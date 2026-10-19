"""Fork: hard per-child limits (tokens, tool calls), default child model,
opt-in discoverable toolsets, and errored-child token accounting."""

from __future__ import annotations

import os
import threading
import time
import unittest
from unittest import mock

from hermes_dynamic_workflows.child import runner as runner_mod
from hermes_dynamic_workflows.child.runner import HermesChildAgentRunner, _resolve_child_toolsets
from hermes_dynamic_workflows.core.config import PluginConfig, load_config
from hermes_dynamic_workflows.core.errors import ChildAgentError, ChildLimitExceeded
from hermes_dynamic_workflows.core.types import ChildAgentRequest
from hermes_dynamic_workflows.child.worktree import WorkspaceLease
from hermes_dynamic_workflows.core.errors import WorkflowLimitExceeded
from hermes_dynamic_workflows.core.types import ChildAgentRunner
from hermes_dynamic_workflows.engine.runtime import WorkflowOptions, run_workflow


def _request(**kw):
    base = dict(id=1, prompt="work", label="w", phase=None, toolsets=[])
    base.update(kw)
    return ChildAgentRequest(**base)


class BurningChild:
    """Spends `step` tokens per loop until interrupted (like a runaway agent)."""

    session_completion_tokens = 0
    session_reasoning_tokens = 0
    session_cache_read_tokens = 0
    session_cache_write_tokens = 0
    model = "test"

    def __init__(self, step=500, tool_hook=None):
        self.session_prompt_tokens = 0
        self.step = step
        self.interrupted = threading.Event()
        self.closed = False
        self.tool_hook = tool_hook

    def run_conversation(self, **_):
        while not self.interrupted.is_set():
            self.session_prompt_tokens += self.step
            if self.tool_hook:
                self.tool_hook()
            time.sleep(0.01)
            if self.session_prompt_tokens > 10_000_000:
                break
        return {"final_response": "", "messages": [], "interrupted": True}

    def interrupt(self, *a, **k):
        self.interrupted.set()
        return True

    def close(self):
        self.closed = True


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        cfg = PluginConfig()
        self.assertEqual(cfg.max_child_tokens, 0)
        self.assertEqual(cfg.max_child_tool_calls, 0)
        self.assertFalse(cfg.include_discoverable_toolsets)
        self.assertEqual(cfg.default_child_model, "")

    def test_load_from_raw_and_env(self):
        env = {
            "HERMES_DYNAMIC_WORKFLOWS_MAX_CHILD_TOKENS": "150000",
            "HERMES_DYNAMIC_WORKFLOWS_MAX_CHILD_TOOL_CALLS": "25",
            "HERMES_DYNAMIC_WORKFLOWS_INCLUDE_DISCOVERABLE_TOOLSETS": "1",
            "HERMES_DYNAMIC_WORKFLOWS_DEFAULT_CHILD_MODEL": "ag/gemini-3.8-flash-high",
        }
        with mock.patch.dict(os.environ, env):
            cfg = load_config()
        self.assertEqual(cfg.max_child_tokens, 150_000)
        self.assertEqual(cfg.max_child_tool_calls, 25)
        self.assertTrue(cfg.include_discoverable_toolsets)
        self.assertEqual(cfg.default_child_model, "ag/gemini-3.8-flash-high")


class ToolsetTests(unittest.TestCase):
    def test_runner_does_not_add_discoverable_by_default(self):
        with mock.patch.object(runner_mod, "_discoverable_child_toolsets", return_value=["mcp-x"]):
            self.assertEqual(
                runner_mod._child_toolsets_for(PluginConfig(), _request(), None, None),
                ["web", "file", "terminal", "skills"],
            )

    def test_runner_adds_discoverable_when_opted_in(self):
        cfg = PluginConfig(include_discoverable_toolsets=True)
        with mock.patch.object(runner_mod, "_discoverable_child_toolsets", return_value=["mcp-x"]):
            self.assertIn("mcp-x", runner_mod._child_toolsets_for(cfg, _request(), None, None))

    def test_explicit_toolsets_never_get_discoverable(self):
        cfg = PluginConfig(include_discoverable_toolsets=True)
        with mock.patch.object(runner_mod, "_discoverable_child_toolsets", return_value=["mcp-x"]):
            self.assertEqual(
                runner_mod._child_toolsets_for(cfg, _request(toolsets=["terminal"]), None, None),
                ["terminal"],
            )


class DefaultModelTests(unittest.TestCase):
    def test_default_child_model_applies_when_unset(self):
        cfg = PluginConfig(default_child_model="ag/flash")
        self.assertEqual(runner_mod._apply_default_child_model(cfg, _request()).model, "ag/flash")

    def test_explicit_model_wins(self):
        cfg = PluginConfig(default_child_model="ag/flash")
        req = _request(model="cc/opus")
        self.assertEqual(runner_mod._apply_default_child_model(cfg, req).model, "cc/opus")

    def test_inherit_keyword_wins(self):
        cfg = PluginConfig(default_child_model="ag/flash")
        req = _request(model="inherit")
        self.assertEqual(runner_mod._apply_default_child_model(cfg, req).model, "inherit")

    def test_empty_config_is_noop(self):
        self.assertIsNone(runner_mod._apply_default_child_model(PluginConfig(), _request()).model)


class HardLimitTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(runner_mod, "LIMIT_POLL_SECONDS", 0.02)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, cfg, child, request=None):
        lease = WorkspaceLease(task_id="workflow-limit-test", cwd="/tmp")
        return HermesChildAgentRunner(cfg)._run_child_with_timeout(
            child, request or _request(), lease, None, []
        )

    def test_token_cap_interrupts_running_child(self):
        child = BurningChild(step=500)
        t0 = time.monotonic()
        with self.assertRaises(ChildLimitExceeded) as ctx:
            self._run(PluginConfig(max_child_tokens=5_000), child)
        self.assertTrue(child.interrupted.is_set())
        self.assertTrue(child.closed)
        self.assertLess(time.monotonic() - t0, 5)
        self.assertIn("token", str(ctx.exception))
        # Overshoot bounded to a few poll intervals, not unbounded.
        self.assertLess(child.session_prompt_tokens, 50_000)

    def test_request_token_limit_tighter_than_config(self):
        child = BurningChild(step=500)
        with self.assertRaises(ChildLimitExceeded):
            self._run(PluginConfig(max_child_tokens=10_000_000), child, _request(token_limit=3_000))
        self.assertLess(child.session_prompt_tokens, 40_000)

    def test_limit_error_is_a_child_error(self):
        self.assertTrue(issubclass(ChildLimitExceeded, ChildAgentError))

    def test_tool_call_cap_interrupts(self):
        lease_id = "workflow-limit-test"
        holder = {}

        def hook():
            cb = holder.get("cb")
            if cb:
                cb({"type": "tool_call", "tool_name": "terminal"})

        child = BurningChild(step=1, tool_hook=hook)
        real_register = None
        from hermes_dynamic_workflows.adapters import hooks as hooks_mod

        real_register = hooks_mod.register_child_observer

        def capture(task_id, cb):
            if task_id == lease_id:
                holder["cb"] = cb
            return real_register(task_id, cb)

        with mock.patch.object(hooks_mod, "register_child_observer", side_effect=capture):
            with self.assertRaises(ChildLimitExceeded) as ctx:
                self._run(PluginConfig(max_child_tool_calls=5), child)
        self.assertIn("tool call", str(ctx.exception))
        self.assertTrue(child.interrupted.is_set())

    def test_no_limits_means_normal_completion(self):
        class Quick(BurningChild):
            def run_conversation(self, **_):
                self.session_prompt_tokens = 123
                return {"final_response": "ok", "messages": [], "completed": True}

        result = self._run(PluginConfig(), Quick())
        self.assertEqual(result.content, "ok")
        self.assertEqual(result.metadata["tokens"], 123)

    def test_child_under_cap_completes(self):
        class Quick(BurningChild):
            def run_conversation(self, **_):
                self.session_prompt_tokens = 900
                return {"final_response": "ok", "messages": [], "completed": True}

        result = self._run(PluginConfig(max_child_tokens=1_000), Quick())
        self.assertEqual(result.content, "ok")


class ErroredChildAccountingTests(unittest.TestCase):
    def test_errored_child_tokens_count_toward_run_budget(self):
        """A child that burns tokens then errors must still charge the budget,
        so the next agent() is refused."""

        class FailThenRun(ChildAgentRunner):
            def __init__(self):
                self.calls = 0

            def run(self, request):
                self.calls += 1
                if self.calls == 1:
                    if request.on_update:
                        request.on_update({"tokens": 42_000})
                    raise ChildLimitExceeded("child token limit exceeded")
                from hermes_dynamic_workflows.core.types import ChildAgentResult

                return ChildAgentResult(content="ok", metadata={"tokens": 1})

        script = """
meta = {"name": "err-budget", "description": "Test workflow"}

try:
    await agent("a", {"label": "a"})
except Exception:
    pass
return await agent("b", {"label": "b"})
"""
        with self.assertRaises(WorkflowLimitExceeded):
            run_workflow(
                script,
                WorkflowOptions(
                    config=PluginConfig(),
                    child_runner=FailThenRun(),
                    token_budget_total=10_000,
                ),
            )


if __name__ == "__main__":
    unittest.main()
