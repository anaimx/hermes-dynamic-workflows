"""Import shims for Hermes core symbols that moved between releases.

Hermes core decomposed ``tools/approval.py`` and ``tools/terminal_tool.py``
in Sep-2026 (upstream issue #7). Each helper tries the legacy location first (so
older cores and test doubles win) and falls back to the new module, so the plugin runs on both old and new cores.
Every lookup is lazy: importing this module never imports Hermes itself.
"""

from __future__ import annotations

import importlib
from typing import Any


def _resolve(name: str, *modules: str) -> Any:
    last_error: Exception | None = None
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - depends on host core
            last_error = exc
            continue
        if hasattr(module, name):
            return getattr(module, name)
    raise ImportError(
        f"cannot resolve {name!r} from any of {modules!r}"
        + (f" (last error: {last_error})" if last_error else "")
    )


def get_approval_mode_fn():
    return _resolve("_get_approval_mode", "tools.approval", "tools.approval_context")


def get_smart_approve_fn():
    return _resolve("_smart_approve", "tools.approval", "tools.approval_smart")


def get_set_current_session_key_fn():
    return _resolve("set_current_session_key", "tools.approval", "tools.approval_context")


def get_reset_current_session_key_fn():
    return _resolve("reset_current_session_key", "tools.approval", "tools.approval_context")


def get_cleanup_vm_fn():
    return _resolve("cleanup_vm", "tools.terminal_tool", "tools.terminal_tool_lifecycle")
