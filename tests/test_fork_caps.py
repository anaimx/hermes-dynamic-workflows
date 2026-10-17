"""Fork: default_token_budget config + env parsing."""

import os
import unittest
from unittest import mock

from hermes_dynamic_workflows.core.config import PluginConfig, load_config


class DefaultTokenBudgetTests(unittest.TestCase):
    def test_default_is_disabled(self):
        self.assertEqual(PluginConfig().default_token_budget, 0)

    def test_env_sets_budget(self):
        with mock.patch.dict(os.environ, {"HERMES_DYNAMIC_WORKFLOWS_DEFAULT_TOKEN_BUDGET": "2000000"}):
            self.assertEqual(load_config().default_token_budget, 2_000_000)

    def test_garbage_env_falls_back(self):
        with mock.patch.dict(os.environ, {"HERMES_DYNAMIC_WORKFLOWS_DEFAULT_TOKEN_BUDGET": "lots"}):
            self.assertEqual(load_config().default_token_budget, 0)

    def test_a2a_blocked_for_children(self):
        self.assertIn("a2a", PluginConfig().blocked_child_toolsets)


if __name__ == "__main__":
    unittest.main()
