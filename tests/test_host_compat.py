"""Compat shims must resolve against the running Hermes core.

Skipped when Hermes is not importable (standalone CI). On a real host this
catches the silent-deny regression from upstream issue #7: the old imports
lived inside try/except blocks that fell back to "deny".
"""

import importlib.util

import pytest

from hermes_dynamic_workflows.host import compat

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("tools") is None or importlib.util.find_spec("tools.approval") is None,
    reason="Hermes core not importable",
)


@pytest.mark.parametrize(
    "getter",
    [
        compat.get_approval_mode_fn,
        compat.get_smart_approve_fn,
        compat.get_set_current_session_key_fn,
        compat.get_reset_current_session_key_fn,
        compat.get_cleanup_vm_fn,
    ],
)
def test_shim_resolves_callable(getter):
    assert callable(getter())


def test_approval_mode_returns_known_value():
    assert compat.get_approval_mode_fn()() in {"manual", "smart", "off"}


def test_unresolvable_symbol_raises_import_error():
    with pytest.raises(ImportError):
        compat._resolve("definitely_not_a_symbol", "tools.approval")
