"""Shared pytest fixtures.

Keeps the whole suite hermetic with respect to the features added around the
CLI's side-effects: no network for the update check, no nag from the MCP hint,
and the global ``.env`` / caches redirected into a tmp dir so tests never read
or pollute the user's real ``~/.config/mitos`` or ``~/.cache/mitos``. Tests that
exercise those features re-enable them explicitly (``monkeypatch.delenv(...)``).
"""

import pytest


@pytest.fixture(autouse=True)
def hermetic_mitos_env(monkeypatch, tmp_path):
    """Isolates per-test config/cache and silences the CLI's network/nag side-effects."""
    monkeypatch.setenv("MITOS_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("MITOS_NO_MCP_HINT", "1")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg_config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg_cache"))
