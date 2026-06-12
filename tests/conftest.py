"""Shared pytest fixtures.

Keeps the whole suite hermetic with respect to the features added around the
CLI's side-effects: no network for the update check, no nag from the MCP hint,
and the global ``.env`` / caches redirected into a tmp dir so tests never read
or pollute the user's real ``~/.config/mitos`` or ``~/.cache/mitos``. Tests that
exercise those features re-enable them explicitly (``monkeypatch.delenv(...)``).
"""

import os

import pytest


@pytest.fixture(autouse=True)
def hermetic_mitos_env(monkeypatch, tmp_path):
    """Isolates per-test config/cache and silences the CLI's network/nag side-effects."""
    monkeypatch.setenv("MITOS_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("MITOS_NO_MCP_HINT", "1")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg_config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg_cache"))


@pytest.fixture(scope="session", autouse=True)
def sweep_leaked_qdrant_collections():
    """After the whole suite, delete test-ONLY collections from the shared Qdrant.

    Tests that build a workspace from ``tempfile.mkdtemp()`` create per-run
    collections named ``mitos-tmp*`` (and the adversarial suite uses
    ``mitos_adversarial_*``); not all of them clean up, so without this they
    accumulate on the shared instance. Pattern-restricted so it can NEVER touch a
    real project collection (``mitos-cartolina`` etc.), and fully best-effort:
    if Qdrant is unreachable, it simply does nothing.
    """
    yield
    url = os.environ.get("QDRANT_URL", "http://localhost:7333")
    try:
        import requests

        resp = requests.get(f"{url}/collections", timeout=2)
        if not resp.ok:
            return
        names = [c["name"] for c in resp.json().get("result", {}).get("collections", [])]
    except Exception:
        return
    for name in names:
        if name.startswith("mitos-tmp") or name.startswith("mitos_adversarial"):
            try:
                requests.delete(f"{url}/collections/{name}", timeout=5)
            except Exception:
                pass
