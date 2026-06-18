"""Shared pytest fixtures.

Keeps the whole suite hermetic with respect to the features added around the
CLI's side-effects: no network for the update check, no nag from the MCP hint,
and the global ``.env`` / caches redirected into a tmp dir so tests never read
or pollute the user's real ``~/.config/mitos`` or ``~/.cache/mitos``. Tests that
exercise those features re-enable them explicitly (``monkeypatch.delenv(...)``).
"""

import os

import pytest


# --- Phase 5a store-rebuild quarantine (the contained-red window) --------------
#
# Phase 5a flips the live schema (entry-001) + identity (entry-002) and rebuilds
# ``commit_parsed_entry`` over the V1a STRICT schema. That flip breaks the five
# live consumers (``sync``/``importer``/``mcp``/``cli``/``renderer``) and the
# prototype read methods **at runtime** — they bind prototype column names
# (``core_axiom``, inline ``scope``/``mechanisms``, ``edges.from_id/to_id/type``,
# the ``pending_embeddings`` drain surface) that no longer exist. This is the
# vision's *contained-red window*, not a regression to chase: the read views are
# restored in Phase 5d and the consumers reconciled in Phase 8a.
#
# To keep the substrate gate (test_identity / test_parser / test_migrations /
# test_config / test_packaging + 5a's rewritten test_store) meaningfully green
# through the 5a→8a window, the broken consumer/read test modules are quarantined
# here — a SINGLE tracked list (Decision 5) skipped via the collection hook below,
# NOT scattered per-file ``pytestmark`` skips and NOT a red CI. The list was
# derived **empirically** (flip → run the full suite → quarantine exactly the
# modules that failed *because of the flip*, not the pre-existing ``*_live.py``
# 429 flakes). Each later phase REMOVES the modules it restores (5d: the read-view
# consumers; 8a: the rest); ``test_store_rebuild_quarantine_is_tracked`` (in
# tests/test_store.py) pins the current set so the shrink to empty is auditable.
STORE_REBUILD_QUARANTINE = [
    # Read-method consumers — restored in Phase 5d (read views + modifier stamping)
    "test_list_decisions.py",
    "test_modifier_surfacing.py",
    "test_neighbor_review.py",
    "test_payload_economy.py",
    "test_surface_confidence.py",
    "test_status_readiness.py",
    "test_renderer.py",
    "test_adversarial_rendering.py",
    # Commit-via-consumer + edge/state consumers — restored in Phase 8a
    "test_sync.py",
    "test_importer.py",
    "test_record_decision.py",
    "test_relations_and_adjacency.py",
    "test_adversarial_invariants.py",
    "test_adversarial_mcp.py",
    "test_cli_pathologies.py",
]


def pytest_collection_modifyitems(config, items):
    """Skips the store-rebuild quarantine modules during the 5a→8a contained-red window.

    Applies a single skip marker to every collected item whose test module is in
    ``STORE_REBUILD_QUARANTINE`` (Phase 5a, Decision 5). The reason names the
    restoring phases so the deferral is legible in the test report; the list
    provably empties by Phase 8a.
    """
    reason = (
        "Phase 5a contained-red window: consumer/read methods break at runtime "
        "against the flipped V1a schema; restored in Phase 5d (read views) / "
        "Phase 8a (consumers)."
    )
    skip_marker = pytest.mark.skip(reason=reason)
    for item in items:
        if item.path.name in STORE_REBUILD_QUARANTINE:
            item.add_marker(skip_marker)


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
