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
# 429 flakes). Each later phase REMOVES the modules it restores;
# ``test_store_rebuild_quarantine_is_tracked`` (in tests/test_store.py) pins the
# current set so the shrink to empty is auditable.
#
# Phase 5d re-bucketed 5a's empirical labels (WIRING_LEDGER entry-003, §16): of the
# 8 modules 5a labelled "restored in 5d", only **2** were genuinely store-only
# (``test_renderer`` + ``test_adversarial_rendering`` — removed below as 5d
# restored them). The other 6 were mis-bucketed: ``test_status_readiness`` is 8×
# ``cli.cmd_status`` (gated on the **6b** cmd_status rebuild), and the remaining 5
# drive the sync consumer write path (``record_decision_entry``), the MCP/CLI
# surfaces, and/or ``amends``/``narrows`` edges (unrepresentable until V1b) — all
# **8a**'s charter. The store-level modifier (T12) + C4 (T5) proofs those would
# have given are delivered in ``tests/test_store.py`` instead.
#
# Phase 6b restored ``test_status_readiness`` (the ``cmd_status`` rebuild it gated
# on landed; the prototype-shape ``ParsedEntry`` fixture was reworked to V1a),
# leaving 12 modules — all 8a's.
#
# Phase 8a DRAINED the quarantine to EMPTY (entry-003 closed): the five live
# consumers (sync/importer/mcp_server/cli) were reconciled to the V1a substrate
# (parse_entry_stream + compute_node_id + get_node_state + the V1a drain surface),
# ``--corrects`` was wired, and every restored module was re-greened — V1b-
# unrepresentable assertions (amends/narrows modifiers, OQ parked/resolved) pared
# or deferred with a logged note (OD1; never silent-skip/coerce). The list reaching
# 0 is the contained-red window closing.
STORE_REBUILD_QUARANTINE: list[str] = []


def pytest_collection_modifyitems(config, items):
    """Skips the store-rebuild quarantine modules during the 5a→8a contained-red window.

    Applies a single skip marker to every collected item whose test module is in
    ``STORE_REBUILD_QUARANTINE`` (Phase 5a, Decision 5). The reason names the
    restoring phases so the deferral is legible in the test report; the list
    provably empties by Phase 8a.
    """
    reason = (
        "Phase 5a contained-red window: consumer methods break at runtime against "
        "the flipped V1a schema (the read views were restored in Phase 5d); "
        "restored in Phase 6b (cmd_status) / Phase 8a (consumers)."
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
