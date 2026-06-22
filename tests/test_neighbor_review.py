"""Tests for the pre-commit near-duplicate / possible-tension review (AX P4).

Loop-Claude's friction: a new decision's nearest neighbour only surfaced in the
POST-commit `related` echo — one step too late to point an amends/supersedes at it
(a re-record is a no-op, so the link could never be added). `record_decision` now
embeds the axiom BEFORE the write, and if it is >=0.85 similar to an existing decision
the author did not reference, it PAUSES (`status: needs_review`, nothing written) so the
author can re-record with the relation or `acknowledge_neighbors=True`.

The pause needs embeddings, so the suite injects a fake embed provider + vector store to
drive it deterministically offline.
"""

import json
import logging
import shutil
import tempfile
from typing import Iterator, Tuple

import pytest
from unittest.mock import patch

from mitos.config import MitosConfig
from mitos.cli import cmd_init, cmd_record
from mitos.store import GraphStore
from mitos.sync import (MitosSyncManager, _has_negation, _polarity_mismatch,
                        _NEIGHBOR_REVIEW_THRESHOLD)


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def ws(offline) -> Iterator[Tuple[MitosConfig, MitosSyncManager]]:
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    cmd_init(config)
    yield config, MitosSyncManager(config)
    shutil.rmtree(tmp, ignore_errors=True)


class _FakeEmbed:
    def get_embedding(self, text, is_query=False):
        return [0.1, 0.2, 0.3]


class _FakeVector:
    def __init__(self, matches):
        self._matches = matches

    def query(self, vector, limit=5):
        return self._matches

    def upsert(self, *a, **k):
        pass


def _arm(m, matches):
    """Wire fake embeddings/vector so the review runs deterministically."""
    m.embed_provider = _FakeEmbed()
    m.vector_store = _FakeVector(matches)


# --------------------------------------------------------------------------- #
# Polarity helpers
# --------------------------------------------------------------------------- #

def test_has_negation_detects_cues():
    assert _has_negation("It is never a per-persona field")
    assert _has_negation("Names are not interpolated")
    assert not _has_negation("It is a per-persona field")


def test_polarity_mismatch():
    assert _polarity_mismatch("X is a field", "X is never a field")
    assert not _polarity_mismatch("X is a field", "X is the field")


# --------------------------------------------------------------------------- #
# The pause (sync layer, real _review_neighbors via injected fakes)
# --------------------------------------------------------------------------- #

def test_high_similar_unreferenced_pauses_and_writes_nothing(ws):
    config, m = ws
    m.record_decision_entry("Catalog owns the per-persona gallery markers.", "rej",
                            ["catalog"], slug="catalog-owns-markers")  # offline → commits
    _arm(m, [{"slug": "catalog-owns-markers", "score": 0.9}])
    res = m.record_decision_entry("The catalog module owns per-persona gallery markers.",
                                  "rej", ["catalog"], slug="catalog-module-markers")
    assert res["status"] == "needs_review"
    assert res["code"] == "similar_decision_exists"
    assert res["neighbors"][0]["slug"] == "catalog-owns-markers"
    # Nothing written: the new node never hit the graph or the buffer.
    store = GraphStore(config.db_path)
    assert store.get_node_by_slug("catalog-module-markers") is None
    with open(config.decisions_file, encoding="utf-8") as f:
        assert "catalog-module-markers" not in f.read()


def test_acknowledge_neighbors_commits(ws):
    config, m = ws
    m.record_decision_entry("Use SQLite for the store.", "rej", ["db"], slug="use-sqlite")
    _arm(m, [{"slug": "use-sqlite", "score": 0.9}])
    res = m.record_decision_entry("Adopt SQLite as the storage engine.", "rej", ["db"],
                                  slug="adopt-sqlite", acknowledge_neighbors=True)
    assert res["status"] == "created"
    assert GraphStore(config.db_path).get_node_by_slug("adopt-sqlite") is not None


def test_declared_relation_skips_pause(ws):
    """Linking the neighbour (amends) means the author already saw it — no pause."""
    config, m = ws
    m.record_decision_entry("Use SQLite for the store.", "rej", ["db"], slug="use-sqlite")
    _arm(m, [{"slug": "use-sqlite", "score": 0.9}])
    res = m.record_decision_entry("Use SQLite with WAL mode.", "rej", ["db"],
                                  slug="use-sqlite-wal", amends="use-sqlite")
    assert res["status"] == "created"


def test_below_threshold_commits(ws):
    config, m = ws
    m.record_decision_entry("Use SQLite for the store.", "rej", ["db"], slug="use-sqlite")
    _arm(m, [{"slug": "use-sqlite", "score": 0.70}])  # loose neighbour, under the bar
    res = m.record_decision_entry("Adopt a Postgres cluster.", "rej", ["db"], slug="adopt-pg")
    assert res["status"] == "created"


def test_offline_never_pauses(ws):
    """No embeddings → the review can't run → it must never block a write."""
    config, m = ws
    m.record_decision_entry("Use SQLite for the store.", "rej", ["db"], slug="use-sqlite")
    # m left offline (no fakes armed)
    res = m.record_decision_entry("Adopt SQLite as the engine.", "rej", ["db"], slug="adopt-sqlite")
    assert res["status"] == "created"


def test_possible_tension_flagged_on_polarity_flip(ws):
    config, m = ws
    m.record_decision_entry("The marker is never a per-persona field.", "rej",
                            ["chrome"], slug="marker-not-persona")
    _arm(m, [{"slug": "marker-not-persona", "score": 0.9}])
    res = m.record_decision_entry("The marker is a per-persona field.", "rej",
                                  ["chrome"], slug="marker-is-persona")
    assert res["status"] == "needs_review"
    assert res["neighbors"][0]["possible_tension"] is True


def test_threshold_is_inclusive(ws):
    config, m = ws
    m.record_decision_entry("Use SQLite for the store.", "rej", ["db"], slug="use-sqlite")
    _arm(m, [{"slug": "use-sqlite", "score": _NEIGHBOR_REVIEW_THRESHOLD}])
    res = m.record_decision_entry("Adopt SQLite engine.", "rej", ["db"], slug="adopt-sqlite")
    assert res["status"] == "needs_review"


# --------------------------------------------------------------------------- #
# MCP + CLI surfaces (patch _review_neighbors for determinism)
# --------------------------------------------------------------------------- #

_FLAGGED = [{"slug": "existing", "axiom": "An existing decision.", "score": 0.9,
             "possible_tension": False}]


def test_mcp_record_decision_pauses(ws):
    from mitos import mcp_server
    config, _ = ws
    with patch("mitos.mcp_server.MitosConfig", return_value=config), \
         patch.object(MitosSyncManager, "_review_neighbors", return_value=_FLAGGED):
        res = json.loads(mcp_server.record_decision("A new call.", "rej", ["s"], slug="newcall"))
    assert res["status"] == "needs_review" and res["neighbors"] == _FLAGGED
    assert GraphStore(config.db_path).get_node_by_slug("newcall") is None


def test_mcp_record_decision_acknowledge_commits(ws):
    from mitos import mcp_server
    config, _ = ws
    # _review_neighbors must NOT even be consulted when acknowledged.
    with patch("mitos.mcp_server.MitosConfig", return_value=config), \
         patch.object(MitosSyncManager, "_review_neighbors", return_value=_FLAGGED):
        res = json.loads(mcp_server.record_decision("A new call.", "rej", ["s"],
                                                    slug="newcall", acknowledge_neighbors=True))
    assert res["status"] == "created"
    assert GraphStore(config.db_path).get_node_by_slug("newcall") is not None


def test_cli_record_pause_exits_nonzero(ws, capsys):
    config, _ = ws
    with patch.object(MitosSyncManager, "_review_neighbors", return_value=_FLAGGED):
        with pytest.raises(SystemExit) as exc:
            cmd_record(config, axiom="A new call.", rejected="rej", slug="newcall")
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "Paused" in err and "existing" in err and "acknowledge-neighbors" in err


def test_cli_record_acknowledge_commits(ws):
    config, _ = ws
    with patch.object(MitosSyncManager, "_review_neighbors", return_value=_FLAGGED):
        cmd_record(config, axiom="A new call.", rejected="rej", slug="newcall",
                   acknowledge_neighbors=True)
    assert GraphStore(config.db_path).get_node_by_slug("newcall") is not None


# --------------------------------------------------------------------------- #
# Transitive-lineage near-dup suppression (Phase 3b)
#
# The near-dup gate is direct-edge-only in V1a: declaring a relation to the
# near-twin suppresses the pause for THAT node, but still pauses on the twin's
# older ancestors reachable through an amends/narrows/supersedes chain. 3b closes
# that gap: a declared edge to a chain HEAD suppresses the pause for the chain's
# transitive predecessors (consuming GraphStore.get_lineage), while an UNDECLARED
# near-twin still pauses (T12); V1a's direct suppression of all nine types is
# preserved by construction (T13 slice, must-not-regress).
#
# CLI ⇄ MCP parity is automatic: both `mitos record` and MCP `record_decision`
# route through `record_decision_entry` (the single gate home, exercised by the
# MCP/CLI tests above), so the transitive suppression is shared by construction —
# verified, not re-implemented (CLAUDE.md behavioural-sync rule).
# --------------------------------------------------------------------------- #


def _seed(m, slug, axiom, **rels):
    """Record a decision OFFLINE (commits, no pause — _review_neighbors is empty until armed)."""
    res = m.record_decision_entry(axiom, "rej", ["s"], slug=slug, **rels)
    assert res["status"] == "created", res
    return res


def _inject_raw_edge(store, source_id, target_id, edge_type):
    """Insert an ``edges`` row directly, BYPASSING ``_reconcile_edges``.

    The supported write path rejects a mutation cycle-closer, so the only way to
    seed a corrupt cycle the homeostasis bound must survive is a raw INSERT (the
    graph is a rebuildable derivative — M7/P6). Mirrors the idiom in
    ``tests/test_lineage_and_cycles.py``.
    """
    ts = "2026-06-23T00:00:00.000000+00:00"
    conn = store._get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO edges (source_id, source_kind, target_id, "
                "target_kind, edge_type, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (source_id, "decision", target_id, "decision", edge_type, ts),
            )
    finally:
        conn.close()


def test_transitive_suppression_amends_chain_head(ws):
    """T12 core: declaring `amends` the chain HEAD suppresses a near-dup grandparent.

    Chain C1 ← C2 ← C3 via `amends` (all stay active — amends is non-kill). The new
    entry declares `amends=C3` (the head); the armed near-dup neighbour is C1 (the
    grandparent, two hops back). C1 ∈ get_lineage(C3), so its slug joins the
    suppression set transitively → no pause, the entry commits.
    """
    config, m = ws
    _seed(m, "chain-c1", "The store uses SQLite for persistence.")
    _seed(m, "chain-c2", "The store uses SQLite with a single connection.", amends="chain-c1")
    _seed(m, "chain-c3", "The store uses SQLite with WAL journaling.", amends="chain-c2")
    # Arm the grandparent (active, NOT the declared target) as the high-similarity match.
    _arm(m, [{"slug": "chain-c1", "score": 0.9}])
    res = m.record_decision_entry(
        "The store uses SQLite with WAL and a busy timeout.", "rej", ["s"],
        slug="chain-c4", amends="chain-c3",
    )
    assert res["status"] == "created", res
    assert GraphStore(config.db_path).get_node_by_slug("chain-c4") is not None


def test_undeclared_near_twin_still_pauses(ws):
    """T12 negative (P10 — the load-bearing proof): an UNDECLARED near-twin still pauses.

    Same C1 ← C2 ← C3 chain, same armed grandparent C1 — but the new entry declares
    NO mutation edge to the chain. C1 is neither a direct nor a transitive target, so
    the pause fires. Stashing the 3b augmentation must leave THIS green while turning
    `test_transitive_suppression_amends_chain_head` RED.
    """
    config, m = ws
    _seed(m, "chain-c1", "The store uses SQLite for persistence.")
    _seed(m, "chain-c2", "The store uses SQLite with a single connection.", amends="chain-c1")
    _seed(m, "chain-c3", "The store uses SQLite with WAL journaling.", amends="chain-c2")
    _arm(m, [{"slug": "chain-c1", "score": 0.9}])
    res = m.record_decision_entry(
        "The store uses SQLite with WAL and a busy timeout.", "rej", ["s"],
        slug="chain-c4-undeclared",
    )
    assert res["status"] == "needs_review"
    assert res["code"] == "similar_decision_exists"
    assert res["neighbors"][0]["slug"] == "chain-c1"
    # Nothing written — the pause is byte-for-byte non-destructive.
    store = GraphStore(config.db_path)
    assert store.get_node_by_slug("chain-c4-undeclared") is None
    with open(config.decisions_file, encoding="utf-8") as f:
        assert "chain-c4-undeclared" not in f.read()


def test_undeclared_unrelated_edge_still_pauses(ws):
    """T12 negative variant: a mutation edge to an UNRELATED node does not suppress the twin.

    The new entry amends an unrelated decision (not on the C1 chain), so C1 is reachable
    through neither the direct target nor its lineage → the pause still fires.
    """
    config, m = ws
    _seed(m, "chain-c1", "The store uses SQLite for persistence.")
    _seed(m, "chain-c2", "The store uses SQLite with a single connection.", amends="chain-c1")
    _seed(m, "unrelated", "The renderer emits MADR markdown.")
    _arm(m, [{"slug": "chain-c1", "score": 0.9}])
    res = m.record_decision_entry(
        "The store uses SQLite with a connection pool.", "rej", ["s"],
        slug="chain-c-unrelated", amends="unrelated",
    )
    assert res["status"] == "needs_review"
    assert res["neighbors"][0]["slug"] == "chain-c1"


def test_supersedes_bridge_freebie_suppresses_across_kill_edge(ws):
    """Success Criterion 3: the union walk bridges a `supersedes` kill-edge for free.

    Chain A ← B (B amends A) ← C (C supersedes B). C supersedes B, so B goes inactive
    and never surfaces as a near-dup; A stays active (only amended). get_lineage(C)
    walks the UNION (supersedes ∪ amends), so it reaches A across the kill-edge bridge
    with no mixed-chain special case. Declaring `amends=C` suppresses the armed A.

    Boundary (suppression-only, vision §6.2): this does NOT re-declare amends/narrows —
    it only quiets the pause. So A reads as amended_by B (its real modifier), NOT by the
    new entry — modifier stamping (a different consumer) is unaffected.
    """
    config, m = ws
    _seed(m, "bridge-a", "Config is read from a single .env file.")
    _seed(m, "bridge-b", "Config is read from .env then the environment.", amends="bridge-a")
    _seed(m, "bridge-c", "Config resolves env then project then global .env.", supersedes="bridge-b")
    store = GraphStore(config.db_path)
    assert store.get_node_state(store.resolve_slug("bridge-b")[0]) == "superseded"
    assert store.get_node_state(store.resolve_slug("bridge-a")[0]) == "active"
    # Arm the deepest STILL-ACTIVE member, reachable from C only across the bridge.
    _arm(m, [{"slug": "bridge-a", "score": 0.9}])
    res = m.record_decision_entry(
        "Config resolves env then project then global, with a cache.", "rej", ["s"],
        slug="bridge-d", amends="bridge-c",
    )
    assert res["status"] == "created", res
    # Boundary (suppression ≠ re-declaration): the new entry amends only C, NOT A. The
    # suppression quieted the pause but synthesized no amends→A edge, so modifier
    # stamping (a different consumer) is unaffected — A is not amended_by the new entry.
    a_id = store.resolve_slug("bridge-a")[0]
    d_id = store.resolve_slug("bridge-d")[0]
    edges = store.get_edges()
    assert not [e for e in edges if e["source_id"] == d_id and e["target_id"] == a_id], \
        "suppression must not synthesize a re-declaration edge to the bridged predecessor"
    assert [e for e in edges if e["source_id"] == d_id and e["edge_type"] == "amends"
            and e["target_id"] == store.resolve_slug("bridge-c")[0]], \
        "the new entry's only mutation edge is the declared amends→C"


@pytest.mark.parametrize("relation", ["amends", "cites", "depends_on", "narrows", "supersedes"])
def test_direct_suppression_preserved_all_relation_types(ws, relation):
    """T13 slice (DoD #13b must-not-regress): a DIRECTLY-declared edge suppresses its neighbour.

    The armed neighbour IS the declared target (not a transitive ancestor), so this is
    V1a's direct suppression — now riding the 3b code path. A representative spread of
    the newly-committing types must each suppress exactly as `supersedes` did in V1a.
    """
    config, m = ws
    slug = "direct-new-" + relation.replace("_", "-")  # slugs hyphenate underscores
    _seed(m, "direct-target", "The vector store batches embedding upserts.")
    _arm(m, [{"slug": "direct-target", "score": 0.9}])
    res = m.record_decision_entry(
        "The vector store batches embedding upserts per sync.", "rej", ["s"],
        slug=slug, **{relation: "direct-target"},
    )
    assert res["status"] == "created", res
    assert GraphStore(config.db_path).get_node_by_slug(slug) is not None


def test_transitive_suppression_offline_is_no_op(ws):
    """Success Criterion 7: offline, the gate is a no-op — the walk is skipped, the write commits.

    No embeddings/vector store → `_review_neighbors` returns [] regardless, so the
    transitive walk is short-circuited (never attempted) and a declared chain-head edge
    still commits with no error.
    """
    config, m = ws
    _seed(m, "off-c1", "The CLI router uses argparse subparsers.")
    _seed(m, "off-c2", "The CLI router uses argparse subparsers with aliases.", amends="off-c1")
    # m left OFFLINE (no _arm) — the augmentation is gated behind embed/vector presence.
    res = m.record_decision_entry(
        "The CLI router uses argparse with command aliases.", "rej", ["s"],
        slug="off-c3", amends="off-c2",
    )
    assert res["status"] == "created", res


def test_corrupt_cycle_suppression_is_loud_partial_and_never_hangs(ws, caplog):
    """Success Criterion 5 (Decision 5): the consumer tolerates a corrupt cycle — no hang.

    A raw-injected mutation cycle X ↔ Y (the write path rejects cycle-closers, so it can
    only enter out-of-band) is reachable from the declared target X. get_lineage(X)
    truncates at the cycle, emits a loud WARNING, and returns the PARTIAL lineage — so
    the gate suppresses what was walked (Y) and the `record` completes rather than
    hanging the hot path. 3b adds no cycle handling of its own; it trusts the 3a bound.
    """
    config, m = ws
    _seed(m, "cyc-x", "The sync manager holds a file lock during commit.")
    _seed(m, "cyc-y", "The sync manager holds a file lock with a timeout.")
    store = m.store
    x_id = store.resolve_slug("cyc-x")[0]
    y_id = store.resolve_slug("cyc-y")[0]
    # Close the cycle out-of-band (the reconciler would reject these as cycle_violation).
    _inject_raw_edge(store, source_id=x_id, target_id=y_id, edge_type="amends")
    _inject_raw_edge(store, source_id=y_id, target_id=x_id, edge_type="amends")
    # Arm Y (active, in the partial lineage of X) as the near-dup neighbour.
    _arm(m, [{"slug": "cyc-y", "score": 0.9}])
    with caplog.at_level(logging.WARNING):
        res = m.record_decision_entry(
            "The sync manager holds a file lock with a bounded timeout.", "rej", ["s"],
            slug="cyc-new", amends="cyc-x",
        )
    # The gate completed (did not hang) and suppressed the partial-lineage member.
    assert res["status"] == "created", res
    assert GraphStore(config.db_path).get_node_by_slug("cyc-new") is not None
    # Loud, non-fatal: get_lineage emitted the homeostasis WARNING naming the cycle.
    assert any("cycle" in r.getMessage().lower() for r in caplog.records), \
        "expected a loud homeostasis WARNING on the corrupt cycle"
