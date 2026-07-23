"""Tests for the pre-commit near-duplicate review (AX P4).

Loop-Claude's friction: a new decision's nearest neighbour only surfaced in the
POST-commit `related` echo (since deleted) — one step too late to point an amends/supersedes at it
(a re-record is a no-op, so the link could never be added). `record_decision` now
embeds the axiom BEFORE the write, and if it is >=0.80 similar to an existing decision
(the strong-match band floor — ADR `record-pause-floor-lowered-to-strong-match-band`)
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
from mitos.conflict import ConflictUnavailableReason, Unavailable
from mitos.errors import DatabaseError, EmbeddingError, VectorStoreError
from mitos.parser import ParsedEntry
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager, _NEIGHBOR_REVIEW_THRESHOLD


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


def test_multi_value_supersedes_suppresses_pause_on_all_targets(ws):
    """A comma-separated --supersedes suppresses the pause on EVERY declared target.

    Regression for the multi-value record fix: ``declared_targets`` is built from the
    SPLIT slugs, not the raw comma-string — otherwise each target spuriously re-pauses
    (the exact friction hit committing the Q3 4-supersede ADR).
    """
    config, m = ws
    m.record_decision_entry("Axiom A.", "rej", ["s"], slug="prior-a")
    m.record_decision_entry("Axiom B.", "rej", ["s"], slug="prior-b")
    _arm(m, [{"slug": "prior-a", "score": 0.9}, {"slug": "prior-b", "score": 0.9}])
    res = m.record_decision_entry("Unifying axiom.", "rej", ["s"], slug="unifier",
                                  supersedes="prior-a, prior-b")
    assert res["status"] == "created", res


def test_multi_value_extra_relation_suppresses_pause_on_all_targets(ws):
    """A comma-separated non-kill relation (--cites) likewise suppresses every target."""
    config, m = ws
    m.record_decision_entry("Axiom A.", "rej", ["s"], slug="cited-a")
    m.record_decision_entry("Axiom B.", "rej", ["s"], slug="cited-b")
    _arm(m, [{"slug": "cited-a", "score": 0.9}, {"slug": "cited-b", "score": 0.9}])
    res = m.record_decision_entry("Citing axiom.", "rej", ["s"], slug="citer",
                                  cites="cited-a, cited-b")
    assert res["status"] == "created", res


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


def test_strong_match_band_pauses(ws):
    """A 0.80–0.85 unreferenced active neighbour now pauses (the lowered floor's
    whole point — ADR `record-pause-floor-lowered-to-strong-match-band`: at 0.85
    this band fell to the visibility-only `related` echo (since deleted), the
    mechanism that already failed the five-week prose-obsoletion trap)."""
    config, m = ws
    m.record_decision_entry("Use SQLite for the store.", "rej", ["db"], slug="use-sqlite")
    _arm(m, [{"slug": "use-sqlite", "score": 0.82}])
    res = m.record_decision_entry("Adopt SQLite as the engine.", "rej", ["db"],
                                  slug="adopt-sqlite")
    assert res["status"] == "needs_review"
    assert res["neighbors"][0]["slug"] == "use-sqlite"


def test_strong_match_band_declared_target_still_commits(ws):
    """A 0.82 neighbour that IS the declared target stays exempt at the new floor."""
    config, m = ws
    m.record_decision_entry("Use SQLite for the store.", "rej", ["db"], slug="use-sqlite")
    _arm(m, [{"slug": "use-sqlite", "score": 0.82}])
    res = m.record_decision_entry("Use SQLite with WAL mode.", "rej", ["db"],
                                  slug="use-sqlite-wal", amends="use-sqlite")
    assert res["status"] == "created", res


def test_threshold_value_matches_adr():
    """The floor is 0.80 — pinned so a drive-by retune goes through the ADR."""
    assert _NEIGHBOR_REVIEW_THRESHOLD == 0.80


def test_threshold_is_inclusive(ws):
    config, m = ws
    m.record_decision_entry("Use SQLite for the store.", "rej", ["db"], slug="use-sqlite")
    _arm(m, [{"slug": "use-sqlite", "score": _NEIGHBOR_REVIEW_THRESHOLD}])
    res = m.record_decision_entry("Adopt SQLite engine.", "rej", ["db"], slug="adopt-sqlite")
    assert res["status"] == "needs_review"


# --------------------------------------------------------------------------- #
# The enriched payload (candidate_payload shape + modifier stamps)
# --------------------------------------------------------------------------- #

def test_pause_neighbor_carries_enriched_letter_shape(ws):
    """Each pause neighbour is the full candidate_payload finding — the Letter core
    (axiom / scope / rejected_paths) plus score — with no polarity guess: the retired
    ``possible_tension`` key must not reappear (nothing guessed replaces it; the
    authoring agent judges tenability from the enrichment itself)."""
    config, m = ws
    m.record_decision_entry(
        "Catalog owns the per-persona gallery markers.",
        "Rejected a per-persona marker table: markers are catalog facts.",
        ["catalog"], slug="catalog-owns-markers")  # offline → commits
    _arm(m, [{"slug": "catalog-owns-markers", "score": 0.9}])
    res = m.record_decision_entry("The catalog module owns per-persona gallery markers.",
                                  "rej", ["catalog"], slug="catalog-module-markers")
    assert res["status"] == "needs_review"
    n = res["neighbors"][0]
    assert n["slug"] == "catalog-owns-markers"
    assert n["axiom"] == "Catalog owns the per-persona gallery markers."
    assert n["scope"] == ["catalog"]
    assert n["rejected_paths"] == (
        "Rejected a per-persona marker table: markers are catalog facts.")
    assert n["score"] == 0.9
    assert "possible_tension" not in n


def test_amended_but_active_neighbor_surfaces_amended_by(ws):
    """An active-but-amended neighbour carries its ``amended_by`` stamp on the pause.

    The pause is a decision-read surface, so the every-read-surface stamping
    invariant binds it: the agent judging tenability must see the neighbour has
    already moved on, never read it as the final word (the "amended axioms read
    as live" trap).

    There is deliberately no ``superseded_by``/``corrected_by`` fixture: the
    gather keeps only ``active ∪ drifted`` nodes, and a kill edge
    (supersedes/corrects) never points at a live target (store.py's kill-edge
    rule) — a kill-stamped neighbour is structurally unreachable on this
    surface, a documented impossibility rather than a coverage gap.
    """
    config, m = ws
    m.record_decision_entry("Use SQLite for the store.", "rej", ["db"], slug="use-sqlite")
    m.record_decision_entry("Use SQLite with WAL mode.", "rej", ["db"],
                            slug="use-sqlite-wal", amends="use-sqlite")
    _arm(m, [{"slug": "use-sqlite", "score": 0.9}])
    res = m.record_decision_entry("Adopt SQLite as the storage engine.", "rej", ["db"],
                                  slug="adopt-sqlite")
    assert res["status"] == "needs_review"
    n = res["neighbors"][0]
    assert n["slug"] == "use-sqlite"
    assert n["amended_by"] == ["use-sqlite-wal"]


def test_unmodified_neighbor_carries_no_stamp_keys(ws):
    """A never-modified neighbour carries NO stamp keys — stamps are conditional
    (present only when non-empty), so the unmodified majority stays clean."""
    config, m = ws
    m.record_decision_entry("Use SQLite for the store.", "rej", ["db"], slug="use-sqlite")
    _arm(m, [{"slug": "use-sqlite", "score": 0.9}])
    res = m.record_decision_entry("Adopt SQLite as the storage engine.", "rej", ["db"],
                                  slug="adopt-sqlite")
    assert res["status"] == "needs_review"
    n = res["neighbors"][0]
    for key in ("amended_by", "narrowed_by", "superseded_by", "corrected_by"):
        assert key not in n


# --------------------------------------------------------------------------- #
# MCP + CLI surfaces (patch _review_neighbors for determinism)
# --------------------------------------------------------------------------- #

# The enriched candidate_payload shape in its contractual key order (slug, axiom,
# scope, score, rejected_paths[, stamps]). No stamp key here — this is the
# unmodified-majority shape; the stamped case rides the real-path tests below.
_FLAGGED = [{"slug": "existing", "axiom": "An existing decision.", "scope": ["s"],
             "score": 0.9,
             "rejected_paths": "Rejected the obvious alternative, with reasons."}]


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
    # The enriched render: each neighbour block carries its rejected_paths
    # fragment and scope tag; the retired polarity marker must not reappear.
    assert "Rejected the obvious alternative" in err
    assert "scope: s" in err
    assert "possible tension" not in err


def test_cli_record_acknowledge_commits(ws):
    config, _ = ws
    with patch.object(MitosSyncManager, "_review_neighbors", return_value=_FLAGGED):
        cmd_record(config, axiom="A new call.", rejected="rej", slug="newcall",
                   acknowledge_neighbors=True)
    assert GraphStore(config.db_path).get_node_by_slug("newcall") is not None


# --------------------------------------------------------------------------- #
# CLI⇄MCP parity (Phase 2b)
#
# Both surfaces build their own MitosSyncManager internally, so the REAL review
# path (gather → screen → candidate_payload) is armed via a shared factory
# patched in at each surface's own resolution point: `mitos.cli.MitosSyncManager`
# (module-top import) and `mitos.sync.MitosSyncManager` (the MCP tool's
# function-local import, resolved at call time).
# --------------------------------------------------------------------------- #

def _armed_real_manager_factory(matches):
    """A side_effect factory both surface patches share: real manager, armed fakes.

    Captures the real class at build time, BEFORE any patch goes up — a factory
    body that resolved ``mitos.sync.MitosSyncManager`` while the patch is active
    would get the mock back and recurse.
    """
    real_cls = MitosSyncManager

    def factory(config):
        m = real_cls(config)
        _arm(m, matches)
        return m

    return factory


def test_cli_mcp_record_pause_parity(ws, capsys):
    """T7 (KDD-5): both surfaces emit the identical enriched pause payload.

    Drives the real review path on both surfaces — injected dicts structurally
    cannot prove the whole chain (gather → screen → candidate_payload → both
    serializers) agrees — over one seeded near-dup carrying an `amends` modifier
    stamp. Compares the *parsed* CLI --json stdout against the *parsed* MCP tool
    return (the two emissions differ in indentation by design), and pins the
    protocol semantics riding the shared message: both exits named, no
    tension/judged wording.
    """
    from mitos import mcp_server
    config, m = ws
    # Seed through the directly-built manager BEFORE the patches: offline →
    # _review_neighbors returns [] → seeding never pauses. The paused record
    # must NOT declare these slugs (a declared target is screened out).
    m.record_decision_entry("Use SQLite for the store.",
                            "Rejected Postgres: operational weight.", ["db"],
                            slug="use-sqlite")
    m.record_decision_entry("Use SQLite with WAL mode.", "rej", ["db"],
                            slug="use-sqlite-wal", amends="use-sqlite")
    factory = _armed_real_manager_factory([{"slug": "use-sqlite", "score": 0.9}])

    with patch("mitos.cli.MitosSyncManager", side_effect=factory), \
         patch("mitos.sync.MitosSyncManager", side_effect=factory), \
         patch("mitos.mcp_server.MitosConfig", return_value=config):
        capsys.readouterr()
        with pytest.raises(SystemExit) as exc:
            cmd_record(config, axiom="Adopt SQLite as the storage engine.",
                       rejected="rej", scope=["db"], slug="adopt-sqlite",
                       as_json=True)
        assert exc.value.code == 2
        cli_payload = json.loads(capsys.readouterr().out)
        mcp_payload = json.loads(mcp_server.record_decision(
            "Adopt SQLite as the storage engine.", "rej", ["db"],
            slug="adopt-sqlite"))

    assert cli_payload["neighbors"] == mcp_payload["neighbors"]
    assert cli_payload["code"] == mcp_payload["code"] == "similar_decision_exists"
    assert cli_payload["message"] == mcp_payload["message"]

    # The enriched shape in its contractual key order, the stamp riding last.
    n = cli_payload["neighbors"][0]
    assert list(n.keys()) == ["slug", "axiom", "scope", "score", "rejected_paths",
                              "amended_by"]
    assert n["slug"] == "use-sqlite"
    assert n["rejected_paths"] == "Rejected Postgres: operational weight."
    assert n["amended_by"] == ["use-sqlite-wal"]

    msg = cli_payload["message"]
    assert "amends/supersedes/contradicts/cites" in msg      # the relation exit
    assert "acknowledge_neighbors=True" in msg               # the independence exit
    assert "tension" not in msg.lower() and "judged" not in msg.lower()
    # A pause on either surface writes nothing.
    assert GraphStore(config.db_path).get_node_by_slug("adopt-sqlite") is None


def test_cli_record_pause_renders_enrichment_and_stamp(ws, capsys):
    """The human pause render shows each neighbour's enrichment — rejected_paths
    fragment, scope tag — and the `amended_by` stamp slug on the head line (a
    stamped neighbour must never read as final). Real path via the armed
    factory: the stamp rides only because the amends edge is genuinely in the
    store."""
    config, m = ws
    m.record_decision_entry("Use SQLite for the store.",
                            "Rejected Postgres: operational weight.", ["db"],
                            slug="use-sqlite")
    m.record_decision_entry("Use SQLite with WAL mode.", "rej", ["db"],
                            slug="use-sqlite-wal", amends="use-sqlite")
    factory = _armed_real_manager_factory([{"slug": "use-sqlite", "score": 0.9}])

    with patch("mitos.cli.MitosSyncManager", side_effect=factory):
        with pytest.raises(SystemExit) as exc:
            cmd_record(config, axiom="Adopt SQLite as the storage engine.",
                       rejected="rej", scope=["db"], slug="adopt-sqlite")
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "use-sqlite" in err
    assert "Rejected Postgres" in err
    assert "scope: db" in err
    assert "amended by: use-sqlite-wal" in err
    assert "possible tension" not in err


def test_mcp_record_docstring_pins_enriched_protocol():
    """The greppable halves of T6/KD-2 on the docstring (P13 prompt code): the
    retired `possible_tension` key must not reappear, and the degraded-commit
    key `neighbor_review_unavailable` must stay documented."""
    from mitos import mcp_server
    doc = mcp_server.record_decision.__doc__
    assert "possible_tension" not in doc
    assert "neighbor_review_unavailable" in doc


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


# --------------------------------------------------------------------------- #
# Gather-composed pause: OQ-safety + honest degradation (Phase 1a)
#
# The pause's discovery now composes the Conflict sensor's gather+screen stages.
# Two properties the old pause-local scan lacked, pinned here:
#   * OQ-safe by construction — an open question in the KNN window is screened by
#     kind at gather, never crashed on (the old scan KeyError'd on `core_axiom`
#     and its blanket catch converted that into a silently EMPTY pause, hiding
#     genuine duplicates).
#   * "Couldn't check" ≠ "checked, clean" — a degraded pause read (embed/vector
#     Unavailable, or a propagated graph fault caught at the record call site)
#     fails open: the record commits and the created receipt carries ONE calm
#     `neighbor_review_unavailable` notice. Clean-empty, offline-unconfigured,
#     and acknowledged-past records carry NO notice.
# --------------------------------------------------------------------------- #


class _RaisingEmbed:
    """An embed provider that always raises — the constructor-injected fault shape."""

    def __init__(self, exc):
        self._exc = exc

    def get_embedding(self, text, is_query=False):
        raise self._exc


class _RaisingVector:
    """A vector store whose query always raises (upsert stays a harmless no-op)."""

    def __init__(self, exc):
        self._exc = exc

    def query(self, vector, limit=5):
        raise self._exc

    def upsert(self, *a, **k):
        pass


def test_active_oq_in_window_pauses_on_genuine_decision(ws):
    """T1 (the mandatory regression): an active OQ in the KNN window never empties the pause.

    Mirror of test_conflict_gather.py::test_open_question_match_is_dropped_by_kind_
    not_crashed_on at the pause level. The old scan subscripted `core_axiom` (which an
    OQ lacks) and its blanket catch turned the KeyError into an empty pause — an agent
    could sail past a genuine duplicate because an unrelated OQ happened to be nearby.
    The gather screens the OQ by kind; the decision near-dup still pauses.
    """
    config, m = ws
    _seed(m, "cache-policy", "Cache aggressively at the edge.")
    # Seed an open question via parse→commit (the deterministic keyless path —
    # record_decision_entry only mints decisions).
    oq = ParsedEntry("open_question", "oq-cache-cadence", 40, 50)
    oq.topic = "cache invalidation cadence"
    oq.questions_raised = ["How often should the cache invalidate?"]
    oq.scope = ["s"]
    m.store.commit_parsed_entry(oq)
    # Sanity anchors (the exact trap): kill-edge state calls the OQ 'active', and it
    # lacks the `core_axiom` key the old scan hard-subscripted.
    oq_node = m.store.get_node_by_slug("oq-cache-cadence")
    assert oq_node is not None and oq_node["kind"] == "open_question"
    assert m.store.get_node_state(oq_node["id"]) == "active"
    assert "core_axiom" not in oq_node
    # The OQ is the NEAREST match, beside a genuine decision near-dup.
    _arm(m, [{"slug": "oq-cache-cadence", "score": 0.95},
             {"slug": "cache-policy", "score": 0.9}])
    res = m.record_decision_entry("Cache aggressively at the CDN edge.", "rej", ["s"],
                                  slug="cache-policy-v2")
    assert res["status"] == "needs_review"
    assert [n["slug"] for n in res["neighbors"]] == ["cache-policy"]
    # The pause wrote nothing.
    assert GraphStore(config.db_path).get_node_by_slug("cache-policy-v2") is None


def test_declared_weak_edge_exempt_while_undeclared_pauses(ws):
    """T2 (invariant 3): the pause screens on its own BROAD declared set, in one window.

    A declared weak-edge (`cites`) target at 0.9 is exempt while an UNDECLARED 0.9
    neighbour in the same window still pauses — proving the broad all-relations set
    (not `declared_strong_targets`, which ignores weak edges) reached screen_candidates.
    """
    config, m = ws
    _seed(m, "weak-cited", "Embedding upserts are batched per sync.")
    _seed(m, "weak-undeclared", "Embedding upserts are batched per run.")
    _arm(m, [{"slug": "weak-cited", "score": 0.9},
             {"slug": "weak-undeclared", "score": 0.9}])
    res = m.record_decision_entry("Embedding upserts batch across the sync run.",
                                  "rej", ["s"], slug="weak-new", cites="weak-cited")
    assert res["status"] == "needs_review"
    assert [n["slug"] for n in res["neighbors"]] == ["weak-undeclared"]


def test_degraded_embedding_commits_with_notice(ws):
    """T3: an EmbeddingError during the pause read fails open — commit + calm notice."""
    config, m = ws
    m.embed_provider = _RaisingEmbed(EmbeddingError("quota exhausted"))
    m.vector_store = _FakeVector([])
    res = m.record_decision_entry("The renderer emits MADR markdown.", "rej", ["s"],
                                  slug="deg-embed")
    assert res["status"] == "created"
    notice = res["neighbor_review_unavailable"]
    assert "embedding service unavailable" in notice
    assert "mitos check" in notice
    # Unavailable.detail is logging-only — never rendered into the receipt.
    assert "quota exhausted" not in notice
    assert GraphStore(config.db_path).get_node_by_slug("deg-embed") is not None


def test_degraded_vector_store_commits_with_notice(ws):
    """T3: a VectorStoreError during the pause read fails open — commit + calm notice."""
    config, m = ws
    m.embed_provider = _FakeEmbed()
    m.vector_store = _RaisingVector(VectorStoreError("connection refused"))
    res = m.record_decision_entry("The renderer emits MADR markdown.", "rej", ["s"],
                                  slug="deg-vector")
    assert res["status"] == "created"
    notice = res["neighbor_review_unavailable"]
    assert "vector store unavailable" in notice
    assert "mitos check" in notice
    assert "connection refused" not in notice
    assert GraphStore(config.db_path).get_node_by_slug("deg-vector") is not None


def test_graph_fault_during_pause_read_commits_with_notice(ws):
    """T3 (the call-site catch): a propagated DatabaseError fails open, never crashes.

    gather_candidates propagates graph-store faults by design; the record call site
    catches exactly (DatabaseError, ValidationError). Phase B uses get_node /
    commit_parsed_entry / get_outgoing_edges — none of them the patched read — so the
    commit proceeds. (Nothing post-commit calls the patched method anymore: the
    `related` echo and its blanket-catch scan were deleted.)
    """
    config, m = ws
    _seed(m, "gf-prior", "The sync lock is held during commit.")
    _arm(m, [{"slug": "gf-prior", "score": 0.9}])
    with patch.object(m.store, "get_node_by_slug",
                      side_effect=DatabaseError("disk I/O error")):
        res = m.record_decision_entry("The sync lock is held for the commit duration.",
                                      "rej", ["s"], slug="gf-new")
    assert res["status"] == "created"
    notice = res["neighbor_review_unavailable"]
    assert "graph read failed" in notice
    assert "mitos check" in notice
    assert "disk I/O error" not in notice
    assert GraphStore(config.db_path).get_node_by_slug("gf-new") is not None


def test_clean_empty_carries_no_notice(ws):
    """T3 negative: a healthy check that found nothing is silent — no degradation key."""
    config, m = ws
    _arm(m, [])  # substrate healthy, window empty
    res = m.record_decision_entry("A decision with no neighbours at all.", "rej", ["s"],
                                  slug="clean-empty")
    assert res["status"] == "created"
    assert "neighbor_review_unavailable" not in res


def test_offline_unconfigured_carries_no_notice(ws):
    """Structural absence (a graph-only workspace) is healthy, not degraded — no notice."""
    config, m = ws
    # m left offline: no embed provider, no vector store.
    res = m.record_decision_entry("A decision recorded offline.", "rej", ["s"],
                                  slug="offline-clean")
    assert res["status"] == "created"
    assert "neighbor_review_unavailable" not in res


def test_acknowledge_bypass_carries_no_notice(ws):
    """A declined check is not a failed one: acknowledge_neighbors=True runs no gather,
    so even a broken embed provider yields a clean receipt with no notice."""
    config, m = ws
    m.embed_provider = _RaisingEmbed(EmbeddingError("would raise if consulted"))
    m.vector_store = _FakeVector([])
    res = m.record_decision_entry("An acknowledged independent decision.", "rej", ["s"],
                                  slug="ack-clean", acknowledge_neighbors=True)
    assert res["status"] == "created"
    assert "neighbor_review_unavailable" not in res


# --------------------------------------------------------------------------- #
# The retired `related` echo (T4)
# --------------------------------------------------------------------------- #

def test_created_receipt_carries_no_related_echo(ws):
    """T4: the post-commit `related` echo is DELETED, not merely offline — an armed
    high-similarity neighbour plus acknowledge_neighbors=True (the exact scenario
    the old echo fired on: pause bypassed, embed + vector store answering) yields
    a created receipt with no `related` key."""
    config, m = ws
    _seed(m, "echo-prior", "Retries use exponential backoff with jitter.")
    _arm(m, [{"slug": "echo-prior", "score": 0.9}])
    res = m.record_decision_entry("Retry policy is exponential backoff plus jitter.",
                                  "rej", ["s"], slug="echo-new",
                                  acknowledge_neighbors=True)
    assert res["status"] == "created"
    assert "related" not in res


def test_cli_record_renders_degraded_notice(ws, capsys):
    """The human surface: receipt on stdout, one calm notice line after it on stderr."""
    config, _ = ws
    unavailable = Unavailable(reason=ConflictUnavailableReason.EMBEDDING, detail="quota")
    with patch.object(MitosSyncManager, "_review_neighbors", return_value=unavailable):
        cmd_record(config, axiom="A new call.", rejected="rej", slug="degcall")
    out, err = capsys.readouterr()
    assert "Recorded decision 'degcall'" in out
    assert "Near-duplicate review could not run" in err
    assert "mitos check" in err
    assert "quota" not in err  # detail is logging-only, never rendered


def test_cli_record_json_carries_degraded_notice(ws, capsys):
    """The JSON surface: the receipt dict rides verbatim, notice key included."""
    config, _ = ws
    unavailable = Unavailable(reason=ConflictUnavailableReason.VECTOR_STORE, detail="down")
    with patch.object(MitosSyncManager, "_review_neighbors", return_value=unavailable):
        cmd_record(config, axiom="A new call.", rejected="rej", slug="degjson",
                   as_json=True)
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "created"
    assert "vector store unavailable" in data["neighbor_review_unavailable"]
    assert "down" not in data["neighbor_review_unavailable"]
