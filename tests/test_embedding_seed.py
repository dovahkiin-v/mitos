"""The two substrate questions a covering write needs answered.

Phase 1c reserves creating an absent Qdrant collection for a write that **covers the
workspace's active set**, and the two callers derive that from two deliberately
different sources — which is the whole design, not an inconsistency:

* a **single-node** write (a ``record``, an ``import`` entry) asks the graph:
  ``has_active_node_other_than(node_id)``. Its answer is a property of the running
  write, and it must agree with ``get_active_node_ids()`` exactly, because a
  disagreement in either direction is the defect (claiming coverage it does not deliver,
  or refusing the fresh-project carve-out).
* a **drain** asks the ``embedding_seed`` marker, which records that a *prior* command
  bounded the outbox to the active set. A drain cannot see its own covering-ness from
  inside — ``mitos sync --embed-only`` has no buffer at all — so the knowledge travels to
  the substrate rather than being re-derived there.

These rows pin the marker's lifecycle and the gate's agreement. The write paths that
consume them are pinned in ``tests/test_covering_write_gate.py``; the end-to-end
listing-invariance claims are live-tier (``tests/test_collection_absence_live.py``).

Run under ``./venv/bin/python -m pytest`` (PATTERNS — bare ``python`` lacks deps).
"""

import os
import re
import sqlite3
from typing import List, Optional, Set

import pytest

from mitos.config import MitosConfig
from mitos.errors import DatabaseError
from mitos.migrations import EMBEDDING_SEED_ESTABLISHED_BY
from mitos.parser import ParsedEntry
from mitos.store import GraphStore, open_connection


@pytest.fixture
def store(tmp_path) -> GraphStore:
    """A fresh graph, laddered to the live head."""
    db_path = str(tmp_path / ".mitos" / "graph.sqlite")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return GraphStore(db_path)


def _decision(
    store: GraphStore,
    slug: str,
    axiom: str,
    *,
    supersedes: Optional[List[str]] = None,
    resolves: Optional[List[str]] = None,
) -> str:
    """Commits a minimal decision through parse→commit and returns its node id."""
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = axiom
    entry.rejected_paths = "None."
    if supersedes:
        entry.supersedes = supersedes
    if resolves:
        entry.resolves = resolves
    return store.commit_parsed_entry(entry).node_id


def _open_question(store: GraphStore, slug: str, topic: str, question: str) -> str:
    """Commits a minimal open question and returns its node id."""
    entry = ParsedEntry("open_question", slug, 1, 5)
    entry.topic = topic
    entry.questions_raised = [question]
    return store.commit_parsed_entry(entry).node_id


def _seed_row_count(store: GraphStore) -> int:
    """Counts marker rows directly — the single-row property must be structural."""
    conn = open_connection(store.db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM embedding_seed;").fetchone()[0]
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# The marker — lifecycle and serialization contract
# --------------------------------------------------------------------------- #

def test_a_fresh_graph_has_the_table_and_no_marker(store: GraphStore) -> None:
    """Healthy and empty: the table ships present, and no covering act stands.

    Absence of the row is the false state. There is deliberately no ``covers = 0`` row —
    a two-valued column would let a stale ``0`` and a missing row mean the same thing two
    ways, and only one of them would ever be looked at.
    """
    assert store.embedding_seed() is None
    assert _seed_row_count(store) == 0


def test_stamp_read_clear_roundtrip(store: GraphStore) -> None:
    """stamp → read back → clear → read back None, with strings on both sides."""
    store.stamp_embedding_seed("rebuild")

    seed = store.embedding_seed()
    assert seed is not None
    assert seed["established_by"] == "rebuild"
    # Plain strings cross the boundary — no datetime objects, no tuples. The stamp is
    # application-supplied UTC ISO-8601 (MI-10), not a DDL default.
    assert isinstance(seed["established_at"], str)
    assert seed["established_at"].endswith("+00:00")
    assert set(seed) == {"established_by", "established_at"}

    store.clear_embedding_seed()
    assert store.embedding_seed() is None
    assert _seed_row_count(store) == 0


def test_clearing_an_absent_marker_is_a_noop(store: GraphStore) -> None:
    """The drain clears on every drain-to-empty, so the no-marker case must be quiet."""
    store.clear_embedding_seed()
    store.clear_embedding_seed()
    assert store.embedding_seed() is None


def test_a_second_stamp_replaces_rather_than_raising(store: GraphStore) -> None:
    """Re-stamping records the LATEST covering act; it never accumulates or raises.

    A ``rebuild`` followed by a ``reconcile`` before either drains is an ordinary
    operator sequence, and the marker is a claim about the current outbox, not a history.
    """
    store.stamp_embedding_seed("rebuild")
    first = store.embedding_seed()
    assert first is not None

    store.stamp_embedding_seed("reconcile")

    second = store.embedding_seed()
    assert second is not None
    assert second["established_by"] == "reconcile"
    assert _seed_row_count(store) == 1  # replaced, never a second row


def test_every_declared_verb_is_accepted(store: GraphStore) -> None:
    """Each member of the vocabulary round-trips — no dead member in the CHECK."""
    for verb in EMBEDDING_SEED_ESTABLISHED_BY:
        store.stamp_embedding_seed(verb)
        seed = store.embedding_seed()
        assert seed is not None and seed["established_by"] == verb


def test_an_unknown_established_by_is_refused_at_write_time(store: GraphStore) -> None:
    """Adversarial row (success criterion 11): a malformed verb never lands.

    The marker is the ONLY thing standing between a swept project and a permanently
    unindexed corpus once the inline path can no longer create, so a row nobody can
    interpret must not exist at all. The ``CHECK`` refuses it, and the boundary error
    names the vocabulary rather than leaking a bare SQLite string.
    """
    with pytest.raises(DatabaseError) as excinfo:
        store.stamp_embedding_seed("mitos-sync-probably")

    for verb in EMBEDDING_SEED_ESTABLISHED_BY:
        assert verb in str(excinfo.value)
    assert store.embedding_seed() is None
    assert _seed_row_count(store) == 0


def test_the_single_row_property_is_structural_not_conventional(store: GraphStore) -> None:
    """``CHECK (id = 1)`` — a second row cannot be smuggled in past the accessors."""
    conn = open_connection(store.db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO embedding_seed (id, established_by, established_at) "
                "VALUES (2, 'rebuild', '2026-07-30T00:00:00.000000+00:00');"
            )
    finally:
        conn.close()
    assert _seed_row_count(store) == 0


def test_the_ddl_declares_no_current_timestamp_default(store: GraphStore) -> None:
    """MI-10: every timestamp is application-supplied; the DDL carries no default."""
    conn = open_connection(store.db_path)
    try:
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='embedding_seed';"
        ).fetchone()[0]
    finally:
        conn.close()
    assert "CURRENT_TIMESTAMP" not in sql.upper()
    assert "STRICT" in sql.upper()


def test_the_ddl_check_set_equals_the_python_constant(store: GraphStore) -> None:
    """The meta-test the vocabulary needs: one set, interpolated, never retyped.

    ``established_by`` is the only value crossing a component boundary here — three
    producers write it and one ``CHECK`` accepts it. A literal retyped into the DDL would
    pass every unit test that stamps through the same Python constant and only red at
    runtime, on an operator's machine, mid-heal. So read the constraint text back out of
    what SQLite actually stored and compare it to the constant.
    """
    conn = open_connection(store.db_path)
    try:
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='embedding_seed';"
        ).fetchone()[0]
    finally:
        conn.close()

    match = re.search(r"CHECK\s*\(\s*established_by\s+IN\s*\(([^)]*)\)", sql, re.I)
    assert match is not None, f"no established_by CHECK found in:\n{sql}"
    declared = {token.strip().strip("'") for token in match.group(1).split(",")}
    assert declared == set(EMBEDDING_SEED_ESTABLISHED_BY)


def test_an_unreadable_marker_fails_closed_toward_deferral(store: GraphStore) -> None:
    """Criterion 11's second half: a hand-truncated graph defers, never creates.

    Once the inline path can no longer create, the marker is the sole authority for the
    drain — so the interesting question is which way it fails when it cannot be read at
    all. Toward deferral: an unreadable marker is reported as *absent*, because the safe
    answer to "may this write create the index?" is no, and deferral costs only an outbox
    row that was already written. Raising here would turn a recoverable graph into a
    failed ``sync`` while still creating nothing — strictly worse, with no compensation.
    """
    store.stamp_embedding_seed("rebuild")
    assert store.embedding_seed() is not None

    conn = open_connection(store.db_path)
    try:
        conn.execute("DROP TABLE embedding_seed;")
        conn.commit()
    finally:
        conn.close()

    assert store.embedding_seed() is None


def test_all_three_producers_are_live_and_their_set_is_the_vocabulary(tmp_path) -> None:
    """Every declared verb has a real producer, and no producer emits an undeclared one.

    Drives all three write sites for real — ``reconcile``'s enqueue pass and the
    ``rebuild``/``cutover`` prune — and asserts the observed set is exactly the declared
    one. That makes the vocabulary a closed loop rather than three literals that happen
    to match today: a member added without a producer reds here, and a producer emitting
    something unlisted reds at the ``CHECK``.
    """
    from unittest.mock import MagicMock

    from mitos.cutover import _prune_embedding_queue_to_active
    from mitos.sync import MitosSyncManager

    observed: Set[str] = set()

    # --- rebuild / cutover: the prune stamps in its own transaction ---
    for expected in ("rebuild", "cutover"):
        ws = tmp_path / f"prune-{expected}"
        os.makedirs(ws / ".mitos", exist_ok=True)
        db_path = str(ws / ".mitos" / "graph.sqlite")
        aside_store = GraphStore(db_path)
        node_id = _decision(aside_store, "d1", "An axiom.")
        _prune_embedding_queue_to_active(
            db_path, {node_id}, established_by=expected
        )
        seed = GraphStore(db_path).embedding_seed()
        assert seed is not None
        observed.add(seed["established_by"])

    # --- reconcile: stamped after enqueuing, before the drain ---
    ws = tmp_path / "reconcile"
    os.makedirs(ws / ".mitos", exist_ok=True)
    config = MitosConfig(str(ws))
    graph = GraphStore(config.db_path)
    _decision(graph, "d1", "An axiom.")
    for row in graph.get_pending_embeddings():
        graph.remove_pending_embedding(row["node_id"])

    manager = MitosSyncManager(config)
    manager.embed_provider = MagicMock()
    manager.embed_provider.get_embedding = MagicMock(return_value=[0.1, 0.2, 0.3])
    manager.vector_store = MagicMock()
    manager.vector_store.list_point_ids = MagicMock(return_value=set())
    # Capture the marker as the drain saw it: the drain clears it on the way out, so
    # reading afterwards would find nothing and prove nothing.
    seen: List[Optional[str]] = []
    real_drain = manager.drain_pending_embeddings

    def _spy_drain() -> None:
        seed = manager.store.embedding_seed()
        seen.append(seed["established_by"] if seed else None)
        real_drain()

    manager.drain_pending_embeddings = _spy_drain  # type: ignore[method-assign]
    manager.reconcile_embeddings()

    assert seen == ["reconcile"]
    observed.add("reconcile")

    assert observed == set(EMBEDDING_SEED_ESTABLISHED_BY)


# --------------------------------------------------------------------------- #
# The graph gate — it must agree with the shipped active set, exactly
# --------------------------------------------------------------------------- #

def _agrees(store: GraphStore, node_id: str) -> bool:
    """The gate's answer, and the definition it must never diverge from."""
    gate = store.has_active_node_other_than(node_id)
    reference = len(store.get_active_node_ids() - {node_id}) > 0
    assert gate == reference, (
        f"has_active_node_other_than={gate} but the shipped active set says "
        f"{reference} — the gate and get_active_node_ids have diverged"
    )
    return gate


def test_an_empty_graph_covers_its_first_node(store: GraphStore) -> None:
    """The fresh-project carve-out: with nothing else live, one point IS the active set."""
    node_id = _decision(store, "first", "The first axiom.")
    assert _agrees(store, node_id) is False


def test_a_second_node_is_not_covered_by_a_single_write(store: GraphStore) -> None:
    """A populated graph: one write leaves live architecture unindexed, so it defers."""
    _decision(store, "first", "The first axiom.")
    second = _decision(store, "second", "The second axiom.")
    assert _agrees(store, second) is True


def test_a_superseded_node_does_not_hold_the_gate_open(store: GraphStore) -> None:
    """Liveness is the kill-edge anti-join, so a retired node needs no embedding."""
    _decision(store, "old", "The old axiom.")
    new = _decision(store, "new", "The new axiom.", supersedes=["old"])
    # `old` is superseded, so `new` alone IS the active set.
    assert store.get_active_node_ids() == {new}
    assert _agrees(store, new) is False


def test_a_parked_open_question_holds_the_gate_open(store: GraphStore) -> None:
    """Open questions are in the active set too — a decisions-only gate would over-create."""
    _open_question(store, "an-unsettled-topic", "Caching", "Which cache?")
    decision = _decision(store, "d1", "An unrelated axiom.")
    assert _agrees(store, decision) is True


def test_a_RESOLVED_open_question_still_holds_the_gate_open(store: GraphStore) -> None:
    """Success criterion 14, inverted — and this is the row that had to be measured.

    ``resolves`` is **not** a kill edge, so a resolved open question is still returned by
    ``get_open_questions`` and is still in ``get_active_node_ids`` (its own docstring says
    so: *"``resolved`` OQs are RETURNED; callers foreground ``parked``"*). It therefore
    genuinely needs an embedding, and a fresh project holding one plus its first decision
    must **defer**, not create — a one-point collection over a two-node active set is
    exactly the partial index this phase exists to prevent.

    The plan predicted the opposite ("still creates") from a ``_OQ_RESOLVED_PREDICATE``
    clause that belongs to ``get_scope_counts``' parked aggregate, not to the active-view
    read. Adding that clause to the gate would have made it disagree with the shipped
    active set — the same defect, mirrored, on a path no developer reaches by hand. So the
    assertion is pinned against ``get_active_node_ids()`` itself rather than against a
    hand-written expectation.
    """
    oq_id = _open_question(store, "an-unsettled-topic", "Caching", "Which cache?")
    decision = _decision(
        store, "we-use-lru", "We use an LRU cache.", resolves=["an-unsettled-topic"]
    )

    # The OQ reads as resolved…
    states = {q["slug"]: q["state"] for q in store.get_open_questions()}
    assert states == {"an-unsettled-topic": "resolved"}
    # …and is nonetheless active, which is what the gate must honour.
    assert store.get_active_node_ids() == {oq_id, decision}
    assert _agrees(store, decision) is True


def test_the_gate_agrees_with_the_active_set_across_the_matrix(store: GraphStore) -> None:
    """One row walking the whole fixture matrix, so agreement is a gate not a claim.

    Each step re-asserts the equivalence for every node in the graph, including the
    inactive ones — the gate is asked about arbitrary ids, not only about live ones.
    """
    def _check_all() -> None:
        for node in store.get_all_nodes():
            _agrees(store, node["id"])
        # An id that is not in the graph at all must still answer honestly.
        _agrees(store, "f" * 64)

    _check_all()  # empty graph
    _decision(store, "d1", "Axiom one.")
    _check_all()
    _open_question(store, "q1", "Caching", "Which cache?")
    _check_all()
    _decision(store, "d2", "Axiom two.", resolves=["q1"])
    _check_all()
    _decision(store, "d3", "Axiom three.", supersedes=["d1"])
    _check_all()
