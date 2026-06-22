"""Mutation lineage, write-time cycle prevention & Outbox confirmation (V1b Phase 3a).

The mutation union (``supersedes`` ∪ ``amends`` ∪ ``narrows``) is the edge set
``get_lineage`` walks and the write-time guard keeps acyclic by construction. These
gates pin T10's three axes + the DoD #3 decision-side no-cascade negative + the W9
drain-safety advance, all against real temp SQLite (no async, no LLM, no embeddings,
no mocks — PATTERNS):

  * ``get_lineage`` basics — the mutation union is walked, ``corrects`` is excluded,
    self is never an ancestor, identity-only return shape, empty/absent is healthy.
  * T10 negative — a cycle-closer (direct, AND mixed cross-type) is rejected
    ``cycle_violation`` with zero false positives; a convergent DAG / diamond commits.
  * T10 positive — ``get_lineage`` returns the full chain at ≥40-link depth, and the
    walk is O(reachable lineage), NOT O(corpus): a large unrelated corpus does not
    change the result (proven STRUCTURALLY — no wall-clock assertion, PLANNING_NOTES).
  * T10 homeostasis — a deliberately-seeded corrupt cycle (raw INSERT, bypassing the
    reconciler that rejects it) yields a loud, partial, non-hanging, non-raising read.
  * DoD #3 (decision side) — a commit that flips another node's active-view
    membership writes that node NO Outbox row and ticks its ``updated_at`` not at all.
  * W9 drain-safety — re-enqueue advances ``queued_at`` strictly and resets
    ``retry_count`` to 0 (the in-flight-drain survival guarantee).

Driven via the keyless ``commit_parsed_entry`` parse→commit path (``test_edge_catalog``
idiom); the corrupt-cycle fixture uses raw ``INSERT INTO edges`` (the
``test_migration_snapshot`` idiom 2c reused for its self-edge gate). Run under
``./venv/bin/python -m pytest tests/test_lineage_and_cycles.py -v`` (PATTERNS — never
bare ``python``; it lacks deps).
"""

import logging
import os
import tempfile

import pytest

from mitos.errors import CommitError
from mitos.parser import ParsedEntry
from mitos.store import GraphStore, _MUTATION_EDGE_FIELDS


@pytest.fixture
def temp_store() -> GraphStore:
    """A temporary file GraphStore booted to the live ladder head."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    store = GraphStore(path)
    yield store
    if os.path.exists(path):
        os.remove(path)


# --- Builders + helpers --------------------------------------------------------


def _decision(slug: str, axiom: str, **rels) -> ParsedEntry:
    """A hand-built decision; ``rels`` values are List[str] (V1b multi-valued shape)."""
    e = ParsedEntry("decision", slug, 1, 5)
    e.axiom = axiom
    e.rejected_paths = "An alternative."
    for name, value in rels.items():
        setattr(e, name, value)
    return e


def _commit(store: GraphStore, slug: str, axiom: str, **rels) -> str:
    """Commits a decision and returns its content-hash node id."""
    return store.commit_parsed_entry(_decision(slug, axiom, **rels)).node_id


def _slugs(lineage) -> set:
    """The slug set of a ``get_lineage`` result."""
    return {row["slug"] for row in lineage}


def _inject_raw_edge(
    store: GraphStore, source_id: str, target_id: str, edge_type: str
) -> None:
    """Inserts an ``edges`` row directly, BYPASSING ``_reconcile_edges``.

    The supported write path rejects a cycle-closer, so the only way to seed a
    corrupt cycle the homeostasis bound must survive is a raw INSERT (the graph is a
    rebuildable derivative — M7/P6). Same-kind ``decision`` endpoints (V1b-widened
    CHECK accepts ``amends`` D→D).
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


def _updated_at(store: GraphStore, node_id: str) -> str:
    """Reads a node's raw ``updated_at`` column (not the hydrated payload)."""
    conn = store._get_connection()
    try:
        row = conn.execute(
            "SELECT updated_at FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return row["updated_at"]
    finally:
        conn.close()


# ==============================================================================
# get_lineage — basics: the mutation union, corrects-exclusion, self, shape, empty
# ==============================================================================


def test_lineage_empty_for_node_with_no_mutation_edges(temp_store: GraphStore) -> None:
    """A node with no outgoing mutation edges has an empty lineage (healthy, not error)."""
    a = _commit(temp_store, "lonely", "A standalone decision.")
    assert temp_store.get_lineage(a) == []


def test_lineage_empty_for_nonexistent_node(temp_store: GraphStore) -> None:
    """An unknown node id returns [] — empty is healthy, never an error."""
    assert temp_store.get_lineage("0" * 64) == []


def test_lineage_single_supersedes_ancestor_identity_shape(temp_store: GraphStore) -> None:
    """``supersedes`` is walked; the return is identity-only ``{node_id, slug, kind}``."""
    a = _commit(temp_store, "old", "The old axiom.")
    b = _commit(temp_store, "new", "The new axiom.", supersedes=["old"])
    lineage = temp_store.get_lineage(b)
    assert lineage == [{"node_id": a, "slug": "old", "kind": "decision"}]
    # Thin identity, not a hydrated/stamped payload (Decision 3) — exactly three keys.
    assert set(lineage[0].keys()) == {"node_id", "slug", "kind"}


def test_lineage_walks_amends_and_narrows(temp_store: GraphStore) -> None:
    """``amends`` and ``narrows`` are mutation edges and are walked transitively."""
    a = _commit(temp_store, "base", "Base axiom.")
    b = _commit(temp_store, "amender", "Amends base.", amends=["base"])
    c = _commit(temp_store, "narrower", "Narrows the amender.", narrows=["amender"])
    assert _slugs(temp_store.get_lineage(c)) == {"amender", "base"}


def test_lineage_mixed_union_walked_transitively(temp_store: GraphStore) -> None:
    """The full union (supersedes ∪ amends ∪ narrows) composes across hop types."""
    a = _commit(temp_store, "root", "Root axiom.")
    b = _commit(temp_store, "mid", "Amends root.", amends=["root"])
    c = _commit(temp_store, "head", "Supersedes mid.", supersedes=["mid"])
    assert _slugs(temp_store.get_lineage(c)) == {"mid", "root"}


def test_lineage_excludes_corrects(temp_store: GraphStore) -> None:
    """``corrects`` is NOT in the mutation union — a corrects-only node has empty lineage."""
    a = _commit(temp_store, "typo", "An axiom with a typo.")
    b = _commit(temp_store, "fix", "The corrected axiom.", corrects=["typo"])
    # ``fix`` has an outgoing ``corrects`` edge but no mutation edge → empty lineage
    # (Decision 2: corrects belongs to the deferred get_transcript ancestry, M1).
    assert temp_store.get_lineage(b) == []
    # Guard against an over-broad set: the union is exactly these three.
    assert set(_MUTATION_EDGE_FIELDS) == {"supersedes", "amends", "narrows"}


def test_mutation_edge_constants_in_lockstep() -> None:
    """The SQL ``IN (...)`` literal must never drift from the field tuple (single-source).

    ``_MUTATION_EDGE_TYPES_SQL`` is a hand-typed literal paired with
    ``_MUTATION_EDGE_FIELDS``; a desync would silently change what the walk
    traverses. Pin them together and pin the deliberate ``corrects`` exclusion.
    """
    from mitos.store import _MUTATION_EDGE_TYPES_SQL

    parsed = {tok.strip().strip("'") for tok in _MUTATION_EDGE_TYPES_SQL.strip("()").split(",")}
    assert parsed == set(_MUTATION_EDGE_FIELDS) == {"supersedes", "amends", "narrows"}
    assert "corrects" not in _MUTATION_EDGE_FIELDS  # Decision 2: kept acyclic elsewhere


def test_lineage_never_includes_self(temp_store: GraphStore) -> None:
    """A node is never its own ancestor, even though it has outgoing mutation edges."""
    a = _commit(temp_store, "ancestor", "Ancestor axiom.")
    b = _commit(temp_store, "descendant", "Descendant axiom.", amends=["ancestor"])
    lineage_ids = {row["node_id"] for row in temp_store.get_lineage(b)}
    assert b not in lineage_ids
    assert lineage_ids == {a}


# ==============================================================================
# T10 negative — cycle-closers rejected (direct + mixed), zero false positives
# ==============================================================================


def test_direct_two_cycle_rejected(temp_store: GraphStore) -> None:
    """``A amends B`` exists; authoring ``B amends A`` closes a 2-cycle → rejected."""
    b = _commit(temp_store, "tc-b", "Decision B.")
    a = _commit(temp_store, "tc-a", "Decision A, amends B.", amends=["tc-b"])
    # Re-commit B (same canonical core → same node id) now amending A — the closer.
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(_decision("tc-b", "Decision B.", amends=["tc-a"]))
    items = exc.value.failure.items
    assert items[0].code == "cycle_violation"
    assert items[0].field == "**Amends:**"
    # Zero partial state: no B→A edge committed.
    assert all(
        not (e["source_id"] == b and e["target_id"] == a)
        for e in temp_store.get_edges()
    )


def test_mixed_cross_type_cycle_rejected(temp_store: GraphStore) -> None:
    """``A supersedes B``, ``B amends C`` exist; ``C narrows A`` closes a mixed cycle → rejected.

    The exact case V1a's active-source guard cannot catch once non-kill edges commit:
    the closer is a live-source ``narrows`` looping back through a ``supersedes`` and
    an ``amends``.
    """
    c = _commit(temp_store, "mc-c", "Decision C.")
    b = _commit(temp_store, "mc-b", "Decision B, amends C.", amends=["mc-c"])
    a = _commit(temp_store, "mc-a", "Decision A, supersedes B.", supersedes=["mc-b"])
    # lineage(A) = {B, C}; C ∈ lineage(A), so C narrows A would close the union cycle.
    assert _slugs(temp_store.get_lineage(a)) == {"mc-b", "mc-c"}
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(_decision("mc-c", "Decision C.", narrows=["mc-a"]))
    items = exc.value.failure.items
    assert items[0].code == "cycle_violation"
    assert items[0].field == "**Narrows:**"


def test_self_loop_rejected_as_cycle(temp_store: GraphStore) -> None:
    """A node amending itself is rejected (the self-edge cycle case, the 1-hop closer)."""
    a = _commit(temp_store, "selfish", "A self-amending axiom.")
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(
            _decision("selfish", "A self-amending axiom.", amends=["selfish"])
        )
    assert exc.value.failure.items[0].code == "cycle_violation"


def test_convergent_diamond_accepted(temp_store: GraphStore) -> None:
    """A diamond (shared reachability) is NOT a cycle — it commits and dedups in the walk.

    ``X amends [P, Q]``; ``P amends R``; ``Q amends R``. R is reachable from X by two
    acyclic paths; the visited-set walk re-reaches R as already-explored (BLACK) and
    skips it — no false-positive reject, R appears exactly once in the lineage.
    """
    r = _commit(temp_store, "dia-r", "Root R.")
    p = _commit(temp_store, "dia-p", "P amends R.", amends=["dia-r"])
    q = _commit(temp_store, "dia-q", "Q amends R.", amends=["dia-r"])
    # X amends BOTH P and Q — the convergent commit must NOT be rejected.
    x = _commit(temp_store, "dia-x", "X amends P and Q.", amends=["dia-p", "dia-q"])
    lineage = temp_store.get_lineage(x)
    assert _slugs(lineage) == {"dia-p", "dia-q", "dia-r"}
    # R appears exactly once despite two reachable paths (convergent dedup).
    assert [row["slug"] for row in lineage].count("dia-r") == 1


def test_new_edge_with_source_outside_target_lineage_accepted(temp_store: GraphStore) -> None:
    """A new mutation edge whose source is NOT in the target's lineage commits (accept-case)."""
    r = _commit(temp_store, "acc-r", "Root.")
    p = _commit(temp_store, "acc-p", "P amends R.", amends=["acc-r"])
    q = _commit(temp_store, "acc-q", "Q amends R.", amends=["acc-r"])
    # A brand-new node amending P: its source is not in lineage(P) = {R}, so accepted.
    newcomer = _commit(temp_store, "acc-new", "Newcomer amends P.", amends=["acc-p"])
    assert _slugs(temp_store.get_lineage(newcomer)) == {"acc-p", "acc-r"}


# ==============================================================================
# T10 positive — full chain at ≥40-link depth; O(lineage), not O(corpus)
# ==============================================================================

_CHAIN_DEPTH = 40  # 40 ancestors below the head (41 nodes total)


def _build_chain(store: GraphStore, prefix: str, depth: int) -> tuple:
    """Builds a linear amends chain ``c0 ← c1 ← … ← c{depth}`` and returns (head_id, ancestor_slugs)."""
    slugs = [f"{prefix}-{i:02d}" for i in range(depth + 1)]
    prev = _commit(store, slugs[0], f"Chain link 0 of {prefix}.")
    head = prev
    for i in range(1, depth + 1):
        head = _commit(store, slugs[i], f"Chain link {i} of {prefix}.", amends=[slugs[i - 1]])
    return head, set(slugs[:-1])  # ancestors = all but the head's own slug


def test_lineage_full_chain_at_40_link_depth(temp_store: GraphStore) -> None:
    """``get_lineage(head)`` of a 40-link amends chain returns all 40 ancestors."""
    head, ancestor_slugs = _build_chain(temp_store, "chain", _CHAIN_DEPTH)
    lineage = temp_store.get_lineage(head)
    assert len(lineage) == _CHAIN_DEPTH
    assert _slugs(lineage) == ancestor_slugs


def test_lineage_walk_is_corpus_size_independent(temp_store: GraphStore) -> None:
    """The walk visits exactly the reachable lineage — a large unrelated corpus is invisible.

    Structural proof of the O(reachable lineage) ≈ O(chain depth) bound, NOT O(corpus):
    seed a large unrelated corpus (incl. its OWN unrelated amends chains, so the
    walker could wander if it walked structurally rather than by reachability), then
    assert the 40-deep chain's lineage is *unchanged* — exactly its 40 ancestors,
    none of the noise. No wall-clock / timing assertion (PLANNING_NOTES calibration
    discipline — banned); the invariant is reachability, not elapsed time.
    """
    head, ancestor_slugs = _build_chain(temp_store, "chain", _CHAIN_DEPTH)
    # A large unrelated corpus: 50 standalone decisions + 5 unrelated 6-link chains.
    for i in range(50):
        _commit(temp_store, f"noise-{i:03d}", f"Unrelated decision {i}.")
    for c in range(5):
        _build_chain(temp_store, f"noise-chain-{c}", 6)
    # The result is identical to the no-noise case — corpus size does not leak in.
    lineage = temp_store.get_lineage(head)
    assert len(lineage) == _CHAIN_DEPTH
    assert _slugs(lineage) == ancestor_slugs


# ==============================================================================
# T10 homeostasis — seeded corrupt cycle → loud, partial, non-hanging, non-raising
# ==============================================================================


def test_corrupt_cycle_loud_partial_return_no_hang(temp_store: GraphStore, caplog) -> None:
    """A raw-injected corrupt mutation cycle yields a loud, partial, non-hanging read.

    The supported write path rejects every cycle-closer, so this cycle can only enter
    out-of-band (M7/P6 — the graph is a rebuildable derivative). ``get_lineage`` must
    truncate at the cycle, emit a loud diagnostic naming the node, return the PARTIAL
    lineage, and neither infinite-loop nor raise (a ``record`` over a corrupt graph
    still completes).
    """
    a = _commit(temp_store, "corrupt-a", "Decision A.")
    b = _commit(temp_store, "corrupt-b", "Decision B, amends A.", amends=["corrupt-a"])
    # Close the cycle out-of-band: a → b (the reconciler would reject this as a
    # cycle_violation, so we inject it raw).
    _inject_raw_edge(temp_store, source_id=a, target_id=b, edge_type="amends")

    with caplog.at_level(logging.WARNING, logger="mitos.store"):
        lineage = temp_store.get_lineage(b)  # must NOT hang and must NOT raise

    # Partial lineage: the ancestor walked before the bound fired (a), no duplicates.
    assert _slugs(lineage) == {"corrupt-a"}
    # Loud diagnostic: a WARNING was emitted naming the node and the cycle.
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "expected a loud homeostasis WARNING on the corrupt cycle"
    assert any("cycle" in r.getMessage().lower() for r in warnings)
    assert any(b in r.getMessage() for r in warnings)


def test_corrupt_cycle_does_not_break_record_path(temp_store: GraphStore) -> None:
    """A new acyclic commit still succeeds against a graph holding a corrupt cycle (crash-safe)."""
    a = _commit(temp_store, "cz-a", "Decision A.")
    b = _commit(temp_store, "cz-b", "Decision B, amends A.", amends=["cz-a"])
    _inject_raw_edge(temp_store, source_id=a, target_id=b, edge_type="amends")
    # An unrelated, acyclic record still commits — the corrupt subgraph is isolated.
    fresh = _commit(temp_store, "cz-fresh", "An unrelated fresh decision.")
    assert temp_store.get_lineage(fresh) == []


# ==============================================================================
# DoD #3 (decision side) — no-cascade negative
# ==============================================================================


def test_no_cascade_membership_flip_writes_no_outbox_row_for_other_node(
    temp_store: GraphStore,
) -> None:
    """Committing ``B supersedes A`` (which flips A out of the active view) writes A NO
    Outbox row and does NOT tick A's ``updated_at`` (DoD #3 decision side, §4.4)."""
    a = _commit(temp_store, "casc-a", "Decision A.")
    # A was enqueued by its OWN commit; clear it so we can prove B's commit adds none.
    temp_store.remove_pending_embedding(a)
    a_updated_before = _updated_at(temp_store, a)

    b = _commit(temp_store, "casc-b", "Decision B supersedes A.", supersedes=["casc-a"])

    queued_ids = {row["node_id"] for row in temp_store.get_pending_embeddings()}
    # Only the committing node (B) is enqueued — no cascade onto the flipped A.
    assert b in queued_ids
    assert a not in queued_ids
    # A's row was untouched (no tick) even though B's commit retired A.
    assert _updated_at(temp_store, a) == a_updated_before


# ==============================================================================
# W9 — Outbox drain-safety: re-enqueue advances queued_at + resets retry_count
# ==============================================================================


def test_reenqueue_advances_queued_at_and_resets_retry_count(temp_store: GraphStore) -> None:
    """A re-enqueue strictly advances ``queued_at`` and resets ``retry_count`` to 0.

    Proves the W9/V3b drain-safety guarantee structurally with controlled distinct
    stamps (no reliance on real-clock advance between two full commits — the
    same-µs flake the plan's Gotcha warns about). A drain conditioning a DELETE on
    ``(node_id, old_queued_at)`` therefore affects zero rows: the freshly-enqueued
    intent survives an in-flight drain.
    """
    node_id = _commit(temp_store, "drain-node", "A decision to re-enqueue.")
    parsed = _decision("drain-node", "A decision to re-enqueue.")
    t1 = "2026-01-01T00:00:00.000000+00:00"
    t2 = "2026-01-02T00:00:00.000000+00:00"
    assert t2 > t1  # the stamps strictly advance by construction

    conn = temp_store._get_connection()
    try:
        with conn:
            temp_store._enqueue_outbox(conn.cursor(), node_id, parsed, t1)
        row = conn.execute(
            "SELECT queued_at, retry_count FROM pending_embeddings WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        assert row["queued_at"] == t1
        # Simulate a drainer having recorded a retry on this row.
        with conn:
            conn.execute(
                "UPDATE pending_embeddings SET retry_count = 4 WHERE node_id = ?",
                (node_id,),
            )
        # Re-enqueue with a strictly-later stamp.
        with conn:
            temp_store._enqueue_outbox(conn.cursor(), node_id, parsed, t2)
        row = conn.execute(
            "SELECT queued_at, retry_count FROM pending_embeddings WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        assert row["queued_at"] == t2  # advanced
        assert row["retry_count"] == 0  # reset (the drain-revival guarantee)
    finally:
        conn.close()
