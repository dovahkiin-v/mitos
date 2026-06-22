"""Tests for the Mitos SQLite GraphStore — V1a per-entry commit core (Phase 5a).

Covers the Phase 5a rebuild of ``commit_parsed_entry`` against the live V1a STRICT
schema (the dual deferred-flip — entry-001 schema + entry-002 identity — landed
here): slug-free content-hash identity (``compute_node_id``), the in-place
commentary UPDATE with the MI-4 canonical-core + ``source`` fence, ``node_scopes``
reconciliation (MI-9), write-once-preserve ``transcripts`` (V1-D16), the
idempotent no-op (MI-3 / V1-D17), the ``source`` enum coverage matrix (V1-D20),
atomic rollback (V1-D10), and the 5b/5c deferred seams' forcing-function tests.

Read methods are unavailable until Phase 5d, so node/scope/transcript assertions
go through **raw SQL** on the store's own connection. The connection-suite /
PRAGMA / version-guard tests (Phase 2a) are schema-agnostic and stay green.
"""

import sqlite3
import tempfile
import os
import json
import logging
import pytest
from mitos.store import GraphStore, ValidationError, compute_hash
from mitos.identity import compute_node_id
from mitos.errors import DatabaseError, CommitError
from mitos.parser import ParsedEntry, parse_entry_stream


@pytest.fixture
def temp_store() -> GraphStore:
    """Initializes a temporary file GraphStore (boots the V1a schema post-5a)."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    store = GraphStore(path)
    yield store
    if os.path.exists(path):
        os.remove(path)


# --- Test builders + raw-SQL read helpers (reads unavailable until 5d) ---------


def _decision(
    slug: str = "d-slug",
    axiom: str = "An axiom.",
    rejected: str = "An alternative.",
    mechanisms=None,
    scope=None,
    source=None,
    transcript=None,
    invalidates_if=None,
    context=None,
) -> ParsedEntry:
    """Builds a hand-made decision ``ParsedEntry`` on the V1a (``axiom``) surface."""
    e = ParsedEntry("decision", slug, 1, 5)
    e.axiom = axiom
    e.rejected_paths = rejected
    e.mechanisms = list(mechanisms) if mechanisms else []
    e.scope = list(scope) if scope else []
    e.source = source
    e.transcript = transcript
    e.invalidates_if = invalidates_if
    e.context = context
    return e


def _open_question(
    slug: str = "oq-slug", topic: str = "A topic.", questions=None, scope=None
) -> ParsedEntry:
    """Builds a hand-made open_question ``ParsedEntry`` on the V1a surface."""
    e = ParsedEntry("open_question", slug, 1, 5)
    e.topic = topic
    e.questions_raised = list(questions) if questions else ["A question?"]
    e.scope = list(scope) if scope else []
    return e


def _node_row(store: GraphStore, node_id: str):
    """Reads a single ``nodes`` row as a dict via raw SQL, or None."""
    conn = store._get_connection()
    try:
        row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _scopes(store: GraphStore, node_id: str):
    """Reads a node's ``node_scopes`` tags as a sorted list via raw SQL."""
    conn = store._get_connection()
    try:
        return sorted(
            r["scope"]
            for r in conn.execute(
                "SELECT scope FROM node_scopes WHERE node_id = ?", (node_id,)
            )
        )
    finally:
        conn.close()


def _transcript(store: GraphStore, node_id: str):
    """Reads a node's stored transcript text via raw SQL, or None."""
    conn = store._get_connection()
    try:
        row = conn.execute(
            "SELECT transcript_text FROM transcripts WHERE node_id = ?", (node_id,)
        ).fetchone()
        return row["transcript_text"] if row else None
    finally:
        conn.close()


def _count(store: GraphStore, table: str) -> int:
    """Counts rows in a table via raw SQL (table name is a code-internal literal)."""
    conn = store._get_connection()
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def _edges(store: GraphStore):
    """Reads all ``edges`` rows as dicts via raw SQL (read methods quarantined until 5d)."""
    conn = store._get_connection()
    try:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM edges ORDER BY source_id, target_id, edge_type"
            )
        ]
    finally:
        conn.close()


def _is_active(store: GraphStore, node_id: str) -> bool:
    """True iff the node has no incoming kill-edge (the inline active-view anti-join)."""
    conn = store._get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM edges WHERE target_id = ? "
            "AND edge_type IN ('supersedes', 'corrects') LIMIT 1",
            (node_id,),
        ).fetchone()
        return row is None
    finally:
        conn.close()


def _pending(store: GraphStore):
    """Reads all ``pending_embeddings`` (Outbox) rows as dicts via raw SQL.

    Read methods stay quarantined until 5d, so the 5c Outbox assertions go through
    raw SQL on the store's own connection (the ``_edges``/``_scopes`` pattern).
    """
    conn = store._get_connection()
    try:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM pending_embeddings ORDER BY node_id"
            )
        ]
    finally:
        conn.close()


# ===========================================================================
# Phase 5a — per-entry commit core (node identity, commentary, scope, transcript)
# ===========================================================================


def test_new_decision_inserts_one_node(temp_store: GraphStore) -> None:
    """A decision commits one nodes row at the slug-free compute_node_id id."""
    e = _decision(
        slug="core-isolation",
        axiom="We will isolate the pure logic core.",
        rejected="pgvector, or direct coupling.",
        mechanisms=["sqlite", "wal"],
        scope=["substrate"],
    )
    delta = temp_store.commit_parsed_entry(e)

    assert delta.node_id == compute_node_id(
        kind="decision", axiom=e.axiom, mechanism_refs=e.mechanisms
    )
    assert delta.node_id != compute_hash(
        "decision", "core-isolation", e.axiom, e.mechanisms
    )
    assert delta.node_scope == ["substrate"]
    assert delta.commentary_fields_changed is False  # a fresh INSERT is not a "change"
    assert _count(temp_store, "nodes") == 1

    row = _node_row(temp_store, delta.node_id)
    assert row["kind"] == "decision"
    assert row["slug"] == "core-isolation"
    assert row["slug_casefold"] == "core-isolation"
    assert row["axiom"] == e.axiom
    assert json.loads(row["mechanism_refs_json"]) == ["sqlite", "wal"]
    assert row["rejected_paths_json"] == "pgvector, or direct coupling."
    assert row["source"] == "user"  # absent **Source:** -> "user" (V1-D20)
    assert row["created_at"] == row["updated_at"]  # one stamp on INSERT (MI-10)
    # off-kind columns are NULL (not json.dumps("") of an absent core field)
    assert row["topic"] is None
    assert row["questions_raised_json"] is None
    assert _scopes(temp_store, delta.node_id) == ["substrate"]


def test_new_open_question_inserts_one_node(temp_store: GraphStore) -> None:
    """An open_question commits topic + questions; decision columns are NULL."""
    e = _open_question(
        slug="auth-roadblock",
        topic="Session handling",
        questions=["How do we handle sessions?", "Stateless or stateful?"],
        scope=["auth"],
    )
    delta = temp_store.commit_parsed_entry(e)

    assert delta.node_id == compute_node_id(
        kind="open_question", topic=e.topic, questions_raised=e.questions_raised
    )
    row = _node_row(temp_store, delta.node_id)
    assert row["kind"] == "open_question"
    assert row["topic"] == "Session handling"
    assert json.loads(row["questions_raised_json"]) == e.questions_raised
    assert row["axiom"] is None
    assert row["mechanism_refs_json"] is None
    assert row["rejected_paths_json"] is None
    assert row["created_at"] == row["updated_at"]
    assert _scopes(temp_store, delta.node_id) == ["auth"]


def test_commentary_update_slug_rename_same_id(temp_store: GraphStore) -> None:
    """Same core + new slug → one node, slug renamed in place, updated_at ticks (V1-D16)."""
    e1 = _decision(slug="use-sqlite", axiom="Use SQLite.", mechanisms=["sqlite"])
    d1 = temp_store.commit_parsed_entry(e1)
    row1 = _node_row(temp_store, d1.node_id)

    e2 = _decision(slug="use-sqlite-renamed", axiom="Use SQLite.", mechanisms=["sqlite"])
    d2 = temp_store.commit_parsed_entry(e2)

    assert d2.node_id == d1.node_id  # slug excluded from identity (Q5)
    assert d2.commentary_fields_changed is True
    assert _count(temp_store, "nodes") == 1  # a rename, not a second node

    row2 = _node_row(temp_store, d1.node_id)
    assert row2["slug"] == "use-sqlite-renamed"
    assert row2["slug_casefold"] == "use-sqlite-renamed"
    assert row2["axiom"] == "Use SQLite."  # canonical core unchanged (fenced)
    assert row2["created_at"] == row1["created_at"]  # created_at stable
    assert row2["updated_at"] > row1["updated_at"]  # updated_at ticked (V1-D17)


def test_mi5_commentary_update_emits_no_edge(temp_store: GraphStore) -> None:
    """MI-5: a commentary-only update touches ``edges`` zero times (§5.3 / §12 anchor).

    An in-place commentary UPDATE — slug/scope/context changes on the same
    canonical core, no declared relationship field — is a node + ``node_scopes``
    mutation only. Edges are reserved for declared relations, so a commentary edit
    must never write (or churn) an ``edges`` row. This is the dedicated MI-5
    verification anchor the closeout §5.3 audit names (distinct from the
    failed-commit rollback cases, which leave 0 edges for a different reason).
    """
    e1 = _decision(slug="d", axiom="A.", scope=["one"], context="First.")
    d1 = temp_store.commit_parsed_entry(e1)
    assert _count(temp_store, "edges") == 0

    # Same core, changed commentary (slug rename + scope + context) → an in-place
    # UPDATE (one node), and STILL zero edges — the commentary edit emits no edge.
    e2 = _decision(slug="d-renamed", axiom="A.", scope=["two"], context="Revised.")
    d2 = temp_store.commit_parsed_entry(e2)
    assert d2.node_id == d1.node_id
    assert d2.commentary_fields_changed is True
    assert _count(temp_store, "nodes") == 1
    assert _count(temp_store, "edges") == 0


def test_mi4_source_fence_and_core_change(temp_store: GraphStore) -> None:
    """source is fenced on a same-id re-commit; a changed axiom mints a new node."""
    e1 = _decision(slug="d", axiom="Axiom A.", source=None)  # -> "user"
    d1 = temp_store.commit_parsed_entry(e1)
    assert _node_row(temp_store, d1.node_id)["source"] == "user"

    # same canonical core, different **Source:** -> stored source unchanged (MI-4)
    e2 = _decision(slug="d", axiom="Axiom A.", source="capture_llm")
    d2 = temp_store.commit_parsed_entry(e2)
    assert d2.node_id == d1.node_id
    assert _node_row(temp_store, d1.node_id)["source"] == "user"

    # changed axiom -> a NEW id (new node), not an in-place update. A DISTINCT
    # slug is used: under 5b two active nodes may not share a casefold(slug), so a
    # same-slug different-axiom pair is an independent slug_collision (V1-D4 case 3,
    # covered by test_independent_collision_rolls_back) — not what this test probes.
    e3 = _decision(slug="d-b", axiom="Axiom B.")
    d3 = temp_store.commit_parsed_entry(e3)
    assert d3.node_id != d1.node_id
    assert _count(temp_store, "nodes") == 2


def test_idempotent_recommit_is_noop(temp_store: GraphStore) -> None:
    """A byte-identical re-commit is a true no-op — updated_at does not tick (MI-3)."""
    e = _decision(slug="d", axiom="A.", mechanisms=["m"], scope=["s"], transcript="T")
    d1 = temp_store.commit_parsed_entry(e)
    before = _node_row(temp_store, d1.node_id)["updated_at"]

    e2 = _decision(slug="d", axiom="A.", mechanisms=["m"], scope=["s"], transcript="T")
    d2 = temp_store.commit_parsed_entry(e2)

    assert d2.node_id == d1.node_id
    assert d2.commentary_fields_changed is False
    assert _count(temp_store, "nodes") == 1
    assert _node_row(temp_store, d1.node_id)["updated_at"] == before  # no tick


def test_scope_reconciliation_converges_and_casefolds(temp_store: GraphStore) -> None:
    """Scopes casefold-collapse on commit and converge idempotently on re-commit (MI-9)."""
    e = _decision(slug="d", axiom="A.", scope=["Substrate", "substrate", "Auth"])
    d = temp_store.commit_parsed_entry(e)
    assert _scopes(temp_store, d.node_id) == ["auth", "substrate"]  # casefold collapse

    # re-commit with one tag removed and one added -> rows converge
    e2 = _decision(slug="d", axiom="A.", scope=["substrate", "render"])
    d2 = temp_store.commit_parsed_entry(e2)
    assert d2.node_id == d.node_id
    assert _scopes(temp_store, d.node_id) == ["render", "substrate"]
    assert d2.commentary_fields_changed is True  # a scope change is a footprint change


def test_transcript_write_once_preserve(temp_store: GraphStore) -> None:
    """Transcripts are write-once-preserve: a strip-and-re-sync never deletes (V1-D16)."""
    e = _decision(slug="d", axiom="A.", transcript="ORIGINAL")
    d = temp_store.commit_parsed_entry(e)
    assert _transcript(temp_store, d.node_id) == "ORIGINAL"
    assert _count(temp_store, "transcripts") == 1

    # strip the [DECISION_TRANSCRIPT] block and re-sync -> row PRESERVED
    e2 = _decision(slug="d", axiom="A.", transcript=None)
    d2 = temp_store.commit_parsed_entry(e2)
    assert _transcript(temp_store, d.node_id) == "ORIGINAL"
    assert d2.commentary_fields_changed is False  # absent transcript is not a change

    # a changed transcript -> updated
    e3 = _decision(slug="d", axiom="A.", transcript="REVISED")
    d3 = temp_store.commit_parsed_entry(e3)
    assert _transcript(temp_store, d.node_id) == "REVISED"
    assert d3.commentary_fields_changed is True


@pytest.mark.parametrize(
    "source_line, expected",
    [
        ("**Source:** user\n", "user"),
        ("**Source:** capture_llm\n", "capture_llm"),
        ("**Source:** import_llm\n", "import_llm"),
        ("", "user"),  # absent -> "user"
    ],
)
def test_source_enum_coverage_through_parse(
    temp_store: GraphStore, source_line: str, expected: str
) -> None:
    """All three source enum values (plus absent→user) flow through parse→commit (Lesson 13)."""
    text = (
        "### src-test\n"
        "**Decided:** an axiom for source coverage\n"
        "**Rejected:** an alternative\n"
        + source_line
    )
    entries = parse_entry_stream(text, "decision")
    assert len(entries) == 1
    delta = temp_store.commit_parsed_entry(entries[0])
    assert _node_row(temp_store, delta.node_id)["source"] == expected


def test_source_out_of_enum_rejected_by_ddl_check(temp_store: GraphStore) -> None:
    """An out-of-enum **Source:** is rejected by the DDL CHECK and the entry rolls back."""
    text = (
        "### bad-src\n"
        "**Decided:** an axiom\n"
        "**Rejected:** an alternative\n"
        "**Source:** banana\n"
    )
    entries = parse_entry_stream(text, "decision")
    with pytest.raises(DatabaseError) as exc:
        temp_store.commit_parsed_entry(entries[0])
    assert "banana" in str(exc.value) or "source" in str(exc.value)
    assert _count(temp_store, "nodes") == 0  # nothing committed


def test_structural_guard_rejects_empty_canonical_core(temp_store: GraphStore) -> None:
    """A hand-built entry that bypassed the parser (empty axiom) fails with a clear vector."""
    e = ParsedEntry("decision", "no-axiom", 1, 5)  # .axiom defaults to ""
    e.rejected_paths = "alt"
    with pytest.raises(ValidationError):
        temp_store.commit_parsed_entry(e)
    assert _count(temp_store, "nodes") == 0


# --- Deferred-seam forcing functions (Decision 4) ------------------------------


def test_5b_supersedes_commits_one_kill_edge(temp_store: GraphStore) -> None:
    """FORCING-FUNCTION INVERSION — Phase 5b wires the kill-edge.

    Through 5a the ``_reconcile_edges`` seam was a no-op and a declared
    ``Supersedes:`` committed ZERO edges (``test_5a_commits_no_edges_yet``). 5b
    fills the seam: a declared kill-edge now commits exactly one ``edges`` row
    new→old and removes the target from the active view. This is the conscious
    inversion of the 5a tripwire.
    """
    old = temp_store.commit_parsed_entry(_decision(slug="old-choice", axiom="Old."))
    e = _decision(slug="new-choice", axiom="New.")
    e.supersedes = ["old-choice"]
    new = temp_store.commit_parsed_entry(e)

    rows = _edges(temp_store)
    assert len(rows) == 1
    edge = rows[0]
    assert edge["source_id"] == new.node_id
    assert edge["target_id"] == old.node_id
    assert edge["source_kind"] == "decision"
    assert edge["target_kind"] == "decision"
    assert edge["edge_type"] == "supersedes"
    assert edge["created_at"]  # application-supplied ISO stamp (MI-10), not NULL
    # The superseded node leaves the active view; the superseding node stays active.
    assert _is_active(temp_store, old.node_id) is False
    assert _is_active(temp_store, new.node_id) is True


def test_5c_commit_enqueues_one_outbox_row(temp_store: GraphStore) -> None:
    """FORCING-FUNCTION INVERSION — Phase 5c wires the Outbox enqueue.

    Through 5a the ``_enqueue_outbox`` seam was a no-op and a commit enqueued ZERO
    ``pending_embeddings`` rows (``test_5a_enqueues_no_outbox_yet``). 5c fills the
    seam: every commit now UPSERTs exactly one row keyed on the committing node's
    id (``queued_at`` stamped, ``retry_count`` 0). This is the conscious inversion
    of the 5a forcing function.
    """
    delta = temp_store.commit_parsed_entry(_decision(slug="d", axiom="A."))
    rows = _pending(temp_store)
    assert len(rows) == 1
    row = rows[0]
    assert row["node_id"] == delta.node_id
    assert row["queued_at"]  # application-supplied ISO stamp (MI-10), not NULL
    assert row["retry_count"] == 0
    # The Outbox stamp shares the commit's single ``now`` (MI-10): on a new commit
    # ``queued_at`` equals the node's ``created_at``.
    assert row["queued_at"] == _node_row(temp_store, delta.node_id)["created_at"]


def test_5c_open_question_enqueues_identically(temp_store: GraphStore) -> None:
    """An open_question commit enqueues one Outbox row too — the enqueue is kind-agnostic."""
    delta = temp_store.commit_parsed_entry(_open_question(slug="oq", topic="T"))
    rows = _pending(temp_store)
    assert len(rows) == 1
    assert rows[0]["node_id"] == delta.node_id
    assert rows[0]["retry_count"] == 0


def test_5c_enqueue_unconditional_on_commentary_only_recommit(
    temp_store: GraphStore,
) -> None:
    """A commentary-only re-commit (slug rename → same canonical core → same id) re-enqueues
    and RESETS the row's drain state — slug is not identity (M2)."""
    d1 = temp_store.commit_parsed_entry(_decision(slug="d-one", axiom="A."))
    # Advance the drain state via raw SQL so the re-stamp's reset is observable.
    conn = temp_store._get_connection()
    try:
        with conn:
            conn.execute(
                "UPDATE pending_embeddings SET queued_at = ?, retry_count = ? "
                "WHERE node_id = ?",
                ("2000-01-01T00:00:00+00:00", 5, d1.node_id),
            )
    finally:
        conn.close()

    d2 = temp_store.commit_parsed_entry(_decision(slug="d-two", axiom="A."))
    assert d2.node_id == d1.node_id  # slug is not part of identity (M2)
    assert d2.commentary_fields_changed is True  # a rename is a commentary change
    rows = _pending(temp_store)
    assert len(rows) == 1  # still exactly one row (UPSERT on the PK, not a dup)
    assert rows[0]["retry_count"] == 0  # drain state reset
    assert rows[0]["queued_at"] != "2000-01-01T00:00:00+00:00"  # re-stamped


def test_5c_byte_identical_recommit_resets_drain_state_node_noop(
    temp_store: GraphStore,
) -> None:
    """The load-bearing P5 self-healing case: a byte-identical re-commit re-stamps the
    Outbox row (resets a dead-lettered ``retry_count``) EVEN THOUGH the node row is a
    true no-op (MI-3).

    MI-3's "true no-op" governs the NODE (no tick, no new node); MI-12's reset clause
    governs the OUTBOX (a deliberate retry must revive a node V3b dead-lettered on a
    transient outage). The two invariants coexist on purpose — this is the proof.
    """
    e = _decision(slug="d", axiom="A.", mechanisms=["m"], scope=["s"], transcript="T")
    d1 = temp_store.commit_parsed_entry(e)
    node_before = _node_row(temp_store, d1.node_id)
    # Simulate a V3b dead-letter: a transient provider outage drove retry_count up.
    conn = temp_store._get_connection()
    try:
        with conn:
            conn.execute(
                "UPDATE pending_embeddings SET retry_count = 3 WHERE node_id = ?",
                (d1.node_id,),
            )
    finally:
        conn.close()

    # Re-commit BYTE-IDENTICALLY.
    e2 = _decision(slug="d", axiom="A.", mechanisms=["m"], scope=["s"], transcript="T")
    d2 = temp_store.commit_parsed_entry(e2)

    # The NODE is a true no-op (MI-3): same id, no commentary change, one row, no tick.
    assert d2.node_id == d1.node_id
    assert d2.commentary_fields_changed is False
    assert _count(temp_store, "nodes") == 1
    assert _node_row(temp_store, d1.node_id)["updated_at"] == node_before["updated_at"]

    # The OUTBOX row IS revived (MI-12): retry_count reset to 0 (a bare no-op would
    # have left it at 3) — the dead-letter gets a fresh drain attempt.
    rows = _pending(temp_store)
    assert len(rows) == 1
    assert rows[0]["retry_count"] == 0


def test_5c_idempotent_upsert_keeps_single_row(temp_store: GraphStore) -> None:
    """Two commits of the same node keep EXACTLY one Outbox row (UPSERT on the PK).

    The F2 idempotent-enqueue substrate assertion the index routes to 5c.
    """
    d1 = temp_store.commit_parsed_entry(_decision(slug="d", axiom="A."))
    d2 = temp_store.commit_parsed_entry(_decision(slug="d", axiom="A."))
    assert d2.node_id == d1.node_id
    rows = _pending(temp_store)
    assert len(rows) == 1
    assert rows[0]["node_id"] == d1.node_id


def test_5c_rollback_enqueues_nothing(temp_store: GraphStore) -> None:
    """A store-stage failure (missing_target) rolls back the speculative Outbox row too.

    The enqueue at the seam writes the row BEFORE the slug-collision gate; a
    ``CommitError`` raised later rolls the whole transaction back, including the
    speculative ``pending_embeddings`` row (V1-D10 / MI-12 — commits with the node
    or not at all).
    """
    e = _decision(slug="new", axiom="New.")
    e.supersedes = ["nonexistent"]  # missing_target — fails store-stage, rolls back
    with pytest.raises(CommitError):
        temp_store.commit_parsed_entry(e)
    assert _count(temp_store, "pending_embeddings") == 0  # speculative row rolled back
    assert _count(temp_store, "nodes") == 0


# --- First-order CommitDelta cascade (5c finalizes cascade_affected_scopes) -----


def test_5c_cascade_committing_node_only(temp_store: GraphStore) -> None:
    """A decision in scopes {x, y} with no edges → cascade is its own scopes (5a behavior preserved)."""
    delta = temp_store.commit_parsed_entry(
        _decision(slug="d", axiom="A.", scope=["x", "y"])
    )
    assert delta.cascade_affected_scopes == ["x", "y"]


def test_5c_cascade_includes_kill_edge_target_scope(temp_store: GraphStore) -> None:
    """B supersedes A: B's delta names BOTH its own scope and A's now-deactivated scope (C3).

    When B supersedes A, A leaves the active view, so every scope A was in needs a
    re-render — the first-order render-targeting signal V3b/V4 consume.
    """
    temp_store.commit_parsed_entry(_decision(slug="a", axiom="Old.", scope=["x"]))
    e = _decision(slug="b", axiom="New.", scope=["y"])
    e.supersedes = ["a"]
    delta = temp_store.commit_parsed_entry(e)
    assert delta.cascade_affected_scopes == ["x", "y"]


def test_5c_cascade_on_edge_only_change(temp_store: GraphStore) -> None:
    """Re-committing a node to ADD a Supersedes: line (no commentary/scope change) surfaces
    the target's scope via the ``edges_changed`` gate, even though the node's own footprint
    is unchanged (so the node's own scope is NOT re-added — only A's view membership flipped)."""
    temp_store.commit_parsed_entry(_decision(slug="a", axiom="Old.", scope=["x"]))
    b1 = temp_store.commit_parsed_entry(_decision(slug="b", axiom="New.", scope=["y"]))
    # Re-commit B identically EXCEPT for the added Supersedes: line.
    e = _decision(slug="b", axiom="New.", scope=["y"])
    e.supersedes = ["a"]
    delta = temp_store.commit_parsed_entry(e)
    assert delta.node_id == b1.node_id
    assert delta.commentary_fields_changed is False  # footprint unchanged; only edges
    assert delta.cascade_affected_scopes == ["x"]  # the target's scope, not B's own


def test_5c_cascade_empty_on_byte_identical_recommit(temp_store: GraphStore) -> None:
    """A byte-identical re-commit produces an empty cascade — even though the Outbox is re-stamped."""
    temp_store.commit_parsed_entry(_decision(slug="d", axiom="A.", scope=["x"]))
    delta = temp_store.commit_parsed_entry(_decision(slug="d", axiom="A.", scope=["x"]))
    assert delta.cascade_affected_scopes == []
    assert _count(temp_store, "pending_embeddings") == 1  # but the Outbox IS re-enqueued


def test_5c_commit_delta_struct_shape_preserved(temp_store: GraphStore) -> None:
    """The CommitDelta still carries all five fields with the correct types (V1b extends, never rebuilds)."""
    delta = temp_store.commit_parsed_entry(
        _decision(slug="d", axiom="A.", scope=["x"])
    )
    assert isinstance(delta.node_id, str)
    assert isinstance(delta.node_scope, list)
    assert isinstance(delta.self_old_scope, list)
    assert isinstance(delta.commentary_fields_changed, bool)
    assert isinstance(delta.cascade_affected_scopes, list)
    # to_dict() carries exactly the five fields, JSON-safe (no tuples).
    d = delta.to_dict()
    assert set(d.keys()) == {
        "node_id",
        "node_scope",
        "self_old_scope",
        "commentary_fields_changed",
        "cascade_affected_scopes",
    }


def test_atomic_commit_rolls_back_on_midcommit_failure(
    temp_store: GraphStore, monkeypatch
) -> None:
    """An induced failure mid-commit leaves ZERO partial rows across all tables (V1-D10)."""
    e = _decision(slug="atomic", axiom="ATOM.", scope=["x"], transcript="t")

    def boom(cursor, node_id, parsed, now):
        raise sqlite3.IntegrityError("forced mid-commit failure")

    monkeypatch.setattr(temp_store, "_reconcile_edges", boom)

    with pytest.raises(DatabaseError):
        temp_store.commit_parsed_entry(e)

    assert _count(temp_store, "nodes") == 0
    assert _count(temp_store, "node_scopes") == 0
    assert _count(temp_store, "transcripts") == 0


# --- Deferred-flip tripwires (consciously INVERTED in Phase 5a) ----------------


def test_v1_schema_is_live_migration_step_after_phase_5a(temp_store: GraphStore) -> None:
    """DEFERRED-FLIP TRIPWIRE INVERTED — Phase 5a registered _v1_schema as live step 1.

    Through 2b–4b this asserted the schema was authored-but-not-live (the boot stayed
    on the prototype ``_init_db`` at ``user_version == 0``, the step absent from the
    registry). Phase 5a flipped it (entry-001): step 1 is registered and a fresh boot
    ladders the V1a STRICT schema in. The boot now lands at the live ladder head (read
    programmatically — V1b's step 2 ladders past 1; the STRICT ``nodes`` table proves
    the V1a schema is present regardless of how far the head has advanced).
    """
    from mitos.migrations import MIGRATION_STEPS, _pending_head, _v1_schema

    assert (1, _v1_schema) in MIGRATION_STEPS
    conn = temp_store._get_connection()
    try:
        assert conn.execute("PRAGMA user_version;").fetchone()[0] == _pending_head(
            MIGRATION_STEPS
        )
        assert bool(
            conn.execute(
                "SELECT strict FROM pragma_table_list WHERE name='nodes';"
            ).fetchone()[0]
        )
    finally:
        conn.close()


def test_identity_hash_is_slug_free_after_phase_5a(temp_store: GraphStore) -> None:
    """DEFERRED-FLIP TRIPWIRE INVERTED — Phase 5a points the commit at compute_node_id.

    Through 3a–4b this asserted the commit path still minted via the slug-INCLUSIVE
    prototype ``compute_hash``. Phase 5a flipped it (entry-002): the live mint is the
    slug-free ``compute_node_id``, so two same-core/different-slug entries converge to
    ONE node and a rename is a V1-D16 in-place commentary UPDATE. (Phase 8a retires
    ``compute_hash`` itself + reconciles its remaining importers — its 8a tail.)
    """
    AXIOM = "Use SQLite for the graph store"
    MECHS = ["sqlite", "wal"]

    e1 = _decision(slug="use-sqlite", axiom=AXIOM, rejected="Postgres.", mechanisms=list(MECHS))
    d1 = temp_store.commit_parsed_entry(e1)

    slug_free_id = compute_node_id(kind="decision", axiom=AXIOM, mechanism_refs=MECHS)
    assert d1.node_id == slug_free_id  # minted via the slug-free hash
    assert d1.node_id != compute_hash("decision", "use-sqlite", AXIOM, MECHS)

    # Same canonical core, different slug -> the SAME node id (converged)
    e2 = _decision(
        slug="use-sqlite-renamed", axiom=AXIOM, rejected="Postgres.", mechanisms=list(MECHS)
    )
    d2 = temp_store.commit_parsed_entry(e2)
    assert d2.node_id == d1.node_id
    assert _count(temp_store, "nodes") == 1
    assert _node_row(temp_store, d1.node_id)["slug"] == "use-sqlite-renamed"


# ===========================================================================
# Phase 2a — connection hardening (PRAGMA suite, version guard, ladder boot)
#
# Schema-agnostic; these stay green across the 5a flip. The empty-ladder-boot
# test is updated to assert the live ladder head (read via `_pending_head`, not a
# literal — it advances as later visions append rungs, e.g. V1b's step 2) now that
# the registry is populated. The WAL concurrency test's raw INSERT/UPDATE moved to
# V1a `nodes` columns.
# ===========================================================================


def test_wal_concurrency_multi_reader(temp_store: GraphStore) -> None:
    """Verifies SQLite WAL concurrency permits multiple parallel readers and a writer."""
    now = "2026-06-18T00:00:00.000000+00:00"
    # 1. Open main connection and write an initial V1a node row
    conn_writer = temp_store._get_connection()
    cursor = conn_writer.cursor()
    cursor.execute(
        "INSERT INTO nodes "
        "(id, kind, slug, slug_casefold, source, axiom, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("test-id", "decision", "test-slug", "test-slug", "user", "My core axiom", now, now),
    )
    conn_writer.commit()

    # 2. Start a transaction on the writer but do not commit it yet
    conn_writer.execute("BEGIN IMMEDIATE TRANSACTION;")
    conn_writer.execute("UPDATE nodes SET axiom = 'Axiom Modified' WHERE id = 'test-id'")

    # 3. A separate reader sees the pre-write snapshot (WAL snapshot isolation)
    conn_reader = temp_store._get_connection()
    cursor_reader = conn_reader.cursor()
    cursor_reader.execute("SELECT axiom FROM nodes WHERE id = 'test-id'")
    assert cursor_reader.fetchone()["axiom"] == "My core axiom"

    # 4. Commit the write
    conn_writer.commit()
    conn_writer.close()

    # 5. The reader now sees the modified state on a fresh query
    cursor_reader.execute("SELECT axiom FROM nodes WHERE id = 'test-id'")
    assert cursor_reader.fetchone()["axiom"] == "Axiom Modified"
    conn_reader.close()


def test_write_connection_issues_full_pragma_suite(temp_store: GraphStore) -> None:
    """A write/file connection reports the full §5.2.8 PRAGMA suite (MI-8)."""
    conn = temp_store._get_connection()
    try:
        assert conn.execute("PRAGMA foreign_keys;").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout;").fetchone()[0] == 5000
        assert conn.execute("PRAGMA journal_mode;").fetchone()[0].lower() == "wal"
        # synchronous: 0=OFF, 1=NORMAL, 2=FULL — NORMAL is the V1-D12 posture.
        assert conn.execute("PRAGMA synchronous;").fetchone()[0] == 1
    finally:
        conn.close()


def test_read_only_connection_keeps_fk_and_busy_timeout(temp_store: GraphStore) -> None:
    """A read-only connection still issues FK + busy_timeout, skips WAL/synchronous cleanly."""
    ro_store = GraphStore(temp_store.db_path, read_only=True)
    conn = ro_store._get_connection()
    try:
        assert conn.execute("PRAGMA foreign_keys;").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout;").fetchone()[0] == 5000
    finally:
        conn.close()


def test_foreign_keys_enforced_at_connection_level(temp_store: GraphStore) -> None:
    """The FK PRAGMA actually takes effect: an orphan child insert is rejected."""
    conn = temp_store._get_connection()
    try:
        conn.execute("CREATE TABLE _fk_parent (id INTEGER PRIMARY KEY);")
        conn.execute(
            "CREATE TABLE _fk_child ("
            "id INTEGER PRIMARY KEY, "
            "parent_id INTEGER REFERENCES _fk_parent(id));"
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO _fk_child (id, parent_id) VALUES (1, 999);")
            conn.commit()
    finally:
        conn.close()


def test_sqlite_version_guard_rejects_old_runtime(
    temp_store: GraphStore, monkeypatch
) -> None:
    """A sub-3.37 linked SQLite makes the connection helper fail fast with guidance."""
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 36, 0))
    with pytest.raises(DatabaseError) as exc_info:
        temp_store._get_connection()
    message = str(exc_info.value)
    assert "3.37" in message  # names the required floor
    assert "3.36" in message  # names the detected version


def test_boot_ladders_fresh_store_to_head(temp_store: GraphStore) -> None:
    """5a's boot ladders a fresh store up to the live ladder head (not the empty ``0``).

    Inverts the 2a-era ``test_boot_through_empty_ladder_leaves_user_version_zero``:
    with the registry populated, a fresh boot lands at the live head over the STRICT
    schema rather than the empty-ladder ``0``. The head is read programmatically
    (``_pending_head``) — it advances as later visions append rungs (V1b's step 2),
    so this never re-pins to a stale literal.
    """
    from mitos.migrations import MIGRATION_STEPS, _pending_head

    conn = temp_store._get_connection()
    try:
        assert conn.execute("PRAGMA user_version;").fetchone()[0] == _pending_head(
            MIGRATION_STEPS
        )
    finally:
        conn.close()


def test_init_boot_guard_refuses_prototype_graph() -> None:
    """The RW ``__init__`` boot guard REFUSES a prototype graph, routing to cutover (§10.1).

    Phase 5a (entry-001) wires ``is_pre_v1a_schema`` into ``GraphStore.__init__``:
    opening a real pre-V1a (prototype ``_init_db``) graph must **raise** + name the
    one-time cutover, never silently ladder-advance it into an undiagnosable hybrid
    (R3/R11). The predicate itself is unit-tested in ``test_migrations.py``; this
    proves the boot actually *invokes* the guard and surfaces the remedy — a
    regression that dropped the guard would pass every predicate test but fail here.
    """
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        # Build the real prototype (pre-V1a) schema in place, bypassing the
        # V1a-booting __init__ (the §16 retained-``_init_db`` fixture pattern).
        proto = GraphStore.__new__(GraphStore)
        proto.db_path = path
        proto.read_only = False
        proto._init_db()

        # A normal RW construction over that prototype graph must refuse + route.
        with pytest.raises(DatabaseError) as exc:
            GraphStore(path)
        assert "cutover" in str(exc.value).lower()  # names the route-to-cutover remedy

        # The refused boot did NOT ladder-advance the prototype (no R3/R11 hybrid):
        # ``user_version`` stays 0, the prototype schema is untouched on disk.
        conn = proto._get_connection()
        try:
            assert conn.execute("PRAGMA user_version;").fetchone()[0] == 0
        finally:
            conn.close()
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_store_rebuild_quarantine_is_tracked() -> None:
    """FORCING FUNCTION — the Phase 5a contained-red quarantine is a conscious, shrinking set.

    The quarantine list (``tests/conftest.py``, Decision 5) must stay an explicit,
    tracked set that provably empties by Phase 8a. Pinning it here makes any change a
    conscious edit (and any restoring phase must update this set), never a silent
    drift. The set was derived empirically from the flip (not the pre-existing
    ``*_live.py`` 429 flakes, which are not quarantined).

    Phase 5d removed the 2 genuinely store-only modules it restored
    (``test_renderer`` + ``test_adversarial_rendering``) and re-bucketed the other
    6 of 5a's "restored in 5d" labels (WIRING_LEDGER entry-003, §16):
    ``test_status_readiness`` → 6b (gated on the ``cmd_status`` rebuild) and the 5
    consumer-entangled modules → 8a. Phase 6b then restored ``test_status_readiness``
    (the ``cmd_status`` rebuild landed), leaving 12 — all 8a's.
    """
    from conftest import STORE_REBUILD_QUARANTINE

    # Phase 8a drained the contained-red window to EMPTY: all 12 consumer modules
    # were restored against the reconciled V1a consumers (entry-003 closed). The set
    # provably reaching 0 is the closing of the 5a→8a window.
    assert set(STORE_REBUILD_QUARANTINE) == set()


# ===========================================================================
# Phase 5b — edge reconciliation, referential integrity & slug-collision
#
# The store becomes the authority on the graph's referential truth: the two
# kill-edges commit by declarative mirror, the slug-collision assertion enforces
# one active node per casefold(slug), and the five source="store" codes fire.
# All assertions are raw-SQL on edges/nodes (read methods quarantined until 5d).
# ===========================================================================


def _commit_kill(store, slug, axiom, edge_type, target, **kw):
    """Builds a decision declaring one kill-edge (bare-slug, agentic shape) + commits it."""
    e = _decision(slug=slug, axiom=axiom, **kw)
    setattr(e, edge_type, [target])  # List[str] shape (V1b multi-valued)
    return store.commit_parsed_entry(e)


# --- Kill-edge commit (both types, both kinds) ---------------------------------


def test_corrects_commits_one_kill_edge(temp_store: GraphStore) -> None:
    """A declared ``Corrects:`` commits one edge of edge_type 'corrects'."""
    buggy = temp_store.commit_parsed_entry(_decision(slug="buggy", axiom="Buggy."))
    fix = _commit_kill(temp_store, "fixed", "Fixed.", "corrects", "buggy")

    rows = _edges(temp_store)
    assert len(rows) == 1
    assert rows[0]["edge_type"] == "corrects"
    assert rows[0]["source_id"] == fix.node_id
    assert rows[0]["target_id"] == buggy.node_id
    assert _is_active(temp_store, buggy.node_id) is False


def test_open_question_supersedes_open_question_same_kind(temp_store: GraphStore) -> None:
    """An OQ→OQ kill-edge commits (the CHECK permits same-kind, forbids cross-kind)."""
    oq1 = temp_store.commit_parsed_entry(_open_question(slug="oq1", topic="T1"))
    e = _open_question(slug="oq2", topic="T2")
    e.supersedes = ["oq1"]
    oq2 = temp_store.commit_parsed_entry(e)

    rows = _edges(temp_store)
    assert len(rows) == 1
    assert rows[0]["source_kind"] == "open_question"
    assert rows[0]["target_kind"] == "open_question"
    assert rows[0]["source_id"] == oq2.node_id
    assert rows[0]["target_id"] == oq1.node_id
    assert _is_active(temp_store, oq1.node_id) is False


def test_bracket_and_bare_citation_both_resolve(temp_store: GraphStore) -> None:
    """A bracketed citation (corpus shape) and a bare slug (agentic shape) both resolve."""
    old = temp_store.commit_parsed_entry(_decision(slug="old-choice", axiom="Old."))
    # Corpus/cutover shape: the parser stores the value raw, brackets included.
    text = (
        "### new-choice\n"
        "**Decided:** A new axiom.\n"
        "**Rejected:** An alternative.\n"
        "**Supersedes:** [old-choice]\n"
    )
    entries = parse_entry_stream(text, "decision")
    assert entries[0].supersedes == ["[old-choice]"]  # brackets retained by the parser
    new = temp_store.commit_parsed_entry(entries[0])

    rows = _edges(temp_store)
    assert len(rows) == 1
    assert rows[0]["source_id"] == new.node_id
    assert rows[0]["target_id"] == old.node_id


# --- Declarative mirror (V1-D21 / §4.5.1) --------------------------------------


def test_declarative_mirror_drop_resurrects_then_readd(temp_store: GraphStore) -> None:
    """Dropping a Supersedes line DELETEs the edge (target resurrects); re-adding re-inserts."""
    old = temp_store.commit_parsed_entry(_decision(slug="old", axiom="Old."))
    new = _commit_kill(temp_store, "new", "New.", "supersedes", "old")
    assert _count(temp_store, "edges") == 1
    assert _is_active(temp_store, old.node_id) is False

    # Re-commit 'new' WITHOUT the Supersedes line -> edge DELETEd, 'old' resurrected.
    temp_store.commit_parsed_entry(_decision(slug="new", axiom="New."))
    assert _count(temp_store, "edges") == 0
    assert _is_active(temp_store, old.node_id) is True

    # Re-add the Supersedes line -> edge re-inserted.
    _commit_kill(temp_store, "new", "New.", "supersedes", "old")
    assert _count(temp_store, "edges") == 1
    assert _is_active(temp_store, old.node_id) is False


def test_idempotent_redeclaration_no_churn(temp_store: GraphStore) -> None:
    """Re-declaring an unchanged kill-edge is a no-op — no duplicate, no updated_at tick (MI-5)."""
    temp_store.commit_parsed_entry(_decision(slug="old", axiom="Old."))
    new = _commit_kill(temp_store, "new", "New.", "supersedes", "old")
    before = _node_row(temp_store, new.node_id)["updated_at"]

    # Byte-identical re-commit (same commentary, same declared edge).
    again = _commit_kill(temp_store, "new", "New.", "supersedes", "old")
    assert again.node_id == new.node_id
    assert _count(temp_store, "edges") == 1  # no duplicate
    assert _node_row(temp_store, new.node_id)["updated_at"] == before  # no tick (MI-3)


# --- Net-new-only / the self-strangling trap (Decision 1) ----------------------


def test_recommit_superseding_node_with_changed_commentary_keeps_edge(
    temp_store: GraphStore,
) -> None:
    """The self-strangle fix: re-committing a superseding node retains its kill-edge.

    Re-resolving a RETAINED ``Supersedes:`` against the active view would fail (the
    target is inactive *because of that edge*) and roll back the commentary edit.
    Net-new-only re-resolution keeps the edge intact and the target inactive.
    """
    old = temp_store.commit_parsed_entry(_decision(slug="old", axiom="Old."))
    new = _commit_kill(temp_store, "new", "New.", "supersedes", "old", context="v1")
    assert _is_active(temp_store, old.node_id) is False

    # Re-commit 'new' with CHANGED commentary, STILL declaring Supersedes: old.
    again = _commit_kill(
        temp_store, "new", "New.", "supersedes", "old", context="v2"
    )
    assert again.node_id == new.node_id
    assert _count(temp_store, "edges") == 1  # edge intact, never re-resolved
    assert _is_active(temp_store, old.node_id) is False  # target still inactive
    assert _node_row(temp_store, new.node_id)["context"] == "v2"  # commentary edited


# --- The four V1-D4 collision cases (Decision 4) -------------------------------


def test_fm1_same_slug_supersession_commits(temp_store: GraphStore) -> None:
    """FM1 — a same-slug supersession commits: the predecessor goes inactive."""
    a = temp_store.commit_parsed_entry(_decision(slug="x", axiom="Axiom A."))
    b = _commit_kill(temp_store, "x", "Axiom B.", "supersedes", "x")

    assert b.node_id != a.node_id
    assert _count(temp_store, "nodes") == 2
    assert _count(temp_store, "edges") == 1
    assert _is_active(temp_store, a.node_id) is False  # predecessor inactive
    assert _is_active(temp_store, b.node_id) is True  # one live answer for 'x'


def test_fm2_removing_supersedes_rolls_back_on_collision(temp_store: GraphStore) -> None:
    """FM2 — removing the Supersedes line resurrects the predecessor → slug_collision rollback."""
    a = temp_store.commit_parsed_entry(_decision(slug="x", axiom="Axiom A."))
    b = _commit_kill(temp_store, "x", "Axiom B.", "supersedes", "x")

    # Re-commit B without the Supersedes line: resurrecting A recreates the collision.
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(_decision(slug="x", axiom="Axiom B."))
    codes = [item.code for item in exc.value.failure.items]
    assert codes == ["slug_collision"]
    # Rolled back: the edge survives, A stays inactive (no resurrection on disk).
    assert _count(temp_store, "edges") == 1
    assert _is_active(temp_store, a.node_id) is False


def test_cross_slug_resurrection_collision_rolls_back(temp_store: GraphStore) -> None:
    """Dropping a Supersedes line must not resurrect a predecessor onto a slug a
    *different* active entry has since taken (the cross-slug resurrection collision).

    Distinct from FM2 (same-slug): here the superseding node 'b' has its OWN slug, a
    third entry independently reused the predecessor's freed slug 'c' while it was
    inactive, and dropping b's Supersedes line would reactivate 'c' — leaving TWO
    active entries under 'c'. A committing-slug-only assertion (it would only check
    'b') misses it; the resurrected-slug check catches it and rolls back so MI-13
    stays a hard global invariant (5d's get_node_by_slug depends on ≤1 active/slug).
    """
    c = temp_store.commit_parsed_entry(_decision(slug="c", axiom="Axiom C."))
    _commit_kill(temp_store, "b", "Axiom B.", "supersedes", "c")  # c -> inactive
    d = temp_store.commit_parsed_entry(_decision(slug="c", axiom="Axiom D."))  # reuse 'c'
    assert _is_active(temp_store, c.node_id) is False
    assert _is_active(temp_store, d.node_id) is True

    # Re-commit 'b' WITHOUT the Supersedes line: resurrecting 'c' collides with 'd'.
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(_decision(slug="b", axiom="Axiom B."))
    assert [i.code for i in exc.value.failure.items] == ["slug_collision"]

    # Rolled back: the b→c edge survives, 'c' stays inactive, exactly one active 'c'.
    assert _count(temp_store, "edges") == 1
    assert _is_active(temp_store, c.node_id) is False
    assert _is_active(temp_store, d.node_id) is True
    assert [n for n in (c.node_id, d.node_id) if _is_active(temp_store, n)] == [
        d.node_id
    ]


def test_independent_collision_rolls_back(temp_store: GraphStore) -> None:
    """An independent same-slug entry (no supersession) rolls back slug_collision."""
    temp_store.commit_parsed_entry(_decision(slug="dup", axiom="Axiom A."))
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(_decision(slug="dup", axiom="Axiom B."))
    assert [i.code for i in exc.value.failure.items] == ["slug_collision"]
    assert _count(temp_store, "nodes") == 1  # the second entry never committed


def test_rename_onto_active_slug_rolls_back(temp_store: GraphStore) -> None:
    """Renaming an entry onto an existing active slug rolls back slug_collision."""
    temp_store.commit_parsed_entry(_decision(slug="a", axiom="Axiom A."))
    b = temp_store.commit_parsed_entry(_decision(slug="b", axiom="Axiom B."))

    # Re-commit B (same core => same id) renamed to 'a' -> collides with the active A.
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(_decision(slug="a", axiom="Axiom B."))
    assert [i.code for i in exc.value.failure.items] == ["slug_collision"]
    assert _node_row(temp_store, b.node_id)["slug"] == "b"  # rename rolled back


def test_case_variant_zombie_rolls_back_under_casefold(temp_store: GraphStore) -> None:
    """A case-variant slug (use-x vs Use-X) collides under casefold and rolls back."""
    temp_store.commit_parsed_entry(_decision(slug="use-x", axiom="Axiom A."))
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(_decision(slug="Use-X", axiom="Axiom B."))
    assert [i.code for i in exc.value.failure.items] == ["slug_collision"]
    assert _count(temp_store, "nodes") == 1


# --- The five store-stage codes (each: right code + source + full rollback) ----


def test_missing_target_rolls_back(temp_store: GraphStore) -> None:
    """Citing a slug not in the graph fires missing_target and rolls the entry back."""
    e = _decision(slug="new", axiom="New.")
    e.supersedes = ["nonexistent"]
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(e)
    item = exc.value.failure.items[0]
    assert item.code == "missing_target"
    assert item.source == "store"
    assert exc.value.failure.slug == "new"
    assert _count(temp_store, "nodes") == 0  # whole entry rolled back
    assert _count(temp_store, "edges") == 0


def test_dangling_edge_reports_one_hop_killer(temp_store: GraphStore) -> None:
    """Citing an inactive target fires dangling_edge and names its immediate 1-hop killer."""
    temp_store.commit_parsed_entry(_decision(slug="a", axiom="Axiom A."))
    _commit_kill(temp_store, "b", "Axiom B.", "supersedes", "a")  # a is now inactive

    # C cites the now-inactive 'a' (it should have cited the active 'b').
    e = _decision(slug="c", axiom="Axiom C.")
    e.supersedes = ["a"]
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(e)
    item = exc.value.failure.items[0]
    assert item.code == "dangling_edge"
    assert "b" in item.message  # the 1-hop killer's slug is named
    assert _count(temp_store, "nodes") == 2  # C rolled back


def test_cycle_violation_self_edge(temp_store: GraphStore) -> None:
    """A node citing its own slug (resolving to self) fires cycle_violation."""
    e = _decision(slug="x", axiom="Self.")
    e.supersedes = ["x"]
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(e)
    assert exc.value.failure.items[0].code == "cycle_violation"
    assert _count(temp_store, "nodes") == 0
    assert _count(temp_store, "edges") == 0


def test_cycle_violation_inactive_source(temp_store: GraphStore) -> None:
    """A net-new kill-edge from an already-superseded source fires cycle_violation."""
    a = temp_store.commit_parsed_entry(_decision(slug="a", axiom="Axiom A."))
    temp_store.commit_parsed_entry(_decision(slug="c", axiom="Axiom C."))
    _commit_kill(temp_store, "b", "Axiom B.", "supersedes", "a")  # a is now inactive

    # Re-commit A (same core => same id, now inactive) declaring a NEW kill-edge.
    e = _decision(slug="a", axiom="Axiom A.")
    e.supersedes = ["c"]
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(e)
    assert exc.value.failure.items[0].code == "cycle_violation"
    # No A→C edge minted; only the original B→A edge remains.
    rows = _edges(temp_store)
    assert len(rows) == 1
    assert rows[0]["source_id"] != a.node_id  # the surviving edge is B→A, not A→C


def test_kind_constraint_violation_via_ddl_check(temp_store: GraphStore) -> None:
    """A cross-kind kill-edge fires kind_constraint_violation from the DDL CHECK (Lesson 2)."""
    temp_store.commit_parsed_entry(_open_question(slug="oq1", topic="A topic."))
    e = _decision(slug="d", axiom="A decision.")
    e.supersedes = ["oq1"]  # decision -> open_question is cross-kind
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(e)
    assert exc.value.failure.items[0].code == "kind_constraint_violation"
    assert _count(temp_store, "edges") == 0
    assert _count(temp_store, "nodes") == 1  # only the OQ; the decision rolled back


# --- The seven formerly-deferred edge types now COMMIT (V1b 2a flip) -----------


def test_formerly_deferred_edge_types_now_commit(
    temp_store: GraphStore, caplog
) -> None:
    """The non-kill types commit their edges (no warn, no defer) as of V1b 2a.

    Pre-flip these seven warn-deferred — logged a WARNING, committed no edge. The
    flip removes that tail: ``amends`` / ``cites`` now author a real edge, both
    endpoints stay active (non-kill), and no "deferred to V1b" notice is logged.
    """
    t1 = temp_store.commit_parsed_entry(_decision(slug="t1", axiom="Target one."))
    t2 = temp_store.commit_parsed_entry(_decision(slug="t2", axiom="Target two."))
    e = _decision(slug="d", axiom="A decision.")
    e.amends = ["t1"]  # formerly warn-deferred — now commits
    e.cites = ["t2"]  # formerly warn-deferred — now commits

    with caplog.at_level(logging.WARNING, logger="mitos.store"):
        delta = temp_store.commit_parsed_entry(e)

    # The node + both non-kill edges committed.
    assert _node_row(temp_store, delta.node_id) is not None
    rows = _edges(temp_store)
    assert {(r["edge_type"], r["target_id"]) for r in rows} == {
        ("amends", t1.node_id),
        ("cites", t2.node_id),
    }
    # Non-kill edges retire nothing — both targets stay active.
    assert _is_active(temp_store, t1.node_id) is True
    assert _is_active(temp_store, t2.node_id) is True
    # The warn-defer tail is gone — no "deferred to V1b" notice is logged.
    assert "deferred to V1b" not in caplog.text


# --- updated_at ticks on an edge-set change (V1-D17) ---------------------------


def test_updated_at_ticks_on_edge_only_change(temp_store: GraphStore) -> None:
    """An edge-only re-commit ticks updated_at; a byte-identical re-commit does not."""
    temp_store.commit_parsed_entry(_decision(slug="old", axiom="Old."))
    b = temp_store.commit_parsed_entry(_decision(slug="b", axiom="B axiom."))
    before = _node_row(temp_store, b.node_id)["updated_at"]

    # Re-commit B with the SAME commentary but a NEW outgoing kill-edge.
    again = _commit_kill(temp_store, "b", "B axiom.", "supersedes", "old")
    assert again.node_id == b.node_id
    after = _node_row(temp_store, b.node_id)["updated_at"]
    assert after > before  # edges_changed feeds the tick even with no commentary change


# --- Code-name pin (the cross-vision §5.2.2 contract) --------------------------


def test_store_failure_codes_pin() -> None:
    """STORE_FAILURE_CODES is EXACTLY the five reserved names (a typo is a silent break)."""
    from mitos.errors import STORE_FAILURE_CODES

    assert STORE_FAILURE_CODES == frozenset(
        {
            "slug_collision",
            "missing_target",
            "dangling_edge",
            "kind_constraint_violation",
            "cycle_violation",
        }
    )


# ===========================================================================
# Phase 5d — read views, retrieval surfaces & modifier stamping
#
# The read side completes the substrate: every read method is rebuilt over the
# V1a STRICT schema to (a) compute activeness via the kill-edge anti-join (the
# SAME definition ``_is_active`` encodes — M3, never the prototype
# ``compute_all_states``), (b) return the prototype reader-key dict so the
# unchanged consumers keep working, and (c) stamp reverse-relation modifiers
# through ONE bulk join. Plus the two new primitives: ``get_transcript`` and the
# C4 ``query_letter``. Fixtures commit via ``commit_parsed_entry`` (the store
# path), never ``record_decision_entry`` (the 8a consumer path).
# ===========================================================================


def _insert_drifted_signal(store: GraphStore, node_id: str) -> None:
    """Raw-SQL inserts a ``drifted`` signal (no V1a writer exists — reserved channel)."""
    conn = store._get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO signals (node_id, signal_type, created_at) "
                "VALUES (?, 'drifted', ?)",
                (node_id, "2026-06-18T00:00:00.000000+00:00"),
            )
    finally:
        conn.close()


# --- Active-view anti-join & reader-key aliasing (the core) --------------------


def test_5d_active_decisions_excludes_superseded(temp_store: GraphStore) -> None:
    """get_active_decisions returns only nodes with no incoming kill-edge."""
    old = temp_store.commit_parsed_entry(_decision(slug="old", axiom="Old."))
    _commit_kill(temp_store, "new", "New.", "supersedes", "old")

    slugs = [n["slug"] for n in temp_store.get_active_decisions()]
    assert slugs == ["new"]
    # The read's anti-join matches the inline ``_is_active`` helper definition (Lesson 14).
    assert _is_active(temp_store, old.node_id) is False


def test_5d_active_reads_never_call_get_node_state(
    temp_store: GraphStore, monkeypatch
) -> None:
    """The active read methods compute activeness via the anti-join, NOT per-node state.

    The prototype ``compute_all_states`` DAG was retired in Phase 8a; its single-node
    successor is ``get_node_state``. Monkeypatching it to raise proves the active read
    views still derive activeness from the inline SQL kill-edge anti-join (one
    definition of "active", Lesson 14) and never fall back to an N+1 per-node state
    call. A regression re-introducing a per-node filter fails loudly here.
    """
    old = temp_store.commit_parsed_entry(_decision(slug="old", axiom="Old.", scope=["z"]))
    _commit_kill(temp_store, "new", "New.", "supersedes", "old", scope=["z"])
    temp_store.commit_parsed_entry(_open_question(slug="oq", topic="T", scope=["z"]))

    def _boom(_node_id):
        raise AssertionError("get_node_state must not be called by the active reads")

    monkeypatch.setattr(temp_store, "get_node_state", _boom)

    assert [n["slug"] for n in temp_store.get_active_decisions()] == ["new"]
    assert [n["slug"] for n in temp_store.get_active_decisions(scope="z")] == ["new"]
    assert [n["slug"] for n in temp_store.get_open_questions()] == ["oq"]
    assert temp_store.get_node_by_slug("new")["slug"] == "new"
    assert [p["slug"] for p in temp_store.query_letter(scope="z")] == ["new"]
    assert {n["slug"] for n in temp_store.get_decisions(state="all")} == {"old", "new"}


def test_5d_reader_keys_aliased_no_v1a_leak(temp_store: GraphStore) -> None:
    """A decision read returns the prototype reader keys; V1a column names don't leak."""
    temp_store.commit_parsed_entry(
        _decision(slug="d", axiom="The axiom.", mechanisms=["beta", "alpha"], scope=["s"])
    )
    (node,) = temp_store.get_active_decisions()
    assert node["core_axiom"] == "The axiom."
    assert node["mechanisms"] == ["beta", "alpha"]  # list, decode order preserved
    assert node["scope"] == ["s"]
    assert node["rejected_paths"] == "An alternative."  # raw string
    assert node["is_drifted"] is False
    # The V1a column names must NOT appear as keys (the alias is total).
    for leaked in ("axiom", "mechanism_refs_json", "rejected_paths_json", "questions_raised_json"):
        assert leaked not in node
    # A decision carries no OQ keys.
    assert "topic" not in node and "questions_raised" not in node


def test_5d_rejected_paths_is_raw_string_not_json(temp_store: GraphStore) -> None:
    """rejected_paths round-trips the RAW string (5a §14) — never JSON-decoded."""
    temp_store.commit_parsed_entry(
        _decision(slug="d", axiom="A.", rejected='["not", "a", "list"]')
    )
    (node,) = temp_store.get_active_decisions()
    # If the read JSON-decoded it, this would be a list; it must stay the raw string.
    assert node["rejected_paths"] == '["not", "a", "list"]'
    assert isinstance(node["rejected_paths"], str)


def test_5d_drifted_node_included_and_annotated(temp_store: GraphStore) -> None:
    """A drifted decision stays in the active view with is_drifted True (annotate, not retire)."""
    d = temp_store.commit_parsed_entry(_decision(slug="d", axiom="A."))
    _insert_drifted_signal(temp_store, d.node_id)

    (node,) = temp_store.get_active_decisions()
    assert node["slug"] == "d"
    assert node["is_drifted"] is True  # forward-correct EXISTS derivation lights up


def test_5d_active_decisions_scope_filter(temp_store: GraphStore) -> None:
    """The scope filter narrows correctly; a multi-scope node appears under each scope."""
    temp_store.commit_parsed_entry(_decision(slug="a", axiom="A.", scope=["be"]))
    temp_store.commit_parsed_entry(_decision(slug="b", axiom="B.", scope=["fe", "be"]))

    assert {n["slug"] for n in temp_store.get_active_decisions(scope="be")} == {"a", "b"}
    assert {n["slug"] for n in temp_store.get_active_decisions(scope="fe")} == {"b"}
    assert temp_store.get_active_decisions(scope="nope") == []


def test_5d_get_node_aliases_any_state(temp_store: GraphStore) -> None:
    """get_node returns the reader shape for active AND inactive ids (not active-scoped)."""
    old = temp_store.commit_parsed_entry(_decision(slug="old", axiom="Old."))
    _commit_kill(temp_store, "new", "New.", "supersedes", "old")

    n = temp_store.get_node(old.node_id)
    assert n is not None and n["slug"] == "old" and n["core_axiom"] == "Old."
    assert temp_store.get_node("nonexistent-id") is None


def test_5d_get_node_by_slug_active_only(temp_store: GraphStore) -> None:
    """get_node_by_slug returns the single ACTIVE node; casefold matches; unknown → None."""
    temp_store.commit_parsed_entry(_decision(slug="be-choice", axiom="A."))
    assert temp_store.get_node_by_slug("be-choice")["slug"] == "be-choice"
    assert temp_store.get_node_by_slug("BE-CHOICE")["slug"] == "be-choice"  # str.casefold
    assert temp_store.get_node_by_slug("unknown") is None


def test_5d_get_node_by_slug_resolves_one_after_reslug(temp_store: GraphStore) -> None:
    """After a supersede + slug reuse, get_node_by_slug still resolves ≤1 active (MI-13)."""
    # Commit 'c'; supersede it with 'b'; reuse slug 'c' on a NEW independent decision.
    temp_store.commit_parsed_entry(_decision(slug="c", axiom="C-old."))
    _commit_kill(temp_store, "b", "B.", "supersedes", "c")
    temp_store.commit_parsed_entry(_decision(slug="c", axiom="C-new-independent."))

    node = temp_store.get_node_by_slug("c")
    assert node is not None
    assert node["core_axiom"] == "C-new-independent."  # the lone active 'c'


def test_5d_open_questions_reader_keys_and_anti_join(temp_store: GraphStore) -> None:
    """get_open_questions returns the OQ reader shape and applies the Stage-1 anti-join."""
    oq1 = temp_store.commit_parsed_entry(
        _open_question(slug="oq1", topic="Topic one", questions=["a?", "b?"], scope=["x"])
    )
    (node,) = temp_store.get_open_questions()
    assert node["topic"] == "Topic one"
    assert node["questions_raised"] == ["a?", "b?"]
    assert node["scope"] == ["x"]
    # OQ carries no decision keys.
    for leaked in ("core_axiom", "mechanisms", "rejected_paths"):
        assert leaked not in node

    # A corrects kill-edge on the OQ removes it from the active OQ view (V1-D18 Stage-1).
    e = _open_question(slug="oq2", topic="Topic two")
    e.corrects = ["oq1"]
    temp_store.commit_parsed_entry(e)
    assert [n["slug"] for n in temp_store.get_open_questions()] == ["oq2"]


def test_5d_get_decisions_computed_state(temp_store: GraphStore) -> None:
    """get_decisions attaches computed_state (active/superseded/corrected) and filters on it."""
    temp_store.commit_parsed_entry(_decision(slug="sup-old", axiom="O."))
    _commit_kill(temp_store, "sup-new", "N.", "supersedes", "sup-old")
    temp_store.commit_parsed_entry(_decision(slug="cor-old", axiom="CO."))
    _commit_kill(temp_store, "cor-new", "CN.", "corrects", "cor-old")

    states = {n["slug"]: n["computed_state"] for n in temp_store.get_decisions(state="all")}
    assert states == {
        "sup-old": "superseded",
        "sup-new": "active",
        "cor-old": "corrected",
        "cor-new": "active",
    }
    assert {n["slug"] for n in temp_store.get_decisions()} == {"sup-new", "cor-new"}  # live set
    assert [n["slug"] for n in temp_store.get_decisions(state="superseded")] == ["sup-old"]
    assert [n["slug"] for n in temp_store.get_decisions(state="corrected")] == ["cor-old"]


def test_5d_get_all_nodes_mixed_kinds_with_state(temp_store: GraphStore) -> None:
    """get_all_nodes returns both kinds (any state) with computed_state attached."""
    temp_store.commit_parsed_entry(_decision(slug="d", axiom="A."))
    temp_store.commit_parsed_entry(_open_question(slug="oq", topic="T"))
    got = {(n["slug"], n["kind"], n["computed_state"]) for n in temp_store.get_all_nodes()}
    assert got == {("d", "decision", "active"), ("oq", "open_question", "active")}


# --- Modifier engine (T12 — store-level) --------------------------------------


def test_5d_modifiers_map_superseded_by(temp_store: GraphStore) -> None:
    """get_modifiers_map populates superseded_by for an inactive (superseded) target."""
    old = temp_store.commit_parsed_entry(_decision(slug="old", axiom="O."))
    _commit_kill(temp_store, "new", "N.", "supersedes", "old")
    assert temp_store.get_modifiers_map([old.node_id]) == {
        old.node_id: {"superseded_by": ["new"]}
    }
    assert temp_store.get_modifiers(old.node_id) == {"superseded_by": ["new"]}


def test_5d_modifiers_map_corrected_by(temp_store: GraphStore) -> None:
    """get_modifiers_map populates corrected_by for an inactive (corrected) target."""
    buggy = temp_store.commit_parsed_entry(_decision(slug="buggy", axiom="B."))
    _commit_kill(temp_store, "fixed", "F.", "corrects", "buggy")
    assert temp_store.get_modifiers_map([buggy.node_id]) == {
        buggy.node_id: {"corrected_by": ["fixed"]}
    }


def test_5d_modifiers_map_unmodified_absent_and_empty_input(temp_store: GraphStore) -> None:
    """An unmodified node is absent from the map; empty input → {}."""
    d = temp_store.commit_parsed_entry(_decision(slug="d", axiom="A."))
    assert temp_store.get_modifiers_map([d.node_id]) == {}  # active, no incoming kill-edge
    assert temp_store.get_modifiers(d.node_id) == {}
    assert temp_store.get_modifiers_map([]) == {}


def test_5d_reserved_modifier_keys_never_populate_in_v1a(temp_store: GraphStore) -> None:
    """amended_by / narrowed_by are reserved-empty in V1a but stay in MODIFIER_EDGE_KEYS (the seam).

    Only supersedes/corrects edges can exist (the edges CHECK + parser warn-defer),
    so only superseded_by/corrected_by ever populate. The reserved keys remaining in
    MODIFIER_EDGE_KEYS is what makes V1b lighting up amends/narrows a one-line change
    with zero read-surface edits (the C4 FORWARD HAZARD seam).
    """
    from mitos.store import MODIFIER_EDGE_KEYS

    assert MODIFIER_EDGE_KEYS == {
        "supersedes": "superseded_by",
        "amends": "amended_by",
        "narrows": "narrowed_by",
        "corrects": "corrected_by",
    }
    old = temp_store.commit_parsed_entry(_decision(slug="old", axiom="O."))
    _commit_kill(temp_store, "new", "N.", "supersedes", "old")
    mods = temp_store.get_modifiers_map([old.node_id])[old.node_id]
    assert "amended_by" not in mods and "narrowed_by" not in mods


def test_5d_active_surfaces_modifier_empty_seam_ships(temp_store: GraphStore) -> None:
    """The active view is provably modifier-empty, but every surface still runs the stamp pass.

    An active node has no incoming kill-edge (that IS the anti-join), so active reads
    carry no modifier keys; an inactive node read via get_decisions(state='superseded')
    DOES carry its stamped superseded_by — proving the stamping machinery is wired on
    the surfaces that can return an inactive node (the seam ships; C4 FORWARD HAZARD).
    """
    old = temp_store.commit_parsed_entry(_decision(slug="old", axiom="O."))
    _commit_kill(temp_store, "new", "N.", "supersedes", "old")

    for node in temp_store.get_active_decisions():
        assert "superseded_by" not in node and "corrected_by" not in node
    (inactive,) = temp_store.get_decisions(state="superseded")
    assert inactive["superseded_by"] == ["new"]
    assert temp_store.get_node(old.node_id)["superseded_by"] == ["new"]


def test_5d_modifier_stamp_is_one_bulk_call_never_n_plus_1(
    temp_store: GraphStore, monkeypatch
) -> None:
    """A read over N nodes issues exactly ONE modifier query (never N+1, P11)."""
    for i in range(5):
        temp_store.commit_parsed_entry(_decision(slug=f"d{i}", axiom=f"A{i}.", scope=["s"]))

    calls = {"n": 0}
    original = temp_store._modifiers_map

    def _counting(conn, node_ids):
        calls["n"] += 1
        return original(conn, node_ids)

    monkeypatch.setattr(temp_store, "_modifiers_map", _counting)
    assert len(temp_store.get_active_decisions()) == 5
    assert calls["n"] == 1  # one bulk join for all five nodes


# --- C4 Letter query (T5 substrate) -------------------------------------------


def test_5d_query_letter_by_slug_shape(temp_store: GraphStore) -> None:
    """query_letter(slug=…) returns the Letter projection (axiom, NOT core_axiom)."""
    temp_store.commit_parsed_entry(
        _decision(slug="d", axiom="The axiom.", rejected="Why not.", scope=["s"])
    )
    (payload,) = temp_store.query_letter(slug="d")
    assert payload == {
        "slug": "d",
        "axiom": "The axiom.",  # the C4 projection name (not core_axiom)
        "scope": ["s"],
        "rejected_paths": "Why not.",
    }


def test_5d_query_letter_by_node_id_and_scope(temp_store: GraphStore) -> None:
    """query_letter filters by node_id (PK) and by scope+kind on the active view."""
    d = temp_store.commit_parsed_entry(_decision(slug="d", axiom="A.", scope=["x"]))
    temp_store.commit_parsed_entry(_decision(slug="e", axiom="E.", scope=["y"]))

    assert [p["slug"] for p in temp_store.query_letter(node_id=d.node_id)] == ["d"]
    assert [p["slug"] for p in temp_store.query_letter(scope="x", kind="decision")] == ["d"]
    assert temp_store.query_letter(scope="none-such") == []


def test_5d_query_letter_brief_drops_rejected_keeps_axiom_and_modifiers(
    temp_store: GraphStore,
) -> None:
    """brief=True drops rejected_paths but keeps axiom and any modifier keys."""
    temp_store.commit_parsed_entry(_decision(slug="d", axiom="A.", rejected="Heavy."))
    (full,) = temp_store.query_letter(slug="d")
    (brief,) = temp_store.query_letter(slug="d", brief=True)
    assert "rejected_paths" in full and full["rejected_paths"] == "Heavy."
    assert "rejected_paths" not in brief
    assert brief["axiom"] == "A." and brief["scope"] == []


def test_5d_query_letter_active_view_only(temp_store: GraphStore) -> None:
    """query_letter never returns an inactive (superseded) node."""
    temp_store.commit_parsed_entry(_decision(slug="old", axiom="O."))
    _commit_kill(temp_store, "new", "N.", "supersedes", "old")
    assert [p["slug"] for p in temp_store.query_letter()] == ["new"]
    assert temp_store.query_letter(slug="old") == []


def test_5d_query_letter_has_no_semantic_path() -> None:
    """V1a has no vector/embedding query — the word 'semantic' must not appear in query_letter."""
    import inspect

    src = inspect.getsource(GraphStore.query_letter)
    assert "semantic" not in src.lower()


# --- get_transcript (new primitive) -------------------------------------------


def test_5d_get_transcript_returns_self_or_none(temp_store: GraphStore) -> None:
    """get_transcript returns the node's own committed transcript text, or None."""
    with_tx = temp_store.commit_parsed_entry(
        _decision(slug="d", axiom="A.", transcript="User: why?\nLLM: because.")
    )
    without_tx = temp_store.commit_parsed_entry(_decision(slug="e", axiom="E."))
    assert temp_store.get_transcript(with_tx.node_id) == "User: why?\nLLM: because."
    assert temp_store.get_transcript(without_tx.node_id) is None
    assert temp_store.get_transcript("nonexistent-id") is None


def test_5d_get_transcript_supersession_does_not_borrow(temp_store: GraphStore) -> None:
    """A superseding node's transcript is its OWN — the predecessor's is not borrowed.

    The corrects-only transitive walk is V1b/V5 (pinned in the docstring); for the
    V1a direct read the rule holds trivially: get_transcript(new) returns only new's
    own transcript (None here), never the superseded predecessor's.
    """
    old = temp_store.commit_parsed_entry(
        _decision(slug="old", axiom="O.", transcript="OLD transcript.")
    )
    new = _decision(slug="new", axiom="N.")
    new.supersedes = ["old"]
    new_delta = temp_store.commit_parsed_entry(new)
    assert temp_store.get_transcript(new_delta.node_id) is None  # not borrowed from old
    assert temp_store.get_transcript(old.node_id) == "OLD transcript."
