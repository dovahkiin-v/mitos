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
    e.supersedes = "old-choice"
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


def test_5a_enqueues_no_outbox_yet(temp_store: GraphStore) -> None:
    """FORCING FUNCTION — 5a enqueues NOTHING into pending_embeddings.

    The ``_enqueue_outbox`` seam is a no-op in 5a. This flips RED when Phase 5c
    wires the Outbox enqueue.
    """
    temp_store.commit_parsed_entry(_decision(slug="d", axiom="A."))
    assert _count(temp_store, "pending_embeddings") == 0


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
    registry). Phase 5a flipped it (entry-001): the step is registered and a fresh
    boot lands ``user_version == 1`` over the V1a STRICT schema.
    """
    from mitos.migrations import MIGRATION_STEPS, _v1_schema

    assert (1, _v1_schema) in MIGRATION_STEPS
    conn = temp_store._get_connection()
    try:
        assert conn.execute("PRAGMA user_version;").fetchone()[0] == 1
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
# test is updated to assert the V1a head (user_version == 1) now that step 1 is
# live. The WAL concurrency test's raw INSERT/UPDATE moved to V1a `nodes` columns.
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


def test_boot_lands_v1a_schema_at_user_version_one(temp_store: GraphStore) -> None:
    """5a's boot ladders a fresh store to the V1a schema head (user_version == 1).

    Inverts the 2a-era ``test_boot_through_empty_ladder_leaves_user_version_zero``:
    with step 1 live, a fresh boot lands ``user_version == 1`` over the V1a STRICT
    schema rather than the empty-ladder ``0``.
    """
    conn = temp_store._get_connection()
    try:
        assert conn.execute("PRAGMA user_version;").fetchone()[0] == 1
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
    tracked set that provably empties by Phase 8a: Phase 5d removes the read-view
    consumers it restores, Phase 8a removes the rest. Pinning it here makes any
    change a conscious edit (and any restoring phase must update this set), never a
    silent drift. The set was derived empirically from the flip (not the pre-existing
    ``*_live.py`` 429 flakes, which are not quarantined).
    """
    from conftest import STORE_REBUILD_QUARANTINE

    assert set(STORE_REBUILD_QUARANTINE) == {
        # restored in Phase 5d (read views + modifier stamping)
        "test_list_decisions.py",
        "test_modifier_surfacing.py",
        "test_neighbor_review.py",
        "test_payload_economy.py",
        "test_surface_confidence.py",
        "test_status_readiness.py",
        "test_renderer.py",
        "test_adversarial_rendering.py",
        # restored in Phase 8a (consumer preservation)
        "test_sync.py",
        "test_importer.py",
        "test_record_decision.py",
        "test_relations_and_adjacency.py",
        "test_adversarial_invariants.py",
        "test_adversarial_mcp.py",
        "test_cli_pathologies.py",
    }


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
    setattr(e, edge_type, target)
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
    e.supersedes = "oq1"
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
    assert entries[0].supersedes == "[old-choice]"  # brackets retained by the parser
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
    e.supersedes = "nonexistent"
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
    e.supersedes = "a"
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(e)
    item = exc.value.failure.items[0]
    assert item.code == "dangling_edge"
    assert "b" in item.message  # the 1-hop killer's slug is named
    assert _count(temp_store, "nodes") == 2  # C rolled back


def test_cycle_violation_self_edge(temp_store: GraphStore) -> None:
    """A node citing its own slug (resolving to self) fires cycle_violation."""
    e = _decision(slug="x", axiom="Self.")
    e.supersedes = "x"
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
    e.supersedes = "c"
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
    e.supersedes = "oq1"  # decision -> open_question is cross-kind
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(e)
    assert exc.value.failure.items[0].code == "kind_constraint_violation"
    assert _count(temp_store, "edges") == 0
    assert _count(temp_store, "nodes") == 1  # only the OQ; the decision rolled back


# --- Warn-defer the seven non-V1a edge types -----------------------------------


def test_deferred_edge_types_warn_not_fail(
    temp_store: GraphStore, caplog
) -> None:
    """A V1b relationship field (e.g. Amends:) is logged, NOT committed, NOT a failure."""
    target = temp_store.commit_parsed_entry(_decision(slug="t", axiom="Target."))
    e = _decision(slug="d", axiom="A decision.")
    e.amends = "t"  # a V1b type — warn-deferred
    e.cites = "t"  # another V1b type
    e.supersedes = "t"  # a V1a kill-edge — this DOES commit

    with caplog.at_level(logging.WARNING, logger="mitos.store"):
        delta = temp_store.commit_parsed_entry(e)

    # The node + the kill-edge committed; the two deferred types wrote NO edges.
    assert _node_row(temp_store, delta.node_id) is not None
    rows = _edges(temp_store)
    assert len(rows) == 1
    assert rows[0]["edge_type"] == "supersedes"
    assert _is_active(temp_store, target.node_id) is False
    # Both deferred types are logged loudly (WARNING), naming the entry + field.
    assert "deferred to V1b" in caplog.text
    assert "amends" in caplog.text
    assert "cites" in caplog.text


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
