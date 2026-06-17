"""Adversarial test suite for the Mitos SQLite GraphStore.

Verifies database schema initialization, content-hash identity, computed states (M3),
declarative edge reconciliation (V1-D21), signals insert-or-ignore (MI-4), and
the CommitDelta cascade contract (V1-D22/D18).
"""

import sqlite3
import tempfile
import os
import pytest
from mitos.store import GraphStore, ValidationError
from mitos.errors import DatabaseError
from mitos.parser import ParsedEntry

@pytest.fixture
def temp_store() -> GraphStore:
    """Fixture that initializes a temporary in-memory-like file GraphStore."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    store = GraphStore(path)
    yield store
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


def test_store_commit_and_retrieve(temp_store: GraphStore) -> None:
    """Verifies basic commit and retrieval of nodes."""
    entry = ParsedEntry("decision", "core-isolation", 1, 10)
    entry.core_axiom = "We will isolate the pure logic core."
    entry.rejected_paths = "pgvector, or direct coupling."
    entry.scope = ["substrate"]

    delta = temp_store.commit_parsed_entry(entry)
    assert delta.node_id is not None
    assert delta.node_scope == ["substrate"]
    assert not delta.commentary_fields_changed

    node = temp_store.get_node(delta.node_id)
    assert node is not None
    assert node["slug"] == "core-isolation"
    assert node["core_axiom"] == "We will isolate the pure logic core."
    assert node["scope"] == ["substrate"]


def test_computed_state_traversal(temp_store: GraphStore) -> None:
    """Tests the M3 computed state derivation for supersession chains."""
    # 1. Commit first node (active)
    entry1 = ParsedEntry("decision", "db-choice", 1, 10)
    entry1.core_axiom = "Use pgvector."
    entry1.rejected_paths = "None."
    entry1.scope = ["database"]
    d1 = temp_store.commit_parsed_entry(entry1)

    # 2. Verify it is active
    conn = temp_store._get_connection()
    states = temp_store.compute_all_states(conn)
    assert states[d1.node_id] == "active"
    conn.close()

    # 3. Commit second node superseding the first
    entry2 = ParsedEntry("decision", "db-choice-new", 1, 10)
    entry2.core_axiom = "Use SQLite."
    entry2.rejected_paths = "pgvector."
    entry2.supersedes = "db-choice"
    entry2.scope = ["database"]
    d2 = temp_store.commit_parsed_entry(entry2)

    # 4. Verify computed states: d1 is superseded, d2 is active
    conn = temp_store._get_connection()
    states = temp_store.compute_all_states(conn)
    assert states[d1.node_id] == "superseded"
    assert states[d2.node_id] == "active"
    conn.close()


def test_declarative_edge_reconciliation(temp_store: GraphStore) -> None:
    """Verifies V1-D21 outgoing edges mirror the buffer and delete retired links."""
    # Create target nodes first
    target1 = ParsedEntry("decision", "t1", 1, 2)
    target1.core_axiom = "Target 1."
    target1.rejected_paths = "None."
    temp_store.commit_parsed_entry(target1)

    target2 = ParsedEntry("decision", "t2", 1, 2)
    target2.core_axiom = "Target 2."
    target2.rejected_paths = "None."
    temp_store.commit_parsed_entry(target2)

    # Commit source node pointing to t1
    source = ParsedEntry("decision", "src", 1, 5)
    source.core_axiom = "Source."
    source.rejected_paths = "None."
    source.supersedes = "t1"
    
    d_src = temp_store.commit_parsed_entry(source)
    edges = temp_store.get_edges()
    assert len(edges) == 1
    assert edges[0]["from_id"] == d_src.node_id
    assert edges[0]["type"] == "supersedes"

    # Re-commit source node pointing to t2 instead
    source.supersedes = "t2"
    d_src_updated = temp_store.commit_parsed_entry(source)
    
    # Assert outgoing edge reconciled: old deleted, new inserted
    edges_updated = temp_store.get_edges()
    assert len(edges_updated) == 1
    assert edges_updated[0]["from_id"] == d_src_updated.node_id
    assert edges_updated[0]["to_id"] != edges[0]["to_id"]


def test_signals_insert_or_ignore(temp_store: GraphStore) -> None:
    """Tests the MI-4 partial unique index insert-or-ignore rule."""
    entry = ParsedEntry("decision", "sig-test", 1, 5)
    entry.core_axiom = "Axiom."
    entry.rejected_paths = "None."
    d = temp_store.commit_parsed_entry(entry)

    # Write drifted signal
    temp_store.write_signal(d.node_id, "drifted")
    # Duplicate write should be a silent no-op (insert-or-ignore enforced)
    temp_store.write_signal(d.node_id, "drifted")

    # Assert drifted state is derived
    conn = temp_store._get_connection()
    states = temp_store.compute_all_states(conn)
    assert states[d.node_id] == "drifted"
    conn.close()


def test_commit_delta_cascade_scopes(temp_store: GraphStore) -> None:
    """Tests that CommitDelta returns accurate cascade scopes on status flips."""
    # 1. Commit target question in scope "auth"
    oq = ParsedEntry("open_question", "auth-roadblock", 1, 5)
    oq.questions_raised = ["How do we handle sessions?"]
    oq.scope = ["auth"]
    d_oq = temp_store.commit_parsed_entry(oq)

    # Verify open question is parked initially
    conn = temp_store._get_connection()
    assert temp_store.compute_all_states(conn)[d_oq.node_id] == "parked"
    conn.close()

    # 2. Commit resolving decision in scope "core"
    res = ParsedEntry("decision", "resolve-auth", 1, 5)
    res.core_axiom = "Use stateless JWTs."
    res.rejected_paths = "Sessions."
    res.resolves = "auth-roadblock"
    res.scope = ["core"]
    
    delta = temp_store.commit_parsed_entry(res)
    
    # Assert resolving decision caused state flip of the OQ, returning its scope in cascade
    assert "auth" in delta.cascade_affected_scopes


def test_wal_concurrency_multi_reader(temp_store: GraphStore) -> None:
    """Verifies SQLite WAL concurrency permits multiple parallel readers and a writer."""
    # 1. Open main connection and write initial node
    conn_writer = temp_store._get_connection()
    cursor = conn_writer.cursor()
    cursor.execute(
        "INSERT INTO nodes (id, slug, kind, core_axiom, rejected_paths) VALUES (?, ?, ?, ?, ?)",
        ("test-id", "test-slug", "decision", "My core axiom", "None")
    )
    conn_writer.commit()

    # 2. Start a transaction on the writer connection but do not commit it yet
    conn_writer.execute("BEGIN IMMEDIATE TRANSACTION;")
    conn_writer.execute(
        "UPDATE nodes SET core_axiom = 'Axiom Modified' WHERE id = 'test-id'"
    )

    # 3. Open a separate reader connection
    conn_reader = temp_store._get_connection()
    cursor_reader = conn_reader.cursor()
    
    # In WAL mode, the reader is not blocked by the writer's immediate transaction!
    # The reader sees the state before the uncommitted write (snapshot isolation).
    cursor_reader.execute("SELECT core_axiom FROM nodes WHERE id = 'test-id'")
    row = cursor_reader.fetchone()
    assert row["core_axiom"] == "My core axiom"
    
    # 4. Commit the write
    conn_writer.commit()
    conn_writer.close()

    # 5. Reader now sees modified state after a new query
    cursor_reader.execute("SELECT core_axiom FROM nodes WHERE id = 'test-id'")
    row_after = cursor_reader.fetchone()
    assert row_after["core_axiom"] == "Axiom Modified"
    conn_reader.close()


# --- Phase 2a: connection hardening (PRAGMA suite, version guard, ladder boot) ---


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
    """The FK PRAGMA actually takes effect: an orphan child insert is rejected.

    Generic proof that ``foreign_keys=ON`` is live (the V1a kind-matrix schema
    arrives in 2b). With FK off, the orphan insert would silently succeed.
    """
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


def test_boot_through_empty_ladder_leaves_user_version_zero(
    temp_store: GraphStore,
) -> None:
    """Option-A wiring boots the empty ladder as a no-op (user_version stays 0)."""
    conn = temp_store._get_connection()
    try:
        assert conn.execute("PRAGMA user_version;").fetchone()[0] == 0
    finally:
        conn.close()


def test_v1_schema_authored_but_not_live_until_phase_5a(
    temp_store: GraphStore,
) -> None:
    """DEFERRED-FLIP TRIPWIRE — Phase 2b authors _v1_schema but does NOT make it live.

    Phase 2b builds and fully proves the V1a STRICT schema (``_v1_schema``) but
    deliberately does NOT register it (Key Decision 1): the live boot stays on the
    prototype ``_init_db`` so the suite stays green through 2b–4b. **Phase 5a** is
    the flipping phase — it appends ``(1, _v1_schema)`` to ``MIGRATION_STEPS`` and
    retires ``_init_db`` in lockstep with the ``commit_parsed_entry`` rebuild that
    writes the V1a schema. When 5a does that, this test MUST be consciously
    updated; it is the forcing function that makes the deferred wiring visible.

    Two assertions pin the deferred state from both sides — the boot still lands on
    the prototype (``user_version == 0``) AND the schema step is absent from the
    live registry.
    """
    from mitos.migrations import MIGRATION_STEPS, _v1_schema

    conn = temp_store._get_connection()
    try:
        assert conn.execute("PRAGMA user_version;").fetchone()[0] == 0
    finally:
        conn.close()
    assert (1, _v1_schema) not in MIGRATION_STEPS
    assert MIGRATION_STEPS == []
