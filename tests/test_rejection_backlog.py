"""The rejection index's graph half: ladder step 5, the field-aware enqueue, the primitives.

Recall-confidence phase 4a. One rung adds ``rejection_backlog`` (one row per decision
whose rejected items await a write), ``rejection_seed`` (the rejection index's own seed
marker) and the decision outbox's keyset index. ``commit_parsed_entry`` enqueues a
backlog row in its own transaction exactly when the committed decision is new with an
item, or its stored ``rejected_paths`` changed. The rows here run against real SQLite
with no mocks except where a fault is injected:

- the rung (A1–A6): fresh boot, upgrade with its snapshot, replay, an older build over
  the upgraded graph, the DDL, the query plans;
- a read-only store over a pre-rung graph (R1–R2): the reads fall back on exactly the
  missing table, and any other fault still raises;
- the enqueue (E1–E7): which commits enqueue, which do not, and the transaction;
- the primitives and the death story (P1–P5): the seq compare (ABA included), the
  claim walk, the foreign key, and the marker.
"""

import hashlib
import os
import re
import sqlite3
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from mitos.errors import CommitError, DatabaseError
from mitos.migrations import (
    MIGRATION_STEPS,
    REJECTION_SEED_ESTABLISHED_BY,
    _pending_head,
    _v4a_rejection_substrate,
    run_migrations,
)
from mitos.parser import ParsedEntry
from mitos.store import (
    GraphStore,
    _boot_migrations,
    open_connection,
    write_rejection_seed,
)

_TS = "2026-10-16T00:00:00.000000+00:00"
_OLD = "2000-01-01T00:00:00.000000+00:00"
_NUMBERED = "1. A sentinel point — it sits in every search.\n2. A graph row — it outlives the collection."


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def store(tmp_path) -> GraphStore:
    """A fresh graph, laddered to the live head."""
    db_path = str(tmp_path / ".mitos" / "graph.sqlite")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return GraphStore(db_path)


def _decision(
    slug: str = "d-one",
    axiom: str = "Use a sibling collection.",
    rejected: str = _NUMBERED,
    **fields: Any,
) -> ParsedEntry:
    """Builds a hand-made decision ``ParsedEntry`` (fields set as attributes)."""
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = axiom
    entry.rejected_paths = rejected
    for name, value in fields.items():
        setattr(entry, name, value)
    return entry


def _open_question(slug: str = "q-one") -> ParsedEntry:
    entry = ParsedEntry("open_question", slug, 1, 5)
    entry.topic = "A topic."
    entry.questions_raised = ["A question?"]
    return entry


def _raw(path: str, sql: str, params: Tuple = ()) -> List[Tuple]:
    conn = sqlite3.connect(path)
    try:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _user_version(path: str) -> int:
    return _raw(path, "PRAGMA user_version")[0][0]


def _sha(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _backlog(store: GraphStore) -> List[Tuple]:
    """Every backlog row as ``(node_id, enqueue_seq, retry_count, queued_at)``."""
    return _raw(
        store.db_path,
        "SELECT node_id, enqueue_seq, retry_count, queued_at FROM rejection_backlog "
        "ORDER BY enqueue_seq",
    )


def _outbox_stamp(store: GraphStore, node_id: str) -> Optional[str]:
    rows = _raw(
        store.db_path,
        "SELECT queued_at FROM pending_embeddings WHERE node_id = ?",
        (node_id,),
    )
    return rows[0][0] if rows else None


def _age_outbox(store: GraphStore, node_id: str) -> None:
    conn = open_connection(store.db_path)
    try:
        conn.execute(
            "UPDATE pending_embeddings SET queued_at = ? WHERE node_id = ?",
            (_OLD, node_id),
        )
        conn.commit()
    finally:
        conn.close()


def _build_step4_graph(path: str) -> None:
    """A populated graph at ladder step 4, the shape every pre-4a graph has.

    A writable ``GraphStore`` always ladders to the live head, so the graph is laddered
    by injection and filled with raw inserts (the step-3 idiom in test_scope_ordinal).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = open_connection(path)
    try:
        run_migrations(conn, MIGRATION_STEPS[:4])
        for n in range(3):
            node_id = f"{n:064x}"
            conn.execute(
                "INSERT INTO nodes (id, kind, slug, slug_casefold, source, axiom, "
                "mechanism_refs_json, rejected_paths_json, created_at, updated_at) "
                "VALUES (?, 'decision', ?, ?, 'user', ?, '[]', ?, ?, ?)",
                (node_id, f"d-{n}", f"d-{n}", f"Axiom {n}.", f"Not {n}.", _TS, _TS),
            )
            conn.execute(
                "INSERT INTO pending_embeddings (node_id, queued_at) VALUES (?, ?)",
                (node_id, _TS),
            )
        conn.commit()
    finally:
        conn.close()


def _rows(path: str, table: str) -> List[Tuple]:
    return _raw(path, f"SELECT * FROM {table} ORDER BY rowid")


def _table_sql(path: str, name: str) -> str:
    return _raw(
        path, "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    )[0][0]


def _tables(path: str) -> set:
    return {r[0] for r in _raw(path, "SELECT name FROM sqlite_master WHERE type = 'table'")}


def _indexes(path: str) -> set:
    return {r[0] for r in _raw(path, "SELECT name FROM sqlite_master WHERE type = 'index'")}


# --------------------------------------------------------------------------- #
# A — the rung
# --------------------------------------------------------------------------- #


def test_a1_a_fresh_boot_reaches_the_head_with_both_tables_empty_and_the_index(
    store: GraphStore,
) -> None:
    path = store.db_path
    assert _user_version(path) == _pending_head(MIGRATION_STEPS)
    assert {"rejection_backlog", "rejection_seed", "sqlite_sequence"} <= _tables(path)
    assert "idx_pending_embeddings_queue" in _indexes(path)
    assert _rows(path, "rejection_backlog") == []
    assert _rows(path, "rejection_seed") == []
    assert not [f for f in os.listdir(os.path.dirname(path)) if ".snapshot_v" in f]


def test_a1_the_runners_own_default_registry_reaches_the_head(tmp_path) -> None:
    """``run_migrations``' default argument binds the registry *object* at def-time, so
    a rung registered by rebinding ``MIGRATION_STEPS`` is invisible to it."""
    conn = open_connection(str(tmp_path / "default.sqlite"))
    try:
        assert run_migrations(conn) == _pending_head(MIGRATION_STEPS)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        conn.close()
    assert {"rejection_backlog", "rejection_seed"} <= tables


def test_a2_an_upgraded_step4_graph_gains_the_rung_keeps_its_rows_and_its_snapshot(
    tmp_path,
) -> None:
    path = str(tmp_path / ".mitos" / "graph.sqlite")
    _build_step4_graph(path)
    nodes_before = _rows(path, "nodes")
    outbox_before = _rows(path, "pending_embeddings")
    assert "rejection_backlog" not in _tables(path)

    GraphStore(path)

    assert _user_version(path) == _pending_head(MIGRATION_STEPS)
    assert os.path.exists(path + ".snapshot_v4")
    assert _user_version(path + ".snapshot_v4") == 4
    assert _rows(path, "rejection_backlog") == []  # no backfill
    assert _rows(path, "rejection_seed") == []
    assert _rows(path, "nodes") == nodes_before
    assert _rows(path, "pending_embeddings") == outbox_before
    assert "idx_pending_embeddings_queue" in _indexes(path)


def test_a3_replaying_the_ladder_and_the_step_itself_changes_nothing(
    store: GraphStore,
) -> None:
    store.commit_parsed_entry(_decision())
    schema = _raw(store.db_path, "SELECT type, name, sql FROM sqlite_master ORDER BY name")
    backlog = _backlog(store)

    conn = open_connection(store.db_path)
    try:
        assert run_migrations(conn, MIGRATION_STEPS) == _pending_head(MIGRATION_STEPS)
        conn.execute("BEGIN")
        _v4a_rejection_substrate(conn)  # IF NOT EXISTS throughout: raises nothing
        conn.execute("COMMIT")
    finally:
        conn.close()

    assert _raw(store.db_path, "SELECT type, name, sql FROM sqlite_master ORDER BY name") == schema
    assert _backlog(store) == backlog


def test_a4_an_older_build_runs_nothing_over_the_head_and_its_writer_still_commits(
    store: GraphStore,
) -> None:
    head = _pending_head(MIGRATION_STEPS)
    snapshots_before = sorted(os.listdir(os.path.dirname(store.db_path)))

    _boot_migrations(store.db_path, MIGRATION_STEPS[:4])  # the 0.19.x boot
    conn = open_connection(store.db_path)
    try:
        assert run_migrations(conn, MIGRATION_STEPS[:4]) == head
    finally:
        conn.close()

    assert _user_version(store.db_path) == head
    assert sorted(os.listdir(os.path.dirname(store.db_path))) == snapshots_before
    assert {"rejection_backlog", "rejection_seed"} <= _tables(store.db_path)

    # The older commit's own statements: a node and its outbox row, no backlog.
    conn = open_connection(store.db_path)
    try:
        conn.execute(
            "INSERT INTO nodes (id, kind, slug, slug_casefold, source, axiom, "
            "mechanism_refs_json, rejected_paths_json, created_at, updated_at) "
            "VALUES (?, 'decision', 'old', 'old', 'user', 'Old.', '[]', ?, ?, ?)",
            ("a" * 64, _NUMBERED, _TS, _TS),
        )
        conn.execute(
            "INSERT INTO pending_embeddings (node_id, queued_at, retry_count) "
            "VALUES (?, ?, 0) ON CONFLICT(node_id) DO UPDATE SET "
            "queued_at = excluded.queued_at, retry_count = 0",
            ("a" * 64, _TS),
        )
        conn.commit()
    finally:
        conn.close()
    assert _raw(store.db_path, "SELECT COUNT(*) FROM nodes WHERE id = ?", ("a" * 64,)) == [(1,)]
    assert _backlog(store) == []


@pytest.mark.parametrize("table", ["rejection_backlog", "rejection_seed"])
def test_a5_the_tables_are_strict_and_carry_no_current_timestamp(
    store: GraphStore, table: str
) -> None:
    sql = _table_sql(store.db_path, table)
    assert "CURRENT_TIMESTAMP" not in sql.upper()
    assert "STRICT" in sql.upper()


def test_a5_the_marker_check_set_equals_the_python_constant(store: GraphStore) -> None:
    sql = _table_sql(store.db_path, "rejection_seed")
    match = re.search(r"CHECK\s*\(\s*established_by\s+IN\s*\(([^)]*)\)", sql, re.I)
    assert match is not None, f"no established_by CHECK found in:\n{sql}"
    declared = {token.strip().strip("'") for token in match.group(1).split(",")}
    assert declared == set(REJECTION_SEED_ESTABLISHED_BY)
    assert "import" in declared


def test_a5_the_marker_refuses_an_unknown_verb_and_a_second_row(store: GraphStore) -> None:
    conn = open_connection(store.db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO rejection_seed (id, established_by, established_at) "
                "VALUES (1, 'sync', ?)",
                (_TS,),
            )
        conn.execute(
            "INSERT INTO rejection_seed (id, established_by, established_at) "
            "VALUES (1, 'import', ?)",
            (_TS,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO rejection_seed (id, established_by, established_at) "
                "VALUES (2, 'rebuild', ?)",
                (_TS,),
            )
    finally:
        conn.close()
    with pytest.raises(DatabaseError, match="rebuild, cutover, reconcile, import"):
        store.stamp_rejection_seed("sync")


class _RecordingCursor:
    """Forwards to a cursor, recording each statement with its bound parameters."""

    def __init__(self, cursor: sqlite3.Cursor, seen: List[Tuple[str, Tuple]]) -> None:
        self._cursor = cursor
        self._seen = seen

    def execute(self, sql: str, params: Any = ()) -> Any:
        self._seen.append((sql, tuple(params)))
        return self._cursor.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


class _RecordingConnection(_RecordingCursor):
    """A connection whose ``execute`` and cursors record what they run."""

    def cursor(self) -> _RecordingCursor:
        return _RecordingCursor(self._cursor.cursor(), self._seen)


def _statements(store: GraphStore, call: Callable[[], Any]) -> List[Tuple[str, Tuple]]:
    """Runs ``call`` and returns every SELECT the store ran, with its bound parameters.

    The parameters stay bound, never inlined: SQLite plans a literal differently from
    a parameter (measured: the ``OR`` keyset seeks the index with literals and scans it
    with parameters), and the store always binds.
    """
    seen: List[Tuple[str, Tuple]] = []
    original = store._get_connection
    store._get_connection = lambda: _RecordingConnection(original(), seen)  # type: ignore[method-assign]
    try:
        call()
    finally:
        store._get_connection = original  # type: ignore[method-assign]
    return [(sql, params) for sql, params in seen if sql.lstrip().upper().startswith("SELECT")]


def _plan(store: GraphStore, statement: Tuple[str, Tuple]) -> str:
    sql, params = statement
    conn = sqlite3.connect(store.db_path)
    try:
        rows = conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
    finally:
        conn.close()
    return " | ".join(r[3] for r in rows)


def test_a6_the_outbox_claims_seek_the_new_index(store: GraphStore) -> None:
    head = _statements(store, lambda: store.claim_pending_embeddings("d", 10))
    keyset = _statements(
        store, lambda: store.claim_pending_embeddings("d", 10, after=(_TS, "a" * 64))
    )
    assert len(head) == 1 and len(keyset) == 1
    head_plan, keyset_plan = _plan(store, head[0]), _plan(store, keyset[0])
    assert "idx_pending_embeddings_queue" in head_plan, head_plan
    assert "TEMP B-TREE" not in head_plan, head_plan
    # The keyset must SEEK to its cursor, not walk the index from its head.
    assert "SEARCH" in keyset_plan and "idx_pending_embeddings_queue" in keyset_plan, keyset_plan
    assert "TEMP B-TREE" not in keyset_plan, keyset_plan


def test_a6_the_backlog_claims_walk_the_primary_key(store: GraphStore) -> None:
    head = _statements(store, lambda: store.claim_rejection_backlog(10))
    keyset = _statements(store, lambda: store.claim_rejection_backlog(10, after_seq=3))
    assert len(head) == 1 and len(keyset) == 1
    head_plan, keyset_plan = _plan(store, head[0]), _plan(store, keyset[0])
    assert "TEMP B-TREE" not in head_plan, head_plan
    assert "INTEGER PRIMARY KEY" in keyset_plan, keyset_plan
    assert "TEMP B-TREE" not in keyset_plan, keyset_plan


# --------------------------------------------------------------------------- #
# R — read-only over a pre-rung graph
# --------------------------------------------------------------------------- #


def test_r1_a_read_only_store_over_a_step4_graph_reads_empty_and_absent(tmp_path) -> None:
    path = str(tmp_path / ".mitos" / "graph.sqlite")
    _build_step4_graph(path)
    sha = _sha(path)

    ro = GraphStore(path, read_only=True)
    assert ro.get_rejection_backlog_row(f"{0:064x}") is None
    assert ro.claim_rejection_backlog(10) == []
    assert ro.claim_rejection_backlog(10, after_seq=1) == []
    assert ro.rejection_seed() is None

    assert _user_version(path) == 4
    assert _sha(path) == sha


class _FaultingConnection:
    """A connection whose every ``execute`` raises the given SQLite error."""

    def __init__(self, error: sqlite3.Error) -> None:
        self.error = error

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise self.error

    def close(self) -> None:
        return None


@pytest.mark.parametrize(
    "message", ["database is locked", "database disk image is malformed"]
)
def test_r2_any_other_read_fault_still_raises(store: GraphStore, message: str) -> None:
    store._get_connection = lambda: _FaultingConnection(  # type: ignore[method-assign]
        sqlite3.OperationalError(message)
    )
    with pytest.raises(DatabaseError, match=message):
        store.get_rejection_backlog_row("a" * 64)
    with pytest.raises(DatabaseError, match=message):
        store.claim_rejection_backlog(10)
    with pytest.raises(DatabaseError, match=message):
        store.claim_rejection_backlog(10, after_seq=1)


def test_r2_the_writes_never_fall_back_on_a_missing_table(tmp_path) -> None:
    path = str(tmp_path / ".mitos" / "graph.sqlite")
    _build_step4_graph(path)
    # A writable store would ladder it; drive the writes over the pre-rung file.
    store = GraphStore.__new__(GraphStore)
    store.db_path = path
    store.read_only = False
    with pytest.raises(DatabaseError, match="no such table"):
        store.clear_rejection_backlog_row("a" * 64, enqueue_seq=1)
    with pytest.raises(DatabaseError, match="no such table"):
        store.note_rejection_backlog_failure("a" * 64, enqueue_seq=1)
    with pytest.raises(DatabaseError, match="no such table"):
        store.stamp_rejection_seed("reconcile")
    with pytest.raises(DatabaseError, match="no such table"):
        store.clear_rejection_seed()


# --------------------------------------------------------------------------- #
# E — the enqueue
# --------------------------------------------------------------------------- #


def test_e1_a_new_decision_with_items_enqueues_one_fresh_row(store: GraphStore) -> None:
    node_id = store.commit_parsed_entry(_decision()).node_id
    rows = _backlog(store)
    assert len(rows) == 1
    owner, seq, retries, queued_at = rows[0]
    assert owner == node_id and retries == 0 and seq > 0
    updated_at = _raw(store.db_path, "SELECT updated_at FROM nodes WHERE id = ?", (node_id,))
    assert queued_at == updated_at[0][0]
    assert store.get_rejection_backlog_row(node_id) == (node_id, queued_at, seq, 0)


def test_e2_a_new_decision_with_no_item_and_an_open_question_enqueue_nothing(
    store: GraphStore,
) -> None:
    # Unreachable through the parser (an empty Rejected is a parse failure), so the
    # entry is hand-built, as a caller bypassing it would be.
    store.commit_parsed_entry(_decision(rejected="   \n\t "))
    store.commit_parsed_entry(_decision("d-two", "Another axiom.", rejected=""))
    store.commit_parsed_entry(_open_question())
    assert _backlog(store) == []
    # Their outbox rows were written as always.
    assert len(_raw(store.db_path, "SELECT * FROM pending_embeddings")) == 3


def test_e5_a_change_to_empty_requeues_and_none_against_empty_does_not(
    store: GraphStore,
) -> None:
    node_id = store.commit_parsed_entry(_decision()).node_id
    first = store.get_rejection_backlog_row(node_id)

    store.commit_parsed_entry(_decision(rejected=""))
    emptied = store.get_rejection_backlog_row(node_id)
    assert emptied is not None and emptied.enqueue_seq > first.enqueue_seq

    # A NULL stored field against an incoming "" is no change.
    assert store.clear_rejection_backlog_row(node_id, enqueue_seq=emptied.enqueue_seq)
    conn = open_connection(store.db_path)
    try:
        conn.execute("UPDATE nodes SET rejected_paths_json = NULL WHERE id = ?", (node_id,))
        conn.commit()
    finally:
        conn.close()
    store.commit_parsed_entry(_decision(rejected=""))
    assert _backlog(store) == []


_EDITS: Dict[str, Callable[[], ParsedEntry]] = {
    "slug rename": lambda: _decision(slug="d-renamed"),
    "scope change": lambda: _decision(scope=["elsewhere"]),
    "context change": lambda: _decision(context="New context."),
    "invalidates_if change": lambda: _decision(invalidates_if="If it breaks."),
    "edge added": lambda: _decision(depends_on=["d-anchor"]),
    "transcript only": lambda: _decision(transcript="A transcript."),
    "byte-identical": lambda: _decision(),
}


@pytest.mark.parametrize("edit", sorted(_EDITS))
def test_e4_no_other_edit_touches_a_standing_row(store: GraphStore, edit: str) -> None:
    store.commit_parsed_entry(_decision("d-anchor", "An anchor."))
    node_id = store.commit_parsed_entry(_decision()).node_id
    before = store.get_rejection_backlog_row(node_id)
    _age_outbox(store, node_id)

    store.commit_parsed_entry(_EDITS[edit]())

    assert store.get_rejection_backlog_row(node_id) == before
    # The decision outbox stays unconditional: every re-commit re-stamps it.
    assert _outbox_stamp(store, node_id) not in (None, _OLD)


def test_e4_an_edge_removed_touches_no_standing_row(store: GraphStore) -> None:
    store.commit_parsed_entry(_decision("d-anchor", "An anchor."))
    node_id = store.commit_parsed_entry(_decision(depends_on=["d-anchor"])).node_id
    before = store.get_rejection_backlog_row(node_id)
    _age_outbox(store, node_id)

    store.commit_parsed_entry(_decision())  # the depends_on edge dropped

    assert _raw(store.db_path, "SELECT COUNT(*) FROM edges WHERE source_id = ?", (node_id,)) == [(0,)]
    assert store.get_rejection_backlog_row(node_id) == before
    assert _outbox_stamp(store, node_id) not in (None, _OLD)


def test_e4_a_recommit_is_not_the_rows_retry(store: GraphStore) -> None:
    node_id = store.commit_parsed_entry(_decision()).node_id
    row = store.get_rejection_backlog_row(node_id)
    assert store.note_rejection_backlog_failure(node_id, enqueue_seq=row.enqueue_seq)
    assert store.note_rejection_backlog_failure(node_id, enqueue_seq=row.enqueue_seq)

    store.commit_parsed_entry(_decision())  # byte-identical

    after = store.get_rejection_backlog_row(node_id)
    assert after.enqueue_seq == row.enqueue_seq
    assert after.retry_count == 2


def test_e6_a_fault_after_the_backlog_write_leaves_no_row(store: GraphStore) -> None:
    original = store._enqueue_rejection_backlog

    def write_then_fail(cursor: sqlite3.Cursor, node_id: str, now: str) -> None:
        original(cursor, node_id, now)
        assert _raw_count_via(cursor, "rejection_backlog") == 1  # it did write
        raise RuntimeError("injected after the backlog write")

    store._enqueue_rejection_backlog = write_then_fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="injected"):
        store.commit_parsed_entry(_decision())
    assert _raw(store.db_path, "SELECT COUNT(*) FROM nodes") == [(0,)]
    assert _raw(store.db_path, "SELECT COUNT(*) FROM pending_embeddings") == [(0,)]
    assert _backlog(store) == []


def _raw_count_via(cursor: sqlite3.Cursor, table: str) -> int:
    return cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_e6_a_slug_collision_after_both_enqueues_leaves_neither_row(
    store: GraphStore,
) -> None:
    keeper = store.commit_parsed_entry(_decision("taken", "The first axiom.")).node_id
    standing = _backlog(store)

    with pytest.raises(CommitError):
        store.commit_parsed_entry(_decision("taken", "A different axiom."))

    assert _raw(store.db_path, "SELECT id FROM nodes") == [(keeper,)]
    assert _raw(store.db_path, "SELECT node_id FROM pending_embeddings") == [(keeper,)]
    assert _backlog(store) == standing


def test_e7_a_rebuild_replay_queues_every_decision_holding_items(tmp_path) -> None:
    from mitos.config import MitosConfig
    from mitos.cutover import rebuild_and_gate

    root = tmp_path / "ws"
    os.makedirs(root / ".mitos", exist_ok=True)
    config = MitosConfig(str(root))
    with open(config.decisions_file, "w", encoding="utf-8") as handle:
        handle.write(
            "<!-- BEGIN ENTRIES -->\n\n"
            "### d-new\n\n**Decided:** The new axiom.\n"
            "**Rejected:**\n1. Keep the old — it drifted.\n2. Do nothing — it rots.\n"
            "**Supersedes:** d-old\n\n"
            "### d-old\n\n**Decided:** The old axiom.\n**Rejected:** n/a\n\n"
            "### d-live\n\n**Decided:** A live axiom.\n**Rejected:** Something else.\n"
        )
    with open(config.questions_file, "w", encoding="utf-8") as handle:
        handle.write(
            "<!-- BEGIN ENTRIES -->\n\n"
            "### q-one\n\n**Topic:** A topic.\n**Questions:** Why?\n"
        )

    result = rebuild_and_gate(config, aside_db_path=config.db_path + ".aside")
    aside = GraphStore(result.aside_db_path)

    decisions = {
        n["id"] for n in aside.get_all_nodes() if n["kind"] == "decision"
    }
    assert len(decisions) == 3
    assert {row[0] for row in _backlog(aside)} == decisions
    assert all(row[2] == 0 for row in _backlog(aside))


# --------------------------------------------------------------------------- #
# E3 — the real amend route, on a live and on a retired decision
# --------------------------------------------------------------------------- #


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A real ``cmd_init`` workspace, keyless and serviceless (test_amend_commentary)."""
    from unittest.mock import MagicMock

    from mitos.cli import cmd_init
    from mitos.config import MitosConfig
    from mitos.sync import MitosSyncManager

    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:9")
    down = MagicMock(side_effect=Exception("backend down"))
    monkeypatch.setattr("mitos.sync.GeminiEmbeddingProvider", down)
    monkeypatch.setattr("mitos.sync.QdrantVectorStore", down)
    root = tmp_path / "ws"
    root.mkdir()
    config = MitosConfig(str(root))
    cmd_init(config)
    return MitosSyncManager(config)


def _record(m: Any, slug: str, **relations: Any) -> str:
    result = m.record_decision_entry(
        f"The {slug} axiom.", f"The {slug} rejected reasoning.", ["alpha"],
        mechanisms=[f"{slug}-mechanism"], context=f"The {slug} context.", slug=slug,
        acknowledge_neighbors=True, **relations,
    )
    assert result["status"] == "created", result
    return result["id"]


@pytest.mark.parametrize("target", ["live", "retired"])
def test_e3_a_rejected_paths_amend_requeues_with_a_larger_seq(ws: Any, target: str) -> None:
    live_id = _record(ws, "live")
    old_id = _record(ws, "old")
    _record(ws, "successor", supersedes="old")
    node_id = live_id if target == "live" else old_id
    slug = "live" if target == "live" else "old"
    if target == "retired":
        assert ws.store.get_node_state(old_id) == "superseded"

    row = ws.store.get_rejection_backlog_row(node_id)
    assert row is not None
    assert ws.store.note_rejection_backlog_failure(node_id, enqueue_seq=row.enqueue_seq)

    result = ws.amend_commentary(slug, {"rejected_paths": "1. A new option — newly turned down."})
    assert result["status"] == "amended", result

    after = ws.store.get_rejection_backlog_row(node_id)
    assert after.enqueue_seq > row.enqueue_seq
    assert after.retry_count == 0


def test_e3_a_context_amend_requeues_nothing(ws: Any) -> None:
    node_id = _record(ws, "live")
    row = ws.store.get_rejection_backlog_row(node_id)
    assert ws.amend_commentary("live", {"context": "Another context."})["status"] == "amended"
    assert ws.store.get_rejection_backlog_row(node_id) == row


# --------------------------------------------------------------------------- #
# P — the primitives and the death story
# --------------------------------------------------------------------------- #


def test_p1_a_clear_lands_only_on_the_seq_it_read_aba_included(store: GraphStore) -> None:
    node_id = store.commit_parsed_entry(_decision()).node_id
    first = store.get_rejection_backlog_row(node_id)

    # A re-queue since the read defeats the stale clear; the row stands.
    store.commit_parsed_entry(_decision(rejected="1. Only one option — now."))
    second = store.get_rejection_backlog_row(node_id)
    assert not store.clear_rejection_backlog_row(node_id, enqueue_seq=first.enqueue_seq)
    assert store.get_rejection_backlog_row(node_id) == second

    # The current seq clears.
    assert store.clear_rejection_backlog_row(node_id, enqueue_seq=second.enqueue_seq)
    assert store.get_rejection_backlog_row(node_id) is None

    # ABA: cleared, re-inserted — a clear holding the FIRST seq still loses.
    store.commit_parsed_entry(_decision())
    third = store.get_rejection_backlog_row(node_id)
    assert third.enqueue_seq > second.enqueue_seq > first.enqueue_seq
    assert not store.clear_rejection_backlog_row(node_id, enqueue_seq=first.enqueue_seq)
    assert not store.clear_rejection_backlog_row(node_id, enqueue_seq=second.enqueue_seq)
    assert store.get_rejection_backlog_row(node_id) == third


def test_p2_a_failure_note_lands_only_on_its_seq(store: GraphStore) -> None:
    node_id = store.commit_parsed_entry(_decision()).node_id
    first = store.get_rejection_backlog_row(node_id)
    assert store.note_rejection_backlog_failure(node_id, enqueue_seq=first.enqueue_seq)
    assert store.get_rejection_backlog_row(node_id).retry_count == 1

    store.commit_parsed_entry(_decision(rejected="1. Only one option — now."))
    second = store.get_rejection_backlog_row(node_id)
    assert second.retry_count == 0
    assert not store.note_rejection_backlog_failure(node_id, enqueue_seq=first.enqueue_seq)
    assert store.get_rejection_backlog_row(node_id) == second


def test_p3_the_claim_walks_in_enqueue_order_and_a_requeue_goes_to_the_back(
    store: GraphStore,
) -> None:
    a = store.commit_parsed_entry(_decision("d-a", "Axiom A.")).node_id
    b = store.commit_parsed_entry(_decision("d-b", "Axiom B.")).node_id
    c = store.commit_parsed_entry(_decision("d-c", "Axiom C.")).node_id

    page = store.claim_rejection_backlog(2)
    assert [r.node_id for r in page] == [a, b]
    rest = store.claim_rejection_backlog(2, after_seq=page[-1].enqueue_seq)
    assert [r.node_id for r in rest] == [c]

    # A commit re-queues A mid-walk: it reappears behind C in the same walk.
    store.commit_parsed_entry(_decision("d-a", "Axiom A.", rejected="1. Changed — yes."))
    rest = store.claim_rejection_backlog(10, after_seq=page[-1].enqueue_seq)
    assert [r.node_id for r in rest] == [c, a]
    assert store.claim_rejection_backlog(10, after_seq=rest[-1].enqueue_seq) == []


def test_p4_a_backlog_row_naming_no_node_cannot_be_inserted(store: GraphStore) -> None:
    conn = open_connection(store.db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO rejection_backlog (node_id, queued_at) VALUES (?, ?)",
                ("f" * 64, _TS),
            )
    finally:
        conn.close()


def test_p5_the_marker_stamps_reads_replaces_and_spends_only_what_it_read(
    store: GraphStore,
) -> None:
    assert store.rejection_seed() is None
    store.stamp_rejection_seed("reconcile")
    first = store.rejection_seed()
    assert first["established_by"] == "reconcile"

    store.clear_rejection_seed(established_at="1999-01-01T00:00:00+00:00")
    assert store.rejection_seed() == first  # a stale stamp spends nothing

    store.stamp_rejection_seed("import")
    second = store.rejection_seed()
    assert second["established_by"] == "import"
    assert _raw(store.db_path, "SELECT COUNT(*) FROM rejection_seed") == [(1,)]

    store.clear_rejection_seed(established_at=second["established_at"])
    assert store.rejection_seed() is None

    # The in-transaction twin writes the same row on a caller's connection.
    conn = open_connection(store.db_path)
    try:
        write_rejection_seed(conn, "rebuild", now=_TS)
        conn.commit()
    finally:
        conn.close()
    assert store.rejection_seed() == {"established_by": "rebuild", "established_at": _TS}
    store.clear_rejection_seed()
    assert store.rejection_seed() is None


def test_p5_neither_marker_touches_the_others_table(store: GraphStore) -> None:
    store.stamp_rejection_seed("reconcile")
    assert store.embedding_seed() is None
    store.stamp_embedding_seed("reconcile")
    store.clear_embedding_seed()
    assert store.rejection_seed() is not None
    store.stamp_embedding_seed("rebuild")
    store.clear_rejection_seed()
    assert store.embedding_seed() is not None
    assert store.rejection_seed() is None


def test_p5_an_unreadable_marker_reads_as_absent(store: GraphStore) -> None:
    store.stamp_rejection_seed("reconcile")
    conn = open_connection(store.db_path)
    try:
        conn.execute("DROP TABLE rejection_seed")
        conn.commit()
    finally:
        conn.close()
    assert store.rejection_seed() is None
