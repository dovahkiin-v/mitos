"""Behavioral tests for the forward-only SQLite migration ladder (Phase 2a).

Proves the ladder mechanism end-to-end with synthetic steps, against real SQLite
(no mocks): the empty registry boots as a no-op, synthetic steps apply in
ascending order and gate on ``user_version``, a failing step rolls back its DDL
*and* its version bump together (MI-3 atomicity), and replaying an at-head ladder
changes nothing (MI-3 idempotency). The real ``MIGRATION_STEPS`` registry stays
empty in 2a; steps are injected via the ``steps=`` parameter so the empty-no-op
guarantee stays honest.
"""

import sqlite3

import pytest

from mitos.errors import DatabaseError
from mitos.migrations import (
    MIGRATION_STEPS,
    _v1_schema,
    is_pre_v1a_schema,
    run_migrations,
)
from mitos.store import GraphStore, open_connection


def _fresh_conn() -> sqlite3.Connection:
    """Opens a throwaway in-memory connection for pure ladder-logic tests."""
    return sqlite3.connect(":memory:")


def _user_version(conn: sqlite3.Connection) -> int:
    """Reads ``PRAGMA user_version`` back through the DB (never asserts a literal)."""
    return conn.execute("PRAGMA user_version;").fetchone()[0]


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Reports whether a table is present in the schema."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (name,)
    ).fetchone()
    return row is not None


def _create_step(table: str):
    """Builds a synthetic step that creates ``table`` (no IF NOT EXISTS).

    Omitting IF NOT EXISTS is deliberate: if the version gate ever failed and a
    step re-ran, ``CREATE TABLE`` would raise "table already exists" — so a clean
    idempotent-replay pass is real proof the gate held, not a masked double-apply.
    """

    def step(conn: sqlite3.Connection) -> None:
        conn.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY);")

    return step


def test_default_registry_is_empty() -> None:
    """The real registry ships empty in V1a 2a (the empty-case-first-class lever)."""
    assert MIGRATION_STEPS == []


def test_empty_ladder_is_noop() -> None:
    """The empty ladder on a fresh DB leaves user_version=0, creates nothing."""
    conn = _fresh_conn()
    head = run_migrations(conn)
    assert head == 0
    assert _user_version(conn) == 0
    # No tables of any kind were created.
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    ).fetchall()
    assert tables == []


def test_empty_ladder_replay_is_noop() -> None:
    """Replaying the empty ladder still changes nothing and raises nothing (MI-3)."""
    conn = _fresh_conn()
    assert run_migrations(conn) == 0
    assert run_migrations(conn) == 0
    assert _user_version(conn) == 0


def test_synthetic_steps_apply_and_advance_version() -> None:
    """Synthetic steps apply on a fresh DB and advance user_version to the head."""
    conn = _fresh_conn()
    steps = [(1, _create_step("alpha")), (2, _create_step("beta"))]
    head = run_migrations(conn, steps)
    assert head == 2
    assert _user_version(conn) == 2
    assert _table_exists(conn, "alpha")
    assert _table_exists(conn, "beta")


def test_steps_apply_in_ascending_order_regardless_of_registry_order() -> None:
    """Steps run in ascending version order even when the registry is unsorted."""
    conn = _fresh_conn()
    applied: list = []

    def recorder(version: int):
        def step(conn: sqlite3.Connection) -> None:
            applied.append(version)

        return step

    steps = [(3, recorder(3)), (1, recorder(1)), (2, recorder(2))]
    head = run_migrations(conn, steps)
    assert head == 3
    assert applied == [1, 2, 3]
    assert _user_version(conn) == 3


def test_gating_skips_steps_at_or_below_current_version() -> None:
    """A DB already at user_version=1 runs only step 2 (step 1 is gated out)."""
    conn = _fresh_conn()
    conn.execute("PRAGMA user_version = 1;")
    steps = [(1, _create_step("alpha")), (2, _create_step("beta"))]
    head = run_migrations(conn, steps)
    assert head == 2
    assert not _table_exists(conn, "alpha")  # gated — never ran
    assert _table_exists(conn, "beta")
    assert _user_version(conn) == 2


def test_at_head_ladder_runs_nothing() -> None:
    """A DB already at the ladder head applies no step (pure no-op)."""
    conn = _fresh_conn()
    conn.execute("PRAGMA user_version = 2;")
    steps = [(1, _create_step("alpha")), (2, _create_step("beta"))]
    head = run_migrations(conn, steps)
    assert head == 2
    assert not _table_exists(conn, "alpha")
    assert not _table_exists(conn, "beta")
    assert _user_version(conn) == 2


def test_idempotent_replay_of_applied_ladder_is_noop() -> None:
    """Running the same ladder twice is a zero-change no-op the second time (MI-3).

    The synthetic steps omit IF NOT EXISTS, so a re-run would raise on the second
    CREATE — a clean pass proves the version gate held.
    """
    conn = _fresh_conn()
    steps = [(1, _create_step("alpha")), (2, _create_step("beta"))]
    assert run_migrations(conn, steps) == 2
    assert run_migrations(conn, steps) == 2  # raises nothing; gate holds
    assert _user_version(conn) == 2
    assert _table_exists(conn, "alpha")
    assert _table_exists(conn, "beta")


def test_failing_step_rolls_back_ddl_and_version_together() -> None:
    """A step that raises mid-DDL leaves its table absent AND user_version unchanged."""
    conn = _fresh_conn()

    def doomed_step(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE doomed (id INTEGER PRIMARY KEY);")
        raise RuntimeError("boom mid-DDL")

    with pytest.raises(RuntimeError, match="boom"):
        run_migrations(conn, [(1, doomed_step)])

    assert not _table_exists(conn, "doomed")  # DDL rolled back
    assert _user_version(conn) == 0  # version bump rolled back with it


def test_prior_committed_step_survives_a_later_failing_step() -> None:
    """Each rung is its own transaction: an earlier success is not undone by a later raise."""
    conn = _fresh_conn()

    def doomed_step(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE doomed (id INTEGER PRIMARY KEY);")
        raise RuntimeError("boom")

    steps = [(1, _create_step("good")), (2, doomed_step)]
    with pytest.raises(RuntimeError, match="boom"):
        run_migrations(conn, steps)

    assert _table_exists(conn, "good")  # step 1 committed independently
    assert not _table_exists(conn, "doomed")  # step 2 rolled back
    assert _user_version(conn) == 1  # advanced to 1, not 2


def test_partial_ladder_resumes_from_mid_version() -> None:
    """A DB partway up the ladder applies only the remaining rungs."""
    conn = _fresh_conn()
    conn.execute("PRAGMA user_version = 1;")
    steps = [
        (1, _create_step("alpha")),
        (2, _create_step("beta")),
        (3, _create_step("gamma")),
    ]
    head = run_migrations(conn, steps)
    assert head == 3
    assert not _table_exists(conn, "alpha")  # already applied before this run
    assert _table_exists(conn, "beta")
    assert _table_exists(conn, "gamma")
    assert _user_version(conn) == 3


def test_rejects_nonpositive_version() -> None:
    """A non-positive step version is rejected before any step runs."""
    conn = _fresh_conn()
    with pytest.raises(DatabaseError):
        run_migrations(conn, [(0, _create_step("alpha"))])
    assert not _table_exists(conn, "alpha")


def test_rejects_duplicate_versions() -> None:
    """Two steps sharing a version are rejected (strict monotonicity)."""
    conn = _fresh_conn()
    with pytest.raises(DatabaseError):
        run_migrations(conn, [(1, _create_step("alpha")), (1, _create_step("beta"))])


def test_rejects_non_integer_version() -> None:
    """A non-integer step version is rejected (guards the P8 interpolation carve-out)."""
    conn = _fresh_conn()
    with pytest.raises(DatabaseError):
        run_migrations(conn, [("1", _create_step("alpha"))])


# --- Phase 2b: the V1a STRICT-table schema as ladder step 1 ------------------
#
# The schema is authored + proven here via INJECTION — run_migrations(conn,
# steps=[(1, _v1_schema)]) — exactly as 2a proved the runner with synthetic
# steps; the live MIGRATION_STEPS stays empty and store.py is byte-for-byte
# unchanged (Phase 5a does the live flip). Achieved state is always read back
# through the DB (PRAGMA / introspection), never asserted against a literal we
# also wrote. Constraint-firing tests run through ``open_connection`` so
# ``foreign_keys=ON`` is live (MI-8) — a bare ``sqlite3.connect`` would make every
# FK/CHECK-via-FK assertion silently pass nothing.

# A representative app-supplied UTC ISO-8601 µs timestamp (the DDL carries no
# CURRENT_TIMESTAMP default; every writer supplies one — MI-10).
_TS = "2026-06-18T00:00:00.000000+00:00"

_V1A_TABLES = (
    "nodes",
    "node_scopes",
    "edges",
    "transcripts",
    "signals",
    "pending_embeddings",
)


def _v1_conn(tmp_path, name: str = "v1a.sqlite") -> sqlite3.Connection:
    """Opens an FK-on file DB and advances it to the V1a schema head (user_version 1).

    Goes through ``open_connection`` (the MI-8 chokepoint) so the FK/kind-matrix
    constraint tests actually enforce; the schema is applied via injection so the
    live registry stays empty.
    """
    conn = open_connection(str(tmp_path / name))
    run_migrations(conn, steps=[(1, _v1_schema)])
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> dict:
    """Reads back a table's columns as ``{name: {'notnull': int, 'pk': int}}``."""
    return {
        row[0]: {"notnull": row[1], "pk": row[2]}
        for row in conn.execute(
            'SELECT name, "notnull", pk FROM pragma_table_info(?);', (table,)
        )
    }


def _is_strict(conn: sqlite3.Connection, table: str) -> bool:
    """Reports whether ``table`` is a STRICT table (read back via pragma_table_list)."""
    row = conn.execute(
        "SELECT strict FROM pragma_table_list WHERE name = ?;", (table,)
    ).fetchone()
    return row is not None and bool(row[0])


def _index_unique_flags(conn: sqlite3.Connection, table: str) -> dict:
    """Reads back named indexes as ``{index_name: unique_flag}`` for a table."""
    return {
        row[0]: row[1]
        for row in conn.execute(
            'SELECT name, "unique" FROM pragma_index_list(?);', (table,)
        )
    }


def _unique_index_colsets(conn: sqlite3.Connection, table: str) -> list:
    """Reads back the column tuples of every UNIQUE index on ``table``."""
    colsets = []
    for row in conn.execute(
        'SELECT name, "unique" FROM pragma_index_list(?);', (table,)
    ):
        if not row[1]:
            continue
        cols = [r[0] for r in conn.execute("SELECT name FROM pragma_index_info(?);", (row[0],))]
        colsets.append(tuple(cols))
    return colsets


def _insert_node(
    conn: sqlite3.Connection,
    node_id: str,
    kind: str = "decision",
    source: str = "user",
    slug: str = None,
) -> None:
    """Inserts a fully-populated ``nodes`` row (every NOT NULL column supplied, §8)."""
    slug = slug if slug is not None else node_id
    conn.execute(
        "INSERT INTO nodes "
        "(id, kind, slug, slug_casefold, source, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?);",
        (node_id, kind, slug, slug.casefold(), source, _TS, _TS),
    )


def _insert_edge(
    conn: sqlite3.Connection,
    source_id: str,
    source_kind: str,
    target_id: str,
    target_kind: str,
    edge_type: str = "supersedes",
) -> None:
    """Inserts an ``edges`` row with every NOT NULL column supplied."""
    conn.execute(
        "INSERT INTO edges "
        "(source_id, source_kind, target_id, target_kind, edge_type, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?);",
        (source_id, source_kind, target_id, target_kind, edge_type, _TS),
    )


# --- From-scratch boot check + shape ---

def test_v1_schema_creates_six_strict_tables_and_advances_version(tmp_path) -> None:
    """Injecting step 1 on a fresh DB yields user_version 1 and exactly the six STRICT tables."""
    conn = _v1_conn(tmp_path)
    try:
        assert _user_version(conn) == 1
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            )
        }
        assert tables == set(_V1A_TABLES)
        for table in _V1A_TABLES:
            assert _is_strict(conn, table), f"{table} is not STRICT"
    finally:
        conn.close()


def test_v1_schema_nodes_columns_keys_and_unique_id_kind(tmp_path) -> None:
    """nodes has the §8.2 columns, the NOT NULL/PK flags, and the UNIQUE (id, kind) the edge FK needs."""
    conn = _v1_conn(tmp_path)
    try:
        cols = _columns(conn, "nodes")
        assert set(cols) == {
            "id", "kind", "slug", "slug_casefold", "source", "axiom",
            "mechanism_refs_json", "topic", "questions_raised_json",
            "rejected_paths_json", "invalidates_if", "context", "confirmed_by",
            "confirmed_at", "created_at", "updated_at",
        }
        assert cols["id"]["pk"] == 1
        for not_null_col in ("kind", "slug", "slug_casefold", "source", "created_at", "updated_at"):
            assert cols[not_null_col]["notnull"] == 1, f"{not_null_col} should be NOT NULL"
        # The composite FK target: an explicit UNIQUE (id, kind) must exist (the PK
        # on id alone is NOT sufficient — §7 / scout composite-FK gotcha).
        assert ("id", "kind") in _unique_index_colsets(conn, "nodes")
    finally:
        conn.close()


def test_v1_schema_child_table_columns_and_keys(tmp_path) -> None:
    """The five child tables carry the §8.2 columns, NOT NULL flags, and composite/PK keys."""
    conn = _v1_conn(tmp_path)
    try:
        scopes = _columns(conn, "node_scopes")
        assert set(scopes) == {"node_id", "scope"}
        assert scopes["node_id"]["pk"] and scopes["scope"]["pk"]  # composite PK

        edges = _columns(conn, "edges")
        assert set(edges) == {
            "source_id", "source_kind", "target_id", "target_kind",
            "edge_type", "created_at",
        }
        for col in edges.values():
            assert col["notnull"] == 1  # every edges column is NOT NULL

        transcripts = _columns(conn, "transcripts")
        assert set(transcripts) == {"node_id", "transcript_text"}
        assert transcripts["node_id"]["pk"] == 1
        assert transcripts["transcript_text"]["notnull"] == 1

        signals = _columns(conn, "signals")
        assert set(signals) == {
            "node_id", "signal_type", "source", "created_at", "payload_json",
        }
        # Composite PK over (node_id, signal_type, source); payload_json nullable.
        assert signals["node_id"]["pk"] and signals["signal_type"]["pk"] and signals["source"]["pk"]
        assert signals["payload_json"]["notnull"] == 0

        pending = _columns(conn, "pending_embeddings")
        assert set(pending) == {"node_id", "queued_at", "retry_count"}
        assert pending["node_id"]["pk"] == 1
        assert pending["queued_at"]["notnull"] == 1
        assert pending["retry_count"]["notnull"] == 1
    finally:
        conn.close()


def test_v1_schema_creates_the_three_indexes(tmp_path) -> None:
    """The three §8.2 indexes exist, and idx_nodes_slug_casefold is NON-unique (V1-D4/M3)."""
    conn = _v1_conn(tmp_path)
    try:
        node_idx = _index_unique_flags(conn, "nodes")
        assert "idx_nodes_slug_casefold" in node_idx
        assert node_idx["idx_nodes_slug_casefold"] == 0  # non-unique by design
        assert "idx_node_scopes_scope" in _index_unique_flags(conn, "node_scopes")
        assert "idx_edges_target" in _index_unique_flags(conn, "edges")
    finally:
        conn.close()


def test_v1_schema_idempotent_replay_is_noop(tmp_path) -> None:
    """Re-injecting step 1 over an at-head DB is a no-op: user_version stays 1, no error (MI-3)."""
    conn = _v1_conn(tmp_path)
    try:
        assert run_migrations(conn, steps=[(1, _v1_schema)]) == 1
        assert _user_version(conn) == 1
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            )
        }
        assert tables == set(_V1A_TABLES)
    finally:
        conn.close()


def test_v1_schema_declares_no_current_timestamp_default(tmp_path) -> None:
    """No V1a table declares DEFAULT CURRENT_TIMESTAMP — every timestamp is app-supplied (MI-10)."""
    conn = _v1_conn(tmp_path)
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table';"
        ).fetchall()
        names = {row[0] for row in rows}
        assert names == set(_V1A_TABLES)  # only the V1a tables are present
        for name, sql in rows:
            assert "CURRENT_TIMESTAMP" not in sql.upper(), f"{name} carries CURRENT_TIMESTAMP"
    finally:
        conn.close()


# --- Constraint firing (proves the structural guards + that foreign_keys=ON took) ---

def test_nodes_check_rejects_bad_kind(tmp_path) -> None:
    """A nodes row whose kind is outside the enum is rejected by the CHECK."""
    conn = _v1_conn(tmp_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            _insert_node(conn, "n1", kind="mechanism")
    finally:
        conn.close()


def test_nodes_check_rejects_bad_source(tmp_path) -> None:
    """A nodes row whose source is outside the enum is rejected by the CHECK."""
    conn = _v1_conn(tmp_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            _insert_node(conn, "n1", source="robot")
    finally:
        conn.close()


def test_nodes_accepts_every_valid_enum_value(tmp_path) -> None:
    """Both kinds and all three source values are accepted (the paired positive of the CHECK)."""
    conn = _v1_conn(tmp_path)
    try:
        _insert_node(conn, "d1", kind="decision", source="user")
        _insert_node(conn, "q1", kind="open_question", source="capture_llm")
        _insert_node(conn, "d2", kind="decision", source="import_llm")
        count = conn.execute("SELECT COUNT(*) FROM nodes;").fetchone()[0]
        assert count == 3
    finally:
        conn.close()


def test_edges_fk_rejects_orphan_target(tmp_path) -> None:
    """An edge whose target has no node row is rejected by the composite FK (proves FK resolves)."""
    conn = _v1_conn(tmp_path)
    try:
        _insert_node(conn, "d1", kind="decision")
        with pytest.raises(sqlite3.IntegrityError):
            _insert_edge(conn, "d1", "decision", "missing", "decision")
    finally:
        conn.close()


def test_edges_check_rejects_cross_kind(tmp_path) -> None:
    """A cross-kind edge (both FKs resolvable) is rejected by the kind-matrix CHECK."""
    conn = _v1_conn(tmp_path)
    try:
        _insert_node(conn, "d1", kind="decision")
        _insert_node(conn, "q1", kind="open_question")
        # Both FKs resolve (kinds match the real nodes), so only the
        # source_kind = target_kind CHECK can fail — isolating the kind matrix.
        with pytest.raises(sqlite3.IntegrityError):
            _insert_edge(conn, "d1", "decision", "q1", "open_question")
    finally:
        conn.close()


def test_edges_check_rejects_unwhitelisted_edge_type(tmp_path) -> None:
    """An edge_type outside {supersedes, corrects} is rejected by the CHECK (V1a two-edge whitelist)."""
    conn = _v1_conn(tmp_path)
    try:
        _insert_node(conn, "d1", kind="decision")
        _insert_node(conn, "d2", kind="decision")
        with pytest.raises(sqlite3.IntegrityError):
            _insert_edge(conn, "d1", "decision", "d2", "decision", edge_type="amends")
    finally:
        conn.close()


def test_edges_accepts_same_kind_decision_supersession(tmp_path) -> None:
    """A same-kind decision->decision supersedes edge is accepted (the composite FK resolves)."""
    conn = _v1_conn(tmp_path)
    try:
        _insert_node(conn, "d1", kind="decision")
        _insert_node(conn, "d2", kind="decision")
        _insert_edge(conn, "d2", "decision", "d1", "decision", edge_type="supersedes")
        _insert_edge(conn, "d2", "decision", "d1", "decision", edge_type="corrects")
        count = conn.execute("SELECT COUNT(*) FROM edges;").fetchone()[0]
        assert count == 2
    finally:
        conn.close()


def test_edges_accepts_open_question_to_open_question(tmp_path) -> None:
    """A same-kind OQ->OQ edge is accepted — V1-D18 Stage 1 relies on it."""
    conn = _v1_conn(tmp_path)
    try:
        _insert_node(conn, "q1", kind="open_question")
        _insert_node(conn, "q2", kind="open_question")
        _insert_edge(conn, "q2", "open_question", "q1", "open_question", edge_type="supersedes")
        count = conn.execute("SELECT COUNT(*) FROM edges;").fetchone()[0]
        assert count == 1
    finally:
        conn.close()


def test_signals_pk_rejects_duplicate_triple_but_allows_distinct_source(tmp_path) -> None:
    """signals' composite PK rejects an identical (node, type, source) triple; a distinct source is fine."""
    conn = _v1_conn(tmp_path)
    try:
        _insert_node(conn, "d1", kind="decision")
        conn.execute(
            "INSERT INTO signals (node_id, signal_type, source, created_at) VALUES (?, ?, ?, ?);",
            ("d1", "drifted", "sensor-a", _TS),
        )
        # Same node + type, different source → distinct PK, both rows coexist.
        conn.execute(
            "INSERT INTO signals (node_id, signal_type, source, created_at) VALUES (?, ?, ?, ?);",
            ("d1", "drifted", "sensor-b", _TS),
        )
        assert conn.execute("SELECT COUNT(*) FROM signals;").fetchone()[0] == 2
        # Identical triple → PK conflict.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO signals (node_id, signal_type, source, created_at) VALUES (?, ?, ?, ?);",
                ("d1", "drifted", "sensor-a", _TS),
            )
    finally:
        conn.close()


def test_signals_check_rejects_bad_signal_type(tmp_path) -> None:
    """A signal_type outside the §8.2 whitelist is rejected by the CHECK."""
    conn = _v1_conn(tmp_path)
    try:
        _insert_node(conn, "d1", kind="decision")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO signals (node_id, signal_type, source, created_at) VALUES (?, ?, ?, ?);",
                ("d1", "stale", "sensor-a", _TS),
            )
    finally:
        conn.close()


def test_pending_embeddings_pk_conflict_and_upsert_roundtrip(tmp_path) -> None:
    """A second pending row for one node conflicts; an ON CONFLICT UPSERT round-trips (pre-proves 5c)."""
    conn = _v1_conn(tmp_path)
    try:
        _insert_node(conn, "d1", kind="decision")
        conn.execute(
            "INSERT INTO pending_embeddings (node_id, queued_at) VALUES (?, ?);",
            ("d1", _TS),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO pending_embeddings (node_id, queued_at) VALUES (?, ?);",
                ("d1", _TS),
            )
        # The idempotent-enqueue path 5c will use: UPSERT bumps retry_count.
        conn.execute(
            "INSERT INTO pending_embeddings (node_id, queued_at) VALUES (?, ?) "
            "ON CONFLICT(node_id) DO UPDATE SET retry_count = retry_count + 1;",
            ("d1", _TS),
        )
        row = conn.execute(
            "SELECT retry_count FROM pending_embeddings WHERE node_id = ?;", ("d1",)
        ).fetchone()
        assert row[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM pending_embeddings;").fetchone()[0] == 1
    finally:
        conn.close()


def test_node_scopes_pk_dedupes_and_fk_rejects_orphan(tmp_path) -> None:
    """node_scopes' composite PK rejects a duplicate (node, scope); the FK rejects an orphan node_id."""
    conn = _v1_conn(tmp_path)
    try:
        _insert_node(conn, "d1", kind="decision")
        conn.execute("INSERT INTO node_scopes (node_id, scope) VALUES (?, ?);", ("d1", "core"))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO node_scopes (node_id, scope) VALUES (?, ?);", ("d1", "core"))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO node_scopes (node_id, scope) VALUES (?, ?);", ("ghost", "core"))
    finally:
        conn.close()


def test_transcripts_pk_one_per_node_and_fk_rejects_orphan(tmp_path) -> None:
    """transcripts holds one row per node (PK) and the FK rejects an orphan node_id."""
    conn = _v1_conn(tmp_path)
    try:
        _insert_node(conn, "d1", kind="decision")
        conn.execute(
            "INSERT INTO transcripts (node_id, transcript_text) VALUES (?, ?);",
            ("d1", "raw capture text"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO transcripts (node_id, transcript_text) VALUES (?, ?);",
                ("d1", "second text"),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO transcripts (node_id, transcript_text) VALUES (?, ?);",
                ("ghost", "orphan text"),
            )
    finally:
        conn.close()


# --- is_pre_v1a_schema: the pre-V1a detection guard (wired in 5a/6b) ---

def _proto_nodes(conn: sqlite3.Connection, strict: bool = False, with_casefold: bool = False) -> None:
    """Builds a prototype-shaped ``nodes`` table (user_version stays 0) for guard tests."""
    cols = "id TEXT PRIMARY KEY, slug TEXT, kind TEXT"
    if with_casefold:
        cols += ", slug_casefold TEXT"
    strict_clause = " STRICT" if strict else ""
    conn.execute(f"CREATE TABLE nodes ({cols}){strict_clause};")


def test_is_pre_v1a_true_for_real_prototype_schema(tmp_path) -> None:
    """The live prototype graph (real _init_db schema) is detected as pre-V1a → must route to cutover."""
    store = GraphStore(str(tmp_path / "proto.sqlite"))
    conn = store._get_connection()
    try:
        assert is_pre_v1a_schema(conn) is True
    finally:
        conn.close()


def test_is_pre_v1a_true_for_minimal_non_strict_nodes() -> None:
    """A non-STRICT nodes table with no slug_casefold at user_version 0 is pre-V1a."""
    conn = _fresh_conn()
    _proto_nodes(conn, strict=False, with_casefold=False)
    assert is_pre_v1a_schema(conn) is True


def test_is_pre_v1a_true_when_non_strict_even_with_casefold_column() -> None:
    """Non-STRICT alone flags pre-V1a — a stray slug_casefold column does not rescue it."""
    conn = _fresh_conn()
    _proto_nodes(conn, strict=False, with_casefold=True)
    assert is_pre_v1a_schema(conn) is True


def test_is_pre_v1a_true_when_strict_but_missing_slug_casefold() -> None:
    """A STRICT nodes lacking slug_casefold flags pre-V1a — the missing-column branch fires."""
    conn = _fresh_conn()
    _proto_nodes(conn, strict=True, with_casefold=False)
    assert is_pre_v1a_schema(conn) is True


def test_is_pre_v1a_false_for_fresh_empty_db() -> None:
    """A fresh empty DB (no nodes table) is NOT pre-V1a — empty is healthy, not broken."""
    conn = _fresh_conn()
    assert is_pre_v1a_schema(conn) is False


def test_is_pre_v1a_false_for_fresh_v1a_db(tmp_path) -> None:
    """A DB at the V1a schema head (user_version 1) is NOT pre-V1a."""
    conn = _v1_conn(tmp_path)
    try:
        assert is_pre_v1a_schema(conn) is False
    finally:
        conn.close()


def test_is_pre_v1a_false_at_advanced_version_regardless_of_shape() -> None:
    """The user_version >= 1 gate short-circuits: a prototype-shaped DB at version 1 is NOT pre-V1a."""
    conn = _fresh_conn()
    _proto_nodes(conn, strict=False, with_casefold=False)
    conn.execute("PRAGMA user_version = 1;")
    assert is_pre_v1a_schema(conn) is False
