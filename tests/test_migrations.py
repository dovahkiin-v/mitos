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
    _pending_head,
    _v1_schema,
    _v1b_schema,
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


def test_registry_has_v1a_and_v1b_schema_steps() -> None:
    """The live registry carries both ladder rungs: V1a step 1 and V1b step 2.

    Phase 5a appended ``(1, _v1_schema)`` (entry-001 flip); Phase 1b appends
    ``(2, _v1b_schema)`` (the ``mechanisms`` DDL + widened ``edges`` CHECK) via
    ``.append`` — never a rebind, so ``run_migrations``'s def-time-bound default arg
    sees both on the live boot. This retires the ``len == 1`` dormancy tripwire 1a
    left for 1b.
    """
    assert (1, _v1_schema) in MIGRATION_STEPS
    assert (2, _v1b_schema) in MIGRATION_STEPS
    assert len(MIGRATION_STEPS) == 2  # the V1a rung + the V1b rung


def test_empty_ladder_is_noop() -> None:
    """An explicitly-empty ladder on a fresh DB leaves user_version=0, creates nothing.

    Pass ``steps=[]`` explicitly: the module default ``MIGRATION_STEPS`` is no
    longer empty (5a registered step 1), so the empty-ladder-noop guarantee is
    proven with an injected empty registry, not the live default.
    """
    conn = _fresh_conn()
    head = run_migrations(conn, steps=[])
    assert head == 0
    assert _user_version(conn) == 0
    # No tables of any kind were created.
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    ).fetchall()
    assert tables == []


def test_empty_ladder_replay_is_noop() -> None:
    """Replaying the empty ladder still changes nothing and raises nothing (MI-3).

    Uses an injected empty registry (``steps=[]``): the live default now carries
    step 1 (5a), so the empty-ladder replay guarantee is proven explicitly.
    """
    conn = _fresh_conn()
    assert run_migrations(conn, steps=[]) == 0
    assert run_migrations(conn, steps=[]) == 0
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
    """The real prototype graph (real _init_db schema) is detected as pre-V1a → route to cutover.

    Post-5a a normal ``GraphStore(...)`` boots the V1a schema (user_version 1), so
    building the *real* prototype DDL requires bypassing ``__init__`` and calling
    the retained ``_init_db`` method directly — the canonical prototype-schema
    definition Phase 7's cutover relies on (§16). This preserves the "real
    prototype DDL is detected → route-to-cutover" coverage; the sibling
    ``test_is_pre_v1a_true_for_minimal_non_strict_nodes`` covers the minimal shape.
    """
    store = GraphStore.__new__(GraphStore)  # bypass the V1a-booting __init__
    store.db_path = str(tmp_path / "proto.sqlite")
    store.read_only = False
    store._init_db()  # build the real prototype (pre-V1a) schema in place
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


# --- Phase 1b: mechanisms DDL + edges kind-CHECK widening (ladder step 2) -------
#
# Step 2 is authored + proven here the same way step 1 was: injected through
# ``run_migrations(conn, steps=[(1, _v1_schema), (2, _v1b_schema)])`` against real
# on-disk temp DBs (the rebuild's INSERT...SELECT + ALTER RENAME need a file the FK
# pragma is live on — ``_v1_conn`` already opens through ``open_connection``). The
# real-registry snapshot-reversal half (the fault-injection T15 variant) lives in
# ``tests/test_migration_snapshot.py``; these are the DDL-level shape/widening/
# faithfulness/replay fixtures. Achieved state is always read back through the DB /
# introspection, never asserted against a literal we also wrote; no ``user_version``
# literal (PLANNING_NOTES — bind to instance state).

# The full injected ladder to head 2 (mirrors the live ``MIGRATION_STEPS``).
_V1B_STEPS = [(1, _v1_schema), (2, _v1b_schema)]


def _v1b_conn(tmp_path, name: str = "v1b.sqlite") -> sqlite3.Connection:
    """Opens an FK-on file DB laddered through the injected registry to head 2 (V1b)."""
    conn = open_connection(str(tmp_path / name))
    run_migrations(conn, steps=_V1B_STEPS)
    return conn


def test_full_ladder_boots_fresh_db_to_head_2_via_live_registry(tmp_path) -> None:
    """A fresh DB run through the LIVE registry ladders to head 2 with both schemas.

    PLANNING_NOTES:57 — the from-scratch boot (not just step-2-on-a-seeded-v1) is what
    catches a "step 2 assumes ``edges`` exists but nothing created it" bug: the live
    ``MIGRATION_STEPS`` applies step 1 then step 2 from ``user_version`` 0. The head is
    read dynamically (``_pending_head``), never a literal.
    """
    conn = open_connection(str(tmp_path / "fresh.sqlite"))
    try:
        head = run_migrations(conn, MIGRATION_STEPS)
        assert head == _pending_head(MIGRATION_STEPS)
        assert _user_version(conn) == _pending_head(MIGRATION_STEPS)
        # All six V1a tables + the new mechanisms registry are present.
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            )
        }
        assert tables == set(_V1A_TABLES) | {"mechanisms"}
        # mechanisms laddered in STRICT with its canonical_name PK (full shape +
        # prototype-negative assertions live in the dedicated mechanisms tests below).
        assert _is_strict(conn, "mechanisms")
        assert _columns(conn, "mechanisms")["canonical_name"]["pk"] == 1
        # idx_edges_target survived the rebuild (DROP TABLE took it; the step recreates it).
        assert "idx_edges_target" in _index_unique_flags(conn, "edges")
        # The widened CHECK is in effect: a cross-kind cites edge inserts (v1 forbade it).
        _insert_node(conn, "d1", kind="decision")
        _insert_node(conn, "q1", kind="open_question")
        _insert_edge(conn, "d1", "decision", "q1", "open_question", edge_type="cites")
        assert conn.execute("SELECT COUNT(*) FROM edges;").fetchone()[0] == 1
    finally:
        conn.close()


# --- mechanisms DDL constraints (§9 #5) ---

def test_v1b_mechanisms_table_shape_and_no_prototype_artifacts(tmp_path) -> None:
    """mechanisms has the §8.2 columns/PK/STRICT and NONE of the dead prototype's shape."""
    conn = _v1b_conn(tmp_path)
    try:
        assert _is_strict(conn, "mechanisms")
        cols = _columns(conn, "mechanisms")
        assert set(cols) == {"canonical_name", "authored_name", "source", "created_at"}
        assert cols["canonical_name"]["pk"] == 1
        for col in ("canonical_name", "authored_name", "source", "created_at"):
            assert cols[col]["notnull"] == 1, f"{col} should be NOT NULL"
        # Negative assertions against the dead ``_init_db`` prototype trap (vision §2/§7):
        # no ``kind`` column, no ``name``-PK column, no ``node_mechanisms`` junction.
        assert "kind" not in cols
        assert "name" not in cols
        assert not _table_exists(conn, "node_mechanisms")
        # No CURRENT_TIMESTAMP default — created_at is application-supplied (MI-10).
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='mechanisms';"
        ).fetchone()[0]
        assert "CURRENT_TIMESTAMP" not in sql.upper()
    finally:
        conn.close()


def test_v1b_mechanisms_check_rejects_bad_source(tmp_path) -> None:
    """A mechanisms row whose source is outside the enum is rejected by the DDL CHECK.

    Proves the CHECK shipped; the writer-tied all-three-values parameterized coverage
    is Phase 5a's (§6.2 Lesson 13).
    """
    conn = _v1b_conn(tmp_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO mechanisms (canonical_name, authored_name, source, created_at) "
                "VALUES (?, ?, ?, ?);",
                ("lint:wal", "LINT:wal", "robot", _TS),
            )
    finally:
        conn.close()


def test_v1b_mechanisms_accepts_valid_row_and_pk_dedupes(tmp_path) -> None:
    """A valid mechanisms row commits; a duplicate canonical_name conflicts on the PK."""
    conn = _v1b_conn(tmp_path)
    try:
        conn.execute(
            "INSERT INTO mechanisms (canonical_name, authored_name, source, created_at) "
            "VALUES (?, ?, ?, ?);",
            ("lint:wal", "LINT:wal", "user", _TS),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO mechanisms (canonical_name, authored_name, source, created_at) "
                "VALUES (?, ?, ?, ?);",
                ("lint:wal", "lint-wal", "capture_llm", _TS),
            )
        assert conn.execute("SELECT COUNT(*) FROM mechanisms;").fetchone()[0] == 1
    finally:
        conn.close()


# --- Widening is effective: the rebuild's accept/reject at the raw DDL level (§9 #3) ---
# 1b verifies the DDL CHECK; Phase 2a owns the store-layer kind_constraint_violation
# mapping over it (DoD #8b).

def test_v1b_edges_accept_the_four_new_same_kind_types(tmp_path) -> None:
    """The four new same-kind edge types now insert (decision→decision and OQ→OQ)."""
    conn = _v1b_conn(tmp_path)
    try:
        _insert_node(conn, "d1", kind="decision")
        _insert_node(conn, "d2", kind="decision")
        for edge_type in ("amends", "narrows", "depends_on", "contradicts"):
            _insert_edge(conn, "d1", "decision", "d2", "decision", edge_type=edge_type)
        # OQ→OQ same-kind also holds for the non-cross types.
        _insert_node(conn, "q1", kind="open_question")
        _insert_node(conn, "q2", kind="open_question")
        _insert_edge(conn, "q1", "open_question", "q2", "open_question", edge_type="narrows")
        assert conn.execute("SELECT COUNT(*) FROM edges;").fetchone()[0] == 5
    finally:
        conn.close()


def test_v1b_edges_accept_the_three_cross_kind_shapes(tmp_path) -> None:
    """The three cross-kind clauses insert: cites any→any, derives_from OQ→D, resolves D→OQ."""
    conn = _v1b_conn(tmp_path)
    try:
        _insert_node(conn, "d1", kind="decision")
        _insert_node(conn, "q1", kind="open_question")
        # cites is any→any: exercise a cross-kind direction the v1 CHECK forbade.
        _insert_edge(conn, "d1", "decision", "q1", "open_question", edge_type="cites")
        # derives_from is OQ→D only.
        _insert_edge(conn, "q1", "open_question", "d1", "decision", edge_type="derives_from")
        # resolves is D→OQ only.
        _insert_edge(conn, "d1", "decision", "q1", "open_question", edge_type="resolves")
        assert conn.execute("SELECT COUNT(*) FROM edges;").fetchone()[0] == 3
    finally:
        conn.close()


def test_v1b_edges_reject_bogus_type_and_kind_violations(tmp_path) -> None:
    """An unknown edge_type and kind-violating cross-kind shapes are still rejected."""
    conn = _v1b_conn(tmp_path)
    try:
        _insert_node(conn, "d1", kind="decision")
        _insert_node(conn, "d2", kind="decision")
        _insert_node(conn, "q1", kind="open_question")
        # Unknown edge_type — no clause admits it.
        with pytest.raises(sqlite3.IntegrityError):
            _insert_edge(conn, "d1", "decision", "d2", "decision", edge_type="bogus")
        # resolves is D→OQ only: a D→D resolves is kind-violating (the canonical #8b case).
        with pytest.raises(sqlite3.IntegrityError):
            _insert_edge(conn, "d1", "decision", "d2", "decision", edge_type="resolves")
        # derives_from is OQ→D only: the reverse D→OQ is rejected.
        with pytest.raises(sqlite3.IntegrityError):
            _insert_edge(conn, "d1", "decision", "q1", "open_question", edge_type="derives_from")
        # A same-kind-only type across kinds (amends D→OQ) is rejected.
        with pytest.raises(sqlite3.IntegrityError):
            _insert_edge(conn, "d1", "decision", "q1", "open_question", edge_type="amends")
        assert conn.execute("SELECT COUNT(*) FROM edges;").fetchone()[0] == 0
    finally:
        conn.close()


# --- Faithfulness — DoD #8a (R13): every edge row survives the rebuild (§9 #2) ---

def test_v1b_edges_rebuild_preserves_every_row_incl_archived_entry(tmp_path) -> None:
    """The widening carries forward every edge row — incl. one on an 'archived' entry (R13).

    Seed a v1 graph with a spread of kill-edges (the only v1-legal types), including an
    edge whose target is an otherwise-untouched node standing in for an already-archived
    entry NOT re-derivable from the buffer (DoD #8a). Apply step 2 and assert the full
    edge set survives byte-for-byte, the count is unchanged, and ``foreign_key_check`` is
    clean.
    """
    conn = _v1_conn(tmp_path)  # at head 1 — seed under the narrow v1 schema
    try:
        for node_id, kind in (
            ("d_archived", "decision"),
            ("d_live", "decision"),
            ("d_two", "decision"),
            ("q_root", "open_question"),
            ("q_child", "open_question"),
        ):
            _insert_node(conn, node_id, kind=kind)
        # A spread of v1 kill-edges, incl. one TARGETING the archived entry.
        _insert_edge(conn, "d_live", "decision", "d_archived", "decision", edge_type="supersedes")
        _insert_edge(conn, "d_two", "decision", "d_live", "decision", edge_type="corrects")
        _insert_edge(conn, "q_child", "open_question", "q_root", "open_question", edge_type="supersedes")

        select_all = (
            "SELECT source_id, source_kind, target_id, target_kind, edge_type, created_at "
            "FROM edges ORDER BY source_id, target_id, edge_type;"
        )
        # Materialize as plain tuples (open_connection sets row_factory=Row, which a
        # literal-tuple membership check below would not match).
        before_rows = [tuple(r) for r in conn.execute(select_all).fetchall()]

        # Apply step 2 (step 1 is gated — already at version 1).
        assert run_migrations(conn, steps=_V1B_STEPS) == 2

        after_rows = [tuple(r) for r in conn.execute(select_all).fetchall()]
        assert len(after_rows) == len(before_rows)  # no silent loss
        assert after_rows == before_rows  # every column value identical
        # The archived-entry edge specifically survived.
        assert (
            "d_live",
            "decision",
            "d_archived",
            "decision",
            "supersedes",
            _TS,
        ) in after_rows
        assert conn.execute("PRAGMA foreign_key_check;").fetchall() == []
    finally:
        conn.close()


# --- MI-3 replay-safety (§9 #4) ---

def test_v1b_full_ladder_replay_is_noop(tmp_path) -> None:
    """Re-running the full ladder over an at-head v2 DB changes nothing, raises nothing (MI-3)."""
    conn = _v1_conn(tmp_path)
    try:
        _insert_node(conn, "d1", kind="decision")
        _insert_node(conn, "d2", kind="decision")
        _insert_edge(conn, "d1", "decision", "d2", "decision", edge_type="supersedes")
        assert run_migrations(conn, steps=_V1B_STEPS) == 2

        edges_before = conn.execute("SELECT * FROM edges;").fetchall()
        mech_sql_before = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='mechanisms';"
        ).fetchone()[0]
        edges_sql_before = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='edges';"
        ).fetchone()[0]

        assert run_migrations(conn, steps=_V1B_STEPS) == 2  # raises nothing; gate holds
        assert _user_version(conn) == 2
        assert conn.execute("SELECT * FROM edges;").fetchall() == edges_before
        assert (
            conn.execute("SELECT sql FROM sqlite_master WHERE name='mechanisms';").fetchone()[0]
            == mech_sql_before
        )
        assert (
            conn.execute("SELECT sql FROM sqlite_master WHERE name='edges';").fetchone()[0]
            == edges_sql_before
        )
    finally:
        conn.close()


def test_v1b_schema_reinvoked_directly_is_noop_via_skip_guard(tmp_path) -> None:
    """Re-invoking ``_v1b_schema`` against an already-v2 DB is a true no-op (skip-guard, MI-3).

    The version gate skips step 2 on full-ladder replay, so a DIRECT re-invocation is
    what actually exercises the widened-CHECK skip-guard: the rebuild must NOT run again
    (no churn, no row loss, no error, no orphan ``edges_new``) when ``edges`` already
    carries the ``'cites'`` marker.
    """
    conn = _v1b_conn(tmp_path)
    try:
        _insert_node(conn, "d1", kind="decision")
        _insert_node(conn, "d2", kind="decision")
        _insert_edge(conn, "d1", "decision", "d2", "decision", edge_type="amends")
        edges_before = conn.execute("SELECT * FROM edges;").fetchall()
        edges_sql_before = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='edges';"
        ).fetchone()[0]

        # Direct re-invocation outside the runner — the skip-guard must short-circuit.
        _v1b_schema(conn)

        assert conn.execute("SELECT * FROM edges;").fetchall() == edges_before
        assert (
            conn.execute("SELECT sql FROM sqlite_master WHERE name='edges';").fetchone()[0]
            == edges_sql_before
        )
        assert not _table_exists(conn, "edges_new")  # no stray rebuild table
    finally:
        conn.close()
