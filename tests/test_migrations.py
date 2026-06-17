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
from mitos.migrations import MIGRATION_STEPS, run_migrations


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
