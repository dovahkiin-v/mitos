"""Fault-injection tests for the pre-ladder DB snapshot harness (Phase 1a).

The first populated-schema migration (Phase 1b) rewrites a graph holding real,
irreplaceable rows — including edges on already-archived entries that are NOT
re-derivable from the buffer. The 1b faithfulness gates catch a *detectably*-lossy
migration; the snapshot reverses a *successful-but-buggy* one (a silent semantic
corruption no gate covered) and is retained on success as the fallback.

These tests prove the harness **now**, before 1b's real step 2 exists, by injecting
synthetic steps through the established ``run_migrations(conn, steps=...)`` idiom
(``tests/test_migrations.py``). All version/path values are read back through the DB
or derived from the live registry head — no ``user_version`` literal, no migration
head is hardcoded (PLANNING_NOTES: bind to instance state). Everything runs against
real on-disk temp DBs (``tmp_path``) — the snapshot path needs a file VACUUM INTO
can write and ``os.replace`` can rename, so the in-memory ``_fresh_conn`` idiom of
``test_migrations.py`` does not apply (Scout Brief discrepancy 2).

Run under ``./venv/bin/python -m pytest`` (PATTERNS — bare ``python`` lacks deps).
"""

import os

import pytest

from mitos.migrations import (
    MIGRATION_STEPS,
    _pending_head,
    _snapshot_path,
    _v1_schema,
    restore_from_snapshot,
    run_migrations,
    take_pre_ladder_snapshot,
)
from mitos.store import _boot_migrations, open_connection

# A representative app-supplied UTC ISO-8601 µs timestamp (the V1a DDL carries no
# CURRENT_TIMESTAMP default; every writer supplies one — MI-10).
_TS = "2026-06-22T00:00:00.000000+00:00"


# --- read-back helpers (never assert a literal we also wrote) ------------------


def _user_version(conn) -> int:
    """Reads ``PRAGMA user_version`` back through the DB."""
    return conn.execute("PRAGMA user_version;").fetchone()[0]


def _table_exists(conn, name: str) -> bool:
    """Reports whether a table is present in the schema."""
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (name,)
        ).fetchone()
        is not None
    )


def _insert_node(conn, node_id: str, kind: str = "decision", source: str = "user") -> None:
    """Inserts a fully-populated ``nodes`` row (every NOT NULL column supplied, §8)."""
    conn.execute(
        "INSERT INTO nodes "
        "(id, kind, slug, slug_casefold, source, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?);",
        (node_id, kind, node_id, node_id.casefold(), source, _TS, _TS),
    )


def _insert_edge(conn, source_id, target_id, edge_type="supersedes", kind="decision") -> None:
    """Inserts an ``edges`` row (same-kind, V1a-whitelisted by default)."""
    conn.execute(
        "INSERT INTO edges "
        "(source_id, source_kind, target_id, target_kind, edge_type, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?);",
        (source_id, kind, target_id, kind, edge_type, _TS),
    )


def _edge_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM edges;").fetchone()[0]


def _seed_v1_db(db_path: str) -> None:
    """Creates a V1a-schema DB at ``user_version`` 1 with seeded rows.

    Two decision nodes and a supersedes edge between them. ``d_archived`` stands in
    for an already-archived entry whose edge is NOT re-derivable from the buffer —
    the exact row a faulty 1b rebuild could silently drop (DoD #8a / R13). The
    connection is closed so the file is settled before ``_boot_migrations`` opens
    its own.
    """
    conn = open_connection(db_path)
    try:
        # Apply the V1a schema as step 1 purely to seed rows (the snapshot tests key
        # their head off the injected synthetic steps, never this).
        run_migrations(conn, steps=[(1, _v1_schema)])
        _insert_node(conn, "d_archived", kind="decision")
        _insert_node(conn, "d_live", kind="decision")
        _insert_edge(conn, "d_live", "d_archived", edge_type="supersedes")
        conn.commit()
    finally:
        conn.close()


def _table_creating_step(table: str):
    """A synthetic step that creates ``table`` (visible, commits via the runner)."""

    def step(conn) -> None:
        conn.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY);")

    return step


def _raising_step(conn) -> None:
    """A synthetic step that does DDL then raises (the post-commit-gate-fail model).

    Mirrors ``test_migrations.doomed_step``: issues a CREATE then raises, so the
    runner rolls *this* step back while a prior committed step's change persists —
    forcing the in-place DB to diverge from the snapshot before restore.
    """
    conn.execute("CREATE TABLE half_migrated (id INTEGER PRIMARY KEY);")
    raise RuntimeError("boom: faithfulness gate failed after the rebuild committed")


# --- T15-fault: restore-on-failure ---------------------------------------------


def test_fault_injection_restores_pre_migration_graph(tmp_path) -> None:
    """A ladder failure leaves the live graph restored to its pre-migration state.

    Inject ``[(2, sentinel), (3, raising)]``: step 2 commits a visible sentinel
    (advancing to 2), step 3 raises. The sentinel-then-raise across two steps forces
    the in-place DB to *diverge* from the snapshot before restore, so a no-op restore
    cannot pass. After ``_boot_migrations`` re-raises, the live DB must be back at the
    pre-migration version, the sentinel absent, every seeded row (incl. the
    archived-entry edge) intact, the snapshot consumed, and no orphan sidecars left.
    """
    db_path = str(tmp_path / ".mitos" / "graph.sqlite")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    _seed_v1_db(db_path)

    # Read the pre-migration version back, then key the expected snapshot path off
    # it (no literal) — the version migrated *from*.
    conn = open_connection(db_path)
    pre_version = _user_version(conn)
    conn.close()
    snapshot_path = _snapshot_path(db_path, pre_version)

    steps = [
        (pre_version + 1, _table_creating_step("sentinel_marker")),
        (pre_version + 2, _raising_step),
    ]
    with pytest.raises(RuntimeError, match="boom"):
        _boot_migrations(db_path, steps)

    # No half-migrated DB / no orphan sidecars — assert BEFORE reopening (a read
    # connection re-creates a -wal).
    assert not os.path.exists(db_path + "-wal")
    assert not os.path.exists(db_path + "-shm")
    # The snapshot was CONSUMED by the restore (it became the live DB).
    assert not os.path.exists(snapshot_path)

    conn = open_connection(db_path)
    try:
        assert _user_version(conn) == pre_version  # rolled all the way back
        assert not _table_exists(conn, "sentinel_marker")  # step 2's commit undone
        assert not _table_exists(conn, "half_migrated")  # step 3 never committed
        # Every seeded row survived, including the archived-entry edge (R13).
        assert _edge_count(conn) == 1
        node_ids = {r[0] for r in conn.execute("SELECT id FROM nodes;")}
        assert node_ids == {"d_archived", "d_live"}
        edge = conn.execute(
            "SELECT source_id, target_id, edge_type FROM edges;"
        ).fetchone()
        assert tuple(edge) == ("d_live", "d_archived", "supersedes")
    finally:
        conn.close()


# --- T15-success: retain-on-pass -----------------------------------------------


def test_successful_migration_retains_snapshot(tmp_path) -> None:
    """A successful migration leaves the snapshot on disk (retained, never dropped).

    Inject ``[(2, benign)]``: the ladder advances cleanly. The live DB ends at the
    new version with the benign change, AND the snapshot file remains on disk with
    content equal to the pre-migration graph (version unchanged, benign change
    absent) — the silent-corruption fallback (P5 Ironclad).
    """
    db_path = str(tmp_path / ".mitos" / "graph.sqlite")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    _seed_v1_db(db_path)

    conn = open_connection(db_path)
    pre_version = _user_version(conn)
    conn.close()
    snapshot_path = _snapshot_path(db_path, pre_version)

    steps = [(pre_version + 1, _table_creating_step("v1b_marker"))]
    _boot_migrations(db_path, steps)  # succeeds, no raise

    conn = open_connection(db_path)
    try:
        assert _user_version(conn) == pre_version + 1  # advanced
        assert _table_exists(conn, "v1b_marker")  # benign change committed live
    finally:
        conn.close()

    # The snapshot is RETAINED and faithful to the pre-migration graph.
    assert os.path.exists(snapshot_path)
    snap = open_connection(snapshot_path, read_only=True)
    try:
        assert _user_version(snap) == pre_version  # VACUUM INTO preserved it
        assert not _table_exists(snap, "v1b_marker")  # pre-migration content
        assert snap.execute("SELECT COUNT(*) FROM edges;").fetchone()[0] == 1
        node_ids = {r[0] for r in snap.execute("SELECT id FROM nodes;")}
        assert node_ids == {"d_archived", "d_live"}
    finally:
        snap.close()


# --- Precondition no-op: don't snapshot needlessly -----------------------------


def test_no_snapshot_when_db_already_at_head(tmp_path) -> None:
    """A DB already at the ladder head takes no snapshot (no pending step)."""
    db_path = str(tmp_path / ".mitos" / "graph.sqlite")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    _seed_v1_db(db_path)

    conn = open_connection(db_path)
    try:
        current = _user_version(conn)
        # Head == current: no pending step.
        steps = [(current, _table_creating_step("noop"))]
        result = take_pre_ladder_snapshot(conn, db_path, steps)
        assert result is None
    finally:
        conn.close()
    assert _snapshot_files(db_path) == []


def test_no_snapshot_on_fresh_empty_db(tmp_path) -> None:
    """A fresh DB (user_version 0) takes no snapshot — no rows to lose."""
    db_path = str(tmp_path / ".mitos" / "graph.sqlite")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = open_connection(db_path)  # untouched: user_version 0
    try:
        assert _user_version(conn) == 0
        steps = [(1, _table_creating_step("first"))]
        result = take_pre_ladder_snapshot(conn, db_path, steps)
        assert result is None
    finally:
        conn.close()
    assert _snapshot_files(db_path) == []


# --- Tripwire: the harness fires when a pending populated step exists -----------


def test_snapshot_taken_when_populated_db_has_pending_step(tmp_path) -> None:
    """Tripwire — a populated DB (>=1) with a pending synthetic step IS snapshotted.

    A regression that breaks the precondition (so the snapshot silently stops firing)
    is caught here, now, before 1b's real step 2 makes it live. The injected step
    keeps this honest while the live registry head stays at 1.
    """
    db_path = str(tmp_path / ".mitos" / "graph.sqlite")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    _seed_v1_db(db_path)

    conn = open_connection(db_path)
    try:
        current = _user_version(conn)
        steps = [(current + 1, _table_creating_step("pending"))]
        result = take_pre_ladder_snapshot(conn, db_path, steps)
        assert result == _snapshot_path(db_path, current)
        assert os.path.exists(result)
    finally:
        conn.close()


def test_real_boot_is_dormant_no_snapshot(tmp_path) -> None:
    """With the LIVE registry, a populated V1a graph takes no snapshot (dormant).

    Behavioural proof of the dormant-until-1b contract, keyed off the actual
    ``MIGRATION_STEPS`` head — no literal: a populated graph seeded at the live head
    has no pending step, so the precondition ``current < head`` cannot hold and the
    harness never fires on a real boot. 1b's ``.append((2, _v1b_schema))`` makes
    ``current < head`` true and flips it live (at which point this dormancy test is
    1b's to retire — the deliberate tripwire).
    """
    db_path = str(tmp_path / ".mitos" / "graph.sqlite")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    _seed_v1_db(db_path)  # seeded at user_version 1 == the live ladder head

    conn = open_connection(db_path)
    try:
        # The seeded version equals the live head, so there is no pending step.
        assert _user_version(conn) == _pending_head(MIGRATION_STEPS)
        result = take_pre_ladder_snapshot(conn, db_path, MIGRATION_STEPS)
        assert result is None  # dormant: nothing pending at the live head
    finally:
        conn.close()
    assert _snapshot_files(db_path) == []


# --- WAL-consistency: VACUUM INTO captures uncheckpointed frames ----------------


def test_snapshot_is_wal_consistent(tmp_path) -> None:
    """A snapshot taken while a -wal exists still captures the full committed state.

    Seed rows that leave uncheckpointed WAL frames (commit without an explicit
    checkpoint, connection kept open), take the snapshot while ``-wal`` is present and
    non-empty, then read the snapshot back and assert the WAL-resident row is there —
    proving ``VACUUM INTO`` captured the WAL'd state a bare ``cp`` of the main file
    alone would have lost (§9, Decision 1).
    """
    db_path = str(tmp_path / ".mitos" / "graph.sqlite")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    _seed_v1_db(db_path)

    # Re-open and write a marker that stays in the WAL: do NOT checkpoint, do NOT
    # close before the snapshot. open_connection runs WAL + synchronous=NORMAL.
    conn = open_connection(db_path)
    try:
        current = _user_version(conn)
        _insert_node(conn, "d_wal_resident", kind="decision")
        conn.commit()  # committed to the -wal, not yet checkpointed to the main file
        assert os.path.exists(db_path + "-wal")
        assert os.path.getsize(db_path + "-wal") > 0  # uncheckpointed frames present

        steps = [(current + 1, _table_creating_step("pending"))]
        snapshot_path = take_pre_ladder_snapshot(conn, db_path, steps)
        assert snapshot_path is not None
    finally:
        conn.close()

    snap = open_connection(snapshot_path, read_only=True)
    try:
        node_ids = {r[0] for r in snap.execute("SELECT id FROM nodes;")}
        # The WAL-resident row a bare main-file cp would have missed is present.
        assert "d_wal_resident" in node_ids
        assert node_ids == {"d_archived", "d_live", "d_wal_resident"}
    finally:
        snap.close()


# --- Idempotent retry: a stale snapshot from a crashed attempt self-heals -------


def test_stale_snapshot_is_discarded_before_take(tmp_path) -> None:
    """A leftover snapshot from a crashed prior attempt is discarded, not an error.

    ``VACUUM INTO`` errors if its target exists; pre-creating a stale file at the
    snapshot path must NOT break the take — the harness discards it and writes a
    fresh, valid snapshot (P5 idempotency).
    """
    db_path = str(tmp_path / ".mitos" / "graph.sqlite")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    _seed_v1_db(db_path)

    conn = open_connection(db_path)
    try:
        current = _user_version(conn)
        snapshot_path = _snapshot_path(db_path, current)
        # Simulate a crashed prior attempt: a stale (non-DB) file at the path, plus
        # an orphan sidecar.
        with open(snapshot_path, "w") as f:
            f.write("stale leftover from a crashed migration attempt")
        with open(snapshot_path + "-wal", "w") as f:
            f.write("orphan wal")

        steps = [(current + 1, _table_creating_step("pending"))]
        result = take_pre_ladder_snapshot(conn, db_path, steps)
        assert result == snapshot_path
    finally:
        conn.close()

    # The stale file was replaced by a valid snapshot (opens + reads back clean).
    assert not os.path.exists(snapshot_path + "-wal")  # orphan sidecar cleared
    snap = open_connection(snapshot_path, read_only=True)
    try:
        assert {r[0] for r in snap.execute("SELECT id FROM nodes;")} == {
            "d_archived",
            "d_live",
        }
    finally:
        snap.close()


# --- restore_from_snapshot: orphan-WAL guard + atomic replace -------------------


def test_restore_clears_orphan_wal_and_replaces(tmp_path) -> None:
    """``restore_from_snapshot`` clears the destination's orphan WAL then replaces.

    A stale ``-wal`` left beside the destination would be mis-applied to the restored
    file on next open (``SQLITE_CORRUPT``, the cutover's R11 lesson). Restore must
    clear it before the atomic ``os.replace`` and leave the snapshot consumed.
    """
    db_path = str(tmp_path / ".mitos" / "graph.sqlite")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    _seed_v1_db(db_path)

    conn = open_connection(db_path)
    try:
        current = _user_version(conn)
        # Take a real snapshot to restore from (inject a pending step so it fires).
        steps = [(current + 1, _table_creating_step("pending"))]
        snapshot_path = take_pre_ladder_snapshot(conn, db_path, steps)
        assert snapshot_path is not None
    finally:
        conn.close()

    # Plant a stale orphan -wal beside the destination (the hazard R11 guards).
    with open(db_path + "-wal", "w") as f:
        f.write("stale orphan wal that must not survive the restore")

    restore_from_snapshot(db_path, snapshot_path)

    assert not os.path.exists(db_path + "-wal")  # orphan cleared before replace
    assert not os.path.exists(snapshot_path)  # snapshot consumed (renamed in)
    # The restored DB opens clean (no SQLITE_CORRUPT) and carries the seeded rows.
    conn = open_connection(db_path)
    try:
        assert _user_version(conn) == current
        assert {r[0] for r in conn.execute("SELECT id FROM nodes;")} == {
            "d_archived",
            "d_live",
        }
    finally:
        conn.close()


def _snapshot_files(db_path: str) -> list:
    """Lists any ``…snapshot_v*`` siblings of ``db_path`` (for no-op assertions)."""
    import glob

    return sorted(glob.glob(db_path + ".snapshot_v*"))
