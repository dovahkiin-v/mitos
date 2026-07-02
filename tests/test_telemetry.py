"""Behavioral tests for the sibling telemetry store (Phase 1b).

Proves ``mitos/telemetry.py`` end-to-end against a **real temp SQLite** (no mocks —
this is a typed observability surface, so round-trips against a real file are the
point): the ladder boots green from scratch and replays to a no-op, a long/multi-
line judged axiom round-trips verbatim and uncapped, NULL semantics bind exactly,
the ``judgment_batches`` side-table attributes cost/latency exactly once (RF-2), a
partial write rolls the whole batch back, the writer is append-only, and the
sibling store survives a real ``mitos rebuild`` swap (T8).

Dynamic values only: ``PRAGMA user_version`` is read back through the DB, never a
bare literal beyond the single head expectation; metric fields are fixture inputs
echoed back and asserted for round-trip equality, not a computed count.
"""

import os
import sqlite3

import pytest

from mitos.config import MitosConfig
from mitos.cutover import default_aside_db_path, perform_swap, rebuild_and_gate
from mitos.errors import DatabaseError
from mitos.store import GraphStore, open_connection
from mitos.telemetry import (
    _CONFLICT_CHECKS_COLUMNS,
    _JUDGMENT_BATCHES_COLUMNS,
    ConflictCheckRow,
    JudgmentBatch,
    TelemetryStore,
)

CREATED_AT = "2026-07-03T01:46:29.359834+00:00"
SENTINEL = "<!-- BEGIN ENTRIES — newest first -->"


# --- builders + read helpers ---------------------------------------------------


def _row(**overrides) -> ConflictCheckRow:
    """Builds a ConflictCheckRow with sensible defaults; override any field."""
    base = dict(
        batch_id="batch-1",
        sync_run_id="sync-1",
        judged_axiom="Use SQLite for the graph store.",
        proposal_rejected_paths="Considered Postgres; rejected for portability.",
        proposal_scope="storage, persistence",
        proposed_hash_if_any="deadbeef",
        candidate_slug="graph-store-is-sqlite",
        candidate_hash="cafef00d",
        candidate_rejected_paths="Rejected a document store.",
        candidate_scope="storage",
        tenable=True,
        confidence=0.91,
        surfaced=True,
        candidate_source="embedding_topk",
        model_alias="CLAUDE_SONNET",
        prompt_version="conflict-judge-v1",
        mitos_version="0.5.21",
        rationale="Both axioms fix the storage engine and disagree.",
    )
    base.update(overrides)
    return ConflictCheckRow(**base)


def _batch(**overrides) -> JudgmentBatch:
    """Builds a JudgmentBatch with sensible defaults; override any field."""
    base = dict(
        batch_id="batch-1",
        token_input=1200,
        token_output=340,
        token_cache_read=0,
        token_cache_creation=800,
        elapsed_ms=2500,
    )
    base.update(overrides)
    return JudgmentBatch(**base)


def _telemetry_path(tmp_path) -> str:
    """A telemetry path whose parent ``.mitos/`` does NOT yet exist.

    Constructing the store here also exercises the makedirs-before-open guard —
    SQLite raises "unable to open database file" on a missing directory.
    """
    return str(tmp_path / ".mitos" / "telemetry.sqlite")


def _open(path: str) -> sqlite3.Connection:
    """Opens a read-back connection through the MI-8 chokepoint (row_factory=Row)."""
    return open_connection(path)


def _user_version(conn: sqlite3.Connection) -> int:
    """Reads ``PRAGMA user_version`` back through the DB (never a bare literal)."""
    return conn.execute("PRAGMA user_version;").fetchone()[0]


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Reports whether a table is present in the schema."""
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;", (name,)
        ).fetchone()
        is not None
    )


def _schema_snapshot(conn: sqlite3.Connection):
    """Returns the full stored DDL as comparable tuples (byte-identity check)."""
    return [
        tuple(r)
        for r in conn.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name;"
        ).fetchall()
    ]


def _count(conn: sqlite3.Connection, table: str, batch_id: str = None) -> int:
    """Counts rows in ``table``, optionally scoped to a ``batch_id``."""
    if batch_id is None:
        return conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]
    return conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE batch_id=?;", (batch_id,)
    ).fetchone()[0]


# --- lockstep: to_params order matches the INSERT column tuples ----------------


def test_to_params_arity_matches_column_tuples() -> None:
    """``to_params`` emits exactly one value per INSERT column (order-lockstep pin)."""
    assert len(_row().to_params(CREATED_AT)) == len(_CONFLICT_CHECKS_COLUMNS)
    assert len(_batch().to_params()) == len(_JUDGMENT_BATCHES_COLUMNS)


# --- criterion 1: fresh boot ---------------------------------------------------


def test_fresh_boot_creates_schema(tmp_path) -> None:
    """Constructing on an absent path creates the file, both tables, user_version=1.

    Also proves the makedirs-before-open guard: the parent ``.mitos/`` does not
    exist before construction.
    """
    path = _telemetry_path(tmp_path)
    assert not os.path.exists(path)

    TelemetryStore(path)

    assert os.path.exists(path)
    conn = _open(path)
    try:
        assert _table_exists(conn, "conflict_checks")
        assert _table_exists(conn, "judgment_batches")
        assert _user_version(conn) == 1
    finally:
        conn.close()


# --- criterion 2: replay idempotency x2 ----------------------------------------


def test_replay_idempotency_is_noop(tmp_path) -> None:
    """Booting the ladder a second and third time changes nothing and raises nothing."""
    path = _telemetry_path(tmp_path)
    TelemetryStore(path)
    conn = _open(path)
    try:
        first_schema = _schema_snapshot(conn)
        assert _user_version(conn) == 1
    finally:
        conn.close()

    # Re-boot twice over an at-head file — the user_version gate skips the applied
    # rung; the IF NOT EXISTS DDL is the structural backstop.
    TelemetryStore(path)
    TelemetryStore(path)

    conn = _open(path)
    try:
        assert _user_version(conn) == 1
        assert _schema_snapshot(conn) == first_schema
    finally:
        conn.close()


# --- criterion 3: verbatim uncapped round-trip ---------------------------------


def test_verbatim_uncapped_roundtrip(tmp_path) -> None:
    """A long, multi-line axiom/rationale/rejected_paths reads back byte-identical."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)

    long_axiom = (
        "Line one of the judged axiom.\n" * 500
        + "  trailing détail — ß, Straße, and a Greek final sigma ς.\n"
    )
    multiline_rationale = "First reason.\n\nSecond reason.\n- bullet a\n- bullet b\n"
    multiline_rejected = "Rejected path A.\nRejected path B.\n\tindented C\n"

    store.record_judged_batch(
        _batch(),
        [
            _row(
                judged_axiom=long_axiom,
                rationale=multiline_rationale,
                candidate_rejected_paths=multiline_rejected,
            )
        ],
        CREATED_AT,
    )

    conn = _open(path)
    try:
        got = conn.execute(
            "SELECT judged_axiom, rationale, candidate_rejected_paths, created_at "
            "FROM conflict_checks;"
        ).fetchone()
    finally:
        conn.close()
    assert got["judged_axiom"] == long_axiom
    assert got["rationale"] == multiline_rationale
    assert got["candidate_rejected_paths"] == multiline_rejected
    assert got["created_at"] == CREATED_AT


# --- criterion 4: NULL semantics ------------------------------------------------


def test_null_semantics_roundtrip(tmp_path) -> None:
    """Nullable fields bind None; present nullables bind their value, exactly."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)

    store.record_judged_batch(
        _batch(),
        [
            _row(
                proposal_scope=None,
                candidate_scope=None,
                proposal_rejected_paths=None,
                sync_run_id="sync-9",
                proposed_hash_if_any="hash-9",
            )
        ],
        CREATED_AT,
    )

    conn = _open(path)
    try:
        got = conn.execute(
            "SELECT proposal_scope, candidate_scope, proposal_rejected_paths, "
            "sync_run_id, proposed_hash_if_any FROM conflict_checks;"
        ).fetchone()
    finally:
        conn.close()
    assert got["proposal_scope"] is None
    assert got["candidate_scope"] is None
    assert got["proposal_rejected_paths"] is None
    assert got["sync_run_id"] == "sync-9"
    assert got["proposed_hash_if_any"] == "hash-9"


def test_not_null_field_fails_loudly_and_rolls_back(tmp_path) -> None:
    """Omitting the M5-required ``candidate_rejected_paths`` raises + lands nothing."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)

    # candidate_rejected_paths is NOT NULL (M5 requires it on every decision).
    bad = _row(candidate_rejected_paths=None)
    with pytest.raises(DatabaseError):
        store.record_judged_batch(_batch(), [bad], CREATED_AT)

    conn = _open(path)
    try:
        assert _count(conn, "conflict_checks") == 0
        assert _count(conn, "judgment_batches") == 0
    finally:
        conn.close()


def test_check_constraint_rejects_out_of_range_confidence(tmp_path) -> None:
    """The belt-and-suspenders CHECK rejects a confidence outside [0, 1]."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)

    with pytest.raises(DatabaseError):
        store.record_judged_batch(_batch(), [_row(confidence=2.0)], CREATED_AT)

    conn = _open(path)
    try:
        assert _count(conn, "conflict_checks") == 0
        assert _count(conn, "judgment_batches") == 0
    finally:
        conn.close()


# --- criterion 5: batch attribution (RF-2) -------------------------------------


def test_batch_attribution_exactly_once(tmp_path) -> None:
    """N candidate rows share one batch; metrics attribute exactly once (no N x)."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)

    n = 4
    batch = _batch(batch_id="b-rf2", token_input=1500)
    rows = [
        _row(batch_id="b-rf2", candidate_slug=f"cand-{i}", candidate_hash=f"h{i}")
        for i in range(n)
    ]
    store.record_judged_batch(batch, rows, CREATED_AT)

    conn = _open(path)
    try:
        assert _count(conn, "conflict_checks", "b-rf2") == n
        assert _count(conn, "judgment_batches", "b-rf2") == 1
        # A naive SUM over the side-table returns TRUE spend, not n x it.
        total = conn.execute(
            "SELECT SUM(token_input) FROM judgment_batches;"
        ).fetchone()[0]
        assert total == 1500
    finally:
        conn.close()


# --- criterion 6: batch atomicity ----------------------------------------------


def test_batch_atomicity_rolls_back_whole_batch(tmp_path) -> None:
    """A bad row late in a batch rolls back the batch row AND every earlier row."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)

    good = [
        _row(candidate_slug=f"ok-{i}", candidate_hash=f"h{i}") for i in range(3)
    ]
    bad = _row(candidate_slug="bad", candidate_hash="hbad", candidate_rejected_paths=None)

    with pytest.raises(DatabaseError):
        store.record_judged_batch(_batch(), good + [bad], CREATED_AT)

    conn = _open(path)
    try:
        assert _count(conn, "conflict_checks") == 0
        assert _count(conn, "judgment_batches") == 0
    finally:
        conn.close()


# --- criterion 7: append-only ---------------------------------------------------


def test_append_only_second_batch_preserves_first(tmp_path) -> None:
    """A second batch adds its rows without touching the first batch's rows."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)

    store.record_judged_batch(_batch(batch_id="b1"), [_row(batch_id="b1")], CREATED_AT)
    store.record_judged_batch(
        _batch(batch_id="b2"),
        [_row(batch_id="b2", candidate_slug="other", candidate_hash="h-other")],
        "2026-07-04T00:00:00+00:00",
    )

    conn = _open(path)
    try:
        assert _count(conn, "conflict_checks") == 2
        assert _count(conn, "judgment_batches") == 2
        # The first batch's row is byte-for-byte as written (untouched by batch 2).
        first = conn.execute(
            "SELECT created_at, candidate_slug FROM conflict_checks WHERE batch_id=?;",
            ("b1",),
        ).fetchone()
        assert first["created_at"] == CREATED_AT
        assert first["candidate_slug"] == "graph-store-is-sqlite"
    finally:
        conn.close()


# --- bool -> INTEGER storage (STRICT has no BOOLEAN affinity) -------------------


def test_bools_stored_as_integers(tmp_path) -> None:
    """``tenable``/``surfaced`` land as 0/1 INTEGERs under STRICT typing."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)

    store.record_judged_batch(
        _batch(), [_row(tenable=True, surfaced=False)], CREATED_AT
    )

    conn = _open(path)
    try:
        got = conn.execute(
            "SELECT tenable, surfaced FROM conflict_checks;"
        ).fetchone()
    finally:
        conn.close()
    assert got["tenable"] == 1
    assert got["surfaced"] == 0
    assert isinstance(got["tenable"], int)
    assert isinstance(got["surfaced"], int)


# --- criterion 8: rebuild-survival (T8 unit) -----------------------------------


def _decision_block(slug: str, decided: str, rejected: str = "n/a") -> str:
    """One decision entry block in spec order (mirrors test_cutover's ``_decision``)."""
    return f"### {slug}\n\n**Decided:** {decided}\n**Rejected:** {rejected}"


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def test_telemetry_survives_rebuild_swap(tmp_path) -> None:
    """A populated ``telemetry.sqlite`` + its rows survive a real ``mitos rebuild`` swap.

    The T8 guarantee: telemetry sits OUTSIDE the graph swap/backup/sidecar set, so
    ``rebuild_and_gate`` + ``perform_swap`` (which touch only ``graph.sqlite``-derived
    paths) leave the sibling store untouched.
    """
    config = MitosConfig(str(tmp_path))

    # A real (fresh) V1a graph at config.db_path so the completeness gate has an old
    # graph to read, plus a one-entry corpus for the rebuild to replay.
    GraphStore(config.db_path)
    _write(
        config.decisions_file,
        SENTINEL + "\n\n" + _decision_block("alpha", "Alpha axiom.") + "\n",
    )

    # Populate the telemetry sibling BEFORE the swap.
    telemetry = TelemetryStore(config.telemetry_path)
    telemetry.record_judged_batch(
        _batch(batch_id="survivor"),
        [_row(batch_id="survivor", judged_axiom="A judgment that must outlive rebuild.")],
        CREATED_AT,
    )

    # Drive the REAL swap.
    aside = default_aside_db_path(config)
    rebuild_and_gate(config, aside_db_path=aside)
    perform_swap(config, aside, timestamp="20260618-120000")

    # The sibling file and its row are untouched.
    assert os.path.exists(config.telemetry_path)
    conn = _open(config.telemetry_path)
    try:
        assert _count(conn, "conflict_checks", "survivor") == 1
        assert _count(conn, "judgment_batches", "survivor") == 1
        got = conn.execute(
            "SELECT judged_axiom FROM conflict_checks WHERE batch_id=?;", ("survivor",)
        ).fetchone()
        assert got["judged_axiom"] == "A judgment that must outlive rebuild."
    finally:
        conn.close()
