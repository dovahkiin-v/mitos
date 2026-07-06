"""Behavioral tests for the sibling telemetry store (Phase 1b).

Proves ``mitos/telemetry.py`` end-to-end against a **real temp SQLite** (no mocks —
this is a typed observability surface, so round-trips against a real file are the
point): the ladder boots green from scratch and replays to a no-op, a long/multi-
line judged axiom round-trips verbatim and uncapped, NULL semantics bind exactly,
the ``judgment_batches`` side-table attributes cost/latency exactly once (RF-2), a
partial write rolls the whole batch back, the writer is append-only, and the
sibling store survives a real ``mitos rebuild`` swap (T8).

Dynamic values only: ``PRAGMA user_version`` expectations come from
``_pending_head(TELEMETRY_MIGRATION_STEPS)``, never a hardcoded rung literal (the
sliced-ladder ``== 1`` in the upgrade fixture is the one deliberate exception —
it pins the *pre-upgrade* rung, not the head); metric fields are fixture inputs
echoed back and asserted for round-trip equality, not a computed count.
"""

import dataclasses
import inspect
import os
import sqlite3

import pytest

import mitos.telemetry as telemetry_module
from mitos.config import MitosConfig
from mitos.cutover import default_aside_db_path, perform_swap, rebuild_and_gate
from mitos.errors import DatabaseError
from mitos.migrations import _pending_head, run_migrations
from mitos.store import BUSY_TIMEOUT_MS, GraphStore, open_connection
from mitos.telemetry import (
    _CONFLICT_CHECKS_COLUMNS,
    _JUDGMENT_BATCHES_COLUMNS,
    TELEMETRY_MIGRATION_STEPS,
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
        surface="sync",
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
        model_id="test-model-versioned-1",
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
    # Rung 2 widened both tuples — the INSERTs self-assemble from these.
    assert "surface" in _CONFLICT_CHECKS_COLUMNS
    assert "model_id" in _JUDGMENT_BATCHES_COLUMNS


def test_rung2_fields_have_no_defaults() -> None:
    """``surface``/``model_id`` are required at construction — the writers' fence.

    The schema's permanent ``DEFAULT 'sync'`` can never catch a writer omitting the
    column (an omitting INSERT silently gets ``'sync'``), so the enforcement is the
    defaultless dataclass field: omission is a ``TypeError`` at construction
    (CHK-D7). ``model_id=None`` stays a legal *explicit* value; silence is not.
    """
    row_fields = {f.name: f for f in dataclasses.fields(ConflictCheckRow)}
    surface = row_fields["surface"]
    assert surface.default is dataclasses.MISSING
    assert surface.default_factory is dataclasses.MISSING

    batch_fields = {f.name: f for f in dataclasses.fields(JudgmentBatch)}
    model_id = batch_fields["model_id"]
    assert model_id.default is dataclasses.MISSING
    assert model_id.default_factory is dataclasses.MISSING


# --- criterion 1: fresh boot ---------------------------------------------------


def test_fresh_boot_creates_schema(tmp_path) -> None:
    """Constructing on an absent path creates the file, all three tables, head version.

    Also proves the makedirs-before-open guard: the parent ``.mitos/`` does not
    exist before construction. The head expectation is dynamic
    (``_pending_head``) — never a hardcoded rung literal.
    """
    path = _telemetry_path(tmp_path)
    assert not os.path.exists(path)

    TelemetryStore(path)

    assert os.path.exists(path)
    conn = _open(path)
    try:
        assert _table_exists(conn, "conflict_checks")
        assert _table_exists(conn, "judgment_batches")
        assert _table_exists(conn, "check_runs")
        assert _user_version(conn) == _pending_head(TELEMETRY_MIGRATION_STEPS)
    finally:
        conn.close()


def _table_info(conn: sqlite3.Connection, table: str):
    """Returns ``PRAGMA table_info`` as (name, type, notnull, pk) tuples."""
    return [
        (r["name"], r["type"], r["notnull"], r["pk"])
        for r in conn.execute(f"PRAGMA table_info({table});").fetchall()
    ]


def test_check_runs_column_contract(tmp_path) -> None:
    """``check_runs`` matches the §3/W1 contract exactly — 2d's writer builds on this.

    Nullability is the semantic line and is pinned NOW (SQLite constraints are
    rebuild-only to change): NOT NULL = always known at run end; NULL = "value not
    computable this run", distinct from a genuine zero (CHK-D10).
    """
    path = _telemetry_path(tmp_path)
    TelemetryStore(path)
    conn = _open(path)
    try:
        got = _table_info(conn, "check_runs")
    finally:
        conn.close()
    assert got == [
        ("run_id", "TEXT", 1, 1),
        ("mode", "TEXT", 1, 0),
        ("started_at", "TEXT", 1, 0),
        ("ended_at", "TEXT", 1, 0),
        ("exit_code", "INTEGER", 1, 0),
        ("nodes_swept", "INTEGER", 1, 0),
        ("pairs_judged_fresh", "INTEGER", 1, 0),
        ("pairs_reused", "INTEGER", 1, 0),
        ("findings_new", "INTEGER", 0, 0),
        ("findings_known", "INTEGER", 0, 0),
        ("coverage_exclusions", "INTEGER", 0, 0),
        ("degraded_reason", "TEXT", 0, 0),
        ("mitos_version", "TEXT", 1, 0),
    ]


def test_widened_columns_shapes(tmp_path) -> None:
    """``conflict_checks.surface`` is NOT NULL DEFAULT 'sync'; ``model_id`` is nullable TEXT."""
    path = _telemetry_path(tmp_path)
    TelemetryStore(path)
    conn = _open(path)
    try:
        checks = {
            r["name"]: (r["type"], r["notnull"], r["dflt_value"])
            for r in conn.execute("PRAGMA table_info(conflict_checks);").fetchall()
        }
        batches = {
            r["name"]: (r["type"], r["notnull"], r["dflt_value"])
            for r in conn.execute("PRAGMA table_info(judgment_batches);").fetchall()
        }
    finally:
        conn.close()
    # The DEFAULT is the rung-2 backfill mechanism (and permanent — SQLite requires
    # it on ADD COLUMN NOT NULL); the writers' lockstep is the real fence.
    assert checks["surface"] == ("TEXT", 1, "'sync'")
    assert batches["model_id"] == ("TEXT", 0, None)


def test_check_runs_checks_reject_out_of_contract_values(tmp_path) -> None:
    """The belt-and-suspenders CHECKs pin the closed sets: mode, exit_code, one-row-per-run."""
    path = _telemetry_path(tmp_path)
    TelemetryStore(path)
    insert = (
        "INSERT INTO check_runs (run_id, mode, started_at, ended_at, exit_code, "
        "nodes_swept, pairs_judged_fresh, pairs_reused, findings_new, findings_known, "
        "coverage_exclusions, degraded_reason, mitos_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"
    )
    good = ("run-1", "corpus", CREATED_AT, CREATED_AT, 0, 5, 2, 3, 1, 0, 0, None, "test-v")
    conn = _open(path)
    try:
        with conn:
            conn.execute(insert, good)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(insert, ("run-2", "watch") + good[2:])  # mode outside the set
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(insert, ("run-3", "staged", CREATED_AT, CREATED_AT, 3) + good[5:])
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(insert, good)  # duplicate run_id — one row per run is structural
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
        assert _user_version(conn) == _pending_head(TELEMETRY_MIGRATION_STEPS)
    finally:
        conn.close()

    # Re-boot twice over an at-head file — the user_version gate skips the applied
    # rungs (rung 2's bare ALTERs are NOT re-runnable; the guard is what makes the
    # replay a no-op, MI-3).
    TelemetryStore(path)
    TelemetryStore(path)

    conn = _open(path)
    try:
        assert _user_version(conn) == _pending_head(TELEMETRY_MIGRATION_STEPS)
        assert _schema_snapshot(conn) == first_schema
    finally:
        conn.close()


# --- rung-2 upgrade: 'sync' backfill + NULL model_id on legacy rows -------------


def test_upgrade_backfills_surface_and_null_model_id(tmp_path) -> None:
    """A rung-1 DB with legacy rows boots to head: ``surface='sync'``, ``model_id`` NULL.

    The post-rung-2 writer cannot author a pre-rung-2 row (its INSERT names
    ``surface``), so the legacy fixture is built by running the SLICED ladder and
    INSERTing over the rung-1 column set via raw parameterized SQL. Every other
    field must read back byte-unchanged (``ADD COLUMN … DEFAULT`` is a schema
    operation — no row is UPDATEd; the append-only corpus is untouched).
    """
    path = _telemetry_path(tmp_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = open_connection(path)
    try:
        run_migrations(conn, TELEMETRY_MIGRATION_STEPS[:1])
        conn.execute(
            "INSERT INTO judgment_batches (batch_id, token_input, token_output, "
            "token_cache_read, token_cache_creation, elapsed_ms) "
            "VALUES (?, ?, ?, ?, ?, ?);",
            ("legacy-batch", 1200, 340, 0, 800, 2500),
        )
        conn.execute(
            "INSERT INTO conflict_checks (batch_id, sync_run_id, judged_axiom, "
            "proposal_rejected_paths, proposal_scope, proposed_hash_if_any, "
            "candidate_slug, candidate_hash, candidate_rejected_paths, "
            "candidate_scope, tenable, confidence, surfaced, candidate_source, "
            "model_alias, prompt_version, mitos_version, rationale, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
            (
                "legacy-batch", "sync-legacy", "Legacy judged axiom.", None, None,
                "hash-legacy", "legacy-cand", "cafe1234", "Rejected X.", None,
                0, 0.9, 1, "embedding_topk", "SONNET", "conflict-judge-v1",
                "0.5.21", "Legacy rationale.", CREATED_AT,
            ),
        )
        row_before = dict(conn.execute("SELECT * FROM conflict_checks;").fetchone())
        batch_before = dict(conn.execute("SELECT * FROM judgment_batches;").fetchone())
        assert _user_version(conn) == 1  # the deliberate pre-upgrade rung
    finally:
        conn.close()

    TelemetryStore(path)  # advance the ladder 1 -> head

    conn = _open(path)
    try:
        assert _user_version(conn) == _pending_head(TELEMETRY_MIGRATION_STEPS)
        row_after = dict(conn.execute("SELECT * FROM conflict_checks;").fetchone())
        batch_after = dict(conn.execute("SELECT * FROM judgment_batches;").fetchone())
        upgraded_schema = _schema_snapshot(conn)
    finally:
        conn.close()
    assert row_after.pop("surface") == "sync"  # the DEFAULT backfill
    assert row_after == row_before  # every pre-existing field byte-unchanged
    assert batch_after.pop("model_id") is None  # pre-rung-2 provenance is unknowable
    assert batch_after == batch_before

    # Stretch pin: fresh-install and upgraded-install execute the identical DDL
    # sequence (rung 1 then rung 2), so their stored schemas are byte-identical.
    fresh_path = str(tmp_path / ".mitos" / "fresh-telemetry.sqlite")
    TelemetryStore(fresh_path)
    conn = _open(fresh_path)
    try:
        assert _schema_snapshot(conn) == upgraded_schema
    finally:
        conn.close()


# --- MI-8: every connection routes through the store.open_connection chokepoint --


def test_connections_route_through_mi8_chokepoint(tmp_path, monkeypatch) -> None:
    """Boot + write open exactly two connections, both fully PRAGMA-configured.

    Asserts the chokepoint discipline behaviourally: a spy wraps
    ``mitos.telemetry.open_connection`` (the import-time binding — patching
    ``mitos.store`` would not intercept) and records each connection's live
    ``foreign_keys``/``busy_timeout``. A bare ``sqlite3.connect`` anywhere in the
    module would bypass the spy — the source-level pin closes that gap.
    """
    path = _telemetry_path(tmp_path)
    real_open = telemetry_module.open_connection
    pragmas = []

    def spy(db_path: str, read_only: bool = False) -> sqlite3.Connection:
        conn = real_open(db_path, read_only=read_only)
        pragmas.append(
            (
                conn.execute("PRAGMA foreign_keys;").fetchone()[0],
                conn.execute("PRAGMA busy_timeout;").fetchone()[0],
            )
        )
        return conn

    monkeypatch.setattr("mitos.telemetry.open_connection", spy)
    store = TelemetryStore(path)
    store.record_judged_batch(_batch(), [_row()], CREATED_AT)

    assert len(pragmas) == 2  # one boot connection + one per-write connection
    assert all(fk == 1 for fk, _ in pragmas)
    assert all(busy == BUSY_TIMEOUT_MS for _, busy in pragmas)
    # Supplementary source pin: telemetry never opens a bare sqlite3.connect.
    assert "sqlite3.connect" not in inspect.getsource(telemetry_module)


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
        # 'check' (not the builder's 'sync' default) proves the writer's VALUE binds
        # — a 'sync' read-back could be the schema DEFAULT masking an omitted column.
        _batch(model_id="claude-test-9"),
        [
            _row(
                judged_axiom=long_axiom,
                rationale=multiline_rationale,
                candidate_rejected_paths=multiline_rejected,
                surface="check",
            )
        ],
        CREATED_AT,
    )

    conn = _open(path)
    try:
        got = conn.execute(
            "SELECT judged_axiom, rationale, candidate_rejected_paths, created_at, "
            "surface FROM conflict_checks;"
        ).fetchone()
        got_batch = conn.execute(
            "SELECT model_id FROM judgment_batches;"
        ).fetchone()
    finally:
        conn.close()
    assert got["judged_axiom"] == long_axiom
    assert got["rationale"] == multiline_rationale
    assert got["candidate_rejected_paths"] == multiline_rejected
    assert got["created_at"] == CREATED_AT
    assert got["surface"] == "check"
    assert got_batch["model_id"] == "claude-test-9"


# --- criterion 4: NULL semantics ------------------------------------------------


def test_null_semantics_roundtrip(tmp_path) -> None:
    """Nullable fields bind None; present nullables bind their value, exactly."""
    path = _telemetry_path(tmp_path)
    store = TelemetryStore(path)

    store.record_judged_batch(
        # model_id=None is a legal EXPLICIT value (resolution failed) → SQL NULL,
        # never the empty string.
        _batch(model_id=None),
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
        got_batch = conn.execute(
            "SELECT model_id FROM judgment_batches;"
        ).fetchone()
    finally:
        conn.close()
    assert got["proposal_scope"] is None
    assert got["candidate_scope"] is None
    assert got["proposal_rejected_paths"] is None
    assert got["sync_run_id"] == "sync-9"
    assert got["proposed_hash_if_any"] == "hash-9"
    assert got_batch["model_id"] is None


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
