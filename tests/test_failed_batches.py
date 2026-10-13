"""Rung 6: the failed-judgment-batch table, its writer, and the spend union (Phase 2d, B7).

A corpus-check batch that fails after a response arrived was billed and, before this
rung, recorded nowhere. These rows pin the table's shape (usage whole or absent, no
``detail``), the writer's manners, the rung's upgrade path, and ``SPEND_UNION_SQL`` —
total spend across both batch tables, ``UNION ALL`` so identical counts both count.
The engine side (the carry and the row through ``cmd_check``) lives in
``test_check_failure_diagnosis.py``.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from typing import Any, Tuple

import pytest

from mitos import telemetry
from mitos.errors import DatabaseError
from mitos.migrations import _pending_head, run_migrations
from mitos.store import open_connection
from mitos.telemetry import (
    COVERAGE_RUNG,
    FAILED_BATCHES_RUNG,
    SPEND_UNION_SQL,
    TELEMETRY_MIGRATION_STEPS,
    CoverageMarks,
    FailedJudgmentBatch,
    JudgmentBatch,
    TelemetryStore,
    _FAILED_JUDGMENT_BATCHES_COLUMNS,
)

from test_check_coverage import _check_run_row

_USAGE_COLUMNS = ("token_input", "token_output", "token_cache_read",
                  "token_cache_creation")


def _failed(batch_id: str = "failed-1", *, usage: Any = (100, 40, 0, 0),
            stop_reason: Any = "max_tokens") -> FailedJudgmentBatch:
    """A valid failed-batch row; ``usage=None`` is the no-response shape."""
    tokens = usage if usage is not None else (None, None, None, None)
    return FailedJudgmentBatch(
        batch_id=batch_id,
        run_id="run-1",
        created_at="2026-09-21T00:00:00+00:00",
        reason="judgment_truncated",
        proposal_id="p" * 64,
        partner_ids=["a" * 64, "b" * 64],
        token_input=tokens[0],
        token_output=tokens[1],
        token_cache_read=tokens[2],
        token_cache_creation=tokens[3],
        stop_reason=stop_reason,
        model_id="claude-test",
        model_alias="SONNET",
        prompt_version="conflict-tenability-v1",
        mitos_version="test",
    )


def _rows(path: str) -> list:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM failed_judgment_batches ORDER BY batch_id")]
    finally:
        conn.close()


@pytest.fixture
def tel(tmp_path) -> TelemetryStore:
    return TelemetryStore(str(tmp_path / "telemetry.sqlite"))


# --------------------------------------------------------------------------- #
# T — the rung
# --------------------------------------------------------------------------- #

def test_fresh_store_boots_to_head_with_the_failed_batches_table(tmp_path) -> None:
    """T1: head reached; nullability as declared; the rung's step is this table's."""
    path = str(tmp_path / "telemetry.sqlite")
    TelemetryStore(path)
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == _pending_head(
            TELEMETRY_MIGRATION_STEPS)
        info = {r[1]: r for r in conn.execute(
            "PRAGMA table_info(failed_judgment_batches)")}
    finally:
        conn.close()
    assert tuple(info) == _FAILED_JUDGMENT_BATCHES_COLUMNS
    nullable = {name for name, r in info.items() if r[3] == 0}
    assert nullable == {*_USAGE_COLUMNS, "stop_reason", "model_id"}
    assert [name for name, r in info.items() if r[5]] == ["batch_id"]
    assert (dict(TELEMETRY_MIGRATION_STEPS)[FAILED_BATCHES_RUNG]
            is telemetry._failed_judgment_batches_schema)
    assert FAILED_BATCHES_RUNG > COVERAGE_RUNG


def test_rung_five_file_upgrades_with_rows_intact_and_replay_is_a_no_op(tmp_path) -> None:
    """T2: a rung-5 file with a run row and coverage upgrades to head; replay changes nothing."""
    path = str(tmp_path / "telemetry.sqlite")
    conn = open_connection(path)
    run_migrations(conn, TELEMETRY_MIGRATION_STEPS[:COVERAGE_RUNG])
    conn.close()
    conn = open_connection(path)
    with conn:
        conn.execute(telemetry._INSERT_CHECK_RUN_SQL,
                     _check_run_row("old-run").to_params())
        conn.executemany(
            telemetry._UPSERT_COVERAGE_SQL,
            CoverageMarks(run_id="old-run", marked_at="t", covered=("n1", "n2"),
                          excluded=("n3",)).to_params(),
        )
    conn.close()

    store = TelemetryStore(path)
    store.record_failed_batch(_failed())

    def snapshot() -> Tuple[int, list, list, list, list]:
        c = sqlite3.connect(path)
        try:
            return (
                c.execute("PRAGMA user_version").fetchone()[0],
                c.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall(),
                c.execute("SELECT run_id FROM check_runs").fetchall(),
                c.execute("SELECT * FROM check_coverage ORDER BY node_id").fetchall(),
                c.execute("SELECT * FROM failed_judgment_batches").fetchall(),
            )
        finally:
            c.close()

    before = snapshot()
    assert before[0] == _pending_head(TELEMETRY_MIGRATION_STEPS)
    assert before[2] == [("old-run",)]
    assert [r[:2] for r in before[3]] == [
        ("n1", "covered"), ("n2", "covered"), ("n3", "excluded")]
    assert len(before[4]) == 1
    conn = open_connection(path)
    run_migrations(conn, TELEMETRY_MIGRATION_STEPS)
    conn.close()
    TelemetryStore(path)
    assert snapshot() == before


# --------------------------------------------------------------------------- #
# W — the writer
# --------------------------------------------------------------------------- #

def test_to_params_arity_matches_the_column_tuple() -> None:
    """The lockstep pin: one value per INSERT column."""
    assert len(_failed().to_params()) == len(_FAILED_JUDGMENT_BATCHES_COLUMNS)


def test_a_failed_batch_row_round_trips_with_nulls_kept_null(tel) -> None:
    """W1: ``partner_ids`` reads back a JSON array; no-response usage reads back NULL."""
    billed = _failed("billed-1")
    silent = _failed("silent-1", usage=None, stop_reason=None)
    tel.record_failed_batch(billed)
    tel.record_failed_batch(silent)

    by_id = {r["batch_id"]: r for r in _rows(tel.telemetry_path)}
    for row in (billed, silent):
        stored = by_id[row.batch_id]
        assert json.loads(stored["partner_ids"]) == row.partner_ids
        stored["partner_ids"] = json.loads(stored["partner_ids"])
        assert FailedJudgmentBatch(**stored) == row
    assert stored["partner_ids"] == ["a" * 64, "b" * 64]
    assert [by_id["silent-1"][c] for c in _USAGE_COLUMNS] == [None] * 4
    assert by_id["silent-1"]["stop_reason"] is None
    assert [by_id["billed-1"][c] for c in _USAGE_COLUMNS] == [100, 40, 0, 0]


@pytest.mark.parametrize("hole", range(4))
def test_a_half_null_usage_set_is_refused(tel, hole: int) -> None:
    """W2: usage is whole or absent — one missing token column refuses the row."""
    usage = [1, 2, 3, 4]
    usage[hole] = None
    with pytest.raises(DatabaseError, match="Failed to persist failed batch"):
        tel.record_failed_batch(_failed(usage=tuple(usage)))
    assert _rows(tel.telemetry_path) == []


def test_a_duplicate_batch_id_is_refused_and_the_first_row_stands(tel) -> None:
    """W3: ``batch_id`` is the PK."""
    tel.record_failed_batch(_failed("dup"))
    with pytest.raises(DatabaseError):
        tel.record_failed_batch(_failed("dup", usage=None, stop_reason=None))
    (row,) = _rows(tel.telemetry_path)
    assert row["token_input"] == 100


def test_detail_has_no_field_and_no_column(tel) -> None:
    """W4: ``Unavailable.detail`` can carry a request id; it has nowhere to land."""
    assert not any("detail" in f.name for f in dataclasses.fields(FailedJudgmentBatch))
    conn = sqlite3.connect(tel.telemetry_path)
    try:
        columns = [r[1] for r in conn.execute(
            "PRAGMA table_info(failed_judgment_batches)")]
    finally:
        conn.close()
    assert columns and not any("detail" in c for c in columns)


# --------------------------------------------------------------------------- #
# U — the spend union
# --------------------------------------------------------------------------- #

def test_spend_union_counts_identical_batches_twice_and_counts_no_usage_rows(tel) -> None:
    """U1: judged + billed-failed sum per column (identical counts both kept); 1 no-usage row."""
    tel.record_judged_batch(
        JudgmentBatch(batch_id="judged-1", model_id="claude-test", token_input=100,
                      token_output=40, token_cache_read=0, token_cache_creation=0,
                      elapsed_ms=10, stop_reason="tool_use"),
        [], "2026-09-21T00:00:00+00:00",
    )
    tel.record_failed_batch(_failed("billed-1", usage=(100, 40, 0, 0)))
    tel.record_failed_batch(_failed("silent-1", usage=None, stop_reason=None))

    conn = sqlite3.connect(tel.telemetry_path)
    try:
        conn.row_factory = sqlite3.Row
        total = dict(conn.execute(SPEND_UNION_SQL).fetchone())
    finally:
        conn.close()
    assert total == {
        "token_input": 200, "token_output": 80, "token_cache_read": 0,
        "token_cache_creation": 0, "failed_batches_without_usage": 1,
    }


def test_spend_union_over_empty_tables_is_zero(tel) -> None:
    """A workspace that never judged reads zero spend, not NULL."""
    conn = sqlite3.connect(tel.telemetry_path)
    try:
        assert conn.execute(SPEND_UNION_SQL).fetchone() == (0, 0, 0, 0, 0)
    finally:
        conn.close()
