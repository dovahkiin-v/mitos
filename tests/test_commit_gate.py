"""Tests for the commit gate (G1) — hermetic, no network, no spend.

G1's hook asks one question: has an unscoped corpus check been *attempted* since the
uncovered set last changed? This module grows one section per phase:

* 3a — the last-attempt record: telemetry rung 7 (``check_attempt``), its writers
  (the ``started`` entry write and the outcome at the run-end seam), the read-only
  reader ``read_last_attempt``, the pure builder ``check.attempt_outcome_from_result``
  and ``cmd_check``'s two writes;
* 3b — the refused spend (``spend_not_authorized`` + the planned batch count, on the
  attempt's own record), the unrecorded-attempt disclosure on every answer
  ``cmd_check`` gives, and the refusal's on-record clause;
* 3c1 — the hook predicate (``commit_gate.evaluate_gate``): the shared key test
  ``config.judge_api_key``, the five rows each proven as a transition from
  ``blocked``, ``cmd_hook_run``'s channels and block code, a subprocess probe that no
  row loads an LLM SDK, and ``--staged``'s blindness to an MCP-path record;
* 3c2 onward — the verb's own boundary, the status gate row and the standing notice.

Every ``cmd_check`` row injects both seams (substrate and judge): offline, a run with
no embedding provider over a non-empty corpus exits before the seam, and a run-end
row that injected only the judge would pass for that reason. Values that shift
between runs (ids, timestamps, fingerprints) are read back and compared across rows,
never hardcoded. Real SQLite throughout.
"""

import dataclasses
import json
import os
import sqlite3
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple

import pytest

from mitos import check, cli, telemetry
from mitos.audit_debt import AuditDebt, derive_audit_debt
from mitos.check import CheckFinding
from mitos.cli import cmd_init
from mitos.commit_gate import (
    GATE_ATTEMPTED,
    GATE_BLOCKED,
    GATE_CAUSES,
    GATE_KEYLESS,
    GATE_NOTHING_UNCOVERED,
    GATE_ROWS,
    GATE_UNREADABLE,
    HOOK_BLOCK_EXIT,
    GateVerdict,
    evaluate_gate,
)
from mitos.config import MitosConfig, judge_api_key
from mitos.conflict import ConflictUnavailableReason, Unavailable
from mitos.errors import DatabaseError, MitosError
from mitos.migrations import _pending_head, run_migrations
from mitos.recall import provenance_line
from mitos.store import GraphStore, open_connection
from mitos.sync import MitosSyncManager
from mitos.telemetry import (
    ATTEMPT_COULD_NOT_COMPLETE,
    ATTEMPT_NEW_FINDINGS,
    ATTEMPT_NO_NEW_FINDINGS,
    ATTEMPT_RUN_END_STATES,
    ATTEMPT_RUNG,
    ATTEMPT_SPEND_NOT_AUTHORIZED,
    ATTEMPT_STARTED,
    FAILED_BATCHES_RUNG,
    TELEMETRY_MIGRATION_STEPS,
    AttemptOutcome,
    AttemptRefusal,
    AttemptStart,
    CoverageMarks,
    LastAttempt,
    ReuseUnavailable,
    TelemetryAbsent,
    TelemetryStore,
    TelemetryUnreadable,
    read_last_attempt,
)

from _conflict_helpers import _drain_outbox, _match
from test_check_cli import (  # noqa: F401  (offline is autouse; the rest fixtures)
    _FakeStdin,
    _FaultStore,
    _assert_parses,
    _pair,
    _read_check_runs,
    _recipe,
    _wire_judge,
    _wire_substrate,
    offline,
    recipe_workspace,
    workspace,
)
from test_check_coverage import (
    _CommittingJudge,
    _check_run_row,
    _coverage,
    _healthy_result,
    _judge_for,
    _run,
    _scoped_corpus,
    _tables,
)
from test_check_probe import _commit, _poison, _seed_verdict


# =========================================================================== #
# Phase 3a — the last-attempt record
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_ATTEMPT_COLUMNS = (
    "slot", "attempt_id", "started_at", "fingerprint", "state", "run_id",
    "outcome_at", "degradation_tokens", "new_pairs", "findings_known",
    "batches_planned",
)

_OUTCOME_COLUMNS = (
    "run_id", "outcome_at", "degradation_tokens", "new_pairs", "findings_known",
)


def _attempt_rows(path: str) -> list:
    """Every ``check_attempt`` row, as dicts over every column."""
    conn = sqlite3.connect(path)
    try:
        return [
            dict(zip(_ATTEMPT_COLUMNS, row))
            for row in conn.execute(
                f"SELECT {', '.join(_ATTEMPT_COLUMNS)} FROM check_attempt"
            )
        ]
    finally:
        conn.close()


def _attempt(path: str) -> Optional[Dict[str, Any]]:
    """The one attempt row, or ``None``; asserts there is never more than one."""
    rows = _attempt_rows(path)
    assert len(rows) <= 1
    return rows[0] if rows else None


def _fingerprint(config: MitosConfig) -> str:
    """The leaf's fingerprint of the uncovered set, right now."""
    debt = derive_audit_debt(config.db_path, config.telemetry_path)
    assert isinstance(debt, AuditDebt)
    return debt.fingerprint


def _assert_started(row: Optional[Dict[str, Any]], fingerprint: str) -> None:
    """The row is an entry record: ``started``, the fingerprint, no outcome."""
    assert row is not None
    assert row["state"] == ATTEMPT_STARTED
    assert row["fingerprint"] == fingerprint
    assert all(row[c] is None for c in _OUTCOME_COLUMNS)
    assert row["batches_planned"] is None


def _outcome(**overrides: Any) -> AttemptOutcome:
    base = dict(
        attempt_id="att-1", state=ATTEMPT_NO_NEW_FINDINGS, run_id="run-1",
        outcome_at="2026-09-21T00:01:00.000000+00:00", degradation_tokens=(),
        new_pairs=(), findings_known=0,
    )
    base.update(overrides)
    return AttemptOutcome(**base)


def _start(attempt_id: str = "att-1", fingerprint: str = "f" * 64) -> AttemptStart:
    return AttemptStart(attempt_id=attempt_id,
                        started_at="2026-09-21T00:00:00.000000+00:00",
                        fingerprint=fingerprint)


def _raw_attempt_update(path: str, sql: str, *params: Any) -> None:
    conn = sqlite3.connect(path)
    with conn:
        conn.execute(sql, params)
    conn.close()


# --------------------------------------------------------------------------- #
# 1 — the rung
# --------------------------------------------------------------------------- #

def test_fresh_store_boots_to_head_with_the_attempt_table(tmp_path) -> None:
    """Criterion 1: a fresh store reaches the head with ``check_attempt`` present."""
    path = str(tmp_path / "telemetry.sqlite")
    TelemetryStore(path)
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == _pending_head(
            TELEMETRY_MIGRATION_STEPS)
        assert "check_attempt" in _tables(conn)
    finally:
        conn.close()
    assert dict(TELEMETRY_MIGRATION_STEPS)[ATTEMPT_RUNG] is telemetry._check_attempt_schema
    assert ATTEMPT_RUNG > FAILED_BATCHES_RUNG


def test_rung_six_file_upgrades_with_rows_intact_and_replay_is_a_no_op(tmp_path) -> None:
    """Criterion 1: a rung-6 file upgrades keeping its rows; replay at head changes nothing."""
    path = str(tmp_path / "telemetry.sqlite")
    conn = open_connection(path)
    run_migrations(conn, TELEMETRY_MIGRATION_STEPS[:FAILED_BATCHES_RUNG])
    conn.close()
    conn = open_connection(path)
    with conn:
        conn.execute(telemetry._INSERT_CHECK_RUN_SQL,
                     _check_run_row("old-run").to_params())
        conn.executemany(
            telemetry._UPSERT_COVERAGE_SQL,
            CoverageMarks(run_id="old-run", marked_at="t", covered=("n1",),
                          excluded=()).to_params(),
        )
    conn.close()

    store = TelemetryStore(path)
    store.record_attempt_start(_start())

    def snapshot() -> Tuple[int, list, list, list, list]:
        c = sqlite3.connect(path)
        try:
            return (
                c.execute("PRAGMA user_version").fetchone()[0],
                c.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall(),
                c.execute("SELECT run_id FROM check_runs").fetchall(),
                c.execute("SELECT * FROM check_coverage").fetchall(),
                c.execute("SELECT * FROM check_attempt").fetchall(),
            )
        finally:
            c.close()

    before = snapshot()
    assert before[0] == _pending_head(TELEMETRY_MIGRATION_STEPS)
    assert before[2] == [("old-run",)]
    assert [r[:2] for r in before[3]] == [("n1", "covered")]
    assert len(before[4]) == 1
    conn = open_connection(path)
    run_migrations(conn, TELEMETRY_MIGRATION_STEPS)
    conn.close()
    TelemetryStore(path)
    assert snapshot() == before


# --------------------------------------------------------------------------- #
# 2–3 — the read-only reader and the round trip
# --------------------------------------------------------------------------- #

def test_reader_on_an_absent_path_creates_nothing(tmp_path) -> None:
    """Criterion 2: no file (or no parent) → ``TelemetryAbsent('no_file')``, still no file."""
    path = tmp_path / "telemetry.sqlite"
    assert read_last_attempt(str(path)) == TelemetryAbsent("no_file")
    assert not os.path.exists(path)

    nested = tmp_path / "missing-dir" / "telemetry.sqlite"
    assert read_last_attempt(str(nested)) == TelemetryAbsent("no_file")
    assert not os.path.exists(nested.parent)


def test_reader_on_a_rung_six_file_leaves_it_untouched(tmp_path) -> None:
    """Criterion 2: a rung-6 file → ``below_rung``; its bytes and user_version unchanged."""
    path = tmp_path / "telemetry.sqlite"
    conn = open_connection(str(path))
    run_migrations(conn, TELEMETRY_MIGRATION_STEPS[:FAILED_BATCHES_RUNG])
    conn.close()
    before = path.read_bytes()

    assert read_last_attempt(str(path)) == TelemetryAbsent("below_rung")
    assert path.read_bytes() == before
    conn = sqlite3.connect(str(path))
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == FAILED_BATCHES_RUNG
        assert "check_attempt" not in _tables(conn)
    finally:
        conn.close()


def test_reader_on_a_corrupt_file_or_a_directory_is_unreadable(tmp_path) -> None:
    """Criterion 2: a file that exists and cannot be read is a fault, never "no attempt"."""
    corrupt = tmp_path / "telemetry.sqlite"
    corrupt.write_bytes(b"this is not a sqlite database, not even close" * 20)
    assert isinstance(read_last_attempt(str(corrupt)), TelemetryUnreadable)

    directory = tmp_path / "a-directory"
    directory.mkdir()
    assert isinstance(read_last_attempt(str(directory)), TelemetryUnreadable)


def test_reader_on_an_empty_table_is_none_not_absent(tmp_path) -> None:
    """Criterion 2: a rung-7 file with no row → ``None``, a different type from Absent."""
    path = str(tmp_path / "telemetry.sqlite")
    TelemetryStore(path)
    assert read_last_attempt(path) is None


def test_round_trip_start_then_outcome(tmp_path) -> None:
    """Criterion 3: a start then an outcome read back as tuples, nulls as ``None``."""
    path = str(tmp_path / "telemetry.sqlite")
    store = TelemetryStore(path)
    start = _start()
    store.record_attempt_start(start)

    assert read_last_attempt(path) == LastAttempt(
        attempt_id=start.attempt_id, started_at=start.started_at,
        fingerprint=start.fingerprint, state=ATTEMPT_STARTED, run_id=None,
        outcome_at=None, degradation_tokens=None, new_pairs=None,
        findings_known=None, batches_planned=None,
    )

    outcome = _outcome(state=ATTEMPT_COULD_NOT_COMPLETE,
                       degradation_tokens=("judgment", "judgment_truncated"),
                       new_pairs=(("p1", "q1"), ("p2", "q2")), findings_known=None)
    store.record_run_end(_check_run_row("run-1"), coverage=None, attempt=outcome)

    read = read_last_attempt(path)
    assert read == LastAttempt(
        attempt_id=start.attempt_id, started_at=start.started_at,
        fingerprint=start.fingerprint, state=ATTEMPT_COULD_NOT_COMPLETE,
        run_id="run-1", outcome_at=outcome.outcome_at,
        degradation_tokens=("judgment", "judgment_truncated"),
        new_pairs=(("p1", "q1"), ("p2", "q2")), findings_known=None,
        batches_planned=None,
    )
    assert isinstance(read.new_pairs[0], tuple)
    # Persisted as JSON lists, never tuples.
    raw = _attempt(path)
    assert json.loads(raw["new_pairs"]) == [["p1", "q1"], ["p2", "q2"]]
    assert json.loads(raw["degradation_tokens"]) == ["judgment", "judgment_truncated"]


def test_reader_returns_an_unknown_state_verbatim(tmp_path) -> None:
    """Criterion 3: a newer build's state is returned as-is, never rejected."""
    path = str(tmp_path / "telemetry.sqlite")
    TelemetryStore(path).record_attempt_start(_start())
    _raw_attempt_update(path, "UPDATE check_attempt SET state = ?", "a_later_state")

    read = read_last_attempt(path)
    assert isinstance(read, LastAttempt) and read.state == "a_later_state"


@pytest.mark.parametrize("column, damaged", [
    ("new_pairs", "not json at all"),
    ("new_pairs", '[["only-one-id"]]'),
    ("new_pairs", '[["p", 7]]'),
    ("degradation_tokens", '{"not": "a list"}'),
], ids=["pairs-not-json", "pair-of-one", "pair-not-strings", "tokens-not-a-list"])
def test_a_damaged_json_column_is_unreadable(tmp_path, column: str, damaged: str) -> None:
    """Criterion 3: a column that does not decode is a damaged row → ``TelemetryUnreadable``."""
    path = str(tmp_path / "telemetry.sqlite")
    TelemetryStore(path).record_attempt_start(_start())
    _raw_attempt_update(path, f"UPDATE check_attempt SET {column} = ?", damaged)

    assert isinstance(read_last_attempt(path), TelemetryUnreadable)


# --------------------------------------------------------------------------- #
# 4 — the boundary shapes refuse before a connection opens
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("state", [ATTEMPT_STARTED, ATTEMPT_SPEND_NOT_AUTHORIZED, "done", ""])
def test_outcome_refuses_a_state_that_is_not_a_run_end_state(state: str) -> None:
    """Criterion 4: ``started`` and any unknown state are refused at construction."""
    with pytest.raises(ValueError, match="run-end state"):
        _outcome(state=state)


def test_outcome_refuses_a_pair_that_is_not_two_ids() -> None:
    with pytest.raises(ValueError, match="two node ids"):
        _outcome(new_pairs=(("only-one",),))


@pytest.mark.parametrize("field", ["attempt_id", "started_at", "fingerprint"])
def test_start_refuses_an_empty_field(field: str) -> None:
    """Criterion 4: every entry field is a non-empty string."""
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(_start(), **{field: ""})


def test_refusal_happens_before_any_connection(tmp_path, monkeypatch) -> None:
    """Criterion 4: a refused shape never reaches the writer, so nothing opens."""
    opened = []
    monkeypatch.setattr(telemetry, "open_connection",
                        lambda *a, **k: opened.append(a) or pytest.fail("opened"))
    with pytest.raises(ValueError):
        _outcome(state=ATTEMPT_STARTED)
    with pytest.raises(ValueError):
        _start(attempt_id="")
    assert opened == []


# --------------------------------------------------------------------------- #
# 5 — the builder, over synthetic results
# --------------------------------------------------------------------------- #

def _finding(proposal: str, partner: str, novelty: Optional[str]) -> CheckFinding:
    return CheckFinding(
        proposal_hash=proposal, partner_hash=partner,
        proposal_node={"id": proposal}, partner_node={"id": partner},
        score=0.9, rationale="Synthetic.", confidence=0.9, reused=novelty == "known",
        source_batch_id="b0", source_created_at="2026-09-21T00:00:00+00:00",
        novelty=novelty,
    )


_JUDGMENT_FAILURE = (Unavailable(reason=ConflictUnavailableReason.JUDGMENT, detail="x"),)

_BUILDER_CASES = {
    "healthy-known-only": (
        dict(findings=(_finding("a", "b", "known"), _finding("a", "c", "known"))),
        ATTEMPT_NO_NEW_FINDINGS, (), 2,
    ),
    "healthy-one-new": (
        dict(findings=(_finding("a", "b", "new"), _finding("a", "c", "known"))),
        ATTEMPT_NEW_FINDINGS, (("a", "b"),), 1,
    ),
    "degraded-one-new": (
        dict(findings=(_finding("a", "b", "new"),), judgment_failures=_JUDGMENT_FAILURE,
             batches_failed=1),
        ATTEMPT_COULD_NOT_COMPLETE, (("a", "b"),), 0,
    ),
    "degraded-none": (
        dict(judgment_failures=_JUDGMENT_FAILURE, batches_failed=1),
        ATTEMPT_COULD_NOT_COMPLETE, (), 0,
    ),
    "reuse-read-novelty-none": (
        dict(findings=(_finding("a", "b", None),),
             reuse_unavailable=ReuseUnavailable("broken read")),
        ATTEMPT_COULD_NOT_COMPLETE, (), None,
    ),
}


@pytest.mark.parametrize("case", list(_BUILDER_CASES), ids=list(_BUILDER_CASES))
def test_builder_rules(case: str) -> None:
    """Criterion 5: degraded dominates the state and keeps the pairs; None is never new."""
    overrides, state, pairs, known = _BUILDER_CASES[case]
    result = _healthy_result(**overrides)

    outcome = check.attempt_outcome_from_result(result, attempt_id="att-x")

    assert outcome.state == state
    assert outcome.new_pairs == pairs
    assert outcome.findings_known == known
    assert outcome.degradation_tokens == check.run_degradations(result)
    assert (outcome.attempt_id, outcome.run_id, outcome.outcome_at) == (
        "att-x", result.run_id, result.ended_at)
    row = check.check_run_row_from_result(
        result, mode="corpus", exit_code=check.exit_code_for(result))
    assert outcome.findings_known == row.findings_known


def test_builder_keeps_tokens_in_declaration_order() -> None:
    """Criterion 5: two tokens arrive in ``_DEGRADATION_TOKENS`` order, as the row joins them."""
    result = _healthy_result(judgment_failures=_JUDGMENT_FAILURE, batches_failed=1,
                             telemetry_write_failures=("batch b0: disk full",))
    outcome = check.attempt_outcome_from_result(result, attempt_id="att-x")
    assert outcome.degradation_tokens == ("judgment", "telemetry_write")
    row = check.check_run_row_from_result(
        result, mode="corpus", exit_code=check.exit_code_for(result))
    assert ",".join(outcome.degradation_tokens) == row.degraded_reason


# --------------------------------------------------------------------------- #
# 6 — the entry write precedes the substrate
# --------------------------------------------------------------------------- #

def test_substrate_unavailable_leaves_started(workspace, capsys) -> None:
    """Criterion 6: no embedding provider over a non-empty corpus → exit 2, ``started``."""
    config, store, _tel = workspace
    _pair(store)
    _drain_outbox(store)
    entry = _fingerprint(config)
    # No substrate monkeypatch: real construction has no embedding provider offline.

    code, obj = _run(config, capsys)

    assert code == 2 and obj["code"] == "substrate_unavailable"
    _assert_started(_attempt(config.telemetry_path), entry)


def test_a_substrate_that_raises_still_leaves_started(workspace, monkeypatch, capsys) -> None:
    """Criterion 6 (scout W-1): the entry write sits before the substrate is built at all."""
    config, store, _tel = workspace
    _pair(store)
    _drain_outbox(store)
    entry = _fingerprint(config)

    def _raising(config: MitosConfig) -> Any:
        raise MitosError("substrate construction blew up")

    monkeypatch.setattr(cli, "_build_check_substrate", _raising)

    code, obj = _run(config, capsys)

    assert code == 2 and obj["code"] == "check_faulted"
    _assert_started(_attempt(config.telemetry_path), entry)


def test_a_store_fault_at_the_start_probe_leaves_started(workspace, monkeypatch, capsys) -> None:
    """Criterion 6: ``check_faulted`` after entry leaves ``started``, with no run row."""
    config, store, _tel = workspace
    _a, _b, nbhds = _pair(store)
    _drain_outbox(store)
    entry = _fingerprint(config)
    _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, None)
    # Replaces only cli's name; the leaf imports GraphStore from mitos.store.
    monkeypatch.setattr(cli, "GraphStore", lambda path: _FaultStore(GraphStore(path)))

    code, obj = _run(config, capsys)

    assert code == 2 and obj["code"] == "check_faulted"
    assert _read_check_runs(config) == []
    _assert_started(_attempt(config.telemetry_path), entry)


# --------------------------------------------------------------------------- #
# 7–10 — the run-end outcome, through cmd_check
# --------------------------------------------------------------------------- #

def _reported_new_pairs(obj: Dict[str, Any]) -> list:
    """The new findings the run's own report names, as ``[proposal_id, partner_id]``."""
    return [[f["proposal"]["id"], f["partner"]["id"]]
            for f in obj["findings"] if f["novelty"] == "new"]


def _assert_outcome_joins_its_run(config: MitosConfig, obj: Dict[str, Any],
                                  row: Dict[str, Any], entry: str) -> None:
    (run,) = [r for r in _read_check_runs(config) if r["run_id"] == obj["run_id"]]
    assert row["run_id"] == run["run_id"]
    assert row["outcome_at"] == run["ended_at"]
    assert row["fingerprint"] == entry


def test_a_reused_known_finding_records_no_new_findings(workspace, monkeypatch, capsys) -> None:
    """Criterion 7: known only → ``no_new_findings``, ``findings_known == 1``."""
    config, store, tel = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    _seed_verdict(tel, proposal_hash=a_id, candidate_hash=b_id, tenable=False,
                  confidence=0.9, batch_id="gate-prior-batch",
                  created_at="2026-06-01T00:00:00.000000+00:00")
    entry = _fingerprint(config)
    _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, None)  # reuse-only: the judge is never built

    code, obj = _run(config, capsys)

    assert code == 0
    row = _attempt(config.telemetry_path)
    assert row["state"] == ATTEMPT_NO_NEW_FINDINGS
    assert row["findings_known"] == 1
    assert json.loads(row["new_pairs"]) == []
    assert json.loads(row["degradation_tokens"]) == []
    _assert_outcome_joins_its_run(config, obj, row, entry)


def test_a_first_ever_finding_records_new_findings(workspace, monkeypatch, capsys) -> None:
    """Criterion 7: one new finding → ``new_findings`` with its pair."""
    config, store, tel = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    entry = _fingerprint(config)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel, tenable=False))

    code, obj = _run(config, capsys)

    assert code == 1
    row = _attempt(config.telemetry_path)
    assert row["state"] == ATTEMPT_NEW_FINDINGS
    pairs = json.loads(row["new_pairs"])
    assert pairs == _reported_new_pairs(obj)
    assert len(pairs) == 1 and set(pairs[0]) == {a_id, b_id}
    assert row["findings_known"] == 0
    _assert_outcome_joins_its_run(config, obj, row, entry)
    assert read_last_attempt(config.telemetry_path).new_pairs == (tuple(pairs[0]),)


def test_a_failing_batch_records_could_not_complete(workspace, monkeypatch, capsys) -> None:
    """Criterion 7: a failing batch, no finding → ``could_not_complete`` naming judgment."""
    config, store, tel = workspace
    _a, _b, nbhds = _pair(store)
    _drain_outbox(store)
    entry = _fingerprint(config)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    failing = Unavailable(reason=ConflictUnavailableReason.JUDGMENT, detail="judge died")
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel, overrides={0: failing}))

    code, obj = _run(config, capsys)

    assert code == 2
    row = _attempt(config.telemetry_path)
    assert row["state"] == ATTEMPT_COULD_NOT_COMPLETE
    assert "judgment" in json.loads(row["degradation_tokens"])
    assert json.loads(row["new_pairs"]) == []
    _assert_outcome_joins_its_run(config, obj, row, entry)


def test_a_degraded_run_keeps_its_new_pair(workspace, monkeypatch, capsys) -> None:
    """Criterion 8: one failing batch plus one new finding → tokens **and** the pair."""
    config, store, tel = workspace
    a_axiom, b_axiom, c_axiom = (
        "Gate axiom alpha.", "Gate axiom beta.", "Gate axiom gamma.")
    a_id = _commit(store, "gate-a", a_axiom)
    b_id = _commit(store, "gate-b", b_axiom)
    c_id = _commit(store, "gate-c", c_axiom)
    nbhds = {a_axiom: [_match("gate-b", 0.9)], b_axiom: [],
             c_axiom: [_match("gate-b", 0.9)]}
    _drain_outbox(store)
    entry = _fingerprint(config)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    failing = Unavailable(reason=ConflictUnavailableReason.JUDGMENT, detail="judge died")
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel, tenable=False,
                                        overrides={0: failing}))

    code, obj = _run(config, capsys)

    assert code == 2  # degraded dominates the exit code ...
    row = _attempt(config.telemetry_path)
    assert row["state"] == ATTEMPT_COULD_NOT_COMPLETE  # ... and never the record
    assert "judgment" in json.loads(row["degradation_tokens"])
    pairs = json.loads(row["new_pairs"])
    assert pairs == _reported_new_pairs(obj)
    assert len(pairs) == 1 and b_id in pairs[0] and {a_id, c_id} & set(pairs[0])
    _assert_outcome_joins_its_run(config, obj, row, entry)


def test_the_fingerprint_is_not_rewritten(workspace, monkeypatch, capsys) -> None:
    """Criterion 9: a decision committed mid-run leaves the entry-time fingerprint standing."""
    config, store, tel = workspace
    _a, _b, nbhds = _pair(store)
    _drain_outbox(store)
    entry = _fingerprint(config)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    judge = _CommittingJudge(_judge_for(store, embed, vector, tel), store)
    _wire_judge(monkeypatch, judge)

    code, obj = _run(config, capsys)

    assert judge.committed is not None and obj["degradations"] == []
    row = _attempt(config.telemetry_path)
    assert row["state"] in (ATTEMPT_NO_NEW_FINDINGS, ATTEMPT_NEW_FINDINGS)
    assert row["fingerprint"] == entry
    after = derive_audit_debt(config.db_path, config.telemetry_path)
    assert after.uncovered_ids == frozenset({judge.committed})
    assert row["fingerprint"] != after.fingerprint


class _OverlappingJudge:
    """Delegates to a canned judge; on its first call a second attempt replaces the record."""

    def __init__(self, inner: Any, tel: TelemetryStore, start: AttemptStart) -> None:
        self._inner = inner
        self._tel = tel
        self._start = start
        self.replaced = False

    def __call__(self, prompt: Any) -> Any:
        if not self.replaced:
            self._tel.record_attempt_start(self._start)
            self.replaced = True
        return self._inner(prompt)


def test_an_overlapping_attempt_keeps_the_record(workspace, monkeypatch, capsys) -> None:
    """Criterion 10: A's outcome lands nowhere once B replaced it; A's run row and coverage land."""
    config, store, tel = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    b_start = _start(attempt_id="attempt-b", fingerprint="b" * 64)
    judge = _OverlappingJudge(_judge_for(store, embed, vector, tel), tel, b_start)
    _wire_judge(monkeypatch, judge)

    code, obj = _run(config, capsys)

    assert judge.replaced and code == 0
    row = _attempt(config.telemetry_path)
    assert (row["attempt_id"], row["fingerprint"]) == ("attempt-b", "b" * 64)
    _assert_started(row, "b" * 64)
    assert obj["summary_row_written"] is True
    assert [r["run_id"] for r in _read_check_runs(config)] == [obj["run_id"]]
    cov = _coverage(config.telemetry_path)
    assert set(cov) == {a_id, b_id}
    assert {run_id for _, run_id, _ in cov.values()} == {obj["run_id"]}


# --------------------------------------------------------------------------- #
# 11–13 — who writes, and what survives
# --------------------------------------------------------------------------- #

def test_scoped_and_staged_runs_write_nothing(workspace, monkeypatch, capsys) -> None:
    """Criterion 11: ``--scope x``, ``--scope ""`` and ``--staged`` leave the row byte-identical."""
    config, store, tel = workspace
    _a, _b, nbhds = _pair(store)
    _drain_outbox(store)
    tel.record_attempt_start(_start(fingerprint="s" * 64))
    tel.record_run_end(
        _check_run_row("seeded-run"), coverage=None,
        attempt=_outcome(run_id="seeded-run", new_pairs=(("p", "q"),),
                         degradation_tokens=("judgment",),
                         state=ATTEMPT_COULD_NOT_COMPLETE, findings_known=3))
    _raw_attempt_update(config.telemetry_path,
                        "UPDATE check_attempt SET batches_planned = ?", 9)
    seeded = _attempt(config.telemetry_path)
    assert all(v is not None for v in seeded.values())  # every column compared
    _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, None)

    for scope in ("no-such-tag", ""):
        code, _obj = _run(config, capsys, scope=scope)
        assert code == 0
        assert _attempt(config.telemetry_path) == seeded

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=True)
    capsys.readouterr()
    assert code == 0
    assert _attempt(config.telemetry_path) == seeded


def _refuse(config: MitosConfig, capsys: Any, monkeypatch: Any, form: str, *,
            scope: Optional[str] = None) -> Tuple[int, int, str, str]:
    """Runs a refused ``cmd_check`` in one of its three refusal forms.

    Returns ``(exit, batches_planned, out, err)``. The batch count is read off the
    refusal's own output (the ``--json`` key, or the leading count of the text
    line), never hand-counted.
    """
    if form == "tty":
        monkeypatch.setattr("sys.stdin", _FakeStdin(tty=True))
        monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    else:
        monkeypatch.setattr("sys.stdin", _FakeStdin(tty=False))
    code = cli.cmd_check(config, scope=scope, fresh=False, assume_yes=False,
                         as_json=form == "json")
    captured = capsys.readouterr()
    if form == "json":
        obj = json.loads(captured.out)
        assert obj["code"] == "confirmation_required"
        planned = obj["batches_planned"]
    else:
        text = captured.err if form == "text" else captured.out
        planned = int(text.split(" judgment batches pending", 1)[0].split()[-1])
    return code, planned, captured.out, captured.err


@pytest.mark.parametrize("form", ["text", "json", "tty"])
def test_a_refused_spend_records_spend_not_authorized(workspace, monkeypatch, capsys,
                                                       form: str) -> None:
    """Criterion 12 (3a), inverted by 3b criterion 4: the refusal marks its own record.

    Exit 2, no judge, no ``check_runs`` row; the record reads ``spend_not_authorized``
    with the refusal's batch count, the entry fingerprint intact and no ``run_id``.
    The refusal tells the caller the attempt is on record (3b criterion 12).
    """
    config, store, _tel = workspace
    _a, _b, nbhds = _pair(store)
    _drain_outbox(store)
    entry = _fingerprint(config)
    _wire_substrate(monkeypatch, nbhds)
    invoked = _wire_judge(monkeypatch, None)
    monkeypatch.setattr(check, "CHECK_CONFIRM_BATCHES", 0)

    code, planned, out, err = _refuse(config, capsys, monkeypatch, form)

    assert code == 2
    assert invoked == []
    assert _read_check_runs(config) == []
    row = _attempt(config.telemetry_path)
    assert row["state"] == ATTEMPT_SPEND_NOT_AUTHORIZED
    assert row["batches_planned"] == planned >= 1
    assert row["fingerprint"] == entry
    assert row["run_id"] is None and row["outcome_at"] is not None
    assert all(row[c] is None for c in ("degradation_tokens", "new_pairs", "findings_known"))
    said = {"text": err, "json": json.loads(out)["error"] if form == "json" else "",
            "tty": out}[form]
    assert cli._ATTEMPT_ON_RECORD_CLAUSE in said


def test_a_second_attempt_replaces_the_first_whole(workspace, monkeypatch, capsys) -> None:
    """Criterion 13 (D1): one row, the second attempt's; no column carries over."""
    config, store, tel = workspace
    _a, _b, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel))
    code, first_obj = _run(config, capsys)
    assert code == 0
    first = _attempt(config.telemetry_path)
    # A column only a later phase writes, seeded on the first attempt.
    _raw_attempt_update(config.telemetry_path,
                        "UPDATE check_attempt SET batches_planned = ?", 5)

    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel))
    code, second_obj = _run(config, capsys)

    assert code == 0
    rows = _attempt_rows(config.telemetry_path)
    assert len(rows) == 1
    (second,) = rows
    assert second["attempt_id"] != first["attempt_id"]
    assert second["run_id"] == second_obj["run_id"] != first_obj["run_id"]
    assert second["state"] == ATTEMPT_NO_NEW_FINDINGS
    assert second["batches_planned"] is None


# --------------------------------------------------------------------------- #
# 14–15 — faults: the entry write is best-effort, the outcome is transactional
# --------------------------------------------------------------------------- #

class _FailingStartTelemetry:
    """Delegates to a real store; only ``record_attempt_start`` raises ``exc``."""

    def __init__(self, inner: TelemetryStore, exc: Exception) -> None:
        self._inner = inner
        self._exc = exc

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def record_attempt_start(self, start: AttemptStart) -> None:
        raise self._exc


def test_an_entry_write_fault_does_not_touch_the_run(workspace, monkeypatch, capsys) -> None:
    """Criterion 14: the check runs as if unfaulted; there is just no attempt row."""
    config, store, tel = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel))
    faulted = _FailingStartTelemetry(tel, DatabaseError("provoked entry-write fault"))
    monkeypatch.setattr(cli, "_build_check_telemetry", lambda config: faulted)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=True)
    captured = capsys.readouterr()
    obj = json.loads(captured.out)

    assert code == 0 and obj["degradations"] == []
    assert obj["summary_row_written"] is True
    assert captured.err == ""  # a known fault is remembered silently
    assert set(_coverage(config.telemetry_path)) == {a_id, b_id}
    assert _attempt(config.telemetry_path) is None


def test_begin_attempt_without_telemetry_is_telemetry_unavailable(workspace) -> None:
    config, _store, _tel = workspace
    assert cli._begin_check_attempt(config, None) == cli._AttemptUnrecorded(
        "telemetry_unavailable")


def test_begin_attempt_over_a_corrupt_graph_is_fingerprint_unreadable(tmp_path) -> None:
    config = MitosConfig(str(tmp_path))
    tel = TelemetryStore(config.telemetry_path)
    os.makedirs(os.path.dirname(config.db_path), exist_ok=True)
    with open(config.db_path, "wb") as fh:
        fh.write(b"this is not a sqlite database, not even close" * 20)

    assert cli._begin_check_attempt(config, tel) == cli._AttemptUnrecorded(
        "fingerprint_unreadable")
    assert _attempt(config.telemetry_path) is None


def test_begin_attempt_write_fault_is_write_failed_silently(workspace, capsys) -> None:
    config, _store, tel = workspace
    faulted = _FailingStartTelemetry(tel, DatabaseError("provoked"))
    assert cli._begin_check_attempt(config, faulted) == cli._AttemptUnrecorded(
        "write_failed")
    assert capsys.readouterr().err == ""


def test_begin_attempt_unexpected_exception_warns_once(workspace, capsys) -> None:
    """Criterion 14 (D5): a bug is still ``write_failed``, and says so on stderr, once."""
    config, _store, tel = workspace
    faulted = _FailingStartTelemetry(tel, RuntimeError("a bug in the write path"))
    assert cli._begin_check_attempt(config, faulted) == cli._AttemptUnrecorded(
        "write_failed")
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("[Warning]") and "a bug in the write path" in lines[0]


def test_begin_attempt_lands_the_started_record(workspace) -> None:
    """The happy path: the key it returns is the key on the row."""
    config, store, tel = workspace
    _pair(store)
    entry = _fingerprint(config)

    got = cli._begin_check_attempt(config, tel)

    assert isinstance(got, cli._AttemptOnRecord)
    row = _attempt(config.telemetry_path)
    assert row["attempt_id"] == got.attempt_id
    _assert_started(row, entry)


def test_the_unrecorded_causes_are_a_closed_set() -> None:
    assert cli._ATTEMPT_UNRECORDED_CAUSES == (
        "telemetry_unavailable", "fingerprint_unreadable", "write_failed")


def test_a_seam_fault_leaves_started(workspace, monkeypatch, capsys) -> None:
    """Criterion 15: the outcome rides the seam's transaction; a fault lands none of it."""
    config, store, tel = workspace
    _a, _b, nbhds = _pair(store)
    _drain_outbox(store)
    entry = _fingerprint(config)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel))
    monkeypatch.setattr(telemetry, "_UPSERT_COVERAGE_SQL", "NOT SQL")

    code, obj = _run(config, capsys)

    assert code == 2 and obj["summary_row_written"] is False
    assert _read_check_runs(config) == []
    assert _coverage(config.telemetry_path) == {}
    _assert_started(_attempt(config.telemetry_path), entry)


# =========================================================================== #
# Phase 3b — the refused spend, and the unrecorded attempt
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class _FailingRefusalTelemetry:
    """Delegates to a real store; only ``record_attempt_refusal`` raises ``exc``."""

    def __init__(self, inner: TelemetryStore, exc: Exception) -> None:
        self._inner = inner
        self._exc = exc

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def record_attempt_refusal(self, refusal: AttemptRefusal) -> None:
        raise self._exc


def _refusal(**overrides: Any) -> AttemptRefusal:
    base = dict(attempt_id="att-1", refused_at="2026-09-21T00:02:00.000000+00:00",
                batches_planned=3)
    base.update(overrides)
    return AttemptRefusal(**base)


def _unrecorded(cause: str) -> "cli._AttemptUnrecorded":
    return cli._AttemptUnrecorded(cause)


def _refusable_pair(config: MitosConfig, store: GraphStore, monkeypatch: Any) -> list:
    """A pair planning one fresh group, a refusing threshold, both seams; returns ``invoked``."""
    _a, _b, nbhds = _pair(store)
    _drain_outbox(store)
    _wire_substrate(monkeypatch, nbhds)
    monkeypatch.setattr(check, "CHECK_CONFIRM_BATCHES", 0)
    return _wire_judge(monkeypatch, None)


# --------------------------------------------------------------------------- #
# 1–3 — the refusal's boundary shape and round trip
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("overrides", [
    {"attempt_id": ""}, {"refused_at": ""}, {"batches_planned": 0},
    {"batches_planned": -1}, {"batches_planned": True}, {"batches_planned": "3"},
], ids=["empty-id", "empty-time", "zero", "negative", "bool", "string"])
def test_a_malformed_refusal_is_refused_before_any_connection(monkeypatch,
                                                               overrides) -> None:
    """Criterion 1: the shape refuses at construction, so nothing opens."""
    opened = []
    monkeypatch.setattr(telemetry, "open_connection",
                        lambda *a, **k: opened.append(a) or pytest.fail("opened"))
    with pytest.raises(ValueError, match="AttemptRefusal"):
        _refusal(**overrides)
    assert opened == []


def test_a_refusal_round_trips_through_the_reader(tmp_path) -> None:
    """Criterion 2: start → refusal → the reader returns exactly the refusal record."""
    path = str(tmp_path / "telemetry.sqlite")
    store = TelemetryStore(path)
    start = _start(fingerprint="e" * 64)
    store.record_attempt_start(start)
    refusal = _refusal(attempt_id=start.attempt_id, batches_planned=4)

    store.record_attempt_refusal(refusal)

    assert read_last_attempt(path) == LastAttempt(
        attempt_id=start.attempt_id, started_at=start.started_at,
        fingerprint=start.fingerprint, state=ATTEMPT_SPEND_NOT_AUTHORIZED,
        run_id=None, outcome_at=refusal.refused_at, degradation_tokens=None,
        new_pairs=None, findings_known=None, batches_planned=4,
    )


def test_the_refusal_state_is_not_a_run_end_state() -> None:
    """Criterion 3: the run-end seam can never write it (the refused-state row above)."""
    assert ATTEMPT_SPEND_NOT_AUTHORIZED == "spend_not_authorized"
    assert ATTEMPT_SPEND_NOT_AUTHORIZED not in ATTEMPT_RUN_END_STATES
    with pytest.raises(ValueError, match="run-end state"):
        _outcome(state=ATTEMPT_SPEND_NOT_AUTHORIZED)


# --------------------------------------------------------------------------- #
# 5–7 — who the refusal write reaches, and its faults
# --------------------------------------------------------------------------- #

def test_a_refusal_leaves_a_replaced_record_untouched(workspace, monkeypatch,
                                                      capsys) -> None:
    """Criterion 5: a newer attempt replaced the row before the refusal → it stays as is."""
    config, store, tel = workspace
    _refusable_pair(config, store, monkeypatch)
    original = cli._confirm_spend

    def _replacing_confirm(n: int, **kw: Any) -> Optional[int]:
        # Strictly between the entry write and the refusal write.
        TelemetryStore(config.telemetry_path).record_attempt_start(
            _start(attempt_id="newer", fingerprint="n" * 64))
        return original(n, **kw)

    monkeypatch.setattr(cli, "_confirm_spend", _replacing_confirm)

    code, _planned, _out, _err = _refuse(config, capsys, monkeypatch, "json")

    assert code == 2
    row = _attempt(config.telemetry_path)
    assert row == {**dict.fromkeys(_ATTEMPT_COLUMNS), "slot": 1, "attempt_id": "newer",
                   "started_at": _start().started_at, "fingerprint": "n" * 64,
                   "state": ATTEMPT_STARTED}


@pytest.mark.parametrize("scope, threshold", [("x", 0), ("", -1)],
                         ids=["scope-x", "scope-empty"])
def test_a_scoped_refusal_writes_nothing(workspace, monkeypatch, capsys, scope: str,
                                         threshold: int) -> None:
    """Criterion 6: a scoped refusal leaves a seeded record byte-identical, says no clause.

    ``--scope ""`` matches no decision, so it plans no batch; a negative threshold
    makes zero batches refuse, which is the only way to reach its refusal.
    """
    config, store, tel = workspace
    a_axiom, b_axiom = "Scoped refusal axiom alpha.", "Scoped refusal axiom beta."
    _commit(store, "scoped-a", a_axiom, scope=["x"])
    _commit(store, "scoped-b", b_axiom, scope=["x"])
    _drain_outbox(store)
    _wire_substrate(monkeypatch, {a_axiom: [_match("scoped-b", 0.9)], b_axiom: []})
    invoked = _wire_judge(monkeypatch, None)
    monkeypatch.setattr(check, "CHECK_CONFIRM_BATCHES", threshold)
    tel.record_attempt_start(_start(fingerprint="s" * 64))
    tel.record_run_end(
        _check_run_row("seeded-run"), coverage=None,
        attempt=_outcome(run_id="seeded-run", new_pairs=(("p", "q"),),
                         degradation_tokens=("judgment",),
                         state=ATTEMPT_COULD_NOT_COMPLETE, findings_known=3))
    _raw_attempt_update(config.telemetry_path,
                        "UPDATE check_attempt SET batches_planned = ?", 9)
    seeded = _attempt(config.telemetry_path)
    assert all(v is not None for v in seeded.values())

    for form in ("json", "text"):
        code, _planned, out, err = _refuse(config, capsys, monkeypatch, form,
                                           scope=scope)
        assert code == 2 and invoked == []
        assert _attempt(config.telemetry_path) == seeded
        assert cli._ATTEMPT_ON_RECORD_CLAUSE not in out + err
        if form == "json":
            assert json.loads(out)["attempt_unrecorded"] is None


@pytest.mark.parametrize("exc", [DatabaseError("provoked refusal-write fault"),
                                 RuntimeError("a bug in the refusal write")],
                         ids=["database-error", "runtime-error"])
def test_a_refusal_write_fault_changes_nothing_the_caller_reads(workspace, monkeypatch,
                                                                capsys, exc) -> None:
    """Criterion 7: exit 2, one JSON object, the row stays ``started``; a bug warns once."""
    config, store, tel = workspace
    _refusable_pair(config, store, monkeypatch)
    entry = _fingerprint(config)
    faulted = _FailingRefusalTelemetry(TelemetryStore(config.telemetry_path), exc)
    monkeypatch.setattr(cli, "_build_check_telemetry", lambda c: faulted)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=True)
    captured = capsys.readouterr()

    assert code == 2
    obj = json.loads(captured.out)  # the whole buffer is exactly one object
    assert obj["code"] == "confirmation_required"
    # The clause rests on the entry write, which landed.
    assert cli._ATTEMPT_ON_RECORD_CLAUSE in obj["error"]
    _assert_started(_attempt(config.telemetry_path), entry)
    if isinstance(exc, DatabaseError):
        assert captured.err == ""
    else:
        lines = captured.err.splitlines()
        assert len(lines) == 1
        assert lines[0].startswith("[Warning]") and "a bug in the refusal write" in lines[0]


# --------------------------------------------------------------------------- #
# 8–11 — the unrecorded attempt is said on every answer
# --------------------------------------------------------------------------- #

def test_every_unrecorded_cause_has_words_that_name_the_way_past() -> None:
    """Criterion 8: the words map is closed over the causes; each message names the exit."""
    assert cli._ATTEMPT_UNRECORDED_WORDS.keys() == set(cli._ATTEMPT_UNRECORDED_CAUSES)
    for cause in cli._ATTEMPT_UNRECORDED_CAUSES:
        got = cli._attempt_unrecorded_json(_unrecorded(cause))
        assert got["cause"] == cause
        assert cli._ATTEMPT_UNRECORDED_WORDS[cause] in got["message"]
        assert "git commit --no-verify" in got["message"]
        assert cli._attempt_unrecorded_line(_unrecorded(cause)) == got["message"]
    for attempt in (None, cli._AttemptOnRecord("att-1")):
        assert cli._attempt_unrecorded_json(attempt) is None
        assert cli._attempt_unrecorded_line(attempt) is None


def _telemetry_none_run(config: MitosConfig, store: GraphStore, monkeypatch: Any,
                        as_json: bool, capsys: Any) -> Tuple[int, str, str]:
    """A completed run over no telemetry: the entry write is ``telemetry_unavailable``."""
    _a, _b, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, None))
    monkeypatch.setattr(cli, "_build_check_telemetry", lambda c: None)
    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False,
                         as_json=as_json)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_the_report_says_an_unrecorded_attempt_after_the_history_note(
        workspace, monkeypatch, capsys) -> None:
    """Criterion 9 (text): the disclosure is the line right after the check-history note."""
    config, store, _tel = workspace
    code, out, _err = _telemetry_none_run(config, store, monkeypatch, False, capsys)

    assert code == 2
    lines = out.splitlines()
    (note_at,) = [i for i, line in enumerate(lines)
                  if "not recorded to check history" in line]
    expected = cli._attempt_unrecorded_line(_unrecorded("telemetry_unavailable"))
    assert lines[note_at + 1].strip() == expected


def test_the_json_carries_an_unrecorded_attempt_beside_the_row_flag(
        workspace, monkeypatch, capsys) -> None:
    """Criterion 9 (``--json``): the key names the cause; its message is the text line."""
    config, store, _tel = workspace
    code, out, _err = _telemetry_none_run(config, store, monkeypatch, True, capsys)

    assert code == 2
    obj = json.loads(out)
    assert obj["summary_row_written"] is False
    assert obj["attempt_unrecorded"] == cli._attempt_unrecorded_json(
        _unrecorded("telemetry_unavailable"))


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_substrate_unavailable_says_an_unrecorded_attempt(workspace, monkeypatch,
                                                          capsys, as_json) -> None:
    """Criterion 10: no embedding provider, no telemetry → the exit carries the disclosure."""
    config, store, _tel = workspace
    _pair(store)
    _drain_outbox(store)
    monkeypatch.setattr(cli, "_build_check_telemetry", lambda c: None)
    expected = cli._attempt_unrecorded_json(_unrecorded("telemetry_unavailable"))

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False,
                         as_json=as_json)
    captured = capsys.readouterr()

    assert code == 2
    if as_json:
        obj = json.loads(captured.out)
        assert obj["code"] == "substrate_unavailable"
        assert obj["attempt_unrecorded"] == expected
    else:
        lines = captured.err.splitlines()
        assert lines[-2].startswith("check could not run: cannot audit")
        assert lines[-1] == expected["message"]


@pytest.mark.parametrize("form", ["text", "json", "tty"])
def test_a_refusal_says_an_unrecorded_attempt_instead_of_the_clause(
        workspace, monkeypatch, capsys, form) -> None:
    """Criterion 10: a refusal over no telemetry swaps the on-record clause for the disclosure."""
    config, store, _tel = workspace
    invoked = _refusable_pair(config, store, monkeypatch)
    monkeypatch.setattr(cli, "_build_check_telemetry", lambda c: None)
    expected = cli._attempt_unrecorded_json(_unrecorded("telemetry_unavailable"))

    code, _planned, out, err = _refuse(config, capsys, monkeypatch, form)

    assert code == 2 and invoked == []
    assert cli._ATTEMPT_ON_RECORD_CLAUSE not in out + err
    if form == "json":
        assert json.loads(out)["attempt_unrecorded"] == expected
    elif form == "text":
        assert err.splitlines()[-1] == expected["message"]
    else:
        lines = out.splitlines()
        assert lines[-2] == "Aborted — nothing spent."
        assert lines[-1] == expected["message"]


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_check_faulted_says_an_unrecorded_attempt(workspace, monkeypatch, capsys,
                                                  as_json) -> None:
    """Criterion 10: a store fault after an entry write that failed → the disclosure."""
    config, store, tel = workspace
    _a, _b, nbhds = _pair(store)
    _drain_outbox(store)
    _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, None)
    monkeypatch.setattr(cli, "GraphStore", lambda path: _FaultStore(GraphStore(path)))
    faulted = _FailingStartTelemetry(TelemetryStore(config.telemetry_path),
                                     DatabaseError("provoked entry-write fault"))
    monkeypatch.setattr(cli, "_build_check_telemetry", lambda c: faulted)
    expected = cli._attempt_unrecorded_json(_unrecorded("write_failed"))

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False,
                         as_json=as_json)
    captured = capsys.readouterr()

    assert code == 2
    if as_json:
        obj = json.loads(captured.out)
        assert obj["code"] == "check_faulted"
        assert obj["attempt_unrecorded"] == expected
    else:
        lines = captured.err.splitlines()
        assert lines[-2].startswith("check could not run:")
        assert lines[-1] == expected["message"]


def test_an_entry_write_fault_is_named_on_json(workspace, monkeypatch, capsys) -> None:
    """Criterion 10: the silent entry-write fault of 3a's row 14 now carries the key."""
    config, store, tel = workspace
    _a, _b, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel))
    faulted = _FailingStartTelemetry(tel, DatabaseError("provoked entry-write fault"))
    monkeypatch.setattr(cli, "_build_check_telemetry", lambda config: faulted)

    code, obj = _run(config, capsys)

    assert code == 0
    assert obj["attempt_unrecorded"]["cause"] == "write_failed"


def test_clean_and_scoped_runs_carry_null_and_say_nothing(workspace, monkeypatch,
                                                          capsys) -> None:
    """Criterion 11: an entry write that landed, and a scoped run, disclose nothing."""
    config, store, tel = workspace
    _a, _b, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)

    for scope in (None, "no-such-tag"):
        _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel))
        code, obj = _run(config, capsys, scope=scope)
        assert code == 0 and obj["attempt_unrecorded"] is None

        _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel))
        code = cli.cmd_check(config, scope=scope, fresh=False, assume_yes=False,
                             as_json=False)
        captured = capsys.readouterr()
        assert code == 0
        assert "was not recorded" not in captured.out + captured.err


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_a_graph_that_cannot_open_made_no_attempt(workspace, monkeypatch, capsys,
                                                  as_json) -> None:
    """Criterion 11: ``GraphStore`` raising is no attempt at all → null, no line."""
    config, _store, _tel = workspace

    def _raising(path: str) -> Any:
        raise DatabaseError("provoked graph-open fault")

    monkeypatch.setattr(cli, "GraphStore", _raising)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False,
                         as_json=as_json)
    captured = capsys.readouterr()

    assert code == 2
    if as_json:
        obj = json.loads(captured.out)
        assert obj["code"] == "check_faulted" and obj["attempt_unrecorded"] is None
    else:
        assert "was not recorded" not in captured.err
        assert captured.err.splitlines()[-1].startswith("check could not run:")


# =========================================================================== #
# Phase 3c1 — the predicate, the shared key test, the block code, `hook-run`
# =========================================================================== #
#
# Every pass row is a transition: it starts from a fixture verified ``blocked``,
# changes the one thing the row names, and asserts the new row by name on the
# verdict. Rows 2–4 look the same from outside (silence, exit 0), so a row that
# asserted only "passed" would prove nothing. The judge key is a dummy string
# nothing ever sends: every ``cmd_check`` that reaches judging wires its judge.

_DUMMY_KEY = "sk-dummy-never-sent"


def _keyed(config: MitosConfig, monkeypatch: Any, key: str = _DUMMY_KEY) -> MitosConfig:
    """A fresh config for the same workspace with a judge key in its env.

    ``config.env`` is resolved at construction, so the key only counts on a config
    built after it is set. The project name is carried so recipes keep it.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    return MitosConfig(config.workspace_dir, project=config.project)


def _keyless(config: MitosConfig, monkeypatch: Any) -> MitosConfig:
    """A fresh config for the same workspace with no judge key anywhere."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return MitosConfig(config.workspace_dir, project=config.project)


def _blocked_pair(config: MitosConfig, store: GraphStore,
                  monkeypatch: Any) -> Tuple[str, str, Dict[str, Any], MitosConfig]:
    """Two committed decisions, telemetry at head, no attempt; asserts ``blocked``.

    Returns the pair's ids, their neighbourhoods and the keyed config.
    """
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    keyed = _keyed(config, monkeypatch)
    assert evaluate_gate(keyed) == GateVerdict(GATE_BLOCKED, uncovered=2)
    return a_id, b_id, nbhds, keyed


def _corrupt(path: str) -> None:
    """Overwrites a store file with bytes SQLite cannot read, dropping WAL siblings."""
    for sibling in (path + "-wal", path + "-shm"):
        if os.path.exists(sibling):
            os.remove(sibling)
    with open(path, "wb") as f:
        f.write(b"this is not a sqlite database" * 64)


def _tree(root: str) -> list:
    """Every path under ``root``, relative and sorted."""
    return sorted(
        os.path.relpath(os.path.join(d, name), root)
        for d, dirs, files in os.walk(root) for name in dirs + files
    )


# --------------------------------------------------------------------------- #
# 1–2 — the key test
# --------------------------------------------------------------------------- #

def test_the_key_test_reads_the_workspace_env(workspace, monkeypatch) -> None:
    """Criterion 1: the key when set; ``None`` when unset or exported empty."""
    config, _store, _tel = workspace

    assert judge_api_key(_keyed(config, monkeypatch)) == _DUMMY_KEY
    assert judge_api_key(_keyless(config, monkeypatch)) is None
    assert judge_api_key(_keyed(config, monkeypatch, key="")) is None


@pytest.mark.parametrize("key", [None, "", "   ", _DUMMY_KEY],
                         ids=["unset", "empty", "whitespace", "dummy"])
def test_check_and_the_gate_agree_on_keyless(workspace, monkeypatch, key) -> None:
    """Criterion 2: ``_build_check_judge`` is ``None`` exactly when the key test is.

    The real builder, unstubbed: a dummy key builds a real client offline and sends
    nothing. The whitespace case is what a second spelling in the builder (a
    ``.strip()``) would split on.
    """
    config, _store, _tel = workspace
    target = (_keyless(config, monkeypatch) if key is None
              else _keyed(config, monkeypatch, key=key))

    assert (cli._build_check_judge(target) is None) == (judge_api_key(target) is None)


# --------------------------------------------------------------------------- #
# 3–10 — the predicate rows as transitions
# --------------------------------------------------------------------------- #

def test_uncovered_decisions_with_no_attempt_block(workspace, monkeypatch) -> None:
    """Criterion 3, row 5: two committed decisions and no attempt → ``blocked``, N = 2."""
    config, store, _tel = workspace
    _blocked_pair(config, store, monkeypatch)
    assert read_last_attempt(config.telemetry_path) is None


def test_a_keyless_workspace_is_inactive_and_opens_no_store(workspace, monkeypatch) -> None:
    """Criterion 4, row 2: removing the key → ``keyless``, even over a corrupt graph.

    The corrupt graph is D2's proof: were the key asked after the stores are read,
    it would read ``unreadable`` instead.
    """
    config, store, _tel = workspace
    _blocked_pair(config, store, monkeypatch)

    assert evaluate_gate(_keyless(config, monkeypatch)) == GateVerdict(GATE_KEYLESS)

    _corrupt(config.db_path)
    assert evaluate_gate(_keyless(config, monkeypatch)) == GateVerdict(GATE_KEYLESS)
    assert evaluate_gate(_keyed(config, monkeypatch)).row == GATE_UNREADABLE


def test_a_clean_run_leaves_nothing_uncovered(workspace, monkeypatch, capsys) -> None:
    """Criterion 5, row 3: a clean unscoped check covers both → ``nothing_uncovered``."""
    config, store, tel = workspace
    _a, _b, nbhds, keyed = _blocked_pair(config, store, monkeypatch)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel))

    code, obj = _run(config, capsys)

    assert code == 0 and obj["degradations"] == []
    assert evaluate_gate(keyed) == GateVerdict(GATE_NOTHING_UNCOVERED, uncovered=0)


def test_an_empty_corpus_has_nothing_uncovered(workspace, monkeypatch) -> None:
    """Criterion 5: a built graph with zero decisions lands on row 3 directly."""
    config, _store, _tel = workspace
    assert evaluate_gate(_keyed(config, monkeypatch)) == GateVerdict(
        GATE_NOTHING_UNCOVERED, uncovered=0)


def test_a_decision_recorded_mid_run_blocks_the_next_commit(workspace, monkeypatch,
                                                             capsys) -> None:
    """Criterion 6: the attempt's fingerprint is the set at entry, not the set now.

    The run covers what it swept and leaves exactly the new decision uncovered, so
    the recorded attempt no longer matches and the gate blocks on one. (Row 4 is
    reachable with N > 0 only through an attempt that covered nothing, which is why
    the next row carries its proof.)
    """
    config, store, tel = workspace
    _a, _b, nbhds, keyed = _blocked_pair(config, store, monkeypatch)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    judge = _CommittingJudge(_judge_for(store, embed, vector, tel), store)
    _wire_judge(monkeypatch, judge)

    code, _obj = _run(config, capsys)

    assert code in (0, 1)
    assert read_last_attempt(config.telemetry_path).state in ATTEMPT_RUN_END_STATES
    assert evaluate_gate(keyed) == GateVerdict(GATE_BLOCKED, uncovered=1)


def _substrate_unavailable(config, store, tel, monkeypatch, capsys, nbhds) -> None:
    # No substrate patch: real construction has no embedding provider offline → exit 2
    # before the run end, so the record stays ``started``.
    code, obj = _run(config, capsys)
    assert code == 2 and obj["code"] == "substrate_unavailable"


def _refused(config, store, tel, monkeypatch, capsys, nbhds) -> None:
    _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, None)
    monkeypatch.setattr(check, "CHECK_CONFIRM_BATCHES", 0)
    code, _planned, _out, _err = _refuse(config, capsys, monkeypatch, "json")
    assert code == 2


def _failing_batch(config, store, tel, monkeypatch, capsys, nbhds) -> None:
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    failing = Unavailable(reason=ConflictUnavailableReason.JUDGMENT, detail="judge died")
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel, overrides={0: failing}))
    code, _obj = _run(config, capsys)
    assert code == 2


_FAILED_ATTEMPTS = {
    ATTEMPT_STARTED: _substrate_unavailable,
    ATTEMPT_SPEND_NOT_AUTHORIZED: _refused,
    ATTEMPT_COULD_NOT_COMPLETE: _failing_batch,
}


@pytest.mark.parametrize("state", list(_FAILED_ATTEMPTS))
def test_a_failed_attempt_opens_the_gate(workspace, monkeypatch, capsys, state) -> None:
    """Criterion 7, row 4: an attempt that covered nothing opens the gate, whatever its state.

    ``started`` never reached its run end and ``spend_not_authorized`` is not a run-end
    state at all; both open it, because the gate answers forgetting, not failure.
    """
    config, store, tel = workspace
    _a, _b, nbhds, keyed = _blocked_pair(config, store, monkeypatch)

    _FAILED_ATTEMPTS[state](config, store, tel, monkeypatch, capsys, nbhds)

    assert read_last_attempt(config.telemetry_path).state == state
    assert evaluate_gate(keyed) == GateVerdict(GATE_ATTEMPTED, uncovered=2)


def test_a_record_after_an_attempt_blocks_again(workspace, monkeypatch, capsys) -> None:
    """Criterion 8: from ``attempted``, one more decision → ``blocked`` over N + 1."""
    config, store, tel = workspace
    _a, _b, nbhds, keyed = _blocked_pair(config, store, monkeypatch)
    _substrate_unavailable(config, store, tel, monkeypatch, capsys, nbhds)
    assert evaluate_gate(keyed) == GateVerdict(GATE_ATTEMPTED, uncovered=2)

    _commit(store, "gate-late", "A decision recorded after the attempt.")

    assert evaluate_gate(keyed) == GateVerdict(GATE_BLOCKED, uncovered=3)


def test_a_scoped_run_shrinks_the_set_but_is_no_attempt(workspace, monkeypatch,
                                                        capsys) -> None:
    """Criterion 9: ``--scope x`` covers what it swept and writes no attempt → still blocked."""
    config, store, tel = workspace
    _x1, _x2, _y1, nbhds = _scoped_corpus(store)
    _drain_outbox(store)
    keyed = _keyed(config, monkeypatch)
    assert evaluate_gate(keyed) == GateVerdict(GATE_BLOCKED, uncovered=3)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel, scope="x"))

    code, obj = _run(config, capsys, scope="x")

    assert code == 0
    swept = obj["nodes_swept"]
    assert 0 < swept < 3
    assert read_last_attempt(config.telemetry_path) is None
    assert evaluate_gate(keyed) == GateVerdict(GATE_BLOCKED, uncovered=3 - swept)


def test_an_excluded_node_does_not_hold_the_count_up(workspace, monkeypatch,
                                                     capsys) -> None:
    """Criterion 10: a clean run that excludes one poison node → ``nothing_uncovered``."""
    config, store, tel = workspace
    a_id, _b, nbhds, keyed = _blocked_pair(config, store, monkeypatch)
    _poison(store, a_id)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel))

    code, _obj = _run(config, capsys)

    assert code == 0
    assert evaluate_gate(keyed) == GateVerdict(GATE_NOTHING_UNCOVERED, uncovered=0)


# --------------------------------------------------------------------------- #
# 11–14 — row 1, and absent is not unreadable
# --------------------------------------------------------------------------- #

def test_no_graph_is_unreadable_and_creates_nothing(tmp_path, monkeypatch) -> None:
    """Criterion 11: no ``graph.sqlite`` → ``unreadable`` / ``no_graph``; the tree is unchanged."""
    keyed = _keyed(MitosConfig(str(tmp_path)), monkeypatch)
    before = _tree(str(tmp_path))

    assert evaluate_gate(keyed) == GateVerdict(GATE_UNREADABLE, cause="no_graph")
    assert _tree(str(tmp_path)) == before


@pytest.mark.parametrize("store_path, cause", [
    ("db_path", "graph"), ("telemetry_path", "telemetry"),
])
def test_an_unreadable_store_lets_the_commit_through(workspace, monkeypatch,
                                                     store_path: str, cause: str) -> None:
    """Criterion 12: a corrupt graph or telemetry file → ``unreadable`` naming it."""
    config, store, _tel = workspace
    _blocked_pair(config, store, monkeypatch)

    _corrupt(getattr(config, store_path))

    assert evaluate_gate(_keyed(config, monkeypatch)) == GateVerdict(
        GATE_UNREADABLE, cause=cause)


def test_an_unreadable_attempt_record_is_row_one_not_a_block(workspace,
                                                             monkeypatch) -> None:
    """Criterion 13 (D3): coverage reads fine, the attempt row does not → row 1, never 5."""
    config, store, tel = workspace
    _a, _b, _nbhds, keyed = _blocked_pair(config, store, monkeypatch)
    tel.record_attempt_start(_start(fingerprint=_fingerprint(config)))
    _raw_attempt_update(config.telemetry_path,
                        "UPDATE check_attempt SET new_pairs = ?", "not json at all")

    assert evaluate_gate(keyed) == GateVerdict(GATE_UNREADABLE, cause="telemetry")


def _graph_only_workspace(tmp_path: Any) -> MitosConfig:
    """A workspace with two committed decisions and no telemetry file."""
    config = MitosConfig(str(tmp_path))
    store = GraphStore(config.db_path)
    _commit(store, "gate-a", "Graph-only axiom alpha.")
    _commit(store, "gate-b", "Graph-only axiom beta.")
    return config


def test_a_telemetry_file_below_the_attempt_rung_blocks(tmp_path, monkeypatch) -> None:
    """Criterion 14: a rung-6 file holds coverage but no attempt → ``blocked``, N = M.

    The first commit after an upgrade. The file's ``user_version`` is unchanged.
    """
    config = _graph_only_workspace(tmp_path)
    conn = open_connection(config.telemetry_path)
    run_migrations(conn, TELEMETRY_MIGRATION_STEPS[:FAILED_BATCHES_RUNG])
    conn.close()

    assert evaluate_gate(_keyed(config, monkeypatch)) == GateVerdict(
        GATE_BLOCKED, uncovered=2)
    conn = sqlite3.connect(config.telemetry_path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == FAILED_BATCHES_RUNG
    finally:
        conn.close()


def test_no_telemetry_file_blocks_and_creates_none(tmp_path, monkeypatch) -> None:
    """Criterion 14: no telemetry file → ``blocked``, N = M, and still no file afterwards."""
    config = _graph_only_workspace(tmp_path)
    assert not os.path.exists(config.telemetry_path)

    assert evaluate_gate(_keyed(config, monkeypatch)) == GateVerdict(
        GATE_BLOCKED, uncovered=2)
    assert not os.path.exists(config.telemetry_path)


# --------------------------------------------------------------------------- #
# 15–19 — the handler
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("extra, subject", [
    (False, "2 decisions have"), (True, "3 decisions have"),
], ids=["two", "three"])
def test_a_block_echoes_then_names_the_count_and_the_check(workspace, monkeypatch, capsys,
                                                           extra: bool, subject: str) -> None:
    """Criterion 15: exit ``HOOK_BLOCK_EXIT``; nothing on stdout; stderr is the echo, then the message."""
    config, store, _tel = workspace
    _a, _b, _nbhds, keyed = _blocked_pair(config, store, monkeypatch)
    if extra:
        _commit(store, "gate-third", "A third uncovered decision.")

    code = cli.cmd_hook_run(keyed)
    out, err = capsys.readouterr()

    assert code == HOOK_BLOCK_EXIT
    assert out == ""
    lines = err.splitlines()
    assert lines[0] == provenance_line(keyed)
    assert len(lines) == 2
    assert lines[1].startswith(f"{subject} not been covered by a contradiction check. Run `mitos check -p ")
    assert "the gate asks only that the check was attempted" in lines[1]


def test_a_block_over_one_decision_is_singular(tmp_path, monkeypatch, capsys) -> None:
    """Criterion 15: one uncovered decision reads "1 decision has"."""
    config = MitosConfig(str(tmp_path))
    _commit(GraphStore(config.db_path), "gate-only", "The only decision.")
    keyed = _keyed(config, monkeypatch)

    assert cli.cmd_hook_run(keyed) == HOOK_BLOCK_EXIT
    err = capsys.readouterr().err
    assert "1 decision has not been covered" in err
    assert "1 decisions" not in err


def test_the_block_recipe_parses_to_check_on_this_workspace(recipe_workspace, monkeypatch,
                                                            capsys) -> None:
    """Criterion 16: the recipe parses through the real parser to ``check`` with the selector."""
    config, store, _tel = recipe_workspace
    _pair(store)
    keyed = _keyed(config, monkeypatch)

    assert cli.cmd_hook_run(keyed) == HOOK_BLOCK_EXIT
    recipe = _recipe(capsys.readouterr().err, "not been covered")
    _assert_parses(recipe, "check", config.project)


def _to_keyless(config, store, tel, monkeypatch, capsys, nbhds) -> MitosConfig:
    return _keyless(config, monkeypatch)


def _to_nothing_uncovered(config, store, tel, monkeypatch, capsys, nbhds) -> MitosConfig:
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel))
    assert _run(config, capsys)[0] == 0
    return _keyed(config, monkeypatch)


def _to_attempted(config, store, tel, monkeypatch, capsys, nbhds) -> MitosConfig:
    _substrate_unavailable(config, store, tel, monkeypatch, capsys, nbhds)
    return _keyed(config, monkeypatch)


_QUIET_PASSES = {
    GATE_KEYLESS: _to_keyless,
    GATE_NOTHING_UNCOVERED: _to_nothing_uncovered,
    GATE_ATTEMPTED: _to_attempted,
}


@pytest.mark.parametrize("row", list(_QUIET_PASSES))
def test_the_quiet_passes_print_nothing(workspace, monkeypatch, capsys, row: str) -> None:
    """Criterion 17: from ``blocked``, each quiet pass exits 0 with both channels empty."""
    config, store, tel = workspace
    _a, _b, nbhds, keyed = _blocked_pair(config, store, monkeypatch)
    assert cli.cmd_hook_run(keyed) == HOOK_BLOCK_EXIT
    capsys.readouterr()

    moved = _QUIET_PASSES[row](config, store, tel, monkeypatch, capsys, nbhds)
    capsys.readouterr()
    assert evaluate_gate(moved).row == row

    assert cli.cmd_hook_run(moved) == 0
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("cause", GATE_CAUSES)
def test_an_unreadable_row_says_one_line_and_passes(workspace, tmp_path, monkeypatch,
                                                    capsys, cause: str) -> None:
    """Criterion 18: each row-1 cause → exit 0, stdout empty, exactly one stderr line."""
    config, store, tel = workspace
    if cause == "no_graph":
        target = _keyed(MitosConfig(str(tmp_path)), monkeypatch)
    else:
        _blocked_pair(config, store, monkeypatch)
        _corrupt(config.db_path if cause == "graph" else config.telemetry_path)
        target = _keyed(config, monkeypatch)
    assert evaluate_gate(target) == GateVerdict(GATE_UNREADABLE, cause=cause)

    code = cli.cmd_hook_run(target)
    out, err = capsys.readouterr()

    assert code == 0 and out == ""
    assert len(err.splitlines()) == 1 and err.endswith("\n")
    assert "let through" in err and "`" not in err


def test_an_unreadable_attempt_record_says_one_line_too(workspace, monkeypatch,
                                                        capsys) -> None:
    """Criterion 18: the lazily read attempt's fault (D3) is the telemetry line."""
    config, store, tel = workspace
    _a, _b, _nbhds, keyed = _blocked_pair(config, store, monkeypatch)
    tel.record_attempt_start(_start(fingerprint=_fingerprint(config)))
    _raw_attempt_update(config.telemetry_path,
                        "UPDATE check_attempt SET new_pairs = ?", "not json at all")

    assert cli.cmd_hook_run(keyed) == 0
    out, err = capsys.readouterr()
    assert out == "" and err == cli._HOOK_UNREADABLE_LINES["telemetry"] + "\n"


def test_the_block_code_is_one_nothing_else_produces() -> None:
    """Criterion 19: ``HOOK_BLOCK_EXIT`` is outside every status another cause can produce.

    The installed hook maps this status alone to a failed commit, so it must never
    be something a crash or the shell could return:

    * 0 — success;
    * 1 — Python's uncaught exception, and ``main()``'s fault arms;
    * 2 — argparse's usage error (and the interpreter's own);
    * 120 — the interpreter when flushing stdout fails at exit;
    * 124–127 — ``timeout``, ``env``, and the shell's not-executable / not-found;
    * 128–255 — death by signal.
    """
    excluded = {0, 1, 2, 120} | set(range(124, 128)) | set(range(128, 256))
    assert type(HOOK_BLOCK_EXIT) is int
    assert 0 <= HOOK_BLOCK_EXIT <= 255
    assert HOOK_BLOCK_EXIT not in excluded


def test_the_rows_are_a_closed_set_in_table_order() -> None:
    """The five row names, in the vision's table order; ``blocks`` is row 5 alone."""
    assert GATE_ROWS == ("unreadable", "keyless", "nothing_uncovered", "attempted",
                         "blocked")
    assert [GateVerdict(r).blocks for r in GATE_ROWS] == [False] * 4 + [True]


# --------------------------------------------------------------------------- #
# 20 — through main(), in a subprocess: no row loads an LLM SDK
# --------------------------------------------------------------------------- #

_HOOK_PROBE = """
import json, sys
sys.argv = ["mitos", "hook-run", "-p", {ws!r}]
from mitos.cli import main
try:
    main()
    code = 0
except SystemExit as e:
    code = e.code
print(json.dumps([code, sorted(m for m in ("anthropic", "google.genai") if m in sys.modules)]))
"""


def _probe_hook(ws: str, *, keyed: bool) -> Tuple[int, list]:
    """Runs ``mitos hook-run -p <ws>`` through ``main()`` in a fresh interpreter.

    Returns the exit status and which LLM SDKs were imported. The environment is
    this test's own (hermetic ``XDG_*``, credentials stripped) plus, when ``keyed``,
    a dummy judge key.
    """
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env["MITOS_NO_UPDATE_CHECK"] = "1"
    if keyed:
        env["ANTHROPIC_API_KEY"] = _DUMMY_KEY
    out = subprocess.run([sys.executable, "-c", _HOOK_PROBE.format(ws=ws)],
                         capture_output=True, text=True, env=env, check=True)
    code, sdks = json.loads(out.stdout.strip().splitlines()[-1])
    return code, sdks


def test_hook_run_loads_no_llm_sdk_on_any_row(tmp_path, capsys) -> None:
    """Criterion 20 (D6): rows 3, 2, 5, 4 through the real verb; no SDK on any of them.

    One ``init``'d workspace moved between states. A block must not buy the SDK
    either. In-process this would be vacuous: the test process has imported both.
    """
    ws = str(tmp_path / "hookws")
    os.makedirs(ws)
    config = MitosConfig(ws)
    cmd_init(config)
    capsys.readouterr()
    config = MitosConfig(ws)

    assert _probe_hook(ws, keyed=True) == (0, [])          # row 3: empty corpus

    store = GraphStore(config.db_path)
    _commit(store, "probe-a", "Probe axiom alpha.")
    _commit(store, "probe-b", "Probe axiom beta.")
    assert _probe_hook(ws, keyed=False) == (0, [])         # row 2: keyless
    assert _probe_hook(ws, keyed=True) == (HOOK_BLOCK_EXIT, [])  # row 5

    TelemetryStore(config.telemetry_path).record_attempt_start(
        _start(fingerprint=_fingerprint(config)))
    assert _probe_hook(ws, keyed=True) == (0, [])          # row 4


# --------------------------------------------------------------------------- #
# 22 — constraint 12: `--staged` finds nothing pending after an MCP-path record
# --------------------------------------------------------------------------- #

def test_staged_sees_nothing_after_an_mcp_path_record(tmp_path, monkeypatch, capsys) -> None:
    """Criterion 22 (vision §2.3): ``check --staged`` is blind to a recorded decision.

    ``record_decision`` commits to the graph as it writes the buffer, so the entry is
    never pending and ``--staged`` answers its free one-line clear without building a
    judge or a substrate. That is why the commit gate asks about coverage, not about
    pending entries; whoever revisits ``--staged`` inverts this row.
    """
    config = MitosConfig(str(tmp_path))
    cmd_init(config)
    manager = MitosSyncManager(config)
    manager.record_decision_entry(
        "Gate rows are evaluated in a fixed order.", "Evaluate them in any order.",
        [], slug="gate-order")
    capsys.readouterr()

    def _never(config: MitosConfig) -> Any:
        raise AssertionError("--staged built a check seam with nothing pending")

    monkeypatch.setattr(cli, "_build_check_judge", _never)
    monkeypatch.setattr(cli, "_build_check_substrate", _never)

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=False)

    assert code == 0
    assert capsys.readouterr().out == "Gate clear — no pending decisions to check.\n"
