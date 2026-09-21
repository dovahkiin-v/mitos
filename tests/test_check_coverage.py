"""Tests for check coverage — telemetry rung 5, the run-end seam, the read-only reader.

Coverage is the primary record of which decisions an undegraded corpus check has
actually swept: one row per node, ``covered`` or ``excluded``, written only by an
undegraded run and in the same transaction as that run's ``check_runs`` row. The
rows here pin four things:

* which runs cover — an undegraded run covers (exit 0 *and* exit 1), a degraded run
  writes its row and no coverage, a seam fault writes neither;
* what a run covers — only the nodes it swept, so a ``--scope`` run never marks a node
  outside its scope and a decision recorded mid-run stays uncovered;
* the mark rule — the last undegraded run to sweep a node wins, both promotion
  (``excluded → covered``) and the write of an exclusion;
* the reader — ``read_coverage`` returns data, ``TelemetryAbsent`` or
  ``TelemetryUnreadable`` and never creates or migrates the file.

Every ``cmd_check`` row injects both seams (substrate and judge): an offline run with
no embedding provider over a non-empty corpus exits before the seam, and a row that
injected only the judge would pass for that reason. Real SQLite throughout.
"""

import dataclasses
import itertools
import json
import os
import sqlite3
from typing import Any, Dict, Optional, Tuple

import pytest

from mitos import check, cli, telemetry
from mitos.check import BacklogRow, CheckRunResult, ProbeUnavailable, StaleProbe
from mitos.conflict import ConflictUnavailableReason, Unavailable
from mitos.migrations import _pending_head, run_migrations
from mitos.parser import ParsedEntry
from mitos.store import open_connection
from mitos.telemetry import (
    COVERAGE_RUNG,
    TELEMETRY_MIGRATION_STEPS,
    CheckRunRow,
    CoverageMarks,
    CoverageRead,
    ReuseUnavailable,
    TelemetryAbsent,
    TelemetryStore,
    TelemetryUnreadable,
    read_coverage,
)

from _conflict_helpers import _drain_outbox, _match
from test_check_cli import (  # noqa: F401  (offline is autouse; workspace a fixture)
    PRODUCTION_ALIAS,
    _pair,
    _read_check_runs,
    _wire_judge,
    _wire_substrate,
    offline,
    workspace,
)
from test_check_probe import _canned_judge, _commit, _poison


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _coverage(path: str) -> Dict[str, Tuple[str, str, str]]:
    """Reads every coverage row as ``{node_id: (mark, run_id, marked_at)}``."""
    conn = sqlite3.connect(path)
    try:
        return {
            node_id: (mark, run_id, marked_at)
            for node_id, mark, run_id, marked_at in conn.execute(
                "SELECT node_id, mark, run_id, marked_at FROM check_coverage"
            )
        }
    finally:
        conn.close()


_BATCH_PREFIXES = (f"cov-batch-{i}" for i in itertools.count())


def _judge_for(
    store: Any, embed: Any, vector: Any, tel: Any, *,
    scope: Optional[str] = None, fresh: bool = False,
    tenable: bool = True, confidence: float = 0.9,
    overrides: Optional[Dict[int, Any]] = None,
) -> Any:
    """A canned judge matching the plan ``cmd_check`` will build (one execution per group).

    Each judge mints its own batch-id prefix: ``batch_id`` is the
    ``judgment_batches`` PK, so two runs sharing one would degrade the second.
    """
    plan = check.plan_corpus_check(
        store=store, embed_provider=embed, vector_store=vector, telemetry=tel,
        model_alias=PRODUCTION_ALIAS, scope=scope, fresh=fresh,
    )
    return _canned_judge(plan, tenable=tenable, confidence=confidence,
                         batch_prefix=next(_BATCH_PREFIXES), overrides=overrides)


def _run(config: Any, capsys: Any, *, scope: Optional[str] = None,
         fresh: bool = False) -> Tuple[int, Dict[str, Any]]:
    """Runs ``cmd_check`` in corpus mode with ``--json``; returns (exit, payload)."""
    code = cli.cmd_check(config, scope=scope, fresh=fresh, assume_yes=False,
                         as_json=True)
    return code, json.loads(capsys.readouterr().out)


def _run_row(config: Any, run_id: str) -> Dict[str, Any]:
    """The ``check_runs`` row for one run."""
    (row,) = [r for r in _read_check_runs(config) if r["run_id"] == run_id]
    return row


def _check_run_row(run_id: str) -> CheckRunRow:
    """A minimal valid corpus row for driving ``record_run_end`` directly."""
    return CheckRunRow(
        run_id=run_id, mode="corpus",
        started_at="2026-09-21T00:00:00+00:00", ended_at="2026-09-21T00:01:00+00:00",
        exit_code=0, nodes_swept=2, pairs_judged_fresh=0, pairs_reused=0,
        findings_new=0, findings_known=0, coverage_exclusions=0,
        degraded_reason=None, mitos_version="test",
    )


# --------------------------------------------------------------------------- #
# 1 — the rung
# --------------------------------------------------------------------------- #

def _tables(conn: sqlite3.Connection) -> set:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_fresh_store_boots_to_head_with_the_coverage_table(tmp_path) -> None:
    """A fresh ``TelemetryStore`` reaches the ladder head with ``check_coverage`` present."""
    path = str(tmp_path / "telemetry.sqlite")
    TelemetryStore(path)
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == _pending_head(
            TELEMETRY_MIGRATION_STEPS)
        assert "check_coverage" in _tables(conn)
    finally:
        conn.close()
    assert dict(TELEMETRY_MIGRATION_STEPS)[COVERAGE_RUNG] is telemetry._check_coverage_schema


def test_rung_four_file_upgrades_with_rows_intact_and_replay_is_a_no_op(tmp_path) -> None:
    """A rung-4 file upgrades to 5 keeping its rows; the ladder re-run at head changes nothing."""
    path = str(tmp_path / "telemetry.sqlite")
    conn = open_connection(path)
    run_migrations(conn, TELEMETRY_MIGRATION_STEPS[:4])
    conn.close()
    conn = open_connection(path)
    with conn:
        conn.execute(telemetry._INSERT_CHECK_RUN_SQL, _check_run_row("old-run").to_params())
    conn.close()

    store = TelemetryStore(path)
    store.record_run_end(
        _check_run_row("new-run"),
        coverage=CoverageMarks(run_id="new-run", marked_at="t", covered=("n1",),
                               excluded=()),
    )

    def snapshot() -> Tuple[int, list, list, list]:
        c = sqlite3.connect(path)
        try:
            return (
                c.execute("PRAGMA user_version").fetchone()[0],
                c.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall(),
                c.execute("SELECT run_id FROM check_runs ORDER BY run_id").fetchall(),
                c.execute("SELECT * FROM check_coverage").fetchall(),
            )
        finally:
            c.close()

    before = snapshot()
    assert before[0] == COVERAGE_RUNG
    assert before[2] == [("new-run",), ("old-run",)]
    conn = open_connection(path)
    run_migrations(conn, TELEMETRY_MIGRATION_STEPS)
    conn.close()
    TelemetryStore(path)
    assert snapshot() == before


# --------------------------------------------------------------------------- #
# 2–9 — which runs cover, and what
# --------------------------------------------------------------------------- #

def test_undegraded_run_covers_its_sweep(workspace, monkeypatch, capsys) -> None:
    """Criterion 2: a clean run covers both ids; run_id and marked_at join its row."""
    config, store, tel = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel))

    code, obj = _run(config, capsys)

    assert code == 0 and obj["degradations"] == []
    row = _run_row(config, obj["run_id"])
    assert _coverage(config.telemetry_path) == {
        a_id: ("covered", row["run_id"], row["ended_at"]),
        b_id: ("covered", row["run_id"], row["ended_at"]),
    }


def test_exit_one_run_still_covers(workspace, monkeypatch, capsys) -> None:
    """Criterion 3 (D1): finding a new contradiction is not failing to look."""
    config, store, tel = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel, tenable=False))

    code, obj = _run(config, capsys)

    assert code == 1
    cov = _coverage(config.telemetry_path)
    assert {n: m for n, (m, _, _) in cov.items()} == {a_id: "covered", b_id: "covered"}


def test_degraded_run_writes_its_row_and_no_coverage_then_a_clean_rerun_covers(
    workspace, monkeypatch, capsys
) -> None:
    """Criterion 4: one failing batch → exit 2, a degraded row, zero coverage; then covered."""
    config, store, tel = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    failing = Unavailable(reason=ConflictUnavailableReason.JUDGMENT, detail="judge died")
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel, overrides={0: failing}))

    code, obj = _run(config, capsys)

    assert code == 2
    assert "judgment" in _run_row(config, obj["run_id"])["degraded_reason"]
    assert _coverage(config.telemetry_path) == {}

    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel))
    code, obj = _run(config, capsys)

    assert code == 0
    cov = _coverage(config.telemetry_path)
    assert set(cov) == {a_id, b_id}
    assert {run_id for _, run_id, _ in cov.values()} == {obj["run_id"]}


def test_exclusion_then_promotion(workspace, monkeypatch, capsys) -> None:
    """Criterion 5: a poisoned swept node is ``excluded``; drained, a clean re-run covers it."""
    config, store, tel = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    _poison(store, a_id)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel))

    code, first = _run(config, capsys)

    assert code == 0 and first["degradations"] == []
    cov = _coverage(config.telemetry_path)
    assert cov[a_id][0] == "excluded" and cov[b_id][0] == "covered"

    store.remove_pending_embedding(a_id)
    _wire_judge(monkeypatch, None)  # the pair is reused now; no fresh groups
    code, second = _run(config, capsys)

    assert code == 0
    cov = _coverage(config.telemetry_path)
    assert cov[a_id][:2] == ("covered", second["run_id"])
    assert second["run_id"] != first["run_id"]


def _scoped_corpus(store: Any) -> Tuple[str, str, str, Dict[str, Any]]:
    """Two scope-x decisions (one discovers the other) and one scope-y decision."""
    x1_axiom, x2_axiom, y1_axiom = (
        "Scoped axiom x-one.", "Scoped axiom x-two.", "Scoped axiom y-one.")
    x1 = _commit(store, "cov-x1", x1_axiom, scope=["x"])
    x2 = _commit(store, "cov-x2", x2_axiom, scope=["x"])
    y1 = _commit(store, "cov-y1", y1_axiom, scope=["y"])
    nbhds = {x1_axiom: [_match("cov-x2", 0.9)], x2_axiom: [], y1_axiom: []}
    return x1, x2, y1, nbhds


def test_scoped_run_covers_only_what_it_swept(workspace, monkeypatch, capsys) -> None:
    """Criterion 6: ``--scope x`` → rows for x's decisions only."""
    config, store, tel = workspace
    x1, x2, y1, nbhds = _scoped_corpus(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel, scope="x"))

    code, _obj = _run(config, capsys, scope="x")

    assert code == 0
    assert set(_coverage(config.telemetry_path)) == {x1, x2}


def test_scoped_run_never_marks_outside_its_scope(workspace, monkeypatch, capsys) -> None:
    """Criterion 6b (D3's bound): an exclusion the run did not sweep gets no row, ever."""
    config, store, tel = workspace
    x1, x2, y1, nbhds = _scoped_corpus(store)
    oq = ParsedEntry("open_question", "cov-oq", 1, 5)
    oq.topic = "a parked coverage topic"
    oq.questions_raised = ["is this ever swept?"]
    oq_id = store.commit_parsed_entry(oq).node_id
    _drain_outbox(store)
    _poison(store, y1)
    _poison(store, oq_id)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel, scope="x"))

    code, obj = _run(config, capsys, scope="x")

    assert code == 0
    # Non-vacuous: the run named both exclusions and still marked neither.
    assert _run_row(config, obj["run_id"])["coverage_exclusions"] == 2
    assert set(_coverage(config.telemetry_path)) == {x1, x2}

    # The transition: an unscoped run marks y1 excluded (never the open question) ...
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel))
    code, _obj = _run(config, capsys)
    assert code == 0
    before = _coverage(config.telemetry_path)
    assert before[y1][0] == "excluded"
    assert oq_id not in before

    # ... and once y1 drains, scope-x runs leave its row byte-identical.
    store.remove_pending_embedding(y1)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel, scope="x"))
    code, third = _run(config, capsys, scope="x")
    assert code == 0
    after = _coverage(config.telemetry_path)
    assert after[y1] == before[y1]
    assert after[x1][1] == third["run_id"]


def test_fresh_run_covers(workspace, monkeypatch, capsys) -> None:
    """Criterion 7: ``--fresh`` covers like an unscoped clean run, moving the run_id."""
    config, store, tel = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel))
    _run(config, capsys)

    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel, fresh=True))
    code, obj = _run(config, capsys, fresh=True)

    assert code == 0 and obj["fresh"] is True
    cov = _coverage(config.telemetry_path)
    assert set(cov) == {a_id, b_id}
    assert {run_id for _, run_id, _ in cov.values()} == {obj["run_id"]}


def test_seam_fault_inside_the_transaction_lands_nothing(
    workspace, monkeypatch, capsys
) -> None:
    """Criterion 8: the coverage upsert fails after the row INSERT ran → nothing lands."""
    config, store, tel = workspace
    _a, _b, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel))
    monkeypatch.setattr(telemetry, "_UPSERT_COVERAGE_SQL", "NOT SQL")

    code, obj = _run(config, capsys)

    assert code == 2
    assert obj["summary_row_written"] is False
    assert _read_check_runs(config) == []
    assert _coverage(config.telemetry_path) == {}


class _CommittingJudge:
    """Delegates to a canned judge, committing a new decision on its first call.

    The commit's outbox row is removed at once, so the end probe sees no transient
    backlog and the run stays undegraded; the only thing that differs from a clean
    run is a decision the sweep never saw.
    """

    def __init__(self, inner: Any, store: Any) -> None:
        self._inner = inner
        self._store = store
        self.committed: Optional[str] = None

    def __call__(self, prompt: Any) -> Any:
        if self.committed is None:
            self.committed = _commit(self._store, "cov-mid-run",
                                     "A decision recorded while the check ran.")
            self._store.remove_pending_embedding(self.committed)
        return self._inner(prompt)


def test_decision_recorded_mid_run_stays_uncovered(workspace, monkeypatch, capsys) -> None:
    """D2's guard: coverage comes from what the sweep consumed, not a store re-read."""
    config, store, tel = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    judge = _CommittingJudge(_judge_for(store, embed, vector, tel), store)
    _wire_judge(monkeypatch, judge)

    code, obj = _run(config, capsys)

    assert judge.committed is not None
    assert code in (0, 1) and obj["degradations"] == []
    cov = _coverage(config.telemetry_path)
    assert set(cov) == {a_id, b_id}
    assert judge.committed not in cov


# --------------------------------------------------------------------------- #
# The seam's own contract (direct)
# --------------------------------------------------------------------------- #

def test_coverage_marks_reject_an_id_in_both_tuples() -> None:
    """A node carries one mark; the boundary shape refuses both, before any connection."""
    with pytest.raises(ValueError, match="one coverage mark"):
        CoverageMarks(run_id="r", marked_at="t", covered=("n1", "n2"), excluded=("n2",))


def test_duplicate_run_id_rolls_its_coverage_back(tmp_path) -> None:
    """A failed row INSERT takes the same transaction's coverage with it."""
    path = str(tmp_path / "telemetry.sqlite")
    store = TelemetryStore(path)
    store.record_run_end(_check_run_row("r1"), coverage=None)

    with pytest.raises(telemetry.DatabaseError):
        store.record_run_end(
            _check_run_row("r1"),
            coverage=CoverageMarks(run_id="r1", marked_at="t", covered=("n1",),
                                   excluded=()),
        )
    assert _coverage(path) == {}


def test_record_check_run_writes_the_row_alone(tmp_path) -> None:
    """The staged writer delegates to the seam with no coverage."""
    path = str(tmp_path / "telemetry.sqlite")
    TelemetryStore(path).record_check_run(_check_run_row("staged-run"))
    assert _coverage(path) == {}
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT run_id FROM check_runs").fetchall() == [("staged-run",)]
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# 10–11 — the read-only reader
# --------------------------------------------------------------------------- #

def test_reader_on_an_absent_path_creates_nothing(tmp_path) -> None:
    """No file (or no parent dir) → ``TelemetryAbsent('no_file')``, and still no file."""
    path = tmp_path / "telemetry.sqlite"
    assert read_coverage(str(path)) == TelemetryAbsent("no_file")
    assert not os.path.exists(path)

    nested = tmp_path / "missing-dir" / "telemetry.sqlite"
    assert read_coverage(str(nested)) == TelemetryAbsent("no_file")
    assert not os.path.exists(nested.parent)


def test_reader_on_a_below_rung_file_leaves_it_untouched(tmp_path) -> None:
    """A rung-4 file → ``below_rung``; its user_version and tables are unchanged."""
    path = str(tmp_path / "telemetry.sqlite")
    conn = open_connection(path)
    run_migrations(conn, TELEMETRY_MIGRATION_STEPS[:4])
    conn.close()

    assert read_coverage(path) == TelemetryAbsent("below_rung")
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
        assert "check_coverage" not in _tables(conn)
    finally:
        conn.close()


def test_reader_on_an_empty_file_reads_below_rung(tmp_path) -> None:
    """A 0-byte file is an empty database at user_version 0: it holds nothing."""
    path = tmp_path / "telemetry.sqlite"
    path.write_bytes(b"")
    assert read_coverage(str(path)) == TelemetryAbsent("below_rung")


def test_reader_on_a_corrupt_file_or_a_directory_is_unreadable(tmp_path) -> None:
    """A file that exists and cannot be read is a fault, never an empty set."""
    corrupt = tmp_path / "telemetry.sqlite"
    corrupt.write_bytes(b"this is not a sqlite database, not even close" * 20)
    assert isinstance(read_coverage(str(corrupt)), TelemetryUnreadable)

    directory = tmp_path / "a-directory"
    directory.mkdir()
    assert isinstance(read_coverage(str(directory)), TelemetryUnreadable)


def test_reader_splits_marks_and_empty_is_a_different_type(tmp_path) -> None:
    """A healthy file → ``CoverageRead`` by mark; a healthy empty one is not ``Absent``."""
    path = str(tmp_path / "telemetry.sqlite")
    store = TelemetryStore(path)

    empty = read_coverage(path)
    assert empty == CoverageRead(covered=frozenset(), excluded=frozenset())
    assert not isinstance(empty, TelemetryAbsent)

    store.record_run_end(
        _check_run_row("r1"),
        coverage=CoverageMarks(run_id="r1", marked_at="t", covered=("n1", "n2"),
                               excluded=("n3",)),
    )
    assert read_coverage(path) == CoverageRead(
        covered=frozenset({"n1", "n2"}), excluded=frozenset({"n3"}))


def test_reader_ignores_an_unknown_mark(tmp_path) -> None:
    """Criterion 11: a newer build's mark is in neither set and does not raise."""
    path = str(tmp_path / "telemetry.sqlite")
    TelemetryStore(path)
    conn = sqlite3.connect(path)
    with conn:
        conn.execute("INSERT INTO check_coverage VALUES ('n9', 'stale', 'r', 't')")
    conn.close()

    assert read_coverage(path) == CoverageRead(covered=frozenset(), excluded=frozenset())


# --------------------------------------------------------------------------- #
# 12 — coverage_marks_from_result
# --------------------------------------------------------------------------- #

def _healthy_result(**overrides: Any) -> CheckRunResult:
    """A minimal undegraded result sweeping ``a`` and ``b``."""
    base = CheckRunResult(
        run_id="unit-run", started_at="2026-09-21T00:00:00+00:00",
        ended_at="2026-09-21T00:01:00+00:00",
        nodes_total=2, nodes_swept=2, swept_node_ids=("b", "a"),
        sweep_degraded=None, findings=(), pairs_judged_fresh=0, pairs_reused=0,
        batches_planned=0, batches_executed=0, batches_failed=0, batches_skipped=0,
        judgment_failures=(), judgment_abort=None, reuse_unavailable=None,
        telemetry_write_failures=(),
        start_probe=StaleProbe(transient=(), excluded=()),
        end_probe=StaleProbe(transient=(), excluded=()),
    )
    return dataclasses.replace(base, **overrides)


def _poison_row(node_id: str) -> BacklogRow:
    return BacklogRow(node_id=node_id, queued_at="2026-09-21T00:00:00+00:00",
                      retry_count=check.CHECK_STALE_RETRY_TOLERANCE)


def test_marks_cover_the_sweep_sorted_and_stamped() -> None:
    marks = check.coverage_marks_from_result(_healthy_result())
    assert marks == CoverageMarks(run_id="unit-run",
                                  marked_at="2026-09-21T00:01:00+00:00",
                                  covered=("a", "b"), excluded=())


@pytest.mark.parametrize("overrides", [
    {"judgment_failures": (Unavailable(reason=ConflictUnavailableReason.JUDGMENT,
                                       detail="x"),)},
    {"reuse_unavailable": ReuseUnavailable("broken read")},
    {"telemetry_write_failures": ("batch b0: disk full",)},
    {"start_probe": StaleProbe(transient=(_poison_row("a"),), excluded=())},
    # The exclusion set is only half known here; marks would cover from a guess.
    {"end_probe": ProbeUnavailable("backlog read failed")},
], ids=["judgment", "reuse_read", "telemetry_write", "stale_index", "probe_read"])
def test_any_degradation_token_means_no_marks(overrides: Dict[str, Any]) -> None:
    result = _healthy_result(**overrides)
    assert check.run_degradations(result)
    assert check.coverage_marks_from_result(result) is None


def test_swept_exclusion_is_split_out_and_unswept_exclusion_is_dropped() -> None:
    """An excluded id the run swept is ``excluded``; one it did not sweep gets nothing."""
    result = _healthy_result(
        start_probe=StaleProbe(transient=(), excluded=(_poison_row("a"),)),
        end_probe=StaleProbe(transient=(), excluded=(_poison_row("z"),)),
    )
    assert check.run_degradations(result) == ()
    marks = check.coverage_marks_from_result(result)
    assert marks.covered == ("b",)
    assert marks.excluded == ("a",)
