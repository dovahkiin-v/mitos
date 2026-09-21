"""A failed judgment batch says WHY, not just that it failed.

``mitos check`` reported coverage honestly — *"Judged 66 of 67 judgment batches
(1 failed)"* — and then stopped. The reason existed the whole time and travelled the
whole way out: ``check.py``'s ``record_batch_failure`` appends every ``Unavailable``
to ``judgment_failures``, and ``CheckRunResult.judgment_failures`` is documented as
*"EVERY judgment-stage degradation this run recorded"*. Its only consumer collapsed it
to a bool — ``"judgment": bool(result.judgment_failures)`` in the degradation-token map
— and nothing outside ``check.py`` read it at all. So two consecutive corpus audits
each reported one failed batch and neither could say whether it was a judge timeout, an
unparseable response, or a truncation.

The fix renders the typed reasons, grouped and counted, under the coverage line.

The constraint this module exists to hold: **``Unavailable.detail`` is never rendered.**
``conflict.py`` documents it as *"for logging/telemetry ONLY — never rendered to a user
(the surface owns UX wording)"*. Pasting the exception text into the report is the
obvious implementation and it breaks a stated contract, so
``test_detail_never_reaches_the_report`` pins the boundary rather than trusting it.

The second half (Phase 2d, B7) pins what a failed batch leaves behind: its pairs,
pin and billed usage on ``CheckRunResult.failed_batches`` and in telemetry's
``failed_judgment_batches``, with NULL usage where no response arrived, and never the
detail. The table's own rows live in ``test_failed_batches.py``.

Run under ``./venv/bin/python -m pytest``.
"""

import itertools
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pytest

from mitos import __version__, check, cli
from mitos.check import CheckRunResult, StaleProbe
from mitos.conflict import (
    CONFLICT_PROMPT_VERSION,
    ConflictUnavailableReason,
    JudgmentExecution,
    Unavailable,
)
from mitos.conflict_judgment import BilledUnavailable
from mitos.errors import DatabaseError
from mitos.models import get_model_id
from mitos.telemetry import TelemetryStore

from _conflict_helpers import _drain_outbox, _execution
from test_check_cli import (  # noqa: F401  (offline is autouse; workspace a fixture)
    PRODUCTION_ALIAS,
    _FailingWriteTelemetry,
    _read_check_runs,
    _wire_judge,
    _wire_substrate,
    offline,
    workspace,
)
from test_check_engine import (  # noqa: F401  (fixtures)
    _disjoint_pairs_corpus,
    _plan,
    temp_store,
    temp_telemetry,
)
from test_check_probe import _canned_judge


def _failure(reason: ConflictUnavailableReason, detail: str = "raw sdk text") -> Unavailable:
    """One judgment-stage degradation, carrying a detail that must never be printed."""
    return Unavailable(reason=reason, detail=detail)


def _result(
    failures: Tuple[Unavailable, ...],
    *,
    planned: int = 67,
    executed: int = 67,
    failed: int = 1,
) -> Any:
    """A degraded run whose coverage line fires (``batches_judged < batches_planned``)."""
    probe = StaleProbe(transient=(), excluded=())
    return CheckRunResult(
        run_id="diagnosis-run",
        started_at="2026-09-18T11:17:04+00:00",
        ended_at="2026-09-18T11:28:02+00:00",
        nodes_total=444,
        nodes_swept=444,
        swept_node_ids=(),
        sweep_degraded=None,
        findings=(),
        pairs_judged_fresh=151,
        pairs_reused=1218,
        batches_planned=planned,
        batches_executed=executed,
        batches_failed=failed,
        batches_skipped=0,
        judgment_failures=failures,
        judgment_abort=None,
        reuse_unavailable=None,
        telemetry_write_failures=(),
        start_probe=probe,
        end_probe=probe,
    )


def _render(result: Any, capsys: pytest.CaptureFixture) -> str:
    """Renders the human report and returns stdout."""
    cli._print_check_report(
        result,
        exclusions=[],
        denominator=444,
        scope=None,
        row_written=True,
        transient_count=0,
    )
    return capsys.readouterr().out


# --------------------------------------------------------------------------- #
# The silence this closes
# --------------------------------------------------------------------------- #


def test_a_failed_batch_names_its_reason(capsys: pytest.CaptureFixture) -> None:
    """The 2026-09-18 shape: one failed batch of 67, and now it says why."""
    out = _render(_result((_failure(ConflictUnavailableReason.JUDGMENT_TIMEOUT),)), capsys)

    assert "Judged 66 of 67 judgment batches (1 failed)." in out
    assert "Why: 1 × the judge timed out or returned an error." in out


def test_reasons_are_grouped_and_counted_not_listed(
    capsys: pytest.CaptureFixture,
) -> None:
    """400 timeouts are one fact, not 400 lines."""
    out = _render(
        _result(
            tuple(
                _failure(ConflictUnavailableReason.JUDGMENT_TIMEOUT) for _ in range(3)
            )
            + (_failure(ConflictUnavailableReason.JUDGMENT),),
            planned=67,
            executed=67,
            failed=4,
        ),
        capsys,
    )

    assert "3 × the judge timed out or returned an error" in out
    assert "1 × the judge's response could not be parsed" in out
    assert out.count("Why:") == 1, "one breakdown line, however many failures"


def test_breakdown_order_is_the_vocabulary_not_occurrence(
    capsys: pytest.CaptureFixture,
) -> None:
    """The sentence reads the same way every run, whatever order the failures arrived."""
    out = _render(
        _result(
            (
                _failure(ConflictUnavailableReason.JUDGMENT_TRUNCATED),
                _failure(ConflictUnavailableReason.JUDGMENT_TIMEOUT),
            ),
            failed=2,
        ),
        capsys,
    )

    timeout = out.index("the judge timed out")
    truncated = out.index("hit max_tokens")
    assert timeout < truncated, "printed order must follow the declared vocabulary"


def test_healthy_run_prints_no_breakdown(capsys: pytest.CaptureFixture) -> None:
    """No failures, no furniture — the line appears only when something failed."""
    out = _render(_result((), planned=67, executed=67, failed=0), capsys)

    assert "Why:" not in out


# --------------------------------------------------------------------------- #
# The contract the obvious implementation breaks
# --------------------------------------------------------------------------- #


def test_detail_never_reaches_the_report(capsys: pytest.CaptureFixture) -> None:
    """``Unavailable.detail`` is logging/telemetry only — the surface owns the wording.

    The tempting fix for "the report can't say why" is to print the exception text.
    ``conflict.py`` forbids exactly that, so the breakdown is built from the typed
    reason and this row fails the moment someone reaches for ``detail``.
    """
    secret = "anthropic.APIStatusError: request-id req_0123456789 overloaded_error"
    out = _render(
        _result((_failure(ConflictUnavailableReason.JUDGMENT_TIMEOUT, secret),)), capsys
    )

    assert secret not in out
    assert "req_0123456789" not in out
    assert "the judge timed out or returned an error" in out, (
        "the typed reason still reaches the reader — detail is withheld, not the fact"
    )


def test_every_unavailable_reason_has_wording() -> None:
    """The map is total, so a reason added later cannot KeyError the renderer mid-report.

    ``ConflictUnavailableReason`` is documented as additive — members were added by
    three separate phases. A partial map would render the whole report and then die on
    the one run that hit the new member, which is the worst possible moment.
    """
    missing = [
        reason.value
        for reason in ConflictUnavailableReason
        if reason.value not in cli._JUDGMENT_FAILURE_WORDS
    ]

    assert not missing, (
        f"these Unavailable reasons have no rendered wording: {missing} — add them to "
        f"cli._JUDGMENT_FAILURE_WORDS"
    )


def test_json_carries_reasons_and_not_details() -> None:
    """The machine surface gets the same split: typed reasons, no detail text.

    Built through the real ``_check_json_object`` rather than re-deriving the list
    here — a test that recomputes the value it is checking proves only that the test
    can do arithmetic.
    """
    secret = "anthropic.APIStatusError: request-id req_0123456789 overloaded_error"
    result = _result((_failure(ConflictUnavailableReason.JUDGMENT_TIMEOUT, secret),))
    row = check.check_run_row_from_result(result, mode="corpus", exit_code=2)

    obj = cli._check_json_object(
        result,
        row,
        exclusions=[],
        exit_code=2,
        row_written=True,
        scope=None,
        fresh=False,
        transient_count=0,
    )

    assert obj["judgment_failure_reasons"] == ["judgment_timeout"]
    assert "judgment" in obj["degradations"], "the token stays — this ADDS the reason"
    assert secret not in json.dumps(obj), "detail must not reach the machine surface"


# =========================================================================== #
# B7 (Phase 2d) — a failed batch carries its pairs and its spend, and leaves a
# ``failed_judgment_batches`` row. Driven through ``cmd_check`` (both seams wired,
# a distinct batch prefix per run) and, for the result carry, ``execute_corpus_check``.
# =========================================================================== #

_B7_PREFIXES = (f"b7-batch-{i}" for i in itertools.count())
_B7_BILLED_IDS = (f"b7-billed-{i}" for i in itertools.count())


def _billed(*, reason: ConflictUnavailableReason = ConflictUnavailableReason.JUDGMENT_TRUNCATED,
            usage: Tuple[int, int, int, int] = (900, 2000, 3, 4),
            stop_reason: str = "max_tokens",
            detail: str = "judgment truncated") -> BilledUnavailable:
    """The executor's billed-failure shape, with a batch id no canned judge mints (G4)."""
    return BilledUnavailable(
        reason=reason, detail=detail,
        billed=JudgmentExecution(
            raw_text="", batch_id=next(_B7_BILLED_IDS), model_alias=PRODUCTION_ALIAS,
            token_input=usage[0], token_output=usage[1], token_cache_read=usage[2],
            token_cache_creation=usage[3], elapsed_ms=7, stop_reason=stop_reason,
        ),
    )


def _plain(reason: ConflictUnavailableReason, detail: str = "raw sdk text") -> Unavailable:
    return Unavailable(reason=reason, detail=detail)


def _wire_corpus(workspace, monkeypatch, n: int,
                 overrides: Optional[Dict[int, Any]] = None) -> Any:
    """Commits ``n`` one-pair groups, wires both seams; returns the plan ``cmd_check`` builds."""
    config, store, tel = workspace
    _keys, nbhds = _disjoint_pairs_corpus(store, n)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    plan = check.plan_corpus_check(
        store=store, embed_provider=embed, vector_store=vector, telemetry=tel,
        model_alias=PRODUCTION_ALIAS,
    )
    assert len(plan.fresh_groups) == n
    _wire_judge(monkeypatch, _canned_judge(
        plan, tenable=True, batch_prefix=next(_B7_PREFIXES), overrides=overrides))
    return plan


def _cmd_check(config: Any, capsys: pytest.CaptureFixture) -> Tuple[int, Dict[str, Any]]:
    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=True)
    return code, json.loads(capsys.readouterr().out)


def _failed_rows(path: str) -> List[Dict[str, Any]]:
    """Every ``failed_judgment_batches`` row, in write order."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM failed_judgment_batches ORDER BY rowid")]
    finally:
        conn.close()


def _checks_for(path: str, batch_id: str) -> int:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT COUNT(*) FROM conflict_checks WHERE batch_id = ?",
                            (batch_id,)).fetchone()[0]
    finally:
        conn.close()


_TOKENS = ("token_input", "token_output", "token_cache_read", "token_cache_creation")


def test_each_failed_batch_leaves_a_row_with_its_pairs_pin_and_usage(
    workspace, monkeypatch, capsys,
) -> None:
    """C1: billed usage where a response arrived, NULLs where none did, pairs from the plan."""
    config, _store, _tel = workspace
    truncated = _billed()
    plan = _wire_corpus(workspace, monkeypatch, 4, overrides={
        0: truncated,
        2: _plain(ConflictUnavailableReason.JUDGMENT_REJECTED),
        3: _plain(ConflictUnavailableReason.JUDGMENT_TIMEOUT),
    })

    code, payload = _cmd_check(config, capsys)

    assert code == 2
    assert (payload["batches_failed"], payload["batches_skipped"]) == (3, 0)
    rows = _failed_rows(config.telemetry_path)
    assert [r["reason"] for r in rows] == [
        "judgment_truncated", "judgment_rejected", "judgment_timeout"]
    trunc, rejected, timeout = rows
    assert trunc["batch_id"] == truncated.billed.batch_id
    assert [trunc[c] for c in _TOKENS] == [900, 2000, 3, 4]
    assert trunc["stop_reason"] == "max_tokens"
    for silent in (rejected, timeout):
        assert [silent[c] for c in _TOKENS] == [None] * 4
        assert silent["stop_reason"] is None
        assert len(silent["batch_id"]) == 32
    for row, group in zip(rows, (plan.fresh_groups[i] for i in (0, 2, 3))):
        assert row["proposal_id"] == group.proposal_hash
        assert json.loads(row["partner_ids"]) == [p.partner_hash for p in group.pairs]
    (run_row,) = _read_check_runs(config)
    for row in rows:
        assert row["run_id"] == run_row["run_id"] == payload["run_id"]
        assert row["model_alias"] == PRODUCTION_ALIAS
        assert row["prompt_version"] == CONFLICT_PROMPT_VERSION
        assert row["model_id"] == get_model_id(PRODUCTION_ALIAS)
        assert row["mitos_version"] == __version__
        stamp = datetime.fromisoformat(row["created_at"])
        assert stamp.utcoffset() == timedelta(0)
        assert row["created_at"].endswith("+00:00")


def test_a_parse_failure_row_carries_that_executions_usage(
    workspace, monkeypatch, capsys,
) -> None:
    """C2: an empty verdict array fails the parse; the row is the execution that arrived."""
    config, _store, _tel = workspace
    _wire_corpus(workspace, monkeypatch, 2,
                 overrides={1: _execution([], batch_id="b7-parse-fail")})

    code, payload = _cmd_check(config, capsys)

    assert code == 2
    assert payload["judgment_failure_reasons"] == ["judgment_unavailable"]
    (row,) = _failed_rows(config.telemetry_path)
    assert row["batch_id"] == "b7-parse-fail"
    assert [row[c] for c in _TOKENS] == [100, 40, 0, 0]
    assert row["stop_reason"] is None


def test_the_result_carries_the_failed_rows_in_occurrence_order(
    temp_store, temp_telemetry,
) -> None:
    """C3: ``len(failed_batches) == batches_failed``, occurrence order, equal to telemetry."""
    _keys, nbhds = _disjoint_pairs_corpus(temp_store, 4)
    _drain_outbox(temp_store)
    plan = _plan(temp_store, nbhds, temp_telemetry)
    judge = _canned_judge(plan, batch_prefix=next(_B7_PREFIXES), overrides={
        1: _plain(ConflictUnavailableReason.JUDGMENT),
        3: _billed(),
    })

    result = check.execute_corpus_check(
        plan, judge=judge, telemetry=temp_telemetry, store=temp_store)

    assert result.batches_failed == len(result.failed_batches) == 2
    assert [r.proposal_id for r in result.failed_batches] == [
        plan.fresh_groups[1].proposal_hash, plan.fresh_groups[3].proposal_hash]
    stored = _failed_rows(temp_telemetry.telemetry_path)
    for row in stored:
        row["partner_ids"] = json.loads(row["partner_ids"])
    assert [check.FailedJudgmentBatch(**r) for r in stored] == list(result.failed_batches)


def test_the_row_needs_no_other_row_and_survives_a_failed_run_end_seam(
    workspace, monkeypatch, capsys,
) -> None:
    """C4: zero ``conflict_checks`` rows for it, and no ``check_runs`` row at all (D2)."""
    config, _store, _tel = workspace
    _wire_corpus(workspace, monkeypatch, 2, overrides={0: _billed()})
    monkeypatch.setattr(cli, "_build_check_telemetry", lambda config: (
        _FailingWriteTelemetry(TelemetryStore(config.telemetry_path))))

    code, payload = _cmd_check(config, capsys)

    assert code == 2
    assert payload["summary_row_written"] is False
    assert _read_check_runs(config) == []
    (row,) = _failed_rows(config.telemetry_path)
    assert row["run_id"] == payload["run_id"]
    assert _checks_for(config.telemetry_path, row["batch_id"]) == 0


def test_a_failed_batch_moves_no_reuse_answer(workspace, monkeypatch, capsys) -> None:
    """C5: the reuse index is unchanged, and the next run judges those pairs fresh again."""
    config, store, tel = workspace

    def index() -> Dict[Any, Any]:
        loaded = tel.load_reuse_index(prompt_version=CONFLICT_PROMPT_VERSION,
                                      model_alias=PRODUCTION_ALIAS)
        return dict(loaded.iter_pairs())

    before = index()
    _wire_corpus(workspace, monkeypatch, 1, overrides={0: _billed()})
    code, _payload = _cmd_check(config, capsys)
    assert code == 2
    assert len(_failed_rows(config.telemetry_path)) == 1
    assert index() == before

    embed, vector = cli._build_check_substrate(config)[:2]
    replan = check.plan_corpus_check(
        store=store, embed_provider=embed, vector_store=vector, telemetry=tel,
        model_alias=PRODUCTION_ALIAS,
    )
    assert len(replan.fresh_groups) == 1 and replan.reused == ()
    _wire_judge(monkeypatch, _canned_judge(replan, tenable=True,
                                           batch_prefix=next(_B7_PREFIXES)))
    code, payload = _cmd_check(config, capsys)
    assert code == 0
    assert (payload["pairs_judged_fresh"], payload["pairs_reused"]) == (1, 0)


def test_detail_never_reaches_a_failed_batch_row(workspace, monkeypatch, capsys) -> None:
    """C6: a planted request id in ``detail`` lands in no column of any row."""
    config, _store, _tel = workspace
    _wire_corpus(workspace, monkeypatch, 2, overrides={
        0: _billed(detail="truncated req_PLANTED"),
        1: _plain(ConflictUnavailableReason.JUDGMENT_REJECTED, "HTTP 400 req_PLANTED"),
    })

    _cmd_check(config, capsys)

    rows = _failed_rows(config.telemetry_path)
    assert len(rows) == 2
    assert not any("req_PLANTED" in str(value) for row in rows for value in row.values())


def test_no_judge_writes_no_failed_row(temp_store, temp_telemetry) -> None:
    """C7: the keyless abort attempted no batch, so it carries and writes nothing."""
    _keys, nbhds = _disjoint_pairs_corpus(temp_store, 2)
    _drain_outbox(temp_store)
    plan = _plan(temp_store, nbhds, temp_telemetry)

    result = check.execute_corpus_check(
        plan, judge=None, telemetry=temp_telemetry, store=temp_store)

    assert result.failed_batches == ()
    assert result.batches_failed == 0
    assert result.judgment_abort is result.judgment_failures[-1]
    assert _failed_rows(temp_telemetry.telemetry_path) == []


class _FailingFailedBatchTelemetry:
    """Wraps a real store; only ``record_failed_batch`` raises."""

    def __init__(self, inner: TelemetryStore) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def record_failed_batch(self, row: Any) -> None:
        raise DatabaseError("provoked failed-batch write fault")


def test_a_failed_row_write_degrades_and_never_aborts(temp_store, temp_telemetry) -> None:
    """C8: the next healthy batch still persists; ``telemetry_write`` joins ``judgment``."""
    _keys, nbhds = _disjoint_pairs_corpus(temp_store, 2)
    _drain_outbox(temp_store)
    plan = _plan(temp_store, nbhds, temp_telemetry)
    prefix = next(_B7_PREFIXES)
    judge = _canned_judge(plan, batch_prefix=prefix, overrides={0: _billed()})

    result = check.execute_corpus_check(
        plan, judge=judge, telemetry=_FailingFailedBatchTelemetry(temp_telemetry),
        store=temp_store)

    assert result.batches_executed == 2 and result.batches_judged == 1
    assert _checks_for(temp_telemetry.telemetry_path, f"{prefix}-1") == 1
    degradations = check.run_degradations(result)
    assert "telemetry_write" in degradations and "judgment" in degradations
    assert len(result.failed_batches) == 1
    assert result.telemetry_write_failures == (
        f"failed batch {result.failed_batches[0].batch_id}: "
        "provoked failed-batch write fault",)
    assert _failed_rows(temp_telemetry.telemetry_path) == []


def test_a_failed_batch_without_telemetry_is_disclosed(temp_store, temp_telemetry) -> None:
    """``telemetry=None``: the failed row is carried, and its unwritten row is a write failure."""
    _keys, nbhds = _disjoint_pairs_corpus(temp_store, 1)
    _drain_outbox(temp_store)
    plan = _plan(temp_store, nbhds, temp_telemetry)
    judge = _canned_judge(plan, batch_prefix=next(_B7_PREFIXES), overrides={0: _billed()})

    result = check.execute_corpus_check(plan, judge=judge, telemetry=None, store=temp_store)

    (row,) = result.failed_batches
    assert result.telemetry_write_failures == (
        f"failed batch {row.batch_id}: telemetry store unavailable (never constructed)",)
    assert "telemetry_write" in check.run_degradations(result)
