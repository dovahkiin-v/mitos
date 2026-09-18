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

Run under ``./venv/bin/python -m pytest``.
"""

from typing import Any, Tuple

import pytest

from mitos import cli
from mitos.check import CheckRunResult, StaleProbe
from mitos.conflict import ConflictUnavailableReason, Unavailable


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
    import json

    from mitos import check

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
