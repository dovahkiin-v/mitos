"""Machinery for the Layer-B conflict eval — live SONNET judge, banded (Conflict §6.3).

The conflict twin of `_semantic_harness.py`. Where the retrieval harness scores the
shipped embedding + Qdrant path, this one drives the shipped **conflict facade**
(`mitos.conflict.run_conflict_check`) with the **real SONNET judge** over the frozen
Harbor corpus, and turns the facade's own return shape (`judged_pairs` + `surfaced`
flags) into the conflict metrics report + a soft baseline diff. It is a *measurement*
layer: it reads the gate off the facade output, it never recomputes it (D2).

**This module is the anthropic quarantine boundary (plan D5).** It is the sole golden
harness file that pulls `anthropic` into `sys.modules` — via `mitos.conflict_judgment`,
the one module that imports the SDK. It is imported ONLY by `test_conflict_eval_live.py`;
`_semantic_harness.py` and `test_retrieval_live.py` stay anthropic-free by design, so the
retrieval eval never drags the judgment SDK in. It *reuses* the anthropic-free shared IO
(`provenance`, `write_report`, `populate_index`, `qdrant_reachable`, `GOLDEN_DIR`) by
importing `_semantic_harness`.

The judge is stochastic, so the discipline is Layer B's: the live test hard-asserts only
the DETERMINISTIC screening (the declared-target drop is pure graph logic) and a single
lenient smoke floor; the tenable/confidence judgment quality goes into the report + a soft
`conflict_baseline_diff` that WARNS for human review, never reds CI, never auto-seeds the
baseline (`MITOS_UPDATE_BASELINE=1` only).
"""

import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

import anthropic

from mitos.conflict import (
    CONFLICT_PROMPT_VERSION,
    CONFLICT_SIMILARITY_FLOOR,
    CONFLICT_SURFACE_THRESHOLD,
    CONFLICT_TOP_K,
    Unavailable,
    run_conflict_check,
)
from mitos.conflict_judgment import make_judgment_executor

sys.path.insert(0, os.path.dirname(__file__))
import _semantic_harness as H  # noqa: E402
from metrics import (  # noqa: E402
    confidence_calibration_curve,
    not_tenable_precision,
    not_tenable_recall,
    recommend_floor,
    same_polarity_fp_rate,
)

CONFLICT_BASELINE_PATH = os.path.join(H.GOLDEN_DIR, "conflict.baseline.metrics.json")

# Conflict metrics where a HIGHER value is better (a drop past the band is a regression).
CONFLICT_HIGHER_IS_BETTER = ("not_tenable_recall", "not_tenable_precision")
# Conflict metrics where a LOWER value is better (a rise past the band is a regression).
CONFLICT_LOWER_IS_BETTER = ("same_polarity_fp_rate",)

# Distinct `reason` values on a per-fixture outcome, telling the two judged=false
# meanings apart (§7): the EXPECTED declared-target drop vs the UNEXPECTED retrieval
# miss at the active floor (the loud-but-soft 4b signal).
REASON_JUDGED = "judged"
REASON_DECLARED_DROP = "declared_drop"
REASON_RETRIEVAL_MISS = "unexpected_retrieval_miss"


# ---------------------------------------------------------------------------
# Live judge construction
# ---------------------------------------------------------------------------

def make_live_judge() -> Callable:
    """Constructs the real SONNET judge from ``ANTHROPIC_API_KEY`` in the environment.

    Homes the ``import anthropic`` + client construction inside this quarantine module
    so the live test never touches the SDK directly. Client construction issues no API
    call (it cannot 429 — that is why the anthropic degradation is inspected off the
    returned :class:`Unavailable`, not caught here; Warning E), so this raises only on a
    genuinely absent key, which the caller gates on before calling.

    Returns:
        The one-arg ``judge`` callable the facade expects (``RenderedPrompt`` →
        ``JudgmentExecution | Unavailable``).
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return make_judgment_executor(client)


# ---------------------------------------------------------------------------
# Conflict eval
# ---------------------------------------------------------------------------

def _outcome_and_fixture(
    fx: Dict[str, Any], result: Any
) -> Dict[str, Any]:
    """Derives one fixture's outcome + report record from a facade result.

    Reads the NAMED candidate off the facade's ``judged_pairs`` (other judged pairs
    from the over-fetch are noise — not oracle-scored). The named candidate being
    absent means ``judged=false``, whose *meaning* splits on the oracle: an expected
    declared-drop (fixture 4) vs an unexpected retrieval miss at the active floor.
    The gate (``pair.surfaced``) and verdict (``pair.judgment.*``) are read off the
    result, NEVER recomputed (D2).

    Args:
        fx: The oracle conflict fixture.
        result: A non-degraded :class:`~mitos.conflict.ConflictCheckResult`.

    Returns:
        A dict ``{"outcome": <metrics record>, "fixture": <report record>}``.
    """
    candidate_slug = fx["candidate"]
    pair = next(
        (p for p in result.judged_pairs if p.candidate.slug == candidate_slug), None
    )
    if pair is not None:
        judged = True
        actual_tenable: Optional[bool] = pair.judgment.tenable_together
        actual_surfaced = pair.surfaced
        confidence: Optional[float] = pair.judgment.confidence
        reason = REASON_JUDGED
        # The Qdrant similarity S5 gates on (`candidate.score >= floor`) — the SOLE
        # input to 4b's floor calibration. `null` when the named candidate never
        # retrieved/reached the judge (below).
        similarity: Optional[float] = pair.candidate.score
    else:
        judged = False
        actual_tenable = None
        actual_surfaced = False
        confidence = None
        similarity = None
        # judged=false is EXPECTED for the declared-drop fixture, an unexpected soft
        # signal otherwise (a below-floor retrieval miss the eval records but never reds).
        reason = (
            REASON_DECLARED_DROP
            if not fx["expected_candidate_judged"]
            else REASON_RETRIEVAL_MISS
        )

    # One batched judgment call fires per proposal, so `result.execution` is this
    # fixture's whole batch cost/latency (§8). `execution is None` on a clean-empty
    # (all-screened) result ⇒ no LLM call fired ⇒ `null` token/latency. NOTE:
    # JudgmentExecution has FOUR scalar token fields, not a `token_usage` object —
    # assemble the dict here (reading `execution.token_usage` would AttributeError).
    ex = result.execution
    if ex is not None:
        token_usage: Optional[Dict[str, int]] = {
            "input": ex.token_input,
            "output": ex.token_output,
            "cache_read": ex.token_cache_read,
            "cache_creation": ex.token_cache_creation,
        }
        elapsed_ms: Optional[int] = ex.elapsed_ms
    else:
        token_usage = None
        elapsed_ms = None

    # The metrics record — the minimal shape metrics.py consumes.
    outcome = {
        "kind": fx["kind"],
        "expected_tenable": fx["expected_tenable"],
        "expected_surfaced": fx["expected_surfaced"],
        "judged": judged,
        "actual_tenable": actual_tenable,
        "actual_surfaced": actual_surfaced,
        "confidence": confidence,
    }
    # The report record — the full per-fixture trace 4b recomputes the curve from.
    fixture = {
        "proposal": fx["proposal"],
        "candidate": candidate_slug,
        "kind": fx["kind"],
        "expected_candidate_judged": fx["expected_candidate_judged"],
        "expected_tenable": fx["expected_tenable"],
        "expected_surfaced": fx["expected_surfaced"],
        "judged": judged,
        "actual_tenable": actual_tenable,
        "actual_surfaced": actual_surfaced,
        "confidence": confidence,
        "reason": reason,
        # 4b additions — the floor-calibration input + the prompt-fit budget/latency.
        "similarity": similarity,
        "token_usage": token_usage,
        "elapsed_ms": elapsed_ms,
    }
    return {"outcome": outcome, "fixture": fixture}


def run_conflict_eval(
    oracle: Dict[str, Any],
    entries_by_slug: Dict[str, Any],
    provider: Any,
    vstore: Any,
    store: Any,
    judge: Callable,
    *,
    floor: float = CONFLICT_SIMILARITY_FLOOR,
    top_k: int = CONFLICT_TOP_K,
    surface_threshold: float = CONFLICT_SURFACE_THRESHOLD,
) -> Any:
    """Runs every conflict fixture through the real facade and computes the metrics.

    For each ``conflict`` fixture: look up the proposal :class:`ParsedEntry` by slug,
    drive the shipped :func:`~mitos.conflict.run_conflict_check`, derive one outcome
    record for the NAMED candidate, and roll the outcomes up into the aggregate +
    calibration curve. A returned :class:`~mitos.conflict.Unavailable` (substrate or
    judge degradation) short-circuits the whole run — it is environmental, so the eval
    hands it back for the caller to skip loudly (never a measured signal).

    Args:
        oracle: The parsed ``oracle.semantic.json``.
        entries_by_slug: ``{slug: ParsedEntry}`` over the corpus (the proposal source).
        provider: A ``GeminiEmbeddingProvider``.
        vstore: A ``QdrantVectorStore`` bound to the populated test collection.
        store: The populated ``GraphStore`` (2a's computed-state source).
        judge: The bound SONNET judge (from :func:`make_live_judge`).
        floor: The similarity floor passed to the facade (default the calibrated
            ``CONFLICT_SIMILARITY_FLOOR``; the probe run passes ``floor=0.0``).
        top_k: The judged-batch cap (default ``CONFLICT_TOP_K``).
        surface_threshold: The CONF-D4 confidence gate (default
            ``CONFLICT_SURFACE_THRESHOLD``).

    Returns:
        A report dict ``{provenance, params, fixtures, aggregate, calibration}`` on a
        healthy run, or the :class:`~mitos.conflict.Unavailable` verbatim if any
        fixture degraded (the caller skips loudly).
    """
    fixtures_out: List[Dict[str, Any]] = []
    outcomes: List[Dict[str, Any]] = []
    judgment_model: Optional[str] = None

    for fx in oracle["conflict"]:
        entry = entries_by_slug[fx["proposal"]]
        result = run_conflict_check(
            entry,
            embed_provider=provider,
            vector_store=vstore,
            store=store,
            judge=judge,
            floor=floor,
            top_k=top_k,
            surface_threshold=surface_threshold,
        )
        if isinstance(result, Unavailable):
            # Environmental degradation (embedding / vector-store / judge timeout-or-5xx):
            # NOT a measured outcome. Hand it back — the caller turns it into a loud skip.
            return result

        # Stamp the judge alias from the first live execution (P19-clean — the public
        # field off JudgmentExecution, never the private module constant; Warning D).
        if result.execution is not None and judgment_model is None:
            judgment_model = result.execution.model_alias

        derived = _outcome_and_fixture(fx, result)
        outcomes.append(derived["outcome"])
        fixtures_out.append(derived["fixture"])

    aggregate = {
        "not_tenable_recall": not_tenable_recall(outcomes),
        "not_tenable_precision": not_tenable_precision(outcomes),
        "same_polarity_fp_rate": same_polarity_fp_rate(outcomes),
    }
    calibration = confidence_calibration_curve(outcomes)

    return {
        "provenance": H.provenance(
            judgment_model=judgment_model, prompt_version=CONFLICT_PROMPT_VERSION
        ),
        "params": {
            "floor": floor,
            "top_k": top_k,
            "surface_threshold": surface_threshold,
        },
        "fixtures": fixtures_out,
        "aggregate": aggregate,
        "calibration": calibration,
    }


# ---------------------------------------------------------------------------
# Baseline diff (soft gate) — direction-aware, conflict-named
# ---------------------------------------------------------------------------

def conflict_baseline_diff(
    report: Dict[str, Any], baseline: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Compares a run's conflict aggregate to a stored baseline, flagging regressions.

    The conflict twin of :func:`_semantic_harness.baseline_diff`: a SOFT gate that
    returns the list of regressions (empty == clean) for the caller to WARN on; it never
    hard-fails and never auto-accepts. Direction-aware over the conflict metric names —
    ``not_tenable_recall`` / ``not_tenable_precision`` regress on a drop,
    ``same_polarity_fp_rate`` on a rise. Per-metric bands are read from the baseline's
    ``bands`` block so tolerance is versioned with the numbers.

    Args:
        report: A report dict from :func:`run_conflict_eval`.
        baseline: The parsed ``conflict.baseline.metrics.json``
            (``{provenance, params, aggregate, bands}``).

    Returns:
        A list of regression dicts, each ``{metric, baseline, current, band, direction}``.
    """
    regressions: List[Dict[str, Any]] = []
    base_agg = baseline.get("aggregate", {})
    bands = baseline.get("bands", {})
    current = report["aggregate"]
    for metric in CONFLICT_HIGHER_IS_BETTER + CONFLICT_LOWER_IS_BETTER:
        if metric not in base_agg:
            continue
        band = bands.get(metric, 0.0)
        base_val = base_agg[metric]
        cur_val = current[metric]
        if metric in CONFLICT_HIGHER_IS_BETTER:
            regressed = cur_val < base_val - band
            direction = "drop"
        else:
            regressed = cur_val > base_val + band
            direction = "rise"
        if regressed:
            regressions.append(
                {
                    "metric": metric,
                    "baseline": base_val,
                    "current": cur_val,
                    "band": band,
                    "direction": direction,
                }
            )
    return regressions


# ---------------------------------------------------------------------------
# Conflict baseline IO (separate file — Warning A / plan §8)
# ---------------------------------------------------------------------------

def load_conflict_baseline() -> Optional[Dict[str, Any]]:
    """Loads ``conflict.baseline.metrics.json``, or None if it has not been seeded yet.

    4a deliberately ships NO seeded baseline — the judge is live/costly and the floor is
    still provisional, so a meaningful baseline is 4b's calibrated, reviewed act. Absent
    → the soft-diff test skips loudly.
    """
    if not os.path.exists(CONFLICT_BASELINE_PATH):
        return None
    with open(CONFLICT_BASELINE_PATH, encoding="utf-8") as f:
        return json.load(f)


def write_conflict_baseline(report: Dict[str, Any], bands: Dict[str, float]) -> None:
    """Freezes a run's conflict aggregate as the reviewed baseline (explicit-flag only).

    Called ONLY under ``MITOS_UPDATE_BASELINE=1`` — never from an ordinary run, so a
    quota-degraded or jittery judge run can never silently become ground truth. A
    SEPARATE file from the retrieval baseline (different metrics, cadence, stochasticity),
    reviewed in isolation. Carries ``params`` (floor/top_k/surface_threshold) instead of
    the retrieval baseline's ``k`` — conflict has no k.

    Args:
        report: A report dict from :func:`run_conflict_eval`.
        bands: Per-metric regression tolerances stored alongside the numbers.
    """
    payload = {
        "provenance": report["provenance"],
        "params": report["params"],
        "aggregate": report["aggregate"],
        "bands": bands,
    }
    with open(CONFLICT_BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# Human summary
# ---------------------------------------------------------------------------

def conflict_human_summary(report: Dict[str, Any]) -> str:
    """Renders a short human-readable summary of a conflict report for the test log.

    Loudly flags any ``unexpected_retrieval_miss`` — a fixture whose candidate was
    expected to reach the judge but did not at the active floor (a soft 4b signal,
    never a CI red).

    Args:
        report: A report dict from :func:`run_conflict_eval`.

    Returns:
        A multi-line string: provenance header, per-fixture verdict line, aggregate,
        and a loud retrieval-miss callout when present.
    """
    p = report["provenance"]
    params = report["params"]
    lines = [
        f"Layer-B conflict eval — judge={p['judgment_model']} "
        f"prompt={p['prompt_version']} embed={p['embedding_model']} @ mitos "
        f"{p['mitos_version']} ({p['commit_sha']}, dirty={p['dirty_tree']})",
        f"floor={params['floor']} top_k={params['top_k']} "
        f"surface_threshold={params['surface_threshold']}",
    ]
    misses: List[str] = []
    for f in report["fixtures"]:
        conf = f["confidence"]
        conf_s = f"{conf:.2f}" if conf is not None else "  — "
        sim = f.get("similarity")
        sim_s = f"{sim:.4f}" if sim is not None else "  —   "
        lines.append(
            f"  {f['kind']:26} judged={str(f['judged']):5} "
            f"tenable={str(f['actual_tenable']):5} surfaced={str(f['actual_surfaced']):5} "
            f"conf={conf_s} sim={sim_s} [{f['reason']}]  {f['proposal']} ✗ {f['candidate']}"
        )
        if f["reason"] == REASON_RETRIEVAL_MISS:
            misses.append(f"  {f['proposal']} ✗ {f['candidate']} ({f['kind']})")
    agg = report["aggregate"]
    lines.append(
        f"AGGREGATE: not_tenable_recall={agg['not_tenable_recall']:.3f} "
        f"not_tenable_precision={agg['not_tenable_precision']:.3f} "
        f"same_polarity_fp_rate={agg['same_polarity_fp_rate']:.3f}"
    )

    # Calibration readout (§5): the contradiction similarity table that SETS the floor,
    # plus the recommended value vs the landed constant. `similarity`-bearing report ⇒
    # a real readout; a legacy report without it ⇒ the recommendation is None.
    contradictions = [
        f for f in report["fixtures"]
        if f["expected_tenable"] is False and f["judged"] and f.get("similarity") is not None
    ]
    if contradictions:
        lines.append("CALIBRATION (contradiction similarities — the floor's binding set):")
        for f in sorted(contradictions, key=lambda r: r["similarity"]):
            lines.append(
                f"  sim={f['similarity']:.4f}  {f['kind']:26} "
                f"{f['proposal']} ✗ {f['candidate']}"
            )
        rec = recommend_floor(report["fixtures"])
        rec_s = f"{rec:.4f}" if rec is not None else "None (no judged contradictions)"
        lines.append(
            f"  → recommended floor = min(contradiction sim) − margin = {rec_s}   "
            f"(landed CONFLICT_SIMILARITY_FLOOR = {CONFLICT_SIMILARITY_FLOOR})"
        )

    if misses:
        lines.append(
            "⚠ RETRIEVAL MISSES (expected to be judged, did not retrieve above the "
            "floor — soft signal for floor calibration, NOT a defect):"
        )
        lines.extend(misses)
    return "\n".join(lines)
