"""Deterministic retrieval-metric math for the golden dataset (Layer B).

Pure functions over ranked slug lists and expected-outcome sets — no services, no
embeddings, no Qdrant, no randomness. This is the measurement core of Layer B's
retrieval eval: the live harness (`_semantic_harness.py`) produces ranked results
against real embeddings, and these functions turn them into the numbers the metrics
report and baseline diff are built from. Kept in a separate module precisely so the
math is unit-testable in bare CI (`test_metrics.py`) while the flaky, live parts stay
integration-gated.

Design decisions pinned here (see MITOS_GOLDEN_DATASET_SPEC Part C):

* **Set-based at the k boundary, never rank-exact.** Embedding order drifts by design
  and Qdrant score ties make the exact order of near-equal results unstable, so every
  metric is defined over the *set* of the top-k slugs, not their precise ranks (MRR is
  the one exception — it needs the rank of the first relevant hit, which is stable
  because a genuine relevant hit sits well clear of the tie band).
* **Duplicates are collapsed first.** A ranked list may in principle repeat a slug;
  `_dedupe_preserve_order` collapses it so top-k counts distinct results.
* **Empty-set conventions are explicit and documented** (see each function) so a
  degenerate fixture yields a defined number, never a ``ZeroDivisionError``.
"""

from typing import Any, Dict, List, Optional, Sequence, Set


def _dedupe_preserve_order(slugs: Sequence[str]) -> List[str]:
    """Collapses repeated slugs, keeping first-seen order.

    Args:
        slugs: A ranked list of result slugs, best-first, possibly with repeats.

    Returns:
        The same slugs with later duplicates removed, order preserved.
    """
    seen: Set[str] = set()
    out: List[str] = []
    for s in slugs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def recall_at_k(ranked: Sequence[str], expect_relevant: Sequence[str], k: int) -> float:
    """Computes recall@k — the fraction of relevant slugs present in the top-k.

    Empty-set convention: an empty ``expect_relevant`` returns ``1.0`` (vacuously
    satisfied — there is nothing to fail to retrieve). A fixture that genuinely
    tests recall always names at least one relevant slug.

    Args:
        ranked: Ranked result slugs, best-first (duplicates collapsed internally).
        expect_relevant: Slugs that should appear in the top-k.
        k: The cutoff. Values larger than the result count simply take all results.

    Returns:
        A value in ``[0.0, 1.0]``.
    """
    relevant = set(expect_relevant)
    if not relevant:
        return 1.0
    top_k = set(_dedupe_preserve_order(ranked)[:k])
    return len(relevant & top_k) / len(relevant)


def precision_at_k(ranked: Sequence[str], expect_relevant: Sequence[str], k: int) -> float:
    """Computes precision@k — the fraction of the top-k that are relevant.

    The denominator is ``min(k, number_of_distinct_results)`` so a run that returns
    fewer than k results is not penalised for the empty tail. Empty-set convention:
    an empty result list returns ``0.0`` (nothing retrieved → nothing correct).

    Args:
        ranked: Ranked result slugs, best-first (duplicates collapsed internally).
        expect_relevant: Slugs considered relevant.
        k: The cutoff.

    Returns:
        A value in ``[0.0, 1.0]``.
    """
    deduped = _dedupe_preserve_order(ranked)
    denom = min(k, len(deduped))
    if denom == 0:
        return 0.0
    relevant = set(expect_relevant)
    top_k = deduped[:k]
    hits = sum(1 for s in top_k if s in relevant)
    return hits / denom


def mrr(ranked: Sequence[str], expect_relevant: Sequence[str]) -> float:
    """Computes the reciprocal rank of the first relevant slug.

    Ranks are 1-based: a relevant slug in first position scores ``1.0``, second
    ``0.5``, and so on. Returns ``0.0`` when no relevant slug appears at all.

    Args:
        ranked: Ranked result slugs, best-first (duplicates collapsed internally).
        expect_relevant: Slugs considered relevant.

    Returns:
        A value in ``[0.0, 1.0]``.
    """
    relevant = set(expect_relevant)
    for i, slug in enumerate(_dedupe_preserve_order(ranked), start=1):
        if slug in relevant:
            return 1.0 / i
    return 0.0


def hard_negative_fp_rate(
    ranked: Sequence[str], expect_absent: Sequence[str], k: int
) -> float:
    """Computes the fraction of hard-negative slugs that crack the top-k.

    This is the precision-side guard: ``expect_absent`` names anti-relevant slugs
    (plausible-but-wrong neighbours) that a healthy ranking keeps out of the top-k.
    Empty-set convention: an empty ``expect_absent`` returns ``0.0`` (no hard
    negatives declared → none can leak).

    Args:
        ranked: Ranked result slugs, best-first (duplicates collapsed internally).
        expect_absent: Slugs that must NOT appear in the top-k.
        k: The cutoff.

    Returns:
        A value in ``[0.0, 1.0]`` — 0.0 is the healthy outcome.
    """
    absent = set(expect_absent)
    if not absent:
        return 0.0
    top_k = set(_dedupe_preserve_order(ranked)[:k])
    return len(absent & top_k) / len(absent)


def evaluate_fixture(
    ranked: Sequence[str],
    expect_relevant: Sequence[str],
    expect_absent: Sequence[str],
    k: int,
) -> Dict[str, float]:
    """Computes the full metric set for one retrieval fixture.

    A convenience combiner used by both the unit tests and the live harness so the
    per-fixture metric block is defined in exactly one place.

    Args:
        ranked: Ranked result slugs, best-first.
        expect_relevant: Slugs that should appear in the top-k.
        expect_absent: Hard-negative slugs that must not appear in the top-k.
        k: The cutoff.

    Returns:
        A dict with ``recall_at_k``, ``precision_at_k``, ``mrr``, and
        ``hard_negative_fp_rate`` keys.
    """
    return {
        "recall_at_k": recall_at_k(ranked, expect_relevant, k),
        "precision_at_k": precision_at_k(ranked, expect_relevant, k),
        "mrr": mrr(ranked, expect_relevant),
        "hard_negative_fp_rate": hard_negative_fp_rate(ranked, expect_absent, k),
    }


# =========================================================================== #
# Conflict-sensor metric math (Part C / Conflict-sensor vision §6.3)
# --------------------------------------------------------------------------- #
# The retrieval math above scores a ranked slug list; these score the *conflict
# judgment* — did the sensor flag the real contradictions, hold fire on the merely
# agreeing pairs, and calibrate its confidence honestly? Same discipline: pure
# functions, no services, documented empty-set conventions, unit-tested in bare CI.
#
# Each function takes a list of per-fixture **outcome records** — one dict per
# conflict fixture, produced by the live harness (`_conflict_harness.run_conflict_eval`)
# from the real `run_conflict_check` facade output. The record shape:
#
#   {
#     "kind": str,                    # the fixture's kind (e.g. "same-polarity-agreement")
#     "expected_tenable": bool|None,  # oracle: the judge's verdict IF judged; None when not judged
#     "expected_surfaced": bool,      # oracle: the final ≥0.85 gate outcome
#     "judged": bool,                 # did the NAMED candidate reach the judge?
#     "actual_tenable": bool|None,    # the judge's real verdict; None when not judged
#     "actual_surfaced": bool,        # the facade's real gate result (read off `pair.surfaced`)
#     "confidence": float|None,       # the judge's raw confidence; None when not judged
#   }
#
# "Not judged" (candidate absent from `judged_pairs`) means either an expected
# declared-target drop (fixture 4) or an unexpected retrieval miss at the provisional
# floor — the harness distinguishes them; the metrics only care that the pair carries
# no verdict (its `actual_*` fields are None/False and it is excluded from the
# judged-only aggregates below).
# =========================================================================== #


def not_tenable_recall(outcomes: Sequence[Dict[str, Any]]) -> float:
    """Fraction of genuine (judged) contradictions the sensor actually surfaced.

    "Did we catch the real contradictions?" The denominator is the fixtures that
    are genuinely not-tenable AND reached the judge (``expected_tenable is False and
    judged``); the numerator is those the facade surfaced (``actual_surfaced``).
    A not-judged contradiction (declared-dropped, or a retrieval miss) is excluded
    from the denominator — it was never given to the judge, so it can't count as a
    recall failure of the *judge*. Higher-is-better.

    Empty-set convention: no genuinely-not-tenable-and-judged fixtures ⇒ ``1.0``
    (vacuously — there was nothing to catch).

    Args:
        outcomes: Per-fixture outcome records (see module header for the shape).

    Returns:
        A value in ``[0.0, 1.0]``.
    """
    judged_contradictions = [
        o for o in outcomes if o["expected_tenable"] is False and o["judged"]
    ]
    if not judged_contradictions:
        return 1.0
    surfaced = sum(1 for o in judged_contradictions if o["actual_surfaced"])
    return surfaced / len(judged_contradictions)


def not_tenable_precision(outcomes: Sequence[Dict[str, Any]]) -> float:
    """Fraction of what the sensor surfaced that was a genuine contradiction.

    "Of what we flagged, how much was real?" The denominator is the fixtures the
    facade actually surfaced (``actual_surfaced``); the numerator is those that were
    genuinely not-tenable (``expected_tenable is False``). A false positive — a
    tenable pair the judge wrongly surfaced — drops it. Higher-is-better.

    Empty-set convention: nothing surfaced ⇒ ``1.0`` (vacuously — nothing was
    wrongly flagged).

    Args:
        outcomes: Per-fixture outcome records (see module header for the shape).

    Returns:
        A value in ``[0.0, 1.0]``.
    """
    surfaced = [o for o in outcomes if o["actual_surfaced"]]
    if not surfaced:
        return 1.0
    genuine = sum(1 for o in surfaced if o["expected_tenable"] is False)
    return genuine / len(surfaced)


def same_polarity_fp_rate(outcomes: Sequence[Dict[str, Any]]) -> float:
    """Fraction of same-polarity-agreement fixtures the sensor wrongly surfaced.

    The #34 CONF-D4 guard: two decisions that merely *agree* (same-polarity, e.g.
    a base decision and its stricter refinement) must NEVER be flagged as a conflict.
    Of the ``kind == "same-polarity-agreement"`` fixtures, the fraction that got
    surfaced — which should be ``0.0``. Lower-is-better.

    Empty-set convention: no same-polarity fixtures ⇒ ``0.0`` (none present → none
    can be a false positive).

    Args:
        outcomes: Per-fixture outcome records (see module header for the shape).

    Returns:
        A value in ``[0.0, 1.0]`` — 0.0 is the healthy outcome.
    """
    same_polarity = [o for o in outcomes if o["kind"] == "same-polarity-agreement"]
    if not same_polarity:
        return 0.0
    false_positives = sum(1 for o in same_polarity if o["actual_surfaced"])
    return false_positives / len(same_polarity)


def _confidence_bin_index(confidence: float, n_bins: int) -> int:
    """Maps a confidence in ``[0, 1]`` to its bin index in ``[0, n_bins - 1]``.

    Bins are equal-width over ``[0, 1]``; the top bin is right-closed so a confidence
    of exactly ``1.0`` lands in the last bin rather than overflowing.

    Args:
        confidence: The judge's confidence, in ``[0, 1]``.
        n_bins: The number of equal-width bins.

    Returns:
        The bin index, clamped to ``[0, n_bins - 1]``.
    """
    return min(int(confidence * n_bins), n_bins - 1)


# The recall-first jitter margin subtracted from the min-contradiction similarity when
# recommending the floor (plan D1 / §14). Small and deliberate: it hedges recall against
# future corpus/embedding drift (err LOW), without dropping the floor into the
# clearly-irrelevant tail (which would re-admit noise and raise judge cost/FP surface).
# 0.03 sits mid-range of the plan's ≈0.02–0.05 band. It is a RECOMMENDATION margin — the
# actual landed CONFLICT_SIMILARITY_FLOOR is a reviewed constant (its Calibration block
# records the measured min, this margin, and the resulting value).
DEFAULT_FLOOR_MARGIN = 0.03


def recommend_floor(
    records: Sequence[Dict[str, Any]], margin: float = DEFAULT_FLOOR_MARGIN
) -> Optional[float]:
    """Recommends the similarity floor from measured per-fixture report records.

    Recall-first (OpEcon §11): the highest cutoff that still admits *every* known
    contradiction that reached the judge — ``min(similarity)`` over the fixtures whose
    oracle ``expected_tenable is False`` and that were ``judged`` — minus a small jitter
    ``margin`` (err low, hedge recall against drift). The genuine / cross-domain /
    multilingual contradictions set the floor; tenable pairs and screened candidates do
    not constrain it.

    This reads the **report** records (which carry ``similarity``), NOT the leaner metrics
    ``outcome`` records (which do not). A judged contradiction always carries a non-``None``
    ``similarity`` (its candidate reached the judge); the ``None`` guard is defensive.

    Empty-set convention: no judged contradiction fixtures ⇒ ``None`` (there is nothing to
    calibrate against — the caller keeps the standing floor rather than inventing one).
    Never raises.

    Args:
        records: Per-fixture report records (each ``{expected_tenable, judged, similarity,
            ...}`` — the ``fixtures`` list of a ``run_conflict_eval`` report).
        margin: The recall-first jitter margin subtracted from the min (default
            :data:`DEFAULT_FLOOR_MARGIN`).

    Returns:
        The recommended floor as a ``float``, or ``None`` when no judged contradiction
        fixture is present.
    """
    sims = [
        r["similarity"]
        for r in records
        if r.get("expected_tenable") is False
        and r.get("judged")
        and r.get("similarity") is not None
    ]
    if not sims:
        return None
    return min(sims) - margin


def confidence_calibration_curve(
    outcomes: Sequence[Dict[str, Any]], n_bins: int = 4
) -> List[Dict[str, Any]]:
    """Bins the judged fixtures by confidence and reports the observed conflict rate.

    The raw (confidence ↔ actual-not-tenable) relationship 4b reads to validate the
    pinned ``CONFLICT_SURFACE_THRESHOLD = 0.85`` (CONF-D4). ``observed_not_tenable_fraction``
    is the fraction of the bin the judge itself ruled not-tenable (``actual_tenable is
    False``) — the judge's own verdicts, NOT the oracle ground truth (``expected_tenable``);
    for a well-calibrated judge this should rise with stated confidence, and the 0.85 gate
    should sit where it is high. 4b, wanting a true reliability diagram against ground
    truth, recomputes from the per-fixture report records (which carry both
    ``expected_tenable`` and the raw ``confidence``) rather than reading this curve blind.

    **Honestly sparse at n=6.** With only six fixtures (fewer once the declared-drop
    and any retrieval miss are excluded — those are not judged, so carry no
    confidence), most bins hold zero or one fixture. The curve is a scaffold that
    grows dense as the conflict corpus grows; do not read a trend into it yet. Bins
    with no members are still returned (deterministic structure) with ``count = 0``
    and ``None`` for the two fraction/mean fields.

    Only **judged** fixtures (``judged`` true and a non-``None`` ``confidence``) are
    binned; a declared-drop / retrieval-miss fixture has no verdict to place.

    Empty-set convention: an empty ``outcomes`` list ⇒ ``[]`` (no curve at all).
    A non-empty run in which nothing was judged returns the full ``n_bins`` scaffold
    with every bin at ``count = 0``.

    Args:
        outcomes: Per-fixture outcome records (see module header for the shape).
        n_bins: The number of equal-width confidence bins over ``[0, 1]``.

    Returns:
        A list of ``n_bins`` bin dicts (or ``[]`` for empty input), each
        ``{lo, hi, count, observed_not_tenable_fraction, mean_confidence}``, where
        the last two are ``None`` for an empty bin.
    """
    if not outcomes:
        return []
    judged = [
        o for o in outcomes if o["judged"] and o.get("confidence") is not None
    ]
    curve: List[Dict[str, Any]] = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        members = [
            o for o in judged if _confidence_bin_index(o["confidence"], n_bins) == i
        ]
        count = len(members)
        observed: Optional[float]
        mean_conf: Optional[float]
        if count:
            not_tenable = sum(1 for o in members if o["actual_tenable"] is False)
            observed = not_tenable / count
            mean_conf = sum(o["confidence"] for o in members) / count
        else:
            observed = None
            mean_conf = None
        curve.append(
            {
                "lo": lo,
                "hi": hi,
                "count": count,
                "observed_not_tenable_fraction": observed,
                "mean_confidence": mean_conf,
            }
        )
    return curve
