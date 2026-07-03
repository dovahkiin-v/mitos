"""Bare-CI unit tests for the Layer-B retrieval-metric math (`metrics.py`).

Deterministic: synthetic ranked lists with hand-computed expected values, no
services. These pin the empty-set/edge conventions the live harness depends on so a
future refactor of the math can never silently move a metric. Every edge case Fable's
review flagged (#9) is exercised: empty results, ``k`` larger than the result count,
empty ``expect_relevant``, duplicate slugs, and hard-negative accounting with fewer
than ``k`` results.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from metrics import (  # noqa: E402
    DEFAULT_FLOOR_MARGIN,
    _dedupe_preserve_order,
    confidence_calibration_curve,
    evaluate_fixture,
    hard_negative_fp_rate,
    mrr,
    not_tenable_precision,
    not_tenable_recall,
    precision_at_k,
    recall_at_k,
    recommend_floor,
    same_polarity_fp_rate,
)

# A small stable universe. Ranked lists below are hand-authored, best-first.
REL = ["a", "b"]          # relevant
ABS = ["x", "y"]          # hard negatives


class TestRecallAtK:
    def test_all_relevant_in_top_k(self):
        assert recall_at_k(["a", "b", "c"], REL, k=5) == 1.0

    def test_half_relevant_in_top_k(self):
        assert recall_at_k(["a", "c", "d"], REL, k=5) == 0.5

    def test_relevant_below_cutoff_missed(self):
        # 'b' sits at rank 4 but k=3 → not recalled.
        assert recall_at_k(["a", "c", "d", "b"], REL, k=3) == 0.5

    def test_empty_results(self):
        assert recall_at_k([], REL, k=5) == 0.0

    def test_empty_expect_relevant_is_vacuously_one(self):
        # Documented convention: nothing to fail to retrieve → 1.0.
        assert recall_at_k(["a", "b"], [], k=5) == 1.0

    def test_k_larger_than_results(self):
        assert recall_at_k(["a", "b"], REL, k=50) == 1.0


class TestPrecisionAtK:
    def test_all_top_k_relevant(self):
        assert precision_at_k(["a", "b"], REL, k=2) == 1.0

    def test_half_top_k_relevant(self):
        assert precision_at_k(["a", "c"], REL, k=2) == 0.5

    def test_denominator_is_min_k_and_result_count(self):
        # Only 2 distinct results but k=5 → denom is 2, not 5, so 1 hit → 0.5.
        assert precision_at_k(["a", "z"], REL, k=5) == 0.5

    def test_empty_results(self):
        assert precision_at_k([], REL, k=5) == 0.0

    def test_no_relevant_hits(self):
        assert precision_at_k(["c", "d"], REL, k=2) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mrr(["a", "c"], REL) == 1.0

    def test_second_position(self):
        assert mrr(["c", "a"], REL) == 0.5

    def test_third_position(self):
        assert mrr(["c", "d", "b"], REL) == 1.0 / 3.0

    def test_no_relevant(self):
        assert mrr(["c", "d"], REL) == 0.0

    def test_empty_results(self):
        assert mrr([], REL) == 0.0


class TestHardNegativeFPRate:
    def test_no_leak(self):
        assert hard_negative_fp_rate(["a", "b", "c"], ABS, k=5) == 0.0

    def test_one_of_two_leaks(self):
        assert hard_negative_fp_rate(["a", "x", "c"], ABS, k=5) == 0.5

    def test_both_leak(self):
        assert hard_negative_fp_rate(["x", "y"], ABS, k=5) == 1.0

    def test_leak_below_cutoff_does_not_count(self):
        # 'x' is at rank 4 but k=3 → does not crack the top-k.
        assert hard_negative_fp_rate(["a", "b", "c", "x"], ABS, k=3) == 0.0

    def test_empty_expect_absent_is_zero(self):
        assert hard_negative_fp_rate(["a", "b"], [], k=5) == 0.0

    def test_fewer_results_than_k(self):
        # Only 2 results, neither a hard negative → 0.0 despite k=5.
        assert hard_negative_fp_rate(["a", "b"], ABS, k=5) == 0.0


class TestDuplicateHandling:
    def test_dedupe_preserves_order(self):
        assert _dedupe_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_duplicates_do_not_inflate_precision(self):
        # Without dedupe, ["a","a","c"] top-3 would read as 2/3 relevant; deduped it
        # is a/c → 1 hit over min(3, 2)=2 → 0.5.
        assert precision_at_k(["a", "a", "c"], REL, k=3) == 0.5

    def test_duplicate_relevant_does_not_break_recall(self):
        assert recall_at_k(["a", "a", "b"], REL, k=5) == 1.0

    def test_duplicate_pushes_real_result_past_cutoff(self):
        # Deduped ["a","b"] → 'b' survives at rank 2; a naive [:k] on the raw list
        # ["a","a","b"] with k=2 would drop 'b'. Dedupe-first prevents that.
        assert recall_at_k(["a", "a", "b"], REL, k=2) == 1.0


class TestEvaluateFixture:
    def test_combiner_matches_individual_functions(self):
        ranked = ["a", "x", "c", "b"]
        result = evaluate_fixture(ranked, REL, ABS, k=5)
        assert result == {
            "recall_at_k": recall_at_k(ranked, REL, k=5),
            "precision_at_k": precision_at_k(ranked, REL, k=5),
            "mrr": mrr(ranked, REL),
            "hard_negative_fp_rate": hard_negative_fp_rate(ranked, ABS, k=5),
        }

    def test_combiner_concrete_values(self):
        # ranked a,x,c,b  rel {a,b} abs {x,y}  k=5
        #   recall: both a,b in top5 → 1.0
        #   precision: 2 relevant of 4 distinct → 0.5
        #   mrr: 'a' at rank 1 → 1.0
        #   hard-neg fp: 'x' leaks (1 of 2) → 0.5
        result = evaluate_fixture(["a", "x", "c", "b"], REL, ABS, k=5)
        assert result == {
            "recall_at_k": 1.0,
            "precision_at_k": 0.5,
            "mrr": 1.0,
            "hard_negative_fp_rate": 0.5,
        }


# =========================================================================== #
# Conflict-sensor metric math (Part C / Conflict-sensor vision §6.3)
# --------------------------------------------------------------------------- #
# Hand-authored outcome records with hand-computed expected values; no services.
# These pin the conflict-metric empty-set/edge conventions the live conflict harness
# (`_conflict_harness.run_conflict_eval`) depends on. Outcome-record shape is defined
# in the metrics.py conflict-section header.
# =========================================================================== #


def _oc(
    kind="genuine-contradiction",
    expected_tenable=False,
    expected_surfaced=True,
    judged=True,
    actual_tenable=False,
    actual_surfaced=True,
    confidence=0.9,
):
    """Builds one conflict outcome record with contradiction-catch defaults."""
    return {
        "kind": kind,
        "expected_tenable": expected_tenable,
        "expected_surfaced": expected_surfaced,
        "judged": judged,
        "actual_tenable": actual_tenable,
        "actual_surfaced": actual_surfaced,
        "confidence": confidence,
    }


# A not-judged fixture (declared drop / retrieval miss): no verdict, nothing surfaced.
def _not_judged(kind="declared-contradiction"):
    return _oc(
        kind=kind,
        expected_tenable=None,
        expected_surfaced=False,
        judged=False,
        actual_tenable=None,
        actual_surfaced=False,
        confidence=None,
    )


class TestNotTenableRecall:
    def test_all_judged_contradictions_surfaced(self):
        outs = [_oc(actual_surfaced=True), _oc(actual_surfaced=True)]
        assert not_tenable_recall(outs) == 1.0

    def test_half_surfaced(self):
        # Two judged contradictions, one missed (judged tenable-together, not surfaced).
        outs = [
            _oc(actual_surfaced=True),
            _oc(actual_tenable=True, actual_surfaced=False),
        ]
        assert not_tenable_recall(outs) == 0.5

    def test_not_judged_contradiction_excluded_from_denominator(self):
        # A judged+surfaced contradiction, plus a declared-dropped one (judged=False) and
        # a tenable pair — only the first counts → 1/1 = 1.0.
        outs = [
            _oc(actual_surfaced=True),
            _not_judged(),
            _oc(kind="same-polarity-agreement", expected_tenable=True,
                actual_tenable=True, actual_surfaced=False),
        ]
        assert not_tenable_recall(outs) == 1.0

    def test_empty_convention_no_judged_contradictions(self):
        # Only a tenable pair → no not-tenable-and-judged fixtures → vacuous 1.0.
        outs = [_oc(kind="same-polarity-agreement", expected_tenable=True,
                    actual_tenable=True, actual_surfaced=False)]
        assert not_tenable_recall(outs) == 1.0

    def test_empty_list(self):
        assert not_tenable_recall([]) == 1.0


class TestNotTenablePrecision:
    def test_all_surfaced_genuine(self):
        outs = [_oc(actual_surfaced=True), _oc(actual_surfaced=True)]
        assert not_tenable_precision(outs) == 1.0

    def test_false_positive_drops_precision(self):
        # Two surfaced: one genuine contradiction, one wrongly-surfaced tenable pair.
        outs = [
            _oc(actual_surfaced=True),
            _oc(kind="same-polarity-agreement", expected_tenable=True,
                actual_tenable=False, actual_surfaced=True),
        ]
        assert not_tenable_precision(outs) == 0.5

    def test_empty_convention_nothing_surfaced(self):
        outs = [
            _oc(actual_surfaced=False),
            _oc(kind="same-polarity-agreement", expected_tenable=True,
                actual_tenable=True, actual_surfaced=False),
        ]
        assert not_tenable_precision(outs) == 1.0

    def test_empty_list(self):
        assert not_tenable_precision([]) == 1.0


class TestSamePolarityFPRate:
    def test_no_false_positive(self):
        outs = [_oc(kind="same-polarity-agreement", expected_tenable=True,
                    actual_tenable=True, actual_surfaced=False)]
        assert same_polarity_fp_rate(outs) == 0.0

    def test_the_34_case_surfaced(self):
        # The #34 must-not-flag failure: a same-polarity pair got surfaced → FP rate 1.0.
        outs = [_oc(kind="same-polarity-agreement", expected_tenable=True,
                    actual_tenable=False, actual_surfaced=True)]
        assert same_polarity_fp_rate(outs) == 1.0

    def test_mixed_same_polarity(self):
        outs = [
            _oc(kind="same-polarity-agreement", expected_tenable=True,
                actual_tenable=True, actual_surfaced=False),
            _oc(kind="same-polarity-agreement", expected_tenable=True,
                actual_tenable=False, actual_surfaced=True),
        ]
        assert same_polarity_fp_rate(outs) == 0.5

    def test_ignores_other_kinds(self):
        # A surfaced genuine contradiction is NOT a same-polarity false positive.
        outs = [
            _oc(actual_surfaced=True),
            _oc(kind="same-polarity-agreement", expected_tenable=True,
                actual_tenable=True, actual_surfaced=False),
        ]
        assert same_polarity_fp_rate(outs) == 0.0

    def test_empty_convention_no_same_polarity(self):
        outs = [_oc(actual_surfaced=True)]
        assert same_polarity_fp_rate(outs) == 0.0

    def test_empty_list(self):
        assert same_polarity_fp_rate([]) == 0.0


class TestConfidenceCalibrationCurve:
    def test_empty_outcomes_is_empty_list(self):
        assert confidence_calibration_curve([]) == []

    def test_two_distinct_bins(self):
        # n_bins=4 → widths of 0.25. conf 0.1 → bin0 (tenable, not_tenable frac 0.0);
        # conf 0.9 → bin3 (not-tenable, frac 1.0). Bins 1 & 2 stay empty.
        outs = [
            _oc(actual_tenable=True, actual_surfaced=False, confidence=0.1),
            _oc(actual_tenable=False, actual_surfaced=True, confidence=0.9),
        ]
        curve = confidence_calibration_curve(outs, n_bins=4)
        assert len(curve) == 4
        assert curve[0]["lo"] == 0.0 and curve[0]["hi"] == 0.25
        assert curve[0]["count"] == 1
        assert curve[0]["observed_not_tenable_fraction"] == 0.0
        assert curve[0]["mean_confidence"] == 0.1
        assert curve[1]["count"] == 0
        assert curve[1]["observed_not_tenable_fraction"] is None
        assert curve[1]["mean_confidence"] is None
        assert curve[2]["count"] == 0
        assert curve[3]["count"] == 1
        assert curve[3]["observed_not_tenable_fraction"] == 1.0
        assert curve[3]["mean_confidence"] == 0.9

    def test_not_judged_excluded(self):
        # A not-judged fixture (confidence None) is never binned; only the judged one is.
        outs = [_not_judged(), _oc(confidence=0.9)]
        curve = confidence_calibration_curve(outs, n_bins=4)
        assert sum(b["count"] for b in curve) == 1
        assert curve[3]["count"] == 1

    def test_multiple_in_one_bin_mean_and_fraction(self):
        # Two judged fixtures both land in bin3 [0.75, 1.0]: one not-tenable, one tenable →
        # observed fraction 0.5, mean confidence (0.8 + 0.9) / 2 = 0.85.
        outs = [
            _oc(actual_tenable=False, confidence=0.8),
            _oc(actual_tenable=True, actual_surfaced=False, confidence=0.9),
        ]
        curve = confidence_calibration_curve(outs, n_bins=4)
        assert curve[3]["count"] == 2
        assert curve[3]["observed_not_tenable_fraction"] == 0.5
        assert curve[3]["mean_confidence"] == pytest.approx(0.85)

    def test_confidence_one_lands_in_top_bin(self):
        # A confidence of exactly 1.0 must not overflow past the last bin.
        curve = confidence_calibration_curve([_oc(confidence=1.0)], n_bins=4)
        assert curve[3]["count"] == 1
        assert curve[3]["observed_not_tenable_fraction"] == 1.0


# --------------------------------------------------------------------------- #
# recommend_floor — the floor-calibration selector (plan D1). Hand-authored REPORT
# records (which carry `similarity`, unlike the leaner `_oc` outcome records) with
# hand-computed min−margin. Pins the selector, the margin arithmetic, and the
# empty-contradiction-set convention the live calibration test relies on.
# --------------------------------------------------------------------------- #


def _rec(expected_tenable=False, judged=True, similarity=0.7, kind="genuine-contradiction"):
    """Builds one conflict REPORT record (the `fixtures` shape carrying `similarity`)."""
    return {
        "kind": kind,
        "expected_tenable": expected_tenable,
        "judged": judged,
        "similarity": similarity,
    }


class TestRecommendFloor:
    def test_min_contradiction_minus_margin(self):
        # Three judged contradictions at distinct similarities; the min (0.60) sets it.
        recs = [
            _rec(similarity=0.72),
            _rec(similarity=0.60, kind="cross-domain-structural"),
            _rec(similarity=0.81, kind="multilingual"),
        ]
        assert recommend_floor(recs) == pytest.approx(0.60 - DEFAULT_FLOOR_MARGIN)

    def test_explicit_margin_overrides_default(self):
        recs = [_rec(similarity=0.60)]
        assert recommend_floor(recs, margin=0.05) == pytest.approx(0.55)

    def test_tenable_and_screened_fixtures_do_not_constrain(self):
        # A tenable pair (higher-tenable) and a not-judged declared-drop are ignored;
        # only the single judged contradiction at 0.65 counts → 0.65 − margin.
        recs = [
            _rec(similarity=0.65),
            _rec(expected_tenable=True, similarity=0.40, kind="same-polarity-agreement"),
            _rec(expected_tenable=None, judged=False, similarity=None,
                 kind="declared-contradiction"),
        ]
        assert recommend_floor(recs) == pytest.approx(0.65 - DEFAULT_FLOOR_MARGIN)

    def test_none_similarity_is_skipped(self):
        # A contradiction whose candidate never retrieved (similarity None) is defensively
        # skipped; the 0.70 one remains the min.
        recs = [
            _rec(similarity=None),
            _rec(similarity=0.70, kind="cross-domain-structural"),
        ]
        assert recommend_floor(recs) == pytest.approx(0.70 - DEFAULT_FLOOR_MARGIN)

    def test_empty_convention_no_judged_contradictions(self):
        # Only a tenable pair → nothing to calibrate against → None (keep the standing floor).
        recs = [_rec(expected_tenable=True, similarity=0.90, kind="same-polarity-agreement")]
        assert recommend_floor(recs) is None

    def test_empty_list_is_none(self):
        assert recommend_floor([]) is None
