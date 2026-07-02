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

sys.path.insert(0, os.path.dirname(__file__))
from metrics import (  # noqa: E402
    _dedupe_preserve_order,
    evaluate_fixture,
    hard_negative_fp_rate,
    mrr,
    precision_at_k,
    recall_at_k,
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
