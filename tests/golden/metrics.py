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

from typing import Dict, List, Sequence, Set


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
