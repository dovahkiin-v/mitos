"""Tests for the Conflict sensor's candidate filter/rank stage (Phase 2b, §6.5 S4–S6).

``screen_candidates`` is the second and final candidate-pipeline stage: over 2a's raw
``list[Candidate]`` it drops the author's declared strong-relationship targets and the
proposal's own slug (S4), gates on the similarity floor (S5), then ranks and truncates to
``top_k`` (S6) — returning the judged batch (possibly ``[]``, a clean short-circuit, never
``Unavailable``). ``declared_strong_targets`` centralizes the CONF-D7 strong-set (weak
edges deliberately excluded). Both are pure, storeless functions over in-memory data.

Discipline (plan §9): no mocks — construct ``Candidate``/``ParsedEntry`` directly.
``screen_candidates`` never reads ``candidate.node``, so a minimal ``node`` is fine. Every
test passes an **explicit** ``floor`` (never asserts against ``CONFLICT_SIMILARITY_FLOOR``,
which 4b recalibrates — the anti-chase rule, CONF-D2).
"""

from typing import Any, Dict, List

from mitos.conflict import (
    Candidate,
    declared_strong_targets,
    screen_candidates,
)
from mitos.parser import ParsedEntry


# --------------------------------------------------------------------------- #
# Helpers — direct construction, no fixtures (pure/storeless functions)
# --------------------------------------------------------------------------- #

def _cand(slug: str, score: float) -> Candidate:
    """A minimal Candidate; ``node`` is unread by screen_candidates (plan §9)."""
    return Candidate(slug=slug, score=score, node={}, state="active")


def _entry(slug: str = "proposal", **relationships: List[str]) -> ParsedEntry:
    """A ParsedEntry with the given relationship lists set (fields default ``[]``)."""
    entry = ParsedEntry("decision", slug, 0, 0)
    for field, targets in relationships.items():
        setattr(entry, field, targets)
    return entry


def _slugs(candidates: List[Candidate]) -> List[str]:
    """The slugs of a candidate list, in order (for order-sensitive assertions)."""
    return [c.slug for c in candidates]


# --------------------------------------------------------------------------- #
# Test 1 — the §6.5 trace (the load-bearing pin)
# --------------------------------------------------------------------------- #

def test_the_6_5_trace_returns_rows_3_5_6_7_ranked_descending() -> None:
    """Feed the vision's 8-row §6.5 scenario → judged batch is exactly rows [3,5,6,7].

    Proposal ``cache-policy-v2`` declares ``Supersedes: cache-policy``; illustrative
    floor 0.55, top_k 5. Rows 1–2 drop at S4 (declared / own-slug), rows 8–9 drop at S5
    (below floor), rows 3/5/6/7 survive — ranked similarity-descending. Assert slugs AND
    order (the whole point is the deterministic, ordered batch).
    """
    candidates = [
        _cand("cache-policy", 0.91),                # row 1 — declared Supersedes: → drop
        _cand("cache-policy-v2", 0.90),             # row 2 — own slug → drop
        _cand("cache-ttl-fixed", 0.78),             # row 3 — judged (rank 1)
        _cand("cache-invalidation-manual", 0.71),   # row 5 — judged (rank 2)
        _cand("metrics-naming", 0.68),              # row 6 — judged (weak Cites: only)
        _cand("global-no-cache", 0.66),             # row 7 — judged (scope-blind recall)
        _cand("db-conn-pooling", 0.41),             # row 8 — below floor
        _cand("logging-format", 0.22),              # row 9 — below floor
    ]
    result = screen_candidates(
        candidates,
        declared_targets={"cache-policy"},
        own_slug="cache-policy-v2",
        floor=0.55,
        top_k=5,
    )
    assert _slugs(result) == [
        "cache-ttl-fixed",
        "cache-invalidation-manual",
        "metrics-naming",
        "global-no-cache",
    ]


# --------------------------------------------------------------------------- #
# Test 2 — S4 declared-target drop (even at high similarity)
# --------------------------------------------------------------------------- #

def test_declared_target_is_dropped_even_at_top_similarity() -> None:
    """A candidate whose slug is a declared strong target is dropped before the floor."""
    result = screen_candidates(
        [_cand("declared-x", 0.99), _cand("undeclared-y", 0.60)],
        declared_targets={"declared-x"},
        own_slug="proposal",
        floor=0.55,
        top_k=5,
    )
    assert _slugs(result) == ["undeclared-y"]


# --------------------------------------------------------------------------- #
# Test 3 — S4 own-slug guard (false-self-conflict, RF-1)
# --------------------------------------------------------------------------- #

def test_own_slug_candidate_is_dropped_no_self_conflict() -> None:
    """A candidate matching own_slug (the prior version) never self-conflicts."""
    result = screen_candidates(
        [_cand("cache-policy-v2", 0.95), _cand("other", 0.60)],
        declared_targets=set(),
        own_slug="cache-policy-v2",
        floor=0.55,
        top_k=5,
    )
    assert _slugs(result) == ["other"]


# --------------------------------------------------------------------------- #
# Test 4 — weak edges do NOT contribute to the drop set
# --------------------------------------------------------------------------- #

def test_declared_strong_targets_excludes_weak_edges() -> None:
    """Only the five strong fields count; cites/depends_on/derives_from/resolves don't."""
    entry = _entry(
        supersedes=["s"],
        cites=["x"],
        depends_on=["y"],
        derives_from=["z"],
        resolves=["w"],
    )
    assert declared_strong_targets(entry) == {"s"}


def test_candidate_declared_only_via_cites_survives_s4() -> None:
    """A neighbour declared solely via a weak Cites: edge still reaches judgment (row 6)."""
    entry = _entry(supersedes=["cache-policy"], cites=["metrics-naming"])
    result = screen_candidates(
        [_cand("metrics-naming", 0.68)],
        declared_targets=declared_strong_targets(entry),
        own_slug=entry.slug,
        floor=0.55,
        top_k=5,
    )
    assert _slugs(result) == ["metrics-naming"]


def test_declared_strong_targets_unions_all_five_strong_fields() -> None:
    """Every strong field contributes; within-field duplicates collapse via the set."""
    entry = _entry(
        supersedes=["a", "a"],   # within-field duplicate collapses
        amends=["b"],
        narrows=["c"],
        contradicts=["d"],
        corrects=["e"],
    )
    assert declared_strong_targets(entry) == {"a", "b", "c", "d", "e"}


def test_declared_strong_targets_empty_when_no_relationships() -> None:
    """The common case: an entry declaring nothing yields the empty set."""
    assert declared_strong_targets(_entry()) == set()


# --------------------------------------------------------------------------- #
# Test 5 — casefold discipline on both sides of every slug compare (P9, Lesson 22)
# --------------------------------------------------------------------------- #

def test_declared_target_drop_is_casefolded() -> None:
    """A mixed-case declared target drops a differently-cased candidate slug."""
    result = screen_candidates(
        [_cand("cache-policy", 0.80)],
        declared_targets={"Cache-Policy".casefold()},
        own_slug="proposal",
        floor=0.55,
        top_k=5,
    )
    assert result == []


def test_own_slug_drop_is_casefolded() -> None:
    """own_slug is folded before comparison — case never leaks a self-conflict through."""
    result = screen_candidates(
        [_cand("cache-policy-v2", 0.80)],
        declared_targets=set(),
        own_slug="CACHE-POLICY-V2",
        floor=0.55,
        top_k=5,
    )
    assert result == []


def test_declared_strong_targets_folds_at_the_boundary() -> None:
    """declared_strong_targets casefolds each declared slug (incl. non-ASCII, P9)."""
    entry = _entry(supersedes=["Cache-Policy"], amends=["IŠIMTIS"])
    # Lithuanian "IŠIMTIS".casefold() folds the Š — the fold, not .lower(), is what
    # the S4 compare relies on for a load-bearing non-ASCII slug.
    assert declared_strong_targets(entry) == {"cache-policy", "IŠIMTIS".casefold()}


# --------------------------------------------------------------------------- #
# Test 6 — S5 floor is inclusive (`>=`)
# --------------------------------------------------------------------------- #

def test_floor_is_inclusive_keeps_exact_boundary_drops_just_below() -> None:
    """score == floor is kept; a hair below is dropped (the ``>= floor`` boundary)."""
    result = screen_candidates(
        [_cand("at-floor", 0.70), _cand("below-floor", 0.699)],
        declared_targets=set(),
        own_slug="proposal",
        floor=0.70,
        top_k=5,
    )
    assert _slugs(result) == ["at-floor"]


# --------------------------------------------------------------------------- #
# Test 7 — S6 rank + truncate to top_k
# --------------------------------------------------------------------------- #

def test_rank_and_truncate_keeps_top_k_by_score_descending() -> None:
    """Six above-floor survivors, top_k=5 → the top 5 by score, lowest dropped, ordered."""
    candidates = [
        _cand("c0.60", 0.60),
        _cand("c0.90", 0.90),
        _cand("c0.70", 0.70),
        _cand("c0.95", 0.95),
        _cand("c0.80", 0.80),
        _cand("c0.65", 0.65),
    ]
    result = screen_candidates(
        candidates,
        declared_targets=set(),
        own_slug="proposal",
        floor=0.55,
        top_k=5,
    )
    assert _slugs(result) == ["c0.95", "c0.90", "c0.80", "c0.70", "c0.65"]
    assert "c0.60" not in _slugs(result)  # the lowest above-floor survivor is truncated


# --------------------------------------------------------------------------- #
# Test 8 — shadowing: S4 drop happens BEFORE S6 truncate (CONF-D7)
# --------------------------------------------------------------------------- #

def test_declared_drops_before_truncate_so_undeclared_are_not_shadowed() -> None:
    """2 declared/own at the top by similarity + 5 undeclared above floor, top_k=5.

    All 5 undeclared must be present. A truncate-first bug (S6 before S4) would keep the
    2 high-similarity declared/own rows, leaving only 3 undeclared in the window — the
    orthogonal-conflict recall the vision promises, silently narrowed.
    """
    candidates = [
        _cand("declared-hi", 0.99),   # declared strong target, top similarity
        _cand("own-hi", 0.98),        # own slug, second
        _cand("u1", 0.90),
        _cand("u2", 0.85),
        _cand("u3", 0.80),
        _cand("u4", 0.75),
        _cand("u5", 0.70),
    ]
    result = screen_candidates(
        candidates,
        declared_targets={"declared-hi"},
        own_slug="own-hi",
        floor=0.55,
        top_k=5,
    )
    assert _slugs(result) == ["u1", "u2", "u3", "u4", "u5"]


# --------------------------------------------------------------------------- #
# Test 9 — empty short-circuits (clean, never Unavailable)
# --------------------------------------------------------------------------- #

def test_empty_input_returns_empty_list() -> None:
    """No candidates → clean empty batch (3b makes no LLM call)."""
    assert screen_candidates(
        [], declared_targets=set(), own_slug="proposal", floor=0.55, top_k=5
    ) == []


def test_all_below_floor_returns_empty_list() -> None:
    """Every candidate below the floor → clean empty batch (a novel decision)."""
    result = screen_candidates(
        [_cand("a", 0.40), _cand("b", 0.30)],
        declared_targets=set(),
        own_slug="proposal",
        floor=0.55,
        top_k=5,
    )
    assert result == []


def test_all_declared_or_self_returns_empty_list() -> None:
    """Every candidate declared/self → clean empty batch (all already reasoned about)."""
    result = screen_candidates(
        [_cand("declared", 0.90), _cand("proposal", 0.88)],
        declared_targets={"declared"},
        own_slug="proposal",
        floor=0.55,
        top_k=5,
    )
    assert result == []


# --------------------------------------------------------------------------- #
# Test 10 — floor/top_k default to the module constants (D2), injectable
# --------------------------------------------------------------------------- #

def test_defaults_read_the_module_constants_without_pinning_their_values() -> None:
    """Omitting floor/top_k reads the module constants at runtime (not asserted by value).

    Two candidates far apart in similarity — one comfortably above any sane floor, one
    comfortably below — so the split is stable regardless of the PROVISIONAL floor's exact
    value (never pinned here; 4b owns it). Proves the defaults wire through.
    """
    result = screen_candidates(
        [_cand("clear-keep", 0.99), _cand("clear-drop", 0.01)],
        declared_targets=set(),
        own_slug="proposal",
    )
    assert _slugs(result) == ["clear-keep"]
