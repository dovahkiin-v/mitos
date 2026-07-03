"""Tests for the Conflict sensor's pipeline facade (Phase 3b — ``run_conflict_check``).

``run_conflict_check`` composes the five shipped stages (2a gather → 2b screen → 3a render →
injected executor → 3a parse → CONF-D4 gate) into deliverable 1: the reusable core. It
returns a ``ConflictCheckResult`` (clean-empty / judged-none / judged-some) or a typed
``Unavailable`` (degraded), writing nothing.

Discipline (scout brief / plan §9): deterministic, keyless, **no SDK** — inject hand-rolled
fake ``embed_provider``/``vector_store``/``store`` (the 2a idiom) + a plain fake ``judge``
function (a closure returning a canned ``JudgmentExecution`` or ``Unavailable``). Every test
passes an **explicit** ``floor``/``top_k``/``surface_threshold`` so behaviour keys on the
injected values, never the PROVISIONAL ``CONFLICT_SIMILARITY_FLOOR`` (the anti-chase rule).
Facade fake ``node`` dicts are RICHER than 2b's ``{}`` — the facade hits
``node["core_axiom"]`` (``judge_input_from_node``) and ``slug``/``scope``/``rejected_paths``
(``candidate_payload`` → ``letter_payload``).
"""

import json
from typing import Any, Dict, List, Optional

import pytest

from mitos.conflict import (
    ConflictCheckResult,
    ConflictUnavailableReason,
    JudgmentExecution,
    Unavailable,
    compute_node_id,
    judge_input_from_entry,
    judge_input_from_node,
    run_conflict_check,
)
from mitos.errors import EmbeddingError
from mitos.parser import ParsedEntry


# --------------------------------------------------------------------------- #
# Fixtures + hand-rolled fakes (the 2a idiom — synchronous, no unittest.mock)
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key, no reachable service — the injected fakes are the only substrate."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


class _FakeEmbed:
    """Returns a fixed document-space vector, or raises the configured error."""

    def __init__(self, raises: Optional[BaseException] = None) -> None:
        self._raises = raises

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        if self._raises is not None:
            raise self._raises
        return [0.1, 0.2, 0.3]


class _FakeVector:
    """Returns canned ``(slug, score)`` matches for the over-fetch query."""

    def __init__(self, matches: List[Dict[str, Any]]) -> None:
        self._matches = matches

    def query(self, vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        return list(self._matches)


class _FakeStore:
    """Resolves slugs to hydrated nodes from a dict; every resolved node is ``active``."""

    def __init__(self, nodes: Dict[str, Dict[str, Any]]) -> None:
        self._nodes = nodes

    def get_node_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        return self._nodes.get(slug)

    def get_node_state(self, node_id: str) -> str:
        return "active"


class _RecordingJudge:
    """A fake ``judge`` returning a canned value; records whether/how often it was called."""

    def __init__(self, ret: Any) -> None:
        self._ret = ret
        self.called = False
        self.calls = 0
        self.last_prompt: Any = None

    def __call__(self, prompt: Any) -> Any:
        self.called = True
        self.calls += 1
        self.last_prompt = prompt
        return self._ret


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def _node(slug: str, *, axiom: Optional[str] = None, scope: Optional[List[str]] = None,
          rejected: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    """A hydrated store node carrying the keys the facade reads (+ optional modifier stamps)."""
    node = {
        "id": f"id-{slug}",
        "slug": slug,
        "core_axiom": axiom if axiom is not None else f"axiom for {slug}",
        "scope": scope if scope is not None else [],
        "rejected_paths": rejected if rejected is not None else f"rejected for {slug}",
    }
    node.update(extra)
    return node


def _match(slug: str, score: float) -> Dict[str, Any]:
    return {"slug": slug, "score": score}


def _entry(
    slug: str = "proposal",
    *,
    axiom: str = "The proposal axiom.",
    mechanisms: Optional[List[str]] = None,
    scope: Optional[List[str]] = None,
    rejected: str = "Rejected the obvious alternative.",
    **relationships: List[str],
) -> ParsedEntry:
    """A proposal ParsedEntry with the canonical-core + M5 fields the facade reads."""
    entry = ParsedEntry("decision", slug, 0, 0)
    entry.axiom = axiom
    entry.mechanisms = mechanisms if mechanisms is not None else []
    entry.scope = scope if scope is not None else []
    entry.rejected_paths = rejected
    for field, targets in relationships.items():
        setattr(entry, field, targets)
    return entry


def _execution(
    verdicts: List[tuple],
    *,
    batch_id: str = "batch-fixed-id",
    elapsed_ms: int = 12,
    token_input: int = 100,
    token_output: int = 40,
) -> JudgmentExecution:
    """Builds a JudgmentExecution whose ``raw_text`` is the judge JSON for ``verdicts``.

    ``verdicts``: list of ``(slug, tenable_together, confidence, rationale)``. Order is free
    (3a's parse realigns by slug); the set/count must match the screened batch.
    """
    raw = json.dumps(
        [
            {
                "slug": slug,
                "rationale": rationale,
                "tenable_together": tenable,
                "confidence": confidence,
            }
            for (slug, tenable, confidence, rationale) in verdicts
        ]
    )
    return JudgmentExecution(
        raw_text=raw,
        batch_id=batch_id,
        model_alias="SONNET",
        token_input=token_input,
        token_output=token_output,
        token_cache_read=0,
        token_cache_creation=0,
        elapsed_ms=elapsed_ms,
    )


def _run(
    judge: Any,
    *,
    entry: Optional[ParsedEntry] = None,
    matches: Optional[List[Dict[str, Any]]] = None,
    nodes: Optional[Dict[str, Dict[str, Any]]] = None,
    embed: Optional[_FakeEmbed] = None,
    floor: float = 0.5,
    top_k: int = 5,
    surface_threshold: float = 0.85,
) -> Any:
    """Drives the facade with hand-rolled fakes + explicit dials."""
    return run_conflict_check(
        entry if entry is not None else _entry(),
        embed_provider=embed if embed is not None else _FakeEmbed(),
        vector_store=_FakeVector(matches if matches is not None else []),
        store=_FakeStore(nodes if nodes is not None else {}),
        judge=judge,
        floor=floor,
        top_k=top_k,
        surface_threshold=surface_threshold,
    )


# --------------------------------------------------------------------------- #
# 1. Happy path — one pair surfaces; all pairs carried for telemetry
# --------------------------------------------------------------------------- #

def test_happy_path_surfaces_only_the_gated_pair_and_carries_all_pairs() -> None:
    """A not-tenable-at-high-confidence pair surfaces; every judged pair is in judged_pairs."""
    matches = [_match("cand-alpha", 0.90), _match("cand-beta", 0.80), _match("cand-gamma", 0.70)]
    nodes = {
        # cand-alpha carries a modifier stamp — assert it rides onto the finding payload.
        "cand-alpha": _node("cand-alpha", amended_by="some-amender"),
        "cand-beta": _node("cand-beta"),
        "cand-gamma": _node("cand-gamma"),
    }
    judge = _RecordingJudge(
        _execution(
            [
                ("cand-alpha", False, 0.90, "alpha reverses the proposal"),
                ("cand-beta", True, 0.95, "beta merely elaborates"),
                ("cand-gamma", False, 0.60, "gamma weakly tenses"),
            ]
        )
    )

    result = _run(judge, matches=matches, nodes=nodes)

    assert isinstance(result, ConflictCheckResult)
    # Exactly one surfaced finding — cand-alpha (not-tenable @ 0.90 ≥ 0.85).
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.slug == "cand-alpha"
    assert finding.confidence == 0.90
    assert "reverses" in finding.rationale
    assert finding.payload["slug"] == "cand-alpha"
    assert finding.payload["amended_by"] == "some-amender"  # 2b stamp rides along.
    # All three pairs carried, with correct surfaced flags.
    flags = {p.candidate.slug: p.surfaced for p in result.judged_pairs}
    assert flags == {"cand-alpha": True, "cand-beta": False, "cand-gamma": False}
    assert result.execution is judge._ret  # the batch metrics threaded through.


# --------------------------------------------------------------------------- #
# 2. Gate boundaries — the CONF-D4 surface predicate
# --------------------------------------------------------------------------- #

def test_gate_boundaries_of_the_conf_d4_predicate() -> None:
    """0.85 surfaces; 0.84999 does not; tenable never surfaces; a low not-tenable is judged-not-surfaced."""
    matches = [
        _match("edge-eq", 0.90),      # not-tenable @ 0.85 exactly → surfaces.
        _match("edge-below", 0.89),   # not-tenable @ 0.84999 → does not surface.
        _match("tenable-high", 0.88), # tenable @ 0.99 → never surfaces.
        _match("low-nt", 0.87),       # not-tenable @ 0.20 → does not surface, but IS judged.
    ]
    nodes = {m["slug"]: _node(m["slug"]) for m in matches}
    judge = _RecordingJudge(
        _execution(
            [
                ("edge-eq", False, 0.85, "exactly at threshold"),
                ("edge-below", False, 0.84999, "a hair below"),
                ("tenable-high", True, 0.99, "compatible despite closeness"),
                ("low-nt", False, 0.20, "contradiction but unsure"),
            ]
        )
    )

    result = _run(judge, matches=matches, nodes=nodes, surface_threshold=0.85)

    surfaced = {f.slug for f in result.findings}
    assert surfaced == {"edge-eq"}
    # All four are judged (telemetry), regardless of the gate.
    assert {p.candidate.slug for p in result.judged_pairs} == {
        "edge-eq", "edge-below", "tenable-high", "low-nt"
    }
    flags = {p.candidate.slug: p.surfaced for p in result.judged_pairs}
    assert flags == {
        "edge-eq": True, "edge-below": False, "tenable-high": False, "low-nt": False
    }


# --------------------------------------------------------------------------- #
# 3. Clean-empty — nothing screens through; the judge is NOT called
# --------------------------------------------------------------------------- #

def test_clean_empty_returns_result_without_calling_the_judge() -> None:
    """All matches fall below the floor → clean-empty result, no LLM call (DoD-2)."""
    matches = [_match("far-1", 0.30), _match("far-2", 0.20)]
    nodes = {"far-1": _node("far-1"), "far-2": _node("far-2")}
    judge = _RecordingJudge(_execution([]))

    result = _run(judge, matches=matches, nodes=nodes, floor=0.55)

    assert isinstance(result, ConflictCheckResult)
    assert result.findings == []
    assert result.judged_pairs == []
    assert result.execution is None
    assert judge.called is False  # no judge on a below-floor entry (CONF-D2).


def test_clean_empty_on_no_matches_at_all() -> None:
    """No neighbours at all → the same clean-empty result, judge untouched."""
    judge = _RecordingJudge(_execution([]))
    result = _run(judge, matches=[], nodes={})
    assert isinstance(result, ConflictCheckResult)
    assert result.judged_pairs == [] and result.execution is None
    assert judge.called is False


# --------------------------------------------------------------------------- #
# 4. 2a degradation propagates — judge never called
# --------------------------------------------------------------------------- #

def test_gather_unavailable_propagates_and_judge_not_called() -> None:
    """An EMBEDDING degradation from 2a returns verbatim; the judge is never reached."""
    judge = _RecordingJudge(_execution([]))
    result = _run(
        judge,
        embed=_FakeEmbed(raises=EmbeddingError("gemini down")),
        matches=[_match("x", 0.9)],
        nodes={"x": _node("x")},
    )
    assert isinstance(result, Unavailable)
    assert result.reason is ConflictUnavailableReason.EMBEDDING
    assert judge.called is False


# --------------------------------------------------------------------------- #
# 5. 3b degradation propagates
# --------------------------------------------------------------------------- #

def test_executor_unavailable_propagates() -> None:
    """A JUDGMENT_TIMEOUT from the injected executor is returned verbatim."""
    unavailable = Unavailable(
        reason=ConflictUnavailableReason.JUDGMENT_TIMEOUT, detail="timed out after 15s"
    )
    judge = _RecordingJudge(unavailable)
    matches = [_match("cand-a", 0.9)]
    result = _run(judge, matches=matches, nodes={"cand-a": _node("cand-a")})
    assert result is unavailable
    assert judge.called is True  # it WAS called (a real batch existed).


# --------------------------------------------------------------------------- #
# 6. Malformed batch → Unavailable(JUDGMENT)
# --------------------------------------------------------------------------- #

def test_malformed_batch_degrades_to_judgment_unavailable() -> None:
    """A non-JSON raw_text makes 3a's parse return Unavailable(JUDGMENT); facade returns it."""
    bad = JudgmentExecution(
        raw_text="not json at all",
        batch_id="b",
        model_alias="SONNET",
        token_input=10,
        token_output=5,
        token_cache_read=0,
        token_cache_creation=0,
        elapsed_ms=3,
    )
    judge = _RecordingJudge(bad)
    matches = [_match("cand-a", 0.9)]
    result = _run(judge, matches=matches, nodes={"cand-a": _node("cand-a")})
    assert isinstance(result, Unavailable)
    assert result.reason is ConflictUnavailableReason.JUDGMENT  # NOT JUDGMENT_TIMEOUT.


# --------------------------------------------------------------------------- #
# 7. proposed_hash_if_any join (DoD-6 seed)
# --------------------------------------------------------------------------- #

def test_proposed_hash_mirrors_the_commit_path_compute_node_id() -> None:
    """The join hash equals a direct compute_node_id over {kind, axiom, mechanisms} (D5)."""
    entry = _entry(axiom="We standardize on SQLite WAL.", mechanisms=["sqlite", "wal"])
    judge = _RecordingJudge(_execution([]))
    result = _run(judge, entry=entry, matches=[], nodes={})

    expected = compute_node_id(
        kind="decision", axiom=entry.axiom, mechanism_refs=entry.mechanisms
    )
    assert isinstance(result, ConflictCheckResult)
    assert result.proposed_hash_if_any == expected


# --------------------------------------------------------------------------- #
# 8. Fed context is carried verbatim (what 5b persists)
# --------------------------------------------------------------------------- #

def test_fed_context_is_carried_verbatim() -> None:
    """proposal_input and each pair.candidate_input equal the exact JudgeInputs fed."""
    entry = _entry(axiom="Proposal axiom X.", scope=["cache"], rejected="No pgvector.")
    matches = [_match("cand-a", 0.9), _match("cand-b", 0.8)]
    nodes = {
        "cand-a": _node("cand-a", axiom="A axiom", scope=["db"], rejected="A rejects"),
        "cand-b": _node("cand-b", axiom="B axiom", scope=[], rejected=""),
    }
    judge = _RecordingJudge(
        _execution(
            [("cand-a", True, 0.9, "ok"), ("cand-b", True, 0.9, "ok")]
        )
    )

    result = _run(judge, entry=entry, matches=matches, nodes=nodes)

    assert result.proposal_input == judge_input_from_entry(entry)
    by_slug = {p.candidate.slug: p.candidate_input for p in result.judged_pairs}
    assert by_slug["cand-a"] == judge_input_from_node(nodes["cand-a"])
    assert by_slug["cand-b"] == judge_input_from_node(nodes["cand-b"])


# --------------------------------------------------------------------------- #
# 9. Execution (batch metrics) threaded onto the result
# --------------------------------------------------------------------------- #

def test_execution_metrics_ride_onto_the_result() -> None:
    """The batch_id / tokens / elapsed from the fake judge are threaded onto result.execution."""
    execution = _execution(
        [("cand-a", True, 0.9, "ok")],
        batch_id="the-batch-42",
        elapsed_ms=77,
        token_input=321,
        token_output=88,
    )
    judge = _RecordingJudge(execution)
    result = _run(judge, matches=[_match("cand-a", 0.9)], nodes={"cand-a": _node("cand-a")})

    assert isinstance(result, ConflictCheckResult)
    assert result.execution is execution
    assert result.execution.batch_id == "the-batch-42"
    assert result.execution.elapsed_ms == 77
    assert result.execution.token_input == 321


# --------------------------------------------------------------------------- #
# 10. floor / top_k / surface_threshold are injectable
# --------------------------------------------------------------------------- #

def test_top_k_truncates_the_judged_batch() -> None:
    """top_k=1 sends only the highest-scoring candidate to the judge."""
    matches = [_match("hi", 0.90), _match("mid", 0.80), _match("lo", 0.70)]
    nodes = {m["slug"]: _node(m["slug"]) for m in matches}
    # Only the top candidate ("hi") survives → the judge JSON must cover exactly it.
    judge = _RecordingJudge(_execution([("hi", True, 0.9, "ok")]))

    result = _run(judge, matches=matches, nodes=nodes, top_k=1)

    assert isinstance(result, ConflictCheckResult)
    assert [p.candidate.slug for p in result.judged_pairs] == ["hi"]


def test_floor_gates_out_below_threshold_candidates() -> None:
    """An injected floor of 0.85 drops everything under it before judgment."""
    matches = [_match("hi", 0.90), _match("mid", 0.80), _match("lo", 0.70)]
    nodes = {m["slug"]: _node(m["slug"]) for m in matches}
    judge = _RecordingJudge(_execution([("hi", True, 0.9, "ok")]))

    result = _run(judge, matches=matches, nodes=nodes, floor=0.85)

    assert [p.candidate.slug for p in result.judged_pairs] == ["hi"]


def test_surface_threshold_is_injectable() -> None:
    """A low surface_threshold (0.5) surfaces a not-tenable finding at confidence 0.6."""
    matches = [_match("cand-a", 0.9)]
    nodes = {"cand-a": _node("cand-a")}
    judge = _RecordingJudge(_execution([("cand-a", False, 0.60, "contradicts")]))

    result = _run(judge, matches=matches, nodes=nodes, surface_threshold=0.5)

    assert isinstance(result, ConflictCheckResult)
    assert [f.slug for f in result.findings] == ["cand-a"]
