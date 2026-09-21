"""The standing-finding carry: a finding leaves the report only by being resolved.

``mitos check`` partitions findings into new (gating, exit 1) and known (standing,
non-gating) — ``check-findings-gate-on-novelty-not-standing``, whose rejected path (2)
refuses hiding known findings outright: *"an unresolved contradiction that stops being
reported is a silent audit hole"*. It refused the outcome but its mechanism delivered
it anyway, because "known" was derived only for pairs THIS RUN'S SWEEP rediscovered:
``plan_corpus_check`` built its pair set from the sweep and the reuse partition then
iterated only that set.

Measured on the 2026-09-18 corpus audit. Between two runs the corpus grew 412 → 444
nodes, one pair was pushed out of every node's ``CONFLICT_TOP_K = 5`` window by nearer
neighbours, its stored verdict was never looked up, and its standing finding left the
report with nothing resolved and nothing said. ``findings_known`` was never "open
findings" — it was "open findings that happened to be re-screened this run".

``standing-findings-carry-on-the-reuse-index-not-the-sweep`` amends the mechanism and
keeps the axiom: standing findings are carried off the REUSE INDEX, screened on the
same two facts the sweep screens on. This module pins the three screens and, most
importantly, the property that makes the carry safe rather than nagging —
**a pair with a declared strong edge is never resurrected** (test_edge_declared_*).
Without that screen the fix would re-report exactly the findings an author resolved,
which is the crying-wolf sensor the parent vision's trust law exists against.

Discipline (PATTERNS / plan §8): no mocks — real ``ReuseIndex``/``StoredVerdict``/
snapshot construction, real hashes from a real store for the end-to-end row. Run under
``./venv/bin/python -m pytest`` (bare ``python`` lacks the deps).
"""

import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import pytest

from _conflict_helpers import _keyed_substrate

from mitos.check import (
    CorpusSnapshot,
    DepartedFinding,
    build_strong_edge_index,
    carry_standing_findings,
    plan_corpus_check,
)
from mitos.conflict import CONFLICT_PROMPT_VERSION, CONFLICT_SURFACE_THRESHOLD
from mitos.parser import ParsedEntry
from mitos.store import GraphStore
from mitos.telemetry import ReuseIndex, StoredVerdict, TelemetryStore


PRODUCTION_ALIAS = "SONNET"


# --------------------------------------------------------------------------- #
# Helpers — direct construction (the function under test is pure and storeless)
# --------------------------------------------------------------------------- #


def _node(node_id: str, slug: str) -> Dict[str, Any]:
    """A hydrated node dict — the two keys the carry reads plus a real-shaped axiom."""
    return {"id": node_id, "slug": slug, "axiom": f"Axiom for {slug}."}


def _verdict(
    *,
    tenable: bool = False,
    confidence: float = 0.92,
    rationale: str = "Stored rationale, carried verbatim.",
    batch_id: str = "prior-batch",
    created_at: str = "2026-09-07T12:00:00+00:00",
) -> StoredVerdict:
    """A stored prior verdict; finding-grade by default (not tenable, above threshold)."""
    return StoredVerdict(
        tenable=tenable,
        confidence=confidence,
        rationale=rationale,
        batch_id=batch_id,
        created_at=created_at,
    )


def _index(*entries: Tuple[str, str, StoredVerdict]) -> ReuseIndex:
    """Builds a real ``ReuseIndex`` on the same sorted-pair key ``lookup`` uses."""
    return ReuseIndex({tuple(sorted((a, b))): v for a, b, v in entries})


def _snapshot(
    nodes: List[Dict[str, Any]], edges: Optional[List[Dict[str, str]]] = None
) -> CorpusSnapshot:
    """A snapshot over the given live nodes and (optionally) declared edges."""
    return CorpusSnapshot(
        nodes=tuple(nodes), edge_index=build_strong_edge_index(edges or [])
    )


# --------------------------------------------------------------------------- #
# The defect this exists against
# --------------------------------------------------------------------------- #


def test_finding_the_sweep_never_rediscovered_is_still_reported() -> None:
    """THE regression: a standing finding outside every top-k window still stands.

    The 2026-09-18 shape exactly — the pair is live, undeclared and finding-grade,
    and this run's sweep produced nothing at all for it.
    """
    a, b = _node("hash-a", "alpha"), _node("hash-b", "beta")
    carried, departed = carry_standing_findings(
        reuse_index=_index(("hash-a", "hash-b", _verdict())),
        snapshot=_snapshot([a, b]),
        swept_pairs=[],
    )

    assert len(carried) == 1, "the finding vanished — this is the silent audit hole"
    assert departed == []
    assert {carried[0].proposal_node["slug"], carried[0].partner_node["slug"]} == {
        "alpha",
        "beta",
    }


def test_carried_finding_keeps_the_stored_verdict_verbatim() -> None:
    """M8: the carry re-reports a prior, it never re-renders one."""
    verdict = _verdict(
        confidence=0.88,
        rationale="The two cannot both supply the bound.",
        batch_id="batch-from-the-first-audit",
        created_at="2026-09-07T13:45:00+00:00",
    )
    carried, _ = carry_standing_findings(
        reuse_index=_index(("hash-a", "hash-b", verdict)),
        snapshot=_snapshot([_node("hash-a", "alpha"), _node("hash-b", "beta")]),
        swept_pairs=[],
    )

    assert carried[0].verdict == verdict
    assert carried[0].verdict.created_at == "2026-09-07T13:45:00+00:00", (
        "a carried finding's first-reported stamp is the PRIOR's, not this run's"
    )


# --------------------------------------------------------------------------- #
# The screen that keeps the carry honest rather than nagging
# --------------------------------------------------------------------------- #


def test_edge_declared_departs_and_is_never_carried() -> None:
    """The anti-resurrection property: a resolved pair must not come back.

    Declaring ``Supersedes:``/``Amends:``/``Narrows:``/``Contradicts:`` IS the
    resolution path. A carry that skipped this screen would re-report every finding
    an author had already settled — strictly worse than the bug it fixes.
    """
    carried, departed = carry_standing_findings(
        reuse_index=_index(("hash-a", "hash-b", _verdict())),
        snapshot=_snapshot(
            [_node("hash-a", "alpha"), _node("hash-b", "beta")],
            edges=[
                {
                    "source_id": "hash-a",
                    "target_id": "hash-b",
                    "edge_type": "contradicts",
                }
            ],
        ),
        swept_pairs=[],
    )

    assert carried == []
    assert [d.reason for d in departed] == ["edge-declared"]


def test_edge_declared_screen_is_orientation_blind() -> None:
    """An edge authored the other way round resolves the pair just the same."""
    carried, departed = carry_standing_findings(
        reuse_index=_index(("hash-a", "hash-b", _verdict())),
        snapshot=_snapshot(
            [_node("hash-a", "alpha"), _node("hash-b", "beta")],
            edges=[
                {"source_id": "hash-b", "target_id": "hash-a", "edge_type": "amends"}
            ],
        ),
        swept_pairs=[],
    )

    assert carried == []
    assert [d.reason for d in departed] == ["edge-declared"]


def test_a_weak_edge_does_not_resolve_a_pair() -> None:
    """Only the five strong relationship types resolve — ``cites`` is not one."""
    carried, departed = carry_standing_findings(
        reuse_index=_index(("hash-a", "hash-b", _verdict())),
        snapshot=_snapshot(
            [_node("hash-a", "alpha"), _node("hash-b", "beta")],
            edges=[
                {"source_id": "hash-a", "target_id": "hash-b", "edge_type": "cites"}
            ],
        ),
        swept_pairs=[],
    )

    assert len(carried) == 1
    assert departed == []


def test_dead_side_departs_as_no_longer_live() -> None:
    """A side that is no longer an active decision at its judged hash ends the pair."""
    carried, departed = carry_standing_findings(
        reuse_index=_index(("hash-a", "hash-gone", _verdict())),
        snapshot=_snapshot([_node("hash-a", "alpha")]),
        swept_pairs=[],
    )

    assert carried == []
    assert [d.reason for d in departed] == ["side-no-longer-live"]
    assert {departed[0].proposal_hash, departed[0].partner_hash} == {
        "hash-a",
        "hash-gone",
    }


# --------------------------------------------------------------------------- #
# Not a finding, and not a departure either
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "verdict,why",
    [
        (_verdict(tenable=True, confidence=0.99), "a tenable prior is not a finding"),
        (
            _verdict(tenable=False, confidence=CONFLICT_SURFACE_THRESHOLD - 0.01),
            "a below-threshold prior never surfaced",
        ),
    ],
)
def test_non_finding_priors_are_neither_carried_nor_departed(
    verdict: StoredVerdict, why: str
) -> None:
    """A prior that never stood cannot stop standing — it is silently skipped.

    Counting these as departures would report a "resolution" for a pair that was
    never a finding, inflating the delta with noise on every run forever.
    """
    carried, departed = carry_standing_findings(
        reuse_index=_index(("hash-a", "hash-b", verdict)),
        snapshot=_snapshot([_node("hash-a", "alpha"), _node("hash-b", "beta")]),
        swept_pairs=[],
    )

    assert carried == [], why
    assert departed == [], why


def test_threshold_is_inclusive_at_the_surface_boundary() -> None:
    """The gate is ``>=`` (CONF-D4) — the carry re-derives it, never its own formula."""
    carried, _ = carry_standing_findings(
        reuse_index=_index(
            ("hash-a", "hash-b", _verdict(confidence=CONFLICT_SURFACE_THRESHOLD))
        ),
        snapshot=_snapshot([_node("hash-a", "alpha"), _node("hash-b", "beta")]),
        swept_pairs=[],
    )

    assert len(carried) == 1


# --------------------------------------------------------------------------- #
# No double-reporting: the carry covers what the sweep MISSED
# --------------------------------------------------------------------------- #


class _Pair:
    """The two attributes the carry reads off a swept ``CorpusPair``."""

    def __init__(self, proposal_hash: str, partner_hash: str) -> None:
        self.proposal_hash = proposal_hash
        self.partner_hash = partner_hash


def test_a_pair_the_sweep_found_is_not_carried_as_well() -> None:
    """It already reports through the normal reuse path; carrying would duplicate it."""
    carried, departed = carry_standing_findings(
        reuse_index=_index(("hash-a", "hash-b", _verdict())),
        snapshot=_snapshot([_node("hash-a", "alpha"), _node("hash-b", "beta")]),
        swept_pairs=[_Pair("hash-a", "hash-b")],
    )

    assert carried == []
    assert departed == []


def test_swept_exclusion_is_orientation_blind() -> None:
    """The sweep's discovery direction is an accident (CHK-D2) — it must not matter."""
    carried, _ = carry_standing_findings(
        reuse_index=_index(("hash-a", "hash-b", _verdict())),
        snapshot=_snapshot([_node("hash-a", "alpha"), _node("hash-b", "beta")]),
        swept_pairs=[_Pair("hash-b", "hash-a")],
    )

    assert carried == [], "the pair was swept — the reversed key must still match"


# --------------------------------------------------------------------------- #
# Degradation and determinism
# --------------------------------------------------------------------------- #


def test_unavailable_index_carries_nothing_and_claims_no_departures() -> None:
    """A run that cannot read its history must not assert what is still standing."""
    carried, departed = carry_standing_findings(
        reuse_index=None,
        snapshot=_snapshot([_node("hash-a", "alpha"), _node("hash-b", "beta")]),
        swept_pairs=[],
    )

    assert carried == []
    assert departed == [], (
        "no history means no knowledge of departures — silence, never a claim"
    )


def test_output_order_is_the_pair_key_not_map_iteration() -> None:
    """A run's report order is a pure function of its corpus (mirrors dedup ordering)."""
    nodes = [_node(f"hash-{c}", c) for c in "abcd"]
    carried, _ = carry_standing_findings(
        reuse_index=_index(
            ("hash-c", "hash-d", _verdict()),
            ("hash-a", "hash-b", _verdict()),
        ),
        snapshot=_snapshot(nodes),
        swept_pairs=[],
    )

    assert [(c.proposal_hash, c.partner_hash) for c in carried] == [
        ("hash-a", "hash-b"),
        ("hash-c", "hash-d"),
    ]


# --------------------------------------------------------------------------- #
# End to end, through the real planner
# --------------------------------------------------------------------------- #


@pytest.fixture
def temp_store() -> GraphStore:
    """A real graph store on a temp file (real content hashes, for free)."""
    directory = tempfile.mkdtemp()
    yield GraphStore(os.path.join(directory, "graph.sqlite"))


@pytest.fixture
def temp_telemetry() -> TelemetryStore:
    """A real telemetry store on a temp file."""
    directory = tempfile.mkdtemp()
    yield TelemetryStore(os.path.join(directory, "telemetry.sqlite"))


def _commit(store: GraphStore, slug: str, axiom: str) -> str:
    """Commits a decision and returns its content-hash node id."""
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = axiom
    entry.rejected_paths = "An alternative."
    return store.commit_parsed_entry(entry).node_id


def test_planner_carries_a_standing_finding_the_sweep_returned_nothing_for(
    temp_store: GraphStore, temp_telemetry: TelemetryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: empty neighbourhoods, a seeded prior, and the finding still stands.

    The planner's own pair set is empty here — precisely the state that used to make
    a standing finding disappear — so everything on ``plan.carried`` got there by the
    carry and nothing by the sweep.
    """
    monkeypatch.setenv("MITOS_NO_LIVE_TESTS", "1")
    a_axiom, b_axiom = "Planner axiom alpha.", "Planner axiom beta."
    a_id = _commit(temp_store, "planner-alpha", a_axiom)
    b_id = _commit(temp_store, "planner-beta", b_axiom)

    from mitos.telemetry import ConflictCheckRow, JudgmentBatch
    from mitos import __version__

    row = ConflictCheckRow(
        batch_id="carry-seed",
        sync_run_id="seed-run",
        surface="sync",
        judged_axiom=a_axiom,
        proposal_rejected_paths=None,
        proposal_scope=None,
        proposed_hash_if_any=min(a_id, b_id),
        candidate_slug="planner-beta",
        candidate_hash=max(a_id, b_id),
        candidate_rejected_paths="Seeded alternative.",
        candidate_scope=None,
        tenable=False,
        confidence=0.93,
        surfaced=True,
        candidate_source="embedding_topk",
        model_alias=PRODUCTION_ALIAS,
        prompt_version=CONFLICT_PROMPT_VERSION,
        mitos_version=__version__,
        rationale="Seeded standing finding.",
    )
    temp_telemetry.record_judged_batch(
        JudgmentBatch(
            batch_id="carry-seed",
            model_id=None,
            token_input=1,
            token_output=1,
            token_cache_read=0,
            token_cache_creation=0,
            elapsed_ms=1,
        ),
        [row],
        "2026-09-07T12:00:00+00:00",
    )

    # Every node present, nobody's neighbour — the sweep finds no pairs at all.
    embed, vector = _keyed_substrate({a_axiom: [], b_axiom: []})
    plan = plan_corpus_check(
        store=temp_store,
        embed_provider=embed,
        vector_store=vector,
        telemetry=temp_telemetry,
        model_alias=PRODUCTION_ALIAS,
    )

    assert plan.pairs == (), "precondition: the sweep discovered nothing"
    assert len(plan.carried) == 1, "the seeded standing finding went silent"
    assert plan.departed == ()
    assert {
        plan.carried[0].proposal_node["slug"],
        plan.carried[0].partner_node["slug"],
    } == {"planner-alpha", "planner-beta"}


# --------------------------------------------------------------------------- #
# The surface — the departure delta is a bounded line, never a growing wall
# --------------------------------------------------------------------------- #


def _result(departed: Tuple[DepartedFinding, ...]) -> Any:
    """A minimal healthy ``CheckRunResult`` carrying only the departures under test."""
    from mitos.check import CheckRunResult, StaleProbe

    probe = StaleProbe(transient=(), excluded=())
    return CheckRunResult(
        run_id="render-run",
        started_at="2026-09-18T12:00:00+00:00",
        ended_at="2026-09-18T12:01:00+00:00",
        nodes_total=2,
        nodes_swept=2,
        swept_node_ids=(),
        sweep_degraded=None,
        findings=(),
        pairs_judged_fresh=0,
        pairs_reused=0,
        batches_planned=0,
        batches_executed=0,
        batches_failed=0,
        batches_skipped=0,
        judgment_failures=(),
        judgment_abort=None,
        reuse_unavailable=None,
        telemetry_write_failures=(),
        start_probe=probe,
        end_probe=probe,
        departed=departed,
    )


def _render(result: Any, capsys: pytest.CaptureFixture) -> str:
    """Renders the human report and returns stdout."""
    from mitos import cli

    cli._print_check_report(
        result,
        exclusions=[],
        denominator=2,
        scope=None,
        row_written=True,
        transient_count=0,
    )
    return capsys.readouterr().out


def test_departure_line_counts_by_reason_and_names_no_pairs(
    capsys: pytest.CaptureFixture,
) -> None:
    """One bounded line: counts and reasons, never an enumeration.

    A resolution stays true forever, so a narrated section would re-print the same
    settled pairs on every run until the end of time — the nagging surface the
    parent vision's trust law forbids.
    """
    out = _render(
        _result(
            (
                DepartedFinding("h1", "h2", "edge-declared"),
                DepartedFinding("h3", "h4", "side-no-longer-live"),
                DepartedFinding("h5", "h6", "edge-declared"),
            )
        ),
        capsys,
    )

    assert (
        "3 previously-reported findings no longer standing "
        "(2 resolved by a declared relationship, 1 no longer a live pair)." in out
    )
    for hash_ in ("h1", "h2", "h3", "h4", "h5", "h6"):
        assert hash_ not in out, "the line counts departures, it does not list them"


def test_departure_line_is_absent_when_nothing_left(
    capsys: pytest.CaptureFixture,
) -> None:
    """Silence on the healthy path — the report gains no permanent furniture."""
    assert "no longer standing" not in _render(_result(()), capsys)


def test_departure_line_agrees_in_number_on_a_single_departure(
    capsys: pytest.CaptureFixture,
) -> None:
    """Calm ASCII (P9) includes not saying "1 findings"."""
    out = _render(_result((DepartedFinding("h1", "h2", "edge-declared"),)), capsys)

    assert "1 previously-reported finding no longer standing" in out
