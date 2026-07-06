"""Tests for the stale-index probe, exit derivation and ``check_runs`` row (Phase 2d).

The CHK-D4 honesty hardware over 2c's engine, plus the CHK-D7 run memory. The
load-bearing properties under test (plan §9):

* the probe partition (T11) — a transient backlog row (``retry_count`` below
  ``CHECK_STALE_RETRY_TOLERANCE``) gates the run partial (exit 2, findings intact);
  an only-poison backlog (``retry_count`` at or above tolerance) NEVER gates — the
  run completes and exits on its findings with the exclusions disclosed by node id
  (the poison-row escape: without it one poison row turns the gate red forever);
* fault dispositions split by stage (KD2) — a start-probe fault propagates (nothing
  spent yet); an end-probe fault degrades typed to ``ProbeUnavailable`` (the paid-for
  findings must survive), catching BOTH raw ``sqlite3.Error`` and wrapped
  ``DatabaseError`` (the 1b twin-catch disease);
* one derivation site (KD4) — ``run_degradations`` feeds both ``exit_code_for``
  (2-dominates-1) and the row's ``degraded_reason``, so process exit and persisted
  row can never fork;
* the scalar-equality law (T12) — every ``check_runs`` scalar derives from the same
  ``CheckRunResult`` the report reads, including the NULL edges (NULL = "could not
  tell", never a masqueraded zero).

Discipline (PATTERNS + the 2c suite): hand-rolled synchronous fakes, a real temp
``GraphStore`` seeded via ``commit_parsed_entry`` (real content hashes; every commit
auto-enqueues an Outbox row — fixtures state their backlog posture explicitly), a
real temp ``TelemetryStore``, production pins (``"SONNET"`` + the default
``CONFLICT_PROMPT_VERSION``). Zero LLM, zero live keys. Run under
``./venv/bin/python -m pytest``.
"""

import os
import shutil
import sqlite3
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import pytest

from mitos import __version__
from mitos.check import (
    CHECK_STALE_RETRY_TOLERANCE,
    BacklogRow,
    CheckPlan,
    ProbeUnavailable,
    StaleProbe,
    check_run_row_from_result,
    coverage_exclusion_ids,
    execute_corpus_check,
    exit_code_for,
    plan_corpus_check,
    probe_stale_index,
    run_degradations,
)
from mitos.conflict import (
    CONFLICT_PROMPT_VERSION,
    CONFLICT_SURFACE_THRESHOLD,
    ConflictUnavailableReason,
    Unavailable,
)
from mitos.errors import DatabaseError, VectorStoreError
from mitos.parser import ParsedEntry
from mitos.store import GraphStore
from mitos.telemetry import ConflictCheckRow, JudgmentBatch, TelemetryStore

from _conflict_helpers import (
    _SequenceJudge,
    _drain_outbox,
    _execution,
    _keyed_substrate,
    _match,
)

# Production judge pin (the 2c suite's discipline — synthetic pins can't execute).
PRODUCTION_ALIAS = "SONNET"


# --------------------------------------------------------------------------- #
# Fixtures — offline env + real temp graph/telemetry stores (the 2c per-file block)
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key, no reachable service — the injected fakes are the only substrate."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def temp_store() -> GraphStore:
    """A temporary file GraphStore booted to the live ladder head."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    store = GraphStore(path)
    yield store
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def temp_telemetry() -> TelemetryStore:
    """A real temp TelemetryStore booted to the telemetry ladder head."""
    tmpdir = tempfile.mkdtemp()
    store = TelemetryStore(os.path.join(tmpdir, ".mitos", "telemetry.sqlite"))
    yield store
    shutil.rmtree(tmpdir, ignore_errors=True)


def _commit(
    store: GraphStore,
    slug: str,
    axiom: str,
    *,
    scope: Optional[List[str]] = None,
    rejected: str = "An alternative.",
    **rels: List[str],
) -> str:
    """Commits a decision and returns its content-hash node id (real hashes for free)."""
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = axiom
    entry.rejected_paths = rejected
    if scope is not None:
        entry.scope = scope
    for name, value in rels.items():
        setattr(entry, name, value)
    return store.commit_parsed_entry(entry).node_id


def _pair_corpus(
    store: GraphStore,
) -> Tuple[str, str, Dict[str, List[Dict[str, Any]]]]:
    """Commits one (a, b) pair whose a-side sweep discovers b; returns ids + neighbourhoods."""
    a_axiom = "Probe corpus axiom alpha."
    b_axiom = "Probe corpus axiom beta."
    a_id = _commit(store, "probe-a", a_axiom)
    b_id = _commit(store, "probe-b", b_axiom)
    neighbourhoods = {a_axiom: [_match("probe-b", 0.9)], b_axiom: []}
    return a_id, b_id, neighbourhoods


def _plan(
    store: Any,
    neighbourhoods: Dict[str, List[Dict[str, Any]]],
    telemetry: Any,
    **kwargs: Any,
) -> CheckPlan:
    """Plans over the keyed substrate at production pins (overridable via kwargs)."""
    embed, vector = _keyed_substrate(
        neighbourhoods, vector_raises=kwargs.pop("vector_raises", None)
    )
    kwargs.setdefault("model_alias", PRODUCTION_ALIAS)
    return plan_corpus_check(
        store=store,
        embed_provider=embed,
        vector_store=vector,
        telemetry=telemetry,
        **kwargs,
    )


def _canned_judge(
    plan: CheckPlan,
    *,
    tenable: bool = False,
    confidence: float = 0.9,
    batch_prefix: str = "probe-batch",
    overrides: Optional[Dict[int, Any]] = None,
) -> _SequenceJudge:
    """One valid canned execution per fresh group, in plan order (distinct batch ids)."""
    rets: List[Any] = []
    for i, group in enumerate(plan.fresh_groups):
        verdicts = [
            (pair.partner_node["slug"], tenable, confidence, f"Probe rationale {i}.")
            for pair in group.pairs
        ]
        rets.append(_execution(verdicts, batch_id=f"{batch_prefix}-{i}"))
    for index, entry in (overrides or {}).items():
        rets[index] = entry
    return _SequenceJudge(rets)


def _seed_verdict(
    telemetry: TelemetryStore,
    *,
    proposal_hash: str,
    candidate_hash: str,
    tenable: bool,
    confidence: float,
    batch_id: str,
    created_at: str,
) -> None:
    """Seeds one prior verdict at PRODUCTION pins through the REAL writer."""
    row = ConflictCheckRow(
        batch_id=batch_id,
        sync_run_id="seed-run",
        surface="sync",
        judged_axiom="Seeded proposal axiom.",
        proposal_rejected_paths=None,
        proposal_scope=None,
        proposed_hash_if_any=proposal_hash,
        candidate_slug="seeded-candidate",
        candidate_hash=candidate_hash,
        candidate_rejected_paths="Seeded alternative.",
        candidate_scope=None,
        tenable=tenable,
        confidence=confidence,
        surfaced=(not tenable) and confidence >= CONFLICT_SURFACE_THRESHOLD,
        candidate_source="embedding_topk",
        model_alias=PRODUCTION_ALIAS,
        prompt_version=CONFLICT_PROMPT_VERSION,
        mitos_version=__version__,
        rationale="Seeded prior rationale.",
    )
    batch = JudgmentBatch(
        batch_id=batch_id,
        model_id=None,
        token_input=1,
        token_output=1,
        token_cache_read=0,
        token_cache_creation=0,
        elapsed_ms=1,
    )
    telemetry.record_judged_batch(batch, [row], created_at)


def _poison(store: GraphStore, node_id: str) -> None:
    """Marks one EXISTING backlog row poison: retry_count reaches the tolerance.

    Order matters (the UPSERT gotcha): ``add_pending_embedding`` after incrementing
    would silently reset ``retry_count`` to 0 and un-poison the row — so the row is
    (re-)added FIRST, then incremented up to the tolerance.
    """
    store.add_pending_embedding(node_id)
    for _ in range(CHECK_STALE_RETRY_TOLERANCE):
        store.increment_pending_attempts(node_id)


class _ProbeFaultStore:
    """Wraps a real store; the Nth ``get_pending_embeddings`` call raises.

    Everything else passes through, so plan/sweep run against the real graph —
    only the probe read is armed (the KD2 fault-disposition seam).
    """

    def __init__(self, inner: GraphStore, *, fail_on: int, exc: BaseException) -> None:
        self._inner = inner
        self._fail_on = fail_on
        self._exc = exc
        self.probe_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def get_pending_embeddings(self) -> List[Dict[str, Any]]:
        self.probe_calls += 1
        if self.probe_calls == self._fail_on:
            raise self._exc
        return self._inner.get_pending_embeddings()


class _OrderSpyStore:
    """Wraps a real store, logging read-method call order (the §9-5 KD1 ordering pin)."""

    def __init__(self, inner: GraphStore) -> None:
        self._inner = inner
        self.log: List[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def get_pending_embeddings(self) -> List[Dict[str, Any]]:
        self.log.append("get_pending_embeddings")
        return self._inner.get_pending_embeddings()

    def get_active_decisions(self, scope: Optional[str] = None) -> List[Dict[str, Any]]:
        self.log.append("get_active_decisions")
        return self._inner.get_active_decisions(scope)

    def get_edges(self) -> List[Dict[str, str]]:
        self.log.append("get_edges")
        return self._inner.get_edges()


class _FailingWriteTelemetry:
    """A real store whose Nth write raises — the KD6 write-failure seam (2c's shape)."""

    def __init__(self, inner: TelemetryStore, fail_on: int) -> None:
        self._inner = inner
        self._fail_on = fail_on
        self.write_attempts = 0

    @property
    def telemetry_path(self) -> str:
        return self._inner.telemetry_path

    def load_reuse_index(self, *, prompt_version: str, model_alias: str) -> Any:
        return self._inner.load_reuse_index(
            prompt_version=prompt_version, model_alias=model_alias
        )

    def record_judged_batch(self, batch: Any, rows: Any, created_at: str) -> None:
        self.write_attempts += 1
        if self.write_attempts == self._fail_on:
            raise DatabaseError("simulated telemetry write failure")
        self._inner.record_judged_batch(batch, rows, created_at)


# --------------------------------------------------------------------------- #
# §9-1 [TDD] T11 — the probe partition: transient gates, poison never does
# --------------------------------------------------------------------------- #

def test_transient_backlog_gates_exit_2_findings_intact(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-1 (TDD-first): an undrained fresh commit (retry 0) labels the run partial —
    ``"stale_index"`` fires, exit 2 dominates the new finding, and the finding still
    rides the result (partial results labeled, never discarded)."""
    _, _, neighbourhoods = _pair_corpus(temp_store)  # commits enqueue: transient backlog

    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    result = execute_corpus_check(
        plan, judge=_canned_judge(plan), telemetry=temp_telemetry, store=temp_store
    )

    assert plan.start_probe.transient  # every commit enqueued at retry 0
    assert not plan.start_probe.excluded
    assert "stale_index" in run_degradations(result)
    assert exit_code_for(result) == 2  # degraded dominates the new finding
    assert len(result.findings) == 1 and result.findings[0].novelty == "new"


def test_only_poison_backlog_never_gates_exits_on_findings(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-1: an only-poison backlog (retry >= tolerance) produces NO gate — the run
    exits on its findings (here: a new finding, exit 1) and the poison node is
    disclosed by id in the exclusion projection (the CHK-D4 escape)."""
    a_id, b_id, neighbourhoods = _pair_corpus(temp_store)
    _drain_outbox(temp_store)
    _poison(temp_store, b_id)

    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    result = execute_corpus_check(
        plan, judge=_canned_judge(plan), telemetry=temp_telemetry, store=temp_store
    )

    assert not result.start_probe.transient
    assert [row.node_id for row in result.start_probe.excluded] == [b_id]
    assert "stale_index" not in run_degradations(result)
    assert exit_code_for(result) == 1  # the new finding, not the poison, decides
    assert coverage_exclusion_ids(result) == (b_id,)


def test_only_poison_backlog_with_no_findings_exits_0(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-1: poison + tenable verdicts → a certified-clean exit 0, exclusions still
    disclosed — the wound is visible and non-fatal, forever, every run."""
    a_id, _, neighbourhoods = _pair_corpus(temp_store)
    _drain_outbox(temp_store)
    _poison(temp_store, a_id)

    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    result = execute_corpus_check(
        plan,
        judge=_canned_judge(plan, tenable=True),
        telemetry=temp_telemetry,
        store=temp_store,
    )

    assert run_degradations(result) == ()
    assert exit_code_for(result) == 0
    assert coverage_exclusion_ids(result) == (a_id,)


def test_mixed_backlog_gates_and_still_discloses_the_poison(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-1: transient + poison together → the transient row wins (gate fires), and
    the poison node is still named in the exclusions."""
    a_id, b_id, neighbourhoods = _pair_corpus(temp_store)
    _drain_outbox(temp_store)
    _poison(temp_store, a_id)
    temp_store.add_pending_embedding(b_id)  # a fresh transient row (retry 0)

    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    result = execute_corpus_check(
        plan,
        judge=_canned_judge(plan, tenable=True),
        telemetry=temp_telemetry,
        store=temp_store,
    )

    assert [row.node_id for row in result.start_probe.transient] == [b_id]
    assert "stale_index" in run_degradations(result)
    assert exit_code_for(result) == 2
    assert coverage_exclusion_ids(result) == (a_id,)


def test_drained_corpus_probes_healthy_no_token(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-1: a drained outbox is HEALTHY on both probes — empty ``StaleProbe``s, no
    token, exit derived from findings alone (empty-is-healthy, never a degradation)."""
    _, _, neighbourhoods = _pair_corpus(temp_store)
    _drain_outbox(temp_store)

    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    result = execute_corpus_check(
        plan,
        judge=_canned_judge(plan, tenable=True),
        telemetry=temp_telemetry,
        store=temp_store,
    )

    assert result.start_probe == StaleProbe((), ())
    assert result.end_probe == StaleProbe((), ())
    assert run_degradations(result) == ()
    assert exit_code_for(result) == 0


# --------------------------------------------------------------------------- #
# §9-2 [TDD] T12 — the scalar-equality law: the row derives from the one result
# --------------------------------------------------------------------------- #

def test_check_run_row_scalars_equal_the_report(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-2 (TDD-first): one run carrying a known (reused) finding, a new (fresh)
    finding, and a poison exclusion — every row scalar equals the independently
    computed report value, timestamps echoed, ``exit_code`` the passed value."""
    a_axiom = "Row corpus axiom alpha."
    b_axiom = "Row corpus axiom beta."
    c_axiom = "Row corpus axiom gamma."
    d_axiom = "Row corpus axiom delta."
    a_id = _commit(temp_store, "row-a", a_axiom)
    b_id = _commit(temp_store, "row-b", b_axiom)
    c_id = _commit(temp_store, "row-c", c_axiom)
    d_id = _commit(temp_store, "row-d", d_axiom)
    neighbourhoods = {
        a_axiom: [_match("row-b", 0.9)],
        b_axiom: [],
        c_axiom: [_match("row-d", 0.9)],
        d_axiom: [],
    }
    # The (a, b) pair rides a seeded prior FINDING verdict → reused, novelty "known".
    _seed_verdict(
        temp_telemetry,
        proposal_hash=a_id,
        candidate_hash=b_id,
        tenable=False,
        confidence=0.95,
        batch_id="seed-batch-row",
        created_at="2026-07-06T00:00:00+00:00",
    )
    _drain_outbox(temp_store)
    _poison(temp_store, d_id)

    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    result = execute_corpus_check(
        plan, judge=_canned_judge(plan), telemetry=temp_telemetry, store=temp_store
    )
    exit_code = exit_code_for(result)
    row = check_run_row_from_result(result, mode="corpus", exit_code=exit_code)

    # Independently computed report values — the law: row == report, structurally.
    assert row.run_id == result.run_id
    assert row.mode == "corpus"
    assert row.started_at == result.started_at
    assert row.ended_at == result.ended_at
    assert row.exit_code == exit_code == 1  # new finding, healthy run (poison escapes)
    assert row.nodes_swept == result.nodes_swept == 4
    assert row.pairs_judged_fresh == result.pairs_judged_fresh == 1
    assert row.pairs_reused == result.pairs_reused == 1
    assert row.findings_new == sum(1 for f in result.findings if f.novelty == "new") == 1
    assert (
        row.findings_known
        == sum(1 for f in result.findings if f.novelty == "known")
        == 1
    )
    assert row.coverage_exclusions == len(coverage_exclusion_ids(result)) == 1
    assert row.degraded_reason is None  # poison produces no token — healthy
    assert row.mitos_version == __version__


def test_degraded_run_row_joins_the_tokens_in_declaration_order(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-2: a run degraded on several axes stamps ``degraded_reason`` as the
    comma-joined ``run_degradations`` tuple — the SAME derivation the exit reads
    (KD4: the row and the process exit can never fork)."""
    _, _, neighbourhoods = _pair_corpus(temp_store)  # undrained → stale_index
    failing = _FailingWriteTelemetry(temp_telemetry, fail_on=1)  # → telemetry_write

    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    result = execute_corpus_check(
        plan, judge=_canned_judge(plan), telemetry=failing, store=temp_store
    )
    row = check_run_row_from_result(result, mode="corpus", exit_code=exit_code_for(result))

    assert run_degradations(result) == ("telemetry_write", "stale_index")
    assert row.degraded_reason == "telemetry_write,stale_index"
    assert row.exit_code == 2


def test_healthy_run_row_zeros_are_genuine_zeros(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-2: a healthy zero-finding run stamps 0/0/0 (partitioned, none; both probes
    clean of poison) and ``degraded_reason`` NULL — zeros are facts, never NULL."""
    _, _, neighbourhoods = _pair_corpus(temp_store)
    _drain_outbox(temp_store)

    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    result = execute_corpus_check(
        plan,
        judge=_canned_judge(plan, tenable=True),
        telemetry=temp_telemetry,
        store=temp_store,
    )
    row = check_run_row_from_result(result, mode="corpus", exit_code=exit_code_for(result))

    assert row.exit_code == 0
    assert row.findings_new == 0 and row.findings_known == 0
    assert row.coverage_exclusions == 0
    assert row.degraded_reason is None


def test_reuse_unavailable_stamps_null_partition_even_with_zero_findings(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-2 (the NULL edge that costs something): under ``reuse_unavailable`` the
    partition columns stamp NULL even when the finding count is ZERO — the rule is
    one-flag-derivable, and a degraded run must never read as a clean data point."""
    _, _, neighbourhoods = _pair_corpus(temp_store)
    _drain_outbox(temp_store)

    plan = _plan(temp_store, neighbourhoods, None)  # reuse read unavailable at plan
    result = execute_corpus_check(
        plan,
        judge=_canned_judge(plan, tenable=True),  # zero findings
        telemetry=temp_telemetry,
        store=temp_store,
    )
    row = check_run_row_from_result(result, mode="corpus", exit_code=exit_code_for(result))

    assert not result.findings
    assert row.findings_new is None and row.findings_known is None
    assert row.exit_code == 2  # reuse_read — never a clean trend point
    assert row.pairs_reused == 0  # a TRUE zero: the run genuinely reused nothing


# --------------------------------------------------------------------------- #
# §9-3 / §9-4 — probe mechanics: end-probe growth, fault dispositions
# --------------------------------------------------------------------------- #

def test_mid_run_commit_is_caught_by_the_end_probe(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-3: a clean start probe does not certify alone — a backlog row appearing
    between plan and execute (the mid-run-commit case) turns the END probe
    transient and the run partial (two reads seal the truth)."""
    a_id, _, neighbourhoods = _pair_corpus(temp_store)
    _drain_outbox(temp_store)

    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    assert plan.start_probe == StaleProbe((), ())
    temp_store.add_pending_embedding(a_id)  # a commit landing mid-run
    result = execute_corpus_check(
        plan,
        judge=_canned_judge(plan, tenable=True),
        telemetry=temp_telemetry,
        store=temp_store,
    )

    assert isinstance(result.end_probe, StaleProbe)
    assert [row.node_id for row in result.end_probe.transient] == [a_id]
    assert "stale_index" in run_degradations(result)
    assert exit_code_for(result) == 2


def test_end_probe_raw_sqlite_fault_degrades_typed_findings_preserved(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-4 (twin-catch, raw half): a raw ``sqlite3.Error`` from the end-probe read
    degrades to ``ProbeUnavailable`` — findings preserved, ``"probe_read"`` fires,
    exit 2, and the row's ``coverage_exclusions`` is NULL (never a fake zero)."""
    _, _, neighbourhoods = _pair_corpus(temp_store)
    _drain_outbox(temp_store)
    faulty = _ProbeFaultStore(
        temp_store, fail_on=2, exc=sqlite3.OperationalError("simulated query fault")
    )

    plan = _plan(faulty, neighbourhoods, temp_telemetry)
    result = execute_corpus_check(
        plan, judge=_canned_judge(plan), telemetry=temp_telemetry, store=faulty
    )
    row = check_run_row_from_result(result, mode="corpus", exit_code=exit_code_for(result))

    assert isinstance(result.end_probe, ProbeUnavailable)
    assert "simulated query fault" in result.end_probe.detail
    assert len(result.findings) == 1  # the paid-for finding survives
    assert "probe_read" in run_degradations(result)
    assert exit_code_for(result) == 2
    assert row.coverage_exclusions is None
    # Disclose what you know: the start-probe half still projects (empty here).
    assert coverage_exclusion_ids(result) == ()


def test_end_probe_wrapped_database_error_also_degrades_typed(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-4 (twin-catch, wrapped half): a Mitos ``DatabaseError`` (the open-failure
    wrap) is caught the same way — catch one type only and the other escapes."""
    _, _, neighbourhoods = _pair_corpus(temp_store)
    _drain_outbox(temp_store)
    faulty = _ProbeFaultStore(
        temp_store, fail_on=2, exc=DatabaseError("simulated open failure")
    )

    plan = _plan(faulty, neighbourhoods, temp_telemetry)
    result = execute_corpus_check(
        plan,
        judge=_canned_judge(plan, tenable=True),
        telemetry=temp_telemetry,
        store=faulty,
    )

    assert isinstance(result.end_probe, ProbeUnavailable)
    assert "probe_read" in run_degradations(result)


def test_start_probe_fault_propagates(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-4: at plan entry NOTHING is spent — a store fault propagates exactly like
    ``snapshot_corpus``'s (KD2: a broken graph must never masquerade as clean),
    and ``probe_stale_index`` itself owns no disposition (no try/except)."""
    _, _, neighbourhoods = _pair_corpus(temp_store)
    faulty = _ProbeFaultStore(
        temp_store, fail_on=1, exc=sqlite3.OperationalError("simulated entry fault")
    )

    with pytest.raises(sqlite3.OperationalError, match="simulated entry fault"):
        _plan(faulty, neighbourhoods, temp_telemetry)


# --------------------------------------------------------------------------- #
# §9-5 — determinism + the KD1 ordering pin
# --------------------------------------------------------------------------- #

def test_probe_partitions_and_sorts_deterministically(temp_store: GraphStore) -> None:
    """§9-5: both tuples are node-id-sorted regardless of insert/UPSERT order, the
    partition boundary is exactly the tolerance, and re-probing an unchanged
    backlog is byte-identical (the DoD-3 determinism half)."""
    ids = sorted(
        _commit(temp_store, f"det-{i}", f"Determinism axiom {i}.") for i in range(4)
    )
    _drain_outbox(temp_store)
    # Re-seed in reverse-sorted order: 2 poison (>= tolerance), 2 transient (below).
    for node_id in reversed(ids):
        temp_store.add_pending_embedding(node_id)
    _poison(temp_store, ids[3])
    _poison(temp_store, ids[0])
    for _ in range(CHECK_STALE_RETRY_TOLERANCE - 1):
        temp_store.increment_pending_attempts(ids[2])  # tolerance-1 stays transient

    probe = probe_stale_index(temp_store)

    assert [row.node_id for row in probe.transient] == [ids[1], ids[2]]
    assert [row.node_id for row in probe.excluded] == [ids[0], ids[3]]
    assert all(isinstance(row, BacklogRow) for row in probe.transient + probe.excluded)
    assert all(
        row.retry_count < CHECK_STALE_RETRY_TOLERANCE for row in probe.transient
    )
    assert all(
        row.retry_count >= CHECK_STALE_RETRY_TOLERANCE for row in probe.excluded
    )
    assert probe_stale_index(temp_store) == probe  # unchanged backlog → identical


def test_probe_reads_once_per_stage_and_fires_before_the_snapshot(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-5 (the KD1 ordering pin): exactly one ``get_pending_embeddings`` per
    stage, and at plan it fires BEFORE the snapshot reads — the backlog that
    predates the snapshot is exactly what thins the sweep the snapshot defines."""
    _, _, neighbourhoods = _pair_corpus(temp_store)
    _drain_outbox(temp_store)
    spy = _OrderSpyStore(temp_store)

    plan = _plan(spy, neighbourhoods, temp_telemetry)
    plan_log = list(spy.log)
    result = execute_corpus_check(
        plan,
        judge=_canned_judge(plan, tenable=True),
        telemetry=temp_telemetry,
        store=spy,
    )

    assert plan_log.count("get_pending_embeddings") == 1
    assert plan_log.index("get_pending_embeddings") < plan_log.index(
        "get_active_decisions"
    )
    execute_log = spy.log[len(plan_log):]
    assert execute_log.count("get_pending_embeddings") == 1
    assert isinstance(result.end_probe, StaleProbe)


# --------------------------------------------------------------------------- #
# §9-6 / §9-7 — the exit contract: each degradation alone → 2; 2 dominates 1
# --------------------------------------------------------------------------- #

def test_sweep_trip_alone_exits_2(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-6: a mid-sweep vector trip → ``"sweep"`` token, exit 2."""
    a_axiom = "Sweep trip axiom alpha."
    b_axiom = "Sweep trip axiom beta."
    _commit(temp_store, "trip-a", a_axiom)
    _commit(temp_store, "trip-b", b_axiom)
    _drain_outbox(temp_store)
    neighbourhoods = {a_axiom: [], b_axiom: []}

    plan = _plan(
        temp_store,
        neighbourhoods,
        temp_telemetry,
        vector_raises={a_axiom: VectorStoreError("vector store down")},
    )
    result = execute_corpus_check(
        plan, judge=_canned_judge(plan), telemetry=temp_telemetry, store=temp_store
    )

    assert run_degradations(result) == ("sweep",)
    assert exit_code_for(result) == 2


def test_judgment_trip_alone_exits_2(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-6: a judgment ``Unavailable`` → ``"judgment"`` token, exit 2."""
    _, _, neighbourhoods = _pair_corpus(temp_store)
    _drain_outbox(temp_store)

    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    judge = _canned_judge(
        plan,
        overrides={
            0: Unavailable(
                reason=ConflictUnavailableReason.JUDGMENT, detail="judge died"
            )
        },
    )
    result = execute_corpus_check(
        plan, judge=judge, telemetry=temp_telemetry, store=temp_store
    )

    assert run_degradations(result) == ("judgment",)
    assert exit_code_for(result) == 2


def test_telemetry_write_failure_alone_exits_2(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-6: a per-batch write failure → ``"telemetry_write"`` token, exit 2 — the
    finding still reports (the run is degraded, the judgment loop never aborts)."""
    _, _, neighbourhoods = _pair_corpus(temp_store)
    _drain_outbox(temp_store)
    failing = _FailingWriteTelemetry(temp_telemetry, fail_on=1)

    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    result = execute_corpus_check(
        plan, judge=_canned_judge(plan), telemetry=failing, store=temp_store
    )

    assert run_degradations(result) == ("telemetry_write",)
    assert exit_code_for(result) == 2
    assert len(result.findings) == 1


def test_new_finding_healthy_exits_1_and_known_only_exits_0(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-7: a healthy run with a new finding → 1; re-running (the finding now a
    standing prior, reused) → 0 — the repetition story the summary row records."""
    _, _, neighbourhoods = _pair_corpus(temp_store)
    _drain_outbox(temp_store)

    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    first = execute_corpus_check(
        plan, judge=_canned_judge(plan), telemetry=temp_telemetry, store=temp_store
    )
    assert exit_code_for(first) == 1

    replan = _plan(temp_store, neighbourhoods, temp_telemetry)
    second = execute_corpus_check(
        replan, judge=_SequenceJudge([]), telemetry=temp_telemetry, store=temp_store
    )
    assert [f.novelty for f in second.findings] == ["known"]
    assert exit_code_for(second) == 0


def test_clean_corpus_exits_0(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-7: no findings, healthy probes → 0."""
    _, _, neighbourhoods = _pair_corpus(temp_store)
    _drain_outbox(temp_store)

    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    result = execute_corpus_check(
        plan,
        judge=_canned_judge(plan, tenable=True),
        telemetry=temp_telemetry,
        store=temp_store,
    )

    assert exit_code_for(result) == 0


def test_unpartitioned_findings_structurally_cannot_reach_exit_1(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-7 (the structural pin from §3): findings with ``novelty=None`` occur only
    under ``reuse_unavailable``, which already yields ``"reuse_read"`` → 2 — the
    exit function reads novelty, never re-derives finding-ness, and never sees an
    unpartitioned finding on a non-degraded run."""
    _, _, neighbourhoods = _pair_corpus(temp_store)
    _drain_outbox(temp_store)

    plan = _plan(temp_store, neighbourhoods, None)  # reuse read unavailable
    result = execute_corpus_check(
        plan, judge=_canned_judge(plan), telemetry=temp_telemetry, store=temp_store
    )

    assert [f.novelty for f in result.findings] == [None]
    assert "reuse_read" in run_degradations(result)
    assert exit_code_for(result) == 2  # never 1


# --------------------------------------------------------------------------- #
# §9-9 (builder half) — loud ValueError on out-of-contract mode/exit_code
# --------------------------------------------------------------------------- #

def test_row_builder_rejects_out_of_contract_mode_and_exit_code(
    temp_store: GraphStore, temp_telemetry: TelemetryStore
) -> None:
    """§9-9: the builder validates ``mode``/``exit_code`` with a loud ``ValueError``
    (a programming error, cheaper than the CHECK's ``IntegrityError`` at write)."""
    _, _, neighbourhoods = _pair_corpus(temp_store)
    _drain_outbox(temp_store)
    plan = _plan(temp_store, neighbourhoods, temp_telemetry)
    result = execute_corpus_check(
        plan,
        judge=_canned_judge(plan, tenable=True),
        telemetry=temp_telemetry,
        store=temp_store,
    )

    with pytest.raises(ValueError, match="mode"):
        check_run_row_from_result(result, mode="watch", exit_code=0)
    with pytest.raises(ValueError, match="exit_code"):
        check_run_row_from_result(result, mode="corpus", exit_code=3)
    # 'staged' is inside the closed set — the builder does not police semantics.
    staged = check_run_row_from_result(result, mode="staged", exit_code=0)
    assert staged.mode == "staged"
