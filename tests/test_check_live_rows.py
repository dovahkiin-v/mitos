"""``check``'s stale-index probe counts live decisions' rows only.

Phase 3a of the recall-confidence vision (ADR
``check-stale-index-probe-counts-only-live-decisions-rows``). ``probe_stale_index`` reads
the decision outbox once and, over a non-empty backlog, keeps a row only when its node is
in ``GraphStore.get_active_decision_ids()`` — the population the corpus sweep judges,
drifted decisions included. A retired decision's row, an open question's row and a
node-less row hide nothing from the sweep, so they neither make a run partial
(``stale_index``) nor appear among its coverage exclusions.

Each state is driven through ``cmd_check`` twice: the corpus run (``--json``) and the
staged gate. A staged check probes only when the buffer holds a pending entry, so the
staged half writes one and wires a judge that finds a conflict with ``active-q``: an
undegraded staged run exits 1 on that finding, a degraded one exits 2.

Run under ``./venv/bin/python -m pytest`` (PATTERNS — bare ``python`` lacks deps).
"""

import inspect
import sqlite3
from typing import Any, Dict, FrozenSet, List, Tuple

import pytest

from mitos import cli
from mitos.check import (
    CHECK_STALE_RETRY_TOLERANCE,
    ProbeUnavailable,
    exit_code_for,
    execute_corpus_check,
    probe_stale_index,
    run_degradations,
)
from mitos.errors import DatabaseError
from mitos.protocols import GraphStoreProtocol
from mitos.store import GraphStore

from _conflict_helpers import _drain_outbox, _match
from test_check_cli import (  # noqa: F401  (offline is autouse; workspace a fixture)
    _read_check_runs,
    _wire_judge,
    _wire_substrate,
    offline,
    workspace,
)
from test_check_coverage import _judge_for, _run, _run_row
from test_check_probe import (  # noqa: F401  (temp_store / temp_telemetry are fixtures)
    _canned_judge,
    _commit,
    _plan,
    _poison,
    temp_store,
    temp_telemetry,
)
from test_check_staged import _ACTIVE_AXIOM, _PENDING_AXIOM, _finding_judge, _write_decisions
from test_embedding_seed import _open_question

_A = "Live corpus axiom alpha."
_B = "Live corpus axiom beta."
_X1 = "Superseding axiom for x."
_Y1 = "Correcting axiom for y."
_RES = "Resolving axiom for the question."
_DRIFT = "Drifted but still live axiom."

GHOST = "f" * 64


def _world(store: GraphStore) -> Tuple[Dict[str, str], Dict[str, List[Dict[str, Any]]]]:
    """Commits one node in every state the probe must tell apart, then drains the outbox.

    Returns the ids by slug and the neighbourhoods of every axiom the corpus sweep embeds
    (the live decisions'). ``active-q`` is the staged gate's conflict target.
    """
    ids: Dict[str, str] = {}
    ids["live-a"] = _commit(store, "live-a", _A)
    ids["live-b"] = _commit(store, "live-b", _B)
    ids["active-q"] = _commit(store, "active-q", _ACTIVE_AXIOM)
    ids["old-x"] = _commit(store, "old-x", "Retired axiom for x.")
    ids["live-x"] = _commit(store, "live-x", _X1, supersedes=["old-x"])
    ids["wrong-y"] = _commit(store, "wrong-y", "Corrected-away axiom for y.")
    ids["fixed-y"] = _commit(store, "fixed-y", _Y1, corrects=["wrong-y"])
    ids["q-parked"] = _open_question(store, "q-parked", "Parking", "Which way to park?")
    ids["q-resolved"] = _open_question(store, "q-resolved", "Resolving", "Which way?")
    ids["res-d"] = _commit(store, "res-d", _RES, resolves=["q-resolved"])
    ids["drift-d"] = _commit(store, "drift-d", _DRIFT)
    store.write_signal(ids["drift-d"], "drifted")
    _drain_outbox(store)
    neighbourhoods = {
        _A: [_match("live-b", 0.9)], _B: [], _ACTIVE_AXIOM: [], _X1: [], _Y1: [],
        _RES: [], _DRIFT: [],
    }
    return ids, neighbourhoods


def _plant_ghost_row(db_path: str, retry_count: int = 0) -> None:
    """A row whose node is not in the graph. The outbox's node_id is a foreign key, so
    the row goes in through a raw connection, where foreign keys are off by default."""
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO pending_embeddings (node_id, queued_at, retry_count) "
                "VALUES (?, ?, ?)",
                (GHOST, "2026-10-16T00:00:00+00:00", retry_count),
            )
    finally:
        conn.close()


def _queue(store: GraphStore, config: Any, ids: Dict[str, str], state: str) -> None:
    """Queues one transient row for the node in ``state``."""
    if state == "node-less":
        _plant_ghost_row(config.db_path)
    else:
        store.add_pending_embedding(ids[state])


# Which state's transient row gates the run (C1, C5) and which leaves it (C2, C3, C4, C6).
_GATES = {
    "live-a": True,        # C1, the control
    "drift-d": True,       # C5
    "old-x": False,        # C2, superseded
    "wrong-y": False,      # C3, corrected
    "q-parked": False,     # C4
    "q-resolved": False,   # C4
    "node-less": False,    # C6
}


@pytest.mark.parametrize("state", list(_GATES))
def test_c1_to_c6_a_corpus_run_gates_on_live_decisions_rows_only(
    workspace, monkeypatch, capsys, state
) -> None:
    config, store, tel = workspace
    ids, neighbourhoods = _world(store)
    _queue(store, config, ids, state)
    embed, vector = _wire_substrate(monkeypatch, neighbourhoods)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel))

    code, obj = _run(config, capsys)

    if _GATES[state]:
        assert code == 2
        assert "stale_index" in obj["degradations"]
    else:
        assert code == 0
        assert "stale_index" not in obj["degradations"]
        assert _run_row(config, obj["run_id"])["degraded_reason"] is None
    assert obj["coverage_exclusions"] == []


@pytest.mark.parametrize("state", list(_GATES))
def test_c1_to_c6_the_staged_gate_gates_on_live_decisions_rows_only(
    workspace, monkeypatch, capsys, state
) -> None:
    config, store, tel = workspace
    ids, _ = _world(store)
    _queue(store, config, ids, state)
    _write_decisions(config, ("pending-y", _PENDING_AXIOM))
    _wire_substrate(monkeypatch, {_PENDING_AXIOM: [_match("active-q", 0.9)]})
    _wire_judge(monkeypatch, _finding_judge())

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=False)

    (row,) = _read_check_runs(config)
    if _GATES[state]:
        assert code == 2
        assert "stale_index" in row["degraded_reason"]
    else:
        assert code == 1  # the finding, undegraded
        assert row["degraded_reason"] is None
    assert "[Conflict]" in capsys.readouterr().out


def test_c7_a_retired_poison_row_is_no_exclusion_a_live_one_is(
    workspace, monkeypatch, capsys
) -> None:
    config, store, tel = workspace
    ids, neighbourhoods = _world(store)
    _poison(store, ids["old-x"])
    _poison(store, ids["q-parked"])
    _poison(store, ids["live-b"])
    _plant_ghost_row(config.db_path, retry_count=CHECK_STALE_RETRY_TOLERANCE)
    embed, vector = _wire_substrate(monkeypatch, neighbourhoods)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel))

    code, obj = _run(config, capsys)

    assert code == 0
    assert {e["id"] for e in obj["coverage_exclusions"]} == {ids["live-b"]}
    assert _run_row(config, obj["run_id"])["coverage_exclusions"] == 1
    excluded = probe_stale_index(store).excluded
    assert [row.node_id for row in excluded] == [ids["live-b"]]


def test_c8_a_mixed_backlog_keeps_exactly_the_live_rows_sorted(workspace) -> None:
    config, store, _ = workspace
    ids, _ = _world(store)
    for slug in ("live-b", "live-a", "old-x", "q-resolved", "res-d"):
        store.add_pending_embedding(ids[slug])
    for slug in ("fixed-y", "drift-d", "wrong-y", "q-parked"):
        _poison(store, ids[slug])
    _plant_ghost_row(config.db_path)

    probe = probe_stale_index(store)

    assert [row.node_id for row in probe.transient] == sorted(
        ids[slug] for slug in ("live-a", "live-b", "res-d"))
    assert [row.node_id for row in probe.excluded] == sorted(
        ids[slug] for slug in ("drift-d", "fixed-y"))
    assert probe_stale_index(store) == probe


class _LivenessCountingStore:
    """Wraps a real store and counts real calls of the liveness read (not lookups)."""

    def __init__(self, inner: GraphStore) -> None:
        self._inner = inner
        self.liveness_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def get_active_decision_ids(self) -> FrozenSet[str]:
        self.liveness_calls += 1
        return self._inner.get_active_decision_ids()


def test_c9_an_empty_backlog_never_reads_liveness(workspace) -> None:
    _, store, _ = workspace
    ids, _ = _world(store)
    spy = _LivenessCountingStore(store)

    assert probe_stale_index(spy).transient == ()
    assert spy.liveness_calls == 0

    store.add_pending_embedding(ids["live-a"])  # the control: a backlog reads it once
    probe_stale_index(spy)
    assert spy.liveness_calls == 1


def test_c10_the_liveness_read_is_on_the_graph_protocol() -> None:
    assert "get_active_decision_ids" in GraphStoreProtocol.__dict__
    proto = inspect.signature(GraphStoreProtocol.get_active_decision_ids)
    real = inspect.signature(GraphStore.get_active_decision_ids)
    assert list(proto.parameters) == list(real.parameters)
    assert proto.return_annotation == real.return_annotation


class _LivenessFaultStore(_LivenessCountingStore):
    """The Nth liveness read raises; everything else passes through."""

    def __init__(self, inner: GraphStore, *, fail_on: int, exc: BaseException) -> None:
        super().__init__(inner)
        self._fail_on = fail_on
        self._exc = exc

    def get_active_decision_ids(self) -> FrozenSet[str]:
        self.liveness_calls += 1
        if self.liveness_calls == self._fail_on:
            raise self._exc
        return self._inner.get_active_decision_ids()


def test_c11_a_liveness_fault_at_plan_propagates(temp_store, temp_telemetry) -> None:
    a_id = _commit(temp_store, "probe-a", "Probe corpus axiom alpha.")
    _drain_outbox(temp_store)
    _poison(temp_store, a_id)  # a non-empty backlog, so the liveness read runs
    faulty = _LivenessFaultStore(
        temp_store, fail_on=1, exc=sqlite3.OperationalError("simulated liveness fault"))

    with pytest.raises(sqlite3.OperationalError, match="simulated liveness fault"):
        _plan(faulty, {"Probe corpus axiom alpha.": []}, temp_telemetry)


@pytest.mark.parametrize("exc", [
    sqlite3.OperationalError("simulated liveness fault"),
    DatabaseError("simulated liveness open failure"),
], ids=["raw", "wrapped"])
def test_c11_a_liveness_fault_at_execute_end_degrades_typed(
    temp_store, temp_telemetry, exc
) -> None:
    a_axiom, b_axiom = "Probe corpus axiom alpha.", "Probe corpus axiom beta."
    _commit(temp_store, "probe-a", a_axiom)
    b_id = _commit(temp_store, "probe-b", b_axiom)
    _drain_outbox(temp_store)
    _poison(temp_store, b_id)  # excluded, never gating: only the fault can make this 2
    faulty = _LivenessFaultStore(temp_store, fail_on=2, exc=exc)

    plan = _plan(faulty, {a_axiom: [_match("probe-b", 0.9)], b_axiom: []}, temp_telemetry)
    result = execute_corpus_check(
        plan, judge=_canned_judge(plan), telemetry=temp_telemetry, store=faulty
    )

    assert isinstance(result.end_probe, ProbeUnavailable)
    assert len(result.findings) == 1
    assert "probe_read" in run_degradations(result)
    assert exit_code_for(result) == 2
