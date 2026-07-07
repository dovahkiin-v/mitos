"""Tests for the ``mitos check --staged`` gate mode (Phase 3b).

Drives ``cli.cmd_check(config, staged=True, …)`` / ``cli._run_staged_check`` directly
against a real tmp-workspace ``GraphStore`` + ``TelemetryStore`` (the ``test_check_cli``
fixture manner), with keyed fakes for the external substrate + judge. The load-bearing
surface under test (plan §9):

* the pure-read pending predicate (KD1, W9) — a committed entry's content hash is
  excluded, an edited (new-hash) entry is pending, an OQ entry is never selected, and
  the graph is never written;
* the fail-closed 0/1/2 exit contract (§3) — *any* pending contradiction gates (exit 1,
  no novelty partition), a clean buffer passes (0), no pending entries short-circuits
  (0, zero LLM contact, no probe, no row), and a gate that cannot run says so (2);
* the no-row-unless-judged rule (KD2/KD8) and ``surface='check'`` telemetry attribution
  (KD7) — a judged run leaves exactly one ``mode='staged'`` ``check_runs`` row + its
  ``conflict_checks`` rows under the run's id; a no-pending/refused/clean-empty run none;
* the machine-stable staged ``--json`` object (§8/KD9), through ``_emit_json`` only.

Discipline (PATTERNS + the 3a suite): hand-rolled synchronous fakes, real temp stores
seeded via ``commit_parsed_entry`` (every commit auto-enqueues an Outbox row — fixtures
state their backlog posture explicitly via ``_drain_outbox``), a real working-tree
``decisions.md`` parsed with ``parse_entry_stream``, production pins. Zero LLM, zero live
keys. Run under ``./venv/bin/python -m pytest``.
"""

import json
import os
import shutil
import sqlite3
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import pytest

from mitos import __version__, check, cli
from mitos.config import MitosConfig
from mitos.conflict import ConflictUnavailableReason, Unavailable
from mitos.errors import DatabaseError, VectorStoreError
from mitos.parser import ParsedEntry, parse_entry_stream
from mitos.store import GraphStore, open_connection
from mitos.telemetry import TelemetryStore

from mitos.conflict import (
    Candidate, ConflictCheckResult, JudgeInput, JudgedPair, Judgment,
)

from _conflict_helpers import _drain_outbox, _execution, _keyed_substrate, _match, _SequenceJudge
from test_check_probe import _commit, _poison

PRODUCTION_ALIAS = "SONNET"


# --------------------------------------------------------------------------- #
# Fixtures — offline env + a real temp workspace (config + graph + telemetry)
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key, no reachable service — the injected fakes are the only substrate."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def workspace() -> Tuple[MitosConfig, GraphStore, TelemetryStore]:
    """A temp workspace whose graph + telemetry live at ``config``'s own paths."""
    tmpdir = tempfile.mkdtemp()
    config = MitosConfig(tmpdir)
    store = GraphStore(config.db_path)
    telemetry = TelemetryStore(config.telemetry_path)
    yield config, store, telemetry
    shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Helpers — decisions.md authoring, seam wiring, telemetry readback
# --------------------------------------------------------------------------- #

def _decisions_md(*entries: Tuple[str, str]) -> str:
    """Renders a working-tree ``decisions.md`` from ``(slug, axiom)`` pairs.

    Emits the ``parse_entry_stream`` shape (a BEGIN-ENTRIES sentinel + ``## slug``
    blocks with the M5-required ``**Rejected:**``). No ``**Mechanisms:**`` field, so
    the parsed ``.mechanisms`` is ``[]`` — matching ``_commit``'s default, so an entry
    whose ``**Decided:**`` text equals a committed axiom hashes to the same node id.
    """
    lines = ["<!-- BEGIN ENTRIES -->\n\n"]
    for slug, axiom in entries:
        lines.append(f"## 2026-07-05 — {slug} — Title\n")
        lines.append(f"**Decided:** {axiom}\n")
        lines.append(f"**Rejected:** A rejected alternative for {slug}.\n\n")
    return "".join(lines)


def _write_decisions(config: MitosConfig, *entries: Tuple[str, str]) -> None:
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write(_decisions_md(*entries))


def _wire_substrate(
    monkeypatch: pytest.MonkeyPatch,
    neighbourhoods: Dict[str, List[Dict[str, Any]]],
    *,
    vector_raises: Optional[Dict[str, BaseException]] = None,
) -> Tuple[Any, Any]:
    """Monkeypatches ``cli._build_check_substrate`` to keyed fakes; returns them."""
    embed, vector = _keyed_substrate(neighbourhoods, vector_raises=vector_raises)
    monkeypatch.setattr(cli, "_build_check_substrate",
                        lambda config: (embed, vector, None, None))
    return embed, vector


def _wire_judge(monkeypatch: pytest.MonkeyPatch, judge: Any) -> List[bool]:
    """Monkeypatches ``cli._build_check_judge`` to return ``judge``; logs invocation."""
    invoked: List[bool] = []

    def builder() -> Any:
        invoked.append(True)
        return judge
    monkeypatch.setattr(cli, "_build_check_judge", builder)
    return invoked


def _read_check_runs(config: MitosConfig) -> List[Dict[str, Any]]:
    if not os.path.exists(config.telemetry_path):
        return []
    conn = open_connection(config.telemetry_path, read_only=True)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM check_runs").fetchall()]
    finally:
        conn.close()


def _read_conflict_checks(config: MitosConfig) -> List[Dict[str, Any]]:
    if not os.path.exists(config.telemetry_path):
        return []
    conn = open_connection(config.telemetry_path, read_only=True)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM conflict_checks").fetchall()]
    finally:
        conn.close()


class _FakeStdin:
    """A stdin stand-in with a controllable ``isatty`` (the 3a pattern)."""

    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class _FailingWriteTelemetry:
    """Wraps a real telemetry store; only ``record_check_run`` raises (the KD8 seam)."""

    def __init__(self, inner: TelemetryStore) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def record_check_run(self, row: Any) -> None:
        raise DatabaseError("provoked summary-row write fault")


# A pending↔committed pair whose pending sweep discovers the committed decision.
_PENDING_AXIOM = "Pending gate axiom that may conflict with the active corpus."
_ACTIVE_AXIOM = "Active corpus axiom the pending entry may contradict."


def _seed_active(store: GraphStore) -> str:
    """Commits one active decision the pending entry can be found to conflict with."""
    node_id = _commit(store, "active-q", _ACTIVE_AXIOM)
    _drain_outbox(store)
    return node_id


def _finding_judge(batch_id: str = "staged-b0") -> _SequenceJudge:
    """A one-call judge returning a not-tenable, high-confidence verdict for active-q.

    ``batch_id`` is the ``judgment_batches`` PK — a re-run must mint a DISTINCT id (as
    the real executor's per-call uuid does) or the second persist collides on the PK.
    """
    return _SequenceJudge([
        _execution([("active-q", False, 0.9, "They cannot both stand.")], batch_id=batch_id),
    ])


# --------------------------------------------------------------------------- #
# 1 — the pure-read pending predicate (KD1, W9)
# --------------------------------------------------------------------------- #

def test_1a_committed_entry_excluded_edited_entry_pending(workspace) -> None:
    """A committed entry's hash is excluded; an edited (new-hash) entry is pending."""
    config, store, _telemetry = workspace
    _commit(store, "committed-x", "A committed axiom already in the graph.")
    text = _decisions_md(
        ("committed-x", "A committed axiom already in the graph."),  # excluded
        ("pending-y", "A never-committed axiom, still in the buffer."),  # pending
    )
    entries = parse_entry_stream(text, "decision")

    pending = cli._pending_decision_entries(store, entries)

    assert [e.slug for e in pending] == ["pending-y"]


def test_1b_oq_entry_never_selected(workspace) -> None:
    """The ``entry.kind == 'decision'`` safety-belt drops an open_question entry."""
    config, store, _telemetry = workspace
    oq = ParsedEntry("open_question", "some-oq", 1, 3)
    oq.topic = "a parked topic"
    oq.questions_raised = ["what about X?"]

    assert cli._pending_decision_entries(store, [oq]) == []


def test_1c_predicate_writes_nothing_to_the_graph(workspace) -> None:
    """The predicate touches the store only through ``get_node`` — no graph write (KD1)."""
    config, store, _telemetry = workspace
    committed = _commit(store, "committed-x", "A committed axiom already in the graph.")
    before = store.get_node(committed)["source"]
    text = _decisions_md(("committed-x", "A committed axiom already in the graph."))
    entries = parse_entry_stream(text, "decision")

    cli._pending_decision_entries(store, entries)

    # note_source_reencounter (the write two lines below the copied predicate) would
    # have mutated the source; it must not have been called.
    assert store.get_node(committed)["source"] == before


# --------------------------------------------------------------------------- #
# 2 / 3 / 5 — the gate verdict (exit 1 / 0 / re-attempt)
# --------------------------------------------------------------------------- #

def test_2_pending_contradiction_exits_1_names_both_sides(workspace, monkeypatch, capsys) -> None:
    """A pending undeclared contradiction → exit 1; report names both sides + rationale."""
    config, store, telemetry = workspace
    _seed_active(store)
    _write_decisions(config, ("pending-y", _PENDING_AXIOM))
    _wire_substrate(monkeypatch, {_PENDING_AXIOM: [_match("active-q", 0.9)]})
    _wire_judge(monkeypatch, _finding_judge())

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=False)

    assert code == 1
    out = capsys.readouterr().out
    assert "[Conflict]" in out
    assert "pending-y" in out and "active-q" in out
    assert "They cannot both stand." in out
    assert "Resolve by declaring a relationship" in out


def test_3_clean_buffer_exits_0(workspace, monkeypatch, capsys) -> None:
    """Pending entries that judge clean (no candidate) → exit 0, no findings."""
    config, store, telemetry = workspace
    _seed_active(store)
    _write_decisions(config, ("pending-y", _PENDING_AXIOM))
    # Empty neighbourhood ⇒ clean-empty facade result (no batch fires).
    _wire_substrate(monkeypatch, {_PENDING_AXIOM: []})
    _wire_judge(monkeypatch, _SequenceJudge([]))

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=False)

    assert code == 0
    assert "Gate clear" in capsys.readouterr().out
    assert _read_check_runs(config) == []  # clean-empty ⇒ no judgment ⇒ no row


def test_5_unchanged_bad_buffer_reattempt_exits_1_again(workspace, monkeypatch) -> None:
    """The gate does not stop blocking on a retry of the same bad buffer (CHK-D10)."""
    config, store, telemetry = workspace
    _seed_active(store)
    _write_decisions(config, ("pending-y", _PENDING_AXIOM))
    _wire_substrate(monkeypatch, {_PENDING_AXIOM: [_match("active-q", 0.9)]})

    for attempt in range(2):
        # Fresh judge per attempt (distinct batch id per run — the batches PK).
        _wire_judge(monkeypatch, _finding_judge(batch_id=f"staged-b{attempt}"))
        code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                             assume_yes=False, as_json=False)
        assert code == 1


# --------------------------------------------------------------------------- #
# 4 — the no-pending short-circuit (KD2)
# --------------------------------------------------------------------------- #

def test_4_no_pending_exits_0_zero_contact_no_probe_no_row(workspace, monkeypatch, capsys) -> None:
    """No pending entries → exit 0; probe/substrate/judge seams untouched, no row."""
    config, store, telemetry = workspace
    _commit(store, "committed-x", "A committed axiom already in the graph.")
    _drain_outbox(store)
    _write_decisions(config, ("committed-x", "A committed axiom already in the graph."))

    probe_calls: List[bool] = []
    monkeypatch.setattr(check, "probe_stale_index",
                        lambda store: probe_calls.append(True))
    sub_calls: List[bool] = []
    monkeypatch.setattr(cli, "_build_check_substrate",
                        lambda config: sub_calls.append(True) or (None, None, "x", "x"))
    judge_calls = _wire_judge(monkeypatch, None)

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=False)

    assert code == 0
    assert "no pending decisions" in capsys.readouterr().out.lower()
    assert probe_calls == [] and sub_calls == [] and judge_calls == []
    assert _read_check_runs(config) == []


def test_4b_absent_decisions_file_exits_0(workspace, monkeypatch) -> None:
    """An absent working-tree decisions.md is an empty pending set → exit 0."""
    config, store, telemetry = workspace
    assert not os.path.exists(config.decisions_file)
    judge_calls = _wire_judge(monkeypatch, None)

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=False)

    assert code == 0
    assert judge_calls == []
    assert _read_check_runs(config) == []


# --------------------------------------------------------------------------- #
# 6 — flag-combo rejection (pre-store, exit 2)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("scope,fresh", [("auth", False), (None, True)])
def test_6_flag_combo_rejected_exit_2_no_store_contact(workspace, monkeypatch, scope, fresh) -> None:
    """``--staged --scope`` / ``--staged --fresh`` → exit 2 before any store contact."""
    config, store, telemetry = workspace
    # Arm the substrate to explode if touched — the guard must return before it.
    def _boom(config):
        raise AssertionError("substrate built despite an invalid flag combo")
    monkeypatch.setattr(cli, "_build_check_substrate", _boom)

    code = cli.cmd_check(config, staged=True, scope=scope, fresh=fresh,
                         assume_yes=False, as_json=False)

    assert code == 2
    assert _read_check_runs(config) == []


def test_6b_flag_combo_rejected_json_error_object(workspace, capsys) -> None:
    """The flag-combo refusal emits an error object on ``--json`` (never a bare exit)."""
    config, store, telemetry = workspace
    code = cli.cmd_check(config, staged=True, scope="auth", fresh=False,
                         assume_yes=False, as_json=True)
    assert code == 2
    obj = json.loads(capsys.readouterr().out)
    assert obj["code"] == "invalid_flags"


# --------------------------------------------------------------------------- #
# 7 / 8 — fail-closed preconditions (no key / no providers), exit 2, no row
# --------------------------------------------------------------------------- #

def test_7_no_anthropic_key_with_pending_exits_2_no_row(workspace, monkeypatch, capsys) -> None:
    """A missing judge key with pending entries → fail-closed exit 2, no row (KD5)."""
    config, store, telemetry = workspace
    _seed_active(store)
    _write_decisions(config, ("pending-y", _PENDING_AXIOM))
    _wire_substrate(monkeypatch, {_PENDING_AXIOM: [_match("active-q", 0.9)]})
    _wire_judge(monkeypatch, None)  # keyless

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=False)

    assert code == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert _read_check_runs(config) == []


def test_8_providers_absent_with_pending_exits_2_no_row(workspace, monkeypatch, capsys) -> None:
    """Embed/vector absent with pending entries → fail-closed exit 2, names the component."""
    config, store, telemetry = workspace
    _seed_active(store)
    _write_decisions(config, ("pending-y", _PENDING_AXIOM))
    monkeypatch.setattr(cli, "_build_check_substrate",
                        lambda config: (None, object(), "no GEMINI key", None))
    judge_calls = _wire_judge(monkeypatch, _finding_judge())

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=False)

    assert code == 2
    assert "embeddings" in capsys.readouterr().err
    assert judge_calls == []  # fail-closed before the judge is built
    assert _read_check_runs(config) == []


# --------------------------------------------------------------------------- #
# 9 — stale-index disposition (transient gates; only-poison never gates)
# --------------------------------------------------------------------------- #

def test_9a_transient_backlog_and_finding_exits_2_finding_intact(workspace, monkeypatch, capsys) -> None:
    """A transient backlog + a finding → exit 2 (2 dominates 1), finding still printed."""
    config, store, telemetry = workspace
    _commit(store, "active-q", _ACTIVE_AXIOM)  # NOT drained ⇒ transient backlog row
    _write_decisions(config, ("pending-y", _PENDING_AXIOM))
    _wire_substrate(monkeypatch, {_PENDING_AXIOM: [_match("active-q", 0.9)]})
    _wire_judge(monkeypatch, _finding_judge())

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=False)

    assert code == 2
    out = capsys.readouterr().out
    assert "[Conflict]" in out and "active-q" in out  # finding intact
    assert "behind the vector index" in out
    rows = _read_check_runs(config)
    assert len(rows) == 1
    assert rows[0]["mode"] == "staged"
    assert rows[0]["exit_code"] == 2
    assert "stale_index" in rows[0]["degraded_reason"]


def test_9b_only_poison_backlog_never_gates_exit_1_exclusion_disclosed(workspace, monkeypatch, capsys) -> None:
    """An over-tolerance (poison) backlog is disclosed but never gates → exit 1 on the finding."""
    config, store, telemetry = workspace
    active_id = _commit(store, "active-q", _ACTIVE_AXIOM)
    _drain_outbox(store)
    _poison(store, active_id)  # retry_count ≥ tolerance ⇒ excluded, never gating
    _write_decisions(config, ("pending-y", _PENDING_AXIOM))
    _wire_substrate(monkeypatch, {_PENDING_AXIOM: [_match("active-q", 0.9)]})
    _wire_judge(monkeypatch, _finding_judge())

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=False)

    assert code == 1
    out = capsys.readouterr().out
    assert "[Conflict]" in out
    assert "Coverage exclusions" in out and "active-q" in out
    rows = _read_check_runs(config)
    assert len(rows) == 1 and rows[0]["exit_code"] == 1
    assert rows[0]["coverage_exclusions"] == 1
    assert rows[0]["degraded_reason"] is None  # poison never degrades


# --------------------------------------------------------------------------- #
# 10 — the aggregate breaker (one penalty, not N)
# --------------------------------------------------------------------------- #

def test_10_breaker_trips_on_first_unavailable_partial_exit_2(workspace, monkeypatch, capsys) -> None:
    """Facade Unavailable on entry k → remaining skipped, prior findings partial, exit 2."""
    config, store, telemetry = workspace
    _seed_active(store)
    p1_axiom = "First pending axiom that finds the active corpus decision."
    p2_axiom = "Second pending axiom whose vector query goes dark."
    _write_decisions(config, ("pending-1", p1_axiom), ("pending-2", p2_axiom))
    _wire_substrate(
        monkeypatch,
        {p1_axiom: [_match("active-q", 0.9)], p2_axiom: []},
        vector_raises={p2_axiom: VectorStoreError("qdrant down mid-run")},
    )
    # Only pending-1 ever reaches the judge (pending-2 dies in gather).
    _wire_judge(monkeypatch, _SequenceJudge([
        _execution([("active-q", False, 0.9, "They cannot both stand.")], batch_id="b0"),
    ]))

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=False)

    assert code == 2
    out = capsys.readouterr().out
    assert "[Conflict]" in out  # pending-1's finding survives
    assert "Checked 1 of 2" in out
    rows = _read_check_runs(config)
    assert len(rows) == 1
    assert rows[0]["nodes_swept"] == 1
    # A VECTOR_STORE Unavailable reads as the semantic-substrate `sweep` token.
    assert "sweep" in rows[0]["degraded_reason"]


def test_10b_judge_degradation_reads_as_judgment_token(workspace, monkeypatch, capsys) -> None:
    """A judge-side ``Unavailable`` (vs substrate) trips the breaker as the ``judgment`` token."""
    config, store, telemetry = workspace
    _seed_active(store)
    _write_decisions(config, ("pending-y", _PENDING_AXIOM))
    _wire_substrate(monkeypatch, {_PENDING_AXIOM: [_match("active-q", 0.9)]})
    _wire_judge(monkeypatch, _SequenceJudge([
        Unavailable(reason=ConflictUnavailableReason.JUDGMENT_TIMEOUT, detail="timed out"),
    ]))

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=True)

    assert code == 2
    obj = json.loads(capsys.readouterr().out)
    assert obj["degradations"] == ["judgment"]
    assert obj["nodes_swept"] == 0
    assert _read_check_runs(config) == []  # no batch fired ⇒ no row


# --------------------------------------------------------------------------- #
# 11 — telemetry attribution (surface='check' + one staged run row)
# --------------------------------------------------------------------------- #

def test_11_judged_run_persists_check_surface_and_one_staged_row(workspace, monkeypatch) -> None:
    """A judged staged run writes ``surface='check'`` rows + one ``mode='staged'`` row joined by id."""
    config, store, telemetry = workspace
    _seed_active(store)
    _write_decisions(config, ("pending-y", _PENDING_AXIOM))
    _wire_substrate(monkeypatch, {_PENDING_AXIOM: [_match("active-q", 0.9)]})
    _wire_judge(monkeypatch, _finding_judge())

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=False)

    assert code == 1
    runs = _read_check_runs(config)
    assert len(runs) == 1
    run = runs[0]
    assert run["mode"] == "staged" and run["exit_code"] == 1
    assert run["pairs_reused"] == 0 and run["findings_known"] == 0
    assert run["findings_new"] == 1 and run["mitos_version"] == __version__
    checks = _read_conflict_checks(config)
    assert len(checks) == 1
    assert checks[0]["surface"] == "check"
    # The one-thread-of-truth join: conflict_checks.sync_run_id == the run PK.
    assert checks[0]["sync_run_id"] == run["run_id"]


# --------------------------------------------------------------------------- #
# 12 — lazy telemetry (clean-empty needs no telemetry / no row)
# --------------------------------------------------------------------------- #

def test_12_clean_empty_with_telemetry_none_exits_0_no_row(workspace, monkeypatch) -> None:
    """All-clean-empty pending set with telemetry=None → exit 0, no row (KD7 lazy)."""
    config, store, telemetry = workspace
    _seed_active(store)
    _write_decisions(config, ("pending-y", _PENDING_AXIOM))
    _wire_substrate(monkeypatch, {_PENDING_AXIOM: []})  # no candidate ⇒ no batch fires
    monkeypatch.setattr(cli, "_build_check_telemetry", lambda config: None)
    _wire_judge(monkeypatch, _SequenceJudge([]))  # built (key present), never called

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=False)

    assert code == 0
    assert _read_check_runs(config) == []


def test_12b_summary_write_failure_moves_exit_to_2(workspace, monkeypatch) -> None:
    """A judged run whose ``record_check_run`` fails → exit 2, no persisted row (KD8)."""
    config, store, telemetry = workspace
    _seed_active(store)
    _write_decisions(config, ("pending-y", _PENDING_AXIOM))
    _wire_substrate(monkeypatch, {_PENDING_AXIOM: [_match("active-q", 0.9)]})
    _wire_judge(monkeypatch, _finding_judge())
    monkeypatch.setattr(cli, "_build_check_telemetry",
                        lambda config: _FailingWriteTelemetry(telemetry))

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=False)

    assert code == 2
    assert _read_check_runs(config) == []  # the failed row is discarded, never persisted


# --------------------------------------------------------------------------- #
# 13 — the staged --json shape (§8 / KD9)
# --------------------------------------------------------------------------- #

def test_13_json_shape_staged_snapshot(workspace, monkeypatch, capsys) -> None:
    """The staged ``--json`` object: key set, ``mode:'staged'``, scope absent, novelty:'new'."""
    config, store, telemetry = workspace
    _seed_active(store)
    _write_decisions(config, ("pending-y", _PENDING_AXIOM))
    _wire_substrate(monkeypatch, {_PENDING_AXIOM: [_match("active-q", 0.9)]})
    _wire_judge(monkeypatch, _finding_judge())

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=True)

    assert code == 1
    obj = json.loads(capsys.readouterr().out)
    assert obj["mode"] == "staged"
    assert "scope" not in obj  # staged never scopes
    assert obj["fresh"] is False
    assert obj["pairs_reused"] == 0 and obj["findings_known"] == 0
    assert obj["findings_new"] == 1 and obj["exit_code"] == 1
    assert obj["nodes_total"] == 1 and obj["nodes_swept"] == 1
    assert obj["batches_planned"] == obj["batches_executed"] + obj["batches_skipped"]
    assert obj["summary_row_written"] is True
    finding = obj["findings"][0]
    assert finding["novelty"] == "new"
    assert finding["proposal"]["slug"] == "pending-y"
    assert finding["partner"]["slug"] == "active-q"
    assert finding["partner"]["id"] is not None
    # The cross-surface invariant a CI consumer relies on.
    assert (obj["exit_code"] == 1) == (obj["findings_new"] > 0)


def test_13b_json_no_pending_shape(workspace, monkeypatch, capsys) -> None:
    """A no-pending run still emits exactly one machine object (exit 0, empty findings)."""
    config, store, telemetry = workspace
    _commit(store, "committed-x", "A committed axiom already in the graph.")
    _drain_outbox(store)
    _write_decisions(config, ("committed-x", "A committed axiom already in the graph."))

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=True)

    assert code == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["mode"] == "staged" and obj["exit_code"] == 0
    assert obj["nodes_total"] == 0 and obj["findings"] == []
    assert obj["summary_row_written"] is False


# --------------------------------------------------------------------------- #
# 14 — confirm parity (the same three refusals as corpus, on the pending count)
# --------------------------------------------------------------------------- #

def test_14a_above_threshold_headless_exits_2_zero_spend(workspace, monkeypatch) -> None:
    """Above-threshold pending count, no TTY, no ``--yes`` → exit 2, judge never built."""
    config, store, telemetry = workspace
    _seed_active(store)
    _write_decisions(config, ("pending-y", _PENDING_AXIOM))
    _wire_substrate(monkeypatch, {_PENDING_AXIOM: [_match("active-q", 0.9)]})
    monkeypatch.setattr(check, "CHECK_CONFIRM_BATCHES", 0)  # n=1 > 0 gates
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin(False))
    judge_calls = _wire_judge(monkeypatch, _finding_judge())

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=False)

    assert code == 2
    assert judge_calls == []  # zero spend — confirm precedes the judge build
    assert _read_check_runs(config) == []


def test_14b_above_threshold_json_error_object(workspace, monkeypatch, capsys) -> None:
    """Above-threshold on ``--json`` → a confirmation_required error object, exit 2."""
    config, store, telemetry = workspace
    _seed_active(store)
    _write_decisions(config, ("pending-y", _PENDING_AXIOM))
    _wire_substrate(monkeypatch, {_PENDING_AXIOM: [_match("active-q", 0.9)]})
    monkeypatch.setattr(check, "CHECK_CONFIRM_BATCHES", 0)
    _wire_judge(monkeypatch, _finding_judge())

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=True)

    assert code == 2
    obj = json.loads(capsys.readouterr().out)
    assert obj["code"] == "confirmation_required"


def test_14c_above_threshold_interactive_decline_exits_2(workspace, monkeypatch, capsys) -> None:
    """Above-threshold with a TTY, declined at the prompt → exit 2 'nothing spent'."""
    config, store, telemetry = workspace
    _seed_active(store)
    _write_decisions(config, ("pending-y", _PENDING_AXIOM))
    _wire_substrate(monkeypatch, {_PENDING_AXIOM: [_match("active-q", 0.9)]})
    monkeypatch.setattr(check, "CHECK_CONFIRM_BATCHES", 0)
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin(True))
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    _wire_judge(monkeypatch, _finding_judge())

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=False, as_json=False)

    assert code == 2
    assert "nothing spent" in capsys.readouterr().out.lower()


# --------------------------------------------------------------------------- #
# The staged mapper (KD7) — surface='check', MI-9 coercions, distinct-alias resolve
# --------------------------------------------------------------------------- #

def test_persist_staged_batch_surface_check_and_mi9(workspace) -> None:
    """Direct ``_persist_staged_batch``: ``surface='check'``, MI-9 ``""``/``[]`` → NULL, run id joins.

    The staged sibling of ``test_conflict_sync.test_persist_mi9_empty_maps_to_null_unit`` —
    guards the third duplicated ``ConflictCheckResult`` → rows mapper (the writers' lockstep):
    the two staged-specific values (``surface='check'`` + ``sync_run_id`` = the run id) and the
    load-bearing empty-proposal coercions the parse-driven path can't reach.
    """
    config, store, telemetry = workspace
    result = ConflictCheckResult(
        proposal_input=JudgeInput(axiom="A global pending axiom.", rejected_paths="", scope=[]),
        proposed_hash_if_any="proposal-hash",
        findings=[],
        judged_pairs=[JudgedPair(
            candidate=Candidate(slug="cand", score=0.9, node={"id": "cand-hash"}, state="active"),
            candidate_input=JudgeInput(axiom="Cand axiom.", rejected_paths="", scope=[]),
            judgment=Judgment(slug="cand", rationale="why", tenable_together=False, confidence=0.9),
            surfaced=True,
        )],
        execution=_execution([("cand", False, 0.9, "why")], batch_id="staged-unit-b0"),
    )

    detail = cli._persist_staged_batch(telemetry, result, run_id="staged-run-77")

    assert detail is None
    rows = _read_conflict_checks(config)
    assert len(rows) == 1
    row = rows[0]
    assert row["surface"] == "check"           # the staged difference from the sync mapper
    assert row["sync_run_id"] == "staged-run-77"
    assert row["proposal_rejected_paths"] is None  # "" → NULL (MI-9)
    assert row["proposal_scope"] is None            # [] → NULL (MI-9)
    assert row["candidate_scope"] is None           # [] → NULL (MI-9)
    assert row["candidate_rejected_paths"] == ""     # NOT NULL — degenerate "" verbatim
    assert row["candidate_hash"] == "cand-hash"


def test_persist_staged_batch_none_telemetry_returns_detail(workspace) -> None:
    """A ``None`` telemetry store is a write failure the caller degrades on (never raises)."""
    config, store, telemetry = workspace
    result = ConflictCheckResult(
        proposal_input=JudgeInput(axiom="ax", rejected_paths="r", scope=[]),
        proposed_hash_if_any="h", findings=[],
        judged_pairs=[JudgedPair(
            candidate=Candidate(slug="c", score=0.9, node={"id": "cid"}, state="active"),
            candidate_input=JudgeInput(axiom="cax", rejected_paths="cr", scope=[]),
            judgment=Judgment(slug="c", rationale="w", tenable_together=False, confidence=0.9),
            surfaced=True,
        )],
        execution=_execution([("c", False, 0.9, "w")], batch_id="b"),
    )

    assert cli._persist_staged_batch(None, result, run_id="r") is not None


def test_14d_above_threshold_with_yes_proceeds(workspace, monkeypatch) -> None:
    """``--yes`` waives the confirm even above threshold → the gate runs (exit 1)."""
    config, store, telemetry = workspace
    _seed_active(store)
    _write_decisions(config, ("pending-y", _PENDING_AXIOM))
    _wire_substrate(monkeypatch, {_PENDING_AXIOM: [_match("active-q", 0.9)]})
    monkeypatch.setattr(check, "CHECK_CONFIRM_BATCHES", 0)
    _wire_judge(monkeypatch, _finding_judge())

    code = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                         assume_yes=True, as_json=False)

    assert code == 1
