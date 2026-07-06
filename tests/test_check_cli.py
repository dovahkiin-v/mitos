"""Tests for the ``mitos check`` CLI verb — corpus mode (Phase 3a).

Drives ``cmd_check(config, …)`` directly against a real tmp-workspace
``GraphStore`` + ``TelemetryStore`` (the ``test_check_probe.py`` fixture manner),
with keyed fakes for the external substrate. The load-bearing surface under test:

* the shipped 0/1/2 exit contract (CHK-C2) — the eight index-enumerated exit cases
  (§9 battery), red-first;
* the CHK-D5 spend confirm (KD3) — strictly-above threshold, three refusal surfaces,
  all exit 2, zero spend, no row;
* the no-row-on-refusal/pre-execute rule (KD4) and the run-end seam order (KD5) —
  a completed run leaves exactly one ``check_runs`` row whose ``exit_code`` equals
  the process exit; a refused/failed-before-execute run leaves none;
* the machine-stable ``--json`` object (§8/KD7), through ``_emit_json`` only.

Discipline (PATTERNS + the 2c/2d suites): hand-rolled synchronous fakes, real temp
stores seeded via ``commit_parsed_entry`` (every commit auto-enqueues an Outbox row —
fixtures state their backlog posture explicitly via ``_drain_outbox``), production
pins (``"SONNET"`` + the default ``CONFLICT_PROMPT_VERSION``). Zero LLM, zero live
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
from mitos.conflict import CONFLICT_PROMPT_VERSION
from mitos.errors import DatabaseError, MitosError
from mitos.store import GraphStore, open_connection
from mitos.telemetry import TelemetryStore

from _conflict_helpers import _drain_outbox, _keyed_substrate, _match
from test_check_probe import _canned_judge, _commit, _poison, _seed_verdict

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
    """A temp workspace whose graph + telemetry live at ``config``'s own paths.

    ``cmd_check`` builds its own ``GraphStore(config.db_path)`` /
    ``TelemetryStore(config.telemetry_path)`` — the same files these fixtures seed,
    so a test seeds decisions/verdicts here and the verb reads them back.
    """
    tmpdir = tempfile.mkdtemp()
    config = MitosConfig(tmpdir)
    store = GraphStore(config.db_path)
    telemetry = TelemetryStore(config.telemetry_path)
    yield config, store, telemetry
    shutil.rmtree(tmpdir, ignore_errors=True)


def _pair(store: GraphStore) -> Tuple[str, str, Dict[str, List[Dict[str, Any]]]]:
    """Commits an (a, b) pair whose a-side sweep discovers b; returns ids + neighbourhoods."""
    a_axiom = "Corpus axiom alpha for the check verb."
    b_axiom = "Corpus axiom beta for the check verb."
    a_id = _commit(store, "cli-a", a_axiom)
    b_id = _commit(store, "cli-b", b_axiom)
    neighbourhoods = {a_axiom: [_match("cli-b", 0.9)], b_axiom: []}
    return a_id, b_id, neighbourhoods


def _wire_substrate(
    monkeypatch: pytest.MonkeyPatch,
    neighbourhoods: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Any, Any]:
    """Monkeypatches ``cli._build_check_substrate`` to keyed fakes; returns them."""
    embed, vector = _keyed_substrate(neighbourhoods)
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


def _canned_for(
    store: GraphStore, embed: Any, vector: Any, telemetry: Any, *,
    scope: Optional[str] = None, fresh: bool = False,
    tenable: bool = False, confidence: float = 0.9,
) -> Any:
    """Builds a per-group canned judge from a plan matching ``cmd_check``'s internal one."""
    plan = check.plan_corpus_check(
        store=store, embed_provider=embed, vector_store=vector, telemetry=telemetry,
        model_alias=PRODUCTION_ALIAS, scope=scope, fresh=fresh,
    )
    return _canned_judge(plan, tenable=tenable, confidence=confidence)


def _read_check_runs(config: MitosConfig) -> List[Dict[str, Any]]:
    """Reads every ``check_runs`` row back through a real read-only connection."""
    if not os.path.exists(config.telemetry_path):
        return []
    conn = open_connection(config.telemetry_path, read_only=True)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM check_runs").fetchall()]
    finally:
        conn.close()


class _FakeStdin:
    """A stdin stand-in with a controllable ``isatty`` (test_cutover.py:958 pattern)."""

    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class _FaultStore:
    """Wraps a real store; ``get_pending_embeddings`` raises (the start-probe fault seam)."""

    def __init__(self, inner: GraphStore) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def get_pending_embeddings(self) -> List[Dict[str, Any]]:
        raise sqlite3.OperationalError("provoked probe fault")


class _FailingWriteTelemetry:
    """Wraps a real telemetry store; only ``record_check_run`` raises (KD5 write-fault seam)."""

    def __init__(self, inner: TelemetryStore) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def record_check_run(self, row: Any) -> None:
        raise DatabaseError("provoked summary-row write fault")


# --------------------------------------------------------------------------- #
# The TDD battery (index-enumerated) — §9
# --------------------------------------------------------------------------- #

def test_1_reused_standing_finding_exits_0(workspace, monkeypatch, capsys):
    """A reused surfaced verdict → known finding → exit 0, standing section, no vector block."""
    config, store, telemetry = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    _seed_verdict(telemetry, proposal_hash=a_id, candidate_hash=b_id,
                  tenable=False, confidence=0.9, batch_id="prior-batch",
                  created_at="2026-06-01T00:00:00.000000+00:00")
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    invoked = _wire_judge(monkeypatch, None)  # never built — no fresh groups

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=False)

    assert code == 0
    out = capsys.readouterr().out
    assert "standing (previously reported)" in out
    assert "[Conflict]" not in out
    assert invoked == []  # reuse-only → judge never constructed


def test_2_first_ever_finding_exits_1(workspace, monkeypatch, capsys):
    """A first-ever undeclared contradiction → new finding → exit 1, full vector printed."""
    config, store, telemetry = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    judge = _canned_for(store, embed, vector, telemetry, tenable=False, confidence=0.9)
    _wire_judge(monkeypatch, judge)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=False)

    assert code == 1
    out = capsys.readouterr().out
    assert "[Conflict]" in out
    assert "cli-a" in out and "cli-b" in out
    assert "Resolve by declaring a relationship in decisions.md" in out


def test_3_fresh_reconfirmation_of_standing_exits_0(workspace, monkeypatch, capsys):
    """``--fresh`` re-judge of a standing (previously surfaced) finding stays known → exit 0."""
    config, store, telemetry = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    _seed_verdict(telemetry, proposal_hash=a_id, candidate_hash=b_id,
                  tenable=False, confidence=0.9, batch_id="prior-batch",
                  created_at="2026-06-01T00:00:00.000000+00:00")
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    judge = _canned_for(store, embed, vector, telemetry, fresh=True, tenable=False, confidence=0.9)
    _wire_judge(monkeypatch, judge)

    code = cli.cmd_check(config, scope=None, fresh=True, assume_yes=False, as_json=False)

    assert code == 0
    assert "standing (previously reported)" in capsys.readouterr().out


def test_4_fresh_flip_of_tenable_pair_exits_1(workspace, monkeypatch, capsys):
    """``--fresh`` re-judge flipping a previously-tenable pair to a finding → new → exit 1."""
    config, store, telemetry = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    _seed_verdict(telemetry, proposal_hash=a_id, candidate_hash=b_id,
                  tenable=True, confidence=0.9, batch_id="prior-tenable",
                  created_at="2026-06-01T00:00:00.000000+00:00")
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    judge = _canned_for(store, embed, vector, telemetry, fresh=True, tenable=False, confidence=0.9)
    _wire_judge(monkeypatch, judge)

    code = cli.cmd_check(config, scope=None, fresh=True, assume_yes=False, as_json=False)

    assert code == 1
    assert "[Conflict]" in capsys.readouterr().out


def test_5_degraded_plus_findings_exits_2(workspace, monkeypatch, capsys):
    """A transient backlog (undrained) + a new finding → exit 2 (2 dominates 1); findings still shown."""
    config, store, telemetry = workspace
    a_id, b_id, nbhds = _pair(store)
    # Deliberately NOT drained → the commit's transient Outbox rows gate stale_index.
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    judge = _canned_for(store, embed, vector, telemetry, tenable=False, confidence=0.9)
    _wire_judge(monkeypatch, judge)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=True)

    assert code == 2
    obj = json.loads(capsys.readouterr().out)
    assert "stale_index" in obj["degradations"]
    assert len(obj["findings"]) == 1  # the finding rides at exit 2


def test_6_headless_above_threshold_without_yes_exits_2_zero_llm(workspace, monkeypatch, capsys):
    """Headless above-threshold without ``--yes`` → exit 2, judge never built, no row."""
    config, store, telemetry = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    _wire_substrate(monkeypatch, nbhds)
    invoked = _wire_judge(monkeypatch, None)
    monkeypatch.setattr(check, "CHECK_CONFIRM_BATCHES", 0)  # any fresh group is "above"
    monkeypatch.setattr("sys.stdin", _FakeStdin(tty=False))

    def _no_input(prompt: str = "") -> str:
        raise AssertionError("input() must never be called headless")

    monkeypatch.setattr("builtins.input", _no_input)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=False)

    assert code == 2
    assert invoked == []  # the judge builder is never reached — zero spend
    assert _read_check_runs(config) == []  # refusal writes no row (KD4)


def test_7_json_shape_snapshot(workspace, monkeypatch, capsys):
    """``--json`` emits exactly the §8 key set; per-finding carries the pinned shape."""
    config, store, telemetry = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    judge = _canned_for(store, embed, vector, telemetry, tenable=False, confidence=0.9)
    _wire_judge(monkeypatch, judge)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=True)

    assert code == 1
    obj = json.loads(capsys.readouterr().out)
    assert set(obj.keys()) == {
        "run_id", "mode", "exit_code", "started_at", "ended_at", "fresh",
        "nodes_total", "nodes_swept", "pairs_judged_fresh", "pairs_reused",
        "batches_planned", "batches_executed", "batches_skipped", "findings",
        "findings_new", "findings_known", "degradations", "coverage_exclusions",
        "index_backlog_transient", "summary_row_written",
    }
    assert "scope" not in obj  # MI-9 — absent, never ""
    assert obj["mode"] == "corpus" and obj["exit_code"] == 1
    finding = obj["findings"][0]
    assert set(finding.keys()) == {
        "novelty", "confidence", "rationale", "score", "reused",
        "source_batch_id", "source_created_at", "proposal", "partner",
    }
    for side in ("proposal", "partner"):
        assert set(finding[side].keys()) >= {"id", "slug", "axiom", "scope", "rejected_paths"}
    assert finding["novelty"] == "new"


def test_8_no_tty_below_threshold_runs_to_completion(workspace, monkeypatch, capsys):
    """No-TTY, below threshold → never prompts, runs to completion (exit per findings)."""
    config, store, telemetry = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    judge = _canned_for(store, embed, vector, telemetry, tenable=False, confidence=0.9)
    _wire_judge(monkeypatch, judge)
    monkeypatch.setattr("sys.stdin", _FakeStdin(tty=False))

    def _no_input(prompt: str = "") -> str:
        raise AssertionError("input() must never be called below threshold headless")

    monkeypatch.setattr("builtins.input", _no_input)

    # default CHECK_CONFIRM_BATCHES (10) ≥ n (1) → no gate.
    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=False)

    assert code == 1


# --------------------------------------------------------------------------- #
# Beyond the battery — each maps to a §4 decision or a W-flip
# --------------------------------------------------------------------------- #

def test_interactive_confirm_accept_spends(workspace, monkeypatch, capsys):
    """A TTY 'y' at the confirm authorizes the spend and the run completes."""
    config, store, telemetry = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    judge = _canned_for(store, embed, vector, telemetry, tenable=False, confidence=0.9)
    _wire_judge(monkeypatch, judge)
    monkeypatch.setattr(check, "CHECK_CONFIRM_BATCHES", 0)
    monkeypatch.setattr("sys.stdin", _FakeStdin(tty=True))
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=False)

    assert code == 1
    assert len(_read_check_runs(config)) == 1  # a completed run writes exactly one row


def test_interactive_confirm_decline_exits_2_no_row(workspace, monkeypatch, capsys):
    """A TTY 'n' declines → exit 2, 'nothing spent', no row (KD3 — not cutover's exit 1)."""
    config, store, telemetry = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    invoked = _wire_judge(monkeypatch, None)
    monkeypatch.setattr(check, "CHECK_CONFIRM_BATCHES", 0)
    monkeypatch.setattr("sys.stdin", _FakeStdin(tty=True))
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=False)

    assert code == 2
    assert "Aborted — nothing spent." in capsys.readouterr().out
    assert invoked == []
    assert _read_check_runs(config) == []


def test_json_above_threshold_refusal_object(workspace, monkeypatch, capsys):
    """``--json`` above threshold without ``--yes`` → the refusal object, exit 2, no row."""
    config, store, telemetry = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, None)
    monkeypatch.setattr(check, "CHECK_CONFIRM_BATCHES", 0)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=True)

    assert code == 2
    obj = json.loads(capsys.readouterr().out)
    assert obj["code"] == "confirmation_required"
    assert obj["batches_planned"] == 1
    assert _read_check_runs(config) == []


def test_empty_corpus_exits_0_row_written(workspace, monkeypatch, capsys):
    """An empty corpus → the quiet healthy-empty line, exit 0, row written."""
    config, store, telemetry = workspace
    _wire_substrate(monkeypatch, {})
    _wire_judge(monkeypatch, None)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=False)

    assert code == 0
    assert "corpus is empty" in capsys.readouterr().out
    rows = _read_check_runs(config)
    assert len(rows) == 1 and rows[0]["exit_code"] == 0 and rows[0]["nodes_swept"] == 0


def test_empty_corpus_both_providers_none_exits_0(workspace, monkeypatch, capsys):
    """KD2 structural pin: empty corpus + both providers None → same healthy path, exit 0."""
    config, store, telemetry = workspace
    # Real substrate construction (offline) → (None, None); empty snapshot never
    # touches the providers, so the run stays on the one engine path.
    monkeypatch.setattr(cli, "_build_check_judge", lambda: None)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=False)

    assert code == 0
    assert len(_read_check_runs(config)) == 1


def test_zero_match_scope_exits_0(workspace, monkeypatch, capsys):
    """A ``--scope`` matching no live decision → the '0 of N' line, exit 0."""
    config, store, telemetry = workspace
    _commit(store, "scoped-a", "A scoped axiom.", scope=["alpha"])
    _commit(store, "scoped-b", "Another scoped axiom.", scope=["alpha"])
    _drain_outbox(store)
    _wire_substrate(monkeypatch, {})
    _wire_judge(monkeypatch, None)

    code = cli.cmd_check(config, scope="nonesuch", fresh=False, assume_yes=False, as_json=False)

    assert code == 0
    out = capsys.readouterr().out
    assert "0 of 2 live decisions match scope 'nonesuch'" in out


def test_providers_none_nonempty_exits_2_no_row(workspace, monkeypatch, capsys):
    """Providers None + a non-empty corpus → exit 2, message names the component, no row."""
    config, store, telemetry = workspace
    _commit(store, "solo", "A live decision that cannot be audited.")
    _drain_outbox(store)
    # No substrate monkeypatch — real construction is (None, None) offline.

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=False)

    assert code == 2
    err = capsys.readouterr().err
    assert "cannot audit 1 live decision" in err
    assert "embeddings" in err or "vector store" in err
    assert _read_check_runs(config) == []


def test_provoked_store_fault_exits_2_no_traceback(workspace, monkeypatch, capsys):
    """A store fault at the start probe → calm exit 2, no traceback (KD1a)."""
    config, store, telemetry = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    _wire_substrate(monkeypatch, nbhds)  # providers present → we reach plan
    _wire_judge(monkeypatch, None)
    monkeypatch.setattr(cli, "GraphStore", lambda path: _FaultStore(GraphStore(path)))

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=False)

    captured = capsys.readouterr()
    assert code == 2
    assert "check could not run" in captured.err
    assert "Traceback" not in captured.err and "Traceback" not in captured.out
    assert _read_check_runs(config) == []


def test_shared_boundary_conditional_check_vs_list(monkeypatch, capsys):
    """A bad ``-C`` maps to exit 2 under ``check`` but stays exit 1 under ``list`` (KD1)."""
    missing = os.path.join(tempfile.gettempdir(), "definitely-not-a-mitos-dir-xyz")

    monkeypatch.setattr("sys.argv", ["mitos", "-C", missing, "check"])
    with pytest.raises(SystemExit) as exc_check:
        cli.main()
    assert exc_check.value.code == 2

    monkeypatch.setattr("sys.argv", ["mitos", "-C", missing, "list"])
    with pytest.raises(SystemExit) as exc_list:
        cli.main()
    assert exc_list.value.code == 1


def test_run_end_seam_row_equals_report(workspace, monkeypatch, capsys):
    """W7 evidence: the persisted row's scalars equal the JSON, and its exit_code the return."""
    config, store, telemetry = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    judge = _canned_for(store, embed, vector, telemetry, tenable=False, confidence=0.9)
    _wire_judge(monkeypatch, judge)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=True)
    obj = json.loads(capsys.readouterr().out)

    rows = _read_check_runs(config)
    assert len(rows) == 1
    row = rows[0]
    assert row["exit_code"] == code == 1
    assert row["run_id"] == obj["run_id"]
    assert row["nodes_swept"] == obj["nodes_swept"]
    assert row["pairs_judged_fresh"] == obj["pairs_judged_fresh"]
    assert row["pairs_reused"] == obj["pairs_reused"]
    assert row["findings_new"] == obj["findings_new"] == 1
    assert row["findings_known"] == obj["findings_known"] == 0
    assert obj["summary_row_written"] is True
    assert row["mitos_version"] == __version__


def test_telemetry_none_unpartitioned_exit_2(workspace, monkeypatch, capsys):
    """Telemetry None → unpartitioned findings, exit 2, summary_row_written false, no row."""
    config, store, telemetry = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    # With telemetry None the reuse index is unavailable → all pairs are fresh.
    judge = _canned_for(store, embed, vector, None, tenable=False, confidence=0.9)
    _wire_judge(monkeypatch, judge)
    monkeypatch.setattr(cli, "_build_check_telemetry", lambda config: None)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=True)

    assert code == 2
    obj = json.loads(capsys.readouterr().out)
    assert "reuse_read" in obj["degradations"]
    assert obj["findings_new"] is None and obj["findings_known"] is None
    assert obj["findings"][0]["novelty"] is None
    assert obj["summary_row_written"] is False
    assert _read_check_runs(config) == []


def test_record_check_run_failure_exits_2_no_row(workspace, monkeypatch, capsys):
    """A summary-row write failure → exit 2, disclosure present, no row (KD5 last fallible act)."""
    config, store, telemetry = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    judge = _canned_for(store, embed, vector, telemetry, tenable=False, confidence=0.9)
    _wire_judge(monkeypatch, judge)
    monkeypatch.setattr(cli, "_build_check_telemetry",
                        lambda config: _FailingWriteTelemetry(TelemetryStore(config.telemetry_path)))

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=True)

    assert code == 2
    obj = json.loads(capsys.readouterr().out)
    assert obj["summary_row_written"] is False
    assert _read_check_runs(config) == []


def test_poison_exclusion_disclosed_exits_0(workspace, monkeypatch, capsys):
    """W6: a poison (chronically un-embedded) node never gates → exit 0, disclosed by slug."""
    config, store, telemetry = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)          # clear the transient backlog first
    _poison(store, a_id)          # re-add a_id at the retry tolerance → a poison row
    embed, vector = _wire_substrate(monkeypatch, nbhds)
    judge = _canned_for(store, embed, vector, telemetry, tenable=True, confidence=0.9)
    _wire_judge(monkeypatch, judge)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=True)

    assert code == 0  # poison never gates; the tenable verdict yields no finding
    obj = json.loads(capsys.readouterr().out)
    assert "stale_index" not in obj["degradations"]  # poison is excluded, not transient
    exclusions = {e["id"]: e["slug"] for e in obj["coverage_exclusions"]}
    assert a_id in exclusions and exclusions[a_id] == "cli-a"  # slug resolved live (MI-2)


def test_keyless_reuse_only_exits_0(workspace, monkeypatch, capsys):
    """P14: a reuse-only run needs no key — judge never built, exit 0, one row."""
    config, store, telemetry = workspace
    a_id, b_id, nbhds = _pair(store)
    _drain_outbox(store)
    _seed_verdict(telemetry, proposal_hash=a_id, candidate_hash=b_id,
                  tenable=True, confidence=0.9, batch_id="prior-tenable",
                  created_at="2026-06-01T00:00:00.000000+00:00")
    _wire_substrate(monkeypatch, nbhds)
    invoked = _wire_judge(monkeypatch, None)

    code = cli.cmd_check(config, scope=None, fresh=False, assume_yes=False, as_json=False)

    assert code == 0  # the reused prior was tenable → silence, not a finding
    assert invoked == []  # no fresh groups → no client (no ANTHROPIC key needed)
    assert len(_read_check_runs(config)) == 1
