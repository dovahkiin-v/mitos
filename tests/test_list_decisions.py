"""Tests for the exhaustive enumeration path (item ①): the `list_decisions` MCP
tool, the `mitos list` CLI twin (`--json`, live-set semantics, alias), the
store-level `get_decisions`, and the "use the exhaustive path" note that
`surface` now appends.

Driven by loop-Claude's AX friction: semantic recall (surface/query) is ranked
and capped, so it can't answer "have I seen *everything*?". These assert the
complete-set path is unbounded, deterministic, and reachable from every surface.

Forced fully offline (unreachable Qdrant + no keys) so they exercise the pure
graph read and never depend on the machine's running services.
"""

import json
import shutil
import sys
import tempfile
from typing import Iterator, Tuple

import pytest
from unittest.mock import patch

from mitos.config import MitosConfig
from mitos.cli import cmd_init, cmd_list, main
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager


@pytest.fixture
def offline(monkeypatch):
    """Forces degraded graph-only mode: unreachable Qdrant, no embedding keys."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")  # nothing listens here
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def ws(offline) -> Iterator[Tuple[MitosConfig, MitosSyncManager]]:
    """An initialised temp workspace + a manager, in offline graph-only mode."""
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    cmd_init(config)
    yield config, MitosSyncManager(config)
    shutil.rmtree(tmp, ignore_errors=True)


def _record(m: MitosSyncManager, slug: str, scope, supersedes=None) -> None:
    res = m.record_decision_entry(
        axiom=f"Axiom for {slug}.",
        rejected_paths=f"Rejected alternative for {slug}.",
        scope=scope,
        slug=slug,
        supersedes=supersedes,
    )
    assert "error" not in res, res


# --------------------------------------------------------------------------- #
# store.get_decisions — the single source of truth
# --------------------------------------------------------------------------- #

def test_get_decisions_is_unbounded(ws) -> None:
    """Returns EVERY matching decision — no top-k cap (the recall-cliff fix)."""
    config, m = ws
    for i in range(7):  # > surface's hard limit of 5
        _record(m, f"bulk-{i}", scope=["bulk"])
    decisions = GraphStore(config.db_path).get_decisions(scope="bulk")
    assert len(decisions) == 7
    assert {d["slug"] for d in decisions} == {f"bulk-{i}" for i in range(7)}


def test_get_decisions_scope_filter(ws) -> None:
    """Scope filter returns only decisions tagged with that scope."""
    config, m = ws
    _record(m, "auth-one", scope=["auth"])
    _record(m, "db-one", scope=["db"])
    store = GraphStore(config.db_path)
    assert {d["slug"] for d in store.get_decisions(scope="auth")} == {"auth-one"}
    assert {d["slug"] for d in store.get_decisions()} == {"auth-one", "db-one"}


def test_get_decisions_state_filter(ws) -> None:
    """'active' excludes superseded; 'all' includes it; exact state isolates it."""
    config, m = ws
    _record(m, "old-call", scope=["x"])
    _record(m, "new-call", scope=["x"], supersedes="old-call")
    store = GraphStore(config.db_path)

    active = {d["slug"] for d in store.get_decisions(state="active")}
    assert active == {"new-call"}  # superseded one is gone

    every = {d["slug"] for d in store.get_decisions(state="all")}
    assert every == {"old-call", "new-call"}

    superseded = {d["slug"] for d in store.get_decisions(state="superseded")}
    assert superseded == {"old-call"}


# --------------------------------------------------------------------------- #
# MCP list_decisions tool
# --------------------------------------------------------------------------- #

def test_mcp_list_decisions_complete_set(ws) -> None:
    """The MCP tool returns the full set with the documented shape, unbounded."""
    from mitos import mcp_server
    config, m = ws
    for i in range(6):
        _record(m, f"item-{i}", scope=["zone"])

    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.list_decisions(scope="zone"))

    assert resp["total"] == 6
    assert len(resp["decisions"]) == 6
    assert resp["scope"] == "zone"
    assert resp["state"] == "active"
    # Letter-mode shape per decision, plus computed state.
    sample = resp["decisions"][0]
    assert set(sample) == {"slug", "axiom", "rejected_paths", "scope", "state"}
    assert "transcript" not in sample and "core_axiom" not in sample


def test_mcp_list_decisions_registered() -> None:
    """list_decisions is a registered MCP tool alongside surface/query/record."""
    import asyncio
    from mitos.mcp_server import mcp
    names = [t.name for t in asyncio.run(mcp.list_tools())]
    assert "list_decisions" in names


def test_mcp_list_decisions_state_all_includes_superseded(ws) -> None:
    """state='all' surfaces superseded decisions the default 'active' hides."""
    from mitos import mcp_server
    config, m = ws
    _record(m, "v1", scope=["api"])
    _record(m, "v2", scope=["api"], supersedes="v1")
    store = GraphStore(config.db_path, read_only=True)

    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        active = json.loads(mcp_server.list_decisions(scope="api"))
        every = json.loads(mcp_server.list_decisions(scope="api", state="all"))

    assert {d["slug"] for d in active["decisions"]} == {"v2"}
    assert {d["slug"] for d in every["decisions"]} == {"v1", "v2"}


# --------------------------------------------------------------------------- #
# CLI: cmd_list --json, alias, surface note
# --------------------------------------------------------------------------- #

def test_cmd_list_json_shape(ws, capsys) -> None:
    """`mitos list --json` emits the same structured shape as the MCP tool."""
    config, m = ws
    for i in range(3):
        _record(m, f"c-{i}", scope=["cli"])
    capsys.readouterr()  # drain the init banner
    cmd_list(config, scope="cli", as_json=True)
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 3
    assert {d["slug"] for d in out["decisions"]} == {"c-0", "c-1", "c-2"}
    assert out["state"] == "active"


def test_cmd_list_empty_graph_message(ws, capsys) -> None:
    """An empty graph still reports 'empty' (pinned by the pathologies suite)."""
    config, _ = ws
    capsys.readouterr()
    cmd_list(config)
    assert "empty" in capsys.readouterr().out.lower()


@patch("mitos.cli.cmd_list")
def test_list_decisions_alias_routes(mock_list, monkeypatch) -> None:
    """The MCP-name alias `list_decisions` routes to cmd_list with --json plumbed."""
    monkeypatch.setattr(sys, "argv", ["mitos", "list_decisions", "--scope", "api", "--json"])
    main()
    mock_list.assert_called_once()
    _, kwargs = mock_list.call_args
    assert kwargs["scope"] == "api"
    assert kwargs["as_json"] is True


def test_surface_note_points_to_exhaustive_path(ws) -> None:
    """surface_decisions appends a note steering to list_decisions when it has hits."""
    from mitos import mcp_server
    config, m = ws
    _record(m, "settled", scope=["db"])
    store = GraphStore(config.db_path, read_only=True)
    # No embed/vector → falls back to scope pre-filter, which populates results.
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.surface_decisions(query="anything", scope="db"))
    assert resp["active_decisions"]
    assert "note" in resp and "list_decisions" in resp["note"]


# --------------------------------------------------------------------------- #
# T8: absent-from-live scope hard-filter recovery (3d)
#
# A scoped read (`list`/`open-questions`/`list_decisions`) that returns empty
# because the scope tag isn't in the live vocabulary surfaces a bounded
# self-correction vector (text) / an in-band exit-0 signal (scope_known:false +
# scope_recovery under --json/MCP) — never a silent empty, never a hard error.
# A live-but-empty read stays clean honest-empty. (recovery payload carries no
# node id → nothing to modifier-stamp.)
# --------------------------------------------------------------------------- #

from mitos.parser import ParsedEntry
from mitos.recall import scope_filter_recovery
from mitos.cli import cmd_open_questions


def _commit_oq_scope(store: GraphStore, slug: str, scope) -> None:
    """Commits a parked open question in a scope (makes the scope live-via-OQ)."""
    e = ParsedEntry("open_question", slug, 1, 5)
    e.topic = f"Topic for {slug}"
    e.questions_raised = [f"What about {slug}?"]
    e.scope = list(scope)
    store.commit_parsed_entry(e)


# ----- pure-function unit tests (the fast lane) ----------------------------- #

def test_scope_filter_recovery_absent_scope_returns_note() -> None:
    """An absent-from-live scope yields a {'note': ...} payload with the vector."""
    counts = {"auth": {"active_decisions": 3, "parked_open_questions": 0}}
    rec = scope_filter_recovery(scope="ath", scope_counts=counts, surface="cli")
    assert rec is not None
    note = rec["note"]
    assert "unused scope tag" in note
    assert "'auth'" in note  # did-you-mean
    assert "--state all" in note  # the hard-filter affordance
    assert not note.endswith(" ")  # rstripped at the recovery boundary


def test_scope_filter_recovery_live_scope_returns_none() -> None:
    """A live scope recovers nothing — it is real data, not a typo."""
    counts = {"auth": {"active_decisions": 3, "parked_open_questions": 0}}
    assert scope_filter_recovery(scope="auth", scope_counts=counts, surface="cli") is None


def test_scope_filter_recovery_none_scope_and_none_counts_return_none() -> None:
    """No scope, or uncheckable counts, degrade calmly to None (no fabricated hint)."""
    counts = {"auth": {"active_decisions": 1, "parked_open_questions": 0}}
    assert scope_filter_recovery(scope=None, scope_counts=counts, surface="cli") is None
    assert scope_filter_recovery(scope="ath", scope_counts=None, surface="cli") is None


def test_scope_filter_recovery_surface_wording_differs() -> None:
    """MCP pointers use tool call-forms; CLI pointers use shell commands (T7)."""
    counts = {"auth": {"active_decisions": 1, "parked_open_questions": 0}}
    cli = scope_filter_recovery(scope="ath", scope_counts=counts, surface="cli")["note"]
    mcp = scope_filter_recovery(scope="ath", scope_counts=counts, surface="mcp")["note"]
    assert "mitos list --scope 'ath' --state all" in cli
    assert "list_decisions(" not in cli  # the MCP-tool-leak invariant
    assert "list_decisions(scope='ath', state='all')" in mcp


# ----- MCP list_decisions in-band signal + CLI⇄MCP parity ------------------- #

def test_mcp_list_decisions_absent_scope_in_band_signal(ws) -> None:
    """list_decisions(scope=absent) → scope_known:false + scope_recovery, exit-0."""
    from mitos import mcp_server
    config, m = ws
    _record(m, "a", scope=["auth"])
    _record(m, "d", scope=["db"])
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.list_decisions(scope="ath"))
    assert resp["decisions"] == [] and resp["open_questions"] == []
    assert resp["scope_known"] is False
    rec = resp["scope_recovery"]
    assert "unused scope tag" in rec and "'auth'" in rec
    # MCP-worded pointers, never the CLI `mitos` shell forms.
    assert "list_decisions(scope='ath', state='all')" in rec
    assert "mitos list" not in rec


def test_mcp_list_decisions_live_scope_envelope_unchanged(ws) -> None:
    """A live scope returns the existing envelope — no additive fields (truly additive)."""
    from mitos import mcp_server
    config, m = ws
    _record(m, "a", scope=["auth"])
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.list_decisions(scope="auth"))
    assert resp["decisions"]
    assert "scope_known" not in resp and "scope_recovery" not in resp


def test_cli_mcp_fire_same_signal(ws, capsys) -> None:
    """cmd_list --json and list_decisions fire the same in-band signal, surface-worded."""
    from mitos import mcp_server
    config, m = ws
    _record(m, "a", scope=["auth"])
    capsys.readouterr()
    cmd_list(config, scope="ath", as_json=True)
    cli = json.loads(capsys.readouterr().out)
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        mcp = json.loads(mcp_server.list_decisions(scope="ath"))
    assert cli["scope_known"] is False and mcp["scope_known"] is False
    # Both name the same did-you-mean; the call-form differs by surface.
    assert "'auth'" in cli["scope_recovery"] and "'auth'" in mcp["scope_recovery"]
    assert "mitos list --scope 'ath' --state all" in cli["scope_recovery"]
    assert "list_decisions(scope='ath', state='all')" in mcp["scope_recovery"]


# ----- CLI text + json fire the vector -------------------------------------- #

def test_cmd_list_absent_scope_text_vector(ws, capsys) -> None:
    """`mitos list --scope <absent>` prints the bounded vector, exit-0 (no raise)."""
    config, m = ws
    _record(m, "a", scope=["auth"])
    _record(m, "b", scope=["db"])
    capsys.readouterr()
    cmd_list(config, scope="ath")  # must not raise
    out = capsys.readouterr().out
    assert "unused scope tag" in out
    assert "'auth'" in out  # did-you-mean
    assert "Live scopes (busiest first)" in out
    assert "--state all" in out
    assert "No decisions match the given filters." not in out


def test_cmd_list_absent_scope_json_in_band(ws, capsys) -> None:
    """`mitos list --scope <absent> --json` carries the additive fields."""
    config, m = ws
    _record(m, "a", scope=["auth"])
    capsys.readouterr()
    cmd_list(config, scope="ath", as_json=True)
    out = json.loads(capsys.readouterr().out)
    assert out["decisions"] == [] and out["open_questions"] == []
    assert out["scope_known"] is False
    assert "unused scope tag" in out["scope_recovery"]


def test_cmd_open_questions_absent_scope_text_and_json(ws, capsys) -> None:
    """`mitos open-questions --scope <absent>` fires the vector in both modes."""
    config, m = ws
    _record(m, "a", scope=["auth"])
    capsys.readouterr()
    cmd_open_questions(config, scope="ath")
    text = capsys.readouterr().out
    assert "unused scope tag" in text and "'auth'" in text
    assert "Zero parked open questions found." not in text

    cmd_open_questions(config, scope="ath", as_json=True)
    js = json.loads(capsys.readouterr().out)
    assert js["scope_known"] is False
    assert "unused scope tag" in js["scope_recovery"]


# ----- negative pins: live-but-empty stays honest-empty --------------------- #

def test_live_via_parked_oq_is_not_unused(ws, capsys) -> None:
    """A scope live ONLY via a parked OQ is a real tag — `list` returns no decisions
    for it, but it must NOT trip the unused-scope vector."""
    config, m = ws
    store = GraphStore(config.db_path)
    _commit_oq_scope(store, "q-foo", scope=["foo"])
    capsys.readouterr()
    # list --scope foo returns no *decisions*, but foo is in the live map.
    cmd_list(config, scope="foo")
    out = capsys.readouterr().out
    assert "unused scope tag" not in out
    # json path: no additive fields either.
    cmd_list(config, scope="foo", as_json=True)
    js = json.loads(capsys.readouterr().out)
    assert "scope_known" not in js


def test_open_questions_live_scope_no_parked_oq_is_honest_empty(ws, capsys) -> None:
    """A scope live via a decision but with no parked OQ → honest-empty, no vector.

    Proves the gate keys on live-membership, not on emptiness alone."""
    config, m = ws
    _record(m, "a", scope=["auth"])  # auth is live (decision), but has no parked OQ
    capsys.readouterr()
    cmd_open_questions(config, scope="auth")
    out = capsys.readouterr().out
    assert "Zero parked open questions found." in out
    assert "unused scope tag" not in out
    cmd_open_questions(config, scope="auth", as_json=True)
    js = json.loads(capsys.readouterr().out)
    assert "scope_known" not in js


# ----- empty-graph precedence + --state all dead-scope recovery ------------- #

def test_cmd_list_empty_graph_precedence_over_vector(ws, capsys) -> None:
    """A fresh graph yields the 'empty, run sync' line, NOT the unused-scope vector."""
    config, _ = ws
    capsys.readouterr()
    cmd_list(config, scope="anything")
    out = capsys.readouterr().out
    assert "Graph database is empty. Run 'mitos sync' to ingest entries." in out
    assert "unused scope tag" not in out


def test_state_all_dead_scope_pair(ws, capsys) -> None:
    """A superseded-only scope: default `active` is empty → vector with `--state all`
    pointer; `--state all` returns the superseded decisions → no recovery (non-empty)."""
    config, m = ws
    _record(m, "v1", scope=["legacy"])
    _record(m, "v2", scope=["legacy"], supersedes="v1")
    # v2 also moves out so the scope has NO active decisions left.
    _record(m, "v3", scope=["other"], supersedes="v2")
    capsys.readouterr()

    # default active view → empty → vector pointing at --state all.
    cmd_list(config, scope="legacy")
    out = capsys.readouterr().out
    assert "unused scope tag" in out
    assert "--state all" in out

    # --state all → superseded decisions surface → honest non-empty, no recovery.
    cmd_list(config, scope="legacy", state_filter="all", as_json=True)
    js = json.loads(capsys.readouterr().out)
    assert js["decisions"]  # superseded nodes present
    assert "scope_known" not in js
