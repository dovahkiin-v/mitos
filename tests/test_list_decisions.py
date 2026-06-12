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
