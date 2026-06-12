"""Tests for round-2 AX fixes: read-payload economy + write-side polish.

From loop-Claude's second round of feedback:
- #1a brevity knob — surface/list can omit the heavy `rejected_paths` for a quick scan.
- #1b session dedup — the MCP server marks already-surfaced decisions `seen` and stops
  re-paying their `rejected_paths` within a session (one persistent serve process).
- #5b record returns the path to the markdown it wrote (so the agent can eyeball it).
- #2 residual — a truncated auto-slug nudges for an explicit `slug=`.
- #5a — `open_questions` is omitted when no scope was given (absent = not scanned).

Offline (unreachable Qdrant + no keys) so behaviour is deterministic; the live brief
path against real embeddings is in test_integration_live.py.
"""

import json
import shutil
import tempfile
from typing import Iterator, Tuple

import pytest
from unittest.mock import patch

from mitos import mcp_server
from mitos.config import MitosConfig
from mitos.cli import cmd_init, cmd_surface, cmd_list, cmd_record
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def ws(offline) -> Iterator[Tuple[MitosConfig, MitosSyncManager]]:
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    cmd_init(config)
    yield config, MitosSyncManager(config)
    shutil.rmtree(tmp, ignore_errors=True)


def _ro_store(config):
    return GraphStore(config.db_path, read_only=True)


# --------------------------------------------------------------------------- #
# #1a Brevity knob — surface / list
# --------------------------------------------------------------------------- #

def test_mcp_surface_brief_omits_rejected_paths(ws):
    """surface(brief=True) drops rejected_paths; default keeps it (the killer field)."""
    config, m = ws
    m.record_decision_entry("Use SQLite WAL.", "Postgres too heavy.", ["db"], slug="sqlite-wal")
    store = _ro_store(config)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        full = json.loads(mcp_server.surface_decisions("storage", scope="db"))
        brief = json.loads(mcp_server.surface_decisions("storage", scope="db", brief=True))
    assert "rejected_paths" in full["active_decisions"][0]
    assert "rejected_paths" not in brief["active_decisions"][0]
    assert brief["active_decisions"][0]["axiom"] == "Use SQLite WAL."  # axiom stays


def test_mcp_list_brief_omits_rejected_paths(ws):
    """list_decisions(brief=True) drops rejected_paths across the whole set."""
    config, m = ws
    for i in range(3):
        m.record_decision_entry(f"Axiom {i}.", "rej", ["z"], slug=f"z-{i}")
    store = _ro_store(config)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        brief = json.loads(mcp_server.list_decisions(scope="z", brief=True))
    assert brief["total"] == 3
    assert all("rejected_paths" not in d for d in brief["decisions"])
    assert all(d["axiom"] for d in brief["decisions"])


def test_cli_surface_brief_json(ws, capsys):
    """`mitos surface --brief` (CLI) omits rejected_paths in JSON output."""
    config, m = ws
    m.record_decision_entry("Adopt hexagonal arch.", "Layered leaks IO.", ["arch"], slug="hex")
    capsys.readouterr()
    cmd_surface(config, "architecture", scope="arch", as_json=True, brief=True)
    out = json.loads(capsys.readouterr().out)
    assert out["active_decisions"]
    assert "rejected_paths" not in out["active_decisions"][0]


def test_cli_list_brief_json(ws, capsys):
    """`mitos list --brief` (CLI) omits rejected_paths in JSON output."""
    config, m = ws
    m.record_decision_entry("Some call.", "rej", ["k"], slug="k-1")
    capsys.readouterr()
    cmd_list(config, scope="k", as_json=True, brief=True)
    out = json.loads(capsys.readouterr().out)
    assert "rejected_paths" not in out["decisions"][0]


# --------------------------------------------------------------------------- #
# #1b Session dedup (MCP-only)
# --------------------------------------------------------------------------- #

def test_mcp_surface_dedup_marks_seen_within_session(ws):
    """A precedent surfaced twice in a session comes back `seen`, without re-paying."""
    config, m = ws
    m.record_decision_entry("Single PSP is Stripe.", "Adyen heavier.", ["pay"], slug="stripe-psp")
    store = _ro_store(config)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        first = json.loads(mcp_server.surface_decisions("payments", scope="pay"))
        second = json.loads(mcp_server.surface_decisions("payments", scope="pay"))
    d1 = first["active_decisions"][0]
    d2 = second["active_decisions"][0]
    assert "rejected_paths" in d1 and "seen" not in d1      # first sight: full
    assert d2.get("seen") is True and "rejected_paths" not in d2  # re-hit: lightweight


def test_seen_set_reset_between_tests():
    """Guards the conftest reset — the dedup set starts empty each test."""
    assert mcp_server._SEEN_SLUGS == set()


# --------------------------------------------------------------------------- #
# #5b record returns the markdown path
# --------------------------------------------------------------------------- #

def test_record_returns_decisions_path(ws):
    """The write tool hands back the path to the human-readable markdown it wrote."""
    config, m = ws
    res = m.record_decision_entry("A decision.", "rej", ["s"], slug="pathed")
    assert res["status"] == "created"
    assert res["path"] == config.decisions_file


def test_record_exists_also_returns_path(ws):
    """Idempotent re-record still points at the markdown."""
    config, m = ws
    m.record_decision_entry("A decision.", "rej", ["s"], slug="dup")
    res = m.record_decision_entry("A decision.", "rej", ["s"], slug="dup")
    assert res["status"] == "exists"
    assert res["path"] == config.decisions_file


def test_cli_record_prints_path(ws, capsys):
    """cmd_record surfaces the written path to the human."""
    config, _ = ws
    cmd_record(config, axiom="Printed path.", rejected="rej", slug="printed")
    out = capsys.readouterr().out
    assert config.decisions_file in out and "Written:" in out


# --------------------------------------------------------------------------- #
# #2 residual — explicit-slug nudge on truncation
# --------------------------------------------------------------------------- #

_LONG_AXIOM = ("The art catalog listing endpoint resolves persona scoped collections "
               "through the catalog data module rather than inlining them per persona")


def test_record_long_auto_slug_emits_hint(ws):
    """A truncated auto-derived slug nudges for an explicit slug."""
    config, m = ws
    res = m.record_decision_entry(_LONG_AXIOM, "rej", ["catalog"])
    assert res["status"] == "created"
    assert "slug_hint" in res and "slug=" in res["slug_hint"]


def test_record_long_explicit_slug_no_hint(ws):
    """An explicit slug on a long axiom suppresses the nudge."""
    config, m = ws
    res = m.record_decision_entry(_LONG_AXIOM, "rej", ["catalog"], slug="catalog-endpoint-resolves")
    assert "slug_hint" not in res


def test_record_short_axiom_no_hint(ws):
    """A short axiom (no truncation) gets no nudge."""
    config, m = ws
    res = m.record_decision_entry("Short and sweet.", "rej", ["s"])
    assert "slug_hint" not in res


# --------------------------------------------------------------------------- #
# #5a open_questions disambiguation
# --------------------------------------------------------------------------- #

def test_surface_omits_open_questions_without_scope(ws):
    """No scope → open_questions key ABSENT (it wasn't scanned, not 'none')."""
    config, m = ws
    m.record_decision_entry("Decision.", "rej", ["s"], slug="d")
    store = _ro_store(config)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.surface_decisions("anything"))
    assert "open_questions" not in resp


def test_surface_includes_open_questions_with_scope(ws):
    """A scope → open_questions key PRESENT (possibly [], meaning none parked here)."""
    config, m = ws
    m.record_decision_entry("Decision.", "rej", ["scoped"], slug="d")
    store = _ro_store(config)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.surface_decisions("anything", scope="scoped"))
    assert "open_questions" in resp and resp["open_questions"] == []
