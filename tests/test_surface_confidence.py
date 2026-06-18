"""Tests for the surface-recall confidence signal (AX P5).

Loop-Claude's friction: `surface_decisions` returned a capped list of mid-score
neighbours that looked identical to a real precedent, and an empty result that looked
identical to "the real precedent is hiding just below the cap." Neither the agent could
trust. Now every response carries a `confidence` (strong/weak/none) and an action note,
the policy lives in `mitos.recall` (shared by the MCP tool and the CLI twin), and a
semantic run that finds nothing no longer dumps an unranked scope listing dressed as
matches.

The unit tests pin the pure policy; the integration tests drive the MCP tool with a fake
vector store so scores are deterministic without Qdrant/keys.
"""

import json
import shutil
import tempfile
from typing import Iterator, Tuple

import pytest
from unittest.mock import patch

from mitos.config import MitosConfig
from mitos.cli import cmd_init
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager
from mitos.recall import assess_surface_recall, SURFACE_STRONG_THRESHOLD, SURFACE_WEAK_THRESHOLD


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


def _rec(m, slug, scope=None):
    res = m.record_decision_entry(f"Axiom for {slug}.", f"Rejected for {slug}.",
                                  scope or [], slug=slug)
    assert "error" not in res, res
    return res


class _FakeEmbed:
    def get_embedding(self, text, is_query=False):
        return [0.1, 0.2, 0.3]


class _FakeVector:
    def __init__(self, matches):
        self._matches = matches

    def query(self, vector, limit=5, filter_scope=None):
        return self._matches


# --------------------------------------------------------------------------- #
# Pure policy — mitos.recall.assess_surface_recall
# --------------------------------------------------------------------------- #

def test_policy_strong_when_top_score_clears_threshold():
    conf, note = assess_surface_recall(semantic_ran=True, top_score=0.9, result_count=2,
                                       scope="db", scope_decision_count=None)
    assert conf == "strong" and "list_decisions" in note


def test_policy_strong_at_exact_threshold():
    """The threshold is inclusive — a score exactly at the bar is strong."""
    conf, _ = assess_surface_recall(semantic_ran=True, top_score=SURFACE_STRONG_THRESHOLD,
                                    result_count=1, scope=None, scope_decision_count=None)
    assert conf == "strong"


def test_policy_weak_below_threshold_names_the_score():
    conf, note = assess_surface_recall(semantic_ran=True, top_score=0.61, result_count=3,
                                       scope=None, scope_decision_count=None)
    assert conf == "weak"
    assert "0.61" in note


def test_policy_off_axis_below_weak_threshold():
    conf, note = assess_surface_recall(semantic_ran=True, top_score=0.55, result_count=3,
                                       scope=None, scope_decision_count=None)
    assert conf == "none"
    assert "0.55" in note and "off-axis" in note.lower()


def test_policy_none_no_match_points_to_list():
    conf, note = assess_surface_recall(semantic_ran=True, top_score=None, result_count=0,
                                       scope=None, scope_decision_count=None)
    assert conf == "none" and "No semantic match" in note and "list_decisions" in note


def test_policy_none_scope_unused_says_zero_decisions():
    conf, note = assess_surface_recall(semantic_ran=True, top_score=None, result_count=0,
                                       scope="ghost", scope_decision_count=0)
    assert conf == "none" and "0 decisions" in note and "tag unused" in note


def test_policy_degraded_with_results_is_not_a_ranking():
    conf, note = assess_surface_recall(semantic_ran=False, top_score=None, result_count=4,
                                       scope="db", scope_decision_count=None)
    assert conf is None
    assert "unavailable" in note and "NOT a relevance ranking" in note and "list_decisions" in note


def test_policy_degraded_empty_scope_unused():
    conf, note = assess_surface_recall(semantic_ran=False, top_score=None, result_count=0,
                                       scope="ghost", scope_decision_count=0)
    assert conf is None and "unavailable" in note and "0 decisions" in note


# --------------------------------------------------------------------------- #
# MCP surface_decisions — confidence end to end (fake vector store)
# --------------------------------------------------------------------------- #

def _surface_with(matches, ws, query="some claim", scope=None):
    from mitos import mcp_server
    config, _ = ws
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, _FakeEmbed(), _FakeVector(matches))):
        return json.loads(mcp_server.surface_decisions(query, scope=scope))


def test_mcp_surface_strong_hit(ws):
    config, m = ws
    _rec(m, "real-precedent", scope=["db"])
    resp = _surface_with([{"slug": "real-precedent", "score": 0.91}], ws, scope="db")
    assert resp["confidence"] == "strong"
    assert resp["active_decisions"][0]["slug"] == "real-precedent"


def test_mcp_surface_weak_hit_flagged(ws):
    config, m = ws
    _rec(m, "loose-neighbour", scope=["db"])
    resp = _surface_with([{"slug": "loose-neighbour", "score": 0.62}], ws, scope="db")
    assert resp["confidence"] == "weak"
    assert "Twilight zone" in resp["note"]
    assert resp["active_decisions"]  # still returned, just flagged weak


def test_mcp_surface_no_match_scope_unused(ws):
    config, m = ws
    _rec(m, "elsewhere", scope=["other"])
    resp = _surface_with([], ws, scope="ghost")  # semantic ran, found nothing, ghost scope empty
    assert resp["confidence"] == "none"
    assert resp["active_decisions"] == []
    assert "0 decisions" in resp["note"]


def test_mcp_surface_semantic_empty_does_not_dump_scope_listing(ws):
    """KEY P5 behaviour: a semantic run that finds nothing in a POPULATED scope returns
    empty + confidence none — it must NOT fall back to dumping the scope's decisions as
    if they were matches (that was the false-precedent ambiguity)."""
    config, m = ws
    _rec(m, "in-scope-but-not-matched", scope=["db"])
    resp = _surface_with([], ws, scope="db")  # FakeVector returns no matches
    assert resp["confidence"] == "none"
    assert resp["active_decisions"] == []     # scope listing NOT dumped


def test_mcp_surface_degraded_has_no_confidence(ws):
    """Offline (no embed/vector) → degraded: a note but no `confidence`, and the scope
    listing fallback still fires so a CLI-only agent gets something."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "settled", scope=["db"])
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.surface_decisions("anything", scope="db"))
    assert "confidence" not in resp
    assert resp["active_decisions"]                       # degraded fallback fired
    assert "unavailable" in resp["note"] and "list_decisions" in resp["note"]
