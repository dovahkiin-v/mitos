"""Tests for the surface-recall confidence signal (AX P5) + unused-scope recovery (3c).

Loop-Claude's friction: `surface_decisions` returned a capped list of mid-score
neighbours that looked identical to a real precedent, and an empty result that looked
identical to "the real precedent is hiding just below the cap." Neither the agent could
trust. Now every response carries a `confidence` (strong/weak/none) and an action note,
the policy lives in `mitos.recall` (shared by the MCP tool and the CLI twin), and a
semantic run that finds nothing no longer dumps an unranked scope listing dressed as
matches.

3c makes the recall core **surface-agnostic** (each surface words its own pointer — CLI
shell verbs vs MCP tool call-forms, single-sourced from `_SURFACE_POINTERS`) and
replaces the old unbounded `"Valid scopes are: …"` enumeration with a **bounded
self-correction vector** (did-you-mean + top-K busiest-first + overflow pointer + a
static `mitos sync` hedge). The unused-scope signal keys on **live-vocabulary
membership** (`get_scope_counts`), so a scope live only via a parked open question is a
real tag, not a typo.

The unit tests pin the pure policy; the integration tests drive the MCP tool and the CLI
twin with a fake vector store so scores are deterministic without Qdrant/keys.
"""

import io
import json
import shutil
import tempfile
from contextlib import redirect_stdout
from typing import Iterator, Tuple

import pytest
from unittest.mock import patch

from mitos.config import MitosConfig
from mitos.cli import cmd_init, cmd_surface
from mitos.parser import ParsedEntry
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


def _commit_oq(store: GraphStore, slug: str, scope) -> None:
    """Commits a hand-built parked open_question through the write path (no embed).

    `commit_parsed_entry` returns a `CommitDelta` and *raises* on failure — do NOT
    `assert "error" not in res` on it (3b gotcha).
    """
    e = ParsedEntry("open_question", slug, 1, 5)
    e.topic = f"Topic for {slug}"
    e.questions_raised = [f"What about {slug}?"]
    e.scope = list(scope)
    store.commit_parsed_entry(e)


class _FakeEmbed:
    def get_embedding(self, text, is_query=False):
        return [0.1, 0.2, 0.3]


class _FakeVector:
    def __init__(self, matches):
        self._matches = matches

    def query(self, vector, limit=5):
        return self._matches


def _counts(*names_and_counts):
    """Builds a busiest-first `get_scope_counts`-shaped map from (name, n) pairs.

    The caller lists pairs in the order they want them to arrive at the policy (the real
    callsite pre-orders via `order_scope_counts`); active-decision count carries `n`.
    """
    return {name: {"active_decisions": n, "parked_open_questions": 0}
            for name, n in names_and_counts}


# --------------------------------------------------------------------------- #
# Pure policy — mitos.recall.assess_surface_recall
# --------------------------------------------------------------------------- #

def test_policy_strong_when_top_score_clears_threshold():
    conf, note = assess_surface_recall(semantic_ran=True, top_score=0.9, result_count=2,
                                       scope="db", surface="cli")
    assert conf == "strong"
    assert "mitos list" in note and "list_decisions" not in note


def test_policy_strong_mcp_uses_mcp_callform():
    conf, note = assess_surface_recall(semantic_ran=True, top_score=0.9, result_count=2,
                                       scope="db", surface="mcp")
    assert conf == "strong"
    assert "list_decisions(scope='db')" in note


def test_policy_strong_at_exact_threshold():
    """The threshold is inclusive — a score exactly at the bar is strong."""
    conf, _ = assess_surface_recall(semantic_ran=True, top_score=SURFACE_STRONG_THRESHOLD,
                                    result_count=1, scope=None, surface="cli")
    assert conf == "strong"


def test_policy_weak_below_threshold_names_the_score():
    conf, note = assess_surface_recall(semantic_ran=True, top_score=0.61, result_count=3,
                                       scope=None, surface="cli")
    assert conf == "weak"
    assert "0.61" in note


def test_policy_off_axis_below_weak_threshold():
    conf, note = assess_surface_recall(semantic_ran=True, top_score=0.55, result_count=3,
                                       scope=None, surface="cli")
    assert conf == "none"
    assert "0.55" in note and "off-axis" in note.lower()


def test_policy_none_no_match_points_to_list():
    conf, note = assess_surface_recall(semantic_ran=True, top_score=None, result_count=0,
                                       scope=None, surface="cli")
    assert conf == "none" and "No semantic match" in note
    assert "mitos list" in note and "list_decisions" not in note


def test_policy_none_scope_unused_bounded_vector():
    """Migrated from the old `Valid scopes are: db` enumeration → bounded vector."""
    conf, note = assess_surface_recall(semantic_ran=True, top_score=None, result_count=0,
                                       scope="ghost", scope_counts=_counts(("db", 1)),
                                       surface="cli")
    assert conf == "none"
    assert "unused scope tag" in note and "db" in note
    assert "Valid scopes are" not in note


def test_policy_weak_scope_unused_but_has_matches():
    conf, note = assess_surface_recall(semantic_ran=True, top_score=0.65, result_count=1,
                                       scope="ghost", scope_counts=_counts(("auth", 1)),
                                       surface="cli")
    assert conf == "weak"
    assert "unused scope tag" in note
    assert "auth" in note and "Valid scopes are" not in note
    assert "matched semantically (twilight zone" in note


def test_policy_degraded_with_results_is_not_a_ranking():
    conf, note = assess_surface_recall(semantic_ran=False, top_score=None, result_count=4,
                                       scope="db", surface="cli")
    assert conf is None
    assert "unavailable" in note and "NOT a relevance ranking" in note
    assert "mitos list" in note and "list_decisions" not in note


def test_policy_degraded_empty_scope_unused():
    conf, note = assess_surface_recall(semantic_ran=False, top_score=None, result_count=0,
                                       scope="ghost", scope_counts={}, surface="cli")
    assert conf is None and "unavailable" in note and "unused scope tag" in note


# --------------------------------------------------------------------------- #
# Bounded unused-scope vector (3c, W9 / T7)
# --------------------------------------------------------------------------- #

def test_unused_vector_did_you_mean():
    _, note = assess_surface_recall(semantic_ran=True, top_score=None, result_count=0,
                                    scope="ath", scope_counts=_counts(("auth", 3)),
                                    surface="cli")
    assert "Did you mean 'auth'?" in note


def test_unused_vector_top_k_and_overflow_bounded():
    """At most K busiest-first tags + a discovery pointer; the (K+1)th tag is absent."""
    counts = _counts(("substrate", 9), ("store", 8), ("schema", 7), ("vector", 6),
                     ("parser", 5), ("config", 4), ("render", 3))  # 7 live > K=5
    _, note = assess_surface_recall(semantic_ran=True, top_score=None, result_count=0,
                                    scope="ghost", scope_counts=counts, surface="cli")
    assert "Live scopes (busiest first): substrate, store, schema, vector, parser." in note
    assert "config" not in note and "render" not in note   # the 6th/7th are not listed
    assert "mitos scopes" in note                          # overflow pointer (CLI form)


def test_unused_vector_overflow_pointer_mcp_form():
    counts = _counts(("a1", 9), ("b2", 8), ("c3", 7), ("d4", 6), ("e5", 5), ("f6", 4))
    _, note = assess_surface_recall(semantic_ran=True, top_score=None, result_count=0,
                                    scope="ghost", scope_counts=counts, surface="mcp")
    assert "list_scopes" in note and "mitos scopes" not in note


def test_unused_vector_sync_hedge_present():
    _, note = assess_surface_recall(semantic_ran=True, top_score=None, result_count=0,
                                    scope="ghost", scope_counts=_counts(("auth", 1)),
                                    surface="cli")
    assert "mitos sync" in note


def test_unused_vector_empty_project_is_calm():
    """A fresh/empty project: just the unused-tag statement + sync hedge — no list, no
    did-you-mean."""
    _, note = assess_surface_recall(semantic_ran=True, top_score=None, result_count=0,
                                    scope="ghost", scope_counts={}, surface="cli")
    assert "unused scope tag" in note and "mitos sync" in note
    assert "Did you mean" not in note and "Live scopes" not in note


def test_unused_signal_keys_on_live_map_not_active_count():
    """A scope present in the live map (e.g. live only via a parked OQ → count 0/1) is
    NOT flagged unused — membership, not active-decision count, is the oracle."""
    counts = {"auth": {"active_decisions": 0, "parked_open_questions": 1}}
    _, note = assess_surface_recall(semantic_ran=True, top_score=None, result_count=0,
                                    scope="auth", scope_counts=counts, surface="cli")
    assert "unused scope tag" not in note


def test_none_scope_counts_never_fabricates_unused():
    """`scope_counts=None` (callsite couldn't compute) → calm degradation, never a typo
    hint."""
    _, note = assess_surface_recall(semantic_ran=True, top_score=None, result_count=0,
                                    scope="ghost", scope_counts=None, surface="cli")
    assert "unused scope tag" not in note


def test_surface_leak_gate_cli_never_emits_mcp_callforms():
    """T7 load-bearing pin: no CLI-surfaced note carries an MCP *tool* call-form across
    the unused / degraded / completeness / no-match branches."""
    counts = _counts(("auth", 3), ("store", 2))
    cases = [
        dict(semantic_ran=True, top_score=None, result_count=0, scope="ghost"),    # unused, no match
        dict(semantic_ran=False, top_score=None, result_count=0, scope="ghost"),   # degraded, unused
        dict(semantic_ran=False, top_score=None, result_count=4, scope="auth"),    # degraded, populated
        dict(semantic_ran=True, top_score=0.9, result_count=2, scope="auth"),      # completeness, scoped
        dict(semantic_ran=True, top_score=0.9, result_count=2, scope=None),        # completeness, no scope
        dict(semantic_ran=True, top_score=None, result_count=0, scope=None),       # no match, no scope
    ]
    for c in cases:
        _, note = assess_surface_recall(scope_counts=counts, surface="cli", **c)
        assert "list_decisions(" not in note, c
        assert "list_scopes(" not in note, c


def test_cli_mcp_signal_parity_for_unused_scope():
    """Same unused-scope *signal* on both surfaces; only the pointer wording differs."""
    counts = _counts(("auth", 3))
    _, cli_note = assess_surface_recall(semantic_ran=True, top_score=None, result_count=0,
                                        scope="ghost", scope_counts=counts, surface="cli")
    _, mcp_note = assess_surface_recall(semantic_ran=True, top_score=None, result_count=0,
                                        scope="ghost", scope_counts=counts, surface="mcp")
    assert "unused scope tag" in cli_note and "unused scope tag" in mcp_note
    assert "list_decisions(" not in cli_note
    # MCP keeps its tool call-forms; CLI keeps shell verbs — same signal, worded per surface.


def test_surface_is_required_keyword():
    with pytest.raises(TypeError):
        assess_surface_recall(semantic_ran=True, top_score=0.9, result_count=1, scope=None)


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


def _cli_surface_json(matches, ws, query="some claim", scope=None):
    """Drives the CLI `cmd_surface` end-to-end with deterministic scores and returns the
    parsed `--json` payload. `matches=None` exercises the degraded (no embed/vector) path."""
    from mitos import cli
    config, _ = ws
    manager = MitosSyncManager(config)
    if matches is None:
        manager.embed_provider = None
        manager.vector_store = None
    else:
        manager.embed_provider = _FakeEmbed()
        manager.vector_store = _FakeVector(matches)
    buf = io.StringIO()
    with patch.object(cli, "MitosSyncManager", return_value=manager):
        with redirect_stdout(buf):
            cmd_surface(config, query, as_json=True, scope=scope)
    return json.loads(buf.getvalue())


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
    resp = _surface_with([], ws, scope="ghost")  # semantic ran, found nothing, ghost scope unused
    assert resp["confidence"] == "none"
    assert resp["active_decisions"] == []
    note = resp["note"]
    assert "unused scope tag" in note and "other" in note
    assert "Valid scopes are" not in note
    assert "list_decisions(" not in note  # MCP discovery pointer is `list_scopes`, not the list verb


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


def test_mcp_surface_parked_oq_scope_is_not_unused(ws):
    """A scope live ONLY via a parked open question is a real tag — the unused-scope
    vector must NOT fire (membership keys on the live map, not active-decision count)."""
    config, m = ws
    store = GraphStore(config.db_path)
    _commit_oq(store, "q-auth", scope=["auth"])           # auth: 0 active, 1 parked OQ
    resp = _surface_with([], ws, scope="auth")            # semantic ran, no matches
    assert resp["confidence"] == "none"
    assert "unused scope tag" not in resp["note"]


# --------------------------------------------------------------------------- #
# CLI cmd_surface — end-to-end surface-leak gate + parity (3c, T7)
# --------------------------------------------------------------------------- #

def test_cli_surface_unused_scope_no_mcp_leak(ws):
    """T7 from the CLI verb entry: an unused scope self-corrects with CLI verbs and
    never an MCP tool call-form."""
    config, m = ws
    for s in ("substrate", "store", "schema", "vector", "parser", "config"):  # 6 live > K
        _rec(m, f"{s}-dec", scope=[s])
    resp = _cli_surface_json([], ws, scope="ghost")        # semantic ran, no matches, ghost unused
    note = resp["note"]
    assert resp["confidence"] == "none"
    assert "unused scope tag" in note
    assert "Live scopes (busiest first):" in note
    assert "mitos scopes" in note                          # overflow pointer (CLI form)
    assert "mitos sync" in note                            # authored-but-unsynced hedge
    assert "list_decisions(" not in note and "list_scopes(" not in note


def test_cli_and_mcp_unused_scope_signal_parity(ws):
    """Both surfaces fire the unused-scope signal for the same scope; only the overflow
    pointer wording differs (CLI `mitos scopes` vs MCP `list_scopes`)."""
    config, m = ws
    for s in ("auth", "store", "schema", "vector", "parser", "config"):  # 6 live > K
        _rec(m, f"{s}-dec", scope=[s])
    cli_resp = _cli_surface_json([], ws, scope="ghost")
    mcp_resp = _surface_with([], ws, scope="ghost")
    assert "unused scope tag" in cli_resp["note"] and "unused scope tag" in mcp_resp["note"]
    assert "auth" in cli_resp["note"] and "auth" in mcp_resp["note"]   # alpha-first → in top-K
    assert "mitos scopes" in cli_resp["note"] and "list_decisions(" not in cli_resp["note"]
    assert "list_scopes" in mcp_resp["note"]               # MCP: tool call-form


def test_cli_surface_degraded_no_mcp_leak(ws):
    """Degraded CLI path (no embed/vector) still words its completeness pointer as a CLI
    verb, never the MCP `list_decisions()` call-form."""
    config, m = ws
    _rec(m, "settled", scope=["db"])
    resp = _cli_surface_json(None, ws, scope="db")         # degraded; db populated → fallback fires
    note = resp["note"]
    assert "unavailable" in note
    assert "mitos list" in note and "list_decisions(" not in note
