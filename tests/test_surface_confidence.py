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

import ast
import inspect
import io
import json
import os
import shutil
import tempfile
from contextlib import redirect_stdout
from typing import Iterator, Tuple

import pytest
from unittest.mock import patch

from mitos.config import MitosConfig
from mitos.cli import cmd_init, cmd_query, cmd_surface
from mitos.errors import CollectionMissingError
from mitos.parser import ParsedEntry
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager
from mitos.provider_cause import AUTH, provider_cause_phrase
from mitos.recall import (ShortWindow, short_window_clause)  # noqa: E402  (2f)
from mitos.recall import degraded_section_clause  # noqa: E402  (5d)
from mitos.recall import (_SURFACE_POINTERS, assess_query_recall,
                          assess_surface_recall, SURFACE_STRONG_THRESHOLD,
                          SURFACE_WEAK_THRESHOLD, EDGE_KIND_MID_CLUSTER,
                          EDGE_KIND_UNMEASURED, RANKED_EDGE_MARGIN, WindowEdge,
                          edge_clause, window_edge)

from _point_hits import CannedVector, orphan_hit


# The cause the degraded rows inject. It is the phrase the "no providers" route
# really produces (``degraded_reason_from_error(None, …)``), so an end-to-end row
# on that route and a pure-policy row compare against the same words.
_DEGRADED = "embeddings/Qdrant unavailable"


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


def _rec(m, slug, scope=None, **relations):
    res = m.record_decision_entry(f"Axiom for {slug}.", f"Rejected for {slug}.",
                                  scope or [], slug=slug, **relations)
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


class _FakeVector(CannedVector):
    """Canned matches through the shared leaf (every hit carries its ``point_id``)."""


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
    conf, note = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, top_score=0.9, result_count=2,
                                       scope="db", surface="cli")
    assert conf == "strong"
    assert "mitos list" in note and "list_decisions" not in note


def test_policy_strong_mcp_uses_mcp_callform():
    conf, note = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, top_score=0.9, result_count=2,
                                       scope="db", surface="mcp")
    assert conf == "strong"
    assert "list_decisions(scope='db')" in note


def test_policy_strong_at_exact_threshold():
    """The threshold is inclusive — a score exactly at the bar is strong."""
    conf, _ = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, top_score=SURFACE_STRONG_THRESHOLD,
                                    result_count=1, scope=None, surface="cli")
    assert conf == "strong"


def test_policy_weak_below_threshold_names_the_score():
    conf, note = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, top_score=0.61, result_count=3,
                                       scope=None, surface="cli")
    assert conf == "weak"
    assert "0.61" in note


def test_policy_off_axis_below_weak_threshold():
    conf, note = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, top_score=0.55, result_count=3,
                                       scope=None, surface="cli")
    assert conf == "none"
    assert "0.55" in note and "off-axis" in note.lower()


def test_policy_none_no_match_points_to_list():
    conf, note = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, top_score=None, result_count=0,
                                       scope=None, surface="cli")
    assert conf == "none" and "No semantic match" in note
    assert "mitos list" in note and "list_decisions" not in note


def test_policy_none_scope_unused_bounded_vector():
    """Migrated from the old `Valid scopes are: db` enumeration → bounded vector."""
    conf, note = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, top_score=None, result_count=0,
                                       scope="ghost", scope_counts=_counts(("db", 1)),
                                       surface="cli")
    assert conf == "none"
    assert "unused scope tag" in note and "db" in note
    assert "Valid scopes are" not in note


def test_policy_weak_scope_unused_but_has_matches():
    conf, note = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, top_score=0.65, result_count=1,
                                       scope="ghost", scope_counts=_counts(("auth", 1)),
                                       surface="cli")
    assert conf == "weak"
    assert "unused scope tag" in note
    assert "auth" in note and "Valid scopes are" not in note
    assert "matched semantically (twilight zone" in note


def test_policy_degraded_with_results_is_not_a_ranking():
    # 5d: on the dump arm (`scope_total` given), the only degraded arm with results.
    conf, note = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=False, degraded_reason=_DEGRADED, top_score=None, result_count=4,
                                       scope="db", scope_total=4, surface="cli")
    assert conf is None
    assert "unavailable" in note and "NOT a relevance ranking" in note
    assert "mitos list" in note and "list_decisions" not in note


def test_policy_degraded_empty_scope_unused():
    conf, note = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=False, degraded_reason=_DEGRADED, top_score=None, result_count=0,
                                       scope="ghost", scope_counts={}, surface="cli")
    assert conf is None and "unavailable" in note and "unused scope tag" in note


# --------------------------------------------------------------------------- #
# Bounded unused-scope vector (3c, W9 / T7)
# --------------------------------------------------------------------------- #

def test_unused_vector_did_you_mean():
    _, note = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, top_score=None, result_count=0,
                                    scope="ath", scope_counts=_counts(("auth", 3)),
                                    surface="cli")
    assert "Did you mean 'auth'?" in note


def test_unused_vector_top_k_and_overflow_bounded():
    """At most K busiest-first tags + a discovery pointer; the (K+1)th tag is absent."""
    counts = _counts(("substrate", 9), ("store", 8), ("schema", 7), ("vector", 6),
                     ("parser", 5), ("config", 4), ("render", 3))  # 7 live > K=5
    _, note = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, top_score=None, result_count=0,
                                    scope="ghost", scope_counts=counts, surface="cli")
    assert "Live scopes (busiest first): substrate, store, schema, vector, parser." in note
    assert "config" not in note and "render" not in note   # the 6th/7th are not listed
    assert "mitos scopes" in note                          # overflow pointer (CLI form)


def test_unused_vector_overflow_pointer_mcp_form():
    counts = _counts(("a1", 9), ("b2", 8), ("c3", 7), ("d4", 6), ("e5", 5), ("f6", 4))
    _, note = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, top_score=None, result_count=0,
                                    scope="ghost", scope_counts=counts, surface="mcp")
    assert "list_scopes" in note and "mitos scopes" not in note


def test_unused_vector_sync_hedge_present():
    _, note = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, top_score=None, result_count=0,
                                    scope="ghost", scope_counts=_counts(("auth", 1)),
                                    surface="cli")
    assert "mitos sync" in note


def test_unused_vector_empty_project_is_calm():
    """A fresh/empty project: just the unused-tag statement + sync hedge — no list, no
    did-you-mean."""
    _, note = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, top_score=None, result_count=0,
                                    scope="ghost", scope_counts={}, surface="cli")
    assert "unused scope tag" in note and "mitos sync" in note
    assert "Did you mean" not in note and "Live scopes" not in note


def test_unused_signal_keys_on_live_map_not_active_count():
    """A scope present in the live map (e.g. live only via a parked OQ → count 0/1) is
    NOT flagged unused — membership, not active-decision count, is the oracle."""
    counts = {"auth": {"active_decisions": 0, "parked_open_questions": 1}}
    _, note = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, top_score=None, result_count=0,
                                    scope="auth", scope_counts=counts, surface="cli")
    assert "unused scope tag" not in note


def test_none_scope_counts_never_fabricates_unused():
    """`scope_counts=None` (callsite couldn't compute) → calm degradation, never a typo
    hint."""
    _, note = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, top_score=None, result_count=0,
                                    scope="ghost", scope_counts=None, surface="cli")
    assert "unused scope tag" not in note


def test_surface_leak_gate_cli_never_emits_mcp_callforms():
    """T7 load-bearing pin: no CLI-surfaced note carries an MCP *tool* call-form across
    the unused / degraded / completeness / no-match branches."""
    counts = _counts(("auth", 3), ("store", 2))
    cases = [
        dict(semantic_ran=True, top_score=None, result_count=0, scope="ghost"),    # unused, no match
        dict(semantic_ran=False, top_score=None, result_count=0, scope="ghost"),   # degraded, unused
        dict(semantic_ran=False, top_score=None, result_count=4, scope="auth",
             scope_total=4),                                                       # degraded, populated (the dump; 5d)
        dict(semantic_ran=True, top_score=0.9, result_count=2, scope="auth"),      # completeness, scoped
        dict(semantic_ran=True, top_score=0.9, result_count=2, scope=None),        # completeness, no scope
        dict(semantic_ran=True, top_score=None, result_count=0, scope=None),       # no match, no scope
    ]
    for c in cases:
        _, note = assess_surface_recall(
            lead=None, short=None,
            edge=None, scope_counts=counts, surface="cli",
            degraded_reason=None if c["semantic_ran"] else _DEGRADED, **c)
        assert "list_decisions(" not in note, c
        assert "list_scopes(" not in note, c


def test_cli_mcp_signal_parity_for_unused_scope():
    """Same unused-scope *signal* on both surfaces; only the pointer wording differs."""
    counts = _counts(("auth", 3))
    _, cli_note = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, top_score=None, result_count=0,
                                        scope="ghost", scope_counts=counts, surface="cli")
    _, mcp_note = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, top_score=None, result_count=0,
                                        scope="ghost", scope_counts=counts, surface="mcp")
    assert "unused scope tag" in cli_note and "unused scope tag" in mcp_note
    assert "list_decisions(" not in cli_note
    # MCP keeps its tool call-forms; CLI keeps shell verbs — same signal, worded per surface.


def test_surface_is_required_keyword():
    # `edge=None` passed so this row keeps testing `surface`, not the newer keyword.
    with pytest.raises(TypeError):
        assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, top_score=0.9, result_count=1, scope=None)


# --------------------------------------------------------------------------- #
# MCP surface_decisions — confidence end to end (fake vector store)
# --------------------------------------------------------------------------- #

def _surface_with(matches, ws, query="some claim", scope=None, full_top=None, brief=False,
                  limit=None):
    """`matches=None` drives no providers; a vector-store instance drives its own route."""
    from mitos import mcp_server
    config, _ = ws
    store = GraphStore(config.db_path, read_only=True)
    vector = _FakeVector(matches, store=store) if isinstance(matches, list) else matches
    embed = None if matches is None else _FakeEmbed()
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, embed, vector)):
        kw = {} if limit is None else {"limit": limit}
        return json.loads(mcp_server.surface_decisions(
            query, scope=scope, brief=brief, project=config.workspace_dir,
            full_top=full_top, **kw))


def _cli_surface_json(matches, ws, query="some claim", scope=None, full_top=None,
                      brief=False, limit=None):
    """Drives the CLI `cmd_surface` end-to-end with deterministic scores and returns the
    parsed `--json` payload. `matches=None` exercises the degraded (no embed/vector) path;
    a vector-store instance is passed through as-is, so a fault stub drives its own route."""
    from mitos import cli
    config, _ = ws
    manager = MitosSyncManager(config)
    if matches is None:
        manager.embed_provider = None
        manager.vector_store = None
    else:
        manager.embed_provider = _FakeEmbed()
        manager.vector_store = (_FakeVector(matches) if isinstance(matches, list)
                                else matches)
    buf = io.StringIO()
    with patch.object(cli, "MitosSyncManager", return_value=manager):
        with redirect_stdout(buf):
            cmd_surface(config, query, as_json=True, scope=scope, brief=brief,
                        full_top=full_top, limit=limit)
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
        resp = json.loads(mcp_server.surface_decisions("anything", scope="db", project=config.workspace_dir))
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


# --------------------------------------------------------------------------- #
# A1 — the `query` register (3a)
#
# The band `surface` has carried since June, wired onto the targeted recall verbs
# with a register of their own. The band is the SAME three tokens on all four
# surfaces (one classification, `_classify_recall`); the NOTE is not shared, and
# none of it may be inherited — `surface`'s wording is a verdict on the corpus
# ("likely no settled precedent", "decide and record it", "the scope is
# populated"), and a targeted lookup measured none of that. The half-register is
# the failure that ships green: fresh `"none"` sentences beside `strong`/`weak`
# still speaking `surface`'s language, symmetric across boundaries so every parity
# fixture agrees. T5 below is its only mechanical detector.
# --------------------------------------------------------------------------- #

class _StubConfig:
    """The duck-typed config `assess_query_recall` reads — `project` and nothing else.

    The composer takes a config rather than a `project=` string so a call site
    cannot hand it a value it computed itself; it reads the attribute on
    `corpus_provenance`'s own `getattr` idiom, which is what keeps `recall.py` a
    Tier-1 leaf with zero `mitos` imports.
    """

    def __init__(self, project="/home/user/projects/demo"):
        self.project = project


# The four reachable input tuples on this verb, computed rather than judged: `scope`
# arrives structurally None at both `query` call sites, so every `scope_unused`
# branch of the shipped policy is dead here and exactly four sentences survive. A
# phase dutifully writing the shipped policy's eight has written four it can never
# emit.
_QUERY_CASES = [
    ("strong", dict(top_score=0.91, result_count=2)),
    ("weak", dict(top_score=0.61, result_count=3)),
    ("none-with-results", dict(top_score=0.55, result_count=3)),
    ("none-empty", dict(top_score=None, result_count=0)),
]

def _query_pointer(surface, config):
    """The redirect call-form as composed, so a row can price it out of a length.

    Derived from the table rather than re-spelled here: a hand-copy of the
    production template is a second source for the one string the table exists to
    single-source. The literal spelling is pinned by its own rows below (the CLI
    form names `mitos surface -p`, and its `repr` survives a project carrying a
    space), so deriving costs no coverage.
    """
    from mitos.shell_quote import shell_flag

    return _SURFACE_POINTERS[surface]["precedent_scan"].format(
        selector=shell_flag("-p", config.project)
    )


# The edge shapes the register and length rows range over (2e): none, a mid-cluster
# window with every rank held whole, one already axiom-only, one at the ceiling (no
# lever), and an unmeasured edge (no lever).
_EDGE_SHAPES = [
    None,
    WindowEdge(EDGE_KIND_MID_CLUSTER, 5, 5),
    WindowEdge(EDGE_KIND_MID_CLUSTER, 5, 0),
    WindowEdge(EDGE_KIND_MID_CLUSTER, 50, None),
    WindowEdge(EDGE_KIND_UNMEASURED, 5, None),
]


def _query_note(surface, config=None, edge=None, lead=None, short=None, **kw):
    return assess_query_recall(lead=lead, short=short, edge=edge, config=config or _StubConfig(), surface=surface, **kw)[1]


# --------------------------------------------------------------------------- #
# T4 — the four register sentences, one per reachable input tuple
# --------------------------------------------------------------------------- #

def test_query_strong_is_a_legend_and_names_no_verb():
    """Displaces: "Ranked top matches. For the COMPLETE set … call {complete_hint}."

    Every clause of that sentence is gone. The scope-completeness framing is
    structurally dead on a scopeless verb, and its `list_decisions` redirect
    answers a question nobody raised — the caller named a thing and got it, so a
    redirect here is a per-answer turn tax wearing a legend (P15). What must NOT
    follow is "so `strong` carries no note": this is the band a caller meets most
    often, and dropping it leaves the common answer with no legend at all.
    """
    conf, note = assess_query_recall(lead=None, short=None, edge=None, top_score=0.91, result_count=2,
                                     config=_StubConfig(), surface="cli")
    assert conf == "strong"
    assert note                                        # it keeps a note
    for verb in ("mitos list", "list_decisions", "mitos surface",
                 "surface_decisions", "mitos sync", "mitos query"):
        assert verb not in note, f"the strong band named {verb!r}"


def test_query_weak_names_the_score_and_redirects():
    """Displaces: "Twilight zone: top score … Check carefully before deciding."

    The score and the twilight property survive; the closing instruction does not
    — "check carefully before deciding" is the precedent-check register, and that
    is `surface`'s question, not this verb's.
    """
    conf, note = assess_query_recall(lead=None, short=None, edge=None, top_score=0.61, result_count=3,
                                     config=_StubConfig(), surface="cli")
    assert conf == "weak"
    assert "0.61" in note
    assert "before deciding" not in note
    assert "mitos surface -p" in note                   # the redirect


def test_query_none_with_results_drops_the_scope_clause():
    """Displaces: "Very likely off-axis: … The scope is populated, but nothing
    matches your query. Treat as no-precedent and decide fresh."

    "The scope is populated" can never be true on a verb that takes no scope, and
    "treat as no-precedent and decide fresh" is a decide instruction off a
    targeted miss.
    """
    conf, note = assess_query_recall(lead=None, short=None, edge=None, top_score=0.55, result_count=3,
                                     config=_StubConfig(), surface="cli")
    assert conf == "none"
    assert "0.55" in note
    assert "The scope is populated" not in note
    assert "mitos surface -p" in note


def test_query_none_empty_drops_the_corpus_verdict():
    """Displaces: "No semantic match for {scope_phrase} — likely no settled
    precedent. Decide and record it, or call {complete_hint} …"

    "Likely no settled precedent" is a claim about the corpus that a lookup miss
    does not license, and "decide and record it" is exactly how an agent that
    reached for the wrong verb mints a duplicate of a decision the corpus already
    holds. The band is honest about the ranking and silent about the corpus.
    """
    conf, note = assess_query_recall(lead=None, short=None, edge=None, top_score=None, result_count=0,
                                     config=_StubConfig(), surface="cli")
    assert conf == "none"
    assert "no settled precedent" not in note
    assert "mitos surface -p" in note


# The `query` register lock (ADR query-band-register-states-ranking-never-instructs-a-write):
# no write instruction, no verdict on the corpus. One list, read by every row that
# polices a sentence in that register (4b's withheld clause included).
_QUERY_REGISTER_BANNED = ("decide", "record it", "before deciding", "the scope is populated",
                          "no settled precedent")


@pytest.mark.parametrize("label,kw", _QUERY_CASES)
@pytest.mark.parametrize("surface", ["cli", "mcp"])
def test_no_query_note_instructs_a_write_or_judges_the_corpus(surface, label, kw):
    """The content rule that binds all four, on both boundaries.

    Opera C2: a normal degraded-index state — Qdrant up, ranking ran, the precedent
    in the graph and absent from this ranking — banded honestly must not answer
    with a write instruction. Checked as an absence on every sentence rather than
    on the one branch where the shipped wording is most obviously wrong.
    """
    note = _query_note(surface, **kw)
    lowered = note.casefold()
    for banned in _QUERY_REGISTER_BANNED:
        assert banned not in lowered, f"{label}/{surface} note carries {banned!r}"


@pytest.mark.parametrize("label,kw", _QUERY_CASES)
def test_the_query_redirect_never_reimports_a_bare_recipe(label, kw):
    """G9: the CLI redirect is selectored; no bare `mitos …` recipe rides beside it.

    `complete_hint` (the bare `mitos list`) belongs to the ancestors these
    sentences displace, and `state_all` is reachable only through the scope-unused
    vector, which is dead on this verb. Both routes into the table's bare entries
    are shut — which is what makes the one selectored pointer consistent rather
    than a lone correct pointer beside broken siblings in the same message.
    """
    config = _StubConfig("/home/user/my projects/demo")   # a space: the runnability bound
    note = _query_note("cli", config=config, **kw)
    assert "mitos list" not in note and "mitos scopes" not in note
    if "mitos surface" in note:
        # repr-rendered, so a name or path carrying a space stays one shell word.
        assert "-p '/home/user/my projects/demo'" in note


@pytest.mark.parametrize("label,kw", _QUERY_CASES)
def test_the_query_register_leaks_no_call_form_across_the_boundary(label, kw):
    """G5 — the shipped `surface` leak gate's twin, in both directions."""
    cli_note = _query_note("cli", **kw)
    mcp_note = _query_note("mcp", **kw)
    assert "surface_decisions(" not in cli_note
    assert "list_decisions(" not in cli_note and "list_scopes(" not in cli_note
    assert "mitos surface" not in mcp_note and "-p " not in mcp_note


@pytest.mark.parametrize("edge", _EDGE_SHAPES)
@pytest.mark.parametrize("label,kw", _QUERY_CASES)
def test_the_mcp_query_note_is_no_longer_than_the_one_it_displaces(label, kw, edge):
    """The comparative bound (D3), stated where nothing data-dependent muddies it.

    On MCP both pointers are bare call-forms, so the whole composed note compares
    like for like: the `query` register may not be wordier than the `surface`
    sentence it displaces at the same inputs. The register is a per-answer cost
    (P15) — it earns its bytes by replacing prose, not by adding to it.
    """
    q = _query_note("mcp", edge=edge, **kw)
    s = assess_surface_recall(lead=None, short=None, edge=edge, semantic_ran=True, degraded_reason=None, scope=None, surface="mcp", **kw)[1]
    assert len(q) <= len(s), f"{label}: query note {len(q)} > surface note {len(s)}"


@pytest.mark.parametrize("edge", _EDGE_SHAPES)
@pytest.mark.parametrize("label,kw", _QUERY_CASES)
def test_the_cli_query_prose_is_no_longer_than_the_one_it_displaces(label, kw, edge):
    """The same bound on the CLI, with each side's pointer priced out — and why.

    The CLI redirect carries a selector and a `repr` because a response note is
    read wherever the caller was standing (D2), so its length is the caller's own
    workspace path — data, not prose this phase authored. Counting it would make
    the bound a property of someone's directory name: red on a long path, green on
    a short one, and governing nothing either way. So the comparison is over the
    authored prose, which is what the bound is actually about. The MCP row above
    is the unqualified statement; this one is the same claim with the one
    data-dependent term removed from both sides.
    """
    config = _StubConfig()
    q = _query_note("cli", config=config, edge=edge, **kw)
    s = assess_surface_recall(lead=None, short=None, edge=edge, semantic_ran=True, degraded_reason=None, scope=None, surface="cli", **kw)[1]
    q_prose = len(q) - (len(_query_pointer("cli", config)) if "mitos surface" in q else 0)
    s_prose = len(s) - (len(_SURFACE_POINTERS["cli"]["complete"])
                        if _SURFACE_POINTERS["cli"]["complete"] in s else 0)
    assert q_prose <= s_prose, f"{label}: query prose {q_prose} > surface prose {s_prose}"


def test_the_query_composer_has_no_degraded_and_no_scope_arm():
    """The two fences that are structural rather than asserted about behaviour.

    No `semantic_ran`: the composer has no degraded arm to author a degraded
    sentence in. No `scope`: a parameter that can only ever be None grows exactly
    the dead branches this verb's enumeration computes away — there is no fifth
    sentence to write. A band that is never None is the same fence on the return.
    """
    params = inspect.signature(assess_query_recall).parameters
    assert "semantic_ran" not in params and "scope" not in params
    assert "scope_counts" not in params
    for _label, kw in _QUERY_CASES:
        assert assess_query_recall(lead=None, short=None, edge=None, config=_StubConfig(), surface="cli", **kw)[0] is not None


def test_the_precedent_scan_pointer_exists_under_both_outer_keys():
    """G6 — a one-sided entry is a `KeyError` at runtime, not a failure at rest.

    The table's OUTER keys are the boundary axis and stay exactly two; a new
    pointer that lands under one of them ships the redirect on one surface and
    raises on the other, and nothing else in the tree pins the key set.
    """
    assert set(_SURFACE_POINTERS) == {"cli", "mcp"}
    assert set(_SURFACE_POINTERS["cli"]) == set(_SURFACE_POINTERS["mcp"])
    assert "precedent_scan" in _SURFACE_POINTERS["cli"]


# --------------------------------------------------------------------------- #
# T5 — register divergence. R11's only mechanical detector.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("label,kw", _QUERY_CASES)
@pytest.mark.parametrize("surface", ["cli", "mcp"])
def test_the_query_note_differs_from_the_surface_note_at_every_input(surface, label, kw):
    """Quantified over emitted NOTES, not over a symbol name.

    A half-register — fresh `"none"` sentences with `strong`/`weak` still
    inheriting `surface`'s — is symmetric across boundaries, so the parity
    fixtures agree, the call-site set is complete and every band value is correct.
    Equality against the sibling's note at the same inputs is the one thing that
    catches it, and it has to hold on all four or the rule has been read as being
    about the branch where the shipped wording is most obviously dangerous.
    """
    q = _query_note(surface, **kw)
    s = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, scope=None, surface=surface, **kw)[1]
    assert q != s, f"{label}/{surface}: the query register inherited surface's sentence"


def test_the_divergence_holds_on_a_note_a_driven_call_site_emitted(ws):
    """The same claim from the other end — four inequality rows over a composer
    nothing calls would pass just as well (T6's failure mode arriving in T5's home).

    Driven end to end on a real ranked answer: what `mitos query --json` actually
    put on the wire is not what `surface` would have said at those inputs.
    """
    config, m = ws
    _rec(m, "cache-strategy", scope=["db"])
    resp = _cli_query_json([{"slug": "cache-strategy", "score": 0.91}], ws)
    surface_note = assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=True, degraded_reason=None, top_score=0.91,
                                         result_count=1, scope=None, surface="cli")[1]
    assert resp["confidence"] == "strong"
    assert resp["note"] != surface_note
    # One hit at limit 5 over a one-point fetch: 2f's `ran_out` clause rides it.
    assert resp["note"] == _query_note("cli", config=config, top_score=0.91, result_count=1,
                                       short=ShortWindow("ran_out", 1, 5, 1, 0))


# --------------------------------------------------------------------------- #
# T6 — the call-site meta-test: a new read surface lands as a failing set
# comparison, never as silent non-coverage.
# --------------------------------------------------------------------------- #

def _band_composer_names():
    """Every public band composer in `recall.py`, DERIVED rather than listed.

    Keyed on the `assess_*` prefix, not on the string "confidence" (which this
    tree also uses for a numeric conflict-judgment field in four other modules)
    and not on "takes a `surface` keyword" (which over-collects
    `scope_filter_recovery` and `missing_graph_note`, neither of which bands
    anything). A third composer named outside the prefix reds the pin below rather
    than slipping the net silently.
    """
    from mitos import recall
    return {name for name, obj in vars(recall).items()
            if name.startswith("assess_") and callable(obj)}


def _band_call_sites():
    """(module, enclosing function) for every band-composer call in the package.

    An AST walk rather than a grep: the enclosing function is the half that
    matters, and a grep cannot see it. A call inside a nested def would be
    attributed to both, which over-collects rather than under-collects — there are
    none on this path today.

    Swept over **every** `mitos/*.py`, not just the two surfaces that band today.
    The claim this pin is here to make is "a new read surface without a band lands
    red", and a two-file walk makes that claim only about those two files — a
    surface introduced in a new module would go unwatched by the very test written
    to watch for it. The set is identical either way at this tip, which is exactly
    when the wider sweep is free to take.
    """
    from mitos import recall
    names = _band_composer_names()
    pkg = os.path.dirname(recall.__file__)
    sites = set()
    for entry in sorted(os.listdir(pkg)):
        if not entry.endswith(".py"):
            continue
        with open(os.path.join(pkg, entry), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id in names):
                    sites.add((entry[:-3], fn.name))
    return sites


def test_the_band_composer_set_is_exactly_the_two_we_know_about():
    """The assertion that closes the hole in the call-site pin below.

    The walk keys on a set derived from `recall.py`; if a third composer lands
    without a call-site row, the walk simply never looks for it. Pinning the
    derived set makes that a red here, with the message naming what is owed.
    """
    assert _band_composer_names() == {"assess_surface_recall", "assess_query_recall"}, (
        "a new band composer landed in recall.py — add its call sites to the pin "
        "below, or the meta-test goes on watching only the ones it knew about"
    )


def test_every_band_composer_call_site_is_accounted_for():
    """A set comparison, so an unbanded new read surface lands red.

    Keyed on the composer SYMBOLS, never on the name `assess_surface_recall`:
    under the split build that name holds only `surface`'s two call sites, so a
    row pinning it passes green while `query`'s go unwatched — a vacuous frame,
    not a vacuous row.
    """
    assert _band_call_sites() == {
        ("cli", "cmd_surface"), ("cli", "cmd_query"),
        ("mcp_server", "surface_decisions"), ("mcp_server", "query_decisions"),
    }


# --------------------------------------------------------------------------- #
# T7 — the two structural-unreachability pins.
#
# Each is a claim about the CALLERS that no shape of the policy can assert about
# itself: a later refactor that routed a degraded read into the band would redden
# nothing while shipping a confidence label over an answer that was never ranked.
# --------------------------------------------------------------------------- #

class _Boom:
    """A vector store that fails mid-query — the generic degraded route."""

    def query(self, vector, limit=5, *, with_payload=True):
        raise RuntimeError("qdrant fell over")

    def delete_points(self, point_ids):
        raise RuntimeError("qdrant fell over")


class _MissingCollection:
    """Qdrant is up and says that collection does not exist."""

    def query(self, vector, limit=5, *, with_payload=True):
        raise CollectionMissingError(
            "Qdrant collection 'mitos-tmp-absent' does not exist.",
            collection="mitos-tmp-absent",
        )

    def delete_points(self, point_ids):
        raise CollectionMissingError(
            "Qdrant collection 'mitos-tmp-absent' does not exist.",
            collection="mitos-tmp-absent",
        )


class _NoMatches:
    """Present, and simply returns nothing — the healthy-empty arm's other shape."""

    def query(self, vector, limit=5, *, with_payload=True):
        return []

    def delete_points(self, point_ids):
        pass  # nothing is stored, so nothing is left to delete


def _sentinel(*_args, **_kwargs):
    raise AssertionError("a degraded route reached the band composer")


def test_no_degraded_cli_query_route_reaches_the_band_composer(ws):
    """All four `cmd_query` degraded routes, re-derived from the source rather than
    inherited: store construction, no providers, the is-a-gap collection arm, and
    the generic mid-query fault.

    Patched as imported into `mitos.cli` — the binding is module-local, so a patch
    on `mitos.recall` would not intercept.
    """
    config, m = ws
    _rec(m, "settled", scope=["db"])                  # populated → the collection arm IS a gap
    store = GraphStore(config.db_path, read_only=True)
    routes = [
        ("store construction", None),
        ("no providers", _StubManager(store, None, None)),
        ("absent collection over a populated graph",
         _StubManager(store, _FakeEmbed(), _MissingCollection())),
        ("mid-query fault", _StubManager(store, _FakeEmbed(), _Boom())),
    ]
    with patch("mitos.cli.assess_query_recall", _sentinel):
        for label, stub in routes:
            buf = io.StringIO()
            if stub is None:
                mm = patch("mitos.cli.MitosSyncManager", side_effect=RuntimeError("pre-V1a"))
            else:
                mm = patch("mitos.cli.MitosSyncManager", return_value=stub)
            with mm, redirect_stdout(buf):
                cmd_query(config, "a claim that is not any slug", as_json=True)
            assert json.loads(buf.getvalue())["degraded"] == "lexical", label


def test_no_degraded_mcp_query_route_reaches_the_band_composer(ws):
    """The same four on `query_decisions`, whose degraded routes are its own."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "settled", scope=["db"])
    store = GraphStore(config.db_path, read_only=True)
    routes = [
        ("component construction", None),
        ("no providers", (store, None, None)),
        ("absent collection over a populated graph",
         (store, _FakeEmbed(), _MissingCollection())),
        ("mid-query fault", (store, _FakeEmbed(), _Boom())),
    ]
    with patch.object(mcp_server, "assess_query_recall", _sentinel):
        for label, comps in routes:
            kw = ({"side_effect": RuntimeError("pre-V1a")} if comps is None
                  else {"return_value": comps})
            with patch.object(mcp_server, "get_workspace_components", **kw):
                out = json.loads(mcp_server.query_decisions(
                    "a claim that is not any slug", project=config.workspace_dir))
            assert out["degraded"] == "lexical", label


def test_neither_query_surface_takes_a_scope():
    """The fence is in the signature, and the pin is what catches a verb growing one.

    `scope` arrives structurally None at both call sites, which is what makes the
    four-sentence enumeration a computation rather than a judgement. A later
    `scope` parameter on either verb reopens every branch the register left out.
    """
    from mitos import mcp_server
    assert "scope" not in inspect.signature(cmd_query).parameters
    assert "scope" not in inspect.signature(mcp_server.query_decisions).parameters


# --------------------------------------------------------------------------- #
# T8 — the band end to end: four surfaces × exits × encodings.
#
# The module had no CLI TEXT frame at all before this — 21 policy rows and one
# `--json` harness, so every text obligation would have been proved by nothing
# while reading as covered. The frame is lifted from `test_modifier_surfacing`.
# --------------------------------------------------------------------------- #

class _StubManager:
    """Stub MitosSyncManager: real read store + injected embed/vector providers."""

    def __init__(self, store, embed_provider, vector_store):
        self.store = store
        self.embed_provider = embed_provider
        self.vector_store = vector_store


def _cli_query(matches, ws, query="a claim that is not any slug", as_json=False,
               config=None, full_top=None, brief=False, limit=None):
    """Drives `cmd_query` end to end and returns the raw captured stdout.

    `matches=None` drives the degraded (no embed/vector) path; a vector-store
    instance is passed through as-is, so a fault stub drives its own route.
    `config` overrides the `ws` one (a keyed config must be built after its key).
    """
    from mitos import cli
    config = config or ws[0]
    store = GraphStore(config.db_path, read_only=True)
    if matches is None:
        stub = _StubManager(store, None, None)
    elif isinstance(matches, list):
        stub = _StubManager(store, _FakeEmbed(), _FakeVector(matches))
    else:
        stub = _StubManager(store, _FakeEmbed(), matches)
    buf = io.StringIO()
    with patch.object(cli, "MitosSyncManager", return_value=stub):
        with redirect_stdout(buf):
            cmd_query(config, query, as_json=as_json, brief=brief, full_top=full_top,
                      limit=limit)
    return buf.getvalue()


def _cli_query_json(matches, ws, query="a claim that is not any slug", full_top=None,
                    brief=False, limit=None):
    return json.loads(_cli_query(matches, ws, query=query, as_json=True,
                                 full_top=full_top, brief=brief, limit=limit))


def _mcp_query(matches, ws, query="a claim that is not any slug", full_top=None,
               brief=False, limit=None):
    from mitos import mcp_server
    config, _ = ws
    store = GraphStore(config.db_path, read_only=True)
    vector = _FakeVector(matches, store=store) if isinstance(matches, list) else matches
    embed = None if matches is None else _FakeEmbed()
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, embed, vector)):
        kw = {} if limit is None else {"limit": limit}
        return json.loads(mcp_server.query_decisions(
            query, brief=brief, project=config.workspace_dir, full_top=full_top,
            **kw))


_BAND_LINE_PREFIX = "⚠ confidence:"


def _band_lines(out):
    """The band LINE only — never a bare `⚠` search, which the modifier marker on a
    ranked text render already satisfies."""
    return [ln for ln in out.splitlines() if ln.startswith(_BAND_LINE_PREFIX)]


def _cli_surface_text(matches, ws, query="some claim", scope=None, full_top=None,
                      limit=None):
    """`cmd_surface`'s text render — the module drove only its `--json` twin."""
    from mitos import cli
    config, _ = ws
    manager = MitosSyncManager(config)
    manager.embed_provider = _FakeEmbed()
    manager.vector_store = _FakeVector(matches)
    buf = io.StringIO()
    with patch.object(cli, "MitosSyncManager", return_value=manager):
        with redirect_stdout(buf):
            cmd_surface(config, query, scope=scope, full_top=full_top, limit=limit)
    return buf.getvalue()


def test_strong_prints_no_band_line_on_either_verb(ws):
    """The shipped `surface` half of the same rule, stated rather than assumed —
    it is the baseline the new `query` line was placed against, and no test in the
    tree pinned it."""
    config, m = ws
    _rec(m, "cache-strategy", scope=["db"])
    matches = [{"slug": "cache-strategy", "score": 0.91}]
    assert _band_lines(_cli_surface_text(matches, ws)) == []
    assert _band_lines(_cli_query(matches, ws)) == []


def test_cli_query_ranked_strong_carries_the_note_and_no_band_line(ws):
    """`strong` prints no band line on either verb — the quantifier is over
    BRANCHES, never over band labels. A label-axis reading composes a `strong`
    line carrying `⚠` over a good result, which reads as a warning about an answer
    that is fine."""
    config, m = ws
    _rec(m, "cache-strategy", scope=["db"])
    out = _cli_query([{"slug": "cache-strategy", "score": 0.91}], ws)
    assert "cache-strategy" in out
    assert _band_lines(out) == []
    assert _query_note("cli", config=config, top_score=0.91, result_count=1) in out


@pytest.mark.parametrize("score,band", [(0.61, "weak"), (0.41, "none")])
def test_cli_query_ranked_prints_the_band_line_and_the_note(ws, score, band):
    config, m = ws
    _rec(m, "cache-strategy", scope=["db"])
    out = _cli_query([{"slug": "cache-strategy", "score": score}], ws)
    lines = _band_lines(out)
    assert len(lines) == 1 and band in lines[0]
    assert _query_note("cli", config=config, top_score=score, result_count=1) in out


def test_cli_query_genuine_miss_carries_the_band_on_both_encodings(ws):
    """The text branch gets the line AND the note; its `--json` twin gets
    `confidence` and the same note. `cmd_surface` prints its band line on the
    ranked path only, so a placement-and-all copy would leave this branch carrying
    a note and no band while its JSON twin carried a label — the encoding axis
    reopened inside one verb."""
    config, m = ws
    _rec(m, "unrelated", scope=["x"])
    out = _cli_query([], ws)
    assert "No matching decisions found." in out
    assert len(_band_lines(out)) == 1 and "none" in _band_lines(out)[0]
    expected = _query_note("cli", config=config, top_score=None, result_count=0)
    assert expected in out
    resp = _cli_query_json([], ws)
    assert resp["confidence"] == "none" and resp["note"] == expected


def test_cli_query_blackout_suppresses_the_note_and_keeps_the_label(ws):
    """The override row's deliberate shape: text carries no band label beside a
    `--json` twin that still carries `confidence`. An override reassigns the NOTE
    and leaves the band standing — the band is a fact about the ranking that a
    diagnosis about the graveyard does not contradict."""
    config, m = ws
    _rec(m, "dead-v1", scope=["x"])
    _rec(m, "dead-v2", scope=["x"], supersedes="dead-v1")
    matches = [{"slug": "dead-v1", "score": 0.9}]

    out = _cli_query(matches, ws)
    assert _band_lines(out) == []
    assert "dead-v1" in out and "dead-v2" in out
    assert _query_note("cli", config=config, top_score=None, result_count=0) not in out

    resp = _cli_query_json(matches, ws)
    assert resp["confidence"] == "none"
    assert resp["matches"] == [] and resp["all_superseded"]
    # The 4th blackout_note call site: `--json` carried the handles alone until 3a
    # while the text branch had printed this note since 2d. 2f moved the note to
    # recall and worded its heal per boundary: the CLI names the history read with
    # the caller's selector (the pointer table's `history`).
    from mitos.shell_quote import shell_flag
    history = _SURFACE_POINTERS["cli"]["history"].format(
        selector=shell_flag("-p", config.project))
    assert "superseded" in resp["note"] and history in resp["note"]


def test_mcp_query_blackout_carries_the_note_too(ws):
    """The 5th call site, and it closes a different divergence: there is no text
    surface here, so what withholding it would create is a CLI⇄MCP split — and an
    MCP⇄MCP one, since `surface_decisions` already emits this note on this
    surface."""
    config, m = ws
    _rec(m, "dead-v1", scope=["x"])
    _rec(m, "dead-v2", scope=["x"], supersedes="dead-v1")
    out = _mcp_query([{"slug": "dead-v1", "score": 0.9}], ws)
    assert out["confidence"] == "none"
    assert out["matches"] == [] and out["all_superseded"]
    # 2f: the MCP heal is its own pointer, a read tool, never a shell command.
    assert "superseded" in out["note"] and "list_decisions(state='all')" in out["note"]
    assert "mitos " not in out["note"]


@pytest.mark.parametrize("vector_cls", [_NoMatches, _MissingCollection])
def test_both_mcp_query_empty_envelopes_take_the_band(ws, vector_cls):
    """`cmd_query` builds ONE envelope where this tool builds TWO — the ranked one
    (semantic ran and matched nothing) and the healthy-empty one constructed in the
    `CollectionMissingError` arm. A cross-product written from either surface's
    shape under-counts the other's."""
    config, m = ws
    out = _mcp_query(vector_cls(), ws)
    assert out["matches"] == []
    assert out["confidence"] == "none"
    assert out["note"] == _query_note("mcp", config=config, top_score=None, result_count=0)


def test_the_healthy_empty_arm_bands_where_its_degraded_twin_does_not(ws):
    """The fork's two arms, as separate rows: same `CollectionMissingError`, ten
    lines apart. Over an EMPTY graph the absence is the empty index and the read is
    healthy — it bands. Over a POPULATED one it is a real hole and degrades — no
    `confidence` key at all."""
    config, m = ws
    healthy = _mcp_query(_MissingCollection(), ws)          # empty graph
    assert healthy["confidence"] == "none" and "degraded" not in healthy

    _rec(m, "settled", scope=["db"])                        # now populated
    degraded = _mcp_query(_MissingCollection(), ws)
    assert degraded["degraded"] == "lexical"
    assert "confidence" not in degraded


@pytest.mark.parametrize("driver", ["cli", "mcp"])
def test_no_degraded_query_exit_carries_a_confidence_key(ws, driver):
    config, m = ws
    _rec(m, "settled", scope=["db"])
    out = (_cli_query_json(None, ws) if driver == "cli" else _mcp_query(None, ws))
    assert out["degraded"] == "lexical"
    assert "confidence" not in out and _BAND_LINE_PREFIX not in json.dumps(out)


def test_cli_surface_degraded_still_carries_no_confidence(ws):
    """The unchanged baseline, stated rather than assumed: `cmd_surface` takes a
    zero-byte diff in this phase, and it is the equality baseline the divergence
    rows compare against."""
    config, m = ws
    _rec(m, "settled", scope=["db"])
    assert "confidence" not in _cli_surface_json(None, ws, scope="db")


def test_the_exact_slug_exit_still_carries_no_band(ws):
    """2a's exit stamps provenance and never confidence: a band on a named handle
    reads as "this decision is doubtful", which is a different claim entirely."""
    config, m = ws
    _rec(m, "cache-strategy", scope=["db"])
    out = _mcp_query([], ws, query="cache-strategy")
    assert out["slug"] == "cache-strategy"
    assert "confidence" not in out and "note" not in out


def test_the_cli_query_has_no_exact_slug_branch_to_band(ws):
    """The one legitimate CLI⇄MCP asymmetry, shown as an ABSENCE rather than built
    as a second half: `mitos query` stays semantic-only by ADR
    `cli-query-stays-semantic-not-dereference-twin`, so a real slug takes the
    ordinary ranked path and bands like any other lookup."""
    config, m = ws
    _rec(m, "cache-strategy", scope=["db"])
    resp = _cli_query_json([{"slug": "cache-strategy", "score": 0.91}], ws,
                           query="cache-strategy")
    assert "state" not in resp and resp["matches"][0]["slug"] == "cache-strategy"
    assert resp["confidence"] == "strong"


def test_cli_and_mcp_query_notes_differ_only_in_the_call_form(ws):
    """The cross-surface data contract, stated as values: one classification, one
    register, each boundary's own pointer."""
    config, m = ws
    _rec(m, "cache-strategy", scope=["db"])
    matches = [{"slug": "cache-strategy", "score": 0.61}]
    cli_resp = _cli_query_json(matches, ws)
    mcp_resp = _mcp_query(matches, ws)
    assert cli_resp["confidence"] == mcp_resp["confidence"] == "weak"
    assert cli_resp["note"] != mcp_resp["note"]
    # The short window's `limit` spelling is a call form too (2f's clause rides
    # this one-hit window), so it is swapped beside the redirect.
    assert (cli_resp["note"].replace(_query_pointer("cli", config),
                                     _query_pointer("mcp", config))
            .replace("--limit 5", "limit=5")
            == mcp_resp["note"])


# --- the unbuilt-graph override, on both encodings ------------------------- #

_CLONE_ENTRY = """
### clone-entry-one

**Decided:** A clone carries the corpus but never the graph.
**Rejected:** Committing the binary graph — it is derivative.
**Scope:** clone
"""


@pytest.fixture
def cloned(offline) -> Iterator[Tuple[MitosConfig, None]]:
    """A workspace with corpus entries and a graph holding no nodes.

    Seeded BY HAND: `mitos sync` commits nothing without a `GEMINI_API_KEY` (G8),
    and `record` commits to the graph — the one thing this fixture must not have.
    Shaped as a `(config, manager)` pair so the `ws`-keyed helpers above drive it
    unchanged.
    """
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    cmd_init(config)
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(_CLONE_ENTRY)
    os.remove(config.db_path)
    assert GraphStore(config.db_path).graph_fingerprint()[0] == 0
    yield config, None
    shutil.rmtree(tmp, ignore_errors=True)


def test_the_unbuilt_override_takes_the_text_branch_that_appends(cloned):
    """The one text branch where an override and a band co-occur, and the branch
    APPENDS where the envelope assigns — so the suppression is code, not something
    the copy carries. Left to the copy, a miss over an unbuilt graph would print
    the band note whose redirect sends the caller to `surface`, which answers just
    as empty over that same unbuilt graph: a turn spent one line above the heal.
    """
    config, _ = cloned
    out = _cli_query([], cloned)
    assert "No matching decisions found." in out
    assert "graph is unbuilt" in out and "mitos rebuild -p" in out
    assert _band_lines(out) == []
    assert _query_note("cli", config=config, top_score=None, result_count=0) not in out


@pytest.mark.parametrize("driver", ["cli", "mcp"])
def test_the_unbuilt_override_keeps_the_label_on_the_machine_encodings(cloned, driver):
    """Precedence is confidence note < blackout < unbuilt, and it is the NOTE that
    yields. The band note is assigned first and the overrides reassign it — the
    same straight-line order `cmd_surface` has run since W31."""
    out = (_cli_query_json([], cloned) if driver == "cli" else _mcp_query([], cloned))
    assert out["confidence"] == "none"
    # Both registers name the rebuild: the CLI as a command, MCP as a fact.
    assert "graph is unbuilt" in out["note"] and "rebuild" in out["note"]
    assert "reconcile" not in out["note"]


# --------------------------------------------------------------------------- #
# T9 — the band reads off the SURFACED list, not the raw vector-store return.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", ["cli", "mcp"])
def test_the_band_is_read_off_the_surfaced_list_not_the_raw_return(ws, driver):
    """R12's only detector, and it needs a ranking the filter actually thins.

    `dead-v1` scores 0.95 and is superseded-filtered out; `live-one` survives at
    0.62. Wired off the raw return the band reads `strong` — a confident verdict
    on a match the caller never saw. Wired off the surfaced list it reads `weak`.
    The same fixture catches the fail-open half: a `top_score` no loop ever raised
    is None, which `_classify_recall` admits into `strong` on a non-empty set. A
    fixture whose every hit survives cannot tell the two builds apart, and neither
    shape reddens anything else in the tree.
    """
    config, m = ws
    _rec(m, "live-one", scope=["x"])
    _rec(m, "dead-v1", scope=["x"])
    _rec(m, "dead-v2", scope=["x"], supersedes="dead-v1")
    matches = [{"slug": "dead-v1", "score": 0.95}, {"slug": "live-one", "score": 0.62}]

    if driver == "cli":
        resp = _cli_query_json(matches, ws)
        text = _cli_query(matches, ws)
        assert len(_band_lines(text)) == 1 and "weak" in _band_lines(text)[0]
    else:
        resp = _mcp_query(matches, ws)
    assert [d["slug"] for d in resp["matches"]] == ["live-one"]
    assert resp["confidence"] == "weak"
    assert "0.62" in resp["note"] and "0.95" not in resp["note"]


# --------------------------------------------------------------------------- #
# Phase 4a — ranked reads return decisions only.
#
# Sync embeds open questions into the same collection, `get_node_by_slug` is
# kind-agnostic and `get_node_state` calls a parked OQ `active` — so an OQ in the
# window reached the payload shaper, raised `KeyError` on `core_axiom`, and the
# loop's blanket catch turned a healthy read into degraded recall (or, on
# `surface` at rank 2, a cut list with no band). The record path's gather learned
# this first (`conflict.gather_candidates`). Every fake point here carries only
# `{"slug", "score"}`: the rows pass because the STORE says `open_question`
# (D1), never because the point does.
# --------------------------------------------------------------------------- #

def _seed_4a(ws):
    """Two decisions and one parked OQ, with the exact trap asserted up front."""
    config, m = ws
    _rec(m, "dec-a", scope=["x"])
    _rec(m, "dec-b", scope=["x"])
    _commit_oq(m.store, "oq-parked", scope=["x"])
    oq = m.store.get_node_by_slug("oq-parked")
    assert oq is not None and oq["kind"] == "open_question"
    assert m.store.get_node_state(oq["id"]) == "active"
    assert "core_axiom" not in oq


def _seed_retired_oq(m):
    """G5: `oq-old` superseded by `oq-new` — a legal same-kind lineage."""
    _commit_oq(m.store, "oq-old", scope=["x"])
    e = ParsedEntry("open_question", "oq-new", 1, 5)
    e.topic = "Topic for oq-new"
    e.questions_raised = ["What about oq-new?"]
    e.scope = ["x"]
    e.supersedes = ["oq-old"]
    m.store.commit_parsed_entry(e)
    assert m.store.get_node_by_slug("oq-old") is None
    assert m.store.resolve_slug_kinds("oq-old") == ["open_question"]


# driver → (call, list key, verb, surface)
_4A_DRIVERS = {
    "mcp-surface": (_surface_with, "active_decisions", "surface", "mcp"),
    "cli-surface": (_cli_surface_json, "active_decisions", "surface", "cli"),
    "mcp-query": (_mcp_query, "matches", "query", "mcp"),
    "cli-query": (_cli_query_json, "matches", "query", "cli"),
}


def _4a_band(verb, surface, config, top_score, n, short=None):
    """The band the surfaced decisions alone earn, from the policy itself.

    ``short`` is 2f's: a window under ``limit`` over a fetch that returned points
    ends its note with the short-window clause, so a row driving one passes it."""
    if verb == "surface":
        return assess_surface_recall(lead=None, short=short, edge=None, semantic_ran=True, degraded_reason=None, top_score=top_score,
                                     result_count=n, scope=None, surface=surface)
    return assess_query_recall(lead=None, short=short, edge=None, top_score=top_score, result_count=n, config=config,
                               surface=surface)


@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
@pytest.mark.parametrize("oq_rank", [1, 2])
def test_an_open_question_in_the_window_is_skipped_and_the_hits_survive(ws, driver, oq_rank):
    """Rows 1–2. At rank 1 today every surface degrades to lexical (`KeyError`).

    At rank 2 `surface` does not degrade: it keeps the rank-1 decision, drops the
    one below, and writes no `confidence` at all (G2) — so the row asserts the
    decision BELOW the OQ and the band, not merely "no `degraded` key".
    """
    config, _ = ws
    _seed_4a(ws)
    call, key, verb, surface = _4A_DRIVERS[driver]
    decs = [{"slug": "dec-a", "score": 0.8}, {"slug": "dec-b", "score": 0.7}]
    matches = list(decs)
    matches.insert(oq_rank - 1, {"slug": "oq-parked", "score": 0.85 if oq_rank == 1 else 0.75})
    resp = call(matches, ws)
    assert "degraded" not in resp
    assert [d["slug"] for d in resp[key]] == ["dec-a", "dec-b"]
    # Two decisions at limit 5 over a three-point fetch: 2f's `ran_out` clause.
    short = ShortWindow("ran_out", 2, 5, len(matches), 0)
    assert (resp["confidence"], resp["note"]) == _4a_band(verb, surface, config, 0.8, 2,
                                                          short=short)


@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
def test_the_band_is_read_off_decisions_never_off_a_skipped_open_question(ws, driver):
    """Row 3 (G3). The OQ scores strong at rank 1; both decisions sit below strong.

    A fix that skips the OQ AFTER raising `top_score` passes rows 1–2 and reads
    `strong` here — a verdict on a point the caller never saw. The fixture proves
    it can tell: the band with the OQ's score counted must differ from the right one.
    """
    config, _ = ws
    _seed_4a(ws)
    call, key, verb, surface = _4A_DRIVERS[driver]
    oq_score = min(1.0, SURFACE_STRONG_THRESHOLD + 0.1)
    hi = (SURFACE_STRONG_THRESHOLD + SURFACE_WEAK_THRESHOLD) / 2
    lo = SURFACE_WEAK_THRESHOLD - 0.05
    matches = [{"slug": "oq-parked", "score": oq_score},
               {"slug": "dec-a", "score": hi}, {"slug": "dec-b", "score": lo}]
    # Two decisions at limit 5 over a three-point fetch: 2f's `ran_out` clause.
    right = _4a_band(verb, surface, config, hi, 2,
                     short=ShortWindow("ran_out", 2, 5, len(matches), 0))
    assert right[0] != _4a_band(verb, surface, config, oq_score, 2)[0]
    resp = call(matches, ws)
    assert "degraded" not in resp
    assert [d["slug"] for d in resp[key]] == ["dec-a", "dec-b"]
    assert (resp["confidence"], resp["note"]) == right


@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
def test_a_retired_open_question_is_never_offered_as_a_retired_precedent(ws, driver):
    """Row 4. Alone in the window — beside a live decision the blackout never fires.

    Before the screen, the slug-keyed handle builder resolved the superseded
    OQ's slug without reading its kind, and the blackout presented a parked
    question as a settled decision's retired precedent. Since 2c the walk skips
    it as a non-decision before it can be retired. The positive control
    (a superseded DECISION alone) proves the retired arm is screened, not disabled.
    """
    config, m = ws
    _seed_4a(ws)
    _seed_retired_oq(m)
    _rec(m, "dead-v1", scope=["x"])
    _rec(m, "dead-v2", scope=["x"], supersedes="dead-v1")
    call, key, verb, surface = _4A_DRIVERS[driver]

    resp = call([{"slug": "oq-old", "score": 0.9}], ws)
    assert "all_superseded" not in resp and "degraded" not in resp
    assert resp[key] == []
    assert (resp["confidence"], resp["note"]) == _4a_band(verb, surface, config, None, 0)

    control = call([{"slug": "dead-v1", "score": 0.9}], ws)
    assert [h["slug"] for h in control["all_superseded"]] == ["dead-v1"]


@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
def test_beside_a_retired_decision_only_the_decision_is_a_retired_handle(ws, driver):
    """Row 5 — pins D2's boundary. Green before the fix on the decision's side
    (its handle was always kept); the OQ's handle is what it removes."""
    _, m = ws
    _seed_4a(ws)
    _seed_retired_oq(m)
    _rec(m, "dead-v1", scope=["x"])
    _rec(m, "dead-v2", scope=["x"], supersedes="dead-v1")
    call, key, _, _ = _4A_DRIVERS[driver]
    resp = call([{"slug": "oq-old", "score": 0.9}, {"slug": "dead-v1", "score": 0.8}], ws)
    assert resp[key] == []
    assert [h["slug"] for h in resp["all_superseded"]] == ["dead-v1"]


def test_an_open_questions_exact_slug_falls_through_to_the_ranked_search(ws):
    """Row 6 (MCP `query_decisions` only — the CLI twin has no exact-slug arm).

    The OQ's slug is not a dereference: it goes on to the ranked search by its
    kind, and the ranked window (where the OQ also sits) lists decisions only.
    """
    config, _ = ws
    _seed_4a(ws)
    matches = [{"slug": "oq-parked", "score": 0.9}, {"slug": "dec-a", "score": 0.8}]
    resp = _mcp_query(matches, ws, query="oq-parked")
    assert "axiom" not in resp and "degraded" not in resp
    assert [d["slug"] for d in resp["matches"]] == ["dec-a"]
    assert (resp["confidence"], resp["note"]) == _4a_band(
        "query", "mcp", config, 0.8, 1, short=ShortWindow("ran_out", 1, 5, 2, 0))


def test_an_exact_slug_composition_fault_is_loud_not_a_silent_ranked_search(ws, monkeypatch):
    """Row 7 — D3's proof, and the only row that can tell "falls through by kind"
    from "falls through by exception".

    A DECISION-kind node missing `core_axiom` is a defect in the exact-slug
    composition. Kept inside the old `except Exception: pass`, it slid silently
    into the ranked search and returned a healthy-looking envelope; with the
    `try` narrowed to the store reads, it raises. Only the planted slug is
    malformed; the plant is live because the exact-slug branch is the one place
    left that looks a node up by slug (the ranked read resolves by point id).
    """
    from mitos import mcp_server
    config, _ = ws
    _seed_4a(ws)
    real = GraphStore.get_node_by_slug
    dec_a_id = GraphStore(config.db_path, read_only=True).get_node_by_slug("dec-a")["id"]

    def planted(self, slug):
        if slug == "planted-slug":
            return {"id": dec_a_id, "slug": "planted-slug", "kind": "decision",
                    "rejected_paths": [], "scope": ["x"]}
        return real(self, slug)

    monkeypatch.setattr(GraphStore, "get_node_by_slug", planted)
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, _FakeEmbed(),
                                    _FakeVector([{"slug": "dec-b", "score": 0.8}]))):
        with pytest.raises(KeyError):
            mcp_server.query_decisions("planted-slug", project=config.workspace_dir)


@pytest.mark.parametrize("verb", ["surface", "query"])
def test_text_renders_rank_the_decisions_past_an_open_question(ws, verb):
    """Row 8. Unscoped on purpose: a scoped degraded dump prints `(score 1.000)`
    lines too. The degraded lexical render prints no score at all, so a scored
    ranked line is the marker — never the degraded note's wording (7a rewrites it)."""
    _seed_4a(ws)
    matches = [{"slug": "oq-parked", "score": 0.85},
               {"slug": "dec-a", "score": 0.8}, {"slug": "dec-b", "score": 0.7}]
    out = _cli_surface_text(matches, ws) if verb == "surface" else _cli_query(matches, ws)
    assert "1. dec-a  (score 0.800)" in out
    assert "2. dec-b  (score 0.700)" in out
    assert "oq-parked" not in out


def test_every_ranked_read_screens_kind_through_the_walk():
    """Stretch: the structural twin of the matrix above, moved by 2c to where the
    screen now lives. The four ranked surfaces resolve no slug in a loop — each calls
    the shared walk — and the walk's `fill` compares `kind` before it reads
    `computed_state` (a resolved open question reports `active`). A fifth surface
    that kept a private slug loop fails the first assertion."""
    import mitos.cli
    import mitos.mcp_server
    import mitos.retrieval_walk

    surfaces = {"surface_decisions", "query_decisions", "cmd_surface", "cmd_query"}
    calls_walk = set()
    for module in (mitos.mcp_server, mitos.cli):
        for fn in ast.walk(ast.parse(inspect.getsource(module))):
            if not isinstance(fn, ast.FunctionDef):
                continue
            for loop in (n for n in ast.walk(fn) if isinstance(n, ast.For)):
                calls = {c.func.attr for c in ast.walk(loop)
                         if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)}
                assert "get_node_by_slug" not in calls, f"{fn.name} resolves hits by slug"
            if any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                   and c.func.id == "walk_ranked" for c in ast.walk(fn)):
                calls_walk.add(fn.name)
    assert calls_walk == surfaces

    fill = next(n for n in ast.walk(ast.parse(inspect.getsource(mitos.retrieval_walk)))
                if isinstance(n, ast.FunctionDef) and n.name == "fill")

    def first_line(key):
        return min(c.lineno for c in ast.walk(fill)
                   if isinstance(c, ast.Subscript) and isinstance(c.slice, ast.Constant)
                   and c.slice.value == key)

    assert first_line("kind") < first_line("computed_state")


# --------------------------------------------------------------------------- #
# Phase 3g2 — the standing check notice on every `surface` exit, never `query`'s.
#
# Seeding goes through 3g1's real writers in `test_commit_gate` (imported
# function-locally, as the receipt rows do). Every row sets its own key: the
# `offline` fixture strips it, so every row above this section is the keyless
# regression net for "a healthy or keyless surface pays zero bytes".
# --------------------------------------------------------------------------- #

import shlex  # noqa: E402
import sys  # noqa: E402
from contextlib import redirect_stderr  # noqa: E402

from mitos.check_notice import check_notice_line  # noqa: E402
from mitos.telemetry import (ATTEMPT_COULD_NOT_COMPLETE, ATTEMPT_NO_NEW_FINDINGS,  # noqa: E402
                             ATTEMPT_SPEND_NOT_AUTHORIZED)


def _gate():
    """3g1's seeding and parsing helpers (`_seed_attempt`, `_keyed`, `_SHOWN`, …)."""
    import test_commit_gate
    return test_commit_gate


class _PlantedFault(Exception):
    """A raise planted at a call site; its class must come back out unchanged."""


_HIT = [{"slug": "settled", "score": 0.91}]

# Every route into a `surface` exit, as (components, scope). `None` components
# means store construction raises (the no-graph route).
_ROUTES = ("semantic", "scope_dump", "lexical_empty", "lexical_no_graph")


def _components(route, store):
    return {
        "semantic": (store, _FakeEmbed(), _FakeVector(_HIT)),
        "no_matches": (store, _FakeEmbed(), _NoMatches()),
        "collection_missing": (store, _FakeEmbed(), _MissingCollection()),
        "scope_dump": (store, None, None),
        "lexical_empty": (store, None, None),
        "lexical_no_graph": None,
    }[route]


def _scope(route):
    return "db" if route == "scope_dump" else None


def _mcp_surface(config, route, query="some claim"):
    """`surface_decisions` on one route; returns (envelope, stderr)."""
    from mitos import mcp_server
    comps = _components(route, GraphStore(config.db_path, read_only=True))
    kw = ({"side_effect": RuntimeError("pre-V1a")} if comps is None
          else {"return_value": comps})
    err = io.StringIO()
    with patch.object(mcp_server, "get_workspace_components", **kw), redirect_stderr(err):
        out = mcp_server.surface_decisions(query, scope=_scope(route),
                                           project=config.workspace_dir)
    return json.loads(out), err.getvalue()


def _cli_surface(config, route, *, as_json, query="some claim", argv0="mitos"):
    """`cmd_surface` on one route with `argv[0]` pinned; returns (stdout, stderr).

    `config` is the caller's (keyed or not): the notice reads only its env.
    """
    from mitos import cli
    comps = _components(route, GraphStore(config.db_path, read_only=True))
    mm = (patch.object(cli, "MitosSyncManager", side_effect=RuntimeError("pre-V1a"))
          if comps is None else
          patch.object(cli, "MitosSyncManager", return_value=_StubManager(*comps)))
    out, err = io.StringIO(), io.StringIO()
    with mm, patch.object(sys, "argv", [argv0]), redirect_stdout(out), redirect_stderr(err):
        cmd_surface(config, query, scope=_scope(route), as_json=as_json)
    return (json.loads(out.getvalue()) if as_json else out.getvalue()), err.getvalue()


def _two_decisions(ws):
    """Commits `settled` and `other` (scope `db`); returns their node ids."""
    config, m = ws
    _rec(m, "settled", scope=["db"])
    _rec(m, "other", scope=["db"])
    return tuple(m.store.get_node_by_slug(s)["id"] for s in ("settled", "other"))


def _seed_resolvable(ws):
    """A `could_not_complete` record holding one pair of real node ids."""
    config, _ = ws
    ids = _two_decisions(ws)
    _gate()._seed_attempt(config, ATTEMPT_COULD_NOT_COMPLETE, tokens=("sweep",),
                          pairs=(ids,))
    return ids


def _assert_route_is(route, envelope):
    """Guards against a vacuous row: the envelope is the exit the row names."""
    if route in ("lexical_empty", "lexical_no_graph"):
        assert envelope["degraded"] == "lexical"
    else:
        assert "degraded" not in envelope and envelope["active_decisions"], route
        # Semantic ranking ran (a band) — or it did not, and this is the scope dump.
        assert ("confidence" in envelope) == (route == "semantic"), route


def _data(notice):
    return {k: v for k, v in notice.items() if k != "line"}


@pytest.mark.parametrize("route", _ROUTES)
def test_the_notice_rides_every_surface_envelope_equal_on_both_boundaries(
        ws, monkeypatch, route):
    """Criterion 1: semantic, scope dump, and both lexical routes; MCP == `--json`."""
    config, _ = ws
    settled_id, other_id = _seed_resolvable(ws)
    keyed = _gate()._keyed(config, monkeypatch)

    mcp, _ = _mcp_surface(config, route)
    cli_json, _ = _cli_surface(keyed, route, as_json=True)
    _assert_route_is(route, mcp)
    _assert_route_is(route, cli_json)

    notice = mcp["check_notice"]
    assert cli_json["check_notice"] == notice
    assert set(notice) == _gate()._NOTICE_KEYS
    assert notice["state"] == ATTEMPT_COULD_NOT_COMPLETE
    assert notice["line"] == check_notice_line(_data(notice))
    (pair,) = notice["new_pairs"]
    assert (pair["proposal"]["id"], pair["partner"]["id"]) == (settled_id, other_id)
    if route == "lexical_no_graph":
        # No second store is opened: the handles are whole ids, never resolved.
        assert pair["proposal"]["slug"] is None and pair["partner"]["slug"] is None
    else:
        assert (pair["proposal"]["slug"], pair["partner"]["slug"]) == ("settled", "other")


@pytest.mark.parametrize("case", sorted(
    ["started", "new_findings", "could_not_complete", "could_not_complete_with_pairs",
     "spend_not_authorized", "unknown_state", "no_new_findings"]))
def test_every_shown_state_rides_the_semantic_envelope_and_the_healthy_one_does_not(
        ws, monkeypatch, case):
    """Criterion 2: 3g1's `_SHOWN` cases ride; `no_new_findings` is absent on both."""
    config, m = ws
    _rec(m, "settled", scope=["db"])
    gate = _gate()
    if case == "no_new_findings":
        gate._seed_attempt(config, ATTEMPT_NO_NEW_FINDINGS)
    else:
        seed = dict(gate._SHOWN[case])
        gate._seed_attempt(config, seed.pop("state"), **seed)
    keyed = gate._keyed(config, monkeypatch)

    mcp, _ = _mcp_surface(config, "semantic")
    cli_json, _ = _cli_surface(keyed, "semantic", as_json=True)
    if case == "no_new_findings":
        assert "check_notice" not in mcp and "check_notice" not in cli_json
    else:
        assert mcp["check_notice"]["state"] == gate._SHOWN[case]["state"]
        assert cli_json["check_notice"] == mcp["check_notice"]


@pytest.mark.parametrize("route", ["collection_missing", "no_matches"])
def test_the_healthy_empty_index_carries_the_notice(ws, monkeypatch, route):
    """Criterion 3: an absent collection over an empty expected set is the empty index
    a fresh clone meets — `semantic_ran` stays True and the envelope carries it."""
    config, _ = ws
    seed = dict(_gate()._SHOWN["could_not_complete"])
    _gate()._seed_attempt(config, seed.pop("state"), **seed)
    keyed = _gate()._keyed(config, monkeypatch)

    mcp, _ = _mcp_surface(config, route)
    cli_json, _ = _cli_surface(keyed, route, as_json=True)
    for env in (mcp, cli_json):
        assert "degraded" not in env and env["active_decisions"] == []
        assert env["check_notice"]["state"] == ATTEMPT_COULD_NOT_COMPLETE
    assert cli_json["check_notice"] == mcp["check_notice"]


def _text_notice(err):
    (line,) = _gate()._notice_lines(err)
    return line


@pytest.mark.parametrize("route", ["semantic", "no_matches", "lexical_empty",
                                   "lexical_no_graph"])
def test_the_text_exits_close_with_the_line_and_a_recipe_that_parses(
        ws, monkeypatch, route):
    """Criterion 4: full results, clean-empty, and both lexical routes. The notice
    is the last thing on stderr, never on stdout, and its recipe resolves back to
    the workspace with no `--yes`."""
    config, _ = ws
    _seed_resolvable(ws)
    keyed = _gate()._keyed(config, monkeypatch)
    notice, _ = _mcp_surface(config, route)
    notice = notice["check_notice"]

    out, err = _cli_surface(keyed, route, as_json=False)
    if route == "no_matches":
        assert "No active precedents found" in out  # the clean-empty exit
    line = _text_notice(err)
    assert err.rstrip().endswith(line)
    assert notice["line"] not in out
    assert line == f"{notice['line']} `mitos check -p {keyed.project!r}` attempts it again."
    recipe = _gate()._last_backticked(line)
    assert "--yes" not in recipe
    _gate()._resolves_to(recipe, keyed)


def test_a_refused_spend_names_a_person_at_a_terminal_on_surface(ws, monkeypatch):
    """Criterion 4: `spend_not_authorized` hands the prompt to a person, never `--yes`."""
    config, m = ws
    _rec(m, "settled", scope=["db"])
    _gate()._seed_attempt(config, ATTEMPT_SPEND_NOT_AUTHORIZED)
    keyed = _gate()._keyed(config, monkeypatch)

    _, err = _cli_surface(keyed, "semantic", as_json=False)
    line = _text_notice(err)
    assert line.endswith(f" A person runs `mitos check -p {keyed.project!r}` at a "
                         "terminal and confirms the prompt.")
    assert "--yes" not in line


def test_the_surface_clause_names_the_invoked_build(ws, monkeypatch, tmp_path):
    """Criterion 4 (W16): an absolute `argv[0]` whose realpath is not `PATH`'s mitos
    is quoted as the command, and the rest of the recipe still parses."""
    from mitos import cli
    config, _ = ws
    _seed_resolvable(ws)
    keyed = _gate()._keyed(config, monkeypatch)
    stub = str(tmp_path / "a venv" / "bin" / "mitos")
    monkeypatch.setattr(cli.shutil, "which", lambda name: str(tmp_path / "other" / "mitos"))

    _, err = _cli_surface(keyed, "semantic", as_json=False, argv0=stub)
    recipe = _gate()._last_backticked(_text_notice(err))
    tokens = shlex.split(recipe)
    assert tokens[0] == stub and recipe.startswith(shlex.quote(stub) + " ")
    args = cli._build_parser().parse_args(tokens[1:])
    assert args.command == "check" and args.project_post == keyed.project


def test_mcp_carries_the_payload_alone(ws, monkeypatch):
    """Criterion 5: no command on the MCP line, and nothing beyond 3g1's payload."""
    config, _ = ws
    _seed_resolvable(ws)
    keyed = _gate()._keyed(config, monkeypatch)
    for route in _ROUTES:
        mcp, _ = _mcp_surface(config, route)
        notice = mcp["check_notice"]
        assert "`" not in notice["line"] and "mitos " not in notice["line"], route
        assert set(notice) == _gate()._NOTICE_KEYS
        assert _cli_surface(keyed, route, as_json=True)[0]["check_notice"] == notice


def _read_spies(monkeypatch):
    """Counts calls to each boundary's own `read_last_attempt` binding."""
    from mitos import cli, mcp_server
    calls = []
    for mod in (cli, mcp_server):
        real = mod.read_last_attempt

        def spy(path, _real=real, _mod=mod.__name__):
            calls.append(_mod)
            return _real(path)

        monkeypatch.setattr(mod, "read_last_attempt", spy)
    return calls


def test_a_keyless_surface_reads_nothing_then_shows_once_keyed(ws, monkeypatch):
    """Criterion 6, a transition: keyless pays no read on any exit; the same calls
    show the notice once a key is set."""
    from mitos import cli
    config, _ = ws
    _seed_resolvable(ws)
    opens = _gate()._attempt_rung_opens(monkeypatch)
    reads = _read_spies(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    keyless = MitosConfig(config.workspace_dir, project=config.project)

    for route in _ROUTES:
        mcp, _ = _mcp_surface(config, route)
        cli_json, _ = _cli_surface(keyless, route, as_json=True)
        _, err = _cli_surface(keyless, route, as_json=False)
        assert "check_notice" not in mcp and "check_notice" not in cli_json, route
        assert _gate()._notice_lines(err) == [], route
    assert opens == [] and reads == []

    keyed = _gate()._keyed(config, monkeypatch)
    mcp, _ = _mcp_surface(config, "semantic")
    _, err = _cli_surface(keyed, "semantic", as_json=False)
    assert mcp["check_notice"]["state"] == ATTEMPT_COULD_NOT_COMPLETE
    assert len(_gate()._notice_lines(err)) == 1
    assert opens and set(reads) == {cli.__name__, "mitos.mcp_server"}


@pytest.mark.parametrize("record", ["no_new_findings", "no_telemetry_file"])
def test_a_healthy_keyed_surface_is_byte_identical_to_a_keyless_one(ws, monkeypatch, record):
    """Criterion 7: keyed + a healthy record (or no telemetry at all) adds nothing."""
    config, m = ws
    _rec(m, "settled", scope=["db"])
    if record == "no_new_findings":
        _gate()._seed_attempt(config, ATTEMPT_NO_NEW_FINDINGS)
    else:
        assert not os.path.exists(config.telemetry_path)

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    keyless = MitosConfig(config.workspace_dir, project=config.project)
    before = {r: (_mcp_surface(config, r)[0], _cli_surface(keyless, r, as_json=True)[0],
                  _cli_surface(keyless, r, as_json=False)) for r in _ROUTES}
    keyed = _gate()._keyed(config, monkeypatch)
    after = {r: (_mcp_surface(config, r)[0], _cli_surface(keyed, r, as_json=True)[0],
                 _cli_surface(keyed, r, as_json=False)) for r in _ROUTES}
    assert after == before
    # Absent, never null — a relative comparison alone is blind to a key both sides set.
    for mcp, cli_json, (_out, err) in after.values():
        assert "check_notice" not in mcp and "check_notice" not in cli_json
        assert _gate()._notice_lines(err) == []


def test_query_carries_no_notice_on_any_exit(ws, monkeypatch):
    """Criterion 8: `query_decisions` (exact slug, ranked, lexical) and `mitos query`
    (`--json` and text; ranked and the no-provider lexical exit) — keyed, with a
    shown record on file — carry no key, print no line and open no attempt rung.
    The CLI has no exact-slug branch (ADR `cli-query-stays-semantic-not-dereference-twin`)."""
    from mitos import mcp_server
    config, _ = ws
    _seed_resolvable(ws)
    # `cmd_query` takes its config as an argument, and `config.env` is resolved at
    # construction: only a config built after the key is set is keyed. The `ws`
    # one is keyless, and a keyless silence would prove nothing about `query`.
    keyed = _gate()._keyed(config, monkeypatch)
    opens = _gate()._attempt_rung_opens(monkeypatch)

    store = GraphStore(config.db_path, read_only=True)
    mcp_exits = {
        "exact_slug": ("settled", (store, _FakeEmbed(), _FakeVector(_HIT))),
        "ranked": ("some claim", (store, _FakeEmbed(), _FakeVector(_HIT))),
        "lexical": ("some claim", (store, None, None)),
    }
    for label, (query, comps) in mcp_exits.items():
        err = io.StringIO()
        with patch.object(mcp_server, "get_workspace_components", return_value=comps), \
                redirect_stderr(err):
            out = json.loads(mcp_server.query_decisions(query, project=config.workspace_dir))
        assert ("degraded" in out) == (label == "lexical"), label  # the exit it names
        assert "check_notice" not in out, label
        assert _gate()._notice_lines(err.getvalue()) == [], label

    for matches in (_HIT, None):  # ranked; the no-embed/vector lexical exit
        for as_json in (True, False):
            err = io.StringIO()
            with redirect_stderr(err), patch.object(sys, "argv", ["mitos"]):
                out = _cli_query(matches, ws, as_json=as_json, config=keyed)
            assert "check_notice" not in out
            assert _gate()._notice_lines(err.getvalue()) == []
    assert opens == []

    # The control: the same keyed setup shows the notice on `surface`, so the
    # silence above is `query`'s and not the fixture's.
    assert _mcp_surface(config, "semantic")[0]["check_notice"]
    assert _cli_surface(keyed, "semantic", as_json=True)[0]["check_notice"]


@pytest.mark.parametrize("boundary", ["mcp", "cli"])
def test_a_raise_inside_the_read_leaves_the_ranking_whole(ws, monkeypatch, boundary):
    """Criterion 9, the degrade: a raise planted at the boundary's `read_last_attempt`
    is absorbed by the total composer. The ranked envelope comes back equal to an
    unplanted one minus the notice — same decisions, confidence and note, never
    `degraded` — with exactly one warning. Green under a misplaced call too (D3):
    this row proves totality, not placement."""
    from mitos import cli, mcp_server
    config, _ = ws
    _seed_resolvable(ws)
    keyed = _gate()._keyed(config, monkeypatch)

    def run():
        if boundary == "mcp":
            return _mcp_surface(config, "semantic")
        return _cli_surface(keyed, "semantic", as_json=True)

    whole, _ = run()
    assert "check_notice" in whole

    def planted(path):
        raise _PlantedFault("planted in the read")

    monkeypatch.setattr(mcp_server if boundary == "mcp" else cli,
                        "read_last_attempt", planted)
    degraded, err = run()
    expected = {k: v for k, v in whole.items() if k != "check_notice"}
    assert degraded == expected
    assert "degraded" not in degraded and degraded["confidence"] == whole["confidence"]
    assert len(_gate()._notice_warnings(err)) == 1


@pytest.mark.parametrize("boundary", ["mcp", "cli"])
def test_a_raise_at_the_call_site_propagates_and_is_never_degraded_recall(
        ws, monkeypatch, boundary):
    """Criterion 10, the placement row (4a's tripwire). A raise at the call site is
    a defect in this code, not a fault of the read. Composed inside the ranked
    `try`, the loop's blanket `except` would catch it and render a healthy ranked
    read as a lexical-degraded envelope with a nonsense reason — §4.5's latent fault
    by a new door. So the planted class must propagate out of the verb."""
    from mitos import cli, mcp_server
    config, _ = ws
    _seed_resolvable(ws)
    keyed = _gate()._keyed(config, monkeypatch)

    def planted(*args, **kwargs):
        raise _PlantedFault("planted at the call site")

    monkeypatch.setattr(mcp_server if boundary == "mcp" else cli,
                        "compose_check_notice", planted)
    with pytest.raises(_PlantedFault):
        if boundary == "mcp":
            _mcp_surface(config, "semantic")
        else:
            _cli_surface(keyed, "semantic", as_json=True)


def test_the_blackout_and_the_unbuilt_graph_carry_the_notice_too(ws, cloned, monkeypatch):
    """Stretch (belt-and-braces over D1): the two note overrides still carry it."""
    config, m = ws
    _rec(m, "dead-v1", scope=["x"])
    _rec(m, "dead-v2", scope=["x"], supersedes="dead-v1")
    seed = dict(_gate()._SHOWN["started"])
    _gate()._seed_attempt(config, seed.pop("state"))
    _gate()._keyed(config, monkeypatch)
    from mitos import mcp_server
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(
            store, _FakeEmbed(),
            _FakeVector([{"slug": "dead-v1", "score": 0.9}], store=store))):
        blackout = json.loads(mcp_server.surface_decisions("c", project=config.workspace_dir))
    assert blackout["all_superseded"]
    assert blackout["check_notice"]["state"] == _gate()._SHOWN["started"]["state"]

    clone, _ = cloned
    _gate()._seed_attempt(clone, _gate()._SHOWN["started"]["state"])
    unbuilt, _ = _mcp_surface(clone, "no_matches")
    assert "graph is unbuilt" in unbuilt["note"]
    assert unbuilt["check_notice"]["state"] == _gate()._SHOWN["started"]["state"]


# --------------------------------------------------------------------------- #
# Phase 4b — A1 depth: `full_top`, `rejected_paths_withheld`, the refusals.
#
# Ranks 1..N keep `rejected_paths`, lower ranks are axiom-only, and a thinned
# answer says so by a count and one clause naming the by-handle read. Rank is
# rank among the decisions RETURNED: the fixture puts 4a's parked OQ at
# point-rank 1, so an implementation that ranks by `enumerate(matches)` spends
# rank 1 on a hit it skipped. Four decisions, so a mid value has a real middle.
# --------------------------------------------------------------------------- #

from mitos import cli as _cli  # noqa: E402
from mitos.recall import count_withheld, withheld_clause  # noqa: E402

_4B_POINTS = [{"slug": "oq-parked", "score": 0.85}, {"slug": "dec-a", "score": 0.8},
              {"slug": "dec-b", "score": 0.7}, {"slug": "dec-c", "score": 0.65},
              {"slug": "dec-d", "score": 0.62}]
_4B_SLUGS = ["dec-a", "dec-b", "dec-c", "dec-d"]


def _seed_4b(ws):
    """4a's seed (two decisions + the parked OQ) and two more decisions."""
    _, m = ws
    _seed_4a(ws)
    _rec(m, "dec-c", scope=["x"])
    _rec(m, "dec-d", scope=["x"])


def _whole(hits):
    return ["rejected_paths" in h for h in hits]


def _clause_for(surface, config, count):
    """The clause the envelope must end with. MCP names no project, so its config is
    irrelevant; the CLI drivers hand `cmd_*` the `ws` config itself."""
    return withheld_clause(count, surface=surface, config=config)


@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
def test_an_omitted_full_top_is_the_default_byte_for_byte(ws, driver):
    """R1: no `full_top`, `full_top=None` and `full_top` ≥ the hits all answer the same
    bytes — every hit whole, and no `rejected_paths_withheld` key at all."""
    _seed_4b(ws)
    call, key, _verb, _surface = _4A_DRIVERS[driver]
    default = call(_4B_POINTS, ws)
    assert [h["slug"] for h in default[key]] == _4B_SLUGS
    assert all(_whole(default[key]))
    assert "rejected_paths_withheld" not in default
    for full_top in (None, 4, 50):
        assert json.dumps(call(_4B_POINTS, ws, full_top=full_top)) == json.dumps(default)


@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
def test_full_top_zero_is_brief(ws, driver):
    """R2: `full_top=0` answers exactly what `brief=True` answers, and counts every hit."""
    _seed_4b(ws)
    call, key, _verb, _surface = _4A_DRIVERS[driver]
    zero = call(_4B_POINTS, ws, full_top=0)
    assert json.dumps(zero) == json.dumps(call(_4B_POINTS, ws, brief=True))
    assert _whole(zero[key]) == [False] * 4
    assert zero["rejected_paths_withheld"] == len(zero[key]) == 4


@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
def test_a_skipped_open_question_consumes_no_rank(ws, driver):
    """R3: with the OQ at point-rank 1, `full_top=1` keeps the first DECISION whole."""
    _seed_4b(ws)
    call, key, _verb, _surface = _4A_DRIVERS[driver]
    resp = call(_4B_POINTS, ws, full_top=1)
    assert [h["slug"] for h in resp[key]] == _4B_SLUGS
    assert _whole(resp[key]) == [True, False, False, False]
    assert resp["rejected_paths_withheld"] == len(resp[key]) - 1


@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
def test_a_mid_full_top_thins_below_the_cut_and_says_so(ws, driver):
    """R4: ranks ≤ N whole, ranks > N axiom-only, K the thinned count, and the note ends
    with the clause. A thinned hit is today's brief hit, key for key."""
    config, _ = ws
    _seed_4b(ws)
    call, key, _verb, surface = _4A_DRIVERS[driver]
    resp = call(_4B_POINTS, ws, full_top=2)
    whole = call(_4B_POINTS, ws)
    assert _whole(resp[key]) == [True, True, False, False]
    assert resp["rejected_paths_withheld"] == 2 == count_withheld(resp[key])
    assert resp["note"].endswith(" " + _clause_for(surface, config, 2))
    assert resp[key][:2] == whole[key][:2]
    for thin, full in zip(resp[key][2:], whole[key][2:]):
        assert thin == {k: v for k, v in full.items() if k != "rejected_paths"}
    # The count sits after the final note and before any notice (3g2: notice last).
    keys = list(resp)
    assert keys.index("rejected_paths_withheld") == keys.index("note") + 1


_4B_BANDS = {
    "strong": [0.95, 0.9, 0.85, 0.8],
    "weak": [0.7, 0.68, 0.66, 0.64],
    "none": [0.3, 0.25, 0.2, 0.15],
}


@pytest.mark.parametrize("driver", ["mcp-surface", "mcp-query"])
def test_the_cut_never_follows_the_band(ws, driver):
    """R5: the same `full_top` thins the same ranks over a strong, a weak and a
    none-with-results ranking — depth is the caller's, never the band's."""
    _seed_4b(ws)
    call, key, _verb, _surface = _4A_DRIVERS[driver]
    seen = {}
    for band, scores in _4B_BANDS.items():
        points = [{"slug": s, "score": sc} for s, sc in zip(_4B_SLUGS, scores)]
        resp = call(points, ws, full_top=1)
        assert resp["confidence"] == band
        seen[band] = (_whole(resp[key]), resp["rejected_paths_withheld"])
    assert all(v == ([True, False, False, False], 3) for v in seen.values()), seen


@pytest.mark.parametrize("route", ["no providers", "mid-query fault"])
@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
def test_the_lexical_envelope_honours_full_top_by_position(ws, driver, route):
    """R6: the degraded lexical answer thins by position among its matches and carries
    the key and the clause. The query term-matches every fixture axiom ("Axiom for …"),
    so the row cannot pass on an empty envelope."""
    config, _ = ws
    _seed_4b(ws)
    call, _key, _verb, surface = _4A_DRIVERS[driver]
    matches = None if route == "no providers" else _Boom()
    resp = call(matches, ws, query="axiom", full_top=1)
    assert resp["degraded"] == "lexical"
    assert len(resp["matches"]) >= 2
    assert _whole(resp["matches"]) == [True] + [False] * (len(resp["matches"]) - 1)
    assert resp["rejected_paths_withheld"] == len(resp["matches"]) - 1
    assert resp["note"].endswith(" " + _clause_for(surface, config, len(resp["matches"]) - 1))
    # Whole by default on the same route: the key exists only when something was cut.
    assert "rejected_paths_withheld" not in call(matches, ws, query="axiom")


@pytest.mark.parametrize("driver", ["mcp-surface", "cli-surface"])
def test_the_scope_dump_honours_full_top_by_position(ws, driver):
    """R7: `surface`'s degraded scoped dump thins by list position, with key and clause.
    Four decisions sit under the default window of five; 4c's rows drive `limit`."""
    config, _ = ws
    _seed_4b(ws)
    call, key, _verb, surface = _4A_DRIVERS[driver]
    resp = call(None, ws, scope="x", full_top=1)
    assert "degraded" not in resp
    assert sorted(h["slug"] for h in resp[key]) == _4B_SLUGS
    assert _whole(resp[key]) == [True, False, False, False]
    assert resp["rejected_paths_withheld"] == 3
    assert resp["note"].endswith(" " + _clause_for(surface, config, 3))


def test_an_exact_slug_hit_is_never_thinned(ws):
    """R8: the dereference exit ignores `full_top` — whole, and no count."""
    _seed_4b(ws)
    resp = _mcp_query([], ws, query="dec-b", full_top=0)
    assert resp["slug"] == "dec-b"
    assert resp["rejected_paths"] == "Rejected for dec-b."
    assert "rejected_paths_withheld" not in resp


@pytest.mark.parametrize("argv", [
    ["surface", "q", "--brief", "--full-top", "2"],
    ["query", "q", "--brief", "--full-top", "2"],
    ["surface", "q", "--full-top", "-1"],
    ["query", "q", "--full-top=-1"],
    ["surface", "q", "--full-top", "two"],
])
def test_the_cli_refuses_a_depth_fault_through_argparse(argv, capsys):
    """R10: both-together and a negative value exit 2 before anything resolves."""
    with pytest.raises(SystemExit) as exc:
        _cli._build_parser().parse_args(["-p", "x", *argv])
    assert exc.value.code == 2
    assert "--full-top" in capsys.readouterr().err


def test_the_cli_parses_full_top_and_leaves_it_none_by_default():
    parse = _cli._build_parser().parse_args
    assert parse(["surface", "q", "--full-top", "2"]).full_top == 2
    assert parse(["query", "q", "--full-top", "0"]).full_top == 0
    assert parse(["surface", "q"]).full_top is None
    assert parse(["query", "q", "--brief"]).brief is True


@pytest.mark.parametrize("verb", ["surface", "query"])
def test_cli_and_mcp_thin_alike(ws, verb):
    """R11: on one fixture and one `full_top`, both boundaries agree on the hits'
    key sets, the count, and the clause modulo the call form."""
    config, _ = ws
    _seed_4b(ws)
    mcp_call, key, _, _ = _4A_DRIVERS[f"mcp-{verb}"]
    cli_call = _4A_DRIVERS[f"cli-{verb}"][0]
    mcp_out = mcp_call(_4B_POINTS, ws, full_top=2)
    cli_out = cli_call(_4B_POINTS, ws, full_top=2)
    assert [set(h) for h in mcp_out[key]] == [set(h) for h in cli_out[key]]
    assert mcp_out["rejected_paths_withheld"] == cli_out["rejected_paths_withheld"] == 2
    mcp_clause, cli_clause = _clause_for("mcp", config, 2), _clause_for("cli", config, 2)
    assert mcp_out["note"].endswith(mcp_clause) and cli_out["note"].endswith(cli_clause)
    pointers = {s: _SURFACE_POINTERS[s]["dereference"] for s in ("cli", "mcp")}
    from mitos.shell_quote import shell_flag
    cli_form = pointers["cli"].format(selector=shell_flag("-p", config.project))
    assert cli_clause.replace(cli_form, pointers["mcp"]) == mcp_clause


def _cli_text(verb, ws, full_top):
    if verb == "surface":
        return _cli_surface_text(_4B_POINTS, ws, full_top=full_top)
    return _cli_query(_4B_POINTS, ws, full_top=full_top)


@pytest.mark.parametrize("verb", ["surface", "query"])
def test_the_printed_clause_carries_a_recipe_that_parses(ws, verb):
    """R12: the recipe on the printed `→` note parses to `show` on this workspace."""
    config, _ = ws
    _seed_4b(ws)
    out = _cli_text(verb, ws, full_top=1)
    note = next(ln for ln in out.splitlines()
                if ln.startswith("→ ") and "omit rejected_paths" in ln)
    recipes = [r for r in note.split("`")[1::2] if r.startswith("mitos show")]
    assert len(recipes) == 1
    args = _cli._build_parser().parse_args(shlex.split(recipes[0])[1:])
    assert args.command == "show"
    assert args.project_post == config.project
    assert args.ident == "<slug>"


@pytest.mark.parametrize("verb", ["surface", "query"])
def test_a_thinned_text_hit_prints_no_rejected_line(ws, verb):
    """R16: one whole hit prints one `Rejected:` line; the clause rides the `→` note."""
    config, _ = ws
    _seed_4b(ws)
    out = _cli_text(verb, ws, full_top=1)
    assert sum(ln.strip().startswith("Rejected:") for ln in out.splitlines()) == 1
    assert _clause_for("cli", config, 3) in out
    assert not any(_clause_for("cli", config, 3) in ln
                   for ln in _cli_text(verb, ws, full_top=None).splitlines())


def test_the_lexical_text_render_prints_the_clause_on_its_note_line(ws):
    """W1: the degraded text render prints its note bare (no `→`), clause included."""
    config, _ = ws
    _seed_4b(ws)
    out = _cli_query(None, ws, query="axiom", full_top=1)
    assert sum(ln.strip().startswith("Rejected:") for ln in out.splitlines()) == 1
    assert any(ln.endswith(_clause_for("cli", config, 3)) for ln in out.splitlines())


@pytest.mark.parametrize("count", [1, 3])
def test_the_clause_names_count_field_and_a_read(count):
    """R13/R14: singular and plural right, the MCP form names no shell command, neither
    crosses the boundary's call forms, and both pass the `query` register lock."""
    cli = withheld_clause(count, surface="cli", config=_StubConfig("/home/user/my projects/demo"))
    mcp = withheld_clause(count, surface="mcp", config=_StubConfig())
    for clause in (cli, mcp):
        assert clause.startswith(f"{count} hit{'' if count == 1 else 's'} ")
        assert "rejected_paths" in clause
        lowered = clause.casefold()
        for banned in _QUERY_REGISTER_BANNED:
            assert banned not in lowered
    assert "mitos " not in mcp and "-p " not in mcp and "show_node(" in mcp
    assert "show_node(" not in cli and "surface_decisions(" not in cli
    assert "-p '/home/user/my projects/demo' -- <slug>" in cli


def test_every_lexical_route_forwards_full_top():
    """G2: each call into a lexical helper passes the effective `full_top` along — a
    route that passed a literal would answer whole with every other row still green."""
    import mitos.mcp_server as mcp_mod
    targets = {"_lexical_degraded_response", "_emit_lexical_degraded"}
    seen = 0
    for mod in (mcp_mod, _cli):
        for node in ast.walk(ast.parse(inspect.getsource(mod))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in targets):
                seen += 1
                kw = {k.arg: k.value for k in node.keywords}
                assert isinstance(kw.get("full_top"), ast.Name), ast.unparse(node)
                assert kw["full_top"].id == "full_top", ast.unparse(node)
    assert seen == 12


# --------------------------------------------------------------------------- #
# Phase 2e — the edge clause, written over 4c's lever block on intent: every 4c
# row survives as the edge row testing the same intent (IMPLEMENTATION_NOTES,
# "Changed expectations"), plus a scope dump that follows `limit` and says how much
# of the scope it shows (4c, unchanged).
#
# Lock 1's clause rode any full weak/none window. The edge clause rides ANY ranked
# band when the rank just past the window scores within `RANKED_EDGE_MARGIN` of the
# last one returned, says an unmeasured edge is unmeasured, and is otherwise absent.
# `_FakeVector` honours `limit` and returns at most its canned list, so five points
# at limit 5 are an EXHAUSTED fetch: no edge and no clause (G1). An edge fixture adds
# a sixth live decision whose gap to the fifth is derived from the margin BY NAME and
# guarded (G2), so a moved margin breaks the fixture loudly rather than flipping it.
# Expected strings come from `edge_clause` and the composers, never a hand copy —
# except the dump's sentences, whose exact words are the thing pinned.
# --------------------------------------------------------------------------- #

import os.path  # noqa: E402
import re  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from mitos import recall as _recall  # noqa: E402
from mitos import retrieval_walk  # noqa: E402

_4C_SLUGS = ["dec-a", "dec-b", "dec-c", "dec-d", "dec-e"]
_4C_SCORES = {
    "strong": [0.9, 0.85, 0.8, 0.78, 0.76],
    "weak": [0.7, 0.68, 0.66, 0.64, 0.62],
    "none": [0.5, 0.45, 0.4, 0.35, 0.3],
}
_SIXTH = "dec-f"


def _seed_4c(ws):
    """4b's seed (four decisions + the parked OQ) and a fifth decision."""
    _, m = ws
    _seed_4b(ws)
    _rec(m, "dec-e", scope=["x"])


def _seed_edge(ws):
    """4c's seed and the sixth decision an edge fixture ranks just past the window."""
    _, m = ws
    _seed_4c(ws)
    _rec(m, _SIXTH, scope=["x"])


def _points(band, n=5):
    return [{"slug": s, "score": sc} for s, sc in zip(_4C_SLUGS[:n], _4C_SCORES[band][:n])]


def _with_sixth(band, gap):
    points = _points(band) + [{"slug": _SIXTH, "score": _4C_SCORES[band][4] - gap}]
    return points, points[4]["score"] - points[5]["score"]


def _mid_points(band):
    """Five points and a sixth inside the margin of the fifth (guarded, G2)."""
    points, gap = _with_sixth(band, RANKED_EDGE_MARGIN / 2)
    assert gap < RANKED_EDGE_MARGIN, (band, gap)
    return points


def _clear_points(band):
    """Five points and a sixth past the margin below the fifth (guarded, G2)."""
    points, gap = _with_sixth(band, RANKED_EDGE_MARGIN * 2)
    assert gap >= RANKED_EDGE_MARGIN, (band, gap)
    return points


def _mid(limit, held):
    return WindowEdge(EDGE_KIND_MID_CLUSTER, limit, held)


def _edge_for(surface, limit, held):
    return edge_clause(_mid(limit, held), surface=surface)


def _unmeasured_for(surface, limit):
    return edge_clause(WindowEdge(EDGE_KIND_UNMEASURED, limit, None), surface=surface)


def _edge_stems(surface, limit):
    """What every variant of the clause at this window starts with — the absence probes."""
    mid = os.path.commonprefix([_edge_for(surface, limit, limit), _edge_for(surface, limit, 0),
                                _edge_for(surface, limit, None)])
    return mid, _unmeasured_for(surface, limit)


def _assert_no_edge(note, surface, limit):
    for stem in _edge_stems(surface, limit):
        assert stem not in note, (stem, note)


def _base_note(driver, config, top_score, n):
    """The band note the surfaced decisions earn with no edge clause."""
    _call, _key, verb, surface = _4A_DRIVERS[driver]
    return _4a_band(verb, surface, config, top_score, n)[1]


# ---- policy ---------------------------------------------------------------- #

def test_the_edge_is_a_required_keyword_on_both_composers():
    """ADR register-selecting-keyword-is-required-on-every-axis-never-defaulted: a
    forgotten call site is a TypeError, not a silently missing clause."""
    with pytest.raises(TypeError, match="edge"):
        assess_surface_recall(semantic_ran=True, degraded_reason=None, top_score=0.7, result_count=1,
                              scope=None, surface="mcp", lead=None, short=None)
    with pytest.raises(TypeError, match="edge"):
        assess_query_recall(top_score=0.7, result_count=1, config=_StubConfig(),
                            surface="mcp", lead=None, short=None)


_FOUND = retrieval_walk.EDGE_FOUND
_UNMEASURED_END = retrieval_walk.EDGE_UNMEASURED
_SHORT = retrieval_walk.SHORT_BOUND
_EXHAUSTED = retrieval_walk.EXHAUSTED
# The table injects its own margin, so every gap is exact in binary floating point and
# the at-margin row compares 0.25 with 0.25, not two roundings of 0.0156.
_TABLE_MARGIN = 0.25


def _walk(n, *, edge=None, end=_EXHAUSTED, limit=5, last=0.75):
    """A duck-typed walk: ``window``/``edge``/``end``/``limit`` are all `window_edge` reads."""
    return SimpleNamespace(window=[SimpleNamespace(score=last)] * n,
                           edge=None if edge is None else SimpleNamespace(score=edge),
                           end=end, limit=limit)


@pytest.mark.parametrize("label,walk,full_top,expected", [
    ("within the margin", _walk(5, edge=0.625, end=_FOUND), None, _mid(5, 5)),
    ("at exactly the margin is a clear drop", _walk(5, edge=0.5, end=_FOUND), None, None),
    ("beyond the margin", _walk(5, edge=0.25, end=_FOUND), None, None),
    ("a negative gap is mid-cluster (G3)", _walk(5, edge=0.875, end=_FOUND), None, _mid(5, 5)),
    ("unmeasured", _walk(5, end=_UNMEASURED_END), None,
     WindowEdge(EDGE_KIND_UNMEASURED, 5, None)),
    ("unmeasured holds no full_top", _walk(5, end=_UNMEASURED_END), 2,
     WindowEdge(EDGE_KIND_UNMEASURED, 5, None)),
    ("short window at the bound", _walk(3, end=_SHORT), None, None),
    ("exhausted with a full window (G1)", _walk(5, end=_EXHAUSTED), None, None),
    ("exhausted short", _walk(3, end=_EXHAUSTED), None, None),
    ("empty window", _walk(0, end=_EXHAUSTED), None, None),
    ("empty window beside an edge", _walk(0, edge=0.74, end=_FOUND), None, None),
    ("below the ceiling names the lever", _walk(49, edge=0.75, end=_FOUND, limit=49), None,
     _mid(49, 49)),
    ("at the ceiling names none", _walk(50, edge=0.75, end=_FOUND, limit=50), None,
     _mid(50, None)),
    ("at the ceiling ignores full_top", _walk(50, edge=0.75, end=_FOUND, limit=50), 2,
     _mid(50, None)),
    ("full_top held", _walk(5, edge=0.7, end=_FOUND), 2, _mid(5, 2)),
    ("full_top capped at the window", _walk(5, edge=0.7, end=_FOUND), 9, _mid(5, 5)),
    ("brief", _walk(5, edge=0.7, end=_FOUND), 0, _mid(5, 0)),
], ids=lambda v: v if isinstance(v, str) else "")
def test_window_edge_decides_from_the_walk(label, walk, full_top, expected):
    """Policy row 1, inverted from `window_lever`'s points table: the edge's gap, the
    walk's end and the ceiling decide — never the window's length alone."""
    got = window_edge(walk, margin=_TABLE_MARGIN, ceiling=50, full_top=full_top)
    assert got == expected, label


def test_the_end_token_agrees_with_the_walk():
    """The one data-level join: `recall.py` may not import the walk, so it spells the
    end value it reads, and this row keeps the two spellings one."""
    assert _recall._WALK_END_UNMEASURED == retrieval_walk.EDGE_UNMEASURED


def test_window_edge_keywords_are_required():
    with pytest.raises(TypeError, match="margin"):
        window_edge(_walk(5), ceiling=50, full_top=None)
    with pytest.raises(TypeError, match="ceiling"):
        window_edge(_walk(5), margin=0.1, full_top=None)


_RETIRED_LEVER_NAMES = ("window_lever", "WindowLever", "limit_clause", "_with_lever")


def test_the_lever_names_are_gone_from_mitos():
    """D5: retired, not aliased — a living alias keeps a second rule for when the
    lever applies."""
    for name in _RETIRED_LEVER_NAMES:
        assert not hasattr(_recall, name), name
    mitos_dir = os.path.dirname(_recall.__file__)
    for fn in sorted(os.listdir(mitos_dir)):
        if not fn.endswith(".py"):
            continue
        with open(os.path.join(mitos_dir, fn), encoding="utf-8") as fh:
            text = fh.read()
        for name in _RETIRED_LEVER_NAMES:
            assert not re.search(rf"\b{re.escape(name)}\b", text), (fn, name)


@pytest.mark.parametrize("surface", ["cli", "mcp"])
def test_the_held_zero_clause_names_limit_only(surface):
    """With nothing held the clause names no full_top — naming it hands a brief caller
    a re-send the boundary refuses."""
    ptr = _SURFACE_POINTERS[surface]
    held = _edge_for(surface, 5, 3)
    zero = _edge_for(surface, 5, 0)
    assert ptr["limit_arg"].format(n=5) in held and ptr["limit_arg"].format(n=5) in zero
    assert ptr["full_top_arg"].format(n=3) in held
    assert ptr["full_top_arg"].split("{")[0].strip() not in zero


@pytest.mark.parametrize("surface", ["cli", "mcp"])
@pytest.mark.parametrize("edge", [_mid(50, None), WindowEdge(EDGE_KIND_UNMEASURED, 5, None)],
                         ids=["ceiling", "unmeasured"])
def test_a_clause_without_a_lever_offers_no_re_send(surface, edge):
    """The ceiling and the unmeasured edge name the window once and nothing to widen:
    no `full_top`, no second `limit`, no "higher limit"."""
    ptr = _SURFACE_POINTERS[surface]
    clause = edge_clause(edge, surface=surface)
    limit_stem = ptr["limit_arg"].split("{")[0]
    assert clause.count(limit_stem) == 1 and ptr["limit_arg"].format(n=edge.limit) in clause
    assert ptr["full_top_arg"].split("{")[0].strip() not in clause
    assert "higher limit" not in clause


_EDGE_CLAUSES = [_mid(5, 0), _mid(5, 5), _mid(50, None), WindowEdge(EDGE_KIND_UNMEASURED, 5, None)]


@pytest.mark.parametrize("edge", _EDGE_CLAUSES, ids=["held-0", "held-5", "ceiling", "unmeasured"])
def test_the_clause_is_register_clean_and_stays_on_its_boundary(edge):
    """No corpus verdict, no write instruction, no claim the further ranks score higher
    or are relevant; MCP names argument spellings and no shell; the CLI names flags and
    no call form."""
    cli = edge_clause(edge, surface="cli")
    mcp = edge_clause(edge, surface="mcp")
    for clause in (cli, mcp):
        lowered = clause.casefold()
        for banned in _QUERY_REGISTER_BANNED + ("higher score", "score higher", "may hold",
                                                "precedent", "relevant"):
            assert banned not in lowered, (clause, banned)
        assert "!" not in clause and "⚠" not in clause and "**" not in clause
    assert "mitos " not in mcp and "-p " not in mcp and "--" not in mcp
    assert "(" not in cli and "=" not in cli


@pytest.mark.parametrize("edge", _EDGE_SHAPES)
@pytest.mark.parametrize("label,kw", _QUERY_CASES)
@pytest.mark.parametrize("surface", ["cli", "mcp"])
def test_no_query_note_with_the_clause_instructs_a_write(surface, label, kw, edge):
    """The register lock holds with the clause riding, and neither boundary's note
    borrows the other's call forms."""
    note = _query_note(surface, edge=edge, **kw)
    lowered = note.casefold()
    for banned in _QUERY_REGISTER_BANNED:
        assert banned not in lowered, f"{label}/{surface} note carries {banned!r}"
    if surface == "cli":
        assert "surface_decisions(" not in note and "list_decisions(" not in note
    else:
        assert "mitos surface" not in note and "-p " not in note


def _band_forks():
    """Every (fork, composer call) of the matrix, with whether the clause may ride."""
    forks = []
    for verb in ("surface", "query"):
        for label, top, n, scope_unused, rides in (
            ("strong", 0.9, 3, False, True),
            ("strong-scope-unused", 0.9, 3, True, True),
            ("weak", 0.65, 3, False, True),
            ("weak-scope-unused", 0.65, 3, True, True),
            ("none-results", 0.3, 3, False, True),
            ("none-results-scope-unused", 0.3, 3, True, True),
            ("none-empty", None, 0, False, False),
            ("none-empty-scope-unused", None, 0, True, False),
        ):
            if verb == "query" and scope_unused:
                continue  # the query composer has no scope arm
            forks.append((verb, label, top, n, scope_unused, rides))
    # 5d dropped the "degraded-partial" fork (results, no scope_total): that arm now
    # raises on every edge shape (test_a_degraded_answer_with_results_and_no_scope_total_raises).
    for label, n, total in (("degraded-dump", 3, 8), ("degraded-empty", 0, None)):
        forks.append(("surface-degraded", label, None, n, total, False))
    return forks


@pytest.mark.parametrize("surface", ["cli", "mcp"])
@pytest.mark.parametrize("fork", _band_forks(), ids=lambda f: f"{f[0]}-{f[1]}")
@pytest.mark.parametrize("edge", _EDGE_SHAPES[1:])
def test_the_clause_rides_every_ranked_fork_and_no_other(surface, fork, edge):
    """Inverted from 4c's weak/none-only matrix: the clause appears iff the fork holds
    ranked results — `strong` included — and is appended after the whole note; the
    nothing-ranked fork and every degraded arm drop it."""
    verb, label, top, n, extra, rides = fork
    clause = edge_clause(edge, surface=surface)

    def compose(e):
        if verb == "query":
            return assess_query_recall(lead=None, short=None, edge=e, top_score=top, result_count=n,
                                       config=_StubConfig(), surface=surface)[1]
        if verb == "surface-degraded":
            return assess_surface_recall(lead=None, short=None, edge=e, semantic_ran=False, degraded_reason=_DEGRADED, top_score=None,
                                         result_count=n, scope="x", scope_total=extra,
                                         scope_counts=_counts(("x", 8)), surface=surface)[1]
        scope, counts = ("ghost", _counts(("x", 3))) if extra else (None, None)
        return assess_surface_recall(lead=None, short=None, edge=e, semantic_ran=True, degraded_reason=None, top_score=top,
                                     result_count=n, scope=scope, scope_counts=counts,
                                     surface=surface)[1]

    bare, with_edge = compose(None), compose(edge)
    if rides:
        assert with_edge == f"{bare} {clause}"
    else:
        assert with_edge == bare
        _assert_no_edge(with_edge, surface, edge.limit)


# ---- drivers: the clause on the window's edge ------------------------------ #

@pytest.mark.parametrize("band", ["strong", "weak", "none"])
@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
def test_a_full_window_whose_fetch_ran_out_names_no_edge(ws, driver, band):
    """G1, inverted from `test_a_filled_weak_or_none_window_names_the_lever`: five
    points at limit 5 is a fetch that ran out, so there is no next rank to compare and
    no clause — where the old clause offered a wider window over nothing."""
    config, _ = ws
    _seed_4c(ws)
    call, key, _verb, surface = _4A_DRIVERS[driver]
    resp = call(_points(band), ws)
    assert resp["confidence"] == band
    assert len(resp[key]) == 5
    assert resp["note"] == _base_note(driver, config, _4C_SCORES[band][0], 5)
    _assert_no_edge(resp["note"], surface, 5)


@pytest.mark.parametrize("band", ["weak", "none"])
@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
def test_a_mid_cluster_window_names_the_edge(ws, driver, band):
    """Five live decisions and a sixth inside the margin of the fifth → the band note,
    one space, the clause — the same clause characters on `surface` and `query` of one
    boundary, and the sixth decision is not returned."""
    config, _ = ws
    _seed_edge(ws)
    call, key, _verb, surface = _4A_DRIVERS[driver]
    resp = call(_mid_points(band), ws)
    assert resp["confidence"] == band
    assert [h["slug"] for h in resp[key]] == _4C_SLUGS
    base = _base_note(driver, config, _4C_SCORES[band][0], 5)
    assert resp["note"] == f"{base} {_edge_for(surface, 5, 5)}"


@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
def test_the_rank_six_case_under_strong_names_the_edge(ws, driver):
    """Success criterion 1, inverted from `test_a_strong_filled_window_names_no_lever`:
    the case lock 1's clause could never flag — a `strong` answer whose sixth live
    decision scores as close as its fifth."""
    config, _ = ws
    _seed_edge(ws)
    call, _key, _verb, surface = _4A_DRIVERS[driver]
    resp = call(_mid_points("strong"), ws)
    assert resp["confidence"] == "strong"
    assert resp["note"] == f"{_base_note(driver, config, 0.9, 5)} {_edge_for(surface, 5, 5)}"


@pytest.mark.parametrize("band", ["strong", "weak", "none"])
@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
def test_a_clear_drop_names_no_edge(ws, driver, band):
    """A sixth live decision past the margin below the fifth: the window ended at a
    drop, and the note is the band note alone."""
    config, _ = ws
    _seed_edge(ws)
    call, key, _verb, surface = _4A_DRIVERS[driver]
    resp = call(_clear_points(band), ws)
    assert resp["confidence"] == band
    assert len(resp[key]) == 5
    assert resp["note"] == _base_note(driver, config, _4C_SCORES[band][0], 5)
    _assert_no_edge(resp["note"], surface, 5)


@pytest.mark.parametrize("band", ["weak", "none"])
@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
def test_a_short_window_states_why_after_todays_band_note(ws, driver, band):
    """Four points at limit 5 — the fetch ran out short. 2e pinned the band note
    alone and left the short-window statement to 2f; 2f inverts it on intent: the
    band note byte for byte, then the `ran_out` clause, and still no edge."""
    config, _ = ws
    _seed_4c(ws)
    call, _key, _verb, surface = _4A_DRIVERS[driver]
    resp = call(_points(band, 4), ws)
    assert resp["confidence"] == band
    short = short_window_clause(ShortWindow("ran_out", 4, 5, 4, 0), surface=surface)
    assert resp["note"] == f"{_base_note(driver, config, _4C_SCORES[band][0], 4)} {short}"
    _assert_no_edge(resp["note"], surface, 5)


@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
def test_a_full_window_that_surfaces_nothing_names_no_edge(ws, driver):
    """Five points (the parked OQ and four unresolvable slugs) surface nothing —
    `none`'s nothing-ranked fork. The walk fills the window with live decisions only,
    so these points fill nothing and there is no edge to state."""
    config, _ = ws
    _seed_4c(ws)
    call, key, _verb, surface = _4A_DRIVERS[driver]
    # The four ghosts are points no node carries — orphans, declared as such.
    points = [{"slug": "oq-parked", "score": 0.7}] + [
        orphan_hit(0.65, slug=f"ghost-{i}", tag=f"ghost-{i}") for i in range(4)]
    resp = call(points, ws)
    assert resp["confidence"] == "none"
    assert resp[key] == []
    assert resp["note"] == _base_note(driver, config, None, 0)
    _assert_no_edge(resp["note"], surface, 5)


@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
def test_the_edge_is_the_next_live_decision_not_the_next_point(ws, driver):
    """Inverted from `test_the_window_counts_live_decisions_not_points`: an open
    question just past the window, inside the margin, is not the edge — the next live
    decision is, and it sits at a drop. Moved to the top, the OQ takes no slot and the
    sixth decision inside the margin is the edge."""
    config, _ = ws
    _seed_edge(ws)
    call, key, _verb, surface = _4A_DRIVERS[driver]
    clear = _clear_points("weak")
    oq = {"slug": "oq-parked", "score": _4C_SCORES["weak"][4] - RANKED_EDGE_MARGIN / 2}
    resp = call(clear[:5] + [oq] + clear[5:], ws)
    assert [h["slug"] for h in resp[key]] == _4C_SLUGS
    assert resp["note"] == _base_note(driver, config, 0.7, 5)

    resp = call([{"slug": "oq-parked", "score": 0.72}] + _mid_points("weak"), ws)
    assert [h["slug"] for h in resp[key]] == _4C_SLUGS
    assert resp["note"] == f"{_base_note(driver, config, 0.7, 5)} {_edge_for(surface, 5, 5)}"


@pytest.mark.parametrize("route", ["no providers", "mid-query fault"])
@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
def test_no_degraded_envelope_names_the_edge(ws, driver, route):
    """The lexical envelopes carry no band and no clause, on every route; the scope
    dump (surface only) neither, pinned by its whole sentence."""
    _seed_4c(ws)
    call, _key, verb, surface = _4A_DRIVERS[driver]
    matches = None if route == "no providers" else _Boom()
    resp = call(matches, ws, query="axiom", limit=3)
    assert resp["degraded"] == "lexical"
    _assert_no_edge(resp["note"], surface, 3)
    if verb == "surface":
        dump = call(matches, ws, scope="x", limit=3)
        assert "degraded" not in dump and len(dump["active_decisions"]) == 3
        assert "confidence" not in dump
        reason = _DEGRADED if route == "no providers" else "Qdrant unreachable"
        assert dump["note"] == _dump_envelope_note(surface, 3, 5, scope="x", reason=reason)


def _commit_live(store: GraphStore, prefix: str, n: int) -> None:
    """Commits ``n`` live decisions ``{prefix}-0..`` straight through the store."""
    for i in range(n):
        e = ParsedEntry("decision", f"{prefix}-{i}", 1, 5)
        e.axiom = f"Axiom for {prefix}-{i}."
        e.rejected_paths = f"Rejected for {prefix}-{i}."
        e.scope = ["x"]
        store.commit_parsed_entry(e)


@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
def test_the_ceiling_names_the_edge_without_a_lever(ws, driver):
    """Inverted from `test_no_lever_at_the_clamp_ceiling` (G5), whose `limit=50` half
    passed because its fetch ran out, not because of the ceiling (the vacuous frame).
    51 live decisions: at limit 50 the edge is the 51st, inside the margin → the
    clause, no lever (a higher limit clamps back). At 49 the same fetch names the lever.
    And a fetch of exactly 50 at limit 50 ran out → no clause at all."""
    config, m = ws
    _commit_live(m.store, "deep", 51)
    call, _key, _verb, surface = _4A_DRIVERS[driver]
    points = [{"slug": "deep-0", "score": 0.7}] + [
        {"slug": f"deep-{i}", "score": 0.61} for i in range(1, 51)]
    assert points[49]["score"] - points[50]["score"] < RANKED_EDGE_MARGIN
    for limit, held in ((50, None), (49, 49)):
        resp = call(points, ws, limit=limit)
        assert resp["confidence"] == "weak"
        base = _base_note(driver, config, 0.7, limit)
        assert resp["note"] == f"{base} {_edge_for(surface, limit, held)}"
    resp = call(points[:50], ws, limit=50)
    assert resp["note"] == _base_note(driver, config, 0.7, 50)


@pytest.mark.parametrize("depth,held", [({"full_top": 2}, 2), ({"full_top": 9}, 5),
                                        ({"full_top": 0}, 0), ({"brief": True}, 0)])
@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
def test_the_clause_holds_the_effective_full_top(ws, driver, depth, held):
    """The held count is the post-normalisation `full_top`, capped at the window;
    brief and full_top=0 get the limit-only form."""
    _seed_edge(ws)
    call, _key, _verb, surface = _4A_DRIVERS[driver]
    resp = call(_mid_points("weak"), ws, **depth)
    clause = _edge_for(surface, 5, held)
    assert clause in resp["note"]
    if held == 0:
        assert _SURFACE_POINTERS[surface]["full_top_arg"].split("{")[0].strip() \
            not in resp["note"]


@pytest.mark.parametrize("driver", list(_4A_DRIVERS))
def test_the_edge_clause_sits_between_the_band_note_and_the_withheld_clause(ws, driver):
    """Band note → edge clause (inside the composer) → withheld clause (appended by
    the call site after every override)."""
    config, _ = ws
    _seed_edge(ws)
    call, _key, _verb, surface = _4A_DRIVERS[driver]
    resp = call(_mid_points("weak"), ws, full_top=2)
    edge = _edge_for(surface, 5, 2)
    withheld = _clause_for(surface, config, 3)
    assert resp["rejected_paths_withheld"] == 3
    assert resp["note"] == f"{_base_note(driver, config, 0.7, 5)} {edge} {withheld}"
    assert resp["note"].index(edge) < resp["note"].index(withheld)


@pytest.mark.parametrize("verb", ["surface", "query"])
def test_the_printed_cli_flags_parse_on_both_verbs(ws, verb):
    """The flags the text render prints, spliced into a real argv on each verb, parse
    through the real parser to the window and the held count they name."""
    _seed_edge(ws)
    if verb == "surface":
        out = _cli_surface_text(_mid_points("weak"), ws, full_top=2)
    else:
        out = _cli_query(_mid_points("weak"), ws, full_top=2)
    line = next(ln for ln in out.splitlines() if _edge_stems("cli", 5)[0] in ln)
    limit_m = re.search(r"--limit (\d+)", line)
    held_m = re.search(r"--full-top (\d+)", line)
    fragment = [*limit_m.group(0).split(), *held_m.group(0).split()]
    for target in ("surface", "query"):
        args = _cli._build_parser().parse_args([target, "-p", "x", "q", *fragment])
        assert args.limit == int(limit_m.group(1)) == 5
        assert args.full_top == int(held_m.group(1)) == 2


def test_the_edge_pointer_keys_exist_under_both_outer_keys():
    """A one-sided key is a KeyError on the other boundary."""
    for key in ("limit_arg", "full_top_arg"):
        for side in ("cli", "mcp"):
            assert "{n}" in _SURFACE_POINTERS[side][key], (side, key)


# ---- the scope dump -------------------------------------------------------- #

_DUMP_SLUGS = [f"dump-{i}" for i in range(1, 9)]


def _seed_dump(ws):
    """Eight active decisions in scope 'y', and one in scope 'solo'."""
    _, m = ws
    for slug in _DUMP_SLUGS:
        _rec(m, slug, scope=["y"])
    _rec(m, "only-one", scope=["solo"])


def _dump_note(surface, shown, total, scope="y", reason=_DEGRADED):
    return assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=False, degraded_reason=reason,
                                 top_score=None, result_count=shown, scope=scope,
                                 scope_total=total, surface=surface)[1]


# 5d: the dump's envelope note is the composer's dump sentence, then its rejection
# section's clause (searched by words, states read), then any withheld clause.
_DUMP_SECTION = degraded_section_clause(computed=True, retired_dropped=True)


def _dump_envelope_note(surface, shown, total, scope="y", reason=_DEGRADED):
    return f"{_dump_note(surface, shown, total, scope=scope, reason=reason)} {_DUMP_SECTION}"


def test_the_dump_sentence_names_how_much_of_the_scope_it_shows():
    """Row 14's words, pinned once here: part, all, and the singular — with today's
    complete_hint tail. The head is 7a's: it names the injected cause, and the
    hard-coded "(embeddings/Qdrant down)" it replaced is gone (a cause the note never
    saw, which 4a caught blaming Qdrant for a KeyError)."""
    reason = provider_cause_phrase(AUTH)
    head = f"Semantic recall unavailable ({reason}) — showing "
    # 2g: the dump says what it could not compute, after the complete_hint tail
    # (5d: three items again — rejected options are now matched by their words).
    tail = (" as a fallback, NOT a relevance ranking. For the authoritative set use "
            "list_decisions(scope='y') (pure graph read). "
            "Not computed without similarity scores: the retired partition, the "
            "window's edge and the confidence band.")
    assert (_dump_note("mcp", 7, 8, reason=reason)
            == f"{head}7 of the 8 active decisions in scope 'y'{tail}")
    assert (_dump_note("mcp", 8, 8, reason=reason)
            == f"{head}all 8 active decisions in scope 'y'{tail}")
    assert (_dump_note("mcp", 1, 1, reason=reason)
            == f"{head}the one active decision in scope 'y'{tail}")
    assert "embeddings/Qdrant down" not in _dump_note("mcp", 7, 8)


def test_a_degraded_answer_with_results_and_no_scope_total_raises():
    """Row 14's other half, rewritten by 5d (D6, WIRING_LEDGER entry-014): the
    partial-list arm (degraded, results, no ``scope_total``) was reached by no
    surface — with recall down no walk runs, so only the dump fills the list — and
    its omission clause rode a note whose ``not_computed`` nothing set. It now
    raises, naming the contract, on every edge shape; the missing-reason guard is
    still asked first."""
    for edge in _EDGE_SHAPES:
        with pytest.raises(ValueError, match="scope dump's; pass scope_total"):
            assess_surface_recall(lead=None, short=None, edge=edge, semantic_ran=False,
                                  degraded_reason=_DEGRADED, top_score=None,
                                  result_count=2, scope="x", scope_total=None,
                                  surface="mcp")
    with pytest.raises(ValueError, match="degraded_reason"):
        assess_surface_recall(lead=None, short=None, edge=None, semantic_ran=False,
                              degraded_reason=None, top_score=None, result_count=2,
                              scope="x", scope_total=None, surface="mcp")


@pytest.mark.parametrize("route", ["no providers", "mid-query fault"])
@pytest.mark.parametrize("limit,shown", [(7, 7), (None, 5), (20, 8)])
def test_the_dump_follows_limit_and_counts_on_both_surfaces(ws, limit, shown, route):
    """Rows 13 and 14 / K4, K5: the dump shows the clamped limit's worth of the scope,
    its note says how many of how many, and CLI⇄MCP differ only in call forms."""
    _seed_dump(ws)
    matches = None if route == "no providers" else _Boom()
    got = {}
    for surface, call in (("mcp", _surface_with), ("cli", _cli_surface_json)):
        resp = call(matches, ws, scope="y", limit=limit)
        assert "degraded" not in resp and "confidence" not in resp
        assert len(resp["active_decisions"]) == shown
        assert set(h["slug"] for h in resp["active_decisions"]) <= set(_DUMP_SLUGS)
        reason = _DEGRADED if route == "no providers" else "Qdrant unreachable"
        assert resp["note"] == _dump_envelope_note(surface, shown, 8, reason=reason)
        got[surface] = resp
    assert ([h["slug"] for h in got["mcp"]["active_decisions"]]
            == [h["slug"] for h in got["cli"]["active_decisions"]])
    cli_hint = _SURFACE_POINTERS["cli"]["complete_scope"].format(scope="y")
    mcp_hint = _SURFACE_POINTERS["mcp"]["complete_scope"].format(scope="y")
    assert got["mcp"]["note"].replace(mcp_hint, cli_hint) == got["cli"]["note"]


@pytest.mark.parametrize("driver", ["mcp-surface", "cli-surface"])
def test_a_one_decision_scope_dump_reads_singular(ws, driver):
    """Row 14: singular grammar at a one-decision scope."""
    _seed_dump(ws)
    call, _key, _verb, surface = _4A_DRIVERS[driver]
    resp = call(None, ws, scope="solo")
    assert [h["slug"] for h in resp["active_decisions"]] == ["only-one"]
    assert resp["note"] == _dump_envelope_note(surface, 1, 1, scope="solo")


@pytest.mark.parametrize("driver", ["mcp-surface", "cli-surface"])
def test_the_dump_thins_by_position_at_a_wider_limit(ws, driver):
    """Row 15: 4b's rank-by-position thinning is unchanged on a seven-wide dump."""
    config, _ = ws
    _seed_dump(ws)
    call, _key, _verb, surface = _4A_DRIVERS[driver]
    resp = call(None, ws, scope="y", limit=7, full_top=2)
    assert _whole(resp["active_decisions"]) == [True, True] + [False] * 5
    assert resp["rejected_paths_withheld"] == 5
    assert resp["note"] == f"{_dump_envelope_note(surface, 7, 8)} {_clause_for(surface, config, 5)}"
