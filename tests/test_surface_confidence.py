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
from mitos.recall import (_SURFACE_POINTERS, assess_query_recall,
                          assess_surface_recall, SURFACE_STRONG_THRESHOLD,
                          SURFACE_WEAK_THRESHOLD)


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
        return json.loads(mcp_server.surface_decisions(query, scope=scope, project=config.workspace_dir))


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
    return _SURFACE_POINTERS[surface]["precedent_scan"].format(
        project=repr(config.project)
    )


def _query_note(surface, config=None, **kw):
    return assess_query_recall(config=config or _StubConfig(), surface=surface, **kw)[1]


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
    conf, note = assess_query_recall(top_score=0.91, result_count=2,
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
    conf, note = assess_query_recall(top_score=0.61, result_count=3,
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
    conf, note = assess_query_recall(top_score=0.55, result_count=3,
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
    conf, note = assess_query_recall(top_score=None, result_count=0,
                                     config=_StubConfig(), surface="cli")
    assert conf == "none"
    assert "no settled precedent" not in note
    assert "mitos surface -p" in note


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
    for banned in ("decide", "record it", "before deciding", "the scope is populated",
                   "no settled precedent"):
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


@pytest.mark.parametrize("label,kw", _QUERY_CASES)
def test_the_mcp_query_note_is_no_longer_than_the_one_it_displaces(label, kw):
    """The comparative bound (D3), stated where nothing data-dependent muddies it.

    On MCP both pointers are bare call-forms, so the whole composed note compares
    like for like: the `query` register may not be wordier than the `surface`
    sentence it displaces at the same inputs. The register is a per-answer cost
    (P15) — it earns its bytes by replacing prose, not by adding to it.
    """
    q = _query_note("mcp", **kw)
    s = assess_surface_recall(semantic_ran=True, scope=None, surface="mcp", **kw)[1]
    assert len(q) <= len(s), f"{label}: query note {len(q)} > surface note {len(s)}"


@pytest.mark.parametrize("label,kw", _QUERY_CASES)
def test_the_cli_query_prose_is_no_longer_than_the_one_it_displaces(label, kw):
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
    q = _query_note("cli", config=config, **kw)
    s = assess_surface_recall(semantic_ran=True, scope=None, surface="cli", **kw)[1]
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
        assert assess_query_recall(config=_StubConfig(), surface="cli", **kw)[0] is not None


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
    s = assess_surface_recall(semantic_ran=True, scope=None, surface=surface, **kw)[1]
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
    surface_note = assess_surface_recall(semantic_ran=True, top_score=0.91,
                                         result_count=1, scope=None, surface="cli")[1]
    assert resp["confidence"] == "strong"
    assert resp["note"] != surface_note
    assert resp["note"] == _query_note("cli", config=config, top_score=0.91, result_count=1)


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

    def query(self, vector, limit=5):
        raise RuntimeError("qdrant fell over")


class _MissingCollection:
    """Qdrant is up and says that collection does not exist."""

    def query(self, vector, limit=5):
        raise CollectionMissingError(
            "Qdrant collection 'mitos-tmp-absent' does not exist.",
            collection="mitos-tmp-absent",
        )


class _NoMatches:
    """Present, and simply returns nothing — the healthy-empty arm's other shape."""

    def query(self, vector, limit=5):
        return []


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


def _cli_query(matches, ws, query="a claim that is not any slug", as_json=False):
    """Drives `cmd_query` end to end and returns the raw captured stdout.

    `matches=None` drives the degraded (no embed/vector) path; a vector-store
    instance is passed through as-is, so a fault stub drives its own route.
    """
    from mitos import cli
    config, _ = ws
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
            cmd_query(config, query, as_json=as_json)
    return buf.getvalue()


def _cli_query_json(matches, ws, query="a claim that is not any slug"):
    return json.loads(_cli_query(matches, ws, query=query, as_json=True))


def _mcp_query(matches, ws, query="a claim that is not any slug"):
    from mitos import mcp_server
    config, _ = ws
    store = GraphStore(config.db_path, read_only=True)
    vector = _FakeVector(matches) if isinstance(matches, list) else matches
    embed = None if matches is None else _FakeEmbed()
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, embed, vector)):
        return json.loads(mcp_server.query_decisions(query, project=config.workspace_dir))


_BAND_LINE_PREFIX = "⚠ confidence:"


def _band_lines(out):
    """The band LINE only — never a bare `⚠` search, which the modifier marker on a
    ranked text render already satisfies."""
    return [ln for ln in out.splitlines() if ln.startswith(_BAND_LINE_PREFIX)]


def _cli_surface_text(matches, ws, query="some claim", scope=None):
    """`cmd_surface`'s text render — the module drove only its `--json` twin."""
    from mitos import cli
    config, _ = ws
    manager = MitosSyncManager(config)
    manager.embed_provider = _FakeEmbed()
    manager.vector_store = _FakeVector(matches)
    buf = io.StringIO()
    with patch.object(cli, "MitosSyncManager", return_value=manager):
        with redirect_stdout(buf):
            cmd_surface(config, query, scope=scope)
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
    # while the text branch had printed this note since 2d.
    assert "superseded" in resp["note"] and "mitos list --state all" in resp["note"]


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
    assert "superseded" in out["note"] and 'list_decisions(state="all")' in out["note"]


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
    assert (cli_resp["note"].replace(_query_pointer("cli", config),
                                     _query_pointer("mcp", config))
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
    assert "graph is unbuilt" in out and "mitos sync" in out
    assert _band_lines(out) == []
    assert _query_note("cli", config=config, top_score=None, result_count=0) not in out


@pytest.mark.parametrize("driver", ["cli", "mcp"])
def test_the_unbuilt_override_keeps_the_label_on_the_machine_encodings(cloned, driver):
    """Precedence is confidence note < blackout < unbuilt, and it is the NOTE that
    yields. The band note is assigned first and the overrides reassign it — the
    same straight-line order `cmd_surface` has run since W31."""
    out = (_cli_query_json([], cloned) if driver == "cli" else _mcp_query([], cloned))
    assert out["confidence"] == "none"
    assert "graph is unbuilt" in out["note"] and "mitos sync" in out["note"]
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
