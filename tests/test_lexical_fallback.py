"""Tests for the deterministic lexical fallback on the semantic read verbs.

ADR ``read-verbs-degrade-to-lexical-decisions-md-fallback``: when semantic
recall or the graph is unavailable for any reason, ``surface``/``query`` (CLI
and MCP twins) degrade to a case-insensitive term-match over decisions.md —
presented honestly as a grep over the markdown corpus (decisions.md and
decisions/archive/; degraded header, ``degraded: "lexical"`` JSON
marker, no ``confidence``), modifier-stamped when the graph is readable, with
a stamps-unavailable disclosure when it is not. The clean-empty "No active
precedents found" header must never co-occur with a degraded note.
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
from mitos.cli import cmd_init, cmd_query, cmd_surface
from mitos.errors import (CollectionMissingError, DatabaseError, EmbeddingError,
                          VectorStoreError)
from mitos.lexical import (
    DISPLAY_STOPWORDS,
    LEXICAL_MIN_TERM_LEN,
    degraded_reason_from_error,
    lexical_fallback,
    _query_terms,
)
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


def _rec(m, slug, axiom=None, **kwargs):
    res = m.record_decision_entry(
        axiom or f"Axiom for {slug}.", f"Rejected for {slug}.", [], slug=slug,
        **kwargs,
    )
    assert "error" not in res, res
    return res


def _capture(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


class _Boom:
    """Embedding provider whose query embedding raises (e.g. a 429)."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def get_embedding(self, text, is_query=False):
        raise self.exc


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestTermMatching:
    def test_terms_drop_short_and_dedupe(self):
        assert _query_terms("to be or NOT to be Cache cache") == ["not", "cache"]

    def test_reason_classifies_429_not_raw_blob(self):
        exc = EmbeddingError(
            '429 {"error": {"status": "RESOURCE_EXHAUSTED", "message": "..."}}'
        )
        reason = degraded_reason_from_error(exc, surface="cli", project="p")
        assert reason == "embedding provider rate-limited (429)"  # the shipped phrase
        assert "RESOURCE_EXHAUSTED" not in reason

    def test_reason_pre_v1a(self):
        exc = DatabaseError("This graph predates the V1a schema (a prototype ...)")
        assert "V1a" in degraded_reason_from_error(exc, surface="cli", project="p")

    def test_reason_none_means_unwired(self):
        for surface in ("cli", "mcp"):
            assert degraded_reason_from_error(
                None, surface=surface, project="p") == "embeddings/Qdrant unavailable"

    def test_reason_collection_missing_beats_the_vector_store_arm(self):
        """G1: the subclass arm must precede the one it subclasses, or it is dead.

        ``isinstance(exc, VectorStoreError)`` is True for a
        ``CollectionMissingError``, so an arm ordered after it never runs and every
        read reports "Qdrant unavailable" — the blame-the-infrastructure phrase this
        phase exists to remove, on the most-used read verb. The classifier is the
        single shared leaf behind all four read surfaces, so the ordering is worth
        pinning here as well as through them.
        """
        exc = CollectionMissingError(
            "Qdrant collection 'mitos-x' does not exist", collection="mitos-x"
        )
        reason = degraded_reason_from_error(exc, surface="cli", project="p")

        assert "mitos-x" in reason
        assert "mitos reconcile" in reason
        assert "Qdrant unavailable" not in reason
        # The broad arm still answers for a genuine outage.
        assert degraded_reason_from_error(
            VectorStoreError("Qdrant connection refused"), surface="cli", project="p"
        ) == "Qdrant unavailable"

    def test_reason_collection_missing_without_a_name_still_reads(self):
        """The name is an affordance, not a dependency — an unnamed instance degrades."""
        reason = degraded_reason_from_error(
            CollectionMissingError("gone"), surface="cli", project="p")
        assert "collection missing" in reason
        assert "mitos reconcile" in reason
        assert "''" not in reason  # no empty-quote artefact


class TestLexicalFallbackCore:
    def _md(self, tmp_path, entries):
        p = tmp_path / "decisions.md"
        marker = (
            "<!-- BEGIN ENTRIES — new decisions go directly below this line, "
            "newest first -->"
        )
        blocks = [marker]
        for slug, axiom in entries:
            blocks.append(
                f"### {slug}\n\n**Decided:** {axiom}\n**Rejected:** none.\n"
            )
        p.write_text("# Decisions\n\n" + "\n\n".join(blocks), encoding="utf-8")
        return str(p)

    def test_ranking_by_distinct_terms_then_recency(self, tmp_path):
        path = self._md(tmp_path, [
            ("newer-cache-entry", "About cache things."),
            ("older-cache-strategy", "The cache strategy for redis."),
            ("unrelated", "Totally different."),
        ])
        env = lexical_fallback("cache strategy", corpus_paths=[path], reason="test", store=None)
        slugs = [m["slug"] for m in env["matches"]]
        # older-cache-strategy matches 2 terms → first; newer-cache-entry 1 term.
        assert slugs == ["older-cache-strategy", "newer-cache-entry"]
        # Tie-break check: two 1-term matches keep file order (newer first).
        env2 = lexical_fallback("cache", corpus_paths=[path], reason="test", store=None)
        assert [m["slug"] for m in env2["matches"]] == [
            "newer-cache-entry", "older-cache-strategy",
        ]

    def test_envelope_shape_no_confidence_no_scores(self, tmp_path):
        path = self._md(tmp_path, [("cache-entry", "A cache axiom.")])
        env = lexical_fallback("cache", corpus_paths=[path], reason="test cause", store=None)
        assert env["degraded"] == "lexical"
        assert env["degraded_reason"] == "test cause"
        assert "confidence" not in env
        assert env["stamps_unavailable"] is True
        m = env["matches"][0]
        assert "score" not in m and "confidence" not in m
        assert m["rejected_paths"] == "none."
        assert "Semantic recall unavailable (test cause)" in env["note"]
        assert "stamps not applied" in env["note"]

    def test_limit_and_brief(self, tmp_path):
        # The boundaries' `brief` reaches this leaf as `full_top=0`.
        path = self._md(tmp_path, [(f"cache-{i}", "cache") for i in range(6)])
        env = lexical_fallback("cache", corpus_paths=[path], reason="r", store=None, limit=3,
                               full_top=0)
        assert len(env["matches"]) == 3
        assert all("rejected_paths" not in m for m in env["matches"])

    def test_full_top_thins_by_position_among_returned_matches(self, tmp_path):
        path = self._md(tmp_path, [(f"cache-{i}", "cache") for i in range(6)])
        env = lexical_fallback("cache", corpus_paths=[path], reason="r", store=None, limit=4,
                               full_top=2)
        assert ["rejected_paths" in m for m in env["matches"]] == [True, True, False, False]
        # The leaf has no surface: the count and its clause are the boundaries' to add.
        assert "rejected_paths_withheld" not in env

    def test_zero_matches_notice(self, tmp_path):
        path = self._md(tmp_path, [("cache-entry", "A cache axiom.")])
        env = lexical_fallback("zebra quantum", corpus_paths=[path], reason="r", store=None)
        assert env["matches"] == []
        # The grep pointer names the whole corpus: after rotation the buffer alone
        # is the wrong place to grep "to be sure".
        assert "`decisions.md` and `decisions/archive/`" in env["note"]


# ---------------------------------------------------------------------------
# F2 — display stopwords: dropped from `matched_terms`, counted, never re-ranked
# ---------------------------------------------------------------------------

# The EC's own query (F2, live): mostly function words around three content terms.
_EC_QUERY = "how does the pause decide which neighbours to show to the author"
_CLEAN_QUERY = "cache strategy redis eviction"

_GOLDEN_CORPUS = [
    ("pause-display-rule", "How the review pause does show which entries matter."),
    ("which-way-does-the-flow-go", "How does the thing work, and which way."),
    ("cache-strategy", "The cache strategy decides where results live."),
    ("author-credit", "Author credit lines are kept."),
    ("redis-eviction", "Redis eviction runs nightly."),
    ("unrelated", "Totally different."),
]

# Captured on `9ef2f8a` (7c), before F2 touched `lexical.py`: `lexical_fallback(q,
# corpus_paths=[<_GOLDEN_CORPUS>], reason="golden", store=None)`, `json.dumps` with
# default separators and `ensure_ascii=False`. The pre-change envelopes F2 is held to.
_GOLDEN_NOTE = (
    "Semantic recall unavailable (golden) — deterministic text match over the markdown "
    "corpus (decisions.md + decisions/archive/) (degraded): Graph unavailable — "
    "state/modifier stamps not applied; entries come straight from the markdown corpus "
    "and may include superseded ones."
)
_GOLDEN_EC = (
    '{"degraded": "lexical", "degraded_reason": "golden", "matches": ['
    '{"slug": "pause-display-rule", "axiom": "How the review pause does show which '
    'entries matter.", "scope": [], "matched_terms": ["how", "does", "the", "pause", '
    '"which", "show"], "rejected_paths": "none."}, '
    '{"slug": "which-way-does-the-flow-go", "axiom": "How does the thing work, and '
    'which way.", "scope": [], "matched_terms": ["how", "does", "the", "which"], '
    '"rejected_paths": "none."}, '
    '{"slug": "cache-strategy", "axiom": "The cache strategy decides where results '
    'live.", "scope": [], "matched_terms": ["the", "decide"], "rejected_paths": '
    '"none."}, '
    '{"slug": "author-credit", "axiom": "Author credit lines are kept.", "scope": [], '
    '"matched_terms": ["author"], "rejected_paths": "none."}], '
    '"stamps_unavailable": true, "note": ' + json.dumps(_GOLDEN_NOTE, ensure_ascii=False)
    + '}'
)
_GOLDEN_CLEAN = (
    '{"degraded": "lexical", "degraded_reason": "golden", "matches": ['
    '{"slug": "cache-strategy", "axiom": "The cache strategy decides where results '
    'live.", "scope": [], "matched_terms": ["cache", "strategy"], "rejected_paths": '
    '"none."}, '
    '{"slug": "redis-eviction", "axiom": "Redis eviction runs nightly.", "scope": [], '
    '"matched_terms": ["redis", "eviction"], "rejected_paths": "none."}], '
    '"stamps_unavailable": true, "note": ' + json.dumps(_GOLDEN_NOTE, ensure_ascii=False)
    + '}'
)


class TestDisplayStopwords:
    def _run(self, tmp_path, query, entries=_GOLDEN_CORPUS):
        path = TestLexicalFallbackCore()._md(tmp_path, entries)
        return lexical_fallback(query, corpus_paths=[path], reason="golden", store=None)

    def test_stopwords_dropped_and_counted_in_query_order(self, tmp_path):
        """R9: the EC's noise leaves the display, is counted, and order is kept."""
        m = self._run(tmp_path, _EC_QUERY)["matches"][0]
        assert m["slug"] == "pause-display-rule"
        assert m["matched_terms"] == ["pause", "show"]
        assert m["stopwords_dropped"] == 4

    def test_a_stopword_only_hit_is_still_returned_with_its_count(self, tmp_path):
        """R10: an empty display list carries a non-zero count; the hit stays."""
        env = self._run(tmp_path, _EC_QUERY)
        [m] = [m for m in env["matches"] if m["slug"] == "which-way-does-the-flow-go"]
        assert m["matched_terms"] == []
        assert m["stopwords_dropped"] == 4

    def test_rank_still_counts_every_matched_term(self, tmp_path):
        """R11: three stopwords outrank one content term — the display filter is not
        a scoring filter, so the first hit explains itself by its count alone."""
        env = self._run(tmp_path, "which does the author", [
            ("author-note", "Author notes."),
            ("which-does-the", "Which does the job."),
        ])
        first, second = env["matches"]
        assert first["slug"] == "which-does-the"
        assert first["matched_terms"] == [] and first["stopwords_dropped"] == 3
        assert second["slug"] == "author-note" and second["matched_terms"] == ["author"]
        assert "stopwords_dropped" not in second

    def test_order_and_matches_equal_the_pre_change_golden(self, tmp_path):
        """R12: same slugs in the same order; each match is the golden's once its
        stopwords are removed from `matched_terms` and the count is dropped."""
        env = self._run(tmp_path, _EC_QUERY)
        golden = json.loads(_GOLDEN_EC)
        assert [m["slug"] for m in env["matches"]] == [m["slug"] for m in golden["matches"]]
        for got, was in zip(env["matches"], golden["matches"]):
            got = dict(got)
            dropped = got.pop("stopwords_dropped", 0)
            filtered = [t for t in was["matched_terms"] if t not in DISPLAY_STOPWORDS]
            assert dropped == len(was["matched_terms"]) - len(filtered)
            assert got == {**was, "matched_terms": filtered}

    def test_absent_when_zero_the_envelope_is_byte_identical(self, tmp_path):
        """R13: a query with no stopword among its hits returns today's bytes."""
        env = self._run(tmp_path, _CLEAN_QUERY)
        assert json.dumps(env, ensure_ascii=False) == _GOLDEN_CLEAN
        assert all("stopwords_dropped" not in m for m in env["matches"])

    def test_list_hygiene(self):
        """R14: no dead entries, casefolded, the EC's noise in, `show` out."""
        for word in DISPLAY_STOPWORDS:
            assert word == word.casefold()
            assert len(word) >= LEXICAL_MIN_TERM_LEN, word
        assert {"how", "does", "the", "which"} <= DISPLAY_STOPWORDS
        assert "show" not in DISPLAY_STOPWORDS


# ---------------------------------------------------------------------------
# CLI wiring — each failure mode routes to the fallback
# ---------------------------------------------------------------------------


class TestCliFailureModes:
    def test_surface_embed_error_routes_to_fallback(self, ws):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        exc = EmbeddingError('429 {"status": "RESOURCE_EXHAUSTED"}')
        with patch("mitos.cli.MitosSyncManager") as MM:
            mgr = MitosSyncManager(config)
            mgr.embed_provider = _Boom(exc)
            mgr.vector_store = object()
            MM.return_value = mgr
            out = _capture(cmd_surface, config, "cache strategy")
        assert "Semantic recall unavailable" in out
        assert "429" in out
        assert "RESOURCE_EXHAUSTED" not in out
        assert "cache-strategy" in out
        assert "No active precedents found" not in out

    def test_surface_no_providers_routes_to_fallback(self, ws):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        out = _capture(cmd_surface, config, "cache strategy")
        assert "deterministic text match over the markdown corpus" in out
        assert "cache-strategy" in out
        assert "No active precedents found" not in out

    def test_text_never_prints_terms_while_json_carries_both_keys(self, ws):
        """R15: two queries over one entry that differ only in their matched terms
        and dropped count print the same text; `--json` carries both keys."""
        config, m = ws
        _rec(m, "pause-display-rule", "How the review pause does show which entries matter.")
        noisy = _capture(cmd_surface, config, "the pause show")
        plain = _capture(cmd_surface, config, "pause")
        assert "pause-display-rule" in noisy
        assert noisy == plain
        assert "stopwords_dropped" not in noisy and "matched_terms" not in noisy
        match = json.loads(_capture(cmd_surface, config, "the pause show",
                                    as_json=True))["matches"][0]
        assert match["matched_terms"] == ["pause", "show"]
        assert match["stopwords_dropped"] == 1

    def test_surface_json_degraded_marker(self, ws):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        out = _capture(cmd_surface, config, "cache strategy", as_json=True)
        data = json.loads(out)
        assert data["degraded"] == "lexical"
        assert isinstance(data["degraded_reason"], str)
        assert "confidence" not in data
        assert data["matches"][0]["slug"] == "cache-strategy"

    def test_surface_pre_v1a_graph_falls_back_without_graph(self, ws):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        exc = DatabaseError(
            "This graph predates the V1a schema (a prototype layout was "
            "detected)."
        )
        with patch("mitos.cli.MitosSyncManager", side_effect=exc):
            out = _capture(cmd_surface, config, "cache strategy", as_json=True)
        data = json.loads(out)
        assert data["degraded"] == "lexical"
        assert "V1a" in data["degraded_reason"]
        assert data["stamps_unavailable"] is True
        assert data["matches"][0]["slug"] == "cache-strategy"
        # No state/modifier stamps without a graph.
        assert "state" not in data["matches"][0]

    def test_surface_modifier_stamps_when_graph_readable(self, ws):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        _rec(m, "cache-strategy-amendment", "Amend the cache strategy.",
             amends="cache-strategy")
        out = _capture(cmd_surface, config, "cache write-through", as_json=True)
        data = json.loads(out)
        assert data["degraded"] == "lexical"
        by_slug = {mm["slug"]: mm for mm in data["matches"]}
        assert by_slug["cache-strategy"]["amended_by"] == [
            "cache-strategy-amendment"
        ]
        assert by_slug["cache-strategy"]["state"] == "active"

    def test_surface_superseded_filtered_when_graph_readable(self, ws):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        _rec(m, "cache-strategy-v2", "Use a write-back cache.",
             supersedes="cache-strategy")
        out = _capture(cmd_surface, config, "cache", as_json=True)
        data = json.loads(out)
        slugs = [mm["slug"] for mm in data["matches"]]
        assert "cache-strategy-v2" in slugs
        assert "cache-strategy" not in slugs

    def test_query_embed_error_routes_to_fallback(self, ws):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        exc = EmbeddingError('429 {"status": "RESOURCE_EXHAUSTED"}')
        with patch("mitos.cli.MitosSyncManager") as MM:
            mgr = MitosSyncManager(config)
            mgr.embed_provider = _Boom(exc)
            mgr.vector_store = object()
            MM.return_value = mgr
            out = _capture(cmd_query, config, "cache strategy")
        assert "Semantic recall unavailable" in out
        assert "RESOURCE_EXHAUSTED" not in out
        assert "cache-strategy" in out

    def test_query_no_providers_routes_to_fallback(self, ws):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        out = _capture(cmd_query, config, "cache strategy", as_json=True)
        data = json.loads(out)
        assert data["degraded"] == "lexical"
        assert data["matches"][0]["slug"] == "cache-strategy"

    def test_query_pre_v1a_falls_back(self, ws):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        exc = DatabaseError("This graph predates the V1a schema.")
        with patch("mitos.cli.MitosSyncManager", side_effect=exc):
            out = _capture(cmd_query, config, "cache", as_json=True)
        data = json.loads(out)
        assert data["degraded"] == "lexical"
        assert data["stamps_unavailable"] is True

    def test_no_lexical_match_still_degraded_never_clean_empty(self, ws):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        out = _capture(cmd_surface, config, "zebra quantum entanglement")
        assert "Semantic recall unavailable" in out
        assert "`decisions.md` and `decisions/archive/`" in out
        assert "No active precedents found" not in out

    def test_exit_code_zero_via_main(self, ws, monkeypatch):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        from mitos.cli import main
        with patch("sys.argv", ["mitos", "-p", config.workspace_dir,
                                "surface", "zebra quantum"]):
            rc = main()
        assert rc in (0, None)


# ---------------------------------------------------------------------------
# MCP twins
# ---------------------------------------------------------------------------


class TestMcpParity:
    def _components(self, config, embed=None, vec=None):
        from mitos.store import GraphStore
        store = GraphStore(config.db_path, read_only=True)
        return store, embed, vec

    def test_mcp_surface_embed_error(self, ws):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        from mitos import mcp_server
        exc = EmbeddingError('429 {"status": "RESOURCE_EXHAUSTED"}')
        comps = self._components(config, embed=_Boom(exc), vec=object())
        with patch.object(mcp_server, "get_workspace_components",
                          return_value=comps):
            out = json.loads(mcp_server.surface_decisions("cache strategy", project=config.workspace_dir))
        assert out["degraded"] == "lexical"
        assert "429" in out["degraded_reason"]
        assert "RESOURCE_EXHAUSTED" not in out["degraded_reason"]
        assert out["matches"][0]["slug"] == "cache-strategy"
        assert "confidence" not in out

    def test_mcp_matches_carry_the_cli_json_terms_and_count(self, ws):
        """R15, MCP twin: per match, the same `matched_terms` and `stopwords_dropped`
        as CLI `--json` over the same corpus and query."""
        config, m = ws
        _rec(m, "pause-display-rule", "How the review pause does show which entries matter.")
        _rec(m, "which-does-the", "Which does the job.")
        from mitos import mcp_server
        cli_env = json.loads(_capture(cmd_surface, config, _EC_QUERY, as_json=True))
        with patch.object(mcp_server, "get_workspace_components",
                          return_value=self._components(config)):
            mcp_env = json.loads(mcp_server.surface_decisions(
                _EC_QUERY, project=config.workspace_dir))
        assert mcp_env["degraded"] == "lexical"

        def evidence(env):
            return [(d["slug"], d["matched_terms"], d.get("stopwords_dropped"))
                    for d in env["matches"]]
        assert evidence(mcp_env) == evidence(cli_env)
        assert ("which-does-the", [], 3) in evidence(mcp_env)

    def test_mcp_surface_pre_v1a(self, ws):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        from mitos import mcp_server
        exc = DatabaseError("This graph predates the V1a schema.")
        with patch.object(mcp_server, "get_workspace_components",
                          side_effect=exc):
            out = json.loads(mcp_server.surface_decisions("cache", project=config.workspace_dir))
        assert out["degraded"] == "lexical"
        assert out["stamps_unavailable"] is True

    def test_mcp_query_no_providers(self, ws):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        from mitos import mcp_server
        comps = self._components(config)
        with patch.object(mcp_server, "get_workspace_components",
                          return_value=comps):
            out = json.loads(mcp_server.query_decisions("cache strategy", project=config.workspace_dir))
        assert out["degraded"] == "lexical"
        assert out["matches"][0]["slug"] == "cache-strategy"
        assert "error" not in out

    def test_mcp_query_embed_error(self, ws):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        from mitos import mcp_server
        exc = EmbeddingError("boom connection refused")
        comps = self._components(config, embed=_Boom(exc), vec=object())
        with patch.object(mcp_server, "get_workspace_components",
                          return_value=comps):
            out = json.loads(mcp_server.query_decisions("cache strategy", project=config.workspace_dir))
        assert out["degraded"] == "lexical"
        assert "error" not in out
        assert out["matches"][0]["slug"] == "cache-strategy"

    def test_mcp_stamps_when_graph_readable(self, ws):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        _rec(m, "cache-strategy-amendment", "Amend the cache strategy.",
             amends="cache-strategy")
        from mitos import mcp_server
        comps = self._components(config)
        with patch.object(mcp_server, "get_workspace_components",
                          return_value=comps):
            out = json.loads(mcp_server.surface_decisions("cache", project=config.workspace_dir))
        by_slug = {mm["slug"]: mm for mm in out["matches"]}
        assert by_slug["cache-strategy"]["amended_by"] == [
            "cache-strategy-amendment"
        ]


# ---------------------------------------------------------------------------
# I8 — an absent Qdrant collection on the four semantic read surfaces
#
# The two things that must be true at once, and holding both IS the gate:
#   * absence over a POPULATED graph speaks — a real hole in recall, worded as
#     itself with `mitos reconcile`, never as "Qdrant unavailable";
#   * absence over an EMPTY graph stays quiet — a just-initialized project has an
#     empty index by definition, and making that read as broken would break the
#     "empty/fresh is healthy" line on the very first `mitos query` a new keyed
#     project runs.
#
# A fixture with only the populated row passes under EITHER behaviour, which is
# exactly how the regression would ship. Both halves, all four surfaces.
# ---------------------------------------------------------------------------

_ABSENT = "mitos-tmp-absent-collection"


class _MissingCollection:
    """A vector store answering: Qdrant is up, that collection does not exist."""

    def query(self, vector, limit=5):
        raise CollectionMissingError(
            f"Qdrant collection '{_ABSENT}' does not exist "
            "(Qdrant is up and answered 404 to the query).",
            collection=_ABSENT,
        )


class _Embeds:
    """An embedding provider that succeeds — the fault under test is downstream."""

    def get_embedding(self, text, is_query=False):
        return [0.1, 0.2, 0.3]


class TestAbsentCollectionOnTheReadSurfaces:
    def _cli(self, config, verb, **kwargs):
        with patch("mitos.cli.MitosSyncManager") as MM:
            mgr = MitosSyncManager(config)
            mgr.embed_provider = _Embeds()
            mgr.vector_store = _MissingCollection()
            MM.return_value = mgr
            return _capture(verb, config, "cache strategy", **kwargs)

    def _mcp(self, config, tool):
        from mitos.store import GraphStore
        from mitos import mcp_server
        comps = (GraphStore(config.db_path, read_only=True),
                 _Embeds(), _MissingCollection())
        with patch.object(mcp_server, "get_workspace_components",
                          return_value=comps):
            return json.loads(getattr(mcp_server, tool)("cache strategy", project=config.workspace_dir))

    # -- populated graph: absence announces itself, by name, with the heal ----

    @pytest.mark.parametrize("verb_name", ["cmd_query", "cmd_surface"])
    def test_cli_populated_graph_names_the_collection_and_the_heal(self, ws, verb_name):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        verb = {"cmd_query": cmd_query, "cmd_surface": cmd_surface}[verb_name]

        out = self._cli(config, verb)

        assert "Semantic recall unavailable" in out
        assert _ABSENT in out
        assert "mitos reconcile" in out
        # The phrase this whole phase exists to stop: Qdrant is RUNNING.
        assert "Qdrant unavailable" not in out
        assert "Traceback" not in out
        assert "cache-strategy" in out          # the lexical fallback still answers

    @pytest.mark.parametrize("tool", ["query_decisions", "surface_decisions"])
    def test_mcp_populated_graph_names_the_collection_and_the_heal(
        self, ws, tool
    ):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")

        out = self._mcp(config, tool)

        assert out["degraded"] == "lexical"
        # 7a inverted this row on purpose: it pinned `mitos reconcile` on MCP, a shell
        # command handed to an agent (ROADMAP G14). The collection is still named and
        # the heal still stated, in the MCP register: a fact and a person as the actor.
        assert _ABSENT in out["degraded_reason"]
        assert "mitos " not in out["degraded_reason"]
        assert "reconcile" in out["degraded_reason"]
        assert "a person with a shell" in out["degraded_reason"]
        assert "Qdrant unavailable" not in out["degraded_reason"]
        assert out["matches"][0]["slug"] == "cache-strategy"

    # -- empty graph: the ordinary nothing-found result, no diagnostic --------

    def test_cli_query_empty_graph_stays_the_ordinary_miss(self, ws):
        config, _m = ws
        out = self._cli(config, cmd_query)

        assert "No matching decisions found." in out
        assert "Semantic recall unavailable" not in out
        assert "reconcile" not in out

    def test_cli_query_empty_graph_json_is_the_clean_envelope(self, ws):
        config, _m = ws
        data = json.loads(self._cli(config, cmd_query, as_json=True))

        assert data["matches"] == []
        assert "degraded" not in data
        assert "all_superseded" not in data
        assert data["collection"]                # provenance still rides

    def test_cli_surface_empty_graph_stays_the_ordinary_miss(self, ws):
        config, _m = ws
        out = self._cli(config, cmd_surface)

        assert "No active precedents found" in out
        assert "Semantic recall unavailable" not in out

    @pytest.mark.parametrize("tool", ["query_decisions", "surface_decisions"])
    def test_mcp_empty_graph_is_the_clean_envelope(self, ws, tool):
        config, _m = ws
        out = self._mcp(config, tool)

        assert "degraded" not in out
        assert "degraded_reason" not in out
        assert out.get("matches", out.get("active_decisions")) == []

    # -- the gate reads the set `reconcile` would index, not just decisions ---

    def test_a_graph_holding_only_an_open_question_still_speaks(self, ws):
        """``get_active_node_ids`` is decisions ∪ open questions — the set the heal covers.

        Gated on ``get_active_decisions`` instead, a workspace whose only content is
        a parked open question would read as clean-empty while ``mitos reconcile``
        did in fact have a node to index. The gate and the heal must agree by
        construction, not by coincidence.
        """
        from mitos.parser import ParsedEntry
        from mitos.store import GraphStore

        config, _m = ws
        oq = ParsedEntry("open_question", "an-unsettled-topic", 1, 5)
        oq.topic = "Cache eviction policy"
        oq.questions_raised = ["Which cache eviction policy?"]
        GraphStore(config.db_path).commit_parsed_entry(oq)

        out = self._cli(config, cmd_query)

        assert "Semantic recall unavailable" in out
        assert "mitos reconcile" in out


# ---------------------------------------------------------------------------
# W31 — the UNBUILT graph on the same four semantic read surfaces
#
# The sibling above answers "is an absent COLLECTION a gap?" on the graph. This
# one answers "is an empty GRAPH a gap?" on the corpus, and it is the state the
# absolute-path escape hatch made routine: a clone carries the committed
# `.mitos/config.toml` and a `decisions.md` holding real entries, but not the
# gitignored `*.sqlite`. Every read over it returns the clean empty envelope, and
# the agent that asked reads *no precedents* for a project that has hundreds.
#
# The pair is the fixture, again and for the same reason: the clone AND a fresh
# workspace whose sample-only corpus sits above the `BEGIN ENTRIES` sentinel,
# which must keep answering exactly as it does today. The fresh half is already
# covered by `TestAbsentCollectionOnTheReadSurfaces`' empty-graph rows above —
# they run on the shipped `ws` fixture, which is a bare `cmd_init` — so this class
# adds the clone half and re-asserts the twin only where the composition differs.
# ---------------------------------------------------------------------------

_CLONE_ENTRY = """
### clone-entry-one

**Decided:** A clone carries the corpus but never the graph.
**Rejected:** Committing the binary graph — it is derivative.
**Scope:** clone
"""


@pytest.fixture
def cloned(offline):
    """A workspace with entries below the sentinel and a graph holding no nodes.

    The corpus is seeded BY HAND: `mitos sync` commits nothing without a
    `GEMINI_API_KEY` (it parses, then refuses) and `record` commits to the graph,
    which is the one thing this fixture must not have. The graph file is deleted
    after `init` and then re-created empty — because that is the reachable steady
    state on this surface: `MitosSyncManager` opens the store read-write, so the
    first read over a clone leaves a 0-node `graph.sqlite` behind and every read
    after it sees exactly this shape.
    """
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    cmd_init(config)
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(_CLONE_ENTRY)
    import os as _os
    _os.remove(config.db_path)
    from mitos.store import GraphStore
    assert GraphStore(config.db_path).graph_fingerprint()[0] == 0
    yield config
    shutil.rmtree(tmp, ignore_errors=True)


class TestUnbuiltGraphOnTheReadSurfaces:
    def _cli(self, config, verb, **kwargs):
        """The seam from the class above: a healthy embedder, an absent collection.

        Reused deliberately — an unbuilt clone with a key and a reachable Qdrant
        raises `CollectionMissingError`, `missing_index_is_a_gap` calls the absence
        healthy (the active set IS empty), and the read lands on the ordinary
        empty-result path. That is the exact composition this class is about.
        """
        with patch("mitos.cli.MitosSyncManager") as MM:
            mgr = MitosSyncManager(config)
            mgr.embed_provider = _Embeds()
            mgr.vector_store = _MissingCollection()
            MM.return_value = mgr
            return _capture(verb, config, "cache strategy", **kwargs)

    def _mcp(self, config, tool):
        from mitos.store import GraphStore
        from mitos import mcp_server
        comps = (GraphStore(config.db_path, read_only=True),
                 _Embeds(), _MissingCollection())
        with patch.object(mcp_server, "get_workspace_components",
                          return_value=comps):
            return json.loads(getattr(mcp_server, tool)("cache strategy", project=config.workspace_dir))

    # -- the clone: the empty answer says why it is empty ---------------------

    def test_cli_query_text_names_the_unbuilt_graph_and_rebuild(self, cloned):
        """The heal is `rebuild`, not `sync`, even when the entries sit in the buffer:
        one state, one heal — `rebuild` reads the buffer too, and `sync` reads
        nothing else, so over a rotated corpus it would build nothing.
        """
        out = self._cli(cloned, cmd_query)

        assert "No matching decisions found." in out
        assert "graph is unbuilt" in out
        assert "mitos rebuild -p" in out
        assert "mitos sync" not in out
        assert "reconcile" not in out

    def test_cli_query_json_carries_the_note(self, cloned):
        data = json.loads(self._cli(cloned, cmd_query, as_json=True))

        assert data["matches"] == []
        assert "graph is unbuilt" in data["note"]
        assert "mitos rebuild -p" in data["note"]
        assert data["collection"]              # the provenance stamp still rides

    def test_cli_surface_names_the_unbuilt_graph_and_rebuild(self, cloned):
        out = self._cli(cloned, cmd_surface)

        assert "No active precedents found" in out
        assert "graph is unbuilt" in out
        assert "mitos rebuild -p" in out

    @pytest.mark.parametrize("tool", ["query_decisions", "surface_decisions"])
    def test_mcp_tools_carry_the_note_in_their_own_register(
        self, cloned, tool
    ):
        """Same predicate, same composer, a different closing clause: an agent on
        this surface is handed no shell command (it would run it) and no tool (there
        is none that rebuilds), so the clause states the fact and names a person as
        the next actor — beating letting it hunt for a tool that does not exist.
        """
        out = self._mcp(cloned, tool)

        assert out.get("matches", out.get("active_decisions")) == []
        assert "graph is unbuilt" in out["note"]
        assert "mitos " not in out["note"]
        assert "no tool on this surface performs" in out["note"]
        assert "a person" in out["note"]
        assert "reconcile" not in out["note"]

    def test_the_note_is_not_a_degradation_the_envelope_stays_clean(
        self, cloned
    ):
        """It annotates a successful read; it does not claim the read failed.

        `degraded: "lexical"` means "I could not run semantic recall". Here recall
        ran and there was genuinely nothing indexed — a different fact, and blurring
        the two would put a diagnosis on the wrong axis.
        """
        out = self._mcp(cloned, "surface_decisions")

        assert "degraded" not in out
        assert "degraded_reason" not in out

    # -- the twin: a fresh workspace is unchanged in every respect ------------

    def test_the_fresh_twin_says_nothing_about_a_graph(self, ws):
        config, _m = ws
        out = self._cli(config, cmd_query)

        assert "No matching decisions found." in out
        assert "unbuilt" not in out

    @pytest.mark.parametrize("tool", ["query_decisions", "surface_decisions"])
    def test_the_fresh_twin_mcp_envelope_carries_no_graph_note(
        self, ws, tool
    ):
        config, _m = ws
        out = self._mcp(config, tool)

        assert "unbuilt" not in json.dumps(out)

    def test_a_populated_graph_over_a_populated_corpus_says_nothing_either(self, ws):
        """The control that keeps the gate honest end to end: once anything is
        committed, the note must go away even though the corpus is non-empty.
        """
        config, m = ws
        with open(config.decisions_file, "a", encoding="utf-8") as f:
            f.write(_CLONE_ENTRY)
        _rec(m, "some-other-decision", "An unrelated axiom.")

        out = self._cli(config, cmd_query)

        assert "unbuilt" not in out


class _NoMatches:
    """A vector store that is present and simply returns nothing.

    The other way an empty answer arrives: the collection EXISTS (so nothing
    raises) and the query matched no points — which is what an unbuilt clone looks
    like the moment anything has created its collection. `query_decisions` builds
    two different empty envelopes for the two shapes, so both need a row or the
    verb reads as done while one exit says nothing (3e's per-EXIT lesson).
    """

    def query(self, vector, limit=5):
        return []


class TestUnbuiltGraphOnTheOrdinaryEmptyEnvelope:
    def _mcp(self, config, tool, vector_store):
        from mitos.store import GraphStore
        from mitos import mcp_server
        comps = (GraphStore(config.db_path, read_only=True), _Embeds(), vector_store)
        with patch.object(mcp_server, "get_workspace_components",
                          return_value=comps):
            return json.loads(getattr(mcp_server, tool)("cache strategy", project=config.workspace_dir))

    @pytest.mark.parametrize("tool", ["query_decisions", "surface_decisions"])
    def test_a_present_but_empty_collection_still_names_the_unbuilt_graph(
        self, cloned, tool
    ):
        out = self._mcp(cloned, tool, _NoMatches())

        assert out.get("matches", out.get("active_decisions")) == []
        assert "graph is unbuilt" in out["note"]
        assert "a person" in out["note"] and "mitos " not in out["note"]

    @pytest.mark.parametrize("tool", ["query_decisions", "surface_decisions"])
    def test_the_fresh_twin_on_the_same_envelope_says_nothing(
        self, ws, tool
    ):
        config, _m = ws
        out = self._mcp(config, tool, _NoMatches())

        assert "unbuilt" not in json.dumps(out)
