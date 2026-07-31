"""Tests for the deterministic lexical fallback on the semantic read verbs.

ADR ``read-verbs-degrade-to-lexical-decisions-md-fallback``: when semantic
recall or the graph is unavailable for any reason, ``surface``/``query`` (CLI
and MCP twins) degrade to a case-insensitive term-match over decisions.md —
presented honestly as a grep (degraded header, ``degraded: "lexical"`` JSON
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
        reason = degraded_reason_from_error(exc)
        assert "429" in reason
        assert "RESOURCE_EXHAUSTED" not in reason

    def test_reason_pre_v1a(self):
        exc = DatabaseError("This graph predates the V1a schema (a prototype ...)")
        assert "V1a" in degraded_reason_from_error(exc)

    def test_reason_none_means_unwired(self):
        assert "unavailable" in degraded_reason_from_error(None)

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
        reason = degraded_reason_from_error(exc)

        assert "mitos-x" in reason
        assert "mitos reconcile" in reason
        assert "Qdrant unavailable" not in reason
        # The broad arm still answers for a genuine outage.
        assert degraded_reason_from_error(
            VectorStoreError("Qdrant connection refused")
        ) == "Qdrant unavailable"

    def test_reason_collection_missing_without_a_name_still_reads(self):
        """The name is an affordance, not a dependency — an unnamed instance degrades."""
        reason = degraded_reason_from_error(CollectionMissingError("gone"))
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
        env = lexical_fallback("cache strategy", path, reason="test", store=None)
        slugs = [m["slug"] for m in env["matches"]]
        # older-cache-strategy matches 2 terms → first; newer-cache-entry 1 term.
        assert slugs == ["older-cache-strategy", "newer-cache-entry"]
        # Tie-break check: two 1-term matches keep file order (newer first).
        env2 = lexical_fallback("cache", path, reason="test", store=None)
        assert [m["slug"] for m in env2["matches"]] == [
            "newer-cache-entry", "older-cache-strategy",
        ]

    def test_envelope_shape_no_confidence_no_scores(self, tmp_path):
        path = self._md(tmp_path, [("cache-entry", "A cache axiom.")])
        env = lexical_fallback("cache", path, reason="test cause", store=None)
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
        path = self._md(tmp_path, [(f"cache-{i}", "cache") for i in range(6)])
        env = lexical_fallback("cache", path, reason="r", store=None, limit=3,
                               brief=True)
        assert len(env["matches"]) == 3
        assert all("rejected_paths" not in m for m in env["matches"])

    def test_zero_matches_notice(self, tmp_path):
        path = self._md(tmp_path, [("cache-entry", "A cache axiom.")])
        env = lexical_fallback("zebra quantum", path, reason="r", store=None)
        assert env["matches"] == []
        assert "grep decisions.md" in env["note"]


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
        assert "deterministic text match over decisions.md" in out
        assert "cache-strategy" in out
        assert "No active precedents found" not in out

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
        assert "grep decisions.md" in out
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

    def test_mcp_surface_embed_error(self, ws, monkeypatch):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        monkeypatch.chdir(config.workspace_dir)
        from mitos import mcp_server
        exc = EmbeddingError('429 {"status": "RESOURCE_EXHAUSTED"}')
        comps = self._components(config, embed=_Boom(exc), vec=object())
        with patch.object(mcp_server, "get_workspace_components",
                          return_value=comps):
            out = json.loads(mcp_server.surface_decisions("cache strategy"))
        assert out["degraded"] == "lexical"
        assert "429" in out["degraded_reason"]
        assert "RESOURCE_EXHAUSTED" not in out["degraded_reason"]
        assert out["matches"][0]["slug"] == "cache-strategy"
        assert "confidence" not in out

    def test_mcp_surface_pre_v1a(self, ws, monkeypatch):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        monkeypatch.chdir(config.workspace_dir)
        from mitos import mcp_server
        exc = DatabaseError("This graph predates the V1a schema.")
        with patch.object(mcp_server, "get_workspace_components",
                          side_effect=exc):
            out = json.loads(mcp_server.surface_decisions("cache"))
        assert out["degraded"] == "lexical"
        assert out["stamps_unavailable"] is True

    def test_mcp_query_no_providers(self, ws, monkeypatch):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        monkeypatch.chdir(config.workspace_dir)
        from mitos import mcp_server
        comps = self._components(config)
        with patch.object(mcp_server, "get_workspace_components",
                          return_value=comps):
            out = json.loads(mcp_server.query_decisions("cache strategy"))
        assert out["degraded"] == "lexical"
        assert out["matches"][0]["slug"] == "cache-strategy"
        assert "error" not in out

    def test_mcp_query_embed_error(self, ws, monkeypatch):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        monkeypatch.chdir(config.workspace_dir)
        from mitos import mcp_server
        exc = EmbeddingError("boom connection refused")
        comps = self._components(config, embed=_Boom(exc), vec=object())
        with patch.object(mcp_server, "get_workspace_components",
                          return_value=comps):
            out = json.loads(mcp_server.query_decisions("cache strategy"))
        assert out["degraded"] == "lexical"
        assert "error" not in out
        assert out["matches"][0]["slug"] == "cache-strategy"

    def test_mcp_stamps_when_graph_readable(self, ws, monkeypatch):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")
        _rec(m, "cache-strategy-amendment", "Amend the cache strategy.",
             amends="cache-strategy")
        monkeypatch.chdir(config.workspace_dir)
        from mitos import mcp_server
        comps = self._components(config)
        with patch.object(mcp_server, "get_workspace_components",
                          return_value=comps):
            out = json.loads(mcp_server.surface_decisions("cache"))
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

    def _mcp(self, config, monkeypatch, tool):
        from mitos.store import GraphStore
        monkeypatch.chdir(config.workspace_dir)
        from mitos import mcp_server
        comps = (GraphStore(config.db_path, read_only=True),
                 _Embeds(), _MissingCollection())
        with patch.object(mcp_server, "get_workspace_components",
                          return_value=comps):
            return json.loads(getattr(mcp_server, tool)("cache strategy"))

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
        self, ws, monkeypatch, tool
    ):
        config, m = ws
        _rec(m, "cache-strategy", "Use a write-through cache.")

        out = self._mcp(config, monkeypatch, tool)

        assert out["degraded"] == "lexical"
        assert _ABSENT in out["degraded_reason"]
        assert "mitos reconcile" in out["degraded_reason"]
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
    def test_mcp_empty_graph_is_the_clean_envelope(self, ws, monkeypatch, tool):
        config, _m = ws
        out = self._mcp(config, monkeypatch, tool)

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

    def _mcp(self, config, monkeypatch, tool):
        from mitos.store import GraphStore
        monkeypatch.chdir(config.workspace_dir)
        from mitos import mcp_server
        comps = (GraphStore(config.db_path, read_only=True),
                 _Embeds(), _MissingCollection())
        with patch.object(mcp_server, "get_workspace_components",
                          return_value=comps):
            return json.loads(getattr(mcp_server, tool)("cache strategy"))

    # -- the clone: the empty answer says why it is empty ---------------------

    def test_cli_query_text_names_the_unbuilt_graph_and_sync(self, cloned):
        out = self._cli(cloned, cmd_query)

        assert "No matching decisions found." in out
        assert "graph is unbuilt" in out
        assert "mitos sync" in out
        assert "reconcile" not in out

    def test_cli_query_json_carries_the_note(self, cloned):
        data = json.loads(self._cli(cloned, cmd_query, as_json=True))

        assert data["matches"] == []
        assert "graph is unbuilt" in data["note"]
        assert "mitos sync" in data["note"]
        assert data["collection"]              # the provenance stamp still rides

    def test_cli_surface_names_the_unbuilt_graph_and_sync(self, cloned):
        out = self._cli(cloned, cmd_surface)

        assert "No active precedents found" in out
        assert "graph is unbuilt" in out
        assert "mitos sync" in out

    @pytest.mark.parametrize("tool", ["query_decisions", "surface_decisions"])
    def test_mcp_tools_carry_the_note_in_their_own_register(
        self, cloned, monkeypatch, tool
    ):
        """Same predicate, same composer, a different closing clause: an agent on
        this surface cannot run a shell command where it stands, and saying so beats
        letting it hunt for a `sync` tool that does not exist.
        """
        out = self._mcp(cloned, monkeypatch, tool)

        assert out.get("matches", out.get("active_decisions")) == []
        assert "graph is unbuilt" in out["note"]
        assert "mitos sync" in out["note"]
        assert "no tool for it on this surface" in out["note"]
        assert "reconcile" not in out["note"]

    def test_the_note_is_not_a_degradation_the_envelope_stays_clean(
        self, cloned, monkeypatch
    ):
        """It annotates a successful read; it does not claim the read failed.

        `degraded: "lexical"` means "I could not run semantic recall". Here recall
        ran and there was genuinely nothing indexed — a different fact, and blurring
        the two would put a diagnosis on the wrong axis.
        """
        out = self._mcp(cloned, monkeypatch, "surface_decisions")

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
        self, ws, monkeypatch, tool
    ):
        config, _m = ws
        out = self._mcp(config, monkeypatch, tool)

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
    def _mcp(self, config, monkeypatch, tool, vector_store):
        from mitos.store import GraphStore
        monkeypatch.chdir(config.workspace_dir)
        from mitos import mcp_server
        comps = (GraphStore(config.db_path, read_only=True), _Embeds(), vector_store)
        with patch.object(mcp_server, "get_workspace_components",
                          return_value=comps):
            return json.loads(getattr(mcp_server, tool)("cache strategy"))

    @pytest.mark.parametrize("tool", ["query_decisions", "surface_decisions"])
    def test_a_present_but_empty_collection_still_names_the_unbuilt_graph(
        self, cloned, monkeypatch, tool
    ):
        out = self._mcp(cloned, monkeypatch, tool, _NoMatches())

        assert out.get("matches", out.get("active_decisions")) == []
        assert "graph is unbuilt" in out["note"]
        assert "mitos sync" in out["note"]

    @pytest.mark.parametrize("tool", ["query_decisions", "surface_decisions"])
    def test_the_fresh_twin_on_the_same_envelope_says_nothing(
        self, ws, monkeypatch, tool
    ):
        config, _m = ws
        out = self._mcp(config, monkeypatch, tool, _NoMatches())

        assert "unbuilt" not in json.dumps(out)
