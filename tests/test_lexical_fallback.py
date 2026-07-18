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
from mitos.errors import DatabaseError, EmbeddingError
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
        monkeypatch.chdir(config.workspace_dir)
        from mitos.cli import main
        with patch("sys.argv", ["mitos", "surface", "zebra quantum"]):
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
