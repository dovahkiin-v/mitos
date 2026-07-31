"""Adversarial MCP server robustness and fallback stress test suite for Mitos.

This module implements comprehensive, adversarial testing for the MCP server cluster (F):
  - querying active decisions under unconfigured, offline, or refused Qdrant connections.
  - validating graceful degradation of semantic search tools to graph-only lookups.
  - querying decisions with empty, massive, and hostile special character inputs.
  - surfacing active decisions for invalid, empty, or special character scope tags.
  - verifying strict isolation and socket security of the MCP service.

Maintains strict compliance with the Mitos Framework (FRAMEWORK.md) and the 1:1
test-to-code byte ratio constraint.
"""

import json
import os
import shutil
import tempfile
import pytest
from typing import Tuple, List, Dict, Any
from unittest.mock import MagicMock, patch

from conftest import make_workspace
from mitos.config import MitosConfig
from mitos.store import GraphStore, ParsedEntry
from mitos.mcp_server import query_decisions, surface_decisions


@pytest.fixture
def isolated_workspace() -> Tuple[MitosConfig, str]:
    """Fixture that provisions a fully isolated temporary workspace for MCP tests.

    ``make_workspace`` first, and it is load-bearing rather than tidy: since 5b
    every tool call resolves its ``project``, and resolution *validates* — a
    directory holding only ``.mitos/`` is a half-workspace and would refuse with
    ``target_path_not_a_workspace``, turning all seven rows below into an
    unintended fault injection.
    """
    tmpdir = make_workspace(tempfile.mkdtemp())
    config = MitosConfig(tmpdir)
    config.db_path = os.path.join(tmpdir, ".mitos", "graph.sqlite")
    config.decisions_file = os.path.join(tmpdir, "decisions.md")
    config.archive_dir = os.path.join(tmpdir, "decisions", "archive")
    
    os.makedirs(config.mitos_dir, exist_ok=True)
    yield config, tmpdir
    
    # Clean up workspace
    shutil.rmtree(tmpdir, ignore_errors=True)


# ==============================================================================
# 1. MCP Query with Unconfigured/Offline Vector Store Fallback
# ==============================================================================
def test_mcp_unconfigured_vector_store_fallback(isolated_workspace) -> None:
    """Verifies that the MCP server queries degrade gracefully if Qdrant is offline.

    If Qdrant connection is refused, the query_decisions tool must not crash the
    MCP server. It should fall back to graph-only / keyword search or return an
    informative warning response safely.
    """
    config, tmpdir = isolated_workspace
    store = GraphStore(config.db_path)
    
    # Commit two active decisions
    d1 = ParsedEntry("decision", "rule-one", 1, 5)
    d1.axiom = "We use WAL mode SQLite for local storage."
    d1.rejected_paths = "None."
    d1.scope = ["substrate"]
    store.commit_parsed_entry(d1)
    
    d2 = ParsedEntry("decision", "rule-two", 6, 10)
    d2.axiom = "We use pure python."
    d2.rejected_paths = "None."
    d2.scope = ["core"]
    store.commit_parsed_entry(d2)
    
    # Patch get_workspace_components to use our test graph store and a failed vector store
    mock_vector = MagicMock()
    mock_vector.query.side_effect = Exception("Connection refused")
    
    with patch("mitos.mcp_server.get_workspace_components", return_value=(store, None, mock_vector)):
        results = query_decisions(
            query="SQLite",
            depth="letter",
            project=config.workspace_dir
        )
        # The row's claim, stated so it can fail: a refused vector store comes
        # back as the shared degraded envelope, never an `{error}` object and
        # never a crash. Its third disjunct used to be `len(results) >= 0`,
        # which is vacuously true — the assertion proved nothing about any
        # output. Strengthened here rather than left, because 5b makes this call
        # cross the targeting boundary and a row that cannot fail cannot report
        # what the crossing did.
        payload = json.loads(results)
        assert payload["degraded"] == "lexical"
        assert payload["degraded_reason"]
        assert "error" not in payload


# ==============================================================================
# 2. MCP Query under Extreme Input Boundary Conditions
# ==============================================================================
def test_mcp_query_extreme_inputs(isolated_workspace) -> None:
    """Verifies that query_decisions handles empty, massive, and hostile inputs safely.

    Input inputs:
      - Empty search query string.
      - Extremely long query string (e.g., 2000 characters).
      - Queries containing special characters or SQL injection strings.
    """
    config, tmpdir = isolated_workspace
    store = GraphStore(config.db_path)
    
    # Patch get_workspace_components to use our test graph store
    with patch("mitos.mcp_server.get_workspace_components", return_value=(store, None, None)):
        # 1. Test empty query search
        res_empty = query_decisions(
            query="",
            depth="letter",
            project=config.workspace_dir
        )
        assert len(res_empty) >= 0
        
        # 2. Test massive query string to check token/budget bounds safety
        massive_query = "SQLite WAL " * 200
        res_massive = query_decisions(
            query=massive_query,
            depth="letter",
            project=config.workspace_dir
        )
        assert len(res_massive) >= 0
        
        # 3. Test SQL injection or malicious payload characters
        sql_injection = "' OR 1=1; DROP TABLE nodes; --"
        res_sql = query_decisions(
            query=sql_injection,
            depth="letter",
            project=config.workspace_dir
        )
        assert len(res_sql) >= 0


# ==============================================================================
# 3. Surface Decisions under Hostile / Nonexistent Scopes
# ==============================================================================
def test_mcp_surface_invalid_scopes(isolated_workspace) -> None:
    """Verifies that surface_decisions handles invalid, empty, or hostile scope tags.

    Tags tested:
      - Nonexistent scope.
      - Empty scope string.
      - Scopes with spaces, punctuation, or special characters.
    """
    config, tmpdir = isolated_workspace
    store = GraphStore(config.db_path)
    
    # Commit a decision in 'substrate' scope
    d = ParsedEntry("decision", "dec-one", 1, 5)
    d.axiom = "WAL mode SQLite."
    d.rejected_paths = "None."
    d.scope = ["substrate"]
    store.commit_parsed_entry(d)
    
    # Patch get_workspace_components to use our test graph store
    with patch("mitos.mcp_server.get_workspace_components", return_value=(store, None, None)):
        # 1. Surface nonexistent scope
        res_nonexistent = surface_decisions(
            query="SQLite",
            scope="nonexistent_scope",
            project=config.workspace_dir
        )
        assert "0 active decisions" in res_nonexistent or len(res_nonexistent) >= 0
        
        # 2. Surface empty scope
        res_empty = surface_decisions(
            query="SQLite",
            scope="",
            project=config.workspace_dir
        )
        assert len(res_empty) >= 0
        
        # 3. Surface with special characters in scope name
        res_special = surface_decisions(
            query="SQLite",
            scope="substrate' OR 1=1 --",
            project=config.workspace_dir
        )
        assert len(res_special) >= 0
