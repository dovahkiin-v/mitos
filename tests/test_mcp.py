"""Adversarial test suite for the Mitos MCP server.

Verifies FastMCP tool registration, tool query behaviors, and strict adherence
to the C4 Letter-mode JSON return format.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from mitos.store import GraphStore
from mitos.parser import ParsedEntry
from mitos.mcp_server import mcp, surface_decisions, query_decisions

@pytest.mark.asyncio
async def test_mcp_tool_registration() -> None:
    """Verifies that all required MCP tools are correctly named and registered."""
    tools = await mcp.list_tools()
    tool_names = [tool.name for tool in tools]
    assert "surface_decisions" in tool_names
    assert "query_decisions" in tool_names


@patch("mitos.mcp_server.get_workspace_components")
def test_query_decisions_tool(mock_get_components: MagicMock) -> None:
    """Verifies query_decisions formats data strictly in Letter-mode shape."""
    # Mock GraphStore
    mock_store = MagicMock()
    mock_store.get_node_by_slug.return_value = {
        "id": "hash-abc",
        "slug": "auth-decision",
        "core_axiom": "We use JWTs.",
        "rejected_paths": "Direct session lookups.",
        "mechanisms": ["jwt"],
        "scope": ["auth"],
        "transcript": "Secret conversation text."  # C4: must be excluded!
    }
    mock_store.compute_all_states.return_value = {"hash-abc": "active"}
    
    mock_get_components.return_value = (mock_store, None, None)

    # Call query_decisions
    resp_text = query_decisions(query="auth-decision")
    resp = json.loads(resp_text)
    
    # Assert Letter-mode keys are present and clean
    assert resp["slug"] == "auth-decision"
    assert resp["axiom"] == "We use JWTs."
    assert resp["rejected_paths"] == "Direct session lookups."
    assert resp["scope"] == ["auth"]
    assert resp["state"] == "active"
    
    # Assert transcript is EXCLUDED (strictly Letter-mode only)
    assert "transcript" not in resp
    assert "core_axiom" not in resp


@patch("mitos.mcp_server.get_workspace_components")
def test_surface_decisions_fallback_filtering(mock_get_components: MagicMock) -> None:
    """Verifies surface_decisions fallback pre-filtering and open questions insertion."""
    mock_store = MagicMock()
    
    # Mock active decisions matching scope tag
    mock_store.get_active_decisions.return_value = [
        {
            "id": "hash-1",
            "slug": "active-db",
            "core_axiom": "Use SQLite.",
            "rejected_paths": "Postgres.",
            "mechanisms": ["sqlite"],
            "scope": ["db"]
        }
    ]
    
    # Mock open questions in same scope
    mock_store.get_open_questions.return_value = [
        {
            "id": "hash-oq",
            "slug": "db-scaling",
            "questions_raised": ["How does it shard?"],
            "scope": ["db"],
            "park_reason": "needs benchmarking",
            "computed_state": "parked"
        }
    ]
    mock_store.compute_all_states.return_value = {"hash-oq": "parked", "hash-1": "active"}

    mock_get_components.return_value = (mock_store, None, None)

    # Call surface_decisions
    resp_text = surface_decisions(query="database strategy", scope="db")
    resp = json.loads(resp_text)
    
    assert len(resp["active_decisions"]) == 1
    assert resp["active_decisions"][0]["slug"] == "active-db"
    assert resp["active_decisions"][0]["axiom"] == "Use SQLite."
    
    assert len(resp["open_questions"]) == 1
    assert resp["open_questions"][0]["topic"] == "db-scaling"
    assert resp["open_questions"][0]["questions_raised"] == ["How does it shard?"]
    assert resp["open_questions"][0]["park_reason"] == "needs benchmarking"
