"""Adversarial test suite for the Mitos Sync Pipeline.

Verifies private snapshotting, advisory locking, LLM enrichment mocks, slug collision
correction prompts, and content-aware archive rotation.
"""

import tempfile
import os
import shutil
import json
import pytest
from typing import Tuple
from unittest.mock import MagicMock, patch

from mitos.config import MitosConfig
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager
from mitos.parser import ParsedEntry

@pytest.fixture
def sync_env() -> Tuple[MitosConfig, MitosSyncManager, str]:
    """Fixture to set up a complete mock sync environment."""
    tmpdir = tempfile.mkdtemp()
    
    # Custom config mapping to temp folder
    config = MitosConfig(tmpdir)
    config.db_path = os.path.join(tmpdir, ".mitos", "graph.sqlite")
    config.decisions_file = os.path.join(tmpdir, "decisions.md")
    config.archive_dir = os.path.join(tmpdir, "decisions", "archive")
    
    # Create required .mitos and files
    os.makedirs(os.path.join(tmpdir, ".mitos"), exist_ok=True)
    
    # Write empty decisions.md with BEGIN ENTRIES marker
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write(
            "# Decisions\n"
            "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->\n"
        )
        
    manager = MitosSyncManager(config)
    yield config, manager, tmpdir
    
    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


@patch("google.genai.Client")
def test_sync_happy_path(mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]) -> None:
    """Verifies that new buffer entries are parsed, LLM-enriched, committed, and rotated."""
    config, manager, tmpdir = sync_env

    # 1. Append valid decision entry to write buffer
    entry_text = (
        "## 2026-05-19 — isolation — Isolation Title\n"
        "**Decided:** Use pure logic cores.\n"
        "**Rejected:** Tight coupling.\n"
        "**Mechanisms:** python\n"
        "**Scope:** core\n"
    )
    
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(entry_text + "\n")

    # 2. Mock Gemini API Client responses
    mock_gen_resp = MagicMock()
    mock_gen_resp.text = json.dumps({
        "refined_core_axiom": "We strictly use pure logic cores.",
        "refined_mechanisms": ["python", "sqlite"],
        "refined_scope": ["core", "substrate"],
        "suggested_relationships": {}
    })
    mock_client.return_value.models.generate_content.return_value = mock_gen_resp

    # Set up environment variables to satisfy provider check
    os.environ["GEMINI_API_KEY"] = "mock_key"

    # 3. Perform sync in auto-accept mode
    manager.perform_sync(auto_accept=True)

    # 4. Assertions
    store = GraphStore(config.db_path)
    nodes = store.get_all_nodes()
    assert len(nodes) == 1
    node = nodes[0]
    
    assert node["slug"] == "isolation"
    # Refined axiom saved
    assert node["core_axiom"] == "We strictly use pure logic cores."
    assert node["mechanisms"] == ["python", "sqlite"]
    assert node["scope"] == ["core", "substrate"]
    # Verify OD3 confirmation metadata populated
    assert node["confirmed_by"] == "gemini-3.1-flash-lite"
    assert node["confirmed_at"] is not None

    # 5. Assert content-aware archive rotation:
    # decisions.md write buffer must be cleared of the entry raw block
    with open(config.decisions_file, "r", encoding="utf-8") as f:
        remaining_content = f.read()
    assert "## 2026-05-19 — isolation" not in remaining_content
    assert "BEGIN ENTRIES" in remaining_content  # Marker preserved

    # Archive folder contains the rotated block
    archives = os.listdir(config.archive_dir)
    assert len(archives) == 1
    with open(os.path.join(config.archive_dir, archives[0]), "r", encoding="utf-8") as f:
        archive_content = f.read()
    assert "## 2026-05-19 — isolation" in archive_content


@patch("google.genai.Client")
def test_sync_stale_entry_detection(mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture) -> None:
    """Verifies that entries drafted >14 days ago trigger a stdout warning."""
    config, manager, tmpdir = sync_env

    # 1. Draft an entry dated 20 days ago (relative to June 2026 current time)
    entry_text = (
        "## 2026-05-10 — stale-slug — A stale decision\n"
        "**Decided:** Use stable algorithms.\n"
        "**Rejected:** Transient models.\n"
    )
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(entry_text + "\n")

    # 2. Mock client response
    mock_gen_resp = MagicMock()
    mock_gen_resp.text = json.dumps({
        "refined_core_axiom": "Use stable algorithms.",
        "refined_mechanisms": [],
        "refined_scope": ["core"],
        "suggested_relationships": {}
    })
    mock_client.return_value.models.generate_content.return_value = mock_gen_resp
    os.environ["GEMINI_API_KEY"] = "mock_key"

    manager.perform_sync(auto_accept=True)

    captured = capsys.readouterr()
    assert "was drafted on 2026-05-10 (>14 days ago) and remains unsynced" in captured.out


@patch("google.genai.Client")
@patch("builtins.input", side_effect=["c", "a"])
def test_sync_slug_collision_correction(mock_input: MagicMock, mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]) -> None:
    """Verifies S4 correction flow on slug collision, checking that a corrects edge is created."""
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)
    os.environ["GEMINI_API_KEY"] = "mock_key"

    # 1. Seed an existing active decision in the graph
    entry1 = ParsedEntry("decision", "database", 1, 10)
    entry1.core_axiom = "We use PostgreSQL."
    entry1.rejected_paths = "No SQL."
    entry1.scope = ["database"]
    store.commit_parsed_entry(entry1)

    # 2. Add colliding slug in decisions.md
    entry2_text = (
        "## 2026-06-01 — database — Database Update\n"
        "**Decided:** We actually use SQLite for local WAL reads.\n"
        "**Rejected:** PostgreSQL dependency.\n"
    )
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(entry2_text + "\n")

    # Mock client response
    mock_gen_resp = MagicMock()
    mock_gen_resp.text = json.dumps({
        "refined_core_axiom": "We actually use SQLite for local WAL reads.",
        "refined_mechanisms": ["sqlite"],
        "refined_scope": ["database"],
        "suggested_relationships": {}
    })
    mock_client.return_value.models.generate_content.return_value = mock_gen_resp

    # 3. Run sync (interactive input mock selects "c" for correction and "a" for accept)
    manager.perform_sync(auto_accept=False)

    # 4. Assert corrects relationship was created in database
    edges = store.get_edges()
    assert len(edges) == 1
    edge = edges[0]
    assert edge["type"] == "corrects"

    # Assert computed states have replaced original active decision with corrected one
    nodes = store.get_all_nodes()
    assert len(nodes) == 2
    
    conn = store._get_connection()
    states = store.compute_all_states(conn)
    conn.close()
    
    corrected_id = [n["id"] for n in nodes if "SQLite" in n["core_axiom"]][0]
    original_id = [n["id"] for n in nodes if "PostgreSQL" in n["core_axiom"]][0]
    
    assert states[corrected_id] == "active"
    assert states[original_id] == "superseded"


@patch("google.genai.Client")
def test_sync_outbox_queue_and_drain(mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]) -> None:
    """Verifies that failed embeddings enter pending_embeddings queue and drain on recovery (C2)."""
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)
    os.environ["GEMINI_API_KEY"] = "mock_key"

    # Mock the embedding provider API call entirely to prevent network requests
    manager.embed_provider.get_embedding = MagicMock(return_value=[0.1, 0.2, 0.3])

    # 1. Force a connection failure on vector store upsert
    manager.vector_store.upsert = MagicMock(side_effect=Exception("Qdrant connection refused"))

    # 2. Append new decision entry
    entry_text = (
        "## 2026-05-19 — queue-test — Queue Test Title\n"
        "**Decided:** Outbox queue works.\n"
        "**Rejected:** Memory only queue.\n"
    )
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(entry_text + "\n")

    # Mock client response
    mock_gen_resp = MagicMock()
    mock_gen_resp.text = json.dumps({
        "refined_core_axiom": "Outbox queue works.",
        "refined_mechanisms": [],
        "refined_scope": ["core"],
        "suggested_relationships": {}
    })
    mock_client.return_value.models.generate_content.return_value = mock_gen_resp

    # 3. Perform sync -> should finish and commit graph, but defer embedding to outbox
    manager.perform_sync(auto_accept=True)

    # 4. Assert node exists in graph but also in pending_embeddings queue
    nodes = store.get_all_nodes()
    assert len(nodes) == 1
    
    pending = store.get_pending_embeddings()
    assert len(pending) == 1
    assert pending[0]["node_id"] == nodes[0]["id"]
    assert pending[0]["embedding_text"] == "Outbox queue works."

    # 5. Restore vector store (mock recovery)
    manager.vector_store.upsert = MagicMock() # success
    
    # 6. Run manual drain
    manager.drain_pending_embeddings()

    # 7. Assert outbox is now drained cleanly
    pending_post = store.get_pending_embeddings()
    assert len(pending_post) == 0
    manager.vector_store.upsert.assert_called_once()


def test_sync_outbox_queue_concurrent_drain(sync_env: Tuple[MitosConfig, MitosSyncManager, str]) -> None:
    """Verifies that concurrent drainers atomically claim distinct rows and prevent double-processing."""
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)
    
    # 1. Commit three valid nodes first to satisfy FK constraints
    e1 = ParsedEntry("decision", "db-1", 1, 5)
    e1.core_axiom = "Axiom 1"
    e1.rejected_paths = "None."
    d1 = store.commit_parsed_entry(e1)
    
    e2 = ParsedEntry("decision", "db-2", 1, 5)
    e2.core_axiom = "Axiom 2"
    e2.rejected_paths = "None."
    d2 = store.commit_parsed_entry(e2)
    
    e3 = ParsedEntry("decision", "db-3", 1, 5)
    e3.core_axiom = "Axiom 3"
    e3.rejected_paths = "None."
    d3 = store.commit_parsed_entry(e3)

    # Pre-populate some pending embeddings in the queue
    store.add_pending_embedding(d1.node_id, "Axiom 1")
    store.add_pending_embedding(d2.node_id, "Axiom 2")
    store.add_pending_embedding(d3.node_id, "Axiom 3")
    
    # Drainer 1 claims a batch of 2
    drainer1_items = store.claim_pending_embeddings("drainer-1", limit=2)
    assert len(drainer1_items) == 2
    
    # Drainer 2 tries to claim a batch of 2 -> should only get the remaining 1 item!
    drainer2_items = store.claim_pending_embeddings("drainer-2", limit=2)
    assert len(drainer2_items) == 1
    assert drainer2_items[0]["node_id"] == d3.node_id
    
    # Verify Drainer 1 items are d1 and d2
    drainer1_node_ids = {item["node_id"] for item in drainer1_items}
    assert drainer1_node_ids == {d1.node_id, d2.node_id}
    
    # Drainer 1 releases its claims
    store.release_pending_embeddings("drainer-1")
    
    # Drainer 3 claims now -> should get d1 and d2
    drainer3_items = store.claim_pending_embeddings("drainer-3", limit=2)
    assert len(drainer3_items) == 2
    drainer3_node_ids = {item["node_id"] for item in drainer3_items}
    assert drainer3_node_ids == {d1.node_id, d2.node_id}


def test_sync_auto_heal_sample_block(sync_env: Tuple[MitosConfig, MitosSyncManager, str]) -> None:
    """Verifies that the decisions.md header and sample format block are auto-restored if modified or missing."""
    config, manager, tmpdir = sync_env
    
    # 1. Write an entry in the buffer with a corrupted header
    original_entries = (
        "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->\n\n"
        "## 2026-05-19 — my-test — Real decision\n"
        "**Decided:** Real core decision.\n"
        "**Rejected:** None.\n"
    )
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write("# Corrupted Header\nSome junk text\n\n" + original_entries)
        
    # 2. Trigger auto-healing
    manager.auto_heal_decisions_file()
    
    # 3. Read back healed decisions.md
    with open(config.decisions_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert "## SAMPLE FORMAT — auto-restored by mitos sync" in content
    assert "### example-slug" in content
    assert "my-test" in content
    assert "Real core decision." in content


