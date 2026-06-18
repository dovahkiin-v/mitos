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


@pytest.mark.skip(reason="V1a defers date-based stale detection (8a): parse_entry_stream "
                         "uses slug-only headers (V1-D7) and does not extract entry.date, so "
                         "the >14-day stale warning has no input. The capability rides dated "
                         "headers, a prototype format V1a's spec dropped — deferred, not silently "
                         "coerced (K5/OD1).")
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
    entry1.axiom = "We use PostgreSQL."
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

    # 4. Assert corrects relationship was created in database (V1a edge column edge_type)
    edges = store.get_edges()
    assert len(edges) == 1
    edge = edges[0]
    assert edge["edge_type"] == "corrects"

    # Assert computed states: the corrector is active; the original is CORRECTED.
    # V1a distinguishes 'corrected' from 'superseded' (the prototype collapsed both) —
    # this is the G12 vocabulary drift the store comment anchors to 8a.
    nodes = store.get_all_nodes()
    assert len(nodes) == 2

    corrected_id = [n["id"] for n in nodes if "SQLite" in n["core_axiom"]][0]
    original_id = [n["id"] for n in nodes if "PostgreSQL" in n["core_axiom"]][0]

    assert store.get_node_state(corrected_id) == "active"
    assert store.get_node_state(original_id) == "corrected"


@patch("google.genai.Client")
def test_sync_outbox_queue_and_drain(mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]) -> None:
    """Verifies that failed embeddings enter pending_embeddings queue and drain on recovery (C2)."""
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)
    os.environ["GEMINI_API_KEY"] = "mock_key"

    # Inject mock embedding deps so the test is hermetic: the manager is built by the
    # fixture before any mocking, so without a reachable Qdrant/GEMINI key (e.g. in CI)
    # it lands in degraded graph-only mode with embed_provider/vector_store == None
    # (sync.py __init__). Assign mocks directly to prevent network requests and exercise
    # the outbox path deterministically regardless of the host environment.
    manager.embed_provider = MagicMock()
    manager.embed_provider.get_embedding = MagicMock(return_value=[0.1, 0.2, 0.3])

    # 1. Force a connection failure on vector store upsert
    manager.vector_store = MagicMock()
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
    # V1a stores NO embedding_text on the row — it is re-derived at drain (C2/M8, 8a).
    assert "embedding_text" not in pending[0]

    # 5. Restore vector store (mock recovery)
    manager.vector_store.upsert = MagicMock() # success
    
    # 6. Run manual drain
    manager.drain_pending_embeddings()

    # 7. Assert outbox is now drained cleanly
    pending_post = store.get_pending_embeddings()
    assert len(pending_post) == 0
    manager.vector_store.upsert.assert_called_once()


def test_sync_outbox_drain_single_writer_semantics(sync_env: Tuple[MitosConfig, MitosSyncManager, str]) -> None:
    """V1a single-writer drain surface: claim is an ordered read, release is a no-op (8a).

    The prototype tested concurrent ``claimed_by`` reservation (two drainers claim
    disjoint rows). V1a defers that claim machinery to V3b (§5.2.8, K3) — it
    serializes writers via ``busy_timeout``, so there is no in-DB claim to contend
    over. The 3-column ``pending_embeddings`` shape carries no ``claimed_by``, so
    ``claim`` is an ordered bounded SELECT (no reservation) and ``release`` is inert.
    This pins the V1a contract; the concurrent-reservation case is V3b's.
    """
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)

    # Commit three valid nodes (the commit also enqueues them — 5c _enqueue_outbox).
    deltas = []
    for slug, ax in (("db-1", "Axiom 1"), ("db-2", "Axiom 2"), ("db-3", "Axiom 3")):
        e = ParsedEntry("decision", slug, 1, 5)
        e.axiom = ax
        e.rejected_paths = "None."
        d = store.commit_parsed_entry(e)
        # add_pending_embedding is the idempotent standalone twin of the commit-time
        # enqueue (V1a 3-column shape, no embedding_text arg) — re-stamps the row.
        store.add_pending_embedding(d.node_id)
        deltas.append(d)
    all_ids = {d.node_id for d in deltas}

    # claim is an ordered read bounded by limit — NO reservation (V1a single-writer).
    batch = store.claim_pending_embeddings("drainer-1", limit=2)
    assert len(batch) == 2

    # release is inert (no claimed_by column to clear) and nothing was consumed.
    store.release_pending_embeddings("drainer-1")
    assert len(store.get_pending_embeddings()) == 3

    # The full pending set is reachable; a row carries no embedding_text (re-derived).
    everyone = store.claim_pending_embeddings("drainer-2", limit=10)
    assert {item["node_id"] for item in everyone} == all_ids
    assert "embedding_text" not in everyone[0]


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


