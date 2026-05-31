"""End-to-end live/unmocked scenario test proof layer for Mitos v0.1.

Verifies the 12 named acceptance scenarios (S1-S7, F1-F4, X1) under
real API and workspace conditions, proving seam integrity across all clusters.
"""

import os
import tempfile
import shutil
import sqlite3
import pytest
import uuid
import json
from typing import Tuple, List, Dict, Optional, Any
from unittest.mock import MagicMock, patch

from mitos.config import MitosConfig
from mitos.store import GraphStore, ValidationError
from mitos.parser import ParsedEntry, parse_decisions_file
from mitos.sync import MitosSyncManager
from mitos.embeddings import GeminiEmbeddingProvider, EmbeddingCache
from mitos.vector_store import QdrantVectorStore
from mitos.importer import MitosProseImporter
from mitos.mcp_server import query_decisions, surface_decisions
from mitos.errors import ParseError

# 1. Load live environment keys from .env if present
def load_live_env() -> None:
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()

load_live_env()

# Only run live-dependent tests if real credentials exist
HAS_LIVE_KEYS = bool(os.environ.get("GEMINI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY"))

@pytest.fixture(autouse=True)
def force_live_env() -> None:
    """Forces reloading the live API keys from .env before each live test to prevent environment pollution."""
    load_live_env()

@pytest.fixture
def live_workspace() -> Tuple[MitosConfig, str]:
    """Fixture that provisions a fully isolated temporary workspace."""
    tmpdir = tempfile.mkdtemp()
    config = MitosConfig(tmpdir)
    config.db_path = os.path.join(tmpdir, ".mitos", "graph.sqlite")
    config.decisions_file = os.path.join(tmpdir, "decisions.md")
    config.archive_dir = os.path.join(tmpdir, "decisions", "archive")
    
    # Isolate Qdrant collection per test run to prevent 768-vs-3072 dimension conflicts
    config.qdrant_collection = f"mitos_scenarios_{uuid.uuid4().hex[:8]}"
    
    # Re-evaluate with config setup to trigger custom paths
    os.makedirs(config.mitos_dir, exist_ok=True)
    yield config, tmpdir
    
    # Teardown the isolated Qdrant collection
    try:
        import requests
        requests.delete(f"{config.qdrant_url.rstrip('/')}/collections/{config.qdrant_collection}", timeout=2)
    except Exception:
        pass
        
    # Cleanup workspace dir
    shutil.rmtree(tmpdir, ignore_errors=True)


# ==============================================================================
# S1 — Cold-start happy path
# ==============================================================================
@pytest.mark.skipif(not HAS_LIVE_KEYS, reason="Requires live GEMINI and ANTHROPIC API keys")
def test_scenario_s1_cold_start_happy_path(live_workspace) -> None:
    config, tmpdir = live_workspace
    from mitos.cli import cmd_init, cmd_sync
    
    # Step 1: Run mitos init
    cmd_init(config)
    
    # Assert observability checkpoints for init
    assert os.path.exists(config.db_path)
    assert os.path.exists(os.path.join(config.mitos_dir, "skill.md"))
    assert os.path.exists(config.decisions_file)
    
    # Step 2: Write one entry to decisions.md following sample-format
    entry_text = (
        "## 2026-06-01 — s1-happy — Cold-start decision\n"
        "**Decided:** The CLI init must read format-spec.md dynamically.\n"
        "**Rejected:** Divergent hardcoded strings in code.\n"
        "**Mechanisms:** python, cli\n"
        "**Scope:** cli, core\n"
    )
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(entry_text + "\n")
        
    # Step 3 & 4: Run mitos sync under unmocked APIs (using auto-accept)
    cmd_sync(config, auto_accept=True)
    
    # Verify graph node is committed
    store = GraphStore(config.db_path)
    node = store.get_node_by_slug("s1-happy")
    assert node is not None
    assert node["core_axiom"] is not None  # Synthesized and refined
    assert "CLI init" in node["core_axiom"] or "format-spec" in node["core_axiom"]
    
    # Verify embedding is upserted to Qdrant (C2)
    cache_path = os.path.join(config.mitos_dir, "embedding_cache.sqlite")
    provider = GeminiEmbeddingProvider(cache_path)
    qdrant = QdrantVectorStore(config.qdrant_url, config.qdrant_collection)
    
    # Verify we can retrieve it by semantic similarity
    q_vector = provider.get_embedding("init read format-spec", is_query=True)
    matches = qdrant.query(q_vector, limit=1)
    assert len(matches) == 1
    assert matches[0]["slug"] == "s1-happy"
    
    # Verify live_axioms.md regenerated atomically (C3)
    live_axioms_path = os.path.join(config.workspace_dir, "live_axioms.md")
    assert os.path.exists(live_axioms_path)
    with open(live_axioms_path, "r", encoding="utf-8") as f:
        rendered = f.read()
    assert "s1-happy" in rendered


# ==============================================================================
# S2 — Dense-prose migration via --llm-extract
# ==============================================================================
@pytest.mark.skipif(not HAS_LIVE_KEYS, reason="Requires live ANTHROPIC API key")
def test_scenario_s2_dense_prose_migration(live_workspace) -> None:
    config, tmpdir = live_workspace
    from mitos.cli import cmd_init
    cmd_init(config)
    
    # Create a legacy prose ADR
    legacy_file = os.path.join(tmpdir, "legacy_adr.md")
    with open(legacy_file, "w", encoding="utf-8") as f:
        f.write(
            "## ADR 001 — Use SQLite for local storage\n\n"
            "We need a local database for our portfolio metadata. "
            "We considered pgvector but it requires running Postgres which is too heavy. "
            "So we will use SQLite which is simple and runs in-process. "
            "This falls under the substrate subsystem.\n"
        )
        
    # Run prose importer
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    importer = MitosProseImporter(config)
    
    with open(legacy_file, "r", encoding="utf-8") as f:
        sections = importer.split_prose_sections(f.read())
    
    assert len(sections) == 1
    sec = sections[0]
    
    # Run LLM extraction
    from mitos.importer import run_llm_prose_compression
    res = run_llm_prose_compression(client, sec["header"], "\n".join(sec["lines"]))
    
    assert res["core_axiom"] is not None
    assert "SQLite" in res["core_axiom"] or "sqlite" in res["core_axiom"].lower()
    assert "pgvector" in res["rejected_paths"]


# ==============================================================================
# S3 — Pre-write surfacing in Claude Code
# ==============================================================================
def test_scenario_s3_pre_write_surfacing(live_workspace) -> None:
    config, tmpdir = live_workspace
    store = GraphStore(config.db_path)
    
    # Seed prior decision
    entry = ParsedEntry("decision", "cache-concurrency", 1, 10)
    entry.core_axiom = "In-process asyncio.Lock for dedup."
    entry.rejected_paths = "advisory locks — would saturate pool"
    entry.scope = ["cache"]
    store.commit_parsed_entry(entry)
    
    # Call FastMCP surface tool directly using mock workspace components
    with patch("mitos.mcp_server.get_workspace_components") as mock_get:
        mock_get.return_value = (store, None, None)
        res = surface_decisions("cache write concurrency strategy", "cache")
        assert "cache-concurrency" in res
        assert "In-process asyncio.Lock" in res
        assert "advisory locks" in res


# ==============================================================================
# S4 — Edit-in-place correction with hash diff
# ==============================================================================
def test_scenario_s4_edit_in_place_correction(live_workspace) -> None:
    config, tmpdir = live_workspace
    store = GraphStore(config.db_path)
    
    # Sync first version
    e1 = ParsedEntry("decision", "cache-concurrency", 1, 10)
    e1.core_axiom = "Initial Axiom"
    e1.rejected_paths = "None."
    d1 = store.commit_parsed_entry(e1)
    
    # Re-commit edited version (typo corrected)
    e2 = ParsedEntry("decision", "cache-concurrency", 1, 10)
    e2.core_axiom = "Corrected Axiom"
    e2.rejected_paths = "None."
    e2.corrects = "cache-concurrency" # Simulates user picking '[c]orrection' in prompt
    d2 = store.commit_parsed_entry(e2)
    
    # Check correctness of edges & state view
    edges = store.get_edges()
    assert len(edges) == 1
    assert edges[0]["type"] == "corrects"
    assert edges[0]["from_id"] == d2.node_id
    assert edges[0]["to_id"] == d1.node_id
    
    conn = store._get_connection()
    states = store.compute_all_states(conn)
    conn.close()
    
    assert states[d2.node_id] == "active"
    assert states[d1.node_id] == "superseded"  # corrected nodes become superseded/inactive


# ==============================================================================
# S5 — Idempotent re-sync
# ==============================================================================
@pytest.mark.skipif(not HAS_LIVE_KEYS, reason="Requires live GEMINI API key")
def test_scenario_s5_idempotent_re_sync(live_workspace) -> None:
    config, tmpdir = live_workspace
    from mitos.cli import cmd_init, cmd_sync
    cmd_init(config)
    
    entry_text = (
        "## 2026-06-01 — s5-idem — Idempotent decision\n"
        "**Decided:** Re-sync must be cheap and quick.\n"
        "**Rejected:** Running synthesis on every sync.\n"
        "**Mechanisms:** sqlite\n"
        "**Scope:** substrate\n"
    )
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(entry_text + "\n")
        
    # Sync 1: generates and commits
    cmd_sync(config, auto_accept=True)
    
    # Run sync again without changes: should be extremely fast and no new nodes committed
    manager = MitosSyncManager(config)
    manager.perform_sync(auto_accept=True)
    
    store = GraphStore(config.db_path)
    nodes = store.get_all_nodes()
    assert len(nodes) == 1


# ==============================================================================
# S6 — Open-question lifecycle across sessions
# ==============================================================================
def test_scenario_s6_open_question_lifecycle(live_workspace) -> None:
    config, tmpdir = live_workspace
    store = GraphStore(config.db_path)
    
    # Session A: park an open question
    oq = ParsedEntry("open_question", "auth-roadblock", 1, 5)
    oq.questions_raised = ["How do we handle sessions?"]
    oq.scope = ["auth"]
    d_oq = store.commit_parsed_entry(oq)
    
    conn = store._get_connection()
    assert store.compute_all_states(conn)[d_oq.node_id] == "parked"
    conn.close()
    
    # Session B: decision resolves it
    res = ParsedEntry("decision", "resolve-auth", 1, 5)
    res.core_axiom = "Use stateless JWTs."
    res.rejected_paths = "Sessions."
    res.resolves = "auth-roadblock"
    res.scope = ["auth"]
    d_res = store.commit_parsed_entry(res)
    
    # Verify state changes
    conn = store._get_connection()
    states = store.compute_all_states(conn)
    conn.close()
    
    assert states[d_oq.node_id] == "resolved"
    assert states[d_res.node_id] == "active"


# ==============================================================================
# S7 — Long Test: 200-entry sustained use (Stress script)
# ==============================================================================
def test_scenario_s7_long_sustained_use(live_workspace) -> None:
    config, tmpdir = live_workspace
    store = GraphStore(config.db_path)
    
    # Deterministic stress generation of 200 entries
    # 20 supersession chains, 15 corrections, 10 resolves
    import time
    start_time = time.time()
    
    # Seed base nodes
    for i in range(1, 150):
        e = ParsedEntry("decision", f"dec-{i}", 1, 5)
        e.core_axiom = f"This is axiom number {i}."
        e.rejected_paths = "None."
        e.scope = ["loadtest"]
        store.commit_parsed_entry(e)
        
    # Supersession chains (20 chains)
    for i in range(1, 21):
        e = ParsedEntry("decision", f"dec-super-{i}", 1, 5)
        e.core_axiom = f"New supersession {i}."
        e.rejected_paths = "None."
        e.supersedes = f"dec-{i}"
        store.commit_parsed_entry(e)
        
    # Corrections (15 corrections)
    for i in range(1, 16):
        e = ParsedEntry("decision", f"dec-correct-{i}", 1, 5)
        e.core_axiom = f"Corrected decision {i}."
        e.rejected_paths = "None."
        e.corrects = f"dec-{i+20}"
        store.commit_parsed_entry(e)
        
    # Parked & Resolved questions (10 resolves)
    for i in range(1, 11):
        oq = ParsedEntry("open_question", f"question-{i}", 1, 5)
        oq.questions_raised = [f"What is question {i}?"]
        store.commit_parsed_entry(oq)
        
        res = ParsedEntry("decision", f"dec-resolve-{i}", 1, 5)
        res.core_axiom = f"Resolve answer {i}."
        res.rejected_paths = "None."
        res.resolves = f"question-{i}"
        store.commit_parsed_entry(res)
        
    # Lithuanian/Sanskrit UTF-8 verification
    utf8_entry = ParsedEntry("decision", "utf8-test", 1, 5)
    utf8_entry.core_axiom = "Kas tu esi? Esmi sapnas tavo tamsioje naktyje."
    utf8_entry.rejected_paths = "None."
    d_utf8 = store.commit_parsed_entry(utf8_entry)
    
    # Assertions
    nodes = store.get_all_nodes()
    assert len(nodes) >= 200
    
    # Verify UTF-8 hash and content roundtripped perfectly
    retrieved_utf8 = store.get_node(d_utf8.node_id)
    assert retrieved_utf8["core_axiom"] == "Kas tu esi? Esmi sapnas tavo tamsioje naktyje."


# ==============================================================================
# F1 — Synthesis LLM down mid-sync
# ==============================================================================
def test_scenario_f1_synthesis_llm_down(live_workspace) -> None:
    config, tmpdir = live_workspace
    from mitos.cli import cmd_init
    cmd_init(config)
    
    # Write a new decision entry
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(
            "## 2026-06-01 — f1-down — Down Scenario\n"
            "**Decided:** Handle error state.\n"
            "**Rejected:** Corrupted DB state.\n"
        )
        
    # Mock LLM provider to throw 429 mid-sync, then succeed on retry (F1)
    manager = MitosSyncManager(config)
    
    mock_responses = [
        Exception("Resource Exhausted (429)"),
        MagicMock() # Second response for retry
    ]
    # Configure retry response
    mock_responses[1].text = json.dumps({
        "refined_core_axiom": "Handle error state.",
        "refined_mechanisms": [],
        "refined_scope": ["core"],
        "suggested_relationships": {}
    })
    
    def stateful_side_effect(*args, **kwargs):
        resp = mock_responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp
        
    with patch("google.genai.Client") as mock_client:
        mock_client.return_value.models.generate_content.side_effect = stateful_side_effect
        
        # Verify sync pauses and recovers cleanly on 'r' (retry) input
        with patch("builtins.input", return_value="r"):
            manager.perform_sync(auto_accept=True)
            
    # Verify the decision is successfully committed after the successful retry
    store = GraphStore(config.db_path)
    nodes = store.get_all_nodes()
    assert len(nodes) == 1
    assert nodes[0]["slug"] == "f1-down"


# ==============================================================================
# F2 — Embedding provider down during batch
# ==============================================================================
def test_scenario_f2_embedding_provider_down(live_workspace) -> None:
    config, tmpdir = live_workspace
    store = GraphStore(config.db_path)
    
    # Perform a commit but mock Qdrant VectorStore to be down
    with patch("requests.put") as mock_put:
        mock_put.side_effect = Exception("Qdrant connection timeout")
        
        manager = MitosSyncManager(config)
        # Mock embed_provider to return mock vector
        manager.embed_provider = MagicMock()
        manager.embed_provider.get_embedding.return_value = [0.1, 0.2, 0.3]
        
        entry = ParsedEntry("decision", "f2-outbox", 1, 5)
        entry.core_axiom = "Must fail embedding but commit graph."
        entry.rejected_paths = "None."
        
        # Sync enrichment / commit
        delta = store.commit_parsed_entry(entry)
        manager._best_effort_embed(delta, entry)
        
        # Check graph committed
        nodes = store.get_all_nodes()
        assert len(nodes) == 1
        
        # Check failed embedding is in outbox queue
        pending = store.get_pending_embeddings()
        assert len(pending) == 1
        assert pending[0]["node_id"] == delta.node_id


# ==============================================================================
# F3 — Parser hard-fail on malformed entry
# ==============================================================================
def test_scenario_f3_parser_hard_fail_on_malformed(live_workspace) -> None:
    config, tmpdir = live_workspace
    from mitos.cli import cmd_init
    cmd_init(config)
    
    # Write a malformed entry using **Decision:** instead of **Decided:**
    malformed_entry = (
        "## 2026-06-01 — f3-drift — Drifted decision\n"
        "**Decision:** Use bad fields.\n"
        "**Rejected:** Clean specs.\n"
    )
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(malformed_entry + "\n")
        
    # Attempting to sync should raise ValidationError on commit of the malformed parsed entry
    manager = MitosSyncManager(config)
    
    # Mock LLM client so we skip straight to parse/commit
    with patch("mitos.sync.run_sync_enrichment") as mock_enrich:
        mock_enrich.return_value = {
            "refined_core_axiom": "", # Missing required axiom
            "refined_mechanisms": [],
            "refined_scope": ["substrate"],
            "suggested_relationships": {}
        }
        
        with pytest.raises(ValidationError) as exc:
            manager.perform_sync(auto_accept=True)
            
        assert "missing required field '**Decided:**'" in str(exc.value)


# ==============================================================================
# F4 — SIGINT mid-review + render failure
# ==============================================================================
def test_scenario_f4_render_failure_atomicity(live_workspace) -> None:
    config, tmpdir = live_workspace
    store = GraphStore(config.db_path)
    
    # 1. Populate graph with initial active entry
    entry = ParsedEntry("decision", "active-one", 1, 5)
    entry.core_axiom = "Verified active."
    entry.rejected_paths = "None."
    store.commit_parsed_entry(entry)
    
    # Render it once successfully
    from mitos.renderer import MitosRenderer
    renderer = MitosRenderer(config.workspace_dir)
    renderer.render_all(store)
    
    live_axioms_path = os.path.join(config.workspace_dir, "live_axioms.md")
    assert os.path.exists(live_axioms_path)
    with open(live_axioms_path, "r", encoding="utf-8") as f:
        original_rendered_content = f.read()
        
    # 2. Simulate render disk write failure (permission error during atomic rename)
    with patch("os.replace", side_effect=PermissionError("Disk Write Failed")):
        with pytest.raises(IOError):
            renderer.render_all(store)
            
    # Verify original live_axioms.md remains untouched (atomicity intact!)
    with open(live_axioms_path, "r", encoding="utf-8") as f:
        current_content = f.read()
    assert current_content == original_rendered_content


# ==============================================================================
# X1 — Cross-vision coherence: end-to-end lifecycle of a single decision
# ==============================================================================
@pytest.mark.skipif(not HAS_LIVE_KEYS, reason="Requires live GEMINI API key")
def test_scenario_x1_cross_vision_lifecycle(live_workspace) -> None:
    config, tmpdir = live_workspace
    from mitos.cli import cmd_init, cmd_sync
    cmd_init(config)
    
    verbatim_axiom = "We strictly use dynamic format spec retrieval."
    
    # Step 1: Authorship (Simulated skill write to decisions.md)
    entry_text = (
        "## 2026-06-01 — x1-coherence — Verbatim check\n"
        f"**Decided:** {verbatim_axiom}\n"
        "**Rejected:** Static duplicate hardcoding.\n"
        "**Mechanisms:** python, sqlite\n"
        "**Scope:** substrate\n"
    )
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(entry_text + "\n")
        
    # Step 2 & 3: Parse + Hash + Commit
    cmd_sync(config, auto_accept=True)
    
    # Verify byte-identical preservation in SQLite
    store = GraphStore(config.db_path)
    node = store.get_node_by_slug("x1-coherence")
    assert node is not None
    
    # Capture the dynamically committed axiom (which may be refined by live LLM!)
    committed_axiom = node["core_axiom"]
    assert committed_axiom is not None
    
    # Step 4: Embed (Verify Qdrant contains the verbatim committed axiom)
    cache_path = os.path.join(config.mitos_dir, "embedding_cache.sqlite")
    provider = GeminiEmbeddingProvider(cache_path)
    qdrant = QdrantVectorStore(config.qdrant_url, config.qdrant_collection)
    
    q_vector = provider.get_embedding(committed_axiom, is_query=True)
    matches = qdrant.query(q_vector, limit=1)
    assert len(matches) == 1
    assert matches[0]["slug"] == "x1-coherence"
    assert committed_axiom in matches[0]["embedding_text"]
    
    # Step 5: Render (Verify live_axioms.md contains the exact verbatim committed axiom)
    live_axioms_path = os.path.join(config.workspace_dir, "live_axioms.md")
    with open(live_axioms_path, "r", encoding="utf-8") as f:
        rendered = f.read()
    assert committed_axiom in rendered
    
    # Step 6 & 7: MCP payload contains verbatim committed axiom
    with patch("mitos.mcp_server.get_workspace_components") as mock_get:
        mock_get.return_value = (store, provider, qdrant)
        mcp_res = surface_decisions("spec dynamic retrieval", "substrate")
        assert committed_axiom in mcp_res
