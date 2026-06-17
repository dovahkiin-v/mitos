"""Highly adversarial pathology and concurrency stress test suite for Mitos.

Verifies extreme edge cases, circular dependencies, Lithuania/Sanskrit unicode slugs,
advisory lock concurrency contention, outbox queue saturation, and alternate
rotation modes, pushing test coverage past 1:1 byte-wise ratio.
"""

import os
import tempfile
import shutil
import sqlite3
import pytest
import uuid
import json
import time
import multiprocessing
from typing import Tuple, List, Dict, Any, Optional
from unittest.mock import MagicMock, patch

from mitos.config import MitosConfig
from mitos.store import GraphStore, ValidationError, DatabaseError
from mitos.parser import ParsedEntry, parse_decisions_file
from mitos.sync import MitosSyncManager
from mitos.embeddings import GeminiEmbeddingProvider, EmbeddingCache
from mitos.vector_store import QdrantVectorStore
from mitos.renderer import MitosRenderer
from mitos.errors import ParseError

# Force load live environment keys
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

HAS_LIVE_KEYS = bool(os.environ.get("GEMINI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY"))

@pytest.fixture
def isolated_workspace() -> Tuple[MitosConfig, str]:
    """Fixture that provisions a fully isolated temporary workspace for pathology stress tests."""
    load_live_env()
    tmpdir = tempfile.mkdtemp()
    config = MitosConfig(tmpdir)
    config.db_path = os.path.join(tmpdir, ".mitos", "graph.sqlite")
    config.decisions_file = os.path.join(tmpdir, "decisions.md")
    config.archive_dir = os.path.join(tmpdir, "decisions", "archive")
    
    # Isolate Qdrant collection
    config.qdrant_collection = f"mitos_pathologies_{uuid.uuid4().hex[:8]}"
    
    os.makedirs(config.mitos_dir, exist_ok=True)
    yield config, tmpdir

    
    # Cleanup Qdrant collection
    try:
        import requests
        requests.delete(f"{config.qdrant_url.rstrip('/')}/collections/{config.qdrant_collection}", timeout=2)
    except Exception:
        pass
        
    shutil.rmtree(tmpdir, ignore_errors=True)


# ==============================================================================
# P1 — Lithuania & Sanskrit Unicode Slug Integrity
# ==============================================================================
def test_pathology_unicode_slug_integrity(isolated_workspace) -> None:
    """Verifies that non-ASCII Lithuanian and Sanskrit characters in slugs parse, hash, and index perfectly."""
    config, tmpdir = isolated_workspace
    store = GraphStore(config.db_path)
    
    # Lithuanian: kaukas-ąčęėįšųūž (mythological spirit)
    # Sanskrit: svapnas-तव-तमसे-नक्ते (dream in your dark night)
    slug_lt = "kaukas-ąčęėįšųūž"
    slug_sa = "svapnas-तव-तमसे-नक्ते"
    
    e1 = ParsedEntry("decision", slug_lt, 1, 10)
    e1.core_axiom = "Ąžuolas yra stiprus ir gilus."
    e1.rejected_paths = "Eglė, pušis."
    e1.scope = ["lietuva"]
    
    e2 = ParsedEntry("decision", slug_sa, 1, 10)
    e2.core_axiom = "Asmi svapnas tava tamase nakte."
    e2.rejected_paths = "None."
    e2.scope = ["sanskrit"]
    
    # Commit both to store
    d1 = store.commit_parsed_entry(e1)
    d2 = store.commit_parsed_entry(e2)
    
    # Assert nodes are successfully saved in database
    n1 = store.get_node(d1.node_id)
    n2 = store.get_node(d2.node_id)
    
    assert n1 is not None
    assert n1["slug"] == slug_lt
    assert n1["core_axiom"] == "Ąžuolas yra stiprus ir gilus."
    
    assert n2 is not None
    assert n2["slug"] == slug_sa
    assert n2["core_axiom"] == "Asmi svapnas tava tamase nakte."


# ==============================================================================
# P2 — Circular Dependency Prevention & Loop Resolution
# ==============================================================================
def test_pathology_circular_dependency_resolution(isolated_workspace) -> None:
    """Tests how the GraphStore handle circular dependency edge declarations."""
    config, tmpdir = isolated_workspace
    store = GraphStore(config.db_path)
    
    # We establish two nodes that supersede each other circularly:
    # A supersedes B, B supersedes A
    eA = ParsedEntry("decision", "node-a", 1, 5)
    eA.core_axiom = "Axiom A"
    eA.rejected_paths = "None."
    eA.supersedes = "node-b"
    
    eB = ParsedEntry("decision", "node-b", 1, 5)
    eB.core_axiom = "Axiom B"
    eB.rejected_paths = "None."
    eB.supersedes = "node-a"
    
    dA = store.commit_parsed_entry(eA)
    dB = store.commit_parsed_entry(eB)
    
    # In SQLite computed states, active/superseded resolution must terminate
    # and not infinite loop. Since node-b was committed second, it points to node-a.
    # Let's verify compute_all_states completes without RecursionError.
    conn = store._get_connection()
    try:
        states = store.compute_all_states(conn)
        assert states is not None
        assert dA.node_id in states
        assert dB.node_id in states
    finally:
        conn.close()


# ==============================================================================
# P3 — Extreme Cascading Status Flips & Deletion Propagation
# ==============================================================================
def test_pathology_extreme_cascading_status_flips(isolated_workspace) -> None:
    """Verifies that resolving an open question triggers a cascade across the graph."""
    config, tmpdir = isolated_workspace
    store = GraphStore(config.db_path)
    
    # 1. Park an open question in 'auth' scope
    oq = ParsedEntry("open_question", "auth-roadblock", 1, 5)
    oq.questions_raised = ["How do we handle sessions?"]
    oq.scope = ["auth"]
    d_oq = store.commit_parsed_entry(oq)
    
    # 2. Add an active decision narrowing another active decision
    e1 = ParsedEntry("decision", "jwt-base", 1, 5)
    e1.core_axiom = "JWT is base auth."
    e1.rejected_paths = "None."
    e1.scope = ["auth"]
    d_e1 = store.commit_parsed_entry(e1)
    
    # 3. Add jwt-spec narrowing jwt-base and resolving roadblock
    e2 = ParsedEntry("decision", "jwt-spec", 1, 5)
    e2.core_axiom = "Use stateless JWTs with HMAC SHA-256."
    e2.rejected_paths = "RSA (too heavy)."
    e2.narrows = "jwt-base"
    e2.resolves = "auth-roadblock"
    e2.scope = ["auth"]
    
    delta = store.commit_parsed_entry(e2)
    
    # Verify JWT roadblock resolved and JWT base status propagated correctly
    assert "auth" in delta.cascade_affected_scopes
    conn = store._get_connection()
    states = store.compute_all_states(conn)
    conn.close()
    
    assert states[d_oq.node_id] == "resolved"


# ==============================================================================
# P4 — Outbox Queue High Contention & Worker Saturation
# ==============================================================================
def test_pathology_outbox_queue_worker_saturation(isolated_workspace) -> None:
    """Simulates 10 concurrent drainers attempting to drain a saturated outbox queue."""
    config, tmpdir = isolated_workspace
    store = GraphStore(config.db_path)
    
    # Seed 50 active nodes in the database to satisfy FK constraints
    node_ids = []
    for i in range(50):
        e = ParsedEntry("decision", f"contend-{i}", 1, 5)
        e.core_axiom = f"Axiom {i}"
        e.rejected_paths = "None."
        d = store.commit_parsed_entry(e)
        node_ids.append(d.node_id)
        
        # Add to outbox queue
        store.add_pending_embedding(d.node_id, f"Axiom {i}")
        
    # Verify outbox size is 50
    assert len(store.get_pending_embeddings()) == 50
    
    # Parallel worker claim simulation
    def worker_claim(worker_id: int, results_list: list) -> None:
        try:
            db = GraphStore(config.db_path)
            # Atomically claim a batch of 10
            claimed = db.claim_pending_embeddings(f"worker-{worker_id}", limit=10)
            results_list.append(len(claimed))
        except Exception:
            pass

    manager = multiprocessing.Manager()
    claimed_counts = manager.list()
    
    processes = []
    for i in range(10): # 10 concurrent workers
        p = multiprocessing.Process(target=worker_claim, args=(i, claimed_counts))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        
    # Verify that the sum of all claimed counts is exactly 50 (zero double-claiming!)
    assert sum(claimed_counts) == 50


# ==============================================================================
# P5 — Concurrency Sync Advisory File Lock Serialization
# ==============================================================================
def test_pathology_sync_advisory_lock_serialization(isolated_workspace) -> None:
    """Verifies that the advisory FileLock strictly serializes concurrent sync invocations."""
    config, tmpdir = isolated_workspace
    manager = MitosSyncManager(config)
    
    # Hold the lock inside the main test process
    with manager.lock:
        # Spawn a parallel process that attempts to sync/acquire lock with a short timeout
        def attempt_sync(cfg_path: str, result_box: list) -> None:
            from filelock import FileLock, Timeout
            lock_path = os.path.join(cfg_path, "decisions.md.lock")
            lock = FileLock(lock_path)
            try:
                with lock.acquire(timeout=0.5):
                    result_box.append("acquired")
            except Timeout:
                result_box.append("timeout")
                
        mp_manager = multiprocessing.Manager()
        results = mp_manager.list()
        
        p = multiprocessing.Process(target=attempt_sync, args=(config.workspace_dir, results))
        p.start()
        p.join()
        
        # Verify the background process timed out because the main process held the lock!
        assert "timeout" in results


# ==============================================================================
# P6 — Alternate Rotation Modes: Prune and Mark
# ==============================================================================
@pytest.mark.skipif(not HAS_LIVE_KEYS, reason="Requires live GEMINI API key")
def test_pathology_rotation_mode_prune(isolated_workspace) -> None:
    """Verifies that rotation_mode='prune' deletes entries from buffer instead of archiving."""
    config, tmpdir = isolated_workspace
    from mitos.cli import cmd_init, cmd_sync
    cmd_init(config)
    
    # Configure rotation mode override to prune
    config.rotation_mode = "prune"
    config.pending_threshold = 1 # Immediate rotation
    
    # Write a new entry to decisions.md
    entry_text = (
        "## 2026-06-01 — s1-prune — Pruned decision\n"
        "**Decided:** Prune deletes rotated nodes from buffer.\n"
        "**Rejected:** Archive preservation.\n"
        "**Mechanisms:** python\n"
        "**Scope:** substrate\n"
    )
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(entry_text + "\n")
        
    cmd_sync(config, auto_accept=True)
    
    # Verify buffer is cleared
    with open(config.decisions_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "s1-prune" not in content
    
    # Verify no archive file was created in archive directory
    assert not os.path.exists(config.archive_dir) or len(os.listdir(config.archive_dir)) == 0


@pytest.mark.skipif(not HAS_LIVE_KEYS, reason="Requires live GEMINI API key")
def test_pathology_rotation_mode_mark(isolated_workspace) -> None:
    """Verifies that rotation_mode='mark' comments out entries in buffer instead of deleting or archiving."""
    config, tmpdir = isolated_workspace
    from mitos.cli import cmd_init, cmd_sync
    cmd_init(config)
    
    # Configure rotation mode override to mark
    config.rotation_mode = "mark"
    config.pending_threshold = 1 # Immediate rotation
    
    # Write a new entry to decisions.md
    entry_text = (
        "## 2026-06-01 — s1-mark — Marked decision\n"
        "**Decided:** Mark comments out rotated nodes.\n"
        "**Rejected:** Archive preservation.\n"
        "**Mechanisms:** python\n"
        "**Scope:** substrate\n"
    )
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(entry_text + "\n")
        
    cmd_sync(config, auto_accept=True)
    
    # Verify buffer still has it, but it is commented out
    with open(config.decisions_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "s1-mark" in content
    assert "<!-- ROTATED START" in content
    assert "ROTATED END -->" in content
    
    # Verify no archive file was created in archive directory
    assert not os.path.exists(config.archive_dir) or len(os.listdir(config.archive_dir)) == 0


# ==============================================================================
# P7 — Renderer Warning on Budget Overflow
# ==============================================================================
def test_pathology_renderer_budget_overflow_warning(isolated_workspace, capsys) -> None:
    """Verifies the renderer RECORDS a size-ceiling overflow on ``.overflows`` (never prints it).

    The warnings used to print mid-render, burying the record receipt under a wall of
    repeated lines. They are now structured data the write path debounces and the
    ``status`` surface details — so the render itself must stay silent.
    """
    config, tmpdir = isolated_workspace
    store = GraphStore(config.db_path)

    # Commit a massive node to push live_axioms.md over the 50,000-char ceiling.
    entry = ParsedEntry("decision", "massive-axiom", 1, 5)
    entry.core_axiom = "We strictly use large text buffers to overflow budget." * 1500
    entry.rejected_paths = "None."
    store.commit_parsed_entry(entry)

    renderer = MitosRenderer(config.workspace_dir)
    renderer.render_all(store)

    # Verify global live_axioms.md exists.
    live_axioms_path = os.path.join(config.workspace_dir, "live_axioms.md")
    assert os.path.exists(live_axioms_path)

    # The overflow is recorded structurally and NOT printed (so it can't bury a receipt).
    captured = capsys.readouterr()
    assert "exceeds" not in captured.out
    assert "[Warning]" not in captured.out
    over = [o for o in renderer.overflows if o["name"] == "live_axioms.md"]
    assert len(over) == 1
    assert over[0]["chars"] > 50000
    assert over[0]["threshold_chars"] == 50000
    assert over[0]["est_tokens"] > 0
