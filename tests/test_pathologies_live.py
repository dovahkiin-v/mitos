"""Highly adversarial pathology and concurrency stress test suite for Mitos.

Verifies extreme edge cases, circular dependencies, Lithuania/Sanskrit unicode slugs,
advisory lock concurrency contention, outbox queue saturation, and alternate
rotation modes, pushing test coverage past 1:1 byte-wise ratio.
"""

import os
import tempfile
import shutil
import pytest
import uuid
import multiprocessing
from typing import Tuple

from mitos.config import MitosConfig
from mitos.store import GraphStore
from mitos.parser import ParsedEntry
from mitos.sync import MitosSyncManager
from mitos.renderer import MitosRenderer

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
    e1.axiom = "Ąžuolas yra stiprus ir gilus."
    e1.rejected_paths = "Eglė, pušis."
    e1.scope = ["lietuva"]
    
    e2 = ParsedEntry("decision", slug_sa, 1, 10)
    e2.axiom = "Asmi svapnas tava tamase nakte."
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
#
# RETIRED (V1b r1): the pre-V1b scalar-`supersedes` cycle this probed cannot form, and
# its body called the phantom `compute_all_states` (retired in Phase 8a). V1b's write-time
# mutation-cycle prevention is covered purpose-built by:
#   - tests/test_lineage_and_cycles.py  (T10: test_direct_two_cycle_rejected,
#     test_mixed_cross_type_cycle_rejected, test_self_loop_rejected_as_cycle,
#     test_convergent_diamond_accepted, the ≥40-link depth + corrupt-cycle homeostasis gates)
#   - tests/test_store.py 5b  (test_cycle_violation_self_edge, test_cycle_violation_inactive_source)
# ==============================================================================


# ==============================================================================
# P3 — Extreme Cascading Status Flips & Deletion Propagation
# ==============================================================================
def test_pathology_extreme_cascading_status_flips(isolated_workspace) -> None:
    """Resolving an open question flips its COMPUTED state but triggers NO cascade.

    Rewritten for V1b reality: there is no transitive cascade (``CommitDelta`` is
    first-order, DoD #3) and OQ Stage-2 state is computed at read time (M3), so the
    resolving commit writes nothing to the OQ node — no ``updated_at`` tick, no
    Outbox re-enqueue. Only the committing decision gets those. (Was authored
    against the phantom ``compute_all_states`` + a transitive ``cascade_affected_scopes``
    assertion that V1b does not ship — T3 OQ side.)
    """
    config, tmpdir = isolated_workspace
    store = GraphStore(config.db_path)

    def oq_meta(node_id: str):
        conn = store._get_connection()
        try:
            updated_at = conn.execute(
                "SELECT updated_at FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()[0]
            row = conn.execute(
                "SELECT queued_at FROM pending_embeddings WHERE node_id = ?", (node_id,)
            ).fetchone()
            return updated_at, (row[0] if row else None)
        finally:
            conn.close()

    def oq_state(slug: str) -> str:
        for oq in store.get_open_questions():
            if oq["slug"] == slug:
                return oq["state"]
        raise ValueError(f"OQ {slug} not in the active OQ view")

    # 1. Park an open question in 'auth' scope
    oq = ParsedEntry("open_question", "auth-roadblock", 1, 5)
    oq.topic = "Auth session strategy"
    oq.questions_raised = ["How do we handle sessions?"]
    oq.scope = ["auth"]
    d_oq = store.commit_parsed_entry(oq)
    assert oq_state("auth-roadblock") == "parked"

    # 2. Add an active decision (the future narrow target)
    e1 = ParsedEntry("decision", "jwt-base", 1, 5)
    e1.axiom = "JWT is base auth."
    e1.rejected_paths = "None."
    e1.scope = ["auth"]
    store.commit_parsed_entry(e1)

    # Fingerprint the OQ's write state BEFORE the resolving commit.
    before = oq_meta(d_oq.node_id)

    # 3. jwt-spec narrows jwt-base AND resolves auth-roadblock (two distinct targets,
    #    so no dangling_edge from stacking edges on one entry to the same target).
    e2 = ParsedEntry("decision", "jwt-spec", 1, 5)
    e2.axiom = "Use stateless JWTs with HMAC SHA-256."
    e2.rejected_paths = "RSA (too heavy)."
    e2.narrows = ["jwt-base"]
    e2.resolves = ["auth-roadblock"]
    e2.scope = ["auth"]
    d_e2 = store.commit_parsed_entry(e2)

    # The OQ's computed state flips to resolved (read at query time)...
    assert oq_state("auth-roadblock") == "resolved"
    # ...but the resolving commit wrote NO cascade to the OQ node: its updated_at
    # and Outbox queued_at are byte-identical to before jwt-spec committed.
    assert oq_meta(d_oq.node_id) == before
    # The committing decision is the one node jwt-spec enqueued for (re-)embedding.
    pending = {row["node_id"] for row in store.get_pending_embeddings()}
    assert d_e2.node_id in pending


# ==============================================================================
# P4 — Outbox Queue High Contention & Worker Saturation
# ==============================================================================
@pytest.mark.skip(reason="V3b: the claimed_by claim-reservation machinery is deferred "
                         "(§5.2.8, K3). V1a is single-writer (busy_timeout), so "
                         "claim_pending_embeddings is an ordered SELECT with no reservation — "
                         "there is no multi-drainer double-claim to gate. The V1a single-writer "
                         "drain surface is pinned by "
                         "test_sync.test_sync_outbox_drain_single_writer_semantics. Deferred to V3b (K5).")
def test_pathology_outbox_queue_worker_saturation(isolated_workspace) -> None:
    """Simulates 10 concurrent drainers attempting to drain a saturated outbox queue."""
    config, tmpdir = isolated_workspace
    store = GraphStore(config.db_path)
    
    # Seed 50 active nodes in the database to satisfy FK constraints
    node_ids = []
    for i in range(50):
        e = ParsedEntry("decision", f"contend-{i}", 1, 5)
        e.axiom = f"Axiom {i}"
        e.rejected_paths = "None."
        d = store.commit_parsed_entry(e)
        node_ids.append(d.node_id)
        
        # Add to outbox queue (V1a 3-column shape: node_id only, no embedding_text)
        store.add_pending_embedding(d.node_id)
        
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
    entry.axiom = "We strictly use large text buffers to overflow budget." * 1500
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
