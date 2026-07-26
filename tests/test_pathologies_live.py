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

from live_helpers import live_tests_disabled
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

HAS_LIVE_KEYS = (not live_tests_disabled()) and bool(os.environ.get("GEMINI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY"))

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
# P6 — Deprecated Rotation Modes: Prune and Mark both behave as Archive
# ==============================================================================
def _write_rotation_mode(config, mode: str) -> None:
    """Writes `rotation_mode` into the workspace's config.toml and reloads it.

    The epoch-1 pinning lives in the config LOADER, so a test that assigns
    ``config.rotation_mode`` directly bypasses the behaviour it means to check.

    Args:
        config: The workspace config to rewrite and reload in place.
        mode: The rotation mode to write.
    """
    from mitos.config import MitosConfig

    os.makedirs(config.mitos_dir, exist_ok=True)
    with open(os.path.join(config.mitos_dir, "config.toml"), "w", encoding="utf-8") as f:
        f.write(f'rotation_mode = "{mode}"\n')
    config.rotation_mode = MitosConfig(config.workspace_dir).rotation_mode
@pytest.mark.skipif(not HAS_LIVE_KEYS, reason="Requires live GEMINI API key")
def test_pathology_rotation_mode_prune_now_behaves_as_archive(isolated_workspace) -> None:
    """A `prune` workspace now ARCHIVES — the deprecation's regression fixture.

    Retargeted from "prune deletes entries from the buffer instead of archiving".
    `prune` removed the block from the buffer and wrote it NOWHERE, so the node had
    no source block and `rebuild` — the tool's own repair story — could never
    reconstruct it. The coercion is strictly preserving: both modes clear the buffer,
    archive additionally keeps the text.

    The mode is loaded from `.mitos/config.toml`, not assigned onto the config
    object, because the pinning lives in the loader — assigning the attribute
    directly would bypass the very behaviour under test.
    """
    config, tmpdir = isolated_workspace
    from mitos.cli import cmd_init, cmd_sync
    cmd_init(config)

    _write_rotation_mode(config, "prune")
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
    
    # Buffer is cleared, exactly as `prune` did.
    with open(config.decisions_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "s1-prune" not in content

    # But the text SURVIVES in the archive now — the whole point of the deprecation.
    # Under `prune` the entry existed in neither the buffer nor an archive, and its
    # node was permanently unreconstructable.
    assert os.path.exists(config.archive_dir)
    archives = os.listdir(config.archive_dir)
    assert len(archives) == 1
    with open(os.path.join(config.archive_dir, archives[0]), "r", encoding="utf-8") as f:
        assert "s1-prune" in f.read()


@pytest.mark.skipif(not HAS_LIVE_KEYS, reason="Requires live GEMINI API key")
def test_pathology_rotation_mode_mark_now_behaves_as_archive(isolated_workspace) -> None:
    """A `mark` workspace now ARCHIVES and writes NO sentinel wrapper.

    Retargeted from "mark comments out entries in the buffer". The wrapper it wrote
    was never recognized as a delimiter by the entry-stream tokenizer, so both
    sentinel lines were absorbed as continuation lines of the adjacent field — and
    when that field is `**Mechanisms:**`, part of the canonical core, the node id
    shifts, for the rotated entry AND for the un-rotated one above it. Here the
    coercion is a rescue, not merely a change.
    """
    config, tmpdir = isolated_workspace
    from mitos.cli import cmd_init, cmd_sync
    cmd_init(config)

    _write_rotation_mode(config, "mark")
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
    
    # No sentinel wrapper is written any more, anywhere.
    with open(config.decisions_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "s1-mark" not in content
    assert "<!-- ROTATED START" not in content
    assert "ROTATED END -->" not in content

    # The entry is archived intact, so `rebuild` can still find its source block.
    assert os.path.exists(config.archive_dir)
    archives = os.listdir(config.archive_dir)
    assert len(archives) == 1
    with open(os.path.join(config.archive_dir, archives[0]), "r", encoding="utf-8") as f:
        archived = f.read()
    assert "s1-mark" in archived
    assert "ROTATED" not in archived


# ==============================================================================
# P7 — Renderer Warning on Budget Overflow
# ==============================================================================
def test_pathology_renderer_budget_overflow_warning(isolated_workspace, capsys) -> None:
    """Verifies an over-ceiling corpus renders silently and degrades to the index.

    The warnings used to print mid-render, burying the record receipt under a wall
    of repeated lines — the render itself must stay silent. And per the
    global-render-degrades ADR, a corpus whose full render would breach the global
    ceiling now writes the oneline index instead, so live_axioms.md itself no
    longer appears in ``.overflows`` (the index fits comfortably).
    """
    config, tmpdir = isolated_workspace
    store = GraphStore(config.db_path)

    # Commit a massive node to push the would-be full render over the 50,000-char ceiling.
    entry = ParsedEntry("decision", "massive-axiom", 1, 5)
    entry.axiom = "We strictly use large text buffers to overflow budget." * 1500
    entry.rejected_paths = "None."
    store.commit_parsed_entry(entry)

    renderer = MitosRenderer(config.workspace_dir)
    renderer.render_all(store)

    # The global file exists — as the degraded oneline index, well under the ceiling.
    live_axioms_path = os.path.join(config.workspace_dir, "live_axioms.md")
    assert os.path.exists(live_axioms_path)
    with open(live_axioms_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert content.startswith("# Live Axioms — Index")
    assert "massive-axiom" in content
    assert len(content) < 50000

    # Nothing is printed (so a record receipt can't be buried), and the index-mode
    # global file no longer reports itself over-ceiling.
    captured = capsys.readouterr()
    assert "exceeds" not in captured.out
    assert "[Warning]" not in captured.out
    assert not [o for o in renderer.overflows if o["name"] == "live_axioms.md"]
