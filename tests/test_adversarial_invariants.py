"""Highly detailed adversarial invariants and digital infrastructure test suite for Mitos.

This test suite comprehensively verifies the core Mitos invariants and principles
(M1–M8, P1–P20) as defined in the Mitos Framework (FRAMEWORK.md) and the Mitos
v0.1 Opera. It tests the system under extreme pressure, circular dependencies,
case-insensitive edge-resolution ambiguities, database corruption recovery,
Lithuanian and Sanskrit multi-byte Unicode slug stability, outbox queue claim-row
concurrency, and cross-project seam integrity.

Test-to-Code ratio is strictly maintained at >1:1 by providing exhaustive,
real-world scenarios with deep structural assertions.
"""

import os
import tempfile
import shutil
import sqlite3
import json
import uuid
import time
import multiprocessing
import pytest
from typing import Tuple, List, Dict, Any, Optional
from unittest.mock import MagicMock, patch

from mitos.config import MitosConfig
from mitos.store import GraphStore, ValidationError, DatabaseError, CommitError
from mitos.identity import compute_node_id
from mitos.parser import ParsedEntry, parse_decisions_file
from mitos.sync import MitosSyncManager
from mitos.renderer import MitosRenderer
from mitos.errors import ParseError


# Load live environment keys from .env if present
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
    """Fixture that provisions a fully isolated temporary workspace for adversarial tests."""
    load_live_env()
    tmpdir = tempfile.mkdtemp()
    config = MitosConfig(tmpdir)
    config.db_path = os.path.join(tmpdir, ".mitos", "graph.sqlite")
    config.decisions_file = os.path.join(tmpdir, "decisions.md")
    config.archive_dir = os.path.join(tmpdir, "decisions", "archive")
    config.qdrant_collection = f"mitos_adversarial_{uuid.uuid4().hex[:8]}"
    
    os.makedirs(config.mitos_dir, exist_ok=True)

    yield config, tmpdir
    
    # Clean up workspace
    shutil.rmtree(tmpdir, ignore_errors=True)


# ==============================================================================
# 1. M1/M2/M3 — Deep DAG Immutability & Computed State Cascade Verification
# ==============================================================================
def test_invariant_m1_m2_m3_deep_dag_and_cascades(isolated_workspace) -> None:
    """Tests Axiom Immutability (M1), in-place edge addition (M1 commentary), and
    Computed State (M3) across the OQ Stage-2 resolution self-healing cascade.

    Constructs the decision/open-question hierarchy and drives it through the full
    Stage-2 lifecycle:
      - Decision A: active.
      - Open Question Q1: parked → resolved by A (an in-place edge addition on A's
        existing node — same canonical core, so the same id; M1 commentary/edges
        are mutable in place).
      - Decision C: active, supersedes A → A inactive → Q1 self-heals back to
        parked (its resolver is no longer active; V1-D18 Stage-2, §4.5.1).

    Verifies that decision state (active/superseded) and OQ Stage-2 state
    (parked/resolved) are each computed at query time off the right surface — the
    helper MUST split: a decision's state lives on ``get_all_nodes``'
    ``computed_state`` (the kill-edge axis), while an OQ's parked/resolved state
    lives ONLY on ``get_open_questions``' ``state`` (a resolved OQ still appears in
    ``get_all_nodes`` reading ``computed_state="active"`` — never parked/resolved).
    """
    config, tmpdir = isolated_workspace
    store = GraphStore(config.db_path)

    def get_node_state(slug: str) -> str:
        # Decisions: kill-edge state (active/superseded/corrected) via get_all_nodes.
        # OQs: Stage-2 state (parked/resolved) via get_open_questions — get_all_nodes'
        # computed_state is the kill-edge axis and reads "active" for a resolved OQ,
        # so the OQ branch must read the dedicated Stage-2 surface instead.
        for n in store.get_all_nodes():
            if n["slug"] == slug:
                if n["kind"] == "open_question":
                    for oq in store.get_open_questions():
                        if oq["slug"] == slug:
                            return oq["state"]
                    raise ValueError(f"OQ {slug} not in the active OQ view")
                return n["computed_state"]
        raise ValueError(f"Node with slug {slug} not found")

    # 1. Commit Decision A (active)
    a = ParsedEntry("decision", "decision-a", 1, 5)
    a.axiom = "We use WAL mode SQLite for local storage."
    a.rejected_paths = "Postgres (too heavy), MongoDB."
    a.scope = ["substrate", "database"]
    a.mechanisms = ["sqlite", "wal-mode"]
    delta_a = store.commit_parsed_entry(a)

    assert delta_a.node_id is not None
    # node_scope comes back scope-sorted (the store's deterministic contract), not
    # in authoring order.
    assert delta_a.node_scope == ["database", "substrate"]
    assert delta_a.self_old_scope == []

    # Verify node A is active
    assert get_node_state("decision-a") == "active"

    # 2. Commit Open Question Q1 (parked)
    q1 = ParsedEntry("open_question", "question-q1", 6, 10)
    q1.topic = "File lock strategy"
    q1.park_reason = "need to determine file lock strategy"
    q1.questions_raised = ["How do we serialize sync calls?"]
    store.commit_parsed_entry(q1)

    # Verify Q1 is parked
    assert get_node_state("question-q1") == "parked"

    # Now re-commit A with a Resolves: edge to Q1. Same canonical core (axiom +
    # mechanisms unchanged) ⇒ SAME node id (M2 content-hash identity) ⇒ the resolves
    # edge lands IN PLACE on the existing node (M1: the axiom is immutable, but
    # commentary/edges are mutable on a matching core), not a new node.
    a_resolves = ParsedEntry("decision", "decision-a", 1, 6)
    a_resolves.axiom = "We use WAL mode SQLite for local storage."
    a_resolves.rejected_paths = "Postgres (too heavy), MongoDB."
    a_resolves.scope = ["substrate", "database"]
    a_resolves.mechanisms = ["sqlite", "wal-mode"]
    a_resolves.resolves = ["question-q1"]

    delta_a_resolves = store.commit_parsed_entry(a_resolves)
    # M2: the re-commit is the SAME node (slug-free content-hash identity), edge
    # added in place — not a fork.
    assert delta_a_resolves.node_id == delta_a.node_id

    # Verify Q1 is resolved and no longer parked
    assert get_node_state("question-q1") == "resolved"

    # 3. Commit Decision C (supersedes A) from a distinct entry
    c = ParsedEntry("decision", "decision-c", 11, 15)
    c.axiom = "We use SQLite in WAL mode with advisory file locking."
    c.rejected_paths = "No locking (leads to write race)."
    c.scope = ["substrate", "locking"]
    c.supersedes = ["decision-a"]

    store.commit_parsed_entry(c)

    # Verify A is now superseded (inactive)
    assert get_node_state("decision-a") == "superseded"

    # Verify C is active
    assert get_node_state("decision-c") == "active"

    # Verify Q1 flips back to parked because A (which resolved it) is no longer
    # active — V1-D18 Stage-2 self-healing, computed at read time (M3), no cascade
    # write. Resolution does not flow transitively through supersedes.
    assert get_node_state("question-q1") == "parked"


# ==============================================================================
# 2. Case-Insensitive Slug Resolution & Ambiguity Safety
# ==============================================================================
def test_invariant_slug_casefold_collision_is_rejected(isolated_workspace) -> None:
    """V1a structurally PREVENTS the case-variant ambiguity (V1-D4 / MI-13, 8a/G8).

    The prototype let 'Use-SQLite' and 'use-sqlite' coexist and then resolved the
    fuzzy NOCASE ambiguity at read/edge time. V1a closes the door upstream: the
    post-mutation slug-collision assertion (5b) enforces at most ONE active node per
    casefold(slug), so committing a second active node that casefolds to an existing
    active slug — with no kill-edge between them — rolls back with ``slug_collision``.
    The "ambiguous resolution" the prototype tested is now structurally unreachable;
    this pins the V1a enforcement that makes it so (8a pared the prototype assertion).
    """
    config, tmpdir = isolated_workspace
    store = GraphStore(config.db_path)

    # Insert node 1: 'Use-SQLite'
    e1 = ParsedEntry("decision", "Use-SQLite", 1, 5)
    e1.axiom = "Axiom one."
    e1.rejected_paths = "None."
    store.commit_parsed_entry(e1)

    # Insert node 2: 'use-sqlite' (differs only in case) — a casefold collision with no
    # kill-edge between them: V1a rolls it back rather than letting both go active.
    e2 = ParsedEntry("decision", "use-sqlite", 6, 10)
    e2.axiom = "Axiom two."
    e2.rejected_paths = "None."
    with pytest.raises(CommitError) as exc:
        store.commit_parsed_entry(e2)
    assert exc.value.failure is not None
    assert any(item.code == "slug_collision" for item in exc.value.failure.items)

    # The first node remains the single, unambiguous active holder of the casefold slug.
    resolved = store.get_node_by_slug("use-sqlite")
    assert resolved is not None and resolved["slug"] == "Use-SQLite"


# ==============================================================================
# 3. M5 — Self-Healing & Database Corruption Recovery (Golden Source Rebuild)
# ==============================================================================
@pytest.mark.skipif(not HAS_LIVE_KEYS, reason="Requires live GEMINI API key")
def test_invariant_m5_database_corruption_and_rebuild(isolated_workspace) -> None:
    """Verifies Data Sovereignty (M5) rebuildability.

    If the SQLite database is completely deleted or corrupted, running `mitos sync`
    must perfectly rebuild the entire graph store and vector indices from the
    original user-authored decisions.md buffer file.
    """
    config, tmpdir = isolated_workspace
    
    # 1. Initialize Mitos workspace
    from mitos.cli import cmd_init, cmd_sync
    cmd_init(config)
    
    # 2. Write multiple decisions to decisions.md write-buffer.
    # Authored newest-first (the buffer convention): the newer d2 — which
    # Depends-On the older d1 — sits ON TOP of d1. Steady-state sync now parses the
    # buffer oldest-first (V1b 4a reverses it), so d1 commits before d2 and the
    # forward-ref lands in a single sync. (A buffer that placed d2 below d1 would
    # quarantine d2's edge on the first pass, then converge in the SAME sync via 4b's
    # intra-sync fixpoint retry — no second sync needed.)
    entry_text = (
        "## 2026-06-01 — d2 — Second decision\n"
        "**Decided:** Second rule of Mitos.\n"
        "**Rejected:** None.\n"
        "**Mechanisms:** sqlite\n"
        "**Scope:** substrate\n"
        "**Depends-On:** d1\n\n"
        "## 2026-06-01 — d1 — First decision\n"
        "**Decided:** First rule of Mitos.\n"
        "**Rejected:** None.\n"
        "**Mechanisms:** python\n"
        "**Scope:** core\n"
    )
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(entry_text + "\n")
        
    # 3. Run sync to build graph
    cmd_sync(config, auto_accept=True)
    
    # Verify database has nodes and edges
    store = GraphStore(config.db_path)
    assert len(store.get_active_decisions()) == 2
    node_d2 = store.get_node_by_slug("d2")
    assert node_d2 is not None
    
    # 4. Simulate catastrophic database loss (deleting SQLite file)
    os.remove(config.db_path)
    assert not os.path.exists(config.db_path)

    # Re-populate the decisions.md file with the original entries to simulate restore from the user's markdown
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write(
            "# Decisions for Mitos\n\n"
            "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->\n\n"
            f"{entry_text}\n"
        )

    # 5. Re-run init to restore DB skeleton and sync to rebuild entire database
    cmd_init(config)
    cmd_sync(config, auto_accept=True)
    
    # 6. Verify that graph has been perfectly and fully restored from decisions.md!
    store_restored = GraphStore(config.db_path)
    active_decisions = store_restored.get_active_decisions()
    assert len(active_decisions) == 2
    
    d1_restored = store_restored.get_node_by_slug("d1")
    d2_restored = store_restored.get_node_by_slug("d2")
    assert d1_restored is not None
    assert d2_restored is not None
    
    # The two decisions rebuild from decisions.md (M5), and the `Depends-On: d1`
    # edge rebuilds with them — as of V1b 2a the non-kill edge commits (it was
    # warn-deferred in V1a), so the rebuild carries it forward faithfully.
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    edges = conn.execute("SELECT * FROM edges").fetchall()
    assert len(edges) == 1
    assert edges[0]["edge_type"] == "depends_on"
    conn.close()


# ==============================================================================
# 4. M6 — Deduplicating Mechanisms Registry Verification
# ==============================================================================
def test_invariant_m6_mechanism_registry_deduplication(isolated_workspace) -> None:
    """Tests the deduplicating mechanism registry (M6).

    V1b shipped the typed mechanism registry: the live ``mechanisms`` table is
    keyed on a ``canonical_name`` primary key (``authored_name``/``source``/
    ``created_at`` alongside), written first-seen-wins by ``commit_parsed_entry``'s
    decision-gated Phase-5a auto-registration writer. This is the adversarial
    suite's M6 invariant gate: mechanism tags declared across multiple entries are
    normalized (whitespace-folded), deduplicated on the canonical PK, and stored
    once each. Broader registry feature coverage lives in
    ``tests/test_mechanisms.py``.
    """
    config, tmpdir = isolated_workspace
    store = GraphStore(config.db_path)
    
    # Commit Decision A with mechanisms sqlite, wal-mode
    a = ParsedEntry("decision", "a", 1, 5)
    a.axiom = "WAL SQLite."
    a.rejected_paths = "None."
    a.mechanisms = ["sqlite", "wal-mode"]
    store.commit_parsed_entry(a)
    
    # Commit Decision B with mechanisms sqlite, python
    b = ParsedEntry("decision", "b", 6, 10)
    b.axiom = "Python SQLite."
    b.rejected_paths = "None."
    b.mechanisms = ["sqlite ", " python"] # with whitespace
    store.commit_parsed_entry(b)
    
    # Fetch from mechanisms registry table directly
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT canonical_name FROM mechanisms ORDER BY canonical_name ASC").fetchall()
    names = [r["canonical_name"] for r in rows]
    conn.close()
    
    # Verify normalization and deduplication
    assert names == ["python", "sqlite", "wal-mode"]


# ==============================================================================
# 5. M2/M5 — Unicode Stability (Lithuanian / Sanskrit Multi-byte Slugs)
# ==============================================================================
def test_invariant_unicode_slug_and_axiom_stability(isolated_workspace) -> None:
    """Verifies that SHA-256 hash-identity computation remains perfectly stable with Unicode slugs.

    Tests multi-byte content in Lithuanian and Sanskrit characters (e.g. Lithuanian letters
    'ąčęėįšųūž' and Sanskrit devanagari/slangs like 'Kas tu esi? Esmi sapnas tavo tamsioje naktyje').
    This ensures that characters are handled as UTF-8 bytes and produce predictable hashes.
    """
    config, tmpdir = isolated_workspace
    store = GraphStore(config.db_path)
    
    # Lithuanian and Sanskrit multi-byte strings
    lithuanian_slug = "kas-tu-esi-esmi-sapnas-tavo-tamsioje-naktyje"
    devanagari_axiom = "कस्त्वमसि अस्मि स्वप्नस्तव तमसे नक्ते"
    lithuanian_axiom = "Kas tu esi? Esmi sapnas tavo tamsioje naktyje."
    
    # 1. Parse and commit Unicode entry
    e = ParsedEntry("decision", lithuanian_slug, 1, 5)
    e.axiom = f"{devanagari_axiom} — {lithuanian_axiom}"
    e.rejected_paths = "Nothing."
    e.mechanisms = ["unicode-utf8"]
    
    delta = store.commit_parsed_entry(e)

    # 2. Re-calculate the V1a slug-free canonical-core id locally (UTF-8 stable). The
    # slug is NOT part of identity (V1-D2) — 8a migrated this off the prototype
    # slug-inclusive compute_hash onto compute_node_id.
    expected_hash = compute_node_id(
        kind="decision",
        axiom=f"{devanagari_axiom} — {lithuanian_axiom}",
        mechanism_refs=["unicode-utf8"],
    )

    assert delta.node_id == expected_hash
    
    # 3. Retrieve node and assert text matches perfectly without corruption
    node = store.get_node(delta.node_id)
    assert node["slug"] == lithuanian_slug
    assert devanagari_axiom in node["core_axiom"]
    assert lithuanian_axiom in node["core_axiom"]


# ==============================================================================
# 6. Outbox Queue Concurrency and Claim-Row Race Condition Safety
# ==============================================================================
@pytest.mark.skip(reason="V3b: the claimed_by claim-reservation machinery is deferred "
                         "(§5.2.8, K3). V1a is single-writer (busy_timeout serializes writers), "
                         "so claim_pending_embeddings is an ordered SELECT with NO reservation — "
                         "there is no multi-drainer race to gate. The V1a single-writer drain "
                         "surface is pinned by test_sync.test_sync_outbox_drain_single_writer_semantics. "
                         "Deferred to V3b, not silently coerced (K5).")
def test_invariant_outbox_queue_drain_concurrency(isolated_workspace) -> None:
    """Verifies that outbox queue drains strictly protect against double-processing.

    Spawns 5 concurrent worker processes in parallel, all calling `claim_pending_embeddings`
    atomically using a unique drainer ID, and asserts that exactly 50 total records
    are claimed without a single duplicate.
    """
    config, tmpdir = isolated_workspace
    store = GraphStore(config.db_path)
    
    # Insert 50 mock nodes and add them all to pending embeddings outbox queue
    for i in range(50):
        d = ParsedEntry("decision", f"dec-{i}", 1, 5)
        d.axiom = f"Rule {i}"
        d.rejected_paths = "None."
        store.commit_parsed_entry(d)
        node_id = compute_node_id(kind="decision", axiom=f"Rule {i}")
        store.add_pending_embedding(node_id)
        
    # Check outbox size is 50
    assert len(store.get_pending_embeddings()) == 50
    
    # Parallel worker claim simulation using multiprocessing
    def worker_drain(worker_id: int, results_list: list, db_path: str) -> None:
        try:
            db = GraphStore(db_path)
            # Atomically claim a batch of 10 items
            claimed = db.claim_pending_embeddings(f"worker-{worker_id}", limit=10)
            results_list.append(len(claimed))
        except Exception:
            pass
            
    mp_manager = multiprocessing.Manager()
    claimed_counts = mp_manager.list()
    
    processes = []
    for i in range(5):
        p = multiprocessing.Process(target=worker_drain, args=(i, claimed_counts, config.db_path))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        
    # Sum of all claimed counts must be exactly 50 (no double-claims, no misses!)
    assert sum(claimed_counts) == 50


# ==============================================================================
# 7. Circular Dependency Prevention (Commit Cycle Validation)
# ==============================================================================
def test_invariant_circular_dependency_gate(isolated_workspace) -> None:
    """Verifies that circular dependency resolution terminates without infinite loops.

    If Decision A depends on B, B depends on C, and C depends on A, the GraphStore
    must handle the cycle gracefully and complete computed state resolution
    without throwing a RecursionError or locking up.
    """
    config, tmpdir = isolated_workspace
    store = GraphStore(config.db_path)
    
    # 1. Commit Decision A
    a = ParsedEntry("decision", "a", 1, 5)
    a.axiom = "Decision A."
    a.rejected_paths = "None."
    store.commit_parsed_entry(a)
    
    # 2. Commit Decision B depends on A
    b = ParsedEntry("decision", "b", 6, 10)
    b.axiom = "Decision B."
    b.rejected_paths = "None."
    b.depends_on = ["a"]
    store.commit_parsed_entry(b)
    
    # 3. Commit Decision C depends on B
    c = ParsedEntry("decision", "c", 11, 15)
    c.axiom = "Decision C."
    c.rejected_paths = "None."
    c.depends_on = ["b"]
    store.commit_parsed_entry(c)
    
    # 4. Update A to depend on C (Creates Cycle: A -> C -> B -> A!)
    a_cycle = ParsedEntry("decision", "a", 1, 5)
    a_cycle.axiom = "Decision A."
    a_cycle.rejected_paths = "None."
    a_cycle.depends_on = ["c"]
    store.commit_parsed_entry(a_cycle)
    
    # Verify that we can resolve all states successfully without recursion or crash!
    nodes = store.get_all_nodes()
    assert len(nodes) == 3
