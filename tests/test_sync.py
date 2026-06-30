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
    """New buffer entries are parsed, committed VERBATIM (strict-deterministic sync — no LLM enrichment), and rotated."""
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

    # 2. Strict-deterministic sync makes no LLM call — the entry commits verbatim.
    #    (The google.genai.Client patch + mock key below only satisfy the key/embed gate.)

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
    # Committed VERBATIM — the authored axiom/mechanisms/scope, never an LLM rewrite.
    assert node["core_axiom"] == "Use pure logic cores."
    assert node["mechanisms"] == ["python"]
    assert node["scope"] == ["core"]
    # OD3 confirmation metadata: deterministic sync stamps the user/author, not a model.
    assert node["confirmed_by"] == "user"
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


# --------------------------------------------------------------------------- #
# Phase 4a — questions.md steady-state ingestion + per-entry commit-stage
# quarantine floor. The quarantine lives in perform_sync ABOVE the commit, so it
# is driven through perform_sync (mock-key + mocked client just satisfy the
# decision-enrichment key gate; the OQ branch never calls the client). Node ids
# are read back from the store, never hardcoded.
# --------------------------------------------------------------------------- #

_QUESTIONS_HEADER = (
    "# Open Questions\n"
    "<!-- BEGIN ENTRIES — new open questions go directly below this line, newest first -->\n\n"
)


def _set_enrichment_passthrough(mock_client: MagicMock) -> None:
    """Wires the mocked Gemini client to return a UNIQUE refined axiom per call.

    Each decision in the batch is enriched once; a per-call distinct axiom keeps
    distinct decisions distinct (a fixed axiom would collapse several decisions to
    one canonical core). ``suggested_relationships`` is empty so the only edges are
    the authored ones. Open questions skip enrichment entirely, so this is never
    called for them.
    """
    counter = {"n": 0}

    def _gen(*args: object, **kwargs: object) -> MagicMock:
        counter["n"] += 1
        resp = MagicMock()
        resp.text = json.dumps(
            {
                "refined_core_axiom": f"Refined axiom number {counter['n']}.",
                "refined_mechanisms": [],
                "refined_scope": ["core"],
                "suggested_relationships": {},
            }
        )
        return resp

    mock_client.return_value.models.generate_content.side_effect = _gen


def _append_decision(config: MitosConfig, text: str) -> None:
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def _write_questions(tmpdir: str, body: str) -> str:
    path = os.path.join(tmpdir, "questions.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_QUESTIONS_HEADER + body)
    return path


_HOST_DECISION = (
    "## 2026-05-19 — host-decision — Host Decision\n"
    "**Decided:** Use the host approach.\n"
    "**Rejected:** The alternatives.\n"
    "**Scope:** core\n"
)


@patch("google.genai.Client")
def test_sync_ingests_questions_md_and_commits_derives_from(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """OQ ingestion happy path: both OQ nodes land and an OQ→D derives_from commits.

    Decisions-first ordering (D1) lands the typical Derives-From: forward-ref on the
    first pass — the host decision commits before the open question that derives
    from it.
    """
    config, manager, tmpdir = sync_env
    os.environ["GEMINI_API_KEY"] = "mock_key"
    _set_enrichment_passthrough(mock_client)

    _append_decision(config, _HOST_DECISION)
    _write_questions(
        tmpdir,
        "### oq-one\n\n"
        "**Topic:** Embedding model selection for v0.2.\n"
        "**Questions:** Do we pin one model or allow per-project choice?\n\n"
        "### oq-two\n\n"
        "**Topic:** Whether the host approach needs revisiting at scale.\n"
        "**Questions:** Does the host approach hold past 1k nodes?\n"
        "**Derives-From:** host-decision\n",
    )

    manager.perform_sync(auto_accept=True)

    store = GraphStore(config.db_path)
    oqs = store.get_open_questions()
    assert {q["slug"] for q in oqs} == {"oq-one", "oq-two"}

    host = store.get_node_by_slug("host-decision")
    assert host is not None
    oq_two_id = next(q["id"] for q in oqs if q["slug"] == "oq-two")

    derives = [e for e in store.get_edges() if e["edge_type"] == "derives_from"]
    assert len(derives) == 1
    assert derives[0]["source_id"] == oq_two_id
    assert derives[0]["target_id"] == host["id"]


@patch("google.genai.Client")
def test_sync_missing_questions_md_is_healthy(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """An absent questions.md is healthy-empty: no FileNotFoundError, decisions commit."""
    config, manager, tmpdir = sync_env
    os.environ["GEMINI_API_KEY"] = "mock_key"
    _set_enrichment_passthrough(mock_client)

    assert not os.path.exists(os.path.join(tmpdir, "questions.md"))
    _append_decision(config, _HOST_DECISION)

    # Must not raise.
    manager.perform_sync(auto_accept=True)

    store = GraphStore(config.db_path)
    assert store.get_node_by_slug("host-decision") is not None
    assert store.get_open_questions() == []


@patch("google.genai.Client")
def test_sync_questions_md_file_level_error_bulkheads_from_decisions(
    mock_client: MagicMock,
    sync_env: Tuple[MitosConfig, MitosSyncManager, str],
    capsys: pytest.CaptureFixture,
) -> None:
    """File-level bulkhead (D4/P7): a broken questions.md warns + yields zero OQs,
    while decisions.md still commits.

    questions.md is made a *directory*, so the snapshot copy raises IsADirectoryError
    (an OSError) — a deterministic file-level failure that is isolated to OQ ingestion.
    """
    config, manager, tmpdir = sync_env
    os.environ["GEMINI_API_KEY"] = "mock_key"
    _set_enrichment_passthrough(mock_client)

    # A directory at the questions.md path: exists() is True, but shutil.copy raises.
    os.makedirs(os.path.join(tmpdir, "questions.md"))
    _append_decision(config, _HOST_DECISION)

    manager.perform_sync(auto_accept=True)

    captured = capsys.readouterr()
    assert "Could not snapshot questions.md" in captured.out

    store = GraphStore(config.db_path)
    assert store.get_node_by_slug("host-decision") is not None  # decisions unaffected
    assert store.get_open_questions() == []  # zero OQ entries, not a crash


@patch("google.genai.Client")
def test_sync_questions_md_undecodable_bytes_bulkheads_from_decisions(
    mock_client: MagicMock,
    sync_env: Tuple[MitosConfig, MitosSyncManager, str],
    capsys: pytest.CaptureFixture,
) -> None:
    """File-level bulkhead, parse axis (D4/P7): a questions.md with invalid UTF-8
    bytes warns + yields zero OQs while decisions.md still commits.

    The snapshot copy is a BINARY copy, so undecodable bytes pass straight through it
    and only blow up when parse_file_reversed re-reads the snapshot as UTF-8. Without
    wrapping the OQ read+parse this UnicodeDecodeError would propagate and abort the
    WHOLE sync (decisions included) — the exact cross-buffer contamination D4 forbids.
    """
    config, manager, tmpdir = sync_env
    os.environ["GEMINI_API_KEY"] = "mock_key"
    _set_enrichment_passthrough(mock_client)

    # Invalid UTF-8 bytes in questions.md (copies fine as binary, fails utf-8 parse).
    with open(os.path.join(tmpdir, "questions.md"), "wb") as f:
        f.write(b"# Open Questions\n<!-- BEGIN ENTRIES -->\n\xff\xfe### oq\n"
                b"**Topic:** x\n**Questions:** y\n")
    _append_decision(config, _HOST_DECISION)

    # Must not raise — the OQ buffer fault is isolated.
    manager.perform_sync(auto_accept=True)

    captured = capsys.readouterr()
    assert "Could not parse questions.md" in captured.out

    store = GraphStore(config.db_path)
    assert store.get_node_by_slug("host-decision") is not None  # decisions unaffected
    assert store.get_open_questions() == []  # zero OQ entries, not a crash


@patch("google.genai.Client")
def test_sync_malformed_decision_entry_does_not_strand_oq(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """Symmetric bulkhead: a malformed DECISION entry is per-entry isolated, and OQ
    ingestion still proceeds (a defect in one buffer never strands the other)."""
    config, manager, tmpdir = sync_env
    os.environ["GEMINI_API_KEY"] = "mock_key"
    _set_enrichment_passthrough(mock_client)

    # A decision missing the required **Rejected:** field (M5) → collector-isolated.
    _append_decision(
        config,
        "## 2026-05-19 — broken-decision — Broken\n"
        "**Decided:** This decision omits the required rejected paths.\n",
    )
    _write_questions(
        tmpdir,
        "### healthy-oq\n\n"
        "**Topic:** A question that should still ingest.\n"
        "**Questions:** Does the OQ buffer survive a decision-side parse defect?\n",
    )

    manager.perform_sync(auto_accept=True)

    store = GraphStore(config.db_path)
    assert store.get_node_by_slug("broken-decision") is None  # isolated, not committed
    assert {q["slug"] for q in store.get_open_questions()} == {"healthy-oq"}


@patch("google.genai.Client")
def test_sync_single_forward_ref_converges_in_one_sync(
    mock_client: MagicMock,
    sync_env: Tuple[MitosConfig, MitosSyncManager, str],
    capsys: pytest.CaptureFixture,
) -> None:
    """4b fixpoint, the headline (DoD #11 axis 1): a single cross-file forward-ref
    converges in ONE sync.

    A decision that Resolves: an open question authored in questions.md hits the
    opposite file order — decisions-first attempts the decision before its OQ target,
    so on the main pass the resolves edge is a forward-ref → missing_target →
    quarantine. Under 4a this stranded the decision for a SECOND sync; under 4b's
    fixpoint the OQ commits on the main pass and the re-attempt then lands the
    decision + its resolves edge in THIS sync. (Was
    test_sync_quarantines_forward_ref_missing_target_as_guiding_vector under 4a; the
    guiding-vector coverage moved to test_sync_unauthored_target_residual_guiding_vector.)
    """
    config, manager, tmpdir = sync_env
    os.environ["GEMINI_API_KEY"] = "mock_key"
    _set_enrichment_passthrough(mock_client)

    _append_decision(
        config,
        "## 2026-05-19 — resolver-decision — Resolver\n"
        "**Decided:** This decision answers the open thread.\n"
        "**Rejected:** Leaving it open.\n"
        "**Resolves:** oq-target\n",
    )
    _write_questions(
        tmpdir,
        "### oq-target\n\n"
        "**Topic:** The open thread the decision resolves.\n"
        "**Questions:** Which approach do we commit to?\n",
    )

    manager.perform_sync(auto_accept=True)

    captured = capsys.readouterr()
    # Converged: nothing left in the residual, so no [Quarantined] vector fired.
    assert "[Quarantined]" not in captured.out
    assert "0 unresolved" in captured.out

    store = GraphStore(config.db_path)
    # BOTH nodes committed in the one sync.
    resolver = store.get_node_by_slug("resolver-decision")
    assert resolver is not None
    oqs = store.get_open_questions()
    assert {q["slug"] for q in oqs} == {"oq-target"}
    oq_target_id = next(q["id"] for q in oqs if q["slug"] == "oq-target")

    # The resolves edge landed in the fixpoint (decision → OQ), endpoints read back.
    resolves = [e for e in store.get_edges() if e["edge_type"] == "resolves"]
    assert len(resolves) == 1
    assert resolves[0]["source_id"] == resolver["id"]
    assert resolves[0]["target_id"] == oq_target_id


@patch("google.genai.Client")
def test_sync_deep_acyclic_chain_converges_in_one_sync(
    mock_client: MagicMock,
    sync_env: Tuple[MitosConfig, MitosSyncManager, str],
) -> None:
    """4b fixpoint, the deep case (DoD #11 axis 1): a cross-file forward-ref chain
    whose dependency direction alternates across files converges in ONE sync.

    Chain: d1 Resolves: q1; q1 Derives-From: d2; d2 Resolves: q2; q2 terminal. The
    `resolves` (D→OQ) and `derives_from` (OQ→D) edges point opposite ways, so neither
    decisions-first nor oldest-first lands the whole chain on the main pass — only the
    terminal q2 commits there; the fixpoint walks the rest (d2 → q1 → d1) over its
    retry passes. All four nodes and all three edges land in a single sync.
    """
    config, manager, tmpdir = sync_env
    os.environ["GEMINI_API_KEY"] = "mock_key"
    _set_enrichment_passthrough(mock_client)

    # Decisions, authored newest-first (the buffer convention): d1 (newer) on top.
    _append_decision(
        config,
        "## 2026-05-21 — d1 — D1\n"
        "**Decided:** The leaf decision, resolving q1.\n"
        "**Rejected:** Leaving q1 open.\n"
        "**Resolves:** q1\n\n"
        "## 2026-05-19 — d2 — D2\n"
        "**Decided:** The mid decision, resolving q2.\n"
        "**Rejected:** Leaving q2 open.\n"
        "**Resolves:** q2\n",
    )
    # Open questions, newest-first: q1 (which derives from d2) on top, terminal q2 below.
    _write_questions(
        tmpdir,
        "### q1\n\n"
        "**Topic:** The question d1 resolves and that derives from d2.\n"
        "**Questions:** Does q1 hold given d2?\n"
        "**Derives-From:** d2\n\n"
        "### q2\n\n"
        "**Topic:** The terminal question d2 resolves.\n"
        "**Questions:** Which approach for q2?\n",
    )

    manager.perform_sync(auto_accept=True)

    store = GraphStore(config.db_path)
    d1 = store.get_node_by_slug("d1")
    d2 = store.get_node_by_slug("d2")
    assert d1 is not None and d2 is not None

    oqs = store.get_open_questions()
    oq_ids = {q["slug"]: q["id"] for q in oqs}
    assert set(oq_ids) == {"q1", "q2"}

    edges = store.get_edges()
    resolves = {(e["source_id"], e["target_id"]) for e in edges if e["edge_type"] == "resolves"}
    derives = {(e["source_id"], e["target_id"]) for e in edges if e["edge_type"] == "derives_from"}
    # d1→q1 and d2→q2 (two resolves); q1→d2 (one derives_from).
    assert resolves == {(d1["id"], oq_ids["q1"]), (d2["id"], oq_ids["q2"])}
    assert derives == {(oq_ids["q1"], d2["id"])}


@patch("google.genai.Client")
def test_sync_fixpoint_is_load_bearing_for_deep_chain(
    mock_client: MagicMock,
    sync_env: Tuple[MitosConfig, MitosSyncManager, str],
) -> None:
    """P10 'provoke the failure' — the same deep chain does NOT fully converge with the
    fixpoint stubbed out, proving the fixpoint is load-bearing (not incidental).

    With _commit_quarantine_fixpoint replaced by a no-op that commits nothing and
    returns the whole quarantine set as residual, only the terminal q2 commits on the
    main pass; the deepest decision d1 (two hops up the chain) does NOT — it would need
    a second sync. This is the RED-without-4b proof the floor-only behaviour leaves.
    """
    config, manager, tmpdir = sync_env
    os.environ["GEMINI_API_KEY"] = "mock_key"
    _set_enrichment_passthrough(mock_client)

    _append_decision(
        config,
        "## 2026-05-21 — d1 — D1\n"
        "**Decided:** The leaf decision, resolving q1.\n"
        "**Rejected:** Leaving q1 open.\n"
        "**Resolves:** q1\n\n"
        "## 2026-05-19 — d2 — D2\n"
        "**Decided:** The mid decision, resolving q2.\n"
        "**Rejected:** Leaving q2 open.\n"
        "**Resolves:** q2\n",
    )
    _write_questions(
        tmpdir,
        "### q1\n\n"
        "**Topic:** The question d1 resolves and that derives from d2.\n"
        "**Questions:** Does q1 hold given d2?\n"
        "**Derives-From:** d2\n\n"
        "### q2\n\n"
        "**Topic:** The terminal question d2 resolves.\n"
        "**Questions:** Which approach for q2?\n",
    )

    # Disable the fixpoint: commit nothing, surface everything as residual.
    def _noop_fixpoint(self, quarantined, synced_blocks):  # type: ignore[no-untyped-def]
        return list(quarantined)

    with patch.object(MitosSyncManager, "_commit_quarantine_fixpoint", _noop_fixpoint):
        manager.perform_sync(auto_accept=True)

    store = GraphStore(config.db_path)
    # The terminal OQ committed on the main pass; the deepest decision did NOT.
    assert {q["slug"] for q in store.get_open_questions()} == {"q2"}
    assert store.get_node_by_slug("d1") is None
    assert store.get_node_by_slug("d2") is None


@patch("google.genai.Client")
def test_sync_true_cycle_surfaces_loud_and_returns(
    mock_client: MagicMock,
    sync_env: Tuple[MitosConfig, MitosSyncManager, str],
    capsys: pytest.CaptureFixture,
) -> None:
    """4b fixpoint, the cycle case (DoD #11 axis 2): a true 2-node mutual-reference
    cycle commits NEITHER node, prints a loud per-entry vector for each, and the sync
    RETURNS (no hang, no exception, no whole-sync abort).

    cycle-decision Resolves: cycle-oq AND cycle-oq Derives-From: cycle-decision — each
    references the other, so neither can commit first. The fixpoint makes zero progress,
    terminates after one no-progress pass, and the residual is reported as a loud vector
    per member. Reaching the assertions at all IS the no-hang proof (a wedge would never
    return); P7 holds — no exception escapes perform_sync.
    """
    config, manager, tmpdir = sync_env
    os.environ["GEMINI_API_KEY"] = "mock_key"
    _set_enrichment_passthrough(mock_client)

    _append_decision(
        config,
        "## 2026-05-19 — cycle-decision — Cycle Decision\n"
        "**Decided:** This decision resolves an OQ that derives from it.\n"
        "**Rejected:** Breaking the cycle.\n"
        "**Resolves:** cycle-oq\n",
    )
    _write_questions(
        tmpdir,
        "### cycle-oq\n\n"
        "**Topic:** A question that derives from the decision that resolves it.\n"
        "**Questions:** Which way does this cycle resolve?\n"
        "**Derives-From:** cycle-decision\n",
    )

    # Must RETURN — not hang, not raise.
    manager.perform_sync(auto_accept=True)

    captured = capsys.readouterr()
    # A loud per-entry vector named each member of the cycle.
    assert captured.out.count("[Quarantined]") == 2
    assert "cycle-decision" in captured.out
    assert "cycle-oq" in captured.out
    # Post-fixpoint (D4) wording — not the optimistic "settles next sync" framing.
    assert "not present anywhere in this corpus" in captured.out

    store = GraphStore(config.db_path)
    # NEITHER node committed.
    assert store.get_node_by_slug("cycle-decision") is None
    assert {q["slug"] for q in store.get_open_questions()} == set()


@patch("google.genai.Client")
def test_sync_unauthored_target_residual_guiding_vector(
    mock_client: MagicMock,
    sync_env: Tuple[MitosConfig, MitosSyncManager, str],
    capsys: pytest.CaptureFixture,
) -> None:
    """4b residual (preserves 4a's guiding-vector UX for the genuinely-unresolvable
    case): a reference to a target authored NOWHERE quarantines after the exhausted
    fixpoint with the post-fixpoint (D4) vector, and the entry stays in its buffer.

    This is the test that REPLACES 4a's guiding-vector coverage: under 4b a forward-ref
    quarantine means the target is truly absent (the fixpoint already retried every
    in-corpus dependency), so the vector names that honestly rather than promising a
    next-sync commit.
    """
    config, manager, tmpdir = sync_env
    os.environ["GEMINI_API_KEY"] = "mock_key"
    _set_enrichment_passthrough(mock_client)

    _append_decision(
        config,
        "## 2026-05-19 — orphan-resolver — Orphan Resolver\n"
        "**Decided:** This decision resolves a question that was never authored.\n"
        "**Rejected:** Authoring the question.\n"
        "**Resolves:** nonexistent-oq\n",
    )

    manager.perform_sync(auto_accept=True)

    captured = capsys.readouterr()
    assert "[Quarantined]" in captured.out
    assert "orphan-resolver" in captured.out
    assert "not present anywhere in this corpus" in captured.out
    # Honest post-fixpoint framing — must NOT carry 4a's optimistic "settles next sync".
    assert "commit on a subsequent sync once its target lands" not in captured.out

    store = GraphStore(config.db_path)
    assert store.get_node_by_slug("orphan-resolver") is None  # never committed

    # Quarantined entry stays in its buffer (never rotated) for a fix-and-re-sync.
    with open(config.decisions_file, "r", encoding="utf-8") as f:
        assert "orphan-resolver" in f.read()


@patch("google.genai.Client")
def test_sync_quarantine_isolates_whole_commit_error_class(
    mock_client: MagicMock,
    sync_env: Tuple[MitosConfig, MitosSyncManager, str],
    capsys: pytest.CaptureFixture,
) -> None:
    """Whole-class quarantine, axis (b): a kind_constraint_violation ALSO isolates and
    does NOT abort the sync — proof the catch is the CommitError CLASS, not a
    missing_target-only filter (P10: a missing_target-only catch would let this abort
    the whole batch, so the healthy OQ below would not commit).

    Under 4b this is a *permanent* failure: the fixpoint retries the kind violation
    once, makes no progress, and falls to the residual — reported via the relocated
    _report_commit_quarantine. End state is unchanged from 4a (violator never commits,
    survivor-OQ does, the code string still prints).
    """
    config, manager, tmpdir = sync_env
    os.environ["GEMINI_API_KEY"] = "mock_key"
    _set_enrichment_passthrough(mock_client)

    # Seed an existing DECISION target so the offending edge is a kind violation
    # (resolves is D→OQ; a resolves D→D is kind_constraint_violation), not a
    # missing_target.
    seed = GraphStore(config.db_path)
    target = ParsedEntry("decision", "target-decision", 1, 5)
    target.axiom = "A pre-existing decision target."
    target.rejected_paths = "None."
    seed.commit_parsed_entry(target)

    _append_decision(
        config,
        "## 2026-05-19 — kind-violator — Kind Violator\n"
        "**Decided:** This decision wrongly resolves another decision.\n"
        "**Rejected:** Authoring it correctly.\n"
        "**Resolves:** target-decision\n",
    )
    _write_questions(
        tmpdir,
        "### survivor-oq\n\n"
        "**Topic:** An open question that must still commit.\n"
        "**Questions:** Does one entry's kind violation abort the batch?\n",
    )

    manager.perform_sync(auto_accept=True)

    captured = capsys.readouterr()
    assert "kind_constraint_violation" in captured.out
    assert "kind-violator" in captured.out

    store = GraphStore(config.db_path)
    assert store.get_node_by_slug("kind-violator") is None  # quarantined
    # The batch did NOT abort — the healthy OQ committed after the rejected entry.
    assert {q["slug"] for q in store.get_open_questions()} == {"survivor-oq"}


@patch("google.genai.Client")
def test_sync_open_questions_never_rotate(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """OQ does not rotate (D5): questions.md is byte-unchanged after sync, no archive
    carries the OQ, while the decision rotates normally."""
    config, manager, tmpdir = sync_env
    os.environ["GEMINI_API_KEY"] = "mock_key"
    _set_enrichment_passthrough(mock_client)

    _append_decision(config, _HOST_DECISION)
    questions_path = _write_questions(
        tmpdir,
        "### persistent-oq\n\n"
        "**Topic:** A persistent open thread.\n"
        "**Questions:** Should this OQ ever be rotated out of its buffer?\n",
    )
    with open(questions_path, "r", encoding="utf-8") as f:
        questions_before = f.read()

    manager.perform_sync(auto_accept=True)

    # questions.md is a persistent buffer — byte-for-byte unchanged.
    with open(questions_path, "r", encoding="utf-8") as f:
        assert f.read() == questions_before

    store = GraphStore(config.db_path)
    assert {q["slug"] for q in store.get_open_questions()} == {"persistent-oq"}

    # The decision rotated to archive; the OQ did not appear there.
    if os.path.isdir(config.archive_dir):
        for name in os.listdir(config.archive_dir):
            with open(os.path.join(config.archive_dir, name), "r", encoding="utf-8") as f:
                archive_text = f.read()
            assert "persistent-oq" not in archive_text
            assert "persistent open thread" not in archive_text


@patch("google.genai.Client")
def test_sync_decisions_oldest_first_amend_commits_in_one_sync(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """Decisions oldest-first (D2): a newer entry (authored on top) that Amends: an
    older in-buffer entry commits in ONE sync — the reversal lands the older entry
    first, so the amend resolves its target on the first pass."""
    config, manager, tmpdir = sync_env
    os.environ["GEMINI_API_KEY"] = "mock_key"
    _set_enrichment_passthrough(mock_client)

    # Authored newest-first (the buffer convention): newer on top, older below.
    _append_decision(
        config,
        "## 2026-05-20 — newer-decision — Newer\n"
        "**Decided:** The newer refinement.\n"
        "**Rejected:** Status quo.\n"
        "**Amends:** older-decision\n\n"
        "## 2026-05-19 — older-decision — Older\n"
        "**Decided:** The original approach.\n"
        "**Rejected:** Nothing considered.\n",
    )

    manager.perform_sync(auto_accept=True)

    store = GraphStore(config.db_path)
    older = store.get_node_by_slug("older-decision")
    newer = store.get_node_by_slug("newer-decision")
    assert older is not None and newer is not None

    amends = [e for e in store.get_edges() if e["edge_type"] == "amends"]
    assert len(amends) == 1
    assert amends[0]["source_id"] == newer["id"]
    assert amends[0]["target_id"] == older["id"]
