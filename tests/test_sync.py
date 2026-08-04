"""Adversarial test suite for the Mitos Sync Pipeline.

Verifies private snapshotting, advisory locking, LLM enrichment mocks, slug collision
correction prompts, and content-aware archive rotation.
"""

import sys
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
    config.env["GEMINI_API_KEY"] = "mock_key"

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
    config.env["GEMINI_API_KEY"] = "mock_key"

    manager.perform_sync(auto_accept=True)

    captured = capsys.readouterr()
    assert "was drafted on 2026-05-10 (>14 days ago) and remains unsynced" in captured.out


@patch("google.genai.Client")
@patch("builtins.input", side_effect=["a"])
def test_sync_slug_collision_correction(mock_input: MagicMock, mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]) -> None:
    """A DECLARED correction at the colliding slug commits as declared, interactively.

    Retargeted: this pinned the ``[c]orrection / [s]upersession`` prompt, which is
    retired — its answer was applied in memory only and never reached the markdown, so
    it minted kill-edges the gold source did not declare. The relation now comes from
    the entry, and the only prompt left on this path is the ordinary accept prompt.
    """
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)
    config.env["GEMINI_API_KEY"] = "mock_key"

    # 1. Seed an existing active decision in the graph
    entry1 = ParsedEntry("decision", "database", 1, 10)
    entry1.axiom = "We use PostgreSQL."
    entry1.rejected_paths = "No SQL."
    entry1.scope = ["database"]
    store.commit_parsed_entry(entry1)

    # 2. Add a colliding slug that DECLARES the correction, as the skip message asks.
    entry2_text = (
        "## 2026-06-01 — database — Database Update\n"
        "**Decided:** We actually use SQLite for local WAL reads.\n"
        "**Rejected:** PostgreSQL dependency.\n"
        "**Corrects:** [database]\n"
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

    # 3. Run sync (interactive; the sole input is "a" for the accept prompt)
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


def _seed_active_decision(store: GraphStore, slug: str, axiom: str) -> str:
    """Commits one active decision under ``slug`` and returns its node id."""
    entry = ParsedEntry("decision", slug, 1, 10)
    entry.axiom = axiom
    entry.rejected_paths = "No SQL."
    entry.scope = ["database"]
    return store.commit_parsed_entry(entry).node_id


@patch("google.genai.Client")
def test_sync_undeclared_slug_collision_is_skipped_under_auto_accept(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """An UNDECLARED slug collision under ``--yes`` is reported and skipped, not auto-retired.

    Auto-mode used to default the collision verb to ``corrects``, minting a killer node
    that retired a real decision on nothing but a slug match. A canonical core can shift
    from any hand-edit to a ``**Mechanisms:**`` line, so that default converted an
    accident into permanent data loss (P5 Ironclad).
    """
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)
    config.env["GEMINI_API_KEY"] = "mock_key"

    _seed_active_decision(store, "database", "We use PostgreSQL.")

    # A colliding entry that declares NO relation at the slug it collides with.
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(
            "\n### database\n\n"
            "**Decided:** We actually use SQLite for local WAL reads.\n"
            "**Rejected:** PostgreSQL dependency.\n"
            "**Mechanisms:** sqlite\n"
        )

    manager.perform_sync(auto_accept=True)

    # Nothing committed, nothing retired.
    nodes = store.get_all_nodes()
    assert len(nodes) == 1, "the colliding entry must not commit"
    assert store.get_edges() == [], "no kill-edge may be minted for an undeclared collision"
    assert store.get_node_state(nodes[0]["id"]) == "active"

    # The entry stays in the buffer, so the author can declare the relation and re-run.
    with open(config.decisions_file, "r", encoding="utf-8") as f:
        assert "### database" in f.read()


@patch("google.genai.Client")
def test_sync_declared_same_slug_supersession_commits_under_auto_accept(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """A DECLARED same-slug supersession commits as declared under ``--yes`` (MI-13 FM1).

    The carve-out that keeps the skip above from breaking a supported pattern: an author
    evolving an axiom while preserving the citation handle. The collision override must
    not rewrite the declared ``supersedes`` into ``corrects``.
    """
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)
    config.env["GEMINI_API_KEY"] = "mock_key"

    original_id = _seed_active_decision(store, "database", "We use PostgreSQL.")

    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(
            "\n### database\n\n"
            "**Decided:** We actually use SQLite for local WAL reads.\n"
            "**Rejected:** PostgreSQL dependency.\n"
            "**Mechanisms:** sqlite\n"
            "**Supersedes:** [database]\n"
        )

    manager.perform_sync(auto_accept=True)

    nodes = store.get_all_nodes()
    assert len(nodes) == 2, "the declared supersession must commit"

    edges = store.get_edges()
    assert len(edges) == 1
    assert edges[0]["edge_type"] == "supersedes", "a declared supersedes must not become corrects"

    successor_id = [n["id"] for n in nodes if n["id"] != original_id][0]
    assert store.get_node_state(original_id) == "superseded"
    assert store.get_node_state(successor_id) == "active"


@patch("google.genai.Client")
def test_sync_outbox_queue_and_drain(mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]) -> None:
    """Verifies that failed embeddings enter pending_embeddings queue and drain on recovery (C2)."""
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)
    config.env["GEMINI_API_KEY"] = "mock_key"

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


def test_sync_outbox_drain_loops_past_batch_limit(sync_env: Tuple[MitosConfig, MitosSyncManager, str]) -> None:
    """A single drain empties an outbox larger than the claim batch (>10).

    Regression: ``claim_pending_embeddings`` bounds each claim to ``limit=10`` and the
    drain used to run exactly once, so a corpus with >10 pending nodes — the state
    right after ``mitos rebuild``/``cutover`` re-seed the whole active set — was left
    with ``vectors < nodes`` until ``sync`` happened to run again (the documented
    single-``sync`` re-embed silently under-delivering). The drain now loops until the
    outbox is empty.
    """
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)

    # Commit 15 nodes (> the 10-row claim batch); each commit enqueues its outbox row.
    n = 15
    for i in range(n):
        e = ParsedEntry("decision", f"drain-{i:02d}", 1, 5)
        e.axiom = f"Axiom {i}"
        e.rejected_paths = "None."
        d = store.commit_parsed_entry(e)
        store.add_pending_embedding(d.node_id)
    assert len(store.get_pending_embeddings()) == n

    # Hermetic embedding deps (see test_sync_outbox_queue_and_drain).
    manager.embed_provider = MagicMock()
    manager.embed_provider.get_embedding = MagicMock(return_value=[0.1, 0.2, 0.3])
    manager.vector_store = MagicMock()
    manager.vector_store.upsert = MagicMock()

    # One drain call must fully empty the outbox — not just the first batch of 10.
    manager.drain_pending_embeddings()

    assert len(store.get_pending_embeddings()) == 0
    assert manager.vector_store.upsert.call_count == n


def test_sync_outbox_drain_stops_on_total_failure(sync_env: Tuple[MitosConfig, MitosSyncManager, str]) -> None:
    """The drain loop's progress guard: a fully-erroring provider terminates, not spins.

    When every claimed row errors on upsert (an embedding-provider outage), the batch
    resolves zero rows; re-claiming would return the same rows forever, so the loop
    breaks after one batch. The rows stay in the outbox (retry_count incremented) to
    drain on a later sync — fail-fast in aggregate: one outage costs one batch of
    attempts, not a hot spin.
    """
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)

    n = 15
    for i in range(n):
        e = ParsedEntry("decision", f"fail-{i:02d}", 1, 5)
        e.axiom = f"Axiom {i}"
        e.rejected_paths = "None."
        d = store.commit_parsed_entry(e)
        store.add_pending_embedding(d.node_id)

    manager.embed_provider = MagicMock()
    manager.embed_provider.get_embedding = MagicMock(return_value=[0.1, 0.2, 0.3])
    manager.vector_store = MagicMock()
    manager.vector_store.upsert = MagicMock(side_effect=Exception("Qdrant down"))

    # Must return (not hang). Only the first batch is attempted, then the guard breaks.
    manager.drain_pending_embeddings()

    # Nothing drained; the loop tried exactly one claim batch (<= the limit of 10)
    # rather than re-claiming the same failing rows.
    assert len(store.get_pending_embeddings()) == n
    assert manager.vector_store.upsert.call_count <= 10


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
    config.env["GEMINI_API_KEY"] = "mock_key"
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
    config.env["GEMINI_API_KEY"] = "mock_key"
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
    config.env["GEMINI_API_KEY"] = "mock_key"
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
    config.env["GEMINI_API_KEY"] = "mock_key"
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
    config.env["GEMINI_API_KEY"] = "mock_key"
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
    config.env["GEMINI_API_KEY"] = "mock_key"
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
    config.env["GEMINI_API_KEY"] = "mock_key"
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
    config.env["GEMINI_API_KEY"] = "mock_key"
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
    config.env["GEMINI_API_KEY"] = "mock_key"
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
    config.env["GEMINI_API_KEY"] = "mock_key"
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
    config.env["GEMINI_API_KEY"] = "mock_key"
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
    config.env["GEMINI_API_KEY"] = "mock_key"
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
    config.env["GEMINI_API_KEY"] = "mock_key"
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


# --- MI-10: confirmed_at is UTC with an explicit offset -------------------------

def _assert_utc_offset(stamp: str, where: str) -> None:
    """Asserts a timestamp is offset-aware UTC (MI-10), not a naive local-time string."""
    from datetime import datetime, timezone

    assert stamp, f"{where}: no stamp written"
    parsed = datetime.fromisoformat(stamp)
    assert parsed.tzinfo is not None, (
        f"{where}: {stamp!r} is naive local time — MI-10 requires an explicit offset"
    )
    assert parsed.utcoffset().total_seconds() == 0, f"{where}: {stamp!r} is not UTC"


@patch("google.genai.Client")
def test_sync_confirmed_at_is_utc_with_offset(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """`perform_sync`'s OD3 stamp is application-supplied UTC ISO-8601 (MI-10).

    Both ``confirmed_at`` writers used ``datetime.now().isoformat()`` while
    ``created_at`` on the very same row used ``_utc_now_iso()`` — so every stamp on
    the live corpus was a naive local-time string sitting beside an offset-aware one.
    """
    config, manager, tmpdir = sync_env
    config.env["GEMINI_API_KEY"] = "mock_key"

    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(
            "\n### utc-stamp\n\n"
            "**Decided:** Timestamps carry their offset.\n"
            "**Rejected:** Naive local time.\n"
            "**Mechanisms:** datetime\n"
        )

    manager.perform_sync(auto_accept=True)

    node = GraphStore(config.db_path).get_all_nodes()[0]
    _assert_utc_offset(node["confirmed_at"], "perform_sync")
    _assert_utc_offset(node["created_at"], "created_at (already correct)")


def test_record_confirmed_at_is_utc_with_offset(
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """`record_decision_entry`'s OD3 stamp carries an offset too (MI-10).

    The second of the two naive writers — the agent-facing write path, which is the
    one that authored 113 of the corpus's 114 confirmation stamps.
    """
    config, manager, tmpdir = sync_env

    result = manager.record_decision_entry(
        "Timestamps carry their offset.", "Naive local time.", ["datetime"],
        slug="utc-stamp-record",
    )
    assert result["state"] == "active", result

    node = GraphStore(config.db_path).get_all_nodes()[0]
    _assert_utc_offset(node["confirmed_at"], "record_decision_entry")


# --- the interactive collision prompt is retired (R6 N1) ------------------------
#
# The prompt asked the author for a relation they could only express in memory:
# `edge_relationship` was set at the prompt and consumed by an in-memory mutation of
# `entry`, and nothing ever spliced the chosen `Corrects:`/`Supersedes:` line into the
# buffer — rotation archives the raw unmodified snapshot slice. So every
# interactively-resolved collision committed a kill-edge the gold source does not
# declare (against P6/M7), which surfaces as removed-edge divergence, offers itself
# for DELETION on the next reconcile, and replays at rebuild without the declaration
# as a permanent casualty. Auto-mode's report-and-skip was the right shape all along:
# it names the exact line to add, so the declaration lands in the markdown.

@patch("google.genai.Client")
@patch("builtins.input", side_effect=AssertionError("no collision prompt may be shown"))
def test_undeclared_collision_is_skipped_interactively_too(
    mock_input: MagicMock, mock_client: MagicMock,
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """An undeclared collision is reported and skipped in BOTH modes, never prompted.

    The skip lands above the accept prompt, so a colliding entry consumes no input at
    all — mocking ``input`` to raise is the assertion.
    """
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)
    config.env["GEMINI_API_KEY"] = "mock_key"

    original_id = _seed_active_decision(store, "database", "We use PostgreSQL.")

    # Sync an empty buffer first so the sample-format header auto-heal — a legitimate
    # write, unrelated to the collision — has already happened when the snapshot is taken.
    manager.perform_sync(auto_accept=False)

    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(
            "\n### database\n\n"
            "**Decided:** We actually use SQLite for local WAL reads.\n"
            "**Rejected:** PostgreSQL dependency.\n"
            "**Mechanisms:** sqlite\n"
        )
    with open(config.decisions_file, "r", encoding="utf-8") as f:
        before = f.read()

    manager.perform_sync(auto_accept=False)

    assert store.get_edges() == [], "no kill-edge the gold source does not declare"
    assert len(store.get_all_nodes()) == 1, "the colliding entry must not commit"
    assert store.get_node_state(original_id) == "active"

    with open(config.decisions_file, "r", encoding="utf-8") as f:
        assert f.read() == before, "the buffer entry must be left byte-unchanged"


@patch("google.genai.Client")
def test_collision_resyncs_cleanly_once_the_author_declares_the_relation(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """The skip is a vector, not a wall: adding the named line commits on re-sync.

    This is what the retired prompt was reaching for, done in the one place a rebuild
    can find it again.
    """
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)
    config.env["GEMINI_API_KEY"] = "mock_key"

    original_id = _seed_active_decision(store, "database", "We use PostgreSQL.")

    entry = ("\n### database\n\n"
             "**Decided:** We actually use SQLite for local WAL reads.\n"
             "**Rejected:** PostgreSQL dependency.\n"
             "**Mechanisms:** sqlite\n")
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(entry)

    manager.perform_sync(auto_accept=True)
    assert store.get_edges() == []

    # The author does what the message told them to.
    with open(config.decisions_file, "r", encoding="utf-8") as f:
        buffered = f.read()
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write(buffered.replace(
            "**Mechanisms:** sqlite\n",
            "**Mechanisms:** sqlite\n**Corrects:** [database]\n",
        ))

    manager.perform_sync(auto_accept=True)

    edges = store.get_edges()
    assert len(edges) == 1 and edges[0]["edge_type"] == "corrects", edges
    assert store.get_node_state(original_id) == "corrected"


@patch("google.genai.Client")
@patch("builtins.input", side_effect=AssertionError("no collision prompt may be shown"))
def test_collision_never_discards_a_kill_edge_authored_at_another_slug(
    mock_input: MagicMock, mock_client: MagicMock,
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """An entry colliding on its own slug never loses a kill-edge authored elsewhere.

    The retired override wholesale-replaced both kill lists (``entry.corrects =
    [entry.slug]; entry.supersedes = []``), so an entry that collided on ``database``
    while declaring ``Supersedes: [legacy-store]`` committed with that authored edge
    silently dropped. Only the interactive path could reach it — auto-mode has skipped
    undeclared collisions since 0.10.2 — which is why retiring the prompt dissolves
    the bug rather than patching it.

    The entry declares nothing at the slug it collides with, so the correct outcome is
    a skip: nothing committed, nothing dropped, and the fix in the author's hands.
    """
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)
    config.env["GEMINI_API_KEY"] = "mock_key"

    _seed_active_decision(store, "database", "We use PostgreSQL.")
    legacy_id = _seed_active_decision(store, "legacy-store", "We use a flat file.")

    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(
            "\n### database\n\n"
            "**Decided:** We actually use SQLite for local WAL reads.\n"
            "**Rejected:** PostgreSQL dependency.\n"
            "**Mechanisms:** sqlite\n"
            "**Supersedes:** [legacy-store]\n"
        )

    manager.perform_sync(auto_accept=False)

    assert store.get_edges() == [], "a skipped entry commits no edges at all"
    assert store.get_node_state(legacy_id) == "active", "nothing may be retired by a skip"
    assert len(store.get_all_nodes()) == 2

    # The author declares at the colliding slug too; now BOTH authored edges land —
    # which is the property the override destroyed.
    with open(config.decisions_file, "r", encoding="utf-8") as f:
        buffered = f.read()
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write(buffered.replace(
            "**Supersedes:** [legacy-store]\n",
            "**Supersedes:** [legacy-store], [database]\n",
        ))

    manager.perform_sync(auto_accept=True)

    types_by_target = {e["target_id"]: e["edge_type"] for e in store.get_edges()}
    assert types_by_target.get(legacy_id) == "supersedes", (
        "the edge authored at another slug must survive the collision branch"
    )
    assert store.get_node_state(legacy_id) == "superseded"


# ===========================================================================
# C′ — the commentary reconcile at the idempotency gate
# ===========================================================================
#
# `sync` skipped every already-committed entry, so a hand-edit to a committed
# entry's commentary was invisible: the receipt said `Regenerated live_axioms.md ✓`
# and the graph kept serving the stale value to every read. The store has ALWAYS
# been able to update commentary in place on a matching canonical core — the branch
# was simply unreachable, gated away by all four callers. C′ routes one caller to it,
# divergence-GATED so a clean corpus behaves exactly as before.

def _seed_committed_buffer(config, manager, *, amends=None, slug="reconcile-me"):
    """Commits one entry through `record`, leaving it IN the buffer for a re-sync.

    Deliberately `record_decision_entry` rather than `perform_sync`: rotation is tied
    to a first sync commit, so a sync-authored entry leaves the buffer immediately and
    is no longer reconcilable (its reconciler is `rebuild`). `record`-authored entries
    never rotate — which is exactly why the live corpus holds 203 entries in the buffer
    against 6 in the archive, and why they are the entries the reconcile actually meets.
    """
    result = manager.record_decision_entry(
        "The reconcilable axiom.", "The original rejected reasoning.",
        ["alpha"], mechanisms=["sqlite"], slug=slug, amends=amends,
        acknowledge_neighbors=True,
    )
    assert result.get("state") == "active", result
    return GraphStore(config.db_path)


def _edit_buffer(config, old, new):
    with open(config.decisions_file, "r", encoding="utf-8") as f:
        text = f.read()
    assert old in text, f"fixture edit target missing: {old!r}"
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write(text.replace(old, new))


@patch("google.genai.Client")
def test_a_clean_corpus_resync_is_byte_identical_behaviour(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """No divergence → the same `continue` as before, and MI-3's no-tick holds.

    The gate is what keeps a 203-entry corpus from becoming 203 prompts and a SONNET
    bill per sync. It is also what keeps `updated_at` still on a byte-identical
    re-commit, which MI-3 requires.
    """
    config, manager, tmpdir = sync_env
    config.env["GEMINI_API_KEY"] = "mock_key"
    store = _seed_committed_buffer(config, manager)
    before = store.get_all_nodes()[0]

    manager.perform_sync(auto_accept=True)

    after = GraphStore(config.db_path).get_all_nodes()[0]
    assert after["updated_at"] == before["updated_at"], "MI-3: no tick without a change"
    assert len(GraphStore(config.db_path).get_all_nodes()) == 1


@patch("google.genai.Client")
def test_a_commentary_edit_is_reconciled_under_auto_accept(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """The brief's flagship case: a corrected `rejected_paths` reaches the graph."""
    config, manager, tmpdir = sync_env
    config.env["GEMINI_API_KEY"] = "mock_key"
    store = _seed_committed_buffer(config, manager)
    node_before = store.get_all_nodes()[0]

    _edit_buffer(config, "The original rejected reasoning.", "The CORRECTED reasoning.")
    manager.perform_sync(auto_accept=True)

    nodes = GraphStore(config.db_path).get_all_nodes()
    assert len(nodes) == 1, "a commentary correction must not mint a second node"
    assert nodes[0]["rejected_paths"] == "The CORRECTED reasoning."
    assert nodes[0]["id"] == node_before["id"], "the canonical core is untouched (M1)"


@patch("google.genai.Client")
def test_a_scope_edit_is_reconciled(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """Scope is the species with real-use evidence — a findability defect, now repairable."""
    config, manager, tmpdir = sync_env
    config.env["GEMINI_API_KEY"] = "mock_key"
    _seed_committed_buffer(config, manager)

    _edit_buffer(config, "**Scope:** alpha", "**Scope:** alpha, beta")
    manager.perform_sync(auto_accept=True)

    assert sorted(GraphStore(config.db_path).get_all_nodes()[0]["scope"]) == ["alpha", "beta"]


@patch("google.genai.Client")
def test_the_reconcile_carries_stored_confirmation_provenance_forward(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """A commentary correction must NOT rewrite or NULL `confirmed_by`/`confirmed_at`.

    Both columns sit in the store's commentary `UPDATE SET`, and at the gate the PARSED
    entry carries `confirmed_by=None` (sync's `"user"` stamp happens further down). So
    a naive reconcile NULLs the stamps, and one that reuses sync's stamping rewrites
    them to `"user"`. Measured on the live corpus: either would destroy 114 stamps —
    113 `agent` plus one model — on a change that touched only prose.
    """
    config, manager, tmpdir = sync_env
    config.env["GEMINI_API_KEY"] = "mock_key"
    store = _seed_committed_buffer(config, manager)
    node_id = store.get_all_nodes()[0]["id"]
    with store._get_connection() as conn:
        conn.execute(
            "UPDATE nodes SET confirmed_by = 'agent', confirmed_at = ? WHERE id = ?",
            ("2026-06-23T13:04:17.147973+00:00", node_id),
        )

    _edit_buffer(config, "The original rejected reasoning.", "The CORRECTED reasoning.")
    manager.perform_sync(auto_accept=True)

    node = GraphStore(config.db_path).get_all_nodes()[0]
    assert node["rejected_paths"] == "The CORRECTED reasoning.", "the fixture must reconcile"
    assert node["confirmed_by"] == "agent", "provenance must survive a commentary fix"
    assert node["confirmed_at"] == "2026-06-23T13:04:17.147973+00:00"


@patch("google.genai.Client")
def test_the_reconcile_does_not_rotate_the_entry(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """Rotation stays tied to a FIRST commit — this is the invisibility mechanism.

    An entry that rotates out of the buffer leaves `sync`'s read-set, so its future
    divergence becomes undetectable by the very surface that just repaired it. A
    reconcile that rotated would quietly re-create the disease it cures.
    """
    config, manager, tmpdir = sync_env
    config.env["GEMINI_API_KEY"] = "mock_key"
    manager.config.pending_threshold = 1
    _seed_committed_buffer(config, manager)

    _edit_buffer(config, "The original rejected reasoning.", "The CORRECTED reasoning.")
    manager.perform_sync(auto_accept=True)

    with open(config.decisions_file, "r", encoding="utf-8") as f:
        assert "### reconcile-me" in f.read(), "a reconciled entry must stay in the buffer"


@patch("google.genai.Client")
def test_an_edge_deletion_is_skipped_under_auto_accept_and_reported(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str],
    capsys: pytest.CaptureFixture,
) -> None:
    """`--yes` widens sync from append-only to mutate — but never to DELETE an edge.

    `commit_parsed_entry` mirrors edges declaratively, so a removed markdown line
    DELETES that edge. You cannot apply an entry's commentary while withholding its
    edge state, so an entry carrying BOTH is wholly skipped — which means its
    commentary divergence stays unreconciled on every non-interactive run. That is the
    right call and it must be stated, not discovered.
    """
    config, manager, tmpdir = sync_env
    config.env["GEMINI_API_KEY"] = "mock_key"
    store = GraphStore(config.db_path)
    _seed_active_decision(store, "predecessor", "The earlier axiom.")

    _seed_committed_buffer(config, manager, amends="predecessor")
    assert len(store.get_edges()) == 1, "the fixture must commit the edge"

    # Drop the relation line AND correct the commentary — the mixed case.
    _edit_buffer(config, "**Amends:** predecessor\n", "")
    _edit_buffer(config, "The original rejected reasoning.", "The CORRECTED reasoning.")
    manager.perform_sync(auto_accept=True)

    after = GraphStore(config.db_path)
    assert len(after.get_edges()) == 1, "the edge must NOT be deleted under --yes"
    node = [n for n in after.get_all_nodes() if n["slug"] == "reconcile-me"][0]
    assert node["rejected_paths"] == "The original rejected reasoning.", (
        "a mixed entry is WHOLLY skipped — commentary cannot be applied alone"
    )
    out = capsys.readouterr().out
    assert "reconcile-me" in out and "skip" in out.lower()


@patch("google.genai.Client")
def test_an_edge_addition_applies_under_auto_accept(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """Additions apply, deletions do not — an asymmetry, matching `record --yes`."""
    config, manager, tmpdir = sync_env
    config.env["GEMINI_API_KEY"] = "mock_key"
    store = GraphStore(config.db_path)
    _seed_active_decision(store, "predecessor", "The earlier axiom.")
    _seed_committed_buffer(config, manager)
    assert store.get_edges() == []

    _edit_buffer(config, "**Scope:** alpha", "**Scope:** alpha\n**Amends:** [predecessor]")
    manager.perform_sync(auto_accept=True)

    edges = GraphStore(config.db_path).get_edges()
    assert len(edges) == 1 and edges[0]["edge_type"] == "amends"


@patch("google.genai.Client")
@patch("builtins.input", side_effect=["s"])
@patch("sys.stdin.isatty", return_value=True)
def test_an_interactive_skip_leaves_the_graph_untouched(
    mock_isatty: MagicMock, mock_input: MagicMock, mock_client: MagicMock,
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """The reconcile has its OWN prompt verb, so answering `[s]kip` changes nothing."""
    config, manager, tmpdir = sync_env
    config.env["GEMINI_API_KEY"] = "mock_key"
    _seed_committed_buffer(config, manager)

    _edit_buffer(config, "The original rejected reasoning.", "The CORRECTED reasoning.")
    manager.perform_sync(auto_accept=False)

    node = GraphStore(config.db_path).get_all_nodes()[0]
    assert node["rejected_paths"] == "The original rejected reasoning."


@patch("google.genai.Client")
@patch("builtins.input", side_effect=["r"])
@patch("sys.stdin.isatty", return_value=True)
def test_an_interactive_reconcile_deletes_exactly_the_dropped_edge(
    mock_isatty: MagicMock, mock_input: MagicMock, mock_client: MagicMock,
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """Interactively, a deletion IS applied — the author saw it named and said yes."""
    config, manager, tmpdir = sync_env
    config.env["GEMINI_API_KEY"] = "mock_key"
    store = GraphStore(config.db_path)
    _seed_active_decision(store, "predecessor", "The earlier axiom.")
    _seed_committed_buffer(config, manager, amends="predecessor")
    assert len(store.get_edges()) == 1

    _edit_buffer(config, "**Amends:** predecessor\n", "")
    manager.perform_sync(auto_accept=False)

    assert GraphStore(config.db_path).get_edges() == []


@patch("google.genai.Client")
def test_the_reconcile_writes_a_write_ahead_audit_row(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """P8: every applied reconcile leaves a durable record of what changed.

    `updated_at` says only THAT something changed. Without this row an agent's edit
    plus `sync --yes` is exactly as unattributable as mutating the graph directly.
    """
    from mitos.telemetry import TelemetryStore

    config, manager, tmpdir = sync_env
    config.env["GEMINI_API_KEY"] = "mock_key"
    _seed_committed_buffer(config, manager)

    _edit_buffer(config, "The original rejected reasoning.", "The CORRECTED reasoning.")
    manager.perform_sync(auto_accept=True)

    rows = TelemetryStore(config.telemetry_path).read_commentary_audit()
    applied = [r for r in rows if r["slug"] == "reconcile-me" and r["outcome"] is None]
    assert len(applied) == 1, f"expected one intent row, got {rows}"
    row = applied[0]
    assert row["fields_changed"] == ["rejected_paths"]
    assert row["prior_values"] == {"rejected_paths": "The original rejected reasoning."}
    assert row["new_values"] == {"rejected_paths": "The CORRECTED reasoning."}


@patch("google.genai.Client")
def test_an_unavailable_audit_store_refuses_the_reconcile(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str],
    capsys: pytest.CaptureFixture,
) -> None:
    """The discipline inversion: mandatory, not best-effort.

    `TelemetryStore` is best-effort by design — a boot failure warns and continues,
    because a telemetry failure must never abort a sync. A P8-MANDATORY row cannot
    inherit that ambient posture: if the audit store is unavailable the reconcile is
    refused and reported, never applied unaudited, or P8 holds only when the disk
    cooperates. This is the sharpest way the design fails in practice if left unstated.
    """
    config, manager, tmpdir = sync_env
    config.env["GEMINI_API_KEY"] = "mock_key"
    _seed_committed_buffer(config, manager)
    _edit_buffer(config, "The original rejected reasoning.", "The CORRECTED reasoning.")

    from mitos.errors import DatabaseError

    with patch("mitos.sync.TelemetryStore.record_commentary_intent",
               side_effect=DatabaseError("audit store unavailable")):
        manager.perform_sync(auto_accept=True)

    node = GraphStore(config.db_path).get_all_nodes()[0]
    assert node["rejected_paths"] == "The original rejected reasoning.", (
        "an unauditable reconcile must NOT be applied"
    )
    # ONE readouterr call: it drains the buffer, so a second returns "" and would
    # silently reduce this to an err-only assertion.
    captured = capsys.readouterr()
    combined = (captured.err + captured.out).lower()
    assert "audit" in combined or "refus" in combined


@patch("google.genai.Client")
def test_a_reconcile_never_carries_a_transcript_rewrite_along(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str],
    capsys: pytest.CaptureFixture,
) -> None:
    """A transcript edit must NOT ride along on a reconcile triggered by another field.

    `commit_parsed_entry` UPDATEs `transcripts` whenever the incoming text differs, and
    the transcript is not in the divergence set — so left attached it landed with no
    printed diff, no `fields_changed`, and no `prior_values`. An agent editing
    `**Rejected:**` and rewriting the transcript block got both applied, and the prior
    transcript then existed nowhere: markdown rewritten, graph updated, audit row
    silent. That is exactly the unattributed graph mutation P8 forbids, inside the
    feature that adds the attribution row.
    """
    from mitos.telemetry import TelemetryStore

    config, manager, tmpdir = sync_env
    config.env["GEMINI_API_KEY"] = "mock_key"
    _seed_committed_buffer(config, manager)

    _edit_buffer(config, "**Scope:** alpha",
                 "**Scope:** alpha\n[DECISION_TRANSCRIPT]\n"
                 "User: the ORIGINAL conversation.\n[/DECISION_TRANSCRIPT]")
    manager.perform_sync(auto_accept=True)
    store = GraphStore(config.db_path)
    node_id = store.get_all_nodes()[0]["id"]
    original_transcript = store.get_transcript(node_id)

    # Now edit a reconcilable field AND rewrite the transcript in the same pass.
    _edit_buffer(config, "The original rejected reasoning.", "The CORRECTED reasoning.")
    _edit_buffer(config, "User: the ORIGINAL conversation.", "User: a FABRICATED replacement.")
    manager.perform_sync(auto_accept=True)

    after = GraphStore(config.db_path)
    assert after.get_all_nodes()[0]["rejected_paths"] == "The CORRECTED reasoning.", (
        "the fixture must actually have reconciled"
    )
    assert after.get_transcript(node_id) == original_transcript, (
        "the stored transcript must be preserved, not silently replaced"
    )
    rows = TelemetryStore(config.telemetry_path).read_commentary_audit()
    applied = [r for r in rows if r["outcome"] is None]
    assert all("transcript" not in (r["fields_changed"] or []) for r in applied)


@patch("google.genai.Client")
def test_sync_without_a_tty_and_without_yes_skips_instead_of_dying(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str],
    capsys: pytest.CaptureFixture,
) -> None:
    """A non-interactive `mitos sync` must not die on the reconcile prompt.

    This gate fires on a corpus state that used to produce ZERO prompts, so prompting
    turned every agent, CI job, cron and piped invocation into a fatal
    `EOF when reading a line`. Skipping is the fail-closed choice: `--yes` is an
    explicit authorization to mutate, and the absence of a terminal is not it.
    """
    config, manager, tmpdir = sync_env
    config.env["GEMINI_API_KEY"] = "mock_key"
    _seed_committed_buffer(config, manager)
    _edit_buffer(config, "The original rejected reasoning.", "The CORRECTED reasoning.")

    # pytest already runs with a non-TTY stdin; assert that explicitly so the test
    # cannot silently start exercising the interactive path.
    assert not sys.stdin.isatty()
    manager.perform_sync(auto_accept=False)  # must not raise

    node = GraphStore(config.db_path).get_all_nodes()[0]
    assert node["rejected_paths"] == "The original rejected reasoning.", "fail closed"
    assert "no terminal" in capsys.readouterr().out


@patch("google.genai.Client")
def test_an_open_question_divergence_does_not_reconcile_forever(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """An open question carrying `**Context:**` must not reconcile on every sync.

    The parser assigns `context`/`invalidates_if` kind-agnostically while the store
    NULLs them for an open question, so a reconcile of one can never converge: it
    printed `Reconciled ✓` and appended a fresh attribution row every single sync, for
    a mutation that provably cannot land.
    """
    from mitos.telemetry import TelemetryStore

    config, manager, tmpdir = sync_env
    config.env["GEMINI_API_KEY"] = "mock_key"
    with open(config.questions_file, "w", encoding="utf-8") as f:
        f.write("# Questions\n<!-- BEGIN ENTRIES -->\n\n### embedding-choice\n\n"
                "**Topic:** which embedding model\n**Questions:**\n- Which dimension?\n")
    manager.perform_sync(auto_accept=True)

    with open(config.questions_file, "r", encoding="utf-8") as f:
        text = f.read()
    with open(config.questions_file, "w", encoding="utf-8") as f:
        f.write(text.replace("**Questions:**",
                             "**Context:** Not permitted on an OQ but parses fine.\n**Questions:**"))
    for _ in range(3):
        manager.perform_sync(auto_accept=True)

    rows = TelemetryStore(config.telemetry_path).read_commentary_audit()
    assert rows == [], f"an open question must never enter the reconcile: {rows}"


@patch("google.genai.Client")
def test_an_unresolvable_citation_is_reported_once_not_audited_forever(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """A citation naming no decision is pre-flighted, not retried into the audit trail.

    The commit raises `missing_target` regardless, and the reconcile correctly never
    quarantines and never retries — but it re-fired every sync, appending an intent row
    PLUS a failure row each time, forever, with retention deliberately deferred. That
    unbounded write also falsified the "reconciles are rare" premise that both the
    retention deferral and the `synchronous=FULL` fsync cost rest on.
    """
    from mitos.telemetry import TelemetryStore

    config, manager, tmpdir = sync_env
    config.env["GEMINI_API_KEY"] = "mock_key"
    _seed_committed_buffer(config, manager)

    _edit_buffer(config, "**Scope:** alpha", "**Scope:** alpha\n**Cites:** ghost-slug")
    for _ in range(3):
        manager.perform_sync(auto_accept=True)

    assert TelemetryStore(config.telemetry_path).read_commentary_audit() == [], (
        "an unreconcilable citation must write no attribution rows at all"
    )
    assert GraphStore(config.db_path).get_edges() == []


@patch("google.genai.Client")
def test_a_kind_illegal_edge_is_reported_once_not_audited_forever(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """A `derives_from` declared on a DECISION can never commit — pre-flight it too.

    The sibling of the unresolvable-citation case above, and the reason that fix was
    too narrow: this target resolves perfectly, so the `missing_target`-only pre-flight
    passed it straight through to a commit the `edges` CHECK can never admit. Every
    sync then wrote an intent row plus a correlated failure row, forever. Measured on
    the cartolina corpus the day it was found: 19 such edges, 6 rows after 3 syncs.

    `derives_from` originates from an open question by definition, so a decision
    declaring it is illegal no matter what it points at.
    """
    from mitos.telemetry import TelemetryStore

    config, manager, tmpdir = sync_env
    config.env["GEMINI_API_KEY"] = "mock_key"
    store = GraphStore(config.db_path)
    _seed_active_decision(store, "a-real-decision", "A perfectly resolvable axiom.")
    _seed_committed_buffer(config, manager)

    _edit_buffer(config, "**Scope:** alpha",
                 "**Scope:** alpha\n**Derives-From:** a-real-decision")
    for _ in range(3):
        manager.perform_sync(auto_accept=True)

    assert TelemetryStore(config.telemetry_path).read_commentary_audit() == [], (
        "a kind-illegal edge must write no attribution rows at all — the commit is "
        "foreclosed, so the write-ahead intent row is itself the defect"
    )
    assert GraphStore(config.db_path).get_edges() == []


def test_preflight_declares_a_disposition_for_every_store_code() -> None:
    """The pre-flight must account for the whole failure class, not an instance of it.

    This is the drift guard that makes §4.3's lesson mechanical. Phase 3 handled
    `missing_target` and silently left `kind_constraint_violation` to repeat forever;
    nothing failed, because nothing was watching the class. Now a store code with no
    declared disposition reds this test, forcing the author to either catch it or
    write down why not.
    """
    from mitos.errors import STORE_FAILURE_CODES
    from mitos.sync import _PREFLIGHT_DISPOSITIONS

    assert set(_PREFLIGHT_DISPOSITIONS) == set(STORE_FAILURE_CODES), (
        "every store failure code needs a pre-flight disposition — declare it caught, "
        "or record the reason it is deliberately not pre-flighted"
    )
    for code, disposition in _PREFLIGHT_DISPOSITIONS.items():
        assert disposition and isinstance(disposition, str), code


@patch("google.genai.Client")
def test_a_legal_edge_to_a_lineage_slug_still_reconciles(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """The pre-flight refuses only when EVERY candidate kind is illegal.

    A slug can name a whole same-slug supersession lineage (MI-13), so a naive
    "first candidate wins" probe would refuse a reconcile the commit would have
    accepted — silently dropping a legitimate repair, which is strictly worse than
    the extra audit row the pre-flight exists to prevent.
    """
    from mitos.telemetry import TelemetryStore

    config, manager, tmpdir = sync_env
    config.env["GEMINI_API_KEY"] = "mock_key"
    store = GraphStore(config.db_path)
    _seed_active_decision(store, "cited-decision", "A cited axiom.")
    _seed_committed_buffer(config, manager)

    _edit_buffer(config, "**Scope:** alpha",
                 "**Scope:** alpha\n**Cites:** cited-decision")
    manager.perform_sync(auto_accept=True)

    assert TelemetryStore(config.telemetry_path).read_commentary_audit(), (
        "a legal edge must still reconcile and still be attributed"
    )
    assert GraphStore(config.db_path).get_edges(), "the legal edge must have committed"


@patch("google.genai.Client")
def test_the_audit_row_records_full_edge_state_not_a_delta(
    mock_client: MagicMock, sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """`prior_values["edges"]` must be what the graph HELD, not merely what was removed.

    The documented read rule is "an intent row over a graph matching its `new_values`
    means applied". With deltas, a node that RETAINS an edge has `prior: []` — a
    positive false claim — and the rule then mismatches and reports a
    successfully-applied reconcile as not applied, inverting the P8 verdict.
    """
    from mitos.telemetry import TelemetryStore

    config, manager, tmpdir = sync_env
    config.env["GEMINI_API_KEY"] = "mock_key"
    store = GraphStore(config.db_path)
    _seed_active_decision(store, "kept-target", "The retained axiom.")
    _seed_active_decision(store, "added-target", "The newly cited axiom.")
    _seed_committed_buffer(config, manager, amends="kept-target")

    _edit_buffer(config, "**Amends:** kept-target",
                 "**Amends:** kept-target\n**Cites:** added-target")
    manager.perform_sync(auto_accept=True)

    rows = [r for r in TelemetryStore(config.telemetry_path).read_commentary_audit()
            if r["outcome"] is None]
    assert len(rows) == 1, rows
    assert rows[0]["prior_values"]["edges"] == ["amends:kept-target"], (
        "the retained edge must appear in the prior STATE"
    )
    assert rows[0]["new_values"]["edges"] == ["amends:kept-target", "cites:added-target"]
