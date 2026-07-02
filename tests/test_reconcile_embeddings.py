"""Tests for `mitos reconcile` — self-healing a Qdrant/graph vector mismatch.

Reconcile diffs the graph's ACTIVE node set against Qdrant's actual point ids,
enqueues the missing active nodes, and drains (reusing the Fix 1 drain loop). It
is the one-command heal for a bare Qdrant wipe (`curl -X DELETE` of the
collection) that leaves the graph populated, Qdrant empty, and the outbox empty —
a state `mitos sync` cannot recover because there is nothing queued to drain.

Dead/superseded nodes are intentionally NOT re-embedded (retrieval filters them
via `has_id` and never returns them — the `cutover-bounds-embedding-seed-to-active`
decision), so reconcile targets the active surface only.
"""

import tempfile
import os
import shutil
import pytest
import requests
from typing import Tuple
from unittest.mock import MagicMock, patch

from mitos.config import MitosConfig
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager
from mitos.parser import ParsedEntry
from mitos.vector_store import QdrantVectorStore, hash_to_uuid
from mitos.errors import VectorStoreError


@pytest.fixture
def sync_env() -> Tuple[MitosConfig, MitosSyncManager, str]:
    """Sets up a hermetic sync environment mapped to a temp workspace."""
    tmpdir = tempfile.mkdtemp()

    config = MitosConfig(tmpdir)
    config.db_path = os.path.join(tmpdir, ".mitos", "graph.sqlite")
    config.decisions_file = os.path.join(tmpdir, "decisions.md")
    config.archive_dir = os.path.join(tmpdir, "decisions", "archive")

    os.makedirs(os.path.join(tmpdir, ".mitos"), exist_ok=True)
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write(
            "# Decisions\n"
            "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->\n"
        )

    manager = MitosSyncManager(config)
    yield config, manager, tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


def _mock_embed_stack(manager: MitosSyncManager, present_uuids: set) -> None:
    """Wires MagicMock embed/vector deps onto a manager for a hermetic reconcile.

    Args:
        manager: The sync manager to instrument.
        present_uuids: The point-id UUID set `list_point_ids` should report.
    """
    manager.embed_provider = MagicMock()
    manager.embed_provider.get_embedding = MagicMock(return_value=[0.1, 0.2, 0.3])
    manager.vector_store = MagicMock()
    manager.vector_store.upsert = MagicMock()
    manager.vector_store.list_point_ids = MagicMock(return_value=present_uuids)


def _commit(store: GraphStore, slug: str, axiom: str, supersedes=None) -> str:
    """Commits a minimal decision node and returns its node id."""
    e = ParsedEntry("decision", slug, 1, 5)
    e.axiom = axiom
    e.rejected_paths = "None."
    if supersedes:
        e.supersedes = supersedes
    return store.commit_parsed_entry(e).node_id


def _clear_outbox(store: GraphStore) -> None:
    """Empties pending_embeddings — simulates the post-Qdrant-wipe empty outbox."""
    for row in store.get_pending_embeddings():
        store.remove_pending_embedding(row["node_id"])
    assert store.get_pending_embeddings() == []


def test_reconcile_heals_partial_index(sync_env: Tuple[MitosConfig, MitosSyncManager, str]) -> None:
    """Reconcile re-embeds exactly the active nodes missing from Qdrant, not the whole corpus."""
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)

    ids = [_commit(store, f"node-{i:02d}", f"Axiom {i}") for i in range(5)]
    _clear_outbox(store)  # bare-wipe scenario: outbox empty, so `sync` would be a no-op

    # Qdrant already holds the first two nodes; the last three are missing.
    present = {hash_to_uuid(ids[0]), hash_to_uuid(ids[1])}
    missing = set(ids[2:])
    _mock_embed_stack(manager, present)

    result = manager.reconcile_embeddings()

    assert result == {"active": 5, "present": 2, "enqueued": 3}
    # Upsert was called exactly for the missing set — not the whole corpus.
    assert manager.vector_store.upsert.call_count == 3
    upserted = {call.args[0] for call in manager.vector_store.upsert.call_args_list}
    assert upserted == missing
    # The heal fully drained what it queued.
    assert store.get_pending_embeddings() == []


def test_reconcile_targets_active_surface_only(sync_env: Tuple[MitosConfig, MitosSyncManager, str]) -> None:
    """The ADR invariant: superseded nodes are never re-embedded, even when absent from Qdrant."""
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)

    keep_id = _commit(store, "keep-me", "Live axiom.")
    old_id = _commit(store, "old-one", "Doomed axiom.")
    new_id = _commit(store, "new-one", "Replacement axiom.", supersedes=["old-one"])
    _clear_outbox(store)

    # Sanity: old-one is superseded, the other two are active.
    active_ids = store.get_active_node_ids()
    assert active_ids == {keep_id, new_id}
    assert old_id not in active_ids

    # Qdrant fully wiped — everything is "missing", but only the active set heals.
    _mock_embed_stack(manager, present_uuids=set())

    result = manager.reconcile_embeddings()

    assert result["active"] == 2
    assert result["enqueued"] == 2
    upserted = {call.args[0] for call in manager.vector_store.upsert.call_args_list}
    assert upserted == {keep_id, new_id}
    assert old_id not in upserted  # superseded stays out of Qdrant


def test_reconcile_is_idempotent(sync_env: Tuple[MitosConfig, MitosSyncManager, str]) -> None:
    """A second reconcile enqueues nothing and calls upsert zero times."""
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)

    ids = [_commit(store, f"node-{i}", f"Axiom {i}") for i in range(3)]
    _clear_outbox(store)

    # Qdrant already holds every active node.
    _mock_embed_stack(manager, present_uuids={hash_to_uuid(i) for i in ids})

    result = manager.reconcile_embeddings()

    assert result == {"active": 3, "present": 3, "enqueued": 0}
    assert manager.vector_store.upsert.call_count == 0
    assert store.get_pending_embeddings() == []


def test_reconcile_degrades_cleanly_when_providers_down(sync_env: Tuple[MitosConfig, MitosSyncManager, str]) -> None:
    """With no embed/vector provider, reconcile returns zeros and leaves the graph untouched."""
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)
    _commit(store, "node-a", "Axiom A")
    _clear_outbox(store)

    # A degraded manager (Qdrant/Gemini down) has None providers.
    manager.embed_provider = None
    manager.vector_store = None

    result = manager.reconcile_embeddings()

    assert result == {"active": 0, "present": 0, "enqueued": 0}
    # Graph and outbox are untouched — no crash, no partial write.
    assert len(store.get_all_nodes()) == 1
    assert store.get_pending_embeddings() == []


def test_cmd_reconcile_returns_error_on_qdrant_unreachable(
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """cmd_reconcile surfaces a Qdrant-scroll outage as a clean non-zero exit, not a crash."""
    from mitos import cli

    config, _manager, tmpdir = sync_env

    fake_manager = MagicMock()
    fake_manager.reconcile_embeddings = MagicMock(
        side_effect=VectorStoreError("Qdrant scroll connection error")
    )
    with patch("mitos.cli.MitosSyncManager", return_value=fake_manager):
        rc = cli.cmd_reconcile(config, as_json=False)

    assert rc == 1


@patch("mitos.vector_store.requests.post")
@patch("mitos.vector_store.requests.get")
def test_list_point_ids_pagination(mock_get: MagicMock, mock_post: MagicMock) -> None:
    """list_point_ids walks every scroll page until next_page_offset is null."""
    # _ensure_collection (called from __init__) sees an existing collection.
    ensure_resp = MagicMock()
    ensure_resp.status_code = 200
    mock_get.return_value = ensure_resp

    page1 = MagicMock()
    page1.status_code = 200
    page1.json.return_value = {
        "result": {"points": [{"id": "uuid-1"}, {"id": "uuid-2"}], "next_page_offset": "cursor-2"}
    }
    page2 = MagicMock()
    page2.status_code = 200
    page2.json.return_value = {
        "result": {"points": [{"id": "uuid-3"}], "next_page_offset": None}
    }
    mock_post.side_effect = [page1, page2]

    vs = QdrantVectorStore("http://localhost:7333", "coll")
    ids = vs.list_point_ids(page_size=2)

    assert ids == {"uuid-1", "uuid-2", "uuid-3"}
    assert mock_post.call_count == 2
    # The second page request carries the offset returned by the first.
    assert mock_post.call_args_list[1].kwargs["json"].get("offset") == "cursor-2"


@patch("mitos.vector_store.requests.post")
@patch("mitos.vector_store.requests.get")
def test_list_point_ids_raises_on_qdrant_error(mock_get: MagicMock, mock_post: MagicMock) -> None:
    """A Qdrant connection failure during scroll surfaces as VectorStoreError."""
    ensure_resp = MagicMock()
    ensure_resp.status_code = 200
    mock_get.return_value = ensure_resp
    mock_post.side_effect = requests.RequestException("connection refused")

    vs = QdrantVectorStore("http://localhost:7333", "coll")
    with pytest.raises(VectorStoreError):
        vs.list_point_ids()
