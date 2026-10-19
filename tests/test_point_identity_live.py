"""Phase 2a, live against Qdrant: a hit's point id is ``hash_to_uuid(node_id)``.

The offline rows pin the client against canned bodies and the resolver against a
real graph. What only a real server can show is that the two meet: a collection
written **through today's write path** (``MitosSyncManager._best_effort_embed``,
reached by ``record_decision_entry``), shaped exactly like one written before this
vision, answers ``query`` with ids that ``resolve_point_ids`` maps back to the nodes
written — and answers ``with_payload=False`` with ids and scores alone.

Needs Qdrant, not Gemini: the embed provider is a deterministic keyless fake whose
vectors are mutually orthogonal, so no record pauses on a neighbour. Collections are
named ``mitos-tmp-pointid-…`` and deleted per test (and swept by conftest).
"""

import os
import uuid
from typing import Iterator, List, Tuple

import pytest
import requests

from mitos.cli import cmd_init
from mitos.config import MitosConfig
from mitos.models import EMBEDDING_DIM
from mitos.sync import MitosSyncManager
from mitos.vector_store import QdrantVectorStore, hash_to_uuid

from live_helpers import live_tests_disabled

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:7333")


def _qdrant_reachable() -> bool:
    """Best-effort probe: this tier needs a service, not a key."""
    try:
        return requests.get(f"{QDRANT_URL}/collections", timeout=2).ok
    except Exception:
        return False


# The brake and an unreachable Qdrant are kept apart, as in
# test_collection_absence_live.py: only the second is an environmental degradation
# the live floor counts, so only it carries the floor's marker.
_BRAKE_ON = live_tests_disabled()
HAS_QDRANT = (not _BRAKE_ON) and _qdrant_reachable()

if _BRAKE_ON:
    _SKIP_REASON = (
        "MITOS_NO_LIVE_TESTS is set — the live tier is switched off, so these rows "
        "did not run and nothing pretended to check."
    )
else:
    _SKIP_REASON = (
        f"Qdrant is not reachable at {QDRANT_URL} — the point-identity rows need a "
        "real instance. Environmental, not a code defect; start it with "
        "`docker compose up -d` in the mitos repo."
    )

pytestmark = pytest.mark.skipif(not HAS_QDRANT, reason=_SKIP_REASON)

_DECISIONS = [
    ("pointid-sqlite-store", "Use SQLite for the graph store."),
    ("pointid-markdown-corpus", "Keep the corpus in plain markdown."),
    ("pointid-qdrant-index", "Index decision embeddings in Qdrant."),
]


class _OrthogonalEmbed:
    """One basis vector per distinct text: deterministic, keyless, never similar."""

    def __init__(self) -> None:
        self._axes: dict = {}

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        axis = self._axes.setdefault(text, len(self._axes) % EMBEDDING_DIM)
        vector = [0.0] * EMBEDDING_DIM
        vector[axis] = 1.0
        return vector


@pytest.fixture
def written(tmp_path, monkeypatch) -> Iterator[Tuple[MitosSyncManager, QdrantVectorStore,
                                                     _OrthogonalEmbed]]:
    """Records three decisions through the shipped write path into a throwaway collection."""
    monkeypatch.setenv("QDRANT_URL", QDRANT_URL)
    collection = f"mitos-tmp-pointid-{uuid.uuid4().hex[:12]}"
    config = MitosConfig(str(tmp_path / "ws"))
    cmd_init(config)
    manager = MitosSyncManager(config)
    embed = _OrthogonalEmbed()
    vstore = QdrantVectorStore(QDRANT_URL, collection)
    manager.embed_provider = embed  # type: ignore[assignment]
    manager.vector_store = vstore  # type: ignore[assignment]
    try:
        for slug, axiom in _DECISIONS:
            receipt = manager.record_decision_entry(axiom, "Rejected the obvious alternative.",
                                                    ["core"], slug=slug)
            assert receipt["status"] == "created", receipt
        assert manager.store.get_pending_embeddings() == [], "the write path did not embed"
        yield manager, vstore, embed
    finally:
        requests.delete(f"{QDRANT_URL}/collections/{collection}", timeout=5)


def _node_ids(manager: MitosSyncManager) -> dict:
    return {slug: manager.store.resolve_slug(slug)[0] for slug, _ in _DECISIONS}


def test_query_returns_hash_to_uuid_of_the_node_written(written) -> None:
    manager, vstore, embed = written
    ids = _node_ids(manager)
    # Any direction will do: the collection holds three points and the limit is wider.
    hits = vstore.query([1.0] + [0.0] * (EMBEDDING_DIM - 1), limit=10)

    assert {h["point_id"] for h in hits} == {hash_to_uuid(i) for i in ids.values()}
    for hit in hits:
        assert hit["point_id"] == hash_to_uuid(ids[hit["slug"]])


def test_resolve_point_ids_maps_the_servers_ids_to_the_written_nodes(written) -> None:
    manager, vstore, _embed = written
    ids = _node_ids(manager)
    hits = vstore.query([1.0] + [0.0] * (EMBEDDING_DIM - 1), limit=10)

    result = manager.store.resolve_point_ids([h["point_id"] for h in hits])

    assert result.orphans == [] and result.ambiguous == {}
    assert {n["id"] for n in result.nodes.values()} == set(ids.values())
    for hit in hits:
        assert result.nodes[hit["point_id"]]["slug"] == hit["slug"]


def test_with_payload_false_returns_only_ids_and_scores_from_the_server(written) -> None:
    manager, vstore, _embed = written
    ids = _node_ids(manager)

    hits = vstore.query([1.0] + [0.0] * (EMBEDDING_DIM - 1), limit=10, with_payload=False)

    assert hits and all(set(h) == {"point_id", "score"} for h in hits)
    assert {h["point_id"] for h in hits} == {hash_to_uuid(i) for i in ids.values()}
