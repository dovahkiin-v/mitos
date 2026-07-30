"""Live I10 rows: a read against an absent collection leaves the instance untouched.

The offline fake (``tests/test_vector_store.py``) can only prove *"mitos dispatched
no create"* — it never observes a server. The claim the vision actually makes is
stronger and different: **the instance's collection listing is byte-identical
afterwards.** Only a real Qdrant can answer that, so this module exists to answer it,
and F2's split is the reason it is a separate file rather than another assertion in
the offline one.

Three rows, all against the live instance on ``:7333``:

* a semantic read (``query``) against a collection that does not exist reports its
  absence and leaves ``GET /collections`` unchanged;
* a scroll (``mitos status``' read path, and ``reconcile``'s pre-write diff) does the
  same;
* a **write** into the same absent collection still creates it — 1b moved *where*
  creation happens, never *whether*, and pinning that is what forces 1c to invert a
  test rather than notice a comment.

Discipline: every collection name here carries the ``mitos-tmp`` prefix so
``conftest``'s ``sweep_leaked_qdrant_collections`` can reclaim it, and each row
deletes its own anyway. No Gemini/Anthropic spend — the vectors are hand-made — but
the module still consults ``live_tests_disabled()`` because it needs a reachable
service, and ``LIVE_MODULES`` registration keeps it inside the live-tier coverage
floor rather than silently outside it.
"""

import os
import uuid
from typing import List, Set

import pytest
import requests

from mitos.errors import CollectionMissingError
from mitos.models import EMBEDDING_DIM
from mitos.vector_store import QdrantVectorStore, scroll_point_ids

from live_helpers import live_tests_disabled

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:7333")


def _qdrant_reachable() -> bool:
    """Best-effort probe: this tier needs a service, not a key."""
    try:
        return requests.get(f"{QDRANT_URL}/collections", timeout=2).ok
    except Exception:
        return False


#: The two skip causes are kept apart on purpose, and only one carries the
#: live-floor's ``not a code defect`` marker. The brake being on means the tier was
#: switched off *wholesale* — nothing pretended to check, which is an honest state
#: the floor must stay silent about. An unreachable Qdrant while the tier is live is
#: the environmental degradation the floor exists to count. Collapsing the two makes
#: every braked inner-loop run report a spurious coverage hole.
_BRAKE_ON = live_tests_disabled()
HAS_QDRANT = (not _BRAKE_ON) and _qdrant_reachable()

if _BRAKE_ON:
    _SKIP_REASON = (
        "MITOS_NO_LIVE_TESTS is set — the live tier is switched off, so these rows "
        "did not run and nothing pretended to check."
    )
else:
    _SKIP_REASON = (
        f"Qdrant is not reachable at {QDRANT_URL} — the collection-listing "
        "invariance rows need a real instance. Environmental, not a code defect; "
        "start it with `docker compose up -d` in the mitos repo."
    )

pytestmark = pytest.mark.skipif(not HAS_QDRANT, reason=_SKIP_REASON)


def _listing() -> Set[str]:
    """The instance's current collection names — the thing that must not change."""
    resp = requests.get(f"{QDRANT_URL}/collections", timeout=5)
    resp.raise_for_status()
    return {c["name"] for c in resp.json()["result"]["collections"]}


def _absent_name() -> str:
    """A collection name that does not exist, prefixed for the conftest sweep."""
    return f"mitos-tmp-absent-{uuid.uuid4().hex[:12]}"


def _vector() -> List[float]:
    """A hand-made unit-ish vector of the right width — no embedding spend."""
    return [0.01] * EMBEDDING_DIM


@pytest.fixture
def absent_collection() -> str:
    """Yields a name that is absent going in, and is gone again coming out."""
    name = _absent_name()
    assert name not in _listing()
    yield name
    requests.delete(f"{QDRANT_URL}/collections/{name}", timeout=5)


def test_a_semantic_read_leaves_the_collection_listing_byte_identical(
    absent_collection: str,
) -> None:
    """T4: reading an absent collection reports it and creates nothing on the server.

    This is the assertion the offline row structurally cannot make. Before 1b the
    store's constructor would have minted the collection here, and the very first
    ``surface_decisions`` after the 1d migration would have handed every swept
    project a clean, empty, permanently-green index over a populated corpus.
    """
    before = _listing()

    store = QdrantVectorStore(QDRANT_URL, absent_collection)
    with pytest.raises(CollectionMissingError) as excinfo:
        store.query(_vector(), limit=5)

    assert excinfo.value.collection == absent_collection
    assert _listing() == before
    assert absent_collection not in _listing()


def test_a_scroll_leaves_the_collection_listing_byte_identical(
    absent_collection: str,
) -> None:
    """The same for the no-create read path ``status`` and ``reconcile`` both use."""
    before = _listing()

    with pytest.raises(CollectionMissingError):
        scroll_point_ids(QDRANT_URL, absent_collection)

    assert _listing() == before


def test_constructing_a_store_leaves_the_listing_byte_identical(
    absent_collection: str,
) -> None:
    """W7 end-to-end: construction alone touches nothing on a real instance."""
    before = _listing()

    QdrantVectorStore(QDRANT_URL, absent_collection)

    assert _listing() == before


def test_a_write_still_creates_the_absent_collection(absent_collection: str) -> None:
    """Success criterion 8: 1b changed WHERE creation happens, never WHETHER.

    The pin 1c must consciously invert: after 1c only a write covering the
    workspace's active set may take this branch, and this row is where that
    narrowing has to argue with an existing assertion.
    """
    assert absent_collection not in _listing()

    store = QdrantVectorStore(QDRANT_URL, absent_collection)
    store.upsert("a" * 64, _vector(), {"slug": "live-absence-row", "scope": []})

    assert absent_collection in _listing()
    # And the created collection actually holds the point — the retry landed, it
    # did not merely stop erroring.
    assert store.list_point_ids() == {
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    }


def test_reconcile_heals_an_absent_collection_end_to_end(
    absent_collection: str, tmp_path
) -> None:
    """T5 against a real instance: the heal creates the collection and repopulates it.

    The whole chain in one row — the pre-write diff reads the absent collection as
    empty, every active node is enqueued, the drain's first upsert creates, and the
    active set lands. Before 1b the diff's 404 raised and ``mitos reconcile``
    reported "Qdrant or embedding provider down" over a healthy Qdrant, so the heal
    this vision prescribes failed on the exact state the collection migration
    creates.

    The embedding provider is faked (hand-made vectors, no Gemini spend); the vector
    store and the graph are real, which is where the claim lives.
    """
    from unittest.mock import MagicMock

    from mitos.config import MitosConfig
    from mitos.parser import ParsedEntry
    from mitos.store import GraphStore
    from mitos.sync import MitosSyncManager
    from mitos.vector_store import hash_to_uuid

    config = MitosConfig(str(tmp_path))
    os.makedirs(config.mitos_dir, exist_ok=True)
    store = GraphStore(config.db_path)
    node_ids = []
    for index in range(3):
        entry = ParsedEntry("decision", f"live-absence-{index}", 1, 5)
        entry.axiom = f"Live absence axiom {index}."
        entry.rejected_paths = "None."
        node_ids.append(store.commit_parsed_entry(entry).node_id)
    for row in store.get_pending_embeddings():
        store.remove_pending_embedding(row["node_id"])

    manager = MitosSyncManager(config)
    manager.store = store
    manager.embed_provider = MagicMock()
    manager.embed_provider.get_embedding = MagicMock(return_value=_vector())
    manager.vector_store = QdrantVectorStore(QDRANT_URL, absent_collection)

    result = manager.reconcile_embeddings()

    assert result == {"active": 3, "present": 0, "enqueued": 3}
    assert absent_collection in _listing()
    assert manager.vector_store.list_point_ids() == {
        hash_to_uuid(nid) for nid in node_ids
    }
    # Idempotent: the second pass finds everything present and enqueues nothing.
    assert manager.reconcile_embeddings() == {"active": 3, "present": 3, "enqueued": 0}
