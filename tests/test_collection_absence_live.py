"""Live I10 rows: absence is honest on a read, and durable across a write.

The offline fake (``tests/test_vector_store.py``) can only prove *"mitos dispatched
no create"* — it never observes a server. The claim the vision actually makes is
stronger and different: **the instance's collection listing is byte-identical
afterwards.** Only a real Qdrant can answer that, so this module exists to answer it,
and F2's split is the reason it is a separate file rather than another assertion in
the offline one.

The read half (1b) is here:

* a semantic read (``query``) against a collection that does not exist reports its
  absence and leaves ``GET /collections`` unchanged;
* a scroll (``mitos status``' read path, and ``reconcile``'s pre-write diff) does the
  same;
* constructing a store touches nothing.

And the write half (1c), which is where absence becomes *durable*:

* a **covering** write still creates — the fresh-project carve-out, unchanged;
* a **non-covering** write does not, and N consecutive records leave the listing
  byte-identical with N rows queued;
* ``rebuild`` → ``sync`` (and its ``--embed-only`` twin) creates the collection holding
  the full active set, which is what makes ``rebuild``'s own printed next step true;
* ``rebuild`` → ``record`` with no sync creates nothing — the marker has exactly one
  consumer;
* ``import`` into a populated graph creates nothing and leaves every row queued;
* ``reconcile`` heals the whole surface in one pass.

Discipline: every collection name here carries the ``mitos-tmp`` prefix so
``conftest``'s ``sweep_leaked_qdrant_collections`` can reclaim it, and each row
deletes its own anyway. No Gemini/Anthropic spend — the vectors are hand-made — but
the module still consults ``live_tests_disabled()`` because it needs a reachable
service, and ``LIVE_MODULES`` registration keeps it inside the live-tier coverage
floor rather than silently outside it.

``-n auto`` is forbidden for this module: the listing assertions are instance-global and
``sweep_leaked_qdrant_collections`` is session-scoped, so under xdist one worker's
teardown deletes a collection another is still asserting against — and the failure mode
is a **skip**, not a red, so a green parallel run proves nothing.
"""

import os
import uuid
from typing import Any, List, Set

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


def test_a_covering_write_still_creates_the_absent_collection(
    absent_collection: str,
) -> None:
    """Creation did not go away — it narrowed. A covering write behaves exactly as before.

    1b moved *where* creation happens; 1c narrows *which writes* may do it. This is the
    surviving half of the row 1b left for this phase to argue with: a write that declares
    it covers the active set creates and lands, unchanged.
    """
    assert absent_collection not in _listing()

    store = QdrantVectorStore(QDRANT_URL, absent_collection)
    store.upsert(
        "a" * 64, _vector(), {"slug": "live-absence-row", "scope": []}, may_create=True
    )

    assert absent_collection in _listing()
    # And the created collection actually holds the point — the retry landed, it
    # did not merely stop erroring.
    assert store.list_point_ids() == {
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    }


def test_a_NON_covering_write_leaves_the_listing_byte_identical(
    absent_collection: str,
) -> None:
    """The inverted half — and the assertion no offline fake can make.

    The offline twin proves mitos dispatched no create-shaped request. This proves the
    instance is genuinely unchanged, which is the claim every downstream net keys on: the
    record notice, ``check``'s fail-closed gate, and the stale-index probe all read
    *absence*, and one write must not be able to spend it.
    """
    before = _listing()

    store = QdrantVectorStore(QDRANT_URL, absent_collection)
    with pytest.raises(CollectionMissingError) as excinfo:
        store.upsert(
            "b" * 64, _vector(), {"slug": "live-refusal-row", "scope": []},
            may_create=False,
        )

    assert excinfo.value.collection == absent_collection
    assert "reconcile" in str(excinfo.value)
    assert _listing() == before
    assert absent_collection not in _listing()


SENTINEL = "<!-- BEGIN ENTRIES — newest first -->"


def _corpus(*slug_axiom_pairs) -> str:
    """Authors a decisions.md stream, newest-first under the sentinel."""
    blocks = [
        f"### {slug}\n\n**Decided:** {axiom}\n**Rejected:** n/a"
        for slug, axiom in slug_axiom_pairs
    ]
    return SENTINEL + "\n\n" + "\n\n".join(blocks) + "\n"


def _live_workspace(tmp_path, collection: str, *slug_axiom_pairs):
    """Builds a real workspace whose graph and corpus agree, bound to ``collection``.

    Returns ``(config, node_ids)``. The corpus is committed through parse→commit (no LLM
    anywhere on the path), and the outbox is emptied afterwards so each row starts from a
    state it chose rather than one the commits happened to leave.
    """
    from mitos.config import MitosConfig
    from mitos.parser import parse_entry_stream
    from mitos.store import GraphStore

    config = MitosConfig(str(tmp_path))
    os.makedirs(config.mitos_dir, exist_ok=True)
    config.qdrant_url = QDRANT_URL
    config.qdrant_collection = collection

    text = _corpus(*slug_axiom_pairs)
    with open(config.decisions_file, "w", encoding="utf-8") as handle:
        handle.write(text)

    store = GraphStore(config.db_path)
    node_ids = []
    failures: List[Any] = []
    entries = parse_entry_stream(text, "decision", failures=failures)
    assert not failures, failures
    # Oldest-first, so any citation's target commits before the citer.
    for entry in reversed(entries):
        node_ids.append(store.commit_parsed_entry(entry).node_id)
    for row in store.get_pending_embeddings():
        store.remove_pending_embedding(row["node_id"])
    return config, node_ids


def _manager_with_faked_embeddings(config, collection: str):
    """A real sync manager on a real Qdrant, with the embedding call faked.

    The vector store and the graph are real — that is where every claim in this module
    lives. Only the embedding provider is faked, so no row spends Gemini quota.
    """
    from unittest.mock import MagicMock

    from mitos.sync import MitosSyncManager

    manager = MitosSyncManager(config)
    manager.embed_provider = MagicMock()
    manager.embed_provider.get_embedding = MagicMock(return_value=_vector())
    manager.vector_store = QdrantVectorStore(QDRANT_URL, collection)
    return manager


def test_n_consecutive_records_leave_the_listing_byte_identical(
    absent_collection: str, tmp_path
) -> None:
    """The durable-absence pair, end to end: absence survives N writes, not one.

    This is the hazard in its exact shape. Before this phase the first ``record`` into a
    swept project created the collection holding **one point out of N**, so from the
    *second* record onward the conflict gather succeeded against a near-empty index,
    nothing cleared the 0.80 floor, and the pause rendered clean — *"couldn't check"*
    became *"checked, clean"*, which is the precise inversion the record-write-pause
    vision shipped its whole degradation contract to prevent.

    So the load-bearing assertion is about the **second** record, and about the outbox
    still holding every row afterwards: the signal is intact and the work is queued.
    """
    from mitos.parser import ParsedEntry

    config, _ = _live_workspace(
        tmp_path, absent_collection, ("existing", "An axiom already in the graph.")
    )
    manager = _manager_with_faked_embeddings(config, absent_collection)
    before = _listing()

    for index in range(3):
        entry = ParsedEntry("decision", f"newcomer-{index}", 1, 5)
        entry.axiom = f"Newly recorded axiom {index}."
        entry.rejected_paths = "None."
        delta = manager.store.commit_parsed_entry(entry)
        assert manager._best_effort_embed(delta, entry) is None

    assert _listing() == before
    assert absent_collection not in _listing()
    # Nothing was lost: every newcomer's commit-time outbox row survives the deferral.
    assert len(manager.store.get_pending_embeddings()) == 3


def test_a_fresh_projects_first_record_does_create_live(
    absent_collection: str, tmp_path
) -> None:
    """Success criterion 2 against a real instance: the carve-out is not theoretical.

    An empty graph's first record IS the whole active set, so there is no partial index to
    fear and nothing for absence to protect. The row matters because the narrowing is easy
    to over-apply: a gate that also refused here would leave every genuinely new project
    with no index until someone ran ``reconcile``, for no benefit at all.
    """
    from mitos.parser import ParsedEntry
    from mitos.vector_store import hash_to_uuid

    config, _ = _live_workspace(tmp_path, absent_collection)
    manager = _manager_with_faked_embeddings(config, absent_collection)
    assert absent_collection not in _listing()

    entry = ParsedEntry("decision", "first", 1, 5)
    entry.axiom = "The first axiom."
    entry.rejected_paths = "None."
    delta = manager.store.commit_parsed_entry(entry)
    manager._best_effort_embed(delta, entry)

    assert absent_collection in _listing()
    assert manager.vector_store.list_point_ids() == {hash_to_uuid(delta.node_id)}


def _rebuild_and_swap(config) -> None:
    """Runs what ``mitos rebuild`` runs: rebuild-aside, gate, atomic swap.

    Driven through the same two functions ``cmd_rebuild`` calls rather than through the
    verb, because the verb's own wiring (prompting, ``--allow-drops``, the JSON envelope)
    is already pinned offline in ``tests/test_rebuild.py``. What is live-only here is what
    happens to Qdrant afterwards.
    """
    from mitos.cutover import default_aside_db_path, perform_swap, rebuild_and_gate

    result = rebuild_and_gate(
        config, aside_db_path=default_aside_db_path(config), strict=False, quiet=True
    )
    assert not result.residual_casualties
    perform_swap(config, result.aside_db_path, timestamp="20260730-000000")


@pytest.mark.parametrize("entry_point", ["embed_only_verb", "commit_path_drain"])
def test_rebuild_then_sync_creates_the_collection_with_the_full_active_set(
    absent_collection: str, tmp_path, entry_point: str
) -> None:
    """Success criteria 3 and 4: the printed next-step is true, on both drain entry points.

    ``rebuild`` and ``cutover`` issue no upsert of their own — they bound the outbox to
    exactly the active set — so the covering write is the **next** command's drain. That
    is what ``mitos rebuild`` already promises in print (*"Re-embed so semantic
    surface/query reflect the rebuild: mitos sync"*), and a rule that could only read
    intent from the running call site would make that shipped line false.

    Both entry points matter because ``--embed-only`` is a drain with **no buffer at
    all**: its covering-ness is entirely a property of outbox state a prior command
    established, so the operator who knows the tool better must not be the one punished.
    ``embed_only_verb`` drives the real verb (``cli.cmd_sync(embed_only=True)``);
    ``commit_path_drain`` drives the method ``perform_sync``'s step 6 calls. The full
    ``perform_sync`` is deliberately not driven here — it short-circuits on its own
    ``GEMINI_API_KEY`` env guard before reaching any drain, so a row built on it would
    prove the guard rather than the drain, and the drain it eventually reaches is exactly
    the method this parametrization already calls.
    """
    from unittest.mock import patch

    from mitos import cli
    from mitos.store import GraphStore
    from mitos.vector_store import hash_to_uuid

    config, _ = _live_workspace(
        tmp_path,
        absent_collection,
        ("beta", "Beta axiom."),
        ("alpha", "Alpha axiom."),
    )
    assert absent_collection not in _listing()

    _rebuild_and_swap(config)

    # The rebuild left the outbox bounded to the active set and said so, and it created
    # nothing on the instance.
    post_rebuild = GraphStore(config.db_path)
    active_ids = post_rebuild.get_active_node_ids()
    assert len(active_ids) == 2
    seed = post_rebuild.embedding_seed()
    assert seed is not None and seed["established_by"] == "rebuild"
    assert absent_collection not in _listing()

    manager = _manager_with_faked_embeddings(config, absent_collection)
    if entry_point == "embed_only_verb":
        with patch("mitos.cli.MitosSyncManager", return_value=manager):
            cli.cmd_sync(config, embed_only=True)
    else:
        manager.drain_pending_embeddings()

    assert absent_collection in _listing()
    assert manager.vector_store.list_point_ids() == {
        hash_to_uuid(nid) for nid in active_ids
    }
    # Coverage delivered: the outbox is empty and the claim is spent.
    assert manager.store.get_pending_embeddings() == []
    assert manager.store.embedding_seed() is None


def test_rebuild_then_record_with_no_sync_creates_nothing(
    absent_collection: str, tmp_path
) -> None:
    """Gotcha 1: the marker has exactly ONE consumer, and this is the row that proves it.

    If a ``record`` consulted the marker after a ``rebuild``, the first record into a
    swept project would mint a one-point collection — the precise hazard this phase exists
    to close, reintroduced through the phase's own mechanism. The marker authorizes a
    *drain*; nothing else may read it.
    """
    from mitos.parser import ParsedEntry
    from mitos.store import GraphStore

    config, _ = _live_workspace(
        tmp_path,
        absent_collection,
        ("beta", "Beta axiom."),
        ("alpha", "Alpha axiom."),
    )
    _rebuild_and_swap(config)
    assert GraphStore(config.db_path).embedding_seed() is not None
    before = _listing()

    manager = _manager_with_faked_embeddings(config, absent_collection)
    entry = ParsedEntry("decision", "newcomer", 1, 5)
    entry.axiom = "A newly recorded axiom."
    entry.rejected_paths = "None."
    delta = manager.store.commit_parsed_entry(entry)
    manager._best_effort_embed(delta, entry)

    assert _listing() == before
    assert absent_collection not in _listing()
    # The marker is untouched — the inline path neither reads nor spends it, so the
    # operator's next `sync` still heals.
    assert manager.store.embedding_seed() is not None


def test_import_into_a_populated_graph_creates_nothing_live(
    absent_collection: str, tmp_path
) -> None:
    """Success criterion 7: one line, no collection, N rows still queued.

    ``import`` is a separate implementation sharing the ``_best_effort_embed`` name, so a
    symbol grep returns both and reads as one. Leaving it out would let ``mitos import``
    mint exactly the partial collection the clause exists to prevent, one entry deep.
    """
    from unittest.mock import MagicMock

    from mitos.importer import MitosProseImporter
    from mitos.parser import ParsedEntry

    config, _ = _live_workspace(
        tmp_path, absent_collection, ("existing", "An axiom already in the graph.")
    )
    before = _listing()

    importer = MitosProseImporter(config)
    importer.embed_provider = MagicMock()
    importer.embed_provider.get_embedding = MagicMock(return_value=_vector())
    importer.vector_store = QdrantVectorStore(QDRANT_URL, absent_collection)

    for index in range(3):
        entry = ParsedEntry("decision", f"imported-{index}", 1, 5)
        entry.axiom = f"Imported axiom {index}."
        entry.rejected_paths = "None."
        delta = importer.store.commit_parsed_entry(entry)
        importer._best_effort_embed(delta, entry)

    assert _listing() == before
    assert absent_collection not in _listing()
    # Every imported node is still queued — the importer never dequeues, so its rows
    # survive by default; what the added arm buys is the notice, not the durability.
    assert len(importer.store.get_pending_embeddings()) == 3


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
