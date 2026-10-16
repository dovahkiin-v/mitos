"""Tests for `mitos reconcile` — self-healing a Qdrant/graph vector mismatch.

Reconcile diffs the decision index's EXPECTED set — every decision, live or
retired, plus the active open questions (`GraphStore.get_decision_index_expected_ids`)
— against Qdrant's actual point ids, enqueues the missing nodes, and drains (handing
the drain the listing it already took). It is the one-command heal for a bare Qdrant
wipe (`curl -X DELETE` of the collection) that leaves the graph populated, Qdrant
empty, and the outbox empty — a state `mitos sync` cannot recover because there is
nothing queued to drain.

A retired decision is seeded like a live one: its vector lives exactly as long as its
node, and the read surfaces partition it by computed state (ADR
`graveyard-is-deliberate-every-decision-vector-lives-as-long-as-its-node`). Only an
open question that was superseded or corrected stays out.

The result's shipped keys (`active`, `present`, `enqueued`) keep their meaning; 3b
added `expected`, `expected_present`, `written`, `left_queued` and `drain_stop`, so
rows assert the shipped keys' values plus the new keys rather than one whole dict.
"""

import tempfile
import json
import os
import shutil
import pytest
import requests
from typing import Any, Dict, Tuple
from unittest.mock import MagicMock, patch

from mitos.config import MitosConfig
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager
from mitos.parser import ParsedEntry
from mitos.vector_store import QdrantVectorStore, hash_to_uuid
from mitos.errors import CollectionMissingError, VectorStoreError


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


_SHIPPED = ("active", "present", "enqueued")
_ADDED = ("expected", "expected_present", "written", "left_queued", "drain_stop")


def _shipped(result: Dict[str, Any]) -> Dict[str, Any]:
    """The three keys reconcile shipped before 3b, which keep their meaning."""
    return {key: result[key] for key in _SHIPPED}


def _added(result: Dict[str, Any]) -> Dict[str, Any]:
    """The five keys 3b added beside them."""
    return {key: result[key] for key in _ADDED}


def _clear_outbox(store: GraphStore) -> None:
    """Empties pending_embeddings — simulates the post-Qdrant-wipe empty outbox."""
    for row in store.get_pending_embeddings():
        store.remove_pending_embedding(row["node_id"])
    assert store.get_pending_embeddings() == []


def test_reconcile_heals_partial_index(sync_env: Tuple[MitosConfig, MitosSyncManager, str]) -> None:
    """Reconcile re-embeds exactly the nodes missing from Qdrant, not the whole corpus."""
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)

    ids = [_commit(store, f"node-{i:02d}", f"Axiom {i}") for i in range(5)]
    _clear_outbox(store)  # bare-wipe scenario: outbox empty, so `sync` would be a no-op

    # Qdrant already holds the first two nodes; the last three are missing.
    present = {hash_to_uuid(ids[0]), hash_to_uuid(ids[1])}
    missing = set(ids[2:])
    _mock_embed_stack(manager, present)

    result = manager.reconcile_embeddings()

    assert _shipped(result) == {"active": 5, "present": 2, "enqueued": 3}
    assert _added(result) == {"expected": 5, "expected_present": 2, "written": 3,
                              "left_queued": 0, "drain_stop": None}
    # Upsert was called exactly for the missing set — not the whole corpus.
    assert manager.vector_store.upsert.call_count == 3
    upserted = {call.args[0] for call in manager.vector_store.upsert.call_args_list}
    assert upserted == missing
    # The heal fully drained what it queued.
    assert store.get_pending_embeddings() == []


def test_reconcile_seeds_the_retired_decision_too(
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """Inverted in 3b: a superseded decision missing from Qdrant is embedded like a live one.

    Was ``test_reconcile_targets_active_surface_only``, which pinned the opposite (the
    superseded node never upserted) under ``cutover-bounds-embedding-seed-to-active``.
    The graveyard is now deliberate (ADR
    ``graveyard-is-deliberate-every-decision-vector-lives-as-long-as-its-node``): every
    decision's vector lives as long as its node, so reconcile diffs the expected set.
    """
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)

    keep_id = _commit(store, "keep-me", "Live axiom.")
    old_id = _commit(store, "old-one", "Doomed axiom.")
    new_id = _commit(store, "new-one", "Replacement axiom.", supersedes=["old-one"])
    _clear_outbox(store)

    # Sanity: old-one is superseded, the other two are active, and all three expected.
    active_ids = store.get_active_node_ids()
    assert active_ids == {keep_id, new_id}
    assert old_id not in active_ids
    assert store.get_decision_index_expected_ids() == {keep_id, old_id, new_id}

    # Qdrant fully wiped — everything is missing, and everything expected heals.
    _mock_embed_stack(manager, present_uuids=set())

    result = manager.reconcile_embeddings()

    assert result["active"] == 2
    assert result["expected"] == 3
    assert result["enqueued"] == 3
    upserted = {call.args[0] for call in manager.vector_store.upsert.call_args_list}
    assert upserted == {keep_id, old_id, new_id}


def test_reconcile_is_idempotent(sync_env: Tuple[MitosConfig, MitosSyncManager, str]) -> None:
    """A second reconcile enqueues nothing and calls upsert zero times."""
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)

    ids = [_commit(store, f"node-{i}", f"Axiom {i}") for i in range(3)]
    _clear_outbox(store)

    # Qdrant already holds every active node.
    _mock_embed_stack(manager, present_uuids={hash_to_uuid(i) for i in ids})

    result = manager.reconcile_embeddings()

    assert _shipped(result) == {"active": 3, "present": 3, "enqueued": 0}
    # Nothing missing, so no drain ran: `written` / `drain_stop` are 0 / None.
    assert _added(result) == {"expected": 3, "expected_present": 3, "written": 0,
                              "left_queued": 0, "drain_stop": None}
    assert manager.vector_store.upsert.call_count == 0
    assert store.get_pending_embeddings() == []


def test_reconcile_degrades_cleanly_when_providers_down(sync_env: Tuple[MitosConfig, MitosSyncManager, str]) -> None:
    """With no embed/vector provider, reconcile returns zeros and leaves the graph untouched."""
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)
    _commit(store, "node-a", "Axiom A")
    _clear_outbox(store)

    # A degraded manager has None providers. Since 1b the vector store constructs
    # without touching the network, so this state comes from a MISSING KEY (or a
    # provider that raised), never from Qdrant being down — a down Qdrant now
    # surfaces at the scroll instead, and its row is the cmd_reconcile one below.
    manager.embed_provider = None
    manager.vector_store = None

    result = manager.reconcile_embeddings()

    # The provider-down return keeps the one key set every path carries.
    assert _shipped(result) == {"active": 0, "present": 0, "enqueued": 0}
    assert _added(result) == {"expected": 0, "expected_present": 0, "written": 0,
                              "left_queued": 0, "drain_stop": None}
    # Graph and outbox are untouched — no crash, no partial write.
    assert len(store.get_all_nodes()) == 1
    assert store.get_pending_embeddings() == []


def test_cmd_reconcile_heals_an_absent_collection_and_exits_0(
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """T5: the verb heals rather than blaming Qdrant for a state Qdrant is fine with."""
    from mitos import cli

    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)
    _commit(store, "node-a", "Axiom A")
    _clear_outbox(store)

    _mock_embed_stack(manager, present_uuids=set())
    manager.vector_store.list_point_ids = MagicMock(
        side_effect=CollectionMissingError(
            "Qdrant collection 'mitos-x' does not exist", collection="mitos-x"
        )
    )

    with patch("mitos.cli.MitosSyncManager", return_value=manager):
        rc = cli.cmd_reconcile(config, as_json=True)

    assert rc == 0
    assert manager.vector_store.upsert.call_count == 1


def test_cmd_reconcile_now_actually_reaches_its_qdrant_down_arm(
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """T5's twin, and it is NEWLY satisfied rather than merely preserved.

    Before 1b an unreachable Qdrant made the store's constructor raise, so
    ``MitosSyncManager`` left ``vector_store = None``, ``reconcile_embeddings``
    returned zeros, and the verb exited **0** with a confusing double message. Now
    the store constructs, the scroll raises a connection ``VectorStoreError``, and
    the verb exits **1** with an accurate line. A reviewer meeting the changed exit
    code without this row would read a behaviour improvement as a regression.

    Driven through a REAL ``QdrantVectorStore`` against a dead port — a fake would
    not prove the path is reachable, which is the entire claim.
    """
    from mitos import cli

    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)
    _commit(store, "node-a", "Axiom A")
    _clear_outbox(store)

    manager.embed_provider = MagicMock()
    manager.vector_store = QdrantVectorStore("http://localhost:9", "mitos-tmp-dead")

    with patch("mitos.cli.MitosSyncManager", return_value=manager):
        rc = cli.cmd_reconcile(config, as_json=False)

    assert rc == 1


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


def test_reconcile_reads_an_absent_collection_as_empty(
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """An absent collection is the state reconcile exists FOR — it heals, never errors.

    W9. Before 1b the pre-write diff's 404 raised and ``cmd_reconcile`` rendered
    "Qdrant or embedding provider down", so the heal this vision prescribes failed
    on the very state the collection migration creates, and blamed infrastructure
    for it. The absent collection now reads as an empty point set: every expected
    node is missing, every one is enqueued, and the drain's first upsert creates
    the collection.
    """
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)
    ids = [_commit(store, f"node-{i}", f"Axiom {i}") for i in range(3)]
    _clear_outbox(store)

    _mock_embed_stack(manager, present_uuids=set())
    manager.vector_store.list_point_ids = MagicMock(
        side_effect=CollectionMissingError(
            "Qdrant collection 'mitos-x' does not exist", collection="mitos-x"
        )
    )

    result = manager.reconcile_embeddings()

    assert _shipped(result) == {"active": 3, "present": 0, "enqueued": 3}
    assert _added(result) == {"expected": 3, "expected_present": 0, "written": 3,
                              "left_queued": 0, "drain_stop": None}
    upserted = {call.args[0] for call in manager.vector_store.upsert.call_args_list}
    assert upserted == set(ids)
    assert store.get_pending_embeddings() == []


def test_reconcile_still_raises_when_qdrant_is_unreachable(
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """The refused-connection twin: only ABSENCE reads as empty, an outage propagates.

    The narrow catch is the whole point — widened to ``VectorStoreError`` it would
    swallow a genuine outage and re-embed the entire corpus against a dead Qdrant.
    """
    config, manager, tmpdir = sync_env
    store = GraphStore(config.db_path)
    _commit(store, "node-a", "Axiom A")
    _clear_outbox(store)

    _mock_embed_stack(manager, present_uuids=set())
    manager.vector_store.list_point_ids = MagicMock(
        side_effect=VectorStoreError("Qdrant scroll connection error: refused")
    )

    with pytest.raises(VectorStoreError):
        manager.reconcile_embeddings()


@patch("mitos.vector_store.requests.post")
@patch("mitos.vector_store.requests.get")
def test_list_point_ids_404_is_a_collection_missing_error(
    mock_get: MagicMock, mock_post: MagicMock
) -> None:
    """The typed 404 arrives through the real store, not only through a fake."""
    mock_get.return_value = MagicMock(status_code=200)
    mock_post.return_value = MagicMock(status_code=404, text="doesn't exist")

    vs = QdrantVectorStore("http://localhost:7333", "coll")
    with pytest.raises(CollectionMissingError) as excinfo:
        vs.list_point_ids()
    assert excinfo.value.collection == "coll"


@patch("mitos.vector_store.requests.post")
@patch("mitos.vector_store.requests.get")
def test_list_point_ids_pagination(mock_get: MagicMock, mock_post: MagicMock) -> None:
    """list_point_ids walks every scroll page until next_page_offset is null."""
    # Construction contacts nothing since 1b; `requests.get` is patched only so a
    # stray call would be visible rather than hitting a real localhost.
    mock_get.return_value = MagicMock(status_code=200)

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
    mock_get.return_value = MagicMock(status_code=200)
    mock_post.side_effect = requests.RequestException("connection refused")

    vs = QdrantVectorStore("http://localhost:7333", "coll")
    with pytest.raises(VectorStoreError):
        vs.list_point_ids()


# --------------------------------------------------------------------------- #
# 3b — the expected set, the drain's report, and the listing hand-off
# --------------------------------------------------------------------------- #
#
# These rows drive explicit doubles with real counters (``test_outbox_drain``'s
# ``_Vector`` / ``_Embed``), never a ``MagicMock`` vector store: a MagicMock absorbs a new
# keyword silently, and the listing count is the whole claim of the hand-off rows.

from mitos.provider_cause import QUOTA  # noqa: E402
from mitos.sync import DRAIN_STOP_OUTAGE  # noqa: E402

from test_outbox_drain import POISON, _Embed, _Vector  # noqa: E402
from test_provider_cause import _REPRESENTATIVE  # noqa: E402


def _graveyard(store: GraphStore) -> Tuple[set, set]:
    """Two live decisions and two retired ones; returns ``(active_ids, retired_ids)``."""
    live_a = _commit(store, "live-a", "Live axiom A.")
    old_b = _commit(store, "old-b", "Retired axiom B.")
    new_b = _commit(store, "new-b", "Replacement axiom B.", supersedes=["old-b"])
    old_c = _commit(store, "old-c", "Retired axiom C.")
    new_c = _commit(store, "new-c", "Replacement axiom C.", supersedes=["old-c"])
    _clear_outbox(store)
    return {live_a, new_b, new_c}, {old_b, old_c}


def _wire(manager: MitosSyncManager, vector: _Vector, embed: _Embed) -> None:
    manager.embed_provider = embed  # type: ignore[assignment]
    manager.vector_store = vector  # type: ignore[assignment]


def test_reconcile_over_an_active_only_collection_embeds_exactly_the_retired(
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """The vision's §4.5 case: the collection holds the active points only (today's path).

    Reconcile enqueues the retired decisions and nothing else, the drain presence-skips
    nothing it was not handed (the active points are not queued), the upsert spy sees only
    the missing ids, and the report says what was written. A second run is a no-op.
    """
    config, manager, _ = sync_env
    store = GraphStore(config.db_path)
    active, retired = _graveyard(store)

    vector, embed = _Vector(present=tuple(active)), _Embed()
    _wire(manager, vector, embed)

    result = manager.reconcile_embeddings()

    assert _shipped(result) == {"active": 3, "present": 3, "enqueued": 2}
    assert _added(result) == {"expected": 5, "expected_present": 3, "written": 2,
                              "left_queued": 0, "drain_stop": None}
    assert set(vector.written) == retired
    assert len(embed.calls) == 2
    assert store.get_pending_embeddings() == []

    again = manager.reconcile_embeddings()
    assert _shipped(again) == {"active": 3, "present": 5, "enqueued": 0}
    assert again["expected_present"] == 5 and again["written"] == 0
    assert set(vector.written) == retired  # no second write


def test_a_per_row_failure_leaves_one_queued_and_written_one_short(
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """``left_queued`` is the standing outbox, and a per-row failure stands in it."""
    config, manager, _ = sync_env
    store = GraphStore(config.db_path)
    for i in range(3):
        _commit(store, f"node-{i}", f"Axiom {i}.")
    poisoned = _commit(store, "poisoned", f"{POISON} axiom.")
    _clear_outbox(store)

    _wire(manager, _Vector(), _Embed())
    result = manager.reconcile_embeddings()

    assert result["enqueued"] == 4
    assert result["written"] == 3
    assert result["left_queued"] == 1
    assert result["drain_stop"] is None
    assert [row["node_id"] for row in store.get_pending_embeddings()] == [poisoned]


def test_an_outage_on_row_one_of_33_leaves_33_queued(
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """The drain's ``failed`` counts rows attempted (1); 33 stand, and the report says 33."""
    config, manager, _ = sync_env
    store = GraphStore(config.db_path)
    for i in range(33):
        _commit(store, f"node-{i:02d}", f"Axiom {i}.")
    _clear_outbox(store)

    _wire(manager, _Vector(), _Embed(fail_all=_REPRESENTATIVE[QUOTA]))
    result = manager.reconcile_embeddings()

    assert result["enqueued"] == 33
    assert result["written"] == 0
    assert result["left_queued"] == 33
    assert result["drain_stop"] == DRAIN_STOP_OUTAGE


def test_left_queued_is_read_not_inferred_from_the_counters(
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """Rows queued before reconcile make the arithmetic shortcuts wrong; the read is right.

    A present node's per-commit row is presence-skipped under the marker, and a
    superseded open question's row (outside the expected set, so never enqueued by
    reconcile) is embedded by the drain — ``written`` counts it, so ``enqueued - written``
    would report ``-1`` rows left. The report reads the outbox instead.
    """
    config, manager, _ = sync_env
    store = GraphStore(config.db_path)
    present_node = _commit(store, "already-indexed", "Present axiom.")
    missing = [_commit(store, f"missing-{i}", f"Missing axiom {i}.") for i in range(2)]
    q1 = ParsedEntry("open_question", "q-old", 1, 5)
    q1.topic, q1.questions_raised = "A topic.", ["Old question?"]
    store.commit_parsed_entry(q1)
    q2 = ParsedEntry("open_question", "q-new", 1, 5)
    q2.topic, q2.questions_raised = "A topic.", ["New question?"]
    q2.supersedes = ["q-old"]
    q_new = store.commit_parsed_entry(q2).node_id
    _clear_outbox(store)
    # Queued BEFORE reconcile: a present node's row and the killed open question's row.
    store.add_pending_embedding(present_node)
    killed_id = next(n["id"] for n in store.get_all_nodes() if n["slug"] == "q-old")
    assert killed_id not in store.get_decision_index_expected_ids()
    store.add_pending_embedding(killed_id)

    vector = _Vector(present=(present_node,))
    _wire(manager, vector, _Embed())
    result = manager.reconcile_embeddings()

    assert result["enqueued"] == 3  # the two missing decisions + the live open question
    assert set(missing) | {q_new} <= set(vector.written)
    assert result["written"] == 4  # …plus the killed open question's pre-queued row
    assert result["enqueued"] - result["written"] == -1  # the shortcut is wrong here
    assert result["left_queued"] == 0 == len(store.get_pending_embeddings())


def test_the_listing_is_handed_to_the_drain_once_per_reconcile(
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """A present collection is scrolled once per reconcile, not again by its drain."""
    config, manager, _ = sync_env
    store = GraphStore(config.db_path)
    active, _retired = _graveyard(store)
    vector = _Vector(present=tuple(active))
    _wire(manager, vector, _Embed())

    manager.reconcile_embeddings()

    assert vector.list_calls == 1


def test_an_absent_collection_is_listed_twice_reconciles_404_then_the_drains_own(
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """Over a 404 nothing is handed: the drain lists for itself; its spend rule sees the truth."""
    config, manager, _ = sync_env
    store = GraphStore(config.db_path)
    _graveyard(store)
    vector = _Vector(absent=True)
    _wire(manager, vector, _Embed())

    result = manager.reconcile_embeddings()

    assert vector.list_calls == 2
    assert result["written"] == 5
    assert vector.created


def test_an_absent_collection_with_every_row_failing_keeps_the_marker(
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """3a's absent-collection spend rule, untouched by the hand-off (B7's row).

    The walk completes with the collection still absent and rows standing, so the marker
    reconcile stamped is kept. Handing the drain ``listing=set()`` over the 404 would fake
    a present collection and spend it.
    """
    config, manager, _ = sync_env
    store = GraphStore(config.db_path)
    for i in range(3):
        _commit(store, f"poison-{i}", f"{POISON} axiom {i}.")
    _clear_outbox(store)
    _wire(manager, _Vector(absent=True), _Embed())

    result = manager.reconcile_embeddings()

    assert result["written"] == 0
    assert result["left_queued"] == 3
    seed = store.embedding_seed()
    assert seed is not None and seed["established_by"] == "reconcile"


def test_cmd_reconcile_text_and_json_carry_the_same_numbers(
    sync_env: Tuple[MitosConfig, MitosSyncManager, str], capsys
) -> None:
    """Both renderings over a graveyard fixture where ``present != expected_present``.

    The collection holds the active points plus one orphan point, so ``present`` (4) is
    not what was already indexed (3). The text line no longer calls ``present`` "already
    indexed" (the stale §4.11 line); ``--json`` stdout is one object whose three shipped
    keys keep their values.
    """
    from mitos import cli

    config, manager, _ = sync_env
    store = GraphStore(config.db_path)
    active, _retired = _graveyard(store)
    orphan = "e" * 64

    def _run(as_json: bool) -> str:
        vector = _Vector(present=tuple(active) + (orphan,))
        _wire(manager, vector, _Embed())
        for node_id in store.get_decision_index_expected_ids() - active:
            store.remove_pending_embedding(node_id)
        with patch("mitos.cli.MitosSyncManager", return_value=manager):
            assert cli.cmd_reconcile(config, as_json=as_json) == 0
        return capsys.readouterr().out

    payload = json.loads(_run(True))
    assert _shipped(payload) == {"active": 3, "present": 4, "enqueued": 2}
    assert _added(payload) == {"expected": 5, "expected_present": 3, "written": 2,
                               "left_queued": 0, "drain_stop": None}

    text = _run(False)
    assert "Reconciled: 5 node(s) the decision index should hold (3 active), " \
           "3 already indexed, 2 enqueued, 2 embedded." in text
    assert "4 point(s) already indexed" not in text
    assert "still queued" not in text


def test_cmd_reconcile_text_names_rows_left_queued(
    sync_env: Tuple[MitosConfig, MitosSyncManager, str], capsys
) -> None:
    """Where today "N re-embedded" counted what was enqueued, a drain that left rows says so."""
    from mitos import cli

    config, manager, _ = sync_env
    store = GraphStore(config.db_path)
    _commit(store, "poisoned", f"{POISON} axiom.")
    _clear_outbox(store)
    _wire(manager, _Vector(), _Embed())

    with patch("mitos.cli.MitosSyncManager", return_value=manager):
        assert cli.cmd_reconcile(config, as_json=False) == 0
    out = capsys.readouterr().out

    assert "1 enqueued, 0 embedded." in out
    assert "1 embedding row(s) still queued" in out


def test_provider_down_still_reads_the_standing_outbox(
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """``left_queued`` is a read on every path: a provider-down reconcile does nothing,
    but the rows already queued still stand and the report says so."""
    config, manager, _ = sync_env
    store = GraphStore(config.db_path)
    _commit(store, "queued-a", "Axiom A.")
    _commit(store, "queued-b", "Axiom B.")  # both commits self-enqueue
    manager.embed_provider = None
    manager.vector_store = None

    result = manager.reconcile_embeddings()

    assert _shipped(result) == {"active": 0, "present": 0, "enqueued": 0}
    assert result["written"] == 0 and result["drain_stop"] is None
    assert result["left_queued"] == 2
