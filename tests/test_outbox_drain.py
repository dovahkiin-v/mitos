"""The decision outbox's drain: one walk, presence-skip, and the marker's spend rule.

Phase 3a of the recall-confidence vision. ``drain_pending_embeddings`` walks the outbox in
queue order with a keyset cursor and attempts each row once per walk, so no head of
failing rows can stop a later drain before the rows behind it. An outage (the provider
classifier's ``auth`` / ``quota`` / ``transient`` on the embed call, or the vector store
failing the write) still stops the walk at once. Under a standing ``embedding_seed``
marker the drain lists the collection once and removes, without embedding, every row
whose point is already present. The marker is spent when the walk completes with the
collection present, or when the backlog is empty; an outage or a refusal keeps it, and a
re-stamp made while the walk ran survives.

Rows marked ``cmd_sync(embed_only=True)`` run through the verb, with
``mitos.cli.MitosSyncManager`` patched to return a manager wired to explicit doubles.
The doubles are plain classes with a real ``present`` set and real call counters, never
a ``MagicMock``: a ``MagicMock`` vector store "lists" an object whose ``in`` answers
False, so a marker-standing drain over one would silently list empty.

Run under ``./venv/bin/python -m pytest`` (PATTERNS — bare ``python`` lacks deps).
"""

import json
import os
import sqlite3
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from unittest.mock import patch

import pytest

from mitos import cli
from mitos.config import MitosConfig
from mitos.errors import CollectionMissingError, DatabaseError, VectorStoreError
from mitos.parser import ParsedEntry
from mitos.provider_cause import AUTH, QUOTA, TRANSIENT, UNKNOWN, provider_cause_phrase
from mitos.store import GraphStore
from mitos.sync import (
    DRAIN_MARKER_KEPT,
    DRAIN_MARKER_SPENT,
    DRAIN_STOP_CLAIM_FAILED,
    DRAIN_STOP_OUTAGE,
    DRAIN_STOP_REFUSED,
    DrainOutcome,
    MitosSyncManager,
)
from mitos.vector_store import hash_to_uuid

from test_provider_cause import _REPRESENTATIVE

POISON = "POISON"


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #

class _Vector:
    """A vector store with a real point set and real counters.

    ``present`` holds node ids whose points the collection already has (stored as point
    ids, the way ``list_point_ids`` returns them). ``absent=True`` answers like a real
    store whose collection does not exist: the listing raises, a covering write creates,
    a non-covering one is refused. ``upsert_exc`` makes every write raise it.
    ``on_upsert`` runs before each write, for the mid-walk rows.
    """

    def __init__(self, *, present: Tuple[str, ...] = (), absent: bool = False,
                 upsert_exc: Optional[BaseException] = None,
                 list_exc: Optional[BaseException] = None,
                 payloads: Optional[Dict[str, Dict[str, Any]]] = None,
                 on_upsert: Optional[Callable[[str], None]] = None) -> None:
        self.points: Set[str] = {hash_to_uuid(node_id) for node_id in present}
        self.absent = absent
        self.created = False
        self.upsert_exc = upsert_exc
        self.list_exc = list_exc
        self.payloads: Dict[str, Dict[str, Any]] = dict(payloads or {})
        self.on_upsert = on_upsert
        self.list_calls = 0
        self.declarations: List[bool] = []
        self.written: List[str] = []

    def _missing(self) -> CollectionMissingError:
        return CollectionMissingError(
            "Qdrant collection 'mitos-tmp-fake' does not exist", collection="mitos-tmp-fake"
        )

    def list_point_ids(self, page_size: int = 256) -> Set[str]:
        self.list_calls += 1
        if self.list_exc is not None:
            raise self.list_exc
        if self.absent and not self.created:
            raise self._missing()
        return set(self.points)

    def upsert(self, point_id: str, vector: List[float], payload: Dict[str, Any],
               *, may_create: bool) -> None:
        self.declarations.append(may_create)
        if self.on_upsert is not None:
            self.on_upsert(point_id)
        if self.upsert_exc is not None:
            raise self.upsert_exc
        if self.absent and not self.created:
            if not may_create:
                raise self._missing()
            self.created = True
        self.points.add(hash_to_uuid(point_id))
        self.payloads[point_id] = dict(payload)
        self.written.append(point_id)

    def query(self, vector: List[float], limit: int = 5, *,
              with_payload: bool = True) -> List[Dict[str, Any]]:
        return []


class _Embed:
    """Counts calls; text carrying ``POISON`` fails with ``poison()``'s exception."""

    def __init__(self, poison: Optional[Callable[[], BaseException]] = None,
                 fail_all: Optional[Callable[[], BaseException]] = None) -> None:
        self.calls: List[str] = []
        self.poison = poison or (lambda: _REPRESENTATIVE[UNKNOWN]())
        self.fail_all = fail_all

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        self.calls.append(text)
        if self.fail_all is not None:
            raise self.fail_all()
        if POISON in text:
            raise self.poison()
        return [0.1, 0.2, 0.3]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _workspace(tmp_path) -> MitosConfig:
    root = tmp_path / "ws"
    os.makedirs(root / ".mitos", exist_ok=True)
    config = MitosConfig(str(root))
    with open(config.decisions_file, "w", encoding="utf-8") as handle:
        handle.write(
            "# Decisions\n"
            "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->\n"
        )
    return config


def _decision(store: GraphStore, slug: str, axiom: str) -> str:
    """Commits a decision through parse→commit (which queues its row) and returns its id."""
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = axiom
    entry.rejected_paths = "None."
    return store.commit_parsed_entry(entry).node_id


def _manager(config: MitosConfig, vector: _Vector, embed: _Embed) -> MitosSyncManager:
    manager = MitosSyncManager(config)
    manager.embed_provider = embed  # type: ignore[assignment]
    manager.vector_store = vector  # type: ignore[assignment]
    return manager


def _sync_embed_only(config: MitosConfig, manager: MitosSyncManager) -> DrainOutcome:
    """Runs ``mitos sync --embed-only`` against ``manager`` and returns the drain's outcome."""
    outcomes: List[DrainOutcome] = []
    real = manager.drain_pending_embeddings

    def _spy() -> DrainOutcome:
        outcome = real()
        outcomes.append(outcome)
        return outcome

    manager.drain_pending_embeddings = _spy  # type: ignore[method-assign]
    try:
        with patch.object(cli, "MitosSyncManager", return_value=manager):
            cli.cmd_sync(config, embed_only=True)
    finally:
        del manager.drain_pending_embeddings
    assert len(outcomes) == 1
    return outcomes[0]


def _retries(store: GraphStore) -> Dict[str, int]:
    return {row["node_id"]: row["retry_count"] for row in store.get_pending_embeddings()}


# --------------------------------------------------------------------------- #
# D1 — a head of failing rows larger than one claim blocks nothing
# --------------------------------------------------------------------------- #

def test_d1_a_head_of_failing_rows_blocks_nothing_behind_it(tmp_path, capsys) -> None:
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    poison = [_decision(store, f"poison-{i:02d}", f"{POISON} axiom {i}.") for i in range(12)]
    healthy = [_decision(store, f"healthy-{i}", f"Healthy axiom {i}.") for i in range(5)]

    vector, embed = _Vector(), _Embed()
    manager = _manager(config, vector, embed)

    outcome = _sync_embed_only(config, manager)

    assert outcome == DrainOutcome(written=5, skipped_present=0, pruned=0, failed=12,
                                   stop=None, marker=None)
    assert sorted(vector.written) == sorted(healthy)
    assert len(embed.calls) == 17  # each row tried exactly once
    assert _retries(store) == {node_id: 1 for node_id in poison}

    # A second drain does the same: every poison row once more, and rows queued behind
    # them are still reached.
    later = [_decision(store, f"later-{i}", f"Later axiom {i}.") for i in range(5)]
    outcome = _sync_embed_only(config, manager)

    assert outcome == DrainOutcome(written=5, skipped_present=0, pruned=0, failed=12,
                                   stop=None, marker=None)
    assert set(vector.written) == set(healthy) | set(later)
    assert _retries(store) == {node_id: 2 for node_id in poison}


# --------------------------------------------------------------------------- #
# D2 / D3 — an outage stops the walk after one attempted row
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cause", [AUTH, QUOTA, TRANSIENT])
def test_d2_a_provider_outage_stops_at_the_first_row(tmp_path, capsys, cause) -> None:
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    ids = [_decision(store, f"row-{i}", f"Axiom {i}.") for i in range(5)]

    vector = _Vector()
    embed = _Embed(fail_all=_REPRESENTATIVE[cause])
    manager = _manager(config, vector, embed)
    capsys.readouterr()

    outcome = _sync_embed_only(config, manager)

    assert outcome.stop == DRAIN_STOP_OUTAGE
    assert outcome.failed == 1 and outcome.written == 0
    assert len(embed.calls) == 1
    assert vector.declarations == []
    assert _retries(store) == {ids[0]: 1, **{node_id: 0 for node_id in ids[1:]}}

    out = capsys.readouterr().out
    failure = f"[Warning] Failed to drain embedding for 'row-0': {provider_cause_phrase(cause)}"
    assert out.count("Failed to drain embedding for '") == 1
    assert failure in out
    assert out.index(failure) < out.index("Stopping the drain")


def test_d3_a_store_outage_on_the_write_stops_the_walk(tmp_path, capsys) -> None:
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    ids = [_decision(store, f"row-{i}", f"Axiom {i}.") for i in range(5)]

    vector = _Vector(upsert_exc=VectorStoreError("Qdrant returned 500"))
    embed = _Embed()
    manager = _manager(config, vector, embed)

    outcome = _sync_embed_only(config, manager)

    assert outcome.stop == DRAIN_STOP_OUTAGE
    assert len(vector.declarations) == 1
    assert len(embed.calls) == 1
    assert _retries(store) == {ids[0]: 1, **{node_id: 0 for node_id in ids[1:]}}


def test_d3_twin_a_missing_collection_with_no_marker_is_still_the_refusal(
    tmp_path, capsys
) -> None:
    """The refusal is not an outage: one line, no increment, nothing created."""
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    for i in range(3):
        _decision(store, f"row-{i}", f"Axiom {i}.")
    before = _retries(store)

    vector = _Vector(absent=True)
    manager = _manager(config, vector, _Embed())
    capsys.readouterr()

    outcome = _sync_embed_only(config, manager)

    assert outcome.stop == DRAIN_STOP_REFUSED
    assert outcome.marker is None
    assert vector.declarations == [False]
    assert vector.created is False
    assert vector.list_calls == 0
    assert _retries(store) == before
    assert capsys.readouterr().out.count("Stopping the drain") == 1


# --------------------------------------------------------------------------- #
# D4–D7 — presence-skip, only under the marker
# --------------------------------------------------------------------------- #

def test_d4_a_seeded_drain_skips_present_points_without_embedding(tmp_path, capsys) -> None:
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    ids = [_decision(store, f"row-{i}", f"Axiom {i}.") for i in range(5)]
    store.stamp_embedding_seed("rebuild")

    vector = _Vector(present=tuple(ids[:4]))
    embed = _Embed()
    manager = _manager(config, vector, embed)
    capsys.readouterr()

    outcome = _sync_embed_only(config, manager)

    assert outcome == DrainOutcome(written=1, skipped_present=4, pruned=0, failed=0,
                                   stop=None, marker=DRAIN_MARKER_SPENT)
    assert len(embed.calls) == 1
    assert vector.written == [ids[4]]
    assert vector.list_calls == 1
    assert store.get_pending_embeddings() == []
    assert store.embedding_seed() is None
    out = capsys.readouterr().out
    assert "Skipped 4 already-indexed row(s): the embedding queue was seeded by `rebuild`." in out
    assert "Successfully drained" in out and out.count("Successfully drained") == 1


def test_d5_a_seeded_drain_over_an_absent_collection_creates_it(tmp_path) -> None:
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    for i in range(3):
        _decision(store, f"row-{i}", f"Axiom {i}.")
    store.stamp_embedding_seed("reconcile")

    vector = _Vector(absent=True)
    embed = _Embed()
    manager = _manager(config, vector, embed)

    outcome = _sync_embed_only(config, manager)

    assert vector.list_calls == 1
    assert vector.declarations == [True, True, True]
    assert vector.created is True
    assert len(embed.calls) == 3
    assert outcome == DrainOutcome(written=3, skipped_present=0, pruned=0, failed=0,
                                   stop=None, marker=DRAIN_MARKER_SPENT)
    assert store.embedding_seed() is None


def test_d6_an_ordinary_drain_re_upserts_a_present_point(tmp_path) -> None:
    """No marker: nothing is listed, and a stale payload slug is healed by the write."""
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    node_id = _decision(store, "new-name", "An axiom renamed while embedding was down.")

    vector = _Vector(present=(node_id,), payloads={node_id: {"slug": "old-name"}})
    embed = _Embed()
    manager = _manager(config, vector, embed)

    outcome = _sync_embed_only(config, manager)

    assert vector.list_calls == 0
    assert vector.written == [node_id]
    assert vector.payloads[node_id]["slug"] == "new-name"
    assert outcome.written == 1 and outcome.skipped_present == 0
    assert outcome.marker is None


def test_d7_the_residual_a_per_commit_row_under_a_marker_is_skipped(tmp_path) -> None:
    """Known and pinned: under a standing marker a present node's row is not re-upserted,
    so its stale payload slug waits for the node's next commit."""
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    node_id = _decision(store, "new-name", "An axiom renamed while embedding was down.")
    store.stamp_embedding_seed("rebuild")

    vector = _Vector(present=(node_id,), payloads={node_id: {"slug": "old-name"}})
    embed = _Embed()
    manager = _manager(config, vector, embed)

    outcome = manager.drain_pending_embeddings()

    assert outcome.skipped_present == 1
    assert vector.written == []
    assert embed.calls == []
    assert vector.payloads[node_id]["slug"] == "old-name"


# --------------------------------------------------------------------------- #
# D8–D13 — the spend rule
# --------------------------------------------------------------------------- #

def test_d8_a_poison_row_left_with_the_collection_present_spends_the_marker(
    tmp_path, capsys
) -> None:
    """The listing alone proves presence: every healthy row is skipped, the poison row
    fails at the embed call, no upsert runs, and the marker is still spent. The next
    ordinary drain neither lists nor may create."""
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    healthy = [_decision(store, f"row-{i}", f"Axiom {i}.") for i in range(3)]
    poison = _decision(store, "poison", f"{POISON} axiom.")
    store.stamp_embedding_seed("rebuild")

    vector = _Vector(present=tuple(healthy))
    manager = _manager(config, vector, _Embed())

    outcome = _sync_embed_only(config, manager)

    assert outcome == DrainOutcome(written=0, skipped_present=3, pruned=0, failed=1,
                                   stop=None, marker=DRAIN_MARKER_SPENT)
    assert vector.declarations == []
    assert store.embedding_seed() is None
    assert set(_retries(store)) == {poison}

    # The next ordinary drain, over a collection that has since gone absent.
    fresh = _decision(store, "fresh", "A fresh healthy axiom.")
    absent = _Vector(absent=True)
    manager.vector_store = absent  # type: ignore[assignment]

    outcome = _sync_embed_only(config, manager)

    assert absent.list_calls == 0
    assert absent.declarations == [False]  # the poison row failed before any upsert
    assert absent.created is False
    assert outcome.stop == DRAIN_STOP_REFUSED
    assert outcome.marker is None
    assert set(_retries(store)) == {poison, fresh}


def test_d9_an_absent_collection_with_rows_left_keeps_the_marker(tmp_path) -> None:
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    for i in range(3):
        _decision(store, f"poison-{i}", f"{POISON} axiom {i}.")
    store.stamp_embedding_seed("rebuild")

    vector = _Vector(absent=True)
    manager = _manager(config, vector, _Embed())

    outcome = manager.drain_pending_embeddings()

    assert outcome == DrainOutcome(written=0, skipped_present=0, pruned=0, failed=3,
                                   stop=None, marker=DRAIN_MARKER_KEPT)
    assert vector.created is False
    assert store.embedding_seed() is not None


def test_d10_an_outage_under_the_marker_keeps_it(tmp_path) -> None:
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    for i in range(3):
        _decision(store, f"row-{i}", f"Axiom {i}.")
    store.stamp_embedding_seed("reconcile")

    manager = _manager(config, _Vector(), _Embed(fail_all=_REPRESENTATIVE[QUOTA]))

    outcome = manager.drain_pending_embeddings()

    assert outcome.stop == DRAIN_STOP_OUTAGE
    assert outcome.marker == DRAIN_MARKER_KEPT
    assert store.embedding_seed()["established_by"] == "reconcile"


def test_an_outage_after_skips_still_reports_the_skipped_count(tmp_path, capsys) -> None:
    """A skip prints no per-row line, so the count is the only trace of the rows it
    removed; an outage stop must not swallow it."""
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    ids = [_decision(store, f"row-{i}", f"Axiom {i}.") for i in range(2)]
    _decision(store, "row-2", f"{POISON} axiom 2.")
    store.stamp_embedding_seed("rebuild")

    vector = _Vector(present=tuple(ids))
    manager = _manager(config, vector, _Embed(poison=_REPRESENTATIVE[QUOTA]))
    capsys.readouterr()

    outcome = manager.drain_pending_embeddings()

    assert outcome == DrainOutcome(written=0, skipped_present=2, pruned=0, failed=1,
                                   stop=DRAIN_STOP_OUTAGE, marker=DRAIN_MARKER_KEPT)
    out = capsys.readouterr().out
    assert out.count("Skipped 2 already-indexed row(s): the embedding queue was "
                     "seeded by `rebuild`.") == 1
    assert out.index("Stopping the drain") < out.index("Skipped 2")


def test_d10_a_listing_outage_under_the_marker_keeps_it_before_any_row(tmp_path) -> None:
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    for i in range(3):
        _decision(store, f"row-{i}", f"Axiom {i}.")
    store.stamp_embedding_seed("rebuild")

    embed = _Embed()
    manager = _manager(config, _Vector(list_exc=VectorStoreError("connection refused")), embed)

    outcome = manager.drain_pending_embeddings()

    assert outcome == DrainOutcome(written=0, skipped_present=0, pruned=0, failed=0,
                                   stop=DRAIN_STOP_OUTAGE, marker=DRAIN_MARKER_KEPT)
    assert embed.calls == []
    assert set(_retries(store).values()) == {0}
    assert store.embedding_seed() is not None


def test_d10_a_refusal_under_the_marker_keeps_it(tmp_path) -> None:
    """A collection that lists as present but refuses the write (it vanished mid-walk)."""
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    for i in range(2):
        _decision(store, f"row-{i}", f"Axiom {i}.")
    store.stamp_embedding_seed("rebuild")

    vector = _Vector(upsert_exc=CollectionMissingError(
        "Qdrant collection 'mitos-tmp-fake' does not exist", collection="mitos-tmp-fake"))
    manager = _manager(config, vector, _Embed())

    outcome = manager.drain_pending_embeddings()

    assert outcome.stop == DRAIN_STOP_REFUSED
    assert outcome.marker == DRAIN_MARKER_KEPT
    assert vector.declarations == [True]
    assert set(_retries(store).values()) == {0}
    assert store.embedding_seed() is not None


def test_d11_a_marker_re_stamped_mid_walk_survives_the_spend(tmp_path) -> None:
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    for i in range(3):
        _decision(store, f"row-{i}", f"Axiom {i}.")
    store.stamp_embedding_seed("rebuild")
    read_at = store.embedding_seed()["established_at"]

    stamped: List[str] = []

    def _restamp(point_id: str) -> None:
        if not stamped:
            store.stamp_embedding_seed("reconcile")
            stamped.append(point_id)

    manager = _manager(config, _Vector(on_upsert=_restamp), _Embed())

    outcome = manager.drain_pending_embeddings()

    assert outcome.written == 3 and outcome.stop is None
    seed = store.embedding_seed()
    assert seed is not None
    assert seed["established_by"] == "reconcile"
    assert seed["established_at"] != read_at


def test_d12_a_row_enqueued_mid_walk_is_written_in_the_same_walk(tmp_path) -> None:
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    ids = [_decision(store, f"row-{i}", f"Axiom {i}.") for i in range(3)]

    late: List[str] = []

    def _enqueue(point_id: str) -> None:
        if not late:
            late.append(_decision(store, "late", "An axiom committed mid-walk."))

    vector = _Vector(on_upsert=_enqueue)
    manager = _manager(config, vector, _Embed())

    outcome = manager.drain_pending_embeddings()

    assert outcome.written == 4 and outcome.stop is None
    assert set(vector.written) == set(ids) | set(late)
    assert store.get_pending_embeddings() == []


def test_d13_an_empty_backlog_spends_the_marker_without_listing(tmp_path) -> None:
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    store.stamp_embedding_seed("cutover")

    vector = _Vector()
    manager = _manager(config, vector, _Embed())

    outcome = manager.drain_pending_embeddings()

    assert outcome == DrainOutcome(written=0, skipped_present=0, pruned=0, failed=0,
                                   stop=None, marker=DRAIN_MARKER_SPENT)
    assert vector.list_calls == 0
    assert store.embedding_seed() is None


# --------------------------------------------------------------------------- #
# D14 — `reconcile --json` prints one object on stdout
# --------------------------------------------------------------------------- #

def test_d14_reconcile_json_keeps_the_drains_lines_off_stdout(tmp_path, capsys) -> None:
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    _decision(store, "row-0", "Axiom 0.")

    vector = _Vector()
    manager = _manager(config, vector, _Embed())
    capsys.readouterr()

    with patch.object(cli, "MitosSyncManager", return_value=manager):
        code = cli.cmd_reconcile(config, as_json=True)

    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["enqueued"] == 1
    assert "Draining pending embeddings queue" in captured.err
    assert "Successfully drained embedding for 'row-0'" in captured.err


# --------------------------------------------------------------------------- #
# The unspecified arms (G4 and the walk's other exits)
# --------------------------------------------------------------------------- #

def test_g4_a_remove_fault_carrying_429_is_a_per_row_failure(tmp_path) -> None:
    """The classifier reads only the embed call's exception: a graph error whose text
    carries "429" after a successful write is per-row, and the walk continues."""
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    ids = [_decision(store, f"row-{i}", f"Axiom {i}.") for i in range(3)]

    vector = _Vector()
    manager = _manager(config, vector, _Embed())
    real_remove = manager.store.remove_pending_embedding
    faults: List[str] = []

    def _remove(node_id: str) -> None:
        if not faults:
            faults.append(node_id)
            raise DatabaseError("disk I/O error near page 429")
        real_remove(node_id)

    manager.store.remove_pending_embedding = _remove  # type: ignore[method-assign]

    outcome = manager.drain_pending_embeddings()

    assert outcome.stop is None
    assert outcome.written == 2 and outcome.failed == 1
    assert vector.written == ids
    assert _retries(store) == {ids[0]: 1}


def test_a_failed_claim_keeps_the_marker(tmp_path, capsys) -> None:
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    _decision(store, "row-0", "Axiom 0.")
    store.stamp_embedding_seed("rebuild")

    manager = _manager(config, _Vector(), _Embed())

    def _claim(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        raise DatabaseError("database is locked")

    manager.store.claim_pending_embeddings = _claim  # type: ignore[method-assign]

    outcome = manager.drain_pending_embeddings()

    assert outcome.stop == DRAIN_STOP_CLAIM_FAILED
    assert outcome.marker == DRAIN_MARKER_KEPT
    assert store.embedding_seed() is not None
    assert "Failed to claim outbox queue" in capsys.readouterr().out


def test_a_node_less_row_is_pruned(tmp_path) -> None:
    """Close to unreachable in production (the outbox's node_id is a foreign key), so the
    row is planted through a raw connection, where foreign keys are off by default."""
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    node_id = _decision(store, "row-0", "Axiom 0.")
    ghost = "f" * 64
    conn = sqlite3.connect(config.db_path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO pending_embeddings (node_id, queued_at, retry_count) "
                "VALUES (?, ?, 0)",
                (ghost, "2000-01-01T00:00:00+00:00"),
            )
    finally:
        conn.close()

    vector = _Vector()
    manager = _manager(config, vector, _Embed())

    outcome = manager.drain_pending_embeddings()

    assert outcome == DrainOutcome(written=1, skipped_present=0, pruned=1, failed=0,
                                   stop=None, marker=None)
    assert vector.written == [node_id]
    assert store.get_pending_embeddings() == []
