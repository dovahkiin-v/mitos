"""Every write site declares whether it may create an absent collection.

The offline half of phase 1c. ``tests/test_embedding_seed.py`` pins the two substrate
questions; this module pins the four write paths that ask them and what each does when the
answer is *no*:

* ``MitosSyncManager._best_effort_embed`` — the inline record/sync embed, declaring from
  the graph gate. Three reach sites share it: ``record_decision_entry``, the sync commit
  path, and an injected ``embed_fn=`` callback into ``replay.commit_quarantine_fixpoint``.
  The third is the one a test plan forgets, and nothing else exercises it.
* ``drain_pending_embeddings`` — declaring from the ``embedding_seed`` marker, refusing as
  a whole rather than per row, and clearing the marker when the outbox empties.
* ``MitosProseImporter._best_effort_embed`` — the same graph gate, and the one site that
  *adds* a defer path rather than reusing one.
* ``reconcile_embeddings`` — stamping the marker so its own drain is a covering write.

**Every assertion about which value was declared is made against an explicit fake, never a
``MagicMock``.** A MagicMock accepts ``may_create=<anything>`` and records it, so no
MagicMock-backed row can red on a wrong declaration — which is precisely the class of
silent pass this phase exists to prevent. The fakes below therefore spell the parameter
out and, where the intent matters, raise ``CollectionMissingError`` exactly as a real store
would.

Run under ``./venv/bin/python -m pytest`` (PATTERNS — bare ``python`` lacks deps).
"""

import os
from typing import Any, Dict, List, Set

import pytest

from mitos.config import MitosConfig
from mitos.errors import CollectionMissingError
from mitos.importer import MitosProseImporter
from mitos.parser import ParsedEntry
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager


class _RecordingVector:
    """Records every declared ``may_create``; optionally behaves as absent.

    ``absent=True`` makes it answer like a real store whose collection does not exist:
    a declared-covering write succeeds and "creates", a non-covering one raises the typed
    ``CollectionMissingError``. That is what turns these rows from "the parameter was
    passed" into "the refusal actually fires".
    """

    def __init__(self, *, absent: bool = False) -> None:
        self.absent = absent
        self.created = False
        self.declarations: List[bool] = []
        self.landed: List[str] = []

    def upsert(self, point_id: str, vector: List[float], payload: Dict[str, Any],
               *, may_create: bool) -> None:
        self.declarations.append(may_create)
        if self.absent and not self.created:
            if not may_create:
                raise CollectionMissingError(
                    "Qdrant collection 'mitos-tmp-fake' does not exist",
                    collection="mitos-tmp-fake",
                )
            self.created = True
        self.landed.append(point_id)

    def query(self, vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        return []

    def list_point_ids(self, page_size: int = 256) -> Set[str]:
        if self.absent and not self.created:
            raise CollectionMissingError(
                "Qdrant collection 'mitos-tmp-fake' does not exist",
                collection="mitos-tmp-fake",
            )
        return set()


class _FakeEmbed:
    """A fixed vector per call — no Gemini spend, and a call counter worth asserting."""

    def __init__(self) -> None:
        self.calls = 0

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        self.calls += 1
        return [0.1, 0.2, 0.3]


def _workspace(tmp_path, name: str = "ws") -> MitosConfig:
    """A minimal initialized workspace (``.mitos/`` + a decisions buffer header)."""
    root = tmp_path / name
    os.makedirs(root / ".mitos", exist_ok=True)
    config = MitosConfig(str(root))
    with open(config.decisions_file, "w", encoding="utf-8") as handle:
        handle.write(
            "# Decisions\n"
            "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->\n"
        )
    return config


def _decision(store: GraphStore, slug: str, axiom: str) -> str:
    """Commits a minimal decision through parse→commit and returns its node id."""
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = axiom
    entry.rejected_paths = "None."
    return store.commit_parsed_entry(entry).node_id


def _entry(slug: str, axiom: str) -> ParsedEntry:
    """A prepared (uncommitted) decision entry."""
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = axiom
    entry.rejected_paths = "None."
    return entry


def _manager(config: MitosConfig, vector: _RecordingVector) -> MitosSyncManager:
    """A sync manager wired to explicit fakes (never a MagicMock — see the docstring)."""
    manager = MitosSyncManager(config)
    manager.embed_provider = _FakeEmbed()  # type: ignore[assignment]
    manager.vector_store = vector  # type: ignore[assignment]
    return manager


def _outbox_slugs(store: GraphStore) -> Set[str]:
    """The slugs currently queued for embedding."""
    return {
        store.get_node(row["node_id"])["slug"]
        for row in store.get_pending_embeddings()
    }


# --------------------------------------------------------------------------- #
# The inline embed — declares from the graph gate
# --------------------------------------------------------------------------- #

def test_a_fresh_projects_first_record_declares_covering(tmp_path) -> None:
    """Success criterion 2: the carve-out survives, and it is a real create.

    The one node being written IS the whole active set, so there is nothing this write
    leaves unindexed and nothing for absence to protect.
    """
    config = _workspace(tmp_path)
    vector = _RecordingVector(absent=True)
    manager = _manager(config, vector)

    entry = _entry("first", "The first axiom.")
    delta = manager.store.commit_parsed_entry(entry)
    manager._best_effort_embed(delta, entry)

    assert vector.declarations == [True]
    assert vector.created is True
    assert vector.landed == [delta.node_id]
    # Indexed, so the commit's outbox row is dropped.
    assert manager.store.get_pending_embeddings() == []


def test_a_record_into_a_populated_graph_defers(tmp_path, capsys) -> None:
    """A single write into a populated graph refuses, and the outbox row survives.

    The durability half: nothing is lost, because the commit already enqueued this node
    and the refusal deliberately does not dequeue it. ``mitos reconcile`` completes it.
    """
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    _decision(store, "existing", "An axiom already in the graph.")

    vector = _RecordingVector(absent=True)
    manager = _manager(config, vector)
    entry = _entry("newcomer", "A newly recorded axiom.")
    delta = manager.store.commit_parsed_entry(entry)

    assert manager._best_effort_embed(delta, entry) is None

    assert vector.declarations == [False]
    assert vector.created is False
    assert vector.landed == []
    assert _outbox_slugs(manager.store) == {"existing", "newcomer"}

    err = capsys.readouterr().err
    # Named as itself (the collection is missing), never as an outage, and carrying the
    # heal. The channel is stderr because this path is shared with the MCP write tool,
    # whose stdout is the JSON-RPC transport.
    assert "collection is missing" in err
    assert "mitos reconcile" in err
    assert "down" not in err.lower()


def test_the_latch_prints_once_and_spends_nothing_further(tmp_path, capsys) -> None:
    """A multi-entry sync reports the deferral ONCE and skips N−1 round trips.

    Per-node noise on a state that is a property of the whole command is the wrong shape,
    and each skipped attempt also saves an embedding call — deferral costs no extra
    provider spend at all.
    """
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    _decision(store, "existing", "An axiom already in the graph.")

    vector = _RecordingVector(absent=True)
    manager = _manager(config, vector)

    for index in range(4):
        entry = _entry(f"newcomer-{index}", f"Axiom {index}.")
        delta = manager.store.commit_parsed_entry(entry)
        manager._best_effort_embed(delta, entry)

    # One attempt, then the latch: no further upserts and no further embeddings.
    assert vector.declarations == [False]
    assert manager.embed_provider.calls == 1  # type: ignore[union-attr]
    err = capsys.readouterr().err
    assert err.count("collection is missing") == 1
    # Every node stayed queued — four newcomers plus the pre-existing one.
    assert len(manager.store.get_pending_embeddings()) == 5


def test_the_latch_is_per_manager_and_never_read_by_the_drain(tmp_path) -> None:
    """D4's boundary: a covering drain later in the same command must still create.

    If the drain consulted the inline latch, ``rebuild`` → ``sync`` would refuse to heal
    whenever an earlier entry in that same sync had met the absent collection — the phase
    breaking its own headline promise through its own suppression mechanism.
    """
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    _decision(store, "existing", "An axiom already in the graph.")

    vector = _RecordingVector(absent=True)
    manager = _manager(config, vector)

    entry = _entry("newcomer", "A newly recorded axiom.")
    delta = manager.store.commit_parsed_entry(entry)
    manager._best_effort_embed(delta, entry)
    assert manager._collection_absent is True

    # A prior command bounded the outbox to the active set; the drain is covering.
    manager.store.stamp_embedding_seed("rebuild")
    manager.drain_pending_embeddings()

    assert vector.created is True
    assert manager.store.get_pending_embeddings() == []


def test_the_fixpoint_retry_path_takes_the_same_defer_branch(tmp_path, capsys) -> None:
    """Success criterion 13: the third reach site, which nothing else exercises.

    A forward-ref entry commits only on a fixpoint retry pass, and its embed arrives
    through the ``embed_fn=`` callback injected into ``replay.commit_quarantine_fixpoint``
    — not through either of the two direct call sites. Because the declaration lives in
    the shared helper, this path inherits it for free; the row exists to prove the
    inheritance rather than to assume it.
    """
    from mitos.replay import commit_quarantine_fixpoint

    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    _decision(store, "existing", "An axiom already in the graph.")

    vector = _RecordingVector(absent=True)
    manager = _manager(config, vector)

    # A quarantined entry whose target now exists, so the retry pass commits it. The
    # edge is ``amends`` rather than ``supersedes`` on purpose: a kill-edge would retire
    # ``existing`` in the same transaction, leaving the new node as the whole active set
    # and (correctly) declaring covering — which would test the carve-out instead of the
    # defer branch this row is about.
    forward = _entry("late-arrival", "An axiom that could not commit first time.")
    forward.amends = ["existing"]
    committed, _passes, residual = commit_quarantine_fixpoint(
        manager.store,
        [(forward, "", None)],  # type: ignore[list-item]
        embed_fn=manager._best_effort_embed,
    )

    assert committed == 1 and residual == []
    assert vector.declarations == [False]
    assert vector.created is False
    assert "collection is missing" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# The consequence the phase exists for: the honest notice keeps firing
# --------------------------------------------------------------------------- #

class _AbsenceAwareVector(_RecordingVector):
    """A fake whose READS fail while the collection is absent, exactly as Qdrant's do.

    ``_RecordingVector`` only models the write. This models the pair, which is the only
    way to observe the actual hazard: the record notice is raised by the conflict
    *gather* (a read), so whether it fires on the **second** record depends entirely on
    whether the **first** record's write created the collection.
    """

    def query(self, vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        if self.absent and not self.created:
            raise CollectionMissingError(
                "Qdrant collection 'mitos-tmp-fake' does not exist",
                collection="mitos-tmp-fake",
            )
        return []


@pytest.mark.parametrize("gate_forced_open", [False, True])
def test_the_honest_notice_still_fires_on_the_second_record(
    tmp_path, monkeypatch, gate_forced_open: bool
) -> None:
    """Success criterion 1, in the shape of the inversion it prevents.

    ``"couldn't check"`` must not quietly become ``"checked, clean"``. Under the shipped
    gate the first record's upsert defers, the collection stays absent, and the *second*
    record's gather still meets absence and still carries
    ``neighbor_review_unavailable`` — the notice fires every time, which is what makes it
    trustworthy.

    The forced-open arm is the counterfactual, and it is why this row bites: with the gate
    pinned open the first record creates a collection holding **one point out of the
    corpus's N**, so the second record's gather *succeeds* against a near-empty index,
    nothing clears the 0.80 conflict floor, and the receipt is clean. Same fixture, same
    corpus, opposite verdict — the difference is only the gate.
    """
    monkeypatch.setattr(
        GraphStore,
        "has_active_node_other_than",
        lambda self, node_id: not gate_forced_open,
    )

    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    for index in range(3):
        _decision(store, f"existing-{index}", f"An axiom already in the graph {index}.")

    vector = _AbsenceAwareVector(absent=True)
    manager = _manager(config, vector)

    first = manager.record_decision_entry(
        "The first newly recorded axiom.", "None.", ["s"], slug="newcomer-one"
    )
    # Record 1 always carries the notice — absence was real when it read.
    assert first["status"] == "created"
    assert "neighbor_review_unavailable" in first
    assert "mitos reconcile" in first["neighbor_review_unavailable"]

    # The state that decides record 2's answer: how much of the corpus the index holds
    # after exactly one write. Four active nodes; one point, or none.
    assert len(manager.store.get_active_node_ids()) == 4
    points_after_one_write = len(vector.landed)
    assert points_after_one_write == (1 if gate_forced_open else 0)

    second = manager.record_decision_entry(
        "The second newly recorded axiom.", "None.", ["s"], slug="newcomer-two"
    )
    assert second["status"] == "created"

    if gate_forced_open:
        # The hazard, reproduced: one write spent the signal, so record 2's gather
        # succeeded against an index holding one point out of four active nodes and the
        # receipt came back clean. "Couldn't check" became "checked, clean".
        assert vector.created is True
        assert "neighbor_review_unavailable" not in second
    else:
        # The shipped behaviour: absence is durable, so the notice is too.
        assert vector.created is False
        assert vector.landed == []
        assert "neighbor_review_unavailable" in second
        assert "mitos reconcile" in second["neighbor_review_unavailable"]


# --------------------------------------------------------------------------- #
# The drain — declares from the marker
# --------------------------------------------------------------------------- #

def test_a_drain_with_no_marker_refuses_once_and_leaves_retry_counts_alone(
    tmp_path, capsys
) -> None:
    """Success criterion 8: one line, an immediate stop, and no counter poisoned.

    ``retry_count`` exists to describe genuine failure. The row was not rejected — the
    drain declined — so inflating it would make an ordinary deferral look like a node
    that keeps failing to embed, and would eventually mislead whoever reads the outbox.
    """
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    for index in range(3):
        _decision(store, f"queued-{index}", f"Axiom {index}.")

    before = {row["node_id"]: row["retry_count"] for row in store.get_pending_embeddings()}
    assert len(before) == 3

    vector = _RecordingVector(absent=True)
    manager = _manager(config, vector)
    assert manager.store.embedding_seed() is None

    manager.drain_pending_embeddings()

    # One attempt then a hard stop — not one attempt per row.
    assert vector.declarations == [False]
    assert vector.created is False
    out = capsys.readouterr().out
    assert out.count("Stopping the drain") == 1
    assert "mitos reconcile" in out

    after = {row["node_id"]: row["retry_count"] for row in store.get_pending_embeddings()}
    assert after == before  # nothing left the outbox, nothing was penalised


def test_a_marked_drain_creates_and_clears_the_marker(tmp_path) -> None:
    """Success criterion 9, first half: coverage delivered, so the claim is spent.

    Leaving the marker standing would let a later, genuinely non-covering drain create —
    the marker authorizing exactly what it exists to prevent.
    """
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    for index in range(3):
        _decision(store, f"queued-{index}", f"Axiom {index}.")

    vector = _RecordingVector(absent=True)
    manager = _manager(config, vector)
    manager.store.stamp_embedding_seed("rebuild")

    manager.drain_pending_embeddings()

    assert vector.declarations == [True, True, True]
    assert vector.created is True
    assert len(vector.landed) == 3
    assert manager.store.get_pending_embeddings() == []
    assert manager.store.embedding_seed() is None  # cleared on drain-to-empty


def test_a_drain_that_makes_no_progress_retains_the_marker(tmp_path) -> None:
    """Success criterion 9, second half: rows remain, so the claim still holds.

    The zero-resolving progress guard exists for a dead embedding provider. Clearing the
    marker there would punish an outage by making the *next* drain non-covering — the heal
    would then defer forever, with nothing naming why.
    """
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    _decision(store, "queued", "An axiom.")

    class _DeadEmbed:
        def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
            raise RuntimeError("embedding provider 429")

    vector = _RecordingVector()
    manager = _manager(config, vector)
    manager.embed_provider = _DeadEmbed()  # type: ignore[assignment]
    manager.store.stamp_embedding_seed("reconcile")

    manager.drain_pending_embeddings()

    assert len(manager.store.get_pending_embeddings()) == 1
    seed = manager.store.embedding_seed()
    assert seed is not None and seed["established_by"] == "reconcile"


def test_a_refused_drain_retains_the_marker_it_never_consumed(tmp_path) -> None:
    """A refusal with a marker standing is a contradiction, so this pins the pairing.

    The only way here is a marker written while the collection was reachable and a
    collection that vanished before the drain ran. Rows remain, so the marker stays — the
    next drain is still covering and completes the heal.
    """
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    for index in range(2):
        _decision(store, f"queued-{index}", f"Axiom {index}.")

    class _AlwaysAbsentVector(_RecordingVector):
        def upsert(self, point_id: str, vector: List[float], payload: Dict[str, Any],
                   *, may_create: bool) -> None:
            self.declarations.append(may_create)
            raise CollectionMissingError(
                "Qdrant collection 'mitos-tmp-fake' does not exist",
                collection="mitos-tmp-fake",
            )

    vector = _AlwaysAbsentVector()
    manager = _manager(config, vector)
    manager.store.stamp_embedding_seed("rebuild")

    manager.drain_pending_embeddings()

    assert vector.declarations == [True]  # stopped on the first row, not per row
    assert len(manager.store.get_pending_embeddings()) == 2
    assert manager.store.embedding_seed() is not None


def test_an_already_empty_outbox_clears_a_standing_marker(tmp_path) -> None:
    """The marker's claim is about the outbox, so an empty outbox spends it.

    Reached when a project's active set was empty at ``rebuild`` time: the prune stamps
    over a queue that is already empty. Leaving the marker armed indefinitely would make
    it a standing creation licence rather than a record of pending coverage.
    """
    config = _workspace(tmp_path)
    vector = _RecordingVector()
    manager = _manager(config, vector)
    manager.store.stamp_embedding_seed("cutover")

    manager.drain_pending_embeddings()

    assert vector.declarations == []
    assert manager.store.embedding_seed() is None


def test_the_marker_never_authorizes_an_inline_write(tmp_path) -> None:
    """Gotcha 1: ``rebuild`` → ``record`` (no sync) must create nothing.

    The marker's whole safety rests on having exactly one consumer. If a ``record``
    consulted it after a ``rebuild``, the first record into a swept project would mint a
    one-point collection — the precise hazard this phase closes, reintroduced through its
    own mechanism.
    """
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    _decision(store, "existing", "An axiom already in the graph.")

    vector = _RecordingVector(absent=True)
    manager = _manager(config, vector)
    manager.store.stamp_embedding_seed("rebuild")  # as `mitos rebuild` would leave it

    entry = _entry("newcomer", "A newly recorded axiom.")
    delta = manager.store.commit_parsed_entry(entry)
    manager._best_effort_embed(delta, entry)

    assert vector.declarations == [False]
    assert vector.created is False
    # And the marker is untouched — the inline path neither reads nor spends it.
    assert manager.store.embedding_seed() is not None


# --------------------------------------------------------------------------- #
# reconcile — stamps so its own drain covers
# --------------------------------------------------------------------------- #

def test_reconcile_against_an_absent_collection_creates_the_whole_active_set(
    tmp_path,
) -> None:
    """The heal in one pass, and the stamp is what makes its drain a covering write.

    Without the stamp the drain would declare ``False``, refuse, and ``mitos reconcile``
    would become a command that cannot reconcile the one state it exists for.
    """
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    ids = [_decision(store, f"node-{index}", f"Axiom {index}.") for index in range(3)]
    for row in store.get_pending_embeddings():
        store.remove_pending_embedding(row["node_id"])

    vector = _RecordingVector(absent=True)
    manager = _manager(config, vector)

    result = manager.reconcile_embeddings()

    assert result == {"active": 3, "present": 0, "enqueued": 3}
    assert vector.created is True
    assert set(vector.landed) == set(ids)
    assert manager.store.get_pending_embeddings() == []
    # Coverage delivered, so the marker the pass wrote is gone again.
    assert manager.store.embedding_seed() is None


def test_a_deferred_record_is_reconciled_leaving_no_orphan_outbox_row(
    tmp_path, capsys
) -> None:
    """The round trip: defer → heal → nothing left over, and nothing indexed twice.

    Deferral is only cheap if the queued row is genuinely the work item. So the sequence
    that matters is the whole one: a record defers and keeps its row, the operator runs
    ``reconcile``, and afterwards the index holds every active node exactly once and the
    outbox is empty — no orphan row, no duplicate point, no second embedding paid for the
    node that deferred.

    The same read that made record 2's notice fire is the operation ``check``'s
    fail-closed precondition keys on (1b re-keyed it off store construction), so this row
    also pins that the operator is not handed a green check in between: the collection is
    still absent until the heal runs.
    """
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    _decision(store, "existing", "An axiom already in the graph.")
    for row in store.get_pending_embeddings():
        store.remove_pending_embedding(row["node_id"])

    vector = _AbsenceAwareVector(absent=True)
    manager = _manager(config, vector)

    entry = _entry("newcomer", "A newly recorded axiom.")
    delta = manager.store.commit_parsed_entry(entry)
    manager._best_effort_embed(delta, entry)

    # Deferred: the row stands, and reads still meet absence (check stays fail-closed).
    assert _outbox_slugs(manager.store) == {"newcomer"}
    with pytest.raises(CollectionMissingError):
        vector.query([0.1, 0.2, 0.3])

    result = manager.reconcile_embeddings()

    assert result == {"active": 2, "present": 0, "enqueued": 2}
    assert vector.created is True
    # Every active node indexed exactly once — the deferred node was not double-written.
    assert sorted(vector.landed) == sorted(manager.store.get_active_node_ids())
    assert len(vector.landed) == len(set(vector.landed))
    assert manager.store.get_pending_embeddings() == []
    assert manager.store.embedding_seed() is None


def test_reconcile_stamps_nothing_when_the_index_is_already_complete(tmp_path) -> None:
    """Nothing missing means no coverage claim to make — and no drain to authorize."""
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    _decision(store, "node", "An axiom.")
    for row in store.get_pending_embeddings():
        store.remove_pending_embedding(row["node_id"])

    from mitos.vector_store import hash_to_uuid

    node_id = store.get_active_node_ids().pop()

    class _CompleteVector(_RecordingVector):
        def list_point_ids(self, page_size: int = 256) -> Set[str]:
            return {hash_to_uuid(node_id)}

    vector = _CompleteVector()
    manager = _manager(config, vector)

    assert manager.reconcile_embeddings() == {"active": 1, "present": 1, "enqueued": 0}
    assert vector.declarations == []
    assert manager.store.embedding_seed() is None


# --------------------------------------------------------------------------- #
# import — the one site that ADDS a defer path
# --------------------------------------------------------------------------- #

def test_import_into_a_populated_graph_defers_with_one_line(tmp_path, capsys) -> None:
    """Success criterion 7: creates nothing, prints ONE line, leaves N outbox rows.

    Leaving ``import`` out would let it mint exactly the partial collection the clause
    exists to prevent, one entry deep. And the notice is the substance of the added path:
    the importer's bare ``except Exception: pass`` already swallows the refusal (a
    ``CollectionMissingError`` is a ``VectorStoreError`` is an ``Exception``), and it never
    dequeues, so durability came free — silence did not.
    """
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    _decision(store, "existing", "An axiom already in the graph.")

    importer = MitosProseImporter(config)
    importer.embed_provider = _FakeEmbed()  # type: ignore[assignment]
    vector = _RecordingVector(absent=True)
    importer.vector_store = vector  # type: ignore[assignment]

    for index in range(3):
        entry = _entry(f"imported-{index}", f"Imported axiom {index}.")
        delta = importer.store.commit_parsed_entry(entry)
        importer._best_effort_embed(delta, entry)

    assert vector.declarations == [False]  # the latch: one attempt for the whole import
    assert vector.created is False
    out = capsys.readouterr().out
    assert out.count("Embeddings deferred for this import") == 1
    assert "mitos reconcile" in out
    # Every imported node plus the pre-existing one is still queued: the importer never
    # dequeues, so its rows survive by default.
    assert len(importer.store.get_pending_embeddings()) == 4


def test_import_into_a_fresh_project_still_creates_on_its_first_entry(tmp_path) -> None:
    """The carve-out reaches ``import`` too: entry 1 covers, and 2..N land normally.

    Worth pinning because it is where the 404-only scoping shows: entries after the first
    declare ``False`` and still succeed, because by then the collection exists and the
    declaration is never read.
    """
    config = _workspace(tmp_path)
    importer = MitosProseImporter(config)
    importer.embed_provider = _FakeEmbed()  # type: ignore[assignment]
    vector = _RecordingVector(absent=True)
    importer.vector_store = vector  # type: ignore[assignment]

    for index in range(3):
        entry = _entry(f"imported-{index}", f"Imported axiom {index}.")
        delta = importer.store.commit_parsed_entry(entry)
        importer._best_effort_embed(delta, entry)

    assert vector.declarations == [True, False, False]
    assert vector.created is True
    assert len(vector.landed) == 3


# --------------------------------------------------------------------------- #
# Force the predicate to a constant (PATTERNS)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("forced", [True, False])
def test_forcing_the_gate_to_a_constant_reds_a_different_set_of_rows(
    tmp_path, monkeypatch, forced: bool
) -> None:
    """The twin-fixture proof: each constant breaks a DIFFERENT expectation.

    A gate whose quiet branch only fires in a state a developer never reaches by hand
    (populated graph **and** absent collection) is exactly where a one-sided fixture
    passes under either behaviour. Pinning the predicate to each constant and confirming
    the fresh-project and populated-graph cases disagree converts "the twin exists" into
    "the twin bites": under a forced ``True`` the populated graph would create, and under a
    forced ``False`` the fresh project would not.
    """
    monkeypatch.setattr(
        GraphStore, "has_active_node_other_than", lambda self, node_id: not forced
    )

    # Fresh project.
    fresh_config = _workspace(tmp_path, "fresh")
    fresh_vector = _RecordingVector(absent=True)
    fresh_manager = _manager(fresh_config, fresh_vector)
    fresh_entry = _entry("first", "The first axiom.")
    fresh_manager._best_effort_embed(
        fresh_manager.store.commit_parsed_entry(fresh_entry), fresh_entry
    )

    # Populated graph.
    full_config = _workspace(tmp_path, "full")
    full_store = GraphStore(full_config.db_path)
    _decision(full_store, "existing", "An axiom already in the graph.")
    full_vector = _RecordingVector(absent=True)
    full_manager = _manager(full_config, full_vector)
    full_entry = _entry("newcomer", "A newly recorded axiom.")
    full_manager._best_effort_embed(
        full_manager.store.commit_parsed_entry(full_entry), full_entry
    )

    # Both sides follow the FORCED predicate, not the graph — which is what proves the
    # real behaviour in the rows above came from the gate and not from the fixture.
    assert fresh_vector.created is forced
    assert full_vector.created is forced
