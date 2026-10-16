"""The rejection backlog's drain: one at a time, whole-set replacement, convergence.

Recall-confidence phase 4c. ``MitosSyncManager.drain_rejection_backlog`` is the only
writer that removes from the rejection index. It runs one at a time per workspace
under a non-blocking OS lock of its own, walks the backlog once in ``enqueue_seq``
order, and per owner reads the seq (the claim) and only then the field, lists what the
owner holds, writes what it lacks, removes what its field no longer states, and clears
against the seq it read. A clear that answers False is a re-queue; the same walk takes
the owner again and converges it on the latest field. Batches cross owners; an outage
stops the walk and charges no row; a failing owner costs only its own row. Only the
rejection seed marker licenses creating the collection.

The rows drive ``cli.main`` for ``sync``, ``sync --embed-only`` and ``reconcile`` and
the manager method for the walk's internals, with the three constructors bound at
``mitos.sync``: a recording provider (never a MagicMock), a decision vector store, and
``FakeRejectionIndex``. Rows are queued through ``commit_parsed_entry``, which enqueues
and never embeds, so no inline leg runs unless a row asks for one. A real
``GraphStore`` in ``tmp_path`` throughout.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import pytest
from filelock import FileLock

from mitos import cli
from mitos.errors import EmbeddingError, VectorStoreError, VectorStoreUnreachableError
from mitos.parser import ParsedEntry
from mitos.rejections import rejection_points
from mitos.store import REJECTION_BACKLOG_FAILURE_TOLERANCE, GraphStore
from mitos.sync import (
    DRAIN_MARKER_KEPT,
    DRAIN_MARKER_SPENT,
    DRAIN_STOP_CLAIM_FAILED,
    DRAIN_STOP_OUTAGE,
    DRAIN_STOP_REFUSED,
    DRAIN_STOP_UNAVAILABLE,
    DRAIN_STOPS,
    REJECTION_DRAIN_STOP_HELD,
    REJECTION_DRAIN_STOPS,
    MitosSyncManager,
    RejectionDrainOutcome,
    rejection_drain_lock_path,
    rejection_drain_report,
)
from mitos.vector_store import hash_to_uuid

from _rejection_index_double import FakeRejectionIndex
from _stdio_fake_providers import FAULT_FIELDS, FakeVectorStore, _SdkLikeError, fake_vector
from test_cli_selector import _run
from test_corpus_provenance import _init_workspace
from test_rejection_inline import MODEL, RecordingProvider, Rig, _append_entry, _bind

DEAD_QDRANT_URL = "http://127.0.0.1:9"
POISON = "POISON"
F_ABC = "1. Alpha — a.\n2. Bravo — b.\n3. Charlie — c."
F_BD = "1. Bravo — b.\n2. Delta — d."
F_ONE = "1. Echo — e."
REFUSAL_PREFIX = "Rejection index: "


# --------------------------------------------------------------------------- #
# Doubles and the rig
# --------------------------------------------------------------------------- #


def _sdk_error(cause: str) -> EmbeddingError:
    body = f"planted {cause}"
    try:
        raise _SdkLikeError(body, **FAULT_FIELDS[cause])
    except _SdkLikeError as sdk:
        error = EmbeddingError(f"Gemini embedding API call failed: {body}")
        error.__cause__ = sdk
        return error


class DrainProvider(RecordingProvider):
    """``RecordingProvider``, plus a batch carrying ``POISON`` fails ``unknown``."""

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if self.fault is None and any(POISON in text for text in texts):
            self.document_calls.append(list(texts))
            raise _sdk_error("unknown")
        return super().embed_documents(texts)


def _decision_store_class() -> type:
    """A per-test decision vector store that can also list (reconcile needs it)."""

    def list_point_ids(self, page_size: int = 256) -> Set[str]:
        from mitos.errors import CollectionMissingError

        points = self._collections.get(self.collection)
        if points is None:
            raise CollectionMissingError("missing", collection=self.collection)
        return {hash_to_uuid(node_id) for node_id in points}

    return type("DecisionStore", (FakeVectorStore,),
                {"_collections": {}, "list_point_ids": list_point_ids})


def _rig(tmp_path, monkeypatch, *, present: bool = True, name: str = "ws") -> Rig:
    monkeypatch.setenv("QDRANT_URL", DEAD_QDRANT_URL)
    monkeypatch.chdir(tmp_path)
    root = _init_workspace(tmp_path / name)
    made = Rig(root, DrainProvider(), FakeRejectionIndex(present=present),
               _decision_store_class())
    _bind(monkeypatch, made)
    return made


@pytest.fixture
def d(tmp_path, monkeypatch) -> Rig:
    return _rig(tmp_path, monkeypatch)


def _commit(d: Rig, slug: str, field: str) -> str:
    """Commits (or re-commits) a decision through the store: queues its row, embeds nothing."""
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = f"The {slug} axiom stands on its own words."
    entry.rejected_paths = field
    return d.store().commit_parsed_entry(entry).node_id


def _hold(d: Rig, owner: str, field: str) -> None:
    """Puts an owner's items in the index, as an earlier write would have."""
    for pid in rejection_points(owner, field):
        d.index.points[pid] = {"vector": fake_vector(pid),
                               "payload": {"owner_id": owner, "model": MODEL}}


def _ids(owner: str, field: str) -> Set[str]:
    return set(rejection_points(owner, field))


def _texts(owner: str, field: str) -> List[str]:
    return [ref.text for ref in rejection_points(owner, field).values()]


def _held(d: Rig, owner: str) -> Set[str]:
    return {pid for pid, p in d.index.points.items() if p["payload"]["owner_id"] == owner}


def _drain(d: Rig) -> RejectionDrainOutcome:
    return d.manager().drain_rejection_backlog()


def _retry(d: Rig, owner: str) -> Optional[int]:
    row = d.row(owner)
    return None if row is None else row.retry_count


def _stamp(d: Rig) -> None:
    d.store().stamp_rejection_seed("reconcile")


def _ops(d: Rig) -> List[str]:
    return [name for name, _ in d.index.calls]


def _summary(out: str) -> List[str]:
    return [line for line in out.splitlines() if line.startswith(REFUSAL_PREFIX)]


def _outcome(**fields: Any) -> RejectionDrainOutcome:
    base = dict(owners_cleared=0, items_embedded=0, items_skipped_present=0,
                items_removed=0, owners_failed=0, owners_past_tolerance=0, calls=0,
                left_queued=0, stop=None, marker=None)
    base.update(fields)
    return RejectionDrainOutcome(**base)


def _once(method: str, action: Callable[[Dict[str, Any]], None],
          when: Callable[[Dict[str, Any]], bool] = lambda details: True):
    """An ``on_call`` hook that fires ``action`` on the first matching call only."""
    fired: List[bool] = []

    def hook(name: str, details: Dict[str, Any]) -> None:
        if not fired and name == method and when(details):
            fired.append(True)
            action(details)

    return hook


# --------------------------------------------------------------------------- #
# The vocabulary
# --------------------------------------------------------------------------- #


def test_the_stops_extend_the_decision_drains_without_widening_them() -> None:
    assert REJECTION_DRAIN_STOP_HELD == "held"
    assert REJECTION_DRAIN_STOPS == DRAIN_STOPS + ("held",)
    assert "held" not in DRAIN_STOPS
    assert REJECTION_BACKLOG_FAILURE_TOLERANCE == 3


def test_to_dict_carries_every_field_and_round_trips_through_json() -> None:
    outcome = _outcome(owners_cleared=2, left_queued=None, owners_past_tolerance=None,
                       stop=DRAIN_STOP_OUTAGE, marker=DRAIN_MARKER_KEPT)
    data = json.loads(json.dumps(outcome.to_dict()))
    assert data == outcome.to_dict()
    assert RejectionDrainOutcome(**data) == outcome
    assert list(data) == [
        "owners_cleared", "items_embedded", "items_skipped_present", "items_removed",
        "owners_failed", "owners_past_tolerance", "calls", "left_queued", "stop",
        "marker",
    ]


# --------------------------------------------------------------------------- #
# R-replace / R-order — whole-set replacement, write before remove
# --------------------------------------------------------------------------- #


def test_r_replace_writes_only_the_new_item_then_removes_the_dropped_ones(d) -> None:
    owner = _commit(d, "amended", F_ABC)
    _hold(d, owner, F_ABC)
    other = _commit(d, "untouched", F_ONE)
    _hold(d, other, F_ONE)
    d.store().clear_rejection_backlog_row(other, enqueue_seq=d.row(other).enqueue_seq)
    _commit(d, "amended", F_BD)  # the amend: a fresh seq, the field B, D

    outcome = _drain(d)

    assert outcome == _outcome(owners_cleared=1, items_embedded=1,
                               items_skipped_present=1, items_removed=2, calls=1)
    assert d.provider.document_calls == [["Delta — d."]]
    assert _held(d, owner) == _ids(owner, F_BD)
    assert _held(d, other) == _ids(other, F_ONE)
    assert d.row(owner) is None
    removed = [details["ids"] for name, details in d.index.calls if name == "delete_points"]
    assert removed == [sorted(_ids(owner, F_ABC) - _ids(owner, F_BD))]


def test_r_order_every_write_precedes_its_owners_delete(d) -> None:
    owners = []
    for slug in ("one", "two", "three"):
        owner = _commit(d, slug, F_ABC)
        _hold(d, owner, F_ABC)
        owners.append(owner)
    for slug in ("one", "two", "three"):
        _commit(d, slug, F_BD)
    poisoned = _commit(d, "poisoned", F_ABC)
    _hold(d, poisoned, F_ABC)
    _commit(d, "poisoned", f"1. Bravo — b.\n2. {POISON} — p.")

    _drain(d)

    for owner in [*owners, poisoned]:
        mine = _ids(owner, F_ABC) | _ids(owner, F_BD) | set(rejection_points(
            owner, f"1. Bravo — b.\n2. {POISON} — p."))
        writes = [i for i, (name, det) in enumerate(d.index.calls)
                  if name == "upsert_points" and set(det["ids"]) & mine]
        deletes = [i for i, (name, det) in enumerate(d.index.calls)
                   if name == "delete_points" and set(det["ids"]) & mine]
        if owner == poisoned:
            assert writes == [] and deletes == []
        else:
            assert writes and deletes and max(writes) < min(deletes)
    assert _retry(d, poisoned) == 1


def test_r_killed_between_leaves_old_and_new_and_the_next_drain_finishes(d) -> None:
    owner = _commit(d, "amended", F_ABC)
    _hold(d, owner, F_ABC)
    _commit(d, "amended", F_BD)
    seq = d.row(owner).enqueue_seq

    def interrupt(details: Dict[str, Any]) -> None:
        raise KeyboardInterrupt

    d.index.on_call = _once("delete_points", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _drain(d)

    # Both sets present (never none), the row untouched, the lock released.
    assert _held(d, owner) == _ids(owner, F_ABC) | _ids(owner, F_BD)
    assert d.row(owner).enqueue_seq == seq and _retry(d, owner) == 0
    probe = FileLock(rejection_drain_lock_path(d.config), timeout=0)
    probe.acquire()
    probe.release()

    d.index.on_call = None
    outcome = _drain(d)
    assert outcome.owners_cleared == 1 and outcome.calls == 0
    assert _held(d, owner) == _ids(owner, F_BD)
    assert d.row(owner) is None


# --------------------------------------------------------------------------- #
# R-requeue — convergence on the latest field
# --------------------------------------------------------------------------- #

F_NEW = "1. Bravo — b.\n2. Foxtrot — f."


@pytest.mark.parametrize("latest", [F_NEW, F_ABC], ids=["new-items", "revert"])
def test_r_requeue_mid_replacement_converges_in_the_same_drain(d, latest) -> None:
    """A real amend (its inline write runs) lands after the listing, before the delete.

    In the ``revert`` case the amend restores the first field, so the delete then
    removes items the latest field states again; the retake re-adds them.
    """
    from test_rejection_inline import _mcp_amend, _record

    owner = _record(d.manager(), "first", F_ABC)["id"]  # licensed: creates, clears
    assert d.row(owner) is None
    _mcp_amend(d, "first", rejected_paths=F_BD)  # inline adds Delta; the row stands
    assert _held(d, owner) == _ids(owner, F_ABC) | _ids(owner, F_BD)
    d.index.calls.clear()
    d.provider.document_calls.clear()

    d.index.on_call = _once(
        "delete_points", lambda details: _mcp_amend(d, "first", rejected_paths=latest))
    outcome = _drain(d)

    assert outcome.stop is None
    assert _held(d, owner) == _ids(owner, latest)
    assert d.row(owner) is None
    relists = [det for name, det in d.index.calls
               if name == "list_owner_point_ids" and det["owner_id"] == owner]
    assert len(relists) == 2  # the first take and the retake


@pytest.mark.parametrize("when", ["before", "after"])
def test_r_requeue_around_the_field_read_converges(d, monkeypatch, when) -> None:
    """D2: the seq is read first. A commit after it, around the field read, re-queues
    the owner, so the take's clear fails and the retake reads the newer field."""
    owner = _commit(d, "amended", F_ABC)
    _hold(d, owner, F_ABC)
    _commit(d, "amended", F_BD)
    real = GraphStore.get_rejection_owner_fields
    fired: List[bool] = []

    def read(self, node_ids):
        if not fired and when == "before":
            fired.append(True)
            _commit(d, "amended", F_NEW)
        answer = real(self, node_ids)
        if not fired and when == "after":
            fired.append(True)
            _commit(d, "amended", F_NEW)
        return answer

    monkeypatch.setattr(GraphStore, "get_rejection_owner_fields", read)
    outcome = _drain(d)

    assert outcome.stop is None
    assert _held(d, owner) == _ids(owner, F_NEW)
    assert d.row(owner) is None


# --------------------------------------------------------------------------- #
# R-second-drain / R-killed-holding-lock — one at a time, dies with its process
# --------------------------------------------------------------------------- #


def _bounded(call: Callable[[], Any], seconds: float = 10.0) -> Any:
    """Runs ``call`` on a daemon thread and fails the row if it does not return."""
    box: Dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = call()
        except BaseException as exc:  # pragma: no cover - surfaced below
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), f"the call did not return within {seconds}s"
    if "error" in box:
        raise box["error"]
    return box["value"]


def test_r_second_drain_leaves_the_backlog_to_the_holder(d, capsys) -> None:
    owner = _commit(d, "queued", F_ABC)
    seq = d.row(owner).enqueue_seq
    holder = FileLock(rejection_drain_lock_path(d.config), timeout=0)
    holder.acquire()
    try:
        capsys.readouterr()
        outcome = _bounded(lambda: _drain(d))
        assert outcome == RejectionDrainOutcome(0, 0, 0, 0, 0, None, 0, None,
                                                stop="held", marker=None)
        assert d.provider.document_calls == [] and d.index.calls == []
        assert d.row(owner).enqueue_seq == seq and _retry(d, owner) == 0
        assert "another drain holds the rejection backlog" in capsys.readouterr().out

        code = _bounded(lambda: _run(["-p", d.root, "sync", "--embed-only"]))
        assert code in (0, None)
        assert "another drain holds the rejection backlog" in capsys.readouterr().out
        assert d.index.calls == [] and d.row(owner).enqueue_seq == seq
    finally:
        holder.release()


def test_r_the_drain_never_takes_the_buffer_lock(d) -> None:
    """A held ``decisions.md`` lock does not hold the drain (it is not its lock)."""
    owner = _commit(d, "queued", F_ONE)
    buffer_lock = FileLock(d.config.decisions_file + ".lock", timeout=0)
    buffer_lock.acquire()
    try:
        outcome = _bounded(lambda: _drain(d))
    finally:
        buffer_lock.release()
    assert outcome.stop is None and outcome.owners_cleared == 1
    assert d.row(owner) is None


def test_r_killed_holding_lock_frees_it_with_no_operator_step(d) -> None:
    owner = _commit(d, "queued", F_ONE)
    sentinel = os.path.join(os.path.dirname(d.root), "held.sentinel")
    child = textwrap_child(d.root, sentinel)
    proc = subprocess.Popen([sys.executable, "-c", child],
                            cwd=os.path.dirname(d.root))
    try:
        deadline = time.monotonic() + 30
        while not os.path.exists(sentinel):
            assert proc.poll() is None, "the child died before holding the lock"
            assert time.monotonic() < deadline, "the child never held the lock"
            time.sleep(0.05)
        assert _drain(d).stop == "held"  # it really holds it
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    # A killed holder may leave the file; the kernel dropped the lock with it.
    outcome = _drain(d)
    assert outcome.stop is None and outcome.owners_cleared == 1
    assert d.row(owner) is None and _held(d, owner) == _ids(owner, F_ONE)


def textwrap_child(root: str, sentinel: str) -> str:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return (
        "import sys, time\n"
        f"sys.path.insert(0, {repo!r})\n"
        "from filelock import FileLock\n"
        "from mitos.config import MitosConfig\n"
        "from mitos.sync import rejection_drain_lock_path\n"
        f"lock = FileLock(rejection_drain_lock_path(MitosConfig({root!r})), timeout=0)\n"
        "lock.acquire()\n"
        f"open({sentinel!r}, 'w').close()\n"
        "time.sleep(120)\n"
    )


def test_g1_a_lock_that_cannot_be_taken_is_unavailable_not_held(d, monkeypatch, capsys) -> None:
    owner = _commit(d, "queued", F_ONE)
    blocker = os.path.join(os.path.dirname(d.root), "a-file")
    open(blocker, "w").close()
    monkeypatch.setattr("mitos.sync.rejection_drain_lock_path",
                        lambda config: os.path.join(blocker, "rejection_drain.lock"))
    capsys.readouterr()
    outcome = _drain(d)
    assert outcome.stop == DRAIN_STOP_UNAVAILABLE and outcome.left_queued == 1
    assert "lock could not be taken" in capsys.readouterr().out
    assert d.index.calls == [] and _retry(d, owner) == 0


def test_an_escaped_fault_is_an_outcome_and_releases_the_lock(d, monkeypatch, capsys) -> None:
    owner = _commit(d, "queued", F_ONE)

    def boom(self):
        raise RuntimeError("planted walk fault")

    monkeypatch.setattr("mitos.sync._RejectionDrainWalk.run", boom)
    outcome = _drain(d)
    assert outcome.stop == DRAIN_STOP_UNAVAILABLE and outcome.left_queued == 1
    assert "planted walk fault" in capsys.readouterr().out
    probe = FileLock(rejection_drain_lock_path(d.config), timeout=0)
    probe.acquire()
    probe.release()
    assert _retry(d, owner) == 0


def test_an_unbound_manager_is_silent_and_unavailable(d, capsys) -> None:
    _commit(d, "queued", F_ONE)
    m = d.manager()
    m.rejection_index = None
    capsys.readouterr()
    outcome = m.drain_rejection_backlog()
    assert outcome == _outcome(left_queued=1, stop=DRAIN_STOP_UNAVAILABLE)
    assert capsys.readouterr() == ("", "")
    assert not os.path.exists(rejection_drain_lock_path(d.config))


@pytest.mark.parametrize("model", ["", None], ids=["empty", "none"])
def test_a_provider_naming_no_model_charges_no_row(d, capsys, model) -> None:
    """The provider's own fault stops the walk before the claim (fresh-eyes 4c).

    Read per batch, it classified ``unknown`` and noted every owner of every page:
    three such syncs would have pushed the whole backlog past the tolerance.
    """
    owners = [_commit(d, f"owner-{n}", F_ONE) for n in range(3)]
    _stamp(d)
    m = d.manager()  # the factory stamps the configured model; unset it after
    d.provider.model_id = model
    capsys.readouterr()

    outcome = m.drain_rejection_backlog()

    assert outcome == _outcome(left_queued=3, stop=DRAIN_STOP_UNAVAILABLE)
    assert d.provider.document_calls == [] and d.index.calls == []
    assert [_retry(d, owner) for owner in owners] == [0, 0, 0]
    assert d.store().rejection_seed() is not None  # read nothing, spent nothing
    assert "reported no model id" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Failure: owner-granular quarantine, outages stop and charge nothing
# --------------------------------------------------------------------------- #


def test_r_failing_delete_holds_its_owner_and_the_walk_goes_on(d, capsys) -> None:
    first = _commit(d, "refused-delete", F_ABC)
    _hold(d, first, F_ABC)
    _commit(d, "refused-delete", F_BD)
    second = _commit(d, "next-owner", F_ONE)
    d.index.refuse_delete = True
    capsys.readouterr()

    outcome = _drain(d)

    assert outcome.owners_failed == 1 and outcome.owners_cleared == 1
    assert outcome.items_removed == 0
    assert _retry(d, first) == 1 and d.row(second) is None
    assert _held(d, first) == _ids(first, F_ABC) | _ids(first, F_BD)
    out = capsys.readouterr().out
    assert any("'refused-delete'" in line and "old items were not removed" in line
               for line in out.splitlines())


def test_r_poison_in_batch_costs_only_its_owner(d) -> None:
    healthy = []
    for slug in ("healthy-one", "healthy-two"):
        owner = _commit(d, slug, F_ABC)
        _hold(d, owner, F_ABC)
        _commit(d, slug, F_BD)
        healthy.append(owner)
    poison = _commit(d, "poisoned", f"1. {POISON} — p.")

    outcome = _drain(d)

    assert len(d.provider.document_calls) == 4  # one batch, then three solo
    assert outcome.calls == 4 and outcome.owners_failed == 1
    assert all(d.row(owner) is None for owner in healthy)
    assert all(_held(d, owner) == _ids(owner, F_BD) for owner in healthy)
    assert _retry(d, poison) == 1


@pytest.fixture
def two_per_batch(monkeypatch):
    """Shrinks the byte bound so a batch holds exactly two of the rows' texts."""
    size = len(("search_document: " + "Item 00 — xxxxxxxx.").encode("utf-8"))
    monkeypatch.setattr("mitos.embeddings.EMBED_BATCH_MAX_BYTES", 2 * size)


def _items(*names: str) -> str:
    return "\n".join(f"{i + 1}. {name} — xxxxxxxx." for i, name in enumerate(names))


@pytest.mark.parametrize("a_items", [("PoiA 01", "Item 02"),
                                     ("PoiA 01", "Item 02", "Item 03")],
                         ids=["tail-of-one", "multi-chunk-solo"])
def test_r_split_owner_failing_first_never_sends_its_tail_on(
        d, monkeypatch, two_per_batch, a_items) -> None:
    monkeypatch.setattr("test_rejection_drain.POISON", "PoiA")
    h1 = _commit(d, "h1", _items("Item 10"))
    a = _commit(d, "a", _items(*a_items))
    h2 = _commit(d, "h2", _items("Item 20"))
    a_texts = _texts(a, _items(*a_items))

    outcome = _drain(d)

    calls = d.provider.document_calls
    assert calls[0] == [_texts(h1, _items("Item 10"))[0], a_texts[0]]  # batch 1 fails
    assert calls[1] == _texts(h1, _items("Item 10"))  # h1 solo
    assert calls[2] == a_texts[:2]  # a solo, chunk 1 fails; chunk 2 never sent
    assert calls[3] == _texts(h2, _items("Item 20"))  # batch 2 holds no tail of a
    assert len(calls) == 4 and outcome.calls == 4
    sent = [text for call in calls for text in call]
    assert all(text not in sent for text in a_texts[2:])
    assert _retry(d, a) == 1 and outcome.owners_failed == 1
    assert d.row(h1) is None and d.row(h2) is None
    assert not [det for name, det in d.index.calls
                if name == "upsert_points" and set(det["ids"]) & _ids(a, _items(*a_items))]


def test_r_outage_after_a_landed_batch_finalizes_the_owner_it_finished(
        d, two_per_batch) -> None:
    """G6: an owner whose items all landed in batch 1 is finalized before batch 2."""
    first = _commit(d, "first", _items("Item 01", "Item 02"))
    _hold(d, first, _items("Item 09"))  # a stale item to remove
    second = _commit(d, "second", _items("Item 03"))

    def quota_on_second(texts):
        if len(d.provider.document_calls) == 1:
            d.provider.fault = "quota"

    real = d.provider.embed_documents

    def embed(texts):
        quota_on_second(texts)
        return real(texts)

    d.provider.embed_documents = embed  # type: ignore[method-assign]
    outcome = _drain(d)

    assert outcome.stop == DRAIN_STOP_OUTAGE and outcome.calls == 2
    assert d.row(first) is None and _held(d, first) == _ids(first, _items("Item 01", "Item 02"))
    assert _retry(d, second) == 0


def test_r_head_of_failing_rows_blocks_nothing_behind_it(d, monkeypatch) -> None:
    monkeypatch.setattr("mitos.sync._REJECTION_DRAIN_CLAIM_SIZE", 2)
    poison = [_commit(d, f"poison-{i}", f"1. {POISON} {i} — p.") for i in range(3)]
    healthy = [_commit(d, f"healthy-{i}", f"1. Fine {i} — f.") for i in range(2)]

    outcome = _drain(d)

    assert all(d.row(owner) is None for owner in healthy)
    assert all(_retry(d, owner) == 1 for owner in poison)
    assert outcome.owners_failed == 3 and outcome.owners_cleared == 2
    assert outcome.stop is None and outcome.left_queued == 3


OUTAGES = {
    "auth": lambda d: setattr(d.provider, "fault", "auth"),
    "quota": lambda d: setattr(d.provider, "fault", "quota"),
    "transient": lambda d: setattr(d.provider, "fault", "transient"),
    "upsert-unreachable": lambda d: setattr(
        d.index, "on_call", _once("upsert_points", _raise(VectorStoreUnreachableError))),
    "listing-unreachable": lambda d: setattr(d.index, "unreachable", True),
}


def _raise(error: type) -> Callable[[Dict[str, Any]], None]:
    def action(details: Dict[str, Any]) -> None:
        raise error("planted")
    return action


@pytest.mark.parametrize("marker", [False, True], ids=["no-marker", "marker"])
@pytest.mark.parametrize("outage", list(OUTAGES))
def test_r_outage_stops_at_once_and_charges_no_row(d, capsys, outage, marker) -> None:
    owners = [_commit(d, f"row-{i}", f"1. Item {i} — x.") for i in range(4)]
    if marker:
        _stamp(d)
    OUTAGES[outage](d)
    capsys.readouterr()

    outcome = _drain(d)

    assert outcome.stop == DRAIN_STOP_OUTAGE
    assert outcome.calls == (1 if outage in ("auth", "quota", "transient",
                                             "upsert-unreachable") else 0)
    assert outcome.marker == (DRAIN_MARKER_KEPT if marker else None)
    assert all(_retry(d, owner) == 0 for owner in owners)
    assert outcome.owners_failed == 0 and outcome.left_queued == 4
    assert (d.store().rejection_seed() is not None) is marker
    out = capsys.readouterr().out
    assert len([line for line in out.splitlines() if "stopping the drain" in line]) == 1


# --------------------------------------------------------------------------- #
# R-listing-faults — D3's listing rows
# --------------------------------------------------------------------------- #


def _requeue_on_first(d: Rig, method: str, owner_slug: str, field: str):
    return _once(method, lambda details: _commit(d, owner_slug, field))


@pytest.mark.parametrize("where", ["entry", "mid-walk", "retake"])
@pytest.mark.parametrize("fault", ["unreachable", "refused"])
def test_r_listing_faults_are_outages(d, monkeypatch, where, fault) -> None:
    first = _commit(d, "first", F_ABC)
    second = _commit(d, "second", F_ONE)
    error = VectorStoreUnreachableError if fault == "unreachable" else VectorStoreError
    if where == "entry":
        _stamp(d)
        d.index.on_call = _once("list_point_owners", _raise(error))
    elif where == "mid-walk":
        monkeypatch.setattr("mitos.sync._REJECTION_DRAIN_CLAIM_SIZE", 1)
        d.index.on_call = _once(
            "list_owner_point_ids", _raise(error),
            when=lambda det: det["owner_id"] == second)
    else:
        _stamp(d)
        requeue = _requeue_on_first(d, "upsert_points", "first", F_BD)
        failing = _once("list_owner_point_ids", _raise(error),
                        when=lambda det: det["owner_id"] == first)

        def hook(name, details):
            requeue(name, details)
            failing(name, details)

        d.index.on_call = hook

    outcome = _drain(d)

    assert outcome.stop == DRAIN_STOP_OUTAGE and outcome.owners_failed == 0
    if where == "mid-walk":
        assert d.row(first) is None  # page 1 finished before page 2's listing failed
        assert _retry(d, second) == 0
    else:
        assert _retry(d, first) == 0  # never noted, its re-queued row included
        assert outcome.marker == DRAIN_MARKER_KEPT
        assert d.store().rejection_seed() is not None


def test_r_listing_missing_with_no_marker_is_refused(tmp_path, monkeypatch) -> None:
    d = _rig(tmp_path, monkeypatch, present=False)
    owner = _commit(d, "first", F_ONE)
    outcome = _drain(d)
    assert outcome.stop == DRAIN_STOP_REFUSED and outcome.calls == 0
    assert _ops(d) == ["list_owner_point_ids"] and d.index.created == 0
    assert _retry(d, owner) == 0


def test_r_listing_missing_on_a_retake_under_the_marker_writes_the_owner(d) -> None:
    owner = _commit(d, "first", F_ABC)
    _hold(d, owner, F_ABC)
    _commit(d, "first", F_BD)
    _stamp(d)
    requeue = _requeue_on_first(d, "delete_points", "first", F_NEW)

    def vanish(details: Dict[str, Any]) -> None:
        d.index.present = False  # the retake's own listing meets the absence
        d.index.points.clear()

    gone = _once("list_owner_point_ids", vanish)

    def hook(name: str, details: Dict[str, Any]) -> None:
        requeue(name, details)
        gone(name, details)

    d.index.on_call = hook
    outcome = _drain(d)

    assert _ops(d).count("list_owner_point_ids") == 1  # the retake, and it met a 404
    assert outcome.stop is None and d.index.created == 1
    assert _held(d, owner) == _ids(owner, F_NEW)
    assert d.row(owner) is None


# --------------------------------------------------------------------------- #
# The marker: licence, listing mode, spend rules
# --------------------------------------------------------------------------- #


def test_r_marker_listing_absent_lists_once_and_the_first_write_creates(
        tmp_path, monkeypatch) -> None:
    d = _rig(tmp_path, monkeypatch, present=False)
    owners = [_commit(d, f"row-{i}", f"1. Item {i} — x.") for i in range(3)]
    _stamp(d)

    outcome = _drain(d)

    assert _ops(d).count("list_point_owners") == 1
    assert "list_owner_point_ids" not in _ops(d)
    upserts = [det for name, det in d.index.calls if name == "upsert_points"]
    assert upserts and all(det["may_create"] for det in upserts)
    assert d.index.created == 1 and outcome.calls == 1
    assert outcome.owners_cleared == 3 and outcome.marker == DRAIN_MARKER_SPENT
    assert all(d.row(owner) is None for owner in owners)


def test_r_retake_relists_under_the_marker(d) -> None:
    owner = _commit(d, "first", F_ABC)
    _stamp(d)
    d.index.on_call = _requeue_on_first(d, "upsert_points", "first", F_BD)

    outcome = _drain(d)

    assert [det["owner_id"] for name, det in d.index.calls
            if name == "list_owner_point_ids"] == [owner]
    assert _held(d, owner) == _ids(owner, F_BD) and d.row(owner) is None
    assert outcome.marker == DRAIN_MARKER_SPENT


def test_a_row_queued_after_the_entry_listing_relists(d, monkeypatch) -> None:
    """The watermark: a commit landing after the marker's listing re-lists its owner."""
    monkeypatch.setattr("mitos.sync._REJECTION_DRAIN_CLAIM_SIZE", 1)
    first = _commit(d, "first", F_ONE)
    _stamp(d)
    late: List[str] = []
    d.index.on_call = _once("list_point_owners",
                            lambda details: late.append(_commit(d, "late", F_ABC)))
    _drain(d)
    assert [det["owner_id"] for name, det in d.index.calls
            if name == "list_owner_point_ids"] == late
    assert d.row(first) is None and d.row(late[0]) is None


def test_k12_an_unlicensed_write_meeting_a_vanished_collection_creates_nothing(
        d, capsys) -> None:
    """No marker: the owner listing found the collection, then it went before the write."""
    owner = _commit(d, "first", F_ONE)

    def vanish(details: Dict[str, Any]) -> None:
        d.index.present = False
        d.index.points.clear()

    d.index.on_call = _once("upsert_points", vanish)
    capsys.readouterr()
    outcome = _drain(d)
    assert outcome.stop == DRAIN_STOP_REFUSED and d.index.created == 0
    assert [det["may_create"] for name, det in d.index.calls if name == "upsert_points"] == [False]
    assert _retry(d, owner) == 0
    assert "wait until the rejection index is built" in capsys.readouterr().out


def test_k13_the_decision_seed_marker_licenses_nothing_here(tmp_path, monkeypatch) -> None:
    d = _rig(tmp_path, monkeypatch, present=False)
    owner = _commit(d, "first", F_ONE)
    d.store().stamp_embedding_seed("reconcile")
    outcome = _drain(d)
    assert outcome.stop == DRAIN_STOP_REFUSED and outcome.marker is None
    assert d.index.created == 0 and _retry(d, owner) == 0


def test_r_seeded_poison_spends_and_the_next_ordinary_drain_cannot_create(
        tmp_path, monkeypatch) -> None:
    d = _rig(tmp_path, monkeypatch, present=False)
    healthy = [_commit(d, f"healthy-{i}", f"1. Fine {i} — f.") for i in range(2)]
    poison = _commit(d, "poisoned", f"1. {POISON} — p.")
    _stamp(d)

    outcome = _drain(d)

    assert d.index.created == 1
    assert all(d.row(owner) is None for owner in healthy)
    assert _retry(d, poison) == 1
    assert outcome.marker == DRAIN_MARKER_SPENT and d.store().rejection_seed() is None

    d.index.present = False
    d.index.points.clear()
    d.index.calls.clear()
    d.provider.document_calls.clear()
    again = _drain(d)
    assert again.stop == DRAIN_STOP_REFUSED and again.calls == 0
    assert _ops(d) == ["list_owner_point_ids"] and d.index.created == 1


def test_r_spend_on_an_empty_backlog_lists_nothing(d) -> None:
    _stamp(d)
    outcome = _drain(d)
    assert outcome.marker == DRAIN_MARKER_SPENT and d.index.calls == []
    assert d.store().rejection_seed() is None


def test_r_spend_when_the_walk_completes_present(d) -> None:
    _commit(d, "first", F_ONE)
    _stamp(d)
    assert _drain(d).marker == DRAIN_MARKER_SPENT
    assert d.store().rejection_seed() is None


def test_r_spend_kept_when_still_absent_with_rows_left(tmp_path, monkeypatch) -> None:
    d = _rig(tmp_path, monkeypatch, present=False)
    _commit(d, "poisoned", f"1. {POISON} — p.")
    _stamp(d)
    outcome = _drain(d)
    assert outcome.stop is None and d.index.created == 0
    assert outcome.marker == DRAIN_MARKER_KEPT and d.store().rejection_seed() is not None


def test_r_spend_when_still_absent_and_nothing_is_left(tmp_path, monkeypatch) -> None:
    d = _rig(tmp_path, monkeypatch, present=False)
    owner = _commit(d, "emptied", F_ONE)
    _commit(d, "emptied", "")  # nothing to write, nothing held
    _stamp(d)
    outcome = _drain(d)
    assert d.row(owner) is None and d.index.created == 0
    assert outcome.marker == DRAIN_MARKER_SPENT


def test_r_spend_kept_on_a_refusal_and_a_claim_fault(tmp_path, monkeypatch) -> None:
    d = _rig(tmp_path, monkeypatch, present=False)
    _commit(d, "first", F_ONE)
    _stamp(d)
    d.index.create_then_missing = True
    refused = _drain(d)
    assert refused.stop == DRAIN_STOP_REFUSED and refused.marker == DRAIN_MARKER_KEPT

    def broken(self, limit, *, after_seq=None):
        from mitos.errors import DatabaseError
        raise DatabaseError("planted claim fault")

    monkeypatch.setattr(GraphStore, "claim_rejection_backlog", broken)
    faulted = _drain(d)
    assert faulted.stop == DRAIN_STOP_CLAIM_FAILED and faulted.marker == DRAIN_MARKER_KEPT
    assert d.store().rejection_seed() is not None


def test_r_spend_a_restamp_made_mid_walk_survives(d) -> None:
    _commit(d, "first", F_ONE)
    _stamp(d)
    before = d.store().rejection_seed()

    def restamp(details: Dict[str, Any]) -> None:
        time.sleep(0.002)  # a distinct stamp
        d.store().stamp_rejection_seed("rebuild")

    d.index.on_call = _once("upsert_points", restamp)
    outcome = _drain(d)
    assert outcome.marker == DRAIN_MARKER_SPENT
    after = d.store().rejection_seed()
    assert after is not None and after != before and after["established_by"] == "rebuild"


def test_r_refused_create_is_worded_apart_from_absence(tmp_path, monkeypatch, capsys) -> None:
    d = _rig(tmp_path, monkeypatch, present=False)
    _commit(d, "first", F_ONE)
    _stamp(d)
    d.index.create_then_missing = True
    capsys.readouterr()
    _drain(d)
    out = capsys.readouterr().out
    assert "created and still answered missing" in out
    assert "wait until the rejection index is built" not in out


# --------------------------------------------------------------------------- #
# Small shapes: empty field, presence only, node gone, tolerance
# --------------------------------------------------------------------------- #


def test_r_empty_field_removes_every_held_item_with_no_call(d) -> None:
    owner = _commit(d, "emptied", F_ABC)
    _hold(d, owner, F_ABC)
    _commit(d, "emptied", "")
    outcome = _drain(d)
    assert outcome == _outcome(owners_cleared=1, items_removed=3)
    assert _held(d, owner) == set() and d.row(owner) is None
    assert d.provider.document_calls == []


def test_r_presence_only_spends_no_call(d) -> None:
    owner = _commit(d, "present", F_ABC)
    _hold(d, owner, F_ABC)
    outcome = _drain(d)
    assert outcome == _outcome(owners_cleared=1, items_skipped_present=3)
    assert d.provider.document_calls == [] and d.row(owner) is None


def test_a_row_whose_node_is_gone_is_cleared_and_deletes_nothing(d) -> None:
    owner = _commit(d, "gone", F_ONE)
    _hold(d, owner, F_ONE)
    import sqlite3
    conn = sqlite3.connect(d.config.db_path)
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DELETE FROM nodes WHERE id = ?", (owner,))
        conn.commit()
    finally:
        conn.close()
    outcome = _drain(d)
    assert outcome.owners_cleared == 1 and outcome.items_removed == 0
    assert d.row(owner) is None and _held(d, owner) == _ids(owner, F_ONE)
    assert "delete_points" not in _ops(d)


def test_r_tolerance_is_reported_and_the_row_is_still_attempted(d, capsys) -> None:
    owner = _commit(d, "poisoned", f"1. {POISON} — p.")
    seq = d.row(owner).enqueue_seq
    for _ in range(REJECTION_BACKLOG_FAILURE_TOLERANCE):
        d.store().note_rejection_backlog_failure(owner, enqueue_seq=seq)
    capsys.readouterr()

    outcome = _drain(d)

    assert len(d.provider.document_calls) == 1  # attempted, never dropped
    assert _retry(d, owner) == REJECTION_BACKLOG_FAILURE_TOLERANCE + 1
    assert outcome.owners_past_tolerance == 1
    lines = _summary(capsys.readouterr().out)
    assert any("1 backlog row(s) have failed 3 or more times" in line and "stay queued" in line
               for line in lines)


def test_a_failed_batched_owner_read_passes_its_page_and_charges_nothing(
        d, monkeypatch, capsys) -> None:
    owner = _commit(d, "first", F_ONE)

    def broken(self, node_ids):
        from mitos.errors import DatabaseError
        raise DatabaseError("planted read fault")

    monkeypatch.setattr(GraphStore, "get_rejection_owner_fields", broken)
    capsys.readouterr()
    outcome = _drain(d)
    assert outcome.stop is None and _retry(d, owner) == 0
    assert "Could not read 1 rejection backlog owner(s)" in capsys.readouterr().out
    assert d.index.calls == []


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def test_the_report_is_silent_over_nothing_and_names_no_command() -> None:
    assert rejection_drain_report(_outcome().to_dict()) == []
    assert rejection_drain_report(
        _outcome(stop=REJECTION_DRAIN_STOP_HELD, left_queued=None,
                 owners_past_tolerance=None).to_dict()) == []
    lines = rejection_drain_report(_outcome(
        owners_cleared=12, items_embedded=31, items_skipped_present=40,
        items_removed=6, owners_failed=1, left_queued=2,
        owners_past_tolerance=1).to_dict())
    assert lines[0] == (
        "Rejection index: 12 owner(s) replaced — 31 item(s) embedded, 40 already "
        "present, 6 removed; 1 owner(s) failed; 2 row(s) left queued.")
    assert len(lines) == 2
    assert all("mitos" not in line for line in lines)


# --------------------------------------------------------------------------- #
# Verbs: refusal through sync, wiring, reconcile's key, the commit loop
# --------------------------------------------------------------------------- #


def test_r_full_backlog_absent_no_marker_through_sync_creates_nothing(
        tmp_path, monkeypatch, capsys) -> None:
    d = _rig(tmp_path, monkeypatch, present=False)
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
    for slug in ("one", "two", "three"):
        _append_entry(d, slug)
    capsys.readouterr()

    code = _run(["-p", d.root, "sync", "--yes"])

    out = capsys.readouterr().out
    store = d.store()
    owners = [store.get_node_by_slug(s)["id"] for s in ("one", "two", "three")]
    assert code in (0, None)
    assert d.index.created == 0 and d.provider.document_calls == []
    assert all(_retry(d, owner) == 0 for owner in owners)
    refusals = [line for line in out.splitlines() if "wait until the rejection index" in line]
    assert refusals == [
        "Rejection index: 3 rejection backlog row(s) wait until the rejection index "
        "is built; nothing was created or embedded."]
    # X-heal: this release builds nothing that would heal it, so no verb is named
    # (4d inverts this when its covering verbs stamp the marker).
    for word in ("reconcile", "rebuild", "mitos"):
        assert word not in refusals[0]


def _spy_order(monkeypatch) -> List[str]:
    order: List[str] = []
    real_decision = MitosSyncManager.drain_pending_embeddings
    real_rejection = MitosSyncManager.drain_rejection_backlog

    def decision(self, *args, **kwargs):
        order.append("decision")
        return real_decision(self, *args, **kwargs)

    def rejection(self, *args, **kwargs):
        order.append("rejection")
        return real_rejection(self, *args, **kwargs)

    monkeypatch.setattr(MitosSyncManager, "drain_pending_embeddings", decision)
    monkeypatch.setattr(MitosSyncManager, "drain_rejection_backlog", rejection)
    return order


@pytest.mark.parametrize("verb", [["sync", "--yes"], ["sync", "--embed-only"], ["reconcile"]])
def test_r_wiring_each_verb_drains_rejections_after_decisions(d, monkeypatch, verb) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
    _append_entry(d, "buffered")
    _commit(d, "queued", F_ONE)
    if verb[0] == "reconcile":
        _run(["-p", d.root, "sync", "--yes"])  # a built graph to reconcile
        _commit(d, "queued", F_BD)
    order = _spy_order(monkeypatch)
    _run(["-p", d.root, *verb])
    assert order[-1] == "rejection" and "decision" in order
    assert order.index("rejection") > order.index("decision") or verb[0] == "reconcile"
    assert d.row(d.store().get_node_by_slug("queued")["id"]) is None


def _raising_rejection_drain(monkeypatch) -> None:
    def boom(self, *args, **kwargs):
        raise RuntimeError("planted rejection drain fault")

    monkeypatch.setattr(MitosSyncManager, "drain_rejection_backlog", boom)


@pytest.mark.parametrize("verb", [["sync", "--yes"], ["sync", "--embed-only"]])
def test_r_wiring_a_raising_drain_leaves_sync_whole(d, monkeypatch, capsys, verb) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
    _append_entry(d, "buffered")
    _raising_rejection_drain(monkeypatch)
    capsys.readouterr()
    code = _run(["-p", d.root, *verb])
    out, err = capsys.readouterr()
    assert code in (0, None)
    assert "Rejection backlog drain failed: planted rejection drain fault" in err
    if verb[1] == "--yes":
        assert d.store().get_node_by_slug("buffered") is not None


def _reconcile(d: Rig, capsys, *, as_json: bool) -> Any:
    capsys.readouterr()
    argv = ["-p", d.root, "reconcile"] + (["--json"] if as_json else [])
    code = _run(argv)
    out, err = capsys.readouterr()
    return code, out, err


@pytest.mark.parametrize("as_json", [True, False], ids=["json", "text"])
def test_r_wiring_a_raising_drain_leaves_reconcile_whole(d, monkeypatch, capsys, as_json) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
    _append_entry(d, "buffered")
    _run(["-p", d.root, "sync", "--yes"])
    owner = _commit(d, "queued", F_ONE)
    _raising_rejection_drain(monkeypatch)
    code, out, err = _reconcile(d, capsys, as_json=as_json)
    assert code == 0
    assert "Rejection backlog drain failed" in err
    if as_json:
        result = json.loads(out)
        assert result["rejection_drain"]["stop"] == DRAIN_STOP_UNAVAILABLE
        assert result["rejection_drain"]["left_queued"] == 1
        assert result["expected"] >= 1
    else:
        assert ("Rejection index: not drained this run; 1 backlog row(s) stay queued."
                in out.splitlines())
        assert "Reconciled:" in out
    assert _retry(d, owner) == 0


def test_r_reconcile_json_is_one_object_on_every_path(tmp_path, monkeypatch, capsys) -> None:
    keys = set(_outcome().to_dict())
    # normal, with a row to drain (its progress lines must not reach stdout)
    d = _rig(tmp_path, monkeypatch, name="normal")
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
    _append_entry(d, "buffered")
    _run(["-p", d.root, "sync", "--yes"])
    owner = _commit(d, "queued", F_ABC)
    _hold(d, owner, F_ABC)
    _commit(d, "queued", F_BD)
    _commit(d, "poisoned", f"1. {POISON} — p.")
    code, out, err = _reconcile(d, capsys, as_json=True)
    normal = json.loads(out)  # exactly one object: anything else fails to parse
    assert set(normal["rejection_drain"]) == keys
    assert normal["rejection_drain"]["owners_cleared"] == 1
    assert normal["rejection_drain"]["owners_failed"] == 1
    assert "'poisoned'" in err and "Rejection index" not in out

    # provider down. A scoped patch, never ``monkeypatch.undo()``: that would also
    # undo conftest's autouse redirects (the registry's XDG home among them).
    with pytest.MonkeyPatch.context() as scoped:
        scoped.setattr(MitosSyncManager, "__init__",
                       _down_init(MitosSyncManager.__init__))
        _commit(d, "queued", F_ABC)
        code, out, err = _reconcile(d, capsys, as_json=True)
    down = json.loads(out)
    assert down["rejection_drain"]["stop"] == DRAIN_STOP_UNAVAILABLE
    assert down["rejection_drain"]["left_queued"] == 2  # the re-queue and the poison
    assert set(down) == set(normal)

    # unbuilt graph: a corpus with entries, a graph holding no node
    u = _rig(tmp_path, monkeypatch, name="unbuilt")
    _append_entry(u, "never-synced")
    code, out, err = _reconcile(u, capsys, as_json=True)
    unbuilt = json.loads(out)
    assert code == 1 and unbuilt["graph_unbuilt"] is True
    assert set(unbuilt["rejection_drain"]) == keys and set(unbuilt) == set(normal)


def _down_init(real):
    def init(self, config):
        real(self, config)
        self.embed_provider = None
    return init


def test_r_reconcile_text_prints_the_summary_once(d, monkeypatch, capsys) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
    _append_entry(d, "buffered")
    _run(["-p", d.root, "sync", "--yes"])
    owner = _commit(d, "queued", F_ABC)
    _hold(d, owner, F_ABC)
    _commit(d, "queued", F_BD)
    code, out, err = _reconcile(d, capsys, as_json=False)
    assert code == 0
    assert len([line for line in out.splitlines()
                if line.startswith("Rejection index: 1 owner(s) replaced")]) == 1


def test_r_sync_loop_defers_to_its_drain(d, monkeypatch, capsys) -> None:
    """D5: the commit loop and the fixpoint run no inline rejection leg; the drain
    clears every row in fewer calls than entries."""
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
    inline: List[str] = []
    real = MitosSyncManager._write_rejection_items_inline

    def spy(self, delta, entry):
        inline.append(entry.slug)
        return real(self, delta, entry)

    monkeypatch.setattr(MitosSyncManager, "_write_rejection_items_inline", spy)
    _append_entry(d, "plain")
    _append_entry(d, "authored-later")
    with open(d.config.decisions_file, "a", encoding="utf-8") as handle:
        handle.write(
            "\n## 2026-10-17 — forward-ref — Forward Ref\n"
            "**Decided:** The axiom citing an entry authored above it.\n"
            "**Rejected:**\n1. Waiting — it never converges.\n"
            "**Cites:** authored-later\n"
            "**Mechanisms:** forward-ref-mechanism\n"
            "**Scope:** alpha\n"
        )
    capsys.readouterr()

    code = _run(["-p", d.root, "sync", "--yes"])

    out = capsys.readouterr().out
    assert code in (0, None)
    assert "[Fixpoint] converged 1 quarantined entry" in out  # non-vacuity
    assert inline == []
    store = d.store()
    owners = [store.get_node_by_slug(s)["id"] for s in ("plain", "authored-later", "forward-ref")]
    assert all(d.row(owner) is None for owner in owners)
    assert len(d.provider.document_calls) == 1 < len(owners)


def test_record_and_amend_still_write_inline(d) -> None:
    from test_rejection_inline import _mcp_amend, _record

    owner = _record(d.manager(), "first", F_ABC)["id"]
    assert _held(d, owner) == _ids(owner, F_ABC) and d.row(owner) is None
    _mcp_amend(d, "first", rejected_paths=F_BD)
    assert _ids(owner, F_BD) <= _held(d, owner)


def test_r_check_is_untouched_by_a_row_past_tolerance(tmp_path, monkeypatch) -> None:
    from mitos.check import probe_stale_index

    d = _rig(tmp_path, monkeypatch)
    owner = _commit(d, "poisoned", f"1. {POISON} — p.")
    row = d.row(owner)
    probes = [repr(probe_stale_index(d.store()))]
    codes = [_run(["-p", d.root, "check"])]

    for _ in range(REJECTION_BACKLOG_FAILURE_TOLERANCE):
        d.store().note_rejection_backlog_failure(owner, enqueue_seq=row.enqueue_seq)
    assert d.store().count_rejection_backlog(
        min_retry=REJECTION_BACKLOG_FAILURE_TOLERANCE) == 1
    probes.append(repr(probe_stale_index(d.store())))
    codes.append(_run(["-p", d.root, "check"]))

    assert probes[0] == probes[1]
    assert codes[0] == codes[1]


def test_the_stdio_fake_index_is_loud_on_the_owner_listing() -> None:
    from _stdio_fake_providers import _rejection_index_class

    with pytest.raises(NotImplementedError):
        _rejection_index_class()().list_point_owners()
