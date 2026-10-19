"""The covering verbs build and collect the rejection index; ``status`` reports it.

Recall-confidence phase 4d. The rejection index's expected contents are computable from
the graph alone — every decision's ``rejected_paths``, enumerated by the one leaf — so
``status`` and ``reconcile`` diff the index by point id (``rejections.diff_rejection_index``)
and the heal clears its own warning by construction. ``reconcile`` lists the index once,
queues every owner lacking an item, stamps ``rejection_seed("reconcile")`` when anything
is missing, runs 4c's drain, and collects stale and orphaned items under the drain's
lock, judging the listing against a graph read taken after it and never over a graph
holding no node. ``rebuild``/``cutover`` stamp the marker in the aside transaction; an
import into a graph holding no decision stamps it after its commits. ``status`` words
the index from the diff — never from the backlog's row count — and none of it moves
``ready``, the verdict or an exit code.

The rows drive ``cli.main`` for ``reconcile`` and ``status`` (and the manager for the
collection's internals), with ``FakeRejectionIndex`` bound at both seams
(``mitos.sync`` and ``mitos.cli``), a recording provider (never a MagicMock), the
per-test decision store, and a real ``GraphStore`` in ``tmp_path``.
"""

import json
import os
import sqlite3
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pytest
from filelock import FileLock

from mitos import cli, mcp_server, rejections
from mitos.errors import (
    CollectionMissingError,
    DatabaseError,
    VectorStoreError,
    VectorStoreUnreachableError,
)
from mitos.parser import ParsedEntry
from mitos.recall import missing_rejection_index_is_a_gap
from mitos.rejections import (
    RejectionIndexDiff,
    diff_rejection_index,
    expected_rejection_points,
    rejection_points,
)
from mitos.shell_quote import shell_flag
from mitos.store import REJECTION_BACKLOG_FAILURE_TOLERANCE, GraphStore
from mitos.sync import (
    DRAIN_STOPS,
    REJECTION_COLLECT_HELD,
    REJECTION_COLLECT_JUDGED,
    REJECTION_COLLECT_OUTCOMES,
    REJECTION_COLLECT_UNAVAILABLE,
    REJECTION_COLLECT_UNBUILT,
    REJECTION_COLLECT_UNREAD,
    MitosSyncManager,
    rejection_drain_lock_path,
)
from mitos.vector_store import hash_to_uuid

from _rejection_index_double import FakeRejectionIndex
from test_cli_selector import _run
from test_expected_set import _decision, _matrix, _open_question, store  # noqa: F401
from test_rejection_drain import F_ABC, F_BD, F_ONE, _commit, _hold, _ids
from test_rejection_drain import _rig as _drain_rig
from test_rejection_inline import MODEL, Rig, _append_entry, _mcp_amend, _mcp_record
from test_status_overview import _register, _workspace, qdrant  # noqa: F401

A = "a" * 64
B = "b" * 64
C = "c" * 64

_REJECTION_CHECK_KEYS = (
    "rejection_collection_exists", "rejection_expected_items", "missing_rejection_items",
    "stale_rejection_items", "orphan_rejection_items", "rejection_backlog_past_tolerance",
)
_INDEX_KEYS = (
    "expected_items", "expected_owners", "present_items", "collection_present",
    "missing_items", "owners_queued", "seed_stamped", "stale_items", "orphan_items",
    "deleted", "delete_failed", "collected",
)


# --------------------------------------------------------------------------- #
# The rig: 4c's drain rig, plus status's seams
# --------------------------------------------------------------------------- #


def _rig(tmp_path, monkeypatch, *, present: bool = False, name: str = "ws") -> Rig:
    """4c's rig with the provider key set and ``status``' three reads bound too.

    The rejection index answers at both seams (``mitos.sync`` for reconcile's manager,
    ``mitos.cli`` for status's own listing); ``status``' decision probe and scroll
    answer from the decision store's collections.
    """
    d = _drain_rig(tmp_path, monkeypatch, present=present, name=name)
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
    monkeypatch.setattr("mitos.cli.QdrantRejectionIndex", lambda url, coll: d.index)
    collections = d.vector_cls._collections

    def check(url: str, coll: str) -> Dict[str, Any]:
        points = collections.get(coll)
        return {"reachable": True, "collection_exists": points is not None,
                "points": None if points is None else len(points)}

    def scroll(url: str, coll: str, page_size: int = 256) -> Set[str]:
        points = collections.get(coll)
        if points is None:
            raise CollectionMissingError("missing", collection=coll)
        return {hash_to_uuid(node_id) for node_id in points}

    monkeypatch.setattr(cli, "_check_qdrant", check)
    monkeypatch.setattr(cli, "scroll_point_ids", scroll)
    return d


@pytest.fixture
def d(tmp_path, monkeypatch) -> Rig:
    return _rig(tmp_path, monkeypatch, present=True)


@pytest.fixture
def absent(tmp_path, monkeypatch) -> Rig:
    return _rig(tmp_path, monkeypatch, present=False)


def _seat(d: Rig, slug: str, field: str) -> str:
    """A decision the index lacks and no row names: the upgraded workspace's shape."""
    owner = _commit(d, slug, field)
    row = d.row(owner)
    if row is not None:
        assert d.store().clear_rejection_backlog_row(owner, enqueue_seq=row.enqueue_seq)
    return owner


def _plant(d: Rig, point_id: str, owner: Optional[str]) -> None:
    payload: Dict[str, Any] = {"model": MODEL}
    if owner is not None:
        payload["owner_id"] = owner
    d.index.points[point_id] = {"vector": [0.0], "payload": payload}


def _sel(d: Rig) -> str:
    return shell_flag("-p", d.config.project)


def _reconcile(d: Rig, capsys) -> Tuple[Any, Dict[str, Any], str]:
    capsys.readouterr()
    code = _run(["-p", d.root, "reconcile", "--json"])
    out, err = capsys.readouterr()
    return code, json.loads(out), err


def _reconcile_text(d: Rig, capsys) -> Tuple[Any, str, str]:
    capsys.readouterr()
    code = _run(["-p", d.root, "reconcile"])
    out, err = capsys.readouterr()
    return code, out, err


def _status(d: Rig, capsys) -> Tuple[Any, Dict[str, Any]]:
    capsys.readouterr()
    code = _run(["-p", d.root, "status", "--json"])
    return code, json.loads(capsys.readouterr().out)


def _status_text(d: Rig, capsys) -> Tuple[Any, str]:
    capsys.readouterr()
    code = _run(["-p", d.root, "status"])
    return code, capsys.readouterr().out


def _rkeys(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: payload["checks"][key] for key in _REJECTION_CHECK_KEYS}


def _three(payload: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    checks = payload["checks"]
    return (checks["missing_rejection_items"], checks["stale_rejection_items"],
            checks["orphan_rejection_items"])


def _rejection_lines(out: str) -> List[str]:
    return [line for line in out.splitlines()
            if "rejection index" in line.lower() or "rejected item" in line
            or "rejected items" in line]


def _all_items(d: Rig) -> Set[str]:
    return set(d.index.points)


# --------------------------------------------------------------------------- #
# D-leaf / D-memo — one expected-set derivation, one diff
# --------------------------------------------------------------------------- #


def test_d_leaf_the_diff_splits_missing_stale_and_orphaned_by_id_and_owner() -> None:
    fields = {A: F_ABC, B: F_ONE}
    expected = expected_rejection_points(fields)
    assert expected == {**{pid: A for pid in _ids(A, F_ABC)},
                        **{pid: B for pid in _ids(B, F_ONE)}}

    a_ids = sorted(_ids(A, F_ABC))
    listing = {
        a_ids[0]: C,      # an expected id under a foreign owner's payload: present
        a_ids[1]: A,
        "stale-1": A,     # owner a graph decision, item not stated
        "orphan-1": C,    # owner not a graph decision
        "orphan-2": None,  # unnamed owner
    }
    diff = diff_rejection_index(expected, listing, owners=fields.keys())

    assert isinstance(diff, RejectionIndexDiff)
    assert diff.missing == {pid: owner for pid, owner in expected.items()
                            if pid not in (a_ids[0], a_ids[1])}
    assert diff.stale == frozenset({"stale-1"})
    assert diff.orphaned == frozenset({"orphan-1", "orphan-2"})
    assert diff.missing_owners == frozenset({A, B})


def test_d_leaf_identical_items_collapse_and_empty_fields_expect_nothing() -> None:
    assert len(expected_rejection_points({A: "1. Same — s.\n2. Same — s."})) == 1
    assert expected_rejection_points({A: "", B: "  \n\t "}) == {}
    with pytest.raises(ValueError):
        expected_rejection_points({"not-a-node-id": F_ONE})


def test_d_leaf_the_memo_enumerates_an_unchanged_owner_once(monkeypatch) -> None:
    seen: List[Tuple[str, str]] = []
    real = rejections.rejection_points

    def spy(owner: str, field: str):
        seen.append((owner, field))
        return real(owner, field)

    monkeypatch.setattr(rejections, "rejection_points", spy)
    memo: Dict[Tuple[str, str], Tuple[str, ...]] = {}
    first = expected_rejection_points({A: F_ABC, B: F_ONE}, memo)
    second = expected_rejection_points({A: F_ABC, B: F_BD}, memo)
    assert seen == [(A, F_ABC), (B, F_ONE), (B, F_BD)]
    assert set(first) - _ids(B, F_ONE) == set(second) - _ids(B, F_BD)


def _spy_enumerations(monkeypatch) -> List[Tuple[str, str]]:
    seen: List[Tuple[str, str]] = []
    real = rejections.rejection_points

    def spy(owner: str, field: str):
        seen.append((owner, field))
        return real(owner, field)

    monkeypatch.setattr(rejections, "rejection_points", spy)
    return seen


def test_d_memo_reconcile_enumerates_each_owner_and_field_once(d, monkeypatch) -> None:
    one = _seat(d, "one", F_ABC)
    _hold(d, one, F_ABC)
    two = _seat(d, "two", F_ONE)
    _hold(d, two, F_ONE)
    seen = _spy_enumerations(monkeypatch)

    result = d.manager().reconcile_embeddings()

    assert result["rejection_index"]["collected"] == REJECTION_COLLECT_JUDGED
    assert sorted(seen) == sorted([(one, F_ABC), (two, F_ONE)])


def test_d_memo_an_owner_amended_between_r1_and_r5_is_enumerated_twice(
        d, monkeypatch) -> None:
    one = _seat(d, "one", F_ABC)
    _hold(d, one, F_ABC)
    two = _seat(d, "two", F_ONE)
    _hold(d, two, F_ONE)
    fired: List[bool] = []

    def amend(name: str, details: Dict[str, Any]) -> None:
        if name == "list_point_owners" and not fired:
            fired.append(True)
            _commit(d, "one", F_BD)  # the amend lands after R1, before R5

    d.index.on_call = amend
    seen = _spy_enumerations(monkeypatch)

    d.manager().reconcile_embeddings()

    assert seen.count((one, F_ABC)) == 1 and seen.count((one, F_BD)) == 1
    assert seen.count((two, F_ONE)) == 1


# --------------------------------------------------------------------------- #
# G-gap — keyed on this index's own expected set
# --------------------------------------------------------------------------- #


def test_g_gap_fires_on_an_item_bearing_decision_and_never_on_questions(store) -> None:
    assert store.rejection_index_expects_items() is False
    assert missing_rejection_index_is_a_gap(store) is False

    _open_question(store, "q-only", "Only a question?")
    assert store.rejection_index_expects_items() is False
    assert missing_rejection_index_is_a_gap(store) is False

    live = _decision(store, "live", "A live axiom.")
    assert store.rejection_index_expects_items() is True
    assert missing_rejection_index_is_a_gap(store) is True

    _raw_field(store, live, "  \n ")  # whitespace enumerates nothing
    assert store.rejection_index_expects_items() is False
    assert missing_rejection_index_is_a_gap(store) is False


def test_g_gap_a_retired_decision_still_owns_items(store) -> None:
    _decision(store, "old", "An old axiom.")
    new = _decision(store, "new", "Its successor.", supersedes=["old"])
    _raw_field(store, new, "")
    assert store.rejection_index_expects_items() is True
    assert missing_rejection_index_is_a_gap(store) is True


def test_g_gap_an_unreadable_graph_is_the_loud_answer() -> None:
    class _Raising:
        def rejection_index_expects_items(self) -> bool:
            raise DatabaseError("planted")

    assert missing_rejection_index_is_a_gap(_Raising()) is True


def _raw_field(store: GraphStore, node_id: str, value: str) -> None:
    conn = store._get_connection()
    try:
        conn.execute("UPDATE nodes SET rejected_paths_json = ? WHERE id = ?", (value, node_id))
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# E-enqueue / A-active — the store
# --------------------------------------------------------------------------- #


def test_e_enqueue_inserts_for_rowless_owners_and_never_touches_a_standing_row(d) -> None:
    standing = _commit(d, "standing", F_ONE)
    row = d.row(standing)
    store = d.store()
    assert store.note_rejection_backlog_failure(standing, enqueue_seq=row.enqueue_seq)
    rowless = _seat(d, "rowless", F_ONE)

    assert store.enqueue_rejection_backlog_owners([standing, rowless, rowless]) == 1

    kept = d.row(standing)
    assert (kept.enqueue_seq, kept.retry_count, kept.queued_at) == (
        row.enqueue_seq, 1, row.queued_at)
    assert d.row(rowless).retry_count == 0
    assert store.enqueue_rejection_backlog_owners([]) == 0


def test_a_active_the_ids_only_read_equals_the_hydrating_derivation(store) -> None:
    ids = _matrix(store)
    store.write_signal(ids["d-active"], "drifted")
    hydrating = ({n["id"] for n in store.get_active_decisions()}
                 | {n["id"] for n in store.get_open_questions()})

    active = store.get_active_node_ids()

    assert isinstance(active, set)
    assert active == hydrating
    assert {ids["d-active"], ids["q-parked"], ids["q-resolved"]} <= active
    assert not active & {ids[s] for s in ("d-superseded", "d-corrected", "q-superseded",
                                          "q-corrected")}


def test_the_owner_fields_read_is_every_decision_raw_and_coalesced(store) -> None:
    ids = _matrix(store)
    _raw_field(store, ids["d-active"], "")
    conn = store._get_connection()
    try:
        conn.execute("UPDATE nodes SET rejected_paths_json = NULL WHERE id = ?",
                     (ids["d-resolver"],))
        conn.commit()
    finally:
        conn.close()

    fields = store.get_rejection_index_owner_fields()

    assert set(fields) == {i for s, i in ids.items() if s.startswith("d-")}
    assert fields[ids["d-active"]] == "" and fields[ids["d-resolver"]] == ""
    assert fields[ids["d-superseded"]] == "None."
    assert store.holds_any_decision() is True


def test_holds_any_decision_ignores_open_questions(store) -> None:
    assert store.holds_any_decision() is False
    _open_question(store, "q", "A question?")
    assert store.holds_any_decision() is False


def test_the_collect_vocabulary_is_closed_and_beside_the_drain_stops() -> None:
    assert REJECTION_COLLECT_OUTCOMES == ("judged", "held", "unbuilt", "unread", "unavailable")
    assert "unavailable" in DRAIN_STOPS


# --------------------------------------------------------------------------- #
# R-build / R-noop / R-stale / R-orphan — reconcile
# --------------------------------------------------------------------------- #


def test_r_build_reconcile_queues_stamps_and_the_drain_writes_every_item(
        absent, capsys) -> None:
    owners = {_seat(absent, "one", F_ABC): F_ABC, _seat(absent, "two", F_ONE): F_ONE,
              _seat(absent, "three", F_BD): F_BD}
    expected = set().union(*(_ids(o, f) for o, f in owners.items()))
    assert all(absent.row(o) is None for o in owners)

    code, result, _err = _reconcile(absent, capsys)

    index = result["rejection_index"]
    assert code == 0
    assert index == {
        "expected_items": len(expected), "expected_owners": 3, "present_items": 0,
        "collection_present": False, "missing_items": len(expected), "owners_queued": 3,
        "seed_stamped": True, "stale_items": 0, "orphan_items": 0, "deleted": 0,
        "delete_failed": 0, "collected": REJECTION_COLLECT_JUDGED,
    }
    drain = result["rejection_drain"]
    assert drain["items_embedded"] == index["missing_items"]
    assert drain["owners_cleared"] == index["owners_queued"]
    assert drain["stop"] is None and drain["left_queued"] == 0
    assert absent.index.created == 1
    assert _all_items(absent) == expected
    _code, after = _status(absent, capsys)
    assert _three(after) == (0, 0, 0)
    assert after["checks"]["rejection_collection_exists"] is True


def test_r_noop_a_complete_index_queues_stamps_and_deletes_nothing(
        d, monkeypatch, capsys) -> None:
    one = _seat(d, "one", F_ABC)
    _hold(d, one, F_ABC)
    # Spied, not read back after: the drain spends a marker on an empty backlog.
    stamps: List[str] = []
    real_stamp = GraphStore.stamp_rejection_seed

    def spy(self, established_by: str) -> None:
        stamps.append(established_by)
        real_stamp(self, established_by)

    monkeypatch.setattr(GraphStore, "stamp_rejection_seed", spy)

    _code, result, _err = _reconcile(d, capsys)

    index = result["rejection_index"]
    assert index["missing_items"] == 0 and index["owners_queued"] == 0
    assert index["seed_stamped"] is False
    assert (index["stale_items"], index["orphan_items"], index["deleted"]) == (0, 0, 0)
    assert index["collected"] == REJECTION_COLLECT_JUDGED
    assert stamps == [] and d.store().rejection_seed() is None
    assert not [c for c in d.index.calls if c[0] == "delete_points"]


def test_r_stale_status_names_it_reconcile_deletes_it_status_clears(d, capsys) -> None:
    owner = _seat(d, "amended", F_ABC)
    _hold(d, owner, F_ABC)
    # The older build: the field changed and the enqueue never ran.
    _commit(d, "amended", F_BD)
    _seat(d, "amended", F_BD)
    _hold(d, owner, F_BD)
    stale = _ids(owner, F_ABC) - _ids(owner, F_BD)

    _code, before = _status(d, capsys)
    assert _three(before) == (0, len(stale), 0)

    _code, result, _err = _reconcile(d, capsys)
    index = result["rejection_index"]
    assert (index["stale_items"], index["orphan_items"]) == (len(stale), 0)
    assert index["deleted"] == len(stale)
    assert _all_items(d) == _ids(owner, F_BD)

    _code, after = _status(d, capsys)
    assert _three(after) == (0, 0, 0)


def test_r_orphan_an_ownerless_and_an_unnamed_item_are_both_deleted(d, capsys) -> None:
    owner = _seat(d, "one", F_ONE)
    _hold(d, owner, F_ONE)
    _plant(d, "11111111-1111-1111-1111-111111111111", C)
    _plant(d, "22222222-2222-2222-2222-222222222222", None)

    _code, before = _status(d, capsys)
    assert _three(before) == (0, 0, 2)

    _code, result, _err = _reconcile(d, capsys)
    index = result["rejection_index"]
    assert (index["stale_items"], index["orphan_items"], index["deleted"]) == (0, 2, 2)
    assert _all_items(d) == _ids(owner, F_ONE)
    deleted = [details["ids"] for name, details in d.index.calls if name == "delete_points"]
    assert deleted == [sorted(["11111111-1111-1111-1111-111111111111",
                               "22222222-2222-2222-2222-222222222222"])]


def test_r_text_reconcile_words_the_index_and_its_collection(d, capsys) -> None:
    owner = _seat(d, "one", F_ONE)
    _hold(d, owner, F_ONE)
    _plant(d, "11111111-1111-1111-1111-111111111111", C)
    second = _seat(d, "two", F_ABC)

    code, out, _err = _reconcile_text(d, capsys)

    assert code == 0
    total = len(_ids(owner, F_ONE)) + len(_ids(second, F_ABC))
    assert (f"Rejection index: {total} item(s) of 2 decision(s) expected, "
            f"{len(_ids(second, F_ABC))} missing — 1 decision(s) newly queued for the "
            f"drain.") in out
    assert "Rejection index collected: 1 stale or orphaned item(s) deleted." in out
    assert "None" not in "\n".join(_rejection_lines(out))


# --------------------------------------------------------------------------- #
# R-mid-collect (T12) / R-held / R-empty-listing / R-unbuilt
# --------------------------------------------------------------------------- #


def test_r_mid_collect_a_record_and_an_amend_during_reconcile_keep_their_items(
        d, capsys) -> None:
    kept = _seat(d, "kept", F_ONE)
    _hold(d, kept, F_ONE)
    amended = _mcp_record(d, "amended")["id"]  # in the buffer, so it can be amended
    landed: Dict[str, str] = {}

    def land(name: str, details: Dict[str, Any]) -> None:
        # Fires inside R2's listing, before it answers: both writes are listed, and
        # their owners commit before R5's graph read.
        if name == "list_point_owners" and not landed:
            landed["record"] = _mcp_record(d, "landed")["id"]
            _mcp_amend(d, "amended", rejected_paths=F_BD)
            landed["amend"] = amended

    d.index.on_call = land

    _code, result, _err = _reconcile(d, capsys)

    from test_rejection_inline import FIELD
    assert set(landed) == {"record", "amend"}
    assert set(_ids(landed["record"], FIELD)) <= _all_items(d)
    assert _ids(amended, F_BD) <= _all_items(d)
    assert not (_ids(amended, FIELD) - _ids(amended, F_BD)) & _all_items(d)
    assert _ids(kept, F_ONE) <= _all_items(d)
    index = result["rejection_index"]
    assert index["collected"] == REJECTION_COLLECT_JUDGED
    assert index["orphan_items"] == 0


def test_r_mid_collect_status_twin_a_record_during_status_never_reads_orphaned(
        d, capsys) -> None:
    kept = _seat(d, "kept", F_ONE)
    _hold(d, kept, F_ONE)
    landed: List[str] = []

    def land(name: str, details: Dict[str, Any]) -> None:
        if name == "list_point_owners" and not landed:
            landed.append(_mcp_record(d, "landed")["id"])

    d.index.on_call = land

    _code, payload = _status(d, capsys)

    assert landed
    assert payload["checks"]["orphan_rejection_items"] == 0
    assert payload["checks"]["stale_rejection_items"] == 0


def test_r_held_a_held_drain_lock_collects_nothing_and_names_the_rerun(
        d, capsys) -> None:
    owner = _seat(d, "one", F_ONE)
    _hold(d, owner, F_ONE)
    _plant(d, "11111111-1111-1111-1111-111111111111", C)
    holder = FileLock(rejection_drain_lock_path(d.config), timeout=0)
    holder.acquire()
    try:
        _code, result, _err = _reconcile(d, capsys)
        code, out, _err = _reconcile_text(d, capsys)
    finally:
        holder.release()

    index = result["rejection_index"]
    assert index["collected"] == REJECTION_COLLECT_HELD
    assert (index["stale_items"], index["orphan_items"], index["deleted"]) == (None, None, 0)
    assert "11111111-1111-1111-1111-111111111111" in d.index.points
    assert code == 0
    assert (f"Rejection index not collected — a drain held the rejection backlog's lock; "
            f"`mitos reconcile {_sel(d)}` again collects it.") in out


def test_r_empty_listing_a_first_build_takes_no_lock_and_reads_the_graph_once(
        absent, monkeypatch, capsys) -> None:
    _seat(absent, "one", F_ABC)
    reads: List[int] = []
    real_read = GraphStore.get_rejection_index_owner_fields

    def counted(self):
        reads.append(1)
        return real_read(self)

    monkeypatch.setattr(GraphStore, "get_rejection_index_owner_fields", counted)
    locks: List[str] = []
    import mitos.sync as sync_module
    real_lock = sync_module.FileLock

    def lock_spy(path: str, *args: Any, **kwargs: Any):
        locks.append(path)
        return real_lock(path, *args, **kwargs)

    monkeypatch.setattr(sync_module, "FileLock", lock_spy)

    result = absent.manager().reconcile_embeddings()

    index = result["rejection_index"]
    assert index["collected"] == REJECTION_COLLECT_JUDGED
    assert (index["stale_items"], index["orphan_items"]) == (0, 0)
    assert len(reads) == 1
    # One acquisition only: the drain's own (R4). R5 took none.
    assert locks.count(rejection_drain_lock_path(absent.config)) == 1


def test_r_unbuilt_a_zero_node_graph_judges_and_deletes_nothing_in_either_index(
        d, capsys) -> None:
    _append_entry(d, "unsynced")  # the corpus holds an entry; the graph holds none
    _plant(d, "11111111-1111-1111-1111-111111111111", A)
    d.vector_cls._collections[d.config.qdrant_collection] = {
        A: ([0.0], {"slug": "ghost"})}

    code, result, err = _reconcile(d, capsys)
    text_code, _out, text_err = _reconcile_text(d, capsys)

    index = result["rejection_index"]
    assert index["collected"] == REJECTION_COLLECT_UNBUILT
    assert (index["stale_items"], index["orphan_items"], index["deleted"]) == (None, None, 0)
    assert "11111111-1111-1111-1111-111111111111" in d.index.points
    assert A in d.vector_cls._collections[d.config.qdrant_collection]
    assert result["deleted"] == 0 and result["graph_unbuilt"] is True
    assert code == 1 and text_code == 1
    assert "mitos rebuild" in text_err
    assert not [c for c in d.index.calls if c[0] == "delete_points"]


def test_r_unbuilt_over_an_empty_corpus_names_the_items_not_judged(d, capsys) -> None:
    # A zero-node graph whose corpus is empty is not "unbuilt" (no refusal speaks), so
    # the collection's own outcome must: the decision side's "not judged" twin.
    _plant(d, "11111111-1111-1111-1111-111111111111", A)

    code, out, _err = _reconcile_text(d, capsys)

    assert code == 0
    assert ("1 rejected item(s) not judged — the graph holds no nodes; reconcile "
            "deletes items only over a built graph.") in out
    assert "11111111-1111-1111-1111-111111111111" in d.index.points


# --------------------------------------------------------------------------- #
# R-faults / R-keys
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fault", ["refuse_list", "unreachable"])
def test_r_faults_a_listing_fault_queues_and_stamps_nothing(
        d, capsys, fault) -> None:
    owner = _seat(d, "one", F_ABC)
    setattr(d.index, fault, True)

    code, result, err = _reconcile(d, capsys)

    index = result["rejection_index"]
    assert code == 0
    assert index["collected"] == REJECTION_COLLECT_UNAVAILABLE
    assert index["owners_queued"] == 0 and index["seed_stamped"] is False
    assert index["missing_items"] is None and index["stale_items"] is None
    assert d.row(owner) is None and d.store().rejection_seed() is None
    assert "The rejection index could not be measured against the graph" in err
    assert result["expected"] == 1 and result["enqueued"] == 1  # the decision heal intact


def test_r_faults_a_refused_chunk_is_counted_and_the_next_is_tried(
        d, monkeypatch, capsys) -> None:
    owner = _seat(d, "one", F_ONE)
    _hold(d, owner, F_ONE)
    _plant(d, "11111111-1111-1111-1111-111111111111", C)
    _plant(d, "22222222-2222-2222-2222-222222222222", C)
    monkeypatch.setattr("mitos.sync._ORPHAN_DELETE_CHUNK", 1)
    deletes: List[int] = []

    def refuse_first(name: str, details: Dict[str, Any]) -> None:
        if name == "delete_points":
            deletes.append(1)
            if len(deletes) == 1:
                raise VectorStoreError("planted refusal")

    d.index.on_call = refuse_first

    _code, result, err = _reconcile(d, capsys)

    index = result["rejection_index"]
    assert (index["deleted"], index["delete_failed"]) == (1, 1)
    assert len(deletes) == 2
    assert "1 rejected item(s) not deleted" in err


def test_r_faults_an_unreachable_delete_stops_and_counts_every_unconfirmed(
        d, monkeypatch, capsys) -> None:
    owner = _seat(d, "one", F_ONE)
    _hold(d, owner, F_ONE)
    _plant(d, "11111111-1111-1111-1111-111111111111", C)
    _plant(d, "22222222-2222-2222-2222-222222222222", C)
    monkeypatch.setattr("mitos.sync._ORPHAN_DELETE_CHUNK", 1)

    def unreachable(name: str, details: Dict[str, Any]) -> None:
        if name == "delete_points":
            raise VectorStoreUnreachableError("planted outage")

    d.index.on_call = unreachable

    _code, result, err = _reconcile(d, capsys)

    index = result["rejection_index"]
    assert (index["deleted"], index["delete_failed"]) == (0, 2)
    assert len([c for c in d.index.calls if c[0] == "delete_points"]) == 1
    assert "Stopped deleting rejected items; 2 not deleted" in err


def test_r_faults_a_graph_fault_under_the_lock_is_unread(d, monkeypatch, capsys) -> None:
    owner = _seat(d, "one", F_ONE)
    _hold(d, owner, F_ONE)
    _plant(d, "11111111-1111-1111-1111-111111111111", C)
    calls: List[int] = []
    real = GraphStore.get_rejection_index_owner_fields

    def second_faults(self):
        calls.append(1)
        if len(calls) == 2:
            raise DatabaseError("planted R5 fault")
        return real(self)

    monkeypatch.setattr(GraphStore, "get_rejection_index_owner_fields", second_faults)

    code, out, err = _reconcile_text(d, capsys)

    assert code == 0
    assert "11111111-1111-1111-1111-111111111111" in d.index.points
    assert "planted R5 fault" in err
    assert (f"Rejection index not collected — the graph could not be read, so nothing "
            f"was deleted from it; `mitos reconcile {_sel(d)}` again collects it.") in out


def test_r_keys_every_path_carries_the_rejection_index_key_set(d, absent, capsys) -> None:
    results = [d.manager().reconcile_embeddings()]
    down = d.manager()
    down.embed_provider = None
    results.append(down.reconcile_embeddings())
    no_index = d.manager()
    no_index.rejection_index = None
    results.append(no_index.reconcile_embeddings())
    d.index.unreachable = True
    results.append(d.manager().reconcile_embeddings())

    for result in results:
        assert tuple(result["rejection_index"]) == _INDEX_KEYS
        assert result["rejection_index"]["collected"] in REJECTION_COLLECT_OUTCOMES
    for silent in results[1:3]:
        assert silent["rejection_index"] == {
            "expected_items": None, "expected_owners": None, "present_items": None,
            "collection_present": None, "missing_items": None, "owners_queued": 0,
            "seed_stamped": False, "stale_items": None, "orphan_items": None,
            "deleted": 0, "delete_failed": 0, "collected": REJECTION_COLLECT_UNAVAILABLE,
        }


def test_r_keys_the_silent_paths_print_nothing_about_the_rejection_index(d, capsys) -> None:
    no_index = d.manager()
    no_index.rejection_index = None
    capsys.readouterr()
    no_index.reconcile_embeddings()
    out, err = capsys.readouterr()
    assert "rejection index" not in (out + err).lower()


# --------------------------------------------------------------------------- #
# S-* — status
# --------------------------------------------------------------------------- #


def _state_absent_gap(d: Rig) -> None:
    d.index.present = False
    _seat(d, "one", F_ABC)


def _state_absent_no_gap(d: Rig) -> None:
    d.index.present = False
    _open_question(d.store(), "q-only", "Only a question?")


def _state_present_complete(d: Rig) -> None:
    owner = _seat(d, "one", F_ABC)
    _hold(d, owner, F_ABC)


def _state_missing(d: Rig) -> None:
    owner = _seat(d, "one", F_ABC)
    _hold(d, owner, F_ONE.replace("Echo", "Other"))
    _hold(d, owner, F_ABC)
    for pid in sorted(_ids(owner, F_ABC))[:2]:
        d.index.points.pop(pid)
    for pid in list(d.index.points):
        if pid not in _ids(owner, F_ABC):
            d.index.points.pop(pid)


def _state_stale(d: Rig) -> None:
    owner = _seat(d, "one", F_ABC)
    _hold(d, owner, F_ABC)
    _hold(d, owner, F_ONE)


def _state_orphan(d: Rig) -> None:
    owner = _seat(d, "one", F_ABC)
    _hold(d, owner, F_ABC)
    _plant(d, "11111111-1111-1111-1111-111111111111", C)


def _state_zero_node(d: Rig) -> None:
    _plant(d, "11111111-1111-1111-1111-111111111111", A)


def _expected_abc() -> int:
    return len(_ids(A, F_ABC))


S_STATES: Dict[str, Tuple[Callable[[Rig], None], Callable[[], Dict[str, Any]]]] = {
    "absent-gap": (_state_absent_gap, lambda: dict(
        rejection_collection_exists=False, rejection_expected_items=_expected_abc(),
        missing_rejection_items=_expected_abc(), stale_rejection_items=0,
        orphan_rejection_items=0, rejection_backlog_past_tolerance=0)),
    "absent-no-gap": (_state_absent_no_gap, lambda: dict(
        rejection_collection_exists=False, rejection_expected_items=0,
        missing_rejection_items=0, stale_rejection_items=0, orphan_rejection_items=0,
        rejection_backlog_past_tolerance=0)),
    "present-complete": (_state_present_complete, lambda: dict(
        rejection_collection_exists=True, rejection_expected_items=_expected_abc(),
        missing_rejection_items=0, stale_rejection_items=0, orphan_rejection_items=0,
        rejection_backlog_past_tolerance=0)),
    "missing": (_state_missing, lambda: dict(
        rejection_collection_exists=True, rejection_expected_items=_expected_abc(),
        missing_rejection_items=2, stale_rejection_items=0, orphan_rejection_items=0,
        rejection_backlog_past_tolerance=0)),
    "stale": (_state_stale, lambda: dict(
        rejection_collection_exists=True, rejection_expected_items=_expected_abc(),
        missing_rejection_items=0, stale_rejection_items=1, orphan_rejection_items=0,
        rejection_backlog_past_tolerance=0)),
    "orphan": (_state_orphan, lambda: dict(
        rejection_collection_exists=True, rejection_expected_items=_expected_abc(),
        missing_rejection_items=0, stale_rejection_items=0, orphan_rejection_items=1,
        rejection_backlog_past_tolerance=0)),
    "zero-node": (_state_zero_node, lambda: dict(
        rejection_collection_exists=True, rejection_expected_items=0,
        missing_rejection_items=0, stale_rejection_items=None, orphan_rejection_items=None,
        rejection_backlog_past_tolerance=0)),
}


@pytest.mark.parametrize("state", list(S_STATES))
def test_s_keys_status_reports_the_index_from_the_diff(d, capsys, state) -> None:
    build, expected = S_STATES[state]
    build(d)
    _code, payload = _status(d, capsys)
    assert payload["rejection_collection"] == d.config.rejection_collection
    assert payload["rejection_collection"] != payload["collection"]
    assert _rkeys(payload) == expected()


def test_s_keys_an_unreachable_qdrant_leaves_the_listing_side_unknown(
        d, monkeypatch, capsys) -> None:
    _state_present_complete(d)
    monkeypatch.setattr(cli, "_check_qdrant", lambda url, coll: {
        "reachable": False, "collection_exists": False, "points": None})

    _code, payload = _status(d, capsys)
    code, out = _status_text(d, capsys)

    assert _rkeys(payload) == dict(
        rejection_collection_exists=None, rejection_expected_items=_expected_abc(),
        missing_rejection_items=None, stale_rejection_items=None,
        orphan_rejection_items=None, rejection_backlog_past_tolerance=0)
    assert not [c for c in d.index.calls if c[0] == "list_point_owners"]
    assert f"— rejection index '{d.config.rejection_collection}'   → needs Qdrant up" in out


@pytest.mark.parametrize("state", [s for s in S_STATES])
def test_s_ready_no_rejection_state_moves_ready_the_verdict_or_the_exit(
        d, capsys, state) -> None:
    S_STATES[state][0](d)
    code, payload = _status(d, capsys)
    text_code, out = _status_text(d, capsys)
    assert (payload["ready"], code, text_code) == (True, 0, 0)
    assert "READY ✓" in out.splitlines()[1]


def test_s_text_absent_with_a_gap_prices_the_heal_on_the_row(absent, capsys) -> None:
    owner = _seat(absent, "one", F_ABC)
    _code, out = _status_text(absent, capsys)
    n = len(_ids(owner, F_ABC))
    assert (f"  — rejection index '{absent.config.rejection_collection}'   → missing — "
            f"{n} rejected item(s) of 1 decision(s) have no point; run "
            f"`mitos reconcile {_sel(absent)}` ({n} embedding(s) — this index keeps no "
            f"embedding cache)") in out


def test_s_text_each_rung_names_the_selectored_reconcile_and_one_read_claim(
        d, capsys) -> None:
    """4d planted the word-absence loop for 5a to invert, on the stale line only: the
    ranked reads now verify every match against its decision's current field, so the
    stale line says no read serves those items. Every other rejection line still
    makes no read claim."""
    owner = _seat(d, "one", F_ABC)
    _hold(d, owner, F_ONE)  # stale
    _plant(d, "11111111-1111-1111-1111-111111111111", C)  # orphan; ABC missing
    d.store().stamp_rejection_seed("reconcile")

    _code, out = _status_text(d, capsys)
    lines = _rejection_lines(out)

    n = len(_ids(owner, F_ABC))
    assert (f"  ⚠ rejection index incomplete — {n} rejected item(s) of 1 decision(s) have "
            f"no point. `mitos reconcile {_sel(d)}` embeds them: {n} embedding(s), none "
            f"cached. Informational, not a readiness blocker.") in out
    assert "the rejection backlog was seeded by `reconcile`" in out
    assert f"the next `mitos sync {_sel(d)}` builds it" in out
    stale_line = (f"  • 1 rejected item(s) await removal — their decision's field no "
                  f"longer states them, so no read serves them; `mitos reconcile "
                  f"{_sel(d)}` removes them.")
    assert stale_line in lines
    assert (f"  ⚠ 1 rejected item(s) belong to no decision in the graph — "
            f"`mitos reconcile {_sel(d)}` deletes them.") in out
    assert f"  ✓ rejection index '{d.config.rejection_collection}'" in out
    assert "      (2 item(s) indexed)" in out
    for line in lines:
        assert "None" not in line
        for word in ("find", "match", "serve", "search"):
            if line == stale_line and word == "serve":
                continue
            assert word not in line.lower(), (word, line)


def test_s_healthy_amend_reads_no_warning_and_names_no_backlog(d, capsys) -> None:
    owner = _commit(d, "amended", F_ABC)
    _hold(d, owner, F_ABC)
    d.store().clear_rejection_backlog_row(owner, enqueue_seq=d.row(owner).enqueue_seq)
    _commit(d, "amended", F_BD)  # the amend: its row stands
    _hold(d, owner, F_BD)        # and the inline leg added every new item
    assert d.row(owner) is not None

    _code, payload = _status(d, capsys)
    _code, out = _status_text(d, capsys)

    stale = len(_ids(owner, F_ABC) - _ids(owner, F_BD))
    assert _three(payload) == (0, stale, 0)
    lines = _rejection_lines(out)
    assert not [line for line in lines if "⚠" in line]
    assert any(line.startswith(f"  • {stale} rejected item(s) await removal") for line in lines)
    assert not [line for line in lines if "backlog" in line]


@pytest.mark.parametrize("failures,named", [(REJECTION_BACKLOG_FAILURE_TOLERANCE, True),
                                            (REJECTION_BACKLOG_FAILURE_TOLERANCE - 1, False)])
def test_s_tolerance_a_row_is_named_only_at_the_tolerance(
        d, capsys, failures, named) -> None:
    owner = _commit(d, "poisoned", F_ONE)
    _hold(d, owner, F_ONE)
    store = d.store()
    for _ in range(failures):
        assert store.note_rejection_backlog_failure(owner, enqueue_seq=d.row(owner).enqueue_seq)

    _code, payload = _status(d, capsys)
    _code, out = _status_text(d, capsys)

    assert payload["checks"]["rejection_backlog_past_tolerance"] == (1 if named else 0)
    line = (f"  ⚠ 1 decision(s)' rejected items failed to index "
            f"{REJECTION_BACKLOG_FAILURE_TOLERANCE} or more times; they stay queued, and "
            f"every `mitos sync {_sel(d)}` and `mitos reconcile {_sel(d)}` attempts them "
            f"again.")
    assert (line in out) is named


def test_s_tolerance_reconcile_never_resets_a_poison_rows_count(d, capsys) -> None:
    """R3 queues owners only where no row stands, so a poison row keeps its count."""
    from test_rejection_drain import POISON

    field = f"1. {POISON} — p."
    owner = _commit(d, "poisoned", field)  # its items are missing: R3 would queue it
    store = d.store()
    for _ in range(REJECTION_BACKLOG_FAILURE_TOLERANCE):
        assert store.note_rejection_backlog_failure(owner, enqueue_seq=d.row(owner).enqueue_seq)

    _code, result, _err = _reconcile(d, capsys)
    _code, payload = _status(d, capsys)

    assert result["rejection_index"]["owners_queued"] == 0
    assert d.row(owner).retry_count == REJECTION_BACKLOG_FAILURE_TOLERANCE + 1
    assert payload["checks"]["rejection_backlog_past_tolerance"] == 1


def test_s_questions_a_questions_only_workspace_is_calm(absent, capsys) -> None:
    _open_question(absent.store(), "q-only", "Only a question?")
    _code, payload = _status(absent, capsys)
    _code, out = _status_text(absent, capsys)
    assert _three(payload) == (0, 0, 0)
    assert (f"  — rejection index '{absent.config.rejection_collection}'   → created when "
            f"a decision's rejected items are first indexed — none expected yet") in out
    assert not [line for line in _rejection_lines(out) if "⚠" in line or "•" in line]


def test_s_none_safe_a_graph_fault_renders_the_count_free_hint(
        absent, monkeypatch, capsys) -> None:
    _seat(absent, "one", F_ABC)

    def boom(self):
        raise DatabaseError("planted status read fault")

    monkeypatch.setattr(GraphStore, "get_rejection_index_owner_fields", boom)

    _code, payload = _status(absent, capsys)
    _code, out = _status_text(absent, capsys)

    keys = _rkeys(payload)
    assert keys["rejection_collection_exists"] is False
    assert (keys["rejection_expected_items"], *_three(payload)) == (None, None, None, None)
    assert (f"→ missing — could not read the graph to price it; run "
            f"`mitos reconcile {_sel(absent)}`") in out
    assert "None" not in "\n".join(_rejection_lines(out))


def test_s_unbuilt_the_row_defers_to_the_graph_rung(absent, capsys) -> None:
    _append_entry(absent, "unsynced")
    _code, out = _status_text(absent, capsys)
    assert (f"  — rejection index '{absent.config.rejection_collection}'   → nothing "
            f"indexed yet — the graph is unbuilt (see below)") in out


def test_s_pre_v1a_the_row_defers_to_the_schema_row_and_names_no_reconcile(
        absent, capsys) -> None:
    # A prototype graph is read by nothing here, so the row cannot price a gap — and
    # `reconcile` is not its heal (`cutover` is, on the graph schema row).
    os.remove(absent.config.db_path)
    proto = GraphStore.__new__(GraphStore)
    proto.db_path = absent.config.db_path
    proto.read_only = False
    proto._init_db()

    code, out = _status_text(absent, capsys)
    _code, payload = _status(absent, capsys)

    assert payload["pre_v1a"] is True and code == 1
    assert (f"  — rejection index '{absent.config.rejection_collection}'   → not read — "
            f"the graph predates V1a (see the graph schema row)") in out
    assert not [line for line in _rejection_lines(out) if "reconcile" in line]
    assert (_rkeys(payload)["rejection_expected_items"], *_three(payload)) == (
        None, None, None, None)


# --------------------------------------------------------------------------- #
# C-rebuild / I-import — the marker's other producers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strict,verb", [(False, "rebuild"), (True, "cutover")])
def test_c_rebuild_stamps_its_own_verb_in_the_aside_and_the_next_drain_builds(
        absent, strict, verb) -> None:
    from mitos.cutover import perform_swap, rebuild_and_gate

    for slug, field in (("one", F_ABC), ("two", F_ONE)):
        _append_entry(absent, slug, field)
    config = absent.config
    result = rebuild_and_gate(config, aside_db_path=config.db_path + ".aside", strict=strict)

    aside = GraphStore(result.aside_db_path)
    assert aside.rejection_seed()["established_by"] == verb
    assert aside.embedding_seed()["established_by"] == verb
    assert GraphStore(config.db_path).rejection_seed() is None  # the live graph: none yet

    perform_swap(config, result.aside_db_path, timestamp="20261017-000001")
    live = GraphStore(config.db_path)
    assert live.rejection_seed()["established_by"] == verb

    outcome = absent.manager().drain_rejection_backlog(report=False)

    owners = {n["id"]: live.get_rejection_owner_fields([n["id"]])[n["id"]][1]
              for n in live.get_all_nodes() if n["kind"] == "decision"}
    assert outcome.stop is None and absent.index.created == 1
    assert _all_items(absent) == set(expected_rejection_points(owners))


def test_c_rebuild_a_refused_rebuild_leaves_no_marker(absent) -> None:
    from mitos.cutover import rebuild_and_gate

    _append_entry(absent, "one", F_ABC)
    config = absent.config
    result = rebuild_and_gate(config, aside_db_path=config.db_path + ".aside", strict=False)
    os.remove(result.aside_db_path)  # what a refusal does with the aside file

    assert GraphStore(config.db_path).rejection_seed() is None


def _legacy(tmp_path) -> str:
    path = tmp_path / "legacy.md"
    path.write_text(
        "## Use a queue\nWe chose a queue.\n\n## Keep SQLite\nSQLite stays.\n",
        encoding="utf-8")
    return str(path)


def _import(d: Rig, path: str) -> None:
    from mitos.importer import MitosProseImporter

    MitosProseImporter(d.config).import_from_file(path, use_llm_extract=False)


def test_i_import_into_a_decisionless_graph_stamps_import(absent, tmp_path) -> None:
    _open_question(absent.store(), "q-only", "A question owns no item?")
    _import(absent, _legacy(tmp_path))
    assert absent.store().rejection_seed()["established_by"] == "import"


def test_i_import_into_a_populated_graph_stamps_nothing(absent, tmp_path) -> None:
    _seat(absent, "already", F_ONE)
    _import(absent, _legacy(tmp_path))
    assert absent.store().rejection_seed() is None


def test_i_import_of_nothing_stamps_nothing(absent, tmp_path) -> None:
    path = tmp_path / "empty.md"
    path.write_text("## Only a heading\n", encoding="utf-8")
    _import(absent, str(path))
    _import(absent, str(path))  # the second run imports zero (identical content)
    store = absent.store()
    seed = store.rejection_seed()
    # The first run imported one into an empty graph (stamped); clear it and re-run.
    assert seed is not None and seed["established_by"] == "import"
    store.clear_rejection_seed(established_at=seed["established_at"])
    _import(absent, str(path))
    assert store.rejection_seed() is None


def test_i_import_a_stamp_fault_warns_and_the_import_reports_its_count(
        absent, tmp_path, monkeypatch, capsys) -> None:
    def boom(self, established_by):
        raise DatabaseError("planted stamp fault")

    monkeypatch.setattr(GraphStore, "stamp_rejection_seed", boom)
    capsys.readouterr()
    _import(absent, _legacy(tmp_path))
    out = capsys.readouterr().out
    assert "Successfully imported 2 nodes" in out
    assert "The rejection index's seed marker was not stamped: planted stamp fault" in out
    assert f"`mitos reconcile {_sel(absent)}` builds the rejection index" in out


# --------------------------------------------------------------------------- #
# O-overview — presence from the one listing
# --------------------------------------------------------------------------- #


def _overview_ws(tmp_path, name: str, *, entries: bool) -> str:
    root = _workspace(tmp_path / name)
    if entries:
        with open(os.path.join(root, "decisions.md"), "w", encoding="utf-8") as handle:
            handle.write("<!-- BEGIN ENTRIES -->\n\n### a-slug\n**Decided:** x\n")
    return root


def test_o_overview_reports_presence_from_the_one_listing(tmp_path, qdrant) -> None:
    from mitos import overview
    from mitos.config import MitosConfig

    both = _overview_ws(tmp_path, "both", entries=True)
    lacks = _overview_ws(tmp_path, "lacks", entries=True)
    _register(both=both, lacks=lacks)
    url = MitosConfig(both).qdrant_url
    qdrant.names[url] = [MitosConfig(both).qdrant_collection,
                         MitosConfig(both).rejection_collection,
                         MitosConfig(lacks).qdrant_collection]

    payload = overview.build_overview()

    projects = {p["name"]: p for p in payload["projects"]}
    assert projects["both"]["rejection_collection"] == MitosConfig(both).rejection_collection
    assert projects["both"]["rejection_collection_present"] is True
    assert projects["lacks"]["rejection_collection_present"] is False
    assert qdrant.urls == [f"{url}/collections"]  # one GET per instance, nothing else
    assert cli._overview_notes(projects["both"], payload) == []
    assert cli._overview_notes(projects["lacks"], payload) == [
        f"   rejection index {MitosConfig(lacks).rejection_collection} absent — "
        f"`mitos status 'lacks'` for the heal"]


def test_o_overview_the_note_needs_the_decision_collection_and_a_populated_corpus(
        tmp_path, qdrant) -> None:
    from mitos import overview
    from mitos.config import MitosConfig

    empty = _overview_ws(tmp_path, "empty", entries=False)
    swept = _overview_ws(tmp_path, "swept", entries=True)
    _register(empty=empty, swept=swept)
    url = MitosConfig(empty).qdrant_url
    qdrant.names[url] = [MitosConfig(empty).qdrant_collection]

    payload = overview.build_overview()
    projects = {p["name"]: p for p in payload["projects"]}

    assert cli._overview_notes(projects["empty"], payload) == []
    swept_notes = cli._overview_notes(projects["swept"], payload)
    assert len(swept_notes) == 1 and "no vector collection" in swept_notes[0]


def test_o_overview_an_unusable_listing_leaves_presence_unknown(tmp_path, qdrant) -> None:
    from mitos import overview
    from mitos.config import MitosConfig

    root = _overview_ws(tmp_path, "p", entries=True)
    _register(p=root)
    qdrant.faults[MitosConfig(root).qdrant_url] = VectorStoreUnreachableError("down")

    payload = overview.build_overview()
    assert payload["projects"][0]["rejection_collection_present"] is None


# --------------------------------------------------------------------------- #
# T11 — the upgrade path end to end, through cli.main
# --------------------------------------------------------------------------- #


def _to_step4(db_path: str) -> None:
    """Rewinds a head graph to ladder step 4: the shape every pre-4a graph has."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE rejection_backlog")
        conn.execute("DROP TABLE rejection_seed")
        conn.execute("DROP INDEX IF EXISTS idx_pending_embeddings_queue")
        conn.execute("DELETE FROM pending_embeddings")
        conn.execute("PRAGMA user_version = 4")
        conn.commit()
    finally:
        conn.close()


def _upgraded_workspace(d: Rig) -> Dict[str, str]:
    """A workspace an earlier build wrote: decisions (one retired), active points only."""
    from test_expected_set import _decision as commit_decision

    store = d.store()
    ids = {
        "old": commit_decision(store, "old", "The old axiom."),
        "live": commit_decision(store, "live", "The live axiom."),
    }
    ids["new"] = commit_decision(store, "new", "Its successor.", supersedes=["old"])
    collection = d.vector_cls._collections.setdefault(d.config.qdrant_collection, {})
    for slug in ("live", "new"):
        collection[ids[slug]] = ([1.0] + [0.0] * 7, {"slug": slug, "state": "active"})
    _to_step4(d.config.db_path)
    return ids


def test_t11_the_upgrade_path_end_to_end(absent, capsys) -> None:
    d = absent
    ids = _upgraded_workspace(d)

    # A ranked read over the upgraded graph answers its decision ranking whole (the
    # writable open applies the rung).
    for verb, key in (("surface", "active_decisions"), ("query", "matches")):
        capsys.readouterr()
        _run(["-p", d.root, verb, "the live axiom", "--json"])
        answer = json.loads(capsys.readouterr().out)
        assert {m["slug"] for m in answer[key]} == {"live", "new"}, answer
    conn = sqlite3.connect(d.config.db_path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 5
    finally:
        conn.close()

    # status names both gaps, each with reconcile.
    _code, before = _status(d, capsys)
    _code, text = _status_text(d, capsys)
    expected_items = len(expected_rejection_points(
        d.store().get_rejection_index_owner_fields()))
    assert before["checks"]["missing_retired_vectors"] == 1
    assert before["checks"]["missing_rejection_items"] == expected_items == 3
    assert f"`mitos reconcile {_sel(d)}`" in text
    assert "retired decision(s) have no vector" in text
    assert "rejected item(s) of 3 decision(s) have no point" in text

    # A record and an amend queue rows and create nothing.
    recorded = _mcp_record(d, "recorded")["id"]
    _mcp_amend(d, "recorded", rejected_paths=F_BD)
    assert d.index.created == 0 and not d.index.present
    assert d.store().count_rejection_backlog() == 1  # the amend re-queued the record's row
    assert d.row(recorded) is not None

    # One reconcile re-seeds the graveyard and builds the rejection index.
    code, result, _err = _reconcile(d, capsys)
    assert code == 0
    assert result["enqueued"] == 1 and result["written"] == 1  # the retired decision
    index = result["rejection_index"]
    # The three the earlier build wrote are queued; the recorded one already stands.
    assert index["seed_stamped"] is True and index["owners_queued"] == 3
    drain = result["rejection_drain"]
    assert drain["calls"] >= 1 and drain["items_skipped_present"] == 0
    assert drain["stop"] is None and drain["left_queued"] == 0

    _code, after = _status(d, capsys)
    assert after["checks"]["missing_retired_vectors"] == 0
    assert _three(after) == (0, 0, 0)
    assert ids["old"] in d.vector_cls._collections[d.config.qdrant_collection]


def test_t11_rollback_a_write_the_previous_build_made_reads_stale_and_missing(
        d, capsys) -> None:
    owner = _seat(d, "one", F_ABC)
    _hold(d, owner, F_ABC)
    # The previous build amends: the field changes, no row is queued, nothing indexed.
    _commit(d, "one", F_BD)
    _seat(d, "one", F_BD)

    _code, before = _status(d, capsys)
    stale = len(_ids(owner, F_ABC) - _ids(owner, F_BD))
    missing = len(_ids(owner, F_BD) - _ids(owner, F_ABC))
    assert _three(before) == (missing, stale, 0)

    _code, result, _err = _reconcile(d, capsys)
    assert result["rejection_index"]["deleted"] == stale
    assert _all_items(d) == _ids(owner, F_BD)
    _code, after = _status(d, capsys)
    assert _three(after) == (0, 0, 0)
