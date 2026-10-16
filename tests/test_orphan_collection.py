"""The decision index's death story: `reconcile` collects node-less points, over a built graph only.

Phase 3c of the recall-confidence vision (ADRs
``reconcile-heals-every-decision-vector-and-collects-only-node-less-points`` and
``reconcile-orphan-collection-runs-over-a-built-graph-and-expects-active-open-questions``).
A point whose node is still in the graph but not active is **history**, kept; a point
whose node is not in the graph at all is **garbage**, and `reconcile` deletes it by its
listed id. `status` and `reconcile` partition the non-active points with one function
(`vector_store.partition_points`), so the heal clears its own warning by construction.
Over a graph holding no nodes every point would look node-less, so neither surface
judges there, and `reconcile` over an unbuilt clone refuses, names `rebuild`, exits 1.

The rows (plan §8.1):

* O1 — an orphan deleted and counted; retired nodes' points (killed open questions
  included) kept; a second run deletes nothing.
* O2 — a record landing mid-collection keeps its point, on `reconcile` and in `status`.
* O3 — the unbuilt graph and its neighbours: no graph file, an empty graph file, a
  zero-node graph with an empty corpus, a fresh `init`, an unbuilt clone over an absent
  collection, and a zero-node graph beside an unreadable corpus.
* O4 — a failing delete is reported and not counted; unreachable and missing stop.
* O5 — `status`'s partition sums to `orphan_points`; the not-judged states.
* O6 — one partition on both surfaces over §3.7's four states; agreement with
  `resolve_point_ids`.
* O7 — the heal clears its own warning.
* O8 — `status` over a zero-node graph with points names no collector.
* O9 — the shipped keys keep their values; every `reconcile` return path carries the
  same key set.
* O10 — `get_all_node_ids` agrees with `get_all_nodes`.

Run under ``./venv/bin/python -m pytest`` (PATTERNS — bare ``python`` lacks deps).
"""

import json
import os
import stat
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from unittest.mock import patch

import pytest

import mitos.sync
from mitos import cli
from mitos.config import MitosConfig
from mitos.errors import (CollectionMissingError, VectorStoreError,
                          VectorStoreUnreachableError)
from mitos.shell_quote import shell_flag
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager
from mitos.vector_store import hash_to_uuid, partition_points

from test_expected_set import _decision, _graveyard, _matrix, _open_question, _status_ws
from test_outbox_drain import _Embed
from test_status_deep_report import _append_entry


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

#: Two ids no node carries, in Qdrant's canonical (lowercase, hyphenated) form.
NODELESS = (hash_to_uuid("a" * 64), hash_to_uuid("b" * 64))


class _Index:
    """A collection with a real point set, a delete that records, and planted faults.

    ``delete_faults`` maps a delete call's index to the exception it raises.
    ``on_list`` runs inside each listing (``reconcile``'s and ``status``' scroll alike)
    before the points are returned, for the mid-collection rows.
    """

    def __init__(self, points: Sequence[str] = (), *, absent: bool = False,
                 delete_faults: Optional[Dict[int, BaseException]] = None,
                 on_list: Optional[Callable[["_Index"], None]] = None) -> None:
        self.points: Set[str] = set(points)
        self.absent = absent
        self.delete_faults = dict(delete_faults or {})
        self.on_list = on_list
        self.delete_calls: List[List[str]] = []

    def _missing(self) -> CollectionMissingError:
        return CollectionMissingError(
            "Qdrant collection 'mitos-tmp-orphans' does not exist",
            collection="mitos-tmp-orphans")

    def list_point_ids(self, page_size: int = 256) -> Set[str]:
        if self.absent:
            raise self._missing()
        if self.on_list is not None:
            self.on_list(self)
        return set(self.points)

    def upsert(self, point_id: str, vector: List[float], payload: Dict[str, Any],
               *, may_create: bool) -> None:
        if self.absent:
            if not may_create:
                raise self._missing()
            self.absent = False
        self.points.add(hash_to_uuid(point_id))

    def query(self, vector: List[float], limit: int = 5, *,
              with_payload: bool = True) -> List[Dict[str, Any]]:
        return []

    def delete_points(self, point_ids: Sequence[str]) -> None:
        call = len(self.delete_calls)
        self.delete_calls.append(list(point_ids))
        if call in self.delete_faults:
            raise self.delete_faults[call]
        if self.absent:
            raise self._missing()
        self.points.difference_update(point_ids)


def _status_reads(monkeypatch, index: _Index) -> None:
    """Points `status`' collection probe and scroll at ``index``."""
    monkeypatch.setattr(cli, "_check_qdrant", lambda url, coll: {
        "reachable": True, "collection_exists": not index.absent,
        "points": None if index.absent else len(index.points)})

    def _scroll(base_url: str, collection: str, page_size: int = 256) -> Set[str]:
        return index.list_point_ids(page_size)

    monkeypatch.setattr(cli, "scroll_point_ids", _scroll)


def _status(config: MitosConfig, capsys, *, as_json: bool) -> Tuple[int, Any]:
    capsys.readouterr()
    rc = cli.cmd_status(config.workspace_dir, as_json=as_json)
    out = capsys.readouterr().out
    return rc, (json.loads(out)["checks"] if as_json else out)


def _manager(config: MitosConfig, index: _Index,
             embed: Optional[_Embed] = None) -> MitosSyncManager:
    manager = MitosSyncManager(config)
    manager.embed_provider = embed if embed is not None else _Embed()  # type: ignore[assignment]
    manager.vector_store = index  # type: ignore[assignment]
    return manager


def _reconcile(config: MitosConfig, index: _Index, capsys, *, as_json: bool,
               manager: Optional[MitosSyncManager] = None) -> Tuple[int, str, str]:
    """Runs ``cmd_reconcile`` against ``index``; returns ``(rc, stdout, stderr)``."""
    manager = manager if manager is not None else _manager(config, index)
    capsys.readouterr()
    with patch("mitos.cli.MitosSyncManager", return_value=manager):
        rc = cli.cmd_reconcile(config, as_json=as_json)
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def _clear_outbox(config: MitosConfig) -> None:
    store = GraphStore(config.db_path)
    for row in store.get_pending_embeddings():
        store.remove_pending_embedding(row["node_id"])


def _full_graveyard(config: MitosConfig) -> Dict[str, str]:
    """`_graveyard` plus a corrected open question: both kill types on both kinds."""
    ids = _graveyard(config)
    store = GraphStore(config.db_path)
    ids["q-typo"] = _open_question(store, "q-typo", "Typo?")
    ids["q-fixed"] = _open_question(store, "q-fixed", "Fixed?", corrects=["q-typo"])
    _clear_outbox(config)
    return ids


_ACTIVE = ("live-a", "new-b", "new-c", "q-new", "q-fixed")
_RETIRED = ("old-b", "old-c", "q-old", "q-typo")


def _every_point(ids: Dict[str, str]) -> Set[str]:
    return {hash_to_uuid(node_id) for node_id in ids.values()}


def _clone(tmp_path, monkeypatch, name: str = "clone") -> MitosConfig:
    """An unbuilt clone: config and a corpus holding an entry, NO graph file."""
    (tmp_path / name).mkdir()
    config = _status_ws(tmp_path / name, monkeypatch)
    _append_entry(config.workspace_dir)
    os.remove(config.db_path)
    return MitosConfig(config.workspace_dir)


def _ws(tmp_path, monkeypatch, name: str = "ws") -> MitosConfig:
    (tmp_path / name).mkdir()
    return _status_ws(tmp_path / name, monkeypatch)


# --------------------------------------------------------------------------- #
# O1 — an orphan deleted and counted
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("as_json", [False, True])
def test_o1_reconcile_deletes_exactly_the_nodeless_points_and_keeps_history(
    tmp_path, monkeypatch, capsys, as_json
) -> None:
    config = _ws(tmp_path, monkeypatch)
    ids = _full_graveyard(config)
    index = _Index(_every_point(ids) | set(NODELESS))

    rc, out, err = _reconcile(config, index, capsys, as_json=as_json)

    assert rc == 0
    # The ids reach the delete verbatim, as the listing gave them — never re-hashed.
    assert index.delete_calls == [sorted(NODELESS)]
    assert index.points == _every_point(ids)  # history kept, killed open questions too
    for slug in _RETIRED:
        assert hash_to_uuid(ids[slug]) in index.points
    if as_json:
        result = json.loads(out)
        assert result["deleted"] == 2
        assert result["delete_failed"] == 0
        assert result["nodeless_points"] == 2
        assert result["retired_node_points"] == len(_RETIRED)
        assert result["graph_unbuilt"] is False
    else:
        assert "Collected: 2 node-less point(s) deleted." in out
        assert f"{len(_RETIRED)} point(s) of retired nodes kept as history." in out
        assert "could not be deleted" not in out

    # Idempotent: a second run judges the same history and deletes nothing.
    rc, out, _err = _reconcile(config, index, capsys, as_json=True)
    assert rc == 0
    second = json.loads(out)
    assert len(index.delete_calls) == 1
    assert (second["deleted"], second["nodeless_points"]) == (0, 0)
    assert second["retired_node_points"] == len(_RETIRED)


def test_o1_the_summary_path_leads_with_the_echo_on_stdout_even_when_a_drain_ran(
    tmp_path, monkeypatch, capsys
) -> None:
    """The echo rides the response's stream: the summary answers on stdout, so the
    echo is its first line, ahead of the drain's progress lines."""
    config = _ws(tmp_path, monkeypatch)
    ids = _full_graveyard(config)
    present = _every_point(ids) - {hash_to_uuid(ids["old-b"])}
    index = _Index(present | set(NODELESS))

    rc, out, err = _reconcile(config, index, capsys, as_json=False)

    assert rc == 0
    lines = out.splitlines()
    assert lines[0] == (f"corpus: {config.project} · {config.qdrant_collection} · "
                        f"{config.workspace_dir}")
    assert "Draining pending embeddings queue" in out
    assert "Not reconciled" not in err


# --------------------------------------------------------------------------- #
# O2 — a record landing mid-collection keeps its point
# --------------------------------------------------------------------------- #

def _late_record(config: MitosConfig, landed: List[str]) -> Callable[[_Index], None]:
    """Commits a record (node, then its point) inside the listing, once; its id lands
    in ``landed``."""

    def _on_list(index: _Index) -> None:
        if landed:
            return
        node_id = _decision(GraphStore(config.db_path), "late", "A late axiom.")
        landed.append(node_id)
        index.points.add(hash_to_uuid(node_id))

    return _on_list


def test_o2_reconcile_keeps_a_point_whose_record_landed_during_the_listing(
    tmp_path, monkeypatch, capsys
) -> None:
    config = _ws(tmp_path, monkeypatch)
    ids = _full_graveyard(config)
    landed: List[str] = []
    index = _Index(_every_point(ids), on_list=_late_record(config, landed))

    rc, out, _err = _reconcile(config, index, capsys, as_json=True)

    result = json.loads(out)
    assert rc == 0
    assert result["nodeless_points"] == 0
    assert result["deleted"] == 0
    assert index.delete_calls == []
    assert hash_to_uuid(landed[0]) in index.points


def test_o2_status_counts_no_nodeless_point_for_a_record_landing_during_its_scroll(
    tmp_path, monkeypatch, capsys
) -> None:
    config = _ws(tmp_path, monkeypatch)
    ids = _full_graveyard(config)
    landed: List[str] = []
    index = _Index(_every_point(ids), on_list=_late_record(config, landed))
    _status_reads(monkeypatch, index)

    _rc, checks = _status(config, capsys, as_json=True)

    assert landed  # the record really landed inside the scroll
    assert checks["nodeless_points"] == 0
    assert checks["retired_node_points"] == len(_RETIRED)


# --------------------------------------------------------------------------- #
# O3 — the unbuilt graph and its neighbours
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("graph_file", ["absent", "empty"])
def test_o3ab_reconcile_over_an_unbuilt_graph_deletes_nothing_and_refuses(
    tmp_path, monkeypatch, capsys, as_json, graph_file
) -> None:
    """(a) no graph file — constructing the manager creates it, so on `reconcile` (a)
    arrives as (b); (b) the empty file a read surface leaves behind."""
    config = _clone(tmp_path, monkeypatch)
    if graph_file == "empty":
        GraphStore(config.db_path)  # what any read surface's construction does
    index = _Index(set(NODELESS) | {hash_to_uuid("c" * 64)})

    rc, out, err = _reconcile(config, index, capsys, as_json=as_json)

    assert rc == 1
    assert index.delete_calls == []
    assert len(index.points) == 3
    assert "Reconciled" not in out and "Reconciled" not in err
    if as_json:
        result = json.loads(out)  # still exactly one object on stdout
        assert result["graph_unbuilt"] is True
        assert result["nodeless_points"] is None
        assert result["retired_node_points"] is None
        assert result["deleted"] == 0
        assert result["workspace"] == config.workspace_dir
    else:
        assert out == ""
        assert "the graph holds no nodes" in err
        assert "nothing was deleted" in err
        assert f"`mitos rebuild {shell_flag('-p', config.project)}`" in err
        assert "mitos reconcile" not in err
        assert config.qdrant_collection in err  # the echo, on the refusal's stream


def test_o3a_status_over_no_graph_file_reads_every_point_key_unknown(
    tmp_path, monkeypatch, capsys
) -> None:
    """The true no-file shape reaches neither the scroll nor the absent branch, so
    all three point keys stay `None` and no not-judged line prints; the unbuilt rung
    names `rebuild`."""
    config = _clone(tmp_path, monkeypatch)
    index = _Index(set(NODELESS))
    _status_reads(monkeypatch, index)

    _rc, checks = _status(config, capsys, as_json=True)
    assert (checks["orphan_points"], checks["retired_node_points"],
            checks["nodeless_points"]) == (None, None, None)
    assert checks["graph_unbuilt"] is True
    _rc, out = _status(config, capsys, as_json=False)
    assert "not judged" not in out
    assert "mitos rebuild" in out


def test_o3c_a_zero_node_graph_with_an_empty_corpus_is_not_judged_and_exits_0(
    tmp_path, monkeypatch, capsys
) -> None:
    config = _ws(tmp_path, monkeypatch)  # a fresh init: zero nodes, sample-only corpus
    index = _Index(set(NODELESS))

    rc, out, err = _reconcile(config, index, capsys, as_json=False)

    assert rc == 0
    assert index.delete_calls == []
    assert "Reconciled:" in out
    assert ("2 point(s) not judged — the graph holds no nodes; reconcile deletes "
            "points only over a built graph.") in out
    assert "rebuild" not in out + err

    rc, out, _err = _reconcile(config, index, capsys, as_json=True)
    result = json.loads(out)
    assert rc == 0
    assert (result["retired_node_points"], result["nodeless_points"]) == (None, None)
    assert result["graph_unbuilt"] is False


@pytest.mark.parametrize("as_json", [False, True])
def test_o3e_a_fresh_init_over_an_absent_collection_judges_an_empty_listing(
    tmp_path, monkeypatch, capsys, as_json
) -> None:
    config = _ws(tmp_path, monkeypatch)
    index = _Index(absent=True)

    rc, out, err = _reconcile(config, index, capsys, as_json=as_json)

    assert rc == 0
    assert "not judged" not in out + err
    if as_json:
        result = json.loads(out)
        assert (result["retired_node_points"], result["nodeless_points"]) == (0, 0)
        assert result["graph_unbuilt"] is False


@pytest.mark.parametrize("as_json", [False, True])
def test_o3f_an_unbuilt_clone_over_an_absent_collection_refuses_with_zeros(
    tmp_path, monkeypatch, capsys, as_json
) -> None:
    config = _clone(tmp_path, monkeypatch)
    index = _Index(absent=True)

    rc, out, err = _reconcile(config, index, capsys, as_json=as_json)

    assert rc == 1
    if as_json:
        result = json.loads(out)
        assert (result["retired_node_points"], result["nodeless_points"]) == (0, 0)
        assert result["graph_unbuilt"] is True
    else:
        assert "Not reconciled" in err and "mitos rebuild" in err


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="environmental: root reads a mode-000 file, so no corpus "
                           "is unreadable; run as a normal user")
def test_o3d_a_zero_node_graph_beside_an_unreadable_corpus_deletes_nothing(
    tmp_path, monkeypatch, capsys
) -> None:
    """G3: the corpus scan reads an unreadable file as empty, so the unbuilt predicate
    says "built and empty" — the guard keys on the node count and still deletes
    nothing."""
    config = _clone(tmp_path, monkeypatch)
    GraphStore(config.db_path)
    os.chmod(config.decisions_file, 0)
    try:
        index = _Index(set(NODELESS))
        rc, out, err = _reconcile(config, index, capsys, as_json=True)
    finally:
        os.chmod(config.decisions_file, stat.S_IRUSR | stat.S_IWUSR)

    result = json.loads(out)
    assert result["graph_unbuilt"] is False  # the predicate was fooled
    assert index.delete_calls == []  # the guard was not
    assert index.points == set(NODELESS)
    assert result["nodeless_points"] is None
    assert rc == 0


def test_o3_a_keyless_unbuilt_clone_warns_then_refuses_and_exits_1(
    tmp_path, monkeypatch, capsys
) -> None:
    """The provider-down return still computes `graph_unbuilt`: a keyless clone is
    still a clone, so it now exits 1 where it used to print zeros and exit 0."""
    config = _clone(tmp_path, monkeypatch)
    manager = MitosSyncManager(config)
    manager.embed_provider = None
    manager.vector_store = None

    rc, out, err = _reconcile(config, _Index(), capsys, as_json=False, manager=manager)

    assert rc == 1
    assert "Cannot reconcile: Embedding provider or vector store down." in err
    assert "Not reconciled" in err
    assert err.index("Cannot reconcile") < err.index("Not reconciled")
    assert "Reconciled" not in out


# --------------------------------------------------------------------------- #
# O4 — a failing delete is reported and not counted
# --------------------------------------------------------------------------- #

_SIX = tuple(sorted(hash_to_uuid(c * 64) for c in "abcdef"))


def _three_chunks(tmp_path, monkeypatch, faults: Dict[int, BaseException]
                  ) -> Tuple[MitosConfig, _Index]:
    monkeypatch.setattr(mitos.sync, "_ORPHAN_DELETE_CHUNK", 2)
    config = _ws(tmp_path, monkeypatch)
    node_id = _decision(GraphStore(config.db_path), "live", "A live axiom.")
    _clear_outbox(config)
    return config, _Index({hash_to_uuid(node_id), *_SIX}, delete_faults=faults)


def test_o4_a_refused_chunk_is_counted_failed_and_the_next_is_tried(
    tmp_path, monkeypatch, capsys
) -> None:
    config, index = _three_chunks(
        tmp_path, monkeypatch, {1: VectorStoreError("Qdrant delete failed: 400")})

    rc, out, err = _reconcile(config, index, capsys, as_json=False)

    assert rc == 0  # re-running reconcile completes the heal
    assert index.delete_calls == [list(_SIX[0:2]), list(_SIX[2:4]), list(_SIX[4:6])]
    assert index.points & set(_SIX) == set(_SIX[2:4])
    assert ("Collected: 4 node-less point(s) deleted, 2 could not be deleted — Qdrant "
            "refused or did not confirm; `mitos reconcile -p") in out
    assert "again retries them." in out
    assert "2 node-less point(s) not deleted" in err

    rc, out, _err = _reconcile(config, index, capsys, as_json=True)
    result = json.loads(out)
    assert (result["deleted"], result["delete_failed"]) == (2, 0)  # the retry


@pytest.mark.parametrize("fault, at", [
    (VectorStoreUnreachableError("Qdrant connection error during delete"), 1),
    (CollectionMissingError("gone", collection="mitos-tmp-orphans"), 0),
])
def test_o4_unreachable_or_missing_stops_the_loop_and_counts_the_rest_failed(
    tmp_path, monkeypatch, capsys, fault, at
) -> None:
    config, index = _three_chunks(tmp_path, monkeypatch, {at: fault})

    rc, out, err = _reconcile(config, index, capsys, as_json=True)

    result = json.loads(out)
    assert rc == 0
    assert len(index.delete_calls) == at + 1  # nothing sent after the stop
    assert result["deleted"] == 2 * at
    assert result["delete_failed"] == 6 - 2 * at
    assert "Stopped deleting node-less points" in err


# --------------------------------------------------------------------------- #
# O5 — status: the partition sums to orphan_points
# --------------------------------------------------------------------------- #

def test_o5_status_partition_sums_to_the_orphan_count(tmp_path, monkeypatch, capsys) -> None:
    config = _ws(tmp_path, monkeypatch)
    ids = _full_graveyard(config)
    present = _every_point(ids) | set(NODELESS)
    _status_reads(monkeypatch, _Index(present))

    _rc, checks = _status(config, capsys, as_json=True)

    active = {hash_to_uuid(ids[s]) for s in _ACTIVE}
    assert checks["orphan_points"] == len(present - active)  # its shipped value
    assert checks["retired_node_points"] == len(_RETIRED)
    assert checks["nodeless_points"] == 2
    assert checks["retired_node_points"] + checks["nodeless_points"] == checks["orphan_points"]


def test_o5_the_sum_holds_when_the_active_read_names_an_id_the_node_read_lacks(
    tmp_path, monkeypatch, capsys
) -> None:
    """A rebuild swap between the two reads: the partition intersects active with the
    nodes, so the sum is an identity rather than a coincidence."""
    config = _ws(tmp_path, monkeypatch)
    ids = _full_graveyard(config)
    phantom = "f" * 64
    present = _every_point(ids) | {hash_to_uuid(phantom)}
    _status_reads(monkeypatch, _Index(present))
    real = GraphStore.get_active_node_ids
    monkeypatch.setattr(GraphStore, "get_active_node_ids",
                        lambda self: real(self) | {phantom})

    _rc, checks = _status(config, capsys, as_json=True)

    assert checks["nodeless_points"] == 1
    assert checks["retired_node_points"] + checks["nodeless_points"] == checks["orphan_points"]


def test_o5_a_scroll_failure_leaves_all_three_unknown(tmp_path, monkeypatch, capsys) -> None:
    config = _ws(tmp_path, monkeypatch)
    _full_graveyard(config)
    monkeypatch.setattr(cli, "_check_qdrant", lambda url, coll: {
        "reachable": True, "collection_exists": True, "points": 9})

    def _fails(base_url, collection, page_size=256):
        raise VectorStoreError("Qdrant scroll connection error")

    monkeypatch.setattr(cli, "scroll_point_ids", _fails)
    _rc, checks = _status(config, capsys, as_json=True)
    assert (checks["orphan_points"], checks["retired_node_points"],
            checks["nodeless_points"]) == (None, None, None)


def test_o5_an_absent_collection_over_a_populated_graph_reads_all_three_zero(
    tmp_path, monkeypatch, capsys
) -> None:
    config = _ws(tmp_path, monkeypatch)
    _full_graveyard(config)
    _status_reads(monkeypatch, _Index(absent=True))
    _rc, checks = _status(config, capsys, as_json=True)
    assert (checks["orphan_points"], checks["retired_node_points"],
            checks["nodeless_points"]) == (0, 0, 0)


def test_o5_a_graph_fault_after_the_scroll_leaves_the_point_keys_unknown_only(
    tmp_path, monkeypatch, capsys
) -> None:
    """The pair read has its own fault boundary: the slug diffs stay as computed and
    the scroll-failure line never prints, because the scroll did not fail."""
    config = _ws(tmp_path, monkeypatch)
    ids = _full_graveyard(config)
    present = (_every_point(ids) - {hash_to_uuid(ids["live-a"]),
                                    hash_to_uuid(ids["old-b"])}) | set(NODELESS)
    _status_reads(monkeypatch, _Index(present))

    def _boom(self):
        raise RuntimeError("graph unreadable mid-report")

    monkeypatch.setattr(GraphStore, "get_all_node_ids", _boom)

    _rc, checks = _status(config, capsys, as_json=True)
    assert (checks["orphan_points"], checks["retired_node_points"],
            checks["nodeless_points"]) == (None, None, None)
    assert checks["missing_active_slugs"] == ["live-a"]
    assert checks["missing_retired_slugs"] == ["old-b"]
    _rc, out = _status(config, capsys, as_json=False)
    assert "could not verify vector completeness" not in out
    assert "vector index incomplete" in out


# --------------------------------------------------------------------------- #
# O6 — one partition, two surfaces
# --------------------------------------------------------------------------- #

def _pair(checks_or_result: Dict[str, Any]) -> Tuple[Any, Any]:
    return checks_or_result["retired_node_points"], checks_or_result["nodeless_points"]


def _both(config: MitosConfig, index: _Index, monkeypatch, capsys) -> Tuple[Any, Any]:
    """`status`' pair, then `reconcile`'s over the same collection (before its writes)."""
    _status_reads(monkeypatch, index)
    _rc, checks = _status(config, capsys, as_json=True)
    _rc, out, _err = _reconcile(config, index, capsys, as_json=True)
    return _pair(checks), _pair(json.loads(out))


def test_o6_state_1_a_faulted_graph_pair_is_unknown_on_both(
    tmp_path, monkeypatch, capsys
) -> None:
    config = _ws(tmp_path, monkeypatch)
    ids = _full_graveyard(config)

    def _boom(self):
        raise RuntimeError("graph unreadable")

    monkeypatch.setattr(GraphStore, "get_all_node_ids", _boom)
    status_pair, reconcile_pair = _both(
        config, _Index(_every_point(ids) | set(NODELESS)), monkeypatch, capsys)
    assert status_pair == reconcile_pair == (None, None)


@pytest.mark.parametrize("graph", ["populated", "empty"])
@pytest.mark.parametrize("collection", ["absent", "empty"])
def test_o6_state_2_an_empty_listing_is_zero_on_any_graph(
    tmp_path, monkeypatch, capsys, graph, collection
) -> None:
    config = _ws(tmp_path, monkeypatch)
    if graph == "populated":
        _full_graveyard(config)
    index = _Index(absent=(collection == "absent"))
    # reconcile's drain would create and fill the collection; a pure listing row.
    manager = _manager(config, index)
    manager.drain_pending_embeddings = lambda **kw: mitos.sync.DrainOutcome(  # type: ignore
        written=0, skipped_present=0, pruned=0, failed=0, stop=None, marker=None)
    _status_reads(monkeypatch, index)
    _rc, checks = _status(config, capsys, as_json=True)
    _rc, out, _err = _reconcile(config, index, capsys, as_json=True, manager=manager)
    assert _pair(checks) == _pair(json.loads(out)) == (0, 0)


def test_o6_state_3_points_over_a_zero_node_graph_are_unknown_on_both(
    tmp_path, monkeypatch, capsys
) -> None:
    config = _ws(tmp_path, monkeypatch)
    status_pair, reconcile_pair = _both(config, _Index(set(NODELESS)), monkeypatch, capsys)
    assert status_pair == reconcile_pair == (None, None)


def test_o6_state_4_both_surfaces_call_the_one_partition_and_agree(
    tmp_path, monkeypatch, capsys
) -> None:
    config = _ws(tmp_path, monkeypatch)
    ids = _full_graveyard(config)
    calls: List[str] = []

    def _spy(module: Any, name: str) -> None:
        def _wrapped(*args, **kwargs):
            calls.append(name)
            return partition_points(*args, **kwargs)
        monkeypatch.setattr(module, "partition_points", _wrapped)

    _spy(cli, "status")
    _spy(mitos.sync, "reconcile")
    status_pair, reconcile_pair = _both(
        config, _Index(_every_point(ids) | set(NODELESS)), monkeypatch, capsys)
    assert status_pair == reconcile_pair == (len(_RETIRED), 2)
    assert calls == ["status", "reconcile"]


def test_o6_nodeless_agrees_with_the_walks_orphan_test(tmp_path, monkeypatch) -> None:
    """For any listing, `nodeless` equals `resolve_point_ids(listing).orphans` —
    malformed and decimal ids included, and a killed open question kept by both.
    Listings use Qdrant's canonical id form: `resolve_point_ids` also normalizes an
    uppercase or dashless id, which Qdrant never returns, and the delete takes the
    listed string verbatim."""
    config = _ws(tmp_path, monkeypatch)
    ids = _full_graveyard(config)
    store = GraphStore(config.db_path)
    listing = _every_point(ids) | set(NODELESS) | {"not-a-uuid", "7", "٧"}

    split = partition_points(listing, node_ids=store.get_all_node_ids(),
                             active_ids=store.get_active_node_ids())

    assert set(split.nodeless) == set(store.resolve_point_ids(listing).orphans)
    assert set(split.nodeless) == set(NODELESS) | {"not-a-uuid", "7", "٧"}
    assert hash_to_uuid(ids["q-old"]) in split.retired_node
    assert hash_to_uuid(ids["q-typo"]) in split.retired_node


# --------------------------------------------------------------------------- #
# O7 — the heal clears its own warning
# --------------------------------------------------------------------------- #

def test_o7_status_then_reconcile_then_status_clears_the_garbage_rung(
    tmp_path, monkeypatch, capsys
) -> None:
    config = _ws(tmp_path, monkeypatch)
    ids = _full_graveyard(config)
    index = _Index(_every_point(ids) | set(NODELESS))
    _status_reads(monkeypatch, index)

    _rc, before = _status(config, capsys, as_json=False)
    assert "⚠ 2 point(s) belong to no node in the graph" in before
    assert "to delete them" in before

    rc, _out, _err = _reconcile(config, index, capsys, as_json=False)
    assert rc == 0

    _rc, after = _status(config, capsys, as_json=False)
    assert "belong to no node" not in after
    assert f"• {len(_RETIRED)} point(s) belong to retired nodes — history, kept" in after


# --------------------------------------------------------------------------- #
# O8 — status over a zero-node graph with points names no collector
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("corpus", ["empty", "holds entries"])
def test_o8_status_over_a_zero_node_graph_judges_no_point(
    tmp_path, monkeypatch, capsys, corpus
) -> None:
    config = _ws(tmp_path, monkeypatch)
    if corpus == "holds entries":
        _append_entry(config.workspace_dir)
    _status_reads(monkeypatch, _Index(set(NODELESS)))

    _rc, checks = _status(config, capsys, as_json=True)
    assert checks["orphan_points"] == 2  # every point
    assert (checks["retired_node_points"], checks["nodeless_points"]) == (None, None)

    _rc, out = _status(config, capsys, as_json=False)
    line = next(l for l in out.splitlines() if "not judged" in l)
    assert line.strip() == "• 2 point(s) not judged — the graph holds no nodes."
    assert "mitos reconcile" not in out
    assert "left by a rebuild" not in out
    assert "belong to no node" not in out
    assert ("mitos rebuild" in out) is (corpus == "holds entries")


# --------------------------------------------------------------------------- #
# O9 — P1: the shipped keys keep their values; one key set on every path
# --------------------------------------------------------------------------- #

_SHIPPED_AND_3B = ("active", "present", "enqueued", "expected", "expected_present",
                   "written", "left_queued", "drain_stop")
_ADDED_3C = ("retired_node_points", "nodeless_points", "deleted", "delete_failed",
             "graph_unbuilt")


def test_o9_the_shipped_reconcile_and_status_keys_keep_their_values(
    tmp_path, monkeypatch, capsys
) -> None:
    config = _ws(tmp_path, monkeypatch)
    ids = _full_graveyard(config)
    present = _every_point(ids) | set(NODELESS)
    index = _Index(present)
    _status_reads(monkeypatch, index)

    _rc, checks = _status(config, capsys, as_json=True)
    active = {hash_to_uuid(ids[s]) for s in _ACTIVE}
    assert checks["active_nodes"] == len(_ACTIVE)
    assert checks["missing_active_vectors"] == 0
    assert checks["missing_retired_vectors"] == 0
    assert checks["orphan_points"] == len(present - active)
    assert checks["collection_points"] == len(present)

    _rc, out, _err = _reconcile(config, index, capsys, as_json=True)
    result = json.loads(out)
    expected = len(GraphStore(config.db_path).get_decision_index_expected_ids())
    assert {k: result[k] for k in _SHIPPED_AND_3B} == {
        "active": len(_ACTIVE), "present": len(present), "enqueued": 0,
        "expected": expected, "expected_present": expected, "written": 0,
        "left_queued": 0, "drain_stop": None}


def test_o9_every_reconcile_return_path_carries_the_same_key_set(
    tmp_path, monkeypatch, capsys
) -> None:
    keysets = []
    built = _ws(tmp_path, monkeypatch, "built")
    ids = _full_graveyard(built)
    for index in (_Index(_every_point(ids) | set(NODELESS)), _Index(absent=True)):
        keysets.append(set(_manager(built, index).reconcile_embeddings()))
    down = MitosSyncManager(built)
    down.embed_provider = None
    keysets.append(set(down.reconcile_embeddings()))
    clone = _clone(tmp_path, monkeypatch)
    keysets.append(set(_manager(clone, _Index(set(NODELESS))).reconcile_embeddings()))

    assert all(keys == keysets[0] for keys in keysets)
    assert set(_SHIPPED_AND_3B) | set(_ADDED_3C) == keysets[0]


def test_o9_the_provider_down_return_judges_nothing(tmp_path, monkeypatch) -> None:
    config = _ws(tmp_path, monkeypatch)
    _full_graveyard(config)
    manager = MitosSyncManager(config)
    manager.embed_provider = None
    result = manager.reconcile_embeddings()
    assert {k: result[k] for k in _ADDED_3C} == {
        "retired_node_points": None, "nodeless_points": None, "deleted": 0,
        "delete_failed": 0, "graph_unbuilt": False}


# --------------------------------------------------------------------------- #
# O10 — the ids-only node read
# --------------------------------------------------------------------------- #

def test_o10_get_all_node_ids_agrees_with_get_all_nodes(store) -> None:
    assert store.get_all_node_ids() == frozenset()
    _matrix(store)
    ids = store.get_all_node_ids()
    assert isinstance(ids, frozenset)
    assert ids == {node["id"] for node in store.get_all_nodes()}
    assert len(ids) > len(store.get_active_node_ids())


@pytest.fixture
def store(tmp_path) -> GraphStore:
    db_path = str(tmp_path / ".mitos" / "graph.sqlite")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return GraphStore(db_path)
