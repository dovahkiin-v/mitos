"""Phase 2f: the retired partition, its handles, their live ends, the lead and the blackout.

Every ranked envelope lists the retired decisions its window passed as handles —
those that outranked the last live decision returned, or every retired decision in
the fetch when the window came back short — bounded at ``limit``, closest first,
with the rest counted. Each handle carries its node id, its score, its successor
stamps and, when an immediate successor is itself retired, the **live end** of its
kill chain. A lead line opens the note when a retired decision outranks rank 1; the
blackout is that lead's zero-live form. A short window says why it is short.
(Vision §4.4; plan p2f §10.)

Rows enter the outer frame where they can — MCP ``surface_decisions`` /
``query_decisions`` called as tools and ``cmd_surface`` / ``cmd_query`` called as
verbs on both encodings (the 2c drivers) — over a real ``GraphStore``. The store's
anomaly fixtures (fan-out, cycle) are planted with raw SQL into a throwaway graph,
because the supported write path refuses to author either.
"""

import inspect
import io
import json
import logging
import sqlite3
import threading
import time
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from mitos import cli, display, mcp_server, recall, retrieval_walk
from mitos.cli import cmd_init, cmd_show, cmd_surface
from mitos.config import MitosConfig
from mitos.errors import DatabaseError
from mitos.parser import ParsedEntry
from mitos.recall import (RetiredPartition, ShortWindow, _SURFACE_POINTERS,
                          assess_query_recall, assess_surface_recall, blackout_note,
                          edge_clause, retired_lead_clause, retired_partition,
                          short_window, short_window_clause, stamps_unread_clause,
                          window_edge, withheld_clause)
from mitos.retrieval_walk import (EDGE_FOUND, EDGE_UNMEASURED, EXHAUSTED,
                                  RANKED_OVERFETCH_BOUND, SHORT_BOUND, PointResolution,
                                  fill, walk_ranked)
from mitos.store import GraphStore, _KILL_EDGE_TYPES_SQL
from mitos.sync import MitosSyncManager

from _point_hits import CannedVector, orphan_hit, point_hit
from test_mcp_selector import FORBIDDEN_SYNTAX
from test_point_identity import _record, same_slug_lineage
from test_ranked_walk_surfaces import (DRIVERS, JSON_DRIVERS, TEXT_DRIVERS, _Forwarding,
                                       _StampFault, _cli, _mcp, _text_ranks)
from test_surface_confidence import (_FakeEmbed, _MissingCollection, _commit_oq, _rec,
                                     offline, ws)  # noqa: F401

_CLAIM = "a claim that is not any slug"
_LEAD_STEM = "The closest match was retired:"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _id(m: MitosSyncManager, slug: str) -> str:
    ids = m.store.resolve_slug(slug)
    assert len(ids) == 1, ids
    return ids[0]


def _retire(m: MitosSyncManager, slug: str) -> str:
    """Records ``slug`` and a live ``{slug}-next`` superseding it; returns ``slug``'s id."""
    _rec(m, slug)
    _rec(m, f"{slug}-next", supersedes=slug)
    return _id(m, slug)


def _chain(m: MitosSyncManager, slug: str) -> str:
    """A two-deep kill chain: ``slug`` ← ``{slug}-mid`` ← ``{slug}-end`` (live)."""
    _rec(m, slug)
    _rec(m, f"{slug}-mid", supersedes=slug)
    _rec(m, f"{slug}-end", supersedes=f"{slug}-mid")
    return _id(m, slug)


def _live(m: MitosSyncManager, *slugs: str) -> List[str]:
    for slug in slugs:
        _rec(m, slug)
    return [_id(m, s) for s in slugs]


def _note_line(out: str) -> str:
    """The text note: the last ``→`` line, or — on ``cmd_query``'s blackout exit,
    which prints the note bare since 2d — the last non-empty line."""
    lines = [ln for ln in out.splitlines() if ln.startswith("→ ")]
    if lines:
        return lines[-1][2:]
    return [ln for ln in out.splitlines() if ln.strip()][-1]


def _note(resp: Any, key: Optional[str]) -> str:
    return resp["note"] if key else _note_line(resp)


def _block(out: str) -> str:
    """The CLI text partition block: from its header to the note line."""
    assert "Retired decisions this ranking passed" in out, out
    return out.split("Retired decisions this ranking passed", 1)[1].split("→ ", 1)[0]


# --------------------------------------------------------------------------- #
# Walk: membership (row 1)
# --------------------------------------------------------------------------- #

def _node(nid: str, state: str = "active", kind: str = "decision") -> Dict[str, Any]:
    return {"id": nid, "slug": nid, "kind": kind, "computed_state": state}


def _walk_of(spec: str, *, limit: int, bound: int = 100):
    """``spec`` is one letter per hit: L live, R retired, O orphan. Pure ``fill``."""
    nodes, hits = {}, []
    for i, c in enumerate(spec):
        pid = f"p{i}"
        hits.append({"point_id": pid, "score": 1 - i / 100})
        if c != "O":
            nodes[pid] = _node(f"{c}{i}", "active" if c == "L" else "superseded")
    resolution = PointResolution(nodes=nodes, orphans=[], ambiguous={})
    return fill(hits, resolution, limit=limit, bound=bound)


def _ids_of(hits) -> List[str]:
    return [h.node["id"] for h in hits]


def test_a_full_window_passes_only_the_retired_hits_above_its_last_live():
    """Row 1: a retired hit between the last live and the edge is not a member."""
    walk = _walk_of("RLRLRL", limit=2)
    assert walk.end == EDGE_FOUND
    assert _ids_of(walk.window) == ["L1", "L3"]
    assert _ids_of(walk.retired) == ["R0", "R2", "R4"]
    assert _ids_of(walk.passed) == ["R0", "R2"]


def test_a_full_unmeasured_window_passes_only_what_it_outranked():
    """G7: an unmeasured edge lists no retired point below the last live one."""
    walk = _walk_of("RLLRR", limit=2, bound=5)
    assert walk.end == EDGE_UNMEASURED
    assert _ids_of(walk.passed) == ["R0"]


@pytest.mark.parametrize("spec,end", [("RRLRR", EXHAUSTED), ("RRRR", EXHAUSTED)])
def test_a_short_window_passes_every_retired_hit(spec, end):
    """Row 1: short (one live) and zero live — every retired hit is a member."""
    walk = _walk_of(spec, limit=2)
    assert walk.end == end
    assert walk.passed == walk.retired and len(walk.passed) == spec.count("R")


def test_a_short_bound_window_passes_every_retired_hit():
    walk = _walk_of("RLRRO", limit=3, bound=5)
    assert walk.end == SHORT_BOUND
    assert _ids_of(walk.passed) == ["R0", "R2", "R3"]


def test_passed_is_a_prefix_of_retired_so_its_shown_part_is_already_stamped():
    """2c's lesson kept: ``passed[:limit]`` ⊆ ``retired[:limit]`` ⊆ the stamp ids."""
    for spec in ("RLRLRL", "RRLRR", "LRRRL", "RRRRRL", "LLR"):
        for limit in (1, 2, 3):
            walk = _walk_of(spec, limit=limit)
            assert walk.passed == walk.retired[:len(walk.passed)], (spec, limit)
            stamped = set(retrieval_walk.stamp_ids_of(walk))
            assert set(_ids_of(walk.passed[:limit])) <= stamped, (spec, limit)


def test_fill_accepts_limit_at_or_above_bound_for_the_harness():
    """The horizon harness drives ``limit >= bound``; membership stays well defined."""
    walk = _walk_of("RLR", limit=5, bound=3)
    assert _ids_of(walk.passed) == ["R0", "R2"]


# --------------------------------------------------------------------------- #
# Walk: the fourth read and its fault domain (rows 2, 3)
# --------------------------------------------------------------------------- #

def _commit(store: GraphStore, slug: str, *, supersedes: Optional[List[str]] = None,
            corrects: Optional[List[str]] = None) -> str:
    entry = ParsedEntry("decision", slug, 1, 10)
    entry.axiom = f"Axiom for {slug}."
    entry.rejected_paths = "Rejected the obvious alternative."
    entry.scope = ["core"]
    if supersedes:
        entry.supersedes = supersedes
    if corrects:
        entry.corrects = corrects
    return store.commit_parsed_entry(entry).node_id


@pytest.fixture
def store(tmp_path) -> GraphStore:
    return GraphStore(str(tmp_path / "graph.sqlite"))


class _LiveEndSpy:
    """Forwards to a real store; ``get_live_ends`` is scripted per test."""

    def __init__(self, inner: GraphStore, live_ends: Any = None, raises: Any = None):
        self._inner = inner
        self._live_ends = live_ends
        self._raises = raises
        self.ended: List[List[str]] = []

    def resolve_point_ids(self, ids):
        return self._inner.resolve_point_ids(ids)

    def get_modifiers_map(self, ids):
        return self._inner.get_modifiers_map(ids)

    def get_live_ends(self, ids):
        self.ended.append(list(ids))
        if self._raises is not None:
            raise self._raises
        if self._live_ends is not None:
            return self._live_ends
        return self._inner.get_live_ends(ids)


def _chain_store(store: GraphStore):
    """old ← mid ← new, plus two live decisions; ``old`` outranks both."""
    old = _commit(store, "old")
    _commit(store, "mid", supersedes=["old"])
    _commit(store, "new", supersedes=["mid"])
    a, b = _commit(store, "a"), _commit(store, "b")
    return old, a, b


def test_the_fourth_read_names_the_live_end_of_the_shown_members(store):
    old, a, b = _chain_store(store)
    graph = _LiveEndSpy(store)
    walk = walk_ranked([0.1], vector_store=CannedVector(
        [point_hit(old, .9), point_hit(a, .8), point_hit(b, .7)], store=store),
        graph=graph, limit=2)
    assert graph.ended == [[old]]
    assert walk.live_ends == {old: ["new"]}
    assert walk.stamps[old] == {"superseded_by": ["mid"]}


def test_a_non_dict_live_end_map_fails_loud(store):
    """F14c's row: the walk guards the type, as it does the stamp map's."""
    old, a, b = _chain_store(store)
    with pytest.raises(retrieval_walk.WalkContractError, match="get_live_ends"):
        walk_ranked([0.1], vector_store=CannedVector(
            [point_hit(old, .9), point_hit(a, .8)], store=store),
            graph=_LiveEndSpy(store, live_ends=[]), limit=1)


def test_a_live_end_fault_alone_costs_the_stamps_and_the_live_ends(store):
    """Row 3 (D2, scout D-K): the stamp read succeeds and is discarded; the window,
    edge and partition stand."""
    old, a, b = _chain_store(store)
    hits = [point_hit(old, .9), point_hit(a, .8), point_hit(b, .7)]
    healthy = walk_ranked([0.1], vector_store=CannedVector(hits, store=store),
                          graph=store, limit=1)
    graph = _LiveEndSpy(store, raises=DatabaseError("planted live-end fault"))
    walk = walk_ranked([0.1], vector_store=CannedVector(hits, store=store),
                       graph=graph, limit=1)
    assert graph.ended == [[old]]
    assert walk.stamps == {} and walk.live_ends == {} and walk.stamps_unread is True
    assert _ids_of(walk.window) == _ids_of(healthy.window) == [a]
    assert walk.edge.node["id"] == healthy.edge.node["id"] == b
    assert walk.passed == healthy.passed and walk.end == healthy.end


def test_the_live_end_catch_is_narrow(store):
    """The live-end twin of K7's row: a defect is not a fault, it propagates."""
    old, a, b = _chain_store(store)
    with pytest.raises(RuntimeError, match="defect"):
        walk_ranked([0.1], vector_store=CannedVector(
            [point_hit(old, .9), point_hit(a, .8)], store=store),
            graph=_LiveEndSpy(store, raises=RuntimeError("defect")), limit=1)


def test_a_real_sqlite_live_end_fault_is_typed_and_contained(store, monkeypatch):
    """The store wraps the CTE's ``sqlite3.Error``; the walk contains it."""
    old, a, b = _chain_store(store)
    original = store._get_connection

    class _Faulting:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, params=()):
            if "WITH RECURSIVE" in sql:
                raise sqlite3.OperationalError("database is locked")
            return self._conn.execute(sql, params)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    monkeypatch.setattr(store, "_get_connection", lambda: _Faulting(original()))
    with pytest.raises(DatabaseError, match="live ends"):
        store.get_live_ends([old])
    walk = walk_ranked([0.1], vector_store=CannedVector(
        [point_hit(old, .9), point_hit(a, .8)], store=store), graph=store, limit=1)
    assert _ids_of(walk.window) == [a] and walk.stamps_unread is True
    assert walk.stamps == {} and walk.live_ends == {}


# --------------------------------------------------------------------------- #
# Store: get_live_ends (rows 4, 5)
# --------------------------------------------------------------------------- #

def _plant_edge(store: GraphStore, source: str, target: str,
                edge_type: str = "supersedes") -> None:
    """An out-of-band kill edge (G2): the write path refuses fan-out and cycles."""
    conn = store._get_connection()
    try:
        conn.execute(
            "INSERT INTO edges (source_id, source_kind, target_id, target_kind, "
            "edge_type, created_at) VALUES (?, 'decision', ?, 'decision', ?, ?)",
            (source, target, edge_type, "2026-10-16T00:00:00+00:00"))
        conn.commit()
    finally:
        conn.close()


def test_a_seed_whose_killer_is_live_is_absent(store):
    old = _commit(store, "old")
    _commit(store, "new", supersedes=["old"])
    live = _commit(store, "live")
    assert store.get_live_ends([old, live]) == {}


def test_a_two_deep_chain_names_its_live_end(store):
    """G12: the liveness test on the CTE's alias — the unaliased predicate would call
    `mid` live and answer nothing for this seed."""
    old = _commit(store, "old")
    _commit(store, "mid", supersedes=["old"])
    _commit(store, "new", supersedes=["mid"])
    assert store.get_live_ends([old]) == {old: ["new"]}


@pytest.mark.parametrize("first,second", [("corrects", "supersedes"),
                                          ("supersedes", "corrects")])
def test_a_corrects_hop_is_a_kill_hop(store, first, second):
    old = _commit(store, "old")
    _commit(store, "mid", **{first: ["old"]})
    _commit(store, "new", **{second: ["mid"]})
    assert store.get_live_ends([old]) == {old: ["new"]}


def test_a_drifted_decision_is_a_live_end(store):
    """Drift annotates; it does not kill (F6's row)."""
    old = _commit(store, "old")
    _commit(store, "mid", supersedes=["old"])
    new = _commit(store, "new", supersedes=["mid"])
    store.write_signal(new, "drifted")
    assert store.get_node_state(new) == "drifted"
    assert store.get_live_ends([old]) == {old: ["new"]}


def test_a_fan_out_names_both_ends_in_order_and_warns_once(store, caplog):
    seed = _commit(store, "seed")
    mid = _commit(store, "mid", supersedes=["seed"])
    _commit(store, "b-end", supersedes=["mid"])
    a_end = _commit(store, "a-end")
    _plant_edge(store, a_end, mid)
    with caplog.at_level(logging.WARNING, logger="mitos.store"):
        assert store.get_live_ends([seed]) == {seed: ["a-end", "b-end"]}
    warnings = [r for r in caplog.records if r.name == "mitos.store"]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "fan-out" in message and seed in message and "mitos rebuild" in message


def test_live_ends_sort_by_casefold_then_slug():
    """The result rule's order, over the slugs the walk reached."""
    ends = ["b-end", "A-end", "a-end"]
    assert sorted(ends, key=lambda s: (s.casefold(), s)) == ["A-end", "a-end", "b-end"]
    assert "slug.casefold(), slug" in inspect.getsource(GraphStore.get_live_ends)


def test_the_store_answers_its_live_ends_in_casefold_then_slug_order(store):
    """The order as the store returns it, not as ``sorted`` would: three live ends
    whose plain slug order and casefold order disagree (the chain analysis sorts by
    id first, so only the result rule's own sort can put them right). Slugs collide
    case-insensitively at commit, so the casefold tie the ``slug`` tie-break breaks
    cannot be built through the store; the row above pins that half."""
    seed = _commit(store, "seed")
    mid = _commit(store, "mid", supersedes=["seed"])
    for slug in ("Zed-end", "alpha-end", "Mid-end"):
        _plant_edge(store, _commit(store, slug), mid)
    assert store.get_live_ends([seed]) == {seed: ["alpha-end", "Mid-end", "Zed-end"]}


def _bounded(fn, seconds: float = 10.0):
    """G3: a cycle row runs under a guard, so a non-terminating walk reds, never hangs."""
    box: Dict[str, Any] = {}

    def run():
        box["value"] = fn()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), "the kill-chain walk did not terminate on a cycle"
    return box["value"]


def test_a_closed_cycle_answers_empty_and_warns(store, caplog):
    seed = _commit(store, "seed")
    m = _commit(store, "m", supersedes=["seed"])
    n = _commit(store, "n")
    _plant_edge(store, n, m)
    _plant_edge(store, m, n)
    with caplog.at_level(logging.WARNING, logger="mitos.store"):
        assert _bounded(lambda: store.get_live_ends([seed])) == {seed: []}
    messages = [r.getMessage() for r in caplog.records if r.name == "mitos.store"]
    assert len(messages) == 1 and "cycle" in messages[0] and seed in messages[0]


def test_a_cycle_with_an_exit_names_the_exit(store, caplog):
    seed = _commit(store, "seed")
    m = _commit(store, "m", supersedes=["seed"])
    n = _commit(store, "n")
    exit_ = _commit(store, "exit")
    _plant_edge(store, n, m)
    _plant_edge(store, m, n)
    _plant_edge(store, exit_, n)
    with caplog.at_level(logging.WARNING, logger="mitos.store"):
        assert _bounded(lambda: store.get_live_ends([seed])) == {seed: ["exit"]}
    messages = [r.getMessage() for r in caplog.records if r.name == "mitos.store"]
    assert len(messages) == 1 and "cycle" in messages[0]


def test_several_seeds_are_one_select_on_one_connection(store, monkeypatch):
    """F4's row: one statement for the whole list, counted on the method's own
    connection after its opener's PRAGMAs."""
    seeds = []
    for i in range(3):
        seeds.append(_commit(store, f"old-{i}"))
        _commit(store, f"mid-{i}", supersedes=[f"old-{i}"])
        _commit(store, f"new-{i}", supersedes=[f"mid-{i}"])
    recorder = _record(store, monkeypatch)
    assert store.get_live_ends(seeds) == {s: [f"new-{i}"] for i, s in enumerate(seeds)}
    assert recorder.connections == 1
    assert len(recorder.statements) == 1
    assert recorder.statements[0][0].lstrip().startswith("WITH RECURSIVE")


def test_the_live_end_read_drives_from_the_chain_never_scans_edges(store, monkeypatch):
    """P17: the outer join is pinned to drive from the chain (``CROSS JOIN``). Left to
    the planner it scanned every edge — 24.5 ms against 0.4 ms on the burial horizon."""
    seed = _commit(store, "old")
    _commit(store, "mid", supersedes=["old"])
    _commit(store, "new", supersedes=["mid"])
    recorder = _record(store, monkeypatch)
    store.get_live_ends([seed])
    sql, params = recorder.statements[0]
    conn = sqlite3.connect(store.db_path)
    try:
        plan = [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + sql, params)]
    finally:
        conn.close()
    assert not [step for step in plan if step.startswith("SCAN e")], plan
    assert "CROSS JOIN edges e" in sql


def test_an_empty_input_reads_nothing(store, monkeypatch):
    recorder = _record(store, monkeypatch)
    assert store.get_live_ends([]) == {}
    assert recorder.connections == 0 and recorder.statements == []


def test_the_kill_types_come_from_the_one_liveness_atom():
    """Row 5 (PATTERNS "Active-view liveness is ONE SQL atom"): no second hand-rolled
    tuple, and never the unaliased active-view predicate inside the aliased CTE."""
    import ast
    import textwrap
    fn = ast.parse(textwrap.dedent(inspect.getsource(GraphStore._walk_kill_chains))).body[0]
    fn.body = fn.body[1:]                                       # the code, not its docstring
    source = ast.unparse(fn)
    assert "_KILL_EDGE_TYPES_SQL" in source
    assert "'supersedes'" not in source and "'corrects'" not in source
    assert "_ACTIVE_VIEW_PREDICATE" not in source
    assert "UNION ALL" not in source and "UNION" in source
    assert _KILL_EDGE_TYPES_SQL == "('supersedes', 'corrects')"


def test_the_live_end_read_is_on_the_graph_protocol():
    from mitos.protocols import GraphStoreProtocol
    assert callable(getattr(GraphStoreProtocol, "get_live_ends", None))


# --------------------------------------------------------------------------- #
# The constructor (row 6)
# --------------------------------------------------------------------------- #

_RETIRED = {"id": "n1", "slug": "old", "computed_state": "superseded"}


def test_a_handle_carries_its_identity_state_and_score():
    h = display.retired_handle(_RETIRED, score=0.81, stamps={}, live_ends={})
    assert h == {"slug": "old", "id": "n1", "state": "superseded", "score": 0.81}


def test_a_handle_carries_each_kill_stamp_when_stamped_and_no_other():
    stamps = {"n1": {"superseded_by": ["mid"], "corrected_by": ["fix"],
                     "amended_by": ["x"]}}
    h = display.retired_handle(_RETIRED, score=0.8, stamps=stamps, live_ends={})
    assert h["superseded_by"] == ["mid"] and h["corrected_by"] == ["fix"]
    assert "amended_by" not in h and "live_end" not in h


def test_live_end_is_present_exactly_when_the_map_carries_the_id():
    stamps = {"n1": {"superseded_by": ["mid"]}}
    h = display.retired_handle(_RETIRED, score=0.8, stamps=stamps,
                               live_ends={"n1": ["new"]})
    assert h["live_end"] == ["new"]
    closed = display.retired_handle(_RETIRED, score=0.8, stamps=stamps,
                                    live_ends={"n1": []})
    assert closed["live_end"] == []                # membership, not truthiness
    other = display.retired_handle(_RETIRED, score=0.8, stamps=stamps,
                                   live_ends={"n2": ["x"]})
    assert "live_end" not in other


def test_on_a_fault_a_handle_carries_no_successor_key():
    h = display.retired_handle(_RETIRED, score=0.8, stamps={}, live_ends={})
    assert set(h) == {"slug", "id", "state", "score"}


def test_the_constructor_takes_its_maps_by_keyword_only():
    params = inspect.signature(display.retired_handle).parameters
    assert [p for p, v in params.items() if v.kind is v.KEYWORD_ONLY] == [
        "score", "stamps", "live_ends"]


def test_display_blackout_note_is_gone():
    """D4 / F16: the note moved to recall; a stray alias in the leaf reds here."""
    assert not hasattr(display, "blackout_note")
    assert "blackout_note" not in display.__dict__


# --------------------------------------------------------------------------- #
# The partition rule and the short window (row 7)
# --------------------------------------------------------------------------- #

def _rank(rank: int) -> SimpleNamespace:
    return SimpleNamespace(rank=rank, score=1 - rank / 100, node={"id": f"r{rank}"})


def _fake_walk(*, window=(), passed=(), limit=5, returned=10, end=EXHAUSTED,
               retired=None, edge=None) -> SimpleNamespace:
    window = [_rank(r) for r in window]
    passed = [_rank(r) for r in passed]
    return SimpleNamespace(window=window, passed=passed, limit=limit, returned=returned,
                           end=end, retired=passed if retired is None else retired,
                           edge=edge)


def test_the_partition_is_bounded_at_limit_and_counts_the_rest():
    part = retired_partition(_fake_walk(window=[8], passed=[1, 2, 3, 4], limit=2))
    assert [h.rank for h in part.shown] == [1, 2] and part.omitted == 2
    whole = retired_partition(_fake_walk(window=[8], passed=[1, 2], limit=2))
    assert len(whole.shown) == 2 and whole.omitted == 0


def test_the_partition_leads_only_when_its_first_member_outranks_rank_one():
    assert retired_partition(_fake_walk(window=[2, 4], passed=[1, 3], limit=2)).leads
    assert not retired_partition(_fake_walk(window=[1, 4], passed=[2, 3], limit=2)).leads
    assert not retired_partition(_fake_walk(window=[], passed=[1, 2])).leads   # blackout
    assert not retired_partition(_fake_walk(window=[1, 2], passed=[])).leads


def test_an_empty_partition_shows_nothing():
    assert retired_partition(_fake_walk(window=[1])) == RetiredPartition([], 0, False)


@pytest.mark.parametrize("end,kind", [(EXHAUSTED, "ran_out"), (SHORT_BOUND, "bound_retired")])
def test_a_short_window_says_why(end, kind):
    walk = _fake_walk(window=[3], passed=[1, 2], limit=5, returned=7, end=end)
    assert short_window(walk) == ShortWindow(kind, 1, 5, 7, 2)


def test_a_full_window_or_an_empty_fetch_says_nothing_about_shortness():
    assert short_window(_fake_walk(window=[1, 2], limit=2, end=EDGE_FOUND)) is None
    assert short_window(_fake_walk(window=[1, 2], limit=2, end=EXHAUSTED)) is None
    assert short_window(_fake_walk(window=[], returned=0, end=EXHAUSTED)) is None


def test_a_short_window_states_its_reason_at_the_ceiling_too():
    walk = _fake_walk(window=[1], limit=display.RANKED_LIMIT_CEILING,
                      returned=RANKED_OVERFETCH_BOUND, end=SHORT_BOUND)
    assert short_window(walk).kind == "bound_retired"


def test_the_short_window_end_values_agree_with_the_walk():
    assert recall._WALK_END_EXHAUSTED == retrieval_walk.EXHAUSTED
    assert recall._WALK_END_SHORT_BOUND == retrieval_walk.SHORT_BOUND


# --------------------------------------------------------------------------- #
# Note pieces: order and register (row 8)
# --------------------------------------------------------------------------- #

_CONFIG = SimpleNamespace(project="my proj")
_LEAD = {"slug": "old", "id": "n1", "state": "superseded", "score": 0.9,
         "superseded_by": ["mid"], "live_end": ["new"]}
_SHORT = ShortWindow("ran_out", 1, 5, 3, 1)
_EDGE = recall.WindowEdge(recall.EDGE_KIND_MID_CLUSTER, 5, 5)
_FORKS = [("surface", dict(top_score=0.9, result_count=2)),
          ("surface", dict(top_score=0.65, result_count=2)),
          ("surface", dict(top_score=0.3, result_count=2)),
          ("query", dict(top_score=0.9, result_count=2)),
          ("query", dict(top_score=0.65, result_count=2)),
          ("query", dict(top_score=0.3, result_count=2))]


def _compose(verb, surface, kw, *, lead, edge, short):
    if verb == "surface":
        return assess_surface_recall(semantic_ran=True, degraded_reason=None, scope=None,
                                     surface=surface, lead=lead, edge=edge, short=short,
                                     **kw)[1]
    return assess_query_recall(config=_CONFIG, surface=surface, lead=lead, edge=edge,
                               short=short, **kw)[1]


@pytest.mark.parametrize("surface", ["cli", "mcp"])
@pytest.mark.parametrize("verb,kw", _FORKS)
def test_every_ranked_fork_reads_lead_band_then_edge_or_short(verb, kw, surface):
    band = _compose(verb, surface, kw, lead=None, edge=None, short=None)
    lead = retired_lead_clause(_LEAD, surface=surface)
    assert _compose(verb, surface, kw, lead=_LEAD, edge=None, short=_SHORT) == (
        f"{lead} {band} {short_window_clause(_SHORT, surface=surface)}")
    assert _compose(verb, surface, kw, lead=_LEAD, edge=_EDGE, short=None) == (
        f"{lead} {band} {edge_clause(_EDGE, surface=surface)}")


def test_the_lead_opens_before_the_unused_scope_prefix():
    counts = {"auth": {"active_decisions": 2, "parked_open_questions": 0}}
    note = assess_surface_recall(semantic_ran=True, degraded_reason=None, top_score=0.9,
                                 result_count=2, scope="nope", scope_counts=counts,
                                 surface="mcp", lead=_LEAD, edge=None, short=None)[1]
    lead = retired_lead_clause(_LEAD, surface="mcp")
    assert note.startswith(f"{lead} ") and "nope" in note[len(lead):]


@pytest.mark.parametrize("verb", ["surface", "query"])
def test_the_nothing_ranked_forks_ignore_lead_and_short(verb):
    kw = dict(top_score=None, result_count=0)
    plain = _compose(verb, "mcp", kw, lead=None, edge=None, short=None)
    assert _compose(verb, "mcp", kw, lead=_LEAD, edge=None, short=_SHORT) == plain


def test_the_degraded_arm_ignores_lead_and_short():
    # On the dump arm (5d: the only degraded arm with results takes `scope_total`).
    kw = dict(semantic_ran=False, degraded_reason="the provider is down", top_score=None,
              result_count=2, scope=None, scope_total=2, surface="mcp", edge=None)
    plain = assess_surface_recall(lead=None, short=None, **kw)[1]
    assert assess_surface_recall(lead=_LEAD, short=_SHORT, **kw)[1] == plain


def test_lead_and_short_are_required_keywords_on_both_composers():
    with pytest.raises(TypeError, match="lead"):
        assess_surface_recall(semantic_ran=True, degraded_reason=None, top_score=0.7,
                              result_count=1, scope=None, surface="mcp", edge=None,
                              short=None)
    with pytest.raises(TypeError, match="short"):
        assess_query_recall(top_score=0.7, result_count=1, config=_CONFIG, surface="mcp",
                            edge=None, lead=None)


@pytest.mark.parametrize("spec", ["LLLLLLL", "LRLRL", "RRLRRL", "LLRRR", "RRR", "L", "LLL"])
@pytest.mark.parametrize("limit", [1, 2, 3, 5])
@pytest.mark.parametrize("bound", [4, 100])
def test_the_edge_and_the_short_clause_never_co_occur(spec, limit, bound):
    if limit >= bound:
        return
    walk = _walk_of(spec[:bound], limit=limit, bound=bound)
    edge = window_edge(walk, margin=1.0, ceiling=50, full_top=None)
    assert not (edge is not None and short_window(walk) is not None), (spec, limit)


def test_the_lead_names_the_live_end_then_the_stamps_then_says_unread():
    assert retired_lead_clause(_LEAD, surface="mcp") == (
        "The closest match was retired: old (superseded); its current word lives in new.")
    stamped = {k: v for k, v in _LEAD.items() if k != "live_end"}
    assert retired_lead_clause(stamped, surface="mcp").endswith("lives in mid.")
    both = dict(stamped, corrected_by=["fix"])
    assert retired_lead_clause(both, surface="mcp").endswith("lives in mid, fix.")
    closed = dict(_LEAD, live_end=[])
    assert retired_lead_clause(closed, surface="mcp").endswith(
        "its kill chain reaches no live decision.")
    unread = {"slug": "old", "id": "n1", "state": "superseded", "score": 0.9}
    assert retired_lead_clause(unread, surface="mcp").endswith(
        "its successor could not be read.")


def test_the_lead_is_the_same_on_both_boundaries_and_names_no_pointer():
    assert retired_lead_clause(_LEAD, surface="cli") == retired_lead_clause(
        _LEAD, surface="mcp")
    with pytest.raises(KeyError):
        retired_lead_clause(_LEAD, surface="web")


def _clauses(surface: str) -> List[str]:
    handles = [_LEAD, dict(_LEAD, slug="other", live_end=[])]
    return [
        retired_lead_clause(_LEAD, surface=surface),
        short_window_clause(ShortWindow("ran_out", 0, 5, 1, 1), surface=surface),
        short_window_clause(ShortWindow("bound_retired", 2, 50, 350, 300), surface=surface),
        blackout_note(handles, short=_SHORT, surface=surface, config=_CONFIG),
    ]


@pytest.mark.parametrize("surface", ["cli", "mcp"])
def test_no_2f_clause_says_unmeasured_or_instructs_a_write(surface):
    for text in _clauses(surface):
        low = text.lower()
        assert "unmeasured" not in low
        for word in ("record", "decide", "write ", "sync"):
            assert word not in low, (word, text)


@pytest.mark.parametrize("surface", ["cli", "mcp"])
def test_the_short_clause_names_no_lever(surface):
    for short in (ShortWindow("ran_out", 1, 5, 3, 0),
                  ShortWindow("bound_retired", 1, 5, 350, 340)):
        text = short_window_clause(short, surface=surface).lower()
        for lever in ("higher", "raise", "increase", "widen", "re-send", "resend"):
            assert lever not in text, (lever, text)


def test_the_mcp_wording_names_no_shell_command():
    """G5's new row on the note's new home: the MCP blackout, lead and short clauses
    carry no ``mitos `` and no ``FORBIDDEN_SYNTAX`` member (F12's row)."""
    for text in _clauses("mcp"):
        assert "mitos " not in text, text
        for syntax in FORBIDDEN_SYNTAX:
            assert syntax not in text, (syntax, text)


def test_each_boundary_words_its_own_blackout_heal():
    cli_note = blackout_note([_LEAD], short=None, surface="cli", config=_CONFIG)
    mcp_note = blackout_note([_LEAD], short=None, surface="mcp", config=_CONFIG)
    assert "`mitos show -p 'my proj' -- <id>`" in cli_note
    assert "`mitos list -p 'my proj' --state all`" in cli_note
    assert "show_node(ident='<id>')" in mcp_note
    assert "list_decisions(state='all')" in mcp_note


def test_the_pointer_keys_exist_under_both_surfaces():
    for key in ("dereference_id", "history"):
        assert key in _SURFACE_POINTERS["cli"] and key in _SURFACE_POINTERS["mcp"]
        assert "mitos " in _SURFACE_POINTERS["cli"][key]
        assert "mitos " not in _SURFACE_POINTERS["mcp"][key]


def test_the_blackout_names_the_live_end_never_the_dead_hop():
    """F11's row at the composer: ``old``'s immediate successor ``mid`` is retired."""
    note = blackout_note([_LEAD], short=None, surface="mcp", config=_CONFIG)
    assert "old (superseded → new)" in note and "mid" not in note


# --------------------------------------------------------------------------- #
# Surfaces: the partition on a graze (rows 9, 10)
# --------------------------------------------------------------------------- #

def _graze_world(m) -> List[Dict[str, Any]]:
    """limit=2: old0 · a · old1 · b · old2 · c — members old0, old1; old2 sits
    between the last live and the edge."""
    old = [_retire(m, f"old{i}") for i in range(3)]
    a, b, c = _live(m, "a", "b", "c")
    return [point_hit(old[0], .95), point_hit(a, .9), point_hit(old[1], .85),
            point_hit(b, .8), point_hit(old[2], .75), point_hit(c, .7)]


def _partition_slugs(resp, key) -> Optional[List[str]]:
    if key:
        return ([h["slug"] for h in resp["retired_partition"]]
                if "retired_partition" in resp else None)
    if "Retired decisions this ranking passed" not in resp:
        return None
    return [ln.split()[0] for ln in _block(resp).splitlines()[1:]
            if ln.startswith("  ") and "(+" not in ln]


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_a_graze_carries_the_retired_decisions_it_passed(ws, driver):
    """Row 9 (F1's row): the members, not the retired hit below the last live."""
    _, m = ws
    hits = _graze_world(m)
    call, _verb, _surface, key = DRIVERS[driver]
    resp = call(ws, CannedVector(hits), limit=2)
    assert _partition_slugs(resp, key) == ["old0", "old1"]
    if key:
        assert [h["slug"] for h in resp[key]] == ["a", "b"]
        assert "retired_partition_omitted" not in resp and "all_superseded" not in resp
        assert all(h["id"] == _id(m, h["slug"]) for h in resp["retired_partition"])
        assert [h["score"] for h in resp["retired_partition"]] == [.95, .85]
    else:
        assert _text_ranks(resp) == ["a", "b"]


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_the_partition_is_bounded_at_limit_and_the_rest_counted(ws, driver):
    """Row 9, bounded (F2/F3's rows): four members on a full window at limit 2."""
    _, m = ws
    old = [_retire(m, f"old{i}") for i in range(4)]
    a, b, c = _live(m, "a", "b", "c")
    hits = ([point_hit(o, .99 - i / 100) for i, o in enumerate(old)]
            + [point_hit(a, .9), point_hit(b, .8), point_hit(c, .7)])
    call, _verb, _surface, key = DRIVERS[driver]
    resp = call(ws, CannedVector(hits), limit=2)
    assert _partition_slugs(resp, key) == ["old0", "old1"]
    if key:
        assert resp["retired_partition_omitted"] == len(old) - 2
    else:
        assert f"(+{len(old) - 2} more retired not listed)" in resp


@pytest.mark.parametrize("driver", list(DRIVERS))
@pytest.mark.parametrize("shape", ["nothing-retired", "all-below-last-live"])
def test_no_partition_when_nothing_was_passed(ws, driver, shape):
    """Row 9: absent when nothing passed, and absent when every retired hit ranked
    below the last live one on a full window (the partition is not "every retired
    point in the fetch")."""
    _, m = ws
    a, b, c = _live(m, "a", "b", "c")
    hits = [point_hit(a, .9), point_hit(b, .8), point_hit(c, .7)]
    if shape == "all-below-last-live":
        hits.insert(2, point_hit(_retire(m, "old"), .75))
    call, _verb, _surface, key = DRIVERS[driver]
    resp = call(ws, CannedVector(hits), limit=2)
    assert _partition_slugs(resp, key) is None
    assert _LEAD_STEM not in _note(resp, key)
    if key:
        assert "retired_partition_omitted" not in resp


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_the_lead_opens_the_note_when_a_retired_decision_outranks_rank_one(ws, driver):
    """Row 10 (F9's row): present at rank 1, absent when the retired hit is mid-window."""
    _, m = ws
    call, _verb, _surface, key = DRIVERS[driver]
    resp = call(ws, CannedVector(_graze_world(m)), limit=2)
    assert _note(resp, key).startswith(
        f"{_LEAD_STEM} old0 (superseded); its current word lives in old0-next. ")

    mid = [point_hit(_id(m, "a"), .95), point_hit(_id(m, "old0"), .9),
           point_hit(_id(m, "b"), .85), point_hit(_id(m, "c"), .8)]
    resp = call(ws, CannedVector(mid), limit=2)
    assert _partition_slugs(resp, key) == ["old0"]
    assert _LEAD_STEM not in _note(resp, key)


# --------------------------------------------------------------------------- #
# Live ends through the surfaces (rows 11, 12)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", list(DRIVERS))
def test_a_two_deep_chain_names_its_live_end_in_handle_and_lead(ws, driver):
    """Row 11: the handle keeps the write-fact stamp and adds ``live_end``; the lead
    sends the agent to the live end, never to the dead hop."""
    _, m = ws
    x = _chain(m, "x")
    a, b, c = _live(m, "a", "b", "c")
    call, _verb, _surface, key = DRIVERS[driver]
    resp = call(ws, CannedVector([point_hit(x, .95), point_hit(a, .9), point_hit(b, .8),
                                  point_hit(c, .7)]), limit=2)
    note = _note(resp, key)
    assert note.startswith(f"{_LEAD_STEM} x (superseded); its current word lives in x-end.")
    assert "x-mid" not in note
    if key:
        (handle,) = resp["retired_partition"]
        assert handle == {"slug": "x", "id": x, "state": "superseded", "score": .95,
                          "superseded_by": ["x-mid"], "live_end": ["x-end"]}
    else:
        assert "superseded_by: x-mid  live_end: x-end" in _block(resp)


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_the_blackout_names_each_handles_live_end(ws, driver):
    """Row 11 on the blackout (F11's row): it no longer names the dead hop."""
    _, m = ws
    x = _chain(m, "x")
    y = _retire(m, "y")
    call, _verb, _surface, key = DRIVERS[driver]
    resp = call(ws, CannedVector([point_hit(x, .9), point_hit(y, .8)]))
    note = _note(resp, key)
    assert "x (superseded → x-end)" in note and "→ x-mid" not in note
    assert "y (superseded → y-next)" in note


@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_a_blackout_writes_one_handle_list_to_both_keys(ws, driver):
    """Row 12 (F13's row): ``all_superseded`` equals ``retired_partition``."""
    _, m = ws
    x = _chain(m, "x")
    y = _retire(m, "y")
    call, _verb, _surface, key = DRIVERS[driver]
    resp = call(ws, CannedVector([point_hit(x, .9), point_hit(y, .8)]))
    assert resp[key] == []
    assert resp["all_superseded"] == resp["retired_partition"]
    assert [h["slug"] for h in resp["all_superseded"]] == ["x", "y"]
    assert resp["all_superseded"][0]["live_end"] == ["x-end"]
    assert "live_end" not in resp["all_superseded"][1]


def test_the_mcp_surface_writes_the_same_list_object_to_both_keys(ws):
    """Row 12 at the object: the surface passes one list, so the two keys cannot
    drift apart (F13 planted a second comprehension)."""
    config, m = ws
    _chain(m, "x")
    seen: Dict[str, Any] = {}
    original = mcp_server.dumps_display

    def capture(obj, **kw):
        seen.update(obj)
        return original(obj, **kw)

    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, _FakeEmbed(),
                                    CannedVector([point_hit(_id(m, "x"), .9)]))), \
            patch.object(mcp_server, "dumps_display", capture):
        mcp_server.surface_decisions(_CLAIM, project=config.workspace_dir)
    assert seen["all_superseded"] is seen["retired_partition"]


# --------------------------------------------------------------------------- #
# By-handle reachability (row 13)
# --------------------------------------------------------------------------- #

def test_a_handles_id_reads_the_retired_node_while_its_slug_reads_the_live_bearer(ws):
    """Row 13: the handle's id dereferences to that node on both surfaces."""
    config, m = ws
    retired, successor = same_slug_lineage(m.store)
    resp = _mcp("surface", ws, CannedVector([point_hit(retired, .9)]))
    (handle,) = resp["all_superseded"]
    assert handle["id"] == retired and handle["slug"] == "database"

    by_id = json.loads(mcp_server.show_node(handle["id"], project=config.workspace_dir))
    by_slug = json.loads(mcp_server.show_node(handle["slug"], project=config.workspace_dir))
    assert by_id["id"] == retired and by_id["state"] == "superseded"
    assert by_slug["id"] == successor and by_slug["state"] == "active"

    for ident, want in ((handle["id"], retired), (handle["slug"], successor)):
        buf = io.StringIO()
        with redirect_stdout(buf):
            cmd_show(config, ident, as_json=True)
        assert json.loads(buf.getvalue())["id"] == want


# --------------------------------------------------------------------------- #
# Faults (row 14 and row 3 at the surface)
# --------------------------------------------------------------------------- #

class _LiveEndFault(_Forwarding):
    def get_live_ends(self, node_ids):
        raise DatabaseError("planted live-end fault")


@pytest.mark.parametrize("wrap", [_StampFault, _LiveEndFault], ids=["stamps", "live-ends"])
@pytest.mark.parametrize("driver", list(DRIVERS))
def test_a_fault_on_a_read_with_handles_keeps_the_handles_and_states_it(ws, driver, wrap):
    """Row 14 (F8's row): the ranking and the handles stand, no successor key, the
    lead says the successor could not be read, and the clause closes the note."""
    config, m = ws
    hits = _graze_world(m)
    call, _verb, surface, key = DRIVERS[driver]
    healthy = call(ws, CannedVector(hits), limit=2)
    faulted = call(ws, CannedVector(hits), limit=2, wrap=wrap)
    clause = stamps_unread_clause(surface=surface, config=config)
    note = _note(faulted, key)
    assert note.startswith(f"{_LEAD_STEM} old0 (superseded); its successor could not be read.")
    assert note.endswith(clause)
    assert _partition_slugs(faulted, key) == _partition_slugs(healthy, key) == ["old0", "old1"]
    if key:
        assert faulted["stamps_unread"] is True
        for h in faulted["retired_partition"]:
            assert set(h) == {"slug", "id", "state", "score"}
        assert [h["slug"] for h in faulted[key]] == [h["slug"] for h in healthy[key]]
    else:
        assert "superseded_by" not in _block(faulted)
        assert _text_ranks(faulted) == _text_ranks(healthy)


@pytest.mark.parametrize("surface", ["mcp", "cli"])
def test_one_envelope_reads_lead_band_withheld_then_stamps_unread(ws, surface):
    """Row 8 end to end: on one real envelope (a thinned graze with a stamp fault)
    every 2f piece sits where §3.4 orders it, the call sites' clauses after them."""
    config, m = ws
    hits = _graze_world(m)
    if surface == "mcp":
        store = _StampFault(GraphStore(config.db_path, read_only=True))
        with patch.object(mcp_server, "get_workspace_components",
                          return_value=(store, _FakeEmbed(), CannedVector(hits))):
            resp = json.loads(mcp_server.surface_decisions(
                _CLAIM, project=config.workspace_dir, limit=2, brief=True))
    else:
        manager = MitosSyncManager(config)
        manager.embed_provider, manager.vector_store = _FakeEmbed(), CannedVector(hits)
        manager.store = _StampFault(manager.store)
        buf = io.StringIO()
        with patch.object(cli, "MitosSyncManager", return_value=manager), redirect_stdout(buf):
            cmd_surface(config, _CLAIM, as_json=True, limit=2, brief=True)
        resp = json.loads(buf.getvalue())
    note = resp["note"]
    withheld = withheld_clause(2, surface=surface, config=config)
    unread = stamps_unread_clause(surface=surface, config=config)
    assert resp["rejected_paths_withheld"] == 2 and resp["stamps_unread"] is True
    assert note.startswith(_LEAD_STEM)
    band = note.index("Ranked top matches.") if "Ranked top matches." in note else note.index(
        "Twilight zone")
    assert 0 < band < note.index(withheld) < note.index(unread)
    assert note.endswith(unread)


# --------------------------------------------------------------------------- #
# The short window (row 15)
# --------------------------------------------------------------------------- #

def _bound_fill(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pads a fetch with orphans to exactly the bound, so the walk ends short_bound."""
    pad = RANKED_OVERFETCH_BOUND - len(hits)
    return hits + [orphan_hit(0.1 - i / 10_000, tag=f"pad-{i}") for i in range(pad)]


@pytest.mark.parametrize("driver", list(DRIVERS))
@pytest.mark.parametrize("kind", ["ran_out", "bound_retired"])
def test_a_short_window_says_why_on_every_driver(ws, driver, kind):
    """Row 15 (F10's row lives in row 7): one live decision at limit 5."""
    _, m = ws
    (a,) = _live(m, "a")
    old = [_retire(m, f"old{i}") for i in range(2)]
    hits = [point_hit(a, .9)] + [point_hit(o, .5 - i / 100) for i, o in enumerate(old)]
    if kind == "bound_retired":
        hits = _bound_fill(hits)
    call, _verb, surface, key = DRIVERS[driver]
    resp = call(ws, CannedVector(hits))
    short = ShortWindow(kind, 1, 5, len(hits), len(old))
    assert _note(resp, key).endswith(short_window_clause(short, surface=surface))


@pytest.mark.parametrize("driver", list(DRIVERS))
@pytest.mark.parametrize("kind", ["ran_out", "bound_retired"])
def test_a_blackout_states_its_short_window_too(ws, driver, kind):
    config, m = ws
    old = [_retire(m, f"old{i}") for i in range(2)]
    hits = [point_hit(o, .9 - i / 100) for i, o in enumerate(old)]
    if kind == "bound_retired":
        hits = _bound_fill(hits)
    call, _verb, surface, key = DRIVERS[driver]
    resp = call(ws, CannedVector(hits))
    handles = [display.retired_handle(
        {"id": o, "slug": f"old{i}", "computed_state": "superseded"}, score=.9 - i / 100,
        stamps={o: {"superseded_by": [f"old{i}-next"]}}, live_ends={})
        for i, o in enumerate(old)]
    expected = blackout_note(handles, short=ShortWindow(kind, 0, 5, len(hits), 2),
                             surface=surface, config=config)
    assert _note(resp, key) == expected
    if key:
        assert resp["all_superseded"] == handles


# --------------------------------------------------------------------------- #
# CLI text (row 16) and the healthy-empty arm (row 16a)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", TEXT_DRIVERS)
def test_the_text_block_shows_every_handle_whole_and_the_count(ws, driver):
    _, m = ws
    old = [_retire(m, f"old{i}") for i in range(4)]
    a, b, c = _live(m, "a", "b", "c")
    hits = ([point_hit(o, .99 - i / 100) for i, o in enumerate(old)]
            + [point_hit(a, .9), point_hit(b, .8), point_hit(c, .7)])
    call, *_ = DRIVERS[driver]
    out = call(ws, CannedVector(hits), limit=2)
    block = _block(out)
    assert f"  old0  (superseded, score 0.990)  id {old[0]}  superseded_by: old0-next" in block
    assert f"id {old[1]}" in block and "old2" not in block
    assert "(+2 more retired not listed)" in block
    # After the hit list, before the note line.
    assert out.index("2. b") < out.index("Retired decisions") < out.index("→ ")


@pytest.mark.parametrize("driver", TEXT_DRIVERS)
def test_the_text_blackout_exit_prints_the_block(ws, driver):
    """F14's row: ``cmd_query``'s early return and ``cmd_surface``'s empty header."""
    _, m = ws
    old = [_retire(m, f"old{i}") for i in range(3)]
    call, *_ = DRIVERS[driver]
    out = call(ws, CannedVector([point_hit(o, .9 - i / 100) for i, o in enumerate(old)]),
               limit=2)
    block = _block(out)
    assert f"id {old[0]}" in block and f"id {old[1]}" in block
    assert "(+1 more retired not listed)" in block
    assert out.index("Retired decisions") < out.index("All 2 nearest matches")


def test_the_text_surface_blackout_with_parked_open_questions_prints_the_block(ws):
    """F14b's row: a blackout with parked open questions skips the empty header and
    takes the lower branch, which must render the handles too."""
    config, m = ws
    old = _retire(m, "old")
    _commit_oq(m.store, "oq-parked", ["x"])
    manager = MitosSyncManager(config)
    manager.embed_provider = _FakeEmbed()
    manager.vector_store = CannedVector([point_hit(old, .9)])
    buf = io.StringIO()
    with patch.object(cli, "MitosSyncManager", return_value=manager), redirect_stdout(buf):
        cmd_surface(config, _CLAIM, scope="x")
    out = buf.getvalue()
    assert "[open question in scope] oq-parked" in out
    block = _block(out)
    assert f"  old  (superseded, score 0.900)  id {old}" in block
    assert out.index("Retired decisions") < out.index("[open question in scope]")


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_the_healthy_empty_missing_collection_arm_carries_no_partition(ws, driver):
    """Row 16a: an empty active set and no collection — no walk, so no partition, no
    lead, no short clause, and no unbound name on any surface."""
    call, _verb, _surface, key = DRIVERS[driver]
    resp = call(ws, _MissingCollection())
    text = json.dumps(resp) if key else resp
    assert "retired_partition" not in text and "Retired decisions" not in text
    assert _LEAD_STEM not in text and "This window is short" not in text
    if key:
        assert resp["confidence"] == "none" and "degraded" not in resp


# --------------------------------------------------------------------------- #
# Parity (row 17)
# --------------------------------------------------------------------------- #

def _parity_world(m, shape: str) -> List[Dict[str, Any]]:
    if shape == "graze":
        return _graze_world(m)
    x = _chain(m, "x")
    if shape == "blackout":
        return [point_hit(x, .9), point_hit(_retire(m, "y"), .8)]
    (a,) = _live(m, "a")
    return [point_hit(x, .95), point_hit(a, .9)]                 # short window


#: Keys whose values the two harness builds legitimately differ on: the MCP call
#: resolves the registered name and the CLI row hands a path-form config (3d's
#: `_resolve_like_main` note), and only the MCP surface composes a notice here.
_HARNESS_KEYS = {"project", "collection", "workspace", "check_notice", "note"}


@pytest.mark.parametrize("verb", ["surface", "query"])
@pytest.mark.parametrize("shape", ["graze", "blackout", "short"])
def test_cli_json_and_mcp_agree_on_the_partition(ws, verb, shape):
    _, m = ws
    hits = _parity_world(m, shape)
    mcp_resp = _mcp(verb, ws, CannedVector(hits), limit=2)
    cli_resp = _cli(verb, ws, CannedVector(hits), as_json=True, limit=2)
    assert set(mcp_resp) - {"check_notice"} == set(cli_resp) - {"check_notice"}
    assert "retired_partition" in mcp_resp
    for k in set(mcp_resp) - _HARNESS_KEYS:
        assert mcp_resp[k] == cli_resp[k], k


# --------------------------------------------------------------------------- #
# Constraint 4 — the ten-to-one neighbourhood at the ceiling (row 18, T4)
# --------------------------------------------------------------------------- #

_RETIRED_PER_LIVE = 10


@pytest.fixture(scope="module")
def ten_to_one(tmp_path_factory):
    """A neighbourhood of 10-deep kill chains, each ending in one live decision,
    ranked chain by chain: ``r0 … r9, L`` per group. Built through the public commit
    path; the build time is reported in the row."""
    root = tmp_path_factory.mktemp("ten_to_one")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("XDG_CONFIG_HOME", str(root / "xdg"))
        mp.setenv("QDRANT_URL", "http://127.0.0.1:9")
        for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
            mp.delenv(k, raising=False)
        config = MitosConfig(str(root / "ws"))
        cmd_init(config)
        store = GraphStore(config.db_path)
        started = time.perf_counter()
        groups = -(-RANKED_OVERFETCH_BOUND // (_RETIRED_PER_LIVE + 1))
        ranked: List[str] = []
        for g in range(groups):
            ids = []
            for i in range(_RETIRED_PER_LIVE):
                prior = [f"g{g}-r{i - 1}"] if i else None
                ids.append(_commit(store, f"g{g}-r{i}", supersedes=prior))
            ids.append(_commit(store, f"g{g}-live",
                               supersedes=[f"g{g}-r{_RETIRED_PER_LIVE - 1}"]))
            ranked.extend(ids)
        elapsed = time.perf_counter() - started
    return config, ranked, elapsed


@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_ten_retired_to_each_live_at_the_ceiling(ten_to_one, driver):
    """Row 18: a canned fetch of exactly the bound, ten retired to each live, at
    ``limit=50`` — the walk ends short_bound with fewer than 50 live, says so with
    its retired count, and shows 50 handles closest first with the rest counted.
    Every expected number is derived here from the fixture."""
    config, ranked, elapsed = ten_to_one
    assert elapsed < 30, f"the public-path build took {elapsed:.1f}s"
    limit = display.RANKED_LIMIT_CEILING
    fetch = ranked[:RANKED_OVERFETCH_BOUND]
    hits = [point_hit(nid, 0.99 - i / 1000) for i, nid in enumerate(fetch)]
    group = _RETIRED_PER_LIVE + 1
    live_n = sum(1 for i in range(len(fetch)) if i % group == _RETIRED_PER_LIVE)
    retired_n = len(fetch) - live_n
    assert live_n < limit

    call, _verb, surface, key = DRIVERS[driver]
    resp = call((config, None), CannedVector(hits), limit=limit)
    assert len(resp[key]) == live_n
    clause = short_window_clause(
        ShortWindow("bound_retired", live_n, limit, RANKED_OVERFETCH_BOUND, retired_n),
        surface=surface)
    assert resp["note"].endswith(clause)
    partition = resp["retired_partition"]
    expected = [nid for i, nid in enumerate(fetch) if i % group != _RETIRED_PER_LIVE][:limit]
    assert [h["id"] for h in partition] == expected
    assert [h["score"] for h in partition] == sorted(
        [h["score"] for h in partition], reverse=True)
    assert resp["retired_partition_omitted"] == retired_n - limit
    # Live ends: every member but a chain's last retired names its group's live.
    for h in partition:
        g, i = h["slug"].split("-r")
        if int(i) < _RETIRED_PER_LIVE - 1:
            assert h["live_end"] == [f"{g}-live"], h
        else:
            assert "live_end" not in h and h["superseded_by"] == [f"{g}-live"]
