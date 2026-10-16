"""Phase 2c: the shared identity walk (``mitos/retrieval_walk.py``).

``fill`` is the pure core: rows drive it over hand-built resolutions, so every rule
of the walk is pinned without a store. ``walk_ranked`` is the shell production calls:
rows drive it over a real temp ``GraphStore`` and a ``CannedVector`` (which records
each call), and pin its three reads — one vector call at the bound with ids only, one
batched resolve, one batched stamp read — and its contract errors, raised before the
query so a wrong double fails on an empty fetch too.

Ids are derived (``commit_parsed_entry``'s delta, ``hash_to_uuid``), never hardcoded.
"""

import sqlite3
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

import mitos.retrieval_walk as walk_module
from mitos.display import RANKED_LIMIT_CEILING
from mitos.errors import DatabaseError
from mitos.parser import ParsedEntry
from mitos.retrieval_walk import (EDGE_FOUND, EDGE_UNMEASURED, EXHAUSTED,
                                  LIVE_STATES, RANKED_OVERFETCH_BOUND, SHORT_BOUND,
                                  SKIP_KEYS, PointResolution, WalkContractError,
                                  fill, walk_ranked)
from mitos.store import GraphStore
from mitos.vector_store import hash_to_uuid

from _point_hits import CannedVector, orphan_hit, point_hit


# --------------------------------------------------------------------------- #
# The pure core over hand-built resolutions
# --------------------------------------------------------------------------- #

def _node(tag: str, *, kind: str = "decision", state: str = "active") -> Dict[str, Any]:
    return {"id": f"{tag:0>64}"[-64:], "slug": tag, "kind": kind, "computed_state": state}


class _World:
    """Point ids → nodes, plus orphans and ambiguous ids, built one hit at a time."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.orphans: List[str] = []
        self.ambiguous: Dict[str, List[str]] = {}
        self.hits: List[Dict[str, Any]] = []

    def hit(self, pid: str, score: float, node: Optional[Dict[str, Any]] = None, *,
            orphan: bool = False, ambiguous: bool = False) -> "_World":
        if orphan:
            self.orphans.append(pid)
        elif ambiguous:
            self.ambiguous[pid] = ["a" * 64, "b" * 64]
        else:
            self.nodes[pid] = node
        self.hits.append({"point_id": pid, "score": score})
        return self

    def live(self, tag: str, score: float, state: str = "active") -> "_World":
        return self.hit(f"p-{tag}", score, _node(tag, state=state))

    def retired(self, tag: str, score: float, state: str = "superseded") -> "_World":
        return self.hit(f"p-{tag}", score, _node(tag, state=state))

    def resolution(self) -> PointResolution:
        return PointResolution(nodes=dict(self.nodes), orphans=list(self.orphans),
                               ambiguous=dict(self.ambiguous))


def _fill(world: _World, *, limit: int, bound: int):
    return fill(world.hits, world.resolution(), limit=limit, bound=bound)


def _slugs(hits) -> List[str]:
    return [h.node["slug"] for h in hits]


def test_edge_found_holds_the_window_and_the_next_live_rank() -> None:
    w = _World().live("a", .9).retired("r", .85).live("b", .8).live("c", .7).live("d", .6)
    walk = _fill(w, limit=2, bound=10)
    assert walk.end == EDGE_FOUND
    assert _slugs(walk.window) == ["a", "b"]
    assert walk.edge.node["slug"] == "c" and walk.edge.rank == 4
    # The walk stopped at the edge: `d` was never read.
    assert _slugs(walk.retired) == ["r"]
    assert walk.returned == 5 and walk.bound == 10 and walk.limit == 2


def test_edge_unmeasured_is_a_full_window_at_the_bound_with_no_further_live() -> None:
    w = _World().live("a", .9).live("b", .8).retired("r", .7)
    walk = _fill(w, limit=2, bound=3)
    assert walk.end == EDGE_UNMEASURED
    assert _slugs(walk.window) == ["a", "b"] and walk.edge is None


def test_short_bound_is_fewer_than_limit_live_when_the_bound_is_reached() -> None:
    w = _World().retired("r1", .9).live("a", .8).retired("r2", .7)
    walk = _fill(w, limit=2, bound=3)
    assert walk.end == SHORT_BOUND
    assert _slugs(walk.window) == ["a"] and _slugs(walk.retired) == ["r1", "r2"]


@pytest.mark.parametrize("n_live", [1, 2])
def test_exhausted_is_a_fetch_shorter_than_the_bound_full_or_short(n_live) -> None:
    w = _World()
    for i in range(n_live):
        w.live(f"l{i}", .9 - i / 10)
    walk = _fill(w, limit=2, bound=5)
    assert walk.end == EXHAUSTED and walk.edge is None
    assert len(walk.window) == n_live


def test_an_empty_fetch_is_exhausted() -> None:
    walk = _fill(_World(), limit=5, bound=350)
    assert walk.end == EXHAUSTED
    assert walk.window == [] and walk.retired == [] and walk.edge is None
    assert walk.stamps == {} and walk.stamps_unread is False


def test_each_skip_is_counted_under_its_own_key_and_all_four_keys_always_exist() -> None:
    a = _node("a")
    w = (_World().hit("p-orphan", .99, orphan=True)
         .hit("p-amb", .98, ambiguous=True)
         .hit("p-oq", .97, _node("oq", kind="open_question"))
         .hit("p-a", .96, a)
         .hit("p-a-again", .95, a))   # the same node under a second point
    walk = _fill(w, limit=5, bound=350)
    assert walk.skipped == {"orphan": 1, "ambiguous": 1, "non_decision": 1, "duplicate": 1}
    assert _slugs(walk.window) == ["a"]
    assert tuple(_fill(_World(), limit=1, bound=2).skipped) == SKIP_KEYS


def test_a_point_absent_from_every_bucket_counts_as_an_orphan_never_guessed() -> None:
    walk = fill([{"point_id": "never-asked", "score": .9}],
                PointResolution(nodes={}, orphans=[], ambiguous={}), limit=1, bound=5)
    assert walk.skipped["orphan"] == 1 and walk.window == []


def test_a_duplicate_retired_node_is_counted_once_too() -> None:
    r = _node("r", state="superseded")
    w = _World().hit("p-r", .9, r).hit("p-r2", .8, r)
    walk = _fill(w, limit=1, bound=350)
    assert _slugs(walk.retired) == ["r"] and walk.skipped["duplicate"] == 1


@pytest.mark.parametrize("state", ["active", "resolved", "parked", "superseded"])
def test_kind_is_tested_before_state(state) -> None:
    """A resolved open question reports `active` (PATTERNS "scope-blind KNN"): it is a
    non-decision, never live and never retired."""
    w = _World().hit("p-oq", .9, _node("oq", kind="open_question", state=state))
    walk = _fill(w, limit=1, bound=350)
    assert walk.window == [] and walk.retired == []
    assert walk.skipped["non_decision"] == 1


def test_drifted_is_live_and_superseded_and_corrected_are_retired() -> None:
    w = (_World().live("d", .9, state="drifted").retired("s", .8, state="superseded")
         .retired("c", .7, state="corrected"))
    walk = _fill(w, limit=5, bound=350)
    assert _slugs(walk.window) == ["d"] and _slugs(walk.retired) == ["s", "c"]


def test_every_rank_is_the_hits_position_in_the_fetch() -> None:
    w = (_World().hit("p-o", .99, orphan=True).retired("r1", .9).live("a", .8)
         .hit("p-oq", .7, _node("oq", kind="open_question")).retired("r2", .6)
         .live("b", .5).retired("r3", .4))
    walk = _fill(w, limit=1, bound=350)
    assert [(h.node["slug"], h.rank) for h in walk.window] == [("a", 3)]
    assert [(h.node["slug"], h.rank) for h in walk.retired] == [("r1", 2), ("r2", 5)]
    assert walk.edge.rank == 6
    # The retired hit past the edge was never passed.
    assert "r3" not in _slugs(walk.retired)


def test_fill_accepts_a_limit_at_or_past_the_bound() -> None:
    """2b's horizon harness drives B = limit + 1 and below; only `walk_ranked` refuses."""
    w = _World().live("a", .9).live("b", .8)
    assert _fill(w, limit=2, bound=2).end == EDGE_UNMEASURED
    assert _fill(w, limit=5, bound=2).end == SHORT_BOUND


@pytest.mark.parametrize("hit", [{"point_id": "p-1"}, {"point_id": "p-1", "score": None},
                                 {"point_id": "p-1", "score": True}],
                         ids=["no-score", "null-score", "bool-score"])
def test_a_hit_without_a_numeric_score_is_a_contract_error(hit) -> None:
    """Left to a bare `KeyError`, a missing score would raise inside a surface's broad
    catch and read as degraded recall."""
    with pytest.raises(WalkContractError, match="score"):
        fill([hit], PointResolution({}, [], {}), limit=1, bound=5)


def test_a_hit_without_a_string_point_id_is_a_contract_error() -> None:
    with pytest.raises(WalkContractError, match="point_id"):
        fill([{"score": .9}], PointResolution({}, [], {}), limit=1, bound=5)
    with pytest.raises(WalkContractError, match="point_id"):
        fill([{"point_id": 7, "score": .9}], PointResolution({}, [], {}), limit=1, bound=5)


# --------------------------------------------------------------------------- #
# The shell over a real store
# --------------------------------------------------------------------------- #

@pytest.fixture
def store(tmp_path) -> GraphStore:
    return GraphStore(str(tmp_path / "graph.sqlite"))


def _commit(store: GraphStore, slug: str, *, kind: str = "decision",
            supersedes: Optional[List[str]] = None,
            amends: Optional[List[str]] = None) -> str:
    entry = ParsedEntry(kind, slug, 1, 10)
    if kind == "decision":
        entry.axiom = f"Axiom for {slug}."
        entry.rejected_paths = "Rejected the obvious alternative."
    else:
        entry.topic = f"Topic for {slug}"
        entry.questions_raised = [f"What about {slug}?"]
    entry.scope = ["core"]
    if supersedes:
        entry.supersedes = supersedes
    if amends:
        entry.amends = amends
    return store.commit_parsed_entry(entry).node_id


class _SpyGraph:
    """Forwards the two batched reads to a real store and counts them; every per-hit
    read a slug-keyed loop used raises, so a walk that fell back to one is loud."""

    def __init__(self, inner: GraphStore) -> None:
        self._inner = inner
        self.resolved: List[List[str]] = []
        self.stamped: List[List[str]] = []

    def resolve_point_ids(self, point_ids):
        ids = list(point_ids)
        self.resolved.append(ids)
        return self._inner.resolve_point_ids(ids)

    def get_modifiers_map(self, node_ids):
        self.stamped.append(list(node_ids))
        return self._inner.get_modifiers_map(node_ids)

    def _per_hit(self, *_a, **_k):
        raise AssertionError("the walk made a per-hit store read")

    get_node_by_slug = get_node_state = get_modifiers = resolve_slug = _per_hit
    resolve_slug_kinds = get_node = _per_hit


def _ids(store: GraphStore, *slugs: str) -> List[str]:
    return [store.resolve_slug(s)[0] for s in slugs]


def test_walk_ranked_makes_exactly_its_three_reads(store) -> None:
    a, b, c = (_commit(store, s) for s in ("a", "b", "c"))
    old = _commit(store, "old")
    _commit(store, "new", supersedes=["old"])
    _commit(store, "amender", amends=["a"])
    hits = [point_hit(old, .95), point_hit(a, .9), point_hit(b, .8), point_hit(c, .7)]
    vec = CannedVector(hits, store=store)
    graph = _SpyGraph(store)

    walk = walk_ranked([0.1], vector_store=vec, graph=graph, limit=2)

    assert vec.calls == [(RANKED_OVERFETCH_BOUND, False)]
    assert graph.resolved == [[h["point_id"] for h in hits]]
    assert graph.stamped == [[a, b, old]]          # window, then retired[:limit]
    assert _slugs(walk.window) == ["a", "b"] and walk.edge.node["id"] == c
    assert walk.stamps == {a: {"amended_by": ["amender"]}, old: {"superseded_by": ["new"]}}
    assert walk.stamps_unread is False


def test_only_the_first_limit_retired_hits_are_stamped(store) -> None:
    olds = [_commit(store, f"old-{i}") for i in range(4)]
    for i in range(4):
        _commit(store, f"new-{i}", supersedes=[f"old-{i}"])
    graph = _SpyGraph(store)
    walk = walk_ranked([0.1], vector_store=CannedVector(
        [point_hit(o, .9 - i / 10) for i, o in enumerate(olds)], store=store),
        graph=graph, limit=2)
    assert _slugs(walk.retired) == [f"old-{i}" for i in range(4)]
    assert graph.stamped == [olds[:2]]
    assert set(walk.stamps) == set(olds[:2])


@pytest.mark.parametrize("shape", ["short_bound", "exhausted"])
def test_one_vector_call_never_a_refetch(store, shape) -> None:
    """§4.3: a short window is stated, never chased with a second, deeper call."""
    live = _commit(store, "live")
    olds = [_commit(store, f"old-{i}") for i in range(3)]
    for i in range(3):
        _commit(store, f"new-{i}", supersedes=[f"old-{i}"])
    hits = [point_hit(o, .9) for o in olds] + [point_hit(live, .5)]
    vec = CannedVector(hits, store=store)
    bound = 4 if shape == "short_bound" else RANKED_OVERFETCH_BOUND
    walk = walk_ranked([0.1], vector_store=vec, graph=store, limit=2, bound=bound)
    assert walk.end == (SHORT_BOUND if shape == "short_bound" else EXHAUSTED)
    assert vec.queries == 1


def test_an_empty_fetch_makes_no_stamp_read(store) -> None:
    graph = _SpyGraph(store)
    walk = walk_ranked([0.1], vector_store=CannedVector([], store=store), graph=graph,
                       limit=5)
    assert walk.end == EXHAUSTED and graph.stamped == []


def test_read_stamps_false_makes_no_stamp_read_and_needs_no_stamp_reader(store) -> None:
    a = _commit(store, "a")
    _commit(store, "amender", amends=["a"])

    class _ResolveOnly:
        def __init__(self, inner):
            self._inner = inner

        def resolve_point_ids(self, ids):
            return self._inner.resolve_point_ids(ids)

    walk = walk_ranked([0.1], vector_store=CannedVector([point_hit(a, .9)], store=store),
                       graph=_ResolveOnly(store), limit=1, read_stamps=False)
    assert _slugs(walk.window) == ["a"]
    assert walk.stamps == {} and walk.stamps_unread is False


# ---- the stamp fault: costs the stamps alone ------------------------------- #

class _StampFault(_SpyGraph):
    def __init__(self, inner, exc):
        super().__init__(inner)
        self._exc = exc

    def get_modifiers_map(self, node_ids):
        self.stamped.append(list(node_ids))
        raise self._exc


def test_a_stamp_fault_costs_the_stamps_alone(store) -> None:
    a = _commit(store, "a")
    _commit(store, "amender", amends=["a"])
    walk = walk_ranked([0.1], vector_store=CannedVector([point_hit(a, .9)], store=store),
                       graph=_StampFault(store, DatabaseError("planted")), limit=1)
    assert _slugs(walk.window) == ["a"]
    assert walk.stamps == {} and isinstance(walk.stamps, dict)
    assert walk.stamps_unread is True


def test_a_real_sqlite_stamp_fault_is_typed_and_contained(store, monkeypatch) -> None:
    """The store wraps the modifier query's `sqlite3.Error` as `DatabaseError`, so a
    locked or malformed graph costs the stamps, not the read (scout D1)."""
    a = _commit(store, "a")
    original = store._get_connection

    class _Faulting:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, params=()):
            if "FROM edges e" in sql:
                raise sqlite3.OperationalError("database is locked")
            return self._conn.execute(sql, params)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    monkeypatch.setattr(store, "_get_connection", lambda: _Faulting(original()))
    with pytest.raises(DatabaseError, match="modifiers"):
        store.get_modifiers_map([a])
    walk = walk_ranked([0.1], vector_store=CannedVector([point_hit(a, .9)], store=store),
                       graph=store, limit=1)
    assert _slugs(walk.window) == ["a"] and walk.stamps_unread is True


def test_the_stamp_catch_is_narrow(store) -> None:
    """K7's row: a defect in the stamp reader is not a stamp fault — it propagates."""
    a = _commit(store, "a")
    with pytest.raises(RuntimeError, match="defect"):
        walk_ranked([0.1], vector_store=CannedVector([point_hit(a, .9)], store=store),
                    graph=_StampFault(store, RuntimeError("defect")), limit=1)


def test_a_resolver_fault_propagates_for_the_surface_to_degrade(store) -> None:
    a = _commit(store, "a")

    class _ResolveFault(_SpyGraph):
        def resolve_point_ids(self, ids):
            raise DatabaseError("planted resolver fault")

    with pytest.raises(DatabaseError, match="resolver"):
        walk_ranked([0.1], vector_store=CannedVector([point_hit(a, .9)], store=store),
                    graph=_ResolveFault(store), limit=1)


# ---- the contract ---------------------------------------------------------- #

class _NoPointId:
    """Wraps a canned store and strips every hit's point id."""

    def __init__(self, inner):
        self._inner = inner

    def query(self, vector, limit=5, *, with_payload=True):
        return [{"score": h["score"]}
                for h in self._inner.query(vector, limit, with_payload=with_payload)]


def test_a_magicmock_store_fails_loud(store) -> None:
    """Ledger entry-002: a MagicMock absorbs `resolve_point_ids` and returns a
    MagicMock; the isinstance check refuses it."""
    a = _commit(store, "a")
    with pytest.raises(WalkContractError, match="PointResolution"):
        walk_ranked([0.1], vector_store=CannedVector([point_hit(a, .9)], store=store),
                    graph=MagicMock(), limit=1)


def test_non_dict_stamps_fail_loud(store) -> None:
    a = _commit(store, "a")

    class _ListStamps(_SpyGraph):
        def get_modifiers_map(self, node_ids):
            return []

    with pytest.raises(WalkContractError, match="dict"):
        walk_ranked([0.1], vector_store=CannedVector([point_hit(a, .9)], store=store),
                    graph=_ListStamps(store), limit=1)


def test_a_hit_without_point_id_fails_loud(store) -> None:
    a = _commit(store, "a")
    with pytest.raises(WalkContractError, match="point_id"):
        walk_ranked([0.1], vector_store=_NoPointId(CannedVector([point_hit(a, .9)],
                                                                store=store)),
                    graph=store, limit=1)


@pytest.mark.parametrize("missing", ["resolve_point_ids", "get_modifiers_map"])
def test_a_missing_store_method_fails_at_entry_even_on_an_empty_fetch(store, missing):
    """K16's row: the entry check runs before the query and before any empty-window
    skip, so a double lacking a method cannot pass on a zero-hit fetch."""

    class _Partial:
        pass

    graph = _Partial()
    for name in ("resolve_point_ids", "get_modifiers_map"):
        if name != missing:
            setattr(graph, name, getattr(store, name))
    vec = CannedVector([], store=store)
    with pytest.raises(WalkContractError, match=missing):
        walk_ranked([0.1], vector_store=vec, graph=graph, limit=1)
    assert vec.queries == 0


def test_a_vector_store_without_query_fails_at_entry(store) -> None:
    with pytest.raises(WalkContractError, match="query"):
        walk_ranked([0.1], vector_store=object(), graph=store, limit=1)


@pytest.mark.parametrize("limit", [0, -1, RANKED_OVERFETCH_BOUND, RANKED_OVERFETCH_BOUND + 1,
                                   True, 2.0])
def test_limit_outside_one_to_bound_minus_one_fails_at_entry(store, limit) -> None:
    vec = CannedVector([], store=store)
    with pytest.raises(WalkContractError, match="limit"):
        walk_ranked([0.1], vector_store=vec, graph=store, limit=limit)
    assert vec.queries == 0


def test_the_contract_error_is_a_type_error() -> None:
    assert issubclass(WalkContractError, TypeError)


# --------------------------------------------------------------------------- #
# Identity through the store: a same-slug pair and a rename
# --------------------------------------------------------------------------- #

def test_a_same_slug_pair_resolves_each_point_to_its_own_node(store) -> None:
    from test_point_identity import same_slug_lineage
    retired, successor = same_slug_lineage(store)
    hits = [point_hit(retired, .9), point_hit(successor, .8)]
    walk = walk_ranked([0.1], vector_store=CannedVector(hits, store=store), graph=store,
                       limit=5)
    assert [(h.node["id"], h.score) for h in walk.window] == [(successor, .8)]
    assert [(h.node["id"], h.score) for h in walk.retired] == [(retired, .9)]
    assert walk.stamps[retired] == {"superseded_by": ["database"]}


def test_an_orphan_point_is_skipped_and_the_window_fills(store) -> None:
    a, b = _commit(store, "a"), _commit(store, "b")
    hits = [orphan_hit(.99, slug="a"), point_hit(a, .9), point_hit(b, .8)]
    walk = walk_ranked([0.1], vector_store=CannedVector(hits, store=store), graph=store,
                       limit=2)
    assert _slugs(walk.window) == ["a", "b"] and walk.skipped["orphan"] == 1


# --------------------------------------------------------------------------- #
# Agreement and identity rows
# --------------------------------------------------------------------------- #

def test_live_states_agree_with_the_stores() -> None:
    from mitos.store import LIVE_STATES as STORE_LIVE_STATES
    assert LIVE_STATES == STORE_LIVE_STATES


def test_point_resolution_is_one_class_reachable_from_the_store() -> None:
    import mitos.protocols
    import mitos.store
    assert mitos.store.PointResolution is walk_module.PointResolution
    assert mitos.protocols.PointResolution is walk_module.PointResolution


def test_the_bound_leaves_room_for_the_edge_at_the_ceiling() -> None:
    assert RANKED_OVERFETCH_BOUND > RANKED_LIMIT_CEILING
    assert RANKED_OVERFETCH_BOUND == 350


def test_a_point_id_round_trips_to_its_node_through_the_walk(store) -> None:
    a = _commit(store, "a")
    walk = walk_ranked([0.1], vector_store=CannedVector([point_hit(a, .9)], store=store),
                       graph=store, limit=1)
    assert walk.window[0].point_id == hash_to_uuid(a)


# --------------------------------------------------------------------------- #
# 2d: last_score, the score of the fetch's last hit
# --------------------------------------------------------------------------- #

def test_last_score_is_the_fetchs_last_hit_even_when_the_walk_stops_early() -> None:
    """The gather judges a cut on the last *raw* score, which the walk alone sees."""
    world = _World().live("a", 0.9).live("b", 0.8).live("c", 0.7).retired("d", 0.6)
    walk = _fill(world, limit=1, bound=4)
    assert walk.end == EDGE_FOUND          # stopped at rank 2
    assert walk.last_score == 0.6          # still the fetch's last hit


def test_last_score_is_none_for_an_empty_fetch() -> None:
    assert _fill(_World(), limit=1, bound=5).last_score is None


def test_a_last_hit_past_the_edge_without_a_score_still_breaks_the_contract() -> None:
    """The walk may stop before the last hit; that hit is checked anyway."""
    world = _World().live("a", 0.9).live("b", 0.8)
    hits = world.hits + [{"point_id": "p-tail"}]
    with pytest.raises(WalkContractError):
        fill(hits, world.resolution(), limit=1, bound=3)
