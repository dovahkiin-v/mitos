"""Phase 5a: the rejection section on the four ranked reads.

Every ranked read (MCP ``surface_decisions`` and ``query_decisions``' ranked branch,
``mitos surface`` and ``mitos query``) searches the rejection index with the query
vector it already holds, inside the shared walk (``retrieval_walk.walk_ranked``'s
leg). The read keeps only matches the owner's **current** field still states,
shapes each by where its owner already sits in the answer, and names every way the
section was not computed whole (gap, not searched, unavailable, short) without
touching the decision answer.

The rows drive a real SQLite graph, ``FakeRejectionIndex`` bound at the read seam
(``_ranked_rejection_index``) and a canned decision vector, so every score is placed
deliberately: the query vector is one axis and a point's vector sits at a chosen
cosine to it. The floor is always read from ``recall`` (it is provisional, PENDING
5b), never hard-coded.
"""

import ast
import inspect
import io
import json
import math
import os
import re
import shutil
import tempfile
from contextlib import redirect_stdout
from typing import Any, Dict, Iterator, List, Optional, Tuple
from unittest.mock import patch

import pytest

import mitos.cli as cli
import mitos.mcp_server as mcp_server
import mitos.retrieval_walk as walk_module
from mitos.cli import cmd_init, cmd_query, cmd_surface
from mitos.config import MitosConfig
from mitos.display import RANKED_LIMIT_CEILING, rejection_matches_payload
from mitos.errors import CollectionMissingError, DatabaseError
from mitos.parser import ParsedEntry
from mitos.recall import (REJECTION_MATCH_FLOOR, REJECTION_SECTION_BOUND,
                          count_withheld, count_withheld_decisions,
                          rejection_owners_whole, rejection_section_clause,
                          rejection_section_state, withheld_clause)
from mitos.rejections import point_items
from mitos.retrieval_walk import (REJECTION_DROP_KEYS, REJECTION_OVERFETCH_BOUND,
                                  REJECTION_SEARCH_BUDGET_SECONDS, PointResolution,
                                  RejectionLeg, WalkContractError, fill_rejections,
                                  stamp_ids_of, walk_ranked)
from mitos.shell_quote import shell_flag
from mitos.store import GraphStore
from mitos.vector_store import RejectionHit, RejectionSearch, hash_to_uuid

from _point_hits import CannedVector, point_hit
from _rejection_index_double import FakeRejectionIndex

DIM = 8
FLOOR = REJECTION_MATCH_FLOOR
QUERY = "a claim that is not any slug"
SECTION_KEYS = ("rejection_matches", "rejection_section", "rejection_dropped")

F_ABC = "1. Alpha — a.\n2. Bravo — b.\n3. Charlie — c."
F_BD = "1. Bravo — b.\n2. Delta — d."
F_ONE = "Rejected the obvious alternative."
F_TWO = "1. Keep the old — it drifted.\n2. Buy a tool — it locks us in."


def _query_vector() -> List[float]:
    return [1.0] + [0.0] * (DIM - 1)


def _at(score: float) -> List[float]:
    """A unit vector at cosine ``score`` to :func:`_query_vector`."""
    vector = [0.0] * DIM
    vector[0] = score
    vector[1] = math.sqrt(max(0.0, 1.0 - score * score))
    return vector


ABOVE = round(min(0.99, FLOOR + 0.2), 4)
JUST_ABOVE = round(FLOOR + 0.05, 4)
BELOW = round(FLOOR - 0.2, 4)


class _Embed:
    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        return _query_vector()


class _StubManager:
    def __init__(self, store, embed_provider, vector_store):
        self.store = store
        self.embed_provider = embed_provider
        self.vector_store = vector_store


# --------------------------------------------------------------------------- #
# The workspace and the fixtures' graph
# --------------------------------------------------------------------------- #

@pytest.fixture
def ws(monkeypatch) -> Iterator[MitosConfig]:
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    cmd_init(config)
    yield config
    shutil.rmtree(tmp, ignore_errors=True)


def _commit(config: MitosConfig, slug: str, field: str = F_ONE, *,
            supersedes: Optional[List[str]] = None, amends: Optional[List[str]] = None,
            scope: Optional[List[str]] = None) -> str:
    """Commits (or re-commits) a decision through the store; embeds nothing."""
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = f"The {slug} axiom stands on its own words."
    entry.rejected_paths = field
    entry.scope = list(scope or ["core"])
    if supersedes:
        entry.supersedes = supersedes
    if amends:
        entry.amends = amends
    return GraphStore(config.db_path).commit_parsed_entry(entry).node_id


def _commit_oq(config: MitosConfig, slug: str) -> str:
    entry = ParsedEntry("open_question", slug, 1, 5)
    entry.topic = f"Topic for {slug}"
    entry.questions_raised = [f"What about {slug}?"]
    entry.scope = ["core"]
    return GraphStore(config.db_path).commit_parsed_entry(entry).node_id


def _plant(index: FakeRejectionIndex, owner: str, field: str,
           scores: Dict[int, float]) -> Dict[int, str]:
    """Puts an owner's items in the index, each ordinal at its score; returns the ids."""
    ids = {}
    for pid, item in point_items(owner, field).items():
        if item.ordinal in scores:
            index.points[pid] = {"vector": _at(scores[item.ordinal]),
                                 "payload": {"owner_id": owner, "model": "m"}}
            ids[item.ordinal] = pid
    return ids


def _bind(monkeypatch, index: Any) -> None:
    monkeypatch.setattr(cli, "_ranked_rejection_index", lambda config: index)
    monkeypatch.setattr(mcp_server, "_ranked_rejection_index", lambda config: index)


def _bind_leg(monkeypatch, index: Any, **kw: Any) -> None:
    """Binds a leg with non-default knobs (``fetch``, ``budget_seconds``) at both."""
    def leg(config):
        return RejectionLeg(index=index, items_of=point_items, floor=FLOOR,
                            bound=REJECTION_SECTION_BOUND, **kw)
    monkeypatch.setattr(cli, "_rejection_leg", leg)
    monkeypatch.setattr(mcp_server, "_rejection_leg", leg)


# --------------------------------------------------------------------------- #
# The four surfaces (JSON) and the two CLI text renders
# --------------------------------------------------------------------------- #

def _vector(matches, store):
    return CannedVector(matches, store=store) if isinstance(matches, list) else matches


def _mcp_surface(config, matches, **kw):
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, _Embed(), _vector(matches, store))):
        return json.loads(mcp_server.surface_decisions(
            QUERY, project=config.workspace_dir, **kw))


def _mcp_query(config, matches, query=QUERY, **kw):
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, _Embed(), _vector(matches, store))):
        return json.loads(mcp_server.query_decisions(
            query, project=config.workspace_dir, **kw))


def _cli_run(config, matches, verb, *, as_json, **kw):
    store = GraphStore(config.db_path, read_only=True)
    stub = _StubManager(store, _Embed(), _vector(matches, store))
    buf = io.StringIO()
    with patch.object(cli, "MitosSyncManager", return_value=stub):
        with redirect_stdout(buf):
            (cmd_surface if verb == "surface" else cmd_query)(
                config, QUERY, as_json=as_json, **kw)
    return buf.getvalue()


def _cli_surface(config, matches, **kw):
    return json.loads(_cli_run(config, matches, "surface", as_json=True, **kw))


def _cli_query(config, matches, **kw):
    return json.loads(_cli_run(config, matches, "query", as_json=True, **kw))


def _cli_surface_text(config, matches, **kw):
    return _cli_run(config, matches, "surface", as_json=False, **kw)


def _cli_query_text(config, matches, **kw):
    return _cli_run(config, matches, "query", as_json=False, **kw)


# driver → (call, list key, surface)
DRIVERS = {
    "mcp-surface": (_mcp_surface, "active_decisions", "mcp"),
    "mcp-query": (_mcp_query, "matches", "mcp"),
    "cli-surface": (_cli_surface, "active_decisions", "cli"),
    "cli-query": (_cli_query, "matches", "cli"),
}
TEXT_DRIVERS = {"cli-surface": _cli_surface_text, "cli-query": _cli_query_text}


def _call(driver, config, matches, **kw):
    return DRIVERS[driver][0](config, matches, **kw)


def _strip_section(envelope: Dict[str, Any], surface: str, config) -> Dict[str, Any]:
    """The envelope minus the section's keys and its clause."""
    out = {k: v for k, v in envelope.items() if k not in SECTION_KEYS}
    state = envelope.get("rejection_section")
    if state is not None:
        clause = rejection_section_clause(state, dropped=envelope.get("rejection_dropped"),
                                          surface=surface, config=config)
        assert f" {clause}" in out["note"], out["note"]
        out["note"] = out["note"].replace(f" {clause}", "", 1)
    return out


# --------------------------------------------------------------------------- #
# The pure core: fill_rejections over hand-built resolutions
# --------------------------------------------------------------------------- #

class _Item:
    def __init__(self, ordinal: int, text: str) -> None:
        self.ordinal = ordinal
        self.text = text


def _owner(tag: str, *, kind: str = "decision", state: str = "active") -> Dict[str, Any]:
    owner_id = (tag * 64)[:64]
    return {"id": owner_id, "slug": f"s-{tag}", "kind": kind, "computed_state": state,
            "rejected_paths": f"field-{tag}", "core_axiom": f"axiom {tag}", "scope": []}


class _Pure:
    """Owners, their items and a search answer, built one hit at a time."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.ambiguous: Dict[str, List[str]] = {}
        self.orphans: List[str] = []
        self.items: Dict[str, Dict[str, _Item]] = {}
        self.hits: List[RejectionHit] = []
        self.items_calls: List[Tuple[str, str]] = []

    def owner(self, node: Dict[str, Any], items: Dict[str, Tuple[int, str]]) -> "_Pure":
        self.nodes[hash_to_uuid(node["id"])] = node
        self.items[node["id"]] = {pid: _Item(o, t) for pid, (o, t) in items.items()}
        return self

    def hit(self, pid: str, owner_id: Optional[str], score: float) -> "_Pure":
        self.hits.append(RejectionHit(point_id=pid, owner_id=owner_id, score=score))
        return self

    def items_of(self, owner_id: str, field: str):
        self.items_calls.append((owner_id, field))
        return self.items.get(owner_id, {})

    def fill(self, *, bound: int = 3, fetch: int = 100, searched: bool = True):
        resolution = PointResolution(nodes=dict(self.nodes), orphans=list(self.orphans),
                                     ambiguous=dict(self.ambiguous))
        return fill_rejections(RejectionSearch(searched=searched, hits=list(self.hits)),
                               resolution, items_of=self.items_of, floor=FLOOR,
                               bound=bound, fetch=fetch)


def _a_owner(world: _Pure, n: int = 3) -> Dict[str, Any]:
    node = _owner("a")
    world.owner(node, {f"a{i}": (i, f"item {i}\n") for i in range(1, n + 1)})
    return node


def test_fill_the_floor_stops_the_walk() -> None:
    world = _Pure()
    a = _a_owner(world)
    world.hit("a1", a["id"], ABOVE).hit("a2", a["id"], BELOW).hit("a3", a["id"], ABOVE)
    walk = world.fill()
    assert [m.point_id for m in walk.matches] == ["a1"]
    assert walk.short is False and walk.dropped == {k: 0 for k in REJECTION_DROP_KEYS}


def test_fill_two_parts_of_one_item_collapse_to_one_match_at_the_best_score() -> None:
    world = _Pure()
    a = _owner("a")
    world.owner(a, {"p1": (1, "one long item  \n"), "p2": (1, "one long item  \n"),
                    "q": (2, "two")})
    world.hit("p2", a["id"], 0.95).hit("p1", a["id"], 0.9).hit("q", a["id"], 0.8)
    walk = world.fill()
    assert [(m.point_id, m.ordinal, m.score) for m in walk.matches] == [
        ("p2", 1, 0.95), ("q", 2, 0.8)]
    assert walk.matches[0].item == "one long item"  # the whole item, rstripped
    assert sum(walk.dropped.values()) == 0          # a collapse is not a drop


def test_fill_the_bound_counts_items_not_points() -> None:
    world = _Pure()
    a = _owner("a")
    world.owner(a, {"p1": (1, "x"), "p1b": (1, "x"), "p2": (2, "y"), "p3": (3, "z"),
                    "p4": (4, "w")})
    for pid, score in (("p1", .95), ("p1b", .94), ("p2", .9), ("p3", .85), ("p4", .8)):
        world.hit(pid, a["id"], score)
    walk = world.fill(bound=3, fetch=5)
    assert [m.ordinal for m in walk.matches] == [1, 2, 3]
    assert walk.short is False                      # the bound filled


def test_fill_every_drop_is_counted_by_its_reason() -> None:
    world = _Pure()
    a = _a_owner(world)
    oq = _owner("b", kind="open_question")
    world.owner(oq, {"b1": (1, "q")})
    mismatched = _owner("c")
    # A prefix match whose whole id differs from the payload's owner: an orphan.
    world.nodes[hash_to_uuid("c" * 64)] = dict(mismatched, id="c" * 32 + "d" * 32)
    world.ambiguous[hash_to_uuid("e" * 64)] = ["e" * 64, "e" * 32 + "f" * 32]
    (world.hit("gone", a["id"], .99)              # stale: the field no longer states it
          .hit("x1", None, .98)                   # ownerless
          .hit("x2", "f" * 64, .97)               # orphan: no node
          .hit("x3", "not-a-node-id", .96)        # orphan: not a node id, never raises
          .hit("x4", "c" * 64, .95)               # orphan: prefix match, other id
          .hit("x5", "e" * 64, .94)               # ambiguous
          .hit("b1", oq["id"], .93)               # non_decision
          .hit("a1", a["id"], .92))
    walk = world.fill(fetch=8)
    assert [m.point_id for m in walk.matches] == ["a1"]
    assert walk.dropped == {"stale": 1, "ownerless": 1, "orphan": 3, "non_decision": 1,
                            "ambiguous": 1}
    # The fetch came back full and its last hit is above the floor: short.
    assert walk.short is True and walk.returned == walk.fetch == 8


def test_fill_short_only_when_the_fetch_is_full_and_still_above_the_floor() -> None:
    def walk_of(n_hits: int, fetch: int, last: float):
        world = _Pure()
        a = _a_owner(world, n=1)
        world.hit("a1", a["id"], .95)
        for i in range(n_hits - 2):
            world.hit(f"stale{i}", a["id"], .9)
        world.hit("last", a["id"], last)
        return world.fill(fetch=fetch)

    assert walk_of(4, 4, JUST_ABOVE).short is True       # full and above: short
    assert walk_of(4, 5, JUST_ABOVE).short is False      # ran out: complete
    assert walk_of(4, 4, BELOW).short is False           # crossed the floor: complete


def test_fill_a_fetch_that_crosses_the_floor_after_drops_is_complete() -> None:
    world = _Pure()
    a = _a_owner(world, n=1)
    world.hit("s1", a["id"], .95).hit("s2", a["id"], .9).hit("lo", a["id"], BELOW)
    walk = world.fill(fetch=3)
    assert walk.matches == [] and walk.dropped["stale"] == 2 and walk.short is False


def test_fill_an_unsearched_answer_is_over_budget() -> None:
    walk = _Pure().fill(searched=False)
    assert walk.state == "over_budget" and walk.matches == [] and walk.short is False


def test_fill_item_counts_count_the_distinct_items_a_field_can_serve() -> None:
    """A byte-identical item collapses into its first occurrence (rejection_points'
    rule), so it is not a second item the reader has yet to see."""
    world = _Pure()
    a = _owner("a")
    world.owner(a, {"p1": (1, "same"), "p2": (2, "other")})  # three enumeration items
    world.hit("p1", a["id"], .9)
    assert world.fill().item_counts == {a["id"]: 2}


@pytest.mark.parametrize("hit", [
    RejectionHit(point_id=None, owner_id=None, score=.9),
    RejectionHit(point_id="p", owner_id=None, score=True),
    RejectionHit(point_id="p", owner_id=7, score=.9),
    {"point_id": "p", "score": .9},
])
def test_fill_a_malformed_hit_is_a_walk_contract_error(hit) -> None:
    with pytest.raises(WalkContractError):
        fill_rejections(RejectionSearch(searched=True, hits=[hit]),
                        PointResolution({}, [], {}), items_of=lambda o, f: {},
                        floor=FLOOR, bound=3, fetch=10)


# --------------------------------------------------------------------------- #
# The shell: walk_ranked's reads with a leg
# --------------------------------------------------------------------------- #

class _SpyGraph:
    """Forwards the three batched reads to a real store and records them."""

    def __init__(self, inner: GraphStore) -> None:
        self._inner = inner
        self.resolved: List[List[str]] = []
        self.stamped: List[List[str]] = []
        self.ended: List[List[str]] = []
        self.fault: Optional[str] = None

    def resolve_point_ids(self, point_ids):
        ids = list(point_ids)
        self.resolved.append(ids)
        return self._inner.resolve_point_ids(ids)

    def get_modifiers_map(self, node_ids):
        self.stamped.append(list(node_ids))
        if self.fault == "stamps":
            raise DatabaseError("planted")
        return self._inner.get_modifiers_map(node_ids)

    def get_live_ends(self, node_ids):
        self.ended.append(list(node_ids))
        return self._inner.get_live_ends(node_ids)


def _leg(index, **kw) -> RejectionLeg:
    return RejectionLeg(index=index, items_of=point_items, floor=FLOOR,
                        bound=REJECTION_SECTION_BOUND, **kw)


def test_walk_with_a_leg_one_search_one_resolve_holding_the_owners_one_stamp_read(ws) -> None:
    a = _commit(ws, "a")
    owner = _commit(ws, "owner", F_TWO, amends=None)
    _commit(ws, "amender", amends=["owner"])
    index = FakeRejectionIndex(present=True)
    _plant(index, owner, F_TWO, {1: ABOVE, 2: JUST_ABOVE})
    store = GraphStore(ws.db_path, read_only=True)
    vec = CannedVector([point_hit(a, .9)], store=store)
    graph = _SpyGraph(store)

    walk = walk_ranked(_query_vector(), vector_store=vec, graph=graph, limit=2,
                       rejections=_leg(index))

    assert [name for name, _ in index.calls] == ["search"]
    assert index.calls[0][1] == {"limit": REJECTION_OVERFETCH_BOUND,
                                 "budget_seconds": REJECTION_SEARCH_BUDGET_SECONDS}
    assert vec.queries == 1
    assert graph.resolved == [[hash_to_uuid(a), hash_to_uuid(owner)]]
    assert graph.stamped == [[a, owner]]
    assert graph.ended == []
    assert stamp_ids_of(walk) == [a, owner]
    assert walk.rejections.state == "searched"
    assert [m.ordinal for m in walk.rejections.matches] == [1, 2]
    assert walk.stamps[owner] == {"amended_by": ["amender"]}


def test_walk_a_window_hits_owner_resolves_once(ws) -> None:
    owner = _commit(ws, "owner", F_TWO)
    index = FakeRejectionIndex(present=True)
    _plant(index, owner, F_TWO, {1: ABOVE})
    store = GraphStore(ws.db_path, read_only=True)
    graph = _SpyGraph(store)
    walk_ranked(_query_vector(), vector_store=CannedVector([point_hit(owner, .9)],
                                                           store=store),
                graph=graph, limit=2, rejections=_leg(index))
    assert graph.resolved == [[hash_to_uuid(owner)]]
    assert graph.stamped == [[owner]]


def test_walk_the_live_end_read_covers_the_retired_owners(ws) -> None:
    old = _commit(ws, "old", F_TWO)
    _commit(ws, "mid", supersedes=["old"])
    _commit(ws, "new", supersedes=["mid"])
    index = FakeRejectionIndex(present=True)
    _plant(index, old, F_TWO, {1: ABOVE})
    store = GraphStore(ws.db_path, read_only=True)
    graph = _SpyGraph(store)
    walk = walk_ranked(_query_vector(), vector_store=CannedVector([], store=store),
                       graph=graph, limit=2, rejections=_leg(index))
    # The zero-window, zero-retired read the section exists for: one stamp read, one
    # live-end read, both over the section's owner.
    assert graph.stamped == [[old]] and graph.ended == [[old]]
    assert walk.live_ends == {old: ["new"]}
    assert walk.rejections.matches[0].live is False


def test_walk_owners_below_the_floor_never_join_the_batch(ws) -> None:
    a = _commit(ws, "a")
    owner = _commit(ws, "owner", F_TWO)
    index = FakeRejectionIndex(present=True)
    _plant(index, owner, F_TWO, {1: BELOW, 2: BELOW})
    index.points["bogus"] = {"vector": _at(ABOVE), "payload": {"owner_id": "Not-Hex"}}
    store = GraphStore(ws.db_path, read_only=True)
    graph = _SpyGraph(store)
    walk = walk_ranked(_query_vector(), vector_store=CannedVector([point_hit(a, .9)],
                                                                  store=store),
                       graph=graph, limit=2, rejections=_leg(index))
    assert graph.resolved == [[hash_to_uuid(a)]]   # the decision hits only
    assert walk.rejections.matches == []
    assert walk.rejections.dropped["orphan"] == 1  # the foreign owner, never formatted


@pytest.mark.parametrize("knob,state", [
    ("absent", "absent"), ("over_budget", "over_budget"), ("unreachable", "error"),
    ("refuse_search", "error"),
])
def test_walk_each_leg_failure_is_a_state_and_the_window_stands(ws, knob, state) -> None:
    a = _commit(ws, "a")
    index = FakeRejectionIndex(present=knob != "absent")
    if knob != "absent":
        setattr(index, knob, True)
    store = GraphStore(ws.db_path, read_only=True)
    with_leg = walk_ranked(_query_vector(), vector_store=CannedVector(
        [point_hit(a, .9)], store=store), graph=store, limit=2, rejections=_leg(index))
    without = walk_ranked(_query_vector(), vector_store=CannedVector(
        [point_hit(a, .9)], store=store), graph=store, limit=2)
    assert with_leg.rejections.state == state and with_leg.rejections.matches == []
    from dataclasses import replace
    assert replace(with_leg, rejections=None) == without


def test_walk_a_spent_budget_reads_as_over_budget(ws) -> None:
    index = FakeRejectionIndex(present=True)
    store = GraphStore(ws.db_path, read_only=True)
    walk = walk_ranked(_query_vector(), vector_store=CannedVector([], store=store),
                       graph=store, limit=2, rejections=_leg(index, budget_seconds=0.0))
    assert walk.rejections.state == "over_budget"


class _NoSearch:
    pass


class _BadAnswer(FakeRejectionIndex):
    """The shared double whose search answers a malformed value."""

    def __init__(self, answer):
        super().__init__(present=True)
        self.answer = answer

    def search(self, vector, limit, *, budget_seconds):
        return self.answer


@pytest.mark.parametrize("index", [
    _NoSearch(), _BadAnswer(None), _BadAnswer(RejectionSearch(searched="yes", hits=[])),
    _BadAnswer(RejectionSearch(searched=True, hits=None)),
    _BadAnswer(RejectionSearch(searched=True, hits=[RejectionHit("p", None, "x")])),
])
def test_walk_a_wrong_index_double_is_loud(ws, index) -> None:
    store = GraphStore(ws.db_path, read_only=True)
    with pytest.raises(WalkContractError):
        walk_ranked(_query_vector(), vector_store=CannedVector([], store=store),
                    graph=store, limit=2, rejections=_leg(index))


class _BrokenSearch(FakeRejectionIndex):
    """The shared double whose search fails with a defect, not an index error."""

    def search(self, vector, limit, *, budget_seconds):
        raise TypeError("planted: a defect inside the search")


def test_walk_a_search_defect_is_loud(ws) -> None:
    """The leg catches the index's own error classes only; a defect (here a
    TypeError) is never read as an unavailable index (P10)."""
    store = GraphStore(ws.db_path, read_only=True)
    with pytest.raises(TypeError):
        walk_ranked(_query_vector(), vector_store=CannedVector([], store=store),
                    graph=store, limit=2, rejections=_leg(_BrokenSearch(present=True)))


def test_walk_a_stamp_fault_costs_the_owners_their_stamps_too(ws) -> None:
    owner = _commit(ws, "owner", F_TWO)
    _commit(ws, "amender", amends=["owner"])
    index = FakeRejectionIndex(present=True)
    _plant(index, owner, F_TWO, {1: ABOVE})
    store = GraphStore(ws.db_path, read_only=True)
    graph = _SpyGraph(store)
    graph.fault = "stamps"
    walk = walk_ranked(_query_vector(), vector_store=CannedVector([], store=store),
                       graph=graph, limit=2, rejections=_leg(index))
    assert walk.stamps == {} and walk.stamps_unread is True
    assert len(walk.rejections.matches) == 1      # the section survives the fault


def test_walk_without_a_leg_stamp_ids_are_unchanged(ws) -> None:
    a = _commit(ws, "a")
    store = GraphStore(ws.db_path, read_only=True)
    walk = walk_ranked(_query_vector(), vector_store=CannedVector([point_hit(a, .9)],
                                                                  store=store),
                       graph=store, limit=2)
    assert walk.rejections is None and stamp_ids_of(walk) == [a]


# --------------------------------------------------------------------------- #
# T13: the three owner shapes, on every surface
# --------------------------------------------------------------------------- #

def _letter_keys(m: Dict[str, Any]) -> set:
    return {"axiom", "scope", "rejected_paths"} & set(m)


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_t13_a_live_owner_outside_the_window_carries_its_letter_once(
        ws, monkeypatch, driver) -> None:
    a = _commit(ws, "a")
    owner = _commit(ws, "owner", F_TWO)
    index = FakeRejectionIndex(present=True)
    _plant(index, owner, F_TWO, {1: ABOVE, 2: JUST_ABOVE})
    _bind(monkeypatch, index)
    env = _call(driver, ws, [{"slug": "a", "score": .9}])
    matches = env["rejection_matches"]
    assert [m["letter"] for m in matches] == ["here", "section"]
    here, later = matches
    assert here["slug"] == "owner" and here["id"] == owner and here["state"] == "active"
    assert (here["ordinal"], here["item"]) == (1, "1. Keep the old — it drifted.")
    assert here["axiom"] == "The owner axiom stands on its own words."
    assert here["rejected_paths"] == F_TWO and here["scope"] == ["core"]
    assert _letter_keys(later) == set()
    assert (later["ordinal"], later["item"]) == (2, "2. Buy a tool — it locks us in.")
    assert "rejection_section" not in env and "rejection_dropped" not in env
    # Placed after the decision answer's own keys, after the retired partition.
    keys = list(env)
    assert keys.index("rejection_matches") > keys.index("note")


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_t13_a_live_owner_in_the_window_is_pointed_at(ws, monkeypatch, driver) -> None:
    _commit(ws, "a")
    owner = _commit(ws, "owner", F_TWO)
    _commit(ws, "amender", amends=["owner"])
    index = FakeRejectionIndex(present=True)
    _plant(index, owner, F_TWO, {2: ABOVE})
    _bind(monkeypatch, index)
    env = _call(driver, ws, [{"slug": "a", "score": .9}, {"slug": "owner", "score": .8}])
    (m,) = env["rejection_matches"]
    assert m["letter"] == "window" and m["rank"] == 2
    assert _letter_keys(m) == set()
    assert m["amended_by"] == ["amender"]          # the one stamp map, stamped on
    assert m["item"] == "2. Buy a tool — it locks us in."


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_t13_a_retired_owner_arrives_as_its_handle(ws, monkeypatch, driver) -> None:
    _commit(ws, "a")
    old = _commit(ws, "old", F_TWO)
    _commit(ws, "mid", supersedes=["old"])
    _commit(ws, "new", supersedes=["mid"])
    index = FakeRejectionIndex(present=True)
    _plant(index, old, F_TWO, {1: ABOVE})
    _bind(monkeypatch, index)
    env = _call(driver, ws, [{"slug": "a", "score": .9}])
    (m,) = env["rejection_matches"]
    assert m["letter"] == "retired" and m["state"] == "superseded"
    assert m["superseded_by"] == ["mid"] and m["live_end"] == ["new"]
    assert _letter_keys(m) == set()


def _depth_kw(depth: str) -> Dict[str, Any]:
    return {"brief": {"brief": True}, "below": {"full_top": 1}, "at": {"full_top": 2},
            "whole": {}}[depth]


@pytest.mark.parametrize("depth,whole", [("brief", False), ("below", False),
                                         ("at", True), ("whole", True)])
@pytest.mark.parametrize("driver", list(DRIVERS))
def test_t13_the_owner_follows_the_depth_of_the_windows_last_rank(
        ws, monkeypatch, driver, depth, whole) -> None:
    _commit(ws, "a")
    _commit(ws, "b")
    owner = _commit(ws, "owner", F_TWO)
    index = FakeRejectionIndex(present=True)
    _plant(index, owner, F_TWO, {1: ABOVE})
    _bind(monkeypatch, index)
    env = _call(driver, ws, [{"slug": "a", "score": .9}, {"slug": "b", "score": .85}],
                **_depth_kw(depth))
    (m,) = env["rejection_matches"]
    assert m["letter"] == "here" and m["item"] == "1. Keep the old — it drifted."
    assert ("rejected_paths" in m) is whole
    assert "axiom" in m


def test_rejection_owners_whole_reads_the_last_rank() -> None:
    assert rejection_owners_whole(None, 3) is True
    assert rejection_owners_whole(0, 3) is False
    assert rejection_owners_whole(2, 3) is False
    assert rejection_owners_whole(3, 3) is True
    assert rejection_owners_whole(1, 0) is True    # a blackout: only brief withholds
    assert rejection_owners_whole(0, 0) is False


# --------------------------------------------------------------------------- #
# The withheld rule, over decisions
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", list(DRIVERS))
def test_withheld_a_thinned_window_hit_whose_one_item_matched_is_not_counted(
        ws, monkeypatch, driver) -> None:
    _commit(ws, "a")
    owner = _commit(ws, "owner", F_ONE)
    index = FakeRejectionIndex(present=True)
    _plant(index, owner, F_ONE, {1: ABOVE})
    _bind(monkeypatch, index)
    env = _call(driver, ws, [{"slug": "a", "score": .9}, {"slug": "owner", "score": .8}],
                full_top=1)
    assert env["rejection_matches"][0]["letter"] == "window"
    assert "rejected_paths_withheld" not in env
    assert "omit" not in env["note"]


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_withheld_a_here_owner_with_an_unshown_item_is_counted(
        ws, monkeypatch, driver) -> None:
    _commit(ws, "a")
    owner = _commit(ws, "owner", F_TWO)
    index = FakeRejectionIndex(present=True)
    _plant(index, owner, F_TWO, {1: ABOVE})
    _bind(monkeypatch, index)
    env = _call(driver, ws, [{"slug": "a", "score": .9}], brief=True)
    assert env["rejection_matches"][0]["letter"] == "here"
    # The window hit plus the section's owner, each a decision whose reasoning is out.
    assert env["rejected_paths_withheld"] == 2
    surface = DRIVERS[driver][2]
    assert env["note"].endswith(" " + withheld_clause(2, surface=surface, config=ws))


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_withheld_a_here_owner_whose_every_item_is_shown_is_not_counted(
        ws, monkeypatch, driver) -> None:
    _commit(ws, "a")
    owner = _commit(ws, "owner", F_TWO)
    index = FakeRejectionIndex(present=True)
    _plant(index, owner, F_TWO, {1: ABOVE, 2: JUST_ABOVE})
    _bind(monkeypatch, index)
    env = _call(driver, ws, [{"slug": "a", "score": .9}], brief=True)
    assert [m["letter"] for m in env["rejection_matches"]] == ["here", "section"]
    assert env["rejected_paths_withheld"] == 1          # the window hit alone


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_withheld_with_no_matches_the_decision_count_equals_the_hit_count(
        ws, monkeypatch, driver) -> None:
    for slug in ("a", "b", "c"):
        _commit(ws, slug)
    env = _call(driver, ws, [{"slug": s, "score": .9 - i / 20}
                             for i, s in enumerate(("a", "b", "c"))], full_top=1)
    key = DRIVERS[driver][1]
    assert "rejection_matches" not in env
    assert env["rejected_paths_withheld"] == count_withheld(env[key]) == 2


def test_count_withheld_decisions_dedupes_and_exempts_the_covered() -> None:
    assert count_withheld_decisions(thinned_ids=["a", "b", "a", "c"],
                                    shown_ordinals={"a": {1, 2}, "c": {1}},
                                    item_counts={"a": 2, "c": 2}) == 2  # b and c
    assert count_withheld_decisions(thinned_ids=[], shown_ordinals={},
                                    item_counts={}) == 0


# --------------------------------------------------------------------------- #
# Stale: never served (T11's read-time half)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", list(DRIVERS))
def test_a_stale_item_is_never_served_and_the_ordinal_is_recomputed(
        ws, monkeypatch, driver) -> None:
    owner = _commit(ws, "owner", F_ABC)
    _plant(index := FakeRejectionIndex(present=True), owner, F_ABC,
           {1: .99, 2: .9, 3: .95})
    _commit(ws, "owner", F_BD)  # Alpha and Charlie are gone; Bravo is now item 1
    _bind(monkeypatch, index)
    env = _call(driver, ws, [])
    (m,) = env["rejection_matches"]
    assert (m["ordinal"], m["item"]) == (1, "1. Bravo — b.")
    # The fetch ran out (3 < the bound), so the drops cost nothing: not short.
    assert "rejection_section" not in env and "rejection_dropped" not in env


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_a_stale_drop_on_a_full_fetch_says_short_with_the_count(
        ws, monkeypatch, driver) -> None:
    owner = _commit(ws, "owner", F_ABC)
    index = FakeRejectionIndex(present=True)
    _plant(index, owner, F_ABC, {1: .99, 2: .9, 3: .95})
    _commit(ws, "owner", F_BD)
    _bind_leg(monkeypatch, index, fetch=3)
    env = _call(driver, ws, [])
    assert len(env["rejection_matches"]) == 1
    assert env["rejection_section"] == "short" and env["rejection_dropped"] == 2
    surface = DRIVERS[driver][2]
    clause = rejection_section_clause("short", dropped=2, surface=surface, config=ws)
    assert f" {clause}" in env["note"] and "2 rejection matches" in clause


def test_a_write_then_read_serves_no_item_an_amend_removed(tmp_path, monkeypatch) -> None:
    """One index at both seams: the record's and the amend's inline writes land in the
    index the read searches, and the amend's old item (which the inline write never
    removes) stays in it, unserved."""
    from test_rejection_inline import (Rig, RecordingProvider, _bind as bind_write,
                                       _mcp_amend, _mcp_record, _vector_store_class)
    from test_corpus_provenance import _init_workspace

    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:9")
    monkeypatch.chdir(tmp_path)
    root = _init_workspace(tmp_path / "ws")
    rig = Rig(root, RecordingProvider(), FakeRejectionIndex(), _vector_store_class())
    bind_write(monkeypatch, rig)
    _bind(monkeypatch, rig.index)
    owner = _mcp_record(rig, "owner", F_ABC)["id"]
    _mcp_amend(rig, "owner", rejected_paths=F_BD)
    assert set(point_items(owner, F_ABC)) <= set(rig.index.points)  # nothing removed
    for point in rig.index.points.values():
        point["vector"] = _at(ABOVE)
    calls_before = len(rig.index.calls)
    env = _mcp_query(rig.config, [])
    assert rig.index.calls[calls_before][0] == "search"   # the SAME instance searched
    items = sorted(m["item"] for m in env["rejection_matches"])
    assert items == ["1. Bravo — b.", "2. Delta — d."]


# --------------------------------------------------------------------------- #
# The gap, and the other not-computed states
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", list(DRIVERS))
def test_an_absent_index_over_stated_items_is_a_gap(ws, monkeypatch, driver) -> None:
    _commit(ws, "a")
    _bind(monkeypatch, FakeRejectionIndex(present=False))
    env = _call(driver, ws, [{"slug": "a", "score": .9}])
    surface = DRIVERS[driver][2]
    assert env["rejection_section"] == "gap" and "rejection_matches" not in env
    clause = rejection_section_clause("gap", dropped=None, surface=surface, config=ws)
    assert env["note"].endswith(" " + clause)
    if surface == "cli":
        assert f"`mitos reconcile {shell_flag('-p', ws.project)}`" in clause
    else:
        assert "mitos" not in clause and "a person with a shell" in clause
        tool_names = [name for name in dir(mcp_server)
                      if callable(getattr(mcp_server, name)) and not name.startswith("_")]
        for name in ("reconcile", "rebuild", "sync", *tool_names):
            if len(name) > 4:
                assert f"{name}(" not in clause, name
        assert "reconcile" not in clause


@pytest.mark.parametrize("graph", ["questions-only", "empty"])
@pytest.mark.parametrize("driver", list(DRIVERS))
def test_an_absent_index_over_no_stated_item_adds_nothing(
        ws, monkeypatch, driver, graph) -> None:
    if graph == "questions-only":
        _commit_oq(ws, "oq")
    _bind(monkeypatch, FakeRejectionIndex(present=False))
    absent = _call(driver, ws, [])
    _bind(monkeypatch, FakeRejectionIndex(present=True))
    present = _call(driver, ws, [])
    assert absent == present
    assert not set(SECTION_KEYS) & set(absent)


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_an_index_deleted_mid_session_states_the_gap_on_the_next_read(
        ws, monkeypatch, driver) -> None:
    owner = _commit(ws, "owner", F_TWO)
    index = FakeRejectionIndex(present=True)
    _plant(index, owner, F_TWO, {1: ABOVE})
    _bind(monkeypatch, index)
    first = _call(driver, ws, [])
    assert len(first["rejection_matches"]) == 1
    index.present = False
    index.points.clear()
    second = _call(driver, ws, [])
    assert second["rejection_section"] == "gap" and "rejection_matches" not in second


@pytest.mark.parametrize("knob,state", [("unreachable", "unavailable"),
                                        ("refuse_search", "unavailable"),
                                        ("over_budget", "not_searched")])
@pytest.mark.parametrize("driver", list(DRIVERS))
def test_an_erroring_or_slow_index_is_named_and_the_decision_answer_is_whole(
        ws, monkeypatch, driver, knob, state) -> None:
    _commit(ws, "a")
    _commit(ws, "b")
    owner = _commit(ws, "owner", F_TWO)
    window = [{"slug": "a", "score": .9}, {"slug": "b", "score": .85}]
    healthy_index = FakeRejectionIndex(present=True)
    _bind(monkeypatch, healthy_index)
    healthy = _call(driver, ws, window, full_top=1)
    failing = FakeRejectionIndex(present=True)
    _plant(failing, owner, F_TWO, {1: ABOVE})
    setattr(failing, knob, True)
    _bind(monkeypatch, failing)
    env = _call(driver, ws, window, full_top=1)
    surface = DRIVERS[driver][2]
    assert env["rejection_section"] == state and "rejection_matches" not in env
    assert _strip_section(env, surface, ws) == healthy
    # The note order: band note, the section's clause, then the withheld clause.
    clause = rejection_section_clause(state, dropped=None, surface=surface, config=ws)
    assert env["note"].endswith(
        f" {clause} {withheld_clause(1, surface=surface, config=ws)}")


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_a_seam_that_built_no_index_is_unavailable(ws, monkeypatch, driver) -> None:
    _commit(ws, "a")
    _bind(monkeypatch, None)
    env = _call(driver, ws, [{"slug": "a", "score": .9}])
    assert env["rejection_section"] == "unavailable"


def test_the_seam_constructs_never_connects_and_answers_none_on_a_fault(ws) -> None:
    """The production seam, unpatched: its source re-executed in its module's
    namespace (the autouse replaced the module attribute)."""
    for module in (cli, mcp_server):
        real = _original_seam(module)
        index = real(ws)
        assert type(index).__name__ == "QdrantRejectionIndex"
        assert index.collection == ws.rejection_collection

        class _Broken:
            @property
            def qdrant_url(self):
                raise RuntimeError("planted")
        assert real(_Broken()) is None


def _original_seam(module):
    """The module's own ``_ranked_rejection_index``, before the autouse's patch."""
    source = inspect.getsource(module)
    tree = ast.parse(source)
    (fn,) = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name == "_ranked_rejection_index"]
    namespace = dict(vars(module))
    # The autouse refuses the class name at `mitos.cli` (the status seam); the read
    # seam must still construct the real client.
    from mitos.vector_store import QdrantRejectionIndex
    namespace["QdrantRejectionIndex"] = QdrantRejectionIndex
    exec(compile(ast.Module(body=[fn], type_ignores=[]), module.__file__, "exec"),
         namespace)
    return namespace["_ranked_rejection_index"]


def test_the_state_vocabulary_maps_one_to_one() -> None:
    class W:
        def __init__(self, state, short=False):
            self.state, self.short = state, short
    asked = []

    def gap():
        asked.append(1)
        return True
    assert rejection_section_state(W("searched"), gap=gap) is None
    assert rejection_section_state(W("over_budget"), gap=gap) == "not_searched"
    assert rejection_section_state(W("error"), gap=gap) == "unavailable"
    assert asked == []                                   # the gap is asked only on absent
    assert rejection_section_state(W("absent"), gap=gap) == "gap"
    assert rejection_section_state(W("absent"), gap=lambda: False) is None
    assert rejection_section_state(W("searched", short=True), gap=gap) == "short"
    assert rejection_section_state(None, gap=gap) == "unavailable"
    with pytest.raises(ValueError):
        rejection_section_state(W("bogus"), gap=gap)


def test_the_leg_states_agree_with_recall() -> None:
    import mitos.recall as recall
    assert (recall._LEG_ABSENT, recall._LEG_OVER_BUDGET, recall._LEG_ERROR,
            recall._LEG_SEARCHED) == (walk_module.REJECTIONS_ABSENT,
                                      walk_module.REJECTIONS_OVER_BUDGET,
                                      walk_module.REJECTIONS_ERROR,
                                      walk_module.REJECTIONS_SEARCHED)


def test_the_owner_id_predicate_agrees_with_rejections() -> None:
    import mitos.rejections as rejections
    assert walk_module._OWNER_ID_RE.pattern == rejections._OWNER_ID_RE.pattern


# --------------------------------------------------------------------------- #
# Constraint 4: writes meanwhile
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arm", ["absent", "unreachable", "over_budget"])
def test_writes_land_and_check_is_unchanged_while_the_read_index_fails(
        tmp_path, monkeypatch, arm) -> None:
    from mitos.check import probe_stale_index
    from test_rejection_inline import (Rig, RecordingProvider, _bind as bind_write,
                                       _mcp_amend, _mcp_record, _vector_store_class)
    from test_corpus_provenance import _init_workspace

    outcomes = {}
    for side in ("healthy", arm):
        monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:9")
        root = _init_workspace(tmp_path / side)
        index = FakeRejectionIndex(present=side != "absent")
        if side not in ("healthy", "absent"):
            setattr(index, side, True)
        rig = Rig(root, RecordingProvider(), index, _vector_store_class())
        bind_write(monkeypatch, rig)
        _bind(monkeypatch, index)
        # A decision whose items the read would search for, committed store-side.
        entry = ParsedEntry("decision", "seed", 1, 5)
        entry.axiom = "The seed axiom stands on its own words."
        entry.rejected_paths = F_ONE
        rig.store().commit_parsed_entry(entry)
        read = _mcp_query(rig.config, [])
        receipt = _mcp_record(rig, "first", F_TWO)
        amended = _mcp_amend(rig, "first", rejected_paths=F_ABC)
        store = rig.store()
        outcomes[side] = {
            "receipt": {k: receipt.get(k) for k in ("status", "embedding", "audit_debt")},
            "amended": amended["status"],
            "outbox": sorted(row["node_id"] for row in store.get_pending_embeddings()),
            # The probe's rows carry their enqueue time; everything else must agree.
            "probe": re.sub(r"queued_at='[^']*'", "", repr(probe_stale_index(store))),
            "matches": read.get("matches"),
        }
        expected = {"healthy": None, "absent": "gap", "unreachable": "unavailable",
                    "over_budget": "not_searched"}[side]
        assert read.get("rejection_section") == expected
    assert outcomes["healthy"]["receipt"] == outcomes[arm]["receipt"]
    assert outcomes["healthy"]["amended"] == outcomes[arm]["amended"] == "amended"
    assert outcomes["healthy"]["probe"] == outcomes[arm]["probe"]
    assert outcomes["healthy"]["outbox"] == outcomes[arm]["outbox"]
    assert outcomes["healthy"]["matches"] == outcomes[arm]["matches"]


# --------------------------------------------------------------------------- #
# The other envelopes
# --------------------------------------------------------------------------- #

def test_an_exact_slug_hit_composes_no_section(ws, monkeypatch) -> None:
    owner = _commit(ws, "owner", F_TWO)
    index = FakeRejectionIndex(present=False)  # a gap the ranked branch would state
    _plant(index, owner, F_TWO, {1: ABOVE})
    _bind(monkeypatch, index)
    env = _mcp_query(ws, [], query="owner")
    assert env["slug"] == "owner"
    assert not set(SECTION_KEYS) & set(env)
    assert index.calls == []


class _MissingCollection:
    def query(self, vector, limit=5, *, with_payload=True):
        raise CollectionMissingError("absent", collection="c")

    def delete_points(self, point_ids):
        raise CollectionMissingError("absent", collection="c")


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_the_healthy_empty_arm_composes_no_section(ws, monkeypatch, driver) -> None:
    _bind(monkeypatch, FakeRejectionIndex(present=False))
    env = _call(driver, ws, _MissingCollection())
    assert "degraded" not in env
    assert not set(SECTION_KEYS) & set(env)


def _blackout(ws) -> str:
    _commit(ws, "old")
    _commit(ws, "new", supersedes=["old"])
    owner = _commit(ws, "owner", F_TWO)
    return owner


@pytest.mark.parametrize("brief", [False, True])
@pytest.mark.parametrize("driver", list(DRIVERS))
def test_a_blackout_carries_the_section_beside_all_superseded(
        ws, monkeypatch, driver, brief) -> None:
    owner = _blackout(ws)
    window = [{"slug": "old", "score": .9}]
    _bind(monkeypatch, FakeRejectionIndex(present=True))
    bare = _call(driver, ws, window, brief=brief)
    index = FakeRejectionIndex(present=True)
    _plant(index, owner, F_TWO, {1: ABOVE})
    _bind(monkeypatch, index)
    env = _call(driver, ws, window, brief=brief)
    surface = DRIVERS[driver][2]
    assert env["all_superseded"] == bare["all_superseded"]
    assert env["rejection_matches"][0]["letter"] == "here"
    if brief:
        # The withheld clause survives the blackout override, after it.
        assert env["rejected_paths_withheld"] == 1
        assert env["note"] == (
            f"{bare['note']} {withheld_clause(1, surface=surface, config=ws)}")
    else:
        assert "rejected_paths_withheld" not in env and env["note"] == bare["note"]


@pytest.mark.parametrize("brief", [False, True])
@pytest.mark.parametrize("driver", list(DRIVERS))
def test_a_miss_whose_query_names_a_rejected_option_carries_the_section(
        ws, monkeypatch, driver, brief) -> None:
    owner = _commit(ws, "owner", F_TWO)
    index = FakeRejectionIndex(present=True)
    _plant(index, owner, F_TWO, {1: ABOVE})
    _bind(monkeypatch, index)
    env = _call(driver, ws, [], brief=brief)
    key = DRIVERS[driver][1]
    assert env[key] == [] and env["rejection_matches"][0]["letter"] == "here"
    assert ("rejected_paths" in env["rejection_matches"][0]) is not brief
    assert env.get("rejected_paths_withheld") == (1 if brief else None)


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_an_unbuilt_graph_override_keeps_the_tail(ws, monkeypatch, driver) -> None:
    """The unbuilt-graph note overrides the band note, and the section's clause still
    follows it: the tail is composed after every override."""
    from mitos.recall import missing_graph_note
    for module in (cli, mcp_server):
        monkeypatch.setattr(module, "corpus_holds_entries", lambda config: True)
    _bind(monkeypatch, None)
    env = _call(driver, ws, [])
    surface = DRIVERS[driver][2]
    clause = rejection_section_clause("unavailable", dropped=None, surface=surface,
                                      config=ws)
    assert env["rejection_section"] == "unavailable"
    assert env["note"] == f"{missing_graph_note(surface, ws)} {clause}"


# --------------------------------------------------------------------------- #
# CLI text: every ranked exit prints the section
# --------------------------------------------------------------------------- #

def _text_lines(out: str) -> List[str]:
    return out.splitlines()


@pytest.mark.parametrize("exit_", ["miss", "blackout", "populated"])
@pytest.mark.parametrize("verb", list(TEXT_DRIVERS))
def test_text_every_ranked_exit_prints_the_section(ws, monkeypatch, verb, exit_) -> None:
    if exit_ == "blackout":
        owner = _blackout(ws)
        window = [{"slug": "old", "score": .9}]
    else:
        _commit(ws, "a")
        owner = _commit(ws, "owner", F_TWO)
        window = [] if exit_ == "miss" else [{"slug": "a", "score": .9}]
    index = FakeRejectionIndex(present=True)
    _plant(index, owner, F_TWO, {1: ABOVE, 2: JUST_ABOVE})
    _bind(monkeypatch, index)
    out = TEXT_DRIVERS[verb](ws, window)
    assert "Rejected options matching this query (closest first):" in out
    assert f"owner  (active, score {ABOVE:.3f})  item 1: 1. Keep the old — it drifted." in out
    assert "item 2: 2. Buy a tool — it locks us in." in out
    assert "    Decided:  The owner axiom stands on its own words." in out
    assert f"    Rejected: {F_TWO.splitlines()[0]}" in out


@pytest.mark.parametrize("verb", list(TEXT_DRIVERS))
def test_text_the_state_clause_rides_the_note(ws, monkeypatch, verb) -> None:
    _commit(ws, "a")
    _bind(monkeypatch, FakeRejectionIndex(present=False))
    out = TEXT_DRIVERS[verb](ws, [{"slug": "a", "score": .9}])
    clause = rejection_section_clause("gap", dropped=None, surface="cli", config=ws)
    assert any(line.startswith("→") and clause in line for line in _text_lines(out)), out


@pytest.mark.parametrize("verb", list(TEXT_DRIVERS))
def test_text_window_and_retired_owners_print_their_pointers(ws, monkeypatch, verb) -> None:
    _commit(ws, "a")
    old = _commit(ws, "old", F_ONE)
    _commit(ws, "mid", supersedes=["old"])
    _commit(ws, "new", supersedes=["mid"])
    a_owner = _commit(ws, "a", F_TWO)
    index = FakeRejectionIndex(present=True)
    _plant(index, a_owner, F_TWO, {1: ABOVE})
    _plant(index, old, F_ONE, {1: JUST_ABOVE})
    _bind(monkeypatch, index)
    out = TEXT_DRIVERS[verb](ws, [{"slug": "a", "score": .9}])
    assert "    see rank 1 above" in out
    assert "    superseded_by: mid live_end: new" in out


# --------------------------------------------------------------------------- #
# Parity: CLI --json against MCP
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("verb", ["surface", "query"])
@pytest.mark.parametrize("kw", [{}, {"brief": True}, {"full_top": 1}])
def test_cli_json_and_mcp_agree_on_the_section(ws, monkeypatch, verb, kw) -> None:
    _commit(ws, "a")
    owner = _commit(ws, "owner", F_TWO)
    old = _commit(ws, "old", F_ONE)
    _commit(ws, "new", supersedes=["old"])
    index = FakeRejectionIndex(present=True)
    _plant(index, owner, F_TWO, {1: ABOVE, 2: JUST_ABOVE})
    _plant(index, old, F_ONE, {1: JUST_ABOVE - 0.01})
    _bind(monkeypatch, index)
    window = [{"slug": "a", "score": .9}]
    mcp = (_mcp_surface if verb == "surface" else _mcp_query)(ws, window, **kw)
    cli_env = (_cli_surface if verb == "surface" else _cli_query)(ws, window, **kw)
    assert set(mcp) == set(cli_env)
    for key in mcp:
        if key in ("note", "project", "collection", "workspace"):
            continue
        assert mcp[key] == cli_env[key], key


# --------------------------------------------------------------------------- #
# Structural and pin rows
# --------------------------------------------------------------------------- #

MITOS_DIR = os.path.dirname(os.path.abspath(cli.__file__))


def _walk_calls(filename: str):
    """Every ``walk_ranked(...)`` call in a mitos module, with its enclosing function."""
    path = os.path.join(MITOS_DIR, filename)
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = []
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "walk_ranked"):
                    out.append((fn.name, {kw.arg for kw in node.keywords}))
    return out


def test_the_gather_passes_no_leg() -> None:
    calls = _walk_calls("conflict.py")
    assert calls, "the sweep found no walk_ranked call in conflict.py"
    assert all("rejections" not in kws for _fn, kws in calls)


def test_every_ranked_walk_call_passes_the_leg() -> None:
    found = {}
    for filename in ("mcp_server.py", "cli.py"):
        for fn, kws in _walk_calls(filename):
            found[(filename, fn)] = "rejections" in kws
    assert found == {("mcp_server.py", "surface_decisions"): True,
                     ("mcp_server.py", "query_decisions"): True,
                     ("cli.py", "cmd_surface"): True,
                     ("cli.py", "cmd_query"): True}


def test_the_rejection_search_is_called_only_by_the_walk() -> None:
    """The twin of the one-vector-call row: an index ``.search(`` call in ``mitos/``
    lives in ``retrieval_walk.walk_ranked`` alone (``re`` and compiled patterns'
    ``.search`` are not index calls)."""
    sites = set()
    for filename in sorted(os.listdir(MITOS_DIR)):
        if not filename.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(MITOS_DIR, filename), encoding="utf-8").read())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "search"
                        and any(kw.arg == "budget_seconds" for kw in node.keywords)):
                    sites.add((filename, fn.name))
    assert sites == {("retrieval_walk.py", "walk_ranked")}


def test_the_fetch_bound_exceeds_the_limit_ceiling() -> None:
    assert REJECTION_OVERFETCH_BOUND > RANKED_LIMIT_CEILING


def test_the_floor_is_pending_5b() -> None:
    from golden.test_calibrated_constants import CALIBRATED_CONSTANTS, PENDING
    assert PENDING["mitos.recall.REJECTION_MATCH_FLOOR"] == "5b"
    assert "mitos.recall.REJECTION_MATCH_FLOOR" not in CALIBRATED_CONSTANTS


def test_the_offline_autouse_binds_a_fake_at_both_read_seams(ws) -> None:
    """P26's guard: without the autouse an offline ranked read sends a real search to
    whatever QDRANT_URL names."""
    for module in (cli, mcp_server):
        index = module._ranked_rejection_index(ws)
        assert isinstance(index, FakeRejectionIndex)
        assert index.present is True and index.points == {}


# --------------------------------------------------------------------------- #
# T11 rollback, through cli.main: status names the stale items, no read serves them
# --------------------------------------------------------------------------- #

def test_t11_rollback_status_names_the_stale_items_and_no_ranked_read_serves_them(
        tmp_path, monkeypatch, capsys) -> None:
    """The previous build amended a field without touching the index: its old items
    stand in it. ``status`` names them stale (4d); every ranked read checks each
    match against the field as it is now, so none of them is shown."""
    from _stdio_fake_providers import fake_vector
    from test_cli_selector import _run
    from test_rejection_covering import _rig as covering_rig, _seat, _status
    from test_rejection_drain import _commit as drain_commit, _hold

    d = covering_rig(tmp_path, monkeypatch, present=True)
    owner = _seat(d, "one", F_ABC)
    _hold(d, owner, F_ABC)
    drain_commit(d, "one", F_BD)        # the previous build's amend: no row, no write
    _seat(d, "one", F_BD)

    _code, report = _status(d, capsys)
    stale = set(point_items(owner, F_ABC)) - set(point_items(owner, F_BD))
    assert report["checks"]["stale_rejection_items"] == len(stale) == 2

    query = "which option did we turn away"
    q = fake_vector(query)
    d.vector_cls._collections.setdefault(d.config.qdrant_collection, {})[owner] = (
        q, {"slug": "one", "state": "active"})
    for point in d.index.points.values():
        point["vector"] = list(q)       # every held item, stale ones included, at 1.0
    _bind(monkeypatch, d.index)
    for verb in ("surface", "query"):
        capsys.readouterr()
        _run(["-p", d.root, verb, query, "--json"])
        env = json.loads(capsys.readouterr().out)
        assert [(m["ordinal"], m["item"]) for m in env["rejection_matches"]] == [
            (1, "1. Bravo — b.")], env
        assert "rejection_section" not in env      # the fetch ran out: complete


# --------------------------------------------------------------------------- #
# The light resolver (the trigger 5a fired): light walk + one hydrate of the shown
# --------------------------------------------------------------------------- #

class _LightSpy(_SpyGraph):
    def __init__(self, inner: GraphStore) -> None:
        super().__init__(inner)
        self.light: List[List[str]] = []
        self.drop_on_hydrate: Optional[str] = None

    def resolve_point_states(self, point_ids):
        ids = list(point_ids)
        self.light.append(ids)
        return self._inner.resolve_point_states(ids)

    def resolve_point_ids(self, point_ids):
        res = super().resolve_point_ids(point_ids)
        if self.drop_on_hydrate is not None:
            res.nodes.pop(self.drop_on_hydrate, None)
        return res


def test_resolve_point_states_is_the_whole_resolves_partition_with_light_nodes(ws) -> None:
    a = _commit(ws, "a")
    old = _commit(ws, "old")
    _commit(ws, "new", supersedes=["old"])
    oq = _commit_oq(ws, "oq")
    store = GraphStore(ws.db_path, read_only=True)
    pids = [hash_to_uuid(a), hash_to_uuid(old), hash_to_uuid(oq), "not-a-point",
            hash_to_uuid("f" * 64)]
    whole = store.resolve_point_ids(pids)
    light = store.resolve_point_states(pids)
    assert light.orphans == whole.orphans and light.ambiguous == whole.ambiguous
    assert set(light.nodes) == set(whole.nodes)
    for pid, node in light.nodes.items():
        assert set(node) == {"id", "kind", "computed_state"}
        assert {k: whole.nodes[pid][k] for k in node} == node
    assert light.nodes[hash_to_uuid(old)]["computed_state"] == "superseded"
    assert store.resolve_point_states([]) == PointResolution({}, [], {})


def test_a_light_walk_resolves_light_then_hydrates_only_what_it_shows(ws) -> None:
    ids = [_commit(ws, f"d{i}") for i in range(6)]
    old = _commit(ws, "old", F_TWO)
    _commit(ws, "new", supersedes=["old"])
    owner = _commit(ws, "owner", F_TWO)
    index = FakeRejectionIndex(present=True)
    _plant(index, owner, F_TWO, {1: ABOVE})
    store = GraphStore(ws.db_path, read_only=True)
    hits = ([point_hit(old, .95)] + [point_hit(i, .9 - n / 50) for n, i in enumerate(ids)])
    graph = _LightSpy(store)
    walk = walk_ranked(_query_vector(), vector_store=CannedVector(hits, store=store),
                       graph=graph, limit=2, rejections=_leg(index), light=True)
    every = [h["point_id"] for h in hits] + [hash_to_uuid(owner)]
    assert graph.light == [every]
    # One hydrate: window (d0, d1), edge (d2), passed[:limit] (old), the owner.
    assert graph.resolved == [[hash_to_uuid(i) for i in (ids[0], ids[1], ids[2], old, owner)]]
    assert graph.stamped == [[ids[0], ids[1], old, owner]]
    for hit in walk.window + [walk.edge] + walk.passed:
        assert "core_axiom" in hit.node and "scope" in hit.node
    assert walk.rejections.matches[0].owner["rejected_paths"] == F_TWO
    # The same walk, whole: the same answer.
    whole = walk_ranked(_query_vector(), vector_store=CannedVector(hits, store=store),
                        graph=store, limit=2, rejections=_leg(index))
    for got, want in ((walk.window, whole.window), (walk.passed, whole.passed),
                      ([walk.edge], [whole.edge])):
        assert [h.node for h in got] == [h.node for h in want]
    assert walk.rejections == whole.rejections and walk.stamps == whole.stamps
    assert walk.live_ends == whole.live_ends and walk.end == whole.end


def test_a_light_walk_without_a_shown_hit_hydrates_nothing_it_does_not_show(ws) -> None:
    old = _commit(ws, "old")
    _commit(ws, "new", supersedes=["old"])
    retired = [_commit(ws, f"r{i}") for i in range(4)]
    for i in range(4):
        _commit(ws, f"k{i}", supersedes=[f"r{i}"])
    store = GraphStore(ws.db_path, read_only=True)
    hits = [point_hit(r, .9 - n / 50) for n, r in enumerate(retired)]
    graph = _LightSpy(store)
    walk = walk_ranked(_query_vector(), vector_store=CannedVector(hits, store=store),
                       graph=graph, limit=2, light=True)
    assert graph.resolved == [[hash_to_uuid(r) for r in retired[:2]]]   # passed[:limit]
    assert "core_axiom" not in walk.retired[3].node                     # light beyond


def test_a_node_gone_between_the_two_reads_is_a_store_fault(ws) -> None:
    a = _commit(ws, "a")
    store = GraphStore(ws.db_path, read_only=True)
    graph = _LightSpy(store)
    graph.drop_on_hydrate = hash_to_uuid(a)
    with pytest.raises(DatabaseError):
        walk_ranked(_query_vector(), vector_store=CannedVector([point_hit(a, .9)],
                                                               store=store),
                    graph=graph, limit=2, light=True)


def test_a_light_walk_needs_the_light_resolver(ws) -> None:
    store = GraphStore(ws.db_path, read_only=True)
    with pytest.raises(WalkContractError):
        walk_ranked(_query_vector(), vector_store=CannedVector([], store=store),
                    graph=_SpyGraph(store), limit=2, light=True)


def _light_kw_of(filename: str):
    path = os.path.join(MITOS_DIR, filename)
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "walk_ranked"):
                    light = [kw.value for kw in node.keywords if kw.arg == "light"]
                    out[fn.name] = (ast.literal_eval(light[0]) if light else None)
    return out


def test_the_four_ranked_reads_walk_light_and_the_gather_whole() -> None:
    assert _light_kw_of("mcp_server.py") == {"surface_decisions": True,
                                             "query_decisions": True}
    assert _light_kw_of("cli.py") == {"cmd_surface": True, "cmd_query": True}
    assert set(_light_kw_of("conflict.py").values()) == {None}
