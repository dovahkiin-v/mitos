"""Phase 7a: the record pause's tombstone partition.

A draft that sits at or above the tombstone floor to a retired decision may be
re-treading ground the corpus already tried and abandoned. The gather already hands
the record path every retired decision its live fill passed (``Gathering.retired``,
2d); ``MitosSyncManager._review_tombstones`` turns them into the pause's second
partition: exempted by node id before the bound (a declared target, its mutation
lineage, or anything its forward kill chain reaches), bounded at ``CONFLICT_TOP_K``
with the rest counted, each entry a retired handle plus its whole axiom and two facts
about its live successors. (Vision §4.9; plan p7a §3, §4 D3–D4, §6.)

The rows here drive the review directly over a real ``GraphStore``, on the store-only
manager shell the replay worker and Layer C build, with the gathering's retired hits
resolved by point id exactly as the gather resolves them.
"""

from typing import Any, Dict, List, Optional, Tuple

import pytest

from mitos import sync
from mitos.conflict import CONFLICT_TOP_K, Gathering
from mitos.errors import DatabaseError
from mitos.parser import ParsedEntry
from mitos.retrieval_walk import EXHAUSTED, WalkHit
from mitos.store import GraphStore
from mitos.sync import DeclaredReview, MitosSyncManager, TombstoneReview
from mitos.vector_store import hash_to_uuid

_FLOOR = sync._TOMBSTONE_FLOOR
_ABOVE = round(_FLOOR + 0.05, 4)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

@pytest.fixture
def store(tmp_path) -> GraphStore:
    return GraphStore(str(tmp_path / "graph.sqlite"))


def _commit(store: GraphStore, slug: str, axiom: Optional[str] = None, **rels) -> str:
    """Commits a decision through the write path; ``rels`` values are slug lists."""
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = axiom or f"Axiom for {slug}."
    entry.rejected_paths = "An alternative."
    for name, value in rels.items():
        setattr(entry, name, value)
    return store.commit_parsed_entry(entry).node_id


def _shell(store: GraphStore) -> MitosSyncManager:
    """The store-only manager shell the replay worker and Layer C build."""
    shell = object.__new__(MitosSyncManager)
    shell.store = store
    return shell


def _gathering(store: GraphStore, scored: List[Tuple[str, float]]) -> Gathering:
    """A healthy gathering whose retired partition is ``scored`` (id, score), resolved
    by point id like the gather's own walk (the resolver's node, ``computed_state``
    included)."""
    pids = [hash_to_uuid(node_id) for node_id, _ in scored]
    nodes = store.resolve_point_ids(pids).nodes
    retired = [WalkHit(node=nodes[pid], score=score, point_id=pid, rank=i)
               for i, (pid, (_, score)) in enumerate(zip(pids, scored), 1)]
    for hit in retired:
        assert hit.node["computed_state"] not in ("active", "drifted"), hit.node["slug"]
    return Gathering(live=[], retired=retired, end=EXHAUSTED, returned=len(retired),
                     bound=675, last_score=scored[-1][1] if scored else None)


def _declared(store: GraphStore, supersedes: Optional[str] = None,
              corrects: Optional[str] = None, **extra: str) -> DeclaredReview:
    return _shell(store)._review_declared(supersedes, corrects, extra)


_NONE = DeclaredReview(slugs=set(), ids_by_target={}, ids=set())


def _review(store: GraphStore, scored: List[Tuple[str, float]],
            declared: DeclaredReview = _NONE, shown_neighbors=(), **kw) -> TombstoneReview:
    return _shell(store)._review_tombstones(_gathering(store, scored), declared,
                                            shown_neighbors=list(shown_neighbors), **kw)


def _slugs(review: TombstoneReview) -> List[str]:
    return [t["slug"] for t in review.shown]


class _Counting:
    """Forwards to a real store, counting (or failing) chosen reads."""

    def __init__(self, inner: GraphStore, fail: Optional[str] = None) -> None:
        self._inner = inner
        self._fail = fail
        self.calls: Dict[str, int] = {}

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._inner, name)
        if not callable(target):
            return target

        def call(*args, **kwargs):
            self.calls[name] = self.calls.get(name, 0) + 1
            if name == self._fail:
                raise DatabaseError(f"injected {name} fault")
            return target(*args, **kwargs)
        return call


# --------------------------------------------------------------------------- #
# Membership and the entry
# --------------------------------------------------------------------------- #

def test_a_retired_hit_at_the_floor_is_a_tombstone_with_its_handle_and_axiom(store):
    old = _commit(store, "old", "We store sessions in signed JWTs.")
    _commit(store, "new", supersedes=["old"])
    review = _review(store, [(old, _FLOOR)])
    assert review.omitted == 0 and review.cleared_by == set()
    assert review.shown == [{
        "slug": "old", "id": old, "state": "superseded", "score": _FLOOR,
        "superseded_by": ["new"], "axiom": "We store sessions in signed JWTs.",
    }]


def test_a_retired_hit_below_the_floor_reads_nothing(store):
    old = _commit(store, "old")
    _commit(store, "new", supersedes=["old"])
    counting = _Counting(store)
    shell = _shell(counting)
    review = shell._review_tombstones(
        _gathering(store, [(old, round(_FLOOR - 0.0001, 4))]), _NONE, shown_neighbors=[])
    assert review == TombstoneReview(shown=[], omitted=0, cleared_by=set())
    assert counting.calls == {}


def test_the_floor_reads_the_constant_at_call_time_and_a_caller_may_lower_it(store, monkeypatch):
    old = _commit(store, "old")
    _commit(store, "new", supersedes=["old"])
    assert _slugs(_review(store, [(old, 0.77)])) == []
    assert _slugs(_review(store, [(old, 0.77)], floor=0.75)) == ["old"]
    monkeypatch.setattr(sync, "_TOMBSTONE_FLOOR", 0.76)
    assert _slugs(_review(store, [(old, 0.77)])) == ["old"]


def test_tombstones_keep_the_gathers_rank_order(store):
    ids = []
    for slug in ("a", "b", "c"):
        ids.append(_commit(store, slug))
        _commit(store, f"{slug}-next", supersedes=[slug])
    scored = [(ids[1], 0.95), (ids[2], 0.9), (ids[0], 0.85)]
    assert _slugs(_review(store, scored)) == ["b", "c", "a"]


def test_a_retired_successor_names_the_live_end(store):
    old = _commit(store, "old")
    _commit(store, "mid", supersedes=["old"])
    _commit(store, "end", supersedes=["mid"])
    entry = _review(store, [(old, _ABOVE)]).shown[0]
    assert entry["superseded_by"] == ["mid"] and entry["live_end"] == ["end"]


def test_a_corrected_tombstone_carries_its_corrector(store):
    old = _commit(store, "old")
    _commit(store, "fix", corrects=["old"])
    entry = _review(store, [(old, _ABOVE)]).shown[0]
    assert entry["state"] == "corrected" and entry["corrected_by"] == ["fix"]
    assert "superseded_by" not in entry and "live_end" not in entry


# --------------------------------------------------------------------------- #
# Exemption by node id, before the bound (D4)
# --------------------------------------------------------------------------- #

def test_superseding_x_clears_xs_retired_predecessor(store):
    p = _commit(store, "p")
    _commit(store, "x", supersedes=["p"])
    review = _review(store, [(p, _ABOVE)], _declared(store, supersedes="x"))
    assert review.shown == [] and review.omitted == 0 and review.cleared_by == {"x"}


def test_a_corrects_chain_clears_through_the_kill_chain_walk(store):
    """P corrected by X, the draft amends X: `get_lineage` leaves `corrects` out, so
    only the forward kill-chain walk from P reaches X."""
    p = _commit(store, "p")
    _commit(store, "x", corrects=["p"])
    declared = _declared(store, amends="x")
    assert _shell(store).store.get_lineage(declared.ids_by_target["x"].copy().pop()) == []
    review = _review(store, [(p, _ABOVE)], declared)
    assert review.shown == [] and review.cleared_by == {"x"}


def test_a_lineage_that_holds_the_corrector_clears(store):
    """The draft amends Y; Y amends C; C corrected P. P's chain reaches C, which is in
    Y's lineage."""
    p = _commit(store, "p")
    _commit(store, "c", corrects=["p"])
    _commit(store, "y", amends=["c"])
    review = _review(store, [(p, _ABOVE)], _declared(store, amends="y"))
    assert review.shown == [] and review.cleared_by == {"y"}


def test_the_exemption_is_per_tombstone(store):
    p = _commit(store, "p")
    _commit(store, "x", supersedes=["p"])
    q = _commit(store, "q")
    _commit(store, "q-next", supersedes=["q"])
    review = _review(store, [(p, 0.9), (q, 0.85)], _declared(store, supersedes="x"))
    assert _slugs(review) == ["q"] and review.cleared_by == {"x"}


def test_a_lineage_ancestor_of_an_amends_target_is_exempt_by_its_own_id(store):
    """B amends A, and A was superseded by an unrelated C: A's kill chain reaches C,
    not B, so only A's id in B's lineage exempts it."""
    a = _commit(store, "a")
    _commit(store, "b", amends=["a"])
    _commit(store, "c", supersedes=["a"])
    declared = _declared(store, amends="b")
    assert a in declared.ids_by_target["b"]
    review = _review(store, [(a, _ABOVE)], declared)
    assert review.shown == [] and review.cleared_by == {"b"}


def test_a_declared_retired_target_exempts_itself(store):
    """A draft may declare a relation to a uniquely-slugged retired decision; its id is
    in the exempt set."""
    old = _commit(store, "old")
    _commit(store, "new", supersedes=["old"])
    review = _review(store, [(old, _ABOVE)], _declared(store, cites="old"))
    assert review.shown == [] and review.cleared_by == {"old"}


def test_a_same_slug_declaration_clears_nothing_by_slug(store):
    """The retired `database` shares its slug with its live successor. The slug names
    two nodes, so a declaration of it resolves to no exempting id (Phase A refuses it
    on the record path); the slug alone must not clear the tombstone (constraint 8)."""
    from test_point_identity import same_slug_lineage
    retired, _successor = same_slug_lineage(store)
    declared = _declared(store, cites="database")
    assert "database" in declared.slugs and declared.ids_by_target == {}
    assert _slugs(_review(store, [(retired, _ABOVE)], declared)) == ["database"]


def test_exemption_runs_before_the_bound(store):
    """Exempt hits never take a slot: seven retired hits above the floor, two exempt,
    show five and omit none."""
    hits = []
    for i in range(7):
        node = _commit(store, f"t{i}")
        _commit(store, f"t{i}-next", supersedes=[f"t{i}"])
        hits.append((node, round(0.95 - 0.01 * i, 4)))
    declared = _declared(store, cites="t0-next,t1-next")
    review = _review(store, hits, declared)
    assert _slugs(review) == ["t2", "t3", "t4", "t5", "t6"] and review.omitted == 0
    assert review.cleared_by == {"t0-next", "t1-next"}


# --------------------------------------------------------------------------- #
# The bound and the count
# --------------------------------------------------------------------------- #

def _seven(store: GraphStore) -> List[Tuple[str, float]]:
    hits = []
    for i in range(7):
        node = _commit(store, f"t{i}")
        _commit(store, f"t{i}-next", supersedes=[f"t{i}"])
        hits.append((node, round(0.95 - 0.01 * i, 4)))
    return hits


def test_seven_tombstones_show_the_five_closest_and_count_two(store):
    review = _review(store, _seven(store))
    assert CONFLICT_TOP_K == 5
    assert _slugs(review) == ["t0", "t1", "t2", "t3", "t4"] and review.omitted == 2


def test_an_unbounded_review_shows_every_tombstone(store):
    review = _review(store, _seven(store), bound=None)
    assert len(review.shown) == 7 and review.omitted == 0


# --------------------------------------------------------------------------- #
# The successor facts
# --------------------------------------------------------------------------- #

def test_a_successor_among_the_shown_neighbours_is_on_the_pause(store):
    old = _commit(store, "old")
    _commit(store, "New-Way", supersedes=["old"])
    entry = _review(store, [(old, _ABOVE)], shown_neighbors=[{"slug": "new-way"}]).shown[0]
    assert entry["successor_on_pause"] == ["New-Way"]


def test_a_successor_not_shown_is_not_on_the_pause(store):
    old = _commit(store, "old")
    _commit(store, "new", supersedes=["old"])
    entry = _review(store, [(old, _ABOVE)], shown_neighbors=[{"slug": "other"}]).shown[0]
    assert "successor_on_pause" not in entry


def test_the_live_end_is_the_successor_on_the_pause(store):
    old = _commit(store, "old")
    _commit(store, "mid", supersedes=["old"])
    _commit(store, "end", supersedes=["mid"])
    entry = _review(store, [(old, _ABOVE)],
                    shown_neighbors=[{"slug": "mid"}, {"slug": "end"}]).shown[0]
    assert entry["successor_on_pause"] == ["end"]


def test_a_same_slug_successor_is_ambiguous(store):
    from test_point_identity import same_slug_lineage
    retired, _successor = same_slug_lineage(store)
    entry = _review(store, [(retired, _ABOVE)]).shown[0]
    assert entry["superseded_by"] == ["database"] and entry["successor_ambiguous"] is True


def test_a_distinct_slug_successor_carries_no_ambiguity_key(store):
    old = _commit(store, "old")
    _commit(store, "new", supersedes=["old"])
    assert "successor_ambiguous" not in _review(store, [(old, _ABOVE)]).shown[0]


def test_ambiguity_is_the_validators_own_predicate(store, monkeypatch):
    """The fact reads `_ambiguous_target`, the predicate the three refusals read."""
    old = _commit(store, "old")
    _commit(store, "new", supersedes=["old"])
    monkeypatch.setattr(sync, "_ambiguous_target", lambda ids: True)
    assert _review(store, [(old, _ABOVE)]).shown[0]["successor_ambiguous"] is True


# --------------------------------------------------------------------------- #
# Faults propagate (D3)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("read", ["get_kill_chains", "get_modifiers_map", "resolve_slug"])
def test_a_tombstone_read_fault_propagates_typed(store, read):
    old = _commit(store, "old")
    _commit(store, "new", supersedes=["old"])
    shell = _shell(_Counting(store, fail=read))
    with pytest.raises(DatabaseError, match="injected"):
        shell._review_tombstones(_gathering(store, [(old, _ABOVE)]), _NONE, shown_neighbors=[])


def test_a_raw_sqlite_fault_on_a_slug_read_is_typed(store, monkeypatch):
    import sqlite3
    old = _commit(store, "old")
    _commit(store, "new", supersedes=["old"])

    def broken(slug):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(store, "resolve_slug", broken)
    with pytest.raises(DatabaseError, match="resolve slug"):
        _review(store, [(old, _ABOVE)])


# --------------------------------------------------------------------------- #
# The declarations (one lineage walk, three fields)
# --------------------------------------------------------------------------- #

def test_declared_review_slugs_equal_the_suppression_set_the_pause_used(store):
    a = _commit(store, "a")
    _commit(store, "b", supersedes=["a"])
    _commit(store, "c", amends=["b"])
    _commit(store, "d")
    declared = _declared(store, amends="c", cites="d")
    assert declared.slugs == {"c", "b", "a", "d"}
    assert declared.ids_by_target["c"] == {store.resolve_slug(s)[0] for s in ("c", "b", "a")}
    assert declared.ids_by_target["d"] == {store.resolve_slug("d")[0]}
    assert declared.ids == {store.resolve_slug(s)[0] for s in ("a", "b", "c", "d")}
    assert a in declared.ids


def test_review_declared_targets_keeps_its_provider_gate(store):
    _commit(store, "a")
    _commit(store, "b", amends=["a"])
    shell = _shell(store)
    shell.embed_provider = None
    shell.vector_store = None
    assert shell._review_declared_targets(None, None, {"amends": "b"}) == {"b"}
    shell.embed_provider = object()
    shell.vector_store = object()
    assert shell._review_declared_targets(None, None, {"amends": "b"}) == {"a", "b"}


def test_a_raw_sqlite_lineage_fault_is_typed(store, monkeypatch):
    import sqlite3
    _commit(store, "a")
    _commit(store, "b", amends=["a"])

    def broken(node_id):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(store, "get_lineage", broken)
    with pytest.raises(DatabaseError, match="lineage"):
        _declared(store, amends="b")


# --------------------------------------------------------------------------- #
# Store: get_kill_chains
# --------------------------------------------------------------------------- #

def test_get_kill_chains_returns_reached_and_slugs(store):
    old = _commit(store, "old")
    mid = _commit(store, "mid", supersedes=["old"])
    end = _commit(store, "end", supersedes=["mid"])
    chains, slugs = store.get_kill_chains([old, old])
    assert list(chains) == [old]
    assert chains[old].reached == [mid, end] and chains[old].killers == [mid]
    assert slugs == {mid: "mid", end: "end"}
    assert store.get_kill_chains([]) == ({}, {})


def test_get_kill_chains_chunks_its_seeds(store, monkeypatch):
    """One `?` per seed, chunked at 500 like the point-id reads."""
    from mitos import store as store_module
    old = _commit(store, "old")
    mid = _commit(store, "mid", supersedes=["old"])
    seen = []
    original = store._walk_kill_chains

    def spy(conn, node_ids):
        seen.append(len(node_ids))
        return original(conn, node_ids)
    monkeypatch.setattr(store, "_walk_kill_chains", spy)
    monkeypatch.setattr(store_module, "_POINT_ID_CHUNK", 2)
    seeds = [old, "a" * 64, "b" * 64, "c" * 64, "d" * 64]
    chains, _ = store.get_kill_chains(seeds)
    assert seen == [2, 2, 1] and set(chains) == set(seeds)
    assert chains[old].reached == [mid]


def test_get_kill_chains_wraps_a_broken_connection(store, monkeypatch):
    import sqlite3

    class _Broken:
        def execute(self, sql, params=()):
            raise sqlite3.OperationalError("database is locked")

        def close(self):
            pass
    monkeypatch.setattr(store, "_get_connection", lambda: _Broken())
    with pytest.raises(DatabaseError, match="kill chains"):
        store.get_kill_chains(["a" * 64])
