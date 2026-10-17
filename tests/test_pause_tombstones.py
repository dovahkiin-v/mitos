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


_NONE = DeclaredReview(slugs=set(), ids_by_target={}, ids=set(), direct_ids={})


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


def test_a_sixty_deep_chain_answers_alike_through_both_readers(store):
    """Row 13: the refactor is behaviour-identical on a deep chain — `get_live_ends`
    and `live_ends_from_chains` over `get_kill_chains` name the one live end."""
    from mitos.store import live_ends_from_chains
    ids = [_commit(store, "link-00")]
    for i in range(1, 61):
        ids.append(_commit(store, f"link-{i:02d}", supersedes=[f"link-{i - 1:02d}"]))
    seeds = ids[:3]
    assert store.get_live_ends(seeds) == {i: ["link-60"] for i in seeds}
    chains, slugs = store.get_kill_chains(seeds)
    assert live_ends_from_chains(chains, slugs) == store.get_live_ends(seeds)
    assert len(chains[ids[0]].reached) == 60 and chains[ids[0]].live_ends == [ids[-1]]


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


# =========================================================================== #
# The record path (T17's 7a rows): the partition on the pause, MCP and CLI
# =========================================================================== #

import json  # noqa: E402
from unittest.mock import patch  # noqa: E402

from mitos.cli import cmd_record  # noqa: E402
from mitos.conflict import CONFLICT_OVERFETCH_LIMIT  # noqa: E402

from mitos.shell_quote import shell_arg  # noqa: E402

from _point_hits import point_hit  # noqa: E402
from test_neighbor_review import (_FakeEmbed, _FakeVector, _RaisingEmbed,  # noqa: E402,F401
                                  _arm, _armed_real_manager_factory, _seed, offline,
                                  small_bound, ws)

_HIGH = 0.9
_BETWEEN = round((sync._NEIGHBOR_REVIEW_THRESHOLD + _FLOOR) / 2, 4)  # live floor ≤ x < tombstone floor
_DRAFT = "The draft walks a road the corpus already tried."


def _id_of(m, slug):
    ids = m.store.resolve_slug(slug)
    assert len(ids) == 1, ids
    return ids[0]


def _retire(m, slug, axiom=None, by=None, how="supersedes"):
    """Seeds ``slug`` and a live ``by`` (default ``{slug}-next``) that kills it."""
    _seed(m, slug, axiom or f"Axiom for {slug}.")
    _seed(m, by or f"{slug}-next", f"Axiom for {by or slug + '-next'}.", **{how: slug})
    return _id_of(m, slug)


def _buffer(config):
    with open(config.decisions_file, encoding="utf-8") as fh:
        return fh.read()


def _draft(m, slug="draft", **rels):
    return m.record_decision_entry(_DRAFT, "rej", ["s"], slug=slug, **rels)


def _surfaces(config, matches, **rels):
    """The same draft through MCP `record_decision` and CLI `record` (text and --json),
    each surface building its own armed real manager. Returns (mcp, cli_json, cli_text,
    cli_exit)."""
    from mitos import mcp_server
    factory = _armed_real_manager_factory(matches)
    with patch("mitos.cli.MitosSyncManager", side_effect=factory), \
         patch("mitos.sync.MitosSyncManager", side_effect=factory), \
         patch("mitos.mcp_server.MitosConfig", return_value=config):
        mcp = json.loads(mcp_server.record_decision(_DRAFT, "rej", ["s"], slug="draft",
                                                    project=config.workspace_dir, **rels))
        import io
        from contextlib import redirect_stderr, redirect_stdout
        out = io.StringIO()
        with redirect_stdout(out), pytest.raises(SystemExit) as exc:
            cmd_record(config, axiom=_DRAFT, rejected="rej", scope=["s"], slug="draft",
                       as_json=True, **rels)
        cli_json = json.loads(out.getvalue())
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()), pytest.raises(SystemExit) as exc2:
            cmd_record(config, axiom=_DRAFT, rejected="rej", scope=["s"], slug="draft", **rels)
    assert exc.value.code == exc2.value.code
    return mcp, cli_json, err.getvalue(), exc2.value.code


# -- Row 1: a re-tread pauses on its tombstone alone -------------------------------------

def test_a_re_tread_pauses_on_its_tombstone_alone_and_writes_nothing(ws):
    config, m = ws
    old = _retire(m, "old-way", "We store sessions in signed JWTs.", by="new-way")
    before = _buffer(config)
    _arm(m, [{"slug": "old-way", "score": _HIGH}])
    res = _draft(m)
    assert res["status"] == "needs_review" and res["code"] == "similar_decision_exists"
    assert res["neighbors"] == []
    assert res["tombstones"] == [{
        "slug": "old-way", "id": old, "state": "superseded", "score": _HIGH,
        "superseded_by": ["new-way"], "axiom": "We store sessions in signed JWTs."}]
    assert "tombstones_omitted" not in res and "neighbors_omitted" not in res
    msg = res["message"]
    assert "similar to 0" not in msg and "this corpus retired" in msg
    assert f"≥{sync._floor_text(_FLOOR)}" in msg and "acknowledge_neighbors=True" in msg
    assert "mitos " not in msg
    assert _buffer(config) == before
    assert m.store.get_node_by_slug("draft") is None


@pytest.mark.parametrize("floor, text", [
    (0.80, "0.80"), (0.8265, "0.8265"), (0.815, "0.815"), (0.82649, "0.82649"),
])
def test_a_floor_prints_every_place_it_holds_and_never_rounds_up(floor, text):
    # A floor past four places printed at four would round up and state a bound the
    # partition does not keep (fresh eyes: 0.82649 printed "0.8265").
    assert sync._floor_text(floor) == text


def test_a_retired_hit_between_the_floors_does_not_pause(ws):
    """The two floors are separate: a retired neighbour at or above the live floor but
    under the tombstone floor is no tombstone."""
    config, m = ws
    assert sync._NEIGHBOR_REVIEW_THRESHOLD <= _BETWEEN < _FLOOR
    _retire(m, "old-way")
    _arm(m, [{"slug": "old-way", "score": _BETWEEN}])
    assert _draft(m)["status"] == "created"


def test_both_surfaces_pause_on_a_tombstone_alone(ws):
    config, m = ws
    old = _retire(m, "old-way", "We store sessions in signed JWTs.", by="new-way")
    mcp, cli_json, text, code = _surfaces(config, [{"slug": "old-way", "score": _HIGH}])
    assert code == 2
    for res in (mcp, cli_json):
        assert res["status"] == "needs_review" and res["neighbors"] == []
        assert [t["id"] for t in res["tombstones"]] == [old]
    assert "this corpus retired" in text and "0 existing" not in text
    assert f"† old-way  (superseded, score {_HIGH:.3f})  id {old}" in text
    assert "We store sessions in signed JWTs." in text
    assert "superseded by: new-way" in text
    assert f"-- {shell_arg(old)}`" in text and "mitos show" in text
    assert "--acknowledge-neighbors" in text and "Judge each neighbour" not in text


# -- Row 2: the kill-chain exemption --------------------------------------------------------

def test_superseding_x_does_not_pause_on_xs_retired_predecessor(ws):
    config, m = ws
    _retire(m, "p", by="x")
    _arm(m, [{"slug": "p", "score": _HIGH}])
    assert _draft(m, supersedes="x")["status"] == "created"


def test_amending_the_corrector_does_not_pause_on_the_corrected(ws):
    config, m = ws
    _retire(m, "p", by="x", how="corrects")
    _arm(m, [{"slug": "p", "score": _HIGH}])
    assert _draft(m, amends="x")["status"] == "created"


def test_a_lineage_holding_the_corrector_does_not_pause(ws):
    config, m = ws
    _retire(m, "p", by="c", how="corrects")
    _seed(m, "y", "Axiom for y.", amends="c")
    _arm(m, [{"slug": "p", "score": _HIGH}])
    assert _draft(m, amends="y")["status"] == "created"


def test_an_undeclared_tombstone_in_the_same_draft_still_pauses(ws):
    config, m = ws
    _retire(m, "p", by="x")
    _retire(m, "q")
    _arm(m, [{"slug": "p", "score": 0.92}, {"slug": "q", "score": 0.88}])
    res = _draft(m, supersedes="x")
    assert res["status"] == "needs_review"
    assert [t["slug"] for t in res["tombstones"]] == ["q"]


# -- Row 3: declared target and lineage --------------------------------------------------

def test_a_retired_lineage_ancestor_of_an_amends_target_is_never_a_tombstone(ws):
    config, m = ws
    _seed(m, "a", "Axiom for a.")
    _seed(m, "b", "Axiom for b.", amends="a")
    _seed(m, "c", "Axiom for c.", supersedes="a")
    _arm(m, [{"slug": "a", "score": _HIGH}])
    assert _draft(m, amends="b")["status"] == "created"


# -- Row 4: the same-slug successor ------------------------------------------------------

def test_a_same_slug_successor_says_acknowledgement_only(ws):
    config, m = ws
    from test_point_identity import same_slug_lineage
    retired, _successor = same_slug_lineage(m.store)
    _arm(m, [point_hit(retired, _HIGH)])
    res = _draft(m)
    assert res["status"] == "needs_review"
    (t,) = res["tombstones"]
    assert t["id"] == retired and t["successor_ambiguous"] is True
    assert "successor_ambiguous clears only with acknowledge_neighbors=True" in res["message"]


def test_a_distinct_slug_successor_carries_no_ambiguity(ws):
    config, m = ws
    _retire(m, "old-way", by="new-way")
    _arm(m, [{"slug": "old-way", "score": _HIGH}])
    res = _draft(m)
    assert "successor_ambiguous" not in res["tombstones"][0]
    assert "successor_ambiguous" not in res["message"]


# -- Row 5: the successor on the pause ---------------------------------------------------

def test_a_tombstone_stays_beside_its_successor_and_one_relation_clears_both(ws):
    config, m = ws
    _retire(m, "old-way", by="new-way")
    matches = [{"slug": "old-way", "score": _HIGH}, {"slug": "new-way", "score": 0.88}]
    _arm(m, matches)
    res = _draft(m)
    assert [n["slug"] for n in res["neighbors"]] == ["new-way"]
    (t,) = res["tombstones"]
    assert t["slug"] == "old-way" and t["successor_on_pause"] == ["new-way"]
    assert "successor_on_pause" in res["message"]
    _arm(m, matches)
    assert _draft(m, cites="new-way")["status"] == "created"


# -- Row 6: the bound and both counts ----------------------------------------------------

def test_seven_tombstones_show_five_and_count_two(ws):
    config, m = ws
    for i in range(7):
        _retire(m, f"t{i}")
    _arm(m, [{"slug": f"t{i}", "score": round(0.95 - 0.01 * i, 4)} for i in range(7)])
    res = _draft(m)
    assert [t["slug"] for t in res["tombstones"]] == ["t0", "t1", "t2", "t3", "t4"]
    assert res["tombstones_omitted"] == 2
    assert "2 more tombstone(s) above the floor are not shown" in res["message"]


def test_seven_live_neighbours_show_five_and_count_two(ws):
    config, m = ws
    for i in range(7):
        _seed(m, f"n{i}", f"Axiom for n{i}.")
    _arm(m, [{"slug": f"n{i}", "score": round(0.95 - 0.01 * i, 4)} for i in range(7)])
    res = _draft(m)
    assert [n["slug"] for n in res["neighbors"]] == ["n0", "n1", "n2", "n3", "n4"]
    assert res["neighbors_omitted"] == 2 and "tombstones" not in res
    assert "and 2 more above the floor are not shown" in res["message"]


def test_at_the_bound_neither_count_is_present(ws):
    config, m = ws
    for i in range(5):
        _seed(m, f"n{i}", f"Axiom for n{i}.")
        _retire(m, f"t{i}")
    _arm(m, [{"slug": f"n{i}", "score": 0.9} for i in range(5)]
         + [{"slug": f"t{i}", "score": 0.89} for i in range(5)])
    res = _draft(m)
    assert len(res["neighbors"]) == 5 and len(res["tombstones"]) == 5
    assert "neighbors_omitted" not in res and "tombstones_omitted" not in res


# -- Row 7: D7, the full review ----------------------------------------------------------

def _live_corpus(m, n):
    for i in range(n):
        _seed(m, f"live-{i:02d}", f"Axiom for live {i}.")
    return [{"slug": f"live-{i:02d}", "score": round(0.97 - 0.005 * i, 4)} for i in range(n)]


def test_a_full_review_says_its_counts_cover_what_it_read(ws):
    config, m = ws
    _arm(m, _live_corpus(m, CONFLICT_OVERFETCH_LIMIT + 1))
    res = _draft(m)
    full = res["neighbor_review_full"]
    assert f"all {CONFLICT_OVERFETCH_LIMIT} of its live candidates" in full
    assert "a live neighbour or a tombstone may lie past them" in full
    assert "mitos" not in full and "neighbor_review_cut" not in res
    assert res["neighbors_omitted"] == CONFLICT_OVERFETCH_LIMIT - 5


def test_an_exhausted_review_is_not_full(ws):
    config, m = ws
    _arm(m, _live_corpus(m, CONFLICT_OVERFETCH_LIMIT))
    assert "neighbor_review_full" not in _draft(m)


def test_a_full_review_whose_last_candidate_is_under_both_floors_is_not_full(ws):
    config, m = ws
    matches = _live_corpus(m, CONFLICT_OVERFETCH_LIMIT + 1)
    matches[CONFLICT_OVERFETCH_LIMIT - 1]["score"] = 0.5
    matches[CONFLICT_OVERFETCH_LIMIT]["score"] = 0.49
    _arm(m, matches)
    assert "neighbor_review_full" not in _draft(m)


def test_a_cut_review_is_never_full(ws, small_bound):
    from _point_hits import orphan_hit
    config, m = ws
    _retire(m, "old-way")
    hits = [{"slug": "old-way", "score": 0.95}]
    hits += [orphan_hit(0.94 - 0.001 * i, tag=f"cut-{i}") for i in range(CONFLICT_OVERFETCH_LIMIT)]
    _arm(m, hits)
    res = _draft(m)
    assert res["status"] == "needs_review" and "neighbor_review_full" not in res
    cut = res["neighbor_review_cut"]
    assert "a live neighbour or a tombstone may lie past what it read" in cut


def test_a_cut_between_the_floors_names_only_the_live_partition(ws, small_bound):
    from _point_hits import orphan_hit
    config, m = ws
    _seed(m, "prior", "Axiom for prior.")
    tail = _BETWEEN
    hits = [{"slug": "prior", "score": 0.95}]
    hits += [orphan_hit(0.94 - (0.94 - tail) * i / (CONFLICT_OVERFETCH_LIMIT - 1), tag=f"cut-{i}")
             for i in range(CONFLICT_OVERFETCH_LIMIT)]
    _arm(m, hits)
    cut = _draft(m)["neighbor_review_cut"]
    assert "so a live neighbour may lie past what it read" in cut


# -- Row 8: the failure state ------------------------------------------------------------

@pytest.mark.parametrize("read", ["get_kill_chains", "get_modifiers_map"])
def test_a_tombstone_read_fault_fails_open_and_never_reads_as_clean(ws, monkeypatch, read):
    config, m = ws
    _retire(m, "old-way")
    _arm(m, [{"slug": "old-way", "score": _HIGH}])
    calls = []

    def broken(*a, **k):
        calls.append(a)
        raise DatabaseError(f"injected {read} fault")
    monkeypatch.setattr(m.store, read, broken)
    res = _draft(m)
    assert res["status"] == "created" and calls
    assert "could not run (graph read failed)" in res["neighbor_review_unavailable"]
    assert "tombstones" not in res


def test_a_fault_also_discards_the_live_neighbours(ws, monkeypatch):
    config, m = ws
    _retire(m, "old-way")
    _seed(m, "live-twin", "Axiom for live twin.")
    _arm(m, [{"slug": "old-way", "score": _HIGH}, {"slug": "live-twin", "score": 0.88}])
    monkeypatch.setattr(m.store, "get_kill_chains",
                        lambda ids: (_ for _ in ()).throw(DatabaseError("injected")))
    res = _draft(m)
    assert res["status"] == "created" and "neighbor_review_unavailable" in res


def test_no_retired_hit_at_the_floor_reads_nothing(ws, monkeypatch):
    config, m = ws
    _retire(m, "old-way")
    _arm(m, [{"slug": "old-way", "score": _BETWEEN}])
    calls = []
    monkeypatch.setattr(m.store, "get_kill_chains", lambda ids: calls.append(ids))
    res = _draft(m)
    assert res["status"] == "created" and calls == []
    assert "neighbor_review_unavailable" not in res


def test_an_unavailable_gather_does_no_tombstone_work(ws, monkeypatch):
    from mitos.errors import EmbeddingError
    config, m = ws
    _retire(m, "old-way")
    _arm(m, [{"slug": "old-way", "score": _HIGH}])
    m.embed_provider = _RaisingEmbed(EmbeddingError("down"))
    calls = []
    monkeypatch.setattr(m.store, "get_kill_chains", lambda ids: calls.append(ids))
    res = _draft(m)
    assert res["status"] == "created" and calls == []
    assert "embedding service unavailable" in res["neighbor_review_unavailable"]


# -- Row 9: the acknowledged path --------------------------------------------------------

def test_acknowledging_past_a_tombstone_commits_with_the_live_list_unchanged(ws):
    config, m = ws
    _retire(m, "old-way")
    _arm(m, [{"slug": "old-way", "score": _HIGH}])
    res = _draft(m, acknowledge_neighbors=True)
    assert res["status"] == "created"
    assert res["acknowledged_neighbors"] == []
    assert "tombstones" not in res and "tombstones_omitted" not in res


# -- Row 10: D9, a declaration that cleared a tombstone is not "no near match" -----------

def test_a_declaration_that_cleared_a_tombstone_sits_in_declared(ws):
    config, m = ws
    _retire(m, "p", by="x")
    _seed(m, "live-twin", "Axiom for live twin.")
    _arm(m, [{"slug": "p", "score": _HIGH}, {"slug": "live-twin", "score": 0.88}])
    res = _draft(m, cites="x")
    assert res["status"] == "needs_review" and "tombstones" not in res
    assert res["declared"] == ["x"] and "declared_no_near_match" not in res


def test_without_the_tombstone_the_same_declaration_is_no_near_match(ws):
    """D9's control: the exclusion is the cleared tombstone, nothing else."""
    config, m = ws
    _retire(m, "p", by="x")
    _seed(m, "live-twin", "Axiom for live twin.")
    _arm(m, [{"slug": "live-twin", "score": 0.88}])
    res = _draft(m, cites="x")
    assert res["declared_no_near_match"] == [{"slug": "x"}]


# -- Row 12: parity ----------------------------------------------------------------------

def test_cli_json_and_mcp_carry_one_pause_and_the_text_names_each_id(ws):
    config, m = ws
    old = _retire(m, "old-way", by="new-way")
    other = _retire(m, "older-way")
    _seed(m, "live-twin", "Axiom for live twin.")
    matches = [{"slug": "old-way", "score": 0.93}, {"slug": "older-way", "score": 0.91},
               {"slug": "live-twin", "score": 0.88}, {"slug": "new-way", "score": 0.86}]
    mcp, cli_json, text, code = _surfaces(config, matches)
    assert code == 2
    echo = {"project", "collection", "workspace"}
    assert set(mcp) == set(cli_json)
    assert {k: v for k, v in mcp.items() if k not in echo} == \
        {k: v for k, v in cli_json.items() if k not in echo}
    assert [t["id"] for t in mcp["tombstones"]] == [old, other]
    assert mcp["tombstones"][0]["successor_on_pause"] == ["new-way"]
    assert "mitos " not in mcp["message"]
    for tid in (old, other):
        assert f"-- {shell_arg(tid)}`" in text
    assert "its successor is on this pause: new-way" in text
    assert "Judge each neighbour" in text and "For a tombstone the draft does not re-tread" in text
