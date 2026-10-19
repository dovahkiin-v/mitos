"""The conflict gather on the shared identity walk (recall-confidence vision, Phase 2d).

``conflict.gather_candidates`` makes one vector call at ``CONFLICT_RAW_BOUND``, resolves
every hit by point id through ``retrieval_walk.walk_ranked`` and fills
``CONFLICT_OVERFETCH_LIMIT`` (20) live candidates, returning them in a ``Gathering``
beside the retired hits it passed and the fetch's end. The rows here pin:

* **A1** — each live candidate's node equals what ``get_node_by_slug`` returns for it,
  dict for dict (``computed_state`` popped, modifier stamps merged), so every payload
  built from a candidate is unchanged by the move;
* **identity** — a same-slug successor is gathered once at its own score with its
  retired twin in ``retired``; a renamed decision is gathered, screened and judged under
  its current slug while its point's payload still carries the old one;
* **live-fill and the cut** — the ten-to-one neighbourhood fills at the raw bound, and
  ``cut_above`` reads the last *raw* score against the consumer's floor, inclusively;
* **the contract** — a store missing a method the gather reads, a ``MagicMock`` store
  and a non-dict stamp map raise ``WalkContractError`` and are never read as
  ``Unavailable``;
* **S2** — no judge path reads the retired partition.

Every score and bound is read from the constants or the fixture, never hard-coded.
"""

import ast
import inspect
import shutil
import tempfile
import textwrap
from typing import Any, Dict, Iterator, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from mitos import check, cli, conflict, sync
from mitos.cli import cmd_init
from mitos.config import MitosConfig
from mitos.conflict import (
    CONFLICT_OVERFETCH_LIMIT,
    CONFLICT_SIMILARITY_FLOOR,
    Gathering,
    Unavailable,
    gather_candidates,
    run_conflict_check,
    screen_candidates,
)
from mitos.parser import ParsedEntry
from mitos.retrieval_walk import (
    EDGE_FOUND,
    EDGE_UNMEASURED,
    EXHAUSTED,
    SHORT_BOUND,
    WalkContractError,
)
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager

from _conflict_helpers import _execution
from _point_hits import CannedVector, orphan_hit, point_hit
from test_point_identity import same_slug_lineage


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def ws() -> Iterator[Tuple[MitosConfig, MitosSyncManager]]:
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    cmd_init(config)
    yield config, MitosSyncManager(config)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def store(tmp_path) -> GraphStore:
    return GraphStore(str(tmp_path / "graph.sqlite"))


class _Embed:
    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        return [0.1, 0.2, 0.3]


def _commit(store: GraphStore, slug: str, axiom: str, **relations: List[str]) -> str:
    entry = ParsedEntry("decision", slug, 1, 10)
    entry.axiom = axiom
    entry.rejected_paths = f"Rejected the alternative to {slug}."
    entry.scope = ["core"]
    for name, targets in relations.items():
        setattr(entry, name, targets)
    return store.commit_parsed_entry(entry).node_id


def _gather(hits: List[Dict[str, Any]], store: Any) -> Any:
    return gather_candidates("A draft axiom.", embed_provider=_Embed(),
                             vector_store=CannedVector(hits, store=store), store=store)


def _gathering(end: str, last_score: Optional[float], live: int = 0) -> Gathering:
    return Gathering(live=[None] * live, retired=[], end=end, returned=0,
                     bound=conflict.CONFLICT_RAW_BOUND, last_score=last_score)


# --------------------------------------------------------------------------- #
# The constants
# --------------------------------------------------------------------------- #

def test_the_raw_bound_leaves_room_for_the_live_count_and_the_ten_to_one_row() -> None:
    """The walk requires ``limit < bound``; the ten-to-one neighbourhood must fill."""
    assert conflict.CONFLICT_RAW_BOUND > CONFLICT_OVERFETCH_LIMIT
    assert conflict.CONFLICT_RAW_BOUND >= 200 + CONFLICT_OVERFETCH_LIMIT


def test_conflict_no_longer_spells_its_own_live_states() -> None:
    """Liveness is the walk's ``LIVE_STATES``, agreement-pinned to the store's."""
    assert not hasattr(conflict, "_LIVE_STATES")


# --------------------------------------------------------------------------- #
# A1 — the candidate's node is get_node_by_slug's, dict for dict
# --------------------------------------------------------------------------- #

def test_a1_every_gathered_node_equals_get_node_by_slug(store: GraphStore) -> None:
    """Over every live node of a graph holding stamps of every kind the gather can see.

    ``amended``, ``narrowed`` and a node amended twice carry stamps; a superseded and a
    corrected node are retired and must not appear. Break B6 (no stamp merge) and B7
    (``computed_state`` left on the node) red this row.
    """
    _commit(store, "base", "The base decision.")
    _commit(store, "amender", "Amends the base.", amends=["base"])
    _commit(store, "narrower", "Narrows the base.", narrows=["base"])
    _commit(store, "second", "A second decision.")
    _commit(store, "second-amender", "Amends the second.", amends=["second"])
    _commit(store, "old", "An old decision.")
    _commit(store, "new", "Supersedes the old one.", supersedes=["old"])
    _commit(store, "typo", "A decision with a typo.")
    _commit(store, "fixed", "Corrects the typo.", corrects=["typo"])
    ids = [store.resolve_slug(s)[0] for s in
           ("base", "amender", "narrower", "second", "second-amender", "old", "new",
            "typo", "fixed")]
    hits = [point_hit(node_id, 0.99 - 0.01 * i) for i, node_id in enumerate(ids)]
    gathered = _gather(hits, store)

    assert {c.slug for c in gathered.live} == {"base", "amender", "narrower", "second",
                                               "second-amender", "new", "fixed"}
    assert {h.node["slug"] for h in gathered.retired} == {"old", "typo"}
    stamped = 0
    for candidate in gathered.live:
        expected = store.get_node_by_slug(candidate.slug)
        assert candidate.node == expected, candidate.slug
        assert candidate.slug == expected["slug"]
        assert candidate.state == store.get_node_state(expected["id"])
        stamped += int(any(k in candidate.node for k in ("amended_by", "narrowed_by")))
    assert stamped == 2  # the row exercised the merge, not an empty one


# --------------------------------------------------------------------------- #
# Identity: the same slug, and the rename
# --------------------------------------------------------------------------- #

def test_a_same_slug_successor_is_gathered_once_at_its_own_score(store: GraphStore) -> None:
    """Retired X and live Y share slug ``database``. X's point outranks Y's.

    Before 2d the payload-slug join resolved X's point to Y, so Y was gathered twice
    and once at X's score. Now X is a retired hit and Y is live once, at its own score.
    """
    retired, successor = same_slug_lineage(store)
    gathered = _gather([point_hit(retired, 0.93, slug="database"),
                        point_hit(successor, 0.81, slug="database")], store)

    assert [(c.node["id"], c.score) for c in gathered.live] == [(successor, 0.81)]
    assert [(h.node["id"], h.score) for h in gathered.retired] == [(retired, 0.93)]


def test_the_pause_files_a_same_slug_successor_under_its_own_score(ws) -> None:
    """The pause's gathered index holds Y's own score, not its retired twin's."""
    config, m = ws
    retired, successor = same_slug_lineage(m.store)
    m.embed_provider = _Embed()
    m.vector_store = CannedVector([point_hit(retired, 0.93, slug="database"),
                                   point_hit(successor, 0.81, slug="database")],
                                  store_of=lambda: m.store)
    entry = ParsedEntry("decision", "draft", 1, 10)
    entry.axiom = "A draft about the database."
    index: Dict[str, Tuple[str, float]] = {}
    m._review_neighbors(entry, set(), gathered_index=index)
    assert index == {"database": ("database", 0.81)}


def test_a_renamed_decision_is_gathered_screened_and_judged_under_its_current_slug(
        ws) -> None:
    """The point's payload says ``use-sqlite``; the node is ``sqlite-store``.

    The candidate carries the node's slug, so the own-slug drop and a declared target
    named by the current slug both fire, and the judge's prompt names the current slug
    (a separate row from P1, which holds no rename).
    """
    config, m = ws
    m.record_decision_entry("Use SQLite for the store.", "rej", ["db"], slug="use-sqlite")
    node_id = m.store.resolve_slug("use-sqlite")[0]
    assert m.amend_commentary("use-sqlite", {"slug": "sqlite-store"})["status"] == "amended"
    vector = CannedVector([point_hit(node_id, 0.9, slug="use-sqlite")], store=m.store)

    gathered = gather_candidates("A draft.", embed_provider=_Embed(), vector_store=vector,
                                 store=m.store)
    assert [c.slug for c in gathered.live] == ["sqlite-store"]
    assert screen_candidates(gathered.live, declared_targets=set(),
                             own_slug="sqlite-store") == []
    assert screen_candidates(gathered.live, declared_targets={"sqlite-store"},
                             own_slug="draft") == []

    prompts: List[Any] = []

    def judge(prompt: Any) -> Any:
        prompts.append(prompt)
        return _execution([(s, True, 0.1, "Fine.") for s in prompt.candidate_slugs])

    draft = ParsedEntry("decision", "draft", 1, 10)
    draft.axiom = "Store data in SQLite."
    draft.rejected_paths = "rej"
    result = run_conflict_check(draft, embed_provider=_Embed(), vector_store=vector,
                                store=m.store, judge=judge)
    assert not isinstance(result, Unavailable)
    assert list(prompts[0].candidate_slugs) == ["sqlite-store"]
    assert "sqlite-store" in prompts[0].user and "use-sqlite" not in prompts[0].user


# --------------------------------------------------------------------------- #
# Live-fill: the ten-to-one neighbourhood (constraint 4, the gather's half)
# --------------------------------------------------------------------------- #

def _ten_to_one(store: GraphStore, *, retired_low: float) -> Tuple[List[str], List[str], List[Dict]]:
    """200 retired decisions outranking 20 live ones, all in one neighbourhood.

    The retired scores fall evenly from 0.99 to ``retired_low``; the live ones sit below.
    """
    retired = [_commit(store, f"old-{i:03d}", f"Retired decision number {i}.")
               for i in range(200)]
    _commit(store, "killer", "Retires the two hundred.",
            supersedes=[f"old-{i:03d}" for i in range(200)])
    live = [_commit(store, f"live-{i:02d}", f"Live decision number {i}.") for i in range(20)]
    step = (0.99 - retired_low) / 199
    hits = [point_hit(node_id, 0.99 - step * i) for i, node_id in enumerate(retired)]
    low = min(retired_low - 0.01, 0.75)
    hits += [point_hit(node_id, low - 0.001 * i) for i, node_id in enumerate(live)]
    return retired, live, hits


def test_ten_to_one_fills_twenty_live_at_the_raw_bound(store: GraphStore) -> None:
    """At ``CONFLICT_RAW_BOUND`` the gather holds the 20 live; the 200 ride ``retired``.

    Break B1 (the bound back at ~20 raw) reds this row and P2.
    """
    retired, live, hits = _ten_to_one(store, retired_low=0.80)
    gathered = _gather(hits, store)
    assert [c.node["id"] for c in gathered.live] == live
    assert gathered.end != SHORT_BOUND
    assert not gathered.cut_above(CONFLICT_SIMILARITY_FLOOR)
    assert [h.node["id"] for h in gathered.retired] == retired
    assert [h.rank for h in gathered.retired] == list(range(1, 201))


@pytest.mark.parametrize("retired_low, cut", [(0.80, True), (0.70, False)])
def test_ten_to_one_under_a_200_bound_is_cut_only_above_the_floor(
        store: GraphStore, monkeypatch: pytest.MonkeyPatch, retired_low: float,
        cut: bool) -> None:
    """With the bound moved to 200 every slot is retired: short of 20 live.

    It is a cut above the floor exactly when the last raw score (the 200th retired
    point's) is at or above it. The bound is read at call time, so this monkeypatch
    reaches the gather — bound as a default argument it would not, and the row would
    pass vacuously.
    """
    monkeypatch.setattr(conflict, "CONFLICT_RAW_BOUND", 200)
    _, _, hits = _ten_to_one(store, retired_low=retired_low)
    gathered = _gather(hits, store)
    assert len(gathered.live) < CONFLICT_OVERFETCH_LIMIT
    assert gathered.end == SHORT_BOUND and gathered.bound == 200
    assert gathered.last_score == pytest.approx(retired_low)
    assert gathered.cut_above(CONFLICT_SIMILARITY_FLOOR) is cut


# --------------------------------------------------------------------------- #
# cut_above's boundary
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("delta, cut", [(0.0, True), (0.001, True), (-0.001, False)])
def test_cut_above_is_inclusive_at_the_floor(delta: float, cut: bool) -> None:
    floor = CONFLICT_SIMILARITY_FLOOR
    assert _gathering(SHORT_BOUND, floor + delta).cut_above(floor) is cut


@pytest.mark.parametrize("end", [EDGE_FOUND, EDGE_UNMEASURED, EXHAUSTED])
def test_a_full_or_exhausted_gather_is_never_cut(end: str) -> None:
    """``edge_unmeasured`` is exactly 20 live at the bound: full, not cut."""
    assert not _gathering(end, 0.99, live=CONFLICT_OVERFETCH_LIMIT).cut_above(0.0)


def test_cut_above_reads_the_last_raw_score_not_the_last_live_one(store: GraphStore,
                                                                    monkeypatch) -> None:
    """The last live candidate scores above the floor; the last raw point below it.

    Scores fall down the fetch, so only a raw tail at or above the floor can hide an
    above-floor live neighbour. Break B4 (reading the last live score) reds this row.
    """
    monkeypatch.setattr(conflict, "CONFLICT_RAW_BOUND", CONFLICT_OVERFETCH_LIMIT + 1)
    live = _commit(store, "live-one", "The one live decision.")
    floor = CONFLICT_SIMILARITY_FLOOR
    hits = [point_hit(live, 0.95)]
    hits += [orphan_hit(0.94 - (0.94 - (floor - 0.05)) * i / CONFLICT_OVERFETCH_LIMIT,
                        tag=f"o{i}") for i in range(CONFLICT_OVERFETCH_LIMIT)]
    gathered = _gather(hits, store)
    assert gathered.end == SHORT_BOUND
    assert gathered.live[-1].score >= floor > gathered.last_score
    assert gathered.cut_above(floor) is False


def test_an_empty_fetch_is_exhausted_and_never_cut(store: GraphStore) -> None:
    gathered = _gather([], store)
    assert gathered.end == EXHAUSTED and gathered.last_score is None
    assert not gathered.cut_above(0.0)


# --------------------------------------------------------------------------- #
# The contract: a wrong double fails loud, never as Unavailable
# --------------------------------------------------------------------------- #

class _Partial:
    """A store providing only the methods named."""

    def __init__(self, inner: GraphStore, *names: str,
                 stamps: Any = None) -> None:
        self._inner = inner
        for name in names:
            setattr(self, name, getattr(inner, name))
        if stamps is not None:
            self.get_modifiers_map = lambda ids: stamps


@pytest.mark.parametrize("make", [
    pytest.param(lambda s: _Partial(s, "get_modifiers_map"), id="no-resolver"),
    pytest.param(lambda s: _Partial(s, "resolve_point_ids"), id="no-stamp-reader"),
    pytest.param(lambda s: MagicMock(), id="magicmock"),
    pytest.param(lambda s: _Partial(s, "resolve_point_ids", stamps=[("not", "a dict")]),
                 id="non-dict-stamps"),
])
@pytest.mark.parametrize("hits", ["empty", "one-live"])
def test_a_store_breaking_the_contract_raises_out_of_the_gather(store: GraphStore, make,
                                                                hits: str) -> None:
    """On an empty fetch too, except the non-dict stamp map, which needs a window.

    Break B13 (a non-dict stamp map accepted) reds the non-dict leg.
    """
    node_id = _commit(store, "live-one", "The one live decision.")
    canned = [] if hits == "empty" else [point_hit(node_id, 0.9)]
    double = make(store)
    if hits == "empty" and isinstance(double, _Partial) and \
            callable(getattr(double, "get_modifiers_map", None)) and \
            callable(getattr(double, "resolve_point_ids", None)):
        pytest.skip("a non-dict stamp map is only read over a non-empty window")
    with pytest.raises(WalkContractError):
        gather_candidates("A draft.", embed_provider=_Embed(),
                          vector_store=CannedVector(canned, store=store), store=double)


# --------------------------------------------------------------------------- #
# S2 — the judge paths never read the retired partition
# --------------------------------------------------------------------------- #

JUDGE_PATHS = [
    (conflict, "run_conflict_check"),
    (check, "sweep_node"),
    (check, "iter_sweep"),
    (check, "plan_corpus_check"),
    (check, "execute_corpus_check"),
    (sync.MitosSyncManager, "_run_and_surface_conflict"),
    (cli, "_run_staged_check"),
]


@pytest.mark.parametrize("owner, name", JUDGE_PATHS, ids=[n for _, n in JUDGE_PATHS])
def test_s2_no_judge_path_reads_the_retired_partition(owner: Any, name: str) -> None:
    """``Gathering.retired`` is the record path's (7a's tombstones), never the judge's.

    Break B5 (``run_conflict_check`` screening live + retired) reds this row and P2.
    """
    source = textwrap.dedent(inspect.getsource(getattr(owner, name)))
    reads = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Attribute) and node.attr == "retired"]
    assert reads == [], f"{name} reads .retired"


def test_s2_is_not_vacuous() -> None:
    """The row's detector sees a planted read."""
    planted = "def f(g):\n    return g.live + g.retired\n"
    assert any(isinstance(n, ast.Attribute) and n.attr == "retired"
               for n in ast.walk(ast.parse(planted)))
