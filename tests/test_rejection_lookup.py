"""Phase 5c: the field-scoped lookup — ``query`` asks the rejection index alone.

``query_decisions(…, field="rejected_paths")`` and ``mitos query --field
rejected_paths`` answer *was this rejected, and by what?*: one embedding, one search
of the rejection index (the decision collection is never read), one whole resolve of
the hits' owners, ``fill_rejections`` with no floor and a bound of ``limit + 1``
items, then the window's stamps. The rejection ranking is the whole answer, shaped by
the section's own shaper; ``full_top`` thins owners, never items; a slug-shaped query
is ranked like any claim. Every exit that cannot rank items answers with a labelled
lexical match over the corpus's parsed rejected options.

The rows drive a real SQLite graph seeded from markdown blocks (the lexical arm reads
the same text the graph committed), ``FakeRejectionIndex`` bound at the read seam
(``_ranked_rejection_index``) with every point placed at a chosen cosine to the query,
and a counting decision vector store that must never be queried. No score is
hard-coded into an expectation: orders are derived from the planted scores.
"""

import inspect
import io
import json
import math
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import patch

import pytest

import mitos.cli as cli
import mitos.mcp_server as mcp_server
from mitos.cli import cmd_init, cmd_query
from mitos.config import MitosConfig
from mitos.display import RANKED_LIMIT_CEILING, rejection_matches_payload
from mitos.errors import CollectionMissingError, DatabaseError, EmbeddingError
from mitos.lexical import lexical_rejection_matches
from mitos.parser import parse_entry_stream
from mitos.provider_cause import QUOTA, provider_cause_phrase
from mitos.recall import (LOOKUP_DEGRADED_NOT_COMPUTED, LOOKUP_FIELDS,
                          RANKED_EDGE_MARGIN, REJECTION_GAP_POINTERS,
                          lookup_edge, lookup_edge_clause, lookup_omission_clause,
                          missing_graph_note, rejection_lookup)
from mitos.rejections import enumerate_rejected_paths, point_items
from mitos.retrieval_walk import (EDGE_FOUND, EDGE_UNMEASURED, EXHAUSTED, NO_FLOOR,
                                  REJECTION_LOOKUP_BUDGET_SECONDS,
                                  REJECTION_OVERFETCH_BOUND, SHORT_BOUND,
                                  RejectionLookupWalk, WalkContractError,
                                  walk_rejection_lookup)
from mitos.shell_quote import shell_flag
from mitos.store import GraphStore
from mitos.vector_store import (QdrantVectorStore, RejectionHit, RejectionSearch,
                                hash_to_uuid)

from _point_hits import CannedVector
from _rejection_index_double import FakeRejectionIndex
from conftest import resolve_like_main

DIM = 8
QUERY = "a claim about the webhook queue"

F_DB = ("1. Postgres as the webhook queue — too heavy for one consumer.\n"
        "2. Redis lists — they lose messages on a restart.\n"
        "3. SQS — a cloud lock-in for a local tool.")
F_CACHE = ("1. Memcached — no persistence across restarts.\n"
           "2. A Redis cache — another daemon to run.")
F_PROSE = "Rejected a nightly batch rebuild: it hides drift for a day."


def _query_vector() -> List[float]:
    return [1.0] + [0.0] * (DIM - 1)


def _at(score: float) -> List[float]:
    """A unit vector at cosine ``score`` to :func:`_query_vector`."""
    vector = [0.0] * DIM
    vector[0] = score
    vector[1] = math.sqrt(max(0.0, 1.0 - score * score))
    return vector


class _Embed:
    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        return _query_vector()


class _QuotaEmbed:
    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        raise EmbeddingError("429 RESOURCE_EXHAUSTED: quota exceeded")


class _StubManager:
    def __init__(self, store, embed_provider, vector_store):
        self.store = store
        self.embed_provider = embed_provider
        self.vector_store = vector_store


# --------------------------------------------------------------------------- #
# The workspace: markdown blocks committed through the sync path
# --------------------------------------------------------------------------- #

@pytest.fixture
def ws(monkeypatch) -> Iterator[MitosConfig]:
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    tmp = tempfile.mkdtemp()
    with redirect_stdout(io.StringIO()):
        cmd_init(MitosConfig(tmp))
    yield resolve_like_main(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


def _block(slug: str, rejected: str, *, supersedes: Optional[str] = None) -> str:
    lines = [f"### {slug}", "", f"**Decided:** The {slug} axiom stands on its own words.",
             f"**Rejected:** {rejected}", "**Scope:** core"]
    if supersedes:
        lines.append(f"**Supersedes:** {supersedes}")
    return "\n".join(lines) + "\n"


def _write_buffer(config: MitosConfig, blocks: List[str]) -> str:
    """Replaces the buffer's entry stream with ``blocks`` (newest first)."""
    with open(config.decisions_file, encoding="utf-8") as f:
        text = f.read()
    marker = text.index("<!-- BEGIN ENTRIES")
    head = text[:text.index("\n", marker) + 1]
    body = "\n".join(blocks)
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write(f"{head}\n{body}")
    return body


def _seed(config: MitosConfig, blocks: List[str], *, commit: bool = True) -> Dict[str, str]:
    """Writes the blocks (newest first) and commits them oldest first; slug → node id."""
    body = _write_buffer(config, blocks)
    if not commit:
        return {}
    store = GraphStore(config.db_path)
    ids: Dict[str, str] = {}
    for entry in reversed(parse_entry_stream(body, "decision", failures=[])):
        ids[entry.slug] = store.commit_parsed_entry(entry).node_id
    return ids


def _field_of(config: MitosConfig, node_id: str) -> str:
    return GraphStore(config.db_path, read_only=True).get_node(node_id)["rejected_paths"]


def _plant(index: FakeRejectionIndex, config: MitosConfig, owner: str,
           scores: Dict[int, float], *, field: Optional[str] = None) -> Dict[int, str]:
    """Puts an owner's items (its stored field, or ``field``) in the index at scores."""
    ids = {}
    for pid, item in point_items(owner, field or _field_of(config, owner)).items():
        if item.ordinal in scores:
            index.points[pid] = {"vector": _at(scores[item.ordinal]),
                                 "payload": {"owner_id": owner, "model": "m"}}
            ids[item.ordinal] = pid
    return ids


def _plant_stale(index: FakeRejectionIndex, owner: str, n: int, score: float) -> None:
    """Points the owner's current field does not produce (an edit after indexing)."""
    for k in range(n):
        pid = f"00000000-0000-4000-8000-{k:012d}"
        index.points[pid] = {"vector": _at(score - k * 1e-6),
                             "payload": {"owner_id": owner, "model": "m"}}


def _items(config: MitosConfig, owner: str) -> Dict[int, str]:
    """Ordinal → the whole item text the lookup serves for that owner."""
    return {item.ordinal: item.text.rstrip()
            for item in point_items(owner, _field_of(config, owner)).values()}


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #

@contextmanager
def _seam(index: Any):
    with patch.object(cli, "_ranked_rejection_index", lambda config: index), \
            patch.object(mcp_server, "_ranked_rejection_index", lambda config: index):
        yield


def _mcp(config, index, *, query=QUERY, embed=None, vector=None, store=None,
         components_raise: Optional[BaseException] = None, **kw) -> Dict[str, Any]:
    store = store or GraphStore(config.db_path, read_only=True)
    vector = vector if vector is not None else CannedVector([], store=store)
    if components_raise is not None:
        patcher = patch.object(mcp_server, "get_workspace_components",
                               side_effect=components_raise)
    else:
        patcher = patch.object(mcp_server, "get_workspace_components",
                               return_value=(store, embed or _Embed(), vector))
    with patcher, _seam(index):
        return json.loads(mcp_server.query_decisions(
            query, project=config.workspace_dir, field="rejected_paths", **kw))


def _cli(config, index, *, as_json=True, query=QUERY, embed=None, vector=None,
         store=None, components_raise: Optional[BaseException] = None, **kw):
    if components_raise is not None:
        patcher = patch.object(cli, "MitosSyncManager", side_effect=components_raise)
    else:
        store = store or GraphStore(config.db_path, read_only=True)
        vector = vector if vector is not None else CannedVector([], store=store)
        patcher = patch.object(cli, "MitosSyncManager",
                               return_value=_StubManager(store, embed or _Embed(), vector))
    buf = io.StringIO()
    with patcher, _seam(index), redirect_stdout(buf):
        cmd_query(config, query, as_json=as_json, field="rejected_paths", **kw)
    out = buf.getvalue()
    return json.loads(out) if as_json else out


JSON_DRIVERS = {"mcp": _mcp, "cli": _cli}


def _call(driver, config, index, **kw):
    return JSON_DRIVERS[driver](config, index, **kw)


def _arg(surface: str, name: str, n: int) -> str:
    return f"{name}={n}" if surface == "mcp" else f"--{name.replace('_', '-')} {n}"


# --------------------------------------------------------------------------- #
# The walk: its reads, its ends, its failures
# --------------------------------------------------------------------------- #

class _SpyGraph:
    """Forwards the walk's three reads to a real store and records each batch."""

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
        if self.fault == "ends":
            raise DatabaseError("planted")
        return self._inner.get_live_ends(node_ids)


def _walk(config, index, *, limit=5, graph=None, **kw) -> RejectionLookupWalk:
    graph = graph or GraphStore(config.db_path, read_only=True)
    return walk_rejection_lookup(_query_vector(), index=index, graph=graph,
                                 items_of=point_items, limit=limit, **kw)


def test_walk_makes_one_search_one_owner_resolve_one_stamp_read(ws) -> None:
    ids = _seed(ws, [_block("db", F_DB), _block("cache", F_CACHE)])
    index = FakeRejectionIndex(present=True)
    _plant(index, ws, ids["db"], {1: .95, 2: .90})
    _plant(index, ws, ids["cache"], {2: .93})
    graph = _SpyGraph(GraphStore(ws.db_path, read_only=True))

    walk = _walk(ws, index, graph=graph)

    assert index.calls == [("search", {"limit": REJECTION_OVERFETCH_BOUND,
                                       "budget_seconds": REJECTION_LOOKUP_BUDGET_SECONDS})]
    owners = [hash_to_uuid(i) for i in (ids["db"], ids["cache"])]
    assert graph.resolved == [owners]
    assert graph.stamped == [[ids["db"], ids["cache"]]]
    assert graph.ended == []  # no retired owner in the window
    assert [(m.owner["slug"], m.ordinal) for m in walk.window] == [
        ("db", 1), ("cache", 2), ("db", 2)]
    assert walk.state == "searched" and walk.end == EXHAUSTED and walk.edge is None


def test_walk_fills_limit_plus_one_the_last_is_the_edge(ws) -> None:
    ids = _seed(ws, [_block("db", F_DB)])
    index = FakeRejectionIndex(present=True)
    _plant(index, ws, ids["db"], {1: .95, 2: .90, 3: .85})

    walk = _walk(ws, index, limit=2)

    assert [m.ordinal for m in walk.window] == [1, 2]
    assert walk.edge is not None and walk.edge.ordinal == 3
    assert walk.end == EDGE_FOUND and walk.short is False


def test_walk_collapses_an_items_parts_into_its_best_part(ws) -> None:
    ids = _seed(ws, [_block("db", F_DB)])
    owner = ids["db"]
    field = _field_of(ws, owner)
    real = point_items(owner, field)
    first = next(item for item in real.values() if item.ordinal == 1)
    second = next(item for item in real.values() if item.ordinal == 2)

    def items_of(owner_id, text):
        # Item 1 has two parts here: its own point and one more.
        out = dict(point_items(owner_id, text))
        out["11111111-1111-4111-8111-111111111111"] = first
        return out

    index = FakeRejectionIndex(present=True)
    _plant(index, ws, owner, {1: .90, 2: .85})
    index.points["11111111-1111-4111-8111-111111111111"] = {
        "vector": _at(.95), "payload": {"owner_id": owner, "model": "m"}}

    walk = walk_rejection_lookup(_query_vector(), index=index,
                                 graph=GraphStore(ws.db_path, read_only=True),
                                 items_of=items_of, limit=5)

    assert [m.ordinal for m in walk.window] == [1, 2]
    assert walk.window[0].point_id == "11111111-1111-4111-8111-111111111111"
    assert walk.window[0].score == pytest.approx(.95)
    assert walk.window[0].item == first.text.rstrip()
    assert walk.window[1].item == second.text.rstrip()


def test_walk_a_full_window_whose_edge_was_not_fetched_is_unmeasured_not_short(ws) -> None:
    ids = _seed(ws, [_block("db", F_DB)])
    index = FakeRejectionIndex(present=True)
    _plant(index, ws, ids["db"], {1: .95, 2: .94})
    _plant_stale(index, ids["db"], 3, .90)

    walk = _walk(ws, index, limit=2, fetch=4)

    assert len(walk.window) == 2 and walk.edge is None
    assert walk.end == EDGE_UNMEASURED
    assert walk.short is False


def test_walk_a_short_window_over_a_full_fetch_is_short(ws) -> None:
    ids = _seed(ws, [_block("db", F_DB)])
    index = FakeRejectionIndex(present=True)
    _plant(index, ws, ids["db"], {1: .95})
    _plant_stale(index, ids["db"], 3, .90)

    walk = _walk(ws, index, limit=2, fetch=4)

    assert [m.ordinal for m in walk.window] == [1]
    assert walk.end == SHORT_BOUND and walk.short is True
    assert walk.dropped["stale"] == 3


def test_walk_has_no_floor(ws) -> None:
    ids = _seed(ws, [_block("db", F_DB)])
    index = FakeRejectionIndex(present=True)
    _plant(index, ws, ids["db"], {1: .05, 2: -0.5})
    assert NO_FLOOR == float("-inf")
    assert [m.ordinal for m in _walk(ws, index).window] == [1, 2]


def test_walk_an_overrun_walks_nothing(ws) -> None:
    _seed(ws, [_block("db", F_DB)])
    index = FakeRejectionIndex(present=True)
    index.over_budget = True
    graph = _SpyGraph(GraphStore(ws.db_path, read_only=True))

    walk = _walk(ws, index, graph=graph)

    assert walk.state == "over_budget" and walk.window == [] and walk.returned == 0
    assert graph.resolved == [] and graph.stamped == []


@pytest.mark.parametrize("knob", ["absent", "unreachable", "refuse_search"])
def test_walk_lets_the_search_fault_propagate(ws, knob) -> None:
    """The search is the answer: no fault becomes an empty window (G7)."""
    index = FakeRejectionIndex(present=knob != "absent")
    if knob != "absent":
        setattr(index, knob, True)
    expected = CollectionMissingError if knob == "absent" else Exception
    with pytest.raises(expected) as raised:
        _walk(ws, index)
    assert not isinstance(raised.value, WalkContractError)


def test_walk_a_stamp_fault_costs_the_stamps_alone(ws) -> None:
    ids = _seed(ws, [_block("db", F_DB)])
    index = FakeRejectionIndex(present=True)
    _plant(index, ws, ids["db"], {1: .95})
    graph = _SpyGraph(GraphStore(ws.db_path, read_only=True))
    graph.fault = "stamps"

    walk = _walk(ws, index, graph=graph)

    assert walk.stamps_unread is True and walk.stamps == {} and len(walk.window) == 1


def test_walk_reads_live_ends_over_the_windows_retired_owners_only(ws) -> None:
    ids = _seed(ws, [_block("new", F_CACHE, supersedes="old"), _block("old", F_DB)])
    old_id = ids["old"]
    index = FakeRejectionIndex(present=True)
    _plant(index, ws, old_id, {1: .95})
    _plant(index, ws, ids["new"], {1: .90})
    graph = _SpyGraph(GraphStore(ws.db_path, read_only=True))

    walk = _walk(ws, index, graph=graph)

    assert [m.live for m in walk.window] == [False, True]
    assert graph.ended == [[old_id]]


class _NoSearch:
    pass


def test_walk_contract_faults_are_loud(ws) -> None:
    graph = GraphStore(ws.db_path, read_only=True)
    with pytest.raises(WalkContractError):
        walk_rejection_lookup(_query_vector(), index=_NoSearch(), graph=graph,
                              items_of=point_items, limit=5)
    for limit in (0, REJECTION_OVERFETCH_BOUND, True):
        with pytest.raises(WalkContractError):
            walk_rejection_lookup(_query_vector(), index=FakeRejectionIndex(present=True),
                                  graph=graph, items_of=point_items, limit=limit)


class _BadAnswer(FakeRejectionIndex):
    def search(self, vector, limit, *, budget_seconds):
        return RejectionSearch(searched=True, hits=[RejectionHit("p", None, "high")])


def test_walk_a_malformed_answer_is_a_contract_error(ws) -> None:
    with pytest.raises(WalkContractError):
        _walk(ws, _BadAnswer(present=True))


# --------------------------------------------------------------------------- #
# The healthy answer, on both surfaces
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", list(JSON_DRIVERS))
def test_the_answer_is_the_item_ranking_alone(ws, driver) -> None:
    ids = _seed(ws, [_block("db", F_DB), _block("cache", F_CACHE), _block("prose", F_PROSE)])
    index = FakeRejectionIndex(present=True)
    scores = {(ids["db"], 1): .70, (ids["db"], 2): .95, (ids["cache"], 2): .90,
              (ids["prose"], 1): .80, (ids["db"], 3): .60}
    for (owner, ordinal), score in scores.items():
        _plant(index, ws, owner, {ordinal: score})
    store = GraphStore(ws.db_path, read_only=True)
    vector = CannedVector([], store=store)

    out = _call(driver, ws, index, store=store, vector=vector, limit=4)

    expected = sorted(scores, key=lambda k: -scores[k])[:4]
    got = [(m["id"], m["ordinal"]) for m in out["rejection_matches"]]
    assert got == expected
    assert [m["score"] for m in out["rejection_matches"]] == pytest.approx(
        [scores[k] for k in expected])
    texts = {owner: _items(ws, owner) for owner in ids.values()}
    assert [m["item"] for m in out["rejection_matches"]] == [
        texts[owner][ordinal] for owner, ordinal in expected]
    assert out["field"] == "rejected_paths" and out["query"] == QUERY
    assert "matches" not in out and "retired_partition" not in out
    assert out["confidence"] == "strong"
    assert vector.queries == 0  # the decision collection is never read
    assert {m["letter"] for m in out["rejection_matches"]} <= {"here", "section"}


@pytest.mark.parametrize("driver", list(JSON_DRIVERS))
def test_a_slug_shaped_query_is_ranked_never_dereferenced(ws, driver) -> None:
    ids = _seed(ws, [_block("db", F_DB)])
    index = FakeRejectionIndex(present=True)
    _plant(index, ws, ids["db"], {2: .9})
    store = GraphStore(ws.db_path, read_only=True)
    slug_reads = []
    original = store.get_node_by_slug
    store.get_node_by_slug = lambda *a, **k: slug_reads.append(a) or original(*a, **k)

    out = _call(driver, ws, index, query="db", store=store)

    assert [m["ordinal"] for m in out["rejection_matches"]] == [2]
    assert "axiom" not in out and "mechanisms" not in out
    assert "thinning_not_applied" not in out
    assert slug_reads == []


def test_without_field_the_exact_slug_exit_still_answers(ws) -> None:
    _seed(ws, [_block("db", F_DB)])
    store = GraphStore(ws.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, _Embed(), CannedVector([], store=store))):
        out = json.loads(mcp_server.query_decisions("db", project=ws.workspace_dir))
    assert out["slug"] == "db" and "rejected_paths" in out and "field" not in out


@pytest.mark.parametrize("driver", list(JSON_DRIVERS))
def test_an_owner_letters_once_and_full_top_thins_owners_never_items(ws, driver) -> None:
    ids = _seed(ws, [_block("db", F_DB), _block("cache", F_CACHE)])
    index = FakeRejectionIndex(present=True)
    _plant(index, ws, ids["db"], {1: .95, 2: .93, 3: .91})
    _plant(index, ws, ids["cache"], {1: .94})

    whole = _call(driver, ws, index)
    letters = [(m["slug"], m["letter"]) for m in whole["rejection_matches"]]
    assert letters == [("db", "here"), ("cache", "here"), ("db", "section"),
                       ("db", "section")]
    assert all("rejected_paths" in m for m in whole["rejection_matches"]
               if m["letter"] == "here")
    assert all("axiom" not in m for m in whole["rejection_matches"]
               if m["letter"] == "section")
    assert "rejected_paths_withheld" not in whole  # every owner whole

    one = _call(driver, ws, index, full_top=1)
    here = {m["slug"]: m for m in one["rejection_matches"] if m["letter"] == "here"}
    assert "rejected_paths" in here["db"] and "rejected_paths" not in here["cache"]
    assert [m["item"] for m in one["rejection_matches"]] == [
        m["item"] for m in whole["rejection_matches"]]
    # cache shows item 1 of its 2: its reasoning was withheld.
    assert one["rejected_paths_withheld"] == 1

    brief = _call(driver, ws, index, brief=True)
    assert all("rejected_paths" not in m for m in brief["rejection_matches"])
    assert all(m["item"] for m in brief["rejection_matches"])
    # db shows all three of its items, so only cache counts.
    assert brief["rejected_paths_withheld"] == 1


@pytest.mark.parametrize("driver", list(JSON_DRIVERS))
def test_a_stale_item_is_never_served_and_a_short_window_says_so(ws, driver) -> None:
    ids = _seed(ws, [_block("db", F_DB)])
    index = FakeRejectionIndex(present=True)
    _plant(index, ws, ids["db"], {1: .95})
    # The field before an edit: an item the owner no longer states.
    _plant(index, ws, ids["db"], {1: .99}, field="1. A queue in a spreadsheet — no.")
    _plant_stale(index, ids["db"], REJECTION_OVERFETCH_BOUND, .90)

    out = _call(driver, ws, index)

    assert [m["ordinal"] for m in out["rejection_matches"]] == [1]
    assert out["rejection_matches"][0]["item"] == _items(ws, ids["db"])[1]
    assert out["rejection_section"] == "short"
    assert out["rejection_dropped"] == REJECTION_OVERFETCH_BOUND - 1
    assert "Verification dropped" in out["note"]


@pytest.mark.parametrize("driver", list(JSON_DRIVERS))
def test_a_full_window_with_an_unfetched_edge_is_unmeasured_and_not_short(ws, driver) -> None:
    ids = _seed(ws, [_block("db", F_DB)])
    index = FakeRejectionIndex(present=True)
    _plant(index, ws, ids["db"], {1: .95, 2: .94})
    _plant_stale(index, ids["db"], REJECTION_OVERFETCH_BOUND, .90)

    out = _call(driver, ws, index, limit=2)

    assert len(out["rejection_matches"]) == 2
    assert "rejection_section" not in out and "rejection_dropped" not in out
    assert "is unmeasured: the read fetched no further rejected option" in out["note"]


@pytest.mark.parametrize("driver", list(JSON_DRIVERS))
def test_the_index_running_out_is_said(ws, driver) -> None:
    ids = _seed(ws, [_block("db", F_DB)])
    index = FakeRejectionIndex(present=True)
    _plant(index, ws, ids["db"], {1: .9, 2: .8})
    out = _call(driver, ws, index, limit=5)
    # Each surface spells the argument its own way (`--limit 5` on the CLI).
    assert ("reached the end of the rejection index before it filled "
            f"{_arg(driver, 'limit', 5)}.") in out["note"]
    assert "rejection_section" not in out


# G3: the edge's lever is counted in owners.

def _cluster(ws, index, owners_items, *, edge_owner, edge_ordinal=1):
    """Plants descending scores 0.95, 0.94, … then an edge item inside the margin."""
    score = .95
    for owner, ordinal in owners_items:
        _plant(index, ws, owner, {ordinal: score})
        score -= 0.01
    _plant(index, ws, edge_owner, {edge_ordinal: score + 0.01 - RANKED_EDGE_MARGIN / 4})


@pytest.mark.parametrize("driver", list(JSON_DRIVERS))
def test_the_edge_lever_counts_owners_at_the_default(ws, driver) -> None:
    ids = _seed(ws, [_block("db", F_DB), _block("cache", F_CACHE), _block("prose", F_PROSE)])
    index = FakeRejectionIndex(present=True)
    db, cache = ids["db"], ids["cache"]
    _cluster(ws, index, [(db, 1), (db, 2), (cache, 1), (db, 3), (cache, 2)],
             edge_owner=ids["prose"])

    out = _call(driver, ws, index, limit=5)

    assert "ended mid-cluster at" in out["note"]
    assert f"with {_arg(driver, 'full_top', 2)} the decisions it adds" in out["note"]
    assert _arg(driver, "full_top", 5) not in out["note"]


@pytest.mark.parametrize("driver", list(JSON_DRIVERS))
def test_the_edge_lever_is_capped_by_owners_under_full_top(ws, driver) -> None:
    ids = _seed(ws, [_block("db", F_DB), _block("cache", F_CACHE)])
    index = FakeRejectionIndex(present=True)
    db = ids["db"]
    _cluster(ws, index, [(db, 1), (db, 2), (db, 3)], edge_owner=ids["cache"])

    out = _call(driver, ws, index, limit=3, full_top=2)

    assert f"with {_arg(driver, 'full_top', 1)} the decisions it adds" in out["note"]


@pytest.mark.parametrize("brief", [False, True])
@pytest.mark.parametrize("driver", list(JSON_DRIVERS))
def test_an_all_retired_window_names_limit_alone(ws, driver, brief) -> None:
    ids = _seed(ws, [_block("new", F_PROSE, supersedes="old"), _block("old", F_DB)])
    index = FakeRejectionIndex(present=True)
    old = ids["old"]
    _cluster(ws, index, [(old, 1), (old, 2)], edge_owner=old, edge_ordinal=3)

    out = _call(driver, ws, index, limit=2, **({"brief": True} if brief else {}))

    assert [m["letter"] for m in out["rejection_matches"]] == ["retired", "retired"]
    assert out["rejection_matches"][0]["superseded_by"] == ["new"]
    assert "the next rejected option scores as close as the last; a higher limit " \
           "reaches it." in out["note"]
    assert "full_top" not in out["note"] and "--full-top" not in out["note"]
    assert "rejected_paths_withheld" not in out  # a handle is never a thinned owner


def test_the_edge_at_the_ceiling_names_no_lever() -> None:
    class _W:
        def __init__(self, n):
            self.window = [type("M", (), {"score": .9})() for _ in range(n)]
            self.edge = type("M", (), {"score": .9})()
            self.end = EDGE_FOUND
            self.limit = n

    edge = lookup_edge(_W(RANKED_LIMIT_CEILING), owners=0, margin=RANKED_EDGE_MARGIN,
                       ceiling=RANKED_LIMIT_CEILING, full_top=None)
    clause = lookup_edge_clause(edge, ceiling=RANKED_LIMIT_CEILING, surface="mcp")
    assert clause.endswith("scores as close as the last.")


@pytest.mark.parametrize("driver", list(JSON_DRIVERS))
def test_a_stamp_fault_is_stated(ws, driver) -> None:
    ids = _seed(ws, [_block("db", F_DB)])
    index = FakeRejectionIndex(present=True)
    _plant(index, ws, ids["db"], {1: .9})
    store = GraphStore(ws.db_path, read_only=True)

    def boom(_ids):
        raise DatabaseError("planted")

    store.get_modifiers_map = boom
    out = _call(driver, ws, index, store=store)
    assert out["stamps_unread"] is True
    assert "Modifier stamps could not be read" in out["note"]


def test_the_band_note_on_items_claims_no_verdict() -> None:
    """G6: a strong item score is closeness, never "a real hit"."""
    from mitos.recall import _lookup_band_note
    note = _lookup_band_note("strong", top_score=.9, result_count=1, surface="mcp",
                             config=type("C", (), {"project": "p"})())
    assert "real hit" not in note and "is rejected" not in note
    assert "judge whether that rejection covers your option" in note


# --------------------------------------------------------------------------- #
# The empty answers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", list(JSON_DRIVERS))
def test_an_absent_index_over_an_empty_expected_set_is_the_healthy_empty_answer(
        ws, driver) -> None:
    out = _call(driver, ws, FakeRejectionIndex(present=False))
    assert out["confidence"] == "none"
    assert out["note"].startswith("No rejected option ranked — this ranking returned nothing.")
    assert "rejection_matches" not in out and "degraded" not in out
    assert "rejection_section" not in out and out["field"] == "rejected_paths"


@pytest.mark.parametrize("present", [False, True])
@pytest.mark.parametrize("driver", list(JSON_DRIVERS))
def test_an_empty_answer_over_an_unbuilt_graph_says_so(ws, driver, present) -> None:
    """W31 on both healthy empty exits: the absent collection over nothing expected,
    and an empty window over an existing one."""
    _seed(ws, [_block("db", F_DB)], commit=False)
    out = _call(driver, ws, FakeRejectionIndex(present=present))
    assert out["note"] == missing_graph_note(driver, ws)
    assert out["confidence"] == "none" and "degraded" not in out


# --------------------------------------------------------------------------- #
# The lexical answer: every exit that cannot rank items
# --------------------------------------------------------------------------- #

def _gap_heal(surface: str, config) -> str:
    return REJECTION_GAP_POINTERS[surface]["build"].format(
        selector=shell_flag("-p", config.project))


@pytest.mark.parametrize("driver", list(JSON_DRIVERS))
def test_an_absent_index_over_stated_items_answers_lexically_with_the_gap(ws, driver) -> None:
    ids = _seed(ws, [_block("db", F_DB), _block("cache", F_CACHE)])

    out = _call(driver, ws, FakeRejectionIndex(present=False), query="redis restart")

    assert out["degraded"] == "lexical" and out["rejection_section"] == "gap"
    assert out["not_computed"] == list(LOOKUP_DEGRADED_NOT_COMPUTED) == ["band", "edge"]
    assert "confidence" not in out
    assert _gap_heal(driver, ws) in out["degraded_reason"]
    assert out["note"].count(_gap_heal(driver, ws)) == 1  # the heal is stated once
    assert "no rejected option was searched" not in out["note"]
    matches = out["rejection_matches"]
    assert all("score" not in m for m in matches)
    assert (matches[0]["id"], matches[0]["ordinal"]) == (ids["db"], 2)
    assert matches[0]["item"] == _items(ws, ids["db"])[2]
    assert matches[0]["matched_terms"] == ["redis", "restart"]


def test_the_mcp_gap_names_no_command(ws) -> None:
    _seed(ws, [_block("db", F_DB)])
    out = _mcp(ws, FakeRejectionIndex(present=False), query="redis")
    assert "mitos reconcile" not in json.dumps(out)


@pytest.mark.parametrize("cause", ["over_budget", "unreachable", "refuse_search",
                                   "embedding", "no-index", "graph"])
@pytest.mark.parametrize("driver", list(JSON_DRIVERS))
def test_every_unrankable_exit_answers_lexically_with_a_calm_reason(ws, driver, cause) -> None:
    ids = _seed(ws, [_block("db", F_DB)])
    index: Any = FakeRejectionIndex(present=True)
    _plant(index, ws, ids["db"], {1: .9})
    kw: Dict[str, Any] = {"query": "redis lists"}
    if cause in ("over_budget", "unreachable", "refuse_search"):
        setattr(index, cause, True)
    elif cause == "embedding":
        kw["embed"] = _QuotaEmbed()
    elif cause == "no-index":
        index = None
    elif cause == "graph":
        kw["components_raise"] = DatabaseError(
            "This graph predates the V1a schema (a prototype layout was detected).")

    out = _call(driver, ws, index, **kw)

    assert out["degraded"] == "lexical" and "confidence" not in out
    assert "rejection_section" not in out
    reason = out["degraded_reason"]
    expected = {"over_budget": lambda r: r == "the rejection search ran past its time budget",
                "unreachable": lambda r: r == "Qdrant unavailable",
                "refuse_search": lambda r: r == "Qdrant unavailable",
                "embedding": lambda r: r == provider_cause_phrase(QUOTA),
                "no-index": lambda r: r == "embeddings/Qdrant unavailable",
                "graph": lambda r: "V1a" in r}[cause]
    assert expected(reason), reason
    assert [(m["id"], m["ordinal"]) for m in out["rejection_matches"]] == [(ids["db"], 2)]
    if cause == "graph":
        assert out["stamps_unavailable"] is True
        assert "Rejected options of retired decisions are not listed" not in out["note"]
    else:
        assert "stamps_unavailable" not in out
        assert out["rejection_matches"][0]["state"] == "active"


@pytest.mark.parametrize("driver", list(JSON_DRIVERS))
def test_a_resolver_fault_answers_lexically_never_a_clean_empty(ws, driver) -> None:
    ids = _seed(ws, [_block("db", F_DB)])
    index = FakeRejectionIndex(present=True)
    _plant(index, ws, ids["db"], {1: .9})
    store = GraphStore(ws.db_path, read_only=True)

    def boom(_ids):
        raise DatabaseError("planted resolver fault")

    store.resolve_point_ids = boom
    out = _call(driver, ws, index, store=store, query="redis lists")
    assert out["degraded"] == "lexical" and "confidence" not in out
    assert "planted" not in out["degraded_reason"]  # classified, never the raw text
    assert [(m["id"], m["ordinal"]) for m in out["rejection_matches"]] == [(ids["db"], 2)]


def test_the_lexical_match_is_by_item_never_the_raw_field(ws) -> None:
    ids = _seed(ws, [_block("db", F_DB)])
    out = lexical_rejection_matches("postgres restart", corpus_paths=[ws.decisions_file],
                                    store=None, limit=None, full_top=None)
    items = _items(ws, ids["db"])
    assert [(m["ordinal"], m["item"], m["matched_terms"]) for m in out["matches"]] == [
        (1, items[1], ["postgres"]), (2, items[2], ["restart"])]
    assert out["matches"][0]["letter"] == "here" and out["matches"][1]["letter"] == "section"
    assert out["item_counts"] == {ids["db"]: 3}


def test_the_lexical_match_lists_one_of_two_byte_identical_items(ws) -> None:
    field = "1. Keep the old cron job — it drifts.\n2. Keep the old cron job — it drifts."
    ids = _seed(ws, [_block("cron", field)])
    assert len(enumerate_rejected_paths(_field_of(ws, ids["cron"])).items) == 2
    out = lexical_rejection_matches("cron", corpus_paths=[ws.decisions_file],
                                    store=GraphStore(ws.db_path, read_only=True),
                                    limit=None, full_top=None)
    assert [m["ordinal"] for m in out["matches"]] == [1]
    assert out["item_counts"] == {ids["cron"]: 1}


def test_the_lexical_match_drops_retired_owners_when_the_graph_reads(ws) -> None:
    ids = _seed(ws, [_block("new", F_CACHE, supersedes="old"), _block("old", F_DB)])
    store = GraphStore(ws.db_path, read_only=True)
    read = lexical_rejection_matches("redis", corpus_paths=[ws.decisions_file], store=store,
                                     limit=None, full_top=None)
    assert [m["slug"] for m in read["matches"]] == ["new"]
    unread = lexical_rejection_matches("redis", corpus_paths=[ws.decisions_file], store=None,
                                       limit=None, full_top=None)
    assert {m["slug"] for m in unread["matches"]} == {"new", "old"}
    assert unread["graph_read"] is False
    assert ids["old"] in {m["id"] for m in unread["matches"]}
    assert lookup_omission_clause(retired_dropped=False).count("retired") == 0
    assert "retired decisions are not listed" in lookup_omission_clause(retired_dropped=True)


@pytest.mark.parametrize("driver", list(JSON_DRIVERS))
def test_the_lexical_withheld_count_is_over_owners_with_the_coverage_exemption(
        ws, driver) -> None:
    ids = _seed(ws, [_block("db", F_DB), _block("cache", F_CACHE)])
    absent = FakeRejectionIndex(present=False)
    # "lose" hits db 2 only, "restart" hits db 2 and cache 1, "redis" db 2, cache 2.
    whole = _call(driver, ws, absent, query="redis restart persistence")
    owners = {m["id"] for m in whole["rejection_matches"]}
    assert owners == {ids["db"], ids["cache"]}
    assert any(m["letter"] == "section" for m in whole["rejection_matches"])
    assert "rejected_paths_withheld" not in whole  # whole depth: nothing withheld

    brief = _call(driver, ws, absent, query="redis restart persistence", brief=True)
    shown = {}
    for m in brief["rejection_matches"]:
        shown.setdefault(m["id"], set()).add(m["ordinal"])
    # cache shows both of its two items: exempt. db shows two of three: withheld.
    assert shown[ids["cache"]] == {1, 2}
    assert brief["rejected_paths_withheld"] == 1


# --------------------------------------------------------------------------- #
# The refusals
# --------------------------------------------------------------------------- #

def test_mcp_refuses_another_field_before_resolving_a_project() -> None:
    out = json.loads(mcp_server.query_decisions("x", field="axiom"))
    assert set(out) == {"error"}
    assert "'axiom'" in out["error"] and "'rejected_paths'" in out["error"]


def test_cli_refuses_another_field_for_a_direct_caller(ws) -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_query(ws, "x", as_json=True, field="axiom")
    out = json.loads(buf.getvalue())
    assert "'axiom'" in out["error"] and out["project"] == ws.project
    with pytest.raises(ValueError, match="'rejected_paths'"):
        cmd_query(ws, "x", field="axiom")


def test_the_field_flag_takes_only_the_accepted_values(ws) -> None:
    parser = cli._build_parser()
    action = next(a for a in parser._subparsers._group_actions[0].choices["query"]._actions
                  if "--field" in a.option_strings)
    assert tuple(action.choices) == LOOKUP_FIELDS == ("rejected_paths",)
    err = io.StringIO()
    with redirect_stderr(err), pytest.raises(SystemExit) as code:
        parser.parse_args(["query", "--field", "axiom", "x"])
    assert code.value.code == 2 and "invalid choice" in err.getvalue()


def test_the_field_flag_reaches_cmd_query(ws) -> None:
    seen = {}
    argv = ["mitos", "query", "-p", ws.workspace_dir, "--field", "rejected_paths", "x"]
    with patch.object(cli, "cmd_query", side_effect=lambda *a, **k: seen.update(k)), \
            patch.object(sys, "argv", argv):
        cli.main()
    assert seen["field"] == "rejected_paths"


def test_surface_takes_no_field_flag(ws) -> None:
    err = io.StringIO()
    argv = ["mitos", "surface", "-p", ws.workspace_dir, "--field", "rejected_paths", "x"]
    with redirect_stderr(err), pytest.raises(SystemExit), patch.object(sys, "argv", argv):
        cli.main()
    assert "mitos query" in err.getvalue()


# --------------------------------------------------------------------------- #
# Parity and the text render
# --------------------------------------------------------------------------- #

def _exits(ws) -> Dict[str, Dict[str, Any]]:
    ids = _seed(ws, [_block("db", F_DB), _block("cache", F_CACHE)])
    healthy = FakeRejectionIndex(present=True)
    _plant(healthy, ws, ids["db"], {1: .95, 2: .90})
    _plant(healthy, ws, ids["cache"], {1: .93})
    short = FakeRejectionIndex(present=True)
    _plant(short, ws, ids["db"], {1: .95})
    _plant_stale(short, ids["db"], REJECTION_OVERFETCH_BOUND, .9)
    over = FakeRejectionIndex(present=True)
    over.over_budget = True
    return {"healthy": {"index": healthy}, "short": {"index": short},
            "empty": {"index": FakeRejectionIndex(present=True)},
            "gap": {"index": FakeRejectionIndex(present=False), "query": "redis restart"},
            "over_budget": {"index": over, "query": "redis restart"}}


_WORDED = ("note", "degraded_reason")


def test_mcp_and_cli_json_agree_on_every_exit(ws) -> None:
    for name, spec in _exits(ws).items():
        kw = {k: v for k, v in spec.items() if k != "index"}
        mcp = _mcp(ws, spec["index"], **kw)
        cli_out = _cli(ws, spec["index"], **kw)
        assert list(mcp) == list(cli_out), name
        for key in mcp:
            if key not in _WORDED:
                assert mcp[key] == cli_out[key], (name, key)


def test_the_text_render_carries_the_note_and_every_item(ws) -> None:
    for name, spec in _exits(ws).items():
        kw = {k: v for k, v in spec.items() if k != "index"}
        envelope = _cli(ws, spec["index"], **kw)
        text = _cli(ws, spec["index"], as_json=False, **kw)
        assert envelope["note"] in text, name
        for match in envelope.get("rejection_matches", []):
            assert match["item"] in text, (name, match["item"])
        assert ws.project in text


def test_the_composer_takes_no_decision_window() -> None:
    """The lookup shapes with ``window_ids=[]``: no match is ever ``window``."""
    seen = {}

    def shape(walk, **kw):
        seen.update(kw)
        return []

    class _Walk:
        window = [type("M", (), {"score": .5})()]
        edge = None
        end = EXHAUSTED
        limit = 5
        returned = 1
        dropped = {}
        item_counts = {}
        stamps = {}
        live_ends = {}
        stamps_unread = False
        short = False

    rejection_lookup(_Walk(), query="q", full_top=3, shape=shape, margin=.01,
                     ceiling=50, graph_gap=lambda: False, surface="mcp",
                     config=type("C", (), {"project": "p", "qdrant_collection": "c",
                                           "workspace_dir": "w"})())
    assert seen["window_ids"] == [] and seen["whole_owners"] == 3
    assert seen["owners_whole"] is True
    assert seen.get("stamps") == {} and rejection_matches_payload is not shape


# --------------------------------------------------------------------------- #
# The prose and the agreement rows
# --------------------------------------------------------------------------- #

def test_the_lookup_budget_is_the_decision_searchs_timeout() -> None:
    source = inspect.getsource(QdrantVectorStore.query)
    timeouts = {float(t) for t in re.findall(r"timeout=(\d+(?:\.\d+)?)", source)}
    assert timeouts == {REJECTION_LOOKUP_BUDGET_SECONDS}


def test_the_query_help_no_longer_claims_a_slug_branch() -> None:
    parser = cli._build_parser()
    choices = parser._subparsers._group_actions[0]
    help_line = next(a.help for a in choices._choices_actions if a.dest == "query")
    assert "by slug" not in help_line
    assert "--field rejected_paths" in help_line


def test_the_skill_teaches_the_lookup_and_the_cli_slug_read() -> None:
    body = cli._skill_md_text("<<FORMAT SPEC SENTINEL>>")
    line = next(l for l in body.splitlines() if l.startswith("- `query_decisions`"))
    assert "a slug you are carrying" not in body
    assert '`field="rejected_paths"` (CLI `--field rejected_paths`)' in line
    assert "ask whether it was already rejected" in line
    assert "`mitos show -p . -- <slug>`" in line
    assert body.index("<<FORMAT SPEC SENTINEL>>") < body.index(line)
