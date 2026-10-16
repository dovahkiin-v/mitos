"""Phase 2c: the four ranked surfaces and their blackout, read through the shared walk.

Every row enters the outer frame — MCP ``surface_decisions`` and ``query_decisions``
called as tools, ``cmd_surface`` and ``cmd_query`` called as verbs and read on both
encodings (``--json`` and text) — each over the store construction that surface really
uses: a read-only ``GraphStore`` on MCP, ``MitosSyncManager``'s writable one on the
CLI (CODEBASE_MAP gotcha 8). Hits are authored by node id (``point_hit``), because a
same-slug pair cannot be told apart by slug, which is the point.

What the rows pin (vision §4.2, §4.3, §5.4 lock A; plan p2c §9.2):

- a same-slug pair gives the live decision once, at its own score, and a blackout of
  the retired one describes the retired node;
- a renamed decision is served under its current slug, in the window and as a handle;
- orphans, open questions and retired hits take no slot;
- a stamp fault leaves the ranking and band standing, sets ``stamps_unread`` and ends
  the note with the clause, blackout included; a resolver fault degrades as before;
- a wrong store double raises out of every surface, never as a degraded envelope;
- each read makes one vector call, at the bound, ids only;
- CLI and MCP agree on the window and the handles.
"""

import io
import json
import sqlite3
from contextlib import redirect_stdout
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from mitos import cli, mcp_server
from mitos.cli import cmd_query, cmd_surface
from mitos.display import blackout_note
from mitos.errors import DatabaseError
from mitos.parser import ParsedEntry
from mitos.recall import stamps_unread_clause
from mitos.retrieval_walk import RANKED_OVERFETCH_BOUND, WalkContractError
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager

from _point_hits import CannedVector, orphan_hit, point_hit
from test_point_identity import same_slug_lineage
from test_surface_confidence import _FakeEmbed, _rec, offline, ws  # noqa: F401

_CLAIM = "a claim that is not any slug"

Wrap = Optional[Callable[[GraphStore], Any]]


# --------------------------------------------------------------------------- #
# Drivers: (surface, verb, encoding) → the response (a dict, or the text)
# --------------------------------------------------------------------------- #

def _mcp(tool: str, ws, vector, *, wrap: Wrap = None, limit: Optional[int] = None,
         embed: Any = None):
    config, _ = ws
    store = GraphStore(config.db_path, read_only=True)
    if wrap is not None:
        store = wrap(store)
    kw = {} if limit is None else {"limit": limit}
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, embed or _FakeEmbed(), vector)):
        if tool == "surface":
            return json.loads(mcp_server.surface_decisions(
                _CLAIM, project=config.workspace_dir, **kw))
        return json.loads(mcp_server.query_decisions(
            _CLAIM, project=config.workspace_dir, **kw))


def _cli(verb: str, ws, vector, *, as_json: bool, wrap: Wrap = None,
         limit: Optional[int] = None, embed: Any = None):
    config, _ = ws
    manager = MitosSyncManager(config)
    manager.embed_provider = embed or _FakeEmbed()
    manager.vector_store = vector
    if wrap is not None:
        manager.store = wrap(manager.store)
    buf = io.StringIO()
    with patch.object(cli, "MitosSyncManager", return_value=manager):
        with redirect_stdout(buf):
            (cmd_surface if verb == "surface" else cmd_query)(
                config, _CLAIM, as_json=as_json, limit=limit)
    out = buf.getvalue()
    return json.loads(out) if as_json else out


# name → (call, verb, surface, list key or None for text)
DRIVERS: Dict[str, Tuple[Callable, str, str, Optional[str]]] = {
    "mcp-surface": (lambda ws, v, **k: _mcp("surface", ws, v, **k), "surface", "mcp",
                    "active_decisions"),
    "mcp-query": (lambda ws, v, **k: _mcp("query", ws, v, **k), "query", "mcp", "matches"),
    "cli-surface-json": (lambda ws, v, **k: _cli("surface", ws, v, as_json=True, **k),
                         "surface", "cli", "active_decisions"),
    "cli-query-json": (lambda ws, v, **k: _cli("query", ws, v, as_json=True, **k),
                       "query", "cli", "matches"),
    "cli-surface-text": (lambda ws, v, **k: _cli("surface", ws, v, as_json=False, **k),
                         "surface", "cli", None),
    "cli-query-text": (lambda ws, v, **k: _cli("query", ws, v, as_json=False, **k),
                       "query", "cli", None),
}
JSON_DRIVERS = [d for d, (_, _, _, key) in DRIVERS.items() if key]
TEXT_DRIVERS = [d for d, (_, _, _, key) in DRIVERS.items() if not key]


def _vec(hits: List[Dict[str, Any]]) -> CannedVector:
    return CannedVector(hits)


def _id(m: MitosSyncManager, slug: str) -> str:
    ids = m.store.resolve_slug(slug)
    assert len(ids) == 1, ids
    return ids[0]


def _hit_slugs(resp, key) -> List[str]:
    return [h["slug"] for h in resp[key]]


def _text_ranks(out: str) -> List[str]:
    """`N. slug  (score …)` lines of a text render, slug only."""
    return [ln.split(". ", 1)[1].split("  (score")[0] for ln in out.splitlines()
            if ln[:1].isdigit() and "  (score " in ln]


def _commit_oq(store: GraphStore, slug: str) -> str:
    e = ParsedEntry("open_question", slug, 1, 5)
    e.topic = f"Topic for {slug}"
    e.questions_raised = [f"What about {slug}?"]
    e.scope = ["x"]
    return store.commit_parsed_entry(e).node_id


# --------------------------------------------------------------------------- #
# Same-slug lineage
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", list(DRIVERS))
def test_a_same_slug_pair_serves_the_live_decision_once_at_its_own_score(ws, driver):
    """Before 2c the retired point's 0.90 resolved by slug to the live successor,
    which was served twice — once at its retired twin's score."""
    _, m = ws
    retired, successor = same_slug_lineage(m.store)
    call, _verb, _surface, key = DRIVERS[driver]
    # Both points carry the payload slug they were written under: the shared one.
    resp = call(ws, _vec([point_hit(retired, 0.90, slug="database"),
                          point_hit(successor, 0.80, slug="database")]))
    if key:
        assert [(h["slug"], h["score"]) for h in resp[key]] == [("database", 0.80)]
        assert "all_superseded" not in resp
    else:
        assert _text_ranks(resp) == ["database"]
        assert "(score 0.800)" in resp and "(score 0.900)" not in resp


@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_a_same_slug_retired_point_alone_is_a_blackout_of_the_retired_node(ws, driver):
    _, m = ws
    retired, _successor = same_slug_lineage(m.store)
    call, _verb, _surface, key = DRIVERS[driver]
    resp = call(ws, _vec([point_hit(retired, 0.90, slug="database")]))
    assert resp[key] == []
    assert resp["all_superseded"] == [
        {"slug": "database", "state": "superseded", "superseded_by": ["database"]}]
    assert resp["note"] == blackout_note(resp["all_superseded"])


# --------------------------------------------------------------------------- #
# Rename
# --------------------------------------------------------------------------- #

def _rename(m: MitosSyncManager, old: str, new: str) -> None:
    receipt = m.amend_commentary(old, {"slug": new})
    assert receipt["status"] == "amended", receipt


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_a_renamed_decision_is_served_under_its_current_slug(ws, driver):
    """The point was written under `use-sqlite`; its payload still says so. Before
    2c the slug join found no node and the decision silently left recall."""
    _, m = ws
    _rec(m, "use-sqlite")
    node_id = _id(m, "use-sqlite")
    _rename(m, "use-sqlite", "sqlite-store")
    call, _verb, _surface, key = DRIVERS[driver]
    resp = call(ws, _vec([point_hit(node_id, 0.88, slug="use-sqlite")]))
    if key:
        assert _hit_slugs(resp, key) == ["sqlite-store"]
    else:
        assert _text_ranks(resp) == ["sqlite-store"]
        assert "1. use-sqlite" not in resp


@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_a_renamed_retired_decision_is_handled_under_its_current_slug(ws, driver):
    _, m = ws
    _rec(m, "r-old-name")
    node_id = _id(m, "r-old-name")
    _rename(m, "r-old-name", "r-current")
    _rec(m, "r-successor", supersedes="r-current")
    call, _verb, _surface, key = DRIVERS[driver]
    resp = call(ws, _vec([point_hit(node_id, 0.9, slug="r-old-name")]))
    assert resp[key] == []
    assert resp["all_superseded"] == [
        {"slug": "r-current", "state": "superseded", "superseded_by": ["r-successor"]}]


# --------------------------------------------------------------------------- #
# Live-fill: orphans, open questions and retired hits take no slot
# --------------------------------------------------------------------------- #

def _live_fill_world(m) -> List[Dict[str, Any]]:
    for slug in ("a", "b", "c", "old"):
        _rec(m, slug)
    _rec(m, "new", supersedes="old")
    oq = _commit_oq(m.store, "oq-parked")
    return [orphan_hit(0.99, slug="gone"), point_hit(_id(m, "old"), 0.95),
            point_hit(oq, 0.93), point_hit(_id(m, "a"), 0.9),
            point_hit(_id(m, "b"), 0.8), point_hit(_id(m, "c"), 0.7)]


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_orphans_questions_and_retired_hits_take_no_slot(ws, driver):
    """G3, a deliberate change: before 2c the orphan, the retired decision and the
    open question each took one of the two slots and the window came back empty."""
    _, m = ws
    hits = _live_fill_world(m)
    call, _verb, _surface, key = DRIVERS[driver]
    resp = call(ws, _vec(hits), limit=2)
    if key:
        assert _hit_slugs(resp, key) == ["a", "b"]
        assert "all_superseded" not in resp
    else:
        assert _text_ranks(resp) == ["a", "b"]


@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_a_live_decision_below_retired_ones_ends_the_old_blackout(ws, driver):
    """G4, a deliberate inversion: the top `limit` are retired and a live decision
    sits below them. Before 2c that was a blackout; the walk fills from the fetch."""
    _, m = ws
    hits = []
    for i in range(3):
        _rec(m, f"old-{i}")
        _rec(m, f"new-{i}", supersedes=f"old-{i}")
        hits.append(point_hit(_id(m, f"old-{i}"), 0.9 - i / 100))
    _rec(m, "deep-live")
    hits.append(point_hit(_id(m, "deep-live"), 0.5))
    call, _verb, _surface, key = DRIVERS[driver]
    resp = call(ws, _vec(hits), limit=2)
    assert _hit_slugs(resp, key) == ["deep-live"]
    assert "all_superseded" not in resp


@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_the_blackout_is_bounded_at_limit_closest_first(ws, driver):
    """K14's row: `limit + 2` retired, zero live → `limit` handles, best first."""
    _, m = ws
    hits = []
    for i in range(4):
        _rec(m, f"old-{i}")
        _rec(m, f"new-{i}", supersedes=f"old-{i}")
        hits.append(point_hit(_id(m, f"old-{i}"), 0.9 - i / 100))
    call, _verb, _surface, key = DRIVERS[driver]
    resp = call(ws, _vec(hits), limit=2)
    assert resp[key] == []
    assert [h["slug"] for h in resp["all_superseded"]] == ["old-0", "old-1"]


# --------------------------------------------------------------------------- #
# One vector call, at the bound, ids only
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", list(DRIVERS))
@pytest.mark.parametrize("shape", ["full", "short"])
def test_each_read_makes_one_vector_call_at_the_bound_ids_only(ws, driver, shape):
    """W11 end to end. `short` is a neighbourhood of `limit + 2` retired hits and one
    live one: the window comes back short and is never chased with a second call."""
    _, m = ws
    if shape == "full":
        hits = _live_fill_world(m)
    else:
        hits = []
        for i in range(4):
            _rec(m, f"old-{i}")
            _rec(m, f"new-{i}", supersedes=f"old-{i}")
            hits.append(point_hit(_id(m, f"old-{i}"), 0.9 - i / 100))
        _rec(m, "only-live")
        hits.append(point_hit(_id(m, "only-live"), 0.4))
    vec = _vec(hits)
    call, *_ = DRIVERS[driver]
    call(ws, vec, limit=2)
    assert vec.calls == [(RANKED_OVERFETCH_BOUND, False)]


# --------------------------------------------------------------------------- #
# The stamp fault: the ranking stands, the fault is stated
# --------------------------------------------------------------------------- #

class _Forwarding:
    """Forwards every attribute to a real store; subclasses override one method."""

    def __init__(self, inner: GraphStore) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _StampFault(_Forwarding):
    def get_modifiers_map(self, node_ids):
        raise DatabaseError("planted stamp fault")


def _real_sqlite_stamp_fault(monkeypatch) -> None:
    """The modifier query itself raises, as a locked graph's would (scout D1)."""
    def boom(self, conn, node_ids):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(GraphStore, "_modifiers_map", boom)


def _amended_world(m) -> List[Dict[str, Any]]:
    _rec(m, "base")
    _rec(m, "amender", amends="base")
    return [point_hit(_id(m, "base"), 0.9), point_hit(_id(m, "amender"), 0.85)]


def _healthy_and_faulted(ws, monkeypatch, driver, fault, hits):
    call, *_ = DRIVERS[driver]
    healthy = call(ws, _vec(hits))
    if fault == "wrapper":
        faulted = call(ws, _vec(hits), wrap=_StampFault)
    else:
        _real_sqlite_stamp_fault(monkeypatch)
        faulted = call(ws, _vec(hits))
    return healthy, faulted


@pytest.mark.parametrize("fault", ["wrapper", "real-sqlite"])
@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_a_stamp_fault_leaves_the_ranking_and_band_standing_and_says_so(
        ws, monkeypatch, driver, fault):
    """Inverts D4's characterization: before 2c MCP `surface` served these hits
    unstamped in silence, and the other three surfaces went lexical."""
    config, m = ws
    hits = _amended_world(m)
    _call, _verb, surface, key = DRIVERS[driver]
    healthy, faulted = _healthy_and_faulted(ws, monkeypatch, driver, fault, hits)

    assert healthy[key][0]["amended_by"] == ["amender"]
    assert "stamps_unread" not in healthy
    assert "degraded" not in faulted
    assert _hit_slugs(faulted, key) == _hit_slugs(healthy, key) == ["base", "amender"]
    assert [h["score"] for h in faulted[key]] == [h["score"] for h in healthy[key]]
    assert faulted["confidence"] == healthy["confidence"]
    assert all(not any(k.endswith("_by") for k in h) for h in faulted[key])
    assert faulted["stamps_unread"] is True
    clause = stamps_unread_clause(surface=surface, config=config)
    assert faulted["note"] == f"{healthy['note']} {clause}"


@pytest.mark.parametrize("driver", TEXT_DRIVERS)
def test_a_stamp_fault_ends_the_text_note_with_the_clause(ws, driver):
    config, m = ws
    hits = _amended_world(m)
    call, _verb, surface, _key = DRIVERS[driver]
    healthy = call(ws, _vec(hits))
    faulted = call(ws, _vec(hits), wrap=_StampFault)
    clause = stamps_unread_clause(surface=surface, config=config)
    assert _text_ranks(faulted) == _text_ranks(healthy) == ["base", "amender"]
    assert "amender" in healthy.split("\n2. ")[0]          # the stamp marker on base
    note_lines = [ln for ln in faulted.splitlines() if ln.startswith("→ ")]
    assert note_lines and note_lines[-1].endswith(clause)
    assert clause not in healthy


@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_a_stamp_fault_on_a_blackout_states_it_on_the_blackout_note(ws, driver):
    """K9's row: the clause comes after the blackout override, and the handle omits
    the successor it could not read."""
    config, m = ws
    _rec(m, "old")
    _rec(m, "new", supersedes="old")
    call, _verb, surface, key = DRIVERS[driver]
    resp = call(ws, _vec([point_hit(_id(m, "old"), 0.9)]), wrap=_StampFault)
    handles = [{"slug": "old", "state": "superseded"}]
    assert resp[key] == [] and resp["all_superseded"] == handles
    assert resp["stamps_unread"] is True
    clause = stamps_unread_clause(surface=surface, config=config)
    assert resp["note"] == f"{blackout_note(handles)} {clause}"


def test_a_stamp_fault_on_the_text_blackout_states_it_too(ws):
    """`cmd_query`'s blackout returns early with its own print (G8)."""
    config, m = ws
    _rec(m, "old")
    _rec(m, "new", supersedes="old")
    out = _cli("query", ws, _vec([point_hit(_id(m, "old"), 0.9)]), as_json=False,
               wrap=_StampFault)
    clause = stamps_unread_clause(surface="cli", config=config)
    assert f"{blackout_note([{'slug': 'old', 'state': 'superseded'}])} {clause}" in out


@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_a_healthy_read_carries_no_stamps_unread_key(ws, driver):
    """Constraint 5: absent means the stamps were read (computed and empty)."""
    _, m = ws
    call, *_ = DRIVERS[driver]
    resp = call(ws, _vec(_amended_world(m)))
    assert "stamps_unread" not in resp


# --------------------------------------------------------------------------- #
# Faults that are not stamp faults
# --------------------------------------------------------------------------- #

class _ResolveFault(_Forwarding):
    def resolve_point_ids(self, point_ids):
        raise DatabaseError("planted resolver fault")


@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_a_resolver_fault_degrades_as_a_store_fault_does(ws, driver):
    _, m = ws
    call, *_ = DRIVERS[driver]
    resp = call(ws, _vec(_amended_world(m)), wrap=_ResolveFault)
    assert resp["degraded"] == "lexical"
    assert "stamps_unread" not in resp


class _NoResolver(_Forwarding):
    def __getattr__(self, name: str) -> Any:
        if name == "resolve_point_ids":
            raise AttributeError(name)
        return getattr(self._inner, name)


@pytest.mark.parametrize("wrap", [lambda _store: MagicMock(), _NoResolver],
                         ids=["magicmock", "no-resolver"])
@pytest.mark.parametrize("driver", list(DRIVERS))
def test_a_wrong_store_double_raises_out_of_every_surface(ws, driver, wrap):
    """Ledger entry-002 and D3: the contract error escapes the degraded arm. On MCP
    the raise reaches FastMCP as `isError`; on the CLI it propagates to `main()`."""
    _, m = ws
    call, *_ = DRIVERS[driver]
    with pytest.raises(WalkContractError):
        call(ws, _vec(_amended_world(m)), wrap=wrap)


# --------------------------------------------------------------------------- #
# CLI ⇄ MCP parity on the window and the handles
# --------------------------------------------------------------------------- #

def _mixed_world(m) -> List[Dict[str, Any]]:
    """Live, retired, an orphan, an open question and a renamed node, amended too."""
    _rec(m, "live-a")
    _rec(m, "to-rename")
    renamed = _id(m, "to-rename")
    _rename(m, "to-rename", "renamed")
    _rec(m, "amender", amends="live-a")
    _rec(m, "old")
    _rec(m, "new", supersedes="old")
    oq = _commit_oq(m.store, "oq-parked")
    return [point_hit(_id(m, "old"), 0.97), orphan_hit(0.96, slug="gone"),
            point_hit(oq, 0.95), point_hit(renamed, 0.9, slug="to-rename"),
            point_hit(_id(m, "live-a"), 0.85), point_hit(_id(m, "amender"), 0.8)]


@pytest.mark.parametrize("verb", ["surface", "query"])
def test_cli_and_mcp_agree_on_the_window(ws, verb):
    _, m = ws
    hits = _mixed_world(m)
    key = "active_decisions" if verb == "surface" else "matches"
    mcp_resp = _mcp(verb, ws, _vec(hits), limit=3)
    cli_resp = _cli(verb, ws, _vec(hits), as_json=True, limit=3)
    assert set(mcp_resp) - {"check_notice"} == set(cli_resp) - {"check_notice"}
    assert mcp_resp[key] == cli_resp[key]
    assert _hit_slugs(mcp_resp, key) == ["renamed", "live-a", "amender"]
    assert mcp_resp[key][1]["amended_by"] == ["amender"]
    assert mcp_resp["confidence"] == cli_resp["confidence"]


@pytest.mark.parametrize("verb", ["surface", "query"])
def test_cli_and_mcp_agree_on_the_handles(ws, verb):
    _, m = ws
    retired, _successor = same_slug_lineage(m.store)
    _rec(m, "old")
    _rec(m, "new", supersedes="old")
    hits = [point_hit(retired, 0.9), orphan_hit(0.85), point_hit(_id(m, "old"), 0.8)]
    mcp_resp = _mcp(verb, ws, _vec(hits))
    cli_resp = _cli(verb, ws, _vec(hits), as_json=True)
    assert mcp_resp["all_superseded"] == cli_resp["all_superseded"] == [
        {"slug": "database", "state": "superseded", "superseded_by": ["database"]},
        {"slug": "old", "state": "superseded", "superseded_by": ["new"]},
    ]
