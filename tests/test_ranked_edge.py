"""Phase 2e: the edge clause over the shared walk, on every ranked surface and encoding.

`test_surface_confidence.py` holds the policy and the rows inverted from 4c's lever
block. These rows drive the walk's own ends through the six drivers of
`test_ranked_walk_surfaces` (MCP `surface`/`query`, the CLI's `--json` and text on
both verbs): the unmeasured edge, which needs a fetch that fills the whole bound; the
clause order beside the withheld and stamps-unread clauses; the exits that can carry
no edge; CLI⇄MCP parity; and one row entered through `cli.main()`.

Every fixture's gap is derived from `RANKED_EDGE_MARGIN` by name and guarded, and the
bound from `retrieval_walk.RANKED_OVERFETCH_BOUND`: `walk_ranked`'s `bound` default is
bound at definition time and every surface calls it without `bound=`, so patching the
module constant would change nothing.
"""

import io
import json
import sys
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional

import pytest
from unittest.mock import patch

from mitos import cli, mcp_server
from mitos.cli import cmd_query, cmd_surface
from mitos.display import blackout_note
from mitos.recall import (EDGE_KIND_MID_CLUSTER, EDGE_KIND_UNMEASURED, RANKED_EDGE_MARGIN,
                          WindowEdge, edge_clause, stamps_unread_clause, withheld_clause)
from mitos.retrieval_walk import (EDGE_UNMEASURED, RANKED_OVERFETCH_BOUND, SHORT_BOUND,
                                  walk_ranked)
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager

from _point_hits import CannedVector, orphan_hit, point_hit
from test_ranked_walk_surfaces import DRIVERS, JSON_DRIVERS, TEXT_DRIVERS, _StampFault, _id
from test_surface_confidence import (_FakeEmbed, _query_pointer, _rec, cloned, offline,  # noqa: F401
                                     ws)

_CLAIM = "a claim that is not any slug"
_SLUGS = [f"edge-{i}" for i in range(1, 7)]
_TOP = [0.9, 0.86, 0.83, 0.8, 0.78]      # strong; the fifth is the last returned


def _seed(m) -> None:
    for slug in _SLUGS:
        _rec(m, slug)


def _hits(m, sixth_gap: Optional[float]) -> List[Dict[str, Any]]:
    """Five live decisions, and a sixth ``sixth_gap`` below the fifth (None: no sixth)."""
    hits = [point_hit(_id(m, s), sc) for s, sc in zip(_SLUGS, _TOP)]
    if sixth_gap is not None:
        hits.append(point_hit(_id(m, _SLUGS[5]), _TOP[4] - sixth_gap))
    return hits


def _mid(m) -> List[Dict[str, Any]]:
    hits = _hits(m, RANKED_EDGE_MARGIN / 2)
    assert hits[4]["score"] - hits[5]["score"] < RANKED_EDGE_MARGIN  # G2 guard
    return hits


def _clear(m) -> List[Dict[str, Any]]:
    hits = _hits(m, RANKED_EDGE_MARGIN * 2)
    assert hits[4]["score"] - hits[5]["score"] >= RANKED_EDGE_MARGIN  # G2 guard
    return hits


def _unmeasured(m) -> List[Dict[str, Any]]:
    """A fetch filling the whole bound with five live decisions and orphans after them."""
    hits = _hits(m, None) + [orphan_hit(0.5, tag=f"edge-orphan-{i}")
                             for i in range(RANKED_OVERFETCH_BOUND - 5)]
    assert len(hits) == RANKED_OVERFETCH_BOUND
    return hits


def _note(resp: Any, key: Optional[str]) -> str:
    """The note a response carries: the envelope's, or the text render's last `→` line."""
    if key is not None:
        return resp["note"]
    lines = [ln for ln in resp.splitlines() if ln.startswith("→ ")]
    assert lines, resp
    return lines[-1][2:]


def _mid_clause(surface: str, limit: int = 5, held: Optional[int] = 5) -> str:
    return edge_clause(WindowEdge(EDGE_KIND_MID_CLUSTER, limit, held), surface=surface)


def _unmeasured_clause(surface: str, limit: int = 5) -> str:
    return edge_clause(WindowEdge(EDGE_KIND_UNMEASURED, limit, None), surface=surface)


def _no_edge(note: str, surface: str, limit: int = 5) -> None:
    assert "mid-cluster" not in note and "unmeasured" not in note, note
    assert _mid_clause(surface, limit, None)[:-1] not in note, note


# --------------------------------------------------------------------------- #
# The three edges, on all six drivers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", list(DRIVERS))
def test_mid_cluster_appends_the_clause_to_the_clear_drop_note(ws, driver):
    """Row 8, the rank-6 case under `strong`: the same five returned, the sixth inside
    the margin, and the note is the clear-drop twin's note plus one sentence."""
    _, m = ws
    _seed(m)
    call, _verb, surface, key = DRIVERS[driver]
    mid = call(ws, CannedVector(_mid(m)))
    clear = call(ws, CannedVector(_clear(m)))
    if key is not None:
        assert mid["confidence"] == clear["confidence"] == "strong"
        assert [h["slug"] for h in mid[key]] == [h["slug"] for h in clear[key]] == _SLUGS[:5]
    assert _note(mid, key) == f"{_note(clear, key)} {_mid_clause(surface)}"
    _no_edge(_note(clear, key), surface)


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_an_unmeasured_edge_says_so_and_offers_no_lever(ws, driver):
    """Row 11 / D3: the fetch reached its bound holding exactly `limit` live, so the
    edge is unmeasured — stated, because absence would read as a clear drop — and no
    lever is named, because a higher limit re-reads the same fetch."""
    _, m = ws
    _seed(m)
    call, _verb, surface, key = DRIVERS[driver]
    resp = call(ws, CannedVector(_unmeasured(m)))
    clear = call(ws, CannedVector(_clear(m)))
    note = _note(resp, key)
    assert note == f"{_note(clear, key)} {_unmeasured_clause(surface)}"
    assert "higher limit" not in note
    full_top_stem = {"cli": "--full-top", "mcp": "full_top="}[surface]
    assert full_top_stem not in note


def test_an_unmeasured_fetch_reread_at_a_higher_limit_is_short_at_the_same_live_set(ws):
    """D3's premise: the walk fetches its fixed bound whatever `limit` is, so the same
    350 points read at `limit + 5` hold the same five live decisions and end short —
    the lever the unmeasured clause withholds would have done nothing."""
    config, m = ws
    _seed(m)
    hits = _unmeasured(m)
    store = GraphStore(config.db_path, read_only=True)
    at_limit = walk_ranked([0.1], vector_store=CannedVector(hits, store=store), graph=store,
                           limit=5)
    wider = walk_ranked([0.1], vector_store=CannedVector(hits, store=store), graph=store,
                        limit=10)
    assert at_limit.end == EDGE_UNMEASURED and wider.end == SHORT_BOUND
    assert ([h.node["slug"] for h in wider.window]
            == [h.node["slug"] for h in at_limit.window] == _SLUGS[:5])


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_an_unmeasured_fetch_reread_wider_carries_no_clause(ws, driver):
    """The companion through every surface: the short window states no edge."""
    _, m = ws
    _seed(m)
    call, _verb, surface, key = DRIVERS[driver]
    resp = call(ws, CannedVector(_unmeasured(m)), limit=10)
    _no_edge(_note(resp, key), surface, 10)


# --------------------------------------------------------------------------- #
# Order: band → edge → withheld → stamps-unread (row 14)
# --------------------------------------------------------------------------- #

def _mcp_full_top(tool: str, ws, vector, wrap=None, full_top=None):
    config, _ = ws
    store = GraphStore(config.db_path, read_only=True)
    if wrap is not None:
        store = wrap(store)
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, _FakeEmbed(), vector)):
        fn = mcp_server.surface_decisions if tool == "surface" else mcp_server.query_decisions
        return json.loads(fn(_CLAIM, project=config.workspace_dir, full_top=full_top))


def _cli_full_top(verb: str, ws, vector, *, as_json, wrap=None, full_top=None):
    config, _ = ws
    manager = MitosSyncManager(config)
    manager.embed_provider = _FakeEmbed()
    manager.vector_store = vector
    if wrap is not None:
        manager.store = wrap(manager.store)
    buf = io.StringIO()
    with patch.object(cli, "MitosSyncManager", return_value=manager):
        with redirect_stdout(buf):
            (cmd_surface if verb == "surface" else cmd_query)(
                config, _CLAIM, as_json=as_json, full_top=full_top)
    out = buf.getvalue()
    return json.loads(out) if as_json else out


_FULL_TOP_DRIVERS = {
    "mcp-surface": lambda ws, v, **k: _mcp_full_top("surface", ws, v, **k),
    "mcp-query": lambda ws, v, **k: _mcp_full_top("query", ws, v, **k),
    "cli-surface-json": lambda ws, v, **k: _cli_full_top("surface", ws, v, as_json=True, **k),
    "cli-query-json": lambda ws, v, **k: _cli_full_top("query", ws, v, as_json=True, **k),
    "cli-surface-text": lambda ws, v, **k: _cli_full_top("surface", ws, v, as_json=False, **k),
    "cli-query-text": lambda ws, v, **k: _cli_full_top("query", ws, v, as_json=False, **k),
}


@pytest.mark.parametrize("driver", list(_FULL_TOP_DRIVERS))
def test_the_edge_sits_after_the_band_and_before_withheld_and_stamps_unread(ws, driver):
    """`full_top=2` thins three ranks and a stamp fault costs the stamps: the note is
    the band note, the edge clause holding `full_top=2`, the withheld clause, then the
    stamps-unread clause — the first from the composer, the last two from the call site."""
    config, m = ws
    _seed(m)
    call = _FULL_TOP_DRIVERS[driver]
    _c, _verb, surface, key = DRIVERS[driver]
    healthy_clear = call(ws, CannedVector(_clear(m)))
    resp = call(ws, CannedVector(_mid(m)), wrap=_StampFault, full_top=2)
    band = _note(healthy_clear, key)
    expected = " ".join([band, _mid_clause(surface, 5, 2),
                         withheld_clause(3, surface=surface, config=config),
                         stamps_unread_clause(surface=surface, config=config)])
    assert _note(resp, key) == expected


# --------------------------------------------------------------------------- #
# Exits that hold no edge (row 13)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", list(DRIVERS))
def test_a_blackout_carries_no_edge_clause(ws, driver):
    """Every hit retired: no live window, so no edge, and the blackout note stands."""
    _, m = ws
    _rec(m, "old")
    _rec(m, "new", supersedes="old")
    call, _verb, surface, key = DRIVERS[driver]
    resp = call(ws, CannedVector([point_hit(_id(m, "old"), 0.9)]))
    handles = [{"slug": "old", "state": "superseded", "superseded_by": ["new"]}]
    if key is not None:
        assert resp[key] == [] and resp["all_superseded"] == handles
        assert resp["note"] == blackout_note(handles)
    else:
        assert blackout_note(handles) in resp
    _no_edge(resp if key is None else resp["note"], surface)


@pytest.mark.parametrize("driver", ["cli", "mcp"])
def test_an_unbuilt_graph_carries_no_edge_clause(cloned, driver):
    """No nodes, so no window and no edge: the unbuilt-graph note stands alone."""
    from test_surface_confidence import _cli_query_json, _mcp_query
    resp = (_cli_query_json if driver == "cli" else _mcp_query)([], cloned)
    assert "graph is unbuilt" in resp["note"]
    _no_edge(resp["note"], driver)


def test_the_exact_slug_exit_carries_no_note_and_so_no_edge(ws):
    """MCP `query_decisions`' exact-slug exit returns before the ranked branch."""
    config, m = ws
    _seed(m)
    resp = _mcp_full_top("query", ws, CannedVector(_mid(m)))
    assert "note" in resp  # the ranked branch, as a control
    from test_surface_confidence import _mcp_query
    exact = _mcp_query(CannedVector(_mid(m)), ws, query=_SLUGS[0])
    assert exact["slug"] == _SLUGS[0] and "note" not in exact and "confidence" not in exact


# --------------------------------------------------------------------------- #
# Parity (row 16) and the entered frame (row 17)
# --------------------------------------------------------------------------- #

_CLI_TO_MCP = (("--limit 5", "limit=5"), ("--full-top 5", "full_top=5"),
               ("mitos list", "list_decisions()"))


@pytest.mark.parametrize("verb", ["surface", "query"])
@pytest.mark.parametrize("shape", ["mid", "unmeasured"])
def test_cli_and_mcp_differ_only_in_the_spellings(ws, verb, shape):
    """No key added: the `--json` and MCP key sets are equal, and the notes differ only
    in each boundary's pointer spellings; the CLI text prints the `--json` note."""
    config, m = ws
    _seed(m)
    hits = _mid(m) if shape == "mid" else _unmeasured(m)
    mcp = DRIVERS[f"mcp-{verb}"][0](ws, CannedVector(hits))
    cli_json = DRIVERS[f"cli-{verb}-json"][0](ws, CannedVector(hits))
    text = DRIVERS[f"cli-{verb}-text"][0](ws, CannedVector(hits))
    assert set(mcp) == set(cli_json)
    mapped = cli_json["note"]
    for cli_form, mcp_form in _CLI_TO_MCP:
        mapped = mapped.replace(cli_form, mcp_form)
    if verb == "query":
        mapped = mapped.replace(_query_pointer("cli", config), _query_pointer("mcp", config))
    assert mapped == mcp["note"]
    assert _note(text, None) == cli_json["note"]


def test_the_rank_six_case_through_cli_main(ws, monkeypatch, capsys):
    """Row 17: `mitos surface -p <ws> … --limit 5` entered through `main()`, whose
    resolution, dispatch and printing all run, prints the mid-cluster clause."""
    config, m = ws
    _seed(m)
    manager = MitosSyncManager(config)
    manager.embed_provider = _FakeEmbed()
    manager.vector_store = CannedVector(_mid(m))
    monkeypatch.setattr(sys, "argv", ["mitos", "surface", "-p", config.workspace_dir,
                                      _CLAIM, "--limit", "5"])
    with patch.object(cli, "MitosSyncManager", return_value=manager):
        cli.main()
    out = capsys.readouterr().out
    assert _note(out, None).endswith(f" {_mid_clause('cli')}")
