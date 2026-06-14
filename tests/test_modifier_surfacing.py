"""Tests for reverse-relation modifier surfacing (the "amended axioms read as live" fix).

Driven by loop-Claude's most-repeated AX friction: an `amends`/`narrows` leaves its
target `active` with the ORIGINAL axiom text, and no retrieval surface signalled that a
later decision had moved on from it — so a fresh agent cited a superseded mechanism
(worst case, a relocated architecture) with full confidence. The graph always knew (the
edges exist); only the payload lied.

These pin the fix end-to-end: `GraphStore.get_modifiers[_map]` reads the incoming
modifying edges, and every read surface (MCP surface/query/list, the CLI twins,
`mitos show`, and the rendered `live_axioms.md`) stamps `superseded_by` / `amended_by` /
`narrowed_by` / `corrected_by` so the reader knows which decision to chase.

Forced fully offline (unreachable Qdrant + no keys) so the behaviour is deterministic and
never depends on running services.
"""

import json
import shutil
import tempfile
from typing import Iterator, Tuple

import pytest
from unittest.mock import patch

from mitos.config import MitosConfig
from mitos.cli import cmd_init, cmd_list, cmd_show, cmd_surface
from mitos.store import GraphStore, MODIFIER_EDGE_KEYS
from mitos.sync import MitosSyncManager
from mitos.renderer import render_node_markdown, MitosRenderer


@pytest.fixture
def offline(monkeypatch):
    """Forces degraded graph-only mode: unreachable Qdrant, no embedding keys."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")  # nothing listens here
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def ws(offline) -> Iterator[Tuple[MitosConfig, MitosSyncManager]]:
    """An initialised temp workspace + a manager, in offline graph-only mode."""
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    cmd_init(config)
    yield config, MitosSyncManager(config)
    shutil.rmtree(tmp, ignore_errors=True)


def _rec(m: MitosSyncManager, slug: str, scope=None, **relations) -> dict:
    res = m.record_decision_entry(
        axiom=f"Axiom for {slug}.",
        rejected_paths=f"Rejected alternative for {slug}.",
        scope=scope or [],
        slug=slug,
        **relations,
    )
    assert "error" not in res, res
    return res


# --------------------------------------------------------------------------- #
# store.get_modifiers[_map] — the source of truth
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("relation,reverse_key", [
    ("supersedes", "superseded_by"),
    ("amends", "amended_by"),
    ("narrows", "narrowed_by"),
])
def test_get_modifiers_one_per_relation(ws, relation, reverse_key) -> None:
    """Each writable modifying edge surfaces under its reverse key on the TARGET."""
    config, m = ws
    target = _rec(m, "target")
    _rec(m, "modifier", **{relation: "target"})
    store = GraphStore(config.db_path)
    assert store.get_modifiers(target["id"]) == {reverse_key: ["modifier"]}


def test_get_modifiers_corrects_edge(ws) -> None:
    """`corrects` (a valid graph edge, e.g. from import — not exposed to the agentic
    write path) maps to `corrected_by`, covering the full MODIFIER_EDGE_KEYS table."""
    import sqlite3
    config, m = ws
    target = _rec(m, "target")
    corrector = _rec(m, "corrector")
    conn = sqlite3.connect(config.db_path)
    conn.execute("INSERT INTO edges (from_id, to_id, type) VALUES (?, ?, 'corrects')",
                 (corrector["id"], target["id"]))
    conn.commit()
    conn.close()
    assert GraphStore(config.db_path).get_modifiers(target["id"]) == {"corrected_by": ["corrector"]}


def test_get_modifiers_only_on_target_not_modifier(ws) -> None:
    """The modifying decision itself is unmodified — the edge points one way."""
    config, m = ws
    _rec(m, "target")
    modifier = _rec(m, "modifier", amends="target")
    store = GraphStore(config.db_path)
    assert store.get_modifiers(modifier["id"]) == {}


def test_get_modifiers_unmodified_is_empty(ws) -> None:
    """A solo decision returns {} — the healthy, common case stays uncluttered."""
    config, m = ws
    solo = _rec(m, "solo")
    assert GraphStore(config.db_path).get_modifiers(solo["id"]) == {}


def test_get_modifiers_accumulates_multiple_edges(ws) -> None:
    """A hub node accumulates edges: amended by one decision AND narrowed by another.

    The multi-edge case loop-Claude flagged — a long-lived hub ADR collects modifiers,
    and ALL of them must surface, not just the latest.
    """
    config, m = ws
    hub = _rec(m, "hub")
    _rec(m, "amender", amends="hub")
    _rec(m, "narrower", narrows="hub")
    mods = GraphStore(config.db_path).get_modifiers(hub["id"])
    assert mods == {"amended_by": ["amender"], "narrowed_by": ["narrower"]}


def test_get_modifiers_two_of_same_relation_sorted(ws) -> None:
    """Two decisions amending one node both appear, deterministically ordered."""
    config, m = ws
    hub = _rec(m, "hub")
    _rec(m, "zeta-amend", amends="hub")
    _rec(m, "alpha-amend", amends="hub")
    mods = GraphStore(config.db_path).get_modifiers(hub["id"])
    assert mods == {"amended_by": ["alpha-amend", "zeta-amend"]}  # slug-sorted


def test_get_modifiers_map_batches(ws) -> None:
    """The batch map keys each modified node and omits unmodified ones."""
    config, m = ws
    a = _rec(m, "a")
    b = _rec(m, "b")
    solo = _rec(m, "solo")
    _rec(m, "a-mod", amends="a")
    _rec(m, "b-mod", supersedes="b")
    store = GraphStore(config.db_path)
    mp = store.get_modifiers_map([a["id"], b["id"], solo["id"]])
    assert mp == {a["id"]: {"amended_by": ["a-mod"]},
                  b["id"]: {"superseded_by": ["b-mod"]}}
    assert solo["id"] not in mp


def test_get_modifiers_map_empty_input(ws) -> None:
    """No ids → empty map, no query."""
    config, _ = ws
    assert GraphStore(config.db_path).get_modifiers_map([]) == {}


# --------------------------------------------------------------------------- #
# MCP read surfaces
# --------------------------------------------------------------------------- #

def test_mcp_query_exact_slug_surfaces_amended_by(ws) -> None:
    """THE headline case: an exact-slug read of an amended decision reads `active`
    but now also carries `amended_by` so the stale axiom can't masquerade as live."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "boundary-rule")
    _rec(m, "boundary-rule-v2", amends="boundary-rule")
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.query_decisions("boundary-rule"))
    assert resp["state"] == "active"           # amends does NOT retire the parent
    assert resp["amended_by"] == ["boundary-rule-v2"]


def test_mcp_query_exact_slug_unmodified_has_no_modifier_keys(ws) -> None:
    """An unmodified decision's payload is unchanged (no spurious modifier keys)."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "clean")
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.query_decisions("clean"))
    assert not any(k in resp for k in MODIFIER_EDGE_KEYS.values())


def test_mcp_list_decisions_surfaces_modifiers(ws) -> None:
    """list_decisions stamps modifier slugs onto the affected decision only."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "amended-one", scope=["z"])
    _rec(m, "amender", scope=["z"], amends="amended-one")
    _rec(m, "untouched", scope=["z"])
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.list_decisions(scope="z"))
    by_slug = {d["slug"]: d for d in resp["decisions"]}
    assert by_slug["amended-one"]["amended_by"] == ["amender"]
    assert not any(k in by_slug["untouched"] for k in MODIFIER_EDGE_KEYS.values())


def test_mcp_surface_scope_fallback_surfaces_modifiers(ws) -> None:
    """The offline scope-fallback path on surface_decisions also stamps modifiers."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "fb-target", scope=["db"])
    _rec(m, "fb-amender", scope=["db"], amends="fb-target")
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.surface_decisions(query="anything", scope="db"))
    target = next(d for d in resp["active_decisions"] if d["slug"] == "fb-target")
    assert target["amended_by"] == ["fb-amender"]


class _FakeEmbed:
    """Stand-in embedding provider — returns a constant vector, never calls the API."""

    def get_embedding(self, text: str, is_query: bool = False):
        return [0.1, 0.2, 0.3]


class _FakeVector:
    """Stand-in vector store — replays a fixed ranked match list."""

    def __init__(self, matches):
        self._matches = matches

    def query(self, vector, limit: int = 5, filter_scope=None):
        return self._matches


def test_mcp_surface_semantic_path_surfaces_modifiers(ws) -> None:
    """The SEMANTIC ranking loop (the AX loop's primary path) stamps modifiers too —
    not just the offline exact/scope-fallback paths. Driven by a fake vector store so
    it is deterministic without Qdrant/keys."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "sem-target", scope=["x"])
    _rec(m, "sem-amender", scope=["x"], amends="sem-target")
    store = GraphStore(config.db_path, read_only=True)
    fake_vec = _FakeVector([{"slug": "sem-target", "score": 0.91}])
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, _FakeEmbed(), fake_vec)):
        resp = json.loads(mcp_server.surface_decisions(query="how do we normalize rows"))
    target = next(d for d in resp["active_decisions"] if d["slug"] == "sem-target")
    assert target["amended_by"] == ["sem-amender"]


def test_mcp_query_semantic_path_surfaces_modifiers(ws) -> None:
    """query_decisions' semantic branch (claim, not slug) stamps modifiers on matches."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "q-target", scope=["x"])
    _rec(m, "q-amender", scope=["x"], amends="q-target")
    store = GraphStore(config.db_path, read_only=True)
    fake_vec = _FakeVector([{"slug": "q-target", "score": 0.88}])
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, _FakeEmbed(), fake_vec)):
        # A spaced claim resolves to no exact slug → falls through to semantic search.
        resp = json.loads(mcp_server.query_decisions("a claim that is not any slug"))
    match = next(mt for mt in resp["matches"] if mt["slug"] == "q-target")
    assert match["amended_by"] == ["q-amender"]


def test_mcp_query_exact_slug_superseded_carries_superseded_by(ws) -> None:
    """A superseded node read by exact slug reports BOTH state and which decision replaced it."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "v1")
    _rec(m, "v2", supersedes="v1")
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        resp = json.loads(mcp_server.query_decisions("v1"))
    assert resp["state"] == "superseded"
    assert resp["superseded_by"] == ["v2"]


def test_modifiers_survive_brief_trim(ws) -> None:
    """The staleness flag is ALWAYS attached — even on a brief payload where
    rejected_paths is trimmed. A lightweight scan that lost the heavy field still needs
    to know the axiom has been moved on from."""
    from mitos import mcp_server
    config, m = ws
    _rec(m, "hot")
    _rec(m, "hot-v2", amends="hot")
    store = GraphStore(config.db_path, read_only=True)
    node = store.get_node_by_slug("hot")

    brief = mcp_server._decision_payload(node, 1.0, brief=True, store=store)
    assert "rejected_paths" not in brief and brief["amended_by"] == ["hot-v2"]

    full = mcp_server._decision_payload(node, 1.0, brief=False, store=store)
    assert full["rejected_paths"] and full["amended_by"] == ["hot-v2"]


# --------------------------------------------------------------------------- #
# CLI read surfaces
# --------------------------------------------------------------------------- #

def test_cli_show_prints_modifier(ws, capsys) -> None:
    """`mitos show` on an amended decision prints the 'Amended by' annotation."""
    config, m = ws
    _rec(m, "shown")
    _rec(m, "shown-v2", amends="shown")
    capsys.readouterr()
    cmd_show(config, "shown")
    out = capsys.readouterr().out
    assert "Amended by" in out and "shown-v2" in out


def test_cli_list_text_marks_modified(ws, capsys) -> None:
    """`mitos list` text output flags a modified-but-live decision with ⚠."""
    config, m = ws
    _rec(m, "listed", scope=["s"])
    _rec(m, "listed-v2", scope=["s"], narrows="listed")
    capsys.readouterr()
    cmd_list(config, scope="s")
    out = capsys.readouterr().out
    assert "⚠" in out and "narrowed by" in out and "listed-v2" in out


def test_cli_list_json_carries_modifiers(ws, capsys) -> None:
    """`mitos list --json` carries the modifier key for agent consumption."""
    config, m = ws
    _rec(m, "j-target", scope=["s"])
    _rec(m, "j-amender", scope=["s"], amends="j-target")
    capsys.readouterr()
    cmd_list(config, scope="s", as_json=True)
    out = json.loads(capsys.readouterr().out)
    target = next(d for d in out["decisions"] if d["slug"] == "j-target")
    assert target["amended_by"] == ["j-amender"]


def test_cli_surface_json_carries_modifiers(ws, capsys) -> None:
    """`mitos surface --json` (scope fallback) carries the modifier key."""
    config, m = ws
    _rec(m, "s-target", scope=["db"])
    _rec(m, "s-amender", scope=["db"], amends="s-target")
    capsys.readouterr()
    cmd_surface(config, query="anything", scope="db", as_json=True)
    out = json.loads(capsys.readouterr().out)
    target = next(d for d in out["active_decisions"] if d["slug"] == "s-target")
    assert target["amended_by"] == ["s-amender"]


# --------------------------------------------------------------------------- #
# Renderer (live_axioms.md is a primary agent-context artifact)
# --------------------------------------------------------------------------- #

def test_render_node_markdown_emits_marker() -> None:
    """A live node with modifiers renders a ⚠ chase-it line; without, nothing extra."""
    node = {"slug": "x", "core_axiom": "Do the thing.", "scope": [], "mechanisms": []}
    plain = render_node_markdown(node)
    assert "⚠" not in plain
    marked = render_node_markdown(node, {"amended_by": ["x-v2"]})
    assert "⚠ Amended by:** x-v2" in marked and "chase" in marked


def test_render_all_writes_modifier_marker(ws) -> None:
    """render_all annotates an amended-but-active decision in live_axioms.md."""
    config, m = ws
    _rec(m, "rendered", scope=["r"])
    _rec(m, "rendered-v2", scope=["r"], amends="rendered")
    store = GraphStore(config.db_path)
    MitosRenderer(config.workspace_dir).render_all(store)
    with open(f"{config.workspace_dir}/live_axioms.md", encoding="utf-8") as f:
        content = f.read()
    assert "⚠ Amended by:** rendered-v2" in content
