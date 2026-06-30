"""Phase 4c — OQ Stage-2 resolution, self-healing, modifier de-projection, and the
minimal CLI/MCP visibility reads that light up over ``oq_state_view``.

V1a shipped OQ **Stage 1** only — ``get_open_questions`` applied the kill-edge
anti-join (a typo-fixed OQ drops out) but carried no resolution state: a parked OQ
and a resolved one were indistinguishable, and the four shipped OQ-visibility
consumers read a ``computed_state`` key the method never set (the CLI verbs
``KeyError``-ed on a non-empty OQ set; the MCP twins silently degraded to ``[]``).

4c makes ``get_open_questions`` *be* ``oq_state_view``: each Stage-1 survivor gets a
query-time ``state ∈ {parked, resolved}`` (``resolved`` iff ≥1 incoming ``resolves``
from a STILL-ACTIVE decision), and the five consumers (CLI ``open-questions`` /
``list`` / ``surface`` + MCP ``surface_decisions`` / ``list_decisions``) foreground
``parked`` and propagate the OQ's ``amended_by`` / ``narrowed_by`` modifiers.

The elegance the vision pins (M3 — state computed, never stored): **self-healing is
free**. Tear up the decision that resolved a question and the OQ re-surfaces as
``parked`` automatically, because ``state`` is a view recomputed at read time, not a
stored flag.

Forced fully offline (unreachable Qdrant + no keys) so every gate is deterministic
graph state — no LLM, no mocks, real temp SQLite. The established keyless DoD
pattern (``tests/test_v1a_closeout.py`` / ``tests/test_modifier_surfacing.py``).
"""

import json
import shutil
import tempfile
from typing import Any, Dict, Iterator, Optional, Tuple

import pytest
from unittest.mock import patch

from mitos.config import MitosConfig
from mitos.cli import cmd_init, cmd_list, cmd_open_questions, cmd_surface
from mitos.parser import ParsedEntry
from mitos.store import GraphStore


# --------------------------------------------------------------------------- #
# Keyless / offline harness
# --------------------------------------------------------------------------- #


@pytest.fixture
def offline(monkeypatch) -> None:
    """Forces degraded graph-only mode: unreachable Qdrant, no embedding keys.

    Makes a key-bearing dev box behave byte-for-byte like keyless CI — commits go
    to real temp SQLite, ``commit_parsed_entry`` never embeds, and every assertion
    keys on graph state, never on a live service.
    """
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")  # nothing listens here
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def ws(offline) -> Iterator[Tuple[MitosConfig, GraphStore]]:
    """An initialised temp workspace + a graph store bound to it, offline.

    Yields ``(config, store)``: the store drives direct ``commit_parsed_entry``
    graph mutation; ``config`` lets the CLI verbs / MCP tools open their own store
    on the same ``db_path``.
    """
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    cmd_init(config)
    try:
        yield config, GraphStore(config.db_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _commit_oq(store: GraphStore, slug: str, *, topic: Optional[str] = None,
               questions=None, scope=None, **relations):
    """Commits a hand-built open_question ``ParsedEntry`` through the write path.

    ``relations`` are edge fields (``amends`` / ``narrows`` / ``supersedes`` /
    ``corrects`` …) given as a slug or a list of slugs — normalised to the
    ``List[str]`` the commit path iterates (a bare ``str`` would iterate
    per-character).
    """
    e = ParsedEntry("open_question", slug, 1, 5)
    e.topic = topic or f"Topic for {slug}"
    e.questions_raised = list(questions) if questions else [f"What about {slug}?"]
    e.scope = list(scope) if scope else []
    for name, val in relations.items():
        setattr(e, name, val if isinstance(val, list) else [val])
    return store.commit_parsed_entry(e)


def _commit_dec(store: GraphStore, slug: str, *, axiom: Optional[str] = None,
                rejected: str = "None.", scope=None, **relations):
    """Commits a hand-built decision ``ParsedEntry`` through the write path."""
    e = ParsedEntry("decision", slug, 1, 5)
    e.axiom = axiom or f"Axiom for {slug}."
    e.rejected_paths = rejected
    e.scope = list(scope) if scope else []
    for name, val in relations.items():
        setattr(e, name, val if isinstance(val, list) else [val])
    return store.commit_parsed_entry(e)


def _oq(store: GraphStore, slug: str, scope: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The full ``oq_state_view`` dict for ``slug`` (or None if not in the view)."""
    for oq in store.get_open_questions(scope=scope):
        if oq["slug"] == slug:
            return oq
    return None


def _state(store: GraphStore, slug: str, scope: Optional[str] = None) -> Optional[str]:
    """The OQ's computed Stage-2 ``state`` from ``oq_state_view`` (or None)."""
    oq = _oq(store, slug, scope=scope)
    return oq["state"] if oq is not None else None


def _node_meta(store: GraphStore, node_id: str) -> Tuple[str, Optional[str]]:
    """Reads a node's ``(updated_at, pending_embeddings.queued_at)`` via raw SQL.

    The cascade-write fingerprint: a commit that touches this node would tick
    ``updated_at`` and re-stamp ``queued_at``; one that does not leaves both
    byte-identical.
    """
    conn = store._get_connection()
    try:
        updated_at = conn.execute(
            "SELECT updated_at FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()[0]
        row = conn.execute(
            "SELECT queued_at FROM pending_embeddings WHERE node_id = ?", (node_id,)
        ).fetchone()
        return updated_at, (row[0] if row else None)
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Stage-2 resolution + self-healing (T7, DoD #7)
# --------------------------------------------------------------------------- #


def test_oq_parked_until_active_resolver(ws) -> None:
    """An OQ reads ``parked`` until a STILL-ACTIVE decision resolves it (T7 positive)."""
    _, store = ws
    _commit_oq(store, "q-auth")
    assert _state(store, "q-auth") == "parked"

    _commit_dec(store, "d-jwt", resolves="q-auth")
    assert _state(store, "q-auth") == "resolved"


def test_oq_reparks_when_resolver_superseded_without_reattach(ws) -> None:
    """Self-healing: supersede the resolver WITHOUT re-declaring ``Resolves:`` →
    the OQ flips ``resolved → parked`` automatically (M3 query-time, §4.5.1).

    No 4c mechanism does this — it falls out of the source-liveness gate: the
    moment ``d-jwt`` gains an incoming kill-edge, ``_OQ_RESOLVED_SQL``'s EXISTS
    fails and the view recomputes ``parked``. Resolution does NOT flow transitively
    through ``supersedes``.
    """
    _, store = ws
    _commit_oq(store, "q-auth")
    _commit_dec(store, "d-jwt", resolves="q-auth")
    assert _state(store, "q-auth") == "resolved"

    # Kill the resolver from a SEPARATE entry (the 2a/2b fixture gotcha: never stack
    # a kill-edge + another edge to the same target in one entry — dangling_edge).
    _commit_dec(store, "d-jwt-v2", supersedes="d-jwt")
    assert _state(store, "q-auth") == "parked"  # re-surfaced, fail-safe-and-loud


def test_oq_resolution_carries_forward_only_by_explicit_redeclare(ws) -> None:
    """Resolution is carried forward ONLY by an explicit re-declared ``Resolves:``.

    A successor that supersedes the resolver AND re-declares ``Resolves: q-auth``
    (two distinct targets, so no dangling_edge) keeps the OQ ``resolved`` — the
    active re-declaration is what counts, not transitive inheritance.
    """
    _, store = ws
    _commit_oq(store, "q-auth")
    _commit_dec(store, "d-jwt", resolves="q-auth")
    _commit_dec(store, "d-jwt-v2", supersedes="d-jwt", resolves="q-auth")
    assert _state(store, "q-auth") == "resolved"


# --------------------------------------------------------------------------- #
# Stage 1 still holds — kill-edged OQs are excluded entirely (V1a regression)
# --------------------------------------------------------------------------- #


def test_kill_edged_oq_excluded_not_parked_forever(ws) -> None:
    """A typo-fixed (``corrects`` kill-edge) OQ leaves the view, not "parked forever".

    Stage 1 (the kill-edge anti-join) is unchanged: a corrected OQ is excluded from
    ``get_open_questions`` outright — it never even reaches Stage-2 labeling.
    """
    _, store = ws
    _commit_oq(store, "q-typo")
    _commit_oq(store, "q-fixed", corrects="q-typo")  # OQ→OQ corrects, same kind
    slugs = [oq["slug"] for oq in store.get_open_questions()]
    assert "q-typo" not in slugs
    assert "q-fixed" in slugs


# --------------------------------------------------------------------------- #
# OQ modifier stamping + de-projection on the view (T6 OQ side, DoD #6 layer i)
# --------------------------------------------------------------------------- #


def test_amended_active_oq_carries_amended_by_then_deprojects(ws) -> None:
    """An amended-but-active OQ reads ``amended_by`` on the view; killing the amender
    de-projects it — the 2b source-liveness filter reaching the OQ surface for free.

    The "amended axioms read as live" trap, OQ side: the OQ stays ``parked``
    (``amends`` is not ``resolves``) but must not read as the final word while a
    later ``amends`` has moved on from it.
    """
    _, store = ws
    _commit_oq(store, "q-base")
    _commit_oq(store, "q-amender", amends="q-base")  # OQ amends OQ, both active

    base = _oq(store, "q-base")
    assert base["state"] == "parked"
    assert base["amended_by"] == ["q-amender"]

    # Supersede the amender from a SEPARATE entry → de-projects (fail-safe; the
    # amender's axiom survives in the graph, the projection merely drops).
    _commit_oq(store, "q-killer", supersedes="q-amender")
    base2 = _oq(store, "q-base")
    assert base2["state"] == "parked"
    assert "amended_by" not in base2


def test_narrowed_active_oq_carries_narrowed_by_then_deprojects(ws) -> None:
    """The ``narrows`` mirror of OQ modifier de-projection."""
    _, store = ws
    _commit_oq(store, "q-base")
    _commit_oq(store, "q-narrower", narrows="q-base")
    assert _oq(store, "q-base")["narrowed_by"] == ["q-narrower"]

    _commit_oq(store, "q-killer", supersedes="q-narrower")
    assert "narrowed_by" not in _oq(store, "q-base")


# --------------------------------------------------------------------------- #
# No-cascade OQ-state flip (T3 OQ side, DoD #3)
# --------------------------------------------------------------------------- #


def test_resolving_oq_writes_no_cascade_to_the_oq(ws) -> None:
    """Resolving an OQ flips its COMPUTED state but writes NOTHING to the OQ node.

    The OQ's ``state`` becomes ``resolved`` at read time (no row rewrite), so the
    resolving commit must leave the OQ's ``updated_at`` and Outbox ``queued_at``
    byte-identical — only the committing decision gets an ``updated_at`` tick + an
    Outbox row. V1b ships no transitive cascade (DoD #3).
    """
    _, store = ws
    d_oq = _commit_oq(store, "q-auth", scope=["auth"])
    before = _node_meta(store, d_oq.node_id)

    d_res = _commit_dec(store, "d-jwt", resolves="q-auth", scope=["auth"])

    # The OQ now reads resolved (computed at read time)...
    assert _state(store, "q-auth") == "resolved"
    # ...but the resolving commit wrote no cascade to the OQ row.
    assert _node_meta(store, d_oq.node_id) == before
    # The committing decision is the only node the commit enqueued.
    pending = {row["node_id"] for row in store.get_pending_embeddings()}
    assert d_res.node_id in pending


# --------------------------------------------------------------------------- #
# CLI bounded-visibility reads (T14, DoD #14) — the latent-KeyError gate
# --------------------------------------------------------------------------- #


def test_cmd_open_questions_lists_only_parked_no_crash(ws, capsys) -> None:
    """``mitos open-questions`` foregrounds parked OQs and does NOT crash on a
    non-empty OQ set (the latent ``KeyError`` the empty-DB test never caught)."""
    config, store = ws
    _commit_oq(store, "q-open", scope=["auth"])
    _commit_oq(store, "q-answered", scope=["auth"])
    _commit_dec(store, "d-answer", resolves="q-answered", scope=["auth"])

    capsys.readouterr()
    cmd_open_questions(config)  # must not raise
    out = capsys.readouterr().out
    assert "q-open" in out
    assert "q-answered" not in out  # resolved OQ is bounded out of the default view


def test_cmd_list_json_open_questions_only_parked(ws, capsys) -> None:
    """``mitos list --json`` carries only parked OQs in ``open_questions``."""
    config, store = ws
    _commit_oq(store, "q-open", scope=["s"])
    _commit_oq(store, "q-answered", scope=["s"])
    _commit_dec(store, "d-answer", resolves="q-answered", scope=["s"])

    capsys.readouterr()
    cmd_list(config, scope="s", as_json=True)
    out = json.loads(capsys.readouterr().out)
    topics = {oq["topic"] for oq in out["open_questions"]}
    assert topics == {"q-open"}


def test_cmd_open_questions_json_shape_parked_only(ws, capsys) -> None:
    """`open-questions --json`: parked OQs appear with the cross-verb per-OQ shape
    (`topic`/`questions_raised`/`park_reason`); a resolved OQ is bounded out."""
    config, store = ws
    _commit_oq(store, "q-open", scope=["s"])
    _commit_oq(store, "q-answered", scope=["s"])
    _commit_dec(store, "d-answer", resolves="q-answered", scope=["s"])

    capsys.readouterr()
    cmd_open_questions(config, scope="s", as_json=True)
    out = json.loads(capsys.readouterr().out)
    topics = {oq["topic"] for oq in out["open_questions"]}
    assert topics == {"q-open"}
    assert out["total"] == 1
    assert out["scope"] == "s"
    entry = out["open_questions"][0]
    assert set(entry) >= {"topic", "questions_raised", "park_reason"}


def test_cmd_open_questions_json_empty_is_honest_envelope(ws, capsys) -> None:
    """`open-questions --json` on an empty graph emits a clean honest-empty envelope
    (exit-0, no crash, never an error) — empty/fresh is first-class."""
    config, store = ws

    capsys.readouterr()
    cmd_open_questions(config, as_json=True)  # must not raise
    out = json.loads(capsys.readouterr().out)
    assert out == {"open_questions": [], "total": 0, "scope": None}


def test_cmd_open_questions_json_matches_list_json_oq_shape(ws, capsys) -> None:
    """The cross-verb contract: a parked OQ's per-entry dict is byte-identical between
    `open-questions --json` and `list --json`'s `open_questions[]` array."""
    config, store = ws
    _commit_oq(store, "q-listed", scope=["s"])
    _commit_oq(store, "q-amender", scope=["s"], amends="q-listed")

    capsys.readouterr()
    cmd_open_questions(config, scope="s", as_json=True)
    oq_entries = json.loads(capsys.readouterr().out)["open_questions"]

    cmd_list(config, scope="s", as_json=True)
    list_entries = json.loads(capsys.readouterr().out)["open_questions"]

    assert oq_entries == list_entries


def test_cmd_list_text_marks_amended_parked_oq(ws, capsys) -> None:
    """``mitos list`` text flags an amended-but-active parked OQ with a ⚠ marker."""
    config, store = ws
    _commit_oq(store, "q-listed", scope=["s"])
    _commit_oq(store, "q-amender", scope=["s"], amends="q-listed")

    capsys.readouterr()
    cmd_list(config, scope="s")
    out = capsys.readouterr().out
    assert "q-listed" in out
    assert "⚠" in out and "amended by" in out and "q-amender" in out


def test_cmd_surface_lists_parked_oq_with_modifier(ws, capsys) -> None:
    """``mitos surface --scope`` (the plan-missed 5th consumer) foregrounds parked
    OQs, excludes resolved, and carries ``amended_by`` — CLI⇄MCP parity.

    Before 4c this read the unset ``computed_state`` inside a swallowing
    ``except Exception: pass``, so the CLI surface silently returned NO open
    questions while the MCP twin returned them. The fix restores parity.
    """
    config, store = ws
    _commit_oq(store, "q-open", scope=["auth"])
    _commit_oq(store, "q-amender", scope=["auth"], amends="q-open")
    _commit_oq(store, "q-answered", scope=["auth"])
    _commit_dec(store, "d-answer", scope=["auth"], resolves="q-answered")

    capsys.readouterr()
    cmd_surface(config, query="anything", scope="auth", as_json=True)
    out = json.loads(capsys.readouterr().out)
    by_topic = {oq["topic"]: oq for oq in out["open_questions"]}
    assert "q-open" in by_topic            # parked present
    assert "q-answered" not in by_topic     # resolved absent
    assert by_topic["q-open"]["amended_by"] == ["q-amender"]


# --------------------------------------------------------------------------- #
# MCP parity (present/absent, not just no-crash — §7 masked-bug note)
# --------------------------------------------------------------------------- #


def test_mcp_surface_decisions_oq_present_absent_amended(ws) -> None:
    """``surface_decisions`` returns the parked OQ PRESENT, the resolved OQ ABSENT,
    and an amended-but-active OQ carries ``amended_by`` (§9.4 layer ii).

    Asserting PRESENCE, not just "no crash": the ``except Exception: pass`` would
    have passed a no-crash test even with a silently-empty list.
    """
    from mitos import mcp_server
    config, store = ws
    _commit_oq(store, "q-open", scope=["auth"])
    _commit_oq(store, "q-amender", scope=["auth"], amends="q-open")
    _commit_oq(store, "q-answered", scope=["auth"])
    _commit_dec(store, "d-answer", scope=["auth"], resolves="q-answered")

    ro = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(ro, None, None)):
        resp = json.loads(mcp_server.surface_decisions(query="anything", scope="auth"))
    by_topic = {oq["topic"]: oq for oq in resp["open_questions"]}
    assert "q-open" in by_topic
    assert "q-answered" not in by_topic
    assert by_topic["q-open"]["amended_by"] == ["q-amender"]


def test_mcp_list_decisions_oq_present_absent(ws) -> None:
    """``list_decisions`` returns the parked OQ present and the resolved OQ absent."""
    from mitos import mcp_server
    config, store = ws
    _commit_oq(store, "q-open", scope=["auth"])
    _commit_oq(store, "q-answered", scope=["auth"])
    _commit_dec(store, "d-answer", scope=["auth"], resolves="q-answered")

    ro = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(ro, None, None)):
        resp = json.loads(mcp_server.list_decisions(scope="auth"))
    topics = {oq["topic"] for oq in resp["open_questions"]}
    assert "q-open" in topics
    assert "q-answered" not in topics


def _surface_oq(config, scope):
    """Returns ``{topic: oq_subdict}`` from MCP ``surface_decisions`` for a scope."""
    from mitos import mcp_server
    ro = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(ro, None, None)):
        resp = json.loads(mcp_server.surface_decisions(query="anything", scope=scope))
    return {oq["topic"]: oq for oq in resp["open_questions"]}


def test_oq_modifier_deprojects_at_the_output_layer(ws) -> None:
    """The 2b de-projection reaches the USER-FACING output, not just the payload
    (DoD #6 OQ layer ii — both layers).

    While the amender is active the MCP surface output carries ``amended_by``; once
    the amender is superseded the SAME output drops it — so an amended-then-healed
    OQ never reads as the final word at the surface the architect actually sees.
    """
    config, store = ws
    _commit_oq(store, "q-base", scope=["auth"])
    _commit_oq(store, "q-amender", scope=["auth"], amends="q-base")
    assert _surface_oq(config, "auth")["q-base"]["amended_by"] == ["q-amender"]

    # Supersede the amender from a separate entry → output de-projects.
    _commit_oq(store, "q-killer", scope=["auth"], supersedes="q-amender")
    assert "amended_by" not in _surface_oq(config, "auth")["q-base"]
