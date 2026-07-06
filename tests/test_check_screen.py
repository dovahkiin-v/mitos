"""Tests for the corpus-audit either-direction strong-edge pair screen (Phase 2a).

``mitos check`` sweeps the whole corpus and would, naively, spend an LLM judgment on
every similar pair — including pairs the author has ALREADY reasoned about with a
declared relationship. The sync-time sensor drops those via
``declared_strong_targets(entry)``, but that reads an entry's *forward* declarations
only. In corpus mode a pair can be declared from *either* side (the harbor global
declares nothing; its scoped exception declares ``Narrows: <global>``), so the corpus
screen must read the relationship id-natively, from either endpoint.

This pins ``mitos.check``'s two composition pieces:

* ``build_strong_edge_index(edges)`` — one pass over ``store.get_edges()`` → an
  orientation-blind ``{node_id: frozenset(partner_ids)}`` adjacency, keeping only the
  five ``_STRONG_RELATIONSHIP_FIELDS`` edge types, recording BOTH endpoints. Direct
  edges only — no lineage walk.
* ``screen_strong_edge_pairs(proposal_id, candidates, index)`` — drop every candidate
  whose ``node['id']`` is a strong-edge partner of ``proposal_id`` (either direction).
  Order-preserving, pure, no store.

Discipline (PATTERNS / plan §8): no mocks — construct ``Candidate``/edge dicts directly;
one real temp-SQLite round-trip pins the ``get_edges()`` dict-key contract. No async, no
LLM, no key. Run under ``./venv/bin/python -m pytest`` (bare ``python`` lacks the deps).
"""

import os
import subprocess
import sys
import tempfile
from typing import Dict, List

import pytest

from mitos.check import (
    StrongEdgeIndex,
    build_strong_edge_index,
    screen_strong_edge_pairs,
)
from mitos.conflict import Candidate, _STRONG_RELATIONSHIP_FIELDS
from mitos.parser import ParsedEntry
from mitos.store import GraphStore


# --------------------------------------------------------------------------- #
# Helpers — direct construction (pure/storeless functions; no fixtures needed)
# --------------------------------------------------------------------------- #

def _candidate(slug: str, node_id: str, score: float = 0.9) -> Candidate:
    """A Candidate whose ``node`` carries an ``id`` — the key the screen reads.

    Unlike ``test_conflict_screen.py``'s ``_cand`` (``node={}``): this screen is
    id-native, so ``candidate.node['id']`` MUST be present (a ``node={}`` would
    ``KeyError``).
    """
    return Candidate(slug=slug, score=score, node={"id": node_id}, state="active")


def _edge(source_id: str, target_id: str, edge_type: str) -> Dict[str, str]:
    """A synthetic ``get_edges()`` row — only the three keys the builder reads."""
    return {"source_id": source_id, "target_id": target_id, "edge_type": edge_type}


def _ids(candidates: List[Candidate]) -> List[str]:
    """The ``node['id']``s of a candidate list, in order (order-sensitive asserts)."""
    return [c.node["id"] for c in candidates]


# --------------------------------------------------------------------------- #
# Real-store round-trip (§8-1, §8-2) — the minimal harbor ``narrows`` shape
# --------------------------------------------------------------------------- #

@pytest.fixture
def temp_store() -> GraphStore:
    """A temporary file GraphStore booted to the live ladder head."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    store = GraphStore(path)
    yield store
    if os.path.exists(path):
        os.remove(path)


def _commit(store: GraphStore, slug: str, axiom: str, **rels) -> str:
    """Commits a decision and returns its content-hash node id."""
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = axiom
    entry.rejected_paths = "An alternative."
    for name, value in rels.items():
        setattr(entry, name, value)
    return store.commit_parsed_entry(entry).node_id


@pytest.fixture
def narrows_pair(temp_store: GraphStore):
    """The minimal harbor shape: a declares-nothing global + a scoped exception.

    Commit the global FIRST (it declares nothing), then the health endpoint that
    declares ``Narrows: <global>`` — the target must exist for the slug to resolve at
    reconcile time (the ordering the lineage tests demonstrate). Yields
    ``(store, global_id, health_id)``. This gives a clean SINGLE ``narrows`` edge in
    ``get_edges()`` — no golden-harness coupling, no third-node narrows muddying it.
    """
    global_id = _commit(
        temp_store, "harbor-all-endpoints-authenticated", "All endpoints authenticate."
    )
    health_id = _commit(
        temp_store,
        "harbor-health-endpoint-public",
        "The health endpoint is public.",
        narrows=["harbor-all-endpoints-authenticated"],
    )
    return temp_store, global_id, health_id


def test_harbor_narrows_pair_dropped_both_orientations(narrows_pair) -> None:
    """§8-1 (DoD-2 / T2 unit half): the harbor ``narrows`` pair is screened EITHER way.

    Build the index from a real ``store.get_edges()``. The pair must drop whichever
    node the sweep is iterating — the forward-only bug (a 0.97-confidence FP) removed
    structurally, with zero judgment involvement.
    """
    store, global_id, health_id = narrows_pair
    index = build_strong_edge_index(store.get_edges())

    global_as_candidate = _candidate("harbor-all-endpoints-authenticated", global_id)
    health_as_candidate = _candidate("harbor-health-endpoint-public", health_id)

    # Sweep the global (declares nothing forward) → health-public still drops.
    assert screen_strong_edge_pairs(global_id, [health_as_candidate], index) == []
    # Sweep the declarer → the global drops too.
    assert screen_strong_edge_pairs(health_id, [global_as_candidate], index) == []

    # And the index itself is orientation-blind: each is the other's partner.
    assert health_id in index.partners(global_id)
    assert global_id in index.partners(health_id)


def test_get_edges_exposes_the_dict_key_contract(narrows_pair) -> None:
    """§8-2: pin the real ``get_edges()`` dict keys the builder binds to.

    Fail loud here (not silently mis-screen) if the store ever renames a key — the
    cross-system data-level-join-key discipline (don't trust the field names, prove
    them against a real round-trip).
    """
    store, _global_id, _health_id = narrows_pair
    edges = store.get_edges()
    assert len(edges) == 1  # exactly the one authored narrows edge
    edge = edges[0]
    assert {"source_id", "target_id", "edge_type"} <= set(edge)
    assert edge["edge_type"] == "narrows"


# --------------------------------------------------------------------------- #
# Pure-logic cases (§8-3, 4, 5, 7, 8) — synthetic edge dicts + inline Candidates
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("strong_type", _STRONG_RELATIONSHIP_FIELDS)
def test_every_strong_type_screens_both_orientations(strong_type: str) -> None:
    """§8-3 (generalized): each of the five strong types screens either direction.

    An edge of any strong type between X and Y makes both ``screen(X, [Y])`` and
    ``screen(Y, [X])`` drop — the ``contradicts`` case (CONF-D9) plus the other four.
    """
    index = build_strong_edge_index([_edge("hX", "hY", strong_type)])
    assert screen_strong_edge_pairs("hX", [_candidate("y", "hY")], index) == []
    assert screen_strong_edge_pairs("hY", [_candidate("x", "hX")], index) == []


@pytest.mark.parametrize("weak_type", ["cites", "depends_on", "derives_from", "resolves"])
def test_weak_edges_do_not_screen(weak_type: str) -> None:
    """§8-4: a pair joined ONLY by a weak edge survives (reaches judgment).

    Weak edges express dependence, not a resolved tension, so they must never shield an
    undeclared conflict from the judge.
    """
    index = build_strong_edge_index([_edge("hX", "hY", weak_type)])
    survivor = _candidate("y", "hY")
    assert screen_strong_edge_pairs("hX", [survivor], index) == [survivor]
    assert screen_strong_edge_pairs("hY", [_candidate("x", "hX")], index) != []


def test_lineage_boundary_pair_is_judged_not_screened() -> None:
    """§8-5: direct-edge-only — a superseded declarer re-opens the pair.

    ``B narrows A`` then ``B' supersedes B``: the pair ``{A, B'}`` has NO direct strong
    edge, so it must reach judgment (the author never declared how the *new* B' relates
    to A). A lineage walk would over-screen and hide a real re-opened conflict.
    """
    index = build_strong_edge_index([
        _edge("hB", "hA", "narrows"),        # B narrows A
        _edge("hBprime", "hB", "supersedes"),  # B' supersedes B
    ])
    # A's partners are {B} only — NOT the transitively-related B'.
    assert index.partners("hA") == frozenset({"hB"})
    assert "hBprime" not in index.partners("hA")

    b_prime = _candidate("b-prime", "hBprime")
    assert screen_strong_edge_pairs("hA", [b_prime], index) == [b_prime]


def test_strong_type_set_is_single_sourced_from_conflict() -> None:
    """§8-7: the screen binds the EXACT ``_STRONG_RELATIONSHIP_FIELDS`` object.

    Identity (``is``) against the imported constant — mirrors
    ``test_mutation_edge_constants_in_lockstep``. A future edit to the strong set (a
    sixth type, a dropped one) must flow through one edit, never a hand-copied 5-tuple
    in ``check.py`` (binding ``_MUTATION_EDGE_FIELDS`` would silently miss
    ``contradicts``/``corrects``).
    """
    import mitos.check as check

    assert check._STRONG_RELATIONSHIP_FIELDS is _STRONG_RELATIONSHIP_FIELDS
    assert set(_STRONG_RELATIONSHIP_FIELDS) == {
        "supersedes", "amends", "narrows", "contradicts", "corrects"
    }


def test_empty_index_screens_nothing() -> None:
    """§8-8: an empty index (no edges) drops nothing — every candidate survives."""
    index = build_strong_edge_index([])
    candidates = [_candidate("a", "hA"), _candidate("b", "hB")]
    assert screen_strong_edge_pairs("hX", candidates, index) == candidates
    assert index.partners("hAnything") == frozenset()


def test_node_with_no_strong_edges_screens_nothing() -> None:
    """§8-8: a swept node absent from the index screens nothing (partners → ∅ default).

    The index carries strong edges among OTHER nodes; the swept node shares none, so the
    healthy common case is a full pass-through.
    """
    index = build_strong_edge_index([_edge("hP", "hQ", "narrows")])
    candidates = [_candidate("a", "hA"), _candidate("b", "hB")]
    assert screen_strong_edge_pairs("hX", candidates, index) == candidates


def test_screen_is_order_preserving_and_drops_only_partners() -> None:
    """The survivors keep their input order; only the strong-edge partner is dropped."""
    index = build_strong_edge_index([_edge("hProp", "hMid", "amends")])
    first = _candidate("first", "hFirst")
    partner = _candidate("mid", "hMid")   # the one strong-edge partner
    last = _candidate("last", "hLast")
    survivors = screen_strong_edge_pairs("hProp", [first, partner, last], index)
    assert _ids(survivors) == ["hFirst", "hLast"]


def test_multiple_strong_partners_accumulate_and_all_drop() -> None:
    """A node with strong edges to TWO partners drops both in one screen call.

    Pins the builder's per-node accumulation (a second strong edge must add to the
    partner set, not replace it) — the shape 2b relies on for a node that declares
    several strong relationships (e.g. ``supersedes`` one decision and ``narrows``
    another).
    """
    index = build_strong_edge_index([
        _edge("hProp", "hOne", "supersedes"),
        _edge("hTwo", "hProp", "narrows"),  # reverse orientation — still a partner
    ])
    assert index.partners("hProp") == frozenset({"hOne", "hTwo"})
    one, two = _candidate("one", "hOne"), _candidate("two", "hTwo")
    fresh = _candidate("fresh", "hFresh")
    assert screen_strong_edge_pairs("hProp", [one, fresh, two], index) == [fresh]


def test_weak_and_strong_mixed_keeps_only_the_weak_partner() -> None:
    """Two candidates, one strong-linked and one weak-linked → only the strong drops."""
    index = build_strong_edge_index([
        _edge("hProp", "hStrong", "contradicts"),
        _edge("hProp", "hWeak", "cites"),
    ])
    strong = _candidate("strong", "hStrong")
    weak = _candidate("weak", "hWeak")
    assert screen_strong_edge_pairs("hProp", [strong, weak], index) == [weak]


def test_self_edge_does_not_create_self_partnership() -> None:
    """A corrupt ``source_id == target_id`` strong edge leaves the node partnerless.

    The write path rejects a self-edge, but the rebuildable graph could carry one
    out-of-band; skipping it in the builder mirrors ``get_contradictions``'s ``!= ?``
    guard. Self-drop is ``screen_candidates``'s ``own_slug`` job, not this screen's.
    """
    index = build_strong_edge_index([_edge("hX", "hX", "narrows")])
    assert index.partners("hX") == frozenset()
    own = _candidate("x", "hX")
    assert screen_strong_edge_pairs("hX", [own], index) == [own]


def test_index_returns_the_declared_type() -> None:
    """``StrongEdgeIndex`` is the plan's named seam (2b's sweep planner inherits it)."""
    index = build_strong_edge_index([_edge("hX", "hY", "narrows")])
    assert isinstance(index, StrongEdgeIndex)


# --------------------------------------------------------------------------- #
# The Tier-1 dependency-free contract (§8-9, scout recommendation)
# --------------------------------------------------------------------------- #

def test_importing_check_drags_no_heavy_dependency() -> None:
    """A fresh interpreter importing ``mitos.check`` pulls no LLM dep.

    ``check.py`` sits above ``conflict.py`` in the import DAG and must carry the same
    Tier-1 discipline: no ``anthropic``/genai at module scope (the judge arrives
    injected in later phases). Mirrors
    ``test_importing_conflict_drags_no_heavy_dependency`` — locks the contract before
    2c/3a are tempted to add a heavy module-scope import. Run in a subprocess so the
    assertion sees a clean import graph.
    """
    heavy = ["anthropic", "google", "google.genai"]
    probe = (
        "import sys; import mitos.check; "
        f"leaked = [m for m in {heavy!r} if m in sys.modules]; "
        "assert not leaked, 'mitos.check leaked heavy deps: ' + repr(leaked); "
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"dep-free import probe failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout
