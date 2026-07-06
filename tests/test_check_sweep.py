"""Tests for the corpus sweep assembly (Phase 2b) — snapshot, dedup, orientation, grouping.

``mitos check``'s deterministic plan half: snapshot the live sweep set once
(``snapshot_corpus`` — one ``get_active_decisions`` + one ``get_edges`` → the 2a
strong-edge index built ONCE), sweep each node lazily through the shipped pipeline
stages (``iter_sweep``: gather → 2a's either-direction screen → ``screen_candidates``),
dedup the discovered unordered pairs corpus-wide (``dedup_oriented_pairs`` — each pair
exactly once, oriented to the lexicographically smaller content hash), and group the
oriented pairs into judgment-sized batches (``group_judgment_batches``).

The load-bearing properties under test (plan §3):

* healthy-empty (``[]``) vs degraded (``Unavailable``) are different TYPES — never blur;
* laziness IS the breaker seam — no gather work beyond the last consumed yield;
* each unordered pair exactly once, retention order-independent (never keep-first);
* orientation is replayable and discovery-free (lex-smaller hash, derived at test time
  from the fixture's ACTUAL computed ids — real hashes fall where they fall);
* grouping is a pure function of the pair set — byte-identical across sweep orders.

Discipline (PATTERNS + scout brief): hand-rolled synchronous fakes (the keyed
``_conflict_helpers`` variants — per-proposal neighbourhoods); a real temp ``GraphStore``
seeded via ``commit_parsed_entry`` (never embeds → keyless + deterministic, and the ids
are REAL content hashes); zero LLM, zero telemetry, zero writes. Run under
``./venv/bin/python -m pytest`` (bare ``python`` lacks the deps).
"""

import os
import tempfile
from typing import Any, Dict, List, Optional

import pytest

from mitos.check import (
    CorpusPair,
    CorpusSnapshot,
    JudgmentGroup,
    NodeSweep,
    build_strong_edge_index,
    dedup_oriented_pairs,
    group_judgment_batches,
    iter_sweep,
    snapshot_corpus,
    sweep_node,
)
from mitos.conflict import (
    Candidate,
    ConflictUnavailableReason,
    Unavailable,
    judge_input_from_node,
)
from mitos.errors import DatabaseError, EmbeddingError, VectorStoreError
from mitos.parser import ParsedEntry
from mitos.store import GraphStore

from _conflict_helpers import _keyed_substrate, _match


# --------------------------------------------------------------------------- #
# Fixtures — offline env + a real temp store (the test_check_screen idiom)
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key, no reachable service — the injected fakes are the only substrate."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def temp_store() -> GraphStore:
    """A temporary file GraphStore booted to the live ladder head."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    store = GraphStore(path)
    yield store
    if os.path.exists(path):
        os.remove(path)


def _commit(
    store: GraphStore,
    slug: str,
    axiom: str,
    *,
    scope: Optional[List[str]] = None,
    **rels: List[str],
) -> str:
    """Commits a decision and returns its content-hash node id (real hashes for free)."""
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = axiom
    entry.rejected_paths = "An alternative."
    if scope is not None:
        entry.scope = scope
    for name, value in rels.items():
        setattr(entry, name, value)
    return store.commit_parsed_entry(entry).node_id


def _pure_node(node_id: str, slug: str) -> Dict[str, Any]:
    """A minimal hand-built 'hydrated node' — the keys the pure post-passes read."""
    return {
        "id": node_id,
        "slug": slug,
        "core_axiom": f"Axiom of {slug}.",
        "rejected_paths": "",
        "scope": [],
    }


def _cand(node: Dict[str, Any], score: float) -> Candidate:
    """A Candidate wrapping a node dict — the shape gather hands the post-passes."""
    return Candidate(slug=node["slug"], score=score, node=node, state="active")


def _sweep_all(store: GraphStore, snapshot: CorpusSnapshot, embed: Any, vector: Any) -> List[NodeSweep]:
    """Consumes the full lazy sweep — the healthy-run shape (2c stops early on a trip)."""
    return list(
        iter_sweep(snapshot, embed_provider=embed, vector_store=vector, store=store)
    )


def _keys(pairs: List[CorpusPair]) -> List[tuple]:
    """The oriented pair keys, in output order (order-sensitive determinism asserts)."""
    return [(pair.proposal_hash, pair.partner_hash) for pair in pairs]


# --------------------------------------------------------------------------- #
# The §6.4 trace fixture — THE integration test (and W3's wiring proof)
# --------------------------------------------------------------------------- #

@pytest.fixture
def trace_corpus(temp_store: GraphStore):
    """The §6.4 trace corpus: five decisions A–E, ``B narrows A``, ``E supersedes D``.

    Commit ordering is load-bearing: a declared relationship's target must exist when
    the declarer commits (slug resolution at reconcile time) — A before B, D before E.
    The snapshot is {A, B, C, E}; D is superseded out of the active view. Yields
    ``(store, ids, axioms)`` — ids are the ACTUAL computed content hashes; every
    orientation expectation below derives from them at test time (never hardcoded —
    the §6.4 ``hash(E) < hash(C)`` illustration is not authorable).
    """
    axioms = {
        "trace-a": "Cache reads go through the shared LRU layer.",
        "trace-b": "Session cache reads bypass the shared LRU layer.",
        "trace-c": "Every cache layer is process-local.",
        "trace-d": "Cache invalidation is broadcast over the bus.",
        "trace-e": "Cache invalidation is polled from the bus.",
    }
    ids = {}
    ids["trace-a"] = _commit(temp_store, "trace-a", axioms["trace-a"])
    ids["trace-b"] = _commit(
        temp_store, "trace-b", axioms["trace-b"], narrows=["trace-a"]
    )
    ids["trace-c"] = _commit(temp_store, "trace-c", axioms["trace-c"])
    ids["trace-d"] = _commit(temp_store, "trace-d", axioms["trace-d"])
    ids["trace-e"] = _commit(
        temp_store, "trace-e", axioms["trace-e"], supersedes=["trace-d"]
    )
    return temp_store, ids, axioms


def _trace_substrate(axioms: Dict[str, str]):
    """The trace's scripted neighbourhoods (sim ≥ floor throughout; D at 0.78 ≥ floor
    so its absence from every pair proves gather's live re-verify dropped it — not
    the similarity floor)."""
    return _keyed_substrate(
        {
            axioms["trace-a"]: [_match("trace-b", 0.88), _match("trace-c", 0.81)],
            axioms["trace-b"]: [_match("trace-a", 0.88)],
            axioms["trace-c"]: [
                _match("trace-a", 0.81),
                _match("trace-e", 0.79),
                _match("trace-d", 0.78),
            ],
            axioms["trace-e"]: [_match("trace-c", 0.79)],
        }
    )


def test_trace_declared_pair_screened_both_sides_and_d_never_seen(trace_corpus) -> None:
    """§9-1 (W3 wiring proof): the declared pair drops from EITHER side's sweep.

    The screen is wired between gather and ``screen_candidates`` over a REAL
    ``get_edges()``: B (candidate-side edge) is dropped from A's sweep, and A from
    B's — B's sweep lands healthy-EMPTY (a list, never ``Unavailable``). D is
    superseded: never iterated (absent from the snapshot) and, fed as a canned KNN
    match above the floor, dropped by gather's live re-verify — never a candidate.
    """
    store, ids, axioms = trace_corpus
    snapshot = snapshot_corpus(store)
    assert {node["id"] for node in snapshot.nodes} == {
        ids["trace-a"], ids["trace-b"], ids["trace-c"], ids["trace-e"]
    }

    embed, vector = _trace_substrate(axioms)
    sweeps = _sweep_all(store, snapshot, embed, vector)
    by_id = {sweep.node["id"]: sweep for sweep in sweeps}

    # Forward orientation: B declared `narrows A`, so A's sweep keeps only C.
    assert [c.node["id"] for c in by_id[ids["trace-a"]].result] == [ids["trace-c"]]
    # Reverse orientation: A drops from B's sweep — healthy empty, type-distinct.
    b_result = by_id[ids["trace-b"]].result
    assert isinstance(b_result, list) and b_result == []
    # D is never a candidate anywhere (live re-verify, not the floor, dropped it).
    for sweep in sweeps:
        assert all(c.node["id"] != ids["trace-d"] for c in sweep.result)


def test_trace_assembles_exactly_the_two_oriented_pairs(trace_corpus) -> None:
    """§9-1 continued: exactly {A,C} and {C,E}, each once, oriented to the lex-smaller
    ACTUAL hash — and byte-identical when the same sweeps arrive in reversed order."""
    store, ids, axioms = trace_corpus
    snapshot = snapshot_corpus(store)
    embed, vector = _trace_substrate(axioms)
    sweeps = _sweep_all(store, snapshot, embed, vector)

    pairs = dedup_oriented_pairs(sweeps)
    expected_keys = sorted(
        [
            tuple(sorted((ids["trace-a"], ids["trace-c"]))),
            tuple(sorted((ids["trace-c"], ids["trace-e"]))),
        ]
    )
    assert _keys(pairs) == expected_keys
    for pair in pairs:
        assert pair.proposal_hash < pair.partner_hash
        assert pair.proposal_node["id"] == pair.proposal_hash
        assert pair.partner_node["id"] == pair.partner_hash
    # Replayability: sweep iteration order must not leak into the plan structures.
    assert dedup_oriented_pairs(list(reversed(sweeps))) == pairs


def test_trace_groups_are_a_pure_function_of_the_pair_set(trace_corpus) -> None:
    """§9-1 tail: groups keyed+ordered by proposal hash, partners partner-hash-sorted,
    byte-identical across sweep orders — the whole plan is replayable."""
    store, ids, axioms = trace_corpus
    snapshot = snapshot_corpus(store)
    embed, vector = _trace_substrate(axioms)
    sweeps = _sweep_all(store, snapshot, embed, vector)

    pairs = dedup_oriented_pairs(sweeps)
    groups = group_judgment_batches(pairs)
    assert groups == group_judgment_batches(
        dedup_oriented_pairs(list(reversed(sweeps)))
    )
    assert [g.proposal_hash for g in groups] == sorted({p.proposal_hash for p in pairs})
    assert {key for g in groups for key in _keys(list(g.pairs))} == set(_keys(pairs))
    for group in groups:
        assert isinstance(group, JudgmentGroup)
        assert list(group.pairs) == sorted(group.pairs, key=lambda p: p.partner_hash)
        assert group.proposal_node["id"] == group.proposal_hash


# --------------------------------------------------------------------------- #
# §9-2 — drop-before-floor-before-truncate is observable at the sweep level
# --------------------------------------------------------------------------- #

def test_declared_strong_neighbour_never_shadows_an_undeclared_conflict(
    temp_store: GraphStore,
) -> None:
    """With ``top_k=1``, a declared-strong candidate at 0.95 must not consume the one
    slot: the strong-edge drop runs UPSTREAM of truncation, so the undeclared 0.80
    candidate survives. (Screening after ``screen_candidates`` would return [].)"""
    _commit(temp_store, "shadow-base", "Shadow base axiom.")
    _commit(
        temp_store, "shadow-declared", "Shadow declared axiom.", narrows=["shadow-base"]
    )
    id_undeclared = _commit(temp_store, "shadow-undeclared", "Shadow undeclared axiom.")

    index = build_strong_edge_index(temp_store.get_edges())
    base_node = temp_store.get_node_by_slug("shadow-base")
    embed, vector = _keyed_substrate(
        {
            "Shadow base axiom.": [
                _match("shadow-declared", 0.95),
                _match("shadow-undeclared", 0.80),
            ]
        }
    )
    result = sweep_node(
        base_node,
        edge_index=index,
        embed_provider=embed,
        vector_store=vector,
        store=temp_store,
        top_k=1,
    )
    assert [c.node["id"] for c in result] == [id_undeclared]


# --------------------------------------------------------------------------- #
# §9-3 / §9-4 — dedup exactly-once + order-independent retention + orientation
# --------------------------------------------------------------------------- #

def test_dedup_is_exactly_once_and_retention_is_order_independent() -> None:
    """A pair discovered from both sides yields ONE CorpusPair, and the retained
    fields (score, node dicts) are identical whichever sweep order the discoveries
    arrive in. The two discoveries carry DIFFERENT scores, so a keep-first rule
    would be caught red-handed here (KD3's order-independent retention pin)."""
    lo = _pure_node("hash-a", "pair-lo")
    hi = _pure_node("hash-b", "pair-hi")
    sweep_lo = NodeSweep(node=lo, result=[_cand(hi, 0.91)])
    sweep_hi = NodeSweep(node=hi, result=[_cand(lo, 0.83)])

    forward = dedup_oriented_pairs([sweep_lo, sweep_hi])
    backward = dedup_oriented_pairs([sweep_hi, sweep_lo])

    assert len(forward) == 1
    assert forward == backward  # byte-identical, retained score + dicts included
    (pair,) = forward
    assert (pair.proposal_hash, pair.partner_hash) == ("hash-a", "hash-b")
    # The retention rule is deterministic in the PAIR, not the sweep order: whichever
    # order arrived, the same discovery's score is kept.
    assert pair.score == backward[0].score
    assert group_judgment_batches(forward) == group_judgment_batches(backward)


def test_orientation_is_discovery_free_both_branches() -> None:
    """§9-4: the proposal is the lex-smaller hash whichever side discovered the pair —
    both the discoverer-wins and the undiscovered-side-wins branches."""
    node_a = _pure_node("hash-a", "orient-a")
    node_b = _pure_node("hash-b", "orient-b")

    # Branch 1: the discoverer IS the proposal (a discovers b; hash-a < hash-b).
    (pair,) = dedup_oriented_pairs(
        [
            NodeSweep(node=node_a, result=[_cand(node_b, 0.9)]),
            NodeSweep(node=node_b, result=[]),
        ]
    )
    assert pair.proposal_hash == "hash-a"
    assert pair.proposal_node["slug"] == "orient-a"
    assert pair.partner_node["slug"] == "orient-b"

    # Branch 2: the UNDISCOVERING side is the proposal (only b's sweep found the pair).
    (pair2,) = dedup_oriented_pairs(
        [
            NodeSweep(node=node_a, result=[]),
            NodeSweep(node=node_b, result=[_cand(node_a, 0.9)]),
        ]
    )
    assert pair2.proposal_hash == "hash-a"
    assert pair2.proposal_node["slug"] == "orient-a"  # arrived via Candidate.node
    assert pair2.partner_node["slug"] == "orient-b"


# --------------------------------------------------------------------------- #
# §9-5 — grouping + splitting
# --------------------------------------------------------------------------- #

def test_grouping_splits_oversize_groups_in_partner_hash_order() -> None:
    """Three pairs sharing one oriented proposal, ``top_k=2`` → split 2+1 in
    partner-hash order (fed shuffled to prove the sort is the function's)."""
    proposal = _pure_node("hash-a", "group-prop")
    partners = [_pure_node(f"hash-p{i}", f"group-p{i}") for i in (1, 2, 3)]
    pairs = [
        CorpusPair(
            proposal_hash="hash-a",
            partner_hash=partner["id"],
            proposal_node=proposal,
            partner_node=partner,
            score=0.8,
        )
        for partner in partners
    ]
    groups = group_judgment_batches([pairs[2], pairs[0], pairs[1]], top_k=2)
    assert [len(g.pairs) for g in groups] == [2, 1]
    assert [p.partner_hash for p in groups[0].pairs] == ["hash-p1", "hash-p2"]
    assert [p.partner_hash for p in groups[1].pairs] == ["hash-p3"]
    assert all(g.proposal_hash == "hash-a" for g in groups)
    assert all(g.proposal_node["id"] == "hash-a" for g in groups)


def test_grouping_orders_by_proposal_hash_and_exact_top_k_is_one_group() -> None:
    """Groups come out proposal-hash-sorted regardless of input order; a group with
    exactly ``top_k`` partners does NOT split (the boundary case)."""
    prop_a = _pure_node("hash-a", "gb-a")
    prop_b = _pure_node("hash-b", "gb-b")
    partner_y = _pure_node("hash-y", "gb-y")
    partner_z = _pure_node("hash-z", "gb-z")
    pair_b = CorpusPair("hash-b", "hash-z", prop_b, partner_z, 0.8)
    pair_a1 = CorpusPair("hash-a", "hash-y", prop_a, partner_y, 0.8)
    pair_a2 = CorpusPair("hash-a", "hash-z", prop_a, partner_z, 0.8)

    groups = group_judgment_batches([pair_b, pair_a2, pair_a1], top_k=2)
    assert [g.proposal_hash for g in groups] == ["hash-a", "hash-b"]
    # Exactly top_k partners → ONE group, partner-hash-sorted inside.
    assert [p.partner_hash for p in groups[0].pairs] == ["hash-y", "hash-z"]
    assert [p.partner_hash for p in groups[1].pairs] == ["hash-z"]


# --------------------------------------------------------------------------- #
# §9-6 / §9-7 — typed per-node degradation + laziness (the breaker seam)
# --------------------------------------------------------------------------- #

def test_embed_fault_degrades_that_node_only_and_dedup_skips_it(
    temp_store: GraphStore,
) -> None:
    """§9-6: one node's embed raising degrades THAT NodeSweep only — typed
    ``Unavailable(EMBEDDING)``, never blurred with a healthy ``[]``. Dedup skips the
    degraded sweep without raising, and the pair the OTHER side discovered with the
    degraded node still stands (partial coverage is labeled by 2c, not compensated)."""
    id_a = _commit(temp_store, "deg-a", "Degradation axiom alpha.")
    id_b = _commit(temp_store, "deg-b", "Degradation axiom beta.")
    id_c = _commit(temp_store, "deg-c", "Degradation axiom gamma.")

    snapshot = snapshot_corpus(temp_store)
    embed, vector = _keyed_substrate(
        {
            "Degradation axiom alpha.": [_match("deg-b", 0.9)],
            "Degradation axiom beta.": [],
            "Degradation axiom gamma.": [],
        },
        embed_raises={"Degradation axiom beta.": EmbeddingError("quota exhausted")},
    )
    sweeps = _sweep_all(temp_store, snapshot, embed, vector)
    by_id = {sweep.node["id"]: sweep for sweep in sweeps}

    degraded = by_id[id_b].result
    assert isinstance(degraded, Unavailable)
    assert degraded.reason is ConflictUnavailableReason.EMBEDDING
    # The type fork: a healthy node with zero survivors is a LIST — consumers fork on
    # isinstance, never on emptiness.
    healthy_empty = by_id[id_c].result
    assert isinstance(healthy_empty, list) and healthy_empty == []

    pairs = dedup_oriented_pairs(sweeps)  # no raise — degraded contributes zero pairs
    assert _keys(pairs) == [tuple(sorted((id_a, id_b)))]


def test_vector_fault_degrades_typed_through_the_keyed_substrate(
    temp_store: GraphStore,
) -> None:
    """The ``vector_raises`` seam (folded into ``_keyed_substrate`` FOR 2c's breaker
    tests) degrades that node's sweep to ``Unavailable(VECTOR_STORE)`` — pinned here
    so the helper path 2c inherits is proven, not shipped blind."""
    _commit(temp_store, "vec-a", "Vector fault axiom.")
    snapshot = snapshot_corpus(temp_store)
    embed, vector = _keyed_substrate(
        {"Vector fault axiom.": []},
        vector_raises={"Vector fault axiom.": VectorStoreError("qdrant severed")},
    )
    (sweep,) = _sweep_all(temp_store, snapshot, embed, vector)
    assert isinstance(sweep.result, Unavailable)
    assert sweep.result.reason is ConflictUnavailableReason.VECTOR_STORE
    assert dedup_oriented_pairs([sweep]) == []


def test_iter_sweep_is_lazy_no_gather_work_beyond_consumption(
    temp_store: GraphStore,
) -> None:
    """§9-7: consume up to the first ``Unavailable`` and stop — the substrate fakes
    saw ONLY the consumed nodes (no embed, no KNN for the remainder). This is 2c's
    breaker seam: stopping consumption structurally skips the rest of the corpus.

    The degraded node is picked from the ACTUAL snapshot order at test time (row
    order is a DB accident — never assumed)."""
    for name in ("lazy-a", "lazy-b", "lazy-c", "lazy-d"):
        _commit(temp_store, name, f"Laziness axiom {name}.")
    snapshot = snapshot_corpus(temp_store)
    texts = [node["core_axiom"] for node in snapshot.nodes]

    embed, vector = _keyed_substrate(
        {text: [] for text in texts},
        embed_raises={texts[1]: EmbeddingError("substrate severed")},
    )
    consumed: List[NodeSweep] = []
    for sweep in iter_sweep(
        snapshot, embed_provider=embed, vector_store=vector, store=temp_store
    ):
        consumed.append(sweep)
        if isinstance(sweep.result, Unavailable):
            break

    assert len(consumed) == 2
    assert [text for text, _ in embed.calls] == texts[:2]  # nothing beyond the trip
    assert len(vector.queried) == 1  # the degraded node raised before its KNN


# --------------------------------------------------------------------------- #
# §9-8 — scope: proposal-set filter only; recall stays scope-blind
# --------------------------------------------------------------------------- #

def test_scope_filters_the_sweep_set_only_recall_stays_scope_blind(
    temp_store: GraphStore,
) -> None:
    """A scoped snapshot sweeps only matching nodes, but a scoped sweep node pairing
    with an OUT-OF-SCOPE live candidate is a feature (CONF-D2) — the pair survives
    all the way to a group."""
    id_scoped = _commit(temp_store, "scoped-s", "Scoped sweep axiom.", scope=["api"])
    id_global = _commit(temp_store, "global-t", "Out of scope axiom.", scope=["cache"])

    snapshot = snapshot_corpus(temp_store, scope="api")
    assert [node["id"] for node in snapshot.nodes] == [id_scoped]

    embed, vector = _keyed_substrate(
        {"Scoped sweep axiom.": [_match("global-t", 0.9)]}
    )
    sweeps = _sweep_all(temp_store, snapshot, embed, vector)
    pairs = dedup_oriented_pairs(sweeps)
    assert _keys(pairs) == [tuple(sorted((id_scoped, id_global)))]
    groups = group_judgment_batches(pairs)
    assert len(groups) == 1 and len(groups[0].pairs) == 1


def test_zero_match_scope_is_healthy_empty_everything(temp_store: GraphStore) -> None:
    """A zero-match scope yields an empty snapshot → empty sweep, no pairs, no groups —
    healthy, not degraded, no error (the '0 of N' wording is 3a's; 2b returns the
    faithful empty)."""
    _commit(temp_store, "any-node", "Some axiom.", scope=["api"])
    snapshot = snapshot_corpus(temp_store, scope="nomatch")
    assert snapshot.nodes == ()

    embed, vector = _keyed_substrate({})  # never called — nothing to sweep
    sweeps = _sweep_all(temp_store, snapshot, embed, vector)
    assert sweeps == []
    assert dedup_oriented_pairs(sweeps) == []
    assert group_judgment_batches([]) == []
    assert embed.calls == [] and vector.queried == []


# --------------------------------------------------------------------------- #
# §9-9 / §9-10 — self-echo dropped; both sides feed the node adapter
# --------------------------------------------------------------------------- #

def test_knn_self_echo_never_becomes_a_pair(temp_store: GraphStore) -> None:
    """§9-9: a KNN result echoing the swept node's own slug is dropped at the sweep
    level (``own_slug`` threading proven) — ``{X, X}`` never reaches dedup."""
    id_x = _commit(temp_store, "echo-x", "Echo axiom ex.")
    id_y = _commit(temp_store, "echo-y", "Echo axiom why.")
    snapshot = snapshot_corpus(temp_store)
    embed, vector = _keyed_substrate(
        {
            "Echo axiom ex.": [_match("echo-x", 0.99), _match("echo-y", 0.90)],
            "Echo axiom why.": [],
        }
    )
    sweeps = _sweep_all(temp_store, snapshot, embed, vector)
    by_id = {sweep.node["id"]: sweep for sweep in sweeps}
    assert [c.node["id"] for c in by_id[id_x].result] == [id_y]
    assert _keys(dedup_oriented_pairs(sweeps)) == [tuple(sorted((id_x, id_y)))]


def test_both_sides_of_a_pair_feed_the_node_adapter(temp_store: GraphStore) -> None:
    """§9-10: when orientation makes the UNDISCOVERING side the proposal, both
    ``proposal_node`` and ``partner_node`` project cleanly through
    ``judge_input_from_node`` — non-empty axioms, the phantom-empty guard exercised
    on 2b's real structures. The discovery direction is built AFTER comparing the
    actual hashes, so the branch under test occurs by construction."""
    id_one = _commit(temp_store, "adapter-one", "Adapter axiom one.")
    id_two = _commit(temp_store, "adapter-two", "Adapter axiom two.")
    lo_slug, hi_slug = (
        ("adapter-one", "adapter-two") if id_one < id_two else ("adapter-two", "adapter-one")
    )
    lo_node = temp_store.get_node_by_slug(lo_slug)
    hi_node = temp_store.get_node_by_slug(hi_slug)

    # The HIGHER-hash node discovers the pair → the lex-smaller, undiscovering side
    # becomes the proposal, sourced from Candidate.node.
    (pair,) = dedup_oriented_pairs(
        [NodeSweep(node=hi_node, result=[_cand(lo_node, 0.9)])]
    )
    assert pair.proposal_hash == lo_node["id"]

    proposal_input = judge_input_from_node(pair.proposal_node)
    partner_input = judge_input_from_node(pair.partner_node)
    assert proposal_input.axiom == lo_node["core_axiom"] and proposal_input.axiom
    assert partner_input.axiom == hi_node["core_axiom"] and partner_input.axiom


# --------------------------------------------------------------------------- #
# §9-11 — store faults PROPAGATE (KD5 inheritance; never masked, never empty)
# --------------------------------------------------------------------------- #

class _EdgeFaultStore:
    """``get_edges`` raises — snapshot_corpus must propagate, never fall back empty."""

    def get_active_decisions(self, scope: Optional[str] = None) -> List[Dict[str, Any]]:
        return []

    def get_edges(self) -> List[Dict[str, str]]:
        raise DatabaseError("edges table unreadable")


class _ReVerifyFaultStore:
    """Gather's S3 re-verify read raises — a graph fault, not semantic degradation."""

    def get_node_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        raise DatabaseError("graph read failed mid-sweep")


def test_get_edges_fault_propagates_from_snapshot_corpus() -> None:
    """§9-11a: a ``get_edges()`` fault raises out of ``snapshot_corpus`` — no fallback
    empty index (which would silently screen nothing and mislabel every pair)."""
    with pytest.raises(DatabaseError):
        snapshot_corpus(_EdgeFaultStore())


def test_gather_internal_store_fault_propagates_out_of_iter_sweep() -> None:
    """§9-11b: a graph-store fault inside gather's re-verify propagates out of the
    sweep — never converted to ``Unavailable`` (KD5: only the two semantic-substrate
    faults degrade)."""
    node = _pure_node("hash-a", "fault-a")
    snapshot = CorpusSnapshot(nodes=(node,), edge_index=build_strong_edge_index([]))
    embed, vector = _keyed_substrate(
        {node["core_axiom"]: [_match("neighbour", 0.9)]}
    )
    with pytest.raises(DatabaseError):
        list(
            iter_sweep(
                snapshot,
                embed_provider=embed,
                vector_store=vector,
                store=_ReVerifyFaultStore(),
            )
        )


# --------------------------------------------------------------------------- #
# The build-once seam (plan gotcha) — one snapshot read each, never per node
# --------------------------------------------------------------------------- #

class _CountingReadsStore:
    """Wraps a real store, counting the two snapshot reads (the build-once probe)."""

    def __init__(self, inner: GraphStore) -> None:
        self._inner = inner
        self.active_calls = 0
        self.edges_calls = 0

    def get_active_decisions(self, scope: Optional[str] = None) -> List[Dict[str, Any]]:
        self.active_calls += 1
        return self._inner.get_active_decisions(scope)

    def get_edges(self) -> List[Dict[str, str]]:
        self.edges_calls += 1
        return self._inner.get_edges()

    def get_node_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        return self._inner.get_node_by_slug(slug)

    def get_node_state(self, node_id: str) -> str:
        return self._inner.get_node_state(node_id)


def test_snapshot_reads_the_store_exactly_once_each(temp_store: GraphStore) -> None:
    """``snapshot_corpus`` makes ONE ``get_active_decisions`` + ONE ``get_edges`` call,
    and a full sweep afterwards makes no more of either — the edge index is built once
    at run start, never rebuilt per node (O(edges), not O(edges)×O(nodes))."""
    axioms = {}
    for name in ("once-a", "once-b", "once-c"):
        axiom = f"Build-once axiom {name}."
        _commit(temp_store, name, axiom)
        axioms[axiom] = []

    spy = _CountingReadsStore(temp_store)
    snapshot = snapshot_corpus(spy)
    assert (spy.active_calls, spy.edges_calls) == (1, 1)

    embed, vector = _keyed_substrate(axioms)
    _sweep_all(spy, snapshot, embed, vector)
    assert (spy.active_calls, spy.edges_calls) == (1, 1)  # untouched by the sweep
