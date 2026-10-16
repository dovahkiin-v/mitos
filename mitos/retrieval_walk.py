"""The shared identity walk: one over-fetched vector call, resolved by point id.

Every ranked read — MCP ``surface_decisions`` and ``query_decisions``' ranked
branch, ``mitos surface`` and ``mitos query`` — and the blackout they render map
each vector hit to its node through the hit's **point id**, never its payload
slug. A slug is a handle: mutable (MI-2) and, across a same-slug supersession,
shared (MI-13). A slug join hands a retired point's score to the live decision
that took its name and loses a renamed decision whose payload still carries the
old one. The point id is the node's identity (``hash_to_uuid(node_id)``: the
first 128 bits of its content hash), so a hit resolves to exactly the node it was
written for (ADR ``vector-hits-resolve-to-nodes-by-point-id-prefix-never-by-payload-slug``).

The walk makes three reads and no more (ADR
``readers-fill-their-window-with-live-hits-from-one-over-fetched-call``): **one**
``VectorStore.query`` at :data:`RANKED_OVERFETCH_BOUND` with ``with_payload=False``,
**one** batched ``resolve_point_ids`` over every hit, and **one** batched
``get_modifiers_map`` over the window and the first ``limit`` retired hits. In
between, :func:`fill` walks the hits in memory until it holds ``limit`` live
decisions plus one more for the edge, so retired and open-question points never
take a caller's slots. It never re-fetches.

A stamp fault costs the stamps alone: the walk returns ``stamps={}`` with
``stamps_unread=True`` and the window stands (lock A, failure state). A resolver
fault (``DatabaseError``) propagates, and each surface degrades it as a store
fault. A wrong store double raises :class:`WalkContractError`, which every
surface re-raises ahead of its degraded arm, so a double that absorbs every call
(a ``MagicMock``) fails loud instead of passing an absence-asserting row over an
empty window.

Why a module of its own
-----------------------
The walk cannot live in ``store.py``: the conflict gather (``conflict.gather_candidates``)
calls it, and
the check family's fence (``tests/test_conflict_closeout.py``) pins that
``store.py`` never enters that family's runtime-import closure. It cannot live in
``recall.py`` or ``display.py`` either — both sit inside that closure already and
must not grow a vector-store concern. So the walk is a leaf of its own over a
duck-typed store, and :class:`PointResolution` moved here from ``store.py`` so the
walk can ``isinstance``-check the resolver's result without importing the store
(``store.py`` and ``protocols.py`` import it from here; ``mitos.store.PointResolution``
is this class).

Its admission to the check family's fence
-----------------------------------------
Stdlib (``dataclasses``, ``typing``) plus ``mitos.errors``, which is already a
fence member and imports only ``typing``: the leaf's runtime closure is exactly
``{retrieval_walk.py, errors.py}``. It reads two methods off a store it is handed
and writes nothing, so admitting it costs the fence no write path and no
``store``/``vector_store``/``telemetry`` reach. ``conflict.py``'s import of it grew
the exact-set pin by this module alone. Keep it that way: never import ``store``,
``protocols``, ``vector_store``, ``display``, ``recall`` or ``config`` here.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from mitos.errors import DatabaseError

# The ranked over-fetch bound: how many points one ranked read fetches before it
# walks them to `limit + 1` live decisions. A budget constant, not a calibrated
# threshold (vision §6.3), so it stays outside Layer C's pin population.
#
# Set by phase 2b's ruled runs `20261016-0813-c772d96-burial`,
# `20261016-0819-c772d96-timing` and `20261016-0823-c772d96-dogfood`, by Vinga's
# ruling of 2026-10-16: the burial horizon the bound covers is W-emp + M (40-deep
# supersession chains on live's measured geometry, plus mixed depths). Its smallest
# cut-free bound is B_min = 338, so 350. Core p95 at 350 measured 24.0 ms (burial)
# and 26.6 ms (timing) against P16's 50 ms letter-retrieval budget. W-max (every
# predecessor at the single largest measured offset) is a sensitivity row, not a
# reason to raise it. The net: the first cut on a real corpus is the build order
# for the `active` hint (ADR
# `over-fetch-bound-covers-measured-burial-horizon-and-a-real-corpus-cut-builds-the-active-hint`).
#
# The light resolver stays unbuilt (ADR
# `ranked-walk-resolves-every-hit-whole-light-resolver-unbuilt-until-core-p95-passes-40ms`):
# at 350 the whole-node resolve is about half the core, and an (id, kind, state)
# resolve over every hit plus hydrating the window would save about 6.5 ms at p50.
# Trigger: the first phase whose measured ranked-read core p95 at this bound
# exceeds 40 ms on either horizon graph builds it, before adding its own cost.
#
# Must stay above `display.RANKED_LIMIT_CEILING` so `limit + 1` always fits
# (pinned by a row, not an import: this leaf imports no `display`).
RANKED_OVERFETCH_BOUND: int = 350

# The computed states a ranked read serves as live. Agreement-pinned to
# `store.LIVE_STATES` by a row; spelled here because this leaf cannot import the
# store.
LIVE_STATES: Tuple[str, ...] = ("active", "drifted")

# The end states a walk can reach (`RankedWalk.end`).
EDGE_FOUND = "edge_found"
EDGE_UNMEASURED = "edge_unmeasured"
SHORT_BOUND = "short_bound"
EXHAUSTED = "exhausted"

# The skip reasons, always all four keys in `RankedWalk.skipped`.
SKIP_KEYS: Tuple[str, ...] = ("orphan", "ambiguous", "non_decision", "duplicate")


@dataclass(frozen=True)
class PointResolution:
    """The partition of a batch of vector-index point ids by the node each names.

    Every distinct input string lands in exactly one of the three buckets. Keys
    are the point ids exactly as given, so a caller indexes by a hit's own
    ``point_id``. Built by ``GraphStore.resolve_point_ids``; it lives here so the
    walk can type-check the resolver's result without importing the store.

    Attributes:
        nodes: Point id → the one node carrying its prefix, in the reader-facing
            shape plus ``computed_state`` (``get_node_state``'s semantics) and no
            modifier stamps. Each entry is its own shallow copy.
        orphans: Point ids no node carries (or that are not a well-formed point
            id), in input order, deduplicated.
        ambiguous: Point id → the sorted ids of the two or more nodes sharing its
            128-bit prefix; resolved to neither.
    """

    nodes: Dict[str, Dict[str, Any]]
    orphans: List[str]
    ambiguous: Dict[str, List[str]]


class WalkContractError(TypeError):
    """A store or vector double does not honour the walk's contract.

    Raised at ``walk_ranked`` entry (a missing method, ``limit`` outside
    ``1..bound-1``) and where results arrive (a resolver result that is not a
    :class:`PointResolution`, a stamp map that is not a ``dict``, a hit without a
    string ``point_id`` or a numeric ``score``). Every surface re-raises it ahead of
    its degraded arm: swallowed, it would read as "semantic recall degraded".
    """


@dataclass(frozen=True)
class WalkHit:
    """One resolved decision hit.

    Attributes:
        node: The resolver's node (reader-facing shape plus ``computed_state``).
        score: The hit's similarity score.
        point_id: The point id exactly as the vector store returned it.
        rank: The hit's 1-based position in the fetch — not its position in the
            window, which surfaces read by index instead.
    """

    node: Dict[str, Any]
    score: float
    point_id: str
    rank: int


@dataclass(frozen=True)
class RankedWalk:
    """What one ranked read found. Runtime only, never persisted.

    Attributes:
        window: Live decisions in rank order, at most ``limit``.
        edge: The next live decision after the window (live rank ``limit + 1``),
            or None when the fetch held none.
        retired: Every retired decision passed before the walk stopped, in rank
            order.
        end: ``edge_found`` (``limit + 1`` live reached), ``edge_unmeasured``
            (the bound was reached holding exactly ``limit`` live),
            ``short_bound`` (the bound was reached holding fewer) or
            ``exhausted`` (the fetch returned fewer hits than the bound: the
            corpus ran out).
        returned: How many hits the fetch returned.
        bound: The ``limit`` the fetch asked the vector store for.
        limit: The caller's clamped window size.
        skipped: Hits skipped, by reason: ``orphan``, ``ambiguous``,
            ``non_decision``, ``duplicate`` — always all four keys.
        stamps: Node id → modifier stamps, for the window and ``retired[:limit]``.
            Always a dict: ``{}`` when not asked for or when the read faulted.
        stamps_unread: True only when the stamp read faulted.
        last_score: The score of the fetch's last hit (rank ``returned``), or None
            for an empty fetch. The gather reads it to tell a cut above its floor
            from one below it; the ranked surfaces ignore it.
    """

    window: List[WalkHit]
    edge: Optional[WalkHit]
    retired: List[WalkHit]
    end: str
    returned: int
    bound: int
    limit: int
    skipped: Dict[str, int]
    stamps: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    stamps_unread: bool = False
    last_score: Optional[float] = None


def _point_id_of(hit: Any) -> str:
    """Returns a hit's point id, or raises when the hit breaks the protocol.

    A hit is ``{point_id: str, score: number}`` at least. A missing score is checked
    here too: left to a bare ``KeyError`` inside a surface's broad catch, a double
    that dropped it would read as degraded recall rather than as a broken double.
    """
    pid = hit.get("point_id") if isinstance(hit, dict) else None
    if not isinstance(pid, str):
        raise WalkContractError(
            f"a vector hit must carry a string point_id; got {hit!r}"
        )
    score = hit.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise WalkContractError(f"a vector hit must carry a numeric score; got {hit!r}")
    return pid


def fill(hits: List[Dict[str, Any]], resolution: PointResolution, *,
         limit: int, bound: int) -> RankedWalk:
    """Walks ranked hits to ``limit`` live decisions plus the edge. Pure; no I/O.

    Each hit is classified by its resolution, in this order: an orphan or
    ambiguous point is skipped and counted (a point id absent from every bucket
    counts as an orphan, never guessed); a node whose ``kind`` is not
    ``decision`` is skipped as ``non_decision`` — kind before state, because a
    resolved open question reports ``active``; a decision already seen is skipped
    as ``duplicate``; a decision whose ``computed_state`` is in
    :data:`LIVE_STATES` is live; any other decision is retired. The walk stops at
    ``limit + 1`` live.

    ``limit >= bound`` is accepted here (2b's horizon harness drives it); the
    production entry, :func:`walk_ranked`, refuses it.

    Args:
        hits: The fetch's hits in rank order, each carrying ``point_id`` and
            ``score``.
        resolution: The resolution of (at least) every hit's point id.
        limit: The window size.
        bound: The ``limit`` the fetch asked for; ``len(hits) < bound`` means the
            corpus ran out.

    Returns:
        A :class:`RankedWalk` with ``stamps`` empty, ``stamps_unread`` False and
        ``last_score`` the last hit's score.

    Raises:
        WalkContractError: If a hit carries no string ``point_id`` or no numeric
            ``score``.
    """
    orphans = set(resolution.orphans)
    ambiguous = resolution.ambiguous
    nodes = resolution.nodes
    window: List[WalkHit] = []
    edge: Optional[WalkHit] = None
    retired: List[WalkHit] = []
    skipped: Dict[str, int] = {key: 0 for key in SKIP_KEYS}
    seen: set = set()
    for rank, hit in enumerate(hits, start=1):
        pid = _point_id_of(hit)
        if pid in ambiguous:
            skipped["ambiguous"] += 1
            continue
        node = nodes.get(pid)
        if pid in orphans or node is None:
            skipped["orphan"] += 1
            continue
        if node["kind"] != "decision":
            skipped["non_decision"] += 1
            continue
        if node["id"] in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(node["id"])
        walk_hit = WalkHit(node=node, score=hit["score"], point_id=pid, rank=rank)
        if node["computed_state"] in LIVE_STATES:
            if len(window) < limit:
                window.append(walk_hit)
            else:
                edge = walk_hit
                break
        else:
            retired.append(walk_hit)
    if edge is not None:
        end = EDGE_FOUND
    elif len(hits) < bound:
        end = EXHAUSTED
    elif len(window) == limit:
        end = EDGE_UNMEASURED
    else:
        end = SHORT_BOUND
    # The walk may stop before the last hit; that hit is still checked, so its
    # score is a number whichever way the walk ended.
    last_score = hits[-1]["score"] if hits and _point_id_of(hits[-1]) else None
    return RankedWalk(window=window, edge=edge, retired=retired, end=end,
                      returned=len(hits), bound=bound, limit=limit, skipped=skipped,
                      last_score=last_score)


def _require_callable(obj: Any, name: str, role: str) -> None:
    if not callable(getattr(obj, name, None)):
        raise WalkContractError(f"the walk's {role} must provide {name}(); "
                                f"{type(obj).__name__} does not")


def walk_ranked(vector: List[float], *, vector_store: Any, graph: Any, limit: int,
                bound: int = RANKED_OVERFETCH_BOUND,
                read_stamps: bool = True) -> RankedWalk:
    """Runs one ranked read: query, resolve, fill, stamp — three reads, no more.

    Every contract check runs at entry, before the query, so a double missing a
    method fails on an empty fetch too. The vector call is
    ``vector_store.query(vector, limit=bound, with_payload=False)``; the hits are
    resolved by ``graph.resolve_point_ids`` in one batch; the stamp ids are the
    window's plus the first ``limit`` retired hits' (what a bounded retired
    partition can show), read in one ``graph.get_modifiers_map`` call. No stamp
    read is made when both lists are empty, or when ``read_stamps`` is False.

    Args:
        vector: The query embedding.
        vector_store: Duck-typed; must provide ``query``.
        graph: The duck-typed store; must provide ``resolve_point_ids``, plus
            ``get_modifiers_map`` when ``read_stamps`` is set.
        limit: The caller's clamped window size, ``1 <= limit < bound``.
        bound: The over-fetch bound.
        read_stamps: False for a reader that must do no stamp work: the conflict
            gather, which reads its live window's stamps itself and must never stamp
            the retired hits it passed.

    Returns:
        A :class:`RankedWalk`.

    Raises:
        WalkContractError: On a contract fault (see the class).
        DatabaseError: From the resolver; the caller degrades it as a store fault.
        Exception: Whatever ``vector_store.query`` raises (``CollectionMissingError``
            included), propagated unchanged.
    """
    _require_callable(vector_store, "query", "vector store")
    _require_callable(graph, "resolve_point_ids", "graph store")
    if read_stamps:
        _require_callable(graph, "get_modifiers_map", "graph store")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit < bound:
        raise WalkContractError(f"limit must satisfy 1 <= limit < bound ({bound}); "
                                f"got {limit!r}")

    hits = vector_store.query(vector, limit=bound, with_payload=False)
    point_ids = [_point_id_of(hit) for hit in hits]
    resolution = graph.resolve_point_ids(point_ids)
    if not isinstance(resolution, PointResolution):
        raise WalkContractError(
            "resolve_point_ids must return a PointResolution; got "
            f"{type(resolution).__name__}"
        )
    walk = fill(hits, resolution, limit=limit, bound=bound)
    if not read_stamps:
        return walk
    stamp_ids = stamp_ids_of(walk)
    if not stamp_ids:
        return walk
    try:
        stamps = graph.get_modifiers_map(stamp_ids)
    except DatabaseError:
        return replace(walk, stamps={}, stamps_unread=True)
    if not isinstance(stamps, dict):
        raise WalkContractError(
            f"get_modifiers_map must return a dict; got {type(stamps).__name__}"
        )
    return replace(walk, stamps=stamps)


def stamp_ids_of(walk: RankedWalk) -> List[str]:
    """The node ids :func:`walk_ranked` stamps: the window, then ``retired[:limit]``."""
    return ([h.node["id"] for h in walk.window]
            + [h.node["id"] for h in walk.retired[:walk.limit]])
