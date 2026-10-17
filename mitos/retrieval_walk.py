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

The walk makes four reads and no more (ADR
``readers-fill-their-window-with-live-hits-from-one-over-fetched-call``): **one**
``VectorStore.query`` at :data:`RANKED_OVERFETCH_BOUND` with ``with_payload=False``,
**one** batched ``resolve_point_ids`` over every hit, **one** batched
``get_modifiers_map`` over the window and the first ``limit`` retired hits, and
**one** batched ``get_live_ends`` over the retired partition's shown members. In
between, :func:`fill` walks the hits in memory until it holds ``limit`` live
decisions plus one more for the edge, so retired and open-question points never
take a caller's slots. It never re-fetches.

The fourth read is conditional, because it is the one whose cost grows with chain
depth. It is made only when the partition shows a member (``passed[:limit]``): a
full window whose retired hits all rank below the last live one shows none, and a
read that passed no retired decision has none to show. Membership is a rank
comparison :func:`fill` makes, so the ids are known before the read.

A stamp fault costs the stamps alone: the walk returns ``stamps={}`` with
``stamps_unread=True`` and the window stands (lock A, failure state). The live-end
read shares that fault domain, because a live end is a successor fact read off the
same kill edges as ``superseded_by`` / ``corrected_by``: a ``DatabaseError`` from
either read returns ``stamps={}``, ``live_ends={}`` and ``stamps_unread=True``. A resolver
fault (``DatabaseError``) propagates, and each surface degrades it as a store
fault. A wrong store double raises :class:`WalkContractError`, which every
surface re-raises ahead of its degraded arm, so a double that absorbs every call
(a ``MagicMock``) fails loud instead of passing an absence-asserting row over an
empty window.

The rejection leg
-----------------
A ranked surface may hand the walk a :class:`RejectionLeg`. The walk then makes
**one** more call and no more graph reads: one ``RejectionIndex.search`` with the
same query vector, at the leg's own fetch bound and under its own time budget
(ADR ``rejection-search-on-ranked-reads-runs-under-its-own-time-budget``). The
owners of the hits at or above the leg's floor join the one resolve batch (an
owner's point-id form is its decision point's id, so a window hit's owner resolves
once), the one stamp read and the one live-end read. :func:`fill_rejections` then
checks every hit against its owner's **current** enumeration, injected as
``items_of`` (``rejections.point_items``), so an item the field no longer states is
never served (ADR
``rejection-index-keeps-its-own-backlog-and-reads-serve-no-item-its-owner-no-longer-states``).
The leg's own failures never escape the walk: an absent collection, an overrun
budget and an index error each become a state on ``RankedWalk.rejections``, and the
decision answer is the one a read without the leg returns. The conflict gather
never passes a leg.

The light resolver
------------------
The ranked reads pass ``light=True``: the resolve over every hit reads only each
node's id, kind and state (``resolve_point_states``), and one more batched read
(``resolve_point_ids``) hydrates only what the answer shows — the window, the edge,
the partition's shown members and the leg's owners. 5a's rejection leg took the
ranked read's core p95 past the 40 ms trigger at the horizon, which builds it (ADR
``ranked-walk-resolves-every-hit-whole-light-resolver-unbuilt-until-core-p95-passes-40ms``).
The gather keeps the one whole resolve: it consumes every retired node it passed.

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
Stdlib (``dataclasses``, ``re``, ``typing``, ``uuid``) plus ``mitos.errors``, which
is already a fence member and imports only ``typing``: the leaf's runtime closure is
exactly ``{retrieval_walk.py, errors.py}``. It reads three methods off a store it is
handed, and one off a rejection index, and writes nothing, so admitting it costs the
fence no write path and no ``store``/``vector_store``/``telemetry`` reach.
``conflict.py``'s import of it grew the exact-set pin by this module alone. Keep it
that way: never import ``store``, ``protocols``, ``vector_store``, ``display``,
``recall``, ``rejections`` or ``config`` here. The leg's enumeration, floor and
bound arrive as arguments.
"""

import re
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from mitos.errors import CollectionMissingError, DatabaseError, VectorStoreError

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

# The rejection leg's fetch bound: how many items one rejection search asks for
# before `fill_rejections` walks them to the floor or the section bound. A budget
# constant, not a calibrated threshold (vision §6.3), so it stays outside Layer C's
# pin population.
#
# MEASUREMENT PENDING (§7.1 of phase 5a sets it and cites its runs here).
#
# Must stay above `display.RANKED_LIMIT_CEILING`: the field-scoped lookup (5c) walks
# to a caller's `limit` with this one bound (pinned by a row, not an import).
REJECTION_OVERFETCH_BOUND: int = 100

# The rejection search's time budget, in seconds: the search's socket timeout
# (`RejectionIndex.search`'s contract), so the wall it bounds is about twice this
# (connect, then read). A budget constant, not a calibrated threshold. An overrun
# costs the section alone, stated as `not_searched`; the decision answer is whole
# (ADRs `rejection-search-on-ranked-reads-runs-under-its-own-time-budget` and
# `rejection-search-time-budget-sits-inside-the-50ms-letter-retrieval-budget`).
#
# MEASUREMENT PENDING (§7.1 of phase 5a sets it and cites its runs here).
REJECTION_SEARCH_BUDGET_SECONDS: float = 0.008

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

# The rejection leg's states (`RejectionWalk.state`).
REJECTIONS_SEARCHED = "searched"
REJECTIONS_ABSENT = "absent"
REJECTIONS_OVER_BUDGET = "over_budget"
REJECTIONS_ERROR = "error"

# The rejection hits verification drops, always all five keys in
# `RejectionWalk.dropped`.
REJECTION_DROP_KEYS: Tuple[str, ...] = (
    "stale", "ownerless", "orphan", "non_decision", "ambiguous",
)

# A node id: 64 lowercase hex characters. The same predicate as
# `rejections._OWNER_ID_RE` (agreement-pinned by a row); spelled here because this
# leaf cannot import `rejections`. An owner id outside it is never UUID-formatted.
_OWNER_ID_RE = re.compile(r"[0-9a-f]{64}")


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
    :class:`PointResolution`, a stamp or live-end map that is not a ``dict``, a hit without a
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
class RejectionLeg:
    """What a ranked surface hands the walk to search the rejection index too.

    Attributes:
        index: The duck-typed rejection index; must provide
            ``search(vector, limit, *, budget_seconds)``.
        items_of: ``rejections.point_items``, injected: ``(owner_id, field)`` →
            ``{point_id: item}``, each item carrying ``ordinal`` and ``text``.
        floor: The rejection-match floor (``recall.REJECTION_MATCH_FLOOR``); a hit
            below it ends the walk.
        bound: The section bound in items (``recall.REJECTION_SECTION_BOUND``).
            No default: the value lives in ``recall``, which this leaf cannot
            import, so every caller passes it.
        fetch: How many items the search asks for.
        budget_seconds: The search's time budget.
    """

    index: Any
    items_of: Callable[[str, str], Mapping[str, Any]]
    floor: float
    bound: int
    fetch: int = REJECTION_OVERFETCH_BOUND
    budget_seconds: float = REJECTION_SEARCH_BUDGET_SECONDS


@dataclass(frozen=True)
class RejectionMatch:
    """One verified rejection match. Runtime only.

    Attributes:
        owner: The resolved owner node (reader-facing shape plus
            ``computed_state``).
        point_id: The best-scoring part's point id.
        score: The best-scoring part's score.
        ordinal: The item's ordinal in the owner's **current** enumeration.
        item: The whole item's text, ``item.text.rstrip()`` (never a part).
        live: Whether the owner's computed state is live (:data:`LIVE_STATES`).
    """

    owner: Dict[str, Any]
    point_id: str
    score: float
    ordinal: int
    item: str
    live: bool


@dataclass(frozen=True)
class RejectionWalk:
    """What the rejection leg found. Runtime only, ``RankedWalk.rejections``.

    Attributes:
        state: ``searched``, ``absent`` (the collection does not exist),
            ``over_budget`` (the search ran past its time budget) or ``error``
            (the index could not be read).
        matches: Verified matches in score order, at most ``bound`` distinct items.
        short: The bound did not fill, the fetch came back full and its last hit
            was still at or above the floor: an unfetched hit could have qualified.
        dropped: Hits at or above the floor that verification dropped, by reason —
            always all of :data:`REJECTION_DROP_KEYS`.
        item_counts: Owner id → how many distinct items its current field can
            serve (the distinct ordinals of ``items_of``), for every matched owner.
        returned: How many hits the search returned.
        fetch: The ``limit`` the search asked for.
    """

    state: str
    matches: List[RejectionMatch]
    short: bool
    dropped: Dict[str, int]
    item_counts: Dict[str, int]
    returned: int
    fetch: int


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
        stamps_unread: True only when the stamp read or the live-end read faulted.
        last_score: The score of the fetch's last hit (rank ``returned``), or None
            for an empty fetch. The gather reads it to tell a cut above its floor
            from one below it; the ranked surfaces ignore it.
        passed: The retired partition's members, in rank order and unbounded. On a
            full window (``len(window) == limit``) they are the retired hits ranked
            above the last live one; on a short window, empty included, every
            retired hit in the fetch (a short walk never stops early). Always a
            prefix of ``retired``, so ``passed[:limit]`` is already stamped.
        live_ends: Node id → the live-end slugs of its kill chain, for
            ``passed[:limit]`` and the rejection section's retired owners, and only
            for the ids whose immediate successor is itself retired
            (``GraphStore.get_live_ends``' result rule). ``{}`` when not read or
            when the read faulted.
        rejections: What the rejection leg found, or None when no leg ran. Never
            set by :func:`fill`. When set, ``stamps`` covers the section's owners
            too.
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
    passed: List[WalkHit] = field(default_factory=list)
    live_ends: Dict[str, List[str]] = field(default_factory=dict)
    rejections: Optional[RejectionWalk] = None


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

    The retired partition's membership is decided here too, because it is a rank
    comparison and the live-end read needs the ids: on a full window the retired
    hits ranked above the last live one, otherwise every retired hit (``passed``).

    Returns:
        A :class:`RankedWalk` with ``stamps`` and ``live_ends`` empty,
        ``stamps_unread`` False, ``last_score`` the last hit's score and ``passed``
        the partition's members.

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
    last_score = None
    if hits:
        _point_id_of(hits[-1])
        last_score = hits[-1]["score"]
    # The retired partition's members. A full window shows only what outranked its
    # last live decision; a retired hit between the last live and the edge was not
    # passed over while filling, so it is not a member.
    if len(window) == limit:
        cutoff = window[-1].rank
        passed = [h for h in retired if h.rank < cutoff]
    else:
        passed = list(retired)
    return RankedWalk(window=window, edge=edge, retired=retired, end=end,
                      returned=len(hits), bound=bound, limit=limit, skipped=skipped,
                      last_score=last_score, passed=passed)


def _owner_point_id(owner_id: Optional[str]) -> Optional[str]:
    """Returns an owner's point-id form, or None for an id that is not a node id.

    ``vector_store.hash_to_uuid``'s output, spelled with ``uuid`` because this leaf
    cannot import the vector store: the first 128 bits of the id as a lowercase
    8-4-4-4-12 UUID, byte-identical to the owner's decision point id, so the
    resolver dedupes it against a window hit by raw string.
    """
    if not isinstance(owner_id, str) or _OWNER_ID_RE.fullmatch(owner_id) is None:
        return None
    return str(uuid.UUID(hex=owner_id[:32]))


def _rejection_hits_of(answer: Any) -> List[Tuple[str, Optional[str], float]]:
    """Returns a searched answer's hits as ``(point_id, owner_id, score)``.

    Attribute access, because a rejection hit is a ``RejectionHit`` named tuple, not
    a decision hit's dict.

    Raises:
        WalkContractError: If the answer carries no ``hits`` list, or a hit carries
            no string ``point_id``, no numeric ``score`` or an ``owner_id`` that is
            neither a string nor None.
    """
    hits = getattr(answer, "hits", None)
    if not isinstance(hits, list):
        raise WalkContractError(
            f"a rejection search answer must carry a hits list; got {answer!r}"
        )
    out = []
    for hit in hits:
        pid = getattr(hit, "point_id", None)
        score = getattr(hit, "score", None)
        owner = getattr(hit, "owner_id", None)
        if not isinstance(pid, str):
            raise WalkContractError(
                f"a rejection hit must carry a string point_id; got {hit!r}"
            )
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise WalkContractError(
                f"a rejection hit must carry a numeric score; got {hit!r}"
            )
        if owner is not None and not isinstance(owner, str):
            raise WalkContractError(
                f"a rejection hit's owner_id must be a string or None; got {hit!r}"
            )
        out.append((pid, owner, score))
    return out


def _unsearched(state: str, fetch: int) -> RejectionWalk:
    """The leg's result when no hit came back to walk."""
    return RejectionWalk(state=state, matches=[], short=False,
                         dropped={key: 0 for key in REJECTION_DROP_KEYS},
                         item_counts={}, returned=0, fetch=fetch)


def fill_rejections(search: Any, resolution: PointResolution, *,
                    items_of: Callable[[str, str], Mapping[str, Any]],
                    floor: float, bound: int, fetch: int) -> RejectionWalk:
    """Walks a rejection search's hits to verified matches. Pure; no I/O.

    The hits arrive in score order. The walk stops at the first hit below
    ``floor`` (nothing past it can qualify) or once ``bound`` distinct items are
    held. Each hit above the floor is checked in this order: no payload owner is
    ``ownerless``; an owner that is not a node id, or whose point-id form resolved
    to no node or to a node with a different id, is an ``orphan``; a prefix two
    nodes share is ``ambiguous``; an owner that is not a decision is
    ``non_decision``; a point id the owner's current field no longer produces
    (``items_of(owner_id, owner["rejected_paths"])``) is ``stale``. A hit whose
    ``(owner, ordinal)`` is already held is a lower-scoring part of a shown item and
    collapses into it, counted as nothing.

    Args:
        search: The ``RejectionSearch`` the index answered. ``searched=False``
            gives the ``over_budget`` state.
        resolution: The walk's one resolution, which holds the owners' point-id
            forms.
        items_of: ``(owner_id, field)`` → ``{point_id: item}``.
        floor: The rejection-match floor.
        bound: The section bound in items.
        fetch: The ``limit`` the search asked for.

    Returns:
        A :class:`RejectionWalk`. ``short`` is True only when the bound did not
        fill, the fetch came back full and its last hit was still at or above the
        floor; a fetch that ran out or crossed the floor is complete, however many
        hits were dropped.

    Raises:
        WalkContractError: On an answer or a hit that breaks the search contract.
    """
    searched = getattr(search, "searched", None)
    if not isinstance(searched, bool):
        raise WalkContractError(
            f"a rejection search answer must carry a bool searched; got {search!r}"
        )
    hits = _rejection_hits_of(search)
    if not searched:
        return _unsearched(REJECTIONS_OVER_BUDGET, fetch)
    dropped = {key: 0 for key in REJECTION_DROP_KEYS}
    matches: List[RejectionMatch] = []
    item_counts: Dict[str, int] = {}
    held: set = set()
    enumerations: Dict[str, Mapping[str, Any]] = {}
    for pid, owner_id, score in hits:
        if score < floor or len(matches) >= bound:
            break
        if owner_id is None:
            dropped["ownerless"] += 1
            continue
        owner_pid = _owner_point_id(owner_id)
        if owner_pid is not None and owner_pid in resolution.ambiguous:
            dropped["ambiguous"] += 1
            continue
        node = resolution.nodes.get(owner_pid) if owner_pid is not None else None
        # The resolver matches a 128-bit prefix; only the payload's whole id is the
        # owner, never a node that merely shares its prefix.
        if node is None or node["id"] != owner_id:
            dropped["orphan"] += 1
            continue
        if node["kind"] != "decision":
            dropped["non_decision"] += 1
            continue
        items = enumerations.get(owner_id)
        if items is None:
            items = items_of(owner_id, node["rejected_paths"])
            enumerations[owner_id] = items
        item = items.get(pid)
        if item is None:
            dropped["stale"] += 1
            continue
        if (owner_id, item.ordinal) in held:
            continue
        held.add((owner_id, item.ordinal))
        item_counts[owner_id] = len({i.ordinal for i in items.values()})
        matches.append(RejectionMatch(
            owner=node, point_id=pid, score=score, ordinal=item.ordinal,
            item=item.text.rstrip(), live=node["computed_state"] in LIVE_STATES,
        ))
    short = (len(matches) < bound and len(hits) >= fetch
             and bool(hits) and hits[-1][2] >= floor)
    return RejectionWalk(state=REJECTIONS_SEARCHED, matches=matches, short=short,
                         dropped=dropped, item_counts=item_counts,
                         returned=len(hits), fetch=fetch)


def _require_callable(obj: Any, name: str, role: str) -> None:
    if not callable(getattr(obj, name, None)):
        raise WalkContractError(f"the walk's {role} must provide {name}(); "
                                f"{type(obj).__name__} does not")


def walk_ranked(vector: List[float], *, vector_store: Any, graph: Any, limit: int,
                bound: int = RANKED_OVERFETCH_BOUND,
                read_stamps: bool = True,
                rejections: Optional[RejectionLeg] = None,
                light: bool = False) -> RankedWalk:
    """Runs one ranked read: query, resolve, fill, stamp, live ends — four reads, no more.

    Every contract check runs at entry, before the query, so a double missing a
    method fails on an empty fetch too. The vector call is
    ``vector_store.query(vector, limit=bound, with_payload=False)``; the hits are
    resolved by ``graph.resolve_point_ids`` in one batch; the stamp ids are the
    window's plus the first ``limit`` retired hits' (what a bounded retired
    partition can show), read in one ``graph.get_modifiers_map`` call. No stamp
    read is made when both lists are empty, or when ``read_stamps`` is False. Then
    one ``graph.get_live_ends`` call over the partition's shown members
    (``passed[:limit]``), skipped when there are none. The two reads share one fault
    domain: a ``DatabaseError`` from either costs the stamps and the live ends
    together, flagged ``stamps_unread``.

    With a rejection leg, one ``search`` follows the query (same vector, the leg's
    fetch and budget). The owners of its hits at or above the floor join the one
    resolve batch, the section's owners join the one stamp read, and its retired
    owners join the one live-end read: one more network call, no more graph reads.
    The leg's failures become its state, never a raise: ``CollectionMissingError`` →
    ``absent``, an overrun budget → ``over_budget``, any other ``VectorStoreError``
    (an unreachable index included) → ``error``.

    Args:
        vector: The query embedding.
        vector_store: Duck-typed; must provide ``query``.
        graph: The duck-typed store; must provide ``resolve_point_ids``, plus
            ``get_modifiers_map`` and ``get_live_ends`` when ``read_stamps`` is set.
        limit: The caller's clamped window size, ``1 <= limit < bound``.
        bound: The over-fetch bound.
        read_stamps: False for a reader that must do no stamp work: the conflict
            gather, which reads its live window's stamps itself and must never stamp
            the retired hits it passed. It makes neither the stamp read nor the
            live-end read.
        rejections: The rejection leg, or None (the gather, and every read that
            composes no section). Without it the walk is the four reads above.
        light: Walk every hit on ``graph.resolve_point_states`` (id, kind and
            state), then hydrate only what the answer shows — the window, the edge,
            ``passed[:limit]`` and the leg's owners — in one ``resolve_point_ids``
            (the ranked reads; ADR
            ``ranked-walk-resolves-every-hit-whole-light-resolver-unbuilt-until-core-p95-passes-40ms``).
            False keeps the one whole resolve: the gather, which consumes every
            retired node it passed. On a light walk ``retired`` beyond
            ``passed[:limit]`` holds light nodes.

    Returns:
        A :class:`RankedWalk`; ``rejections`` is set exactly when a leg was passed.

    Raises:
        WalkContractError: On a contract fault (see the class), the rejection
            index's and its answer's included.
        DatabaseError: From the resolver; the caller degrades it as a store fault.
        Exception: Whatever ``vector_store.query`` raises (``CollectionMissingError``
            included), propagated unchanged.
    """
    _require_callable(vector_store, "query", "vector store")
    _require_callable(graph, "resolve_point_ids", "graph store")
    if light:
        _require_callable(graph, "resolve_point_states", "graph store")
    if read_stamps:
        _require_callable(graph, "get_modifiers_map", "graph store")
        _require_callable(graph, "get_live_ends", "graph store")
    if rejections is not None:
        _require_callable(rejections.index, "search", "rejection index")
        if not callable(rejections.items_of):
            raise WalkContractError("the rejection leg's items_of must be callable")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit < bound:
        raise WalkContractError(f"limit must satisfy 1 <= limit < bound ({bound}); "
                                f"got {limit!r}")

    hits = vector_store.query(vector, limit=bound, with_payload=False)
    point_ids = [_point_id_of(hit) for hit in hits]
    search = None
    leg_state: Optional[str] = None
    batch = point_ids
    owners: List[str] = []
    if rejections is not None:
        # Caught by class: a WalkContractError is a TypeError, never swallowed here.
        try:
            search = rejections.index.search(vector, rejections.fetch,
                                             budget_seconds=rejections.budget_seconds)
        except CollectionMissingError:
            leg_state = REJECTIONS_ABSENT
        except VectorStoreError:
            leg_state = REJECTIONS_ERROR
        if leg_state is None and not isinstance(getattr(search, "searched", None), bool):
            raise WalkContractError(
                f"a rejection search answer must carry a bool searched; got {search!r}"
            )
        if leg_state is None and search.searched:
            # Only the owners `fill_rejections` can reach: those of hits at or above
            # the floor, each once, after the decision hits (raw-string dedupe).
            seen = set(point_ids)
            for _pid, owner_id, score in _rejection_hits_of(search):
                if score < rejections.floor:
                    break
                owner_pid = _owner_point_id(owner_id)
                if owner_pid is not None and owner_pid not in owners:
                    owners.append(owner_pid)
            batch = point_ids + [pid for pid in owners if pid not in seen]
    resolver = "resolve_point_states" if light else "resolve_point_ids"
    resolution = _resolution(getattr(graph, resolver)(batch), resolver)
    walk = fill(hits, resolution, limit=limit, bound=bound)
    if light:
        walk, resolution = _hydrate_shown(walk, owners, graph)
    if rejections is not None:
        # Attached before every return below, so no exit drops the section.
        section = (_unsearched(leg_state, rejections.fetch) if leg_state is not None
                   else fill_rejections(search, resolution, items_of=rejections.items_of,
                                        floor=rejections.floor, bound=rejections.bound,
                                        fetch=rejections.fetch))
        walk = replace(walk, rejections=section)
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
    shown_ids = [h.node["id"] for h in walk.passed[:walk.limit]]
    if walk.rejections is not None:
        for match in walk.rejections.matches:
            if not match.live and match.owner["id"] not in shown_ids:
                shown_ids.append(match.owner["id"])
    if not shown_ids:
        return replace(walk, stamps=stamps)
    try:
        live_ends = graph.get_live_ends(shown_ids)
    except DatabaseError:
        # One fault domain with the stamps: the healthy stamps are discarded too,
        # so a handle never carries `superseded_by` without its live end.
        return replace(walk, stamps={}, live_ends={}, stamps_unread=True)
    if not isinstance(live_ends, dict):
        raise WalkContractError(
            f"get_live_ends must return a dict; got {type(live_ends).__name__}"
        )
    return replace(walk, stamps=stamps, live_ends=live_ends)


def _resolution(result: Any, resolver: str) -> PointResolution:
    if not isinstance(result, PointResolution):
        raise WalkContractError(
            f"{resolver} must return a PointResolution; got {type(result).__name__}"
        )
    return result


def _hydrate_shown(walk: RankedWalk, owners: List[str],
                   graph: Any) -> Tuple[RankedWalk, PointResolution]:
    """Swaps a light walk's shown hits for whole nodes, in one ``resolve_point_ids``.

    Shown means what a ranked answer composes from: the window, the edge,
    ``passed[:limit]`` (the partition's handles) and the leg's owners. A shown hit
    whose node the second read no longer holds (a graph rebuilt between the two
    reads) raises ``DatabaseError``, which every surface degrades as a store fault.

    Returns:
        The walk with whole nodes on its shown hits, and the hydrating resolution
        (the leg's owners are verified against it).
    """
    shown = ([h.point_id for h in walk.window]
             + ([walk.edge.point_id] if walk.edge is not None else [])
             + [h.point_id for h in walk.passed[:walk.limit]])
    hydrated = _resolution(graph.resolve_point_ids(list(dict.fromkeys(shown + owners))),
                           "resolve_point_ids")

    def whole(hit: WalkHit) -> WalkHit:
        node = hydrated.nodes.get(hit.point_id)
        if node is None:
            raise DatabaseError(f"point {hit.point_id} left the graph between two reads")
        return replace(hit, node=node)

    shown_passed = [whole(h) for h in walk.passed[:walk.limit]]
    by_point = {h.point_id: h for h in shown_passed}
    return replace(
        walk,
        window=[whole(h) for h in walk.window],
        edge=whole(walk.edge) if walk.edge is not None else None,
        retired=[by_point.get(h.point_id, h) for h in walk.retired],
        passed=shown_passed + walk.passed[walk.limit:],
    ), hydrated


def stamp_ids_of(walk: RankedWalk) -> List[str]:
    """The node ids :func:`walk_ranked` stamps: the window, then ``retired[:limit]``,
    then the rejection section's owners not already among them (when a leg ran)."""
    ids = ([h.node["id"] for h in walk.window]
           + [h.node["id"] for h in walk.retired[:walk.limit]])
    if walk.rejections is not None:
        for match in walk.rejections.matches:
            if match.owner["id"] not in ids:
                ids.append(match.owner["id"])
    return ids
