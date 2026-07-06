"""The corpus-audit engine's core — strong-edge pair screen (2a) + sweep assembly (2b).

This module is the home of ``mitos check``: the corpus-wide conflict audit that sweeps
every decision and judges the genuinely *undeclared* tensions between similar pairs. The
sync-time sensor (``conflict.py``) judges one proposal against its neighbours at write
time; the corpus audit has no privileged direction — it must read a relationship as a
fact about a *pair*, not about whichever node the sweep happened to iterate first.

Phase 2a landed the first two composition pieces: the **either-direction strong-edge
screen** that removes, id-natively, every pair the author already settled with a declared
strong relationship — so the LLM judge is never asked to second-guess a carve-out the
author was most careful to make. A relationship is stored once, as authored
(``source → target``), but "A narrows B" and "B is narrowed by A" are the same settled
decision, so the index records **both** endpoints and the screen is direction-free.

Phase 2b assembles the deterministic **corpus sweep** around that screen — the plan half
of the run engine (CHK-D2 steps 1–3): a run-start snapshot (:func:`snapshot_corpus` —
one ``get_active_decisions`` + one ``get_edges``, the index built once), a lazy per-node
sweep composing the shipped pipeline stages (:func:`iter_sweep`: gather → strong-edge
screen → ``screen_candidates``), corpus-wide exactly-once pair dedup with replayable
orientation (:func:`dedup_oriented_pairs` — the oriented proposal is the
lexicographically smaller content hash, independent of which side's sweep discovered the
pair), and judgment-sized batch grouping (:func:`group_judgment_batches`). Everything
here is deterministic and LLM-free: the run engine (2c) consumes these structures to
*plan* a run — and disclose its cost — before spending a single judgment token, and its
aggregate breaker is simply "stop consuming the generator".

The screen is **direct-edge-only, by design** (CHK-D2): it reads edges as stored, nothing
transitive. If ``B narrows A`` and then ``B' supersedes B``, the pair ``{A, B'}`` is
genuinely unexamined — the author never declared how the *new* ``B'`` relates to ``A`` —
so it must reach judgment. A lineage walk would bury that real, re-opened question under a
lapsed relationship; knowing when a relationship has lapsed is as load-bearing as knowing
when it holds.

**Tier-1 leaf, same discipline as ``conflict.py``.** This module must never import a
higher-tier ``mitos`` module or a heavy dependency (``anthropic``, the Qdrant/genai
clients) at module scope — the judge arrives injected in later phases, and the CHK-*
constants land here as later phases build the sweep/run engine. ``check`` sits *above*
``conflict`` in the import DAG (``check → conflict``, never the reverse); its only
module-scope import is ``conflict``, itself a dep-free leaf. A dep-free import guard pins
this (``test_importing_check_drags_no_heavy_dependency``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from mitos.conflict import (
    CONFLICT_SIMILARITY_FLOOR,
    CONFLICT_TOP_K,
    Candidate,
    Unavailable,
    _STRONG_RELATIONSHIP_FIELDS,
    gather_candidates,
    screen_candidates,
)

if TYPE_CHECKING:
    # Runtime-injected, duck-typed collaborators — annotated only for the type checker.
    # Importing ``mitos.protocols`` at runtime pulls ``parser`` + ``store`` (not a
    # leaf-cheap import), so these stay behind the guard (the conflict.py idiom).
    from mitos.protocols import EmbeddingProvider, GraphStoreProtocol, VectorStore


class StrongEdgeIndex:
    """An orientation-blind strong-edge adjacency: ``node_id → its strong partners``.

    Built once per run by :func:`build_strong_edge_index` from ``store.get_edges()`` and
    read per swept node via :meth:`partners` — the "build once, look up per node" seam
    Phase 2b threads into its run-start snapshot. Two nodes are partners iff a strong edge
    (any of the five :data:`_STRONG_RELATIONSHIP_FIELDS` types) joins them in *either*
    direction. The stored partner sets are frozen — the index is an immutable snapshot of
    the graph's strong-edge structure at build time.
    """

    def __init__(self, partners: Dict[str, "frozenset[str]"]) -> None:
        """Wraps a pre-built ``{node_id: frozenset(partner_ids)}`` adjacency.

        Args:
            partners: The fully-built adjacency; only nodes with at least one strong
                partner appear. Callers use :func:`build_strong_edge_index`, not this
                constructor directly.
        """
        self._partners = partners

    def partners(self, node_id: str) -> "frozenset[str]":
        """The strong-edge partners of ``node_id``, or ``∅`` for an unknown node.

        The empty-frozenset default is deliberate: a swept node that shares no strong
        edge (the healthy common case) is not an error, it simply has no partners. This
        keeps the lookup total so a consumer can never ``KeyError`` on a fresh node.

        Args:
            node_id: The content-hash id of the node to look up.

        Returns:
            The frozenset of partner node ids (empty if the node is absent).
        """
        return self._partners.get(node_id, frozenset())

    def __len__(self) -> int:
        """The number of nodes carrying at least one strong-edge partner."""
        return len(self._partners)


def build_strong_edge_index(edges: "List[Dict[str, str]]") -> StrongEdgeIndex:
    """One pass over the edge list → an orientation-blind strong-edge adjacency.

    For each edge whose ``edge_type`` is one of the five strong relationship types
    (:data:`_STRONG_RELATIONSHIP_FIELDS` — bound to the single source in ``conflict.py``,
    never a hand-listed lookalike), record **both** endpoints as partners of each other.
    This is what makes the screen direction-free: an edge authored ``source=X, target=Y``
    puts ``Y ∈ partners(X)`` *and* ``X ∈ partners(Y)``.

    Direct edges only — the index is built from edges verbatim, with **no** transitive or
    lineage closure (CHK-D2). A superseded declarer re-opens its pair by construction:
    ``{A, B'}`` where ``B narrows A`` and ``B' supersedes B`` has no direct strong edge, so
    it survives the screen and reaches judgment. A self-edge (``source_id == target_id``,
    which the write path rejects but a rebuilt graph could carry out-of-band) is skipped —
    a node is never its own strong partner; self-drop is ``screen_candidates``'s
    ``own_slug`` job downstream, not this screen's (mirrors ``get_contradictions``'s
    ``!= ?`` guard).

    Pure and store-free: takes a pre-fetched edge list so the logic is testable without a
    store. Phase 2b makes the single ``store.get_edges()`` call at run-start and passes the
    result here (one build, reused across the whole sweep). A ``get_edges()`` fault is the
    caller's to surface — this function does not mask it (there is no fallback empty index,
    which would silently screen nothing and mislabel every pair as fresh).

    Args:
        edges: The edge rows from ``store.get_edges()``; each a dict exposing
            ``source_id``, ``target_id``, and ``edge_type`` (extra keys ignored).

    Returns:
        The built :class:`StrongEdgeIndex`.
    """
    adjacency: Dict[str, Set[str]] = {}
    for edge in edges:
        if edge["edge_type"] not in _STRONG_RELATIONSHIP_FIELDS:
            continue
        source = edge["source_id"]
        target = edge["target_id"]
        if source == target:
            continue  # a node is never its own partner (own_slug covers self downstream)
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    return StrongEdgeIndex(
        {node_id: frozenset(partner_ids) for node_id, partner_ids in adjacency.items()}
    )


def screen_strong_edge_pairs(
    proposal_id: str,
    candidates: "List[Candidate]",
    index: StrongEdgeIndex,
) -> "List[Candidate]":
    """Drop every candidate that shares a strong edge with the swept node (either way).

    Given a swept node (``proposal_id``, a content hash) and its gathered candidates,
    remove each candidate whose ``node['id']`` is a strong-edge partner of ``proposal_id``
    in the index — the author already reasoned about that pair, so the audit stays silent
    and spends no judgment on it. Compares content-hash ids directly (never slugs — the
    slug is a mutable historical citation, M2).

    Order-preserving, pure, and storeless — no I/O, no state re-verify. Placed *upstream*
    of ``screen_candidates`` by the sweep (Phase 2b) so a declared strong neighbour is
    dropped before the similarity floor and the ``top_k`` truncation, and can never consume
    a slot that would shadow a genuine undeclared conflict out of the window (the CONF-D7
    drop-before-floor-before-truncate order).

    Args:
        proposal_id: The swept node's content-hash id.
        candidates: The gathered neighbours to screen (order preserved in the result).
        index: The strong-edge index built once at run-start.

    Returns:
        The candidates that share no strong edge with the swept node — the survivors that
        reach judgment (possibly empty; possibly the whole input unchanged).
    """
    partners = index.partners(proposal_id)
    return [candidate for candidate in candidates if candidate.node["id"] not in partners]


# --------------------------------------------------------------------------- #
# Phase 2b — the corpus sweep assembly (snapshot → lazy sweep → dedup → groups)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CorpusSnapshot:
    """The run-start snapshot: the sweep set + the strong-edge index, fixed together.

    One :func:`snapshot_corpus` call pins both in one place, so the whole run reads a
    single consistent view of the graph — mid-run commits are the next run's problem.
    ``len(snapshot.nodes)`` is the denominator for 2c's swept-vs-skipped accounting.

    Attributes:
        nodes: The sweep set — ``get_active_decisions(scope)``'s hydrated,
            modifier-stamped decision dicts (active ∪ drifted), in snapshot order.
            That order is a DB accident (the SELECT has no ORDER BY); nothing
            downstream may depend on it — output determinism comes from the
            hash-ordered post-passes, never from iteration order.
        edge_index: The either-direction strong-edge index, built ONCE from a single
            ``get_edges()`` pass and reused across every swept node.
    """

    nodes: Tuple[Dict[str, Any], ...]
    edge_index: StrongEdgeIndex


def snapshot_corpus(
    store: "GraphStoreProtocol", *, scope: Optional[str] = None
) -> CorpusSnapshot:
    """Snapshots the live sweep set and the strong-edge index at run start.

    Exactly two graph reads — one ``get_active_decisions(scope)`` for the sweep set,
    one ``get_edges()`` for the index — never re-read per node (the build-once seam
    the 2a handoff pins; a per-node rebuild would be O(edges) × O(nodes) for zero
    benefit). ``scope`` filters the *proposal set only* (CHK-D2/CONF-D2): a scoped
    snapshot sweeps only matching nodes, but their gathered partners may be any live
    decision — a scoped carve-out can fight a global, so candidate recall stays
    scope-blind downstream. A zero-match scope yields an empty snapshot: empty sweep,
    no pairs, no groups — healthy, not degraded (the "0 of N" wording is the CLI's).

    Store faults propagate (KD5): a failing ``get_edges()`` or
    ``get_active_decisions()`` raises here — no fallback empty index (which would
    silently screen nothing and mislabel every settled pair as fresh), no fallback
    empty sweep set (which would mislabel a broken graph as a clean corpus).

    Args:
        store: The graph store to snapshot.
        scope: An optional scope tag filtering the sweep set (passed through verbatim
            to ``get_active_decisions`` — folding is the CLI's concern).

    Returns:
        The :class:`CorpusSnapshot` the whole run iterates.
    """
    nodes = tuple(store.get_active_decisions(scope))
    edge_index = build_strong_edge_index(store.get_edges())
    return CorpusSnapshot(nodes=nodes, edge_index=edge_index)


def sweep_node(
    node: Dict[str, Any],
    *,
    edge_index: StrongEdgeIndex,
    embed_provider: "EmbeddingProvider",
    vector_store: "VectorStore",
    store: "GraphStoreProtocol",
    floor: float = CONFLICT_SIMILARITY_FLOOR,
    top_k: int = CONFLICT_TOP_K,
) -> "List[Candidate] | Unavailable":
    """Discovers ONE swept node's undeclared conflict candidates (the per-node stage).

    Composes the shipped pipeline stages in the pinned order — ``gather_candidates`` →
    :func:`screen_strong_edge_pairs` → ``screen_candidates`` — so the id-native
    declared-pair drop runs *upstream* of the floor and the ``top_k`` truncation
    (CONF-D7 shadowing: a declared strong neighbour must never consume a slot that
    would shadow a genuine undeclared conflict out of the window).

    The axiom passed to gather is ``node["core_axiom"]`` — a direct subscript, loud
    ``KeyError`` if the shape drifts (the phantom-empty foot-gun's loud half). Gather
    routes it through ``identity.embedding_text``, so the vector is byte-identical to
    what the outbox embedded for this node → the sweep's embeds are cache hits.
    ``declared_targets=set()`` is correct, not a gap: the strong-edge screen already
    did the declared drop id-natively, both directions; ``screen_candidates`` keeps
    the ``own_slug`` self-drop (a KNN echoing the swept node itself), the floor, and
    the rank/truncate.

    Args:
        node: The hydrated swept decision (a ``CorpusSnapshot.nodes`` element).
        edge_index: The run's strong-edge index (built once in the snapshot).
        embed_provider: The injected embedding provider.
        vector_store: The injected vector store.
        store: The injected graph store (gather's live re-verify source).
        floor: The inclusive similarity floor (default ``CONFLICT_SIMILARITY_FLOOR``).
        top_k: The cap on the surviving batch (default ``CONFLICT_TOP_K``).

    Returns:
        The surviving :class:`Candidate` list (possibly ``[]`` — healthy-empty), or
        the typed :class:`Unavailable` gather returned (degraded; passed through as a
        VALUE, verbatim — never blurred into ``[]``, never raised). Graph-store
        faults inside gather propagate (KD5) — only the two semantic-substrate faults
        degrade.
    """
    gathered = gather_candidates(
        node["core_axiom"],
        embed_provider=embed_provider,
        vector_store=vector_store,
        store=store,
    )
    if isinstance(gathered, Unavailable):
        return gathered
    undeclared = screen_strong_edge_pairs(node["id"], gathered, edge_index)
    return screen_candidates(
        undeclared,
        declared_targets=set(),
        own_slug=node["slug"],
        floor=floor,
        top_k=top_k,
    )


@dataclass(frozen=True)
class NodeSweep:
    """One swept node paired with its typed per-node outcome.

    ``result`` is a list (healthy — possibly empty) **or** the shipped
    :class:`~mitos.conflict.Unavailable` (degraded). Consumers fork on
    ``isinstance(result, Unavailable)``, never on emptiness — the same
    type-distinguished-degradation discipline as 1b's ``ReuseIndex`` vs
    ``ReuseUnavailable`` on the telemetry side.

    Attributes:
        node: The hydrated swept decision dict.
        result: ``List[Candidate]`` (healthy) or ``Unavailable`` (degraded).
    """

    node: Dict[str, Any]
    result: "List[Candidate] | Unavailable"


def iter_sweep(
    snapshot: CorpusSnapshot,
    *,
    embed_provider: "EmbeddingProvider",
    vector_store: "VectorStore",
    store: "GraphStoreProtocol",
    floor: float = CONFLICT_SIMILARITY_FLOOR,
    top_k: int = CONFLICT_TOP_K,
) -> "Iterator[NodeSweep]":
    """Lazily sweeps the snapshot, one typed :class:`NodeSweep` per node.

    A generator — laziness IS the breaker seam: no gather work happens for nodes
    beyond the last consumed yield, so the run engine's aggregate breaker needs no
    hook, no callback, no policy parameter. On the first :class:`Unavailable` it
    simply stops consuming and the remainder is structurally skipped — no post-trip
    node ever eats its own timeout (the one-penalty-per-run property, delivered by
    generator semantics instead of a flag). Consumed-count vs ``len(snapshot.nodes)``
    is the caller's swept-vs-skipped accounting.

    Yields in snapshot order (a DB accident — see :class:`CorpusSnapshot`); the
    hash-ordered post-passes make the plan structures order-independent.

    Args:
        snapshot: The run-start snapshot (sweep set + edge index).
        embed_provider: The injected embedding provider.
        vector_store: The injected vector store.
        store: The injected graph store.
        floor: The inclusive similarity floor, threaded to every per-node sweep.
        top_k: The per-node survivor cap, threaded likewise.

    Yields:
        One :class:`NodeSweep` per snapshot node, in snapshot order.
    """
    for node in snapshot.nodes:
        yield NodeSweep(
            node=node,
            result=sweep_node(
                node,
                edge_index=snapshot.edge_index,
                embed_provider=embed_provider,
                vector_store=vector_store,
                store=store,
                floor=floor,
                top_k=top_k,
            ),
        )


@dataclass(frozen=True)
class CorpusPair:
    """One deduped, oriented undeclared pair — the unit the judge will price and see.

    The oriented proposal is the **lexicographically smaller content hash** — a pure
    function of the pair, independent of which side's sweep discovered it, of sweep
    order, and of timestamps. That replayability is what lets verdict reuse (CHK-D3)
    key consistently run after run, and it is why the run engine stamps telemetry's
    ``judged_axiom``/``proposed_hash_if_any`` from the oriented proposal, never from
    discovery context. Both nodes ride whole (hydrated dicts): the engine needs live
    slugs for prompt rendering and report display (MI-2) plus the fed-context fields
    for telemetry — pre-projecting ``JudgeInput``\\ s here would duplicate what the
    nodes already carry.

    Attributes:
        proposal_hash: The lex-smaller content hash — THE oriented proposal.
        partner_hash: The other side's content hash.
        proposal_node: The proposal's hydrated node dict (either side may be the
            "undiscovering" one — same shape, same adapter).
        partner_node: The partner's hydrated node dict.
        score: The gathered similarity — informational context only, never an
            ordering key (two discoveries of one pair carry independently-computed
            floats; hash order is the only ordering anywhere in the plan).
    """

    proposal_hash: str
    partner_hash: str
    proposal_node: Dict[str, Any]
    partner_node: Dict[str, Any]
    score: float


def dedup_oriented_pairs(sweeps: "Iterable[NodeSweep]") -> "List[CorpusPair]":
    """Dedups the discovered pairs corpus-wide — each unordered pair exactly once.

    The dedup key is the unordered content-hash pair (``tuple(sorted((a, b)))`` — the
    same convention 1b's ``ReuseIndex`` keys on, so the engine's reuse lookup joins
    without adaptation). Ids come from ``node["id"]`` direct subscripts — real strings
    on every hydrated node, which is what keeps ``ReuseIndex.lookup``'s fail-loud
    ``None`` contract intact downstream.

    Degraded sweeps contribute zero pairs and raise nothing — but a pair the *other*
    side discovered with a degraded node still stands (partial coverage is labeled by
    the engine holding the same list, not compensated for here). When one pair is
    discovered from both sides, retention is **order-independent** (never keep-first):
    the discovery whose sweep node IS the oriented proposal wins, falling back to the
    partner-side discovery — deterministic whichever side the consumer iterated first,
    so the retained ``score`` and node dicts never depend on sweep order.

    Args:
        sweeps: The consumed :class:`NodeSweep`\\ s (a list or any iterable — the
            engine hands over exactly what it consumed before any breaker trip).

    Returns:
        The oriented pairs, sorted by pair key — a deterministic, replayable pure
        function of the healthy sweep results.
    """
    retained: Dict[Tuple[str, str], CorpusPair] = {}
    for sweep in sweeps:
        if isinstance(sweep.result, Unavailable):
            continue
        sweep_id = sweep.node["id"]
        for candidate in sweep.result:
            partner_id = candidate.node["id"]
            proposal_hash, partner_hash = sorted((sweep_id, partner_id))
            key = (proposal_hash, partner_hash)
            discovered_from_proposal = sweep_id == proposal_hash
            if not discovered_from_proposal and key in retained:
                continue  # the incumbent stands (proposal-side wins; first otherwise)
            if discovered_from_proposal:
                proposal_node, partner_node = sweep.node, candidate.node
            else:
                proposal_node, partner_node = candidate.node, sweep.node
            retained[key] = CorpusPair(
                proposal_hash=proposal_hash,
                partner_hash=partner_hash,
                proposal_node=proposal_node,
                partner_node=partner_node,
                score=candidate.score,
            )
    return [retained[key] for key in sorted(retained)]


@dataclass(frozen=True)
class JudgmentGroup:
    """One judgment-sized batch: an oriented proposal + its partner pairs (≤ top_k).

    The shape the run engine turns into one batched judgment call after the reuse
    partition removes the already-judged pairs (which is why grouping is a separate
    seam from dedup — reused pairs never enter a batch).

    Attributes:
        proposal_hash: The group's oriented proposal hash.
        proposal_node: The proposal's hydrated node dict.
        pairs: The group's :class:`CorpusPair`\\ s, partner-hash-sorted, ≤ ``top_k``.
    """

    proposal_hash: str
    proposal_node: Dict[str, Any]
    pairs: Tuple[CorpusPair, ...]


def group_judgment_batches(
    pairs: "Iterable[CorpusPair]",
    *,
    top_k: int = CONFLICT_TOP_K,
) -> "List[JudgmentGroup]":
    """Groups oriented pairs into judgment-sized batches — a pure, replayable pass.

    Groups are keyed and ordered by proposal hash; partners within a group are
    partner-hash-sorted; a group with more than ``top_k`` partners splits into
    consecutive chunks of ≤ ``top_k`` in that order. Hash order everywhere, score
    nowhere: two runs over the same snapshot and gather results produce
    byte-identical batch structures — replayability is worth more than mirroring the
    sync batch's similarity-descending cosmetic order (the judge sees the whole batch
    regardless).

    Args:
        pairs: The deduped oriented pairs (any iterable).
        top_k: The batch-size cap (default ``CONFLICT_TOP_K`` — the same window the
            sync-time judge sees).

    Returns:
        The :class:`JudgmentGroup`\\ s, proposal-hash-ordered (split chunks adjacent,
        in partner-hash order). ``len(...)`` of the post-reuse remainder is the exact
        batch count the CHK-D5 disclosure names.
    """
    by_proposal: Dict[str, List[CorpusPair]] = {}
    for pair in pairs:
        by_proposal.setdefault(pair.proposal_hash, []).append(pair)
    groups: List[JudgmentGroup] = []
    for proposal_hash in sorted(by_proposal):
        ordered = sorted(by_proposal[proposal_hash], key=lambda pair: pair.partner_hash)
        for start in range(0, len(ordered), top_k):
            chunk = tuple(ordered[start:start + top_k])
            groups.append(
                JudgmentGroup(
                    proposal_hash=proposal_hash,
                    proposal_node=chunk[0].proposal_node,
                    pairs=chunk,
                )
            )
    return groups
