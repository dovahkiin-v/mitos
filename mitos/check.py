"""The corpus-audit engine's core — the either-direction strong-edge pair screen (2a).

This module is the home of ``mitos check``: the corpus-wide conflict audit that sweeps
every decision and judges the genuinely *undeclared* tensions between similar pairs. The
sync-time sensor (``conflict.py``) judges one proposal against its neighbours at write
time; the corpus audit has no privileged direction — it must read a relationship as a
fact about a *pair*, not about whichever node the sweep happened to iterate first.

Phase 2a lands the first two composition pieces: the **either-direction strong-edge
screen** that removes, id-natively, every pair the author already settled with a declared
strong relationship — so the LLM judge is never asked to second-guess a carve-out the
author was most careful to make. A relationship is stored once, as authored
(``source → target``), but "A narrows B" and "B is narrowed by A" are the same settled
decision, so the index records **both** endpoints and the screen is direction-free.

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

from typing import Dict, List, Set

from mitos.conflict import Candidate, _STRONG_RELATIONSHIP_FIELDS


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
