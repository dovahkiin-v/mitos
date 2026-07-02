"""Formal typing protocols for Mitos components.

This module defines Protocol classes for GraphStore, VectorStore, and
EmbeddingProvider to decouple Mitos components and support seamless swap-ins.
"""

from typing import Protocol, List, Dict, Optional, Any, Tuple, Set
from mitos.parser import ParsedEntry
from mitos.store import CommitDelta

class EmbeddingProvider(Protocol):
    """Protocol for cache-aware, prefix-based text embedding providers."""

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """Gets embedding vector for a single text.

        Args:
            text: The text block to embed.
            is_query: Whether this is a query prefix or document prefix.

        Returns:
            The embedding vector.
        """
        ...

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Gets embedding vectors for a batch of texts.

        Args:
            texts: List of text blocks.

        Returns:
            List of embedding vectors.
        """
        ...

    def get_stats(self) -> Tuple[int, int, float]:
        """Returns hit/miss cache statistics.

        Returns:
            A tuple of (hits, misses, hit_rate).
        """
        ...

    def reset_stats(self) -> None:
        """Resets the observability statistics counters."""
        ...


class VectorStore(Protocol):
    """Protocol for point-management and semantic similarity query interfaces."""

    def upsert(self, point_id: str, vector: List[float], payload: Dict[str, Any]) -> None:
        """Upserts a single vector and payload into the vector index.

        Args:
            point_id: The SHA-256 node ID.
            vector: The embedding vector values.
            payload: Node metadata.
        """
        ...

    def query(
        self,
        vector: List[float],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Queries for the semantically nearest vectors.

        Recall is scope-blind by contract — scope is a downstream discoverability
        hint, never a recall filter (see :meth:`QdrantVectorStore.query`).

        Args:
            vector: The query embedding vector.
            limit: Maximum matches to return.

        Returns:
            A list of dictionary results with payload and scores.
        """
        ...

    def list_point_ids(self, page_size: int = 256) -> Set[str]:
        """Lists every point id currently in the index.

        Args:
            page_size: Maximum points fetched per page.

        Returns:
            The set of point-id strings in the index.
        """
        ...


class GraphStoreProtocol(Protocol):
    """Protocol for the relational, SQLite-backed decision graph."""

    def resolve_slug(self, slug: str) -> List[str]:
        """Resolves a slug to matching node IDs via casefold-exact match (V1-D23).

        Single-tier ``str.casefold()`` match against ``slug_casefold`` — no fuzzy
        alias-fallback tier (MI-9); a renamed-away slug is not silently repaired.

        Args:
            slug: The slug string to find.

        Returns:
            A list of matching node IDs.
        """
        ...

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single node by its ID.

        Args:
            node_id: The primary key ID of the node.

        Returns:
            A dictionary containing the node data, or None if not found.
        """
        ...

    def get_node_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single node by slug, raising on ambiguity.

        Args:
            slug: The slug identifier.

        Returns:
            The node dictionary or None.
        """
        ...

    def write_signal(self, node_id: str, stype: str, source: Optional[str] = None) -> None:
        """Writes a signal row (drifted, source_reencounter).

        Args:
            node_id: The node to signal.
            stype: One of 'drifted', 'source_reencounter'.
            source: Optional source enum value the signal carries (uniform with
                ``nodes.source`` — V1-D14).
        """
        ...

    def note_source_reencounter(
        self, node_id: str, stored_source: str, new_source: str
    ) -> bool:
        """Emits one source_reencounter signal iff the re-encountering source differs.

        The V1-D14 policy wrapper over ``write_signal``, called at the four
        node-exists short-circuit gates (MI-4 cross-source provenance audit).

        Args:
            node_id: The re-encountered node's content-hash id.
            stored_source: The node's first-seen (MI-4-fenced) source.
            new_source: The re-encountering source enum value.

        Returns:
            True iff a signal write was committed (or idempotently ignored); False if
            the source was unchanged or the best-effort write was dropped.
        """
        ...

    def commit_parsed_entry(self, parsed: ParsedEntry) -> CommitDelta:
        """Commits a single ParsedEntry within an atomic transaction.

        Args:
            parsed: The ParsedEntry representation.

        Returns:
            A structured CommitDelta payload.
        """
        ...

    def get_active_decisions(self, scope: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves all currently active or drifted decisions.

        Args:
            scope: Optional scope filter.

        Returns:
            List of active decision node dicts.
        """
        ...

    def get_open_questions(self, scope: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves all open questions.

        Args:
            scope: Optional scope filter.

        Returns:
            List of open question node dicts.
        """
        ...

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Retrieves all nodes with computed states.

        Returns:
            List of all node dicts.
        """
        ...

    def get_edges(self) -> List[Dict[str, str]]:
        """Retrieves all graph edges.

        Returns:
            List of edge dictionaries.
        """
        ...

    def get_modifiers_map(self, node_ids: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """Maps each node to the slugs of later decisions that modify it.

        Args:
            node_ids: The node IDs to look up modifiers for.

        Returns:
            A mapping of node_id -> {reverse_relation_key: [modifier_slug, ...]},
            containing only nodes that actually have modifiers.
        """
        ...

    def get_modifiers(self, node_id: str) -> Dict[str, List[str]]:
        """Returns the reverse-relation modifiers for a single node.

        Args:
            node_id: The node ID to look up modifiers for.

        Returns:
            A mapping of reverse-relation key to modifier slugs, or ``{}``.
        """
        ...

    def get_contradictions(self, node_id: str) -> List[Dict[str, str]]:
        """Returns the nodes that contradict ``node_id``, from EITHER direction.

        ``contradicts`` is symmetric and stored once (A->B); this is the single
        safe bidirectional read — no consumer hand-rolls
        ``WHERE source=X OR target=X``. Returns counterparts regardless of either
        endpoint's active-view state (v0.1 takes no active-view action on
        ``contradicts``). Identity only, not a hydrated decision payload.

        Args:
            node_id: The node whose contradictions to read.

        Returns:
            One ``{node_id, slug, kind}`` dict per distinct counterpart node,
            deduplicated across the two edge directions; ``[]`` when none.
        """
        ...

    def get_lineage(self, node_id: str) -> List[Dict[str, str]]:
        """Returns ``node_id``'s transitive mutation ancestors.

        Walks ``node_id``'s outgoing mutation edges (``supersedes`` ∪ ``amends`` ∪
        ``narrows``; ``corrects`` excluded) transitively, new→old. Identity only,
        not a hydrated decision payload. On a corrupt/out-of-band cycle: truncate,
        log a loud diagnostic naming the node, return the partial lineage (never
        loops, never raises).

        Args:
            node_id: The node whose mutation ancestry to read.

        Returns:
            One ``{node_id, slug, kind}`` dict per distinct ancestor, sorted by
            slug, excluding ``node_id`` itself; ``[]`` when none.
        """
        ...

    def get_unregistered_mechanisms(self, parsed: ParsedEntry) -> List[str]:
        """Returns the subset of ``parsed``'s mechanism refs NOT yet registered.

        Read-only pre-commit feedback query (registers nothing): given a parsed
        entry, returns which cited mechanisms are not yet in the ``mechanisms``
        registry, so an interactive-review surface can flag a typo/alias before
        auto-registration fires. Keys each ref by ``mechanism_canonical_norm``,
        returns the not-yet-present ones (authored form when available), order-stable
        and deduped by ``canonical_name``. Decision-gated — an open question carries
        no mechanisms → ``[]``.

        Args:
            parsed: The entry whose mechanism refs to check against the registry.

        Returns:
            The authored (or ref-fallback) form of each unregistered cited
            mechanism, first-seen order; ``[]`` when all registered / none cited /
            open question.
        """
        ...

    def get_decisions(
        self, scope: Optional[str] = None, state: str = "active"
    ) -> List[Dict[str, Any]]:
        """Enumerates decision nodes matching scope and computed state.

        Args:
            scope: Optional scope filter.
            state: ``"active"`` (live set), ``"all"``, or an exact computed state.

        Returns:
            List of decision node dicts with ``computed_state`` attached.
        """
        ...

    def get_transcript(self, node_id: str) -> Optional[str]:
        """Returns a node's own committed transcript text, or None.

        Args:
            node_id: The node whose transcript to read.

        Returns:
            The raw transcript text, or None.
        """
        ...

    def query_letter(
        self,
        *,
        scope: Optional[str] = None,
        kind: str = "decision",
        slug: Optional[str] = None,
        node_id: Optional[str] = None,
        brief: bool = False,
    ) -> List[Dict[str, Any]]:
        """Structured-filter Letter query over the active view (C4) — no semantic path.

        Args:
            scope: Optional scope tag.
            kind: ``"decision"`` or ``"open_question"``.
            slug: Optional exact slug (casefolded).
            node_id: Optional exact content-hash id.
            brief: When True, omit ``rejected_paths``.

        Returns:
            Letter payload dicts (active-view only), each modifier-stamped.
        """
        ...

    def add_pending_embedding(self, node_id: str) -> None:
        """Enqueues a node onto the pending-embeddings outbox (V1a 3-column shape).

        Aligned to the V1a ``pending_embeddings`` schema (Phase 8a): no
        ``embedding_text`` argument — the drainer re-derives ``embedding_text(node)``
        at drain time from the node's immutable core (C2/M8), so nothing is stored.

        Args:
            node_id: The node ID to enqueue.
        """
        ...

    def get_pending_embeddings(self) -> List[Dict[str, Any]]:
        """Retrieves all queued pending embeddings.

        Returns:
            List of pending embedding dictionaries.
        """
        ...

    def remove_pending_embedding(self, node_id: str) -> None:
        """Removes a resolved node from the outbox queue.

        Args:
            node_id: The node ID to remove.
        """
        ...

    def increment_pending_attempts(self, node_id: str) -> None:
        """Increments the retry attempt count for a queued embedding.

        Args:
            node_id: The node ID to increment.
        """
        ...
