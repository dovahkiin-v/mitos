"""Formal typing protocols for Mitos components.

This module defines Protocol classes for GraphStore, VectorStore, and
EmbeddingProvider to decouple Mitos components and support seamless swap-ins.
"""

from typing import Protocol, List, Dict, Optional, Any, Tuple
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
        filter_scope: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Queries for similar vectors, supporting optional scope pre-filtering.

        Args:
            vector: The query embedding vector.
            limit: Maximum matches to return.
            filter_scope: Optional scope tag to filter results by.

        Returns:
            A list of dictionary results with payload and scores.
        """
        ...


class GraphStoreProtocol(Protocol):
    """Protocol for the relational, SQLite-backed decision graph."""

    def resolve_slug(self, slug: str) -> List[str]:
        """Resolves a slug to matching node IDs via casefold-exact match (V1-D23).

        Single-tier ``str.casefold()`` match against ``slug_casefold`` — no fuzzy
        alias-fallback tier (MI-9; the ``slug_aliases`` subsystem is V1b/MI-2).

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

    def write_signal(self, node_id: str, stype: str, actor: Optional[str] = None) -> None:
        """Writes a signal row (drifted, source_reencounter).

        Args:
            node_id: The node to signal.
            stype: One of 'drifted', 'source_reencounter'.
            actor: Optional string actor name.
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
