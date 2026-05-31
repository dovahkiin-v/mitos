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
        """Resolves a slug to matching node IDs (case-insensitive).

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

    def add_pending_embedding(self, node_id: str, embedding_text: str) -> None:
        """Adds a node to the pending outbox queue.

        Args:
            node_id: The node ID.
            embedding_text: The canonical text to embed.
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
