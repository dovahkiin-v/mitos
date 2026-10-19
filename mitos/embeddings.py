"""Embedding pipeline for Mitos.

This module implements the embedding provider (D), standardizing cache checks,
observability counters, and provider integrations.
"""

import sqlite3
import json
import hashlib
import os
from typing import List, Dict, Optional, Any, Sequence, Tuple
from mitos.models import get_embedding_model_id
from mitos.errors import EmbeddingError

# The asymmetric prefix every document embedding is sent with. One spelling for
# the two senders (the cached ``get_embedding`` and the uncached
# ``embed_documents``); ``rejections.DOCUMENT_PREFIX_BYTES`` derives the item-part
# bound from its byte length, and ``tests/test_rejections.py`` tripwires the bytes
# each path actually sends.
DOCUMENT_PREFIX = "search_document: "

# The batch bounds of one ``embed_content`` call: at most this many texts, and at
# most this many UTF-8 bytes of PREFIXED text. ``fit_batch`` and ``embed_documents``
# both read them at call time, so the batches the one batched path sends are within
# the bounds literally.
EMBED_BATCH_MAX_TEXTS = 100
EMBED_BATCH_MAX_BYTES = 40_000


def _prefixed_bytes(text: str) -> int:
    """Returns the UTF-8 byte length of ``text`` as sent, with the document prefix."""
    return len((DOCUMENT_PREFIX + text).encode("utf-8"))


def fit_batch(texts: Sequence[str]) -> int:
    """Counts how many leading texts fit one uncached embedding call.

    Pure. A text fits while the batch holds at most ``EMBED_BATCH_MAX_TEXTS``
    texts and at most ``EMBED_BATCH_MAX_BYTES`` prefixed UTF-8 bytes in total.
    Non-empty input always yields at least 1, so a caller walking a long list
    always progresses; that rule is for progress only and is not a guard — a
    single text over the byte bound is still refused by ``embed_documents``.

    Args:
        texts: The unprefixed texts, in the order they would be sent.

    Returns:
        The length of the longest leading run that fits (0 for empty input).
    """
    count = 0
    total = 0
    for text in texts:
        size = _prefixed_bytes(text)
        if count >= EMBED_BATCH_MAX_TEXTS or total + size > EMBED_BATCH_MAX_BYTES:
            break
        count += 1
        total += size
    if count == 0 and texts:
        return 1
    return count

class EmbeddingCache:
    """SQLite-backed persistent cache for text embeddings."""

    def __init__(self, cache_path: str) -> None:
        self.cache_path = cache_path
        self._init_db()

    def _init_db(self) -> None:
        db_dir = os.path.dirname(self.cache_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    content_hash TEXT PRIMARY KEY,
                    vector TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def get(self, content_hash: str) -> Optional[List[float]]:
        """Retrieves a cached vector by content hash."""
        try:
            with sqlite3.connect(self.cache_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT vector FROM cache WHERE content_hash = ?", (content_hash,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row["vector"])
        except Exception:
            pass
        return None

    def set(self, content_hash: str, vector: List[float]) -> None:
        """Stores a vector in the cache, enforcing a 5000-record ceiling."""
        try:
            with sqlite3.connect(self.cache_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (content_hash, vector) VALUES (?, ?)",
                    (content_hash, json.dumps(vector))
                )
                
                # Enforce size ceiling of 5000 records
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM cache")
                count = cursor.fetchone()[0]
                if count > 5000:
                    # Evict the oldest records
                    conn.execute(
                        """
                        DELETE FROM cache WHERE content_hash IN (
                            SELECT content_hash FROM cache 
                            ORDER BY created_at ASC, rowid ASC 
                            LIMIT ?
                        )
                        """,
                        (count - 5000,)
                    )
        except Exception:
            pass


class GeminiEmbeddingProvider:
    """Gemini-based implementation of the embedding provider."""

    def __init__(
        self,
        cache_path: str,
        *,
        api_key: Optional[str],
        model_id: Optional[str] = None,
    ) -> None:
        """Builds the provider for one workspace's key and embedding model.

        Args:
            cache_path: The content-hash embedding cache (its key carries **no**
                model id, which is why ``model_id`` must arrive resolved for the
                right target rather than be read here).
            api_key: The workspace's ``GEMINI_API_KEY``. ``None`` means the caller
                supplied nothing and the provider refuses — it consults no
                environment of its own (5c). ``""`` is a supplied answer and
                refuses the same way, which is what keeps
                ``env GEMINI_API_KEY= mitos …`` a keyless run.
            model_id: The resolved embedding model id, or ``None`` for the
                baseline. Never stored beyond ``self.model_id``; the key is never
                stored at all (P8).
        """
        self.cache = EmbeddingCache(cache_path)
        self.hits = 0
        self.misses = 0

        # API Client initialization using the new google-genai SDK
        if not api_key:
            # A vector, not a wall (P3): the old wording named the process
            # environment, which is now one of three places the key can live and
            # the least likely of them. Name the search order and the verb that
            # writes one. The provider itself reads no environment.
            raise EmbeddingError(
                "GEMINI_API_KEY is not set for this workspace — mitos reads it "
                "from the process environment, then the workspace's .env, then "
                "the global .env (`mitos set-key <key>` writes one)"
            )
        # Imported here, not at module scope: `google.genai` costs ~0.7s to import,
        # and every `mitos` verb imports this module whether or not it embeds.
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model_id = (
            model_id if model_id is not None else get_embedding_model_id()
        )

    def get_stats(self) -> Tuple[int, int, float]:
        """Returns cache stats: (hits, misses, hit_rate)."""
        total = self.hits + self.misses
        rate = (self.hits / total) if total > 0 else 0.0
        return self.hits, self.misses, rate

    def reset_stats(self) -> None:
        """Resets the observability hit/miss counters."""
        self.hits = 0
        self.misses = 0

    def compute_content_hash(self, text: str) -> str:
        """Computes SHA-256 hash over text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """Gets embedding for a single text using prefix and cache.

        Args:
            text: The text block to embed.
            is_query: Whether this is a query prefix or document prefix.

        Returns:
            The embedding vector.
        """
        # Apply asymmetric prefixing
        prefix = "search_query: " if is_query else DOCUMENT_PREFIX
        prefixed_text = prefix + text
        
        content_hash = self.compute_content_hash(prefixed_text)
        
        # 1. Check cache
        cached_vector = self.cache.get(content_hash)
        if cached_vector is not None:
            self.hits += 1
            return cached_vector

        self.misses += 1
        
        # 2. Call API on cache miss
        try:
            # We use types.TaskType.RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY if needed,
            # but the direct API expects the prefix as part of text in text-embedding-004.
            response = self.client.models.embed_content(
                model=self.model_id,
                contents=prefixed_text
            )
            
            # Extract vector
            if not response.embeddings or len(response.embeddings) == 0:
                raise EmbeddingError("Gemini API returned an empty embedding list")
                
            vector = response.embeddings[0].values
            
            # Cache the result
            self.cache.set(content_hash, vector)
            return vector
            
        except Exception as e:
            raise EmbeddingError(f"Gemini embedding API call failed: {str(e)}") from e

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Embeds documents in exactly one provider call, never touching the cache.

        The rejection index's embedding path. It neither reads nor writes
        ``self.cache`` and never moves ``hits``/``misses`` (they describe the
        cache): an item's point id derives from its text, so the index's own
        presence answer is its cache (ADR
        ``rejection-index-embeddings-bypass-the-shared-embedding-cache``).

        The bounds are checked absolutely, before any call, so "one call within
        the declared bounds" holds for every caller: more than
        ``EMBED_BATCH_MAX_TEXTS`` texts, or more than ``EMBED_BATCH_MAX_BYTES``
        prefixed UTF-8 bytes in total (a single oversized text included), is a
        caller's programming error. ``fit_batch`` sizes a batch that passes.

        Args:
            texts: The unprefixed document texts; each is sent with
                ``DOCUMENT_PREFIX``.

        Returns:
            One non-empty vector per text, in input order. Empty input returns
            ``[]`` with no call.

        Raises:
            ValueError: If the batch exceeds either bound.
            EmbeddingError: If the call fails (chained from the SDK exception,
                so ``provider_cause`` can classify it) or the answer is
                malformed: no embeddings, a count other than ``len(texts)``, or
                an element whose ``values`` is missing or empty.
        """
        if not texts:
            return []
        if len(texts) > EMBED_BATCH_MAX_TEXTS:
            raise ValueError(
                f"embed_documents takes at most {EMBED_BATCH_MAX_TEXTS} texts per "
                f"call, got {len(texts)}"
            )
        total = sum(_prefixed_bytes(t) for t in texts)
        if total > EMBED_BATCH_MAX_BYTES:
            raise ValueError(
                f"embed_documents takes at most {EMBED_BATCH_MAX_BYTES} prefixed "
                f"UTF-8 bytes per call, got {total}"
            )
        # One ``Content`` per text, never a list of strings: measured 2026-10-17
        # (google-genai 2.8.0, gemini-embedding-2), a list of N strings is read as
        # ONE content of N parts and answers ONE embedding, while N ``Content``
        # answer N, each identical to the single-text call. The answer check below
        # is what caught it; ``tests/test_rejection_inline_live.py`` pins it live.
        # Imported here, not at module scope, for the reason ``__init__`` gives.
        from google.genai import types

        contents = [
            types.Content(parts=[types.Part(text=DOCUMENT_PREFIX + t)]) for t in texts
        ]
        # The try wraps the SDK call alone, so the malformed-answer errors below
        # are raised once, never re-wrapped and chained to themselves.
        try:
            response = self.client.models.embed_content(
                model=self.model_id,
                contents=contents,
            )
        except Exception as e:
            raise EmbeddingError(
                f"Gemini embedding API call failed: {str(e)}"
            ) from e
        # Malformed-answer messages stay constant (no counts, no text):
        # ``provider_cause`` matches "429" anywhere in a message, so an
        # interpolated number could misclassify this as quota.
        embeddings = getattr(response, "embeddings", None)
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise EmbeddingError(
                "Gemini API returned a document embedding list that does not "
                "match the batch"
            )
        vectors: List[List[float]] = []
        for emb in embeddings:
            values = getattr(emb, "values", None)
            if not isinstance(values, list) or not values:
                raise EmbeddingError(
                    "Gemini API returned a document embedding with no values"
                )
            vectors.append(values)
        return vectors
