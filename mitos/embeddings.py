"""Embedding pipeline for Mitos.

This module implements the embedding provider (D), standardizing cache checks,
observability counters, and provider integrations.
"""

import sqlite3
import json
import hashlib
import os
from typing import List, Dict, Optional, Any, Tuple
from google import genai
from mitos.models import get_embedding_model_id
from mitos.errors import EmbeddingError

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

    def __init__(self, cache_path: str) -> None:
        self.cache = EmbeddingCache(cache_path)
        self.hits = 0
        self.misses = 0
        
        # API Client initialization using the new google-genai SDK
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EmbeddingError("GEMINI_API_KEY environment variable is not set")
        self.client = genai.Client(api_key=api_key)
        self.model_id = get_embedding_model_id()

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
        prefix = "search_query: " if is_query else "search_document: "
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
            raise EmbeddingError(f"Gemini embedding API call failed: {str(e)}")

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Gets embeddings for a batch of documents, utilizing token-aware bounds.

        Args:
            texts: List of text blocks.

        Returns:
            List of embedding vectors.
        """
        results: List[Optional[List[float]]] = [None] * len(texts)
        miss_indices = []
        miss_texts = []

        # 1. Resolve cache hits first
        for idx, text in enumerate(texts):
            prefix = "search_document: "
            prefixed_text = prefix + text
            content_hash = self.compute_content_hash(prefixed_text)
            
            cached_vector = self.cache.get(content_hash)
            if cached_vector is not None:
                self.hits += 1
                results[idx] = cached_vector
            else:
                self.misses += 1
                miss_indices.append(idx)
                # Keep prefixed text for bulk API call
                miss_texts.append(prefixed_text)

        # 2. Process cache misses in token-aware batches
        if miss_texts:
            max_batch_count = 100
            max_batch_chars = 40000  # ~10,000 tokens
            
            current_batch = []
            current_batch_chars = 0
            batches = []
            
            for prefixed_t in miss_texts:
                if len(current_batch) >= max_batch_count or (current_batch_chars + len(prefixed_t)) > max_batch_chars:
                    batches.append(current_batch)
                    current_batch = [prefixed_t]
                    current_batch_chars = len(prefixed_t)
                else:
                    current_batch.append(prefixed_t)
                    current_batch_chars += len(prefixed_t)
            if current_batch:
                batches.append(current_batch)

            # 3. Call bulk API for each batch and populate results/cache
            processed_count = 0
            for batch in batches:
                try:
                    response = self.client.models.embed_content(
                        model=self.model_id,
                        contents=batch
                    )
                    
                    if not response.embeddings:
                        raise EmbeddingError("Gemini API returned empty embeddings in batch call")
                    
                    for i, emb in enumerate(response.embeddings):
                        vector = emb.values
                        # Store in results at original index
                        orig_idx = miss_indices[processed_count + i]
                        results[orig_idx] = vector
                        
                        # Cache the result
                        text_item = batch[i]
                        content_hash = self.compute_content_hash(text_item)
                        self.cache.set(content_hash, vector)
                        
                    processed_count += len(batch)
                except Exception as e:
                    raise EmbeddingError(f"Batch embedding API call failed: {str(e)}")

        # Filter out any None values (should be none)
        return [r for r in results if r is not None]
