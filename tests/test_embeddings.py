"""Adversarial test suite for the Mitos embedding pipeline.

Verifies SQLite-backed caching logic, asymmetric prefixing rules, cache stats
observability, and mock provider API interaction.
"""

import tempfile
import os
import pytest
from unittest.mock import MagicMock, patch
from mitos.embeddings import EmbeddingCache, GeminiEmbeddingProvider
from mitos.errors import EmbeddingError

@pytest.fixture
def temp_cache_path() -> str:
    """Fixture returning path to a temporary cache database file."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_embedding_cache_get_set(temp_cache_path: str) -> None:
    """Tests basic SQLite embedding cache get/set operations."""
    cache = EmbeddingCache(temp_cache_path)
    
    # Try getting nonexistent
    assert cache.get("nonexistent") is None
    
    # Set and get vector
    vector = [0.1, 0.2, 0.3, 0.4]
    cache.set("hash-123", vector)
    
    retrieved = cache.get("hash-123")
    assert retrieved == vector


@patch("google.genai.Client")
def test_embedding_provider_observability(mock_client: MagicMock, temp_cache_path: str) -> None:
    """Tests asymmetric prefixing, cache hits/misses, and API client fallback."""
    os.environ["GEMINI_API_KEY"] = "mock_key"
    
    provider = GeminiEmbeddingProvider(temp_cache_path)
    assert provider.model_id == "gemini-embedding-2"
    
    # Mock embed API response
    mock_resp = MagicMock()
    mock_emb = MagicMock()
    mock_emb.values = [0.9, 0.8, 0.7]
    mock_resp.embeddings = [mock_emb]
    mock_client.return_value.models.embed_content.return_value = mock_resp

    # 1. First call -> Cache Miss
    vec1 = provider.get_embedding("My text block", is_query=False)
    assert vec1 == [0.9, 0.8, 0.7]
    
    hits, misses, rate = provider.get_stats()
    assert hits == 0
    assert misses == 1
    assert rate == 0.0
    
    # Verify document prefix applied
    mock_client.return_value.models.embed_content.assert_called_with(
        model="gemini-embedding-2",
        contents="search_document: My text block"
    )

    # 2. Second call -> Cache Hit
    vec2 = provider.get_embedding("My text block", is_query=False)
    assert vec2 == [0.9, 0.8, 0.7]
    
    hits, misses, rate = provider.get_stats()
    assert hits == 1
    assert misses == 1
    assert rate == 0.5


@patch("google.genai.Client")
def test_embeddings_batch_token_aware(mock_client: MagicMock, temp_cache_path: str) -> None:
    """Tests that get_embeddings_batch executes bulk requests and utilizes cache correctly."""
    os.environ["GEMINI_API_KEY"] = "mock_key"
    provider = GeminiEmbeddingProvider(temp_cache_path)

    # Mock embed API response for batch of 2
    mock_resp = MagicMock()
    mock_emb1 = MagicMock()
    mock_emb1.values = [0.1, 0.1, 0.1]
    mock_emb2 = MagicMock()
    mock_emb2.values = [0.2, 0.2, 0.2]
    mock_resp.embeddings = [mock_emb1, mock_emb2]
    mock_client.return_value.models.embed_content.return_value = mock_resp

    texts = ["First block", "Second block"]
    vectors = provider.get_embeddings_batch(texts)
    
    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.1, 0.1]
    assert vectors[1] == [0.2, 0.2, 0.2]
    
    # Assert cache stats show misses
    hits, misses, rate = provider.get_stats()
    assert misses == 2
    assert hits == 0

    # Repeat call: should hit cache entirely
    provider.reset_stats()
    vectors_cached = provider.get_embeddings_batch(texts)
    assert len(vectors_cached) == 2
    hits, misses, rate = provider.get_stats()
    assert hits == 2
    assert misses == 0


def test_embedding_cache_eviction_ceiling(temp_cache_path: str) -> None:
    """Verifies that the cache evicts the oldest entries when exceeding 5000 records."""
    import sqlite3
    import json
    
    cache = EmbeddingCache(temp_cache_path)
    
    # Pre-populate 5000 entries efficiently inside a single transaction
    with sqlite3.connect(temp_cache_path) as conn:
        conn.execute("BEGIN TRANSACTION;")
        for i in range(5000):
            conn.execute(
                "INSERT INTO cache (content_hash, vector) VALUES (?, ?)",
                (f"hash-{i}", json.dumps([0.1 * i]))
            )
        conn.commit()
        
    # Check count is 5000
    with sqlite3.connect(temp_cache_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        assert count == 5000
        
    # Add 5001st record. This should trigger eviction!
    cache.set("hash-new", [0.99])
    
    # Verify count remains 5000
    with sqlite3.connect(temp_cache_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        assert count == 5000
        
        # Verify "hash-0" (the oldest record) has been evicted
        oldest = conn.execute("SELECT 1 FROM cache WHERE content_hash = 'hash-0'").fetchone()
        assert oldest is None
        
        # Verify the newest record exists
        newest = conn.execute("SELECT 1 FROM cache WHERE content_hash = 'hash-new'").fetchone()
        assert newest is not None

