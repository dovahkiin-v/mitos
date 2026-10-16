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
    # No env write: conftest's credential strip retired the collection-order
    # leak this row once fed (the ~113-row crutch of 5c §7.2) — since 5c the
    # provider consults no environment at all, the explicit key below is the one
    # it uses.
    provider = GeminiEmbeddingProvider(temp_cache_path, api_key="mock_key")
    # No `model_id=`: this asserts the leaf's own baseline fallback, which is the
    # `model_id is None` branch and stays live (UNROUTED_MODEL_RESOLVES).
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
    provider = GeminiEmbeddingProvider(temp_cache_path, api_key="mock_key")

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



# --------------------------------------------------------------------------- #
# The uncached document batch (recall-confidence 4b): fit_batch, embed_documents
# --------------------------------------------------------------------------- #

from types import SimpleNamespace

from mitos.embeddings import (
    DOCUMENT_PREFIX,
    EMBED_BATCH_MAX_BYTES,
    EMBED_BATCH_MAX_TEXTS,
    fit_batch,
)
from mitos.provider_cause import QUOTA, UNKNOWN, classify_provider_error


class _SdkError(Exception):
    """An SDK-shaped error carrying the typed fields the classifier reads."""

    def __init__(self, message: str, **fields) -> None:
        super().__init__(message)
        for name, value in fields.items():
            setattr(self, name, value)


def _sent_texts(contents) -> list:
    """The texts one ``embed_content`` call carried: one ``Content`` per text."""
    return [content.parts[0].text for content in contents]


def _recording_provider(cache_path: str, answer=None):
    """A real provider whose client is a recorder: zero calls is a list length."""
    sent: list = []

    def embed_content(*, model: str, contents):
        sent.append(contents)
        if answer is not None:
            return answer(contents)
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1, 0.2]) for _ in contents])

    provider = GeminiEmbeddingProvider(cache_path, api_key="dummy", model_id="m")
    provider.client = SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
    return provider, sent


def _cache_count(path: str) -> int:
    import sqlite3

    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    finally:
        conn.close()


def _prefixed(text: str) -> int:
    return len((DOCUMENT_PREFIX + text).encode("utf-8"))


def test_fit_batch_counts_texts_and_prefixed_utf8_bytes() -> None:
    assert fit_batch([]) == 0
    assert fit_batch(["a"] * (EMBED_BATCH_MAX_TEXTS + 5)) == EMBED_BATCH_MAX_TEXTS
    room = EMBED_BATCH_MAX_BYTES - _prefixed("")
    assert fit_batch(["x" * room]) == 1
    assert fit_batch(["x" * room, "y"]) == 1
    half = EMBED_BATCH_MAX_BYTES // 2 - len(DOCUMENT_PREFIX)
    assert fit_batch(["x" * half, "x" * half, "z"]) == 2
    # Bytes, not characters: two-byte letters fill the bound at half the length.
    cyrillic = "ж" * (room // 2)
    assert _prefixed(cyrillic) <= EMBED_BATCH_MAX_BYTES
    assert fit_batch([cyrillic, "ж" * 10]) == 1
    assert len(DOCUMENT_PREFIX + cyrillic) + len(DOCUMENT_PREFIX + "ж" * 10) < EMBED_BATCH_MAX_BYTES


def test_fit_batch_always_progresses_on_an_oversized_head() -> None:
    assert fit_batch(["x" * (EMBED_BATCH_MAX_BYTES + 1), "y"]) == 1


def test_embed_documents_empty_input_calls_nothing(temp_cache_path: str) -> None:
    provider, sent = _recording_provider(temp_cache_path)
    assert provider.embed_documents([]) == []
    assert sent == []


@pytest.mark.parametrize("texts", [
    ["t"] * (EMBED_BATCH_MAX_TEXTS + 1),
    ["x" * EMBED_BATCH_MAX_BYTES],
    ["x" * (EMBED_BATCH_MAX_BYTES // 2), "y" * (EMBED_BATCH_MAX_BYTES // 2)],
], ids=["count", "one-oversized-text", "total-bytes"])
def test_embed_documents_refuses_an_oversize_batch_before_any_call(
    temp_cache_path: str, texts
) -> None:
    provider, sent = _recording_provider(temp_cache_path)
    with pytest.raises(ValueError):
        provider.embed_documents(texts)
    assert sent == []


def test_embed_documents_one_call_prefixed_in_order_and_uncached(temp_cache_path: str) -> None:
    provider, sent = _recording_provider(temp_cache_path)
    before = _cache_count(temp_cache_path)
    vectors = provider.embed_documents(["one", "two", "three"])
    assert len(sent) == 1
    assert _sent_texts(sent[0]) == [DOCUMENT_PREFIX + t for t in ("one", "two", "three")]
    # One content per text, never a bare list of strings (the live API reads that
    # as one content of many parts and answers one embedding).
    assert all(len(content.parts) == 1 for content in sent[0])
    assert vectors == [[0.1, 0.2]] * 3
    assert _cache_count(temp_cache_path) == before
    assert (provider.hits, provider.misses) == (0, 0)
    # The same texts again: still a call, still no cache row (presence is the cache).
    provider.embed_documents(["one", "two", "three"])
    assert len(sent) == 2 and _cache_count(temp_cache_path) == before


@pytest.mark.parametrize("answer", [
    lambda contents: SimpleNamespace(embeddings=None),
    lambda contents: SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1])]),
    lambda contents: SimpleNamespace(
        embeddings=[SimpleNamespace(values=[0.1]), SimpleNamespace(values=None)]),
    lambda contents: SimpleNamespace(
        embeddings=[SimpleNamespace(values=[0.1]), SimpleNamespace(values=[])]),
    lambda contents: SimpleNamespace(),
], ids=["embeddings-none", "short", "values-none", "values-empty", "no-attribute"])
def test_embed_documents_a_malformed_answer_is_an_embedding_error(
    temp_cache_path: str, answer
) -> None:
    provider, sent = _recording_provider(temp_cache_path, answer)
    with pytest.raises(EmbeddingError) as raised:
        provider.embed_documents(["one", "two"])
    assert len(sent) == 1
    # Raised once, never chained to itself, and it classifies as unknown.
    assert not isinstance(raised.value.__cause__, EmbeddingError)
    assert classify_provider_error(raised.value) == UNKNOWN


def test_embed_documents_chains_the_sdk_error_for_the_classifier(temp_cache_path: str) -> None:
    def refuse(contents):
        raise _SdkError("quota exhausted", code=429, status="RESOURCE_EXHAUSTED")

    provider, sent = _recording_provider(temp_cache_path, refuse)
    with pytest.raises(EmbeddingError) as raised:
        provider.embed_documents(["one"])
    assert isinstance(raised.value.__cause__, _SdkError)
    assert classify_provider_error(raised.value) == QUOTA
    assert (provider.hits, provider.misses) == (0, 0)
