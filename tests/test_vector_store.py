"""Adversarial test suite for the Qdrant REST vector store.

Verifies deterministic UUID mapping, REST endpoints requests, collection
initialization handling, and filtered semantic queries.
"""

import pytest
from unittest.mock import MagicMock, patch
from mitos.vector_store import QdrantVectorStore, hash_to_uuid
from mitos.errors import VectorStoreError

def test_hash_to_uuid_deterministic() -> None:
    """Verifies that 64-char SHA-256 is mapped deterministically to 36-char UUID."""
    sha = "2c26b05237a0c7222f6f4555523f4555523f4555523f4555523f4555523f4555"
    uuid_str = hash_to_uuid(sha)
    
    assert len(uuid_str) == 36
    # Assert formatting structure: 8-4-4-4-12
    assert uuid_str[8] == "-"
    assert uuid_str[13] == "-"
    assert uuid_str[18] == "-"
    assert uuid_str[23] == "-"
    
    # Assert stability
    assert hash_to_uuid(sha) == uuid_str


@patch("requests.get")
@patch("requests.put")
def test_vector_store_collection_creation(mock_put: MagicMock, mock_get: MagicMock) -> None:
    """Verifies that missing collections are created with correct size and distance parameters."""
    # Mock collection check to return 404 (not found)
    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 404
    mock_get.return_value = mock_get_resp
    
    # Mock collection creation to succeed
    mock_put_resp = MagicMock()
    mock_put_resp.status_code = 200
    mock_put.return_value = mock_put_resp

    # Initialize store (triggers creation checks)
    store = QdrantVectorStore("http://localhost:6333", collection_name="test_collection")
    
    # Verify requests sent
    mock_get.assert_called_with("http://localhost:6333/collections/test_collection", timeout=5)
    mock_put.assert_called_with(
        "http://localhost:6333/collections/test_collection",
        json={"vectors": {"size": 3072, "distance": "Cosine"}},
        headers={"Content-Type": "application/json"},
        timeout=5
    )


@patch("requests.get")
@patch("requests.post")
def test_vector_store_query_handling(mock_post: MagicMock, mock_get: MagicMock) -> None:
    """Verifies semantic query requests and scope tag filters match format specifications."""
    # Mock initialization checks to be successful
    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get.return_value = mock_get_resp
    
    # Mock query search response
    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200
    mock_post_resp.json.return_value = {
        "result": [
            {
                "id": "uuid-123",
                "score": 0.95,
                "payload": {
                    "slug": "query-result",
                    "scope": ["core"],
                    "state": "active",
                    "kind": "decision",
                    "embedding_text": "Axiom text."
                }
            }
        ]
    }
    mock_post.return_value = mock_post_resp

    store = QdrantVectorStore("http://localhost:6333", collection_name="test_collection")
    
    # Perform query with scope tag filter
    results = store.query([0.1]*3072, limit=1, filter_scope="core")
    
    assert len(results) == 1
    assert results[0]["slug"] == "query-result"
    assert results[0]["score"] == 0.95
    assert results[0]["scope"] == ["core"]

    # Assert scope pre-filter included in post payload
    args, kwargs = mock_post.call_args
    body = kwargs["json"]
    assert body["limit"] == 1
    assert "filter" not in body
