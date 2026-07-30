"""Adversarial test suite for the Qdrant REST vector store.

Verifies deterministic UUID mapping, REST endpoints requests, collection
initialization handling, and filtered semantic queries.
"""

import pytest
from unittest.mock import MagicMock, patch
from mitos.models import EMBEDDING_DIM
from mitos.vector_store import QdrantVectorStore, hash_to_uuid, scroll_point_ids
from mitos.errors import CollectionMissingError, VectorStoreError

_URL = "http://localhost:6333"
_COLLECTION = "test_collection"
_CREATE_URL = f"{_URL}/collections/{_COLLECTION}"


def _resp(status: int, body: object = None) -> MagicMock:
    """Builds a fake ``requests`` response with a status and a JSON body."""
    resp = MagicMock()
    resp.status_code = status
    resp.text = "fake body"
    if isinstance(body, Exception):
        resp.json.side_effect = body
    else:
        resp.json.return_value = body
    return resp


def _is_create_shaped(call: object) -> bool:
    """Reports whether a recorded ``requests.put`` call would CREATE a collection.

    Deliberately not "is it a PUT": ``upsert`` is itself a PUT (to
    ``…/collections/{c}/points``), so "no PUT" would be meaningless here. A create
    is the PUT to the **collection root** carrying a ``vectors`` config body, and
    matching that shape keeps the assertion honest if the store ever grows other
    PUTs.
    """
    args, kwargs = call
    url = args[0] if args else kwargs.get("url", "")
    body = kwargs.get("json")
    return url == _CREATE_URL and isinstance(body, dict) and "vectors" in body


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


# --------------------------------------------------------------------------- #
# I10 — creation leaves store construction (W7)
# --------------------------------------------------------------------------- #

@patch("mitos.vector_store.requests.put")
@patch("mitos.vector_store.requests.post")
@patch("mitos.vector_store.requests.get")
def test_construction_dispatches_zero_requests(
    mock_get: MagicMock, mock_post: MagicMock, mock_put: MagicMock
) -> None:
    """Constructing a store contacts Qdrant not at all — the strongest W7 pin.

    This is the offline regression net for the change that would actually happen:
    someone restoring the constructor's existence probe. It runs in bare CI, where
    the live listing-invariance row cannot.
    """
    QdrantVectorStore(_URL, collection_name=_COLLECTION)

    assert mock_get.call_args_list == []
    assert mock_post.call_args_list == []
    assert mock_put.call_args_list == []


@patch("mitos.vector_store.requests.put")
@patch("mitos.vector_store.requests.post")
def test_a_read_dispatches_no_create_shaped_request(
    mock_post: MagicMock, mock_put: MagicMock
) -> None:
    """A semantic read against an absent collection creates nothing (offline half).

    What this proves is precisely "mitos dispatched no create" — it never observes
    a server, so it cannot speak to the collection listing. The live twin
    (``tests/test_collection_absence_live.py``) is what asserts that.
    """
    mock_post.return_value = _resp(404)

    store = QdrantVectorStore(_URL, collection_name=_COLLECTION)
    with pytest.raises(CollectionMissingError):
        store.query([0.1] * EMBEDDING_DIM, limit=1)

    assert not any(_is_create_shaped(c) for c in mock_put.call_args_list)


@patch("mitos.vector_store.requests.put")
@patch("mitos.vector_store.requests.get")
def test_upsert_creates_the_absent_collection_and_retries_once(
    mock_get: MagicMock, mock_put: MagicMock
) -> None:
    """The write is where creation lives now: 404 → create → retry, and it lands."""
    mock_get.return_value = _resp(404)  # _ensure_collection's existence probe
    mock_put.side_effect = [
        _resp(404),  # the first upsert — no collection
        _resp(200),  # the create
        _resp(200),  # the retried upsert
    ]

    store = QdrantVectorStore(_URL, collection_name=_COLLECTION)
    store.upsert("a" * 64, [0.1] * EMBEDDING_DIM, {"slug": "s"})

    creates = [c for c in mock_put.call_args_list if _is_create_shaped(c)]
    assert len(creates) == 1
    assert creates[0].kwargs["json"] == {
        "vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"}
    }
    # Three PUTs total: upsert → create → upsert. The retry is the third.
    assert len(mock_put.call_args_list) == 3


@patch("mitos.vector_store.requests.put")
@patch("mitos.vector_store.requests.get")
def test_upsert_retries_exactly_once_then_raises(
    mock_get: MagicMock, mock_put: MagicMock
) -> None:
    """A second 404 raises rather than looping — a create-then-retry loop against a
    persistently-404ing endpoint would hammer a live service forever."""
    mock_get.return_value = _resp(404)
    mock_put.side_effect = [
        _resp(404),  # the first upsert
        _resp(200),  # the create reports success…
        _resp(404),  # …and the endpoint 404s anyway (a misclassified 404)
    ]

    store = QdrantVectorStore(_URL, collection_name=_COLLECTION)
    with pytest.raises(CollectionMissingError):
        store.upsert("a" * 64, [0.1] * EMBEDDING_DIM, {"slug": "s"})

    # upsert → create → upsert, and then it stops.
    assert len(mock_put.call_args_list) == 3


# --------------------------------------------------------------------------- #
# I10 — missing ≠ unreachable (W8)
# --------------------------------------------------------------------------- #

@patch("mitos.vector_store.requests.post")
def test_query_404_is_a_typed_collection_missing_error(mock_post: MagicMock) -> None:
    """A query 404 names the collection and stays catchable as a VectorStoreError."""
    mock_post.return_value = _resp(404)

    store = QdrantVectorStore(_URL, collection_name=_COLLECTION)
    with pytest.raises(CollectionMissingError) as excinfo:
        store.query([0.1] * EMBEDDING_DIM, limit=1)

    assert excinfo.value.collection == _COLLECTION
    assert _COLLECTION in str(excinfo.value)
    # Subclassing is the whole compatibility story — every shipped
    # `except VectorStoreError` net must keep catching this unchanged.
    assert isinstance(excinfo.value, VectorStoreError)


@patch("mitos.vector_store.requests.post")
def test_scroll_404_is_a_typed_collection_missing_error(mock_post: MagicMock) -> None:
    """The module-level scroll (and so ``list_point_ids``) classifies 404 the same way."""
    mock_post.return_value = _resp(404)

    with pytest.raises(CollectionMissingError):
        scroll_point_ids(_URL, _COLLECTION)

    store = QdrantVectorStore(_URL, collection_name=_COLLECTION)
    with pytest.raises(CollectionMissingError):
        store.list_point_ids()


@patch("mitos.vector_store.requests.post")
def test_query_500_stays_a_plain_vector_store_error(mock_post: MagicMock) -> None:
    """Only 404 means absence — every other non-200 keeps its old class and wording."""
    mock_post.return_value = _resp(500)

    store = QdrantVectorStore(_URL, collection_name=_COLLECTION)
    with pytest.raises(VectorStoreError) as excinfo:
        store.query([0.1] * EMBEDDING_DIM, limit=1)

    assert not isinstance(excinfo.value, CollectionMissingError)
    assert "Qdrant query failed" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# A 200 with a hostile body is a substrate fault, not a healthy empty answer
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("body", [
    pytest.param(ValueError("Expecting value"), id="non-json"),
    pytest.param("just a string", id="not-an-object"),
    pytest.param({"status": "ok"}, id="no-result-key"),
    pytest.param({"result": "not-a-list"}, id="result-not-a-list"),
    pytest.param({"result": ["not-an-object"]}, id="items-not-objects"),
    # The sharpest of the set: a JSON `null` result. Coerced with `or []` it reads
    # as "nothing matched" and a fail-closed consumer certifies a corpus it never
    # actually read — the absent-key guard's own failure mode by another spelling.
    pytest.param({"result": None}, id="result-null"),
])
@patch("mitos.vector_store.requests.post")
def test_query_malformed_200_raises_typed(mock_post: MagicMock, body: object) -> None:
    """A 200 of the wrong shape must reach callers as a VectorStoreError.

    Unguarded it escapes as a raw ``AttributeError``/``ValueError`` — outside
    ``requests.RequestException`` and outside every net keyed on the typed class —
    and a fail-closed consumer would meet it as a generic fatal, or worse, as a
    healthy empty answer. It is the one shape that could slip past a
    status-code-keyed classifier.
    """
    mock_post.return_value = _resp(200, body)

    store = QdrantVectorStore(_URL, collection_name=_COLLECTION)
    with pytest.raises(VectorStoreError):
        store.query([0.1] * EMBEDDING_DIM, limit=1)


@pytest.mark.parametrize("body", [
    pytest.param(ValueError("Expecting value"), id="non-json"),
    pytest.param({"status": "ok"}, id="no-result-key"),
    pytest.param({"result": {"points": "not-a-list"}}, id="points-not-a-list"),
    pytest.param({"result": {"points": [{"no": "id"}]}}, id="point-without-id"),
    pytest.param({"result": None}, id="result-null"),
])
@patch("mitos.vector_store.requests.post")
def test_scroll_malformed_200_raises_typed(mock_post: MagicMock, body: object) -> None:
    """The scroll's body parse is hardened the same way (it feeds status' id-diff)."""
    mock_post.return_value = _resp(200, body)

    with pytest.raises(VectorStoreError):
        scroll_point_ids(_URL, _COLLECTION)


@patch("mitos.vector_store.requests.post")
def test_vector_store_query_handling(mock_post: MagicMock) -> None:
    """Verifies a semantic query request is scope-blind: no scope filter in the
    Qdrant payload, verbatim limit, and raw (unboosted) scores returned."""
    mock_post.return_value = _resp(200, {
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
    })

    store = QdrantVectorStore(_URL, collection_name=_COLLECTION)

    # Recall is scope-blind: query takes no scope filter.
    results = store.query([0.1] * EMBEDDING_DIM, limit=1)

    assert len(results) == 1
    assert results[0]["slug"] == "query-result"
    # Raw semantic score is returned untouched — no scope boost inflating it.
    assert results[0]["score"] == 0.95
    assert results[0]["scope"] == ["core"]

    # Scope must NOT reach the Qdrant payload — no filter, and limit is verbatim.
    args, kwargs = mock_post.call_args
    body = kwargs["json"]
    assert body["limit"] == 1
    assert "filter" not in body
