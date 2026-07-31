"""Adversarial test suite for the Qdrant REST vector store.

Verifies deterministic UUID mapping, REST endpoints requests, collection
initialization handling, and filtered semantic queries.
"""

import pytest
from unittest.mock import MagicMock, patch
from mitos.models import EMBEDDING_DIM
from mitos.vector_store import (QdrantVectorStore, hash_to_uuid, list_collection_names,
                                scroll_point_ids)
from mitos.errors import (CollectionMissingError, VectorStoreError,
                          VectorStoreUnreachableError)

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
    store.upsert("a" * 64, [0.1] * EMBEDDING_DIM, {"slug": "s"}, may_create=True)

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
        store.upsert("a" * 64, [0.1] * EMBEDDING_DIM, {"slug": "s"}, may_create=True)

    # upsert → create → upsert, and then it stops.
    assert len(mock_put.call_args_list) == 3


# --------------------------------------------------------------------------- #
# I10 write half — a non-covering write leaves an absent collection absent
# --------------------------------------------------------------------------- #

@patch("mitos.vector_store.requests.put")
@patch("mitos.vector_store.requests.get")
def test_a_non_covering_upsert_dispatches_no_create_and_raises(
    mock_get: MagicMock, mock_put: MagicMock
) -> None:
    """Success criterion 12, the offline regression twin.

    What this proves is precisely *"mitos dispatched no create"* — it never observes a
    server, so it cannot speak to the collection listing (F2's split; the live twin in
    ``tests/test_collection_absence_live.py`` asserts that). It is the net that runs in
    bare CI, and it catches the change that would actually happen: someone widening the
    creation branch back out.

    Assert the absence of a **create-shaped** request rather than "no PUT" — the upsert
    is itself a PUT, so the weaker claim would be vacuous.
    """
    mock_put.return_value = _resp(404)

    store = QdrantVectorStore(_URL, collection_name=_COLLECTION)
    with pytest.raises(CollectionMissingError) as excinfo:
        store.upsert("a" * 64, [0.1] * EMBEDDING_DIM, {"slug": "s"}, may_create=False)

    assert not any(_is_create_shaped(c) for c in mock_put.call_args_list)
    # And no existence probe either — the refusal is decided before any recovery.
    assert mock_get.call_args_list == []
    # Exactly one PUT: the attempted upsert. No create, no retry.
    assert len(mock_put.call_args_list) == 1
    # Worded as the recoverable state it is, never as an outage, and it carries the heal.
    assert excinfo.value.collection == _COLLECTION
    assert "reconcile" in str(excinfo.value)
    # Catchable by every shipped `except VectorStoreError` net, unchanged.
    assert isinstance(excinfo.value, VectorStoreError)


@patch("mitos.vector_store.requests.put")
@patch("mitos.vector_store.requests.get")
def test_may_create_is_read_only_on_a_404(
    mock_get: MagicMock, mock_put: MagicMock
) -> None:
    """A healthy collection ignores the declaration entirely — the blast radius pin.

    Every write on a healthy project declares ``may_create=False`` forever after the
    first, and nothing about it changes: the parameter is consulted at exactly one place,
    the 404 recovery branch. Without this row a reader would reasonably assume the
    narrowing is a behaviour change on the healthy path.
    """
    mock_put.return_value = _resp(200)

    store = QdrantVectorStore(_URL, collection_name=_COLLECTION)
    store.upsert("a" * 64, [0.1] * EMBEDDING_DIM, {"slug": "s"}, may_create=False)

    assert len(mock_put.call_args_list) == 1
    assert mock_get.call_args_list == []


def test_may_create_is_required_and_keyword_only() -> None:
    """D1's contract, asserted rather than commented.

    A default would be the quiet failure: ``False`` loses the covering case for any
    future call site that forgets to declare (it defers forever and reads as working,
    because the outbox merely grows), ``True`` re-arms the hazard. Positional passing is
    refused for the same reason — a fourth positional argument is easy to add by accident
    and impossible to read at the call site.
    """
    store = QdrantVectorStore(_URL, collection_name=_COLLECTION)

    with pytest.raises(TypeError):
        store.upsert("a" * 64, [0.1] * EMBEDDING_DIM, {"slug": "s"})  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        store.upsert("a" * 64, [0.1] * EMBEDDING_DIM, {"slug": "s"}, True)  # type: ignore[misc]


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


# --- the instance-scoped collection listing ---------------------------------

@patch("mitos.vector_store.requests.get")
def test_list_collection_names_reads_the_instance_endpoint_once(mock_get: MagicMock) -> None:
    """One call, addressed to the INSTANCE — never to a collection.

    The bulk shape the global overview needs: whether N projects' collections exist
    is one request plus local set membership, so the call budget scales with distinct
    instances and never with project count.
    """
    mock_get.return_value = _resp(200, {"result": {"collections": [
        {"name": "mitos-a-1234abcd"}, {"name": "mitos-b-5678efgh"}]}})

    names = list_collection_names(_URL, timeout=3.0)

    assert names == {"mitos-a-1234abcd", "mitos-b-5678efgh"}
    assert mock_get.call_count == 1
    assert mock_get.call_args[0][0] == f"{_URL}/collections"
    assert mock_get.call_args[1]["timeout"] == 3.0


@patch("mitos.vector_store.requests.get")
def test_list_collection_names_reads_an_empty_instance_as_empty(mock_get: MagicMock) -> None:
    """Zero collections is a legitimate answer, not a fault — a fresh instance."""
    mock_get.return_value = _resp(200, {"result": {"collections": []}})

    assert list_collection_names(_URL, timeout=3.0) == set()


@pytest.mark.parametrize("status", [404, 401, 500])
@patch("mitos.vector_store.requests.get")
def test_list_collection_names_never_classifies_a_non_200_as_a_missing_collection(
        mock_get: MagicMock, status: int) -> None:
    """404 included, and 404 is the point.

    This endpoint addresses the instance, not a collection, so routing it through
    ``_raise_response_error`` (which maps 404 → ``CollectionMissingError``) would
    file a wrong base URL or a proxy in the way as *your collection is missing* — at
    exactly the boundary a consumer's reachable/listing tri-state depends on.
    """
    mock_get.return_value = _resp(status)

    with pytest.raises(VectorStoreError) as excinfo:
        list_collection_names(_URL, timeout=3.0)
    assert not isinstance(excinfo.value, CollectionMissingError)
    assert not isinstance(excinfo.value, VectorStoreUnreachableError)


@patch("mitos.vector_store.requests.get")
def test_list_collection_names_types_an_unreachable_instance_apart(
        mock_get: MagicMock) -> None:
    """The distinction a caller reporting instance health cannot do without.

    *Answered with something unusable* must not read as *did not answer*, and the
    two are one class today, so a consumer could otherwise only sniff message text.
    A subclass, so every shipped ``except VectorStoreError`` net keeps catching it.
    """
    import requests as _requests
    mock_get.side_effect = _requests.exceptions.ConnectionError("refused")

    with pytest.raises(VectorStoreUnreachableError) as excinfo:
        list_collection_names(_URL, timeout=3.0)
    assert isinstance(excinfo.value, VectorStoreError)


@pytest.mark.parametrize("body", [
    pytest.param(ValueError("Expecting value"), id="non-json"),
    pytest.param("just a string", id="not-an-object"),
    pytest.param({"status": "ok"}, id="no-result-key"),
    pytest.param({"result": "not-an-object"}, id="result-not-an-object"),
    pytest.param({"result": {}}, id="no-collections-key"),
    pytest.param({"result": {"collections": "not-a-list"}}, id="collections-not-a-list"),
    pytest.param({"result": {"collections": ["bare-string"]}}, id="items-not-objects"),
    pytest.param({"result": {"collections": [{"id": 1}]}}, id="item-without-name"),
    pytest.param({"result": {"collections": [{"name": 7}]}}, id="name-not-a-string"),
    # The sharpest of the set, and the reason there is no `or []` here: coerced, a
    # null listing reads as an empty instance, which tells every project on it that
    # its vectors are gone. An empty listing and an unusable one are exactly the two
    # states this function exists to keep apart.
    pytest.param({"result": {"collections": None}}, id="collections-null"),
])
@patch("mitos.vector_store.requests.get")
def test_list_collection_names_malformed_200_raises_typed(
        mock_get: MagicMock, body: object) -> None:
    """A 200 of the wrong shape is a substrate fault, not an empty answer.

    And it is emphatically **not** the unreachable class: the instance answered, so
    a consumer's tri-state must read *up, but its listing is unusable*. Routing a
    shape guard through :class:`VectorStoreUnreachableError` would flip every
    malformed listing to *no answer* with the whole suite green — the same silent
    mis-filing the 404 row above fences from the other side.
    """
    mock_get.return_value = _resp(200, body)

    with pytest.raises(VectorStoreError) as excinfo:
        list_collection_names(_URL, timeout=3.0)
    assert not isinstance(excinfo.value, VectorStoreUnreachableError)
