"""The rejection index's vector half, offline: its name, its client, its protocol, its double.

Recall-confidence phase 4a. The rejection index is a sibling collection holding one
point per rejected item, named after the decision collection
(``config.rejection_collection_name``) and spoken to by ``QdrantRejectionIndex``, a
client apart from the decision index's. Every client row patches
``mitos.vector_store.requests`` (the ``tests/test_vector_store.py`` idiom) except C6b,
which provokes a real overrun against a silent socket. The structural row (S1) holds
every class defining a rejection-only method to the whole ``RejectionIndex`` protocol,
and S2 runs the shared contract scenario over ``FakeRejectionIndex`` — the same
function the live lane runs over Qdrant.
"""

import ast
import inspect
import os
import re
import socket
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import requests

from _rejection_index_double import (
    OWNER_A,
    FakeRejectionIndex,
    axis_vector,
    exercise_rejection_index_contract,
    scenario_points,
)
from mitos.config import (
    REJECTION_COLLECTION_SUFFIX,
    MitosConfig,
    default_collection_name,
    rejection_collection_name,
)
from mitos.errors import (
    CollectionMissingError,
    VectorStoreError,
    VectorStoreUnreachableError,
)
from mitos.models import EMBEDDING_DIM
from mitos.protocols import RejectionIndex
from mitos.vector_store import (
    QdrantRejectionIndex,
    QdrantVectorStore,
    RejectionHit,
    RejectionPoint,
    hash_to_uuid,
    scroll_point_ids,
)
from test_vector_store_protocol import (
    _NO_DEFAULT,
    _ast_params,
    _classes,
    _methods,
    _tree_files,
)

_URL = "http://localhost:6333"
_COLLECTION = "mitos-ws-0123abcd-rejections"
_POINTS_URL = f"{_URL}/collections/{_COLLECTION}/points"
_COMPLETED = {"result": {"operation_id": 7, "status": "completed"}, "status": "ok"}
_ACKNOWLEDGED = {"result": {"operation_id": 7, "status": "acknowledged"}, "status": "ok"}
_CREATED = {"result": True, "status": "ok"}
_MODEL = "gemini-embedding-test"


def _resp(status: int, body: object = None) -> MagicMock:
    """A fake ``requests`` response; an exception body makes ``.json()`` raise."""
    resp = MagicMock()
    resp.status_code = status
    resp.text = "fake body"
    if isinstance(body, Exception):
        resp.json.side_effect = body
    else:
        resp.json.return_value = body
    return resp


def _index() -> QdrantRejectionIndex:
    return QdrantRejectionIndex(_URL, _COLLECTION)


def _points(n: int, owner: str = OWNER_A) -> List[RejectionPoint]:
    return [
        RejectionPoint(
            point_id=f"00000000-0000-0000-0000-{i:012x}", owner_id=owner, vector=[0.1]
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# N — the name
# --------------------------------------------------------------------------- #


def test_n1_the_sibling_is_the_decision_name_plus_the_suffix(tmp_path) -> None:
    import tempfile

    assert REJECTION_COLLECTION_SUFFIX == "-rejections"
    assert rejection_collection_name("mitos-ws-0123abcd") == "mitos-ws-0123abcd-rejections"
    workspace = tempfile.mkdtemp()
    try:
        config = MitosConfig(workspace)
        assert config.rejection_collection == config.qdrant_collection + "-rejections"
        # The leaked-collection sweep deletes names starting ``mitos-tmp``; the sibling
        # of a temp workspace is among them.
        assert config.qdrant_collection.startswith("mitos-tmp")
        assert config.rejection_collection.startswith("mitos-tmp")
    finally:
        os.rmdir(workspace)


@pytest.mark.parametrize("path", [
    "/home/someone/проект",
    "/",
    "/srv/deadbeef",
    "/srv/mitos-0123abcd",
    "/srv/rejections",
    "/srv/a-rejections",
    "/srv/ws-rejections-0123abcd",
])
def test_n2_no_sibling_can_equal_any_decision_name(path: str) -> None:
    decision = default_collection_name(path)
    sibling = rejection_collection_name(decision)
    assert re.search(r"-[0-9a-f]{8}$", decision), decision
    assert not re.search(r"-[0-9a-f]{8}$", sibling), sibling
    assert sibling.startswith(decision)


def test_n3_the_sibling_name_is_derived_never_persisted_or_overridden(tmp_path) -> None:
    root = tmp_path / "ws"
    (root / ".mitos").mkdir(parents=True)
    (root / ".mitos" / "config.toml").write_text(
        'rejection_collection = "somewhere-else"\n', encoding="utf-8"
    )
    config = MitosConfig(str(root), warn_unknown_keys=False)
    assert config.rejection_collection == rejection_collection_name(config.qdrant_collection)
    assert "rejection_collection" not in config.to_dict()
    assert config.rejection_collection not in config.to_dict().values()


# --------------------------------------------------------------------------- #
# C — the client
# --------------------------------------------------------------------------- #


@patch("mitos.vector_store.requests.get")
@patch("mitos.vector_store.requests.put")
@patch("mitos.vector_store.requests.post")
def test_c1_construction_dispatches_nothing(post, put, get) -> None:
    _index()
    assert post.call_count == put.call_count == get.call_count == 0


@patch("mitos.vector_store.requests.put")
def test_c2_upsert_sends_ids_as_given_the_payload_exactly_and_waits(put) -> None:
    put.return_value = _resp(200, _COMPLETED)
    points = _points(130)

    _index().upsert_points(points, model=_MODEL, may_create=False)

    assert put.call_count == 3
    sizes = [len(c.kwargs["json"]["points"]) for c in put.call_args_list]
    assert sizes == [64, 64, 2]
    sent = [p for c in put.call_args_list for p in c.kwargs["json"]["points"]]
    assert [p["id"] for p in sent] == [p.point_id for p in points]  # never re-hashed
    assert all(p["payload"] == {"owner_id": OWNER_A, "model": _MODEL} for p in sent)
    for call in put.call_args_list:
        assert call.args[0] == f"{_POINTS_URL}?wait=true"
        assert call.kwargs["timeout"] == 5


@patch("mitos.vector_store.requests.put")
def test_c2_an_empty_upsert_sends_nothing(put) -> None:
    _index().upsert_points([], model=_MODEL, may_create=True)
    assert put.call_count == 0


@pytest.mark.parametrize("answer", [
    _resp(200, _ACKNOWLEDGED),
    _resp(200, {"result": {"operation_id": 7}, "status": "ok"}),
    _resp(200, {"result": None}),
    _resp(200, {"status": "ok"}),
    _resp(200, ValueError("not json")),
    _resp(500),
    _resp(400),
], ids=["acknowledged", "no-status", "null-result", "no-result", "not-json", "500", "400"])
@patch("mitos.vector_store.requests.put")
def test_c3_upsert_accepts_only_completed(put, answer) -> None:
    put.return_value = answer
    with pytest.raises(VectorStoreError) as excinfo:
        _index().upsert_points(_points(1), model=_MODEL, may_create=False)
    assert type(excinfo.value) is VectorStoreError


@patch("mitos.vector_store.requests.put")
def test_c3_upsert_types_a_transport_fault_as_unreachable(put) -> None:
    put.side_effect = requests.ConnectionError("refused")
    with pytest.raises(VectorStoreUnreachableError):
        _index().upsert_points(_points(1), model=_MODEL, may_create=True)


@patch("mitos.vector_store.requests.put")
def test_c3_a_chunk_failing_after_one_landed_raises(put) -> None:
    put.side_effect = [_resp(200, _COMPLETED), _resp(500)]
    with pytest.raises(VectorStoreError):
        _index().upsert_points(_points(70), model=_MODEL, may_create=False)
    assert put.call_count == 2


def _create_sequence(*middle: Any) -> List[Any]:
    """The PUTs a licensed 404 makes: points (404), then ``middle``, then a retry."""
    return [_resp(404), *middle]


@patch("mitos.vector_store.requests.put")
def test_c4_a_licensed_404_creates_the_collection_and_its_owner_index_then_retries(put) -> None:
    put.side_effect = _create_sequence(
        _resp(200, _CREATED), _resp(200, _COMPLETED), _resp(200, _COMPLETED)
    )
    _index().upsert_points(_points(2), model=_MODEL, may_create=True)

    urls = [c.args[0] for c in put.call_args_list]
    assert urls == [
        f"{_POINTS_URL}?wait=true",
        f"{_URL}/collections/{_COLLECTION}",
        f"{_URL}/collections/{_COLLECTION}/index?wait=true",
        f"{_POINTS_URL}?wait=true",
    ]
    assert put.call_args_list[1].kwargs["json"] == {
        "vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"}
    }
    assert put.call_args_list[2].kwargs["json"] == {
        "field_name": "owner_id", "field_schema": "keyword"
    }


@patch("mitos.vector_store.requests.put")
def test_c4_a_second_404_after_creating_raises(put) -> None:
    put.side_effect = _create_sequence(
        _resp(200, _CREATED), _resp(200, _COMPLETED), _resp(404)
    )
    with pytest.raises(CollectionMissingError):
        _index().upsert_points(_points(1), model=_MODEL, may_create=True)


@patch("mitos.vector_store.requests.put")
def test_c4_an_unlicensed_404_raises_and_sends_nothing_create_shaped(put) -> None:
    put.return_value = _resp(404)
    with pytest.raises(CollectionMissingError) as excinfo:
        _index().upsert_points(_points(1), model=_MODEL, may_create=False)
    assert excinfo.value.collection == _COLLECTION
    assert put.call_count == 1
    assert put.call_args.args[0] == f"{_POINTS_URL}?wait=true"


@pytest.mark.parametrize("index_answer, error", [
    (_resp(200, _ACKNOWLEDGED), VectorStoreError),
    (_resp(500), VectorStoreError),
    (_resp(200, ValueError("not json")), VectorStoreError),
    (_resp(200, {"result": None}), VectorStoreError),
    (requests.ConnectionError("refused"), VectorStoreUnreachableError),
], ids=["acknowledged", "500", "not-json", "null-result", "transport"])
@patch("mitos.vector_store.requests.put")
def test_c4_a_failing_owner_index_raises(put, index_answer, error) -> None:
    put.side_effect = _create_sequence(_resp(200, _CREATED), index_answer)
    with pytest.raises(error) as excinfo:
        _index().upsert_points(_points(1), model=_MODEL, may_create=True)
    assert type(excinfo.value) is error
    assert put.call_count == 3  # no retry over an unindexed collection


@pytest.mark.parametrize("create_answer, error", [
    (requests.ConnectionError("refused"), VectorStoreUnreachableError),
    (_resp(200, {"result": False}), VectorStoreError),
    (_resp(500), VectorStoreError),
], ids=["transport", "not-true", "500"])
@patch("mitos.vector_store.requests.put")
def test_c4_a_failing_collection_create_raises(put, create_answer, error) -> None:
    put.side_effect = _create_sequence(create_answer)
    with pytest.raises(error) as excinfo:
        _index().upsert_points(_points(1), model=_MODEL, may_create=True)
    assert type(excinfo.value) is error
    assert put.call_count == 2


@patch("mitos.vector_store.requests.put")
def test_c4_a_racing_create_409_reads_as_exists(put) -> None:
    """R-409 (4c, D7): another writer created it between the 404 and this PUT."""
    put.side_effect = _create_sequence(
        _resp(409), _resp(200, _COMPLETED), _resp(200, _COMPLETED)
    )
    _index().upsert_points(_points(1), model=_MODEL, may_create=True)  # no raise

    assert put.call_count == 4
    assert [c.args[0] for c in put.call_args_list] == [
        f"{_POINTS_URL}?wait=true",
        f"{_URL}/collections/{_COLLECTION}",
        f"{_URL}/collections/{_COLLECTION}/index?wait=true",
        f"{_POINTS_URL}?wait=true",
    ]


@patch("mitos.vector_store.requests.put")
def test_c4_a_409_then_a_refused_index_still_raises(put) -> None:
    put.side_effect = _create_sequence(_resp(409), _resp(500))
    with pytest.raises(VectorStoreError):
        _index().upsert_points(_points(1), model=_MODEL, may_create=True)
    assert put.call_count == 3


def test_c4_model_and_may_create_are_required_and_keyword_only() -> None:
    index = _index()
    with pytest.raises(TypeError):
        index.upsert_points(_points(1), model=_MODEL)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        index.upsert_points(_points(1), may_create=True)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        index.upsert_points(_points(1), _MODEL, True)  # type: ignore[misc]


@pytest.mark.parametrize("owner", [None, "", 5])
@patch("mitos.vector_store.requests.put")
def test_c5_a_point_without_an_owner_is_refused_before_any_request(put, owner) -> None:
    points = [*_points(1), RejectionPoint(point_id="x", owner_id=owner, vector=[0.1])]
    with pytest.raises(ValueError):
        _index().upsert_points(points, model=_MODEL, may_create=True)
    assert put.call_count == 0


@pytest.mark.parametrize("model", ["", None])
@patch("mitos.vector_store.requests.put")
def test_c5_an_empty_model_is_refused_before_any_request(put, model) -> None:
    with pytest.raises(ValueError):
        _index().upsert_points(_points(1), model=model, may_create=True)
    assert put.call_count == 0


@patch("mitos.vector_store.requests.post")
def test_c6_search_asks_only_for_the_owner_under_the_callers_budget(post) -> None:
    post.return_value = _resp(200, {"result": [
        {"id": "p-1", "score": 0.91, "payload": {"owner_id": OWNER_A}},
        {"id": "p-2", "score": 0.80, "payload": {}},
        {"id": 7, "score": 0.70, "payload": {"owner_id": 5}},
        {"id": "p-4", "score": 0.60, "payload": {"owner_id": ""}},
        {"id": "p-5", "score": 0.50},
    ]})
    answer = _index().search([0.1, 0.2], 5, budget_seconds=0.25)

    assert post.call_args.args[0] == f"{_POINTS_URL}/search"
    assert post.call_args.kwargs["json"] == {
        "vector": [0.1, 0.2], "limit": 5, "with_payload": ["owner_id"]
    }
    assert post.call_args.kwargs["timeout"] == 0.25
    assert answer.searched is True
    assert answer.hits == [
        RejectionHit("p-1", OWNER_A, 0.91),
        RejectionHit("p-2", None, 0.80),
        RejectionHit("7", None, 0.70),
        RejectionHit("p-4", None, 0.60),
        RejectionHit("p-5", None, 0.50),
    ]


@pytest.mark.parametrize("body", [
    {"result": [{"score": 0.9, "payload": {"owner_id": OWNER_A}}]},
    {"result": [{"id": None, "score": 0.9}]},
    {"result": [{"id": "p", "score": "high"}]},
    {"result": None},
    {"result": {"points": []}},
    ValueError("not json"),
], ids=["no-id", "null-id", "string-score", "null-result", "object-result", "not-json"])
@patch("mitos.vector_store.requests.post")
def test_c6_a_malformed_search_answer_raises(post, body) -> None:
    post.return_value = _resp(200, body)
    with pytest.raises(VectorStoreError) as excinfo:
        _index().search([0.1], 3, budget_seconds=1.0)
    assert type(excinfo.value) is VectorStoreError


@pytest.mark.parametrize("timeout", [
    requests.ReadTimeout("slow"), requests.ConnectTimeout("slow connect"),
], ids=["read", "connect"])
@patch("mitos.vector_store.requests.post")
def test_c6_any_timeout_is_not_searched_never_a_raise(post, timeout) -> None:
    post.side_effect = timeout
    answer = _index().search([0.1], 3, budget_seconds=0.05)
    assert answer.searched is False and answer.hits == []


@pytest.mark.parametrize("spent", [0.0, -0.5, float("nan")], ids=["zero", "negative", "nan"])
@patch("mitos.vector_store.requests.post")
def test_c6_a_spent_budget_sends_nothing_and_is_not_searched(post, spent) -> None:
    # `requests` refuses a non-positive timeout with a raw ValueError; a caller's
    # remaining budget can reach zero, and that must cost matches, not a raise.
    answer = _index().search([0.1], 3, budget_seconds=spent)
    assert answer.searched is False and answer.hits == []
    post.assert_not_called()


@patch("mitos.vector_store.requests.post")
def test_c6_search_types_an_outage_and_an_absent_collection(post) -> None:
    post.side_effect = requests.ConnectionError("refused")
    with pytest.raises(VectorStoreUnreachableError):
        _index().search([0.1], 3, budget_seconds=1.0)
    post.side_effect = None
    post.return_value = _resp(404)
    with pytest.raises(CollectionMissingError):
        _index().search([0.1], 3, budget_seconds=1.0)
    post.return_value = _resp(500)
    with pytest.raises(VectorStoreError) as excinfo:
        _index().search([0.1], 3, budget_seconds=1.0)
    assert type(excinfo.value) is VectorStoreError


def test_c6b_an_overrun_against_a_silent_socket_is_not_searched() -> None:
    """A listener that accepts (the kernel completes the handshake from the backlog)
    and never answers: the read waits past the budget, and the search says so."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    try:
        port = listener.getsockname()[1]
        index = QdrantRejectionIndex(f"http://127.0.0.1:{port}", _COLLECTION)
        start = time.monotonic()
        answer = index.search(axis_vector(0), 3, budget_seconds=0.05)
        elapsed = time.monotonic() - start
    finally:
        listener.close()
    assert answer.searched is False and answer.hits == []
    assert elapsed < 1.0, elapsed


@patch("mitos.vector_store.requests.post")
def test_c7_retrieve_asks_by_ids_and_answers_the_present_subset(post) -> None:
    ids = ["00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002"]
    post.return_value = _resp(200, {"result": [{"id": ids[1]}]})
    assert _index().retrieve_present(ids) == {ids[1]}
    assert post.call_args.args[0] == _POINTS_URL
    assert post.call_args.kwargs["json"] == {
        "ids": ids, "with_payload": False, "with_vector": False
    }
    assert post.call_args.kwargs["timeout"] == 5


@patch("mitos.vector_store.requests.post")
def test_c7_an_empty_retrieve_sends_nothing(post) -> None:
    assert _index().retrieve_present([]) == set()
    assert post.call_count == 0


_MALFORMED_LISTS = [
    {"result": None},
    {"result": {"points": []}},
    {"result": [{"payload": {}}]},
    {"result": [{"id": None}]},
    {"result": ["p-1"]},
    {"status": "ok"},
    ValueError("not json"),
]
_MALFORMED_IDS = ["null-result", "object-result", "no-id", "null-id", "bare-string",
                  "no-result", "not-json"]


@pytest.mark.parametrize("body", _MALFORMED_LISTS, ids=_MALFORMED_IDS)
@patch("mitos.vector_store.requests.post")
def test_c7_a_malformed_retrieve_answer_raises_plain(post, body) -> None:
    post.return_value = _resp(200, body)
    with pytest.raises(VectorStoreError) as excinfo:
        _index().retrieve_present(["p-1"])
    assert type(excinfo.value) is VectorStoreError


@pytest.mark.parametrize("answer, error", [
    (requests.ConnectionError("refused"), VectorStoreUnreachableError),
    (requests.ReadTimeout("slow"), VectorStoreUnreachableError),
    (_resp(404), CollectionMissingError),
    (_resp(500), VectorStoreError),
], ids=["transport", "timeout", "404", "500"])
@patch("mitos.vector_store.requests.post")
def test_c7_retrieve_types_each_failure(post, answer, error) -> None:
    if isinstance(answer, Exception):
        post.side_effect = answer
    else:
        post.return_value = answer
    with pytest.raises(error) as excinfo:
        _index().retrieve_present(["p-1"])
    assert type(excinfo.value) is error


@patch("mitos.vector_store.requests.post")
def test_c8_the_owner_scroll_filters_on_the_owner_and_pages(post) -> None:
    post.side_effect = [
        _resp(200, {"result": {"points": [{"id": "p-1"}], "next_page_offset": "p-2"}}),
        _resp(200, {"result": {"points": [{"id": "p-2"}], "next_page_offset": None}}),
    ]
    assert _index().list_owner_point_ids(OWNER_A, page_size=1) == {"p-1", "p-2"}
    first, second = (c.kwargs["json"] for c in post.call_args_list)
    assert post.call_args_list[0].args[0] == f"{_POINTS_URL}/scroll"
    assert first == {
        "limit": 1, "with_payload": False, "with_vector": False,
        "filter": {"must": [{"key": "owner_id", "match": {"value": OWNER_A}}]},
    }
    assert second["offset"] == "p-2" and second["filter"] == first["filter"]


@patch("mitos.vector_store.requests.post")
def test_c8_the_full_listing_is_unfiltered(post) -> None:
    post.return_value = _resp(200, {"result": {"points": [{"id": 3}], "next_page_offset": None}})
    assert _index().list_point_ids() == {"3"}
    assert "filter" not in post.call_args.kwargs["json"]


@patch("mitos.vector_store.requests.post")
def test_c8_the_owner_map_takes_back_the_owner_only_and_pages(post) -> None:
    post.side_effect = [
        _resp(200, {"result": {"points": [
            {"id": "p-1", "payload": {"owner_id": OWNER_A}},
            {"id": 7, "payload": {"owner_id": ""}},
        ], "next_page_offset": "p-2"}}),
        _resp(200, {"result": {"points": [
            {"id": "p-2", "payload": {"owner_id": OWNER_A}},
            {"id": "p-3"},
            {"id": "p-4", "payload": {"owner_id": 5}},
        ], "next_page_offset": None}}),
    ]
    assert _index().list_point_owners(page_size=2) == {
        "p-1": OWNER_A, "7": None, "p-2": OWNER_A, "p-3": None, "p-4": None,
    }
    first, second = (c.kwargs["json"] for c in post.call_args_list)
    assert first == {"limit": 2, "with_payload": ["owner_id"], "with_vector": False}
    assert second["offset"] == "p-2" and "filter" not in second


def _list_owner(index: QdrantRejectionIndex) -> Any:
    return index.list_owner_point_ids(OWNER_A)


def _list_all(index: QdrantRejectionIndex) -> Any:
    return index.list_point_ids()


def _list_owners(index: QdrantRejectionIndex) -> Any:
    return index.list_point_owners()


@pytest.mark.parametrize("listing", [_list_owner, _list_all, _list_owners],
                         ids=["owner", "all", "owners"])
@pytest.mark.parametrize("answer, error", [
    (requests.ConnectionError("refused"), VectorStoreUnreachableError),
    (_resp(404), CollectionMissingError),
    (_resp(500), VectorStoreError),
    (_resp(200, {"result": None}), VectorStoreError),
    (_resp(200, {"result": {"points": "p-1"}}), VectorStoreError),
    (_resp(200, {"result": {"points": [{"payload": {}}]}}), VectorStoreError),
], ids=["transport", "404", "500", "null-result", "points-not-list", "no-id"])
@patch("mitos.vector_store.requests.post")
def test_c8_the_scrolls_type_each_failure(post, answer, error, listing) -> None:
    if isinstance(answer, Exception):
        post.side_effect = answer
    else:
        post.return_value = answer
    with pytest.raises(error) as excinfo:
        listing(_index())
    assert type(excinfo.value) is error


@patch("mitos.vector_store.requests.post")
def test_c8_the_public_scroll_keeps_its_plain_transport_error(post) -> None:
    post.side_effect = requests.ConnectionError("refused")
    with pytest.raises(VectorStoreError) as excinfo:
        scroll_point_ids(_URL, "mitos-ws-0123abcd")
    assert type(excinfo.value) is VectorStoreError


@patch("mitos.vector_store.requests.post")
def test_c10_the_decision_query_is_unchanged(post) -> None:
    post.return_value = _resp(200, {"result": []})
    QdrantVectorStore(_URL, "mitos-ws-0123abcd").query([0.1], limit=4)
    assert post.call_args.kwargs["timeout"] == 5
    assert post.call_args.kwargs["json"] == {"vector": [0.1], "limit": 4, "with_payload": True}


@patch("mitos.vector_store.requests.put")
def test_c10_the_decision_upsert_still_hashes_its_node_id(put) -> None:
    put.return_value = _resp(200, {"result": {"status": "acknowledged"}})
    QdrantVectorStore(_URL, "mitos-ws-0123abcd").upsert(OWNER_A, [0.1], {}, may_create=False)
    assert put.call_args.kwargs["json"]["points"][0]["id"] == hash_to_uuid(OWNER_A)
    assert put.call_args.kwargs["timeout"] == 5


# --------------------------------------------------------------------------- #
# S — the protocol and the double
# --------------------------------------------------------------------------- #

#: A class defining any of these is a rejection index. Never ``list_point_ids`` or
#: ``delete_points``, which every decision double defines too.
REJECTION_ONLY_METHODS = {"search", "upsert_points", "retrieve_present", "list_owner_point_ids",
                          "list_point_owners"}
#: Every method a rejection index must define or inherit.
REJECTION_INDEX_METHODS = (
    "search", "retrieve_present", "list_owner_point_ids", "list_point_ids",
    "list_point_owners", "upsert_points", "delete_points",
)
#: Found at 4a: 3 (the protocol, the client, the shared double). A floor.
MIN_REJECTION_INDEX_CLASSES = 3


def _reference(method: str) -> List[Any]:
    sig = inspect.signature(getattr(RejectionIndex, method))
    return [(p.name, p.kind.name,
             _NO_DEFAULT if p.default is inspect.Parameter.empty else p.default)
            for p in sig.parameters.values()]


def rejection_index_faults(files: List[Path], universe: List[Path]) -> List[str]:
    """Rejection indexes in ``files`` lacking a protocol method, or with a drifted one.

    The population is every class defining a rejection-only method. Each protocol
    method is resolved on the class or, by base-class name over ``universe``, on an
    ancestor, and its parameters compared by name, kind and default.
    """
    by_name: Dict[str, List[ast.ClassDef]] = {}
    for _path, cls in _classes(universe):
        by_name.setdefault(cls.name, []).append(cls)

    def resolve(cls: ast.ClassDef, method: str, seen: frozenset) -> Optional[ast.FunctionDef]:
        if method in _methods(cls):
            return _methods(cls)[method]
        for base in cls.bases:
            name = ast.unparse(base).split(".")[-1]
            for candidate in by_name.get(name, []):
                if name not in seen:
                    found = resolve(candidate, method, seen | {name})
                    if found is not None:
                        return found
        return None

    faults = []
    for path, cls in _classes(files):
        if not REJECTION_ONLY_METHODS & set(_methods(cls)):
            continue
        for method in REJECTION_INDEX_METHODS:
            fn = resolve(cls, method, frozenset({cls.name}))
            if fn is None:
                faults.append(f"{path.name}::{cls.name} lacks {method}")
            elif _ast_params(fn) != _reference(method):
                faults.append(f"{path.name}::{cls.name}.{method}{_ast_params(fn)}")
    return faults


def test_s1_every_rejection_index_in_the_tree_honours_the_whole_protocol() -> None:
    assert _reference("search") == [
        ("self", "POSITIONAL_OR_KEYWORD", _NO_DEFAULT),
        ("vector", "POSITIONAL_OR_KEYWORD", _NO_DEFAULT),
        ("limit", "POSITIONAL_OR_KEYWORD", _NO_DEFAULT),
        ("budget_seconds", "KEYWORD_ONLY", _NO_DEFAULT),
    ]
    assert _reference("upsert_points") == [
        ("self", "POSITIONAL_OR_KEYWORD", _NO_DEFAULT),
        ("points", "POSITIONAL_OR_KEYWORD", _NO_DEFAULT),
        ("model", "KEYWORD_ONLY", _NO_DEFAULT),
        ("may_create", "KEYWORD_ONLY", _NO_DEFAULT),
    ]
    files = _tree_files()
    population = [
        (p.name, c.name) for p, c in _classes(files)
        if REJECTION_ONLY_METHODS & set(_methods(c))
    ]
    assert len(population) >= MIN_REJECTION_INDEX_CLASSES, population
    assert {
        ("protocols.py", "RejectionIndex"),
        ("vector_store.py", "QdrantRejectionIndex"),
        ("_rejection_index_double.py", "FakeRejectionIndex"),
    } <= set(population)
    assert rejection_index_faults(files, files) == []


def test_s1_the_row_reports_a_planted_drifted_double(tmp_path) -> None:
    """Positive control: a drifted ``search``, a lacking double, and one inheriting
    every method but overriding ``search`` correctly (which conforms)."""
    planted = tmp_path / "test_planted_rejection.py"
    planted.write_text(textwrap.dedent('''
        from _rejection_index_double import FakeRejectionIndex
        def test_x():
            class _Drifted(FakeRejectionIndex):
                def search(self, vector, limit=5, budget_seconds=1.0):
                    return None
            class _Lacking:
                def retrieve_present(self, point_ids):
                    return set()
            class _Fine(FakeRejectionIndex):
                def search(self, vector, limit, *, budget_seconds):
                    return None
    '''), encoding="utf-8")
    universe = [planted, *_tree_files()]
    faults = rejection_index_faults([planted], universe)
    assert [f.split("::")[1].split(".")[0].split(" ")[0] for f in faults if "_Drifted" in f] == ["_Drifted"]
    assert sorted(f.split(" lacks ")[1] for f in faults if "_Lacking" in f) == sorted(
        set(REJECTION_INDEX_METHODS) - {"retrieve_present"}
    )
    assert not [f for f in faults if "_Fine" in f]


def test_s2_the_contract_scenario_over_the_double() -> None:
    index = FakeRejectionIndex()

    def overrun(vector: List[float]) -> Any:
        index.over_budget = True
        try:
            return index.search(vector, 3, budget_seconds=0.001)
        finally:
            index.over_budget = False

    exercise_rejection_index_contract(index, overrun=overrun)
    assert index.created == 1
    assert all(p["payload"]["model"] for p in index.points.values())


def test_s2_the_doubles_knobs_answer_as_the_client_does() -> None:
    points = list(scenario_points().values())
    index = FakeRejectionIndex(present=True)
    index.upsert_points(points, model="m", may_create=False)

    index.unreachable = True
    for call in (
        lambda: index.retrieve_present([points[0].point_id]),
        lambda: index.list_point_ids(),
        lambda: index.list_owner_point_ids(OWNER_A),
        lambda: index.upsert_points(points, model="m", may_create=True),
        lambda: index.delete_points([points[0].point_id]),
        lambda: index.search(axis_vector(0), 3, budget_seconds=1.0),
    ):
        with pytest.raises(VectorStoreUnreachableError):
            call()
    # An empty input sends nothing, so it meets no fault.
    assert index.retrieve_present([]) == set()
    index.delete_points([])
    index.unreachable = False

    index.refuse_upsert = True
    with pytest.raises(VectorStoreError) as excinfo:
        index.upsert_points(points, model="m", may_create=False)
    assert type(excinfo.value) is VectorStoreError
    index.refuse_delete = True
    with pytest.raises(VectorStoreError) as excinfo:
        index.delete_points([points[0].point_id])
    assert type(excinfo.value) is VectorStoreError
    assert len(index.list_point_ids()) == 3

    with pytest.raises(ValueError):
        index.upsert_points(points, model="", may_create=True)
    assert index.calls[-1][0] == "upsert_points"
    assert index.calls[-1][1]["model"] == ""
