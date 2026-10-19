"""Live rows for the rejection index's client, against a real Qdrant (recall-confidence 4a).

The offline rows (``tests/test_rejection_index.py``) prove what the client sends and how
it reads each answer. These prove the server agrees: the same contract scenario the
shared double passes runs here over ``QdrantRejectionIndex`` (L1); the model each point
records reads back through a standard filtered scroll (L2); creation leaves the owner
index in place (L3); a budget passes when Qdrant is fast (L4); and nothing here touches
the decision collection beside it (L5).

The overrun itself is not provoked here: a tiny budget never overruns on loopback
(measured at plan time, 0 of 20 at four settings), so the scenario's overrun step asks
a silent socket, as offline row C6b does.

Discipline: the workspace lives under ``tempfile.mkdtemp()``, so both its collections
start ``mitos-tmp`` and ``conftest``'s session sweep reclaims anything a row leaves;
each row deletes its own anyway. No key is needed and none is spent — the vectors are
hand-made. Serial only (``-n auto`` lets one worker's sweep delete another's
collection, which degrades to a skip, not a red).
"""

import os
import shutil
import socket
import tempfile
from typing import Any, Iterator, List

import pytest
import requests

from _rejection_index_double import (
    OWNER_A,
    SCENARIO_MODEL,
    axis_vector,
    exercise_rejection_index_contract,
    scenario_points,
)
from mitos.config import MitosConfig
from mitos.vector_store import (
    QdrantRejectionIndex,
    QdrantVectorStore,
    RejectionPoint,
    RejectionSearch,
    hash_to_uuid,
)
from mitos.rejections import item_point_id

from live_helpers import live_tests_disabled

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:7333")


def _qdrant_reachable() -> bool:
    try:
        return requests.get(f"{QDRANT_URL}/collections", timeout=2).ok
    except Exception:
        return False


#: The two skip causes are kept apart; only the unreachable one carries the
#: live-floor's ``not a code defect`` marker (see test_collection_absence_live).
_BRAKE_ON = live_tests_disabled()
HAS_QDRANT = (not _BRAKE_ON) and _qdrant_reachable()

if _BRAKE_ON:
    _SKIP_REASON = (
        "MITOS_NO_LIVE_TESTS is set — the live tier is switched off, so these rows "
        "did not run and nothing pretended to check."
    )
else:
    _SKIP_REASON = (
        f"Qdrant is not reachable at {QDRANT_URL} — the rejection index's live rows "
        "need a real instance. Environmental, not a code defect; start it with "
        "`docker compose up -d` in the mitos repo."
    )

pytestmark = pytest.mark.skipif(not HAS_QDRANT, reason=_SKIP_REASON)


def _drop(collection: str) -> None:
    requests.delete(f"{QDRANT_URL}/collections/{collection}", timeout=5)


@pytest.fixture
def config() -> Iterator[MitosConfig]:
    """A temp workspace's config; both its collections are dropped afterwards."""
    workspace = tempfile.mkdtemp()
    config = MitosConfig(workspace)
    assert config.qdrant_collection.startswith("mitos-tmp")
    assert config.rejection_collection.startswith("mitos-tmp")
    try:
        yield config
    finally:
        _drop(config.rejection_collection)
        _drop(config.qdrant_collection)
        shutil.rmtree(workspace, ignore_errors=True)


def _silent_overrun(collection: str) -> Any:
    """Searches through a listener that never answers, under a 0.05 s budget."""

    def overrun(vector: List[float]) -> RejectionSearch:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        try:
            port = listener.getsockname()[1]
            silent = QdrantRejectionIndex(f"http://127.0.0.1:{port}", collection)
            return silent.search(vector, 3, budget_seconds=0.05)
        finally:
            listener.close()

    return overrun


def _built(config: MitosConfig) -> QdrantRejectionIndex:
    index = QdrantRejectionIndex(QDRANT_URL, config.rejection_collection)
    index.upsert_points(list(scenario_points().values()), model=SCENARIO_MODEL, may_create=True)
    return index


def test_l1_the_contract_scenario_over_qdrant(config: MitosConfig) -> None:
    index = QdrantRejectionIndex(QDRANT_URL, config.rejection_collection)
    exercise_rejection_index_contract(
        index, overrun=_silent_overrun(config.rejection_collection)
    )


def test_l2_the_model_reads_back_from_the_points(config: MitosConfig) -> None:
    index = _built(config)
    other = RejectionPoint(
        point_id=item_point_id(OWNER_A, "Written under another model."),
        owner_id=OWNER_A,
        vector=axis_vector(3),
    )
    index.upsert_points([other], model="another-model", may_create=False)

    def not_written_under(model: str) -> List[str]:
        resp = requests.post(
            f"{QDRANT_URL}/collections/{config.rejection_collection}/points/scroll",
            json={
                "limit": 100, "with_payload": True, "with_vector": False,
                "filter": {"must_not": [{"key": "model", "match": {"value": model}}]},
            },
            timeout=5,
        )
        resp.raise_for_status()
        return [p["id"] for p in resp.json()["result"]["points"]]

    assert not_written_under(SCENARIO_MODEL) == [other.point_id]
    assert sorted(not_written_under("another-model")) == sorted(
        p.point_id for p in scenario_points().values()
    )
    payload = requests.post(
        f"{QDRANT_URL}/collections/{config.rejection_collection}/points",
        json={"ids": [other.point_id], "with_payload": True, "with_vector": False},
        timeout=5,
    ).json()["result"][0]["payload"]
    assert payload == {"owner_id": OWNER_A, "model": "another-model"}


def test_l3_creation_leaves_the_owner_index(config: MitosConfig) -> None:
    _built(config)
    info = requests.get(
        f"{QDRANT_URL}/collections/{config.rejection_collection}", timeout=5
    ).json()["result"]
    assert "owner_id" in info["payload_schema"]
    assert info["payload_schema"]["owner_id"]["data_type"] == "keyword"


def test_l4_a_budget_passes_when_qdrant_is_fast(config: MitosConfig) -> None:
    index = _built(config)
    answer = index.search(axis_vector(0), 3, budget_seconds=0.05)
    assert answer.searched is True
    assert answer.hits and answer.hits[0].owner_id == OWNER_A


def test_l5_the_decision_collection_is_untouched(config: MitosConfig) -> None:
    decisions = QdrantVectorStore(QDRANT_URL, config.qdrant_collection)
    decisions.upsert(OWNER_A, axis_vector(0), {"slug": "d"}, may_create=True)

    def count(collection: str) -> int:
        resp = requests.post(
            f"{QDRANT_URL}/collections/{collection}/points/count",
            json={"exact": True}, timeout=5,
        )
        resp.raise_for_status()
        return resp.json()["result"]["count"]

    before = count(config.qdrant_collection)
    index = QdrantRejectionIndex(QDRANT_URL, config.rejection_collection)
    exercise_rejection_index_contract(
        index, overrun=_silent_overrun(config.rejection_collection)
    )
    index.delete_points(sorted(index.list_point_ids()))
    assert count(config.qdrant_collection) == before == 1
    assert decisions.list_point_ids() == {hash_to_uuid(OWNER_A)}
