"""Live rows for the rejection backlog's drain (recall-confidence 4c).

The offline rows (``tests/test_rejection_drain.py``) prove the walk over
``FakeRejectionIndex``. These prove the real client and the real lock agree, against
Qdrant on :7333: an amended owner is replaced whole through the owner-filtered scroll
and the delete by ids (L-replace); a drain killed between its write and its delete
leaves old and new items beside each other, and the next drain takes the lock the dead
process held and finishes (L-kill-mid-replacement); two writers racing to create one
collection both land, the 409 read as "exists" (L-409, plus the measured answers that
behaviour rests on).

No row spends: the provider is a recording double whose vectors are hand-made at the
collection's width. Workspaces live under ``tempfile.mkdtemp()``, so every collection
starts ``mitos-tmp`` and ``conftest``'s session sweep reclaims anything a row leaves;
each row drops its own anyway. Serial only.
"""

import hashlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, Iterator, List, Sequence

import pytest
import requests

from mitos.cli import cmd_init
from mitos.config import MitosConfig
from mitos.models import EMBEDDING_DIM
from mitos.parser import ParsedEntry
from mitos.rejections import rejection_points
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager
from mitos.vector_store import QdrantRejectionIndex, RejectionPoint

from conftest import resolve_like_main
from live_helpers import live_tests_disabled

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:7333")


def _qdrant_reachable() -> bool:
    try:
        return requests.get(f"{QDRANT_URL}/collections", timeout=2).ok
    except Exception:
        return False


_BRAKE_ON = live_tests_disabled()
HAS_QDRANT = (not _BRAKE_ON) and _qdrant_reachable()

if _BRAKE_ON:
    _SKIP_REASON = (
        "MITOS_NO_LIVE_TESTS is set — the live tier is switched off, so these rows "
        "did not run and nothing pretended to check."
    )
else:
    _SKIP_REASON = (
        f"Qdrant is not reachable at {QDRANT_URL} — the rejection drain's live rows "
        "need a real instance. Environmental, not a code defect; start it with "
        "`docker compose up -d` in the mitos repo."
    )

pytestmark = pytest.mark.skipif(not HAS_QDRANT, reason=_SKIP_REASON)

MODEL = "live-lane-recording-model"
F_ABC = "1. Alpha — a.\n2. Bravo — b.\n3. Charlie — c."
F_BD = "1. Bravo — b.\n2. Delta — d."


def _vector(text: str) -> List[float]:
    """A deterministic unit-ish vector at the collection's width."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vector = [0.0] * EMBEDDING_DIM
    for i in range(8):
        vector[int.from_bytes(digest[i * 2:i * 2 + 2], "big") % EMBEDDING_DIM] += 1.0
    return vector


class _Provider:
    model_id = MODEL

    def __init__(self) -> None:
        self.document_calls: List[List[str]] = []

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        return _vector(text)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        self.document_calls.append(list(texts))
        return [_vector(t) for t in texts]


def _drop(collection: str) -> None:
    requests.delete(f"{QDRANT_URL}/collections/{collection}", timeout=5)


@pytest.fixture
def ws(monkeypatch) -> Iterator[Dict[str, Any]]:
    monkeypatch.setenv("QDRANT_URL", QDRANT_URL)
    workspace = tempfile.mkdtemp()
    cmd_init(MitosConfig(workspace))
    config = resolve_like_main(os.path.realpath(workspace))
    assert config.rejection_collection.startswith("mitos-tmp")
    provider = _Provider()
    monkeypatch.setattr("mitos.sync.GeminiEmbeddingProvider", lambda *a, **k: provider)
    try:
        yield {"root": os.path.realpath(workspace), "config": config, "provider": provider}
    finally:
        _drop(config.rejection_collection)
        _drop(config.qdrant_collection)
        shutil.rmtree(workspace, ignore_errors=True)


def _commit(ws: Dict[str, Any], slug: str, field: str) -> str:
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = f"The {slug} axiom stands on its own words."
    entry.rejected_paths = field
    return GraphStore(ws["config"].db_path).commit_parsed_entry(entry).node_id


def _index(ws: Dict[str, Any]) -> QdrantRejectionIndex:
    return QdrantRejectionIndex(QDRANT_URL, ws["config"].rejection_collection)


def _seed(ws: Dict[str, Any], owner: str, field: str) -> None:
    """Writes an owner's items through the real client, creating the collection."""
    _index(ws).upsert_points(
        [RejectionPoint(pid, owner, _vector(ref.text))
         for pid, ref in rejection_points(owner, field).items()],
        model=MODEL, may_create=True,
    )


def _row(ws: Dict[str, Any], owner: str):
    return GraphStore(ws["config"].db_path).get_rejection_backlog_row(owner)


def test_l_replace_over_the_real_collection(ws) -> None:
    owner = _commit(ws, "amended", F_ABC)
    _seed(ws, owner, F_ABC)
    other = _commit(ws, "untouched", F_BD)
    _seed(ws, other, F_BD)
    GraphStore(ws["config"].db_path).clear_rejection_backlog_row(
        other, enqueue_seq=_row(ws, other).enqueue_seq)
    _commit(ws, "amended", F_BD)

    outcome = MitosSyncManager(ws["config"]).drain_rejection_backlog()

    assert outcome.stop is None and outcome.owners_cleared == 1
    assert (outcome.items_embedded, outcome.items_skipped_present, outcome.items_removed) == (1, 1, 2)
    assert ws["provider"].document_calls == [["Delta — d."]]
    index = _index(ws)
    assert index.list_owner_point_ids(owner) == set(rejection_points(owner, F_BD))
    assert index.list_owner_point_ids(other) == set(rejection_points(other, F_BD))
    owners = index.list_point_owners()
    assert {pid for pid, o in owners.items() if o == owner} == set(rejection_points(owner, F_BD))
    assert _row(ws, owner) is None


_CHILD = """
import sys, time
sys.path.insert(0, {repo!r})
import mitos.sync as sync
from mitos.config import MitosConfig
from mitos.vector_store import QdrantRejectionIndex

class Provider:
    model_id = {model!r}
    def get_embedding(self, text, is_query=False):
        raise RuntimeError("the decision leg is not this row's")
    def embed_documents(self, texts):
        import hashlib
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = [0.0] * {dim}
            for i in range(8):
                vector[int.from_bytes(digest[i * 2:i * 2 + 2], "big") % {dim}] += 1.0
            out.append(vector)
        return out

class Blocking(QdrantRejectionIndex):
    def upsert_points(self, points, *, model, may_create):
        super().upsert_points(points, model=model, may_create=may_create)
        open({sentinel!r}, "w").close()
        time.sleep(120)

sync.GeminiEmbeddingProvider = lambda *a, **k: Provider()
sync.QdrantRejectionIndex = Blocking
sync.MitosSyncManager(MitosConfig({root!r})).drain_rejection_backlog()
"""


def test_l_kill_mid_replacement_leaves_both_sets_and_the_next_drain_finishes(ws) -> None:
    owner = _commit(ws, "amended", F_ABC)
    _seed(ws, owner, F_ABC)
    _commit(ws, "amended", F_BD)
    seq = _row(ws, owner).enqueue_seq
    sentinel = os.path.join(ws["root"], "upserted.sentinel")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    child = _CHILD.format(repo=repo, model=MODEL, dim=EMBEDDING_DIM,
                          sentinel=sentinel, root=ws["root"])
    env = dict(os.environ, QDRANT_URL=QDRANT_URL)
    proc = subprocess.Popen([sys.executable, "-c", child], cwd=ws["root"], env=env)
    try:
        deadline = time.monotonic() + 60
        while not os.path.exists(sentinel):
            assert proc.poll() is None, "the child drain exited before its write"
            assert time.monotonic() < deadline, "the child drain never wrote"
            time.sleep(0.05)
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    index = _index(ws)
    both = set(rejection_points(owner, F_ABC)) | set(rejection_points(owner, F_BD))
    assert index.list_owner_point_ids(owner) == both  # old beside new, never none
    row = _row(ws, owner)
    assert row is not None and row.enqueue_seq == seq and row.retry_count == 0

    outcome = MitosSyncManager(ws["config"]).drain_rejection_backlog()

    assert outcome.stop is None and outcome.owners_cleared == 1
    assert outcome.calls == 0 and outcome.items_removed == 2
    assert index.list_owner_point_ids(owner) == set(rejection_points(owner, F_BD))
    assert _row(ws, owner) is None


@pytest.fixture
def scratch_collection() -> Iterator[str]:
    name = f"mitos-tmp-409-{uuid.uuid4().hex[:12]}"
    try:
        yield name
    finally:
        _drop(name)


def test_l_409_the_answers_the_race_rests_on(scratch_collection) -> None:
    """Measured on the running Qdrant: a second create answers 409, and an index PUT
    over an index that already exists answers 200 ``completed``."""
    url = f"{QDRANT_URL}/collections/{scratch_collection}"
    body = {"vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"}}
    assert requests.put(url, json=body, timeout=5).status_code == 200
    assert requests.put(url, json=body, timeout=5).status_code == 409
    index = {"field_name": "owner_id", "field_schema": "keyword"}
    for _ in range(2):
        resp = requests.put(f"{url}/index?wait=true", json=index, timeout=5)
        assert resp.status_code == 200 and resp.json()["result"]["status"] == "completed"


def test_l_409_two_racing_creators_both_land(scratch_collection) -> None:
    owners = ["a" * 64, "b" * 64]
    barrier = threading.Barrier(2)
    errors: List[BaseException] = []

    def write(owner: str) -> None:
        client = QdrantRejectionIndex(QDRANT_URL, scratch_collection)
        point = RejectionPoint(str(uuid.uuid4()), owner, _vector(owner))
        barrier.wait()
        try:
            client.upsert_points([point], model=MODEL, may_create=True)
        except BaseException as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(o,)) for o in owners]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    assert errors == []
    listed = QdrantRejectionIndex(QDRANT_URL, scratch_collection).list_point_owners()
    assert sorted(listed.values()) == owners
    info = requests.get(f"{QDRANT_URL}/collections/{scratch_collection}", timeout=5)
    assert info.json()["result"]["payload_schema"]["owner_id"]["data_type"] == "keyword"
