"""Live rows for the rejection index's covering verbs (recall-confidence 4d).

The offline rows (``tests/test_rejection_covering.py``) prove ``reconcile``'s build and
collection and ``status``' diff over ``FakeRejectionIndex``. These prove the real
client, the real listing and the real lock agree, against Qdrant on :7333:
``reconcile`` creates the sibling collection under its own stamp and its ids are the
expected set (L-build); stale and orphaned items are deleted by id, and a delete of an
id no longer present completes — what lets the collection re-delete an item the drain
already removed (L-collect); ``status --json`` over the real listing reports the same
keys as the offline rows (L-status); a drain lock held by another process leaves the
collection ``held`` (L-held).

No row spends: the provider is a recording double whose vectors are hand-made at the
collection's width. Workspaces live under ``tempfile.mkdtemp()``, so every collection
starts ``mitos-tmp`` and ``conftest``'s session sweep reclaims anything a row leaves;
each row drops its own anyway. Serial only.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterator, List, Sequence

import pytest
import requests

from mitos import cli
from mitos.cli import cmd_init
from mitos.config import MitosConfig
from mitos.models import EMBEDDING_DIM
from mitos.parser import ParsedEntry
from mitos.rejections import expected_rejection_points, rejection_points
from mitos.store import GraphStore
from mitos.sync import (REJECTION_COLLECT_HELD, REJECTION_COLLECT_JUDGED, MitosSyncManager,
                        rejection_drain_lock_path)
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
        f"Qdrant is not reachable at {QDRANT_URL} — the covering verbs' live rows "
        "need a real instance. Environmental, not a code defect; start it with "
        "`docker compose up -d` in the mitos repo."
    )

pytestmark = pytest.mark.skipif(not HAS_QDRANT, reason=_SKIP_REASON)

MODEL = "live-lane-recording-model"
F_ABC = "1. Alpha — a.\n2. Bravo — b.\n3. Charlie — c."
F_BD = "1. Bravo — b.\n2. Delta — d."
F_ONE = "1. Echo — e."
FOREIGN = "f" * 64


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


def _store(ws: Dict[str, Any]) -> GraphStore:
    return GraphStore(ws["config"].db_path)


def _seat(ws: Dict[str, Any], slug: str, field: str) -> str:
    """Commits a decision and drops its row: what an earlier build left."""
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = f"The {slug} axiom stands on its own words."
    entry.rejected_paths = field
    store = _store(ws)
    owner = store.commit_parsed_entry(entry).node_id
    row = store.get_rejection_backlog_row(owner)
    if row is not None:
        store.clear_rejection_backlog_row(owner, enqueue_seq=row.enqueue_seq)
    return owner


def _index(ws: Dict[str, Any]) -> QdrantRejectionIndex:
    return QdrantRejectionIndex(QDRANT_URL, ws["config"].rejection_collection)


def _expected(ws: Dict[str, Any]) -> Dict[str, str]:
    return expected_rejection_points(_store(ws).get_rejection_index_owner_fields())


def _reconcile(ws: Dict[str, Any]) -> Dict[str, Any]:
    return MitosSyncManager(ws["config"]).reconcile_embeddings()


def test_l_build_reconcile_creates_the_sibling_under_its_stamp(ws) -> None:
    for slug, field in (("one", F_ABC), ("two", F_ONE)):
        _seat(ws, slug, field)

    result = _reconcile(ws)

    index = result["rejection_index"]
    assert index["seed_stamped"] is True and index["collection_present"] is False
    assert index["collected"] == REJECTION_COLLECT_JUDGED
    assert result["rejection_drain"]["stop"] is None
    listed = _index(ws).list_point_owners()
    assert listed == _expected(ws)
    assert index["missing_items"] == len(listed) == 4


def test_l_collect_deletes_stale_and_orphaned_by_id_and_an_absent_id_completes(ws) -> None:
    owner = _seat(ws, "amended", F_ABC)
    _reconcile(ws)
    # The previous build amends (no row), and another workspace's item lands here.
    _seat(ws, "amended", F_BD)
    _index(ws).upsert_points(
        [RejectionPoint(pid, owner, _vector(ref.text))
         for pid, ref in rejection_points(owner, F_BD).items()]
        + [RejectionPoint("99999999-9999-9999-9999-999999999999", FOREIGN, _vector("x"))],
        model=MODEL, may_create=False)
    stale = set(rejection_points(owner, F_ABC)) - set(rejection_points(owner, F_BD))

    result = _reconcile(ws)

    index = result["rejection_index"]
    assert (index["stale_items"], index["orphan_items"]) == (len(stale), 1)
    assert (index["deleted"], index["delete_failed"]) == (len(stale) + 1, 0)
    assert _index(ws).list_point_owners() == _expected(ws)
    # Re-deleting what is already gone completes on this Qdrant (no raise).
    _index(ws).delete_points(sorted(stale))


def test_l_status_reports_the_real_listing_with_the_offline_keys(ws, capsys) -> None:
    owner = _seat(ws, "one", F_ABC)
    _reconcile(ws)
    _index(ws).upsert_points(
        [RejectionPoint("99999999-9999-9999-9999-999999999999", FOREIGN, _vector("x"))],
        model=MODEL, may_create=False)
    _seat(ws, "two", F_ONE)  # missing: one item

    capsys.readouterr()
    cli.cmd_status(ws["root"], as_json=True)
    payload = json.loads(capsys.readouterr().out)

    checks = payload["checks"]
    assert payload["rejection_collection"] == ws["config"].rejection_collection
    assert {key: checks[key] for key in (
        "rejection_collection_exists", "rejection_expected_items",
        "missing_rejection_items", "stale_rejection_items", "orphan_rejection_items",
        "rejection_backlog_past_tolerance")} == {
        "rejection_collection_exists": True,
        "rejection_expected_items": len(rejection_points(owner, F_ABC)) + 1,
        "missing_rejection_items": 1, "stale_rejection_items": 0,
        "orphan_rejection_items": 1, "rejection_backlog_past_tolerance": 0,
    }


_HOLDER = """
import sys, time
from filelock import FileLock
lock = FileLock({path!r}, timeout=0)
lock.acquire()
open({sentinel!r}, "w").close()
time.sleep(120)
"""


def test_l_held_a_lock_held_by_another_process_leaves_the_collection_held(ws) -> None:
    owner = _seat(ws, "one", F_ONE)
    _reconcile(ws)
    _index(ws).upsert_points(
        [RejectionPoint("99999999-9999-9999-9999-999999999999", FOREIGN, _vector("x"))],
        model=MODEL, may_create=False)
    sentinel = os.path.join(ws["root"], "held.sentinel")
    holder = _HOLDER.format(path=rejection_drain_lock_path(ws["config"]), sentinel=sentinel)
    proc = subprocess.Popen([sys.executable, "-c", holder], cwd=ws["root"])
    try:
        deadline = time.monotonic() + 30
        while not os.path.exists(sentinel):
            assert proc.poll() is None, "the holder exited before taking the lock"
            assert time.monotonic() < deadline, "the holder never took the lock"
            time.sleep(0.05)

        result = _reconcile(ws)
    finally:
        proc.kill()
        proc.wait(timeout=10)

    index = result["rejection_index"]
    assert index["collected"] == REJECTION_COLLECT_HELD
    assert index["deleted"] == 0
    assert "99999999-9999-9999-9999-999999999999" in _index(ws).list_point_owners()
    assert set(rejection_points(owner, F_ONE)) <= set(_index(ws).list_point_owners())
