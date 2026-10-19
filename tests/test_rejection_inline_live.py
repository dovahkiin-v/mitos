"""Live rows for the rejection index's inline write path (recall-confidence 4b).

The offline rows (``tests/test_rejection_inline.py``) prove the leg's decisions over
``FakeRejectionIndex``. These prove the real client agrees, against Qdrant on :7333:
the first record creates ``<collection>-rejections`` with its owner index (V1); a
second record and an amend probe and write through the real client, the dropped item
stays and every payload carries the model (V2); a collection deleted mid-session makes
the next record spend no embedding call and leave its row (V3).

V1–V3 need no key and spend none: the provider is a recording double whose vectors are
hand-made at the collection's width, and the decision index's real client runs beside
it. V4 alone spends: four tiny real embedding calls with the repo's key, because the
one fact no double can carry is the SDK's own reading of a batch — measured 2026-10-17,
a list of strings is one multi-part content that answers ONE embedding, and only one
``Content`` per text answers one each (``embed_documents``' comment). The workspace lives under ``tempfile.mkdtemp()``, so both collections start
``mitos-tmp`` and ``conftest``'s session sweep reclaims anything a row leaves; each row
drops its own anyway. Serial only.
"""

import hashlib
import json
import os
import shutil
import tempfile
from typing import Any, Dict, Iterator, List, Sequence

import pytest
import requests

from mitos import mcp_server
from mitos.cli import cmd_init
from mitos.config import MitosConfig
from mitos.models import EMBEDDING_DIM
from mitos.rejections import rejection_points
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager

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
        f"Qdrant is not reachable at {QDRANT_URL} — the inline rejection leg's live "
        "rows need a real instance. Environmental, not a code defect; start it with "
        "`docker compose up -d` in the mitos repo."
    )

pytestmark = pytest.mark.skipif(not HAS_QDRANT, reason=_SKIP_REASON)

FIELD = "1. Keep the old — it drifted.\n2. Do nothing — it rots.\n3. Buy a tool — it locks us in."
AMENDED = "1. Keep the old — it drifted.\n2. Do nothing — it rots.\n3. Fork it — we would own it."


def _vector(text: str) -> List[float]:
    """A deterministic unit-ish vector at the collection's width."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vector = [0.0] * EMBEDDING_DIM
    for i in range(8):
        vector[int.from_bytes(digest[i * 2:i * 2 + 2], "big") % EMBEDDING_DIM] += 1.0
    return vector


class _Provider:
    model_id = "live-lane-recording-model"

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
    assert config.qdrant_collection.startswith("mitos-tmp")
    assert config.rejection_collection == config.qdrant_collection + "-rejections"
    provider = _Provider()
    monkeypatch.setattr("mitos.sync.GeminiEmbeddingProvider",
                        lambda *a, **k: provider)
    try:
        yield {"root": os.path.realpath(workspace), "config": config, "provider": provider}
    finally:
        _drop(config.rejection_collection)
        _drop(config.qdrant_collection)
        shutil.rmtree(workspace, ignore_errors=True)


def _record(ws: Dict[str, Any], slug: str) -> str:
    result = json.loads(mcp_server.record_decision(
        f"The {slug} axiom stands on its own words.", FIELD, ["alpha"], slug,
        mechanisms=[f"{slug}-mechanism"], acknowledge_neighbors=True, project=ws["root"],
    ))
    assert result["status"] == "created", result
    return result["id"]


def _points(collection: str, ids: List[str]) -> Dict[str, Dict[str, Any]]:
    resp = requests.post(f"{QDRANT_URL}/collections/{collection}/points",
                         json={"ids": ids, "with_payload": True}, timeout=5)
    resp.raise_for_status()
    return {str(p["id"]): p["payload"] for p in resp.json()["result"]}


def test_v1_the_first_record_creates_the_sibling_with_its_owner_index(ws) -> None:
    owner = _record(ws, "first")
    collection = ws["config"].rejection_collection

    info = requests.get(f"{QDRANT_URL}/collections/{collection}", timeout=5)
    assert info.ok, info.text
    schema = info.json()["result"]["payload_schema"]
    assert schema["owner_id"]["data_type"] == "keyword"
    ids = list(rejection_points(owner, FIELD))
    assert set(_points(collection, ids)) == set(ids)
    assert GraphStore(ws["config"].db_path).get_rejection_backlog_row(owner) is None
    assert len(ws["provider"].document_calls) == 1


def test_v2_a_second_record_and_an_amend_go_through_the_real_client(ws) -> None:
    _record(ws, "first")
    owner = _record(ws, "second")
    collection = ws["config"].rejection_collection

    result = json.loads(mcp_server.amend_commentary(
        "second", rejected_paths=AMENDED, project=ws["root"]))
    assert result["status"] == "amended", result

    before, after = rejection_points(owner, FIELD), rejection_points(owner, AMENDED)
    every = sorted(set(before) | set(after))
    stored = _points(collection, every)
    assert set(stored) == set(every)  # the dropped item is still present
    assert {p["model"] for p in stored.values()} == {_Provider.model_id}
    assert {p["owner_id"] for p in stored.values()} == {owner}
    # The amend embedded only its new item, and its row stands.
    added = [ref.text for pid, ref in after.items() if pid not in before]
    assert ws["provider"].document_calls[-1] == added
    assert GraphStore(ws["config"].db_path).get_rejection_backlog_row(owner) is not None


def test_v3_a_collection_deleted_mid_session_spends_nothing_on_the_next_record(ws) -> None:
    _record(ws, "first")
    _record(ws, "second")
    _drop(ws["config"].rejection_collection)
    calls = len(ws["provider"].document_calls)

    owner = _record(ws, "third")

    assert len(ws["provider"].document_calls) == calls
    assert GraphStore(ws["config"].db_path).get_rejection_backlog_row(owner) is not None
    info = requests.get(f"{QDRANT_URL}/collections/{ws['config'].rejection_collection}",
                        timeout=5)
    assert info.status_code == 404


def _repo_gemini_key() -> str:
    from mitos.env import resolve_key

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return resolve_key("GEMINI_API_KEY", repo, "/nonexistent").value or ""


def test_v4_embed_documents_answers_one_vector_per_text_on_the_live_api(tmp_path) -> None:
    from mitos.embeddings import GeminiEmbeddingProvider
    from mitos.models import get_embedding_model_id

    from live_helpers import skip_on_embed_quota

    key = _repo_gemini_key()
    if not key:
        pytest.skip("No GEMINI_API_KEY in the repo .env — V4 needs the real provider. "
                    "Environmental, not a code defect.")
    provider = GeminiEmbeddingProvider(str(tmp_path / "c.sqlite"), api_key=key,
                                       model_id=get_embedding_model_id({}))
    texts = ["Keep the old — it drifted.", "Do nothing — it rots.", "Buy a tool — it locks us in."]
    with skip_on_embed_quota():
        batch = provider.embed_documents(texts)
        single = provider.get_embedding(texts[1], is_query=False)
    assert len(batch) == 3 and all(len(v) == EMBEDDING_DIM for v in batch)

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        return dot / ((sum(x * x for x in a) * sum(y * y for y in b)) ** 0.5)

    # The middle slot is that text's own vector, not a pooled or shifted one.
    assert cosine(batch[1], single) > 0.999
    assert cosine(batch[0], batch[2]) < 0.999
