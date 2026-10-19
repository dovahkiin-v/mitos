"""Live rows for the ranked reads' rejection section (recall-confidence 5a).

The offline rows (``tests/test_rejection_section.py``) prove the leg, verification and
every state over ``FakeRejectionIndex``. These prove the real client and the real
collection agree, against Qdrant on :7333: a record's inline write puts one owner's
items in a real rejection collection and a ranked read shows the verified match
(L-match); after an amend the old item still sits in the collection and no read serves
it (L-stale); the collection deleted between two reads states the gap (L-gap); and a
search that overruns its budget (a listener that never answers) reads
``not_searched`` with the decision answer whole (L-budget). Both read seams run unpatched (this module is in ``LIVE_MODULES``).

No row spends: the provider is a recording double whose vectors are hand-made at the
collection's width, so a query naming an item's text scores it at 1.0. Workspaces live
under ``tempfile.mkdtemp()``, so both collections start ``mitos-tmp`` and
``conftest``'s session sweep reclaims anything a row leaves; each row drops its own
anyway. Serial only.
"""

import hashlib
import io
import json
import os
import socket
import tempfile
from contextlib import redirect_stdout
from typing import Any, Dict, Iterator, List, Sequence

import pytest
import requests

from mitos import cli, mcp_server
from mitos.cli import cmd_init
from mitos.config import MitosConfig
from mitos.models import EMBEDDING_DIM
from mitos.recall import REJECTION_MATCH_FLOOR, REJECTION_SECTION_BOUND
from mitos.rejections import point_items
from mitos.retrieval_walk import RejectionLeg
from mitos.vector_store import QdrantRejectionIndex

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
        f"Qdrant is not reachable at {QDRANT_URL} — the rejection section's live rows "
        "need a real instance. Environmental, not a code defect; start it with "
        "`docker compose up -d` in the mitos repo."
    )

pytestmark = pytest.mark.skipif(not HAS_QDRANT, reason=_SKIP_REASON)

MODEL = "live-lane-recording-model"
F_ABC = "1. Alpha — a.\n2. Bravo — b.\n3. Charlie — c."
F_BD = "1. Bravo — b.\n2. Delta — d."
SECTION_KEYS = ("rejection_matches", "rejection_section", "rejection_dropped")


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
    factory = lambda *a, **k: provider  # noqa: E731
    monkeypatch.setattr("mitos.sync.GeminiEmbeddingProvider", factory)
    monkeypatch.setattr("mitos.mcp_server.GeminiEmbeddingProvider", factory)
    try:
        yield {"root": os.path.realpath(workspace), "config": config}
    finally:
        _drop(config.rejection_collection)
        _drop(config.qdrant_collection)


def _record(ws, slug: str, field: str) -> str:
    result = json.loads(mcp_server.record_decision(
        f"The {slug} axiom stands on its own words.", field, ["alpha"], slug,
        acknowledge_neighbors=True, project=ws["root"]))
    assert result["status"] == "created", result
    return result["id"]


def _mcp_query(ws, text: str) -> Dict[str, Any]:
    return json.loads(mcp_server.query_decisions(text, project=ws["root"]))


def _cli_query(ws, text: str) -> Dict[str, Any]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.cmd_query(ws["config"], text, as_json=True)
    return json.loads(buf.getvalue())


def _part(owner: str, field: str, ordinal: int) -> str:
    """The embedded text of an item's first part: a query naming it scores 1.0."""
    for item in point_items(owner, field).values():
        if item.ordinal == ordinal:
            return item.parts[0].text
    raise AssertionError(ordinal)


def test_l_match_a_ranked_read_shows_the_verified_item(ws) -> None:
    owner = _record(ws, "cache", F_ABC)
    index = QdrantRejectionIndex(QDRANT_URL, ws["config"].rejection_collection)
    assert index.retrieve_present(list(point_items(owner, F_ABC))) == set(
        point_items(owner, F_ABC))
    text = _part(owner, F_ABC, 2)
    for read in (_mcp_query, _cli_query):
        env = read(ws, text)
        first = env["rejection_matches"][0]
        assert (first["slug"], first["ordinal"], first["item"]) == (
            "cache", 2, "2. Bravo — b."), env
        assert first["score"] >= REJECTION_MATCH_FLOOR
        assert "rejection_section" not in env


def test_l_stale_an_amended_away_item_stays_in_the_collection_and_is_never_served(ws) -> None:
    owner = _record(ws, "cache", F_ABC)
    result = json.loads(mcp_server.amend_commentary("cache", project=ws["root"],
                                                    rejected_paths=F_BD))
    assert result["status"] == "amended", result
    index = QdrantRejectionIndex(QDRANT_URL, ws["config"].rejection_collection)
    alpha = next(pid for pid, item in point_items(owner, F_ABC).items()
                 if item.ordinal == 1)
    assert index.retrieve_present([alpha]) == {alpha}   # the inline write only adds
    for read in (_mcp_query, _cli_query):
        # Alpha's own text scores its stale point at 1.0; it is never served.
        env = read(ws, _part(owner, F_ABC, 1))
        assert "rejection_matches" not in env, env
        # The amend's inline write added Delta; it is served, numbered from now.
        env = read(ws, _part(owner, F_BD, 2))
        assert [(m["ordinal"], m["item"]) for m in env["rejection_matches"]][:1] == [
            (2, "2. Delta — d.")], env


def test_l_gap_the_collection_deleted_between_reads_states_the_gap(ws) -> None:
    owner = _record(ws, "cache", F_ABC)
    text = _part(owner, F_ABC, 2)
    assert _mcp_query(ws, text)["rejection_matches"]
    _drop(ws["config"].rejection_collection)
    for read in (_mcp_query, _cli_query):
        env = read(ws, text)
        assert env["rejection_section"] == "gap" and "rejection_matches" not in env


def test_l_budget_an_overrun_search_is_not_searched_and_the_answer_is_whole(
        ws, monkeypatch) -> None:
    """The real client's timeout path. A tiny budget never overruns on loopback
    (measured: a 1e-6 s socket timeout against :7333 completes), so the leg's index
    is a listener that accepts and never answers, under a 0.05 s budget."""
    owner = _record(ws, "cache", F_ABC)
    text = _part(owner, F_ABC, 2)
    healthy = _mcp_query(ws, text)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    port = listener.getsockname()[1]

    def leg(config):
        return RejectionLeg(index=QdrantRejectionIndex(f"http://127.0.0.1:{port}",
                                                       config.rejection_collection),
                            items_of=point_items, floor=REJECTION_MATCH_FLOOR,
                            bound=REJECTION_SECTION_BOUND, budget_seconds=0.05)

    monkeypatch.setattr(mcp_server, "_rejection_leg", leg)
    monkeypatch.setattr(cli, "_rejection_leg", leg)
    try:
        for read in (_mcp_query, _cli_query):
            env = read(ws, text)
            assert env["rejection_section"] == "not_searched", env
            assert env["matches"] == healthy["matches"]
            assert env["confidence"] == healthy["confidence"]
    finally:
        listener.close()


def test_l_sweep_the_rejection_sibling_is_reclaimed_by_the_session_sweep(ws) -> None:
    """conftest's sweep deletes by prefix: the sibling inherits the workspace's
    ``mitos-tmp`` prefix (its name is the decision collection's plus a suffix)."""
    import conftest

    source = __import__("inspect").getsource(conftest.sweep_leaked_qdrant_collections)
    assert 'name.startswith("mitos-tmp")' in source
    config = ws["config"]
    assert config.rejection_collection.startswith(config.qdrant_collection)
    assert config.qdrant_collection.startswith("mitos-tmp")
