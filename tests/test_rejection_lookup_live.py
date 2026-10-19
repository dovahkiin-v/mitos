"""Live rows for the field-scoped lookup (recall-confidence 5c).

The offline rows (``tests/test_rejection_lookup.py``) prove the walk, the composer,
every exit and parity over ``FakeRejectionIndex``. These prove the real client and a
real rejection collection agree, against Qdrant on :7333, on both surfaces: a record's
inline write puts its items in the collection and ``query(field="rejected_paths")``
ranks the named item first (L-lookup); a slug-shaped query is ranked over items, never
dereferenced (L-slug); the collection deleted between two reads answers lexically with
the gap (L-gap); and a search that overruns its budget (a listener that never answers)
answers lexically with the overrun named (L-budget). Both read seams run unpatched
except where a row says (this module is in ``LIVE_MODULES``).

No row spends: the provider is a recording double whose vectors are hand-made at the
collection's width, so a query naming an item's text scores it at 1.0. Workspaces live
under ``tempfile.mkdtemp()``, so both collections start ``mitos-tmp`` and
``conftest``'s session sweep reclaims anything a row leaves; each row drops its own
anyway. Serial only.
"""

import functools
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
from mitos import retrieval_walk
from mitos.cli import cmd_init
from mitos.config import MitosConfig
from mitos.models import EMBEDDING_DIM
from mitos.recall import REJECTION_LOOKUP_OVER_BUDGET_REASON
from mitos.rejections import point_items
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
        f"Qdrant is not reachable at {QDRANT_URL} — the field-scoped lookup's live rows "
        "need a real instance. Environmental, not a code defect; start it with "
        "`docker compose up -d` in the mitos repo."
    )

pytestmark = pytest.mark.skipif(not HAS_QDRANT, reason=_SKIP_REASON)

MODEL = "live-lane-recording-model"
F_ABC = "1. Alpha — a.\n2. Bravo — b.\n3. Charlie — c."
F_ED = "1. Echo — e.\n2. Delta — d."


def _vector(text: str) -> List[float]:
    """A deterministic unit-ish vector at the collection's width."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vector = [0.0] * EMBEDDING_DIM
    for i in range(8):
        vector[int.from_bytes(digest[i * 2:i * 2 + 2], "big") % EMBEDDING_DIM] += 1.0
    return vector


class _Provider:
    model_id = MODEL

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        return _vector(text)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [_vector(t) for t in texts]


def _drop(collection: str) -> None:
    requests.delete(f"{QDRANT_URL}/collections/{collection}", timeout=5)


@pytest.fixture
def ws(monkeypatch) -> Iterator[Dict[str, Any]]:
    monkeypatch.setenv("QDRANT_URL", QDRANT_URL)
    workspace = tempfile.mkdtemp()
    with redirect_stdout(io.StringIO()):
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


def _mcp_lookup(ws, text: str) -> Dict[str, Any]:
    return json.loads(mcp_server.query_decisions(text, project=ws["root"],
                                                 field="rejected_paths"))


def _cli_lookup(ws, text: str) -> Dict[str, Any]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.cmd_query(ws["config"], text, as_json=True, field="rejected_paths")
    return json.loads(buf.getvalue())


READS = (_mcp_lookup, _cli_lookup)


def _part(owner: str, field: str, ordinal: int) -> str:
    """The embedded text of an item's first part: a query naming it scores 1.0."""
    for item in point_items(owner, field).values():
        if item.ordinal == ordinal:
            return item.parts[0].text
    raise AssertionError(ordinal)


def test_l_lookup_ranks_the_named_item_first_over_a_real_collection(ws) -> None:
    owner = _record(ws, "cache", F_ABC)
    _record(ws, "queue", F_ED)
    text = _part(owner, F_ABC, 2)
    for read in READS:
        env = read(ws, text)
        first = env["rejection_matches"][0]
        assert (first["slug"], first["ordinal"], first["item"]) == (
            "cache", 2, "2. Bravo — b."), env
        assert first["score"] == pytest.approx(1.0, abs=1e-4)
        assert first["letter"] == "here" and "rejected_paths" in first
        assert env["field"] == "rejected_paths" and "matches" not in env
        assert env["confidence"] == "strong" and "degraded" not in env
        # Every item both decisions state comes back: no floor, five slots.
        assert len(env["rejection_matches"]) == 5


def test_l_slug_a_slug_shaped_query_is_ranked_never_dereferenced(ws) -> None:
    _record(ws, "cache", F_ABC)
    for read in READS:
        env = read(ws, "cache")
        assert "axiom" not in env and env["rejection_matches"], env
        assert all(m["slug"] == "cache" for m in env["rejection_matches"])


def test_l_gap_the_collection_deleted_answers_lexically_with_the_gap(ws) -> None:
    owner = _record(ws, "cache", F_ABC)
    assert _mcp_lookup(ws, _part(owner, F_ABC, 2))["rejection_matches"]
    _drop(ws["config"].rejection_collection)
    for read in READS:
        env = read(ws, "Bravo")
        assert env["degraded"] == "lexical" and env["rejection_section"] == "gap", env
        assert [(m["ordinal"], m["item"]) for m in env["rejection_matches"]] == [
            (2, "2. Bravo — b.")]
        assert "confidence" not in env


def test_l_budget_an_overrun_answers_lexically_with_the_overrun_named(
        ws, monkeypatch) -> None:
    """The real client's timeout path: a listener that accepts and never answers,
    under a 0.05 s budget (a tiny budget against :7333 completes on loopback)."""
    _record(ws, "cache", F_ABC)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    port = listener.getsockname()[1]

    def index(config):
        return QdrantRejectionIndex(f"http://127.0.0.1:{port}", config.rejection_collection)

    walk = functools.partial(retrieval_walk.walk_rejection_lookup, budget_seconds=0.05)
    for module in (mcp_server, cli):
        monkeypatch.setattr(module, "_ranked_rejection_index", index)
        monkeypatch.setattr(module, "walk_rejection_lookup", walk)
    try:
        for read in READS:
            env = read(ws, "Bravo")
            assert env["degraded"] == "lexical", env
            assert env["degraded_reason"] == REJECTION_LOOKUP_OVER_BUDGET_REASON
            assert [m["ordinal"] for m in env["rejection_matches"]] == [2]
    finally:
        listener.close()
