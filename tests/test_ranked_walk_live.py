"""Phase 2c, live against Qdrant: the four ranked surfaces over a collection written by
today's write path (T3).

The offline rows (``test_ranked_walk_surfaces.py``) feed the walk canned hits. What
only a real server shows is that the walk meets a collection shaped exactly like one
written before this vision: points upserted by ``record_decision_entry``'s inline
embed (``MitosSyncManager._best_effort_embed``), carrying the slug they were written
under in their payload. The fixture holds the three cases a slug join gets wrong:

- a **same-slug lineage** — a retired ``database`` and its live successor, which the
  record path refuses to author (a shared slug arises from the corpus, MI-13), so
  the successor is committed through the commit path and embedded through the same
  ``_best_effort_embed`` the record path uses;
- a decision **renamed after its point was written**, while embedding was down — the
  rename waits in the outbox, so the point's payload keeps the old slug until the next
  drain (with the provider up, ``amend_commentary`` re-upserts the point at once);
- an **orphan** — a point whose node the graph does not hold.

Needs Qdrant, not Gemini: the embedder gives each decision's keyword its own axis, so
every stored vector is orthogonal to the others and a read's query vector says
exactly which points are near. Collections are ``mitos-tmp-walk-…``, deleted per test
(and swept by conftest).
"""

import math
import os
import uuid
from typing import Dict, Iterator, List, Tuple

import pytest
import requests

from mitos.cli import cmd_init
from mitos.config import MitosConfig
from mitos.models import EMBEDDING_DIM
from mitos.parser import ParsedEntry
from mitos.retrieval_walk import RANKED_OVERFETCH_BOUND
from mitos.sync import MitosSyncManager
from mitos.vector_store import QdrantVectorStore, hash_to_uuid

from live_helpers import live_tests_disabled
from test_ranked_walk_surfaces import DRIVERS, JSON_DRIVERS, _text_ranks

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
        f"Qdrant is not reachable at {QDRANT_URL} — the ranked-walk rows need a real "
        "instance. Environmental, not a code defect; start it with "
        "`docker compose up -d` in the mitos repo."
    )

pytestmark = pytest.mark.skipif(not HAS_QDRANT, reason=_SKIP_REASON)

# One axis per keyword; an axiom carries exactly one keyword.
_AXES = {"alphakw": 0, "betakw": 1, "gammakw": 2, "deltakw": 3}
_SPARE_AXIS = 10


def _axis_vector(weights: Dict[str, float]) -> List[float]:
    vector = [0.0] * EMBEDDING_DIM
    for keyword, weight in weights.items():
        vector[_AXES[keyword]] = weight
    return vector


class _KeywordEmbed:
    """Document embedder: the keyword in the text picks the axis. Keyless, exact."""

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        hits = {k: 1.0 for k in _AXES if k in text.lower()}
        if not hits:
            vector = [0.0] * EMBEDDING_DIM
            vector[_SPARE_AXIS] = 1.0
            return vector
        return _axis_vector(hits)


class _FixedQuery:
    """The read's embedder: every query asks with one chosen vector."""

    def __init__(self, weights: Dict[str, float]) -> None:
        self._vector = _axis_vector(weights)

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        return list(self._vector)


def _cosine(weights: Dict[str, float], keyword: str) -> float:
    return weights[keyword] / math.sqrt(sum(w * w for w in weights.values()))


def _workspace(tmp_path, monkeypatch, name: str):
    monkeypatch.setenv("QDRANT_URL", QDRANT_URL)
    collection = f"mitos-tmp-walk-{uuid.uuid4().hex[:12]}"
    config = MitosConfig(str(tmp_path / name))
    cmd_init(config)
    manager = MitosSyncManager(config)
    vstore = QdrantVectorStore(QDRANT_URL, collection)
    manager.embed_provider = _KeywordEmbed()  # type: ignore[assignment]
    manager.vector_store = vstore  # type: ignore[assignment]
    return config, manager, vstore, collection


def _record(manager: MitosSyncManager, slug: str, axiom: str, **relations) -> str:
    receipt = manager.record_decision_entry(axiom, "Rejected the obvious alternative.",
                                            ["core"], slug=slug, **relations)
    assert receipt["status"] == "created", receipt
    return manager.store.resolve_slug(slug)[0]


def _commit_successor(manager: MitosSyncManager, slug: str, axiom: str, *,
                      supersedes: str, embed: bool) -> str:
    """A same-slug successor through the commit path a corpus edit takes; embedded
    through the record path's own inline embed when ``embed``."""
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = axiom
    entry.rejected_paths = "Rejected the obvious alternative."
    entry.scope = ["core"]
    entry.supersedes = [supersedes]
    delta = manager.store.commit_parsed_entry(entry)
    if embed:
        assert manager._best_effort_embed(delta, entry) is not None, "successor not embedded"
    return delta.node_id


@pytest.fixture
def written(tmp_path, monkeypatch) -> Iterator[Tuple[Tuple, MitosSyncManager, QdrantVectorStore, Dict[str, str]]]:
    config, manager, vstore, collection = _workspace(tmp_path, monkeypatch, "ws")
    try:
        ids = {
            "retired": _record(manager, "database", "Use the alphakw store."),
            "renamed": _record(manager, "ren-old", "Keep the gammakw policy."),
        }
        ids["live"] = _commit_successor(manager, "database", "Use the betakw store.",
                                        supersedes="database", embed=True)
        # With the provider up, `amend_commentary` re-upserts the point and its payload
        # follows the rename. With it down the point keeps the old slug until the next
        # drain — the stale case a slug join loses — so the rename lands then.
        manager.embed_provider = None  # type: ignore[assignment]
        receipt = manager.amend_commentary("ren-old", {"slug": "ren-current"})
        assert receipt["status"] == "amended", receipt
        manager.embed_provider = _KeywordEmbed()  # type: ignore[assignment]
        orphan_node = uuid.uuid4().hex + uuid.uuid4().hex
        vstore.upsert(orphan_node, _axis_vector({"deltakw": 1.0}),
                      {"slug": "gone", "scope": ["core"], "state": "active",
                       "kind": "decision", "embedding_text": "deltakw"}, may_create=False)
        ids["orphan_point"] = hash_to_uuid(orphan_node)
        yield (config, manager), manager, vstore, ids
    finally:
        requests.delete(f"{QDRANT_URL}/collections/{collection}", timeout=5)


def test_the_collection_is_shaped_as_today_writes_it(written) -> None:
    """The fixture's premise: four points, ids from `hash_to_uuid`, and the renamed
    and same-slug points carrying the payload slugs a slug join would misread."""
    _ws, manager, vstore, ids = written
    hits = vstore.query(_axis_vector({"alphakw": 1.0}), limit=10)
    by_point = {h["point_id"]: h["slug"] for h in hits}
    assert by_point == {
        hash_to_uuid(ids["retired"]): "database",
        hash_to_uuid(ids["live"]): "database",
        hash_to_uuid(ids["renamed"]): "ren-old",
        ids["orphan_point"]: "gone",
    }
    assert manager.store.resolve_slug("ren-old") == []


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_the_live_successor_is_served_once_at_its_own_score(written, driver) -> None:
    ws, _manager, vstore, _ids = written
    weights = {"alphakw": 0.9, "betakw": 0.8}
    call, _verb, _surface, key = DRIVERS[driver]
    resp = call(ws, vstore, embed=_FixedQuery(weights))
    if key:
        top = resp[key][0]
        assert top["slug"] == "database"
        assert top["score"] == pytest.approx(_cosine(weights, "betakw"), abs=1e-4)
        assert [h["slug"] for h in resp[key]].count("database") == 1
    else:
        assert _text_ranks(resp)[0] == "database"
        assert _text_ranks(resp).count("database") == 1
        assert f"(score {_cosine(weights, 'betakw'):.3f})" in resp


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_the_renamed_decision_is_found_under_its_current_slug(written, driver) -> None:
    ws, _manager, vstore, _ids = written
    call, _verb, _surface, key = DRIVERS[driver]
    resp = call(ws, vstore, embed=_FixedQuery({"gammakw": 1.0}))
    ranks = [h["slug"] for h in resp[key]] if key else _text_ranks(resp)
    assert ranks[0] == "ren-current"
    assert "ren-old" not in ranks


@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_the_orphan_is_skipped_and_never_served(written, driver) -> None:
    ws, _manager, vstore, _ids = written
    call, _verb, _surface, key = DRIVERS[driver]
    resp = call(ws, vstore, embed=_FixedQuery({"deltakw": 0.9, "betakw": 0.5}))
    assert [h["slug"] for h in resp[key]][0] == "database"
    assert "gone" not in [h["slug"] for h in resp[key]]
    assert "degraded" not in resp


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_one_vector_call_at_the_bound_ids_only_against_the_server(written, driver) -> None:
    ws, _manager, vstore, _ids = written

    class _Counting:
        def __init__(self, inner):
            self._inner = inner
            self.calls: List[Tuple[int, bool]] = []

        def query(self, vector, limit=5, *, with_payload=True):
            self.calls.append((limit, with_payload))
            return self._inner.query(vector, limit, with_payload=with_payload)

        def delete_points(self, point_ids):
            self._inner.delete_points(point_ids)

    counting = _Counting(vstore)
    call, *_ = DRIVERS[driver]
    call(ws, counting, embed=_FixedQuery({"alphakw": 1.0}))
    assert counting.calls == [(RANKED_OVERFETCH_BOUND, False)]


@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_the_blackout_appears_when_only_the_retired_point_is_near(tmp_path, monkeypatch,
                                                                  driver) -> None:
    """A collection holding only the retired point (its same-slug successor is in the
    graph, not yet embedded): the fetch holds zero live decisions, so the blackout
    names the retired node, its state and its immediate successor's slug."""
    config, manager, vstore, collection = _workspace(tmp_path, monkeypatch, "solo")
    try:
        retired = _record(manager, "database", "Use the alphakw store.")
        _commit_successor(manager, "database", "Use the betakw store.",
                          supersedes="database", embed=False)
        call, _verb, _surface, key = DRIVERS[driver]
        resp = call((config, manager), vstore, embed=_FixedQuery({"alphakw": 1.0}))
        assert resp[key] == []
        # 2f, on intent: the handle gains the retired node's id and its score; the
        # successor is live, so no `live_end` (the stamp already names it).
        (handle,) = resp["all_superseded"]
        assert isinstance(handle.pop("score"), float)
        assert handle == {"slug": "database", "id": retired, "state": "superseded",
                          "superseded_by": ["database"]}
    finally:
        requests.delete(f"{QDRANT_URL}/collections/{collection}", timeout=5)
