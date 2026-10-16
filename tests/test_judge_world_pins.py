"""The judge's world is pinned (vision §6.2 constraint 2, the 2d part; T5).

The conflict gather decides what the judge sees, and the judge costs money and
attention. Phase 2d moves the gather onto the shared identity walk, so its candidate
set becomes a function of the live graph alone. Two rows keep that change honest:

* **P1 — every judge prompt is byte-identical** over a neighbourhood where 20 raw
  points already held every above-floor live candidate (no rename, no shared slug,
  few retired points). The snapshot ``tests/fixtures/judge_prompts_2d.json`` was
  generated on the pre-2d tree and committed on its own, before the gather changed,
  so this row was green before the change and must stay green after it. It touches
  only APIs that are stable across the change: ``run_conflict_check``,
  ``plan_corpus_check`` / ``execute_corpus_check`` and a recording judge. It never
  reads ``gather_candidates``' return.

The geometry is real cosine ranking (``FakeVectorStore``, the suite's one ranking
double), over a real temp ``GraphStore`` committed through ``commit_parsed_entry``.
Every vector is authored: a draft sits on axis 0 and each decision at a chosen
cosine to it, on an axis of its own, so every score below is derived from the
fixture, never hard-coded into an assertion.

Regenerate the snapshot only when a prompt change is intended
(``MITOS_WRITE_JUDGE_PROMPTS=1``), and review the diff like an oracle diff.
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import pytest

from mitos.check import execute_corpus_check, plan_corpus_check
from mitos.conflict import Unavailable, run_conflict_check
from mitos.identity import embedding_text
from mitos.parser import ParsedEntry
from mitos.store import GraphStore

from _conflict_helpers import _execution
from _stdio_fake_providers import FakeVectorStore

SNAPSHOT = os.path.join(os.path.dirname(__file__), "fixtures", "judge_prompts_2d.json")
WRITE_ENV = "MITOS_WRITE_JUDGE_PROMPTS"
PRODUCTION_ALIAS = "SONNET"
DIMENSIONS = 24


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def _at_cosine(cosine: float, axis: int) -> List[float]:
    """A unit vector at ``cosine`` to axis 0, leaning on its own ``axis``."""
    vector = [0.0] * DIMENSIONS
    vector[0] = cosine
    vector[axis] = math.sqrt(1.0 - cosine * cosine)
    return vector


class _GeoEmbed:
    """Answers each authored axiom with its authored vector; refuses any other text.

    Keyed on ``embedding_text`` — the text the gather embeds — so a fixture axiom and
    the gather's lookup cannot disagree on normalisation.
    """

    def __init__(self) -> None:
        self._vectors: Dict[str, List[float]] = {}

    def add(self, axiom: str, vector: List[float]) -> None:
        self._vectors[embedding_text({"kind": "decision", "axiom": axiom})] = vector

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        return list(self._vectors[text])


class _PromptJudge:
    """Records every rendered prompt and answers each batch "tenable", well-formed."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.prompts: List[Any] = []

    def __call__(self, prompt: Any) -> Any:
        self.prompts.append(prompt)
        verdicts = [(slug, True, 0.2, "Holds together.") for slug in prompt.candidate_slugs]
        return _execution(verdicts, batch_id=f"{self.label}-{len(self.prompts)}",
                          model_alias=PRODUCTION_ALIAS)


class World:
    """A committed graph, its authored vectors, and a private ``FakeVectorStore``."""

    def __init__(self, store: GraphStore, collection: str) -> None:
        self.store = store
        self.embed = _GeoEmbed()
        self.vectors = FakeVectorStore("http://unused", collection)
        self.collection = collection
        self.ids: Dict[str, str] = {}

    def commit(self, slug: str, axiom: str, cosine: float, axis: int, *,
               rejected: str = "Rejected the obvious alternative.",
               scope: Optional[List[str]] = None, **relations: List[str]) -> str:
        entry = ParsedEntry("decision", slug, 1, 10)
        entry.axiom = axiom
        entry.rejected_paths = rejected
        entry.scope = scope if scope is not None else ["core"]
        for name, targets in relations.items():
            setattr(entry, name, targets)
        node_id = self.store.commit_parsed_entry(entry).node_id
        vector = _at_cosine(cosine, axis)
        self.embed.add(axiom, vector)
        self.vectors.upsert(node_id, vector,
                            {"slug": slug, "kind": "decision", "scope": entry.scope},
                            may_create=True)
        self.ids[slug] = node_id
        return node_id

    def drop_points(self, node_ids: List[str]) -> None:
        points = FakeVectorStore._collections[self.collection]
        for node_id in node_ids:
            del points[node_id]


@pytest.fixture
def world(tmp_path, request) -> World:
    collection = f"judge-world-{request.node.name}-{os.getpid()}"
    built = World(GraphStore(str(tmp_path / "graph.sqlite")), collection)
    yield built
    FakeVectorStore._collections.pop(collection, None)


def _draft(slug: str, axiom: str, **relations: List[str]) -> ParsedEntry:
    entry = ParsedEntry("decision", slug, 1, 10)
    entry.axiom = axiom
    entry.rejected_paths = "Rejected keeping it as it was."
    entry.scope = ["core"]
    for name, targets in relations.items():
        setattr(entry, name, targets)
    return entry


# --------------------------------------------------------------------------- #
# P1's neighbourhood: ten points, so 20 raw always held every live candidate
# --------------------------------------------------------------------------- #

DRAFT_ONE = ("draft-cache-policy", "Session data is cached in process memory only.")
DRAFT_TWO = ("draft-cache-ttl", "Cached session entries expire after five minutes.")


def build_p1(world: World) -> Tuple[ParsedEntry, ParsedEntry]:
    """Eight live decisions near two drafts, two retired predecessors, one amends edge.

    Ten points in all, so a 20-point raw window held every one of them before 2d and
    the live-filled gather holds the same live set after it.
    """
    world.commit("cache-in-redis", "Session cache lives in a shared Redis instance.",
                 0.74, 9, rejected="Rejected process-local caches.")
    world.commit("cache-ttl-hour", "Session cache entries live for one hour.",
                 0.71, 10)
    world.commit("cache-is-shared", "The session cache is shared across workers.",
                 0.95, 1, rejected="Rejected per-process caches: they diverge.",
                 supersedes=["cache-in-redis"])
    world.commit("cache-warm-start", "Workers warm the session cache at start.",
                 0.92, 2, scope=["core", "ops"])
    world.commit("cache-eviction-lru", "The session cache evicts least recently used.",
                 0.89, 3)
    world.commit("cache-eviction-size", "Cache eviction triggers at 512 MiB.",
                 0.86, 4, amends=["cache-eviction-lru"])
    world.commit("cache-ttl-ten", "Session cache entries live for ten minutes.",
                 0.83, 5, supersedes=["cache-ttl-hour"])
    world.commit("cache-metrics", "Cache hit rate is exported as a metric.",
                 0.80, 6, scope=["ops"])
    world.commit("cache-encryption", "Cached session data is encrypted at rest.",
                 0.78, 7, rejected="Rejected plaintext caching of tokens.")
    world.commit("logging-format", "Logs are structured JSON lines.", 0.40, 8)
    world.embed.add(DRAFT_ONE[1], _at_cosine(1.0, 11))
    world.embed.add(DRAFT_TWO[1], _at_cosine(0.97, 12))
    return (_draft(*DRAFT_ONE),
            _draft(*DRAFT_TWO, narrows=["cache-ttl-ten"]))


def capture_prompts(world: World, drafts: Tuple[ParsedEntry, ...]) -> Dict[str, Any]:
    """Every prompt the judge receives: one per draft, then the corpus check's batches."""
    captured: List[Dict[str, Any]] = []
    systems = set()

    def keep(path: str, prompt: Any) -> None:
        systems.add(prompt.system)
        captured.append({"path": path, "user": prompt.user,
                         "candidate_slugs": list(prompt.candidate_slugs)})

    for draft in drafts:
        judge = _PromptJudge(draft.slug)
        result = run_conflict_check(draft, embed_provider=world.embed,
                                    vector_store=world.vectors, store=world.store,
                                    judge=judge)
        assert not isinstance(result, Unavailable), result
        assert len(judge.prompts) == 1, f"{draft.slug} reached no judge"
        keep(f"run_conflict_check:{draft.slug}", judge.prompts[0])

    plan = plan_corpus_check(store=world.store, embed_provider=world.embed,
                             vector_store=world.vectors, telemetry=None,
                             model_alias=PRODUCTION_ALIAS)
    assert plan.sweep_degraded is None
    judge = _PromptJudge("corpus")
    result = execute_corpus_check(plan, judge=judge, telemetry=None, store=world.store)
    assert result.judgment_degraded is None
    assert judge.prompts, "the corpus check judged nothing"
    for index, prompt in enumerate(judge.prompts):
        keep(f"corpus:batch-{index}", prompt)

    assert len(systems) == 1, "the system prefix must be one static string"
    return {"system": systems.pop(), "prompts": captured}


def test_p1_every_judge_prompt_is_byte_identical_to_the_pre_2d_snapshot(world) -> None:
    """P1: the sync sensor's, ``check --staged``'s and the corpus check's prompts.

    The sensor and ``check --staged`` share ``run_conflict_check``, so one capture per
    draft covers both. Draft two declares ``narrows`` on a neighbour (the declared
    drop); the corpus sweep crosses an ``amends`` edge (the strong-edge screen) and
    two supersessions (retired points in the fetch).
    """
    captured = capture_prompts(world, build_p1(world))
    if os.environ.get(WRITE_ENV) == "1":
        with open(SNAPSHOT, "w", encoding="utf-8") as handle:
            json.dump(captured, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    with open(SNAPSHOT, encoding="utf-8") as handle:
        snapshot = json.load(handle)
    assert captured["system"] == snapshot["system"]
    assert [p["path"] for p in captured["prompts"]] == [p["path"] for p in snapshot["prompts"]]
    for got, want in zip(captured["prompts"], snapshot["prompts"]):
        assert got == want, f"prompt {want['path']} changed"
