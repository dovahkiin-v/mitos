"""Layer C's service-touching half: its own collection, and the two recording readers.

Layer C records similarity values through the readers that ship, in both embedding
spaces. The collection holds every Harbor node, the retired ones included, which is
what makes a retired precedent's value measurable at all. Layer B's ``populate_index``
indexes the active set only, and so does the conflict eval that reuses it; when 3b lands
the single expected-set store read, `populate_layer_c` should call it.

Each read goes through a `RecordingVectorStore` wrapped around the collection. The
query form calls MCP ``surface_decisions`` with the workspace components patched to
hand it the proxy; the draft form calls ``conflict.gather_candidates`` with the proxy.
Both readers make exactly one ``query`` call, and the proxy keeps what it returned,
scores included, before the reader drops retired hits into a list without scores.

Hits are joined to nodes by payload slug inside this collection only, guarded at
population by an assertion that every payload slug is unique. When 2a returns point ids
from ``VectorStore.query``, the join moves to ``hash_to_uuid(node_id)``.

This module imports ``_semantic_harness`` and never ``_conflict_harness``, which pulls
``anthropic`` in; Layer C makes no judge call.
"""

import importlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import mitos
from mitos import conflict, sync
from mitos.identity import embedding_text
from mitos.mcp_server import surface_decisions

sys.path.insert(0, os.path.dirname(__file__))
import _calibration_record as R  # noqa: E402
import _semantic_harness as H  # noqa: E402

# The query form's window. The draft form reads the gather's own window
# (``conflict.CONFLICT_OVERFETCH_LIMIT``), which is what the pause sees.
RECORD_LIMIT = 10
READER = "mcp.surface_decisions+conflict.gather_candidates"
# The reader's own live predicate (``surface_decisions`` and ``gather_candidates`` both).
LIVE_STATES = ("active", "drifted")
MITOS_DIR = os.path.dirname(os.path.abspath(mitos.__file__))


class RecordingVectorStore:
    """Wraps a vector store and keeps every ``query`` call's limit and raw result.

    Duck-typed on purpose: the readers call only ``query``, and every other attribute
    passes through. It defines no ``__len__`` or ``__bool__``, so the readers' truthiness
    test on the store sees a plain object.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: List[Tuple[int, List[Dict[str, Any]]]] = []

    def query(self, vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        result = self._inner.query(vector, limit=limit)
        self.calls.append((limit, result))
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@dataclass
class CalibrationIndex:
    """The built Harbor graph, its Layer C collection and what a read needs to label hits."""

    store: Any
    provider: Any
    vstore: Any
    workspace: str
    nodes: List[Dict[str, Any]]
    nodes_by_slug: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    live_by_slug: Dict[str, bool] = field(default_factory=dict)

    @property
    def decision_count(self) -> int:
        return sum(1 for n in self.nodes if n["kind"] == "decision")


@dataclass
class FormRead:
    """One fixture read: the derived result plus what the live rows check it against."""

    result: Dict[str, Any]
    calls: List[Tuple[int, List[Dict[str, Any]]]]
    envelope: Optional[Dict[str, Any]] = None
    candidates: Optional[List[Any]] = None


class DraftReadUnavailable(Exception):
    """``gather_candidates`` returned ``Unavailable``; the caller decides skip or fail."""

    def __init__(self, unavailable: Any) -> None:
        super().__init__(f"Unavailable(reason={unavailable.reason.value}): {unavailable.detail}")
        self.unavailable = unavailable


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

def expected_set(store: Any) -> List[Dict[str, Any]]:
    """Returns every decision node, any state, plus the open questions the graph holds.

    The production expected set (vision §4.5). Decisions come from ``get_all_nodes``,
    which keeps retired ones; open questions from ``get_open_questions``.
    """
    decisions = [n for n in store.get_all_nodes() if n["kind"] == "decision"]
    return decisions + list(store.get_open_questions())


def populate_layer_c(store: Any, provider: Any, vstore: Any) -> CalibrationIndex:
    """Embeds the expected set into the Layer C collection, with production's payload.

    The payload's ``state`` is the literal ``"active"`` on every point, a retired one
    included, because every production write path writes that literal; the reader then
    sees what it would see in production. ``live`` per slug is the readers' own
    predicate: a decision whose ``get_node_state`` is active or drifted. An open question
    is never live, and ``get_node_state`` is never asked about one (it answers
    ``"active"`` for a resolved question, since state comes only from kill edges).

    Args:
        store: The built Harbor ``GraphStore``.
        provider: The embedding provider on the model-keyed golden cache.
        vstore: The ``QdrantVectorStore`` bound to the Layer C collection.

    Returns:
        A `CalibrationIndex` without a workspace (the caller sets it).
    """
    nodes = expected_set(store)
    slugs = [n["slug"] for n in nodes]
    duplicates = sorted({s for s in slugs if slugs.count(s) > 1})
    assert not duplicates, (
        f"payload slugs must be unique for the harness's slug join (until 2a returns "
        f"point ids); duplicated: {duplicates}"
    )
    for node in nodes:
        text = H._derive_embedding_text(node)
        vector = provider.get_embedding(text, is_query=False)
        payload = {
            "slug": node["slug"],
            "scope": node.get("scope") or [],
            "state": "active",
            "kind": node["kind"],
            "embedding_text": text,
        }
        # The raw 64-hex node id; upsert hashes it. A covering write: the collection holds
        # the whole expected set, so it may bring the collection into existence.
        vstore.upsert(node["id"], vector, payload, may_create=True)
    index = CalibrationIndex(store=store, provider=provider, vstore=vstore, workspace="", nodes=nodes)
    for node in nodes:
        index.nodes_by_slug[node["slug"]] = node
        index.live_by_slug[node["slug"]] = (
            node["kind"] == "decision" and store.get_node_state(node["id"]) in LIVE_STATES
        )
    return index


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def label_hits(index: CalibrationIndex, raw: List[Dict[str, Any]], exclude: List[str]) -> List[Dict[str, Any]]:
    """Labels raw hits with kind, liveness and self for `derive_result`."""
    hits = []
    for h in raw:
        node = index.nodes_by_slug.get(h["slug"])
        assert node is not None, f"hit {h['slug']!r} is not a node this collection was built from"
        hits.append({
            "slug": h["slug"],
            "value": h["score"],
            "kind": node["kind"],
            "live": index.live_by_slug[h["slug"]],
            "self": h["slug"] in exclude,
        })
    return hits


def _one_call(proxy: RecordingVectorStore, fixture: Dict[str, Any]) -> List[Dict[str, Any]]:
    assert len(proxy.calls) == 1, (
        f"fixture {fixture['id']!r}: the reader made {len(proxy.calls)} proxied query "
        f"calls; exactly one is the shipped shape"
    )
    return proxy.calls[0][1]


def read_query_form(index: CalibrationIndex, fixture: Dict[str, Any]) -> FormRead:
    """Reads a query-form fixture through MCP ``surface_decisions``.

    The text is embedded once here first, through the same provider: the reader
    swallows an embedding fault into a degraded envelope, so this is where a quota
    fault can propagate. The reader's own embed is then a cache hit with the same vector.
    """
    index.provider.get_embedding(fixture["text"], is_query=True)
    proxy = RecordingVectorStore(index.vstore)
    with patch("mitos.mcp_server.get_workspace_components") as components:
        components.return_value = (index.store, index.provider, proxy)
        envelope = json.loads(
            surface_decisions(fixture["text"], project=index.workspace, limit=RECORD_LIMIT, brief=True)
        )
    raw = _one_call(proxy, fixture)
    hits = label_hits(index, raw, fixture.get("exclude", []))
    result = R.derive_result(fixture, hits, envelope.get("confidence"), None)
    return FormRead(result=result, calls=proxy.calls, envelope=envelope)


def read_draft_form(index: CalibrationIndex, fixture: Dict[str, Any]) -> FormRead:
    """Reads a draft-form fixture through ``conflict.gather_candidates`` (the pause's gather).

    Embeds the gather's own text once first, for the same reason as `read_query_form`:
    the gather turns an embedding fault into ``Unavailable``.

    Raises:
        DraftReadUnavailable: When the gather returns ``Unavailable``.
    """
    index.provider.get_embedding(
        embedding_text({"kind": "decision", "axiom": fixture["text"]}), is_query=False
    )
    proxy = RecordingVectorStore(index.vstore)
    candidates = conflict.gather_candidates(
        fixture["text"], embed_provider=index.provider, vector_store=proxy, store=index.store
    )
    if isinstance(candidates, conflict.Unavailable):
        raise DraftReadUnavailable(candidates)
    raw = _one_call(proxy, fixture)
    hits = label_hits(index, raw, fixture.get("exclude", []))
    result = R.derive_result(fixture, hits, None, sync._NEIGHBOR_REVIEW_THRESHOLD)
    return FormRead(result=result, calls=proxy.calls, candidates=candidates)


def read_fixture(index: CalibrationIndex, fixture: Dict[str, Any]) -> FormRead:
    """Reads one fixture through the reader its form names."""
    if fixture["form"] == "query":
        return read_query_form(index, fixture)
    return read_draft_form(index, fixture)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def constants_in_force() -> Dict[str, Any]:
    """Returns the runtime value of every name in the threshold population: the "before"."""
    out: Dict[str, Any] = {}
    for dotted in R.threshold_population(MITOS_DIR):
        module, _, name = dotted.rpartition(".")
        out[dotted] = getattr(importlib.import_module(module), name)
    return out


def run_provenance(provider: Any) -> Dict[str, Any]:
    """Extends ``_semantic_harness.provenance()`` for a Layer C run.

    The model is the provider's own, so one value flows to the cache path, the provider
    and the record. The three judge keys are dropped: Layer C makes no judge call.
    """
    prov = H.provenance()
    for key in ("judgment_model", "judgment_model_id", "prompt_version"):
        prov.pop(key, None)
    prov["embedding_model"] = provider.model_id
    prov["reader"] = READER
    prov["record_limit"] = {"query": RECORD_LIMIT, "draft": conflict.CONFLICT_OVERFETCH_LIMIT}
    prov["constants_in_force"] = constants_in_force()
    return prov


def build_run(index: CalibrationIndex, results: List[Dict[str, Any]], corpus: str = "harbor") -> Dict[str, Any]:
    """Assembles a §3.4 run from fixture results (``run_id`` is minted by ``append_run``)."""
    return {
        "corpus": corpus,
        "decision_count": index.decision_count,
        "node_count": len(index.nodes),
        "provenance": run_provenance(index.provider),
        "fixtures": results,
        "classes": R.summarize_classes(results),
    }


# ---------------------------------------------------------------------------
# The smoke: Layer B's fixtures in Layer C's shape, unlabelled
# ---------------------------------------------------------------------------

def adapt_layer_b_oracle(oracle: Dict[str, Any], axiom_by_slug: Dict[str, str]) -> List[Dict[str, Any]]:
    """Adapts ``oracle.semantic.json`` into unlabelled Layer C fixtures for the smoke.

    Each ``retrieval`` entry becomes a query-form read (targets = ``expect_relevant``,
    negatives = ``expect_absent``); each ``conflict`` entry becomes a draft-form read of
    its proposal's axiom, target = its candidate, with the proposal excluded as the
    self-hit. Ids come from the position, because two retrieval entries share a query.
    Every class is ``null``, so ``append_run`` refuses a run built from these.
    """
    fixtures = []
    for i, entry in enumerate(oracle["retrieval"]):
        fixtures.append({
            "id": f"layerb-retrieval-{i}", "class": None, "form": "query",
            "text": entry["query"], "targets": list(entry["expect_relevant"]),
            "negatives": list(entry.get("expect_absent", [])), "measure": "target",
            "notes": f"Layer B retrieval[{i}]" + (f" ({entry['probe']})" if entry.get("probe") else ""),
        })
    for i, entry in enumerate(oracle["conflict"]):
        fixtures.append({
            "id": f"layerb-conflict-{i}", "class": None, "form": "draft",
            "text": axiom_by_slug[entry["proposal"]], "targets": [entry["candidate"]],
            "negatives": [], "measure": "target", "exclude": [entry["proposal"]],
            "notes": f"Layer B conflict[{i}] ({entry['kind']})",
        })
    return R.validate_fixtures(fixtures, require_class=False)
