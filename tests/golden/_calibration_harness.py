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

Hits are joined to nodes by their point id: ``VectorStore.query`` returns each hit's
``point_id`` (2a), and `index_graph` maps ``hash_to_uuid(node_id)`` to each node of the
expected set. The map is built here, independently of ``GraphStore.resolve_point_ids``:
an instrument that resolved through the code it measures could not catch that code being
wrong. Labels stay slugs, so the sealed fixtures and every earlier run stay comparable;
fixtures still name nodes by slug, which is why the slug-uniqueness assertion stays.

A rejection-target fixture is also read against two more throwaway collections, the two
rejection-index shapes ruling 2 compares: one point per rejected item part
(``rejections.rejection_points``) and one point per decision holding its whole
``rejected_paths`` field. No shipped reader searches them before 5a, so `search_points`
reads them raw with ``exact`` search, which makes "every point valued" a declaration —
``VectorStore.query`` has no exact knob, and production has no reason to gain one. Both
shape reads join hits to the graph by point id, never by payload. 5a's switch to the
shipped rejection search retires `search_points`.

This module imports ``_semantic_harness`` and never ``_conflict_harness``, which pulls
``anthropic`` in; Layer C makes no judge call.
"""

import importlib
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import requests

import mitos
from mitos import conflict, sync
from mitos.identity import canonical_core_string_norm, embedding_text
from mitos.mcp_server import surface_decisions
from mitos.rejections import enumerate_rejected_paths, rejection_points
from mitos.vector_store import hash_to_uuid

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

    def query(self, vector: List[float], limit: int = 5, *,
              with_payload: bool = True) -> List[Dict[str, Any]]:
        # Forwards the keyword: from 2c the readers pass ``with_payload=False``, and a
        # wrapper rejecting it would raise inside the reader's catch-all and turn every
        # read into a degraded one.
        result = self._inner.query(vector, limit=limit, with_payload=with_payload)
        self.calls.append((limit, result))
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@dataclass
class ShapeCollections:
    """The two rejection-index shapes, populated, and the maps a read labels hits with.

    ``item_points`` maps a per-item point id to ``(owner slug, ordinal, part index)``;
    ``field_points`` maps a per-decision point id to its owner slug. Both ids are the
    lowercase dashed UUIDs Qdrant returns.
    """

    items_vstore: Any
    fields_vstore: Any
    item_points: Dict[str, Tuple[str, int, int]]
    field_points: Dict[str, str]

    @property
    def counts(self) -> Dict[str, int]:
        return {"per_item": len(self.item_points), "per_decision": len(self.field_points)}


@dataclass
class CalibrationIndex:
    """The built Harbor graph, its Layer C collection and what a read needs to label hits.

    ``shapes`` is set by the live module's shape fixture; an item-measured read without
    it raises rather than recording a result with no items.
    """

    store: Any
    provider: Any
    vstore: Any
    workspace: str
    nodes: List[Dict[str, Any]]
    nodes_by_slug: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    nodes_by_point_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    live_by_slug: Dict[str, bool] = field(default_factory=dict)
    shapes: Optional[ShapeCollections] = None

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
    shape_searches: Optional[Dict[str, int]] = None


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


def index_graph(store: Any, provider: Any = None, vstore: Any = None) -> CalibrationIndex:
    """Builds the index's maps over the expected set; touches no service.

    ``live`` per slug is the readers' own predicate: a decision whose ``get_node_state``
    is active or drifted. An open question is never live, and ``get_node_state`` is never
    asked about one (it answers ``"active"`` for a resolved question, since state comes
    only from kill edges). The offline fixture rows call this with no provider and no
    vector store, to resolve item refs over a built graph.

    Returns:
        A `CalibrationIndex` without a workspace (the caller sets it).
    """
    nodes = expected_set(store)
    slugs = [n["slug"] for n in nodes]
    duplicates = sorted({s for s in slugs if slugs.count(s) > 1})
    assert not duplicates, (
        f"slugs must be unique: fixtures name nodes by slug and labels are slugs "
        f"(hits join by point id); duplicated: {duplicates}"
    )
    index = CalibrationIndex(store=store, provider=provider, vstore=vstore, workspace="", nodes=nodes)
    for node in nodes:
        index.nodes_by_slug[node["slug"]] = node
        index.nodes_by_point_id[hash_to_uuid(node["id"])] = node
        index.live_by_slug[node["slug"]] = (
            node["kind"] == "decision" and store.get_node_state(node["id"]) in LIVE_STATES
        )
    assert len(index.nodes_by_point_id) == len(nodes), "two nodes share a point id"
    return index


def populate_layer_c(store: Any, provider: Any, vstore: Any) -> CalibrationIndex:
    """Embeds the expected set into the Layer C collection, with production's payload.

    The payload's ``state`` is the literal ``"active"`` on every point, a retired one
    included, because every production write path writes that literal; the reader then
    sees what it would see in production. The maps come from `index_graph`.

    Args:
        store: The built Harbor ``GraphStore``.
        provider: The embedding provider on the model-keyed golden cache.
        vstore: The ``QdrantVectorStore`` bound to the Layer C collection.

    Returns:
        A `CalibrationIndex` without a workspace (the caller sets it).
    """
    index = index_graph(store, provider, vstore)
    for node in index.nodes:
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
    return index


def _decisions(index: CalibrationIndex) -> List[Dict[str, Any]]:
    """Every decision in the expected set, retired ones included (§4.6: their items stay)."""
    return [n for n in index.nodes if n["kind"] == "decision"]


def populate_shape_collections(index: CalibrationIndex, items_vstore: Any, fields_vstore: Any) -> ShapeCollections:
    """Embeds both rejection-index shapes over every decision, each text as a document.

    Per item: one point per ``rejections.rejection_points(node id, field)`` entry, its
    part text embedded. Per decision: one point at ``hash_to_uuid(node id)``, holding
    ``canonical_core_string_norm(field)`` embedded. An empty field contributes no point
    in either shape. The payload names the owner's node id and nothing a read takes from
    the graph (ADR ``rejection-point-payload-names-its-owner-and-nothing-a-read-takes-from-the-graph``).

    ``upsert`` maps its id through ``hash_to_uuid``, which slices hex, so an item point is
    written under ``uuid.UUID(pid).hex`` and round-trips to ``pid``; the population row
    checks that against ``list_point_ids()``.

    Args:
        index: A populated `CalibrationIndex` (its provider embeds).
        items_vstore: A ``QdrantVectorStore`` bound to the per-item collection.
        fields_vstore: A ``QdrantVectorStore`` bound to the per-decision collection.

    Returns:
        The `ShapeCollections`.
    """
    item_points: Dict[str, Tuple[str, int, int]] = {}
    field_points: Dict[str, str] = {}
    for node in _decisions(index):
        field_text = node.get("rejected_paths") or ""
        payload = {"owner": node["id"]}
        for pid, ref in rejection_points(node["id"], field_text).items():
            vector = index.provider.get_embedding(ref.text, is_query=False)
            items_vstore.upsert(uuid.UUID(pid).hex, vector, payload, may_create=True)
            item_points[pid] = (node["slug"], ref.ordinal, ref.part_index)
        whole = canonical_core_string_norm(field_text)
        if whole:
            vector = index.provider.get_embedding(whole, is_query=False)
            fields_vstore.upsert(node["id"], vector, payload, may_create=True)
            field_points[hash_to_uuid(node["id"])] = node["slug"]
    return ShapeCollections(items_vstore, fields_vstore, item_points, field_points)


def resolve_item_refs(index: CalibrationIndex, fixture: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Resolves a fixture's ``{owner, option}`` refs to items of the built graph.

    ``option`` names the one item of the owner's field whose identity text (the item
    without its marker) starts with it, both under ``canonical_core_string_norm``. The
    match is case-sensitive. An ordinal is never authored, because a later edit of the
    line can shift it; the prefix survives a renumbering.

    Args:
        index: An index over the built graph (`index_graph` suffices).
        fixture: A validated fixture.

    Returns:
        ``{"targets": [...], "negatives": [...]}``, each ref as ``{owner, option, ordinal,
        part_ids, field_id}``: the item's part point ids and its owner's whole-field id.

    Raises:
        ValueError: Naming the fixture, on an owner that is not a decision of the graph,
            or an option matching no item or several.
    """
    out: Dict[str, List[Dict[str, Any]]] = {"targets": [], "negatives": []}
    for key, role in (("target_items", "targets"), ("negative_items", "negatives")):
        for ref in fixture.get(key, []):
            node = index.nodes_by_slug.get(ref["owner"])
            if node is None or node["kind"] != "decision":
                raise ValueError(f"fixture {fixture['id']!r}: item owner {ref['owner']!r} is not a decision of the graph")
            field_text = node.get("rejected_paths") or ""
            option = canonical_core_string_norm(ref["option"])
            items = enumerate_rejected_paths(field_text).items
            matches = [i for i in items if canonical_core_string_norm(i.identity_text).startswith(option)]
            if len(matches) != 1:
                named = [i.identity_text[:60] for i in matches]
                raise ValueError(
                    f"fixture {fixture['id']!r}: option {ref['option']!r} of {ref['owner']!r} matches "
                    f"{len(matches)} items {named}; it must name exactly one"
                )
            item = matches[0]
            part_ids = [pid for pid, p in rejection_points(node["id"], field_text).items()
                        if p.ordinal == item.ordinal]
            out[role].append({"owner": ref["owner"], "option": ref["option"], "ordinal": item.ordinal,
                              "part_ids": part_ids, "field_id": hash_to_uuid(node["id"])})
    return out


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def label_hits(index: CalibrationIndex, raw: List[Dict[str, Any]], exclude: List[str]) -> List[Dict[str, Any]]:
    """Labels raw hits with slug, kind, liveness and self for `derive_result`.

    Each hit joins to its node through its ``point_id`` — never its payload slug, which
    an ids-only read does not carry — and the label is the node's slug. A point id this
    collection was not built from asserts: the collection holds exactly the expected set.
    """
    hits = []
    for h in raw:
        node = index.nodes_by_point_id.get(h["point_id"])
        assert node is not None, f"hit {h['point_id']!r} is not a point this collection was built from"
        slug = node["slug"]
        hits.append({
            "slug": slug,
            "value": h["score"],
            "kind": node["kind"],
            "live": index.live_by_slug[slug],
            "self": slug in exclude,
        })
    return hits


def _one_call(proxy: RecordingVectorStore, fixture: Dict[str, Any]) -> List[Dict[str, Any]]:
    assert len(proxy.calls) == 1, (
        f"fixture {fixture['id']!r}: the reader made {len(proxy.calls)} proxied query "
        f"calls; exactly one is the shipped shape"
    )
    return proxy.calls[0][1]


def search_points(base_url: str, collection: str, vector: List[float], limit: int) -> List[Dict[str, Any]]:
    """One raw ``POST …/points/search`` with ``exact`` set, for the two shape collections.

    ``exact`` makes "every point valued" a declaration rather than a consequence of the
    collection sitting below HNSW's indexing threshold. ``VectorStore.query`` returns
    point ids too since 2a, but it has no exact knob, so this read stays until 5a's
    switch to the shipped rejection search retires it.

    Returns:
        ``[{"id", "score", "payload"}]``, best first.

    Raises:
        requests.HTTPError: On a non-2xx answer.
        ValueError: If the body's ``result`` is not a list of hits.
    """
    resp = requests.post(
        f"{base_url.rstrip('/')}/collections/{collection}/points/search",
        json={"vector": vector, "limit": limit, "with_payload": True, "params": {"exact": True}},
        timeout=5,
    )
    resp.raise_for_status()
    result = resp.json().get("result")
    if not isinstance(result, list):
        raise ValueError(f"search on {collection!r} returned result {result!r}, not a list")
    return [{"id": h["id"], "score": h["score"], "payload": h.get("payload")} for h in result]


def read_rejection_shapes(
    index: CalibrationIndex, fixture: Dict[str, Any], vector: List[float]
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Searches both shape collections with the fixture's query vector and derives ``items``.

    Each search takes the whole collection (``limit`` = its point count) and must return
    every point, so ``absent`` on a ref can only mean an unresolved ref, never a censored
    low score. Hits are labelled from the population maps by point id.

    Returns:
        The pure ``items`` value and the number of searches made per shape.

    Raises:
        ValueError: If ``index.shapes`` is unset, naming the fixture.
    """
    shapes = index.shapes
    if shapes is None:
        raise ValueError(
            f"fixture {fixture['id']!r}: measure `item` needs the shape collections "
            f"(CalibrationIndex.shapes is unset)"
        )
    refs = resolve_item_refs(index, fixture)
    searches = {}
    by_shape: Dict[str, List[Dict[str, Any]]] = {}
    for shape, vstore, points in (("per_item", shapes.items_vstore, shapes.item_points),
                                  ("per_decision", shapes.fields_vstore, shapes.field_points)):
        raw = search_points(vstore.base_url, vstore.collection, vector, len(points))
        searches[shape] = 1
        assert len(raw) == len(points), (
            f"fixture {fixture['id']!r}: {shape} search returned {len(raw)} of {len(points)} points"
        )
        hits = []
        for h in raw:
            assert h["id"] in points, f"{shape} hit {h['id']!r} is not a point this population wrote"
            if shape == "per_item":
                owner, ordinal, part = points[h["id"]]
            else:
                owner, ordinal, part = points[h["id"]], None, None
            hits.append({"point_id": h["id"], "owner": owner, "ordinal": ordinal, "part": part,
                         "value": h["score"], "owner_live": index.live_by_slug[owner]})
        by_shape[shape] = hits
    return R.derive_item_result(fixture, by_shape, refs), searches


def read_query_form(index: CalibrationIndex, fixture: Dict[str, Any]) -> FormRead:
    """Reads a query-form fixture through MCP ``surface_decisions``.

    The text is embedded once here first, through the same provider: the reader
    swallows an embedding fault into a degraded envelope, so this is where a quota
    fault can propagate. The reader's own embed is then a cache hit with the same vector.
    An item-measured fixture also reads both rejection shapes with that vector.
    """
    vector = index.provider.get_embedding(fixture["text"], is_query=True)
    proxy = RecordingVectorStore(index.vstore)
    with patch("mitos.mcp_server.get_workspace_components") as components:
        components.return_value = (index.store, index.provider, proxy)
        envelope = json.loads(
            surface_decisions(fixture["text"], project=index.workspace, limit=RECORD_LIMIT, brief=True)
        )
    raw = _one_call(proxy, fixture)
    hits = label_hits(index, raw, fixture.get("exclude", []))
    result = R.derive_result(fixture, hits, envelope.get("confidence"), None)
    searches = None
    if fixture["measure"] == "item":
        result["items"], searches = read_rejection_shapes(index, fixture, vector)
    return FormRead(result=result, calls=proxy.calls, envelope=envelope, shape_searches=searches)


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


def run_provenance(provider: Any, shapes: Optional[ShapeCollections] = None) -> Dict[str, Any]:
    """Extends ``_semantic_harness.provenance()`` for a Layer C run.

    The model is the provider's own, so one value flows to the cache path, the provider
    and the record. The three judge keys are dropped: Layer C makes no judge call. With
    the shape collections, ``record_limit.items`` is ``"all"`` (each shape read takes the
    whole collection) and ``shape_points`` gives the two point counts.
    """
    prov = H.provenance()
    for key in ("judgment_model", "judgment_model_id", "prompt_version"):
        prov.pop(key, None)
    prov["embedding_model"] = provider.model_id
    prov["reader"] = READER
    prov["record_limit"] = {"query": RECORD_LIMIT, "draft": conflict.CONFLICT_OVERFETCH_LIMIT}
    if shapes is not None:
        prov["record_limit"]["items"] = "all"
        prov["shape_points"] = shapes.counts
    prov["constants_in_force"] = constants_in_force()
    return prov


def build_run(index: CalibrationIndex, results: List[Dict[str, Any]], corpus: str = "harbor") -> Dict[str, Any]:
    """Assembles a run from fixture results (``run_id`` is minted by ``append_run``).

    A harbor run carries ``shapes``, the rejection-shape comparison, computed by the pure
    module from the item-measured results; a dense run never does.
    """
    run = {
        "corpus": corpus,
        "decision_count": index.decision_count,
        "node_count": len(index.nodes),
        "provenance": run_provenance(index.provider, index.shapes),
        "fixtures": results,
        "classes": R.summarize_classes(results),
    }
    if corpus == "harbor":
        run["shapes"] = R.summarize_shapes(results)
    return run


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
