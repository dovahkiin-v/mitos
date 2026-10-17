"""Layer C — similarity values recorded through the shipped readers, live and serial.

Layer B keeps the order of results and drops every score, so it cannot see a model swap
that rescales similarity. Layer C records the values. It builds the frozen Harbor graph,
embeds every node (the retired ones included) into a throwaway ``mitos-tmp-golden-layerc-*``
collection through the model-keyed golden cache, and reads fixtures through MCP
``surface_decisions`` (query space) and ``conflict.gather_candidates`` (document space)
with a recording proxy on the vector store. See ``tests/golden/calibration/README.md``.

Gating mirrors ``test_retrieval_live.py``: it runs only with a Gemini key and the live
brake off, skips loudly when Qdrant does not answer, and turns an embedding quota fault
into a named skip. No Anthropic call is made here.

Rejection-target fixtures are also read against two more throwaway collections, the two
rejection-index shapes (one point per rejected item part, one per decision's whole field),
built by the module-scoped ``shape_collections`` fixture. Since 5b the per-item one is
written and searched through the shipped ``QdrantRejectionIndex`` in production's shape,
and bound as the reads' per-item index: every query-form read then carries the shipped
rejection section and checks it against the instrument's own ranking.

Hard asserts are structural only: the collections hold every point they should, each read
makes one proxied call (and one search per shape), every target has a value, and the
envelope agrees with the raw hits.
Values go to a gitignored report and, for labelled fixtures, into a soft diff that
warns and never fails. A run enters the committed record only under
``MITOS_UPDATE_CALIBRATION=1``.
"""

import json
import os
import sys
import uuid
import warnings

import pytest
import requests

from mitos import conflict, sync
from mitos.embeddings import GeminiEmbeddingProvider
from mitos.errors import EmbeddingError
from mitos.models import get_embedding_model_id
from mitos.rejections import rejection_points
from mitos.vector_store import QdrantRejectionIndex, QdrantVectorStore, hash_to_uuid

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # tests/ for live_helpers
from _harness import build_reference_graph  # noqa: E402
import _calibration_harness as CH  # noqa: E402
import _calibration_record as R  # noqa: E402
import _semantic_harness as H  # noqa: E402
from live_helpers import (  # noqa: E402
    EMBED_QUOTA_SKIP_REASON,
    _is_embed_quota_exhausted,
    live_tests_disabled,
    skip_on_embed_quota,
)

UPDATE_FLAG = "MITOS_UPDATE_CALIBRATION"
RETIRED_NODES = ("harbor-auth-jwt-v1", "harbor-auth-jwt-v2", "harbor-legacy-ftp-gateway")
# Verbatim harbor-auth-jwt-v1's axiom, read as a query: the raw hits should hold a JWT
# node with a value while the envelope lists none of them as live.
RETIRED_PROBE = {
    "id": "retired-probe-jwt", "class": None, "form": "query",
    "text": "Harbor authenticates API calls with JWTs",
    "targets": ["harbor-auth-jwt-v1"], "negatives": [], "measure": "target",
    "notes": "the proxy measures a retired precedent the envelope hides",
}


def _load_live_env() -> None:
    """Loads keys from the repo-root .env into os.environ (mirrors the live suites)."""
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"
    )
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


_load_live_env()
HAS_LIVE_KEYS = (not live_tests_disabled()) and bool(os.environ.get("GEMINI_API_KEY"))
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:7333")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

pytestmark = pytest.mark.skipif(
    not HAS_LIVE_KEYS,
    reason="GEMINI_API_KEY not set — Layer C records similarity values from live embeddings.",
)


@pytest.fixture(scope="module")
def calibration_index(tmp_path_factory):
    """Builds Harbor, embeds every node into a throwaway collection, yields the index."""
    if not H.qdrant_reachable(QDRANT_URL):
        pytest.skip(
            f"Qdrant unreachable at {QDRANT_URL} — Layer C needs a live vector store. "
            f"Environmental, not a code defect."
        )
    from conftest import make_workspace

    collection = f"mitos-tmp-golden-layerc-{uuid.uuid4().hex[:8]}"
    store = build_reference_graph(str(tmp_path_factory.mktemp("layerc-graph") / "graph.sqlite"))
    model_id = get_embedding_model_id()
    cache_path = R.golden_cache_path(model_id)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    provider = GeminiEmbeddingProvider(cache_path, api_key=GEMINI_API_KEY, model_id=model_id)
    # Construction is pure; the first covering upsert creates the collection.
    vstore = QdrantVectorStore(QDRANT_URL, collection)
    try:
        with skip_on_embed_quota():
            index = CH.populate_layer_c(store, provider, vstore)
        index.workspace = make_workspace(tmp_path_factory.mktemp("layerc-ws"))
        yield index
    finally:
        try:
            requests.delete(f"{QDRANT_URL.rstrip('/')}/collections/{collection}", timeout=5)
        except requests.RequestException:
            pass  # the conftest sweep removes mitos-tmp-* as a backstop


@pytest.fixture(scope="module")
def shape_collections(calibration_index):
    """Builds both rejection-index shapes over every Harbor decision and binds the per-item
    one as the reads' item index; deletes both after."""
    index = calibration_index
    suffix = uuid.uuid4().hex[:8]
    names = (f"mitos-tmp-golden-layerc-items-{suffix}", f"mitos-tmp-golden-layerc-fields-{suffix}")
    items_index = QdrantRejectionIndex(QDRANT_URL, names[0])
    fields_vstore = QdrantVectorStore(QDRANT_URL, names[1])
    try:
        with skip_on_embed_quota():
            index.shapes = CH.populate_shape_collections(index, items_index, fields_vstore)
        index.items = index.shapes.items
        yield index.shapes
    finally:
        index.shapes = None
        index.items = None
        for name in names:
            try:
                requests.delete(f"{QDRANT_URL.rstrip('/')}/collections/{name}", timeout=5)
            except requests.RequestException:
                pass  # the conftest sweep removes mitos-tmp-* as a backstop


def _read(index, fixture):
    """Reads one fixture, turning an environmental embedding fault into a named skip."""
    try:
        with skip_on_embed_quota():
            return CH.read_fixture(index, fixture)
    except CH.DraftReadUnavailable as exc:
        u = exc.unavailable
        if (u.reason is conflict.ConflictUnavailableReason.EMBEDDING
                and _is_embed_quota_exhausted(EmbeddingError(u.detail))):
            pytest.skip(EMBED_QUOTA_SKIP_REASON)
        pytest.fail(f"fixture {fixture['id']!r}: the gather degraded: {exc}")


def _stats_line(index) -> str:
    hits, misses, rate = index.provider.get_stats()
    return f"[layer-c] embedding cache this session: hits={hits} misses={misses} rate={rate:.2f}"


def _value_ranges(results) -> str:
    """Decision-ranking target values per space; item-measured fixtures are left out,
    since their slug targets are the owner and adopter, recorded as context."""
    lines = []
    for space in R.SPACES:
        values = [t["value"] for r in results if r["space"] == space and r["measure"] != "item"
                  for t in r["targets"] if t["value"] is not None]
        if values:
            lines.append(f"  {space:8} targets n={len(values)} min={min(values):.4f} max={max(values):.4f}")
    return "\n".join(lines)


def test_collection_holds_every_harbor_node(calibration_index):
    """The collection is production's expected set: every decision, retired ones too, plus OQs."""
    index = calibration_index
    expected = CH.expected_set(index.store)
    assert len(index.nodes) == len(expected) == 31, (
        f"expected 31 nodes (29 decisions + 2 open questions), populated {len(index.nodes)}; "
        f"Layer B's active-only 28 is not this set"
    )
    assert index.decision_count == 29
    ids = index.vstore.list_point_ids()
    assert ids == {hash_to_uuid(n["id"]) for n in expected}
    by_slug = {n["slug"]: n for n in expected}
    for slug in RETIRED_NODES:
        assert hash_to_uuid(by_slug[slug]["id"]) in ids, f"retired {slug} has no point"
        assert index.live_by_slug[slug] is False
    assert len(index.nodes_by_slug) == len(index.nodes), "slugs are not unique (fixtures resolve by slug)"
    assert len(index.nodes_by_point_id) == len(index.nodes), "point ids are not unique (hits join by point id)"


def test_shape_collections_hold_every_rejection_point(calibration_index, shape_collections):
    """Per item: every rejection point of every decision, retired owners included, listed
    through the shipped client. Per decision: one point per decision with a non-empty field."""
    index = calibration_index
    decisions = [n for n in index.nodes if n["kind"] == "decision"]
    expected_items = set()
    for n in decisions:
        expected_items |= set(rejection_points(n["id"], n["rejected_paths"] or ""))
    assert shape_collections.items_index.list_point_ids() == expected_items == set(shape_collections.item_points)
    assert set(index.items.points) == expected_items
    expected_fields = {hash_to_uuid(n["id"]) for n in decisions if (n["rejected_paths"] or "").strip()}
    assert shape_collections.fields_vstore.list_point_ids() == expected_fields == set(shape_collections.field_points)
    by_slug = {n["slug"]: n for n in decisions}
    for slug in RETIRED_NODES:
        owned = set(rejection_points(by_slug[slug]["id"], by_slug[slug]["rejected_paths"]))
        assert owned and owned <= expected_items, f"retired {slug}'s items are not in the per-item shape"
    print(f"\n[layer-c] shapes: {shape_collections.counts}\n{_stats_line(index)}")


def test_smoke_records_values_in_both_spaces(calibration_index):
    """Layer B's 14 fixtures, read through both shipped readers, each yield values."""
    index = calibration_index
    axioms = {n["slug"]: n["core_axiom"] for n in index.nodes if n["kind"] == "decision"}
    fixtures = CH.adapt_layer_b_oracle(H.load_semantic_oracle(), axioms)
    assert len(fixtures) == 14
    reads = [_read(index, fx) for fx in fixtures]

    problems = []
    for fx, read in zip(fixtures, reads):
        r = read.result
        raw = read.calls[0][1]
        if not r["hits"]:
            problems.append(f"{fx['id']}: no hits")
        for t in r["targets"]:
            if t["value"] is None:
                problems.append(f"{fx['id']}: target {t['slug']} absent from the window (a finding)")
        # Keyed through the node: the walk's hits carry no payload slug (ids only).
        raw_score = {index.nodes_by_point_id[h["point_id"]]["slug"]: h["score"] for h in raw}
        if fx["form"] == "query":
            assert "confidence" in read.envelope, f"{fx['id']}: degraded envelope {read.envelope}"
            for d in read.envelope["active_decisions"]:
                assert d["score"] == raw_score.get(d["slug"]), (
                    f"{fx['id']}: envelope score for {d['slug']} is not the proxied hit's"
                )
        else:
            for c in read.candidates.live:
                assert c.score == raw_score.get(c.slug), (
                    f"{fx['id']}: candidate {c.slug} is not a proxied hit with an equal score"
                )
            assert r["pause_floor"] == sync._NEIGHBOR_REVIEW_THRESHOLD
    assert not problems, "smoke findings:\n" + "\n".join(problems)

    results = [read.result for read in reads]
    report = {"provenance": CH.run_provenance(index.provider), "results": results}
    path = H.write_report(report, "calibration-smoke-latest")
    lines = [f"Layer C smoke — {len(results)} fixtures, {report['provenance']['embedding_model']}"]
    for r in results:
        values = ", ".join(f"{t['slug']}={t['value']:.4f}" for t in r["targets"])
        lines.append(f"  {r['id']:22} {r['space']:8} band={r['envelope_band']} {values}")
    lines += [_value_ranges(results), _stats_line(index), f"[report] {path}"]
    print("\n" + "\n".join(lines))


def test_retired_values_are_measured_where_the_envelope_hides_them(calibration_index):
    """The proxy holds a retired node's value; the envelope lists no retired node as live."""
    index = calibration_index
    read = _read(index, R.validate_fixtures([dict(RETIRED_PROBE)], require_class=False)[0])
    retired = [h for h in read.result["hits"] if h["slug"] in RETIRED_NODES[:2]]
    assert retired, f"no JWT node among the raw hits: {[h['slug'] for h in read.result['hits']]}"
    assert all(isinstance(h["value"], float) and not h["live"] for h in retired)
    listed = {d["slug"] for d in read.envelope["active_decisions"]}
    assert not listed & set(RETIRED_NODES), f"the envelope lists a retired node as live: {listed}"
    target = read.result["targets"][0]
    assert target["value"] is not None and target["live_rank"] is None
    print(f"\n[layer-c] retired probe: jwt-v1={target['value']:.4f} "
          f"band={read.result['envelope_band']}\n{_stats_line(index)}")


def _labelled_reads(index):
    """Reads every labelled fixture, or skips while there are none."""
    fixtures = R.load_fixtures()
    if not fixtures:
        pytest.skip("no Layer C fixtures yet — calibration/fixtures.json is empty")
    R.check_fixture_slugs(fixtures, index.nodes_by_slug)
    return fixtures, [_read(index, fx) for fx in fixtures]


def _fmt(v):
    return "  —   " if v is None else f"{v:.4f}"


def _class_lines(run) -> str:
    """One line per (class, space): targets and negatives, min / median / max."""
    lines = []
    for cls, spaces in run["classes"].items():
        for space, b in spaces.items():
            neg = b["negatives"]
            lines.append(
                f"  {cls:18} {space:8} {b['measure']:8} n={b['n']:2} "
                f"targets {_fmt(b['min'])} {_fmt(b['median'])} {_fmt(b['max'])} absent={b['absent']}  "
                f"negatives n={neg['n']:2} {_fmt(neg['min'])} {_fmt(neg['median'])} {_fmt(neg['max'])}"
            )
    for shape, block in run.get("shapes", {}).items():
        t, n, m = block["targets"], block["negatives"], block["margin"]
        lines.append(
            f"  shape {shape:12} targets median={_fmt(t['median'])} p10={_fmt(t['p10'])}  "
            f"negatives median={_fmt(n['median'])} p90={_fmt(n['p90'])}  "
            f"rank-one={block['target_rank_one']:.2f} margin median={_fmt(m['median'])}"
        )
    return "\n".join(lines)


def _floor_lines(run) -> str:
    """The rejection floor's reading of this run (5b D3), and the section's appearance
    rate on the query fixtures that re-propose nothing (the generic attractor's evidence)."""
    from mitos.recall import REJECTION_MATCH_FLOOR

    results = run["fixtures"]
    boundary = R.rejection_floor_boundary(results)
    lines = [f"  rejection floor: boundary {boundary} (in force {REJECTION_MATCH_FLOOR}); carrying-target "
             f"retention at boundary {R.rejection_floor_retention(results, boundary) if boundary else None}"]
    others = [r for r in results if r["space"] == "query" and r["class"] != "rejection-target"]
    shown = sum(1 for r in others if r.get("section"))
    lines.append(f"  sections on non-rejection query fixtures: {shown}/{len(others)} "
                 f"(matches {sum(len(r.get('section') or []) for r in others)})")
    return "\n".join(lines)


def test_calibration_classes(calibration_index, shape_collections):
    """Reads the labelled fixtures, reports the class distributions, soft-diffs the record."""
    index = calibration_index
    fixtures, reads = _labelled_reads(index)
    for fx, read in zip(fixtures, reads):
        raw = read.calls[0][1]
        # Keyed through the node: the walk's hits carry no payload slug (ids only).
        raw_score = {index.nodes_by_point_id[h["point_id"]]["slug"]: h["score"] for h in raw}
        if fx["form"] == "query":
            assert "confidence" in read.envelope, f"{fx['id']}: degraded envelope {read.envelope}"
            for d in read.envelope["active_decisions"]:
                assert d["score"] == raw_score.get(d["slug"]), f"{fx['id']}: envelope score for {d['slug']}"
        if fx["form"] == "query":
            # The read checked the shipped section against the instrument (it raises on a
            # disagreement); the section was searched and whole.
            assert read.result["section_state"] is None, fx["id"]
            assert len(read.item_ranking) == len(index.items.points), fx["id"]
        if fx["measure"] == "item":
            assert read.shape_searches == {"per_item": 1, "per_decision": 1}, fx["id"]
            assert set(read.result["items"]) == set(R.SHAPES), fx["id"]
            for shape in R.SHAPES:
                refs = read.result["items"][shape]["targets"] + read.result["items"][shape]["negatives"]
                assert all(not r["absent"] for r in refs), f"{fx['id']}: an item ref with no value in {shape}"
        else:
            assert read.shape_searches is None and "items" not in read.result, fx["id"]
        if fx["form"] == "draft" and fx["class"] == "retired-precedent":
            # 7a §6.6: the read asked the shipped tombstone review (it raises on a
            # disagreement), and every target at or above the floor came back shown.
            assert read.tombstones is not None, f"{fx['id']}: the tombstone agreement did not run"
            for t in read.result["targets"]:
                if t["value"] is not None and t["value"] >= sync._TOMBSTONE_FLOOR:
                    assert t["slug"] in read.tombstones, fx["id"]
    run = CH.build_run(index, [read.result for read in reads])
    expected = {(f["class"], R.SPACE_OF_FORM[f["form"]]) for f in fixtures}
    assert {(c, s) for c, spaces in run["classes"].items() for s in spaces} == expected
    assert set(run["shapes"]) == set(R.SHAPES)
    path = H.write_report(run, "calibration-latest")
    print(f"\n[layer-c] classes (min median max):\n{_class_lines(run)}\n{_floor_lines(run)}\n"
          f"{_stats_line(index)}\n[report] {path}")
    record = R.load_record()
    baseline_id = R.latest_run_id(record, "harbor")
    if baseline_id is None:
        pytest.skip(f"the record holds no harbor run yet — record one with {UPDATE_FLAG}=1, then commit it")
    diffs = R.soft_diff(run, R.load_run(baseline_id))
    regressions = [d for d in diffs if d["regression"]]
    if diffs:
        warnings.warn(
            f"Layer C values differ from run {baseline_id} (REVIEW, do not rubber-stamp; "
            f"{len(regressions)} flagged):\n" + "\n".join(f"  {json.dumps(d)}" for d in diffs),
            UserWarning,
        )


@pytest.mark.skipif(
    os.environ.get(UPDATE_FLAG) != "1",
    reason=f"recording a run is explicit-only — set {UPDATE_FLAG}=1 to append one to the record.",
)
def test_record_run(calibration_index, shape_collections):
    """Explicit-only: appends the labelled run to the committed record. Review before committing."""
    _, reads = _labelled_reads(calibration_index)
    run = CH.build_run(calibration_index, [read.result for read in reads])
    run_id = R.append_run(run)
    print(f"\n[layer-c] recorded run {run_id} — REVIEW before committing")
