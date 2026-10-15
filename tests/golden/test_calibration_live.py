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

Hard asserts are structural only: the collection holds every node, each read makes one
proxied call, every target has a value, and the envelope agrees with the raw hits.
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
from mitos.vector_store import QdrantVectorStore, hash_to_uuid

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
    lines = []
    for space in R.SPACES:
        values = [t["value"] for r in results if r["space"] == space
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
    assert len(index.nodes_by_slug) == len(index.nodes), "payload slugs are not unique"


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
        raw_score = {h["slug"]: h["score"] for h in raw}
        if fx["form"] == "query":
            assert "confidence" in read.envelope, f"{fx['id']}: degraded envelope {read.envelope}"
            for d in read.envelope["active_decisions"]:
                assert d["score"] == raw_score.get(d["slug"]), (
                    f"{fx['id']}: envelope score for {d['slug']} is not the proxied hit's"
                )
        else:
            for c in read.candidates:
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


def _labelled_run(index):
    """Reads every labelled fixture and builds the run, or skips while there are none."""
    fixtures = R.load_fixtures()
    if not fixtures:
        pytest.skip("no Layer C fixtures yet — 1c labels them in calibration/fixtures.json")
    R.check_fixture_slugs(fixtures, index.nodes_by_slug)
    results = [_read(index, fx).result for fx in fixtures]
    return CH.build_run(index, results)


def test_calibration_classes(calibration_index):
    """Reads the labelled fixtures, reports the class distributions, soft-diffs the record."""
    run = _labelled_run(calibration_index)
    path = H.write_report(run, "calibration-latest")
    print(f"\n[layer-c] classes: {json.dumps({c: [b['n'], b['min'], b['median'], b['max']] for c, b in run['classes'].items()})}\n[report] {path}")
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
def test_record_run(calibration_index):
    """Explicit-only: appends the labelled run to the committed record. Review before committing."""
    run = _labelled_run(calibration_index)
    run_id = R.append_run(run)
    print(f"\n[layer-c] recorded run {run_id} — REVIEW before committing")
