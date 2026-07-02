"""Layer-B retrieval eval — live, integration-gated, banded (see MITOS_GOLDEN_DATASET_SPEC Part C).

Runs the golden retrieval fixtures through the shipped embedding + Qdrant path against
the frozen Harbor corpus, and scores them. This is a MEASUREMENT layer: read-only over
the existing surfaces, banded not exact. The tight numbers live in a metrics report +
a soft baseline diff (`_semantic_harness.py`); the hard asserts here are deliberately
loose recall floors — a "retrieval is fundamentally working" smoke, not a rank pin
(embedding order drifts by design; Fable #8).

Gating (mirrors the project's other `*_live.py` suites):
  * skips cleanly without `GEMINI_API_KEY` (embeddings) — HAS_LIVE_KEYS.
  * skips loudly if Qdrant is unreachable — `QdrantVectorStore.__init__` RAISES on a
    refused connection, which `skip_on_embed_quota` would NOT catch, so we probe first
    (Fable #3).
  * degrades a 429 embed-quota exhaustion to a loud skip via `skip_on_embed_quota`.
  * uses a throwaway `mitos-tmp-golden-*` collection (swept by conftest even if teardown
    misses) plus its own teardown.

Baseline discipline: the baseline is written ONLY under `MITOS_UPDATE_BASELINE=1`
(`test_seed_baseline`), never from an ordinary run — a quota-degraded run can't silently
become ground truth (Fable #4). Absent baseline → the diff test skips loudly.
"""

import os
import sys
import uuid
import warnings

import pytest
import requests

from mitos.embeddings import GeminiEmbeddingProvider
from mitos.mcp_server import surface_decisions
from mitos.models import get_embedding_model_id
from mitos.vector_store import QdrantVectorStore

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # tests/ for live_helpers
from _harness import build_reference_graph  # noqa: E402
import _semantic_harness as H  # noqa: E402
from live_helpers import skip_on_embed_quota  # noqa: E402

# Loose smoke floor — at least half of each relevant set must surface in the top-k.
# NOT a calibrated per-fixture floor: the real measurement is the report + baseline
# diff. Raise deliberately (reviewed, like --update-golden) once measured.
SMOKE_RECALL_FLOOR = 0.5
TOP_K = 5

# Per-metric regression bands stored with a seeded baseline (Fable #10).
BASELINE_BANDS = {
    "recall_at_k": 0.10,
    "precision_at_k": 0.15,
    "mrr": 0.15,
    "hard_negative_fp_rate": 0.10,
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
HAS_LIVE_KEYS = bool(os.environ.get("GEMINI_API_KEY"))  # retrieval needs embeddings only
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:7333")

pytestmark = pytest.mark.skipif(
    not HAS_LIVE_KEYS,
    reason="GEMINI_API_KEY not set — Layer-B retrieval eval needs live embeddings.",
)


@pytest.fixture(scope="module")
def populated_index():
    """Builds the Harbor graph, embeds it into a throwaway Qdrant collection, yields it.

    Yields:
        A tuple ``(store, provider, vstore, collection)`` over the populated index.
    """
    if not H.qdrant_reachable(QDRANT_URL):
        pytest.skip(
            f"Qdrant unreachable at {QDRANT_URL} — Layer-B retrieval eval needs a live "
            f"vector store. Environmental, not a code defect."
        )

    import tempfile

    collection = f"mitos-tmp-golden-{uuid.uuid4().hex[:8]}"
    tmp = tempfile.mkdtemp(prefix="mitos-golden-live-")
    store = build_reference_graph(os.path.join(tmp, "graph.sqlite"))

    cache_dir = os.path.join(H.GOLDEN_DIR, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    # Model-keyed cache PATH — the EmbeddingCache key is content-hash only (no model
    # column), so a model swap must land on a different file or it returns stale
    # vectors and masks embedding drift (Fable #11).
    cache_path = os.path.join(cache_dir, f"embeddings-{get_embedding_model_id()}.sqlite")
    provider = GeminiEmbeddingProvider(cache_path)
    vstore = QdrantVectorStore(QDRANT_URL, collection)

    try:
        with skip_on_embed_quota():
            n = H.populate_index(store, provider, vstore)
        assert n == 28, f"expected 28 corpus nodes indexed, got {n}"
        yield store, provider, vstore, collection
    finally:
        try:
            requests.delete(f"{QDRANT_URL.rstrip('/')}/collections/{collection}", timeout=5)
        except requests.RequestException:
            pass  # conftest sweep catches mitos-tmp-* as a backstop


def test_retrieval_smoke_floors(populated_index, request):
    """Every gating fixture clears the loose recall floor; writes the metrics report."""
    store, provider, vstore, _ = populated_index
    oracle = H.load_semantic_oracle()
    with skip_on_embed_quota():
        report = H.run_retrieval_eval(oracle, provider, vstore, k=TOP_K)

    # Human summary to the test log (report JSON to the gitignored reports dir).
    summary = H.human_summary(report)
    path = H.write_report(report, "retrieval-latest")
    print(f"\n{summary}\n[report] {path}")

    failures = []
    for fx in report["fixtures"]:
        if fx["measure_only"]:
            continue
        recall = fx["metrics"]["recall_at_k"]
        if recall < SMOKE_RECALL_FLOOR:
            failures.append(f"  recall={recall:.2f} < {SMOKE_RECALL_FLOOR} «{fx['query'][:60]}»")
    assert not failures, "retrieval recall below smoke floor:\n" + "\n".join(failures)


def test_baseline_diff_soft_gate(populated_index):
    """Soft gate: diff aggregate metrics vs the seeded baseline; WARN, never hard-fail."""
    store, provider, vstore, _ = populated_index
    baseline = H.load_baseline()
    if baseline is None:
        pytest.skip(
            "no baseline.metrics.json yet — seed it with MITOS_UPDATE_BASELINE=1 "
            "(reviewed, like --update-golden), then commit it."
        )
    oracle = H.load_semantic_oracle()
    with skip_on_embed_quota():
        report = H.run_retrieval_eval(oracle, provider, vstore, k=TOP_K)
    regressions = H.baseline_diff(report, baseline)
    if regressions:
        # Layer B is banded + service-dependent — a semantic regression flags for
        # human review, it does not red CI (never auto-accept, never hard-fail).
        warnings.warn(
            "Layer-B retrieval regression vs baseline (REVIEW — do not rubber-stamp):\n"
            + "\n".join(
                f"  {r['metric']}: {r['baseline']:.3f} -> {r['current']:.3f} "
                f"({r['direction']}, band {r['band']})"
                for r in regressions
            ),
            UserWarning,
        )


def test_mcp_surface_smoke(populated_index):
    """One realism pass through the MCP surface against the test index (Fable #2).

    Patches `get_workspace_components` to inject the test (store, provider, vstore)
    so `surface_decisions` resolves the golden corpus, not the repo's real .mitos.
    """
    from unittest.mock import patch

    store, provider, vstore, _ = populated_index
    with patch("mitos.mcp_server.get_workspace_components") as mock_get:
        mock_get.return_value = (store, provider, vstore)
        with skip_on_embed_quota():
            res = surface_decisions(
                "How do we resolve two people editing the same file at the same time?"
            )
    # The concurrent-edit precedent should surface through the real MCP path.
    assert "harbor-sync" in res, f"expected a sync decision in MCP surface output; got: {res[:400]}"


@pytest.mark.skipif(
    os.environ.get("MITOS_UPDATE_BASELINE") != "1",
    reason="baseline seeding is explicit-only — set MITOS_UPDATE_BASELINE=1 to (re)seed.",
)
def test_seed_baseline(populated_index):
    """Explicit-only: freezes the current aggregate as the reviewed baseline (Fable #4)."""
    store, provider, vstore, _ = populated_index
    oracle = H.load_semantic_oracle()
    with skip_on_embed_quota():
        report = H.run_retrieval_eval(oracle, provider, vstore, k=TOP_K)
    H.write_baseline(report, BASELINE_BANDS)
    print(f"\n[baseline seeded — REVIEW before committing]\n{H.human_summary(report)}")
