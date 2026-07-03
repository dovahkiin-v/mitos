"""Layer-B conflict eval — live, integration-gated, banded (Conflict-sensor §6.3, T7 scaffold).

Drives the SHIPPED conflict facade (`mitos.conflict.run_conflict_check`) with the REAL
SONNET judge over the six golden `conflict:` fixtures against the frozen Harbor corpus,
and scores it. The conflict twin of `test_retrieval_live.py`: read-only over the shipped
pipeline, banded not exact. The tight numbers live in a metrics report + a soft baseline
diff (`_conflict_harness.py`); the only HARD asserts here are the DETERMINISTIC screening
(the declared-target drop is pure graph logic) and one lenient "sensor fundamentally
works" smoke floor — the judge is stochastic, so its verdict quality is soft-diffed, never
red (per Layer B: a semantic regression WARNS for review, never hard-fails, never
auto-seeds the baseline).

Gating (mirrors the other `*_live.py` suites):
  * skips cleanly without BOTH `GEMINI_API_KEY` (embeddings) and `ANTHROPIC_API_KEY`
    (judgment) — conflict needs both — HAS_LIVE_KEYS.
  * skips loudly if Qdrant is unreachable (`QdrantVectorStore.__init__` RAISES on refusal).
  * degrades an embed-quota 429 in populate to a loud skip via `skip_on_embed_quota`.
  * degrades ANY facade `Unavailable` (embed / vector-store / judge timeout-or-5xx) to a
    loud named skip — the executor fail-opens, so a quota-exhausted judge RETURNS
    `Unavailable(JUDGMENT_TIMEOUT)` rather than raising; we inspect the return (Warning E),
    a `try/except anthropic.*` would never fire.
  * uses a throwaway `mitos-tmp-golden-*` collection (swept by conftest even if teardown
    misses) plus its own teardown.

Baseline discipline: 4a ships NO seeded conflict baseline (the judge is live/costly, the
floor is still PROVISIONAL — a meaningful baseline is 4b's calibrated, reviewed act). The
soft-diff test skips loudly when it is absent; seeding is `MITOS_UPDATE_BASELINE=1`-only.
"""

import os
import sys
import uuid
import warnings

import pytest
import requests

from mitos.conflict import CONFLICT_SIMILARITY_FLOOR, Unavailable
from mitos.embeddings import GeminiEmbeddingProvider
from mitos.models import get_embedding_model_id
from mitos.parser import parse_entry_stream
from mitos.vector_store import QdrantVectorStore

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # tests/ for live_helpers
from _harness import CORPUS_PATH, build_reference_graph  # noqa: E402
import _conflict_harness as CH  # noqa: E402  (pulls anthropic in — the quarantine boundary)
import _semantic_harness as H  # noqa: E402
from live_helpers import skip_on_embed_quota  # noqa: E402
from metrics import recommend_floor  # noqa: E402

# Loose smoke floor — of the genuine contradictions that reached the judge, at least half
# must surface. NOT a calibrated per-fixture floor (the real measurement is the report +
# baseline diff): the judge is stochastic and the floor is provisional, so keep it lenient
# enough that ordinary judge jitter never reds it. Raise deliberately (reviewed) once 4b
# has measured the numbers. Mirrors `SMOKE_RECALL_FLOOR` in the retrieval suite.
SMOKE_NOT_TENABLE_RECALL_FLOOR = 0.5

# The calibration probe floor (plan D1/§7). At 0.0 the S5 gate screens NOTHING, so every
# candidate ranked in the top-K is scored and its raw similarity captured — the worst case
# for truncation (a lower floor only ADDS candidates, never removes the named one). Fixture
# 4's declared-drop (S4) is floor-independent, so it still drops here.
CALIBRATION_PROBE_FLOOR = 0.0

# Per-metric regression bands stored with a seeded baseline (4b seeds; 4a only ships the hook).
CONFLICT_BASELINE_BANDS = {
    "not_tenable_recall": 0.20,
    "not_tenable_precision": 0.20,
    "same_polarity_fp_rate": 0.20,
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
# Conflict needs BOTH surfaces: Gemini embeddings (candidate gather) + Anthropic judgment.
HAS_LIVE_KEYS = bool(
    os.environ.get("GEMINI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")
)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:7333")

pytestmark = pytest.mark.skipif(
    not HAS_LIVE_KEYS,
    reason="GEMINI_API_KEY and ANTHROPIC_API_KEY both required — the Layer-B conflict "
    "eval drives live embeddings AND the live SONNET judge.",
)


def _skip_if_unavailable(result) -> None:
    """Turns a facade `Unavailable` into a loud, named skip (environmental, not a defect).

    The conflict facade fail-opens: an embed-quota 429, a Qdrant fault, or a judge
    quota/5xx/timeout all RETURN a typed `Unavailable` (never raise past the seam), so
    the eval hands it back rather than measuring garbage. Each reason is a distinct
    environmental cause, none a code defect — skip loudly with the reason so a keys-present
    CI never reds on a live outage.

    Args:
        result: A `run_conflict_eval` return value.
    """
    if isinstance(result, Unavailable):
        pytest.skip(
            f"conflict facade degraded to Unavailable(reason={result.reason.value}) — "
            f"environmental (embed/vector-store quota or the live SONNET judge "
            f"timeout/5xx/quota); NOT a code defect. detail: {result.detail}"
        )


@pytest.fixture(scope="module")
def conflict_index():
    """Builds the Harbor graph + index + the proposal entry map for the conflict eval.

    Yields:
        A tuple ``(store, provider, vstore, entries_by_slug, collection)``.
    """
    if not H.qdrant_reachable(QDRANT_URL):
        pytest.skip(
            f"Qdrant unreachable at {QDRANT_URL} — the Layer-B conflict eval needs a live "
            f"vector store. Environmental, not a code defect."
        )

    import tempfile

    collection = f"mitos-tmp-golden-{uuid.uuid4().hex[:8]}"
    tmp = tempfile.mkdtemp(prefix="mitos-golden-conflict-")
    store = build_reference_graph(os.path.join(tmp, "graph.sqlite"))

    # The proposal source: parse the SAME frozen corpus into {slug: ParsedEntry}. The
    # facade is driven with the proposal's own ParsedEntry (2a's own-slug guard drops the
    # self-match; the over-fetch leaves margin for the real candidate).
    failures: list = []
    entries = parse_entry_stream(
        open(CORPUS_PATH, encoding="utf-8").read(), "decision", failures=failures
    )
    assert not failures, f"reference corpus failed to parse cleanly: {failures}"
    entries_by_slug = {e.slug: e for e in entries}

    cache_dir = os.path.join(H.GOLDEN_DIR, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"embeddings-{get_embedding_model_id()}.sqlite")
    provider = GeminiEmbeddingProvider(cache_path)
    vstore = QdrantVectorStore(QDRANT_URL, collection)

    try:
        with skip_on_embed_quota():
            H.populate_index(store, provider, vstore)
        yield store, provider, vstore, entries_by_slug, collection
    finally:
        try:
            requests.delete(
                f"{QDRANT_URL.rstrip('/')}/collections/{collection}", timeout=5
            )
        except requests.RequestException:
            pass  # conftest sweep catches mitos-tmp-* as a backstop


def test_conflict_declared_drop_and_smoke_floor(conflict_index):
    """Hard-asserts the deterministic declared drop + a lenient aggregate smoke floor.

    The declared-target drop (fixture 4: `harbor-sync-crdt-merge` declares
    `Contradicts: harbor-sync-last-write-wins`) is pure 2b graph logic, independent of the
    stochastic judge — so it is the one per-fixture behaviour we HARD-assert. The smoke
    floor is deliberately loose (`SMOKE_NOT_TENABLE_RECALL_FLOOR`): a "the sensor
    fundamentally judges contradictions" line, not a calibrated gate.
    """
    store, provider, vstore, entries_by_slug, _ = conflict_index
    oracle = H.load_semantic_oracle()
    with skip_on_embed_quota():
        report = CH.run_conflict_eval(
            oracle, entries_by_slug, provider, vstore, store, CH.make_live_judge()
        )
    _skip_if_unavailable(report)

    summary = CH.conflict_human_summary(report)
    path = H.write_report(report, "conflict-latest")
    print(f"\n{summary}\n[report] {path}")

    # (1) DETERMINISTIC: the declared-contradiction candidate never reached the judge.
    declared = [f for f in report["fixtures"] if f["kind"] == "declared-contradiction"]
    assert len(declared) == 1, "expected exactly one declared-contradiction fixture"
    dfx = declared[0]
    assert not dfx["judged"], (
        f"declared-target drop FAILED: {dfx['candidate']} reached the judge though "
        f"{dfx['proposal']} declares Contradicts it — 2b's declared screen regressed."
    )

    # (2) SMOKE: of the contradictions that WERE judged, at least half surfaced.
    recall = report["aggregate"]["not_tenable_recall"]
    assert recall >= SMOKE_NOT_TENABLE_RECALL_FLOOR, (
        f"not_tenable_recall={recall:.2f} < {SMOKE_NOT_TENABLE_RECALL_FLOOR} smoke floor "
        f"— the sensor is missing genuine contradictions it judged. See the report."
    )


def test_conflict_floor_calibration(conflict_index):
    """Probe run at floor=0.0: capture similarities, recommend the floor, HARD-assert recall.

    The 4b calibration act. Runs the eval at :data:`CALIBRATION_PROBE_FLOOR` so every named
    candidate is scored (nothing screened by S5), writes the calibration readout, and computes
    :func:`recommend_floor`. The one HARD invariant (deterministic — embeddings are stable for
    fixed text+model, so this reds CI on a genuine recall regression, not judge jitter): every
    judged contradiction fixture retrieves at a similarity **>= the landed
    ``CONFLICT_SIMILARITY_FLOOR``** — i.e. the calibrated floor never screens a real
    contradiction (recall-first). Everything judge-quality stays soft (the report + the
    soft-diff test), never red — the Layer B law.
    """
    store, provider, vstore, entries_by_slug, _ = conflict_index
    oracle = H.load_semantic_oracle()
    with skip_on_embed_quota():
        report = CH.run_conflict_eval(
            oracle, entries_by_slug, provider, vstore, store, CH.make_live_judge(),
            floor=CALIBRATION_PROBE_FLOOR,
        )
    _skip_if_unavailable(report)

    summary = CH.conflict_human_summary(report)
    path = H.write_report(report, "conflict-calibration-probe")
    recommended = recommend_floor(report["fixtures"])
    print(
        f"\n{summary}\n"
        f"[recommended floor] {recommended}  "
        f"(landed CONFLICT_SIMILARITY_FLOOR = {CONFLICT_SIMILARITY_FLOOR})\n"
        f"[report] {path}"
    )

    # The judged contradictions (oracle expected_tenable is False) — the recall-first set.
    contradictions = [
        f for f in report["fixtures"]
        if f["expected_tenable"] is False and f["judged"]
    ]
    assert contradictions, (
        "no judged contradiction fixtures in the probe run — the corpus/oracle changed, or "
        "every contradiction was screened/truncated even at floor=0.0 (a top_k-ceiling / hub "
        "signal, not a floor issue). See the report."
    )
    for f in contradictions:
        # A judged contradiction MUST carry a similarity (its candidate reached the judge).
        assert f["similarity"] is not None, (
            f"judged contradiction {f['proposal']} ✗ {f['candidate']} ({f['kind']}) has no "
            f"similarity — inconsistent report record."
        )
        # RECALL-FIRST: the landed floor must not screen a genuine contradiction.
        assert f["similarity"] >= CONFLICT_SIMILARITY_FLOOR, (
            f"recall-first VIOLATION: {f['kind']} {f['proposal']} ✗ {f['candidate']} "
            f"retrieves at similarity={f['similarity']:.4f} < landed "
            f"CONFLICT_SIMILARITY_FLOOR={CONFLICT_SIMILARITY_FLOOR} — the calibrated floor "
            f"would screen this real contradiction before judgment. Recalibrate the floor "
            f"DOWN (min contradiction similarity − margin). See the calibration readout."
        )


def test_conflict_baseline_diff_soft_gate(conflict_index):
    """Soft gate: diff the conflict aggregate vs the seeded baseline; WARN, never hard-fail."""
    store, provider, vstore, entries_by_slug, _ = conflict_index
    baseline = CH.load_conflict_baseline()
    if baseline is None:
        pytest.skip(
            "no conflict.baseline.metrics.json yet — 4a ships the seed hook but not the "
            "baseline (the floor is provisional; seeding is 4b's calibrated act). Seed it "
            "with MITOS_UPDATE_BASELINE=1 (reviewed), then commit it."
        )
    oracle = H.load_semantic_oracle()
    with skip_on_embed_quota():
        report = CH.run_conflict_eval(
            oracle, entries_by_slug, provider, vstore, store, CH.make_live_judge()
        )
    _skip_if_unavailable(report)
    regressions = CH.conflict_baseline_diff(report, baseline)
    if regressions:
        # Layer B is banded + service-dependent + judge-stochastic — a regression flags
        # for human review, it does NOT red CI (never auto-accept, never hard-fail).
        warnings.warn(
            "Layer-B conflict regression vs baseline (REVIEW — do not rubber-stamp):\n"
            + "\n".join(
                f"  {r['metric']}: {r['baseline']:.3f} -> {r['current']:.3f} "
                f"({r['direction']}, band {r['band']})"
                for r in regressions
            ),
            UserWarning,
        )


@pytest.mark.skipif(
    os.environ.get("MITOS_UPDATE_BASELINE") != "1",
    reason="baseline seeding is explicit-only — set MITOS_UPDATE_BASELINE=1 to (re)seed.",
)
def test_seed_conflict_baseline(conflict_index):
    """Explicit-only: freezes the current conflict aggregate as the reviewed baseline.

    4a ships this hook but does NOT run it — a meaningful conflict baseline is 4b's
    calibrated, reviewed act (the floor is still provisional; the judge is live/costly).
    """
    store, provider, vstore, entries_by_slug, _ = conflict_index
    oracle = H.load_semantic_oracle()
    with skip_on_embed_quota():
        report = CH.run_conflict_eval(
            oracle, entries_by_slug, provider, vstore, store, CH.make_live_judge()
        )
    _skip_if_unavailable(report)
    CH.write_conflict_baseline(report, CONFLICT_BASELINE_BANDS)
    print(
        f"\n[conflict baseline seeded — REVIEW before committing]\n"
        f"{CH.conflict_human_summary(report)}"
    )
