"""Phase 6b — the live real-corpus dogfood (the §1.2 P10 belt-and-suspenders proof).

The one test that runs the WHOLE shipped stack — Gemini embeddings + Qdrant + the live
SONNET judge — against the real ``Forge/mt/decisions.md`` corpus and proves a genuine
contradiction surfaces at high confidence. Where ``test_conflict_closeout.py`` proves the
gates keyless-deterministically and ``tests/golden/test_conflict_eval_live.py`` scores the
judge over the frozen Harbor fixtures, this closes the §1.2 wording exactly: *"on a real
corpus, an entry that opposes an active decision it does not declare a relationship with is
flagged at sync-time … at high precision."*

READ-ONLY against a TEMP-indexed copy (plan D6). ``/home/vinga/Forge/mt/decisions.md`` is
the framework's live gold source, so the fixture indexes it into a throwaway workspace +
``mitos-tmp-golden-*`` Qdrant collection and drives the **facade**
(``run_conflict_check``) directly — it NEVER runs a committing ``mitos sync`` against the
real ``decisions.md`` and writes no ``conflict_checks`` corpus into ``Forge/mt``.

Gating (mirrors the other ``*_live.py`` suites — a live outage must never red CI, only a
real defect):
  * skips cleanly without BOTH ``GEMINI_API_KEY`` (embeddings) and ``ANTHROPIC_API_KEY``
    (the judge) — ``HAS_LIVE_KEYS``.
  * skips loudly if Qdrant is unreachable (``QdrantVectorStore.__init__`` RAISES on refusal).
  * skips loudly if the gold corpus is absent (a box without the Forge checkout).
  * degrades an embed-quota 429 to a loud skip via ``skip_on_embed_quota``.
  * degrades ANY facade ``Unavailable`` (embed / vector-store / judge timeout-or-5xx) to a
    loud named skip — the executor fail-opens, so a quota-exhausted judge RETURNS
    ``Unavailable(JUDGMENT_TIMEOUT)`` rather than raising; we inspect the return.

No new dependency (P19): ``anthropic`` rides in only through the golden
``_conflict_harness`` quarantine module (``make_live_judge``); ``requests`` + the Gemini/
Qdrant clients are already present.
"""

import os
import sys
import uuid

import pytest
import requests

from mitos.conflict import (
    CONFLICT_SURFACE_THRESHOLD,
    Unavailable,
    run_conflict_check,
)
from mitos.embeddings import GeminiEmbeddingProvider
from mitos.models import get_embedding_model_id
from mitos.parser import ParsedEntry
from mitos.store import GraphStore
from mitos.vector_store import QdrantVectorStore

# The golden harness lives under tests/golden/; add it (and tests/) to the path the same way
# test_conflict_eval_live.py does, then reuse its service-touching pieces — do NOT re-author
# the index/judge scaffold (plan D1: reuse, don't duplicate).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "golden"))
sys.path.insert(0, os.path.dirname(__file__))
import _semantic_harness as H  # noqa: E402
from _conflict_harness import make_live_judge  # noqa: E402  (pulls anthropic — the quarantine boundary)
from live_helpers import skip_on_embed_quota  # noqa: E402

# The framework's live gold decision corpus (241 lines). READ-ONLY — never synced against.
DOGFOOD_CORPUS_PATH = "/home/vinga/Forge/mt/decisions.md"

# The indexed decision the probe proposal is authored to blatantly contradict: CONF-D4's
# "high precision over recall, surface only >= 0.85" ruling. The proposal argues the exact
# opposite (recall-first, surface everything) — an unambiguous not-tenable a temp-0.3 SONNET
# judge rates at high confidence. Retrieval is deterministic for fixed text+model, so this
# candidate reaches the judge on every run; the judge verdict is the (reliable) live part.
CONTRADICTED_DECISION_SLUG = "conflict-surface-threshold-085"


def _load_live_env() -> None:
    """Loads keys from the repo-root .env into os.environ (mirrors the live suites)."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


_load_live_env()
# The dogfood needs BOTH surfaces: Gemini embeddings (candidate gather) + Anthropic judgment.
HAS_LIVE_KEYS = bool(
    os.environ.get("GEMINI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")
)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:7333")

pytestmark = pytest.mark.skipif(
    not HAS_LIVE_KEYS,
    reason="GEMINI_API_KEY and ANTHROPIC_API_KEY both required — the live dogfood drives "
    "real embeddings AND the live SONNET judge against the real corpus.",
)


def _build_dogfood_graph(db_path: str) -> GraphStore:
    """Parses the real gold corpus and commits it oldest-first into a throwaway graph.

    Mirrors ``tests/golden/_harness.build_reference_graph`` but over the live
    ``Forge/mt/decisions.md`` (which is authored newest-first, so ``reversed`` gives the
    oldest-first commit order an edge's target needs to pre-exist). Read-only w.r.t. the
    source file — it only ever reads it.

    Args:
        db_path: Filesystem path for the throwaway SQLite graph.

    Returns:
        The populated ``GraphStore``.

    Raises:
        AssertionError: If the gold corpus fails to parse cleanly.
    """
    from mitos.parser import parse_entry_stream

    store = GraphStore(db_path)
    failures: list = []
    text = open(DOGFOOD_CORPUS_PATH, encoding="utf-8").read()
    entries = parse_entry_stream(text, "decision", failures=failures)
    assert not failures, f"gold corpus failed to parse cleanly: {failures}"
    for entry in reversed(entries):  # oldest-first
        store.commit_parsed_entry(entry)
    return store


@pytest.fixture(scope="module")
def dogfood_index():
    """Indexes the real gold corpus into a throwaway graph + Qdrant collection (read-only).

    Yields:
        A tuple ``(store, provider, vstore, collection)``.
    """
    if not os.path.exists(DOGFOOD_CORPUS_PATH):
        pytest.skip(
            f"gold corpus absent at {DOGFOOD_CORPUS_PATH} (no Forge checkout on this box) — "
            f"the real-corpus dogfood is environmental, not a code defect."
        )
    if not H.qdrant_reachable(QDRANT_URL):
        pytest.skip(
            f"Qdrant unreachable at {QDRANT_URL} — the live dogfood needs a vector store. "
            f"Environmental, not a code defect."
        )

    import tempfile

    collection = f"mitos-tmp-golden-{uuid.uuid4().hex[:8]}"
    tmp = tempfile.mkdtemp(prefix="mitos-dogfood-conflict-")
    store = _build_dogfood_graph(os.path.join(tmp, "graph.sqlite"))

    cache_dir = os.path.join(H.GOLDEN_DIR, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"embeddings-{get_embedding_model_id()}.sqlite")
    provider = GeminiEmbeddingProvider(cache_path)
    vstore = QdrantVectorStore(QDRANT_URL, collection)

    try:
        with skip_on_embed_quota():
            H.populate_index(store, provider, vstore)
        yield store, provider, vstore, collection
    finally:
        try:
            requests.delete(
                f"{QDRANT_URL.rstrip('/')}/collections/{collection}", timeout=5
            )
        except requests.RequestException:
            pass  # conftest sweep catches mitos-tmp-* as a backstop


def _recall_first_proposal() -> ParsedEntry:
    """Builds the probe proposal — a recall-first axiom that contradicts CONF-D4.

    A blatant, unambiguous opposite of the indexed ``conflict-surface-threshold-085``
    ("high precision over recall; surface only >= 0.85"): surface everything at any
    confidence, tuned entirely for recall. Its own slug/declarations are fresh, so 2b's
    own-slug guard and the declared-drop leave the real candidate in the batch.

    Returns:
        A populated ``ParsedEntry`` standing as the proposal.
    """
    entry = ParsedEntry("decision", "conflict-surface-tune-for-recall", 1, 1)
    entry.axiom = (
        "The conflict sensor surfaces every detected tension at the sync-time accept prompt "
        "regardless of confidence — including low-confidence and merely-possible "
        "contradictions — tuning entirely for recall so no potential conflict is ever "
        "suppressed, accepting a noisy false-positive stream as the price of never missing one."
    )
    entry.rejected_paths = (
        "Rejected a high-precision confidence gate that stays silent below a threshold "
        "(the very posture this decision overturns)."
    )
    entry.scope = ["conflict-sensor"]
    entry.mechanisms = []
    return entry


def test_dogfood_real_corpus_surfaces_high_confidence_contradiction(dogfood_index):
    """A recall-first proposal surfaces a >= 0.85 not-tenable finding on the REAL corpus.

    The §1.2 real-corpus P10 proof through the full live stack. HARD-asserts:
      (1) the run did not degrade (else a loud environmental skip);
      (2) at least one candidate reached the judge (retrieval + judgment fired on the real
          corpus) — deterministic given fixed embeddings;
      (3) at least one >= 0.85 not-tenable finding surfaced — the sensor flagged a genuine
          contradiction at high precision (``findings`` are, by construction, exactly the
          not-tenable-at-confidence pairs).
    The judge is stochastic but the proposal is a blatant CONF-D4 inversion, so a temp-0.3
    SONNET reliably rates it not-tenable; a genuine miss (never an outage) is worth a red.
    """
    store, provider, vstore, _ = dogfood_index
    proposal = _recall_first_proposal()

    with skip_on_embed_quota():
        result = run_conflict_check(
            proposal,
            embed_provider=provider,
            vector_store=vstore,
            store=store,
            judge=make_live_judge(),
        )

    # (1) Any substrate/judge degradation is environmental — skip loudly, never red.
    if isinstance(result, Unavailable):
        pytest.skip(
            f"live dogfood degraded to Unavailable(reason={result.reason.value}) — "
            f"environmental (embed/vector-store quota or the SONNET judge timeout/5xx/quota); "
            f"NOT a code defect. detail: {result.detail}"
        )

    # A human-readable trace for the test log (and IMPLEMENTATION_NOTES observation).
    judged = ", ".join(
        f"{p.candidate.slug}(sim={p.candidate.score:.3f}, "
        f"tenable={p.judgment.tenable_together}, conf={p.judgment.confidence:.2f}"
        f"{', SURFACED' if p.surfaced else ''})"
        for p in result.judged_pairs
    )
    print(f"\n[dogfood] judged {len(result.judged_pairs)} candidate(s): {judged}")
    print(f"[dogfood] findings: {[(f.slug, round(f.confidence, 2)) for f in result.findings]}")

    # (2) Retrieval + judgment fired on the real corpus.
    assert result.judged_pairs, (
        "no candidate reached the judge on the real corpus — the recall-first proposal "
        "retrieved nothing above the floor (a retrieval/floor regression, not judge jitter)."
    )

    # (3) A genuine contradiction surfaced at high precision (>= the surface threshold).
    assert result.findings, (
        "no >= 0.85 not-tenable finding surfaced on the real corpus for a proposal that "
        "blatantly inverts CONF-D4 (conflict-surface-threshold-085). Judged pairs: "
        f"{judged}"
    )
    for finding in result.findings:
        assert finding.confidence >= CONFLICT_SURFACE_THRESHOLD  # findings are gated by construction
