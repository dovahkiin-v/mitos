"""Tests for the Conflict sensor's §8 constants catalog (Phase 1a).

Two contracts:
1. The five constants exist with the pinned values (the four fixed dials pinned
   exactly; the provisional similarity floor pinned only as present + in (0, 1),
   since Phase 4b recalibrates it against golden fixtures).
2. ``mitos.conflict`` is a Tier-1 leaf: importing it drags NO heavy dependency
   (``anthropic``, the Qdrant/genai clients). This pins the boundary for the whole
   vision — 2a/3b must keep those imports function-local / ``TYPE_CHECKING``-guarded.
"""

import subprocess
import sys

from mitos import conflict


# ---------------------------------------------------------------------------
# The five §8 constants
# ---------------------------------------------------------------------------

def test_fixed_constants_have_pinned_values() -> None:
    """The four first-principles-derived dials carry their §8 values exactly."""
    assert conflict.CONFLICT_SURFACE_THRESHOLD == 0.85
    assert conflict.CONFLICT_TOP_K == 5
    assert conflict.CONFLICT_JUDGMENT_TEMPERATURE == 0.3
    assert conflict.CONFLICT_LLM_TIMEOUT_S == 15


def test_similarity_floor_is_present_numeric_and_in_unit_range() -> None:
    """The provisional floor exists and is a sane (0, 1) similarity — value NOT pinned.

    It is the one corpus-empirical constant, recalibrated against golden fixtures
    in Phase 4b; pinning its exact number here would just make this test chase it.
    """
    floor = conflict.CONFLICT_SIMILARITY_FLOOR
    assert isinstance(floor, (int, float)) and not isinstance(floor, bool)
    assert 0.0 < floor < 1.0


def test_module_exposes_the_five_constants_and_the_2a_pipeline_symbols() -> None:
    """The five §8 constants plus the 2a candidate-gathering surface are all present.

    Before 2a this was ``exposes_only_the_five_constants`` (exact-set equality). 2a
    added the first pipeline stage, so the leaf is no longer constants-only and the
    exact-set form is retired (it would only chase leaf-safe stdlib imports on every
    refactor — the real "no heavy dep leaks" guard is the subprocess test below).
    This now pins the *intended* public API: the five catalog constants (unchanged)
    plus the 2a over-fetch dial, stage entry point, and three shared types. 3b extends
    this set (the judgment call); update it there, and never hide a symbol to pass it.
    """
    public = {name for name in vars(conflict) if not name.startswith("__")}
    expected_api = {
        # The five §8 constants (still exactly these — pinned by value below / above).
        "CONFLICT_SURFACE_THRESHOLD",
        "CONFLICT_TOP_K",
        "CONFLICT_JUDGMENT_TEMPERATURE",
        "CONFLICT_LLM_TIMEOUT_S",
        "CONFLICT_SIMILARITY_FLOOR",
        # The 2a candidate-gathering surface.
        "CONFLICT_OVERFETCH_LIMIT",
        "gather_candidates",
        "Candidate",
        "Unavailable",
        "ConflictUnavailableReason",
        # The 2b candidate filter + Letter-payload surface.
        "declared_strong_targets",
        "screen_candidates",
        "candidate_payload",
        # The 3a judgment render + parse surface.
        "JudgeInput",
        "RenderedPrompt",
        "Judgment",
        "judge_input_from_entry",
        "judge_input_from_node",
        "render_judgment_prompt",
        "parse_judgment_response",
        "CONFLICT_PROMPT_VERSION",
    }
    missing = expected_api - public
    assert not missing, f"conflict.py is missing intended public symbols: {sorted(missing)}"


def test_overfetch_limit_is_a_bounded_margin_above_top_k() -> None:
    """The 2a over-fetch dial is a single bounded window wider than the final top-K.

    CONF-D3/D7: the raw KNN window must exceed CONFLICT_TOP_K so S3's non-live drops
    and 2b's declared/own-slug drops cannot shadow an undeclared neighbour out of the
    final batch. Pinned as an int strictly greater than TOP_K (not the exact value —
    it is an operational tuning dial, like a config knob, not a calibrated constant).
    """
    limit = conflict.CONFLICT_OVERFETCH_LIMIT
    assert isinstance(limit, int) and not isinstance(limit, bool)
    assert limit > conflict.CONFLICT_TOP_K


# ---------------------------------------------------------------------------
# The Tier-1 dependency-free contract
# ---------------------------------------------------------------------------

def test_importing_conflict_drags_no_heavy_dependency() -> None:
    """A fresh interpreter importing mitos.conflict pulls no LLM dep.

    Run in a subprocess so the assertion sees a clean import graph (this test
    process has already imported plenty). The heavy deps whose absence is the
    Tier-1 contract are the ones 2a/3b will inject, never import at module scope:
    ``anthropic`` (the 3b judgment client) and ``google.genai`` (the embeddings
    SDK behind candidate gathering). Qdrant access is ``requests``-based (stdlib-
    adjacent, not a heavy client), so it is not part of this guard.
    """
    heavy = ["anthropic", "google", "google.genai"]
    probe = (
        "import sys; import mitos.conflict; "
        f"leaked = [m for m in {heavy!r} if m in sys.modules]; "
        "assert not leaked, 'mitos.conflict leaked heavy deps: ' + repr(leaked); "
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"dep-free import probe failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout
