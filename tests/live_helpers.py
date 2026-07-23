"""Shared robustness helpers for the ``*_live.py`` suites (Phase r2).

The live integration suites make real Gemini/Anthropic calls and exercise the
real ``mitos`` binary as a subprocess. Three *environmental* (non-code) conditions
otherwise paint the suite red and train the team to ignore live-red — which masks
a real future regression (P10/P16: live-red must stay trustworthy):

1. The Gemini free-tier daily embed quota is spent → ``429 RESOURCE_EXHAUSTED``.
2. The globally-installed (pipx) ``mitos`` binary lags the source under test, so a
   subprocess test asserting V1b-new behaviour fails cryptically (``0 == 1``).
3. The Gemini free-tier daily GENERATIVE enrichment quota is spent (a SEPARATE
   bucket from embed) or the model 503s under high demand → ``cmd_sync`` swallows the
   wrapped ``SynthesisError`` and skips the entry, so a corruption-rebuild test
   (m5) reds as a cryptic ``AssertionError`` (fewer than the expected committed
   decisions), with no ``EmbeddingError`` ever propagating.

These helpers degrade each condition to a **LOUD** ``pytest.skip`` — a named,
actionable cause, never a silent skip (the invisible-failure class PATTERNS.md
forbids). Every catch is scoped **narrowly** to the specific signature so a real
defect still fails red.

The logic is single-sourced here (DRY) because both ``test_scenarios_live.py`` and
``test_integration_live.py`` need it; a future change to the 429 signature or the
skip reason touches one place. The bare-module import (``from live_helpers import
…``) resolves because ``tests/`` is on ``sys.path`` under pytest's default prepend
mode (no ``tests/__init__.py``) — proven by the existing ``from conftest import
STORE_REBUILD_QUARANTINE`` at ``tests/test_store.py``.
"""

import contextlib
import os
import subprocess
from typing import Optional, Tuple

import pytest
from google import genai

from mitos import __version__ as SOURCE_VERSION
from mitos.errors import EmbeddingError, SynthesisError
from mitos.parser import ParsedEntry
from mitos.sync import run_sync_enrichment

# Loud, actionable skip reason naming the quota + retry + that it is environmental.
EMBED_QUOTA_SKIP_REASON = (
    "Gemini free-tier embed quota spent (429 RESOURCE_EXHAUSTED — limit 1000/day "
    "on gemini-embedding-2, retry ~55s). Environmental, resets daily; NOT a code "
    "defect (PATTERNS.md). Live embed suite not exercisable until the quota resets."
)


# ---------------------------------------------------------------------------
# Embed-quota (429) robustness
# ---------------------------------------------------------------------------

def _is_embed_quota_exhausted(exc: BaseException) -> bool:
    """Return whether an exception carries the Gemini embed-quota (429) signature.

    Narrow by design: matches only the quota/429 signature, so every other
    ``EmbeddingError`` (auth, malformed response, network) falls through
    unchanged and still fails red. ``EmbeddingError`` wraps the genai error as
    ``f"Gemini embedding API call failed: {str(e)}"``, and the genai 429 string
    carries both ``429`` and ``RESOURCE_EXHAUSTED``, so matching ``str(exc)`` is
    reliable.

    Args:
        exc: The exception to inspect.

    Returns:
        True if the exception message names the 429/quota signature.
    """
    msg = str(exc)
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg


@contextlib.contextmanager
def skip_on_embed_quota():
    """Degrade a propagating embed-quota 429 inside the block to a loud skip.

    For the s1/x1 boundary, where the 429 hits an **explicit**
    ``provider.get_embedding(...)`` in the test body and raises an
    ``EmbeddingError`` loudly. Catches **only** ``EmbeddingError``, skips **only**
    on the quota signature, and re-raises everything else (including a non-quota
    ``EmbeddingError``). Zero extra API cost — it guards the call that already
    raises.

    Yields:
        None — control returns to the guarded block.

    Raises:
        Skipped: When the guarded call raises a quota-signature ``EmbeddingError``.
    """
    try:
        yield
    except EmbeddingError as exc:
        if _is_embed_quota_exhausted(exc):
            pytest.skip(EMBED_QUOTA_SKIP_REASON)
        raise


def skip_if_embed_quota_exhausted(embed_provider) -> None:
    """Probe an embed provider and skip loudly if the quota is exhausted.

    For the record-pause boundary, where production swallows the embed error
    (the pre-commit near-dup review fails OPEN) so the 429 never reaches the
    test as an ``EmbeddingError`` — it surfaces only as a ``"created"`` result
    where a pause was expected (with a ``neighbor_review_unavailable`` notice,
    or without one when the seed vector never upserted). This fires an
    **active** probe (an explicit embed call the test can see 429) **only on
    that degraded path**, so it costs an extra embed only when the test is
    already degraded (zero extra cost on the healthy path). If the pause is
    absent for a *non-quota* reason (a real bug), the probe succeeds (no 429)
    → returns → the caller's assertion fires loudly.

    Args:
        embed_provider: The provider to probe (a ``GeminiEmbeddingProvider`` or
            ``None``). ``None`` returns without skipping — an absent provider
            can't be probed, so the caller's assertion is left to fire rather
            than silently masked.

    Returns:
        None.

    Raises:
        Skipped: When the probe raises a quota-signature ``EmbeddingError``.
    """
    if embed_provider is None:
        return  # can't probe → let the caller's assert fire (don't silently skip)
    with skip_on_embed_quota():
        embed_provider.get_embedding(
            "mitos r2 live-suite embed-quota probe", is_query=True
        )


# ---------------------------------------------------------------------------
# Generative-enrichment-quota (429/503) robustness
# ---------------------------------------------------------------------------

# Loud, actionable skip reason — names the GENERATIVE quota, that it is a SEPARATE
# bucket from embed, environmental, resets daily, and NOT a code defect.
ENRICHMENT_QUOTA_SKIP_REASON = (
    "Gemini free-tier GENERATIVE enrichment quota spent / model unavailable "
    "(429 RESOURCE_EXHAUSTED on generate_content_free_tier_requests, or 503 "
    "UNAVAILABLE 'model experiencing high demand' on gemini-3.1-flash-lite). A "
    "SEPARATE bucket from the embed quota (confirmed independent). Environmental, "
    "resets daily; NOT a code defect (PATTERNS.md). The m5 corruption-rebuild test "
    "is not exercisable until the generative quota/availability recovers."
)


def _is_enrichment_quota_exhausted(exc: BaseException) -> bool:
    """Return whether an exception carries the generative-quota / unavailable signature.

    Narrow by design: matches only 429 / 503 / RESOURCE_EXHAUSTED / UNAVAILABLE —
    the signatures the generative enrichment model raises under a spent quota or a
    transient outage. Every other ``SynthesisError`` (auth, malformed JSON, network)
    falls through unchanged and still fails red. ``run_sync_enrichment`` wraps the
    genai error as ``f"LLM enrichment call failed: {str(e)}"`` and the genai
    429/503 string carries the signature, so matching ``str(exc)`` is reliable.

    Wider than ``_is_embed_quota_exhausted``'s 429/RESOURCE_EXHAUSTED set because
    the generative model also 503s under high demand (the Phase-1-boundary flake
    was a 503 UNAVAILABLE, not a 429); keep the two matchers separate.

    Args:
        exc: The exception to inspect.

    Returns:
        True if the exception message names the 429/503/quota/unavailable signature.
    """
    msg = str(exc)
    return (
        "RESOURCE_EXHAUSTED" in msg
        or "UNAVAILABLE" in msg
        or "429" in msg
        or "503" in msg
    )


def _enrichment_probe_entry() -> ParsedEntry:
    """Build a minimal decision ParsedEntry to drive the enrichment probe call.

    ``run_sync_enrichment`` reads only ``slug``/``axiom``/``rejected_paths``/
    ``mechanisms``/``scope``/``context``; the rest default safely on a fresh
    ``ParsedEntry`` (``mechanisms``/``scope`` MUST stay ``list[str]`` — they are
    ``','.join``-ed in the prompt).

    Returns:
        A populated ``ParsedEntry`` suitable for a single enrichment probe call.
    """
    pe = ParsedEntry(kind="decision", slug="mitos-r3-enrichment-probe", line_start=1, line_end=1)
    pe.axiom = "Probe the generative enrichment model for live-quota availability."
    pe.core_axiom = pe.axiom
    pe.rejected_paths = "None."
    pe.mechanisms = ["probe"]
    pe.scope = ["test"]
    pe.context = "r3 live-suite enrichment-quota active probe."
    return pe


def skip_if_enrichment_quota_exhausted(genai_client=None) -> None:
    """Probe the generative enrichment model; skip loudly if its quota is exhausted.

    The swallowed-boundary twin of :func:`skip_if_embed_quota_exhausted`, but against
    the GENERATIVE enrichment model. m5 calls ``cmd_sync``, which on a 429/503 swallows
    the wrapped ``SynthesisError`` in its F1 branch (``sync.py:609-626``) and skips the
    entry — so the failure never reaches the test as an exception; it surfaces only as
    fewer-than-2 committed decisions (an ``AssertionError`` downstream). Fire this ONLY
    on that degraded path (``len(active_decisions) < 2``): it exercises the exact
    production call (``run_sync_enrichment``) and, if the live generative quota is spent,
    raises a quota-signature ``SynthesisError`` -> loud skip. If the call SUCCEEDS (quota
    healthy) the shortfall is a real bug -> return -> the caller's assertion fires loudly.
    Zero extra API cost on the healthy path (gated behind the ``< 2`` check).

    Args:
        genai_client: A genai client (or a test fake exposing
            ``models.generate_content``). ``None`` -> construct one from
            ``GEMINI_API_KEY``; if no key is set, return without skipping (can't probe
            -> let the caller's assert fire rather than silently mask it).

    Returns:
        None.

    Raises:
        Skipped: When the probe raises a quota-signature ``SynthesisError``.
        SynthesisError: When the probe raises a non-quota ``SynthesisError`` (a real
            enrichment bug, never silently skipped).
    """
    if genai_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return  # no key -> can't probe -> let the caller's assert fire (don't mask)
        genai_client = genai.Client(api_key=api_key)
    try:
        run_sync_enrichment(genai_client, _enrichment_probe_entry(), [])
    except SynthesisError as exc:
        if _is_enrichment_quota_exhausted(exc):
            pytest.skip(ENRICHMENT_QUOTA_SKIP_REASON)
        raise


# ---------------------------------------------------------------------------
# Stale-global-binary (version-lag) robustness
# ---------------------------------------------------------------------------

def _parse_semver(text: str) -> Optional[Tuple[int, ...]]:
    """Parse a version string into an integer tuple, tolerating a name prefix.

    Handles both producer formats: the global probe ``mitos --version`` emits
    ``"mitos 0.4.0\\n"`` (name-prefixed, trailing newline); ``SOURCE_VERSION`` is
    bare ``"0.4.0"``. Strips, takes the last whitespace token, splits on ``.``.

    Args:
        text: The raw version text (e.g. ``"mitos 0.4.0"`` or ``"0.3.2"``).

    Returns:
        A tuple of ints (e.g. ``(0, 4, 0)``), or ``None`` on an empty string or a
        non-integer token.
    """
    tokens = text.strip().split()
    if not tokens:
        return None
    parts = tokens[-1].split(".")
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


def _is_stale(global_text: str, source_text: str) -> bool:
    """Return whether a global binary version lags the source version.

    Args:
        global_text: The global binary's reported version (e.g. ``"mitos 0.3.2"``).
        source_text: The source version under test (e.g. ``"0.4.0"``).

    Returns:
        True only if both parse and the global version is strictly behind the
        source. ``False`` if either is unparseable (uncertain → don't skip) or
        the global is equal/ahead.
    """
    g = _parse_semver(global_text)
    s = _parse_semver(source_text)
    if g is None or s is None:
        return False
    return g < s


def skip_if_global_mitos_stale(mitos_bin: str) -> None:
    """Skip loudly if the resolved global ``mitos`` binary lags the source.

    Version-GATED: when the versions **match** (or the global is ahead), this does
    NOT skip → the test still runs against the real binary, preserving its
    packaging-drift-catching value. Only when the global is strictly behind the
    source does it skip with a named, actionable reason. ``mitos --version`` is
    argparse ``action="version"`` → prints ``mitos X.Y.Z`` to stdout, exit 0.

    Args:
        mitos_bin: Path or name of the ``mitos`` binary to probe (typically
            ``shutil.which("mitos") or "mitos"``).

    Returns:
        None.

    Raises:
        Skipped: When the global binary's version is strictly behind
            ``SOURCE_VERSION``.
    """
    try:
        out = subprocess.run(
            [mitos_bin, "--version"], capture_output=True, text=True, timeout=10
        )
    except Exception:
        return  # can't probe → run the test (uncertainty must not mask it)
    text = out.stdout or out.stderr or ""
    if _is_stale(text, SOURCE_VERSION):
        pytest.skip(
            f"global mitos {text.strip()!r} lags source {SOURCE_VERSION} — run "
            f"`pipx install --force git+https://github.com/dovahkiin-v/mitos`. "
            f"Environmental install-lag, not a code defect."
        )
