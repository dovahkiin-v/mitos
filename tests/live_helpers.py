"""Shared robustness helpers for the ``*_live.py`` suites (Phase r2).

The live integration suites make real Gemini/Anthropic calls and exercise the
real ``mitos`` binary as a subprocess. Two *environmental* (non-code) conditions
otherwise paint the suite red and train the team to ignore live-red — which masks
a real future regression (P10/P16: live-red must stay trustworthy):

1. The Gemini free-tier daily embed quota is spent → ``429 RESOURCE_EXHAUSTED``.
2. The globally-installed (pipx) ``mitos`` binary lags the source under test, so a
   subprocess test asserting V1b-new behaviour fails cryptically (``0 == 1``).

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
import subprocess
from typing import Optional, Tuple

import pytest

from mitos import __version__ as SOURCE_VERSION
from mitos.errors import EmbeddingError

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

    For the adjacency boundary, where production swallows the embed error
    (fail-silent echo) so the 429 never reaches the test as an ``EmbeddingError``
    — it surfaces only as an empty ``related`` echo. This fires an **active**
    probe (an explicit embed call the test can see 429) **only on the empty-echo
    path**, so it costs an extra embed only when the test is already degraded
    (zero extra cost on the healthy path). If the echo is empty for a *non-quota*
    reason (a real bug), the probe succeeds (no 429) → returns → the caller's
    assertion fires loudly.

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
