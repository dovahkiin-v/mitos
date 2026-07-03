"""The Conflict sensor's judgment executor — the one live Anthropic SONNET call (Phase 3b).

This is the **single module** in the Conflict pipeline that imports ``anthropic`` at
module scope, deliberately quarantined here so the Tier-1 leaf ``mitos.conflict`` stays
dependency-free (the dep-free subprocess guard in ``test_conflict_constants.py`` asserts
``anthropic`` never lands in ``sys.modules`` on ``import mitos.conflict``). The facade
(:func:`mitos.conflict.run_conflict_check`) receives the executor as an injected ``judge``
callable and names this module nowhere — the only real import edge is
``conflict_judgment → conflict`` (this module imports the boundary types + constants FROM
the leaf), never the reverse (plan D1).

**Tier 2 (logic).** Imports Tier-1 (`conflict`, `models`) + `anthropic`. Imported by
Tier-3 orchestration (5a's sync surface), never by the leaf.

The executor's job is narrow (plan D2): make the one batched tenability call, cap it hard,
measure it, and hand back a :class:`~mitos.conflict.JudgmentExecution` (raw text + batch_id
+ usage + elapsed) — or a typed :class:`~mitos.conflict.Unavailable` on a timeout or any
Anthropic error (**fail-open**: it never raises past the seam, never blocks a commit). The
raw text is parsed by the *facade* (3a's :func:`~mitos.conflict.parse_judgment_response`),
not here, so ``candidate_slugs`` — the parse's realignment key — stays facade-side.
"""

from __future__ import annotations

import time
from typing import Callable
from uuid import uuid4

import anthropic

from mitos.conflict import (
    CONFLICT_JUDGMENT_TEMPERATURE,
    CONFLICT_LLM_TIMEOUT_S,
    ConflictUnavailableReason,
    JudgmentExecution,
    RenderedPrompt,
    Unavailable,
)
from mitos.models import get_model_id

# The model family+tier alias (P19 — never a raw versioned id). Rides on every
# ``JudgmentExecution`` so 5b stamps each telemetry row's ``model_alias``.
_JUDGMENT_MODEL_ALIAS = "SONNET"

# Bounds the JSON-array OUTPUT (≤ CONFLICT_TOP_K rationale+verdict objects), not the input.
# Sized for the worst-case 5-candidate batch's rationales; a generous cap, cheap because
# output is what is billed and the judge emits at most K short objects.
_JUDGMENT_MAX_TOKENS = 2000


def execute_judgment(
    prompt: RenderedPrompt,
    *,
    client: "anthropic.Anthropic",
    timeout_s: float = CONFLICT_LLM_TIMEOUT_S,
) -> "JudgmentExecution | Unavailable":
    """Runs one batched SONNET tenability call; returns raw text + metrics, or a typed failure.

    Mirrors the importer's ``messages.create`` precedent (``importer.py:56/65``) plus the
    three additions 3b owns: ``system=prompt.system`` (the RF-3 cache anchor), reading
    ``message.usage.*``, and a hard ``timeout`` cap. Retries are disabled via
    ``client.with_options(max_retries=0, timeout=timeout_s)`` so ``CONFLICT_LLM_TIMEOUT_S`` is
    a **true** wall-clock ceiling — the SDK defaults to ``max_retries=2``, which would burn
    ~3×``timeout_s`` + backoff and defeat the P14 cap (``max_retries`` is NOT a
    ``messages.create`` kwarg in anthropic 0.109.1 — only ``timeout`` is — so it MUST be set on
    the client / via ``with_options``). 5b's aggregate breaker owns the retry-vs-trip policy,
    not the SDK.

    Fail-open (plan D4): an :class:`~anthropic.APITimeoutError` and the broader
    :class:`~anthropic.AnthropicError` (rate-limit, 5xx, connection) both map to
    ``Unavailable(JUDGMENT_TIMEOUT)`` — the same surface disposition (CONF-D10/D5), the
    ``detail`` string discriminating the exact cause for logs. The executor never raises past
    this seam and never blocks the commit.

    Args:
        prompt: The rendered judgment prompt (from 3a's ``render_judgment_prompt``); its
            ``system`` is passed as the cache-anchored prefix, its ``user`` as the single
            user-message content.
        client: The injected Anthropic client (5a constructs the real one). Keyword-only.
        timeout_s: The hard per-call wall-clock cap in seconds (default
            ``CONFLICT_LLM_TIMEOUT_S``). Keyword-only.

    Returns:
        A :class:`~mitos.conflict.JudgmentExecution` (raw text + batch_id + usage + elapsed)
        on success, or an :class:`~mitos.conflict.Unavailable` with
        ``reason=JUDGMENT_TIMEOUT`` on a timeout or any Anthropic error.
    """
    # Mint the batch id up front (W8) — one per batched call, shared by every
    # ``conflict_checks`` row 5b writes for this batch. A plain unique ``str``.
    batch_id = uuid4().hex

    started = time.perf_counter()
    try:
        message = client.with_options(
            max_retries=0, timeout=timeout_s
        ).messages.create(
            model=get_model_id(_JUDGMENT_MODEL_ALIAS),
            max_tokens=_JUDGMENT_MAX_TOKENS,
            temperature=CONFLICT_JUDGMENT_TEMPERATURE,
            system=prompt.system,  # static cache-anchored prefix; cache_control OFF (RF-3).
            messages=[{"role": "user", "content": prompt.user}],
        )
    except anthropic.APITimeoutError as exc:
        # The P14 headline case — "slow AI is failed AI". Distinct detail, same disposition.
        return Unavailable(
            reason=ConflictUnavailableReason.JUDGMENT_TIMEOUT,
            detail=f"judgment call timed out after {timeout_s}s: {exc}",
        )
    except anthropic.AnthropicError as exc:
        # Rate-limit / 5xx / connection — grouped into the SAME disposition (CONF-D10/D5),
        # the detail carrying the cause. Prefer the typed SDK base over a bare ``except`` so a
        # genuine programming bug in this function isn't masked as a judge outage.
        return Unavailable(
            reason=ConflictUnavailableReason.JUDGMENT_TIMEOUT,
            detail=f"anthropic error: {exc}",
        )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    # The response text, verbatim — the facade (not the executor) parses it (D2).
    raw_text = message.content[0].text

    # Usage capture. Cache fields can be ``None`` when caching is off (RF-3, the sync
    # surface); coerce None → 0 at the read boundary so 5b's ``NOT NULL INTEGER`` columns
    # never see a None. Attribute names verified against anthropic 0.109.1.
    usage = message.usage
    return JudgmentExecution(
        raw_text=raw_text,
        batch_id=batch_id,
        model_alias=_JUDGMENT_MODEL_ALIAS,
        token_input=getattr(usage, "input_tokens", 0) or 0,
        token_output=getattr(usage, "output_tokens", 0) or 0,
        token_cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
        token_cache_creation=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        elapsed_ms=elapsed_ms,
    )


def make_judgment_executor(
    client: "anthropic.Anthropic",
) -> "Callable[[RenderedPrompt], JudgmentExecution | Unavailable]":
    """Binds a client into the one-arg ``judge`` callable the facade expects (the 5a seam).

    5a calls this once with the constructed Anthropic client and passes the returned callable
    as ``run_conflict_check(..., judge=...)``. The closure keeps the facade's ``judge`` a
    clean one-arg function of a :class:`~mitos.conflict.RenderedPrompt`, so the facade never
    imports this module or touches the SDK (plan D1) — and stays trivially testable with a
    plain fake function (no SDK mock).

    Args:
        client: The Anthropic client to bind (5a constructs it, e.g. with ``max_retries=0``).

    Returns:
        A one-arg callable ``(RenderedPrompt) -> JudgmentExecution | Unavailable`` that drives
        :func:`execute_judgment` with the bound client.
    """

    def judge(prompt: "RenderedPrompt") -> "JudgmentExecution | Unavailable":
        return execute_judgment(prompt, client=client)

    return judge
