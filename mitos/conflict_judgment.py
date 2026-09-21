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

The executor's job is narrow (plan D2): make the one batched tenability call via tool-use
(``tool_choice=tool``), cap it hard, measure it, and hand back a
:class:`~mitos.conflict.JudgmentExecution` (serialized verdict array + batch_id + usage +
elapsed) — or a typed :class:`~mitos.conflict.Unavailable` on a timeout, any Anthropic
error, or a truncated response (**fail-open**: it never raises past the seam, never blocks
a commit). A failure decided from a response that arrived and was billed comes back as a
:class:`BilledUnavailable`, which carries that response's usage so the corpus check can
record the spend (B7). The verdict array is serialized into ``raw_text`` as JSON, so both
``parse_judgment_response`` consumers are untouched.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
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
from mitos.models import accepts_temperature, get_model_id

# The model family+tier alias (P19 — never a raw versioned id). Rides on every
# ``JudgmentExecution`` so 5b stamps each telemetry row's ``model_alias``.
_JUDGMENT_MODEL_ALIAS = "SONNET"

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BilledUnavailable(Unavailable):
    """A judgment failure decided from a response that arrived and was billed.

    Returned for a truncated response, a response with no ``tool_use`` block, and a
    tool call without its ``verdicts`` key. ``reason`` and ``detail`` are exactly what a
    plain :class:`~mitos.conflict.Unavailable` would carry, so every consumer that tests
    ``isinstance(result, Unavailable)`` and switches on ``reason`` is unchanged. The
    exhausted ladder and a refusal stay plain: no usage object came back for either.

    The corpus check reads ``billed`` by that attribute name without importing this
    module (``check._billed_execution``), so the name is a join key across modules.

    Attributes:
        billed: The response's usage, ``stop_reason``, ``batch_id`` and alias, as a
            :class:`~mitos.conflict.JudgmentExecution`. Its ``raw_text`` is ``""``: it
            rides only inside this failure, which the corpus loop's ``Unavailable``
            branch takes first, so it never reaches the parser.
    """

    billed: JudgmentExecution


def _execution_from(
    message: Any, *, raw_text: str, batch_id: str, elapsed_ms: int
) -> JudgmentExecution:
    """Builds the execution record for a response that arrived, success or failure.

    The one usage read, so the success path and the billed failures cannot drift. A
    ``None`` usage field reads as 0: a response did arrive, so an absent cache field is
    caching off, a true zero. A fake message must set ``usage`` to real ints; a
    ``MagicMock`` usage would put mocks in the token fields.

    Args:
        message: The Anthropic ``Message`` that came back.
        raw_text: The serialized verdict array, or ``""`` for a billed failure.
        batch_id: The executor's minted batch id.
        elapsed_ms: The call's wall-clock time across the ladder.

    Returns:
        The :class:`~mitos.conflict.JudgmentExecution` for that response.
    """
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
        stop_reason=message.stop_reason,
    )


# Escalating backoff for transient API errors (429, 5xx, timeouts). Three fast
# retries, then slower ones to wait out rate-limit windows. The last two 60s
# waits are the final attempt — if the quota is exhausted, we stop loudly.
_RETRY_BACKOFFS_S: Tuple[float, ...] = (1, 1, 1, 10, 20, 30, 60, 60)


def _is_transient(exc: Exception) -> bool:
    """Reports whether retrying could plausibly succeed (timeouts, connection loss, 429, 5xx).

    Any other status is the API refusing the request itself — a rejected parameter, a
    revoked key, an unknown model id — and nine attempts buy the same refusal after
    ~183s of sleeping (measured 2026-09-18: 186s per batch on a 400). An exception with
    no status keeps the benefit of the doubt the ladder always gave it.
    """
    if isinstance(exc, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        return True
    return status == 429 or status >= 500


# Defence-in-depth budget for the tool-use response. The tool schema bounds shape;
# this bounds length. Measured: tool-use verdicts emit ~160 tokens/verdict (812 for 5),
# so 2000 accommodates the schema output with margin. The parser still rejects a
# truncated response — the tool_choice=tool constraint does not guarantee completeness
# when `stop_reason` is `max_tokens`.
_JUDGMENT_MAX_TOKENS = 2000

# The tool schema the judge is forced to call, built per batch. Property order
# preserves the CONF-D3 chain-of-thought lever: rationale BEFORE tenable_together, so
# the model reasons before it rules.
#
# ``slug`` is an enum of exactly the batch's candidate slugs and the tool is ``strict``,
# so the API enforces the echo by constrained decoding rather than by instruction.
# Measured 2026-09-19 (bridge/Mitos/RESULT-judge-parse-failure-20260919.md): at
# temperature 0.3 the judge echoed one slug with a segment spliced in from that
# candidate's own axiom prose, five runs out of six, and the parser rightly refused the
# batch each time — billed, unpersisted, and re-bought on the next run. Fencing the
# identifier makes a mangled echo impossible; it does not touch how the verdict is
# reached, which is why it rides no CONFLICT_PROMPT_VERSION bump (Vinga's ruling,
# 2026-09-19). Two costs, both accepted: the tool block now varies per batch, so
# nothing ahead of it in the request is prefix-cacheable while the enum is per batch
# (caching is already declined for check — see the ADR
# check-prompt-caching-declined-sub-minimum-prefix); and strict mode compiles a grammar
# per distinct schema, i.e. once per batch.
_VERDICT_TOOL_NAME = "record_verdicts"


def verdict_tool(candidate_slugs: "Sequence[str]") -> "Dict[str, Any]":
    """Builds the strict ``record_verdicts`` tool definition fenced to one batch.

    Args:
        candidate_slugs: The batch's candidate slugs, verbatim, in candidate order — the
            list the caller will parse the verdicts against.

    Returns:
        The tool definition dict for ``messages.create(tools=[...])``: ``strict`` with
        every object closed (``additionalProperties: false``) and every property
        required, and ``slug`` constrained to ``enum: candidate_slugs``.

    Raises:
        ValueError: If the batch is empty or holds a duplicate slug — neither can be
            fenced 1:1, and the parser could not align such a batch anyway. Raised
            before any request is built, so nothing is spent.
    """
    slugs = list(candidate_slugs)
    if not slugs:
        raise ValueError("cannot fence an empty candidate batch")
    if len(set(slugs)) != len(slugs):
        raise ValueError("cannot fence a candidate batch holding a duplicate slug")
    return {
        "name": _VERDICT_TOOL_NAME,
        "description": (
            "Record the tenability verdicts for every candidate in this batch."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "verdicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "slug": {"type": "string", "enum": slugs},
                            "rationale": {"type": "string"},
                            "tenable_together": {"type": "boolean"},
                            "confidence": {"type": "number"},
                        },
                        "required": [
                            "slug",
                            "rationale",
                            "tenable_together",
                            "confidence",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["verdicts"],
            "additionalProperties": False,
        },
    }


def execute_judgment(
    prompt: RenderedPrompt,
    *,
    client: "anthropic.Anthropic",
    timeout_s: float = CONFLICT_LLM_TIMEOUT_S,
    model_id: Optional[str] = None,
) -> "JudgmentExecution | Unavailable":
    """Runs one batched SONNET tenability call via tool-use; returns verdicts + metrics, or a typed failure.

    Forces the model to call the ``record_verdicts`` tool (``tool_choice=tool``), reads
    the verdict array from the tool_use block's ``input``, and serializes it into
    ``raw_text`` as a JSON array — so both ``parse_judgment_response`` consumers (the
    corpus path in ``check.py`` and the sync path in ``conflict.py``) are untouched.
    The tool is built per batch by :func:`verdict_tool`: strict, with ``slug`` fenced to
    ``prompt.candidate_slugs``, so the echo the parser aligns on cannot be mangled.

    SDK retries are disabled (``max_retries=0``) so ``CONFLICT_LLM_TIMEOUT_S`` is a
    true wall-clock ceiling per attempt. Transient errors (429, 5xx, timeouts,
    connection loss — :func:`_is_transient`) are retried with an escalating backoff
    ladder (``_RETRY_BACKOFFS_S``): 3×1s, then 10s, 20s, 30s, 60s, 60s. If every
    attempt fails, the ``Unavailable`` detail names the error class, the attempt count,
    and the total elapsed time — so the cause is diagnosable without reading logs.

    Fail-open (plan D4): an exhausted ladder maps to ``Unavailable(JUDGMENT_TIMEOUT)``;
    any other 4xx returns at once as ``Unavailable(JUDGMENT_REJECTED)`` with the
    API's own message in the detail. ``stop_reason='max_tokens'`` maps to
    ``Unavailable(JUDGMENT_TRUNCATED)`` — checked BEFORE touching ``message.content``,
    since a truncated forced-tool response can carry an incomplete or absent
    ``tool_use`` block. The executor never raises past this seam and never blocks the
    commit.

    ``temperature`` is sent only to a model that accepts it
    (:func:`mitos.models.accepts_temperature`) — the models that reject sampling
    parameters answer every call carrying one with a 400.

    Args:
        prompt: The rendered judgment prompt (from 3a's ``render_judgment_prompt``); its
            ``system`` is passed as the cache-anchored prefix, its ``user`` as the single
            user-message content.
        client: The injected Anthropic client (5a constructs the real one). Keyword-only.
        timeout_s: The hard per-call wall-clock cap in seconds (default
            ``CONFLICT_LLM_TIMEOUT_S``). Keyword-only.
        model_id: The resolved versioned id for ``_JUDGMENT_MODEL_ALIAS``, taken
            off the calling workspace's ``config.env`` by whichever orchestrator
            bound the client (2c). ``None`` falls back to the baseline for the
            alias — an override reaches this call only by being passed, because
            the model registry reads no process environment. Keyword-only.

    Returns:
        A :class:`~mitos.conflict.JudgmentExecution` (raw text + batch_id + usage + elapsed)
        on success, or an :class:`~mitos.conflict.Unavailable` naming why not —
        ``JUDGMENT_TIMEOUT`` (ladder exhausted), ``JUDGMENT_REJECTED`` (the request
        was refused), ``JUDGMENT_TRUNCATED`` or ``JUDGMENT`` (the one response that
        came back was unusable). The last two are a :class:`BilledUnavailable`
        carrying that response's usage; the first two are plain.

    Raises:
        ValueError: If ``prompt.candidate_slugs`` cannot be fenced (empty, or a
            duplicate slug) — a caller defect, refused before any spend, like the
            pin-mismatch guards in ``check.py``.
    """
    # The fence first — a batch that cannot be fenced is refused before any spend.
    tool = verdict_tool(prompt.candidate_slugs)

    # Mint the batch id up front (W8) — one per batched call, shared by every
    # ``conflict_checks`` row 5b writes for this batch. A plain unique ``str``.
    batch_id = uuid4().hex

    resolved_model = (
        model_id if model_id is not None
        else get_model_id(_JUDGMENT_MODEL_ALIAS)
    )
    create_kwargs = dict(
        model=resolved_model,
        max_tokens=_JUDGMENT_MAX_TOKENS,
        system=prompt.system,
        messages=[{"role": "user", "content": prompt.user}],
        tools=[tool],
        tool_choice={"type": "tool", "name": _VERDICT_TOOL_NAME},
    )
    if accepts_temperature(resolved_model):
        create_kwargs["temperature"] = CONFLICT_JUDGMENT_TEMPERATURE

    last_error: Optional[Exception] = None
    rejected: Optional[Exception] = None
    attempts = 0
    started = time.perf_counter()

    for attempt_backoff in (0, *_RETRY_BACKOFFS_S):
        if attempt_backoff > 0:
            _log.info(
                "judgment retry %d/%d after %s — waiting %.0fs",
                attempts, len(_RETRY_BACKOFFS_S), last_error, attempt_backoff,
            )
            time.sleep(attempt_backoff)
        attempts += 1
        try:
            message = client.with_options(
                max_retries=0, timeout=timeout_s
            ).messages.create(**create_kwargs)
            break  # success
        except anthropic.AnthropicError as exc:
            if not _is_transient(exc):
                rejected = exc
                break  # the API refused the request; retrying buys the same refusal
            last_error = exc
    else:
        total_s = time.perf_counter() - started
        return Unavailable(
            reason=ConflictUnavailableReason.JUDGMENT_TIMEOUT,
            detail=(
                f"judgment failed after {attempts} attempts over {total_s:.0f}s — "
                f"last error: {last_error}"
            ),
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    # Returned past the ladder, not from inside it: a refusal is decided from the one
    # error response that came back, which is what files it first-attempt.
    if rejected is not None:
        status = getattr(rejected, "status_code", None)
        return Unavailable(
            reason=ConflictUnavailableReason.JUDGMENT_REJECTED,
            detail=(
                f"judgment rejected (HTTP {status}) on attempt {attempts} "
                f"for model {resolved_model!r}: {rejected}"
            ),
        )

    def billed() -> JudgmentExecution:
        return _execution_from(
            message, raw_text="", batch_id=batch_id, elapsed_ms=elapsed_ms
        )

    # Truncation check BEFORE touching content — a forced-tool response truncated at
    # max_tokens can carry an incomplete or absent tool_use block.
    if message.stop_reason == "max_tokens":
        return BilledUnavailable(
            reason=ConflictUnavailableReason.JUDGMENT_TRUNCATED,
            detail=(
                f"judgment truncated: stop_reason='max_tokens' "
                f"(budget={_JUDGMENT_MAX_TOKENS})"
            ),
            billed=billed(),
        )

    # Extract the tool_use block by type, not by position.
    tool_block = None
    for block in message.content:
        if block.type == "tool_use":
            tool_block = block
            break
    if tool_block is None:
        return BilledUnavailable(
            reason=ConflictUnavailableReason.JUDGMENT,
            detail="no tool_use block in response",
            billed=billed(),
        )

    # A forced tool call is not a guarantee of the schema's required key; a payload
    # without it is a malformed batch, returned typed like the absent block above.
    if "verdicts" not in (tool_block.input or {}):
        return BilledUnavailable(
            reason=ConflictUnavailableReason.JUDGMENT,
            detail="tool_use block carries no 'verdicts' key",
            billed=billed(),
        )

    # Serialize the verdicts array into raw_text so both parse_judgment_response
    # consumers (check.py corpus path AND conflict.py sync path) are untouched.
    raw_text = json.dumps(tool_block.input["verdicts"])
    return _execution_from(
        message, raw_text=raw_text, batch_id=batch_id, elapsed_ms=elapsed_ms
    )


def make_judgment_executor(
    client: "anthropic.Anthropic",
    *,
    model_id: Optional[str] = None,
) -> "Callable[[RenderedPrompt], JudgmentExecution | Unavailable]":
    """Binds a client into the one-arg ``judge`` callable the facade expects (the 5a seam).

    5a calls this once with the constructed Anthropic client and passes the returned callable
    as ``run_conflict_check(..., judge=...)``. The closure keeps the facade's ``judge`` a
    clean one-arg function of a :class:`~mitos.conflict.RenderedPrompt`, so the facade never
    imports this module or touches the SDK (plan D1) — and stays trivially testable with a
    plain fake function (no SDK mock).

    Args:
        client: The Anthropic client to bind (5a constructs it, e.g. with ``max_retries=0``).
        model_id: The resolved versioned model id to bind alongside it (2c) — the
            id the calling workspace's ``config.env`` resolved for
            ``_JUDGMENT_MODEL_ALIAS``, or ``None`` for the baseline. It rides the
            closure rather than the facade, so the facade's ``judge`` stays a
            one-arg function of a ``RenderedPrompt``.

    Returns:
        A one-arg callable ``(RenderedPrompt) -> JudgmentExecution | Unavailable`` that drives
        :func:`execute_judgment` with the bound client.
    """

    def judge(prompt: "RenderedPrompt") -> "JudgmentExecution | Unavailable":
        return execute_judgment(prompt, client=client, model_id=model_id)

    return judge
