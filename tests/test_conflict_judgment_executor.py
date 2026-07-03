"""Tests for the Conflict sensor's judgment executor (Phase 3b — the one live SONNET call).

``execute_judgment`` makes the single batched Anthropic tenability call, caps it hard
(``with_options(max_retries=0, timeout=…)``), captures usage + elapsed + a minted
``batch_id``, and returns a ``JudgmentExecution`` — OR a typed ``Unavailable(JUDGMENT_TIMEOUT)``
on a timeout or any Anthropic error (**fail-open**, never raises past the seam).
``make_judgment_executor`` binds a client into the one-arg ``judge`` callable the facade uses.

Discipline (scout brief / plan §9): SDK-faked via a plain ``MagicMock`` client passed as a
param — the SDK call is **synchronous** (no ``AsyncMock``, no ``pytest-asyncio``). The
executor takes the client as an argument, so no ``@patch("anthropic.Anthropic")`` is needed.
Anthropic exception classes need ``httpx.Request``/``httpx.Response`` to construct (see
``_req`` / the error recipes). ``ANTHROPIC_API_KEY`` is stripped — the fake client is the
only substrate.
"""

from typing import Any, Optional
from unittest.mock import MagicMock

import httpx
import pytest

import anthropic

from mitos.conflict import (
    CONFLICT_JUDGMENT_TEMPERATURE,
    CONFLICT_LLM_TIMEOUT_S,
    ConflictUnavailableReason,
    JudgmentExecution,
    RenderedPrompt,
    Unavailable,
)
from mitos.conflict_judgment import execute_judgment, make_judgment_executor
from mitos.models import get_model_id


# --------------------------------------------------------------------------- #
# Fixtures + helpers
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fake client is the only substrate — no real key reaches anything."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def _prompt() -> RenderedPrompt:
    """A minimal RenderedPrompt — the executor only reads ``.system`` and ``.user``."""
    return RenderedPrompt(
        system="SYSTEM-PREFIX", user="USER-BLOCK", prompt_version="conflict-tenability-v1"
    )


def _fake_message(
    text: str = "[]",
    *,
    input_tokens: int = 120,
    output_tokens: int = 45,
    cache_read: Optional[int] = 0,
    cache_creation: Optional[int] = 0,
) -> MagicMock:
    """A fake ``messages.create`` return — ``.content[0].text`` + a ``.usage`` with four attrs."""
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage = MagicMock(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
    )
    return msg


def _client_returning(message: MagicMock) -> MagicMock:
    """A fake client whose ``with_options(...).messages.create(...)`` returns ``message``.

    ``with_options`` returns a NEW client, so the create call lives on
    ``client.with_options.return_value.messages.create``. A ``MagicMock`` auto-creates that
    chain; we pin its return value and keep the handles for kwarg assertions.
    """
    client = MagicMock()
    client.with_options.return_value.messages.create.return_value = message
    return client


def _client_raising(exc: BaseException) -> MagicMock:
    """A fake client whose ``with_options(...).messages.create(...)`` raises ``exc``."""
    client = MagicMock()
    client.with_options.return_value.messages.create.side_effect = exc
    return client


def _req() -> httpx.Request:
    """A dummy request — the anthropic exception classes need one to construct."""
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


# --------------------------------------------------------------------------- #
# 1. Call shape — model / temperature / system / message / timeout / retries
# --------------------------------------------------------------------------- #

def test_call_shape_mirrors_importer_plus_system_timeout_and_no_retries() -> None:
    """The create call carries the right model/temp/system/user, and retries are disabled."""
    message = _fake_message()
    client = _client_returning(message)

    execute_judgment(_prompt(), client=client, timeout_s=15)

    # The hard-cap wiring rides on ``with_options`` (max_retries + timeout).
    client.with_options.assert_called_once_with(max_retries=0, timeout=15)

    # The call shape rides on the returned client's ``messages.create``.
    create = client.with_options.return_value.messages.create
    create.assert_called_once()
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == get_model_id("SONNET")  # read it, never hardcode the id.
    assert kwargs["temperature"] == CONFLICT_JUDGMENT_TEMPERATURE
    assert kwargs["system"] == "SYSTEM-PREFIX"
    assert kwargs["messages"] == [{"role": "user", "content": "USER-BLOCK"}]
    assert "max_tokens" in kwargs and isinstance(kwargs["max_tokens"], int)


def test_timeout_default_is_the_conflict_llm_timeout_constant() -> None:
    """Called without ``timeout_s``, the executor caps at ``CONFLICT_LLM_TIMEOUT_S``."""
    client = _client_returning(_fake_message())
    execute_judgment(_prompt(), client=client)
    client.with_options.assert_called_once_with(
        max_retries=0, timeout=CONFLICT_LLM_TIMEOUT_S
    )


# --------------------------------------------------------------------------- #
# 2. Happy path — the JudgmentExecution shape
# --------------------------------------------------------------------------- #

def test_happy_path_returns_execution_with_raw_text_metrics_and_batch_id() -> None:
    """A successful call yields the raw text, provenance, the four token counts, elapsed_ms."""
    message = _fake_message(
        text='[{"slug": "x", "rationale": "r", "tenable_together": true, "confidence": 0.9}]',
        input_tokens=200,
        output_tokens=60,
        cache_read=0,
        cache_creation=0,
    )
    client = _client_returning(message)

    result = execute_judgment(_prompt(), client=client)

    assert isinstance(result, JudgmentExecution)
    assert result.raw_text == message.content[0].text
    assert isinstance(result.batch_id, str) and result.batch_id  # non-empty.
    assert result.model_alias == "SONNET"
    assert result.token_input == 200
    assert result.token_output == 60
    assert result.token_cache_read == 0
    assert result.token_cache_creation == 0
    assert isinstance(result.elapsed_ms, int) and result.elapsed_ms >= 0


# --------------------------------------------------------------------------- #
# 3. Cache usage fields None → 0 (caching off, RF-3)
# --------------------------------------------------------------------------- #

def test_none_cache_usage_fields_coerce_to_zero() -> None:
    """A ``None`` cache usage field becomes 0 (5b's columns are NOT NULL INTEGER)."""
    message = _fake_message(cache_read=None, cache_creation=None)
    client = _client_returning(message)

    result = execute_judgment(_prompt(), client=client)

    assert isinstance(result, JudgmentExecution)
    assert result.token_cache_read == 0
    assert result.token_cache_creation == 0


# --------------------------------------------------------------------------- #
# 4. Timeout → Unavailable(JUDGMENT_TIMEOUT), never raises
# --------------------------------------------------------------------------- #

def test_timeout_maps_to_unavailable_judgment_timeout() -> None:
    """An ``APITimeoutError`` becomes a typed ``Unavailable(JUDGMENT_TIMEOUT)``; detail names it."""
    client = _client_raising(anthropic.APITimeoutError(_req()))

    result = execute_judgment(_prompt(), client=client, timeout_s=15)

    assert isinstance(result, Unavailable)
    assert result.reason is ConflictUnavailableReason.JUDGMENT_TIMEOUT
    assert "timed out" in result.detail.lower()


# --------------------------------------------------------------------------- #
# 5. Any Anthropic error → Unavailable(JUDGMENT_TIMEOUT), never raises
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "exc",
    [
        anthropic.RateLimitError(
            "rate limited", response=httpx.Response(429, request=_req()), body=None
        ),
        anthropic.APIConnectionError(message="connection boom", request=_req()),
    ],
)
def test_anthropic_errors_map_to_unavailable_and_never_raise(exc: BaseException) -> None:
    """Rate-limit / connection errors fail open — same disposition, cause in the detail."""
    client = _client_raising(exc)

    result = execute_judgment(_prompt(), client=client)

    assert isinstance(result, Unavailable)
    assert result.reason is ConflictUnavailableReason.JUDGMENT_TIMEOUT
    assert result.detail  # carries the cause for logs.


# --------------------------------------------------------------------------- #
# 6. batch_id uniqueness
# --------------------------------------------------------------------------- #

def test_batch_id_is_unique_per_call() -> None:
    """Two calls mint two distinct batch ids (the batch⋈checks join key, W8)."""
    client = _client_returning(_fake_message())
    first = execute_judgment(_prompt(), client=client)
    second = execute_judgment(_prompt(), client=client)
    assert isinstance(first, JudgmentExecution) and isinstance(second, JudgmentExecution)
    assert first.batch_id != second.batch_id


# --------------------------------------------------------------------------- #
# 7. make_judgment_executor — the bound one-arg seam
# --------------------------------------------------------------------------- #

def test_make_judgment_executor_binds_client_into_one_arg_callable() -> None:
    """The factory returns a one-arg ``judge`` that drives execute_judgment with the client."""
    message = _fake_message()
    client = _client_returning(message)

    judge = make_judgment_executor(client)
    result = judge(_prompt())

    assert isinstance(result, JudgmentExecution)
    # The bound client's create was driven exactly once through the seam.
    client.with_options.return_value.messages.create.assert_called_once()


def test_make_judgment_executor_propagates_unavailable() -> None:
    """A degraded call through the bound seam returns the typed Unavailable, never raises."""
    client = _client_raising(anthropic.APITimeoutError(_req()))
    judge = make_judgment_executor(client)
    result = judge(_prompt())
    assert isinstance(result, Unavailable)
    assert result.reason is ConflictUnavailableReason.JUDGMENT_TIMEOUT
