"""Tests for the google-genai duplicate-key banner suppression (AX P6).

Loop-Claude's friction: every embedding-touching command (`status`/`surface`/`record`)
printed `Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.` — an SDK
WARNING fired on every client construction when both keys are in the environment, sitting
as recurring noise right above the output an agent is parsing. Mitos installs a logging
filter (at package import) that drops ONLY that message and leaves every other SDK warning
intact.
"""

import logging

import mitos


def test_duplicate_key_banner_is_dropped(caplog) -> None:
    """The exact duplicate-key banner never reaches a handler (filtered at the logger)."""
    mitos._silence_genai_duplicate_key_warning()  # idempotent; ensure installed
    logger = logging.getLogger("google_genai._api_client")
    with caplog.at_level(logging.WARNING, logger="google_genai._api_client"):
        logger.warning("Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.")
    assert not any("Both GOOGLE_API_KEY" in r.getMessage() for r in caplog.records)


def test_other_sdk_warnings_pass_through(caplog) -> None:
    """Unrelated warnings on the same logger are untouched — we suppress one message only."""
    mitos._silence_genai_duplicate_key_warning()
    logger = logging.getLogger("google_genai._api_client")
    with caplog.at_level(logging.WARNING, logger="google_genai._api_client"):
        logger.warning("A genuinely important SDK warning.")
    assert any("genuinely important" in r.getMessage() for r in caplog.records)


def test_filter_install_is_idempotent() -> None:
    """Re-installing doesn't stack duplicate filters on the logger."""
    logger = logging.getLogger("google_genai._api_client")
    mitos._silence_genai_duplicate_key_warning()
    mitos._silence_genai_duplicate_key_warning()
    matching = [f for f in logger.filters
                if isinstance(f, mitos._DuplicateKeyWarningFilter)]
    assert len(matching) == 1
