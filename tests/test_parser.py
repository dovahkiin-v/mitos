"""Adversarial test suite for the Mitos strict parser.

Tests edge cases, formatting violations, HTML comment preservation, and
structural invariants to verify deterministic parsing and robust validation.
"""

import pytest
import tempfile
import os
from mitos.parser import (
    strip_html_comments,
    parse_header,
    parse_decisions_file,
    ParsedEntry
)
from mitos.errors import ParseError, ValidationError
from mitos.store import GraphStore

def test_strip_html_comments_outside_protected() -> None:
    """Verifies that comments are stripped outside but preserved inside fenced and transcript blocks."""
    raw = (
        "This is text <!-- strip me --> here.\n"
        "<!-- strip line -->\n"
        "```python\n"
        "# <!-- keep me --> in code block\n"
        "```\n"
        "[DECISION_TRANSCRIPT]\n"
        "User: <!-- keep me --> in transcript\n"
        "[/DECISION_TRANSCRIPT]\n"
    )
    cleaned = strip_html_comments(raw)
    
    # Assert line count is preserved exactly
    assert len(cleaned.splitlines()) == len(raw.splitlines())
    
    # Assert stripping outside
    assert "strip me" not in cleaned
    assert "strip line" not in cleaned
    
    # Assert preservation inside
    assert "keep me --> in code block" in cleaned
    assert "keep me --> in transcript" in cleaned


def test_parse_header_formats() -> None:
    """Tests header extraction for standard and slug-only formats."""
    # Standard format with em-dash
    slug, date, title = parse_header("## 2026-05-21 \u2014 test-slug \u2014 My Title")
    assert slug == "test-slug"
    assert date == "2026-05-21"
    assert title == "My Title"

    # Slug only
    slug, date, title = parse_header("### example-slug")
    assert slug == "example-slug"
    assert date is None
    assert title is None


def test_parse_decisions_valid_entry() -> None:
    """Tests parsing a standard, well-formed decision entry."""
    entry_text = (
        "<!-- BEGIN ENTRIES -->\n"
        "## 2026-05-19 — valid-slug — Well-formed title\n\n"
        "**Decided:** The core architecture must use pure logic cores.\n"
        "**Rejected:**\n"
        "- framework-tight-coupling — locks our code to transient dependencies\n\n"
        "**Mechanisms:** sqlite, python\n"
        "**Scope:** substrate, auth\n"
        "**Context:** Background detail goes here.\n\n"
        "[DECISION_TRANSCRIPT]\n"
        "User: Can we couple it?\n"
        "Claude: No.\n"
        "[/DECISION_TRANSCRIPT]\n"
    )
    entries = parse_decisions_file(entry_text)
    assert len(entries) == 1
    entry = entries[0]
    
    assert entry.kind == "decision"
    assert entry.slug == "valid-slug"
    assert entry.date == "2026-05-19"
    assert entry.core_axiom == "The core architecture must use pure logic cores."
    assert "framework-tight-coupling" in entry.rejected_paths
    assert entry.mechanisms == ["sqlite", "python"]
    assert entry.scope == ["substrate", "auth"]
    assert entry.context == "Background detail goes here."
    assert "User: Can we couple it?" in entry.transcript


def test_parse_decisions_missing_axiom_throws() -> None:
    """Verifies that missing required **Decided:** field throws a ValidationError on store commit."""
    entry_text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### missing-axiom\n"
        "**Rejected:** Alternative paths.\n"
    )
    entries = parse_decisions_file(entry_text)
    assert len(entries) == 1
    
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        store = GraphStore(path)
        with pytest.raises(ValidationError) as exc:
            store.commit_parsed_entry(entries[0])
        assert "missing required field '**Decided:**'" in str(exc.value)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_parse_decisions_missing_rejected_throws() -> None:
    """Verifies that missing required **Rejected:** field throws a ValidationError on store commit."""
    entry_text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### missing-rejected\n"
        "**Decided:** We will build a pure core.\n"
    )
    entries = parse_decisions_file(entry_text)
    assert len(entries) == 1
    
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        store = GraphStore(path)
        with pytest.raises(ValidationError) as exc:
            store.commit_parsed_entry(entries[0])
        assert "missing required field '**Rejected:**'" in str(exc.value)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_parse_open_question() -> None:
    """Tests parsing parked open questions with required **Questions:** fields."""
    entry_text = (
        "<!-- BEGIN ENTRIES -->\n"
        "[DECISION_PARKED: cache-stampede — needs benchmark data]\n"
        "**Questions:**\n"
        "- How does standard locking behave under 10000x load?\n"
        "- Do we need a distributed lock?\n"
    )
    entries = parse_decisions_file(entry_text)
    assert len(entries) == 1
    oq = entries[0]
    
    assert oq.kind == "open_question"
    assert oq.slug == "cache-stampede"
    assert oq.park_reason == "needs benchmark data"
    assert len(oq.questions_raised) == 2
    assert "How does standard locking behave under 10000x load?" in oq.questions_raised


def test_parse_inline_markers() -> None:
    """Verifies that inline NOTE and PARKED markers are extracted from entries."""
    entry_text = (
        "<!-- BEGIN ENTRIES -->\n"
        "## 2026-05-19 — valid-slug — Well-formed title\n"
        "**Decided:** The core architecture must use pure logic cores.\n"
        "**Rejected:** Direct coupling.\n"
        "**Context:** Background detail [NOTE: this is an inline note].\n"
        "And we also have [PARKED: what is the benchmark rate?].\n"
    )
    entries = parse_decisions_file(entry_text)
    assert len(entries) == 1
    entry = entries[0]
    
    assert entry.notes == ["this is an inline note"]
    assert entry.parked_questions == ["what is the benchmark rate?"]
