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
    parse_entry_stream,
    ParsedEntry,
    _normalize_mechanism_list,
    _normalize_scope_list,
    _normalize_questions_list,
)
from mitos import identity
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


def test_unknown_field_reports_true_line_range() -> None:
    """Verifies a malformed field's ParseError points at the actual offending line (C5).

    Regression: the unknown-field branch used to cite a stale scan-loop variable,
    so in multi-entry buffers it reported the file's last line instead of the bad one.
    """
    entry_text = (
        "<!-- BEGIN ENTRIES -->\n"          # line 1
        "## 2026-05-19 — slug-a — Title\n"  # line 2  (section header)
        "**Decided:** Something valid.\n"    # line 3
        "**Bogus:** not a real field.\n"     # line 4  <- the offender
        "**Rejected:** an alternative.\n"    # line 5
    )
    with pytest.raises(ParseError) as exc:
        parse_decisions_file(entry_text)
    assert exc.value.line_start == 4
    assert exc.value.line_end == 4
    assert "Bogus" in exc.value.message


def test_malformed_entry_isolated_when_collector_supplied() -> None:
    """Verifies §7.2-A: one malformed entry is recorded and skipped; the rest still parse."""
    entry_text = (
        "<!-- BEGIN ENTRIES -->\n"
        "## 2026-05-19 — bad-entry — Bad\n"
        "**Decided:** something\n"
        "**Bogus:** nope\n"
        "## 2026-05-19 — good-entry — Good\n"
        "**Decided:** a valid axiom\n"
        "**Rejected:** the rejected alternative\n"
    )
    errors: list = []
    entries = parse_decisions_file(entry_text, errors=errors)

    assert len(entries) == 1
    assert entries[0].slug == "good-entry"
    assert len(errors) == 1
    assert isinstance(errors[0], ParseError)
    assert "Bogus" in errors[0].message


def test_malformed_entry_raises_in_strict_mode() -> None:
    """Verifies the default (no collector) still hard-fails on the first malformed entry (OD1)."""
    entry_text = (
        "<!-- BEGIN ENTRIES -->\n"
        "## 2026-05-19 — bad-entry — Bad\n"
        "**Decided:** something\n"
        "**Bogus:** nope\n"
    )
    with pytest.raises(ParseError):
        parse_decisions_file(entry_text)


# ===========================================================================
# Phase 4a — parse_entry_stream (V1a core deterministic tokenizer)
#
# The new path is built UNWIRED alongside the prototype above; the prototype
# tests stay and stay green. Goldens are authored from format-spec.md §3/§4,
# quoting the canonical samples as standalone stream input.
# ===========================================================================

# The §3 decision sample (format-spec.md), quoted as a standalone entry stream
# below the sentinel. ``Source`` is omitted (tool-only) so it must parse to None.
_DECISION_SAMPLE_STREAM = (
    "<!-- BEGIN ENTRIES — newest first -->\n"
    "### example-slug\n"
    "\n"
    "**Decided:** We will use SQLite in WAL mode for the graph store.\n"
    "**Rejected:** pgvector (too heavy for local-first portfolio audience), "
    "sqlite-vec (defer to v0.2 to preserve V1 ship date).\n"
    "**Mechanisms:** sqlite, wal-mode\n"
    "**Scope:** substrate\n"
    "**Context:** We need a local-first graph that supports concurrent reads "
    "and writes gracefully.\n"
    "\n"
    "[DECISION_TRANSCRIPT]\n"
    "User: Let's use Postgres.\n"
    "Claude: That breaks the local-first requirement in P10. Let's use SQLite.\n"
    "[/DECISION_TRANSCRIPT]\n"
)

# The §4 open-question sample (format-spec.md), quoted as a standalone stream.
# Its two questions are authored inline on one line (no bullets).
_OQ_SAMPLE_STREAM = (
    "<!-- BEGIN ENTRIES — newest first -->\n"
    "### example-open-question\n"
    "\n"
    "**Topic:** Embedding model selection for v0.2 semantic surface\n"
    "**Questions:** Do we pin one embedding model or allow per-project choice? "
    "What is the re-embed cost if we switch after the corpus grows past ~1k nodes?\n"
    "**Scope:** substrate, embeddings\n"
)


def test_parse_entry_stream_decision_golden() -> None:
    """The §3 decision sample tokenizes to the expected ParsedEntry (Success #1)."""
    entries = parse_entry_stream(_DECISION_SAMPLE_STREAM, "decision")
    assert len(entries) == 1
    entry = entries[0]

    assert entry.kind == "decision"
    assert entry.slug == "example-slug"
    assert entry.axiom == "We will use SQLite in WAL mode for the graph store."
    # **Decided:** lands on the NEW axiom attribute; the prototype core_axiom is
    # left untouched by the new path (the 8a rename has not happened yet).
    assert entry.core_axiom == ""
    assert entry.mechanisms == ["sqlite", "wal-mode"]
    assert entry.scope == ["substrate"]
    assert "pgvector" in entry.rejected_paths
    assert entry.context.startswith("We need a local-first graph")
    assert entry.source is None  # tool-only; absent -> None (5a defaults to "user")
    assert entry.transcript is not None
    assert "User: Let's use Postgres." in entry.transcript
    assert "Claude: That breaks the local-first requirement" in entry.transcript


def test_parse_entry_stream_open_question_golden() -> None:
    """The §4 OQ sample tokenizes under kind="open_question" (Success #2).

    The two questions are authored inline on one line (no bullets), so the
    no-auto-split-on-? rule yields exactly ONE question item — the deterministic
    behavior the plan pins. No [DECISION_PARKED] marker is involved.
    """
    entries = parse_entry_stream(_OQ_SAMPLE_STREAM, "open_question")
    assert len(entries) == 1
    oq = entries[0]

    assert oq.kind == "open_question"
    assert oq.slug == "example-open-question"
    assert oq.topic == "Embedding model selection for v0.2 semantic surface"
    assert oq.park_reason is None  # no marker on the V1a OQ path
    assert len(oq.questions_raised) == 1
    assert oq.questions_raised[0].startswith("Do we pin one embedding model")
    assert oq.questions_raised[0].endswith("grows past ~1k nodes?")
    assert oq.scope == ["substrate", "embeddings"]


def test_parse_entry_stream_sentinel_skips_preamble() -> None:
    """A canonical sample ABOVE the sentinel yields zero entries (Success #3a)."""
    text = (
        "# decisions.md\n"
        "Some preamble prose.\n"
        "### example-slug\n"
        "**Decided:** This sample is above the sentinel and must be skipped.\n"
        "**Rejected:** none\n"
        "<!-- BEGIN ENTRIES — newest first -->\n"
    )
    assert parse_entry_stream(text, "decision") == []


def test_parse_entry_stream_no_sentinel_parses_whole_file() -> None:
    """A sentinel-less file is treated wholly as the entry stream (Success #3b)."""
    text = (
        "### no-sentinel-entry\n"
        "**Decided:** parsed because there is no sentinel\n"
        "**Rejected:** none\n"
    )
    entries = parse_entry_stream(text, "decision")
    assert len(entries) == 1
    assert entries[0].slug == "no-sentinel-entry"
    assert entries[0].axiom == "parsed because there is no sentinel"


def test_parse_entry_stream_sample_below_deleted_sentinel_surfaces() -> None:
    """A canonical sample left below a deleted sentinel surfaces as a real entry.

    Correct P6 behavior — the parser does not special-case the sample slug; once
    the sentinel is gone the sample is just an entry (Success #3c).
    """
    # Same body as the §3 sample but with no sentinel line at all.
    text = _DECISION_SAMPLE_STREAM.split("\n", 1)[1]
    entries = parse_entry_stream(text, "decision")
    assert len(entries) == 1
    assert entries[0].slug == "example-slug"


def test_parse_entry_stream_kind_is_caller_declared() -> None:
    """The same slug block parses as either kind per the caller's param (Success #7).

    No [DECISION_PARKED] marker; kind is the caller's declaration, not sniffed.
    """
    block = (
        "<!-- BEGIN ENTRIES -->\n"
        "### shared-slug\n"
        "**Topic:** a topic that also reads as prose\n"
    )
    assert parse_entry_stream(block, "decision")[0].kind == "decision"
    assert parse_entry_stream(block, "open_question")[0].kind == "open_question"


def test_parse_entry_stream_unknown_kind_raises() -> None:
    """An unknown declared kind fails loudly (caller-contract violation)."""
    with pytest.raises(ValueError):
        parse_entry_stream("### x\n**Decided:** y\n", "banana")


def test_parse_entry_stream_transcript_aware_delimiting() -> None:
    """A ##-header and a **Decided:**-shaped line inside a transcript are literal text.

    The latent prototype bug fixed here: the outer section split must suppress
    delimiter detection while in a [DECISION_TRANSCRIPT] block, and field-shaped
    lines inside it are captured verbatim, not parsed (Success #4, Gotcha #1).
    """
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### real-entry\n"
        "**Decided:** the real axiom\n"
        "**Rejected:** the real rejected path\n"
        "[DECISION_TRANSCRIPT]\n"
        "User: pasting a doc with markdown in it\n"
        "## Looks Like A New Entry\n"
        "**Decided:** this is transcript text, not a field\n"
        "Claude: understood\n"
        "[/DECISION_TRANSCRIPT]\n"
    )
    entries = parse_entry_stream(text, "decision")
    assert len(entries) == 1  # the inner ## did NOT start a second entry
    entry = entries[0]
    assert entry.axiom == "the real axiom"  # the inner **Decided:** did NOT override
    assert "## Looks Like A New Entry" in entry.transcript
    assert "this is transcript text, not a field" in entry.transcript


def test_parse_entry_stream_literal_comments_preserved() -> None:
    """In-stream HTML comments are literal field text, not stripped (Success #5, V1-D7)."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### with-comment\n"
        "**Decided:** an axiom with a comment <!-- keep me literal --> trailing\n"
        "**Rejected:** none\n"
    )
    entry = parse_entry_stream(text, "decision")[0]
    assert "<!-- keep me literal -->" in entry.axiom


def test_parse_entry_stream_scope_normalization() -> None:
    """Scope case-variants collapse to one casefolded tag (Success #6, MI-9)."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### scope-norm\n"
        "**Decided:** x\n"
        "**Rejected:** y\n"
        "**Scope:** Substrate, substrate, , SUBSTRATE\n"
    )
    entry = parse_entry_stream(text, "decision")[0]
    assert entry.scope == ["substrate"]  # casefolded, deduped, empties dropped


def test_parse_entry_stream_mechanism_normalization() -> None:
    """Mechanism case-and-punct variants collapse to one folded token (Success #6)."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### mech-norm\n"
        "**Decided:** x\n"
        "**Rejected:** y\n"
        "**Mechanisms:** WAL Mode, wal-mode, SQLite, sqlite\n"
    )
    entry = parse_entry_stream(text, "decision")[0]
    # WAL Mode == wal-mode (fold); SQLite == sqlite (casefold); sorted tag set.
    assert entry.mechanisms == ["sqlite", "wal-mode"]


def test_parse_entry_stream_questions_order_preserved_with_dedup() -> None:
    """Bulleted questions keep authored order with order-preserving dedup (Success #6)."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### q-order\n"
        "**Topic:** ordering matters\n"
        "**Questions:**\n"
        "- third concern raised first?\n"
        "- a second concern?\n"
        "- third concern raised first?\n"
    )
    oq = parse_entry_stream(text, "open_question")[0]
    assert oq.questions_raised == ["third concern raised first?", "a second concern?"]


def test_parse_entry_stream_multiline_join_discipline() -> None:
    """Multi-line Rejected/Context newline-join; a wrapped axiom space-collapses."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### joins\n"
        "**Decided:** an axiom that wraps\n"
        "across two lines\n"
        "**Rejected:**\n"
        "- first rejected path\n"
        "- second rejected path\n"
        "**Context:**\n"
        "line one of context\n"
        "line two of context\n"
    )
    entry = parse_entry_stream(text, "decision")[0]
    assert entry.axiom == "an axiom that wraps across two lines"  # space-joined
    assert entry.rejected_paths == "- first rejected path\n- second rejected path"
    assert entry.context == "line one of context\nline two of context"


def test_parse_entry_stream_source_field_extracted_raw() -> None:
    """**Source:** is extracted raw (no enum validation — that is 5a)."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### with-source\n"
        "**Decided:** x\n"
        "**Rejected:** y\n"
        "**Source:** capture_llm\n"
    )
    entry = parse_entry_stream(text, "decision")[0]
    assert entry.source == "capture_llm"


def test_parse_entry_stream_relationship_fields_raw() -> None:
    """Relationship fields are extracted as raw single-slug strings (resolution is 5b)."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### with-edges\n"
        "**Decided:** x\n"
        "**Rejected:** y\n"
        "**Supersedes:** old-decision\n"
        "**Corrects:** typo-decision\n"
        "**Depends-On:** base-decision\n"
    )
    entry = parse_entry_stream(text, "decision")[0]
    assert entry.supersedes == "old-decision"
    assert entry.corrects == "typo-decision"
    assert entry.depends_on == "base-decision"


def test_parse_entry_stream_slugless_header_fails_fast() -> None:
    """A header with no slug is the one structural fail-fast in 4a (V1-D1)."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"  # line 1
        "###\n"                       # line 2  <- slug-less header
        "**Decided:** x\n"            # line 3
    )
    with pytest.raises(ParseError) as exc:
        parse_entry_stream(text, "decision")
    assert exc.value.line_start == 2
    assert exc.value.line_end == 2


def test_parse_entry_stream_source_path_threaded_inert() -> None:
    """An arbitrary source_path is accepted and threaded; inert in 4a (Success #7)."""
    entries = parse_entry_stream(
        _DECISION_SAMPLE_STREAM, "decision", source_path="archive/2025-Q1.md"
    )
    assert len(entries) == 1
    assert entries[0].slug == "example-slug"


def test_parse_entry_stream_to_dict_roundtrip_safe() -> None:
    """The emitted ParsedEntry.to_dict() carries the new keys and is JSON-safe."""
    import json

    entry = parse_entry_stream(_DECISION_SAMPLE_STREAM, "decision")[0]
    d = entry.to_dict()
    assert d["axiom"] == "We will use SQLite in WAL mode for the graph store."
    assert d["topic"] is None
    assert d["source"] is None
    # Lists are plain lists (never tuples); the dict survives a JSON roundtrip.
    assert json.loads(json.dumps(d)) == d
    assert isinstance(d["mechanisms"], list)
    assert isinstance(d["scope"], list)


# --- Identity cross-check (C1 drift guard) ---------------------------------
# The tokenizer must NOT import identity (C1 purity), so its parser-local
# normalization helpers are re-implementations of the §12 byte-forms. This test
# (tests MAY import identity) asserts they stay byte-equal — drift is caught
# structurally, the boundary stays clean (Key Decision 2 / Success #6).

# Raw mechanism fixtures including the degenerate pure-ASCII-punctuation token
# (folds to "" — identity emits [""], so the parser must too; byte-equality is
# the contract, not a post-fold drop). NFC pair uses \u escapes so an editor
# that silently NFC-normalizes the source cannot collapse the canary.
_MECHANISM_CROSSCHECK_FIXTURES = [
    ["sqlite", "wal-mode"],
    ["SQLite", "sqlite"],
    ["WAL Mode", "wal-mode"],
    ["str_casefold", "str-casefold", "Str Casefold"],
    ["  sqlite  ", "sqlite"],
    ["!!!"],                       # pure ASCII punctuation -> folds to ""
    ["", "  ", "sqlite"],          # empty/whitespace raw items filtered
    ["node-scopes", "node-scopes"],
    ["caf\u00e9", "cafe\u0301"],   # composed e-acute vs e + combining acute -> NFC-converge
    ["wal-mode", "WAL Mode", "sqlite", "SQLite"],  # order-independent set
]

_QUESTION_CROSSCHECK_FIXTURES = [
    ["Is the lock fair?", "Does it scale to 10000x?"],
    ["duplicate question?", "duplicate question?"],
    ["  spaced question?  "],
    ["", "  ", "real question?"],
    ["caf\u00e9?", "cafe\u0301?"],   # composed vs decomposed -> NFC-converge, order preserved
    ["b second?", "a first?", "b second?"],  # order preserved, dedup
]


@pytest.mark.parametrize("raw", _MECHANISM_CROSSCHECK_FIXTURES)
def test_parse_entry_stream_mechanism_norm_matches_identity(raw) -> None:
    """Parser mechanism fold is byte-equal to identity.mechanism_refs_list_norm."""
    assert _normalize_mechanism_list(raw) == identity.mechanism_refs_list_norm(raw)


@pytest.mark.parametrize("raw", _QUESTION_CROSSCHECK_FIXTURES)
def test_parse_entry_stream_question_norm_matches_identity(raw) -> None:
    """Parser question norm is byte-equal to identity.questions_raised_list_norm."""
    assert _normalize_questions_list(raw) == identity.questions_raised_list_norm(raw)


def test_parse_entry_stream_scope_norm_is_parser_pinned() -> None:
    """Scope has no identity counterpart; its byte-form is pinned by its own golden."""
    assert _normalize_scope_list(["Substrate", "substrate", " ", "AUTH"]) == [
        "substrate",
        "auth",
    ]
