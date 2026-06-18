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
    load_dynamic_field_map,
    FIELD_MAP,
    ParsedEntry,
    _normalize_mechanism_list,
    _normalize_scope_list,
    _normalize_questions_list,
)
from mitos import identity
from mitos.errors import (
    ParseError,
    ValidationError,
    MitosError,
    EntryFailure,
    FailureItem,
    PARSER_MALFORMED_ENTRY,
    PARSER_MISSING_REQUIRED_FIELD,
    PARSER_MALFORMED_MARKER,
    PARSER_FAILURE_CODES,
)
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


# NOTE (Phase 5a): the two prototype store-validation tests that lived here
# (``test_parse_decisions_missing_axiom_throws`` / ``..._missing_rejected_throws``)
# were removed. They drove a ``parse_decisions_file`` entry into
# ``commit_parsed_entry`` and asserted the prototype's *store-side* required-field
# ``ValidationError``. Phase 5a (Decision 2) moves format validation to the parser
# (C1) and removes that store gate, so those tests asserted behaviour that no
# longer exists. The parser-stage required-field coverage lives in the Phase 4b
# section below; the store's structural canonical-core guard is covered by
# ``tests/test_store.py::test_structural_guard_rejects_empty_canonical_core``.


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
    """Kind comes from the caller's param, not content-sniffing (Success #7).

    4b enforces required fields per kind, so a fixture valid for one kind is no
    longer auto-valid for the other (a decision needs Decided+Rejected; an OQ
    needs Topic+Questions, and Rejected is forbidden on it). The old single
    shared-slug block tripped required-field validation for both kinds — so this
    uses a minimal *valid* block per kind, still proving the caller declares kind.
    """
    decision_block = (
        "<!-- BEGIN ENTRIES -->\n"
        "### a-decision\n"
        "**Decided:** an axiom\n"
        "**Rejected:** a rejected path\n"
    )
    oq_block = (
        "<!-- BEGIN ENTRIES -->\n"
        "### an-open-question\n"
        "**Topic:** a topic\n"
        "**Questions:** a real question?\n"
    )
    assert parse_entry_stream(decision_block, "decision")[0].kind == "decision"
    assert parse_entry_stream(oq_block, "open_question")[0].kind == "open_question"


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


# ===========================================================================
# Phase 4b — validation, marker recognition & the §5.2.2 failure envelope
#
# The parser becomes the authority on format-level well-formedness (the C1
# boundary). A malformed entry yields a structured EntryFailure anchored to the
# slug (or None), with FailureItems carrying stable cross-vision code names and a
# source="parser" discriminator. Per-stage accumulation, per-entry isolation.
# ===========================================================================

# A minimal well-formed decision / OQ stream (line 2 = the slug header).
_MIN_DECISION = (
    "<!-- BEGIN ENTRIES -->\n"
    "### a-decision\n"
    "**Decided:** an axiom\n"
    "**Rejected:** a rejected path\n"
)
_MIN_OQ = (
    "<!-- BEGIN ENTRIES -->\n"
    "### an-open-question\n"
    "**Topic:** a topic\n"
    "**Questions:** a real question?\n"
)


def test_parse_entry_stream_well_formed_empty_collector() -> None:
    """Success #1: a well-formed stream returns entries with an empty collector."""
    failures: list = []
    entries = parse_entry_stream(_MIN_DECISION, "decision", failures=failures)
    assert len(entries) == 1
    assert failures == []
    # Strict mode (no collector) does not raise on a well-formed stream.
    assert len(parse_entry_stream(_MIN_DECISION, "decision")) == 1


def test_parse_entry_stream_missing_decided() -> None:
    """A decision missing **Decided:** yields one missing_required_field item."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### no-axiom\n"
        "**Rejected:** a rejected path\n"
    )
    failures: list = []
    entries = parse_entry_stream(text, "decision", failures=failures)
    assert entries == []
    assert len(failures) == 1
    env = failures[0]
    assert env.slug == "no-axiom"
    assert len(env.items) == 1
    item = env.items[0]
    assert item.code == "missing_required_field"
    assert item.source == "parser"
    assert item.field == "**Decided:**"
    assert item.line_start == 2  # localized to the entry header line


def test_parse_entry_stream_missing_rejected() -> None:
    """A decision missing **Rejected:** yields one missing_required_field item."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### no-rejected\n"
        "**Decided:** an axiom\n"
    )
    failures: list = []
    entries = parse_entry_stream(text, "decision", failures=failures)
    assert entries == []
    assert len(failures) == 1
    assert [i.field for i in failures[0].items] == ["**Rejected:**"]
    assert failures[0].items[0].code == "missing_required_field"


def test_parse_entry_stream_missing_both_required_accumulates() -> None:
    """Missing BOTH required fields -> ONE envelope with TWO items (§5.2.2 accumulate)."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### empty-decision\n"
        "**Context:** only commentary, no required fields\n"
    )
    failures: list = []
    entries = parse_entry_stream(text, "decision", failures=failures)
    assert entries == []
    assert len(failures) == 1  # ONE envelope, not two
    items = failures[0].items
    assert len(items) == 2  # both required fields accumulate within the stage
    assert {i.code for i in items} == {"missing_required_field"}
    assert {i.field for i in items} == {"**Decided:**", "**Rejected:**"}


def test_parse_entry_stream_oq_missing_topic() -> None:
    """An OQ missing **Topic:** yields a missing_required_field item."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### no-topic\n"
        "**Questions:** a real question?\n"
    )
    failures: list = []
    entries = parse_entry_stream(text, "open_question", failures=failures)
    assert entries == []
    assert [i.field for i in failures[0].items] == ["**Topic:**"]


def test_parse_entry_stream_oq_missing_questions() -> None:
    """An OQ missing **Questions:** yields a missing_required_field item."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### no-questions\n"
        "**Topic:** a topic with no questions\n"
    )
    failures: list = []
    entries = parse_entry_stream(text, "open_question", failures=failures)
    assert entries == []
    assert [i.field for i in failures[0].items] == ["**Questions:**"]


def test_parse_entry_stream_mechanisms_optional() -> None:
    """A decision with NO **Mechanisms:** is valid (optional -> [], not required)."""
    failures: list = []
    entries = parse_entry_stream(_MIN_DECISION, "decision", failures=failures)
    assert len(entries) == 1
    assert entries[0].mechanisms == []
    assert failures == []


def test_parse_entry_stream_empty_required_field_is_missing() -> None:
    """A present-but-empty **Decided:** counts as missing (must be non-empty, §1)."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### empty-axiom\n"
        "**Decided:** \n"
        "**Rejected:** a rejected path\n"
    )
    failures: list = []
    entries = parse_entry_stream(text, "decision", failures=failures)
    assert entries == []
    assert [i.field for i in failures[0].items] == ["**Decided:**"]


def test_parse_entry_stream_slugless_header_envelope() -> None:
    """A slug-less header -> malformed_entry, slug=None, raw_header captured."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"  # line 1
        "###\n"                       # line 2  <- slug-less header
        "**Decided:** x\n"            # line 3
    )
    failures: list = []
    entries = parse_entry_stream(text, "decision", failures=failures)
    assert entries == []
    assert len(failures) == 1
    env = failures[0]
    assert env.slug is None
    assert env.raw_header == "###"
    assert len(env.items) == 1
    item = env.items[0]
    assert item.code == "malformed_entry"
    assert item.source == "parser"
    assert item.line_start == 2 and item.line_end == 2  # the header line


def test_parse_entry_stream_unrecognized_field() -> None:
    """An unrecognized field -> malformed_entry, field/line localized."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"   # line 1
        "### with-bogus\n"            # line 2
        "**Decided:** x\n"           # line 3
        "**Bogus:** not a real field\n"  # line 4  <- the offender
        "**Rejected:** y\n"          # line 5
    )
    failures: list = []
    entries = parse_entry_stream(text, "decision", failures=failures)
    assert entries == []
    assert len(failures) == 1
    items = failures[0].items
    assert len(items) == 1
    assert items[0].code == "malformed_entry"
    assert items[0].field == "**Bogus:**"
    assert items[0].line_start == 4  # the offending field's line


def test_parse_entry_stream_rejected_on_oq_forbidden() -> None:
    """**Rejected:** on an open_question -> malformed_entry (M5, Decision 5)."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"  # line 1
        "### oq-with-rejected\n"     # line 2
        "**Topic:** a topic\n"       # line 3
        "**Questions:** q?\n"        # line 4
        "**Rejected:** forbidden\n"  # line 5  <- M5 violation
    )
    failures: list = []
    entries = parse_entry_stream(text, "open_question", failures=failures)
    assert entries == []
    assert len(failures) == 1
    items = failures[0].items
    assert len(items) == 1
    assert items[0].code == "malformed_entry"
    assert items[0].field == "**Rejected:**"
    assert items[0].line_start == 5
    assert "M5" in items[0].message


def test_parse_entry_stream_rejected_on_oq_flagged_even_when_empty() -> None:
    """An empty **Rejected:** on an OQ is still an M5 violation (flagged on presence)."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### oq-empty-rejected\n"
        "**Topic:** a topic\n"
        "**Questions:** q?\n"
        "**Rejected:** \n"
    )
    failures: list = []
    parse_entry_stream(text, "open_question", failures=failures)
    assert len(failures) == 1
    assert [i.field for i in failures[0].items] == ["**Rejected:**"]


def test_parse_entry_stream_topic_on_decision_is_tolerated() -> None:
    """Scope discipline: a stray **Topic:** on a decision is NOT forbidden (spec silent)."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### decision-with-topic\n"
        "**Decided:** an axiom\n"
        "**Rejected:** a rejected path\n"
        "**Topic:** the spec forbids only Rejected-on-OQ, not Topic-on-decision\n"
    )
    failures: list = []
    entries = parse_entry_stream(text, "decision", failures=failures)
    assert len(entries) == 1  # tolerated — only the one M5 rule is enforced
    assert failures == []


def test_parse_entry_stream_unclosed_transcript_marker() -> None:
    """An unclosed [DECISION_TRANSCRIPT] -> malformed_marker (Decision 4).

    The unclosed span absorbs the following sibling entry to EOF (the 4a
    silent-data-loss edge). 4b reports it loudly rather than swallowing in
    silence; recovery (un-absorbing the sibling) is a deliberate non-goal.
    """
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### entry-a\n"
        "**Decided:** axiom a\n"
        "**Rejected:** rejected a\n"
        "[DECISION_TRANSCRIPT]\n"
        "User: a transcript with no close marker\n"
        "### entry-b\n"
        "**Decided:** axiom b\n"
        "**Rejected:** rejected b\n"
    )
    failures: list = []
    entries = parse_entry_stream(text, "decision", failures=failures)
    assert entries == []  # entry-b was absorbed into the open span (not recovered)
    assert len(failures) == 1
    items = failures[0].items
    assert any(i.code == "malformed_marker" for i in items)
    marker = [i for i in items if i.code == "malformed_marker"][0]
    assert marker.field == "[DECISION_TRANSCRIPT]"
    assert marker.line_start == 5  # localized to the unclosed open-marker line


def test_parse_entry_stream_stray_close_marker() -> None:
    """A stray [/DECISION_TRANSCRIPT] (no open) -> malformed_marker (latitude)."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### entry-a\n"
        "**Decided:** axiom a\n"
        "**Rejected:** rejected a\n"
        "[/DECISION_TRANSCRIPT]\n"
    )
    failures: list = []
    entries = parse_entry_stream(text, "decision", failures=failures)
    assert entries == []
    assert len(failures) == 1
    items = failures[0].items
    assert len(items) == 1
    assert items[0].code == "malformed_marker"
    assert items[0].field == "[/DECISION_TRANSCRIPT]"


def test_parse_entry_stream_well_formed_transcript_still_parses() -> None:
    """A balanced [DECISION_TRANSCRIPT] block is valid — no malformed_marker."""
    failures: list = []
    entries = parse_entry_stream(_DECISION_SAMPLE_STREAM, "decision", failures=failures)
    assert len(entries) == 1
    assert failures == []
    assert entries[0].transcript is not None


def test_parse_entry_stream_collector_isolates_per_entry() -> None:
    """§5.2.2 per-entry isolation: one malformed + one good -> good returned, bad enveloped."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### bad-entry\n"
        "**Decided:** something\n"
        "**Bogus:** nope\n"
        "### good-entry\n"
        "**Decided:** a valid axiom\n"
        "**Rejected:** the rejected alternative\n"
    )
    failures: list = []
    entries = parse_entry_stream(text, "decision", failures=failures)
    assert len(entries) == 1
    assert entries[0].slug == "good-entry"
    assert len(failures) == 1
    assert failures[0].slug == "bad-entry"


def test_parse_entry_stream_strict_mode_raises_with_envelope() -> None:
    """Strict mode (no collector): first malformed entry raises ParseError carrying envelope."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### no-axiom\n"
        "**Rejected:** y\n"
    )
    with pytest.raises(ParseError) as exc:
        parse_entry_stream(text, "decision")
    assert exc.value.failure is not None
    assert isinstance(exc.value.failure, EntryFailure)
    assert exc.value.failure.items[0].code == "missing_required_field"


def test_parse_entry_stream_strict_mode_stops_at_first_malformed() -> None:
    """Strict mode raises on the FIRST malformed entry (does not reach the second)."""
    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### first-bad\n"
        "**Rejected:** y\n"          # missing Decided
        "### second-bad\n"
        "**Decided:** x\n"           # missing Rejected
    )
    with pytest.raises(ParseError) as exc:
        parse_entry_stream(text, "decision")
    assert exc.value.failure.slug == "first-bad"


def test_parse_entry_stream_code_names_pinned() -> None:
    """Cross-vision contract: the literal code/source strings are pinned (§5.2.2).

    A typo here is a silent cross-vision break (V3a UX / V5 relay switch on these).
    """
    # The constants ARE the literal contract strings.
    assert PARSER_MALFORMED_ENTRY == "malformed_entry"
    assert PARSER_MISSING_REQUIRED_FIELD == "missing_required_field"
    assert PARSER_MALFORMED_MARKER == "malformed_marker"
    assert PARSER_FAILURE_CODES == {
        "malformed_entry",
        "missing_required_field",
        "malformed_marker",
    }
    # And the parser emits exactly these, with source="parser".
    emitted = {
        "missing_required_field": (
            "<!-- BEGIN ENTRIES -->\n### x\n**Rejected:** y\n", "decision"),
        "malformed_entry": (
            "<!-- BEGIN ENTRIES -->\n### x\n**Decided:** a\n**Bogus:** b\n"
            "**Rejected:** y\n", "decision"),
        "malformed_marker": (
            "<!-- BEGIN ENTRIES -->\n### x\n**Decided:** a\n**Rejected:** y\n"
            "[DECISION_TRANSCRIPT]\nopen\n", "decision"),
    }
    for expected_code, (text, kind) in emitted.items():
        failures: list = []
        parse_entry_stream(text, kind, failures=failures)
        codes = {i.code for env in failures for i in env.items}
        sources = {i.source for env in failures for i in env.items}
        assert expected_code in codes, (expected_code, codes)
        assert sources == {"parser"}
        # Stage-purity: a parser envelope carries ONLY parser-stage codes.
        assert codes <= PARSER_FAILURE_CODES


def test_entry_failure_to_dict_json_roundtrip_safe() -> None:
    """Both envelope structs serialize JSON-roundtrip-safe (cross-vision boundary)."""
    import json

    text = (
        "<!-- BEGIN ENTRIES -->\n"
        "### bad\n"
        "**Bogus:** z\n"
    )
    failures: list = []
    parse_entry_stream(text, "decision", failures=failures)
    assert len(failures) == 1
    d = failures[0].to_dict()
    # Survives a JSON roundtrip; items is a list of plain dicts (never tuples).
    assert json.loads(json.dumps(d)) == d
    assert isinstance(d["items"], list)
    assert all(isinstance(i, dict) for i in d["items"])
    # The item dict carries the contract keys.
    item0 = d["items"][0]
    assert set(item0) == {"code", "source", "message", "field", "line_start", "line_end"}


def test_entry_failure_source_path_threaded() -> None:
    """source_path is threaded from parse_entry_stream onto each envelope."""
    text = "<!-- BEGIN ENTRIES -->\n### bad\n**Bogus:** z\n"
    failures: list = []
    parse_entry_stream(text, "decision", source_path="archive/2025-Q1.md", failures=failures)
    assert failures[0].source_path == "archive/2025-Q1.md"


# --- Baseline removal / §9 spec-pure FIELD_MAP gate (Decision 3) ------------
# The hardcoded baseline mask is gone; FIELD_MAP derives purely from the spec.
# These pin: (a) all expected fields are recognized, (b) spec-derived ⊇ the old
# baseline (no recognition lost), (c) a missing spec fails loudly.

# The exact key set the removed baseline carried (frozen here as the historical
# reference — if a future spec edit drops one of these, (b) fails loudly -> it is
# a 1c spec gap to escalate, NOT a cue to re-add the baseline).
_REMOVED_BASELINE_KEYS = {
    "decided", "mechanisms", "rejected", "invalidates if", "invalidates-if",
    "invalidates_if", "scope", "context", "supersedes", "amends", "narrows",
    "depends-on", "depends on", "depends_on", "resolves", "questions", "corrects",
    "contradicts", "derives-from", "derives from", "derives_from", "cites",
}


def test_field_map_spec_pure_recognizes_all_fields() -> None:
    """The §9 gate: the spec-pure FIELD_MAP recognizes every field + all 9 edges."""
    fm = load_dynamic_field_map()
    # Core / commentary / provenance fields.
    for key in [
        "decided", "mechanisms", "rejected", "invalidates-if", "scope",
        "context", "source", "topic", "questions",
    ]:
        assert key in fm, key
    # All nine relationship names (corrects rides purely on the spec post-removal).
    for rel in [
        "supersedes", "corrects", "amends", "narrows", "depends_on",
        "resolves", "contradicts", "derives_from", "cites",
    ]:
        assert rel in fm, rel
    # special_mappings translate spec names -> attribute names.
    assert fm["decided"] == "core_axiom"
    assert fm["rejected"] == "rejected_paths"
    assert fm["questions"] == "questions_raised"


def test_field_map_spec_derived_superset_of_baseline() -> None:
    """Decision 3 proof: spec-derived ⊇ the old baseline (no recognition lost)."""
    fm = load_dynamic_field_map()
    missing = _REMOVED_BASELINE_KEYS - set(fm)
    assert missing == set(), (
        f"spec-derived FIELD_MAP lost baseline keys {missing} — a 1c spec gap; "
        "escalate, do NOT re-add the baseline (V1-D7)."
    )


def test_load_dynamic_field_map_missing_spec_raises() -> None:
    """A missing/unreadable spec fails loudly (MitosError), not a silent empty map."""
    import builtins

    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if "format-spec.md" in str(path):
            raise FileNotFoundError(path)
        return real_open(path, *args, **kwargs)

    original = builtins.open
    builtins.open = fake_open
    try:
        with pytest.raises(MitosError):
            load_dynamic_field_map()
    finally:
        builtins.open = original


def test_field_map_regex_rejects_malformed_field_names(monkeypatch) -> None:
    """r2 guard: the recognition regex harvests only well-formed [A-Za-z _-] names.

    The pre-r2 class ``[a-zA-Z -_]`` parsed ` -_` as the 0x20-0x5F range, silently
    accepting 35 unintended chars (digits, ``@``, ``/``, ``;`` …). Post-4b this regex
    is the SOLE field-recognition gate (no baseline mask), so a malformed spec bullet
    must NOT be harvested. Well-formed names — including hyphenated ones (hyphen stays
    literal) — must still resolve. Pre-fix this test fails: ``fo0o``/``fo@o``/``a/b``/
    ``a;b`` are harvested by the over-broad range.
    """
    import builtins
    import io
    import re

    synthetic_spec = (
        "# Synthetic format-spec for the r2 regex guard\n"
        "- `**Decided:**` well-formed plain name\n"
        "- `**Invalidates-If:**` well-formed hyphenated name (hyphen literal)\n"
        "- `**Fo0o:**` malformed — digit in name\n"
        "- `**Fo@o:**` malformed — @ in name\n"
        "- `**A/B:**` malformed — slash in name\n"
        "- `**A;B:**` malformed — semicolon in name\n"
    )

    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if "format-spec.md" in str(path):
            return io.StringIO(synthetic_spec)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    fm = load_dynamic_field_map()

    # Well-formed names are harvested — hyphen-as-literal is preserved.
    assert "decided" in fm
    assert "invalidates-if" in fm
    # Malformed names are rejected outright by the tightened class.
    for malformed in ("fo0o", "fo@o", "a/b", "a;b"):
        assert malformed not in fm, f"malformed field name {malformed!r} was harvested"
    # Strong form: NO harvested key carries a char outside [a-z], space, _ or -.
    # (special_mappings + alias_variations only ever add space/underscore/hyphen keys.)
    for key in fm:
        assert re.fullmatch(r"[a-z _-]+", key), f"unexpected chars in harvested key {key!r}"
