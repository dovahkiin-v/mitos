"""Strict deterministic Markdown parser for Mitos decisions.

This module implements the C5 integration contract (Skill -> Parser) under the
OD1 runtime parsing constraint: strictly structured, deterministic, and loud
on any format violation.
"""

import os
import re
import string
import unicodedata
from typing import List, Dict, Optional, Any, Tuple
from mitos.errors import (
    ParseError,
    MitosError,
    EntryFailure,
    FailureItem,
    PARSER_MALFORMED_ENTRY,
    PARSER_MISSING_REQUIRED_FIELD,
    PARSER_MALFORMED_MARKER,
)

def load_dynamic_field_map() -> Dict[str, str]:
    """Builds the FIELD_MAP purely from format-spec.md (C5 single source, V1-D7).

    The map is derived *only* from the field declarations in ``format-spec.md`` —
    there is no hardcoded baseline mask. Post-1c the spec carries every field
    name (``source``/``topic``/``corrects`` + all nine relationship names; the §9
    gate proves spec-derived ⊇ the old baseline), so a baseline would only *hide*
    a future spec omission rather than expose it. ``special_mappings`` translate
    spec names to attribute names (``decided``→``core_axiom``); ``alias_variations``
    register the space-form aliases of multi-word fields.

    Returns:
        A mapping of recognized (lowercased) field names to ParsedEntry attribute
        names.

    Raises:
        MitosError: If ``format-spec.md`` is missing or unreadable. Without it
            the map would be empty and every field would read as unrecognized —
            a debugging nightmare — so this is a loud hard failure (C5
            single-source consequence). The spec ships as package data, so a real
            install always has it.
    """
    import os

    special_mappings = {
        "decided": "core_axiom",
        "rejected": "rejected_paths",
        "questions": "questions_raised",
    }

    alias_variations = {
        "depends on": "depends_on",
        "invalidates if": "invalidates_if",
        "derives from": "derives_from",
    }

    field_map: Dict[str, str] = {}
    spec_path = os.path.join(os.path.dirname(__file__), "format-spec.md")
    try:
        with open(spec_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as exc:
        raise MitosError(
            f"format-spec.md is missing or unreadable at {spec_path}: {exc}. "
            "It is the single source of field-name truth (C5); without it every "
            "field is unrecognized. Reinstall mitos so the spec ships as package "
            "data."
        ) from exc

    # Extract fields declared in the markdown list items: - `**Field:**`
    # Char class is literal letters + space + underscore + hyphen. The hyphen MUST
    # stay last (or be escaped) so it is a literal `-`, NOT a range endpoint — the
    # pre-r2 `[a-zA-Z -_]` silently parsed ` -_` as the 0x20–0x5F range (35 stray
    # chars: digits, @, punctuation). This regex is the SOLE field-recognition gate
    # post-4b (no baseline mask), so a malformed spec name must not be harvested.
    fields_found = re.findall(r'-\s+`\*\*(?P<field>[A-Za-z _-]+):\*\*`', content)
    for f_name in fields_found:
        normalized_key = f_name.strip().lower()

        if normalized_key in special_mappings:
            target_attr = special_mappings[normalized_key]
        else:
            target_attr = normalized_key.replace("-", "_").replace(" ", "_")

        field_map[normalized_key] = target_attr

        # Register alias variations for hyphens/spaces
        if "-" in normalized_key:
            field_map[normalized_key.replace("-", "_")] = target_attr
            field_map[normalized_key.replace("-", " ")] = target_attr

    for k, v in alias_variations.items():
        field_map[k] = v

    return field_map


# Canonical field normalization mapping (dynamically generated to enforce C5 single-source)
FIELD_MAP: Dict[str, str] = load_dynamic_field_map()


def strip_html_comments(text: str) -> str:
    """Strips HTML comments outside fenced code blocks and transcripts.

    This preserves the exact line structure and line count of the source file,
    replacing stripped comment characters with spaces, so that line numbers
    in error reporting remain perfectly accurate.

    Args:
        text: The raw markdown content.

    Returns:
        The comment-stripped markdown content.
    """
    lines = text.splitlines()
    cleaned_lines = []
    in_fenced_code = False
    in_transcript = False
    in_html_comment = False

    for line in lines:
        stripped = line.strip()

        # Track fenced code block state
        if stripped.startswith("```"):
            in_fenced_code = not in_fenced_code

        # Track transcript block state
        if stripped == "[DECISION_TRANSCRIPT]":
            in_transcript = True
        elif stripped == "[/DECISION_TRANSCRIPT]":
            in_transcript = False

        if in_fenced_code or in_transcript:
            # Preserve comments byte-for-byte in protected blocks
            cleaned_lines.append(line)
        else:
            # Replace characters inside <!-- ... --> with spaces to preserve line lengths
            new_chars = list(line)
            i = 0
            while i < len(line):
                if not in_html_comment and line[i:i+4] == "<!--":
                    in_html_comment = True
                    new_chars[i] = ' '
                    new_chars[i+1] = ' '
                    new_chars[i+2] = ' '
                    new_chars[i+3] = ' '
                    i += 4
                elif in_html_comment and line[i:i+3] == "-->":
                    in_html_comment = False
                    new_chars[i] = ' '
                    new_chars[i+1] = ' '
                    new_chars[i+2] = ' '
                    i += 3
                else:
                    if in_html_comment:
                        new_chars[i] = ' '
                    i += 1
            cleaned_lines.append("".join(new_chars))

    return "\n".join(cleaned_lines)


def parse_header(header_line: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Parses a markdown heading line to extract slug, date, and title.

    Supports formats like:
      - ## YYYY-MM-DD — slug — Title
      - ### slug

    Args:
        header_line: The header line string.

    Returns:
        A tuple of (slug, date, title).
    """
    text = header_line.lstrip("#").strip()
    
    # Split by em-dash (—), en-dash (–), or space-dash-space
    parts = re.split(r'\s*[\u2014\u2013]\s*|\s+-\s+', text)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) >= 3:
        if re.match(r'^\d{4}-\d{2}-\d{2}$', parts[0]):
            return parts[1], parts[0], " - ".join(parts[2:])
        else:
            return parts[0], None, " - ".join(parts[1:])
    elif len(parts) == 2:
        if re.match(r'^\d{4}-\d{2}-\d{2}$', parts[0]):
            return parts[1], parts[0], None
        else:
            return parts[0], None, parts[1]
    elif len(parts) == 1:
        return parts[0], None, None
    else:
        raise ValueError(f"Malformed heading line: {header_line}")


class ParsedEntry:
    """Structured representation of a parsed decision or open question."""

    def __init__(self, kind: str, slug: str, line_start: int, line_end: int) -> None:
        self.kind = kind
        self.slug = slug
        self.line_start = line_start
        self.line_end = line_end

        # V1a canonical-core surface (populated only by ``parse_entry_stream``).
        # ``axiom`` is the V1a name for the decision axiom; the prototype's
        # ``core_axiom`` stays alongside it until Phase 8a renames the consumers.
        # ``topic`` is the open_question canonical-core field; ``source`` is the
        # tool-only provenance field (5a validates it and defaults absent -> "user").
        self.axiom: str = ""
        self.topic: Optional[str] = None
        self.source: Optional[str] = None

        # Decision fields
        self.date: Optional[str] = None
        self.title: Optional[str] = None
        self.core_axiom: str = ""
        self.mechanisms: List[str] = []
        self.rejected_paths: str = ""
        self.invalidates_if: Optional[str] = None
        self.scope: List[str] = []
        self.context: Optional[str] = None
        # Relationship fields are comma-separated multi-valued (V1b, ADR
        # ``relationship-fields-comma-separated-multivalued``): each parses to a
        # ``List[str]`` of cited slugs, mirroring ``mechanisms`` / ``scope`` above
        # (``[]`` when absent; a lone slug is a 1-element list, so V1a single-valued
        # authoring is unchanged — additive, no migration). The per-instance ``[]``
        # is minted fresh each ``__init__`` (no shared-mutable-default foot-gun).
        self.supersedes: List[str] = []
        self.corrects: List[str] = []
        self.amends: List[str] = []
        self.narrows: List[str] = []
        self.depends_on: List[str] = []
        self.resolves: List[str] = []
        self.contradicts: List[str] = []
        self.derives_from: List[str] = []
        self.cites: List[str] = []
        self.transcript: Optional[str] = None
        self.confirmed_by: Optional[str] = None
        self.confirmed_at: Optional[str] = None
        self.notes: List[str] = []
        self.parked_questions: List[str] = []

        # Open question fields
        self.park_reason: Optional[str] = None
        self.questions_raised: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        """Serializes ParsedEntry into a JSON-compatible dictionary."""
        return {
            "kind": self.kind,
            "slug": self.slug,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "date": self.date,
            "title": self.title,
            "core_axiom": self.core_axiom,
            "mechanisms": self.mechanisms,
            "rejected_paths": self.rejected_paths,
            "invalidates_if": self.invalidates_if,
            "scope": self.scope,
            "context": self.context,
            "supersedes": self.supersedes,
            "corrects": self.corrects,
            "amends": self.amends,
            "narrows": self.narrows,
            "depends_on": self.depends_on,
            "resolves": self.resolves,
            "contradicts": self.contradicts,
            "derives_from": self.derives_from,
            "cites": self.cites,
            "transcript": self.transcript,
            "confirmed_by": self.confirmed_by,
            "confirmed_at": self.confirmed_at,
            "notes": self.notes,
            "parked_questions": self.parked_questions,
            "park_reason": self.park_reason,
            "questions_raised": self.questions_raised,
            "axiom": self.axiom,
            "topic": self.topic,
            "source": self.source,
        }


def parse_decisions_file(text: str, errors: Optional[List[ParseError]] = None) -> List[ParsedEntry]:
    """Parses a markdown write-buffer text deterministically into parsed entries.

    Handles HTML comment stripping, division into sections below the BEGIN ENTRIES
    marker, field parsing, and strict invariant validation.

    Args:
        text: The raw decisions.md file content.
        errors: Optional collector for per-entry parse failures. When supplied, a
            malformed entry is isolated -- its ParseError is appended here and
            parsing continues with the remaining entries, so one bad entry does
            not block the rest of the sync (the section 7.2 degradation contract:
            "other entries in the same sync continue"). When None (the default),
            the first malformed entry raises immediately (strict mode).

    Returns:
        A list of the successfully parsed ParsedEntry objects.

    Raises:
        ParseError: If any structural syntax is malformed and no ``errors``
            collector was supplied.
    """
    # 1. Strip HTML comments outside protected blocks on the whole file
    clean_text = strip_html_comments(text)
    lines = clean_text.splitlines()

    # 2. Find the line index containing "BEGIN ENTRIES" in raw text
    begin_line_idx = 0
    raw_lines = text.splitlines()
    for idx, line in enumerate(raw_lines):
        if "BEGIN ENTRIES" in line:
            begin_line_idx = idx
            break

    # 3. Identify and scan sections (Decisions or Open Questions)
    sections: List[Dict[str, Any]] = []
    current_section: Optional[Dict[str, Any]] = None

    for idx, line in enumerate(lines, start=1):
        if idx - 1 < begin_line_idx:
            continue
            
        stripped = line.strip()

        # Check section start conditions
        is_decision_start = (line.startswith("##") or line.startswith("###")) and not line.startswith("####")
        is_oq_start = "[DECISION_PARKED:" in line

        if is_decision_start or is_oq_start:
            if current_section:
                current_section["line_end"] = idx - 1
                sections.append(current_section)

            current_section = {
                "type": "decision" if is_decision_start else "open_question",
                "line_start": idx,
                "header_line": line,
                "lines": []
            }

        if current_section:
            current_section["lines"].append(line)

    if current_section:
        current_section["line_end"] = len(lines)
        sections.append(current_section)

    parsed_entries: List[ParsedEntry] = []

    # 4. Parse each section in isolation. A malformed entry is recorded in the
    #    errors collector and skipped so the remaining entries still sync; in
    #    strict mode (no collector) the first malformed entry re-raises.
    for sec in sections:
        try:
            parsed_entries.append(_parse_section(sec))
        except ParseError as exc:
            if errors is None:
                raise
            errors.append(exc)
            continue

    return parsed_entries


def _parse_section(sec: Dict[str, Any]) -> ParsedEntry:
    """Parses a single pre-scanned section into a ParsedEntry.

    Args:
        sec: A section dict from the scanning pass, carrying ``type``,
            ``line_start``, ``line_end``, ``header_line``, and ``lines``.

    Returns:
        The parsed entry for this section.

    Raises:
        ParseError: If the header, a field, or a marker is malformed; the
            reported line range points at the actual offending line.
    """
    line_start = sec["line_start"]
    line_end = sec["line_end"]

    if sec["type"] == "decision":
        # Extract header details
        try:
            slug, date, title = parse_header(sec["header_line"])
        except Exception as e:
            raise ParseError(f"Malformed decision header: {str(e)}", line_start, line_start)

        entry = ParsedEntry("decision", slug, line_start, line_end)
        entry.date = date
        entry.title = title

        # Parse fields
        fields: Dict[str, List[str]] = {}
        current_field: Optional[str] = None
        in_transcript = False
        transcript_lines: List[str] = []

        for offset, line in enumerate(sec["lines"][1:], start=1):  # Skip the header line
            # Absolute file line of this line, so a malformed field is reported
            # at the real offending line rather than a stale scan-loop variable.
            field_line = line_start + offset
            stripped = line.strip()

            # Transcript boundary checks
            if stripped == "[DECISION_TRANSCRIPT]":
                in_transcript = True
                continue
            elif stripped == "[/DECISION_TRANSCRIPT]":
                in_transcript = False
                continue

            if in_transcript:
                transcript_lines.append(line)
                continue

            # Match standard field pattern: **Field**: or **Field:**
            field_match = re.match(r'^\s*\*\*(?P<field>[a-zA-Z -]+)(?::\*\*|\*\*:\s*)(?P<content>.*)$', line)
            if field_match:
                field_name = field_match.group("field").strip().lower()
                content = field_match.group("content").strip()

                if field_name in FIELD_MAP:
                    current_field = FIELD_MAP[field_name]
                    fields[current_field] = [content]
                else:
                    raise ParseError(f"Unknown field '**{field_match.group('field')}**' declared", field_line, field_line)
            else:
                if current_field and stripped:
                    fields[current_field].append(line.strip())

        # Assign parsed fields to entry
        if transcript_lines:
            entry.transcript = "\n".join(transcript_lines).strip()

        # Scan for inline markers (NOTE, PARKED)
        for line in sec["lines"]:
            note_matches = re.findall(r'\[NOTE:\s*([^\]]+)\]', line)
            for nm in note_matches:
                entry.notes.append(nm.strip())

            parked_matches = re.findall(r'\[PARKED:\s*([^\]]+)\]', line)
            for pm in parked_matches:
                entry.parked_questions.append(pm.strip())

        if "core_axiom" in fields:
            entry.core_axiom = " ".join(fields["core_axiom"]).strip()
        if "rejected_paths" in fields:
            entry.rejected_paths = "\n".join(fields["rejected_paths"]).strip()
        if "invalidates_if" in fields:
            entry.invalidates_if = " ".join(fields["invalidates_if"]).strip()
        if "context" in fields:
            entry.context = "\n".join(fields["context"]).strip()
        # Relationship fields are comma-separated multi-valued (V1b) — same split as
        # the live ``parse_entry_stream`` path, kept consistent so this prototype
        # never assigns a bare ``str`` to a now-``List[str]`` attribute.
        for rel in _RELATIONSHIP_FIELDS:
            if rel in fields:
                rel_raw = " ".join(fields[rel]).strip()
                setattr(entry, rel, [c.strip() for c in rel_raw.split(",") if c.strip()])

        if "mechanisms" in fields:
            mech_str = " ".join(fields["mechanisms"]).strip()
            entry.mechanisms = [m.strip() for m in mech_str.split(",") if m.strip()]

        if "scope" in fields:
            scope_str = " ".join(fields["scope"]).strip()
            entry.scope = [s.strip() for s in scope_str.split(",") if s.strip()]

        return entry

    elif sec["type"] == "open_question":
        # Extract topic & reason from inline marker: [DECISION_PARKED: topic — reason]
        header = sec["header_line"]
        match = re.search(r'\[DECISION_PARKED:\s*(?P<content>[^\]]+)\]', header)
        if not match:
            raise ParseError("Malformed [DECISION_PARKED] marker syntax", line_start, line_start)

        content = match.group("content").strip()
        parts = re.split(r'\s*[\u2014\u2013]\s*|\s+-\s+', content, maxsplit=1)

        slug = parts[0].strip()
        park_reason = parts[1].strip() if len(parts) > 1 else None

        entry = ParsedEntry("open_question", slug, line_start, line_end)
        entry.park_reason = park_reason

        # Parse fields (specifically **Questions:**)
        questions_lines: List[str] = []
        current_field = None

        for line in sec["lines"][1:]:  # Skip the marker line
            field_match = re.match(r'^\s*\*\*(?P<field>[a-zA-Z -]+)(?::\*\*|\*\*:\s*)(?P<content>.*)$', line)
            if field_match:
                field_name = field_match.group("field").strip().lower()
                if FIELD_MAP.get(field_name) == "questions_raised":
                    current_field = "questions_raised"
                    content = field_match.group("content").strip()
                    questions_lines.append(content)
                else:
                    current_field = None
            else:
                if current_field == "questions_raised" and line.strip():
                    questions_lines.append(line.strip())

        # Split questions by list markers or newlines
        raw_questions = "\n".join(questions_lines)
        # Find bullet points: - question, * question, 1. question
        bullets = re.split(r'\n\s*[-\*\d\.]+\s*', "\n" + raw_questions)
        entry.questions_raised = [b.strip() for b in bullets if b.strip()]

        if not entry.questions_raised:
            # Fallback to lines if no bullet pattern was matched
            entry.questions_raised = [q.strip() for q in questions_lines if q.strip()]

        return entry


# ---------------------------------------------------------------------------
# V1a entry-stream tokenizer (Phase 4a)
#
# ``parse_entry_stream`` is the V1a half of the C1 boundary (V1-D8): a pure,
# deterministic, *dumb-on-purpose* tokenizer. It turns one entry-stream file of
# a single caller-declared kind into ``ParsedEntry`` objects carrying raw,
# boundary-normalized fields. It does NOT hash (store/5a), does NOT validate
# required fields or build a failure envelope (4b), and does NOT strip HTML
# comments from stream content (V1-D7 — comments are literal field text).
#
# It is built UNWIRED alongside the still-live prototype ``parse_decisions_file``
# (Key Decision 1): no consumer calls it until 5a (commit) and 7a (cutover).
#
# C1 PURITY (Key Decision 2): the normalization helpers below fold/casefold/dedup
# to the SAME §12 byte-forms as ``mitos/identity.py`` (so a token the parser
# stores matches the token identity later hashes), but this module MUST NOT
# import ``identity`` — that would reverse the C1 direction. Drift is caught
# structurally by a cross-check test that imports identity and asserts equality.
# ---------------------------------------------------------------------------

# Maximal run of ASCII whitespace OR ASCII punctuation -> a single hyphen. This
# is byte-identical to ``identity._MECHANISM_FOLD_RE`` (the cross-check test
# pins it). ``re.ASCII`` restricts ``\s`` to ASCII whitespace so the fold never
# collapses a non-ASCII token (CJK, accented letters, a no-break space) — those
# pass through NFC+casefold untouched (V1-D3).
_MECHANISM_FOLD_RE = re.compile(r"[\s" + re.escape(string.punctuation) + r"]+", re.ASCII)

# A field declaration line: ``**Field:** content`` or ``**Field**: content``.
# The field-name class is ASCII letters/space/hyphen only, matching the prototype
# (`Invalidates-If`, `Depends-On`, etc. all parse). Reused read-only from the
# prototype's inline shape; recognition is via ``FIELD_MAP``.
_FIELD_LINE_RE = re.compile(
    r'^\s*\*\*(?P<field>[a-zA-Z -]+)(?::\*\*|\*\*:\s*)(?P<content>.*)$'
)

# Splits a ``**Questions:**`` field body into individual questions on bullet /
# ordinal markers at line starts (``- q``, ``* q``, ``1. q``). Deliberately does
# NOT split on a sentence-final ``?`` — a single question may contain one, and the
# OQ identity hashes the *list*, so over-splitting would silently mint a different
# node id. This is the prototype's split, preserved verbatim.
_QUESTION_BULLET_RE = re.compile(r'\n\s*[-\*\d\.]+\s*')

# Single-slug relationship fields extracted as raw strings (recognition
# completeness is 4b; edge resolution is 5b). Order is irrelevant — assignment
# is per-field.
_RELATIONSHIP_FIELDS = (
    "supersedes", "corrects", "amends", "narrows", "depends_on",
    "resolves", "contradicts", "derives_from", "cites",
)


def _mechanism_canonical_norm(token: str) -> str:
    """Folds a single mechanism token to its canonical, casefolded form.

    NFC, then casefold, then collapse each maximal run of ASCII punctuation or
    ASCII whitespace to a single hyphen and strip leading/trailing hyphens. So
    ``"WAL Mode"`` and ``"wal-mode"`` both fold to ``"wal-mode"``. This is
    byte-identical to ``identity.mechanism_canonical_norm`` (C1 cross-check).

    Args:
        token: The raw mechanism token.

    Returns:
        The casefolded, punctuation-folded canonical token.
    """
    folded = unicodedata.normalize("NFC", token).casefold()
    return _MECHANISM_FOLD_RE.sub("-", folded).strip("-")


def _normalize_mechanism_list(items: List[str]) -> List[str]:
    """Normalizes raw mechanism tokens into a sorted, deduped tag set.

    Filters empty/whitespace-only raw items (``if m.strip()`` on the *raw* item,
    before folding — §12's filter-on-the-raw rule), folds each, then set-dedups
    and code-point-sorts. ``mechanism_refs`` is an unordered set, so reordering
    or duplicating never changes the eventual node id. Byte-identical to
    ``identity.mechanism_refs_list_norm`` (C1 cross-check).

    Args:
        items: The raw mechanism tokens (already comma-split).

    Returns:
        The folded, deduped, code-point-sorted mechanism list.
    """
    return sorted({_mechanism_canonical_norm(m) for m in items if m.strip()})


def _normalize_scope_list(items: List[str]) -> List[str]:
    """Normalizes raw scope tags: casefold, drop empties, order-preserving dedup.

    Scope is a cross-kind tag set. Each tag is stripped and casefolded (Python
    ``str.casefold`` — never SQLite ``NOCASE``/``LOWER``, MI-7/P9), empties are
    dropped, and duplicates are removed preserving first-seen order, so no
    empty/NULL scope row can ever reach the store (MI-9). Scope has **no**
    ``identity.py`` counterpart (it is commentary, not hashed) — its byte-form is
    pinned by its own golden, not by the cross-check.

    Args:
        items: The raw scope tags (already comma-split).

    Returns:
        The casefolded, deduped scope list in authored order.
    """
    return list(dict.fromkeys(s.strip().casefold() for s in items if s.strip()))


def _normalize_questions_list(items: List[str]) -> List[str]:
    """Normalizes raw open-question strings, preserving authored order.

    Filters empty/whitespace-only raw items, NFC-normalizes and end-strips each
    (case preserved — case is content), then applies an order-preserving dedup —
    **never a sort**. Authored order is identity-significant for an open question
    (M1), so reordering mints a new node. Byte-identical to
    ``identity.questions_raised_list_norm`` (C1 cross-check).

    Args:
        items: The raw question strings (already bullet-split).

    Returns:
        The NFC-normalized questions in authored order, duplicates removed.
    """
    return list(
        dict.fromkeys(
            unicodedata.normalize("NFC", q).strip() for q in items if q.strip()
        )
    )


def _split_entry_sections(
    lines: List[str], begin_idx: int
) -> List[Dict[str, Any]]:
    """Splits the entry-stream lines into per-entry section dicts.

    A single transcript-aware pass: a ``##``/``### slug`` line starts a new
    section, EXCEPT inside a ``[DECISION_TRANSCRIPT]…[/DECISION_TRANSCRIPT]``
    block, where such a line is literal transcript text (this is the latent
    prototype bug the new path fixes). ``####`` and single ``#`` are not entry
    delimiters. The transcript marker lines are kept in the section so the
    per-section tokenizer re-tracks the span and captures the transcript body.

    Args:
        lines: All lines of the file (post-``splitlines``).
        begin_idx: Index of the first entry-stream line (the line after the
            sentinel, or 0 when there is no sentinel).

    Returns:
        A list of section dicts, each with ``line_start`` (1-based, absolute),
        ``line_end``, ``header_line``, and ``lines`` (header first).
    """
    sections: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    in_transcript = False

    for i in range(begin_idx, len(lines)):
        line = lines[i]
        file_line = i + 1  # 1-based, absolute file line
        stripped = line.strip()

        if not in_transcript and stripped == "[DECISION_TRANSCRIPT]":
            in_transcript = True
            if current is not None:
                current["lines"].append(line)
            continue
        if in_transcript and stripped == "[/DECISION_TRANSCRIPT]":
            in_transcript = False
            if current is not None:
                current["lines"].append(line)
            continue

        is_header = (
            not in_transcript
            and line.startswith("##")
            and not line.startswith("####")
        )
        if is_header:
            if current is not None:
                current["line_end"] = file_line - 1
                sections.append(current)
            current = {
                "line_start": file_line,
                "header_line": line,
                "lines": [line],
            }
        elif current is not None:
            current["lines"].append(line)

    if current is not None:
        current["line_end"] = len(lines)
        sections.append(current)

    return sections


def _tokenize_entry(
    sec: Dict[str, Any],
    kind: str,
    items: Optional[List[FailureItem]] = None,
) -> ParsedEntry:
    """Tokenizes one pre-split section into a ``ParsedEntry`` of the given kind.

    Extracts every field ``FIELD_MAP`` recognizes, assigns ``**Decided:**`` to
    the new ``axiom`` attribute (not ``core_axiom``), and normalizes the list
    fields at the C1 boundary.

    When an ``items`` collector is supplied (4b validation), structural format
    violations that have an offending line are surfaced into it as
    :class:`~mitos.errors.FailureItem`: an unrecognized ``**Field:**`` line, a
    ``**Rejected:**`` field on an ``open_question`` (forbidden by M5), an
    unclosed ``[DECISION_TRANSCRIPT]`` marker, and a stray
    ``[/DECISION_TRANSCRIPT]`` close. *Absence* failures (a required field that is
    simply not present) have no line and are checked separately by
    :func:`_check_required_fields`. When ``items`` is ``None`` the tokenizer is
    purely permissive (4a behavior): unrecognized fields are silently discarded
    and marker balance is not reported.

    Args:
        sec: A section dict from :func:`_split_entry_sections`.
        kind: The caller-declared kind (``"decision"`` / ``"open_question"``).
        items: Optional collector for structural ``FailureItem``s (4b).

    Returns:
        The tokenized entry.

    Raises:
        ParseError: If the header carries no slug (structural fail-fast, V1-D1).
            The caller (:func:`_validate_section`) folds this into a pre-header
            ``malformed_entry`` envelope.
    """
    line_start = sec["line_start"]
    line_end = sec["line_end"]

    try:
        slug = parse_header(sec["header_line"])[0]
    except ValueError as exc:
        raise ParseError(f"Malformed entry header: {exc}", line_start, line_start)

    entry = ParsedEntry(kind, slug, line_start, line_end)

    fields: Dict[str, List[str]] = {}
    current_field: Optional[str] = None
    in_transcript = False
    transcript_open_line: Optional[int] = None
    transcript_lines: List[str] = []

    # Per-line offset tracking (mirrors the prototype ``_parse_section``) so a
    # structural item can be localized to the offending file line.
    for offset, line in enumerate(sec["lines"][1:], start=1):  # skip the header
        file_line = line_start + offset  # 1-based, absolute file line
        stripped = line.strip()

        # Transcript span: markers toggle, body is captured verbatim. A field- or
        # header-shaped line inside the span is literal transcript text.
        if not in_transcript and stripped == "[DECISION_TRANSCRIPT]":
            in_transcript = True
            transcript_open_line = file_line
            continue
        if in_transcript and stripped == "[/DECISION_TRANSCRIPT]":
            in_transcript = False
            continue
        if in_transcript:
            transcript_lines.append(line)
            continue
        if stripped == "[/DECISION_TRANSCRIPT]":
            # A close marker with no matching open (latitude, Decision 4): a
            # marker line is never field content, so it is consumed here. Loud
            # report when validating; otherwise silently dropped.
            if items is not None:
                items.append(
                    FailureItem(
                        code=PARSER_MALFORMED_MARKER,
                        source="parser",
                        message=(
                            "Stray [/DECISION_TRANSCRIPT] close marker with no "
                            "matching [DECISION_TRANSCRIPT]."
                        ),
                        field="[/DECISION_TRANSCRIPT]",
                        line_start=file_line,
                        line_end=file_line,
                    )
                )
            continue

        field_match = _FIELD_LINE_RE.match(line)
        if field_match:
            raw_name = field_match.group("field").strip()
            name = raw_name.lower()
            content = field_match.group("content").strip()
            if name in FIELD_MAP:
                current_field = FIELD_MAP[name]
                fields[current_field] = [content]
                # M5 (Decision 5): ``**Rejected:**`` is decision-only — its
                # presence on an open question is a format violation. Flagged on
                # the field line (presence, not content), so an empty
                # ``**Rejected:**`` on an OQ is caught too.
                if (
                    kind == "open_question"
                    and current_field == "rejected_paths"
                    and items is not None
                ):
                    items.append(
                        FailureItem(
                            code=PARSER_MALFORMED_ENTRY,
                            source="parser",
                            message=(
                                "Field **Rejected:** is not permitted on an open "
                                "question (decision-only, M5)."
                            ),
                            field="**Rejected:**",
                            line_start=file_line,
                            line_end=file_line,
                        )
                    )
            else:
                # An unrecognized field stops accumulation so its content can't
                # bleed into the prior field. Permissive (discard) without a
                # collector; reported as malformed_entry with a collector (4b).
                current_field = None
                if items is not None:
                    items.append(
                        FailureItem(
                            code=PARSER_MALFORMED_ENTRY,
                            source="parser",
                            message=f"Unrecognized field '**{raw_name}:**'.",
                            field=f"**{raw_name}:**",
                            line_start=file_line,
                            line_end=file_line,
                        )
                    )
        elif current_field is not None and stripped:
            # Continuation line for the current field.
            fields[current_field].append(stripped)

    # Marker balance: an unclosed [DECISION_TRANSCRIPT] swallows every following
    # entry into the open span to EOF (the 4a silent-data-loss edge). 4b detects
    # and reports it; recovery (un-absorbing siblings) is a deliberate non-goal
    # (Decision 4) — the loud failure is the remedy.
    if in_transcript and items is not None:
        open_line = transcript_open_line if transcript_open_line is not None else line_start
        items.append(
            FailureItem(
                code=PARSER_MALFORMED_MARKER,
                source="parser",
                message=(
                    "Unclosed [DECISION_TRANSCRIPT] marker "
                    "(missing [/DECISION_TRANSCRIPT])."
                ),
                field="[DECISION_TRANSCRIPT]",
                line_start=open_line,
                line_end=line_end,
            )
        )

    if transcript_lines:
        entry.transcript = "\n".join(transcript_lines).strip()

    # --- Field assignment with per-field join discipline ---
    # ``**Decided:**`` (FIELD_MAP -> "core_axiom") lands on the NEW ``axiom``
    # attribute; the prototype ``core_axiom`` is left untouched (8a flips it).
    if "core_axiom" in fields:
        entry.axiom = " ".join(fields["core_axiom"]).strip()
    if "topic" in fields:
        entry.topic = " ".join(fields["topic"]).strip()
    if "source" in fields:
        entry.source = " ".join(fields["source"]).strip()

    # Prose / single-value fields: space-join (a wrapped value collapses to one
    # line). Multi-line fields: newline-join (a list survives intact).
    if "invalidates_if" in fields:
        entry.invalidates_if = " ".join(fields["invalidates_if"]).strip()
    if "rejected_paths" in fields:
        entry.rejected_paths = "\n".join(fields["rejected_paths"]).strip()
    if "context" in fields:
        entry.context = "\n".join(fields["context"]).strip()
    # Relationship fields are comma-separated multi-valued (V1b): ``Cites: a, b``
    # tokenizes to ``["a", "b"]``; a lone slug is a 1-element list (mirror the
    # ``mechanisms`` / ``scope`` split below). Edge resolution per citation is the
    # store's job (``_reconcile_edges``); the parser only tokenizes.
    for rel in _RELATIONSHIP_FIELDS:
        if rel in fields:
            rel_raw = " ".join(fields[rel]).strip()
            setattr(entry, rel, [c.strip() for c in rel_raw.split(",") if c.strip()])

    # Boundary-normalized list fields (the C1 fold/dedup, byte-equal to identity
    # for mechanisms/questions; scope is parser-only).
    if "mechanisms" in fields:
        mech_raw = " ".join(fields["mechanisms"]).strip()
        entry.mechanisms = _normalize_mechanism_list(mech_raw.split(","))
    if "scope" in fields:
        scope_raw = " ".join(fields["scope"]).strip()
        entry.scope = _normalize_scope_list(scope_raw.split(","))
    if "questions_raised" in fields:
        raw_questions = "\n".join(fields["questions_raised"])
        bullets = _QUESTION_BULLET_RE.split("\n" + raw_questions)
        entry.questions_raised = _normalize_questions_list(bullets)

    return entry


# Required fields per kind (V1-D8 / format-spec.md §1, §2): the spec field token
# paired with the ParsedEntry attribute that must be present and non-empty.
# ``mechanism_refs`` is intentionally NOT here — it is optional (absent -> []).
_REQUIRED_FIELDS: Dict[str, List[Tuple[str, str]]] = {
    "decision": [("**Decided:**", "axiom"), ("**Rejected:**", "rejected_paths")],
    "open_question": [("**Topic:**", "topic"), ("**Questions:**", "questions_raised")],
}


def _attr_is_present(entry: ParsedEntry, attr: str) -> bool:
    """Reports whether a required ParsedEntry attribute is present and non-empty.

    A required field whose value is empty (or whitespace-only for the string
    fields) counts as absent — format-spec.md requires the field be "present and
    non-empty", so an empty ``**Decided:**`` is a missing axiom, and a
    ``**Questions:**`` that normalizes to no items is missing questions.

    Args:
        entry: The tokenized entry.
        attr: The attribute name (``axiom`` / ``rejected_paths`` / ``topic`` /
            ``questions_raised``).

    Returns:
        ``True`` if the field carries content, ``False`` otherwise.
    """
    value = getattr(entry, attr)
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)  # list field (questions_raised): non-empty


def _check_required_fields(
    entry: ParsedEntry, kind: str, sec: Dict[str, Any], items: List[FailureItem]
) -> None:
    """Appends a ``missing_required_field`` item for each absent required field.

    An absent field has no offending line (there is nothing there), so each item
    is localized to the entry's header line — the place a reader looks to fix it
    (P3 vector error). Failures accumulate: a decision missing both required
    fields yields two items in one envelope (§5.2.2 "accumulate within stage").

    Args:
        entry: The tokenized entry.
        kind: The caller-declared kind.
        sec: The section dict (its ``line_start`` anchors the items).
        items: The collector to append to.
    """
    header_line = sec["line_start"]
    for field_token, attr in _REQUIRED_FIELDS.get(kind, []):
        if not _attr_is_present(entry, attr):
            items.append(
                FailureItem(
                    code=PARSER_MISSING_REQUIRED_FIELD,
                    source="parser",
                    message=f"Missing required field {field_token}.",
                    field=field_token,
                    line_start=header_line,
                    line_end=header_line,
                )
            )


def _validate_section(
    sec: Dict[str, Any], kind: str, source_path: Optional[str]
) -> Tuple[Optional[ParsedEntry], Optional[EntryFailure]]:
    """Tokenizes and format-validates one section against ``format-spec.md``.

    Exactly one of the returned pair is non-``None``:

    - ``(ParsedEntry, None)`` — the section is well-formed.
    - ``(None, EntryFailure)`` — the section has one or more format violations,
      accumulated into the envelope's ``items`` (the §5.2.2 payload).

    A pre-header failure (a structurally untokenizable header — no slug) cannot
    anchor field checks, so it short-circuits to a single ``malformed_entry``
    item with ``slug=None`` and the raw header captured. A tokenizable entry runs
    all checks and accumulates them.

    Args:
        sec: A section dict from :func:`_split_entry_sections`.
        kind: The caller-declared kind.
        source_path: The originating path, threaded onto the envelope.

    Returns:
        The ``(entry, failure)`` pair described above.
    """
    items: List[FailureItem] = []
    try:
        entry = _tokenize_entry(sec, kind, items)
    except ParseError as exc:
        # Pre-header failure: no slug to anchor field checks. Emit one
        # malformed_entry item carrying the raw header (slug stays None).
        envelope = EntryFailure(
            slug=None,
            line_start=exc.line_start,
            line_end=exc.line_end,
            source_path=source_path,
            raw_header=sec["header_line"],
            items=[
                FailureItem(
                    code=PARSER_MALFORMED_ENTRY,
                    source="parser",
                    message=exc.message,
                    field=None,
                    line_start=exc.line_start,
                    line_end=exc.line_end,
                )
            ],
        )
        return None, envelope

    # Header tokenized: required-field presence (the absence checks). Structural
    # violations with a line (unrecognized field, M5, marker balance) were
    # already appended to ``items`` during tokenization.
    _check_required_fields(entry, kind, sec, items)

    if not items:
        return entry, None

    envelope = EntryFailure(
        slug=entry.slug,
        line_start=sec["line_start"],
        line_end=sec["line_end"],
        source_path=source_path,
        items=items,
    )
    return None, envelope


def parse_entry_stream(
    text: str,
    kind: str,
    source_path: Optional[str] = None,
    failures: Optional[List[EntryFailure]] = None,
) -> List[ParsedEntry]:
    """Tokenizes one entry-stream file of a single declared kind (V1a, C1).

    The pure deterministic tokenizer half of the C1 boundary (V1-D8). It splits
    the file into a discarded preamble (everything up to and including the
    ``<!-- BEGIN ENTRIES … -->`` sentinel) and an entry stream, then tokenizes
    each ``##``/``### slug`` block into a :class:`ParsedEntry` of the declared
    kind, with boundary-normalized list fields and verbatim in-stream HTML
    comments.

    Kind is **caller-declared, never filename-sniffed** (V1-D8): the caller
    states ``decision`` or ``open_question`` for the stream it hands over, which
    lets the cutover replay archive/snapshot files under the correct kind. A file
    with **no** sentinel is treated as wholly an entry stream.

    The parser is the authority on **format-level well-formedness** (the C1
    boundary, V1-D8): required-field presence per kind, marker balance, and
    spec-pure field recognition. A malformed entry produces a structured
    :class:`~mitos.errors.EntryFailure` (§5.2.2). It does **not** hash (store/5a),
    does **not** do referential/graph validation — slug collisions, edge targets,
    acyclicity are the store's ``source="store"`` codes (5b) — and does **not**
    strip HTML comments from stream content (V1-D7).

    **Per-entry isolation (§5.2.2):** a malformed entry never aborts the batch.

    - **COLLECTOR mode** (``failures`` supplied): a malformed entry's envelope is
      appended to ``failures`` and the entry is omitted from the return; the
      well-formed entries are returned.
    - **STRICT mode** (default, no collector): the first malformed entry raises
      ``ParseError`` carrying its envelope on ``.failure``.

    Args:
        text: The raw entry-stream file content.
        kind: ``"decision"`` or ``"open_question"`` (caller-declared).
        source_path: The originating path, threaded onto each envelope.
        failures: Optional collector. When supplied, malformed entries are
            isolated into it (collector mode) instead of raising (strict mode).

    Returns:
        The well-formed tokenized entries in file order.

    Raises:
        ValueError: If ``kind`` is neither ``"decision"`` nor ``"open_question"``.
        ParseError: In strict mode (no ``failures`` collector), on the first
            malformed entry — with the §5.2.2 envelope on ``.failure``.
    """
    if kind not in ("decision", "open_question"):
        raise ValueError(
            f"parse_entry_stream: unknown kind {kind!r}; "
            "expected 'decision' or 'open_question'"
        )

    lines = text.splitlines()

    # Preamble cut: entries begin the line AFTER the sentinel. The match is the
    # prototype's substring tolerance (not the full comment). No sentinel ->
    # begin_idx 0 -> the whole file is the entry stream.
    begin_idx = 0
    for i, line in enumerate(lines):
        if "BEGIN ENTRIES" in line:
            begin_idx = i + 1
            break

    sections = _split_entry_sections(lines, begin_idx)

    result: List[ParsedEntry] = []
    for sec in sections:
        entry, failure = _validate_section(sec, kind, source_path)
        if failure is None:
            result.append(entry)
        elif failures is not None:
            # Collector mode: isolate this entry, keep parsing the rest.
            failures.append(failure)
        else:
            # Strict mode: raise on the first malformed entry, carrying the
            # envelope. The ParseError's line range is taken from the first item
            # (so the slug-less header still reports its header line), falling
            # back to the entry span when an item has no line.
            first = failure.items[0]
            ls = first.line_start if first.line_start is not None else failure.line_start
            le = first.line_end if first.line_end is not None else failure.line_end
            raise ParseError(first.message, ls, le, failure=failure)

    return result


def read_text_or_none(path: str) -> Optional[str]:
    """Reads a UTF-8 file, returning ``None`` if it does not exist.

    A missing buffer or archive is a no-op stream, never a crash — an absent
    ``questions.md`` is the live-corpus reality today, both at the one-time cutover
    and in steady-state ``mitos sync`` open-question ingestion (V1b Phase 4a).

    Args:
        path: The file to read.

    Returns:
        The file text, or ``None`` if the file is absent.
    """
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def parse_file_reversed(
    path: str, kind: str, failures: List[EntryFailure]
) -> List[ParsedEntry]:
    """Parses one corpus file in collector mode and reverses it to oldest-first.

    Each corpus file is authored **newest-first** (the ``BEGIN ENTRIES … newest
    first`` convention), so reversing the parsed list yields oldest-first *within*
    the file, landing an older in-buffer entry before a newer one that references
    it. Collector mode (``failures`` supplied) isolates malformed entries into
    ``failures`` instead of raising, so all defects across all files aggregate
    before the caller decides to abort.

    This is the **single source** of the newest-first→oldest-first convention,
    shared by the one-time cutover replay (``cutover.py``) and steady-state
    ``mitos sync`` ingestion (``sync.py``, Phase 4a) — never replicated inline (two
    ``reverse`` definitions drift). The oldest-first order is a *flow heuristic*,
    not a correctness mechanism: its age-proxy is file position, so a
    convention-violating batch simply falls through to the caller's quarantine /
    fixpoint, never a wrong commit.

    Args:
        path: The corpus file to parse (absent → empty stream).
        kind: ``"decision"`` or ``"open_question"`` (caller-declared, V1-D8).
        failures: The shared collector for malformed-entry envelopes.

    Returns:
        The well-formed entries, oldest-first within this file (empty when the
        file is absent).
    """
    text = read_text_or_none(path)
    if text is None:
        return []
    entries = parse_entry_stream(text, kind, source_path=path, failures=failures)
    entries.reverse()
    return entries

