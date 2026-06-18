"""Strict deterministic Markdown parser for Mitos decisions.

This module implements the C5 integration contract (Skill -> Parser) under the
OD1 runtime parsing constraint: strictly structured, deterministic, and loud
on any format violation.
"""

import re
import string
import unicodedata
from typing import List, Dict, Optional, Any, Tuple
from mitos.errors import ParseError

def load_dynamic_field_map() -> Dict[str, str]:
    """Dynamically builds the FIELD_MAP from format-spec.md to enforce C5 single-source truth."""
    import os
    import re
    
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
    
    field_map = {}
    spec_path = os.path.join(os.path.dirname(__file__), "format-spec.md")
    if os.path.exists(spec_path):
        try:
            with open(spec_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Extract fields declared in the markdown list items: - `**Field:**`
            fields_found = re.findall(r'-\s+`\*\*(?P<field>[a-zA-Z -_]+):\*\*`', content)
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
        except Exception:
            pass

    # Safe fallback mapping to ensure baseline fields are always present
    baseline = {
        "decided": "core_axiom",
        "mechanisms": "mechanisms",
        "rejected": "rejected_paths",
        "invalidates if": "invalidates_if",
        "invalidates-if": "invalidates_if",
        "invalidates_if": "invalidates_if",
        "scope": "scope",
        "context": "context",
        "supersedes": "supersedes",
        "amends": "amends",
        "narrows": "narrows",
        "depends-on": "depends_on",
        "depends on": "depends_on",
        "depends_on": "depends_on",
        "resolves": "resolves",
        "questions": "questions_raised",
        "corrects": "corrects",
        "contradicts": "contradicts",
        "derives-from": "derives_from",
        "derives from": "derives_from",
        "derives_from": "derives_from",
        "cites": "cites",
    }
    
    for k, v in baseline.items():
        if k not in field_map:
            field_map[k] = v
            
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
        self.supersedes: Optional[str] = None
        self.corrects: Optional[str] = None
        self.amends: Optional[str] = None
        self.narrows: Optional[str] = None
        self.depends_on: Optional[str] = None
        self.resolves: Optional[str] = None
        self.contradicts: Optional[str] = None
        self.derives_from: Optional[str] = None
        self.cites: Optional[str] = None
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
        if "supersedes" in fields:
            entry.supersedes = " ".join(fields["supersedes"]).strip()
        if "amends" in fields:
            entry.amends = " ".join(fields["amends"]).strip()
        if "narrows" in fields:
            entry.narrows = " ".join(fields["narrows"]).strip()
        if "depends_on" in fields:
            entry.depends_on = " ".join(fields["depends_on"]).strip()
        if "resolves" in fields:
            entry.resolves = " ".join(fields["resolves"]).strip()
        if "corrects" in fields:
            entry.corrects = " ".join(fields["corrects"]).strip()
        if "contradicts" in fields:
            entry.contradicts = " ".join(fields["contradicts"]).strip()
        if "derives_from" in fields:
            entry.derives_from = " ".join(fields["derives_from"]).strip()
        if "cites" in fields:
            entry.cites = " ".join(fields["cites"]).strip()

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


def _tokenize_entry(sec: Dict[str, Any], kind: str) -> ParsedEntry:
    """Tokenizes one pre-split section into a ``ParsedEntry`` of the given kind.

    Permissive by design (4b validates): extracts every field ``FIELD_MAP``
    recognizes, assigns ``**Decided:**`` to the new ``axiom`` attribute (not
    ``core_axiom``), and normalizes the list fields at the C1 boundary. An
    unrecognized ``**Field:**`` line is discarded (it stops field accumulation so
    it can't corrupt a neighbour) rather than rejected — 4a's only hard failure
    is a structurally untokenizable header (no slug).

    Args:
        sec: A section dict from :func:`_split_entry_sections`.
        kind: The caller-declared kind (``"decision"`` / ``"open_question"``).

    Returns:
        The tokenized entry.

    Raises:
        ParseError: If the header carries no slug (structural fail-fast, V1-D1).
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
    transcript_lines: List[str] = []

    for line in sec["lines"][1:]:  # skip the header line
        stripped = line.strip()

        # Transcript span: markers toggle, body is captured verbatim. A field- or
        # header-shaped line inside the span is literal transcript text.
        if not in_transcript and stripped == "[DECISION_TRANSCRIPT]":
            in_transcript = True
            continue
        if in_transcript and stripped == "[/DECISION_TRANSCRIPT]":
            in_transcript = False
            continue
        if in_transcript:
            transcript_lines.append(line)
            continue

        field_match = _FIELD_LINE_RE.match(line)
        if field_match:
            name = field_match.group("field").strip().lower()
            content = field_match.group("content").strip()
            if name in FIELD_MAP:
                current_field = FIELD_MAP[name]
                fields[current_field] = [content]
            else:
                # Permissive: an unrecognized field is not captured and stops
                # accumulation so its content can't bleed into the prior field.
                # Spec-pure field-name recognition + reporting is 4b.
                current_field = None
        elif current_field is not None and stripped:
            # Continuation line for the current field.
            fields[current_field].append(stripped)

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
    for rel in _RELATIONSHIP_FIELDS:
        if rel in fields:
            setattr(entry, rel, " ".join(fields[rel]).strip())

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


def parse_entry_stream(
    text: str,
    kind: str,
    source_path: Optional[str] = None,
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

    This is a pure tokenizer: it does **not** hash (store/5a), **not** validate
    required-field presence or build a structured failure envelope (4b), and
    **not** strip HTML comments from stream content (V1-D7). ``source_path`` is
    accepted and threaded for 4b's failure envelope but is inert here — 4a never
    stats the file or derives kind from it.

    Args:
        text: The raw entry-stream file content.
        kind: ``"decision"`` or ``"open_question"`` (caller-declared).
        source_path: The originating path, threaded for 4b's failure envelope;
            inert in 4a.

    Returns:
        The tokenized entries in file order.

    Raises:
        ValueError: If ``kind`` is neither ``"decision"`` nor ``"open_question"``.
        ParseError: If an entry header is structurally untokenizable (no slug).
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
    return [_tokenize_entry(sec, kind) for sec in sections]

