"""Strict deterministic Markdown parser for Mitos decisions.

This module implements the C5 integration contract (Skill -> Parser) under the
OD1 runtime parsing constraint: strictly structured, deterministic, and loud
on any format violation.
"""

import re
from typing import List, Dict, Optional, Any, Tuple
from mitos.errors import ParseError

# Canonical field normalization mapping
FIELD_MAP: Dict[str, str] = {
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
        }


def parse_decisions_file(text: str) -> List[ParsedEntry]:
    """Parses a markdown write-buffer text deterministically into parsed entries.

    Handles HTML comment stripping, division into sections below the BEGIN ENTRIES
    marker, field parsing, and strict invariant validation.

    Args:
        text: The raw decisions.md file content.

    Returns:
        A list of ParsedEntry objects.

    Raises:
        ParseError: If any structural syntax is malformed.
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

    # 4. Parse the contents of each section
    for sec in sections:
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

            for line in sec["lines"][1:]:  # Skip the header line
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
                        current_field = None  # Ignore unknown fields strictly or skip
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



            parsed_entries.append(entry)

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



            parsed_entries.append(entry)

    return parsed_entries

