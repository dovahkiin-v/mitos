"""Entry-stream markers — the one home for the rules that locate entries in a corpus file.

A corpus file (the ``decisions.md`` buffer, ``questions.md``, an archive) is a preamble
followed by an entry stream. Two modules have to agree about where that stream begins
and where its first entry starts: the parser splits the stream into sections by these
rules, and rotation inserts a batch at the top of the stream, above the first entry
and below any preamble, so a rotated block never becomes the tail of a hand-written
paragraph. Both compose the predicates here; neither restates them, so the two cannot
drift apart (the parser is the reader of what rotation writes, and a disagreement
between them is a silent loss on ``mitos rebuild``).

The rules, as the parser applies them:

* The stream begins on the line after the first line holding ``BEGIN ENTRIES``
  outside an inline-code span. A file without one is wholly an entry stream.
* An entry starts on a line beginning ``##`` but not ``####``, outside a
  ``[DECISION_TRANSCRIPT]`` … ``[/DECISION_TRANSCRIPT]`` span (a heading-shaped line in
  a transcript is transcript text).
* Text between the stream's start and its first entry belongs to no entry and is
  never read back; text after an entry's last field is that field's continuation.

Tier 1: stdlib only, imports nothing from ``mitos``.
"""

import re
from typing import Sequence

ENTRIES_SENTINEL = "BEGIN ENTRIES"
TRANSCRIPT_OPEN = "[DECISION_TRANSCRIPT]"
TRANSCRIPT_CLOSE = "[/DECISION_TRANSCRIPT]"

_INLINE_CODE_RE = re.compile(r'`[^`\n]+`')


def mask_inline_code(line: str) -> str:
    """Blanks the contents of inline-code spans for marker scanning.

    A backtick-quoted token — a documented ``[NOTE: …]``, a quoted
    BEGIN-ENTRIES sentinel — is prose *about* a marker, not the marker, so the
    inline scanners (and sync's structural-token guard, which must agree with
    them) scan a masked copy where span contents are replaced with same-length
    spaces (column positions stay accurate). Single-line spans only; fenced
    blocks are already protected upstream.

    Args:
        line: One raw line of markdown.

    Returns:
        The line with inline-code span contents blanked.
    """
    return _INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), line)


def is_entries_sentinel(line: str) -> bool:
    """Reports whether ``line`` is the sentinel that opens a file's entry stream.

    The match is a substring test over the inline-code-masked line, so a quoted
    sentinel in prose does not open the stream. Only the first such line counts;
    the caller stops scanning at it.

    Args:
        line: One raw line, with or without its line ending.

    Returns:
        ``True`` for the sentinel line.
    """
    return ENTRIES_SENTINEL in mask_inline_code(line)


def is_entry_heading(line: str) -> bool:
    """Reports whether ``line`` has the shape of an entry heading.

    ``##`` and ``###`` open an entry; ``####`` and a single ``#`` do not. Transcript
    state is the caller's: inside a transcript span this shape is literal text.

    Args:
        line: One raw line, with or without its line ending.

    Returns:
        ``True`` for a heading-shaped line.
    """
    return line.startswith("##") and not line.startswith("####")


def entry_stream_start(lines: Sequence[str]) -> int:
    """Returns the index of the first entry-stream line.

    Args:
        lines: The file's lines, with or without line endings.

    Returns:
        The index after the first sentinel line, or ``0`` when the file has none.
    """
    for index, line in enumerate(lines):
        if is_entries_sentinel(line):
            return index + 1
    return 0


def first_entry_index(lines: Sequence[str]) -> int:
    """Returns the index of the line that starts the entry stream's first entry.

    Scans from :func:`entry_stream_start`, transcript-aware, for the first heading
    the parser would open a section on. Rotation inserts a batch at this line, so a
    preamble stays a preamble and the batch's last field never runs into it.

    Args:
        lines: The file's lines, with or without line endings.

    Returns:
        The index of the first entry heading, or ``len(lines)`` when the stream
        holds no entry — a batch then goes at the end of the file.
    """
    in_transcript = False
    for index in range(entry_stream_start(lines), len(lines)):
        stripped = lines[index].strip()
        if not in_transcript and stripped == TRANSCRIPT_OPEN:
            in_transcript = True
            continue
        if in_transcript and stripped == TRANSCRIPT_CLOSE:
            in_transcript = False
            continue
        if not in_transcript and is_entry_heading(lines[index]):
            return index
    return len(lines)
