"""Archive rotation's core — plan the removal, replace each archive whole, then the buffer.

Rotation is the only code in mitos that removes text from the gold source, so its
writes run in one order: every archive is durably replaced, whole, before the buffer
is touched, and the buffer is then replaced whole. A failure anywhere leaves the
buffer as it was and each archive either as it was or holding the rotated blocks in
full. A failure after an archive lands leaves a copy of the rotated blocks in both
files, which is harmless because a replayed block mints the same content-hash node
twice. There is nothing to roll back, which is why this sequence is not routed through
``splice_buffer`` (ADR
``archive-first-makes-rotations-buffer-write-rollback-free-so-it-keeps-its-own-sequence``).
An archive is never appended to: a torn append followed by a retry parses cleanly, and
can replay truncated commentary over the whole copy (ADR
``rotation-rewrites-each-archive-whole-rather-than-appending-to-it``).

Every corpus file is authored newest-first and read reversed, so a batch goes in at
the **top** of the archive's entry stream in reverse batch order: the caller hands the
batch in commit order (oldest first), the newest block lands highest, and the file
replays oldest-first exactly as the buffer would have. The insertion point is the
archive's first entry heading, below any ``BEGIN ENTRIES`` sentinel and any preamble,
by the same rules the parser splits sections with (``mitos.markers``), so a hand-written
paragraph at the top of an archive stays a preamble instead of becoming the last field
of the block inserted above it. An archive with no entry takes the batch at its end
(ADR ``rotation-inserts-each-batch-newest-first-at-the-top-of-the-archives-entry-stream``).

Archives hold both heading forms — the legacy dated ``## YYYY-MM-DD — slug — title``
and the current ``### slug`` — and the parser reads both from one file.

The core never prints and never prompts, reads no config, graph or clock, and knows
nothing of the trigger or the calling verb. Its inputs are injected: the lock (the
caller's own instance — a second ``FileLock`` on the same path deadlocks against it),
the buffer path, the archive directory, and the blocks, each carrying its archive
file name as ``archive_name_for`` computes it from the instant the caller rotates at.
A caller that chooses the blocks from the buffer's own contents passes a selector
instead (``rotate_selected``): it is handed the core's one read of the live buffer,
inside the lock, so the choice and the removal come from the same bytes (ADR
``rotations-eligibility-set-and-removal-derive-from-one-read-of-one-file``). Whatever
the selector reads, it reads — the core still reads nothing but files.

Tier 2: stdlib plus the leaves ``mitos.atomic_file`` and ``mitos.markers``. The lock
arrives as an object, so ``filelock`` is not imported here.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, ContextManager, Dict, List, Sequence, Tuple

from mitos import atomic_file
from mitos.markers import first_entry_index


@dataclass(frozen=True)
class RotationBlock:
    """One committed entry to move from the buffer to an archive.

    Attributes:
        label: The entry's slug, used only to name it in the outcome.
        raw_text: The block exactly as sliced from the buffer text it will be matched
            against.
        archive_name: The basename in the archive directory (e.g. ``2026-Q3.md``).
            The caller supplies it from ``archive_name_for`` — one name per batch, the
            quarter of the rotation instant, so that a later batch never files into an
            earlier quarter and the archives replay in the order the buffer held. Its
            shape must match the archive reader's filename shape or ``mitos rebuild``
            skips the file.
    """

    label: str
    raw_text: str
    archive_name: str


@dataclass
class RotationPlan:
    """The pure result of matching a batch against one read of the buffer.

    Attributes:
        new_buffer: The buffer with every rotated block's span removed.
        archive_texts: Archive name → the text to insert at the top of that
            archive's entry stream: the rotated blocks in **reverse** batch order
            (newest first), each line-terminated and followed by one blank line.
            Keys are in first-appearance order over the batch.
        rotated: The blocks that matched exactly once, in batch order.
        unmatched: Labels of blocks with no line-anchored match (or empty text).
        duplicated: ``(label, count)`` for blocks matching more than once.
        overlapping: Labels of blocks whose single match overlaps another's.
    """

    new_buffer: str
    archive_texts: Dict[str, str]
    rotated: List[RotationBlock]
    unmatched: List[str]
    duplicated: List[Tuple[str, int]]
    overlapping: List[str]


@dataclass
class RotationOutcome:
    """What a rotation did, as data for each caller to word.

    Attributes:
        rotated: The blocks that moved, in batch order.
        unmatched: Labels left in place because no match was found.
        duplicated: ``(label, count)`` left in place because the block is not unique.
        overlapping: Labels left in place because their match overlaps another block.
        archive_paths: The archive files written, in write order.
    """

    rotated: List[RotationBlock]
    unmatched: List[str]
    duplicated: List[Tuple[str, int]]
    overlapping: List[str]
    archive_paths: List[str]


def archive_name_for(instant: str) -> str:
    """Names the archive file for the instant a batch is rotated at.

    The name is the instant's UTC quarter, ``{year}-Q{quarter}.md``. It says when the
    batch was archived — never when a decision was made, and not when the graph first
    saw it. Naming for the rotation instant is what keeps the archives replayable:
    rotation drains the buffer's oldest end, so successive batches carry
    non-decreasing instants and the quarter files, read oldest first, hold the entries
    in the order the buffer did. A catch-up drain files old entries under the current
    quarter, which is accepted: the filename is a hint, never an address, and no
    consumer may derive an entry's archive file from graph state (ADR
    ``archive-filename-is-not-a-derivable-function-of-graph-state``). The shape is the
    writer half of a hand-agreed contract whose reader is
    ``cutover._ARCHIVE_FILENAME_RE``.

    Parsing a stamp is not reading a clock: the caller reads its clock once per
    batch and passes the stamp in, so this core stays clock-free.

    Args:
        instant: An ISO-8601 stamp with a UTC offset, as ``store._utc_now_iso``
            writes one (MI-10).

    Returns:
        The archive basename, e.g. ``"2026-Q3.md"``.

    Raises:
        ValueError: If the stamp does not parse, or carries no offset — converting a
            naive stamp would silently assume local time.
    """
    try:
        parsed = datetime.fromisoformat(instant)
    except (TypeError, ValueError) as e:
        raise ValueError(f"instant {instant!r} is not an ISO-8601 stamp") from e
    if parsed.utcoffset() is None:
        raise ValueError(
            f"instant {instant!r} carries no UTC offset, so its quarter is unknown"
        )
    utc = parsed.astimezone(timezone.utc)
    return f"{utc.year}-Q{(utc.month - 1) // 3 + 1}.md"


def plan_rotation(buffer_text: str, blocks: Sequence[RotationBlock]) -> RotationPlan:
    """Plans which blocks leave the buffer, in one pass and before any write.

    A block is removable when its whole ``raw_text`` occurs exactly once starting at a
    line start and ending at a line end (or, for text without a final newline, at the
    end of the buffer). Two or more such occurrences exclude it as ``duplicated``; none (or an
    empty ``raw_text``) as ``unmatched``; a single match overlapping another block's
    match excludes both as ``overlapping``. Exclusion is per block, so the rest of
    the batch still rotates. Matching is line-anchored so text quoted mid-line inside
    another entry is never counted or cut.

    Args:
        buffer_text: One read of the live buffer.
        blocks: The batch, in commit order (oldest first).

    Returns:
        The plan. For unique, non-overlapping blocks ``new_buffer`` equals the
        sequential ``buffer.replace(raw, "")`` result, and each archive text is the
        accumulation of ``_terminated(raw)`` over the batch in **reverse** order.
    """
    # Line text (without its "\n") → the offsets where such a line starts.
    line_starts: Dict[str, List[int]] = {}
    offset = 0
    length = len(buffer_text)
    while offset <= length:
        newline = buffer_text.find("\n", offset)
        end = length if newline == -1 else newline
        line_starts.setdefault(buffer_text[offset:end], []).append(offset)
        if newline == -1:
            break
        offset = newline + 1

    unmatched: List[str] = []
    duplicated: List[Tuple[str, int]] = []
    spans: List[Tuple[int, int, int]] = []  # (start, end, batch index)
    for index, block in enumerate(blocks):
        raw = block.raw_text
        if not raw:
            unmatched.append(block.label)
            continue
        first_line = raw.split("\n", 1)[0]
        # Anchored at both ends: a block with no final newline (the snapshot's last
        # line) matches only at the end of the buffer, so a last line extended since
        # the snapshot is a changed block, never a prefix cut out of its own line.
        ends_on_line = raw.endswith("\n")
        matches = [
            start for start in line_starts.get(first_line, ())
            if buffer_text.startswith(raw, start)
            and (ends_on_line or start + len(raw) == length)
        ]
        if not matches:
            unmatched.append(block.label)
        elif len(matches) > 1:
            duplicated.append((block.label, len(matches)))
        else:
            spans.append((matches[0], matches[0] + len(raw), index))

    spans.sort()
    overlapping_indices = set()
    reach_end, reach_index = -1, -1
    for start, end, index in spans:
        if start < reach_end:
            overlapping_indices.update((index, reach_index))
        if end > reach_end:
            reach_end, reach_index = end, index

    kept = [span for span in spans if span[2] not in overlapping_indices]
    pieces: List[str] = []
    cursor = 0
    for start, end, _index in kept:
        pieces.append(buffer_text[cursor:start])
        cursor = end
    pieces.append(buffer_text[cursor:])

    removable = {index for _start, _end, index in kept}
    rotated = [block for index, block in enumerate(blocks) if index in removable]
    # Newest first: the caller's batch is in commit order, and every corpus file is
    # read reversed, so the last-committed block goes highest in the file.
    archive_texts: Dict[str, str] = {block.archive_name: "" for block in rotated}
    for block in reversed(rotated):
        archive_texts[block.archive_name] += _terminated(block.raw_text)

    return RotationPlan(
        new_buffer="".join(pieces),
        archive_texts=archive_texts,
        rotated=rotated,
        unmatched=unmatched,
        duplicated=duplicated,
        overlapping=[blocks[index].label for index in sorted(overlapping_indices)],
    )


def rotate(
    lock: ContextManager,
    buffer_path: str,
    archive_dir: str,
    blocks: Sequence[RotationBlock],
) -> RotationOutcome:
    """Moves a batch of blocks from the buffer to their archives, archive first.

    Under ``lock``: reads the live buffer (never a caller's copy — a whole-file
    replace computed from a copy discards every arrival since it was taken), plans,
    reads every target archive, durably replaces each archive whole with its rotated
    blocks inserted newest-first at the top of its entry stream (``_insert_at_top``),
    then replaces the buffer. Nothing is written when nothing matched.

    Args:
        lock: The caller's lock instance for the buffer.
        buffer_path: The buffer file to remove the blocks from.
        archive_dir: The directory holding the archive files.
        blocks: The batch.

    Returns:
        The outcome, including the blocks left in place and why.

    Raises:
        OSError: If a read or write failed. The buffer is unchanged; an archive may
            already hold a copy of the rotated blocks.
        UnicodeError: If the buffer or an archive cannot be decoded, or the text
            encoded. The buffer is unchanged.
    """
    return rotate_selected(lock, buffer_path, archive_dir, lambda _text: blocks)


def rotate_selected(
    lock: ContextManager,
    buffer_path: str,
    archive_dir: str,
    select: Callable[[str], Sequence[RotationBlock]],
) -> RotationOutcome:
    """Moves the blocks a selector chooses from the live buffer, archive first.

    Exactly :func:`rotate`, except the batch is chosen under the lock from the same
    one read the removal is planned against: ``select(buffer_text)`` runs after the
    read and before :func:`plan_rotation`. A selector that raises propagates before
    anything is written.

    Args:
        lock: The caller's lock instance for the buffer.
        buffer_path: The buffer file to remove the blocks from.
        archive_dir: The directory holding the archive files.
        select: Called once with the live buffer text; returns the batch in commit
            order (oldest first), each block sliced from that text.

    Returns:
        The outcome, including the blocks left in place and why.

    Raises:
        OSError: If a read or write failed. The buffer is unchanged; an archive may
            already hold a copy of the rotated blocks.
        UnicodeError: If the buffer or an archive cannot be decoded, or the text
            encoded. The buffer is unchanged.
        Exception: Whatever ``select`` raises. Nothing has been written.
    """
    archive_paths: List[str] = []
    with lock:
        with open(buffer_path, "r", encoding="utf-8") as fh:
            buffer_text = fh.read()
        blocks = select(buffer_text)
        plan = plan_rotation(buffer_text, blocks)
        if plan.rotated:
            # Every archive is read before any is written, so a late read or decode
            # fault leaves every file as it was.
            archive_writes: List[Tuple[str, str]] = []
            for name, text in plan.archive_texts.items():
                archive_path = os.path.join(archive_dir, name)
                existing = _read_archive(archive_path)
                archive_writes.append((archive_path, _insert_at_top(existing, text)))
            for archive_path, content in archive_writes:
                atomic_file.ensure_parent_directory(archive_path)
                atomic_file.write_source(archive_path, content)
                archive_paths.append(archive_path)
            atomic_file.write_source(buffer_path, plan.new_buffer)

    return RotationOutcome(
        rotated=plan.rotated,
        unmatched=plan.unmatched,
        duplicated=plan.duplicated,
        overlapping=plan.overlapping,
        archive_paths=archive_paths,
    )


def _terminated(raw_text: str) -> str:
    """Returns a block ready to stack: line-terminated, then one blank line.

    A snapshot's last block can end without a newline; terminating it keeps the block
    below it a heading on its own line. The blank line after every block is what the
    rotated span usually carries anyway, so an ordinary block's bytes are unchanged.
    """
    return raw_text + ("" if raw_text.endswith("\n") else "\n") + "\n"


def _insert_at_top(existing: str, text: str) -> str:
    """Puts ``text`` at the top of an archive's entry stream.

    The stream's top is its first entry heading as the parser finds it — after any
    ``BEGIN ENTRIES`` sentinel, outside a transcript span. Whatever precedes that
    heading is preamble and stays preamble; whatever follows is older entries, which
    ``text`` now sits above. An archive holding no entry takes ``text`` at its end,
    after a newline when its last line lacks one, so a heading is never glued onto a
    hand-edited line.
    """
    lines = existing.splitlines(keepends=True)
    at = first_entry_index(lines)
    head = "".join(lines[:at])
    separator = "\n" if head and not head.endswith("\n") else ""
    return head + separator + text + "".join(lines[at:])


def _read_archive(path: str) -> str:
    """Reads an archive whole, or ``""`` when it does not exist yet.

    Only absence reads as empty. Any other fault — a non-directory in the path, a
    refused permission, undecodable bytes — propagates before anything is written.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return ""
