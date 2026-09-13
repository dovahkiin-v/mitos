"""Archive rotation's core — plan the removal, append the archive, then replace the buffer.

Rotation is the only code in mitos that removes text from the gold source, so its
writes run in one order: the archive append is durable before the buffer is touched,
and the buffer is then replaced whole. A failure anywhere leaves the buffer as it
was; a failure after the append leaves a copy of the rotated blocks in both files,
which is harmless because a replayed block mints the same content-hash node twice.
There is nothing to roll back, which is why this sequence is not routed through
``splice_buffer`` (ADR
``archive-first-makes-rotations-buffer-write-rollback-free-so-it-keeps-its-own-sequence``).

The core never prints and never prompts, reads no config, graph or clock, and knows
nothing of the trigger or the calling verb. Its inputs are injected: the lock (the
caller's own instance — a second ``FileLock`` on the same path deadlocks against it),
the buffer path, the archive directory, and the blocks, each carrying its archive
file name.

Tier 2: stdlib plus ``mitos.atomic_file``. The lock arrives as an object, so
``filelock`` is not imported here.
"""

import os
from dataclasses import dataclass
from typing import ContextManager, Dict, List, Sequence, Tuple

from mitos import atomic_file


@dataclass(frozen=True)
class RotationBlock:
    """One committed entry to move from the buffer to an archive.

    Attributes:
        label: The entry's slug, used only to name it in the outcome.
        raw_text: The block exactly as sliced from the sync snapshot.
        archive_name: The basename in the archive directory (e.g. ``2026-Q3.md``);
            the caller decides it, and it must match the archive reader's filename
            shape or ``mitos rebuild`` skips the file.
    """

    label: str
    raw_text: str
    archive_name: str


@dataclass
class RotationPlan:
    """The pure result of matching a batch against one read of the buffer.

    Attributes:
        new_buffer: The buffer with every rotated block's span removed.
        archive_texts: Archive name → text to append, in first-appearance order.
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
        archive_paths: The archive files appended to, in write order.
    """

    rotated: List[RotationBlock]
    unmatched: List[str]
    duplicated: List[Tuple[str, int]]
    overlapping: List[str]
    archive_paths: List[str]


def plan_rotation(buffer_text: str, blocks: Sequence[RotationBlock]) -> RotationPlan:
    """Plans which blocks leave the buffer, in one pass and before any write.

    A block is removable when its whole ``raw_text`` occurs exactly once starting at a
    line start. Two or more such occurrences exclude it as ``duplicated``; none (or an
    empty ``raw_text``) as ``unmatched``; a single match overlapping another block's
    match excludes both as ``overlapping``. Exclusion is per block, so the rest of
    the batch still rotates. Matching is line-anchored so text quoted mid-line inside
    another entry is never counted or cut.

    Args:
        buffer_text: One read of the live buffer.
        blocks: The batch, in the order its blocks entered the rotation set.

    Returns:
        The plan. For unique, non-overlapping blocks ``new_buffer`` equals the
        sequential ``buffer.replace(raw, "")`` result, and each archive text is the
        ``raw + "\\n"`` accumulation in batch order.
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
        matches = [
            start for start in line_starts.get(first_line, ())
            if buffer_text.startswith(raw, start)
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
    archive_texts: Dict[str, str] = {}
    for block in rotated:
        archive_texts[block.archive_name] = (
            archive_texts.get(block.archive_name, "") + block.raw_text + "\n"
        )

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
    appends each archive durably, then replaces the buffer. Nothing is written when
    nothing matched.

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
        UnicodeError: If the buffer cannot be decoded or the text encoded. The buffer
            is unchanged.
    """
    archive_paths: List[str] = []
    with lock:
        with open(buffer_path, "r", encoding="utf-8") as fh:
            buffer_text = fh.read()
        plan = plan_rotation(buffer_text, blocks)
        if plan.rotated:
            for name, text in plan.archive_texts.items():
                archive_path = os.path.join(archive_dir, name)
                atomic_file.append_source(archive_path, text)
                archive_paths.append(archive_path)
            atomic_file.write_source(buffer_path, plan.new_buffer)

    return RotationOutcome(
        rotated=plan.rotated,
        unmatched=plan.unmatched,
        duplicated=plan.duplicated,
        overlapping=plan.overlapping,
        archive_paths=archive_paths,
    )
