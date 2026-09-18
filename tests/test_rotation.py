"""Tests for `mitos.rotation` — the archive-first rotation core.

The planner is pure, so the shapes that lose data (a duplicate, a mid-line quote, an
overlap) are pinned without a sync. The I/O rows pin the order the durability claim
rests on: every archive replace returns before the buffer replace starts. The layout
rows pin where a batch lands: newest-first at the top of the archive's entry stream,
which is what the reversing reader needs to replay it in commit order.
"""

import ast
import contextlib
import errno
import os
import stat
import subprocess
import sys
from unittest.mock import patch

import pytest

from mitos import atomic_file, rotation
from mitos.rotation import RotationBlock, archive_name_for, plan_rotation, rotate

_HEADER = "# Decisions\n<!-- BEGIN ENTRIES -->\n"


def _entry(slug: str, body: str = "") -> str:
    return (
        f"## 2026-05-19 — {slug} — Title\n"
        f"**Decided:** The {slug} axiom.{body}\n"
        f"**Rejected:** The alternative.\n"
        f"**Scope:** core\n"
        "\n"
    )


def _block(slug: str, text: str, archive: str = "2026-Q3.md") -> RotationBlock:
    return RotationBlock(slug, text, archive)


def _sequential(buffer: str, blocks) -> "tuple[str, str]":
    """The pre-1c buffer algorithm, and the archive text as the fix stacks it.

    The buffer half is the old sequential ``replace``. The archive half is the batch
    in reverse order, each block line-terminated and followed by one blank line.
    """
    archived = ""
    rotated = []
    for block in blocks:
        if block.raw_text in buffer:
            buffer = buffer.replace(block.raw_text, "")
            rotated.append(block)
    for block in reversed(rotated):
        raw = block.raw_text
        archived += raw + ("" if raw.endswith("\n") else "\n") + "\n"
    return buffer, archived


# --- P1: classification --------------------------------------------------------------

def test_a_unique_block_rotates():
    a, b = _entry("a"), _entry("b")
    plan = plan_rotation(_HEADER + a + b, [_block("a", a)])
    assert [blk.label for blk in plan.rotated] == ["a"]
    assert plan.new_buffer == _HEADER + b
    assert plan.archive_texts == {"2026-Q3.md": a + "\n"}


def test_an_absent_block_is_unmatched_and_the_buffer_is_unchanged():
    buffer = _HEADER + _entry("b")
    plan = plan_rotation(buffer, [_block("a", _entry("a"))])
    assert plan.unmatched == ["a"] and plan.rotated == []
    assert plan.new_buffer == buffer and plan.archive_texts == {}


def test_a_block_present_twice_is_duplicated_and_nothing_is_removed():
    """An unbounded replace removed both copies while archiving one (defect 3)."""
    a = _entry("a")
    buffer = _HEADER + a + a
    plan = plan_rotation(buffer, [_block("a", a)])
    assert plan.duplicated == [("a", 2)]
    assert plan.rotated == [] and plan.new_buffer == buffer


def test_a_mid_line_quote_is_not_a_match():
    """Text quoted inside another line is neither counted nor cut (line anchoring)."""
    quoted = "**Scope:** core\n"
    host = f"## 2026-05-19 — host — Title\n**Decided:** we quote >{quoted}\n"
    buffer = _HEADER + host
    plan = plan_rotation(buffer, [_block("q", quoted)])
    assert plan.unmatched == ["q"] and plan.new_buffer == buffer


def test_a_mid_line_quote_does_not_make_a_real_block_a_duplicate():
    a = _entry("a")
    first_line = a.split("\n", 1)[0]
    quoting = f"## 2026-05-19 — b — Title\n**Decided:** see {a}"
    plan = plan_rotation(_HEADER + quoting + a, [_block("a", a)])
    assert first_line in quoting
    assert [blk.label for blk in plan.rotated] == ["a"]


def test_a_last_line_extended_since_the_snapshot_is_not_cut_mid_line():
    """A block without a final newline matches only at the buffer's end.

    The snapshot's EOF block carries no trailing newline. If its last line was extended
    before the lock was taken, a start-anchored prefix match would archive the old text
    and leave the extension stranded as a fragment of a line — the pre-1c `replace` did.
    """
    tail = "## 2026-05-19 — e — Title\n**Decided:** x.\n**Rejected:** The old"
    buffer = _HEADER + tail + " and the amended ending.\n"
    assert _sequential(buffer, [_block("e", tail)])[0] == _HEADER + " and the amended ending.\n"

    plan = plan_rotation(buffer, [_block("e", tail)])
    assert plan.unmatched == ["e"] and plan.new_buffer == buffer


def test_overlapping_matches_exclude_both_blocks():
    a, b = _entry("a"), _entry("b")
    buffer = _HEADER + a + b
    whole = _block("whole", a + b)
    head = _block("head", a)
    plan = plan_rotation(buffer, [whole, head])
    assert sorted(plan.overlapping) == ["head", "whole"]
    assert plan.rotated == [] and plan.new_buffer == buffer


def test_an_overlap_leaves_the_rest_of_the_batch_rotating():
    a, b, c = _entry("a"), _entry("b"), _entry("c")
    buffer = _HEADER + a + b + c
    plan = plan_rotation(buffer, [_block("ab", a + b), _block("b", b), _block("c", c)])
    assert sorted(plan.overlapping) == ["ab", "b"]
    assert [blk.label for blk in plan.rotated] == ["c"]
    assert plan.new_buffer == _HEADER + a + b


def test_an_empty_raw_text_is_unmatched():
    """An empty string matches everywhere; the core must not trust its caller."""
    buffer = _HEADER + _entry("a")
    plan = plan_rotation(buffer, [_block("empty", "")])
    assert plan.unmatched == ["empty"] and plan.new_buffer == buffer


# --- P2: byte equivalence with the pre-1c buffer algorithm ----------------------------

def test_unique_blocks_match_sequential_replace_byte_for_byte():
    """The buffer keeps the old bytes; the archive stacks the batch newest-first.

    The EOF block carries no final newline, so its terminator is the one place the
    archive bytes are not ``raw + "\\n"``.
    """
    transcript = "\n```\n> user: why?\n> agent: because\n```\n\n"
    entries = [_entry("a"), _entry("b", " With a blank\n\nparagraph."),
               _entry("c") + transcript, _entry("d")]
    tail = "## 2026-05-19 — e — Title\n**Decided:** No final newline.\n**Rejected:** x"
    buffer = _HEADER + "".join(entries) + tail
    blocks = [_block("d", entries[3]), _block("b", entries[1]),
              _block("e", tail), _block("c", entries[2])]

    plan = plan_rotation(buffer, blocks)
    expected_buffer, expected_archive = _sequential(buffer, blocks)

    assert [blk.label for blk in plan.rotated] == ["d", "b", "e", "c"]
    assert plan.new_buffer == expected_buffer
    assert plan.archive_texts == {"2026-Q3.md": expected_archive}
    assert expected_archive == (entries[2] + "\n" + tail + "\n\n" + entries[1] + "\n"
                                + entries[3] + "\n")


def test_a_batch_stacks_newest_first_and_reads_back_in_commit_order(tmp_path):
    """L1: the caller's batch is commit order; the file is read reversed.

    Two batches into one archive: the second's blocks sit above the first's, and
    inside each batch the last-committed block is highest, so ``parse_file_reversed``
    yields the eight in the order they committed.
    """
    from mitos.parser import parse_file_reversed

    first = [_entry(f"first-{i}") for i in range(4)]
    second = [_entry(f"second-{i}") for i in range(4)]
    buffer_path, archive_dir = _workspace(tmp_path, _HEADER + "".join(first + second))
    for batch in (first, second):
        rotate(contextlib.nullcontext(), buffer_path, archive_dir,
               [_block(t.split(" — ")[1], t) for t in batch])

    target = os.path.join(archive_dir, "2026-Q3.md")
    with open(target, encoding="utf-8") as fh:
        assert fh.read() == "".join(t + "\n" for t in list(reversed(second)) + list(reversed(first)))
    failures = []
    slugs = [e.slug for e in parse_file_reversed(target, "decision", failures)]
    assert failures == []
    assert slugs == [f"first-{i}" for i in range(4)] + [f"second-{i}" for i in range(4)]


# --- I/O rows ------------------------------------------------------------------------

def _workspace(tmp_path, buffer: str):
    buffer_path = tmp_path / "decisions.md"
    buffer_path.write_text(buffer, encoding="utf-8")
    return str(buffer_path), str(tmp_path / "decisions" / "archive")


def _temps(directory) -> list:
    return [n for n in os.listdir(directory) if n.endswith(".tmp")]


def _is_dir_fd(fd: int) -> bool:
    return stat.S_ISDIR(os.fstat(fd).st_mode)


def _read_bytes(path) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def test_blocks_for_two_archives_write_two_files_before_the_buffer(tmp_path):
    """P3: 1d's `created_at` partition is a caller-only change — one batch, two files.

    Each archive is a whole-file replace, and every one lands before the buffer's.
    """
    a, b = _entry("a"), _entry("b")
    buffer_path, archive_dir = _workspace(tmp_path, _HEADER + a + b)
    events = []
    real_write = atomic_file.write_source

    def _write(path, content):
        events.append(("write", os.path.basename(path)))
        return real_write(path, content)

    with patch("mitos.atomic_file.write_source", side_effect=_write):
        outcome = rotate(contextlib.nullcontext(), buffer_path, archive_dir,
                         [_block("a", a, "2026-Q2.md"), _block("b", b, "2026-Q3.md")])

    assert events == [("write", "2026-Q2.md"), ("write", "2026-Q3.md"),
                      ("write", "decisions.md")]
    assert [os.path.basename(p) for p in outcome.archive_paths] == ["2026-Q2.md", "2026-Q3.md"]
    with open(os.path.join(archive_dir, "2026-Q2.md"), encoding="utf-8") as fh:
        assert fh.read() == a + "\n"
    with open(os.path.join(archive_dir, "2026-Q3.md"), encoding="utf-8") as fh:
        assert fh.read() == b + "\n"
    with open(buffer_path, encoding="utf-8") as fh:
        assert fh.read() == _HEADER


def test_the_archive_is_fsynced_before_the_buffer(tmp_path):
    """P4: a page-cache-only archive reopens D1's window one layer down."""
    a = _entry("a")
    buffer_path, archive_dir = _workspace(tmp_path, _HEADER + a)
    synced_inodes = []
    real_fsync = os.fsync

    def _fsync(fd):
        synced_inodes.append(os.fstat(fd).st_ino)
        return real_fsync(fd)

    with patch("mitos.atomic_file.os.fsync", side_effect=_fsync):
        rotate(contextlib.nullcontext(), buffer_path, archive_dir, [_block("a", a)])

    archive_ino = os.stat(os.path.join(archive_dir, "2026-Q3.md")).st_ino
    buffer_ino = os.stat(buffer_path).st_ino  # the temp file's inode, after the replace
    assert archive_ino in synced_inodes and buffer_ino in synced_inodes
    assert synced_inodes.index(archive_ino) < synced_inodes.index(buffer_ino)


def test_a_failed_buffer_replace_leaves_the_buffer_whole(tmp_path):
    """The archive replace lands and the buffer's is refused: a copy in both, harmless (M2).

    The injection is filtered to the buffer, because the archive is a replace too.
    """
    a = _entry("a")
    buffer = _HEADER + a
    buffer_path, archive_dir = _workspace(tmp_path, buffer)
    buffer_real = os.path.realpath(buffer_path)
    real_replace = os.replace
    fired = []

    def _replace(src, dst, *args, **kwargs):
        if os.path.realpath(dst) == buffer_real:
            fired.append(dst)
            raise OSError("disk gone")
        return real_replace(src, dst, *args, **kwargs)

    with patch("mitos.atomic_file.os.replace", side_effect=_replace):
        with pytest.raises(OSError):
            rotate(contextlib.nullcontext(), buffer_path, archive_dir, [_block("a", a)])

    assert len(fired) == 1, "the buffer replace must be attempted exactly once"
    with open(buffer_path, encoding="utf-8") as fh:
        assert fh.read() == buffer
    with open(os.path.join(archive_dir, "2026-Q3.md"), encoding="utf-8") as fh:
        assert fh.read() == a + "\n"
    assert _temps(tmp_path) == [] and _temps(archive_dir) == []


# --- 1d: the archive is a whole-file replace ------------------------------------------

def test_f3_a_refused_archive_fsync_leaves_the_archive_byte_identical(tmp_path):
    """F3 (CC-13): an append has already put its bytes down when its fsync raises.

    A torn tail followed by a retry's re-append never fails to parse, and with a
    newline guard the truncated commentary wins on replay. So the archive write must
    land whole or not at all.
    """
    a = _entry("a")
    buffer = _HEADER + a
    buffer_path, archive_dir = _workspace(tmp_path, buffer)
    os.makedirs(archive_dir)
    archive = os.path.join(archive_dir, "2026-Q3.md")
    with open(archive, "w", encoding="utf-8") as fh:
        fh.write(_entry("settled") + "\n")
    archive_before, buffer_before = _read_bytes(archive), _read_bytes(buffer_path)
    real_fsync = os.fsync
    refused = []

    def _fsync(fd):
        if not refused and not _is_dir_fd(fd):
            refused.append(fd)
            raise OSError(errno.EIO, "I/O error")
        return real_fsync(fd)

    with patch("mitos.atomic_file.os.fsync", side_effect=_fsync):
        with pytest.raises(OSError):
            rotate(contextlib.nullcontext(), buffer_path, archive_dir, [_block("a", a)])

    assert len(refused) == 1
    assert _read_bytes(archive) == archive_before
    assert _read_bytes(buffer_path) == buffer_before
    assert _temps(archive_dir) == [] and _temps(tmp_path) == []


def test_f4_a_block_rotated_into_an_archive_without_a_final_newline_parses_back(tmp_path):
    """F4: a hand-edited archive ending mid-line must not swallow the next heading.

    ``PRIOR`` is an archive with no entry, so its stream's top is the end of the file
    and the batch goes after it — glued onto ``PRIOR``, the heading is no heading, and
    the rotated entry silently vanishes from the stream with no failure reported.
    """
    from mitos.parser import parse_file_reversed

    a = _entry("rotate-first")
    buffer_path, archive_dir = _workspace(tmp_path, _HEADER + a)
    os.makedirs(archive_dir)
    archive = os.path.join(archive_dir, "2026-Q3.md")
    with open(archive, "w", encoding="utf-8") as fh:
        fh.write("PRIOR")

    rotate(contextlib.nullcontext(), buffer_path, archive_dir, [_block("rotate-first", a)])

    failures = []
    assert [e.slug for e in parse_file_reversed(archive, "decision", failures)] == [
        "rotate-first"]
    assert failures == []
    with open(archive, encoding="utf-8") as fh:
        assert fh.read() == "PRIOR\n" + a + "\n"


def test_an_archive_ending_in_a_newline_gains_no_separator(tmp_path):
    """The guard fires only on a missing final newline; mitos's own files never need it."""
    a = _entry("a")
    buffer_path, archive_dir = _workspace(tmp_path, _HEADER + a)
    os.makedirs(archive_dir)
    archive = os.path.join(archive_dir, "2026-Q3.md")
    with open(archive, "w", encoding="utf-8") as fh:
        fh.write("PRIOR\n")

    rotate(contextlib.nullcontext(), buffer_path, archive_dir, [_block("a", a)])

    with open(archive, encoding="utf-8") as fh:
        assert fh.read() == "PRIOR\n" + a + "\n"


def test_f5_a_batch_lands_below_an_archives_preamble_and_above_its_first_entry(tmp_path):
    """F5: prose at the top of a hand-edited archive stays a preamble.

    The parser folds any non-blank line after an entry's last field into that field,
    blank lines or not. Inserted at line 1, the batch would end right above ``PRIOR``
    and the rotated entry's ``Scope`` would read ``core PRIOR`` — a silent change to
    the gold source. So the batch goes in at the first entry heading, and every field
    of both entries reads back as authored.
    """
    from mitos.parser import parse_file_reversed

    settled, rotated = _entry("settled"), _entry("rotate-first")
    buffer_path, archive_dir = _workspace(tmp_path, _HEADER + rotated)
    os.makedirs(archive_dir)
    archive = os.path.join(archive_dir, "2026-Q3.md")
    with open(archive, "w", encoding="utf-8") as fh:
        fh.write("PRIOR prose the archivist left here.\n\n" + settled)

    rotate(contextlib.nullcontext(), buffer_path, archive_dir,
           [_block("rotate-first", rotated)])

    with open(archive, encoding="utf-8") as fh:
        assert fh.read() == "PRIOR prose the archivist left here.\n\n" + rotated + "\n" + settled
    failures = []
    entries = {e.slug: e for e in parse_file_reversed(archive, "decision", failures)}
    assert failures == [] and sorted(entries) == ["rotate-first", "settled"]
    for slug, entry in entries.items():
        assert entry.axiom == f"The {slug} axiom."
        assert entry.rejected_paths == "The alternative."
        assert entry.scope == ["core"], "the preamble was folded into the last field"


def test_f6_a_batch_lands_below_the_sentinel_and_leaves_the_sample_above_it_alone(tmp_path):
    """F6: an archive shaped like the buffer — a sample entry above ``BEGIN ENTRIES``.

    The sample is preamble by the parser's rule and must stay outside the stream; the
    batch goes below the sentinel and above the first real entry.
    """
    from mitos.parser import parse_file_reversed

    sample = "## SAMPLE\n\n### example-slug\n\n**Decided:** A sample.\n**Rejected:** x.\n\n"
    sentinel = "<!-- BEGIN ENTRIES — newest first -->\n\n"
    settled, rotated = _entry("settled"), _entry("rotate-first")
    buffer_path, archive_dir = _workspace(tmp_path, _HEADER + rotated)
    os.makedirs(archive_dir)
    archive = os.path.join(archive_dir, "2026-Q3.md")
    with open(archive, "w", encoding="utf-8") as fh:
        fh.write(sample + sentinel + settled)

    rotate(contextlib.nullcontext(), buffer_path, archive_dir,
           [_block("rotate-first", rotated)])

    with open(archive, encoding="utf-8") as fh:
        assert fh.read() == sample + sentinel + rotated + "\n" + settled
    failures = []
    slugs = [e.slug for e in parse_file_reversed(archive, "decision", failures)]
    assert failures == [] and slugs == ["settled", "rotate-first"]


def test_a_sentinel_without_a_final_newline_gains_a_separator_before_the_batch(tmp_path):
    """An archive that is only a sentinel line, unterminated: the batch starts a new line."""
    a = _entry("a")
    buffer_path, archive_dir = _workspace(tmp_path, _HEADER + a)
    os.makedirs(archive_dir)
    archive = os.path.join(archive_dir, "2026-Q3.md")
    with open(archive, "w", encoding="utf-8") as fh:
        fh.write("<!-- BEGIN ENTRIES -->")

    rotate(contextlib.nullcontext(), buffer_path, archive_dir, [_block("a", a)])

    with open(archive, encoding="utf-8") as fh:
        assert fh.read() == "<!-- BEGIN ENTRIES -->\n" + a + "\n"


def test_a_heading_inside_a_preamble_transcript_is_not_the_insertion_point(tmp_path):
    """The insertion scan is transcript-aware, as the parser's section split is."""
    from mitos.parser import parse_file_reversed

    preamble = ("[DECISION_TRANSCRIPT]\n## not a heading\n[/DECISION_TRANSCRIPT]\n\n")
    settled, rotated = _entry("settled"), _entry("rotate-first")
    buffer_path, archive_dir = _workspace(tmp_path, _HEADER + rotated)
    os.makedirs(archive_dir)
    archive = os.path.join(archive_dir, "2026-Q3.md")
    with open(archive, "w", encoding="utf-8") as fh:
        fh.write(preamble + settled)

    rotate(contextlib.nullcontext(), buffer_path, archive_dir,
           [_block("rotate-first", rotated)])

    with open(archive, encoding="utf-8") as fh:
        assert fh.read() == preamble + rotated + "\n" + settled
    failures = []
    slugs = [e.slug for e in parse_file_reversed(archive, "decision", failures)]
    assert failures == [] and slugs == ["settled", "rotate-first"]


def test_an_undecodable_second_archive_changes_no_file(tmp_path):
    """Every archive is read before any is written, so a late decode fault writes nothing."""
    a, b = _entry("a"), _entry("b")
    buffer_path, archive_dir = _workspace(tmp_path, _HEADER + a + b)
    os.makedirs(archive_dir)
    second = os.path.join(archive_dir, "2026-Q3.md")
    with open(second, "wb") as fh:
        fh.write(b"PRIOR \xff\xfe not utf-8\n")
    buffer_before = _read_bytes(buffer_path)

    with pytest.raises(UnicodeDecodeError):
        rotate(contextlib.nullcontext(), buffer_path, archive_dir,
               [_block("a", a, "2026-Q2.md"), _block("b", b, "2026-Q3.md")])

    assert sorted(os.listdir(archive_dir)) == ["2026-Q3.md"]
    assert _read_bytes(second) == b"PRIOR \xff\xfe not utf-8\n"
    assert _read_bytes(buffer_path) == buffer_before


def test_an_existing_archive_keeps_its_mode_and_gains_plain_text_mode_bytes(tmp_path):
    """A rewrite that re-modes would widen or narrow a file the user permissioned."""
    a = _entry("a", " Sprendimas — ąčę ✓")
    buffer_path, archive_dir = _workspace(tmp_path, _HEADER + a)
    os.makedirs(archive_dir)
    archive = os.path.join(archive_dir, "2026-Q3.md")
    reference = tmp_path / "reference.md"
    for path in (archive, reference):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("PRIOR\n")
    with open(reference, "a", encoding="utf-8") as fh:
        fh.write(a + "\n")
    os.chmod(archive, 0o640)
    umask = os.umask(0)
    os.umask(umask)
    assert 0o640 != 0o666 & ~umask, "non-vacuity: the kept mode must not be the default"

    rotate(contextlib.nullcontext(), buffer_path, archive_dir, [_block("a", a)])

    assert _read_bytes(archive) == _read_bytes(reference)
    assert stat.S_IMODE(os.stat(archive).st_mode) == 0o640
    assert _temps(archive_dir) == []


def test_a_fresh_archive_directory_is_durable_before_the_buffer_write(tmp_path):
    """A new archive/ that only the page cache knows about reopens D1 one layer down."""
    a = _entry("a")
    buffer_path, archive_dir = _workspace(tmp_path, _HEADER + a)
    synced = []
    real_fsync = os.fsync

    def _fsync(fd):
        synced.append(os.fstat(fd).st_ino)
        return real_fsync(fd)

    previous = os.umask(0o027)
    try:
        with patch("mitos.atomic_file.os.fsync", side_effect=_fsync):
            rotate(contextlib.nullcontext(), buffer_path, archive_dir, [_block("a", a)])
    finally:
        os.umask(previous)

    archive = os.path.join(archive_dir, "2026-Q3.md")
    assert stat.S_IMODE(os.stat(archive).st_mode) == 0o640
    buffer_at = synced.index(os.stat(buffer_path).st_ino)
    # The file's entry lives in archive/, archive/'s in decisions/, decisions/'s in tmp_path.
    for directory in (archive_dir, os.path.dirname(archive_dir), str(tmp_path)):
        ino = os.stat(directory).st_ino
        assert ino in synced[:buffer_at], f"{directory} was not synced before the buffer"


# --- 1d: archive_name_for ------------------------------------------------------------

@pytest.mark.parametrize("stamp, name", [
    ("2026-03-31T23:59:59.999999+00:00", "2026-Q1.md"),
    ("2026-04-01T00:00:00+00:00", "2026-Q2.md"),
    ("2026-12-31T23:59:59+00:00", "2026-Q4.md"),
    ("2027-01-01T00:00:00+00:00", "2027-Q1.md"),
    ("2026-09-13T17:21:14.640904+00:00", "2026-Q3.md"),
    ("2026-09-13T17:21:14Z", "2026-Q3.md"),
])
def test_the_archive_name_is_the_quarter_of_a_utc_instant(stamp, name):
    """The instant is the caller's rotation clock, passed in as a stamp."""
    assert archive_name_for(stamp) == name


@pytest.mark.parametrize("stamp, name", [
    ("2026-09-30T23:30:00-02:00", "2026-Q4.md"),
    ("2026-10-01T01:00:00+03:00", "2026-Q3.md"),
    ("2026-12-31T22:00:00-05:00", "2027-Q1.md"),
])
def test_an_offset_stamp_is_converted_to_utc_before_it_is_named(stamp, name):
    """Naming the local quarter would file one instant under two names."""
    assert archive_name_for(stamp) == name


@pytest.mark.parametrize("stamp", ["2026-06-23T16:04:18", "2026-09-13"])
def test_a_naive_stamp_is_refused(stamp):
    """Converting it would silently assume local time — the defect being fixed."""
    with pytest.raises(ValueError):
        archive_name_for(stamp)


@pytest.mark.parametrize("stamp", ["garbage", ""])
def test_an_unparseable_stamp_is_refused(stamp):
    with pytest.raises(ValueError):
        archive_name_for(stamp)


def test_every_archive_name_matches_the_archive_readers_filename_shape():
    """The writer half of a hand-agreed contract; the reader's regex is imported, not copied."""
    from mitos.cutover import _ARCHIVE_FILENAME_RE

    names = {
        archive_name_for(f"{year}-{month:02d}-01T00:00:00+00:00")
        for year in (2026, 2027) for month in range(1, 13)
    }
    assert len(names) == 8, "non-vacuity: two years give eight quarters"
    assert all(_ARCHIVE_FILENAME_RE.match(name) for name in names)


class _MutatingLock:
    """A lock whose acquisition prepends an entry — as a capture would before it."""

    def __init__(self, path: str, prepended: str) -> None:
        self.path, self.prepended, self.held = path, prepended, False

    def __enter__(self):
        with open(self.path, encoding="utf-8") as fh:
            text = fh.read()
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(text.replace(_HEADER, _HEADER + self.prepended, 1))
        self.held = True
        return self

    def __exit__(self, *exc):
        self.held = False
        return False


def test_the_one_read_is_the_live_buffer_inside_the_lock(tmp_path):
    """P5: a rotation computed from an earlier copy discards every arrival since."""
    a, arrival = _entry("a"), _entry("arrived")
    buffer_path, archive_dir = _workspace(tmp_path, _HEADER + a)

    rotate(_MutatingLock(buffer_path, arrival), buffer_path, archive_dir, [_block("a", a)])

    with open(buffer_path, encoding="utf-8") as fh:
        assert fh.read() == _HEADER + arrival


def test_a_selector_is_handed_the_live_read_inside_the_lock(tmp_path):
    """3b: the selector chooses from the same bytes the removal is planned against."""
    a, arrival = _entry("a"), _entry("arrived")
    buffer_path, archive_dir = _workspace(tmp_path, _HEADER + a)
    lock = _MutatingLock(buffer_path, arrival)
    seen = []

    def _select(text):
        seen.append((text, lock.held))
        return [_block("a", a)]

    outcome = rotation.rotate_selected(lock, buffer_path, archive_dir, _select)

    assert seen == [(_HEADER + arrival + a, True)], "one call, live text, under the lock"
    assert [b.label for b in outcome.rotated] == ["a"]
    with open(buffer_path, encoding="utf-8") as fh:
        assert fh.read() == _HEADER + arrival


def test_a_selector_that_raises_writes_nothing(tmp_path):
    """3b: a graph read failing inside the selector leaves every file as it was."""
    buffer = _HEADER + _entry("a")
    buffer_path, archive_dir = _workspace(tmp_path, buffer)

    def _select(_text):
        raise OSError("injected: graph read refused")

    with pytest.raises(OSError, match="graph read refused"):
        rotation.rotate_selected(contextlib.nullcontext(), buffer_path, archive_dir, _select)

    assert not os.path.exists(archive_dir)
    with open(buffer_path, encoding="utf-8") as fh:
        assert fh.read() == buffer


def test_entry_heading_indices_follows_the_parsers_section_rule():
    """3b: the count rotation's trigger reads — sentinel, transcript and `####` aware."""
    from mitos.markers import entry_heading_indices, first_entry_index

    lines = (
        "# Decisions\n"
        "### example-slug above the sentinel\n"
        "<!-- BEGIN ENTRIES -->\n"
        "## 2026-05-19 — dated — Title\n"
        "#### not an entry\n"
        "[DECISION_TRANSCRIPT]\n"
        "### inside a transcript\n"
        "[/DECISION_TRANSCRIPT]\n"
        "### current\n"
    ).splitlines(keepends=True)

    assert entry_heading_indices(lines) == [3, 8]
    assert first_entry_index(lines) == 3
    assert entry_heading_indices(["no entries here\n"]) == []
    assert first_entry_index(["no entries here\n"]) == 1


def test_nothing_matchable_writes_nothing(tmp_path):
    """P6: an all-unmatched batch neither rewrites the buffer nor creates an archive."""
    buffer = _HEADER + _entry("b")
    buffer_path, archive_dir = _workspace(tmp_path, buffer)
    os.utime(buffer_path, ns=(1_000_000_000, 1_000_000_000))

    outcome = rotate(contextlib.nullcontext(), buffer_path, archive_dir,
                     [_block("a", _entry("a")), _block("empty", "")])

    assert outcome.archive_paths == [] and outcome.rotated == []
    assert outcome.unmatched == ["a", "empty"]
    assert not os.path.exists(archive_dir)
    assert os.stat(buffer_path).st_mtime_ns == 1_000_000_000
    with open(buffer_path, encoding="utf-8") as fh:
        assert fh.read() == buffer


# --- P7: structure -------------------------------------------------------------------

def _called_names(source: str) -> set:
    return {
        node.func.id for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_the_core_holds_no_input_and_no_print():
    """A prompt wedges a caller without a terminal; a print corrupts the MCP stream."""
    with open(rotation.__file__, encoding="utf-8") as fh:
        names = _called_names(fh.read())
    assert "open" in names, "non-vacuity: the sweep must see the module's own calls"
    assert _called_names("print('x'); input()") >= {"print", "input"}
    assert names.isdisjoint({"print", "input"})


def test_importing_the_core_pulls_in_only_the_leaves():
    """Tier 2: `config`, `store`, `sync` or `filelock` here would bind it to one workspace.

    `markers` is the second leaf: the insertion rule the parser also composes.
    """
    probe = (
        "import sys; import mitos.rotation; "
        "print(','.join(sorted(m for m in sys.modules "
        "if m == 'mitos' or m.startswith('mitos.')))); "
        "print('filelock' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True)
    assert out.stdout.split() == [
        "mitos,mitos.atomic_file,mitos.markers,mitos.rotation", "False"]
