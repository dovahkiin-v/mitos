"""Tests for `mitos.rotation` — the archive-first rotation core.

The planner is pure, so the shapes that lose data (a duplicate, a mid-line quote, an
overlap) are pinned without a sync. The I/O rows pin the order the durability claim
rests on: every archive append returns before the buffer replace starts.
"""

import ast
import contextlib
import os
import subprocess
import sys
from unittest.mock import patch

from mitos import atomic_file, rotation
from mitos.rotation import RotationBlock, plan_rotation, rotate

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
    """Today's pre-1c algorithm, for the byte-equivalence rows."""
    archived = ""
    for block in blocks:
        if block.raw_text in buffer:
            buffer = buffer.replace(block.raw_text, "")
            archived += block.raw_text + "\n"
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


# --- P2: byte equivalence with the pre-1c algorithm -----------------------------------

def test_unique_blocks_match_sequential_replace_byte_for_byte():
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


# --- I/O rows ------------------------------------------------------------------------

def _workspace(tmp_path, buffer: str):
    buffer_path = tmp_path / "decisions.md"
    buffer_path.write_text(buffer, encoding="utf-8")
    return str(buffer_path), str(tmp_path / "decisions" / "archive")


def test_blocks_for_two_archives_write_two_files_before_the_buffer(tmp_path):
    """P3: 1d's `created_at` partition is a caller-only change — one batch, two files."""
    a, b = _entry("a"), _entry("b")
    buffer_path, archive_dir = _workspace(tmp_path, _HEADER + a + b)
    events = []
    real_append, real_write = atomic_file.append_source, atomic_file.write_source

    def _append(path, text):
        events.append(("append", os.path.basename(path)))
        return real_append(path, text)

    def _write(path, content):
        events.append(("write", os.path.basename(path)))
        return real_write(path, content)

    with patch("mitos.atomic_file.append_source", side_effect=_append), \
            patch("mitos.atomic_file.write_source", side_effect=_write):
        outcome = rotate(contextlib.nullcontext(), buffer_path, archive_dir,
                         [_block("a", a, "2026-Q2.md"), _block("b", b, "2026-Q3.md")])

    assert events == [("append", "2026-Q2.md"), ("append", "2026-Q3.md"),
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
    a = _entry("a")
    buffer = _HEADER + a
    buffer_path, archive_dir = _workspace(tmp_path, buffer)

    with patch("mitos.atomic_file.os.replace", side_effect=OSError("disk gone")):
        try:
            rotate(contextlib.nullcontext(), buffer_path, archive_dir, [_block("a", a)])
        except OSError:
            pass
        else:
            raise AssertionError("the replace failure must propagate")

    with open(buffer_path, encoding="utf-8") as fh:
        assert fh.read() == buffer
    with open(os.path.join(archive_dir, "2026-Q3.md"), encoding="utf-8") as fh:
        assert fh.read() == a + "\n"


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


def test_importing_the_core_pulls_in_only_the_leaf():
    """Tier 2: `config`, `store`, `sync` or `filelock` here would bind it to one workspace."""
    probe = (
        "import sys; import mitos.rotation; "
        "print(','.join(sorted(m for m in sys.modules "
        "if m == 'mitos' or m.startswith('mitos.')))); "
        "print('filelock' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["mitos,mitos.atomic_file,mitos.rotation", "False"]
