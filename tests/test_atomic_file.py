"""Tests for `mitos.atomic_file` — the durable write under the gold source.

Each row names the plausible wrong implementation it catches. The primitive's whole
value is in properties a healthy run never shows (a crash, a full disk, a mount that
refuses a directory fsync), so the failures are provoked rather than hoped about.
"""

import errno
import os
import stat
import subprocess
import sys
from unittest.mock import patch

import pytest

from mitos import atomic_file
from mitos.atomic_file import write_derived, write_source

ENTRY_POINTS = [write_source, write_derived]


def _umask_default() -> int:
    """The mode `open(path, "w")` creates at — read once, in a single-threaded test."""
    current = os.umask(0)
    os.umask(current)
    return 0o666 & ~current


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def _temps(directory) -> list:
    return [n for n in os.listdir(directory) if n.endswith(".tmp")]


def _is_dir_fd(fd: int) -> bool:
    return stat.S_ISDIR(os.fstat(fd).st_mode)


# --- R1: bytes -------------------------------------------------------------------

@pytest.mark.parametrize("write", ENTRY_POINTS)
def test_bytes_match_a_plain_text_mode_write(tmp_path, write):
    """A binary-mode or `newline=""` build drifts from today's bytes on some platform."""
    text = "# Sprendimai — ąčęėįšųūž ✓\n\nline two\r\nend\n"
    reference = tmp_path / "reference.md"
    with open(reference, "w", encoding="utf-8") as fh:
        fh.write(text)
    target = tmp_path / "target.md"
    target.write_text("old\n", encoding="utf-8")

    write(str(target), text)

    assert target.read_bytes() == reference.read_bytes()


# --- R2 / R3: the two mode rules ----------------------------------------------------

@pytest.mark.parametrize("write", ENTRY_POINTS)
def test_an_existing_target_keeps_its_mode(tmp_path, write):
    """`mkstemp` + replace re-modes the gold source to 0600 (R11)."""
    fixture_mode = 0o640
    assert fixture_mode != _umask_default(), "fixture would pass vacuously"
    target = tmp_path / "decisions.md"
    target.write_text("old\n", encoding="utf-8")
    os.chmod(target, fixture_mode)

    write(str(target), "new\n")

    assert _mode(target) == fixture_mode
    assert target.read_text(encoding="utf-8") == "new\n"


@pytest.mark.parametrize("write", ENTRY_POINTS)
def test_a_new_target_is_created_at_the_umask_default(tmp_path, write):
    """At a create site there is no mode to preserve, so `mkstemp` would mint 0600."""
    sibling = tmp_path / "plain.md"
    with open(sibling, "w", encoding="utf-8") as fh:
        fh.write("x")
    target = tmp_path / "new.md"

    write(str(target), "fresh\n")

    assert _mode(target) == _mode(sibling) == _umask_default()
    assert target.read_text(encoding="utf-8") == "fresh\n"


# --- R4 / R5: failure before the replace -------------------------------------------

@pytest.mark.parametrize("write", ENTRY_POINTS)
def test_a_failed_replace_leaves_the_target_whole_and_no_temp(tmp_path, write):
    """A truncate-in-place build would have emptied the file before the fault."""
    target = tmp_path / "decisions.md"
    target.write_text("the gold source\n", encoding="utf-8")
    before = target.read_bytes()

    with patch("mitos.atomic_file.os.replace", side_effect=OSError(errno.ENOSPC, "full")):
        with pytest.raises(OSError) as info:
            write(str(target), "never lands\n")

    assert type(info.value) is OSError, "the primitive must not wrap"
    assert target.read_bytes() == before
    assert _temps(tmp_path) == []


@pytest.mark.parametrize("write", ENTRY_POINTS)
def test_an_unencodable_string_leaves_the_target_whole_and_no_temp(tmp_path, write):
    """A cleanup scoped to `OSError` would leak the temp file on this path."""
    target = tmp_path / "decisions.md"
    target.write_text("the gold source\n", encoding="utf-8")
    before = target.read_bytes()

    with pytest.raises(UnicodeEncodeError):
        write(str(target), "lone surrogate \udcff here\n")

    assert target.read_bytes() == before
    assert _temps(tmp_path) == []


def test_a_keyboard_interrupt_mid_write_cleans_up_and_re_raises(tmp_path):
    """`BaseException` must clean up too — an `except Exception` build leaks the temp."""
    target = tmp_path / "decisions.md"
    target.write_text("old\n", encoding="utf-8")

    with patch("mitos.atomic_file.os.fsync", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            write_source(str(target), "new\n")

    assert target.read_text(encoding="utf-8") == "old\n"
    assert _temps(tmp_path) == []


# --- R6: fsync is selected by entry point -----------------------------------------

def test_write_source_fsyncs_the_file_and_opens_the_directory(tmp_path):
    """Dropping the file fsync turns a power loss into LOST, not REVERTED."""
    target = tmp_path / "decisions.md"
    target.write_text("old\n", encoding="utf-8")
    file_syncs = []
    real_fsync, real_open = os.fsync, os.open

    def _fsync(fd):
        if not _is_dir_fd(fd):
            file_syncs.append(fd)
        return real_fsync(fd)

    with patch("mitos.atomic_file.os.fsync", side_effect=_fsync), \
            patch("mitos.atomic_file.os.open", side_effect=real_open) as opened:
        write_source(str(target), "new\n")

    assert len(file_syncs) >= 1
    assert any(c.args[0] == str(tmp_path) for c in opened.call_args_list), \
        "the directory step was never attempted"


def test_write_derived_never_fsyncs(tmp_path):
    """A render reaching the durable path costs a device barrier per file on every write."""
    target = tmp_path / "live_axioms.md"
    real_open = os.open

    with patch("mitos.atomic_file.os.fsync") as fsync, \
            patch("mitos.atomic_file.os.open", side_effect=real_open) as opened:
        write_derived(str(target), "render\n")

    assert fsync.call_count == 0
    assert not any(c.args[0] == str(tmp_path) for c in opened.call_args_list)
    assert target.read_text(encoding="utf-8") == "render\n"


# --- R9: the directory step is best-effort; the file fsync is not ------------------

def test_a_refused_directory_open_is_swallowed(tmp_path):
    """A write-but-not-read directory is a valid setup, not a failed write."""
    target = tmp_path / "decisions.md"
    target.write_text("old\n", encoding="utf-8")
    real_open = os.open

    def _open(path, flags, *args):
        if path == str(tmp_path):
            raise OSError(errno.EACCES, "denied")
        return real_open(path, flags, *args)

    with patch("mitos.atomic_file.os.open", side_effect=_open):
        write_source(str(target), "new\n")

    assert target.read_text(encoding="utf-8") == "new\n"
    assert _temps(tmp_path) == []


def test_a_refused_directory_fsync_is_swallowed(tmp_path):
    """A raising directory step would make every record report a false failed rollback."""
    target = tmp_path / "decisions.md"
    target.write_text("old\n", encoding="utf-8")
    real_fsync = os.fsync
    refused = []

    def _fsync(fd):
        if _is_dir_fd(fd):
            refused.append(fd)
            raise OSError(errno.EINVAL, "not supported on this mount")
        return real_fsync(fd)

    with patch("mitos.atomic_file.os.fsync", side_effect=_fsync):
        write_source(str(target), "new\n")

    assert refused, "the directory fsync was never reached"
    assert target.read_text(encoding="utf-8") == "new\n"
    assert _temps(tmp_path) == []


def test_a_failed_file_fsync_propagates_and_leaves_the_target_untouched(tmp_path):
    """Swallowing EIO on the data blocks would claim durability the disk refused."""
    target = tmp_path / "decisions.md"
    target.write_text("old\n", encoding="utf-8")

    def _fsync(fd):
        raise OSError(errno.EIO, "I/O error")

    with patch("mitos.atomic_file.os.fsync", side_effect=_fsync):
        with pytest.raises(OSError) as info:
            write_source(str(target), "new\n")

    assert info.value.errno == errno.EIO
    assert target.read_text(encoding="utf-8") == "old\n"
    assert _temps(tmp_path) == []


# --- R7: symlinks -------------------------------------------------------------------

@pytest.mark.parametrize("write", ENTRY_POINTS)
def test_a_symlinked_target_stays_a_symlink(tmp_path, write):
    """Replacing the link itself cuts a dotfile-managed buffer loose from its home."""
    real_dir = tmp_path / "dotfiles"
    link_dir = tmp_path / "workspace"
    real_dir.mkdir()
    link_dir.mkdir()
    referent = real_dir / "decisions.md"
    referent.write_text("old\n", encoding="utf-8")
    os.chmod(referent, 0o640)
    link = link_dir / "decisions.md"
    os.symlink(referent, link)

    write(str(link), "new\n")

    assert os.path.islink(link)
    assert referent.read_text(encoding="utf-8") == "new\n"
    assert _mode(referent) == 0o640
    assert _temps(real_dir) == [] and _temps(link_dir) == []


# --- temp-name shape ----------------------------------------------------------------

def test_the_temp_file_is_a_dot_tmp_sibling(tmp_path):
    """Readers that take candidates by shape (`<scope>.md`, `YYYY-Qn.md`) must skip it."""
    target = tmp_path / "decisions.md"
    target.write_text("old\n", encoding="utf-8")
    seen = []
    real_replace = os.replace

    def _replace(src, dst):
        seen.append(src)
        return real_replace(src, dst)

    with patch("mitos.atomic_file.os.replace", side_effect=_replace):
        write_source(str(target), "new\n")

    assert len(seen) == 1
    assert os.path.dirname(seen[0]) == str(tmp_path)
    assert seen[0].endswith(".tmp") and not seen[0].endswith(".md")


# --- R8: tier ------------------------------------------------------------------------

def test_importing_the_leaf_pulls_in_no_other_mitos_module():
    """`cli.py` imports this from 1b on; a convenient import here would cycle or weigh."""
    probe = (
        "import sys; import mitos.atomic_file; "
        "print(','.join(sorted(m for m in sys.modules "
        "if m == 'mitos' or m.startswith('mitos.'))))"
    )
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "mitos,mitos.atomic_file"


# --- ensure_parent_directory (1d): the durable mkdir before a first archive write ----
#
# The bytes-and-mode claim moved to `test_rotation.py`'s archive-mode row: the archive
# file itself is now written by `write_source`, whose shared mode rules apply.

def test_ensure_parent_directory_creates_the_parents_at_the_umask_default(tmp_path):
    target = tmp_path / "decisions" / "archive" / "2026-Q3.md"
    previous = os.umask(0o027)
    try:
        atomic_file.ensure_parent_directory(str(target))
    finally:
        os.umask(previous)

    assert target.parent.is_dir() and not target.exists()
    assert _mode(target.parent) == _mode(target.parent.parent) == 0o750


def test_ensure_parent_directory_fsyncs_each_created_directorys_entry(tmp_path):
    """Without these, a power loss can drop a new archive/ after the buffer replace lands."""
    target = tmp_path / "decisions" / "archive" / "2026-Q3.md"
    file_syncs, dir_syncs = [], []
    real_fsync = os.fsync

    def _fsync(fd):
        (dir_syncs if _is_dir_fd(fd) else file_syncs).append(os.fstat(fd).st_ino)
        return real_fsync(fd)

    with patch("mitos.atomic_file.os.fsync", side_effect=_fsync):
        atomic_file.ensure_parent_directory(str(target))

    assert file_syncs == []
    # `archive/`'s entry lives in `decisions/`, and `decisions/`'s in tmp_path.
    for directory in (target.parent.parent, tmp_path):
        assert os.stat(directory).st_ino in dir_syncs, f"{directory} entry was never synced"

    dir_syncs.clear()
    with patch("mitos.atomic_file.os.fsync", side_effect=_fsync):
        atomic_file.ensure_parent_directory(str(target))
    assert dir_syncs == [], "nothing was created, so there is no entry to persist"


def test_ensure_parent_directory_swallows_a_refused_directory_fsync(tmp_path):
    target = tmp_path / "archive" / "2026-Q3.md"
    real_fsync = os.fsync
    refused = []

    def _fsync(fd):
        if _is_dir_fd(fd):
            refused.append(fd)
            raise OSError(errno.EINVAL, "not supported on this mount")
        return real_fsync(fd)

    with patch("mitos.atomic_file.os.fsync", side_effect=_fsync):
        atomic_file.ensure_parent_directory(str(target))

    assert refused, "the directory fsync was never reached"
    assert target.parent.is_dir()


def test_ensure_parent_directory_propagates_a_blocked_parent_and_writes_nothing(tmp_path):
    """A swallowed makedirs failure would let rotation replace the buffer with no archive."""
    (tmp_path / "decisions").write_text("not a directory\n", encoding="utf-8")
    before = sorted(os.listdir(tmp_path))

    with pytest.raises((NotADirectoryError, FileExistsError)):
        atomic_file.ensure_parent_directory(
            str(tmp_path / "decisions" / "archive" / "2026-Q3.md"))

    assert sorted(os.listdir(tmp_path)) == before
