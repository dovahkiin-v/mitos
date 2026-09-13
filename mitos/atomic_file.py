"""Writes to mitos's files — the one home for tmp+replace, the durable append, and their mode rules.

Three entry points, selected by name rather than by a flag. The two whole-file
replaces differ by the one property that must differ by target, the ``fsync``; the
third is an append, which is durable but not atomic:

* ``write_source`` — for a markdown file ``mitos rebuild`` replays from (the
  ``decisions.md`` buffer, ``questions.md``, ``decisions/archive/*.md``). Temp file →
  write → ``fsync`` → ``os.replace`` → best-effort directory ``fsync``. After a
  process crash **or a power loss** the target holds its old content or its new
  content, whole. Where the filesystem accepts a directory ``fsync``, a returned
  call's new content also persists; where it refuses one, a power loss can revert a
  returned write to its previous whole content.
* ``write_derived`` — for regenerable output only (renders). Same replace, no
  ``fsync``. After a process crash the target holds old or new content, whole. After
  a power loss it **can be lost** (zero-length or garbage), not reverted — acceptable
  only because a render is rebuilt by the next regenerate. Never use it for a
  replayed-from file.
* ``append_source`` — for appending to a replayed-from file (rotation's archive).
  Missing parent directories are created; the file is opened for append, written,
  ``fsync``'d, then its directory (and each directory the call created, on its
  parent) gets a best-effort ``fsync``. **Durable, not atomic:** a crash mid-write can
  leave a partial trailing block, and a raised exception means bytes may already be
  appended. It never re-modes an existing file; a new one is created at ``0o666``
  filtered by the umask. Not covered by the shared contract below.

The contract the two replaces share:

* A symlinked target is resolved first, so the link survives and its referent is
  replaced. A caller's lock keyed on the configured path still serializes, because
  every mitos process builds the same lock path for a workspace and the lock is a
  sibling file the replace never touches.
* Mode: an existing target's permission bits are copied onto the temp file before
  any content is written; a new target is created with ``0o666`` filtered by the
  process umask, which is what ``open(path, "w")`` produces. The umask is never read.
* Bytes on disk are what ``open(path, "w", encoding="utf-8")`` writes for the string.
* **A raised ``Exception`` means the target is untouched**: the temp file is removed
  and the exception propagates unwrapped (``OSError`` stays ``OSError``,
  ``UnicodeEncodeError`` stays itself). The only step after the replace is the
  directory ``fsync``, whose ``OSError`` is swallowed.
* No lock, no rollback, no output. Callers own their lock → read → compute → write
  sequence.

Residuals of replacing rather than truncating, stated rather than fixed: a
hard-linked target stops sharing its inode; the replaced file is owned by the writing
user; extended attributes and ACLs on the old inode do not carry over.

Tier 1: stdlib only, imports nothing from ``mitos``.
"""

import os
import secrets
import stat
from typing import Optional

# Kept as the suffix so readers that take candidates by shape skip an in-flight temp
# file: the archive filename regex and the render tree's ``<scope>.md`` shape both do.
_TEMP_SUFFIX = ".tmp"
_NAME_ATTEMPTS = 16


def write_source(path: str, content: str) -> None:
    """Durably replaces a file ``mitos rebuild`` replays from.

    Args:
        path: The target file. A symlink is followed; its referent is replaced.
        content: The full new text of the file.

    Raises:
        OSError: If the replace did not happen (including a failed file ``fsync``).
            The target is untouched.
        UnicodeEncodeError: If ``content`` cannot be encoded as UTF-8. The target is
            untouched.
    """
    _replace(path, content, durable=True)


def write_derived(path: str, content: str) -> None:
    """Atomically replaces a regenerable file, without ``fsync``.

    For renders only. A power loss can leave the target empty or garbage, so a
    replayed-from file must go through ``write_source`` instead.

    Args:
        path: The target file. A symlink is followed; its referent is replaced.
        content: The full new text of the file.

    Raises:
        OSError: If the replace did not happen. The target is untouched.
        UnicodeEncodeError: If ``content`` cannot be encoded as UTF-8. The target is
            untouched.
    """
    _replace(path, content, durable=False)


def append_source(path: str, text: str) -> None:
    """Durably appends to a file ``mitos rebuild`` replays from.

    Durable, not atomic: after a returned call the bytes survive a power loss (where
    the filesystem accepts a directory ``fsync``, so does a newly created file's
    entry), but a crash mid-write can leave a partial trailing block. Whether a reader
    tolerates that tail or the archive becomes a whole-file replace is Phase 1d's
    decision (CC-13).

    Args:
        path: The target file. Missing parent directories are created.
        text: The text to append, written as ``open(path, "a", encoding="utf-8")``
            writes it.

    Raises:
        OSError: If a directory cannot be made, or the open, write or file ``fsync``
            fails — including ``NotADirectoryError``/``FileExistsError`` when a
            non-directory blocks the parent path. Bytes may be partially appended.
        UnicodeEncodeError: If ``text`` cannot be encoded as UTF-8. Bytes may be
            partially appended.
    """
    target = os.path.realpath(path)
    directory = os.path.dirname(target)

    missing: list = []
    probe = directory
    while probe and not os.path.exists(probe):
        missing.append(probe)
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    os.makedirs(directory, exist_ok=True)

    with open(target, "a", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())

    _fsync_directory_best_effort(directory)
    # Outermost first: each created directory's entry lives in its parent.
    for created in reversed(missing):
        _fsync_directory_best_effort(os.path.dirname(created))


def _replace(path: str, content: str, *, durable: bool) -> None:
    """Writes ``content`` to a sibling temp file and replaces the resolved target."""
    target = os.path.realpath(path)
    directory = os.path.dirname(target)
    try:
        existing_mode: Optional[int] = stat.S_IMODE(os.stat(target).st_mode)
    except FileNotFoundError:
        existing_mode = None

    # Preserve: open private, then widen/narrow to the target's bits before any byte
    # lands. Create: let O_CREAT apply the umask to 0o666, as open(path, "w") does.
    fd, temp_path = _create_temp(target, 0o600 if existing_mode is not None else 0o666)
    try:
        try:
            if existing_mode is not None and hasattr(os, "fchmod"):
                os.fchmod(fd, existing_mode)
            fh = os.fdopen(fd, "w", encoding="utf-8")
        except BaseException:
            os.close(fd)
            raise
        with fh:
            fh.write(content)
            if durable:
                fh.flush()
                os.fsync(fh.fileno())
        os.replace(temp_path, target)
    except BaseException:
        _remove_quietly(temp_path)
        raise

    if durable:
        _fsync_directory_best_effort(directory)


def _create_temp(target: str, mode: int) -> "tuple[int, str]":
    """Creates an exclusive, randomly named ``.tmp`` sibling of ``target``."""
    directory, name = os.path.split(target)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    for attempt in range(_NAME_ATTEMPTS):
        temp_path = os.path.join(
            directory, f".{name}.{secrets.token_hex(6)}{_TEMP_SUFFIX}"
        )
        try:
            return os.open(temp_path, flags, mode), temp_path
        except FileExistsError:
            if attempt == _NAME_ATTEMPTS - 1:
                raise
    raise AssertionError("unreachable")  # pragma: no cover


def _fsync_directory_best_effort(directory: str) -> None:
    """Persists the rename where the filesystem allows it, and never raises.

    Some mounts refuse a directory ``fsync`` (``EINVAL``/``ENOTSUP``) and some
    directories are writable but not readable (``EACCES`` on the open). Both are valid
    setups. The file was already ``fsync``'d, so after a power loss the target holds
    old or new content, whole, whatever this step does; swallowing costs only the
    rename's persistence. The whole ``OSError`` class is swallowed rather than an
    errno list, which would go short on the next unusual mount.
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        dir_fd = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        try:
            os.close(dir_fd)
        except OSError:
            pass


def _remove_quietly(path: str) -> None:
    """Deletes a temp file, ignoring its absence — used on the failure path only."""
    try:
        os.remove(path)
    except OSError:
        pass
