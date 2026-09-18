"""Tests for `mitos capture`'s buffer write — the first rows that enter its lock.

Capture locks, reads the whole buffer, splices the synthesized entry below the
`BEGIN ENTRIES` marker and replaces the file. The synthesis call sits before the
lock and is patched out here, so each row drives the shipped `cmd_capture` from
the lock onwards. Each row names the wrong implementation it catches.
"""

import os
import stat
from unittest.mock import patch

import pytest

from mitos import atomic_file, cli
from mitos.config import MitosConfig
from mitos.sync import MitosSyncManager, _ENTRIES_MARKER

ENTRY = (
    "### captured-entry\n\n"
    "**Axiom:** A captured thought lands whole or not at all.\n\n"
    "**Rejected paths:** truncating the buffer in place.\n\n"
    "**Scope:** substrate\n"
)


@pytest.fixture
def config(tmp_path, monkeypatch):
    """A real `cmd_init` workspace with a dummy key, offline from Qdrant."""
    # qdrant_url is resolved at construction, so the unreachable URL goes in first.
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    cfg = MitosConfig(str(tmp_path))
    cli.cmd_init(cfg)
    cfg.env["GEMINI_API_KEY"] = "test-key"
    return cfg


def _capture(config: MitosConfig) -> None:
    with patch("mitos.cli.run_ambient_capture", return_value=ENTRY), \
            patch("google.genai.Client"):
        cli.cmd_capture(config, "a raw thought")


def _read(config: MitosConfig) -> str:
    with open(config.decisions_file, "r", encoding="utf-8") as f:
        return f.read()


def _write(config: MitosConfig, text: str) -> None:
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write(text)


def _spliced(before: str) -> str:
    """What the handler produces: the entry directly below the first marker."""
    return before.replace(_ENTRIES_MARKER, f"{_ENTRIES_MARKER}\n\n{ENTRY}\n", 1)


def _umask_default() -> int:
    current = os.umask(0)
    os.umask(current)
    return 0o666 & ~current


def _temps(directory: str) -> list:
    return [n for n in os.listdir(directory) if n.endswith(".tmp")]


# --- C1 (RF-1): a marker that occurs twice ------------------------------------------

def test_capture_splices_once_below_the_first_marker_when_the_marker_occurs_twice(config):
    """An uncounted `replace` inserts the entry below every copy of the marker.

    A hand edit can put the marker text inside an entry's context; capture must still
    land exactly one entry, directly below the real (first) marker.
    """
    planted = _read(config) + (
        "\n### hand-edited-entry\n\n"
        "**Axiom:** An entry whose context quotes the marker.\n\n"
        f"**Context:** Someone pasted the sentinel here: {_ENTRIES_MARKER}\n"
    )
    _write(config, planted)
    assert planted.count(_ENTRIES_MARKER) == 2

    _capture(config)

    after = _read(config)
    assert after.count(ENTRY) == 1
    assert after.index(f"{_ENTRIES_MARKER}\n\n{ENTRY}") == after.index(_ENTRIES_MARKER)
    assert after == _spliced(planted)


# --- C2: routed through the leaf, inside the lock -----------------------------------

def test_capture_writes_through_write_source_while_holding_the_buffer_lock(config, capsys):
    """A hand-rolled `open(..., "w")`, or a write moved out of the `with`, reds this row."""
    before = _read(config)
    recorded = []
    calls = []
    real_write_source = atomic_file.write_source

    class _Recording(MitosSyncManager):
        def __init__(self, cfg):
            super().__init__(cfg)
            recorded.append(self)

    def _spy(path, content):
        calls.append((os.path.realpath(path), recorded[0].lock.is_locked))
        return real_write_source(path, content)

    with patch("mitos.cli.MitosSyncManager", _Recording), \
            patch("mitos.atomic_file.write_source", side_effect=_spy):
        _capture(config)

    assert calls == [(os.path.realpath(config.decisions_file), True)]
    assert _read(config) == _spliced(before)
    assert "Appended synthesized decision to decisions.md buffer" in capsys.readouterr().out


# --- C3: an existing buffer keeps its mode ------------------------------------------

def test_capture_preserves_the_buffer_mode(config):
    """A temp-file replace that skips the preserve rule re-modes the gold source."""
    os.chmod(config.decisions_file, 0o640)
    assert 0o640 != _umask_default(), "fixture would pass vacuously"
    before = _read(config)

    _capture(config)

    assert _read(config) == _spliced(before)
    assert stat.S_IMODE(os.stat(config.decisions_file).st_mode) == 0o640


# --- C4: a failed write leaves the buffer whole -------------------------------------

def test_a_failed_capture_write_leaves_the_buffer_byte_identical(config, capsys):
    """A truncate-in-place write leaves a half-written buffer on a full disk.

    The injection is filtered to the buffer's realpath: manager construction runs inside
    the patched call, and a blanket `os.replace` failure could fire there first.
    """
    before = _read(config)
    real_replace = os.replace
    fired = []

    def _fail_buffer_replace(src, dst, *args, **kwargs):
        if os.path.realpath(dst) == os.path.realpath(config.decisions_file):
            fired.append(dst)
            raise OSError(28, "No space left on device")
        return real_replace(src, dst, *args, **kwargs)

    with patch("mitos.atomic_file.os.replace", side_effect=_fail_buffer_replace):
        _capture(config)  # returns normally: capture reports on stdout

    assert len(fired) == 1
    assert _read(config) == before
    assert _temps(config.workspace_dir) == []
    assert "Failed to append captured entry" in capsys.readouterr().out


# --- C5: no marker ------------------------------------------------------------------

def test_capture_appends_at_the_end_of_a_buffer_without_the_marker(config):
    """The fallback keeps today's bytes: no `rstrip`, entry appended after a blank line."""
    before = "# Decisions\n\nA buffer whose marker was deleted by hand.\n"
    _write(config, before)
    targets = []
    real_write_source = atomic_file.write_source

    def _spy(path, content):
        targets.append(os.path.realpath(path))
        return real_write_source(path, content)

    with patch("mitos.atomic_file.write_source", side_effect=_spy):
        _capture(config)

    assert targets == [os.path.realpath(config.decisions_file)]
    assert _read(config) == before + f"\n\n{ENTRY}\n"
