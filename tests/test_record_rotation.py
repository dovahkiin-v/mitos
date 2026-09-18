"""T11 (3c half): rotation reaches `record_decision` — the bound, the receipt field, stdout.

After a `created` record, one bounded rotation runs over the post-write buffer, outside
the write's own lock hold, pre-gated on the bytes the write just produced. Its outcome
rides the receipt as one `rotation` key; a failure is a successful receipt with a
structured cause, and each renderer composes its own recovery.

Seeding discipline (the plan's clock trap): seeds are recorded at the default threshold,
so the gate stays closed while seeding; their nodes are then back-dated in SQL, and only
then is the threshold lowered — so the write under test is the one fresh entry.
"""

import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
from filelock import Timeout

from mitos import atomic_file, settledness
from mitos.cli import cmd_init, cmd_record
from mitos.config import MitosConfig
from mitos.errors import DatabaseError
from mitos.parser import parse_entry_stream
from mitos.store import GraphStore
from mitos.sync import (
    ROTATION_FAILED,
    ROTATION_ROTATED,
    ROTATION_STAGE_FILE,
    ROTATION_STAGE_GRAPH,
    ROTATION_STAGE_LOCK,
    MitosSyncManager,
    _ROTATION_FAILED_NOTE,
)
from test_mcp_selector import FORBIDDEN_SYNTAX

DEAD_QDRANT_URL = "http://127.0.0.1:9"
FRESH = "fresh-write"
FAILURE_PREFIX = "[Warning] Archive rotation failed"


# --------------------------------------------------------------------------- #
# Scaffolding
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Keyless and serviceless: the embed step defers, nothing reaches a network."""
    monkeypatch.setenv("QDRANT_URL", DEAD_QDRANT_URL)
    down = MagicMock(side_effect=Exception("backend down"))
    monkeypatch.setattr("mitos.sync.GeminiEmbeddingProvider", down)
    monkeypatch.setattr("mitos.sync.QdrantVectorStore", down)


@pytest.fixture
def make_ws(tmp_path):
    def _make(name: str = "ws") -> MitosConfig:
        root = tmp_path / name
        root.mkdir()
        config = MitosConfig(str(root))
        cmd_init(config)
        return config
    return _make


@pytest.fixture
def ws(make_ws) -> MitosConfig:
    return make_ws()


def _record(manager: MitosSyncManager, slug: str) -> Dict:
    return manager.record_decision_entry(
        f"The {slug} axiom.", f"The {slug} rejected reasoning.", ["alpha"],
        mechanisms=[f"{slug}-mechanism"], slug=slug, acknowledge_neighbors=True,
    )


def _seed(config: MitosConfig, count: int) -> List[str]:
    """Records `count` entries with the gate closed (default threshold); oldest first."""
    manager = MitosSyncManager(config)
    slugs = []
    for index in range(count):
        slug = f"seed-{index}"
        result = _record(manager, slug)
        assert result["status"] == "created" and "rotation" not in result, result
        slugs.append(slug)
    return slugs


def _back_date(config: MitosConfig, days: int = 30) -> None:
    stamp = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with GraphStore(config.db_path)._get_connection() as conn:
        conn.execute("UPDATE nodes SET updated_at = ?", (stamp,))


def _arm(config: MitosConfig, threshold: int, lag: int = 1) -> None:
    config.rotation_volume_threshold_entries = threshold
    config.rotation_lag_days = lag


def _settled_workspace(config: MitosConfig, seeds: int, threshold: int) -> List[str]:
    slugs = _seed(config, seeds)
    _back_date(config)
    _arm(config, threshold)
    return slugs


def _buffer_slugs(config: MitosConfig) -> List[str]:
    with open(config.decisions_file, encoding="utf-8") as fh:
        return [entry.slug for entry in parse_entry_stream(fh.read(), "decision")]


def _archive_slugs(path: str) -> List[str]:
    with open(path, encoding="utf-8") as fh:
        return re.findall(r"^### (\S+)", fh.read(), flags=re.M)


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class CountingLock:
    """Forwards to the manager's real lock, counting entries; can refuse the Nth."""

    def __init__(self, real, *, fail_on: int = 0) -> None:
        self.real = real
        self.enters = 0
        self.fail_on = fail_on

    def __enter__(self):
        self.enters += 1
        if self.enters == self.fail_on:
            raise Timeout(self.real.lock_file)
        return self.real.__enter__()

    def __exit__(self, *exc):
        return self.real.__exit__(*exc)


def _spy_select(monkeypatch, *, manager=None, raises=None) -> List[Dict]:
    """Wraps the selector: records the text it read and the lock depth at that moment."""
    calls: List[Dict] = []
    real = settledness.select_settled_tail

    def spy(buffer_text, **kwargs):
        calls.append({
            "text": buffer_text,
            "lock_counter": manager.lock.lock_counter if manager is not None else None,
        })
        if raises is not None:
            raise raises
        return real(buffer_text, **kwargs)

    monkeypatch.setattr(settledness, "select_settled_tail", spy)
    return calls


def _spy_rotate(monkeypatch, manager) -> List[int]:
    calls: List[int] = []
    real = manager._rotate_settled

    def spy(*, window):
        calls.append(window)
        return real(window=window)

    monkeypatch.setattr(manager, "_rotate_settled", spy)
    return calls


def _no_rotation_text(stream: str) -> bool:
    return not any("rotat" in line.lower() for line in stream.splitlines())


# --------------------------------------------------------------------------- #
# R1–R6: reach, bound, read, placement, gate, exits
# --------------------------------------------------------------------------- #


def test_a_threshold_crossing_record_moves_the_settled_tail_and_says_where(ws) -> None:
    """R1."""
    seeds = _settled_workspace(ws, 3, threshold=3)

    result = _record(MitosSyncManager(ws), FRESH)

    assert result["status"] == "created"
    report = result["rotation"]
    assert report["outcome"] == ROTATION_ROTATED
    assert "skipped" not in report
    [archive] = report["archives"]
    assert os.path.isabs(archive["path"])
    assert os.path.dirname(archive["path"]) == ws.archive_dir
    assert archive["entries"] == 3 and archive["slugs"] == seeds
    assert _buffer_slugs(ws) == [FRESH]
    assert _archive_slugs(archive["path"]) == list(reversed(seeds))


def test_one_record_moves_one_window_and_the_next_record_moves_the_next(
        ws, monkeypatch) -> None:
    """R2: one bounded call per record; the residual drains across later writes."""
    monkeypatch.setattr(settledness, "ROTATION_WINDOW_ENTRIES", 3)
    seeds = _settled_workspace(ws, 3 + 5, threshold=1)
    manager = MitosSyncManager(ws)
    calls = _spy_rotate(monkeypatch, manager)

    first = _record(manager, FRESH)

    assert calls == [3]
    assert first["rotation"]["archives"][0]["slugs"] == seeds[:3]
    assert _buffer_slugs(ws) == [FRESH] + list(reversed(seeds[3:]))

    second = _record(manager, "fresh-two")

    assert calls == [3, 3]
    assert second["rotation"]["archives"][0]["slugs"] == seeds[3:6]
    assert _buffer_slugs(ws) == ["fresh-two", FRESH] + list(reversed(seeds[6:]))


def test_the_one_read_is_the_post_write_buffer_outside_the_write_hold(
        ws, monkeypatch) -> None:
    """R3 + R3b: the selector reads a buffer holding the new entry, at lock depth 1."""
    _settled_workspace(ws, 3, threshold=3)
    manager = MitosSyncManager(ws)
    calls = _spy_select(monkeypatch, manager=manager)

    result = _record(manager, FRESH)

    [call] = calls
    assert FRESH in [e.slug for e in parse_entry_stream(call["text"], "decision")]
    # filelock counts re-entry: a rotation woven into the write's `with` reads 2.
    assert call["lock_counter"] == 1
    assert result["rotation"]["outcome"] == ROTATION_ROTATED
    assert FRESH in _buffer_slugs(ws)


def test_below_the_threshold_a_record_takes_no_second_acquisition(
        ws, monkeypatch) -> None:
    """R4, closed side: the gate counts the bytes just written, so 4 < 5 stays shut."""
    _settled_workspace(ws, 3, threshold=5)
    manager = MitosSyncManager(ws)
    lock = manager.lock = CountingLock(manager.lock)
    calls = _spy_select(monkeypatch)

    result = _record(manager, FRESH)

    assert result["status"] == "created"
    assert lock.enters == 1
    assert calls == []
    assert "rotation" not in result


def test_at_the_threshold_the_count_includes_the_entry_just_written(
        ws, monkeypatch) -> None:
    """R4, open side, on the boundary: three seeds plus the write make four ≥ 4."""
    _settled_workspace(ws, 3, threshold=4)
    manager = MitosSyncManager(ws)
    lock = manager.lock = CountingLock(manager.lock)

    result = _record(manager, FRESH)

    assert lock.enters == 2
    assert result["rotation"]["outcome"] == ROTATION_ROTATED


def test_a_blocked_tail_writes_nothing_and_reports_nothing(ws) -> None:
    """R5: the gate opens, the tail is recent, nothing moves, and the key is absent."""
    seeds = _seed(ws, 3)
    _arm(ws, 3)
    manager = MitosSyncManager(ws)
    lock = manager.lock = CountingLock(manager.lock)

    result = _record(manager, FRESH)

    assert lock.enters == 2, "non-vacuity: rotation was evaluated"
    assert "rotation" not in result
    assert not os.path.exists(ws.archive_dir)
    assert _buffer_slugs(ws) == [FRESH] + list(reversed(seeds))


def _exists_phase_a(config, manager, monkeypatch):
    result = _record(manager, "seed-0")
    assert result["status"] == "exists"


def _exists_toctou(config, manager, monkeypatch):
    from mitos.parser import ParsedEntry
    real_lock, other = manager.lock, MitosSyncManager(config)

    class RacingLock:
        def __enter__(self):
            racer = ParsedEntry("decision", "raced", 0, 0)
            racer.axiom = "The raced axiom."
            racer.rejected_paths = "setup rejection"
            other.store.commit_parsed_entry(racer)
            return real_lock.__enter__()

        def __exit__(self, *exc):
            return real_lock.__exit__(*exc)

    manager.lock = RacingLock()
    result = manager.record_decision_entry("The raced axiom.", "rej", [], slug="raced",
                                           acknowledge_neighbors=True)
    assert result["status"] == "exists"


def _needs_review(config, manager, monkeypatch):
    monkeypatch.setattr(MitosSyncManager, "_review_neighbors", lambda *a, **k: [
        {"slug": "seed-0", "score": 0.9, "axiom": "The seed-0 axiom."}])
    result = manager.record_decision_entry("The seed-0 axiom, restated.", "rej", ["alpha"],
                                           slug="paused")
    assert result["status"] == "needs_review"


def _slug_collision(config, manager, monkeypatch):
    result = manager.record_decision_entry("A different axiom.", "rej", [], slug="seed-0",
                                           acknowledge_neighbors=True)
    assert result["code"] == "slug_collision"


def _commit_failed(config, manager, monkeypatch):
    monkeypatch.setattr(manager.store, "commit_parsed_entry",
                        MagicMock(side_effect=DatabaseError("boom")))
    result = _record(manager, FRESH)
    assert result["code"] == "commit_failed"


def _forward_write_failed(config, manager, monkeypatch):
    def refuse(path, content):
        raise OSError(28, "No space left on device", path)

    monkeypatch.setattr(atomic_file, "write_source", refuse)
    result = _record(manager, FRESH)
    assert result["code"] == "commit_failed"


@pytest.mark.parametrize("exit_", [
    _exists_phase_a, _exists_toctou, _needs_review, _slug_collision, _commit_failed,
    _forward_write_failed,
], ids=lambda f: f.__name__.strip("_"))
def test_a_write_that_did_not_land_never_evaluates_rotation(ws, monkeypatch, exit_) -> None:
    """R6: every seed is settled and the threshold is 1, so any evaluation would move."""
    _settled_workspace(ws, 3, threshold=1)
    manager = MitosSyncManager(ws)
    calls = _spy_rotate(monkeypatch, manager)

    exit_(ws, manager, monkeypatch)

    assert calls == []
    assert not os.path.exists(ws.archive_dir)


# --------------------------------------------------------------------------- #
# R7: a failure is a successful receipt with a structured cause
# --------------------------------------------------------------------------- #


def _inject_graph(config, manager, monkeypatch) -> List:
    return _spy_select(monkeypatch, raises=sqlite3.OperationalError("database is locked"))


def _inject_file(config, manager, monkeypatch) -> List:
    fired, real = [], atomic_file.write_source

    def refuse_archive(path, content):
        if os.path.dirname(path) == config.archive_dir:
            fired.append(path)
            raise OSError(28, "No space left on device", path)
        return real(path, content)

    monkeypatch.setattr(atomic_file, "write_source", refuse_archive)
    return fired


def _inject_lock(config, manager, monkeypatch) -> List:
    lock = manager.lock = CountingLock(manager.lock, fail_on=2)

    class Fired(list):
        def __len__(self):
            return 1 if lock.enters >= 2 else 0

    return Fired()


@pytest.mark.parametrize("inject, stage, error_type", [
    (_inject_graph, ROTATION_STAGE_GRAPH, "OperationalError"),
    (_inject_file, ROTATION_STAGE_FILE, "OSError"),
    (_inject_lock, ROTATION_STAGE_LOCK, "Timeout"),
], ids=["graph", "file", "lock"])
def test_a_rotation_failure_leaves_a_created_receipt_and_stdout_clean(
        ws, monkeypatch, capsys, inject, stage, error_type) -> None:
    """R7."""
    seeds = _settled_workspace(ws, 3, threshold=3)
    manager = MitosSyncManager(ws)
    fired = inject(ws, manager, monkeypatch)
    capsys.readouterr()

    result = _record(manager, FRESH)

    out, err = capsys.readouterr()
    assert len(fired) >= 1, "the injection fired"
    assert result["status"] == "created"
    assert GraphStore(ws.db_path).get_node_by_slug(FRESH) is not None
    assert _buffer_slugs(ws) == [FRESH] + list(reversed(seeds))
    assert result["rotation"] == {
        "outcome": ROTATION_FAILED,
        "stage": stage,
        "error": result["rotation"]["error"],
        "note": _ROTATION_FAILED_NOTE,
    }
    assert result["rotation"]["error"].startswith(f"{error_type}: ")
    assert "mitos " not in result["rotation"]["note"]
    assert out == ""
    assert _no_rotation_text(err), err


def test_timeout_is_a_lock_fault_not_a_file_fault() -> None:
    """Mutant (g) in isolation: `Timeout` is an `OSError` and must be classified first."""
    from mitos.sync import _rotation_failure_stage
    assert isinstance(Timeout("x"), OSError), "non-vacuity: the MRO this row guards"
    assert _rotation_failure_stage(Timeout("x")) == ROTATION_STAGE_LOCK
    assert _rotation_failure_stage(UnicodeDecodeError("utf-8", b"", 0, 1, "x")) == "file"
    assert _rotation_failure_stage(DatabaseError("x")) == ROTATION_STAGE_GRAPH
    assert _rotation_failure_stage(ValueError("x")) == "unexpected"


def test_the_report_words_skips_per_block_and_crosses_json_as_lists() -> None:
    """The `skipped` shapes no fixture can reach: a mixed batch, a skip-only one, silence."""
    from mitos.cli import _rotation_receipt_lines
    from mitos.rotation import RotationBlock, RotationOutcome
    from mitos.sync import ROTATION_SKIPPED, _rotation_report

    moved = RotationBlock("older-a", "### older-a\n", "2026-Q3.md")
    mixed = _rotation_report(RotationOutcome(
        rotated=[moved], unmatched=["gone"], duplicated=[("dup-x", 2)],
        overlapping=["ovl-y"], archive_paths=["/ws/decisions/archive/2026-Q3.md"],
    ))
    skip_only = _rotation_report(RotationOutcome(
        rotated=[], unmatched=[], duplicated=[("dup-x", 2)], overlapping=[],
        archive_paths=[],
    ))

    assert json.loads(json.dumps(mixed)) == mixed == {
        "outcome": ROTATION_ROTATED,
        "archives": [{"path": "/ws/decisions/archive/2026-Q3.md", "entries": 1,
                      "slugs": ["older-a"]}],
        "skipped": [{"slug": "dup-x", "reason": "duplicated", "occurrences": 2},
                    {"slug": "ovl-y", "reason": "overlapping"}],
    }
    assert skip_only == {"outcome": ROTATION_SKIPPED, "archives": [],
                         "skipped": [{"slug": "dup-x", "reason": "duplicated",
                                      "occurrences": 2}]}
    assert _rotation_report(RotationOutcome([], ["gone"], [], [], [])) is None

    lines = _rotation_receipt_lines(mixed)
    assert lines[0].startswith("  Rotated:   1 older settled entry to ")
    kept = [line for line in lines if line.startswith("  Kept:")]
    assert len(kept) == 2 and all("'dup-x'" in kept[0] and "not moved" in line
                                  for line in kept)
    # A skipped block must not read as the whole rotation having done nothing.
    assert not any("removed nothing" in line for line in lines)


def test_the_lag_zero_residual_can_rotate_the_entry_just_written_and_says_so(
        ws) -> None:
    """R11: the configured semantics, visible on the receipt rather than silent."""
    _seed(ws, 2)
    _arm(ws, 1, lag=0)
    ahead = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()

    with patch("mitos.sync._utc_now_iso", return_value=ahead):
        result = _record(MitosSyncManager(ws), FRESH)

    assert FRESH in result["rotation"]["archives"][0]["slugs"]
    assert FRESH not in _buffer_slugs(ws)


# --------------------------------------------------------------------------- #
# R8–R10: the renderers
# --------------------------------------------------------------------------- #


def _cli_record(config: MitosConfig, **kwargs) -> None:
    cmd_record(config, axiom=f"The {FRESH} axiom.", rejected="rej", slug=FRESH,
               acknowledge_neighbors=True, **kwargs)


def test_cli_text_prints_a_success_receipt_line_on_stdout(ws, capsys) -> None:
    """R8, success."""
    _settled_workspace(ws, 3, threshold=3)
    capsys.readouterr()

    _cli_record(ws)

    out, err = capsys.readouterr()
    [line] = [l for l in out.splitlines() if l.startswith("  Rotated:")]
    assert "3 older settled entries" in line
    assert ws.archive_dir in line
    assert not any(l.startswith(FAILURE_PREFIX) for l in err.splitlines())


def test_cli_text_prints_one_failure_line_on_stderr_with_a_usable_recovery(
        ws, monkeypatch, capsys) -> None:
    """R8, failure: the recipe carries the caller's selector and never names sync."""
    _settled_workspace(ws, 3, threshold=3)
    _inject_graph(ws, None, monkeypatch)
    capsys.readouterr()

    _cli_record(ws)

    out, err = capsys.readouterr()
    [line] = [l for l in err.splitlines() if l.startswith(FAILURE_PREFIX)]
    assert "(graph)" in line and "OperationalError" in line
    assert f"-p {ws.project!r}" in line
    assert "mitos sync" not in line
    assert not any(l.startswith("  Rotated:") for l in out.splitlines())
    assert f"Recorded decision '{FRESH}'" in out


@pytest.mark.parametrize("fail", [False, True], ids=["rotated", "failed"])
def test_cli_json_emits_the_field_verbatim_with_nothing_on_stderr(
        ws, monkeypatch, capsys, fail) -> None:
    """R9."""
    _settled_workspace(ws, 3, threshold=3)
    if fail:
        _inject_graph(ws, None, monkeypatch)
    capsys.readouterr()

    _cli_record(ws, as_json=True)

    out, err = capsys.readouterr()
    report = json.loads(out)["rotation"]
    assert report["outcome"] == (ROTATION_FAILED if fail else ROTATION_ROTATED)
    assert "recovery" not in report
    assert _no_rotation_text(err), err


def _mcp_record(config: MitosConfig) -> Dict:
    from mitos.mcp_server import record_decision
    with patch("mitos.mcp_server.MitosConfig", return_value=config):
        return json.loads(record_decision(
            f"The {FRESH} axiom.", "rej", [], slug=FRESH, acknowledge_neighbors=True,
            project=config.workspace_dir))


def _mcp_register(text: str) -> None:
    flat = " ".join(text.split())
    assert "mitos " not in flat
    assert "-p" not in flat
    for syntax in FORBIDDEN_SYNTAX:
        assert syntax not in flat


def test_mcp_composes_its_own_recovery_on_failure_only(make_ws, monkeypatch, capsys) -> None:
    """R10: recovery on MCP failure alone; every other key equals the CLI's."""
    cli_ws, mcp_ws = make_ws("cli"), make_ws("mcp")
    for config in (cli_ws, mcp_ws):
        _settled_workspace(config, 3, threshold=3)
    _inject_graph(None, None, monkeypatch)
    capsys.readouterr()

    _cli_record(cli_ws, as_json=True)
    cli_report = json.loads(capsys.readouterr()[0])["rotation"]
    mcp_report = _mcp_record(mcp_ws)["rotation"]
    out, err = capsys.readouterr()

    assert "recovery" not in cli_report
    recovery = mcp_report.pop("recovery")
    _mcp_register(recovery)
    for value in mcp_report.values():
        _mcp_register(value)
    assert mcp_report == cli_report
    assert out == ""
    [line] = [l for l in err.splitlines() if l.startswith(FAILURE_PREFIX)]
    assert FRESH in line


def test_mcp_success_carries_no_recovery_and_matches_the_cli(make_ws, capsys) -> None:
    """R10, success parity (paths compared relative to each workspace)."""
    cli_ws, mcp_ws = make_ws("cli"), make_ws("mcp")
    for config in (cli_ws, mcp_ws):
        _settled_workspace(config, 3, threshold=3)
    capsys.readouterr()

    _cli_record(cli_ws, as_json=True)
    cli_report = json.loads(capsys.readouterr()[0])["rotation"]
    mcp_report = _mcp_record(mcp_ws)["rotation"]

    def relative(report, config):
        return json.loads(json.dumps(report).replace(config.workspace_dir, "<ws>"))

    assert "recovery" not in mcp_report
    assert relative(mcp_report, mcp_ws) == relative(cli_report, cli_ws)
    assert mcp_report["outcome"] == ROTATION_ROTATED


# --------------------------------------------------------------------------- #
# R8 ordering + R14: the real CLI frame
# --------------------------------------------------------------------------- #


def _cli_env(tmp_path) -> Dict[str, str]:
    return {
        **os.environ,
        "MITOS_NO_UPDATE_CHECK": "1",
        "XDG_CONFIG_HOME": str(tmp_path / "xdg_config"),
        "GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", "ANTHROPIC_API_KEY": "",
        "QDRANT_URL": DEAD_QDRANT_URL,
    }


def _mitos(ws, env, *argv, combined=False) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "mitos.cli", *argv], cwd=str(ws), env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT if combined else subprocess.PIPE,
        timeout=120,
    )


def _subprocess_workspace(tmp_path, name, env, *, broken_archive=False):
    """Gotcha 9's recipe: seed three, raise nothing yet, then arm and back-date."""
    ws = tmp_path / name
    ws.mkdir()
    assert _mitos(ws, env, "init").returncode == 0
    for index in range(3):
        done = _mitos(ws, env, "-p", str(ws), "record", f"The seed-{index} axiom.",
                      "--rejected", "rej", "--slug", f"seed-{index}",
                      "--acknowledge-neighbors")
        assert done.returncode == 0, done.stderr
    config_path = ws / ".mitos" / "config.toml"
    text = config_path.read_text(encoding="utf-8")
    # Replace, never append: tomllib refuses a duplicate key.
    text, n_threshold = re.subn(r"^rotation_volume_threshold_entries = .*$",
                                "rotation_volume_threshold_entries = 3", text, flags=re.M)
    text, n_lag = re.subn(r"^rotation_lag_days = .*$", "rotation_lag_days = 1", text,
                          flags=re.M)
    assert (n_threshold, n_lag) == (1, 1), text
    config_path.write_text(text, encoding="utf-8")
    stamp = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    conn = sqlite3.connect(ws / ".mitos" / "graph.sqlite")
    with conn:
        conn.execute("UPDATE nodes SET updated_at = ?", (stamp,))
    conn.close()
    if broken_archive:
        (ws / "decisions").mkdir()
        (ws / "decisions" / "archive").write_text("not a directory\n", encoding="utf-8")
    return ws


def test_the_cli_frame_prints_where_the_rotation_went(tmp_path) -> None:
    """R14, success."""
    env = _cli_env(tmp_path)
    ws = _subprocess_workspace(tmp_path, "ws", env)

    done = _mitos(ws, env, "-p", str(ws), "record", "A fresh write.", "--rejected", "rej",
                  "--slug", FRESH)

    assert done.returncode == 0, done.stderr
    [line] = [l for l in done.stdout.splitlines() if l.startswith("  Rotated:")]
    assert f"3 older settled entries to {ws / 'decisions' / 'archive'}" in line
    assert not any(l.startswith(FAILURE_PREFIX) for l in done.stderr.splitlines())


def test_the_failure_line_reaches_a_combined_pipe_after_the_receipt(tmp_path) -> None:
    """R8 ordering + R14 failure: exit 0, receipt, failure line, coherence line last."""
    env = _cli_env(tmp_path)
    ws = _subprocess_workspace(tmp_path, "ws", env, broken_archive=True)

    done = _mitos(ws, env, "-p", str(ws), "record", "A fresh write.", "--rejected", "rej",
                  "--slug", FRESH, combined=True)

    combined = done.stdout
    assert done.returncode == 0, combined
    assert f"{FAILURE_PREFIX} (file): NotADirectoryError" in combined, combined
    failure = combined.index(FAILURE_PREFIX)
    assert combined.index("Handle:") < failure < combined.index("Coherence audit"), combined
