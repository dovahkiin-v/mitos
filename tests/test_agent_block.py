"""Tests for the versioned agent-file block + drift detection.

The block a project pastes into its agent files (AGENTS.md/CLAUDE.md/…) is a copy,
so it carries a version marker: stale copies are detected (`mitos status`,
`mitos agent-block --check`) and refreshed (`mitos agent-block`). These pin the
marker round-trip, the per-file drift classification, the CLI verb, and the status
integration.
"""

import json
import os
import re
import shlex
import sys
from unittest.mock import patch

import pytest

from mitos import cli
from mitos.cli import main
from mitos.config import MitosConfig
from mitos._agent_block import (
    AGENT_GUIDE_VERSION,
    agent_block,
    agent_block_drift,
    marker_version,
    scan_agent_files,
)


def _init(path):
    cli.cmd_init(MitosConfig(str(path)))


def _qdrant(reachable, collection_exists, points=None):
    return lambda url, coll: {
        "reachable": reachable,
        "collection_exists": collection_exists,
        "points": points,
    }


# --------------------------------------------------------------------------- #
# The canonical block + marker round-trip
# --------------------------------------------------------------------------- #

def test_agent_block_carries_current_marker_and_pointers():
    """The emitted block stamps the current version and stays thin (pointers only)."""
    block = agent_block()
    assert f"mitos-agent-guide: v{AGENT_GUIDE_VERSION}" in block
    assert "<!-- /mitos-agent-guide -->" in block          # closing marker for region edits
    assert marker_version(block) == AGENT_GUIDE_VERSION      # round-trips through the parser
    # Durable pointers, not volatile detail: it points at the self-describing tools and
    # the guide rather than inlining the field list / slug cap (which can change).
    # Both habit lines name a TARGET. Post-flip a bare `mitos check` hard-fails and
    # a bare `mitos status` succeeds while answering about the whole machine — the
    # quieter and worse of the two — so the substring `"mitos status"` alone no
    # longer distinguishes a working recipe from a broken one.
    assert "mitos status ." in block
    assert "record_decision" in block
    assert "mitos check -p ." in block  # the habit line pointing at the conflict sweep (4a)
    assert "SETUP.md" in block.replace("setup.md", "SETUP.md") or "github.com/dovahkiin-v/mitos" in block
    assert "≤100" not in block and "100 characters" not in block  # volatile detail stays out


def test_marker_version_absent_is_none():
    assert marker_version("a CLAUDE.md that talks about mitos but has no marker") is None
    assert marker_version("<!-- mitos-agent-guide: v7 -->") == 7


# --------------------------------------------------------------------------- #
# Per-file drift classification
# --------------------------------------------------------------------------- #

def test_scan_current_marker_is_current(tmp_path):
    (tmp_path / "CLAUDE.md").write_text(agent_block(), encoding="utf-8")
    files = scan_agent_files(str(tmp_path))
    assert len(files) == 1
    assert files[0]["file"] == "CLAUDE.md"
    assert files[0]["status"] == "current"
    assert files[0]["marker_version"] == AGENT_GUIDE_VERSION


def test_scan_old_marker_is_outdated(tmp_path):
    (tmp_path / "AGENTS.md").write_text(
        "<!-- mitos-agent-guide: v0 -->\n## Mitos\nrecord_decision etc.", encoding="utf-8")
    files = scan_agent_files(str(tmp_path))
    assert files[0]["status"] == "outdated"
    assert files[0]["marker_version"] == 0


def test_scan_v1_paste_without_habit_line_is_outdated(tmp_path, capsys):
    """A pre-4a v1 paste (no `mitos check` habit line) flags for refresh.

    The habit line landed at guide v2, so a v1 marker is behind the running version:
    `scan_agent_files` reads it `outdated`, `agent_block_drift` reports `stale`, and
    `mitos agent-block --check` exits 1 — the nudge to re-paste the current block.
    """
    (tmp_path / "AGENTS.md").write_text(
        "<!-- mitos-agent-guide: v1 -->\n## Architectural Decisions — Mitos\n"
        "record_decision etc.", encoding="utf-8")
    files = scan_agent_files(str(tmp_path))
    assert files[0]["status"] == "outdated"
    assert files[0]["marker_version"] == 1
    assert agent_block_drift(str(tmp_path))["stale"] is True
    assert cli.cmd_agent_block(str(tmp_path), check=True) == 1


def test_scan_v2_paste_is_outdated(tmp_path, capsys):
    """The v2 paste that exists on real projects' disks TODAY reads `outdated`.

    Spelled as a literal `2`, not `AGENT_GUIDE_VERSION - 1`. The whole module is
    parametrized on the running constant, so it stays green through any bump —
    including one that never happened — and a self-adjusting spelling here would
    restate that tautology instead of the claim, which is about a specific number
    sitting in specific files right now. The marker's integer is also the ONLY
    migration channel these copies have: `scan_agent_files` classifies a paste by
    that integer and never by comparing block text, so a v3 rewrite that skipped
    the bump would leave every stale copy reported `current`.
    """
    (tmp_path / "AGENTS.md").write_text(
        "<!-- mitos-agent-guide: v2 -->\n## Architectural Decisions — Mitos\n"
        "Run `mitos status`. record_decision etc.", encoding="utf-8")
    files = scan_agent_files(str(tmp_path))
    assert files[0]["status"] == "outdated"
    assert files[0]["marker_version"] == 2
    assert agent_block_drift(str(tmp_path))["stale"] is True
    assert cli.cmd_agent_block(str(tmp_path), check=True) == 1


# A pre-marker pasted block: it has the distinctive heading but no version marker.
_LEGACY_BLOCK = (
    "## Architectural Decisions — Mitos (per-project setup)\n"
    "This project uses mitos. Record with `record_decision`; check `surface_decisions`."
)


def test_scan_legacy_block_heading_without_marker_is_unversioned(tmp_path):
    (tmp_path / "GEMINI.md").write_text(_LEGACY_BLOCK, encoding="utf-8")
    files = scan_agent_files(str(tmp_path))
    assert files[0]["status"] == "unversioned"
    assert files[0]["marker_version"] is None


def test_scan_omits_files_that_never_mention_mitos(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("# Just a project\nNo decision memory here.", encoding="utf-8")
    assert scan_agent_files(str(tmp_path)) == []


def test_scan_omits_file_that_only_discusses_mitos(tmp_path):
    """A file that documents mitos/its tools but isn't a pasted block is NOT flagged.

    Precision guard: mitos's own dev guide (and any doc referencing `record_decision`)
    mentions the tools without carrying the block heading — it must not read as a stale
    paste, or the drift signal cries wolf.
    """
    (tmp_path / "CLAUDE.md").write_text(
        "# Working on Mitos\nmitos is an architectural-decision substrate. Its MCP tools "
        "are `record_decision`, `surface_decisions`, `query_decisions`.", encoding="utf-8")
    assert scan_agent_files(str(tmp_path)) == []


def test_drift_stale_only_when_outdated_or_unversioned(tmp_path):
    # current → not stale
    (tmp_path / "CLAUDE.md").write_text(agent_block(), encoding="utf-8")
    assert agent_block_drift(str(tmp_path))["stale"] is False
    # add an unversioned (legacy heading, no marker) sibling → stale
    (tmp_path / "AGENTS.md").write_text(_LEGACY_BLOCK, encoding="utf-8")
    assert agent_block_drift(str(tmp_path))["stale"] is True


# --------------------------------------------------------------------------- #
# The `mitos agent-block` verb
# --------------------------------------------------------------------------- #

def test_cmd_agent_block_prints_block(tmp_path, capsys):
    rc = cli.cmd_agent_block(str(tmp_path))
    out = capsys.readouterr().out
    assert rc == 0
    assert f"mitos-agent-guide: v{AGENT_GUIDE_VERSION}" in out


def test_cmd_agent_block_check_clean_when_current(tmp_path, capsys):
    (tmp_path / "CLAUDE.md").write_text(agent_block(), encoding="utf-8")
    rc = cli.cmd_agent_block(str(tmp_path), check=True)
    assert rc == 0
    assert "✓ CLAUDE.md" in capsys.readouterr().out


def test_cmd_agent_block_check_flags_stale(tmp_path, capsys):
    (tmp_path / "CLAUDE.md").write_text(_LEGACY_BLOCK, encoding="utf-8")
    rc = cli.cmd_agent_block(str(tmp_path), check=True)
    out = capsys.readouterr().out
    assert rc == 1
    assert "⚠ CLAUDE.md" in out and "mitos agent-block" in out


def test_cmd_agent_block_check_no_agent_file(tmp_path, capsys):
    rc = cli.cmd_agent_block(str(tmp_path), check=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "no agent file references mitos" in out


# --------------------------------------------------------------------------- #
# `mitos status` integration
# --------------------------------------------------------------------------- #

def test_status_json_has_agent_fields(tmp_path, monkeypatch, capsys):
    _init(tmp_path)
    capsys.readouterr()  # discard init message
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, False))
    cli.cmd_status(str(tmp_path), as_json=True)
    report = json.loads(capsys.readouterr().out)
    assert report["agent_guide_version"] == AGENT_GUIDE_VERSION
    assert report["agent_files"] == []  # init writes no agent file


def test_status_text_warns_on_stale_agent_note(tmp_path, monkeypatch, capsys):
    _init(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, True, points=1))
    # A legacy, unversioned mitos block in CLAUDE.md.
    (tmp_path / "CLAUDE.md").write_text(_LEGACY_BLOCK, encoding="utf-8")
    cli.cmd_status(str(tmp_path))
    out = capsys.readouterr().out
    assert "agent-file mitos note out of date" in out
    assert "mitos agent-block" in out


def test_status_text_quiet_when_agent_note_current(tmp_path, monkeypatch, capsys):
    _init(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, True, points=1))
    (tmp_path / "CLAUDE.md").write_text(agent_block(), encoding="utf-8")
    cli.cmd_status(str(tmp_path))
    assert "agent-file mitos note out of date" not in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 3i — every refresh recipe the agent-file lines print is selectored and runs
# --------------------------------------------------------------------------- #

def _run(argv):
    """Drives ``main()`` with ``argv``; returns the exit code."""
    with patch.object(sys, "argv", ["mitos"] + list(argv)):
        try:
            main()
        except SystemExit as exc:
            return exc.code
    return 0


def _recipe_selector(out, lead):
    """Parses the one ``mitos agent-block -p …`` recipe on the line holding ``lead``.

    Goes through the real parser and reads the selector the way
    ``_selector_from_args`` does; returns it.
    """
    (line,) = [ln for ln in out.splitlines() if lead in ln]
    (recipe,) = [part for part in line.split("`") if part.startswith("mitos agent-block")]
    tokens = shlex.split(recipe)
    args = cli._build_parser().parse_args(tokens[1:])
    assert args.command == "agent-block"
    assert args.project_post is not None, f"recipe names no project: {recipe!r}"
    return cli._selector_from_args(args)


def _harbor(tmp_path):
    """An `init`ed workspace registered as ``harbor`` with a stale paste in CLAUDE.md."""
    ws = tmp_path / "x" / "harbor"
    ws.mkdir(parents=True)
    _init(ws)
    (ws / "CLAUDE.md").write_text(_LEGACY_BLOCK, encoding="utf-8")
    return ws


def test_status_drift_line_names_the_registered_project(tmp_path, monkeypatch, capsys):
    """R10: the status drift line's recipe names the project status was asked about."""
    ws = _harbor(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, True, points=1))
    capsys.readouterr()

    cli.cmd_status(str(ws), project="harbor")

    selector = _recipe_selector(capsys.readouterr().out, "agent-file mitos note out of date")
    assert selector == "harbor"
    resolved = cli._resolve_selector(selector, "agent-block")
    assert os.path.realpath(resolved.root) == os.path.realpath(ws)


_OUTDATED = "<!-- mitos-agent-guide: v1 -->\n## Architectural Decisions — Mitos\n"


@pytest.mark.parametrize("paste", [_OUTDATED, _LEGACY_BLOCK], ids=["outdated", "unversioned"])
@pytest.mark.parametrize("given", [True, False], ids=["project_given", "project_omitted"])
def test_check_arms_print_a_selectored_recipe(tmp_path, capsys, paste, given):
    """R11: both stale arms; an omitted project falls back to the workspace's path."""
    (tmp_path / "CLAUDE.md").write_text(paste, encoding="utf-8")
    kwargs = {"project": "harbor"} if given else {}

    assert cli.cmd_agent_block(str(tmp_path), check=True, **kwargs) == 1

    selector = _recipe_selector(capsys.readouterr().out, "⚠ CLAUDE.md")
    assert selector == ("harbor" if given else os.path.abspath(str(tmp_path)))


def test_check_through_main_names_the_addressed_project(tmp_path, monkeypatch, capsys):
    """R12: the dispatch passes `config.project`, so the recipe carries the name.

    The fallback would print the path instead and still parse; only this row sees a
    dropped pass.
    """
    ws = _harbor(tmp_path)
    monkeypatch.chdir(tmp_path)
    capsys.readouterr()

    assert _run(["agent-block", "-p", "harbor", "--check"]) == 1

    assert _recipe_selector(capsys.readouterr().out, "⚠ CLAUDE.md") == "harbor"
    assert os.path.realpath(cli._resolve_selector("harbor", "agent-block").root) == \
        os.path.realpath(ws)


@pytest.mark.parametrize("route", ["no_workspace", "malformed_config"])
def test_paste_line_names_the_directory_when_no_name_exists(tmp_path, monkeypatch,
                                                            capsys, route):
    """R13: the no-file arm, over both routes that carry no registered name.

    A directory with no workspace goes through the ordinary dispatch (it echoes a
    corpus on stderr); a workspace whose config does not parse goes through
    `_answer_workspace_optional_verb`, which echoes none. Both print the root.
    """
    target = tmp_path / route
    target.mkdir()
    if route == "malformed_config":
        (target / ".mitos").mkdir()
        (target / ".mitos" / "config.toml").write_text("qdrant_collection = [1, 2\n",
                                                       encoding="utf-8")
        (target / "decisions.md").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    capsys.readouterr()

    assert _run(["agent-block", str(target), "--check"]) == 0

    captured = capsys.readouterr()
    assert ("corpus: " in captured.err) is (route == "no_workspace")
    selector = _recipe_selector(captured.out, "Paste what")
    assert selector == os.path.abspath(str(target))


def test_habit_line_teaches_check_before_commit_and_no_agent_yes():
    """R14: the order that costs no extra turn, and who authorizes a spend.

    `--yes` may appear only inside the sentence that says an agent does not pass it,
    so deleting that sentence or inverting it reds.
    """
    block = agent_block()
    assert "after recording decisions and before committing" in block
    assert "mitos check -p ." in block
    assert "a commit that follows a record waits until a check has been attempted" in block
    sentences = [s for s in re.split(r"(?<=[.;])\s", block) if "--yes" in s]
    assert sentences == ["an agent does not pass `--yes` on its own."]
