"""`mitos status` reports when `.mitos/skill.md` differs from what this mitos writes.

`skill.md` is written by `mitos init` from a pure template plus the installed
format spec, so what it *should* say is computable — the status fact is a
comparison, not a version marker. These rows pin the four states, their two
channels (one text line for `differs`; a top-level `skill_md` key in `--json`),
that none of them moves readiness, and — by running it — that the printed
refresh recipe works from anywhere without tripping the registry.
"""

import json
import os
import re
import shlex
import subprocess
import sys

from conftest import make_workspace
from mitos import cli, registry
from mitos.config import MitosConfig

DEAD_QDRANT_URL = "http://127.0.0.1:9"

_RECIPE = re.compile(r"skill\.md differs[^\n]*`(mitos -C [^`]+)`")


# --- helpers ---------------------------------------------------------------

def _init(path, name=None):
    """Runs ``cmd_init`` on a fresh ``MitosConfig`` for ``path``."""
    cli.cmd_init(MitosConfig(str(path)), name=name)


def _qdrant(reachable=True, collection_exists=False, points=None):
    """Builds a ``_check_qdrant`` stub (no real Qdrant in status tests)."""
    return lambda url, coll: {
        "reachable": reachable,
        "collection_exists": collection_exists,
        "points": points,
    }


def _ready_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, False))


def _skill_path(ws):
    return ws / ".mitos" / "skill.md"


def _status(ws, capsys, **kwargs):
    """Runs text then JSON status; returns ``(rc, text, payload)``."""
    capsys.readouterr()
    rc = cli.cmd_status(str(ws), **kwargs)
    text = capsys.readouterr().out
    rc_json = cli.cmd_status(str(ws), as_json=True, **kwargs)
    payload = json.loads(capsys.readouterr().out)
    assert rc == rc_json
    return rc, text, payload


def _skill_lines(text):
    return [ln for ln in text.splitlines() if "skill.md" in ln]


def _env():
    return {**os.environ, "MITOS_NO_UPDATE_CHECK": "1",
            "GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", "ANTHROPIC_API_KEY": "",
            "QDRANT_URL": DEAD_QDRANT_URL}


def _mitos(cwd, *argv):
    return subprocess.run([sys.executable, "-m", "mitos.cli", *argv], cwd=cwd,
                          env=_env(), capture_output=True, text=True, timeout=300)


def _run_recipe(recipe, cwd):
    argv = shlex.split(recipe)
    assert argv[0] == "mitos"
    return subprocess.run([sys.executable, "-m", "mitos.cli", *argv[1:]], cwd=cwd,
                          env=_env(), capture_output=True, text=True, timeout=300)


# --- the four states --------------------------------------------------------

def test_a_fresh_init_reads_current_and_prints_nothing(tmp_path, monkeypatch, capsys):
    ws = tmp_path / "fresh"
    ws.mkdir()
    _init(ws)
    _ready_env(monkeypatch)

    rc, text, payload = _status(ws, capsys)
    assert payload["skill_md"] == {"status": "current"}
    assert _skill_lines(text) == []
    assert "READY ✓" in text and rc == 0 and payload["ready"] is True


def test_an_edited_skill_md_differs_with_one_line_and_a_recipe(tmp_path, monkeypatch, capsys):
    """One line, direction-neutral, carrying a recipe — and readiness untouched."""
    ws = tmp_path / "edited"
    ws.mkdir()
    _init(ws)
    with open(_skill_path(ws), "a", encoding="utf-8") as f:
        f.write("a local addition\n")
    _ready_env(monkeypatch)

    rc, text, payload = _status(ws, capsys)
    assert payload["skill_md"] == {"status": "differs"}
    lines = _skill_lines(text)
    assert len(lines) == 1
    line = lines[0]
    assert "differs from what this mitos writes" in line
    assert f"`mitos -C {str(ws)!r} init`" in line          # unregistered route: no --name
    assert "MCP" not in line and "mcp" not in line
    for word in ("outdated", "stale", "behind"):
        assert word not in line
    assert "READY ✓" in text and rc == 0 and payload["ready"] is True

    # A caller that addressed a registered name gets `--name` with that name.
    capsys.readouterr()
    cli.cmd_status(str(ws), project="some-name")
    named = _skill_lines(capsys.readouterr().out)
    assert len(named) == 1
    assert f"`mitos -C {str(ws)!r} init --name 'some-name'`" in named[0]


def test_a_deleted_skill_md_is_absent_and_silent(tmp_path, monkeypatch, capsys):
    ws = tmp_path / "deleted"
    ws.mkdir()
    _init(ws)
    _skill_path(ws).unlink()
    _ready_env(monkeypatch)

    rc, text, payload = _status(ws, capsys)
    assert payload["skill_md"] == {"status": "absent"}
    assert _skill_lines(text) == []
    assert "READY ✓" in text and rc == 0


def test_an_undecodable_skill_md_is_unreadable_not_a_crash(tmp_path, monkeypatch, capsys):
    ws = tmp_path / "undecodable"
    ws.mkdir()
    _init(ws)
    _skill_path(ws).write_bytes(b"\xff\xfe\x00 not utf-8 \xc3\x28")
    _ready_env(monkeypatch)

    rc, text, payload = _status(ws, capsys)
    assert payload["skill_md"] == {"status": "unreadable"}
    assert "READY ✓" in text and rc == 0


def test_crlf_line_endings_still_read_current(tmp_path, monkeypatch, capsys):
    """A checkout with CRLF endings is the same text, and the code — not text mode — says so.

    The file is written as bytes so the `\\r\\n` really is on disk; status reads it
    with `newline=""`, so without the explicit normalization this row reds.
    """
    ws = tmp_path / "crlf"
    ws.mkdir()
    _init(ws)
    body = cli._skill_md_text(cli.load_format_spec())
    assert "\n" in body
    _skill_path(ws).write_bytes(body.replace("\n", "\r\n").encode("utf-8"))
    assert b"\r\n" in _skill_path(ws).read_bytes()
    _ready_env(monkeypatch)

    _, _, payload = _status(ws, capsys)
    assert payload["skill_md"] == {"status": "current"}


def test_a_changed_installed_format_spec_is_seen(tmp_path, monkeypatch, capsys):
    """`skill.md` embeds the spec, so a new spec is a new expected text — a marker misses this.

    The workspace's own `format-spec.md` still holds the original, so this also
    reds a build that compares against the workspace copy instead of the
    installed one.
    """
    ws = tmp_path / "spec_moved"
    ws.mkdir()
    _init(ws)
    original = cli.load_format_spec()
    monkeypatch.setattr(cli, "load_format_spec", lambda: original + "\nA new rule.\n")
    _ready_env(monkeypatch)

    _, text, payload = _status(ws, capsys)
    assert payload["skill_md"] == {"status": "differs"}
    assert len(_skill_lines(text)) == 1


# --- shape at the edges ------------------------------------------------------

def test_no_workspace_gives_null_and_malformed_config_keeps_its_shape(
    tmp_path, monkeypatch, capsys
):
    """Uninitialized → `null`; the malformed-config early return stays exactly as it was.

    That early return answers before any workspace file is read, so a `skill_md`
    there would be a claim about a file status never looked at.
    """
    _ready_env(monkeypatch)
    bare = tmp_path / "bare"
    bare.mkdir()
    capsys.readouterr()
    cli.cmd_status(str(bare), as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["initialized"] is False
    assert "skill_md" in payload and payload["skill_md"] is None

    broken = tmp_path / "broken"
    (broken / ".mitos").mkdir(parents=True)
    (broken / ".mitos" / "config.toml").write_text("this is = = not toml [\n", encoding="utf-8")
    cli.cmd_status(str(broken), as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"report", "project", "workspace", "ready",
                            "initialized", "config_error"}


# --- the recipe runs, verbatim, from elsewhere --------------------------------

def test_the_printed_recipe_refreshes_a_workspace_registered_under_another_name(tmp_path):
    """The half-works trap: a bare `init` rewrites skill.md, then the registry refuses.

    The workspace directory is `ws_y` but its registered name is `renamed-x`, so
    only a recipe carrying `--name 'renamed-x'` exits 0. Run from a directory
    that is not the workspace, through the entry point, exactly as printed.
    """
    ws = tmp_path / "ws_y"
    ws.mkdir()
    _init(ws, name="renamed-x")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with open(_skill_path(ws), "a", encoding="utf-8") as f:
        f.write("drift\n")
    registry_before = open(registry.registry_path(), "rb").read()

    status = _mitos(str(elsewhere), "-p", "renamed-x", "status")
    match = _RECIPE.search(status.stdout)
    assert match, status.stdout + status.stderr
    assert "--name 'renamed-x'" in match.group(1)

    run = _run_recipe(match.group(1), str(elsewhere))
    assert run.returncode == 0, run.stdout + run.stderr

    after = _mitos(str(elsewhere), "-p", "renamed-x", "status", "--json")
    assert json.loads(after.stdout)["skill_md"] == {"status": "current"}
    assert open(registry.registry_path(), "rb").read() == registry_before


def test_the_printed_recipe_refreshes_an_unregistered_workspace(tmp_path):
    """An unregistered path gets the bare `-C … init`, which is also its registration path."""
    ws = tmp_path / "loose_ws"
    ws.mkdir()
    root = make_workspace(ws)
    assert root not in registry.load().values()
    _skill_path(ws).write_text("an old skill\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    status = _mitos(str(elsewhere), "status", str(ws))
    match = _RECIPE.search(status.stdout)
    assert match, status.stdout + status.stderr
    assert "--name" not in match.group(1)

    run = _run_recipe(match.group(1), str(elsewhere))
    assert run.returncode == 0, run.stdout + run.stderr

    after = _mitos(str(elsewhere), "status", str(ws), "--json")
    assert json.loads(after.stdout)["skill_md"] == {"status": "current"}
