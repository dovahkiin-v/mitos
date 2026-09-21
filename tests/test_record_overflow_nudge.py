"""B6: the write-time overflow nudge across its three encodings.

`scope_overflow` is composed once on the shared write path and names no command, so it
reaches MCP and CLI `--json` verbatim. Only CLI text adds a recipe, one selectored
`mitos status -p <project>` line on stderr, proven through the real parser. Drivers
follow `test_record_rotation.py`'s (the other post-receipt field on both surfaces).
"""

import json
import shlex
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

import mitos.renderer as R
from mitos import cli
from mitos.cli import cmd_init, cmd_record
from mitos.config import MitosConfig
from test_mcp_selector import FORBIDDEN_SYNTAX

RECIPE_PREFIX = "  Per-file breakdown: "


@pytest.fixture(autouse=True)
def offline_and_over(monkeypatch):
    """Keyless and serviceless, and every render over a squeezed ceiling."""
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:9")
    down = MagicMock(side_effect=Exception("backend down"))
    monkeypatch.setattr("mitos.sync.GeminiEmbeddingProvider", down)
    monkeypatch.setattr("mitos.sync.QdrantVectorStore", down)
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", 100)
    monkeypatch.setattr(R, "GLOBAL_OVERFLOW_WARN_CHARS", 100)


@pytest.fixture
def make_ws(tmp_path):
    def _make(name: str = "ws", project=None) -> MitosConfig:
        root = tmp_path / name
        root.mkdir()
        config = MitosConfig(str(root), project=project)
        cmd_init(config)
        return config
    return _make


def _cli_record(config: MitosConfig, **kwargs) -> None:
    cmd_record(config, axiom="An overflowing axiom.", rejected="rej", scope=["substrate"],
               slug="over", acknowledge_neighbors=True, **kwargs)


def _mcp_record(config: MitosConfig) -> Dict:
    from mitos.mcp_server import record_decision
    with patch("mitos.mcp_server.MitosConfig", return_value=config):
        return json.loads(record_decision(
            "An overflowing axiom.", "rej", ["substrate"], slug="over",
            acknowledge_neighbors=True, project=config.workspace_dir))


def _assert_mcp_register(text: str) -> None:
    """No shell command in the string (R6). Files are named, so no bare `-p` check —
    a scope such as `api-policy` renders `api-policy.md` (scout #3)."""
    flat = " ".join(text.split())
    assert "mitos " not in flat
    assert "`" not in flat
    for syntax in FORBIDDEN_SYNTAX:
        assert syntax not in flat


def test_mcp_scope_overflow_is_the_shared_string_and_names_no_command(make_ws, capsys) -> None:
    """R6: the MCP tool returns the shared string verbatim — the same words CLI --json
    carries for the same corpus — and it names no command."""
    cli_ws, mcp_ws = make_ws("cli"), make_ws("mcp")
    capsys.readouterr()
    _cli_record(cli_ws, as_json=True)
    cli_nudge = json.loads(capsys.readouterr()[0])["scope_overflow"]
    mcp_nudge = _mcp_record(mcp_ws)["scope_overflow"]
    _assert_mcp_register(mcp_nudge)
    assert mcp_nudge == cli_nudge
    assert "substrate.md " in mcp_nudge and "live_axioms.md " in mcp_nudge


@pytest.mark.parametrize("project", [None, "mitos-test-realm", "my realm"],
                         ids=["path", "name", "spaced-name"])
def test_cli_recipe_parses_to_status_with_this_project(make_ws, capsys, project) -> None:
    """R7: the recipe line's backticked span goes through the real parser to
    `status` with this project's selector — a path when unregistered (G7)."""
    config = make_ws(project=project)
    capsys.readouterr()
    _cli_record(config)
    _, err = capsys.readouterr()
    [line] = [l for l in err.splitlines() if l.startswith(RECIPE_PREFIX)]
    recipe = line[len(RECIPE_PREFIX):].strip("`")
    argv = shlex.split(recipe)
    assert argv[0] == "mitos"
    args = cli._build_parser().parse_args(argv[1:])
    assert args.command == "status"
    assert args.project_post == config.project
    assert args.project_post == (project or config.workspace_dir)


def test_cli_text_prints_nudge_then_recipe_after_the_receipt(make_ws, capsys) -> None:
    """R8, text: the receipt is intact on stdout; stderr carries the shared string and,
    on the next line, the recipe."""
    config = make_ws()
    capsys.readouterr()
    _cli_record(config)
    out, err = capsys.readouterr()
    assert "Recorded decision 'over'" in out
    assert "rendered axiom" not in out and RECIPE_PREFIX not in out
    lines = err.splitlines()
    [at] = [i for i, l in enumerate(lines) if l.startswith("⚠ ") and "rendered axiom" in l]
    assert lines[at + 1].startswith(RECIPE_PREFIX)


def test_cli_json_carries_the_mcp_wording_and_no_recipe(make_ws, capsys) -> None:
    """R8, --json: `scope_overflow` is the shared string, and no recipe appears on
    either stream."""
    config = make_ws()
    capsys.readouterr()
    _cli_record(config, as_json=True)
    out, err = capsys.readouterr()
    _assert_mcp_register(json.loads(out)["scope_overflow"])
    assert "Per-file breakdown" not in out and "Per-file breakdown" not in err
    assert "mitos status" not in out + err
