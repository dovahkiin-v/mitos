"""Tests for CLI additions: MCP-name aliases, the `surface` verb, file/stdin
prose input, `--version`, and the MCP-wiring hint."""

import io
import sys

import pytest
from unittest.mock import patch

from mitos import cli
from mitos.cli import main


# --- aliases + surface routing -------------------------------------------------

@patch("mitos.cli.cmd_record")
def test_record_decision_alias_routes(mock_record, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mitos", "record_decision", "ax", "--rejected", "r", "--slug", "s"])
    main()
    assert mock_record.called


@patch("mitos.cli.cmd_surface")
def test_surface_verb_routes_with_scope(mock_surface, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mitos", "surface", "a claim", "--scope", "db"])
    main()
    mock_surface.assert_called_once()
    args, kwargs = mock_surface.call_args
    assert args[1] == "a claim"
    assert kwargs["scope"] == "db"


@patch("mitos.cli.cmd_surface")
def test_surface_decisions_alias_routes(mock_surface, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mitos", "surface_decisions", "claim"])
    main()
    assert mock_surface.called


@patch("mitos.cli.cmd_query")
def test_query_decisions_alias_routes(mock_query, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mitos", "query_decisions", "claim"])
    main()
    assert mock_query.called


@patch("mitos.cli.cmd_query")
def test_query_json_brief_routes(mock_query, monkeypatch):
    """`query c --json --brief` threads as_json=True, brief=True (non-exhaustive)."""
    monkeypatch.setattr(sys, "argv", ["mitos", "query", "claim", "--json", "--brief"])
    main()
    assert mock_query.called
    _, kwargs = mock_query.call_args
    assert kwargs["as_json"] is True and kwargs["brief"] is True


# --- file / stdin prose input --------------------------------------------------

def test_read_text_arg_inline():
    assert cli._read_text_arg("inline", None) == "inline"


def test_read_text_arg_from_file(tmp_path):
    f = tmp_path / "r.txt"
    f.write_text("prose with Camila's apostrophe", encoding="utf-8")
    assert "Camila's" in cli._read_text_arg(None, str(f))


def test_read_text_arg_from_stdin(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("from stdin"))
    assert cli._read_text_arg(None, "-") == "from stdin"


@patch("mitos.cli.cmd_record")
def test_record_reads_rejected_from_file(mock_record, tmp_path, monkeypatch):
    rf = tmp_path / "rej.txt"
    rf.write_text("rejected prose, apostrophe-safe: Camila's", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["mitos", "record", "ax", "--rejected-file", str(rf), "--slug", "s"])
    main()
    _, kwargs = mock_record.call_args
    assert kwargs["rejected"] == "rejected prose, apostrophe-safe: Camila's"


def test_record_requires_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mitos", "record", "ax", "--slug", "s"])  # neither --rejected nor --rejected-file
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


# --- --version -----------------------------------------------------------------

def test_version_flag_prints_and_exits_zero(monkeypatch, capsys):
    from mitos import __version__
    monkeypatch.setattr(sys, "argv", ["mitos", "--version"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


# --- MCP wiring detection + hint ----------------------------------------------

def test_mcp_wired_detection(tmp_path):
    assert cli._mcp_wired(str(tmp_path)) is False
    (tmp_path / ".mcp.json").write_text('{"mcpServers": {"mitos": {"command": "mitos"}}}')
    assert cli._mcp_wired(str(tmp_path)) is True
    (tmp_path / ".mcp.json").write_text('{"mcpServers": {"other": {}}}')
    assert cli._mcp_wired(str(tmp_path)) is False


def test_mcp_hint_fires_then_rate_limits(tmp_path, monkeypatch):
    monkeypatch.delenv("MITOS_NO_MCP_HINT", raising=False)
    first = cli._mcp_hint(str(tmp_path))
    assert first is not None and "wire the MCP" in first
    assert cli._mcp_hint(str(tmp_path)) is None  # within 24h → silent


def test_mcp_hint_silent_when_wired(tmp_path, monkeypatch):
    monkeypatch.delenv("MITOS_NO_MCP_HINT", raising=False)
    (tmp_path / ".mcp.json").write_text('{"mcpServers": {"mitos": {"command": "mitos"}}}')
    assert cli._mcp_hint(str(tmp_path)) is None


def test_mcp_hint_opt_out(tmp_path, monkeypatch):
    monkeypatch.setenv("MITOS_NO_MCP_HINT", "1")
    assert cli._mcp_hint(str(tmp_path)) is None


def test_decision_loop_commands_cover_aliases():
    for verb in ("record", "record_decision", "surface", "surface_decisions",
                 "query", "query_decisions", "list", "list_decisions"):
        assert verb in cli._DECISION_LOOP_COMMANDS
    for non_verb in ("init", "status", "sync", "serve", "set-key"):
        assert non_verb not in cli._DECISION_LOOP_COMMANDS
