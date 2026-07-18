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


@patch("mitos.cli.cmd_query")
def test_query_limit_routes(mock_query, monkeypatch):
    """`query c --limit 7` threads limit=7 through the parser + dispatch."""
    monkeypatch.setattr(sys, "argv", ["mitos", "query", "claim", "--limit", "7"])
    main()
    assert mock_query.called
    _, kwargs = mock_query.call_args
    assert kwargs["limit"] == 7


@patch("mitos.cli.cmd_surface")
def test_surface_limit_routes(mock_surface, monkeypatch):
    """`surface c --limit 7` threads limit=7 through the parser + dispatch."""
    monkeypatch.setattr(sys, "argv", ["mitos", "surface", "claim", "--limit", "7"])
    main()
    assert mock_surface.called
    _, kwargs = mock_surface.call_args
    assert kwargs["limit"] == 7


@patch("mitos.cli.cmd_open_questions")
def test_open_questions_json_routes(mock_oq, monkeypatch):
    """`open-questions --json` threads as_json=True through to the handler."""
    monkeypatch.setattr(sys, "argv", ["mitos", "open-questions", "--json"])
    main()
    assert mock_oq.called
    _, kwargs = mock_oq.call_args
    assert kwargs["as_json"] is True


@patch("mitos.cli.cmd_record")
def test_record_json_routes(mock_record, monkeypatch):
    """`record … --json` threads as_json=True through to the handler."""
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "record", "ax", "--rejected", "r", "--slug", "s", "--json"])
    main()
    assert mock_record.called
    _, kwargs = mock_record.call_args
    assert kwargs["as_json"] is True


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


# --- Phase 6a: help-as-API-doc (gate T12) -------------------------------------

_ALIASES = ("query_decisions", "surface_decisions", "list_decisions", "record_decision")


def test_help_renders_epilog_worked_examples(monkeypatch, capsys):
    """Criterion 1: `mitos --help` exits 0 and renders the worked-examples epilog,
    the surface→record compose, and the relation-edge guidance."""
    monkeypatch.setattr(sys, "argv", ["mitos", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Examples:" in out
    # the surface→record compose appears as runnable example commands
    assert "mitos surface" in out
    assert "mitos record" in out
    # relation-edge guidance — and the recurring "retired" misuse fenced off
    assert "--supersedes" in out and "--corrects" in out
    assert "retired" in out


def test_help_usage_banner_collapsed_no_alias_brace_list(monkeypatch, capsys):
    """Criterion 2: the usage *banner* shows COMMAND and none of the MCP-name
    aliases (they double its width). Assert on the usage block only — the aliases
    legitimately remain in the command-listing body (`query (query_decisions)`)."""
    monkeypatch.setattr(sys, "argv", ["mitos", "--help"])
    with pytest.raises(SystemExit):
        main()
    out = capsys.readouterr().out
    # the usage block is everything before the first blank line (the description)
    usage_block = out.split("\n\n", 1)[0]
    assert "COMMAND" in usage_block
    for alias in _ALIASES:
        assert alias not in usage_block


@patch("mitos.cli.cmd_list")
def test_list_decisions_alias_routes(mock_list, monkeypatch):
    """Criterion 3 (gap fill): the `list_decisions` alias still routes."""
    monkeypatch.setattr(sys, "argv", ["mitos", "list_decisions"])
    main()
    assert mock_list.called


@patch("mitos.cli.cmd_scopes")
def test_list_scopes_alias_routes(mock_scopes, monkeypatch):
    """Criterion 3 (gap fill): the `list_scopes` alias still routes."""
    monkeypatch.setattr(sys, "argv", ["mitos", "list_scopes"])
    main()
    assert mock_scopes.called


def test_surface_decisions_mcp_description_names_compose():
    """Criterion 4 (W15): the surfacing tools' descriptions name the
    surface→record compose so an MCP agent discovers the write-back step."""
    from mitos.mcp_server import surface_decisions, query_decisions
    assert "record_decision" in (surface_decisions.__doc__ or "")
    assert "record_decision" in (query_decisions.__doc__ or "")


# --- --axiom-file (quoting-safe axiom, symmetric with --rejected-file) ----------

@patch("mitos.cli.cmd_record")
def test_record_reads_axiom_from_file(mock_record, tmp_path, monkeypatch):
    af = tmp_path / "axiom.txt"
    af.write_text("Camila's axiom, apostrophe-safe\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "record", "--axiom-file", str(af),
                         "--rejected", "r", "--slug", "s"])
    main()
    _, kwargs = mock_record.call_args
    # The single trailing newline a file/heredoc adds is stripped.
    assert kwargs["axiom"] == "Camila's axiom, apostrophe-safe"


@patch("mitos.cli.cmd_record")
def test_record_reads_axiom_from_stdin(mock_record, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("axiom from stdin\n"))
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "record", "--axiom-file", "-",
                         "--rejected", "r", "--slug", "s"])
    main()
    _, kwargs = mock_record.call_args
    assert kwargs["axiom"] == "axiom from stdin"


def test_record_rejects_both_axiom_sources(tmp_path, monkeypatch, capsys):
    af = tmp_path / "axiom.txt"
    af.write_text("file axiom", encoding="utf-8")
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "record", "inline axiom", "--axiom-file", str(af),
                         "--rejected", "r", "--slug", "s"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "exactly one axiom source" in capsys.readouterr().err


def test_record_rejects_neither_axiom_source(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mitos", "record", "--rejected", "r", "--slug", "s"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "exactly one axiom source" in capsys.readouterr().err


def test_record_neither_axiom_source_json_speaks_json(monkeypatch, capsys):
    """Under --json the dead-end is a structured object on stdout, exit 2 preserved."""
    import json
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "record", "--rejected", "r", "--slug", "s", "--json"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "missing_axiom"
