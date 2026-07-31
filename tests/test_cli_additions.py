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
def test_record_decision_alias_routes(mock_record, monkeypatch, workspace):
    monkeypatch.setattr(sys, "argv", ["mitos", "-p", workspace, "record_decision", "ax", "--rejected", "r", "--slug", "s"])
    main()
    assert mock_record.called


@patch("mitos.cli.cmd_surface")
def test_surface_verb_routes_with_scope(mock_surface, monkeypatch, workspace):
    monkeypatch.setattr(sys, "argv", ["mitos", "-p", workspace, "surface", "a claim", "--scope", "db"])
    main()
    mock_surface.assert_called_once()
    args, kwargs = mock_surface.call_args
    assert args[1] == "a claim"
    assert kwargs["scope"] == "db"


@patch("mitos.cli.cmd_surface")
def test_surface_decisions_alias_routes(mock_surface, monkeypatch, workspace):
    monkeypatch.setattr(sys, "argv", ["mitos", "-p", workspace, "surface_decisions", "claim"])
    main()
    assert mock_surface.called


@patch("mitos.cli.cmd_query")
def test_query_decisions_alias_routes(mock_query, monkeypatch, workspace):
    monkeypatch.setattr(sys, "argv", ["mitos", "-p", workspace, "query_decisions", "claim"])
    main()
    assert mock_query.called


@patch("mitos.cli.cmd_query")
def test_query_json_brief_routes(mock_query, monkeypatch, workspace):
    """`query c --json --brief` threads as_json=True, brief=True (non-exhaustive)."""
    monkeypatch.setattr(sys, "argv", ["mitos", "-p", workspace, "query", "claim", "--json", "--brief"])
    main()
    assert mock_query.called
    _, kwargs = mock_query.call_args
    assert kwargs["as_json"] is True and kwargs["brief"] is True


@patch("mitos.cli.cmd_query")
def test_query_limit_routes(mock_query, monkeypatch, workspace):
    """`query c --limit 7` threads limit=7 through the parser + dispatch."""
    monkeypatch.setattr(sys, "argv", ["mitos", "-p", workspace, "query", "claim", "--limit", "7"])
    main()
    assert mock_query.called
    _, kwargs = mock_query.call_args
    assert kwargs["limit"] == 7


@patch("mitos.cli.cmd_surface")
def test_surface_limit_routes(mock_surface, monkeypatch, workspace):
    """`surface c --limit 7` threads limit=7 through the parser + dispatch."""
    monkeypatch.setattr(sys, "argv", ["mitos", "-p", workspace, "surface", "claim", "--limit", "7"])
    main()
    assert mock_surface.called
    _, kwargs = mock_surface.call_args
    assert kwargs["limit"] == 7


@patch("mitos.cli.cmd_open_questions")
def test_open_questions_json_routes(mock_oq, monkeypatch, workspace):
    """`open-questions --json` threads as_json=True through to the handler."""
    monkeypatch.setattr(sys, "argv", ["mitos", "-p", workspace, "open-questions", "--json"])
    main()
    assert mock_oq.called
    _, kwargs = mock_oq.call_args
    assert kwargs["as_json"] is True


@patch("mitos.cli.cmd_record")
def test_record_json_routes(mock_record, monkeypatch, workspace):
    """`record … --json` threads as_json=True through to the handler."""
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "-p", workspace, "record", "ax", "--rejected", "r", "--slug", "s", "--json"])
    main()
    assert mock_record.called
    _, kwargs = mock_record.call_args
    assert kwargs["as_json"] is True


# --- file / stdin prose input --------------------------------------------------

def test_read_text_arg_inline(workspace):
    assert cli._read_text_arg("inline", None) == "inline"


def test_read_text_arg_from_file(tmp_path, workspace):
    f = tmp_path / "r.txt"
    f.write_text("prose with Camila's apostrophe", encoding="utf-8")
    assert "Camila's" in cli._read_text_arg(None, str(f))


def test_read_text_arg_from_stdin(monkeypatch, workspace):
    monkeypatch.setattr(sys, "stdin", io.StringIO("from stdin"))
    assert cli._read_text_arg(None, "-") == "from stdin"


@patch("mitos.cli.cmd_record")
def test_record_reads_rejected_from_file(mock_record, tmp_path, monkeypatch, workspace):
    rf = tmp_path / "rej.txt"
    rf.write_text("rejected prose, apostrophe-safe: Camila's", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["mitos", "-p", workspace, "record", "ax", "--rejected-file", str(rf), "--slug", "s"])
    main()
    _, kwargs = mock_record.call_args
    assert kwargs["rejected"] == "rejected prose, apostrophe-safe: Camila's"


def test_record_requires_rejected(monkeypatch, workspace):
    monkeypatch.setattr(sys, "argv", ["mitos", "-p", workspace, "record", "ax", "--slug", "s"])  # neither --rejected nor --rejected-file
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

def test_mcp_wired_detection(tmp_path, workspace):
    assert cli._mcp_wired(str(tmp_path)) is False
    (tmp_path / ".mcp.json").write_text('{"mcpServers": {"mitos": {"command": "mitos"}}}')
    assert cli._mcp_wired(str(tmp_path)) is True
    (tmp_path / ".mcp.json").write_text('{"mcpServers": {"other": {}}}')
    assert cli._mcp_wired(str(tmp_path)) is False


def test_mcp_hint_fires_then_rate_limits(tmp_path, monkeypatch, workspace):
    monkeypatch.delenv("MITOS_NO_MCP_HINT", raising=False)
    first = cli._mcp_hint(str(tmp_path))
    assert first is not None and "wire the MCP" in first
    assert cli._mcp_hint(str(tmp_path)) is None  # within 24h → silent


def test_mcp_hint_silent_when_wired(tmp_path, monkeypatch, workspace):
    monkeypatch.delenv("MITOS_NO_MCP_HINT", raising=False)
    (tmp_path / ".mcp.json").write_text('{"mcpServers": {"mitos": {"command": "mitos"}}}')
    assert cli._mcp_hint(str(tmp_path)) is None


def test_mcp_hint_opt_out(tmp_path, monkeypatch, workspace):
    monkeypatch.setenv("MITOS_NO_MCP_HINT", "1")
    assert cli._mcp_hint(str(tmp_path)) is None


def test_decision_loop_commands_cover_aliases(workspace):
    for verb in ("record", "record_decision", "surface", "surface_decisions",
                 "query", "query_decisions", "list", "list_decisions"):
        assert verb in cli._DECISION_LOOP_COMMANDS
    for non_verb in ("init", "status", "sync", "serve", "set-key"):
        assert non_verb not in cli._DECISION_LOOP_COMMANDS


# --- Phase 6a: help-as-API-doc (gate T12) -------------------------------------

_ALIASES = ("query_decisions", "surface_decisions", "list_decisions", "record_decision")


def test_help_renders_epilog_worked_examples(monkeypatch, capsys):
    """Criterion 1: `mitos --help` exits 0 and renders the worked-examples epilog,
    the surface→record compose, and the relation-edge guidance.

    Every worked example names its project since 5a — the epilog's own comment
    promises the block stays runnable, and a selectorless `mitos surface …` is now
    a teaching error rather than a lesson."""
    monkeypatch.setattr(sys, "argv", ["mitos", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Examples:" in out
    # the surface→record compose appears as runnable example commands
    assert "mitos -p myproject surface" in out
    assert "mitos -p myproject record" in out
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
def test_list_decisions_alias_routes(mock_list, monkeypatch, workspace):
    """Criterion 3 (gap fill): the `list_decisions` alias still routes."""
    monkeypatch.setattr(sys, "argv", ["mitos", "-p", workspace, "list_decisions"])
    main()
    assert mock_list.called


@patch("mitos.cli.cmd_scopes")
def test_list_scopes_alias_routes(mock_scopes, monkeypatch, workspace):
    """Criterion 3 (gap fill): the `list_scopes` alias still routes."""
    monkeypatch.setattr(sys, "argv", ["mitos", "-p", workspace, "list_scopes"])
    main()
    assert mock_scopes.called


def test_surface_decisions_mcp_description_names_compose(workspace):
    """Criterion 4 (W15): the surfacing tools' descriptions name the
    surface→record compose so an MCP agent discovers the write-back step."""
    from mitos.mcp_server import surface_decisions, query_decisions
    assert "record_decision" in (surface_decisions.__doc__ or "")
    assert "record_decision" in (query_decisions.__doc__ or "")


# --- --axiom-file (quoting-safe axiom, symmetric with --rejected-file) ----------

@patch("mitos.cli.cmd_record")
def test_record_reads_axiom_from_file(mock_record, tmp_path, monkeypatch, workspace):
    af = tmp_path / "axiom.txt"
    af.write_text("Camila's axiom, apostrophe-safe\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "-p", workspace, "record", "--axiom-file", str(af),
                         "--rejected", "r", "--slug", "s"])
    main()
    _, kwargs = mock_record.call_args
    # The single trailing newline a file/heredoc adds is stripped.
    assert kwargs["axiom"] == "Camila's axiom, apostrophe-safe"


@patch("mitos.cli.cmd_record")
def test_record_reads_axiom_from_stdin(mock_record, monkeypatch, workspace):
    monkeypatch.setattr(sys, "stdin", io.StringIO("axiom from stdin\n"))
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "-p", workspace, "record", "--axiom-file", "-",
                         "--rejected", "r", "--slug", "s"])
    main()
    _, kwargs = mock_record.call_args
    assert kwargs["axiom"] == "axiom from stdin"


def test_record_rejects_both_axiom_sources(tmp_path, monkeypatch, capsys, workspace):
    af = tmp_path / "axiom.txt"
    af.write_text("file axiom", encoding="utf-8")
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "-p", workspace, "record", "inline axiom", "--axiom-file", str(af),
                         "--rejected", "r", "--slug", "s"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "exactly one axiom source" in capsys.readouterr().err


def test_record_rejects_neither_axiom_source(monkeypatch, capsys, workspace):
    monkeypatch.setattr(sys, "argv", ["mitos", "-p", workspace, "record", "--rejected", "r", "--slug", "s"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "exactly one axiom source" in capsys.readouterr().err


def test_record_neither_axiom_source_json_speaks_json(monkeypatch, capsys, workspace):
    """Under --json the dead-end is a structured object on stdout, exit 2 preserved."""
    import json
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "-p", workspace, "record", "--rejected", "r", "--slug", "s", "--json"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "missing_axiom"


# --- stdin arity on record's prose-file args -----------------------------------

@pytest.mark.parametrize("argv_extra, expected_in_msg", [
    (["--axiom-file", "-", "--rejected-file", "-"], "--axiom-file"),
    (["--axiom-file", "-", "--rejected-file", "-", "--context-file", "-"], "--context-file"),
])
def test_multiple_stdin_file_args_fail_with_their_own_error(
    argv_extra, expected_in_msg, monkeypatch, capsys
, workspace):
    """Only one argument can read stdin; asking twice names that, not a missing flag.

    Regression: the first reader drained stdin and the rest came back empty, so the
    failure surfaced downstream as "record requires --rejected or --rejected-file" —
    a wall naming a flag the caller had already passed.
    """
    monkeypatch.setattr(sys, "argv", ["mitos", "-p", workspace, "record", "--slug", "s"] + argv_extra)
    monkeypatch.setattr(sys, "stdin", io.StringIO("some prose"))
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "only one argument can read from stdin" in err
    assert expected_in_msg in err
    assert "requires --rejected" not in err, "must not send the caller after a flag they passed"


def test_single_stdin_file_arg_still_works(monkeypatch, workspace):
    """The guard must not break the ordinary one-arg-from-stdin case."""
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "-p", workspace, "record", "--slug", "s", "--axiom-file", "-", "--rejected", "r"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("An axiom from stdin.\n"))
    with patch("mitos.cli.cmd_record") as mock_record:
        main()
    assert mock_record.called
    assert mock_record.call_args.kwargs["axiom"] == "An axiom from stdin."


# --- the intake leak: repeated --scope, and --axiom's prefix swallow -------------
#
# Both bugs were found from the write side by two different loop Claudes in
# consecutive AX_FEEDBACK rounds (10 and 11, 2026-07-25). They are the *source* of
# the scope divergence the corpus↔graph detector reports, so they close first.

def test_repeated_scope_flags_accumulate(monkeypatch, workspace):
    """`--scope a --scope b` keeps BOTH — it used to silently keep only the last.

    ``nargs="*"`` alone overwrites the destination on each occurrence, and the
    receipt then echoed the truncated value back as if it were the request. All three
    measured scope divergences on the live corpus trace to exactly this.
    """
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "-p", workspace, "record", "ax", "--rejected", "r", "--slug", "s",
                         "--scope", "config", "--scope", "sync", "--scope", "substrate"])
    with patch("mitos.cli.cmd_record") as mock_record:
        main()
    assert mock_record.call_args.kwargs["scope"] == ["config", "sync", "substrate"]


def test_space_separated_scope_still_works(monkeypatch, workspace):
    """The documented spelling keeps working — `extend` must not cost the old form."""
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "-p", workspace, "record", "ax", "--rejected", "r", "--slug", "s",
                         "--scope", "config", "sync"])
    with patch("mitos.cli.cmd_record") as mock_record:
        main()
    assert mock_record.call_args.kwargs["scope"] == ["config", "sync"]


def test_mixed_scope_spellings_accumulate(monkeypatch, workspace):
    """Repeated and space-separated forms compose rather than fight."""
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "-p", workspace, "record", "ax", "--rejected", "r", "--slug", "s",
                         "--scope", "config", "sync", "--scope", "substrate"])
    with patch("mitos.cli.cmd_record") as mock_record:
        main()
    assert mock_record.call_args.kwargs["scope"] == ["config", "sync", "substrate"]


def test_extend_does_not_mutate_the_default_list_in_place(workspace):
    """`action="extend"` + `default=[]` must not accumulate across parses.

    The classic mutable-default gotcha: an action that extends the default list in
    place leaks the first parse's values into every later one. It is only observable
    on ONE parser parsed twice — going through ``main()`` builds a fresh parser each
    time, which re-evaluates ``default=[]`` and hides the bug entirely.
    """
    from mitos.cli import _build_parser

    parser = _build_parser()
    base = ["record", "ax", "--rejected", "r", "--slug", "s"]
    first = parser.parse_args(base + ["--scope", "first", "--mechanisms", "m1"])
    second = parser.parse_args(base + ["--scope", "second", "--mechanisms", "m2"])
    third = parser.parse_args(base)

    assert first.scope == ["first"] and second.scope == ["second"]
    assert first.mechanisms == ["m1"] and second.mechanisms == ["m2"]
    assert third.scope == [], "the default must still be empty after two extends"
    assert third.mechanisms is None


def test_axiom_flag_is_not_swallowed_by_axiom_file(monkeypatch, capsys, workspace):
    """`--axiom "prose"` dies as an argparse error, never as `File name too long`.

    ``allow_abbrev`` defaults to True on every subparser, so ``--axiom`` was an
    unambiguous prefix of ``--axiom-file`` and argparse handed a whole axiom to the
    file reader. The command then died through the outermost boundary as an
    ``[Errno 36] File name too long`` "Fatal Unexpected Error" — a wall pointing at
    nothing the caller could act on.
    """
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "-p", workspace, "record", "--axiom", "A decision stated as prose.",
                         "--rejected", "r", "--slug", "s"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "File name too long" not in combined
    assert "Traceback" not in combined
    assert "--axiom" in combined, "the error must name the flag the caller actually typed"


def test_abbreviation_is_off_on_every_subparser(workspace):
    """No verb abbreviates — a per-parser flag is one forgotten kwarg from regressing.

    Setting ``allow_abbrev=False`` on the top-level parser does NOT propagate to
    ``add_parser()`` children (verified: the child keeps ``allow_abbrev: True``), so
    it is pinned across the whole registered set rather than on ``record`` alone.
    Asserted on the grammar rather than by running a verb: proving a spelling by
    executing it costs a real workspace read, and on a prefix bug a real write.
    """
    from mitos.cli import _build_parser

    verbs = _build_parser()._subparsers._group_actions[0].choices
    assert verbs, "the parser must register verbs for this to mean anything"
    abbreviating = sorted(name for name, p in verbs.items() if p.allow_abbrev)
    assert abbreviating == [], f"these verbs still abbreviate options: {abbreviating}"


def test_abbreviated_option_is_rejected_by_the_grammar(workspace):
    """`--jso` no longer resolves to `--json`; `--axiom` no longer reaches --axiom-file."""
    from mitos.cli import _build_parser

    parser = _build_parser()
    for argv in (["list", "--jso"],
                 ["record", "--axiom", "prose", "--rejected", "r", "--slug", "s"]):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(argv)
        assert exc.value.code == 2, argv


def test_repeated_mechanisms_flags_accumulate(monkeypatch, workspace):
    """`--mechanisms a --mechanisms b` keeps BOTH — and this one is canonical core.

    The same silent-truncation shape as `--scope`, registered on the very next line
    of the same verb, but with a strictly worse consequence: ``mechanisms`` feeds
    ``compute_node_id`` (``mechanism_refs``), so a dropped value gives the decision a
    different content-hash id than the author asked for. Re-recording with the full
    list then mints a *second* node instead of correcting the first, because the core
    is immutable by construction (M1).
    """
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "-p", workspace, "record", "ax", "--rejected", "r", "--slug", "s",
                         "--mechanisms", "sqlite", "--mechanisms", "wal-mode"])
    with patch("mitos.cli.cmd_record") as mock_record:
        main()
    assert mock_record.call_args.kwargs["mechanisms"] == ["sqlite", "wal-mode"]


def test_space_separated_mechanisms_still_works(monkeypatch, workspace):
    """The documented spelling keeps working, and an absent flag still yields None."""
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "-p", workspace, "record", "ax", "--rejected", "r", "--slug", "s",
                         "--mechanisms", "sqlite", "wal-mode"])
    with patch("mitos.cli.cmd_record") as mock_record:
        main()
    assert mock_record.call_args.kwargs["mechanisms"] == ["sqlite", "wal-mode"]

    monkeypatch.setattr(sys, "argv",
                        ["mitos", "-p", workspace, "record", "ax", "--rejected", "r", "--slug", "s"])
    with patch("mitos.cli.cmd_record") as mock_record:
        main()
    assert mock_record.call_args.kwargs["mechanisms"] is None, (
        "absent must stay None — `[]` and None are distinguished downstream"
    )
