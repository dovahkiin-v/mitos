"""Tests for CLI additions: MCP-name aliases, the `surface` verb, file/stdin
prose input, `--version`, and the MCP-wiring hint."""

import io
import re
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


# --- MCP project-entry (shadowing) detection ----------------------------------

def test_mcp_project_entry_detection(tmp_path):
    """The same `.mcp.json` read as the retired `_mcp_wired`, meaning the opposite.

    `True` is now a FINDING — a project-scope entry under the server name `mitos`
    wins by name over the machine-wide registration and erases it, so this is the
    state worth reporting, not the state worth congratulating.
    """
    assert cli._mcp_project_entry(str(tmp_path)) is False
    (tmp_path / ".mcp.json").write_text('{"mcpServers": {"mitos": {"command": "mitos"}}}')
    assert cli._mcp_project_entry(str(tmp_path)) is True
    (tmp_path / ".mcp.json").write_text('{"mcpServers": {"other": {}}}')
    assert cli._mcp_project_entry(str(tmp_path)) is False


def test_mcp_project_entry_keys_on_the_server_name_not_the_command(tmp_path):
    """A server registered under a DIFFERENT key does not shadow, so it is not flagged.

    Precedence is keyed on the entry's name, so `mitos-local` coexists with the
    machine-wide `mitos` entry rather than erasing it. Widening the predicate to
    "any entry whose command mentions mitos" would flag a harmless registration —
    the cries-wolf failure the note exists to avoid.
    """
    (tmp_path / ".mcp.json").write_text(
        '{"mcpServers": {"mitos-local": {"command": "mitos", "args": ["serve"]}}}'
    )
    assert cli._mcp_project_entry(str(tmp_path)) is False


def test_mcp_project_entry_fails_silent_on_a_malformed_file(tmp_path):
    """Unreadable/malformed input reports no finding — it is one row on someone else's report."""
    (tmp_path / ".mcp.json").write_text("{not json at all")
    assert cli._mcp_project_entry(str(tmp_path)) is False
    (tmp_path / ".mcp.json").write_text('["a list, not an object"]')
    assert cli._mcp_project_entry(str(tmp_path)) is False


def test_the_mcp_wiring_nudge_and_its_quiet_switch_are_gone():
    """The retired nudge leaves no symbol behind for a later reader to re-wire.

    Its premise inverted: wiring is a one-time machine-global act now, so a nudge
    that cannot tell whether its advice was already taken would fire in every
    project forever. `MITOS_NO_MCP_HINT` existing at all was evidence it already
    read as a nag; `tests/test_env_routing.py`'s declared read set pins its
    absence from the environment side.
    """
    for symbol in ("_mcp_hint", "_mcp_wired", "_DECISION_LOOP_COMMANDS"):
        assert not hasattr(cli, symbol), f"cli.{symbol} survived its retirement"


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


def test_surface_decisions_mcp_description_names_compose():
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
    argv_extra, expected_in_msg, monkeypatch, capsys, workspace
):
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


# --- the intake leak: repeated --scope, and the retired --axiom prefix swallow ---
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


def test_extend_does_not_mutate_the_default_list_in_place():
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
    """`--axiom "prose"` RECORDS the prose — it never reaches the file reader.

    Inverted deliberately, from the stronger direction. ``allow_abbrev`` defaults to
    True on every subparser, so ``--axiom`` was an unambiguous prefix of
    ``--axiom-file`` and argparse handed a whole axiom to the file reader; the
    command then died through the outermost boundary as an ``[Errno 36] File name
    too long`` "Fatal Unexpected Error". Turning abbreviation off fixed the crash
    and left a wall — a bare usage banner and exit 2 — where every sibling flag had
    taught callers to look. This row asserted that wall; it now asserts the vector.

    The two original negatives are KEPT: they are the founding defect, and neither
    may reappear now that the flag parses for real.
    """
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "-p", workspace, "record", "--axiom", "A decision stated as prose.",
                         "--rejected", "r", "--slug", "s"])
    with patch("mitos.cli.cmd_record") as mock_record:
        main()
    assert mock_record.call_args.kwargs["axiom"] == "A decision stated as prose."
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "File name too long" not in combined
    assert "Traceback" not in combined


def test_abbreviation_is_off_on_every_subparser():
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


def test_abbreviated_option_is_rejected_by_the_grammar():
    """`--jso` no longer resolves to `--json`; `--axiom-fil` no longer reaches --axiom-file.

    The `record` probe was `--axiom`, which is now a declared option resolving
    exactly — so it had to be swapped rather than deleted, or the row would go green
    while testing nothing. `--axiom-fil` is an unambiguous prefix of `--axiom-file`
    and NOT a prefix of `--axiom`: measured, it is accepted under
    ``allow_abbrev=True`` and exits 2 under ``False``, so the row still bites.
    """
    from mitos.cli import _build_parser

    parser = _build_parser()
    for argv in (["list", "--jso"],
                 ["record", "--axiom-fil", "x", "--rejected", "r", "--slug", "s"]):
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


# --- --axiom: three sources, one predicate (B4) ---------------------------------
#
# `--axiom` is where every sibling flag has taught callers to look, and until this
# phase it was a bare usage banner and exit 2 — a wall where a vector belongs. It
# takes its OWN dest (`axiom_flag`): a shared dest with the positional makes
# "supplied both" undetectable (argument order silently decides the winner), which
# is the silently-keep-last defect 1a just fixed, reintroduced on the field that
# CONSTITUTES identity — a dropped axiom is a different node id, not a lost link.

def _record_actions_by_dest():
    """The `record` subparser's argparse actions keyed by dest — positional included.

    Keyed by dest rather than by option string so the POSITIONAL is reachable (it
    has none). Asserts run against ``action.help``, never ``format_help()``: the
    rendered help wraps at a terminal width pytest and the shell do not share, so a
    phrase assertion reds wherever the wrap lands inside it.
    """
    from test_cli_selector import _subparsers
    from mitos.cli import _build_parser

    return {a.dest: a for a in _subparsers(_build_parser())["record"]._actions}


_POSITIONAL_AXIOM = "An axiom typed as a positional."
_FLAG_AXIOM = "An axiom typed as a flag."
_FILE_AXIOM = "An axiom read from a file."

#: The eight axiom-source states. `("flag", "file")` is the invocation the shipped
#: chooser misclassified: it leaves the positional None, so a code picked off that
#: one dest answered `missing_axiom` to a caller who had supplied two.
_AXIOM_MATRIX = [
    ((), "missing_axiom"),
    (("positional",), _POSITIONAL_AXIOM),
    (("flag",), _FLAG_AXIOM),
    (("file",), _FILE_AXIOM),
    (("positional", "flag"), "ambiguous_axiom_source"),
    (("positional", "file"), "ambiguous_axiom_source"),
    (("flag", "file"), "ambiguous_axiom_source"),
    (("positional", "flag", "file"), "ambiguous_axiom_source"),
]


def _axiom_argv(sources, workspace, axiom_file):
    argv = ["mitos", "-p", workspace, "record"]
    if "positional" in sources:
        argv.append(_POSITIONAL_AXIOM)
    if "flag" in sources:
        argv += ["--axiom", _FLAG_AXIOM]
    if "file" in sources:
        argv += ["--axiom-file", str(axiom_file)]
    return argv + ["--rejected", "r", "--slug", "s"]


@pytest.fixture
def axiom_file(tmp_path):
    path = tmp_path / "axiom.txt"
    path.write_text(_FILE_AXIOM + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("sources, expected", _AXIOM_MATRIX,
                         ids=[".".join(s) or "none" for s, _ in _AXIOM_MATRIX])
def test_axiom_source_matrix_text_surface(sources, expected, monkeypatch, capsys,
                                          workspace, axiom_file):
    """All eight states: exactly one source records, none or several refuse with exit 2."""
    monkeypatch.setattr(sys, "argv", _axiom_argv(sources, workspace, axiom_file))
    refuses = expected in ("missing_axiom", "ambiguous_axiom_source")

    if refuses:
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "exactly one axiom source" in err
        # The message names all THREE sources — a refusal that still says "two"
        # sends a caller who typed --axiom looking for a flag it does not mention.
        # `--axiom` is matched with a negative lookahead: the shipped two-source
        # message contains it as a PREFIX of `--axiom-file`, so a bare substring
        # test passes over exactly the omission this asserts against.
        assert re.search(r"--axiom(?!-)", err), err
        assert "--axiom-file" in err, err
        assert "positional" in err, err
    else:
        with patch("mitos.cli.cmd_record") as mock_record:
            main()
        assert mock_record.call_args.kwargs["axiom"] == expected


@pytest.mark.parametrize("sources, expected", [(s, e) for s, e in _AXIOM_MATRIX
                                               if e.endswith("axiom_source")
                                               or e == "missing_axiom"],
                         ids=lambda v: ".".join(v) if isinstance(v, tuple) else str(v))
def test_axiom_source_refusals_carry_their_json_code(sources, expected, monkeypatch,
                                                     capsys, workspace, axiom_file):
    """Every refusal names its code on `--json`, chosen from the COUNT of sources.

    The shipped chooser was an exact discriminator at arity two and wrong at three:
    keyed on one dest, `--axiom X --axiom-file f` reported `missing_axiom` to a
    caller who had supplied two. One count now feeds the guard and the code alike,
    so a single expression cannot drift from the other.
    """
    import json
    monkeypatch.setattr(sys, "argv",
                        _axiom_argv(sources, workspace, axiom_file) + ["--json"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert json.loads(capsys.readouterr().out)["code"] == expected


def test_axiom_flag_reaches_the_record_decision_alias(monkeypatch, workspace):
    """`record` and `record_decision` are ONE parser object — one declaration covers both."""
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "-p", workspace, "record_decision",
                         "--axiom", _FLAG_AXIOM, "--rejected", "r", "--slug", "s"])
    with patch("mitos.cli.cmd_record") as mock_record:
        main()
    assert mock_record.call_args.kwargs["axiom"] == _FLAG_AXIOM


def test_axiom_flag_is_a_plain_store_and_does_not_accumulate():
    """`--axiom` is plain `store` — asserted off the parser, so B1's pattern reds here.

    1a converted nine relation flags in this same block to ``action="append"``, and
    a later pass reaching for that pattern would find `--axiom` one flag over. It is
    forbidden on this one: an accumulating spelling on the identity-constituting
    field mints a joined string into the canonical core, where set semantics have no
    meaning (an axiom is one sentence, not a set) and a merged value is a different
    node id.
    """
    action = _record_actions_by_dest()["axiom_flag"]
    assert type(action).__name__ == "_StoreAction", type(action).__name__
    assert action.nargs is None
    assert action.default is None
    # `metavar` or the usage banner leaks the internal dest as `[--axiom AXIOM_FLAG]`.
    assert action.metavar == "AXIOM"


def test_repeated_axiom_flag_is_last_wins_shipped_class_semantics(monkeypatch, workspace):
    """`--axiom a --axiom b` keeps `b`, silently — pinned, not endorsed.

    The whole single-value class behaves this way (`--rejected`, `--context`,
    `--slug`, `--axiom-file`), so singling `--axiom` out would ship an
    inconsistency; fixing the class is a different item with its own review. The
    three-source check counts SOURCES, not occurrences, so a repeated single flag
    is one source and still records.
    """
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "-p", workspace, "record", "--axiom", "first",
                         "--axiom", "second", "--rejected", "r", "--slug", "s"])
    with patch("mitos.cli.cmd_record") as mock_record:
        main()
    assert mock_record.call_args.kwargs["axiom"] == "second"


def test_the_three_axiom_help_strings_say_three(monkeypatch, capsys, workspace):
    """The positional's help, `--axiom-file`'s help and the refusal all say THREE.

    A sentence that ships saying "exactly one of the two" describes a grammar that
    no longer exists, on the surface a caller reads to find the third.
    """
    actions = _record_actions_by_dest()
    assert "three" in actions["axiom"].help
    assert "three" in actions["axiom_file"].help
    assert actions["axiom_flag"].help

    monkeypatch.setattr(sys, "argv",
                        ["mitos", "-p", workspace, "record", "--rejected", "r", "--slug", "s"])
    with pytest.raises(SystemExit):
        main()
    assert "three" in capsys.readouterr().err


def test_the_flag_and_the_positional_mint_the_same_node_id(monkeypatch, capsys, workspace):
    """MI-4/M2: `--axiom` is a parse-time alias into the canonical core, nothing more.

    Recorded through the positional, then through the flag under a DIFFERENT slug —
    identity is the content, so the second call resolves to the same node id and
    returns `exists`. That is the cheapest demonstration that the new declaration
    reaches the identity-constituting field byte-identically and mutates no existing
    node's core: a flag that mangled the sentence would mint a second node instead.
    """
    import json
    sentence = "Identity is the canonical core, and the flag is only a spelling."

    monkeypatch.setattr(sys, "argv",
                        ["mitos", "-p", workspace, "record", sentence,
                         "--rejected", "r", "--slug", "via-positional", "--json"])
    main()
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "created", first

    monkeypatch.setattr(sys, "argv",
                        ["mitos", "-p", workspace, "record", "--axiom", sentence,
                         "--rejected", "r", "--slug", "via-flag", "--json"])
    main()
    second = json.loads(capsys.readouterr().out)
    assert second["status"] == "exists", second
    assert second["id"] == first["id"]
