"""L9: argparse's walls name the right form instead of only the wrong one.

Three shapes, each re-sighted across releases (AX ledger item 9): a global option
after the verb (`mitos surface -C /tmp x`), an unknown `--flag` for what is a
positional (`--query`), and a flag that belongs to a sibling verb (`show --brief`,
which argparse reported as a missing `ident` — the wrong fault, and once a false
"all three slugs MISSING" finding). Argparse's own line and exit 2 stay; one
teaching line is added under it. Pure parser rows: nothing here touches a workspace.
"""

import pytest

from mitos import cli


def _error(capsys, argv):
    with pytest.raises(SystemExit) as excinfo:
        cli._build_parser().parse_args(argv)
    assert excinfo.value.code == 2
    return capsys.readouterr().err


def test_a_global_option_after_the_verb_says_it_goes_before(capsys) -> None:
    err = _error(capsys, ["surface", "-p", "mitos", "-C", "/tmp", "x"])
    assert "unrecognized arguments: -C" in err
    assert "`-C` is a global option" in err
    assert "mitos -C <dir> surface" in err


def test_the_long_spelling_of_the_global_option_is_taught_too(capsys) -> None:
    err = _error(capsys, ["list", "-p", "mitos", "--directory=/tmp"])
    assert "`--directory` is a global option" in err


def test_a_positional_spelled_as_a_flag_names_the_positional(capsys) -> None:
    err = _error(capsys, ["surface", "-p", "mitos", "--query", "x"])
    assert "unrecognized arguments: --query" in err
    assert "`surface` takes its query as a positional argument" in err
    assert "mitos surface -p <project> <query>" in err


def test_a_near_miss_flag_gets_the_real_one(capsys) -> None:
    err = _error(capsys, ["list", "-p", "mitos", "--onelin"])
    assert "Did you mean `--oneline`?" in err


def test_a_sibling_verbs_flag_names_the_verbs_that_take_it(capsys) -> None:
    """`show --brief`: argparse said `required: ident`; the fault is the flag."""
    err = _error(capsys, ["show", "-p", "mitos", "--brief"])
    assert "`--brief` is not an option of `mitos show`" in err
    assert "mitos surface" in err and "mitos query" in err
    assert "required: ident" not in err


def test_a_sibling_flag_with_the_positional_present_is_taught_the_same(capsys) -> None:
    err = _error(capsys, ["show", "-p", "mitos", "--brief", "some-slug"])
    assert "`--brief` is not an option of `mitos show`" in err


def test_a_flag_no_verb_takes_points_at_the_verbs_help(capsys) -> None:
    err = _error(capsys, ["list", "-p", "mitos", "--zzzzqqq"])
    assert "`mitos list -h` lists its options" in err


def test_a_genuinely_missing_positional_keeps_argparses_own_message(capsys) -> None:
    """No unknown flag in sight: the required-argument error is the right fault."""
    err = _error(capsys, ["show", "-p", "mitos"])
    assert "the following arguments are required: ident" in err
    assert "is not an option" not in err


def test_a_well_formed_call_still_parses() -> None:
    args = cli._build_parser().parse_args(["-C", "/tmp", "surface", "-p", "mitos", "x"])
    assert args.command == "surface" and args.directory == "/tmp"
