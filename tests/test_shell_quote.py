"""The recipe-quoting leaf: every value a printed `mitos …` recipe interpolates.

ROADMAP's census (2026-09-21) found 41 `-p {…!r}` sites and 6 `repr()` ones. `repr` is
a Python literal, not shell quoting: it breaks on a value holding both quote kinds or a
backslash, and every `-p '<name>'` breaks on a registered name that starts with `-`,
because argparse reads the unquoted word as an option. The round trip below is the
property — split the rendered fragment the way a shell would, parse it with mitos's own
parser, and get the value back.
"""

import shlex

import pytest

from mitos import cli
from mitos.shell_quote import shell_arg, shell_flag


AWKWARD = [
    "mitos",
    "my project",
    "/home/me/my projects/x",
    "-dash-led",
    "--double-dash",
    "it's",
    'say "hi"',
    """both ' and " kinds""",
    "back\\slash",
    "ąžuolas",
    "$HOME `x`",
]


@pytest.mark.parametrize("value", AWKWARD)
def test_a_selector_round_trips_through_a_shell_and_the_parser(value) -> None:
    for side in ("pre", "post"):
        fragment = shell_flag("-p", value)
        argv = (shlex.split(fragment) + ["list"] if side == "pre"
                else ["list"] + shlex.split(fragment))
        args = cli._build_parser().parse_args(argv)
        assert (args.project_pre if side == "pre" else args.project_post) == value


@pytest.mark.parametrize("value", AWKWARD)
def test_a_reconcile_entry_round_trips(value) -> None:
    argv = ["sync", "-p", "x"] + shlex.split(shell_flag("--reconcile-entry", value))
    assert cli._build_parser().parse_args(argv).reconcile_entry == [value]


def test_an_ordinary_value_renders_exactly_as_repr_did() -> None:
    """The shipped `-p 'demo'` form is unchanged for every ordinary name."""
    for value in ("demo", "my project", "/home/me/x", "ąžuolas"):
        assert shell_arg(value) == repr(value)
    assert shell_flag("-p", "demo") == "-p 'demo'"


def test_a_dash_led_value_binds_with_equals() -> None:
    assert shell_flag("-p", "-x") == "-p='-x'"


def test_a_control_character_falls_back_to_the_escaped_repr() -> None:
    """Neutralizing a control character is entry-018's axis, deferred; until then a
    value no shell recipe can carry keeps `repr`'s escaping rather than printing a
    raw newline or ESC into the terminal."""
    assert shell_arg("line\nbreak") == repr("line\nbreak")
    assert "\x1b" not in shell_arg("esc\x1b[31m")
