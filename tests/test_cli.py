"""Adversarial test suite for the Mitos CLI entrypoint.

Verifies CLI argument parsing, help commands, and basic dry-runs for commands.
"""

import sys
import tomllib
import pytest
from unittest.mock import MagicMock, patch, ANY
from mitos.cli import main
from mitos.config import toml_scalar

def test_cli_help_menu(workspace) -> None:
    """Verifies that the help menu is printed and exits cleanly with 0."""
    with patch.object(sys, "argv", ["mitos", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0


@patch("mitos.cli.cmd_init")
def test_cli_init_routing(mock_init: MagicMock, workspace) -> None:
    """Verifies that the 'init' command routes to the initialization controller."""
    with patch.object(sys, "argv", ["mitos", "init"]):
        main()
    assert mock_init.called


@patch("mitos.cli.cmd_sync")
def test_cli_sync_routing(mock_sync: MagicMock, workspace) -> None:
    """Verifies that 'sync' command parses flags and routes correctly."""
    with patch.object(sys, "argv", ["mitos", "-p", workspace, "sync", "--yes"]):
        main()
    mock_sync.assert_called_once()
    # Check that auto_accept is True
    args, kwargs = mock_sync.call_args
    assert kwargs["auto_accept"] is True


@patch("mitos.cli.cmd_capture")
def test_cli_capture_routing(mock_capture: MagicMock, workspace) -> None:
    """Verifies that 'capture' routes successfully with text argument."""
    with patch.object(sys, "argv", ["mitos", "-p", workspace, "capture", "Use SQLite WAL"]):
        main()
    mock_capture.assert_called_once()
    args, kwargs = mock_capture.call_args
    assert args[1] == "Use SQLite WAL"


@patch("mitos.cli.cmd_query")
def test_cli_query_routing(mock_query: MagicMock, workspace) -> None:
    """Verifies that semantic 'query' command routes successfully."""
    with patch.object(sys, "argv", ["mitos", "-p", workspace, "query", "cache strategy"]):
        main()
    mock_query.assert_called_once()
    args, kwargs = mock_query.call_args
    assert args[1] == "cache strategy"


@patch("mitos.cli.cmd_show")
def test_cli_show_routing(mock_show: MagicMock, workspace) -> None:
    """Verifies 'show' command queries slugs correctly."""
    with patch.object(sys, "argv", ["mitos", "-p", workspace, "show", "my-slug"]):
        main()
    mock_show.assert_called_once()
    args, kwargs = mock_show.call_args
    assert args[1] == "my-slug"


@patch("mitos.cli.cmd_list")
def test_cli_list_routing(mock_list: MagicMock, workspace) -> None:
    """Verifies 'list' routes with optional scope and state filters."""
    with patch.object(sys, "argv", ["mitos", "-p", workspace, "list", "--scope", "backend", "--state", "active"]):
        main()
    mock_list.assert_called_once_with(
        ANY,
        scope="backend",
        state_filter="active",
        as_json=False,
        brief=False,
        oneline=False
    )


@patch("mitos.cli.cmd_sync")
def test_cli_sync_embed_only_routing(mock_sync: MagicMock, workspace) -> None:
    """Verifies that 'sync --embed-only' routes correctly with embed_only=True."""
    with patch.object(sys, "argv", ["mitos", "-p", workspace, "sync", "--embed-only"]):
        main()
    mock_sync.assert_called_once()
    args, kwargs = mock_sync.call_args
    assert kwargs["embed_only"] is True


@patch("mitos.cli.cmd_query")
def test_cli_query_depth_routing(mock_query: MagicMock, workspace) -> None:
    """Verifies that 'query --depth' routes with the depth parameter."""
    with patch.object(sys, "argv", ["mitos", "-p", workspace, "query", "my claim", "--depth", "trace"]):
        main()
    mock_query.assert_called_once()
    args, kwargs = mock_query.call_args
    assert args[1] == "my claim"
    assert kwargs["depth"] == "trace"


@patch("mitos.cli.cmd_render")
def test_cli_render_format_routing(mock_render: MagicMock, workspace) -> None:
    """Verifies 'render --format' routes format to cmd_render."""
    with patch.object(sys, "argv", ["mitos", "-p", workspace, "render", "--format", "nygard"]):
        main()
    mock_render.assert_called_once()
    args, kwargs = mock_render.call_args
    assert kwargs["render_format"] == "nygard"


@patch("mitos.cli.cmd_serve")
def test_cli_serve_routing(mock_serve: MagicMock, workspace) -> None:
    """Verifies that 'serve' sub-command routes to cmd_serve."""
    with patch.object(sys, "argv", ["mitos", "serve"]):
        main()
    assert mock_serve.called


def test_cli_unexpected_error_exits_1(workspace) -> None:
    """Verifies that unexpected exceptions crash cleanly with exit code 1."""
    with patch("mitos.cli.cmd_init", side_effect=Exception("Unexpected boom!")):
        with patch.object(sys, "argv", ["mitos", "init"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1


def test_cli_malformed_config_exits_clean_no_traceback(
    tmp_path, capsys
) -> None:
    """A malformed `.mitos/config.toml` exits 1 with a one-line error, no traceback.

    The strict config loader (6a) raises a `ConfigError` at `MitosConfig()`
    construction. Because `main()` now builds the config INSIDE its `try:`, the
    `except MitosError` boundary renders it as a clean `Error: …` line rather than
    dumping a raw Python traceback for every command (PLANNING_NOTES Lesson 45 —
    the Letterbox `ConfigError`→traceback trap). Since 5a the config is built from
    the **named** project, so the row names the malformed workspace rather than
    standing in it — the boundary is the same one either way.

    The workspace is a full validity triple: `resolve_project` runs first and
    refuses a directory with no `decisions.md` for its own reason, which would
    prove the wrong thing.
    """
    mitos_dir = tmp_path / ".mitos"
    mitos_dir.mkdir()
    # Unterminated string mid-file → tomllib raises → strict loader raises ConfigError.
    (mitos_dir / "config.toml").write_text(
        'rotation_mode = "archive\nqdrant_url = "x"\n', encoding="utf-8"
    )
    (tmp_path / "decisions.md").write_text("# Decisions\n", encoding="utf-8")

    with patch.object(sys, "argv", ["mitos", "-p", str(tmp_path), "list"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1

    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "Error:" in captured.err
    assert "config" in captured.err.lower()


# ---------------------------------------------------------------------------
# toml_scalar — the shared config/registry serializer (config.py). It moved out of
# cli.py when the registry became its second consumer: the registry leaf cannot
# import cli (cli imports the registry), and config.py already owns the TOML read
# side. The four assertions below are UNCHANGED across that move and the widening
# — they are the byte-compatibility gate for every config.toml line ever seeded.
# ---------------------------------------------------------------------------

def test_toml_scalar_serializes_bool_as_lowercase_literal(workspace) -> None:
    """A bool serializes to native TOML ``true``/``false`` (not ``1``/``0``, not raise).

    ``bool`` subclasses ``int``, so the bool branch must precede the int branch;
    an int-first order would emit ``True`` as ``1``. Regression-pins that ordering.
    """
    assert toml_scalar(True) == "true"
    assert toml_scalar(False) == "false"


def test_toml_scalar_still_serializes_int_and_str(workspace) -> None:
    """The bool branch doesn't disturb the existing int/str scalars."""
    assert toml_scalar(50) == "50"
    assert toml_scalar("archive") == '"archive"'


@pytest.mark.parametrize(
    "value, why",
    [
        ("/x/a\\tb", "a backslash: a basic string would re-read `\\t` as a TAB"),
        ('quote"bearing', "a `\"`: the pre-widening serializer raised on this outright"),
        ("both'and\"quotes", "both quote kinds: neither literal form is available"),
        ("line\nbreak", "a newline: must escape rather than emit a raw break"),
        ("ctrl\x01char", "a control character: illegal raw in either string form"),
        ("ąžuolas", "non-ASCII: legal unescaped, and must not be mangled (P9)"),
        ("plain", "the clean case still takes the shipped basic form"),
    ],
)
def test_toml_scalar_round_trips_the_widened_domain(value, why, workspace) -> None:
    """Every value shape the registry can carry survives a serialize→parse cycle.

    The registry's domain is wider than the config schema's: project names can hold
    a quote and workspace paths can hold a backslash. Both are silent-corruption
    hazards rather than exotic ones — a path written into a TOML *basic* string has
    its escapes interpreted on the next read, so ``/x/a\\tb`` comes back carrying a
    TAB and points at a directory that does not exist.
    """
    assert tomllib.loads(f"k = {toml_scalar(value)}")["k"] == value, why


@pytest.mark.parametrize(
    "key",
    ["plain", "dotted.name", "ąžuolas", 'quote"bearing', "back\\slash"],
)
def test_toml_scalar_also_serves_as_a_quoted_key(key, workspace) -> None:
    """The same serializer quotes registry KEYS, which is why names are never bare.

    A bare dotted key parses as a nested table (``example.com`` — an ordinary
    directory name), and a bare non-ASCII key does not parse at all, which would
    make the whole registry unreadable rather than merely mis-read.
    """
    assert list(tomllib.loads(f'{toml_scalar(key)} = "v"')) == [key]


def test_toml_scalar_refuses_a_non_scalar_loudly(workspace) -> None:
    """A value with no TOML scalar form raises ``TypeError``, never emits a guess.

    Callers at a user-facing boundary convert this into their own calm error; the
    serializer itself stays a pure function that refuses rather than improvises.
    """
    with pytest.raises(TypeError):
        toml_scalar({"not": "a scalar"})
