"""Adversarial test suite for the Mitos CLI entrypoint.

Verifies CLI argument parsing, help commands, and basic dry-runs for commands.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch, ANY
from mitos.cli import main

def test_cli_help_menu() -> None:
    """Verifies that the help menu is printed and exits cleanly with 0."""
    with patch.object(sys, "argv", ["mitos", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0


@patch("mitos.cli.cmd_init")
def test_cli_init_routing(mock_init: MagicMock) -> None:
    """Verifies that the 'init' command routes to the initialization controller."""
    with patch.object(sys, "argv", ["mitos", "init"]):
        main()
    assert mock_init.called


@patch("mitos.cli.cmd_sync")
def test_cli_sync_routing(mock_sync: MagicMock) -> None:
    """Verifies that 'sync' command parses flags and routes correctly."""
    with patch.object(sys, "argv", ["mitos", "sync", "--yes"]):
        main()
    mock_sync.assert_called_once()
    # Check that auto_accept is True
    args, kwargs = mock_sync.call_args
    assert kwargs["auto_accept"] is True


@patch("mitos.cli.cmd_capture")
def test_cli_capture_routing(mock_capture: MagicMock) -> None:
    """Verifies that 'capture' routes successfully with text argument."""
    with patch.object(sys, "argv", ["mitos", "capture", "Use SQLite WAL"]):
        main()
    mock_capture.assert_called_once()
    args, kwargs = mock_capture.call_args
    assert args[1] == "Use SQLite WAL"


@patch("mitos.cli.cmd_query")
def test_cli_query_routing(mock_query: MagicMock) -> None:
    """Verifies that semantic 'query' command routes successfully."""
    with patch.object(sys, "argv", ["mitos", "query", "cache strategy"]):
        main()
    mock_query.assert_called_once()
    args, kwargs = mock_query.call_args
    assert args[1] == "cache strategy"


@patch("mitos.cli.cmd_show")
def test_cli_show_routing(mock_show: MagicMock) -> None:
    """Verifies 'show' command queries slugs correctly."""
    with patch.object(sys, "argv", ["mitos", "show", "my-slug"]):
        main()
    mock_show.assert_called_once()
    args, kwargs = mock_show.call_args
    assert args[1] == "my-slug"


@patch("mitos.cli.cmd_list")
def test_cli_list_routing(mock_list: MagicMock) -> None:
    """Verifies 'list' routes with optional scope and state filters."""
    with patch.object(sys, "argv", ["mitos", "list", "--scope", "backend", "--state", "active"]):
        main()
    mock_list.assert_called_once_with(
        ANY,
        scope="backend",
        state_filter="active",
        as_json=False,
        brief=False
    )


@patch("mitos.cli.cmd_sync")
def test_cli_sync_embed_only_routing(mock_sync: MagicMock) -> None:
    """Verifies that 'sync --embed-only' routes correctly with embed_only=True."""
    with patch.object(sys, "argv", ["mitos", "sync", "--embed-only"]):
        main()
    mock_sync.assert_called_once()
    args, kwargs = mock_sync.call_args
    assert kwargs["embed_only"] is True


@patch("mitos.cli.cmd_query")
def test_cli_query_depth_routing(mock_query: MagicMock) -> None:
    """Verifies that 'query --depth' routes with the depth parameter."""
    with patch.object(sys, "argv", ["mitos", "query", "my claim", "--depth", "trace"]):
        main()
    mock_query.assert_called_once()
    args, kwargs = mock_query.call_args
    assert args[1] == "my claim"
    assert kwargs["depth"] == "trace"


@patch("mitos.cli.cmd_render")
def test_cli_render_format_routing(mock_render: MagicMock) -> None:
    """Verifies 'render --format' routes format to cmd_render."""
    with patch.object(sys, "argv", ["mitos", "render", "--format", "nygard"]):
        main()
    mock_render.assert_called_once()
    args, kwargs = mock_render.call_args
    assert kwargs["render_format"] == "nygard"


@patch("mitos.cli.cmd_serve")
def test_cli_serve_routing(mock_serve: MagicMock) -> None:
    """Verifies that 'serve' sub-command routes to cmd_serve."""
    with patch.object(sys, "argv", ["mitos", "serve"]):
        main()
    assert mock_serve.called


def test_cli_unexpected_error_exits_1() -> None:
    """Verifies that unexpected exceptions crash cleanly with exit code 1."""
    with patch("mitos.cli.cmd_init", side_effect=Exception("Unexpected boom!")):
        with patch.object(sys, "argv", ["mitos", "init"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
