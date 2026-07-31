"""T9 gate: the `-C`/`--directory` process-entry chdir (Phase 4a).

`-C`/`--directory` performs a single `os.chdir(args.directory)` at process entry,
before any env load, config construction, or arg-driven file open — git's `-C`
semantics, a *total* workspace retarget. These tests drive the real entry point
(`main()` with a monkeypatched `sys.argv`) so the chdir + the `main()` reorder are
exercised end-to-end, not via a unit shortcut.

CWD isolation (P10): every test that runs a `-C` invocation anchors a known launch
CWD with `monkeypatch.chdir(tmp_path)`, which auto-restores on teardown — `os.chdir`
mutates process-global CWD, so without it one test would pollute its siblings.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

from conftest import make_workspace
from mitos.cli import main, _enter_target_directory
from mitos.config import default_collection_name
from mitos.errors import MitosError


# ---------------------------------------------------------------------------
# Pure-function unit lane for the helper (fast, keyless).
# ---------------------------------------------------------------------------

def test_enter_target_directory_none_is_noop(tmp_path, monkeypatch) -> None:
    """A None directory (flag absent) leaves the CWD untouched."""
    monkeypatch.chdir(tmp_path)
    before = os.getcwd()
    _enter_target_directory(None)
    assert os.getcwd() == before


def test_enter_target_directory_absent_path_raises_mitos_error(tmp_path, monkeypatch) -> None:
    """A non-existent target raises MitosError (clean P3 error, never raw OSError)."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(MitosError, match="directory not found"):
        _enter_target_directory(str(tmp_path / "does-not-exist"))
    assert os.getcwd() == str(tmp_path)  # no chdir happened on the error path


def test_enter_target_directory_file_path_raises_mitos_error(tmp_path, monkeypatch) -> None:
    """A path that is a file (not a directory) is the same clean MitosError."""
    monkeypatch.chdir(tmp_path)
    a_file = tmp_path / "afile.txt"
    a_file.write_text("x", encoding="utf-8")
    with pytest.raises(MitosError, match="directory not found"):
        _enter_target_directory(str(a_file))


def test_enter_target_directory_real_dir_chdirs(tmp_path, monkeypatch) -> None:
    """A real directory target performs the chdir."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "ws"
    target.mkdir()
    _enter_target_directory(str(target))
    assert os.getcwd() == os.path.realpath(str(target))


# ---------------------------------------------------------------------------
# Per-verb retarget (the core pin): config/collection track `-C`.
# ---------------------------------------------------------------------------

@patch("mitos.cli.cmd_init")
def test_directory_retargets_config_collection(mock_init, tmp_path, monkeypatch) -> None:
    """`mitos -C /ws init` builds its MitosConfig against /ws — collection included."""
    monkeypatch.chdir(tmp_path)          # launch CWD ≠ target
    ws = tmp_path / "ws"
    ws.mkdir()
    with patch.object(sys, "argv", ["mitos", "-C", str(ws), "init"]):
        main()
    mock_init.assert_called_once()
    config = mock_init.call_args.args[0]
    # workspace_dir is abspath('.') computed post-chdir == the current CWD (symlink-safe).
    assert config.workspace_dir == os.getcwd()
    assert config.qdrant_collection == default_collection_name(config.workspace_dir)


@patch("mitos.cli.cmd_list")
def test_directory_reaches_dispatch_and_chdirs(mock_list, tmp_path, monkeypatch) -> None:
    """`-C` is consumed by the parent parser (not the subparser) and the chdir lands.

    `-p .` rather than an absolute path, deliberately and only here: the row's whole
    claim is that the chdir happened before resolution, so a selector that resolves
    *through* the new cwd is the assertion, not a shortcut.
    """
    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "ws"
    make_workspace(ws)
    with patch.object(sys, "argv", ["mitos", "-C", str(ws), "-p", ".", "list"]):
        main()
    mock_list.assert_called_once()
    assert os.getcwd() == os.path.realpath(str(ws))


def test_init_under_directory_creates_mitos_in_target(tmp_path, monkeypatch) -> None:
    """A real `init` under `-C` scaffolds `.mitos` in the target, not the launch CWD."""
    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    with patch.object(sys, "argv", ["mitos", "-C", str(ws), "init"]):
        main()
    assert (ws / ".mitos").is_dir()
    assert not (tmp_path / ".mitos").exists()


# ---------------------------------------------------------------------------
# The R3 partial-retarget pins: set-key writes /ws/.env, and the project .env
# LOAD moved after the chdir (read from /ws/.env).
# ---------------------------------------------------------------------------

def test_set_key_under_directory_writes_target_env(tmp_path, monkeypatch, capsys) -> None:
    """`mitos -C /ws set-key …` writes /ws/.env — NOT the launch CWD's .env (R3 canary)."""
    monkeypatch.chdir(tmp_path)
    ws = make_workspace(tmp_path / "ws")
    with patch.object(sys, "argv",
                      ["mitos", "-C", ws, "-p", ".", "set-key", "SECRET123",
                       "--name", "GEMINI_API_KEY"]):
        main()
    assert os.path.exists(os.path.join(ws, ".env"))
    with open(os.path.join(ws, ".env"), encoding="utf-8") as f:
        assert "SECRET123" in f.read()
    assert not (tmp_path / ".env").exists()


def test_project_env_loaded_from_target_under_directory(tmp_path, monkeypatch) -> None:
    """The project `.env` LOAD moved after the chdir: keys come from /ws/.env (R3).

    If `load_dotenv_file()` had stayed at its pre-parse position, `-C` would
    retarget the graph/collection but read keys from the launch CWD — the
    partial-retarget trap. A custom key name (loaded only from the target `.env`)
    pins that the load now retargets.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MITOS_TEST_DIRKEY", raising=False)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".env").write_text("MITOS_TEST_DIRKEY=from-target\n", encoding="utf-8")
    # A same-named key in the LAUNCH CWD's .env must lose — proving which dir is read.
    (tmp_path / ".env").write_text("MITOS_TEST_DIRKEY=from-launch\n", encoding="utf-8")
    with patch("mitos.cli.cmd_init"):  # keep it light; the env load runs before dispatch
        with patch.object(sys, "argv", ["mitos", "-C", str(ws), "init"]):
            main()
    assert os.environ.get("MITOS_TEST_DIRKEY") == "from-target"


def test_relative_rejected_file_retargets(tmp_path, monkeypatch) -> None:
    """A relative `--rejected-file ./r.txt` opens against the post-chdir CWD (/ws).

    This is `-C`'s surviving role, and the pair `-C /ws -p .` is what the migration
    had to become: **a selector moves the workspace, it does not move where a file
    argument opens.** Rewriting the row as `-p /ws record … --rejected-file ./r.txt`
    would retarget the corpus and still open `./r.txt` against the launch directory
    — a missing file, or worse, a different one.
    """
    monkeypatch.chdir(tmp_path)
    ws = make_workspace(tmp_path / "ws")
    (tmp_path / "ws" / "r.txt").write_text("the rejected alternative prose",
                                           encoding="utf-8")
    with patch("mitos.cli.cmd_record") as mock_record:
        with patch.object(sys, "argv",
                          ["mitos", "-C", ws, "-p", ".", "record", "my axiom",
                           "--rejected-file", "./r.txt", "--slug", "x"]):
            main()
    mock_record.assert_called_once()
    assert mock_record.call_args.kwargs["rejected"] == "the rejected alternative prose"


def test_a_selector_does_not_move_where_a_file_argument_opens(tmp_path, monkeypatch) -> None:
    """G2b's other half, stated as its own row rather than left in a comment.

    Same workspace, no `-C`: `-p /ws` retargets the corpus while `./r.txt` still
    opens against the launch directory. The two files carry different prose, so a
    build that resolved the file against the selector would read the wrong one and
    the assertion would name which.
    """
    monkeypatch.chdir(tmp_path)
    ws = make_workspace(tmp_path / "ws")
    (tmp_path / "ws" / "r.txt").write_text("the workspace copy", encoding="utf-8")
    (tmp_path / "r.txt").write_text("the launch-directory copy", encoding="utf-8")
    with patch("mitos.cli.cmd_record") as mock_record:
        with patch.object(sys, "argv",
                          ["mitos", "-p", ws, "record", "my axiom",
                           "--rejected-file", "./r.txt", "--slug", "x"]):
            main()
    assert mock_record.call_args.args[0].workspace_dir == ws
    assert mock_record.call_args.kwargs["rejected"] == "the launch-directory copy"


# ---------------------------------------------------------------------------
# serve still honours -C — and since 5b that is all -C does for it.
# ---------------------------------------------------------------------------

def test_serve_binds_target(tmp_path, monkeypatch) -> None:
    """`mitos -C /ws serve` moves the process to /ws (no real mcp.run).

    The half this row keeps is the ADR
    `serve-honors-dash-c-via-entry-chdir-not-scoped-out`: `-C` reaches `serve`
    through the process-entry chdir like every other verb, and something must
    still pin that.

    The half it drops is the claim that the launch directory *binds* anything.
    It used to read `MitosConfig().qdrant_collection` inside the fake handler and
    assert the collection derived from the launch dir — true while `_target_config`
    fell back to the cwd, and exactly what 5b removes. A server launched here now
    answers for whichever project each call names, which
    `test_the_launch_directory_binds_nothing` states over a real process. Dropping
    the assertion also drops the zero-arg `MitosConfig()` it needed — one of the
    five 5d must remove, gone here for free.
    """
    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    captured = {}

    def _fake_serve() -> None:
        captured["cwd"] = os.getcwd()

    with patch("mitos.cli.cmd_serve", side_effect=_fake_serve):
        with patch.object(sys, "argv", ["mitos", "-C", str(ws), "serve"]):
            main()
    assert captured["cwd"] == os.path.realpath(str(ws))


# ---------------------------------------------------------------------------
# Positional-wins for status/agent-block.
# ---------------------------------------------------------------------------

def test_status_positional_wins_over_directory(tmp_path, monkeypatch) -> None:
    """`status /path -C /other` reports on /path — the explicit positional wins.

    The positional is a *selector* from phase 3b on, so the target must be a real
    workspace (``.mitos/config.toml`` + ``decisions.md``) or it lands on the
    targeting error before the mocked handler is reached — and the argument the
    handler receives is the canonical root, so it is asserted against
    ``os.path.realpath`` rather than the raw fixture path (the two agree on this
    machine's temp root, which is exactly why the raw spelling would be a
    machine-dependent pin).
    """
    monkeypatch.chdir(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    target = tmp_path / "target"
    (target / ".mitos").mkdir(parents=True)
    (target / ".mitos" / "config.toml").write_text("", encoding="utf-8")
    (target / "decisions.md").write_text("# Decisions\n", encoding="utf-8")
    with patch("mitos.cli.cmd_status", return_value=0) as mock_status:
        with patch.object(sys, "argv",
                          ["mitos", "-C", str(other), "status", str(target)]):
            with pytest.raises(SystemExit) as exc:
                main()
    assert exc.value.code == 0
    assert mock_status.call_args.args[0] == os.path.realpath(str(target))


def test_status_no_positional_is_the_global_overview_not_the_directory(
        tmp_path, monkeypatch) -> None:
    """Inverted at 5a: `-C /ws status` no longer reports on /ws.

    It used to — the chdir landed and `status` resolved the working directory. The
    zero-arg form is now the machine-wide overview, so `-C` has nothing to aim: the
    deep report is never reached at all. Its recovery is the `-p` form below, where
    `-C` still decides what `.` means.
    """
    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    with patch("mitos.cli.cmd_status", return_value=0) as mock_status:
        with patch("mitos.cli.cmd_status_overview", return_value=0) as mock_overview:
            with patch.object(sys, "argv", ["mitos", "-C", str(ws), "status"]):
                with pytest.raises(SystemExit) as exc:
                    main()
    assert exc.value.code == 0
    mock_status.assert_not_called()
    mock_overview.assert_called_once()


def test_status_under_a_directory_still_targets_it_through_a_relative_selector(
        tmp_path, monkeypatch) -> None:
    """`-C /ws -p . status` — the recovery, and the ordering contract for free.

    `.` is absolutized **after** the chdir, so it means /ws rather than the launch
    directory. Canonicalizing before the chdir would change what a relative selector
    means with every absolute-path row still green.
    """
    monkeypatch.chdir(tmp_path)
    ws = make_workspace(tmp_path / "ws")
    with patch("mitos.cli.cmd_status", return_value=0) as mock_status:
        with patch.object(sys, "argv", ["mitos", "-C", ws, "-p", ".", "status"]):
            with pytest.raises(SystemExit):
                main()
    assert mock_status.call_args.args[0] == os.path.realpath(ws)


# ---------------------------------------------------------------------------
# Absent/bad target → one clean error line, exit 1, no traceback.
# ---------------------------------------------------------------------------

def test_absent_directory_clean_error(tmp_path, monkeypatch, capsys) -> None:
    """`mitos -C /nonexistent <verb>` → `Error: directory not found: …`, exit 1, no traceback."""
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "nonexistent"
    with patch.object(sys, "argv", ["mitos", "-C", str(missing), "list"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "Error:" in captured.err
    assert "directory not found" in captured.err


def test_directory_target_is_a_file_clean_error(tmp_path, monkeypatch, capsys) -> None:
    """A `-C` target that is a file is the same clean MitosError, not a NotADirectoryError."""
    monkeypatch.chdir(tmp_path)
    a_file = tmp_path / "afile"
    a_file.write_text("x", encoding="utf-8")
    with patch.object(sys, "argv", ["mitos", "-C", str(a_file), "list"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "directory not found" in captured.err


# ---------------------------------------------------------------------------
# Byte-identity / no-flag path untouched, and CWD-isolation contract (P10).
# ---------------------------------------------------------------------------

@patch("mitos.cli.cmd_list")
def test_no_directory_flag_does_not_chdir(mock_list, tmp_path, monkeypatch) -> None:
    """A flagless invocation never chdirs — the added code is gated on args.directory."""
    monkeypatch.chdir(tmp_path)
    ws = make_workspace(tmp_path / "ws")
    with patch.object(sys, "argv", ["mitos", "-p", ws, "list"]):
        main()
    mock_list.assert_called_once()
    assert os.getcwd() == str(tmp_path)  # launch CWD unchanged


def test_directory_run_mutates_then_isolation_restores_cwd(tmp_path, monkeypatch) -> None:
    """A `-C` run mutates process CWD to the target (the reason P10 isolation matters).

    `monkeypatch.chdir(tmp_path)` auto-restores the real CWD on teardown, so this
    mutation cannot leak into sibling tests — the documented isolation contract.
    """
    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    with patch("mitos.cli.cmd_init"):
        with patch.object(sys, "argv", ["mitos", "-C", str(ws), "init"]):
            main()
    assert os.getcwd() == os.path.realpath(str(ws))
