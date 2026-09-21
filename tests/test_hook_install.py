"""Phase 3e: ``mitos hook-install`` — the repository's own git directory, two markers.

The first file mitos writes outside its own workspace. Two failures are invisible
in a green suite unless a row provokes them: the destructive one (overwriting
someone's hook) and the silent one (a hook that can never fail). The install
matrix runs real ``git`` under isolation (``isolate_git`` from
``test_process_fence``: machine config masked, discovery capped at ``tmp_path``);
the script and block rows run the rendered shell under real ``/bin/sh`` with a
stub standing in for ``mitos``. The proof with the real binary is 3h's.

Every in-process install pins ``sys.argv[0]`` to an absolute stub: under pytest
the running build is pytest's, and a hook baked to pytest's path would pass here
and guard nothing anywhere.
"""

import hashlib
import os
import shlex
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

from mitos import cli, routing
from mitos.cli import cmd_init
from mitos.commit_gate import (
    HOOK_ABSENT, HOOK_BLOCK_EXIT, HOOK_BLOCK_MARKER, HOOK_FILE_KINDS, HOOK_FILE_MARKER,
    HOOK_FOREIGN, HOOK_FOREIGN_WITH_BLOCK, HOOK_OURS, HOOK_UNREADABLE, HookFileState,
    classify_hook_file, hook_script_workspace, render_hook_block, render_hook_script,
    unsafe_shell_path,
)
from mitos.config import MitosConfig
from test_cli_selector import _subparsers
from test_process_fence import (
    _needs_git, add_linked_worktree, isolate_git, make_scratch_repo,
)

_SH = "/bin/sh"


@pytest.fixture(autouse=True)
def _git_isolation(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    isolate_git(monkeypatch, str(tmp_path))
    # pytest's cwd is the checkout. A regression (or a constraint-10 break) that
    # asks git from the cwd instead of the workspace would otherwise install into
    # the developer's own `.git/hooks`; here it lands in a directory git cannot see.
    monkeypatch.chdir(tmp_path)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_STUB = """#!/bin/sh
printf '%s\\0' "$@" > "$MITOS_STUB_LOG"
if [ "${MITOS_STUB_RC:-0}" = signal ]; then
  kill -TERM $$
fi
exit "${MITOS_STUB_RC:-0}"
"""


def _stub(directory, name: str = "mitos") -> str:
    """An executable that logs its argv (NUL-separated) and exits ``$MITOS_STUB_RC``."""
    os.makedirs(str(directory), exist_ok=True)
    path = os.path.join(str(directory), name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_STUB)
    os.chmod(path, 0o755)
    return path


def _stub_argv(log: str) -> Optional[List[str]]:
    if not os.path.exists(log):
        return None
    with open(log, "rb") as f:
        data = f.read()
    return [part.decode("utf-8") for part in data.split(b"\0")[:-1]]


def _run_sh(argv: List[str], *, path_dirs: List[str], log: str, rc: str = "0",
            cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    """Runs ``argv`` with a PATH of exactly ``path_dirs`` (none holds a real mitos)."""
    env = {k: v for k, v in os.environ.items() if k != "PATH"}
    env["PATH"] = os.pathsep.join(path_dirs)
    env["MITOS_STUB_LOG"] = log
    env["MITOS_STUB_RC"] = rc
    return subprocess.run(argv, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=30)


def _workspace(directory) -> str:
    """Initialises a real workspace at ``directory``; returns its realpath."""
    os.makedirs(str(directory), exist_ok=True)
    cmd_init(MitosConfig(str(directory)))
    return os.path.realpath(str(directory))


def _install(workspace: str, argv0: str, capsys: Any) -> Tuple[int, str, str]:
    """``mitos hook-install -p <workspace>`` through real ``main()``, ``argv[0]`` pinned."""
    capsys.readouterr()
    with patch.object(sys, "argv", [argv0, "hook-install", "-p", workspace]):
        try:
            cli.main()
            code = 0
        except SystemExit as exc:
            code = exc.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _main(argv: List[str], argv0: str = "mitos") -> int:
    with patch.object(sys, "argv", [argv0] + list(argv)):
        try:
            cli.main()
        except SystemExit as exc:
            return exc.code
    return 0


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _tree(directory: str) -> Dict[str, str]:
    """Every file under ``directory`` with its sha256 (empty when absent)."""
    found: Dict[str, str] = {}
    for root, _dirs, files in os.walk(directory):
        for name in files:
            path = os.path.join(root, name)
            if os.path.isfile(path) and not os.path.islink(path):
                found[os.path.relpath(path, directory)] = _sha(path)
    return found


def _hook_of(repo: str) -> str:
    return os.path.join(repo, ".git", "hooks", "pre-commit")


def _repo_with_workspace(tmp_path, sub: str = "") -> Tuple[str, str]:
    repo = make_scratch_repo(str(tmp_path / "repo"))
    ws = _workspace(os.path.join(repo, sub) if sub else repo)
    return repo, ws


def _printed_selector(block: str) -> str:
    """The ``-p`` value on the block's first ``hook-run`` line, as ``sh`` splits it."""
    for line in block.splitlines():
        if "hook-run -p" in line:
            tokens = shlex.split(line)
            return tokens[tokens.index("-p") + 1]
    raise AssertionError(f"no hook-run line in:\n{block}")


def _block_of(out: str) -> str:
    """The pasted block within stdout (after the echo line)."""
    start = out.index(HOOK_BLOCK_MARKER)
    return out[start:]


# --------------------------------------------------------------------------- #
# The install matrix
# --------------------------------------------------------------------------- #

@_needs_git
def test_a_fresh_install_writes_an_executable_parseable_hook(tmp_path, capsys) -> None:
    """Criterion 1."""
    repo, ws = _repo_with_workspace(tmp_path)
    exe = _stub(tmp_path / "bin")
    code, out, err = _install(ws, exe, capsys)
    assert code == 0, err
    hook = _hook_of(repo)
    assert os.access(hook, os.X_OK)
    with open(hook, encoding="utf-8") as f:
        text = f.read()
    assert text.split("\n")[:2] == ["#!/bin/sh", HOOK_FILE_MARKER]
    assert subprocess.run([_SH, "-n", hook]).returncode == 0
    assert f"Wrote the commit gate hook to {hook}." in out
    assert f"It gates the workspace at {ws}." in out
    assert f"It runs {exe}." in out
    assert f"To remove the gate, delete {hook}." in out
    assert hook_script_workspace(text) == ws
    assert err == ""


@_needs_git
def test_an_absent_hooks_directory_is_created_under_the_common_dir(tmp_path, capsys) -> None:
    """Criterion 2: ``git init`` templates make one, so the row removes it."""
    repo, ws = _repo_with_workspace(tmp_path)
    shutil.rmtree(os.path.join(repo, ".git", "hooks"))
    code, _out, err = _install(ws, _stub(tmp_path / "bin"), capsys)
    assert code == 0, err
    assert os.access(_hook_of(repo), os.X_OK)


@_needs_git
def test_a_reinstall_for_the_same_workspace_overwrites_quietly(tmp_path, capsys) -> None:
    """Criterion 3, same workspace; a hand ``chmod -x`` does not survive either."""
    repo, ws = _repo_with_workspace(tmp_path)
    exe = _stub(tmp_path / "bin")
    assert _install(ws, exe, capsys)[0] == 0
    os.chmod(_hook_of(repo), 0o644)
    code, out, _err = _install(ws, exe, capsys)
    assert code == 0
    assert "Previously served" not in out
    assert os.access(_hook_of(repo), os.X_OK)


@_needs_git
def test_a_reinstall_for_another_workspace_names_the_previous_one(tmp_path, capsys) -> None:
    """Criterion 3, other workspace: one hook serves one workspace (ADR)."""
    repo = make_scratch_repo(str(tmp_path / "repo"))
    first = _workspace(os.path.join(repo, "a"))
    second = _workspace(os.path.join(repo, "b"))
    exe = _stub(tmp_path / "bin")
    assert _install(first, exe, capsys)[0] == 0
    code, out, _err = _install(second, exe, capsys)
    assert code == 0
    assert f"Previously served {first}." in out
    with open(_hook_of(repo), encoding="utf-8") as f:
        assert hook_script_workspace(f.read()) == second


@_needs_git
def test_an_own_hook_with_an_unreadable_workspace_is_overwritten_and_said(tmp_path,
                                                                           capsys) -> None:
    """The serialization contract: marker on line 2, no parseable workspace → ``ours``."""
    repo, ws = _repo_with_workspace(tmp_path)
    hook = _hook_of(repo)
    with open(hook, "w", encoding="utf-8") as f:
        f.write(f"#!/bin/sh\n{HOOK_FILE_MARKER}\nexit 0\n")
    assert classify_hook_file(hook) == HookFileState(HOOK_OURS, None)
    code, out, _err = _install(ws, _stub(tmp_path / "bin"), capsys)
    assert code == 0
    assert "It replaces a mitos hook whose workspace could not be read." in out


_FOREIGN = "#!/bin/sh\n# someone's own checks\nnpm test\n"


@_needs_git
def test_a_foreign_hook_is_left_byte_identical_and_the_block_printed(tmp_path,
                                                                     capsys) -> None:
    """Criterion 4."""
    repo, ws = _repo_with_workspace(tmp_path)
    hook = _hook_of(repo)
    with open(hook, "w", encoding="utf-8") as f:
        f.write(_FOREIGN)
    before = _sha(hook)
    exe = _stub(tmp_path / "bin")
    code, out, err = _install(ws, exe, capsys)
    assert code == 1
    assert _sha(hook) == before
    assert "mitos did not write it" in err
    block = _block_of(out)
    assert HOOK_FILE_MARKER not in block
    assert _printed_selector(block) == ws
    assert exe in block


@_needs_git
def test_a_foreign_hook_already_carrying_the_block_is_not_reprinted(tmp_path,
                                                                    capsys) -> None:
    """Criterion 5."""
    repo, ws = _repo_with_workspace(tmp_path)
    hook = _hook_of(repo)
    block = render_hook_block(command=["/opt/mitos"], selector=ws, guard_dir=None)
    with open(hook, "w", encoding="utf-8") as f:
        f.write(_FOREIGN + block)
    before = _sha(hook)
    code, out, err = _install(ws, _stub(tmp_path / "bin"), capsys)
    assert code == 1
    assert _sha(hook) == before
    assert "already carries a mitos commit gate block" in err
    assert HOOK_BLOCK_MARKER not in out


@_needs_git
def test_the_whole_file_marker_below_line_two_is_foreign(tmp_path, capsys) -> None:
    """Criterion 6: a mitos hook pasted into the middle of someone's script."""
    repo, ws = _repo_with_workspace(tmp_path)
    hook = _hook_of(repo)
    pasted = render_hook_script(command=["/opt/mitos"], workspace=ws, hook_file=hook)
    with open(hook, "w", encoding="utf-8") as f:
        f.write(_FOREIGN + pasted)
    before = _sha(hook)
    assert classify_hook_file(hook).kind == HOOK_FOREIGN
    code, _out, _err = _install(ws, _stub(tmp_path / "bin"), capsys)
    assert code == 1
    assert _sha(hook) == before


@_needs_git
@pytest.mark.parametrize("dangling", [False, True], ids=["to-a-marker-file", "dangling"])
def test_a_symlinked_pre_commit_is_refused_and_never_written_through(tmp_path, capsys,
                                                                     dangling) -> None:
    """Criterion 7: ``write_source`` follows links, so a link is foreign whatever it names."""
    repo, ws = _repo_with_workspace(tmp_path)
    target = tmp_path / "manager" / "pre-commit"
    target.parent.mkdir()
    if not dangling:
        target.write_text(render_hook_script(command=["/opt/mitos"], workspace=ws,
                                             hook_file=str(target)), encoding="utf-8")
        before = _sha(str(target))
    hook = _hook_of(repo)
    os.symlink(str(target), hook)
    assert classify_hook_file(hook).kind == HOOK_FOREIGN
    code, _out, _err = _install(ws, _stub(tmp_path / "bin"), capsys)
    assert code == 1
    assert os.path.islink(hook)
    if dangling:
        assert not os.path.lexists(str(target))
    else:
        assert _sha(str(target)) == before


@_needs_git
def test_a_directory_named_pre_commit_is_refused(tmp_path, capsys) -> None:
    """Criterion 7, the non-regular sibling."""
    repo, ws = _repo_with_workspace(tmp_path)
    os.mkdir(_hook_of(repo))
    code, _out, _err = _install(ws, _stub(tmp_path / "bin"), capsys)
    assert code == 1
    assert os.path.isdir(_hook_of(repo))


def test_a_fifo_is_foreign_and_never_opened(tmp_path) -> None:
    """Opening a FIFO would block the classifier for ever; ``lstat`` says so first."""
    fifo = str(tmp_path / "pre-commit")
    os.mkfifo(fifo)
    assert classify_hook_file(fifo) == HookFileState(HOOK_FOREIGN)


def test_undecodable_bytes_are_foreign_and_absence_is_absent(tmp_path) -> None:
    path = tmp_path / "pre-commit"
    assert classify_hook_file(str(path)) == HookFileState(HOOK_ABSENT)
    path.write_bytes(b"#!/bin/sh\n" + HOOK_FILE_MARKER.encode() + b"\n\xff\xfe\n")
    assert classify_hook_file(str(path)) == HookFileState(HOOK_FOREIGN)


def test_a_hooks_path_that_is_a_file_is_unreadable_not_absent(tmp_path) -> None:
    """``lstat`` of ``<file>/pre-commit`` is ENOTDIR: nothing may be written there."""
    blocker = tmp_path / "hooks"
    blocker.write_text("")
    assert (classify_hook_file(str(blocker / "pre-commit"))
            == HookFileState(HOOK_UNREADABLE))
    assert set(HOOK_FILE_KINDS) == {HOOK_ABSENT, HOOK_OURS, HOOK_FOREIGN,
                                    HOOK_FOREIGN_WITH_BLOCK, HOOK_UNREADABLE}


@_needs_git
def test_an_in_tree_hooks_path_is_refused_with_the_relative_block(tmp_path, capsys) -> None:
    """Criterion 8: tracked, team-shared; nothing is written anywhere."""
    repo = make_scratch_repo(str(tmp_path / "repo"), hooks_path=".githooks")
    os.mkdir(os.path.join(repo, ".githooks"))
    ws = _workspace(os.path.join(repo, "sub", "ws"))
    before_shared = _tree(os.path.join(repo, ".githooks"))
    before_own = _tree(os.path.join(repo, ".git", "hooks"))
    code, out, err = _install(ws, _stub(tmp_path / "bin"), capsys)
    assert code == 1
    assert "inside this repository's work tree" in err
    assert _tree(os.path.join(repo, ".githooks")) == before_shared
    assert _tree(os.path.join(repo, ".git", "hooks")) == before_own
    block = _block_of(out)
    assert _printed_selector(block) == "./sub/ws"
    assert repo not in block and str(tmp_path) not in block


@_needs_git
def test_a_machine_wide_hooks_path_is_refused_as_outside(tmp_path, capsys,
                                                         monkeypatch) -> None:
    """Criterion 9: a ``core.hooksPath`` from global configuration serves every repository."""
    repo, ws = _repo_with_workspace(tmp_path)
    shared = tmp_path / "shared-hooks"
    shared.mkdir()
    global_config = tmp_path / "gitconfig"
    global_config.write_text(f"[core]\n\thooksPath = {shared}\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    code, out, err = _install(ws, _stub(tmp_path / "bin"), capsys)
    assert code == 1
    assert "outside this repository" in err
    assert os.listdir(str(shared)) == []
    block = _block_of(out)
    assert _printed_selector(block) == "."
    assert str(tmp_path) not in block


@_needs_git
def test_a_linked_worktree_installs_into_the_main_repositorys_hooks(tmp_path,
                                                                    capsys) -> None:
    """Criterion 10."""
    main_repo = make_scratch_repo(str(tmp_path / "main"))
    worktree = add_linked_worktree(main_repo, str(tmp_path / "wt"))
    ws = _workspace(os.path.join(worktree, "ws"))
    code, _out, err = _install(ws, _stub(tmp_path / "bin"), capsys)
    assert code == 0, err
    with open(_hook_of(main_repo), encoding="utf-8") as f:
        assert hook_script_workspace(f.read()) == ws


@_needs_git
def test_a_workspace_below_the_root_bakes_its_own_absolute_path(tmp_path, capsys) -> None:
    """Criterion 11."""
    repo, ws = _repo_with_workspace(tmp_path, sub=os.path.join("pkg", "ws"))
    code, _out, err = _install(ws, _stub(tmp_path / "bin"), capsys)
    assert code == 0, err
    with open(_hook_of(repo), encoding="utf-8") as f:
        assert hook_script_workspace(f.read()) == ws
    assert ws == os.path.join(repo, "pkg", "ws")


@_needs_git
def test_a_space_and_a_quote_install_and_reach_the_stub_exactly(tmp_path, capsys) -> None:
    """Criterion 12: the installed script hands the stub exactly the argv it baked."""
    repo, ws = _repo_with_workspace(tmp_path, sub="it's a ws")
    exe = _stub(tmp_path / "o'dd bin")
    code, _out, err = _install(ws, exe, capsys)
    assert code == 0, err
    hook = _hook_of(repo)
    log = str(tmp_path / "argv.log")
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    result = _run_sh([_SH, hook], path_dirs=[str(empty)], log=log, cwd=repo)
    assert result.returncode == 0, result.stderr
    assert _stub_argv(log) == ["hook-run", "-p", ws, "--hook-file", hook]
    with open(hook, encoding="utf-8") as f:
        assert hook_script_workspace(f.read()) == ws


@_needs_git
def test_a_newline_in_the_workspace_path_is_refused_by_name(tmp_path, capsys) -> None:
    """Criterion 13."""
    repo = make_scratch_repo(str(tmp_path / "repo"))
    ws = _workspace(os.path.join(repo, "a\nb", "ws"))
    code, _out, err = _install(ws, _stub(tmp_path / "bin"), capsys)
    assert code == 1
    assert "control character" in err
    assert not os.path.lexists(_hook_of(repo))


@_needs_git
def test_the_named_workspace_decides_the_repository_not_the_cwd(tmp_path, capsys,
                                                                monkeypatch) -> None:
    """Criterion 14: ``-p B`` from inside repository A."""
    repo_a = make_scratch_repo(str(tmp_path / "a"))
    repo_b = make_scratch_repo(str(tmp_path / "b"))
    ws_b = _workspace(os.path.join(repo_b, "ws"))
    before_a = _tree(os.path.join(repo_a, ".git", "hooks"))
    monkeypatch.chdir(repo_a)
    code, _out, err = _install(ws_b, _stub(tmp_path / "bin"), capsys)
    assert code == 0, err
    assert os.path.exists(_hook_of(repo_b))
    assert _tree(os.path.join(repo_a, ".git", "hooks")) == before_a


def test_outside_a_work_tree_is_refused(tmp_path, capsys) -> None:
    """Criterion 15, first half."""
    ws = _workspace(tmp_path / "plain")
    code, out, err = _install(ws, _stub(tmp_path / "bin"), capsys)
    assert code == 1
    assert "is not inside a git work tree" in err
    assert ws in out  # the echo leads, even on a refusal


def test_no_git_on_path_is_refused(tmp_path, capsys, monkeypatch) -> None:
    """Criterion 15, second half."""
    ws = _workspace(tmp_path / "plain")
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    code, _out, err = _install(ws, _stub(tmp_path / "bin"), capsys)
    assert code == 1
    assert "git was not found" in err


# --------------------------------------------------------------------------- #
# The report: keyless, keyed, unbuilt graph
# --------------------------------------------------------------------------- #

@_needs_git
def test_keyless_says_inactive_and_reads_no_store(tmp_path, capsys, monkeypatch) -> None:
    """Criterion 16, keyless: the key test comes first and opens nothing."""
    _repo, ws = _repo_with_workspace(tmp_path)

    def _never(*args, **kwargs):
        raise AssertionError("a keyless install must not read audit debt")
    monkeypatch.setattr(cli, "derive_audit_debt", _never)
    code, out, err = _install(ws, _stub(tmp_path / "bin"), capsys)
    assert code == 0, err
    assert "the gate stays inactive and every commit passes" in out


@_needs_git
def test_keyed_names_the_count_and_a_recipe_the_parser_reads(tmp_path, capsys,
                                                             monkeypatch) -> None:
    """Criterion 16, keyed, through a workspace path with a space in it."""
    _repo, ws = _repo_with_workspace(tmp_path, sub="my ws")
    assert _main(["-p", ws, "record", "The gate reads coverage, not timestamps.",
                  "--slug", "gate-reads-coverage", "--rejected", "Timestamps."]) == 0
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-dummy-never-sent")
    exe = _stub(tmp_path / "bin")
    code, out, err = _install(ws, exe, capsys)
    assert code == 0, err
    line = next(l for l in out.splitlines() if "not yet covered" in l)
    assert line.startswith("1 decision is not yet covered by a contradiction check")
    recipe = line.split("`")[1]
    tokens = shlex.split(recipe)
    assert tokens[0] == exe
    args = cli._build_parser().parse_args(tokens[1:])
    assert args.command == "check"
    # `init` registered the workspace, so the recipe names it; it must reach this one.
    assert cli._resolve_selector(args.project_post, "check").root == ws


@_needs_git
def test_keyed_with_nothing_uncovered_says_armed(tmp_path, capsys, monkeypatch) -> None:
    _repo, ws = _repo_with_workspace(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-dummy-never-sent")
    code, out, _err = _install(ws, _stub(tmp_path / "bin"), capsys)
    assert code == 0
    assert "The gate is armed, and no decision is uncovered." in out


@_needs_git
def test_an_unbuilt_graph_installs_and_says_the_count_is_unreadable(tmp_path, capsys,
                                                                    monkeypatch) -> None:
    """Criterion 17: ``init`` builds a graph, so the row removes it."""
    _repo, ws = _repo_with_workspace(tmp_path)
    os.remove(MitosConfig(ws).db_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-dummy-never-sent")
    code, out, _err = _install(ws, _stub(tmp_path / "bin"), capsys)
    assert code == 0
    assert "The uncovered count could not be read" in out


# --------------------------------------------------------------------------- #
# The baked command (criterion 18)
# --------------------------------------------------------------------------- #

_PY = "/opt/py 3/bin/python"


def _baked(argv0: str, found: Optional[str]) -> Optional[List[str]]:
    return cli._hook_command_from(argv0, _PY, lambda name: found, cli._MITOS_PACKAGE_DIR)


def test_a_separator_argv0_bakes_the_abspath_not_the_realpath(tmp_path) -> None:
    real = tmp_path / "venv" / "bin" / "mitos"
    real.parent.mkdir(parents=True)
    real.write_text("#!/bin/sh\n")
    link = tmp_path / "bin" / "mitos"
    link.parent.mkdir()
    link.symlink_to(os.path.join("..", "venv", "bin", "mitos"))
    assert _baked(str(link), None) == [str(link)]


@pytest.mark.parametrize("module_file", ["__main__.py", "cli.py"])
def test_a_package_dir_argv0_bakes_the_interpreter(module_file) -> None:
    argv0 = os.path.join(cli._MITOS_PACKAGE_DIR, module_file)
    assert _baked(argv0, "/usr/bin/mitos") == [_PY, "-m", "mitos"]


def test_a_bare_argv0_bakes_what_path_found_or_nothing() -> None:
    assert _baked("mitos", "/usr/local/bin/mitos") == ["/usr/local/bin/mitos"]
    assert _baked("mitos", None) is None


@_needs_git
def test_an_unnameable_executable_refuses_the_install(tmp_path, capsys, monkeypatch) -> None:
    repo, ws = _repo_with_workspace(tmp_path)
    monkeypatch.setattr(cli.shutil, "which", lambda name, *a, **k: None)
    code, _out, err = _install(ws, "mitos", capsys)
    assert code == 1
    assert "cannot tell which mitos executable is running" in err
    assert not os.path.lexists(_hook_of(repo))


@_needs_git
def test_the_path_sentence_appears_only_when_path_resolves_elsewhere(tmp_path, capsys,
                                                                     monkeypatch) -> None:
    _repo, ws = _repo_with_workspace(tmp_path)
    exe = _stub(tmp_path / "bin")
    sentence = "not the mitos on your PATH"
    monkeypatch.setattr(cli.shutil, "which", lambda name, *a, **k: exe)
    assert sentence not in _install(ws, exe, capsys)[1]
    monkeypatch.setattr(cli.shutil, "which",
                        lambda name, *a, **k: str(tmp_path / "other" / "mitos"))
    assert sentence in _install(ws, exe, capsys)[1]
    monkeypatch.setattr(cli.shutil, "which", lambda name, *a, **k: None)
    assert sentence in _install(ws, exe, capsys)[1]


@_needs_git
def test_the_dash_m_form_always_says_it_is_not_the_path_mitos(tmp_path, capsys,
                                                              monkeypatch) -> None:
    """Decided (scout W12): the hook runs this interpreter, never PATH's ``mitos``."""
    repo, ws = _repo_with_workspace(tmp_path)
    exe = _stub(tmp_path / "bin")
    monkeypatch.setattr(cli.shutil, "which", lambda name, *a, **k: exe)
    argv0 = os.path.join(cli._MITOS_PACKAGE_DIR, "__main__.py")
    code, out, _err = _install(ws, argv0, capsys)
    assert code == 0
    assert f"It runs {shlex.quote(sys.executable)} -m mitos." in out
    assert "not the mitos on your PATH" in out


# --------------------------------------------------------------------------- #
# Script semantics, under real sh with a stub (criteria 19, 20, 25)
# --------------------------------------------------------------------------- #

def _script(tmp_path, command: List[str], workspace: str = "/ws") -> str:
    hook = str(tmp_path / "hooks" / "pre-commit")
    os.makedirs(os.path.dirname(hook), exist_ok=True)
    with open(hook, "w", encoding="utf-8") as f:
        f.write(render_hook_script(command=command, workspace=workspace, hook_file=hook))
    return hook


@pytest.mark.parametrize("rc,expected", [
    (str(HOOK_BLOCK_EXIT), HOOK_BLOCK_EXIT),
    ("0", 0), ("1", 0), ("2", 0), ("126", 0), ("127", 0), ("signal", 0),
])
def test_only_the_block_code_fails_the_commit(tmp_path, rc, expected) -> None:
    """Criterion 19: a gate that fails closed on its own faults gets bypassed."""
    exe = _stub(tmp_path / "bin")
    hook = _script(tmp_path, [exe])
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    log = str(tmp_path / "argv.log")
    result = _run_sh([_SH, hook], path_dirs=[str(empty)], log=log, rc=rc)
    assert result.returncode == expected
    assert _stub_argv(log) == ["hook-run", "-p", "/ws", "--hook-file", hook]


def test_a_missing_baked_executable_falls_back_to_path(tmp_path) -> None:
    """Criterion 20, first half."""
    on_path = _stub(tmp_path / "path-bin")
    hook = _script(tmp_path, [str(tmp_path / "gone" / "mitos")])
    log = str(tmp_path / "argv.log")
    result = _run_sh([_SH, hook], path_dirs=[os.path.dirname(on_path)], log=log,
                     rc=str(HOOK_BLOCK_EXIT))
    assert result.returncode == HOOK_BLOCK_EXIT
    assert _stub_argv(log)[:2] == ["hook-run", "-p"]


def test_with_neither_executable_the_commit_passes_silently(tmp_path) -> None:
    """Criterion 20, second half."""
    hook = _script(tmp_path, [str(tmp_path / "gone" / "mitos")])
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    result = _run_sh([_SH, hook], path_dirs=[str(empty)], log=str(tmp_path / "log"))
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


def test_the_dash_m_form_runs_the_interpreter_with_the_module(tmp_path) -> None:
    py = _stub(tmp_path / "o py", name="python")
    hook = _script(tmp_path, [py, "-m", "mitos"])
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    log = str(tmp_path / "argv.log")
    assert _run_sh([_SH, hook], path_dirs=[str(empty)], log=log).returncode == 0
    assert _stub_argv(log) == ["-m", "mitos", "hook-run", "-p", "/ws", "--hook-file", hook]


def test_the_script_passes_its_own_path_and_neither_block_passes_one() -> None:
    """Criterion 25."""
    script = render_hook_script(command=["/opt/mitos"], workspace="/ws",
                                hook_file="/r/.git/hooks/pre-commit")
    assert "--hook-file /r/.git/hooks/pre-commit" in script
    for block in (render_hook_block(command=["/opt/mitos"], selector="/ws", guard_dir=None),
                  render_hook_block(command=None, selector=".", guard_dir="./.mitos")):
        assert "--hook-file" not in block


@pytest.mark.parametrize("workspace", [
    "/plain/ws", "/with space/ws", "/it's/ws", '/say "hi"/ws', "/cost $HOME/ws",
    "/tick `id`/ws",
])
def test_the_served_workspace_round_trips(workspace) -> None:
    """Serialization contract: the reader is the render's inverse, byte-exact."""
    script = render_hook_script(command=["/opt/mitos"], workspace=workspace,
                                hook_file="/r/pre-commit")
    assert hook_script_workspace(script) == workspace


def test_unsafe_shell_path_names_each_reason() -> None:
    assert unsafe_shell_path("/a b/it's \"q\" $x `y`") is None
    assert unsafe_shell_path("relative/x") == "not absolute"
    assert unsafe_shell_path("/a\nb") == "control character"
    assert unsafe_shell_path("/a\x7fb") == "control character"
    assert unsafe_shell_path(os.fsdecode(b"/caf\xe9")) == "not UTF-8"


# --------------------------------------------------------------------------- #
# The blocks inside a host `sh -eu` script (criterion 21)
# --------------------------------------------------------------------------- #

_SENTINEL = "host-after-block"


def _host(tmp_path, block: str, *, sentinel: bool = True) -> str:
    host = tmp_path / "host.sh"
    tail = f"echo {_SENTINEL}\n" if sentinel else ""
    host.write_text(f"set -eu\necho host-before\n{block}{tail}", encoding="utf-8")
    return str(host)


def _abs_block(exe: str) -> str:
    return render_hook_block(command=[exe], selector="/ws", guard_dir=None)


_REL_BLOCK = render_hook_block(command=None, selector="./ws", guard_dir="./ws/.mitos")


@pytest.mark.parametrize("rc", ["0", "2", "1", "127"])
@pytest.mark.parametrize("form", ["absolute", "relative"])
def test_a_passing_block_falls_through_to_the_host(tmp_path, form, rc) -> None:
    exe = _stub(tmp_path / "bin")
    block = _abs_block(exe) if form == "absolute" else _REL_BLOCK
    (tmp_path / "ws" / ".mitos").mkdir(parents=True)
    log = str(tmp_path / "argv.log")
    result = _run_sh([_SH, _host(tmp_path, block)], path_dirs=[str(tmp_path / "bin")],
                     log=log, rc=rc, cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"host-before\n{_SENTINEL}\n"
    assert _stub_argv(log) is not None


@pytest.mark.parametrize("form", ["absolute", "relative"])
def test_a_blocking_block_exits_the_host_with_the_block_code(tmp_path, form) -> None:
    exe = _stub(tmp_path / "bin")
    block = _abs_block(exe) if form == "absolute" else _REL_BLOCK
    (tmp_path / "ws" / ".mitos").mkdir(parents=True)
    result = _run_sh([_SH, _host(tmp_path, block)], path_dirs=[str(tmp_path / "bin")],
                     log=str(tmp_path / "log"), rc=str(HOOK_BLOCK_EXIT),
                     cwd=str(tmp_path))
    assert result.returncode == HOOK_BLOCK_EXIT
    assert _SENTINEL not in result.stdout


def test_the_relative_block_is_silent_where_there_is_no_workspace(tmp_path) -> None:
    _stub(tmp_path / "bin")
    log = str(tmp_path / "argv.log")
    result = _run_sh([_SH, _host(tmp_path, _REL_BLOCK)], path_dirs=[str(tmp_path / "bin")],
                     log=log, rc=str(HOOK_BLOCK_EXIT), cwd=str(tmp_path))
    assert result.returncode == 0
    assert result.stdout == f"host-before\n{_SENTINEL}\n" and result.stderr == ""
    assert _stub_argv(log) is None


def test_the_relative_block_is_silent_without_mitos_on_path(tmp_path) -> None:
    (tmp_path / "ws" / ".mitos").mkdir(parents=True)
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    result = _run_sh([_SH, _host(tmp_path, _REL_BLOCK)], path_dirs=[str(empty)],
                     log=str(tmp_path / "log"), cwd=str(tmp_path))
    assert (result.returncode, result.stderr) == (0, "")
    assert result.stdout == f"host-before\n{_SENTINEL}\n"


@pytest.mark.parametrize("form", ["absolute", "relative"])
def test_a_block_as_the_hosts_final_statement_leaves_status_zero(tmp_path, form) -> None:
    """A bare ``[ … ] && exit`` as the last command would leave 1: every commit blocked."""
    exe = _stub(tmp_path / "bin")
    block = _abs_block(exe) if form == "absolute" else _REL_BLOCK
    (tmp_path / "ws" / ".mitos").mkdir(parents=True)
    result = _run_sh([_SH, _host(tmp_path, block, sentinel=False)],
                     path_dirs=[str(tmp_path / "bin")], log=str(tmp_path / "log"),
                     cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- #
# The printed selector resolves (criterion 22) and the markers (criterion 23)
# --------------------------------------------------------------------------- #

@_needs_git
@pytest.mark.parametrize("sub,expected", [("", "."), ("ws", "./ws"),
                                          (os.path.join("sub", "ws"), "./sub/ws")])
def test_the_printed_selector_is_path_shaped_and_resolves(tmp_path, capsys, monkeypatch,
                                                          sub, expected) -> None:
    repo = make_scratch_repo(str(tmp_path / "repo"), hooks_path=".githooks")
    ws = _workspace(os.path.join(repo, sub) if sub else repo)
    code, out, _err = _install(ws, _stub(tmp_path / "bin"), capsys)
    assert code == 1
    selector = _printed_selector(_block_of(out))
    assert selector == expected
    assert routing.is_path_shaped(selector)
    monkeypatch.chdir(repo)
    assert cli._resolve_selector(selector, "hook-run").root == ws


def test_the_markers_are_disjoint_and_where_they_belong() -> None:
    """Criterion 23."""
    assert HOOK_BLOCK_MARKER not in HOOK_FILE_MARKER
    assert HOOK_FILE_MARKER not in HOOK_BLOCK_MARKER
    for block in (render_hook_block(command=["/opt/mitos"], selector="/ws", guard_dir=None),
                  render_hook_block(command=None, selector=".", guard_dir="./.mitos")):
        assert HOOK_FILE_MARKER not in block and block.startswith(HOOK_BLOCK_MARKER)
    script = render_hook_script(command=["/opt/mitos"], workspace="/ws", hook_file="/h")
    assert script.split("\n")[1] == HOOK_FILE_MARKER
    assert HOOK_BLOCK_MARKER not in script
    assert f"exit {HOOK_BLOCK_EXIT}" in script


# --------------------------------------------------------------------------- #
# The stale line (criterion 24)
# --------------------------------------------------------------------------- #

def test_a_stale_hook_names_itself_in_one_line(tmp_path, capsys) -> None:
    gone = os.path.join(os.path.realpath(str(tmp_path)), "moved-away")
    capsys.readouterr()
    code = _main(["hook-run", "-p", gone, "--hook-file", "/x/pre-commit"])
    out, err = (lambda c: (c.out, c.err))(capsys.readouterr())
    assert code == 0 and out == ""
    assert len(err.splitlines()) == 1 and err.startswith("mitos commit gate: ")
    assert "'/x/pre-commit'" in err and repr(gone) in err
    assert "no longer a Mitos workspace" in err
    assert "let through" in err and "install the hook again" in err
    assert "or delete '/x/pre-commit'" in err
    assert "`" not in err and "--project" not in err


def test_without_the_flag_the_line_is_unchanged(tmp_path, capsys) -> None:
    gone = os.path.join(os.path.realpath(str(tmp_path)), "moved-away")
    capsys.readouterr()
    code = _main(["hook-run", "-p", gone])
    err = capsys.readouterr().err
    assert code == 0
    assert err == (f"mitos commit gate: no Mitos workspace at {gone!r} (project "
                   f"selector {gone!r}), so there is nothing to gate; the commit was "
                   f"let through.\n")


# --------------------------------------------------------------------------- #
# Help (criterion 27)
# --------------------------------------------------------------------------- #

def _flat_help(verb: str) -> str:
    sub = _subparsers(cli._build_parser())[verb]
    return " ".join((sub.description + " " + sub.epilog).split())


def test_hook_install_help_says_what_where_what_it_refuses_and_how_it_dies() -> None:
    text = _flat_help("hook-install")
    for phrase in ("pre-commit hook", "repository's own git directory",
                   "someone else's pre-commit", "shared hooks directory",
                   "Removing the gate is deleting the file it names"):
        assert phrase in text, phrase


def test_hook_run_help_describes_hook_file() -> None:
    sub = _subparsers(cli._build_parser())["hook-run"]
    action = next(a for a in sub._actions if "--hook-file" in a.option_strings)
    helptext = " ".join(action.help.split())
    assert "hook-install writes passes its own path" in helptext
    assert "install again, or delete it" in helptext
