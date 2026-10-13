"""The process fence: one module spawns, and the MCP server cannot reach it.

G1 gives mitos its first subprocess calls — ``git``, asked where a workspace's
repository lives, from three CLI sites. The framework promises that the MCP server
runs no shell commands (P8); until this phase that held only because nothing in
the package spawned at all. These rows make it a property:

- **Spawn sweep.** Nothing in ``mitos/`` except ``_git.py`` spawns a process —
  ``subprocess``/``multiprocessing``/``pty`` imports, ``os.system``/``popen``/
  ``exec*``/``spawn*``/``fork`` through ``os`` or an alias of it, ``from os import``
  of those, ``create_subprocess_*`` on any base, and ``import_module``/
  ``__import__`` of a spawning module by constant name. ``sys.argv``,
  ``sys.executable`` and ``shutil.which`` are not spawns (``which`` is a ``PATH``
  scan) and must not be flagged: ``cli._running_mitos_command`` reads all three.
- **Closure sweep.** ``mcp_server``'s runtime import closure — function-local
  imports included, ``TYPE_CHECKING`` blocks pruned, no lint boundary — contains
  neither ``_git.py`` nor ``cli.py``. ``sync.py`` must be in it, which proves the
  walk descends into function bodies (``mcp_server`` imports ``sync`` only there).

Both are static because a runtime check cannot serve: ``asyncio`` imports
``subprocess`` on its own, and a bare ``import mitos.sync`` already leaves it in
``sys.modules``, so ``sys.modules`` cannot tell a spawner from a bystander.

Honest residual: the sweep reads ``mitos/`` source only. It cannot see a
third-party library that spawns, a spawn reached through ``getattr`` on a string,
or a C extension. It fences mitos's own code, which is the claim P8 needs.

The git-helper rows run the real ``git`` binary against scratch repositories under
masked global/system config with ``GIT_CEILING_DIRECTORIES`` at ``tmp_path``.
``make_scratch_repo``, ``add_linked_worktree``, ``scratch_git`` and
``isolate_git`` are named for reuse by sibling test modules (3e, 3h).
"""

import ast
import glob
import os
import shutil
import subprocess
import sys
import textwrap
from typing import Dict, List, Optional, Set

import pytest

import mitos
import mitos.cli
import mitos.parser
from mitos import _git
from mitos._git import (GitLocation, NotAWorkTree, REASON_GIT_UNAVAILABLE,
                        REASON_OUTSIDE_WORK_TREE, is_within, locate_repository)
from test_conflict_closeout import _module_source_path, _runtime_mitos_imports

_PKG_DIR = os.path.dirname(mitos.__file__)

# A floor, not a count: its only job is to fail loudly if the glob stops finding
# the package (the house precedent is test_workspace_root_discipline's).
_MODULE_FLOOR = 15

_needs_git = pytest.mark.skipif(shutil.which("git") is None,
                                reason="3d: needs a git binary — CI images carry one")


# --- the spawn detector -----------------------------------------------------

_SPAWN_MODULES = frozenset({"subprocess", "multiprocessing", "pty"})
_OS_SPAWN_NAMES = frozenset({"system", "popen", "fork", "forkpty"})
_OS_SPAWN_PREFIXES = ("exec", "spawn", "posix_spawn")
_ASYNC_SPAWN_NAMES = frozenset({"create_subprocess_exec", "create_subprocess_shell"})


def _is_os_spawn_name(name: str) -> bool:
    return name in _OS_SPAWN_NAMES or name.startswith(_OS_SPAWN_PREFIXES)


def _spawn_hits(source: str, filename: str) -> List[str]:
    """Every process-spawning construct in a module's source, as ``line: shape``.

    ``ast.walk`` over the whole tree, so function bodies count. The ``os`` spawn
    names are anchored to ``os`` or a name the module bound to it — never matched
    on any base, so ``sys.executable`` (which starts with ``exec``) stays clean.
    """
    tree = ast.parse(source, filename=filename)
    os_names: Set[str] = {"os"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "os":
                    os_names.add(alias.asname or "os")

    hits: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _SPAWN_MODULES:
                    hits.append(f"{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            root = node.module.split(".")[0]
            if root in _SPAWN_MODULES:
                hits.append(f"{node.lineno}: from {node.module} import")
            elif node.module == "os":
                for alias in node.names:
                    if _is_os_spawn_name(alias.name):
                        hits.append(f"{node.lineno}: from os import {alias.name}")
        elif isinstance(node, ast.Attribute):
            if node.attr in _ASYNC_SPAWN_NAMES:
                hits.append(f"{node.lineno}: .{node.attr}")
            elif (_is_os_spawn_name(node.attr) and isinstance(node.value, ast.Name)
                    and node.value.id in os_names):
                hits.append(f"{node.lineno}: {node.value.id}.{node.attr}")
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            dynamic = ((isinstance(func, ast.Name)
                        and func.id in ("__import__", "import_module"))
                       or (isinstance(func, ast.Attribute) and func.attr == "import_module"))
            first = node.args[0]
            if (dynamic and isinstance(first, ast.Constant) and isinstance(first.value, str)
                    and first.value.split(".")[0] in _SPAWN_MODULES):
                hits.append(f"{node.lineno}: dynamic import of {first.value}")
    return hits


def _package_modules() -> List[str]:
    return sorted(glob.glob(os.path.join(_PKG_DIR, "*.py")))


def test_nothing_in_mitos_spawns_a_process_except_the_git_module() -> None:
    """The spawn sweep: hits outside ``_git.py`` are empty; ``_git.py`` holds some."""
    modules = _package_modules()
    assert len(modules) >= _MODULE_FLOOR

    hits: Dict[str, List[str]] = {}
    for path in modules:
        with open(path, encoding="utf-8") as f:
            found = _spawn_hits(f.read(), os.path.basename(path))
        if found:
            hits[os.path.basename(path)] = found

    git_hits = hits.pop("_git.py", [])
    assert hits == {}
    assert git_hits, "the detector must see the one module it allows"


@pytest.mark.parametrize("source", [
    "import subprocess",
    "def f():\n    import subprocess.run_it",
    "from subprocess import run",
    "import multiprocessing",
    "from pty import spawn",
    "import os\nos.system('x')",
    "import os as _o\ndef f():\n    _o.execvp('x', [])",
    "import os\nos.posix_spawnp('x', [], {})",
    "import os\nos.fork()",
    "from os import popen",
    "from os import spawnl",
    "import asyncio\nasyncio.create_subprocess_exec('x')",
    "loop.create_subprocess_shell('x')",
    "import importlib\nimportlib.import_module('subprocess')",
    "__import__('multiprocessing.pool')",
], ids=["a-import", "a-in-function", "a-from", "a-multiprocessing", "a-pty",
        "b-system", "b-alias-in-function", "b-posix-spawn", "b-fork", "c-popen",
        "c-spawn", "d-exec", "d-shell-any-base", "e-import-module", "e-dunder"])
def test_the_spawn_detector_sees_every_shape(source: str) -> None:
    """In-row positive controls: each shape (a)–(e) is detected, in bodies and via aliases."""
    assert _spawn_hits(source, "<plant>")


def test_the_spawn_detector_does_not_flag_reading_the_interpreter_or_path() -> None:
    """``sys.executable``/``sys.argv``/``shutil.which`` read, never spawn (3c2's reads)."""
    source = textwrap.dedent("""
        import os, shutil, sys
        def running():
            exe = sys.executable
            argv = sys.argv
            found = shutil.which("mitos")
            self.execute(exe)
            return os.path.exists(exe), argv, found
    """)
    assert _spawn_hits(source, "<control>") == []


# --- the closure sweep ------------------------------------------------------

def _mcp_server_closure() -> Set[str]:
    """Basenames of every source file ``mcp_server`` runtime-reaches, function-local included.

    The check family's walker, without its telemetry boundary: this is the whole
    closure, not a lint family.
    """
    seen: Set[str] = set()
    queue = [os.path.join(_PKG_DIR, "mcp_server.py")]
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        with open(path, encoding="utf-8") as f:
            source = f.read()
        for module_name in _runtime_mitos_imports(source, os.path.basename(path)):
            resolved = _module_source_path(module_name)
            if resolved is not None:
                queue.append(resolved)
    return {os.path.basename(p) for p in seen}


def test_the_mcp_server_cannot_reach_the_spawner_or_the_cli() -> None:
    """``_git.py`` and ``cli.py`` are outside ``mcp_server``'s closure; ``sync.py`` inside.

    ``sync.py`` is the witness: ``mcp_server`` imports it only function-locally, so
    its presence proves the walk counts function bodies. The exact set is not
    pinned — the claim is two exclusions, and every legitimate new import would
    churn a pinned set.
    """
    closure = _mcp_server_closure()
    assert "sync.py" in closure
    assert "_git.py" not in closure
    assert "cli.py" not in closure


# --- the loader move --------------------------------------------------------

def test_the_cli_re_exports_the_parsers_loader() -> None:
    assert mitos.cli.load_format_spec is mitos.parser.load_format_spec
    spec_path = os.path.join(_PKG_DIR, "format-spec.md")
    with open(spec_path, encoding="utf-8") as f:
        assert mitos.parser.load_format_spec() == f.read()


_HEAL_PROBE = textwrap.dedent("""
    import os, sys
    from mitos.config import MitosConfig
    from mitos.sync import MitosSyncManager
    ws = sys.argv[1]
    config = MitosConfig(ws)
    config.db_path = os.path.join(ws, ".mitos", "graph.sqlite")
    config.decisions_file = os.path.join(ws, "decisions.md")
    config.archive_dir = os.path.join(ws, "decisions", "archive")
    MitosSyncManager(config).auto_heal_decisions_file()
    print("mitos.cli" in sys.modules)
""")

_MARKER = "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->"


def test_healing_a_decisions_file_does_not_import_the_cli(tmp_path) -> None:
    """Every MCP record runs the heal; after 3d it restores the header without the CLI.

    A fresh interpreter, because this test process has long since imported
    ``mitos.cli``. Non-vacuous: the header was damaged and must come back — the
    heal's own stderr line, the canonical sample block from the spec, and a second
    heal in-process that finds nothing left to change.
    """
    ws = tmp_path / "ws"
    (ws / ".mitos").mkdir(parents=True)
    decisions = ws / "decisions.md"
    decisions.write_text(f"# Decisions\n{_MARKER}\n", encoding="utf-8")

    out = subprocess.run([sys.executable, "-c", _HEAL_PROBE, str(ws)],
                         capture_output=True, text=True, check=True)

    assert out.stdout.strip() == "False"
    assert "Auto-restored decisions.md sample format header block" in out.stderr
    healed = decisions.read_text(encoding="utf-8")
    assert healed.startswith("# Decisions for Mitos\n")
    assert "## SAMPLE FORMAT — auto-restored by mitos sync" in healed.split(_MARKER)[0]

    from mitos.config import MitosConfig
    from mitos.sync import MitosSyncManager
    config = MitosConfig(str(ws))
    config.db_path = str(ws / ".mitos" / "graph.sqlite")
    config.decisions_file = str(decisions)
    config.archive_dir = str(ws / "decisions" / "archive")
    MitosSyncManager(config).auto_heal_decisions_file()
    assert decisions.read_text(encoding="utf-8") == healed


# --- the git helper ---------------------------------------------------------

def isolate_git(monkeypatch: pytest.MonkeyPatch, ceiling: str) -> None:
    """Masks machine git config, caps discovery at ``ceiling``, and drops the locating vars.

    The locating vars are dropped so a suite run from inside a hook cannot leak its
    ``GIT_DIR`` in; a row that wants one sets it back explicitly.
    """
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", ceiling)
    for name in _git.GIT_REPO_LOCATING_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _git_isolation(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    isolate_git(monkeypatch, str(tmp_path))


def scratch_git(cwd: str, *args: str) -> str:
    """Runs a fixture ``git`` command with its own isolation; returns stdout.

    Built independently of ``_git._git_env`` — the unit under test does not build
    its own fixtures. Identity is passed per command.
    """
    env = {k: v for k, v in os.environ.items() if k not in _git.GIT_REPO_LOCATING_ENV}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    out = subprocess.run(
        ["git", "-c", "user.name=mitos-test", "-c", "user.email=test@mitos.invalid",
         "-c", "init.defaultBranch=main", *args],
        cwd=cwd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        check=True)
    return out.stdout


def make_scratch_repo(root: str, *, hooks_path: Optional[str] = None) -> str:
    """Creates a git repository at ``root`` with one empty commit; returns its realpath.

    Args:
        root: The directory to initialise (created if missing).
        hooks_path: If given, written as the repository's ``core.hooksPath``.
    """
    os.makedirs(root, exist_ok=True)
    scratch_git(root, "init", "-q")
    scratch_git(root, "commit", "-q", "--allow-empty", "-m", "scratch")
    if hooks_path is not None:
        scratch_git(root, "config", "core.hooksPath", hooks_path)
    return os.path.realpath(root)


def add_linked_worktree(repo: str, dest: str) -> str:
    """Adds a linked worktree of ``repo`` at ``dest``; returns its realpath."""
    scratch_git(repo, "worktree", "add", "-q", "--detach", dest)
    return os.path.realpath(dest)


def _default_location(root: str) -> GitLocation:
    return GitLocation(top_level=root, common_dir=os.path.join(root, ".git"),
                       hooks_dir=os.path.join(root, ".git", "hooks"))


@_needs_git
def test_a_default_repository_is_located_from_its_root(tmp_path) -> None:
    root = make_scratch_repo(str(tmp_path / "repo"))
    assert locate_repository(root) == _default_location(root)


@_needs_git
def test_a_workspace_below_the_root_gets_the_same_answers(tmp_path) -> None:
    root = make_scratch_repo(str(tmp_path / "repo"))
    ws = os.path.join(root, "sub", "ws")
    os.makedirs(ws)
    assert locate_repository(ws) == _default_location(root)


@_needs_git
def test_an_in_tree_hooks_path_is_honoured(tmp_path) -> None:
    root = make_scratch_repo(str(tmp_path / "repo"), hooks_path=".githooks")
    ws = os.path.join(root, "sub", "ws")
    os.makedirs(ws)
    location = locate_repository(ws)
    assert isinstance(location, GitLocation)
    assert location.hooks_dir == os.path.join(root, ".githooks")
    assert is_within(location.hooks_dir, location.top_level)


@_needs_git
def test_a_machine_wide_hooks_path_from_a_global_config_file(tmp_path, monkeypatch) -> None:
    root = make_scratch_repo(str(tmp_path / "repo"))
    shared = tmp_path / "shared-hooks"
    shared.mkdir()
    global_config = tmp_path / "gitconfig"
    global_config.write_text(f"[core]\n\thooksPath = {shared}\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))

    location = locate_repository(root)
    assert isinstance(location, GitLocation)
    assert location.hooks_dir == os.path.realpath(shared)
    assert not is_within(location.hooks_dir, location.top_level)
    assert not is_within(location.hooks_dir, location.common_dir)


@_needs_git
def test_a_linked_worktree_names_the_main_repositorys_git_dir(tmp_path) -> None:
    main = make_scratch_repo(str(tmp_path / "main"))
    worktree = add_linked_worktree(main, str(tmp_path / "wt"))
    assert locate_repository(worktree) == GitLocation(
        top_level=worktree, common_dir=os.path.join(main, ".git"),
        hooks_dir=os.path.join(main, ".git", "hooks"))


@_needs_git
def test_a_plain_directory_is_outside_a_work_tree(tmp_path, monkeypatch) -> None:
    """Asked from inside a repository, so an answer read off the cwd would be "inside".

    Without the chdir this row's red under a dropped ``cwd=`` depended on pytest
    happening to run from a git checkout (found by 8a1's re-plant in a copy).
    """
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.chdir(make_scratch_repo(str(tmp_path / "cwd-repo")))
    assert locate_repository(str(plain)) == NotAWorkTree(REASON_OUTSIDE_WORK_TREE)


@_needs_git
def test_a_bare_repository_and_the_inside_of_dot_git_are_outside(tmp_path) -> None:
    bare = tmp_path / "bare.git"
    bare.mkdir()
    scratch_git(str(bare), "init", "-q", "--bare")
    root = make_scratch_repo(str(tmp_path / "repo"))
    for directory in (str(bare), os.path.join(root, ".git"),
                      os.path.join(root, ".git", "hooks")):
        assert locate_repository(directory) == NotAWorkTree(REASON_OUTSIDE_WORK_TREE)


def test_a_missing_directory_is_outside_and_never_spawns(tmp_path, monkeypatch) -> None:
    def _no_spawn(*args, **kwargs):
        raise AssertionError("a missing directory must not spawn")
    monkeypatch.setattr(_git.subprocess, "run", _no_spawn)
    assert (locate_repository(str(tmp_path / "absent"))
            == NotAWorkTree(REASON_OUTSIDE_WORK_TREE))


def test_no_git_on_path_is_git_unavailable(tmp_path, monkeypatch) -> None:
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert locate_repository(str(tmp_path)) == NotAWorkTree(REASON_GIT_UNAVAILABLE)


def _fake_git(tmp_path, body: str) -> str:
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    script = bin_dir / "git"
    script.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    script.chmod(0o755)
    return str(bin_dir)


@pytest.mark.skipif(shutil.which("sleep") is None, reason="3d: needs a sleep binary")
def test_a_hung_git_is_git_unavailable(tmp_path, monkeypatch) -> None:
    # `exec`, so the timeout's kill reaches the sleeper itself and no orphan holds
    # the pipes open past it.
    monkeypatch.setenv("PATH", _fake_git(tmp_path, f"exec {shutil.which('sleep')} 30"))
    monkeypatch.setattr(_git, "GIT_TIMEOUT_SECONDS", 0.3)
    assert locate_repository(str(tmp_path)) == NotAWorkTree(REASON_GIT_UNAVAILABLE)


def test_garbled_git_output_is_outside_a_work_tree(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", _fake_git(tmp_path, "printf 'true\\n/somewhere\\n'"))
    assert locate_repository(str(tmp_path)) == NotAWorkTree(REASON_OUTSIDE_WORK_TREE)


def test_a_relative_directory_is_a_programming_error() -> None:
    with pytest.raises(ValueError):
        locate_repository("sub/ws")
    with pytest.raises(ValueError):
        is_within("hooks", "/abs")


# --- the caller's context never leaks ---------------------------------------

@_needs_git
def test_the_callers_cwd_does_not_leak(tmp_path, monkeypatch) -> None:
    repo_a = make_scratch_repo(str(tmp_path / "a"))
    repo_b = make_scratch_repo(str(tmp_path / "b"))
    ws = os.path.join(repo_a, "sub", "ws")
    os.makedirs(ws)
    monkeypatch.chdir(repo_b)
    assert locate_repository(ws) == _default_location(repo_a)


@_needs_git
@pytest.mark.parametrize("with_work_tree", [False, True], ids=["git-dir", "git-dir+work-tree"])
def test_the_callers_git_dir_does_not_leak(tmp_path, monkeypatch, with_work_tree) -> None:
    """GIT_DIR alone is the silent mis-aim (exit 0, B's dirs); with GIT_WORK_TREE, a false "no"."""
    repo_a = make_scratch_repo(str(tmp_path / "a"))
    repo_b = make_scratch_repo(str(tmp_path / "b"))
    ws = os.path.join(repo_a, "sub", "ws")
    os.makedirs(ws)
    monkeypatch.chdir(repo_b)
    monkeypatch.setenv("GIT_DIR", os.path.join(repo_b, ".git"))
    if with_work_tree:
        monkeypatch.setenv("GIT_WORK_TREE", repo_b)
    location = locate_repository(ws)
    assert isinstance(location, GitLocation)
    assert location.top_level == repo_a
    assert location.common_dir == os.path.join(repo_a, ".git")
    assert location.hooks_dir == os.path.join(repo_a, ".git", "hooks")


@_needs_git
def test_a_symlinked_workspace_gets_the_real_paths_answers(tmp_path) -> None:
    root = make_scratch_repo(str(tmp_path / "repo"))
    ws = os.path.join(root, "sub", "ws")
    os.makedirs(ws)
    link = tmp_path / "link-to-ws"
    link.symlink_to(ws, target_is_directory=True)
    assert locate_repository(str(link)) == locate_repository(ws) == _default_location(root)
