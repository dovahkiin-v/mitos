"""Phase 3h: the real-binary proof — real ``git commit``s through this branch's own ``mitos``.

Every earlier G1 row ran in-process or with a stub standing in for ``mitos``. The
hook fails open, so a defect in the seam between the line ``hook-install`` writes
and the binary git runs (a verb that does not parse, a selector that resolves to
nobody, a hook baked to the wrong build) reads exactly like a passing commit. Here
nothing is mocked: real ``git``, real ``/bin/sh``, and the console script that
``sysconfig`` names for this interpreter, proven to import this checkout.

The frame, per row:

- The machine is masked. ``hermetic_mitos_env`` gives each test its own
  ``XDG_CONFIG_HOME`` (so ``init`` registers nothing machine-wide) and strips the
  LLM keys; ``isolate_git`` masks git config and caps discovery at ``tmp_path``.
- One declared environment is handed to every subprocess, ``git`` included,
  because the hook inherits ``git commit``'s environment. It carries a dummy
  judge key (so the gate is armed), no embedding key (so the attempted check can
  never complete or spend), a dead ``QDRANT_URL``, and a ``PATH`` that holds no
  ``mitos`` except the one a row puts there.
- A block is asserted by the hook's own message **and** an unchanged ``HEAD``:
  a commit that fails for any other reason also exits non-zero. A pass is
  asserted as a transition from a block in the same repository, because the
  quiet passes and a hook that could not start ``mitos`` are the same silence.
"""

import os
import shlex
import shutil
import subprocess
import sysconfig
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytest

import mitos
from mitos import routing
from mitos.commit_gate import (
    GATE_ATTEMPTED, GATE_NOTHING_UNCOVERED, HOOK_BLOCK_MARKER,
    evaluate_gate, hook_script_workspace, render_hook_script,
)
from mitos.config import MitosConfig
from mitos.telemetry import ATTEMPT_STARTED, LastAttempt, read_last_attempt
from test_process_fence import _needs_git, isolate_git, make_scratch_repo, scratch_git

# An obvious dummy: it arms the gate and is never sent (no row reaches a judge).
_DUMMY_JUDGE_KEY = "sk-ant-mitos-3h-dummy-never-sent"
# The value test_check_cli's `offline` fixture uses: nothing listens there.
_DEAD_QDRANT = "http://localhost:9"
# render_hook_block's closing line has no exported name; spelled once, here.
_BLOCK_END = "# end of the mitos commit gate block"
_BLOCK_WORDS = "not been covered by a contradiction check"
_GATE_LINE = "mitos commit gate:"
_TIMEOUT = 60
_IDENTITY = ("-c", "user.name=mitos-test", "-c", "user.email=test@mitos.invalid")
_HOST_LOG = ".git/mitos-host.log"
_INSTALL_FIX = ("install this checkout into the interpreter running pytest with "
                "`python -m pip install -e .`; a run from a linked worktree fails "
                "here too, because the editable install maps `mitos` to the main "
                "checkout")


@pytest.fixture(autouse=True)
def _git_isolation(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    isolate_git(monkeypatch, str(tmp_path))
    # A cwd-leaning defect must land where git cannot see a repository, never in
    # the checkout's own `.git/hooks` (3e fresh-eyes).
    monkeypatch.chdir(tmp_path)


# --------------------------------------------------------------------------- #
# The branch binary
# --------------------------------------------------------------------------- #

def _branch_mitos_path() -> str:
    """The console script of the interpreter running pytest: what a user's hook bakes."""
    return os.path.join(sysconfig.get_path("scripts"), "mitos")


_SH_TRAMPOLINE = "'''exec' "
_SH_TRAMPOLINE_ARGS = ' "$0" "$@"'


def _script_interpreter(path: str) -> Optional[str]:
    """The Python a console script runs under, or ``None`` when it names none.

    pip writes a plain ``#!<python>`` unless that path holds a space or exceeds
    the kernel's 127-byte shebang limit; then it writes ``#!/bin/sh`` and an
    ``'''exec' <python> "$0" "$@"`` trampoline on line 2, which is read here too.
    """
    with open(path, "rb") as f:
        first = f.readline().decode("utf-8", "replace").strip()
        second = f.readline().decode("utf-8", "replace").strip()
    if not first.startswith("#!"):
        return None
    interpreter = first[2:].strip()
    if interpreter == "/bin/sh" and second.startswith(_SH_TRAMPOLINE) \
            and second.endswith(_SH_TRAMPOLINE_ARGS):
        quoted = second[len(_SH_TRAMPOLINE):-len(_SH_TRAMPOLINE_ARGS)]
        words = shlex.split(quoted)
        return words[0] if len(words) == 1 else None
    return interpreter


def _identity_problem(path: str, cwd: str) -> Optional[str]:
    """Says why ``path`` is not this checkout's console script, or ``None`` when it is.

    A path that merely exists could be an install of another commit, so the proof
    is the script's own shebang interpreter importing this checkout's package.
    Run from a neutral ``cwd``, since ``-c`` puts the cwd on ``sys.path``.
    """
    if not os.path.isfile(path):
        return f"no console script at {path!r}"
    if not os.access(path, os.X_OK):
        return f"{path!r} is not executable"
    interpreter = _script_interpreter(path)
    if interpreter is None:
        return f"{path!r} names no Python interpreter on its shebang line"
    probe = subprocess.run(
        [interpreter, "-c", "import mitos, os; print(os.path.realpath(mitos.__file__))"],
        cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        timeout=_TIMEOUT)
    if probe.returncode != 0:
        return (f"{interpreter!r} (the shebang of {path!r}) could not import mitos: "
                f"{probe.stderr.strip()}")
    theirs = probe.stdout.strip()
    ours = os.path.realpath(mitos.__file__)
    if theirs != ours:
        return (f"{path!r} imports mitos from {theirs!r}, not this checkout's "
                f"{ours!r}")
    return None


@pytest.fixture(scope="module")
def branch_mitos(tmp_path_factory) -> str:
    """This checkout's console script. Fails, never skips: a skip is the vacuous frame."""
    path = _branch_mitos_path()
    problem = _identity_problem(path, str(tmp_path_factory.mktemp("identity")))
    if problem is not None:
        pytest.fail(f"3h needs this branch's own mitos: {problem}. To fix it, "
                    f"{_INSTALL_FIX}.", pytrace=False)
    return path


# --------------------------------------------------------------------------- #
# The declared environment
# --------------------------------------------------------------------------- #

def _tool_dirs(private_bin: str) -> List[str]:
    """The directories that hold ``git`` and ``sh``, none of which may hold a ``mitos``.

    A directory that does (``git`` in ``~/.local/bin`` beside a pipx ``mitos``)
    would leak another build into every row, so the tool is linked into
    ``private_bin`` instead of adding its directory.
    """
    found: List[str] = []
    for tool in ("git", "sh"):
        where = shutil.which(tool)
        assert where is not None, f"3h needs {tool} on PATH"
        directory = os.path.dirname(where)
        if os.path.lexists(os.path.join(directory, "mitos")):
            os.makedirs(private_bin, exist_ok=True)
            link = os.path.join(private_bin, tool)
            if not os.path.lexists(link):
                os.symlink(where, link)
            continue
        if directory not in found:
            found.append(directory)
    return found


def _row_env(tmp_path, *,
             mitos_on_path: Optional[str] = None) -> Tuple[Dict[str, str], str]:
    """Builds a row's one environment and its private ``bin/``; returns both.

    Args:
        tmp_path: The test's own directory; the registry must live under it.
        mitos_on_path: If given, ``bin/mitos`` is a symlink to it.
    """
    env = dict(os.environ)
    root = os.path.realpath(str(tmp_path))
    for name in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
        value = env.get(name)
        assert value and os.path.realpath(value).startswith(root + os.sep), (
            f"{name}={value!r} is not under this test's tmp_path; a real `init` "
            f"would register into the machine's registry")
    bin_dir = os.path.join(str(tmp_path), "bin")
    os.makedirs(bin_dir, exist_ok=True)
    tools = os.path.join(str(tmp_path), "tools")
    path_dirs = [bin_dir] + _tool_dirs(tools)
    if os.path.isdir(tools):
        path_dirs.insert(1, tools)
    # git also prepends its exec-path to a hook's PATH.
    exec_path = subprocess.run(["git", "--exec-path"], stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, check=True,
                               timeout=_TIMEOUT).stdout.strip()
    for directory in path_dirs + [exec_path]:
        if directory != bin_dir:
            assert not os.path.lexists(os.path.join(directory, "mitos")), directory
    if mitos_on_path is not None:
        os.symlink(mitos_on_path, os.path.join(bin_dir, "mitos"))
    env["PATH"] = os.pathsep.join(path_dirs)
    env["ANTHROPIC_API_KEY"] = _DUMMY_JUDGE_KEY
    env["QDRANT_URL"] = _DEAD_QDRANT
    env.pop("GEMINI_API_KEY", None)
    env.pop("GOOGLE_API_KEY", None)
    return env, bin_dir


# --------------------------------------------------------------------------- #
# Processes
# --------------------------------------------------------------------------- #

def _run(argv: List[str], env: Dict[str, str], cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(argv, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=_TIMEOUT)


def _show(result: subprocess.CompletedProcess) -> str:
    return (f"exit {result.returncode}\n--- stdout\n{result.stdout}"
            f"--- stderr\n{result.stderr}")


def _mitos(binary: str, env: Dict[str, str], cwd: str,
           *args: str) -> subprocess.CompletedProcess:
    return _run([binary, *args], env, cwd)


def _commit(repo: str, env: Dict[str, str], *args: str) -> subprocess.CompletedProcess:
    """A real ``git commit`` under the row's env; never raises on the hook's exit."""
    return _run(["git", *_IDENTITY, "commit", "-q", *args], env, repo)


def _head(repo: str) -> str:
    return scratch_git(repo, "rev-parse", "HEAD").strip()


def _stage(repo: str, name: str, content: str) -> None:
    """Writes one file and stages it by name: the workspace's files stay untracked."""
    with open(os.path.join(repo, name), "w", encoding="utf-8") as f:
        f.write(content)
    scratch_git(repo, "add", "--", name)


def _init(binary: str, env: Dict[str, str], workspace: str, *args: str) -> None:
    os.makedirs(workspace, exist_ok=True)
    result = _mitos(binary, env, workspace, "init", *args)
    assert result.returncode == 0, _show(result)


def _record(binary: str, env: Dict[str, str], selector: str, slug: str) -> None:
    """``record -p <selector>`` from the test's cwd (``tmp_path``), offline."""
    result = _mitos(binary, env, os.getcwd(), "record", "-p", selector, "--slug", slug,
                    f"The {slug} axiom holds for the scratch corpus.",
                    "--rejected", f"The obvious alternative to {slug} was weighed.")
    assert result.returncode == 0, _show(result)


def _install(binary: str, env: Dict[str, str], cwd: str,
             selector: str) -> Tuple[subprocess.CompletedProcess, Optional[str]]:
    """``hook-install -p <selector>``; returns the result and the hook path it printed."""
    result = _mitos(binary, env, cwd, "hook-install", "-p", selector)
    prefix = "Wrote the commit gate hook to "
    for line in result.stdout.splitlines():
        if line.startswith(prefix) and line.endswith("."):
            return result, line[len(prefix):-1]
    return result, None


def _block_recipe(stderr: str) -> List[str]:
    """The first backticked command in the block message, split as a shell would."""
    at = stderr.index(_BLOCK_WORDS)
    start = stderr.index("`", at) + 1
    return shlex.split(stderr[start:stderr.index("`", start)])


def _extract_block(stdout: str) -> str:
    """The pasted block exactly as printed: marker line through the end line."""
    start = stdout.index(HOOK_BLOCK_MARKER)
    end = stdout.index(_BLOCK_END, start) + len(_BLOCK_END)
    return stdout[start:end] + "\n"


def _assert_blocked(step: str, result: subprocess.CompletedProcess, repo: str,
                    head_before: str, uncovered: int) -> None:
    subject = ("1 decision has" if uncovered == 1
               else f"{uncovered} decisions have")
    assert f"{subject} {_BLOCK_WORDS}" in result.stderr, f"{step}: {_show(result)}"
    assert result.returncode != 0, f"{step}: {_show(result)}"
    assert _head(repo) == head_before, f"{step}: HEAD moved under a block"


def _assert_passed(step: str, result: subprocess.CompletedProcess, repo: str,
                   head_before: str) -> None:
    assert result.returncode == 0, f"{step}: {_show(result)}"
    assert _head(repo) != head_before, f"{step}: HEAD did not move"
    assert _BLOCK_WORDS not in result.stderr, f"{step}: {_show(result)}"


def _assert_silent_pass(step: str, result: subprocess.CompletedProcess, repo: str,
                        head_before: str) -> None:
    _assert_passed(step, result, repo, head_before)
    assert _GATE_LINE not in result.stderr, f"{step}: {_show(result)}"
    assert result.stderr == "", f"{step}: {_show(result)}"


def _keyed_config(monkeypatch: pytest.MonkeyPatch, workspace: str) -> MitosConfig:
    """The in-process config under the rows' dummy key (set before it is built)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", _DUMMY_JUDGE_KEY)
    return MitosConfig(workspace)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

@dataclass
class _Gated:
    """A scratch repository whose workspace is installed, recorded, and blocked."""

    repo: str
    workspace: str
    hook_file: str
    env: Dict[str, str]
    bin_dir: str
    head: str


def _gated_and_blocked(tmp_path, binary: str, *, name: str = "gated-ws",
                       mitos_on_path: Optional[str] = None) -> _Gated:
    """``init`` → ``hook-install`` → ``record`` → a staged change, shown blocked.

    The blocked commit runs the unmodified installed hook under the same ``PATH``
    the row's variant will get, so the pass that follows is a transition.
    """
    env, bin_dir = _row_env(tmp_path, mitos_on_path=mitos_on_path)
    repo = make_scratch_repo(str(tmp_path / "repo"))
    workspace = os.path.join(repo, name)
    _init(binary, env, workspace)
    installed, hook_file = _install(binary, env, str(tmp_path), workspace)
    assert installed.returncode == 0 and hook_file, _show(installed)
    _record(binary, env, name, "first-rule")
    _stage(repo, "f", "one\n")
    head = _head(repo)
    _assert_blocked("shown blocked first", _commit(repo, env, "-m", "held"), repo,
                    head, 1)
    return _Gated(repo, workspace, hook_file, env, bin_dir, head)


def _write_hook(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(path, 0o755)


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


# --------------------------------------------------------------------------- #
# The frame
# --------------------------------------------------------------------------- #

def test_the_binary_under_test_is_this_checkouts_console_script(tmp_path) -> None:
    """Criterion 1: the identity guard, as its own row so it is visible and breakable."""
    path = _branch_mitos_path()
    assert _identity_problem(path, str(tmp_path)) is None
    # The guard can say no: a script whose interpreter imports another tree fails.
    other = tmp_path / "other"
    (other / "mitos").mkdir(parents=True)
    (other / "mitos" / "__init__.py").write_text("", encoding="utf-8")
    interpreter = _script_interpreter(path)
    assert interpreter is not None
    fake = tmp_path / "fake-mitos"
    fake.write_text(f"#!{interpreter}\nraise SystemExit(0)\n", encoding="utf-8")
    fake.chmod(0o755)
    assert "not this checkout" in (_identity_problem(str(fake), str(other)) or "")
    assert "no console script" in (_identity_problem(str(tmp_path / "gone"),
                                                      str(tmp_path)) or "")
    # pip's long-or-spaced form: `#!/bin/sh` plus an `'''exec'` trampoline. Reached
    # through a directory with a space, the guard still finds the interpreter.
    # The prefix is linked whole, so a venv still finds its `pyvenv.cfg`.
    spaced_prefix = tmp_path / "py prefix"
    os.symlink(os.path.dirname(os.path.dirname(interpreter)), spaced_prefix)
    spaced = str(spaced_prefix / "bin" / os.path.basename(interpreter))
    trampoline = tmp_path / "trampoline-mitos"
    trampoline.write_text(
        f"#!/bin/sh\n'''exec' \"{spaced}\" \"$0\" \"$@\"\n' '''\n",
        encoding="utf-8")
    trampoline.chmod(0o755)
    assert _script_interpreter(str(trampoline)) == spaced
    assert _identity_problem(str(trampoline), str(tmp_path)) is None


# --------------------------------------------------------------------------- #
# The story
# --------------------------------------------------------------------------- #

@_needs_git
def test_the_story_holds_a_real_commit_until_the_printed_check_is_attempted(
        tmp_path, monkeypatch, branch_mitos) -> None:
    """Criteria 2–8: install → pass → record → blocked → recipe → pass → blocked → amend."""
    env, _bin = _row_env(tmp_path)
    repo = make_scratch_repo(str(tmp_path / "repo"))
    workspace = os.path.join(repo, "story-ws")
    _init(branch_mitos, env, workspace)

    # Criterion 2: the installed hook runs at all, over a fresh graph (row 3).
    installed, hook_file = _install(branch_mitos, env, str(tmp_path), workspace)
    assert installed.returncode == 0 and hook_file, f"criterion 2: {_show(installed)}"
    assert "The gate is armed, and no decision is uncovered." in installed.stdout
    assert os.access(hook_file, os.X_OK), "criterion 2: the hook is not executable"
    script = _read(hook_file)
    assert f"if [ -x {shlex.quote(branch_mitos)} ]; then" in script.splitlines(), script
    assert os.path.realpath(hook_script_workspace(script)) == os.path.realpath(workspace)
    config = _keyed_config(monkeypatch, workspace)
    assert evaluate_gate(config).row == GATE_NOTHING_UNCOVERED
    _stage(repo, "f", "one\n")
    head = _head(repo)
    _assert_silent_pass("criterion 2 (empty corpus)", _commit(repo, env, "-m", "one"),
                        repo, head)

    # Criteria 3 and 8: one record blocks; the installed `hook-run -p …` parsed.
    _record(branch_mitos, env, "story-ws", "first-rule")
    _stage(repo, "f", "two\n")
    head = _head(repo)
    blocked = _commit(repo, env, "-m", "two")
    _assert_blocked("criterion 3", blocked, repo, head, 1)
    lines = blocked.stderr.splitlines()
    assert lines[0].startswith("corpus: story-ws"), f"criterion 3: {_show(blocked)}"
    assert _BLOCK_WORDS in lines[1], f"criterion 3: {_show(blocked)}"
    assert "usage:" not in blocked.stderr, f"criterion 8: {_show(blocked)}"

    # Criterion 4: the block's own recipe, run as printed, is an attempt.
    recipe = _block_recipe(blocked.stderr)
    assert recipe[0] == branch_mitos, f"criterion 4: recipe {recipe!r}"
    assert recipe[1:3] == ["check", "-p"], f"criterion 4: recipe {recipe!r}"
    assert os.path.realpath(routing.resolve_project(recipe[3]).root) == \
        os.path.realpath(workspace), f"criterion 4: {recipe[3]!r} names another corpus"
    attempted = _run(recipe, env, str(tmp_path))
    assert attempted.returncode == 2, f"criterion 4: {_show(attempted)}"
    assert "check could not run: cannot audit" in attempted.stdout + attempted.stderr, \
        f"criterion 4: {_show(attempted)}"
    last = read_last_attempt(config.telemetry_path)
    assert isinstance(last, LastAttempt) and last.state == ATTEMPT_STARTED, last
    assert evaluate_gate(config).row == GATE_ATTEMPTED

    # Criterion 5: the same staged change now passes, in silence (row 4).
    _assert_silent_pass("criterion 5", _commit(repo, env, "-m", "two"), repo, head)

    # Criterion 6: a second record blocks a new change again, naming two.
    _record(branch_mitos, env, "story-ws", "second-rule")
    _stage(repo, "f", "three\n")
    head = _head(repo)
    _assert_blocked("criterion 6", _commit(repo, env, "-m", "three"), repo, head, 2)

    # Criterion 7: `--amend` runs the hook too, and is held.
    _assert_blocked("criterion 7", _commit(repo, env, "--amend", "-m", "amended"),
                    repo, head, 2)


@_needs_git
def test_no_verify_passes_over_a_standing_block(tmp_path, branch_mitos) -> None:
    """Stretch (b): the human bypass SETUP.md names, over the same held change."""
    gated = _gated_and_blocked(tmp_path, branch_mitos)
    result = _commit(gated.repo, gated.env, "--no-verify", "-m", "bypassed")
    _assert_silent_pass("--no-verify", result, gated.repo, gated.head)


# --------------------------------------------------------------------------- #
# Fail-open classes (each a transition from the block above)
# --------------------------------------------------------------------------- #

def _without_the_baked_executable(gated: _Gated, binary: str) -> str:
    """The installed script re-rendered with the baked path gone; nothing else differs."""
    original = _read(gated.hook_file)
    variant = render_hook_script(command=["/nonexistent/mitos"],
                                 workspace=hook_script_workspace(original),
                                 hook_file=gated.hook_file)
    assert original.count(shlex.quote(binary)) == 2, original
    assert original.replace(shlex.quote(binary), "/nonexistent/mitos") == variant
    return variant


@_needs_git
def test_with_no_mitos_reachable_the_commit_passes_silently(tmp_path,
                                                             branch_mitos) -> None:
    """Criterion 9: baked executable gone and an empty ``bin/`` → the script's ``exit 0``."""
    gated = _gated_and_blocked(tmp_path, branch_mitos)
    assert os.listdir(gated.bin_dir) == []
    _write_hook(gated.hook_file, _without_the_baked_executable(gated, branch_mitos))
    _assert_silent_pass("criterion 9", _commit(gated.repo, gated.env, "-m", "passes"),
                        gated.repo, gated.head)


@_needs_git
def test_a_missing_baked_executable_falls_back_to_the_mitos_on_path(
        tmp_path, branch_mitos) -> None:
    """Stretch (a): the same variant with ``bin/mitos`` → this build still blocks."""
    gated = _gated_and_blocked(tmp_path, branch_mitos, mitos_on_path=branch_mitos)
    _write_hook(gated.hook_file, _without_the_baked_executable(gated, branch_mitos))
    blocked = _commit(gated.repo, gated.env, "-m", "held")
    _assert_blocked("fallback", blocked, gated.repo, gated.head, 1)
    assert _block_recipe(blocked.stderr)[0] == "mitos", _show(blocked)


@_needs_git
def test_an_argparse_failure_passes_and_prints_usage(tmp_path, branch_mitos) -> None:
    """Criterion 10: the verb misspelled → argparse's exit 2 → the commit passes."""
    gated = _gated_and_blocked(tmp_path, branch_mitos)
    original = _read(gated.hook_file)
    assert original.count(" hook-run -p ") == 1, original
    _write_hook(gated.hook_file, original.replace(" hook-run -p ", " hook-runx -p "))
    result = _commit(gated.repo, gated.env, "-m", "passes")
    _assert_passed("criterion 10", result, gated.repo, gated.head)
    # The documented residue (3c2, 3e ADR rejected path 4): usage on every commit.
    assert "usage:" in result.stderr, _show(result)


@_needs_git
def test_an_import_failure_passes(tmp_path, branch_mitos) -> None:
    """Criterion 11: a ``PYTHONPATH`` shadow of the package → ``ImportError`` → a pass."""
    gated = _gated_and_blocked(tmp_path, branch_mitos)
    shadow = tmp_path / "shadow" / "mitos"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text(
        'raise ImportError("planted shadow of the mitos package")\n', encoding="utf-8")
    env = dict(gated.env, PYTHONPATH=str(tmp_path / "shadow"))
    # The control: the shadow outranks the editable install, or this row is vacuous.
    control = _mitos(branch_mitos, env, str(tmp_path), "--version")
    assert control.returncode == 1 and "planted shadow" in control.stderr, \
        f"the PYTHONPATH shadow did not take: {_show(control)}"
    result = _commit(gated.repo, env, "-m", "passes")
    _assert_passed("criterion 11", result, gated.repo, gated.head)
    assert "ImportError" in result.stderr and "planted shadow" in result.stderr, \
        _show(result)


# --------------------------------------------------------------------------- #
# The pasted block (constraint 17)
# --------------------------------------------------------------------------- #

@dataclass
class _Pasted:
    """A repository with an in-tree ``core.hooksPath`` and the printed block pasted in."""

    repo: str
    workspace: str
    host: str
    env: Dict[str, str]
    bin_dir: str
    head: str
    blocked: subprocess.CompletedProcess


def _host_ran(repo: str) -> bool:
    return os.path.exists(os.path.join(repo, _HOST_LOG))


def _pasted_and_blocked(tmp_path, binary: str) -> _Pasted:
    """Criterion 13's setup: refused install, block extracted and pasted, then blocked."""
    env, bin_dir = _row_env(tmp_path, mitos_on_path=binary)
    repo = make_scratch_repo(str(tmp_path / "repo"), hooks_path=".githooks")
    workspace = os.path.join(repo, "subdir")
    _init(binary, env, workspace, "--name", "gated-sub")

    refused, written = _install(binary, env, str(tmp_path), "gated-sub")
    assert refused.returncode == 1 and written is None, _show(refused)
    assert "paste this block into its pre-commit:" in refused.stderr, _show(refused)
    assert not os.path.exists(os.path.join(repo, ".githooks")), "something was written"
    assert not os.path.exists(os.path.join(repo, ".git", "hooks", "pre-commit"))
    block = _extract_block(refused.stdout)
    assert "-p ./subdir" in block, block

    os.makedirs(os.path.join(repo, ".githooks"))
    host = os.path.join(repo, ".githooks", "pre-commit")
    _write_hook(host, "#!/bin/sh\nset -eu\n" + block
                + f"printf 'host ran\\n' >> {_HOST_LOG}\n")
    _record(binary, env, "gated-sub", "pasted-rule")
    _stage(repo, "f", "one\n")
    head = _head(repo)
    blocked = _commit(repo, env, "-m", "held")
    _assert_blocked("criterion 13", blocked, repo, head, 1)
    assert not _host_ran(repo), "criterion 13: the block's exit 3 did not end the host"
    return _Pasted(repo, workspace, host, env, bin_dir, head, blocked)


@_needs_git
def test_the_pasted_block_holds_a_real_commit_from_a_set_eu_host(
        tmp_path, branch_mitos) -> None:
    """Criterion 13, plus stretch (c): its bare recipe, run from ``PATH``, opens it."""
    pasted = _pasted_and_blocked(tmp_path, branch_mitos)
    recipe = _block_recipe(pasted.blocked.stderr)
    # D3: with `bin/mitos` → this build on PATH, the running-build rule says bare.
    assert recipe[:3] == ["mitos", "check", "-p"], recipe
    assert os.path.realpath(routing.resolve_project(recipe[3]).root) == \
        os.path.realpath(pasted.workspace)
    assert shutil.which("mitos", path=pasted.env["PATH"]) == \
        os.path.join(pasted.bin_dir, "mitos")
    attempted = _run(recipe, pasted.env, str(tmp_path))
    assert attempted.returncode == 2, _show(attempted)
    result = _commit(pasted.repo, pasted.env, "-m", "passes")
    _assert_silent_pass("stretch (c)", result, pasted.repo, pasted.head)
    assert _host_ran(pasted.repo), "the host's trailing line did not run on a pass"


@_needs_git
def test_the_pasted_block_without_mitos_on_path_passes_and_the_host_runs_on(
        tmp_path, branch_mitos) -> None:
    """Criterion 12: the same host with an empty ``bin/`` → silent pass, host line ran."""
    pasted = _pasted_and_blocked(tmp_path, branch_mitos)
    os.remove(os.path.join(pasted.bin_dir, "mitos"))
    assert shutil.which("mitos", path=pasted.env["PATH"]) is None
    result = _commit(pasted.repo, pasted.env, "-m", "passes")
    _assert_silent_pass("criterion 12", result, pasted.repo, pasted.head)
    assert _host_ran(pasted.repo), "criterion 12: the host's trailing line did not run"


@_needs_git
def test_the_block_spelled_by_name_resolves_nobody_and_lets_the_commit_through(
        tmp_path, branch_mitos) -> None:
    """Criterion 14 (D6): ``-p subdir`` is a name, the registry says ``gated-sub``."""
    pasted = _pasted_and_blocked(tmp_path, branch_mitos)
    host = _read(pasted.host)
    assert host.count("-p ./subdir") == 1, host
    _write_hook(pasted.host, host.replace("-p ./subdir", "-p subdir"))
    result = _commit(pasted.repo, pasted.env, "-m", "passes")
    _assert_passed("criterion 14", result, pasted.repo, pasted.head)
    assert "no project named 'subdir' is registered" in result.stderr, _show(result)
    assert _host_ran(pasted.repo), "criterion 14: the host's trailing line did not run"


# --------------------------------------------------------------------------- #
# The stale hook
# --------------------------------------------------------------------------- #

@_needs_git
def test_a_moved_workspace_passes_and_the_hook_names_itself(tmp_path,
                                                            branch_mitos) -> None:
    """Criterion 15: ``mv`` the workspace → one line naming the printed hook file."""
    gated = _gated_and_blocked(tmp_path, branch_mitos, name="below-root")
    baked = hook_script_workspace(_read(gated.hook_file))
    os.rename(gated.workspace, os.path.join(gated.repo, "moved-away"))
    result = _commit(gated.repo, gated.env, "-m", "passes")
    _assert_passed("criterion 15", result, gated.repo, gated.head)
    gate_lines = [line for line in result.stderr.splitlines() if _GATE_LINE in line]
    assert len(gate_lines) == 1, _show(result)
    line = gate_lines[0]
    assert f"the hook {gated.hook_file!r} gates {baked!r}, which is no longer a Mitos " \
        f"workspace" in line, line
    assert "install the hook again" in line and f"or delete {gated.hook_file!r}" in line


# --------------------------------------------------------------------------- #
# The `-m` form
# --------------------------------------------------------------------------- #

@_needs_git
def test_a_dash_m_install_holds_a_real_commit(tmp_path, branch_mitos) -> None:
    """Stretch (e): ``python -m mitos hook-install`` bakes the interpreter and blocks."""
    env, _bin = _row_env(tmp_path)
    interpreter = _script_interpreter(branch_mitos)
    assert interpreter is not None, _read(branch_mitos)
    repo = make_scratch_repo(str(tmp_path / "repo"))
    workspace = os.path.join(repo, "dash-m")
    _init(branch_mitos, env, workspace)
    installed = _run([interpreter, "-m", "mitos", "hook-install", "-p", workspace], env,
                     str(tmp_path))
    assert installed.returncode == 0, _show(installed)
    hook_file = installed.stdout.split("Wrote the commit gate hook to ", 1)[1] \
        .split("\n", 1)[0][:-1]
    assert " -m mitos" in _read(hook_file), _read(hook_file)
    _record(branch_mitos, env, "dash-m", "dash-m-rule")
    _stage(repo, "f", "one\n")
    head = _head(repo)
    blocked = _commit(repo, env, "-m", "held")
    _assert_blocked("-m form", blocked, repo, head, 1)
    assert _block_recipe(blocked.stderr)[1:3] == ["-m", "mitos"], _show(blocked)
