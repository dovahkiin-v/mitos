"""The one place in mitos that spawns a process: asking ``git`` where a workspace lives.

G1's commit gate needs three answers from git — is this workspace inside a work
tree, where is its top level, and where are its common git directory and hooks
directory. ``locate_repository`` answers all three with a single ``git rev-parse``
and returns either a ``GitLocation`` (every path absolute and realpath'd) or a
``NotAWorkTree`` naming one of two closed reasons. It states locations and makes
no judgement: whether a hooks directory is "ours" is the caller's policy, built
from ``is_within``.

Git guesses which repository it means; this module refuses to let it. The spawn
always runs with ``cwd`` set to the workspace's own directory — never the
caller's, which a subprocess would otherwise inherit — and with git's
repository-locating variables removed from its environment, because an inherited
``GIT_DIR`` overrides ``cwd`` silently (measured 2026-09-21: exit 0, the other
repository's answers). That is the same mis-aim mitos rules out for project
targeting (no cwd fallback, no env-var channel), arriving through git instead.

Process spawning is fenced to this module. ``tests/test_process_fence.py`` pins
statically that nothing else in ``mitos/`` spawns and that nothing in the MCP
server's import closure reaches this module, so the server's "no shell execution"
(P8) is a property rather than an accident. Callers therefore import it
function-locally, inside the verb that asks (``cli`` is the only sanctioned
importer): that keeps ``subprocess`` off every other verb's path, including
``hook-run``'s measured pass path, and out of the ``mitos serve`` process.

Everything here degrades and nothing raises on what git or the filesystem does.
A missing directory, a bare repository, a directory inside ``.git``, or output
that does not parse (a path containing a newline breaks the four-line answer) all
read as ``outside_work_tree``; a missing, unrunnable or hung ``git`` reads as
``git_unavailable``. The one raise is ``ValueError`` on a relative directory — a
programming error, since resolving it would read the process cwd.
"""

import os
import subprocess
from dataclasses import dataclass
from typing import Dict, Tuple, Union

GIT_TIMEOUT_SECONDS = 5.0

# Git's own "local repository" variables, which it clears before running a command
# in another repository (submodules). Copied verbatim from
# ``git rev-parse --local-env-vars`` on git 2.47.3 (2026-09-21); pinned rather than
# spawned at runtime, since the list has been stable for years and a second spawn
# per question would buy nothing. Everything else — PATH, HOME,
# GIT_CONFIG_GLOBAL/SYSTEM, GIT_CEILING_DIRECTORIES — passes through.
GIT_REPO_LOCATING_ENV: Tuple[str, ...] = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_OBJECT_DIRECTORY",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_GRAFT_FILE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_REPLACE_REF_BASE",
    "GIT_PREFIX",
    "GIT_SHALLOW_FILE",
    "GIT_COMMON_DIR",
)

REASON_GIT_UNAVAILABLE = "git_unavailable"
REASON_OUTSIDE_WORK_TREE = "outside_work_tree"
NOT_A_WORK_TREE_REASONS: Tuple[str, ...] = (REASON_GIT_UNAVAILABLE,
                                            REASON_OUTSIDE_WORK_TREE)

_REV_PARSE_ARGV = ["git", "rev-parse", "--is-inside-work-tree", "--show-toplevel",
                   "--git-common-dir", "--git-path", "hooks"]


@dataclass(frozen=True)
class GitLocation:
    """Where git says a workspace's repository lives; every path absolute and realpath'd.

    Attributes:
        top_level: The work tree's top level (``--show-toplevel``); for a linked
            worktree, the worktree itself.
        common_dir: The common git directory (``--git-common-dir``); for a linked
            worktree, the main repository's ``.git``.
        hooks_dir: The hooks directory (``--git-path hooks``), honouring
            ``core.hooksPath``. It may not exist on disk.
    """

    top_level: str
    common_dir: str
    hooks_dir: str


@dataclass(frozen=True)
class NotAWorkTree:
    """A typed negative: the workspace is not in a git work tree, as far as mitos can tell.

    Attributes:
        reason: One of ``NOT_A_WORK_TREE_REASONS``.
    """

    reason: str


def _git_env() -> Dict[str, str]:
    """Returns the process environment, read at call time, minus git's repository-locating variables."""
    return {key: value for key, value in os.environ.items()
            if key not in GIT_REPO_LOCATING_ENV}


def locate_repository(directory: str) -> Union[GitLocation, NotAWorkTree]:
    """Asks git, from ``directory`` itself, where its repository, common dir and hooks dir are.

    Args:
        directory: The workspace's own absolute directory (``config.workspace_dir``,
            or the directory ``init`` is initialising). Never the caller's cwd.

    Returns:
        A ``GitLocation`` when git answers that ``directory`` is inside a work tree;
        otherwise a ``NotAWorkTree`` whose reason is ``outside_work_tree`` (git ran
        and said no, the directory is missing, or the answer did not parse) or
        ``git_unavailable`` (git is missing, unrunnable, or timed out).

    Raises:
        ValueError: If ``directory`` is relative — resolving it would read the
            process cwd, the exact mis-aim this module exists to prevent.
    """
    if not os.path.isabs(directory):
        raise ValueError(f"locate_repository needs an absolute directory, got {directory!r}")
    # Checked before spawning: subprocess raises FileNotFoundError both for a
    # missing git and for a missing cwd, so only this order keeps them apart.
    if not os.path.isdir(directory):
        return NotAWorkTree(REASON_OUTSIDE_WORK_TREE)
    try:
        completed = subprocess.run(
            _REV_PARSE_ARGV,
            cwd=directory,
            env=_git_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return NotAWorkTree(REASON_GIT_UNAVAILABLE)
    if completed.returncode != 0:
        return NotAWorkTree(REASON_OUTSIDE_WORK_TREE)
    text = os.fsdecode(completed.stdout)
    if text.endswith("\n"):
        text = text[:-1]
    lines = text.split("\n")
    if len(lines) != 4 or lines[0] != "true" or not all(lines[1:]):
        return NotAWorkTree(REASON_OUTSIDE_WORK_TREE)
    # Relative answers are relative to the spawn's cwd, so they join to
    # ``directory``; ``join`` leaves an absolute answer alone.
    top_level, common_dir, hooks_dir = (
        os.path.realpath(os.path.join(directory, line)) for line in lines[1:]
    )
    return GitLocation(top_level=top_level, common_dir=common_dir, hooks_dir=hooks_dir)


def is_within(path: str, root: str) -> bool:
    """Reports whether ``path`` is ``root`` or lies beneath it, after resolving symlinks.

    Pure: no spawn. Both arguments must be absolute, for the same reason as
    ``locate_repository``'s.

    Args:
        path: The absolute path to test.
        root: The absolute directory it may lie within.

    Returns:
        True when the realpath of ``path`` equals or descends from the realpath of
        ``root``.

    Raises:
        ValueError: If either argument is relative.
    """
    if not (os.path.isabs(path) and os.path.isabs(root)):
        raise ValueError(f"is_within needs absolute paths, got {path!r} and {root!r}")
    real_path = os.path.realpath(path)
    real_root = os.path.realpath(root)
    return os.path.commonpath([real_path, real_root]) == real_root
