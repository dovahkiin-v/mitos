"""The commit gate's predicate: has anyone at least tried to check these?

One local question, asked on every commit by ``mitos hook-run``: *has a
contradiction check been attempted since the set of uncovered decisions last
changed?* The gate runs no judge, spends nothing and touches no network. It never
judges a finding; it blocks only on forgetting, and any attempt opens it, even one
that failed. The installed hook maps the block code alone to a failed commit, so
the policy lives here and a policy change ships as a release, never as a rewrite
of installed hooks.

The five rows, first match wins (the vision's §4.2 table):

1. ``unreadable``: a store the predicate reads could not be read, or there is no
   graph yet. The commit passes; the verb prints one line.
2. ``keyless``: no judge key is configured, so the gate is inactive. Passes
   silently.
3. ``nothing_uncovered``: every active decision is covered or excluded. Passes
   silently. An empty corpus lands here.
4. ``attempted``: the last attempt's fingerprint equals the current uncovered
   set's, whatever the attempt's state. Passes silently.
5. ``blocked``: decisions are uncovered and no attempt is on record for this set.

Evaluation order, which differs from the table's numbering in one place:

* **The key test is asked before any store is opened.** Row 1 covers a store the
  predicate *reads*, and a keyless predicate reads nothing. This keeps the hook in
  step with the status row (which reads a keyless workspace as inactive) and with
  the record notice (which asks the key test first so a keyless project opens no
  telemetry). For a keyless workspace with a broken store both rows pass the
  commit; the only difference is one stderr line, which a keyless project does
  not get.
* **The attempt record is read lazily**, only when the count is non-zero. Coverage
  and the attempt record share one file, so a coverage fault is caught at the
  derive; a fault that only the attempt read meets (a damaged JSON column) is
  still row 1, never row 5. A block over a record the gate cannot read would be
  the one fail-closed path the gate rules out.
* **Absent is not unreadable.** No telemetry file, or one below the attempt rung,
  holds no attempt, so it blocks. That is the first commit after an upgrade.

Fences:

* Imports ``config``, ``audit_debt`` and ``telemetry`` only (plus the standard
  library): no ``cli``, no ``_git``, no LLM SDK, no environment or
  working-directory read, no subprocess. A subprocess probe through the real verb
  pins that no SDK loads on any row. The one file read outside the stores is
  :func:`classify_hook_file`'s: an ``os.lstat`` and a read of one ``pre-commit``.
* No check-family module imports this one (``tests/test_conflict_closeout.py``'s
  exact-set closure row, via ``audit_debt``).
* Row 4 compares fingerprints only and imports no attempt-state constant: a
  refused spend, a ``started`` attempt that never reached its run end, and a state
  from a newer build all open the gate.
* No catch-all. Both readers return typed results on a file fault; a programming
  error propagates to the hook boundary (``cli._run_hook_boundary``), which prints
  one line naming its class and exits 1, which the installed hook treats as a
  pass. A catch-all here would turn a bug into a silent pass no row
  can see.

The hook ``mitos hook-install`` writes, and the block it prints when it may not
write, are rendered here too, beside the block code they interpolate, so the status
row (3f) can classify a ``pre-commit`` without importing the CLI. Locating the
repository and writing the file are the verb's (``cli.cmd_hook_install``); what the
script says, which file is mitos's, and whether a path can be baked are here, pure.

ADRs: ``telemetry-readers-outside-check-open-read-only-absent-is-empty-not-unreadable``,
``uncovered-decisions-derive-from-coverage-membership-not-created-at-timestamps``,
``unrecorded-check-attempt-is-named-on-the-report-the-hook-gets-no-writability-probe``,
``hook-install-writes-only-the-repos-own-git-dir-marker-means-whole-file``,
``repo-hook-serves-one-workspace-several-deferred-runtime-discovery-rejected``.
"""

import os
import shlex
import stat
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from mitos.audit_debt import AuditDebt, NoGraph, derive_audit_debt
from mitos.config import MitosConfig, judge_api_key
from mitos.telemetry import LastAttempt, TelemetryUnreadable, read_last_attempt

__all__ = [
    "CAUSE_GRAPH",
    "CAUSE_NO_GRAPH",
    "CAUSE_TELEMETRY",
    "GATE_ATTEMPTED",
    "GATE_BLOCKED",
    "GATE_CAUSES",
    "GATE_KEYLESS",
    "GATE_NOTHING_UNCOVERED",
    "GATE_ROWS",
    "GATE_UNREADABLE",
    "GateVerdict",
    "HOOK_ABSENT",
    "HOOK_BLOCK_EXIT",
    "HOOK_BLOCK_MARKER",
    "HOOK_FILE_KINDS",
    "HOOK_FILE_MARKER",
    "HOOK_FOREIGN",
    "HOOK_FOREIGN_WITH_BLOCK",
    "HOOK_OURS",
    "HOOK_UNREADABLE",
    "HookFileState",
    "classify_hook_file",
    "evaluate_gate",
    "hook_script_workspace",
    "render_hook_block",
    "render_hook_script",
    "unsafe_shell_path",
]

GATE_UNREADABLE = "unreadable"
GATE_KEYLESS = "keyless"
GATE_NOTHING_UNCOVERED = "nothing_uncovered"
GATE_ATTEMPTED = "attempted"
GATE_BLOCKED = "blocked"
#: The five rows, in the vision's table order (not the evaluation order).
GATE_ROWS: Tuple[str, ...] = (
    GATE_UNREADABLE,
    GATE_KEYLESS,
    GATE_NOTHING_UNCOVERED,
    GATE_ATTEMPTED,
    GATE_BLOCKED,
)

CAUSE_NO_GRAPH = "no_graph"
CAUSE_GRAPH = "graph"
CAUSE_TELEMETRY = "telemetry"
#: Row 1's causes. ``graph`` and ``telemetry`` are ``DebtUnreadable.source``'s values.
GATE_CAUSES: Tuple[str, ...] = (CAUSE_NO_GRAPH, CAUSE_GRAPH, CAUSE_TELEMETRY)

#: The exit status that blocks a commit, and the only one the installed hook maps
#: to a failure. Chosen outside every status something else can produce: 0; 1
#: (Python's uncaught exception, ``main()``'s fault arms, and the hook boundary's
#: unexpected-exception arm); 2 (argparse's usage
#: error); 120 (the interpreter failing to flush stdout at exit); 124–127
#: (``timeout``, ``env``, and the shell's not-executable / not-found); >= 128
#: (signals). Pinned against that set by a row.
HOOK_BLOCK_EXIT = 3


@dataclass(frozen=True)
class GateVerdict:
    """What the commit gate decided, and the facts it decided on.

    Attributes:
        row: One of :data:`GATE_ROWS`.
        uncovered: N, the uncovered count, when it was read (rows 3–5); else
            ``None``.
        cause: Row 1 only: one of :data:`GATE_CAUSES`; else ``None``.
    """

    row: str
    uncovered: Optional[int] = None
    cause: Optional[str] = None

    @property
    def blocks(self) -> bool:
        """Whether this verdict blocks the commit."""
        return self.row == GATE_BLOCKED


def evaluate_gate(config: MitosConfig) -> GateVerdict:
    """Evaluates the commit gate's five rows for a workspace. Writes nothing.

    Args:
        config: The target workspace's config. The key comes off its resolved
            ``env``; the stores are its ``db_path`` and ``telemetry_path``.

    Returns:
        The first matching row's :class:`GateVerdict`. Never raises on a file
        fault; a programming error propagates.
    """
    if judge_api_key(config) is None:
        return GateVerdict(GATE_KEYLESS)

    debt = derive_audit_debt(config.db_path, config.telemetry_path)
    if isinstance(debt, NoGraph):
        return GateVerdict(GATE_UNREADABLE, cause=CAUSE_NO_GRAPH)
    if not isinstance(debt, AuditDebt):
        return GateVerdict(GATE_UNREADABLE, cause=debt.source)
    if debt.uncovered == 0:
        return GateVerdict(GATE_NOTHING_UNCOVERED, uncovered=0)

    attempt = read_last_attempt(config.telemetry_path)
    if isinstance(attempt, TelemetryUnreadable):
        return GateVerdict(GATE_UNREADABLE, cause=CAUSE_TELEMETRY)
    # The fingerprint alone decides; the attempt's state is never consulted. None
    # (an empty table) and TelemetryAbsent both mean no attempt is on record.
    if isinstance(attempt, LastAttempt) and attempt.fingerprint == debt.fingerprint:
        return GateVerdict(GATE_ATTEMPTED, uncovered=debt.uncovered)
    return GateVerdict(GATE_BLOCKED, uncovered=debt.uncovered)


# --- the installed hook and the pasted block ---------------------------------

#: The whole-file marker: mitos wrote this entire file, and ``hook-install`` may
#: overwrite it whole. It counts only as the file's second line (line 1 is the
#: shebang), so a person's script with a mitos hook pasted into its middle is not
#: mistaken for one mitos owns.
HOOK_FILE_MARKER = ("# mitos-hook: mitos hook-install wrote this whole file "
                    "and overwrites it whole.")
#: The pasted block's marker. It means a person put the gate into their own
#: script; it never licenses a write. Neither marker contains the other.
HOOK_BLOCK_MARKER = ("# mitos-gate-block: pasted by hand; mitos never edits "
                     "the file that holds it.")

HOOK_ABSENT = "absent"
HOOK_OURS = "ours"
HOOK_FOREIGN = "foreign"
HOOK_FOREIGN_WITH_BLOCK = "foreign_with_block"
HOOK_UNREADABLE = "unreadable"
#: What a ``pre-commit`` path can hold. Only ``absent`` and ``ours`` may be written.
HOOK_FILE_KINDS: Tuple[str, ...] = (HOOK_ABSENT, HOOK_OURS, HOOK_FOREIGN,
                                    HOOK_FOREIGN_WITH_BLOCK, HOOK_UNREADABLE)

# The status the rendered shell compares against, spelled from the constant.
_BLOCK = str(HOOK_BLOCK_EXIT)


@dataclass(frozen=True)
class HookFileState:
    """What a ``pre-commit`` path holds, as far as mitos may act on it.

    Attributes:
        kind: One of :data:`HOOK_FILE_KINDS`.
        served_workspace: For ``ours`` only, the workspace the script gates, as
            :func:`hook_script_workspace` reads it back; ``None`` when unreadable
            or for any other kind.
    """

    kind: str
    served_workspace: Optional[str] = None


def unsafe_shell_path(path: str) -> Optional[str]:
    """Says why a path cannot be baked into a hook, or ``None`` when it can.

    Quoting handles spaces, quotes, ``$`` and backticks. It does not make a
    newline safe to print or to read back line by line, so any control character
    is refused, and so is a path that does not encode as UTF-8 (a name decoded
    with ``surrogateescape``). Pure: ``isabs`` reads no working directory.

    Args:
        path: The path to bake.

    Returns:
        ``None`` when safe; otherwise ``"not absolute"``, ``"control character"``
        or ``"not UTF-8"``.
    """
    if not os.path.isabs(path):
        return "not absolute"
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in path):
        return "control character"
    try:
        path.encode("utf-8")
    except UnicodeEncodeError:
        return "not UTF-8"
    return None


def _quoted(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def render_hook_script(*, command: Sequence[str], workspace: str,
                       hook_file: str) -> str:
    """Renders the whole-file ``pre-commit`` script ``hook-install`` writes.

    The script fails a commit on :data:`HOOK_BLOCK_EXIT` alone. Every other status
    ``hook-run`` can end with (a fault, a crash, a usage error, a missing or
    unrunnable program, a signal) exits 0: a gate that fails closed on its own
    faults is the one that gets bypassed. A missing baked executable falls back to
    the ``mitos`` on ``PATH``; with neither, the commit passes silently. Every
    baked value is shell-quoted, and the script passes its own path as
    ``--hook-file`` so a moved workspace names this file.

    This file may ``exit 0`` early; the pasted block may not, so the two are
    separate renderers (:func:`render_hook_block`). Tests are ``if`` conditions
    here too, so neither idiom can drift into the other's bug.

    Args:
        command: The argv prefix that runs this build (``[exe]`` or
            ``[python, "-m", "mitos"]``); every element already checked by
            :func:`unsafe_shell_path` where it is a path.
        workspace: The served workspace's absolute directory.
        hook_file: This script's own absolute path.

    Returns:
        The script text, ``\n``-terminated.
    """
    lines = [
        "#!/bin/sh",
        HOOK_FILE_MARKER,
        "# The mitos commit gate. Remove it by deleting this file.",
        f"if [ -x {shlex.quote(command[0])} ]; then",
        f"  set -- {_quoted(command)}",
        "elif command -v mitos >/dev/null 2>&1; then",
        "  set -- mitos",
        "else",
        "  exit 0",
        "fi",
        "mitos_gate_rc=0",
        (f'"$@" hook-run -p {shlex.quote(workspace)} '
         f"--hook-file {shlex.quote(hook_file)} || mitos_gate_rc=$?"),
        f'if [ "$mitos_gate_rc" -eq {_BLOCK} ]; then',
        f"  exit {_BLOCK}",
        "fi",
        "exit 0",
    ]
    return "\n".join(lines) + "\n"


def render_hook_block(*, command: Optional[Sequence[str]], selector: str,
                      guard_dir: Optional[str]) -> str:
    """Renders the block a person pastes into a ``pre-commit`` mitos may not write.

    The block lives inside someone else's script, so it never exits on a pass,
    assigns every variable it reads (``set -u``), keeps every test inside an
    ``if`` condition, and ends on an ``if`` whose status is 0 when it does not
    block. The last point matters under ``set -e`` and as a host's final
    statement: a bare ``[ … ] && exit`` as the last command leaves status 1, which
    git reads as a failed hook, blocking every commit. The only ``exit`` is the
    block code inside the ``if`` that saw it. It passes no ``--hook-file``: this is
    not a file mitos wrote, and "delete the file" is no way out of someone's own
    script.

    Two forms:

    - **Absolute** (``command`` given, ``guard_dir`` ``None``): a foreign
      ``pre-commit`` holds the repository's own hooks directory. ``command`` is the
      baked argv, falling back to ``PATH``'s ``mitos``; ``selector`` is the absolute
      workspace.
    - **No machine path** (``command`` ``None``): the hooks directory is shared.
      Bare ``mitos``, guarded by ``command -v``; ``selector`` is the path-shaped
      relative spelling (``.`` or ``./sub/ws``), which ``hook-run`` resolves
      against the work-tree root git runs hooks from; ``guard_dir`` is
      ``<selector>/.mitos``, so a repository with no workspace there does nothing.

    Args:
        command: The baked argv prefix, or ``None`` for bare ``mitos``.
        selector: The unquoted ``-p`` value.
        guard_dir: The unquoted directory whose absence makes the block silent, or
            ``None``.

    Returns:
        The block text, ``\n``-terminated.
    """
    run = f"hook-run -p {shlex.quote(selector)} || mitos_gate_rc=$?"
    lines = [HOOK_BLOCK_MARKER, "mitos_gate_rc=0"]
    if command is not None:
        lines += [
            f"if [ -x {shlex.quote(command[0])} ]; then",
            f"  {_quoted(command)} {run}",
            "elif command -v mitos >/dev/null 2>&1; then",
            f"  mitos {run}",
            "fi",
        ]
    else:
        condition = "command -v mitos >/dev/null 2>&1"
        if guard_dir is not None:
            condition = f"[ -d {shlex.quote(guard_dir)} ] && {condition}"
        lines += [f"if {condition}; then", f"  mitos {run}", "fi"]
    lines += [
        f'if [ "$mitos_gate_rc" -eq {_BLOCK} ]; then',
        f"  exit {_BLOCK}",
        "fi",
        "# end of the mitos commit gate block",
    ]
    return "\n".join(lines) + "\n"


def hook_script_workspace(text: str) -> Optional[str]:
    """Reads the served workspace back out of a script :func:`render_hook_script` wrote.

    The inverse of the render: the ``hook-run -p <workspace>`` line, split the way
    ``sh`` would split it. Pure text; no file or cwd read.

    Args:
        text: The script's text.

    Returns:
        The workspace, byte-exact, or ``None`` when no line parses as the render's.
    """
    for line in text.split("\n"):
        if not line.startswith('"$@" hook-run -p '):
            continue
        try:
            tokens: List[str] = shlex.split(line)
        except ValueError:
            return None
        if len(tokens) >= 4 and tokens[:3] == ["$@", "hook-run", "-p"]:
            return tokens[3]
        return None
    return None


def classify_hook_file(path: str) -> HookFileState:
    """Says what a ``pre-commit`` path holds, reading it without following links.

    ``os.lstat`` comes first, so a symlink (dangling or not) is never read through
    and never mistaken for absent: a write would follow it into another file, or
    create the file it points at. Only a regular file is opened, so a FIFO cannot
    block the read. Then, in order: undecodable bytes are foreign; the whole-file
    marker as line 2 exactly is ``ours``; the block marker anywhere is
    ``foreign_with_block``; anything else is ``foreign``.

    Args:
        path: The absolute ``pre-commit`` path.

    Returns:
        The :class:`HookFileState`. ``unreadable`` when the path or the file could
        not be read (a permission, or a hooks path that is itself a file).
    """
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return HookFileState(HOOK_ABSENT)
    except OSError:
        return HookFileState(HOOK_UNREADABLE)
    if not stat.S_ISREG(st.st_mode):
        return HookFileState(HOOK_FOREIGN)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as handle:
            data = handle.read()
    except OSError:
        return HookFileState(HOOK_UNREADABLE)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return HookFileState(HOOK_FOREIGN)
    lines = text.split("\n")
    if len(lines) >= 2 and lines[1] == HOOK_FILE_MARKER:
        return HookFileState(HOOK_OURS, served_workspace=hook_script_workspace(text))
    if HOOK_BLOCK_MARKER in text:
        return HookFileState(HOOK_FOREIGN_WITH_BLOCK)
    return HookFileState(HOOK_FOREIGN)
