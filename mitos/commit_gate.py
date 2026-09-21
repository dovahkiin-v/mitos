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

* Imports ``config``, ``audit_debt`` and ``telemetry`` only: no ``cli``, no LLM
  SDK, no environment or working-directory read, no subprocess. A subprocess
  probe through the real verb pins that no SDK loads on any row.
* No check-family module imports this one (``tests/test_conflict_closeout.py``'s
  exact-set closure row, via ``audit_debt``).
* Row 4 compares fingerprints only and imports no attempt-state constant: a
  refused spend, a ``started`` attempt that never reached its run end, and a state
  from a newer build all open the gate.
* No catch-all. Both readers return typed results on a file fault; a programming
  error propagates to ``main()``'s boundary and exits 1, which the installed hook
  treats as a pass. A catch-all here would turn a bug into a silent pass no row
  can see.

ADRs: ``telemetry-readers-outside-check-open-read-only-absent-is-empty-not-unreadable``,
``uncovered-decisions-derive-from-coverage-membership-not-created-at-timestamps``,
``unrecorded-check-attempt-is-named-on-the-report-the-hook-gets-no-writability-probe``.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

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
    "HOOK_BLOCK_EXIT",
    "evaluate_gate",
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
#: (Python's uncaught exception, ``main()``'s fault arms); 2 (argparse's usage
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
