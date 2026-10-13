"""The single derivation of uncovered decisions: audit debt.

The quantity: the active decisions (``GraphStore.get_active_decision_ids``, the
same population ``check`` sweeps) that no undegraded check has ever looked at,
that is, that hold no row in telemetry's ``check_coverage`` table. Decisions a
sweep saw but could not audit (``excluded``) are counted beside it, never inside
it, so one poison node cannot hold the count above zero for ever.

A surface that reports this number reads it from here and never derives it
itself, so no two surfaces can drift apart. The result is one of three types, and never a false zero:

* :class:`AuditDebt`: the uncovered set, its counts and a membership fingerprint.
  Absent telemetry (no file, or one below the coverage rung) holds no coverage,
  so every active decision is uncovered (N = M).
* :class:`NoGraph`: there is no graph file, so there is nothing to cover.
* :class:`DebtUnreadable`: a file exists and could not be read.

Reading discipline: the graph first, read-only, and only if its file exists
(``GraphStore`` creates its directory even when read-only); then telemetry
through ``telemetry.read_coverage``. Two opens, each through the module that
owns its schema, never an ``ATTACH``. Nothing is created, migrated or written.

Fences, each pinned in ``tests/test_audit_debt.py`` or the check family's
closure pin:

* No check-family module imports this one, function-local imports included
  (``tests/test_conflict_closeout.py``'s exact-set closure row).
* Its ``mitos`` imports are ``store``, ``telemetry`` and ``errors`` only; a
  subprocess probe pins that it adds nothing to their closure.
* It reads no environment variable and no working directory: callers pass both
  paths.

ADRs: ``uncovered-decisions-derive-from-coverage-membership-not-created-at-timestamps``
and ``telemetry-readers-outside-check-open-read-only-absent-is-empty-not-unreadable``.
"""

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from typing import FrozenSet, Iterable

from mitos.errors import DatabaseError
from mitos.store import GraphStore
from mitos.telemetry import (
    TelemetryAbsent,
    TelemetryUnreadable,
    read_coverage,
)

__all__ = [
    "AuditDebt",
    "DebtUnreadable",
    "NoGraph",
    "derive_audit_debt",
    "uncovered_fingerprint",
]


def uncovered_fingerprint(node_ids: Iterable[str]) -> str:
    """Hashes a set of node ids by membership.

    ``sha256`` over the sorted ids joined by ``"\\n"``, as lowercase hex. Node ids
    are 64-char lowercase hex, so the separator cannot occur inside one. It hashes
    the set, never its size: a set that loses one decision and gains another keeps
    its counts and still gets a different fingerprint. The empty set hashes the
    empty string.

    The value is persisted and compared later, so this recipe is a serialization
    contract. It carries no version prefix: if the recipe ever changes, every
    stored fingerprint mismatches once, which reads as "the set changed", not as
    corruption. ``GraphStore.graph_fingerprint`` (a count and a timestamp over
    all nodes) is not a substitute.

    Args:
        node_ids: The ids to hash, in any order. Duplicates collapse.

    Returns:
        A 64-char lowercase hex digest.
    """
    joined = "\n".join(sorted(set(node_ids)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AuditDebt:
    """The active decisions no undegraded check has seen, with their counts.

    Invariant: ``uncovered + excluded <= total``; the rest are covered.

    Attributes:
        uncovered_ids: Active decisions with no coverage row.
        total: M, the number of active decisions.
        excluded: K, the number of active decisions marked excluded.
    """

    uncovered_ids: FrozenSet[str]
    total: int
    excluded: int

    @property
    def uncovered(self) -> int:
        """N, the number of uncovered active decisions."""
        return len(self.uncovered_ids)

    @property
    def fingerprint(self) -> str:
        """The membership fingerprint of :attr:`uncovered_ids`."""
        return uncovered_fingerprint(self.uncovered_ids)


@dataclass(frozen=True)
class NoGraph:
    """There is no graph file at the path, so there is nothing to cover.

    Different from an empty corpus, which is a built graph with zero decisions
    and reads as ``AuditDebt(frozenset(), 0, 0)``.
    """


@dataclass(frozen=True)
class DebtUnreadable:
    """A file exists and could not be read. Never a zero count.

    Attributes:
        source: ``"graph"`` or ``"telemetry"``. Diagnostic only; consumers treat
            both the same.
        detail: The underlying error message.
    """

    source: str
    detail: str


def derive_audit_debt(
    db_path: str, telemetry_path: str
) -> "AuditDebt | NoGraph | DebtUnreadable":
    """Derives the uncovered active decisions from the graph and the coverage table.

    Args:
        db_path: Path to ``graph.sqlite`` (typically ``config.db_path``).
        telemetry_path: Path to ``telemetry.sqlite`` (typically
            ``config.telemetry_path``). Not opened when the graph is missing.

    Returns:
        An :class:`AuditDebt`, :class:`NoGraph` when ``db_path`` does not exist,
        or :class:`DebtUnreadable` when either file exists and cannot be read.
        Never raises on a file fault; a programming error still propagates.
    """
    # Checked before any GraphStore is built: its constructor makes the parent
    # directory even when read-only, and a mode=ro connect on a missing file
    # raises, which would read as unreadable.
    if not os.path.exists(db_path):
        return NoGraph()
    try:
        active = GraphStore(db_path, read_only=True).get_active_decision_ids()
    except (sqlite3.Error, DatabaseError, OSError) as e:
        # A connect failure arrives wrapped (DatabaseError); a corrupt or
        # tableless file raises raw sqlite3 errors at the query.
        return DebtUnreadable(source="graph", detail=str(e))

    coverage = read_coverage(telemetry_path)
    if isinstance(coverage, TelemetryUnreadable):
        return DebtUnreadable(source="telemetry", detail=coverage.detail)
    if isinstance(coverage, TelemetryAbsent):
        covered: FrozenSet[str] = frozenset()
        excluded: FrozenSet[str] = frozenset()
    else:
        covered, excluded = coverage.covered, coverage.excluded

    # Subtract on sets and count at the end: the table keeps rows for nodes that
    # have since been superseded or no longer exist, and those drop out here.
    return AuditDebt(
        uncovered_ids=active - covered - excluded,
        total=len(active),
        excluded=len(active & excluded),
    )
