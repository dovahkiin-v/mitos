"""Forward-only SQLite migration ladder for Mitos.

This module defines the cross-vision schema-versioning primitive every later
vision extends: a step registry keyed by monotonically increasing integers,
applied through ``PRAGMA user_version`` and nothing else (vision §5.2.5). There
is no ``schema_version`` audit table — ``user_version`` is the atomic, in-header,
transactional version marker, so schema state cannot drift out of sync with a
parallel bookkeeping row.

Idempotent replay is *structural*, not remembered (Lesson 2): each step runs only
when the DB's ``user_version`` is strictly below the step's version, inside an
explicit transaction that bumps ``user_version`` on success. SQLite DDL is
transactional, so a crash mid-step rolls back the DDL *and* the version bump
together — there is no partial apply to repair (INVARIANTS.md MI-3).

The registry ships **empty** in V1a Phase 2a: the ladder mechanism is complete
and proven here with synthetic steps, but the first real rung (the V1a schema) is
registered by Phase 2b. The empty ladder boots clean as a no-op that leaves
``user_version`` at 0.
"""

import sqlite3
from typing import Callable, List, Tuple

from mitos.errors import DatabaseError

# A migration step pairs a target schema version with a function that issues its
# DDL against an open connection. The step function does NOT manage the
# transaction or touch ``user_version`` — ``run_migrations`` owns both, so the
# atomic "DDL + version bump roll back together" contract (MI-3) lives in exactly
# one place and a step author cannot accidentally break it (e.g. with
# ``executescript``, which force-commits — see ``run_migrations``).
MigrationStep = Tuple[int, Callable[[sqlite3.Connection], None]]

# The cross-vision step registry. EMPTY in V1a Phase 2a by design — the
# "empty-case-first-class" lever: the ladder boots clean as a no-op before any
# step exists, so Phase 2b is a near-pure addition (append step 1 + flip the live
# boot), not a scaffold-and-populate combo. V1b, V2, V3b and V6 each append a rung
# here; tests inject synthetic steps via ``run_migrations(conn, steps=...)``
# rather than mutating this registry.
MIGRATION_STEPS: List[MigrationStep] = []


def _get_user_version(conn: sqlite3.Connection) -> int:
    """Reads the database's current ``PRAGMA user_version`` header integer.

    Args:
        conn: An open SQLite connection.

    Returns:
        The current ``user_version`` (0 on a fresh database).
    """
    return conn.execute("PRAGMA user_version;").fetchone()[0]


def _validate_steps(steps: List[MigrationStep]) -> List[MigrationStep]:
    """Validates and ascending-sorts a step registry before it is applied.

    Guards the invariant that makes the ``PRAGMA user_version = N`` f-string
    interpolation safe (the P8 carve-out): every version is a code-internal,
    strictly-increasing positive integer — never user/LLM input. Sorting lets a
    registry list its rungs in any order while still applying deterministically;
    the post-sort strict-increase check is therefore a duplicate-version guard.

    Args:
        steps: The step registry to validate.

    Returns:
        The steps sorted by ascending version.

    Raises:
        DatabaseError: If a version is not an int, is non-positive, or a version
            is duplicated.
    """
    ordered = sorted(steps, key=lambda s: s[0])
    prev_version = 0
    for version, _step_fn in ordered:
        # ``bool`` is an ``int`` subclass but ``f"{True}"`` yields ``"True"``, not
        # a number — exclude it so the interpolation can only ever emit a digit.
        if not isinstance(version, int) or isinstance(version, bool):
            raise DatabaseError(
                f"Migration step version must be an int, got {version!r}."
            )
        if version <= 0:
            raise DatabaseError(
                f"Migration step version must be a positive integer, got {version}."
            )
        if version <= prev_version:
            raise DatabaseError(
                f"Migration step versions must be strictly increasing; "
                f"version {version} is duplicated or out of order."
            )
        prev_version = version
    return ordered


def run_migrations(
    conn: sqlite3.Connection,
    steps: List[MigrationStep] = MIGRATION_STEPS,
) -> int:
    """Advances the database to the head of ``steps``, applying each missing rung.

    The ladder is forward-only and idempotent. Each step whose version is strictly
    greater than the DB's current ``user_version`` is applied in ascending order,
    each inside its own explicit transaction that bumps ``user_version`` to the
    step's version on success. A step that raises rolls back its DDL *and* its
    version bump together (SQLite DDL is transactional), then the exception
    propagates — no partial apply is left to repair (MI-3). A prior step that
    already committed is unaffected: each rung is its own transaction.

    Replaying the ladder over an at-head database changes nothing and raises
    nothing — the V1a share of MI-3's idempotent-replay guarantee.

    The runner drives the transaction explicitly (``isolation_level = None`` plus
    ``BEGIN``/``COMMIT``/``ROLLBACK``) and runs each step's statements through
    ``conn.execute``. It deliberately does **not** use ``executescript``, which
    issues an implicit ``COMMIT`` before running and would break the atomic
    "DDL + version bump roll back together" contract above (Decision 4 / §7).

    Args:
        conn: An open, writable SQLite connection. The runner takes ownership of
            its transaction discipline and leaves it in autocommit mode
            (``isolation_level = None``); callers pass a connection dedicated to
            migration (``GraphStore.__init__`` opens one and closes it). Read-only
            stores must not call this — they never migrate.
        steps: The step registry to apply. Defaults to the module-level
            ``MIGRATION_STEPS`` (empty in V1a 2a); tests and future visions inject
            their own without mutating the shared registry.

    Returns:
        The resulting ``user_version`` after all applicable steps have run.

    Raises:
        DatabaseError: If a step version is not a positive, strictly-increasing
            integer.
        Exception: Re-raises whatever a failing step raised, after rolling that
            step's transaction back.
    """
    # Drive transactions ourselves: take the Python layer's implicit management
    # out of the loop so BEGIN/COMMIT/ROLLBACK are ours alone (Decision 4).
    conn.isolation_level = None

    ordered = _validate_steps(steps)
    current = _get_user_version(conn)

    for version, step_fn in ordered:
        if version <= current:
            continue  # already applied — the structural idempotency gate (MI-3)
        conn.execute("BEGIN;")
        try:
            step_fn(conn)
            # PRAGMA cannot be parameterized; ``version`` is a code-internal int
            # validated by ``_validate_steps`` above — never user/LLM input, so
            # the f-string is safe under P8 (a documented carve-out, not a breach).
            conn.execute(f"PRAGMA user_version = {version};")
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        current = version

    return current
