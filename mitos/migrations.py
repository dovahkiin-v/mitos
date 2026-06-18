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

The registry ships **empty** through V1a Phases 2a–4b: the ladder mechanism is
complete and proven with synthetic steps, and Phase 2b authors the first real rung
(the V1a schema, ``_v1_schema``) and proves it via injection — but its live
registration + boot-flip is deferred to Phase 5a (in lockstep with the
``commit_parsed_entry`` rebuild that writes the schema; WIRING_LEDGER entry-001).
So the registry stays empty and the empty ladder boots clean as a no-op that
leaves ``user_version`` at 0.
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

# The cross-vision step registry. EMPTY through V1a Phases 2a–4b by design — the
# "empty-case-first-class" lever: the ladder boots clean as a no-op before any
# step is live, so Phase 5a's flip is a near-pure addition (append step 1 + retire
# _init_db + wire the pre-V1a guard), not a scaffold-and-populate combo. Phase 2b
# authors step 1 (``_v1_schema``) but deliberately does NOT register it here — the
# live flip rides Phase 5a (WIRING_LEDGER entry-001). V1b, V2, V3b and V6 each
# append a rung here; tests inject synthetic steps via
# ``run_migrations(conn, steps=...)`` rather than mutating this registry.
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


# --- V1a schema (Phase 2b): migration ladder step 1 -------------------------
#
# The six STRICT tables of vision §8.2 — the shape every later phase (hashing,
# parser, commit engine, init, cutover) writes against. STRICT eliminates
# SQLite's permissive typing, so type discipline is a property of the tables, not
# of any Python check that could be forgotten (Lesson 2). The enum whitelists are
# CHECK constraints, referential integrity is FK-enforced, and the edge kind
# matrix is a table-level CHECK — a cross-kind or dangling kill-edge cannot even
# be inserted.
#
# Each entry is a SINGLE statement: ``conn.execute`` runs only the first
# statement in a string, so every CREATE TABLE and CREATE INDEX is its own list
# element. NEVER ``executescript`` — it force-commits and would split this DDL
# out of the runner's atomic version-bump transaction (2a Decision 4). No column
# carries ``DEFAULT CURRENT_TIMESTAMP``: every timestamp is application-supplied
# UTC ISO-8601 (MI-10). All identifiers are code-internal literals (no
# interpolation, P8). Table order is parent-first so ``nodes`` exists before any
# child's FK target is referenced.
_V1_SCHEMA_STATEMENTS: List[str] = [
    # nodes — content-hash identity; the parent every other table references.
    # ``UNIQUE (id, kind)`` is required (not redundant with the PK): the ``edges``
    # composite FKs reference ``nodes(id, kind)``, which needs an explicit unique
    # key covering exactly those columns or SQLite raises "foreign key mismatch"
    # at INSERT time (§7). ``idx_nodes_slug_casefold`` is NON-unique by design:
    # V1-D4's "one active node per casefold(slug)" is an application-layer
    # assertion (5b) because M3 forbids a DDL uniqueness predicate over the
    # kill-edge-filtered *active* view — a reflexive UNIQUE here would wrongly
    # reject a legitimate same-slug supersession.
    """
    CREATE TABLE IF NOT EXISTS nodes (
        id TEXT,
        kind TEXT NOT NULL,
        slug TEXT NOT NULL,
        slug_casefold TEXT NOT NULL,
        source TEXT NOT NULL,
        axiom TEXT,
        mechanism_refs_json TEXT,
        topic TEXT,
        questions_raised_json TEXT,
        rejected_paths_json TEXT,
        invalidates_if TEXT,
        context TEXT,
        confirmed_by TEXT,
        confirmed_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (id),
        UNIQUE (id, kind),
        CHECK (kind IN ('decision', 'open_question')),
        CHECK (source IN ('user', 'capture_llm', 'import_llm'))
    ) STRICT;
    """,
    "CREATE INDEX IF NOT EXISTS idx_nodes_slug_casefold ON nodes (slug_casefold);",
    # node_scopes — multi-valued scope tags (MI-9), one casefolded column (no
    # case-preserved twin). The PK is node_id-leading, so the C4 ``WHERE scope=?``
    # membership filter needs its own index (P11; SQLite won't reuse the PK for it).
    """
    CREATE TABLE IF NOT EXISTS node_scopes (
        node_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        PRIMARY KEY (node_id, scope),
        FOREIGN KEY (node_id) REFERENCES nodes(id)
    ) STRICT;
    """,
    "CREATE INDEX IF NOT EXISTS idx_node_scopes_scope ON node_scopes (scope);",
    # edges — the kind-matrix graph. The single table-level CHECK enforces both
    # same-kind-ness (permitting OQ->OQ, forbidding cross-kind) and the two-value
    # V1a whitelist; the composite FKs to nodes(id, kind) make a dangling or
    # cross-kind edge structurally impossible. ``idx_edges_target`` is the
    # incoming-kill-edge anti-join key (the PK is source_id-leading; SQLite does
    # not auto-index FK child columns, P11).
    """
    CREATE TABLE IF NOT EXISTS edges (
        source_id TEXT NOT NULL,
        source_kind TEXT NOT NULL,
        target_id TEXT NOT NULL,
        target_kind TEXT NOT NULL,
        edge_type TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (source_id, target_id, edge_type),
        CHECK (source_kind = target_kind AND edge_type IN ('supersedes', 'corrects')),
        FOREIGN KEY (source_id, source_kind) REFERENCES nodes(id, kind),
        FOREIGN KEY (target_id, target_kind) REFERENCES nodes(id, kind)
    ) STRICT;
    """,
    "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges (target_id, edge_type);",
    # transcripts — off-hot-path raw capture text, one row per node (PK serves).
    """
    CREATE TABLE IF NOT EXISTS transcripts (
        node_id TEXT,
        transcript_text TEXT NOT NULL,
        PRIMARY KEY (node_id),
        FOREIGN KEY (node_id) REFERENCES nodes(id)
    ) STRICT;
    """,
    # signals — the active-state channel. The composite PK (node_id, signal_type,
    # source) does all three jobs (the is_drifted EXISTS off the leading column,
    # the source_reencounter (node, source) uniqueness, the retired singleton);
    # no surrogate id, no secondary index. All three signal_type values are
    # reserved-but-unwritten in V1a.
    """
    CREATE TABLE IF NOT EXISTS signals (
        node_id TEXT NOT NULL,
        signal_type TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        payload_json TEXT,
        PRIMARY KEY (node_id, signal_type, source),
        CHECK (signal_type IN ('drifted', 'source_reencounter', 'retired')),
        FOREIGN KEY (node_id) REFERENCES nodes(id)
    ) STRICT;
    """,
    # pending_embeddings — the Outbox (V1-D-flagless): node_id PK, queued_at,
    # retry_count. No needs_reembed / embedding_text / claimed_by (the prototype's
    # claim machinery defers to V3b).
    """
    CREATE TABLE IF NOT EXISTS pending_embeddings (
        node_id TEXT,
        queued_at TEXT NOT NULL,
        retry_count INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (node_id),
        FOREIGN KEY (node_id) REFERENCES nodes(id)
    ) STRICT;
    """,
]


def _v1_schema(conn: sqlite3.Connection) -> None:
    """Migration step 1: create the V1a STRICT-table schema (vision §8.2).

    Issues the six ``CREATE TABLE ... STRICT`` + three ``CREATE INDEX`` statements
    of ``_V1_SCHEMA_STATEMENTS`` via ``conn.execute`` (one per statement, never
    ``executescript`` — that force-commits and breaks the runner's atomic
    DDL+version-bump rollback, 2a Decision 4). Does NOT touch ``user_version`` or
    manage the transaction — ``run_migrations`` owns both. Idempotent by the
    ``IF NOT EXISTS`` backstop (MI-3 replay); collision with a *prototype* schema
    is the ``is_pre_v1a_schema`` guard's job, not this step's (a silent no-op over
    the old non-STRICT tables would mint an undiagnosable hybrid — R3/R11).

    This step is authored and proven in Phase 2b via injection
    (``run_migrations(conn, steps=[(1, _v1_schema)])``) but is **not** registered
    in the live ``MIGRATION_STEPS``; Phase 5a appends it in lockstep with the
    ``commit_parsed_entry`` rebuild that writes this schema.

    Args:
        conn: An open, writable SQLite connection inside the runner's
            transaction. Its PRAGMA suite (notably ``foreign_keys=ON``) must
            already be live — open it through ``store.open_connection`` (MI-8).
    """
    for statement in _V1_SCHEMA_STATEMENTS:
        conn.execute(statement)


def is_pre_v1a_schema(conn: sqlite3.Connection) -> bool:
    """Reports whether ``conn`` holds the prototype (pre-V1a) schema.

    A prototype graph must be routed to the §2.1 one-time cutover, NEVER
    ladder-advanced: registering step 1 and opening a prototype DB would let
    ``CREATE TABLE IF NOT EXISTS nodes (...STRICT...)`` silently no-op over the old
    non-STRICT ``nodes`` and bump ``user_version`` to 1 — an undiagnosable hybrid
    schema. This predicate is the substrate-level guard that makes the eventual
    flip safe; Phase 5a wires it into the ``__init__`` boot (refuse + route to
    cutover) and Phase 6b into the ``init`` / ``status`` surfaces.

    Defined by §5.2.7's markers: ``user_version == 0`` AND a ``nodes`` table exists
    that is **non-STRICT or lacks ``slug_casefold``**. A fresh/empty DB (no
    ``nodes`` table) and any V1a-or-later DB (``user_version >= 1``) both return
    False — "empty is healthy", and the version gate short-circuits regardless of
    table shape.

    Args:
        conn: An open SQLite connection (read-only is sufficient; this only reads
            ``user_version`` and the schema introspection pragmas).

    Returns:
        True iff ``conn`` holds a prototype graph that must be routed to cutover.
    """
    if _get_user_version(conn) != 0:
        # Any advanced version is V1a-or-later — never a prototype, regardless of
        # the on-disk table shape. The version gate wins.
        return False
    nodes_present = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='nodes';"
    ).fetchone()
    if nodes_present is None:
        # Fresh/empty DB: healthy, not pre-V1a (empty-state-first-class).
        return False
    # ``nodes`` exists at user_version 0 — prototype iff it is non-STRICT OR is
    # missing the V1a ``slug_casefold`` column. Read both markers back through the
    # introspection pragmas (table-valued forms, bound parameters — no
    # interpolation); the 2a version guard already proved SQLite >= 3.37, so
    # ``pragma_table_list``'s ``strict`` column is available.
    strict_row = conn.execute(
        "SELECT strict FROM pragma_table_list WHERE name = ?;", ("nodes",)
    ).fetchone()
    is_strict = strict_row is not None and bool(strict_row[0])
    has_casefold = (
        conn.execute(
            "SELECT 1 FROM pragma_table_info(?) WHERE name = ?;",
            ("nodes", "slug_casefold"),
        ).fetchone()
        is not None
    )
    return (not is_strict) or (not has_casefold)


# --- V1a live registration (Phase 5a) ------------------------------------------
#
# The entry-001 schema boot-flip: 2b authored ``_v1_schema`` but left it
# unregistered so the suite stayed green on the prototype boot through 2b–4b.
# Phase 5a registers it as live ladder step 1 in lockstep with the
# ``commit_parsed_entry`` rebuild that writes this schema. Use ``.append`` — NEVER
# rebind ``MIGRATION_STEPS = [...]``: ``run_migrations``'s default arg binds the
# list *object* at def-time, so an in-place append is seen by the live boot while
# a rebind would be invisible to it (2a IMPL_NOTES; §7 gotcha).
MIGRATION_STEPS.append((1, _v1_schema))
