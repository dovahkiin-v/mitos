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

import os
import sqlite3
from typing import Callable, List, Optional, Tuple

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


# --- V1b schema (Phase 1b): migration ladder step 2 ----------------------------
#
# The first populated-schema migration: it (a) creates the ``mechanisms`` entity
# registry (W1, written by Phase 5a) and (b) rebuilds ``edges`` to widen its
# kind-CHECK from the V1a two-type, same-kind-only whitelist to the full nine-type
# V1b catalog (W3, committed by Phase 2a). Three of the nine are CROSS-kind
# (``cites`` any→any, ``derives_from`` OQ→D, ``resolves`` D→OQ) that
# ``source_kind = target_kind`` structurally forbids, so an ``IN(...)`` extension
# cannot express them and a table REBUILD is required (Decision 2; SQLite cannot
# ``ALTER`` a CHECK).
#
# Appending ``(2, _v1b_schema)`` to ``MIGRATION_STEPS`` activates Phase 1a's dormant
# pre-ladder snapshot (the head becomes 2, so a populated v1 boot now satisfies the
# ``current < head`` precondition) with ZERO change to the 1a harness — so a real
# ``v1→v2`` boot snapshots first and restores on failure, the binary reversal for the
# silent-semantic corruption the in-step faithfulness count-check (Decision 4) cannot
# see. The rebuild runs FK-on and is safe ONLY because nothing references ``edges``
# (``grep "REFERENCES edges"`` → empty, Decision 3); ``PRAGMA foreign_keys`` cannot be
# toggled inside the runner's transaction anyway. There is NO ``edge_type_matrix``
# reference table — the CHECK-with-OR is the whole catalog (P19; ADR
# ``edge-catalog-stays-check-constraint-not-reference-table``).

# ``mechanisms`` — the V1-D5 / V1-D15 entity registry (W1). Pure add:
# ``canonical_name`` PK (slugified by 5a's WRITER via ``mechanism_canonical_norm``,
# NOT this DDL), ``authored_name`` first-seen presentation casing, ``source``
# first-seen-wins provenance, ``created_at`` application-supplied UTC ISO-8601 µs (no
# ``CURRENT_TIMESTAMP`` default — MI-10, mirrors ``_v1_schema``). STRICT enforces the
# TEXT typing. NO ``kind`` column, NO alias storage, NO ``node_mechanisms`` junction
# (the dead ``store._init_db`` prototype's shape — v0.2 adds the ``kind`` flag/aliases
# non-destructively, so P12/P20 forbid pre-creating them here). The ``canonical_name``
# PK is the only index needed (P11); nodes link via the existing
# ``mechanism_refs_json`` column, not a foreign key.
_V1B_MECHANISMS_STATEMENT: str = """
    CREATE TABLE IF NOT EXISTS mechanisms (
        canonical_name TEXT NOT NULL,
        authored_name  TEXT NOT NULL,
        source         TEXT NOT NULL,
        created_at     TEXT NOT NULL,
        PRIMARY KEY (canonical_name),
        CHECK (source IN ('user', 'capture_llm', 'import_llm'))
    ) STRICT;
"""

# ``edges_new`` — the rebuild target carrying the widened nine-type kind-CHECK. Every
# other column/PK/FK is byte-identical to the V1a ``edges`` (step 1's DDL); ONLY the
# CHECK widens. The catalog (vision §4.3): six SAME-kind types (the two kill-edges +
# the four new same-kind non-kill ``amends``/``narrows``/``depends_on``/``contradicts``)
# plus three CROSS-kind clauses (``cites`` any→any, ``derives_from`` OQ→D, ``resolves``
# D→OQ). Node kinds are exactly {'decision', 'open_question'} (the ``nodes`` CHECK), so
# the cross-kind clauses enumerate the full direction matrix. These nine ==
# ``store._KILL_EDGE_FIELDS ∪ _DEFERRED_EDGE_FIELDS`` (the set Phase 2a commits) — a
# typo here is a silent cross-phase mismatch 2a inherits, so the literals are copied
# from the live constants, not retyped from memory.
_V1B_EDGES_NEW_STATEMENT: str = """
    CREATE TABLE edges_new (
        source_id TEXT NOT NULL,
        source_kind TEXT NOT NULL,
        target_id TEXT NOT NULL,
        target_kind TEXT NOT NULL,
        edge_type TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (source_id, target_id, edge_type),
        CHECK (
            (source_kind = target_kind AND edge_type IN
                ('supersedes', 'corrects', 'amends', 'narrows', 'depends_on', 'contradicts'))
         OR (edge_type = 'cites')
         OR (source_kind = 'open_question' AND target_kind = 'decision'      AND edge_type = 'derives_from')
         OR (source_kind = 'decision'      AND target_kind = 'open_question' AND edge_type = 'resolves')
        ),
        FOREIGN KEY (source_id, source_kind) REFERENCES nodes(id, kind),
        FOREIGN KEY (target_id, target_kind) REFERENCES nodes(id, kind)
    ) STRICT;
"""

# A stable marker of the widened CHECK in ``edges``'s stored DDL: the quoted token
# ``'cites'`` appears in the nine-type catalog and never in the V1a two-type CHECK, so
# its presence means the rebuild has already run — the MI-3 replay skip-guard
# (``_rebuild_edges_widened``). ``ALTER TABLE ... RENAME`` rewrites the stored ``sql``
# to name the table ``edges`` while preserving the CHECK body, so the marker survives
# the rename.
_V1B_EDGES_WIDENED_MARKER: str = "'cites'"


def _rebuild_edges_widened(conn: sqlite3.Connection) -> None:
    """Rebuilds ``edges`` in place to widen its kind-CHECK to the 9-type catalog.

    SQLite cannot ``ALTER`` a CHECK, and an ``IN(...)`` list can express neither a
    cross-kind row nor the dropping of ``source_kind = target_kind``, so the widening
    is a table rebuild: create ``edges_new`` with the widened CHECK, copy every row,
    drop ``edges``, rename ``edges_new`` → ``edges``, recreate ``idx_edges_target``
    (``DROP TABLE`` takes the index with it). Runs inside ``run_migrations``'
    transaction — one statement per ``conn.execute``, touching neither
    ``BEGIN``/``COMMIT`` nor ``user_version``.

    Two safety properties make this faithful and reversible:

    * **Faithfulness (R13, Decision 4):** the source row count is captured before the
      copy and compared to ``edges_new`` after it — *before* ``edges`` is dropped — so
      a short ``INSERT ... SELECT`` raises (rolling the whole step back, the 1a
      snapshot restoring) with the original ``edges`` still intact. This turns a silent
      edge loss into a loud, reversible failure. ``PRAGMA foreign_key_check`` then
      asserts the copy left no dangling reference (a read-only probe, legal inside the
      txn; the SQLite FK-disable dance is unneeded because ``edges`` is not an FK
      target — Decision 3).
    * **Replay-safety (MI-3, Decision 5):** if ``edges`` already carries the widened
      CHECK (the ``'cites'`` marker), the rebuild is skipped entirely — a true no-op on
      full-ladder replay or a re-invocation against an already-v2 DB.

    Args:
        conn: An open, writable SQLite connection inside the runner's transaction, with
            ``foreign_keys=ON`` live (opened via ``store.open_connection``, MI-8). The
            ``edges`` table from step 1 must already exist.

    Raises:
        DatabaseError: If the copied edge count differs from the source (a faithfulness
            breach, R13) or ``PRAGMA foreign_key_check`` reports a dangling reference.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='edges';"
    ).fetchone()
    if row is not None and row[0] is not None and _V1B_EDGES_WIDENED_MARKER in row[0]:
        # Already widened — the MI-3 replay no-op (Decision 5).
        return

    # Drop any leftover ``edges_new`` from a crashed prior attempt before the rebuild.
    # Transactional rollback makes a real orphan impossible, but the guard is cheap
    # insurance and keeps a bare re-invocation clean.
    conn.execute("DROP TABLE IF EXISTS edges_new;")
    conn.execute(_V1B_EDGES_NEW_STATEMENT)

    # Faithfulness gate (R13): count BEFORE the copy, compare AFTER — and BEFORE the
    # ``DROP``, so a mismatch rolls back with the original ``edges`` intact.
    before = conn.execute("SELECT COUNT(*) FROM edges;").fetchone()[0]
    conn.execute(
        "INSERT INTO edges_new "
        "(source_id, source_kind, target_id, target_kind, edge_type, created_at) "
        "SELECT source_id, source_kind, target_id, target_kind, edge_type, created_at "
        "FROM edges;"
    )
    after = conn.execute("SELECT COUNT(*) FROM edges_new;").fetchone()[0]
    if before != after:
        raise DatabaseError(
            f"edges rebuild faithfulness check failed: {before} source rows but "
            f"{after} carried into edges_new — aborting to avoid silent edge loss "
            f"(R13). The step rolls back and the pre-ladder snapshot restores the graph."
        )
    # Structural assertion: the copy introduced no dangling FK. The table name is a
    # code-internal literal — the P8 carve-out, same class as ``PRAGMA user_version``.
    fk_violations = conn.execute("PRAGMA foreign_key_check(edges_new);").fetchall()
    if fk_violations:
        raise DatabaseError(
            f"edges rebuild left {len(fk_violations)} dangling foreign-key "
            f"reference(s) in edges_new — aborting (R13). The step rolls back and the "
            f"pre-ladder snapshot restores the graph."
        )

    conn.execute("DROP TABLE edges;")
    conn.execute("ALTER TABLE edges_new RENAME TO edges;")
    # ``DROP TABLE edges`` took ``idx_edges_target`` with it; recreate the
    # incoming-kill-edge anti-join index on the rebuilt table (mirrors the V1a DDL).
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges (target_id, edge_type);"
    )


def _v1b_schema(conn: sqlite3.Connection) -> None:
    """Migration step 2 (V1b): create ``mechanisms`` and widen the ``edges`` CHECK.

    Lands both V1b schema additions as one forward-only ladder rung (``user_version``
    1→2): (a) the STRICT ``mechanisms`` entity registry (W1, written by Phase 5a) and
    (b) the ``edges`` kind-CHECK widened to the nine-type catalog via an in-place table
    rebuild (W3, committed by Phase 2a). Runs inside ``run_migrations``' transaction;
    does NOT touch ``user_version`` or ``BEGIN``/``COMMIT`` (the runner owns both), and
    issues one statement per ``conn.execute`` — never ``executescript`` (it
    force-commits and would split the DDL out of the atomic version-bump, the contract
    ``_v1_schema`` documents).

    Order is mechanisms-first (a trivial ``CREATE ... IF NOT EXISTS``), edges-rebuild
    second (the risky part — Decision 1). Both are idempotent for MI-3 replay: the
    ``mechanisms`` create via ``IF NOT EXISTS``, the edges rebuild via its widened-CHECK
    skip-guard (``_rebuild_edges_widened``).

    Args:
        conn: An open, writable SQLite connection inside the runner's transaction, with
            ``foreign_keys=ON`` live (opened via ``store.open_connection``, MI-8). The
            ``edges`` / ``nodes`` tables from step 1 must already exist — the ladder
            applies step 1 before step 2 on a fresh boot.
    """
    conn.execute(_V1B_MECHANISMS_STATEMENT)
    _rebuild_edges_widened(conn)


# --- Pre-ladder DB snapshot harness (Phase 1a): binary migration reversal ------
#
# The first populated-schema migration (Phase 1b) rewrites a graph that already
# holds real, irreplaceable rows — including edges on archived entries that are
# NOT re-derivable from the buffer. The 1b faithfulness gates catch a migration
# that loses rows *loudly*; they cannot reverse a successful-but-buggy rebuild that
# drops rows *silently*. So before the ladder runs against a populated graph, take
# a WAL-consistent snapshot of the DB file: restored on any ladder failure (leaving
# no half-migrated DB), and RETAINED on success — never auto-dropped-on-pass, the
# load-bearing anti-requirement (a gate-passing migration can still commit a silent
# semantic corruption, so destroying the only pre-migration image the instant the
# gates go green is a wrongful-advance, P5 Ironclad). ADRs:
# ``v1b-migration-takes-pre-ladder-db-snapshot-as-binary-reversal`` and its amend
# ``v1b-migration-snapshot-retained-on-success-not-dropped``.
#
# The harness ships DORMANT in 1a: the live registry head is 1, so the
# ``current >= 1 AND current < head`` precondition is unsatisfiable on a real boot
# until 1b appends ``(2, _v1b_schema)``. It is proven now via injected synthetic
# steps and first fires on a real boot in 1b — with ZERO further change here,
# because the precondition keys off ``_pending_head(steps)``, never a hardcoded
# version. The WAL-sidecar discipline mirrors ``cutover.py``'s idioms but does NOT
# import them — that would form a ``migrations -> cutover -> store -> migrations``
# cycle (Decision 5); the ~4-line ``_clear_sidecars`` is re-implemented here.


def _pending_head(steps: List[MigrationStep]) -> int:
    """Returns the highest version in ``steps`` (the ladder head), 0 if empty.

    Keyed off the passed ``steps`` rather than a hardcoded version so the snapshot
    precondition activates automatically when a later phase appends a rung: 1b's
    single ``MIGRATION_STEPS.append((2, _v1b_schema))`` flips the harness live with
    no change here.

    Args:
        steps: The step registry whose head version to read.

    Returns:
        The maximum step version, or 0 when ``steps`` is empty.
    """
    return max((version for version, _step_fn in steps), default=0)


def _snapshot_path(db_path: str, current: int) -> str:
    """Returns the deterministic pre-ladder snapshot sibling path for ``current``.

    Derived once here so the producer (:func:`take_pre_ladder_snapshot`) and the
    consumer (:func:`restore_from_snapshot`) cannot drift. The path is a sibling of
    ``db_path`` (same filesystem → atomic ``os.replace``) keyed by the version
    migrated *from*, making it deterministic and wall-clock-free (one image per
    version jump). It is distinct from the cutover's ``.rebuild`` / ``.bak_<ts>``
    siblings.

    Args:
        db_path: The live graph DB path.
        current: The pre-migration ``user_version`` the snapshot captures.

    Returns:
        The snapshot path, e.g. ``.mitos/graph.sqlite.snapshot_v1``.
    """
    return f"{db_path}.snapshot_v{current}"


def _clear_sidecars(base_path: str) -> None:
    """Removes a SQLite database's ``-wal`` / ``-shm`` sidecars, if present.

    Re-implements ``cutover._clear_sidecars``'s idiom (deliberately NOT imported —
    that would cycle ``migrations -> cutover -> store -> migrations``, Decision 5).
    Clearing the destination's orphan WAL sidecars *before* a restore is the
    cutover's hard-won R11 lesson: a stale ``-wal`` mis-applied to the restored file
    yields ``SQLITE_CORRUPT``. Absent files are a no-op.

    Args:
        base_path: The database path whose ``-wal`` / ``-shm`` sidecars to clear.
    """
    for suffix in ("-wal", "-shm"):
        try:
            os.remove(base_path + suffix)
        except FileNotFoundError:
            pass


def _discard_stale_snapshot(snapshot_path: str) -> None:
    """Removes a leftover snapshot (+ any orphan sidecars) before a fresh take.

    ``VACUUM INTO`` errors if its target already exists, so a snapshot left by a
    crashed prior attempt must be cleared first — the boot then self-heals on retry
    (P5 idempotency). A ``VACUUM INTO`` output carries no sidecars, but a crashed
    attempt's leftovers might, so all three suffixes are cleared (mirrors
    ``cutover._discard_stale_aside``). Absent files are a no-op.

    Args:
        snapshot_path: The snapshot path to discard.
    """
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(snapshot_path + suffix)
        except FileNotFoundError:
            pass


def take_pre_ladder_snapshot(
    conn: sqlite3.Connection,
    db_path: str,
    steps: List[MigrationStep] = MIGRATION_STEPS,
) -> Optional[str]:
    """Takes a WAL-consistent snapshot of ``db_path`` before a pending migration.

    Snapshots **only** when a populated graph (``user_version >= 1``) has a pending
    ladder step (``current < head(steps)``) — exactly the first populated-schema
    migration the reversal defends. A fresh/empty DB (version 0) has no rows to
    lose, and an at-head DB has no pending step; both return ``None`` and write no
    file, so the common no-op boot copies nothing.

    The snapshot is taken via ``VACUUM INTO``: it reads the connection's committed
    view — **WAL frames included** — and writes a fresh, self-contained DB with no
    ``-wal``/``-shm`` sidecars, so there is exactly one consistent file to restore
    from. A bare ``cp`` of the main DB while a ``-wal`` exists would capture a torn,
    stale image and silently destroy the guarantee (§9, Decision 1). ``VACUUM INTO``
    also preserves ``user_version``, so the restored DB is faithful to the
    pre-migration version.

    Args:
        conn: The open boot connection (the same one the ladder will run on). At the
            call site only read-only schema probes have run, so no write transaction
            is open; a defensive ``commit`` is issued anyway because ``VACUUM``
            cannot run inside a transaction.
        db_path: The live graph DB path the snapshot is a sibling of.
        steps: The step registry whose head decides whether a migration is pending.
            Defaults to the live ``MIGRATION_STEPS`` (head 1 → dormant until 1b
            appends step 2); tests inject synthetic steps.

    Returns:
        The snapshot path if one was taken, else ``None`` (no pending populated
        migration).
    """
    current = _get_user_version(conn)
    if current < 1 or current >= _pending_head(steps):
        return None
    snapshot_path = _snapshot_path(db_path, current)
    # Self-heal an interrupted prior attempt: VACUUM INTO errors on an existing
    # target, so discard any stale snapshot first (P5 idempotency).
    _discard_stale_snapshot(snapshot_path)
    # VACUUM cannot run inside a transaction. No write txn is open here (only
    # ``is_pre_v1a_schema``'s reads have run, under Python's deferred isolation), so
    # this commit is a defensive no-op — cheap insurance before the VACUUM.
    conn.commit()
    # The path is code-derived from ``config.db_path`` — never user/LLM input — so
    # binding it as a parameter satisfies P8 with no carve-out needed.
    conn.execute("VACUUM INTO ?;", (snapshot_path,))
    return snapshot_path


def restore_from_snapshot(db_path: str, snapshot_path: str) -> None:
    """Atomically replaces ``db_path`` with the snapshot, clearing orphan WAL first.

    The failure-path reversal: on any ladder error the live graph is rolled back to
    the pre-migration snapshot, leaving no half-migrated DB. The snapshot is
    **consumed** — it *becomes* the live DB (``os.replace`` is the cross-platform
    atomic overwrite, truly atomic because the snapshot is a same-filesystem
    sibling). Retention is therefore a *success*-path property, not a failure one.

    The destination's ``-wal``/``-shm`` are cleared **before** the replace (the
    cutover's R11 orphan-WAL guard — a stale ``-wal`` applied to the restored file
    yields ``SQLITE_CORRUPT``). The snapshot itself (from ``VACUUM INTO``) has none.

    Precondition: every connection to ``db_path`` is closed (the caller closes its
    boot conn before invoking this).

    Args:
        db_path: The live graph DB path to restore in place.
        snapshot_path: The snapshot produced by :func:`take_pre_ladder_snapshot`.
    """
    _clear_sidecars(db_path)
    os.replace(snapshot_path, db_path)


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


# --- V1b live registration (Phase 1b) ------------------------------------------
#
# Append step 2 — the ``mechanisms`` DDL + the widened ``edges`` CHECK. This single
# ``.append`` is the switch that wakes Phase 1a's dormant pre-ladder snapshot: the
# ladder head becomes 2, so a populated v1 graph now has a pending step and the
# snapshot fires on its real ``v1→v2`` boot (with ZERO change to the 1a harness, which
# keys off ``_pending_head(steps)``). Use ``.append`` — NEVER rebind
# ``MIGRATION_STEPS = [...]``: ``run_migrations``'s default arg binds the list *object*
# at def-time, so an in-place append is seen by the live boot while a rebind would be
# invisible to it (PATTERNS; §7 gotcha).
MIGRATION_STEPS.append((2, _v1b_schema))
