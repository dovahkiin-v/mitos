"""SQLite-backed graph store for Mitos.

This module implements the core architectural Identity & Graph Substrate (B),
computed states (C), edge-reconciliation (V1-D21), and cascade scopes (V1-D22).
"""

import sqlite3
import json
import os
import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Set, Tuple
from mitos.errors import (
    DatabaseError,
    ValidationError,
    CommitError,
    EntryFailure,
    FailureItem,
    STORE_SLUG_COLLISION,
    STORE_MISSING_TARGET,
    STORE_DANGLING_EDGE,
    STORE_KIND_CONSTRAINT_VIOLATION,
    STORE_CYCLE_VIOLATION,
)
from mitos.identity import compute_node_id
from mitos.migrations import (
    MIGRATION_STEPS,
    MigrationStep,
    run_migrations,
    is_pre_v1a_schema,
    take_pre_ladder_snapshot,
    restore_from_snapshot,
)
from mitos.parser import ParsedEntry

# Module logger for non-failing notices. The store is a pure primitive — it logs
# (loud, testable via ``caplog``, no raw stdout I/O) and never prints to the user;
# user-facing notices belong to the sync/cli consumer layer.
logger = logging.getLogger(__name__)

# The SQLite floor V1a's STRICT tables and the migration ladder require. The
# >=3.13 Python floor (Phase 1a) guarantees a bundled SQLite at or above this, but
# a custom build or a rebuilt `sqlite3` module could link an older library — so we
# verify the *linked* version at connect time rather than trusting the interpreter.
SQLITE_MIN_VERSION: Tuple[int, int, int] = (3, 37, 0)

# The §5.2.8 writer-lock-contention budget: SQLITE_BUSY retries are handled inside
# the SQLite C driver via this timeout, never an application-layer Python retry
# loop (Lesson 2 — keep the retry structural, in C).
BUSY_TIMEOUT_MS: int = 5000

# A decision is "live" (still in force) if it is active or merely drifted. Drifted
# means live-but-possibly-stale, not retired — surface_decisions treats both as
# surfaceable, so enumeration must use the same notion of "active".
LIVE_STATES: Tuple[str, ...] = ("active", "drifted")

# Edge types that mean a LATER decision has modified an earlier one. Edges are
# stored from the newer node TO the one it changes, so the INCOMING set of these
# on a node is exactly "who moved on from me". Mapping is edge type -> the
# reverse-relation key surfaced on a retrieved payload. Surfacing these closes the
# "amended axioms read as live" trap: an `amends`/`narrows` leaves the parent
# `active` with its original axiom text, so without this a reader cites a
# superseded mechanism — worst case an architecture relocation — with full
# confidence. `corrects` is included for completeness (it retires like supersedes).
MODIFIER_EDGE_KEYS: Dict[str, str] = {
    "supersedes": "superseded_by",
    "amends": "amended_by",
    "narrows": "narrowed_by",
    "corrects": "corrected_by",
}

# The two V1a kill-edge relationship fields (new→old): a declared citation mints a
# typed edge that removes the target from the active view. Both match the
# ``edges.edge_type`` DDL CHECK whitelist (``'supersedes'`` / ``'corrects'``).
_KILL_EDGE_FIELDS: Tuple[str, ...] = ("supersedes", "corrects")

# Display tokens for every relationship field, used for ``FailureItem.field``
# localization (the store has no finer per-line anchor than the entry span). These
# are the canonical ``decisions.md`` field labels — the two kill-edge labels plus
# the seven from ``sync._EXTRA_RELATIONS``. The store owns its own copy (it is a
# lower tier than ``sync`` and must not import from it); the labels are pinned by
# the test suite so the two can never drift.
_RELATION_TOKENS: Dict[str, str] = {
    "supersedes": "**Supersedes:**",
    "corrects": "**Corrects:**",
    "amends": "**Amends:**",
    "narrows": "**Narrows:**",
    "depends_on": "**Depends-On:**",
    "resolves": "**Resolves:**",
    "contradicts": "**Contradicts:**",
    "derives_from": "**Derives-From:**",
    "cites": "**Cites:**",
}

# Human-readable kind requirement per edge type, for the ``kind_constraint_violation``
# vector error (P3). The six same-kind types share one clause; the two cross-kind
# types name their direction. (``cites`` is any→any and never trips the kind CHECK,
# so its clause is informational only.) The widened ``edges`` CHECK (1b) is the
# structural gate; this just phrases its rejection for the author.
_EDGE_KIND_REQUIREMENT: Dict[str, str] = {
    "supersedes": "connect two entries of the same kind",
    "corrects": "connect two entries of the same kind",
    "amends": "connect two entries of the same kind",
    "narrows": "connect two entries of the same kind",
    "depends_on": "connect two entries of the same kind",
    "contradicts": "connect two entries of the same kind",
    "resolves": "go from a decision to an open question",
    "derives_from": "go from an open question to a decision",
    "cites": "connect any two entries",
}

# The seven NON-KILL relationship types. As of V1b (Phase 2a) these COMMIT their
# edges — both endpoints stay active (unlike the two kill-edges, which retire their
# target). They are kept as a named set because the two reconciliation gates that
# are kill-edge-specific — the "itself superseded" source-active reject (V1-D6) and
# the resurrection re-check — key on membership: a non-kill edge mints from any
# source and resurrects nothing. ``_KILL_EDGE_FIELDS + _DEFERRED_EDGE_FIELDS`` is
# exactly the nine-type catalog the widened ``edges`` CHECK enforces (1b).
_DEFERRED_EDGE_FIELDS: Tuple[str, ...] = (
    "amends",
    "narrows",
    "depends_on",
    "resolves",
    "contradicts",
    "derives_from",
    "cites",
)

# The kill-edge anti-join: a node is INACTIVE iff it is the target of one of these.
# Inlined as a code-internal whitelist (P8 carve-out) — the same active-view
# definition Phase 5d builds its public read methods on.
_KILL_EDGE_TYPES_SQL: str = "('supersedes', 'corrects')"

# --- Phase 5d read-layer SQL fragments -----------------------------------------
# These are the ONE shared active-view / derivation definitions every public read
# method binds, so phrasing can never drift between surfaces (Lesson 14:
# find-first / validate-second). Each is a code-internal literal (no user value
# interpolated — P8); a read method appends the ``?``-bound value separately.

# Activeness is COMPUTED, never stored (M3): a node is inactive iff it is the
# target of an incoming kill-edge. This correlated ``NOT EXISTS`` is the same
# active-view definition 5b/5c compute inline and the ``_is_active`` test helper
# encodes. Bind it in any SELECT whose FROM clause is the unaliased ``nodes``.
_ACTIVE_VIEW_PREDICATE: str = (
    "NOT EXISTS (SELECT 1 FROM edges "
    "WHERE edges.target_id = nodes.id "
    f"AND edges.edge_type IN {_KILL_EDGE_TYPES_SQL})"
)

# ``is_drifted`` is DERIVED from a correlated EXISTS over the drifted-signal
# channel (V1-D11), evaluated INSIDE the main SELECT (one query, never a per-node
# lookup — P11). Structurally always-False in v0.1 (no V1a path writes a drifted
# signal), but forward-correct: it lights up the day a writer ships, for one cheap
# indexed check. Drift ANNOTATES — a drifted node is never excluded from a view
# (drift is loud, not hidden; it does not retire).
_IS_DRIFTED_SQL: str = (
    "EXISTS (SELECT 1 FROM signals "
    "WHERE signals.node_id = nodes.id "
    "AND signals.signal_type = 'drifted') AS is_drifted"
)

# The incoming kill-edge type (if any) for a node, derived inside the main SELECT
# to compute ``computed_state`` (``superseded`` / ``corrected`` / ``active``).
# NULL ⇒ no incoming kill-edge ⇒ active.
_KILLER_TYPE_SQL: str = (
    "(SELECT edges.edge_type FROM edges "
    "WHERE edges.target_id = nodes.id "
    f"AND edges.edge_type IN {_KILL_EDGE_TYPES_SQL} LIMIT 1) AS killer_type"
)

# Indexed scope-membership filter (P11: ``idx_node_scopes_scope`` serves
# ``scope = ?``, then the PK joins back to ``node_id``). Appended to a read SELECT
# and bound with a single ``?`` scope value — never an O(N) Python post-filter.
_SCOPE_FILTER_SQL: str = (
    " AND EXISTS (SELECT 1 FROM node_scopes "
    "WHERE node_scopes.node_id = nodes.id AND node_scopes.scope = ?)"
)


def state_matches(computed_state: str, state_filter: Optional[str]) -> bool:
    """Reports whether a node's computed state passes the requested state filter.

    Args:
        computed_state: The node's computed state (e.g. "active", "superseded").
        state_filter: ``"active"`` (the common case) means the live set —
            ``active`` or ``drifted``. ``None`` or ``"all"`` pass everything.
            Any other value is an exact computed-state match (e.g. "superseded").

    Returns:
        True if the node should be included under the given filter.
    """
    if not state_filter or state_filter == "all":
        return True
    if state_filter == "active":
        return computed_state in LIVE_STATES
    return computed_state == state_filter


def _computed_decision_state(killer_type: Optional[str]) -> str:
    """Derives a node's computed state from its incoming kill-edge type (M3).

    State is computed, never stored: a node with no incoming kill-edge is
    ``"active"``; an incoming ``supersedes`` makes it ``"superseded"`` and an
    incoming ``corrects`` makes it ``"corrected"``. ``is_drifted`` is a separate
    annotation (drift does not retire — it never replaces this state).

    Args:
        killer_type: The incoming kill-edge ``edge_type`` (from the ``killer_type``
            derived column), or None when the node has no incoming kill-edge.

    Returns:
        The computed state string (``"active"`` / ``"superseded"`` / ``"corrected"``).
    """
    if killer_type == "supersedes":
        return "superseded"
    if killer_type == "corrects":
        return "corrected"
    return "active"


def compute_hash(
    kind: str,
    slug: str,
    core_axiom: str = "",
    mechanisms: List[str] = [],
    questions_raised: List[str] = []
) -> str:
    """Computes the PROTOTYPE slug-inclusive SHA-256 id — retained ONLY as a test fixture.

    ⚠ **Retired from production (Phase 8a, entry-002 tail).** The live identity is
    ``identity.compute_node_id`` (slug-free canonical-core hash, V1-D2). No
    production consumer imports or calls this anymore — ``sync.py`` / ``importer.py``
    mint via ``compute_node_id``. This prototype formula (slug-inclusive,
    newline-delimited) survives **only** because the cutover-reference tests
    (``test_cutover``) plant prototype-shaped graphs via the retained ``_init_db``
    fixture and must recompute the matching prototype ids. Do NOT reintroduce a
    production call; mint identity through ``identity.compute_node_id``.

    Args:
        kind: One of "decision" or "open_question".
        slug: Human-readable slug identifier.
        core_axiom: The core decision axiom text.
        mechanisms: Comma-separated or listed mechanisms.
        questions_raised: List of questions for open question kind.

    Returns:
        A 64-character SHA-256 hex string ID.
    """
    hasher = hashlib.sha256()
    if kind == "decision":
        norm_axiom = core_axiom.strip()
        norm_mechs = ",".join(sorted([m.strip() for m in mechanisms if m.strip()]))
        raw_text = f"decision\n{slug.strip().lower()}\n{norm_axiom}\n{norm_mechs}"
    else:
        norm_questions = "\n".join([q.strip() for q in questions_raised if q.strip()])
        raw_text = f"open_question\n{slug.strip().lower()}\n{norm_questions}"
        
    hasher.update(raw_text.encode("utf-8"))
    return hasher.hexdigest()


def _utc_now_iso() -> str:
    """Returns the current UTC instant as an ISO-8601 microsecond string (MI-10).

    The V1a schema carries no ``DEFAULT CURRENT_TIMESTAMP`` — every timestamp is
    application-supplied UTC ISO-8601 with a ``+00:00`` offset and microsecond
    precision (MI-10). One stamp is taken per commit; ``created_at`` and
    ``updated_at`` share it on an INSERT.

    Returns:
        The current time, e.g. ``"2026-06-18T05:54:31.532538+00:00"``.
    """
    return datetime.now(timezone.utc).isoformat()


def _strip_citation(raw: str) -> str:
    """Strips a single surrounding ``[ ... ]`` and whitespace from an edge citation.

    ``format-spec.md`` authors kill-edges as ``**Supersedes:** [slug]`` and the
    deterministic parser stores the value raw (brackets included); the agentic
    write path (``sync.py``) supplies a bare slug. Both shapes must resolve, so a
    single layer of square brackets plus surrounding whitespace is stripped before
    casefolding (MI-7). Nested or unbalanced brackets are left intact — they would
    fail resolution loudly rather than be silently "repaired".

    Args:
        raw: The relationship value as stored on ``ParsedEntry`` — possibly
            ``[bracketed]`` (corpus/cutover shape), possibly bare (agentic shape).

    Returns:
        The bare citation slug, stripped of one bracket layer and whitespace.
    """
    s = raw.strip()
    if len(s) >= 2 and s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    return s


class CommitDelta:
    """Represents the structured delta returned after an entry commit.

    Carries metadata about the committed node and any cascading scope changes
    required for incremental rerendering (V1-D22).
    """

    def __init__(
        self,
        node_id: str,
        node_scope: List[str],
        self_old_scope: List[str],
        commentary_fields_changed: bool,
        cascade_affected_scopes: List[str]
    ) -> None:
        self.node_id = node_id
        self.node_scope = node_scope
        self.self_old_scope = self_old_scope
        self.commentary_fields_changed = commentary_fields_changed
        self.cascade_affected_scopes = cascade_affected_scopes

    def to_dict(self) -> Dict[str, Any]:
        """Converts the CommitDelta into a serializable dictionary."""
        return {
            "node_id": self.node_id,
            "node_scope": self.node_scope,
            "self_old_scope": self.self_old_scope,
            "commentary_fields_changed": self.commentary_fields_changed,
            "cascade_affected_scopes": self.cascade_affected_scopes
        }


def _assert_sqlite_version() -> None:
    """Fails fast if the linked SQLite is too old for V1a's schema substrate.

    Read live from ``sqlite3.sqlite_version_info`` (the linked library version,
    not the interpreter's) on every connect — matching the vision's "at every
    connect()" wording and keeping the guard monkeypatchable in tests. The check
    is a cheap tuple compare; it fires *before* any STRICT DDL could run.

    Raises:
        DatabaseError: If ``sqlite3.sqlite_version_info`` is below
            ``SQLITE_MIN_VERSION``, naming the required version, the detected
            version, and an upgrade path.
    """
    if sqlite3.sqlite_version_info < SQLITE_MIN_VERSION:
        required = ".".join(str(p) for p in SQLITE_MIN_VERSION)
        found = ".".join(str(p) for p in sqlite3.sqlite_version_info)
        raise DatabaseError(
            f"Mitos requires SQLite >= {required} (for STRICT tables and the "
            f"migration ladder), but the linked library is {found}. Upgrade your "
            f"Python's bundled SQLite — a newer python.org build or a distro "
            f"update — or rebuild the `sqlite3` module against a newer libsqlite3."
        )


def open_connection(db_path: str, read_only: bool = False) -> sqlite3.Connection:
    """Opens a SQLite connection with Mitos's full PRAGMA suite (MI-8 chokepoint).

    This is the single connection-open path for the whole codebase — write path,
    read path, test fixtures, the migration runner all flow through it — so "every
    connection is correctly configured" is a structural fact, not a discipline
    anyone can forget. SQLite defaults ``foreign_keys`` OFF *per connection*; a
    forgotten PRAGMA would silently disable the kind-matrix FK enforcement the
    whole edge model rests on (MI-8).

    PRAGMA order follows §5.2.8: version guard (before connect) → connect →
    ``journal_mode=WAL`` → ``synchronous=NORMAL`` (write connections only — a
    ``mode=ro`` connection cannot change them) → ``foreign_keys=ON`` →
    ``busy_timeout`` (every connection; the lock-retry budget handled in the C
    driver, never a Python retry loop).

    Args:
        db_path: Filesystem path to the SQLite database file.
        read_only: If True, open immutably via ``file:...?mode=ro`` and skip the
            write-only PRAGMAs (WAL / synchronous). ``foreign_keys`` and
            ``busy_timeout`` still apply.

    Returns:
        A configured ``sqlite3.Connection`` with ``row_factory = sqlite3.Row``.

    Raises:
        DatabaseError: If the linked SQLite is below the required version, or the
            connection cannot be opened.
    """
    _assert_sqlite_version()
    try:
        if read_only:
            abs_path = os.path.abspath(db_path)
            conn = sqlite3.connect(f"file:{abs_path}?mode=ro", uri=True)
        else:
            conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        if not read_only:
            # WAL + NORMAL synchronous: the V1-D12 single-writer/multi-reader
            # posture. Both are write-connection only — a mode=ro connection
            # cannot change the journal mode or the synchronous level.
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        # busy_timeout last, per §5.2.8: the writer-lock retry budget, handled in
        # the SQLite C driver — never an application-layer Python retry loop.
        conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS};")
        return conn
    except sqlite3.Error as e:
        raise DatabaseError(f"Failed to connect to SQLite: {str(e)}")


def _boot_migrations(
    db_path: str, steps: List[MigrationStep] = MIGRATION_STEPS
) -> None:
    """Boots the migration ladder for a writable store, snapshot-protected.

    The single migration-boot path; ``GraphStore.__init__`` delegates its migration
    block here. Opens a dedicated write connection, refuses a prototype graph
    (routing it to the one-time cutover rather than ladder-advancing it into an
    undiagnosable hybrid, R3/R11), takes a pre-ladder snapshot when a populated
    graph has a pending step, then runs the ladder. On any failure the connection is
    closed and the live graph is atomically restored from the snapshot — leaving no
    half-migrated DB — before the exception propagates; when no snapshot was taken
    (fresh/empty DB, or no pending step) the failure path simply re-raises. On
    success the snapshot is left on disk, retained — never auto-dropped (the
    silent-corruption fallback, P5 Ironclad).

    Args:
        db_path: Filesystem path to the graph DB to migrate. Read-only stores never
            reach here (a ``mode=ro`` connection cannot run the ladder's write
            transaction) — ``__init__`` gates the call on ``not self.read_only``.
        steps: The step registry to apply and to size the snapshot precondition
            against. Defaults to the live ``MIGRATION_STEPS``; tests inject synthetic
            steps to exercise the snapshot/restore path before 1b's real step exists.
    """
    conn = open_connection(db_path)
    snapshot_path: Optional[str] = None
    try:
        # Refuse a prototype graph rather than ladder-advance it into an
        # undiagnosable hybrid (R3/R11) — route it to the §2.1 one-time cutover.
        # Message unchanged from the pre-1a in-__init__ guard (relocated here).
        if is_pre_v1a_schema(conn):
            raise DatabaseError(
                "This graph predates the V1a schema (a prototype layout "
                "was detected). Mitos will not migrate it in place. Run "
                "the one-time cutover to rebuild it into the V1a store."
            )
        snapshot_path = take_pre_ladder_snapshot(conn, db_path, steps)
        run_migrations(conn, steps)
    except Exception:
        # Close before restoring: file replacement needs every handle released (and
        # the orphan-WAL cleared). ``close()`` is idempotent, so the second close in
        # ``finally`` is a safe no-op.
        conn.close()
        if snapshot_path is not None:
            restore_from_snapshot(db_path, snapshot_path)
        raise
    finally:
        conn.close()


class GraphStore:
    """SQLite-backed store managing nodes, edges, signals, and computed state."""

    def __init__(self, db_path: str, read_only: bool = False) -> None:
        self.db_path = db_path
        self.read_only = read_only
        # Ensure the parent directory exists before any connection touches the
        # file. This was the prototype ``_init_db``'s first step; with the
        # ``_init_db()`` boot call removed (Phase 5a, entry-001) it lives here so
        # a fresh store still scaffolds its directory before the ladder boots.
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        if not self.read_only:
            # V1a boot (entry-001 flip): the V1a STRICT schema boots via the
            # migration ladder, now snapshot-protected for the first populated-schema
            # migration (Phase 1a — restore-on-failure, retain-on-success). The whole
            # boot (prototype guard + snapshot + ladder + restore) lives in the
            # module-level ``_boot_migrations`` so a fault-injection test can drive
            # snapshot→fail→restore with injected synthetic steps. Read-only stores
            # never migrate (§7 gotcha): a mode=ro connection cannot run the ladder's
            # write transaction.
            _boot_migrations(self.db_path)

    def _get_connection(self) -> sqlite3.Connection:
        """Opens a configured connection to this store's database.

        A thin instance-method wrapper over the module-level ``open_connection``
        so the PRAGMA suite (MI-8) lives in exactly one place while preserving the
        heavily-bound ``store._get_connection()`` call contract used across the
        codebase and tests.

        Returns:
            A configured ``sqlite3.Connection``.

        Raises:
            DatabaseError: If the linked SQLite is too old or the connection fails.
        """
        return open_connection(self.db_path, self.read_only)

    def _init_db(self) -> None:
        """Builds the prototype (pre-V1a) schema — retained as a test/cutover fixture.

        As of Phase 5a this is **no longer the live boot path**: ``__init__`` boots
        the V1a STRICT schema via the migration ladder. The method is kept (the
        boot *call* was removed, not the definition) because it is the canonical
        prototype-schema definition that ``is_pre_v1a_schema`` detection tests and
        the Phase 7 cutover rely on. Full retirement defers to 8a/Phase 7 (§16).
        Not to be confused with ``embeddings.py``'s unrelated ``_init_db``.
        """
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        schema = """
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            slug TEXT NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('decision', 'open_question')),
            date TEXT,
            title TEXT,
            core_axiom TEXT,
            rejected_paths TEXT,
            invalidates_if TEXT,
            context TEXT,
            transcript TEXT,
            park_reason TEXT,
            questions_raised TEXT, -- JSON array of strings
            mechanisms TEXT,       -- JSON array of strings
            scope TEXT,            -- JSON array of strings
            confirmed_by TEXT,
            confirmed_at TEXT,
            source TEXT NOT NULL DEFAULT 'user',
            source_ref TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS edges (
            from_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
            to_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
            type TEXT NOT NULL CHECK(type IN (
                'supersedes', 'amends', 'narrows', 'depends_on', 
                'contradicts', 'derives_from', 'cites', 'resolves', 'corrects'
            )),
            PRIMARY KEY (from_id, to_id, type)
        );

        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
            type TEXT NOT NULL CHECK(type IN ('drifted', 'source_reencounter')),
            actor TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS mechanisms (
            name TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS node_mechanisms (
            node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
            mechanism_name TEXT NOT NULL REFERENCES mechanisms(name) ON DELETE CASCADE,
            PRIMARY KEY (node_id, mechanism_name)
        );

        CREATE TABLE IF NOT EXISTS pending_embeddings (
            node_id TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
            embedding_text TEXT NOT NULL,
            attempts INTEGER DEFAULT 0,
            claimed_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- partial unique indexes per signals-insert-or-ignore (MI-4)
        CREATE UNIQUE INDEX IF NOT EXISTS idx_signals_drifted 
        ON signals(node_id) WHERE type = 'drifted';

        CREATE UNIQUE INDEX IF NOT EXISTS idx_signals_reencounter 
        ON signals(node_id, actor) WHERE type = 'source_reencounter';
        
        -- index for case-insensitive lookup
        CREATE INDEX IF NOT EXISTS idx_nodes_slug_nocase ON nodes(slug COLLATE NOCASE);
        """
        
        conn = self._get_connection()
        try:
            with conn:
                conn.executescript(schema)
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to initialize schema: {str(e)}")
        finally:
            conn.close()

    def resolve_slug(self, slug: str) -> List[str]:
        """Resolves a slug to node IDs via single-tier casefold-exact match (V1-D23).

        Binds Python ``str.casefold()`` against the indexed ``slug_casefold`` column —
        the same Unicode-correct discipline the authoritative commit path
        (``_reconcile_edges``) and ``get_node_by_slug`` already use (MI-9: never SQLite
        ASCII-only ``COLLATE NOCASE`` / ``LOWER``). There is **no** fuzzy alias-fallback
        tier: V1a mandates single-tier exact match, and the ``slug_aliases`` citation
        subsystem is a separate V1b feature (MI-2), not this resolver. Returns every node
        sharing the casefolded slug — a same-slug supersession lineage yields >1 (MI-13);
        callers own active-view scoping and ambiguity handling.

        Args:
            slug: The slug to find (case-insensitive via ``str.casefold()``).

        Returns:
            A list of matching node IDs (empty if none).
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT id FROM nodes WHERE slug_casefold = ?",
                (slug.casefold(),),
            ).fetchall()
            return [row["id"] for row in rows]
        finally:
            conn.close()

    # --- Phase 5d read-layer helpers (one alias map, one anti-join, one bulk -----
    # scope fetch, one bulk modifier stamp; shared by every public read method so
    # the reader-facing shape and the active-view definition can never drift).

    def _hydrate_node(self, row: sqlite3.Row, scopes: List[str]) -> Dict[str, Any]:
        """Re-keys a raw V1a ``nodes`` row into the prototype reader-facing dict.

        The single alias map (§3) at the heart of the read layer: the V1a column
        names (``axiom`` / ``mechanism_refs_json`` / ``questions_raised_json`` /
        ``rejected_paths_json``) are re-keyed to the prototype reader keys the
        unchanged consumers (``renderer.py`` / ``mcp_server.py``) bind, so the read
        paths keep working with no consumer edit (deferred to Phase 8a). Kind-aware:
        a decision carries ``core_axiom`` / ``mechanisms`` / ``rejected_paths``; an
        open_question carries ``topic`` / ``questions_raised``. Off-kind columns are
        NULL in V1a (5a §14) and are dropped, never surfaced as an empty reader key.

        Args:
            row: A ``nodes`` row joined with the derived ``is_drifted`` column (and,
                for state-bearing reads, ``killer_type``); both helper columns are
                stripped from the returned dict.
            scopes: The node's scope tags, bulk-fetched from ``node_scopes`` (never a
                per-node query — P11).

        Returns:
            The reader-facing node dict (lists are lists, never tuples; the raw
            ``rejected_paths`` string passes through verbatim).
        """
        node = dict(row)
        # Strip the derived helper columns — never part of the reader contract.
        node["is_drifted"] = bool(node.pop("is_drifted", 0))
        node.pop("killer_type", None)
        node["scope"] = list(scopes)

        # Pop all the kind-differentiated canonical-core columns so an off-kind
        # NULL never leaks under a reader key; re-add only the committing kind's.
        axiom = node.pop("axiom", None)
        mechanism_refs_json = node.pop("mechanism_refs_json", None)
        rejected_paths_json = node.pop("rejected_paths_json", None)
        questions_raised_json = node.pop("questions_raised_json", None)
        topic = node.pop("topic", None)
        if node["kind"] == "decision":
            node["core_axiom"] = axiom or ""
            # The degenerate ``["!!!"]→[""]`` mechanism stores ``'[""]'`` (5a §14),
            # which decodes to ``[""]`` — a harmless display artifact; the canonical
            # core is fenced and never re-compared, so do NOT "fix" it here.
            node["mechanisms"] = json.loads(mechanism_refs_json or "[]")
            # ``rejected_paths_json`` holds a RAW string, NOT JSON (5a §14) — passed
            # through verbatim; ``json.loads`` would corrupt or raise.
            node["rejected_paths"] = (
                rejected_paths_json if rejected_paths_json is not None else ""
            )
        else:
            node["topic"] = topic
            node["questions_raised"] = json.loads(questions_raised_json or "[]")
        return node

    def _scopes_for(
        self, conn: sqlite3.Connection, node_ids: List[str]
    ) -> Dict[str, List[str]]:
        """Bulk-fetches scope tags for many nodes in ONE indexed query (never N+1).

        Args:
            conn: An open connection (the caller owns its lifecycle).
            node_ids: The node IDs to fetch scopes for.

        Returns:
            A mapping of node_id -> sorted scope-tag list; an id with no scopes is
            simply absent from the map.
        """
        if not node_ids:
            return {}
        placeholders = ",".join("?" for _ in node_ids)
        out: Dict[str, List[str]] = {}
        for row in conn.execute(
            f"SELECT node_id, scope FROM node_scopes "
            f"WHERE node_id IN ({placeholders}) ORDER BY node_id, scope",
            node_ids,
        ):
            out.setdefault(row["node_id"], []).append(row["scope"])
        return out

    def _modifiers_map(
        self, conn: sqlite3.Connection, node_ids: List[str]
    ) -> Dict[str, Dict[str, List[str]]]:
        """The V1a modifier-map engine over an open connection (one bulk join).

        Shared by the public :meth:`get_modifiers_map` and the per-surface
        :meth:`_stamp_modifiers` pass so a single query — never N+1 — answers "who
        moved on from these nodes?".

        Args:
            conn: An open connection (the caller owns its lifecycle).
            node_ids: The target node IDs to look up incoming modifier edges for.

        Returns:
            A mapping of node_id -> {reverse_key: [modifier_slug, ...]}; only
            modified nodes appear, each with only non-empty, slug-sorted keys.
        """
        if not node_ids:
            return {}
        id_placeholders = ",".join("?" for _ in node_ids)
        type_placeholders = ",".join("?" for _ in MODIFIER_EDGE_KEYS)
        # Source-liveness gate (V1b 2b, the §4.3 FORWARD HAZARD). The join above
        # joins the source node only for its slug; without this gate a dead
        # *modifier* (the edge's source) would ghost-stamp a still-live target — a
        # dead axiom projecting onto a live node. Gate the SOURCE (``e.source_id``),
        # never the target: ``get_node`` deliberately reads inactive nodes and must
        # keep stamping their kill-pointers, so a target-liveness filter would break
        # that read path.
        #
        # ⚠ THE SEMANTIC FORK (do NOT make this filter uniform — it is a regression
        # trap): the kill-edge keys (``superseded_by`` / ``corrected_by``) stay
        # UNFILTERED, only the present-tense projections (``amended_by`` /
        # ``narrowed_by``) are source-gated.
        #   - ``amends`` / ``narrows`` project present-tense policy ("go read X for
        #     the current nuance") onto a still-live target — once X is itself
        #     superseded/corrected that guidance is stale, so de-project (the target
        #     reads un-amended again). De-projection is fail-safe, not data loss: a
        #     superseded amender is superseded, not deleted — its axiom stays in the
        #     graph (recoverable via ``get_lineage``); the target merely stops being
        #     projected onto. Re-amending from the successor is the author's call.
        #   - ``supersedes`` / ``corrects`` are historical "who retired me" pointers
        #     on an already-dead target; a kill-edge can NEVER point at a *live*
        #     target, so leaving it unfiltered cannot reintroduce the hazard. Crucially
        #     ``_KILLER_TYPE_SQL`` / ``get_node_state`` compute ``superseded`` /
        #     ``corrected`` UNFILTERED — gating these keys would empty
        #     ``superseded_by`` while ``get_node_state`` still returns ``superseded``,
        #     desyncing the payload from the state. The kill-pointer-survives guard
        #     test pins this.
        #
        # Single-sourced through ``_KILL_EDGE_TYPES_SQL`` (the one liveness atom every
        # anti-join keys off) — never a re-hand-rolled ``('supersedes','corrects')``
        # literal, never ``_ACTIVE_VIEW_PREDICATE`` verbatim (it binds an UNALIASED
        # ``nodes``; this FROM aliases the source ``n``). Interpolated, not bound
        # (P8 carve-out, exactly as the other kill-edge anti-joins do it), so the
        # binding tuple is unchanged.
        cursor = conn.execute(
            f"""
            SELECT e.target_id AS target_id, e.edge_type AS edge_type, n.slug AS slug
            FROM edges e
            JOIN nodes n ON n.id = e.source_id
            WHERE e.target_id IN ({id_placeholders})
              AND e.edge_type IN ({type_placeholders})
              AND (
                    e.edge_type IN {_KILL_EDGE_TYPES_SQL}
                 OR NOT EXISTS (
                        SELECT 1 FROM edges k
                        WHERE k.target_id = e.source_id
                          AND k.edge_type IN {_KILL_EDGE_TYPES_SQL}
                    )
                  )
            ORDER BY n.slug
            """,
            (*node_ids, *MODIFIER_EDGE_KEYS.keys()),
        )
        result: Dict[str, Dict[str, List[str]]] = {}
        for row in cursor.fetchall():
            key = MODIFIER_EDGE_KEYS[row["edge_type"]]
            result.setdefault(row["target_id"], {}).setdefault(key, []).append(
                row["slug"]
            )
        return result

    def _stamp_modifiers(
        self, conn: sqlite3.Connection, nodes: List[Dict[str, Any]]
    ) -> None:
        """Stamps reverse-relation modifier keys via ONE bulk modifier join, in place.

        Every store read surface routes through this single bulk pass — never N+1,
        never a static-empty stamp. The C4 FORWARD HAZARD seam ships on every read
        even though V1a's active view is provably modifier-empty (an active node has
        no incoming kill-edge by definition): the graph always knew via the edges, so
        the payload must not lie. ``amends`` / ``narrows`` light up that seam as of
        V1b — the warn-defer→commit flip (2a, the moment those edges first exist) plus
        the source-liveness gate inside :meth:`_modifiers_map` (2b) that de-projects a
        dead amender. ``MODIFIER_EDGE_KEYS`` already maps all four keys (the V1a seam);
        the light-up is the new edges committing + the new gate, not a key-map edit.

        Args:
            conn: An open connection (the caller owns its lifecycle).
            nodes: Already-hydrated node dicts (each carrying ``id``) to stamp.
        """
        if not nodes:
            return
        mod_map = self._modifiers_map(conn, [n["id"] for n in nodes])
        for node in nodes:
            for key, slugs in mod_map.get(node["id"], {}).items():
                node[key] = slugs

    def _hydrate_rows(
        self, conn: sqlite3.Connection, rows: List[sqlite3.Row]
    ) -> List[Dict[str, Any]]:
        """Hydrates many rows with ONE bulk scope fetch + ONE bulk modifier stamp.

        Args:
            conn: An open connection (the caller owns its lifecycle).
            rows: Raw ``nodes`` rows (each joined with ``is_drifted``).

        Returns:
            The reader-facing node dicts, scope-joined and modifier-stamped.
        """
        if not rows:
            return []
        ids = [row["id"] for row in rows]
        scopes_map = self._scopes_for(conn, ids)
        nodes = [self._hydrate_node(row, scopes_map.get(row["id"], [])) for row in rows]
        self._stamp_modifiers(conn, nodes)
        return nodes

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single node by its content-hash ID, in the reader-facing shape.

        Returns the node whether it is active or inactive (a direct id read is not
        active-scoped); an inactive node carries its stamped ``superseded_by`` /
        ``corrected_by`` modifier keys so a reader of a moved-on node still sees who
        moved on.

        Args:
            node_id: The primary key ID of the node.

        Returns:
            The reader-facing node dict (aliased columns, ``scope`` list,
            ``is_drifted`` bool, stamped modifiers when non-empty), or None.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                f"SELECT nodes.*, {_IS_DRIFTED_SQL} FROM nodes WHERE nodes.id = ?",
                (node_id,),
            ).fetchone()
            if not row:
                return None
            scopes_map = self._scopes_for(conn, [node_id])
            node = self._hydrate_node(row, scopes_map.get(node_id, []))
            self._stamp_modifiers(conn, [node])
            return node
        finally:
            conn.close()

    def get_node_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """Retrieves the single ACTIVE node for a slug (active-scoped → ≤1, MI-13).

        Resolves the slug directly within the active view — ``casefold(slug)`` match
        AND the kill-edge anti-join — so ``casefold(slug) → active node`` is a
        function, not a relation (a superseded predecessor and its superseder never
        coexist in the active set). MI-13 (enforced upstream at commit, 5b) guarantees
        at most one active node per ``casefold(slug)``, so the ambiguity ``raise``
        below is a defensive vector (P3) that is structurally unreachable.

        Deliberately does NOT route through ``resolve_slug`` because the two methods
        serve different scopes — this is an active-view single read (≤1), while
        ``resolve_slug`` is an all-nodes list (for collision / edge pre-validation).
        Both now share one casing discipline: Python ``str.casefold()`` bound against
        the indexed ``slug_casefold`` column (no SQLite ``NOCASE``, no fuzzy ``LIKE``
        tier; reconciled in r1).

        Args:
            slug: The slug identifier (case-insensitive via ``str.casefold()``).

        Returns:
            The single active node dict (reader-facing shape), or None.

        Raises:
            ValidationError: If more than one active node shares the casefolded slug
                — an MI-13 breach (defensive; structurally unreachable).
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                f"SELECT nodes.*, {_IS_DRIFTED_SQL} FROM nodes "
                f"WHERE nodes.slug_casefold = ? AND {_ACTIVE_VIEW_PREDICATE}",
                (slug.casefold(),),
            ).fetchall()
            if not rows:
                return None
            if len(rows) > 1:
                raise ValidationError(
                    f"Slug '{slug}' resolves to {len(rows)} active nodes "
                    f"(MI-13 breach): {[row['id'] for row in rows]}"
                )
            node_id = rows[0]["id"]
            scopes_map = self._scopes_for(conn, [node_id])
            node = self._hydrate_node(rows[0], scopes_map.get(node_id, []))
            self._stamp_modifiers(conn, [node])
            return node
        finally:
            conn.close()

    def get_node_state(self, node_id: str) -> str:
        """Computes one node's state from its incoming kill-edge (the V1a active view).

        The single-node companion to ``get_decisions``' bulk ``computed_state``
        derivation: it binds the SAME ``_KILLER_TYPE_SQL`` / ``_computed_decision_state``
        logic (M3 — state is computed, never stored), so the store has exactly one
        definition of "active". Replaces the prototype
        ``compute_all_states(conn).get(node_id)`` consumer call sites (Phase 8a).

        Returns ``"active"`` for a node with no incoming kill-edge, ``"superseded"``
        for an incoming ``supersedes``, ``"corrected"`` for an incoming ``corrects``,
        and ``"drifted"`` for an otherwise-active node carrying a drifted signal
        (always-False in v0.1 — the channel is reserved, V1-D11). An absent node
        defaults to ``"active"`` (mirrors the prototype ``.get(id, "active")``; every
        live caller resolves the node first, so the absent branch is a defensive
        vector, P3).

        The state is **decision-centric** (``active`` / ``superseded`` /
        ``corrected`` / ``drifted``). Open-question resolution state
        (``parked`` / ``resolved``) rides the ``resolves`` edge — that edge commits
        as of V1b 2a, but deriving the OQ ``parked`` / ``resolved`` state from it is
        a later V1b phase — so an open_question resolves to its kill-edge state here,
        never ``parked`` / ``resolved`` (G7; V1b owns OQ resolution).

        Args:
            node_id: The content-hash id of the node to derive state for.

        Returns:
            The computed state string.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                f"SELECT {_IS_DRIFTED_SQL}, {_KILLER_TYPE_SQL} "
                "FROM nodes WHERE nodes.id = ?",
                (node_id,),
            ).fetchone()
            if row is None:
                return "active"
            if row["killer_type"] is None and row["is_drifted"]:
                return "drifted"
            return _computed_decision_state(row["killer_type"])
        finally:
            conn.close()

    def write_signal(self, node_id: str, stype: str, actor: Optional[str] = None) -> None:
        """Writes a signal row (drifted, source_reencounter, retired) via INSERT OR IGNORE.

        Aligned to the V1a ``signals`` shape (Phase 8a): ``type``→``signal_type``,
        ``actor``→``source`` (NOT NULL DEFAULT '' — a ``None`` actor maps to ''), and
        the now-required application-supplied ``created_at`` UTC µs stamp (MI-10; the
        V1a schema carries no ``DEFAULT CURRENT_TIMESTAMP``). All three signal_type
        values are reserved-but-unwritten in V1a — no live caller writes a signal —
        so this is a correctness alignment of a latent landmine (K6), not a behavior
        change. ``INSERT OR IGNORE`` keeps the composite-PK uniqueness from crashing.
        """
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO signals (node_id, signal_type, source, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (node_id, stype, actor or "", _utc_now_iso())
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to write signal: {str(e)}")
        finally:
            conn.close()

    def commit_parsed_entry(self, parsed: ParsedEntry) -> CommitDelta:
        """Commits one ParsedEntry as a single atomic V1a transaction (V1-D10).

        Mints the slug-free content-hash id via ``identity.compute_node_id`` and
        writes the committing entry across ``nodes`` + ``node_scopes`` +
        ``transcripts`` in one transaction — partial commit is structurally
        impossible (a crash, lock timeout, or error rolls back the whole entry).

        Identity is the content, not the slug (entry-002 / V1-D2): a new id is an
        INSERT; a re-commit of the same canonical core is an **in-place commentary
        UPDATE** whose ``SET`` covers only commentary — ``slug`` is mutable (a
        rename), but the canonical core *and* ``source`` are fenced (MI-4). A
        byte-identical re-commit is a true no-op: ``updated_at`` does not tick
        (MI-3 / V1-D17). ``node_scopes`` reconcile idempotently (casefolded,
        insert-missing / delete-absent; MI-9) and ``transcripts`` are
        write-once-preserve (insert / update / no-op-on-absent, **never** DELETE;
        V1-D16).

        Edge reconciliation is live as of 5b: ``_reconcile_edges`` commits the two
        kill-edges (``supersedes`` / ``corrects``) via declarative mirror, and a
        post-mutation slug-collision assertion (V1-D4) enforces one active node per
        ``casefold(slug)``. A referential violation rolls the whole entry back with
        a structured ``source="store"`` envelope. Outbox enqueue (5c) and the read
        views (5d) remain deferred seams. Format validation is the parser's job
        (C1 / Decision 2) — the store owns referential truth, not format truth.

        Args:
            parsed: A well-formed ``ParsedEntry`` (the parser's C1 guarantee).

        Returns:
            A ``CommitDelta`` carrying the committed ``node_id`` plus the scope /
            commentary-change metadata.

        Raises:
            ValidationError: If the canonical-core field that feeds the hash is
                empty or the kind is unknown (a caller that bypassed the parser).
            CommitError: On a store-stage referential violation (``missing_target``,
                ``dangling_edge``, ``cycle_violation``, ``kind_constraint_violation``,
                or ``slug_collision``), carrying the §5.2.2 ``EntryFailure`` envelope;
                the whole entry is rolled back (V1-D10).
            DatabaseError: On a SQLite failure (e.g. a ``source`` outside the DDL
                CHECK enum), with the whole entry rolled back.
        """
        # Structural guard, NOT a format re-implementation (Decision 2): the parser
        # owns format validation (C1). This only fences the single value that feeds
        # the hash — an empty canonical core would mint a degenerate node id — so a
        # caller that bypassed the parser fails with a clear vector (P3) instead of
        # silently corrupting identity. A raise (not a bare ``assert``) survives the
        # ``-O`` flag (the 2a IMPL_NOTES precedent: a vector error is durable).
        if parsed.kind == "decision":
            if not parsed.axiom:
                raise ValidationError(
                    f"Decision '{parsed.slug}' reached the store with an empty "
                    "axiom — the parser's format guarantee was bypassed."
                )
            node_id = compute_node_id(
                kind="decision",
                axiom=parsed.axiom,
                mechanism_refs=parsed.mechanisms,
            )
        elif parsed.kind == "open_question":
            if not parsed.topic:
                raise ValidationError(
                    f"Open question '{parsed.slug}' reached the store with an "
                    "empty topic — the parser's format guarantee was bypassed."
                )
            node_id = compute_node_id(
                kind="open_question",
                topic=parsed.topic,
                questions_raised=parsed.questions_raised,
            )
        else:
            # Mirror identity's guard so an unknown kind fails here, not deep in a
            # NULL-column INSERT.
            raise ValidationError(
                f"Cannot commit entry '{parsed.slug}': unknown kind "
                f"{parsed.kind!r} (expected 'decision' or 'open_question')."
            )

        # Kind-aware column values. Off-kind canonical/commentary columns are NULL
        # (never ``json.dumps("")`` of an absent core field): a decision's
        # topic/questions are NULL; an OQ's axiom/mechanism_refs/rejected_paths/
        # invalidates_if/context are NULL. Only id/kind/slug/slug_casefold/source/
        # created_at/updated_at are NOT NULL (§8.2).
        slug = parsed.slug
        slug_casefold = slug.casefold()  # MI-7: Python casefold, never SQLite LOWER
        source = parsed.source or "user"  # V1-D20: absent ``**Source:**`` -> "user"
        if parsed.kind == "decision":
            axiom = parsed.axiom
            mechanism_refs_json = json.dumps(parsed.mechanisms)
            topic = None
            questions_raised_json = None
            rejected_paths_json = parsed.rejected_paths  # raw string (§14 latitude)
            invalidates_if = parsed.invalidates_if
            context = parsed.context
        else:
            axiom = None
            mechanism_refs_json = None
            topic = parsed.topic
            questions_raised_json = json.dumps(parsed.questions_raised)
            rejected_paths_json = None
            invalidates_if = None
            context = None
        confirmed_by = parsed.confirmed_by  # reserved — NULL in V1a in practice
        confirmed_at = parsed.confirmed_at

        # Incoming scopes: strip + casefold + drop-empties (the parser already
        # normalizes; re-applying ``str.casefold()`` is idempotent and keeps a
        # hand-built entry honest — MI-9, never SQLite NOCASE/LOWER). A scope row
        # is never empty/NULL.
        incoming_scopes = {tag for s in parsed.scope if (tag := s.strip().casefold())}
        incoming_transcript = parsed.transcript or None  # falsy -> absent

        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()

                # --- Read prior state (same content-hash id => same node) -------
                prior = cursor.execute(
                    "SELECT * FROM nodes WHERE id = ?", (node_id,)
                ).fetchone()
                prior_scopes = {
                    r["scope"]
                    for r in cursor.execute(
                        "SELECT scope FROM node_scopes WHERE node_id = ?", (node_id,)
                    )
                }
                prior_tx_row = cursor.execute(
                    "SELECT transcript_text FROM transcripts WHERE node_id = ?",
                    (node_id,),
                ).fetchone()
                prior_transcript = (
                    prior_tx_row["transcript_text"] if prior_tx_row else None
                )

                is_new = prior is None

                # --- Diff the direct footprint (same-id path only) --------------
                # The commentary SET is enumerated explicitly (no SELECT *-style
                # overwrite): slug is mutable commentary, the canonical core and
                # ``source`` are fenced (MI-4) and never appear here.
                commentary = {
                    "slug": slug,
                    "slug_casefold": slug_casefold,
                    "rejected_paths_json": rejected_paths_json,
                    "invalidates_if": invalidates_if,
                    "context": context,
                    "confirmed_by": confirmed_by,
                    "confirmed_at": confirmed_at,
                }
                commentary_changed = (not is_new) and any(
                    prior[col] != val for col, val in commentary.items()
                )
                scopes_changed = incoming_scopes != prior_scopes
                # Transcript write-once-preserve (V1-D16): an absent incoming
                # transcript leaves the stored row untouched (never a change).
                transcript_changed = (
                    incoming_transcript is not None
                    and incoming_transcript != prior_transcript
                )
                direct_footprint_changed = (
                    commentary_changed or scopes_changed or transcript_changed
                )

                # --- Timestamp (MI-10): one application-supplied stamp per commit,
                # shared by the node write and any edge ``created_at``. Computed
                # unconditionally (it is cheap) so it is available to
                # ``_reconcile_edges`` (edge rows need it), but only *written* when
                # something actually changed — a byte-identical re-commit writes it
                # nowhere, so ``updated_at`` does not tick (MI-3 / V1-D17).
                now = _utc_now_iso()

                # --- Write the NEW nodes row (FK: must precede edge writes) ------
                # A new node is inserted here so the edge FK
                # (``edges.source_id -> nodes(id, kind)``) is satisfied when
                # ``_reconcile_edges`` runs below. The same-id commentary UPDATE is
                # DEFERRED to after the seam so its single ``updated_at`` tick can
                # also fire on an edge-set change (V1-D17); see the deferred write
                # below. A byte-identical re-commit writes the nodes row nowhere.
                if is_new:
                    cursor.execute(
                        """
                        INSERT INTO nodes (
                            id, kind, slug, slug_casefold, source,
                            axiom, mechanism_refs_json, topic, questions_raised_json,
                            rejected_paths_json, invalidates_if, context,
                            confirmed_by, confirmed_at, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            node_id, parsed.kind, slug, slug_casefold, source,
                            axiom, mechanism_refs_json, topic, questions_raised_json,
                            rejected_paths_json, invalidates_if, context,
                            confirmed_by, confirmed_at, now, now,
                        ),
                    )

                # --- Reconcile node_scopes (insert-missing / delete-absent) -----
                for tag in incoming_scopes - prior_scopes:
                    cursor.execute(
                        "INSERT INTO node_scopes (node_id, scope) VALUES (?, ?)",
                        (node_id, tag),
                    )
                for tag in prior_scopes - incoming_scopes:
                    cursor.execute(
                        "DELETE FROM node_scopes WHERE node_id = ? AND scope = ?",
                        (node_id, tag),
                    )

                # --- Transcripts: write-once-preserve (never DELETE) ------------
                if transcript_changed:
                    if prior_transcript is None:
                        cursor.execute(
                            "INSERT INTO transcripts (node_id, transcript_text) "
                            "VALUES (?, ?)",
                            (node_id, incoming_transcript),
                        )
                    else:
                        cursor.execute(
                            "UPDATE transcripts SET transcript_text = ? "
                            "WHERE node_id = ?",
                            (incoming_transcript, node_id),
                        )

                # --- Edge reconciliation (5b) + Outbox enqueue (5c) ------------
                # Both take the open cursor so they enlist in THIS transaction.
                # ``_reconcile_edges`` returns whether the outgoing edge set changed
                # (feeds the ``updated_at`` tick, V1-D17) plus any ``source="store"``
                # referential failures; it shares the single ``now`` stamp (MI-10)
                # for edge ``created_at``. ``_enqueue_outbox`` UPSERTs the committing
                # node's ``pending_embeddings`` Outbox row (unconditional, flagless,
                # drain-state-resetting; C2) sharing that same ``now`` — a
                # speculative write that the rollback below unwinds on any failure.
                edges_changed, store_failures, resurrected_slugs = (
                    self._reconcile_edges(cursor, node_id, parsed, now)
                )
                self._enqueue_outbox(cursor, node_id, parsed, now)
                store_failures = list(store_failures)

                # --- Same-id commentary write + the single ``updated_at`` tick --
                # Deferred to here (after the seam) so ``edges_changed`` can join
                # the tick condition: a same-id commit ticks ``updated_at`` on ANY
                # footprint change — commentary, scope, transcript, OR outgoing
                # edges (V1-D17). The commentary columns are re-asserted in the same
                # statement (they equal the stored values when only edges changed —
                # a harmless content no-op) so the timestamp is written from exactly
                # one place. The canonical core and ``source`` are NEVER touched
                # (MI-4). A new node already wrote its row above; a byte-identical
                # re-commit changes nothing here (MI-3).
                if (not is_new) and (direct_footprint_changed or edges_changed):
                    cursor.execute(
                        """
                        UPDATE nodes SET
                            slug = ?, slug_casefold = ?, rejected_paths_json = ?,
                            invalidates_if = ?, context = ?, confirmed_by = ?,
                            confirmed_at = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            slug, slug_casefold, rejected_paths_json,
                            invalidates_if, context, confirmed_by,
                            confirmed_at, now, node_id,
                        ),
                    )

                # --- Slug-collision assertion (V1-D4 / MI-13) -------------------
                # Application-layer BY NECESSITY: M3 forbids a DDL uniqueness
                # predicate over the kill-edge-filtered *active* view (a reflexive
                # ``UNIQUE(slug_casefold)`` would wrongly reject a legitimate
                # same-slug supersession). Run AFTER the node + edge writes land
                # (so a rename's new slug and the edge anti-join are both current).
                # Checked slugs: the committing slug PLUS every slug a
                # declarative-mirror DELETE may have resurrected — dropping a
                # ``Supersedes:`` line can reactivate a predecessor whose slug a
                # *different* active node has since taken (the cross-slug
                # resurrection collision; a committing-slug-only check misses it).
                # Each must resolve to exactly one ACTIVE node. Skipped when
                # edge-stage failures already exist — that failure is the actionable
                # one, and a count over a doomed edge set could mislead (Decision 4).
                if not store_failures:
                    slugs_to_check = [slug_casefold] + sorted(
                        resurrected_slugs - {slug_casefold}
                    )
                    for check_casefold in slugs_to_check:
                        active_slugs = [
                            r["slug"]
                            for r in cursor.execute(
                                f"""
                                SELECT n.slug AS slug FROM nodes n
                                WHERE n.slug_casefold = ?
                                  AND NOT EXISTS (
                                      SELECT 1 FROM edges e
                                      WHERE e.target_id = n.id
                                        AND e.edge_type IN {_KILL_EDGE_TYPES_SQL}
                                  )
                                """,
                                (check_casefold,),
                            ).fetchall()
                        ]
                        if len(active_slugs) > 1:
                            # Name the committing slug verbatim when it collides;
                            # otherwise name a representative active holder of the
                            # resurrected slug. One item is enough to roll the entry
                            # back and keeps the envelope single-item (as the
                            # FM2/independent/rename/case-variant tests assert).
                            display = (
                                slug
                                if check_casefold == slug_casefold
                                else active_slugs[0]
                            )
                            store_failures.append(
                                FailureItem(
                                    code=STORE_SLUG_COLLISION,
                                    source="store",
                                    message=(
                                        f"Slug '{display}' would resolve to "
                                        f"{len(active_slugs)} active entries; exactly "
                                        "one active entry may carry a slug. Supersede "
                                        "or rename the colliding entry."
                                    ),
                                    line_start=parsed.line_start,
                                    line_end=parsed.line_end,
                                )
                            )
                            break

                # --- Raise the single store-stage envelope on any failure -------
                # Raised INSIDE ``with conn:`` so the whole entry rolls back
                # (V1-D10). ``CommitError`` is a ``MitosError`` (NOT a
                # ``sqlite3.Error``), so it propagates past the SQLite handlers
                # below, carrying its structured ``EntryFailure`` to the caller.
                if store_failures:
                    raise CommitError(
                        f"Commit of '{parsed.slug}' failed store-stage referential "
                        "validation.",
                        failure=EntryFailure(
                            slug=parsed.slug,
                            line_start=parsed.line_start,
                            line_end=parsed.line_end,
                            items=store_failures,
                        ),
                    )

                # --- Build the delta --------------------------------------------
                # ``cascade_affected_scopes`` is FIRST-ORDER as of 5c (V1b adds the
                # transitive walker, §5.2.4): the union of (i) the committing node's
                # entered/left scopes when its own footprint changed (the 5a
                # behavior) and (ii) the scopes of the nodes this commit currently
                # kill-edges, when the outgoing edge set changed — a superseded /
                # corrected target leaves the active view, so every scope it was in
                # needs a re-render (C3). The targets are read from the
                # POST-reconciliation ``edges`` (the cursor sees 5b's just-written
                # rows; the read runs well after the seam at line 847). The
                # resurrection-target gap is deliberate (a DELETEd kill-edge's target
                # is no longer outgoing → not captured here; V1b's transitive walker
                # owns it — forward-safe: no live V1a consumer reads this field, only
                # ``delta.node_id``). The ``CommitDelta`` field shape is unchanged —
                # V1b extends the population, never rebuilds the struct. A
                # byte-identical re-commit trips neither gate -> ``[]``.
                affected = set()
                if is_new or direct_footprint_changed:
                    affected |= incoming_scopes | prior_scopes
                if edges_changed:
                    affected.update(
                        r["scope"]
                        for r in cursor.execute(
                            f"""
                            SELECT ns.scope AS scope FROM edges e
                            JOIN node_scopes ns ON ns.node_id = e.target_id
                            WHERE e.source_id = ?
                              AND e.edge_type IN {_KILL_EDGE_TYPES_SQL}
                            """,
                            (node_id,),
                        )
                    )
                cascade_affected_scopes = sorted(affected)

                return CommitDelta(
                    node_id=node_id,
                    node_scope=sorted(incoming_scopes),
                    self_old_scope=sorted(prior_scopes),
                    commentary_fields_changed=(
                        False if is_new else direct_footprint_changed
                    ),
                    cascade_affected_scopes=cascade_affected_scopes,
                )
        except sqlite3.IntegrityError as e:
            # The DDL CHECKs/FKs are the structural type gate (Lesson 2). The most
            # likely surface is a ``source`` outside the enum — name the field and
            # the bad value (P3 vector) and roll the whole entry back (V1-D10).
            raise DatabaseError(
                f"Commit of '{parsed.slug}' violated a database constraint "
                f"(source={source!r}?): {e}"
            )
        except sqlite3.Error as e:
            raise DatabaseError(f"SQLite transaction commit failed: {str(e)}")
        finally:
            conn.close()

    def _reconcile_edges(
        self,
        cursor: sqlite3.Cursor,
        node_id: str,
        parsed: ParsedEntry,
        now: str,
    ) -> Tuple[bool, List[FailureItem], Set[str]]:
        """Reconciles the committing node's outgoing edges (V1-D21/D6/D23/D13).

        Declarative mirror inside ``commit_parsed_entry``'s transaction: the node's
        stored outgoing edge set is brought into agreement with the buffer's
        declared relationship citations across **all nine** edge types — the two
        kill-edges (``Supersedes:`` / ``Corrects:``) and the seven non-kill types
        (``Amends:`` / ``Narrows:`` / ``Depends-On:`` / ``Resolves:`` /
        ``Contradicts:`` / ``Derives-From:`` / ``Cites:``). Each relationship field
        is comma-separated multi-valued (V1b): ``Cites: a, b`` reconciles two
        ``cites`` edges; a within-field repeat collapses to one. A declared edge
        already present (matched STRUCTURALLY by its stored target's *current* slug,
        never re-resolved) is RETAINED untouched — re-resolving a retained edge
        against the active view would fail when the target is inactive *because of
        that very edge* (the self-strangling trap). Only a NET-NEW declared edge is
        resolved against the active view and guarded; a stored edge no longer
        declared is DELETEd (the declarative mirror — removing a ``Supersedes:``
        line resurrects the target, §4.5.1).

        The guards fire only on net-new edges and report structured
        ``source="store"`` failures (the entry rolls back on any):

        - ``missing_target`` — no node carries the cited slug (a forward reference
          is always a causal violation; edges point new→old);
        - ``dangling_edge`` — a node carries it but is inactive (the immediate
          1-hop killer is named; no transitive walker — that is V5);
        - ``cycle_violation`` — a self-edge (a node cannot relate to itself), or a
          net-new **kill-edge** from an already-superseded source (the source-active
          reject is a kill-edge invariant per V1-D6; a non-kill edge mints from any
          source — a superseded node may still add a ``Cites:``);
        - ``kind_constraint_violation`` — a kind-violating edge (e.g. a ``resolves``
          D→D, or a decision-source ``derives_from``), caught from the widened DDL
          CHECK on the edge INSERT and mapped (Lesson 2 — never pre-asserted).

        All nine types resolve to the *active* view (V1-D23); a ``cites`` to a
        superseded target therefore fails ``dangling_edge`` (cite the live version).
        Removing a non-kill edge resurrects nothing (it retired no target), so only a
        removed kill-edge feeds the caller's resurrection slug-collision re-check.

        Args:
            cursor: The open cursor inside ``commit_parsed_entry``'s transaction —
                edge writes enlist in the commit transaction with no re-plumbing.
            node_id: The committing (source) node's content-hash id.
            parsed: The committing ``ParsedEntry`` (carries the declared edges).
            now: The commit's single ``_utc_now_iso()`` stamp (MI-10), reused for
                any edge ``created_at`` (the schema has no ``DEFAULT``).

        Returns:
            A ``(edges_changed, failures, resurrected_slugs)`` tuple.
            ``edges_changed`` is True iff an edge row was inserted or deleted (feeds
            the ``updated_at`` tick, V1-D17); ``failures`` is the accumulated list of
            ``source="store"`` ``FailureItem``s (empty on success);
            ``resurrected_slugs`` is the set of ``slug_casefold`` values whose target
            had an outgoing edge DELETEd this commit (declarative-mirror resurrection
            candidates). ``commit_parsed_entry`` runs the slug-collision assertion
            over these too: a resurrected predecessor may reactivate a slug a
            *different* active node has since taken (the cross-slug resurrection
            collision — distinct from the committing-slug FM2 case, and missed by a
            committing-slug-only check).
        """
        failures: List[FailureItem] = []
        edges_changed = False
        # slug_casefolds whose target had an edge DELETEd this commit (resurrection
        # candidates); the caller's slug-collision assertion must also cover these so
        # a removed kill-edge cannot reactivate a node onto an occupied active slug
        # (V1-D4 / MI-13).
        resurrected_slugs: Set[str] = set()

        # --- Declared edges, all nine types (strip brackets, casefold) ----------
        # Each relationship field is a ``List[str]`` of cited slugs (V1b
        # multi-valued); iterate the full nine-type catalog and expand each field
        # into per-citation declarations. ``declared`` drives net-new resolution;
        # ``declared_keys`` (the casefolded target slug + edge_type) drives the
        # declarative-mirror DELETE (V1-D21). A within-field repeat (``Cites: a, a``)
        # collapses to a single declaration — the edge PK is
        # ``(source_id, target_id, edge_type)``, so a second INSERT of the same
        # triple would trip the PK (and mis-map to a kind violation in the catch).
        declared: List[Tuple[str, str, str]] = []  # (edge_type, casefold, display)
        declared_keys: Set[Tuple[str, str]] = set()
        for edge_type in _KILL_EDGE_FIELDS + _DEFERRED_EDGE_FIELDS:
            for raw in (getattr(parsed, edge_type, None) or []):
                citation = _strip_citation(raw)
                if not citation:  # an empty / bracket-only value is not a declaration
                    continue
                citation_casefold = citation.casefold()  # MI-7: never SQLite LOWER
                key = (citation_casefold, edge_type)
                if key in declared_keys:  # within-field / repeated citation → one edge
                    continue
                declared.append((edge_type, citation_casefold, citation))
                declared_keys.add(key)

        # --- Stored outgoing edges, keyed by the target's CURRENT slug (V1-D23) --
        # Matching retained edges by the target's current slug means a renamed-then-
        # cited target resolves correctly, and a stale citation of the OLD name
        # falls through to net-new -> missing_target (the correct strict rejection).
        stored_by_key: Dict[Tuple[str, str], str] = {}
        for row in cursor.execute(
            "SELECT e.target_id AS target_id, e.edge_type AS edge_type, "
            "n.slug_casefold AS slug_casefold "
            "FROM edges e JOIN nodes n ON n.id = e.target_id "
            "WHERE e.source_id = ?",
            (node_id,),
        ).fetchall():
            stored_by_key[(row["slug_casefold"], row["edge_type"])] = row["target_id"]

        # --- The committing (source) node's own activeness ----------------------
        # Shared by every net-new edge this commit: outgoing INSERTs never change
        # the source's *incoming* set, so this is stable across the loop (compute
        # once). A kill-edge mints only from an active source (V1-D6).
        source_active = (
            cursor.execute(
                "SELECT 1 FROM edges WHERE target_id = ? "
                f"AND edge_type IN {_KILL_EDGE_TYPES_SQL} LIMIT 1",
                (node_id,),
            ).fetchone()
            is None
        )

        # --- Resolve + guard each declared edge (all nine types) ----------------
        for edge_type, citation_casefold, citation in declared:
            if (citation_casefold, edge_type) in stored_by_key:
                # RETAINED — matched structurally, never re-resolved (the
                # self-strangle fix). An unchanged re-declaration is a pure no-op:
                # no re-resolution, no guard, no write (this is what satisfies MI-5
                # for a commentary-only re-commit).
                continue

            token = _RELATION_TOKENS[edge_type]

            # NET-NEW — resolve the citation against the *active* view. Multiple
            # nodes may share a casefolded slug (a same-slug supersession lineage);
            # the active one (no incoming kill-edge) is the resolution target.
            candidates = cursor.execute(
                "SELECT id, kind FROM nodes WHERE slug_casefold = ?",
                (citation_casefold,),
            ).fetchall()
            active_candidates = [
                c
                for c in candidates
                if cursor.execute(
                    "SELECT 1 FROM edges WHERE target_id = ? "
                    f"AND edge_type IN {_KILL_EDGE_TYPES_SQL} LIMIT 1",
                    (c["id"],),
                ).fetchone()
                is None
            ]
            # Prefer an active target that is NOT the committing node: in a
            # same-slug supersession the new node shares the slug with the one it
            # supersedes (Q5), so "x" must resolve to the OTHER node, not self.
            active_non_self = [c for c in active_candidates if c["id"] != node_id]

            if active_non_self:
                target = active_non_self[0]
                # The "itself superseded" reject is a KILL-EDGE invariant (V1-D6): a
                # kill-edge mints only from an active source. A non-kill edge (the
                # seven) mints from any source — a superseded node may still add a
                # ``Cites:`` (the mutation-cycle concern for amends/narrows is 3a's
                # write-time reachability probe, not a blanket source-active reject).
                if edge_type in _KILL_EDGE_FIELDS and not source_active:
                    failures.append(
                        FailureItem(
                            code=STORE_CYCLE_VIOLATION,
                            source="store",
                            message=(
                                "This entry is itself superseded, so it cannot "
                                f"{edge_type} '{citation}' — a kill-edge mints only "
                                "from an active source."
                            ),
                            field=token,
                            line_start=parsed.line_start,
                            line_end=parsed.line_end,
                        )
                    )
                    continue
                try:
                    cursor.execute(
                        "INSERT INTO edges (source_id, source_kind, target_id, "
                        "target_kind, edge_type, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            node_id,
                            parsed.kind,
                            target["id"],
                            target["kind"],
                            edge_type,
                            now,
                        ),
                    )
                    edges_changed = True
                except sqlite3.IntegrityError:
                    # The ONLY IntegrityError reachable here is the kind-matrix DDL
                    # CHECK: net-new => no PK dup; a real resolved target with its
                    # true kind => the composite FK is satisfied. Catch tightly +
                    # map (Lesson 2) so it never reaches the outer source-enum
                    # handler in ``commit_parsed_entry``.
                    failures.append(
                        FailureItem(
                            code=STORE_KIND_CONSTRAINT_VIOLATION,
                            source="store",
                            message=(
                                f"'{citation}' has an incompatible kind for a "
                                f"{edge_type} edge; {edge_type} edges must "
                                f"{_EDGE_KIND_REQUIREMENT[edge_type]}."
                            ),
                            field=token,
                            line_start=parsed.line_start,
                            line_end=parsed.line_end,
                        )
                    )
            elif active_candidates:
                # The only active carrier of the slug is the committing node itself:
                # a node cannot supersede/correct itself. Distinct from the FM1
                # collision case, where the citation resolves to the OTHER same-slug
                # node and commits.
                failures.append(
                    FailureItem(
                        code=STORE_CYCLE_VIOLATION,
                        source="store",
                        message=(
                            f"An entry cannot {edge_type} itself ('{citation}' "
                            "resolves to this same entry)."
                        ),
                        field=token,
                        line_start=parsed.line_start,
                        line_end=parsed.line_end,
                    )
                )
            elif candidates:
                # A node carries the slug but every carrier is inactive: report the
                # immediate 1-hop killer of one inactive carrier (a single anti-join
                # hop — NO transitive chain-head walker, that is V1b).
                killer = cursor.execute(
                    "SELECT n.slug AS slug FROM edges e "
                    "JOIN nodes n ON n.id = e.source_id "
                    "WHERE e.target_id = ? "
                    f"AND e.edge_type IN {_KILL_EDGE_TYPES_SQL} LIMIT 1",
                    (candidates[0]["id"],),
                ).fetchone()
                killer_slug = killer["slug"] if killer else "another entry"
                failures.append(
                    FailureItem(
                        code=STORE_DANGLING_EDGE,
                        source="store",
                        message=(
                            f"'{citation}' is inactive (superseded by "
                            f"'{killer_slug}'); a {edge_type} edge must target an "
                            "active entry."
                        ),
                        field=token,
                        line_start=parsed.line_start,
                        line_end=parsed.line_end,
                    )
                )
            else:
                # No node carries the slug at all. Edges point new→old, so a
                # forward reference is always a causal violation (strict rejection,
                # never a "resolve later" deferral — §1.1).
                failures.append(
                    FailureItem(
                        code=STORE_MISSING_TARGET,
                        source="store",
                        message=(
                            f"'{citation}' does not match any entry in the graph. "
                            "Edges point newer→older, so the cited entry must "
                            "already exist."
                        ),
                        field=token,
                        line_start=parsed.line_start,
                        line_end=parsed.line_end,
                    )
                )

        # --- Declarative-mirror DELETE: stored edges no longer declared (V1-D21) -
        for (slug_casefold, edge_type), target_id in stored_by_key.items():
            if (slug_casefold, edge_type) not in declared_keys:
                cursor.execute(
                    "DELETE FROM edges WHERE source_id = ? AND target_id = ? "
                    "AND edge_type = ?",
                    (node_id, target_id, edge_type),
                )
                edges_changed = True
                # Only a removed KILL-edge can reactivate a target (it was inactive
                # *because of* that edge), so only it feeds the caller's resurrection
                # slug-collision re-check (§4.5.1 / V1-D4). A removed non-kill edge
                # (cites/amends/…) changes no node's activeness; adding it here would
                # trigger a spurious collision re-check (Decision 5).
                if edge_type in _KILL_EDGE_FIELDS:
                    resurrected_slugs.add(slug_casefold)

        return edges_changed, failures, resurrected_slugs

    def _enqueue_outbox(
        self, cursor: sqlite3.Cursor, node_id: str, parsed: ParsedEntry, now: str
    ) -> None:
        """Conservative flagless "node-changed" Outbox enqueue (C2 / §5.1).

        UPSERTs exactly ONE ``pending_embeddings`` row for the committing node,
        **unconditionally** — commentary-only and byte-identical re-commits
        included — inside ``commit_parsed_entry``'s transaction (the row commits
        with the node or not at all; a rolled-back commit enqueues nothing —
        MI-12 / V1-D10). The row carries no ``task_type`` / flag and no embedding
        text: the V3b drainer re-derives ``embedding_text(node)`` at drain (a pure
        function of the immutable core, M8) and the embedding cache returns it at
        zero token cost, so a bare "node-changed" signal is the whole contract.

        The enqueue is **never** skipped (recall-over-precision, OPERA §11): a
        missed enqueue is permanent vector staleness (the costly miss), a false one
        is a free drain-time cache-hit (the cheap false positive). Commentary is
        not forever vector-irrelevant — V2's anti-knowledge vector keys on
        ``rejected_paths`` (commentary) and churns on commentary edits by design,
        so a commentary-blind enqueue would silently starve it the moment V2 ships.

        A re-enqueue RESETS drain state (re-stamps ``queued_at``, clears
        ``retry_count`` to 0) rather than a bare idempotent no-op: a node V3b
        dead-lettered on a *transient* provider outage must be revived by a
        deliberate re-commit — and the byte-identical re-commit IS that retry — or
        it would be stranded at max-retry forever (P5 self-healing). This is the
        deliberate exception to MI-3's "true no-op": MI-3 governs the **node** row
        (no tick, no new node), MI-12's reset clause governs the **Outbox** row;
        the two coexist on purpose.

        Kind-agnostic (every committed node may need embedding — no kind branch).
        Writes raw SQL against the three-column V1a shape; it does NOT route through
        the prototype drain methods, which bind the dropped
        ``embedding_text``/``attempts``/``claimed_by`` columns (8a reconciles them).

        Args:
            cursor: The open cursor inside ``commit_parsed_entry``'s transaction —
                the enqueue enlists in the commit transaction with no re-plumbing.
            node_id: The committing node's content-hash id — the
                ``pending_embeddings`` PK and its FK to ``nodes(id)`` (the node row
                already exists by this point, so the FK is always satisfied).
            parsed: The committing ``ParsedEntry`` (retained for seam-shape
                stability; the flagless body does not read it).
            now: The commit's single UTC ISO-8601 µs stamp (MI-10) — shared with
                the node and edge ``created_at`` writes; never re-read inside the
                seam (one stamp per commit).
        """
        cursor.execute(
            """
            INSERT INTO pending_embeddings (node_id, queued_at, retry_count)
            VALUES (?, ?, 0)
            ON CONFLICT(node_id) DO UPDATE SET
                queued_at = excluded.queued_at,
                retry_count = 0
            """,
            (node_id, now),
        )

    def get_active_decisions(self, scope: Optional[str] = None) -> List[Dict[str, Any]]:
        """Exposes the active-view set of decisions (C3 / M3).

        Activeness is the kill-edge anti-join (no incoming ``supersedes`` /
        ``corrects``), computed at query time — never the prototype
        ``compute_all_states`` DAG (which reads dropped edge columns). A drifted
        node is INCLUDED with ``is_drifted == True`` (drift annotates, does not
        retire). Each result is hydrated to the reader-facing shape and
        modifier-stamped (provably empty on the active view, but the seam ships).

        Args:
            scope: Optional scope tag; filtered in SQL via the indexed
                ``node_scopes`` membership join (P11), not an O(N) post-filter.

        Returns:
            Active decision node dicts (reader-facing shape).
        """
        conn = self._get_connection()
        try:
            sql = (
                f"SELECT nodes.*, {_IS_DRIFTED_SQL} FROM nodes "
                f"WHERE nodes.kind = 'decision' AND {_ACTIVE_VIEW_PREDICATE}"
            )
            params: List[str] = []
            if scope is not None:
                sql += _SCOPE_FILTER_SQL
                params.append(scope)
            rows = conn.execute(sql, params).fetchall()
            return self._hydrate_rows(conn, rows)
        finally:
            conn.close()

    def get_all_scopes(self) -> List[str]:
        """Returns a sorted list of all unique scope tags currently in use."""
        conn = self._get_connection()
        try:
            rows = conn.execute("SELECT DISTINCT scope FROM node_scopes ORDER BY scope").fetchall()
            return [row["scope"] for row in rows]
        finally:
            conn.close()

    def get_decisions(self, scope: Optional[str] = None, state: str = "active") -> List[Dict[str, Any]]:
        """Enumerates the COMPLETE set of decision nodes matching scope and state.

        The exhaustive, deterministic counterpart to the semantic recall path
        (``surface``/``query``, which rank and cap at the top few). No top-k, no
        ranking — every matching decision, straight from the graph — so a caller
        can be certain it has seen every settled call in a scope.

        Args:
            scope: Optional scope tag filter; omit for the whole project.
            state: ``"active"`` (default) returns the live set (active + drifted);
                ``"all"`` returns every decision regardless of state; any other
                value is an exact computed-state match (e.g. "superseded").

        Returns:
            Decision node dicts (with ``computed_state`` attached), unbounded.
        """
        conn = self._get_connection()
        try:
            sql = (
                f"SELECT nodes.*, {_IS_DRIFTED_SQL}, {_KILLER_TYPE_SQL} "
                f"FROM nodes WHERE nodes.kind = 'decision'"
            )
            params: List[str] = []
            if scope is not None:
                sql += _SCOPE_FILTER_SQL
                params.append(scope)
            rows = conn.execute(sql, params).fetchall()

            # Derive computed_state from the incoming kill-edge (active vs
            # superseded/corrected) and filter BEFORE hydrating, so the bulk scope
            # fetch + modifier stamp only runs over the kept rows. The exact
            # vocabulary is validated against ``list_decisions`` at 8a (§14) — kept
            # deliberately simple here.
            kept: List[Tuple[sqlite3.Row, str]] = []
            for row in rows:
                computed_state = _computed_decision_state(row["killer_type"])
                if state_matches(computed_state, state):
                    kept.append((row, computed_state))

            nodes = self._hydrate_rows(conn, [row for row, _ in kept])
            for node, (_, computed_state) in zip(nodes, kept):
                node["computed_state"] = computed_state
            return nodes
        finally:
            conn.close()

    def get_open_questions(self, scope: Optional[str] = None) -> List[Dict[str, Any]]:
        """Exposes the active-view set of open questions (V1-D18 Stage-1).

        Applies the SAME kill-edge anti-join as the decision views (an OQ with an
        incoming ``supersedes`` / ``corrects`` is excluded); there is no ``state``
        column. Hydrated to the reader-facing OQ shape (``topic`` /
        ``questions_raised``) and modifier-stamped.

        Args:
            scope: Optional scope tag; filtered in SQL via the indexed
                ``node_scopes`` membership join (P11).

        Returns:
            Active open-question node dicts (reader-facing shape).
        """
        conn = self._get_connection()
        try:
            sql = (
                f"SELECT nodes.*, {_IS_DRIFTED_SQL} FROM nodes "
                f"WHERE nodes.kind = 'open_question' AND {_ACTIVE_VIEW_PREDICATE}"
            )
            params: List[str] = []
            if scope is not None:
                sql += _SCOPE_FILTER_SQL
                params.append(scope)
            rows = conn.execute(sql, params).fetchall()
            return self._hydrate_rows(conn, rows)
        finally:
            conn.close()

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Retrieves every node (any kind, any state) with its computed state.

        Unfiltered — both kinds, active and inactive — each hydrated to the
        reader-facing shape, modifier-stamped, and carrying ``computed_state``
        derived from the kill-edge anti-join (``active`` / ``superseded`` /
        ``corrected``).

        Returns:
            All node dicts (reader-facing shape).
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                f"SELECT nodes.*, {_IS_DRIFTED_SQL}, {_KILLER_TYPE_SQL} FROM nodes"
            ).fetchall()
            states = [_computed_decision_state(row["killer_type"]) for row in rows]
            nodes = self._hydrate_rows(conn, rows)
            for node, computed_state in zip(nodes, states):
                node["computed_state"] = computed_state
            return nodes
        finally:
            conn.close()
            
    def get_edges(self) -> List[Dict[str, str]]:
        """Retrieves all edges in the database."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM edges")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_modifiers_map(self, node_ids: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """Maps each node to the slugs of later decisions that modify it.

        For every node in ``node_ids`` that is the target of one or more
        ``supersedes``/``amends``/``narrows``/``corrects`` edges, returns the
        modifying decisions grouped by reverse-relation key (``superseded_by``,
        ``amended_by``, ``narrowed_by``, ``corrected_by``). A reader stamps these
        onto a retrieved payload so a still-``active`` axiom can't masquerade as the
        live mechanism when a later decision has moved on from it.

        Args:
            node_ids: The node IDs to look up modifiers for.

        Returns:
            A mapping of node_id -> {reverse_key: [modifier_slug, ...]}. Only nodes
            that actually have modifiers appear, and each carries only the
            reverse-relation keys that are non-empty (an unmodified node is absent).
        """
        if not node_ids:
            return {}
        conn = self._get_connection()
        try:
            return self._modifiers_map(conn, node_ids)
        finally:
            conn.close()

    def get_modifiers(self, node_id: str) -> Dict[str, List[str]]:
        """Returns the reverse-relation modifiers for a single node.

        Single-node convenience over :meth:`get_modifiers_map`.

        Args:
            node_id: The node ID to look up modifiers for.

        Returns:
            A mapping of reverse-relation key to modifier slugs, or ``{}`` when the
            node is unmodified.
        """
        return self.get_modifiers_map([node_id]).get(node_id, {})

    def get_transcript(self, node_id: str) -> Optional[str]:
        """Returns a node's own committed transcript text, or None.

        The V1a primitive is the DIRECT self-read: ``transcripts.transcript_text``
        for this node only. It is the single store-owned transcript accessor —
        never reimplemented per consumer.

        **Cross-vision contract (pinned; the transitive walk is NOT in V1a).** The
        ancestry edge set is ``corrects`` ONLY, and the walk STOPS at any
        ``supersedes`` (the decision boundary): a ``corrects`` is a bug-fix on the
        *same* decision, so an ancestor's synthesis still describes it and may be
        borrowed; a ``supersedes`` moves the decision forward, so the predecessor's
        reasoning must NOT be borrowed. The transitive ``corrects``-chain traversal
        lands with V1b's cascade walker / V5 as first consumer (§5.2.3; ADR
        ``transcript-ancestry-corrects-only``) — for the V1a direct read the rule
        holds trivially (a single node's own transcript borrows nothing).

        Args:
            node_id: The node whose transcript to read.

        Returns:
            The raw transcript text, or None when the node has no transcript.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT transcript_text FROM transcripts WHERE node_id = ?",
                (node_id,),
            ).fetchone()
            return row["transcript_text"] if row else None
        finally:
            conn.close()

    def query_letter(
        self,
        *,
        scope: Optional[str] = None,
        kind: str = "decision",
        slug: Optional[str] = None,
        node_id: Optional[str] = None,
        brief: bool = False,
    ) -> List[Dict[str, Any]]:
        """Structured-filter Letter query over the active view (C4) — deterministic only.

        The deterministic C4 retrieval surface: filters the ACTIVE view by any of
        ``scope`` (indexed ``node_scopes`` join), ``kind``, exact ``slug``
        (``slug_casefold``), or ``node_id`` (the content-hash PK) and returns the
        Letter projection ``{slug, axiom, rejected_paths, scope}`` plus stamped
        reverse-relation modifier keys. There is deliberately NO vector / embedding
        path in V1a — this is the structured-filter retrieval, not the ranked recall.

        Note the Letter payload uses ``axiom`` (the C4 projection name), whereas the
        full-node read methods use ``core_axiom`` — two deliberate projections of the
        same column; do not unify them.

        ``brief`` is an explicit per-call argument — NEVER inferred from session or
        connection state (a bare ``/clear`` keeps a connection alive while resetting
        the agent's context, so no connection key can correctly decide "give me
        less"). It drops ``rejected_paths`` only; ``axiom`` and the modifier keys are
        always carried.

        Args:
            scope: Optional scope tag (indexed membership filter).
            kind: ``"decision"`` (default) or ``"open_question"``.
            slug: Optional exact slug (matched case-insensitively via casefold).
            node_id: Optional exact content-hash id.
            brief: When True, omit ``rejected_paths`` (axiom-only scan); modifiers
                and ``axiom`` are still carried.

        Returns:
            Letter payload dicts (active-view only), each modifier-stamped.
        """
        conn = self._get_connection()
        try:
            sql = (
                f"SELECT nodes.* FROM nodes "
                f"WHERE nodes.kind = ? AND {_ACTIVE_VIEW_PREDICATE}"
            )
            params: List[str] = [kind]
            if node_id is not None:
                sql += " AND nodes.id = ?"
                params.append(node_id)
            if slug is not None:
                sql += " AND nodes.slug_casefold = ?"
                params.append(slug.casefold())
            if scope is not None:
                sql += _SCOPE_FILTER_SQL
                params.append(scope)
            rows = conn.execute(sql, params).fetchall()
            if not rows:
                return []

            ids = [row["id"] for row in rows]
            scopes_map = self._scopes_for(conn, ids)
            mod_map = self._modifiers_map(conn, ids)
            payloads: List[Dict[str, Any]] = []
            for row in rows:
                rid = row["id"]
                payload: Dict[str, Any] = {
                    "slug": row["slug"],
                    "axiom": row["axiom"] or "",
                    "scope": scopes_map.get(rid, []),
                }
                if not brief:
                    # rejected_paths_json is a RAW string (5a §14) — verbatim.
                    payload["rejected_paths"] = (
                        row["rejected_paths_json"]
                        if row["rejected_paths_json"] is not None
                        else ""
                    )
                for key, slugs in mod_map.get(rid, {}).items():
                    payload[key] = slugs
                payloads.append(payload)
            return payloads
        finally:
            conn.close()

    def add_pending_embedding(self, node_id: str) -> None:
        """Enqueues a node onto the pending-embeddings Outbox (V1a 3-column shape).

        Aligned to the V1a ``pending_embeddings`` schema (Phase 8a): no
        ``embedding_text`` column — the drainer re-derives ``embedding_text(node)``
        at drain time (a pure function of the immutable core, C2/M8), so nothing is
        stored. A re-enqueue UPSERT resets drain state (re-stamps ``queued_at``,
        clears ``retry_count``) — the same conservative flagless contract
        ``_enqueue_outbox`` (5c) writes inside the commit transaction. This public
        method is the standalone-connection twin of that in-transaction enqueue,
        retained as part of the drain surface ``sync`` binds (R12).
        """
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO pending_embeddings (node_id, queued_at, retry_count)
                    VALUES (?, ?, 0)
                    ON CONFLICT(node_id) DO UPDATE SET
                        queued_at = excluded.queued_at,
                        retry_count = 0
                    """,
                    (node_id, _utc_now_iso())
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to add pending embedding: {str(e)}")
        finally:
            conn.close()

    def get_pending_embeddings(self) -> List[Dict[str, Any]]:
        """Retrieves all pending embeddings in the queue."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM pending_embeddings")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def remove_pending_embedding(self, node_id: str) -> None:
        """Removes a resolved node from the pending embeddings queue."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("DELETE FROM pending_embeddings WHERE node_id = ?", (node_id,))
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to remove pending embedding: {str(e)}")
        finally:
            conn.close()

    def increment_pending_attempts(self, node_id: str) -> None:
        """Increments the retry count for a pending embedding (V1a 3-column shape).

        Aligned to V1a (Phase 8a): ``attempts``→``retry_count``, and the
        ``claimed_by = NULL`` release is dropped — V1a is single-writer
        (``busy_timeout``), so the claim machinery defers to V3b (§5.2.8).
        """
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE pending_embeddings
                    SET retry_count = retry_count + 1
                    WHERE node_id = ?
                    """,
                    (node_id,)
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to increment pending attempts: {str(e)}")
        finally:
            conn.close()

    def claim_pending_embeddings(self, drainer_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Reads a batch of pending embeddings to drain (V1a single-writer semantics).

        Aligned to V1a (Phase 8a): the prototype's ``claimed_by`` claim machinery
        (UPDATE ... RETURNING) defers to V3b (§5.2.8) — V1a is single-writer
        (``busy_timeout``), so a plain ordered SELECT is the whole contract. The
        returned rows carry NO ``embedding_text`` (the column is gone, C2/M8); the
        drainer re-derives it per node. ``drainer_id`` is retained for surface
        stability (R12) and is unused under single-writer semantics.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT node_id, queued_at, retry_count
                FROM pending_embeddings
                ORDER BY queued_at
                LIMIT ?
                """,
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to claim pending embeddings: {str(e)}")
        finally:
            conn.close()

    def release_pending_embeddings(self, drainer_id: str) -> None:
        """No-op under V1a single-writer semantics (Phase 8a; retained for surface stability).

        The prototype released ``claimed_by`` holds; V1a has no ``claimed_by`` column
        (the claim machinery defers to V3b, §5.2.8), so there is nothing to release.
        The method name/signature are preserved because ``sync._drain_embeddings``
        binds it (R12).
        """
        return None
