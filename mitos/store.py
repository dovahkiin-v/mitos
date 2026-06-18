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
from mitos.migrations import run_migrations, is_pre_v1a_schema
from mitos.parser import ParsedEntry

# Module logger for non-failing notices (the warn-defer channel). The store is a
# pure primitive — it logs (loud, testable via ``caplog``, no raw stdout I/O) and
# never prints to the user; user-facing notices belong to the sync/cli consumer
# layer.
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

# Display tokens for the kill-edge fields, used for ``FailureItem.field``
# localization (the store has no finer per-line anchor than the entry span).
_KILL_EDGE_TOKENS: Dict[str, str] = {
    "supersedes": "**Supersedes:**",
    "corrects": "**Corrects:**",
}

# The seven relationship types parsed onto ``ParsedEntry`` but NOT committed in
# V1a — warn-deferred to V1b's reconciler. A deferred declaration is a logged
# notice, never a §5.2.2 failure, and the node + its kill-edges still commit
# (keeps 8b's self-parse closeout achievable).
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


def compute_hash(
    kind: str,
    slug: str,
    core_axiom: str = "",
    mechanisms: List[str] = [],
    questions_raised: List[str] = []
) -> str:
    """Computes a stable, deterministic SHA-256 hash for content-hash identity (M2).

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
            # V1a boot (entry-001 flip): the V1a STRICT schema now boots via the
            # migration ladder (``run_migrations`` -> ``_v1_schema``), not the
            # prototype ``_init_db``. Read-only stores never migrate (§7 gotcha):
            # a mode=ro connection cannot run the ladder's write transaction.
            conn = self._get_connection()
            try:
                # Refuse a prototype graph rather than ladder-advance it into an
                # undiagnosable hybrid (R3/R11). ``is_pre_v1a_schema`` is False for
                # a fresh/empty DB and for any V1a-or-later DB, True only for a
                # prototype graph — route it to the §2.1 one-time cutover (whose
                # tooling ships at Phase 7). The RO "not-ready" status surface is
                # Phase 6b, not 5a.
                if is_pre_v1a_schema(conn):
                    raise DatabaseError(
                        "This graph predates the V1a schema (a prototype layout "
                        "was detected). Mitos will not migrate it in place. Run "
                        "the one-time cutover to rebuild it into the V1a store."
                    )
                run_migrations(conn)
            finally:
                conn.close()

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
        """Resolves a slug to node IDs (case-insensitive with collision check).

        Args:
            slug: The slug string to find.

        Returns:
            A list of matching node IDs.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM nodes WHERE slug = ? COLLATE NOCASE",
                (slug,)
            )
            rows = cursor.fetchall()
            if not rows:
                # Fuzzy fallback for legacy import naming styles
                cursor.execute(
                    "SELECT id FROM nodes WHERE (slug LIKE ? OR slug LIKE ?) COLLATE NOCASE",
                    (f"{slug}:%", f"{slug}-%")
                )
                rows = cursor.fetchall()
            return [row["id"] for row in rows]
        finally:
            conn.close()

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single node by its ID.

        Args:
            node_id: The primary key ID of the node.

        Returns:
            A dictionary containing the node data, or None if not found.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            if not row:
                return None
            
            node = dict(row)
            # Deserialize JSON fields
            for field in ("questions_raised", "mechanisms", "scope"):
                if node.get(field):
                    node[field] = json.loads(node[field])
                else:
                    node[field] = []
            return node
        finally:
            conn.close()

    def get_node_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single node by slug, throwing on ambiguity.

        Args:
            slug: The slug identifier.

        Returns:
            The node dictionary or None.
        """
        node_ids = self.resolve_slug(slug)
        if not node_ids:
            return None
        if len(node_ids) > 1:
            raise ValidationError(
                f"Slug '{slug}' is ambiguous and maps to multiple nodes: {node_ids}"
            )
        return self.get_node(node_ids[0])

    def compute_all_states(self, conn: sqlite3.Connection) -> Dict[str, str]:
        """Computes states for all nodes currently in the transaction connection.

        Implements the M3 principle (computed state derived at runtime).
        - Decision state: 'active | superseded | drifted'
        - Open Question state: 'parked | resolved'

        Returns:
            A mapping of node_id -> state_string.
        """
        cursor = conn.cursor()
        
        # Load all nodes
        cursor.execute("SELECT id, kind, scope FROM nodes")
        nodes = {row["id"]: {"kind": row["kind"], "scope": json.loads(row["scope"] or "[]")} for row in cursor.fetchall()}
        
        # Load all edges of type 'supersedes', 'corrects', 'resolves'
        cursor.execute("SELECT from_id, to_id, type FROM edges WHERE type IN ('supersedes', 'corrects', 'resolves')")
        edges = cursor.fetchall()
        
        # Load drifted signals
        cursor.execute("SELECT node_id FROM signals WHERE type = 'drifted'")
        drifted_nodes = {row["node_id"] for row in cursor.fetchall()}

        # Build incoming edge index: to_id -> list of (from_id, type)
        incoming: Dict[str, List[Tuple[str, str]]] = {}
        for edge in edges:
            to_id = edge["to_id"]
            from_id = edge["from_id"]
            etype = edge["type"]
            incoming.setdefault(to_id, []).append((from_id, etype))

        computed_states: Dict[str, str] = {}

        # First, evaluate Decision nodes. Since it's a DAG of overrides (new corrects/supersedes old),
        # we can evaluate active nodes by checking if they are corrected or superseded
        # by an active node. Since it can be recursive, we do a topological walk.
        # But simpler: in v0.1, a node is superseded if there is ANY active node in its override lineage.
        # Let's perform a simple fixed-point evaluation of active decisions.
        active_decisions: Set[str] = {nid for nid, n in nodes.items() if n["kind"] == "decision"}
        
        changed = True
        while changed:
            changed = False
            to_remove = set()
            for nid in active_decisions:
                # Check if any incoming corrects/supersedes is from another active decision
                for parent_id, etype in incoming.get(nid, []):
                    if etype in ("supersedes", "corrects") and parent_id in active_decisions:
                        to_remove.add(nid)
                        changed = True
                        break
            if to_remove:
                active_decisions -= to_remove

        # Assign states to Decisions
        for nid, node in nodes.items():
            if node["kind"] == "decision":
                if nid in active_decisions:
                    if nid in drifted_nodes:
                        computed_states[nid] = "drifted"
                    else:
                        computed_states[nid] = "active"
                else:
                    computed_states[nid] = "superseded"

        # Assign states to Open Questions
        # Open Question state is 'resolved' if there is an active decision resolving it.
        for nid, node in nodes.items():
            if node["kind"] == "open_question":
                is_resolved = False
                for parent_id, etype in incoming.get(nid, []):
                    if etype == "resolves" and computed_states.get(parent_id) in ("active", "drifted"):
                        is_resolved = True
                        break
                
                computed_states[nid] = "resolved" if is_resolved else "parked"

        return computed_states

    def write_signal(self, node_id: str, stype: str, actor: Optional[str] = None) -> None:
        """Writes a signal row (drifted, source_reencounter) using INSERT OR IGNORE.

        Ensures unique constraints do not crash transactions (signals-insert-or-ignore).
        """
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO signals (node_id, type, actor)
                    VALUES (?, ?, ?)
                    """,
                    (node_id, stype, actor)
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
                # for edge ``created_at``. ``_enqueue_outbox`` is still a no-op (5c).
                edges_changed, store_failures, resurrected_slugs = (
                    self._reconcile_edges(cursor, node_id, parsed, now)
                )
                self._enqueue_outbox(cursor, node_id, parsed)
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
                                """
                                SELECT n.slug AS slug FROM nodes n
                                WHERE n.slug_casefold = ?
                                  AND NOT EXISTS (
                                      SELECT 1 FROM edges e
                                      WHERE e.target_id = n.id
                                        AND e.edge_type IN ('supersedes', 'corrects')
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
                # ``cascade_affected_scopes`` is committing-node first-order in 5a
                # (5c finalizes the kill-edge-target population once 5b's edges
                # exist): a footprint change touches the rendered view of every
                # scope the node entered or left. Only ``delta.node_id`` is read by
                # live consumers, so narrowing the population is consumer-safe.
                if is_new or direct_footprint_changed:
                    cascade_affected_scopes = sorted(incoming_scopes | prior_scopes)
                else:
                    cascade_affected_scopes = []

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
    ) -> Tuple[bool, List[FailureItem]]:
        """Reconciles the committing node's outgoing kill-edges (V1-D21/D6/D23).

        Declarative mirror inside ``commit_parsed_entry``'s transaction: the node's
        stored outgoing edge set is brought into agreement with the buffer's
        declared ``Supersedes:`` / ``Corrects:`` citations. A declared edge already
        present (matched STRUCTURALLY by its stored target's *current* slug, never
        re-resolved) is RETAINED untouched — re-resolving a retained kill-edge
        against the active view would fail, because the target is inactive *because
        of that very edge* (the self-strangling trap). Only a NET-NEW declared edge
        is resolved against the active view and guarded; a stored edge no longer
        declared is DELETEd (the declarative mirror — removing a ``Supersedes:``
        line resurrects the target, §4.5.1).

        The four guards fire only on net-new edges and report structured
        ``source="store"`` failures (the entry rolls back on any):

        - ``missing_target`` — no node carries the cited slug (a forward reference
          is always a causal violation; edges point new→old);
        - ``dangling_edge`` — a node carries it but is inactive (the immediate
          1-hop killer is named; no transitive walker — that is V1b);
        - ``cycle_violation`` — a self-edge (a node cannot supersede/correct
          itself), or a net-new kill-edge from an already-superseded source;
        - ``kind_constraint_violation`` — a cross-kind edge, caught from the DDL
          CHECK on the edge INSERT and mapped (Lesson 2 — never pre-asserted).

        The seven non-V1a relationship types are warn-deferred (logged WARNING,
        node + kill-edges still commit, NOT a failure).

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

        # --- Declared kill-edges (strip brackets, casefold for lookup) ----------
        # ``declared`` drives net-new resolution; ``declared_keys`` (the casefolded
        # target slug + edge_type) drives the declarative-mirror DELETE (V1-D21).
        declared: List[Tuple[str, str, str]] = []  # (edge_type, casefold, display)
        declared_keys: Set[Tuple[str, str]] = set()
        for edge_type in _KILL_EDGE_FIELDS:
            raw = getattr(parsed, edge_type, None)
            if not raw:
                continue
            citation = _strip_citation(raw)
            if not citation:  # an empty / bracket-only value is not a declaration
                continue
            citation_casefold = citation.casefold()  # MI-7: never SQLite LOWER
            declared.append((edge_type, citation_casefold, citation))
            declared_keys.add((citation_casefold, edge_type))

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

        # --- Resolve + guard each declared kill-edge ----------------------------
        for edge_type, citation_casefold, citation in declared:
            if (citation_casefold, edge_type) in stored_by_key:
                # RETAINED — matched structurally, never re-resolved (the
                # self-strangle fix). An unchanged re-declaration is a pure no-op:
                # no re-resolution, no guard, no write (this is what satisfies MI-5
                # for a commentary-only re-commit).
                continue

            token = _KILL_EDGE_TOKENS[edge_type]

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
                if not source_active:
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
                                f"'{citation}' is a different kind of entry; "
                                f"{edge_type} edges must connect two decisions or "
                                "two open questions."
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
                # Deleting this edge may reactivate its target (resurrection,
                # §4.5.1); the target's current slug must be re-checked for a
                # collision by the caller (V1-D4).
                resurrected_slugs.add(slug_casefold)

        # --- Warn-defer the seven non-V1a relationship types --------------------
        # Loud (WARNING, non-deduping — matters at cutover replay), NOT a failure.
        for field in _DEFERRED_EDGE_FIELDS:
            if getattr(parsed, field, None):
                logger.warning(
                    "Entry '%s': '%s' edge deferred to V1b (not committed).",
                    parsed.slug,
                    field,
                )

        return edges_changed, failures, resurrected_slugs

    def _enqueue_outbox(
        self, cursor: sqlite3.Cursor, node_id: str, parsed: ParsedEntry
    ) -> None:
        """Phase 5c seam: ``pending_embeddings`` Outbox enqueue. No-op in 5a.

        5c fills this body: UPSERT a committing-node row into ``pending_embeddings``
        (a conservative flagless "node-changed" signal; a re-enqueue resets
        ``queued_at`` / ``retry_count``) inside this transaction. It takes the open
        ``cursor`` so 5c enlists in the commit's transaction without re-plumbing it.
        In 5a the Outbox stays empty.

        Args:
            cursor: The open cursor inside ``commit_parsed_entry``'s transaction.
            node_id: The committing node's content-hash id.
            parsed: The committing ``ParsedEntry``.
        """
        return None

    def get_active_decisions(self, scope: Optional[str] = None) -> List[Dict[str, Any]]:
        """Exposes view of all active decisions (C3 / M3)."""
        conn = self._get_connection()
        try:
            states = self.compute_all_states(conn)
            active_ids = [nid for nid, state in states.items() if state in ("active", "drifted")]
            
            if not active_ids:
                return []

            placeholders = ",".join("?" for _ in active_ids)
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM nodes WHERE id IN ({placeholders}) AND kind = 'decision'",
                active_ids
            )
            
            results = []
            for row in cursor.fetchall():
                node = dict(row)
                node["mechanisms"] = json.loads(node["mechanisms"] or "[]")
                node["scope"] = json.loads(node["scope"] or "[]")
                
                # Apply scope filter if present
                if scope:
                    if scope not in node["scope"]:
                        continue
                results.append(node)
                
            return results
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
            states = self.compute_all_states(conn)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM nodes WHERE kind = 'decision'")

            results = []
            for row in cursor.fetchall():
                node = dict(row)
                node["mechanisms"] = json.loads(node["mechanisms"] or "[]")
                node["scope"] = json.loads(node["scope"] or "[]")
                node["computed_state"] = states.get(node["id"], "active")

                if scope and scope not in node["scope"]:
                    continue
                if not state_matches(node["computed_state"], state):
                    continue
                results.append(node)

            return results
        finally:
            conn.close()

    def get_open_questions(self, scope: Optional[str] = None) -> List[Dict[str, Any]]:
        """Exposes open questions (resolved or parked)."""
        conn = self._get_connection()
        try:
            states = self.compute_all_states(conn)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM nodes WHERE kind = 'open_question'")
            
            results = []
            for row in cursor.fetchall():
                node = dict(row)
                node["questions_raised"] = json.loads(node["questions_raised"] or "[]")
                node["scope"] = json.loads(node["scope"] or "[]")
                node["computed_state"] = states.get(node["id"], "parked")

                if scope:
                    if scope not in node["scope"]:
                        continue
                results.append(node)
                
            return results
        finally:
            conn.close()

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Retrieves all nodes in the database with their computed states."""
        conn = self._get_connection()
        try:
            states = self.compute_all_states(conn)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM nodes")
            
            results = []
            for row in cursor.fetchall():
                node = dict(row)
                node["mechanisms"] = json.loads(node["mechanisms"] or "[]")
                node["scope"] = json.loads(node["scope"] or "[]")
                node["questions_raised"] = json.loads(node["questions_raised"] or "[]")
                node["computed_state"] = states.get(node["id"], "active")
                results.append(node)
                
            return results
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
            cursor = conn.cursor()
            id_placeholders = ",".join("?" for _ in node_ids)
            type_placeholders = ",".join("?" for _ in MODIFIER_EDGE_KEYS)
            cursor.execute(
                f"""
                SELECT e.to_id AS to_id, e.type AS type, n.slug AS slug
                FROM edges e
                JOIN nodes n ON n.id = e.from_id
                WHERE e.to_id IN ({id_placeholders})
                  AND e.type IN ({type_placeholders})
                ORDER BY n.slug
                """,
                (*node_ids, *MODIFIER_EDGE_KEYS.keys()),
            )
            result: Dict[str, Dict[str, List[str]]] = {}
            for row in cursor.fetchall():
                key = MODIFIER_EDGE_KEYS[row["type"]]
                result.setdefault(row["to_id"], {}).setdefault(key, []).append(row["slug"])
            return result
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

    def add_pending_embedding(self, node_id: str, embedding_text: str) -> None:
        """Adds or updates a node to the pending embeddings queue (C2/F2)."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO pending_embeddings (node_id, embedding_text)
                    VALUES (?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET attempts = attempts + 1
                    """,
                    (node_id, embedding_text)
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
        """Increments the retry attempt count for a pending embedding and releases the claim."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE pending_embeddings
                    SET attempts = attempts + 1, claimed_by = NULL
                    WHERE node_id = ?
                    """,
                    (node_id,)
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to increment pending attempts: {str(e)}")
        finally:
            conn.close()

    def claim_pending_embeddings(self, drainer_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Claims a batch of pending embeddings atomically using UPDATE ... RETURNING (V1-D26)."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE pending_embeddings
                    SET claimed_by = ?
                    WHERE node_id IN (
                        SELECT node_id FROM pending_embeddings
                        WHERE claimed_by IS NULL
                        LIMIT ?
                    )
                    RETURNING node_id, embedding_text
                    """,
                    (drainer_id, limit)
                )
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to claim pending embeddings: {str(e)}")
        finally:
            conn.close()

    def release_pending_embeddings(self, drainer_id: str) -> None:
        """Releases claims held by a drainer, resetting them back to NULL."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    "UPDATE pending_embeddings SET claimed_by = NULL WHERE claimed_by = ?",
                    (drainer_id,)
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to release pending embeddings: {str(e)}")
        finally:
            conn.close()
