"""SQLite-backed graph store for Mitos.

This module implements the core architectural Identity & Graph Substrate (B),
computed states (C), edge-reconciliation (V1-D21), and cascade scopes (V1-D22).
"""

import sqlite3
import json
import os
import hashlib
from typing import List, Dict, Optional, Any, Set, Tuple
from mitos.errors import DatabaseError, ValidationError
from mitos.parser import ParsedEntry

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


class GraphStore:
    """SQLite-backed store managing nodes, edges, signals, and computed state."""

    def __init__(self, db_path: str, read_only: bool = False) -> None:
        self.db_path = db_path
        self.read_only = read_only
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Opens a connection to the SQLite database and configures it."""
        try:
            if self.read_only:
                abs_path = os.path.abspath(self.db_path)
                conn = sqlite3.connect(f"file:{abs_path}?mode=ro", uri=True)
            else:
                conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            if not self.read_only:
                # Enable WAL mode for concurrent multi-reader, single-writer
                conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            return conn
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to connect to SQLite: {str(e)}")

    def _init_db(self) -> None:
        """Initializes the database schema if it does not exist."""
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
        """Commits a single ParsedEntry atomic transaction (C1/V1-D22).

        Computes ID, performs validation, inserts/updates node, reconciles
        outgoing edges (declarative-mirror-edge-reconciliation), evaluates
        cascade consequences (commit-delta-cascade-transitive), and returns
        the structured CommitDelta.

        Args:
            parsed: The ParsedEntry representation.

        Returns:
            A structured CommitDelta payload.
        """
        # 0. Invariant Validation (C1 Seam / M5 invariant)
        if parsed.kind == "decision":
            if not parsed.core_axiom:
                raise ValidationError(f"Decision '{parsed.slug}' is missing required field '**Decided:**'")
            if not parsed.rejected_paths:
                raise ValidationError(f"Decision '{parsed.slug}' is missing required field '**Rejected:**' (P14 / M5 invariant)")
        elif parsed.kind == "open_question":
            if not parsed.questions_raised:
                raise ValidationError(f"Open question '{parsed.slug}' is missing required field '**Questions:**'")

        # 1. Compute node ID
        node_id = compute_hash(
            parsed.kind,
            parsed.slug,
            parsed.core_axiom,
            parsed.mechanisms,
            parsed.questions_raised
        )

        conn = self._get_connection()
        try:
            with conn:
                # Check for prior node with same slug/ID to capture delta metadata
                cursor = conn.cursor()
                cursor.execute("SELECT scope, id FROM nodes WHERE slug = ? COLLATE NOCASE", (parsed.slug,))
                prior_rows = cursor.fetchall()
                
                prior_id: Optional[str] = None
                self_old_scope: List[str] = []
                
                for row in prior_rows:
                    # If matches exact ID, it's a commentary update
                    if row["id"] == node_id:
                        prior_id = node_id
                        self_old_scope = json.loads(row["scope"] or "[]")
                        break
                
                # If no exact ID match but slug exists, we are replacing the node (e.g. correction/supersession)
                if not prior_id and prior_rows:
                    # Use scope of the most recent slug holder
                    self_old_scope = json.loads(prior_rows[-1]["scope"] or "[]")

                # Track global computed states BEFORE write
                states_before = self.compute_all_states(conn)

                # 2. Write Node Row
                serialized_questions = json.dumps(parsed.questions_raised)
                serialized_mechanisms = json.dumps(parsed.mechanisms)
                serialized_scope = json.dumps(parsed.scope)

                # Store or update the node
                cursor.execute(
                    """
                    INSERT INTO nodes (
                        id, slug, kind, date, title, core_axiom, rejected_paths,
                        invalidates_if, context, transcript, park_reason,
                        questions_raised, mechanisms, scope, confirmed_by, confirmed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        date=excluded.date,
                        title=excluded.title,
                        invalidates_if=excluded.invalidates_if,
                        context=excluded.context,
                        transcript=excluded.transcript,
                        park_reason=excluded.park_reason,
                        scope=excluded.scope,
                        confirmed_by=excluded.confirmed_by,
                        confirmed_at=excluded.confirmed_at
                    """,
                    (
                        node_id, parsed.slug, parsed.kind, parsed.date, parsed.title,
                        parsed.core_axiom, parsed.rejected_paths, parsed.invalidates_if,
                        parsed.context, parsed.transcript, parsed.park_reason,
                        serialized_questions, serialized_mechanisms, serialized_scope,
                        parsed.confirmed_by, parsed.confirmed_at
                    )
                )

                # Store mechanisms in the registry (M6)
                for mech in parsed.mechanisms:
                    if mech.strip():
                        cursor.execute(
                            "INSERT OR IGNORE INTO mechanisms (name) VALUES (?)",
                            (mech.strip(),)
                        )

                # 3. Declarative Outgoing Edge Reconciliation (V1-D21)
                # First delete existing outgoing edges originating from this node ID
                cursor.execute("DELETE FROM edges WHERE from_id = ?", (node_id,))

                # Insert declared relationship edges
                declared_relationships = [
                    ("supersedes", parsed.supersedes),
                    ("corrects", parsed.corrects),
                    ("amends", parsed.amends),
                    ("narrows", parsed.narrows),
                    ("depends_on", parsed.depends_on),
                    ("resolves", parsed.resolves),
                    ("contradicts", parsed.contradicts),
                    ("derives_from", parsed.derives_from),
                    ("cites", parsed.cites),
                ]

                for etype, target_slug in declared_relationships:
                    if target_slug:
                        if isinstance(target_slug, list):
                            target_slug = target_slug[0] if target_slug else None
                        if not target_slug:
                            continue
                        # Resolve slug to IDs (case-insensitive)
                        target_ids = self.resolve_slug(str(target_slug).strip())
                        if not target_ids:
                            # Log a warning or create a dangling reference as best-effort
                            # For v0.1: let's ignore or raise if we strictly want validation.
                            # We'll raise to keep graph honest, except for imported where S2 says warning
                            continue
                        
                        if len(target_ids) > 1:
                            raise ValidationError(
                                f"Ambiguous relationship target slug '{target_slug}': resolves to {target_ids}"
                            )
                        
                        # Insert edge
                        cursor.execute(
                            "INSERT OR IGNORE INTO edges (from_id, to_id, type) VALUES (?, ?, ?)",
                            (node_id, target_ids[0], etype)
                        )

                # 4. Evaluate Cascade Consequences (V1-D22)
                states_after = self.compute_all_states(conn)
                
                # Compare states to find flipped view memberships
                cascade_affected_scopes: Set[str] = set()
                
                # Fetch scopes for all nodes to map flipped states to scopes
                cursor.execute("SELECT id, scope FROM nodes")
                all_node_scopes = {row["id"]: json.loads(row["scope"] or "[]") for row in cursor.fetchall()}

                for nid, after_state in states_after.items():
                    before_state = states_before.get(nid)
                    if before_state != after_state:
                        # State membership flipped!
                        scopes = all_node_scopes.get(nid, [])
                        for s in scopes:
                            cascade_affected_scopes.add(s)

                commentary_changed = (prior_id == node_id)

                return CommitDelta(
                    node_id=node_id,
                    node_scope=parsed.scope,
                    self_old_scope=self_old_scope,
                    commentary_fields_changed=commentary_changed,
                    cascade_affected_scopes=list(cascade_affected_scopes)
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"SQLite transaction commit failed: {str(e)}")
        finally:
            conn.close()

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
