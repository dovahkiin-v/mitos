"""SQLite-backed graph store for Mitos.

This module implements the core architectural Identity & Graph Substrate (B),
computed states (C), edge-reconciliation (V1-D21), and cascade scopes (V1-D22).
"""

import sqlite3
import json
import os
import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import (
    List, Dict, Optional, Any, FrozenSet, Iterable, NamedTuple, Sequence, Set, Tuple,
)
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
from mitos.identity import SLUG_MAX_LEN, compute_node_id, mechanism_canonical_norm
from mitos.migrations import (
    EMBEDDING_SEED_ESTABLISHED_BY,
    MIGRATION_STEPS,
    REJECTION_SEED_ESTABLISHED_BY,
    MigrationStep,
    run_migrations,
    is_pre_v1a_schema,
    take_pre_ladder_snapshot,
    restore_from_snapshot,
)
from mitos.parser import ParsedEntry
from mitos.rejections import enumerate_rejected_paths
# The resolver's result type lives in the walk's leaf so the walk can type-check it
# without importing this module (the check family's fence); re-exported here.
from mitos.retrieval_walk import PointResolution
from mitos.scope_tags import normalize_scope_tags

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

# The six same-kind edge types, as the ``edges`` DDL CHECK's first clause lists them.
_SAME_KIND_EDGE_TYPES: frozenset = frozenset(
    {"supersedes", "corrects", "amends", "narrows", "depends_on", "contradicts"}
)


def edge_kind_is_legal(edge_type: str, source_kind: str, target_kind: str) -> bool:
    """Reports whether the kind matrix permits this edge — the DDL CHECK, in Python.

    A pure mirror of the four-clause ``CHECK`` on the ``edges`` table, and it exists
    ONLY so a caller can ask the question **before** attempting the INSERT. The DDL
    stays the structural gate; nothing here relaxes it. It lives beside
    ``_EDGE_KIND_REQUIREMENT`` (which phrases the same matrix for humans) so the two
    move together, and ``test_edge_kind_matrix_matches_ddl`` proves the equivalence
    over the entire ``edge_type × source_kind × target_kind`` grid by attempting a real
    INSERT for every cell — so this cannot drift from the schema undetected.

    Args:
        edge_type: One of the nine relationship types.
        source_kind: The citing node's kind ("decision" or "open_question").
        target_kind: The cited node's kind.

    Returns:
        True if the CHECK would admit the edge.
    """
    if edge_type == "cites":
        return True
    if edge_type in _SAME_KIND_EDGE_TYPES:
        return source_kind == target_kind
    if edge_type == "derives_from":
        return source_kind == "open_question" and target_kind == "decision"
    if edge_type == "resolves":
        return source_kind == "decision" and target_kind == "open_question"
    return False

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

# The MUTATION-UNION relationship fields (new→old) the lineage walk traverses and
# the write-time cycle guard protects: ``supersedes`` (a kill-edge) PLUS the two
# non-kill mutation edges ``amends`` / ``narrows``. This is a THIRD, distinct set —
# it cherry-picks ACROSS ``_KILL_EDGE_FIELDS`` and ``_DEFERRED_EDGE_FIELDS`` and is
# derivable from NEITHER, so do NOT reuse ``_KILL_EDGE_TYPES_SQL``. ``corrects`` (a
# kill-edge) is deliberately EXCLUDED: it is kept acyclic by the source-active
# kill-edge guard, and a typo-fix is not semantic evolution — it belongs to the
# deferred ``get_transcript`` ancestry, not the mutation lineage (M1;
# ``corrects-excluded-from-mutation-cycle-probe``). The other five non-kill types
# (``depends_on`` / ``resolves`` / ``contradicts`` / ``derives_from`` / ``cites``)
# are not lineage edges and never participate in a mutation cycle.
_MUTATION_EDGE_FIELDS: Tuple[str, ...] = ("supersedes", "amends", "narrows")

# The paired SQL ``IN (...)`` literal single-sourcing the mutation-union
# ``edge_type`` filter the lineage walk binds — mirrors ``_KILL_EDGE_TYPES_SQL``'s
# form (a code-internal constant interpolated into the IN clause; no user value is
# interpolated, so it is P8-safe). Kept in lockstep with ``_MUTATION_EDGE_FIELDS``.
_MUTATION_EDGE_TYPES_SQL: str = "('supersedes', 'amends', 'narrows')"

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

# OQ Stage-2 resolution (V1-D18). An open question is ``resolved`` iff it is the
# target of ≥1 incoming ``resolves`` edge whose SOURCE decision is STILL ACTIVE.
# Computed inside the main SELECT (one query, never a per-node lookup — P11), like
# ``_KILLER_TYPE_SQL`` above. This is the THIRD reader of the one liveness atom
# ``_KILL_EDGE_TYPES_SQL`` (after ``_ACTIVE_VIEW_PREDICATE`` and
# ``_modifiers_map``'s source-liveness gate — §6.2, Lesson 4): the inner
# ``NOT EXISTS`` gates the RESOLVER's liveness (``k.target_id = r.source_id``),
# never the OQ's own (the OQ's Stage-1 liveness is the outer
# ``_ACTIVE_VIEW_PREDICATE``). Single-sourced through ``_KILL_EDGE_TYPES_SQL`` +
# the literal ``'resolves'`` (the only ``resolves`` direction the widened CHECK
# accepts, D→OQ) — never a re-hand-rolled anti-join, never ``_ACTIVE_VIEW_PREDICATE``
# verbatim (it binds the UNALIASED ``nodes``; this subquery correlates on the
# aliased ``r.source_id``, so an f-string of it would bind to nothing — the §4.3
# alias trap). Interpolated, not bound (P8 carve-out, exactly as the kill-edge
# anti-joins) — the binding tuple is unchanged. Self-healing falls out for free
# (M3): kill the resolver and this EXISTS recomputes ``parked`` at the next read.
# The bare boolean form (no ``AS is_resolved`` alias) — droppable into a ``CASE`` /
# ``WHERE`` / ``NOT (...)``. Single-sourced here so the aggregate's parked filter
# (``_ACTIVE_VIEW_PREDICATE AND NOT _OQ_RESOLVED_PREDICATE``) and ``get_open_questions``'
# aliased SELECT expression share one definition (DRY / M8). Like
# ``_ACTIVE_VIEW_PREDICATE``, the outer EXISTS correlates on the UNALIASED ``nodes``
# (``r.target_id = nodes.id``) — bind it only in a SELECT whose FROM is the unaliased
# ``nodes`` (the §4.3 alias trap).
_OQ_RESOLVED_PREDICATE: str = (
    "EXISTS (SELECT 1 FROM edges r "
    "WHERE r.target_id = nodes.id AND r.edge_type = 'resolves' "
    "AND NOT EXISTS (SELECT 1 FROM edges k "
    "WHERE k.target_id = r.source_id "
    f"AND k.edge_type IN {_KILL_EDGE_TYPES_SQL}))"
)
_OQ_RESOLVED_SQL: str = f"{_OQ_RESOLVED_PREDICATE} AS is_resolved"

# The decision index's EXPECTED SET (recall-confidence 3b; ADR
# ``decision-index-gap-and-creation-carve-out-key-on-its-expected-set``): every
# decision node in every computed state — a decision's vector lives exactly as long
# as its node (ADR ``graveyard-is-deliberate-every-decision-vector-lives-as-long-as-its-node``)
# — plus the open questions that survive Stage 1. No ``_OQ_RESOLVED_PREDICATE``: a
# *resolved* open question stays in, as it is in ``get_active_node_ids``. So the active
# set is a subset of this one by construction. ONE atom, bound by both readers
# (``get_decision_index_expected_ids`` and ``decision_index_expects_node_other_than``),
# so the sensor (``status``), the heal (``reconcile``), the rebuild/cutover seed, the gap
# predicate and the creation carve-out cannot drift apart. "Decision index" in the name
# because the rejection index has an expected set of its own. Like
# ``_ACTIVE_VIEW_PREDICATE`` it correlates on the UNALIASED ``nodes`` (the §4.3 alias trap).
_DECISION_INDEX_EXPECTED_PREDICATE: str = (
    "(nodes.kind = 'decision' OR "
    f"(nodes.kind = 'open_question' AND {_ACTIVE_VIEW_PREDICATE}))"
)

# The rejection index's OWNERS (recall-confidence 4b): every decision node in every
# computed state, and nothing else — an open question holds no ``rejected_paths``, so
# it never owns an item. Bound by ``rejection_index_expects_owner_other_than`` (the
# inline creation licence: "the graph's only decision, live or retired"), by the
# expected-set read ``get_rejection_index_owner_fields`` (4d), by the gap predicate's
# read ``rejection_index_expects_items`` and by the import's empty-graph read
# ``holds_any_decision`` — one atom, so the licence and the sensor cannot drift. A
# superset of "decisions holding at least one item" only through a decision whose field enumerates nothing, which every write
# route refuses; there the licence errs safe (no create; the row waits for a
# ``reconcile``). Correlates on the UNALIASED ``nodes`` (the §4.3 alias trap).
_REJECTION_INDEX_OWNER_PREDICATE: str = "nodes.kind = 'decision'"

# Indexed scope-membership filter (P11: ``idx_node_scopes_scope`` serves
# ``scope = ?``, then the PK joins back to ``node_id``). Appended to a read SELECT
# and bound with a single ``?`` scope value — never an O(N) Python post-filter.
_SCOPE_FILTER_SQL: str = (
    " AND EXISTS (SELECT 1 FROM node_scopes "
    "WHERE node_scopes.node_id = nodes.id AND node_scopes.scope = ?)"
)

# A node's scope order: authored position first, then the tag as tiebreak. The
# tiebreak makes every read deterministic where two rows share an ordinal (only an
# older mitos inserting at the column's DEFAULT 0 can create that) and reproduces the
# pre-ordinal alphabetical order when a node's ordinals are all 0. One fragment, bound
# by both the hydrating read and the commit path's prior read, so the two orders
# cannot drift.
_SCOPE_ORDER_SQL: str = "ordinal, scope"

# SQLite's message when a read names ``ordinal`` on a graph that has not run ladder
# step 4 (identical on a ``mode=ro`` connection). Matched exactly so a locked,
# corrupt or otherwise-broken read still raises.
_MISSING_ORDINAL_ERROR: str = "no such column: ordinal"
# A query that names the column qualified (``ns.ordinal``) gets a message carrying the
# qualifier, which this does not match — so fallback-guarded reads name it unqualified.


def _scope_discrimination_sql(scope_order: str) -> str:
    """Builds the authored-first / co-tag aggregate behind ``get_scope_discrimination``.

    ``scope_order`` is the per-node tag order that decides "first": ``_SCOPE_ORDER_SQL``,
    or ``"scope"`` on a graph without the ordinal column. It is interpolated
    unqualified into a window over the single ``node_scopes`` read, so a missing
    column raises exactly ``_MISSING_ORDINAL_ERROR``. Ranking reads every row of a
    node before the liveness join drops any node, and ``nodes`` stays unaliased so
    ``_ACTIVE_VIEW_PREDICATE`` binds (the §4.3 alias trap).
    """
    return (
        "WITH ranked AS ("
        "SELECT node_id, scope, "
        f"ROW_NUMBER() OVER (PARTITION BY node_id ORDER BY {scope_order}) AS position "
        "FROM node_scopes), "
        "live AS ("
        "SELECT nodes.id AS node_id FROM nodes "
        f"WHERE nodes.kind = 'decision' AND {_ACTIVE_VIEW_PREDICATE}), "
        "tags AS ("
        "SELECT ranked.node_id, ranked.scope, ranked.position "
        "FROM ranked JOIN live ON live.node_id = ranked.node_id), "
        "firsts AS ("
        "SELECT scope, COUNT(*) AS n FROM tags WHERE position = 1 GROUP BY scope), "
        "partners AS ("
        "SELECT a.scope, COUNT(DISTINCT b.scope) AS n FROM tags a "
        "JOIN tags b ON b.node_id = a.node_id AND b.scope <> a.scope "
        "GROUP BY a.scope) "
        "SELECT t.scope, "
        "COALESCE(firsts.n, 0) AS authored_first_decisions, "
        "COALESCE(partners.n, 0) AS co_tagged_scopes "
        "FROM (SELECT DISTINCT scope FROM tags) t "
        "LEFT JOIN firsts ON firsts.scope = t.scope "
        "LEFT JOIN partners ON partners.scope = t.scope "
        "ORDER BY t.scope"
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


def _computed_oq_state(is_resolved: int) -> str:
    """Derives an open question's Stage-2 state from its resolver liveness (M3).

    The OQ counterpart to :func:`_computed_decision_state`. Stage-2 state is
    computed, never stored: an open question with ≥1 incoming ``resolves`` edge
    from a still-active decision is ``"resolved"``; otherwise it is ``"parked"``.
    (Stage-1 active-view membership — the kill-edge anti-join — is a separate axis
    applied as the outer filter; a kill-edged OQ never reaches this helper.)

    Args:
        is_resolved: The truthy ``is_resolved`` derived column (from
            ``_OQ_RESOLVED_SQL``) — 1 when an active resolver exists, else 0.

    Returns:
        The computed Stage-2 state string (``"resolved"`` / ``"parked"``).
    """
    return "resolved" if is_resolved else "parked"


def _computed_node_state(
    kind: str, killer_type: Optional[str], is_drifted: bool, is_resolved: int
) -> str:
    """Derives one node's state in the vocabulary of its own kind (M3).

    The one kind-aware ladder, shared by the receipt's identification echo
    (:meth:`GraphStore.get_outgoing_edge_targets`) and the by-handle reads
    (:meth:`GraphStore.get_handle_state`). The kill check comes first for both
    kinds, because Stage 1 precedes Stage 2: a killed node reads ``superseded`` /
    ``corrected`` whatever its kind. An unkilled open question then reads its
    Stage-2 ``resolved`` / ``parked`` and takes no drift, as
    :meth:`GraphStore.get_open_questions` does. An unkilled decision reads
    ``drifted`` when a drifted signal is set, else ``active``.

    Args:
        kind: The node's ``kind`` (``"decision"`` or ``"open_question"``).
        killer_type: The incoming kill-edge type, or None when there is none.
        is_drifted: Whether a drifted signal is set on the node.
        is_resolved: The ``is_resolved`` derived column; read for an open
            question only.

    Returns:
        The computed state string.
    """
    if killer_type is not None:
        return _computed_decision_state(killer_type)
    if kind == "open_question":
        return _computed_oq_state(is_resolved)
    if is_drifted:
        return "drifted"
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


# Ids bound per statement in ``GraphStore.created_at_for``: far below SQLite's
# bound-variable limit on every build, including the older 999-variable default.
_CREATED_AT_CHUNK = 500

# Ids bound per statement in ``GraphStore.resolve_point_ids`` (one bind per
# point-id prefix; the range's upper bound is derived in SQL) and in
# ``GraphStore.get_node_states`` (one bind per exact id). The same 500 keeps both
# under the 999-variable floor; ``resolve_point_ids`` runs ``_scopes_for`` once per
# chunk because it does not chunk its own ``IN``.
_POINT_ID_CHUNK = 500

# A normalized point id: what remains of ``hash_to_uuid(node_id)`` once its dashes
# are removed and it is lowercased — the first 32 hex characters of a node id.
_POINT_ID_PREFIX_RE = re.compile(r"[0-9a-f]{32}")


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


# The one spelling of the ``embedding_seed`` upsert. Two callers need it on two
# different connections — ``GraphStore.stamp_embedding_seed`` on its own, and
# ``cutover._prune_embedding_queue_to_expected`` inside the transaction that makes the
# claim true — so the SQL is single-sourced here rather than retyped at the second
# site. The same in-transaction-twin shape ``_enqueue_outbox`` / ``add_pending_embedding``
# already established for the outbox. ``ON CONFLICT(id) DO UPDATE`` so a re-stamp
# replaces the row rather than raising (the marker records the LATEST covering act).
_EMBEDDING_SEED_UPSERT_SQL: str = """
    INSERT INTO embedding_seed (id, established_by, established_at)
    VALUES (1, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
        established_by = excluded.established_by,
        established_at = excluded.established_at
"""


def write_embedding_seed(
    conn: sqlite3.Connection, established_by: str, *, now: Optional[str] = None
) -> None:
    """Stamps the coverage marker on ``conn``, without committing (MI-10 timestamp).

    The in-transaction twin of :meth:`GraphStore.stamp_embedding_seed`: it issues the
    statement and leaves the transaction to the caller, so the act that *makes* the
    outbox cover the decision index's expected set
    (:meth:`GraphStore.get_decision_index_expected_ids`) and the record *claiming* it
    can land together. The
    ``established_by`` vocabulary is enforced by the table's own ``CHECK`` (see
    ``migrations.EMBEDDING_SEED_ESTABLISHED_BY``) — an unknown verb is refused at
    write time, not filtered here.

    Args:
        conn: An open, writable connection whose transaction the caller owns.
        established_by: The verb that established coverage — one of
            ``migrations.EMBEDDING_SEED_ESTABLISHED_BY``.
        now: The UTC ISO-8601 stamp to record; defaults to :func:`_utc_now_iso`
            (MI-10 — application-supplied, never ``CURRENT_TIMESTAMP``).
    """
    conn.execute(
        _EMBEDDING_SEED_UPSERT_SQL, (established_by, now or _utc_now_iso())
    )


# The one spelling of the ``rejection_seed`` upsert, shared by
# ``GraphStore.stamp_rejection_seed`` and :func:`write_rejection_seed` exactly as the
# decision marker's upsert is shared above. The two markers live in separate tables
# (ladder step 5), so neither stamp can touch the other's row.
_REJECTION_SEED_UPSERT_SQL: str = """
    INSERT INTO rejection_seed (id, established_by, established_at)
    VALUES (1, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
        established_by = excluded.established_by,
        established_at = excluded.established_at
"""


def write_rejection_seed(
    conn: sqlite3.Connection, established_by: str, *, now: Optional[str] = None
) -> None:
    """Stamps the rejection index's seed marker on ``conn``, without committing.

    The in-transaction twin of :meth:`GraphStore.stamp_rejection_seed`, mirroring
    :func:`write_embedding_seed`: the statement is issued and the transaction is left
    to the caller. Its caller is ``cutover._prune_embedding_queue_to_expected``
    (``rebuild``/``cutover``, in the aside transaction). The ``established_by`` vocabulary is enforced by the table's own
    ``CHECK`` (``migrations.REJECTION_SEED_ESTABLISHED_BY``).

    Args:
        conn: An open, writable connection whose transaction the caller owns.
        established_by: The verb that established coverage — one of
            ``migrations.REJECTION_SEED_ESTABLISHED_BY``.
        now: The UTC ISO-8601 stamp to record; defaults to :func:`_utc_now_iso`
            (MI-10).
    """
    conn.execute(
        _REJECTION_SEED_UPSERT_SQL, (established_by, now or _utc_now_iso())
    )


# SQLite's message when a read names ``rejection_backlog`` on a graph that has not run
# ladder step 5 (identical on a ``mode=ro`` connection). The backlog reads match it as
# a substring, like ``_MISSING_ORDINAL_ERROR``, so any other fault still raises.
_MISSING_REJECTION_BACKLOG_ERROR: str = "no such table: rejection_backlog"


class RejectionBacklogRow(NamedTuple):
    """One ``rejection_backlog`` row, as the store's reads return it (runtime only).

    Attributes:
        node_id: The owning decision's content-hash id.
        queued_at: The enqueueing commit's UTC ISO-8601 stamp.
        enqueue_seq: The row's compare token. A clear or a failure note lands
            only on the row still carrying it; no later enqueue repeats it.
        retry_count: Failed write attempts since the row was enqueued.
    """

    node_id: str
    queued_at: str
    enqueue_seq: int
    retry_count: int


# Failed attempts at which a rejection backlog row reads as poison. A count, not a
# threshold: the drain reports rows at or past it and keeps attempting them (a row is
# never dropped), and ``status`` reads the same name. ``check`` never reads this backlog.
REJECTION_BACKLOG_FAILURE_TOLERANCE = 3


def _rejection_backlog_row(row: sqlite3.Row) -> RejectionBacklogRow:
    """Builds a :class:`RejectionBacklogRow` from a backlog ``SELECT`` row."""
    return RejectionBacklogRow(
        node_id=row["node_id"],
        queued_at=row["queued_at"],
        enqueue_seq=int(row["enqueue_seq"]),
        retry_count=int(row["retry_count"]),
    )


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


class KillChain(NamedTuple):
    """One seed's forward walk over kill edges (``_walk_kill_chains``).

    Kill edges point new→old (the killer is the edge's source), so the walk goes
    from a retired node to whatever retired it, then to whatever retired that, until
    it reaches nodes nothing has killed: the live ends, where the current word on the
    seed's claim lives. Drifted counts as live — drift annotates, it does not kill.

    Attributes:
        killers: The seed's immediate killers (sources of the kill edges onto it),
            sorted by id.
        reached: Every node the walk reached, the seed excluded, first-seen
            depth-first order.
        live_ends: The ids among ``reached`` that nothing has killed, sorted by
            id; empty when the walk reached none (a closed cycle).
        anomaly: None, ``"fan_out"`` (some reached node, the seed included, has two
            or more killers) or ``"cycle"`` (the walk came back to a node still on
            its path; reported over ``"fan_out"`` when both hold). Only an
            out-of-band edit can author either.
    """

    killers: List[str]
    reached: List[str]
    live_ends: List[str]
    anomaly: Optional[str]


class CommitDelta:
    """Represents the structured delta returned after an entry commit.

    Carries metadata about the committed node and any cascading scope changes
    required for incremental rerendering (V1-D22), plus what the commit decided
    for the rejection index: whether the node is new, and the ``enqueue_seq`` of
    the ``rejection_backlog`` row this commit minted. The seq is handed out of the
    commit's own transaction because it is the only token that pairs with the
    field the commit stored; a row read after the commit could return a newer seq
    a concurrent re-queue minted, and clearing that would lose its write.
    """

    def __init__(
        self,
        node_id: str,
        node_scope: List[str],
        self_old_scope: List[str],
        commentary_fields_changed: bool,
        cascade_affected_scopes: List[str],
        created: bool = False,
        rejection_enqueue_seq: Optional[int] = None,
    ) -> None:
        self.node_id = node_id
        self.node_scope = node_scope
        self.self_old_scope = self_old_scope
        self.commentary_fields_changed = commentary_fields_changed
        self.cascade_affected_scopes = cascade_affected_scopes
        # True when the commit inserted the node (``commit_parsed_entry``'s
        # ``is_new``); False for any re-commit of an existing id.
        self.created = created
        # The ``enqueue_seq`` this commit's rejection-backlog enqueue minted, or
        # None when the commit queued no rejection row (an open question, an edit
        # leaving ``rejected_paths`` unchanged, a new owner with no item).
        self.rejection_enqueue_seq = rejection_enqueue_seq

    def to_dict(self) -> Dict[str, Any]:
        """Converts the CommitDelta into a serializable dictionary."""
        return {
            "node_id": self.node_id,
            "node_scope": self.node_scope,
            "self_old_scope": self.self_old_scope,
            "commentary_fields_changed": self.commentary_fields_changed,
            "cascade_affected_scopes": self.cascade_affected_scopes,
            "created": self.created,
            "rejection_enqueue_seq": self.rejection_enqueue_seq,
        }


def _point_id_prefix(point_id: Any) -> Optional[str]:
    """Normalizes a point id to the node-id prefix it names, or None if it names none.

    The inverse of ``vector_store.hash_to_uuid`` up to its truncation: dashes are
    removed and the rest lowercased, and the result must be exactly 32 hex
    characters. Anything else cannot be the prefix of a node id.

    Args:
        point_id: A point id as the vector index returned it.

    Returns:
        The 32-character lowercase hex prefix, or None.
    """
    if not isinstance(point_id, str):
        return None
    prefix = point_id.replace("-", "").lower()
    if _POINT_ID_PREFIX_RE.fullmatch(prefix) is None:
        return None
    return prefix


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
        # undiagnosable hybrid (R3/R11) — route it to the §2.1 one-time cutover,
        # naming the meanwhile-fallback so a read-only caller isn't dead-ended.
        if is_pre_v1a_schema(conn):
            raise DatabaseError(
                "This graph predates the V1a schema (a prototype layout "
                "was detected). Mitos will not migrate it in place. Run "
                "the one-time cutover to rebuild it into the V1a store. "
                "Meanwhile `mitos surface`/`query` fall back to a text match "
                "over the markdown corpus, and grep over `decisions.md` and "
                "`decisions/archive/` always works — nothing is lost."
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

    def resolve_slug_kinds(self, slug: str) -> List[str]:
        """Resolves a slug to the DISTINCT kinds of the nodes carrying it.

        ``resolve_slug``'s sibling, for callers that need to reason about the kind
        matrix rather than identity — chiefly ``sync``'s reconcile pre-flight, which
        must know whether an edge could ever satisfy the ``edges`` CHECK. Returns kinds
        rather than rows because a slug can name a whole same-slug supersession lineage
        (MI-13), and a pre-flight may only refuse when EVERY candidate is illegal.

        Args:
            slug: The slug to find (case-insensitive via ``str.casefold()``).

        Returns:
            The distinct kinds among matching nodes (empty if the slug resolves to
            nothing — that is ``missing_target``'s case, not this one's).
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT DISTINCT kind FROM nodes WHERE slug_casefold = ?",
                (slug.casefold(),),
            ).fetchall()
            return [row["kind"] for row in rows]
        finally:
            conn.close()

    def resolve_slug(self, slug: str) -> List[str]:
        """Resolves a slug to node IDs via single-tier casefold-exact match (V1-D23).

        Binds Python ``str.casefold()`` against the indexed ``slug_casefold`` column —
        the same Unicode-correct discipline the authoritative commit path
        (``_reconcile_edges``) and ``get_node_by_slug`` already use (MI-9: never SQLite
        ASCII-only ``COLLATE NOCASE`` / ``LOWER``). There is **no** fuzzy alias-fallback
        tier: V1a mandates single-tier exact match. A renamed-away slug is not silently
        repaired — a stale citation breaks loud as ``missing_target`` on the next sync.
        Returns every node sharing the casefolded slug — a same-slug supersession lineage
        yields >1 (MI-13); callers own active-view scoping and ambiguity handling.

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

    def resolve_handle(self, ident: str) -> Optional[Dict[str, Any]]:
        """Resolves an identifier to a node state-agnostically (the ``show`` seam).

        The single dereference primitive shared by ``mitos show`` (5a) and the MCP
        ``show_node`` twin (5b): given a content-hash id or a slug, return the
        reader-facing node dict for *any* computed state (active or superseded), so a
        node that has been moved on from still answers to its own slug instead of
        404-ing. Both surfaces call this one method, making their resolution parity
        structural — they cannot drift.

        **Precedence (exact — 5b reproduces it byte-for-byte):**

        1. **Exact content-hash id** — ``get_node(ident)`` (state-agnostic id read).
           Preserves ``show <id>`` for any node.
        2. **Active-view slug** — ``get_node_by_slug(ident)`` (≤1 per casefolded slug,
           MI-13). A reused slug always resolves to the *live* node.
        3. **Most-recent node in the casefolded-slug lineage** — reached only when no
           active node bears the slug, so it returns a superseded / corrected node.
           Recency = ``created_at`` DESC, ``id`` ASC (the deterministic tie-break;
           ``created_at`` is the app-supplied UTC-µs commit stamp, MI-10, and ISO-8601
           sorts lexicographically == chronologically). The hydrated dict carries its
           stamped modifier keys, so the caller renders it marked-superseded.
        4. Else → ``None`` (genuine absence — a typo or an authored-but-unsynced draft).

        The ``ValidationError`` that ``get_node_by_slug`` may raise on an MI-13 breach
        is **not** caught here: a real breach is not "not found" — it propagates.

        Args:
            ident: A content-hash id or a slug (case-insensitive via ``str.casefold()``).

        Returns:
            The reader-facing node dict (any state, modifier-stamped), or ``None`` for
            genuine absence.
        """
        # 1. Exact content-hash id (state-agnostic).
        node = self.get_node(ident)
        if node:
            return node
        # 2. Active-view slug — a reused slug resolves to the live node (lets a
        #    ValidationError on an MI-13 breach propagate, never swallowed).
        node = self.get_node_by_slug(ident)
        if node:
            return node
        # 3. Most-recent node in the casefolded-slug lineage (only reached when no
        #    active node bears the slug → a superseded / corrected node). Order in SQL
        #    over the live table; the hydrated dict may not carry created_at.
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT id FROM nodes WHERE slug_casefold = ? "
                "ORDER BY created_at DESC, id ASC LIMIT 1",
                (ident.casefold(),),
            ).fetchone()
        finally:
            conn.close()
        if row:
            return self.get_node(row["id"])
        # 4. Genuine absence.
        return None

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
        # OQ Stage-2 derived column (get_open_questions); a raw 0/1 must never leak
        # under a reader key. Popped above the kind branch so both kinds stay clean.
        node.pop("is_resolved", None)
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

    def created_at_for(self, node_ids: Sequence[str]) -> Dict[str, str]:
        """Reads the stored ``created_at`` of many nodes without hydrating them.

        One indexed ``IN`` query per chunk — no scopes, no modifier stamps — so a
        caller stamping a large batch costs no per-node read. Rotation no longer names
        archives from it (the name follows the rotation instant, ADR
        ``rotation-names-the-archive-for-the-rotation-instant-not-created-at``); the
        read stays as a batch primitive.

        Args:
            node_ids: The node ids to look up.

        Returns:
            A mapping of node id → its ``created_at`` string. An id with no node is
            simply absent from the map.

        Raises:
            DatabaseError: If the read fails.
        """
        ids = list(node_ids)
        if not ids:
            return {}
        conn = self._get_connection()
        try:
            out: Dict[str, str] = {}
            for start in range(0, len(ids), _CREATED_AT_CHUNK):
                chunk = ids[start:start + _CREATED_AT_CHUNK]
                placeholders = ",".join("?" for _ in chunk)
                for row in conn.execute(
                    f"SELECT id, created_at FROM nodes WHERE id IN ({placeholders})",
                    chunk,
                ):
                    out[row["id"]] = row["created_at"]
            return out
        except sqlite3.Error as e:
            raise DatabaseError(
                f"Failed to read created_at for {len(ids)} node(s): {str(e)}"
            )
        finally:
            conn.close()

    def resolve_point_ids(self, point_ids: Iterable[str]) -> PointResolution:
        """Maps vector-index point ids to their nodes in one batched, state-agnostic read.

        A point id is ``hash_to_uuid(node_id)``: the first 128 bits of the node's
        content-hash id, so a hit names its node without its payload slug, which is
        a handle (mutable, MI-2; shared across a same-slug supersession, MI-13).
        Each input is normalized by removing dashes and lowercasing and must then be
        32 hex characters; anything else is an orphan. Inputs are deduplicated by
        the raw string, each distinct prefix is read once, and every distinct raw
        string gets its own bucket entry.

        Per chunk this costs one ``FROM nodes`` SELECT — an indexed range join, a
        node matching prefix ``p`` when ``p <= id < p || 'g'`` (ids are lowercase
        hex and ``'g'`` sorts above every hex digit) — plus at most one bulk scope
        fetch. The column is never lowercased, which would forfeit the index; the
        CTE is aliased and ``nodes`` is not, because the derived-state fragments
        correlate on the bare ``nodes`` (the alias trap), and no CTE column is
        selected, because :meth:`_hydrate_node` would carry it onto the node.

        ``computed_state`` mirrors :meth:`get_node_state` exactly (decision
        vocabulary, ``drifted`` for an unkilled node with a drifted signal). An
        orphan is found by the join returning nothing, never by asking a state:
        :meth:`get_node_state` answers ``"active"`` for an id the graph does not
        hold. No modifier stamps are applied — they are a separate batched read
        (:meth:`get_modifiers_map`), so a stamp fault costs only the stamps.

        Args:
            point_ids: Point ids exactly as ``VectorStore.query`` returned them.

        Returns:
            The :class:`PointResolution` partition of the input. Empty input runs
            no statement.

        Raises:
            DatabaseError: If the read fails.
        """
        raw = list(dict.fromkeys(point_ids))
        prefix_of = {pid: _point_id_prefix(pid) for pid in raw}
        prefixes = list(dict.fromkeys(p for p in prefix_of.values() if p is not None))
        matched: Dict[str, List[Dict[str, Any]]] = {}
        if prefixes:
            conn = self._get_connection()
            try:
                for start in range(0, len(prefixes), _POINT_ID_CHUNK):
                    chunk = prefixes[start:start + _POINT_ID_CHUNK]
                    values = ",".join("(?)" for _ in chunk)
                    rows = conn.execute(
                        f"WITH p(lo) AS (VALUES {values}) "
                        f"SELECT nodes.*, {_IS_DRIFTED_SQL}, {_KILLER_TYPE_SQL} "
                        "FROM p JOIN nodes "
                        "ON nodes.id >= p.lo AND nodes.id < p.lo || 'g'",
                        chunk,
                    ).fetchall()
                    scopes_map = self._scopes_for(conn, [row["id"] for row in rows])
                    for row in rows:
                        # Read the state before hydrating: _hydrate_node pops
                        # killer_type and recasts is_drifted.
                        if row["killer_type"] is None and row["is_drifted"]:
                            state = "drifted"
                        else:
                            state = _computed_decision_state(row["killer_type"])
                        node = self._hydrate_node(row, scopes_map.get(row["id"], []))
                        node["computed_state"] = state
                        matched.setdefault(row["id"][:32], []).append(node)
            except sqlite3.Error as e:
                raise DatabaseError(
                    f"Failed to resolve {len(prefixes)} point id(s): {str(e)}"
                )
            finally:
                conn.close()

        nodes: Dict[str, Dict[str, Any]] = {}
        orphans: List[str] = []
        ambiguous: Dict[str, List[str]] = {}
        for pid in raw:
            prefix = prefix_of[pid]
            found = matched.get(prefix, []) if prefix is not None else []
            if not found:
                orphans.append(pid)
            elif len(found) > 1:
                ambiguous[pid] = sorted(node["id"] for node in found)
            else:
                nodes[pid] = dict(found[0])
        return PointResolution(nodes=nodes, orphans=orphans, ambiguous=ambiguous)

    def get_node_states(self, node_ids: Sequence[str]) -> Dict[str, str]:
        """Maps each content-hash id the graph holds to its computed state, in one batched read.

        The exact-id sibling of :meth:`resolve_point_ids`, for a caller that already
        holds whole node ids (the lexical fallback hashes each corpus block the way
        the commit does). State is :meth:`get_node_state`'s decision vocabulary
        (``active`` / ``drifted`` / ``superseded`` / ``corrected``), derived from
        the same ``_IS_DRIFTED_SQL`` / ``_KILLER_TYPE_SQL`` fragments and the same
        drifted rule, so liveness stays one SQL atom. An id the graph does not hold
        is **absent** from the map: existence comes from the ``IN`` read, never from
        asking a state, because :meth:`get_node_state` answers ``"active"`` for an
        unknown id. Nothing is hydrated and no scopes are fetched. ``nodes`` is not
        aliased, because the derived-state fragments correlate on the bare name.

        Args:
            node_ids: Content-hash node ids. Deduplicated here; order is not kept.

        Returns:
            Node id → computed state, for the ids the graph holds. Empty input runs
            no statement and opens no connection.

        Raises:
            DatabaseError: If the read fails.
        """
        ids = list(dict.fromkeys(node_ids))
        if not ids:
            return {}
        conn = self._get_connection()
        try:
            out: Dict[str, str] = {}
            # Read at call time, so a test can shrink the chunk.
            chunk_size = _POINT_ID_CHUNK
            for start in range(0, len(ids), chunk_size):
                chunk = ids[start:start + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                for row in conn.execute(
                    f"SELECT nodes.id, {_IS_DRIFTED_SQL}, {_KILLER_TYPE_SQL} "
                    f"FROM nodes WHERE nodes.id IN ({placeholders})",
                    chunk,
                ):
                    if row["killer_type"] is None and row["is_drifted"]:
                        out[row["id"]] = "drifted"
                    else:
                        out[row["id"]] = _computed_decision_state(row["killer_type"])
            return out
        except sqlite3.Error as e:
            raise DatabaseError(
                f"Failed to read the state of {len(ids)} node(s): {str(e)}"
            )
        finally:
            conn.close()

    def _scopes_for(
        self, conn: sqlite3.Connection, node_ids: List[str]
    ) -> Dict[str, List[str]]:
        """Bulk-fetches scope tags for many nodes in ONE indexed query (never N+1).

        A graph that has not run ladder step 4 has no ``ordinal`` column. A writable
        store migrates at construction, but a read-only one never runs the ladder, so
        on that one error this re-runs the pre-ordinal query ordered by tag — which is
        exactly the order such a graph holds. Any other SQLite error propagates.

        Args:
            conn: An open connection (the caller owns its lifecycle).
            node_ids: The node IDs to fetch scopes for.

        Returns:
            A mapping of node_id -> scope-tag list in authored order (ties, which only
            an older writer or the step-4 backfill leaves, fall back to alphabetical);
            an id with no scopes is simply absent from the map.
        """
        if not node_ids:
            return {}
        placeholders = ",".join("?" for _ in node_ids)
        select = (
            f"SELECT node_id, scope FROM node_scopes WHERE node_id IN ({placeholders})"
        )
        try:
            rows = conn.execute(
                f"{select} ORDER BY node_id, {_SCOPE_ORDER_SQL}", node_ids
            ).fetchall()
        except sqlite3.OperationalError as e:
            if _MISSING_ORDINAL_ERROR not in str(e):
                raise
            rows = conn.execute(
                f"{select} ORDER BY node_id, scope", node_ids
            ).fetchall()
        out: Dict[str, List[str]] = {}
        for row in rows:
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
        ``corrected`` / ``drifted``), and an open_question resolves to its
        kill-edge state here, never ``parked`` / ``resolved``. Its callers — the
        ranked loops, the list text, the conflict gather, the ``exists`` receipts —
        expect decision vocabulary. An open question's Stage-2 state is read by
        :meth:`get_open_questions` on the list surfaces and by
        :meth:`get_handle_state` on the by-handle reads.

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

    def get_handle_state(self, node_id: str) -> str:
        """Computes one node's state in its own kind's vocabulary, for a by-handle read.

        The state behind ``show_node`` and ``mitos show``, which dereference a handle
        that may name an open question. A decision reads what :meth:`get_node_state`
        gives. An open question reads its kill state if it has an incoming kill
        edge, else ``resolved`` / ``parked`` — the same state the receipt's
        identification echo reports, through the same :func:`_computed_node_state`.

        :meth:`get_node_state` stays decision-centric because every other caller
        expects decision vocabulary; this read has exactly the two by-handle callers.
        One SELECT serves both kinds: ``_OQ_RESOLVED_SQL`` yields 0 on a decision
        row. ``FROM nodes`` stays unaliased, because all three fragments correlate on
        the bare ``nodes`` (the §4.3 alias trap). An absent node defaults to
        ``"active"``, as :meth:`get_node_state` does, for the same defensive reason.

        Args:
            node_id: The content-hash id of the node to derive state for.

        Returns:
            The computed state string.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                f"SELECT nodes.kind, {_IS_DRIFTED_SQL}, {_KILLER_TYPE_SQL}, "
                f"{_OQ_RESOLVED_SQL} FROM nodes WHERE nodes.id = ?",
                (node_id,),
            ).fetchone()
            if row is None:
                return "active"
            return _computed_node_state(
                row["kind"],
                row["killer_type"],
                bool(row["is_drifted"]),
                row["is_resolved"],
            )
        finally:
            conn.close()

    def write_signal(self, node_id: str, stype: str, source: Optional[str] = None) -> None:
        """Writes a signal row (drifted, source_reencounter, retired) via INSERT OR IGNORE.

        Aligned to the V1a ``signals`` shape (Phase 8a): ``type``→``signal_type``,
        and the column the signal carries is ``source`` (NOT NULL DEFAULT '' — a
        ``None`` ``source`` maps to '') — uniform with ``nodes.source`` /
        ``mechanisms.source`` / ``ParsedEntry.source`` (V1-D14: the prior ``actor``
        param was conceptual shadowing of the same enum under two names; renamed
        ``actor``→``source`` here in V1b, this being the first live caller). The
        application-supplied ``created_at`` UTC µs stamp is required (MI-10; the V1a
        schema carries no ``DEFAULT CURRENT_TIMESTAMP``). ``INSERT OR IGNORE`` keeps
        the composite-PK uniqueness from crashing on a repeat write.
        """
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO signals (node_id, signal_type, source, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (node_id, stype, source or "", _utc_now_iso())
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to write signal: {str(e)}")
        finally:
            conn.close()

    def note_source_reencounter(
        self, node_id: str, stored_source: str, new_source: str
    ) -> bool:
        """Emits one ``source_reencounter`` signal iff the re-encountering source differs.

        The substrate's first real signal writer (MI-4 provenance audit, V1-D14).
        Called at the four short-circuit gates that skip ``commit_parsed_entry`` for
        an already-stored node (sync, import, and both ``record`` exists-paths). When
        the same canonical decision is met again from a *different* ``source`` (e.g. a
        node first authored ``user``, later swept in by ``mitos import`` as
        ``import_llm``), it records that cross-source re-encounter exactly once per
        ``(node, source)``.

        The signal carries the **new** source; the stored node ``source`` is never
        mutated (first-seen-wins, MI-4-fenced). V1a's composite PK
        ``(node_id, signal_type, source)`` makes a repeat from the same new source a
        clean ``INSERT OR IGNORE`` no-op, so the per-``(node, source)`` audit row is
        written at most once (DoD #1 / §8.2); a third, distinct source mints a second
        row. ``commit_parsed_entry`` gets ZERO change — it is never reached on an
        existing node, so the re-encounter is detected here, at the gate.

        The audit is best-effort: a hard SQLite failure logs and is dropped so a
        non-critical provenance row can never abort a real sync / import / record —
        the decision graph is authoritative, the audit trail is not (P7 bulkhead).
        The next re-encounter re-writes it idempotently (MI-3). A PK collision is not
        a failure (``INSERT OR IGNORE`` swallows it, no raise).

        Args:
            node_id: The re-encountered node's content-hash id.
            stored_source: The node's first-seen (MI-4-fenced) ``source``.
            new_source: The re-encountering ``source`` ∈ {user, capture_llm, import_llm}.

        Returns:
            True iff a signal write was committed (or idempotently ignored); False if
            the new source matched the stored source (no write needed) or the
            best-effort write was dropped on a hard SQLite failure.
        """
        if new_source == stored_source:
            return False
        try:
            self.write_signal(node_id, "source_reencounter", new_source)
            return True
        except DatabaseError as e:
            # Audit is best-effort; the decision graph is not. A failed signal write
            # must never abort a real sync / import / record (the emit sits at gates
            # that lie outside the per-entry commit quarantine) — log loud and drop;
            # the next re-encounter re-writes it idempotently (MI-3 / INSERT OR IGNORE).
            logger.warning(
                "source_reencounter audit signal dropped for node '%s' "
                "(new source '%s'): %s",
                node_id, new_source, e,
            )
            return False

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
        (MI-3 / V1-D17). ``node_scopes`` reconcile idempotently (casefolded and
        deduped, in authored order with each tag's position as its ``ordinal``; a
        changed list — a same-set reorder included — rewrites the node's rows, an
        equal one writes nothing; MI-9) and ``transcripts`` are
        write-once-preserve (insert / update / no-op-on-absent, **never** DELETE;
        V1-D16).

        Edge reconciliation is live as of 5b: ``_reconcile_edges`` commits the two
        kill-edges (``supersedes`` / ``corrects``) via declarative mirror, and a
        post-mutation slug-collision assertion (V1-D4) enforces one active node per
        ``casefold(slug)``. A referential violation rolls the whole entry back with
        a structured ``source="store"`` envelope. The decision outbox row and,
        field-aware, the rejection backlog row are enqueued on the same cursor, so
        they commit with the node or not at all. Format validation is the parser's
        job (C1 / Decision 2) — the store owns referential truth, not format truth.

        Args:
            parsed: A well-formed ``ParsedEntry`` (the parser's C1 guarantee).

        Returns:
            A ``CommitDelta`` carrying the committed ``node_id``, the scope /
            commentary-change metadata, whether the node is new (``created``),
            and the ``enqueue_seq`` of the rejection backlog row this commit
            minted (``rejection_enqueue_seq``, ``None`` when it queued none).

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
        #
        # Slug-length backstop (both kinds): the parser gates this on every file route
        # and the record path gates it early, but the slug is the handle other
        # decisions cite — an over-length one reaching the store means a caller bypassed
        # the parser, so fence it here too (an unbypassable format invariant, MI-11 spirit).
        if len(parsed.slug) > SLUG_MAX_LEN:
            raise ValidationError(
                f"Slug '{parsed.slug}' reached the store at {len(parsed.slug)} "
                f"characters — over the {SLUG_MAX_LEN}-character limit; the parser's "
                "format guarantee was bypassed."
            )
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

        # Incoming scopes through the shared scope-tag rule (`scope_tags`): the
        # parser already normalizes, and re-applying it is idempotent and keeps a
        # hand-built entry honest (MI-9). It is the SAME function the divergence
        # comparator calls, so "equal" here and "not diverged" there cannot drift
        # apart. The list keeps the author's order (first occurrence wins a
        # duplicate): the order is the ``ordinal``, and the first tag is the node's
        # primary scope. Order is commentary-tier — it never reaches
        # ``compute_node_id`` above (C1).
        incoming_scopes = normalize_scope_tags(parsed.scope)
        incoming_transcript = parsed.transcript or None  # falsy -> absent

        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()

                # --- Read prior state (same content-hash id => same node) -------
                prior = cursor.execute(
                    "SELECT * FROM nodes WHERE id = ?", (node_id,)
                ).fetchone()
                prior_scopes = [
                    r["scope"]
                    for r in cursor.execute(
                        "SELECT scope FROM node_scopes WHERE node_id = ? "
                        f"ORDER BY {_SCOPE_ORDER_SQL}",
                        (node_id,),
                    )
                ]
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
                # List equality: a same-set reorder is a footprint change (it moves
                # the primary scope), so it rewrites the rows and ticks updated_at.
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

                # --- Auto-register cited mechanisms (5a; V1-D5 / V1-D15) --------
                # Takes the open cursor so the registry write enlists in THIS
                # transaction (MI-5 atomicity — a later store-stage failure below
                # rolls the mechanism rows back too). Gated ``is_new and decision``:
                # ``mechanism_refs`` is part of the canonical core, so a genuinely
                # new mechanism can ONLY arrive on a node whose id is novel
                # (``is_new``) — a same-id commentary re-commit never carries a new
                # mechanism, so the gate is provably complete, not just an
                # optimization (Decision 2; MI-5 / DoD #2 pass by construction).
                if is_new and parsed.kind == "decision":
                    self._register_mechanisms(cursor, parsed, now)

                # --- Reconcile node_scopes (rewrite in authored order) ----------
                # Only a changed list writes: the node's rows are deleted and
                # re-inserted with each tag's position as its ordinal (a handful of
                # rows under the PK). An equal list writes nothing, so a
                # byte-identical re-commit stays a true no-op (MI-3).
                if scopes_changed:
                    cursor.execute(
                        "DELETE FROM node_scopes WHERE node_id = ?", (node_id,)
                    )
                    for ordinal, tag in enumerate(incoming_scopes):
                        cursor.execute(
                            "INSERT INTO node_scopes (node_id, scope, ordinal) "
                            "VALUES (?, ?, ?)",
                            (node_id, tag, ordinal),
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
                # --- Rejection backlog (field-aware, same transaction) ----------
                # Only a decision can hold rejected items. A new one enqueues when
                # its field enumerates at least one item (the leaf decides what an
                # item is; nothing for a new owner with none). A re-commit enqueues
                # when the stored field changed, compared raw with None read as
                # "" — a change to empty included, since its old items must go.
                # Every other edit leaves a standing row untouched.
                rejection_enqueue_seq: Optional[int] = None
                if parsed.kind == "decision":
                    if is_new:
                        rejection_due = (
                            enumerate_rejected_paths(rejected_paths_json or "").shape
                            != "empty"
                        )
                    else:
                        rejection_due = (
                            (prior["rejected_paths_json"] or "")
                            != (rejected_paths_json or "")
                        )
                    if rejection_due:
                        rejection_enqueue_seq = self._enqueue_rejection_backlog(
                            cursor, node_id, now
                        )
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
                # ``cascade_affected_scopes`` is FIRST-ORDER as of 5c (no V1b
                # cascade; V4 full-regenerates, §5.2.4): the union of (i) the committing node's
                # entered/left scopes when its own footprint changed (the 5a
                # behavior) and (ii) the scopes of the nodes this commit currently
                # kill-edges, when the outgoing edge set changed — a superseded /
                # corrected target leaves the active view, so every scope it was in
                # needs a re-render (C3). The targets are read from the
                # POST-reconciliation ``edges`` (the cursor sees 5b's just-written
                # rows; the read runs well after the seam at line 847). The
                # resurrection-target gap is deliberate (a DELETEd kill-edge's target
                # is no longer outgoing → not captured here; V5 owns the transitive
                # treatment — forward-safe: no live V1a consumer reads this field, only
                # ``delta.node_id``). The ``CommitDelta`` field shape is unchanged —
                # V1b extends the population, never rebuilds the struct. A
                # byte-identical re-commit trips neither gate -> ``[]``.
                affected = set()
                if is_new or direct_footprint_changed:
                    # A set of scope names to re-render, not a node's order.
                    affected |= set(incoming_scopes) | set(prior_scopes)
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
                    # Authored order, the same order the hydrating read returns.
                    node_scope=list(incoming_scopes),
                    self_old_scope=list(prior_scopes),
                    commentary_fields_changed=(
                        False if is_new else direct_footprint_changed
                    ),
                    cascade_affected_scopes=cascade_affected_scopes,
                    created=is_new,
                    rejection_enqueue_seq=rejection_enqueue_seq,
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

    def _register_mechanisms(
        self, cursor: sqlite3.Cursor, parsed: ParsedEntry, now: str
    ) -> None:
        """Auto-registers a committing decision's cited mechanisms (first-seen-wins, in-txn).

        Mints a ``mechanisms`` registry row the first time each cited mechanism's
        ``canonical_name`` is seen, recording who coined it and how they spelled it
        (V1-D5 / V1-D15) — the first-seen provenance v0.2's Drift sensor reads to
        weight hallucination-risk. ``INSERT OR IGNORE`` on the ``canonical_name`` PK
        gives first-seen-wins by construction: a later citation of the same canonical
        no-ops, preserving the earliest ``authored_name`` / ``source`` / ``created_at``
        (and MI-3 replay-safety — a re-run no-ops, never raises).

        Iterates ``parsed.mechanisms`` — the AUTHORITATIVE folded set the node itself
        stores and hashes (``mechanism_refs_json``) — so the registry mirrors the
        node's refs exactly. ``authored_name`` (first-seen presentation casing) is
        resolved by canonical-MATCH against ``parsed.mechanisms_authored`` (the raw
        pre-fold tokens), NOT a direct read: LLM enrichment (``sync.py``) overwrites
        ``mechanisms`` but NOT ``mechanisms_authored``, so a direct read would register
        stale pre-enrichment tokens. Falls back to the ref itself when no raw token
        folds to its canonical (the enrichment-changed-the-set or ``importer.py``
        raw-build path — the accepted R4-class degradation to folded casing).

        An empty-fold token (``mechanism_canonical_norm("!!!") == ""``, the
        self-cleaning degenerate-mechanism case) is skipped — it must never mint a
        junk ``canonical_name=''`` PK row.

        Enlists in the open ``cursor`` (atomic with the node/edge writes): a later
        store-stage failure in the same entry rolls the registration back too (MI-5).

        Args:
            cursor: The open transaction cursor from ``commit_parsed_entry``.
            parsed: The committing decision entry (gated to ``kind == "decision"`` by
                the call site).
            now: The commit's single ``_utc_now_iso()`` stamp (MI-10), reused as the
                registry row's ``created_at``.

        Returns:
            None.
        """
        source = parsed.source or "user"  # V1-D20: absent **Source:** -> "user"
        for ref in parsed.mechanisms:
            canonical = mechanism_canonical_norm(ref)
            if not canonical:
                continue  # empty-fold degenerate token — no junk PK row
            authored = next(
                (
                    raw
                    for raw in parsed.mechanisms_authored
                    if mechanism_canonical_norm(raw) == canonical
                ),
                ref,  # fallback: the ref (enrichment-changed set / importer raw path)
            )
            # Parameterized (P8); INSERT OR IGNORE is the first-seen-wins conflict
            # policy on the ``canonical_name`` PK.
            cursor.execute(
                "INSERT OR IGNORE INTO mechanisms "
                "(canonical_name, authored_name, source, created_at) "
                "VALUES (?, ?, ?, ?)",
                (canonical, authored, source, now),
            )

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
                # ``Cites:`` (the mutation-cycle concern for amends/narrows is the
                # write-time reachability probe below, not a blanket source-active
                # reject).
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
                # --- Write-time mutation-cycle rejection (W8, §4.3) --------------
                # A net-new mutation edge ``node_id → target`` points new→old, so it
                # makes ``node_id`` a DESCENDANT of ``target``. It CLOSES a cycle iff
                # ``target`` already reaches ``node_id`` through the mutation union —
                # i.e. ``node_id`` is already a mutation ancestor of ``target``. This
                # is the exact case the source-active kill-edge reject above CANNOT
                # catch once non-kill edges commit: a live-source ``supersedes`` /
                # ``amends`` / ``narrows`` looping back through an existing non-kill
                # ``amends`` / ``narrows`` (which retire nothing, so the active-source
                # guard never fires). The probe is the SAME walk ``get_lineage``
                # exposes, with the EXACT-reject reaction (zero false positives: a
                # convergent DAG only shares reachability, it never makes the target
                # reach the source). Runs only for the mutation union (Decision 2,
                # ``corrects`` excluded), only on net-new edges (a retained edge
                # already passed). The reachability bound inside the walk keeps even a
                # corrupt existing graph from hanging the hot ``record`` path; the
                # failure funnels through the same ``CommitError`` raise the other
                # rejects use (no new code, no new raise site — §5.2.2).
                if edge_type in _MUTATION_EDGE_FIELDS:
                    ancestor_ids, _ = self._walk_mutation_lineage(
                        cursor, target["id"]
                    )
                    if node_id in ancestor_ids:
                        failures.append(
                            FailureItem(
                                code=STORE_CYCLE_VIOLATION,
                                source="store",
                                message=(
                                    f"A {edge_type} edge to '{citation}' would close "
                                    "a mutation cycle: that entry already descends "
                                    "from this one through the lineage chain "
                                    "(supersedes/amends/narrows), so the edge would "
                                    "make the entry its own ancestor."
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
        is a free drain-time cache-hit (the cheap false positive). The rejection
        index is not driven from this row: it keeps a field-aware row of its own in
        ``rejection_backlog``, written by :meth:`_enqueue_rejection_backlog` in the
        same transaction, so this outbox stays unconditional on its own grounds
        and never serves as the rejected items' queue.

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

    def _enqueue_rejection_backlog(
        self, cursor: sqlite3.Cursor, node_id: str, now: str
    ) -> int:
        """Queues a decision's rejected items for a write, inside the commit's transaction.

        The in-transaction twin of ``_enqueue_outbox`` for the rejection index: it
        runs on ``commit_parsed_entry``'s cursor, so the row commits with the node or
        not at all (MI-12). The caller decides **whether** to enqueue (the decision is
        new with an item, or its stored ``rejected_paths`` changed); this method
        writes the row and returns the seq it minted.

        It deletes the node's standing row, if any, and inserts a fresh one, so every
        enqueue mints a new ``enqueue_seq`` and starts at ``retry_count = 0``. The
        sequence is ``AUTOINCREMENT`` and never reused within a graph, so a reader
        holding an older seq can never clear or note the newer row.

        Unlike the decision outbox, a re-commit is **not** this row's retry. Only a
        field-aware enqueue resets it; a re-commit that leaves ``rejected_paths``
        unchanged never reaches this method, so a standing row keeps its seq and its
        ``retry_count``, and it is retried by the next pass over the backlog.

        Args:
            cursor: The open cursor inside ``commit_parsed_entry``'s transaction.
            node_id: The committing decision's content-hash id (the node row
                already exists by this point, so the FK is satisfied).
            now: The commit's single UTC ISO-8601 stamp (MI-10).

        Returns:
            The new row's ``enqueue_seq``. It is read as ``cursor.lastrowid``
            straight after the INSERT, with nothing run on the cursor in between;
            on an ``INTEGER PRIMARY KEY`` table that is the inserted seq. The
            commit carries it out on ``CommitDelta.rejection_enqueue_seq``.
        """
        cursor.execute(
            "DELETE FROM rejection_backlog WHERE node_id = ?", (node_id,)
        )
        cursor.execute(
            "INSERT INTO rejection_backlog (node_id, queued_at, retry_count) "
            "VALUES (?, ?, 0)",
            (node_id, now),
        )
        return int(cursor.lastrowid)

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

    def get_active_decision_ids(self) -> FrozenSet[str]:
        """Returns the ids of the active-view decisions, unhydrated.

        The same population :meth:`get_active_decisions` returns (and so the same
        one ``check`` sweeps), read as ids only for the uncovered-decision
        derivation (``mitos.audit_debt``): no ``_IS_DRIFTED_SQL``, no
        ``_hydrate_rows``. A drifted decision stays in, because the predicate does
        not retire it and neither does the sweep. Unscoped on purpose: the gate's
        population is the whole corpus.

        The two reads must never disagree, so this binds the shipped
        ``_ACTIVE_VIEW_PREDICATE`` rather than re-encoding it, and ``nodes`` stays
        UNALIASED so the correlated predicate binds (the alias trap the constant's
        own comment warns about). ``tests/test_audit_debt.py`` pins the agreement
        over a fixture matrix (superseded, corrected, drifted, open questions).

        Store faults propagate (``DatabaseError`` / ``sqlite3.Error``); the
        derivation is what turns them into a result.

        Returns:
            The active decision node ids.
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT nodes.id FROM nodes "
                f"WHERE nodes.kind = 'decision' AND {_ACTIVE_VIEW_PREDICATE}"
            ).fetchall()
            return frozenset(row[0] for row in rows)
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
        """Exposes the open-question view with computed Stage-1 + Stage-2 state.

        This method *is* ``oq_state_view`` — the single OQ read surface every
        consumer (CLI/MCP visibility verbs, future V4/V5) calls. It applies two
        independent, query-time axes (M3 — state is computed, never stored):

          - **Stage 1 (active-view membership, V1a):** the SAME kill-edge anti-join
            as the decision views (``_ACTIVE_VIEW_PREDICATE``) — an OQ with an
            incoming ``supersedes`` / ``corrects`` (a typo-fix / retirement) is
            EXCLUDED entirely, exactly as a superseded decision leaves the live set.
          - **Stage 2 (resolution, V1b):** each surviving OQ is labeled
            ``state ∈ {parked, resolved}`` (``_OQ_RESOLVED_SQL``): ``resolved`` iff
            it has ≥1 incoming ``resolves`` edge from a STILL-ACTIVE decision, else
            ``parked``. Resolution does NOT flow transitively through ``supersedes``
            (only direct incoming ``resolves`` + their source's liveness are
            checked), so superseding the resolver without re-declaring ``Resolves:``
            flips the OQ back to ``parked`` automatically — self-healing for free
            (§4.5.1). ``resolved`` OQs are RETURNED (``resolves`` is not a kill
            edge); callers foreground ``parked``.

        Hydrated to the reader-facing OQ shape (``topic`` / ``questions_raised``)
        and modifier-stamped — so an amended-but-active OQ carries ``amended_by`` /
        ``narrowed_by`` and never reads as the final word (the 2b chokepoint reaches
        the OQ surface for free via ``_hydrate_rows``).

        Args:
            scope: Optional scope tag; filtered in SQL via the indexed
                ``node_scopes`` membership join (P11).

        Returns:
            Open-question node dicts (reader-facing shape), each carrying a computed
            ``state`` (``"parked"`` / ``"resolved"``) plus stamped modifier keys
            when a later ``amends`` / ``narrows`` has moved on from it.
        """
        conn = self._get_connection()
        try:
            sql = (
                f"SELECT nodes.*, {_IS_DRIFTED_SQL}, {_OQ_RESOLVED_SQL} FROM nodes "
                f"WHERE nodes.kind = 'open_question' AND {_ACTIVE_VIEW_PREDICATE}"
            )
            params: List[str] = []
            if scope is not None:
                sql += _SCOPE_FILTER_SQL
                params.append(scope)
            rows = conn.execute(sql, params).fetchall()
            nodes = self._hydrate_rows(conn, rows)
            # Zip the computed Stage-2 state onto each hydrated node — mirrors
            # get_decisions' computed_state zip, minus the state_matches filter
            # (oq_state_view returns ALL Stage-1 survivors labeled; the caller
            # foregrounds parked). _hydrate_rows preserves row order.
            for node, row in zip(nodes, rows):
                node["state"] = _computed_oq_state(row["is_resolved"])
            return nodes
        finally:
            conn.close()

    def get_scope_counts(self, include_archived: bool = False) -> Dict[str, Dict[str, int]]:
        """Per scope tag, the live-node counts the discovery surface advertises.

        Computes, in one DB-side state-predicated aggregate over the
        ``node_scopes``↔``nodes`` join, two *live* counts per scope tag: active
        decisions and parked open questions. The counts reuse the read verbs' own
        liveness predicates in SQL (``_ACTIVE_VIEW_PREDICATE`` for the kill-edge
        anti-join; ``_OQ_RESOLVED_PREDICATE`` for OQ resolution) rather than the
        verbs' materialized Python output — so by construction the active-decision
        count for a scope equals ``len(get_decisions(scope=X, state="active"))`` and
        the parked-OQ count equals the ``parked``-labeled subset of
        ``get_open_questions(scope=X)``. The aggregation runs in SQLite; only the
        small tag→counts map crosses into Python (no node hydration).

        Scope keys are returned exactly as stored in ``node_scopes`` — *already
        casefolded* at write time (MI-9 honored by the write path, not here), so no
        runtime fold is applied and the keys are canonical MI-9 forms.

        Args:
            include_archived: When ``False`` (default), returns only *live* domains
                (≥1 active decision OR ≥1 parked open question); a fully
                superseded/resolved domain that computes to 0/0 is excluded (P4
                self-tidying). When ``True``, additionally includes every other tag
                present in ``node_scopes`` at a ``0/0`` floor — the scope-level
                parallel of ``list --state all`` — enumeration-identical to
                ``get_all_scopes()``.

        Returns:
            A map keyed by scope tag, each value
            ``{"active_decisions": int, "parked_open_questions": int}``, ordered
            alphabetically by scope (deterministic; presentation order is the
            caller's job). Empty graph / empty vocabulary returns ``{}``.
        """
        conn = self._get_connection()
        try:
            # One pass, conditional aggregation. The JOIN keeps ``nodes`` UNALIASED
            # so both reused liveness predicates (which correlate on the literal
            # ``nodes.id``) bind correctly — aliasing ``nodes`` would silently make
            # them match nothing (the §4.3 alias trap). ``node_scopes`` IS aliased
            # (``ns``); the predicates never reference it.
            sql = (
                "SELECT ns.scope, "
                f"SUM(CASE WHEN nodes.kind = 'decision' AND {_ACTIVE_VIEW_PREDICATE} "
                "THEN 1 ELSE 0 END) AS active_decisions, "
                f"SUM(CASE WHEN nodes.kind = 'open_question' AND {_ACTIVE_VIEW_PREDICATE} "
                f"AND NOT {_OQ_RESOLVED_PREDICATE} THEN 1 ELSE 0 END) AS parked_open_questions "
                "FROM node_scopes ns "
                "JOIN nodes ON nodes.id = ns.node_id "
                "GROUP BY ns.scope"
            )
            if not include_archived:
                sql += " HAVING active_decisions > 0 OR parked_open_questions > 0"
            sql += " ORDER BY ns.scope"
            rows = conn.execute(sql).fetchall()
            return {
                row["scope"]: {
                    "active_decisions": int(row["active_decisions"]),
                    "parked_open_questions": int(row["parked_open_questions"]),
                }
                for row in rows
            }
        finally:
            conn.close()

    def get_scope_discrimination(self) -> Dict[str, Dict[str, int]]:
        """Per scope tag, how often active decisions chose it first and what it shares.

        The discriminator beside ``get_scope_counts``' liveness counts. A tag carried
        by many active decisions but written first by few has stopped telling those
        decisions apart; the pair makes that visible without judging it:

        - ``authored_first_decisions``: active decisions whose first tag is this one.
          "First" is the first row under the read rule ``_SCOPE_ORDER_SQL`` (authored
          ordinal, tag as tiebreak) — never ``ordinal = 0``, which an older writer
          gives every tag of a node.
        - ``co_tagged_scopes``: the number of distinct other tags carried by at least
          one active decision that also carries this one.

        Both are decision-only, over the live set ``_ACTIVE_VIEW_PREDICATE`` selects
        (the set ``get_decisions(state="active")`` returns). One DB-side aggregate;
        tags are ranked among each node's own rows before liveness filters nodes, and
        only the tag→pair map crosses into Python.

        A read-only store over a graph without ladder step 4 has no ``ordinal``
        column; on exactly that error the aggregate re-runs with primacy ordered by
        tag, the order such a graph holds (as ``_scopes_for`` does). Any other SQLite
        error propagates.

        Returns:
            ``{scope: {"authored_first_decisions": int, "co_tagged_scopes": int}}``
            for every tag carried by at least one active decision, alphabetical by
            the stored (casefolded) tag. An empty graph returns ``{}``.
        """
        conn = self._get_connection()
        try:
            try:
                rows = conn.execute(_scope_discrimination_sql(_SCOPE_ORDER_SQL)).fetchall()
            except sqlite3.OperationalError as e:
                if _MISSING_ORDINAL_ERROR not in str(e):
                    raise
                rows = conn.execute(_scope_discrimination_sql("scope")).fetchall()
            return {
                row["scope"]: {
                    "authored_first_decisions": int(row["authored_first_decisions"]),
                    "co_tagged_scopes": int(row["co_tagged_scopes"]),
                }
                for row in rows
            }
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
            
    def get_all_node_ids(self) -> FrozenSet[str]:
        """Returns every node id in the graph — any kind, any state — unhydrated.

        One ``SELECT id FROM nodes``: the "is this point's node still in the graph?"
        side of the orphan partition (:func:`mitos.vector_store.partition_points`),
        read by ``reconcile`` and ``status`` after their listing. It stays **off**
        :class:`~mitos.protocols.GraphStoreProtocol`: both callers hold a concrete
        ``GraphStore``, so no duck-typed caller reaches it. ``tests/test_orphan_collection.py``
        pins it equal to the ids :meth:`get_all_nodes` returns.

        Store faults propagate; each caller decides what a fault means for it.

        Returns:
            Every node id (SHA-256 hex) the graph holds.
        """
        conn = self._get_connection()
        try:
            rows = conn.execute("SELECT id FROM nodes").fetchall()
            return frozenset(row[0] for row in rows)
        finally:
            conn.close()

    def get_active_node_ids(self) -> Set[str]:
        """Returns the active node id set (active decisions + open questions).

        This is a count of *the live architectural surface*
        and nothing else: ``status``' ``active_nodes`` / ``missing_active_*``,
        ``reconcile``'s ``active`` key and the rebuild's
        ``reconstructed_active_count``. It is **not** what the decision index should
        hold — that is :meth:`get_decision_index_expected_ids`, a superset (it keeps
        every retired decision too), and every "what should the index hold?" consumer
        reads that one (``tests/test_expected_set.py`` pins who calls which).

        One ids-only SQL over the shipped ``_ACTIVE_VIEW_PREDICATE`` (WIRING_LEDGER
        entry-009): the same population as the ids of :meth:`get_active_decisions`
        plus :meth:`get_open_questions` — both apply exactly that predicate unscoped,
        a drifted decision stays in and a parked or resolved open question stays in —
        without hydrating a row. ``nodes`` stays UNALIASED so the correlated
        predicate binds. ``tests/test_rejection_covering.py`` pins the agreement with
        the hydrating derivation over a fixture matrix.

        Returns:
            The set of active decision + open-question node ids (SHA-256 hashes).
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT nodes.id FROM nodes "
                "WHERE nodes.kind IN ('decision', 'open_question') "
                f"AND {_ACTIVE_VIEW_PREDICATE}"
            ).fetchall()
            return {row[0] for row in rows}
        finally:
            conn.close()

    def get_decision_index_expected_ids(self) -> FrozenSet[str]:
        """Returns the decision index's expected set, as ids, unhydrated.

        Every decision node in every computed state (active, drifted, superseded,
        corrected) plus the open questions that survive Stage 1 (parked and
        resolved; a superseded or corrected one is out). A decision's vector lives
        exactly as long as its node (ADR
        ``graveyard-is-deliberate-every-decision-vector-lives-as-long-as-its-node``),
        so this — not :meth:`get_active_node_ids` — is what ``status`` reports
        completeness against, what ``reconcile`` diffs, what the rebuild/cutover
        seed keeps in the outbox and what ``recall.missing_index_is_a_gap`` gates
        on. One read for the sensor and the heal is what lets the heal clear its
        own warning (ADR ``decision-index-gap-and-creation-carve-out-key-on-its-expected-set``).
        The active set is a subset of this one by construction.

        Binds ``_DECISION_INDEX_EXPECTED_PREDICATE`` rather than re-encoding it, and
        ``nodes`` stays UNALIASED so the correlated liveness predicate binds (the
        alias trap the constant's own comment warns about). Its per-record twin is
        :meth:`decision_index_expects_node_other_than`; ``tests/test_expected_set.py``
        pins the agreement over a fixture matrix.

        Store faults propagate (``DatabaseError`` / ``sqlite3.Error``); each caller
        decides what a fault means for it.

        Returns:
            The node ids the decision index should hold a point for.
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT nodes.id FROM nodes "
                f"WHERE {_DECISION_INDEX_EXPECTED_PREDICATE}"
            ).fetchall()
            return frozenset(row[0] for row in rows)
        finally:
            conn.close()

    def decision_index_expects_node_other_than(self, node_id: str) -> bool:
        """Reports whether the decision index's expected set holds any node besides ``node_id``.

        The cheap complement of :meth:`get_decision_index_expected_ids` for the one
        question a single-node write needs answered: *does this write cover what the
        decision index should hold, or is there a node it leaves unindexed?* An empty
        graph, or a graph holding only this node, answers ``False`` — the
        fresh-project carve-out, where one point genuinely **is** the whole expected
        set. A graph holding only retired decisions answers ``True``: they are
        expected too, so a lone record into it must not create a collection holding
        one point.

        Deliberately not ``len(get_decision_index_expected_ids() - {node_id}) > 0``:
        that is a whole-corpus id read on **every** record, not only when it matters.
        This is an ``EXISTS … LIMIT 1`` over the same predicate: an index seek
        (P11/P17), and any decision satisfies the first disjunct.

        The two must never disagree, so this binds the one
        ``_DECISION_INDEX_EXPECTED_PREDICATE`` rather than re-encoding it ("Active-view
        liveness is ONE SQL atom", PATTERNS). Note what is *not* in it: no
        ``_OQ_RESOLVED_PREDICATE`` clause. ``resolves`` is not a kill edge, so a
        *resolved* open question is still returned by ``get_open_questions`` and is
        still expected — it genuinely needs an embedding, and excluding it here would
        make this gate claim coverage the write does not deliver.
        ``tests/test_expected_set.py`` pins the agreement over a fixture matrix rather
        than trusting it.

        Args:
            node_id: The node the calling write covers.

        Returns:
            True iff at least one expected node other than ``node_id`` exists.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT 1 FROM nodes "
                f"WHERE {_DECISION_INDEX_EXPECTED_PREDICATE} "
                "AND nodes.id != ? LIMIT 1",
                (node_id,),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def rejection_index_expects_owner_other_than(self, node_id: str) -> bool:
        """Reports whether any decision node besides ``node_id`` exists, in any state.

        The inline rejection write's creation licence: a write may create the
        absent rejection collection only when its decision is the graph's only
        decision, live or retired (ADR
        ``rejection-collection-creation-follows-the-decision-index-a-seed-marker-of-its-own``).
        Any other decision could own items the write would leave unindexed in a
        collection that then looks built. Open questions never count — the
        rejection index holds none — which is why this is not
        :meth:`decision_index_expects_node_other_than`: that one counts surviving
        open questions, so a project's first decision beside a parked question
        could never create.

        Binds ``_REJECTION_INDEX_OWNER_PREDICATE`` rather than re-encoding it. An
        ``EXISTS … LIMIT 1`` read: it stops at the first other decision row (the
        ``UNIQUE (id, kind)`` autoindex covers it), so no index on ``kind`` is
        needed. Store faults propagate, as the twin's do; the inline leg catches
        them.

        Args:
            node_id: The decision the calling write covers.

        Returns:
            True iff at least one decision node other than ``node_id`` exists.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT 1 FROM nodes "
                f"WHERE {_REJECTION_INDEX_OWNER_PREDICATE} "
                "AND nodes.id != ? LIMIT 1",
                (node_id,),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def stamp_embedding_seed(self, established_by: str) -> None:
        """Records that ``pending_embeddings`` covers the decision index's expected set.

        The standalone-connection twin of :func:`write_embedding_seed`, for the
        producer that is not already inside the transaction it describes
        (``reconcile_embeddings``). Re-stamping replaces the existing row rather than
        raising — the marker records the latest covering act, not a history.

        Args:
            established_by: The verb that established coverage — one of
                ``migrations.EMBEDDING_SEED_ESTABLISHED_BY``.

        Raises:
            DatabaseError: If the write fails, including the ``CHECK`` refusing an
                ``established_by`` outside the declared vocabulary.
        """
        conn = self._get_connection()
        try:
            with conn:
                write_embedding_seed(conn, established_by)
        except sqlite3.Error as e:
            raise DatabaseError(
                f"Failed to stamp the embedding seed marker "
                f"(established_by must be one of "
                f"{', '.join(EMBEDDING_SEED_ESTABLISHED_BY)}): {str(e)}"
            )
        finally:
            conn.close()

    def embedding_seed(self) -> Optional[Dict[str, str]]:
        """Returns the coverage marker, or None when no covering act stands.

        Absence of the row is the false state — there is no ``covers = 0`` row to
        interpret. Two readers: ``MitosSyncManager.drain_pending_embeddings`` acts on
        it (read once at its entry), and ``mitos status`` prints it. The drain is the
        only one that acts.

        An unreadable marker (a hand-truncated graph missing the table entirely) reads
        as **absent**, not as a fault: the drain's safe answer to "may this write
        create the collection?" is *no*, and deferral costs an outbox row that was
        already written. Failing loud here would turn a recoverable graph into a
        failed ``sync`` while still not creating anything.

        Returns:
            ``{"established_by": …, "established_at": …}`` — plain strings, no
            datetime objects crossing the boundary — or ``None``.
        """
        try:
            conn = self._get_connection()
        except DatabaseError:
            return None
        try:
            row = conn.execute(
                "SELECT established_by, established_at FROM embedding_seed WHERE id = 1"
            ).fetchone()
            if row is None:
                return None
            return {
                "established_by": row["established_by"],
                "established_at": row["established_at"],
            }
        except sqlite3.Error:
            return None
        finally:
            conn.close()

    def clear_embedding_seed(self, *, established_at: Optional[str] = None) -> None:
        """Drops the coverage marker — the claim has been delivered or has expired.

        The drain spends the marker it read once its walk has attempted every row
        standing at its entry with the collection present, or once it finds the
        backlog empty (``drain_pending_embeddings`` states the rule). A walk spans
        minutes, and a ``reconcile`` may re-stamp meanwhile, recording a covering
        act this drain did not run under — so the drain passes the
        ``established_at`` it read, and the delete then spends only that marker: a
        re-stamped one survives.

        Args:
            established_at: The stamp of the marker to spend, compared exactly. None
                deletes whatever marker stands.

        Raises:
            DatabaseError: If the delete fails.
        """
        conn = self._get_connection()
        try:
            with conn:
                if established_at is None:
                    conn.execute("DELETE FROM embedding_seed WHERE id = 1")
                else:
                    conn.execute(
                        "DELETE FROM embedding_seed WHERE id = 1 AND established_at = ?",
                        (established_at,),
                    )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to clear the embedding seed marker: {str(e)}")
        finally:
            conn.close()

    # --- The rejection backlog and its seed marker (ladder step 5) ---------------
    #
    # Reads treat a graph without the rung as "rung not applied": a read-only store
    # never migrates, so it can open a step-4 graph. They match exactly the missing
    # table's message; any other SQLite fault raises. Writes never fall back — only a
    # migrated, writable store reaches them, so a missing table there is a fault.

    def get_rejection_backlog_row(self, node_id: str) -> Optional[RejectionBacklogRow]:
        """Returns a decision's standing rejection backlog row, or None.

        Args:
            node_id: The decision's content-hash id.

        Returns:
            The row, or None when no row stands or the graph predates the rung.

        Raises:
            DatabaseError: If the read fails for any other reason.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT node_id, queued_at, enqueue_seq, retry_count "
                "FROM rejection_backlog WHERE node_id = ?",
                (node_id,),
            ).fetchone()
            return _rejection_backlog_row(row) if row is not None else None
        except sqlite3.Error as e:
            if _MISSING_REJECTION_BACKLOG_ERROR in str(e):
                return None
            raise DatabaseError(f"Failed to read the rejection backlog: {str(e)}")
        finally:
            conn.close()

    def claim_rejection_backlog(
        self, limit: int, *, after_seq: Optional[int] = None
    ) -> List[RejectionBacklogRow]:
        """Reads the next rejection backlog rows, in enqueue order.

        A plain ordered ``SELECT``, not a claim — the same single-writer semantics as
        :meth:`claim_pending_embeddings`. Rows come in ``enqueue_seq`` order, and an
        enqueue replaces its row with a fresh, larger seq, so a re-queued owner moves
        to the back of the queue. ``after_seq`` is a keyset cursor on the primary key:
        only rows with a larger seq are returned.

        Args:
            limit: The most rows to return.
            after_seq: The last ``enqueue_seq`` already walked, or None to read from
                the head of the queue.

        Returns:
            Up to ``limit`` rows; ``[]`` when the graph predates the rung.

        Raises:
            DatabaseError: If the read fails for any other reason.
        """
        conn = self._get_connection()
        try:
            if after_seq is None:
                rows = conn.execute(
                    "SELECT node_id, queued_at, enqueue_seq, retry_count "
                    "FROM rejection_backlog ORDER BY enqueue_seq LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT node_id, queued_at, enqueue_seq, retry_count "
                    "FROM rejection_backlog WHERE enqueue_seq > ? "
                    "ORDER BY enqueue_seq LIMIT ?",
                    (after_seq, limit),
                ).fetchall()
            return [_rejection_backlog_row(row) for row in rows]
        except sqlite3.Error as e:
            if _MISSING_REJECTION_BACKLOG_ERROR in str(e):
                return []
            raise DatabaseError(f"Failed to read the rejection backlog: {str(e)}")
        finally:
            conn.close()

    def clear_rejection_backlog_row(self, node_id: str, *, enqueue_seq: int) -> bool:
        """Deletes a decision's backlog row, only if it still carries ``enqueue_seq``.

        A commit that re-queued the owner since the caller read its row replaced
        that row with a larger seq, so a stale clear deletes nothing.

        Args:
            node_id: The decision's content-hash id.
            enqueue_seq: The seq of the row the caller read.

        Returns:
            True iff a row was deleted.

        Raises:
            DatabaseError: If the delete fails.
        """
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM rejection_backlog "
                    "WHERE node_id = ? AND enqueue_seq = ?",
                    (node_id, enqueue_seq),
                )
                return cursor.rowcount == 1
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to clear a rejection backlog row: {str(e)}")
        finally:
            conn.close()

    def note_rejection_backlog_failure(self, node_id: str, *, enqueue_seq: int) -> bool:
        """Counts one failed write attempt on a backlog row, only on ``enqueue_seq``.

        Args:
            node_id: The decision's content-hash id.
            enqueue_seq: The seq of the row the caller read.

        Returns:
            True iff the row's ``retry_count`` was incremented.

        Raises:
            DatabaseError: If the update fails.
        """
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.execute(
                    "UPDATE rejection_backlog SET retry_count = retry_count + 1 "
                    "WHERE node_id = ? AND enqueue_seq = ?",
                    (node_id, enqueue_seq),
                )
                return cursor.rowcount == 1
        except sqlite3.Error as e:
            raise DatabaseError(
                f"Failed to note a rejection backlog failure: {str(e)}"
            )
        finally:
            conn.close()

    def count_rejection_backlog(self, *, min_retry: Optional[int] = None) -> int:
        """Counts the standing rejection backlog rows.

        Args:
            min_retry: Count only rows whose ``retry_count`` is at least this, or
                None for every row.

        Returns:
            The row count; 0 when the graph predates the rung.

        Raises:
            DatabaseError: If the read fails for any other reason.
        """
        conn = self._get_connection()
        try:
            if min_retry is None:
                row = conn.execute("SELECT COUNT(*) FROM rejection_backlog").fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM rejection_backlog WHERE retry_count >= ?",
                    (min_retry,),
                ).fetchone()
            return int(row[0])
        except sqlite3.Error as e:
            if _MISSING_REJECTION_BACKLOG_ERROR in str(e):
                return 0
            raise DatabaseError(f"Failed to count the rejection backlog: {str(e)}")
        finally:
            conn.close()

    def rejection_backlog_last_seq(self) -> Optional[int]:
        """Returns the largest ``enqueue_seq`` standing in the rejection backlog.

        Every enqueue mints a seq larger than any before it, so a row above this
        value was queued after the read.

        Returns:
            The largest standing seq, or None when no row stands or the graph
            predates the rung.

        Raises:
            DatabaseError: If the read fails for any other reason.
        """
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT MAX(enqueue_seq) FROM rejection_backlog").fetchone()
            return None if row is None or row[0] is None else int(row[0])
        except sqlite3.Error as e:
            if _MISSING_REJECTION_BACKLOG_ERROR in str(e):
                return None
            raise DatabaseError(f"Failed to read the rejection backlog: {str(e)}")
        finally:
            conn.close()

    def get_rejection_owner_fields(
        self, node_ids: Sequence[str]
    ) -> Dict[str, Tuple[str, str]]:
        """Reads the slug and the raw ``rejected_paths`` of each named node, in one query.

        ``rejected_paths_json`` holds the **raw** field string, not JSON (see
        :meth:`commit_parsed_entry`), and NULL reads as ``""`` — the same column and
        the same coalesce the field-aware enqueue compared, so the drain enumerates
        exactly what the commit queued it for.

        Args:
            node_ids: The owners to read. Empty asks nothing.

        Returns:
            ``{node_id: (slug, raw_field)}``; a node not in the graph is absent.

        Raises:
            DatabaseError: If the read fails.
        """
        ids = list(dict.fromkeys(node_ids))
        if not ids:
            return {}
        conn = self._get_connection()
        try:
            placeholders = ", ".join("?" for _ in ids)
            rows = conn.execute(
                f"SELECT id, slug, rejected_paths_json FROM nodes "
                f"WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
            return {
                row["id"]: (row["slug"], row["rejected_paths_json"] or "")
                for row in rows
            }
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to read rejection owners: {str(e)}")
        finally:
            conn.close()

    def get_rejection_index_owner_fields(self) -> Dict[str, str]:
        """Reads every rejection index owner's raw ``rejected_paths``, in one query.

        Every decision node in every computed state (``_REJECTION_INDEX_OWNER_PREDICATE``),
        the field coalesced to ``""`` exactly as :meth:`get_rejection_owner_fields`
        does; the column holds the **raw** string, not JSON. This is the graph side
        of the rejection index's expected set: ``rejections.expected_rejection_points``
        turns it into point ids, and ``status`` and ``reconcile`` both diff a listing
        against that, so the heal clears its own warning. It stays off
        :class:`~mitos.protocols.GraphStoreProtocol`: both callers hold a concrete
        ``GraphStore``.

        Returns:
            ``{decision id: raw rejected_paths}`` for every decision node.

        Raises:
            DatabaseError: If the read fails.
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT nodes.id, nodes.rejected_paths_json FROM nodes "
                f"WHERE {_REJECTION_INDEX_OWNER_PREDICATE}"
            ).fetchall()
            return {row[0]: row[1] or "" for row in rows}
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to read rejection index owners: {str(e)}")
        finally:
            conn.close()

    def rejection_index_expects_items(self) -> bool:
        """Reports whether some decision's ``rejected_paths`` holds at least one item.

        The rejection index's expected set is non-empty exactly when this is True,
        so it is what ``recall.missing_rejection_index_is_a_gap`` gates on: an
        absent collection is a gap only when there is something to build. A cursor
        over the decisions with a non-empty column, answering at the first field
        whose enumeration is not ``empty`` (in practice the first row); a
        whitespace-only field is skipped exactly as the enumeration skips it. An
        open question owns no item, so a questions-only graph answers False.

        Store faults propagate; the predicate turns them into its loud answer.

        Returns:
            True iff the rejection index's expected set holds an item.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT nodes.rejected_paths_json FROM nodes "
                f"WHERE {_REJECTION_INDEX_OWNER_PREDICATE} "
                "AND nodes.rejected_paths_json IS NOT NULL "
                "AND nodes.rejected_paths_json != ''"
            )
            for row in cursor:
                if enumerate_rejected_paths(row[0]).shape != "empty":
                    return True
            return False
        finally:
            conn.close()

    def holds_any_decision(self) -> bool:
        """Reports whether the graph holds any decision node, in any state.

        The import's "into an empty graph" question for the rejection seed marker:
        empty means no rejection index owner, so a graph holding only open
        questions is empty for this index. An ``EXISTS … LIMIT 1`` over
        ``_REJECTION_INDEX_OWNER_PREDICATE``.

        Returns:
            True iff at least one decision node exists.

        Raises:
            DatabaseError: If the read fails.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                f"SELECT 1 FROM nodes WHERE {_REJECTION_INDEX_OWNER_PREDICATE} LIMIT 1"
            ).fetchone()
            return row is not None
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to read the graph's decisions: {str(e)}")
        finally:
            conn.close()

    def enqueue_rejection_backlog_owners(self, node_ids: Iterable[str]) -> int:
        """Queues each named owner that has no standing rejection backlog row.

        ``reconcile``'s enqueue, the standalone twin of the commit-only
        ``_enqueue_rejection_backlog``. One transaction; an owner that already has a
        row is left exactly as it stands (``ON CONFLICT(node_id) DO NOTHING``): its
        seq is never re-minted, so a drain mid-walk sees no spurious retake, and its
        ``retry_count`` is never reset, so a poison row keeps reading past the
        tolerance. That deliberately departs from the decision outbox's
        ``add_pending_embedding``, which resets the retry count.

        Args:
            node_ids: The owners to queue (decision ids already in the graph).

        Returns:
            The number of rows inserted.

        Raises:
            DatabaseError: If the write fails.
        """
        ids = list(dict.fromkeys(node_ids))
        if not ids:
            return 0
        now = _utc_now_iso()
        conn = self._get_connection()
        try:
            with conn:
                inserted = 0
                for node_id in ids:
                    cursor = conn.execute(
                        "INSERT INTO rejection_backlog (node_id, queued_at, retry_count) "
                        "VALUES (?, ?, 0) ON CONFLICT(node_id) DO NOTHING",
                        (node_id, now),
                    )
                    inserted += cursor.rowcount
            return inserted
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to queue rejection backlog owners: {str(e)}")
        finally:
            conn.close()

    def stamp_rejection_seed(self, established_by: str) -> None:
        """Records that the rejection backlog covers the rejection index's expected set.

        The standalone-connection twin of :func:`write_rejection_seed`, mirroring
        :meth:`stamp_embedding_seed`. Re-stamping replaces the existing row. Its
        callers are ``reconcile_embeddings`` (when it queued a missing owner) and
        ``MitosProseImporter.import_from_file`` (an import into a graph that held no
        decision).

        Args:
            established_by: The verb that established coverage — one of
                ``migrations.REJECTION_SEED_ESTABLISHED_BY``.

        Raises:
            DatabaseError: If the write fails, including the ``CHECK`` refusing an
                ``established_by`` outside the declared vocabulary.
        """
        conn = self._get_connection()
        try:
            with conn:
                write_rejection_seed(conn, established_by)
        except sqlite3.Error as e:
            raise DatabaseError(
                f"Failed to stamp the rejection seed marker "
                f"(established_by must be one of "
                f"{', '.join(REJECTION_SEED_ESTABLISHED_BY)}): {str(e)}"
            )
        finally:
            conn.close()

    def rejection_seed(self) -> Optional[Dict[str, str]]:
        """Returns the rejection index's seed marker, or None when none stands.

        Mirrors :meth:`embedding_seed`: any unreadable state, a graph without the
        rung included, reads as absent, which is the safe answer to "may this write
        create the collection?".

        Returns:
            ``{"established_by": …, "established_at": …}`` as plain strings, or
            ``None``.
        """
        try:
            conn = self._get_connection()
        except DatabaseError:
            return None
        try:
            row = conn.execute(
                "SELECT established_by, established_at FROM rejection_seed WHERE id = 1"
            ).fetchone()
            if row is None:
                return None
            return {
                "established_by": row["established_by"],
                "established_at": row["established_at"],
            }
        except sqlite3.Error:
            return None
        finally:
            conn.close()

    def clear_rejection_seed(self, *, established_at: Optional[str] = None) -> None:
        """Drops the rejection index's seed marker.

        Mirrors :meth:`clear_embedding_seed`, including the stamp compare: given the
        ``established_at`` the caller read, it spends only that marker, so one
        re-stamped meanwhile survives.

        Args:
            established_at: The stamp of the marker to spend, compared exactly. None
                deletes whatever marker stands.

        Raises:
            DatabaseError: If the delete fails.
        """
        conn = self._get_connection()
        try:
            with conn:
                if established_at is None:
                    conn.execute("DELETE FROM rejection_seed WHERE id = 1")
                else:
                    conn.execute(
                        "DELETE FROM rejection_seed WHERE id = 1 AND established_at = ?",
                        (established_at,),
                    )
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to clear the rejection seed marker: {str(e)}")
        finally:
            conn.close()

    def node_ids_in_commit_order(self) -> List[str]:
        """Returns every node id in the order it was committed into THIS graph file.

        `rowid` order, and the guarantee that buys is stronger than "chronological":
        it is the sequence in which these commits actually **succeeded**, so every
        commit-time constraint held at each step. Replaying in this order therefore
        reproduces a sequence already proven legal.

        That matters because edge citations resolve against the **active view** — a
        `{edge_type}` edge must target an *active* entry — which makes the rule
        commit-time rather than an invariant on the final state. A completed
        supersession permanently points at a retired node, and that is correct: the
        target was active when the citer committed. So any consumer re-materializing a
        SET of entries must emit them in an order where each citation's target still
        commits first, and this is that order.

        Returns:
            Node ids, oldest commit first.
        """
        conn = self._get_connection()
        try:
            return [row["id"] for row in
                    conn.execute("SELECT id FROM nodes ORDER BY rowid")]
        finally:
            conn.close()

    def graph_fingerprint(self) -> Tuple[int, str]:
        """Returns a cheap ``(node_count, max_updated_at)`` change fingerprint.

        The corpus↔graph divergence cache keys on the corpus content hash AND this,
        because divergence is a property of the corpus *versus* the graph: keying on
        corpus content alone means any graph-side change under a static corpus returns
        a stale verdict. The sharp case is a `.bak` restore after a rebuild swap — the
        graph reverts while the corpus keeps its restored blocks, so the corpus hash is
        unchanged, the cache hits, and `status` reports clean over a corpus that has
        diverged the *other* way. P6 invites this directly: "delete the graph and Mitos
        rebuilds" is a sanctioned operation that changes the graph without touching a
        byte of corpus.

        Sufficient against every supported write path: any commentary/scope/edge change
        ticks the committing node's ``updated_at`` through the deferred single-tick
        UPDATE, a new node changes the count, a rebuild re-mints both, and a `.bak`
        restore lowers the MAX. Note one exotic residual, named rather than left to be
        discovered: a wall-clock regression can let a fresh stamp sort below the
        standing MAX with the count unchanged.

        String MAX is safe because every timestamp is offset-aware UTC ISO-8601 with
        one fixed offset (MI-10), so lexical order is chronological order.

        Returns:
            ``(node_count, max_updated_at)``; the string is ``""`` on an empty graph.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(MAX(updated_at), '') AS m FROM nodes"
            ).fetchone()
            return int(row["n"]), str(row["m"])
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

    def get_outgoing_edges(self, node_id: str) -> List[Dict[str, str]]:
        """Lists the committed outgoing edges of a node as write facts.

        Returns each edge as ``{"kind": edge_type, "target": <target's current
        slug>}`` in insertion order. This reads what the commit actually wired,
        never a re-derivation from author input. The repair, amend, divergence,
        restore and settledness paths read it, and the ``created`` receipt falls
        back to it when
        :meth:`get_outgoing_edge_targets` (its healthy-path read) fails, so the
        edges survive with null target keys.

        Args:
            node_id: The source node's id.

        Returns:
            A list of ``{"kind", "target"}`` dicts, empty when the node has no
            outgoing edges.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            rows = cursor.execute(
                "SELECT e.edge_type AS edge_type, n.slug AS slug "
                "FROM edges e JOIN nodes n ON n.id = e.target_id "
                "WHERE e.source_id = ? ORDER BY e.rowid",
                (node_id,),
            ).fetchall()
            return [{"kind": r["edge_type"], "target": r["slug"]} for r in rows]
        finally:
            conn.close()

    def get_outgoing_edge_targets(self, node_id: str) -> List[Dict[str, Any]]:
        """Lists a node's outgoing edges with each target hydrated, stamped and stated.

        The read behind the ``created`` receipt's identification echo (B4): one
        query over the node's outgoing edges joined to their targets, then ONE bulk
        scope fetch and ONE bulk modifier stamp via :meth:`_hydrate_rows` — never a
        per-edge ``get_node`` loop (P11). The state is computed here, at read time
        (M3), in the vocabulary of the target's kind:

          - a decision reads ``superseded`` / ``corrected`` from its incoming
            kill-edge, else ``drifted`` when a drifted signal is set, else
            ``active`` — :meth:`get_node_state`'s logic through the same fragments;
          - an open question reads its kill state if it has one, else its Stage-2
            ``resolved`` / ``parked`` — :meth:`get_open_questions`' logic, so a
            ``resolves`` the caller just wrote is visible in its own read-back.
            (:meth:`get_node_state` is kill-edge-only for an OQ, so it is not used.)

        Both kinds go through :func:`_computed_node_state`, the ladder
        :meth:`get_handle_state` shares.

        The joined ``nodes`` is deliberately UNALIASED and the edges table is
        aliased ``e``: ``_KILLER_TYPE_SQL``, ``_IS_DRIFTED_SQL`` and
        ``_OQ_RESOLVED_SQL`` correlate on the bare ``nodes`` (the §4.3 alias trap).

        Args:
            node_id: The source node's id.

        Returns:
            One dict per edge in insertion order: ``{"kind": edge_type, "target":
            <target's current slug>, "node": <hydrated, modifier-stamped target>,
            "state": <computed state>}``. Empty when the node has no outgoing edges.
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                f"SELECT nodes.*, {_IS_DRIFTED_SQL}, {_KILLER_TYPE_SQL}, "
                f"{_OQ_RESOLVED_SQL}, e.edge_type AS echo_edge_type "
                "FROM edges e JOIN nodes ON nodes.id = e.target_id "
                "WHERE e.source_id = ? ORDER BY e.rowid",
                (node_id,),
            ).fetchall()
            nodes = self._hydrate_rows(conn, rows)
        finally:
            conn.close()
        result: List[Dict[str, Any]] = []
        # _hydrate_rows preserves row order and pops killer_type / is_resolved, so
        # both are read off the raw row; the edge alias is popped so it never rides
        # the hydrated target.
        for node, row in zip(nodes, rows):
            node.pop("echo_edge_type", None)
            state = _computed_node_state(
                node["kind"], row["killer_type"], node["is_drifted"], row["is_resolved"]
            )
            result.append(
                {
                    "kind": row["echo_edge_type"],
                    "target": node["slug"],
                    "node": node,
                    "state": state,
                }
            )
        return result

    def get_incoming_edges(self, node_id: str) -> List[Dict[str, str]]:
        """Lists the committed edges that point AT a node, as write facts.

        The twin of ``get_outgoing_edges``: each edge as ``{"kind": edge_type,
        "source": <citing node's current slug>}`` in insertion order. A rename reads
        this to state which entries still cite the old handle in their markdown.

        Args:
            node_id: The target node's id.

        Returns:
            A list of ``{"kind", "source"}`` dicts, empty when nothing cites the node.
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT e.edge_type AS edge_type, n.slug AS slug "
                "FROM edges e JOIN nodes n ON n.id = e.source_id "
                "WHERE e.target_id = ? ORDER BY e.rowid",
                (node_id,),
            ).fetchall()
            return [{"kind": r["edge_type"], "source": r["slug"]} for r in rows]
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

        Raises:
            DatabaseError: If the read fails. Typed, so a ranked read's stamp fault
                costs the stamps alone (``retrieval_walk`` catches exactly this).
        """
        if not node_ids:
            return {}
        conn = self._get_connection()
        try:
            return self._modifiers_map(conn, node_ids)
        except sqlite3.Error as e:
            raise DatabaseError(
                f"Failed to read modifiers for {len(node_ids)} node(s): {str(e)}"
            )
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

    def _walk_kill_chains(
        self, conn: sqlite3.Connection, node_ids: List[str]
    ) -> Tuple[Dict[str, KillChain], Dict[str, str]]:
        """Walks every seed's kill chain forward, in ONE statement plus memory.

        The shared primitive behind :meth:`get_live_ends` (and any later reader
        that needs ``reached`` — PATTERNS "One walk, N call-sites"). A recursive CTE
        seeded by ``node_ids`` follows each kill edge ``target_id = current`` to its
        ``source_id`` and returns every reached kill edge with its source's slug and
        liveness. ``UNION`` over ``(seed, node)`` rows, never ``UNION ALL``: set
        semantics is what makes the recursion terminate on an out-of-band cycle.

        Liveness is the one atom, composed on the alias: a source is live iff no
        kill edge targets it — ``_KILL_EDGE_TYPES_SQL`` spelled out against
        ``e.source_id`` as ``_modifiers_map``'s source gate does, never
        ``_ACTIVE_VIEW_PREDICATE`` (it binds the unaliased ``nodes``, so inside this
        aliased join it would correlate on nothing and call every node live — the
        alias trap). No ``signals`` read: drifted is live.

        The outer join is a ``CROSS JOIN`` on purpose: it pins SQLite's join order so
        the read drives from the chain rows into ``idx_edges_target``. Left to the
        planner, it scanned every edge and probed the chain (measured on the 50,000-
        decision burial graph: 24.5 ms for five seeds, against 0.4 ms pinned).

        The chain analysis is an iterative three-colour DFS over the fetched edge
        set (``_walk_mutation_lineage``'s idiom), so a cycle is detected rather
        than walked forever, and a convergent DAG is not mistaken for one.

        Args:
            conn: An open connection (the caller owns its lifecycle).
            node_ids: The seeds; repeats and ids naming no node are harmless (an
                unknown id has no chain and gets an empty :class:`KillChain`).

        Returns:
            ``(chains, slugs)``: seed id → :class:`KillChain` for every distinct
            seed, and node id → slug for every node any chain reached.

        Raises:
            sqlite3.Error: From the statement; the public readers wrap it.
        """
        seeds = list(dict.fromkeys(node_ids))
        placeholders = ",".join("?" for _ in seeds)
        rows = conn.execute(
            f"""
            WITH RECURSIVE chain(seed, node) AS (
                SELECT id, id FROM nodes WHERE id IN ({placeholders})
                UNION
                SELECT c.seed, e.source_id
                FROM chain c
                JOIN edges e ON e.target_id = c.node
                WHERE e.edge_type IN {_KILL_EDGE_TYPES_SQL}
            )
            SELECT c.seed AS seed, e.target_id AS target_id,
                   e.source_id AS source_id, n.slug AS slug,
                   NOT EXISTS (
                       SELECT 1 FROM edges k
                       WHERE k.target_id = e.source_id
                         AND k.edge_type IN {_KILL_EDGE_TYPES_SQL}
                   ) AS live
            FROM chain c
            CROSS JOIN edges e ON e.target_id = c.node
            JOIN nodes n ON n.id = e.source_id
            WHERE e.edge_type IN {_KILL_EDGE_TYPES_SQL}
            """,
            seeds,
        ).fetchall()
        killers_of: Dict[str, Dict[str, Set[str]]] = {seed: {} for seed in seeds}
        live: Set[str] = set()
        slugs: Dict[str, str] = {}
        for row in rows:
            killers_of[row["seed"]].setdefault(row["target_id"], set()).add(
                row["source_id"]
            )
            slugs[row["source_id"]] = row["slug"]
            if row["live"]:
                live.add(row["source_id"])

        _WHITE, _GRAY, _BLACK = 0, 1, 2
        chains: Dict[str, KillChain] = {}
        for seed in seeds:
            adjacency = {node: sorted(ks) for node, ks in killers_of[seed].items()}
            fan_out = any(len(ks) > 1 for ks in adjacency.values())
            cycle = False
            reached: List[str] = []
            color: Dict[str, int] = {seed: _GRAY}
            stack: List[Tuple[str, Any]] = [(seed, iter(adjacency.get(seed, [])))]
            while stack:
                node, killers = stack[-1]
                descended = False
                for killer in killers:
                    state = color.get(killer, _WHITE)
                    if state == _GRAY:
                        cycle = True
                        continue
                    if state == _BLACK:
                        continue
                    color[killer] = _GRAY
                    reached.append(killer)
                    stack.append((killer, iter(adjacency.get(killer, []))))
                    descended = True
                    break
                if not descended:
                    color[node] = _BLACK
                    stack.pop()
            anomaly = "cycle" if cycle else ("fan_out" if fan_out else None)
            chains[seed] = KillChain(
                killers=list(adjacency.get(seed, [])),
                reached=reached,
                live_ends=sorted(n for n in reached if n in live),
                anomaly=anomaly,
            )
        return chains, slugs

    def get_live_ends(self, node_ids: List[str]) -> Dict[str, List[str]]:
        """For each id whose kill chain passes through a retired successor, the live ends it reaches.

        A retired decision's ``superseded_by`` / ``corrected_by`` stamps are write
        facts: they name its **immediate** killer whatever that killer's state, so
        on a chain they name a dead hop. This read follows the chain to where the
        current word lives. One batched statement for the whole list
        (:meth:`_walk_kill_chains`).

        Result rule: a seed appears **only** when some immediate killer is itself
        retired. A seed whose killers are all live is absent, because its stamps
        already name the live end; so is a live seed (it has no killer). The value
        lists every live end the walk reached, sorted by ``slug.casefold()`` then
        slug, or ``[]`` when the walk reached none (a cycle with no exit).

        A fan-out (a node with two or more killers) or a cycle can only be authored
        out of band (ADR ``kill-edge-acyclicity-requires-active-source``). Each
        anomalous seed gets one ``logger.warning`` naming it, the anomaly and the
        heal, and the read still answers: non-fatal, never a raise (ADR
        ``get-lineage-cycle-bound-is-nonfatal-partial-return-not-abort``).

        Args:
            node_ids: The retired nodes to follow; ``[]`` reads nothing.

        Returns:
            Node id → live-end slugs, for the admitted seeds only.

        Raises:
            DatabaseError: If the read fails. Typed, so a ranked read's live-end
                fault joins its stamp fault (``retrieval_walk`` catches exactly
                this).
        """
        if not node_ids:
            return {}
        conn = self._get_connection()
        try:
            chains, slugs = self._walk_kill_chains(conn, node_ids)
        except sqlite3.Error as e:
            raise DatabaseError(
                f"Failed to read live ends for {len(node_ids)} node(s): {str(e)}"
            )
        finally:
            conn.close()
        result: Dict[str, List[str]] = {}
        for seed, chain in chains.items():
            if chain.anomaly is not None:
                logger.warning(
                    "get_live_ends: kill-edge %s detected while walking the kill "
                    "chain of node %s (%d node(s) reached, %d live end(s)). The "
                    "supported write path never authors a kill-edge %s; this is an "
                    "out-of-band or corrupt edge — rebuild the derivative graph from "
                    "the markdown corpus (decisions.md and decisions/archive/, the "
                    "authoritative source) with `mitos rebuild`.",
                    chain.anomaly.replace("_", "-"),
                    seed,
                    len(chain.reached),
                    len(chain.live_ends),
                    chain.anomaly.replace("_", "-"),
                )
            live_set = set(chain.live_ends)
            if not any(k not in live_set for k in chain.killers):
                continue
            ends = [slugs[n] for n in chain.live_ends]
            result[seed] = sorted(ends, key=lambda slug: (slug.casefold(), slug))
        return result

    def get_contradictions(self, node_id: str) -> List[Dict[str, str]]:
        """Returns the nodes that contradict ``node_id``, from EITHER direction.

        ``contradicts`` is the catalog's one semantically-symmetric edge — "A
        contradicts B" is the same fact as "B contradicts A" — yet it is stored
        exactly once, as authored (``A`` as ``source_id``, ``B`` as ``target_id``,
        never mirrored: the store-once economy). This is the single safe
        bidirectional read, so no consumer hand-rolls
        ``WHERE source_id = X OR target_id = X`` and silently under-reports
        contradictions by forgetting the ``OR`` (the OD1 hazard, P7).

        Returns counterparts **regardless of either endpoint's active-view state**.
        In v0.1 ``contradicts`` is a flag for v0.2's Conflict pipeline and takes no
        active-view action (§1.1), so a later-superseded endpoint's contradiction is
        still returned — the liveness treatment is v0.2 Conflict's to own. Identity
        only (``{node_id, slug, kind}`` per counterpart), not a hydrated decision
        payload: this is an adjacency read like :meth:`get_edges`, so it stamps no
        modifiers; a consumer wanting a counterpart's full axiom calls
        :meth:`get_node`.

        Args:
            node_id: The node whose contradictions to read.

        Returns:
            One ``{"node_id", "slug", "kind"}`` dict per distinct counterpart node,
            deduplicated across the two edge directions and sorted by slug. Empty
            when the node has no ``contradicts`` edges (or does not exist) — empty is
            healthy, never an error.
        """
        conn = self._get_connection()
        try:
            # UNION (not UNION ALL) dedups the both-directions-authored case
            # (A->B and B->A both present): the counterpart must appear once. The
            # ``!= ?`` guards are cheap insurance against an out-of-band corrupt
            # self-edge (the write path rejects A contradicts A as cycle_violation,
            # but the graph is a rebuildable derivative — M7/P6) so the node never
            # returns itself.
            rows = conn.execute(
                """
                SELECT nodes.id AS node_id, nodes.slug AS slug, nodes.kind AS kind
                FROM edges
                JOIN nodes ON nodes.id = edges.target_id
                WHERE edges.edge_type = 'contradicts'
                  AND edges.source_id = ?
                  AND edges.target_id != ?
                UNION
                SELECT nodes.id AS node_id, nodes.slug AS slug, nodes.kind AS kind
                FROM edges
                JOIN nodes ON nodes.id = edges.source_id
                WHERE edges.edge_type = 'contradicts'
                  AND edges.target_id = ?
                  AND edges.source_id != ?
                ORDER BY slug
                """,
                (node_id, node_id, node_id, node_id),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def _walk_mutation_lineage(
        self, cursor: sqlite3.Cursor, start_id: str
    ) -> Tuple[List[str], Optional[str]]:
        """Walks the transitive mutation-union ancestry outward from ``start_id``.

        The ONE shared walk behind both :meth:`get_lineage` (read) and the
        write-time cycle-rejection guard in :meth:`_reconcile_edges` (the vision's
        "one walk, two call-sites", §4.3). Follows ``start_id``'s OUTGOING mutation
        edges (``supersedes`` ∪ ``amends`` ∪ ``narrows`` — ``_MUTATION_EDGE_FIELDS``;
        ``corrects`` excluded, Decision 2) transitively: mutation edges point
        new→old, so an ancestor is an older node a chain of mutations descends from.

        Depth-first with three-colour cycle detection bounds the traversal so it
        TERMINATES even over a corrupt/out-of-band cyclic graph the supported write
        path never authors (the graph is a rebuildable SQLite derivative — M7/P6). A
        back-edge to a node still on the active DFS path (``_GRAY``) is a true cycle;
        re-reaching a fully-explored node (``_BLACK``) by a second acyclic path — a
        convergent DAG / diamond — is NOT, so a legitimate diamond raises no false
        cycle. On a healthy (acyclic) graph — the invariant the write-guard enforces
        — the walk visits each ancestor exactly once and returns ``cycle_node =
        None``, so the guard's membership test (``source in ancestor_ids``) is exact
        with zero false positives.

        Takes a CURSOR rather than opening a connection so the write-guard reads the
        live transaction's snapshot with no second connection on the hot ``record``
        path. Each node's outgoing edges are materialised (``fetchall``) before
        descending, because the single cursor is reused for every hop.

        Args:
            cursor: An open cursor (the read path's own connection, or the
                committing transaction's cursor).
            start_id: The node whose mutation ancestors to walk. Excluded from the
                result (a node is not its own ancestor).

        Returns:
            A ``(ancestor_ids, cycle_node)`` tuple. ``ancestor_ids`` is the list of
            distinct ancestor node ids in first-seen depth-first order (never
            including ``start_id``). ``cycle_node`` is the id of the node where a
            cycle closed (the back-edge target), or ``None`` when the reachable
            mutation subgraph is acyclic; when a cycle is present ``ancestor_ids`` is
            the partial set walked around the bound (never a hang, never a raise).
        """
        _WHITE, _GRAY, _BLACK = 0, 1, 2
        children_sql = (
            "SELECT target_id FROM edges WHERE source_id = ? "
            f"AND edge_type IN {_MUTATION_EDGE_TYPES_SQL}"
        )

        def _children(node_id: str) -> List[str]:
            # Materialise before descending — the single cursor is reused per hop.
            return [
                row["target_id"]
                for row in cursor.execute(children_sql, (node_id,)).fetchall()
            ]

        ancestors: List[str] = []
        color: Dict[str, int] = {start_id: _GRAY}
        cycle_node: Optional[str] = None
        # Iterative DFS — each stack frame is (node, iterator over its children).
        stack: List[Tuple[str, Any]] = [(start_id, iter(_children(start_id)))]
        while stack:
            node, children = stack[-1]
            descended = False
            for child in children:
                state = color.get(child, _WHITE)
                if state == _GRAY:
                    # Back-edge to a node still on the active path — a true cycle.
                    # Record the first one; keep scanning so the rest of the
                    # reachable set is still collected (the write-guard wants the
                    # complete ancestor set even over a corrupt graph).
                    if cycle_node is None:
                        cycle_node = child
                    continue
                if state == _BLACK:
                    # Already fully explored by another acyclic path — a convergent
                    # DAG, not a cycle. Skip without re-walking.
                    continue
                # _WHITE: a fresh ancestor. ``child`` is never ``start_id`` (it is
                # _GRAY from the top, so it would have matched the back-edge branch).
                color[child] = _GRAY
                ancestors.append(child)
                stack.append((child, iter(_children(child))))
                descended = True
                break
            if not descended:
                color[node] = _BLACK
                stack.pop()
        return ancestors, cycle_node

    def get_lineage(self, node_id: str) -> List[Dict[str, str]]:
        """Returns ``node_id``'s transitive mutation ancestors (``supersedes`` ∪ ``amends`` ∪ ``narrows``).

        Walks OUTWARD from ``node_id`` following its outgoing mutation edges
        (new→old), transitively, via the shared :meth:`_walk_mutation_lineage`.
        ``corrects`` is deliberately excluded (kept acyclic by V1a's active-source
        guard, and walked only by the deferred ``get_transcript`` ancestry — M1,
        Decision 2). The mutation graph is kept acyclic at write time by the guard in
        :meth:`_reconcile_edges` (the same walk), so on the supported path this
        returns the full chain.

        Identity only — ``{node_id, slug, kind}`` per ancestor, the SAME thin shape
        :meth:`get_contradictions` ships — NOT a hydrated/modifier-stamped decision
        payload (this is an adjacency/lineage lookup, not a decision-read surface;
        Decision 3). A consumer wanting an ancestor's axiom calls :meth:`get_node`,
        which stamps.

        Homeostasis bound (P5/P7): on a corrupt/out-of-band cycle the graph never
        meant to hold (M7/P6 — it is a rebuildable derivative), the walk TRUNCATES at
        the cycle, emits a loud ``logger.warning`` naming the node, and returns the
        PARTIAL lineage — it never infinite-loops, never silently truncates, and
        never aborts the caller (a ``record`` over a corrupt graph still completes).

        Args:
            node_id: The node whose mutation ancestry to read.

        Returns:
            One ``{"node_id", "slug", "kind"}`` dict per distinct mutation ancestor,
            sorted by slug; ``node_id`` itself is never included. Empty when the node
            has no outgoing mutation edges (or does not exist) — empty is healthy,
            never an error.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            ancestor_ids, cycle_node = self._walk_mutation_lineage(cursor, node_id)
            if cycle_node is not None:
                # Loud, non-fatal: the supported write path never authors a mutation
                # cycle (the write-time reachability guard rejects every cycle-closer),
                # so this is an out-of-band / corrupt edge. Name the node and surface
                # the repair path; do NOT raise (the hot ``record`` path stays
                # crash-safe over a corrupt graph).
                logger.warning(
                    "get_lineage: mutation cycle detected while walking lineage of "
                    "node %s — the cycle closes at node %s. Returning the partial "
                    "lineage (%d ancestor(s)) walked before the bound fired. The "
                    "supported write path never authors a mutation cycle; this is an "
                    "out-of-band or corrupt edge — rebuild the derivative graph from "
                    "the markdown corpus (decisions.md and decisions/archive/, the "
                    "authoritative source) with `mitos rebuild`.",
                    node_id,
                    cycle_node,
                    len(ancestor_ids),
                )
            if not ancestor_ids:
                return []
            # Hydrate the ancestor ids → identity dicts. ``id`` is the globally-unique
            # ``nodes`` PK, so each id resolves to exactly one row. The placeholder
            # string is built from ``?`` only (no value interpolated) and the ids bind
            # positionally — P8-safe.
            placeholders = ",".join("?" for _ in ancestor_ids)
            rows = cursor.execute(
                "SELECT id AS node_id, slug, kind FROM nodes "
                f"WHERE id IN ({placeholders}) ORDER BY slug",
                ancestor_ids,
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_unregistered_mechanisms(self, parsed: ParsedEntry) -> List[str]:
        """Returns the subset of ``parsed``'s mechanism refs NOT yet in the registry.

        A read-only pre-commit feedback query (registers NOTHING — the whole point is
        to let the author catch a typo or alias *before* the auto-registration ratchet
        fires; V1-D5 / V1-D15). Ships gated with no in-vision consumer — V3a's
        interactive-review surface calls it later. Closure criterion is ship +
        gate-tested, the same P20-forward-wiring convention as
        :meth:`get_contradictions` (W6/W14).

        Iterates ``parsed.mechanisms`` (the authoritative folded set), keys each by
        ``mechanism_canonical_norm`` against the ``mechanisms`` PK, and returns the
        not-yet-present ones in authored presentation form when available
        (canonical-match against ``mechanisms_authored``, else the ref) — order-stable
        (first appearance in ``parsed.mechanisms``) and deduped by ``canonical_name``.
        Empty-fold tokens are skipped (they never register, so they are never
        "unregistered"). Opens its OWN connection (read-only, like :meth:`get_lineage`),
        decoupled from any write transaction.

        Args:
            parsed: The entry whose mechanism refs to check against the registry.

        Returns:
            The authored (or ref-fallback) form of each cited mechanism not yet
            registered, first-seen order, deduped by ``canonical_name``; ``[]`` when
            all are registered, the entry cites none, or it is an open question
            (an OQ carries no mechanisms — empty is healthy, never an error).
        """
        if parsed.kind != "decision":
            return []
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            unregistered: List[str] = []
            seen: Set[str] = set()
            for ref in parsed.mechanisms:
                canonical = mechanism_canonical_norm(ref)
                if not canonical or canonical in seen:
                    continue
                seen.add(canonical)
                row = cursor.execute(
                    "SELECT 1 FROM mechanisms WHERE canonical_name = ?",
                    (canonical,),
                ).fetchone()
                if row is not None:
                    continue  # already registered
                authored = next(
                    (
                        raw
                        for raw in parsed.mechanisms_authored
                        if mechanism_canonical_norm(raw) == canonical
                    ),
                    ref,
                )
                unregistered.append(authored)
            return unregistered
        finally:
            conn.close()

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
        lands with V5 as first consumer (V1b ships no transitive ``corrects`` walk —
        ``get_lineage`` is the mutation-union walk, ``supersedes`` ∪ ``amends`` ∪
        ``narrows``, a distinct edge set; §5.2.3, ADR
        ``transcript-ancestry-corrects-only`` / ``get-transcript-deferred-to-v5``) —
        for the V1a direct read the rule holds trivially (a single node's own
        transcript borrows nothing).

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

    def claim_pending_embeddings(
        self,
        drainer_id: str,
        limit: int = 10,
        *,
        after: Optional[Tuple[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Reads the next rows to drain, in queue order (V1a single-writer semantics).

        A plain ordered ``SELECT`` over ``(queued_at, node_id)`` — not a claim. The
        prototype's ``claimed_by`` machinery (UPDATE ... RETURNING) defers to V3b
        (§5.2.8); V1a is single-writer (``busy_timeout``), so nothing is reserved and
        two concurrent drainers can both read a row. ``node_id`` breaks ties because a
        ``rebuild`` replay can give many rows one ``queued_at`` stamp.

        ``after`` is a keyset cursor: with it, only rows strictly after that
        ``(queued_at, node_id)`` key are returned, by the same order. The drain walks
        the backlog this way, so a row that failed (and so kept its ``queued_at``)
        is passed rather than re-read, and the parameter list stays two values long
        at any backlog size. The keyset is a row-value comparison so that SQLite seeks
        ``idx_pending_embeddings_queue`` (ladder step 5) to the cursor; the equivalent
        ``queued_at > ? OR (queued_at = ? AND …)`` spelling walks the index from its
        head on every claim. The returned rows carry NO ``embedding_text`` (the column
        is gone, C2/M8); the drainer re-derives it per node. ``drainer_id`` is
        retained for surface stability (R12) and is unused.

        Args:
            drainer_id: Unused (R12).
            limit: The most rows to return.
            after: The last ``(queued_at, node_id)`` key already walked, or None to
                read from the head of the queue.

        Returns:
            Up to ``limit`` rows ``{"node_id", "queued_at", "retry_count"}``.

        Raises:
            DatabaseError: If the read fails.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if after is None:
                cursor.execute(
                    """
                    SELECT node_id, queued_at, retry_count
                    FROM pending_embeddings
                    ORDER BY queued_at, node_id
                    LIMIT ?
                    """,
                    (limit,)
                )
            else:
                queued_at, node_id = after
                cursor.execute(
                    """
                    SELECT node_id, queued_at, retry_count
                    FROM pending_embeddings
                    WHERE (queued_at, node_id) > (?, ?)
                    ORDER BY queued_at, node_id
                    LIMIT ?
                    """,
                    (queued_at, node_id, limit)
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
