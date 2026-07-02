"""Sibling telemetry store for the Conflict sensor's non-rebuildable judgments.

The Conflict sensor produces telemetry that **cannot be regenerated** from
``decisions.md``: a SONNET tenability judgment fired once, at temperature, against
a specific prompt + model generation — replay it later and a *different* rationale
comes back, so the one that fired is lost forever unless it is written down whole
(CONF-D8, M8). That corpus is the training set the future ML conflict-classifier
graduates on (OD3), and the observability surface that turns the vision's cost/P95
budgets from assertions into measurements (P15/P16).

Because Mitos's recovery model treats the whole graph as disposable — ``rm
graph.sqlite`` / ``mitos rebuild`` rebuilds it from ``decisions.md`` — a corpus
living *inside* the graph would be silently destroyed on recovery. So it lives in
its **own** file, a sibling ``.mitos/telemetry.sqlite`` fenced off from the
rebuildable graph (P7 bulkhead): the swap/backup machinery in ``cutover.py`` only
ever touches ``graph.sqlite``-derived paths, so a different basename survives the
truth-rebuild untouched (CONF-D8; the T8 guarantee).

This module is deliberately **write-only** in v0.2: an append-only whole-row
writer and nothing more — no retention, decay, or pruning (P4 deferred), and no
consumer-facing read/query surface (the CONF-C1 ``edges ⋈ conflict_checks`` corpus
join belongs to a future vision). Every judged row lands **whole and uncapped** —
the longest, thorniest contradiction is exactly the example the future classifier
will most need (CONF-D8 verbatim-and-whole).

Tier 2 (logic/store): it reuses ``store.open_connection`` (the MI-8 connection
chokepoint) and ``migrations.run_migrations`` (the ladder primitive) rather than
re-implementing either, and defines its **own** ``TELEMETRY_MIGRATION_STEPS`` in a
separate ``user_version`` space so it never touches the graph's ladder. Nothing
here imports the sync/CLI/MCP orchestration layers; the sync surface (Phase 5b)
imports *this*, never the reverse — so ``telemetry -> {store, migrations, errors}``
stays acyclic.
"""

import os
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Tuple

from mitos.errors import DatabaseError
from mitos.migrations import MigrationStep, run_migrations
from mitos.store import open_connection

# --- Boundary-crossing row shapes ---------------------------------------------
#
# Two frozen dataclasses cross the persistence boundary (Python -> parameterized
# INSERT -> STRICT columns -> read-back). Each carries a ``to_params()`` producing
# a value tuple in a *fixed column order* that must match the module-level column
# tuples below, so a single INSERT is fully parameterized (P8) and column/param
# order cannot drift (the per-tuple length is lockstep-pinned by test).


@dataclass(frozen=True)
class ConflictCheckRow:
    """One judged candidate pair — the per-pair grain of the sensor's telemetry.

    Every field is stored **verbatim and uncapped**: the fed contexts
    (``judged_axiom``, both sides' ``rejected_paths``/``scope``) are the primary
    source of what the sensor actually judged (M8), not recoverable from a content
    hash, so nothing is truncated or normalized on the way in. The multi-valued
    ``rejected_paths``/``scope`` arrive **pre-serialized as TEXT** exactly as the
    judge saw them — this writer does not re-encode them (5b/3b own the fed-context
    rendering); no tuples are persisted.

    The additive, overcount-prone batch metrics (``token_*``/``elapsed_ms``) live
    on :class:`JudgmentBatch`, NOT here — copied per-row they would multiply true
    spend by the batch size under a naive ``SUM`` (RF-2/D1). Non-additive
    provenance (``model_alias``/``prompt_version``/``mitos_version``) stays per-row:
    it is never summed, so per-row duplication is harmless.

    Attributes:
        batch_id: Join key to :class:`JudgmentBatch` (a plain column, NOT an FK — a
            training label must outlive graph surgery, CONF-D8). Minted by 3b/5b.
        sync_run_id: The sync run this judgment fired in. Nullable for the deferred
            non-sync write-time path; the 1b/5b sync writer always supplies it.
        judged_axiom: The proposal's axiom text as fed to the judge, verbatim.
        proposal_rejected_paths: The proposal's ``rejected_paths`` fed context, or
            ``None`` when absent on the parsed entry.
        proposal_scope: The proposal's scope fed context, or ``None`` for
            unscoped/global (MI-9: absent = zero tags, never stored as ``""``).
        proposed_hash_if_any: The proposal's content hash. Nullable for the
            deferred write-time draft path; sync rows always supply it.
        candidate_slug: The candidate decision's slug — a debugging/citation handle
            stored verbatim (no casefolding: 1b does no slug comparison).
        candidate_hash: The candidate's M2 content hash (a plain column, not an FK).
        candidate_rejected_paths: The candidate's ``rejected_paths`` fed context.
            NOT NULL — M5 makes ``rejected_paths`` required on every decision.
        candidate_scope: The candidate's scope fed context, or ``None`` for
            unscoped/global (MI-9).
        tenable: The judge's verdict (stored INTEGER 0/1).
        confidence: The judge's raw self-reported confidence in ``[0, 1]``.
        surfaced: Whether the confidence gate fired (stored INTEGER 0/1).
        candidate_source: How the candidate was gathered (``"embedding_topk"`` in
            v0.2).
        model_alias: The family+tier alias of the judging model (never a raw model
            ID — P19; 3b/5b hand it over).
        prompt_version: The judgment prompt version that produced this row.
        mitos_version: The Mitos version that fired the judgment.
        rationale: The judge's rationale — NOT NULL, non-regenerable output (M8).
    """

    batch_id: str
    sync_run_id: Optional[str]
    judged_axiom: str
    proposal_rejected_paths: Optional[str]
    proposal_scope: Optional[str]
    proposed_hash_if_any: Optional[str]
    candidate_slug: str
    candidate_hash: str
    candidate_rejected_paths: str
    candidate_scope: Optional[str]
    tenable: bool
    confidence: float
    surfaced: bool
    candidate_source: str
    model_alias: str
    prompt_version: str
    mitos_version: str
    rationale: str

    def to_params(self, created_at: str) -> Tuple:
        """Produces the INSERT parameter tuple in ``_CONFLICT_CHECKS_COLUMNS`` order.

        The bool -> 0/1 conversion for ``tenable``/``surfaced`` is the primary type
        guard (STRICT SQLite has no BOOLEAN affinity); ``created_at`` is stamped
        here rather than read from the wall clock, so the whole batch shares one
        caller-supplied UTC ISO-8601 timestamp (MI-10, D5).

        Args:
            created_at: The batch-wide UTC ISO-8601 timestamp to stamp on this row.

        Returns:
            A value tuple positionally matching ``_CONFLICT_CHECKS_COLUMNS``.
        """
        return (
            self.batch_id,
            self.sync_run_id,
            self.judged_axiom,
            self.proposal_rejected_paths,
            self.proposal_scope,
            self.proposed_hash_if_any,
            self.candidate_slug,
            self.candidate_hash,
            self.candidate_rejected_paths,
            self.candidate_scope,
            int(self.tenable),
            self.confidence,
            int(self.surfaced),
            self.candidate_source,
            self.model_alias,
            self.prompt_version,
            self.mitos_version,
            self.rationale,
            created_at,
        )


@dataclass(frozen=True)
class JudgmentBatch:
    """The per-batch grain: the additive metrics of one batched judgment call.

    These facts describe the **single batched call**, not any individual candidate
    pair, so they live in their own side-table keyed on ``batch_id`` (PK). That
    makes exactly-once a *structural* property — one metrics row per batch — and
    keeps the append-only writer uniform (no "is-this-the-designated-row" branch
    over ``conflict_checks``). A naive ``SUM(token_input)`` over this table returns
    true spend, never ``N x`` the batch size (RF-2/D1).

    All five metrics are caller-supplied (3b measures them from the Anthropic
    response; 1b only stores) — the writer performs zero wall-clock or API reads.

    Attributes:
        batch_id: The PK; the same minted id string carried by every
            :class:`ConflictCheckRow` of the batch.
        token_input: Input tokens billed for the batched call.
        token_output: Output tokens billed for the batched call.
        token_cache_read: Cache-read tokens billed for the batched call.
        token_cache_creation: Cache-creation tokens billed for the batched call.
        elapsed_ms: Wall-clock latency of the batched call, in milliseconds.
    """

    batch_id: str
    token_input: int
    token_output: int
    token_cache_read: int
    token_cache_creation: int
    elapsed_ms: int

    def to_params(self) -> Tuple:
        """Produces the INSERT parameter tuple in ``_JUDGMENT_BATCHES_COLUMNS`` order.

        Returns:
            A value tuple positionally matching ``_JUDGMENT_BATCHES_COLUMNS``.
        """
        return (
            self.batch_id,
            self.token_input,
            self.token_output,
            self.token_cache_read,
            self.token_cache_creation,
            self.elapsed_ms,
        )


# --- Schema (migration ladder step 1) -----------------------------------------
#
# Telemetry owns a SEPARATE migration ladder in a SEPARATE file with its OWN
# ``user_version`` space — it must NEVER reuse the graph's ``MIGRATION_STEPS`` (its
# default arg is the live graph ladder; omitting ``steps`` would build
# ``nodes``/``edges`` inside ``telemetry.sqlite``). The DDL mirrors
# ``migrations._v1_schema``'s discipline: each CREATE TABLE is a SINGLE statement
# run via ``conn.execute`` (never ``executescript`` — it force-commits and would
# split the DDL out of the runner's atomic version-bump transaction); no
# ``DEFAULT CURRENT_TIMESTAMP`` (``created_at`` is application-supplied, MI-10); all
# identifiers are code-internal literals (P8); STRICT eliminates SQLite's permissive
# typing so type discipline is a property of the tables, not of a forgettable check.

# ``judgment_batches`` — the additive batch metrics (RF-2/D1). ``batch_id`` PRIMARY
# KEY makes "one metrics row per batch" structural. STRICT has no BOOLEAN affinity;
# every metric is a NOT NULL INTEGER.
_JUDGMENT_BATCHES_SCHEMA = """
    CREATE TABLE IF NOT EXISTS judgment_batches (
        batch_id TEXT NOT NULL,
        token_input INTEGER NOT NULL,
        token_output INTEGER NOT NULL,
        token_cache_read INTEGER NOT NULL,
        token_cache_creation INTEGER NOT NULL,
        elapsed_ms INTEGER NOT NULL,
        PRIMARY KEY (batch_id)
    ) STRICT;
"""

# ``conflict_checks`` — one row per judged candidate pair (§8 field contract). No
# PRIMARY KEY: this is an append-only log (the implicit rowid suffices), and a
# natural-key PK could wrongly reject a legitimate re-judgment. No row-size cap —
# the verbatim fed contexts land whole (CONF-D8). ``batch_id`` is a plain column,
# NOT an FK (a training label outlives graph surgery). ``tenable``/``surfaced`` are
# INTEGER (STRICT has no BOOLEAN affinity); the CHECKs are belt-and-suspenders over
# the dataclass bool->int guard (§14 latitude; literal, no interpolation).
_CONFLICT_CHECKS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS conflict_checks (
        batch_id TEXT NOT NULL,
        sync_run_id TEXT,
        judged_axiom TEXT NOT NULL,
        proposal_rejected_paths TEXT,
        proposal_scope TEXT,
        proposed_hash_if_any TEXT,
        candidate_slug TEXT NOT NULL,
        candidate_hash TEXT NOT NULL,
        candidate_rejected_paths TEXT NOT NULL,
        candidate_scope TEXT,
        tenable INTEGER NOT NULL,
        confidence REAL NOT NULL,
        surfaced INTEGER NOT NULL,
        candidate_source TEXT NOT NULL,
        model_alias TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        mitos_version TEXT NOT NULL,
        rationale TEXT NOT NULL,
        created_at TEXT NOT NULL,
        CHECK (tenable IN (0, 1)),
        CHECK (surfaced IN (0, 1)),
        CHECK (confidence >= 0.0 AND confidence <= 1.0)
    ) STRICT;
"""

# One statement per list element — ``conn.execute`` runs only the first statement
# in a string. Order is immaterial (no FK between the two tables).
_TELEMETRY_V1_STATEMENTS: List[str] = [
    _JUDGMENT_BATCHES_SCHEMA,
    _CONFLICT_CHECKS_SCHEMA,
]


def _conflict_checks_schema(conn: sqlite3.Connection) -> None:
    """Migration step 1: create the telemetry schema (``judgment_batches`` + ``conflict_checks``).

    Issues each ``CREATE TABLE ... STRICT`` via ``conn.execute`` (one per statement,
    never ``executescript`` — that force-commits and breaks the runner's atomic
    DDL+version-bump rollback). Does NOT touch ``user_version`` or manage the
    transaction — ``run_migrations`` owns both. Idempotent by the ``IF NOT EXISTS``
    backstop (MI-3 replay).

    Args:
        conn: An open, writable SQLite connection inside the runner's transaction
            (opened via ``store.open_connection``, MI-8).
    """
    for statement in _TELEMETRY_V1_STATEMENTS:
        conn.execute(statement)


# Telemetry's OWN ladder registry — a separate ``user_version`` space from the
# graph's ``migrations.MIGRATION_STEPS``. Authored fresh with step 1 inline; a
# future telemetry migration would ``.append((2, ...))`` (never rebind — a rebind is
# invisible to a def-time-bound default arg; the graph ladder learned this).
TELEMETRY_MIGRATION_STEPS: List[MigrationStep] = [(1, _conflict_checks_schema)]


# --- Parameterized INSERTs ----------------------------------------------------
#
# Column tuples are the single source of INSERT column order; ``to_params`` returns
# values in exactly these orders. The SQL is assembled from code-internal column
# LITERALS and a ``?`` placeholder per column — every VALUE binds via ``?`` (P8);
# the only interpolation is over the whitelisted identifiers/placeholder count,
# never user/LLM data (the documented P8 carve-out for code-internal identifiers).
_CONFLICT_CHECKS_COLUMNS: Tuple[str, ...] = (
    "batch_id",
    "sync_run_id",
    "judged_axiom",
    "proposal_rejected_paths",
    "proposal_scope",
    "proposed_hash_if_any",
    "candidate_slug",
    "candidate_hash",
    "candidate_rejected_paths",
    "candidate_scope",
    "tenable",
    "confidence",
    "surfaced",
    "candidate_source",
    "model_alias",
    "prompt_version",
    "mitos_version",
    "rationale",
    "created_at",
)

_JUDGMENT_BATCHES_COLUMNS: Tuple[str, ...] = (
    "batch_id",
    "token_input",
    "token_output",
    "token_cache_read",
    "token_cache_creation",
    "elapsed_ms",
)


def _insert_sql(table: str, columns: Tuple[str, ...]) -> str:
    """Builds a fully-parameterized INSERT over code-internal column literals.

    Args:
        table: The target table name (a code-internal literal).
        columns: The column names, in the order ``to_params`` emits values.

    Returns:
        An ``INSERT INTO table (cols...) VALUES (?, ?, ...)`` statement with one
        ``?`` per column — every value binds as a parameter (P8).
    """
    placeholders = ", ".join("?" for _ in columns)
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"


_INSERT_CONFLICT_CHECK_SQL = _insert_sql("conflict_checks", _CONFLICT_CHECKS_COLUMNS)
_INSERT_JUDGMENT_BATCH_SQL = _insert_sql("judgment_batches", _JUDGMENT_BATCHES_COLUMNS)


class TelemetryStore:
    """Append-only sibling store for the Conflict sensor's judged batches.

    Boots its own migration ladder on construction (mirroring
    ``GraphStore.__init__``'s boot-on-init shape) and is otherwise
    connection-stateless: every operation opens its own connection through the MI-8
    chokepoint and closes it, holding no long-lived handle. The store is
    write-only in v0.2 — one append-only writer, no query/consumer surface.
    """

    def __init__(self, telemetry_path: str) -> None:
        """Creates/opens the sibling telemetry DB and boots its ladder.

        Args:
            telemetry_path: Filesystem path to ``telemetry.sqlite`` (typically
                ``config.telemetry_path``). Its parent directory is created if
                absent — SQLite will not create a missing directory and would raise
                "unable to open database file" (mirrors ``GraphStore.__init__``).
        """
        self.telemetry_path = telemetry_path
        # Scaffold the parent dir before any connection touches the file — SQLite
        # does not create a missing ``.mitos/`` directory (store.py:538 idiom).
        db_dir = os.path.dirname(telemetry_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        # Bare boot: open the MI-8 chokepoint, run telemetry's OWN ladder, close.
        # No prototype guard, no pre-ladder snapshot — a single append-only store
        # with no risky rebuild needs none of the graph's ``_boot_migrations``
        # machinery (D2). ``run_migrations`` sets this connection to autocommit; it
        # is closed here, so the write path opens a fresh deferred-isolation
        # connection of its own (see ``record_judged_batch``).
        conn = open_connection(telemetry_path)
        try:
            run_migrations(conn, TELEMETRY_MIGRATION_STEPS)
        finally:
            conn.close()

    def record_judged_batch(
        self,
        batch: JudgmentBatch,
        rows: List[ConflictCheckRow],
        created_at: str,
    ) -> None:
        """Persists one judged batch — its metrics row + N candidate rows — atomically.

        Opens its own fresh connection (deferred isolation, so ``with conn:`` frames
        a real transaction — NOT the autocommit boot connection) and issues the one
        ``judgment_batches`` INSERT and the N ``conflict_checks`` INSERTs under a
        single ``with conn:`` block. All-or-nothing: any per-row failure rolls the
        whole batch back — no orphan candidate rows without their batch, no batch
        row without its candidates. Issues only INSERTs (append-only whole-row);
        never UPDATE/DELETE.

        ``created_at`` is the caller-supplied UTC ISO-8601 stamp shared by every
        candidate row of the batch (MI-10, D5) — the writer reads no wall clock.

        Args:
            batch: The per-batch additive metrics (one ``judgment_batches`` row).
            rows: The judged candidate pairs (N ``conflict_checks`` rows). All must
                share ``batch.batch_id`` — the writer stores what it is handed; the
                single transaction is what guarantees the join key always has its
                mate.
            created_at: The batch-wide UTC ISO-8601 timestamp stamped on every row.

        Raises:
            DatabaseError: If the batch cannot be persisted (a constraint violation,
                NOT NULL breach, or any other SQLite error) — the whole batch rolls
                back before this propagates under the CLI's ``MitosError`` boundary.
        """
        conn = open_connection(self.telemetry_path)
        try:
            with conn:
                conn.execute(_INSERT_JUDGMENT_BATCH_SQL, batch.to_params())
                for row in rows:
                    conn.execute(
                        _INSERT_CONFLICT_CHECK_SQL, row.to_params(created_at)
                    )
        except sqlite3.Error as e:
            # ``with conn:`` has already rolled the transaction back; wrap the raw
            # SQLite error in the located ``DatabaseError`` vector so it renders as a
            # one-line ``Error: …`` under the CLI's ``except MitosError`` boundary
            # (§13; mint no new error class).
            raise DatabaseError(f"Failed to persist judged batch: {e}") from e
        finally:
            conn.close()
