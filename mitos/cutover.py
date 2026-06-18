"""Build-aside rebuilder & pre-swap completeness gate for the V1a cutover (R11).

V1a changed the canonical-core hash (the slug left the identity, M2/V1-D2), so the
live dogfood ``.mitos/graph.sqlite`` — minted under the prototype's slug-inclusive
hash — cannot be migrated in place. Because the graph is a derivative projection of
``decisions.md`` + ``questions.md`` + archives (M7/P6), a clean re-parse loses
nothing essential, so the cutover **tears the graph down and rebuilds it** from
that authoritative markdown.

This module is the *safety net* for that teardown. It does NOT touch the live
graph. It:

1. Re-parses the whole corpus and **replays it oldest-first** into a *separate*
   build-aside graph file (the live graph is never opened for writing).
2. Bounds the embedding seed (``pending_embeddings``) to the active set.
3. Runs a **reconstruction-completeness gate**: it reads the still-live old
   prototype graph's active-core set and proves every active core survived into
   the rebuild — surfacing any offender for the operator.

Two failure channels, deliberately distinct (§2.1):

* A **corpus defect** (a parse failure, a ``missing_target``, a Q5-convergence
  self-edge) raises :class:`~mitos.errors.CutoverError` and aborts — the operator
  fixes the markdown and re-runs. The build-aside graph is discarded; the live
  graph was never touched.
* A **completeness shortfall** (an active reference core absent from the
  reconstruction) is NOT raised — it is a verdict on :class:`RebuildResult` the
  operator may override (P6: the markdown is authoritative; a drop may be a
  deliberate purge).

Scope boundary (Phase 7a vs 7b): this module produces a callable, importable
surface (:func:`rebuild_and_gate`) and a verdict (:class:`RebuildResult`). It
**never** swaps the build-aside file into place, never touches WAL sidecars beyond
cleanly closing its own writes, never wipes the Qdrant collection, and never adds a
CLI verb — those are Phase 7b's, which consumes the verdict this module returns.

Governing ADRs: ``v1a-cutover-wipe-and-rebuild``,
``cutover-chronological-replay-by-file-position``,
``cutover-gate-defaults-abort-with-p6-operator-override``,
``cutover-bounds-embedding-seed-to-active``, ``cutover-build-aside-atomic-swap``.

Tier 3 (orchestration): composes ``parser`` (T2), ``store`` (T2), ``identity``
(T1), ``config`` (T1), ``errors`` (T1). Nothing here is imported by those modules.
"""

import glob
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from mitos.config import MitosConfig
from mitos.errors import (
    CommitError,
    CutoverError,
    DatabaseError,
    EntryFailure,
    STORE_CYCLE_VIOLATION,
    ValidationError,
)
from mitos.identity import compute_node_id
from mitos.migrations import is_pre_v1a_schema
from mitos.parser import ParsedEntry, parse_entry_stream
from mitos.store import GraphStore, open_connection

logger = logging.getLogger(__name__)

# Canonical archive filename, ``{year}-Q{quarter}.md`` (config
# ``rotation_archive_path_template``). Anchored so a stray ``notes.md`` in the
# archive dir is ignored, not mis-parsed as a quarter.
_ARCHIVE_FILENAME_RE = re.compile(r"^(\d{4})-Q([1-4])\.md$")

# The kill-edge value set, mirrored from ``store._KILL_EDGE_TYPES_SQL`` as a LOCAL
# constant (G2/G5). The old-graph active filter runs against the *prototype* edges
# schema (``from_id`` / ``to_id`` / ``type``), where V1a's ``_ACTIVE_VIEW_PREDICATE``
# column names (``target_id`` / ``edge_type``) do not apply — so this is the one
# sanctioned hand-written anti-join. Same value set, different columns; the values
# are bound, never f-stringed.
_OLD_GRAPH_KILL_EDGE_TYPES: Tuple[str, str] = ("supersedes", "corrects")

# How much of the core text to surface to the operator on a shortfall — enough to
# recognise the decision, not the whole axiom.
_AXIOM_EXCERPT_LEN: int = 120


@dataclass
class MissingCore:
    """A surfaced completeness-gate offender (P5 transparency).

    One active canonical core that was present in the old prototype graph but is
    absent from the rebuild. Surfaced with human-recognisable handles (``slug`` +
    ``axiom_excerpt``), not just the opaque hash, because the #1 risk (G1) is a
    *false* shortfall from prototype↔V1a normalization drift — the operator reviews
    these to decide whether the offender is genuine loss or a recompute artifact.

    Attributes:
        core_id: The recomputed slug-free node id absent from the reconstruction.
        kind: ``"decision"`` or ``"open_question"``.
        slug: A representative old-graph slug bearing this core (for the operator;
            under Q5 convergence several old slugs may share one core — this is one
            of them).
        axiom_excerpt: The first ~120 chars of the core text — human-recognisable.
    """

    core_id: str
    kind: str
    slug: str
    axiom_excerpt: str

    def to_dict(self) -> Dict[str, object]:
        """Serializes the offender into a JSON-compatible dict (strings only).

        Returns:
            A dict with plain string values (JSON-roundtrip-safe).
        """
        return {
            "core_id": self.core_id,
            "kind": self.kind,
            "slug": self.slug,
            "axiom_excerpt": self.axiom_excerpt,
        }


@dataclass
class RebuildResult:
    """The verdict of a clean :func:`rebuild_and_gate` run (7b consumes it).

    Returned only when the replay succeeded end-to-end (a corpus defect raises
    :class:`~mitos.errors.CutoverError` instead). ``gate_passed`` is *computed*
    from ``missing_cores`` (M3 spirit — state derived, never independently stored),
    so the verdict can never disagree with the offender list.

    Attributes:
        aside_db_path: The build-aside graph file written by this run (left on disk
            for 7b to swap; never the live ``config.db_path``).
        decisions_committed: Count of decision entries replayed into the rebuild.
        open_questions_committed: Count of open-question entries replayed.
        reference_active_count: Distinct active canonical cores read from the
            still-live old prototype graph (post-Q5-convergence dedup); the gate's
            reference baseline. ``0`` when there is no prototype graph to compare
            against (a fresh/absent/already-V1a ``config.db_path`` — a vacuous pass).
        reconstructed_active_count: Active node ids in the rebuild (both kinds).
        missing_cores: The shortfall offenders — active reference cores absent from
            the rebuild. Empty ⇒ the gate passed.
    """

    aside_db_path: str
    decisions_committed: int
    open_questions_committed: int
    reference_active_count: int
    reconstructed_active_count: int
    missing_cores: List[MissingCore]

    @property
    def gate_passed(self) -> bool:
        """Whether the completeness gate passed (no missing active cores).

        Returns:
            ``True`` iff ``missing_cores`` is empty.
        """
        return not self.missing_cores

    def to_dict(self) -> Dict[str, object]:
        """Serializes the verdict into a JSON-compatible dict (lists, never tuples).

        Returns:
            A dict 7b can render as the operator report; ``missing_cores`` is a
            list of offender dicts and ``gate_passed`` is included for convenience.
        """
        return {
            "aside_db_path": self.aside_db_path,
            "decisions_committed": self.decisions_committed,
            "open_questions_committed": self.open_questions_committed,
            "reference_active_count": self.reference_active_count,
            "reconstructed_active_count": self.reconstructed_active_count,
            "missing_cores": [mc.to_dict() for mc in self.missing_cores],
            "gate_passed": self.gate_passed,
        }


def default_aside_db_path(config: MitosConfig) -> str:
    """Returns the conventional build-aside graph path, a sibling of the live graph.

    The build-aside file MUST sit on the same filesystem as ``config.db_path`` so
    Phase 7b's ``os.rename`` swap is atomic (G3) — hence a sibling in
    ``config.mitos_dir``, never a ``/tmp`` path. 7a defines the convention; 7b's
    operator surface reuses it.

    Args:
        config: The active workspace config.

    Returns:
        ``<config.mitos_dir>/graph.sqlite.rebuild``.
    """
    return os.path.join(config.mitos_dir, "graph.sqlite.rebuild")


def rebuild_and_gate(config: MitosConfig, *, aside_db_path: str) -> RebuildResult:
    """Re-parses the corpus, replays it oldest-first, then gates against the old graph.

    The single importable entry point for the V1a cutover's build-aside +
    completeness-gate stage (Phase 7b's operator surface wraps this). The live
    graph (``config.db_path``) is opened read-only for the gate and is otherwise
    never touched; all writes go to ``aside_db_path``.

    Pipeline (§6):

    1. Discard any stale build-aside file (+ its WAL sidecars) from a prior crashed
       run — idempotent retry, no manual step (P5).
    2. Parse every corpus file in collector mode, accumulating all format defects;
       any defect raises :class:`~mitos.errors.CutoverError` before a single commit.
    3. Replay strictly per-entry, oldest-first, into a fresh V1a graph; the first
       referential/validation reject raises ``CutoverError``.
    4. Bound the embedding seed to the active set.
    5. Run the completeness gate against the live old graph (read-only) — a verdict,
       not an abort.

    Args:
        config: The active workspace config (supplies the corpus/archive/old-graph
            paths).
        aside_db_path: Where to build the rebuilt graph — a sibling of the live
            graph for an atomic 7b swap (see :func:`default_aside_db_path`).

    Returns:
        A :class:`RebuildResult` verdict (a clean replay, gate passed or shortfall).

    Raises:
        CutoverError: On a genuine corpus defect — a parse-stage aggregate of
            format failures, or the first referential/validation reject during
            replay. The build-aside file is left for inspection; the live graph is
            untouched.
    """
    # 1. Clean the slate (idempotent retry, P5): a prior crashed run leaves an
    #    orphan aside file; discard it (+ sidecars) so this run starts fresh.
    _discard_stale_aside(aside_db_path)

    # 2. Load + order each kind-stream oldest-first, collecting every format defect
    #    across ALL files before deciding to abort (the operator fixes them at once).
    failures: List[EntryFailure] = []
    decision_entries = _load_decision_stream(config, failures)
    oq_entries = _load_oq_stream(config, failures)
    if failures:
        raise CutoverError(_format_parse_aggregate_message(failures), failure=failures)

    # 3. Replay oldest-first into a fresh V1a graph (a fresh file boots the ladder
    #    to user_version 1; an empty/absent file is never a prototype).
    store = GraphStore(aside_db_path)
    decisions_committed, oq_committed = _replay_oldest_first(
        store, decision_entries, oq_entries
    )

    # 4. Bound the embedding seed to the active set. Every commit self-enqueued one
    #    pending_embeddings row (5c), so the queue holds the whole corpus incl. dead
    #    nodes; prune it to the store's own active set (G5 — never a re-encoded
    #    predicate). Reused as the gate's reconstructed set (step 5).
    reconstructed_ids = _reconstructed_active_ids(store)
    _prune_embedding_queue_to_active(aside_db_path, reconstructed_ids)

    # 5. Completeness gate against the still-live OLD graph (read-only). A shortfall
    #    is a verdict, never an abort (P6 operator override is 7b's flag; 7a simply
    #    does not raise).
    missing_cores, reference_active_count = check_reconstruction_completeness(
        config, reconstructed_ids
    )

    # 6. Close the aside store cleanly before returning. Every helper opens and
    #    closes its own connection (no handle lingers on ``store``), so the on-disk
    #    main+WAL state is already consistent and recoverable for 7b's
    #    checkpoint-before-swap. Clearing the WAL sidecar itself is 7b's job — 7a
    #    must not overstep it.
    return RebuildResult(
        aside_db_path=aside_db_path,
        decisions_committed=decisions_committed,
        open_questions_committed=oq_committed,
        reference_active_count=reference_active_count,
        reconstructed_active_count=len(reconstructed_ids),
        missing_cores=missing_cores,
    )


# --- stream loading & ordering -------------------------------------------------


def _discard_stale_aside(aside_db_path: str) -> None:
    """Removes a stale build-aside graph file and its WAL/SHM sidecars, if present.

    A prior crashed run leaves an orphan; the next run must discard it with no
    manual step (idempotent retry, P5). Absent files are a no-op.

    Args:
        aside_db_path: The build-aside graph path to clear.
    """
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(aside_db_path + suffix)
        except FileNotFoundError:
            pass


def _read_text_or_none(path: str) -> Optional[str]:
    """Reads a UTF-8 file, returning ``None`` if it does not exist.

    A missing buffer or archive is a no-op stream, never a crash — an absent
    ``questions.md`` is the live-corpus reality today (§6).

    Args:
        path: The file to read.

    Returns:
        The file text, or ``None`` if the file is absent.
    """
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _archive_files_oldest_first(archive_dir: str) -> List[str]:
    """Returns the ``{year}-Q{quarter}.md`` archive files, oldest quarter first.

    Globs ``archive_dir`` for ``*.md``, keeps only canonical
    ``{year}-Q{quarter}.md`` names, and sorts by ``(year, quarter)`` ascending so
    that reversing each file at replay lands the corpus globally oldest-first. An
    absent ``archive_dir`` yields an empty list (``glob`` over a missing dir is
    empty) — a workspace may have no archives yet, which is healthy, not an error.

    Args:
        archive_dir: The workspace archive directory (``config.archive_dir``).

    Returns:
        Archive file paths, sorted oldest-quarter-first.
    """
    matches: List[Tuple[int, int, str]] = []
    for path in glob.glob(os.path.join(archive_dir, "*.md")):
        m = _ARCHIVE_FILENAME_RE.match(os.path.basename(path))
        if m:
            matches.append((int(m.group(1)), int(m.group(2)), path))
    matches.sort(key=lambda t: (t[0], t[1]))
    return [path for _, _, path in matches]


def _parse_file_reversed(
    path: str, kind: str, failures: List[EntryFailure]
) -> List[ParsedEntry]:
    """Parses one corpus file in collector mode and reverses it to oldest-first.

    Each corpus file is authored **newest-first** (the ``BEGIN ENTRIES … newest
    first`` convention), so reversing the parsed list yields oldest-first *within*
    the file. Collector mode (``failures`` supplied) isolates malformed entries
    into ``failures`` instead of raising, so all defects across all files
    aggregate before the caller decides to abort.

    Args:
        path: The corpus file to parse (absent → empty stream).
        kind: ``"decision"`` or ``"open_question"`` (caller-declared, V1-D8).
        failures: The shared collector for malformed-entry envelopes.

    Returns:
        The well-formed entries, oldest-first within this file.
    """
    text = _read_text_or_none(path)
    if text is None:
        return []
    entries = parse_entry_stream(text, kind, source_path=path, failures=failures)
    entries.reverse()
    return entries


def _load_decision_stream(
    config: MitosConfig, failures: List[EntryFailure]
) -> List[ParsedEntry]:
    """Builds the globally oldest-first decision stream (archives, then the buffer).

    Concatenates ``[oldest_archive … newest_archive, decisions_file]`` with each
    file reversed to oldest-first. Archives hold entries rotated out *before* the
    buffer's, so the buffer's entries are the newest and come last — a globally
    oldest-first order in which every kill-edge target is authored before its
    superseder (K4: oldest-first replay *is* a valid topological order).

    Args:
        config: The active workspace config.
        failures: The shared parse-failure collector.

    Returns:
        The decision entries, globally oldest-first.
    """
    stream: List[ParsedEntry] = []
    for archive_path in _archive_files_oldest_first(config.archive_dir):
        stream.extend(_parse_file_reversed(archive_path, "decision", failures))
    stream.extend(_parse_file_reversed(config.decisions_file, "decision", failures))
    return stream


def _load_oq_stream(
    config: MitosConfig, failures: List[EntryFailure]
) -> List[ParsedEntry]:
    """Builds the oldest-first open-question stream from ``questions.md``.

    There is no open-question archive convention yet and the live corpus carries
    zero open questions, so the questions buffer is the whole OQ stream today; an
    absent file is a no-op (empty stream). The contract names the file so a future
    OQ corpus is replayed, not silently skipped (§6).

    Args:
        config: The active workspace config.
        failures: The shared parse-failure collector.

    Returns:
        The open-question entries, oldest-first (empty when ``questions.md`` is
        absent).
    """
    return _parse_file_reversed(config.questions_file, "open_question", failures)


def _format_parse_aggregate_message(failures: List[EntryFailure]) -> str:
    """Builds the operator-facing message for a parse-stage abort.

    Lists up to the first five offenders with their slug and located line range
    (P3 vector error — the operator fixes the markdown and re-runs).

    Args:
        failures: The accumulated parse-failure envelopes (always non-empty here).

    Returns:
        A calm, multi-line message naming the offenders.
    """
    n = len(failures)
    located = []
    for f in failures[:5]:
        slug = f.slug if f.slug is not None else "<no slug>"
        loc = f.source_path or "?"
        located.append(f"  - '{slug}' ({loc} lines {f.line_start}-{f.line_end})")
    more = "" if n <= 5 else f"\n  …and {n - 5} more."
    plural = "entry" if n == 1 else "entries"
    return (
        f"Cutover aborted: {n} malformed {plural} in the corpus — nothing was "
        f"committed. Fix the format defect(s) and re-run:\n" + "\n".join(located) + more
    )


# --- replay --------------------------------------------------------------------


def _replay_oldest_first(
    store: GraphStore,
    decision_entries: List[ParsedEntry],
    oq_entries: List[ParsedEntry],
) -> Tuple[int, int]:
    """Replays both kind-streams strictly per-entry into the build-aside graph.

    Decisions then open questions (the two never interleave — no committed edge
    crosses kinds, so their relative order is irrelevant). Each ``commit_parsed_entry``
    is its own atomic transaction (V1-D10); the **first** reject aborts the whole
    rebuild with a :class:`~mitos.errors.CutoverError` surfacing that entry.

    Args:
        store: The fresh build-aside ``GraphStore`` (booted to the V1a head).
        decision_entries: Decision entries, oldest-first.
        oq_entries: Open-question entries, oldest-first.

    Returns:
        ``(decisions_committed, open_questions_committed)``.

    Raises:
        CutoverError: On the first referential/validation reject (the entry that
            failed, its line range, and the store code are surfaced).
    """
    decisions_committed = 0
    for entry in decision_entries:
        _commit_or_abort(store, entry)
        decisions_committed += 1
    oq_committed = 0
    for entry in oq_entries:
        _commit_or_abort(store, entry)
        oq_committed += 1
    return decisions_committed, oq_committed


def _commit_or_abort(store: GraphStore, entry: ParsedEntry) -> None:
    """Commits one entry, translating any store reject into a ``CutoverError`` abort.

    A corpus defect surfaced by the store (a referential violation, or a
    bypassed-parser validation/DB error) is never skipped (G7): it aborts the
    rebuild so the operator fixes the corpus and re-runs against a still-untouched
    live graph (R11).

    Args:
        store: The build-aside store.
        entry: The entry to commit.

    Raises:
        CutoverError: On any commit reject, carrying the store's failure envelope
            when one exists.
    """
    try:
        store.commit_parsed_entry(entry)
    except CommitError as exc:
        raise _cutover_error_for_commit(entry, exc) from exc
    except (ValidationError, DatabaseError) as exc:
        # No structured envelope on these (a bypassed-parser empty core, or a raw
        # SQLite failure) — surface the entry locus and the underlying message.
        raise CutoverError(
            f"Cutover replay aborted at entry '{entry.slug}' "
            f"(lines {entry.line_start}-{entry.line_end}): {exc}"
        ) from exc


def _cutover_error_for_commit(entry: ParsedEntry, exc: CommitError) -> CutoverError:
    """Builds the ``CutoverError`` for a store referential reject during replay.

    A ``cycle_violation`` is the expected one-time Q5 artifact (G4): the corpus
    declares a kill-edge between two entries that now share one canonical core, so
    the edge degenerates to a self-reference. That case gets explicit cleanup
    guidance (drop the degenerate line); every other code gets a located generic
    surface. The store's failure envelope rides on the ``CutoverError`` for 7b.

    Args:
        entry: The entry the store rejected.
        exc: The ``CommitError`` raised by ``commit_parsed_entry``.

    Returns:
        A ``CutoverError`` carrying ``exc.failure``.
    """
    codes = {item.code for item in exc.failure.items} if exc.failure else set()
    if STORE_CYCLE_VIOLATION in codes:
        target = (entry.supersedes or entry.corrects or "<unknown>").strip()
        if len(target) >= 2 and target.startswith("[") and target.endswith("]"):
            target = target[1:-1].strip()
        msg = (
            f"Cutover replay aborted at entry '{entry.slug}' "
            f"(lines {entry.line_start}-{entry.line_end}): a prototype kill-edge "
            f"between two now-converged cores '{entry.slug}' ↔ '{target}'. Under the "
            f"V1a slug-free hash these are one decision (Q5 convergence), so the edge "
            f"degenerates to a self-reference. Drop the now-degenerate "
            f"Supersedes:/Corrects: line from '{entry.slug}' (they are one decision) "
            f"and re-run."
        )
    else:
        code_str = ", ".join(sorted(codes)) if codes else "referential violation"
        msg = (
            f"Cutover replay aborted at entry '{entry.slug}' "
            f"(lines {entry.line_start}-{entry.line_end}): the store rejected it "
            f"({code_str}). {exc}"
        )
    return CutoverError(msg, failure=exc.failure)


# --- embedding-seed bounding ---------------------------------------------------


def _reconstructed_active_ids(store: GraphStore) -> Set[str]:
    """Returns the active node id set of the rebuilt graph (both kinds).

    Reads through the store's own active-view read methods — the single source of
    the V1a active predicate (G5, Lesson 14). Never re-encodes the kill-edge
    anti-join on the new graph.

    Args:
        store: The build-aside store.

    Returns:
        The active decision + open-question node ids.
    """
    active: Set[str] = {node["id"] for node in store.get_active_decisions()}
    active.update(node["id"] for node in store.get_open_questions())
    return active


def _prune_embedding_queue_to_active(
    aside_db_path: str, active_ids: Set[str]
) -> None:
    """Prunes ``pending_embeddings`` to exactly the active id set (both kinds).

    Every commit self-enqueued one ``pending_embeddings`` row unconditionally
    (5c), so the queue holds the whole corpus including dead/superseded nodes; the
    embedding seed must be bounded to the active set (``cutover-bounds-embedding-
    seed-to-active``).

    Uses a connection-local temp table of the active ids rather than a bound
    ``NOT IN (?, ?, …)`` list: at the Dyson horizon (tens of thousands of active
    decisions) a bound list would exceed SQLite's host-parameter limit (~32k),
    while the temp-table anti-join has no such ceiling (P11/P17, the Dyson check).
    The active ids still come from the store's own read methods (G5); this changes
    only *how* they bind, not the active definition. An empty active set correctly
    empties the whole queue (``NOT IN`` over an empty subquery matches every row,
    never the ``NOT IN ()`` syntax error).

    Args:
        aside_db_path: The build-aside graph (writable).
        active_ids: The active node ids to keep.
    """
    conn = open_connection(aside_db_path)
    try:
        # TEMP table is connection-scoped and dropped on close; no transaction
        # gymnastics needed — populate, delete, commit the DELETE, close.
        conn.execute("CREATE TEMP TABLE _active_ids (id TEXT PRIMARY KEY NOT NULL)")
        conn.executemany(
            "INSERT OR IGNORE INTO _active_ids (id) VALUES (?)",
            [(node_id,) for node_id in active_ids],
        )
        conn.execute(
            "DELETE FROM pending_embeddings "
            "WHERE node_id NOT IN (SELECT id FROM _active_ids)"
        )
        conn.commit()
    finally:
        conn.close()


# --- completeness gate ---------------------------------------------------------


def check_reconstruction_completeness(
    config: MitosConfig, reconstructed_ids: Set[str]
) -> Tuple[List[MissingCore], int]:
    """Proves every active core in the old graph survived into the reconstruction.

    Opens the still-live old graph **read-only** and reads its active-core set
    under the *same* kill-edge active lens the reconstructed set uses (a
    derivative-to-derivative regression baseline). Each active old core is
    recomputed to its slug-free :func:`~mitos.identity.compute_node_id` id and
    compared against ``reconstructed_ids``; any reference core absent from the
    rebuild is surfaced as a :class:`MissingCore`. **Never raises** — a shortfall
    is a verdict the operator may override (P6).

    Two robustness guards run first (G7): an **absent** old graph means no
    reference baseline (nothing could be lost → vacuous pass); an old graph that is
    **not a prototype** (already V1a-or-later, or empty) means no prototype core
    columns to read → a clear vacuous-pass verdict rather than a cryptic
    ``no such column: core_axiom``.

    The comparison keys on the **canonical core**, not the raw slug, so Q5
    convergence (two old slugs sharing one core) dedups silently — both old slugs
    recompute to the one core id that is present once in the rebuild.

    Args:
        config: The active workspace config (``config.db_path`` is the old graph).
        reconstructed_ids: The rebuilt graph's active node ids (both kinds).

    Returns:
        ``(missing_cores, reference_active_count)`` — the shortfall offenders (empty
        ⇒ gate passed) and the count of distinct active reference cores.
    """
    db_path = config.db_path
    if not os.path.exists(db_path):
        logger.info(
            "Cutover gate: no live graph at %s — vacuous pass (no reference "
            "baseline, nothing could be lost).",
            db_path,
        )
        return [], 0

    conn = open_connection(db_path, read_only=True)
    try:
        if not is_pre_v1a_schema(conn):
            logger.info(
                "Cutover gate: the live graph at %s is not a prototype graph "
                "(already V1a-or-later, or empty) — no prototype reference to "
                "compare against; nothing to cut over.",
                db_path,
            )
            return [], 0
        reference_cores = _read_prototype_reference_cores(conn)
    finally:
        conn.close()

    missing_ids = set(reference_cores) - reconstructed_ids
    missing_cores = [
        MissingCore(
            core_id=core_id,
            kind=reference_cores[core_id]["kind"],
            slug=reference_cores[core_id]["slug"],
            axiom_excerpt=reference_cores[core_id]["axiom_excerpt"],
        )
        # Deterministic order (slug, then id) for a stable operator report.
        for core_id in sorted(
            missing_ids, key=lambda c: (reference_cores[c]["slug"], c)
        )
    ]
    return missing_cores, len(reference_cores)


def _read_prototype_reference_cores(
    conn: sqlite3.Connection,
) -> Dict[str, Dict[str, str]]:
    """Reads the old prototype graph's active cores, keyed on the recomputed id.

    Raw SQL against the **prototype** schema (``core_axiom`` / ``mechanisms`` /
    ``questions_raised`` columns; ``edges.from_id`` / ``to_id`` / ``type``) — never
    the 5d read API, which assumes the V1a columns (G2). The active filter is the
    1-hop ``NOT EXISTS`` kill-edge anti-join, mirroring V1a's ``_ACTIVE_VIEW_PREDICATE``
    exactly (a node is inactive iff it is the target of any incoming
    ``supersedes`` / ``corrects`` edge). Each active row is recomputed to its
    slug-free core id via :func:`~mitos.identity.compute_node_id` (the same identity
    path the rebuild minted node ids through, so the keys are structurally
    comparable); Q5 convergence dedups here (first slug wins as the representative).

    For an open question the prototype has no ``topic`` column — its core text lives
    in the general ``core_axiom`` column — so ``topic`` recomputes from
    ``core_axiom`` (G6). The live corpus has zero open questions, so this branch is
    fixture-only.

    Args:
        conn: A read-only connection to the old prototype graph.

    Returns:
        ``{core_id: {"kind", "slug", "axiom_excerpt"}}`` for each active core.
    """
    placeholders = ",".join("?" for _ in _OLD_GRAPH_KILL_EDGE_TYPES)
    sql = (
        "SELECT id, slug, kind, core_axiom, mechanisms, questions_raised "
        "FROM nodes "
        "WHERE NOT EXISTS (SELECT 1 FROM edges "
        f"WHERE edges.to_id = nodes.id AND edges.type IN ({placeholders}))"
    )
    reference_cores: Dict[str, Dict[str, str]] = {}
    for row in conn.execute(sql, _OLD_GRAPH_KILL_EDGE_TYPES).fetchall():
        kind = row["kind"]
        core_axiom = row["core_axiom"] or ""
        if kind == "decision":
            core_id = compute_node_id(
                kind="decision",
                axiom=core_axiom,
                mechanism_refs=json.loads(row["mechanisms"] or "[]"),
            )
        elif kind == "open_question":
            core_id = compute_node_id(
                kind="open_question",
                topic=core_axiom,
                questions_raised=json.loads(row["questions_raised"] or "[]"),
            )
        else:
            # The prototype CHECK constrains kind to the two known values; guard
            # defensively rather than mis-key an unknown row.
            continue
        if core_id not in reference_cores:
            reference_cores[core_id] = {
                "kind": kind,
                "slug": row["slug"],
                "axiom_excerpt": core_axiom[:_AXIOM_EXCERPT_LEN],
            }
    return reference_cores
