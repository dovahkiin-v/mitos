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
import shutil
import sqlite3
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

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
from mitos.parser import ParsedEntry, parse_file_reversed
from mitos.replay import commit_quarantine_fixpoint
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
class Casualty:
    """A corpus entry the rebuild replay could not commit — a punch-list item.

    Surfaced by ``mitos rebuild`` (the resilient, non-strict caller of
    :func:`rebuild_and_gate`): the entry stays in the buffer/markdown (the source of
    truth, M7/P6), it simply did not enter the rebuilt graph. The common class is a
    citation to a node that has since been superseded (``dangling_edge``) or never
    authored (``missing_target``). ``cutover`` (the strict caller) never returns
    these — a casualty there raises a :class:`~mitos.errors.CutoverError` instead.

    Attributes:
        slug: The entry's slug (or ``"<unknown>"`` for a pre-header failure).
        line_start: 1-based start line of the entry's section span.
        line_end: 1-based end line of the entry's section span.
        codes: The store failure codes (e.g. ``["dangling_edge"]``); empty for a
            non-``CommitError`` hard failure (validation / raw DB error).
        detail: The human-readable rejection reason(s).
    """

    slug: str
    line_start: int
    line_end: int
    codes: List[str]
    detail: str

    def to_dict(self) -> Dict[str, object]:
        """Serializes the casualty into a JSON-compatible dict.

        Returns:
            A dict for the ``--json`` rebuild report.
        """
        return {
            "slug": self.slug,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "codes": list(self.codes),
            "detail": self.detail,
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
        residual_casualties: Entries the replay could not commit (``mitos rebuild``'s
            resilient path). Always empty for the strict ``cutover`` caller, which
            raises on a casualty instead.
    """

    aside_db_path: str
    decisions_committed: int
    open_questions_committed: int
    reference_active_count: int
    reconstructed_active_count: int
    missing_cores: List[MissingCore]
    residual_casualties: List[Casualty] = field(default_factory=list)

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
            "residual_casualties": [c.to_dict() for c in self.residual_casualties],
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


def rebuild_and_gate(
    config: MitosConfig, *, aside_db_path: str, strict: bool = True
) -> RebuildResult:
    """Re-parses the corpus, replays it oldest-first, then gates against the old graph.

    The shared build-aside + completeness-gate engine behind both the one-time
    ``cutover`` (prototype→V1a) and the recurring ``mitos rebuild`` (re-commit the
    corpus through the current catalog). The live graph (``config.db_path``) is
    opened read-only for the gate and is otherwise never touched; all writes go to
    ``aside_db_path``.

    Pipeline (§6):

    1. Discard any stale build-aside file (+ its WAL sidecars) from a prior crashed
       run — idempotent retry, no manual step (P5).
    2. Parse every corpus file in collector mode, accumulating all format defects;
       any defect raises :class:`~mitos.errors.CutoverError` before a single commit.
    3. Replay per-entry, oldest-first, into a fresh graph, draining forward-refs via
       the per-entry quarantine + intra-sync fixpoint (so an acyclic chain converges
       in one pass regardless of authoring order). Entries that still cannot commit
       are **casualties**.
    4. Bound the embedding seed to the active set.
    5. Run the completeness gate against the live old graph (read-only) — a verdict,
       not an abort.

    **Per-caller policy on casualties (the shared-helper decision).** ``strict=True``
    (the ``cutover`` default) raises a ``CutoverError`` on the first casualty — a
    one-time prototype migration must halt on a genuine corpus defect. ``strict=False``
    (``mitos rebuild``) carries the casualties back on ``RebuildResult.residual_casualties``
    as a punch-list — an upgrade re-commit surfaces stale citations rather than aborting.

    Args:
        config: The active workspace config (supplies the corpus/archive/old-graph
            paths).
        aside_db_path: Where to build the rebuilt graph — a sibling of the live
            graph for an atomic swap (see :func:`default_aside_db_path`).
        strict: When ``True``, a replay casualty raises ``CutoverError``; when
            ``False``, casualties are returned on the verdict.

    Returns:
        A :class:`RebuildResult` verdict (committed counts, gate verdict, and — for
        ``strict=False`` — any residual casualties).

    Raises:
        CutoverError: On a parse-stage aggregate of format failures (always), or —
            when ``strict=True`` — the first replay casualty. The build-aside file is
            left for inspection; the live graph is untouched.
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

    # 3. Replay oldest-first into a fresh graph (a fresh file boots the ladder to the
    #    current head). Forward-refs drain via the per-entry quarantine + intra-sync
    #    fixpoint; entries that still cannot commit come back as casualties.
    store = GraphStore(aside_db_path)
    decisions_committed, oq_committed, casualties = replay_corpus_oldest_first(
        store, decision_entries, oq_entries
    )
    # Per-caller policy (the shared-helper decision): cutover (strict) halts on a
    # genuine defect — reproducing _commit_or_abort's contract on the first casualty;
    # rebuild (resilient) carries the casualties out as a punch-list.
    if strict and casualties:
        entry, exc = casualties[0]
        if isinstance(exc, CommitError):
            raise _cutover_error_for_commit(entry, exc) from exc
        raise CutoverError(
            f"Cutover replay aborted at entry '{entry.slug}' "
            f"(lines {entry.line_start}-{entry.line_end}): {exc}"
        ) from exc
    residual_casualties = [_casualty_from(entry, exc) for entry, exc in casualties]

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
        residual_casualties=residual_casualties,
    )


# --- atomic swap (Phase 7b) ----------------------------------------------------


def _clear_sidecars(base_path: str) -> None:
    """Removes a SQLite database's ``-wal`` / ``-shm`` sidecars, if present.

    Mirrors :func:`_discard_stale_aside`'s absent-is-a-no-op idiom but clears only
    the WAL sidecars, never the main file — :func:`perform_swap` calls this on both
    the aside (after its checkpoint folds every frame into the main file) and the
    destination (before the atomic rename, so no stale orphan ``-wal`` survives to
    be mis-applied to the freshly-swapped graph → ``SQLITE_CORRUPT``, R11).

    Args:
        base_path: The database path whose ``-wal`` / ``-shm`` sidecars to clear.
    """
    for suffix in ("-wal", "-shm"):
        try:
            os.remove(base_path + suffix)
        except FileNotFoundError:
            pass


def perform_swap(
    config: MitosConfig, aside_db_path: str, *, timestamp: str
) -> Optional[str]:
    """Atomically swaps the rebuilt build-aside graph into place; backs up the old.

    The destructive twin of :func:`rebuild_and_gate`: it takes the proven
    build-aside file 7a left on disk and makes it the live graph in a single
    crash-safe instant. The whole crash-safety story is *structural*, not
    disciplinary (Lesson 2 / P5 Ironclad): the only destructive primitive is one
    POSIX-atomic :func:`os.rename` within a single filesystem (guaranteed by 7a's
    sibling :func:`default_aside_db_path`), and every destructive-adjacent step
    precedes it. Because the old graph is **copied** (not moved) to the ``.bak``,
    ``config.db_path`` is never absent — at every instant it opens as either the
    intact old graph or the new V1a graph, so a crash at any point leaves a
    re-runnable workspace needing no manual restore (P5 Unplugged).

    Two WAL hazards are handled, both fatal if mishandled (R11, §2.1):

    * *New side:* the rebuilt graph is in WAL mode, so committed frames may still
      sit in ``<aside>-wal``. ``PRAGMA wal_checkpoint(TRUNCATE)`` folds them into
      the main file and empties the WAL **before** the rename — only then is the
      rebuilt main file self-contained. Skipping this silently loses data.
    * *Old/destination side:* the old graph's ``-wal`` / ``-shm`` must be gone from
      the destination **before** the new main lands there, else SQLite applies a
      stale orphan WAL to the new file on next open → ``SQLITE_CORRUPT``.

    After the swap ``config.db_path`` carries **no sidecars at all**; SQLite
    recreates fresh ``-wal`` / ``-shm`` on the next open.

    The old graph is *copied* (not checkpointed-then-copied): the 5a boot guard
    refuses a prototype graph read-write, the workspace is quiesced (an operator
    precondition), and the markdown corpus is the authoritative recovery source
    regardless (M7/P6) — so the ``.bak`` is a best-effort courtesy, and
    ``perform_swap`` never mutates the graph it is discarding (a clean bulkhead).

    Governing ADRs: ``cutover-build-aside-atomic-swap``,
    ``v1a-cutover-wipe-and-rebuild``,
    ``cutover-gate-defaults-abort-with-p6-operator-override``.

    Args:
        config: The active workspace config (supplies ``config.db_path``, the swap
            destination).
        aside_db_path: The rebuilt build-aside graph to swap in (a sibling of
            ``config.db_path`` for atomicity — see :func:`default_aside_db_path`).
        timestamp: A caller-supplied label for the ``.bak`` filename. Passed in
            (never wall-clocked here) so the helper stays deterministic for
            fixtures (PLANNING_NOTES: pin nothing to wall-clock).

    Returns:
        The ``<db_path>.bak_<timestamp>`` backup path, or ``None`` when there was
        no old graph to back up.

    Raises:
        CutoverError: If ``aside_db_path`` is absent — a defensive guard; the
            caller (:func:`~mitos.cli.cmd_cutover`) only invokes this after a
            successful :func:`rebuild_and_gate`.
    """
    # 1. Guard (defensive — cmd_cutover only calls this after a clean rebuild).
    if not os.path.exists(aside_db_path):
        raise CutoverError(
            "build-aside graph missing — the rebuild did not complete; "
            "re-run the cutover."
        )

    # 2. Checkpoint the aside DB (TRUNCATE): fold every WAL frame into the main
    #    file so the rebuilt graph is self-contained before it is renamed alone.
    #    A fresh write connection is the only open handle (GraphStore is
    #    connection-stateless; rebuild_and_gate left none open), so the TRUNCATE
    #    cannot be blocked by a concurrent reader.
    conn = open_connection(aside_db_path)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()

    # 3. Clear the aside's now-empty sidecars: the rebuilt main file is fully
    #    self-contained from here on.
    _clear_sidecars(aside_db_path)

    # 4. Back up the old graph by COPY (not move), so config.db_path is never
    #    absent (the entire crash-safety guarantee — do not reorder 4–6).
    bak_path: Optional[str] = None
    if os.path.exists(config.db_path):
        bak_path = config.db_path + ".bak_" + timestamp
        shutil.copy2(config.db_path, bak_path)

    # 5. Clear the destination's old sidecars BEFORE the rename — the R11
    #    orphan-WAL guard. A leftover prototype `-wal` beside the freshly-renamed
    #    main is applied on next open → SQLITE_CORRUPT.
    _clear_sidecars(config.db_path)

    # 6. The one atomic, destructive primitive: same-filesystem POSIX rename.
    os.rename(aside_db_path, config.db_path)

    # 7. config.db_path now holds the rebuilt V1a graph with no sidecars.
    return bak_path


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
        stream.extend(parse_file_reversed(archive_path, "decision", failures))
    stream.extend(parse_file_reversed(config.decisions_file, "decision", failures))
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
    return parse_file_reversed(config.questions_file, "open_question", failures)


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


def replay_corpus_oldest_first(
    store: GraphStore,
    decision_entries: List[ParsedEntry],
    oq_entries: List[ParsedEntry],
    *,
    embed_fn: Optional[Callable] = None,
) -> Tuple[int, int, List[Tuple[ParsedEntry, Exception]]]:
    """Replays both kind-streams oldest-first with quarantine + intra-sync fixpoint.

    The shared rebuild replay (``cutover`` and ``mitos rebuild``). Decisions then
    open questions, each a single ``commit_parsed_entry`` transaction (V1-D10). A
    ``CommitError`` quarantines the entry for the fixpoint to retry — so a forward-ref
    whose oldest-first target lands later in the pass converges here rather than
    aborting; a validation / raw-DB error is a non-retry-eligible hard failure
    recorded immediately. The fixpoint
    (:func:`mitos.replay.commit_quarantine_fixpoint`) drains the quarantine to
    convergence; whatever never commits is a **casualty**. This function never raises
    on a casualty — :func:`rebuild_and_gate` applies the strict-vs-resilient policy.

    Args:
        store: The fresh build-aside ``GraphStore`` (booted to the current head).
        decision_entries: Decision entries, oldest-first.
        oq_entries: Open-question entries, oldest-first.
        embed_fn: Optional per-commit embed callback. The rebuild leaves embeddings
            to a later sync (the queue is pruned to active afterward), so this is
            ``None`` by default.

    Returns:
        ``(decisions_committed, open_questions_committed, casualties)`` — the commit
        counts across both the main pass and the fixpoint, and the entries that never
        committed, each ``(entry, exc)``.
    """
    counts = {"decision": 0, "open_question": 0}
    quarantined: List[Tuple[ParsedEntry, str, Optional[CommitError]]] = []
    casualties: List[Tuple[ParsedEntry, Exception]] = []
    for entry in list(decision_entries) + list(oq_entries):
        try:
            delta = store.commit_parsed_entry(entry)
        except CommitError as exc:
            quarantined.append((entry, "", exc))
            continue
        except (ValidationError, DatabaseError) as exc:
            # No §5.2.2 envelope and not retry-eligible — an immediate casualty.
            casualties.append((entry, exc))
            continue
        counts[entry.kind] += 1
        if embed_fn is not None:
            embed_fn(delta, entry)

    def _count_commit(entry: ParsedEntry, _raw: str) -> None:
        counts[entry.kind] += 1

    _committed, _passes, residual = commit_quarantine_fixpoint(
        store, quarantined, embed_fn=embed_fn, on_commit=_count_commit
    )
    casualties.extend((entry, exc) for entry, _raw, exc in residual)
    return counts["decision"], counts["open_question"], casualties


def _casualty_from(entry: ParsedEntry, exc: Exception) -> Casualty:
    """Builds a :class:`Casualty` punch-list item from a rejected entry.

    Args:
        entry: The entry the store could not commit.
        exc: The rejection — a ``CommitError`` (carries the §5.2.2 envelope) or a
            validation / raw-DB error (no codes, message only).

    Returns:
        A serializable casualty carrying the slug, line span, failure codes, and
        human-readable detail.
    """
    failure = getattr(exc, "failure", None)
    items = failure.items if failure is not None else []
    codes = sorted({item.code for item in items})
    detail = "; ".join(item.message for item in items) or str(exc)
    return Casualty(
        slug=entry.slug or "<unknown>",
        line_start=entry.line_start,
        line_end=entry.line_end,
        codes=codes,
        detail=detail,
    )


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
        # supersedes/corrects are now List[str] (V1b multi-valued); take the first
        # declared kill-edge target for the diagnostic (a degenerate self-reference
        # carries a single citation in practice).
        _kill = entry.supersedes or entry.corrects
        target = (_kill[0] if _kill else "<unknown>").strip()
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

    The reference baseline depends on the live graph's shape. An **absent** old graph
    means nothing could be lost → vacuous pass. A **prototype** graph is read through
    the prototype-schema reader (the one-time ``cutover`` path). A current
    **V1a/V1b** graph is read through the store's own active-view
    (:func:`_read_current_graph_reference_cores` — the ``mitos rebuild`` path), so a
    decision active in the live graph but dropped by a re-commit (e.g. an entry whose
    now-stale citation the current catalog rejects) surfaces as a shortfall. An empty
    graph of either shape yields no reference cores → a correct vacuous pass.

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
        prototype = is_pre_v1a_schema(conn)
        reference_cores = (
            _read_prototype_reference_cores(conn) if prototype else None
        )
    finally:
        conn.close()
    if reference_cores is None:
        # A current (V1a/V1b) graph is the rebuild baseline — the prototype reader
        # cannot read its columns. Read the active set through the store's own
        # active-view (G5); an empty graph yields no cores → a correct vacuous pass.
        reference_cores = _read_current_graph_reference_cores(db_path)

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


def _read_current_graph_reference_cores(db_path: str) -> Dict[str, Dict[str, str]]:
    """Reads a current (V1a/V1b) graph's active cores as the rebuild gate baseline.

    The non-prototype twin of :func:`_read_prototype_reference_cores`: opens the live
    graph **read-only** (no migration boot) and reads its active set through the
    store's own ``get_active_decisions`` / ``get_open_questions`` — the single source
    of the active predicate (G5), never a re-encoded anti-join. Each active node is
    keyed on its already-minted slug-free id, directly comparable to the rebuild's.
    ``mitos rebuild`` compares these against the rebuilt active set; an id present
    here but absent there is a dropped node (e.g. an entry whose now-stale citation
    the current catalog rejects).

    Args:
        db_path: The live graph path.

    Returns:
        ``{node_id: {"kind", "slug", "axiom_excerpt"}}`` for each active node.
    """
    store = GraphStore(db_path, read_only=True)
    cores: Dict[str, Dict[str, str]] = {}
    for node in store.get_active_decisions():
        cores[node["id"]] = {
            "kind": "decision",
            "slug": node["slug"],
            "axiom_excerpt": (node.get("core_axiom") or "")[:_AXIOM_EXCERPT_LEN],
        }
    for node in store.get_open_questions():
        text = node.get("topic") or " ".join(node.get("questions_raised") or [])
        cores[node["id"]] = {
            "kind": "open_question",
            "slug": node["slug"],
            "axiom_excerpt": (text or "")[:_AXIOM_EXCERPT_LEN],
        }
    return cores


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
