"""Corpus-replay primitive: per-entry quarantine + intra-sync fixpoint.

The shared engine behind two callers that replay already-parsed entries into a
:class:`~mitos.store.GraphStore`:

- ``MitosSyncManager._commit_quarantine_fixpoint`` (steady-state sync, Phase 4b) —
  drains the entries the main sync pass quarantined.
- ``cutover.replay_corpus_oldest_first`` (the cutover / ``mitos rebuild`` rebuild
  engine) — replays the whole corpus oldest-first.

Both want the same order-independent convergence: re-attempt the quarantined set
until a pass commits nothing new, so any acyclic cross-file forward-ref chain
lands in ONE pass while a genuine defect (a never-authored target, a true
mutual-reference cycle) surfaces as a residual. Each retry is the *same* isolated
``commit_parsed_entry`` transaction (V1-D10) — no batching, no ordering. Keeping
the loop here is the single source of truth the "shared helper, per-caller policy"
decision committed to: callers differ only in how they treat the residual (sync
reports it; cutover raises; rebuild surfaces it as a casualty list).
"""

from typing import Callable, List, Optional, Tuple

from mitos.errors import CommitError
from mitos.parser import ParsedEntry
from mitos.store import CommitDelta, GraphStore

# One quarantined entry: the parsed entry, its decisions-snapshot raw text ("" when
# the caller does not rotate — e.g. an OQ, or the cutover/rebuild replay), and its
# latest ``CommitError``. ``None`` is accepted as the initial failure for an entry
# that has not been attempted yet (it is replaced on the first failed attempt).
QuarantinedEntry = Tuple[ParsedEntry, str, Optional[CommitError]]


def commit_quarantine_fixpoint(
    store: GraphStore,
    quarantined: List[QuarantinedEntry],
    *,
    embed_fn: Optional[Callable[[CommitDelta, ParsedEntry], object]] = None,
    on_commit: Optional[Callable[[ParsedEntry, str], None]] = None,
) -> Tuple[int, int, List[Tuple[ParsedEntry, str, CommitError]]]:
    """Drains a per-entry quarantine set to a fixpoint.

    Re-attempts ``quarantined`` in repeated passes until a pass commits nothing new.
    The progress metric is *a commit succeeded this pass*, not *the set is
    non-empty*: a committed entry leaves the set and is never revisited (none enter
    after the initial set), so the set is monotonically non-increasing and the loop
    stops on the first zero-progress pass. A true mutual-reference cycle (or a
    never-authored target) makes zero progress → one no-progress pass → exit → the
    residual is returned for the caller to surface. Worst case is O(k) passes for a
    depth-k forward-ref chain.

    Each retry is the **same** isolated ``commit_parsed_entry(entry)`` call on the
    **same** already-prepared entry: a failed retry rolls back wholly (the
    ``CommitError`` is raised inside ``commit_parsed_entry``'s ``with conn:``), so a
    clean DB plus any targets committed earlier in the same pass is what the next
    retry sees — the safety property that makes re-attempt sound.

    Args:
        store: The graph store to commit into.
        quarantined: The entries to drain, each ``(entry, raw, latest_error)``.
        embed_fn: Optional callback invoked ``embed_fn(delta, entry)`` on each
            successful commit (sync embeds; the rebuild leaves embeddings to a
            later sync, so it passes ``None``).
        on_commit: Optional callback invoked ``on_commit(entry, raw)`` on each
            successful commit (sync records the block for rotation; the rebuild
            counts committed entries per kind).

    Returns:
        ``(committed, passes, residual)`` — the number of entries committed across
        all passes, the number of passes run, and the residual entries that never
        committed (each carrying its latest ``CommitError``); ``residual`` is ``[]``
        when everything converged.
    """
    if not quarantined:
        return 0, 0, []

    pending: List[QuarantinedEntry] = list(quarantined)
    committed = 0
    passes = 0
    while pending:
        passes += 1
        progressed = False
        still_pending: List[QuarantinedEntry] = []
        for entry, raw, _exc in pending:
            try:
                delta = store.commit_parsed_entry(entry)
            except CommitError as new_exc:
                # Still blocked — keep it (carrying its LATEST failure) for the next
                # pass. The whole entry rolled back inside commit_parsed_entry's
                # ``with conn:``, so the DB stays consistent for the other retries.
                still_pending.append((entry, raw, new_exc))
                continue
            progressed = True
            committed += 1
            print(f"Committed node: {entry.slug} ✓")
            if embed_fn is not None:
                embed_fn(delta, entry)
            if on_commit is not None:
                on_commit(entry, raw)
        pending = still_pending
        if not progressed:
            # A full pass committed nothing — no remaining entry can ever make
            # progress (the set only shrinks), so stop and leave them as residual.
            break

    # ``_exc`` is ``None`` only for an entry never attempted; every residual entry
    # was attempted at least once (pass 1), so its third element is a real
    # ``CommitError`` by here. Narrow the type for callers.
    residual: List[Tuple[ParsedEntry, str, CommitError]] = [
        (entry, raw, exc) for entry, raw, exc in pending if exc is not None
    ]
    return committed, passes, residual
