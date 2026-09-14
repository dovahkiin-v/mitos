"""Settledness — which of the buffer's oldest entries rotation may move, and why it stops.

``decisions.md`` is the working set: entries still live, recently touched, or not yet
in agreement with the graph stay where every repair path can reach them. An entry is
**settled** when it is committed (a node exists for its content-hash id), quiet
(``now - node.updated_at > lag``) and not diverged (``divergence.entry_divergence``
reports nothing) — vision D3. *Committed* is load-bearing: the divergence leaf skips an
entry with no node, so a never-committed draft reads as undiverged. *Not diverged*
reads the pure leaf, never the corpus fold, which builds a fresh lock on the buffer's
path and would deadlock inside rotation's hold (ADR
``write-path-lifecycle-predicates-read-the-divergence-leaf-never-the-fold``).

The work is split by price (vision D2). The trigger is a heading count over the whole
buffer — the parser's own section rule, no field tokenized, no graph read. Only when
the count reaches the threshold is the expensive predicate evaluated, and only over a
window of the last ``window`` entry blocks: the buffer's **tail**, because both tool
writers prepend, so the tail is the oldest (ADR
``the-buffers-oldest-first-window-is-its-tail``). The bound is on entries examined,
never on entries found (ADR
``settledness-scan-is-bounded-on-entries-examined-not-on-eligible-found``).

The batch is the **contiguous** settled run counted from the tail, stopping at the
first entry that is not settled. It is never the settled subset of the window: an
archive has to be an exact prefix of the buffer's history, or ``mitos rebuild`` replays
a supersede before the amend it retires and refuses the swap. The price is that one
stuck tail entry stalls the drain, which is the safe direction — a stalled entry is in
the buffer where ``status`` names its heal, while a wrongly rotated one stays wrongly
placed.

Nothing here is stored and nothing prints. The buffer text arrives from rotation's one
read of the live buffer; the graph arrives as an object with ``get_node`` and
``get_outgoing_edges`` (a ``GraphStore`` satisfies it); the clock arrives as a stamp.

Tier 2: stdlib plus ``markers``, ``parser``, ``identity``, ``divergence`` and
``rotation``. No store, sync, lock or SDK — a subprocess probe pins it.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, List, Optional, Tuple, Union

from mitos import markers
from mitos.divergence import entry_divergence, has_divergence
from mitos.errors import EntryFailure
from mitos.identity import compute_node_id
from mitos.parser import ParsedEntry, parse_entry_stream
from mitos.rotation import RotationBlock

# The most entries one rotation evaluates, and so the most it moves, per acquisition of
# the buffer lock. A contract derived from the opera §7.3 lock window, not a tuning
# knob, so it is a constant and not a config key. Calibration (surface-entropy 3b
# IMPLEMENTATION_NOTES): the hold has a floor W does not control — the whole-file
# archive and buffer rewrites, ~36 ms p95 on today's 1.5 MB buffer, ~21 ms once it
# drains — and W adds ~0.5 ms per entry of graph reads. 20 is the largest window whose
# draining p95 stays inside the ~50 ms window at today's size; the 60 s lock timeout
# is three orders of magnitude away.
ROTATION_WINDOW_ENTRIES = 20

# Why the batch stopped, as data. Each caller words its own reason.
UNCOMMITTED = "uncommitted"
RECENT = "recent"
DIVERGED = "diverged"
UNSTAMPED = "unstamped"
UNPARSEABLE = "unparseable"
WINDOW_MISMATCH = "window_mismatch"

# Opens the window text, so that a later `BEGIN ENTRIES` inside an entry body is not
# the first sentinel and cannot swallow the entries above it.
_WINDOW_SENTINEL = f"<!-- {markers.ENTRIES_SENTINEL} -->\n"


@dataclass
class Selection:
    """What one settledness evaluation chose. Runtime-only; never persisted.

    Attributes:
        blocks: The batch, oldest first — commit order, as the archive writer takes it.
        buffered: Entry headings after the buffer's sentinel.
        examined: Entries the predicate evaluated, never more than the window.
        stopped_at: ``(slug, reason)`` of the first unsettled tail entry, or ``None``
            when nothing stopped the walk. ``slug`` is ``""`` when there is none to
            name (a slug-less parse failure, a window mismatch).
    """

    blocks: List[RotationBlock] = field(default_factory=list)
    buffered: int = 0
    examined: int = 0
    stopped_at: Optional[Tuple[str, str]] = None


def select_settled_tail(
    buffer_text: str,
    *,
    graph: Any,
    now: str,
    lag_days: int,
    threshold: int,
    window: int,
    archive_name: str,
) -> Selection:
    """Selects the contiguous settled run at the buffer's tail.

    Args:
        buffer_text: One read of the live buffer.
        graph: Has ``get_node(node_id)`` and ``get_outgoing_edges(node_id)``.
        now: The rotation instant, an ISO-8601 stamp with a UTC offset.
        lag_days: How long an entry must have gone untouched to be quiet.
        threshold: The buffered count at which evaluation starts.
        window: The most entries to evaluate.
        archive_name: The archive basename every selected block is filed under.

    Returns:
        The selection. Below the threshold it is empty, with nothing parsed and no
        graph read.

    Raises:
        ValueError: If ``now`` does not parse or carries no offset.
        Exception: Whatever a graph read raises.
    """
    lines = buffer_text.splitlines(keepends=True)
    heads = markers.entry_heading_indices(lines)
    buffered = len(heads)
    if buffered == 0 or buffered < threshold or window < 1:
        return Selection(buffered=buffered)

    instant = _parse_stamp(now)
    if instant is None:
        raise ValueError(f"rotation instant {now!r} is not an offset ISO-8601 stamp")

    tail = heads[-window:]
    # Line 1 is the synthetic sentinel, so the parser's 1-based line numbers index
    # `window_lines` directly: no offset anywhere.
    window_lines = [_WINDOW_SENTINEL] + lines[tail[0]:]
    failures: List[EntryFailure] = []
    entries = parse_entry_stream("".join(window_lines), "decision", failures=failures)
    if len(entries) + len(failures) != len(tail):
        return Selection(buffered=buffered, stopped_at=("", WINDOW_MISMATCH))

    sections: List[Union[ParsedEntry, EntryFailure]] = [*entries, *failures]
    sections.sort(key=lambda section: section.line_start, reverse=True)

    selection = Selection(buffered=buffered)
    lag = timedelta(days=lag_days)
    for section in sections:
        selection.examined += 1
        if isinstance(section, EntryFailure):
            selection.stopped_at = (section.slug or "", UNPARSEABLE)
            break
        reason = _unsettled_reason(section, graph, instant, lag)
        if reason is not None:
            selection.stopped_at = (section.slug, reason)
            break
        raw = "".join(window_lines[section.line_start - 1:section.line_end])
        selection.blocks.append(RotationBlock(section.slug, raw, archive_name))
    return selection


def _unsettled_reason(
    entry: ParsedEntry, graph: Any, instant: datetime, lag: timedelta
) -> Optional[str]:
    """Returns why ``entry`` is not settled, or ``None`` when it is.

    The conjuncts run cheapest first: one node read, a stamp comparison, then the
    edge read and the divergence leaf.
    """
    node_id = compute_node_id(
        kind="decision",
        axiom=entry.axiom,
        mechanism_refs=entry.mechanisms,
        topic=entry.topic,
        questions_raised=entry.questions_raised,
    )
    node = graph.get_node(node_id)
    if not node:
        return UNCOMMITTED
    stamp = _parse_stamp(node.get("updated_at"))
    if stamp is None:
        return UNSTAMPED
    if instant - stamp <= lag:
        return RECENT
    report = entry_divergence(
        entry, node, node.get("scope") or [], graph.get_outgoing_edges(node_id)
    )
    if has_divergence(report):
        return DIVERGED
    return None


def _parse_stamp(value: Any) -> Optional[datetime]:
    """Parses an offset ISO-8601 stamp; ``None`` for anything else, naive included."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.utcoffset() is not None else None
