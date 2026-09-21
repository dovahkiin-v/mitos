"""The standing check notice: the last contradiction check's outcome, carried to whoever comes next.

A blocked commit and ``mitos check``'s own report reach only the session that ran
them. The last-attempt record (``telemetry.check_attempt``) holds the outcome, and
this module turns it into one dated, plain line for the next agent or person to
touch the corpus: new contradictions named by handle, a check that could not
complete and why, a spend nobody was allowed to authorize, or a check that started
and recorded no outcome. The line is dated and claims nothing about the present.

Three public names:

* :func:`check_notice_line` — the pure renderer. Every sentence lives here, and none
  names a shell command; a CLI boundary appends its own recovery clause.
* :func:`compose_check_notice` — the total composer. It applies the show rule, reads
  through an injected ``read_attempt`` and resolves pair handles through an injected
  ``get_node``. It never raises, so a host can call it without a ``try``.
* :data:`NOTICE_PAIRS_SHOWN` — how many pairs are named; the rest are counted.

Why a module of its own: ``display`` and ``recall`` sit inside the check family's
fenced import closure (``tests/test_conflict_closeout.py``), and the renderer must
match states by ``telemetry``'s ``ATTEMPT_*`` constants rather than re-spell them.
Importing ``telemetry`` from ``display`` would pull ``telemetry`` and ``store`` into
that closure. Nothing in the closure imports this module.

The read is injected, never called from here: every boundary passes
``telemetry.read_last_attempt`` and its own store's ``get_node``, so the boundary owns
which reader and which connection run, and a test can spy the read. The ``created``
record receipt, ``mitos sync`` and every ``surface`` exit (MCP and CLI; never
``query``) carry the notice. The receipt carries it as
the fact alone (ADR ``receipt-dict-strings-are-mcp-boundary-so-recovery-splits-per-renderer``),
and its coherence line keeps the one ``mitos check`` recipe
(ADR ``created-receipt-names-its-recovery-once-on-the-unconditional-line``).
The ``check_notice`` key is present only when a notice is shown, so a healthy corpus
pays zero bytes.
"""

import sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from mitos.config import judge_api_key
from mitos.display import handle_text, node_handles
from mitos.telemetry import (
    ATTEMPT_COULD_NOT_COMPLETE,
    ATTEMPT_NEW_FINDINGS,
    ATTEMPT_NO_NEW_FINDINGS,
    ATTEMPT_SPEND_NOT_AUTHORIZED,
    ATTEMPT_STARTED,
    LastAttempt,
)

NOTICE_PAIRS_SHOWN = 5

# One past-tense phrase per ``check._DEGRADATION_TOKENS`` token, pinned total by a
# row. No value names a command, and none repeats ``cli._JUDGMENT_FAILURE_WORDS``:
# the notice states what happened, and each boundary says what to do about it.
# ``collection_missing`` rides with ``sweep`` and ``judgment_truncated`` with
# ``judgment``, so each pair reads as cause after effect.
_REASON_WORDS: Dict[str, str] = {
    "sweep": "the corpus sweep degraded",
    "judgment": "some judgment batches did not complete",
    "reuse_read": "prior verdicts could not be read",
    "telemetry_write": "some batch results could not be recorded",
    "stale_index": "some embeddings were not yet in the vector index",
    "probe_read": "completeness could not be certified",
    "collection_missing": "the vector collection was missing",
    "judgment_truncated": "a judge response was cut off at its token limit",
}

# The payload's data keys, in every form. ``line`` is rendered from these alone.
_DATA_KEYS = (
    "state",
    "started_at",
    "reasons",
    "new_pairs",
    "new_pair_count",
    "batches_planned",
)

_PAIR_STATES = (ATTEMPT_NEW_FINDINGS, ATTEMPT_COULD_NOT_COMPLETE)


def _when(started_at: Any) -> str:
    """Renders the attempt's time as ``YYYY-MM-DD HH:MM:SS UTC``.

    Seconds are kept: for a ``started`` record the age is the information. A time
    with no offset is read as UTC, since every writer mints UTC. Text that
    ``fromisoformat`` rejects prints verbatim.
    """
    try:
        moment = datetime.fromisoformat(str(started_at))
    except ValueError:
        return str(started_at)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _contradictions(n: int) -> str:
    """Returns ``"1 new contradiction"`` or ``"N new contradictions"``."""
    return f"{n} new contradiction" + ("" if n == 1 else "s")


def _pair_list(pairs: Sequence[Mapping[str, Any]], count: int) -> str:
    """Names each shown pair as ``A ✗ B`` and counts the rest.

    The one place the cap's tail is built, shared by the two forms that name pairs.
    """
    named = ", ".join(
        f"{handle_text(p['proposal'])} ✗ {handle_text(p['partner'])}" for p in pairs
    )
    rest = count - len(pairs)
    return f"{named}, and {rest} more" if rest > 0 else named


def check_notice_line(notice: Mapping[str, Any]) -> str:
    """Renders the notice's one line from its data keys.

    Five forms, one per state, plus a residual for a state this build does not
    know. Every form is dated and past tense, and none names a command. Pure: no
    I/O, no clock, and ``line`` itself is never read.

    Args:
        notice: A payload holding the six data keys (``state``, ``started_at``,
            ``reasons``, ``new_pairs``, ``new_pair_count``, ``batches_planned``).

    Returns:
        The notice sentence (two sentences when an incomplete check still
        reported pairs).
    """
    state = notice.get("state")
    head = f"The last contradiction check ({_when(notice.get('started_at'))})"
    pairs = notice.get("new_pairs") or []
    count = notice.get("new_pair_count") or 0

    if state == ATTEMPT_NEW_FINDINGS:
        if not count:
            # A hand-damaged record (NULL pairs): the state is the fact, no list.
            return f"{head} found new contradictions."
        return f"{head} found {_contradictions(count)}: {_pair_list(pairs, count)}."
    if state == ATTEMPT_COULD_NOT_COMPLETE:
        reasons = notice.get("reasons") or []
        clause = "; ".join(_REASON_WORDS.get(t, t) for t in reasons)
        line = f"{head} could not complete" + (f": {clause}." if clause else ".")
        if count:
            line += f" It reported {_contradictions(count)}: {_pair_list(pairs, count)}."
        return line
    if state == ATTEMPT_SPEND_NOT_AUTHORIZED:
        planned = notice.get("batches_planned")
        if planned is None:
            batches = "an unknown number of judgment batches"
        else:
            batches = f"{planned} judgment batch" + ("" if planned == 1 else "es")
        return (
            f"{head} planned {batches} and was not authorized to spend; "
            f"a person authorizes that."
        )
    if state == ATTEMPT_STARTED:
        return (
            f"{head} started and has recorded no outcome: it is still running, "
            f"or it ended without one."
        )
    return f"{head} recorded an outcome this build does not know: {state}."


def _payload(
    attempt: LastAttempt,
    get_node: Callable[[str], Optional[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Builds the notice payload for a shown attempt; resolves only the shown pairs."""
    new_pairs: Optional[List[Dict[str, Any]]] = None
    new_pair_count: Optional[int] = None
    if attempt.state in _PAIR_STATES:
        stored = list(attempt.new_pairs or ())
        new_pair_count = len(stored)
        shown = stored[:NOTICE_PAIRS_SHOWN]
        ids = [side for pair in shown for side in pair]
        handles = node_handles(ids, get_node) if ids else []
        new_pairs = [
            {"proposal": handles[2 * i], "partner": handles[2 * i + 1]}
            for i in range(len(shown))
        ]
    tokens = attempt.degradation_tokens
    notice: Dict[str, Any] = {
        "state": attempt.state,
        "started_at": attempt.started_at,
        "reasons": list(tokens) if tokens is not None else None,
        "new_pairs": new_pairs,
        "new_pair_count": new_pair_count,
        "batches_planned": attempt.batches_planned,
    }
    notice["line"] = check_notice_line({k: notice[k] for k in _DATA_KEYS})
    return notice


def compose_check_notice(
    config: Any,
    *,
    read_attempt: Callable[[str], Any],
    get_node: Callable[[str], Optional[Mapping[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Composes the standing check notice, or returns None when there is nothing to show.

    The show rule, in order: a keyless workspace shows nothing and reads nothing
    (the commit gate's own key test); a missing, below-rung or unreadable record
    shows nothing (``mitos status`` owns that diagnosis); ``no_new_findings`` shows
    nothing; every other state is shown, a newer build's state included.

    Never raises. An unexpected fault in the read, the handle lookup or the render
    returns None and writes one ``[Warning]`` line to stderr (stdout is JSON-RPC on
    the MCP server, which shares this path).

    Args:
        config: The workspace's ``MitosConfig``.
        read_attempt: Reads the last-attempt record from a telemetry path;
            ``telemetry.read_last_attempt`` at every call site.
        get_node: Returns a node dict by id, or None; the caller's store's
            ``get_node``.

    Returns:
        The ``check_notice`` payload (six data keys plus ``line``), or None.
    """
    try:
        if judge_api_key(config) is None:
            return None
        attempt = read_attempt(config.telemetry_path)
        if not isinstance(attempt, LastAttempt):
            return None
        if attempt.state == ATTEMPT_NO_NEW_FINDINGS:
            return None
        return _payload(attempt, get_node)
    except Exception as e:
        print(
            f"[Warning] Check notice could not be composed: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return None
