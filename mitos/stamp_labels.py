"""The stamp label map: a short label for every live decision a stamp names (recall-confidence 8a).

A modifier stamp is a bare slug: ``amended_by: ["x"]`` says something moved on, not
what; a retired handle's ``live_end: ["y"]`` says where the current word lives, not
what it says. Each response on the carrying list therefore gains one
``stamp_labels`` map, keyed by slug, holding the first clause of the axiom of every
live decision that its stamps and live ends name and whose axiom the response does
not already carry. It is read for the whole answer in **one** batched store query
(:meth:`~mitos.store.GraphStore.get_live_axioms`), none when there is nothing to
label. A fault in that read costs the map alone: ``stamp_labels_unread``, never the
answer. This is the cheap partial read of Trace's composed answer; composing the
deltas is Trace's, so nothing here summarises — a label is cut, never composed.

**This module is the one place the rule lives.** Who is labelled:

* **named** — on every entry the response shows: ``amended_by`` ∪ ``narrowed_by``
  always (the store's source gate makes them live); then, on an entry carrying
  ``live_end``, its slugs, else ``superseded_by`` ∪ ``corrected_by`` (live by
  construction when no live end was written). That is the successor rule
  ``sync._live_successors`` and ``recall._successor_slugs`` also spell (a row pins
  agreement with the first), so a retired intermediate hop is never named.
* **carried** — the slugs of the live decisions the response shows with their
  axiom: ranked window hits, listed decisions in a live state, rejection matches
  whose letter is not ``retired`` (their owner's axiom is in the response, as a
  ``here`` Letter or a window hit), lexical decision matches, the scope dump's
  decisions and pause neighbours. A retired entry is never carried, even one that
  shows an axiom (a tombstone): MI-13 permits a same-slug supersession, so counting
  it would suppress exactly the label the reader needs.
* **targets** = named − carried, compared by ``str.casefold()``, kept in
  first-appearance order under every distinct verbatim spelling, so each stamp
  finds its key. The store's live filter is a guard (against the window between
  the stamp read and this one, and against a listed retired decision's unwalked
  kill stamp), not where the rule lives.

**Not targets, by rule:** the pause's declared echo (``declared``,
``declared_no_near_match``) and ``descends_from_declared``, which echo the caller's
own relation arguments, as a write receipt's echo does; ``successor_on_pause``,
whose slugs are shown neighbours and so carried; any stamp on a by-handle read
(lock 2). Each is a key this module never reads. **An open question's stamps are
never labelled**: ``amends``/``narrows``/``supersedes``/``corrects`` are same-kind
edges, so an open question's stamps name open questions, and none is a decision —
no surface hands them here.

**A listed retired decision's kill stamp is unwalked** (``list_decisions`` reads no
live ends): it may name a retired hop whose slug a live decision now holds (MI-13
permits it). That decision's axiom then labels it, which is the answer
``show_node(slug)`` gives for the same handle.

**The read is by slug, by design.** Constraint 8 governs how a vector hit or a
lexical corpus entry resolves to its node, and both still resolve by id; a stamp is
a slug, so the label answers the question the stamp poses — which live decision
does this handle name now — exactly as its dereference would.

Tier 1: stdlib and ``mitos.errors`` only. ``recall.py`` never imports this module
(the check family's fence and its zero-import pin); it takes the labels injected.
"""

from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, NamedTuple, Sequence, Tuple

from mitos.errors import DatabaseError

# The label's budget in UTF-8 bytes, its ``…`` mark included. A byte bound holds
# roughly one token count across scripts (a code-point bound would let a CJK label
# cost about three times a Latin one). Set by 8a's M2 from the copy's replay — see
# measure/labels/ — as REJECTION_SECTION_BOUND is set from its tokens.
LABEL_CUT_BYTES: int = 120

# The ellipsis every truncation seam already uses: a label ends with it exactly when
# non-whitespace text of the axiom was dropped.
CUT_MARK = "…"

# Clause marks that cut only when whitespace follows them: scripts that space their
# words, where the same character also sits inside a token (``v0.19.1``,
# ``recall.py``, ``mech:created_at``, ``l·l``).
_SPACED_BOUNDARIES: FrozenSet[str] = frozenset({
    ".", ";", ":", "!", "?",      # Latin (ASCII); ";" is also NFC's Greek question mark
    "؛", "؟", "۔",                # Arabic semicolon, question mark, full stop
    "।", "॥",                     # Devanagari danda, double danda
    "։",                          # Armenian full stop
    "።", "፤",                     # Ethiopic full stop, semicolon
    ";", "·",           # Greek question mark, ano teleia (as typed)
    "·",                     # middle dot: NFC's form of the ano teleia
})

# Clause marks that cut with or without whitespace after them: scripts that write
# without word spaces (CJK full-width and ideographic marks), and the em dash.
_CLOSED_BOUNDARIES: FrozenSet[str] = frozenset({
    "。", "；", "：", "！", "？", "．",  # ideographic / full-width
    "—",                              # em dash
})

# Every clause mark the cut recognises. The comma is excluded on purpose: on the
# dogfood graph (626 decisions) a comma sits at a median 92 characters against 134
# without it, and its early cuts are list commas ("A, B and C").
CLAUSE_BOUNDARIES: FrozenSet[str] = _SPACED_BOUNDARIES | _CLOSED_BOUNDARIES

# The computed states served as live. Agreement-pinned to ``store.LIVE_STATES`` by a
# row; spelled here because this leaf cannot import the store.
LIVE_STATES: Tuple[str, ...] = ("active", "drifted")

# A rejection match whose owner is retired: its letter, and the one letter that is
# not carried.
_RETIRED_LETTER = "retired"


def _fits(text: str, budget: int) -> bool:
    return len(text.encode("utf-8")) <= budget


def _first_cut(text: str) -> int:
    """The index of the first qualifying clause boundary that cuts, or -1."""
    last = len(text) - 1
    for index in range(1, last):
        char = text[index]
        if char in _CLOSED_BOUNDARIES:
            return index
        if char in _SPACED_BOUNDARIES and text[index + 1].isspace():
            return index
    return -1


def cut_label(axiom: str) -> str:
    """Cuts an axiom to its label: the first clause, else a length bound, marked when cut.

    Pure and total. The axiom is stripped and its whitespace runs (newlines
    included) collapse to single spaces. The scan takes the first **qualifying**
    clause boundary at a position past 0: a mark in :data:`CLAUSE_BOUNDARIES` that
    non-whitespace text follows, a spaced mark only when whitespace follows it (so
    ``v0.19.1`` and ``recall.py`` are skipped and the scan goes on). A terminal mark
    is no cut: ``We use SQLite.`` stays whole. When the clause before that boundary
    fits :data:`LABEL_CUT_BYTES` with its mark, it is the label. Otherwise the whole
    axiom when it fits; else a cut at the last whitespace inside the bound; with none
    (CJK, one long token), at the last whole code point inside it, never
    mid-sequence. The label ends with :data:`CUT_MARK` exactly when non-whitespace
    text was dropped, and the mark counts toward the bound.

    Args:
        axiom: The decision's axiom, as stored.

    Returns:
        The label; ``""`` for an empty or whitespace axiom.
    """
    text = " ".join(axiom.split())
    if not text:
        return ""
    room = LABEL_CUT_BYTES - len(CUT_MARK.encode("utf-8"))
    boundary = _first_cut(text)
    if boundary > 0:
        clause = text[:boundary].rstrip()
        if _fits(clause, room):
            return clause + CUT_MARK
    if _fits(text, LABEL_CUT_BYTES):
        return text
    kept = ""
    for char in text:
        if not _fits(kept + char, room):
            break
        kept += char
    # A word cut inside the bound, unless the bound already fell on a space.
    space = kept.rfind(" ")
    if space > 0 and not text[len(kept)].isspace():
        kept = kept[:space]
    return kept.rstrip() + CUT_MARK


class StampLabels(NamedTuple):
    """One response's labels: the map, or the fact that it could not be read.

    Attributes:
        labels: Verbatim slug → label, in first-appearance order. Never mutated.
        unread: True when the label read faulted; ``labels`` is then empty.
    """

    labels: Dict[str, str]
    unread: bool

    def envelope_keys(self) -> Dict[str, Any]:
        """The two keys a response carries: the map when non-empty, the flag on a fault.

        For a surface that places them together (the list, the pause); the ranked
        envelopes and the lookup place them through ``recall``. An absent map means
        computed and nothing to label.
        """
        if self.unread:
            return {"stamp_labels_unread": True}
        if self.labels:
            return {"stamp_labels": dict(self.labels)}
        return {}


NO_LABELS = StampLabels({}, False)


def _named_slugs(entry: Mapping[str, Any]) -> List[str]:
    """The slugs one entry's stamps and live end name, by the successor rule."""
    named = list(entry.get("amended_by") or []) + list(entry.get("narrowed_by") or [])
    # Membership, as ``display.retired_handle`` writes it: `[]` is a stated fact (the
    # chain reaches no live decision), and it hides the immediate kill stamps.
    if "live_end" in entry:
        named += list(entry["live_end"] or [])
    else:
        named += (list(entry.get("superseded_by") or [])
                  + list(entry.get("corrected_by") or []))
    return named


def label_targets(*, named: Iterable[Mapping[str, Any]],
                  carried: Iterable[str]) -> List[str]:
    """The slugs to label: named − carried, by casefold, every verbatim spelling once.

    Args:
        named: The response's entries whose stamps may name a decision, in the order
            the response shows them.
        carried: The slugs of the live decisions the response shows with their axiom.

    Returns:
        Verbatim slugs in first-appearance order; a spelling whose casefold is
        carried is left out, and two spellings of one casefold are both kept (each
        stamp finds its key; the read folds them into one).
    """
    carried_folds = {slug.casefold() for slug in carried}
    targets: List[str] = []
    for entry in named:
        for slug in _named_slugs(entry):
            if slug.casefold() not in carried_folds and slug not in targets:
                targets.append(slug)
    return targets


def read_stamp_labels(store: "object", *, named: Iterable[Mapping[str, Any]],
                      carried: Iterable[str]) -> StampLabels:
    """Reads one response's labels in one batched query, or none when nothing is targeted.

    Total over ``DatabaseError``: a fault returns ``StampLabels({}, True)`` and costs
    the map alone. Contract faults are loud — a store with no callable
    ``get_live_axioms`` while there are targets, or a result that is not a dict of
    strings, raises ``TypeError`` — and every boundary calls this outside its degrade
    ``try``, so a defect propagates and never poses as degraded recall (ADR
    ``ranked-read-rider-is-composed-once-outside-the-degrade-try-and-a-call-site-raise-propagates``).

    Args:
        store: The graph store (duck-typed: ``get_live_axioms``).
        named: The response's stamped entries, in response order.
        carried: The slugs of the live decisions shown with their axiom.

    Returns:
        A :class:`StampLabels`; :data:`NO_LABELS` with no read when nothing is
        targeted. A target whose read value cuts to ``""`` is left out: it labels
        nothing.

    Raises:
        TypeError: On a store or result that breaks the read's contract.
    """
    targets = label_targets(named=named, carried=carried)
    if not targets:
        return NO_LABELS
    reader = getattr(store, "get_live_axioms", None)
    if not callable(reader):
        raise TypeError(
            f"read_stamp_labels: {type(store).__name__} has no get_live_axioms")
    try:
        axioms = reader(targets)
    except DatabaseError:
        return StampLabels({}, True)
    if not isinstance(axioms, dict):
        raise TypeError(
            f"read_stamp_labels: get_live_axioms returned {type(axioms).__name__}, "
            "not a dict")
    labels: Dict[str, str] = {}
    for slug in targets:
        if slug not in axioms:
            continue
        axiom = axioms[slug]
        if not isinstance(axiom, str):
            raise TypeError(
                f"read_stamp_labels: the axiom read for {slug!r} is "
                f"{type(axiom).__name__}, not str")
        label = cut_label(axiom)
        if label:
            labels[slug] = label
    return StampLabels(labels, False)


def _shown_letter(match: Mapping[str, Any]) -> bool:
    return match.get("letter") != _RETIRED_LETTER


def ranked_labels(store: "object", *, window: Sequence[Mapping[str, Any]],
                  handles: Sequence[Mapping[str, Any]],
                  matches: Sequence[Mapping[str, Any]]) -> StampLabels:
    """The ranked family's labels: window hits, retired handles, rejection matches.

    Carried: every window hit and every match whose letter is not ``retired``. The
    blackout's ``all_superseded`` is the same handles, so it is passed once.

    Args:
        store: The graph store.
        window: The window's hit payloads (``active_decisions`` / ``matches``).
        handles: The retired partition's shown handles.
        matches: The rejection section's matches.

    Returns:
        A :class:`StampLabels`.
    """
    return read_stamp_labels(
        store, named=[*window, *handles, *matches],
        carried=[*(hit["slug"] for hit in window),
                 *(match["slug"] for match in matches if _shown_letter(match))],
    )


def match_labels(store: "object", matches: Sequence[Mapping[str, Any]], *,
                 decisions: Sequence[Mapping[str, Any]] = ()) -> StampLabels:
    """The lookup's and the degraded exits' labels: rejection matches beside decisions.

    Carried: every match whose letter is not ``retired``, and every entry of
    ``decisions`` — a lexical read that dropped retired matches holds live ones only,
    an unsynced block has no graph node to be named, and the scope dump's decisions
    are live by its read. (An unsynced draft sharing its slug with a live decision —
    a pending same-slug supersession — counts as carrying that decision while
    showing the draft's own axiom; rare, and the draft names what it replaces.)

    Args:
        store: The graph store.
        matches: The rejection matches (the lookup's, or a degraded section's).
        decisions: The decision matches or dump decisions shown beside them.

    Returns:
        A :class:`StampLabels`.
    """
    return read_stamp_labels(
        store, named=[*decisions, *matches],
        carried=[*(entry["slug"] for entry in decisions),
                 *(match["slug"] for match in matches if _shown_letter(match))],
    )


def list_labels(store: "object", *,
                decisions: Sequence[Mapping[str, Any]]) -> StampLabels:
    """The listing's labels, at its default and ``brief`` tiers (never ``oneline``).

    Carried: every listed decision whose ``state`` is live. A listed retired decision
    is named and never carried, so a same-slug successor it names is labelled.

    Args:
        store: The graph store.
        decisions: The listed decision payloads (``slug``, ``state``, stamps).

    Returns:
        A :class:`StampLabels`.
    """
    return read_stamp_labels(
        store, named=decisions,
        carried=[entry["slug"] for entry in decisions
                 if entry.get("state") in LIVE_STATES],
    )


def pause_labels(store: "object", *, neighbors: Sequence[Mapping[str, Any]],
                 tombstones: Sequence[Mapping[str, Any]]) -> StampLabels:
    """The record pause's labels: neighbours' amenders and tombstones' live successors.

    Carried: every shown neighbour (a live decision with its axiom). A tombstone is
    retired, so it is named and never carried.

    Args:
        store: The graph store.
        neighbors: The pause's shown neighbours.
        tombstones: The pause's shown tombstones.

    Returns:
        A :class:`StampLabels`.
    """
    return read_stamp_labels(
        store, named=[*neighbors, *tombstones],
        carried=[neighbor["slug"] for neighbor in neighbors],
    )
