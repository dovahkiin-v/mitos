"""The one composer of what a degraded read says, shared by both boundaries.

When semantic recall is down, both read verbs end in one of two degraded exits:
``surface``'s unranked scope dump (a scope was given and holds active decisions),
or the lexical envelope (a term-match over the markdown corpus, on both verbs). This
module exists so each key and clause of those exits is composed once: the CLI and
the MCP server each render what it returns and add nothing of their own, so the two
surfaces cannot drift (CLI⇄MCP parity is a standing contract). Per-surface wording
enters only through the ``surface`` keyword, which selects the existing pointer
clauses.

Every degraded exit states what it did not compute: ``not_computed`` (the
``recall.DEGRADED_NOT_COMPUTED`` tokens) plus the omission clause on the note. And
every degraded exit carries a rejection section matched by words: the rejected
options of the corpus, term-matched item by item, beside the decisions, labelled
``rejection_section: "lexical"`` with no score, and the section clause after the
omission clause. The lexical envelope adds all of it here; the dump's note gains the
omission clause from ``recall.assess_surface_recall``, and :func:`dump_section`
composes the dump's keys and the rest of its note, which its boundary only places.

The term-match itself stays in ``lexical`` (``lexical_degraded_read``,
``lexical_window_section``), which has no surface, so ``lexical_fallback``'s
store-less golden rows stay byte-valid.

The field-scoped lookup's lexical answer (``query`` with ``field``) is composed here
too, by :func:`lexical_rejection_envelope` over ``lexical.lexical_rejection_matches``:
its matches are rejected options, so it has its own header, its own ``not_computed``
tokens and its own omission clause.

Tier 2: composition over ``lexical``, ``recall``, ``display`` and ``divergence``. It
never imports ``cli`` or ``mcp_server``, and nothing in the check family imports it.
"""

from typing import Any, Dict, List, NamedTuple, Optional

from mitos.display import clamp_limit, letter_payload
from mitos.divergence import _corpus_files
from mitos.lexical import (_STAMPS_UNAVAILABLE_DISCLOSURE, lexical_degraded_read,
                           lexical_fallback_envelope, lexical_rejection_header,
                           lexical_rejection_matches, lexical_window_section)
from mitos.recall import (DEGRADED_NOT_COMPUTED, DEGRADED_SECTION_TOKEN,
                          LOOKUP_DEGRADED_NOT_COMPUTED, REJECTED_PATHS_FIELD,
                          REJECTION_SECTION_BOUND, corpus_provenance,
                          count_withheld_decisions, degraded_omission_clause,
                          degraded_section_clause, lookup_omission_clause,
                          rejection_owners_whole, stamps_unread_clause,
                          withheld_clause)
from mitos.store import MODIFIER_EDGE_KEYS

# The state token every degraded section carries when it was searched: matched by
# words, so never a ranked token (``recall.RejectionSection``'s vocabulary gains it).
LEXICAL_SECTION_STATE = "lexical"


class ScopeDump(NamedTuple):
    """The degraded scope dump of ``surface``: the shown decisions and the scope's size.

    Attributes:
        decisions: Letter payloads, stamped, at most ``limit`` of them.
        total: How many active decisions the scope holds (the dump note's count).
        ids: The shown decisions' node ids, in rank order (the section's window and
            the withheld population).
        states: Each shown decision's state (``active`` or ``drifted``, from its
            hydration), aligned with ``ids``.
    """

    decisions: List[Dict[str, Any]]
    total: int
    ids: List[str]
    states: List[str]


def _shown_ordinals(matches: List[Dict[str, Any]]) -> Dict[str, set]:
    """Node id → the ordinals a section's matches show of that owner."""
    shown: Dict[str, set] = {}
    for match in matches:
        shown.setdefault(match["id"], set()).add(match["ordinal"])
    return shown


def _unread_sentence(unread: List[str]) -> str:
    """The lexical leaf's sentence naming the corpus files it could not read."""
    return f"{len(unread)} corpus file(s) could not be read: {', '.join(unread)}."


def lexical_envelope(
    query: str,
    *,
    config: Any,
    surface: str,
    reason: str,
    store: Optional[Any],
    limit: Optional[int],
    full_top: Optional[int],
    open_questions: Optional[List[Dict[str, Any]]] = None,
    check_notice: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Composes the lexical degraded envelope for either boundary.

    One lexical pass (``lexical.lexical_degraded_read``) yields the decision matches
    and the rejection section, on the same two graph reads. Keys, in order: the
    leaf's (``lexical.lexical_fallback_envelope``), the query and the corpus
    provenance, ``not_computed``, ``rejection_matches`` (absent when none),
    ``rejection_section: "lexical"``, the withheld count when ``full_top`` withheld
    a decision's reasoning the section does not show, then the open questions and
    the check notice when given. The note reads: the leaf's note, omission, section
    clause, withheld, stamps-unread (one ``stamps_unread`` covers both lists).

    Args:
        query: The claim/topic the caller was trying to recall.
        config: The call's one resolved workspace config (corpus files, provenance,
            and the CLI pointers' selector).
        surface: ``"cli"`` or ``"mcp"`` — selects the pointer clauses' register.
        reason: One-line cause phrase (see ``lexical.degraded_reason_from_error``).
        store: A readable graph store, or None when the graph itself is down.
        limit: The caller's limit, unclamped or clamped (the leaf clamps; clamping
            twice is idempotent).
        full_top: How many top matches keep ``rejected_paths`` (None ⇒ all, 0 ⇒
            none).
        open_questions: An already-computed scoped parked-OQ list (present-if-
            scanned semantics — None means omitted).
        check_notice: The standing check notice ``surface`` composed; None means
            omitted.

    Returns:
        The degraded envelope dict, ready to encode.
    """
    read = lexical_degraded_read(
        query, corpus_paths=_corpus_files(config), store=store, limit=limit,
        full_top=full_top, bound=REJECTION_SECTION_BOUND,
        owners_whole=lambda window_len: rejection_owners_whole(full_top, window_len),
    )
    envelope = lexical_fallback_envelope(read, reason=reason)
    envelope["query"] = query
    envelope.update(corpus_provenance(config))
    envelope["not_computed"] = list(DEGRADED_NOT_COMPUTED)
    if read.section:
        envelope["rejection_matches"] = read.section
    envelope["rejection_section"] = LEXICAL_SECTION_STATE
    envelope["note"] = (
        f"{envelope['note']} {degraded_omission_clause()} "
        f"{degraded_section_clause(computed=True, retired_dropped=read.graph_read)}"
    )
    # Over decisions (the ranked rule): every match past `full_top`, plus every
    # `here` owner whose Letter the depth withheld, each exempt when the section
    # shows every item its field can serve.
    thinned = read.match_ids[full_top:] if full_top is not None else []
    if not rejection_owners_whole(full_top, len(read.match_ids)):
        thinned += [m["id"] for m in read.section if m["letter"] == "here"]
    withheld = count_withheld_decisions(
        thinned_ids=thinned, shown_ordinals=_shown_ordinals(read.section),
        item_counts=read.item_counts,
    )
    if withheld:
        envelope["note"] = (
            f"{envelope['note']} {withheld_clause(withheld, surface=surface, config=config)}"
        )
        envelope["rejected_paths_withheld"] = withheld
    if read.stamps_unread:
        envelope["note"] = (
            f"{envelope['note']} {stamps_unread_clause(surface=surface, config=config)}"
        )
    if open_questions is not None:
        envelope["open_questions"] = open_questions
    if check_notice is not None:
        envelope["check_notice"] = check_notice
    return envelope


# The lookup's lexical no-match line: the grep a person can run to be sure.
_NO_REJECTION_MATCH_LINE = (
    "No rejected option matched the text either — grep the `**Rejected:**` fields of "
    "the workspace's `decisions.md` and `decisions/archive/` to be sure."
)


def lexical_rejection_envelope(
    query: str,
    *,
    config: Any,
    surface: str,
    reason: str,
    store: Optional[Any],
    limit: Optional[int],
    full_top: Optional[int],
    gap: bool,
) -> Dict[str, Any]:
    """Composes the field-scoped lookup's lexical answer for either boundary.

    Every exit of the lookup that cannot rank items by score answers here: a text
    match over the parsed rejected options (``lexical.lexical_rejection_matches``),
    labelled lexical and band-free. Keys, in order: ``query``, ``field``,
    ``degraded``, ``degraded_reason``, ``rejection_matches`` (absent when none),
    ``rejection_section: "gap"`` when the index is a gap, ``not_computed``
    (:data:`recall.LOOKUP_DEGRADED_NOT_COMPUTED`), provenance, ``note``, then
    ``stamps_unavailable`` / ``stamps_unread`` / ``unread_files`` /
    ``rejected_paths_withheld`` when they apply. The note reads: the header, the
    stamps disclosure, the unread files, the no-match line, the omission clause, the
    withheld clause, the stamps-unread clause.

    Args:
        query: The claim the caller asked about.
        config: The call's one resolved workspace config.
        surface: ``"cli"`` or ``"mcp"`` — selects the pointer clauses' register.
        reason: The cause phrase for the header (on the gap,
            ``recall.rejection_lookup_gap_reason``, which carries the heal).
        store: A readable graph store, or None when the graph itself is down.
        limit: The caller's limit, in items.
        full_top: How many owners keep ``rejected_paths`` (None ⇒ all, 0 ⇒ none).
        gap: Whether the rejection index is missing over a non-empty expected set.

    Returns:
        The envelope dict, ready to encode.
    """
    leaf = lexical_rejection_matches(
        query, corpus_paths=_corpus_files(config), store=store, limit=limit,
        full_top=full_top,
    )
    matches = leaf["matches"]
    envelope: Dict[str, Any] = {
        "query": query, "field": REJECTED_PATHS_FIELD, "degraded": "lexical",
        "degraded_reason": reason,
    }
    if matches:
        envelope["rejection_matches"] = matches
    if gap:
        envelope["rejection_section"] = "gap"
    envelope["not_computed"] = list(LOOKUP_DEGRADED_NOT_COMPUTED)
    envelope.update(corpus_provenance(config))
    note = lexical_rejection_header(reason)
    if not leaf["graph_read"]:
        note = f"{note} {_STAMPS_UNAVAILABLE_DISCLOSURE}"
    unread = leaf["unread_files"]
    if unread:
        note = (f"{note} {len(unread)} corpus file(s) could not be read: "
                f"{', '.join(unread)}.")
    if not matches:
        note = f"{note} {_NO_REJECTION_MATCH_LINE}"
    note = f"{note} {lookup_omission_clause(retired_dropped=leaf['graph_read'])}"
    here = [m["id"] for m in matches if m["letter"] == "here"]
    shown: Dict[str, set] = {}
    for match in matches:
        shown.setdefault(match["id"], set()).add(match["ordinal"])
    withheld = count_withheld_decisions(
        thinned_ids=here[full_top:] if full_top is not None else [],
        shown_ordinals=shown, item_counts=leaf["item_counts"],
    )
    if withheld:
        note = f"{note} {withheld_clause(withheld, surface=surface, config=config)}"
    if leaf["stamps_unread"]:
        note = f"{note} {stamps_unread_clause(surface=surface, config=config)}"
    envelope["note"] = note
    if not leaf["graph_read"]:
        envelope["stamps_unavailable"] = True
    if leaf["stamps_unread"]:
        envelope["stamps_unread"] = True
    if unread:
        envelope["unread_files"] = unread
    if withheld:
        envelope["rejected_paths_withheld"] = withheld
    return envelope


def scope_dump(
    store: Any, scope: str, *, limit: Optional[int], full_top: Optional[int]
) -> Optional[ScopeDump]:
    """Lists a scope's active decisions, unranked, for ``surface`` when recall is down.

    One read, ``store.get_active_decisions(scope=scope)``, which hydrates and stamps
    its nodes on one connection, so the dump makes no modifiers read of its own:
    its decisions and their stamps are one fact, stamped or not shown.

    Args:
        store: The graph store (duck-typed).
        scope: The scope tag the caller passed.
        limit: The caller's limit (clamped here, as the ranked reads clamp it).
        full_top: How many top decisions keep ``rejected_paths`` (None ⇒ all).

    Returns:
        The dump, or None when the read raised or the scope holds no active
        decision. None is the caller's cue to take the lexical exit, so ``dump is
        not None`` is exactly the partition between the two degraded exits.
    """
    # The read IS the degraded attempt, and ``get_active_decisions`` wraps no
    # SQLite error, so its whole fault domain — its stamp pass included — is this
    # one broad catch, routed to the lexical exit (which reads and states its own
    # stamp fault). Only the store call is inside: shaping below is loud.
    try:
        active = store.get_active_decisions(scope=scope)
    except Exception:
        return None
    if not active:
        return None
    decisions: List[Dict[str, Any]] = []
    shown = active[:clamp_limit(limit)]
    for rank, node in enumerate(shown, start=1):
        payload = letter_payload(
            node, brief=full_top is not None and rank > full_top,
            extras={"score": 1.0},
        )
        # ``letter_payload`` returns an un-stamped core; the hydrated node carries
        # its stamps.
        for key in MODIFIER_EDGE_KEYS.values():
            if node.get(key):
                payload[key] = list(node[key])
        decisions.append(payload)
    return ScopeDump(
        decisions=decisions, total=len(active), ids=[node["id"] for node in shown],
        # The active view holds live decisions only; drift annotates, never retires.
        states=["drifted" if node.get("is_drifted") else "active" for node in shown],
    )


class DumpSection(NamedTuple):
    """The scope dump's rejection section and the rest of its degraded statement.

    Read by each boundary's tail exactly where a ranked answer's
    ``recall.RejectionSection`` is (``matches``, ``state``, ``dropped``,
    ``withheld``, ``clause`` and :meth:`envelope_keys`), plus ``stamps_unread``,
    which the boundary folds into its one stamps-unread check.

    Attributes:
        matches: The shaped matches (``[]`` when none, or when not searched).
        state: ``"lexical"`` (searched by words) or ``"unavailable"`` (the state
            read failed, so nothing was searched).
        dropped: Always None (no verification ran).
        withheld: ``recall.count_withheld_decisions`` over the dump and the section.
        clause: The section clause, then the unread-files sentence when some.
        stamps_unread: Whether the section owners' stamp read failed.
        not_computed: The dump's ``not_computed`` tokens.
        unread_files: Corpus files that could not be read.
    """

    matches: List[Dict[str, Any]]
    state: str
    dropped: Optional[int]
    withheld: int
    clause: str
    stamps_unread: bool
    not_computed: List[str]
    unread_files: List[str]

    def envelope_keys(self) -> Dict[str, Any]:
        """The dump's degraded keys, in order: ``not_computed``, ``rejection_matches``
        (when some), ``rejection_section``, ``unread_files`` (when some),
        ``stamps_unread`` (when true)."""
        keys: Dict[str, Any] = {"not_computed": self.not_computed}
        if self.matches:
            keys["rejection_matches"] = self.matches
        keys["rejection_section"] = self.state
        if self.unread_files:
            keys["unread_files"] = self.unread_files
        if self.stamps_unread:
            keys["stamps_unread"] = True
        return keys


def dump_section(query: str, dump: ScopeDump, *, config: Any, store: Any,
                 full_top: Optional[int]) -> DumpSection:
    """Composes the scope dump's rejection section, matched by words, for both boundaries.

    The dump's decisions are a graph read; its section is a text match over the
    corpus's rejected items (``lexical.lexical_window_section``), with the dump's
    shown decisions as the window: their items are ``window`` with their ``rank``,
    stated and stamped from the dump's own hydration, never read. Owners outside
    the dump cost at most one state read and one stamp read, each only when it has
    ids. A state-read fault leaves the section **not searched** (``unavailable``,
    no matches, ``rejection_matches`` back on ``not_computed``): the dump's
    decisions are filtered and stamped, so an unfiltered section beside them would
    misdescribe the answer. A stamp-read fault costs the section's stamps alone.
    Wording is surface-free; the boundary appends ``clause``, then its withheld and
    stamps-unread clauses.

    Args:
        query: The claim/topic the caller was trying to recall.
        dump: The non-empty :class:`ScopeDump` the boundary is answering with.
        config: The call's one resolved workspace config (the corpus files).
        store: The graph store the dump read.
        full_top: The effective depth (None ⇒ whole, 0 ⇒ brief).

    Returns:
        A :class:`DumpSection`.
    """
    owners_whole = rejection_owners_whole(full_top, len(dump.ids))
    stamps = {
        node_id: {key: payload[key] for key in MODIFIER_EDGE_KEYS.values() if key in payload}
        for node_id, payload in zip(dump.ids, dump.decisions)
    }
    read = lexical_window_section(
        query, corpus_paths=_corpus_files(config), store=store,
        bound=REJECTION_SECTION_BOUND, owners_whole=owners_whole,
        window_ids=dump.ids, window_states=dict(zip(dump.ids, dump.states)),
        window_stamps=stamps,
    )
    computed = read.graph_read
    matches = read.section if computed else []
    not_computed = list(DEGRADED_NOT_COMPUTED)
    if not computed:
        not_computed.append(DEGRADED_SECTION_TOKEN)
    clause = degraded_section_clause(computed=computed, retired_dropped=computed)
    if read.unread_files:
        clause = f"{clause} {_unread_sentence(read.unread_files)}"
    thinned = [node_id for rank, node_id in enumerate(dump.ids, start=1)
               if full_top is not None and rank > full_top]
    if not owners_whole:
        thinned += [m["id"] for m in matches if m["letter"] == "here"]
    withheld = count_withheld_decisions(
        thinned_ids=thinned, shown_ordinals=_shown_ordinals(matches),
        item_counts=read.item_counts,
    )
    return DumpSection(
        matches=matches, state=LEXICAL_SECTION_STATE if computed else "unavailable",
        dropped=None, withheld=withheld, clause=clause,
        stamps_unread=computed and read.stamps_unread, not_computed=not_computed,
        unread_files=read.unread_files,
    )
