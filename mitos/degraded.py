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
``recall.DEGRADED_NOT_COMPUTED`` tokens) plus the omission clause on the note. The
lexical envelope adds both here; the dump's note gains the clause from
``recall.assess_surface_recall`` and its boundary sets the key.

The term-match itself stays in ``lexical.lexical_fallback``, which has no surface,
so its store-less golden rows stay byte-valid.

Tier 2: composition over ``lexical``, ``recall``, ``display`` and ``divergence``. It
never imports ``cli`` or ``mcp_server``, and nothing in the check family imports it.
"""

from typing import Any, Dict, List, NamedTuple, Optional

from mitos.display import clamp_limit, letter_payload
from mitos.divergence import _corpus_files
from mitos.lexical import lexical_fallback
from mitos.recall import (DEGRADED_NOT_COMPUTED, corpus_provenance, count_withheld,
                          degraded_omission_clause, stamps_unread_clause,
                          withheld_clause)
from mitos.store import MODIFIER_EDGE_KEYS


class ScopeDump(NamedTuple):
    """The degraded scope dump of ``surface``: the shown decisions and the scope's size.

    Attributes:
        decisions: Letter payloads, stamped, at most ``limit`` of them.
        total: How many active decisions the scope holds (the dump note's count).
    """

    decisions: List[Dict[str, Any]]
    total: int


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

    Runs ``lexical_fallback`` over the workspace's corpus (buffer plus archives),
    then adds, in order: the query and the corpus provenance; ``not_computed`` and
    the omission clause; the withheld count and clause when ``full_top`` thinned a
    match; the stamps-unread clause when the leaf could not read the stamps; then
    the open questions and the check notice when given. The note reads: the leaf's
    note, omission, withheld, stamps-unread.

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
    envelope = lexical_fallback(
        query, corpus_paths=_corpus_files(config), reason=reason, store=store,
        limit=limit, full_top=full_top,
    )
    envelope["query"] = query
    envelope.update(corpus_provenance(config))
    envelope["not_computed"] = list(DEGRADED_NOT_COMPUTED)
    envelope["note"] = f"{envelope['note']} {degraded_omission_clause()}"
    withheld = count_withheld(envelope["matches"])
    if withheld:
        envelope["note"] = (
            f"{envelope['note']} {withheld_clause(withheld, surface=surface, config=config)}"
        )
        envelope["rejected_paths_withheld"] = withheld
    if envelope.get("stamps_unread"):
        envelope["note"] = (
            f"{envelope['note']} {stamps_unread_clause(surface=surface, config=config)}"
        )
    if open_questions is not None:
        envelope["open_questions"] = open_questions
    if check_notice is not None:
        envelope["check_notice"] = check_notice
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
    for rank, node in enumerate(active[:clamp_limit(limit)], start=1):
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
    return ScopeDump(decisions=decisions, total=len(active))
