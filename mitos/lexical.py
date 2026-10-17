"""Deterministic lexical fallback for the semantic read verbs.

When semantic recall or the graph is unavailable *for any reason* — an
embedding provider failure (classified by ``provider_cause``: a refused key, a 429
mid-loop, a transient outage, or an unknown error), Qdrant down or the collection
missing, or a pre-V1a graph that takes SQLite reads down with it — ``mitos surface`` and
``mitos query`` degrade to a case-insensitive term-match over the entries of the
markdown corpus — ``decisions.md`` and ``decisions/archive/`` — (slug + axiom)
instead of dead-ending (ADR
``read-verbs-degrade-to-lexical-decisions-md-fallback``). The fallback is a
grep and is presented as one: an explicit degraded header, no similarity
scores, no ``confidence`` field, and a ``degraded: "lexical"`` marker on the
JSON envelope. From the same corpus pass, each entry's rejected options are
matched item by item (never the raw field), so a degraded answer still names the
options a decision already ruled out (:func:`lexical_degraded_read`; the scope
dump's twin, :func:`lexical_window_section`).

The scan covers the whole corpus, not only the buffer: rotation moves committed
entries into the archives, so a buffer-only grep would answer "no matches" over
the decisions that moved. A corpus file that exists but cannot be read is named in
the envelope (``unread_files``) and in the note, never skipped quietly.

Factored here once so the CLI verbs and their MCP twins share the identical
behaviour (CLI⇄MCP parity is a standing contract, not per-surface luck).

Identity and stamping: each matched block is parsed with the commit's own
tokenizer (``parser.parse_entry_stream``) and named by the content-hash id the
commit minted for it (``identity.compute_node_id`` over its axiom and mechanisms),
never by its slug, which a same-slug supersession shares and a rename moves (ADR
``lexical-fallback-resolves-corpus-entries-by-content-hash-id-never-by-slug``).
When the graph is readable, one ``GraphStore.get_node_states`` read for the whole
answer drops the retired matches and states the live ones, and one
``GraphStore.get_modifiers_map`` read stamps them (the every-read-surface rule). A
block whose id the graph does not hold (authored but unsynced) stays, with no state
and no stamps. Two faults are stated, never swallowed: when the graph is down (the
pre-V1a case) or its state read fails, a one-line disclosure replaces the filter and
the stamps (``stamps_unavailable``: entries come straight from markdown and may
include superseded ones); when only the modifiers read fails, the filter and states
stand and the envelope says ``stamps_unread``.
"""

import os
from typing import (Any, Callable, Dict, Iterator, List, Mapping, NamedTuple, Optional,
                    Sequence, Set, Tuple)

from mitos.display import clamp_limit, letter_payload
from mitos.errors import (
    CollectionMissingError,
    DatabaseError,
    EmbeddingError,
    VectorStoreError,
)
from mitos.identity import compute_node_id
from mitos.parser import parse_entry_stream
from mitos.provider_cause import QUOTA, classify_provider_error, provider_cause_phrase
from mitos.rejections import enumerate_rejected_paths, point_items
from mitos.shell_quote import shell_flag
from mitos.store import LIVE_STATES

# Terms shorter than this are dropped from the query before matching — they are
# stop-word noise ("a", "of", "to") that would match nearly every entry. Longer
# stop-words still match and still rank; ``DISPLAY_STOPWORDS`` below only hides them.
LEXICAL_MIN_TERM_LEN: int = 3

# English function words hidden from each match's displayed ``matched_terms`` —
# display only, and counted: the match carries ``stopwords_dropped`` for what was
# hidden, and ranking still counts every matched term, so a hit ranked on these
# words alone shows an empty list beside a non-zero count rather than vanishing.
# English because the list is; the corpus need not be (P9) — a non-English query
# loses nothing, it just drops nothing. Source: the NLTK English stop-word list,
# less contractions and entries shorter than ``LEXICAL_MIN_TERM_LEN`` (they never
# reach a match). Content-bearing common words (``show``) are deliberately absent.
DISPLAY_STOPWORDS: frozenset[str] = frozenset({
    "about", "above", "after", "again", "against", "all", "and", "any", "are",
    "because", "been", "before", "being", "below", "between", "both", "but",
    "can", "did", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "into", "its", "itself", "just",
    "more", "most", "myself", "nor", "not", "now", "off", "once", "only",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "too", "under", "until", "very", "was", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "you", "your",
    "yours", "yourself", "yourselves",
})

_STAMPS_UNAVAILABLE_DISCLOSURE = (
    "Graph unavailable — state/modifier stamps not applied; entries come "
    "straight from the markdown corpus and may include superseded ones."
)

_NO_MATCH_LINE = (
    "No lexical matches either — grep the workspace's `decisions.md` and "
    "`decisions/archive/` to be sure."
)


# Per-surface recovery clauses for the two degraded causes that have a heal, on the
# ``recall._SURFACE_POINTERS`` idiom: the classifier states the cause, each surface
# supplies its own closing clause. The CLI names the command with its selector, the
# ``{selector}`` slot filled by ``shell_quote.shell_flag`` at composition, because a registered name may
# carry a space and an unregistered target's ``project`` is a path. The MCP clause
# names no command and no tool — it states the fact and a person with a shell as the
# next actor (``recall.MISSING_GRAPH_POINTERS["mcp"]``'s register), because an agent
# handed a shell command runs it, and neither heal is a tool.
_RECOVERY_CLAUSES: Dict[str, Dict[str, str]] = {
    "cli": {
        "cutover": "run `mitos cutover {selector}`",
        "reconcile": "run `mitos reconcile {selector}`",
    },
    "mcp": {
        "cutover": (
            "only a cutover migrates it, which no tool on this surface performs — "
            "a person with a shell in that project runs it"
        ),
        "reconcile": (
            "only a reconcile re-creates it and embeds every decision it is missing, which "
            "no tool on this surface performs — a person with a shell in that project "
            "runs it"
        ),
    },
}


def degraded_reason_from_error(
    exc: Optional[BaseException], *, surface: str, project: str
) -> str:
    """Classifies a recall failure into one calm human-readable cause.

    Never returns the raw provider blob — a Gemini error dumps a JSON body into
    ``str(exc)``, and re-printing that is exactly the AX failure the fallback
    exists to replace. An embedding-provider failure is named by
    ``provider_cause``'s closed classifier, the same phrase the scope-dump note
    and the write-path warning carry.

    The arm order is load-bearing: the pre-V1a text first, then
    ``CollectionMissingError`` (its message carries a derived collection name whose
    8-hex path hash can contain ``429``, so the quota text arm must not see it),
    then the 429 text (an ``EmbeddingError`` often wraps a 429), then the other
    typed arms, with ``VectorStoreError`` after the subclass above.

    Args:
        exc: The exception that broke semantic recall, or None when recall was
            never attempted (no provider/vector store wired).
        surface: ``"cli"`` or ``"mcp"`` — selects the recovery clause's register.
            Required, never defaulted: the MCP register must be unable to emit a
            shell command.
        project: The target's ``config.project``, read only by the CLI recipes.

    Returns:
        A short cause phrase for the degraded header.
    """
    recovery = _RECOVERY_CLAUSES[surface]
    if exc is None:
        return "embeddings/Qdrant unavailable"
    text = str(exc)
    if "predates the V1a schema" in text:
        return "graph predates the V1a schema — " + recovery["cutover"].format(
            selector=shell_flag("-p", project))
    # Before the VectorStoreError arm, which it subclasses — otherwise the most-used
    # read verb answers a missing collection with "Qdrant unavailable", the exact
    # blame-the-infrastructure phrase the typed error exists to replace. And before
    # the quota text arm: `mitos-foo-a4291b2c` holds "429", and a missing collection
    # read as a rate limit sends the operator to wait instead of to `reconcile`.
    if isinstance(exc, CollectionMissingError):
        named = f" '{exc.collection}'" if exc.collection else ""
        return f"vector collection{named} missing — " + recovery["reconcile"].format(
            selector=shell_flag("-p", project))
    if classify_provider_error(exc) == QUOTA:
        return provider_cause_phrase(QUOTA)
    if isinstance(exc, EmbeddingError):
        return provider_cause_phrase(classify_provider_error(exc))
    if isinstance(exc, VectorStoreError):
        return "Qdrant unavailable"
    lowered = text.lower()
    if "qdrant" in lowered or "connection" in lowered or "connect" in lowered:
        return "Qdrant unreachable"
    return f"{type(exc).__name__}: recall failed"


def lexical_header(reason: str) -> str:
    """Builds the degraded header line for the given cause."""
    return (
        f"Semantic recall unavailable ({reason}) — deterministic text match "
        f"over the markdown corpus (decisions.md + decisions/archive/) (degraded):"
    )


def lexical_rejection_header(reason: str) -> str:
    """Builds the field-scoped lookup's degraded header line for the given cause.

    Not :func:`lexical_header`, which names a slug-and-axiom match: this one says
    what ran, a text match over the rejected options themselves.
    """
    return (
        f"Semantic search over rejected options unavailable ({reason}) — a "
        f"deterministic text match over the rejected options in the markdown corpus "
        f"(decisions.md + decisions/archive/) (degraded):"
    )


def _query_terms(query: str) -> List[str]:
    """Splits a query into distinct casefolded match terms (len ≥ 3)."""
    seen: List[str] = []
    for token in query.split():
        term = token.strip(".,;:!?'\"()[]{}").casefold()
        if len(term) >= LEXICAL_MIN_TERM_LEN and term not in seen:
            seen.append(term)
    return seen


def _corpus_entries(corpus_paths: Sequence[str],
                    unread: List[str]) -> Iterator[Tuple[int, int, Any]]:
    """Yields the corpus's decision entries newest first, each slug once.

    The one corpus loop both lexical matches share, so they cannot drift: files in
    ``divergence._corpus_files`` order reversed (the buffer first), each read and
    parsed under one guard, then every entry deduplicated by slug before any caller
    matches it.

    Args:
        corpus_paths: The corpus files, oldest archive first, buffer last.
        unread: Appended with the basename of every file that exists and could not
            be read, decoded or parsed.

    Yields:
        ``(file_rank, position, entry)``: the file's rank (0 = newest), the entry's
        position in its file, and the parsed entry.
    """
    seen_slugs: Set[str] = set()
    for file_rank, path in enumerate(reversed(list(corpus_paths))):
        # Read AND parse per file, one guard: a file that exists and cannot be
        # read, decoded or parsed is disclosed, never answered around in silence.
        # A file gone since the listing is simply not part of the corpus now. The
        # commit's tokenizer, so the id a caller computes is the one the commit
        # minted; a malformed entry is dropped into a discarded collector.
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
            entries = (
                parse_entry_stream(text, "decision", source_path=path, failures=[])
                if text else []
            )
        except FileNotFoundError:
            continue
        except Exception:
            unread.append(os.path.basename(path))
            continue
        for position, entry in enumerate(entries):
            # Dedup by handle only — an entry with no slug has nothing to be a
            # duplicate of, and must not shadow another slugless entry. The newest
            # block per slug is kept: a same-slug successor is written after its
            # predecessor, so it is the live one whenever one is live. Every entry
            # is seen here, matched or not.
            key = (entry.slug or "").casefold()
            if key and key in seen_slugs:
                continue
            seen_slugs.add(key)
            yield file_rank, position, entry


class LexicalRead(NamedTuple):
    """One lexical pass over the corpus: the decision matches and the rejection section.

    Attributes:
        matches: The decision matches, :func:`lexical_fallback`'s shape (``[]`` when
            the pass matched no decisions).
        match_ids: Each decision match's content-hash id, in order (unsynced
            blocks included: the id is computed, never read).
        section: The rejection matches, :func:`lexical_rejection_matches`' shape plus
            the ``window`` letter.
        item_counts: Each section owner's id → how many distinct items its field
            can serve.
        graph_read: Whether the graph's states were read (False: no store, or the
            state read raised ``DatabaseError``).
        stamps_unread: Whether the stamp read raised ``DatabaseError``.
        unread_files: Basenames of the corpus files that exist and could not be read.
    """

    matches: List[Dict[str, Any]]
    match_ids: List[str]
    section: List[Dict[str, Any]]
    item_counts: Dict[str, int]
    graph_read: bool
    stamps_unread: bool
    unread_files: List[str]


def _scan(corpus_paths: Sequence[str], terms: List[str], *, decisions: bool,
          items: bool, unread: List[str]) -> Tuple[List[Any], List[Any], Dict[str, int]]:
    """Term-matches every corpus entry once, for its decision text and its items.

    The one pass every lexical answer makes (``_corpus_entries``): each entry's slug
    and axiom when ``decisions``, each **item** of its enumerated ``rejected_paths``
    when ``items`` — never the raw field — keeping only the items the rejection index
    could serve (``rejections.point_items``' ordinals). Both lists come back ranked:
    most distinct terms, then file (newest first), then position (then ordinal).

    Returns:
        ``(decision_ranked, item_ranked, servable)``: decision tuples ``(count,
        file_rank, position, hit, entry, node_id)``, item tuples ``(count, file_rank,
        position, ordinal, hit, entry, node_id, item)``, and each item owner's id →
        its servable item count.
    """
    decision_ranked: List[Any] = []
    item_ranked: List[Any] = []
    servable: Dict[str, int] = {}
    for file_rank, position, entry in _corpus_entries(corpus_paths, unread):
        node_id: Optional[str] = None
        if decisions:
            haystack = f"{entry.slug}\n{entry.axiom}".casefold()
            hit = [t for t in terms if t in haystack]
            if hit:
                node_id = compute_node_id(
                    kind="decision", axiom=entry.axiom,
                    mechanism_refs=entry.mechanisms,
                )
                decision_ranked.append((len(hit), file_rank, position, hit, entry, node_id))
        if not items:
            continue
        hits = []
        for item in enumerate_rejected_paths(entry.rejected_paths).items:
            item_text = item.text.casefold()
            item_hit = [t for t in terms if t in item_text]
            if item_hit:
                hits.append((item, item_hit))
        if not hits:
            continue
        if node_id is None:
            node_id = compute_node_id(
                kind="decision", axiom=entry.axiom, mechanism_refs=entry.mechanisms,
            )
        ordinals = {i.ordinal for i in point_items(node_id, entry.rejected_paths).values()}
        servable[node_id] = len(ordinals)
        for item, item_hit in hits:
            if item.ordinal in ordinals:
                item_ranked.append((len(item_hit), file_rank, position, item.ordinal,
                                    item_hit, entry, node_id, item))
    # Most distinct terms first; ties by file (newest first), then by position —
    # every buffer and rotation-written archive is newest-first below its marker.
    decision_ranked.sort(key=lambda t: (-t[0], t[1], t[2]))
    item_ranked.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
    return decision_ranked, item_ranked, servable


def _read_states(store: Optional[Any], ids: List[str]) -> Tuple[bool, Dict[str, str]]:
    """The one state read: ``(graph_read, states)``; a ``DatabaseError`` reads as no
    store. Only the store call sits in the ``try``: a store missing the method raises."""
    if store is None:
        return False, {}
    try:
        return True, store.get_node_states(ids)
    except DatabaseError:
        return False, {}


def _shape_decisions(ranked: List[Any], states: Dict[str, str], *, graph_read: bool,
                     cap: int, full_top: Optional[int]
                     ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Shapes the ranked decision candidates into matches: ``(matches, match_ids)``.

    Retired candidates are dropped when states were read, before the cap and the
    ``full_top`` thinning, which count kept matches only.
    """
    matches: List[Dict[str, Any]] = []
    match_ids: List[str] = []
    for _count, _file_rank, _position, hit, entry, node_id in ranked:
        # An id the graph does not hold is an authored-but-unsynced block: it
        # stays, with no state and no stamps — dropping the gold source on a graph
        # miss would be a false empty.
        state = states.get(node_id) if graph_read else None
        if state is not None and state not in LIVE_STATES:
            continue
        payload: Dict[str, Any] = {
            "slug": entry.slug,
            "axiom": entry.axiom,
            "scope": entry.scope,
            # Display only: ``hit`` above ranked the entry and stays whole there.
            "matched_terms": [t for t in hit if t not in DISPLAY_STOPWORDS],
        }
        dropped = len(hit) - len(payload["matched_terms"])
        if dropped:
            payload["stopwords_dropped"] = dropped
        # Rank among the matches returned: nothing is appended between here and
        # the append below, and a filtered entry never reaches it.
        if full_top is None or len(matches) + 1 <= full_top:
            payload["rejected_paths"] = entry.rejected_paths
        if state is not None:
            payload["state"] = state
        matches.append(payload)
        match_ids.append(node_id)
        if len(matches) >= cap:
            break
    return matches, match_ids


def _shape_items(ranked: List[Any], states: Mapping[str, str], *, graph_read: bool,
                 cap: int, window: Mapping[str, int], owners_whole: bool,
                 whole_owners: Optional[int]) -> List[Dict[str, Any]]:
    """Shapes the ranked item candidates into rejection matches, at most ``cap``.

    An owner in ``window`` (id → its 1-based rank in the answer's list) is
    ``window`` on every match, with ``rank`` and no Letter. Otherwise an owner's
    first match is ``here`` (its Letter follows: whole when ``owners_whole`` and,
    with ``whole_owners`` set, while fewer than that many owners were lettered —
    ``display.rejection_matches_payload``'s two knobs) and later ones ``section``.
    Retired owners are dropped when states were read; ``states`` may also hold
    states the caller already knew.
    """
    matches: List[Dict[str, Any]] = []
    lettered: List[str] = []
    for _count, _file_rank, _position, ordinal, hit, entry, node_id, item in ranked:
        state = states.get(node_id) if graph_read or node_id in window else None
        if state is not None and state not in LIVE_STATES:
            continue
        payload: Dict[str, Any] = {"slug": entry.slug, "id": node_id}
        if state is not None:
            payload["state"] = state
        payload["ordinal"] = ordinal
        payload["item"] = item.text.rstrip()
        payload["matched_terms"] = [t for t in hit if t not in DISPLAY_STOPWORDS]
        dropped = len(hit) - len(payload["matched_terms"])
        if dropped:
            payload["stopwords_dropped"] = dropped
        if node_id in window:
            payload["letter"] = "window"
            payload["rank"] = window[node_id]
        elif node_id in lettered:
            payload["letter"] = "section"
        else:
            whole = owners_whole and (whole_owners is None
                                      or len(lettered) < whole_owners)
            lettered.append(node_id)
            payload["letter"] = "here"
            owner = {"slug": entry.slug, "core_axiom": entry.axiom,
                     "scope": entry.scope, "rejected_paths": entry.rejected_paths}
            for key, value in letter_payload(owner, brief=not whole).items():
                if key not in payload:
                    payload[key] = value
        matches.append(payload)
        if len(matches) >= cap:
            break
    return matches


def _stamp(payloads: List[Dict[str, Any]], ids: List[str],
           stamps: Mapping[str, Mapping[str, List[str]]]) -> None:
    """Copies each payload's owner stamps onto it, in place (``ids`` aligned)."""
    for payload, node_id in zip(payloads, ids):
        for key, slugs in stamps.get(node_id, {}).items():
            payload[key] = list(slugs)


def _stated(payloads: List[Dict[str, Any]], ids: List[str]) -> List[str]:
    """The distinct ids, in order, of the payloads that carry a ``state``."""
    return list(dict.fromkeys(i for p, i in zip(payloads, ids) if "state" in p))


def _read(query: str, *, corpus_paths: Sequence[str], store: Optional[Any],
          decision_cap: Optional[int], full_top: Optional[int], item_cap: Optional[int],
          owners_whole: Callable[[int], bool], whole_owners: Optional[int]) -> LexicalRead:
    """The lexical envelope's pass: decisions and items, one state read, one stamp read.

    ``decision_cap`` None matches no decisions; ``item_cap`` None matches no items.
    The two reads are unconditional once a store is given (2g's pinned two reads:
    each short-circuits without a connection on no ids), over the union of both
    lists' owners.
    """
    terms = _query_terms(query)
    unread: List[str] = []
    decision_ranked, item_ranked, servable = _scan(
        corpus_paths, terms, decisions=decision_cap is not None,
        items=item_cap is not None, unread=unread)

    # One state read for every candidate, before the walks: the filter must run
    # before the caps and the thinning, which count kept matches only. Called even
    # over no candidates, so the answer's reads are the same two whatever it matched.
    candidate_ids = [t[5] for t in decision_ranked]
    seen = set(candidate_ids)
    candidate_ids += [i for i in dict.fromkeys(t[6] for t in item_ranked) if i not in seen]
    graph_read, states = _read_states(store, candidate_ids)

    matches: List[Dict[str, Any]] = []
    match_ids: List[str] = []
    if decision_cap is not None:
        matches, match_ids = _shape_decisions(decision_ranked, states,
                                              graph_read=graph_read, cap=decision_cap,
                                              full_top=full_top)
    section: List[Dict[str, Any]] = []
    if item_cap is not None:
        window: Dict[str, int] = {}
        for rank, node_id in enumerate(match_ids, start=1):
            window.setdefault(node_id, rank)
        section = _shape_items(item_ranked, states, graph_read=graph_read, cap=item_cap,
                               window=window, owners_whole=owners_whole(len(match_ids)),
                               whole_owners=whole_owners)
    section_ids = [m["id"] for m in section]

    # One stamp read for the stated matches of both lists, after the walks.
    # Unconditional once the state read succeeded, even over no ids. A fault here
    # costs the stamps alone and is stated.
    stamps_unread = False
    if graph_read:
        stated = _stated(matches, match_ids)
        stated += [i for i in _stated(section, section_ids) if i not in set(stated)]
        try:
            stamps = store.get_modifiers_map(stated)
        except DatabaseError:
            stamps_unread = True
        else:
            _stamp(matches, match_ids, stamps)
            _stamp(section, section_ids, stamps)
    return LexicalRead(
        matches=matches, match_ids=match_ids, section=section,
        item_counts={i: servable[i] for i in section_ids},
        graph_read=graph_read, stamps_unread=stamps_unread, unread_files=unread,
    )


def lexical_fallback_envelope(read: LexicalRead, *, reason: str) -> Dict[str, Any]:
    """Composes :func:`lexical_fallback`'s envelope from a lexical pass's decisions.

    ``{degraded, degraded_reason, matches, [stamps_unavailable], [stamps_unread],
    [unread_files], note}``; the note is the header, then the stamps-unavailable
    disclosure, the unread files and the no-match line when they apply. The one
    composition of that envelope, so the lexical envelope that also carries a
    section reads exactly as the store-less goldens pin it.
    """
    note = lexical_header(reason)
    envelope: Dict[str, Any] = {
        "degraded": "lexical",
        "degraded_reason": reason,
        "matches": read.matches,
    }
    if not read.graph_read:
        envelope["stamps_unavailable"] = True
        note = f"{note} {_STAMPS_UNAVAILABLE_DISCLOSURE}"
    if read.stamps_unread:
        envelope["stamps_unread"] = True
    if read.unread_files:
        envelope["unread_files"] = read.unread_files
        note = (f"{note} {len(read.unread_files)} corpus file(s) could not be read: "
                f"{', '.join(read.unread_files)}.")
    if not read.matches:
        note = f"{note} {_NO_MATCH_LINE}"
    envelope["note"] = note
    return envelope


def lexical_fallback(
    query: str,
    *,
    corpus_paths: Sequence[str],
    reason: str,
    store: Optional[Any] = None,
    limit: Optional[int] = None,
    full_top: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs the deterministic lexical fallback over the markdown corpus.

    Term-matches the query (case-insensitive, distinct terms ≥ 3 chars) against
    each entry's slug + axiom across every corpus file, ranks by number of distinct
    terms matched, then by file (newest first: the buffer, then archives newest
    quarter first), then by position in the file (newest first within the buffer
    and every rotation-written archive). A slug seen in two files is listed once,
    from the newer file — the buffer wins, as it does in the corpus fold. Honest by
    construction: no scores, no ``confidence`` key, an explicit ``degraded`` marker,
    and every file that could not be read named rather than skipped.

    Args:
        query: The claim/topic the caller was trying to recall.
        corpus_paths: The corpus files in ``divergence._corpus_files`` order —
            oldest archive first, buffer last. Reversed here, so no caller carries
            ordering logic. Keyword-only and plural so a stale single-path caller
            fails loudly.
        reason: The one-line cause phrase (see ``degraded_reason_from_error``).
        store: A readable ``GraphStore``, or None when the graph itself is down
            (pre-V1a). When given, it is read exactly twice per answer, whatever
            the match count: ``get_node_states`` over every ranked candidate's id
            (retired matches are dropped, live ones carry ``state``), then
            ``get_modifiers_map`` over the stated matches (their stamps). A
            ``DatabaseError`` from the first is treated as no store; one from the
            second costs the stamps alone and sets ``stamps_unread``. Any other
            exception propagates. When None, a stamps-unavailable disclosure rides
            the note instead.
        limit: Max matches to return, clamped as the ranked reads clamp it
            (``display.clamp_limit``): None ⇒ 5, any value ⇒ ``[1,
            RANKED_LIMIT_CEILING]``.
        full_top: How many top matches keep ``rejected_paths``, by position among
            the matches returned; the rest are axiom-only. None ⇒ every match whole,
            0 ⇒ none (the boundaries' ``brief``). The count and its note clause are
            the boundaries' to add — this leaf has no surface.

    Returns:
        ``{degraded: "lexical", degraded_reason, matches: [...], note}`` — each
        match a Letter-shaped dict (slug, axiom, scope, rejected_paths unless
        thinned by ``full_top``, matched_terms) plus state + modifier stamps when the graph is
        readable and holds the block. ``matched_terms`` omits ``DISPLAY_STOPWORDS``;
        ``stopwords_dropped`` counts the omitted terms and is present only when
        non-zero. ``stamps_unavailable: True`` is set when the graph is not readable
        (no store, or its state read failed), ``stamps_unread: True`` when only the
        stamps could not be read (the leaf adds no clause for it: the wording is
        per-surface, the composer's), and ``unread_files`` (basenames) only when some
        corpus file could not be read. The two stamp keys never co-occur.

    Raises:
        TypeError: When ``corpus_paths`` is a single string, which would otherwise
            iterate its characters as paths.
    """
    if isinstance(corpus_paths, (str, bytes)):
        raise TypeError(
            "lexical_fallback: corpus_paths is a sequence of corpus file paths, "
            "not one path string"
        )
    read = _read(query, corpus_paths=corpus_paths, store=store,
                 decision_cap=clamp_limit(limit), full_top=full_top, item_cap=None,
                 owners_whole=lambda _n: True, whole_owners=None)
    return lexical_fallback_envelope(read, reason=reason)


def lexical_degraded_read(
    query: str,
    *,
    corpus_paths: Sequence[str],
    store: Optional[Any],
    limit: Optional[int],
    full_top: Optional[int],
    bound: int,
    owners_whole: Callable[[int], bool],
) -> LexicalRead:
    """Matches decisions and rejected options in one pass, for a degraded answer.

    The lexical envelope's leaf: :func:`lexical_fallback`'s decision matches and,
    from the same corpus pass and the same two graph reads, a rejection section —
    each rejected **item** term-matched on its own text, at most ``bound`` of them,
    beside the decisions. A section owner among the decision matches (joined by
    content-hash id, unsynced blocks included) is ``window`` with its ``rank``;
    any other owner's first match is ``here`` with its Letter, later ones
    ``section``. Retired owners' items are dropped when states were read.

    When the graph is readable it is read exactly twice, whatever the match count:
    ``get_node_states`` over the union of decision candidates and item owners, then
    ``get_modifiers_map`` over the stated decisions and section owners (one stamp
    map for both lists). The fault rules are :func:`lexical_fallback`'s.

    Args:
        query: The claim/topic the caller was trying to recall.
        corpus_paths: The corpus files in ``divergence._corpus_files`` order.
        store: A readable ``GraphStore``, or None when the graph itself is down.
        limit: Max decision matches, clamped as the ranked reads clamp it.
        full_top: How many top decision matches keep ``rejected_paths``.
        bound: Max section items (``recall.REJECTION_SECTION_BOUND``, passed in:
            this leaf imports no ``recall``).
        owners_whole: Given the decision list's length, whether a ``here`` owner
            keeps its ``rejected_paths`` (``recall.rejection_owners_whole`` bound
            to ``full_top``).

    Returns:
        A :class:`LexicalRead`; compose its decisions with
        :func:`lexical_fallback_envelope`.

    Raises:
        TypeError: When ``corpus_paths`` is a single string.
    """
    if isinstance(corpus_paths, (str, bytes)):
        raise TypeError(
            "lexical_degraded_read: corpus_paths is a sequence of corpus file "
            "paths, not one path string"
        )
    return _read(query, corpus_paths=corpus_paths, store=store,
                 decision_cap=clamp_limit(limit), full_top=full_top, item_cap=bound,
                 owners_whole=owners_whole, whole_owners=None)


def lexical_window_section(
    query: str,
    *,
    corpus_paths: Sequence[str],
    store: Any,
    bound: int,
    owners_whole: bool,
    window_ids: Sequence[str],
    window_states: Mapping[str, str],
    window_stamps: Mapping[str, Mapping[str, List[str]]],
) -> LexicalRead:
    """Matches rejected options beside a list of decisions the caller already read.

    The scope dump's leaf: no decision term-match (the list is a graph read), only
    the items, at most ``bound``. An owner in ``window_ids`` is ``window`` with its
    ``rank``, its state and stamps taken from what the caller passed — never read.
    Every other owner is ``here`` / ``section`` and costs the reads: at most one
    ``get_node_states`` over those owners and at most one ``get_modifiers_map`` over
    the stated ones, each made only when its id list is non-empty. A
    ``DatabaseError`` from the state read leaves ``graph_read`` False (the caller
    decides what an unfiltered section means beside its list); one from the stamp
    read sets ``stamps_unread``.

    Args:
        query: The claim/topic the caller was trying to recall.
        corpus_paths: The corpus files in ``divergence._corpus_files`` order.
        store: The readable ``GraphStore`` the list came from.
        bound: Max section items.
        owners_whole: Whether a ``here`` owner keeps its ``rejected_paths``.
        window_ids: The list's node ids, in rank order.
        window_states: Each listed id's state, as the caller read it.
        window_stamps: Each listed id's modifier stamps, as the caller read them.

    Returns:
        A :class:`LexicalRead` with no decision matches.

    Raises:
        TypeError: When ``corpus_paths`` is a single string.
    """
    if isinstance(corpus_paths, (str, bytes)):
        raise TypeError(
            "lexical_window_section: corpus_paths is a sequence of corpus file "
            "paths, not one path string"
        )
    unread: List[str] = []
    _, item_ranked, servable = _scan(corpus_paths, _query_terms(query),
                                     decisions=False, items=True, unread=unread)
    window: Dict[str, int] = {}
    for rank, node_id in enumerate(window_ids, start=1):
        window.setdefault(node_id, rank)
    outside = [i for i in dict.fromkeys(t[6] for t in item_ranked) if i not in window]
    graph_read, states = True, {}
    if outside:
        graph_read, states = _read_states(store, outside)
    states = {**states, **{i: window_states[i] for i in window if i in window_states}}
    section = _shape_items(item_ranked, states, graph_read=graph_read, cap=bound,
                           window=window, owners_whole=owners_whole, whole_owners=None)
    section_ids = [m["id"] for m in section]

    stamps_unread = False
    _stamp(section, section_ids, window_stamps)
    stated = [i for i in _stated(section, section_ids) if i not in window]
    if graph_read and stated:
        try:
            stamps = store.get_modifiers_map(stated)
        except DatabaseError:
            stamps_unread = True
        else:
            _stamp(section, section_ids, stamps)
    return LexicalRead(
        matches=[], match_ids=[], section=section,
        item_counts={i: servable[i] for i in section_ids},
        graph_read=graph_read, stamps_unread=stamps_unread, unread_files=unread,
    )


def lexical_rejection_matches(
    query: str,
    *,
    corpus_paths: Sequence[str],
    store: Optional[Any],
    limit: Optional[int],
    full_top: Optional[int],
) -> Dict[str, Any]:
    """Term-matches the query against each rejected option in the markdown corpus.

    The lexical twin of the rejection ranking, for an answer that cannot rank by
    score: each entry's ``rejected_paths`` is enumerated
    (``rejections.enumerate_rejected_paths``) and each **item** is matched on its own
    text, never the raw field, with :func:`lexical_fallback`'s terms (casefold,
    distinct, at least ``LEXICAL_MIN_TERM_LEN`` characters). Only the items the
    rejection index could serve are candidates (``rejections.point_items``' ordinals:
    of two byte-identical items, the first), so this answer and the semantic one name
    the same items. Ranked by most distinct terms, then file (newest first), then
    position, then ordinal. The corpus is read through the loop
    :func:`lexical_fallback` reads it with.

    When the graph is readable it is read twice, whatever the match count:
    ``get_node_states`` over every candidate's owner — a retired owner's items are
    dropped, a live one's carry ``state``, an unsynced owner's stay without one —
    then ``get_modifiers_map`` over the stated owners shown. A ``DatabaseError`` from
    the first is treated as no store (nothing is dropped); one from the second costs
    the stamps alone.

    Surface-free: no header, no reason and no note. The lookup's lexical envelope
    composes those. The degraded sections of the two ranked verbs match items through
    the same steps (:func:`lexical_degraded_read`, :func:`lexical_window_section`),
    capped by the section bound instead of ``limit``.

    Args:
        query: The claim the caller asked about.
        corpus_paths: The corpus files in ``divergence._corpus_files`` order.
        store: A readable ``GraphStore``, or None when the graph itself is down.
        limit: Max matched items, clamped as the ranked reads clamp it.
        full_top: How many owners keep ``rejected_paths`` on their Letter, by first
            appearance (None ⇒ all, 0 ⇒ none). Matched items are never thinned.

    Returns:
        ``{matches, item_counts, graph_read, stamps_unread, unread_files}``. Each
        match carries ``slug``, ``id`` (the owner's content-hash id), ``state`` when
        known, ``ordinal``, ``item``, ``matched_terms`` (``stopwords_dropped`` when
        non-zero) and ``letter``: ``here`` on an owner's first match, followed by its
        Letter (``axiom``, ``scope``, ``rejected_paths`` unless thinned), or
        ``section`` on its later ones; then the owner's stamps when read. No
        ``score``. ``item_counts`` maps each matched owner to how many distinct items
        its field can serve.

    Raises:
        TypeError: When ``corpus_paths`` is a single string.
    """
    if isinstance(corpus_paths, (str, bytes)):
        raise TypeError(
            "lexical_rejection_matches: corpus_paths is a sequence of corpus file "
            "paths, not one path string"
        )
    # Here the section IS the answer: capped by `limit`, and `full_top` counts owners.
    read = _read(query, corpus_paths=corpus_paths, store=store, decision_cap=None,
                 full_top=None, item_cap=clamp_limit(limit),
                 owners_whole=lambda _n: True, whole_owners=full_top)
    return {
        "matches": read.section,
        "item_counts": read.item_counts,
        "graph_read": read.graph_read,
        "stamps_unread": read.stamps_unread,
        "unread_files": read.unread_files,
    }
