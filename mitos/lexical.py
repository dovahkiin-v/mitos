"""Deterministic lexical fallback for the semantic read verbs.

When semantic recall or the graph is unavailable *for any reason* — embedding
provider errors (a 429 mid-loop), Qdrant down or the collection missing, or a
pre-V1a graph that takes SQLite reads down with it — ``mitos surface`` and
``mitos query`` degrade to a case-insensitive term-match over the entries of the
markdown corpus — ``decisions.md`` and ``decisions/archive/`` — (slug + axiom)
instead of dead-ending (ADR
``read-verbs-degrade-to-lexical-decisions-md-fallback``). The fallback is a
grep and is presented as one: an explicit degraded header, no similarity
scores, no ``confidence`` field, and a ``degraded: "lexical"`` marker on the
JSON envelope.

The scan covers the whole corpus, not only the buffer: rotation moves committed
entries into the archives, so a buffer-only grep would answer "no matches" over
the decisions that moved. A corpus file that exists but cannot be read is named in
the envelope (``unread_files``) and in the note, never skipped quietly.

Factored here once so the CLI verbs and their MCP twins share the identical
behaviour (CLI⇄MCP parity is a standing contract, not per-surface luck).

Stamping: when the graph is readable the matches are filtered to active
entries and stamped via ``GraphStore.get_modifiers`` (the every-read-surface
rule); when it is not (the pre-V1a case), a one-line disclosure replaces the
stamps — entries come straight from markdown and may include superseded ones.
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Set

from mitos.parser import parse_decisions_file

# Terms shorter than this are dropped from the query before matching — they are
# stop-word noise ("a", "of", "to") that would match nearly every entry.
LEXICAL_MIN_TERM_LEN: int = 3

# Default result cap when the caller passes no limit. Mirrors the spirit of the
# ranked verbs' clamp without importing their ceiling — a grep needs no 50-row
# mode by default.
LEXICAL_DEFAULT_LIMIT: int = 10

_STAMPS_UNAVAILABLE_DISCLOSURE = (
    "Graph unavailable — state/modifier stamps not applied; entries come "
    "straight from the markdown corpus and may include superseded ones."
)

_NO_MATCH_LINE = (
    "No lexical matches either — grep the workspace's `decisions.md` and "
    "`decisions/archive/` to be sure."
)


def degraded_reason_from_error(exc: Optional[BaseException]) -> str:
    """Classifies a recall failure into one calm human-readable cause.

    Never returns the raw provider blob — a Gemini 429 dumps a JSON error body
    into ``str(exc)``, and re-printing that is exactly the AX failure the
    fallback exists to replace.

    Args:
        exc: The exception that broke semantic recall, or None when recall was
            never attempted (no provider/vector store wired).

    Returns:
        A short cause phrase for the degraded header.
    """
    if exc is None:
        return "embeddings/Qdrant unavailable"
    text = str(exc)
    if "predates the V1a schema" in text:
        return "graph predates the V1a schema — run `mitos cutover`"
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return "embedding provider rate-limited (429)"
    # Late import avoided — errors.py is a leaf, safe at module level, but the
    # string checks above must win first (an EmbeddingError often wraps a 429).
    from mitos.errors import CollectionMissingError, EmbeddingError, VectorStoreError

    if isinstance(exc, EmbeddingError):
        return "embedding provider error"
    # Before the VectorStoreError arm, which it subclasses — otherwise the most-used
    # read verb answers a missing collection with "Qdrant unavailable", the exact
    # blame-the-infrastructure phrase the typed error exists to replace.
    if isinstance(exc, CollectionMissingError):
        named = f" '{exc.collection}'" if exc.collection else ""
        return f"vector collection{named} missing — run `mitos reconcile`"
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


def _query_terms(query: str) -> List[str]:
    """Splits a query into distinct casefolded match terms (len ≥ 3)."""
    seen: List[str] = []
    for token in query.split():
        term = token.strip(".,;:!?'\"()[]{}").casefold()
        if len(term) >= LEXICAL_MIN_TERM_LEN and term not in seen:
            seen.append(term)
    return seen


def lexical_fallback(
    query: str,
    *,
    corpus_paths: Sequence[str],
    reason: str,
    store: Optional[Any] = None,
    limit: Optional[int] = None,
    brief: bool = False,
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
            (pre-V1a). When given, superseded entries are filtered out and each
            match is modifier-stamped; when None, a stamps-unavailable
            disclosure rides the note instead.
        limit: Max matches to return; None ⇒ ``LEXICAL_DEFAULT_LIMIT``.
        brief: Omit ``rejected_paths`` from each match.

    Returns:
        ``{degraded: "lexical", degraded_reason, matches: [...], note}`` — each
        match a Letter-shaped dict (slug, axiom, scope, rejected_paths unless
        brief, matched_terms) plus state + modifier stamps when the graph is
        readable. ``stamps_unavailable: True`` is set when it is not, and
        ``unread_files`` (basenames) only when some corpus file could not be read.

    Raises:
        TypeError: When ``corpus_paths`` is a single string, which would otherwise
            iterate its characters as paths.
    """
    if isinstance(corpus_paths, (str, bytes)):
        raise TypeError(
            "lexical_fallback: corpus_paths is a sequence of corpus file paths, "
            "not one path string"
        )
    cap = limit if limit and limit > 0 else LEXICAL_DEFAULT_LIMIT
    terms = _query_terms(query)

    ranked: List[Any] = []
    unread: List[str] = []
    seen_slugs: Set[str] = set()
    for file_rank, path in enumerate(reversed(list(corpus_paths))):
        # Read AND parse per file, one guard: a file that exists and cannot be
        # read, decoded or parsed is disclosed, never answered around in silence.
        # A file gone since the listing is simply not part of the corpus now.
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
            entries = parse_decisions_file(text, errors=[]) if text else []
        except FileNotFoundError:
            continue
        except Exception:
            unread.append(os.path.basename(path))
            continue
        for position, entry in enumerate(entries):
            if entry.kind != "decision":
                continue
            # Dedup by handle only — an entry with no slug has nothing to be a
            # duplicate of, and must not shadow another slugless entry.
            key = (entry.slug or "").casefold()
            if key and key in seen_slugs:
                continue
            seen_slugs.add(key)
            axiom = entry.core_axiom or entry.axiom or ""
            haystack = f"{entry.slug}\n{axiom}".casefold()
            hit = [t for t in terms if t in haystack]
            if hit:
                ranked.append((len(hit), file_rank, position, hit, entry))
    # Most distinct terms first; ties by file (newest first), then by position —
    # every buffer and rotation-written archive is newest-first below its marker.
    ranked.sort(key=lambda t: (-t[0], t[1], t[2]))

    matches: List[Dict[str, Any]] = []
    for _count, _file_rank, _position, hit, entry in ranked:
        payload: Dict[str, Any] = {
            "slug": entry.slug,
            "axiom": entry.core_axiom or entry.axiom or "",
            "scope": entry.scope,
            "matched_terms": hit,
        }
        if not brief:
            payload["rejected_paths"] = entry.rejected_paths
        if store is not None:
            # Graph readable: filter to active, stamp modifiers. An entry the
            # graph doesn't know (authored-but-unsynced) stays, unstamped —
            # dropping the gold source on a graph miss would be a false empty.
            # resolve_slug (state-agnostic) rather than the active-view
            # get_node_by_slug, so a superseded entry is seen and filtered.
            node_id = None
            try:
                node_ids = store.resolve_slug(entry.slug)
                node_id = node_ids[0] if node_ids else None
            except Exception:
                node_id = None
            if node_id:
                try:
                    state = store.get_node_state(node_id)
                except Exception:
                    state = None
                if state is not None and state not in ("active", "drifted"):
                    continue
                if state is not None:
                    payload["state"] = state
                try:
                    payload.update(store.get_modifiers(node_id))
                except Exception:
                    pass
        matches.append(payload)
        if len(matches) >= cap:
            break

    note = lexical_header(reason)
    envelope: Dict[str, Any] = {
        "degraded": "lexical",
        "degraded_reason": reason,
        "matches": matches,
    }
    if store is None:
        envelope["stamps_unavailable"] = True
        note = f"{note} {_STAMPS_UNAVAILABLE_DISCLOSURE}"
    if unread:
        envelope["unread_files"] = unread
        note = (f"{note} {len(unread)} corpus file(s) could not be read: "
                f"{', '.join(unread)}.")
    if not matches:
        note = f"{note} {_NO_MATCH_LINE}"
    envelope["note"] = note
    return envelope
