"""MCP Server implementation for Mitos.

This module implements the MCP Server (F) and the C4 integration contract,
exposing surface_decisions and query_decisions tools to LLM clients.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
from mcp.server.fastmcp import FastMCP

from mitos.display import blackout_note, clamp_limit, dumps_display, letter_payload, oneline_payload, order_scope_counts, show_payload, SHOW_NOT_FOUND_HINT
from mitos.config import MitosConfig
from mitos.store import GraphStore, MODIFIER_EDGE_KEYS
from mitos.embeddings import GeminiEmbeddingProvider
from mitos.vector_store import QdrantVectorStore
from mitos.lexical import degraded_reason_from_error, lexical_fallback
from mitos.recall import (assess_surface_recall, corpus_provenance,
                          scope_filter_recovery)

# Create FastMCP server instance
mcp = FastMCP("Mitos")

# No cross-call "seen" dedup state — deliberately. An earlier design cached
# already-surfaced slugs in a process-global set and trimmed `rejected_paths` (the
# relitigation-stopping field) from re-hits, flagging them `seen`. But `mitos serve`
# outlives a single agent session — the orchestrator `/clear`s and respawns the agent
# against the SAME long-running server — so the set leaked across session resets: a
# brand-new fresh-eyes session was handed `seen: true` with `rejected_paths` withheld,
# exactly the field it needed, with no way to tell it was being short-changed. No
# connection/session key is fully correct either (a bare `/clear` keeps the connection
# while resetting the agent's context), so the only correct shape is to hold no
# cross-call state at all. A caller who wants a lightweight scan passes `brief=True` —
# explicit, per-call, stateless. (V5 owns the rebuilt MCP server; carry this forward.)


def _attach_modifiers(payload: Dict[str, Any], node: Dict[str, Any], store: GraphStore) -> Dict[str, Any]:
    """Stamps reverse-relation modifier keys onto a decision payload, in place.

    Adds ``superseded_by`` / ``amended_by`` / ``narrowed_by`` / ``corrected_by``
    (only the non-empty ones) so a reader knows a later decision has moved on from
    this axiom and which one to chase — the fix for amended/narrowed nodes that stay
    ``active`` with their original (now-stale) mechanism text. Always applied,
    independent of ``brief``: the staleness flag matters even on an axiom-only scan,
    where ``rejected_paths`` is trimmed but the trap remains.
    Fail-silent — a modifier lookup error never breaks the recall response.

    Args:
        payload: The decision payload to augment.
        node: The store node dict (must carry ``id``).
        store: The graph store to read reverse edges from.

    Returns:
        The same payload dict, with any modifier keys added.
    """
    try:
        for key, slugs in store.get_modifiers(node["id"]).items():
            payload[key] = slugs
    except Exception:
        pass
    return payload


def _retired_handle(store: GraphStore, slug: str) -> Optional[Dict[str, Any]]:
    """Builds a retired-handle pointer for a superseded-filtered ranked match.

    The MCP twin of ``cli._retired_handle`` — kept independent per surface (the
    ranked loops collect retired handles separately, mirroring the deliberate
    payload-shaper asymmetry), but emitting the identical ``{"slug", "state"}`` (+
    ``superseded_by`` successor) shape so the blackout ``all_superseded`` field is
    byte-equal CLI⇄MCP (T5 parity). State is read authoritatively from the computed
    ``get_node_state`` via the state-agnostic ``resolve_slug`` (the vector payload's
    ``state`` is stale-at-embed-time). Calm degradation (P9): an unresolvable slug
    returns ``None`` (omitted by the caller), a failed state read falls back to
    ``"superseded"``.

    Args:
        store: The graph store to resolve the slug and read state/modifiers from.
        slug: The slug of the superseded-filtered match.

    Returns:
        The retired-handle dict, or ``None`` if the slug does not resolve.
    """
    try:
        node_ids = store.resolve_slug(slug)
    except Exception:
        return None
    if not node_ids:
        return None
    node_id = node_ids[0]
    try:
        state = store.get_node_state(node_id)
    except Exception:
        state = "superseded"
    handle: Dict[str, Any] = {"slug": slug, "state": state}
    try:
        successors = store.get_modifiers(node_id).get("superseded_by")
        if successors:
            handle["superseded_by"] = successors
    except Exception:
        pass
    return handle


def _decision_payload(node: Dict[str, Any], score: float, *, brief: bool,
                      store: Optional[GraphStore] = None) -> Dict[str, Any]:
    """Shapes a Letter-mode decision payload.

    ``rejected_paths`` — the heavy, high-value field whose reasoning stops
    relitigation — is always included unless ``brief`` (the caller explicitly asked
    for an axiom-only scan). There is deliberately no cross-call "seen" trimming; see
    the module-level note on why that state was removed.

    Args:
        node: A store node dict (``slug``, ``core_axiom``, ``rejected_paths``, ``scope``).
        score: The relevance score to attach.
        brief: Drop ``rejected_paths`` for an axiom-only scan.
        store: When given, reverse-relation modifier keys are stamped on (always,
            even for brief payloads).

    Returns:
        A Letter-mode decision dict.
    """
    payload = letter_payload(node, brief=brief, extras={"score": score})
    if store is not None:
        _attach_modifiers(payload, node, store)
    return payload

def get_workspace_components() -> Tuple[GraphStore, Optional[GeminiEmbeddingProvider], Optional[QdrantVectorStore]]:
    """Loads and returns the graph store (read-only), embedding provider, and vector store."""
    config = MitosConfig()
    store = GraphStore(config.db_path, read_only=True)
    
    embed_provider = None
    vector_store = None
    try:
        cache_path = os.path.join(config.mitos_dir, "embedding_cache.sqlite")
        embed_provider = GeminiEmbeddingProvider(cache_path)
        vector_store = QdrantVectorStore(config.qdrant_url, config.qdrant_collection)
    except Exception:
        pass
        
    return store, embed_provider, vector_store


def _lexical_degraded_response(query: str, *, reason: str,
                               store: Optional[GraphStore], brief: bool,
                               limit: int,
                               open_questions: Optional[List[Dict[str, Any]]] = None) -> str:
    """Builds the degraded lexical-fallback JSON for the MCP read tools.

    The MCP twin of ``cli._emit_lexical_degraded`` (ADR
    ``read-verbs-degrade-to-lexical-decisions-md-fallback``): the shared
    ``lexical_fallback`` runs the term-match over decisions.md, so the two
    surfaces cannot drift. The envelope carries ``degraded: "lexical"`` and a
    ``degraded_reason`` — never an ``{error}`` object or raw provider text.

    Args:
        query: The claim/topic the caller was trying to recall.
        reason: One-line cause phrase (see ``degraded_reason_from_error``).
        store: A readable graph store for active-filtering + modifier stamps,
            or None when the graph itself is down (pre-V1a).
        brief: Omit ``rejected_paths`` from each match.
        limit: Max matches to return.
        open_questions: An already-computed scoped parked-OQ list to carry on
            the envelope (present-if-scanned semantics — None means omitted).

    Returns:
        The degraded envelope as a JSON string.
    """
    envelope = lexical_fallback(
        query, MitosConfig().decisions_file, reason=reason, store=store,
        limit=limit, brief=brief,
    )
    envelope["query"] = query
    envelope.update(corpus_provenance(MitosConfig()))
    if open_questions is not None:
        envelope["open_questions"] = open_questions
    return dumps_display(envelope, ensure_ascii=False, indent=2)


def _oq_payload(oq: Dict[str, Any]) -> Dict[str, Any]:
    """Builds the open-question output sub-dict for the MCP visibility tools.

    Mirrors the CLI twins (``cmd_surface`` / ``cmd_list``): the
    ``{topic, questions_raised, park_reason}`` shape PLUS any reverse-relation
    modifier keys (``amended_by`` / ``narrowed_by``) already stamped on the OQ by
    ``get_open_questions``' 2b modifier chokepoint, read straight off the payload —
    so an amended-but-active OQ never reads as the final word, and the MCP surface
    stays behaviourally in sync with its CLI mirror (CLI⇄MCP parity).

    Args:
        oq: A hydrated, modifier-stamped open-question dict from
            ``get_open_questions``.

    Returns:
        The OQ output sub-dict, carrying present modifier keys when non-empty.
    """
    payload: Dict[str, Any] = {
        "topic": oq["slug"],
        "questions_raised": oq["questions_raised"],
        "park_reason": oq.get("park_reason"),
    }
    payload.update({key: oq[key] for key in MODIFIER_EDGE_KEYS.values() if oq.get(key)})
    return payload


@mcp.tool()
def surface_decisions(query: str, scope: Optional[str] = None, brief: bool = False, limit: int = 5) -> str:
    """Surface active precedents for a CLAIM before you decide — the recall loop, use first.

    The broad "is there a settled decision near this?" scan: a ranked, capped (top
    few) semantic match. Reach for this when deciding something; reach for
    query_decisions to look up a SPECIFIC slug or claim, and list_decisions for the
    EXHAUSTIVE set in a scope. Each returned precedent carries its `rejected_paths`
    (why alternatives were ruled out) — the field that actually stops relitigation.
    Every hit carries its full `rejected_paths` unless you pass `brief=True`. Closing
    the loop: after you decide, `record_decision` the outcome so the next agent
    inherits it instead of relitigating.

    Args:
        query: The semantic claim or topic string (e.g. 'cache strategy').
        scope: Optional scope hint — does NOT filter the semantic search. Recall is
            scope-blind by design, so a mis-guessed tag can't hide cross-scope
            precedent; scope only narrows the `open_questions` scan and shapes the
            recall `note` (incl. the "unused tag → valid scopes" redirect). For
            scope-RESTRICTED retrieval use list_decisions(scope=...) — the only
            surface that hard-filters by scope.
        brief: If True, omit `rejected_paths` from every result (axiom-only — a quick
            "is there anything nearby?" scan). Default False keeps the full reasoning.
        limit: Ranked top-k to retrieve (default 5; clamped to 1–50). Raise it to dig
            deeper, lower it to save context — a context-budget dial, not a cap at 5.

    Returns:
        A JSON string with `active_decisions` (ranked, Letter-mode), plus
        `open_questions` ONLY when a scope was given (absent = not scanned, [] = none
        parked in that scope). Each decision: slug, axiom, scope, score, and
        rejected_paths unless brief. A precedent a later decision has moved on from
        also carries the modifying slugs under
        `superseded_by`/`amended_by`/`narrowed_by`/`corrected_by` (always present when
        they apply, even on a brief scan) — chase those before treating its axiom as
        the current mechanism. Also includes `confidence` (`strong`/`weak`/`none` when
        semantic ranking ran) and a `note`: `weak` or `none` means no settled precedent
        on this claim — treat it as no-precedent and decide, or call
        list_decisions(scope=...) for a certain check (don't read weak neighbours as a
        settled decision). When ranked recall retrieved precedents but every one is
        superseded (a blackout), `active_decisions` stays empty and a sibling
        `all_superseded` list carries the retired handles (`slug`, `state`, and the live
        `superseded_by` successor when known) with the `note` naming them — that is a
        recoverable "it was settled before, go read the history", not a true miss.
    """
    top_k = clamp_limit(limit)
    # A pre-V1a graph raises at store construction — the graph is unusable, so
    # the lexical fallback parses decisions.md directly (no graph access).
    try:
        store, embed_provider, vector_store = get_workspace_components()
    except Exception as e:
        return _lexical_degraded_response(
            query, reason=degraded_reason_from_error(e), store=None,
            brief=brief, limit=top_k,
        )

    results: Dict[str, Any] = {"active_decisions": []}
    results.update(corpus_provenance(MitosConfig()))
    semantic_ran = False
    top_score: Optional[float] = None
    retired: List[Dict[str, Any]] = []
    degraded_error: Optional[Exception] = None

    # 1. Semantic search if embeddings and vector store are active
    if embed_provider and vector_store:
        try:
            # Generate query vector
            q_vector = embed_provider.get_embedding(query, is_query=True)
            matches = vector_store.query(q_vector, limit=top_k)
            semantic_ran = True

            for m in matches:
                slug = m["slug"]
                node = store.get_node_by_slug(slug)
                if not node:
                    handle = _retired_handle(store, slug)
                    if handle:
                        retired.append(handle)
                    continue

                # Verify computed active status in SQLite (M3 computed state is source-of-truth)
                node_state = store.get_node_state(node["id"])
                if node_state not in ("active", "drifted"):
                    # Stale vector reference — a retired handle for the blackout vector.
                    handle = _retired_handle(store, slug)
                    if handle:
                        retired.append(handle)
                    continue

                results["active_decisions"].append(
                    _decision_payload(node, m["score"], brief=brief, store=store)
                )
                if top_score is None or m["score"] > top_score:
                    top_score = m["score"]
        except Exception as e:
            # Degrade to exact/scope filtering only
            semantic_ran = False
            degraded_error = e

    # 2. Scope pre-filtering fallback — ONLY when semantic recall is down (degraded).
    #    When semantic ran and simply found nothing, do NOT dump an unranked scope
    #    listing dressed as matches — that's the false-precedent ambiguity P5 closes.
    if not semantic_ran and not results["active_decisions"] and scope:
        try:
            active_decs = store.get_active_decisions(scope=scope)
            for d in active_decs[:5]:
                results["active_decisions"].append(
                    _decision_payload(d, 1.0, brief=brief, store=store)
                )
        except Exception:
            pass

    # 3. Append Open Questions ONLY when a scope was given (C4 resolves clause).
    #    Omitting the key when no scope disambiguates "not scanned" from "none here".
    if scope:
        open_questions = []
        try:
            for q in store.get_open_questions(scope=scope):
                if q["state"] == "parked":
                    open_questions.append(_oq_payload(q))
        except Exception:
            pass
        results["open_questions"] = open_questions

    # Degraded and empty-handed on decisions: route into the deterministic
    # lexical fallback (ADR read-verbs-degrade-to-lexical-decisions-md-fallback)
    # instead of the self-contradicting clean-empty result + unavailable note.
    # The scoped open-questions scan (a pure graph read that survived) rides
    # along on the degraded envelope.
    if not semantic_ran and not results["active_decisions"]:
        return _lexical_degraded_response(
            query, reason=degraded_reason_from_error(degraded_error),
            store=store, brief=brief, limit=top_k,
            open_questions=results.get("open_questions"),
        )

    # Confidence signal — let the agent tell a settled precedent from loose neighbours
    # or genuine absence, instead of a boilerplate note that read the same every time
    # (AX P5). Pass the live scope-count map (busiest-first) when a scope is given: it is
    # the unused-scope oracle (a tag absent from it gets a bounded self-correction
    # vector) and the did-you-mean / top-K source. Calm-degrade to None on error.
    scope_counts: Optional[Dict[str, Dict[str, int]]] = None
    if scope:
        try:
            scope_counts = order_scope_counts(store.get_scope_counts())
        except Exception:
            pass

    confidence, note = assess_surface_recall(
        semantic_ran=semantic_ran,
        top_score=top_score,
        result_count=len(results["active_decisions"]),
        scope=scope,
        scope_counts=scope_counts,
        surface="mcp",
    )
    if confidence is not None:
        results["confidence"] = confidence
    results["note"] = note

    # Blackout: semantic ranking ran and retrieved precedents, but every one was
    # superseded-filtered. Override the note with the recovery vector and attach the
    # retired handles (CLI⇄MCP-identical shape, T5 parity). Distinct from a true miss
    # (where `retired` is empty); fires regardless of any parked open questions.
    if semantic_ran and not results["active_decisions"] and retired:
        results["note"] = blackout_note(retired)
        results["all_superseded"] = retired

    return dumps_display(results, ensure_ascii=False, indent=2)


@mcp.tool()
def list_decisions(scope: Optional[str] = None, state: str = "active", brief: bool = False,
                   oneline: bool = False) -> str:
    """Enumerate the COMPLETE set of decisions (optionally scope-filtered) — no ranking, no top-k.

    surface_decisions / query_decisions are SEMANTIC and capped at the top few
    matches: ideal for "is there precedent for this claim?", but they cannot tell
    you whether you have seen *everything*. When you need certainty — a completeness
    pass over a scope, an audit, "show me every settled call in `auth`" — use this.
    It returns every matching decision deterministically, straight from the graph,
    so nothing hides below a relevance cliff. Needs no API key or Qdrant (pure graph
    read), so it works even when semantic recall is degraded. This is also the ONLY
    retrieval surface that hard-filters by scope — surface/query are scope-blind, so
    when you want results restricted to a scope, this is the verb.

    Args:
        scope: Optional scope tag filter (e.g. 'auth') — a true hard filter (this is
            the only retrieval surface that restricts by scope). Omit for the whole
            project.
        state: 'active' (default) returns the live set (active + drifted); 'all'
            returns every decision regardless of state (including superseded); any
            other value is an exact computed-state match (e.g. 'superseded').
        brief: If True, omit `rejected_paths` from every decision (axiom-only). Useful
            here — an exhaustive scope can otherwise return many full reasoning walls.
        oneline: If True, return the orientation/table-of-contents tier: one minimal
            object per decision — {slug, axiom_oneline (word-boundary-truncated),
            state} plus modifier slugs when present. For big scopes where even
            brief=True blows the result ceiling (measured: a 45-decision scope did) —
            scan the map here, then dereference the few that matter with query/show.
            Letter-complete stays the default depth; this is an explicit opt-down,
            never a default. Mutually exclusive with brief.

    Returns:
        A JSON string: {decisions, open_questions, total, scope, state}. Each
        decision carries the same Letter-mode shape as surface_decisions (slug,
        axiom, rejected_paths, scope) plus its computed `state`, and — when a later
        decision modifies it — `superseded_by`/`amended_by`/`narrowed_by`/
        `corrected_by` modifier slugs (the stamps survive every thinner tier,
        including oneline). UNBOUNDED.
    """
    if brief and oneline:
        return dumps_display(
            {"error": "brief and oneline are mutually exclusive — pick one depth tier."},
            ensure_ascii=False, indent=None)

    store, _embed, _vec = get_workspace_components()

    nodes = store.get_decisions(scope=scope, state=state)
    modifiers = store.get_modifiers_map([n["id"] for n in nodes])
    decisions = []
    for n in nodes:
        # oneline swaps the Letter core for the minimal {slug, axiom_oneline, state}
        # object (same shape as the CLI's `list --oneline --json` — parity seam in
        # display.oneline_payload); modifier stamps ride either shape.
        if oneline:
            d = oneline_payload(n)
        else:
            d = letter_payload(n, brief=brief, extras={"state": n["computed_state"]})
        d.update(modifiers.get(n["id"], {}))
        decisions.append(d)

    open_questions = []
    try:
        for oq in store.get_open_questions(scope=scope):
            if oq["state"] == "parked":
                open_questions.append(_oq_payload(oq))
    except Exception:
        pass

    payload = {
        "decisions": decisions,
        "open_questions": open_questions,
        "total": len(decisions),
        "scope": scope,
        "state": state,
        **corpus_provenance(MitosConfig()),
    }

    # On an empty scoped read, distinguish a genuinely-fresh scope from a misspelled one:
    # an absent-from-live scope rides two additive, in-band fields (never an error object
    # or non-zero exit — an LLM agent reads those as a call-syntax fault and thrashes).
    # Only the miss path pays the get_scope_counts() read. The recovery payload carries no
    # node id, so there is nothing to modifier-stamp here.
    if scope and not decisions and not open_questions:
        scope_counts: Optional[Dict[str, Dict[str, int]]] = None
        try:
            scope_counts = order_scope_counts(store.get_scope_counts())
        except Exception:
            pass
        recovery = scope_filter_recovery(
            scope=scope, scope_counts=scope_counts, surface="mcp"
        )
        if recovery:
            payload["scope_known"] = False
            payload["scope_recovery"] = recovery["note"]

    return dumps_display(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def list_scopes(include_archived: bool = False) -> str:
    """List the project's scope-tag vocabulary with each domain's live-node counts.

    The map an agent reads BEFORE recording or recalling: every scope tag that
    carries a live node, ranked busiest-domain-first (total active decisions +
    parked open questions, descending; ties alphabetical). record_decision /
    surface_decisions / query_decisions / list_decisions all let you write into a
    scope or read from one — but only this tells you *what scopes exist and how
    alive each is*, so you can pick the project's real vocabulary instead of
    inventing a near-duplicate tag. A pure graph read — no API key or Qdrant needed,
    so it works even when semantic recall is degraded.

    This returns a tag→counts AGGREGATE, not decision payloads: there is no node id
    to stamp, so — unlike surface/query/list_decisions — it carries no
    `superseded_by`/`amended_by`/… modifier keys (that is correct, not a missing
    stamp). An empty/fresh project returns `{}` — a valid empty vocabulary, never an
    error.

    Args:
        include_archived: When False (default), returns only live domains (≥1 active
            decision OR ≥1 parked open question). When True, additionally includes
            every other scope tag present in the graph at a `{active_decisions: 0,
            parked_open_questions: 0}` floor — the scope-level parallel of
            list_decisions(state="all").

    Returns:
        A JSON string: an ordered map `{scope: {active_decisions, parked_open_questions}}`,
        busiest domain first. The key order IS the deliverable — iterate it as-is.
    """
    store, _embed, _vec = get_workspace_components()
    return dumps_display(
        order_scope_counts(store.get_scope_counts(include_archived=include_archived)),
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def show_node(ident: str) -> str:
    """Dereference ONE decision or open question by exact handle — slug OR content-hash id.

    The exact-handle lookup that reaches the graveyard: it resolves a node
    state-agnostically (active-first, else the most-recent superseded node in the
    casefolded-slug lineage), so it answers for a SUPERSEDED node that
    query_decisions' slug branch — active-view-only — cannot reach. Use it to
    reconstruct *why* a now-retired call was made (don't relitigate a settled
    rejection). Not a search: pass the precise slug or id you already hold, not a
    claim (for ranked recall use query_decisions / surface_decisions).

    Args:
        ident: A content-hash id or a slug (case-insensitive) — the exact handle.

    Returns:
        A JSON string. A found **decision** is a Letter-complete object (`axiom` +
        `rejected_paths`) with `kind`/`id`/`slug`/`scope`/`state`; a found **open
        question** carries `topic`/`questions_raised`/`park_reason`. Both stamp the
        present reverse-relation modifier keys — a superseded node names its
        `superseded_by`, an amended one its `amended_by`/`narrowed_by` — so a
        moved-on node never reads as the final word. A genuinely-absent handle
        returns `{found: false, ident, hint}` (never an error), the hint pointing
        at `mitos sync` for an authored-but-unsynced draft.
    """
    store, _embed, _vec = get_workspace_components()

    # State-agnostic resolution via the SHARED 5a seam — the identical method
    # cmd_show calls, so the resolution selection cannot drift between surfaces.
    # A genuine MI-13 breach raises ValidationError out of resolve_handle; we do
    # NOT swallow it into not-found (a breach is not "not found").
    node = store.resolve_handle(ident)
    if not node:
        return dumps_display(
            {"found": False, "ident": ident, "hint": SHOW_NOT_FOUND_HINT},
            ensure_ascii=False,
            indent=2,
        )

    # state from the separate computed-state read (never node.get("state") —
    # absent on the resolved dict); modifiers are the one kind-agnostic stamp
    # source. Stamping is LOAD-BEARING: surfacing the superseded is this tool's
    # whole job, so the superseded_by stamp is not decoration.
    state = store.get_node_state(node["id"])
    modifiers = store.get_modifiers(node["id"])
    payload = show_payload(node, state=state, modifiers=modifiers)
    return dumps_display(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def query_decisions(query: str, depth: str = "letter", brief: bool = False, limit: int = 5) -> str:
    """Look up a SPECIFIC decision by slug or claim — the targeted lookup.

    Use this when you know roughly what you're after (a slug you're carrying, or a
    pointed claim). For the broad "is there precedent near this?" scan before
    deciding, use surface_decisions; for the EXHAUSTIVE set in a scope, list_decisions.
    If query matches a unique slug exactly, returns that one decision (full); otherwise
    a ranked semantic search for the claim. Its slug branch is active-view-only — to
    dereference an EXACT handle including a superseded node it can't reach, use show_node.
    Once you decide, `record_decision` the outcome so the next agent inherits it.

    Args:
        query: Unique decision slug identifier OR a semantic claim search query.
        depth: The retrieval depth (e.g. 'letter', 'trace', 'vibe'). v0.1 enforces Letter mode.
        brief: If True, omit `rejected_paths` from ranked semantic matches (axiom-only).
            An exact-slug hit is always returned in full (you asked for that one).
        limit: Ranked top-k for the SEMANTIC branch (default 5; clamped to 1–50). Raise
            it to dig deeper, lower it to save context. Ignored by an exact-slug hit
            (that returns the one decision you named).

    Returns:
        A JSON string containing the ranked results in Letter-mode payload shape.
        A decision a later one has moved on from also carries `superseded_by`/
        `amended_by`/`narrowed_by`/`corrected_by` (the modifying slugs) — an exact-slug
        hit on an amended decision still reads `state: "active"`, so chase these before
        trusting its axiom's mechanism. When the semantic branch retrieved precedents but
        every one is superseded (a blackout), `matches` stays empty and a sibling
        `all_superseded` list carries the retired handles (`slug`, `state`, live
        `superseded_by` when known) — settled before, not a true miss; read the history
        with list_decisions(state="all").
    """
    if depth != "letter":
        return dumps_display({"error": f"Depth mode '{depth}' is not yet implemented in v0.1 (Letter-only retrieval)."}, ensure_ascii=False, indent=None)

    # A pre-V1a graph raises at store construction — the graph is unusable, so
    # the lexical fallback parses decisions.md directly (no graph access).
    try:
        store, embed_provider, vector_store = get_workspace_components()
    except Exception as e:
        return _lexical_degraded_response(
            query, reason=degraded_reason_from_error(e), store=None,
            brief=brief, limit=clamp_limit(limit),
        )

    # 1. Try resolving query as direct slug first
    try:
        node = store.get_node_by_slug(query)
        if node:
            state = store.get_node_state(node["id"])

            output = {
                "slug": node["slug"],
                "axiom": node["core_axiom"],
                "rejected_paths": node["rejected_paths"],
                "scope": node["scope"],
                "state": state,
                "depth_mode": "letter"
            }
            output.update(store.get_modifiers(node["id"]))
            return dumps_display(output, ensure_ascii=False, indent=2)
    except Exception:
        # Not a slug collision or lookup failed; proceed to semantic claim lookup
        pass

    # 2. Perform ranked semantic claim search
    if embed_provider and vector_store:
        try:
            top_k = clamp_limit(limit)
            q_vector = embed_provider.get_embedding(query, is_query=True)
            matches = vector_store.query(q_vector, limit=top_k)

            output_list = []
            retired: List[Dict[str, Any]] = []
            for m in matches:
                slug = m["slug"]
                node = store.get_node_by_slug(slug)
                if not node:
                    handle = _retired_handle(store, slug)
                    if handle:
                        retired.append(handle)
                    continue

                node_state = store.get_node_state(node["id"])
                if node_state not in ("active", "drifted"):
                    handle = _retired_handle(store, slug)
                    if handle:
                        retired.append(handle)
                    continue

                match = letter_payload(
                    node,
                    brief=brief,
                    extras={"state": node_state, "score": m["score"], "depth_mode": "letter"},
                )
                match.update(store.get_modifiers(node["id"]))
                output_list.append(match)

            # Blackout: retrieved precedents but every one superseded-filtered.
            # Add the retired handles so the agent gets a pointer, not a false miss
            # (CLI⇄MCP-identical `all_superseded` shape, T5 parity).
            envelope: Dict[str, Any] = {"query": query, "depth_mode": "letter", "matches": output_list}
            envelope.update(corpus_provenance(MitosConfig()))
            if not output_list and retired:
                envelope["all_superseded"] = retired
            return dumps_display(envelope, ensure_ascii=False, indent=2)
        except Exception as e:
            # Embedding/Qdrant failure mid-query (e.g. a 429): never the raw
            # provider blob — the deterministic lexical fallback instead.
            return _lexical_degraded_response(
                query, reason=degraded_reason_from_error(e), store=store,
                brief=brief, limit=clamp_limit(limit),
            )

    # No embedding provider / vector store wired at all — degrade lexically.
    return _lexical_degraded_response(
        query, reason=degraded_reason_from_error(None), store=store,
        brief=brief, limit=clamp_limit(limit),
    )


@mcp.tool()
def record_decision(axiom: str, rejected_paths: str, scope: List[str], slug: str,
                    mechanisms: Optional[List[str]] = None, context: Optional[str] = None,
                    supersedes: Optional[str] = None, corrects: Optional[str] = None,
                    amends: Optional[str] = None,
                    narrows: Optional[str] = None, depends_on: Optional[str] = None,
                    resolves: Optional[str] = None, contradicts: Optional[str] = None,
                    derives_from: Optional[str] = None, cites: Optional[str] = None,
                    acknowledge_neighbors: bool = False) -> str:
    """Record a decision you just made, with the alternatives you rejected and why,
    so future sessions and other agents inherit it instead of relitigating it.

    Call this the moment you commit to a foundational choice — a schema, a library,
    a pattern, or a path you've decided to abandon. `rejected_paths` is required:
    recording WHY you ruled options out is what stops you (or the next agent) from
    re-proposing them. If this decision relates to an earlier one, look the earlier
    one up first with query_decisions/surface_decisions and pass its EXACT slug to the
    matching relation arg below (each is validated to point at a real decision). Each
    relation arg also accepts a comma-separated list to link several at once
    (e.g. supersedes="a, b").
    Returns the decision's slug; look it up afterwards with query_decisions.

    Args:
        axiom: The decision as a single clear sentence true going forward.
        rejected_paths: The alternatives considered and rejected, and why. REQUIRED.
        scope: Area tags, e.g. ["database", "auth"].
        mechanisms: Concrete technologies/entities involved, e.g. ["sqlite", "wal-mode"].
        context: Optional background on why this was decided.
        supersedes: Exact slug of a prior decision this one REPLACES (the old one
            becomes superseded). Use this for decision evolution.
        corrects: Exact slug of a prior decision this one CORRECTS (an in-buffer
            correction — the old one leaves the active view, like supersedes; use
            this when the earlier decision was wrong rather than outgrown).
        amends: Exact slug of a decision this one amends (modifies without replacing).
        narrows: Exact slug of a decision this one narrows the scope of.
        depends_on: Exact slug of a decision this one depends on.
        resolves: Exact slug of an open question this one resolves (the resolves edge is decision→open_question only).
        contradicts: Exact slug of a decision this one is in tension with.
        derives_from: Not valid when recording a decision — a derives_from edge
            originates from an open question (open_question -> decision), so a
            decision cannot be its source. Use cites to link a decision this one
            builds on.
        cites: Exact slug of a decision this one cites.
        slug: The short, descriptive handle for the decision (e.g. 'sqlite-wal-mode').
            Keep it to at most 100 characters — the slug is the permanent citation
            handle, so an over-length one is rejected (not silently truncated).
        acknowledge_neighbors: Record past the near-duplicate review (the decision is genuinely independent). you have looked at the flagged neighbour(s) and this decision is
            genuinely independent. Leave False (default) on the first attempt.

    Returns:
        A JSON string: {slug, id, state, embedding, status} or {error, code}.
        status="created" means newly recorded; status="exists" is a SUCCESS — the
        identical decision was already recorded and is now confirmed present, not an
        error and not something to retry. Only a top-level {error, code} is a failure.
        status="needs_review" (code "similar_decision_exists") is a PAUSE, not a failure
        and not a write: your decision is ≥0.80 similar to existing `neighbors` you did
        not reference. Inspect them — if this amends/supersedes/contradicts/cites one,
        re-record with that relation arg pointing at the neighbour's slug (`possible_tension`
        on a neighbour flags a likely contradiction, not a duplicate); if it is genuinely
        independent, re-record with acknowledge_neighbors=True. Nothing was written, so a
        re-record is the right move here (unlike an "exists" no-op).
        The "created" result also carries `edges_created` — the relation edges this
        record actually wired, each `{kind, target}` (write facts read back from the
        committed graph, so an empty list means no edge landed) — and the resolved
        `scope`/`mechanisms` as committed.
        On the "created" path the result MAY include `related`: the nearest existing
        live decisions to the one you just recorded — a write-time adjacency hint, so
        you notice an adjacent or contradictory prior decision. If one is genuinely
        related, record the link (re-record with the matching relation arg, or capture
        a follow-up). NOTE: identity is (slug + axiom + mechanisms). Re-recording an
        existing decision is a no-op — a changed `context`/`rejected_paths`/`scope` or
        relation on a re-record is NOT saved. To record different reasoning or a new
        relationship, make a NEW decision (a distinct axiom), don't resubmit the old one.
        The result MAY also carry `scope_overflow`: a one-line, debounced (≤once/24h)
        health nudge that the generated context files have grown past their size ceiling
        — not an error and not about this decision; run `mitos status` for the breakdown.
    """
    # Build our own writable manager — do NOT reuse get_workspace_components()
    # (it opens a read_only=True store). Workspace resolves from cwd, like the read tools.
    from mitos.sync import MitosSyncManager
    config = MitosConfig()
    manager = MitosSyncManager(config)
    result = manager.record_decision_entry(
        axiom=axiom,
        rejected_paths=rejected_paths,
        scope=scope,
        mechanisms=mechanisms,
        context=context,
        supersedes=supersedes,
        corrects=corrects,
        amends=amends,
        narrows=narrows,
        depends_on=depends_on,
        resolves=resolves,
        contradicts=contradicts,
        derives_from=derives_from,
        cites=cites,
        slug=slug,
        acknowledge_neighbors=acknowledge_neighbors,
    )
    return dumps_display(result, ensure_ascii=False, indent=None)
