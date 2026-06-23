"""MCP Server implementation for Mitos.

This module implements the MCP Server (F) and the C4 integration contract,
exposing surface_decisions and query_decisions tools to LLM clients.
"""

import os
import json
from typing import Optional, List, Dict, Any, Tuple
from mcp.server.fastmcp import FastMCP

from mitos.config import MitosConfig
from mitos.store import GraphStore, MODIFIER_EDGE_KEYS
from mitos.embeddings import GeminiEmbeddingProvider
from mitos.vector_store import QdrantVectorStore
from mitos.recall import assess_surface_recall

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
    payload: Dict[str, Any] = {
        "slug": node["slug"],
        "axiom": node["core_axiom"],
        "scope": node["scope"],
        "score": score,
    }
    if not brief:
        payload["rejected_paths"] = node["rejected_paths"]
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
def surface_decisions(query: str, scope: Optional[str] = None, brief: bool = False) -> str:
    """Surface active precedents for a CLAIM before you decide — the recall loop, use first.

    The broad "is there a settled decision near this?" scan: a ranked, capped (top
    few) semantic match. Reach for this when deciding something; reach for
    query_decisions to look up a SPECIFIC slug or claim, and list_decisions for the
    EXHAUSTIVE set in a scope. Each returned precedent carries its `rejected_paths`
    (why alternatives were ruled out) — the field that actually stops relitigation.
    Every hit carries its full `rejected_paths` unless you pass `brief=True`.

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
        settled decision).
    """
    store, embed_provider, vector_store = get_workspace_components()

    results: Dict[str, Any] = {"active_decisions": []}
    semantic_ran = False
    top_score: Optional[float] = None

    # 1. Semantic search if embeddings and vector store are active
    if embed_provider and vector_store:
        try:
            # Generate query vector
            q_vector = embed_provider.get_embedding(query, is_query=True)
            matches = vector_store.query(q_vector, limit=5)
            semantic_ran = True

            for m in matches:
                slug = m["slug"]
                node = store.get_node_by_slug(slug)
                if not node:
                    continue

                # Verify computed active status in SQLite (M3 computed state is source-of-truth)
                node_state = store.get_node_state(node["id"])
                if node_state not in ("active", "drifted"):
                    # Stale vector reference, skip
                    continue

                results["active_decisions"].append(
                    _decision_payload(node, m["score"], brief=brief, store=store)
                )
                if top_score is None or m["score"] > top_score:
                    top_score = m["score"]
        except Exception as e:
            # Degrade to exact/scope filtering only
            semantic_ran = False

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

    # Confidence signal — let the agent tell a settled precedent from loose neighbours
    # or genuine absence, instead of a boilerplate note that read the same every time
    # (AX P5). Compute the scope's active-decision count only when it disambiguates an
    # empty result ("tag unused" vs "populated but nothing matched").
    scope_decision_count: Optional[int] = None
    all_scopes: Optional[List[str]] = None
    if scope:
        try:
            scope_decision_count = len(store.get_active_decisions(scope=scope))
            if scope_decision_count == 0:
                all_scopes = store.get_all_scopes()
        except Exception:
            pass

    confidence, note = assess_surface_recall(
        semantic_ran=semantic_ran,
        top_score=top_score,
        result_count=len(results["active_decisions"]),
        scope=scope,
        scope_decision_count=scope_decision_count,
        all_scopes=all_scopes,
    )
    if confidence is not None:
        results["confidence"] = confidence
    results["note"] = note

    return json.dumps(results, indent=2)


@mcp.tool()
def list_decisions(scope: Optional[str] = None, state: str = "active", brief: bool = False) -> str:
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

    Returns:
        A JSON string: {decisions, open_questions, total, scope, state}. Each
        decision carries the same Letter-mode shape as surface_decisions (slug,
        axiom, rejected_paths, scope) plus its computed `state`, and — when a later
        decision modifies it — `superseded_by`/`amended_by`/`narrowed_by`/
        `corrected_by` modifier slugs. UNBOUNDED.
    """
    store, _embed, _vec = get_workspace_components()

    nodes = store.get_decisions(scope=scope, state=state)
    modifiers = store.get_modifiers_map([n["id"] for n in nodes])
    decisions = []
    for n in nodes:
        d = {
            "slug": n["slug"],
            "axiom": n["core_axiom"],
            "scope": n["scope"],
            "state": n["computed_state"],
        }
        if not brief:
            d["rejected_paths"] = n["rejected_paths"]
        d.update(modifiers.get(n["id"], {}))
        decisions.append(d)

    open_questions = []
    try:
        for oq in store.get_open_questions(scope=scope):
            if oq["state"] == "parked":
                open_questions.append(_oq_payload(oq))
    except Exception:
        pass

    return json.dumps({
        "decisions": decisions,
        "open_questions": open_questions,
        "total": len(decisions),
        "scope": scope,
        "state": state,
    }, indent=2)


@mcp.tool()
def query_decisions(query: str, depth: str = "letter", brief: bool = False) -> str:
    """Look up a SPECIFIC decision by slug or claim — the targeted lookup.

    Use this when you know roughly what you're after (a slug you're carrying, or a
    pointed claim). For the broad "is there precedent near this?" scan before
    deciding, use surface_decisions; for the EXHAUSTIVE set in a scope, list_decisions.
    If query matches a unique slug exactly, returns that one decision (full); otherwise
    a ranked semantic search for the claim.

    Args:
        query: Unique decision slug identifier OR a semantic claim search query.
        depth: The retrieval depth (e.g. 'letter', 'trace', 'vibe'). v0.1 enforces Letter mode.
        brief: If True, omit `rejected_paths` from ranked semantic matches (axiom-only).
            An exact-slug hit is always returned in full (you asked for that one).

    Returns:
        A JSON string containing the ranked results in Letter-mode payload shape.
        A decision a later one has moved on from also carries `superseded_by`/
        `amended_by`/`narrowed_by`/`corrected_by` (the modifying slugs) — an exact-slug
        hit on an amended decision still reads `state: "active"`, so chase these before
        trusting its axiom's mechanism.
    """
    if depth != "letter":
        return json.dumps({"error": f"Depth mode '{depth}' is not yet implemented in v0.1 (Letter-only retrieval)."})

    store, embed_provider, vector_store = get_workspace_components()
    
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
            return json.dumps(output, indent=2)
    except Exception:
        # Not a slug collision or lookup failed; proceed to semantic claim lookup
        pass

    # 2. Perform ranked semantic claim search
    if embed_provider and vector_store:
        try:
            q_vector = embed_provider.get_embedding(query, is_query=True)
            matches = vector_store.query(q_vector, limit=5)
            
            output_list = []
            for m in matches:
                slug = m["slug"]
                node = store.get_node_by_slug(slug)
                if not node:
                    continue
                    
                node_state = store.get_node_state(node["id"])
                if node_state not in ("active", "drifted"):
                    continue

                match = {
                    "slug": node["slug"],
                    "axiom": node["core_axiom"],
                    "scope": node["scope"],
                    "state": node_state,
                    "score": m["score"],
                    "depth_mode": "letter"
                }
                if not brief:
                    match["rejected_paths"] = node["rejected_paths"]
                match.update(store.get_modifiers(node["id"]))
                output_list.append(match)
                
            return json.dumps({"query": query, "depth_mode": "letter", "matches": output_list}, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Semantic claim query failed: {str(e)}"})

    return json.dumps({"error": f"Could not resolve slug or run semantic query for '{query}'"})


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
    matching relation arg below (each is validated to point at a real decision).
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
        derives_from: Exact slug of a decision this one is derived from.
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
        and not a write: your decision is ≥0.85 similar to existing `neighbors` you did
        not reference. Inspect them — if this amends/supersedes/contradicts/cites one,
        re-record with that relation arg pointing at the neighbour's slug (`possible_tension`
        on a neighbour flags a likely contradiction, not a duplicate); if it is genuinely
        independent, re-record with acknowledge_neighbors=True. Nothing was written, so a
        re-record is the right move here (unlike an "exists" no-op).
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
    return json.dumps(result)
