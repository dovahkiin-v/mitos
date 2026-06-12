"""MCP Server implementation for Mitos.

This module implements the MCP Server (F) and the C4 integration contract,
exposing surface_decisions and query_decisions tools to LLM clients.
"""

import os
import json
from typing import Optional, List, Dict, Any, Tuple
from mcp.server.fastmcp import FastMCP

from mitos.config import MitosConfig
from mitos.store import GraphStore
from mitos.embeddings import GeminiEmbeddingProvider
from mitos.vector_store import QdrantVectorStore

# Create FastMCP server instance
mcp = FastMCP("Mitos")

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


@mcp.tool()
def surface_decisions(query: str, scope: Optional[str] = None) -> str:
    """Surfaces active decisions relevant to the query, supporting scope filtering.

    This implements the C4 Letter-mode-only retrieval contract. If a scope filter
    is provided, it pre-filters semantic matches and appends open questions in
    that scope.

    Args:
        query: The semantic claim or topic string (e.g. 'cache strategy').
        scope: Optional scope tag filter (e.g. 'auth', 'database').

    Returns:
        A JSON string containing a ranked list of relevant active decisions
        formatted strictly in Letter mode, and any relevant open questions.
    """
    store, embed_provider, vector_store = get_workspace_components()
    
    results: Dict[str, Any] = {
        "active_decisions": [],
        "open_questions": []
    }

    # 1. Semantic search if embeddings and vector store are active
    if embed_provider and vector_store:
        try:
            # Generate query vector
            q_vector = embed_provider.get_embedding(query, is_query=True)
            matches = vector_store.query(q_vector, limit=5, filter_scope=scope)
            
            for m in matches:
                slug = m["slug"]
                node = store.get_node_by_slug(slug)
                if not node:
                    continue
                    
                # Verifycomputed active status in SQLite (M3 computed state is source-of-truth)
                node_state = store.compute_all_states(store._get_connection()).get(node["id"])
                if node_state not in ("active", "drifted"):
                    # Stale vector reference, skip
                    continue

                # Strictly Letter-mode payload per C4 contract
                results["active_decisions"].append({
                    "slug": node["slug"],
                    "axiom": node["core_axiom"],
                    "rejected_paths": node["rejected_paths"],
                    "scope": node["scope"],
                    "score": m["score"]
                })
        except Exception as e:
            # Degrade to exact/scope filtering only
            pass

    # 2. Scope pre-filtering fallback if semantic search is down
    if not results["active_decisions"] and scope:
        try:
            active_decs = store.get_active_decisions(scope=scope)
            for d in active_decs[:5]:
                results["active_decisions"].append({
                    "slug": d["slug"],
                    "axiom": d["core_axiom"],
                    "rejected_paths": d["rejected_paths"],
                    "scope": d["scope"],
                    "score": 1.0
                })
        except Exception:
            pass

    # 3. Append Open Questions if scope matches (C4 resolves clause)
    if scope:
        try:
            oqs = store.get_open_questions(scope=scope)
            for q in oqs:
                if q["computed_state"] == "parked":
                    results["open_questions"].append({
                        "topic": q["slug"],
                        "questions_raised": q["questions_raised"],
                        "park_reason": q.get("park_reason")
                    })
        except Exception:
            pass

    # Recall here is SEMANTIC and capped at the top few matches — great for "is
    # there precedent for this claim?", but it cannot prove you have seen
    # everything. Point the agent at the exhaustive path so a completeness pass
    # doesn't mistake the ranked top-k for the full set (closes the "am I seeing
    # everything?" trust gap).
    if results["active_decisions"]:
        results["note"] = (
            "Ranked top matches only (semantic, capped). For the COMPLETE set of "
            "decisions in a scope — a completeness pass, not just the most relevant "
            "few — call list_decisions(scope=...)."
        )

    return json.dumps(results, indent=2)


@mcp.tool()
def list_decisions(scope: Optional[str] = None, state: str = "active") -> str:
    """Enumerate the COMPLETE set of decisions (optionally scope-filtered) — no ranking, no top-k.

    surface_decisions / query_decisions are SEMANTIC and capped at the top few
    matches: ideal for "is there precedent for this claim?", but they cannot tell
    you whether you have seen *everything*. When you need certainty — a completeness
    pass over a scope, an audit, "show me every settled call in `auth`" — use this.
    It returns every matching decision deterministically, straight from the graph,
    so nothing hides below a relevance cliff. Needs no API key or Qdrant (pure graph
    read), so it works even when semantic recall is degraded.

    Args:
        scope: Optional scope tag filter (e.g. 'auth'). Omit for the whole project.
        state: 'active' (default) returns the live set (active + drifted); 'all'
            returns every decision regardless of state (including superseded); any
            other value is an exact computed-state match (e.g. 'superseded').

    Returns:
        A JSON string: {decisions, open_questions, total, scope, state}. Each
        decision carries the same Letter-mode shape as surface_decisions (slug,
        axiom, rejected_paths, scope) plus its computed `state` — but UNBOUNDED.
    """
    store, _embed, _vec = get_workspace_components()

    decisions = [
        {
            "slug": n["slug"],
            "axiom": n["core_axiom"],
            "rejected_paths": n["rejected_paths"],
            "scope": n["scope"],
            "state": n["computed_state"],
        }
        for n in store.get_decisions(scope=scope, state=state)
    ]

    open_questions = []
    try:
        for oq in store.get_open_questions(scope=scope):
            if oq["computed_state"] == "parked":
                open_questions.append({
                    "topic": oq["slug"],
                    "questions_raised": oq["questions_raised"],
                    "park_reason": oq.get("park_reason"),
                })
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
def query_decisions(query: str, depth: str = "letter") -> str:
    """Performs an on-demand claim or slug lookup with depth control.

    If query matches a unique slug exactly, returns that decision. Otherwise,
    executes a ranked semantic search for matches matching the claim.

    Args:
        query: Unique decision slug identifier OR a semantic claim search query.
        depth: The retrieval depth (e.g. 'letter', 'trace', 'vibe'). v0.1 enforces Letter mode.

    Returns:
        A JSON string containing the ranked results in Letter-mode payload shape.
    """
    if depth != "letter":
        return json.dumps({"error": f"Depth mode '{depth}' is not yet implemented in v0.1 (Letter-only retrieval)."})

    store, embed_provider, vector_store = get_workspace_components()
    
    # 1. Try resolving query as direct slug first
    try:
        node = store.get_node_by_slug(query)
        if node:
            conn = store._get_connection()
            state = store.compute_all_states(conn).get(node["id"], "active")
            conn.close()
            
            output = {
                "slug": node["slug"],
                "axiom": node["core_axiom"],
                "rejected_paths": node["rejected_paths"],
                "scope": node["scope"],
                "state": state,
                "depth_mode": "letter"
            }
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
                    
                node_state = store.compute_all_states(store._get_connection()).get(node["id"])
                if node_state not in ("active", "drifted"):
                    continue

                output_list.append({
                    "slug": node["slug"],
                    "axiom": node["core_axiom"],
                    "rejected_paths": node["rejected_paths"],
                    "scope": node["scope"],
                    "state": node_state,
                    "score": m["score"],
                    "depth_mode": "letter"
                })
                
            return json.dumps({"query": query, "depth_mode": "letter", "matches": output_list}, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Semantic claim query failed: {str(e)}"})

    return json.dumps({"error": f"Could not resolve slug or run semantic query for '{query}'"})


@mcp.tool()
def record_decision(axiom: str, rejected_paths: str, scope: List[str],
                    mechanisms: Optional[List[str]] = None, context: Optional[str] = None,
                    supersedes: Optional[str] = None, amends: Optional[str] = None,
                    narrows: Optional[str] = None, depends_on: Optional[str] = None,
                    resolves: Optional[str] = None, contradicts: Optional[str] = None,
                    derives_from: Optional[str] = None, cites: Optional[str] = None,
                    slug: Optional[str] = None) -> str:
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
        amends: Exact slug of a decision this one amends (modifies without replacing).
        narrows: Exact slug of a decision this one narrows the scope of.
        depends_on: Exact slug of a decision this one depends on.
        resolves: Exact slug of an open question / decision this one resolves.
        contradicts: Exact slug of a decision this one is in tension with.
        derives_from: Exact slug of a decision this one is derived from.
        cites: Exact slug of a decision this one references.
        slug: Optional explicit slug; derived from the axiom if omitted.

    Returns:
        A JSON string: {slug, id, state, embedding, status} or {error, code}.
        status="created" means newly recorded; status="exists" is a SUCCESS — the
        identical decision was already recorded and is now confirmed present, not an
        error and not something to retry. Only a top-level {error, code} is a failure.
        On the "created" path the result MAY include `related`: the nearest existing
        live decisions to the one you just recorded — a write-time adjacency hint, so
        you notice an adjacent or contradictory prior decision. If one is genuinely
        related, record the link (re-record with the matching relation arg, or capture
        a follow-up). NOTE: identity is (slug + axiom + mechanisms). Re-recording an
        existing decision is a no-op — a changed `context`/`rejected_paths`/`scope` or
        relation on a re-record is NOT saved. To record different reasoning or a new
        relationship, make a NEW decision (a distinct axiom), don't resubmit the old one.
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
        amends=amends,
        narrows=narrows,
        depends_on=depends_on,
        resolves=resolves,
        contradicts=contradicts,
        derives_from=derives_from,
        cites=cites,
        slug=slug,
    )
    return json.dumps(result)
