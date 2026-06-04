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

    return json.dumps(results, indent=2)


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
                    supersedes: Optional[str] = None, slug: Optional[str] = None) -> str:
    """Record a decision you just made, with the alternatives you rejected and why,
    so future sessions and other agents inherit it instead of relitigating it.

    Call this the moment you commit to a foundational choice — a schema, a library,
    a pattern, or a path you've decided to abandon. `rejected_paths` is required:
    recording WHY you ruled options out is what stops you (or the next agent) from
    re-proposing them. If this decision replaces an earlier one, look the earlier one
    up first with query_decisions/surface_decisions and pass its exact slug as
    `supersedes`. Returns the decision's slug; look it up afterwards with query_decisions.

    Args:
        axiom: The decision as a single clear sentence true going forward.
        rejected_paths: The alternatives considered and rejected, and why. REQUIRED.
        scope: Area tags, e.g. ["database", "auth"].
        mechanisms: Concrete technologies/entities involved, e.g. ["sqlite", "wal-mode"].
        context: Optional background on why this was decided.
        supersedes: Optional exact slug of a prior decision this one replaces.
        slug: Optional explicit slug; derived from the axiom if omitted.

    Returns:
        A JSON string: {slug, id, state, embedding, status} or {error, code}.
        status="created" means newly recorded; status="exists" is a SUCCESS — the
        identical decision was already recorded and is now confirmed present, not an
        error and not something to retry. Only a top-level {error, code} is a failure.
        NOTE: identity is (slug + axiom + mechanisms). Re-recording a decision that
        already exists is a no-op — a changed `context`, `rejected_paths`, `scope`, or
        `supersedes` on a re-record is NOT saved. To record different reasoning or a new
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
        slug=slug,
    )
    return json.dumps(result)
