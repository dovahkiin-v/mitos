"""CLI entry point for Mitos.

This module implements the command-line interface for Mitos, coordinating
initialization, sync, ambient capture, querying, list, render, import, and
MCP serving.
"""

import sys
import os
import argparse
from typing import List, Optional
from google import genai

from mitos.config import MitosConfig
from mitos.errors import MitosError, ParseError, ValidationError
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager, run_ambient_capture
from mitos.renderer import MitosRenderer
from mitos.importer import MitosProseImporter


def load_format_spec() -> str:
    """Loads the canonical format specification from the package's single source of truth."""
    spec_path = os.path.join(os.path.dirname(__file__), "format-spec.md")
    with open(spec_path, "r", encoding="utf-8") as f:
        return f.read()


def cmd_init(config: MitosConfig) -> None:
    """Initializes the Mitos workspace."""
    os.makedirs(config.mitos_dir, exist_ok=True)
    
    # 0. Ensure format-spec.md exists and is the single source of truth
    format_spec_path = os.path.join(config.workspace_dir, "format-spec.md")
    format_spec_content = load_format_spec()
    
    if not os.path.exists(format_spec_path):
        with open(format_spec_path, "w", encoding="utf-8") as f:
            f.write(format_spec_content)

    # Extract sample block from format-spec.md
    import re
    match = re.search(r'## 3\.\s+Sample Entry.*?\n```markdown\n(.*?)\n```', format_spec_content, re.DOTALL | re.IGNORECASE)
    sample_block = match.group(1).strip() if match else ""
    
    # 1. Create config.toml if missing
    config_path = os.path.join(config.mitos_dir, "config.toml")
    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(
                "# Mitos Workspace Configuration\n"
                'rotation_mode = "archive" # "archive" | "mark" | "prune"\n'
                "pending_threshold = 30\n"
                'qdrant_url = "http://localhost:6333"\n'
                'qdrant_collection = "mitos"\n'
            )
                # 2. Always write/overwrite skill.md (by inclusion of format-spec.md)
    skill_path = os.path.join(config.mitos_dir, "skill.md")
    with open(skill_path, "w", encoding="utf-8") as f:
        f.write(
            "# Mitos Architecture Skill\n\n"
            "You are operating in a workspace governed by Mitos, an architectural decision graph.\n"
            "When you make an architectural decision or change a foundational pattern, you MUST record it in `decisions.md`.\n\n"
            "## Canonical Format Specification\n"
            "Your entries MUST adhere EXACTLY to the following markdown format (loaded from format-spec.md):\n\n"
            f"{format_spec_content}\n\n"
            "## MCP Tools\n"
            "You have access to the Mitos MCP server.\n"
            "- Use `record_decision` the moment you commit to a foundational choice (a schema, a library, a pattern, a path you're abandoning) to persist it — along with the alternatives you rejected and why — so future sessions and other agents inherit it instead of relitigating it.\n"
            "- Use `query_decisions` to semantically search the architectural graph if you are unsure about existing precedents.\n"
            "- Use `surface_decisions` to load all active axioms for a given scope.\n"
        )

    # 3. Create decisions.md buffer if missing (utilizing the extracted sample block)
    if not os.path.exists(config.decisions_file):
        with open(config.decisions_file, "w", encoding="utf-8") as f:
            f.write(
                "# Decisions for Mitos\n\n"
                "<!-- This file is managed by mitos. LLM integration: see .mitos/skill.md once V5 ships. -->\n"
                "<!-- DO NOT MODIFY ABOVE THIS LINE -->\n\n"
                "## SAMPLE FORMAT — auto-restored by mitos sync, do not modify or delete\n\n"
                f"{sample_block}\n\n"
                "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->\n"
            )
            
    # Touch database to initialize
    GraphStore(config.db_path)
    print(f"Initialized Mitos workspace at {config.workspace_dir} ✓")


def cmd_sync(config: MitosConfig, auto_accept: bool = False, embed_only: bool = False, verbose: bool = False) -> None:
    """Synchronizes the decisions write buffer with the graph store."""
    manager = MitosSyncManager(config)
    if embed_only:
        manager.drain_pending_embeddings()
    else:
        try:
            manager.perform_sync(auto_accept=auto_accept, verbose=verbose)
        except ParseError as e:
            print(f"Sync Aborted: Parse error in write-buffer.\n{str(e)}", file=sys.stderr)
            sys.exit(1)
        except ValidationError as e:
            print(f"Sync Aborted: Validation error.\n{str(e)}", file=sys.stderr)
            sys.exit(1)


def cmd_capture(config: MitosConfig, text: str) -> None:
    """Captures a raw architectural thought and appends it to decisions.md."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY environment variable is not set. Capture requires it.")
        return
        
    client = genai.Client(api_key=api_key)
    print("Synthesizing canonical decision entry ...")
    
    try:
        entry_text = run_ambient_capture(client, text)
    except Exception as e:
        print(f"Ambient capture failed: {str(e)}")
        return

    # Append below BEGIN ENTRIES line under advisory lock
    manager = MitosSyncManager(config)
    try:
        with manager.lock:
            with open(config.decisions_file, "r", encoding="utf-8") as f:
                content = f.read()

            marker = "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->"
            if marker in content:
                content = content.replace(marker, f"{marker}\n\n{entry_text}\n")
            else:
                content += f"\n\n{entry_text}\n"

            with open(config.decisions_file, "w", encoding="utf-8") as f:
                f.write(content)
        print(f"Appended synthesized decision to decisions.md buffer ✓")
    except Exception as e:
        print(f"Failed to append captured entry: {str(e)}")


def cmd_query(config: MitosConfig, query_text: str, depth: str = "letter") -> None:
    """Queries the vector store semantically for similar decisions."""
    if depth != "letter":
        raise ValueError(f"Depth mode '{depth}' is not yet implemented in v0.1 (Letter-only retrieval).")
    manager = MitosSyncManager(config)
    if not manager.embed_provider or not manager.vector_store:
        print("Semantic query unavailable (Qdrant or Gemini embedding provider down).")
        return


    try:
        q_vector = manager.embed_provider.get_embedding(query_text, is_query=True)
        matches = manager.vector_store.query(q_vector, limit=5)
        
        if not matches:
            print("No matching decisions found.")
            return

        print(f"\nSemantic Query Matches for: '{query_text}'")
        print("-" * 60)
        for idx, m in enumerate(matches, start=1):
            print(f"{idx}. {m['slug']} (Score: {m['score']:.4f})")
            print(f"   Decided: {m['embedding_text']}")
            print(f"   Scope:   {', '.join(m['scope'])}")
            print(f"   State:   {m['state']}")
            print()
    except Exception as e:
        print(f"Query failed: {str(e)}")


def cmd_show(config: MitosConfig, ident: str) -> None:
    """Shows full details of a specific node by ID or slug."""
    store = GraphStore(config.db_path)
    
    # Try resolving as ID first, then as slug
    node = store.get_node(ident)
    if not node:
        node = store.get_node_by_slug(ident)
        
    if not node:
        print(f"Node with ID or Slug '{ident}' not found.")
        return

    # Compute current active/superseded state
    conn = store._get_connection()
    states = store.compute_all_states(conn)
    state = states.get(node["id"], "active")
    conn.close()

    print(f"\n[{node['kind'].upper()}] {node['slug']}")
    print(f"ID:           {node['id']}")
    print(f"State:        {state}")
    if node.get("date"):
        print(f"Date:         {node['date']}")
    if node.get("title"):
        print(f"Title:        {node['title']}")
        
    if node["kind"] == "decision":
        print(f"Decided:      {node['core_axiom']}")
        print(f"Rejected:     {node['rejected_paths']}")
        print(f"Mechanisms:   {', '.join(node['mechanisms'])}")
        print(f"Scope:        {', '.join(node['scope'])}")
        if node.get("invalidates_if"):
            print(f"Invalidates:  {node['invalidates_if']}")
        if node.get("context"):
            print(f"Context:      {node['context']}")
    else:
        print(f"Park Reason:  {node.get('park_reason') or 'None'}")
        print("Questions Raised:")
        for q in node["questions_raised"]:
            print(f"  - {q}")

    if node.get("transcript"):
        print("\n[Transcript]")
        print(node["transcript"])
    print()


def cmd_list(config: MitosConfig, scope: Optional[str] = None, state_filter: Optional[str] = None) -> None:
    """Lists all nodes in the database, with optional filters."""
    store = GraphStore(config.db_path)
    nodes = store.get_all_nodes()
    
    if not nodes:
        print("Graph database is empty. Run 'mitos sync' to ingest entries.")
        return

    filtered_nodes = []
    for n in nodes:
        if scope and scope not in n["scope"]:
            continue
        if state_filter and n["computed_state"] != state_filter:
            continue
        filtered_nodes.append(n)

    if not filtered_nodes:
        print("No nodes match the given filters.")
        return

    print(f"\nNodes List ({len(filtered_nodes)} found):")
    print("-" * 80)
    for n in filtered_nodes:
        scopes = f"[{', '.join(n['scope'])}]" if n["scope"] else ""
        axiom_snip = n.get("core_axiom", "")
        if len(axiom_snip) > 50:
            axiom_snip = axiom_snip[:47] + "..."
        print(f"{n['computed_state']:11} | {n['kind']:13} | {n['slug']:30} {scopes}")
        if axiom_snip:
            print(f"              {axiom_snip}")
    print()


def cmd_open_questions(config: MitosConfig, scope: Optional[str] = None) -> None:
    """Lists all parked open questions."""
    store = GraphStore(config.db_path)
    oqs = store.get_open_questions(scope=scope)
    
    parked = [q for q in oqs if q["computed_state"] == "parked"]
    if not parked:
        print("Zero parked open questions found.")
        return

    print(f"\nParked Open Questions ({len(parked)} found):")
    print("-" * 80)
    for q in parked:
        reason = f"({q['park_reason']})" if q.get("park_reason") else ""
        print(f"Topic: {q['slug']} {reason}")
        for question in q["questions_raised"]:
            print(f"  - {question}")
    print()


def cmd_import(config: MitosConfig, filepath: str, use_llm_extract: bool = False) -> None:
    """Imports legacy prose ADR file."""
    importer = MitosProseImporter(config)
    importer.import_from_file(filepath, use_llm_extract)


def cmd_render(config: MitosConfig, scope: Optional[str] = None, render_format: str = "live-axioms") -> None:
    """Statelessly regenerates live_axioms.md and scope axioms."""
    if render_format != "live-axioms":
        print(f"Warning: format '{render_format}' is not supported in v0.1. Falling back to live-axioms.")
    store = GraphStore(config.db_path)
    renderer = MitosRenderer(config.workspace_dir)
    rendered = renderer.render_all(store, scope)
    print("Render complete. Generated files:")
    for path in rendered:
        print(f"  - {path}")


def cmd_record(
    config: MitosConfig,
    axiom: str,
    rejected: str,
    scope: Optional[List[str]] = None,
    mechanisms: Optional[List[str]] = None,
    context: Optional[str] = None,
    supersedes: Optional[str] = None,
    slug: Optional[str] = None,
) -> None:
    """Records a decision directly to the write-buffer and graph (thin wrapper)."""
    manager = MitosSyncManager(config)
    result = manager.record_decision_entry(
        axiom=axiom,
        rejected_paths=rejected,
        scope=scope or [],
        mechanisms=mechanisms,
        context=context,
        supersedes=supersedes,
        slug=slug,
    )
    if "error" in result:
        print(f"Record failed [{result['code']}]: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"Recorded decision '{result['slug']}' ({result['status']}) ✓")
    print(f"  ID:        {result['id']}")
    print(f"  State:     {result['state']}")
    print(f"  Embedding: {result['embedding']}")


def cmd_serve() -> None:
    """Starts the FastMCP server over standard stdio."""
    # Importing mcp instance from mcp_server inside the function prevents early execution issues
    from mitos.mcp_server import mcp
    print("Starting Mitos MCP Server on stdio ...")
    mcp.run()


def main() -> None:
    """Main CLI execution router."""
    parser = argparse.ArgumentParser(
        description="Mitos: Architectural Decision Substrate for LLM-native workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    subparsers.add_parser("init", help="Initialize Mitos in current workspace.")

    # sync
    sync_p = subparsers.add_parser("sync", help="Sync buffer decisions to graph database.")
    sync_p.add_argument("--yes", action="store_true", help="Auto-accept all parsed changes.")
    sync_p.add_argument("--embed-only", action="store_true", help="Drain the pending embeddings outbox queue only.")
    sync_p.add_argument("--verbose", action="store_true", help="Show verbose cache statistics.")

    # capture
    cap_p = subparsers.add_parser("capture", help="Synthesize and append a decision.")
    cap_p.add_argument("text", help="Raw decision description.")

    # query
    q_p = subparsers.add_parser("query", help="Semantic lookup for precedents.")
    q_p.add_argument("claim", help="Assertion or subsystem query.")
    q_p.add_argument("--depth", default="letter", help="Depth (default: letter).")

    # show
    show_p = subparsers.add_parser("show", help="Display details of a specific node.")
    show_p.add_argument("ident", help="Slug or ID of node.")

    # list
    list_p = subparsers.add_parser("list", help="List all graph nodes.")
    list_p.add_argument("--scope", help="Filter by scope tag.")
    list_p.add_argument("--state", help="Filter by computed state.")

    # open-questions
    oq_p = subparsers.add_parser("open-questions", help="List active open questions.")
    oq_p.add_argument("--scope", help="Filter by scope tag.")

    # import
    imp_p = subparsers.add_parser("import", help="Import legacy prose ADR.")
    imp_p.add_argument("path", help="Path to markdown prose file.")
    imp_p.add_argument("--from", dest="import_from", default="prose", help="Import source kind.")
    imp_p.add_argument("--llm-extract", action="store_true", help="Use Sonnet compression pass.")

    # render
    ren_p = subparsers.add_parser("render", help="Regenerate rendered outputs.")
    ren_p.add_argument("--format", default="live-axioms", help="Target format.")
    ren_p.add_argument("--scope", help="Optional scope filter.")

    # record
    rec_p = subparsers.add_parser("record", help="Record a decision directly to buffer and graph.")
    rec_p.add_argument("axiom", help="The decision as a single clear sentence true going forward.")
    rec_p.add_argument("--rejected", required=True, help="Alternatives considered and rejected, and why (REQUIRED).")
    rec_p.add_argument("--scope", nargs="*", default=[], help="Area tags, e.g. --scope database auth.")
    rec_p.add_argument("--mechanisms", nargs="*", default=None, help="Concrete technologies/entities, e.g. --mechanisms sqlite wal-mode.")
    rec_p.add_argument("--context", default=None, help="Optional background on why this was decided.")
    rec_p.add_argument("--supersedes", default=None, help="Exact slug of a prior decision this one replaces.")
    rec_p.add_argument("--slug", default=None, help="Optional explicit slug; derived from the axiom if omitted.")

    # serve
    subparsers.add_parser("serve", help="Launch Mitos FastMCP server on stdio.")

    args = parser.parse_args()
    config = MitosConfig()

    try:
        if args.command == "init":
            cmd_init(config)
        elif args.command == "sync":
            cmd_sync(config, auto_accept=args.yes, embed_only=args.embed_only, verbose=args.verbose)
        elif args.command == "capture":
            cmd_capture(config, args.text)
        elif args.command == "query":
            cmd_query(config, args.claim, depth=args.depth)
        elif args.command == "show":
            cmd_show(config, args.ident)
        elif args.command == "list":
            cmd_list(config, scope=args.scope, state_filter=args.state)
        elif args.command == "open-questions":
            cmd_open_questions(config, scope=args.scope)
        elif args.command == "import":
            cmd_import(config, args.path, use_llm_extract=args.llm_extract)
        elif args.command == "render":
            cmd_render(config, scope=args.scope, render_format=args.format)
        elif args.command == "record":
            cmd_record(
                config,
                axiom=args.axiom,
                rejected=args.rejected,
                scope=args.scope,
                mechanisms=args.mechanisms,
                context=args.context,
                supersedes=args.supersedes,
                slug=args.slug,
            )
        elif args.command == "serve":
            cmd_serve()
    except MitosError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Fatal Unexpected Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
