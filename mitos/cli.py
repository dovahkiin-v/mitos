"""CLI entry point for Mitos.

This module implements the command-line interface for Mitos, coordinating
initialization, sync, ambient capture, querying, list, render, import, and
MCP serving.
"""

import sys
import os
import argparse
from typing import List, Optional, Dict, Any
from google import genai

from mitos import __version__
from mitos.config import MitosConfig, default_collection_name, global_env_path
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


def _ensure_gitignore_entry(gitignore_path: str, entry: str) -> None:
    """Ensures ``entry`` is present in ``.gitignore``, creating the file if needed.

    Keeps a scaffolded ``.env`` (which will hold real API keys) out of version
    control. Idempotent — a no-op when the entry is already a line in the file.

    Args:
        gitignore_path: Path to the workspace ``.gitignore``.
        entry: The line to ensure is present (e.g. ``".env"``).
    """
    existing = ""
    if os.path.exists(gitignore_path):
        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                existing = f.read()
        except OSError:
            return
        if entry in existing.splitlines():
            return
    sep = "" if (not existing or existing.endswith("\n")) else "\n"
    try:
        with open(gitignore_path, "a", encoding="utf-8") as f:
            f.write(f"{sep}{entry}\n")
    except OSError:
        pass


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
        collection = default_collection_name(config.workspace_dir)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(
                "# Mitos Workspace Configuration\n"
                'rotation_mode = "archive" # "archive" | "mark" | "prune"\n'
                "pending_threshold = 30\n"
                "# Qdrant REST endpoint. Defaults to Mitos's dedicated :7333 (not the\n"
                "# standard :6333) so Mitos never co-locates its collections in another\n"
                "# Qdrant you run. Set QDRANT_URL before `init` or edit this line.\n"
                f'qdrant_url = "{config.qdrant_url}"\n'
                "# Per-project collection: keeps this project's vectors isolated\n"
                "# from other Mitos workspaces sharing the same Qdrant instance.\n"
                f'qdrant_collection = "{collection}"\n'
            )

    # 1b. Scaffold a gitignored .env with the credential slots, so a human or
    #     LLM setting Mitos up knows exactly where to drop keys (empty by default).
    env_path = os.path.join(config.workspace_dir, ".env")
    if not os.path.exists(env_path):
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(
                "# ============================================================\n"
                "# Mitos API keys — fill in the value(s), then run `mitos sync`.\n"
                "# This file is gitignored; never commit real keys.\n"
                "# ============================================================\n\n"
                "# Google Gemini API key — REQUIRED (unless already set globally).\n"
                "# One key covers BOTH embeddings (semantic surface/query) AND\n"
                "# decision synthesis (sync/capture).\n"
                "# Tip: set it ONCE for every project with\n"
                "#   mitos set-key --global <KEY>   (stored in ~/.config/mitos/.env)\n"
                "# or drop a project-specific key on the line below to override it.\n"
                "# Get one: https://aistudio.google.com/app/apikey\n"
                "GEMINI_API_KEY=\n\n"
                "# Anthropic (Claude) API key — OPTIONAL. Only used by\n"
                "# `mitos import --llm-extract` to convert legacy prose ADRs.\n"
                "# Get one: https://console.anthropic.com/settings/keys\n"
                "ANTHROPIC_API_KEY=\n"
            )
    # Never let the .env (real keys) get committed.
    _ensure_gitignore_entry(os.path.join(config.workspace_dir, ".gitignore"), ".env")

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
            "## Setup — API Keys\n"
            "Mitos reads keys from a `.env` file at the workspace root (`mitos init` scaffolds it with empty slots; it is gitignored). Set exactly one required key:\n"
            "- **`GEMINI_API_KEY`** (Google Gemini) — REQUIRED for semantic `surface_decisions`/`query_decisions` and for `mitos sync`. One key covers both embeddings and synthesis.\n"
            "- `ANTHROPIC_API_KEY` — OPTIONAL, only for `mitos import --llm-extract` (legacy prose import).\n"
            "Without `GEMINI_API_KEY`, `record_decision` still works (it commits to the local graph; the embedding is queued and drains on the next `mitos sync` once the key is set), but semantic surface/query are unavailable. If a tool reports a missing key, tell the human to put it in `.env`.\n\n"
            "Mitos uses its own Qdrant on **:7333** (not the standard :6333), started with `docker compose up -d`. If semantic tools report Qdrant unreachable, tell the human to start it; `record_decision` still works meanwhile (embeddings queue and drain once it's up).\n\n"
            "## Recording & recall — MCP tools (preferred) or CLI fallback\n"
            "All decisions you record, surface, and query are scoped to THIS project's decision graph and its own Qdrant collection — you will not see, and cannot contaminate, other projects' decisions.\n"
            "If the Mitos MCP server is wired into your agent, call these tools directly — best experience: structured args, no shell-quoting. If it is NOT wired, each maps to a CLI verb (and the CLI also accepts the long names as aliases, e.g. `mitos record_decision`):\n"
            "- `record_decision`  (CLI: `mitos record`) — the moment you commit to a foundational choice (a schema, a library, a pattern, a path you're abandoning), persist it WITH the alternatives you rejected and why, so future sessions inherit it instead of relitigating. Recording rich prose via the CLI? Use `--rejected-file -` / `--context-file -` to read from stdin and avoid shell-quoting.\n"
            "- `surface_decisions` (CLI: `mitos surface`) — surface active precedents for a claim/scope BEFORE you decide, so you don't relitigate a settled call. This is the recall loop — use it first.\n"
            "- `query_decisions`   (CLI: `mitos query`) — semantic or slug lookup when unsure whether a precedent exists.\n"
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


def _read_text_arg(inline: Optional[str], file_path: Optional[str]) -> Optional[str]:
    """Resolves a text argument from an inline value or a file.

    Lets agents pass multi-sentence prose without fighting shell quoting: a
    ``--*-file`` path (or ``-`` for stdin) sidesteps apostrophe/quote escaping
    that would otherwise force the prose to be degraded to satisfy bash.

    Args:
        inline: The value passed directly on the command line, if any.
        file_path: A file path to read instead, or ``"-"`` for stdin.

    Returns:
        The resolved text, or None if neither source was provided.
    """
    if file_path is not None:
        if file_path == "-":
            return sys.stdin.read()
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return inline


def cmd_surface(config: MitosConfig, query: str, scope: Optional[str] = None,
                as_json: bool = False) -> None:
    """Surfaces active decisions relevant to a query — the CLI twin of the MCP
    ``surface_decisions`` tool (the precedent-recall half of Mitos).

    Mirrors ``mcp_server.surface_decisions`` so a CLI-only agent (or a human) can
    run the recall loop without the MCP wired. Semantic match first, scope
    pre-filter fallback, plus any parked open questions in scope.

    Args:
        config: The active workspace configuration.
        query: The claim or topic to find precedents for.
        scope: Optional scope tag filter.
        as_json: Emit a machine-readable JSON report (for agents) instead of text.
    """
    import json as _json
    manager = MitosSyncManager(config)
    store = manager.store
    results: Dict[str, Any] = {"active_decisions": [], "open_questions": []}

    if manager.embed_provider and manager.vector_store:
        try:
            q_vector = manager.embed_provider.get_embedding(query, is_query=True)
            for m in manager.vector_store.query(q_vector, limit=5, filter_scope=scope):
                node = store.get_node_by_slug(m["slug"])
                if not node:
                    continue
                state = store.compute_all_states(store._get_connection()).get(node["id"])
                if state not in ("active", "drifted"):
                    continue
                results["active_decisions"].append({
                    "slug": node["slug"], "axiom": node["core_axiom"],
                    "rejected_paths": node["rejected_paths"], "scope": node["scope"],
                    "score": m["score"],
                })
        except Exception:
            pass

    if not results["active_decisions"] and scope:
        try:
            for d in store.get_active_decisions(scope=scope)[:5]:
                results["active_decisions"].append({
                    "slug": d["slug"], "axiom": d["core_axiom"],
                    "rejected_paths": d["rejected_paths"], "scope": d["scope"], "score": 1.0,
                })
        except Exception:
            pass

    if scope:
        try:
            for oq in store.get_open_questions(scope=scope):
                if oq["computed_state"] == "parked":
                    results["open_questions"].append({
                        "topic": oq["slug"], "questions_raised": oq["questions_raised"],
                        "park_reason": oq.get("park_reason"),
                    })
        except Exception:
            pass

    if as_json:
        print(_json.dumps(results, indent=2))
        return

    ad, oqs = results["active_decisions"], results["open_questions"]
    if not ad and not oqs:
        scope_note = f" (scope: {scope})" if scope else ""
        print(f"No active precedents found for: '{query}'{scope_note}")
        print("→ Nothing settled here yet — safe to decide, then record it.")
        return
    print(f"\nPrecedents for: '{query}'" + (f"  (scope: {scope})" if scope else ""))
    print("-" * 60)
    for i, d in enumerate(ad, start=1):
        print(f"{i}. {d['slug']}  (score {d['score']:.3f})")
        print(f"   Decided:  {d['axiom']}")
        print(f"   Rejected: {d['rejected_paths']}")
        if d["scope"]:
            print(f"   Scope:    {', '.join(d['scope'])}")
        print()
    for oq in oqs:
        print(f"[open question in scope] {oq['topic']}")


def cmd_serve() -> None:
    """Starts the FastMCP server over standard stdio."""
    # Importing mcp instance from mcp_server inside the function prevents early execution issues
    from mitos.mcp_server import mcp
    print("Starting Mitos MCP Server on stdio ...")
    mcp.run()


def _check_qdrant(qdrant_url: str, collection: str) -> Dict[str, Any]:
    """Probes Qdrant reachability and the project's collection (best-effort).

    Args:
        qdrant_url: The configured Qdrant REST endpoint.
        collection: The project's collection name.

    Returns:
        ``{reachable, collection_exists, points}`` — ``collection_exists`` and
        ``points`` are ``None`` when Qdrant is unreachable.
    """
    import requests
    out: Dict[str, Any] = {"reachable": False, "collection_exists": None, "points": None}
    try:
        r = requests.get(
            f"{qdrant_url.rstrip('/')}/collections/{collection}", timeout=3
        )
        out["reachable"] = True
        if r.status_code == 200:
            out["collection_exists"] = True
            out["points"] = r.json().get("result", {}).get("points_count")
        elif r.status_code == 404:
            out["collection_exists"] = False
    except Exception:
        pass
    return out


def _env_file_has_key(env_path: str, name: str) -> bool:
    """True if ``env_path`` assigns ``name`` a non-empty value on ANY line.

    Skips empty assignments (the scaffolded ``GEMINI_API_KEY=`` slot) and keeps
    scanning, so a key added on a later line is still found — matching
    ``load_dotenv_file``'s "first non-empty value wins" semantics.
    """
    if not os.path.exists(env_path):
        return False
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{name}="):
                    if line.split("=", 1)[1].strip().strip('"').strip("'"):
                        return True
    except OSError:
        pass
    return False


def _gemini_key_source(workspace_dir: str) -> Optional[str]:
    """Reports where GEMINI_API_KEY comes from, in precedence order.

    Files are checked before the live environment so the report attributes the
    key to its durable home (``main()`` also loads both files into the
    environment, which would otherwise mask the distinction).

    Args:
        workspace_dir: The project directory to inspect.

    Returns:
        ``"project .env"``, ``"global .env"``, ``"environment"``, or None.
    """
    if _env_file_has_key(os.path.join(workspace_dir, ".env"), "GEMINI_API_KEY"):
        return "project .env"
    if _env_file_has_key(global_env_path(), "GEMINI_API_KEY"):
        return "global .env"
    if os.environ.get("GEMINI_API_KEY"):
        return "environment"
    return None


def _gemini_key_present(workspace_dir: str) -> bool:
    """True if GEMINI_API_KEY is available — env, project .env, or global .env."""
    return _gemini_key_source(workspace_dir) is not None


def _mcp_wired(workspace_dir: str) -> bool:
    """True if a project-scoped ``.mcp.json`` wires the mitos MCP server.

    This is the Claude Code convention (a ``mitos`` entry under ``mcpServers``).
    It's a *recommendation* signal for agents, never a readiness blocker — other
    harnesses wire the MCP elsewhere, and humans don't need it at all.
    """
    import json as _json
    path = os.path.join(workspace_dir, ".mcp.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        return "mitos" in (data.get("mcpServers") or {})
    except (OSError, ValueError, AttributeError):
        return False


def _upsert_env_var(env_path: str, name: str, value: str) -> None:
    """Inserts or replaces ``name=value`` in a ``.env`` file, preserving the rest.

    Replaces an existing (possibly empty) ``name=`` line in place; otherwise
    appends one. Creates the file (and parent dirs) if absent, and tightens the
    file to ``0600`` since it holds secrets.

    Args:
        env_path: Path to the ``.env`` file to write.
        name: The variable name (e.g. ``GEMINI_API_KEY``).
        value: The value to store.
    """
    os.makedirs(os.path.dirname(os.path.abspath(env_path)), exist_ok=True)
    lines: List[str] = []
    found = False
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith(f"{name}="):
                    lines.append(f"{name}={value}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        if lines and not lines[-1].endswith("\n"):
            lines.append("\n")
        lines.append(f"{name}={value}\n")
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    try:
        os.chmod(env_path, 0o600)
    except OSError:
        pass


def cmd_set_key(value: str, name: str = "GEMINI_API_KEY", is_global: bool = False) -> None:
    """Stores an API key in the global or project ``.env``.

    Args:
        value: The API key value to store.
        name: The env var name (default ``GEMINI_API_KEY``).
        is_global: If True, write the shared ``~/.config/mitos/.env`` (serves
            every project); otherwise write ``./.env`` for the current project.
    """
    if is_global:
        env_path = global_env_path()
    else:
        env_path = os.path.join(os.getcwd(), ".env")
    _upsert_env_var(env_path, name, value)
    scope = "globally (all projects)" if is_global else "for this project"
    print(f"Stored {name} {scope} → {env_path}")
    if not is_global:
        _ensure_gitignore_entry(os.path.join(os.getcwd(), ".gitignore"), ".env")


def cmd_status(workspace_dir: str, as_json: bool = False) -> int:
    """Reports whether Mitos is set up for a project, and what (if anything) is missing.

    Designed to be run by a human OR an LLM in a new project: it answers "is Mitos
    ready here?" with a clear ✓/⚠/✗ report, a one-line verdict, and an exit code
    (0 = fully ready, 1 = needs attention / not set up). When not ready it prints
    concise next steps and points at the full SETUP walkthrough.

    Args:
        workspace_dir: The project directory to inspect.
        as_json: Emit a machine-readable JSON report instead of the text report.

    Returns:
        ``0`` if fully ready, ``1`` otherwise.
    """
    import json as _json
    workspace_dir = os.path.abspath(workspace_dir)
    config = MitosConfig(workspace_dir)

    mitos_dir_ok = os.path.isdir(config.mitos_dir) and os.path.exists(
        os.path.join(config.mitos_dir, "config.toml")
    )
    decisions_ok = os.path.exists(config.decisions_file)
    spec_ok = os.path.exists(os.path.join(workspace_dir, "format-spec.md"))
    key_source = _gemini_key_source(workspace_dir)
    key_ok = key_source is not None
    q = _check_qdrant(config.qdrant_url, config.qdrant_collection)
    mcp_wired = _mcp_wired(workspace_dir)

    graph_nodes = None
    if os.path.exists(config.db_path):
        try:
            graph_nodes = len(GraphStore(config.db_path, read_only=True).get_all_nodes())
        except Exception:
            graph_nodes = None

    initialized = mitos_dir_ok and decisions_ok
    # A fresh, initialized project has NO Qdrant collection yet — it auto-creates
    # on the first `record_decision`. So an absent (or empty) collection is a
    # normal ready state, NOT a blocker: a project with .mitos/, a key, and a
    # reachable Qdrant is ready to record its first decision. Only an unreachable
    # Qdrant degrades semantic surface/query.
    ready = initialized and key_ok and q["reachable"]

    if as_json:
        print(_json.dumps({
            "workspace": workspace_dir,
            "ready": ready,
            "initialized": initialized,
            "qdrant_url": config.qdrant_url,
            "collection": config.qdrant_collection,
            "checks": {
                "mitos_workspace": mitos_dir_ok,
                "decisions_buffer": decisions_ok,
                "format_spec": spec_ok,
                "gemini_api_key": key_ok,
                "qdrant_reachable": q["reachable"],
                "collection_exists": q["collection_exists"],
                "collection_points": q["points"],
                "graph_nodes": graph_nodes,
                "mcp_wired": mcp_wired,
            },
        }, indent=2))
        return 0 if ready else 1

    verdict = "READY ✓" if ready else ("NEEDS ATTENTION ⚠" if initialized else "NOT SET UP ✗")
    mark = lambda ok: "✓" if ok is True else ("✗" if ok is False else "—")
    # An absent collection on a reachable Qdrant is normal (auto-creates on the
    # first record), so show it as a neutral "—" with a note — never a ✗ that
    # would contradict an otherwise-READY verdict.
    if not q["reachable"]:
        coll_mark, coll_hint = None, "needs Qdrant up (see above)"
    elif q["collection_exists"]:
        coll_mark, coll_hint = True, None
    else:
        coll_mark, coll_hint = None, "auto-created on first record — none recorded yet"
    checks = [
        ("workspace (.mitos/ + config.toml)", mitos_dir_ok, "run `mitos init`"),
        ("decisions.md buffer", decisions_ok, "created by `mitos init`"),
        ("format-spec.md", spec_ok, "created by `mitos init`"),
        ("GEMINI_API_KEY" + (f" (from {key_source})" if key_source else ""), key_ok,
         "set it once for all projects: `mitos set-key --global <KEY>`"),
        (f"Qdrant reachable ({config.qdrant_url})", q["reachable"],
         "start it: `docker compose up -d` in the mitos repo"),
        (f"collection '{config.qdrant_collection}'", coll_mark, coll_hint),
        # Recommendation, not a requirement — never a ✗. Agents get the best AX
        # (ambient surface/record, structured args, no shell-quoting) via the MCP.
        ("MCP wired (recommended for agents)", True if mcp_wired else None,
         "agents: wire `mitos serve` — see SETUP.md §3 (CLI works without it)"),
    ]
    print(f"\nMITOS STATUS for {workspace_dir} — {verdict}\n")
    for label, ok, hint in checks:
        line = f"  {mark(ok)} {label}"
        if ok is not True and hint:
            line += f"   → {hint}"
        print(line)
    if q["reachable"] and q["collection_exists"] and q["points"] is not None:
        print(f"      ({q['points']} vector(s) indexed)")
    if graph_nodes is not None:
        print(f"  • graph holds {graph_nodes} node(s)")
    print()
    if not ready:
        print("Next steps:")
        n = 1
        if not initialized:
            print(f"  {n}. `mitos init` here (creates .mitos/, decisions.md, scaffolds .env)"); n += 1
        if not key_ok:
            print(f"  {n}. Set your GEMINI_API_KEY once for all projects: "
                  f"`mitos set-key --global <KEY>` (or per-project: `mitos set-key <KEY>`)"); n += 1
        if not q["reachable"]:
            print(f"  {n}. Start Mitos's Qdrant: `docker compose up -d` from the mitos repo"); n += 1
        print("  Full walkthrough → SETUP.md "
              "(https://github.com/dovahkiin-v/mitos/blob/main/SETUP.md)")
        print()
    return 0 if ready else 1


def load_dotenv_file(path: str = ".env") -> None:
    """Loads ``KEY=value`` pairs from a ``.env`` file into the environment.

    Mitos reads its credentials (``GEMINI_API_KEY``, ``ANTHROPIC_API_KEY``) and
    ``QDRANT_URL`` straight from ``os.environ``. This loads them from a workspace
    ``.env`` so a key dropped in that file takes effect without a manual
    ``export`` — the same manual parse the live test-suite already uses, with no
    new dependency (P19 — Dependency Skepticism). An empty value is skipped, and
    an existing environment value is never overridden (an explicit ``export``
    wins over the file).

    Args:
        path: Path to the ``.env`` file (default: ``.env`` in the cwd, i.e. the
            workspace root where ``mitos`` is invoked).
    """
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val and key not in os.environ:
                    os.environ[key] = val
    except OSError:
        pass


def main() -> None:
    """Main CLI execution router."""
    # Project .env (cwd) wins; the global ~/.config/mitos/.env fills any gaps —
    # load_dotenv_file never overwrites an already-set key, so loading project
    # first then global yields the precedence env > project > global.
    load_dotenv_file()
    load_dotenv_file(global_env_path())
    parser = argparse.ArgumentParser(
        description="Mitos: Architectural Decision Substrate for LLM-native workflows."
    )
    parser.add_argument("--version", action="version", version=f"mitos {__version__}")
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

    # query (alias: query_decisions — MCP tool name)
    q_p = subparsers.add_parser("query", aliases=["query_decisions"], help="Semantic lookup for precedents.")
    q_p.add_argument("claim", help="Assertion or subsystem query.")
    q_p.add_argument("--depth", default="letter", help="Depth (default: letter).")

    # surface (alias: surface_decisions — MCP tool name) — the precedent-recall loop
    surf_p = subparsers.add_parser("surface", aliases=["surface_decisions"],
                                   help="Surface active decisions relevant to a query (precedent check before deciding).")
    surf_p.add_argument("query", help="The claim or topic to find precedents for.")
    surf_p.add_argument("--scope", default=None, help="Optional scope tag filter.")
    surf_p.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON.")

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

    # record (alias: record_decision — the MCP tool name, so an agent's first instinct works)
    rec_p = subparsers.add_parser("record", aliases=["record_decision"], help="Record a decision directly to buffer and graph.")
    rec_p.add_argument("axiom", help="The decision as a single clear sentence true going forward.")
    rec_p.add_argument("--rejected", default=None, help="Alternatives considered and rejected, and why (REQUIRED — or use --rejected-file).")
    rec_p.add_argument("--rejected-file", default=None, dest="rejected_file",
                       help="Read --rejected from a file ('-' = stdin) to avoid shell-quoting long prose.")
    rec_p.add_argument("--scope", nargs="*", default=[], help="Area tags, e.g. --scope database auth.")
    rec_p.add_argument("--mechanisms", nargs="*", default=None, help="Concrete technologies/entities, e.g. --mechanisms sqlite wal-mode.")
    rec_p.add_argument("--context", default=None, help="Optional background on why this was decided.")
    rec_p.add_argument("--context-file", default=None, dest="context_file",
                       help="Read --context from a file ('-' = stdin).")
    rec_p.add_argument("--supersedes", default=None, help="Exact slug of a prior decision this one replaces.")
    rec_p.add_argument("--slug", default=None, help="Optional explicit slug; derived from the axiom if omitted.")

    # serve
    subparsers.add_parser("serve", help="Launch Mitos FastMCP server on stdio.")

    # status — is Mitos set up for this project? (human- and LLM-friendly check)
    status_p = subparsers.add_parser("status", help="Check whether Mitos is set up for a project.")
    status_p.add_argument("path", nargs="?", default=None, help="Project directory to check (default: current directory).")
    status_p.add_argument("--json", action="store_true", dest="as_json", help="Emit a machine-readable JSON report.")

    # set-key — store an API key globally (all projects) or for this project
    sk_p = subparsers.add_parser("set-key", help="Store an API key globally (all projects) or for this project.")
    sk_p.add_argument("value", help="The API key value.")
    sk_p.add_argument("--name", default="GEMINI_API_KEY", help="Env var name to store (default: GEMINI_API_KEY).")
    sk_p.add_argument("--global", action="store_true", dest="is_global",
                      help="Write the global ~/.config/mitos/.env (shared by ALL projects) instead of this project's .env.")

    args = parser.parse_args()
    config = MitosConfig()

    try:
        if args.command == "init":
            cmd_init(config)
        elif args.command == "sync":
            cmd_sync(config, auto_accept=args.yes, embed_only=args.embed_only, verbose=args.verbose)
        elif args.command == "capture":
            cmd_capture(config, args.text)
        elif args.command in ("query", "query_decisions"):
            cmd_query(config, args.claim, depth=args.depth)
        elif args.command in ("surface", "surface_decisions"):
            cmd_surface(config, args.query, scope=args.scope, as_json=args.as_json)
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
        elif args.command in ("record", "record_decision"):
            rejected = _read_text_arg(args.rejected, args.rejected_file)
            if not (rejected and rejected.strip()):
                print("record requires --rejected or --rejected-file "
                      "(the rejected alternatives are mandatory).", file=sys.stderr)
                sys.exit(2)
            context = _read_text_arg(args.context, args.context_file)
            cmd_record(
                config,
                axiom=args.axiom,
                rejected=rejected,
                scope=args.scope,
                mechanisms=args.mechanisms,
                context=context,
                supersedes=args.supersedes,
                slug=args.slug,
            )
        elif args.command == "serve":
            cmd_serve()
        elif args.command == "status":
            sys.exit(cmd_status(args.path or os.getcwd(), as_json=args.as_json))
        elif args.command == "set-key":
            cmd_set_key(args.value, name=args.name, is_global=args.is_global)
    except MitosError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Fatal Unexpected Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Best-effort 'new version available' nudge, AFTER the command's own
        # output (the finally runs even on the sys.exit paths above). Skipped for
        # the long-running MCP server; fully fail-silent so it never disrupts work.
        if args.command != "serve":
            try:
                from mitos._update import update_notice
                _notice = update_notice(__version__)
                if _notice:
                    print(_notice, file=sys.stderr)
            except Exception:
                pass


if __name__ == "__main__":
    main()
