"""Stateless renderer for Mitos active axioms.

This module implements the Renderer capability (E) and the C3 integration contract:
generating global and per-scope markdown files atomically from primary source data.
"""

import os
import tempfile
from typing import List, Dict, Any, Optional
from mitos.protocols import GraphStoreProtocol

def render_node_markdown(node: Dict[str, Any]) -> str:
    """Renders a single active decision node as markdown."""
    slug = node.get("slug", "")
    axiom = node.get("core_axiom", "")
    scopes = ", ".join(node.get("scope", []))
    mechs = ", ".join(node.get("mechanisms", []))
    rejected = node.get("rejected_paths", "")

    lines = [
        f"## {slug}",
        f"- **Decided:** {axiom}"
    ]
    if scopes:
        lines.append(f"- **Scope:** {scopes}")
    if mechs:
        lines.append(f"- **Mechanisms:** {mechs}")
    if rejected:
        # Format rejected paths nicely (possibly multiline)
        rejected_indented = "\n  ".join(rejected.splitlines())
        lines.append(f"- **Rejected:**\n  {rejected_indented}")
    
    return "\n".join(lines) + "\n"


def atomic_write(filepath: str, content: str) -> None:
    """Writes content to filepath atomically using a tempfile and replace.

    Prevents partial/corrupted files during failure (F4b).
    """
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
        
    # Write to a temporary file in the same directory (to ensure on same filesystem for rename)
    fd, temp_path = tempfile.mkstemp(dir=dirpath or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        # Atomic rename
        os.replace(temp_path, filepath)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise IOError(f"Atomic write failed for {filepath}: {str(e)}")


class MitosRenderer:
    """Renderer creating active-axiom markdown assets for LLM context ingestion."""

    def __init__(self, workspace_dir: str = ".") -> None:
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.mitos_dir = os.path.join(self.workspace_dir, ".mitos")
        self.axioms_dir = os.path.join(self.mitos_dir, "axioms")

    def render_all(self, store: GraphStoreProtocol, scope: Optional[str] = None) -> List[str]:
        """Statelessly regenerates live_axioms.md and per-scope files.

        Args:
            store: The initialized GraphStore database.
            scope: Optional scope filter. If specified, only that scope is rendered.

        Returns:
            A list of paths rendered.
        """
        # Fetch active decisions directly from database
        active_decisions = store.get_active_decisions()
        
        rendered_paths = []

        # 1. Generate global live_axioms.md
        global_filepath = os.path.join(self.workspace_dir, "live_axioms.md")
        
        global_header = (
            "# Live Axioms\n"
            "*Generated automatically by Mitos. Derived statelessly from primary sources (M8).*\n\n"
        )
        
        global_content = global_header
        if active_decisions:
            global_content += "\n".join(render_node_markdown(d) for d in active_decisions)
        else:
            global_content += "*No active decisions committed in this workspace.*\n"
            
        atomic_write(global_filepath, global_content)
        rendered_paths.append(global_filepath)

        if len(global_content) > 50000:
            print(f"[Warning] 'live_axioms.md' exceeds 50,000 characters ({len(global_content)} chars). Large context files can increase LLM costs and latency. Consider dividing decisions into specialized scopes.")

        # 2. Generate per-scope files
        # Group active decisions by scope tag
        scope_groups: Dict[str, List[Dict[str, Any]]] = {}
        for dec in active_decisions:
            scopes = dec.get("scope", [])
            for s in scopes:
                scope_groups.setdefault(s, []).append(dec)

        # Ensure axioms directory exists
        os.makedirs(self.axioms_dir, exist_ok=True)

        # If a specific scope is requested, only write that one
        scopes_to_render = [scope] if scope else scope_groups.keys()

        for s in scopes_to_render:
            if not s:
                continue
            
            scope_filepath = os.path.join(self.axioms_dir, f"{s}.md")
            scope_header = (
                f"# Active Axioms for Scope: {s}\n"
                f"*Generated automatically by Mitos. Derived statelessly from primary sources (M8).*\n\n"
            )
            
            scope_decisions = scope_groups.get(s, [])
            scope_content = scope_header
            if scope_decisions:
                scope_content += "\n".join(render_node_markdown(d) for d in scope_decisions)
            else:
                scope_content += "*No active decisions committed in this scope.*\n"
                
            atomic_write(scope_filepath, scope_content)
            rendered_paths.append(scope_filepath)

            if len(scope_content) > 20000:
                print(f"[Warning] Scope axiom file '{s}.md' exceeds 20,000 characters ({len(scope_content)} chars). Consider dividing into smaller sub-scopes.")

        return rendered_paths
