"""Stateless renderer for Mitos active axioms.

This module implements the Renderer capability (E) and the C3 integration contract:
generating global and per-scope markdown files atomically from primary source data.
"""

import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from mitos.display import truncate_words
from mitos.protocols import GraphStoreProtocol

# Size ceilings for the generated context files, named in CHARACTERS — the unit the
# check actually measures. (A `len(content)` is characters, so the threshold is named
# in characters, not tokens, to keep the name honest about what it guards.) A rough
# chars→tokens estimate is reported alongside so an author sees the LLM-context cost
# the ceiling is really about. live_axioms.md aggregates every active axiom while a
# per-scope file holds a single scope's slice, so the global ceiling is the looser one.
GLOBAL_OVERFLOW_WARN_CHARS = 50_000
SCOPE_OVERFLOW_WARN_CHARS = 20_000
_CHARS_PER_TOKEN = 4

# Width of the truncated axiom in a secondary-scope pointer line (chars).
POINTER_AXIOM_CHARS = 70

# Section heading grouping the secondary-scope pointer lines below a scope
# file's full entries (see render_pointer_line).
POINTER_SECTION_HEADING = "## Also scoped here (full entries elsewhere)"


def render_pointer_line(node: Dict[str, Any], primary_scope: str) -> str:
    """Renders the one-line secondary-scope pointer for a multi-tag decision.

    Per the render-dedupe ADR, a decision's full Letter-complete body renders only
    under its PRIMARY tag (the first tag in its scope list as hydrated); every
    secondary tag's file carries this pointer instead — slug, word-boundary-
    truncated axiom, and where the full body lives — so scope-file weight stops
    converging toward tags× corpus while the decision stays discoverable from
    every scope it touches.

    Args:
        node: The decision node dict.
        primary_scope: The decision's primary scope tag (its first, author order).

    Returns:
        The pointer line, newline-terminated.
    """
    slug = node.get("slug", "")
    axiom = truncate_words(node.get("core_axiom", ""), POINTER_AXIOM_CHARS)
    return f"- **{slug}** — {axiom} → full entry: {primary_scope}.md\n"


def estimate_tokens(char_count: int) -> int:
    """Estimates an LLM token count from a character count.

    Uses the standard ~4-characters-per-token heuristic. Deliberately rough — it
    exists so a size report can show "~13k tokens" next to a raw char count, giving
    an author the context-cost framing the ceiling is really guarding (not an exact
    tokeniser count).

    Args:
        char_count: Number of characters.

    Returns:
        The estimated token count.
    """
    return char_count // _CHARS_PER_TOKEN


def render_node_markdown(node: Dict[str, Any],
                         modifiers: Optional[Dict[str, List[str]]] = None) -> str:
    """Renders a single active decision node as markdown.

    Args:
        node: The decision node dict.
        modifiers: Optional reverse-relation modifiers (from
            ``GraphStore.get_modifiers``). A live-but-amended/narrowed decision is
            rendered with a ``⚠ Amended by`` line so this generated context file
            can't present a moved-on axiom as the final word.

    Returns:
        The node's markdown block.
    """
    slug = node.get("slug", "")
    axiom = node.get("core_axiom", "")
    scopes = ", ".join(node.get("scope", []))
    mechs = ", ".join(node.get("mechanisms", []))
    rejected = node.get("rejected_paths", "")

    lines = [
        f"## {slug}",
        f"- **Decided:** {axiom}"
    ]
    for key, label in (("amended_by", "Amended by"), ("narrowed_by", "Narrowed by"),
                       ("corrected_by", "Corrected by"), ("superseded_by", "Superseded by")):
        targets = (modifiers or {}).get(key)
        if targets:
            lines.append(f"- **⚠ {label}:** {', '.join(targets)} "
                         f"(chase before treating this axiom as current)")
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


def assemble_render(store: GraphStoreProtocol) -> Dict[str, Any]:
    """Builds the global and per-scope axiom markdown in memory, without writing.

    The single source of truth for both ``MitosRenderer.render_all`` (which writes the
    files) and the read-only size report a health surface shows — so the two can never
    drift on what a file's content (and therefore its measured size) is.

    Args:
        store: The initialized GraphStore to read active decisions from.

    Returns:
        A dict ``{"global": <file>, "scopes": {scope: <file>}}`` where each ``<file>``
        is ``{"name", "scope", "content", "decisions"}`` and ``decisions`` is a list of
        ``(slug, char_count)`` per rendered decision block — enough for a caller to find
        the largest contributors to a file's size. Only scopes with at least one active
        decision appear in ``scopes``.
    """
    active_decisions = store.get_active_decisions()
    # Reverse-relation modifiers, so a live-but-amended axiom carries its
    # "chase the later decision" marker instead of reading as the final word.
    modifiers = store.get_modifiers_map([d["id"] for d in active_decisions])

    def blocks_for(decisions: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        return [(d.get("slug", ""), render_node_markdown(d, modifiers.get(d["id"])))
                for d in decisions]

    global_header = (
        "# Live Axioms\n"
        "*Generated automatically by Mitos. Derived statelessly from primary sources (M8).*\n\n"
    )
    global_blocks = blocks_for(active_decisions)
    if global_blocks:
        global_content = global_header + "\n".join(b for _, b in global_blocks)
    else:
        global_content = global_header + "*No active decisions committed in this workspace.*\n"

    scope_groups: Dict[str, List[Dict[str, Any]]] = {}
    for dec in active_decisions:
        for s in dec.get("scope", []):
            scope_groups.setdefault(s, []).append(dec)

    scopes: Dict[str, Dict[str, Any]] = {}
    for s, decs in scope_groups.items():
        header = (
            f"# Active Axioms for Scope: {s}\n"
            f"*Generated automatically by Mitos. Derived statelessly from primary sources (M8).*\n\n"
        )
        # Dedupe by primary tag (the render-dedupe ADR): the full Letter-complete
        # body renders only under a decision's FIRST scope tag; under every
        # secondary tag a one-line pointer names the primary file. Single-tag
        # decisions therefore render exactly as before. Note: the ADR says
        # "author order", but the graph does not persist it — node_scopes is
        # committed as a sorted set and hydrated sorted — so "first tag" here is
        # the first of the node's scope list as every read surface presents it.
        primaries = [d for d in decs if d.get("scope", [None])[0] == s]
        secondaries = [d for d in decs if d.get("scope", [None])[0] != s]
        s_blocks = blocks_for(primaries)
        pointers = [(d.get("slug", ""), render_pointer_line(d, d["scope"][0]))
                    for d in secondaries]
        content = header + "\n".join(b for _, b in s_blocks)
        if pointers:
            # Full blocks end with "\n"; the extra "\n" leaves one blank line
            # before the pointer section (or sits flush under the header's own
            # trailing blank line when the file is pointers-only).
            if s_blocks:
                content += "\n"
            content += POINTER_SECTION_HEADING + "\n" + "".join(p for _, p in pointers)
        scopes[s] = {
            "name": f"{s}.md",
            "scope": s,
            "content": content,
            # Size-contributor accounting reflects the real contents: full blocks
            # at body weight, pointers at their one-line weight.
            "decisions": ([(slug, len(b)) for slug, b in s_blocks]
                          + [(slug, len(p)) for slug, p in pointers]),
        }

    return {
        "global": {
            "name": "live_axioms.md",
            "scope": None,
            "content": global_content,
            "decisions": [(slug, len(b)) for slug, b in global_blocks],
        },
        "scopes": scopes,
    }


def _empty_scope_file(s: str) -> Dict[str, Any]:
    """Builds the empty-state file record for an explicitly-requested scope with no decisions."""
    return {
        "name": f"{s}.md",
        "scope": s,
        "content": (
            f"# Active Axioms for Scope: {s}\n"
            f"*Generated automatically by Mitos. Derived statelessly from primary sources (M8).*\n\n"
            f"*No active decisions committed in this scope.*\n"
        ),
        "decisions": [],
    }


def _ceiling_for(file_info: Dict[str, Any]) -> int:
    """Returns the char ceiling for an assembled file (the looser global one vs per-scope)."""
    return GLOBAL_OVERFLOW_WARN_CHARS if file_info["scope"] is None else SCOPE_OVERFLOW_WARN_CHARS


def _overflow_entry(file_info: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
    """Builds the overflow record for one over-ceiling file.

    Args:
        file_info: An assembled file record (from ``assemble_render``).
        top_n: How many of the largest decisions in the file to list.

    Returns:
        A JSON-serializable record with the file's char/estimated-token size, the
        ceiling it breached, and its ``top_decisions`` (largest first) — so a reader
        knows which decisions to consider re-scoping.
    """
    chars = len(file_info["content"])
    top = sorted(file_info["decisions"], key=lambda t: t[1], reverse=True)[:top_n]
    return {
        "name": file_info["name"],
        "scope": file_info["scope"],
        "chars": chars,
        "est_tokens": estimate_tokens(chars),
        "threshold_chars": _ceiling_for(file_info),
        "top_decisions": [
            {"slug": slug, "chars": c, "est_tokens": estimate_tokens(c)} for slug, c in top
        ],
    }


def overflow_report(store: GraphStoreProtocol, top_n: int = 5) -> List[Dict[str, Any]]:
    """Read-only report of which rendered files exceed their size ceiling.

    Assembles the same content ``render_all`` would write (without writing it) and
    returns one entry per over-ceiling file — each with its char/estimated-token size
    and the top-N largest decisions in it — so a health surface (``mitos status``) can
    tell an author *what* to re-scope. Returns an empty list when nothing is over.

    Args:
        store: The initialized GraphStore to read from.
        top_n: How many of the largest decisions to list per over-ceiling file.

    Returns:
        A list of overflow records (see ``_overflow_entry``), largest file first.
    """
    assembled = assemble_render(store)
    files = [assembled["global"]] + list(assembled["scopes"].values())
    over = [_overflow_entry(f, top_n) for f in files if len(f["content"]) > _ceiling_for(f)]
    over.sort(key=lambda e: e["chars"], reverse=True)
    return over


def summarize_overflows(overflows: List[Dict[str, Any]]) -> Optional[str]:
    """One-line write-path summary of files over their size ceiling, or None.

    Returns ``None`` when nothing is over threshold, so the caller can print a clean
    success receipt and only append a warning when there is genuinely one to show. The
    detail (which files, which decisions) lives on ``mitos status`` — this is the
    debounced nudge that points there, replacing the per-file wall of lines that used
    to print on every write.

    Args:
        overflows: The overflow records (e.g. from ``MitosRenderer.overflows``).

    Returns:
        A one-line summary, or None.
    """
    if not overflows:
        return None
    n = len(overflows)
    noun = "file" if n == 1 else "files"
    return (f"⚠ {n} rendered axiom {noun} over the size ceiling "
            f"— run `mitos status` for the breakdown.")


class MitosRenderer:
    """Renderer creating active-axiom markdown assets for LLM context ingestion."""

    def __init__(self, workspace_dir: str = ".") -> None:
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.mitos_dir = os.path.join(self.workspace_dir, ".mitos")
        self.axioms_dir = os.path.join(self.mitos_dir, "axioms")
        # Populated by render_all: one record per written file that breached its size
        # ceiling. Read (not printed) so the write path can present a single debounced
        # summary AFTER its success receipt instead of a wall of per-file warnings.
        self.overflows: List[Dict[str, Any]] = []

    def render_all(self, store: GraphStoreProtocol, scope: Optional[str] = None) -> List[str]:
        """Statelessly regenerates live_axioms.md and per-scope files.

        Size-ceiling overflows are recorded on ``self.overflows`` (not printed), so the
        write path can present a single debounced summary AFTER its success receipt and
        route the full breakdown to ``mitos status`` — see ``summarize_overflows`` and
        ``overflow_report``.

        Args:
            store: The initialized GraphStore database.
            scope: Optional scope filter. If specified, only that scope is rendered.

        Returns:
            A list of paths rendered.
        """
        assembled = assemble_render(store)
        rendered_paths: List[str] = []
        written_files: List[Dict[str, Any]] = []

        # 1. Global live_axioms.md (always rendered).
        global_info = assembled["global"]
        global_filepath = os.path.join(self.workspace_dir, "live_axioms.md")
        atomic_write(global_filepath, global_info["content"])
        rendered_paths.append(global_filepath)
        written_files.append(global_info)

        # 2. Per-scope files (filtered to one scope when requested).
        os.makedirs(self.axioms_dir, exist_ok=True)
        scopes_to_render = [scope] if scope else list(assembled["scopes"].keys())
        for s in scopes_to_render:
            if not s:
                continue
            # An explicitly-requested scope with no active decisions still gets an
            # empty-state file (preserves the pre-refactor `render --scope` behaviour).
            info = assembled["scopes"].get(s) or _empty_scope_file(s)
            scope_filepath = os.path.join(self.axioms_dir, f"{s}.md")
            atomic_write(scope_filepath, info["content"])
            rendered_paths.append(scope_filepath)
            written_files.append(info)

        # Record (don't print) which written files breached their size ceiling.
        self.overflows = [
            _overflow_entry(f) for f in written_files if len(f["content"]) > _ceiling_for(f)
        ]

        return rendered_paths
