"""Stateless renderer for Mitos active axioms.

This module implements the Renderer capability (E) and the C3 integration contract:
generating global and per-scope markdown files atomically from primary source data.
"""

import json
import os
import shlex
from typing import List, Dict, Any, Optional, Tuple
from mitos import atomic_file
from mitos.display import oneline_axiom, truncate_words
from mitos.protocols import GraphStoreProtocol
from mitos.store import MODIFIER_EDGE_KEYS

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


def _index_marker(modifiers: Optional[Dict[str, List[str]]]) -> str:
    """Builds the compact modifier marker riding an index row (global or per-scope).

    Mirrors the CLI oneline tier's ``⚠ amended by: <slug>`` marker (the
    stamps-survive-every-thinner-tier rule): a still-active decision that a later
    ``amends``/``narrows``/``corrects``/``supersedes`` has moved on from must not
    read as the final word even in a one-line index.

    Args:
        modifiers: Reverse-relation modifiers for the node (from
            ``GraphStore.get_modifiers_map``), or None.

    Returns:
        A ``  ⚠ …`` suffix, or ``""`` when the node is unmodified.
    """
    if not modifiers:
        return ""
    parts = []
    for key in MODIFIER_EDGE_KEYS.values():
        slugs = modifiers.get(key)
        if slugs:
            parts.append(f"{key.replace('_', ' ')}: {', '.join(slugs)}")
    return ("  ⚠ " + "; ".join(parts)) if parts else ""


def render_index_row(node: Dict[str, Any],
                     modifiers: Optional[Dict[str, List[str]]] = None) -> str:
    """Renders one index line for a decision, in either index tier.

    The row is identical in ``live_axioms.md``'s index and in a degraded per-scope
    file: one degrade rule, one row.

    Slug + word-boundary-truncated axiom (via ``oneline_axiom`` — the same
    truncation seam as the CLI/MCP oneline tier, one seam not two) + a compact
    modifier marker when the decision has incoming modifier edges.

    Args:
        node: The decision node dict.
        modifiers: Optional reverse-relation modifiers for the node.

    Returns:
        The index row, newline-terminated.
    """
    slug = node.get("slug", "")
    return f"- **{slug}** — {oneline_axiom(node)}{_index_marker(modifiers)}\n"


def _assemble_global_index(
    active_decisions: List[Dict[str, Any]],
    modifiers: Dict[str, Dict[str, List[str]]],
    full_chars: int,
) -> Tuple[str, List[Tuple[str, int]]]:
    """Builds the over-ceiling global file: a oneline index grouped by primary scope.

    Per the global-render-degrades ADR: once the full global render would exceed
    ``GLOBAL_OVERFLOW_WARN_CHARS``, live_axioms.md becomes an index — one line per
    decision, grouped under its PRIMARY scope tag with a pointer to that scope's
    per-scope file (the canonical full render). Untagged decisions gather in a
    final unscoped group.

    Args:
        active_decisions: The active decision nodes, hydrated.
        modifiers: Reverse-relation modifiers keyed by node id.
        full_chars: The char size the full render would have been (for the banner).

    Returns:
        ``(content, decisions)`` where ``decisions`` is the ``(slug, char_count)``
        accounting list at index-row weight.
    """
    banner = (
        "# Live Axioms — Index\n"
        "*Generated automatically by Mitos. Derived statelessly from primary sources (M8).*\n\n"
        f"The full render of this corpus ({full_chars:,} chars, "
        f"~{estimate_tokens(full_chars):,} tokens) exceeds the global size ceiling "
        f"({GLOBAL_OVERFLOW_WARN_CHARS:,} chars), so this file is a one-line index of "
        "every active decision. The per-scope files named under each heading are the "
        "canonical full renders.\n"
    )
    groups: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for dec in active_decisions:
        primary = (dec.get("scope") or [None])[0]
        groups.setdefault(primary, []).append(dec)

    sections: List[str] = []
    accounting: List[Tuple[str, int]] = []
    ordered = sorted((s for s in groups if s is not None)) + ([None] if None in groups else [])
    for s in ordered:
        if s is None:
            heading = "## (unscoped) — no scope file; full entries live in decisions.md"
        else:
            heading = f"## {s} — full entries: .mitos/axioms/{s}.md"
        rows = [(d.get("slug", ""), render_index_row(d, modifiers.get(d["id"])))
                for d in groups[s]]
        accounting.extend((slug, len(r)) for slug, r in rows)
        sections.append(heading + "\n" + "".join(r for _, r in rows))
    return banner + "\n" + "\n".join(sections), accounting


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
    """Writes a render file atomically using a temp file and replace, without fsync.

    Prevents partial/corrupted files during failure (F4b). Delegates to
    ``atomic_file.write_derived``, so after a power loss the file can be lost rather
    than reverted — acceptable only because a render regenerates. Never use this for a
    file ``mitos rebuild`` replays from; ``atomic_file.write_source`` is that path.
    """
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    try:
        atomic_file.write_derived(filepath, content)
    except Exception as e:
        raise IOError(f"Atomic write failed for {filepath}: {str(e)}") from e


def assemble_render(store: GraphStoreProtocol) -> Dict[str, Any]:
    """Builds the global and per-scope axiom markdown in memory, without writing.

    The single source of truth for both ``MitosRenderer.render_all`` (which writes the
    files) and the read-only size report a health surface shows — so the two can never
    drift on what a file's content (and therefore its measured size) is.

    The per-scope degrade set is decided in one pass before any row is written (ADR
    ``degrade-set-is-decided-in-one-pass-before-any-row-is-written``): pass one builds
    every scope file in its maximum form (full bodies plus the pointer section) and
    marks each one over ``SCOPE_OVERFLOW_WARN_CHARS``; pass two emits a marked scope as
    its index and an unmarked one as exactly the pass-one file. Nothing written in pass
    two feeds back into the set, so it depends neither on the order scopes are visited
    nor on the order the store returns decisions.

    Args:
        store: The initialized GraphStore to read active decisions from.

    Returns:
        A dict ``{"global": <file>, "scopes": {scope: <file>}}`` where each ``<file>``
        is ``{"name", "scope", "content", "decisions", "mode"}`` and ``decisions`` is a
        list of ``(slug, char_count)`` per rendered decision block or row — enough for
        a caller to find the largest contributors to a file's size. Only scopes with at
        least one active decision appear in ``scopes``. ``mode`` is ``"full"`` (the
        Letter-complete file, under its ceiling) or ``"index"`` (the one-line index a
        file over its ceiling degrades to), on the global file and every scope file.
    """
    active_decisions = store.get_active_decisions()
    # Reverse-relation modifiers, so a live-but-amended axiom carries its
    # "chase the later decision" marker instead of reading as the final word.
    modifiers = store.get_modifiers_map([d["id"] for d in active_decisions])
    # Read at call time, once, so the degrade predicate and the number a degraded
    # header states are the same value.
    scope_ceiling = SCOPE_OVERFLOW_WARN_CHARS

    scope_groups: Dict[str, List[Dict[str, Any]]] = {}
    for dec in active_decisions:
        for s in dec.get("scope", []):
            scope_groups.setdefault(s, []).append(dec)

    # Pass one: every scope in its maximum form, and the fixed degrade set.
    full_files = {s: _full_scope_file(s, decs, modifiers) for s, decs in scope_groups.items()}
    degraded = {s for s, f in full_files.items() if len(f["content"]) > scope_ceiling}

    global_header = (
        "# Live Axioms\n"
        "*Generated automatically by Mitos. Derived statelessly from primary sources (M8).*\n\n"
    )
    global_blocks = [(d.get("slug", ""), render_node_markdown(d, modifiers.get(d["id"])))
                     for d in active_decisions]
    if global_blocks:
        global_content = global_header + "\n".join(b for _, b in global_blocks)
    else:
        global_content = global_header + "*No active decisions committed in this workspace.*\n"
    global_decisions = [(slug, len(b)) for slug, b in global_blocks]
    global_mode = "full"
    # The same degrade rule at the global tier (the global-render-degrades ADR): while
    # the full render fits the global ceiling, the file is byte-identical to the
    # pre-ADR output; once it would exceed the ceiling, live_axioms.md becomes a
    # oneline index instead — a pure deterministic function of the would-be rendered
    # size, no config knob. The size-contributor accounting follows the real contents
    # (index-row weight), so the overflow report stays honest in either mode: an
    # index rarely breaches its ceiling, but one that does is still reported.
    if len(global_content) > GLOBAL_OVERFLOW_WARN_CHARS:
        global_content, global_decisions = _assemble_global_index(
            active_decisions, modifiers, len(global_content))
        global_mode = "index"

    # Pass two: rows written against the fixed set.
    scopes: Dict[str, Dict[str, Any]] = {}
    for s, decs in scope_groups.items():
        if s in degraded:
            scopes[s] = _index_scope_file(s, decs, modifiers, scope_ceiling)
        else:
            scopes[s] = full_files[s]

    return {
        "global": {
            "name": "live_axioms.md",
            "scope": None,
            "content": global_content,
            "decisions": global_decisions,
            "mode": global_mode,
        },
        "scopes": scopes,
    }


def _split_by_primacy(
    s: str, decs: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits a scope's decisions into those whose primary tag is ``s`` and the rest.

    The first tag is the author's first tag: node_scopes persists it as the
    ``ordinal`` and hydrates in that order (ADR scope-primacy-is-the-authored-first-
    tag-persisted-as-a-node-scopes-ordinal). A graph that has not re-committed or
    rebuilt since the ordinal migration still reads its alphabetical backfill here.
    Both lists keep the store's order.
    """
    primaries = [d for d in decs if d.get("scope", [None])[0] == s]
    secondaries = [d for d in decs if d.get("scope", [None])[0] != s]
    return primaries, secondaries


def _full_scope_file(s: str, decs: List[Dict[str, Any]],
                     modifiers: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
    """Builds a scope file in its maximum (full) form.

    Dedupe by primary tag (the render-dedupe ADR): the full Letter-complete body
    renders only under a decision's primary scope tag; under every secondary tag a
    one-line pointer names the primary file. Single-tag decisions therefore render
    exactly as before. This one function is both the degrade predicate's measure and
    the emitted file of every scope that does not degrade.

    Args:
        s: The scope tag.
        decs: The active decisions tagged ``s``, in store order.
        modifiers: Reverse-relation modifiers keyed by node id.

    Returns:
        The file record, ``mode == "full"``.
    """
    header = (
        f"# Active Axioms for Scope: {s}\n"
        f"*Generated automatically by Mitos. Derived statelessly from primary sources (M8).*\n\n"
    )
    primaries, secondaries = _split_by_primacy(s, decs)
    s_blocks = [(d.get("slug", ""), render_node_markdown(d, modifiers.get(d["id"])))
                for d in primaries]
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
    return {
        "name": f"{s}.md",
        "scope": s,
        "content": content,
        # Size-contributor accounting reflects the real contents: full blocks
        # at body weight, pointers at their one-line weight.
        "decisions": ([(slug, len(b)) for slug, b in s_blocks]
                      + [(slug, len(p)) for slug, p in pointers]),
        "mode": "full",
    }


def _size_clause(char_count: int) -> str:
    """Renders a size as ``N chars, ~T tokens`` — the shape every index header states."""
    return f"{char_count:,} chars, ~{estimate_tokens(char_count):,} tokens"


def _scope_tool_pointer(s: str) -> str:
    """Renders the bounded oneline scope tier in both addressing forms, as one bullet.

    A generated file states the addressing *form*, never the machine's answer to it
    (ADR generated-file-states-the-addressing-form-never-the-machines-answer): the CLI
    form carries ``-p .`` with its root-relative caveat, the MCP form a described
    ``project`` slot. The tag is quoted for each reader, because a scope tag is
    user-authored: ``shlex.quote`` for the shell, and the ``--scope=`` spelling so a
    tag that starts with ``-`` is not read as an option; a JSON string literal for
    the tool call.

    Args:
        s: The scope tag.

    Returns:
        The bullet, newline-terminated.
    """
    cli = f"mitos list --scope={shlex.quote(s)} --oneline -p ."
    mcp = (f"list_decisions(scope={json.dumps(s, ensure_ascii=False)}, oneline=True, "
           "project=<absolute path of the workspace directory this file's .mitos/ sits in>)")
    # The MCP form comes first because it names the absolute path the CLI's
    # "from anywhere" clause refers back to (the skill.md Addressing order).
    return (f"- The same list through the bounded tool tier: over MCP, `{mcp}`; on the "
            f"CLI, `{cli}` from the workspace root, or `-p <that absolute path>` from "
            "anywhere.\n")


def _corpus_pointer() -> str:
    """Renders the no-tool route to the authored text of every decision, as one bullet."""
    return ("- The authored text of every decision is in `decisions.md` and any archives "
            "under `decisions/archive/`; `grep` reaches it there.\n")


def _index_scope_file(s: str, decs: List[Dict[str, Any]],
                      modifiers: Dict[str, Dict[str, List[str]]],
                      ceiling: int) -> Dict[str, Any]:
    """Builds a degraded scope file: a self-declaring header over one index row per decision.

    Every decision tagged ``s`` gets one ``render_index_row`` line, primaries first and
    then secondaries, each in store order, with its modifier stamps — the stamps are
    what the raw corpus cannot supply to a reader joining the two by hand. There is no
    per-row pointer and no pointer section; the header carries the file's one pointer.

    The header has two registers. The file is composed with the short one first; if
    that whole file is over ``ceiling`` it is recomposed with the long one, which only
    adds a sentence, so the choice is monotonic and needs no loop. The long register
    states the size of the index rows, never the file's own length, which would change
    as its digits did.

    Args:
        s: The scope tag.
        decs: The active decisions tagged ``s``, in store order.
        modifiers: Reverse-relation modifiers keyed by node id.
        ceiling: The per-scope ceiling the degrade predicate applied.

    Returns:
        The file record, ``mode == "index"``, with ``decisions`` at index-row weight.
    """
    primaries, secondaries = _split_by_primacy(s, decs)
    rows = [(d.get("slug", ""), render_index_row(d, modifiers.get(d["id"])))
            for d in primaries + secondaries]
    rows_text = "".join(r for _, r in rows)
    count = len(rows)
    noun = "decision" if count == 1 else "decisions"

    lead = (
        f"# Active Axioms for Scope: {s} — Index\n"
        "*Generated automatically by Mitos. Derived statelessly from primary sources (M8).*\n\n"
        f"The full render of this scope would exceed the per-scope size ceiling, so this "
        f"file is an index of the {count} active {noun} tagged `{s}`: one line each, with "
        "their modifier stamps. Their full bodies are not in this file.\n"
    )
    pointers = _scope_tool_pointer(s) + _corpus_pointer()
    content = lead + pointers + "\n" + rows_text
    if len(content) > ceiling:
        over = (f"This index is itself over the per-scope size ceiling ({ceiling:,} "
                f"chars); its rows alone come to {_size_clause(len(rows_text))}.\n")
        content = lead + over + pointers + "\n" + rows_text
    return {
        "name": f"{s}.md",
        "scope": s,
        "content": content,
        "decisions": [(slug, len(r)) for slug, r in rows],
        "mode": "index",
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
        "mode": "full",
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

    def __init__(self, workspace_dir: str) -> None:
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
