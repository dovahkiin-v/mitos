"""Stateless renderer for Mitos active axioms.

This module implements the Renderer capability (E) and the C3 integration contract:
generating global and per-scope markdown files atomically from primary source data.
"""

import json
import os
import shlex
import sys
from typing import AbstractSet, List, Dict, Any, Optional, Tuple
from mitos import atomic_file
from mitos.display import oneline_axiom, truncate_words
from mitos.protocols import GraphStoreProtocol
from mitos.scope_tags import normalize_scope_tags
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

# The write-path overflow nudge's cadence: ``summarize_overflows`` says "once a day",
# and the write path passes this window to ``config.hint_due`` — one source, so the
# sentence and the debounce cannot drift apart silently (a row pins the pair).
OVERFLOW_HINT_WINDOW_SECONDS = 24 * 60 * 60
# How many over-ceiling files the nudge names (largest first); the rest are counted.
# The full list is ``mitos status``'s; a once-a-day line must not become the wall of
# per-file lines the debounce retired.
OVERFLOW_NUDGE_FILE_CAP = 5

# Width of the truncated axiom in a secondary-scope pointer line (chars).
POINTER_AXIOM_CHARS = 70

# The addressing skeletons every generated tool pointer is composed from, concrete
# (a real tag) or slot form (`<scope>` / `<slug>`), so the two cannot drift. A
# generated file states the addressing form, never the machine's answer to it (ADR
# generated-file-states-the-addressing-form-never-the-machines-answer).
_MCP_PROJECT_SLOT = ("project=<absolute path of the workspace directory this file's "
                     ".mitos/ sits in>")
_SCOPE_SLOT = "<scope>"
_SLUG_SLOT = "<slug>"


def _list_forms(cli_scope: str, mcp_scope: str) -> Tuple[str, str]:
    """Returns ``(mcp, cli)`` for the bounded oneline scope tier, from rendered scope tokens."""
    return (f"list_decisions(scope={mcp_scope}, oneline=True, {_MCP_PROJECT_SLOT})",
            f"mitos list --scope={cli_scope} --oneline -p .")


def _show_forms(cli_slug: str, mcp_slug: str) -> Tuple[str, str]:
    """Returns ``(mcp, cli)`` for the exact decision lookup, from rendered slug tokens.

    The CLI form puts ``--`` before the slug because a hand-authored slug may start
    with ``-``, and ``show`` takes it as a positional with no ``ident=`` spelling.
    ``-p .`` stays before the ``--``, where it is still read as the selector.
    """
    return (f"show_node(ident={mcp_slug}, {_MCP_PROJECT_SLOT})",
            f"mitos show -p . -- {cli_slug}")


def _addressing_clause(forms: Tuple[str, str]) -> str:
    """Renders ``(mcp, cli)`` as the two-form clause every generated pointer uses.

    The MCP form comes first because it names the absolute path the CLI's "from
    anywhere" clause refers back to (the skill.md Addressing order).
    """
    mcp, cli = forms
    return (f"over MCP, `{mcp}`; on the CLI, `{cli}` from the workspace root, or "
            "`-p <that absolute path>` from anywhere")


# The constant a secondary row ends in when its primary scope's file is an index:
# the body is in no file of the tree, so the row names none. It must never be longer
# than the shortest file pointer (`` → full entry: x.md``), so rewriting a row in an
# undegraded file never grows it and the one-pass degrade set stays exact (ADR
# degrade-set-is-decided-in-one-pass-before-any-row-is-written, as amended in 2d).
POINTER_INDEX_TARGET_MARKER = " → read via show"

# The block heading the secondary-scope pointer lines below a scope file's full
# entries (see render_pointer_line): the heading, then one line saying where each
# row's full entry is, carrying the decision-tier recipe once for every row. It is
# constant and pass one's maximum form emits it too, so it adds the same bytes to
# the degrade measure and to the output. The code after it supplies the newline.
POINTER_SECTION_HEADING = (
    "## Also scoped here\n"
    "Each row's full entry is elsewhere: in the scope file the row names, or, for a row "
    f"ending `{POINTER_INDEX_TARGET_MARKER.strip()}` because the decision's primary "
    "scope file is an index, through the exact lookup: "
    f"{_addressing_clause(_show_forms(_SLUG_SLOT, json.dumps(_SLUG_SLOT)))}. Put the "
    "slug in place of `<slug>`, shell-quoted on the CLI if it holds a space or a quote. "
    "Without a tool, the authored text of every decision is in this workspace's "
    "decisions file and its archives."
)


def render_pointer_line(node: Dict[str, Any], primary_scope: str,
                        primary_is_index: bool = False) -> str:
    """Renders the one-line secondary-scope pointer for a multi-tag decision.

    Per the render-dedupe ADR, a decision's full Letter-complete body renders only
    under its PRIMARY tag (the first tag in its scope list as hydrated); every
    secondary tag's file carries this pointer instead — slug and word-boundary-
    truncated axiom — so scope-file weight stops converging toward tags× corpus
    while the decision stays discoverable from every scope it touches. While the
    primary's file is full the row names it; once that file is an index the body is
    in no file of the tree, so the row ends in ``POINTER_INDEX_TARGET_MARKER`` and the
    section heading block carries the lookup.

    Args:
        node: The decision node dict.
        primary_scope: The decision's primary scope tag (its first, author order).
        primary_is_index: Whether the primary scope's file degraded to an index.

    Returns:
        The pointer line, newline-terminated.
    """
    slug = node.get("slug", "")
    axiom = truncate_words(node.get("core_axiom", ""), POINTER_AXIOM_CHARS)
    if primary_is_index:
        return f"- **{slug}** — {axiom}{POINTER_INDEX_TARGET_MARKER}\n"
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


# The clause a global-index group heading carries in place of a file when that
# scope's own file is an index: no file to name, and the route is in the banner.
INDEX_GROUP_CLAUSE = "scope file is an index (routes above)"


def _assemble_global_index(
    active_decisions: List[Dict[str, Any]],
    modifiers: Dict[str, Dict[str, List[str]]],
    full_chars: int,
    degraded: AbstractSet[str],
    ceiling: int,
) -> Tuple[str, List[Tuple[str, int]]]:
    """Builds the over-ceiling global file: a oneline index grouped by primary scope.

    Per the global-render-degrades ADR: once the full global render would exceed the
    global ceiling, live_axioms.md becomes an index — one line per decision, grouped
    under its PRIMARY scope tag. A group heading names that scope's file only while
    the file is full; a scope whose file degraded gets a heading that names no file,
    and untagged decisions gather in a final ``## (unscoped)`` group that names none
    either. The banner carries the routes for those: the scope tier and the decision
    tier in slot form, in both addressing forms, and the corpus.

    The banner picks its register the way a degraded scope header does: compose the
    file short, and if that whole file is over ``ceiling``, recompose it with one
    added sentence stating the rows' size (never the file's own length).

    Args:
        active_decisions: The active decision nodes, hydrated.
        modifiers: Reverse-relation modifiers keyed by node id.
        full_chars: The char size the full render would have been (for the banner).
        degraded: The scope tags whose per-scope file is an index.
        ceiling: The global ceiling the degrade predicate applied.

    Returns:
        ``(content, decisions)`` where ``decisions`` is the ``(slug, char_count)``
        accounting list at index-row weight.
    """
    groups: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for dec in active_decisions:
        primary = (dec.get("scope") or [None])[0]
        groups.setdefault(primary, []).append(dec)

    sections: List[str] = []
    accounting: List[Tuple[str, int]] = []
    ordered = sorted((s for s in groups if s is not None)) + ([None] if None in groups else [])
    for s in ordered:
        if s is None:
            heading = "## (unscoped)"
        elif s in degraded:
            heading = f"## {s} — {INDEX_GROUP_CLAUSE}"
        else:
            heading = f"## {s} — full entries: .mitos/axioms/{s}.md"
        rows = [(d.get("slug", ""), render_index_row(d, modifiers.get(d["id"])))
                for d in groups[s]]
        accounting.extend((slug, len(r)) for slug, r in rows)
        sections.append(heading + "\n" + "".join(r for _, r in rows))
    body = "\n".join(sections)

    has_file = any(s is not None and s not in degraded for s in groups)
    has_index = any(s is not None and s in degraded for s in groups)
    has_unscoped = None in groups
    meaning: List[str] = []
    if has_file:
        meaning.append("A heading that names a file points at that scope's full entries.")
    if has_index:
        meaning.append(f"A heading marked `{INDEX_GROUP_CLAUSE}` names no file, because "
                       "that scope's own file is an index.")
    if has_unscoped:
        meaning.append("`(unscoped)` gathers the decisions with no scope tag, which no "
                       "scope file holds.")
    if has_index or has_unscoped:
        meaning.append("Reach the entries under a heading that names no file through the "
                       "routes below.")

    lead = (
        "# Live Axioms — Index\n"
        "*Generated automatically by Mitos. Derived statelessly from primary sources (M8).*\n\n"
        f"The full render of this corpus ({_size_clause(full_chars)}) exceeds the global "
        f"size ceiling ({ceiling:,} chars), so this file is a one-line index of every "
        "active decision, with modifier stamps.\n"
    )
    scope_slot = _list_forms(_SCOPE_SLOT, json.dumps(_SCOPE_SLOT))
    slug_slot = _show_forms(_SLUG_SLOT, json.dumps(_SLUG_SLOT))
    guide = (
        " ".join(meaning) + "\n"
        f"- One scope, one line per decision: {_addressing_clause(scope_slot)}.\n"
        f"- One decision's full entry: {_addressing_clause(slug_slot)}.\n"
        "- Put a scope tag or slug in place of its slot, shell-quoted on the CLI if it "
        "holds a space or a quote.\n"
        + _corpus_pointer()
    )
    content = lead + guide + "\n" + body
    if len(content) > ceiling:
        rows_chars = sum(size for _, size in accounting)
        over = (f"This index is itself over that ceiling; its rows alone come to "
                f"{_size_clause(rows_chars)}.\n")
        content = lead + over + guide + "\n" + body
    return content, accounting


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
    every scope file in its maximum form (full bodies plus the pointer section, every
    row naming its primary's file) and marks each one over ``SCOPE_OVERFLOW_WARN_CHARS``;
    pass two emits a marked scope as its index, and an unmarked one with each row into
    a marked primary rewritten to the constant marker (the pass-one file itself when no
    row changes). Both rewrites only shrink a file, so nothing under the ceiling in pass
    one crosses it in pass two. Nothing written in pass two feeds back into the set, so
    it depends neither on the order scopes are visited nor on the order the store
    returns decisions.

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
    global_ceiling = GLOBAL_OVERFLOW_WARN_CHARS

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
    if len(global_content) > global_ceiling:
        global_content, global_decisions = _assemble_global_index(
            active_decisions, modifiers, len(global_content), degraded, global_ceiling)
        global_mode = "index"

    # Pass two: rows written against the fixed set.
    scopes: Dict[str, Dict[str, Any]] = {}
    for s, decs in scope_groups.items():
        if s in degraded:
            scopes[s] = _index_scope_file(s, decs, modifiers, scope_ceiling)
        elif any(d["scope"][0] in degraded for d in _split_by_primacy(s, decs)[1]):
            scopes[s] = _full_scope_file(s, decs, modifiers, degraded)
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
                     modifiers: Dict[str, Dict[str, List[str]]],
                     degraded: AbstractSet[str] = frozenset()) -> Dict[str, Any]:
    """Builds a scope file in its full form.

    Dedupe by primary tag (the render-dedupe ADR): the full Letter-complete body
    renders only under a decision's primary scope tag; under every secondary tag a
    one-line pointer names the primary file, or ends in the constant marker when that
    file is in ``degraded``. Single-tag decisions therefore render exactly as before.
    Called with the empty set this is the degrade predicate's maximum-form measure;
    called with the fixed set it is the emitted file of a scope that does not degrade.

    Args:
        s: The scope tag.
        decs: The active decisions tagged ``s``, in store order.
        modifiers: Reverse-relation modifiers keyed by node id.
        degraded: The scope tags whose file is an index (empty in pass one).

    Returns:
        The file record, ``mode == "full"``.
    """
    header = (
        f"{_scope_title(s)}\n"
        f"*Generated automatically by Mitos. Derived statelessly from primary sources (M8).*\n\n"
    )
    primaries, secondaries = _split_by_primacy(s, decs)
    s_blocks = [(d.get("slug", ""), render_node_markdown(d, modifiers.get(d["id"])))
                for d in primaries]
    pointers = [(d.get("slug", ""),
                 render_pointer_line(d, d["scope"][0], d["scope"][0] in degraded))
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
    forms = _list_forms(shlex.quote(s), json.dumps(s, ensure_ascii=False))
    return f"- The same list through the bounded tool tier: {_addressing_clause(forms)}.\n"


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
        f"{_scope_title(s)}{_INDEX_TITLE_SUFFIX}\n"
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
            f"{_scope_title(s)}\n"
            f"*Generated automatically by Mitos. Derived statelessly from primary sources (M8).*\n\n"
            f"*No active decisions committed in this scope.*\n"
        ),
        "decisions": [],
        "mode": "full",
    }


_INDEX_TITLE_SUFFIX = " — Index"


def _scope_title(s: str) -> str:
    """Returns the first line of every scope file the renderer writes for ``s``, unterminated.

    The full and empty-state forms use it as is; the index form appends
    ``_INDEX_TITLE_SUFFIX``. The sweep recognises its own files by this same string,
    so the code that writes a scope file and the code that recognises one share it.
    """
    return f"# Active Axioms for Scope: {s}"


def _is_vacated_scope_render(entry: "os.DirEntry[str]", claimed: AbstractSet[str]) -> bool:
    """Decides whether a directory entry is this renderer's file for an unclaimed scope.

    Candidates are taken positively (ADR render-sweep-takes-its-candidates-positively-
    by-filename-shape), cheapest test first, and a file is a candidate only when its
    name AND its first line are both the renderer's own:

    1. A regular file, never a symlink: the renderer writes through a link but never
       creates one, so a link in the tree is a person's construct.
    2. ``<stem>.md`` where the stem is already a normalized scope tag. That excludes
       ``README.md``, mixed-case names and ``atomic_file``'s temp files, which end in
       ``_TEMP_SUFFIX`` (``.tmp``) and never in ``.md``.
    3. The stem is not in ``claimed``.
    4. The first line, read as bounded bytes and decoded as UTF-8 with one trailing
       ``\\r`` tolerated (a ``core.autocrlf`` checkout), equals ``_scope_title(stem)``
       or its index form. A first line that is not valid UTF-8 is not the renderer's.
       Only the first line decides: bytes after the first newline are never decoded.

    Reading the title decides ownership only; no byte of it reaches any render.

    Args:
        entry: An entry from a flat ``os.scandir`` of the axioms directory.
        claimed: The scope tags the unfiltered render just wrote.

    Returns:
        True when the file may be removed.

    Raises:
        OSError: The title could not be read; the caller keeps the file.
    """
    if not entry.is_file(follow_symlinks=False) or not entry.name.endswith(".md"):
        return False
    stem = entry.name[:-len(".md")]
    if not stem or normalize_scope_tags([stem]) != [stem] or stem in claimed:
        return False
    try:
        stem.encode("utf-8")
    except UnicodeEncodeError:
        # A name the filesystem could not decode (surrogate-escaped bytes): scope
        # tags are text, so the renderer never wrote it.
        return False
    title = _scope_title(stem)
    index_title = title + _INDEX_TITLE_SUFFIX
    # The longest title the stem can have, plus room for "\r\n".
    limit = len(index_title.encode("utf-8")) + 2
    with open(entry.path, "rb") as f:
        head = f.read(limit)
    line = head.split(b"\n", 1)[0]
    if line.endswith(b"\r"):
        line = line[:-1]
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return text in (title, index_title)


def _sweep_vacated_scope_files(axioms_dir: str, claimed: AbstractSet[str],
                               swept: List[str], failures: List[Dict[str, str]]) -> None:
    """Removes every file in a flat listing of ``axioms_dir`` that renders an unclaimed scope.

    Never raises. A listing failure ends the sweep as one failure named for the
    directory; a removal or title-read failure keeps that file and moves on; a file
    already gone (a concurrent render removed it) is a silent success. Subdirectories
    are never entered.

    Args:
        axioms_dir: The render tree's scope directory.
        claimed: The scope tags the unfiltered render just wrote.
        swept: Appended with each basename removed, in name order.
        failures: Appended with ``{"name", "error"}`` per failure.
    """
    try:
        with os.scandir(axioms_dir) as listing:
            entries = sorted(listing, key=lambda e: e.name)
    except OSError as exc:
        failures.append({"name": axioms_dir, "error": str(exc)})
        return
    for entry in entries:
        try:
            if not _is_vacated_scope_render(entry, claimed):
                continue
            os.remove(entry.path)
        except FileNotFoundError:
            continue
        except Exception as exc:
            failures.append({"name": entry.name, "error": str(exc)})
            continue
        swept.append(entry.name)


def _inside_directory(path: str, directory: str) -> bool:
    """Checks lexically that ``path`` names something strictly below ``directory``.

    Lexical on purpose: resolving links would refuse the write-through to a user's
    symlinked render file that ``atomic_file`` performs deliberately. Every escape a
    scope tag can spell (``..`` segments, an absolute tag, mixed forms) is lexical.

    Args:
        path: The destination to check.
        directory: An absolute, normalized directory.

    Returns:
        True when ``path`` is inside ``directory``.
    """
    dest = os.path.normpath(path)
    try:
        return os.path.commonpath([dest, directory]) == directory and dest != directory
    except ValueError:
        return False


def _ceiling_for(file_info: Dict[str, Any]) -> int:
    """Returns the char ceiling for an assembled file (the looser global one vs per-scope)."""
    return GLOBAL_OVERFLOW_WARN_CHARS if file_info["scope"] is None else SCOPE_OVERFLOW_WARN_CHARS


def _overflow_entry(file_info: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
    """Builds the overflow record for one over-ceiling file.

    Args:
        file_info: An assembled file record (from ``assemble_render``).
        top_n: How many of the file's largest accounted entries to list.

    Returns:
        A JSON-serializable record with the file's char/estimated-token size, the
        ceiling it breached, and its ``top_decisions`` (largest first). Every file over
        its ceiling is an index, so those are its longest rows — a size fact about
        what makes the index long, not a list of decisions to move.
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
    and its top-N longest rows — so a health surface (``mitos status``) can say which
    index files are still over and what makes them long. Every such file is an index
    with nowhere further to degrade. Returns an empty list when nothing is over.

    Args:
        store: The initialized GraphStore to read from.
        top_n: How many of the longest rows to list per over-ceiling file.

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
    success receipt and only append a warning when there is genuinely one to show.
    Names each over-ceiling file with its size and its own ceiling (largest first, up
    to ``OVERFLOW_NUDGE_FILE_CAP``, the rest counted) and states the debounce, so an
    absent nudge on a later receipt is not read as the files having shrunk. It names
    no command: the string reaches MCP verbatim, and the CLI text receipt adds its own
    selectored recipe line.

    Args:
        overflows: The overflow records (e.g. from ``MitosRenderer.overflows``, which
            is in write order — a sorted copy is used, the list is not reordered).

    Returns:
        A one-line summary, or None.
    """
    if not overflows:
        return None
    n = len(overflows)
    noun, their = ("file", "its") if n == 1 else ("files", "their")
    ordered = sorted(overflows, key=lambda e: e["chars"], reverse=True)
    named = [f"{e['name']} {e['chars']:,} chars (ceiling {e['threshold_chars']:,})"
             for e in ordered[:OVERFLOW_NUDGE_FILE_CAP]]
    rest = n - len(named)
    if rest:
        named.append(f"and {rest} more")
    return (f"⚠ {n} rendered axiom {noun} over {their} size ceiling: {', '.join(named)}. "
            f"This notice is debounced to once a day per workspace, so a later receipt "
            f"without it does not mean the files shrank.")


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
        # Also populated by render_all and reset on every call: the basenames an
        # unfiltered render removed, the sweep's failures ({"name", "error"}), and the
        # scope files not written because their tag escapes the tree ({"scope", "path"}).
        self.swept: List[str] = []
        self.sweep_failures: List[Dict[str, str]] = []
        self.write_refusals: List[Dict[str, str]] = []

    def render_all(self, store: GraphStoreProtocol, scope: Optional[str] = None) -> List[str]:
        """Statelessly regenerates live_axioms.md and per-scope files.

        An unfiltered call also removes each file in ``.mitos/axioms/`` that renders a
        scope no longer in the active set, so the directory holds only active scopes'
        renders. It removes only a regular file whose name and first line are both this
        renderer's own for that scope, and only after every write has landed. A removal
        that fails is recorded on ``self.sweep_failures``, reported on stderr, and never
        raised; the next unfiltered render retries it.

        A scope whose tag would put its file outside ``.mitos/axioms/`` is not written, on
        any call. It is recorded on ``self.write_refusals`` and reported on stderr, and
        every other file is still written.

        Size-ceiling overflows are recorded on ``self.overflows`` (not printed), so the
        write path can present a single debounced summary AFTER its success receipt,
        while the full breakdown stays on ``mitos status`` — see ``summarize_overflows``
        and ``overflow_report``.

        Args:
            store: The initialized GraphStore database.
            scope: Optional scope filter. If specified, only that scope is rendered, and
                nothing is removed. An empty string renders every scope, like ``None``.

        Returns:
            A list of paths written. A removed file is never in it.
        """
        # Reset before anything can raise, so a failed call never shows the last call's.
        self.swept, self.sweep_failures, self.write_refusals = [], [], []
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
        # One predicate decides both what the loop writes and whether the sweep runs,
        # so the sweep runs exactly when this call wrote every active scope.
        unfiltered = not scope
        scopes_to_render = list(assembled["scopes"].keys()) if unfiltered else [scope]
        axioms_root = os.path.normpath(self.axioms_dir)
        try:
            for s in scopes_to_render:
                if not s:
                    continue
                # An explicitly-requested scope with no active decisions still gets an
                # empty-state file (preserves the pre-refactor `render --scope` behaviour).
                info = assembled["scopes"].get(s) or _empty_scope_file(s)
                scope_filepath = os.path.join(self.axioms_dir, f"{s}.md")
                if not _inside_directory(scope_filepath, axioms_root):
                    # Skipped, not raised: a raise would stall every render in the
                    # workspace on one tag; and not sanitized, because pointers name
                    # the raw tag.
                    self.write_refusals.append(
                        {"scope": s, "path": os.path.normpath(scope_filepath)})
                    continue
                atomic_write(scope_filepath, info["content"])
                rendered_paths.append(scope_filepath)
                written_files.append(info)

            # Only after every write landed: a render that raised above never sweeps.
            if unfiltered:
                try:
                    _sweep_vacated_scope_files(self.axioms_dir, frozenset(assembled["scopes"]),
                                               self.swept, self.sweep_failures)
                except Exception as exc:
                    self.sweep_failures.append({"name": self.axioms_dir, "error": str(exc)})
        finally:
            # Refusals recorded before a later write raised are still reported.
            self._report_render_problems()

        # Record (don't print) which written files breached their size ceiling.
        self.overflows = [
            _overflow_entry(f) for f in written_files if len(f["content"]) > _ceiling_for(f)
        ]

        return rendered_paths

    def _report_render_problems(self) -> None:
        """Prints one stderr line per write refusal and sweep failure; stdout is never touched.

        The renderer reports these itself because one of its callers swallows every
        render exception and another must keep stdout clean. Nothing prints on success.
        """
        if not (self.write_refusals or self.sweep_failures):
            return
        sys.stdout.flush()
        for refusal in self.write_refusals:
            print(f"[Warning] Scope tag {refusal['scope']!r} would put its render file "
                  f"outside .mitos/axioms/, so that file was not written. Every other "
                  f"render file was written.", file=sys.stderr)
        for failure in self.sweep_failures:
            if failure["name"] == self.axioms_dir:
                # The listing itself, or the sweep as a whole, failed: no file is named.
                print(f"[Warning] Could not check {failure['name']!r} for stale scope "
                      f"render files: {failure['error']}. The render and the write before "
                      f"it are unaffected; the next full render retries.", file=sys.stderr)
                continue
            print(f"[Warning] Could not remove stale scope render {failure['name']!r}: "
                  f"{failure['error']}. The render and the write before it are "
                  f"unaffected; the next full render retries.", file=sys.stderr)
