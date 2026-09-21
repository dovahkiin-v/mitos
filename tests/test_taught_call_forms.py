"""Every call form mitos teaches is checked against the arguments the tools declare.

A7 makes an MCP tool's argument names **contract**: FastMCP drops an unknown
argument silently today, and a later phase is planned to refuse it. From then
on, every call form mitos itself hands an agent — the skill text, the pause message, a
degraded index header, a recall pointer, a tool description — becomes a promise
that the named argument exists. This module keeps those promises true, so a
rename reds here instead of turning mitos's own teaching into refusals.

**Population — found by content, never listed by position.** The scanned
surface is every string literal in ``mitos/*.py`` (walked by AST; an f-string's
interpolations collapse to ``{}``; module, class and function docstrings are
developer prose and excluded), the eight tool descriptions as
``mcp.list_tools()`` delivers them, and the shipped markdown — ``SETUP.md``,
``README.md`` and ``mitos/format-spec.md`` (package data, interpolated verbatim
into the skill text), one paragraph per string.

**Two shapes are extracted.** (A) *Call forms* ``<tool>(<args>)`` for a live
tool name: each top-level ``name=value`` piece is a keyword; a piece without
``name=`` (``{}``, ``…``) is a placeholder and carries nothing to check. The
argument list is bracket-matched to its closing paren and split on top-level
commas; a quote opens a string only where a value starts, and ``<…>`` is one
placeholder span, so the apostrophe in ``<… this file's …>`` opens nothing.
(B) *Bare fragments* ``name=`` outside any call form, from two feeds: (i) any
name some tool declares, anywhere in the surface; (ii) any ``[a-z][a-z_]*=``
in a string that also names a tool — the feed that still sees a renamed or
misspelled argument once it has left the declared set. CLI spellings and
attribute chains (``--scope=``, ``x.y=``, ``$v=``) and ``==`` are not fragments.

**Assertions**, each over a collection that must be empty: every call-form
keyword is declared by its tool; a literal value's container matches the
schema type (``[…]`` array, quoted string, ``True``/``False`` boolean, digits
integer; placeholders skipped); every bare fragment is classified in
``CLASSIFIED``, keyed by ``(source, name)``, as attributed to tools (and then
checked as a keyword of each) or as not a call form, with its reason; and no
classification outlives the fragment it classified. The frame is proved
entered: each source class the vision names is found among the scanned
strings, the form count has a floor, and a synthetic bad form fed through the
same checker comes back flagged.

**Not audited, by decision:** argument names mentioned in prose (``new_slug``
in backticks, "pass ``project``") have no call shape to parse, and scanning
every backticked word would need a table entry for each; format-substituted
argument names (``sync``'s ``{relation}='{target}'``) are invisible to both
feeds, and their runtime values are always real relation names; tool names
themselves (FastMCP answers an unknown tool already); and the CLI half, which
other rows put through ``cli._build_parser()``.
"""

import ast
import asyncio
import inspect
import re
from functools import lru_cache
from pathlib import Path
from typing import List, NamedTuple, Tuple

from mitos import _agent_block, cli, mcp_server, parser, recall, renderer

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "mitos"
MARKDOWN = (REPO / "SETUP.md", REPO / "README.md", PACKAGE / "format-spec.md")

PLACEHOLDERS = ("{}", "N", "...", "…")


@lru_cache(maxsize=None)
def _tools():
    return {tool.name: tool for tool in asyncio.run(mcp_server.mcp.list_tools())}


def _arg_type(schema):
    """Returns the JSON type of one property, with an optional's null unwrapped."""
    if "anyOf" in schema:
        types = [b.get("type") for b in schema["anyOf"] if b.get("type") != "null"]
        return types[0] if len(types) == 1 else None
    return schema.get("type")


# --- the surface -------------------------------------------------------------


def _docstring_nodes(tree):
    """Returns the ids of every module, class and function docstring node."""
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, owners) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                found.add(id(first.value))
    return found


def _collapse(node):
    """Renders an f-string with each interpolation as ``{}``."""
    parts = []
    for value in node.values:
        if isinstance(value, ast.Constant):
            parts.append(str(value.value))
        else:
            parts.append("{}")
    return "".join(parts)


def _module_strings(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = _docstring_nodes(tree)
    strings = []

    def visit(node):
        if isinstance(node, ast.JoinedStr):
            strings.append(_collapse(node))
            # Its constant parts are already in the collapsed text; only the
            # interpolated expressions can hold strings of their own.
            for value in node.values:
                if isinstance(value, ast.FormattedValue):
                    visit(value.value)
                    if value.format_spec is not None:
                        visit(value.format_spec)
            return
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in skip
        ):
            strings.append(node.value)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return strings


def _surface(markdown=MARKDOWN):
    """Returns ``[(source, text)]`` over the whole scanned surface."""
    surface = []
    for path in sorted(PACKAGE.glob("*.py")):
        surface += [(path.stem, text) for text in _module_strings(path)]
    for name, tool in sorted(_tools().items()):
        surface.append((f"tool:{name}", tool.description or ""))
    for path in markdown:
        for paragraph in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
            surface.append((path.name, paragraph))
    return surface


# --- extraction --------------------------------------------------------------

_CLOSERS = {"(": ")", "[": "]", "{": "}", "<": ">"}


def _value_end(text, start):
    """Returns the end of the value beginning at ``start``: a quoted string, a
    bracketed span (``<…>`` included) or a run of word characters."""
    if start >= len(text):
        return start
    char = text[start]
    if char in "'\"":
        close = text.find(char, start + 1)
        return len(text) if close < 0 else close + 1
    if char in _CLOSERS:
        depth, index = 0, start
        while index < len(text):
            if text[index] == char:
                depth += 1
            elif text[index] == _CLOSERS[char]:
                depth -= 1
                if depth == 0:
                    return index + 1
            index += 1
        return len(text)
    match = re.compile(r"[\w.…]*").match(text, start)
    return match.end()


def _split_args(text, start):
    """Splits a call's argument list at ``start`` into top-level pieces.

    Returns ``(pieces, end)`` with ``end`` just past the closing paren, or
    ``None`` when the list never closes.
    """
    pieces, current, index = [], [], start
    while index < len(text):
        char = text[index]
        if char == ")":
            pieces.append("".join(current).strip())
            return [p for p in pieces if p], index + 1
        if char == ",":
            pieces.append("".join(current).strip())
            current = []
            index += 1
            continue
        piece = "".join(current)
        at_value = not piece.strip() or piece.rstrip().endswith("=")
        if char in "'\"<[{(" and (at_value or char in "[{("):
            end = _value_end(text, index)
            current.append(text[index:end])
            index = end
            continue
        current.append(char)
        index += 1
    return None


class Form(NamedTuple):
    source: str
    tool: str
    keywords: List[Tuple[str, str]]
    text: str


class Fragment(NamedTuple):
    source: str
    name: str
    value: str
    text: str


_KEYWORD = re.compile(r"^([A-Za-z_]\w*)\s*=(?!=)\s*(.*)$", re.S)
_FRAGMENT = re.compile(r"(?<![-\w.$])([a-z][a-z_]*)=(?!=)")


def _extract(source, text, tool_names, declared):
    """Returns ``(forms, fragments)`` found in one scanned string."""
    forms, spans = [], []
    call = re.compile(r"\b(" + "|".join(sorted(tool_names)) + r")\(")
    for match in call.finditer(text):
        split = _split_args(text, match.end())
        if split is None:
            continue
        pieces, end = split
        keywords = []
        for piece in pieces:
            keyword = _KEYWORD.match(piece)
            if keyword:
                keywords.append((keyword.group(1), keyword.group(2).strip()))
        forms.append(Form(source, match.group(1), keywords, text[match.start():end]))
        spans.append((match.start(), end))

    names_a_tool = any(re.search(rf"\b{name}\b", text) for name in tool_names)
    fragments = []
    for match in _FRAGMENT.finditer(text):
        if any(start <= match.start() < end for start, end in spans):
            continue
        name = match.group(1)
        if name in declared or names_a_tool:
            value = text[match.end():_value_end(text, match.end())]
            fragments.append(Fragment(source, name, value, text))
    return forms, fragments


def _scan(surface):
    tools = _tools()
    declared = {
        arg for tool in tools.values()
        for arg in tool.inputSchema.get("properties", {})
    }
    forms, fragments = [], []
    for source, text in surface:
        found_forms, found_fragments = _extract(source, text, tools, declared)
        forms += found_forms
        fragments += found_fragments
    return forms, fragments



# --- the classification table ------------------------------------------------

# Every tool that takes the `project` selector.
_SELECTORED = (
    "amend_commentary", "list_decisions", "list_scopes", "query_decisions",
    "record_decision", "show_node", "surface_decisions",
)
_RANKED = ("surface_decisions", "query_decisions")


def _to(*tools):
    return ("attributed", tools)


def _not_a_call_form(reason):
    return ("not a call form", reason)


# Each bare `name=` the scan finds, keyed by (source, name): the tools whose
# argument it teaches, or why it teaches none. A new fragment reds until
# someone decides which it is; an entry whose fragment is gone reds as stale.
CLASSIFIED = {
    ("cli", "brief"): _to(*_RANKED),  # skill text: `brief=True`
    ("cli", "full_top"): _to(*_RANKED),  # skill text: `full_top=N`
    ("cli", "clear"): _to("amend_commentary"),  # skill text: `clear=[…]`
    ("cli", "state"): _not_a_call_form("the `mitos list` text header"),
    ("conflict", "slug"): _not_a_call_form("an internal ValueError message"),
    ("mcp_server", "clear"): _to("amend_commentary"),  # the removal hints
    ("mcp_server", "full_top"): _to(*_RANKED),  # "brief is full_top=0"
    ("mcp_server", "oneline"): _to("list_decisions"),  # amend's not-found recovery
    # _example_call's `{}(project='{}', …)` and the registry escape hatch.
    ("mcp_server", "project"): _to(*_SELECTORED),
    ("recall", "full_top"): _to(*_RANKED),  # _SURFACE_POINTERS["mcp"]
    ("recall", "limit"): _to(*_RANKED),
    # _MCP_PROJECT_SLOT, the selector piece of both degraded-header forms.
    ("renderer", "project"): _to("list_decisions", "show_node"),
    ("sync", "acknowledge_neighbors"): _to("record_decision"),  # the pause
    ("sync", "supersedes"): _to("record_decision"),  # the record refusals
    ("sync", "corrects"): _to("record_decision"),
    ("SETUP.md", "acknowledge_neighbors"): _to("record_decision"),
    ("tool:list_decisions", "brief"): _to("list_decisions"),
    ("tool:query_decisions", "brief"): _to("query_decisions"),
    ("tool:query_decisions", "full_top"): _to("query_decisions"),
    ("tool:record_decision", "acknowledge_neighbors"): _to("record_decision"),
    ("tool:record_decision", "supersedes"): _to("record_decision"),
    ("tool:record_decision", "status"): _not_a_call_form(
        "the Returns block's response value, not an argument"
    ),
    ("tool:surface_decisions", "brief"): _to("surface_decisions"),
    ("tool:surface_decisions", "full_top"): _to("surface_decisions"),
}


# --- the checker -------------------------------------------------------------


def _literal_type(value):
    """Returns the JSON type a literal value spells, or None for a placeholder.

    A sentence's closing period is not part of the value (``True.``).
    """
    value = value.strip()
    if value not in PLACEHOLDERS:
        value = value.rstrip(".")
    if (
        not value
        or value in PLACEHOLDERS
        or re.fullmatch(r"\{\w*\}", value)
        or (value.startswith("<") and value.endswith(">"))
    ):
        return None
    if value.startswith("["):
        return "array"
    if value[0] in "'\"":
        return "string"
    if value in ("True", "False"):
        return "boolean"
    if value.isdigit():
        return "integer"
    return None


def _check(forms, fragments, classified=CLASSIFIED):
    """Returns ``{assertion: [offence, …]}``; every list must be empty."""
    tools = _tools()
    offences = {"keyword": [], "container": [], "unclassified": []}

    def check_keyword(where, tool, name, value):
        properties = tools[tool].inputSchema.get("properties", {})
        if name not in properties:
            offences["keyword"].append(
                f"{where}: `{tool}` declares no `{name}` (declares {sorted(properties)})"
            )
            return
        spelled, declared = _literal_type(value), _arg_type(properties[name])
        if spelled is not None and spelled != declared:
            offences["container"].append(
                f"{where}: `{tool}`'s `{name}` is {declared}, taught as {spelled} "
                f"({name}={value})"
            )

    for form in forms:
        for name, value in form.keywords:
            check_keyword(f"{form.source} {form.text!r}", form.tool, name, value)
    for fragment in fragments:
        key = (fragment.source, fragment.name)
        if key not in classified:
            offences["unclassified"].append(
                f"{fragment.source}: `{fragment.name}={fragment.value}` — add "
                f"{key} to CLASSIFIED as the tools it teaches, or as not a call form"
            )
            continue
        kind, owners = classified[key]
        if kind == "attributed":
            for tool in owners:
                check_keyword(
                    f"{fragment.source} `{fragment.name}={fragment.value}`",
                    tool, fragment.name, fragment.value,
                )
    return offences


@lru_cache(maxsize=None)
def _scanned():
    surface = _surface()
    return surface, _scan(surface)


# --- the rows ----------------------------------------------------------------


def test_every_taught_keyword_is_declared_by_its_tool():
    _, (forms, fragments) = _scanned()
    offences = _check(forms, fragments)["keyword"]
    assert not offences, (
        "mitos teaches argument names its tools do not declare — each is "
        "dropped today and refused once argument names are contract; fix the "
        "teaching at its source:\n" + "\n".join(offences)
    )


def test_every_taught_literal_matches_the_declared_container():
    _, (forms, fragments) = _scanned()
    offences = _check(forms, fragments)["container"]
    assert not offences, (
        "mitos teaches a value in the wrong container — an agent copying it "
        "sends a string where a list is declared, or the reverse:\n"
        + "\n".join(offences)
    )


def test_every_bare_fragment_is_classified():
    _, (forms, fragments) = _scanned()
    offences = sorted(set(_check(forms, fragments)["unclassified"]))
    assert not offences, (
        "a taught `name=` fragment nobody has classified:\n" + "\n".join(offences)
    )


def test_no_classification_outlives_its_fragment():
    _, (_, fragments) = _scanned()
    found = {(f.source, f.name) for f in fragments}
    stale = sorted(set(CLASSIFIED) - found)
    assert not stale, (
        f"classified fragment(s) {stale} are no longer found by the scan. If an "
        "argument was renamed, the taught form at the source still spells the "
        "old name and has gone dark to the declared-name feed — rename it there. "
        "Delete an entry only once its form is really gone."
    )


def _entered(surface, source, rendered, marker):
    """True when a string scanned from ``source`` is the one ``rendered`` came
    from: it carries ``marker``, and every literal chunk between its ``{}``
    interpolations appears in the rendered text."""
    for where, text in surface:
        chunks = [chunk for chunk in text.split("{}") if chunk]
        if (
            where == source
            and any(marker in chunk for chunk in chunks)
            and all(chunk in rendered for chunk in chunks)
        ):
            return True
    return False


def test_the_scan_entered_every_named_source():
    """The vision's seven source classes are the frame check, not the population.

    The agent block and ``README.md`` teach tool names only and hold no call
    form today, so nothing here counts forms from them; they are still
    scanned, and a form added to either later is checked like any other.
    """
    surface, (forms, _) = _scanned()
    sources = {where for where, _ in surface}
    list_form, _ = renderer._list_forms("t", '"t"')
    show_form, _ = renderer._show_forms("t", '"t"')
    missing = [
        label for label, entered in (
            ("the skill text", _entered(
                surface, "cli", cli._skill_md_text(parser.load_format_spec()),
                "surface_decisions",
            )),
            ("the agent block", _entered(
                surface, "_agent_block", _agent_block.agent_block(),
                "mitos-agent-guide",
            )),
            ("the degraded list header", _entered(
                surface, "renderer", list_form, "list_decisions(",
            )),
            ("the degraded show header", _entered(
                surface, "renderer", show_form, "show_node(",
            )),
            ("the header selector slot",
             ("renderer", renderer._MCP_PROJECT_SLOT) in surface),
            ("the recall pointers", all(
                ("recall", pointer) in surface
                for pointer in recall._SURFACE_POINTERS["mcp"].values()
            )),
            ("SETUP.md", "SETUP.md" in sources),
            ("README.md", "README.md" in sources),
            ("format-spec.md", "format-spec.md" in sources),
        ) if not entered
    ]
    missing += [
        f"the {name} description" for name, tool in _tools().items()
        if (f"tool:{name}", tool.description) not in surface
    ]
    assert not missing, f"the scan never entered: {missing}"
    assert len(forms) >= 15, (
        f"only {len(forms)} call forms found — the frame has stopped reaching "
        "the surface that teaches them"
    )


def test_the_forms_known_at_planning_are_found():
    _, (forms, fragments) = _scanned()
    found = {(f.tool, name) for f in forms for name, _ in f.keywords}
    found |= {(f.tool, None) for f in forms}
    for pair in (
        ("list_decisions", "scope"),
        ("list_decisions", "oneline"),
        ("list_decisions", "state"),
        ("show_node", "ident"),
        ("surface_decisions", None),
        ("list_projects", None),
    ):
        assert pair in found, f"{pair} is no longer found among the call forms"
    names = {f.name for f in fragments}
    for name in ("brief", "full_top", "limit", "clear", "acknowledge_neighbors",
                 "supersedes", "corrects", "oneline", "project"):
        assert name in names, f"bare `{name}=` is no longer found by the scan"


def test_the_checker_flags_a_synthetic_bad_form():
    """In-row positive control: the same scan and checker, fed a bad form of
    each kind, must come back with each flagged."""
    synthetic = [
        ("synthetic", "look it up with show_node(id='<slug>')"),
        ("synthetic", 'amend_commentary(slug="s", clear="scope")'),
        ("synthetic", "call surface_decisions with fulltop=3"),
    ]
    offences = _check(*_scan(synthetic))
    assert any("`show_node` declares no `id`" in o for o in offences["keyword"])
    assert any("`clear` is array, taught as string" in o for o in offences["container"])
    assert any("`fulltop=3`" in o for o in offences["unclassified"])


def _signature_mismatches(tools, resolve):
    """Returns the tools whose schema properties differ from their signature."""
    return sorted(
        name for name, tool in tools.items()
        if set(tool.inputSchema.get("properties", {}))
        != set(inspect.signature(resolve(name)).parameters)
    )


def test_the_schema_is_the_signature_so_no_alias_layer_hides_a_rename():
    """The join key is the argument name: a taught keyword must equal a schema
    property, and the schema must equal the tool function's parameters."""
    tools = _tools()
    assert len(tools) == len({t.name for t in tools.values()}) and tools
    assert not _signature_mismatches(tools, lambda name: getattr(mcp_server, name))

    def widened(slug, project=None, unused=None):
        return slug

    one = {"show_node": tools["show_node"]}
    assert _signature_mismatches(one, lambda name: widened) == ["show_node"]
