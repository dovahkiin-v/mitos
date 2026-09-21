"""The MCP tool-description budget: front-load what the client cut must carry.

Clients truncate long tool descriptions — measured 2026-08-04 (`AX_FEEDBACK.md`):
`record_decision`'s description was cut mid-sentence *inside the relation
catalog*, and `query_decisions`' likewise, so an agent chose relation types from
the visible half. The cut point is client-side and not ours to control; what we
own is the ordering and the total weight. Three rules, pinned here:

1. **Front-load rule** — the content an agent must not lose lands in the head of
   the description: `record_decision`'s full relation vocabulary (edges are the
   expensive thing to get wrong) and `query_decisions`' verb-choice guidance.
2. **Budget rule** — no description regrows past the ceiling. The ceiling is a
   regression guard, not a target: 0.15.1 brought the worst offender 5,937 →
   under it, and the teaching that was cut from the tail is delivered in-band by
   the responses themselves (the pause lists its own recovery, the receipt
   carries `differs`), so tail loss on an aggressive client is tolerable by
   design.
3. **Required-arguments window** — every required argument's Args entry ends
   inside the head a truncating client still shows, so a first call can be
   made from what arrives. "Required" is the schema's `required` list plus
   `project` wherever a tool declares it: `project` is schema-optional on
   purpose, which leaves the description as the only place that says it is
   needed. This window is measured on the raw text, because the raw text is
   what the client cuts; the phrase rows stay on the flat view.
4. **Container rule** — where an argument's container is a live question, the
   first sentence of its doc says which: an array-typed argument says ``list``;
   a relation argument says ``string`` and ``comma-separated`` (it is one
   string of slugs, never a list); a string argument that shares its name with
   an array-typed argument on another tool says ``string``. The population is
   derived from the schemas, so an argument that recreates the list-vs-string
   asymmetry joins it the day it is declared. Single-valued strings with no
   list homonym (``query``, ``slug``, ``project`` …) state nothing: a shape
   there teaches no one and every session pays for it eagerly.

Descriptions are read off ``mcp.list_tools()`` (the wire truth), not out of the
source, mirroring ``test_mcp_selector._tools``.
"""

import asyncio
import re

import pytest

from mitos import identity, mcp_server

# The regression ceiling (chars). Not a promise the client shows this much —
# only that we never regrow toward the 5,937-char shape that buried the catalog.
DESCRIPTION_BUDGET = 4_800

# The head window the front-load rule guards. Chosen below the smallest observed
# client cut (a 2,221-char description was truncated), with margin.
FRONT_WINDOW = 1_500

# The window every required argument's doc must end inside (raw chars). One
# client, measured on the wire 2026-09-20, cuts a tool description at exactly
# 2,048 characters — characters, not bytes. Only the description is cut; the
# input schema travels separately and arrives whole. 1,900 leaves margin below
# that cut without pretending to know any other client's.
REQUIRED_ARGS_WINDOW = 1_900

RELATION_ARGS = (
    "supersedes",
    "corrects",
    "amends",
    "narrows",
    "contradicts",
    "depends_on",
    "cites",
    "resolves",
    "derives_from",
)


def _descriptions():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    return {tool.name: tool.description or "" for tool in tools}


def _flat(text):
    """Whitespace-collapsed view, for content rows that must survive a re-wrap.

    FastMCP passes the docstring through verbatim, indentation and hard line
    breaks included, so a phrase asserted raw is really an assertion about where
    the source happens to wrap — which reformatting silently breaks while the
    teaching is still there.
    """
    return " ".join(text.split())


def _tools():
    return {tool.name: tool for tool in asyncio.run(mcp_server.mcp.list_tools())}


def _required_args(tool):
    """Names a caller must supply: schema-``required``, plus ``project`` where declared.

    ``project`` joins by property, not by a list of tool names, so a tool that
    gains the selector is measured the day it arrives.
    """
    schema = tool.inputSchema
    names = set(schema.get("required", ()))
    if "project" in schema.get("properties", {}):
        names.add("project")
    return names


def _required_pairs():
    return sorted(
        (name, arg)
        for name, tool in _tools().items()
        for arg in _required_args(tool)
    )


def _arg_entry_span(description, arg):
    """Returns the raw ``(start, end)`` of ``arg``'s entry in the ``Args:`` block.

    The entry starts at its name token and ends after the last character of its
    last continuation line (exclusive) — continuation being the following
    non-blank lines indented deeper than the name line, the rule
    ``test_mcp_selector._project_arg_doc`` uses. Only name lines at the block's
    entry indent count, so a continuation line that happens to begin ``arg:``
    is not mistaken for the entry. The first matching entry wins.
    A missing entry fails rather than skips: a required argument whose doc was
    deleted must red, not drop out of the measurement.
    """
    lines = description.splitlines(keepends=True)
    offset = 0
    args_indent = None
    entry_indent = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if args_indent is None:
            if stripped == "Args:":
                args_indent = indent
        elif stripped and indent <= args_indent:
            break  # the Args: block has ended
        elif stripped and entry_indent is None:
            entry_indent = indent  # the first entry sets the name-line indent
        if (
            entry_indent is not None
            and indent == entry_indent
            and stripped.startswith(f"{arg}:")
        ):
            start = offset + indent
            end = offset + len(line.rstrip("\n"))
            follower_offset = offset + len(line)
            for follower in lines[index + 1:]:
                if not follower.strip():
                    break
                if len(follower) - len(follower.lstrip()) <= indent:
                    break
                end = follower_offset + len(follower.rstrip("\n"))
                follower_offset += len(follower)
            return start, end
        offset += len(line)
    raise AssertionError(f"no `{arg}:` entry in the Args: block")


def test_every_tool_description_within_budget():
    over = {
        name: len(desc)
        for name, desc in _descriptions().items()
        if len(desc) > DESCRIPTION_BUDGET
    }
    assert not over, (
        f"tool description(s) over the {DESCRIPTION_BUDGET}-char budget: {over} — "
        "trim or front-load; clients truncate the tail (AX 2026-08-04)."
    )


def test_record_decision_relation_catalog_is_front_loaded():
    head = _descriptions()["record_decision"][:FRONT_WINDOW]
    missing = [rel for rel in RELATION_ARGS if rel not in head]
    assert not missing, (
        f"relation arg(s) {missing} absent from record_decision's first "
        f"{FRONT_WINDOW} chars — the catalog must precede everything a client "
        "cut can remove, or agents choose edges from the visible half."
    )


def test_record_decision_confusable_pairs_carry_contrast_in_the_head():
    """The confusable pairs read differently only through their contrast words.

    `amends` vs `narrows` was reported genuinely ambiguous from one-line
    definitions (AX 2026-08-04); the catalog carries a worked carve-out example
    for `narrows` and the outgrown-vs-wrong contrast for supersedes/corrects.
    """
    head = _descriptions()["record_decision"][:FRONT_WINDOW]
    assert "health endpoint" in head, (
        "narrows' worked carve-out example left the front window"
    )
    assert "outgrown" in head and "WRONG" in head, (
        "the supersedes-vs-corrects contrast left the front window"
    )


def test_query_decisions_verb_choice_guidance_is_front_loaded():
    head = _descriptions()["query_decisions"][:600]
    assert "surface_decisions" in head, (
        "query_decisions' redirect to surface_decisions for the broad precedent "
        "scan must sit in the description head — it is the verb-choice teaching "
        "the 08-04 session lacked."
    )


def test_query_decisions_states_the_band_axis_inside_the_head_window():
    """The verb-choice redirect now carries WHICH QUESTION each band answers.

    A byte-count gate cannot see a trim that silently deletes the clause, so the
    content gets its own row. Asserted inside the 600-char window rather than
    anywhere in the description: the clause extends the redirect in place, and a
    later edit that moved it behind the window would leave the budget row green
    while the teaching left every truncating client.

    The axis is the QUESTION each answer is a verdict on, never what the two verbs
    retrieve — that is measured identical, and "surface returns the active set" is
    measured false.
    """
    head = _flat(_descriptions()["query_decisions"][:600])
    assert "confidence" in head and "whether precedent exists" in head, (
        "query_decisions' description no longer says which question each recall "
        "verb's band is a verdict on — an agent holding five matches at 0.61 is "
        "back to guessing which verb it should have called."
    )


def test_surface_decisions_states_the_band_axis():
    """The sibling half. Its phrases are gated by the ceiling alone: the window
    this tool is under is the required-arguments one, which measures where
    `query` and `project` end, not where this contrast sits.

    The two descriptions must not converge: neither gains a `Returns:`-block gloss
    for the other's fields, and this one keeps the corpus-level reading of the band
    while `query_decisions` keeps the ranking-level one.
    """
    desc = _flat(_descriptions()["surface_decisions"])
    assert "whether precedent exists" in desc and "confidence" in desc
    assert "rates that ranking, not the corpus" in desc, (
        "surface_decisions no longer contrasts its own band with query_decisions' "
        "— the differentiator sentence was trimmed rather than extended"
    )


_REQUIRED_PAIRS = _required_pairs()


def test_required_argument_population_is_derived_from_the_live_table():
    """The population rows below cannot shrink without this one noticing.

    Re-derived here by set containment rather than compared with a hand list,
    so neither a count nor a tool list sits anywhere to decay.
    """
    measured = set(_REQUIRED_PAIRS)
    for name, tool in _tools().items():
        schema = tool.inputSchema
        if "project" in schema.get("properties", {}):
            assert (name, "project") in measured, (
                f"{name} declares `project` but its doc is not measured against "
                "the required-arguments window"
            )
        for arg in schema.get("required", ()):
            assert (name, arg) in measured, (
                f"{name}'s schema-required `{arg}` is not measured against the "
                "required-arguments window"
            )


@pytest.mark.parametrize(
    ("tool", "arg"), _REQUIRED_PAIRS, ids=[f"{t}/{a}" for t, a in _REQUIRED_PAIRS]
)
def test_required_argument_doc_ends_inside_the_window(tool, arg):
    _, end = _arg_entry_span(_descriptions()[tool], arg)
    assert end <= REQUIRED_ARGS_WINDOW, (
        f"{tool}'s `{arg}` doc ends at raw char {end}, past "
        f"REQUIRED_ARGS_WINDOW={REQUIRED_ARGS_WINDOW} — a client that cuts the "
        "description at 2,048 chars hides it. Move the entry up (required "
        "arguments first in Args:) or shorten what precedes it."
    )


def test_a_deeper_line_beginning_arg_colon_is_not_the_entry():
    """``_arg_entry_span`` matches name lines at the entry indent only (1a fresh-eyes).

    No shipped description has a continuation line that begins with an argument
    name, so without this row the depth rule could be undone and nothing would red
    (found by 8a1's re-plant). A decoy one level deeper, ahead of the real entry,
    must not be measured in its place.
    """
    description = (
        "Summary.\n"
        "\n"
        "Args:\n"
        "    first: The first argument.\n"
        "        slug: a continuation line that starts like an entry.\n"
        "    slug: The real slug entry.\n"
        "        Its continuation.\n"
        "\n"
        "Returns:\n"
        "    Something.\n"
    )
    start, end = _arg_entry_span(description, "slug")
    assert description[start:end] == "slug: The real slug entry.\n        Its continuation."


def test_record_decision_slug_doc_tells_the_truth_about_the_handle():
    """B13: a slug is a mutable handle and is not part of a decision's identity.

    `identity.compute_node_id` hashes kind, axiom and mechanism refs; the slug
    can be renamed (`amend_commentary`). The description must not say otherwise,
    and its slug entry carries the length limit as the literal that
    `identity.SLUG_MAX_LEN`'s comment promises.
    """
    desc = _descriptions()["record_decision"]
    assert "permanent" not in desc
    assert "identity = slug" not in desc
    start, end = _arg_entry_span(desc, "slug")
    assert str(identity.SLUG_MAX_LEN) in desc[start:end], (
        f"record_decision's slug entry no longer states the {identity.SLUG_MAX_LEN}-"
        "char limit that identity.SLUG_MAX_LEN enforces"
    )


@pytest.mark.parametrize("tool", ["show_node", "query_decisions"])
def test_by_handle_reads_say_mechanisms_are_folded_and_what_they_never_return(tool):
    """A6: each by-handle read's description says `mechanisms` is the folded
    identity form, and names what no by-handle read returns — transcripts,
    graph-primary provenance and outgoing edges — with where transcripts live.

    Placed after `project:`'s Args entry (inside `Returns:`), so it spends only the
    budget and moves no window.
    """
    desc = _flat(_descriptions()[tool])
    assert "`mechanisms` is the folded identity form" in desc
    assert "no by-handle read returns transcripts" in desc.lower()
    assert "`decisions.md`" in desc
    assert "graph-primary provenance" in desc
    assert "`created_at`" in desc and "`confirmed_by`/`confirmed_at`" in desc
    assert "outgoing edges" in desc


@pytest.mark.parametrize("tool", ["show_node", "query_decisions"])
def test_by_handle_reads_do_not_gloss_the_prose_fields(tool):
    """A6 (D6): the response delivers `context` / `invalidates_if` in-band, so the
    description does not enumerate them (no Returns-block gloss), and
    `show_node` names no shell command."""
    desc = _descriptions()[tool]
    assert "`context`" not in desc
    assert "`invalidates_if`" not in desc
    if tool == "show_node":
        assert "mitos " not in desc


# `amend_commentary` declares these only to refuse them (A7 §4.9), so their
# container is moot and they carry no Args entry of their own. Each must still
# be declared: when a later phase retires one, the stale-exemption row reds and
# the entry is removed knowingly.
CONTAINER_EXEMPT = frozenset(
    {("amend_commentary", "axiom"), ("amend_commentary", "mechanisms")}
    | {("amend_commentary", rel) for rel in RELATION_ARGS}
)

# The relation args carry no Args entries on record_decision; their shared
# container clause is the catalog header, from here to the first definition.
_RELATION_HEADER_START = "Relation args"
_RELATION_HEADER_END = "supersedes:"


def _arg_type(schema):
    """Returns the JSON type of one property, with an optional's null unwrapped."""
    if "anyOf" in schema:
        types = [b.get("type") for b in schema["anyOf"] if b.get("type") != "null"]
        return types[0] if len(types) == 1 else None
    return schema.get("type")


def _container_population():
    """Derives the ``(tool, arg) -> shape words`` table from the live schemas.

    (a) every array-typed argument; (b) every relation argument, on any tool
    declaring it; (c) every string argument that is a homonym of an
    array-typed argument on another tool. Exemptions are applied here, before
    any doc is looked up, since the exempt arguments have no entry to find.
    """
    types = {
        (name, arg): _arg_type(prop)
        for name, tool in _tools().items()
        for arg, prop in tool.inputSchema.get("properties", {}).items()
    }
    array_names = {arg for (_, arg), kind in types.items() if kind == "array"}
    population = {}
    for (name, arg), kind in types.items():
        if kind == "array":
            population[(name, arg)] = ("list",)
        elif arg in RELATION_ARGS:
            population[(name, arg)] = ("string", "comma-separated")
        elif kind == "string" and arg in array_names:
            population[(name, arg)] = ("string",)
    return {
        pair: words for pair, words in population.items()
        if pair not in CONTAINER_EXEMPT
    }


def _first_sentence(text):
    """Returns the flat text up to the first sentence end, ``e.g.``/``i.e.`` kept.

    A sentence ends at a period followed by whitespace or the end of the text;
    the abbreviations end nothing. No period means the whole text is the
    sentence.
    """
    flat = _flat(text)
    for match in re.finditer(r"\.(?=\s|$)", flat):
        if flat[max(0, match.start() - 3):match.end()].lower() in ("e.g.", "i.e."):
            continue
        return flat[:match.end()]
    return flat


def _container_doc(tool, arg):
    """Returns the text the container rule reads for one population member.

    For a record_decision relation argument that is the whole catalog header
    (it ends at a colon, so no first-sentence cut applies); otherwise the first
    sentence of the argument's Args entry, its ``name:`` token dropped.
    """
    desc = _descriptions()[tool]
    if tool == "record_decision" and arg in RELATION_ARGS:
        start = desc.index(_RELATION_HEADER_START)
        return _flat(desc[start:desc.index(_RELATION_HEADER_END, start)])
    start, end = _arg_entry_span(desc, arg)
    return _first_sentence(desc[start:end].split(":", 1)[1])


_CONTAINER_POPULATION = _container_population()
_CONTAINER_PAIRS = sorted(_CONTAINER_POPULATION)


def test_container_population_is_derived_from_the_live_table():
    """The shape rows below cannot shrink without this one noticing."""
    measured = set(_CONTAINER_PAIRS)
    for pair in (
        ("record_decision", "scope"),
        ("record_decision", "supersedes"),
        ("surface_decisions", "scope"),
    ):
        assert pair in measured, (
            f"{pair} fell out of the container population — the derivation "
            "shrank, and that argument's shape is no longer measured"
        )


def test_container_exemptions_are_still_declared():
    tools = _tools()
    stale = sorted(
        (name, arg) for name, arg in CONTAINER_EXEMPT
        if arg not in tools[name].inputSchema.get("properties", {})
    )
    assert not stale, (
        f"exempt argument(s) {stale} are no longer declared — remove them from "
        "CONTAINER_EXEMPT, so the exemption does not outlive what it excused"
    )


def test_first_sentence_keeps_abbreviations():
    assert _first_sentence('Tags, e.g. ["a"]. An empty list is refused.') == (
        'Tags, e.g. ["a"].'
    )
    assert _first_sentence("One tag (i.e. a string)") == "One tag (i.e. a string)"


@pytest.mark.parametrize(
    ("tool", "arg"), _CONTAINER_PAIRS, ids=[f"{t}/{a}" for t, a in _CONTAINER_PAIRS]
)
def test_container_shape_is_stated_in_the_first_sentence(tool, arg):
    words = _CONTAINER_POPULATION[(tool, arg)]
    doc = _container_doc(tool, arg)
    missing = [w for w in words if not re.search(rf"\b{w}\b", doc)]
    assert not missing, (
        f"{tool}'s `{arg}` doc does not say {missing} in its first sentence "
        f"({doc!r}) — an agent guesses the container, and a list sent where one "
        "string is declared (or the reverse) is the confusion the ledger "
        "recorded six times."
    )
    if "string" in words:
        assert not re.search(r"\blist\b", doc), (
            f"{tool}'s `{arg}` takes one string, but its first sentence says "
            f"'list' ({doc!r}) — the word an agent reads as ['a', 'b']."
        )
