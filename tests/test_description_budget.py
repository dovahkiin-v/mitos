"""The MCP tool-description budget: front-load what the client cut must carry.

Clients truncate long tool descriptions — measured 2026-08-04 (`AX_FEEDBACK.md`):
`record_decision`'s description was cut mid-sentence *inside the relation
catalog*, and `query_decisions`' likewise, so an agent chose relation types from
the visible half. The cut point is client-side and not ours to control; what we
own is the ordering and the total weight. Two rules, pinned here:

1. **Front-load rule** — the content an agent must not lose lands in the head of
   the description: `record_decision`'s full relation vocabulary (edges are the
   expensive thing to get wrong) and `query_decisions`' verb-choice guidance.
2. **Budget rule** — no description regrows past the ceiling. The ceiling is a
   regression guard, not a target: 0.15.1 brought the worst offender 5,937 →
   under it, and the teaching that was cut from the tail is delivered in-band by
   the responses themselves (the pause lists its own recovery, the receipt
   carries `differs`), so tail loss on an aggressive client is tolerable by
   design.

Descriptions are read off ``mcp.list_tools()`` (the wire truth), not out of the
source, mirroring ``test_mcp_selector._tools``.
"""

import asyncio

from mitos import mcp_server

# The regression ceiling (chars). Not a promise the client shows this much —
# only that we never regrow toward the 5,937-char shape that buried the catalog.
DESCRIPTION_BUDGET = 4_800

# The head window the front-load rule guards. Chosen below the smallest observed
# client cut (a 2,221-char description was truncated), with margin.
FRONT_WINDOW = 1_500

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
    """The sibling half, gated by the ceiling alone — there is no front-load row on
    this tool, and a phase told to check a head window here hunts for a gate that
    is not on it.

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
