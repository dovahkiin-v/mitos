"""Rows for `mitos.tool_markup`, the pattern set both write verbs refuse (Phase 7b, B2).

The scanner in isolation, and the reflection row that pins `FIELD_CLOSER_NAMES` against
both tools' live input schemas. The verb rows live beside their verbs:
`test_record_decision.py` (record) and `test_amend.py` / `test_amend_commentary.py` /
the two amend boundary modules (amend).

That the parser still accepts this markup from hand-authored markdown — the sentence
that keeps the refusal a producer-side narrowing (vision §5.4 C5) — is pinned by
`test_modifier_surfacing.py::test_a6_stray_markup_in_context_reads_back_byte_exact`,
and is not repeated here.

The namespaced prefix is built by concatenation: the tooling that writes these rows
parses the literal form.
"""

import asyncio
import json
import subprocess
import sys

import pytest

from mitos import mcp_server
from mitos.tool_markup import (FIELD_CLOSER_NAMES, TOOL_CALL_MARKUP, field_values,
                               find_tool_call_markup)

NS = "ant" + "ml:"

#: One leaked shape per member, keyed by the member's label.
SHAPES = {
    "function_calls": "</function_calls>",
    "invoke": "</invoke>",
    "parameter": '<parameter name="context">',
    "antml": "</" + NS + "parameter>",
    "field_closer": "</rejected_paths>",
}


def test_the_pattern_set_is_exactly_the_five_labelled_members() -> None:
    """D2 — a closed tuple; a sixth member is a decision, not a drift."""
    assert isinstance(TOOL_CALL_MARKUP, tuple)
    assert [label for label, _ in TOOL_CALL_MARKUP] == list(SHAPES)


@pytest.mark.parametrize("label", sorted(SHAPES))
def test_each_member_is_found_verbatim_at_its_character_offset(label) -> None:
    shape = SHAPES[label]
    value = f"Ünïcode lead — {shape} and the rest."
    assert find_tool_call_markup([("context", value)]) == [
        {"field": "context", "span": shape, "offset": value.index(shape)}]


@pytest.mark.parametrize("opener", [
    '<invoke name="record_decision">', "<parameter name=\"axiom\">",
    "<" + NS + 'invoke name="x">', "<function_calls>",
])
def test_the_opening_forms_are_found(opener) -> None:
    assert find_tool_call_markup([("v", f"x {opener} y")])[0]["span"] == opener


@pytest.mark.parametrize("near_miss", [
    "</div>", "<parameter>", "<invoke>", "</Context>", "< /context>", "</ context>",
    NS + "parameter", "a <parameter-list> of generics", "</unknown_field>",
])
def test_near_misses_are_not_markup(near_miss) -> None:
    """R6 — the line D2 draws: literal, case-sensitive, no generic closer."""
    assert find_tool_call_markup([("context", f"About {near_miss} here.")]) == []


def test_a_backticked_tag_is_exempt_and_a_mixed_value_yields_the_unfenced_hit() -> None:
    """R3 / G4 — the match runs on the masked copy, the span comes from the original."""
    assert find_tool_call_markup([("context", "Mention `</context>` as prose.")]) == []
    value = "Quoted `</context>` then leaked </context> here."
    assert find_tool_call_markup([("context", value)]) == [
        {"field": "context", "span": "</context>", "offset": value.rindex("</context>")}]


def test_an_unclosed_backtick_exempts_nothing() -> None:
    """G5 — the shared inline-code regex needs both ends."""
    assert find_tool_call_markup([("context", "`</context> no close")])[0]["offset"] == 1


def test_hits_come_in_field_order_then_offset_with_no_cap() -> None:
    value = "a</parameter> b</context>\n</invoke>"
    hits = find_tool_call_markup([("rejected_paths", value), ("axiom", "x</cites>")])
    assert [(h["field"], h["span"]) for h in hits] == [
        ("rejected_paths", "</parameter>"), ("rejected_paths", "</context>"),
        ("rejected_paths", "</invoke>"), ("axiom", "</cites>")]
    assert [h["offset"] for h in hits[:3]] == sorted(h["offset"] for h in hits[:3])
    many = "</invoke>" * 50
    assert len(find_tool_call_markup([("context", many)])) == 50


def test_non_strings_are_skipped_and_the_scanner_never_raises() -> None:
    """Standing rule 5 — the record verb gains no raise from junk."""
    junk = [("mechanisms[0]", None), ("mechanisms[1]", 3), ("scope", object()),
            ("slug", b"</invoke>"), ("x", ["</invoke>"])]
    assert find_tool_call_markup(junk) == []


def test_list_and_tuple_values_expand_to_indexed_labels() -> None:
    assert field_values("scope", ["a", "b"]) == [("scope[0]", "a"), ("scope[1]", "b")]
    assert field_values("scope", ("a",)) == [("scope[0]", "a")]
    assert field_values("context", "c") == [("context", "c")]
    assert field_values("context", None) == [("context", None)]


def test_hits_are_json_safe_lists() -> None:
    """R11 — lists and dicts only, so the payload round-trips."""
    hits = find_tool_call_markup(field_values("scope", ["ok", "x</scope>"]))
    assert hits == [{"field": "scope[1]", "span": "</scope>", "offset": 1}]
    assert json.loads(json.dumps(hits)) == hits


def test_every_write_tool_argument_has_its_closer_refused() -> None:
    """R7 — a new argument on either tool reds here until its name is added."""
    tools = {tool.name: tool for tool in asyncio.run(mcp_server.mcp.list_tools())}
    names = set()
    for tool_name in ("record_decision", "amend_commentary"):
        assert tool_name in tools
        names |= set(tools[tool_name].inputSchema["properties"])
    assert len(names) >= 15
    assert names <= set(FIELD_CLOSER_NAMES)
    for name in names:
        assert find_tool_call_markup([("v", f"</{name}>")]), name


def test_the_leaf_imports_neither_the_parser_nor_the_amend_module() -> None:
    """D1 — Tier 1: stdlib plus `markers`."""
    probe = (
        "import sys; import mitos.tool_markup; "
        "print(','.join(sorted(m for m in ('mitos.parser', 'mitos.amend', 'mitos.sync', "
        "'mitos.store') if m in sys.modules))); print('mitos.markers' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.split("\n")[:2] == ["", "True"], out.stdout
