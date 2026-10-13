"""A7's boundary: one argument-fault voice, and an unknown argument refused.

``mitos.mcp_server.mcp`` is a ``_MitosFastMCP``, whose ``call_tool`` refuses an
undeclared argument and answers every argument fault of a call at once, before
any tool code runs. The unit rows drive ``_render_argument_faults`` (pure); the
boundary rows go through the in-memory wire on mitos's own instance, read-only —
nothing here adds a tool to it or mutates it. The real ``mitos serve`` frame
lives in ``tests/test_mcp_stdio_harness.py``; 6a's seam canary (S1–S7) in
``tests/test_mcp_seam.py``.
"""

import json

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import ValidationError

from mitos import mcp_server
from mitos.mcp_server import (ARGUMENT_ECHO_MAX, _RenderedToolError,
                              _render_argument_faults, mcp)
from tests.test_mcp_selector import FORBIDDEN_SYNTAX

TOOL_NAMES = ("surface_decisions", "list_decisions", "list_scopes", "show_node",
              "query_decisions", "record_decision", "amend_commentary", "list_projects")

# What no argument-fault body may carry: a shell command (CC-6), pydantic's wall
# (its URL and the `…Arguments` model title), and FastMCP's prefix, which this
# path never gets and must not imitate.
FORBIDDEN_IN_BODY = (*FORBIDDEN_SYNTAX, "mitos ", "errors.pydantic.dev",
                     "Arguments", "Error executing tool", "validation error")


def _refusal(tool: str, arguments: dict) -> str:
    """Runs the boundary in-process on the real tool; returns the rendered body."""
    captured = mcp._captured_arguments[tool]
    known = {k: v for k, v in arguments.items() if k in captured.properties}
    unknowns = [k for k in arguments if k not in captured.properties]
    pre_parsed = captured.metadata.pre_parse_json(known)
    try:
        captured.metadata.arg_model.model_validate(pre_parsed)
        errors = []
    except ValidationError as exc:
        errors = exc.errors()
    assert unknowns or errors, f"{tool}{arguments} fits; nothing to render"
    return _render_argument_faults(tool, captured.properties, arguments, pre_parsed,
                                   unknowns, errors)


async def _wire(tool: str, arguments: dict):
    async with create_connected_server_and_client_session(mcp._mcp_server) as session:
        return await session.call_tool(tool, arguments)


# --- unit rows: the renderer ------------------------------------------------------


def test_each_fault_kind_renders_its_line_in_order_with_an_honest_count():
    """Row 1 — unknown (with and without a suggestion) → missing → mistyped; the
    count is of faulted arguments; the last line lists parameters in signature order."""
    body = _refusal("record_decision", {
        "limit": 5, "supersede": "x", "rejected_paths": 7,
        "scope": [1, 2], "slug": "s"})
    lines = body.split("\n")
    assert lines[0] == "record_decision was not run: 5 argument faults."
    assert lines[1] == "  `limit` is not an argument of record_decision."
    assert lines[2] == ("  `supersede` is not an argument of record_decision"
                        " — did you mean `supersedes`?")
    assert lines[3] == "  `axiom` is required and was not sent."
    assert lines[4] == "  `rejected_paths` expects a string; it received the number 7."
    assert lines[5] == ("  `scope` expects a list of strings; it received the list [1, 2]"
                        " (the faults are at items 0, 1).")
    declared = list(mcp._captured_arguments["record_decision"].properties)
    assert lines[6] == f"  record_decision takes: {', '.join(declared)}."
    assert declared[:4] == ["axiom", "rejected_paths", "scope", "slug"]
    assert len(lines) == 7


def test_one_fault_is_singular_and_a_parameterless_tool_says_so():
    body = _refusal("list_projects", {"project": "x"})
    assert body == ("list_projects was not run: 1 argument fault.\n"
                    "  `project` is not an argument of list_projects.\n"
                    "  list_projects takes no arguments.")


@pytest.mark.parametrize("tool, unknown, expected", [
    ("show_node", "id", ["ident"]),
    ("record_decision", "axiom_scope", ["axiom", "scope"]),  # a ratio tie: signature order
    ("surface_decisions", "fulltop", ["full_top"]),
    ("record_decision", "supersede", ["supersedes"]),
    ("surface_decisions", "projet", ["project"]),
    ("record_decision", "rejected", ["rejected_paths"]),
    ("surface_decisions", "claim", []),  # 0.6 would say `limit` — the K4 trap
    ("record_decision", "tags", []),
    ("show_node", "name", []),
    ("amend_commentary", "rationale", []),
    ("record_decision", "s", []),  # one letter "contains" into nearly everything
])
def test_did_you_mean_is_the_tools_own_names_by_containment_or_a_close_ratio(
        tool, unknown, expected):
    """Row 2 — D6's table."""
    first = _refusal(tool, {unknown: 1}).split("\n")[1]
    if expected:
        assert first.endswith(
            f"did you mean {' or '.join(f'`{name}`' for name in expected)}?"), first
    else:
        assert "did you mean" not in first and first.endswith(f"of {tool}."), first


def test_a_suggestion_never_names_an_argument_already_sent():
    body = _refusal("show_node", {"ident": "x", "id": "y"})
    assert "did you mean" not in body and "`id` is not an argument of show_node." in body


def test_a_mistyped_line_shows_what_arrived_not_what_pydantic_saw():
    """Row 3 — D4: the raw string, the JSON-rewrite note, never the decoded list."""
    body = _refusal("record_decision", {
        "axiom": "a", "rejected_paths": "r", "scope": ["t"], "slug": "s",
        "supersedes": '["a","b"]'})
    line = body.split("\n")[1]
    assert line == ("  `supersedes` expects a string; it received the string "
                    '"[\\"a\\",\\"b\\"]", which reads as JSON and was taken as a list.')
    assert "['a', 'b']" not in body

    as_object = _refusal("surface_decisions", {"query": "q", "scope": '{"a": 1}'})
    assert "which reads as JSON and was taken as an object." in as_object

    sent_as_list = _refusal("surface_decisions", {"query": "q", "scope": ["a"]})
    assert "it received the list [\"a\"]." in sent_as_list
    assert "reads as JSON" not in sent_as_list

    one_item = _refusal("record_decision", {
        "axiom": "a", "rejected_paths": "r", "scope": [1], "slug": "s"})
    assert "(the fault is at item 0)." in one_item


def test_a_string_rewritten_to_null_says_so():
    """`"null"` on a non-`str` field becomes `None` before validation; where that
    faults (a required list, a bool) the line says it was taken as null."""
    body = _refusal("record_decision", {
        "axiom": "a", "rejected_paths": "r", "scope": "null", "slug": "s",
        "acknowledge_neighbors": "null"})
    assert ('`scope` expects a list of strings; it received the string "null", '
            "which reads as JSON and was taken as null.") in body
    assert ('`acknowledge_neighbors` expects true or false; it received the string '
            '"null", which reads as JSON and was taken as null.') in body


def test_a_long_value_is_truncated_at_the_named_constant():
    blob = "x" * (ARGUMENT_ECHO_MAX * 3)
    body = _refusal("surface_decisions", {"query": "q", "brief": blob})
    line = body.split("\n")[1]
    echo = line.split("it received the string ", 1)[1]
    assert echo.endswith("….") and len(echo) == ARGUMENT_ECHO_MAX + 1
    assert blob not in body


def test_an_unknown_arguments_value_is_never_echoed():
    """G5 — the name is the fault; the value may be a prose blob."""
    body = _refusal("amend_commentary", {"slug": "s", "rationale": "SECRET-PROSE"})
    assert "SECRET-PROSE" not in body


def test_null_and_other_json_kinds_read_naturally():
    body = _refusal("record_decision", {
        "axiom": None, "rejected_paths": "r", "scope": ["t"], "slug": "s",
        "acknowledge_neighbors": "maybe", "mechanisms": {"a": 1}})
    assert "`axiom` expects a string; it received null." in body
    assert '`acknowledge_neighbors` expects true or false; it received the string "maybe".' in body
    assert '`mechanisms` expects a list of strings; it received the object {"a": 1}.' in body
    fraction = _refusal("surface_decisions", {"query": "q", "limit": 5.5})
    assert "`limit` expects a whole number; it received the number 5.5." in fraction


def test_no_parameter_is_marked_required_in_the_parameter_line():
    """D5 — `project` is documented-required, not schema-required."""
    body = _refusal("show_node", {"id": "x"})
    tail = body.split("\n")[-1]
    assert tail == "  show_node takes: ident, project." and "required" not in tail


_FORBIDDEN_CASES = [
    ("show_node", {"id": "x"}),
    ("surface_decisions", {"limit": "many"}),
    ("record_decision", {"bogus": 1, "supersedes": '["a"]', "scope": [1]}),
    ("amend_commentary", {"slug": "s", "rationale": "r", "clear": "context"}),
    ("list_projects", {"project": "mitos"}),
    ("list_scopes", {"include_archived": {"x": 1}}),
    ("query_decisions", {"query": 1, "depht": "letter"}),
    ("list_decisions", {"sate": "active", "brief": []}),
]


@pytest.mark.parametrize("tool, arguments", _FORBIDDEN_CASES)
def test_no_body_carries_a_shell_command_pydantics_wall_or_the_framework_prefix(
        tool, arguments):
    """Row 4 — CC-6 plus the wall this replaces."""
    body = _refusal(tool, arguments)
    assert body.startswith(f"{tool} was not run: ")
    for banned in FORBIDDEN_IN_BODY:
        assert banned not in body, (banned, body)


# --- boundary rows: mitos's own instance over the in-memory wire ------------------


@pytest.mark.asyncio
async def test_the_captured_models_are_the_ones_fastmcp_registered():
    """Row 5 — parity (D2) over all eight real tools: names, schemas, and a bad
    call's ``errors()`` against FastMCP's own validation on a throwaway instance."""
    registered = {tool.name: tool.inputSchema for tool in await mcp.list_tools()}
    assert set(registered) == set(TOOL_NAMES)
    assert set(mcp._captured_arguments) == set(registered)

    plain = FastMCP("parity")
    for name in TOOL_NAMES:
        captured = mcp._captured_arguments[name]
        schema = captured.metadata.arg_model.model_json_schema(by_alias=True)
        assert schema == registered[name], name
        assert list(captured.properties) == list(registered[name].get("properties", {}))
        plain.add_tool(getattr(mcp_server, name))

    for name in TOOL_NAMES:
        captured = mcp._captured_arguments[name]
        if not captured.properties:
            continue
        bad = {key: {"wrong": 1} for key in captured.properties}
        with pytest.raises(ValidationError) as ours:
            captured.metadata.arg_model.model_validate(captured.metadata.pre_parse_json(bad))
        with pytest.raises(ToolError) as theirs:
            await plain.call_tool(name, bad)
        assert isinstance(theirs.value.__cause__, ValidationError), name
        assert ours.value.errors() == theirs.value.__cause__.errors(), name


@pytest.mark.asyncio
async def test_a_mixed_call_is_answered_whole_in_one_response():
    """Row 6 — an unknown, a missing and a mistyped argument in one body: the
    batching D1 pre-validates for."""
    result = await _wire("record_decision", {
        "rationale": "why", "rejected_paths": "r", "scope": "not-a-list", "slug": "s"})
    assert result.isError is True
    body = result.content[0].text
    assert body.startswith("record_decision was not run: 3 argument faults.\n")
    assert "`rationale` is not an argument of record_decision." in body
    assert "`axiom` is required and was not sent." in body
    assert ('`scope` expects a list of strings; it received the string "not-a-list".'
            in body)
    assert "Error executing tool" not in body


@pytest.mark.asyncio
async def test_a_valid_call_is_byte_identical_and_pydantic_leniency_survives():
    """Row 7 — a clean call reaches the tool unchanged; `"true"` for a bool still
    coerces (G8), reaching the tool's own returned refusal exactly as a direct call."""
    listed = await _wire("list_projects", {})
    assert listed.isError is False
    assert listed.content[0].text == mcp_server.list_projects()

    lenient = await _wire("list_decisions", {"brief": "true", "oneline": True,
                                             "project": "anything"})
    assert lenient.isError is False
    assert lenient.content[0].text == mcp_server.list_decisions(
        brief=True, oneline=True, project="anything")
    assert json.loads(lenient.content[0].text)["error"].startswith(
        "brief and oneline are mutually exclusive")


@pytest.mark.asyncio
async def test_an_unknown_tool_still_passes_through_to_fastmcp():
    """S6 on the real instance: no captured model, so FastMCP answers."""
    result = await _wire("no_such_tool", {"bogus": 1})
    assert result.isError is True
    assert result.content[0].text == "Unknown tool: no_such_tool"


@pytest.mark.asyncio
async def test_the_boundary_raises_the_rendered_error_before_any_tool_code(monkeypatch):
    """The refusal is ``_RenderedToolError`` and the tool body never runs."""
    ran = []
    monkeypatch.setattr(FastMCP, "call_tool",
                        lambda self, name, arguments: ran.append(name))
    with pytest.raises(_RenderedToolError) as caught:
        await mcp.call_tool("show_node", {"id": "x"})
    assert str(caught.value).startswith("show_node was not run: 2 argument faults.")
    assert ran == []
