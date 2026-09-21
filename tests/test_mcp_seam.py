"""The ``call_tool`` seam canary: what an overriding ``FastMCP`` subclass can see.

Phase 6b is planned to build unknown-argument refusal on one framework seam: a
subclass overriding ``FastMCP.call_tool(self, name, arguments)``, which FastMCP registers
with the low-level server as ``call_tool(validate_input=False)``. These rows pin
the six properties that build stands on, over a **throwaway** server defined
here, never ``mitos.mcp_server.mcp`` (6b owns that instance; a test-time
mutation of the module global would leak across xdist-shared imports).

Every row goes through the in-memory wire
(``mcp.shared.memory.create_connected_server_and_client_session``): calling a
tool function directly skips FastMCP, and with it the raw-argument view and the
structured validation error these rows exist to measure.

A red row here after an ``mcp`` bump says *the framework moved*, which is a
different message from *mitos broke*. Measured green on 1.26.0 (the declared
floor), 1.27.2 and 1.30.0, 2026-09-21.
"""

from typing import List, Optional

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import ValidationError

from mitos.mcp_server import _RenderedToolError


class _SeamProbe(FastMCP):
    """Refuses an argument the tool does not declare; records what it saw.

    ``fault`` is the exception class the refusal raises, so S2 and S3 share one
    override. ``seen`` holds each ``(name, arguments)`` exactly as it arrived,
    and ``declared`` the schema the public accessor gave from inside the call.
    """

    def __init__(self, fault: type) -> None:
        super().__init__("seam-probe")
        self.fault = fault
        self.seen: list = []
        self.declared: dict = {}

    async def call_tool(self, name, arguments):
        self.seen.append((name, dict(arguments)))
        tool = next((t for t in await self.list_tools() if t.name == name), None)
        if tool is not None:
            self.declared[name] = tool.inputSchema
            unknown = sorted(set(arguments) - set(tool.inputSchema["properties"]))
            if unknown:
                raise self.fault(f"unknown argument {unknown[0]} on {name}")
        return await super().call_tool(name, arguments)


def _probe(fault: type = ValueError) -> _SeamProbe:
    server = _SeamProbe(fault)

    @server.tool()
    def f(
        a: str,
        n: int,
        context: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        return repr(context)

    return server


async def _call(server: _SeamProbe, name: str, arguments: dict):
    async with create_connected_server_and_client_session(
        server._mcp_server
    ) as session:
        return await session.call_tool(name, arguments)


@pytest.mark.asyncio
async def test_s1_override_sees_raw_arguments_before_the_null_rewrite() -> None:
    """S1: the override sees ``"null"`` as sent; the tool receives ``None``.

    FastMCP's ``pre_parse_json`` turns the string ``"null"`` into ``None``
    after the override has run, so only the override can say what *arrived*.
    """
    server = _probe()
    result = await _call(server, "f", {"a": "x", "n": 1, "context": "null"})
    assert not result.isError
    assert ("f", {"a": "x", "n": 1, "context": "null"}) in server.seen
    assert result.content[0].text == "None"


@pytest.mark.parametrize("fault", [ValueError, _RenderedToolError])
@pytest.mark.asyncio
async def test_s2_s3_a_fault_raised_in_the_override_arrives_bare(fault) -> None:
    """S2 (any exception) and S3 (``_RenderedToolError``): ``isError`` with a
    body equal to ``str(exc)`` — **no** ``"Error executing tool …"`` prefix, so
    a voice raised from the override must name the tool itself."""
    server = _probe(fault)
    result = await _call(server, "f", {"a": "x", "n": 1, "bogus": 1})
    assert result.isError
    assert result.content[0].text == "unknown argument bogus on f"


@pytest.mark.asyncio
async def test_s4_super_raises_every_argument_fault_as_structured_errors() -> None:
    """S4: delegating a bad call to ``super()`` raises ``ToolError`` whose
    ``__cause__`` is pydantic's ``ValidationError``, listing every fault at
    once — the missing one and the mistyped one together."""
    server = _probe()
    with pytest.raises(ToolError) as caught:
        await server.call_tool("f", {"n": "notint"})
    cause = caught.value.__cause__
    assert isinstance(cause, ValidationError)
    faults = {(e["type"], e["loc"]) for e in cause.errors()}
    assert faults == {("missing", ("a",)), ("int_parsing", ("n",))}

    result = await _call(_probe(), "f", {"n": "notint"})
    assert result.isError
    assert result.content[0].text.startswith("Error executing tool f: ")


@pytest.mark.asyncio
async def test_s5_list_tools_is_the_public_schema_accessor_inside_the_override() -> None:
    """S5: ``await self.list_tools()`` gives the declared properties and
    ``required`` from inside the override. It is public; ``_tool_manager`` is
    the private alternative and is not the answer 6b inherits."""
    server = _probe()
    await _call(server, "f", {"a": "x", "n": 1})
    schema = server.declared["f"]
    assert set(schema["properties"]) == {"a", "n", "context", "tags"}
    assert set(schema["required"]) == {"a", "n"}


@pytest.mark.asyncio
async def test_s6_an_unknown_tool_still_answers_as_it_does_today() -> None:
    """S6: an unknown tool name passes through the override to ``super()``
    and comes back as FastMCP's own ``Unknown tool`` fault."""
    server = _probe()
    result = await _call(server, "nope", {"bogus": 1})
    assert ("nope", {"bogus": 1}) in server.seen
    assert result.isError
    assert result.content[0].text == "Unknown tool: nope"
