"""T11: ``list_scopes`` over a real ``mitos serve`` subprocess, and the harness's own guards.

Every row here drives an actual process — ``sys.executable -m mitos.cli serve``,
through ``cli.main``, over JSON-RPC on the child's pipes — because the two hazards
this vision closes live outside the function call and an in-process tool call is
structurally blind to both. See ``tests/mcp_harness.py`` for the transport.

Keyless and serviceless by construction: no API key is declared for any child, and
the write-path subprocesses point at a dead Qdrant. So this module runs on the bare
CI lane, and it touches neither the shared Qdrant instance nor the developer's real
``~/.config/mitos``. **No mocks** — the "external service" is a real subprocess,
which is the whole point.
"""

from __future__ import annotations

import asyncio
import gc
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mcp_harness import ServerStartupError, harness_env, mitos_server

#: A port nothing listens on. `mitos record` writes, and an absent `QDRANT_URL`
#: would send it at this box's live default instance, where it could leave behind a
#: collection whose path-hashed name conftest's `mitos-tmp*` sweep does not match.
#: Read-only children need no such guard, which is why `harness_env` does not
#: declare it and each write path opts in.
DEAD_QDRANT_URL = "http://127.0.0.1:9"

#: The seven tools the MCP surface exposes at this phase. Asserted as an exact
#: set, against the names as they arrive on the wire — never derived from an
#: import of `mitos.mcp_server`, which would make the row tautological. Phase 3c
#: widened it by one (`list_projects`, the discovery twin of CLI `projects`); a
#: later phase that adds a tool widens it again rather than deleting the row.
EXPECTED_TOOLS = {
    "list_decisions",
    "list_projects",
    "list_scopes",
    "query_decisions",
    "record_decision",
    "show_node",
    "surface_decisions",
}

TESTS_DIR = Path(__file__).resolve().parent

#: A one-tool MCP server that reports the environment its own process received.
#: It exists because **no mitos tool reports the child's config root** — the
#: hazard D2 guards is invisible at the `list_scopes`/`surface_decisions` surface,
#: so the only way to observe what the transport actually handed the child is to
#: ask a child that will say. It imports mitos (a subprocess may; the *harness*
#: may not) so the answer is mitos's own `config_home()`, not a re-derivation.
ENV_PROBE_SERVER = '''\
import json
import os

from mcp.server.fastmcp import FastMCP

from mitos.config import config_home, global_env_path, global_registry_path

mcp = FastMCP("EnvProbe")


@mcp.tool()
def describe_child() -> str:
    """Reports this process's environment and mitos-resolved config paths."""
    return json.dumps({
        "env": dict(os.environ),
        "cwd": os.getcwd(),
        "config_home": config_home(),
        "global_env_path": global_env_path(),
        "registry_path": global_registry_path(),
    })


mcp.run()
'''


# --------------------------------------------------------------------------- #
# Scaffolding
# --------------------------------------------------------------------------- #


def _scaffold_env(root):
    """The declared environment for the subprocesses that BUILD a workspace."""
    return harness_env(root, extra={"QDRANT_URL": DEAD_QDRANT_URL})


def _run_mitos(*args, cwd, env):
    """Runs one `mitos` verb as a subprocess under a declared environment."""
    done = subprocess.run(
        [sys.executable, "-m", "mitos.cli", *args],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, (
        f"`mitos {' '.join(args)}` failed (rc={done.returncode})\n"
        f"stdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    )
    return done


def _record(ws, scope, *, env):
    """Commits one decision carrying `scope`, keylessly, into an initialized workspace."""
    _run_mitos(
        "record", f"The {scope} workspace answers for itself.",
        "--rejected", "A workspace that reports another workspace's graph.",
        "--scope", scope, "--slug", f"{scope}-probe",
        cwd=ws, env=env,
    )


def _workspace(root, name, *, env, scopes=()):
    """Scaffolds a real workspace with a real `mitos init` subprocess."""
    ws = Path(root) / name
    ws.mkdir(parents=True)
    _run_mitos("init", cwd=ws, env=env)
    for scope in scopes:
        _record(ws, scope, env=env)
    return ws


def _tool_json(result):
    """Parses a tool result's payload — a JSON *string* on `content[0].text`.

    Deliberately not `structuredContent["result"]`, which holds the identical
    string: that wrapper is FastMCP's, not mitos's contract, and later phases
    should not come to depend on it.
    """
    assert result.isError is False, f"tool call errored: {result.content}"
    return json.loads(result.content[0].text)


# --------------------------------------------------------------------------- #
# Group 1 — T11: the smoke
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_list_scopes_answers_over_a_real_serve_subprocess(tmp_path):
    """A real server, launched from a real workspace, answers a real tool call.

    The first test in this codebase to watch mitos from the outside as a process.
    A fresh workspace answering `{}` is also I8's healthy-and-empty shape at this
    surface — a valid empty vocabulary, not an error.
    """
    ws = _workspace(tmp_path, "ws_a", env=_scaffold_env(tmp_path))

    async with mitos_server(cwd=ws, env=harness_env(tmp_path)) as server:
        tools = await server.session.list_tools()
        assert {tool.name for tool in tools.tools} == EXPECTED_TOOLS

        result = await server.session.call_tool("list_scopes", {})
        assert result.isError is False
        assert _tool_json(result) == {}


@pytest.mark.asyncio
async def test_the_smoke_needs_neither_a_key_nor_a_reachable_qdrant(tmp_path):
    """The harness's tier: no API key anywhere, Qdrant on a dead port, still green.

    Structural, not lucky. `list_scopes` is a pure graph read; the provider and the
    vector store are built inside `get_workspace_components`'s `try/except`, so a
    keyless provider degrades to `None`; and store construction dispatches no
    request, so a dead `QDRANT_URL` is inert. That is what keeps this module on the
    bare CI lane.

    The two assertions about the declared environment are the point of the row, not
    ceremony: if the child's config root ever escaped the redirect, it would read
    the developer's real `~/.config/mitos/.env` and this row would go green **with
    a key in hand** — proving nothing while looking like proof.
    """
    env = harness_env(tmp_path, extra={"QDRANT_URL": DEAD_QDRANT_URL})
    assert "GEMINI_API_KEY" not in env
    assert "GOOGLE_API_KEY" not in env

    ws = _workspace(tmp_path, "ws_a", env=env)

    # Tier 3 of the child's key resolution is `<config root>/mitos/.env`. Under the
    # redirect that file does not exist, so the child is keyless by construction.
    assert not (Path(env["XDG_CONFIG_HOME"]) / "mitos" / ".env").exists()

    async with mitos_server(cwd=ws, env=env) as server:
        assert _tool_json(await server.session.call_tool("list_scopes", {})) == {}


# --------------------------------------------------------------------------- #
# Group 2 — clean teardown
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_session_tears_down_without_leaking_pipes_or_loops(tmp_path, recwarn):
    """Owning a subprocess and four streams, the harness must leave none of them open.

    `stdio_client` yields `(read_stream, write_stream)` and no process handle, so
    there is no `returncode` to assert — this asserts what is actually observable.
    `recwarn` is mandatory rather than stylistic: pytest's warnings plugin only
    forces `always` for the deprecation warnings, so CPython's default
    `ignore::ResourceWarning` otherwise stands and an equivalent row written with
    `filterwarnings` alone would behave differently.

    The unraisable-exception half is deliberately *not* asserted here: pytest emits
    `PytestUnraisableExceptionWarning` at cleanup, after this body returns, so an
    in-body assertion claiming to check for one structurally cannot. pytest's own
    machinery fails the run on its own terms, loudly.

    **The measurement window is drained before it opens, and that is load-bearing
    rather than tidy** (added in 3c, which is where it first bit). `gc.collect()`
    finalizes garbage from *anywhere* in the pytest process, and `recwarn` records
    whatever warns while this test is running — so the collect below attributes
    every earlier module's unfinalized resource to this session. Phase 3c added the
    first module that drives the MCP tools for real rather than through a patched
    `get_workspace_components`, and those tools open a `GraphStore` they never
    close, so five `ResourceWarning: unclosed database` finalizations landed here
    and reddened a row about *pipes*. Measured, and it survives isolation: the two
    modules run together are green, because refcounting frees the connections
    promptly there; only under a full session do enough of them reach the collector.
    Draining first keeps the row's claim equal to its subject — the harness imports
    nothing from mitos and opens no database, so a sqlite finalization is by
    construction not its leak. 5b and 5c add more in-process rows of the same shape.
    """
    ws = _workspace(tmp_path, "ws_a", env=_scaffold_env(tmp_path))

    # Drain and discard whatever earlier modules left for the collector, so what
    # this row measures is this session and nothing else.
    gc.collect()
    recwarn.clear()

    async with mitos_server(cwd=ws, env=harness_env(tmp_path)) as server:
        assert _tool_json(await server.session.call_tool("list_scopes", {})) == {}

    # A handle that dies with a helper's frame reds this row by refcount alone; the
    # collect is what makes it hold for a leak that lands in a reference cycle
    # (anyio's memory-object streams being the plausible candidates).
    gc.collect()

    leaked = [w for w in recwarn.list if issubclass(w.category, ResourceWarning)]
    assert leaked == [], (
        "the session left resources open: "
        + "; ".join(f"{w.category.__name__}: {w.message}" for w in leaked)
    )


# --------------------------------------------------------------------------- #
# Group 3 — the harness's own guards
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_declared_environment_is_what_the_child_receives(tmp_path):
    """The tripwire between this module and the developer's real `~/.config/mitos`.

    `stdio_client` filters the parent environment down to six names before merging
    whatever we declare, and `XDG_CONFIG_HOME` is not one of them while `HOME` is.
    So conftest's autouse redirect does not reach any child launched here, and a
    harness written like the tree's other subprocess-driving modules — which build
    `env` from `os.environ` and get the redirect for free — would read the real
    global `.env` and write the real `registry.toml`. It would look identical in
    review. This row is the only thing standing in the way.

    It observes the child directly, because no mitos tool reports its config root:
    the server driven here is a one-tool probe that answers with its own
    `os.environ` and its own `mitos.config` resolutions. Drop `env=` from the
    harness's `StdioServerParameters` and every assertion below fails at once.
    """
    probe = tmp_path / "env_probe_server.py"
    probe.write_text(ENV_PROBE_SERVER, encoding="utf-8")
    ws = _workspace(tmp_path, "ws_a", env=_scaffold_env(tmp_path))
    declared = harness_env(tmp_path)

    async with mitos_server(cwd=ws, env=declared, args=(str(probe),)) as server:
        child = _tool_json(await server.session.call_tool("describe_child", {}))

    child_env = child["env"]

    # Every declared name arrived, with the declared value.
    for name, value in declared.items():
        assert child_env.get(name) == value, f"{name} did not reach the child"

    # …and nothing else did, beyond the six the transport always inherits. This is
    # the measurement behind "override, don't expect to clear": the six survive a
    # declared env, so no child launched through this transport has an empty one.
    inherited = {"HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER"}
    undeclared = set(child_env) - set(declared) - inherited
    # LC_CTYPE is the *interpreter's*, not the transport's: a declared environment
    # names no locale, so CPython's legacy-C-locale coercion (PEP 538) mints
    # `LC_CTYPE=C.UTF-8` into the child's own `os.environ` at startup. Named rather
    # than waved through with a blanket subset — any *other* undeclared name still
    # reds this — and admitted rather than required, since the coercion is a
    # platform behaviour (it picks whichever UTF-8 locale exists, and
    # `PYTHONCOERCECLOCALE=0` disables it) and not a property of this harness.
    assert undeclared <= {"LC_CTYPE"}, f"undeclared names leaked: {sorted(undeclared)}"
    if "LC_CTYPE" in child_env:
        assert child_env["LC_CTYPE"].upper().endswith("UTF-8")
    assert "GEMINI_API_KEY" not in child_env

    # The child's own mitos resolutions land under the redirect, not under `~`.
    assert Path(child["config_home"]) == Path(declared["XDG_CONFIG_HOME"])
    assert Path(child["global_env_path"]).is_relative_to(tmp_path)
    assert Path(child["registry_path"]).is_relative_to(tmp_path)


@pytest.mark.asyncio
async def test_the_launch_directory_is_what_the_server_binds(tmp_path):
    """A server answers for the directory it was launched from — today.

    `mcp_server`'s zero-arg `MitosConfig()` reads the process CWD, so the launch
    directory *is* the bound workspace. **Phase 5b flips exactly this**: once each
    tool takes a `project` selector, a server launched in A must be able to answer
    for B. When that lands, this row is the one to re-point — invert it to assert
    the selector wins over the launch directory. It is not to be deleted: the
    launch directory remains the default, and something must still pin it.
    """
    env = _scaffold_env(tmp_path)
    ws_a = _workspace(tmp_path, "ws_a", env=env, scopes=("alpha",))
    _workspace(tmp_path, "ws_b", env=env, scopes=("beta",))

    async with mitos_server(cwd=ws_a, env=env) as server:
        scopes = _tool_json(await server.session.call_tool("list_scopes", {}))

    assert set(scopes) == {"alpha"}


@pytest.mark.asyncio
async def test_a_named_project_retargets_a_server_launched_somewhere_else(tmp_path):
    """The vision's whole point, over a real server: the call says where.

    The row above pins that the launch directory is still the *default*. This one
    pins that it is no longer the *answer*: a server launched in A, told
    `project=B`, answers about B. Only a subprocess can prove it — an in-process
    call shares pytest's environment and never enters `cli.main()`, so it is
    structurally blind to the two hazards that live outside the function call.

    The registry is planted at the path the **child** will read, taken from
    `harness_env`'s own dict. `registry.registry_path()` would resolve the
    *parent's* conftest redirect instead, and the child would find no registry at
    all — an absent registry is a healthy empty state, so the row would go green
    having proven nothing.
    """
    env = _scaffold_env(tmp_path)
    ws_a = _workspace(tmp_path, "ws_a", env=env, scopes=("alpha",))
    ws_b = _workspace(tmp_path, "ws_b", env=env, scopes=("beta",))

    registry_file = Path(env["XDG_CONFIG_HOME"]) / "mitos" / "registry.toml"
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_file.write_text(f'"bee" = "{ws_b}"\n', encoding="utf-8")

    async with mitos_server(cwd=ws_a, env=env) as server:
        by_name = _tool_json(
            await server.session.call_tool("list_scopes", {"project": "bee"}))
        by_path = _tool_json(
            await server.session.call_tool("list_scopes", {"project": str(ws_b)}))
        default = _tool_json(await server.session.call_tool("list_scopes", {}))
        listed = _tool_json(await server.session.call_tool("list_projects", {}))

    assert set(by_name) == {"beta"}
    assert set(by_path) == {"beta"}
    assert set(default) == {"alpha"}
    assert listed["projects"] == [{"name": "bee", "path": str(ws_b)}]


@pytest.mark.asyncio
async def test_a_bad_selector_arrives_error_marked_carrying_the_whole_anatomy(tmp_path):
    """The delivery mechanism, end to end — measured here rather than reasoned about.

    Three shapes were possible and only one is right, so this row pins which one
    the tree actually ships: returning the anatomy makes an addressing failure an
    ordinary success (`isError: False`); letting the typed error propagate marks
    the result as an error but delivers only its terse discriminator-level
    fallback, which carries none of the recovery data; raising the *rendered*
    body delivers both. The middle one is the trap, because it looks entirely
    correct from the outside.

    `_tool_json` is deliberately not used — it asserts `isError is False`, which
    is the opposite of the contract here.
    """
    env = _scaffold_env(tmp_path)
    ws_a = _workspace(tmp_path, "ws_a", env=env, scopes=("alpha",))

    registry_file = Path(env["XDG_CONFIG_HOME"]) / "mitos" / "registry.toml"
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_file.write_text(f'"alpha-project" = "{ws_a}"\n', encoding="utf-8")

    async with mitos_server(cwd=ws_a, env=env) as server:
        result = await server.session.call_tool(
            "surface_decisions", {"query": "anything", "project": "alpha-projekt"})

    assert result.isError is True
    body = result.content[0].text
    # FastMCP's own prefix is kept rather than fought: it names the failing tool.
    assert body.startswith("Error executing tool surface_decisions: ")
    assert "no project named 'alpha-projekt' is registered" in body
    assert "Did you mean: alpha-project" in body
    assert "Registered projects: alpha-project." in body
    assert "surface_decisions(project='alpha-project', …)" in body
    assert "`list_projects()`" in body
    assert "Traceback" not in body
    # The terse fallback would have been this and nothing else.
    assert body.strip() != (
        "Error executing tool surface_decisions: unknown project 'alpha-projekt'")
    for syntax in ("--project", "mitos init", "mitos projects"):
        assert syntax not in body


@pytest.mark.asyncio
async def test_a_server_that_cannot_start_reports_its_stderr(tmp_path):
    """A dead child is a vector, not a wall.

    Unwrapped, this failure arrives as an anyio `ExceptionGroup` wrapping another
    `ExceptionGroup`, and the real cause is at *no* depth of that chain — it exists
    only in the child's stderr, which by default nobody has thought to open. That
    is a burned debugging run for whoever meets it at 2am, so the harness reads the
    file and names the failure. It is also how WIRING_LEDGER entry-001 will read
    when 6a meets it: `mitos serve` cannot start from a fresh pipx install, and
    this is the shape that failure takes.

    A *hung* child is the second half, and it is the failure mode a long-lived 5b
    or 5c session most needs bounded: the `deadline` must actually be enforced, or
    a server that never answers blocks the run rather than failing it.
    """
    ws = _workspace(tmp_path, "ws_a", env=_scaffold_env(tmp_path))
    dying = ("-c", "import sys; sys.stderr.write('boom: no fastmcp\\n'); sys.exit(1)")

    with pytest.raises(ServerStartupError) as excinfo:
        async with mitos_server(cwd=ws, env=harness_env(tmp_path), args=dying):
            raise AssertionError("the harness yielded a session over a dead child")

    error = excinfo.value
    rendered = str(error)
    assert "boom: no fastmcp" in rendered
    assert "boom: no fastmcp" in error.stderr
    assert sys.executable in error.command
    assert str(ws) == error.cwd
    assert "unhandled errors in a TaskGroup" not in rendered

    # A child that starts and then never answers fails on the deadline, not on the
    # patience of whoever is watching the run.
    hanging = ("-c", "import time; time.sleep(120)")
    with pytest.raises(ServerStartupError) as excinfo:
        async with mitos_server(
            cwd=ws, env=harness_env(tmp_path), args=hanging, deadline=3.0
        ):
            raise AssertionError("the harness yielded a session over a hung child")

    assert "within 3.0s" in str(excinfo.value)
    assert excinfo.value.stderr == ""


@pytest.mark.asyncio
async def test_the_harness_never_relabels_a_failure_that_is_not_its_own(tmp_path):
    """The wall D4 removes from the startup path must not survive on the common one.

    Two failures belong to the caller, not to the harness, and both used to arrive
    wearing the transport's clothes:

    A **failed assertion inside the session body** crosses two anyio task groups
    (``stdio_client``'s and ``ClientSession``'s), each of which wraps it — so it
    reached the caller as `ExceptionGroup([ExceptionGroup([AssertionError])])`,
    `pytest.raises(AssertionError)` no longer matched it, and every red row in 3c,
    5b and 5c would have read as a transport crash. Measured, not predicted.

    An **outer cancellation** — `asyncio.wait_for`, a pytest timeout, a sibling
    task — used to be caught by the startup guard and re-labelled
    `ServerStartupError`, so the caller's own timeout reported this harness's error
    instead of its own, with a deadline in the message that had nothing to do with
    the failure.
    """
    ws = _workspace(tmp_path, "ws_a", env=_scaffold_env(tmp_path))

    with pytest.raises(AssertionError, match="the caller's own assertion") as excinfo:
        async with mitos_server(cwd=ws, env=harness_env(tmp_path)) as server:
            assert _tool_json(await server.session.call_tool("list_scopes", {})) == {}
            raise AssertionError("the caller's own assertion")

    # Not merely matchable — the group must be gone from the chain, or it renders
    # above the assertion in the traceback and the reader meets it anyway.
    assert not isinstance(excinfo.value, BaseExceptionGroup)
    assert not isinstance(excinfo.value.__context__, BaseExceptionGroup)

    async def _hold_a_hung_child():
        async with mitos_server(
            cwd=ws, env=harness_env(tmp_path), args=("-c", "import time; time.sleep(120)")
        ):
            raise AssertionError("the harness yielded a session over a hung child")

    # The caller's timeout wins, and says so: `wait_for` can only report its own
    # TimeoutError if the cancellation it sent reaches it as a cancellation.
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(_hold_a_hung_child(), timeout=2.0)


def test_only_json_rpc_reaches_the_transport(tmp_path):
    """Nothing but JSON-RPC reaches the server's stdout — the protocol channel.

    `cmd_serve` prints a startup banner to **stdout**, and gets away with it by
    accident: `mcp.server.stdio.stdio_server` wraps a fresh `TextIOWrapper` around
    `sys.stdout.buffer`, and the original block-buffered text never reaches the
    pipe. A line-buffered TTY, an added `flush()`, or a different wrapper lifetime
    would each put garbage on the protocol channel. That is not this phase's to fix
    (`mitos/` carries no diff here) — it is this phase's to pin, so 3c and 5b
    inherit a net rather than the accident.

    This row cannot use the harness: `stdio_client` consumes the stream, so raw
    stdout is unreachable from inside it. It drives a hand-written request stream
    through a plain subprocess and exits on EOF.

    **It must reach a tool handler, not merely the handshake.** Measured: a stray
    flushed `print` inside a tool handler puts its own line on the pipe *and*
    drags the buffered startup banner out with it, while a handshake-only exchange
    stays pure whether or not such a print exists — so an initialize-only version
    of this row is inert against the regression it exists to catch. The honest
    limit, stated rather than papered over: a stray print that never flushes never
    reaches the pipe either, so this catches exactly the class that can corrupt
    the channel.
    """
    env = _scaffold_env(tmp_path)
    ws = _workspace(tmp_path, "ws_a", env=env)
    exchange = "".join(json.dumps(message) + "\n" for message in (
        {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "transport-probe", "version": "0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "list_scopes", "arguments": {}},
        },
    ))

    done = subprocess.run(
        [sys.executable, "-m", "mitos.cli", "serve"],
        input=exchange, cwd=str(ws), env=env,
        capture_output=True, text=True, timeout=120,
    )

    lines = [line for line in done.stdout.splitlines() if line.strip()]
    assert len(lines) == 2, (
        f"expected the initialize result and the tool result, got {len(lines)} "
        f"lines:\n{done.stdout}\nstderr:\n{done.stderr}"
    )
    for line in lines:
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"non-JSON line on the protocol channel: {line!r} ({exc})"
            ) from exc
    # Named explicitly, because it is the one line we already know wants out.
    assert "Starting Mitos MCP Server" not in done.stdout


def test_the_harness_refuses_an_undeclared_environment_or_launch_directory(tmp_path):
    """Neither the environment nor the launch directory may be left implicit.

    Both are the topology under test — a defaulted `env` would silently inherit
    whatever the transport chose to pass through (and reach the real config root),
    and a defaulted `cwd` would bind whatever directory pytest happened to run in.
    Keyword-only and required, so the refusal is at the call, not at the handshake.
    """
    with pytest.raises(TypeError):
        mitos_server(cwd=tmp_path)
    with pytest.raises(TypeError):
        mitos_server(env=harness_env(tmp_path))
    with pytest.raises(TypeError):
        mitos_server()


def test_the_harness_module_imports_nothing_from_mitos():
    """`tests/mcp_harness.py` is a leaf over `mcp` and the standard library.

    The harness must be able to drive a server whose in-process import is itself
    the thing under test — 5b rewrites `mcp_server`'s eight zero-arg
    `MitosConfig()` sites — so importing mitos into the *driver* would re-create in
    miniature the in-process coupling this whole substrate exists to escape. The
    same subprocess import-closure probe the tree's other leaves use.
    """
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); import mcp_harness; "
        "print(','.join(sorted(m for m in sys.modules "
        "if m == 'mitos' or m.startswith('mitos.'))))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe, str(TESTS_DIR)],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "", f"the harness imported mitos: {out.stdout.strip()}"


@pytest.mark.asyncio
async def test_one_long_lived_session_spans_calls_and_a_filesystem_mutation(tmp_path):
    """The topology phase 5c needs, proven available before 5c has to discover it.

    One long-lived session, launched from a chosen directory under a fully declared
    environment, spanning several calls with a filesystem mutation between them —
    5c's I6/I7 rows are shaped exactly like this, and the reason 3a exists ahead of
    its consumers is so that shape is not improvised mid-red three phases from now.

    It also pins a real property: the server holds no cached view of the graph, so
    a corpus that changes under a live session is visible on the next call.
    """
    env = _scaffold_env(tmp_path)
    ws = _workspace(tmp_path, "ws_a", env=env)

    async with mitos_server(cwd=ws, env=env) as server:
        before = _tool_json(await server.session.call_tool("list_scopes", {}))
        assert before == {}

        _record(ws, "alpha", env=env)

        after = _tool_json(await server.session.call_tool("list_scopes", {}))
        assert set(after) == {"alpha"}
        assert after["alpha"]["active_decisions"] == 1
