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
import threading
import time
from pathlib import Path

import pytest

from mcp_harness import ServerStartupError, harness_env, mitos_server
#: The collection derivation itself, so an expected collection is never a
#: hand-built `f"mitos-{name}"` literal (1d's W11). A pure path→name function:
#: importing it into the *parent* opens no store and reads no environment, and the
#: no-mitos-import rule below binds `tests/mcp_harness.py`, not this module.
from mitos.config import default_collection_name

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
    """Commits one decision carrying `scope`, keylessly, into an initialized workspace.

    The workspace is named by absolute path: `init` registers it under the
    directory's basename, but these fixtures build several workspaces per run and
    the path form is the one that cannot collide. (`init` itself stays bare — it is
    selector-exempt.)
    """
    _run_mitos(
        "-p", str(ws), "record", f"The {scope} workspace answers for itself.",
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
    A fresh workspace answering an empty `scopes` map is also I8's healthy-and-empty
    shape at this surface — a valid empty vocabulary, not an error, now inside the
    provenance envelope that says which project was empty.
    """
    ws = _workspace(tmp_path, "ws_a", env=_scaffold_env(tmp_path))

    async with mitos_server(cwd=ws, env=harness_env(tmp_path)) as server:
        tools = await server.session.list_tools()
        assert {tool.name for tool in tools.tools} == EXPECTED_TOOLS

        result = await server.session.call_tool("list_scopes", {"project": str(ws)})
        assert result.isError is False
        assert _tool_json(result)["scopes"] == {}


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
        assert _tool_json(await server.session.call_tool(
            "list_scopes", {"project": str(ws)}))["scopes"] == {}


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
        assert _tool_json(await server.session.call_tool(
            "list_scopes", {"project": str(ws)}))["scopes"] == {}

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
async def test_the_launch_directory_binds_nothing(tmp_path):
    """Entry-007's second MCP tripwire, INVERTED at 5b — over a real process.

    3a asserted the opposite, and said so: `mcp_server`'s zero-arg `MitosConfig()`
    read the process CWD, so the launch directory *was* the bound workspace, and
    this row was written to be re-pointed rather than deleted. 5b deletes the
    fallback, so the claim turns over: the launch directory binds nothing at all,
    and a call that names no project is refused even when the process is standing
    in a perfectly good workspace holding the very scope the row could name.

    Only a subprocess can state it. In-process, pytest's cwd is the mitos-pub
    checkout — itself a valid workspace — so a surviving fallback would resolve
    *something* and the distinction would be invisible. Here `cwd=ws_a` is a real
    workspace carrying `alpha`, and the assertion is that `alpha` reaches nothing.

    `-C`'s surviving role is unaffected and is pinned elsewhere
    (`test_serve_binds_target`): the flag still moves the process, it just no
    longer decides whose corpus a call touches.
    """
    env = _scaffold_env(tmp_path)
    ws_a = _workspace(tmp_path, "ws_a", env=env, scopes=("alpha",))
    _workspace(tmp_path, "ws_b", env=env, scopes=("beta",))

    async with mitos_server(cwd=ws_a, env=env) as server:
        result = await server.session.call_tool("list_scopes", {})

    # Read the body directly: `_tool_json` asserts `isError is False` and would
    # fail before the anatomy could be looked at.
    assert result.isError is True
    body = result.content[0].text
    assert body.startswith("Error executing tool list_scopes: ")
    assert "no project was named" in body
    assert "alpha" not in body and str(ws_a) not in body
    assert "list_projects()" in body
    assert "Traceback" not in body


@pytest.mark.asyncio
async def test_a_named_project_retargets_a_server_launched_somewhere_else(tmp_path):
    """The vision's whole point, over a real server: the call says where.

    The row above pins that the launch directory binds nothing. This one pins the
    other half in the same long-lived session — success criterion 4 entire: a
    server launched in A answers a `project=B` call about **B**, in both selector
    forms, and refuses the selector-less one. Only a subprocess can prove it — an
    in-process call shares pytest's environment and never enters `cli.main()`, so
    it is structurally blind to the two hazards that live outside the function
    call.

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
        # The third form, and since 5b it is a refusal rather than a default.
        # `ws_a` is unregistered in this row (the registry was overwritten with
        # `bee` alone) and the process is standing in it — the one arrangement
        # where a surviving fallback would answer `alpha` and read as a pass.
        refused = await server.session.call_tool("list_scopes", {})
        listed = _tool_json(await server.session.call_tool("list_projects", {}))

        # An empty registry is a healthy state, and it must arrive as an ordinary
        # SUCCESS — `_tool_json` asserts `isError is False`, which is the half an
        # in-process row cannot see. Driven by removing the file mid-session,
        # which rides the mutation-between-calls topology 3a proved available and
        # doubles as evidence that the server caches no view of the registry.
        registry_file.unlink()
        empty = _tool_json(await server.session.call_tool("list_projects", {}))

    assert set(by_name["scopes"]) == {"beta"}
    assert set(by_path["scopes"]) == {"beta"}
    assert refused.isError is True
    assert "no project was named" in refused.content[0].text
    assert "alpha" not in refused.content[0].text
    assert listed["projects"] == [{"name": "bee", "path": str(ws_b)}]
    assert empty == {"registry_path": str(registry_file), "count": 0, "projects": []}

    # Criterion 5, over the same real server: the answer names the project back.
    # A registered NAME echoes the name; the same workspace reached by its
    # registered PATH echoes that same name via the reverse lookup. There is no
    # third value to assert since 5b — the selector-less call carries no echo at
    # all, because it resolves nothing and the anatomy stands in the stamp's
    # place. The collection is read from the derivation, never hand-built as
    # f"mitos-{name}".
    assert by_name["project"] == "bee"
    assert by_path["project"] == "bee"
    assert by_name["workspace"] == by_path["workspace"] == str(ws_b)
    assert by_name["collection"] == default_collection_name(str(ws_b))
    assert "collection" not in refused.content[0].text
    # Echo and discovery speak ONE vocabulary: the name planted in the registry,
    # the name `list_projects` reports, and the name the envelope echoes are the
    # same string.
    assert listed["projects"][0]["name"] == by_name["project"] == "bee"


@pytest.mark.asyncio
async def test_two_registered_projects_each_echo_their_own_name_in_one_session(tmp_path):
    """The echo is per-CALL, not per-process — the sharpest form of the claim.

    One long-lived server, launched from neither workspace, alternating between
    two registered projects across two different tools. A per-process echo (a
    name cached at startup, or a config built once and reused) answers both calls
    with one name and reds here, while every single-project row stays green.

    Asserted across two tools because the stamp is applied per tool: `list_scopes`
    builds its envelope, `show_node` updates a payload from the shared display
    leaf, and a phase that wired only one of them would look complete from either
    row alone.
    """
    env = _scaffold_env(tmp_path)
    ws_a = _workspace(tmp_path, "ws_a", env=env, scopes=("alpha",))
    ws_b = _workspace(tmp_path, "ws_b", env=env, scopes=("beta",))
    launch = tmp_path / "elsewhere"
    launch.mkdir()

    registry_file = Path(env["XDG_CONFIG_HOME"]) / "mitos" / "registry.toml"
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_file.write_text(f'"ay" = "{ws_a}"\n"bee" = "{ws_b}"\n', encoding="utf-8")

    async with mitos_server(cwd=launch, env=env) as server:
        scopes = {}
        shows = {}
        # Interleaved on purpose: A, B, A. A cached first-call config survives an
        # A→B→B ordering and dies on the return to A.
        for name in ("ay", "bee", "ay"):
            scopes[name] = _tool_json(
                await server.session.call_tool("list_scopes", {"project": name}))
            shows[name] = _tool_json(await server.session.call_tool(
                "show_node", {"ident": f"{'alpha' if name == 'ay' else 'beta'}-probe",
                              "project": name}))

    for name, ws, scope in (("ay", ws_a, "alpha"), ("bee", ws_b, "beta")):
        assert scopes[name]["project"] == name
        assert scopes[name]["workspace"] == str(ws)
        assert scopes[name]["collection"] == default_collection_name(str(ws))
        # The dereference reached that project's own graph, and says so.
        assert shows[name]["slug"] == f"{scope}-probe"
        assert shows[name]["project"] == name
        assert shows[name]["workspace"] == str(ws)

    assert set(scopes["ay"]["scopes"]) == {"alpha"}
    assert set(scopes["bee"]["scopes"]) == {"beta"}
    assert scopes["ay"]["collection"] != scopes["bee"]["collection"]


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
            assert _tool_json(await server.session.call_tool(
                "list_scopes", {"project": str(ws)}))["scopes"] == {}
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


def _raw_stdio_exchange(exchange, *, expect_lines, cwd, env, timeout=120):
    """Drives a hand-written request stream at a real `serve`, reading raw stdout.

    The shape is the whole point, and it is **not** `subprocess.run(input=...)`:
    that writes the exchange and closes stdin at once, and the EOF starts
    `mcp.server.stdio`'s shutdown while the response is still in flight. The
    memory object stream the SDK writes responses into has capacity 0, so
    ``send()`` returns as soon as the writer task has been *handed* the message —
    before it has been scheduled to write and flush it. When the task group is
    then cancelled the message dies unwritten, and the row fails reporting a
    missing line, which is a stdout-hygiene message for a shutdown race.

    So: write, drain the expected lines, and only then send EOF. Reader threads
    on both pipes, because stderr must not fill its buffer while stdout is being
    read, and both keep draining past the deadline so the purity assertion still
    sees everything the child ever wrote — including anything emitted during
    teardown, which is the one window the shipped shape could not observe.

    Measured 2026-08-12 against `mcp` 1.27.2: the succeeding `list_scopes` call
    and all three rendered refusals (absent selector, unknown name, relative
    path) lost the tool-result line 0/40 each under this shape, and a trivial
    non-mitos FastMCP server 0/60 — against 6/130 across the same arms under
    `subprocess.run(input=...)`. See the row's own comment for the breakdown.

    Args:
        exchange: The complete request stream, newline-delimited JSON-RPC.
        expect_lines: How many stdout lines to wait for before sending EOF.
        cwd: Launch directory for the child.
        env: The child's complete environment.
        timeout: Seconds to wait for `expect_lines` before giving up and
            closing stdin anyway — a starved row must fail on its own assertion,
            never hang.

    Returns:
        ``(stdout, stderr)`` as text, complete through the child's exit.
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "mitos.cli", "serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(cwd), env=env, text=True,
    )
    out, err = [], []

    def _drain(stream, sink):
        for line in stream:
            sink.append(line)

    readers = [
        threading.Thread(target=_drain, args=(proc.stdout, out), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, err), daemon=True),
    ]
    for reader in readers:
        reader.start()

    try:
        try:
            proc.stdin.write(exchange)
            proc.stdin.flush()
        except BrokenPipeError:
            # The child died before it read the exchange. Swallowed on purpose:
            # the row's own assertion, quoting the captured stderr, says far more
            # than a BrokenPipeError raised from a line that is not the subject.
            pass
        deadline = time.monotonic() + timeout
        while (len(out) < expect_lines
               and time.monotonic() < deadline
               and proc.poll() is None):
            time.sleep(0.01)
    finally:
        try:
            proc.stdin.close()  # the EOF, now that nothing is in flight
        except BrokenPipeError:  # pragma: no cover — the child died first
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
            proc.wait(timeout=10)
        for reader in readers:
            reader.join(timeout=10)

    return "".join(out), "".join(err)


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
    through a plain subprocess, reads the two responses, and only then sends EOF
    — see `_raw_stdio_exchange` for why the ordering is load-bearing.

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
            # A SUCCEEDING call, and the selector is what makes it one: since 5b
            # a selector-less call is refused, and this row's whole point is
            # reaching a tool *handler* with a response on the wire. Do not
            # "simplify" this back to `{}`.
            #
            # It is no longer also a flake defence. The EOF race that made a
            # RAISING tool drop its response — measured 2026-07-31 at 3/40
            # (absent selector), 4/40 (unknown name), 1/40 (relative path), with
            # the success path at 0/40 — was a property of closing stdin before
            # draining, not of the call class, and the success path's immunity
            # was luck rather than structure: re-measured 2026-08-12 on the same
            # `mcp` 1.27.2, that shape dropped the success line 2/30 and 1/40,
            # and a trivial FastMCP server importing no mitos at all dropped it
            # 3/60 under the identical shape. That control is what places the
            # race in `mcp.server.stdio`'s shutdown rather than anywhere in
            # mitos. `_raw_stdio_exchange` removes the class outright: 6/130
            # lost across the old shape's arms, 0/220 across the new one's
            # (success 0/40, each of the three refusals 0/40, control 0/60).
            #
            # The rendered-refusal rows still live on the harness, and the reason
            # is now the ordinary one rather than a flake: the harness is a real
            # client session and this row is a hand-written stream that exists
            # only because raw stdout is unreachable from inside one.
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "list_scopes",
                       "arguments": {"project": str(ws)}},
        },
    ))

    stdout, stderr = _raw_stdio_exchange(
        exchange, expect_lines=2, cwd=ws, env=env,
    )

    lines = [line for line in stdout.splitlines() if line.strip()]
    assert len(lines) == 2, (
        f"expected the initialize result and the tool result, got {len(lines)} "
        f"lines:\n{stdout}\nstderr:\n{stderr}"
    )
    for line in lines:
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"non-JSON line on the protocol channel: {line!r} ({exc})"
            ) from exc
    # Named explicitly, because it is the one line we already know wants out.
    assert "Starting Mitos MCP Server" not in stdout


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
        before = _tool_json(await server.session.call_tool(
            "list_scopes", {"project": str(ws)}))
        assert before["scopes"] == {}

        _record(ws, "alpha", env=env)

        after = _tool_json(await server.session.call_tool(
            "list_scopes", {"project": str(ws)}))
        assert set(after["scopes"]) == {"alpha"}
        assert after["scopes"]["alpha"]["active_decisions"] == 1
        # The provenance is per-call, not cached with the graph view. The value is
        # the REGISTERED NAME, not the path this row passes: `_workspace` runs a
        # real `mitos init`, `harness_env` derives the child's `XDG_CONFIG_HOME`
        # from the same root, so init and server share one registry and the path
        # form reverse-looks-up. Read it off the listing rather than hard-coding
        # "ws_a", so the row states the join instead of a coincidence.
        registered = {entry["path"]: entry["name"] for entry in _tool_json(
            await server.session.call_tool("list_projects", {}))["projects"]}
        assert before["project"] == after["project"] == registered[str(ws)]


# --------------------------------------------------------------------------- #
# Phase 5c — I6/I7: the credential follows the TARGET, not the launch directory
#
# The vision's founding hazard, one layer down from the addressing one phases 3-5b
# closed. Until 5c, `cli.main()` poured the *launch* directory's `.env` and the
# global `.env` into `os.environ`, and a shim read `os.environ` whenever a caller
# supplied nothing — so a server launched inside `cartolina/` because that is where
# its client happened to start it answered every mitos call for the rest of its life
# with `cartolina`'s key. Not a wrong answer: a wrong credential, spent against
# someone else's quota, resolving from a file nobody named.
#
# Both hazards live outside the function call, so every row here is a real process.
# The observation splits deliberately across the two surfaces, because each has
# exactly one keyless, network-free signal and they are different signals:
#
#   * the MCP surface has no tool that reports key state, so the *negative* claim
#     rides `degraded_reason` — "embeddings/Qdrant unavailable" is returned only
#     when no provider was ever constructed, i.e. the target resolved no key from
#     any tier;
#   * the CLI surface reports the winning TIER by name (`mitos status`), which is
#     the *positive* claim and the sharper one — `global .env` in particular is
#     reachable only if the resolution genuinely ran for the named target and
#     nothing was promoted into tier 1.
#
# The direct "os.environ gained nothing" assertion is in-process and lives in
# `tests/test_directory_global.py::test_no_env_file_is_promoted_into_the_process_
# environment`, which drives `main()` with a launch `.env`, a target `.env` and a
# global `.env` all carrying one name. These rows are its process-level half.
# --------------------------------------------------------------------------- #

#: Sentinel key values. Syntactically plausible, deliberately worthless. A row
#: that reaches a provider with one of these is a row that has already failed.
A_SENTINEL = "AIza-sentinel-belongs-to-A"
B_SENTINEL = "AIza-sentinel-belongs-to-B"
GLOBAL_SENTINEL = "AIza-sentinel-belongs-to-the-machine"


def _write_env(path, text):
    """Writes a `.env`, creating parents. Returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _global_env(root):
    """The path a harness child resolves as its global `.env`.

    Read off `harness_env`'s own declared `XDG_CONFIG_HOME` rather than re-deriving
    the `<root>/home/.config` shape, and never from `mitos.config.global_env_path()`,
    which would answer for the *parent* process. `config_home() ==
    XDG_CONFIG_HOME` for the child is pinned by
    `test_the_child_environment_is_declared_not_inherited`; the `mitos/.env` tail
    is `global_env_path`'s shape, and it is self-checking here — the `global .env`
    case below only resolves if the file this writes is the one the child reads.

    It does not exist by default (asserted by that same row), so a test wanting a
    global tier creates it.
    """
    return Path(harness_env(root)["XDG_CONFIG_HOME"]) / "mitos" / ".env"


@pytest.mark.asyncio
async def test_a_session_launched_inside_A_resolves_no_key_for_a_keyless_B(tmp_path):
    """I6(b)/I7 server half: A's key does not become the whole process's key.

    One long-lived session **launched from inside project A**, where `A/.env`
    carries a real-shaped key. A neutral launch directory would pass while the
    hazard shipped, which is why `cwd=ws_a` is the load-bearing half of the setup.
    B is keyless — no `.env`, and no global `.env` exists — so the correct answer
    is that B's call finds no key anywhere.

    The observable is `degraded_reason`. `lexical.degraded_reason_from_error(None)`
    returns "embeddings/Qdrant unavailable" and `None` is reached **only** when
    `embed_provider` was never built, i.e. `config.env` carried no `GEMINI_API_KEY`
    for B. With the entry load alive, A's sentinel sits in this child's `os.environ`,
    tier 1 answers for B, the provider *is* built, and the reason becomes
    "embedding provider error" or a connection phrase instead. So the row reds on
    the hazard and passes on the fix.

    Cost: the green path constructs no provider and makes no network call, which is
    what keeps this module serviceless. Under **fault injection** (the shim or the
    entry load restored) it does attempt one call with a worthless key — that is the
    injection run, not the gate run.

    The provenance assertions are not decoration: they state that the answer came
    from B's corpus while B's credential was the one not found, so a row that
    silently answered for A could not pass by resolving nothing.
    """
    env = _scaffold_env(tmp_path)
    ws_a = _workspace(tmp_path, "ws_a", env=env, scopes=("alpha",))
    ws_b = _workspace(tmp_path, "ws_b", env=env, scopes=("beta",))
    _write_env(ws_a / ".env", f"GEMINI_API_KEY={A_SENTINEL}\n")
    assert not _global_env(tmp_path).exists()

    async with mitos_server(cwd=ws_a, env=env) as server:
        payload = _tool_json(await server.session.call_tool(
            "surface_decisions", {"query": "beta workspace", "project": str(ws_b)}))

    assert payload["degraded"] == "lexical"
    assert payload["degraded_reason"] == "embeddings/Qdrant unavailable"
    assert payload["workspace"] == str(ws_b)


@pytest.mark.asyncio
async def test_one_session_answers_from_the_graph_file_on_disk_at_each_call(tmp_path):
    """I6(a): swapping the target's graph between two calls changes the answer.

    3a's `…spans_calls_and_a_filesystem_mutation` proves the *corpus* is re-read;
    this is the sharper claim the index asks for — a simulated rebuild. Nothing is
    cached across calls, **including a SQLite handle**: a server that opened the
    graph once at startup would keep answering from the swapped-away file, and a
    `mitos rebuild` under a live session would be invisible to every later call.

    Added beside 3a's row rather than folded into it: the two claims fail
    differently and a merged row would report the wrong one.
    """
    env = _scaffold_env(tmp_path)
    ws = _workspace(tmp_path, "ws_a", env=env, scopes=("alpha",))
    donor = _workspace(tmp_path, "ws_donor", env=env, scopes=("gamma",))

    async with mitos_server(cwd=ws, env=env) as server:
        before = _tool_json(await server.session.call_tool(
            "list_scopes", {"project": str(ws)}))
        assert set(before["scopes"]) == {"alpha"}

        # The rebuild, simulated: a different graph file lands at the same path.
        (ws / ".mitos" / "graph.sqlite").unlink()
        (ws / ".mitos" / "graph.sqlite").write_bytes(
            (donor / ".mitos" / "graph.sqlite").read_bytes())

        after = _tool_json(await server.session.call_tool(
            "list_scopes", {"project": str(ws)}))

    assert set(after["scopes"]) == {"gamma"}
    assert before["project"] == after["project"]


def _status_key_line(root, target, *, cwd, env):
    """Runs `mitos -p <target> status` as a subprocess and returns its key row.

    Not `_run_mitos`: `status` exits non-zero on these workspaces (Qdrant is a dead
    port by design), and the exit code is not what the caller is asking about. The
    attribution prints on every branch.
    """
    done = subprocess.run(
        [sys.executable, "-m", "mitos.cli", "-p", str(target), "status"],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=120,
    )
    lines = [ln for ln in done.stdout.splitlines() if "GEMINI_API_KEY" in ln]
    assert len(lines) == 1, f"expected one key line, got {lines}\n{done.stdout}"
    return lines[0]


def test_the_cli_invoked_from_inside_A_reports_Bs_own_tier(tmp_path):
    """I7 CLI half: the winning tier is named for the TARGET, from inside another project.

    The positive claim, and the one the MCP surface cannot make keylessly. `mitos
    status` names the tier that actually won (`env.resolve_key`, one call, no
    provider), so a sentinel plus a tier string is a complete statement about which
    file answered.

    Two cases, and the second is the discriminator:

    * **B has its own key** → `project .env`. A's key is present in the launch
      directory and must lose.
    * **B has none, the machine has one** → `global .env`. This is the case no
      wrong implementation can reach: the entry load would have promoted A's key
      into tier 1 and reported `environment`, and a resolution that ran for A
      rather than B would have reported `project .env`. It is also D2's argument
      asserted rather than reasoned — deleting the *global* load with the project
      one had to leave the global **tier** intact, which is exactly the setup
      SETUP.md recommends (an empty project slot plus one global key).

    Bound to `env`'s tier constants, imported here for the same reason
    `test_env_routing.py` binds to them: a rename lands as a failing import, not as
    a green row asserting a dead string.
    """
    from mitos.env import TIER_ENVIRONMENT, TIER_GLOBAL_ENV, TIER_PROJECT_ENV

    env = _scaffold_env(tmp_path)
    ws_a = _workspace(tmp_path, "ws_a", env=env)
    ws_b = _workspace(tmp_path, "ws_b", env=env)
    _write_env(ws_a / ".env", f"GEMINI_API_KEY={A_SENTINEL}\n")
    _write_env(_global_env(tmp_path), f"GEMINI_API_KEY={GLOBAL_SENTINEL}\n")

    _write_env(ws_b / ".env", f"GEMINI_API_KEY={B_SENTINEL}\n")
    line = _status_key_line(tmp_path, ws_b, cwd=ws_a, env=env)
    assert f"(from {TIER_PROJECT_ENV})" in line
    assert TIER_ENVIRONMENT not in line
    # P8: the attribution names the tier, never the value.
    assert B_SENTINEL not in line and A_SENTINEL not in line

    (ws_b / ".env").unlink()
    line = _status_key_line(tmp_path, ws_b, cwd=ws_a, env=env)
    assert f"(from {TIER_GLOBAL_ENV})" in line
    assert TIER_PROJECT_ENV not in line
