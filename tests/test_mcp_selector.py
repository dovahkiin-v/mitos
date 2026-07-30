"""Tests for the MCP project selector (`surface_decisions(project='mitos', …)`).

Phase 3c gives ``routing.resolve_project`` its second consumer — the surface the
whole vision exists for. This module covers the parameter (present, `Optional`,
never schema-required, documented as required), the boundary (one resolution per
tool, before the degradation `try`, after an argument fault), the six rendered
wordings and what each may not say, the bound behaviour, and the `list_projects`
twin's parity with CLI `projects --json`.

**No mocks of external services, no live tier.** Every row drives a real registry
file — conftest's autouse ``hermetic_mitos_env`` redirects ``XDG_CONFIG_HOME`` per
test, which is what keeps these writes out of the developer's own
``~/.config/mitos/registry.toml`` — and real, really-initialized workspaces under
``tmp_path``. Keys are stripped and Qdrant is pointed at a dead port, so nothing
here embeds, queries or spends.

Two things are asserted **exactly** rather than by substring, both for the reason
3b measured: every body carries a vocabulary line naming registered projects, so
a bare ``assert name in body`` is green whatever the line under test actually
says. The did-you-mean list and the registered-projects line are therefore parsed
out and compared as values.

The delivery mechanism itself (``isError``, and the anatomy arriving in
``content[0].text``) is proven where it can only be proven — over a real
``mitos serve`` subprocess, in ``tests/test_mcp_stdio_harness.py``. Everything
here is in-process, because the selector's behaviour lives in the function.
"""

import asyncio
import json
import os
import sys
from unittest.mock import patch

import pytest

from mitos import cli, mcp_server, registry, routing
from mitos.cli import cmd_init, cmd_projects
from mitos.config import MitosConfig
from mitos.display import projects_payload, resolve_display_ensure_ascii
from mitos.errors import (
    EXEMPT_EXPLICITLY_GLOBAL,
    TARGETING_DISCRIMINATORS,
    TARGET_EXEMPT_VERB,
    TARGET_MISSING,
    TARGET_PATH_NOT_A_WORKSPACE,
    TARGET_REGISTERED_UNREACHABLE,
    TARGET_RELATIVE_PATH,
    TARGET_UNKNOWN_NAME,
    ProjectTargetingError,
)
from mitos.sync import MitosSyncManager


#: The tools that take a `project`. `list_projects` is deliberately absent — it
#: answers for the machine, so there is nothing for a selector to select.
TARGETING_TOOLS = (
    "surface_decisions",
    "list_decisions",
    "list_scopes",
    "show_node",
    "query_decisions",
    "record_decision",
)

#: Targeting vocabulary that belongs to the *other* surface. `mitos init` is the
#: sharp one: registration is a human setup act, and an agent handed a
#: state-creating shell command is invited to run it.
FORBIDDEN_SYNTAX = ("--project", "-p ", "mitos init", "mitos projects")


# --- fixtures & helpers ----------------------------------------------------

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Keyless, serviceless: nothing in this module embeds, queries or spends."""
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:9")
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def _write_registry(text: str) -> str:
    """Hand-writes the registry file (the hand-editable states we must tolerate)."""
    path = registry.registry_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _register_pairs(pairs) -> str:
    """Writes a registry from ``(name, path)`` pairs, preserving document order."""
    return _write_registry("".join(f'"{name}" = "{path}"\n' for name, path in pairs))


def _register(**entries) -> str:
    """Writes a registry from ``name → path`` keywords, in the order given."""
    return _register_pairs(entries.items())


def _workspace(root, *, scope=None) -> str:
    """Builds a REALLY-initialized workspace and returns its canonical root.

    A real ``mitos init`` — the graph, the config, the buffers — because these
    rows read through the actual store, not through a validity-triple stub. When
    ``scope`` is given, one decision is committed into it, which is what makes two
    workspaces distinguishable by a store read rather than only by provenance.

    ``init`` registers the workspace as a side effect; every row that cares about
    the registry's contents overwrites it afterwards, so that is left alone here.
    """
    os.makedirs(str(root), exist_ok=True)
    config = MitosConfig(str(root))
    cmd_init(config)
    if scope is not None:
        MitosSyncManager(config).record_decision_entry(
            axiom=f"The {scope} workspace answers for itself.",
            rejected_paths="A workspace that answers with another workspace's graph.",
            scope=[scope],
            slug=f"{scope}-probe",
        )
    return os.path.realpath(str(root))


def _raises(selector) -> ProjectTargetingError:
    """Returns the targeting error a selector raises out of the resolver."""
    with pytest.raises(ProjectTargetingError) as excinfo:
        routing.resolve_project(selector)
    return excinfo.value


def _body(tool: str, **kwargs) -> str:
    """Calls a tool expecting a targeting failure, returning the rendered body."""
    with pytest.raises(mcp_server._RenderedToolError) as excinfo:
        getattr(mcp_server, tool)(**kwargs)
    return str(excinfo.value)


def _vocabulary(body: str):
    """Parses the registered-projects line, or None if the body carries none.

    Returned as the line's **value** rather than checked by substring, because
    every body mentions registered projects somewhere: an assertion that scans the
    whole body cannot tell the vocabulary line from the sentence above it, and
    goes green against a bound regression.
    """
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("Registered projects:"):
            return stripped.split(":", 1)[1].strip()
    return None


def _did_you_mean(body: str):
    """Parses the did-you-mean suggestions as a list, or None if absent."""
    for line in body.splitlines():
        if "Did you mean" in line:
            return line.split(":", 1)[1].strip().split(", ")
    return None


def _tools():
    """The live tool table, off ``mcp.list_tools()`` rather than out of an import."""
    return {tool.name: tool for tool in asyncio.run(mcp_server.mcp.list_tools())}


def _project_arg_doc(description: str) -> str:
    """Extracts the ``project:`` Args entry out of a tool description.

    FastMCP puts the **whole** docstring into ``tool.description``, so the
    forbidden-syntax tripwire cannot be applied to it wholesale: `show_node` and
    `record_decision` carry deliberate, shipped `mitos sync` / `mitos check`
    pointers, which are workflow answers rather than targeting recoveries. The
    rule is about the targeting vocabulary, so it is applied to the entry that
    carries it.
    """
    lines = description.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("project:"):
            indent = len(line) - len(line.lstrip())
            entry = [stripped]
            for follower in lines[index + 1:]:
                if not follower.strip():
                    break
                if len(follower) - len(follower.lstrip()) <= indent:
                    break
                entry.append(follower.strip())
            return " ".join(entry)
    raise AssertionError("no `project:` entry in the description")


def _one_of_every_discriminator(tmp_path):
    """One error per discriminator, built from real resolver failures.

    Keyed on ``TARGETING_DISCRIMINATORS`` rather than a hand-written list, so a
    seventh class added to ``errors.py`` reds the sweep below at the coverage
    assertion — which is where an implementer is told to write a branch and a
    row, instead of silently inheriting a neighbour's wording.
    """
    stale = os.path.realpath(str(tmp_path / "stale"))
    os.makedirs(stale, exist_ok=True)
    bare = tmp_path / "bare"
    bare.mkdir(exist_ok=True)
    _register(stale=stale, other=stale)
    built = {
        TARGET_MISSING: _raises(""),
        TARGET_UNKNOWN_NAME: _raises("stale-x"),
        TARGET_RELATIVE_PATH: _raises("./x"),
        TARGET_PATH_NOT_A_WORKSPACE: _raises(str(bare)),
        TARGET_REGISTERED_UNREACHABLE: _raises("stale"),
        TARGET_EXEMPT_VERB: routing.exempt_verb_error(
            "list_projects", EXEMPT_EXPLICITLY_GLOBAL),
    }
    assert set(built) == TARGETING_DISCRIMINATORS, (
        "a targeting discriminator has no fixture here — give it a renderer "
        "branch and a row rather than letting it fall into a neighbour's"
    )
    for discriminator, err in built.items():
        assert err.discriminator == discriminator
    return built


# ---------------------------------------------------------------------------
# Group 1 — the parameter on the wire. Asserted against `mcp.list_tools()`,
# because the schema an agent meets is FastMCP's rendering, not the signature.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool", TARGETING_TOOLS)
def test_project_is_optional_in_the_schema_and_never_required(tool) -> None:
    """Criterion 2: the schema must not reject a missing `project`.

    `required=True` would make FastMCP refuse the call before any mitos code
    runs, so the teaching anatomy could never render and the caller would get a
    bare framework rejection carrying no registered vocabulary — the one failure
    that cannot recover in a single turn. The property is *absence from* the
    `required` array, not the array's absence: four of the seven tools carry one
    already (`query`, `record`, `show_node`, `surface`), so a row spelled
    `"required" not in schema` would be red out of the box and would then get
    "fixed" into something weaker.
    """
    schema = _tools()[tool].inputSchema
    assert schema["properties"]["project"] == {
        "anyOf": [{"type": "string"}, {"type": "null"}],
        "default": None,
        "title": "Project",
    }
    assert "project" not in schema.get("required", [])


def test_the_targeting_tool_set_is_the_live_one_not_a_hand_copied_list() -> None:
    """The fence under the four parametrized rows above, and under §4.10's net.

    Those rows are parametrized over a tuple written by hand, so on their own they
    fence nothing: a seventh tool taking a `project` would simply not be
    parametrized, and its schema, its documented requirement and its
    forbidden-syntax sweep would all be unasserted while every row stayed green.
    This row makes the tuple a claim about the live tool table rather than a list
    someone remembered to update — computed off `mcp.list_tools()`, the same shape
    3b used over `subparsers.choices` for the CLI alias set.

    Membership is by the parameter, not by a name list: a tool exposes `project`
    or it does not, and the one that does not (`list_projects`) has its own row.
    """
    exposing = {
        name for name, tool in _tools().items()
        if "project" in tool.inputSchema.get("properties", {})
    }

    assert exposing == set(TARGETING_TOOLS), (
        "a tool's `project` parameter is not covered by this module's rows — add "
        "it to TARGETING_TOOLS (which extends the schema, description, "
        "forbidden-syntax and delivery rows onto it) rather than leaving it "
        "asserted by nothing"
    )


def test_list_projects_exposes_no_project_parameter() -> None:
    """Criterion 13: the discovery tool answers for the machine, not a workspace.

    There is nothing for a selector to select, so there is no parameter — and no
    exempt-verb machinery either. `routing.exempt_verb_error` has exactly one
    caller in the tree and it is the CLI's.
    """
    assert "project" not in _tools()["list_projects"].inputSchema.get("properties", {})


@pytest.mark.parametrize("tool", TARGETING_TOOLS)
def test_every_project_argument_is_documented_as_required(tool) -> None:
    """Criterion 14 / the §4.10 net: the description is the whole mechanism.

    The parameter is `Optional` at the schema level by design, so nothing but the
    tool description carries "name this on every call". A hand-written sentence in
    six places is exactly the shape that drifts, and this is the assertion that
    risk actually needs — it also fences the next tool a later phase adds.
    """
    entry = _project_arg_doc(_tools()[tool].description).lower()
    assert "required" in entry
    assert "registered" in entry
    assert "absolute path" in entry
    assert "list_projects()" in entry


@pytest.mark.parametrize("tool", TARGETING_TOOLS)
def test_no_cli_targeting_syntax_reaches_a_project_argument_entry(tool) -> None:
    """Criterion 8, half one: the descriptions.

    Scoped to the `project:` entry rather than the whole description on purpose.
    A blanket "no CLI command form in MCP text" rule would red three pre-existing,
    deliberate, shipped strings — `show_node`'s `mitos sync` (via the shared
    `display.SHOW_NOT_FOUND_HINT` leaf) and `record_decision`'s `mitos check` /
    `mitos status` — which are workflow pointers, where naming a CLI command is
    the correct answer. The rule is about *targeting* recoveries.
    """
    entry = _project_arg_doc(_tools()[tool].description)
    for syntax in FORBIDDEN_SYNTAX:
        assert syntax not in entry, f"{tool} leaked {syntax!r}"


# ---------------------------------------------------------------------------
# Group 2 — resolution: the selector actually retargets the call.
# ---------------------------------------------------------------------------

def test_a_registered_name_answers_about_that_workspace_from_another_cwd(
    tmp_path, monkeypatch
) -> None:
    """Criterion 3: the store read AND the provenance stamp both name the target.

    Driven from a cwd that is a *different, valid* workspace, so an implementation
    that still reads the working directory anywhere in the call reds — including
    one that resolves correctly and then rebuilds a cwd-derived config for its
    provenance envelope, which would return the target's decisions labelled with
    the launch directory's collection.

    The expected collection is read from `MitosConfig(root).qdrant_collection`,
    never built as `f"mitos-{name}"`: it is path-hashed, and a hand-built literal
    would pin the derivation rather than the routing.
    """
    target = _workspace(tmp_path / "target", scope="alpha")
    decoy = _workspace(tmp_path / "decoy", scope="beta")
    _register(target=target, decoy=decoy)
    monkeypatch.chdir(decoy)

    payload = json.loads(mcp_server.list_decisions(project="target"))

    assert [d["slug"] for d in payload["decisions"]] == ["alpha-probe"]
    assert payload["project"] == "target"  # the name the CALL used, echoed back
    assert payload["workspace"] == target
    assert payload["collection"] == MitosConfig(target).qdrant_collection
    assert payload["collection"] != MitosConfig(decoy).qdrant_collection


def test_an_absolute_path_reaches_a_valid_but_unregistered_workspace(
    tmp_path, monkeypatch
) -> None:
    """Criterion 4: the escape hatch — a fresh clone, a mid-setup project.

    An unregistered workspace is a correct steady state, not a degraded one: it
    resolves, answers, and draws no warning of any kind.
    """
    target = _workspace(tmp_path / "target", scope="alpha")
    decoy = _workspace(tmp_path / "decoy", scope="beta")
    _register(decoy=decoy)
    monkeypatch.chdir(decoy)

    payload = json.loads(mcp_server.list_scopes(project=target))

    assert set(payload["scopes"]) == {"alpha"}
    # An unregistered path echoes the path itself — the §4.7 escape-hatch rule,
    # cross-checked here on the tool that exercises it.
    assert payload["project"] == target


def test_the_write_tool_records_into_the_project_the_call_named(
    tmp_path, monkeypatch
) -> None:
    """The write is the highest-stakes retarget: a mis-aim lands a real entry.

    `record_decision` builds its own writable manager rather than reusing the
    read tools' read-only store, so it is the one tool whose config threading is
    a separate seam — and the `FileLock` on the target's own `decisions.md.lock`
    is what keeps concurrent same-project writes serialized while calls to
    different projects never contend.
    """
    target = _workspace(tmp_path / "target")
    decoy = _workspace(tmp_path / "decoy")
    _register(target=target, decoy=decoy)
    monkeypatch.chdir(decoy)

    result = json.loads(mcp_server.record_decision(
        axiom="A recorded decision lands in the project the call named.",
        rejected_paths="Landing it wherever the server happened to start.",
        scope=["routing"], slug="written-into-the-target",
        project="target",
    ))

    assert result.get("error") is None
    assert result["slug"] == "written-into-the-target"
    assert "written-into-the-target" in open(
        os.path.join(target, "decisions.md"), encoding="utf-8").read()
    assert "written-into-the-target" not in open(
        os.path.join(decoy, "decisions.md"), encoding="utf-8").read()


@pytest.mark.parametrize("tool", TARGETING_TOOLS)
def test_a_targeting_failure_reaches_the_caller_from_every_tool(
    tmp_path, monkeypatch, tool
) -> None:
    """The boundary raises on every tool, not only on the ones with no `try`.

    `surface_decisions` and `query_decisions` wrap `get_workspace_components()` in
    a bare `except Exception` that returns the degraded lexical envelope. If
    resolution happened inside that `try`, a targeting failure would come back as
    "semantic recall is degraded" with a nonsense reason and `isError: False` —
    the anatomy silently destroyed, on the two highest-traffic tools, with every
    existing row still green. This is the phase's single most likely wrong build,
    which is why the row is parametrized over all six rather than written for the
    two: the two are the hazard, the six are the contract.
    """
    _register(known=_workspace(tmp_path / "known"))
    monkeypatch.chdir(tmp_path)
    required = {
        "surface_decisions": {"query": "anything"},
        "list_decisions": {},
        "list_scopes": {},
        "show_node": {"ident": "anything"},
        "query_decisions": {"query": "anything"},
        "record_decision": {"axiom": "a", "rejected_paths": "b",
                            "scope": ["c"], "slug": "d"},
    }[tool]

    body = _body(tool, project="nosuchproject", **required)

    assert body.startswith("no project named 'nosuchproject' is registered")
    assert f"{tool}(project=" in body, "the example must name the failing tool"
    assert "degraded" not in body


@pytest.mark.parametrize("tool,kwargs,fault", [
    ("list_decisions", {"brief": True, "oneline": True},
     "brief and oneline are mutually exclusive"),
    ("query_decisions", {"query": "q", "depth": "trace"},
     "is not yet implemented"),
])
def test_an_argument_fault_is_answered_before_the_project_is_resolved(
    tool, kwargs, fault
) -> None:
    """A fault in the arguments is answered by naming the argument.

    3b's ordering rule generalizes: the exempt check runs before resolution so a
    selector on `init` is answered by naming the verb, and here a caller who asked
    for two mutually-exclusive depth tiers is told *that*, not told about a
    project they also got wrong. Driven with an unresolvable project precisely so
    a resolve-first implementation reds.
    """
    payload = json.loads(getattr(mcp_server, tool)(project="nosuchproject", **kwargs))

    assert fault in payload["error"]


# ---------------------------------------------------------------------------
# Group 3 — the six wordings. What each must carry, and what none may say.
# ---------------------------------------------------------------------------

def test_every_discriminator_renders_a_distinct_non_empty_body(tmp_path) -> None:
    """Criterion 7: no class falls into a neighbour's words.

    A wrong class teaches the wrong lesson — answering a relative path with
    "unknown project, did you mean 'mitos'?" invites a retry with another relative
    spelling. The renderer's last branch is a documented `else` (the CLI's shape,
    and `errors._fallback_message`'s), which is safe only while every other class
    is handled above it; a seventh class would land there silently. The fixture's
    coverage assertion is what catches that, and this row is what proves the six
    are actually distinguishable.
    """
    bodies = {
        discriminator: mcp_server._render_targeting_error(err, "surface_decisions")
        for discriminator, err in _one_of_every_discriminator(tmp_path).items()
    }

    assert all(body.strip() for body in bodies.values())
    assert len(set(bodies.values())) == len(TARGETING_DISCRIMINATORS)


def test_every_reachable_class_carries_the_whole_anatomy(tmp_path) -> None:
    """Criterion 7's parts: statement, example, pointer, vocabulary line.

    `exempt_verb` is excluded deliberately rather than by omission: it is
    unreachable from this surface (no tool is exempt, and `list_projects` takes no
    selector to refuse), and nothing was ever resolved, so a registered-name
    vocabulary and a `project=` example would both be answers to a question the
    caller did not ask. It gets a terse honest branch — pinned by its own row
    below — instead of a padded one.
    """
    errors = _one_of_every_discriminator(tmp_path)

    for discriminator, err in errors.items():
        if discriminator == TARGET_EXEMPT_VERB:
            continue
        body = mcp_server._render_targeting_error(err, "list_decisions")
        assert body.splitlines()[0].strip(), discriminator
        assert "list_decisions(project=" in body, discriminator
        assert "`list_projects()`" in body, discriminator
        assert _vocabulary(body) == "stale, other.", discriminator


def test_the_exempt_class_stays_terse_and_claims_no_vocabulary(tmp_path) -> None:
    """The class that is unreachable here says so briefly and stops.

    It exists as a branch only so it cannot be rendered in
    `registered_unreachable`'s words. If a future reader finds themselves reaching
    for `routing.exempt_verb_error` in `mcp_server`, the design has drifted.
    """
    err = _one_of_every_discriminator(tmp_path)[TARGET_EXEMPT_VERB]

    body = mcp_server._render_targeting_error(err, "surface_decisions")

    assert body.splitlines() == [
        "the `list_projects` tool takes no `project` argument — it answers for "
        "the machine, not for one workspace."
    ]
    assert _vocabulary(body) is None


def test_no_rendered_body_carries_the_other_surfaces_vocabulary(tmp_path) -> None:
    """Criterion 8, half two: the bodies.

    `mitos init` is the sharp one. In a directory that is not a workspace an agent
    has no mitos action to take, and an error handing it a state-creating shell
    command invites the wrong one — an autonomous `init` scaffolding `.mitos/` and
    claiming a global registration nobody asked to introduce. The server never
    shells out, so the capability boundary holds regardless; this is a wording
    rule, and the wording is its whole enforcement.
    """
    for discriminator, err in _one_of_every_discriminator(tmp_path).items():
        for tool in TARGETING_TOOLS:
            body = mcp_server._render_targeting_error(err, tool)
            for syntax in FORBIDDEN_SYNTAX:
                assert syntax not in body, f"{discriminator} leaked {syntax!r}"


def test_the_rendered_body_is_never_the_terse_shared_fallback(tmp_path) -> None:
    """D1: the anatomy cannot live on the typed error, so it rides a raise.

    Letting `ProjectTargetingError` propagate out of a tool is the trap, and it
    looks correct — `isError` is True, the message is calm, no traceback. But
    `str()` on that error is the discriminator-level fallback, fenced by a
    never-inverted tripwire forbidding exactly the strings a rendered body must
    carry. So the boundary catches, renders, and re-raises something whose `str()`
    IS the finished body.
    """
    for err in _one_of_every_discriminator(tmp_path).values():
        body = mcp_server._render_targeting_error(err, "surface_decisions")
        assert body != str(err)
        assert len(body) > len(str(err))


def test_the_renderer_reads_nothing_and_therefore_cannot_fail(tmp_path) -> None:
    """D4: no cwd line, no second registry read — a pure function of the error.

    The CLI's renderer needs a `try/except (RegistryError, OSError)` guard because
    it re-reads both *inside* an `except` arm, where a raise escapes `main()`
    entirely. This one is not guarded, and must not be: porting the guard would
    imply a read that is not there. Proven by making both reads fatal — if either
    is ever added, every class reds at once rather than one unlucky one later.

    The absent cwd line is a choice, not an omission. On an always-on server
    `os.getcwd()` is fixed for the process's whole life, so a hint derived from it
    is constant across every call; and it would be true for one phase and
    misleading in the next, since once the fallback is gone the same line names a
    project the server will not use.
    """
    errors = _one_of_every_discriminator(tmp_path)

    def _explode(*args, **kwargs):
        raise AssertionError("the MCP renderer read something")

    with patch.object(os, "getcwd", _explode), \
            patch.object(registry, "load", _explode), \
            patch.object(routing.registry, "load", _explode):
        for err in errors.values():
            mcp_server._render_targeting_error(err, "surface_decisions")


def test_the_relative_path_class_is_worded_for_a_surface_with_no_shell(
    tmp_path
) -> None:
    """G9: the realistic MCP mistake, and the CLI's answer is wrong for it.

    Agents reach for relative paths by habit, out of a world where a working
    directory means something. This surface canonicalizes nothing, so the habit
    lands here — and the CLI's line about the shell not expanding an unquoted `~`
    is simply false here, because there is no shell.
    """
    _register(known=_workspace(tmp_path / "known"))

    relative = mcp_server._render_targeting_error(_raises("./mitos-pub"), "list_scopes")
    tilde = mcp_server._render_targeting_error(_raises("~/mitos-pub"), "list_scopes")

    assert "relative path" in relative
    assert "shell" not in relative and "shell" not in tilde
    assert "`~` is not expanded here" in tilde
    assert "~" not in relative.replace("'./mitos-pub'", "")


def test_registered_unreachable_names_a_human_and_a_path_never_the_name_again(
    tmp_path
) -> None:
    """G8: the class whose CLI recovery is closed to this surface.

    The repoint *is* `mitos init --force`, which no MCP body may name — so left
    unspecified this class would hand an agent a failure with no action it is
    permitted to take, and an agent with no named recovery retries or improvises.
    Naming a human as the next actor is a legitimate agent action.

    The example must NOT name a registered project here, and that is the sharp
    edge: the registered names include the one that just failed, so the general
    "use `registered_names[0]`" rule would render "retry the thing that did not
    work" — the dead end this class exists to remove.
    """
    stale = os.path.realpath(str(tmp_path / "stale"))
    os.makedirs(stale)
    _register(gone=stale, alive=stale)

    body = mcp_server._render_targeting_error(_raises("gone"), "show_node")

    assert "needs a human" in body
    assert stale in body, "the recorded path is the value a repoint edits"
    assert "show_node(project='/absolute/path/to/the/workspace', …)" in body
    assert "show_node(project='gone'" not in body
    assert "show_node(project='alive'" not in body


def test_path_not_a_workspace_describes_the_triple_without_prescribing_it(
    tmp_path
) -> None:
    """G10: naming what was looked for is fair; naming the command that makes it is not.

    Its recovery is a registered project, or the absolute path of a workspace that
    already exists.
    """
    _register(known=_workspace(tmp_path / "known"))
    bare = tmp_path / "bare"
    bare.mkdir()

    body = mcp_server._render_targeting_error(_raises(str(bare)), "surface_decisions")

    assert ".mitos/config.toml" in body and "decisions.md" in body
    assert "mitos init" not in body
    assert "surface_decisions(project='known', …)" in body


# ---------------------------------------------------------------------------
# Group 4 — the enumeration bound. The one part of the anatomy that grows with
# the machine, on a surface where it is input tokens on every failure.
# ---------------------------------------------------------------------------

def test_above_the_bound_with_no_close_match_the_names_collapse_to_a_count(
    tmp_path
) -> None:
    """Criterion 10a: the honest answer is a count plus the discovery pointer.

    Asserted on the **parsed vocabulary line**, not by scanning the body for
    names: the example call legitimately carries one name (the bound governs
    enumeration, not the existence of one example), so a whole-body assertion
    would be red for the wrong reason — and, worse, would tempt someone to drop
    the example to make it pass.
    """
    root = os.path.realpath(str(tmp_path))
    names = [f"proj-{suffix}" for suffix in
             ("aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj", "ak")]
    assert len(names) == routing.REGISTERED_NAMES_BOUND + 1
    _register_pairs((name, root) for name in names)

    body = mcp_server._render_targeting_error(_raises("zzzzzz"), "list_scopes")

    assert _vocabulary(body) == "11 — too many to enumerate here."
    assert _did_you_mean(body) is None
    assert "`list_projects()`" in body
    assert "list_scopes(project='proj-aa', …)" in body


def test_above_the_bound_every_case_variant_is_named_past_the_didyoumean_max(
    tmp_path
) -> None:
    """Criterion 10b / G4: `close_project_matches` can return MORE than the max.

    It expands each folded match to every original that folds onto it, so a
    registry holding four case variants of one name legitimately returns four
    with a max of three. Truncating to the max would hide the very distinction
    the caller needs to see — which of `mitos` and `MITOS` they meant.
    """
    root = os.path.realpath(str(tmp_path))
    variants = ["mitos", "Mitos", "MITOS", "MiToS"]
    _register_pairs([(name, root) for name in variants]
                    + [(f"other-{i}", root) for i in range(7)])

    body = mcp_server._render_targeting_error(_raises("mitoss"), "query_decisions")

    assert _did_you_mean(body) == variants
    assert len(variants) > routing.PROJECT_DIDYOUMEAN_MAX
    assert _vocabulary(body) == "11, closest to what you passed: mitos, Mitos, MITOS, MiToS."


def test_an_empty_registry_names_the_escape_hatch_and_never_the_setup_act(
    tmp_path, monkeypatch
) -> None:
    """Criterion 10c: healthy-and-empty, extended to the teaching-error surface.

    A fresh machine is not a broken one. The CLI answers this state by prescribing
    `mitos init`, because registration is a human setup act and the CLI is where a
    human is standing. This surface answers it with the only recovery an agent is
    permitted to take: the absolute path of a workspace that already exists.
    """
    _write_registry("")
    monkeypatch.chdir(tmp_path)

    body = _body("surface_decisions", query="anything", project="")

    assert _vocabulary(body) == (
        "none on this machine yet — a workspace is still reachable by its "
        "absolute path.")
    assert "mitos init" not in body
    assert "surface_decisions(project='/absolute/path/to/the/workspace', …)" in body

    # A *name* selector still reaches `unknown_name` here — an agent guessing a
    # plausible project name is the ordinary way onto a fresh machine — and its
    # recovery must not lead with "retry with a registered name" when there are
    # none. Caught by rendering it through the real binary, not by a test.
    unknown = _body("list_scopes", project="mitos")
    assert "a registered name" not in unknown
    assert "Retry with the absolute path of the workspace you mean" in unknown
    assert "list_scopes(project='/absolute/path/to/the/workspace', …)" in unknown


# ---------------------------------------------------------------------------
# Group 5 — `list_projects`, and its parity with the CLI twin.
# ---------------------------------------------------------------------------

def test_list_projects_is_byte_identical_to_the_cli_json(tmp_path, capsys) -> None:
    """Criterion 11: one shared leaf, so parity is structural rather than asserted.

    Byte-for-byte over the same registry — including **document order**, which is
    the order a reverse lookup resolves its first match in and therefore the order
    that actually decides, and a **non-ASCII** name and path, because Lithuanian
    is load-bearing in this project and the encoding seam takes `ensure_ascii` as
    a parameter that the two surfaces set differently.

    Asserted on the emitted **text** as well as on the parsed payloads, because
    parsed equality is blind to exactly the two things the shared leaf is supposed
    to fix: key order and indent. `json.loads` would call a compact, differently
    ordered CLI payload identical to the MCP one, so a row that only compares
    parsed dicts proves the *values* match and calls it byte parity.

    The text comparison is legitimate here rather than universal, and the
    difference is worth naming: MCP sets `ensure_ascii=False` unconditionally (the
    transport has no terminal to sniff) while the CLI resolves it against the live
    stdout, which under capture is UTF-8 and therefore also non-escaping. On a
    non-UTF-8 terminal the CLI would emit `\\uXXXX` escapes by design — still the
    same payload, still parse-equal, which is why both assertions are here rather
    than only the stricter one.
    """
    root = os.path.realpath(str(tmp_path))
    lithuanian = os.path.join(root, "ąžuolas")
    os.makedirs(lithuanian, exist_ok=True)
    _register_pairs([("zebra", root), ("ąžuolas", lithuanian), ("alpha", root)])

    mcp_text = mcp_server.list_projects()
    cmd_projects(as_json=True)
    cli_text = capsys.readouterr().out
    from_mcp = json.loads(mcp_text)
    from_cli = json.loads(cli_text)

    assert from_mcp == from_cli
    # `print` adds the newline; everything before it must match exactly.
    assert not resolve_display_ensure_ascii(sys.stdout), (
        "the byte assertion below holds only on a non-escaping stdout")
    assert mcp_text == cli_text.rstrip("\n")
    assert "ąžuolas" in mcp_text, "no surface escaped the non-ASCII name here"
    assert [p["name"] for p in from_mcp["projects"]] == ["zebra", "ąžuolas", "alpha"]
    assert from_mcp["projects"][1]["path"] == lithuanian
    assert from_mcp["count"] == 3
    assert from_mcp["registry_path"] == registry.registry_path()


def test_list_projects_over_an_empty_registry_is_a_clean_empty_envelope() -> None:
    """Criterion 12: no teaching error, no diagnostic the caller did not ask for.

    The wall-vs-honest-empty line is drawn by vocabulary membership, not by result
    count: nothing is registered, and saying so plainly *is* the answer. What a
    caller should do about it belongs in this tool's description, not in a data
    shape the CLI shares.
    """
    path = _write_registry("")

    assert json.loads(mcp_server.list_projects()) == {
        "registry_path": path, "count": 0, "projects": [],
    }


def test_the_payload_survives_a_non_ascii_registry_path(tmp_path, monkeypatch) -> None:
    """The Lithuanian case bit 1d twice — once on names, once on paths.

    Here the *registry file's own path* is non-ASCII, which is the spelling
    neither of those covered.
    """
    home = tmp_path / "namų" / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    path = _write_registry(f'"alpha" = "{os.path.realpath(str(tmp_path))}"\n')

    payload = json.loads(mcp_server.list_projects())

    assert payload["registry_path"] == path
    assert "namų" in payload["registry_path"]
    assert payload["count"] == 1


def test_mitos_projects_is_unchanged_on_both_of_its_branches(
    tmp_path, capsys
) -> None:
    """Criterion 16: lifting the shape to the leaf moved nothing the CLI emits.

    Both branches read the one payload — routing only `--json` through the leaf
    would fork the text table off a second, hand-built list, which is the drift
    the lift exists to remove.
    """
    root = os.path.realpath(str(tmp_path))
    path = _register_pairs([("zebra", root), ("alpha", root)])

    cmd_projects()
    table = capsys.readouterr().out
    assert "Projects (2 registered, in registry order):" in table
    assert f"Registry: {path}" in table
    assert table.index("zebra") < table.index("alpha")

    _write_registry("")
    cmd_projects()
    assert "No projects registered yet" in capsys.readouterr().out

    cmd_projects(as_json=True)
    assert json.loads(capsys.readouterr().out) == {
        "registry_path": path, "count": 0, "projects": [],
    }


def test_the_shared_leaf_holds_no_wording_and_stays_a_stdlib_tier_leaf() -> None:
    """The leaf is a data shape, which is what let it live in `display.py`.

    What the composition locus forbids in this module is *targeting wording* —
    `mcp_server` imports it, so a `--project` or a `mitos init` here would be one
    import from an agent. 3b's own check was `grep -ic "project" display.py` → 0;
    that count is deliberately nonzero now, because this leaf is *named* for the
    thing, so the check has to become the rule it was standing in for.

    Applied to **emittable** literals, not to the source text: docstrings name the
    CLI verb the shape is shared with, which is orientation for a maintainer and
    can reach no caller. A rendered string is a literal that is not a docstring,
    and that is exactly the set swept here — the first draft of this row swept the
    whole file and reddened on its own prose, which would have been "fixed" by
    deleting the sentence that explains the leaf.
    """
    import ast

    tree = ast.parse(open(
        os.path.join(os.path.dirname(cli.__file__), "display.py"), encoding="utf-8"
    ).read())
    docstrings = {
        ast.get_docstring(node, clean=False)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef))
    }
    emittable = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value not in docstrings
    ]
    # Non-inertness, proven against a literal that is really there rather than by
    # asserting the set is non-empty: the shared not-found hint carries
    # `mitos sync`, a deliberate workflow pointer. If the filter ever excluded
    # real strings, this row would sweep an empty set and pass proving nothing.
    assert any("mitos sync" in literal for literal in emittable)

    for literal in emittable:
        for syntax in FORBIDDEN_SYNTAX:
            assert syntax not in literal, f"display.py emits {syntax!r}"

    assert projects_payload({}, "/p") == {
        "registry_path": "/p", "count": 0, "projects": []}
    assert isinstance(projects_payload({"a": "/x"}, "/p")["projects"], list)


# ---------------------------------------------------------------------------
# Group 6 — faults of the registry file and of a resolved workspace.
# ---------------------------------------------------------------------------

def test_a_malformed_registry_reaches_the_caller_as_one_calm_line(
    tmp_path, monkeypatch
) -> None:
    """Criterion 15 / G11: `RegistryError` propagates unwrapped, deliberately.

    There is no registered vocabulary to teach when the file holding it cannot be
    read, so dressing it in the anatomy would be a wall built out of data we do
    not have. Over the transport it becomes `Error executing tool X: <this line>`
    with `isError: True` and no traceback.
    """
    from mitos.errors import RegistryError

    path = _write_registry('"broken" = \n')
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RegistryError) as excinfo:
        mcp_server.list_scopes(project="anything")

    message = str(excinfo.value)
    assert path in message
    assert "\n" not in message.strip()
    assert "Did you mean" not in message
    assert "Registered projects" not in message


def test_a_resolved_target_with_a_malformed_config_stays_a_calm_line(
    tmp_path, monkeypatch
) -> None:
    """G11b: resolution proves a workspace's *shape*, not that its config parses.

    That is 2a's deliberate choice — a malformed config must be
    resolvable-then-diagnosed, not unresolvable, which is the wrong failure at the
    wrong altitude for a state `status` exists to report. So a resolved target can
    still raise, and no carve-out is invented here: it is the same pre-existing
    gap the CLI has, and 4b fixes it in one place for both surfaces.
    """
    from mitos.errors import ConfigError

    root = _workspace(tmp_path / "broken")
    with open(os.path.join(root, ".mitos", "config.toml"), "w", encoding="utf-8") as f:
        f.write("!!! not toml\n")
    _register(broken=root)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError) as excinfo:
        mcp_server.list_scopes(project="broken")

    assert "config.toml" in str(excinfo.value)
    assert "Traceback" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Group 7 — the transitional tripwires. INVERT these at 5b, never delete them.
# ---------------------------------------------------------------------------

def test_a_project_less_call_still_resolves_the_cwd_workspace(
    tmp_path, monkeypatch
) -> None:
    """TRANSITIONAL (phase 5b inverts this row). Criterion 5.

    Construction is not migration: this phase makes the selector *sayable* on the
    MCP surface, not *required*, so a call that names no project behaves exactly
    as it did before. 5b removes the working-directory fallback — the single
    `else: MitosConfig()` branch inside `_target_config` — at which point this
    must become an assertion that the same call renders the **missing** anatomy
    and arrives error-marked. The row exists so that flip is a decision someone
    makes rather than an omission someone notices.
    """
    cwd = _workspace(tmp_path / "cwd", scope="alpha")
    _register(other=_workspace(tmp_path / "other", scope="beta"))
    monkeypatch.chdir(cwd)

    payload = json.loads(mcp_server.list_decisions())

    assert [d["slug"] for d in payload["decisions"]] == ["alpha-probe"]
    assert payload["workspace"] == cwd
    # Transitional echo rule: no selector → the resolved workspace's absolute
    # path, so no mid-vision answer is ever unattributed. 5b removes the branch
    # that produces it along with the fallback itself.
    assert payload["project"] == cwd


def test_an_empty_project_renders_the_missing_anatomy_rather_than_falling_back(
    tmp_path, monkeypatch
) -> None:
    """TRANSITIONAL (phase 5b inverts this row's twin). Criterion 6.

    The gate is `is not None`, never truthiness. `project=""` is a **supplied**
    selector that carries no target — under `if project:` it would silently fall
    back to the working directory, and the caller who asked for something would
    get somewhere else. This is also what makes the `missing` class reachable and
    testable while the fallback is still live.

    At 5b the omitted case joins this one, and the row above becomes its
    duplicate; keep them both then — one pins the gate, the other pins the
    absence of a default.
    """
    cwd = _workspace(tmp_path / "cwd", scope="alpha")
    _register(other=_workspace(tmp_path / "other", scope="beta"))
    monkeypatch.chdir(cwd)

    body = _body("list_decisions", project="")

    assert body.startswith("no project was named")
    assert "alpha" not in body, "an empty selector must not resolve the cwd workspace"
    assert _vocabulary(body) == "other."
