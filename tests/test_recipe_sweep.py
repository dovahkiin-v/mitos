"""The consolidated recipe sweep: the census gaps, each through the real parser.

Vision constraint 6 asks that every CLI recipe the AX Hardening 3 vision wrote or
edited is proven by taking the line from real output and putting it through
``cli._build_parser().parse_args``. Most recipes already have such a row beside
their producer. This module holds only the recipes the 8a1 census found proven by
a substring or regex alone, or by no row at all, so there is one row per gap and no
second copy of fixtures that already parse. Every gap names its project with a
literal (`-p .`) or a placeholder, so no row here depends on how a project name is
quoted; the `!r` interpolation sites are proven by their own producers' rows.

Each row drives the real producer, extracts every recipe from what it wrote,
parses it, and resolves its selector back to the workspace through the registry
rather than by comparing strings (``init`` registers, so a name and a path can
both be right).
"""

import os
import re
import shlex

import pytest

from mitos import cli
from mitos.config import MitosConfig


def _parse(recipe: str):
    """Parses one ``mitos …`` recipe, turning argparse's exit into a red."""
    tokens = shlex.split(recipe)
    assert tokens[0] == "mitos", recipe
    try:
        return cli._build_parser().parse_args(tokens[1:])
    except SystemExit as exc:  # argparse refuses by exiting
        pytest.fail(f"{recipe!r} does not parse (argparse exit {exc.code})")


def _resolves_to(args, workspace) -> None:
    """The recipe's selector, from either side of the verb, names ``workspace``."""
    selector = cli._selector_from_args(args)
    assert selector is not None, f"{args.command} names no project"
    resolved = cli._resolve_selector(selector, args.command)
    assert os.path.realpath(resolved.root) == os.path.realpath(str(workspace))


# --- skill.md's CLI twins (4b, 5d edited the lines) -------------------------

# One line per MCP tool: "- `tool` (CLI: `mitos …`)". The same shape
# tests/test_templates.py reads, where the twins were checked by substring only.
_TWIN_LINE = re.compile(r"^- `(\w+)`\s+\(CLI: `(mitos [^`]+)`\)", re.M)

# The twins are a verb map, not complete commands: three name a verb whose
# required argument the line leaves to the reader. The parse completes exactly
# that argument, as the `<slug>` placeholder is replaced below. A twin that grows
# a new required argument reds here until it is listed.
_TWIN_COMPLETIONS = {
    "record": ["--slug", "some-slug"],
    "query": ["some claim"],
    "surface": ["some claim"],
}


def _twin_recipes(workspace) -> dict:
    cli.cmd_init(MitosConfig(str(workspace)))
    text = (workspace / ".mitos" / "skill.md").read_text(encoding="utf-8")
    text = text.replace(cli.load_format_spec(), "")
    return dict(_TWIN_LINE.findall(text))


def test_every_skill_md_cli_twin_parses_and_names_its_workspace(tmp_path,
                                                                 monkeypatch) -> None:
    """Each twin parses, names the verb its tool maps to, and ``-p .`` resolves here.

    ``-p .`` is a literal in the travelling file, so it is resolved from the
    workspace directory, which is where the file tells its reader to stand.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    twins = _twin_recipes(workspace)
    assert len(twins) >= 8  # population floor: the row cannot pass on nothing
    monkeypatch.chdir(workspace)
    for tool, recipe in twins.items():
        if recipe == "mitos projects":  # selector-exempt, correctly bare
            args = _parse(recipe)
            assert args.command == "projects"
            continue
        recipe = recipe.replace("<slug>", "my-slug").replace(" …", "")
        verb = shlex.split(recipe)[1]
        args = _parse(recipe + "".join(
            " " + shlex.quote(t) for t in _TWIN_COMPLETIONS.get(verb, [])))
        assert args.command == verb, tool
        assert args.project_post == ".", f"{tool}'s twin {recipe!r} names no project"
        _resolves_to(args, workspace)


def test_the_twin_completions_are_each_still_needed(tmp_path) -> None:
    """A completion that is no longer needed is a stale entry, not a harmless one.

    Each listed verb's twin must fail to parse as written; otherwise the table is
    hiding nothing and should shrink.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    twins = {shlex.split(r)[1]: r for r in _twin_recipes(workspace).values()}
    for verb in _TWIN_COMPLETIONS:
        with pytest.raises(SystemExit):
            cli._build_parser().parse_args(shlex.split(twins[verb])[1:])


# --- shipped prose: README's audit bullet, SETUP's two placeholder recipes ---

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _doc(name: str) -> str:
    with open(os.path.join(_REPO_ROOT, name), encoding="utf-8") as f:
        return f.read()


def _spans(text: str) -> list:
    return [s for s in re.findall(r"`([^`\n]+)`", text) if s.startswith(("mitos ", "… "))]


def _registered(tmp_path, name: str):
    """An ``init``ed workspace registered under its basename ``name``."""
    workspace = tmp_path / name
    workspace.mkdir()
    cli.cmd_init(MitosConfig(str(workspace)))
    return workspace


def test_readme_audit_bullet_recipes_parse_and_resolve(tmp_path, monkeypatch) -> None:
    """README's "It audits itself" bullet: both recipes, `-p .` from the workspace (3i)."""
    (para,) = [p for p in _doc("README.md").split("\n") if "It audits itself" in p]
    recipes = _spans(para)
    assert len(recipes) >= 2  # population floor
    workspace = _registered(tmp_path, "ws")
    monkeypatch.chdir(workspace)
    verbs = set()
    for recipe in recipes:
        args = _parse(recipe)
        assert args.project_post == ".", recipe
        _resolves_to(args, workspace)
        verbs.add(args.command)
    assert verbs == {"check", "hook-install"}


@pytest.fixture(params=["harbor", "my proj"], ids=["plain", "with-space"])
def named(request, tmp_path):
    return request.param, _registered(tmp_path, request.param)


def test_setup_block_message_recipe_parses_with_the_name_filled(named) -> None:
    """SETUP's commit-gate prose quotes the block's recipe as ``… check -p '<name>'`` (3i).

    ``…`` stands for the ``mitos`` the hook runs and ``<name>`` for the project,
    already single-quoted in the prose; both are filled and the line is parsed.
    """
    name, workspace = named
    (recipe,) = [s for s in _spans(_doc("SETUP.md")) if s.startswith("… check")]
    filled = "mitos" + recipe[1:].replace("<name>", name)
    args = _parse(filled)
    assert args.command == "check"
    _resolves_to(args, workspace)


def test_setup_full_pointer_recipe_parses_with_the_project_filled(named) -> None:
    """SETUP's divergence note names ``mitos sync -p <project> --full`` (7c)."""
    name, workspace = named
    recipes = [s for s in _spans(_doc("SETUP.md"))
               if s.startswith("mitos sync -p <project> --full")]
    assert recipes  # population floor
    for recipe in recipes:
        args = _parse(recipe.replace("<project>", shlex.quote(name)))
        assert args.command == "sync" and args.full
        _resolves_to(args, workspace)


def test_agent_block_check_habit_recipe_parses_and_resolves(tmp_path,
                                                            monkeypatch) -> None:
    """The agent block's "Keep the graph honest" habit names a check an agent runs verbatim (3i)."""
    from mitos._agent_block import agent_block

    (line,) = [ln for ln in agent_block().splitlines() if "Keep the graph honest" in ln]
    recipes = _spans(line)
    assert recipes  # population floor
    workspace = _registered(tmp_path, "ws")
    monkeypatch.chdir(workspace)
    for recipe in recipes:
        args = _parse(recipe)
        assert args.command == "check"
        assert args.project_post == "."
        _resolves_to(args, workspace)


# --- MCP side: the strings the census found with no no-shell row -------------
#
# The house assertion is `"mitos " not in text`. That substring shape would also
# refuse the product's own name in prose ("so mitos never shortens one", the
# `slug_too_long` message), which names no command. So these rows ask the
# question constraint 6 asks, whether the text hands an agent a shell command: a
# `mitos` followed by one of the parser's own verbs.

def _verbs() -> list:
    from test_cli_selector import _subparsers
    return sorted(_subparsers(cli._build_parser()), key=len, reverse=True)


def _shell_tokens(text: str) -> list:
    pattern = r"\bmitos\s+(?:" + "|".join(re.escape(v) for v in _verbs()) + r")\b"
    return re.findall(pattern, text)


def test_the_shell_token_pattern_has_teeth() -> None:
    """In-row control: the pattern finds a command and leaves the product's name."""
    assert _shell_tokens("run `mitos sync -p 'x'` first") == ["mitos sync"]
    assert _shell_tokens("so mitos never shortens one") == []


def _tool_descriptions() -> dict:
    import asyncio
    from mitos import mcp_server
    return {t.name: t.description for t in asyncio.run(mcp_server.mcp.list_tools())}


@pytest.mark.parametrize("tool", ["record_decision", "surface_decisions",
                                  "query_decisions", "list_decisions"])
def test_the_edited_tool_descriptions_name_no_shell_command(tool) -> None:
    """4b–6a edited these four descriptions, which every MCP client receives whole."""
    text = _tool_descriptions()[tool]
    assert len(text) > 500  # the real description, not an empty stand-in
    assert _shell_tokens(text) == []


@pytest.mark.parametrize("brief,full_top", [(True, 1), (False, -1)],
                         ids=["brief-and-full-top", "negative-full-top"])
def test_the_full_top_refusals_name_no_shell_command(brief, full_top) -> None:
    """4b's two depth-argument faults, through the tool itself (both ranked reads)."""
    import json
    from mitos import mcp_server
    for read in (mcp_server.surface_decisions, mcp_server.query_decisions):
        body = json.loads(read("claim", brief=brief, full_top=full_top))
        assert "full_top" in body["error"]
        assert _shell_tokens(body["error"]) == []


def test_the_slug_too_long_refusal_names_no_shell_command(tmp_path) -> None:
    """4e's message, off the record path an MCP `record_decision` returns whole."""
    from mitos.sync import MitosSyncManager
    workspace = _registered(tmp_path, "ws")
    res = MitosSyncManager(MitosConfig(str(workspace))).record_decision_entry(
        "An axiom.", "rej", ["s"], slug="a" * 101)
    assert res.get("code") == "slug_too_long", res
    text = " ".join(str(v) for v in res.values())
    assert "100" in text
    assert _shell_tokens(text) == []


def test_the_pause_message_names_no_shell_command(tmp_path, monkeypatch) -> None:
    """4d/4e's pause body, with both echo groups rendered, off the real record path."""
    import test_neighbor_review as nr
    from mitos.sync import MitosSyncManager
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    workspace = _registered(tmp_path, "ws")
    m = MitosSyncManager(MitosConfig(str(workspace)))
    nr._seed_echo_corpus(m)
    res = nr._echo_pause(m, nr._ECHO_GATHERED, cites="echo-b, echo-d")
    message = res["message"]
    # Every clause the census listed is present, so the row reads the whole body.
    for part in ("amended_by/narrowed_by", "Nothing is held between calls",
                 "draft_digest", "Declared", "Pass every declaration again"):
        assert part in message, part
    assert _shell_tokens(message) == []
