"""The two agent-facing templates mitos *writes*: `.mitos/skill.md` and `agent-block`.

These are the artifacts that teach the next agent how to reach this project, and
they are copies — a repo commits them, a colleague clones them, another machine
reads them. Two properties follow, and this module is where both are pinned:

- **They persist no resolved project identity.** Not a name, not a path, not a
  placeholder. The deciding failure is *mis*-resolution rather than
  non-resolution: a literal that resolves to nothing costs one teaching error,
  while a literal that resolves to a *different real project* on the reader's
  machine costs a cross-project write. So the rows here are **byte-equality
  across two differently-named workspaces at different paths**, not substring
  absence — an absence assertion is satisfied by a template that says nothing at
  all (and by a leak the assertion did not think to name).
- **They teach the current addressing model.** Every claim of the form "the
  template no longer says X" is paired with a claim that it *does* say Y over the
  same artifact, for exactly that reason.

The workspaces are given different basenames deliberately: a same-basename pair
would let a path-leaking template pass by luck, and `cmd_init` registers, so a
collision also errors on the second init.
"""

import json

import pytest

from mitos import cli
from mitos._agent_block import AGENT_GUIDE_VERSION, agent_block
from mitos.config import MitosConfig


# --- helpers ---------------------------------------------------------------

def _init(path):
    """Runs ``cmd_init`` on a fresh ``MitosConfig`` for ``path``."""
    cli.cmd_init(MitosConfig(str(path)))


def _qdrant(reachable=True, collection_exists=False, points=None):
    """Builds a ``_check_qdrant`` stub (no real Qdrant in template/status tests)."""
    return lambda url, coll: {
        "reachable": reachable,
        "collection_exists": collection_exists,
        "points": points,
    }


def _skill(path):
    """Reads the ``skill.md`` written into ``path``'s workspace."""
    return (path / ".mitos" / "skill.md").read_text(encoding="utf-8")


@pytest.fixture
def two_workspaces(tmp_path):
    """Two initialized workspaces at different paths under different basenames.

    Different *names* as well as different paths: the registry rejects a
    same-basename second registration, and — more to the point — a template that
    leaked its workspace's basename would render identically in a same-basename
    pair and the byte-identity row would pass on a defect.
    """
    alpha = tmp_path / "alpha_ws"
    beta = tmp_path / "nested" / "beta_ws"
    alpha.mkdir(parents=True)
    beta.mkdir(parents=True)
    _init(alpha)
    _init(beta)
    return alpha, beta


# --- T21.1 — the travelling artifacts persist no identity -------------------

def test_skill_md_is_byte_identical_across_two_workspaces(two_workspaces):
    """`init` writes the same bytes wherever it runs — the machine-invariance row.

    `init` *always overwrites* `skill.md`, so machine-varying content would also
    dirty a tracked file on every setup in every project that commits `.mitos/`.
    Equality rather than a substring sweep: this is the row that fails when
    someone persists a resolved identity, and it cannot know in advance which
    identity they chose to persist.
    """
    alpha, beta = two_workspaces
    assert _skill(alpha) == _skill(beta)


def test_skill_md_names_no_workspace_path_or_basename(two_workspaces):
    """The direct reading of the same claim, so a failure says *what* leaked.

    The equality row above is the load-bearing one — it catches leaks nobody
    enumerated. This one exists because a bare `assert a == b` failure on a
    30-line template is a wall, and naming the leaked token is a vector.
    """
    alpha, beta = two_workspaces
    text = _skill(alpha)
    for token in (str(alpha), str(beta), "alpha_ws", "beta_ws"):
        assert token not in text, f"skill.md persisted {token!r}"


def test_agent_block_stdout_is_byte_identical_across_two_workspaces(
    two_workspaces, capsys
):
    """The block travels too, and it is driven through the verb, not the function.

    `agent_block()` takes no workspace argument, so asserting it against itself
    proves nothing about travel — the function structurally cannot vary by root.
    The claim is about what `mitos agent-block <root>` *prints*, so the row runs
    the handler against two roots and compares captured stdout. (3e's
    `test_agent_block_stdout_stays_byte_identical` is the complementary channel
    claim: stdout is the paste-ready artifact and the corpus echo rides stderr.)
    """
    alpha, beta = two_workspaces
    capsys.readouterr()

    assert cli.cmd_agent_block(str(alpha)) == 0
    out_alpha = capsys.readouterr().out
    assert cli.cmd_agent_block(str(beta)) == 0
    out_beta = capsys.readouterr().out

    assert out_alpha == out_beta
    for token in (str(alpha), str(beta), "alpha_ws", "beta_ws"):
        assert token not in out_alpha, f"the agent block persisted {token!r}"


# --- T21.2 — the format spec is included, not inlined -----------------------

def test_skill_md_includes_the_installed_format_spec(two_workspaces):
    """The canonical entry format has exactly one source, and `skill.md` includes it.

    Read through `cli.load_format_spec()` — the same call the template body makes
    — rather than against a hardcoded excerpt, so this pins *inclusion* and not a
    snapshot of the spec's current wording. A rewrite that inlines fragments forks
    the format; this is the row that catches it.
    """
    alpha, _ = two_workspaces
    spec = cli.load_format_spec()
    assert spec.strip()          # a vacuous spec would make the next line trivial
    assert spec in _skill(alpha)


# --- the addressing story, present as well as un-false ----------------------

def test_skill_md_teaches_both_addressing_surfaces_and_drops_the_isolation_promise(
    two_workspaces,
):
    """One row, three claims — because the absence claim is worthless alone.

    An assertion that the template "no longer promises isolation" passes against a
    template that says nothing at all, so the two positive claims share the row:
    the MCP half (name the workspace's absolute path) and the CLI half (`-p .`
    from the workspace root, or the positional on `status`/`agent-block`).

    The falsified promise was *"you will not see, and cannot contaminate, other
    projects' decisions"*. Its first clause survives elsewhere in the file — the
    per-project graph and collection are real — but the protection is now
    isolation-by-naming, not isolation-by-inability.
    """
    text = _skill(two_workspaces[0])

    # Present: the MCP form is the absolute path of the workspace being read from.
    assert "absolute path of the workspace directory" in text
    # Present: the CLI form, on both spellings the flip left standing.
    assert "-p ." in text
    assert "mitos status ." in text
    # Present: the echo, which is what makes a mis-aim visible now that it is possible.
    assert "project · collection · workspace" in text

    # Absent: the promise the vision falsified.
    assert "cannot contaminate" not in text
    assert "you will not see" not in text


def test_skill_md_spells_no_bare_workspace_verb_recipe(two_workspaces):
    """Every CLI recipe the template teaches carries a selector.

    Post-flip a selector-less call on a require-list verb hard-fails, so a recipe
    without one is a command mitos ships that mitos rejects. Checked over the
    require-list verbs the template actually spells; `init` is selector-exempt and
    `status` is selector-*optional* (it answers about the machine instead), so
    both are excluded here and covered by the presence rows above.

    Scoped to the template's **own** body — the included `format-spec.md` is
    subtracted first. The spec's one hit (`Source:`'s note that `mitos import`
    emits `import_llm`) is prose naming which verb stamps a field, not a command
    the reader is told to run, and the spec is not this phase's file to edit.
    """
    text = _skill(two_workspaces[0]).replace(cli.load_format_spec(), "")
    for verb in ("record", "surface", "query", "list", "sync", "check", "import"):
        assert f"`mitos {verb}`" not in text, f"bare `mitos {verb}` recipe in skill.md"


def test_skill_md_keeps_the_contracts_older_than_the_addressing_rewrite(two_workspaces):
    """The four steers the rewrite had to carry through — re-flowing is how they get lost.

    Each predates this vision: the long-name CLI aliases (the surface mitos itself
    points agents at), read-then-draft (the recall loop), engage-with-
    `rejected_paths` (what stops the next agent re-proposing a rejected path), and
    declare-the-relation (graph connectivity, all seven edge types).
    """
    text = _skill(two_workspaces[0])
    assert "the CLI also accepts the long names as aliases" in text
    assert "This is the recall loop — use it first." in text
    assert "that is what stops the next agent re-proposing it" in text
    assert "## Linking decisions" in text
    for edge in ("supersedes", "amends", "narrows", "depends_on",
                 "resolves", "contradicts", "cites"):
        assert f"`{edge}`" in text


def test_agent_block_teaches_the_computed_path_and_targeted_habits(two_workspaces):
    """The block carries the *instruction* to compute the path, never a path.

    Its two habit lines are the phase's silent-failure class: a bare `mitos
    status` post-flip does not error — it succeeds and answers about the whole
    machine, with no per-project verdict anywhere in the output — while a bare
    `mitos check` hard-fails. Neither may ship in a v3 block, or the version bump
    is a half-migration.
    """
    block = agent_block()
    assert "absolute path of the directory this file is in" in block
    assert "mitos status ." in block
    assert "mitos check -p ." in block
    assert "`mitos status`" not in block
    assert "`mitos check`" not in block


def test_agent_block_keeps_the_heading_its_own_drift_check_needs(two_workspaces):
    """`_BLOCK_HEADING_RE` is what detects a PRE-marker paste — break it and nothing reds.

    Every unversioned legacy copy in the wild is found by a heading naming both
    "Architectural Decisions" and "Mitos". A rewrite that renames the heading
    silently stops reporting them: a regression whose only symptom is a check that
    finds nothing.
    """
    from mitos._agent_block import _BLOCK_HEADING_RE, scan_agent_files

    assert _BLOCK_HEADING_RE.search(agent_block()) is not None

    alpha, _ = two_workspaces
    legacy = "# Contributing\n\n## Architectural Decisions — Mitos\n\nsome old prose.\n"
    (alpha / "AGENTS.md").write_text(legacy, encoding="utf-8")
    files = scan_agent_files(str(alpha))
    assert [f["status"] for f in files] == ["unversioned"]


# --- T21.3 — the version bump is the pasted copies' only migration channel ---

_V2_PASTE = (
    "# Project\n\n"
    "<!-- mitos-agent-guide: v2 — managed block, refresh with `mitos agent-block`. -->\n"
    "## Architectural Decisions — Mitos (per-project)\n"
    "Run `mitos status`. If it isn't `READY ✓`, follow the setup guide.\n"
    "<!-- /mitos-agent-guide -->\n"
)


def test_a_v2_paste_is_outdated_to_agent_block_check(tmp_path, capsys):
    """The literal `v2` that exists on disk today is reported stale. Consumer 1 of 2.

    A literal `2`, deliberately — not `AGENT_GUIDE_VERSION - 1`. The claim is about
    the pastes that exist in the wild right now; a self-adjusting spelling would
    state a tautology and stay green through any bump, including a missing one.
    """
    ws = tmp_path / "pasted"
    ws.mkdir()
    _init(ws)
    (ws / "AGENTS.md").write_text(_V2_PASTE, encoding="utf-8")
    capsys.readouterr()

    assert cli.cmd_agent_block(str(ws), check=True) == 1
    assert "outdated" in capsys.readouterr().out


def test_a_v2_paste_is_outdated_to_status(tmp_path, monkeypatch, capsys):
    """…and consumer 2: `mitos status`'s drift line and its `--json` version field.

    Two rows rather than one because they are two code paths — `agent-block
    --check` renders its own report, while `status` folds the same scan into the
    readiness surface. A bump that reached only one of them would leave half the
    pasted copies unmigrated.
    """
    ws = tmp_path / "pasted"
    ws.mkdir()
    _init(ws)
    (ws / "AGENTS.md").write_text(_V2_PASTE, encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, False))
    capsys.readouterr()

    cli.cmd_status(str(ws))
    assert "agent-file mitos note out of date" in capsys.readouterr().out

    cli.cmd_status(str(ws), as_json=True)
    data = json.loads(capsys.readouterr().out)
    assert data["agent_guide_version"] == 3 == AGENT_GUIDE_VERSION
    assert [f["marker_version"] for f in data["agent_files"]] == [2]
    assert [f["status"] for f in data["agent_files"]] == ["outdated"]


# --- the shadowing-entry finding, both directions ---------------------------

def test_a_project_scope_mcp_entry_is_reported_as_a_finding(tmp_path, monkeypatch, capsys):
    """A `.mcp.json` naming `mitos` here shadows the machine-wide server — say so.

    The note states the fact rather than prescribing deletion, because mitos
    cannot tell a stale entry from one deliberately kept identical to the
    machine-wide registration, and "delete it" would be wrong advice half the
    time. What is true in both states is the precedence.
    """
    ws = tmp_path / "shadowed"
    ws.mkdir()
    _init(ws)
    (ws / ".mcp.json").write_text(
        '{"mcpServers": {"mitos": {"command": "mitos", "args": ["serve"]}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, False))
    capsys.readouterr()

    cli.cmd_status(str(ws))
    out = capsys.readouterr().out
    assert "declares its own `mitos` MCP server" in out
    assert "takes precedence" in out
    # It states the fact; it does not order a deletion.
    assert "delete it" not in out.lower()


def test_a_fresh_init_reports_no_mcp_row_at_all(tmp_path, monkeypatch, capsys):
    """The negative half, and it is the one that keeps a healthy report clean.

    A project with no `.mcp.json` is now the *correct* state, not an unfinished
    one — empty/fresh is healthy. So there is no row, no `—`, and no nudge: the
    retired per-project wiring recommendation left nothing behind that renders.
    """
    ws = tmp_path / "clean"
    ws.mkdir()
    _init(ws)
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, False))
    capsys.readouterr()

    assert cli.cmd_status(str(ws)) == 0
    out = capsys.readouterr().out
    assert "READY ✓" in out
    assert "MCP" not in out
    assert "mcp" not in out
