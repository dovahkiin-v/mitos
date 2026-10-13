"""T4 (5b half) — the pre-commit hook recipe, end-to-end in a temp git repo.

The `SETUP.md` pre-commit recipe (4b) is the artifact under test: the
`git diff --quiet -- decisions.md` divergence guard + `mitos check --staged`,
reproduced VERBATIM here (KD4 — a reword in either place diverges the test; the
`test_setup_recipe_is_in_lockstep` tripwire catches drift). Two disciplines:

* **Keyless deterministic half (always runs)** — drives the recipe's SHELL LOGIC
  through a real `git commit` with a FAKE `mitos` on PATH: the divergence guard
  fires (verbatim message) BEFORE any check runs, and the recipe wires the check's
  exit code straight to the commit's success/failure. Hermetic — no keys, no judge,
  no Qdrant; it tests the RECIPE, not the engine (the staged engine's bad-buffer→1
  behaviour is W9's `test_check_staged.py`).
* **Live half (`HAS_LIVE_KEYS`-gated)** — one real end-to-end with the shipped
  binary: a real workspace, a pending undeclared contradiction against an indexed
  decision → the hook blocks the commit; a clean (no-pending) buffer → the commit
  passes.

The divergence message's em-dash is U+2014 (byte-confirmed against SETUP.md; no
line number here — the recipe moves, and a stale anchor is worse than none).

Since 6b this module also carries the **doc-shape** rows for SETUP.md's other
recipes: the scheduled sweep, the secretless-CI prose that points at it, the
corpus↔graph repair, and the `SETUP.md → <Heading>` pointers `mitos/` prints.
They live here because this is already the module that reads SETUP.md from disk.
Since 3i it also carries the commit gate's static prose rows (SETUP.md's gating
section, keys section and `-p` table, and README's audit bullet); none reaches a
key or a network.
"""

import asyncio
import os
import pathlib
import re
import shlex
import subprocess
import uuid

import pytest
import requests

from live_helpers import live_tests_disabled
from mitos import cli, mcp_server
from mitos.config import CONFIG_DEFAULTS, default_collection_name

# --- The recipe under test (VERBATIM from SETUP.md — keep in lockstep, KD4) ----
# The shebang is hook scaffolding (implementer's latitude); the load-bearing
# verbatim parts are the guard command and the divergence message.
RECIPE = """#!/bin/sh
# Fail loudly if decisions.md differs between the index and the working tree —
# `mitos check --staged` reads the WORKING TREE, git commits the INDEX; a divergence
# would gate the wrong bytes (a bad entry fixed-but-not-restaged slips the gate).
if ! git diff --quiet -- decisions.md; then
    echo "decisions.md has unstaged changes — stage or stash them before committing" >&2
    exit 1
fi
mitos check --staged -p .
"""

# The verbatim divergence message the guard emits (em-dash U+2014).
DIVERGENCE_MESSAGE = (
    "decisions.md has unstaged changes — stage or stash them before committing"
)
# The verbatim guard command (the recipe's git plumbing).
GUARD_COMMAND = "git diff --quiet -- decisions.md"
# The verbatim gate command. `-p .` is correct **specifically** because git runs
# hooks from the worktree top level, so `.` IS the workspace root at run time — the
# discriminator is *is cwd the workspace root when this runs*, never *does the
# artifact travel*. Added to the lockstep set at 5a: the two constants above name
# the guard and the message, so an edit to this line alone would have left the
# lockstep row green while `RECIPE` and SETUP.md silently diverged.
GATE_COMMAND = "mitos check --staged -p ."

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SETUP_PATH = os.path.join(_REPO_ROOT, "SETUP.md")
_VENV_BIN = os.path.join(_REPO_ROOT, "venv", "bin")

# --- Reference-corpus delete pair (a known-good undeclared contradiction) ------
_HARD_DELETE = (
    "### harbor-delete-is-immediate-hard\n"
    "**Decided:** Harbor deletes are immediate and irreversible — the blob and its "
    "metadata are purged at once.\n"
    "**Rejected:** A grace period — regulated tenants require provable immediate "
    "erasure on request.\n"
    "**Scope:** storage\n"
)
_SOFT_DELETE = (
    "\n### harbor-delete-is-soft-30d\n"
    "**Decided:** Harbor deletes are soft: a deleted file is recoverable for 30 days "
    "before purge.\n"
    "**Rejected:** Immediate hard delete — one fat-fingered call loses a customer's "
    "data with no recourse.\n"
    "**Scope:** storage\n"
)


def _load_live_env() -> None:
    """Loads keys from the repo-root .env into os.environ (mirrors the live suites)."""
    env_path = os.path.join(_REPO_ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


_load_live_env()
HAS_LIVE_KEYS = (not live_tests_disabled()) and bool(
    os.environ.get("GEMINI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")
)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:7333")


# --------------------------------------------------------------------------- #
# Scaffolding
# --------------------------------------------------------------------------- #

def _run(args, cwd, env=None):
    return subprocess.run(
        args, cwd=str(cwd), env=env, capture_output=True, text=True
    )


def _init_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "-q"], repo)
    _run(["git", "config", "user.email", "hook@test"], repo)
    _run(["git", "config", "user.name", "hook test"], repo)
    _run(["git", "config", "commit.gpgsign", "false"], repo)
    return repo


def _install_hook(repo):
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text(RECIPE, encoding="utf-8")
    hook.chmod(0o755)


def _fake_mitos(bin_dir, marker, exit_code):
    """Writes a fake `mitos` that records its invocation and exits a fixed code."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "mitos"
    script.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{marker}"\nexit {exit_code}\n',
        encoding="utf-8",
    )
    script.chmod(0o755)


def _env_with_path(bin_dir, extra=None):
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env.get("PATH", "")
    if extra:
        env.update(extra)
    return env


# --------------------------------------------------------------------------- #
# Lockstep tripwire
# --------------------------------------------------------------------------- #

def test_setup_recipe_is_in_lockstep():
    """The verbatim guard + message this test asserts still appear in SETUP.md (KD4).

    If 4b's recipe is reworded, this fails — the signal to update BOTH in lockstep.
    """
    setup = open(_SETUP_PATH, encoding="utf-8").read()
    assert GUARD_COMMAND in setup, (
        "the divergence guard command drifted from SETUP.md — update in lockstep"
    )
    assert DIVERGENCE_MESSAGE in setup, (
        "the divergence message drifted from SETUP.md — update in lockstep "
        "(check the em-dash is U+2014)"
    )
    assert GATE_COMMAND in setup, (
        "the gate command drifted from SETUP.md — update in lockstep"
    )
    assert GATE_COMMAND in RECIPE
    # The em-dash is U+2014, not a hyphen or U+2013 (byte-confirmed).
    assert "—" in DIVERGENCE_MESSAGE


# --------------------------------------------------------------------------- #
# 6b — doc-shape rows over SETUP.md's other recipes
# --------------------------------------------------------------------------- #

_MITOS_PKG = pathlib.Path(_REPO_ROOT) / "mitos"


def _setup_text():
    return pathlib.Path(_SETUP_PATH).read_text(encoding="utf-8")


def _extract_section(text, heading):
    """Returns the body under ``heading``, up to the next markdown heading.

    Boundaries are matched on ``^#{2,}\\s`` deliberately, **never** on a bare
    ``^#``: SETUP.md carries ``#``-prefixed *shell comments inside fenced blocks*,
    and the scheduled-sweep block opens with two of them. A ``^#`` splitter cuts
    that section off one line above its only two recipe lines, which would leave
    the presence row below red after a correct fix and the absence row passing
    vacuously — the exact pair of wrong answers these rows exist to prevent.
    """
    lines = text.splitlines()
    start = next(
        (i for i, ln in enumerate(lines)
         if re.match(r"^#{2,}\s", ln) and ln.split(" ", 1)[1].strip() == heading),
        None,
    )
    assert start is not None, f"SETUP.md has no heading {heading!r} — it was renamed or lost"
    body = []
    for ln in lines[start + 1:]:
        if re.match(r"^#{2,}\s", ln):
            break
        body.append(ln)
    return "\n".join(body)


def _fenced_blocks(section, lang):
    """Returns the ``lang``-tagged fenced code blocks in a section body, joined.

    Two reasons the language tag is required rather than optional. The absence rows
    below belong on the **recipe**, not on the section — the prose around the cron
    recipe explains *why* ``-p .`` is wrong there, so a whole-section absence claim
    reds on the sentence that teaches the rule (found by running it). And the
    corpus↔graph section carries an *untagged* fence holding a verbatim sample of
    `mitos status` output, whose ``mitos rebuild`` is quoted shipped text this phase
    must not edit; matching every fence would red on the tool's own words.
    """
    return "\n".join(
        re.findall(r"^```" + lang + r"\n(.*?)^```", section, re.S | re.M)
    )


def test_scheduled_sweep_recipe_names_a_registered_project():
    """cron starts in ``$HOME``, so the nightly sweep must name a project, not ``.``.

    Measured: ``mitos check -p .`` from ``/home/vinga`` exits **2** — the
    fail-closed code — so a mis-aimed scheduled job is indistinguishable from a
    real substrate outage in an exit-code branch, on the one surface whose stderr
    goes to a mailbox nobody reads. The registered-name form is safe *here*
    specifically because a crontab is machine-local state that never travels; the
    same literal in a committed file would name a different project on another
    machine.

    Presence and absence are asserted together over the SAME extracted block. An
    absence claim alone is satisfied by a block that says nothing at all — which
    is exactly the state this file was in before 6b (no selector *and* no ``-p .``,
    so the absence half passed vacuously against a broken recipe).
    """
    recipe = _fenced_blocks(
        _extract_section(_setup_text(), "Scheduled corpus sweep (cron)"), "sh"
    )
    assert "mitos check" in recipe, "the scheduled-sweep block lost its recipe"
    assert re.search(r"mitos check[^\n]*-p \w", recipe), (
        "the scheduled-sweep recipe names no project — cron runs from $HOME, so "
        "it must carry `-p <registered-name>`"
    )
    assert "-p ." not in recipe, (
        "the scheduled-sweep recipe uses `-p .`, which resolves to $HOME under cron"
    )


def test_secretless_ci_prose_does_not_re_teach_a_bare_sweep():
    """The third bare `check` site sat in *prose*, where a fenced-block reader misses it.

    The secretless-CI bullet advises running the corpus sweep on a keyed schedule
    — i.e. the cron form — in the one paragraph a reader consults precisely when
    wiring an unattended job. Paired presence/absence again: the section must
    still send the reader to the scheduled recipe, and must not spell a bare
    ``mitos check`` beside that advice.
    """
    block = _extract_section(
        _setup_text(), "Keys, Qdrant, and the secretless-CI consequence"
    )
    assert "corpus sweep" in block, "the secretless-CI bullet lost its subject"
    assert "`mitos check`" not in block, (
        "the secretless-CI bullet spells a bare `mitos check`; it takes the "
        "scheduled (registered-name) form, not the CI one"
    )


def test_corpus_graph_repair_recipe_names_its_project_on_every_line():
    """The one recipe in the file whose mis-aim *writes* another project's gold source.

    ``restore-source --all-graph-only`` splices blocks into ``decisions.md``, so a
    wrong target here is not a failed read — it is a mutation of the file P6 makes
    authoritative. Presence over the section (all three lines, both
    ``restore-source`` forms) plus an absence over the executable fence only — the
    surrounding prose legitimately names ``mitos rebuild`` as a subject, and the
    untagged fence beside it quotes shipped `mitos status` output verbatim.
    """
    section = _extract_section(_setup_text(), "When the corpus and the graph disagree")
    recipe = _fenced_blocks(section, "bash")
    for verb in ("restore-source", "rebuild"):
        assert f"mitos {verb} -p ." in recipe, f"`mitos {verb}` lost its selector"
    assert recipe.count("mitos restore-source -p .") == 2, (
        "both restore-source lines (--dry-run and the real splice) must name the project"
    )
    bare = re.findall(r"mitos (?:restore-source|rebuild)(?! -p\b)", recipe)
    assert not bare, f"selector-less repair recipe in the executable block: {bare}"


def test_every_setup_md_pointer_in_mitos_names_a_heading_that_exists():
    """`SETUP.md → <Heading>` is printed by production code the `.md` sweep cannot see.

    Two `cli.py` strings route a reader to a SETUP.md section **by title**, and
    nothing else in the tree would notice a rename — 6a cleared out both `§`-number
    references, so numbers are free and titles are not. Matching is by prefix after
    stripping trailing punctuation, because the pointers say `Cutover` while the
    heading is `Cutover (migrating a prototype graph to V1a)`.
    """
    headings = [
        ln.split(" ", 1)[1].strip()
        for ln in _setup_text().splitlines()
        if re.match(r"^#{2,6}\s", ln)
    ]
    pointers = set()
    for path in _MITOS_PKG.rglob("*.py"):
        for target in re.findall(
            r"SETUP\.md\s*→\s*(\S+)", path.read_text(encoding="utf-8")
        ):
            pointers.add(target.rstrip('.)",\\'))

    assert pointers, "no `SETUP.md → <Heading>` pointer found — did the regex rot?"
    for target in sorted(pointers):
        assert any(h.startswith(target) for h in headings), (
            f"`mitos/` points at SETUP.md → {target!r}, which no heading matches. "
            f"Headings: {headings}"
        )


# --------------------------------------------------------------------------- #
# 4d — SETUP.md's tool tables and the amend reach, derived from the code
# --------------------------------------------------------------------------- #

def _live_tool_names():
    """The live MCP tool set, off the server — never a hand list."""
    return {tool.name for tool in asyncio.run(mcp_server.mcp.list_tools())}


def _cli_verbs():
    """Every CLI subparser name, aliases included."""
    return set(cli._build_parser()._subparsers._group_actions[0].choices)


def _tables(section):
    """Parses the markdown tables in ``section``, in order.

    Each table is a list of rows, each row a list of stripped cells; the header
    and its ``|---|`` separator are dropped. A row whose cell count differs from
    its header's fails loudly rather than being skipped — a pipe inside a cell
    splits the row, and inline backticks do not protect it.
    """
    tables, current, width = [], None, None
    for ln in section.splitlines():
        if not ln.startswith("|"):
            current = None
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if current is None:
            current, width = [], len(cells)
            tables.append(current)
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        assert len(cells) == width, f"table row has {len(cells)} cells, header {width}: {ln!r}"
        current.append(cells)
    return tables


def _code(cell):
    """Strips a cell to its code text: backticks and the ★ marker removed."""
    return cell.replace("`", "").replace("★", "").strip()


def test_setup_tool_tables_equal_the_live_tool_and_alias_sets():
    """Both tables in "CLI vs MCP" enumerate the tool set, and prose never goes red.

    Capability map: its MCP column (less `—`) equals the live tool set, and each
    tool's CLI cell is a real verb. Name map: its first column equals the tool
    set, its verb column names real verbs, and its alias column (less `—`) equals
    the parser's subparser names that *are* tool names — the alias set derived,
    not typed.
    """
    tools = _live_tool_names()
    verbs = _cli_verbs()
    aliases = verbs & tools
    assert len(tools) >= 8 and "amend_commentary" in tools   # non-vacuity
    assert aliases and not tools <= verbs                     # some, not all, aliased

    section = _extract_section(_setup_text(), "CLI vs MCP — which surface does what")
    tables = _tables(section)
    assert len(tables) == 2, f"expected the capability map and the name map, got {len(tables)}"
    capability, names = tables
    assert all(len(row) == 3 for row in capability + names)

    mcp_cells = {_code(row[2]): _code(row[1]) for row in capability if _code(row[2]) != "—"}
    assert set(mcp_cells) == tools
    for tool, cli_cell in mcp_cells.items():
        assert cli_cell.startswith("mitos ") and cli_cell.split()[1] in verbs, (tool, cli_cell)

    assert {_code(row[0]) for row in names} == tools
    for row in names:
        assert _code(row[1]).split()[1] in verbs, row
    assert {_code(row[2]).removeprefix("mitos ") for row in names if _code(row[2]) != "—"} == aliases


def test_setup_amend_reach_states_the_shipped_rotation_defaults():
    """The reach sentence's two numbers are hand-typed, so each is pinned to the code."""
    section = _extract_section(_setup_text(), "When the corpus and the graph disagree")
    # `[\s>]+`: the sentence sits in a blockquote, so a wrap carries a `> ` prefix.
    volume = re.search(r"`rotation_volume_threshold_entries`[\s>]+\(default (\d+)\)", section)
    lag = re.search(r"`rotation_lag_days`[\s>]+\(default (\d+) days\)", section)
    assert volume and lag, "the amend reach blockquote lost a default"
    assert int(volume.group(1)) == CONFIG_DEFAULTS["rotation_volume_threshold_entries"]
    assert int(lag.group(1)) == CONFIG_DEFAULTS["rotation_lag_days"]


# --------------------------------------------------------------------------- #
# 3i — the commit gate's prose: SETUP.md's gating section, the -p table, README
# --------------------------------------------------------------------------- #

_GATING_HEADING = "Gating commits and auditing with `mitos check`"
_GATE_HEADING = "The commit gate (`mitos hook-install`)"
_STAGED_HEADING = "Staged gate for a hand-authored `decisions.md`"


def _extract_h2_section(text, heading):
    """Returns a ``##`` section's whole body, its ``###`` subsections included.

    `_extract_section` stops at any heading on purpose (its cron row needs that),
    so a claim about "the gating section" needs its own reader: this one stops
    only at the next ``## `` heading.
    """
    lines = text.splitlines()
    start = next(
        (i for i, ln in enumerate(lines)
         if ln.startswith("## ") and ln[3:].strip() == heading),
        None,
    )
    assert start is not None, f"SETUP.md has no section {heading!r} — it was renamed or lost"
    body = []
    for ln in lines[start + 1:]:
        if ln.startswith("## "):
            break
        body.append(ln)
    return "\n".join(body)


def _shell_fences(section):
    """Returns the section's ``sh``/``bash`` fenced blocks, one string each, in order."""
    return re.findall(r"^```(?:sh|bash)\n(.*?)^```", section, re.S | re.M)


def _flat(text):
    """Collapses line wraps, so a pinned phrase survives re-wrapping the paragraph."""
    return " ".join(text.split())


def test_setup_gating_section_leads_with_the_commit_gate():
    """R1: the commit gate is the primary recipe; `--staged` comes second.

    The first fenced block holding a `mitos` line holds `mitos hook-install`, and it
    sits before the block holding the staged gate command.
    """
    blocks = _shell_fences(_extract_h2_section(_setup_text(), _GATING_HEADING))
    with_mitos = [i for i, b in enumerate(blocks)
                  if any(ln.startswith("mitos ") for ln in b.splitlines())]
    staged = [i for i, b in enumerate(blocks) if GATE_COMMAND in b]
    assert with_mitos and staged, "the gating section lost its recipes"
    assert "mitos hook-install" in blocks[with_mitos[0]]
    assert with_mitos[0] < staged[0]


def test_every_gating_recipe_parses_and_names_its_project():
    """R2: each `mitos` line in the gating section's fences runs, with a selector.

    Parsed by the real parser; the selector is read the way `_selector_from_args`
    reads it (either `-p` position or the positional), never through one `dest`.
    `comments=True` because the cron lines carry trailing shell comments.
    """
    parser = cli._build_parser()
    checked = []
    for block in _shell_fences(_extract_h2_section(_setup_text(), _GATING_HEADING)):
        for line in block.splitlines():
            if not line.startswith("mitos "):
                continue
            try:
                args = parser.parse_args(shlex.split(line, comments=True)[1:])
            except SystemExit:
                pytest.fail(f"SETUP.md's gating recipe does not parse: {line!r}")
            assert cli._selector_from_args(args), f"recipe names no project: {line!r}"
            checked.append(args.command)
    assert len(checked) >= 3, checked          # non-vacuity
    assert "hook-install" in checked


def test_staged_recipe_equals_the_tested_recipe_byte_for_byte():
    """R3: the whole `--staged` block is `RECIPE` without its shebang.

    `test_setup_recipe_is_in_lockstep` pins three pieces; the comment lines between
    them are only seen here.
    """
    blocks = [b for b in _shell_fences(_extract_h2_section(_setup_text(), _GATING_HEADING))
              if GATE_COMMAND in b]
    assert len(blocks) == 1
    assert blocks[0] == RECIPE.split("\n", 1)[1]


def test_commit_gate_prose_states_its_reach_first_block_and_keyless_row():
    """R4: reach, bypass, the first-commit block and the keyless row, as 3h proved them.

    The does-not-run sentence must name all four verbs git runs no `pre-commit` for.
    """
    gate = _flat(_extract_section(_setup_text(), _GATE_HEADING))
    assert "on `git commit`" in gate
    assert "`git commit --amend`" in gate
    assert "`git commit --no-verify` skips it" in gate
    does_not = re.search(r"It does not run on [^.]*\.", gate)
    assert does_not, "the does-not-run sentence is gone"
    for verb in ("merge", "revert", "cherry-pick", "rebase"):
        assert verb in does_not.group(0), verb
    assert "the first commit after install is blocked once" in gate
    assert "An empty corpus passes" in gate
    assert "installed and stays inactive" in gate
    assert "any attempt at that check opens it" in gate
    assert "A `--scope` or `--staged` run is a different check and does not" in gate


def test_upgrade_sentence_names_the_error_and_no_version():
    """R5: an old build is recognised by its argparse error, never by a version number.

    The merge mints the version, so a number written here would be a guess.
    """
    gate = _flat(_extract_section(_setup_text(), _GATE_HEADING))
    assert re.search(r"`hook-install` with an argparse `invalid choice` error", gate)
    assert "Prerequisites → Updating" in gate
    section = _extract_h2_section(_setup_text(), _GATING_HEADING)
    assert not re.search(r"\b\d+\.\d+\.\d+\b", section), "a version number in the gating section"


def test_p_table_moves_the_hook_to_the_absolute_path_row():
    """R6: `-p .` no longer names a git hook, and the installed hook has its row.

    The hook `hook-install` writes bakes an absolute path; a hand-written hook is
    still covered by the `-p .` row's "workspace root when the command runs".
    Presence and absence together.
    """
    (table,) = _tables(_extract_section(_setup_text(), "Naming the project on every call"))
    dot = [row for row in table if row[0].startswith("`-p .`")]
    assert len(dot) == 1
    assert "hook" not in dot[0][1]
    assert any("hook-install" in cell for row in table for cell in row)


def test_keys_section_names_what_the_commit_gate_needs():
    """R7: the gate is armed by the Anthropic key alone; the sweep advice stays."""
    block = _flat(_extract_section(
        _setup_text(), "Keys, Qdrant, and the secretless-CI consequence"))
    assert "The commit gate" in block
    assert "`ANTHROPIC_API_KEY`" in block
    assert "no Gemini key and no Qdrant" in block
    assert "corpus sweep" in block


def test_readme_audit_bullet_names_the_commit_gate_and_nothing_wider():
    """R8: README (the PyPI long description) says "on `git commit`" and no more.

    `--staged` must never again read as the pre-commit step. The `mcp-name` line is
    pinned by `test_packaging_meta` too; this is the row a README editor meets.
    """
    readme = pathlib.Path(_REPO_ROOT, "README.md").read_text(encoding="utf-8")
    assert readme.splitlines()[1] == "<!-- mcp-name: io.github.dovahkiin-v/mitos -->"
    (bullet,) = [ln for ln in readme.splitlines() if "It audits itself" in ln]
    assert "on `git commit`" in bullet
    assert "`mitos hook-install -p .`" in bullet
    assert "as a pre-commit" not in bullet


# --------------------------------------------------------------------------- #
# Keyless half — the recipe's shell logic through a real git commit
# --------------------------------------------------------------------------- #

def test_divergence_guard_fires_before_check(tmp_path):
    """A staged/worktree divergence fires the guard with the verbatim message, before check.

    The guard is pure git plumbing — no keys. Stage decisions.md, edit it without
    re-staging, then commit: the pre-commit hook detects the divergence and aborts
    with the verbatim message BEFORE `mitos check --staged` ever runs (the fake
    mitos's marker stays absent — the check never fired).
    """
    repo = _init_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    marker = tmp_path / "mitos_called"
    _fake_mitos(bin_dir, marker, 0)

    (repo / "decisions.md").write_text("### a\n**Decided:** v1.\n", encoding="utf-8")
    _run(["git", "add", "decisions.md"], repo)
    _run(["git", "commit", "--no-verify", "-m", "base"], repo)
    _install_hook(repo)

    # Diverge: worktree != index for decisions.md; stage an UNRELATED change to commit.
    (repo / "decisions.md").write_text("### a\n**Decided:** v2.\n", encoding="utf-8")
    (repo / "other.txt").write_text("x\n", encoding="utf-8")
    _run(["git", "add", "other.txt"], repo)
    r = _run(["git", "commit", "-m", "should-abort"], repo, env=_env_with_path(bin_dir))

    assert r.returncode != 0, "the commit should have been aborted by the guard"
    assert DIVERGENCE_MESSAGE in (r.stdout + r.stderr), (
        f"guard did not emit the verbatim message. output:\n{r.stdout}\n{r.stderr}"
    )
    assert not marker.exists(), (
        "the fake `mitos` was invoked — the guard did NOT precede the check"
    )


@pytest.mark.parametrize("exit_code,should_commit", [(0, True), (1, False)])
def test_recipe_wires_check_exit_to_commit(tmp_path, exit_code, should_commit):
    """With no divergence, the recipe wires `mitos check --staged`'s exit to the commit.

    A clean staged/worktree state passes the guard and runs `mitos check --staged`
    (the fake): exit 0 → the commit succeeds; exit 1 → the commit is blocked. This
    is the recipe's contract — the gate's verdict is the commit's verdict —
    independent of the (separately-tested) engine.
    """
    repo = _init_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    marker = tmp_path / "mitos_called"
    _fake_mitos(bin_dir, marker, exit_code)

    (repo / "decisions.md").write_text("### a\n**Decided:** v1.\n", encoding="utf-8")
    _run(["git", "add", "decisions.md"], repo)
    _run(["git", "commit", "--no-verify", "-m", "base"], repo)
    _install_hook(repo)

    # No divergence (decisions.md unchanged); stage an unrelated change to commit.
    (repo / "other.txt").write_text("x\n", encoding="utf-8")
    _run(["git", "add", "other.txt"], repo)
    r = _run(["git", "commit", "-m", "gated"], repo, env=_env_with_path(bin_dir))

    assert marker.exists(), "the guard passed but `mitos check --staged` never ran"
    if should_commit:
        assert r.returncode == 0, (
            f"exit-0 check should let the commit through. output:\n{r.stdout}\n{r.stderr}"
        )
    else:
        assert r.returncode != 0, "exit-1 check should block the commit"


# --------------------------------------------------------------------------- #
# Live half — the real binary, one full end-to-end
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not HAS_LIVE_KEYS,
    reason="GEMINI_API_KEY and ANTHROPIC_API_KEY both required — the live hook half "
    "drives the real `mitos check --staged` (embeddings + SONNET judge).",
)
def test_live_hook_blocks_bad_buffer_passes_clean(tmp_path):
    """The shipped `mitos check --staged`, wired into a real pre-commit hook, gates a commit.

    Seed an indexed hard-delete decision; a pending soft-delete buffer (the undeclared
    contradiction) → the hook blocks the commit (fail-closed: exit 1 on the finding, or
    exit 2 if a live substrate degrades — either blocks). Then a clean (no-pending)
    buffer → the no-pending short-circuit exits 0 and the commit passes. `mitos` is the
    real venv binary on PATH.
    """
    try:
        if requests.get(f"{QDRANT_URL.rstrip('/')}/collections", timeout=5).status_code != 200:
            pytest.skip(f"Qdrant unreachable at {QDRANT_URL} — environmental.")
    except requests.RequestException:
        pytest.skip(f"Qdrant unreachable at {QDRANT_URL} — environmental.")

    # A workspace dir named so its derived collection (mitos-<basename>-<path digest>)
    # is swept by conftest's mitos-tmp-* backstop even if teardown misses — the prefix
    # survives the digest suffix, which is why the `tmp` in the basename still buys the
    # reclaim. Ask the derivation rather than hand-building the name: the shape is
    # contract, and a second spelling of it here would silently address a collection
    # nothing writes to.
    ws = tmp_path / f"tmp-golden-hook-{uuid.uuid4().hex[:8]}"
    ws.mkdir()
    collection = default_collection_name(str(ws))
    env = _env_with_path(_VENV_BIN)

    try:
        _run(["git", "init", "-q"], ws)
        _run(["git", "config", "user.email", "hook@test"], ws)
        _run(["git", "config", "user.name", "hook test"], ws)
        _run(["git", "config", "commit.gpgsign", "false"], ws)

        init = _run(["mitos", "init"], ws, env=env)
        assert init.returncode == 0, f"mitos init failed:\n{init.stdout}\n{init.stderr}"

        # Seed + index the hard-delete decision (non-interactive).
        (ws / "decisions.md").write_text(_HARD_DELETE, encoding="utf-8")
        # `init` above stays bare — it is selector-exempt and a supplied selector is
        # REFUSED, so `-p .` there would exit non-zero on "`init` takes no project
        # selector". Only the workspace-targeting verbs gain one.
        sync = _run(["mitos", "sync", "--yes", "-p", "."], ws, env=env)
        if sync.returncode != 0:
            # A quota/service outage during seeding is environmental, not a defect.
            pytest.skip(
                f"seed `mitos sync --yes` failed (likely quota/service):\n"
                f"{sync.stdout}\n{sync.stderr}"
            )

        # Baseline commit (hook not installed yet), then install the hook.
        _run(["git", "add", "-A"], ws)
        _run(["git", "commit", "--no-verify", "-m", "base"], ws)
        _install_hook(ws)

        # BAD buffer: a pending undeclared contradiction; staged (no divergence).
        with open(ws / "decisions.md", "a", encoding="utf-8") as f:
            f.write(_SOFT_DELETE)
        _run(["git", "add", "decisions.md"], ws)
        bad = _run(["git", "commit", "-m", "bad-buffer"], ws, env=env)
        assert bad.returncode != 0, (
            f"the hook should BLOCK a pending undeclared contradiction. "
            f"output:\n{bad.stdout}\n{bad.stderr}"
        )

        # CLEAN buffer: revert to the baseline (no pending); stage an unrelated change.
        _run(["git", "reset", "--hard", "HEAD"], ws)
        (ws / "other.txt").write_text("x\n", encoding="utf-8")
        _run(["git", "add", "other.txt"], ws)
        clean = _run(["git", "commit", "-m", "clean-buffer"], ws, env=env)
        assert clean.returncode == 0, (
            f"the hook should PASS a no-pending buffer (short-circuit exit 0). "
            f"output:\n{clean.stdout}\n{clean.stderr}"
        )
    finally:
        try:
            requests.delete(
                f"{QDRANT_URL.rstrip('/')}/collections/{collection}", timeout=5
            )
        except requests.RequestException:
            pass
