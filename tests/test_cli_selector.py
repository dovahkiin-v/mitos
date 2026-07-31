"""Tests for the CLI project selector (``mitos -p NAME VERB`` / ``mitos VERB -p NAME``).

Phase 3b gives ``routing.resolve_project`` its first consumer. This module covers
the grammar (both sides of the verb, two destinations, every subparser), the
boundary (one resolution site, exempt-before-resolve, chdir-before-absolutize),
the ``status``/``agent-block`` positional upgrade with its non-workspace
carve-out, and the teaching anatomy every targeting failure renders.

**No mocks of external services, no async, no live tier.** Every row drives a
real registry file — conftest's autouse ``hermetic_mitos_env`` redirects
``XDG_CONFIG_HOME`` per test, which is what keeps these writes out of the
developer's own ``~/.config/mitos/registry.toml`` — and real workspace
directories under ``tmp_path``. What *is* patched is the verb handler, so each
assertion is about routing rather than about what the verb does; the two
exceptions are the carve-out rows, which drive the real ``cmd_status`` /
``cmd_agent_block`` because the report text is the thing under test.

Paths are asserted against ``os.path.realpath`` of the fixture path, never the
raw fixture path (the suite convention: the canonical spelling is
``registry.canonicalize``, and on this machine's temp root the two agree, so a
raw comparison would be a machine-dependent pin).

Exact string literals appear where they are the contract — the six discriminator
values live in ``mitos.errors`` and are imported, while the *wording* is this
surface's and is asserted on by the part it must carry (an example, a pointer, a
did-you-mean), never by a whole-sentence match.
"""

import os
import sys
from unittest.mock import patch

import pytest

from conftest import make_workspace
from mitos import config, registry, routing
from mitos.errors import RegistryError
from mitos.cli import (
    _EXEMPT_VERB_NOTES,
    _POSITIONAL_SELECTOR_VERBS,
    _SELECTOR_EXEMPT_VERBS,
    _WORKSPACE_OPTIONAL_VERBS,
    _build_parser,
    main,
)


# --- helpers ---------------------------------------------------------------

def _write_registry(text: str) -> str:
    """Hand-writes the registry file (the hand-editable states we must tolerate)."""
    path = registry.registry_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _register(**entries) -> None:
    """Writes a registry from ``name → path`` pairs, in the order given."""
    _write_registry(
        "".join(f'"{name}" = "{path}"\n' for name, path in entries.items())
    )


def _register_pairs(pairs) -> None:
    """Writes a registry from an iterable of ``(name, path)`` pairs.

    The keyword form cannot express a name that is not a Python identifier — and
    the case-variant and over-the-bound registries below need exactly those.
    """
    _write_registry("".join(f'"{name}" = "{path}"\n' for name, path in pairs))


#: The shared workspace builder, lifted to `conftest` at 5a — the flip gave it a
#: dozen more consumers and thirteen private re-spellings would be thirteen chances
#: to write the half-workspace one. Kept under the module-local name so the rows
#: below read unchanged.
_make_workspace = make_workspace


def _run(argv):
    """Drives ``cli.main()`` through argv, returning the exit code (0 if it returns).

    ``SystemExit`` is swallowed rather than asserted on per row: several verbs
    exit through ``sys.exit(handler(...))`` and several do not, and the difference
    is not what any row here is about.
    """
    with patch.object(sys, "argv", ["mitos"] + list(argv)):
        try:
            main()
        except SystemExit as exc:
            return exc.code
    return 0


def _subparsers(parser):
    """Returns the ``name → parser`` map of registered verbs."""
    return parser._subparsers._group_actions[0].choices


def _did_you_mean(stderr: str):
    """Parses the suggestions out of the did-you-mean line, or None if absent.

    Asserted on as a **parsed list** rather than by substring, because the
    ``Registered projects:`` line below it already carries every registered name:
    a bare ``assert "MiToS" in err`` is green whatever the did-you-mean line
    actually says. Measured, not reasoned — the truncation injection this exists
    to catch passed against the substring spelling.
    """
    for line in stderr.splitlines():
        if "Did you mean" in line:
            return line.split(":", 1)[1].strip().split(", ")
    return None


# ---------------------------------------------------------------------------
# Group 1 — the grammar. Asserted on `_build_parser()` directly: verifying an
# option's spelling by running the verb costs a real workspace read.
# ---------------------------------------------------------------------------

def test_every_registered_subparser_carries_the_project_option() -> None:
    """Computed over ``subparsers.choices``, so a verb added later cannot ship without it.

    The point of the loop in ``_build_parser`` rather than a per-verb
    ``add_argument``: a hand-listed set here would pass while the next verb added
    silently lost the post-verb half.
    """
    missing = [
        name for name, sub in _subparsers(_build_parser()).items()
        if not any("--project" in (a.option_strings or []) for a in sub._actions)
    ]
    assert missing == []


def test_the_aliased_verbs_are_one_parser_object_registered_once() -> None:
    """27 names over 22 objects — the reason the registration loop must dedupe by id().

    A second ``add_argument`` on the same object raises ``ArgumentError:
    conflicting option strings``, so this row is also what pins that the five
    aliases (``query_decisions``, ``surface_decisions``, ``list_decisions``,
    ``list_scopes``, ``record_decision``) are free rather than forgotten.
    """
    choices = _subparsers(_build_parser())
    assert len(choices) == 27
    assert len({id(sub) for sub in choices.values()}) == 22
    for alias, canonical in (("query_decisions", "query"),
                             ("surface_decisions", "surface"),
                             ("list_decisions", "list"),
                             ("list_scopes", "scopes"),
                             ("record_decision", "record")):
        assert choices[alias] is choices[canonical]


def test_the_top_level_parser_carries_the_project_option() -> None:
    """The pre-verb half — the one a shell alias can prefix and `-C` sits beside."""
    parser = _build_parser()
    assert any("--project" in (a.option_strings or []) for a in parser._actions)


def test_the_selector_is_discoverable_in_both_help_surfaces() -> None:
    """`-p` shows in `mitos --help` AND in a verb's `--help`.

    The only thing that makes the post-verb half discoverable to a human: a
    caller who reads `mitos record --help` must see the option that verb accepts.
    """
    parser = _build_parser()
    assert "--project" in parser.format_help()
    assert "--project" in _subparsers(parser)["record"].format_help()


@pytest.mark.parametrize("argv,pre,post", [
    (["-p", "A", "list"], "A", None),
    (["list", "-p", "B"], None, "B"),
    (["-p", "A", "list", "-p", "B"], "A", "B"),
])
def test_the_two_spellings_land_in_distinct_destinations(argv, pre, post) -> None:
    """The argparse trap, made unconstructible.

    With ONE shared ``dest`` the first row parses to ``None``: since 3.7
    ``_SubParsersAction`` parses into a fresh namespace and copies every key onto
    the parent, so the sub-namespace's default overwrites what the top-level
    parser stored — the caller's selector, silently discarded. ``SUPPRESS`` fixes
    that row and makes the third indistinguishable from the second. Two dests have
    no shared slot to clobber, and the third row stays visible enough to refuse.
    """
    args = _build_parser().parse_args(argv)
    assert args.project_pre == pre
    assert args.project_post == post


def test_the_selector_is_never_argparse_required() -> None:
    """`required=True` would exit 2 from argparse before any anatomy could render.

    The missing-selector check is post-parse, at the boundary — which is also what
    keeps every zero-arg invocation working until 5a.
    """
    parser = _build_parser()
    for actions in [parser._actions] + [s._actions for s in _subparsers(parser).values()]:
        for action in actions:
            if "--project" in (action.option_strings or []):
                assert action.required is False


# ---------------------------------------------------------------------------
# Group 2 — both spellings actually retarget the workspace.
# ---------------------------------------------------------------------------

# The require-list verbs that take a `config`, with the five aliases named
# explicitly beside their canonical twins. A canonical-names-only list would
# leave five invocable verbs unpinned on exactly the agent-facing path.
_CONFIG_VERBS = [
    (["list"], "cmd_list"),
    (["list_decisions"], "cmd_list"),
    (["scopes"], "cmd_scopes"),
    (["list_scopes"], "cmd_scopes"),
    (["query", "a claim"], "cmd_query"),
    (["query_decisions", "a claim"], "cmd_query"),
    (["surface", "a topic"], "cmd_surface"),
    (["surface_decisions", "a topic"], "cmd_surface"),
    (["record", "an axiom", "--rejected", "r", "--slug", "s"], "cmd_record"),
    (["record_decision", "an axiom", "--rejected", "r", "--slug", "s"], "cmd_record"),
    (["sync"], "cmd_sync"),
    (["show", "a-slug"], "cmd_show"),
    (["open-questions"], "cmd_open_questions"),
    (["render"], "cmd_render"),
    (["reconcile"], "cmd_reconcile"),
]

_CONFIG_VERB_IDS = [row[0][0] for row in _CONFIG_VERBS]


@pytest.mark.parametrize("tail,handler", _CONFIG_VERBS, ids=_CONFIG_VERB_IDS)
@pytest.mark.parametrize("position", ["pre", "post"])
def test_both_spellings_retarget_the_verbs_workspace(
    tmp_path, monkeypatch, capsys, tail, handler, position
) -> None:
    """`mitos -p N VERB` and `mitos VERB -p N` both act on N's workspace.

    One resolution site at the boundary feeding the single ``config`` every verb
    already receives, so ~20 verbs retarget with no per-verb edit.

    The post-verb half additionally asserts argparse's usage banner was never
    reached: that is the one bare framework rejection no boundary handler can
    intercept (exit 2 before mitos code runs, carrying none of the anatomy), and
    it is why the option is registered on every subparser rather than top-level
    only.
    """
    root = _make_workspace(tmp_path / "real")
    _register(proj=root)
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    argv = (["-p", "proj"] + tail) if position == "pre" else (tail + ["-p", "proj"])
    with patch(f"mitos.cli.{handler}", return_value=0) as mock:
        _run(argv)

    assert mock.called, f"{handler} was never reached: {capsys.readouterr().err}"
    assert mock.call_args.args[0].workspace_dir == root
    assert "unrecognized arguments" not in capsys.readouterr().err


def test_a_registered_name_beats_a_same_named_directory_in_cwd(
    tmp_path, monkeypatch
) -> None:
    """The trap the "absolutize anything not already absolute" build cannot pass.

    The decoy is a **valid** workspace on purpose: under that build the call
    resolves *successfully* to the wrong project, which is the failure worth
    fencing — an invalid decoy would red with a targeting error and the row would
    pass for the wrong reason.
    """
    real = _make_workspace(tmp_path / "real_workspace")
    _register(mitos=real)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    decoy = _make_workspace(cwd / "mitos")
    monkeypatch.chdir(cwd)

    with patch("mitos.cli.cmd_list", return_value=0) as mock:
        _run(["-p", "mitos", "list"])

    assert mock.call_args.args[0].workspace_dir == real
    assert mock.call_args.args[0].workspace_dir != decoy


def test_a_relative_selector_resolves_against_the_post_chdir_cwd(
    tmp_path, monkeypatch
) -> None:
    """`mitos -C /a -p ./b list` acts on /a/b — the chdir runs before absolutization.

    Ordering is contract and it is the one thing an implementer can silently
    invert: canonicalize before the chdir and a relative selector means something
    else entirely, with every absolute-path row still green.
    """
    a = tmp_path / "a"
    b = _make_workspace(a / "b")
    monkeypatch.chdir(tmp_path)

    with patch("mitos.cli.cmd_list", return_value=0) as mock:
        _run(["-C", str(a), "-p", "./b", "list"])

    assert mock.call_args.args[0].workspace_dir == b


def test_a_bare_dot_selector_resolves_rather_than_being_refused(
    tmp_path, monkeypatch, capsys
) -> None:
    """`-p .` is an explicitly-typed relative path, so the boundary absolutizes it.

    What this row can prove is that `.` **resolves** — without the boundary's
    absolutization the resolver refuses it as a relative path and the handler is
    never reached. It structurally cannot distinguish the working-directory
    fallback, because `.` and the fallback name the same directory by definition;
    the sibling row above is what pins that a selector is honoured at all.
    """
    root = _make_workspace(tmp_path / "here")
    monkeypatch.chdir(root)

    with patch("mitos.cli.cmd_list", return_value=0) as mock:
        code = _run(["-p", ".", "list"])

    assert mock.called, capsys.readouterr().err
    assert code == 0
    assert mock.call_args.args[0].workspace_dir == root


def test_an_unregistered_absolute_path_is_a_correct_steady_state(
    tmp_path, monkeypatch
) -> None:
    """The escape hatch: a valid workspace nothing has registered still resolves.

    A cloned project that never ran ``init`` is a supported posture, not a
    degraded one — no warning fires and nothing nags.
    """
    root = _make_workspace(tmp_path / "clone")
    _register()  # an empty registry file
    monkeypatch.chdir(tmp_path)

    with patch("mitos.cli.cmd_list", return_value=0) as mock:
        code = _run(["-p", root, "list"])

    assert mock.call_args.args[0].workspace_dir == root
    assert code == 0


# ---------------------------------------------------------------------------
# Group 3 — naming the target twice, and the verbs that take no target at all.
# ---------------------------------------------------------------------------

def test_naming_the_project_twice_with_two_flags_is_refused(
    tmp_path, monkeypatch, capsys
) -> None:
    """Both spellings are named back, so the caller can see which to drop."""
    monkeypatch.chdir(tmp_path)
    with patch("mitos.cli.cmd_list", return_value=0) as mock:
        code = _run(["-p", "A", "list", "-p", "B"])

    err = capsys.readouterr().err
    assert not mock.called
    assert code == 1
    assert "A" in err and "B" in err
    assert "Traceback" not in err


def test_naming_the_project_twice_is_refused_even_when_the_values_match(
    tmp_path, monkeypatch, capsys
) -> None:
    """No equality exception: a rule with one is a rule nobody can predict."""
    root = _make_workspace(tmp_path / "real")
    _register(proj=root)
    monkeypatch.chdir(tmp_path)
    with patch("mitos.cli.cmd_list", return_value=0) as mock:
        code = _run(["-p", "proj", "list", "-p", "proj"])

    assert not mock.called
    assert code == 1
    assert "twice" in capsys.readouterr().err


def test_a_positional_plus_a_flag_is_refused_on_status(
    tmp_path, monkeypatch, capsys
) -> None:
    """`mitos -p A status /ws` names the target twice in two different grammars."""
    root = _make_workspace(tmp_path / "ws")
    _register(proj=root)
    monkeypatch.chdir(tmp_path)
    with patch("mitos.cli.cmd_status", return_value=0) as mock:
        code = _run(["-p", "proj", "status", root])

    err = capsys.readouterr().err
    assert not mock.called
    assert code == 1
    assert "proj" in err and root in err
    assert "Traceback" not in err


def test_import_is_not_a_positional_selector_verb() -> None:
    """The discriminator is what a positional DENOTES, never that it is spelled `path`.

    ``import``'s positional is a source markdown file. A future verb registering a
    positional called ``path`` must not be swept into the selector set either, so
    the set is asserted exactly rather than by absence alone.
    """
    assert "import" not in _POSITIONAL_SELECTOR_VERBS
    assert _POSITIONAL_SELECTOR_VERBS == frozenset({"status", "agent-block"})


def test_the_workspace_optional_carve_out_is_two_verbs_wide() -> None:
    """Only the two verbs that answer *about* a directory rather than acting on it."""
    assert _WORKSPACE_OPTIONAL_VERBS == frozenset({"status", "agent-block"})


def test_status_is_not_an_exempt_verb() -> None:
    """`status <project>` is a supported targeting form, so `-p X status` is its twin."""
    assert "status" not in _SELECTOR_EXEMPT_VERBS
    assert set(_SELECTOR_EXEMPT_VERBS) == {"init", "serve", "projects"}


def test_every_exempt_verb_owns_its_own_recovery_sentence() -> None:
    """The two exempt maps are hand-written, and only one of them is fenced.

    ``_SELECTOR_EXEMPT_VERBS`` (verb → reason) is what the boundary raises on;
    ``_EXEMPT_VERB_NOTES`` (verb → this surface's wording) is what the renderer
    prints, and its lookup carries a ``.get`` fallback. So a verb added to the
    first without the second does not fail — it silently renders the generic
    *"it targets no single workspace"* line, which is honest and useless: the
    whole point of the class is that each exempt verb is meaningless in a
    *different* way. The class of defect is this vision's recurring one — a
    hand-written set proves only what someone remembered to put in it — so the
    two maps are compared rather than each asserted alone.

    ``set-key`` is in the notes and not in the reason map on purpose: its
    membership is conditional (only ``--global`` makes it global), decided at the
    boundary rather than by a table.
    """
    assert set(_EXEMPT_VERB_NOTES) == set(_SELECTOR_EXEMPT_VERBS) | {"set-key"}
    for verb, note in _EXEMPT_VERB_NOTES.items():
        assert f"`{verb}" in note, f"{verb}'s note must name the verb it refuses"


@pytest.mark.parametrize("verb", ["init", "serve", "projects"])
def test_a_selector_on_an_exempt_verb_names_the_verb(
    tmp_path, monkeypatch, capsys, verb
) -> None:
    """A loud explanatory refusal — never silently ignored, never "unknown project".

    The selector is a name nothing has registered, which is what makes the row
    also the exempt-before-resolve ordering pin: resolving first would answer the
    wrong question and teach a recovery (register the name) that leaves the call
    just as malformed.
    """
    monkeypatch.chdir(tmp_path)
    handler = {"init": "cmd_init", "serve": "cmd_serve", "projects": "cmd_projects"}[verb]
    with patch(f"mitos.cli.{handler}", return_value=0) as mock:
        code = _run(["-p", "nosuch", verb])

    err = capsys.readouterr().err
    assert not mock.called
    assert code == 1
    assert verb in err
    assert "unknown project" not in err
    assert "Traceback" not in err


def test_set_key_is_exempt_only_under_the_global_flag(
    tmp_path, monkeypatch, capsys
) -> None:
    """`--global` writes the machine-wide .env, so a project selector contradicts it."""
    monkeypatch.chdir(tmp_path)
    with patch("mitos.cli.cmd_set_key") as mock:
        code = _run(["-p", "nosuch", "set-key", "v", "--global"])

    err = capsys.readouterr().err
    assert not mock.called
    assert code == 1
    assert "set-key" in err and "--global" in err


def test_a_selector_retargets_the_project_env_that_set_key_writes(
    tmp_path, monkeypatch
) -> None:
    """`mitos -p X set-key <val>` writes X's .env and X's .gitignore.

    The alternative was a credential landing in the launch directory while the
    caller named another project — the discard hazard on the worst surface to be
    quiet about.
    """
    root = _make_workspace(tmp_path / "target")
    _register(proj=root)
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    _run(["-p", "proj", "set-key", "secret-value"])

    assert "GEMINI_API_KEY=secret-value" in (
        open(os.path.join(root, ".env"), encoding="utf-8").read())
    assert ".env" in open(os.path.join(root, ".gitignore"), encoding="utf-8").read()
    assert not os.path.exists(cwd / ".env")


def test_import_accepts_a_selector_and_a_file_argument_together(
    tmp_path, monkeypatch
) -> None:
    """The one verb where positional-*plus*-selector is mandatory.

    The config retargets; the file argument is handed on **raw**, because file
    args are cwd/`-C`-rooted and the selector is not.
    """
    root = _make_workspace(tmp_path / "corpus")
    _register(proj=root)
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    with patch("mitos.cli.cmd_import") as mock:
        _run(["-p", "proj", "import", "./legacy.md"])

    assert mock.call_args.args[0].workspace_dir == root
    assert mock.call_args.args[1] == "./legacy.md"


def test_imports_file_argument_opens_against_the_working_directory(
    tmp_path, monkeypatch
) -> None:
    """Observed, not assumed: the file is found in cwd, not in the target workspace.

    ``import_from_file`` answers a missing file with a print and an exit-0 return,
    so a row that only patches ``cmd_import`` never opens anything and cannot show
    where the path resolves. This one records what the importer would actually
    reach for.
    """
    root = _make_workspace(tmp_path / "corpus")
    _register(proj=root)
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    (cwd / "legacy.md").write_text("# legacy\n", encoding="utf-8")
    monkeypatch.chdir(cwd)

    seen = {}

    class _Importer:
        def __init__(self, config):
            seen["workspace"] = config.workspace_dir

        def import_from_file(self, filepath, use_llm_extract=False):
            seen["exists"] = os.path.exists(filepath)
            seen["resolved"] = os.path.abspath(filepath)

    with patch("mitos.cli.MitosProseImporter", _Importer):
        _run(["-p", "proj", "import", "./legacy.md"])

    assert seen["workspace"] == root
    assert seen["exists"] is True
    assert seen["resolved"] == os.path.realpath(str(cwd / "legacy.md"))


# ---------------------------------------------------------------------------
# Group 4 — the status/agent-block positional upgrade and its carve-out.
# ---------------------------------------------------------------------------

def test_status_resolves_a_registered_name_through_the_positional(
    tmp_path, monkeypatch
) -> None:
    """The upgrade: the positional is a name-or-path, not only a path."""
    root = _make_workspace(tmp_path / "ws")
    _register(proj=root)
    monkeypatch.chdir(tmp_path)

    with patch("mitos.cli.cmd_status", return_value=0) as mock:
        _run(["status", "proj"])

    assert mock.call_args.args[0] == root


def test_the_flag_and_positional_forms_of_status_agree(tmp_path, monkeypatch) -> None:
    """`mitos -p X status` behaves identically to `mitos status X`."""
    root = _make_workspace(tmp_path / "ws")
    _register(proj=root)
    monkeypatch.chdir(tmp_path)

    with patch("mitos.cli.cmd_status", return_value=0) as mock:
        _run(["-p", "proj", "status"])

    assert mock.call_args.args[0] == root


def test_the_deprecation_warning_names_the_workspace_the_flag_selected(
    tmp_path, monkeypatch, capsys
) -> None:
    """The `_warn_target` collapse: one config, so the warning and the report agree.

    The flag spelling leaves ``args.path`` empty, which under the old two-reader
    shape meant the warning silently read the working directory instead of the
    workspace the caller named. Nothing reads ``args.path`` any more, so the state
    does not exist.
    """
    root = _make_workspace(tmp_path / "ws")
    with open(os.path.join(root, ".mitos", "config.toml"), "w") as f:
        f.write('rotation_mode = "prune"\n')
    _register(proj=root)
    clean_cwd = tmp_path / "clean"
    clean_cwd.mkdir()
    monkeypatch.chdir(clean_cwd)

    with patch("mitos.cli.cmd_status", return_value=0):
        _run(["-p", "proj", "status"])

    assert capsys.readouterr().err.count("rotation_mode = 'prune' is deprecated") == 1


def test_status_on_a_bare_directory_still_prints_the_not_set_up_report(
    tmp_path, monkeypatch, capsys
) -> None:
    """The carve-out: `status` keeps giving the verdict its whole job is to give.

    Routed naively through the resolver, this same command answers "no Mitos
    workspace at …" and exits 1 — the same exit code with the guidance gone, and
    SETUP.md's agent setup loop is built on that guidance. So the report text is
    asserted, not only the code: an exit-code-only row passes under exactly the
    regression this carve-out exists to prevent.
    """
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:9")
    bare = tmp_path / "bare"
    bare.mkdir()
    monkeypatch.chdir(tmp_path)

    code = _run(["status", str(bare)])

    out = capsys.readouterr().out
    assert code == 1
    assert "NOT SET UP" in out
    assert "Next steps:" in out
    assert "mitos init" in out
    assert "no Mitos workspace at" not in out


def test_agent_block_check_on_a_bare_directory_still_reports(
    tmp_path, monkeypatch, capsys
) -> None:
    """The carve-out's twin. `agent-block`'s plain form is workspace-independent.

    Refusing a pre-`init` repo here would block a legitimate use — the block is
    exactly what a project pastes *before* it is set up.
    """
    bare = tmp_path / "bare"
    bare.mkdir()
    monkeypatch.chdir(tmp_path)

    code = _run(["agent-block", str(bare), "--check"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Agent-file mitos note" in out
    assert str(bare) in out


def test_status_with_an_unknown_name_gets_the_unknown_anatomy(
    tmp_path, monkeypatch, capsys
) -> None:
    """The carve-out is path-form only. A name is a claim about the registry.

    Widened to the name form, this would answer a typo with a NOT SET UP report
    about a directory the caller never named.
    """
    root = _make_workspace(tmp_path / "ws")
    _register(mitos=root)
    monkeypatch.chdir(tmp_path)

    code = _run(["status", "mitoss"])

    err = capsys.readouterr().err
    assert code == 1
    assert "unknown project 'mitoss'" in err
    assert _did_you_mean(err) == ["mitos"]
    assert "NOT SET UP" not in err


def test_status_with_a_stale_registration_names_the_recorded_path(
    tmp_path, monkeypatch, capsys
) -> None:
    """`registered_unreachable` survives the carve-out too, and it should.

    "`X` is registered at `/…`, which no longer holds a workspace" is a better
    answer than a NOT SET UP report about a path the caller never typed — and it
    is the class's whole reason for existing.
    """
    gone = str(tmp_path / "gone")
    _register(proj=gone)
    monkeypatch.chdir(tmp_path)

    code = _run(["status", "proj"])

    err = capsys.readouterr().err
    assert code == 1
    assert "registered" in err
    assert gone in err
    assert "NOT SET UP" not in err


def test_a_non_workspace_path_is_still_refused_for_other_verbs(
    tmp_path, monkeypatch, capsys
) -> None:
    """`mitos -p /some/bare/dir list` still refuses — there is no corpus there."""
    bare = tmp_path / "bare"
    bare.mkdir()
    monkeypatch.chdir(tmp_path)

    with patch("mitos.cli.cmd_list", return_value=0) as mock:
        code = _run(["-p", str(bare), "list"])

    assert not mock.called
    assert code == 1
    assert "no Mitos workspace" in capsys.readouterr().err


def test_an_empty_status_positional_is_a_supplied_selector(
    tmp_path, monkeypatch, capsys
) -> None:
    """`mitos status ""` renders the missing anatomy rather than falling back to cwd.

    A deliberate behaviour change on a non-zero-arg input, recorded rather than
    discovered: the supplied-ness gate is ``is not None`` everywhere, so an empty
    string is a selector carrying no target — not the absence of one. The phase's
    containment criterion is scoped to *zero-arg* invocations, which this is not.
    """
    monkeypatch.chdir(tmp_path)
    with patch("mitos.cli.cmd_status", return_value=0) as mock:
        code = _run(["status", ""])

    assert not mock.called
    assert code == 1
    assert "no project selector" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Group 5 — the renderer. All six discriminators, all six reachable from the CLI.
# ---------------------------------------------------------------------------

def test_the_missing_class_renders_the_full_anatomy(
    tmp_path, monkeypatch, capsys
) -> None:
    """`-p ""` — what's wrong, a concrete example, the discovery pointer, the cwd hint.

    Reachable in *this* phase precisely because the gate is ``is not None``: an
    ``or``-spelled one would swallow the empty selector into the cwd fallback and
    leave this branch dead code until 5a.
    """
    root = _make_workspace(tmp_path / "here")
    _register(here=root)
    monkeypatch.chdir(root)

    code = _run(["-p", "", "list"])

    err = capsys.readouterr().err
    assert code == 1
    assert "no project selector" in err
    assert "--project" in err                       # the concrete example
    assert "mitos projects" in err                  # the discovery pointer
    assert "Registered projects: here" in err       # the vocabulary
    assert "Traceback" not in err


def test_the_unknown_name_class_offers_a_did_you_mean(
    tmp_path, monkeypatch, capsys
) -> None:
    """A typo gets the closest registered name, the vocabulary, and the pointer."""
    root = _make_workspace(tmp_path / "ws")
    _register(mitos=root, cartolina=root)
    monkeypatch.chdir(tmp_path)

    code = _run(["-p", "mitoss", "list"])

    err = capsys.readouterr().err
    assert code == 1
    assert "unknown project 'mitoss'" in err
    assert _did_you_mean(err) == ["mitos"]      # not `cartolina`, and not both
    assert "Registered projects: mitos, cartolina" in err
    assert "mitos projects" in err
    assert "Traceback" not in err


def test_the_relative_path_class_names_both_valid_forms_and_no_did_you_mean(
    tmp_path, monkeypatch, capsys
) -> None:
    """`-p '~/x'` lands here, and that is the honest answer to a quoting mistake.

    ``os.path.abspath("~/x")`` returns ``<cwd>/~/x``, which **is** absolute, so an
    unguarded absolutize would send a nonsense path to the resolver and answer
    "no workspace at …" naming a directory nobody meant. With the guard the
    selector reaches the resolver path-shaped and non-absolute, and the message
    names the actual rule.

    No did-you-mean: the caller claimed a path, and suggesting a name for it says
    the wrong thing is the spelling of the name.
    """
    root = _make_workspace(tmp_path / "ws")
    _register(mitos=root)
    monkeypatch.chdir(tmp_path)

    code = _run(["-p", "~/x", "list"])

    err = capsys.readouterr().err
    assert code == 1
    # Both valid forms, named — not one or the other.
    assert "registered name" in err
    assert "absolute path" in err
    assert "--project mitos" in err
    assert _did_you_mean(err) is None
    # The unguarded absolutize would have named `<cwd>/~/x` as the subject.
    assert os.path.join(str(tmp_path), "~") not in err
    assert "Traceback" not in err


def test_the_path_not_a_workspace_class_names_the_path(
    tmp_path, monkeypatch, capsys
) -> None:
    """The directory is the subject, so the message names it and what it lacks."""
    bare = tmp_path / "empty"
    bare.mkdir()
    monkeypatch.chdir(tmp_path)

    code = _run(["-p", str(bare), "list"])

    err = capsys.readouterr().err
    assert code == 1
    assert os.path.realpath(str(bare)) in err
    assert "decisions.md" in err
    assert "Did you mean" not in err
    assert "Traceback" not in err


def test_the_registered_unreachable_class_teaches_a_repoint(
    tmp_path, monkeypatch, capsys
) -> None:
    """Not the unknown anatomy: the project IS registered, and the fix is a repoint.

    ``path`` carries the registry's **recorded** string rather than the
    canonicalized probe — the value a human edits in their own file.
    """
    gone = str(tmp_path / "vanished")
    _register(proj=gone)
    monkeypatch.chdir(tmp_path)

    code = _run(["-p", "proj", "list"])

    err = capsys.readouterr().err
    assert code == 1
    assert "'proj'" in err and "registered" in err
    assert gone in err
    # A recovery, named concretely — both halves, not either.
    assert "Repoint" in err
    assert "--force" in err
    assert registry.registry_path() in err
    assert _did_you_mean(err) is None
    assert "unknown project" not in err
    assert "Traceback" not in err


def test_the_exempt_verb_class_names_the_verb_and_why(
    tmp_path, monkeypatch, capsys
) -> None:
    """The sixth class, and the one that carries no registry vocabulary at all."""
    monkeypatch.chdir(tmp_path)
    code = _run(["-p", "anything", "init"])

    err = capsys.readouterr().err
    assert code == 1
    assert "`init`" in err
    assert "registration" in err
    assert "Registered projects" not in err
    assert "Traceback" not in err


def test_a_targeting_failure_under_check_exits_two(tmp_path, monkeypatch, capsys) -> None:
    """The shipped exit mapping is inherited, not re-invented: 2 under `check`.

    For CI, "could not run" is one routing class with the verb's own exit-2
    refusals; no other verb's contract moves.
    """
    monkeypatch.chdir(tmp_path)
    with patch("mitos.cli.cmd_check", return_value=0) as mock:
        code = _run(["-p", "nosuch", "check"])

    assert not mock.called
    assert code == 2
    assert "Traceback" not in capsys.readouterr().err


def test_a_targeting_failure_still_speaks_stderr_under_json(
    tmp_path, monkeypatch, capsys
) -> None:
    """No `--json` targeting envelope: the shipped boundary answers on stderr regardless.

    A JSON envelope for one error class would be a new asymmetry — the verb-level
    JSON dead-ends live inside their handlers and are untouched.
    """
    monkeypatch.chdir(tmp_path)
    code = _run(["-p", "nosuch", "list", "--json"])

    captured = capsys.readouterr()
    assert code == 1
    assert "unknown project" in captured.err
    assert captured.out == ""


def test_an_empty_registry_gets_its_own_variant(tmp_path, monkeypatch, capsys) -> None:
    """"No projects registered yet" — never a blank `Registered projects:` line.

    The CLI is the surface allowed to prescribe `mitos init`; the MCP renderer
    never may, because an agent handed a state-creating shell command in an error
    body will run it.
    """
    _register()
    monkeypatch.chdir(tmp_path)

    _run(["-p", "nosuch", "list"])

    err = capsys.readouterr().err
    assert "No projects are registered yet" in err
    assert "mitos init" in err
    assert "Registered projects:" not in err


def test_the_registered_name_list_collapses_above_the_bound(
    tmp_path, monkeypatch, capsys
) -> None:
    """Above the bound with no close match: the count and the pointer, and NO names.

    An empty ``names`` with ``collapsed=True`` is the honest answer — registry
    order is document order, which carries no relevance ranking, so an arbitrary
    slice of it is worse than a count. Spelling the renderer
    ``bounded.names or [...]`` would undo exactly this.
    """
    root = _make_workspace(tmp_path / "ws")
    names = [f"project-alpha-{i}" for i in range(routing.REGISTERED_NAMES_BOUND + 1)]
    _register_pairs((name, root) for name in names)
    monkeypatch.chdir(tmp_path)

    _run(["-p", "zzzzzzzz", "list"])

    err = capsys.readouterr().err
    assert str(len(names)) in err
    assert "mitos projects" in err
    assert not any(name in err for name in names)


def test_the_renderer_shows_every_did_you_mean_even_past_the_max(
    tmp_path, monkeypatch, capsys
) -> None:
    """`close_project_matches` can return MORE than `PROJECT_DIDYOUMEAN_MAX`.

    It takes up to three *folded* matches and expands each to every original that
    folds onto it, so a registry hand-edited to hold several case variants of one
    name yields all of them. A renderer truncating to the max would silently drop
    the very distinction the caller needs to see.
    """
    root = _make_workspace(tmp_path / "ws")
    variants = ["mitos", "Mitos", "MITOS", "MiToS"]
    _register_pairs((name, root) for name in variants)
    monkeypatch.chdir(tmp_path)

    _run(["-p", "mitoss", "list"])

    assert len(variants) > routing.PROJECT_DIDYOUMEAN_MAX
    assert _did_you_mean(capsys.readouterr().err) == variants


def test_the_cwd_hint_names_the_project_the_caller_is_standing_in(
    tmp_path, monkeypatch, capsys
) -> None:
    """The hint fires from a *subdirectory*, which is the common real posture."""
    root = _make_workspace(tmp_path / "mitos-pub")
    _register_pairs([("mitos-pub", root)])
    inner = tmp_path / "mitos-pub" / "src"
    inner.mkdir()
    monkeypatch.chdir(inner)

    _run(["-p", "nosuchproject", "list"])

    err = capsys.readouterr().err
    assert "sits inside registered project 'mitos-pub'" in err
    assert "--project mitos-pub" in err


def test_the_cwd_hint_is_absent_from_a_directory_that_merely_extends_a_registration(
    tmp_path, monkeypatch, capsys
) -> None:
    """Containment is by whole path segment, and this is the shape that proves it.

    The obvious fixture (``mitos`` and ``mitos-pub`` both registered) is green
    against a bare ``startswith`` because longest-registered-path-wins rescues it.
    What it cannot rescue is a directory whose name merely *extends* a
    registration: ``…/mitos-pub-sandbox`` is not inside ``…/mitos-pub``, and a
    prefix test says confidently that it is — on the one line whose whole job is a
    helpful guess.
    """
    root = _make_workspace(tmp_path / "mitos-pub")
    _register_pairs([("mitos-pub", root)])
    sandbox = tmp_path / "mitos-pub-sandbox" / "src"
    sandbox.mkdir(parents=True)
    monkeypatch.chdir(sandbox)

    _run(["-p", "nosuchproject", "list"])

    err = capsys.readouterr().err
    assert "sits inside registered project" not in err


@pytest.mark.parametrize("selector,hinted", [
    ("", True),                 # missing
    ("nosuch", True),           # unknown_name
    ("~/x", False),             # relative_path
    ("/nonexistent/bare", False),   # path_not_a_workspace
], ids=["missing", "unknown_name", "relative_path", "path_not_a_workspace"])
def test_the_cwd_hint_renders_only_where_a_name_was_at_stake(
    tmp_path, monkeypatch, capsys, selector, hinted
) -> None:
    """The hint is a guess about what the caller MEANT, so it needs a name at stake.

    On a path-form failure the caller named something else entirely and a cwd
    nudge is noise; on the two name-shaped classes it is the recovery. Every row
    here runs from *inside* a registered project, so the hint is available in all
    four — which is what makes the two ``False`` rows a real assertion rather than
    an absence with no cause.
    """
    root = _make_workspace(tmp_path / "inside")
    _register(inside=root)
    monkeypatch.chdir(root)

    _run(["-p", selector, "list"])

    err = capsys.readouterr().err
    assert ("sits inside registered project" in err) is hinted


@pytest.mark.parametrize("seam", ["registry_race", "cwd_read"])
def test_a_failing_cwd_hint_never_replaces_the_diagnosis_with_a_traceback(
    tmp_path, monkeypatch, capsys, seam
) -> None:
    """The hint is decoration; the diagnosis is the answer. It may not take it down.

    The renderer runs INSIDE the boundary's ``except`` arm, where a raise escapes
    ``main()`` entirely — a sibling ``except MitosError`` cannot catch what its
    neighbour handler throws — so an unguarded hint turns the one error class this
    phase exists to render well into a traceback. Both causes are reachable only
    here: a targeting failure never constructs a ``MitosConfig``, so the hint can be
    the run's first ``os.getcwd()``, and the registry is re-read after the raise
    site already read it.

    Both seams need care, and the first one taught its own lesson: patching
    ``registry.load`` to raise **outright** proves nothing, because
    ``resolve_project`` reads the registry before anything else, so the run never
    reaches an ``unknown_name`` error at all and renders the (correct) calm
    ``RegistryError`` line instead. The window this guards is a genuine *race* —
    readable at the raise site, unreadable one call later — so the fake succeeds
    once and fails once. The second drives the cwd fault through
    ``nearest_registered_ancestor`` rather than by patching ``os.getcwd``, which is
    process-global and would break pytest around the call.
    """
    root = _make_workspace(tmp_path / "ws")
    _register(proj=root)
    monkeypatch.chdir(tmp_path)

    if seam == "registry_race":
        patcher = patch("mitos.cli.registry.load", side_effect=[
            {"proj": root},                                  # resolution reads it
            RegistryError("registry became unreadable"),      # the hint re-reads it
        ])
    else:
        patcher = patch("mitos.cli.routing.nearest_registered_ancestor",
                        side_effect=OSError("cwd was deleted"))

    with patcher:
        code = _run(["-p", "nosuch", "list"])

    err = capsys.readouterr().err
    assert code == 1
    assert "unknown project 'nosuch'" in err        # the diagnosis survived
    assert "Traceback" not in err
    assert "sits inside registered project" not in err   # only the hint was lost


def test_a_malformed_registry_is_not_dressed_in_the_targeting_anatomy(
    tmp_path, monkeypatch, capsys
) -> None:
    """`RegistryError` propagates unwrapped and renders through the calm boundary.

    There is no registered vocabulary to enumerate when the file holding it cannot
    be read, so the anatomy would be a lie — and the caller needs the parse detail,
    not a did-you-mean.
    """
    path = _write_registry('"broken" = \n')
    monkeypatch.chdir(tmp_path)

    code = _run(["-p", "anything", "list"])

    err = capsys.readouterr().err
    assert code == 1
    assert path in err
    assert "Did you mean" not in err
    assert "Registered projects" not in err
    assert "Traceback" not in err


# ---------------------------------------------------------------------------
# Group 6 — the two transitional tripwires, INVERTED at 5a (entry-007).
#
# They were written to make the flip a decision someone makes rather than an
# omission someone notices. Both were kept and turned around; neither was deleted,
# because the assertion is the only statement that the fallback is gone rather than
# merely untested.
# ---------------------------------------------------------------------------

def test_a_selectorless_call_is_refused_from_inside_a_workspace(
    tmp_path, monkeypatch
) -> None:
    """INVERTED at 5a: standing in the workspace is no longer an answer.

    Before the flip this exact invocation resolved the working directory and ran.
    Now it renders the §4.5 teaching anatomy on stderr and exits 1 — and the mock
    proves the point that matters: no verb ran at all, so there is no write to
    unwind. Standing in the *right* directory is deliberately not a special case;
    a default that is usually right is exactly the kind that fails silently and
    late.
    """
    root = _make_workspace(tmp_path / "ws")
    monkeypatch.chdir(root)

    with patch("mitos.cli.cmd_list", return_value=0) as mock:
        code = _run(["list"])

    assert code == 1
    mock.assert_not_called()


def test_the_refusal_teaches_the_recovery_and_names_where_you_are(
    tmp_path, monkeypatch, capsys
) -> None:
    """The anatomy on the failure the flip makes routine — including the cwd hint.

    The hint is the one working-directory read 5a **keeps** (entry-005): it is the
    recovery line on the most common post-flip failure, and it sits in the error
    renderer rather than on the resolution path precisely so removing the fallback
    could not take it with it. A sweep reading its inherited `os.getcwd()` list as
    "delete them all" leaves every other row here green.
    """
    root = _make_workspace(tmp_path / "ws")
    monkeypatch.chdir(root)
    _register(here=root)

    code = _run(["list"])

    err = capsys.readouterr().err
    assert code == 1
    assert "--project" in err                    # the flag that recovers it
    assert "mitos projects" in err                # where the vocabulary lives
    # The hint asserted by its DISTINCTIVE phrase, never by the bare name: the
    # `Registered projects:` line one row up already carries every registered name,
    # so `assert "here" in err` is green whatever the hint line actually says (this
    # module's own did-you-mean lesson, applied one line over).
    assert "working directory sits inside" in err
    assert "here" in err
    assert "Traceback" not in err


def test_set_key_without_a_selector_writes_no_env_anywhere(
    tmp_path, monkeypatch
) -> None:
    """INVERTED at 5a — and the sharpest of the three, because the payload is a key.

    ``cmd_set_key``'s ``workspace_dir`` is now required with no default, so the
    project form is unconstructible without a named project rather than silently
    cwd-rooted. Criterion 3: **no file anywhere** — not the launch directory's
    ``.env``, not the global one.
    """
    cwd = tmp_path / "here"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    code = _run(["set-key", "a-value"])

    assert code == 1
    assert not os.path.exists(os.path.join(str(cwd), ".env"))
    assert not os.path.exists(config.global_env_path())


def test_set_key_global_is_exempt_and_still_writes_the_machine_wide_env(
    tmp_path, monkeypatch
) -> None:
    """The one form that must SURVIVE the flip, and the proof `_exempt_reason` is right.

    ``set-key --global`` names no project because it *has* none, so it is exempt in
    that form. Fault injection: make the exempt predicate unconditional on
    ``set-key`` and the bare form above goes green for the wrong reason while this
    row reds.
    """
    cwd = tmp_path / "here"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    code = _run(["set-key", "--global", "a-value"])

    assert code == 0
    with open(config.global_env_path(), encoding="utf-8") as f:
        assert "GEMINI_API_KEY=a-value" in f.read()
    assert not os.path.exists(os.path.join(str(cwd), ".env"))


# ---------------------------------------------------------------------------
# Group 7 — the flip itself (5a). Every require-list verb, the three classes,
# and the fail-closed direction the removed fallback used to backstop.
# ---------------------------------------------------------------------------

#: The minimum extra argv each verb needs to get PAST argparse — required
#: positionals and required/mutually-exclusive flags. Argparse exits 2 before any
#: mitos code runs, so a verb missing one would look like a `check`-shaped exit-2
#: refusal while never reaching the targeting boundary at all.
_MINIMAL_ARGS = {
    "capture": ["some prose"],
    "record": ["an axiom", "--slug", "a-slug"],
    "record_decision": ["an axiom", "--slug", "a-slug"],
    "restore-source": ["--all-graph-only"],
    "import": ["notes.md"],
    "query": ["a claim"],
    "query_decisions": ["a claim"],
    "set-key": ["a-value"],
    "show": ["a-slug"],
    "surface": ["a claim"],
    "surface_decisions": ["a claim"],
}

#: The five MCP-name aliases, pinned by name beside the computed list below. A
#: canonical-names-only parametrization would leave five invocable verbs unpinned on
#: exactly the agent-facing path `skill.md` teaches.
_ALIASES = ("query_decisions", "surface_decisions", "list_decisions",
            "list_scopes", "record_decision")


def _require_list():
    """The §4.4 require-list, computed off the live parser rather than hand-listed."""
    return sorted(set(_subparsers(_build_parser()))
                  - set(_SELECTOR_EXEMPT_VERBS) - {"status"})


def test_the_require_list_is_the_parser_minus_the_two_other_classes() -> None:
    """23 verbs, measured — and the five aliases are among them, named.

    The three classes of §3 partition the parser exactly: exempt (a selector is
    refused), optional (`status`, whose absence routes elsewhere), and required
    (everything else). A verb added later lands in `required` by construction, which
    is the safe default — but this row makes the count a decision rather than a
    drift.
    """
    verbs = _require_list()
    assert len(verbs) == 23
    for alias in _ALIASES:
        assert alias in verbs
    assert set(_subparsers(_build_parser())) == (
        set(verbs) | set(_SELECTOR_EXEMPT_VERBS) | {"status"})


@pytest.mark.parametrize("verb", _require_list())
def test_every_require_list_verb_refuses_a_selectorless_call(
    verb, tmp_path, monkeypatch, capsys
) -> None:
    """I1's CLI half: no verb on the require-list resolves the working directory.

    Run from **inside a valid workspace**, deliberately — that is the invocation
    that used to work, and the one whose silent success is the vision's §1 hazard.
    Nothing about standing in the right place is a special case any more.

    `check` keeps its own exit-2 mapping (CI reads "could not run" as one routing
    class with the verb's own refusals); every other verb exits 1. No third exit
    code is invented for the flip.
    """
    root = _make_workspace(tmp_path / "ws")
    monkeypatch.chdir(root)
    # One registration, so the anatomy renders its full vocabulary line rather than
    # the empty-registry variant — the state a real caller meets.
    _register(theproject=root)

    code = _run([verb] + _MINIMAL_ARGS.get(verb, []))

    err = capsys.readouterr().err
    assert code == (2 if verb == "check" else 1), err
    assert "Traceback" not in err
    assert "--project" in err          # the recovery, in one round trip
    assert "theproject" in err          # the registered vocabulary


def test_agent_blocks_bare_form_is_on_the_require_list_too(
    tmp_path, monkeypatch, capsys
) -> None:
    """Surfaced rather than buried, because it reads like a defect.

    `mitos agent-block`'s plain output is a workspace-*independent* block, so
    requiring a selector for it looks arbitrary. It is the vision's call, not an
    oversight: `--check` genuinely scans a workspace's agent files, and a per-form
    carve-out would be exactly the §4.8 grace the vision refuses — on the verb whose
    output gets committed. The recovery is `mitos agent-block -p .`.
    """
    root = _make_workspace(tmp_path / "ws")
    monkeypatch.chdir(root)

    assert _run(["agent-block"]) == 1
    assert "--project" in capsys.readouterr().err

    assert _run(["agent-block", "-p", root]) == 0
    out = capsys.readouterr().out
    assert "mitos-agent-guide" in out and "/mitos-agent-guide" in out


def test_import_keeps_its_positional_as_a_FILE_and_takes_the_selector_by_flag(
    tmp_path, monkeypatch
) -> None:
    """`import`'s positional is a source markdown file, never a workspace.

    So it is deliberately absent from `_POSITIONAL_SELECTOR_VERBS`: post-flip
    `mitos import notes.md` hard-fails and the fix is `mitos -p <name> import
    notes.md` — the file argument unchanged, still opening against cwd. A migrator
    who reads "positional on a require-list verb" as "that's the selector" either
    deletes the path or passes the workspace twice.
    """
    root = _make_workspace(tmp_path / "ws")
    launch = tmp_path / "launch"
    launch.mkdir()
    (launch / "notes.md").write_text("# notes\n", encoding="utf-8")
    monkeypatch.chdir(launch)

    assert _run(["import", "notes.md"]) == 1

    with patch("mitos.cli.cmd_import", return_value=None) as mock:
        assert _run(["-p", root, "import", "notes.md"]) == 0
    assert mock.call_args.args[0].workspace_dir == root
    assert mock.call_args.args[1] == "notes.md"


# --- the fail-closed direction the removed fallback used to backstop --------

def test_a_write_verb_pointed_at_a_non_workspace_writes_nothing(
    tmp_path, monkeypatch, capsys
) -> None:
    """The adversarial row this layer-removal owes.

    Removing the working-directory fallback makes `resolve_project` the SOLE gate on
    which workspace any CLI verb touches. Until now the fallback backstopped a bad
    selector — a call that failed to resolve still had somewhere to land. After this
    phase nothing does, so the fail-closed direction needs its own assertion: a
    selector naming a real, writable directory that is simply not a workspace must
    refuse **before** anything is created in it.
    """
    not_a_workspace = tmp_path / "plain-dir"
    not_a_workspace.mkdir()
    monkeypatch.chdir(_make_workspace(tmp_path / "ws"))
    before = sorted(os.listdir(str(not_a_workspace)))

    code = _run(["-p", str(not_a_workspace), "record", "an axiom",
                 "--rejected", "the alternative", "--slug", "s"])

    assert code == 1
    assert "no Mitos workspace" in capsys.readouterr().err
    assert sorted(os.listdir(str(not_a_workspace))) == before == []


def test_restore_source_refuses_before_opening_any_graph(
    tmp_path, monkeypatch, capsys
) -> None:
    """Criterion 4: the refusal precedes the store, not merely the write.

    `restore-source --all-graph-only` is the widest-blast-radius read in the tree
    (it walks every graph-only node and rewrites `decisions.md`). The row patches
    `GraphStore` so a construction anywhere on the path is visible: the selectorless
    call must not reach it at all.
    """
    monkeypatch.chdir(_make_workspace(tmp_path / "ws"))

    with patch("mitos.cli.GraphStore") as store:
        code = _run(["restore-source", "--all-graph-only"])

    assert code == 1
    store.assert_not_called()
    assert "--project" in capsys.readouterr().err


# --- the help surface's own truth (T20b) -----------------------------------

def test_the_help_surface_no_longer_teaches_the_working_directory_model() -> None:
    """Criterion 8, asserted against `_build_parser()` — no workspace needed.

    Six texts taught a model this phase deletes, and they sit far apart: two `-p`
    help strings that are the same sentence in two places (an audit that fixes the
    pre-verb one and stops leaves the post-verb spelling teaching the dead model),
    the `-C` string that claimed to retarget graph + collection + `.env`, two
    positional helps promising a current-directory default, and `_EPILOG` — a module
    constant five thousand lines from the parser that renders it.
    """
    parser = _build_parser()
    top = parser.format_help()

    assert "instead of the working directory" not in top
    assert "-C /path/to/repo list" not in top          # the dead headline example
    for claim in ("graph, collection, .env/keys", "default: current directory"):
        assert claim not in top

    subs = _subparsers(parser)
    assert "instead of the working directory" not in subs["record"].format_help()
    assert "default: current directory" not in subs["status"].format_help()
    assert "default: current directory" not in subs["agent-block"].format_help()

    # And every worked example in the epilog names its project, because every verb
    # it shows now requires one. `mitos status` is the single exception and is
    # shown as such.
    epilog = top.split("Examples:", 1)[1]
    for line in epilog.splitlines():
        stripped = line.strip()
        if not stripped.startswith("mitos ") or stripped == "mitos status":
            continue
        assert stripped.startswith("mitos -p "), f"selectorless example: {stripped!r}"
