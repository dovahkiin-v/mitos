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

from mitos import registry, routing
from mitos.errors import RegistryError
from mitos.cli import (
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


def _make_workspace(root) -> str:
    """Builds the minimal valid workspace shape and returns its canonical path.

    The shipped validity triple and nothing more: ``.mitos/`` holding a
    ``config.toml``, plus ``decisions.md``. A half-workspace is not a workspace,
    and building only the first two parts is the fixture mistake this phase's plan
    warns is made from habit.
    """
    os.makedirs(os.path.join(str(root), ".mitos"), exist_ok=True)
    with open(os.path.join(str(root), ".mitos", "config.toml"), "w") as f:
        f.write("# a mitos workspace\n")
    with open(os.path.join(str(root), "decisions.md"), "w") as f:
        f.write("# Decisions\n")
    return os.path.realpath(str(root))


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
# Group 6 — the transitional tripwires. INVERT these at 5a, never delete them.
# ---------------------------------------------------------------------------

def test_a_selectorless_call_still_resolves_the_cwd_workspace(
    tmp_path, monkeypatch
) -> None:
    """TRANSITIONAL (phase 5a inverts this row).

    Construction is not migration: this phase makes the selector *sayable*, not
    *required*, so a bare `mitos list` from inside a workspace behaves exactly as
    it did before. Phase 5a removes the working-directory fallback, at which point
    this must become an assertion that the same invocation renders the missing
    anatomy and exits 1. The row exists so that flip is a decision someone makes
    rather than an omission someone notices.
    """
    root = _make_workspace(tmp_path / "ws")
    monkeypatch.chdir(root)

    with patch("mitos.cli.cmd_list", return_value=0) as mock:
        code = _run(["list"])

    assert code == 0
    assert mock.call_args.args[0].workspace_dir == root


def test_set_key_without_a_selector_still_writes_the_cwd_env(
    tmp_path, monkeypatch
) -> None:
    """TRANSITIONAL (phase 5a inverts this row).

    Pins ``cmd_set_key``'s ``workspace_dir=None`` default. 5a deletes the default
    and the two ``os.getcwd()`` calls behind it, making the argument required —
    and this row then asserts the selectorless call is refused instead.
    """
    cwd = tmp_path / "here"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    _run(["set-key", "a-value"])

    assert "GEMINI_API_KEY=a-value" in (
        open(os.path.join(str(cwd), ".env"), encoding="utf-8").read())
