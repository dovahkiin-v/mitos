"""Tests for the routing resolver (``mitos/routing.py``).

The resolver turns a selector — a registered name or an absolute path — into a
validated, canonical workspace root, and raises one typed error carrying
structured data for every way that can fail. This module is the leaf's own suite:
shape discrimination and its totality, both admitted forms, the single
canonicalization, each failure class with its required data, the hint predicate,
the did-you-mean policy and its bound, and the drift net on the duplicated
validity triple.

No services and no mocks: this is a filesystem leaf with no external service and
no async code, so every row drives a real registry file (the autouse
``hermetic_mitos_env`` fixture redirects ``XDG_CONFIG_HOME`` per test) and a real
workspace directory under ``tmp_path``. Paths are asserted against
``os.path.realpath`` of the fixture path, never the raw fixture path — and where a
row is *about* canonicalization it routes through a real ``os.symlink``, because
this machine's temp root is not a symlink and a bare ``tmp_path`` comparison would
pass just as well under ``abspath``.

Exact string literals appear here on purpose and only where they are the
contract: the six discriminator values, the three exempt-reason keys and the two
``via`` values are what the CLI and MCP boundary renderers bind to, so they are
hand-written rather than imported into their own assertions. Everything else
asserts a relation.
"""

import ast
import json
import os
import subprocess
import sys

import pytest

from mitos import cli, registry, routing
from mitos.config import MitosConfig
from mitos.errors import (
    EXEMPT_CREATES_REGISTRATION,
    EXEMPT_EXPLICITLY_GLOBAL,
    EXEMPT_NO_WORKSPACE,
    EXEMPT_REASONS,
    TARGET_EXEMPT_VERB,
    TARGET_MISSING,
    TARGET_PATH_NOT_A_WORKSPACE,
    TARGET_REGISTERED_UNREACHABLE,
    TARGET_RELATIVE_PATH,
    TARGET_UNKNOWN_NAME,
    TARGETING_DISCRIMINATORS,
    MitosError,
    ProjectTargetingError,
    RegistryError,
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


def _make_workspace(root) -> str:
    """Builds the minimal valid workspace shape and returns its canonical path.

    The shipped validity triple and nothing more: ``.mitos/`` holding a
    ``config.toml``, plus ``decisions.md``. Deliberately no graph — a workspace is
    valid without one (the cloned-but-unbuilt state the escape hatch exists for).
    """
    os.makedirs(os.path.join(str(root), ".mitos"), exist_ok=True)
    with open(os.path.join(str(root), ".mitos", "config.toml"), "w") as f:
        f.write("# a mitos workspace\n")
    with open(os.path.join(str(root), "decisions.md"), "w") as f:
        f.write("# Decisions\n")
    return os.path.realpath(str(root))


# --- group 1: shape discrimination -----------------------------------------

# The corpus D1's complement property is asserted over. `my.project` stays a
# name (dotted names are registrable — the registry quotes its keys);
# `src/proj` is path-shaped despite carrying no leading dot, because a
# separator anywhere is what a name may never hold.
_NAME_SHAPED = ["mitos", "my.project", "MITOS", "ąžuolas", "mitos-pub"]
_PATH_SHAPED = ["src/mitos-pub", "./x", "..", "~", "~/x", "/abs", "trail/"]


@pytest.mark.parametrize("selector", _NAME_SHAPED)
def test_a_name_shaped_selector_is_not_path_shaped_and_is_a_legal_name(selector):
    """Name-shaped selectors reach the registry untouched, and are registrable.

    The second half is the load-bearing one: shape-first discrimination is only
    *total* if every string the shape test calls a name is a string the registry
    can actually hold. Asserted against ``validate_name`` rather than argued.
    """
    assert routing.is_path_shaped(selector) is False
    registry.validate_name(selector)  # raises if it is not a legal name


@pytest.mark.parametrize("selector", _PATH_SHAPED)
def test_a_path_shaped_selector_can_never_be_a_registered_name(selector):
    """Every path-shaped form is refused by name legality — the complement half.

    Without this, a form in neither class (``src/proj`` under a literal
    leading-or-trailing reading) lands on the unknown-name anatomy and answers
    "did you mean 'mitos'?" for something obviously a path.
    """
    assert routing.is_path_shaped(selector) is True
    with pytest.raises(RegistryError):
        registry.validate_name(selector)


def test_the_two_residue_strings_are_named_rather_than_left_implicit():
    """Exactly two strings are neither a path nor a legal name, and both are handled.

    The empty string (which becomes the missing class, not an unknown project
    named ``''``) and a control-character string with no separator (which is
    simply a registry miss). Pinned so a future widening of either predicate has
    to notice it is moving the residue.
    """
    for residue in ("", "a\tb"):
        assert routing.is_path_shaped(residue) is False
        with pytest.raises(RegistryError):
            registry.validate_name(residue)


# --- group 2: the two admitted forms ---------------------------------------

def test_a_registered_name_resolves_to_its_recorded_path_not_to_a_cwd_sibling(
    tmp_path, monkeypatch
):
    """The trap row: ``mitos`` must reach the registry, never ``<cwd>/mitos``.

    Driven from a cwd that *does* contain a same-named subdirectory, because the
    obvious implementation — "absolutize anything not already absolute" — resolves
    happily under any other fixture and kills the vision's core feature silently.
    """
    recorded = _make_workspace(tmp_path / "elsewhere" / "mitos")
    decoy = tmp_path / "cwd" / "mitos"
    _make_workspace(decoy)
    _register(mitos=recorded)
    monkeypatch.chdir(str(tmp_path / "cwd"))

    resolved = routing.resolve_project("mitos")

    assert resolved.root == recorded
    assert resolved.root != os.path.realpath(str(decoy))
    assert resolved.name == "mitos"
    assert resolved.via == "name"


def test_an_absolute_path_resolves_untouched_and_reports_the_path_form(tmp_path):
    """The escape hatch: an absolute path resolves whether or not it is registered."""
    root = _make_workspace(tmp_path / "proj")

    resolved = routing.resolve_project(root)

    assert resolved.root == root
    assert resolved.via == "path"


def test_a_path_form_hit_on_a_registered_path_fills_the_name(tmp_path):
    """The echo needs the name, and the resolver already has the registry loaded."""
    root = _make_workspace(tmp_path / "proj")
    _register(alpha=root)

    resolved = routing.resolve_project(root)

    assert resolved.name == "alpha"
    assert resolved.via == "path"


def test_an_unregistered_absolute_path_is_a_correct_steady_state(tmp_path, capsys):
    """A clone that never runs ``init`` resolves with no warning, nag or diagnostic.

    This is the dominant agent path and a *correct* state, not a degraded one, so
    the success path must stay silent — a warning here would fire constantly on
    something that is working exactly as designed.
    """
    root = _make_workspace(tmp_path / "clone")
    _register(other=_make_workspace(tmp_path / "other"))

    resolved = routing.resolve_project(root)

    assert resolved.name is None
    assert resolved.via == "path"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# --- group 3: canonicalization ---------------------------------------------

def test_every_spelling_of_one_workspace_yields_one_identical_root(tmp_path):
    """Trailing slash, ``..``-bearing and symlinked routes agree on one root.

    The symlink is the row that bites: this machine's temp root is not a symlink,
    so a trailing-slash or ``..`` comparison alone passes just as well under
    ``abspath`` and proves nothing about the canonical spelling.
    """
    root = _make_workspace(tmp_path / "proj")
    link = tmp_path / "link-to-proj"
    os.symlink(root, str(link))

    spellings = [
        root,
        root + os.sep,
        os.path.join(root, "..", "proj"),
        str(link),
    ]
    assert os.path.abspath(str(link)) != os.path.realpath(str(link))  # the row bites
    assert {routing.resolve_project(s).root for s in spellings} == {root}


def test_the_resolved_root_and_the_config_derivation_agree_exactly(tmp_path):
    """``MitosConfig(root).workspace_dir == root`` — one root, one collection.

    ``MitosConfig`` absolutizes and does not resolve symlinks, so a root that were
    *not* already canonical would land on a second identity here. Feeding it the
    resolver's output is what keeps the two systems on one collection.
    """
    root = _make_workspace(tmp_path / "proj")
    link = tmp_path / "link-to-proj"
    os.symlink(root, str(link))

    resolved = routing.resolve_project(str(link))

    assert MitosConfig(resolved.root).workspace_dir == resolved.root


# --- group 4: the six classes ----------------------------------------------

def _raises(selector) -> ProjectTargetingError:
    with pytest.raises(ProjectTargetingError) as exc:
        routing.resolve_project(selector)
    return exc.value


@pytest.mark.parametrize("selector", [None, ""])
def test_an_absent_selector_is_the_missing_class(selector):
    """``None`` and ``""`` both carry no target — not an unknown project named ``''``."""
    err = _raises(selector)
    assert err.discriminator == "missing"


def test_an_unregistered_name_is_the_unknown_class_carrying_its_selector(tmp_path):
    _register(mitos=_make_workspace(tmp_path / "mitos"))

    err = _raises("mitoss")

    assert err.discriminator == "unknown_name"
    assert err.selector == "mitoss"
    assert err.close_matches == ["mitos"]


@pytest.mark.parametrize("selector", ["./mitos-pub", "../x", "src/proj", "~/x"])
def test_a_path_shaped_relative_selector_is_its_own_class(selector):
    """Answering these with the unknown-name anatomy invites another relative retry."""
    err = _raises(selector)
    assert err.discriminator == "relative_path"
    assert err.selector == selector
    # Did-you-mean over registered names is noise for a path input.
    assert err.close_matches == []


def test_an_absolute_path_with_no_workspace_is_its_own_class(tmp_path):
    """The escape hatch's commonest failure — a clone whose ``.mitos/`` never shipped."""
    bare = tmp_path / "not-a-workspace"
    bare.mkdir()

    err = _raises(str(bare))

    assert err.discriminator == "path_not_a_workspace"
    assert err.selector == str(bare)
    assert err.path == os.path.realpath(str(bare))


def test_an_absolute_path_with_no_workspace_stays_that_class_even_when_registered(
    tmp_path,
):
    """The unreachable class fires on the **name** form only.

    A path-form selector resolves *through* nothing, so a path that fails
    validation is "no workspace there" whether or not the registry happens to
    name it. Keeping the class to one meaning is what makes its recovery — repoint
    the registration — the true diagnosis every time it fires.
    """
    vanished = tmp_path / "gone"
    vanished.mkdir()
    _register(gone=os.path.realpath(str(vanished)))

    assert _raises(str(vanished)).discriminator == "path_not_a_workspace"


def test_a_registered_name_whose_workspace_vanished_is_the_unreachable_class(tmp_path):
    """Carries the name and the **recorded** path — the string a repoint edits."""
    recorded = os.path.realpath(str(tmp_path / "moved"))
    os.makedirs(recorded)  # a directory, but not a workspace any more
    _register(moved=recorded)

    err = _raises("moved")

    assert err.discriminator == "registered_unreachable"
    assert err.selector == "moved"
    assert err.name == "moved"
    assert err.path == recorded


def test_the_exempt_verb_class_is_built_by_its_factory(tmp_path):
    """2a ships the factory; the only call to it is the boundary's, in a later phase."""
    err = routing.exempt_verb_error("init", EXEMPT_CREATES_REGISTRATION)

    assert isinstance(err, ProjectTargetingError)
    assert err.discriminator == "exempt_verb"
    assert err.verb == "init"
    assert err.exempt_reason == "creates_registration"


def test_an_unknown_exempt_reason_is_a_programming_fault_not_a_user_fault():
    """Prose where a key belongs raises ``ValueError`` — deliberately not a MitosError."""
    with pytest.raises(ValueError):
        routing.exempt_verb_error("init", "because it creates one")


def test_no_failure_class_ever_carries_another_discriminator(tmp_path):
    """The fall-through the whole contract forbids, asserted across every class.

    Each fixture drives exactly one class; a resolver that folded two together
    (the four-discriminator reading) would show up here as one discriminator
    answering two fixtures.
    """
    workspace = _make_workspace(tmp_path / "live")
    stale = os.path.realpath(str(tmp_path / "stale"))
    os.makedirs(stale)
    bare = tmp_path / "bare"
    bare.mkdir()
    _register(live=workspace, stale=stale)

    seen = {
        "missing": _raises(None).discriminator,
        "unknown_name": _raises("nosuch").discriminator,
        "relative_path": _raises("./x").discriminator,
        "path_not_a_workspace": _raises(str(bare)).discriminator,
        "registered_unreachable": _raises("stale").discriminator,
        "exempt_verb": routing.exempt_verb_error(
            "serve", EXEMPT_NO_WORKSPACE
        ).discriminator,
    }

    assert seen == {k: k for k in seen}
    assert set(seen) == TARGETING_DISCRIMINATORS


def test_the_discriminator_vocabulary_is_pinned_to_its_literals():
    """The exact strings both boundary renderers switch on — a typo is a silent break.

    Per-constant literals *and* the set, on the parser/store failure-code idiom:
    the set alone would stay green through two constants swapping values.
    """
    assert TARGET_MISSING == "missing"
    assert TARGET_UNKNOWN_NAME == "unknown_name"
    assert TARGET_RELATIVE_PATH == "relative_path"
    assert TARGET_PATH_NOT_A_WORKSPACE == "path_not_a_workspace"
    assert TARGET_REGISTERED_UNREACHABLE == "registered_unreachable"
    assert TARGET_EXEMPT_VERB == "exempt_verb"
    assert TARGETING_DISCRIMINATORS == frozenset(
        {
            "missing",
            "unknown_name",
            "relative_path",
            "path_not_a_workspace",
            "registered_unreachable",
            "exempt_verb",
        }
    )

    assert EXEMPT_CREATES_REGISTRATION == "creates_registration"
    assert EXEMPT_NO_WORKSPACE == "no_workspace"
    assert EXEMPT_EXPLICITLY_GLOBAL == "explicitly_global"
    assert EXEMPT_REASONS == frozenset(
        {"creates_registration", "no_workspace", "explicitly_global"}
    )


# --- group 5: error hygiene ------------------------------------------------

def test_targeting_errors_render_through_the_shipped_mitos_boundary():
    """A ``MitosError``, so ``main()``'s existing arm renders it as one calm line.

    It matters here more than usual: nothing renders this error for three more
    phases, so the boundary is the only thing between a targeting failure and a
    raw traceback in the meantime.
    """
    assert issubclass(ProjectTargetingError, MitosError)


def _one_of_every_discriminator(tmp_path):
    """Builds one error per discriminator, for the whole-vocabulary sweeps."""
    stale = os.path.realpath(str(tmp_path / "stale"))
    os.makedirs(stale)
    bare = tmp_path / "bare"
    bare.mkdir()
    _register(stale=stale)
    return [
        _raises(None),
        _raises("nosuch"),
        _raises("./x"),
        _raises(str(bare)),
        _raises("stale"),
        routing.exempt_verb_error("projects", EXEMPT_EXPLICITLY_GLOBAL),
    ]


def test_no_surface_syntax_ever_reaches_the_shared_error_body(tmp_path):
    """The fallback message names no CLI flag and no MCP call form, for any class.

    This row is **never inverted**. Each boundary renders its own example from the
    typed data, so a surface-specific string in the shared body is always a defect
    — and while ``--project`` does not exist yet, a message naming it would also
    be a promise the branch cannot keep for three more phases.
    """
    errors = _one_of_every_discriminator(tmp_path)
    assert len(errors) == len(TARGETING_DISCRIMINATORS)
    # The CLI flag and the MCP call-form are the two the composition locus names;
    # the rest is each surface's whole recovery vocabulary, none of which may be
    # decided here (and `mitos init` in particular must never reach an agent).
    forbidden = [
        "--project",
        "record_decision(",
        "list_projects",
        "mitos init",
        "mitos projects",
    ]
    for err in errors:
        for syntax in forbidden:
            assert syntax not in str(err), f"{err.discriminator} leaked {syntax!r}"


def test_an_untrusted_selector_is_echoed_escaped_not_raw(tmp_path):
    """A selector is text a human or an agent handed us — it goes through ``repr``.

    A raw interpolation lets a newline break the surface's line structure or an
    ESC push an ANSI sequence to the terminal. The safe and unsafe spellings
    differ by three characters, so a future tidy reverts it silently unless a row
    holds it.
    """
    err = _raises("evil\n\x1b[31mname")

    message = str(err)
    assert "\n" not in message
    assert "\x1b" not in message
    assert "\\n" in message and "\\x1b" in message


@pytest.mark.parametrize(
    "discriminator, kwargs",
    [
        ("unknown_name", {}),
        ("relative_path", {}),
        ("path_not_a_workspace", {"selector": "/x"}),
        ("registered_unreachable", {"selector": "x", "name": "x"}),
        ("exempt_verb", {"verb": "init"}),
    ],
)
def test_a_half_filled_error_cannot_be_constructed(discriminator, kwargs):
    """Each class's required data is enforced at construction, not at the renderer.

    A boundary renderer must never meet a half-filled error and must never have to
    defend against one — so the fault surfaces where it is made. ``ValueError``,
    because it is a programming fault in mitos rather than anything the user did.
    """
    with pytest.raises(ValueError):
        ProjectTargetingError(discriminator, **kwargs)


def test_an_unknown_discriminator_is_refused():
    with pytest.raises(ValueError):
        ProjectTargetingError("not_a_class")


# --- group 6: data only, and the structural half of the cwd invariant ------

def test_every_error_the_resolver_raises_leaves_the_cwd_hint_unset(tmp_path):
    """The resolver has no cwd branch; the boundary fills ``cwd_hint_name``.

    Half of the invariant's structural leg (the other half is the module sweep
    below). It is also what lets the MCP renderer leave the field unset or reframe
    it, since its cwd is a fixed launch dir rather than anyone's location.
    """
    for err in _one_of_every_discriminator(tmp_path):
        assert err.cwd_hint_name is None


def test_the_routing_module_never_reads_the_working_directory():
    """A source sweep, because the invariant is about the *call graph*, not a value.

    An ``os.getcwd()`` anywhere in this module would let a fallback grow back
    inside the resolution path while every behavioural row above stayed green.
    Swept over the parsed **code** rather than the raw text: the module's prose
    names ``os.getcwd`` repeatedly (it is the thing this design removes), so a
    string sweep would be a row that can only ever fail for the wrong reason.
    """
    tree = ast.parse(open(routing.__file__, encoding="utf-8").read())
    reads = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in ("getcwd", "chdir")
    ]
    assert reads == [], f"the routing leaf touches the working directory: {reads}"


def test_importing_the_resolver_pulls_in_no_higher_tier_module():
    """``import mitos.routing`` must drag in neither the CLI nor the store layers.

    ``cli`` imports *this* module in a later phase, so an import back would cycle;
    and four later phases land consumers on this leaf, each of them a chance for
    one convenient import to make it as heavy as the CLI. The same subprocess probe
    the registry leaf and the tree's other leaves already use.
    """
    probe = (
        "import sys; import mitos.routing; "
        "leaked = sorted(m for m in sys.modules "
        "if m.startswith('mitos.') and m.split('.')[1] in "
        "{'cli', 'store', 'sync', 'vector_store', 'renderer', 'importer', "
        "'parser', 'cutover'}); "
        "print(','.join(leaked)); "
        "print(','.join(sorted(m for m in sys.modules if m.startswith('mitos'))))"
    )
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True)
    leaked, loaded = out.stdout.splitlines()
    assert leaked == "", f"leaked imports: {leaked}"

    # …and the exact closure, because the rule this module states is "nothing
    # else, **ever**" and the blacklist above only names eight modules. `config`
    # rides in transitively through `registry`, which is tier-legal and is why
    # nothing here imports it directly. A new member here is a tier decision to
    # adjudicate, not a set to widen reflexively.
    #
    # `env` and `models` arrived in Phase 2b, one level further down the same
    # transitive edge: `config` imports both to build its resolved-env carrier.
    # The adjudication — both are pure stdlib leaves (`env` imports nothing from
    # `mitos` at all, `models` only `os` and `typing`), so they add no dependency
    # weight and no cycle.
    assert loaded.split(",") == [
        "mitos", "mitos.config", "mitos.env", "mitos.errors", "mitos.models",
        "mitos.registry", "mitos.routing"
    ]


# --- group 7: the hint predicate -------------------------------------------

def test_a_sibling_registration_is_not_an_ancestor(tmp_path):
    """The sibling shape this machine's own registry carries, held as documentation.

    With both ``mitos`` and ``mitos-pub`` registered, a bare ``startswith``
    matches *both* entries from inside ``mitos-pub``. Worth knowing that this row
    does **not** bite on its own — longest-registered-path-wins then picks
    ``mitos-pub`` regardless, so it is green under the bug. The row below is the
    one that reds; this one pins the ordinary answer beside it.
    """
    mitos = os.path.realpath(str(tmp_path / "mitos"))
    mitos_pub = os.path.realpath(str(tmp_path / "mitos-pub"))
    reg = {"mitos": mitos, "mitos-pub": mitos_pub}

    assert routing.nearest_registered_ancestor(
        os.path.join(mitos_pub, "src"), reg
    ) == "mitos-pub"


def test_a_directory_whose_name_merely_extends_a_registration_is_not_inside_it(
    tmp_path,
):
    """The bare-``startswith`` trap, in the one shape longest-wins cannot rescue.

    ``…/mitos-pub-sandbox`` is not inside ``…/mitos-pub``; it just starts with its
    name — the shape a ``cp -r`` or a second clone produces, so it is ordinary
    rather than contrived. A prefix comparison hints ``mitos-pub`` here, which is
    a **confidently wrong** answer on the line whose whole job is to be a helpful
    guess, and the caller's most likely next move is to act on it. Verified to red
    against the bare form.
    """
    registered = os.path.realpath(str(tmp_path / "mitos-pub"))
    sandbox = os.path.realpath(str(tmp_path / "mitos-pub-sandbox"))

    assert routing.nearest_registered_ancestor(
        os.path.join(sandbox, "src"), {"mitos-pub": registered}
    ) is None


def test_the_longest_registered_ancestor_wins(tmp_path):
    """Nested registrations resolve to the nearest one, not the outermost."""
    outer = os.path.realpath(str(tmp_path / "outer"))
    inner = os.path.join(outer, "inner")
    reg = {"outer": outer, "inner": inner}

    assert routing.nearest_registered_ancestor(
        os.path.join(inner, "deep"), reg
    ) == "inner"


def test_the_registered_path_itself_is_its_own_ancestor(tmp_path):
    """Standing *in* a registered workspace hints it — the equality half."""
    root = os.path.realpath(str(tmp_path / "proj"))
    assert routing.nearest_registered_ancestor(root, {"proj": root}) == "proj"


def test_a_directory_under_no_registration_hints_nothing(tmp_path):
    """No ancestor is a real answer; the renderer simply omits the hint line."""
    reg = {"proj": os.path.realpath(str(tmp_path / "proj"))}
    assert routing.nearest_registered_ancestor(
        os.path.realpath(str(tmp_path / "elsewhere")), reg
    ) is None


def test_two_names_for_one_path_resolve_in_document_order(tmp_path):
    """The tolerated hand-edit resolves by an explicit rule, not by dict luck."""
    root = os.path.realpath(str(tmp_path / "proj"))
    assert routing.nearest_registered_ancestor(
        os.path.join(root, "sub"), {"first": root, "second": root}
    ) == "first"


def test_a_workspace_registered_at_the_filesystem_root_behaves():
    """``/`` is the case a naive ``path + os.sep`` comparison turns into ``//``."""
    assert routing.nearest_registered_ancestor("/etc", {"root": "/"}) == "root"


# --- group 8: did-you-mean and the bound -----------------------------------

@pytest.mark.parametrize(
    "selector, expected",
    [("MITOS", "mitos"), ("ĄŽUOLAS", "ąžuolas")],
)
def test_an_all_caps_selector_still_suggests_its_registered_name(selector, expected):
    """The rows a case-sensitive matcher reds on — and only these.

    Measured: a lightly-cased typo (``Mitos``) matches either way, so a row driven
    by one proves nothing about folding. An all-caps selector — what a human
    produces from a shouted config value or a copied heading — suggests **nothing**
    unfolded. The Lithuanian arm is the reason it must be ``casefold()``.
    """
    assert routing.close_project_matches(
        selector, ["mitos", "ąžuolas"]
    ) == [expected]


def test_folding_is_for_matching_only_and_never_enters_resolution(tmp_path):
    """The lookup stays exact — no normalization exists to diverge."""
    _register(mitos=_make_workspace(tmp_path / "mitos"))

    err = _raises("MITOS")

    assert err.discriminator == "unknown_name"   # not resolved through folding
    assert err.close_matches == ["mitos"]        # but routed through did-you-mean


def test_a_short_registry_renders_in_full_in_document_order():
    """Below the bound the full list is the better answer, and order is the file's."""
    names = ["zulu", "alpha", "mike"]

    bounded = routing.bounded_registered_names(names, ["alpha"])

    assert bounded.names == names
    assert bounded.total == 3
    assert bounded.collapsed is False


def test_a_long_registry_collapses_to_the_matches_plus_a_truthful_total():
    """Past the bound the enumeration is a wall in place of a diagnostic."""
    names = [f"proj{i}" for i in range(routing.REGISTERED_NAMES_BOUND + 5)]

    bounded = routing.bounded_registered_names(names, ["proj3"])

    assert bounded.names == ["proj3"]
    assert bounded.total == len(names)
    assert bounded.collapsed is True


def test_a_long_registry_with_no_close_match_names_the_count_not_a_slice():
    """Deliberately not a "first 10 of 40" prefix — document order carries no ranking.

    An arbitrary slice reads as a relevance answer while being none; the honest
    output is an empty list, an intact total, and the surface's discovery pointer.
    """
    names = [f"proj{i}" for i in range(routing.REGISTERED_NAMES_BOUND + 1)]

    bounded = routing.bounded_registered_names(names, [])

    assert bounded.names == []
    assert bounded.total == len(names)
    assert bounded.collapsed is True


def test_the_bound_is_inclusive_at_its_own_threshold():
    """Exactly at the bound, nothing collapses — the boundary the constant names."""
    names = [f"proj{i}" for i in range(routing.REGISTERED_NAMES_BOUND)]
    assert routing.bounded_registered_names(names, []).collapsed is False


# --- group 9: registry faults ----------------------------------------------

@pytest.mark.parametrize("selector", ["somename", "/absolute/path"])
def test_a_malformed_registry_propagates_unwrapped_from_both_forms(selector):
    """A file fault is not a targeting fault, and it is loud on the path form too.

    The path form needs routing for nothing but the echo name, which is exactly
    why the tempting ``except RegistryError: {}`` would go there — hiding a
    corrupt routing table on the dominant agent path, the worst place to be quiet.
    """
    _write_registry("this is not = = valid toml\n")

    with pytest.raises(RegistryError):
        routing.resolve_project(selector)


def test_an_absent_registry_still_resolves_an_absolute_path(tmp_path):
    """No projects registered yet is the healthy fresh-machine state, not a fault."""
    root = _make_workspace(tmp_path / "proj")
    assert not os.path.exists(registry.registry_path())

    resolved = routing.resolve_project(root)

    assert resolved.root == root
    assert resolved.name is None


def test_an_absent_registry_reports_an_empty_vocabulary_on_a_failure(tmp_path):
    """The empty-registry variant each renderer composes keys on this being ``[]``."""
    err = _raises("nosuch")

    assert err.registered_names == []
    assert err.close_matches == []


def test_registered_names_ride_every_resolver_failure_in_document_order(tmp_path):
    """The vocabulary is data on the error; bounding it for display is the renderer's."""
    _register(
        zulu=_make_workspace(tmp_path / "zulu"),
        alpha=_make_workspace(tmp_path / "alpha"),
    )

    for err in [_raises(None), _raises("nosuch"), _raises("./x")]:
        assert err.registered_names == ["zulu", "alpha"]


# --- group 10: the predicate agreement net ---------------------------------

def test_the_workspace_predicate_agrees_with_status_at_every_step(
    tmp_path, monkeypatch, capsys
):
    """``is_workspace`` and ``mitos status``' own verdict move together, file by file.

    The leaf re-spells the shipped validity triple rather than importing it (the
    only shipped spelling is three local variables inside ``cmd_status``, and
    constructing a ``MitosConfig`` to reach them would raise ``ConfigError`` from a
    validity probe). This is that duplication's drift net: a build-up one file at a
    time, asserting the two verdicts agree at each step rather than asserting a
    hand-written expectation of either.

    Note it reads the ``--json`` payload's ``initialized`` field, not the exit
    code: ``cmd_status`` returns *readiness*, which folds in a key and a reachable
    Qdrant.
    """
    monkeypatch.setattr(cli, "_check_qdrant", lambda url, coll: {
        "reachable": True, "collection_exists": False, "points": None})
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")

    root = tmp_path / "proj"
    root.mkdir()

    def _status_says_initialized() -> bool:
        cli.cmd_status(str(root), as_json=True)
        return json.loads(capsys.readouterr().out)["initialized"]

    steps = [
        lambda: None,
        lambda: os.makedirs(str(root / ".mitos")),
        lambda: (root / ".mitos" / "config.toml").write_text("# a mitos workspace\n"),
        lambda: (root / "decisions.md").write_text("# Decisions\n"),
    ]
    verdicts = []
    for step in steps:
        step()
        assert routing.is_workspace(str(root)) == _status_says_initialized()
        verdicts.append(routing.is_workspace(str(root)))

    # …and the build-up actually crosses the boundary, so the agreement is not
    # four rows of "False == False".
    assert verdicts == [False, False, False, True]


def test_the_predicate_admits_a_workspace_with_no_graph(tmp_path):
    """A committed ``.mitos/`` with a gitignored SQLite file is a *valid* workspace.

    Strengthening the predicate with a graph or node-count check would refuse the
    fresh clone the escape hatch exists to serve. The empty-graph state is named by
    the read surfaces, not by refusing to resolve.
    """
    root = _make_workspace(tmp_path / "clone")
    assert not os.path.exists(os.path.join(root, ".mitos", "graph.sqlite"))
    assert routing.is_workspace(root) is True
    assert routing.resolve_project(root).root == root
