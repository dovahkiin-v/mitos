"""Tests for the global project registry (``mitos/registry.py``).

The registry is the ``name → workspace path`` routing map that lets a command
target a project by name instead of inferring one from the process's working
directory. This module is the leaf's own suite: the flat schema and its
round-trip fidelity, both write-time guards, the structure-preserving atomic
write, document order, and the calm-error surface for a hand-broken file.

No services and no mocks: the registry is a filesystem leaf, and the autouse
``hermetic_mitos_env`` fixture already redirects ``XDG_CONFIG_HOME`` per test, so
every row writes into its own ``tmp_path`` config root. Paths are asserted
against ``os.path.realpath`` of the fixture path, never the raw fixture path — a
``/tmp`` that is a symlink (macOS) would otherwise red the suite for a reason
that has nothing to do with the registry.
"""

import os
import stat
import tomllib
from unittest.mock import patch

import pytest

from mitos import registry
from mitos.config import global_registry_path
from mitos.errors import MitosError, RegistryError


# --- helpers ---------------------------------------------------------------

def _write_registry(text: str) -> str:
    """Hand-writes the registry file (the hand-editable states we must tolerate)."""
    path = registry.registry_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _read_registry() -> str:
    with open(registry.registry_path(), "r", encoding="utf-8") as f:
        return f.read()


# --- path resolution -------------------------------------------------------

def test_registry_path_resolves_through_the_one_xdg_mechanism(tmp_path):
    """``registry_path`` delegates to config's XDG resolution, not a second copy.

    A hand-rolled ``expanduser("~/.config/…")`` here would slip the suite's
    per-test config-root redirect entirely, and every one of the suite's ``init``
    calls would register a vanishing tmp path into the developer's real registry.
    """
    assert registry.registry_path() == global_registry_path()
    assert registry.registry_path() == str(
        tmp_path / "xdg_config" / "mitos" / "registry.toml"
    )


# --- T1: the flat schema + adversarial round-trip --------------------------

def test_absent_registry_reads_as_an_empty_map():
    """No registry file (and no config dir) is the healthy empty state, not an error.

    A fresh machine has neither. ``load()`` raising ``FileNotFoundError`` here
    would make "no projects registered yet" indistinguishable from a fault on
    every surface that reads the map.
    """
    assert not os.path.exists(registry.registry_path())
    assert registry.load() == {}
    assert registry.lookup("anything") is None
    assert registry.reverse_lookup("/anywhere") is None


def test_register_then_load_round_trips_a_plain_name_and_path(tmp_path):
    """The base case: a registered pair reads back byte-identical, and only it."""
    workspace = tmp_path / "proj"
    workspace.mkdir()

    outcome = registry.register(str(workspace))

    assert outcome.name == "proj"
    assert outcome.path == os.path.realpath(str(workspace))
    assert outcome.action == "created"
    assert outcome.previous_path is None
    assert registry.load() == {"proj": os.path.realpath(str(workspace))}
    assert registry.lookup("proj") == os.path.realpath(str(workspace))


@pytest.mark.parametrize(
    "name, subdir",
    [
        ('quote"name', "plain"),           # a `"` in the name
        ("ąžuolas", "plain"),              # non-ASCII (P9) — no normalization
        ("dotted.name", "plain"),          # a dot: bare, TOML would nest it
        ("plain", 'back\\slash'),          # a `\` in the PATH — the silent-corruption case
    ],
)
def test_adversarial_names_and_paths_round_trip_byte_identically(tmp_path, name, subdir):
    """A quote-bearing name and a backslash-bearing path survive write→read exactly.

    This pair is the row's whole point, and a fixture using only ordinary paths
    goes green either way. The pre-widening serializer *raised* on the ``"`` and
    would have written the path into a TOML **basic** string, where ``\\t`` is
    re-read as a TAB — registering a path that does not exist, with nothing naming
    the cause. A dotted name written as a bare key would parse as a nested table.
    """
    workspace = tmp_path / subdir
    workspace.mkdir()
    expected = os.path.realpath(str(workspace))

    registry.register(str(workspace), name=name)

    reg = registry.load()
    assert list(reg) == [name]                  # the key decoded back exactly
    assert reg[name] == expected                # and so did the path
    assert registry.lookup(name) == expected
    assert registry.reverse_lookup(expected) == name


def test_a_backslash_t_path_is_not_silently_turned_into_a_tab(tmp_path):
    """The corruption this widening exists to prevent, asserted at the file level.

    Pinned separately from the round-trip row above because it names the exact
    failure: ``tomllib.loads('k = "/x/a\\tb"')`` yields a TAB. A path carrying a
    backslash must therefore never be written into a basic string.
    """
    hostile = str(tmp_path / "a\\tb")
    os.makedirs(hostile, exist_ok=True)

    registry.register(hostile, name="hostile")

    assert "\t" not in _read_registry()
    assert registry.load()["hostile"] == os.path.realpath(hostile)


@pytest.mark.parametrize(
    "line, found",
    [
        ('"proj" = { path = "/x" }', "dict"),   # a table where a path belongs
        ('"proj" = ["/x"]', "list"),            # an array
        ('"proj" = 50', "int"),                 # a number
        ("[proj]", "dict"),                     # a hand-written table header
    ],
)
def test_a_non_string_value_is_the_one_violable_shape(line, found):
    """I3's flat ``name → str`` schema fails loud on any other shape, naming the fix.

    An old reader must fail *loudly* on a shape it does not understand rather than
    misread a table as a path — which is what makes the version-marker-free flat
    file safe. The message also has to name the dotted-key trap: a perfectly
    ordinary directory name (``example.com``) written bare *becomes* a table.
    """
    path = _write_registry(line + "\n")

    with pytest.raises(RegistryError) as exc:
        registry.load()

    message = str(exc.value)
    assert "proj" in message
    assert found in message
    assert path in message
    assert "quote" in message.lower()  # the dotted-key recovery


def test_document_order_is_preserved_by_load_and_by_an_unrelated_write(tmp_path):
    """Read order is document order — the order first-match resolution decides in.

    An alphabetical sort anywhere would make ``mitos projects`` show an order the
    reverse lookup does not actually follow.
    """
    _write_registry('"zulu" = "/z"\n"alpha" = "/a"\n"mike" = "/m"\n')
    assert list(registry.load()) == ["zulu", "alpha", "mike"]

    workspace = tmp_path / "newcomer"
    workspace.mkdir()
    registry.register(str(workspace))

    # The new entry appends; the three existing ones keep their hand-written order.
    assert list(registry.load()) == ["zulu", "alpha", "mike", "newcomer"]


def test_a_bare_hand_written_key_still_reads_and_updates_in_place(tmp_path):
    """A human's unquoted key is honoured on read and matched on write.

    Writes always quote (a dot or a non-ASCII character in a bare key is a parse
    hazard), but the file is hand-editable and a bare ASCII key is perfectly legal
    TOML — a re-init must update *that* line, not append a second definition of
    the same name (which ``tomllib`` would then refuse as a duplicate key).
    """
    workspace = tmp_path / "proj"
    workspace.mkdir()
    _write_registry('proj = "/stale/path"\n')

    outcome = registry.register(str(workspace), force=True)

    assert outcome.action == "repointed"
    assert outcome.previous_path == "/stale/path"
    assert registry.load() == {"proj": os.path.realpath(str(workspace))}
    # Replaced in place, not appended: a second definition of the same name would
    # make the file a duplicate key, which `tomllib` refuses outright on the next read.
    assert _read_registry() == f'"proj" = "{os.path.realpath(str(workspace))}"\n'


# --- T2: guards ------------------------------------------------------------

def test_name_collision_at_a_different_path_names_both_recoveries(tmp_path):
    """A name already registered elsewhere refuses, naming its current path.

    Both recoveries have to be in the message: the caller cannot know from the
    refusal alone whether they meant to move the registration (``--force``) or to
    register a second project that happens to share a directory name (``--name``).
    """
    first = tmp_path / "a" / "proj"
    second = tmp_path / "b" / "proj"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    registry.register(str(first))

    with pytest.raises(RegistryError) as exc:
        registry.register(str(second))

    message = str(exc.value)
    assert "proj" in message
    assert os.path.realpath(str(first)) in message   # the CURRENT recorded path
    assert "--force" in message
    assert "--name" in message
    # Refused means unchanged: the first registration still stands alone.
    assert registry.load() == {"proj": os.path.realpath(str(first))}


def test_force_repoints_a_name_and_reports_the_previous_path(tmp_path):
    """``--force`` waives the name collision and hands back what it displaced."""
    first = tmp_path / "a" / "proj"
    second = tmp_path / "b" / "proj"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    registry.register(str(first))

    outcome = registry.register(str(second), force=True)

    assert outcome.action == "repointed"
    assert outcome.previous_path == os.path.realpath(str(first))
    assert outcome.path == os.path.realpath(str(second))
    assert registry.load() == {"proj": os.path.realpath(str(second))}


@pytest.mark.parametrize("force", [False, True], ids=["plain", "force"])
def test_path_uniqueness_is_not_waived_by_force(tmp_path, force):
    """A workspace already registered under one name refuses a second name — even with ``--force``.

    The natural reading of ``--force`` is "force"; under it this guard stops firing
    on exactly the mistakes it exists for — a re-init in the wrong directory, or a
    rename attempted as ``--force`` — because both arrive *carrying* ``--force``.
    So the flag waives the name collision only.
    """
    workspace = tmp_path / "proj"
    workspace.mkdir()
    registry.register(str(workspace), name="first")

    with pytest.raises(RegistryError) as exc:
        registry.register(str(workspace), name="second", force=force)

    message = str(exc.value)
    assert "first" in message                     # the holding name
    assert registry.registry_path() in message    # renaming is a hand-edit of THIS file
    assert registry.load() == {"first": os.path.realpath(str(workspace))}


def test_the_unwaivable_guard_is_reported_first_when_both_conditions_hold(tmp_path):
    """With both guards tripped, the refusal names the guard ``--force`` cannot clear.

    Reporting the waivable collision first would send the caller to ``--force``,
    have them re-run, and then refuse for the *other* reason — a two-round-trip
    dead end. The unwaivable condition first names the real recovery immediately.
    """
    other = tmp_path / "other"
    workspace = tmp_path / "proj"
    other.mkdir()
    workspace.mkdir()
    registry.register(str(workspace), name="held")     # this PATH is taken…
    registry.register(str(other), name="taken")        # …and this NAME is taken

    with pytest.raises(RegistryError) as exc:
        registry.register(str(workspace), name="taken")

    message = str(exc.value)
    assert "held" in message                    # the path guard's holder — reported first
    assert "is already registered at" not in message  # NOT the name-collision refusal
    # `--force` may appear only to close the door on it, never as the recovery.
    assert "deliberately does not" in message


def test_re_registering_the_same_pair_is_an_idempotent_reassert(tmp_path):
    """Same name, same path → ``reasserted``, and the file is byte-unchanged.

    Re-init is the documented way an existing workspace joins the registry, so it
    must be a quiet no-op rather than a collision with itself.
    """
    workspace = tmp_path / "proj"
    workspace.mkdir()
    registry.register(str(workspace))
    before = _read_registry()

    outcome = registry.register(str(workspace))

    assert outcome.action == "reasserted"
    assert outcome.previous_path is None
    assert _read_registry() == before


def test_reasserting_one_of_two_names_for_one_path_is_not_a_fault(tmp_path):
    """An exact re-assert short-circuits both guards — a tolerated state stays tolerated.

    Two names for one path is a hand-edit the registry accepts (both reach the same
    workspace, so nothing can be corrupted). Running the path guard ahead of the
    exact-match check would turn a re-init of one of those names into a refusal,
    rendering a tolerated shape as a fault.
    """
    workspace = tmp_path / "proj"
    workspace.mkdir()
    real = os.path.realpath(str(workspace))
    _write_registry(f'"first" = "{real}"\n"second" = "{real}"\n')

    outcome = registry.register(str(workspace), name="second")

    assert outcome.action == "reasserted"
    assert registry.load() == {"first": real, "second": real}


# --- T2b: duplicate-path tolerance ----------------------------------------

def test_two_names_for_one_path_resolve_by_first_match_in_file_order():
    """A duplicate path loads cleanly and reverse-resolves to the FIRST name.

    A dict keyed by path would silently keep the *last* one. Making the rule
    explicit — and pinning it — is what lets both guards be write-time guidance
    rather than something downstream relies on for uniqueness.
    """
    _write_registry('"second" = "/shared/ws"\n"first" = "/shared/ws"\n')

    reg = registry.load()
    assert reg == {"second": "/shared/ws", "first": "/shared/ws"}
    assert registry.reverse_lookup("/shared/ws") == "second"  # first in FILE order


def test_a_duplicate_name_surfaces_through_the_malformed_registry_error():
    """A repeated key is already a TOML error, so it needs no guard of its own."""
    path = _write_registry('"proj" = "/a"\n"proj" = "/b"\n')

    with pytest.raises(RegistryError) as exc:
        registry.load()

    assert path in str(exc.value)
    assert "TOML" in str(exc.value)


# --- name legality ---------------------------------------------------------

@pytest.mark.parametrize(
    "name, because",
    [
        ("", "empty — a workspace at `/` has no directory name"),
        (".config", "a leading dot — `mitos init` inside a dot-directory"),
        ("~backup", "a leading tilde — a path selector's shape"),
        (f"a{os.sep}b", "a path separator"),
        ("bad\x01name", "a control character"),
    ],
)
def test_illegal_names_are_refused_with_a_pointer_at_name(name, because):
    """Every illegal shape refuses and points at ``--name``, never registering it.

    Two of these are ordinary inputs rather than hypotheticals: ``mitos init`` at a
    filesystem root yields an empty basename, and inside any dot-directory a
    leading-dot one. Registering either would put an entry in the file that no
    selector can reach.
    """
    with pytest.raises(RegistryError) as exc:
        registry.validate_name(name)

    assert "--name" in str(exc.value), because


@pytest.mark.parametrize("name", ["proj", "ąžuolas", "dotted.name", "UPPER", "a-b_c", "x.y.z"])
def test_legal_names_are_permissive_and_unnormalized(name):
    """Anything that is not path-shaped is legal, and is stored exactly as given.

    Non-ASCII is deliberate (P9), and there is no case-folding or normalization
    here: a resolver compares exactly, so a normalization at the write side would
    be the seam the two sides diverge on.
    """
    registry.validate_name(name)  # must not raise


def test_the_default_name_comes_from_the_canonical_paths_basename(tmp_path):
    """A trailing slash and a symlinked route both register the real directory's name."""
    real = tmp_path / "real-name"
    real.mkdir()
    link = tmp_path / "link-name"
    link.symlink_to(real, target_is_directory=True)

    assert registry.default_name(registry.canonicalize(str(real) + os.sep)) == "real-name"
    assert registry.default_name(registry.canonicalize(str(link))) == "real-name"


def test_canonicalize_resolves_symlinks_unlike_an_abspath(tmp_path):
    """The one canonical spelling is ``realpath`` — abspath does not resolve links.

    ``MitosConfig.workspace_dir`` is ``abspath``-normalized, so the path the
    registry stores can legitimately differ from it for the same workspace. Every
    comparison (registration, the path guard, reverse lookup) must go through this
    one function or identity splits silently.
    """
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    assert registry.canonicalize(str(link)) == os.path.realpath(str(real))
    assert os.path.abspath(str(link)) != registry.canonicalize(str(link))


# --- the structure-preserving atomic write --------------------------------

def test_a_write_preserves_every_unrelated_byte(tmp_path):
    """Comments, blank lines, ordering, and a human's organization all survive.

    The file is meant to be hand-edited, so a full re-serialization would silently
    destroy the editor's work on every ``mitos init``. Only the touched line moves.
    """
    workspace = tmp_path / "gamma"
    workspace.mkdir()
    original = (
        "# my projects\n"
        "\n"
        "# the important one\n"
        '"alpha" = "/ws/alpha"\n'
        "\n"
        '"beta" = "/ws/beta"   # inline note\n'
    )
    _write_registry(original)

    registry.register(str(workspace))

    after = _read_registry()
    assert after.startswith(original)  # every prior byte intact, entry appended
    assert after == original + f'"gamma" = "{os.path.realpath(str(workspace))}"\n'


def test_a_repoint_replaces_only_its_own_line(tmp_path):
    """Updating one entry leaves the surrounding comments and entries untouched."""
    workspace = tmp_path / "beta"
    workspace.mkdir()
    _write_registry(
        "# header\n"
        '"alpha" = "/ws/alpha"\n'
        '"beta" = "/ws/old-beta"\n'
        "# trailing comment\n"
        '"gamma" = "/ws/gamma"\n'
    )

    registry.register(str(workspace), force=True)

    after = _read_registry()
    assert after == (
        "# header\n"
        '"alpha" = "/ws/alpha"\n'
        f'"beta" = "{os.path.realpath(str(workspace))}"\n'
        "# trailing comment\n"
        '"gamma" = "/ws/gamma"\n'
    )


def test_a_file_without_a_trailing_newline_gains_one_before_the_append(tmp_path):
    """An appended entry never lands on the tail of a hand-written last line."""
    workspace = tmp_path / "proj"
    workspace.mkdir()
    _write_registry('"alpha" = "/ws/alpha"')  # no trailing newline

    registry.register(str(workspace))

    assert registry.load() == {
        "alpha": "/ws/alpha",
        "proj": os.path.realpath(str(workspace)),
    }


def test_the_temp_file_lands_in_the_target_directory(tmp_path):
    """The temp file is a sibling of the registry, so ``os.replace`` is atomic.

    A temp in the system temp dir can be on a different filesystem, where the
    replace degrades to a copy — and a reader can then observe a partial file.
    """
    workspace = tmp_path / "proj"
    workspace.mkdir()
    seen = {}
    real_mkstemp = registry.tempfile.mkstemp

    def _recording_mkstemp(*args, **kwargs):
        seen["dir"] = kwargs.get("dir")
        return real_mkstemp(*args, **kwargs)

    with patch.object(registry.tempfile, "mkstemp", _recording_mkstemp):
        registry.register(str(workspace))

    assert seen["dir"] == os.path.dirname(registry.registry_path())


def test_a_failed_replace_leaves_the_previous_file_byte_intact(tmp_path):
    """A write that dies mid-flight neither corrupts nor half-updates the registry.

    The practical form of the atomicity claim: force the replace to fail and assert
    the pre-existing file is unchanged and no temp file is left behind.
    """
    workspace = tmp_path / "proj"
    workspace.mkdir()
    _write_registry('"alpha" = "/ws/alpha"\n')
    before = _read_registry()
    reg_dir = os.path.dirname(registry.registry_path())

    with patch.object(registry.os, "replace", side_effect=OSError("disk gone")):
        with pytest.raises(RegistryError) as exc:
            registry.register(str(workspace))

    assert _read_registry() == before
    assert [n for n in os.listdir(reg_dir) if n.endswith(".tmp")] == []
    assert "unregistered" in str(exc.value)  # names the state the caller is left in


def test_the_config_directory_is_created_on_first_registration(tmp_path):
    """``<config-home>/mitos/`` may not exist — the global ``.env`` path only reads.

    On a fresh machine ``init`` is the first thing that needs the directory, and
    the autouse fixture points the config root at a tmp dir that does not exist
    yet, so every registration row exercises this.
    """
    assert not os.path.exists(os.path.dirname(registry.registry_path()))
    workspace = tmp_path / "proj"
    workspace.mkdir()

    registry.register(str(workspace))

    assert os.path.isfile(registry.registry_path())


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores directory mode bits, so a read-only config dir stays writable",
)
def test_an_unwritable_registry_directory_raises_a_calm_named_error(tmp_path):
    """An unwritable registry names the path, the OS cause, and the recovery.

    And it says the workspace is *initialized but unregistered* — the state
    ``init`` actually leaves behind, since registration never unwinds the scaffold.
    """
    workspace = tmp_path / "proj"
    workspace.mkdir()
    _write_registry('"alpha" = "/ws/alpha"\n')
    reg_dir = os.path.dirname(registry.registry_path())
    os.chmod(reg_dir, stat.S_IRUSR | stat.S_IXUSR)  # 0o500
    try:
        with pytest.raises(RegistryError) as exc:
            registry.register(str(workspace))
    finally:
        os.chmod(reg_dir, stat.S_IRWXU)

    message = str(exc.value)
    assert registry.registry_path() in message
    assert "unregistered" in message
    assert "mitos init" in message


def test_the_whole_file_is_validated_before_the_line_scan(tmp_path):
    """A hand-written ``[table]`` cannot reach the write's per-line key matcher.

    ``tomllib.loads("[proj]")`` decodes to ``{"proj": {}}`` — ONE key — so a naive
    per-line scan would treat a table header as the definition line for ``proj``
    and overwrite it. Loading (and shape-checking) the whole file first makes that
    state unreachable; this row pins the ordering rather than trusting it.
    """
    workspace = tmp_path / "proj"
    workspace.mkdir()
    _write_registry("[proj]\nkey = 1\n")
    before = _read_registry()

    with pytest.raises(RegistryError):
        registry.register(str(workspace))

    assert _read_registry() == before  # nothing was surgically edited


def test_a_multi_line_value_is_refused_rather_than_silently_duplicated(tmp_path):
    """A hand-edit the line surgery cannot amend refuses, leaving the file untouched.

    A value written as a multi-line TOML string is legal and loads fine, but neither
    of its lines reads as a key definition — so the update would *append* a second
    definition of the same name and the registry would raise on its very next read,
    long after ``init`` reported success. The postcondition catches it at the moment
    it happens, with nothing written.
    """
    workspace = tmp_path / "proj"
    workspace.mkdir()
    _write_registry('"proj" = """\n/old/multi/line/path"""\n')
    before = _read_registry()

    with pytest.raises(RegistryError) as exc:
        registry.register(str(workspace), force=True)

    assert _read_registry() == before          # untouched, not half-updated
    assert "Nothing was written" in str(exc.value)
    assert registry.registry_path() in str(exc.value)
    # Still loadable, still the old value (TOML drops the newline right after `"""`).
    assert registry.load() == {"proj": "/old/multi/line/path"}


def test_registry_errors_render_through_the_shipped_mitos_boundary():
    """``RegistryError`` is a ``MitosError``, which is what keeps the CLI calm.

    Every registry fault reaches the user through ``main()``'s existing
    ``except MitosError`` arm as a one-line ``Error: …``. A sibling hierarchy would
    need a second boundary and would surface as a raw traceback until it got one.
    """
    assert issubclass(RegistryError, MitosError)


# --- leaf discipline -------------------------------------------------------

def test_importing_the_registry_pulls_in_no_higher_tier_module():
    """``import mitos.registry`` must drag in neither the CLI nor the store layers.

    The routing leaf is imported *by* ``cli``, so an import back would cycle; and
    every later phase that lands a consumer on top of it (a resolver, a status
    sweep, an echo) is a chance for one convenient import to make this leaf as
    heavy as the CLI. A prose-only tier rule on a vision's first phase decays —
    this is the same subprocess probe the tree already uses for its other leaves.
    """
    import subprocess
    import sys as _sys

    probe = (
        "import sys; import mitos.registry; "
        "leaked = sorted(m for m in sys.modules "
        "if m.startswith('mitos.') and m.split('.')[1] in "
        "{'cli', 'store', 'sync', 'vector_store', 'renderer', 'importer', "
        "'parser', 'cutover'}); "
        "print(','.join(leaked))"
    )
    out = subprocess.run([_sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"leaked imports: {out.stdout.strip()}"


# --- the registry stores routing only -------------------------------------

def test_the_registry_file_carries_nothing_but_name_and_path(tmp_path):
    """No collection, no version marker, no metadata — routing only.

    The registry is derivative (a re-init rebuilds it) and must never become a
    second source of record. Anything a workspace derives for itself — most
    pointedly its Qdrant collection — stays workspace-local, so editing or
    repointing a registration cannot change what a project *is*.
    """
    workspace = tmp_path / "proj"
    workspace.mkdir()
    registry.register(str(workspace))

    with open(registry.registry_path(), "rb") as f:
        data = tomllib.load(f)

    assert data == {"proj": os.path.realpath(str(workspace))}
    assert all(isinstance(v, str) for v in data.values())
    assert "collection" not in _read_registry()
    assert "version" not in _read_registry()
