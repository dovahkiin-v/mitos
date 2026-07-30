"""The collection-binding suite: path-hash derivation + `qdrant_collection` retirement.

A workspace's Qdrant collection is a pure function of its canonicalized absolute
path, with no opt-out. Two halves, and each is unsafe or inert without the other:

* **The derivation** (`default_collection_name`) hashes the *full canonical path*, so
  two same-basename siblings and a `cp -r` sandbox never share a collection.
* **The retirement** (`qdrant_collection` ∈ `RETIRED_CONFIG_KEYS`) means no persisted
  value can override the derivation — not a legacy auto-pin, not a hand-set one, not
  one that arrived inside a `git clone` of a repo that committed `.mitos/`.

That second half is why the fix is total rather than partial: an `init`-time strip
is structurally unreachable on a workspace `init` never runs on, and a clone reached
by absolute path is exactly that state.

Every row here is offline — the derivation is stdlib-pure and the retirement is a
loader path. Assertions are RELATIONAL (distinct / identical / shape) with exactly
one exact-string row, which pins the algorithm against an out-of-band digest. A row
that recomputed the digest with `hashlib` would prove only that the function calls
`hashlib`, and would pass against a wrong step order or a basename-only hash.
"""

import json
import os
import shutil
import tempfile

import pytest

from mitos import cli
from mitos.config import (
    CONFIG_SCHEMA,
    RETIRED_CONFIG_KEYS,
    MitosConfig,
    default_collection_name,
)
from mitos.vector_store import QdrantVectorStore


def _write_pin(workspace_dir, value='"mitos-mitos-pub"') -> str:
    """Writes a `.mitos/config.toml` carrying only a `qdrant_collection` line.

    `cmd_init` seeds `config.toml` only when it is ABSENT, so pre-creating the file
    is how a fixture becomes a pinned workspace: run `init` afterwards and it leaves
    the file alone (the re-init-of-a-pinned-workspace case), or run nothing at all
    and reach the workspace by path (the clone case).

    Args:
        workspace_dir: The workspace root.
        value: The raw TOML right-hand side, so a row can write a wrong TYPE too.

    Returns:
        The path written.
    """
    mitos_dir = os.path.join(str(workspace_dir), ".mitos")
    os.makedirs(mitos_dir, exist_ok=True)
    path = os.path.join(mitos_dir, "config.toml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"qdrant_collection = {value}\n")
    return path


def _qdrant(reachable, collection_exists, points=None):
    """A `cli._check_qdrant` stub — the offline seam for all four collection rows."""
    return lambda url, coll: {
        "reachable": reachable,
        "collection_exists": collection_exists,
        "points": points,
    }


def _scroll(present_uuids):
    """A no-create `scroll_point_ids` stub reporting a fixed point-id set."""
    return lambda base_url, collection, page_size=256: set(present_uuids)


def _commit_one(workspace_dir):
    """Commits one active decision, so a workspace has a populated graph."""
    from mitos.parser import ParsedEntry
    from mitos.store import GraphStore

    store = GraphStore(MitosConfig(str(workspace_dir)).db_path)
    entry = ParsedEntry("decision", "node-00", 1, 5)
    entry.axiom = "Axiom 0"
    entry.rejected_paths = "n/a"
    return store.commit_parsed_entry(entry).node_id


# ---------------------------------------------------------------------------
# The derivation — I9's safety half
# ---------------------------------------------------------------------------

class TestDerivation:
    """`mitos-<safe basename>-<8 hex of sha256(canonical path)>`, and why each part."""

    def test_the_algorithm_is_pinned_against_an_out_of_band_digest(self) -> None:
        """The one exact-string row in this suite — the algorithm itself.

        The digest was verified OUTSIDE Python, so this row can fail on a changed step
        order, a basename-only hash, a different digest length, or a wrong encoding:

            printf '%s' "/x/workshop_mcp" | sha256sum | cut -c1-8   ->  f1d667f5

        The path cannot exist and carries no symlink component, so `realpath` returns
        it unchanged on any machine and the expected value is not machine-dependent.
        The shape is contract and cannot change after release: the name is the address
        of live data, so a changed derivation renames every collection in the wild and
        strands its vectors.
        """
        assert default_collection_name("/x/workshop_mcp") == "mitos-workshop_mcp-f1d667f5"

    def test_same_basename_siblings_get_distinct_collections(self, tmp_path) -> None:
        """The basename alone is not an identity — two `mitos/` dirs are two projects."""
        first = tmp_path / "a" / "mitos"
        second = tmp_path / "b" / "mitos"
        first.mkdir(parents=True)
        second.mkdir(parents=True)

        assert default_collection_name(str(first)) != default_collection_name(str(second))
        # Both keep the same VISIBLE segment — only the digest separates them, which is
        # exactly why the digest cannot be dropped for legibility's sake.
        assert default_collection_name(str(first)).startswith("mitos-mitos-")
        assert default_collection_name(str(second)).startswith("mitos-mitos-")

    def test_a_copied_workspace_does_not_inherit_its_source_collection(self, tmp_path) -> None:
        """The headline failure: `cp -r` to sandbox a branch, then clobber the original.

        A real `shutil.copytree` of a real initialized workspace — `.mitos/` and all —
        because the class of bug is that the COPY carries something.

        Two copies on purpose, and the SAME-NAME one is the row that bites. A renamed
        sandbox (`project` → `project-sandbox`) stays distinct even under a
        basename-only hash, so on its own it would pass against the implementation this
        phase replaces. A copy that keeps its name under a different parent — a
        `cp -r ~/work/proj /tmp/proj`, or a second `git clone` of one repo — is
        separated by nothing but the path digest.
        """
        source = tmp_path / "project"
        source.mkdir()
        cli.cmd_init(MitosConfig(str(source)))
        renamed = tmp_path / "project-sandbox"
        elsewhere = tmp_path / "sandbox" / "project"
        (tmp_path / "sandbox").mkdir()
        shutil.copytree(str(source), str(renamed))
        shutil.copytree(str(source), str(elsewhere))

        names = {
            MitosConfig(str(p)).qdrant_collection for p in (source, renamed, elsewhere)
        }
        assert len(names) == 3

    def test_every_route_to_one_directory_lands_on_one_collection(self, tmp_path) -> None:
        """A trailing slash, a `..`-bearing route, and a symlink are the same workspace.

        The symlink leg goes through a REAL `os.symlink` rather than relying on
        `/tmp` being one: on a machine where the temp root is a real directory, an
        abspath-only implementation passes every other spelling in this row. This is
        the leg that actually kills `abspath`.
        """
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        os.symlink(str(real), str(link))
        expected = default_collection_name(str(real))

        assert default_collection_name(str(real) + os.sep) == expected
        assert default_collection_name(str(tmp_path / "x" / ".." / "real")) == expected
        assert default_collection_name(str(link)) == expected
        assert MitosConfig(str(link)).qdrant_collection == expected

    def test_the_visible_segment_follows_the_resolved_directory(self, tmp_path) -> None:
        """Symlink resolution moves the readable segment too, not only the digest.

        Identity follows the REAL directory, so a workspace reached through
        `link -> real` is named for `real`. Worth pinning because a reader who expects
        canonicalization to affect only the hash will be surprised by the name.
        """
        real = tmp_path / "engine"
        real.mkdir()
        link = tmp_path / "shortcut"
        os.symlink(str(real), str(link))

        assert default_collection_name(str(link)).startswith("mitos-engine-")
        assert "shortcut" not in default_collection_name(str(link))

    def test_a_path_reached_without_a_registry_entry_is_namespaced_identically(
        self, tmp_path
    ) -> None:
        """The derivation is path-only: routing has no say in identity.

        A workspace reached by absolute path, never `init`ed and so never registered,
        derives exactly what the same directory would derive after registration. The
        registry is routing; the workspace is identity.
        """
        ws = tmp_path / "unregistered"
        ws.mkdir()
        before = MitosConfig(str(ws)).qdrant_collection

        cli.cmd_init(MitosConfig(str(ws)))

        assert MitosConfig(str(ws)).qdrant_collection == before

    def test_basenames_that_sanitize_to_nothing_stay_mutually_distinct(self, tmp_path) -> None:
        """P9's own case must not be the one that cross-contaminates.

        `проект`, `日本語` and `/` keep NO character through the sanitization — under a
        bare-`mitos` fallback all three, and every other wholly-non-Latin project on the
        machine, shared one collection. The digest is unconditional, so each is distinct
        and none is the shared name.
        """
        cyrillic = tmp_path / "проект"
        cjk = tmp_path / "日本語"
        cyrillic.mkdir()
        cjk.mkdir()
        names = [
            default_collection_name(str(cyrillic)),
            default_collection_name(str(cjk)),
            default_collection_name("/"),
        ]

        assert len(set(names)) == 3
        assert "mitos" not in names  # never the bare shared collection
        for name in names:
            # Two segments, not three and not `mitos--<digest>`: an empty visible
            # segment is omitted rather than emitted as a double dash.
            assert name.startswith("mitos-")
            assert len(name.split("-")) == 2

    def test_basenames_surviving_as_a_shared_fragment_stay_distinct(self, tmp_path) -> None:
        """The other non-injective class, and the larger half of it.

        Diacritics are DROPPED rather than folded, so `ąžuolas` survives as the
        fragment `uolas` and would collide with a real `uolas/` sibling; `Проект-1`
        survives as `1` and would collide with a real `1/`. These are the pairs a row
        written for "non-ASCII" as one class never constructs — the digest closes them
        because it never comes from the mangled basename.
        """
        for exotic, plain in (("ąžuolas", "uolas"), ("Проект-1", "1")):
            a = tmp_path / exotic
            b = tmp_path / plain
            a.mkdir()
            b.mkdir()
            assert default_collection_name(str(a)) != default_collection_name(str(b)), exotic
            # The collision is real at the visible segment; only the digest separates.
            assert default_collection_name(str(a)).rsplit("-", 1)[0] == (
                default_collection_name(str(b)).rsplit("-", 1)[0]
            ), exotic
            shutil.rmtree(str(a))
            shutil.rmtree(str(b))

    def test_a_path_carrying_a_non_utf8_byte_derives_a_name_instead_of_raising(self) -> None:
        """`os.fsencode`, not `.encode("utf-8")` — the one place the obvious spelling breaks.

        A POSIX path is bytes. `realpath` returns a `str` carrying surrogateescape code
        points for any byte that is not valid UTF-8, and `str.encode("utf-8")` refuses
        surrogates — so the obvious spelling raises `UnicodeEncodeError` from
        `MitosConfig.__init__`, taking down every verb on that workspace including the
        `mitos status` you would run to find out why.

        The assertion is on the SHAPE, never the digest: the point is that it returns.
        """
        path = os.fsdecode(b"/x/caf\xe9")

        name = default_collection_name(path)

        assert name.startswith("mitos-caf-")
        assert len(name.rsplit("-", 1)[1]) == 8

    def test_the_derivation_needs_no_filesystem(self, tmp_path) -> None:
        """Total for a path that does not exist — pure computation, no `exists()` probe.

        `realpath` resolves as far as it can and normalizes the rest, which is what
        keeps the function callable for a workspace not yet created (and what lets the
        exact-string row above use a path that cannot exist). Whether a path IS a
        workspace belongs to the caller asking it.
        """
        absent = tmp_path / "never" / "created"

        assert default_collection_name(str(absent)).startswith("mitos-created-")


# ---------------------------------------------------------------------------
# Determinism — the abspath↔realpath agreement 2a depends on
# ---------------------------------------------------------------------------

class TestCanonicalizationAgreement:
    """One workspace, three producers of its path, one collection."""

    def test_abspath_and_realpath_roots_agree_with_the_config(self, tmp_path) -> None:
        """`MitosConfig` roots with `abspath`; the registry canonicalizes with `realpath`.

        Both must land on one collection or a name-routed command addresses a different
        vector store than a path-routed one. Step 1 realpaths INTERNALLY, which is what
        reconciles the split — the derivation is the third party to it and the only one
        that can settle it.

        The symlinked leg lives in `TestDerivation` deliberately: on a machine whose
        temp root is a real directory this row alone passes under an abspath-only
        implementation, so it proves determinism, not canonicalization.
        """
        ws = tmp_path / "project"
        ws.mkdir()

        assert (
            default_collection_name(str(ws))
            == default_collection_name(os.path.realpath(str(ws)))
            == MitosConfig(str(ws)).qdrant_collection
        )

    def test_the_name_is_stable_across_constructions(self, tmp_path) -> None:
        """Re-derived every time and never persisted, so it must be reproducible."""
        ws = tmp_path / "project"
        ws.mkdir()
        cli.cmd_init(MitosConfig(str(ws)))

        assert MitosConfig(str(ws)).qdrant_collection == MitosConfig(str(ws)).qdrant_collection


# ---------------------------------------------------------------------------
# The retirement — I9's no-opt-out half
# ---------------------------------------------------------------------------

class TestRetirement:
    """No persisted value can change the resolved collection, and none warns."""

    def test_the_key_left_the_file_schema_for_the_retired_set(self) -> None:
        """The constant-level fact every behavioural row below depends on."""
        assert "qdrant_collection" not in CONFIG_SCHEMA
        assert "qdrant_collection" in RETIRED_CONFIG_KEYS

    @pytest.mark.parametrize(
        "raw, parsed",
        [
            ('"mitos-mitos-pub"', "mitos-mitos-pub"),   # the legacy auto-pin shape
            ('"custom_collection"', "custom_collection"),  # a genuinely hand-set value
            ("123", 123),                                # the WRONG TYPE, now inert too
            ('["a", "b"]', ["a", "b"]),                  # not even a scalar
        ],
    )
    def test_a_surviving_pin_of_any_shape_is_inert_and_silent(
        self, tmp_path, capsys, raw, parsed
    ) -> None:
        """Legacy, hand-set, or mistyped — the resolved name is the derived one.

        The mistyped row matters on its own: `qdrant_collection = 123` used to be a
        hard `ConfigError` (fatal before verb dispatch, so it bricked `mitos status`
        too). A retired key is skipped BEFORE the type check, so it is now inert like
        any other retired value. That is the pattern's existing behaviour for its four
        other members; pinned here so it is a decision rather than an accident.

        stderr must be EMPTY, not merely free of the word: a per-command warning on a
        line every pre-vision `init` wrote is exactly the false-alarm-on-every-call
        noise the retired-key silence exists to prevent.
        """
        ws = tmp_path / "project"
        ws.mkdir()
        _write_pin(ws, raw)

        config = MitosConfig(str(ws))

        assert config.qdrant_collection == default_collection_name(str(ws))
        assert config.qdrant_collection != parsed
        assert config.inert_file_keys["qdrant_collection"] == parsed
        assert capsys.readouterr().err.strip() == ""

    def test_a_clone_reached_by_path_with_no_init_resolves_to_its_own_collection(
        self, tmp_path, capsys
    ) -> None:
        """The state every `init`-time mechanism is structurally unable to reach.

        A repo that committed `.mitos/config.toml`, cloned to a second checkout and
        reached by absolute path: `init` never runs, so a strip predicate, a legacy
        shape recognizer, or a re-pin would all be unreachable here. Resolution-time
        retirement covers it because there is nothing left to recognize.
        """
        original = tmp_path / "checkout-a"
        clone = tmp_path / "checkout-b"
        original.mkdir()
        clone.mkdir()
        pinned_name = default_collection_name(str(original))
        _write_pin(clone, f'"{pinned_name}"')

        config = MitosConfig(str(clone))

        assert config.qdrant_collection == default_collection_name(str(clone))
        assert config.qdrant_collection != pinned_name
        assert capsys.readouterr().err.strip() == ""

    def test_a_fresh_init_writes_no_collection_line_at_all(self, tmp_path, capsys) -> None:
        """Nothing to inherit, because nothing is written. The other half of the fix.

        `init` seeds no `qdrant_collection` key and — D3 — writes no `config.toml`
        matcher, strip, or re-pin either. A fresh workspace therefore has no inert key
        of any kind, which is what keeps the note below a fact about the input.
        """
        ws = tmp_path / "project"
        ws.mkdir()
        cli.cmd_init(MitosConfig(str(ws)))
        capsys.readouterr()

        body = open(os.path.join(str(ws), ".mitos", "config.toml"), encoding="utf-8").read()
        assert "qdrant_collection" not in body
        assert MitosConfig(str(ws)).inert_file_keys == {}

    def test_the_derived_name_is_never_persisted_anywhere(self, tmp_path) -> None:
        """Not in `config.toml`, not in the registry — the property that makes copies safe.

        A persisted name is a name a copy can inherit. Asserting the ABSENCE of the
        resolved string from every file `init` writes is the general form of that, and
        it catches a future convenience write no `qdrant_collection`-keyed grep would.
        """
        ws = tmp_path / "project"
        ws.mkdir()
        cli.cmd_init(MitosConfig(str(ws)))
        derived = MitosConfig(str(ws)).qdrant_collection

        written = []
        for root, _dirs, files in os.walk(str(ws)):
            for name in files:
                path = os.path.join(root, name)
                try:
                    written.append(open(path, encoding="utf-8").read())
                except (OSError, UnicodeDecodeError):
                    pass  # the graph is binary; it holds no collection name either
        from mitos.config import global_registry_path

        if os.path.exists(global_registry_path()):
            written.append(open(global_registry_path(), encoding="utf-8").read())

        assert not [body for body in written if derived in body]

    def test_to_dict_still_carries_the_attribute_at_its_computed_default(self, tmp_path) -> None:
        """The retirement pattern's promise: the ATTRIBUTE survives, the override does not.

        Only the file-override capability was removed. Every consumer binding
        `config.qdrant_collection` — the four store constructions, the status rows, the
        corpus-provenance surface — is untouched, which is why this removal needed no
        migration.
        """
        ws = tmp_path / "project"
        ws.mkdir()
        _write_pin(ws)

        payload = MitosConfig(str(ws)).to_dict()

        assert payload["qdrant_collection"] == default_collection_name(str(ws))
        # Runtime-only, and it must not leak into a persistence boundary.
        assert "inert_file_keys" not in payload


# ---------------------------------------------------------------------------
# Reporting the survivor — one renderer, two surfaces, rendered text only
# ---------------------------------------------------------------------------

class TestInertPinReporting:
    """A note belongs beside the resolved value it claims to set."""

    def test_status_names_the_pin_in_every_collection_branch(self, tmp_path, monkeypatch, capsys) -> None:
        """A CONFIG fact, so it renders regardless of what Qdrant is doing.

        All four branches, because the nesting a reader reaches for first — inside the
        `collection_exists` arm — drops the note from the unreachable and absent cases,
        which are precisely where "which name is actually in force?" is the live
        question. A `coll_hint` cannot carry it either: hints render only when the mark
        is not ✓, and the healthy branch sets it to ✓.
        """
        ws = tmp_path / "project"
        ws.mkdir()
        _write_pin(ws)
        cli.cmd_init(MitosConfig(str(ws)))  # leaves the pinned file alone
        monkeypatch.setenv("GEMINI_API_KEY", "testkey")
        derived = MitosConfig(str(ws)).qdrant_collection

        branches = [
            _qdrant(False, False),                # Qdrant unreachable
            _qdrant(True, True, points=0),        # collection present
            _qdrant(True, False),                 # absent over a populated graph
        ]
        _commit_one(ws)
        monkeypatch.setattr(cli, "scroll_point_ids", _scroll(set()))
        for check in branches:
            monkeypatch.setattr(cli, "_check_qdrant", check)
            capsys.readouterr()
            cli.cmd_status(str(ws))
            out = capsys.readouterr().out
            assert 'qdrant_collection = \'mitos-mitos-pub\'' in out
            assert "inert legacy config" in out
            assert derived in out
            # A vector, not a shrug: one recovery that creates no state.
            assert "can be deleted" in out

    def test_status_renders_the_note_over_an_empty_graph_too(self, tmp_path, monkeypatch, capsys) -> None:
        """The fourth branch — absent collection, empty graph, the fresh-project state.

        Split out because it needs a workspace with no committed node, and because a
        pinned FRESH project is a real shape: a clone whose graph has not been rebuilt.
        """
        ws = tmp_path / "project"
        ws.mkdir()
        _write_pin(ws)
        cli.cmd_init(MitosConfig(str(ws)))
        capsys.readouterr()
        monkeypatch.setenv("GEMINI_API_KEY", "testkey")
        monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, False))

        assert cli.cmd_status(str(ws)) == 0  # still READY — the note is not a check

        out = capsys.readouterr().out
        assert "inert legacy config" in out
        assert "none recorded yet" in out  # the branch's own hint is intact

    def test_status_says_nothing_new_for_a_clean_workspace(self, tmp_path, monkeypatch, capsys) -> None:
        """The twin fixture: without it the rows above pass whether the note is conditional."""
        ws = tmp_path / "project"
        ws.mkdir()
        cli.cmd_init(MitosConfig(str(ws)))
        capsys.readouterr()
        monkeypatch.setenv("GEMINI_API_KEY", "testkey")
        monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, True, points=0))
        monkeypatch.setattr(cli, "scroll_point_ids", _scroll(set()))

        assert cli.cmd_status(str(ws)) == 0

        out = capsys.readouterr().out
        assert "qdrant_collection" not in out
        assert "inert legacy config" not in out

    def test_the_note_never_reaches_the_json_payload(self, tmp_path, monkeypatch, capsys) -> None:
        """Rendered text only — the `--json` shape for this is 4b's to settle.

        4b owns the deep report's payload, including whatever field names an inert
        legacy pin there. Adding one now would hand it a shape to revise for a consumer
        that does not exist, so the addition stops at the rendered surface and this row
        forces 4b to invert an assertion rather than notice a comment.
        """
        ws = tmp_path / "project"
        ws.mkdir()
        _write_pin(ws)
        cli.cmd_init(MitosConfig(str(ws)))
        capsys.readouterr()
        monkeypatch.setenv("GEMINI_API_KEY", "testkey")
        monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, True, points=0))
        monkeypatch.setattr(cli, "scroll_point_ids", _scroll(set()))

        assert cli.cmd_status(str(ws), as_json=True) == 0

        payload = capsys.readouterr().out
        # The full phrase, not the bare word: pytest builds the workspace path from
        # the test's own name and `status` echoes that path back.
        assert "inert legacy config" not in payload
        assert "mitos-mitos-pub" not in payload
        # The collection key still carries the RESOLVED name, and readiness is unmoved.
        parsed = json.loads(payload)
        assert parsed["collection"] == MitosConfig(str(ws)).qdrant_collection
        assert parsed["ready"] is True

    def test_the_pinned_value_is_rendered_escaped_not_raw(self, tmp_path, monkeypatch, capsys) -> None:
        """A config value is untrusted text, and this note prints it back to a terminal.

        A TOML basic string carries `\\n` and `\\u001b` escapes, so a raw interpolation
        would let a pinned value break the single-line layout or smuggle an ANSI
        sequence onto the operator's screen. The note renders through `repr`, which
        escapes both — pinned here so a future tidy to a plain interpolation reds.
        """
        ws = tmp_path / "project"
        ws.mkdir()
        _write_pin(ws, '"evil\\u001b[31m\\nREADY"')
        monkeypatch.setenv("GEMINI_API_KEY", "testkey")
        monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, True, points=0))
        monkeypatch.setattr(cli, "scroll_point_ids", _scroll(set()))

        cli.cmd_status(str(ws))

        out = capsys.readouterr().out
        assert "\x1b[31m" not in out  # no raw escape sequence reaches the terminal
        assert "\\x1b[31m" in out     # it is shown as text instead
        # And the note stays ONE line — the injected newline did not split it.
        note_lines = [ln for ln in out.splitlines() if "inert legacy config" in ln]
        assert len(note_lines) == 1
        assert "READY" in note_lines[0]

    def test_init_prints_the_note_once_on_a_pinned_workspace(self, tmp_path, capsys) -> None:
        """`init` is the other place a human is already asking about this workspace.

        It never rewrites `config.toml`, so the line stays and would otherwise silently
        disagree with the collection in force. A bare receipt line: no `reconcile`
        pointer and no old→new mapping — those belong to the phase whose echo makes
        them true.
        """
        ws = tmp_path / "project"
        ws.mkdir()
        _write_pin(ws)

        cli.cmd_init(MitosConfig(str(ws)))

        out = capsys.readouterr().out
        assert out.count("inert legacy config") == 1
        assert MitosConfig(str(ws)).qdrant_collection in out
        assert "reconcile" not in out.lower()

    def test_init_on_a_clean_workspace_prints_no_note(self, tmp_path, capsys) -> None:
        """The twin. Also the reason the 3e echo tripwire in `test_init.py` stays green:
        its workspaces are freshly `init`ed, so they carry no pin and the note is None.
        """
        ws = tmp_path / "project"
        ws.mkdir()

        cli.cmd_init(MitosConfig(str(ws)))

        out = capsys.readouterr().out
        assert "inert legacy config" not in out
        assert "qdrant_collection" not in out


# ---------------------------------------------------------------------------
# The shared-collection default, removed on the same argument
# ---------------------------------------------------------------------------

class TestStoreCollectionIsDeclared:
    """`QdrantVectorStore`'s `collection_name` has no default any more."""

    def test_constructing_without_a_collection_is_a_type_error(self) -> None:
        """The old default was `"mitos"` — the exact shared name this phase abolishes.

        Every construction site already passes it, so the default was dead weight
        today; what it could still do is land a FUTURE site silently in a namespace
        shared across projects. Declared, not defaulted — the property, not the callers
        that hold it now. (Same argument as `may_create` shipping required in 1c.)
        """
        with pytest.raises(TypeError):
            QdrantVectorStore("http://localhost:7333")

    def test_an_explicit_collection_still_binds_positionally(self) -> None:
        """Removing the default changed no existing caller: it stays positional."""
        store = QdrantVectorStore("http://localhost:7333/", "mitos-explicit-00000000")

        assert store.collection == "mitos-explicit-00000000"
        assert store.base_url == "http://localhost:7333"


def test_the_temp_root_is_not_a_symlink_here_so_the_symlinked_row_earns_its_keep() -> None:
    """A meta-row: it documents WHY `os.symlink` appears in this suite.

    On this machine `tempfile.mkdtemp()` returns a path identical to its `realpath`,
    so a "tmp_path vs realpath(tmp_path)" comparison passes under an abspath-only
    implementation — or under no canonicalization at all. If this assertion ever
    fails, the environment changed and the agreement row above became load-bearing on
    its own; the explicit `os.symlink` rows stay correct either way.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        assert os.path.realpath(tmpdir) == tmpdir
