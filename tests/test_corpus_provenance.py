"""Corpus-provenance stamping on the read surfaces.

Every recall answer names which corpus it came from (``project`` +
``collection`` + ``workspace``), so an empty or twilight result is never
ambiguous between "no precedent exists" and "you're standing in the wrong
workspace" (AX 2026-07-01: the reviewing cwd and a vision's decision store can
diverge). ``project`` answers that in the *caller's own* vocabulary — the name
it addressed, not only the path the tool resolved — so an agent holding several
projects sees a mis-aim immediately.

Pins the fields on the CLI JSON envelopes, the MCP twins, and the degraded
lexical fallback, plus the text-mode provenance line; and pins the four value
rules the ``project`` field follows, one per selector form.
"""

import asyncio
import json
import os
import subprocess
import sys

import pytest

from mitos import mcp_server, registry, routing
from mitos.cli import cmd_init, main
from mitos.config import MitosConfig
from mitos.recall import corpus_provenance, provenance_line


class TestHelpers:
    def test_corpus_provenance_fields(self, tmp_path):
        config = MitosConfig(workspace_dir=str(tmp_path))
        p = corpus_provenance(config)
        assert p["project"] == config.project
        assert p["collection"] == config.qdrant_collection
        assert p["workspace"] == config.workspace_dir

    def test_corpus_provenance_field_order_is_contractual(self, tmp_path):
        """Identity first, then the derived collection, then the location.

        Byte-parity between `scopes --json` and `list_scopes` compares serialized
        bodies, so a reordering on one surface only is a red — which means the
        order is part of the contract, not a rendering preference.
        """
        config = MitosConfig(workspace_dir=str(tmp_path))
        assert list(corpus_provenance(config)) == ["project", "collection", "workspace"]

    def test_provenance_line_carries_all_three(self, tmp_path):
        """The text line names every field the dict does, in the same order.

        Rewritten from a `f"corpus: {collection}"` prefix assertion: `project`
        goes first (a decision), which breaks that substring. What the row was
        protecting is that the tokens are *present*, so that is what it asserts —
        preserving a prefix by shuffling the new field to the tail would have hidden
        the decision instead of recording it.
        """
        # A named project so the three tokens are distinct strings: on a
        # selector-less config `project` IS `workspace_dir`, and an ordering
        # assertion over two identical strings proves nothing.
        config = MitosConfig(str(tmp_path), project="named-one")
        line = provenance_line(config)
        assert line.startswith("corpus: ")
        assert config.project in line
        assert config.qdrant_collection in line
        assert config.workspace_dir in line
        assert (line.index(config.project)
                < line.index(config.qdrant_collection)
                < line.index(config.workspace_dir))


@pytest.fixture
def workspace(tmp_path):
    """A minimal initialized workspace with one synced decision."""
    ws = str(tmp_path)
    env = {**os.environ, "GEMINI_API_KEY": "", "GOOGLE_API_KEY": "",
           "QDRANT_URL": "http://localhost:1"}

    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "mitos.cli", *args],
            capture_output=True, text=True, cwd=ws, env=env,
        )

    run("init")
    with open(os.path.join(ws, "decisions.md"), "a", encoding="utf-8") as f:
        f.write(
            "\n### provenance-test-decision\n"
            "**Decided:** Provenance headers ride every read surface.\n"
            "**Rejected paths:** Silent corpus ambiguity.\n"
            "**Scope:** testing\n"
            "**Date:** 2026-07-18\n"
        )
    run("sync", "--yes")
    return ws, run


def _both_tokens(stdout: str, config: MitosConfig) -> bool:
    """Whether a rendered provenance line names both the collection and the project.

    The shipped rows here asserted the `f"corpus: {collection}"` prefix, which
    `project`-first breaks. They are rewritten to check for the tokens they were
    actually protecting rather than for a field order the line no longer has —
    the alternative (moving `project` to the tail to keep the substring) would
    have made a decision look like an accident.
    """
    line = next((l for l in stdout.splitlines() if l.lstrip().startswith("corpus: ")
                 or "[corpus: " in l), "")
    return config.qdrant_collection in line and config.project in line


class TestCliJson:
    def test_list_json_carries_provenance(self, workspace):
        ws, run = workspace
        out = run("list", "--json")
        payload = json.loads(out.stdout)
        assert payload["project"] == os.path.abspath(ws)  # selector-less → the path
        assert payload["collection"] == MitosConfig(workspace_dir=ws).qdrant_collection
        assert payload["workspace"] == os.path.abspath(ws)

    def test_degraded_surface_json_carries_provenance(self, workspace):
        # Qdrant points at a dead port and no key is set → the lexical fallback
        # fires, and its envelope must still name the corpus.
        ws, run = workspace
        out = run("surface", "provenance headers", "--json")
        payload = json.loads(out.stdout)
        assert payload.get("degraded") == "lexical"
        assert payload["project"] == os.path.abspath(ws)
        assert payload["collection"] == MitosConfig(workspace_dir=ws).qdrant_collection
        assert payload["workspace"] == os.path.abspath(ws)

    def test_degraded_surface_text_has_provenance_line(self, workspace):
        ws, run = workspace
        out = run("surface", "provenance headers")
        assert _both_tokens(out.stdout, MitosConfig(workspace_dir=ws))

    def test_list_text_header_has_provenance(self, workspace):
        ws, run = workspace
        out = run("list")
        assert _both_tokens(out.stdout, MitosConfig(workspace_dir=ws))


class TestMcpTwins:
    def test_list_decisions_payload_carries_provenance(self, workspace, monkeypatch):
        ws, _ = workspace
        monkeypatch.chdir(ws)
        monkeypatch.setenv("QDRANT_URL", "http://localhost:1")
        monkeypatch.setenv("GEMINI_API_KEY", "")
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        payload = json.loads(mcp_server.list_decisions())
        assert payload["project"] == os.path.abspath(ws)
        assert payload["collection"] == MitosConfig(workspace_dir=ws).qdrant_collection
        assert payload["workspace"] == os.path.abspath(ws)

    def test_surface_decisions_degraded_carries_provenance(self, workspace, monkeypatch):
        ws, _ = workspace
        monkeypatch.chdir(ws)
        monkeypatch.setenv("QDRANT_URL", "http://localhost:1")
        monkeypatch.setenv("GEMINI_API_KEY", "")
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        payload = json.loads(mcp_server.surface_decisions("provenance headers"))
        assert payload.get("degraded") == "lexical"
        assert payload["project"] == os.path.abspath(ws)
        assert payload["collection"] == MitosConfig(workspace_dir=ws).qdrant_collection


# --------------------------------------------------------------------------- #
# The four value rules — one per selector form, all four on the same tool.
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=False)
def offline(monkeypatch):
    """Keyless and serviceless: nothing in these rows embeds, queries or spends."""
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:9")
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def _init_workspace(root) -> str:
    """Really initializes a workspace (init registers it) and returns its canonical root."""
    os.makedirs(str(root), exist_ok=True)
    cmd_init(MitosConfig(str(root)))
    return os.path.realpath(str(root))


def _write_registry(**entries) -> None:
    """Rewrites the registry from ``name → path`` keywords.

    Written whole rather than via ``registry.register`` because these rows need
    exact control of the *recorded spelling* (the symlink row's whole point) and
    because ``init`` has already registered each workspace under its basename —
    ``force`` waives a name collision but never path uniqueness, so a second name
    for the same path cannot be added through the writer. Conftest's autouse
    ``XDG_CONFIG_HOME`` redirect keeps this off the developer's real registry.
    """
    path = registry.registry_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("".join(f'"{n}" = "{p}"\n' for n, p in entries.items()))


def _echo(**kwargs) -> str:
    """The `project` field `list_scopes` stamps for a given selector."""
    return json.loads(mcp_server.list_scopes(**kwargs))["project"]


class TestEchoValueRules:
    """§4.7: the echoed `project` is defined for every selector form and never empty.

    All four rules collapse to one expression in `MitosConfig.__init__` because
    `resolve_project` already did the reverse lookup — so these rows are what keep
    that collapse honest, one per form, on a single tool.
    """

    def test_a_registered_name_echoes_that_name(self, tmp_path, offline, monkeypatch):
        target = _init_workspace(tmp_path / "target")
        _write_registry(**{"target-name": target})
        monkeypatch.chdir(tmp_path)

        assert _echo(project="target-name") == "target-name"

    def test_a_registered_path_in_path_form_echoes_its_registered_name(
        self, tmp_path, offline, monkeypatch
    ):
        """The reverse lookup, not the path — the caller addressed a known project.

        The registry is written by `register` (i.e. canonically) on purpose: with a
        hand-written non-canonical spelling `reverse_lookup`'s exact-equality
        contract would miss and this row would silently be testing the
        *unregistered* rule below while still reading as a pass.
        """
        target = _init_workspace(tmp_path / "target")
        _write_registry(known=target)  # the canonical spelling, as `register` writes it
        monkeypatch.chdir(tmp_path)

        assert registry.load()["known"] == target  # the premise, asserted
        assert _echo(project=target) == "known"

    def test_an_unregistered_path_echoes_the_absolute_path(
        self, tmp_path, offline, monkeypatch
    ):
        """The escape hatch — a fresh clone, a mid-setup project. Correct, not degraded."""
        target = _init_workspace(tmp_path / "target")
        _write_registry()  # drop init's registration
        monkeypatch.chdir(tmp_path)

        assert registry.load() == {}
        assert _echo(project=target) == target
        assert os.path.isabs(_echo(project=target))

    def test_a_selectorless_call_echoes_the_resolved_workspace_path(
        self, tmp_path, offline, monkeypatch
    ):
        """TRANSITIONAL (phase 3 only, inverted by 5b): no selector still resolves the
        cwd, and even that answer carries a defined echo — the unregistered-path rule.

        When 5b removes the fallback this row inverts to assert the refusal; it is
        here so that no response is ever unattributed mid-vision.
        """
        target = _init_workspace(tmp_path / "target")
        monkeypatch.chdir(target)

        assert _echo() == target

    def test_a_registered_project_reached_through_a_symlink_echoes_its_name(
        self, tmp_path, offline, monkeypatch
    ):
        """G1.3: the value must never be re-derived from the registry at stamp time.

        The fixture is built to *bite*, which took measuring rather than reasoning.
        `resolve_project` canonicalizes before the config exists, so on the obvious
        symlink fixtures — a link-form selector against a canonically-written
        registry — `config.workspace_dir` is already `realpath` and a locally-derived
        `reverse_lookup(config.workspace_dir)` would agree with the taken design.
        The hazard survives only where the registry's *recorded* value is
        non-canonical (a hand-edited link path — a tolerated state, the file is
        editable TOML by design) and the call comes in by NAME: then the taken
        design echoes `linked` while a derive-locally build echoes the realpath,
        because `abspath(realpath(link)) != the recorded link path` under
        `reverse_lookup`'s exact string equality.
        """
        real = _init_workspace(tmp_path / "real")
        link = str(tmp_path / "link")
        os.symlink(real, link)
        assert os.path.realpath(link) == real != link  # the fixture is not vacuous
        _write_registry(linked=link)  # the hand-edited NON-canonical spelling
        monkeypatch.chdir(tmp_path)

        # The premise: a locally-derived reverse lookup would MISS on this registry.
        resolved = routing.resolve_project("linked")
        assert resolved.root == real
        assert registry.reverse_lookup(resolved.root) is None

        assert _echo(project="linked") == "linked"

    def test_every_form_echoes_something(self, tmp_path, offline, monkeypatch):
        """Never-empty at the surface, across all four forms in one row.

        The constructor pins never-empty at the source (`tests/test_config.py`);
        this is the surface-level restatement — a `""` reaching an agent is the
        failure the whole envelope exists to prevent.
        """
        target = _init_workspace(tmp_path / "target")
        unregistered = _init_workspace(tmp_path / "loose")
        _write_registry(reg=target)  # `loose` deliberately left out
        monkeypatch.chdir(target)

        for selector in ({"project": "reg"}, {"project": target},
                         {"project": unregistered}, {}):
            assert _echo(**selector), f"empty echo for {selector}"


#: What each tool needs beyond its selector, so the completeness row below can
#: call every one of them. The tool *set* is computed from the live schema — a
#: seventh targeting tool lands here as a `KeyError`, not as silent non-coverage
#: (3c's lesson: a row parametrized over a hand-written set proves only what
#: someone remembered to put in the set).
TOOL_ARGS = {
    "surface_decisions": {"query": "anything"},
    "list_decisions": {},
    "list_scopes": {},
    "show_node": {"ident": "no-such-handle"},
    "query_decisions": {"query": "anything"},
    "record_decision": {"axiom": "The envelope rides every answer.",
                        "rejected_paths": "An unattributed answer.",
                        "scope": ["echo"], "slug": "envelope-probe"},
}


def test_every_targeting_tool_carries_the_envelope(tmp_path, offline, monkeypatch):
    """Criterion 4: all six tools that resolve a project echo it back.

    The set is read off `mcp.list_tools()` — membership by the `project`
    parameter, not by a remembered name — so a later phase adding a targeting
    tool is told to extend `TOOL_ARGS` rather than discovering afterwards that its
    tool was never covered. `list_projects` is correctly absent: it answers for
    the machine and resolves no project, so it carries no echo (§11's N/A row).
    """
    target = _init_workspace(tmp_path / "target")
    _write_registry(named=target)
    monkeypatch.chdir(tmp_path)

    schemas = {tool.name: (tool.inputSchema.get("properties") or {})
               for tool in asyncio.run(mcp_server.mcp.list_tools())}
    targeting = sorted(name for name, props in schemas.items() if "project" in props)

    assert targeting == sorted(TOOL_ARGS), "a targeting tool is uncovered here"

    # §11's explicit N/A row, so review does not read it as a seventh missing
    # stamp: `list_projects` takes no selector and resolves no workspace — it
    # answers for the MACHINE, and the echo obligation is on responses that
    # resolve a project. Asserted on its payload, not only its schema.
    assert "project" not in schemas["list_projects"]
    listing = json.loads(mcp_server.list_projects())
    assert not {"project", "collection", "workspace"} & set(listing)

    for name in targeting:
        payload = json.loads(
            getattr(mcp_server, name)(project="named", **TOOL_ARGS[name]))
        assert payload["project"] == "named", name
        assert payload["collection"] == MitosConfig(target).qdrant_collection, name
        assert payload["workspace"] == target, name


class TestTheRecordReceipt:
    """The write's receipt names the corpus it landed in — every outcome shape.

    The highest-value stamp in the set: a mis-aimed read wastes a turn, a
    mis-aimed write lands a real entry in another project's gold source. Stamped
    at the tool/CLI boundary on both surfaces, with `record_decision_entry`
    itself untouched.
    """

    def _cli_receipt(self, config, capsys, **kwargs):
        from mitos.cli import cmd_record
        capsys.readouterr()
        try:
            cmd_record(config, **kwargs)
        except SystemExit as exit_code:  # error → 1, needs_review → 2
            assert exit_code.code in (1, 2)
        return json.loads(capsys.readouterr().out)

    @pytest.mark.parametrize("outcome", ["created", "exists", "error"])
    def test_each_outcome_shape_carries_the_provenance(
        self, outcome, tmp_path, offline, monkeypatch, capsys
    ):
        """created / exists / error — the three shapes reachable offline.

        The `needs_review` pause is the fourth; it needs the armed neighbour
        review, so it is asserted where that machinery already lives
        (`tests/test_neighbor_review.py::test_cli_mcp_record_pause_parity`).
        """
        target = _init_workspace(tmp_path / "target")
        _write_registry(named=target)
        monkeypatch.chdir(tmp_path)
        entry = dict(axiom="Receipts name the corpus they wrote to.",
                     rejected_paths="A silent write into a neighbouring project.",
                     scope=["echo"], slug="receipts-name-the-corpus")
        if outcome == "exists":
            mcp_server.record_decision(project="named", **entry)
        if outcome == "error":
            entry["supersedes"] = "no-such-decision"

        receipt = json.loads(mcp_server.record_decision(project="named", **entry))

        expected = {"created": "created", "exists": "exists"}.get(outcome)
        if expected:
            assert receipt["status"] == expected
        else:
            assert "error" in receipt
        assert receipt["project"] == "named"
        assert receipt["collection"] == MitosConfig(target).qdrant_collection
        assert receipt["workspace"] == target

    def test_the_cli_twin_stamps_the_same_receipt(
        self, tmp_path, offline, monkeypatch, capsys
    ):
        """`mitos record --json` carries it too — `cmd_record`'s docstring claims
        the receipt is the same shape the MCP tool serializes, and a one-surface
        stamp would make that false."""
        target = _init_workspace(tmp_path / "target")
        _write_registry(named=target)
        config = MitosConfig(target, project="named")

        receipt = self._cli_receipt(
            config, capsys,
            axiom="Both surfaces stamp the write.",
            rejected="Stamping only the agent-facing one.",
            scope=["echo"], slug="both-surfaces-stamp", as_json=True)

        assert receipt["status"] == "created"
        assert receipt["project"] == "named"
        assert receipt["collection"] == config.qdrant_collection
        assert receipt["workspace"] == target

    def test_the_write_path_itself_is_not_stamped(self, tmp_path, offline):
        """The receipt `record_decision_entry` returns carries NO provenance.

        The locus is the boundary, deliberately: reaching into `sync.py` for the
        resolved config would put a routing concern inside the buffer-first +
        rollback contract. This row is what makes the boundary-only choice
        observable rather than merely stated.
        """
        from mitos.sync import MitosSyncManager

        target = _init_workspace(tmp_path / "target")
        receipt = MitosSyncManager(MitosConfig(target, project="named")).record_decision_entry(
            axiom="The write path returns a bare receipt.",
            rejected_paths="Stamping inside the locked commit.",
            scope=["echo"], slug="bare-receipt")

        assert receipt["status"] == "created"
        for field in ("project", "collection", "workspace"):
            assert field not in receipt


class TestTheTwoResolutionSitesArePinnedIndependently:
    """The value is derived at exactly two Tier-3 sites, and each needs its own row.

    Every other row in this module reaches the echo through the MCP surface (or
    through a config a test built directly), so dropping ``project=target.name``
    from ``cli.main()`` alone would leave the whole module green — and vice versa.
    These two rows are the injections' targets: each reds for its own site only.
    """

    def test_the_cli_resolution_site_carries_the_name(
        self, tmp_path, offline, monkeypatch, capsys
    ):
        """`mitos -p <name> scopes --json` echoes the NAME, through real `main()`.

        Driven through `main()` rather than by calling `cmd_scopes` with a
        hand-built config, because `main()`'s resolution block IS the subject: a
        verb never builds its own config in production.
        """
        target = _init_workspace(tmp_path / "target")
        _write_registry(named=target)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["mitos", "-p", "named", "scopes", "--json"])
        capsys.readouterr()  # drain the init banner

        main()

        payload = json.loads(capsys.readouterr().out)
        assert payload["project"] == "named"
        assert payload["workspace"] == target

    def test_the_mcp_resolution_site_carries_the_name(
        self, tmp_path, offline, monkeypatch
    ):
        """The mirror image, on the other surface, over the same registry shape."""
        target = _init_workspace(tmp_path / "target")
        _write_registry(named=target)
        monkeypatch.chdir(tmp_path)

        assert _echo(project="named") == "named"
