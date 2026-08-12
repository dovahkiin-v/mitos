"""Corpus-provenance stamping on the read surfaces.

Every recall answer names which corpus it came from (``project`` +
``collection`` + ``workspace``), so an empty or twilight result is never
ambiguous between "no precedent exists" and "you're standing in the wrong
workspace" (AX 2026-07-01: the reviewing cwd and a vision's decision store can
diverge). ``project`` answers that in the *caller's own* vocabulary — the name
it addressed, not only the path the tool resolved — so an agent holding several
projects sees a mis-aim immediately.

Pins the fields on the CLI JSON envelopes, the MCP twins, and the degraded
lexical fallback, plus the text-mode provenance line; and pins the value rules
the ``project`` field follows, one per selector form — three that answer, and the
selector-less form, which since 5b is a refusal rather than a fourth value.
"""

import asyncio
import json
import os
import sqlite3
import subprocess
import sys

import pytest

from mitos import cli, mcp_server, registry, routing
from mitos._agent_block import agent_block
from mitos.cli import _build_parser, cmd_init, main
from mitos.config import MitosConfig
from mitos.recall import corpus_provenance, provenance_line

# 3b's in-process driver and its subparser accessor, imported rather than
# re-derived: `_run` calls the real `main()` (which is where the CLI's single
# resolution site and the two dispatch-site echoes live), and `_subparsers` is
# already the tree's one spelling of "every verb the parser accepts".
from test_cli_selector import _run, _subparsers


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


#: The name the `workspace` fixture registers itself under. A FIXED name rather than
#: `init`'s basename default, because since 5a every row below addresses the
#: workspace by selector and the echo answers with the *registered name* — so the
#: assertions read as the §4.7 value rule they exercise instead of as an incidental
#: `tmp_path` basename.
WORKSPACE_NAME = "provenance-ws"


@pytest.fixture
def workspace(tmp_path):
    """A minimal initialized workspace with one synced decision.

    Its runner supplies `-p WORKSPACE_NAME` on every non-exempt verb: 5a removed
    the working-directory fallback, so `cwd=ws` alone no longer targets anything.
    `init` keeps running bare — it is selector-exempt and a supplied selector is
    refused.
    """
    ws = str(tmp_path)
    env = {**os.environ, "GEMINI_API_KEY": "", "GOOGLE_API_KEY": "",
           "QDRANT_URL": "http://localhost:1"}

    def _raw(*args):
        return subprocess.run(
            [sys.executable, "-m", "mitos.cli", *args],
            capture_output=True, text=True, cwd=ws, env=env,
        )

    def run(*args):
        return _raw("-p", WORKSPACE_NAME, *args)

    _raw("init", "--name", WORKSPACE_NAME)
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


def _echo_line(text: str) -> str:
    """The one rendered corpus echo line in ``text``, or ``""`` if there is none.

    Both spellings: the standalone `corpus: …` line the CLI's text sites print, and
    the bracketed `[corpus: …]` form the three shipped read headers carry inline.
    Extracting the LINE is the whole defence — several in-scope verbs already print
    a workspace path of their own, so a bare `workspace_dir in stdout` is green with
    no echo at all (3b's finding: an assertion satisfied by a neighbouring line).
    """
    return next((l for l in text.splitlines() if l.lstrip().startswith("corpus: ")
                 or "[corpus: " in l), "")


def _both_tokens(stdout: str, config: MitosConfig) -> bool:
    """Whether a rendered provenance line names all three fields of the corpus.

    The shipped rows here asserted the `f"corpus: {collection}"` prefix, which
    `project`-first breaks. They are rewritten to check for the tokens they were
    actually protecting rather than for a field order the line no longer has —
    the alternative (moving `project` to the tail to keep the substring) would
    have made a decision look like an accident.

    3e extends it from two fields to three: `workspace` is a field of the rendered
    line and of every `--json` payload, and a helper that checked two of three left
    the third pinned only on the JSON side.
    """
    line = _echo_line(stdout)
    return (config.qdrant_collection in line and config.project in line
            and config.workspace_dir in line)


class TestCliJson:
    """Entry-007's four CLI-side members, INVERTED at 5a — not deleted.

    Each used to assert `project == abspath(ws)`, the transitional rule for a
    selector-less call. There is no selector-less call left on these verbs, so the
    rule they exercise moved one row down §4.7's table: a **registered path**
    reverse-looks-up, so the echo names the registered name. The assertion is the
    only statement that no response is unattributed; turning it around is what
    keeps that statement true rather than merely absent.
    """

    def test_list_json_carries_provenance(self, workspace):
        ws, run = workspace
        out = run("list", "--json")
        payload = json.loads(out.stdout)
        assert payload["project"] == WORKSPACE_NAME   # registered name → the name
        assert payload["collection"] == MitosConfig(workspace_dir=ws).qdrant_collection
        assert payload["workspace"] == os.path.abspath(ws)

    def test_degraded_surface_json_carries_provenance(self, workspace):
        # Qdrant points at a dead port and no key is set → the lexical fallback
        # fires, and its envelope must still name the corpus.
        ws, run = workspace
        out = run("surface", "provenance headers", "--json")
        payload = json.loads(out.stdout)
        assert payload.get("degraded") == "lexical"
        assert payload["project"] == WORKSPACE_NAME
        assert payload["collection"] == MitosConfig(workspace_dir=ws).qdrant_collection
        assert payload["workspace"] == os.path.abspath(ws)

    def test_degraded_surface_text_has_provenance_line(self, workspace):
        ws, run = workspace
        out = run("surface", "provenance headers")
        assert _both_tokens(out.stdout,
                            MitosConfig(workspace_dir=ws, project=WORKSPACE_NAME))

    def test_list_text_header_has_provenance(self, workspace):
        ws, run = workspace
        out = run("list")
        assert _both_tokens(out.stdout,
                            MitosConfig(workspace_dir=ws, project=WORKSPACE_NAME))


class TestMcpTwins:
    # Both rows name the workspace in PATH form and assert the echo comes back as
    # the registered NAME: the fixture's `mitos init --name provenance-ws`
    # registered it, so `resolve_project` reverse-looks-up and §4.7's second value
    # rule applies. Before 5b these called selectorless from a `chdir(ws)`, and the
    # path form was what a cwd-resolved config echoed — the chdir is gone with the
    # fallback, which is also what makes these the CLI twins two screens up rather
    # than a differently-worded sibling.

    def test_list_decisions_payload_carries_provenance(self, workspace, monkeypatch):
        ws, _ = workspace
        monkeypatch.setenv("QDRANT_URL", "http://localhost:1")
        monkeypatch.setenv("GEMINI_API_KEY", "")
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        payload = json.loads(mcp_server.list_decisions(project=ws))
        assert payload["project"] == WORKSPACE_NAME
        assert payload["collection"] == MitosConfig(workspace_dir=ws).qdrant_collection
        assert payload["workspace"] == os.path.abspath(ws)

    def test_surface_decisions_degraded_carries_provenance(self, workspace, monkeypatch):
        ws, _ = workspace
        monkeypatch.setenv("QDRANT_URL", "http://localhost:1")
        monkeypatch.setenv("GEMINI_API_KEY", "")
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        payload = json.loads(
            mcp_server.surface_decisions("provenance headers", project=ws))
        assert payload.get("degraded") == "lexical"
        assert payload["project"] == WORKSPACE_NAME
        assert payload["collection"] == MitosConfig(workspace_dir=ws).qdrant_collection


# --------------------------------------------------------------------------- #
# The value rules — one per selector form, all of them on the same tool. Three
# forms answer since 5b; the fourth (no selector) is a refusal, and has its own
# row here rather than being dropped.
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
    that collapse honest, one per form, on a single tool. The fourth rule (no
    selector → the workspace path) is the constructor's alone since 5b: no call on
    this surface reaches it, and the row that used to exercise it now asserts the
    refusal in its place.
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

    def test_a_selectorless_call_is_refused_from_inside_a_workspace(
        self, tmp_path, offline, monkeypatch
    ):
        """Entry-007's third tripwire, inverted at 5b: there is no fourth form here.

        Phase 3 asserted that a selector-less call still resolved the cwd and that
        even *that* answer carried a defined echo, so no response was unattributed
        mid-vision. 5b deletes the fallback, so the rule turns over rather than
        lapsing: a call naming no project has no workspace to answer for, and the
        §4.5 anatomy stands where the echo would have been.

        The cwd is deliberately the workspace itself — the one arrangement under
        which a surviving fallback would look like a pass. Nothing about the
        directory the process sits in reaches the answer.
        """
        target = _init_workspace(tmp_path / "target")
        monkeypatch.chdir(target)

        with pytest.raises(mcp_server._RenderedToolError) as raised:
            _echo()

        body = str(raised.value)
        assert "no project was named" in body
        assert target not in body          # the cwd is not offered as a default
        assert "list_projects()" in body   # the recovery an agent can actually take

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

        Three forms, not four: §4.7's fourth value rule (no selector → the
        workspace path) is a *constructor* rule and still holds there, but since
        5b no call on this surface can reach it — a selector-less call has no
        answer to stamp. That case is asserted one row up as a refusal rather than
        dropped from the loop, which is why this docstring says the number out
        loud.
        """
        target = _init_workspace(tmp_path / "target")
        unregistered = _init_workspace(tmp_path / "loose")
        _write_registry(reg=target)  # `loose` deliberately left out
        monkeypatch.chdir(target)

        for selector in ({"project": "reg"}, {"project": target},
                         {"project": unregistered}):
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


def test_the_exact_slug_exit_carries_the_envelope(tmp_path, offline, monkeypatch):
    """A verb is stamped per EXIT, not per verb — `query_decisions`' second one.

    Lives BESIDE the require-list above rather than inside `TOOL_ARGS`, and that
    placement is the whole lesson. `TOOL_ARGS` is per-verb by construction — one
    entry per tool, fenced by a set-equality against `mcp.list_tools()` — so a
    second call-form for one tool reds the completeness assertion instead of
    landing. Its `query_decisions` entry passes a string matching no slug, which
    never reaches the dereference branch, so the branch shipped unstamped under a
    green per-verb fixture (ADR
    `per-exit-coverage-rows-live-beside-the-per-verb-require-list-not-in-it`).

    The dereference exit is the one an agent acts on MOST directly: it named a
    handle and got the decision in full, with no ranking to second-guess and no
    empty result to interpret — which is why an answer from the wrong corpus is
    the one that never looks wrong.
    """
    name, target = _register_workspace(tmp_path, monkeypatch)

    payload = json.loads(mcp_server.query_decisions(
        query="the-cli-echo-names-its-corpus", project=name))

    # Non-vacuity FIRST: prove the call entered the dereference branch. A row
    # asserting only the stamp is green against the semantic branch — which is
    # exactly how the defect survived — so pin the shape that tells them apart:
    # the dereference exit returns a top-level `state` and no `matches`, while
    # both the ranked envelope and the lexical-degraded one carry `matches`.
    assert payload["slug"] == "the-cli-echo-names-its-corpus"
    assert "state" in payload
    assert "matches" not in payload

    assert payload["project"] == name
    assert payload["collection"] == MitosConfig(target).qdrant_collection
    assert payload["workspace"] == target

    # Provenance, and deliberately NOT confidence — the two answer different
    # questions and only one of them is asked here. Provenance answers *which
    # corpus replied*, which is ambiguous on every exit; a band answers *how well
    # the semantic ranking did*, which on a named handle is not a question, so a
    # band here reads as "this decision is doubtful". Pinned rather than left to
    # the prose: the band lands on this verb's OTHER exits next, in this same
    # function, and a phase wiring it envelope-by-envelope meets an assertion it
    # has to invert on purpose instead of a paragraph it has to remember.
    assert "confidence" not in payload
    assert "note" not in payload


@pytest.mark.parametrize("tool,kwargs", [
    ("query_decisions", {"query": "q", "depth": "trace"}),
    ("list_decisions", {"brief": True, "oneline": True}),
])
def test_an_argument_fault_carries_no_envelope(tool, kwargs, tmp_path, offline, monkeypatch):
    """The two deliberate non-stamps, asserted as rows rather than left as silences.

    `query_decisions`' depth-mode refusal and `list_decisions`' `brief and oneline`
    refusal are ARGUMENT faults, answered before `_target_config` runs — *"the
    caller's mistake is in the depth tier, not in the target, and resolving first
    would answer a different question than the one they got wrong"*, in the shipped
    comment's own words. No config exists at those exits, so there is nothing to
    stamp from: they are carve-outs, not gaps, and the absence is stated here so
    review does not read it as two missed exits.

    Their CLI twins diverge legitimately for the same resolution-order reason —
    `cmd_query`'s depth-error `--json` arm DOES stamp, because `main()` resolved
    before dispatch.

    The ordering half is owned by
    `tests/test_mcp_selector.py::test_an_argument_fault_is_answered_before_the_project_is_resolved`
    and is cited, not duplicated: a second copy is an enumeration waiting to drift.
    """
    name, _target = _register_workspace(tmp_path, monkeypatch, decisions=False)

    payload = json.loads(getattr(mcp_server, tool)(project=name, **kwargs))

    assert "error" in payload
    assert not {"project", "collection", "workspace"} & set(payload)


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


#: What each require-list verb needs beyond its selector, so the per-verb row below
#: can actually invoke it. Hand-written on purpose — argparse knows the arity, not a
#: value that makes the verb run — and fenced by the computed require-list row, so a
#: verb added in a later phase lands as a failing set comparison rather than as
#: silent non-coverage. (3d's `TOOL_ARGS` fence, with the parser standing in for
#: `mcp.list_tools()`.)
#:
#: Every entry runs OFFLINE: no key, no reachable Qdrant, no judge. Several verbs
#: therefore answer with a refusal rather than a report — which is the point, since
#: a refusal is a response and §4.7's obligation is on responses.
CLI_VERB_ARGS = {
    "agent-block": [],                     # positional path is optional; it is a selector source
    "capture": ["a raw architectural thought"],   # keyless refusal, on stdout
    "check": [],                           # embed absent + live decisions ⇒ fail-closed, on stderr
    "cutover": [],                         # already-V1a graph ⇒ the cheap no-op
    "import": ["legacy.md"],               # a cwd-rooted FILE, not a selector (3b)
    "list": [],
    "list_decisions": [],
    "list_scopes": [],
    "open-questions": [],
    "query": ["a claim to place"],
    "query_decisions": ["a claim to place"],
    "rebuild": [],                         # no TTY ⇒ refuses to prompt, after the echo
    "reconcile": [],
    "record": ["An axiom recorded by the per-verb row.",
               "--slug", "per-verb-row-record",
               "--rejected", "Leaving the corpus unnamed."],
    "record_decision": ["An axiom recorded through the alias.",
                        "--slug", "per-verb-row-record-alias",
                        "--rejected", "Leaving the corpus unnamed."],
    "render": [],
    "restore-source": ["--all-graph-only"],  # --slug/--all-graph-only is a required mutex
    "scopes": [],
    "set-key": ["a-value-that-is-not-a-key"],
    "show": ["no-such-handle"],
    "surface": ["a topic to recall"],
    "surface_decisions": ["a topic to recall"],
    "sync": [],
}

#: In the computed require-list, satisfies §4.7 in-body rather than with the
#: standard echo line: its report IS a statement about one workspace, and 4b owns
#: the header gaining the registered name. Carved out by NAME and reason, exactly as
#: 3d's MCP row carves out `list_projects`, so the gap is a row rather than a
#: silence.
CLI_ECHO_CARVE_OUTS = {"status"}


def _register_workspace(tmp_path, monkeypatch, *, name="named", decisions=True):
    """A registered workspace holding one committed decision, addressed by name.

    The working directory is the workspace's PARENT, which is deliberately not a
    workspace itself: a row that quietly resolved the cwd instead of the selector
    would fail loudly here rather than assert against the wrong corpus and pass.
    """
    target = _init_workspace(tmp_path / "target")
    _write_registry(**{name: target})
    monkeypatch.chdir(tmp_path)
    if decisions:
        # Seeded through `record`, not `sync`: `perform_sync` refuses outright
        # without a GEMINI key, so a keyless `sync` commits nothing and every row
        # downstream would silently assert against an EMPTY graph. `record` commits
        # and defers the embedding, which is exactly the offline write path.
        _run(["-p", name, "record", "Every CLI verb names the corpus it answered from.",
              "--slug", "the-cli-echo-names-its-corpus",
              "--rejected", "A success line that names no target.",
              "--scope", "echo"])
    return name, target


class TestTheCliRequireList:
    """T14's CLI half: every verb the parser accepts, minus the exempt three.

    T14's MCP half is already discharged by
    ``test_every_targeting_tool_carries_the_envelope`` above, which reads the tool
    set off ``mcp.list_tools()`` and asserts all three *values* on every targeting
    tool's payload — it is cited rather than duplicated here.
    """

    def test_the_require_list_is_computed_off_the_parser(self):
        """The set under test is derived, not remembered.

        ``VERB_ARGS`` is hand-written because argparse cannot supply a value that
        makes a verb *run*; the SET is computed, so a verb added in a later phase
        reds this row instead of shipping unstamped. Verified to bite by deleting a
        name from ``CLI_VERB_ARGS``.

        The five shipped aliases (``query_decisions``, ``surface_decisions``,
        ``list_decisions``, ``list_scopes``, ``record_decision``) are members in
        their own right: they are separate ``choices`` keys, and a
        canonical-names-only list would leave five invocable verbs unpinned on
        exactly the agent-facing path.
        """
        require = set(_subparsers(_build_parser())) - set(cli._SELECTOR_EXEMPT_VERBS)

        assert require == set(CLI_VERB_ARGS) | CLI_ECHO_CARVE_OUTS
        assert {"query_decisions", "surface_decisions", "list_decisions",
                "list_scopes", "record_decision"} <= set(CLI_VERB_ARGS)

    @pytest.mark.parametrize("verb", sorted(CLI_VERB_ARGS))
    def test_every_require_list_verb_names_its_corpus(
        self, verb, tmp_path, offline, monkeypatch, capsys
    ):
        """``mitos -p named <verb>`` names project + collection + workspace.

        Driven through real ``main()``: a verb never builds its own config in
        production, and the two dispatch-site echoes (``set-key``, ``agent-block``)
        exist only there.

        The assertion targets the rendered echo LINE, never "the three strings
        appear somewhere in stdout" — on a verb whose report already prints a
        workspace path that check is green with no echo at all (3b's finding). The
        channel is not pinned here (several verbs answer on stderr, and which one
        does is per-branch); the channel discipline has its own rows below.
        """
        name, target = _register_workspace(tmp_path, monkeypatch)
        (tmp_path / "legacy.md").write_text("no headings here\n", encoding="utf-8")
        config = MitosConfig(target, project=name)
        capsys.readouterr()

        _run(["-p", name, verb, *CLI_VERB_ARGS[verb]])

        captured = capsys.readouterr()
        assert _both_tokens(captured.out + captured.err, config), (
            f"{verb} named no corpus:\n--- stdout ---\n{captured.out}"
            f"\n--- stderr ---\n{captured.err}")

    def test_a_json_verb_carries_the_three_keys_not_a_text_line(
        self, tmp_path, offline, monkeypatch, capsys
    ):
        """``--json`` gets FIELDS; the text line must not leak onto that branch.

        The independent half of the pair: a text echo printed unconditionally
        corrupts every ``json.loads`` in the tree, and a stamp added only to the
        text branch leaves the machine surface unattributed. Both are pinned, on
        one verb whose two branches this phase wrote from scratch.
        """
        name, target = _register_workspace(tmp_path, monkeypatch)
        config = MitosConfig(target, project=name)
        capsys.readouterr()

        _run(["-p", name, "open-questions", "--json"])

        out = capsys.readouterr().out
        payload = json.loads(out)          # reds if a text line leaked above it
        assert payload["project"] == name
        assert payload["collection"] == config.qdrant_collection
        assert payload["workspace"] == target
        assert _echo_line(out) == ""       # …and no rendered line on this branch

    def test_a_path_form_selector_binds_the_registered_and_unregistered_rules(
        self, tmp_path, offline, monkeypatch, capsys
    ):
        """T14's named clause — the two value rules that diverge only here.

        A *registered* workspace addressed by absolute path echoes its registered
        NAME (the reverse lookup `resolve_project` already did); an *unregistered*
        one addressed the same way echoes the absolute PATH. Every other selector
        form collapses the two, so this is the row that tells them apart on the CLI
        text surface — W4's last consumer.
        """
        registered = _init_workspace(tmp_path / "registered")
        unregistered = _init_workspace(tmp_path / "loose")
        _write_registry(known=registered)   # `loose` deliberately dropped
        monkeypatch.chdir(tmp_path)
        assert registry.reverse_lookup(unregistered) is None  # the premise, asserted
        capsys.readouterr()

        _run(["-p", registered, "scopes"])
        assert "known" in _echo_line(capsys.readouterr().out)

        _run(["-p", unregistered, "scopes"])
        line = _echo_line(capsys.readouterr().out)
        assert unregistered in line and "known" not in line


class TestTheEchoRidesTheResponsesOwnChannel:
    """D3: a handler answering on stderr carries its echo there, not on stdout.

    An echo pinned to stdout is invisible to a caller reading the stderr answer —
    which is precisely the agent-facing path on ``record``'s pause and on every
    fail-closed refusal.
    """

    def test_record_reports_its_corpus_on_stdout(
        self, tmp_path, offline, monkeypatch, capsys
    ):
        name, target = _register_workspace(tmp_path, monkeypatch)
        config = MitosConfig(target, project=name)
        capsys.readouterr()

        _run(["-p", name, "record", "A decision that lands.",
              "--slug", "a-decision-that-lands",
              "--rejected", "Writing into a neighbouring project."])

        captured = capsys.readouterr()
        assert _both_tokens(captured.out, config)
        assert "Recorded decision" in captured.out

    def test_record_reports_its_corpus_on_stderr_when_it_refuses(
        self, tmp_path, offline, monkeypatch, capsys
    ):
        """The write did NOT land — and that answer rides stderr, so the echo does.

        A citation to a non-existent slug is the cheapest offline refusal shape.
        """
        name, target = _register_workspace(tmp_path, monkeypatch)
        config = MitosConfig(target, project=name)
        capsys.readouterr()

        _run(["-p", name, "record", "A decision that cannot land.",
              "--slug", "a-decision-that-cannot-land",
              "--rejected", "A silent failure.",
              "--supersedes", "no-such-decision"])

        captured = capsys.readouterr()
        assert "Record failed" in captured.err
        assert _both_tokens(captured.err, config)
        assert _echo_line(captured.out) == ""

    def test_check_reports_its_corpus_on_stderr_when_it_cannot_run(
        self, tmp_path, offline, monkeypatch, capsys
    ):
        """The fail-closed refusal is inside the handler, so it echoes (D5).

        Keyless with live decisions ⇒ `check` cannot audit them and refuses on
        stderr, exit 2. Contrast `main()`'s pre-dispatch refusals, which carry no
        echo because no config exists yet to name.
        """
        name, target = _register_workspace(tmp_path, monkeypatch)
        config = MitosConfig(target, project=name)
        capsys.readouterr()

        code = _run(["-p", name, "check"])

        captured = capsys.readouterr()
        assert code == 2
        assert "could not run" in captured.err
        assert _both_tokens(captured.err, config)


class TestTheTwoCarveOuts:
    """§4.7's two exceptions — and they are different exceptions.

    ``check --staged``'s free path is an **obligation** carve-out (the echo rides
    output the surface already emits; a pre-commit hook's near-silence is a shipped
    feature, and its target is a literal in a committed hook so it cannot drift).
    ``agent-block`` is a **channel** carve-out (its stdout is a travelling artifact
    the tool's own text tells the reader to paste into a committed file, so a
    resolved name and a path-hashed collection must not land inside it). Collapsing
    the two into "these verbs are special" loses both arguments.
    """

    def test_agent_block_stdout_stays_byte_identical(
        self, tmp_path, offline, monkeypatch, capsys
    ):
        """The paste-ready block, byte for byte, with the echo out of band.

        Asserted by EQUALITY against `agent_block()` rather than by containment, so
        the row cannot degrade into "the block contains the block". Driven through
        `main()` because the echo lives at the dispatch site — a row calling
        `cmd_agent_block` directly exercises nothing this phase wrote.
        """
        name, target = _register_workspace(tmp_path, monkeypatch, decisions=False)
        config = MitosConfig(target, project=name)
        capsys.readouterr()

        _run(["-p", name, "agent-block"])

        captured = capsys.readouterr()
        assert captured.out == agent_block() + "\n"   # `print` adds exactly one
        assert _both_tokens(captured.err, config)

    def test_agent_block_check_uses_the_same_channel(
        self, tmp_path, offline, monkeypatch, capsys
    ):
        """One rule, both forms. ``--check``'s report is a diagnostic and could
        safely take stdout — but two channels for one verb is a drift seam for no
        gain, and the single rule is what keeps the plain form's stdout clean."""
        name, target = _register_workspace(tmp_path, monkeypatch, decisions=False)
        config = MitosConfig(target, project=name)
        capsys.readouterr()

        _run(["-p", name, "agent-block", "--check"])

        captured = capsys.readouterr()
        assert _echo_line(captured.out) == ""
        assert _both_tokens(captured.err, config)

    def test_set_key_echoes_in_its_project_form(
        self, tmp_path, offline, monkeypatch, capsys
    ):
        """A credential landing in the wrong project's ``.env`` is the mis-aim this
        names. The echo is at the dispatch site because ``cmd_set_key`` takes a bare
        path, not a config — a shape W24 signed deliberately."""
        name, target = _register_workspace(tmp_path, monkeypatch, decisions=False)
        config = MitosConfig(target, project=name)
        capsys.readouterr()

        _run(["-p", name, "set-key", "not-a-real-key"])

        out = capsys.readouterr().out
        assert _both_tokens(out, config)
        assert "Stored GEMINI_API_KEY" in out

    def test_set_key_global_names_no_corpus(self, tmp_path, offline, monkeypatch,
                                            capsys):
        """``--global`` writes the machine-wide ``.env`` shared by every project, so
        there is no corpus to name — and the verb is selector-exempt in that form,
        which is why this row is driven without one."""
        target = _init_workspace(tmp_path / "target")
        monkeypatch.chdir(target)
        capsys.readouterr()

        _run(["set-key", "--global", "not-a-real-key"])

        out = capsys.readouterr().out
        assert "globally (all projects)" in out
        assert _echo_line(out) == ""


#: `--json` verbs whose payload is knowingly NOT parseable at this commit, with the
#: owner of the fix. One row across every JSON verb catches "a text line leaked above
#: the object" for all of them at once — but a silent exclusion would read as "all
#: payloads parse clean", which is false, so an exclusion is named here and logged
#: by the row itself.
#:
#: **Empty since 5c**, and the emptiness is the gate. Its one member was
#: `reconcile`: `MitosSyncManager.reconcile_embeddings` printed its provider-down
#: line to stdout unconditionally and `as_json`-blind, so `mitos reconcile --json`
#: had emitted a stray text line above its object since before this vision. 3e
#: found it and was fenced out by its own empty-`sync.py`-diff proof (T8b); 5c is
#: the next phase T8b lists as touching `sync.py`, and it moved the line to stderr.
#: Removing the member IS the regression gate — the row below asserts
#: `skipped == sorted(JSON_PARSE_EXCLUSIONS)`, so a stale member reds rather than
#: rots, and a re-broken verb reds on the `json.loads`.
JSON_PARSE_EXCLUSIONS = {}


def test_no_json_payload_carries_a_stray_text_line(tmp_path, offline, monkeypatch,
                                                   capsys, caplog):
    """Every `--json` verb on the require-list emits a parseable object and nothing else.

    The independent half of gotcha 5 for the whole surface at once: a leading text
    echo that is not gated off the `--json` branch corrupts the payload, and one
    verb-at-a-time row would catch it one verb at a time. `json.loads` over the
    whole of stdout is the strictest available assertion — it fails on so much as a
    blank prefix line.
    """
    name, target = _register_workspace(tmp_path, monkeypatch)
    parser_choices = _subparsers(_build_parser())
    json_verbs = sorted(
        verb for verb in set(parser_choices) - set(cli._SELECTOR_EXEMPT_VERBS)
        if any(opt == "--json"
               for action in parser_choices[verb]._actions
               for opt in action.option_strings)
    )
    checked, skipped = [], []

    for verb in json_verbs:
        if verb in JSON_PARSE_EXCLUSIONS:
            skipped.append(verb)
            continue
        capsys.readouterr()
        _run(["-p", name, verb, *CLI_VERB_ARGS.get(verb, []), "--json"])
        out = capsys.readouterr().out
        try:
            json.loads(out)
        except json.JSONDecodeError as exc:
            pytest.fail(f"`mitos {verb} --json` did not emit one object ({exc}):\n{out}")
        checked.append(verb)

    # No silent caps: the row states what it covered and what it did not.
    assert len(checked) >= 16, checked
    assert skipped == sorted(JSON_PARSE_EXCLUSIONS), (
        f"checked {checked}; excluded {skipped} — see JSON_PARSE_EXCLUSIONS")


def test_the_projects_verb_carries_no_echo(tmp_path, offline, monkeypatch, capsys):
    """§11's explicit CLI N/A row, mirroring 3d's on `list_projects`.

    `mitos projects` resolves no project — it answers for the MACHINE — so it
    carries no echo, on either branch. Asserted as a ROW rather than left as a
    silence, so review does not read the absence as a missed verb.
    """
    target = _init_workspace(tmp_path / "target")
    _write_registry(named=target)
    monkeypatch.chdir(tmp_path)
    capsys.readouterr()

    _run(["projects"])
    assert _echo_line(capsys.readouterr().out) == ""

    _run(["projects", "--json"])
    listing = json.loads(capsys.readouterr().out)
    assert not {"project", "collection", "workspace"} & set(listing)


def test_a_leading_echo_reaches_a_combined_pipe_before_a_boundary_error(tmp_path):
    """The echo survives a handler that raises one line later.

    Several verbs take the echo at the TOP of the handler, before the store
    construction — and a fault there is rendered by `main()`'s boundary on stderr,
    which is unbuffered while a piped stdout is not. Measured before the fix:
    `mitos scopes | cat` on an unopenable graph printed `Error: …` ABOVE the echo,
    so the reader's opening line was a refusal for a command that had in fact named
    its corpus first. `_echo_corpus` flushes for exactly this; the row exists
    because `capsys` keeps the streams apart and cannot see the inversion, and
    because the fix is one line a future tidy would remove without hesitating.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    env = {
        **os.environ,
        "MITOS_NO_UPDATE_CHECK": "1",
        "XDG_CONFIG_HOME": str(tmp_path / "xdg_config"),
        "GEMINI_API_KEY": "", "GOOGLE_API_KEY": "",
        "QDRANT_URL": "http://localhost:1",
    }

    def run(*argv):
        return subprocess.run(
            [sys.executable, "-m", "mitos.cli", *argv], cwd=str(workspace), env=env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

    run("init")
    # An unopenable graph: `GraphStore(config.db_path)` raises one statement after
    # the echo, and the refusal renders outside the handler (so it carries none).
    graph = workspace / ".mitos" / "graph.sqlite"
    graph.unlink()
    graph.mkdir()

    done = run("-p", str(workspace), "scopes")

    combined = done.stdout
    assert "Error:" in combined
    assert "corpus: " in combined
    assert combined.index("corpus: ") < combined.index("Error:"), (
        f"the boundary error overtook the leading echo:\n{combined}")


def test_the_report_reaches_a_combined_pipe_before_the_refusals(tmp_path):
    """Ordering, proved where it can actually break: ONE pipe, a real subprocess.

    `capsys` keeps the two streams apart and is structurally blind to this: off a
    TTY stdout is block-buffered while stderr never is, so without an explicit
    flush a stderr refusal overtakes the stdout report it annotates, and the
    reader's opening line is the wrong one.

    `restore-source` is the seam: one run can report on stdout AND refuse on
    stderr, and until this phase that branch carried no flush at all. Both channels
    now lead with the echo, and the report precedes the refusals. The fixture damages
    the GRAPH rather than the corpus — a node whose canonical core is gone is exactly
    the state this verb refuses on, and it is the cheapest way to get one restorable
    and one refused block out of the same run.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    env = {
        **os.environ,
        "MITOS_NO_UPDATE_CHECK": "1",
        "XDG_CONFIG_HOME": str(tmp_path / "xdg_config"),
        "GEMINI_API_KEY": "", "GOOGLE_API_KEY": "",
        "QDRANT_URL": "http://localhost:1",
    }

    def run(*argv):
        return subprocess.run(
            [sys.executable, "-m", "mitos.cli", *argv], cwd=str(workspace), env=env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

    run("init")
    for slug, axiom in (("alpha-block", "The alpha decision restores cleanly."),
                        ("beta-block", "The beta decision cannot be rendered.")):
        run("-p", str(workspace), "record", axiom, "--slug", slug,
            "--rejected", f"Losing {slug}.")

    # Make both nodes graph-only (drop their `### ` blocks), then blank one node's
    # axiom so `render_source_block` refuses it while the other still renders.
    decisions = workspace / "decisions.md"
    kept, skipping = [], False
    for line in decisions.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("### "):
            skipping = line.startswith("### alpha-block") or line.startswith("### beta-block")
        elif line.startswith("#") and not line.startswith("###"):
            skipping = False
        if not skipping:
            kept.append(line)
    decisions.write_text("".join(kept), encoding="utf-8")
    graph = sqlite3.connect(str(workspace / ".mitos" / "graph.sqlite"))
    graph.execute("UPDATE nodes SET axiom = '' WHERE slug = 'beta-block'")
    graph.commit()
    graph.close()

    done = run("-p", str(workspace), "restore-source", "--all-graph-only",
               "--dry-run")

    combined = done.stdout
    assert "Would restore 1 source block(s)" in combined
    assert "✗ refused: beta-block" in combined
    assert combined.index("Would restore") < combined.index("✗ refused"), (
        f"the refusal overtook the report it annotates:\n{combined}")
    # Both channels lead with the echo, so BOTH readings name the corpus.
    assert combined.index("corpus: ") < combined.index("Would restore")
    assert combined.count("corpus: ") == 2


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
