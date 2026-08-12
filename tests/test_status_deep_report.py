"""The `mitos status <project>` deep report: the vocabulary routing gave it, and
the one diagnosis the absolute-path escape hatch made routine.

Three things land here, and only the first is cosmetic:

* **The header names the project the caller named it by** — the registered name
  when one resolved, the canonical path when none did. Passed in from ``main()``,
  never reverse-looked-up inside the handler (3d rejected that build by name: it
  misses on a symlinked route whose registry entry is hand-written
  non-canonically, and prints a path for a registered project with every other row
  green).
* **A surviving `qdrant_collection` pin reaches `--json`** as a value while its
  rendered sentence stays on the text surface. The two tripwires 1d left for this
  phase are inverted in place (`test_collection_derivation.py`,
  `test_status_readiness.py`) rather than duplicated here.
* **The unbuilt graph (W31)** — a workspace whose `decisions.md` holds entries and
  whose graph holds no nodes. That is the *clone*: `.mitos/config.toml` and
  `decisions.md` are committed, `*.sqlite` is gitignored, and nobody commits a
  binary graph on purpose. Before this phase such a workspace read **READY ✓, exit
  0**, and every semantic read then answered *no precedents* for a project with
  hundreds — the "couldn't check" → "checked, it's clean" inversion arriving
  through the graph door.

**The fixture is the PAIR, and the pair is the point.** A clone alone passes under
either behaviour; its twin is a genuinely fresh workspace whose sample-only corpus
sits *above* the `BEGIN ENTRIES` sentinel and which must stay silent and `READY ✓`.
That is why the predicate is the sentinel and not "the file is non-empty" or "the
file has a `###` heading" — both of those fire on every `mitos init`.

Every row here is offline: `_check_qdrant` and `scroll_point_ids` are patched as
`cli` module attributes (`_check_qdrant`'s `import requests` is function-local, so
the seam is total). CI sees this whole module.
"""

import json
import os
import sys

import pytest
from unittest.mock import patch

from mitos import cli
from mitos.cli import main
from mitos.config import MitosConfig
from mitos.parser import corpus_has_entries

# The offline `status` seam, reused rather than re-spelled (the richer of the two
# copies in the tree — `test_status_readiness.py` carries a thinner one).
from test_collection_derivation import _qdrant, _scroll, _write_pin


# --------------------------------------------------------------------------- #
# Fixtures — the T16 pair, plus the registry helpers for the header rows
# --------------------------------------------------------------------------- #

_ENTRY_BLOCK = """
### clone-entry-one

**Decided:** A clone carries the corpus but never the graph.
**Rejected:** Committing the binary graph — it is derivative, and it would conflict
on every merge.
**Scope:** clone
"""


def _append_entry(workspace_dir, block=_ENTRY_BLOCK) -> None:
    """Appends a well-formed entry BELOW the sentinel, by hand.

    Not `mitos sync` (it parses, then refuses before any commit without a
    `GEMINI_API_KEY`, so an offline fixture seeded that way has an empty graph by
    accident rather than by construction — 3e, measured) and not `record` (it
    commits to the graph, which is the one thing this fixture must not have).
    """
    with open(os.path.join(str(workspace_dir), "decisions.md"), "a",
              encoding="utf-8") as f:
        f.write(block)


def _clone(tmp_path, name="clone"):
    """The cloned workspace: committed config + populated corpus, NO graph file.

    `mitos init` writes a 94 KB `graph.sqlite` alongside the rest, so the fixture
    deletes it — which is exactly what a clone looks like, since the sqlite is
    gitignored and the two files beside it are not. A fixture that leaves it behind
    still has a graph and proves nothing.
    """
    ws = tmp_path / name
    ws.mkdir()
    cli.cmd_init(MitosConfig(str(ws)))
    _append_entry(ws)
    os.remove(MitosConfig(str(ws)).db_path)
    assert not os.path.exists(MitosConfig(str(ws)).db_path)
    return ws


def _fresh(tmp_path, name="fresh"):
    """The twin: a genuinely fresh workspace — sample-only corpus, empty graph."""
    ws = tmp_path / name
    ws.mkdir()
    cli.cmd_init(MitosConfig(str(ws)))
    assert os.path.exists(MitosConfig(str(ws)).db_path)
    return ws


def _healthy(monkeypatch, collection_exists=False):
    """Everything else about the workspace reads ready: key, Qdrant, no scroll."""
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, collection_exists, points=0))
    monkeypatch.setattr(cli, "scroll_point_ids", _scroll(set()))


def _run(argv):
    """Drives `cli.main()` through argv, returning the exit code.

    The same shape `test_cli_selector.py` uses; re-spelled rather than imported
    because the rows below need `main()` itself, not that module's registry
    fixtures.
    """
    with patch.object(sys, "argv", ["mitos"] + list(argv)):
        try:
            main()
        except SystemExit as exc:
            return exc.code
    return 0


# --------------------------------------------------------------------------- #
# `parser.corpus_has_entries` — the boundary the predicate stands on
# --------------------------------------------------------------------------- #


class TestCorpusHasEntries:
    """The sentinel is the whole predicate, and each row says why a cheaper one fails."""

    def test_a_freshly_initialized_corpus_holds_no_entries(self, tmp_path) -> None:
        """The row that stops this phase flipping the entire suite to NEEDS ATTENTION.

        `mitos init` seeds 816 bytes including a complete `### example-slug` block —
        so both "the file is non-empty" and "the file contains a `###` heading" are
        true of every fresh workspace. The sample sits ABOVE the sentinel; entries go
        below it.
        """
        ws = _fresh(tmp_path)
        corpus = os.path.join(str(ws), "decisions.md")

        assert os.path.getsize(corpus) > 0          # the fixture is not vacuous
        assert "### example-slug" in open(corpus, encoding="utf-8").read()
        assert corpus_has_entries(corpus) is False

    def test_one_entry_below_the_sentinel_is_an_entry(self, tmp_path) -> None:
        ws = _clone(tmp_path)
        assert corpus_has_entries(os.path.join(str(ws), "decisions.md")) is True

    def test_it_agrees_with_the_two_pass_rule_over_every_short_corpus(self, tmp_path) -> None:
        """The scan STREAMS, and streaming is where an equivalent-looking rewrite
        stops being equivalent: a heading above a sentinel that appears later must
        be demoted to preamble, which a naive single pass returns `True` on.

        So the property is checked exhaustively rather than by example — every
        3-line corpus over the alphabet of shapes that matter, against a literal
        transcription of the parser's own two steps (find the first `BEGIN
        ENTRIES`, then apply `_split_entry_sections`' header predicate). ~1000
        corpora, no services, well under a second.
        """
        pieces = ["## h", "### h", "#### h", "# h", "  ### h",
                  "<!-- BEGIN ENTRIES -->", "[DECISION_TRANSCRIPT]",
                  "[/DECISION_TRANSCRIPT]", "text", ""]

        def two_pass(lines):
            begin = 0
            for i, line in enumerate(lines):
                if "BEGIN ENTRIES" in line:
                    begin = i + 1
                    break
            in_transcript = False
            for line in lines[begin:]:
                stripped = line.strip()
                if not in_transcript and stripped == "[DECISION_TRANSCRIPT]":
                    in_transcript = True
                    continue
                if in_transcript and stripped == "[/DECISION_TRANSCRIPT]":
                    in_transcript = False
                    continue
                if line.startswith("##") and not line.startswith("####") and not in_transcript:
                    return True
            return False

        import itertools
        corpus = tmp_path / "sweep.md"
        mismatches = []
        for combo in itertools.product(pieces, repeat=3):
            corpus.write_text("\n".join(combo) + "\n", encoding="utf-8")
            if corpus_has_entries(str(corpus)) != two_pass(list(combo)):
                mismatches.append(combo)
        assert mismatches == []

    def test_a_missing_file_is_not_a_populated_corpus(self, tmp_path) -> None:
        assert corpus_has_entries(str(tmp_path / "nope.md")) is False

    def test_a_directory_in_the_corpus_slot_is_not_a_populated_corpus(self, tmp_path) -> None:
        """The other `OSError`: an unreadable path answers False, never raises.

        This predicate rides four read surfaces and a status rung; a raise here
        would take down the answer it is only annotating.
        """
        assert corpus_has_entries(str(tmp_path)) is False

    def test_with_no_sentinel_the_whole_file_is_the_entry_stream(self, tmp_path) -> None:
        """`parse_entry_stream`'s rule, mirrored: no sentinel → `begin_idx` 0."""
        path = tmp_path / "d.md"
        path.write_text("### only-entry\n\n**Decided:** X\n", encoding="utf-8")
        assert corpus_has_entries(str(path)) is True

    @pytest.mark.parametrize("heading", [
        "#### too-deep",       # `####` is not an entry delimiter
        "# a title",           # a single `#` is not either
        "  ### indented",      # the predicate reads `line`, not `line.strip()`
        "##### deeper still",  # starts with `####`, so excluded with it
    ])
    def test_non_entry_headings_below_the_sentinel_do_not_count(self, tmp_path, heading) -> None:
        path = tmp_path / "d.md"
        path.write_text(f"<!-- BEGIN ENTRIES -->\n{heading}\n", encoding="utf-8")
        assert corpus_has_entries(str(path)) is False

    def test_a_heading_inside_a_transcript_block_is_literal_text(self, tmp_path) -> None:
        """`_split_entry_sections` is transcript-aware, so this scan must be too.

        A `##` line inside `[DECISION_TRANSCRIPT]…[/DECISION_TRANSCRIPT]` is quoted
        conversation, not a delimiter — the latent prototype bug the section splitter
        fixed. A scan that disagreed would report entries the parser would not find.
        """
        path = tmp_path / "d.md"
        path.write_text(
            "<!-- BEGIN ENTRIES -->\n"
            "[DECISION_TRANSCRIPT]\n"
            "### this is quoted conversation, not an entry\n"
            "[/DECISION_TRANSCRIPT]\n",
            encoding="utf-8",
        )
        assert corpus_has_entries(str(path)) is False

    def test_the_sentinel_scan_itself_is_transcript_blind_like_the_parser(self, tmp_path) -> None:
        """Measured rather than assumed (the plan's stretch item).

        `parse_entry_stream` scans for `BEGIN ENTRIES` over EVERY line, transcript
        or not, and takes the first hit — so a sentinel quoted inside a transcript
        block cuts the stream there for the parser too. This scan copies that
        asymmetry deliberately: it is the parser's behaviour, and a "fix" here would
        make the cheap scan and the real parse disagree about where entries start.
        """
        path = tmp_path / "d.md"
        path.write_text(
            "## real-looking-header-above\n"
            "[DECISION_TRANSCRIPT]\n"
            "<!-- BEGIN ENTRIES -->\n"
            "[/DECISION_TRANSCRIPT]\n"
            "### below-the-quoted-sentinel\n",
            encoding="utf-8",
        )
        assert corpus_has_entries(str(path)) is True

        from mitos.parser import parse_entry_stream
        entries = parse_entry_stream(
            path.read_text(encoding="utf-8"), "decision", str(path), failures=[],
        )
        # The parser agrees about the cut: the header above the quoted sentinel is
        # NOT in its stream, so the two never disagree about which side an entry
        # falls on. (Neither entry validates — that is not what this row is about.)
        assert all(e.slug != "real-looking-header-above" for e in entries)


# --------------------------------------------------------------------------- #
# `recall.missing_graph_is_a_gap` — the gate is the node count
# --------------------------------------------------------------------------- #


class _Store:
    """A store answering a fixed fingerprint, recording what was asked of it."""

    def __init__(self, node_count, active=None, raises=False):
        self.node_count = node_count
        self._active = active if active is not None else set()
        self.raises = raises
        self.asked = []

    def graph_fingerprint(self):
        self.asked.append("graph_fingerprint")
        if self.raises:
            raise RuntimeError("graph unreadable")
        return (self.node_count, "2026-07-31T00:00:00+00:00")

    def get_active_node_ids(self):
        self.asked.append("get_active_node_ids")
        return self._active


class _Config:
    def __init__(self, decisions_file):
        self.decisions_file = decisions_file


class TestTheGateIsTheNodeCount:
    """The one way to ship a confidently wrong rung is to copy the sibling's gate.

    `missing_index_is_a_gap` gates on `get_active_node_ids` because that set is
    exactly what `mitos reconcile` enqueues. This predicate must gate on the NODE
    COUNT, because `mitos sync` commits entries to nodes regardless of computed
    state — the gate and the heal agree by construction, and they are different
    heals.
    """

    def test_a_populated_graph_with_an_empty_active_set_is_not_a_gap(self, tmp_path) -> None:
        """The direct contract row, and it is a leaf row on purpose.

        End-to-end this state is **unconstructible**: kill edges are acyclic by
        write-path guard, so a non-empty graph always has at least one node with no
        incoming kill edge (measured — a supersession chain always leaves a live
        tip). That makes the sibling's gate *equivalent in practice* and wrong in
        contract, so the pin has to be taken where the difference exists: at the
        predicate. Measured: forced to the sibling's gate, exactly this row, the
        one below it, and one parameter of the state matrix red — and **nothing
        else in the tree**, which is the whole reason these three exist.
        """
        from mitos.recall import missing_graph_is_a_gap
        corpus = tmp_path / "decisions.md"
        corpus.write_text("<!-- BEGIN ENTRIES -->\n### an-entry\n", encoding="utf-8")
        store = _Store(node_count=7, active=set())

        assert missing_graph_is_a_gap(
            store, _Config(str(corpus)), corpus_has_entries=corpus_has_entries
        ) is False

    def test_the_predicate_never_hydrates_the_active_view(self, tmp_path) -> None:
        """And the node count is also the CHEAP gate — the reachable half of the same
        decision. `get_active_node_ids` runs two full active-view fetches, each
        hydrating the entire live corpus (its own sibling's docstring says so); this
        predicate rides every empty answer on four read surfaces, so it must be a
        `COUNT`, never a corpus read.
        """
        from mitos.recall import missing_graph_is_a_gap
        corpus = tmp_path / "decisions.md"
        corpus.write_text("<!-- BEGIN ENTRIES -->\n### an-entry\n", encoding="utf-8")
        store = _Store(node_count=0)

        missing_graph_is_a_gap(
            store, _Config(str(corpus)), corpus_has_entries=corpus_has_entries
        )

        assert store.asked == ["graph_fingerprint"]

    @pytest.mark.parametrize("store,corpus_text,expected", [
        (None, "<!-- BEGIN ENTRIES -->\n### e\n", True),      # the clone
        (None, "<!-- BEGIN ENTRIES -->\n", False),            # nothing to build
        (_Store(node_count=0), "<!-- BEGIN ENTRIES -->\n### e\n", True),
        (_Store(node_count=0), "<!-- BEGIN ENTRIES -->\n", False),
        (_Store(node_count=3), "<!-- BEGIN ENTRIES -->\n### e\n", False),
        (_Store(0, raises=True), "<!-- BEGIN ENTRIES -->\n### e\n", True),
        (_Store(0, raises=True), "<!-- BEGIN ENTRIES -->\n", False),
    ])
    def test_every_state_answers_on_the_corpus_except_a_populated_graph(
        self, tmp_path, store, corpus_text, expected
    ) -> None:
        """One row per state, including the two the docstring calls out: `store is
        None` is a real state (status genuinely has none over a clone), and an
        unreadable graph answers on the corpus too — the loud direction, because "I
        could not check" must never render as "I checked, it is empty".
        """
        from mitos.recall import missing_graph_is_a_gap
        corpus = tmp_path / "decisions.md"
        corpus.write_text(corpus_text, encoding="utf-8")

        assert missing_graph_is_a_gap(
            store, _Config(str(corpus)), corpus_has_entries=corpus_has_entries
        ) is expected

    def test_the_corpus_scan_is_required_never_defaulted(self) -> None:
        """`recall` keeps zero `mitos` imports, so the scan is injected — and a
        forgotten call site must be a TypeError, not a silent False (1c's
        `may_create` rule, whose lesson is that a defaulted declaration breaks
        implementations quietly).
        """
        from mitos.recall import missing_graph_is_a_gap
        with pytest.raises(TypeError):
            missing_graph_is_a_gap(None, _Config("/nope"))

    def test_the_note_names_sync_and_never_reconcile(self) -> None:
        """One word away from being the wrong answer, on both surfaces.

        `mitos reconcile` over an unbuilt graph diffs an empty active set against an
        absent collection, enqueues nothing, and reports SUCCESS on a workspace it
        did not touch — converting a recoverable state into one the operator
        believes they already fixed.
        """
        from mitos.recall import missing_graph_note
        for surface in ("cli", "mcp"):
            note = missing_graph_note(surface)
            assert "mitos sync" in note
            assert "reconcile" not in note
            assert "unbuilt" in note

    def test_recall_stays_a_zero_mitos_import_leaf(self) -> None:
        """The reason the scan is injected at all, pinned so a later phase has to
        argue with a test rather than notice a comment: `conflict.py` imports this
        module, so an import here lands inside the check family's discovered
        no-write fence (`test_conflict_closeout.py`'s exact-set pin).
        """
        import ast
        import mitos.recall as recall_mod

        with open(recall_mod.__file__, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert [m for m in imported if m.split(".")[0] == "mitos"] == []


# --------------------------------------------------------------------------- #
# The T16 pair — the clone and its fresh twin, on the text report
# --------------------------------------------------------------------------- #


class TestTheClonedWorkspacePair:
    def test_the_clone_names_the_missing_graph_and_mitos_sync(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The RUNG, asserted without the exit code — deliberately.

        The rung and the readiness gate are two changes, and a row that asserted
        both would hide the loss of either behind the other. Dropping
        `graph_unbuilt` from the `ready` conjunction leaves this row green and reds
        the one below it, which is what "independently pinned" means.
        """
        ws = _clone(tmp_path)
        capsys.readouterr()
        _healthy(monkeypatch)

        cli.cmd_status(str(ws))

        out = capsys.readouterr().out
        assert "the graph is unbuilt" in out
        assert "mitos sync" in out

    def test_the_clone_is_not_ready(self, tmp_path, monkeypatch, capsys) -> None:
        """The gate. No new verdict and no new exit code: `initialized` is still True
        on a clone, so the shipped middle value already fits the state.
        """
        ws = _clone(tmp_path)
        capsys.readouterr()
        _healthy(monkeypatch)

        rc = cli.cmd_status(str(ws))

        assert rc == 1
        assert "NEEDS ATTENTION ⚠" in capsys.readouterr().out

    def test_the_clones_rung_never_names_reconcile(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The wrong-heal row. Scoped to the rung's own line rather than the whole
        report, because the vector-completeness rung legitimately names `reconcile`
        elsewhere and a whole-output assertion would be green for the wrong reason.
        """
        ws = _clone(tmp_path)
        capsys.readouterr()
        _healthy(monkeypatch)

        cli.cmd_status(str(ws))

        rung = [ln for ln in capsys.readouterr().out.splitlines()
                if "the graph is unbuilt" in ln and ln.lstrip().startswith("⚠")]
        assert len(rung) == 1
        assert "mitos sync" in rung[0]
        assert "reconcile" not in rung[0]

    def test_the_clone_gets_a_numbered_next_step(self, tmp_path, monkeypatch, capsys) -> None:
        ws = _clone(tmp_path)
        capsys.readouterr()
        _healthy(monkeypatch)

        cli.cmd_status(str(ws))

        out = capsys.readouterr().out
        assert "Next steps:" in out
        assert any("`mitos sync`" in ln and ln.strip()[0].isdigit()
                   for ln in out.splitlines())

    def test_the_collection_row_does_not_contradict_the_rung(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The calm fresh-project sentence is part of the same lie.

        On a clone with a reachable Qdrant and no collection, the collection row used
        to read "auto-created on first record — none recorded yet" over a corpus of
        hundreds. Two lines, one report, opposite claims about whether anything has
        been decided here.
        """
        ws = _clone(tmp_path)
        capsys.readouterr()
        _healthy(monkeypatch)

        cli.cmd_status(str(ws))

        out = capsys.readouterr().out
        assert "none recorded yet" not in out
        assert "the graph is unbuilt" in out

    def test_the_fresh_twin_stays_silent_and_ready(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The half without which the pair proves nothing.

        Empty/fresh is first-class: a just-initialized project has no decisions and
        an empty graph, and that is health, not breakage. Both halves are load-bearing
        and they pull against each other, which is why this is one fixture in two.
        """
        ws = _fresh(tmp_path)
        capsys.readouterr()
        _healthy(monkeypatch)

        rc = cli.cmd_status(str(ws))

        out = capsys.readouterr().out
        assert rc == 0
        assert "READY ✓" in out
        assert "unbuilt" not in out
        assert "graph holds 0 node(s)" in out       # it HAS a graph; it is just empty

    def test_a_populated_graph_over_a_populated_corpus_stays_silent(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The reachable control for the gate: entries in the corpus AND nodes in the
        graph, including a superseded one, is an ordinary healthy project.
        """
        from mitos.parser import ParsedEntry
        from mitos.store import GraphStore

        ws = _fresh(tmp_path)
        _append_entry(ws)
        store = GraphStore(MitosConfig(str(ws)).db_path)
        for slug, axiom, sup in (("old-one", "Doomed axiom.", None),
                                 ("new-one", "Replacement axiom.", ["old-one"])):
            entry = ParsedEntry("decision", slug, 1, 5)
            entry.axiom = axiom
            entry.rejected_paths = "n/a"
            if sup:
                entry.supersedes = sup
            store.commit_parsed_entry(entry)
        assert store.graph_fingerprint()[0] == 2
        capsys.readouterr()
        _healthy(monkeypatch, collection_exists=True)

        rc = cli.cmd_status(str(ws))

        out = capsys.readouterr().out
        assert rc == 0
        assert "unbuilt" not in out

    def test_a_pre_v1a_graph_is_never_told_to_sync(self, tmp_path, monkeypatch, capsys) -> None:
        """Two heals on one report, one of them wrong — the guard that stops it.

        A prototype graph is skipped by both graph reads, so it reaches the predicate
        with no store while being, by definition, POPULATED. Unguarded it would be
        told "the graph is missing, run `mitos sync`" beside the `mitos cutover` line
        it already gets.
        """
        import sqlite3

        from mitos.migrations import is_pre_v1a_schema

        ws = _fresh(tmp_path)
        _append_entry(ws)
        db = MitosConfig(str(ws)).db_path
        os.remove(db)
        conn = sqlite3.connect(db)
        # `is_pre_v1a_schema`'s markers: `user_version == 0` AND a `nodes` table
        # that is non-STRICT or lacks `slug_casefold` (the minimal shape
        # `test_migrations.py::_proto_nodes` builds).
        conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, slug TEXT, kind TEXT);")
        conn.commit()
        assert is_pre_v1a_schema(conn) is True      # the fixture is not vacuous
        conn.close()
        capsys.readouterr()
        _healthy(monkeypatch)

        rc = cli.cmd_status(str(ws))

        out = capsys.readouterr().out
        assert rc == 1
        assert "mitos cutover" in out
        assert "unbuilt" not in out

    def test_a_directory_that_is_not_a_workspace_is_told_to_init_only(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The other wrong heal, found by running it rather than reading it.

        A directory holding a `decisions.md` and no `.mitos/` cannot be synced at
        all — its report already leads with `mitos init`, and a `mitos sync` rung
        there is a second instruction the reader cannot follow.
        """
        ws = tmp_path / "bare"
        ws.mkdir()
        (ws / "decisions.md").write_text(
            "<!-- BEGIN ENTRIES -->\n### an-entry\n", encoding="utf-8"
        )
        _healthy(monkeypatch)

        rc = cli.cmd_status(str(ws))

        out = capsys.readouterr().out
        assert rc == 1
        assert "NOT SET UP ✗" in out
        assert "mitos init" in out
        assert "unbuilt" not in out


# --------------------------------------------------------------------------- #
# The `--json` payload
# --------------------------------------------------------------------------- #


class TestTheJsonPayload:
    def test_graph_unbuilt_rides_beside_its_siblings(self, tmp_path, monkeypatch, capsys) -> None:
        clone, fresh = _clone(tmp_path), _fresh(tmp_path)
        capsys.readouterr()
        _healthy(monkeypatch)

        assert cli.cmd_status(str(clone), as_json=True) == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["checks"]["graph_unbuilt"] is True
        assert payload["ready"] is False
        assert payload["report"] == "project"

        assert cli.cmd_status(str(fresh), as_json=True) == 0
        twin = json.loads(capsys.readouterr().out)
        assert twin["checks"]["graph_unbuilt"] is False
        assert twin["ready"] is True

    def test_graph_unbuilt_is_a_plain_bool_never_a_tristate(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """Its "could not check" arm resolves to True — the loud direction — rather
        than to a `null` a consumer would have to interpret. Silence must never
        render as verified health.
        """
        ws = _clone(tmp_path)
        capsys.readouterr()
        _healthy(monkeypatch)

        cli.cmd_status(str(ws), as_json=True)

        assert json.loads(capsys.readouterr().out)["checks"]["graph_unbuilt"] is True

    def test_the_project_key_carries_the_registered_name_when_one_resolved(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        ws = _fresh(tmp_path)
        capsys.readouterr()
        _healthy(monkeypatch)

        cli.cmd_status(str(ws), as_json=True, project="a-registered-name")

        assert json.loads(capsys.readouterr().out)["project"] == "a-registered-name"

    def test_the_project_key_falls_back_to_the_canonical_path(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """§4.7's fourth value rule, and the one place `MitosConfig.project` spells
        it: an unregistered path (or no selector at all) is named by its path.
        """
        ws = _fresh(tmp_path)
        capsys.readouterr()
        _healthy(monkeypatch)

        cli.cmd_status(str(ws), as_json=True)

        assert json.loads(capsys.readouterr().out)["project"] == os.path.abspath(str(ws))

    def test_a_malformed_config_payload_carries_the_same_keys(
        self, tmp_path, capsys
    ) -> None:
        """The early branch is a payload too — `project` present on both emission
        sites for the same reason `report` is: a key present on one arm only is
        detectable by absence, which is sniffing wearing a discriminator's clothes.
        """
        ws = tmp_path / "broken"
        (ws / ".mitos").mkdir(parents=True)
        (ws / ".mitos" / "config.toml").write_text("qdrant_collection = [1, 2\n",
                                                   encoding="utf-8")
        (ws / "decisions.md").write_text("", encoding="utf-8")

        rc = cli.cmd_status(str(ws), as_json=True, project="broken-name")

        payload = json.loads(capsys.readouterr().out)
        assert rc == 1
        assert payload["report"] == "project"
        assert payload["project"] == "broken-name"
        assert payload["ready"] is False
        assert "config_error" in payload

    def test_the_inert_pin_is_null_when_no_line_survives(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The key is always present; only its value distinguishes the two states.

        (A pinned workspace's non-null value is pinned by the two inverted 1d
        tripwires — `test_collection_derivation.py` and `test_status_readiness.py` —
        rather than re-asserted here.)
        """
        ws = _fresh(tmp_path)
        capsys.readouterr()
        _healthy(monkeypatch)

        cli.cmd_status(str(ws), as_json=True)

        payload = json.loads(capsys.readouterr().out)
        assert "inert_collection_pin" in payload
        assert payload["inert_collection_pin"] is None

    @pytest.mark.parametrize("raw,expected", [
        ('"mitos-legacy"', "mitos-legacy"),
        ("123", 123),
        ('["a", "b"]', ["a", "b"]),
    ])
    def test_the_pinned_value_reaches_json_with_its_own_type(
        self, tmp_path, monkeypatch, capsys, raw, expected
    ) -> None:
        """Not necessarily a string — 1d's retirement rows parametrize a wrong-type
        `int` and a non-scalar array, the loader records both, and `_emit_json`
        serializes all three. No `repr()`: that escaping is the text surface's, and
        it has no business in a machine payload.
        """
        ws = tmp_path / "pinned"
        ws.mkdir()
        _write_pin(ws, raw)
        cli.cmd_init(MitosConfig(str(ws)))
        capsys.readouterr()
        _healthy(monkeypatch)

        cli.cmd_status(str(ws), as_json=True)

        assert json.loads(capsys.readouterr().out)["inert_collection_pin"] == expected


# --------------------------------------------------------------------------- #
# The header, driven through `main()` — the only place `target.name` exists
# --------------------------------------------------------------------------- #


class TestTheHeaderNamesTheProject:
    def test_a_registered_name_reaches_the_header(self, tmp_path, monkeypatch, capsys) -> None:
        ws = tmp_path / "someproject"
        ws.mkdir()
        cli.cmd_init(MitosConfig(str(ws)), name="alpha")
        capsys.readouterr()
        _healthy(monkeypatch)

        _run(["status", "alpha"])

        out = capsys.readouterr().out
        assert "MITOS STATUS for 'alpha' (" in out
        # The directory that answered stays present: an operator addressing a
        # project by name still needs to see which one it resolved to.
        assert os.path.realpath(str(ws)) in out

    def test_a_registered_project_reached_through_a_symlink_is_named_not_pathed(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The one fixture that tells a correct implementation from a plausible one.

        3d's finding, re-used here rather than re-derived: a registry entry written
        NON-canonically for a symlinked route resolves to the real root, and a
        `registry.reverse_lookup` of that root MISSES. So a header that re-derived
        the name inside `cmd_status` would print a path for a registered project with
        every other row in this class still green.
        """
        real = tmp_path / "real"
        real.mkdir()
        cli.cmd_init(MitosConfig(str(real)), name="realname")
        link = str(tmp_path / "link")
        os.symlink(str(real), link)
        assert os.path.realpath(link) == os.path.realpath(str(real)) != link

        from mitos import registry, routing
        registry.registry_path()
        with open(registry.registry_path(), "w", encoding="utf-8") as f:
            f.write(f'"linked" = "{link}"\n')
        resolved = routing.resolve_project("linked")
        assert registry.reverse_lookup(resolved.root) is None  # the fixture bites

        capsys.readouterr()
        _healthy(monkeypatch)

        _run(["status", "linked"])

        assert "MITOS STATUS for 'linked' (" in capsys.readouterr().out

    def test_an_unregistered_path_keeps_the_shipped_path_only_header(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """No name resolved, so the header is byte-identical to the shipped form —
        the fourth §4.7 value rule, and the reason the selectorless call below is
        unchanged in every respect.
        """
        ws = tmp_path / "unregistered"
        ws.mkdir()
        cli.cmd_init(MitosConfig(str(ws)))
        from mitos import registry
        with open(registry.registry_path(), "w", encoding="utf-8") as f:
            f.write("")
        capsys.readouterr()
        _healthy(monkeypatch)

        _run(["status", str(ws)])

        out = capsys.readouterr().out
        assert f"MITOS STATUS for {os.path.realpath(str(ws))} — " in out
        assert "(" not in out.splitlines()[1]      # no name-and-parens form

    def test_a_registered_name_carrying_a_control_character_renders_escaped(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """**Inverted at 6c** — this row used to pin the raw render, on purpose.

        The sibling of `test_status_overview.py::
        test_a_registry_name_carrying_a_control_character_renders_escaped`, and the
        reason is the same one, one surface further on: `registry.load()` validates
        the *value* (a string, absolute) and never the *key*, so a hand edit can
        leave a name holding a newline. It was rendered raw until the audit because
        `mitos projects` and 4a's overview table both did, and a unilateral
        divergence would have given one listing **three** spellings. 6c fixed all
        three surfaces in one edit, so the lockstep survives and `repr` is now the
        shared spelling on all of them.

        (The pin *value* one row over already went through `repr`, for the reason
        that now covers the name too — it simply had no sibling surface to wait for.
        The rule was always untrusted-values-get-escaped; the exception was a
        shipped set of surfaces that had to be fixed together, and they were.)
        """
        ws = tmp_path / "wsdir"
        ws.mkdir()
        cli.cmd_init(MitosConfig(str(ws)))
        from mitos import registry
        with open(registry.registry_path(), "w", encoding="utf-8") as f:
            f.write(f'"line\\nbreak" = "{os.path.realpath(str(ws))}"\n')
        capsys.readouterr()
        _healthy(monkeypatch)

        _run(["status", "line\nbreak"])

        out = capsys.readouterr().out
        assert "line\nbreak (" not in out                    # cannot break the header
        assert f"{'line' + chr(10) + 'break'!r} (" in out    # escaped, like `projects`

    def test_a_direct_call_without_a_name_is_unchanged(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The 44 existing call sites all pass positionally and see no difference —
        which is also why they cannot see a dispatch regression, and why the rows
        above drive `main()`.
        """
        ws = _fresh(tmp_path)
        capsys.readouterr()
        _healthy(monkeypatch)

        cli.cmd_status(str(ws))

        assert f"MITOS STATUS for {os.path.abspath(str(ws))} — " in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# The boundary: a malformed config is these two verbs' ANSWER, not their error
# --------------------------------------------------------------------------- #


@pytest.fixture
def malformed(tmp_path):
    ws = tmp_path / "malformed"
    (ws / ".mitos").mkdir(parents=True)
    (ws / ".mitos" / "config.toml").write_text("qdrant_collection = [1, 2\n",
                                               encoding="utf-8")
    (ws / "decisions.md").write_text("", encoding="utf-8")
    return ws


class TestTheMalformedConfigCarveOut:
    """Measured before it was written: `mitos status --json <malformed-workspace>`
    printed a one-line `Error:` and **not one byte of JSON**, because `main()` built
    the config before dispatch and `cmd_status`'s own contextual branch never ran.
    On `main` the path form still worked (`main()` built its config from cwd while
    `cmd_status` got `args.path`); 3b collapsed the two onto one boundary-resolved
    config, and because the exit code is 1 either way nothing went red.
    """

    def test_status_json_emits_a_payload_through_main(self, malformed, capsys) -> None:
        rc = _run(["status", "--json", str(malformed)])

        payload = json.loads(capsys.readouterr().out)
        assert rc == 1
        assert payload["report"] == "project"
        assert payload["ready"] is False
        assert "config_error" in payload

    def test_status_text_reports_calmly_through_main(self, malformed, capsys) -> None:
        rc = _run(["status", str(malformed)])

        out = capsys.readouterr().out
        assert rc == 1
        assert "NOT SET UP ✗" in out
        assert "config.toml malformed" in out

    def test_agent_block_still_prints_its_block(self, malformed, capsys) -> None:
        """The carve-out covers the whole frozenset, not just `status`.

        `cmd_agent_block` reads no config at all (verified against its body), so
        refusing it over a config it never opens was the same defect. The cost is
        real and stated rather than hidden: the dispatch's stderr corpus echo is
        ABSENT here, because there is no config to name a corpus from and inventing
        a fallback would make the echo claim a collection nobody resolved.
        """
        rc = _run(["agent-block", str(malformed)])

        captured = capsys.readouterr()
        assert rc == 0
        assert "mitos-agent-guide" in captured.out
        assert "corpus: " not in captured.err

    def test_every_other_verb_keeps_the_calm_boundary_error(self, malformed, capsys) -> None:
        """The carve-out is two verbs wide, deliberately. A verb that acts on a
        corpus cannot answer over a config it could not read, so its refusal stays
        the shipped one-line boundary error on stderr.
        """
        rc = _run(["scopes", "-p", str(malformed)])

        captured = capsys.readouterr()
        assert rc == 1
        assert "Error: Malformed config" in captured.err
        assert captured.out == ""


# --------------------------------------------------------------------------- #
# The exit contract, pinned on the NAMED form so 5a must invert it
# --------------------------------------------------------------------------- #


def test_the_named_form_keeps_the_shipped_zero_one_readiness_mapping(
    tmp_path, monkeypatch, capsys
) -> None:
    """`mitos status <project>`: 0 = fully ready, 1 = needs attention / not set up.

    SETUP.md sells that mapping as the machine-readable readiness signal, and this
    phase changes the predicate feeding it without touching the mapping itself.
    Pinned on the **named** form deliberately: 5a redefines the zero-arg form onto
    the global overview, and this phase says nothing about that — a phase's wording
    belongs to the phase that falsifies the old wording.
    """
    # `cmd_init` registers under the directory basename, and re-registering the
    # same workspace under a second name is refused outright (1a's unwaivable
    # guard) — so the fixture names the directories rather than the registrations.
    _fresh(tmp_path, "readyone")
    _clone(tmp_path, "cloneone")
    capsys.readouterr()
    _healthy(monkeypatch)

    assert _run(["status", "readyone"]) == 0
    assert "READY ✓" in capsys.readouterr().out
    assert _run(["status", "cloneone"]) == 1
    assert "NEEDS ATTENTION ⚠" in capsys.readouterr().out


def test_a_selectorless_status_renders_the_global_overview(
    tmp_path, monkeypatch, capsys
) -> None:
    """Inverted at 5a (entry-007): the zero-arg form is no longer a project report.

    It used to resolve the working directory and answer the deep report about it.
    Now it answers the *other* question — what does this machine have — so the whole
    payload changes shape, not merely a value: `report` flips to `"overview"` and
    every per-project readiness key is gone. Standing in a healthy workspace does not
    bring the old answer back; that is the point of the flip.
    """
    ws = _fresh(tmp_path)
    monkeypatch.chdir(str(ws))
    capsys.readouterr()
    _healthy(monkeypatch)

    rc = _run(["status", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["report"] == "overview"
    assert "ready" not in payload and "checks" not in payload
    # The registration `cmd_init` made is what the overview reports on, and the cwd
    # marker names it — the recovery the flip owes a caller who wanted the old answer.
    assert [p["name"] for p in payload["projects"]] == ["fresh"]
    assert payload["cwd_project"] == "fresh"

    # …and the named form still gives the report this row used to assert.
    rc = _run(["status", "fresh"])
    assert rc == 0
    assert "READY ✓" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# B3 — the four resolved corpus locations, on both encodings
# --------------------------------------------------------------------------- #


# The `checks` map as it ships, spelled rather than counted: the assertion is that
# the map did not GROW, and a count says that only until two edits cancel out. A
# resolved path is not a verdict, so none of the four may ever appear in here.
_SHIPPED_CHECK_KEYS = {
    "mitos_workspace", "decisions_buffer", "format_spec", "gemini_api_key",
    "qdrant_reachable", "collection_exists", "collection_points", "graph_nodes",
    "active_nodes", "missing_active_vectors", "missing_active_slugs",
    "orphan_points", "graph_unbuilt", "mcp_project_entry",
}

_PATH_KEYS = ("decisions_file", "questions_file", "archive_dir", "db_path")


def _four(ws):
    """The four resolved values, read off the config rather than hand-built.

    A hand-built path string would be a second derivation of the thing under test:
    it passes while agreeing with the render and disagreeing with the config, which
    is the exact failure a reader of this report cannot detect.
    """
    config = MitosConfig(str(ws))
    return [getattr(config, key) for key in _PATH_KEYS]


class TestTheFourResolvedPaths:
    """`mitos status <project>` resolves four corpus locations and used to print
    none of them. Two agent sessions grepped `.mitos/decisions.md` — because the row
    order teaches that layout — and got well-formed, confident, **zero-hit** answers
    for every slug over a corpus of hundreds. Nothing errored; nothing warned.

    The rows below quantify over the four MEMBERS, not over sites × encodings: a
    site/encoding grid reads complete with three paths in every cell, and the
    archive is the member a phase drops. It is also the load-bearing one —
    `<root>/decisions/archive` follows neither the root convention nor the `.mitos/`
    one, so a reader told the other three still cannot derive it, and a correct path
    makes a wrong absence-answer *more* credible than a pathless one did.
    """

    def test_all_four_render_on_the_wide_text_report(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        ws = _fresh(tmp_path)
        capsys.readouterr()
        _healthy(monkeypatch)

        cli.cmd_status(str(ws))

        out = capsys.readouterr().out
        for value in _four(ws):
            assert value in out

    def test_all_four_ride_the_wide_json_payload(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """`--json` is this verb's only agent-facing encoding — `status` has no MCP
        twin — and the sessions that failed were agents, so a text-only repair would
        have missed the consumer the item was written for.
        """
        ws = _fresh(tmp_path)
        capsys.readouterr()
        _healthy(monkeypatch)

        cli.cmd_status(str(ws), as_json=True)

        payload = json.loads(capsys.readouterr().out)
        for key, value in zip(_PATH_KEYS, _four(ws)):
            assert payload[key] == value
            assert os.path.isabs(payload[key])

    def test_the_decisions_row_carries_its_own_path(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """Scoped to the row's own line: the requirement is that a reader never has
        to correlate a row with a path listed elsewhere on the page to learn which
        file that row is about, and a whole-output assertion is satisfied by a path
        four lines away.
        """
        ws = _fresh(tmp_path)
        capsys.readouterr()
        _healthy(monkeypatch)

        cli.cmd_status(str(ws))

        rows = [ln for ln in capsys.readouterr().out.splitlines()
                if "decisions.md buffer" in ln]
        assert len(rows) == 1
        assert MitosConfig(str(ws)).decisions_file in rows[0]

    def test_a_failing_decisions_row_still_names_the_path(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """Removing `decisions.md` takes `initialized` down with it, so this fixture
        proves more than the glyph: the verdict is **NOT SET UP ✗** and the exit code
        is 1, so the four render on a not-ready report and not merely beside a
        failing check. (The founding sessions failed in workspaces where every row
        read ✓ — which is why nothing here is conditioned on a verdict at all.)
        """
        ws = _fresh(tmp_path)
        decisions_file = MitosConfig(str(ws)).decisions_file
        os.remove(decisions_file)
        capsys.readouterr()
        _healthy(monkeypatch)

        rc = cli.cmd_status(str(ws))

        out = capsys.readouterr().out
        assert rc == 1
        assert "NOT SET UP ✗" in out
        row = [ln for ln in out.splitlines() if "decisions.md buffer" in ln][0]
        assert row.lstrip().startswith("✗")
        assert decisions_file in row
        for value in _four(ws):
            assert value in out

    def test_the_clone_renders_the_graph_path_it_has_no_file_for(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """`• graph holds {n} node(s)` is guarded on a graph existing; the graph PATH
        is not. On a clone the neutral fact disappears and the location stays — which
        is the state a reader most needs it in.
        """
        ws = _clone(tmp_path)
        db_path = MitosConfig(str(ws)).db_path
        assert not os.path.exists(db_path)
        capsys.readouterr()
        _healthy(monkeypatch)

        cli.cmd_status(str(ws))

        out = capsys.readouterr().out
        assert f"• graph: {db_path}" in out
        # Scoped to the neutral line: the unbuilt-graph rung's own prose legitimately
        # says "the graph holds no nodes", so a whole-output assertion here would be
        # red for the wrong reason.
        assert not any(ln.startswith("  • graph holds") for ln in out.splitlines())
        for value in _four(ws):
            assert value in out

    def test_the_config_error_payload_names_none_of_the_four(
        self, tmp_path, capsys
    ) -> None:
        """The sibling of `test_a_malformed_config_payload_carries_the_same_keys`:
        that arm reaches both its encodings **without ever constructing a
        `MitosConfig`**, so the four values do not exist to render there. Absent, not
        null — the arm already omits every key needing a config, and four always-null
        path keys would be the only unresolvable-fact keys in a deliberately minimal
        payload.
        """
        ws = tmp_path / "broken"
        (ws / ".mitos").mkdir(parents=True)
        (ws / ".mitos" / "config.toml").write_text("qdrant_collection = [1, 2\n",
                                                   encoding="utf-8")
        (ws / "decisions.md").write_text("", encoding="utf-8")

        rc = cli.cmd_status(str(ws), as_json=True)

        payload = json.loads(capsys.readouterr().out)
        assert rc == 1
        assert "config_error" in payload
        for key in _PATH_KEYS:
            assert key not in payload

    def test_the_config_error_text_names_none_of_the_four(
        self, tmp_path, capsys
    ) -> None:
        """The cell an enumeration written from the payload's side drops: the
        `ConfigError` arm is one SITE with two encodings, not a `--json`-only site.
        """
        ws = tmp_path / "broken"
        (ws / ".mitos").mkdir(parents=True)
        (ws / ".mitos" / "config.toml").write_text("qdrant_collection = [1, 2\n",
                                                   encoding="utf-8")
        (ws / "decisions.md").write_text("", encoding="utf-8")

        rc = cli.cmd_status(str(ws))

        out = capsys.readouterr().out
        assert rc == 1
        assert "NOT SET UP ✗" in out
        assert "config.toml malformed" in out
        # No config exists to read the four off, so the absence is asserted on the
        # distinguishing tails of the derivations instead.
        assert os.path.join(str(ws), "decisions.md") not in out
        assert "questions.md" not in out
        assert os.path.join("decisions", "archive") not in out
        assert "graph.sqlite" not in out

    def test_no_resolved_path_became_a_check(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The membership fence, on the payload. The row mark is a THREE-valued
        lambda, so a neutral `—` row satisfies "no verdict" to the letter while
        sitting in the list where every reader takes it for a rung. So the rule binds
        on membership, in any glyph, on both encodings.
        """
        ws = _fresh(tmp_path)
        capsys.readouterr()
        _healthy(monkeypatch)

        cli.cmd_status(str(ws), as_json=True)

        checks = json.loads(capsys.readouterr().out)["checks"]
        assert set(checks) == _SHIPPED_CHECK_KEYS
        for key in _PATH_KEYS:
            assert key not in checks

    def test_the_three_rowless_paths_sit_on_no_verdict_line(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The same fence on the text surface. `decisions.md`'s path is exempt by
        design — it rides its own row's label, which is the whole point of D1; the
        three that have no row must not acquire one.
        """
        ws = _fresh(tmp_path)
        config = MitosConfig(str(ws))
        capsys.readouterr()
        _healthy(monkeypatch)

        cli.cmd_status(str(ws))

        lines = capsys.readouterr().out.splitlines()
        for value in (config.questions_file, config.archive_dir, config.db_path):
            carrying = [ln for ln in lines if value in ln]
            assert carrying, value
            assert all(ln.startswith("  • ") for ln in carrying), carrying

    def test_ready_and_the_exit_codes_are_unchanged(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The zero-byte-diff fence, in the same class as the render it fences, so a
        reader meets the rule and its proof together. No path is a rung: a fresh
        workspace has no `decisions/archive` directory and is healthy.
        """
        fresh, clone = _fresh(tmp_path), _clone(tmp_path)
        capsys.readouterr()
        _healthy(monkeypatch)

        assert cli.cmd_status(str(fresh), as_json=True) == 0
        assert json.loads(capsys.readouterr().out)["ready"] is True
        assert cli.cmd_status(str(clone), as_json=True) == 1
        assert json.loads(capsys.readouterr().out)["ready"] is False


class TestNoFilesystemReadBehindAResolvedPath:
    """A resolved path is **rendered, never inspected**. The temptation arrives
    looking free: `cmd_status` already calls `corpus_graph_divergence`, which globs
    the archive through `cutover._archive_files_oldest_first`, so "the listing is
    already paid, a count is just arithmetic" is available on this very verb — and
    it is a second, weaker report of what the divergence rung already covers,
    drifting from it by construction (the rung counts entries; a listing counts
    files). It is also the only thing B3 could add that is not O(1) in corpus size.
    """

    def test_the_archive_renders_with_no_archive_directory(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        ws = _fresh(tmp_path)
        archive_dir = MitosConfig(str(ws)).archive_dir
        assert not os.path.exists(archive_dir)   # `init` creates none; rotation does
        capsys.readouterr()
        _healthy(monkeypatch)

        cli.cmd_status(str(ws))

        lines = [ln for ln in capsys.readouterr().out.splitlines()
                 if archive_dir in ln]
        assert lines == [f"  • decisions archive: {archive_dir}"]

    def test_the_render_never_lists_the_archive(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """Non-vacuity is asserted, not assumed, in two halves.

        `divergence._corpus_files` imports the lister **inside the function**, so a
        module-attribute patch intercepts — the first assertion below resolves it the
        same way and proves the seam is live. And the fixture is the `_clone`
        deliberately: on a fresh workspace the shipped legitimate caller sits inside
        `try/except Exception: pass`, which would SWALLOW the raiser, while the clone
        has no `db_path` so the whole divergence block is skipped by its guard. Any
        listing that fires here is therefore B3's own.
        """
        ws = _clone(tmp_path)
        archive_dir = MitosConfig(str(ws)).archive_dir

        def _boom(*args, **kwargs):
            raise AssertionError("B3 renders a resolved path; it never lists it")

        monkeypatch.setattr("mitos.cutover._archive_files_oldest_first", _boom)
        from mitos.cutover import _archive_files_oldest_first as _late_bound
        with pytest.raises(AssertionError):
            _late_bound(archive_dir)              # the seam bites

        capsys.readouterr()
        _healthy(monkeypatch)

        rc = cli.cmd_status(str(ws))

        assert rc == 1                            # the clone's shipped exit code
        assert f"• decisions archive: {archive_dir}" in capsys.readouterr().out
