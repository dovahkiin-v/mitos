"""T9 — the corpus readers rotation would disarm, over an ENTIRELY archived corpus (3a).

Rotation makes ``decisions.md`` the working set rather than the corpus, and after a
drain it holds its header, its sentinel and nothing else. Every shipped reader that
asked that file a question about *the corpus* then answers the wrong file, and it
answers quietly: ``status`` stays calm, the four semantic read surfaces answer
"no precedent", the degraded text match answers "no matches", and the heal they name
(``mitos sync``) reads only the buffer and reconstructs nothing.

**Non-vacuity is asserted in every fixture, not assumed** (G4): the buffer's own scan
must answer ``False``, or a buffer-only reader passes these rows and proves nothing.
A buffer with one entry left in it is exactly that vacuous fixture.

Each property is a pair — the archived clone and a control. The control is the same
corpus after a keyless ``rebuild``, which is the heal these rows name, so a control
that stays silent is also evidence the heal works on the state it is named for.

New symbols are imported inside the rows, so running this module against a tree that
lacks them reds each row on its own rather than erroring the whole collection.
"""

import io
import json
import os
import re
import shlex
import subprocess
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import Dict, List

import pytest
from unittest.mock import patch

from mitos import cli
from mitos.cli import cmd_init, cmd_query, cmd_rebuild, cmd_surface
from mitos.config import MitosConfig
from mitos.errors import DatabaseError
from mitos.parser import corpus_has_entries
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager

from test_lexical_fallback import _Embeds, _MissingCollection, _NoMatches
from test_mcp_selector import FORBIDDEN_SYNTAX
from test_status_deep_report import _healthy


# --------------------------------------------------------------------------- #
# The corpus — rotation's `### slug` form and the legacy dated heading (G2)
# --------------------------------------------------------------------------- #

_Q3 = """### archived-cache-eviction

**Decided:** Archived cache eviction runs least-recently-used over the warm tier.
**Rejected:** Random eviction — it discards the hottest keys as often as the coldest.
**Scope:** cache

### archived-queue-backpressure

**Decided:** Archived queue producers block when the queue is full rather than drop.
**Rejected:** Dropping on overflow — silent loss is worse than a slow producer.
**Scope:** queue
"""

_Q2_LEGACY = """## 2026-05-21 — legacy-ledger-format — Ledger rows are append-only

**Decided:** Legacy ledger rows are append-only and corrections are new rows.

**Rejected:**
- in-place edits — they erase the audit trail

**Scope:** ledger
"""


def _write(path: str, text: str, mode: str = "w") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        f.write(text)


def _archived_workspace(root) -> MitosConfig:
    """`init`, then an archive-only corpus: two current-form entries, one legacy one.

    The buffer is left exactly as `init` seeds it — a sample block ABOVE the
    sentinel and no entry below it — which is what a drained buffer looks like.
    """
    os.makedirs(str(root), exist_ok=True)
    config = MitosConfig(str(root))
    cmd_init(config)
    _write(os.path.join(config.archive_dir, "2026-Q3.md"), _Q3)
    _write(os.path.join(config.archive_dir, "2026-Q2.md"), _Q2_LEGACY)
    # G4 — the fixture is only a fixture if the buffer alone says "empty".
    assert corpus_has_entries(config.decisions_file) is False
    return config


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")


@pytest.fixture
def status_shape(tmp_path, offline) -> MitosConfig:
    """No graph file at all: what `status` meets on a clone (G5)."""
    config = _archived_workspace(tmp_path / "archived")
    os.remove(config.db_path)
    assert not os.path.exists(config.db_path)
    return config


@pytest.fixture
def read_shape(tmp_path, offline) -> MitosConfig:
    """A graph file holding zero nodes: what a read surface leaves behind (G5)."""
    config = _archived_workspace(tmp_path / "archived")
    os.remove(config.db_path)
    assert GraphStore(config.db_path).graph_fingerprint()[0] == 0
    return config


def _rebuilt(config: MitosConfig) -> MitosConfig:
    """The control: the same corpus, committed by the heal these rows name."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_rebuild(config, allow_drops=False, assume_yes=True, as_json=True)
    report = json.loads(buf.getvalue())
    assert rc == 0 and report["swapped"] is True, report
    assert GraphStore(config.db_path).graph_fingerprint()[0] == 3
    return config


def _capture(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


def _assert_mcp_register(note: str) -> None:
    """The MCP recovery clause: a fact and a human next actor, never a command."""
    assert "unbuilt" in note
    assert "mitos " not in note
    for syntax in FORBIDDEN_SYNTAX:
        assert syntax not in note
    assert "a person" in note
    assert "reconcile" not in note
    assert "decisions/archive/" in note


# --------------------------------------------------------------------------- #
# T9-1 — the status rung
# --------------------------------------------------------------------------- #


class TestTheStatusRung:
    def test_an_archived_clone_fires_the_rung_naming_rebuild_with_its_selector(
        self, status_shape, monkeypatch, capsys
    ) -> None:
        config = status_shape
        capsys.readouterr()
        _healthy(monkeypatch)

        rc = cli.cmd_status(config.workspace_dir)

        out = capsys.readouterr().out
        rung = [ln for ln in out.splitlines()
                if "the graph is unbuilt" in ln and ln.lstrip().startswith("⚠")]
        assert len(rung) == 1
        assert f"mitos rebuild -p {config.workspace_dir!r}" in rung[0]
        assert "mitos sync" not in rung[0]
        assert "reconcile" not in rung[0]
        assert "decisions/archive/" in rung[0]
        step = [ln for ln in out.splitlines()
                if ln.strip()[:1].isdigit() and "mitos rebuild -p" in ln]
        assert len(step) == 1
        assert rc == 1

    def test_the_json_verdict_is_unbuilt_and_not_ready(
        self, status_shape, monkeypatch, capsys
    ) -> None:
        config = status_shape
        capsys.readouterr()
        _healthy(monkeypatch)

        cli.cmd_status(config.workspace_dir, as_json=True)

        data = json.loads(capsys.readouterr().out)
        assert data["checks"]["graph_unbuilt"] is True
        assert data["ready"] is False

    def test_control_the_rebuilt_corpus_carries_no_rung(
        self, read_shape, monkeypatch, capsys
    ) -> None:
        config = _rebuilt(read_shape)
        capsys.readouterr()
        _healthy(monkeypatch)

        cli.cmd_status(config.workspace_dir, as_json=True)

        assert json.loads(capsys.readouterr().out)["checks"]["graph_unbuilt"] is False


# --------------------------------------------------------------------------- #
# T9-2 — the four semantic read surfaces, both boundaries
# --------------------------------------------------------------------------- #


def _cli(config, verb, **kwargs) -> str:
    with patch("mitos.cli.MitosSyncManager") as MM:
        mgr = MitosSyncManager(config)
        mgr.embed_provider = _Embeds()
        mgr.vector_store = _MissingCollection()
        MM.return_value = mgr
        return _capture(verb, config, "cache eviction", **kwargs)


def _mcp(config, tool, vector_store) -> Dict:
    from mitos import mcp_server
    comps = (GraphStore(config.db_path, read_only=True), _Embeds(), vector_store)
    with patch.object(mcp_server, "get_workspace_components", return_value=comps):
        return json.loads(getattr(mcp_server, tool)(
            "cache eviction", project=config.workspace_dir))


class TestTheReadSurfaces:
    def test_cli_query_text_names_rebuild_with_its_selector(self, read_shape) -> None:
        out = _cli(read_shape, cmd_query)

        assert "No matching decisions found." in out
        assert "graph is unbuilt" in out
        assert f"mitos rebuild -p {read_shape.project!r}" in out
        assert "decisions/archive/" in out
        assert "mitos sync" not in out
        assert "reconcile" not in out

    @pytest.mark.parametrize("verb", [cmd_query, cmd_surface])
    def test_cli_json_notes_name_rebuild(self, read_shape, verb) -> None:
        data = json.loads(_cli(read_shape, verb, as_json=True))

        assert "graph is unbuilt" in data["note"]
        assert "mitos rebuild -p" in data["note"]
        assert "decisions/archive/" in data["note"]
        assert "reconcile" not in data["note"]

    @pytest.mark.parametrize("store_kind", ["missing", "empty"])
    @pytest.mark.parametrize("tool", ["query_decisions", "surface_decisions"])
    def test_mcp_notes_name_no_command_and_a_human_next_actor(
        self, read_shape, tool, store_kind
    ) -> None:
        store = _MissingCollection() if store_kind == "missing" else _NoMatches()
        out = _mcp(read_shape, tool, store)

        assert out.get("matches", out.get("active_decisions")) == []
        _assert_mcp_register(out["note"])

    @pytest.mark.parametrize("tool", ["query_decisions", "surface_decisions"])
    def test_control_the_rebuilt_graph_carries_no_unbuilt_note(
        self, read_shape, tool
    ) -> None:
        config = _rebuilt(read_shape)

        assert "unbuilt" not in json.dumps(_mcp(config, tool, _NoMatches()))
        assert "unbuilt" not in _cli(config, cmd_query, as_json=True)


# --------------------------------------------------------------------------- #
# T9-3 — the overview's no-collection warning
# --------------------------------------------------------------------------- #


from test_status_overview import _by_name, _register, _workspace  # noqa: E402
from test_status_overview import qdrant as _overview_qdrant  # noqa: E402

# The module's instance-boundary fixture, bound under the name the rows request.
qdrant = _overview_qdrant


class TestTheOverview:
    def test_an_archived_only_project_keeps_the_warning(self, tmp_path, qdrant) -> None:
        from mitos import overview
        project = _workspace(tmp_path / "project")
        _write(os.path.join(project, "decisions", "archive", "2026-Q3.md"), _Q3)
        assert corpus_has_entries(os.path.join(project, "decisions.md")) is False
        _register(project=project)

        payload = overview.build_overview()
        entry = _by_name(payload)["project"]
        notes = "\n".join(cli._overview_notes(entry, payload))

        assert entry["collection_present"] is False
        assert "no vector collection" in notes
        assert "rebuild" not in notes and "sync" not in notes

    def test_a_fresh_project_stays_silent(self, tmp_path, qdrant) -> None:
        from mitos import overview
        project = _workspace(tmp_path / "project")
        _register(project=project)

        payload = overview.build_overview()

        assert cli._overview_notes(_by_name(payload)["project"], payload) == []

    def test_the_locator_names_the_same_two_paths_the_config_does(
        self, tmp_path, qdrant
    ) -> None:
        """Byte-identical for one workspace. The config is built from the SAME string
        the overview carries (its canonical path), so a symlinked tmp cannot split them.
        """
        from mitos import overview
        project = _workspace(tmp_path / "project")
        _register(project=project)
        payload = overview.build_overview()
        seen: List = []

        def _scan(locator) -> bool:
            seen.append(locator)
            return False

        cli._overview_notes(_by_name(payload)["project"], payload, corpus_scan=_scan)

        config = MitosConfig(project)
        assert len(seen) == 1
        assert seen[0].decisions_file == config.decisions_file
        assert seen[0].archive_dir == config.archive_dir


# --------------------------------------------------------------------------- #
# T9-4 — the lexical fallback over the archived corpus
# --------------------------------------------------------------------------- #


_ARCHIVED_SLUGS = {"archived-cache-eviction", "legacy-ledger-format"}


class TestLexicalOverTheArchives:
    def test_cli_no_providers_matches_archived_entries_of_both_heading_forms(
        self, read_shape
    ) -> None:
        data = json.loads(_capture(cmd_query, read_shape, "eviction ledger",
                                   as_json=True))

        assert data["degraded"] == "lexical"
        assert {m["slug"] for m in data["matches"]} == _ARCHIVED_SLUGS
        assert "decisions/archive/" in data["note"]
        assert "unread_files" not in data

    @pytest.mark.parametrize("tool", ["query_decisions", "surface_decisions"])
    def test_mcp_no_providers_matches_archived_entries(self, read_shape, tool) -> None:
        from mitos import mcp_server
        comps = (GraphStore(read_shape.db_path, read_only=True), None, None)
        with patch.object(mcp_server, "get_workspace_components", return_value=comps):
            out = json.loads(getattr(mcp_server, tool)(
                "eviction ledger", project=read_shape.workspace_dir))

        assert out["degraded"] == "lexical"
        assert {m["slug"] for m in out["matches"]} == _ARCHIVED_SLUGS

    def test_the_pre_v1a_path_never_opens_sqlite(self, status_shape) -> None:
        """G8: `_corpus_files` imports `cutover`, which imports `store` as a MODULE.
        Importing is not opening SQLite, and this row is what says so.
        """
        exc = DatabaseError("This graph predates the V1a schema.")
        with patch("mitos.cli.MitosSyncManager", side_effect=exc), \
                patch("mitos.store.GraphStore", side_effect=AssertionError("opened")):
            data = json.loads(_capture(cmd_query, status_shape, "eviction ledger",
                                       as_json=True))

        assert data["stamps_unavailable"] is True
        assert {m["slug"] for m in data["matches"]} == _ARCHIVED_SLUGS
        assert not os.path.exists(status_shape.db_path)


# --------------------------------------------------------------------------- #
# T9-5 / T9-6 — disclosure, rank and dedup (the leaf, driven by the real file list)
# --------------------------------------------------------------------------- #


_SENTINEL_BUFFER = ("# Decisions\n\n<!-- BEGIN ENTRIES — new decisions go directly "
                    "below this line, newest first -->\n\n")


def _entry(slug: str, axiom: str) -> str:
    return f"### {slug}\n\n**Decided:** {axiom}\n**Rejected:** none.\n\n"


def _corpus(tmp_path, buffer: str, archives: Dict[str, bytes]) -> SimpleNamespace:
    root = tmp_path / "corpus"
    archive_dir = root / "decisions" / "archive"
    archive_dir.mkdir(parents=True)
    (root / "decisions.md").write_text(_SENTINEL_BUFFER + buffer, encoding="utf-8")
    for name, data in archives.items():
        (archive_dir / name).write_bytes(data)
    return SimpleNamespace(decisions_file=str(root / "decisions.md"),
                           archive_dir=str(archive_dir))


def _lexical(locator, query: str) -> Dict:
    from mitos.divergence import _corpus_files
    from mitos.lexical import lexical_fallback
    return lexical_fallback(query, corpus_paths=_corpus_files(locator),
                            reason="test", store=None)


class TestLexicalDisclosure:
    def test_an_undecodable_archive_is_named_and_the_rest_still_answers(
        self, tmp_path
    ) -> None:
        locator = _corpus(tmp_path, "", {
            "2026-Q1.md": b"### broken\n\n**Decided:** cache \xff\xfe\n",
            "2026-Q3.md": _entry("readable-cache", "A cache axiom.").encode(),
        })

        env = _lexical(locator, "cache")

        assert [m["slug"] for m in env["matches"]] == ["readable-cache"]
        assert env["unread_files"] == ["2026-Q1.md"]
        assert "could not be read" in env["note"] and "2026-Q1.md" in env["note"]

    def test_a_zero_match_over_an_unread_file_is_never_a_calm_empty(
        self, tmp_path
    ) -> None:
        locator = _corpus(tmp_path, "", {"2026-Q1.md": b"### x\n\xff\n"})

        env = _lexical(locator, "zebra")

        assert env["matches"] == []
        assert env["unread_files"] == ["2026-Q1.md"]
        assert "could not be read" in env["note"]

    @pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0,
                        reason="environmental: mode 000 is readable by root")
    def test_an_unreadable_archive_is_named(self, tmp_path) -> None:
        locator = _corpus(tmp_path, "", {
            "2026-Q2.md": _entry("locked-cache", "A cache axiom.").encode(),
            "2026-Q3.md": _entry("open-cache", "A cache axiom.").encode(),
        })
        locked = os.path.join(locator.archive_dir, "2026-Q2.md")
        os.chmod(locked, 0)
        try:
            env = _lexical(locator, "cache")
        finally:
            os.chmod(locked, 0o644)

        assert [m["slug"] for m in env["matches"]] == ["open-cache"]
        assert env["unread_files"] == ["2026-Q2.md"]

    def test_an_undecodable_buffer_answers_rather_than_raising(self, tmp_path) -> None:
        """Before 3a the read sat outside the `try`, so this RAISED out of the leaf
        (and out of an MCP tool whose `except` arm called it)."""
        locator = _corpus(tmp_path, "", {
            "2026-Q3.md": _entry("archived-cache", "A cache axiom.").encode(),
        })
        with open(locator.decisions_file, "ab") as f:
            f.write(b"\xff\xfe")

        env = _lexical(locator, "cache")

        assert [m["slug"] for m in env["matches"]] == ["archived-cache"]
        assert env["unread_files"] == ["decisions.md"]

    def test_a_clean_run_carries_no_unread_key(self, tmp_path) -> None:
        locator = _corpus(tmp_path, _entry("buffer-cache", "A cache axiom."), {})

        assert "unread_files" not in _lexical(locator, "cache")

    def test_a_bare_string_is_refused_rather_than_iterated(self, tmp_path) -> None:
        from mitos.lexical import lexical_fallback
        with pytest.raises(TypeError):
            lexical_fallback("cache", corpus_paths=str(tmp_path / "decisions.md"),
                             reason="test")


class TestLexicalRankAndDedup:
    def test_ties_order_buffer_then_newer_archive_then_older(self, tmp_path) -> None:
        locator = _corpus(tmp_path, _entry("buffer-cache", "A cache axiom."), {
            "2026-Q2.md": _entry("older-cache", "A cache axiom.").encode(),
            "2026-Q3.md": _entry("newer-cache", "A cache axiom.").encode(),
        })

        env = _lexical(locator, "cache")

        assert [m["slug"] for m in env["matches"]] == [
            "buffer-cache", "newer-cache", "older-cache"]

    def test_more_terms_still_outrank_a_newer_file(self, tmp_path) -> None:
        locator = _corpus(tmp_path, _entry("buffer-cache", "A cache axiom."), {
            "2026-Q2.md": _entry("older-cache-eviction", "Cache eviction.").encode(),
        })

        env = _lexical(locator, "cache eviction")

        assert [m["slug"] for m in env["matches"]] == [
            "older-cache-eviction", "buffer-cache"]

    def test_a_slug_in_the_buffer_and_an_archive_is_listed_once_from_the_buffer(
        self, tmp_path
    ) -> None:
        locator = _corpus(tmp_path, _entry("Dup-Cache", "The buffer's cache axiom."), {
            "2026-Q3.md": _entry("dup-cache", "The archive's cache axiom.").encode(),
        })

        env = _lexical(locator, "cache")

        assert len(env["matches"]) == 1
        assert env["matches"][0]["axiom"] == "The buffer's cache axiom."


# --------------------------------------------------------------------------- #
# T9-7 — the archive filename regex is shared on purpose (G7)
# --------------------------------------------------------------------------- #


class TestTheRegexAgreement:
    @pytest.mark.parametrize("name", ["notes.md", "2026-q3.md"])
    def test_a_misnamed_archive_is_invisible_to_the_scan_the_replay_and_lexical(
        self, tmp_path, offline, name
    ) -> None:
        from mitos.divergence import corpus_holds_entries
        config = MitosConfig(str(tmp_path / "ws"))
        os.makedirs(config.workspace_dir)
        cmd_init(config)
        _write(os.path.join(config.archive_dir, name), _Q3)

        assert corpus_holds_entries(config) is False
        assert _lexical(config, "eviction")["matches"] == []
        buf = io.StringIO()
        with redirect_stdout(buf):
            cmd_rebuild(config, allow_drops=False, assume_yes=True, as_json=True)
        assert json.loads(buf.getvalue())["decisions_committed"] == 0

    def test_a_canonical_archive_is_seen(self, status_shape) -> None:
        from mitos.divergence import corpus_holds_entries
        assert corpus_holds_entries(status_shape) is True


# --------------------------------------------------------------------------- #
# T9-8 — the heal runs, through the process a person runs
# --------------------------------------------------------------------------- #


def _env(base: str) -> Dict[str, str]:
    """Copied from `test_cluster_c_checkpoint._env` (2g), not generalised."""
    return {**os.environ, "MITOS_NO_UPDATE_CHECK": "1",
            "XDG_CONFIG_HOME": os.path.join(base, "xdg_config"),
            "XDG_CACHE_HOME": os.path.join(base, "xdg_cache"),
            "GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", "ANTHROPIC_API_KEY": "",
            "QDRANT_URL": "http://localhost:1"}


def _mitos(env, cwd, *argv) -> "subprocess.CompletedProcess[str]":
    return subprocess.run([sys.executable, "-m", "mitos.cli", *argv], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)


class TestTheHealRuns:
    def test_the_rungs_printed_recipe_builds_the_graph(self, status_shape, tmp_path) -> None:
        """The recipe is taken from the rung's own text and run with ``--yes`` added.

        It is printed without ``--yes`` on purpose — its reader is a person at a
        terminal who should see the swap prompt — and off a TTY ``cmd_rebuild`` never
        calls ``input()``: it refuses with ``confirmation_required``, so a verbatim
        subprocess run cannot confirm. That one flag is the only edit made to it.
        """
        env = _env(str(tmp_path))
        ws = status_shape.workspace_dir

        status = _mitos(env, str(tmp_path), "status", ws)
        recipe = re.search(r"`(mitos rebuild -p [^`]+)`", status.stdout)
        assert recipe, status.stdout
        argv = shlex.split(recipe.group(1))[1:] + ["--yes"]

        healed = _mitos(env, str(tmp_path), *argv)
        assert healed.returncode == 0, healed.stdout + healed.stderr
        assert "Rebuild complete" in healed.stdout

        after = _mitos(env, str(tmp_path), "status", "--json", ws)
        assert json.loads(after.stdout)["checks"]["graph_unbuilt"] is False
        assert GraphStore(status_shape.db_path).graph_fingerprint()[0] == 3

    def test_json_reports_the_swap_and_mints_no_backup(self, status_shape, capsys) -> None:
        capsys.readouterr()
        rc = cmd_rebuild(status_shape, allow_drops=False, assume_yes=True, as_json=True)

        report = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert report["swapped"] is True
        assert report["bak_path"] is None

    def test_a_directory_without_mitos_still_refuses(self, tmp_path, capsys) -> None:
        bare = MitosConfig(str(tmp_path / "bare"))
        os.makedirs(bare.workspace_dir)
        capsys.readouterr()

        rc = cmd_rebuild(bare, allow_drops=False, assume_yes=True, as_json=True)

        assert rc == 1
        assert json.loads(capsys.readouterr().out)["reason"] == "no_graph"
        assert not os.path.exists(bare.db_path)


# --------------------------------------------------------------------------- #
# T9-9 — a checked-clean buffer question stays a buffer question
# --------------------------------------------------------------------------- #


class TestTheStagedGateIsABufferQuestion:
    def test_an_archived_and_committed_corpus_has_nothing_pending(
        self, read_shape, capsys
    ) -> None:
        """`_run_staged_check` asks which BUFFERED entries are not in the graph. Its
        falsifier is rotation moving an uncommitted entry, which 3b's *committed*
        conjunct forbids — so over archived, committed entries it answers clean
        without any provider (keyless here, so a probe would fail closed at exit 2).
        """
        config = _rebuilt(read_shape)
        capsys.readouterr()

        rc = cli.cmd_check(config, staged=True, scope=None, fresh=False,
                           assume_yes=True, as_json=False)

        assert rc == 0
        assert "Gate clear" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# T9-10 — the recall contract over a corpus that lives in its archive
# --------------------------------------------------------------------------- #


class _Store:
    def __init__(self, node_count: int, raises: bool = False) -> None:
        self.node_count, self.raises = node_count, raises

    def graph_fingerprint(self):
        if self.raises:
            raise RuntimeError("graph unreadable")
        return (self.node_count, "")


class TestTheRecallContract:
    @pytest.mark.parametrize("where", ["archive", "buffer", "nowhere"])
    @pytest.mark.parametrize("store,populated_answer", [
        (None, True), (_Store(0), True), (_Store(0, raises=True), True),
        (_Store(4), False),
    ])
    def test_the_state_matrix_answers_on_buffer_plus_archives(
        self, tmp_path, where, store, populated_answer
    ) -> None:
        from mitos.divergence import corpus_holds_entries
        from mitos.recall import missing_graph_is_a_gap
        locator = _corpus(
            tmp_path,
            _entry("in-buffer", "An axiom.") if where == "buffer" else "",
            {"2026-Q3.md": _entry("in-archive", "An axiom.").encode()}
            if where == "archive" else {},
        )
        expected = populated_answer if where != "nowhere" else False

        assert missing_graph_is_a_gap(
            store, locator, corpus_scan=corpus_holds_entries) is expected

    def test_the_scan_is_required_and_the_error_names_it(self) -> None:
        from mitos.recall import missing_graph_is_a_gap
        with pytest.raises(TypeError, match="corpus_scan"):
            missing_graph_is_a_gap(None, SimpleNamespace(decisions_file="/nope"))

    def test_the_note_names_rebuild_per_boundary_and_never_reconcile(self) -> None:
        from mitos.recall import missing_graph_note
        config = SimpleNamespace(project="my project")

        cli_note = missing_graph_note("cli", config)
        mcp_note = missing_graph_note("mcp", config)

        assert "mitos rebuild -p 'my project'" in cli_note
        assert "reconcile" not in cli_note and "mitos sync" not in cli_note
        _assert_mcp_register(mcp_note)
        assert "my project" not in mcp_note
