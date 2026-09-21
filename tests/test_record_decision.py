"""Test suite for the record_decision write tool (MitosSyncManager.record_decision_entry).

Covers the spec's four layers: unit (serialise/round-trip, validation, structural-token
rejection, slug determinism), integration (full read+write loop, idempotency, supersedes,
collisions, graceful degradation), MCP boundary, and adversarial (TOCTOU, commit rollback,
pathological inputs). Asserts the SPEC-correct behaviour: an exact slug collision returns
`slug_collision` (never an auto-corrects edge that would create two same-slug nodes), and
every error path leaves decisions.md byte-for-byte unchanged.
"""

import os
import inspect
import json
import re
import shlex
import shutil
import sys
import tempfile
import threading
from typing import Tuple, Iterator

import pytest
from unittest.mock import MagicMock, patch

from mitos.config import MitosConfig
from mitos.cli import cmd_init
from mitos.store import GraphStore
from mitos.errors import DatabaseError
from mitos.sync import MitosSyncManager
from mitos.parser import parse_decisions_file


@pytest.fixture
def ws() -> Iterator[Tuple[MitosConfig, MitosSyncManager]]:
    """A fully initialised temporary Mitos workspace and a manager bound to it."""
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    cmd_init(config)
    yield config, MitosSyncManager(config)
    shutil.rmtree(tmp, ignore_errors=True)


def _read(config: MitosConfig) -> str:
    with open(config.decisions_file, "r", encoding="utf-8") as f:
        return f.read()


# --------------------------------------------------------------------------- #
# Unit
# --------------------------------------------------------------------------- #

def test_keystone_round_trip(ws) -> None:
    """Serialise → parse → the committed node equals the parsed (normalised) fields."""
    config, m = ws
    axiom = "Use SQLite in WAL mode for the graph store."
    rejected = "pgvector (too heavy for local-first), sqlite-vec (deferred to v0.2)."
    res = m.record_decision_entry(
        axiom=axiom, rejected_paths=rejected, scope=["substrate", "database"],
        mechanisms=["sqlite", "wal-mode"], context="Local-first concurrent reads/writes.",
        slug="use-sqlite-wal",
    )
    assert "error" not in res and res["status"] == "created"

    parsed = parse_decisions_file(_read(config), errors=[])
    assert len(parsed) == 1
    p = parsed[0]
    assert p.slug == "use-sqlite-wal"
    assert p.core_axiom == axiom
    assert p.rejected_paths == rejected
    assert p.mechanisms == ["sqlite", "wal-mode"]
    assert p.scope == ["substrate", "database"]
    assert p.context == "Local-first concurrent reads/writes."

    # The committed node matches the parsed form (we commit the parsed entry).
    node = GraphStore(config.db_path).get_node_by_slug("use-sqlite-wal")
    assert node["core_axiom"] == axiom
    assert node["rejected_paths"] == rejected


def test_empty_slug_error(ws) -> None:
    """An empty slug returns an empty_slug error, bypassing the fallback."""
    config, m = ws
    res = m.record_decision_entry(
        axiom="Some valid decision.",
        rejected_paths="None.",
        scope=["test"],
        slug="",  # Explicitly empty
    )
    assert res.get("code") == "empty_slug"
    assert "hyphenated handle" in res.get("error")
def test_multiline_rejected_paths_round_trips(ws) -> None:
    """A bulleted rejected_paths list survives serialise→parse intact."""
    config, m = ws
    rejected = "- Postgres — breaks local-first\n- MySQL — licensing"
    res = m.record_decision_entry("Pick the database.", rejected, [], slug="pick-db")
    assert "error" not in res
    p = parse_decisions_file(_read(config), errors=[])[0]
    assert p.rejected_paths == rejected


def test_validation_empty_fields(ws) -> None:
    """Empty/whitespace axiom or rejected_paths return the structured error, no write."""
    config, m = ws
    before = _read(config)
    assert m.record_decision_entry("", "why", [])["code"] == "empty_axiom"
    assert m.record_decision_entry("   \n  ", "why", [])["code"] == "empty_axiom"
    assert m.record_decision_entry("ax", "", [])["code"] == "missing_rejected_paths"
    assert m.record_decision_entry("ax", "  \t ", [])["code"] == "missing_rejected_paths"
    assert _read(config) == before  # nothing written


@pytest.mark.parametrize("field", ["axiom", "rejected", "context"])
@pytest.mark.parametrize("token", [
    "line one\n## a heading",        # column-0 H2 opens a new entry
    "line one\n### a heading",       # column-0 H3
    "text\n**Decided:** injected",   # field-shaped line
    "text\n**Anything:** injected",  # unknown field-shaped line
    "before BEGIN ENTRIES after",
    "x [DECISION_TRANSCRIPT] y",
    "x [NOTE: smuggled] y",
    "x [PARKED: smuggled] y",
])
def test_structural_token_rejected(ws, field, token) -> None:
    """Structural tokens in any content field → parse_failed, buffer unchanged (not sanitised)."""
    config, m = ws
    before = _read(config)
    kwargs = dict(axiom="A clean axiom.", rejected_paths="A clean rejection.", scope=[], context=None)
    if field == "axiom":
        kwargs["axiom"] = token
    elif field == "rejected":
        kwargs["rejected_paths"] = token
    else:
        kwargs["context"] = token
    res = m.record_decision_entry(**kwargs)
    assert res["code"] == "parse_failed"
    assert _read(config) == before


@pytest.mark.parametrize("safe", [
    "# single hash H1 is fine",
    "#### deep heading is fine",
    "  ## indented heading is fine",
    "midline ## hashes are fine",
])
def test_narrow_header_rejection_allows_safe_markdown(ws, safe) -> None:
    """Single #, ####, indented or mid-line ## are SAFE and must commit (no over-rejection)."""
    config, m = ws
    res = m.record_decision_entry("Use markdown in context.", "no markdown", [], context=safe, slug=f"md-{abs(hash(safe))%9999}")
    assert "error" not in res, res


def test_crlf_normalised_for_hash(ws) -> None:
    """The same decision with \\r\\n vs \\n endings yields the same node id (idempotent)."""
    config, m = ws
    a_crlf = "Line one.\r\nLine two."
    a_lf = "Line one.\nLine two."
    r1 = m.record_decision_entry(a_crlf, "rej\r\nmore", [], slug="crlf")
    r2 = m.record_decision_entry(a_lf, "rej\nmore", [], slug="crlf")
    assert "error" not in r1
    assert r2["status"] == "exists"
    assert r1["id"] == r2["id"]


def test_marker_replace_count_one(ws) -> None:
    """A second marker occurrence in the buffer is not corrupted (replace count=1)."""
    config, m = ws
    # Smuggle a second marker into the buffer (simulating a legacy/manual dup).
    marker = "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->"
    content = _read(config) + f"\n\n### legacy\n\n**Decided:** old\n**Rejected:** old\n{marker}\n"
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write(content)
    res = m.record_decision_entry("New decision here.", "rejected stuff", [], slug="new-one")
    assert "error" not in res
    # The new entry is inserted under the FIRST (header) marker only — exactly one new entry header.
    assert _read(config).count("### new-one") == 1


def test_slug_determinism_and_sorted_mechanism_idempotency(ws) -> None:
    """Same axiom+slug → same identity; mechanism order does not change identity (hash sorts mechanisms)."""
    config, m = ws
    r1 = m.record_decision_entry("We MUST use SQLite!!!", "pgvector", [], slug="we-must-use-sqlite",
                                 mechanisms=["sqlite", "wal"])
    assert r1["slug"] == "we-must-use-sqlite"
    r2 = m.record_decision_entry("We MUST use SQLite!!!", "pgvector", [], slug="we-must-use-sqlite",
                                 mechanisms=["wal", "sqlite"])
    assert r2["status"] == "exists" and r2["id"] == r1["id"]


def test_buffer_append_newest_first(ws) -> None:
    """Entries land directly under the marker, newest first."""
    config, m = ws
    m.record_decision_entry("Decision A.", "Reject A.", [], slug="dec-a")
    m.record_decision_entry("Decision B.", "Reject B.", [], slug="dec-b")
    content = _read(config)
    body = content.split("BEGIN ENTRIES", 1)[1]
    assert body.find("### dec-b") < body.find("### dec-a")


# --------------------------------------------------------------------------- #
# Integration
# --------------------------------------------------------------------------- #

def test_full_read_write_loop(ws) -> None:
    """record → decisions.md → SQLite → slug-queryable via query_decisions."""
    config, m = ws
    res = m.record_decision_entry("Adopt event sourcing.", "CRUD loses history.", ["arch"], slug="event-sourcing")
    assert res["status"] == "created"
    assert "### event-sourcing" in _read(config)
    assert GraphStore(config.db_path).get_node_by_slug("event-sourcing") is not None

    # query_decisions resolves the exact slug without needing embeddings.
    with patch("mitos.mcp_server.MitosConfig", return_value=config):
        from mitos.mcp_server import query_decisions
        out = json.loads(query_decisions("event-sourcing", project=config.workspace_dir))
    assert out["slug"] == "event-sourcing"
    assert out["rejected_paths"] == "CRUD loses history."  # anti-knowledge flows end-to-end


def test_idempotency_e2e_no_buffer_dup(ws) -> None:
    """Recording the identical decision twice → one node, 2nd 'exists', buffer has it once."""
    config, m = ws
    a, r = "Write a test suite.", "Manual verification."
    r1 = m.record_decision_entry(a, r, ["testing"], slug="test-suite")
    r2 = m.record_decision_entry(a, r, ["testing"], slug="test-suite")
    assert r1["status"] == "created" and r2["status"] == "exists" and r1["id"] == r2["id"]
    assert len(GraphStore(config.db_path).get_all_nodes()) == 1
    assert _read(config).count("### test-suite") == 1


def test_supersedes_e2e(ws) -> None:
    """record B with supersedes=A → A computed-superseded, B active, one supersedes edge."""
    config, m = ws
    ra = m.record_decision_entry("Axiom A.", "Reject A.", [], slug="dec-a")
    rb = m.record_decision_entry("Axiom B.", "Reject B.", [], supersedes="dec-a", slug="dec-b")
    assert "error" not in rb and rb["status"] == "created"
    store = GraphStore(config.db_path)
    # V1a single-node state derivation (8a): the prototype compute_all_states DAG retired.
    assert store.get_node_state(ra["id"]) == "superseded"
    assert store.get_node_state(rb["id"]) == "active"
    conn = store._get_connection()
    try:
        # V1a edge columns: edge_type / source_id / target_id (was type / from_id / to_id).
        edges = conn.execute("SELECT * FROM edges WHERE edge_type='supersedes'").fetchall()
        assert len(edges) == 1
        assert edges[0]["source_id"] == rb["id"] and edges[0]["target_id"] == ra["id"]
    finally:
        conn.close()


def test_supersedes_not_found_buffer_unchanged(ws) -> None:
    """Unknown supersedes slug → supersedes_not_found, nothing written, buffer untouched."""
    config, m = ws
    before = _read(config)
    res = m.record_decision_entry("New.", "Old.", [], slug="new-decision", supersedes="ghost-slug")
    assert res["code"] == "supersedes_not_found"
    assert _read(config) == before
    assert len(GraphStore(config.db_path).get_all_nodes()) == 0


def test_supersedes_fuzzy_guard(ws) -> None:
    """A prefix (not exact) supersedes target → supersedes_not_found, not a wrong-node edge."""
    config, m = ws
    m.record_decision_entry("Decision foo bar.", "no", [], slug="foo-bar")
    res = m.record_decision_entry("Tries to supersede a prefix.", "no", [], slug="prefix-superseder",
                                  supersedes="foo")
    assert res["code"] == "supersedes_not_found"


def test_slug_collision_returns_error_and_keeps_read_tools_intact(ws) -> None:
    """Exact slug, different axiom, no supersedes → slug_collision; NO duplicate node; reads intact."""
    config, m = ws
    r1 = m.record_decision_entry("Axiom version one.", "Reject.", [], slug="dup")
    before = _read(config)
    r2 = m.record_decision_entry("Axiom version two.", "Reject.", [], slug="dup")
    assert r2["code"] == "slug_collision"
    assert _read(config) == before  # rejected before any write
    store = GraphStore(config.db_path)
    # Exactly one node holds the slug, and get_node_by_slug does NOT raise.
    conn = store._get_connection()
    try:
        rows = conn.execute("SELECT id FROM nodes WHERE slug='dup'").fetchall()
        assert len(rows) == 1
    finally:
        conn.close()
    assert store.get_node_by_slug("dup")["id"] == r1["id"]


def test_slug_prefix_is_not_a_collision(ws) -> None:
    """A new slug that is a prefix of an existing one commits normally (fuzzy match must not block)."""
    config, m = ws
    m.record_decision_entry("Use SQLite WAL.", "no", [], slug="use-sqlite-wal")
    # acknowledge_neighbors: with live embeddings this near-twin pair lands in the
    # strong-match band the 0.80 pause floor now catches (ADR
    # `record-pause-floor-lowered-to-strong-match-band`); the pause is not this
    # test's subject — slug-prefix collision logic is.
    res = m.record_decision_entry("Use SQLite generally.", "no", [], slug="use-sqlite",
                                  acknowledge_neighbors=True)
    assert "error" not in res and res["status"] == "created"


@patch("mitos.sync.QdrantVectorStore")
@patch("mitos.sync.GeminiEmbeddingProvider")
def test_scope_overflow_summary_after_receipt_then_debounced(mock_provider, mock_vector, ws,
                                                            monkeypatch) -> None:
    """An over-ceiling render attaches ONE debounced `scope_overflow` summary to the result.

    Reproduces the AX complaint and pins the fix end-to-end on the shared write path
    (so both the CLI and MCP surfaces inherit it): the receipt fields are always intact,
    the size nudge is a single line pointing at `mitos status` (not the per-write wall),
    and a second record in the same workspace within the window is silent.

    Since the per-scope degrade (2c) an over-ceiling scope file becomes an index, and at
    the default ceilings this one decision's index fits, so nothing would be over. The
    scope ceiling is squeezed below the index itself, which is still reported.
    """
    import mitos.renderer as R
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", 200)
    config, _ = ws
    # Degrade the backends → no network and no P4 near-duplicate pause (which needs
    # embeddings), isolating the overflow-presentation behaviour under test.
    mock_provider.side_effect = Exception("provider down")
    mock_vector.side_effect = Exception("qdrant down")
    m = MitosSyncManager(config)

    big_axiom = "We persist an enormous rationale here. " * 1600  # > 50,000 chars
    first = m.record_decision_entry(big_axiom, "Smaller buffers.", ["substrate"], slug="huge-one")
    assert "error" not in first and first["status"] == "created"
    # Receipt fields are present and intact — never buried or dropped.
    assert first["slug"] == "huge-one" and first["state"] == "active"
    # Exactly one debounced summary line, pointing at the health surface for detail.
    assert "scope_overflow" in first
    assert "mitos status" in first["scope_overflow"]

    # A second record in the same workspace within the 24h window is silent (debounced),
    # even though the corpus is still over the ceiling.
    second = m.record_decision_entry("A small follow-up axiom.", "Nothing.", ["substrate"], slug="small-two")
    assert "error" not in second
    assert "scope_overflow" not in second


@patch("mitos.sync.QdrantVectorStore")
@patch("mitos.sync.GeminiEmbeddingProvider")
def test_no_scope_overflow_field_when_within_budget(mock_provider, mock_vector, ws) -> None:
    """A normal-sized decision records cleanly with NO scope_overflow field."""
    config, _ = ws
    mock_provider.side_effect = Exception("provider down")
    mock_vector.side_effect = Exception("qdrant down")
    m = MitosSyncManager(config)
    res = m.record_decision_entry("Use a small, bounded axiom.", "Sprawl.", ["substrate"], slug="tidy")
    assert "error" not in res and res["status"] == "created"
    assert "scope_overflow" not in res


@patch("mitos.sync.QdrantVectorStore")
@patch("mitos.sync.GeminiEmbeddingProvider")
def test_graceful_degradation(mock_provider, mock_vector, ws) -> None:
    """Embedding backend down → node commits, embedding 'pending', outbox row present."""
    config, _ = ws
    mock_provider.side_effect = Exception("provider down")
    mock_vector.side_effect = Exception("qdrant down")
    m = MitosSyncManager(config)  # rebuilt so the patched providers apply
    res = m.record_decision_entry("Degrade gracefully.", "Crash.", ["reliability"], slug="degrade")
    assert "error" not in res
    assert res["embedding"] == "pending"
    pending = GraphStore(config.db_path).get_pending_embeddings()
    assert any(p["node_id"] == res["id"] for p in pending)


@patch("mitos.sync.QdrantVectorStore")
@patch("mitos.sync.GeminiEmbeddingProvider")
def test_write_path_warnings_go_to_stderr_not_stdout(mock_provider, mock_vector, ws, capsys) -> None:
    """With the backend down, the embedding-deferral warning lands on stderr, never stdout.

    The MCP write tool (record_decision) shares this code path and uses stdout for its
    JSON-RPC channel, so any stray stdout line there corrupts the protocol — every
    write-path warning must go to stderr.
    """
    config, _ = ws
    mock_provider.side_effect = Exception("provider down")
    mock_vector.side_effect = Exception("qdrant down")
    m = MitosSyncManager(config)
    res = m.record_decision_entry("Defer the embedding cleanly.", "Crash.", ["reliability"], slug="defer-clean")
    assert "error" not in res and res["embedding"] == "pending"
    captured = capsys.readouterr()
    assert "[Warning]" not in captured.out  # stdout stays clean for the MCP JSON-RPC channel
    assert "Embedding upsert deferred" in captured.err
    assert "defer-clean" in captured.err


@patch("mitos.sync.QdrantVectorStore")
@patch("mitos.sync.GeminiEmbeddingProvider")
def test_a_sweep_failure_on_the_record_path_leaves_the_commit_and_stdout_clean(
        mock_provider, mock_vector, ws, capsys, monkeypatch) -> None:
    """E7 (2e): the render's stale-scope sweep fails inside a record; the entry is
    committed, stdout (the MCP JSON-RPC channel) stays clean, and the warning is on
    stderr. The next record, with nothing refused, finishes the job."""
    config, _ = ws
    mock_provider.side_effect = Exception("provider down")
    mock_vector.side_effect = Exception("qdrant down")
    m = MitosSyncManager(config)  # rebuilt so the patched providers apply
    axioms = os.path.join(config.workspace_dir, ".mitos", "axioms")
    os.makedirs(axioms, exist_ok=True)
    gone = os.path.join(axioms, "gone.md")
    with open(gone, "w", encoding="utf-8") as f:
        f.write("# Active Axioms for Scope: gone\nStale body.\n")

    real_remove, fired = os.remove, []

    def refuse_gone(path, *args, **kwargs):
        if path == gone:
            fired.append(path)
            raise PermissionError(13, "Permission denied", path)
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(os, "remove", refuse_gone)
    res = m.record_decision_entry("Sweep failures never fail a write.", "Raising.",
                                  ["reliability"], slug="sweep-refused")
    monkeypatch.setattr(os, "remove", real_remove)

    assert fired == [gone]
    assert "error" not in res and res["status"] == "created"
    assert GraphStore(config.db_path).get_node(res["id"]) is not None
    captured = capsys.readouterr()
    assert "[Warning]" not in captured.out
    assert "gone.md" in captured.err
    assert os.path.exists(gone)

    m.record_decision_entry("A second write retries the sweep.", "Nothing.",
                            ["reliability"], slug="sweep-retried")
    assert not os.path.exists(gone)


# --------------------------------------------------------------------------- #
# MCP boundary
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_mcp_advertises_three_tools() -> None:
    """The server advertises record_decision alongside the two read tools."""
    from mitos.mcp_server import mcp
    names = [t.name for t in await mcp.list_tools()]
    assert "record_decision" in names
    assert "surface_decisions" in names and "query_decisions" in names


def test_mcp_tool_returns_well_formed_json(ws) -> None:
    """The MCP tool returns parseable JSON for both success and error, via a writable store."""
    config, _ = ws
    with patch("mitos.mcp_server.MitosConfig", return_value=config):
        from mitos.mcp_server import record_decision
        ok = json.loads(record_decision("A decision.", "A rejection.", ["s"], slug="mcp-ok", project=config.workspace_dir))
        assert ok["status"] == "created" and ok["slug"] == "mcp-ok"
        err = json.loads(record_decision("Another.", "", ["s"], slug="mcp-err", project=config.workspace_dir))  # missing rejected_paths
        assert err["code"] == "missing_rejected_paths"
    # The write actually landed through the MCP entry point (writable store).
    assert GraphStore(config.db_path).get_node_by_slug("mcp-ok") is not None


# --------------------------------------------------------------------------- #
# Adversarial
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("exc", [DatabaseError("boom"), OSError("disk full")])
def test_commit_failed_rolls_back_buffer(ws, exc) -> None:
    """A commit/write failure returns commit_failed AND restores decisions.md byte-for-byte."""
    config, m = ws
    before = _read(config)
    with patch.object(m.store, "commit_parsed_entry", side_effect=exc):
        res = m.record_decision_entry("Will fail to commit.", "Rejection.", [], slug="will-fail")
    assert res["code"] == "commit_failed"
    assert _read(config) == before  # rolled back, no orphan
    assert GraphStore(config.db_path).get_node_by_slug("will-fail") is None


def _temps(config: MitosConfig) -> list:
    directory = os.path.dirname(config.decisions_file)
    return [n for n in os.listdir(directory) if n.endswith(".tmp")]


def test_record_preserves_the_buffer_mode(ws) -> None:
    """The atomic replace must not re-mode the gold source to the temp file's mode."""
    config, m = ws
    os.chmod(config.decisions_file, 0o640)
    res = m.record_decision_entry("Mode survives.", "Rejection.", ["s"], slug="mode-ok")
    assert res["status"] == "created"
    assert os.stat(config.decisions_file).st_mode & 0o777 == 0o640
    assert "mode-ok" in _read(config)


def test_forward_buffer_write_failure_returns_commit_failed_and_leaves_buffer_whole(ws) -> None:
    """A full disk on the forward write: commit_failed, buffer untouched, no node, no temp.

    Injected at the real primitive's replace, so the temp file is genuinely created and
    must be cleaned. The spy proves the failing call is the forward write carrying the
    entry, not an auto-heal write (the `ws` header is canonical, so the heal makes none).
    """
    from mitos import atomic_file

    config, m = ws
    before = _read(config)
    contents = []
    real_write_source = atomic_file.write_source
    real_replace = os.replace
    replaces = {"n": 0}

    def _spy(path, content):
        contents.append(content)
        return real_write_source(path, content)

    def _fail_first_replace(src, dst):
        replaces["n"] += 1
        if replaces["n"] == 1:
            raise OSError(28, "No space left on device")
        return real_replace(src, dst)

    with patch("mitos.atomic_file.write_source", side_effect=_spy), \
            patch("mitos.atomic_file.os.replace", side_effect=_fail_first_replace):
        res = m.record_decision_entry("Never lands.", "Rejection.", [], slug="never-lands")

    assert res["code"] == "commit_failed"
    assert len(contents) == 2, "forward write, then rollback write"
    assert "never-lands" in contents[0] and contents[0] != before
    assert _read(config) == before
    assert GraphStore(config.db_path).get_node_by_slug("never-lands") is None
    assert _temps(config) == []


def test_refused_directory_fsync_does_not_turn_a_record_into_a_failed_rollback(ws) -> None:
    """A mount that refuses directory fsync must still record cleanly.

    Were the directory step fatal, the forward write would land and raise, the rollback
    would land and raise, and the method would report "rollback failed" over a whole,
    correctly rolled-back file — on every write on that mount.
    """
    import stat as _stat

    config, m = ws
    real_fsync = os.fsync
    refused = []

    def _fsync(fd):
        if _stat.S_ISDIR(os.fstat(fd).st_mode):
            refused.append(fd)
            raise OSError(22, "Invalid argument")
        return real_fsync(fd)

    with patch("mitos.atomic_file.os.fsync", side_effect=_fsync):
        res = m.record_decision_entry("Lands anyway.", "Rejection.", ["s"], slug="lands-anyway")

    assert refused, "the directory fsync was never attempted"
    assert "error" not in res and res["status"] == "created"
    assert GraphStore(config.db_path).get_node_by_slug("lands-anyway") is not None
    assert "lands-anyway" in _read(config)


def test_concurrent_distinct_slugs_all_land(ws) -> None:
    """Five threads recording distinct decisions all commit with no buffer corruption."""
    config, m = ws

    def rec(i: int):
        # These template axioms are near-identical, so with live embeddings the P4
        # review would (correctly) flag them as look-alikes; this test is about
        # concurrent buffer integrity, not dedup, so acknowledge past the review.
        return m.record_decision_entry(f"Decision number {i}.", f"Rejection {i}.", ["c"],
                                       slug=f"con-{i}", acknowledge_neighbors=True)

    threads, results = [], {}
    for i in range(5):
        t = threading.Thread(target=lambda i=i: results.__setitem__(i, rec(i)))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all("error" not in r for r in results.values())
    parsed = parse_decisions_file(_read(config), errors=[])
    assert len([p for p in parsed if p.slug.startswith("con-")]) == 5


def test_toctou_same_slug_different_axiom(ws) -> None:
    """The in-lock recheck stops a same-slug/different-axiom racer from making a duplicate slug.

    Simulates the race deterministically: a racer commits a different-axiom node under the same
    slug at lock-acquisition time — i.e. AFTER Phase A's fast-fail but BEFORE the in-lock recheck —
    so only the in-lock recheck can catch it. The call must return slug_collision, no duplicate.
    """
    config, m = ws
    other = MitosSyncManager(config)
    real_lock = m.lock

    class InjectingLock:
        def __enter__(self):
            # A racer lands the colliding node in the window between Phase A and the recheck.
            other.store.commit_parsed_entry(_mk_entry("racer axiom", "toctou"))
            return real_lock.__enter__()

        def __exit__(self, *exc):
            return real_lock.__exit__(*exc)

    m.lock = InjectingLock()
    res = m.record_decision_entry("our axiom", "rej", [], slug="toctou")

    assert res["code"] == "slug_collision"
    store = GraphStore(config.db_path)
    conn = store._get_connection()
    try:
        assert len(conn.execute("SELECT id FROM nodes WHERE slug='toctou'").fetchall()) == 1
    finally:
        conn.close()
    assert store.get_node_by_slug("toctou") is not None  # does not raise


def test_supersedes_ambiguous(ws) -> None:
    """A supersedes target matching >1 same-casefold-slug lineage node → supersedes_ambiguous.

    The V1a ambiguity trigger is a same-slug supersession lineage (MI-13), not the
    retired fuzzy-prefix tier: node-2 supersedes node-1 while both keep slug 'amb', so
    the all-nodes resolve_slug('amb') returns 2 ids (only node-2 is active).
    """
    config, m = ws
    m.store.commit_parsed_entry(_mk_entry("axiom one", "amb"))      # node-1, slug 'amb'
    e2 = _mk_entry("axiom two", "amb")
    e2.supersedes = ["amb"]                                           # resolves to node-1 (active non-self)
    m.store.commit_parsed_entry(e2)                                # node-2 supersedes node-1; both slug 'amb'
    before = _read(config)
    res = m.record_decision_entry("New decision.", "Rejection.", [], slug="new-decision", supersedes="amb")
    assert res["code"] == "supersedes_ambiguous"
    assert _read(config) == before


def test_supersedes_accepts_cased_non_ascii_lithuanian(ws) -> None:
    """A cased non-ASCII supersedes target (Lithuanian 'KABUTĖ' → 'kabutė') commits.

    Guards the resolve_slug layer: pre-fix SQLite COLLATE NOCASE cannot fold Ė/ė, so a
    legal kill-edge was spuriously rejected (supersedes_not_found). Post-fix casefold
    resolves it and the supersession commits. (Targets reach resolve_slug un-slugified —
    only the new entry's own slug is slugified — so the cased literal is what's matched.)
    """
    config, m = ws
    m.store.commit_parsed_entry(_mk_entry("axiom one", "kabutė"))  # node-1, cased non-ASCII slug
    node1_id = GraphStore(config.db_path).get_node_by_slug("kabutė")["id"]
    res = m.record_decision_entry("New axiom.", "Reject.", [], supersedes="KABUTĖ", slug="kabute-v2")
    assert "error" not in res and res["status"] == "created"
    store = GraphStore(config.db_path)
    assert store.get_node_state(node1_id) == "superseded"
    conn = store._get_connection()
    try:
        edges = conn.execute("SELECT * FROM edges WHERE edge_type='supersedes'").fetchall()
        assert len(edges) == 1
        assert edges[0]["source_id"] == res["id"] and edges[0]["target_id"] == node1_id
    finally:
        conn.close()


def test_supersedes_casefold_distinguishes_from_lower_german_ss(ws) -> None:
    """German ß: 'straße'.casefold()=='strasse', so supersede via 'STRASSE' must be accepted.

    This is the ONLY test that catches a regression of the sync.py re-filter back to
    ``.lower()``: ``"straße".lower()=="straße" != "STRASSE".lower()=="strasse"`` would
    reject, whereas both ``.casefold()`` to ``"strasse"``. (Lithuanian alone does not
    catch it — ``.lower()`` folds ``Ė`` fine; only ß/Greek diverge under ``.lower()``.)
    """
    config, m = ws
    m.store.commit_parsed_entry(_mk_entry("axiom one", "straße"))  # slug_casefold == "strasse"
    node1_id = GraphStore(config.db_path).get_node_by_slug("straße")["id"]
    res = m.record_decision_entry("New axiom.", "Reject.", [], supersedes="STRASSE", slug="strasse-v2")
    assert "error" not in res and res["status"] == "created"
    store = GraphStore(config.db_path)
    assert store.get_node_state(node1_id) == "superseded"
    conn = store._get_connection()
    try:
        edges = conn.execute("SELECT * FROM edges WHERE edge_type='supersedes'").fetchall()
        assert len(edges) == 1
        assert edges[0]["source_id"] == res["id"] and edges[0]["target_id"] == node1_id
    finally:
        conn.close()


def test_corrects_accepts_cased_non_ascii(ws) -> None:
    """A cased non-ASCII corrects target ('KABUTĖ' → 'kabutė') commits (kill-edge twin)."""
    config, m = ws
    m.store.commit_parsed_entry(_mk_entry("axiom one", "kabutė"))
    node1_id = GraphStore(config.db_path).get_node_by_slug("kabutė")["id"]
    res = m.record_decision_entry("New axiom.", "Reject.", [], corrects="KABUTĖ", slug="kabute-fix")
    assert "error" not in res and res["status"] == "created"
    store = GraphStore(config.db_path)
    assert store.get_node_state(node1_id) == "corrected"
    conn = store._get_connection()
    try:
        edges = conn.execute("SELECT * FROM edges WHERE edge_type='corrects'").fetchall()
        assert len(edges) == 1
        assert edges[0]["source_id"] == res["id"] and edges[0]["target_id"] == node1_id
    finally:
        conn.close()


def test_relation_target_accepts_cased_non_ascii(ws) -> None:
    """A cased non-ASCII relation target ('amends'='KABUTĖ' → 'kabutė') passes pre-validation.

    Covers ``_validate_relation_target`` (sync.py:828): pre-fix ``.lower()`` rejected it
    as ``relation_target_not_found``; post-fix ``.casefold()`` resolves it and the record
    commits (the non-kill amends edge now commits in V1b; only ``status`` is asserted here).
    """
    config, m = ws
    m.store.commit_parsed_entry(_mk_entry("axiom one", "kabutė"))
    res = m.record_decision_entry("New axiom.", "Reject.", [], amends="KABUTĖ", slug="kabute-amend")
    assert "error" not in res and res["status"] == "created"


def test_pathological_inputs(ws) -> None:
    """Large fields, unicode, and empty scope/mechanisms commit cleanly."""
    config, m = ws
    res = m.record_decision_entry(
        axiom="Adopt ünîçödé and a very long rationale " + ("x" * 5000),
        rejected_paths="Reject ☃ — " + ("y" * 5000),
        scope=[], mechanisms=[], slug="unicode-huge",
    )
    assert "error" not in res and res["status"] == "created"
    assert GraphStore(config.db_path).get_node_by_slug("unicode-huge") is not None


def _mk_entry(axiom: str, slug: str):
    """Builds a minimal committable decision ParsedEntry for racing/ambiguity setup."""
    from mitos.parser import ParsedEntry
    e = ParsedEntry("decision", slug, 0, 0)
    e.axiom = axiom
    e.rejected_paths = "setup rejection"
    return e


# --------------------------------------------------------------------------- #
# Receipt enrichment: edges_created + resolved scope/mechanisms (write facts)
# --------------------------------------------------------------------------- #

def test_receipt_carries_committed_edges_and_resolved_fields(ws) -> None:
    """The "created" receipt echoes the edges the commit actually wired (incl. a
    comma-split multi-target flag), scope as normalised and mechanisms as authored
    (stripped, empties dropped) — write facts read back from the committed entry,
    not the raw input args."""
    config, m = ws
    for slug in ("old-a", "old-b", "cited-c"):
        # acknowledge_neighbors: the seeds are near-twins of each other; with live
        # embeddings the 0.80 pause floor would otherwise pause the later seeds.
        assert m.record_decision_entry(f"Prior axiom {slug}.", "rej", [], slug=slug,
                                       acknowledge_neighbors=True)["status"] == "created"
    res = m.record_decision_entry(
        axiom="Unifying axiom.", rejected_paths="rej",
        scope=[" db ", "", "auth"], mechanisms=[" sqlite ", ""],
        supersedes="old-a, old-b", cites="cited-c", slug="unifier",
    )
    assert res["status"] == "created"
    # Echo of what was committed (whitespace stripped, empties dropped).
    assert res["scope"] == ["db", "auth"]
    assert res["mechanisms"] == ["sqlite"]
    # Edge facts, one per wired edge; order-insensitive compare.
    got = {(e["kind"], e["target"]) for e in res["edges_created"]}
    assert got == {("supersedes", "old-a"), ("supersedes", "old-b"),
                   ("cites", "cited-c")}
    # And they match the committed graph exactly.
    store = GraphStore(config.db_path)
    node_id = store.resolve_slug("unifier")[0]
    committed = {(e["kind"], e["target"]) for e in store.get_outgoing_edges(node_id)}
    assert got == committed


def test_receipt_edges_created_empty_on_bare_record(ws) -> None:
    """No relation flags → edges_created is present and empty (a fact, not an omission)."""
    config, m = ws
    res = m.record_decision_entry("A lone axiom.", "rej", [], slug="lone")
    assert res["status"] == "created"
    assert res["edges_created"] == []
    assert res["scope"] == [] and res["mechanisms"] == []


# The full `amends` echo the two receipt rows below pin (B4): an acting relation
# carries state, stamps and the whole axiom, in that key order.
_PRIOR_AMENDS_ECHO = {
    "kind": "amends", "target": "prior", "target_state": "active",
    "target_stamps": {"amended_by": ["follow-up"]}, "target_axiom": "Prior axiom.",
}


def test_cli_json_receipt_carries_edges_and_fields(ws, capsys) -> None:
    """`mitos record --json` emits the enriched receipt verbatim."""
    from mitos.cli import cmd_record
    config, m = ws
    m.record_decision_entry("Prior axiom.", "rej", [], slug="prior")
    cmd_record(config, axiom="Follow-up axiom.", rejected="rej", scope=["db"],
               mechanisms=["sqlite"], amends="prior", slug="follow-up", as_json=True)
    payload = json.loads(capsys.readouterr().out)
    # B4 (R1): the stamped identification echo — prior's state computed now, the
    # amended_by stamp this very write put on it, and its whole axiom.
    assert payload["edges_created"] == [_PRIOR_AMENDS_ECHO]
    assert payload["scope"] == ["db"] and payload["mechanisms"] == ["sqlite"]


def test_cli_text_receipt_renders_edges_scope_mechanisms(ws, capsys) -> None:
    """The human receipt prints Edges/Scope/Mechanisms lines after Handle, and
    omits them all on a bare record."""
    from mitos.cli import cmd_record
    config, m = ws
    m.record_decision_entry("Prior axiom.", "rej", [], slug="prior")
    cmd_record(config, axiom="Follow-up axiom.", rejected="rej", scope=["db", "auth"],
               mechanisms=["sqlite"], amends="prior", slug="follow-up")
    out = capsys.readouterr().out
    assert "Edges:     amends → prior" in out
    assert "Scope:     db, auth" in out
    assert "Mechanisms: sqlite" in out
    # Bare record: no empty Edges/Scope/Mechanisms lines. (acknowledge_neighbors:
    # under live embeddings the earlier entries sit in the 0.80 pause band.)
    cmd_record(config, axiom="A lone axiom.", rejected="rej", slug="lone",
               acknowledge_neighbors=True)
    out = capsys.readouterr().out
    assert "Edges:" not in out and "Scope:" not in out and "Mechanisms:" not in out


def test_mcp_receipt_carries_edges_and_fields(ws) -> None:
    """The MCP record_decision result carries the same enrichment (CLI⇄MCP sync)."""
    from mitos import mcp_server
    config, m = ws
    m.record_decision_entry("Prior axiom.", "rej", [], slug="prior")
    with patch("mitos.mcp_server.MitosConfig", return_value=config):
        res = json.loads(mcp_server.record_decision(
            "Follow-up axiom.", "rej", ["db"], slug="follow-up",
            mechanisms=["sqlite"], amends="prior", project=config.workspace_dir))
    assert res["status"] == "created"
    assert res["edges_created"] == [_PRIOR_AMENDS_ECHO]
    assert res["scope"] == ["db"] and res["mechanisms"] == ["sqlite"]


#: What an MCP string may never carry: a `mitos <verb>` command, a `-p` selector in a
#: code span, the two repair verbs, a CLI flag. Token-level, so a sentence ending in
#: "mitos" or the tool's own backticked name cannot pass or fail it by accident.
_SHELL_TOKEN_PATTERNS = (r"\bmitos\s+[a-z-]+", r"`[^`]*\s-p\b[^`]*`", r"restore-source",
                         r"\brebuild\b", r"--")


def _shell_tokens(text: str) -> list:
    """Returns the shell-token patterns ``text`` matches; empty means MCP-clean."""
    return [pat for pat in _SHELL_TOKEN_PATTERNS if re.search(pat, text)]


def _names_the_unreachable_pair(note: str) -> bool:
    """Both states the tool cannot reach, as facts, alike to the tool, a person next."""
    return ("no source block" in note and "decisions/archive/" in note
            and "cannot tell the two apart" in note and "a person" in note)


def test_exists_receipt_reports_the_no_op_without_dropping_the_pointer(ws) -> None:
    """A re-record writes nothing and must say so — while still pointing at the entry.

    Regression: the ``exists`` short-circuit's ``path`` was rendered as ``Written: …``
    under a ``Recorded ✓`` headline, so a caller correcting commentary — or restoring
    a source block for a graph-only node — read it as a successful write while
    nothing had changed. The fix moves the *claim* into ``no_op_reason``; the
    ``path`` pointer stays, since "already recorded — so where is it?" is a fair
    question (the #5b contract in test_payload_economy.py).
    """
    config, m = ws
    r1 = m.record_decision_entry("Pin the digest length.", "Leave it to the implementer.",
                                 [], slug="pin-digest-length")
    assert r1["status"] == "created" and r1.get("path")
    assert "no_op_reason" not in r1, "a real write must not claim to be a no-op"

    r2 = m.record_decision_entry("Pin the digest length.", "CORRECTED rejected text.",
                                 [], slug="pin-digest-length")
    assert r2["status"] == "exists"
    assert r2["path"] == config.decisions_file, "the pointer stays (#5b)"
    assert r2.get("no_op_reason"), "a no-op must say so in-band"
    # Inverted twice. From the C′ release the note named `mitos sync`, the first path
    # that worked; 0.18.0 shipped the tool built for this, and the dict is an MCP
    # boundary (`record --json` emits it verbatim), so it now names that tool and no
    # shell command. The CLI's recipes are its own (see the exists-recovery rows).
    assert "amend_commentary" in r2["no_op_reason"]
    assert _shell_tokens(r2["no_op_reason"]) == [], r2["no_op_reason"]


def test_exists_no_op_note_states_the_unreachable_pair_and_claims_neither(ws) -> None:
    """The two states the tool cannot reach are a pair of facts, with a person next.

    A node with no source block and an entry rotated into decisions/archive/ both answer
    `archived` from `amend_commentary`, which cannot tell them apart — and the receipt
    does not know either, so it names both and claims neither for this entry. No verb
    for the repair: on MCP a command is one the agent runs.
    """
    config, m = ws
    m.record_decision_entry("Pin the digest length.", "Leave it to the implementer.",
                            [], slug="pin-digest-length")
    note = m.record_decision_entry("Pin the digest length.", "CORRECTED rejected text.",
                                   [], slug="pin-digest-length")["no_op_reason"]

    assert _names_the_unreachable_pair(note), note
    lowered = note.casefold()
    for claim in ("this entry is archived", "this entry is in decisions/archive",
                  "this node has no", "it sits in decisions/archive"):
        assert claim not in lowered, claim
    assert "rebuild" not in lowered and "restore-source" not in lowered


def test_exists_receipt_names_the_fields_it_ignored(ws) -> None:
    """AX round 10's ask, verbatim: *say what it ignored*.

    A re-record aimed at correcting commentary got a clean `(exists) ✓` with no
    indication that the values it carried differed from the stored ones, so the caller
    had to check by hand to learn nothing had changed. Both surfaces reported success
    for an operation that changed nothing.
    """
    config, m = ws
    m.record_decision_entry("Pin the digest length.", "The original reasoning.",
                            ["alpha"], slug="pin-digest-length")

    same = m.record_decision_entry("Pin the digest length.", "The original reasoning.",
                                   ["alpha"], slug="pin-digest-length")
    assert "differs" not in same, "an identical re-record ignored nothing"

    changed = m.record_decision_entry("Pin the digest length.", "CORRECTED reasoning.",
                                      ["alpha", "beta"], slug="pin-digest-length")
    assert changed["differs"] == ["rejected_paths", "scope"], changed

    # A reorder-only re-record is a scope difference too: the first tag is the
    # primary scope, and the stored order is not the order this call carried. (A
    # separate node, because an `exists` re-record never mutates — the one above
    # still holds `["alpha"]`, so reordering against it would test membership.)
    m.record_decision_entry("Order the scope tags.", "A reason.", ["alpha", "beta"],
                            slug="order-the-scope-tags")
    reordered = m.record_decision_entry("Order the scope tags.", "A reason.",
                                        ["beta", "alpha"], slug="order-the-scope-tags")
    assert reordered["differs"] == ["scope"], reordered


def test_exists_no_op_leaves_a_missing_source_block_missing(ws) -> None:
    """Re-recording does NOT restore a graph-only node's source block.

    The state a decision-corpus audit meets: node live in the graph, ``###`` block
    absent from ``decisions.md``. Re-recording is the obvious repair and does not
    work — so the receipt has to say it did not, or an audit records edges against
    targets that dangle on the next rebuild.
    """
    config, m = ws
    m.record_decision_entry("Restore me later.", "Nothing.", [], slug="graph-only-node")
    text = _read(config)
    assert "### graph-only-node" in text
    # Excise the block, leaving the node in the graph — the graph-only state.
    head, _, tail = text.partition("### graph-only-node")
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write(head + tail.partition("\n### ")[1] + tail.partition("\n### ")[2])

    res = m.record_decision_entry("Restore me later.", "Nothing.", [], slug="graph-only-node")
    assert res["status"] == "exists"
    assert "### graph-only-node" not in _read(config), "re-record must not be believed to restore"
    # The dict states the pair as facts; the verb that restores it is the CLI's to
    # name (test_exists_cli_recipes_parse runs this same graph-only state).
    assert _names_the_unreachable_pair(res["no_op_reason"])


# --------------------------------------------------------------------------- #
# The exists recovery, per boundary (B11): the dict speaks MCP, the CLI text its own
# --------------------------------------------------------------------------- #

def _replay_text(config, m, *, axiom="Pin the digest length.", slug="pin-digest-length",
                 rejected="CORRECTED reasoning.", capsys) -> str:
    """Records once through the manager, replays through the CLI text; returns stdout."""
    from mitos.cli import cmd_record
    m.record_decision_entry(axiom, "The original reasoning.", [], slug=slug)
    capsys.readouterr()
    cmd_record(config, axiom=axiom, rejected=rejected, slug=slug,
               acknowledge_neighbors=True)
    out, err = capsys.readouterr()
    assert err == "", err
    return out


def test_mcp_exists_note_names_the_tool_and_no_shell_command(ws) -> None:
    """R1: the real MCP entry's replay names `amend_commentary`, in real argument names."""
    from mitos import mcp_server
    config, _ = ws
    with patch("mitos.mcp_server.MitosConfig", return_value=config):
        json.loads(mcp_server.record_decision(
            "Pin the digest length.", "rej", ["db"], slug="pin-digest-length",
            project=config.workspace_dir))
        res = json.loads(mcp_server.record_decision(
            "Pin the digest length.", "CORRECTED rej", ["db"], slug="pin-digest-length",
            project=config.workspace_dir))
    assert res["status"] == "exists"
    note = res["no_op_reason"]
    assert "`amend_commentary`" in note
    assert _shell_tokens(note) == [], note
    assert "supersedes or amends" in note, "bare argument names, not CLI flags"

    amend_params = set(inspect.signature(mcp_server.amend_commentary).parameters)
    record_params = set(inspect.signature(mcp_server.record_decision).parameters)
    for name in ("rejected_paths", "scope", "invalidates_if", "context", "new_slug"):
        assert re.search(rf"\b{name}\b", note) and name in amend_params, name
    for name in ("supersedes", "amends"):
        assert re.search(rf"\b{name}\b", note) and name in record_params, name
    # Every snake_case word it cites is a real argument, or the tool's own name.
    cited = set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", note))
    assert cited <= amend_params | record_params | {"amend_commentary"}, cited


def test_json_exists_receipt_carries_the_dict_note_verbatim(ws, capsys) -> None:
    """R2: `record --json` is the dict, so it carries the MCP wording (D1)."""
    from mitos.cli import cmd_record
    from mitos.sync import _EXISTS_NO_OP_NOTE
    config, m = ws
    m.record_decision_entry("Pin the digest length.", "rej", [], slug="pin-digest-length")
    capsys.readouterr()
    cmd_record(config, axiom="Pin the digest length.", rejected="other", slug="pin-digest-length",
               acknowledge_neighbors=True, as_json=True)
    res = json.loads(capsys.readouterr().out)
    assert res["status"] == "exists"
    assert res["no_op_reason"] == _EXISTS_NO_OP_NOTE
    assert _shell_tokens(res["no_op_reason"]) == []


def _excise_block(config, slug: str) -> None:
    """Cuts ``### slug``'s block from decisions.md, leaving the node graph-only."""
    text = _read(config)
    head, _, tail = text.partition(f"### {slug}")
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write(head + tail.partition("\n### ")[1] + tail.partition("\n### ")[2])


@pytest.mark.parametrize("state", ["buffer", "graph-only"])
@pytest.mark.parametrize("project", [None, "my proj", "vinga's proj"])
def test_exists_cli_recipes_parse(ws, capsys, project, state) -> None:
    """R3: every recipe the text receipt prints parses through the real parser, selectored.

    Read off the printed stdout, so a plant that changes what is printed is what reds.
    The graph-only state is the one `restore-source` heals; the receipt prints the
    same recipes on it, as conditionals.
    """
    from mitos import cli
    config, m = ws
    if project is not None:
        config.project = project
    if state == "graph-only":
        m.record_decision_entry("Pin the digest length.", "The original reasoning.", [],
                                slug="pin-digest-length")
        _excise_block(config, "pin-digest-length")
    out = _replay_text(config, m, capsys=capsys)
    if state == "graph-only":
        assert "### pin-digest-length" not in _read(config)

    spans = re.findall(r"`(mitos [^`]+)`", out)
    verbs = set()
    for span in spans:
        args = cli._build_parser().parse_args(shlex.split(span)[1:])
        verbs.add(args.command)
        assert args.project_post == config.project, span
        if args.command == "amend-commentary":
            assert args.handle == "pin-digest-length"
        elif args.command == "restore-source":
            assert args.slug == "pin-digest-length"
        elif args.command == "sync":
            assert args.reconcile_entry == ["pin-digest-length"]
    assert verbs == {"amend-commentary", "sync", "restore-source", "rebuild"}, spans
    # The field flags the amend line lists are read off the real subparser, not trusted.
    amend_line = next(line for line in out.splitlines() if "mitos amend-commentary" in line)
    listed = re.findall(r"--[a-z-]+", amend_line.split("`")[2])
    subparsers = next(a for a in cli._build_parser()._actions
                      if hasattr(a, "choices") and isinstance(a.choices, dict))
    real = set(subparsers.choices["amend-commentary"]._option_string_actions)
    assert listed and set(listed) <= real, (listed, sorted(real))


def test_exists_cli_recipes_bind_a_dash_led_slug(ws, capsys) -> None:
    """A hand-authored `### -foo` keeps its dash, and argparse reads `-foo` as a flag.

    The amend recipe falls back to the id (never dash-led); the flag recipes use `=`.
    """
    from mitos import cli
    from mitos.cli import cmd_record
    config, m = ws
    m.store.commit_parsed_entry(_mk_entry("A dash-led decision.", "-dash-led"))
    capsys.readouterr()
    cmd_record(config, axiom="A dash-led decision.", rejected="setup rejection",
               slug="dash-led", acknowledge_neighbors=True)
    out = capsys.readouterr().out
    assert "Decision '-dash-led' already recorded" in out, out
    node_id = re.search(r"ID:\s+([0-9a-f]+)", out).group(1)
    parsed = {a.command: a for a in (cli._build_parser().parse_args(shlex.split(s)[1:])
                                     for s in re.findall(r"`(mitos [^`]+)`", out))}
    assert parsed["amend-commentary"].handle == node_id
    assert parsed["restore-source"].slug == "-dash-led"
    assert parsed["sync"].reconcile_entry == ["-dash-led"]


@pytest.mark.parametrize("rejected", ["The original reasoning.", "CORRECTED reasoning."])
def test_exists_cli_names_amend_commentary_exactly_once(ws, capsys, rejected) -> None:
    """R4: one recipe per receipt, with or without `differs`; `Ignored:` names no command."""
    config, m = ws
    out = _replay_text(config, m, rejected=rejected, capsys=capsys)
    assert out.count("mitos amend-commentary") == 1, out
    ignored = [line for line in out.splitlines() if "Ignored:" in line]
    if rejected.startswith("CORRECTED"):
        assert len(ignored) == 1 and "rejected_paths" in ignored[0]
        assert not re.search(r"\bmitos\s+[a-z-]+", ignored[0]), ignored[0]
    else:
        assert ignored == []


def test_exists_cli_headline_prints_the_shared_fact_not_the_dict_note(ws, capsys) -> None:
    """R5: the CLI prints `EXISTS_NO_OP_FACT` and composes its own recovery."""
    from mitos.sync import EXISTS_NO_OP_FACT, _EXISTS_NO_OP_NOTE
    config, m = ws
    out = _replay_text(config, m, capsys=capsys)
    assert f"Decision 'pin-digest-length' {EXISTS_NO_OP_FACT}" in out
    mcp_recovery = _EXISTS_NO_OP_NOTE[len(EXISTS_NO_OP_FACT):].strip()
    assert mcp_recovery[:60] not in out
    assert "amend_commentary" not in out, "the tool's name is the dict's, not this boundary's"


def test_created_receipt_prints_no_exists_recovery(ws, capsys) -> None:
    """R6: the text tail is shared, and a `created` receipt carries none of it."""
    from mitos.cli import cmd_record
    config, _ = ws
    cmd_record(config, axiom="A fresh decision.", rejected="rej", slug="fresh")
    out, err = capsys.readouterr()
    assert "Recorded decision 'fresh'" in out
    for token in ("amend-commentary", "restore-source", "rebuild", "already recorded"):
        assert token not in out + err, token


def test_toctou_exists_twin_carries_the_same_note(ws) -> None:
    """R7: the Phase-B `exists` return carries the gate-3 note byte for byte.

    A racer commits the same node between the fast-fail and the in-lock recheck, so
    only the twin can answer.
    """
    config, m = ws
    other = MitosSyncManager(config)
    real_lock = m.lock
    raced = []

    class InjectingLock:
        def __enter__(self):
            if not raced:
                other.store.commit_parsed_entry(_mk_entry("Raced axiom.", "raced"))
                raced.append(True)
            return real_lock.__enter__()

        def __exit__(self, *exc):
            return real_lock.__exit__(*exc)

    m.lock = InjectingLock()
    twin = m.record_decision_entry("Raced axiom.", "setup rejection", [], slug="raced")
    assert raced and twin["status"] == "exists", twin
    gate3 = m.record_decision_entry("Raced axiom.", "setup rejection", [], slug="raced")
    assert gate3["status"] == "exists"
    assert twin["no_op_reason"] == gate3["no_op_reason"]


def test_exists_note_starts_with_the_shared_fact() -> None:
    """R8: one fact, two boundaries — the dict's note is the fact plus its recovery."""
    from mitos.sync import EXISTS_NO_OP_FACT, _EXISTS_NO_OP_NOTE
    assert _EXISTS_NO_OP_NOTE.startswith(EXISTS_NO_OP_FACT)
    assert "already recorded" in EXISTS_NO_OP_FACT


# --------------------------------------------------------------------------- #
# The coherence-audit pointer (B2 / T2)
# --------------------------------------------------------------------------- #
#
# The field's WORDING is the guard, not a matter of taste: `mitos check` is the
# tree's sole Anthropic spend and reuses verdicts, so one deferred run covers N
# writes for less, while `_confirm_spend` only fires above ten fresh groups — an
# agent auditing per write presents ~1 forever and the only spend ring in the tree
# never fires. A line reading "audit this write" therefore converts one owed run
# into N (ADR `record-receipt-states-cumulative-audit-debt-not-per-write-work`).
# So the register is a tested contract here rather than the author's ear.

#: Shapes the note must never take. Each is planted into the checker below to prove
#: it is not vacuous — a negative row that cannot red is worse than no row.
#: Recorded verbatim so a later reader re-runs the proof instead of trusting it.
_PLANTED_NOTE_VIOLATIONS = (
    "Coherence debt is standing — run `mitos check -p '.'` to clear it.",  # a command
    "Coherence debt is standing; run the audit before the next write.",   # imperative
    "This decision committed without a contradiction check.",             # per-entry
    "Contradiction coverage is stale — audit this corpus now.",           # "now"
    "You should check the corpus for contradictions.",                    # "you should"
    "3 decisions have not had a full contradiction check.",                # "full"
    "3 decisions have not been checked since the last audit.",            # "since"
)

#: Substrings forbidden anywhere in the shared field, casefolded.
_NOTE_FORBIDDEN_SUBSTRINGS = ("mitos ", "this decision", "this entry",
                              "audit this", "you should")
#: Forbidden as WORDS — a bare `"now" in text` reds on "known" (1a's casefold lesson,
#: one class over), so these match on word boundaries only.
#: "full" and "since" are A4's: a check row cannot tell a scoped sweep from a whole
#: one, and after an upgrade the count holds decisions older than any check.
_NOTE_FORBIDDEN_WORDS = ("run", "now", "full", "since")


def register_violations(text: str, *, substrings=(), words=()):
    """Every forbidden shape ``text`` carries, as a sorted list (empty == compliant).

    The shared half of a register contract: a checker the test feeds BOTH the shipped
    string and a set of planted violations, so a regex that silently stopped matching
    reds its own row instead of passing forever. 1b built this inline for the
    coherence note and left it un-lifted; 2a's pause echo is the second real consumer,
    which is when a helper is extracted rather than speculatively.

    Two traps live here rather than in each caller. ``substrings`` match anywhere
    (casefolded), which is right for a phrase; ``words`` match on **word boundaries**
    only, because a bare ``"now" in text`` reds on *"known"*. And the caller must scope
    what it hands in to its **own** field or sentence — the same response legitimately
    carries forbidden tokens elsewhere (free-prose ``rejected_paths``, a shipped
    ``acknowledge_neighbors=True``), so a whole-payload sweep reds on shipped text.

    Args:
        text: The one string under contract.
        substrings: Phrases forbidden anywhere in it.
        words: Tokens forbidden as whole words.

    Returns:
        The sorted forbidden shapes found; empty means compliant.
    """
    import re
    folded = text.casefold()
    found = [s for s in substrings if s in folded]
    found += [w for w in words
              if re.search(rf"\b{re.escape(w)}\b", folded)]
    return sorted(found)


def _coherence_note_violations(note: str):
    """Every register rule the note breaks, as a sorted list (empty == compliant)."""
    return register_violations(note, substrings=_NOTE_FORBIDDEN_SUBSTRINGS,
                               words=_NOTE_FORBIDDEN_WORDS)


def _created(m, axiom: str, slug: str) -> dict:
    """One created receipt, past the 0.80 pause floor (seeds are near-twins)."""
    res = m.record_decision_entry(axiom, "rej", ["s"], slug=slug,
                                  acknowledge_neighbors=True)
    assert res["status"] == "created", res
    return res


def test_created_receipt_carries_a_non_empty_coherence_audit_string(ws) -> None:
    """Every `created` return carries `coherence_audit`, and it is a non-empty str.

    The type row exists on its own because it is the only thing that reds a later
    "simplification" of the field into the boolean the ADR rejected — a flag would
    be per-entry by position and would assert a coverage fact the receipt cannot
    know. Reasoning in a rejected-alternative cannot survive a rewrite unaided.
    """
    config, m = ws
    res = _created(m, "The receipt states its standing coherence debt.", "coh-created")
    assert isinstance(res["coherence_audit"], str)
    assert res["coherence_audit"].strip()


def test_coherence_note_states_a_standing_corpus_wide_debt(ws) -> None:
    """The register, asserted: no command, no imperative, no per-entry referent.

    Checked over every shape the field can take: the shipped receipt's line, the
    composer's line at several (N, M, K) points (the K clause and N = 0 included)
    and the unreadable fallback. Non-vacuity is proved in-row — every shape in
    ``_PLANTED_NOTE_VIOLATIONS`` is fed to the same checker and must be caught.
    Without that, a checker whose regex silently stopped matching would pass this
    row forever.
    """
    from mitos.sync import _COHERENCE_AUDIT_NOTE, _audit_debt_line
    config, m = ws
    shipped = _created(m, "The register is enforced by a row.", "coh-register")["coherence_audit"]
    rendered = [_audit_debt_line(n, total, k)
                for n, total, k in ((1, 1, 0), (1, 12, 0), (3, 12, 1), (0, 12, 0))]

    for note in [shipped, *rendered, _COHERENCE_AUDIT_NOTE]:
        assert _coherence_note_violations(note) == [], note
        # Wider than the entry it rides — the debt is the corpus's, not this write's.
        assert "corpus" in note.casefold(), note

    # The injection proof: each planted shape must be caught by the same checker.
    for planted in _PLANTED_NOTE_VIOLATIONS:
        assert _coherence_note_violations(planted), (
            f"the register checker is vacuous — it passed: {planted!r}")


def test_exists_receipt_carries_neither_the_field_nor_the_line(ws, capsys) -> None:
    """A re-record wrote nothing, so it incurs no audit debt and says nothing.

    The exit that bites: `cmd_record`'s text tail is SHARED between `created` and
    `exists` (it branches only on the headline and the path label), so an
    unconditional print would put the pointer — and a second recipe — on a no-op.
    The `exists` short-circuit returns above the embedding step, which is why this
    is also the one record exit whose stderr can honestly be asserted empty.
    """
    from mitos.cli import cmd_record
    config, m = ws
    _created(m, "A decision recorded once.", "coh-exists")

    res = m.record_decision_entry("A decision recorded once.", "rej", ["s"],
                                  slug="coh-exists", acknowledge_neighbors=True)
    assert res["status"] == "exists"
    assert "coherence_audit" not in res
    assert "audit_debt" not in res

    capsys.readouterr()
    cmd_record(config, axiom="A decision recorded once.", rejected="rej",
               slug="coh-exists", acknowledge_neighbors=True)
    out, err = capsys.readouterr()
    assert "already recorded" in out
    assert "mitos check" not in out + err
    assert "coherence" not in (out + err).casefold()
    assert err == "", err


def test_pause_and_error_exits_carry_no_coherence_audit(ws) -> None:
    """`needs_review` wrote nothing and an error exit wrote nothing — neither owes it."""
    config, m = ws
    _created(m, "The sync lock is held during commit.", "coh-prior")
    with patch.object(MitosSyncManager, "_review_neighbors",
                      return_value=[{"slug": "coh-prior", "score": 0.9,
                                     "axiom": "The sync lock is held during commit."}]):
        paused = m.record_decision_entry("The sync lock is held for the commit duration.",
                                         "rej", ["s"], slug="coh-paused")
    assert paused["status"] == "needs_review"
    assert "coherence_audit" not in paused
    assert "audit_debt" not in paused
    # A5: the mechanisms echo and its fold map ride `created` only.
    assert "mechanisms" not in paused and "mechanisms_normalized" not in paused

    failed = m.record_decision_entry("An axiom pointing nowhere.", "rej", ["s"],
                                     slug="coh-dangling", supersedes="no-such-slug")
    assert "error" in failed
    assert "coherence_audit" not in failed
    assert "audit_debt" not in failed


def _record_on_both_encodings(config, capsys) -> Tuple[dict, dict]:
    """One `record --json` write, then one MCP write, as their parsed payloads.

    Distinct slugs deliberately: the CLI call COMMITS, so an MCP call replaying the
    same axiom would return `exists` (which carries neither field) and the row
    would compare a receipt against nothing.
    """
    from mitos import mcp_server
    from mitos.cli import cmd_record

    capsys.readouterr()
    cmd_record(config, axiom="The CLI encoding of the receipt.", rejected="rej",
               slug="coh-cli", acknowledge_neighbors=True, as_json=True)
    cli_payload = json.loads(capsys.readouterr().out)

    with patch("mitos.mcp_server.MitosConfig", return_value=config):
        mcp_payload = json.loads(mcp_server.record_decision(
            "The MCP encoding of the receipt.", "rej", ["s"], slug="coh-mcp",
            acknowledge_neighbors=True, project=config.workspace_dir))

    assert cli_payload["status"] == "created" and mcp_payload["status"] == "created"
    return cli_payload, mcp_payload


def test_both_machine_encodings_compose_the_coherence_audit_from_one_source(ws, capsys) -> None:
    """`record --json` and MCP `record_decision` carry the same composer's output.

    The two payloads come from two different writes, so they are NOT equal: the
    MCP write sees one more decision, and both its ``uncovered`` and ``total`` are
    the CLI's +1. ``==`` on the sentences would red on correct code. The parity
    proof is the transition plus single-sourcing: each field equals
    ``_audit_debt_line`` applied to its own payload's ``audit_debt``, so a
    hand-copied second spelling on either surface reds here.
    """
    from mitos.sync import _audit_debt_line
    config, _ = ws
    cli_payload, mcp_payload = _record_on_both_encodings(config, capsys)

    assert set(cli_payload) == set(mcp_payload)
    cli_debt, mcp_debt = cli_payload["audit_debt"], mcp_payload["audit_debt"]
    assert isinstance(cli_debt, dict) and isinstance(mcp_debt, dict)
    assert mcp_debt == {**cli_debt, "uncovered": cli_debt["uncovered"] + 1,
                        "total": cli_debt["total"] + 1}
    for payload in (cli_payload, mcp_payload):
        assert payload["coherence_audit"] == _audit_debt_line(**payload["audit_debt"])
    assert cli_payload["coherence_audit"] != mcp_payload["coherence_audit"]
    # And neither machine surface carries the recovery — the command is the CLI text
    # renderer's alone, because an agent handed a shell command runs it.
    assert "mitos" not in cli_payload["coherence_audit"]
    assert "mitos" not in mcp_payload["coherence_audit"]


def test_both_machine_encodings_carry_null_when_the_debt_is_unreadable(ws, capsys) -> None:
    """Unreadable reaches both encodings the same way: the fallback and JSON ``null``.

    Here the two fields ARE the one constant, so ``==`` is the honest assertion.
    The key is present with ``None``, never absent and never zeros.
    """
    from mitos.audit_debt import DebtUnreadable
    from mitos.sync import _COHERENCE_AUDIT_NOTE
    config, _ = ws
    with patch("mitos.sync.derive_audit_debt",
               return_value=DebtUnreadable("telemetry", "planted")):
        cli_payload, mcp_payload = _record_on_both_encodings(config, capsys)

    for payload in (cli_payload, mcp_payload):
        assert "audit_debt" in payload and payload["audit_debt"] is None
        assert payload["coherence_audit"] == _COHERENCE_AUDIT_NOTE
    assert cli_payload["coherence_audit"] == mcp_payload["coherence_audit"]


# --------------------------------------------------------------------------- #
# The audit-debt count (A4 / T3)
# --------------------------------------------------------------------------- #
#
# Every row runs in ``ws``, which has no telemetry file (``cmd_init`` builds no
# ``TelemetryStore``), so an unmarked corpus reads N = M. Rows that need coverage
# write it through the real run-end seam themselves (``_mark``); none imports
# either ``workspace`` fixture.

def _mark(config: MitosConfig, run_id: str, covered=(), excluded=()) -> None:
    """Writes one undegraded run's coverage through ``record_run_end``, no SQL.

    Builds the telemetry store, so the row calling it owns a telemetry file.
    """
    from mitos.telemetry import CoverageMarks, TelemetryStore
    from test_check_coverage import _check_run_row
    row = _check_run_row(run_id)
    TelemetryStore(config.telemetry_path).record_run_end(
        row, coverage=CoverageMarks(run_id=run_id, marked_at=row.ended_at,
                                    covered=tuple(covered), excluded=tuple(excluded)),
        attempt=None)


def test_every_created_receipt_counts_the_corpus(ws) -> None:
    """With no coverage, each write's receipt reads N = M = the decisions so far.

    Two consecutive receipts therefore differ — the defect A4 logged was forty
    byte-identical ones. The key set is pinned to the vision's names, so a rename
    reds here rather than in a later consumer.
    """
    from mitos.sync import _audit_debt_line
    config, m = ws
    for k in range(3):
        res = _created(m, f"Counted decision number {k}.", f"coh-count-{k}")
        assert set(res["audit_debt"]) == {"uncovered", "total", "excluded"}
        assert res["audit_debt"] == {"uncovered": k + 1, "total": k + 1, "excluded": 0}
        assert res["audit_debt"]["uncovered"] >= 1
        assert res["coherence_audit"] == _audit_debt_line(**res["audit_debt"])
        assert ";" not in res["coherence_audit"]


def test_the_line_inflects_on_its_own_counts() -> None:
    """The verb agrees with N and the noun with M; the K clause does not inflect.

    N = 1 is the common case (a covered corpus plus the write just made), and
    "1 … have" is the first thing an agent would quote back.
    """
    from mitos.sync import _audit_debt_line
    assert _audit_debt_line(1, 1, 0) == (
        "1 of this corpus's 1 decision has not been covered by a completed "
        "contradiction check.")
    assert _audit_debt_line(1, 12, 0) == (
        "1 of this corpus's 12 decisions has not been covered by a completed "
        "contradiction check.")
    assert _audit_debt_line(3, 12, 1) == (
        "3 of this corpus's 12 decisions have not been covered by a completed "
        "contradiction check; 1 more could not be audited (not embedded).")
    assert _audit_debt_line(0, 12, 0) == (
        "0 of this corpus's 12 decisions have not been covered by a completed "
        "contradiction check.")


def test_coverage_through_the_run_end_seam_drops_the_count(ws) -> None:
    """A transition: N = M, then an undegraded run covers every prior, then N = 1.

    Route (i): the marks go through ``TelemetryStore.record_run_end``, the seam a
    real ``mitos check`` writes, with the ids of decisions actually recorded. That
    is lighter than driving ``cmd_check`` with both seams wired, and 2a's module
    already proves the check → coverage half. The exact figures matter: an
    assertion of only ``N < M`` passes against a wrong M.
    """
    config, m = ws
    prior = [_created(m, f"Prior decision {i}.", f"coh-prior-{i}") for i in range(3)]
    assert prior[-1]["audit_debt"] == {"uncovered": 3, "total": 3, "excluded": 0}

    _mark(config, "run-cover", covered=[r["id"] for r in prior])
    res = _created(m, "The write after the audit.", "coh-after")
    assert res["audit_debt"] == {"uncovered": 1, "total": 4, "excluded": 0}
    assert res["coherence_audit"].startswith("1 of this corpus's 4 decisions has not")


def test_an_excluded_active_decision_adds_the_k_clause(ws) -> None:
    """K counts active decisions a check saw but could not audit, beside N."""
    config, m = ws
    covered = _created(m, "A decision the check covered.", "coh-covered")
    excluded = _created(m, "A decision the check could not embed.", "coh-excluded")

    _mark(config, "run-k", covered=[covered["id"]], excluded=[excluded["id"]])
    res = _created(m, "The write after a partial audit.", "coh-after-k")
    assert res["audit_debt"] == {"uncovered": 1, "total": 3, "excluded": 1}
    assert res["coherence_audit"].endswith(
        "; 1 more could not be audited (not embedded).")


def test_unreadable_telemetry_renders_the_fallback_and_null(ws) -> None:
    """A telemetry file that exists and cannot be read: fallback + ``None``.

    And the split beside it: a 0-byte file reads below the coverage rung, which
    is *absent*, so it gives a real count with N = M — never the fallback.
    """
    from mitos.sync import _COHERENCE_AUDIT_NOTE
    config, m = ws
    with open(config.telemetry_path, "wb") as f:
        f.write(b"this is not a sqlite database at all" * 40)
    res = _created(m, "A write beside a broken telemetry file.", "coh-garbage")
    assert res["coherence_audit"] == _COHERENCE_AUDIT_NOTE
    assert "audit_debt" in res and res["audit_debt"] is None

    open(config.telemetry_path, "wb").close()
    res = _created(m, "A write beside an empty telemetry file.", "coh-empty")
    assert res["audit_debt"] == {"uncovered": 2, "total": 2, "excluded": 0}


def test_a_raising_derivation_still_returns_created(ws, capsys) -> None:
    """The derivation raising cannot fail the write: fallback, ``None``, one warning.

    The patch targets ``mitos.sync.derive_audit_debt``, the name sync looks up at
    call time; patching ``mitos.audit_debt.derive_audit_debt`` would miss it and
    leave this row vacuous.
    """
    from mitos.sync import _COHERENCE_AUDIT_NOTE
    config, m = ws
    capsys.readouterr()
    with patch("mitos.sync.derive_audit_debt", side_effect=RuntimeError("planted")):
        res = _created(m, "A write whose audit read breaks.", "coh-raises")
    assert res["coherence_audit"] == _COHERENCE_AUDIT_NOTE
    assert res["audit_debt"] is None
    assert res["id"] in GraphStore(config.db_path, read_only=True).get_active_decision_ids()

    err = capsys.readouterr().err
    warnings = [line for line in err.splitlines() if "Audit-debt" in line]
    assert warnings == ["[Warning] Audit-debt read failed for 'coh-raises': planted"], err


def test_the_record_path_creates_no_telemetry_file(ws, capsys) -> None:
    """Reading the debt never builds telemetry, on any surface.

    ``ws`` starts with no telemetry file; a write through the manager, the CLI
    text and ``--json`` encodings and MCP must each leave it absent.
    """
    from mitos import mcp_server
    from mitos.cli import cmd_record
    config, m = ws
    assert not os.path.exists(config.telemetry_path)

    assert _created(m, "Recorded through the manager.", "coh-nt-mgr")["audit_debt"]
    cmd_record(config, axiom="Recorded through CLI text.", rejected="rej",
               slug="coh-nt-text", acknowledge_neighbors=True)
    cmd_record(config, axiom="Recorded through CLI JSON.", rejected="rej",
               slug="coh-nt-json", acknowledge_neighbors=True, as_json=True)
    with patch("mitos.mcp_server.MitosConfig", return_value=config):
        payload = json.loads(mcp_server.record_decision(
            "Recorded through MCP.", "rej", ["s"], slug="coh-nt-mcp",
            acknowledge_neighbors=True, project=config.workspace_dir))
    assert payload["audit_debt"] == {"uncovered": 4, "total": 4, "excluded": 0}
    assert not os.path.exists(config.telemetry_path)


def test_json_created_exit_keeps_b2s_text_off_stderr(ws, capsys) -> None:
    """Under `--json` the pointer speaks on stdout only — nothing of B2 on stderr.

    Scoped to B2's own text rather than to stderr as a whole: an offline created
    record legitimately writes ``[Warning] Embedding upsert deferred …`` there
    (measured), so a row spelled ``err == ""`` would red on shipped behaviour.
    """
    from mitos.cli import cmd_record
    config, _ = ws
    cmd_record(config, axiom="The JSON surface stays on stdout.", rejected="rej",
               slug="coh-json", acknowledge_neighbors=True, as_json=True)
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["coherence_audit"]
    assert payload["coherence_audit"] not in err
    assert "mitos check" not in err


def test_cli_created_receipt_names_the_audit_exactly_once_selectored(ws, capsys) -> None:
    """The recovery clause: `mitos check`, once, carrying the caller's own selector.

    Reds in BOTH directions by construction. Zero mentions is the split's easy half
    done and the recovery never composed — worse than the bare command it replaced.
    Two is the degraded notice having kept its clause.
    """
    from mitos.cli import cmd_record
    config, _ = ws
    cmd_record(config, axiom="The human surface carries the recovery.", rejected="rej",
               slug="coh-text", acknowledge_neighbors=True)
    out, err = capsys.readouterr()
    combined = out + err
    assert combined.count("mitos check") == 1, combined
    # The selector is the caller's own vocabulary, repr-rendered — never a literal
    # (a hand-built config echoes the workspace path; through main() it is the name).
    assert f"-p {config.project!r}" in err, err
    assert "Recorded decision 'coh-text'" in out


def test_the_coherence_line_reaches_a_combined_pipe_after_the_receipt(tmp_path) -> None:
    """Ordering, proved where it can break: ONE pipe, a real subprocess.

    ``capsys`` keeps the streams apart and is structurally blind to this — off a TTY
    stdout is block-buffered while stderr never is, so without the flush the pointer
    overtakes the receipt it annotates. The anchor is ``Handle:`` (the receipt's last
    stdout line) rather than "all of stdout": an offline record already writes
    ``[Warning] Embedding upsert deferred …`` to stderr from inside the write path,
    and measured, that warning lands FIRST in the combined pipe — so an assertion
    shaped "stderr follows stdout" is false before this change too.
    """
    import subprocess
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
    done = run("-p", str(workspace), "record", "A decision recorded through a pipe.",
               "--rejected", "rej", "--slug", "piped-record")

    combined = done.stdout
    assert "Recorded decision 'piped-record'" in combined, combined
    assert "mitos check" in combined, combined
    assert combined.index("mitos check") > combined.index("Handle:"), combined


# --------------------------------------------------------------------------- #
# Phase 3g1 — the standing check notice on the `created` receipt
# --------------------------------------------------------------------------- #
#
# `ws` has no telemetry (2c's gotcha D-1): every shown row seeds a record first,
# through 3g1's writer helper. The key is set before the config is built, because
# `config.env` is resolved at construction.

from test_commit_gate import _SHOWN, _seed_attempt  # noqa: E402

_KEY = "sk-dummy-never-sent"


def _keyed_ws(config: MitosConfig, monkeypatch) -> Tuple[MitosConfig, MitosSyncManager]:
    """The same workspace under a config (and manager) built with a judge key set."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", _KEY)
    keyed = MitosConfig(config.workspace_dir, project=config.project)
    return keyed, MitosSyncManager(keyed)


def _seed_shown(config: MitosConfig, case: str) -> None:
    seed = dict(_SHOWN[case])
    _seed_attempt(config, seed.pop("state"), **seed)


def _text_receipt(config: MitosConfig, capsys, axiom: str, slug: str) -> Tuple[str, str]:
    from mitos.cli import cmd_record
    capsys.readouterr()
    cmd_record(config, axiom=axiom, rejected="rej", slug=slug, acknowledge_neighbors=True)
    return capsys.readouterr()


@pytest.mark.parametrize("case", sorted(_SHOWN))
def test_each_shown_outcome_rides_all_three_receipt_encodings(ws, monkeypatch, capsys,
                                                             case) -> None:
    """Criterion 10: MCP and `--json` carry equal notices; text prints the line just
    above the coherence line.

    Equality is right here, unlike the audit-debt parity row: nothing between the
    two writes touches the attempt record, so both read the same one.
    """
    from mitos.check_notice import check_notice_line
    from mitos.cli import _coherence_audit_hint
    config, _ = ws
    _seed_shown(config, case)
    keyed, _ = _keyed_ws(config, monkeypatch)

    cli_payload, mcp_payload = _record_on_both_encodings(keyed, capsys)
    assert cli_payload["check_notice"] == mcp_payload["check_notice"]
    notice = mcp_payload["check_notice"]
    assert notice["state"] == _SHOWN[case]["state"]
    assert notice["line"] == check_notice_line(
        {k: v for k, v in notice.items() if k != "line"})

    _out, err = _text_receipt(keyed, capsys, "The text encoding of the notice.", "cn-text")
    lines = [ln for ln in err.splitlines() if ln]
    at = lines.index(notice["line"])
    assert lines[at + 1].endswith(_coherence_audit_hint(keyed))


@pytest.mark.parametrize("setup", ["no_new_findings", "keyless", "no_telemetry"])
def test_a_healthy_keyless_or_unrecorded_workspace_pays_zero_bytes(ws, monkeypatch, capsys,
                                                                  setup) -> None:
    """Criterion 11: no `check_notice` key on MCP or `--json`, and no line on text."""
    from test_commit_gate import _seed_attempt as seed
    from mitos.telemetry import ATTEMPT_COULD_NOT_COMPLETE, ATTEMPT_NO_NEW_FINDINGS
    config, _ = ws
    if setup == "no_new_findings":
        seed(config, ATTEMPT_NO_NEW_FINDINGS)
        config, _ = _keyed_ws(config, monkeypatch)
    elif setup == "keyless":
        seed(config, ATTEMPT_COULD_NOT_COMPLETE, tokens=("judgment",))
    else:
        config, _ = _keyed_ws(config, monkeypatch)

    cli_payload, mcp_payload = _record_on_both_encodings(config, capsys)
    assert "check_notice" not in cli_payload and "check_notice" not in mcp_payload
    _out, err = _text_receipt(config, capsys, "A text write with nothing to show.", "cn-quiet")
    assert "The last contradiction check" not in err
    if setup == "no_telemetry":
        # The keyed read opens read-only and never builds the file it did not find.
        assert not os.path.exists(config.telemetry_path)


def test_a_raising_notice_read_still_returns_created(ws, monkeypatch, capsys) -> None:
    """Criterion 12 (the degrade): the read raising costs the notice and nothing else.

    Patched at ``mitos.sync.read_last_attempt``, the name the step looks up.
    """
    from mitos.telemetry import ATTEMPT_COULD_NOT_COMPLETE
    config, _ = ws
    _seed_attempt(config, ATTEMPT_COULD_NOT_COMPLETE, tokens=("judgment",))
    keyed, m = _keyed_ws(config, monkeypatch)
    capsys.readouterr()
    with patch("mitos.sync.read_last_attempt", side_effect=RuntimeError("planted")):
        res = _created(m, "A write whose notice read breaks.", "cn-raises")

    assert "check_notice" not in res
    assert res["id"] in GraphStore(keyed.db_path, read_only=True).get_active_decision_ids()
    assert res["coherence_audit"] and isinstance(res["audit_debt"], dict)
    assert res["edges_created"] == [] and res["embedding"]
    err = capsys.readouterr().err
    assert [ln for ln in err.splitlines() if "Check notice" in ln] == [
        "[Warning] Check notice could not be composed: RuntimeError: planted"]


def test_a_standing_notice_leaves_one_mitos_check_on_the_text_receipt(ws, monkeypatch,
                                                                     capsys) -> None:
    """Criterion 13: the notice line adds no second recipe; the coherence line has it.

    ``argv[0]`` is pinned to bare ``mitos``: a recovery clause leaking onto this
    receipt would otherwise spell pytest's path and stay invisible to the count.
    """
    config, _ = ws
    monkeypatch.setattr(sys, "argv", ["mitos"])
    _seed_shown(config, "could_not_complete_with_pairs")
    keyed, _ = _keyed_ws(config, monkeypatch)
    out, err = _text_receipt(keyed, capsys, "The notice stands on this receipt.", "cn-once")
    assert "The last contradiction check" in err
    assert (out + err).count("mitos check") == 1, out + err


@pytest.mark.parametrize("case", sorted(_SHOWN))
def test_the_mcp_notice_names_no_command(ws, monkeypatch, case) -> None:
    """Criterion 14: the MCP line carries no backtick and no `mitos ` command."""
    from mitos import mcp_server
    config, _ = ws
    _seed_shown(config, case)
    keyed, _ = _keyed_ws(config, monkeypatch)
    with patch("mitos.mcp_server.MitosConfig", return_value=keyed):
        payload = json.loads(mcp_server.record_decision(
            "An MCP write under a standing notice.", "rej", ["s"], slug="cn-mcp",
            acknowledge_neighbors=True, project=keyed.workspace_dir))
    line = payload["check_notice"]["line"]
    assert "`" not in line and "mitos " not in line


def test_exists_and_needs_review_carry_no_notice(ws, monkeypatch, capsys) -> None:
    """Criterion 15: only a write that landed carries the notice."""
    config, _ = ws
    _seed_shown(config, "could_not_complete")
    keyed, m = _keyed_ws(config, monkeypatch)
    assert "check_notice" in _created(m, "The sync lock is held during commit.", "cn-prior")

    again = m.record_decision_entry("The sync lock is held during commit.", "rej", ["s"],
                                    slug="cn-prior", acknowledge_neighbors=True)
    assert again["status"] == "exists" and "check_notice" not in again
    with patch.object(MitosSyncManager, "_review_neighbors",
                      return_value=[{"slug": "cn-prior", "score": 0.9,
                                     "axiom": "The sync lock is held during commit."}]):
        paused = m.record_decision_entry("The sync lock is held for the commit duration.",
                                         "rej", ["s"], slug="cn-paused")
    assert paused["status"] == "needs_review" and "check_notice" not in paused

    _out, err = _text_receipt(keyed, capsys, "The sync lock is held during commit.",
                              "cn-prior")
    assert "The last contradiction check" not in err


# --------------------------------------------------------------------------- #
# B12 — an unreadable file argument on `mitos record` is a located usage refusal
# (exit 2, the flag named, `{error, code}` under --json), the twin of
# amend-commentary's `unreadable_file` — never main()'s "Fatal Unexpected Error".
# --------------------------------------------------------------------------- #

_RECORD_FILE_FLAGS = ("--axiom-file", "--rejected-file", "--context-file")


def _record_argv(workspace, flag, path, *extra):
    """A `record` call whose other sources are valid, so only `flag`'s read can fail."""
    argv = ["mitos", "-p", workspace, "record", "--slug", "s", flag, path]
    if flag != "--axiom-file":
        argv.insert(4, "An axiom.")
    if flag != "--rejected-file":
        argv += ["--rejected", "r"]
    return argv + list(extra)


def _run_record_main(monkeypatch, capsys, argv):
    from unittest.mock import patch
    from mitos import cli
    monkeypatch.setattr(sys, "argv", argv)
    capsys.readouterr()
    code = 0
    with patch("mitos.cli.cmd_record") as spy:
        try:
            cli.main()
        except SystemExit as exc:
            code = exc.code
    out, err = capsys.readouterr()
    return code, out, err, spy


def _unreadable_paths(tmp_path):
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"\xff\xfe\xfa")
    return {"missing": str(tmp_path / "missing.txt"), "undecodable": str(binary)}


@pytest.mark.parametrize("kind", ["missing", "undecodable"])
@pytest.mark.parametrize("flag", _RECORD_FILE_FLAGS)
def test_record_unreadable_file_is_a_located_refusal(workspace, tmp_path, monkeypatch,
                                                      capsys, flag, kind):
    """Row 13 — exit 2, the flag named on stderr, no crash line, no dispatch. The
    same call with a readable file reaches `cmd_record` (the transition)."""
    readable = tmp_path / "ok.txt"
    readable.write_text("readable text\n", encoding="utf-8")
    code, out, err, spy = _run_record_main(
        monkeypatch, capsys, _record_argv(workspace, flag, str(readable)))
    spy.assert_called_once()

    code, out, err, spy = _run_record_main(
        monkeypatch, capsys, _record_argv(workspace, flag, _unreadable_paths(tmp_path)[kind]))
    assert code == 2 and out == ""
    assert f"{flag} could not be read" in err and "Fatal" not in err
    spy.assert_not_called()


@pytest.mark.parametrize("flag", _RECORD_FILE_FLAGS)
def test_record_unreadable_file_under_json_is_one_object(workspace, tmp_path, monkeypatch,
                                                          capsys, flag):
    """Row 14 — stdout is exactly one `{error, code}` object, stderr empty, exit 2."""
    from mitos.cli import AMEND_CODE_UNREADABLE_FILE
    for path in _unreadable_paths(tmp_path).values():
        code, out, err, spy = _run_record_main(
            monkeypatch, capsys, _record_argv(workspace, flag, path, "--json"))
        assert code == 2 and err == ""
        payload = json.loads(out)
        assert set(payload) == {"error", "code"}
        assert payload["code"] == AMEND_CODE_UNREADABLE_FILE == "unreadable_file"
        assert payload["error"].startswith(f"{flag} could not be read: ")
        spy.assert_not_called()


# --------------------------------------------------------------------------- #
# A5 — the receipt echoes mechanisms as authored and names each fold
# --------------------------------------------------------------------------- #
#
# `mechanisms` on a `created` receipt is what decisions.md now holds (authored
# spelling and order, read from the entry the record path parsed back), and
# `mechanisms_normalized` maps each token the identity fold changed to its folded
# form — `{}` when none changed, never absent. Identity is untouched: the node id
# still hashes the folded, sorted, deduped list.

# Captured PRE-CHANGE at commit 0b38c88 (Phase 4f), before any 5a source edit, in a
# scratch keyless workspace: axiom "Receipts echo mechanisms as authored.", rejected
# "rej", scope [], mechanisms ["Zebra Timer", "os.replace"]. The slug is not in the
# hash. A literal is the point: `compute_node_id` equality alone would pass even if
# both sides moved together.
_A5_PRE_CHANGE_ID = "1a7464a402c0b3678922dc55637bbeca6c31679029ed734a8581e91a159f8325"


def _mechanisms_line(config: MitosConfig, slug: str) -> str:
    """Returns ``slug``'s mechanisms as decisions.md holds them, joined.

    Reads through ``parse_entry_stream`` — the tokenizer the record path and sync
    use, and the one that fills ``mechanisms_authored``.
    """
    from mitos.parser import parse_entry_stream
    parsed = parse_entry_stream(_read(config), "decision")
    entry = next(e for e in parsed if e.slug == slug)
    return ", ".join(entry.mechanisms_authored)


def test_a5_echo_keeps_authored_spelling_and_order(ws) -> None:
    """EC A5's live probe: the echo is the authored list exactly, not folded or sorted."""
    config, m = ws
    authored = ["Zebra Timer", "CSS transition-delay", "aria_describedby", "Alpha"]
    res = m.record_decision_entry("Receipts keep the authored order.", "rej", [],
                                  mechanisms=authored, slug="a5-order")
    assert res["status"] == "created"
    assert res["mechanisms"] == authored
    assert _mechanisms_line(config, "a5-order") == ", ".join(authored)
    assert f"**Mechanisms:** {', '.join(authored)}\n" in _read(config)


def test_a5_map_names_each_fold_class_and_omits_canonical_tokens(ws) -> None:
    """Case, `_`, `.`, and whitespace each fold; an already-canonical token is absent."""
    config, m = ws
    authored = ["Alpha", "aria_describedby", "os.replace", "127.0.0.1", "Zebra Timer",
                "sqlite"]
    res = m.record_decision_entry("Every fold class is named.", "rej", [],
                                  mechanisms=authored, slug="a5-classes")
    assert res["mechanisms_normalized"] == {
        "Alpha": "alpha",
        "aria_describedby": "aria-describedby",
        "os.replace": "os-replace",
        "127.0.0.1": "127-0-0-1",
        "Zebra Timer": "zebra-timer",
    }
    # Keys ride in authored order.
    assert list(res["mechanisms_normalized"]) == authored[:5]
    # The A5 → A6 join: folded values plus the unchanged tokens are the graph's set.
    folds = res["mechanisms_normalized"]
    joined = set(folds.values()) | {t for t in res["mechanisms"] if t not in folds}
    node = GraphStore(config.db_path).get_node(res["id"])
    assert joined == set(node["mechanisms"])


def test_a5_map_is_present_and_empty_when_nothing_folds(ws) -> None:
    """`{}` rides the receipt — with canonical tokens, and with no mechanisms at all."""
    config, m = ws
    res = m.record_decision_entry("Canonical tokens need no fold.", "rej", [],
                                  mechanisms=["sqlite", "wal-mode"], slug="a5-canon")
    assert "mechanisms_normalized" in res
    assert res["mechanisms_normalized"] == {}
    assert res["mechanisms"] == ["sqlite", "wal-mode"]

    bare = m.record_decision_entry("No mechanisms at all.", "rej", [], slug="a5-bare",
                                   acknowledge_neighbors=True)
    assert bare["status"] == "created"
    assert bare["mechanisms"] == [] and bare["mechanisms_normalized"] == {}


def test_a5_echo_is_the_split_the_gold_source_holds(ws) -> None:
    """D1: the echo comes from the parsed-back entry, never from the argument.

    One element carrying a comma is written into one ``**Mechanisms:**`` line that
    the parser splits in two; an element carrying a newline spills onto a second
    line that the parser joins with a space. The echo says what decisions.md holds.
    """
    config, m = ws
    res = m.record_decision_entry("A comma inside an element splits.", "rej", [],
                                  mechanisms=["alpha, Beta"], slug="a5-comma")
    assert res["mechanisms"] == ["alpha", "Beta"]
    assert res["mechanisms_normalized"] == {"Beta": "beta"}
    assert _mechanisms_line(config, "a5-comma") == ", ".join(res["mechanisms"])
    assert "**Mechanisms:** alpha, Beta\n" in _read(config)

    res = m.record_decision_entry("A newline inside an element joins.", "rej", [],
                                  mechanisms=["alpha\nBeta Gamma"], slug="a5-newline",
                                  acknowledge_neighbors=True)
    assert res["mechanisms"] == ["alpha Beta Gamma"]
    assert res["mechanisms_normalized"] == {"alpha Beta Gamma": "alpha-beta-gamma"}
    assert _mechanisms_line(config, "a5-newline") == "alpha Beta Gamma"


def test_a5_duplicates_echo_as_stored_and_land_as_one_entity(ws) -> None:
    """`["Foo", "foo"]` echoes both, maps only the one that changed, and is one mechanism."""
    config, m = ws
    res = m.record_decision_entry("Spellings that fold alike are one.", "rej", [],
                                  mechanisms=["Foo", "foo"], slug="a5-dup")
    assert res["mechanisms"] == ["Foo", "foo"]
    assert res["mechanisms_normalized"] == {"Foo": "foo"}
    assert GraphStore(config.db_path).get_node(res["id"])["mechanisms"] == ["foo"]

    res = m.record_decision_entry("A repeated spelling is one key.", "rej", [],
                                  mechanisms=["Foo", "Foo"], slug="a5-dup-same",
                                  acknowledge_neighbors=True)
    assert res["mechanisms"] == ["Foo", "Foo"]
    assert res["mechanisms_normalized"] == {"Foo": "foo"}


def test_a5_node_identity_is_unchanged(ws) -> None:
    """The id still hashes the folded list: it equals the pre-change capture."""
    from mitos.identity import compute_node_id, mechanism_refs_list_norm
    config, m = ws
    axiom = "Receipts echo mechanisms as authored."
    res = m.record_decision_entry(axiom, "rej", [], mechanisms=["Zebra Timer", "os.replace"],
                                  slug="a5-id-pin")
    assert res["status"] == "created"
    assert res["id"] == _A5_PRE_CHANGE_ID
    assert res["id"] == compute_node_id(
        kind="decision", axiom=axiom,
        mechanism_refs=mechanism_refs_list_norm(["Zebra Timer", "os.replace"]))


def test_a5_non_ascii_token_on_both_machine_encodings(ws, capsys) -> None:
    """`Ärger` casefolds, so it is in the map — decoded alike on MCP and `--json`."""
    from mitos import mcp_server
    from mitos.cli import cmd_record
    config, m = ws
    capsys.readouterr()
    cmd_record(config, axiom="A non-ASCII mechanism on the CLI.", rejected="rej",
               mechanisms=["Ärger"], slug="a5-umlaut-cli", as_json=True)
    cli_payload = json.loads(capsys.readouterr().out)
    with patch("mitos.mcp_server.MitosConfig", return_value=config):
        mcp_payload = json.loads(mcp_server.record_decision(
            "A non-ASCII mechanism on MCP.", "rej", [], slug="a5-umlaut-mcp",
            mechanisms=["Ärger"], acknowledge_neighbors=True,
            project=config.workspace_dir))
    for payload in (cli_payload, mcp_payload):
        assert payload["status"] == "created"
        assert payload["mechanisms"] == ["Ärger"]
        assert payload["mechanisms_normalized"] == {"Ärger": "ärger"}


def test_a5_both_encodings_single_source_the_map(ws, capsys) -> None:
    """CC-5: each payload's map rebuilds from its own `mechanisms` via the one helper."""
    from mitos import mcp_server
    from mitos.cli import cmd_record
    from mitos.sync import _mechanisms_fold_map
    config, m = ws
    authored = ["Zebra Timer", "sqlite", "os.replace"]
    capsys.readouterr()
    cmd_record(config, axiom="Parity on the CLI encoding.", rejected="rej",
               mechanisms=authored, slug="a5-parity-cli", as_json=True)
    cli_payload = json.loads(capsys.readouterr().out)
    with patch("mitos.mcp_server.MitosConfig", return_value=config):
        mcp_payload = json.loads(mcp_server.record_decision(
            "Parity on the MCP encoding.", "rej", [], slug="a5-parity-mcp",
            mechanisms=authored, acknowledge_neighbors=True,
            project=config.workspace_dir))
    for payload in (cli_payload, mcp_payload):
        assert payload["mechanisms"] == authored
        assert "mechanisms_normalized" in payload
        assert payload["mechanisms_normalized"] == _mechanisms_fold_map(payload["mechanisms"])


def test_a5_cli_text_names_each_fold_and_stays_bare_when_none(ws, capsys) -> None:
    """The text receipt shows the authored list, then a fold line only when one folded."""
    from mitos.cli import cmd_record
    config, m = ws
    capsys.readouterr()
    cmd_record(config, axiom="The text names the fold.", rejected="rej",
               mechanisms=["Zebra Timer", "os.replace", "sqlite"], slug="a5-text")
    out = capsys.readouterr().out
    assert "Mechanisms: Zebra Timer, os.replace, sqlite" in out
    fold_lines = [ln for ln in out.splitlines() if ln.lstrip().startswith("Folded:")]
    assert len(fold_lines) == 1
    assert "Zebra Timer → zebra-timer" in fold_lines[0]
    assert "os.replace → os-replace" in fold_lines[0]
    assert "sqlite" not in fold_lines[0]
    # Nothing to run, and no claim that anything was lost or merged.
    assert "mitos " not in fold_lines[0]
    for word in ("lost", "merged", "collid", "conflat", "wrong"):
        assert word not in fold_lines[0].casefold()
    # The fold line sits beside the line it modifies.
    lines = out.splitlines()
    assert lines.index(fold_lines[0]) == next(
        i for i, ln in enumerate(lines) if "Mechanisms:" in ln) + 1

    cmd_record(config, axiom="Canonical tokens print no fold line.", rejected="rej",
               mechanisms=["sqlite", "wal-mode"], slug="a5-text-canon",
               acknowledge_neighbors=True)
    out = capsys.readouterr().out
    assert "Mechanisms: sqlite, wal-mode" in out
    assert "Folded:" not in out


def test_a5_exists_replay_carries_neither_key(ws) -> None:
    """An `exists` no-op wrote nothing, so it echoes no mechanisms and names no fold."""
    config, m = ws
    first = m.record_decision_entry("A replayed decision with mechanisms.", "rej", [],
                                    mechanisms=["Zebra Timer"], slug="a5-replay")
    assert first["status"] == "created"
    again = m.record_decision_entry("A replayed decision with mechanisms.", "rej", [],
                                    mechanisms=["Zebra Timer"], slug="a5-replay",
                                    acknowledge_neighbors=True)
    assert again["status"] == "exists"
    assert "mechanisms" not in again and "mechanisms_normalized" not in again


def test_a5_description_states_the_fold_without_naming_the_map() -> None:
    """The fold is stated on the argument; the Returns block does not gloss the new key."""
    from test_description_budget import _descriptions, _flat
    description = _flat(_descriptions()["record_decision"])
    assert "Folded for identity" in description
    assert "_FOO and FOO are one" in description
    assert "committed scope, plus mechanisms as authored" in description
    assert "mechanisms_normalized" not in description


# --------------------------------------------------------------------------- #
# B4 — the target echo: each `edges_created` entry names what the write hit
# --------------------------------------------------------------------------- #

from mitos.sync import (  # noqa: E402 — section-local, like the A5 imports above
    _ECHO_AXIOM_RELATIONS, _ECHO_TOPIC_RELATIONS, _EXTRA_RELATIONS, _KILL_EDGE_FIELDS,
)
from test_modifier_surfacing import _commit_oq  # noqa: E402 — the repo's OQ committer

# The eight relations the record path wires, read from the constants that define
# them (derives_from is refused on a decision before any write or read).
_B4_RELATIONS = [n for n, _ in _EXTRA_RELATIONS if n != "derives_from"] + list(
    _KILL_EDGE_FIELDS)
# The echo's base keys, in order, and the one text key each relation adds. This is
# the test's own oracle, not the constants: moving `cites` into the axiom set, or
# dropping a relation out of it, must redden R2.
_B4_BASE_KEYS = ["kind", "target", "target_state", "target_stamps"]
_B4_TEXT_KEY = {"supersedes": "target_axiom", "corrects": "target_axiom",
                "amends": "target_axiom", "narrows": "target_axiom",
                "contradicts": "target_axiom", "resolves": "target_topic",
                "cites": None, "depends_on": None}


def _b4_seed(config, m, relation: str, slug: str) -> str:
    """Seeds the target a relation needs: an open question for `resolves`, else a
    decision. Returns the target's slug."""
    if relation == "resolves":
        _commit_oq(GraphStore(config.db_path), slug)
    else:
        assert m.record_decision_entry(f"Seed axiom {slug}.", "rej", [], slug=slug,
                                       acknowledge_neighbors=True)["status"] == "created"
    return slug


def _b4_keys(relation: str) -> list:
    text = _B4_TEXT_KEY[relation]
    return _B4_BASE_KEYS + ([text] if text else [])


def test_b4_relation_list_is_the_eight_record_path_relations() -> None:
    """The parametrisation below covers exactly the relations the oracle names."""
    assert sorted(_B4_RELATIONS) == sorted(_B4_TEXT_KEY)
    assert len(_B4_RELATIONS) == 8


@pytest.mark.parametrize("relation", _B4_RELATIONS)
def test_b4_entry_shape_is_decided_by_the_relation(ws, relation) -> None:
    """R2 + R7: each entry's keys, in order, are exactly the relation's allowed set —
    no reasoning, no target field beyond the identifying ones."""
    config, m = ws
    target = _b4_seed(config, m, relation, "b4-target")
    res = m.record_decision_entry("A B4 shape probe.", "rej", [], slug="b4-shape",
                                  **{relation: target})
    assert res["status"] == "created", res
    [entry] = res["edges_created"]
    assert list(entry) == _b4_keys(relation)
    assert entry["kind"] == relation and entry["target"] == target
    for forbidden in ("rejected_paths", "scope", "mechanisms", "score", "id",
                      "core_axiom", "context"):
        assert forbidden not in entry
    assert isinstance(entry["target_stamps"], dict)
    assert entry["target_state"] in {"active", "drifted", "superseded", "corrected",
                                     "parked", "resolved"}


def test_b4_relation_constants_partition_the_record_path_relations() -> None:
    """R2: the acting set and the topic set are named, disjoint, and leave the weak
    relations in neither — a future relation gets no text until someone decides."""
    assert _ECHO_AXIOM_RELATIONS == {"supersedes", "corrects", "amends", "narrows",
                                     "contradicts"}
    assert _ECHO_TOPIC_RELATIONS == {"resolves"}
    assert not (_ECHO_AXIOM_RELATIONS & _ECHO_TOPIC_RELATIONS)
    weak = set(_B4_RELATIONS) - _ECHO_AXIOM_RELATIONS - _ECHO_TOPIC_RELATIONS
    assert weak == {"cites", "depends_on"}


def test_b4_derives_from_is_refused_before_any_echo_read(ws) -> None:
    """R2: `derives_from` on a decision is refused before the commit, so no read runs."""
    config, m = ws
    _b4_seed(config, m, "amends", "b4-df-target")
    with patch.object(GraphStore, "get_outgoing_edge_targets") as echo, \
            patch.object(GraphStore, "get_outgoing_edges") as base:
        res = m.record_decision_entry("Derives from a decision.", "rej", [],
                                      slug="b4-df", derives_from="b4-df-target")
    assert "error" in res
    echo.assert_not_called()
    base.assert_not_called()


@pytest.mark.parametrize("relation,state,stamp", [
    ("supersedes", "superseded", "superseded_by"),
    ("corrects", "corrected", "corrected_by"),
])
def test_b4_kill_edges_read_their_effect(ws, relation, state, stamp) -> None:
    """R3: a kill edge's receipt says the target left the active view, stamped by
    this very write (Gotcha 4: the new slug is the confirmation, not filtered)."""
    config, m = ws
    _b4_seed(config, m, relation, "b4-old")
    res = m.record_decision_entry("A B4 kill probe.", "rej", [], slug="b4-new",
                                  **{relation: "b4-old"})
    assert res["status"] == "created", res
    [entry] = res["edges_created"]
    assert entry["target_state"] == state
    assert entry["target_stamps"] == {stamp: ["b4-new"]}
    assert entry["target_axiom"] == "Seed axiom b4-old."


def test_b4_comma_split_supersedes_gives_two_entries_in_rowid_order(ws) -> None:
    """R3: `supersedes="a, b"` wires two edges, echoed in insertion order."""
    config, m = ws
    for slug in ("b4-a", "b4-b"):
        _b4_seed(config, m, "supersedes", slug)
    res = m.record_decision_entry("Unify a and b.", "rej", [], slug="b4-unify",
                                  supersedes="b4-a, b4-b")
    assert res["status"] == "created", res
    assert [(e["kind"], e["target"], e["target_state"], e["target_stamps"])
            for e in res["edges_created"]] == [
        ("supersedes", "b4-a", "superseded", {"superseded_by": ["b4-unify"]}),
        ("supersedes", "b4-b", "superseded", {"superseded_by": ["b4-unify"]}),
    ]


def test_b4_trap_row_a_cited_target_reads_its_standing_amendment(ws) -> None:
    """R4: a target already amended by x, then cited, reads active + amended_by:[x] —
    state without stamps would be the trap — and a weak relation carries no axiom."""
    config, m = ws
    _b4_seed(config, m, "amends", "b4-base")
    assert m.record_decision_entry("The x amendment.", "rej", [], slug="x",
                                   amends="b4-base")["status"] == "created"
    res = m.record_decision_entry("A citing record.", "rej", [], slug="b4-citer",
                                  cites="b4-base")
    [entry] = res["edges_created"]
    assert entry == {"kind": "cites", "target": "b4-base", "target_state": "active",
                     "target_stamps": {"amended_by": ["x"]}}
    assert "target_axiom" not in entry


def test_b4_resolves_reads_resolved_and_the_stored_topic(ws) -> None:
    """R5: the write's own `resolves` is visible in its read-back (Stage-2 state, not
    the kill-edge-only `active`), and the topic is the stored Topic, not the slug."""
    config, m = ws
    _commit_oq(GraphStore(config.db_path), "q-auth")
    res = m.record_decision_entry("Rotate tokens hourly.", "rej", [], slug="b4-resolver",
                                  resolves="q-auth")
    assert res["status"] == "created", res
    [entry] = res["edges_created"]
    assert entry["target_state"] == "resolved"
    assert entry["target_topic"] == "Topic for q-auth"
    assert entry["target_topic"] != entry["target"]
    assert "target_axiom" not in entry
    assert entry["target_stamps"] == {}


def test_b4_axiom_is_whole_and_byte_equal(ws) -> None:
    """R6: a long axiom rides whole — never truncated."""
    config, m = ws
    long_axiom = ("The graph store keeps every amendment chain intact, " * 8).strip()
    assert len(long_axiom) > 300
    assert m.record_decision_entry(long_axiom, "rej", [], slug="b4-long",
                                   acknowledge_neighbors=True)["status"] == "created"
    res = m.record_decision_entry("Narrow the long one.", "rej", [], slug="b4-narrower",
                                  narrows="b4-long")
    [entry] = res["edges_created"]
    stored = GraphStore(config.db_path).get_node_by_slug("b4-long")["core_axiom"]
    assert entry["target_axiom"] == stored
    assert entry["target_axiom"] == long_axiom


def test_b4_echo_is_bound_to_the_stored_slug_not_the_argument(ws) -> None:
    """R1 (K10): a mixed-case target resolves to the stored casing, and the echo
    names what the store holds — never the argument's spelling."""
    config, m = ws
    m.record_decision_entry("Prior axiom.", "rej", [], slug="prior")
    res = m.record_decision_entry("Follow-up axiom.", "rej", [], slug="follow-up",
                                  amends="PRIOR")
    assert res["status"] == "created", res
    assert res["edges_created"] == [_PRIOR_AMENDS_ECHO]


def _b4_seed_three(config, m) -> None:
    for slug in ("b4-prior", "b4-cited"):
        _b4_seed(config, m, "amends", slug)
    _commit_oq(GraphStore(config.db_path), "q-b4")


_B4_THREE = {"amends": "b4-prior", "cites": "b4-cited", "resolves": "q-b4"}


def test_b4_echo_read_raises_once_the_receipt_survives_with_null_keys(ws, capsys) -> None:
    """R8: the echo read raising after the commit degrades the echo, never the
    receipt — `created`, the base edges with null target keys shaped by relation,
    one warning, the entry written and the node resolvable."""
    config, m = ws
    _b4_seed_three(config, m)
    twin = m.record_decision_entry("The healthy twin.", "rej", [], slug="b4-twin",
                                   **_B4_THREE)
    assert twin["status"] == "created", twin
    capsys.readouterr()
    with patch.object(GraphStore, "get_outgoing_edge_targets",
                      side_effect=RuntimeError("planted")):
        res = m.record_decision_entry("The degraded record.", "rej", [], slug="b4-degraded",
                                      **_B4_THREE)
    assert res["status"] == "created", res
    by_kind = {e["kind"]: e for e in res["edges_created"]}
    assert len(by_kind) == 3
    assert by_kind["amends"] == {"kind": "amends", "target": "b4-prior",
                                 "target_state": None, "target_stamps": None,
                                 "target_axiom": None}
    assert by_kind["cites"] == {"kind": "cites", "target": "b4-cited",
                                "target_state": None, "target_stamps": None}
    assert by_kind["resolves"] == {"kind": "resolves", "target": "q-b4",
                                   "target_state": None, "target_stamps": None,
                                   "target_topic": None}
    # The shape stays relation-decided: each entry's keys equal the healthy twin's.
    assert [list(e) for e in res["edges_created"]] == [
        list(e) for e in twin["edges_created"]]
    assert set(res) == set(twin)
    err = capsys.readouterr().err
    warnings = [ln for ln in err.splitlines() if "Edge" in ln]
    assert warnings == ["[Warning] Edge echo read failed for 'b4-degraded': planted"]
    assert "b4-degraded" in _read(config)
    assert res["id"] in GraphStore(config.db_path, read_only=True).get_active_decision_ids()


def test_b4_both_edge_reads_raise_edges_created_is_null(ws, capsys) -> None:
    """R9: when the base read fails too, the edges could not be read back — `null`,
    never a guessed list — and the write still reports `created`."""
    config, m = ws
    _b4_seed_three(config, m)
    capsys.readouterr()
    with patch.object(GraphStore, "get_outgoing_edge_targets",
                      side_effect=RuntimeError("planted")), \
            patch.object(GraphStore, "get_outgoing_edges",
                         side_effect=RuntimeError("base planted")):
        res = m.record_decision_entry("Both reads fail.", "rej", [], slug="b4-blind",
                                      **_B4_THREE)
    assert res["status"] == "created", res
    assert res["edges_created"] is None
    err = capsys.readouterr().err
    assert [ln for ln in err.splitlines() if "Edge" in ln] == [
        "[Warning] Edge echo read failed for 'b4-blind': planted",
        "[Warning] Edge read failed for 'b4-blind': base planted",
    ]
    assert res["id"] in GraphStore(config.db_path, read_only=True).get_active_decision_ids()


def test_b4_cli_null_edges_print_the_unknown_line_and_json_carries_null(ws, capsys) -> None:
    """R9 on the CLI: text prints the `unknown —` line; `--json` carries `null`."""
    from mitos.cli import cmd_record
    config, m = ws
    _b4_seed_three(config, m)
    capsys.readouterr()
    with patch.object(GraphStore, "get_outgoing_edge_targets",
                      side_effect=RuntimeError("planted")), \
            patch.object(GraphStore, "get_outgoing_edges",
                         side_effect=RuntimeError("base planted")):
        cmd_record(config, axiom="Blind text.", rejected="rej", slug="b4-blind-text",
                   amends="b4-prior")
        out = capsys.readouterr().out
        cmd_record(config, axiom="Blind json.", rejected="rej", slug="b4-blind-json",
                   amends="b4-prior", as_json=True)
        payload = json.loads(capsys.readouterr().out)
    edge_lines = [ln for ln in out.splitlines() if "Edges:" in ln]
    assert edge_lines == [
        "  Edges:     unknown — the edges this record wired could not be read back"]
    assert "edges_created" in payload and payload["edges_created"] is None
    assert payload["status"] == "created"


def test_b4_bare_record_keeps_an_empty_list_and_no_edges_line(ws, capsys) -> None:
    """R10: no relation → `[]` (read, held nothing), and the text prints no Edges line."""
    from mitos.cli import cmd_record
    config, m = ws
    capsys.readouterr()
    cmd_record(config, axiom="A bare B4 record.", rejected="rej", slug="b4-bare",
               as_json=True)
    assert json.loads(capsys.readouterr().out)["edges_created"] == []
    cmd_record(config, axiom="Another bare B4 record.", rejected="rej", slug="b4-bare-2",
               acknowledge_neighbors=True)
    out = capsys.readouterr().out
    assert "Edges:" not in out


def test_b4_cli_text_names_state_stamps_and_the_acting_axiom(ws, capsys) -> None:
    """R11: one line per edge with state and stamp words; the whole axiom (every
    line indented) under an acting edge and not under a weak one; no recipe."""
    from mitos.cli import cmd_record
    config, m = ws
    m.record_decision_entry("Prior axiom.", "rej", [], slug="prior")
    _b4_seed(config, m, "cites", "b4-cited")
    capsys.readouterr()
    cmd_record(config, axiom="Follow-up axiom.", rejected="rej", slug="follow-up",
               amends="prior", cites="b4-cited")
    out = capsys.readouterr().out
    lines = out.splitlines()
    first = next(i for i, ln in enumerate(lines) if ln.startswith("  Edges:"))
    assert lines[first].startswith("  Edges:     amends → prior")
    assert "active" in lines[first] and "amended by: follow-up" in lines[first]
    assert lines[first + 1].strip() == "axiom: Prior axiom."
    assert lines[first + 1].startswith(" " * 13)
    assert lines[first + 2].strip().startswith("cites → b4-cited")
    assert "active" in lines[first + 2] and "amended by" not in lines[first + 2]
    # The weak edge carries no axiom line: the next line, if any, is another field.
    rest = lines[first + 3:]
    assert not rest or not rest[0].strip().startswith(("axiom", "topic"))
    for ln in lines[first:first + 3]:
        assert "mitos " not in ln


def test_b4_cli_text_null_echo_says_the_state_could_not_be_read(ws, capsys) -> None:
    """R11: a null echo prints the unknown state in words, never a blank or None."""
    from mitos.cli import cmd_record
    config, m = ws
    _b4_seed_three(config, m)
    capsys.readouterr()
    with patch.object(GraphStore, "get_outgoing_edge_targets",
                      side_effect=RuntimeError("planted")):
        cmd_record(config, axiom="Null echo text.", rejected="rej", slug="b4-null-text",
                   amends="b4-prior", cites="b4-cited")
    lines = capsys.readouterr().out.splitlines()
    first = next(i for i, ln in enumerate(lines) if ln.startswith("  Edges:"))
    assert lines[first].startswith("  Edges:     amends → b4-prior")
    assert "state could not be read" in lines[first]
    assert lines[first + 1].strip() == "axiom could not be read"
    assert "cites → b4-cited" in lines[first + 2]
    assert "state could not be read" in lines[first + 2]
    for ln in lines[first:first + 3]:
        assert "None" not in ln and "mitos " not in ln


def test_b4_cli_text_indents_every_line_of_a_multiline_axiom() -> None:
    """R6/R11: a multi-line stored axiom prints whole, each line under the indent."""
    from mitos.cli import _edge_echo_lines
    lines = "\n".join(_edge_echo_lines([
        {"kind": "amends", "target": "p", "target_state": "active",
         "target_stamps": {}, "target_axiom": "First line.\nSecond line."}])).splitlines()
    assert lines == ["  Edges:     amends → p  [active]",
                     "               axiom: First line.",
                     "               Second line."]


def test_b4_exists_text_receipt_prints_no_edges_line(ws, capsys) -> None:
    """The text tail is shared with `exists`, which carries no `edges_created`: the
    unknown line is keyed on the key's presence, so a no-op prints no Edges line."""
    from mitos.cli import cmd_record
    config, m = ws
    cmd_record(config, axiom="A replayed text record.", rejected="rej", slug="b4-replay")
    capsys.readouterr()
    cmd_record(config, axiom="A replayed text record.", rejected="rej", slug="b4-replay",
               acknowledge_neighbors=True)
    out = capsys.readouterr().out
    assert "already" in out or "exists" in out
    assert "Edges:" not in out


def test_b4_state_is_computed_at_receipt_time_not_at_commit(ws) -> None:
    """Stretch: a target superseded between the commit and the echo read is reported
    as it stands at read time — the read takes no lock and reads now (M3)."""
    config, m = ws
    _b4_seed(config, m, "cites", "b4-moving")
    original = GraphStore.get_outgoing_edge_targets

    def supersede_first(self, node_id):
        other = MitosSyncManager(config).record_decision_entry(
            "Replaces the moving target.", "rej", [], slug="b4-replacer",
            supersedes="b4-moving")
        assert other["status"] == "created", other
        return original(self, node_id)

    with patch.object(GraphStore, "get_outgoing_edge_targets", autospec=True,
                      side_effect=supersede_first):
        res = m.record_decision_entry("Cites the moving target.", "rej", [],
                                      slug="b4-citer-late", cites="b4-moving")
    [entry] = res["edges_created"]
    assert entry["target_state"] == "superseded"
    assert entry["target_stamps"] == {"superseded_by": ["b4-replacer"]}


def test_b4_store_read_strips_its_helper_columns_from_the_target(ws) -> None:
    """Gotcha 10: the joined edge alias and the derived state columns never ride the
    hydrated target node."""
    config, m = ws
    _b4_seed(config, m, "amends", "b4-clean")
    res = m.record_decision_entry("Amends the clean one.", "rej", [], slug="b4-cleaner",
                                  amends="b4-clean")
    [t] = GraphStore(config.db_path).get_outgoing_edge_targets(res["id"])
    for helper in ("echo_edge_type", "killer_type", "is_resolved", "edge_type"):
        assert helper not in t["node"]
    assert t["kind"] == "amends" and t["target"] == "b4-clean"
    assert t["state"] == "active"
    assert t["node"]["amended_by"] == ["b4-cleaner"]
