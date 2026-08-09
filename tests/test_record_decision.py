"""Test suite for the record_decision write tool (MitosSyncManager.record_decision_entry).

Covers the spec's four layers: unit (serialise/round-trip, validation, structural-token
rejection, slug determinism), integration (full read+write loop, idempotency, supersedes,
collisions, graceful degradation), MCP boundary, and adversarial (TOCTOU, commit rollback,
pathological inputs). Asserts the SPEC-correct behaviour: an exact slug collision returns
`slug_collision` (never an auto-corrects edge that would create two same-slug nodes), and
every error path leaves decisions.md byte-for-byte unchanged.
"""

import os
import json
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
def test_scope_overflow_summary_after_receipt_then_debounced(mock_provider, mock_vector, ws) -> None:
    """An over-ceiling render attaches ONE debounced `scope_overflow` summary to the result.

    Reproduces the AX complaint and pins the fix end-to-end on the shared write path
    (so both the CLI and MCP surfaces inherit it): the receipt fields are always intact,
    the size nudge is a single line pointing at `mitos status` (not the per-write wall),
    and a second record in the same workspace within the window is silent.
    """
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
    comma-split multi-target flag) and scope/mechanisms as normalised — write
    facts read back from the store, not the raw input args."""
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
    # Normalised echo of what was committed (whitespace stripped, empties dropped).
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


def test_cli_json_receipt_carries_edges_and_fields(ws, capsys) -> None:
    """`mitos record --json` emits the enriched receipt verbatim."""
    from mitos.cli import cmd_record
    config, m = ws
    m.record_decision_entry("Prior axiom.", "rej", [], slug="prior")
    cmd_record(config, axiom="Follow-up axiom.", rejected="rej", scope=["db"],
               mechanisms=["sqlite"], amends="prior", slug="follow-up", as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["edges_created"] == [{"kind": "amends", "target": "prior"}]
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
    assert res["edges_created"] == [{"kind": "amends", "target": "prior"}]
    assert res["scope"] == ["db"] and res["mechanisms"] == ["sqlite"]


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
    # INVERTED, deliberately and only now: `mitos sync` reconciles a diverged committed
    # entry as of the C′ release, so the note finally names a path that WORKS. Before
    # that it pointed at a silent no-op, and this assertion was what kept the receipt
    # honest in the meantime — the exact bug class the divergence work exists to kill,
    # held off for three releases until the capability was real.
    assert "mitos sync" in r2["no_op_reason"]
    assert "restore-source" in r2["no_op_reason"], (
        "a graph-only node has no entry to edit — the note must name the verb that "
        "re-materializes one"
    )


def test_exists_no_op_note_covers_the_graph_only_case(ws) -> None:
    """The note distinguishes "edit the entry" from "there is no entry to edit".

    Both halves are reachable, and they need different verbs: a buffer entry is
    reconciled by `mitos sync`, while a node with no `### ` block has nothing to edit
    until `mitos restore-source` re-materializes it. Sending a caller to edit an entry
    that does not exist is the failure this replaced.
    """
    config, m = ws
    m.record_decision_entry("Pin the digest length.", "Leave it to the implementer.",
                            [], slug="pin-digest-length")
    note = m.record_decision_entry("Pin the digest length.", "CORRECTED rejected text.",
                                   [], slug="pin-digest-length")["no_op_reason"]

    assert "no `### ` block" in note, "the graph-only case must be named"
    assert "restore-source" in note, "and pointed at its verb"


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
    # The note names the VERB that restores it, not merely the possibility — a
    # re-record cannot, and until `restore-source` shipped there was nothing that could.
    assert "restore-source" in res["no_op_reason"]
    assert "no `### ` block" in res["no_op_reason"]


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
)

#: Substrings forbidden anywhere in the shared field, casefolded.
_NOTE_FORBIDDEN_SUBSTRINGS = ("mitos ", "this decision", "this entry",
                              "audit this", "you should")
#: Forbidden as WORDS — a bare `"now" in text` reds on "known" (1a's casefold lesson,
#: one class over), so these match on word boundaries only.
_NOTE_FORBIDDEN_WORDS = ("run", "now")


def _coherence_note_violations(note: str):
    """Every register rule the note breaks, as a sorted list (empty == compliant)."""
    import re
    folded = note.casefold()
    found = [s for s in _NOTE_FORBIDDEN_SUBSTRINGS if s in folded]
    found += [w for w in _NOTE_FORBIDDEN_WORDS
              if re.search(rf"\b{re.escape(w)}\b", folded)]
    return sorted(found)


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

    Non-vacuity is proved in-row — every shape in ``_PLANTED_NOTE_VIOLATIONS`` is
    fed to the same checker and must be caught. Without that, a checker whose regex
    silently stopped matching would pass this row forever.
    """
    config, m = ws
    note = _created(m, "The register is enforced by a row.", "coh-register")["coherence_audit"]

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

    failed = m.record_decision_entry("An axiom pointing nowhere.", "rej", ["s"],
                                     slug="coh-dangling", supersedes="no-such-slug")
    assert "error" in failed
    assert "coherence_audit" not in failed


def test_both_machine_encodings_carry_the_identical_coherence_audit(ws, capsys) -> None:
    """`record --json` and MCP `record_decision` return the same object's field.

    Distinct slugs deliberately: the CLI call COMMITS, so an MCP call replaying the
    same axiom would return `exists` (which carries no field at all) and the row
    would compare a string against nothing.
    """
    from mitos import mcp_server
    from mitos.cli import cmd_record
    config, m = ws

    cmd_record(config, axiom="The CLI encoding of the receipt.", rejected="rej",
               slug="coh-cli", acknowledge_neighbors=True, as_json=True)
    cli_payload = json.loads(capsys.readouterr().out)

    with patch("mitos.mcp_server.MitosConfig", return_value=config):
        mcp_payload = json.loads(mcp_server.record_decision(
            "The MCP encoding of the receipt.", "rej", ["s"], slug="coh-mcp",
            acknowledge_neighbors=True, project=config.workspace_dir))

    assert cli_payload["status"] == "created" and mcp_payload["status"] == "created"
    assert cli_payload["coherence_audit"] == mcp_payload["coherence_audit"]
    # And identical because they are ONE source rendered twice, not two strings that
    # happen to agree — the single-sourcing is the parity mechanism, and a row that
    # only compared the two payloads would stay green through a hand-copied second
    # spelling on either surface.
    from mitos.sync import _COHERENCE_AUDIT_NOTE
    assert cli_payload["coherence_audit"] == _COHERENCE_AUDIT_NOTE
    # And neither machine surface carries the recovery — the command is the CLI text
    # renderer's alone, because an agent handed a shell command runs it.
    assert "mitos" not in cli_payload["coherence_audit"]
    assert "mitos" not in mcp_payload["coherence_audit"]


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
