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
    """Same axiom → same slug; mechanism order does not change identity (hash sorts mechanisms)."""
    config, m = ws
    r1 = m.record_decision_entry("We MUST use SQLite!!!", "pgvector", [], mechanisms=["sqlite", "wal"])
    assert r1["slug"] == "we-must-use-sqlite"
    r2 = m.record_decision_entry("We MUST use SQLite!!!", "pgvector", [], mechanisms=["wal", "sqlite"])
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
        out = json.loads(query_decisions("event-sourcing"))
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
    res = m.record_decision_entry("New.", "Old.", [], supersedes="ghost-slug")
    assert res["code"] == "supersedes_not_found"
    assert _read(config) == before
    assert len(GraphStore(config.db_path).get_all_nodes()) == 0


def test_supersedes_fuzzy_guard(ws) -> None:
    """A prefix (not exact) supersedes target → supersedes_not_found, not a wrong-node edge."""
    config, m = ws
    m.record_decision_entry("Decision foo bar.", "no", [], slug="foo-bar")
    res = m.record_decision_entry("Tries to supersede a prefix.", "no", [], supersedes="foo")
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
    res = m.record_decision_entry("Use SQLite generally.", "no", [], slug="use-sqlite")
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
        ok = json.loads(record_decision("A decision.", "A rejection.", ["s"], slug="mcp-ok"))
        assert ok["status"] == "created" and ok["slug"] == "mcp-ok"
        err = json.loads(record_decision("Another.", "", ["s"]))  # missing rejected_paths
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
    e2.supersedes = "amb"                                           # resolves to node-1 (active non-self)
    m.store.commit_parsed_entry(e2)                                # node-2 supersedes node-1; both slug 'amb'
    before = _read(config)
    res = m.record_decision_entry("New decision.", "Rejection.", [], supersedes="amb")
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
    commits (V1a warn-defers the non-kill amends edge itself, so only ``status`` is asserted).
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
