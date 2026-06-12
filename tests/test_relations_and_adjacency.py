"""Tests for slug ergonomics (②) and typed relations + write-time adjacency (③).

Driven by loop-Claude's friction:
- Slugs truncated mid-word (`…brazilian-portug`) made a poor handle for chaining
  supersession/relations → `_slugify` now trims to a word boundary.
- Only `supersedes` was writable; the other typed edges existed end-to-end but
  weren't reachable from the agentic write path → record now serializes + validates
  all of them, exactly like supersedes (EXACT target, Phase-A, buffer-rollback safe).
- Recording was silent about neighbours → record returns a best-effort `related`
  adjacency hint (post-commit, fail-silent — never touches the write contract).

Forced fully offline (unreachable Qdrant + no keys) so the graph/contract behaviour
is deterministic; the live adjacency loop is covered in test_integration_live.py.
"""

import json
import re
import shutil
import sys
import tempfile
from typing import Iterator, Tuple

import pytest
from unittest.mock import patch

from mitos.config import MitosConfig
from mitos.cli import cmd_init, cmd_record, main
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager, _slugify, _SLUG_MAX_LEN


@pytest.fixture
def offline(monkeypatch):
    """Forces degraded graph-only mode: unreachable Qdrant, no embedding keys."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def ws(offline) -> Iterator[Tuple[MitosConfig, MitosSyncManager]]:
    """An initialised temp workspace + a manager, in offline graph-only mode."""
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    cmd_init(config)
    yield config, MitosSyncManager(config)
    shutil.rmtree(tmp, ignore_errors=True)


def _read(config: MitosConfig) -> str:
    with open(config.decisions_file, "r", encoding="utf-8") as f:
        return f.read()


def _edge(store: GraphStore, from_slug: str, to_slug: str, etype: str) -> bool:
    fid = store.get_node_by_slug(from_slug)["id"]
    tid = store.get_node_by_slug(to_slug)["id"]
    return any(e["from_id"] == fid and e["to_id"] == tid and e["type"] == etype
              for e in store.get_edges())


# --------------------------------------------------------------------------- #
# ② Slug ergonomics — word-boundary truncation
# --------------------------------------------------------------------------- #

def test_slugify_short_text_unchanged():
    """Short text is untouched — the determinism baseline."""
    assert _slugify("Use SQLite WAL mode") == "use-sqlite-wal-mode"


def test_slugify_trims_to_word_boundary_not_midword():
    """A long axiom trims to whole words — no `…brazilian-portug` fragment."""
    axiom = ("Camila the Portuguese tutor uses the European variant rather than "
             "the Brazilian Portuguese pronunciation")
    slug = _slugify(axiom)
    assert len(slug) <= _SLUG_MAX_LEN
    assert not slug.endswith("-")
    # Every piece of the slug is a whole source word — nothing sliced mid-word.
    source_words = set(re.sub(r"[^a-z0-9 ]", " ", axiom.lower()).split())
    assert all(w in source_words for w in slug.split("-")), slug


def test_slugify_deterministic_for_long_axiom():
    """Same long text → same slug (compute_hash relies on this)."""
    axiom = "A long architectural decision about caching and invalidation " * 3
    assert _slugify(axiom) == _slugify(axiom)


def test_slugify_single_long_token_hard_caps():
    """One huge token with no boundary in range falls back to the hard cap."""
    slug = _slugify("x" * 100)
    assert slug == "x" * _SLUG_MAX_LEN


def test_record_long_axiom_yields_readable_handle(ws):
    """End-to-end: a recorded long-axiom decision gets a clean, carry-able slug."""
    config, m = ws
    res = m.record_decision_entry(
        axiom=("The catalog data module owns all persona neutral gallery markers "
               "for the Brazilian Portuguese variant going forward"),
        rejected_paths="Inlining markers per persona was rejected: duplication drift.",
        scope=["catalog"],
    )
    assert res["status"] == "created"
    assert not res["slug"].endswith("-") and len(res["slug"]) <= _SLUG_MAX_LEN


# --------------------------------------------------------------------------- #
# ③a Typed relations through the write path
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kwarg,etype", [
    ("amends", "amends"),
    ("narrows", "narrows"),
    ("depends_on", "depends_on"),
    ("resolves", "resolves"),
    ("contradicts", "contradicts"),
    ("derives_from", "derives_from"),
    ("cites", "cites"),
])
def test_each_extra_relation_creates_edge(ws, kwarg, etype):
    """Every newly-exposed relation serializes + commits a correctly-typed edge."""
    config, m = ws
    m.record_decision_entry("Target axiom.", "rej", [], slug="target")
    res = m.record_decision_entry("Linker axiom.", "rej", [], slug="linker", **{kwarg: "target"})
    assert res["status"] == "created", res
    assert _edge(GraphStore(config.db_path), "linker", "target", etype)


def test_multiple_relations_in_one_entry(ws):
    """A single decision can declare several typed edges at once."""
    config, m = ws
    for slug in ("dep", "cited", "amended"):
        m.record_decision_entry(f"Axiom {slug}.", "rej", [], slug=slug)
    res = m.record_decision_entry(
        "Hub decision.", "rej", [], slug="hub",
        depends_on="dep", cites="cited", amends="amended",
    )
    assert res["status"] == "created"
    store = GraphStore(config.db_path)
    assert _edge(store, "hub", "dep", "depends_on")
    assert _edge(store, "hub", "cited", "cites")
    assert _edge(store, "hub", "amended", "amends")


def test_relation_target_not_found_buffer_unchanged(ws):
    """Unknown relation target → error, NOTHING written, buffer byte-for-byte intact.

    This is the buffer-first + rollback contract holding for the new relations.
    """
    config, m = ws
    before = _read(config)
    res = m.record_decision_entry("New.", "Old.", [], depends_on="ghost-slug")
    assert res["code"] == "relation_target_not_found"
    assert "depends_on" in res["error"] and "ghost-slug" in res["error"]
    assert _read(config) == before
    assert len(GraphStore(config.db_path).get_all_nodes()) == 0


def test_relation_target_fuzzy_prefix_rejected(ws):
    """A prefix (not exact) relation target is rejected, not silently wrong-linked."""
    config, m = ws
    m.record_decision_entry("Decision foo bar.", "rej", [], slug="foo-bar")
    res = m.record_decision_entry("Tries a prefix link.", "rej", [], amends="foo")
    assert res["code"] == "relation_target_not_found"


def test_relation_target_ambiguous_buffer_unchanged(ws):
    """A relation target matching multiple nodes → ambiguous error, no half-commit."""
    config, m = ws
    m.record_decision_entry("Axiom one.", "rej", [], slug="amb-one")
    m.record_decision_entry("Axiom two.", "rej", [], slug="amb-two")
    before = _read(config)
    res = m.record_decision_entry("Linker.", "rej", [], depends_on="amb")
    assert res["code"] == "relation_target_ambiguous"
    assert _read(config) == before
    assert GraphStore(config.db_path).get_node_by_slug("linker") is None


def test_relation_does_not_change_target_state(ws):
    """A non-supersedes relation leaves its target active (only supersedes retires)."""
    config, m = ws
    rt = m.record_decision_entry("Target stays active.", "rej", [], slug="t")
    m.record_decision_entry("Depends on it.", "rej", [], slug="d", depends_on="t")
    store = GraphStore(config.db_path)
    conn = store._get_connection()
    try:
        states = store.compute_all_states(conn)
        assert states[rt["id"]] == "active"
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# ③ CLI + MCP relation surfaces
# --------------------------------------------------------------------------- #

def test_cli_cmd_record_depends_on(ws):
    """cmd_record threads a relation flag through to a committed edge."""
    config, _ = ws
    cmd_record(config, axiom="Target.", rejected="rej", slug="cli-target")
    cmd_record(config, axiom="Linker.", rejected="rej", slug="cli-linker", depends_on="cli-target")
    assert _edge(GraphStore(config.db_path), "cli-linker", "cli-target", "depends_on")


@patch("mitos.cli.cmd_record")
def test_cli_relation_flags_route(mock_record, monkeypatch):
    """The new --depends-on/--amends/--cites/etc. flags reach cmd_record."""
    monkeypatch.setattr(sys, "argv", [
        "mitos", "record", "ax", "--rejected", "r",
        "--depends-on", "foo", "--amends", "bar", "--cites", "baz",
        "--derives-from", "qux", "--contradicts", "quux", "--narrows", "corge",
        "--resolves", "grault",
    ])
    main()
    _, kwargs = mock_record.call_args
    assert kwargs["depends_on"] == "foo"
    assert kwargs["amends"] == "bar"
    assert kwargs["cites"] == "baz"
    assert kwargs["derives_from"] == "qux"
    assert kwargs["contradicts"] == "quux"
    assert kwargs["narrows"] == "corge"
    assert kwargs["resolves"] == "grault"


def test_mcp_record_decision_with_relation(ws):
    """The MCP record_decision tool accepts a relation arg and commits the edge."""
    config, _ = ws
    with patch("mitos.mcp_server.MitosConfig", return_value=config):
        from mitos.mcp_server import record_decision
        json.loads(record_decision("Target.", "rej", ["s"], slug="mcp-target"))
        res = json.loads(record_decision("Linker.", "rej", ["s"], slug="mcp-linker",
                                         depends_on="mcp-target"))
    assert res["status"] == "created"
    assert _edge(GraphStore(config.db_path), "mcp-linker", "mcp-target", "depends_on")


# --------------------------------------------------------------------------- #
# ③c Adjacency-at-write
# --------------------------------------------------------------------------- #

def test_no_related_field_when_offline(ws):
    """Adjacency is semantic — offline (no vector) it is simply absent, not an error."""
    config, m = ws
    res = m.record_decision_entry("Solo decision.", "rej", [], slug="solo")
    assert res["status"] == "created"
    assert "related" not in res


def test_adjacent_decisions_empty_without_vector(ws):
    """The helper short-circuits to [] when there is no vector to query with."""
    _, m = ws
    assert m._adjacent_decisions(None, exclude_slug="x") == []


def test_adjacent_decisions_excludes_self_missing_and_superseded(ws):
    """Neighbour surfacing drops self, unknown slugs, and non-live decisions."""
    config, m = ws
    m.record_decision_entry("Keep me active.", "rej", [], slug="keep")
    m.record_decision_entry("Old one.", "rej", [], slug="old")
    m.record_decision_entry("New replaces old.", "rej", [], slug="new", supersedes="old")
    m.record_decision_entry("The just-recorded self.", "rej", [], slug="self-node")

    class FakeVectorStore:
        def query(self, vector, limit=5, filter_scope=None):
            return [
                {"slug": "self-node", "score": 1.0},  # itself — must be excluded
                {"slug": "keep", "score": 0.82},      # live — kept
                {"slug": "old", "score": 0.71},       # superseded — filtered
                {"slug": "missing", "score": 0.6},    # not in graph — skipped
            ]

    m.vector_store = FakeVectorStore()
    related = m._adjacent_decisions([0.1, 0.2, 0.3], exclude_slug="self-node", limit=3)
    slugs = [r["slug"] for r in related]
    assert slugs == ["keep"]
    assert related[0]["axiom"] == "Keep me active."
    assert related[0]["score"] == 0.82
