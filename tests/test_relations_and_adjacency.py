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
from mitos.sync import MitosSyncManager, _slugify, _normalize_slug, _SLUG_MAX_LEN


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
    # V1a edge columns: source_id / target_id / edge_type (was from_id / to_id / type).
    # A killed (corrected/superseded) target leaves the active view, so resolve ids
    # via get_all_nodes rather than the active-only get_node_by_slug.
    by_slug = {n["slug"]: n["id"] for n in store.get_all_nodes()}
    fid = by_slug.get(from_slug)
    tid = by_slug.get(to_slug)
    return any(e["source_id"] == fid and e["target_id"] == tid and e["edge_type"] == etype
              for e in store.get_edges())


def _mk_entry(axiom: str, slug: str):
    """Builds a minimal committable decision ParsedEntry for same-slug-lineage setup."""
    from mitos.parser import ParsedEntry
    e = ParsedEntry("decision", slug, 0, 0)
    e.axiom = axiom
    e.rejected_paths = "setup rejection"
    return e


# --------------------------------------------------------------------------- #
# ② Slug ergonomics — explicit slugs are validated, not truncated
#
# The slug is now mandatory + explicit on record, and it is folded into the
# canonical-core identity (permanent once committed). So the write path NORMALISES an
# explicit slug (case/separators) but REJECTS an over-length one with an exact char
# count — never silently truncating it (a silent trim diverges the stored handle from
# the one the author already cited: self-inflicted citation rot). `_slugify` keeps its
# word-boundary truncation, but that is the AUTO-DERIVE path only — not the write path.
# --------------------------------------------------------------------------- #

def test_slugify_short_text_unchanged():
    """Short text is untouched — the determinism baseline."""
    assert _slugify("Use SQLite WAL mode") == "use-sqlite-wal-mode"


def test_slugify_trims_to_word_boundary_not_midword():
    """The auto-derive path trims a long slug to whole words — no `…brazilian-portug`."""
    # Long enough to exceed the cap so the boundary-trim branch actually fires.
    axiom = ("Camila the Portuguese tutor always uses the formal European variant of "
             "the language rather than the informal Brazilian Portuguese pronunciation")
    slug = _slugify(axiom)
    assert len(slug) <= _SLUG_MAX_LEN
    assert len(_normalize_slug(axiom)) > _SLUG_MAX_LEN  # the trim branch really ran
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
    slug = _slugify("x" * 150)
    assert slug == "x" * _SLUG_MAX_LEN


def test_record_long_explicit_slug_under_cap_records_verbatim(ws):
    """A long-but-under-cap explicit slug records with its handle byte-intact (no silent trim).

    This 68-char descriptive handle is exactly the AX failure case: under the old
    64-char cap it committed as `…-over-quarantine` (the trailing `-floor` silently
    dropped), diverging from the form the author cited. The raised cap + no-truncate
    contract now stores it verbatim.
    """
    config, m = ws
    slug = "steady-state-batch-oldest-first-flow-heuristic-over-quarantine-floor"  # 68 chars
    assert len(slug) <= _SLUG_MAX_LEN
    res = m.record_decision_entry("A real decision.", "rej", ["s"], slug=slug)
    assert res["status"] == "created", res
    assert res["slug"] == slug  # stored handle == cited handle, no truncation


def test_record_over_length_slug_rejected_with_exact_count(ws):
    """An explicit slug over the cap is REJECTED (not truncated), stating the exact overrun.

    Nothing is written — the buffer-first + rollback contract holds — and the message
    names the actual length, the overrun, and the limit so the author knows exactly
    how many characters to drop.
    """
    config, m = ws
    before = _read(config)
    over = "a-very-" * 20  # normalises well past the 100-char cap
    normalized = _normalize_slug(over)
    assert len(normalized) > _SLUG_MAX_LEN
    res = m.record_decision_entry("Some decision.", "rej", ["s"], slug=over)
    assert res["code"] == "slug_too_long", res
    assert str(len(normalized)) in res["error"]                       # exact length
    assert str(len(normalized) - _SLUG_MAX_LEN) in res["error"]       # exact overrun
    assert str(_SLUG_MAX_LEN) in res["error"]                         # the limit
    # Nothing written — the contract holds.
    assert _read(config) == before
    assert len(GraphStore(config.db_path).get_all_nodes()) == 0


# --------------------------------------------------------------------------- #
# ③a Typed relations through the write path
#
# V1a commits only the two kill-edges (supersedes / corrects). The other seven
# relationship fields are parsed + serialized + Phase-A target-validated, but
# WARN-DEFERRED to V1b (store._DEFERRED_EDGE_FIELDS) — no edge is committed. So
# the V1a contract these pin is: the field round-trips into the buffer (authored
# for V1b to light up) and the record succeeds, while get_edges() carries no such
# edge yet. (8a pared these from the prototype "creates an edge" assertions —
# K5/G6; the genuine V1b edge-commit is deferred, not silently coerced.)
# --------------------------------------------------------------------------- #

_EXTRA_RELATION_LABELS = {
    "amends": "Amends", "narrows": "Narrows", "depends_on": "Depends-On",
    "resolves": "Resolves", "contradicts": "Contradicts",
    "derives_from": "Derives-From", "cites": "Cites",
}


def test_corrects_creates_kill_edge(ws):
    """The second V1a kill-edge: record --corrects retires the target (8a wiring, K4)."""
    config, m = ws
    rt = m.record_decision_entry("Target axiom.", "rej", [], slug="target")
    res = m.record_decision_entry("Corrector axiom.", "rej", [], slug="corrector",
                                  corrects="target")
    assert res["status"] == "created", res
    store = GraphStore(config.db_path)
    assert _edge(store, "corrector", "target", "corrects")
    # The corrected target leaves the active view (kill-edge) — computed 'corrected'.
    assert store.get_node_state(rt["id"]) == "corrected"
    assert store.get_node_by_slug("target") is None  # gone from the active view


@pytest.mark.parametrize("kwarg,label", sorted(_EXTRA_RELATION_LABELS.items()))
def test_each_extra_relation_serializes_but_warn_defers_edge(ws, kwarg, label):
    """Each non-kill relation round-trips into the buffer but commits NO edge (V1b).

    V1a parses + serializes + Phase-A-validates these seven, then warn-defers the
    edge to V1b. The prototype asserted the edge committed; 8a pares that to the
    V1a truth — the field is authored (present for the V1b reconciler) and the
    record succeeds, while get_edges() stays empty of it.
    """
    config, m = ws
    m.record_decision_entry("Target axiom.", "rej", [], slug="target")
    res = m.record_decision_entry("Linker axiom.", "rej", [], slug="linker",
                                  **{kwarg: "target"})
    assert res["status"] == "created", res
    # The relation is serialized into decisions.md (authored for V1b).
    assert f"**{label}:** target" in _read(config)
    # ...but no edge is committed in V1a (warn-deferred).
    assert not _edge(GraphStore(config.db_path), "linker", "target", kwarg)


def test_multiple_relations_in_one_entry(ws):
    """Several non-kill relations co-author into one buffer entry (edges defer to V1b)."""
    config, m = ws
    for slug in ("dep", "cited", "amended"):
        m.record_decision_entry(f"Axiom {slug}.", "rej", [], slug=slug)
    res = m.record_decision_entry(
        "Hub decision.", "rej", [], slug="hub",
        depends_on="dep", cites="cited", amends="amended",
    )
    assert res["status"] == "created"
    buf = _read(config)
    assert "**Depends-On:** dep" in buf
    assert "**Cites:** cited" in buf
    assert "**Amends:** amended" in buf
    # All three are warn-deferred — no edges committed in V1a.
    store = GraphStore(config.db_path)
    assert not _edge(store, "hub", "dep", "depends_on")
    assert not _edge(store, "hub", "cited", "cites")
    assert not _edge(store, "hub", "amended", "amends")


def test_relation_target_not_found_buffer_unchanged(ws):
    """Unknown relation target → error, NOTHING written, buffer byte-for-byte intact.

    This is the buffer-first + rollback contract holding for the new relations.
    """
    config, m = ws
    before = _read(config)
    res = m.record_decision_entry("New.", "Old.", [], slug="linker", depends_on="ghost-slug")
    assert res["code"] == "relation_target_not_found"
    assert "depends_on" in res["error"] and "ghost-slug" in res["error"]
    assert _read(config) == before
    assert len(GraphStore(config.db_path).get_all_nodes()) == 0


def test_relation_target_fuzzy_prefix_rejected(ws):
    """A prefix (not exact) relation target is rejected, not silently wrong-linked."""
    config, m = ws
    m.record_decision_entry("Decision foo bar.", "rej", [], slug="foo-bar")
    res = m.record_decision_entry("Tries a prefix link.", "rej", [], slug="linker", amends="foo")
    assert res["code"] == "relation_target_not_found"


def test_relation_target_ambiguous_buffer_unchanged(ws):
    """A relation target matching >1 same-casefold-slug lineage node → ambiguous, no half-commit.

    The V1a ambiguity trigger is a same-slug supersession lineage (MI-13), not the
    retired fuzzy-prefix tier: node-2 supersedes node-1 while both keep slug 'amb', so
    the all-nodes resolve_slug('amb') returns 2 ids and _validate_relation_target reports
    relation_target_ambiguous. (Mirrors test_record_decision.py::test_supersedes_ambiguous
    — the relation-target twin of the same fuzzy-tier removal.)
    """
    config, m = ws
    m.store.commit_parsed_entry(_mk_entry("axiom one", "amb"))     # node-1, slug 'amb'
    e2 = _mk_entry("axiom two", "amb")
    e2.supersedes = "amb"                                          # resolves to node-1 (active non-self)
    m.store.commit_parsed_entry(e2)                               # node-2 supersedes node-1; both slug 'amb'
    before = _read(config)
    res = m.record_decision_entry("Linker.", "rej", [], slug="linker", depends_on="amb")
    assert res["code"] == "relation_target_ambiguous"
    assert _read(config) == before
    assert GraphStore(config.db_path).get_node_by_slug("linker") is None


def test_relation_does_not_change_target_state(ws):
    """A non-kill relation leaves its target active (only supersedes/corrects retire)."""
    config, m = ws
    rt = m.record_decision_entry("Target stays active.", "rej", [], slug="t")
    m.record_decision_entry("Depends on it.", "rej", [], slug="d", depends_on="t")
    # V1a single-node state derivation (8a). depends_on is warn-deferred AND non-kill,
    # so the target is unambiguously active.
    assert GraphStore(config.db_path).get_node_state(rt["id"]) == "active"


# --------------------------------------------------------------------------- #
# ③ CLI + MCP relation surfaces
# --------------------------------------------------------------------------- #

def test_cli_cmd_record_depends_on(ws):
    """cmd_record threads a relation flag through to the buffer (edge defers to V1b)."""
    config, _ = ws
    cmd_record(config, axiom="Target.", rejected="rej", slug="cli-target")
    cmd_record(config, axiom="Linker.", rejected="rej", slug="cli-linker", depends_on="cli-target")
    # The flag reaches the buffer; the edge is warn-deferred in V1a (K5/G6).
    assert "**Depends-On:** cli-target" in _read(config)
    assert not _edge(GraphStore(config.db_path), "cli-linker", "cli-target", "depends_on")


def test_cli_cmd_record_corrects_kill_edge(ws):
    """cmd_record --corrects commits the V1a corrects kill-edge end-to-end (8a, G5)."""
    config, _ = ws
    cmd_record(config, axiom="Target.", rejected="rej", slug="cli-ktarget")
    cmd_record(config, axiom="Corrector.", rejected="rej", slug="cli-kcorrector",
               corrects="cli-ktarget")
    assert _edge(GraphStore(config.db_path), "cli-kcorrector", "cli-ktarget", "corrects")


@patch("mitos.cli.cmd_record")
def test_cli_relation_flags_route(mock_record, monkeypatch):
    """The --corrects/--depends-on/--amends/--cites/etc. flags reach cmd_record."""
    monkeypatch.setattr(sys, "argv", [
        "mitos", "record", "ax", "--rejected", "r", "--slug", "the-slug",
        "--corrects", "korrekt",
        "--depends-on", "foo", "--amends", "bar", "--cites", "baz",
        "--derives-from", "qux", "--contradicts", "quux", "--narrows", "corge",
        "--resolves", "grault",
    ])
    main()
    _, kwargs = mock_record.call_args
    assert kwargs["slug"] == "the-slug"
    assert kwargs["corrects"] == "korrekt"
    assert kwargs["depends_on"] == "foo"
    assert kwargs["amends"] == "bar"
    assert kwargs["cites"] == "baz"
    assert kwargs["derives_from"] == "qux"
    assert kwargs["contradicts"] == "quux"
    assert kwargs["narrows"] == "corge"
    assert kwargs["resolves"] == "grault"


def test_mcp_record_decision_with_relation(ws):
    """The MCP record_decision tool accepts a relation arg + serializes it (edge defers to V1b)."""
    config, _ = ws
    with patch("mitos.mcp_server.MitosConfig", return_value=config):
        from mitos.mcp_server import record_decision
        json.loads(record_decision("Target.", "rej", ["s"], slug="mcp-target"))
        res = json.loads(record_decision("Linker.", "rej", ["s"], slug="mcp-linker",
                                         depends_on="mcp-target"))
    assert res["status"] == "created"
    assert "**Depends-On:** mcp-target" in _read(config)
    assert not _edge(GraphStore(config.db_path), "mcp-linker", "mcp-target", "depends_on")


def test_mcp_record_decision_corrects_kill_edge(ws):
    """The MCP record_decision tool commits the V1a corrects kill-edge (8a, G5 parity)."""
    config, _ = ws
    with patch("mitos.mcp_server.MitosConfig", return_value=config):
        from mitos.mcp_server import record_decision
        json.loads(record_decision("Target.", "rej", ["s"], slug="mcp-ktarget"))
        res = json.loads(record_decision("Corrector.", "rej", ["s"], slug="mcp-kcorrector",
                                         corrects="mcp-ktarget"))
    assert res["status"] == "created"
    assert _edge(GraphStore(config.db_path), "mcp-kcorrector", "mcp-ktarget", "corrects")


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
        def query(self, vector, limit=5):
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
