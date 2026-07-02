"""Tests for the fresh-project status fix: an absent/empty Qdrant collection is
a normal READY state, not a blocker (cli.cmd_status)."""

import json

from mitos import cli
from mitos.config import MitosConfig


def _init(path):
    cli.cmd_init(MitosConfig(str(path)))


def _qdrant(reachable, collection_exists, points=None):
    return lambda url, coll: {
        "reachable": reachable,
        "collection_exists": collection_exists,
        "points": points,
    }


def _scroll(present_uuids):
    """Returns a no-create scroll stub reporting a fixed point-id set."""
    return lambda base_url, collection, page_size=256: set(present_uuids)


def _scroll_fails():
    """Returns a scroll stub that raises, simulating an unreachable Qdrant mid-scroll."""
    from mitos.errors import VectorStoreError

    def _raise(base_url, collection, page_size=256):
        raise VectorStoreError("Qdrant scroll connection error")

    return _raise


def _uuids(ids):
    """Maps node ids to their Qdrant point-id UUIDs."""
    from mitos.vector_store import hash_to_uuid
    return {hash_to_uuid(i) for i in ids}


def test_fresh_project_ready_without_collection(tmp_path, monkeypatch):
    _init(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, False))  # collection not created yet
    assert cli.cmd_status(str(tmp_path)) == 0  # READY despite the absent collection


def test_ready_with_existing_collection(tmp_path, monkeypatch):
    _init(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, True, points=3))
    monkeypatch.setattr(cli, "scroll_point_ids", _scroll(set()))  # no committed nodes → nothing to verify
    assert cli.cmd_status(str(tmp_path)) == 0


def test_not_ready_when_qdrant_unreachable(tmp_path, monkeypatch):
    _init(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(False, None))
    assert cli.cmd_status(str(tmp_path)) == 1


def test_not_ready_when_uninitialized(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, False))
    assert cli.cmd_status(str(tmp_path)) == 1  # no .mitos/ → NOT SET UP


def test_not_ready_when_key_missing(tmp_path, monkeypatch):
    _init(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # no key anywhere (XDG is tmp/empty)
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, True, points=1))
    monkeypatch.setattr(cli, "scroll_point_ids", _scroll(set()))
    assert cli.cmd_status(str(tmp_path)) == 1


def test_json_report_ready_and_has_mcp_field(tmp_path, monkeypatch, capsys):
    _init(tmp_path)
    capsys.readouterr()  # discard cmd_init's "Initialized..." message
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, False))
    code = cli.cmd_status(str(tmp_path), as_json=True)
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert data["ready"] is True
    assert "mcp_wired" in data["checks"]
    assert data["checks"]["mcp_wired"] is False  # no .mcp.json in a fresh init
    # A clean (within-budget) project reports an empty size-ceiling list.
    assert data["scope_overflow"] == []


def test_status_reports_scope_overflow_detail(tmp_path, monkeypatch, capsys):
    """status is the detail surface for size-ceiling overflows: per-file sizes + largest
    decisions in the text report, and a structured list in the JSON report. This is where
    the write path's one-line nudge sends the author for the actionable breakdown."""
    import mitos.renderer as R
    from mitos.store import GraphStore
    from mitos.parser import ParsedEntry

    _init(tmp_path)
    capsys.readouterr()  # discard cmd_init's message
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, True, points=1))
    monkeypatch.setattr(cli, "scroll_point_ids", _scroll(set()))  # completeness is not this test's concern
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", 200)  # cross a small ceiling cheaply

    config = MitosConfig(str(tmp_path))
    store = GraphStore(config.db_path)
    big = ParsedEntry("decision", "big-axiom", 1, 5)
    # V1a: commit_parsed_entry reads `parsed.axiom` (not the prototype `core_axiom`);
    # the renderer hydrates it back to the reader key `core_axiom` (5d _hydrate_node),
    # so the overflow surface still works end-to-end.
    big.axiom = "A long rationale block. " * 40
    big.rejected_paths = "n/a"
    big.scope = ["substrate"]
    store.commit_parsed_entry(big)

    # Text report names the over-ceiling file, the largest decision, and a token estimate.
    assert cli.cmd_status(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "over the size ceiling" in out
    assert "substrate.md" in out
    assert "big-axiom" in out
    assert "tokens" in out

    # JSON report carries the structured scope_overflow list with ranked top decisions.
    assert cli.cmd_status(str(tmp_path), as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    over = [o for o in data["scope_overflow"] if o["name"] == "substrate.md"]
    assert len(over) == 1
    assert over[0]["threshold_chars"] == 200
    assert over[0]["top_decisions"][0]["slug"] == "big-axiom"


def _commit_n(tmp_path, n):
    """Commits n active decisions; returns their node ids in order."""
    from mitos.store import GraphStore
    from mitos.parser import ParsedEntry
    store = GraphStore(MitosConfig(str(tmp_path)).db_path)
    ids = []
    for i in range(n):
        e = ParsedEntry("decision", f"node-{i:02d}", 1, 5)
        e.axiom = f"Axiom {i}"
        e.rejected_paths = "n/a"
        ids.append(store.commit_parsed_entry(e).node_id)
    return ids


def _commit_active_and_superseded(tmp_path):
    """Commits 2 active decisions + 1 superseded; returns (keep_id, old_id, new_id)."""
    from mitos.store import GraphStore
    from mitos.parser import ParsedEntry
    store = GraphStore(MitosConfig(str(tmp_path)).db_path)

    def _d(slug, axiom, supersedes=None):
        e = ParsedEntry("decision", slug, 1, 5)
        e.axiom = axiom
        e.rejected_paths = "n/a"
        if supersedes:
            e.supersedes = supersedes
        return store.commit_parsed_entry(e).node_id

    keep_id = _d("keep-me", "Live axiom.")
    old_id = _d("old-one", "Doomed axiom.")
    new_id = _d("new-one", "Replacement axiom.", supersedes=["old-one"])
    return keep_id, old_id, new_id


def test_status_warns_by_id_diff_even_when_count_looks_healthy(tmp_path, monkeypatch, capsys):
    """The incident shape: the point COUNT clears the active count (graveyard slack), yet
    an active node has no vector. The id-diff catches it where `points >= active` couldn't."""
    _init(tmp_path)
    capsys.readouterr()
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    ids = _commit_n(tmp_path, 3)
    # Qdrant holds 2 of the 3 active nodes + 2 graveyard points → 4 points >= 3 active,
    # so the old count proxy would read HEALTHY, but node-02 is genuinely missing.
    present = _uuids(ids[:2]) | {"graveyard-a", "graveyard-b"}
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, True, points=len(present)))
    monkeypatch.setattr(cli, "scroll_point_ids", _scroll(present))

    assert cli.cmd_status(str(tmp_path)) == 0  # informational — NOT a readiness blocker
    out = capsys.readouterr().out
    assert "vector index incomplete" in out
    assert "1 active node(s) have no vector" in out
    assert "node-02" in out  # names the missing slug at small N
    assert "mitos reconcile" in out


def test_status_silent_when_all_active_present(tmp_path, monkeypatch, capsys):
    """No warning when every active node has a vector — even with graveyard points around."""
    _init(tmp_path)
    capsys.readouterr()
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    ids = _commit_n(tmp_path, 3)
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, True, points=3))
    monkeypatch.setattr(cli, "scroll_point_ids", _scroll(_uuids(ids)))

    assert cli.cmd_status(str(tmp_path)) == 0
    assert "vector index incomplete" not in capsys.readouterr().out


def test_status_active_only_superseded_absence_is_not_missing(tmp_path, monkeypatch, capsys):
    """A superseded node absent from Qdrant is NOT a shortfall — the id-diff targets the
    active surface only (proving the check honors the active-only invariant, not all-nodes)."""
    _init(tmp_path)
    capsys.readouterr()
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    keep_id, old_id, new_id = _commit_active_and_superseded(tmp_path)  # 3 nodes, 2 active
    # Qdrant holds only the two ACTIVE nodes; the superseded old-one has no vector.
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, True, points=2))
    monkeypatch.setattr(cli, "scroll_point_ids", _scroll(_uuids([keep_id, new_id])))

    assert cli.cmd_status(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "graph holds 3 node(s)" in out  # all-nodes count still reported informationally
    assert "vector index incomplete" not in out  # both active nodes present → healthy


def test_status_orphan_points_reported_as_info_not_warning(tmp_path, monkeypatch, capsys):
    """Graveyard points (vector present, node inactive) are reported neutrally, never warned —
    they are the all-superseded blackout vector's substrate."""
    _init(tmp_path)
    capsys.readouterr()
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    ids = _commit_n(tmp_path, 2)
    present = _uuids(ids) | {"graveyard-x", "graveyard-y", "graveyard-z"}  # 3 orphans
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, True, points=len(present)))
    monkeypatch.setattr(cli, "scroll_point_ids", _scroll(present))

    assert cli.cmd_status(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "vector index incomplete" not in out  # every active node has a vector
    assert "3 graveyard point(s)" in out
    assert "not an error" in out  # reported neutrally, not as a warning


def test_status_full_collection_wipe_warns_when_graph_populated(tmp_path, monkeypatch, capsys):
    """Collection deleted OUTRIGHT (not just points) with a populated graph → the whole
    active surface is missing. Absence of the collection must not read as calm health."""
    _init(tmp_path)
    capsys.readouterr()
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    _commit_n(tmp_path, 3)
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, False))  # collection absent
    # The collection-absent path must NOT scroll; if it does, this stub raises and
    # the test's "N active node(s) have no vector" assertion fails loudly.
    monkeypatch.setattr(cli, "scroll_point_ids", _scroll_fails())

    assert cli.cmd_status(str(tmp_path)) == 0  # informational, not a readiness blocker
    out = capsys.readouterr().out
    assert "vector index incomplete" in out
    assert "3 active node(s) have no vector" in out
    assert "mitos reconcile" in out
    assert "auto-created on first record" not in out  # no longer misleadingly calm
    assert "missing — 3 active node(s) have no vectors" in out  # accurate check-line hint


def test_status_absent_collection_fresh_project_stays_quiet(tmp_path, monkeypatch, capsys):
    """An absent collection with NO active nodes is a healthy fresh project — no warning,
    no scroll, and the calm 'auto-created on first record' hint is correct here."""
    _init(tmp_path)
    capsys.readouterr()
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, False))  # absent, empty graph
    monkeypatch.setattr(cli, "scroll_point_ids", _scroll_fails())  # must not be called

    assert cli.cmd_status(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "vector index incomplete" not in out
    assert "could not verify" not in out
    assert "auto-created on first record" in out


def test_status_degrades_when_scroll_fails(tmp_path, monkeypatch, capsys):
    """A scroll failure surfaces 'could not verify' — never a silent fallback to the count."""
    _init(tmp_path)
    capsys.readouterr()
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    _commit_n(tmp_path, 3)
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, True, points=99))  # count would look fine
    monkeypatch.setattr(cli, "scroll_point_ids", _scroll_fails())

    assert cli.cmd_status(str(tmp_path)) == 0  # still not a readiness blocker
    out = capsys.readouterr().out
    assert "could not verify vector completeness" in out
    assert "vector index incomplete" not in out  # no fabricated verdict


def test_status_json_carries_missing_and_orphans(tmp_path, monkeypatch, capsys):
    """The JSON surface carries the real missing-active count/slugs and orphan count."""
    _init(tmp_path)
    capsys.readouterr()
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    ids = _commit_n(tmp_path, 3)
    present = _uuids(ids[:2]) | {"graveyard-a"}  # node-02 missing, 1 orphan
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, True, points=len(present)))
    monkeypatch.setattr(cli, "scroll_point_ids", _scroll(present))

    assert cli.cmd_status(str(tmp_path), as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    checks = data["checks"]
    assert checks["missing_active_vectors"] == 1
    assert checks["missing_active_slugs"] == ["node-02"]
    assert checks["orphan_points"] == 1
    assert checks["active_nodes"] == 3


def test_status_json_null_missing_when_scroll_fails(tmp_path, monkeypatch, capsys):
    """Scroll failure → JSON reports null (unknown), never 0 (which would read 'complete')."""
    _init(tmp_path)
    capsys.readouterr()
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    _commit_n(tmp_path, 2)
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, True, points=2))
    monkeypatch.setattr(cli, "scroll_point_ids", _scroll_fails())

    assert cli.cmd_status(str(tmp_path), as_json=True) == 0
    checks = json.loads(capsys.readouterr().out)["checks"]
    assert checks["missing_active_vectors"] is None
    assert checks["missing_active_slugs"] is None
    assert checks["orphan_points"] is None
