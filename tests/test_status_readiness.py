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


def test_fresh_project_ready_without_collection(tmp_path, monkeypatch):
    _init(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, False))  # collection not created yet
    assert cli.cmd_status(str(tmp_path)) == 0  # READY despite the absent collection


def test_ready_with_existing_collection(tmp_path, monkeypatch):
    _init(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(True, True, points=3))
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
