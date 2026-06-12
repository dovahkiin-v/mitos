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
