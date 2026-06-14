"""Tests for the pre-commit near-duplicate / possible-tension review (AX P4).

Loop-Claude's friction: a new decision's nearest neighbour only surfaced in the
POST-commit `related` echo — one step too late to point an amends/supersedes at it
(a re-record is a no-op, so the link could never be added). `record_decision` now
embeds the axiom BEFORE the write, and if it is >=0.85 similar to an existing decision
the author did not reference, it PAUSES (`status: needs_review`, nothing written) so the
author can re-record with the relation or `acknowledge_neighbors=True`.

The pause needs embeddings, so the suite injects a fake embed provider + vector store to
drive it deterministically offline.
"""

import json
import shutil
import tempfile
from typing import Iterator, Tuple

import pytest
from unittest.mock import patch

from mitos.config import MitosConfig
from mitos.cli import cmd_init, cmd_record
from mitos.store import GraphStore
from mitos.sync import (MitosSyncManager, _has_negation, _polarity_mismatch,
                        _NEIGHBOR_REVIEW_THRESHOLD)


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def ws(offline) -> Iterator[Tuple[MitosConfig, MitosSyncManager]]:
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    cmd_init(config)
    yield config, MitosSyncManager(config)
    shutil.rmtree(tmp, ignore_errors=True)


class _FakeEmbed:
    def get_embedding(self, text, is_query=False):
        return [0.1, 0.2, 0.3]


class _FakeVector:
    def __init__(self, matches):
        self._matches = matches

    def query(self, vector, limit=5, filter_scope=None):
        return self._matches

    def upsert(self, *a, **k):
        pass


def _arm(m, matches):
    """Wire fake embeddings/vector so the review runs deterministically."""
    m.embed_provider = _FakeEmbed()
    m.vector_store = _FakeVector(matches)


# --------------------------------------------------------------------------- #
# Polarity helpers
# --------------------------------------------------------------------------- #

def test_has_negation_detects_cues():
    assert _has_negation("It is never a per-persona field")
    assert _has_negation("Names are not interpolated")
    assert not _has_negation("It is a per-persona field")


def test_polarity_mismatch():
    assert _polarity_mismatch("X is a field", "X is never a field")
    assert not _polarity_mismatch("X is a field", "X is the field")


# --------------------------------------------------------------------------- #
# The pause (sync layer, real _review_neighbors via injected fakes)
# --------------------------------------------------------------------------- #

def test_high_similar_unreferenced_pauses_and_writes_nothing(ws):
    config, m = ws
    m.record_decision_entry("Catalog owns the per-persona gallery markers.", "rej",
                            ["catalog"], slug="catalog-owns-markers")  # offline → commits
    _arm(m, [{"slug": "catalog-owns-markers", "score": 0.9}])
    res = m.record_decision_entry("The catalog module owns per-persona gallery markers.",
                                  "rej", ["catalog"], slug="catalog-module-markers")
    assert res["status"] == "needs_review"
    assert res["code"] == "similar_decision_exists"
    assert res["neighbors"][0]["slug"] == "catalog-owns-markers"
    # Nothing written: the new node never hit the graph or the buffer.
    store = GraphStore(config.db_path)
    assert store.get_node_by_slug("catalog-module-markers") is None
    with open(config.decisions_file, encoding="utf-8") as f:
        assert "catalog-module-markers" not in f.read()


def test_acknowledge_neighbors_commits(ws):
    config, m = ws
    m.record_decision_entry("Use SQLite for the store.", "rej", ["db"], slug="use-sqlite")
    _arm(m, [{"slug": "use-sqlite", "score": 0.9}])
    res = m.record_decision_entry("Adopt SQLite as the storage engine.", "rej", ["db"],
                                  slug="adopt-sqlite", acknowledge_neighbors=True)
    assert res["status"] == "created"
    assert GraphStore(config.db_path).get_node_by_slug("adopt-sqlite") is not None


def test_declared_relation_skips_pause(ws):
    """Linking the neighbour (amends) means the author already saw it — no pause."""
    config, m = ws
    m.record_decision_entry("Use SQLite for the store.", "rej", ["db"], slug="use-sqlite")
    _arm(m, [{"slug": "use-sqlite", "score": 0.9}])
    res = m.record_decision_entry("Use SQLite with WAL mode.", "rej", ["db"],
                                  slug="use-sqlite-wal", amends="use-sqlite")
    assert res["status"] == "created"


def test_below_threshold_commits(ws):
    config, m = ws
    m.record_decision_entry("Use SQLite for the store.", "rej", ["db"], slug="use-sqlite")
    _arm(m, [{"slug": "use-sqlite", "score": 0.70}])  # loose neighbour, under the bar
    res = m.record_decision_entry("Adopt a Postgres cluster.", "rej", ["db"], slug="adopt-pg")
    assert res["status"] == "created"


def test_offline_never_pauses(ws):
    """No embeddings → the review can't run → it must never block a write."""
    config, m = ws
    m.record_decision_entry("Use SQLite for the store.", "rej", ["db"], slug="use-sqlite")
    # m left offline (no fakes armed)
    res = m.record_decision_entry("Adopt SQLite as the engine.", "rej", ["db"], slug="adopt-sqlite")
    assert res["status"] == "created"


def test_possible_tension_flagged_on_polarity_flip(ws):
    config, m = ws
    m.record_decision_entry("The marker is never a per-persona field.", "rej",
                            ["chrome"], slug="marker-not-persona")
    _arm(m, [{"slug": "marker-not-persona", "score": 0.9}])
    res = m.record_decision_entry("The marker is a per-persona field.", "rej",
                                  ["chrome"], slug="marker-is-persona")
    assert res["status"] == "needs_review"
    assert res["neighbors"][0]["possible_tension"] is True


def test_threshold_is_inclusive(ws):
    config, m = ws
    m.record_decision_entry("Use SQLite for the store.", "rej", ["db"], slug="use-sqlite")
    _arm(m, [{"slug": "use-sqlite", "score": _NEIGHBOR_REVIEW_THRESHOLD}])
    res = m.record_decision_entry("Adopt SQLite engine.", "rej", ["db"], slug="adopt-sqlite")
    assert res["status"] == "needs_review"


# --------------------------------------------------------------------------- #
# MCP + CLI surfaces (patch _review_neighbors for determinism)
# --------------------------------------------------------------------------- #

_FLAGGED = [{"slug": "existing", "axiom": "An existing decision.", "score": 0.9,
             "possible_tension": False}]


def test_mcp_record_decision_pauses(ws):
    from mitos import mcp_server
    config, _ = ws
    with patch("mitos.mcp_server.MitosConfig", return_value=config), \
         patch.object(MitosSyncManager, "_review_neighbors", return_value=_FLAGGED):
        res = json.loads(mcp_server.record_decision("A new call.", "rej", ["s"], slug="newcall"))
    assert res["status"] == "needs_review" and res["neighbors"] == _FLAGGED
    assert GraphStore(config.db_path).get_node_by_slug("newcall") is None


def test_mcp_record_decision_acknowledge_commits(ws):
    from mitos import mcp_server
    config, _ = ws
    # _review_neighbors must NOT even be consulted when acknowledged.
    with patch("mitos.mcp_server.MitosConfig", return_value=config), \
         patch.object(MitosSyncManager, "_review_neighbors", return_value=_FLAGGED):
        res = json.loads(mcp_server.record_decision("A new call.", "rej", ["s"],
                                                    slug="newcall", acknowledge_neighbors=True))
    assert res["status"] == "created"
    assert GraphStore(config.db_path).get_node_by_slug("newcall") is not None


def test_cli_record_pause_exits_nonzero(ws, capsys):
    config, _ = ws
    with patch.object(MitosSyncManager, "_review_neighbors", return_value=_FLAGGED):
        with pytest.raises(SystemExit) as exc:
            cmd_record(config, axiom="A new call.", rejected="rej", slug="newcall")
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "Paused" in err and "existing" in err and "acknowledge-neighbors" in err


def test_cli_record_acknowledge_commits(ws):
    config, _ = ws
    with patch.object(MitosSyncManager, "_review_neighbors", return_value=_FLAGGED):
        cmd_record(config, axiom="A new call.", rejected="rej", slug="newcall",
                   acknowledge_neighbors=True)
    assert GraphStore(config.db_path).get_node_by_slug("newcall") is not None
