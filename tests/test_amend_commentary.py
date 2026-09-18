"""Manager rows for `MitosSyncManager.amend_commentary` (Phase 4a, W17, T6's 4a half).

Every workspace is a real `cmd_init` one, seeded through `record` (a keyless `sync`
commits nothing), with the embed provider and vector store down. Every result goes
through `_amend`, which asserts it survives a JSON round trip and names no command.
"""

import ast
import hashlib
import inspect
import json
import os
import textwrap
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest
from filelock import FileLock
from unittest.mock import MagicMock, patch

from mitos import amend
from mitos.cli import cmd_init
from mitos.config import MitosConfig
from mitos.divergence import entry_divergence
from mitos.errors import DatabaseError
from mitos.parser import parse_entry_stream
from mitos.restore import BufferFidelityError
from mitos.settledness import RECENT, select_settled_tail
from mitos.store import GraphStore, _utc_now_iso
from mitos.sync import MitosSyncManager, _ENTRIES_MARKER
from mitos.telemetry import TelemetryStore

DEAD_QDRANT_URL = "http://127.0.0.1:9"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Keyless and serviceless: the embed step defers, nothing reaches a network."""
    monkeypatch.setenv("QDRANT_URL", DEAD_QDRANT_URL)
    down = MagicMock(side_effect=Exception("backend down"))
    monkeypatch.setattr("mitos.sync.GeminiEmbeddingProvider", down)
    monkeypatch.setattr("mitos.sync.QdrantVectorStore", down)


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    config = MitosConfig(str(root))
    cmd_init(config)
    return config, MitosSyncManager(config)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _record(m: MitosSyncManager, slug: str, *, scope=("alpha",), **relations: Any) -> Dict:
    result = m.record_decision_entry(
        f"The {slug} axiom.", f"The {slug} rejected reasoning.", list(scope),
        mechanisms=[f"{slug}-mechanism"], context=f"The {slug} context.", slug=slug,
        acknowledge_neighbors=True, **relations,
    )
    assert result["status"] == "created", result
    return result


def _amend(m: MitosSyncManager, slug: str, changes: Any) -> Dict:
    result = m.amend_commentary(slug, changes)
    assert json.loads(json.dumps(result)) == result, "the result must be JSON-safe (4b/4c)"
    assert "mitos " not in json.dumps(result), "a result string names no command"
    return result


def _buffer(config: MitosConfig) -> str:
    with open(config.decisions_file, encoding="utf-8") as fh:
        return fh.read()


def _write_buffer(config: MitosConfig, text: str) -> None:
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write(text)


def _sha(config: MitosConfig) -> str:
    with open(config.decisions_file, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _entry(config: MitosConfig, slug: str):
    return next((e for e in parse_entry_stream(_buffer(config), "decision") if e.slug == slug), None)


def _block_text(config: MitosConfig, slug: str) -> str:
    entry = _entry(config, slug)
    lines = _buffer(config).splitlines(keepends=True)
    return "".join(lines[entry.line_start - 1:entry.line_end])


def _audit(config: MitosConfig) -> List[Dict]:
    if not os.path.exists(config.telemetry_path):
        return []
    return TelemetryStore(config.telemetry_path).read_commentary_audit()


def _intents(config: MitosConfig) -> List[Dict]:
    return [row for row in _audit(config) if row.get("fields_changed") is not None]


def _outcomes(config: MitosConfig) -> List[Dict]:
    return [row for row in _audit(config) if row.get("outcome")]


def _edge_count(config: MitosConfig) -> int:
    with GraphStore(config.db_path)._get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]


def _back_date(config: MitosConfig, days: int = 30) -> None:
    stamp = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with GraphStore(config.db_path)._get_connection() as conn:
        conn.execute("UPDATE nodes SET updated_at = ?", (stamp,))


def _axiom_file(config: MitosConfig, scope: str) -> Optional[str]:
    path = os.path.join(config.mitos_dir, "axioms", f"{scope}.md")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------- #
# A1 — each field edits and re-commits
# --------------------------------------------------------------------------- #

_A1 = [
    ("rejected_paths", [{"rejected_paths": "Line one.\n  line two."}], "Line one.\nline two."),
    ("invalidates_if", [{"invalidates_if": "When it changes."}], "When it changes."),
    ("invalidates_if", [{"invalidates_if": "First."}, {"invalidates_if": "Second."}], "Second."),
    ("invalidates_if", [{"invalidates_if": "First."}, {"invalidates_if": None}], None),
    ("context", [{"context": "Replaced."}], "Replaced."),
    ("context", [{"context": ""}], None),
    ("context", [{"context": None}, {"context": "Added back."}], "Added back."),
    ("scope", [{"scope": ["gamma", "alpha"]}], ["gamma", "alpha"]),
    ("slug", [{"slug": "renamed-target"}], "renamed-target"),
]


@pytest.mark.parametrize("field, steps, stored", _A1)
def test_each_field_edits_the_block_and_recommits_the_node(ws, field, steps, stored) -> None:
    """A1 — id stable, graph and block updated, neighbours byte-identical, attributed."""
    config, m = ws
    _record(m, "other")
    target_id = _record(m, "target")["id"]
    for step in steps[:-1]:
        assert _amend(m, "target", step)["status"] == "amended"

    _back_date(config)
    other_block = _block_text(config, "other")
    before_node = m.store.get_node(target_id)
    edges_before, intents_before = _edge_count(config), len(_intents(config))

    result = _amend(m, "target", steps[-1])

    assert result["status"] == "amended", result
    assert result["id"] == target_id and result["fields_changed"] == [field]
    assert result["path"] == config.decisions_file and result["embedding"] == "pending"
    node = m.store.get_node(target_id)
    if field == "scope":
        assert node["scope"] == stored
    else:
        assert (node[field] or None) == stored
    final_slug = node["slug"]
    entry = _entry(config, final_slug)
    parsed = entry.scope if field == "scope" else (getattr(entry, field) or None)
    assert parsed == stored, "the buffer block carries the edit"
    assert _block_text(config, "other") == other_block, "the neighbour is byte-identical"
    divergence = entry_divergence(entry, node, node["scope"], m.store.get_outgoing_edges(target_id))
    assert not any(divergence.values()), "no residual reconcile is waiting"
    assert node["updated_at"] > before_node["updated_at"]
    assert before_node["confirmed_by"] == "agent" and before_node["confirmed_at"]
    assert (node["confirmed_by"], node["confirmed_at"]) == (
        before_node["confirmed_by"], before_node["confirmed_at"]
    ), "the graph-primary confirmation pair is carried, never re-stamped"
    assert _edge_count(config) == edges_before, "no edge minted (MI-5)"
    new_intents = _intents(config)[intents_before:]
    assert [row["fields_changed"] for row in new_intents] == [result["fields_changed"]]
    assert new_intents[0]["node_id"] == target_id and new_intents[0]["slug"] == before_node["slug"]


def test_a_case_only_rename_is_a_change_and_is_attributed(ws) -> None:
    """A1b — the casefold seams must not swallow it; a repeat is unchanged."""
    config, m = ws
    target_id = _record(m, "target")["id"]
    _back_date(config)
    before = m.store.get_node(target_id)

    result = _amend(m, "target", {"slug": "Target"})

    assert result["status"] == "amended" and result["rename"]["to"] == "Target"
    node = m.store.get_node(target_id)
    assert node["slug"] == "Target" and node["updated_at"] > before["updated_at"]
    assert _intents(config)[-1]["fields_changed"] == ["slug"]
    assert _intents(config)[-1]["new_values"]["slug"] == "Target"
    assert _buffer(config).count("### Target\n") == 1
    assert _amend(m, "target", {"slug": "Target"})["status"] == "unchanged"


def test_a_buffered_transcript_the_graph_lacks_does_not_ride_along(ws) -> None:
    """A1c — the transcript is withheld, as the reconcile withholds it (P8).

    The divergence set excludes transcripts, so a hand-added transcript block would
    otherwise be committed by an edit that names only `context` and attributes nothing.
    """
    config, m = ws
    target_id = _record(m, "target")["id"]
    _write_buffer(config, _buffer(config).replace(
        "**Context:** The target context.\n",
        "**Context:** The target context.\n[DECISION_TRANSCRIPT]\nUser: hand-added.\n"
        "[/DECISION_TRANSCRIPT]\n",
    ))

    result = _amend(m, "target", {"context": "Repaired."})

    assert result["status"] == "amended" and result["fields_changed"] == ["context"]
    with GraphStore(config.db_path)._get_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM transcripts WHERE node_id = ?", (target_id,)
        ).fetchone()[0]
    assert count == 0
    assert "User: hand-added." in _buffer(config)


# --------------------------------------------------------------------------- #
# A2 / A3 — refusals write nothing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("changes", [{"axiom": "Another axiom."}, {"mechanisms": ["other"]}])
def test_the_canonical_core_is_refused_with_all_three_routes(ws, changes) -> None:
    """A2 — refused before the lock: splice never runs, nothing is attributed."""
    config, m = ws
    _record(m, "target")
    sha = _sha(config)
    with patch.object(m, "splice_buffer", wraps=m.splice_buffer) as spy:
        result = _amend(m, "target", changes)
    spy.assert_not_called()
    assert (result["status"], result["reason"]) == ("refused", "canonical_core")
    assert result["routes"] == {"wrong": "corrects", "outgrown": "supersedes", "partial": "amends"}
    assert result["slug"] == "target"
    assert _sha(config) == sha and _audit(config) == []


@pytest.mark.parametrize("changes, reason", [
    ({"cites": "other"}, "edges"),
    ({"source": "agent"}, "not_editable"),
    ({"colour": "blue"}, "unknown_field"),
    ({"rejected_paths": ""}, "invalid_value"),
    ({}, "no_changes"),
])
def test_other_refusals_are_in_band_and_write_nothing(ws, changes, reason) -> None:
    """A3."""
    config, m = ws
    _record(m, "target")
    sha, mtime = _sha(config), os.stat(config.decisions_file).st_mtime_ns
    result = _amend(m, "target", changes)
    assert (result["status"], result["reason"]) == ("refused", reason)
    assert _sha(config) == sha and os.stat(config.decisions_file).st_mtime_ns == mtime
    assert _audit(config) == []


# --------------------------------------------------------------------------- #
# A4 — the fence is the only raise, and it rolls back
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", [
    "c\n### phantom", "c\n**Decided:** other", "c\n**Scope:** evil",
])
def test_a_value_that_disturbs_the_buffer_raises_and_leaves_nothing(ws, value) -> None:
    """A4 — rolled back byte for byte; the node and the audit trail untouched."""
    config, m = ws
    _record(m, "other")
    target_id = _record(m, "target")["id"]
    sha, node = _sha(config), m.store.get_node(target_id)

    with pytest.raises(BufferFidelityError):
        m.amend_commentary("target", {"context": value})

    assert _sha(config) == sha
    after = m.store.get_node(target_id)
    assert (after["context"], after["scope"], after["updated_at"]) == (
        node["context"], node["scope"], node["updated_at"]
    )
    assert _audit(config) == []


# --------------------------------------------------------------------------- #
# A5 / A6 — misses
# --------------------------------------------------------------------------- #

def _archive_one(config: MitosConfig, m: MitosSyncManager) -> str:
    """Records `old-one`, settles it, then lets a real record-path rotation archive it."""
    old_id = _record(m, "old-one")["id"]
    _back_date(config)
    config.rotation_volume_threshold_entries = 1
    config.rotation_lag_days = 1
    fresh = _record(m, "fresh-one")
    assert fresh.get("rotation", {}).get("outcome") == "rotated", fresh
    assert _entry(config, "old-one") is None
    return old_id


def _draft(slug: str, axiom: str) -> str:
    return f"### {slug}\n\n**Decided:** {axiom}\n**Rejected:** Draft rejected.\n"


def _insert_draft(config: MitosConfig, slug: str, axiom: str) -> None:
    text = _buffer(config)
    _write_buffer(config, text.replace(_ENTRIES_MARKER, f"{_ENTRIES_MARKER}\n\n{_draft(slug, axiom)}", 1))


def _assert_wrote_nothing(config: MitosConfig, m: MitosSyncManager, handle: str, expected: Dict) -> None:
    sha = _sha(config)
    with patch.object(m.store, "commit_parsed_entry", wraps=m.store.commit_parsed_entry) as spy:
        result = _amend(m, handle, {"context": "A repair."})
    spy.assert_not_called()
    assert {k: result[k] for k in expected} == expected
    assert _sha(config) == sha and _intents(config) == []


def test_a_handle_nothing_bears_is_not_found(ws) -> None:
    """A5."""
    config, m = ws
    _record(m, "target")
    _assert_wrote_nothing(config, m, "no-such-entry", {"status": "not_found", "slug": "no-such-entry"})


def test_an_archived_node_is_classified_archived(ws) -> None:
    """A5 — archived by a real rotation, never a hand-built archive."""
    config, m = ws
    old_id = _archive_one(config, m)
    _assert_wrote_nothing(config, m, "old-one", {"status": "archived", "id": old_id})


def test_a_buffered_draft_is_uncommitted_even_when_it_reuses_an_archived_slug(ws) -> None:
    """A5 — the author's pending block is the buffer's truth (mutant d)."""
    config, m = ws
    # Archive first: a draft in the buffer is uncommitted, which stops the settled walk.
    _archive_one(config, m)
    _insert_draft(config, "draft-one", "A draft axiom.")
    _assert_wrote_nothing(config, m, "draft-one", {"status": "uncommitted"})
    _insert_draft(config, "old-one", "A different axiom reusing the slug.")
    _assert_wrote_nothing(config, m, "old-one", {"status": "uncommitted"})


def test_an_open_question_handle_is_refused(ws) -> None:
    """A6 — this verb's field of action is decisions.md."""
    config, m = ws
    oq = parse_entry_stream("### oq-one\n\n**Topic:** A topic.\n**Questions:** Why?\n", "open_question")[0]
    m.store.commit_parsed_entry(oq)
    result = _amend(m, "oq-one", {"context": "x"})
    assert (result["status"], result["reason"]) == ("refused", "open_question")


# --------------------------------------------------------------------------- #
# A7 / A8 — renames
# --------------------------------------------------------------------------- #

def test_a_rename_onto_an_active_slug_rolls_back_and_is_attributed_as_failed(ws) -> None:
    """A7 — MI-13."""
    config, m = ws
    _record(m, "other")
    target_id = _record(m, "target")["id"]
    sha, node = _sha(config), m.store.get_node(target_id)

    result = _amend(m, "target", {"slug": "other"})

    assert (result["code"], result["slug"]) == ("slug_collision", "target")
    assert "'other'" in result["error"]
    assert _sha(config) == sha
    after = m.store.get_node(target_id)
    assert (after["slug"], after["updated_at"]) == ("target", node["updated_at"])
    intents, outcomes = _intents(config), _outcomes(config)
    assert len(intents) == 1 and len(outcomes) == 1
    assert outcomes[0]["correlates_to"] == intents[0]["audit_id"]
    assert outcomes[0]["outcome"].startswith("failed:")


def test_a_rename_onto_a_superseded_slug_is_legal_under_active_view_uniqueness(ws) -> None:
    """Gotcha 9 — pinned as the store behaves: MI-13 is uniqueness over the active view."""
    config, m = ws
    _record(m, "old")
    _record(m, "successor", supersedes="old")
    target_id = _record(m, "target")["id"]
    result = _amend(m, "target", {"slug": "old"})
    assert result["status"] == "amended", result
    assert m.store.get_node_by_slug("old")["id"] == target_id


def test_a_rename_returns_the_incoming_citations_and_leaves_them_edge_diverged(ws) -> None:
    """A8 — the fact at rename time; the citing markdown breaks loudly by design."""
    config, m = ws
    target_id = _record(m, "target")["id"]
    _record(m, "citer", cites="target")
    _record(m, "amender", amends="target")

    result = _amend(m, "target", {"slug": "target-renamed"})

    assert result["rename"] == {
        "from": "target", "to": "target-renamed",
        "incoming": [{"kind": "cites", "source": "citer"}, {"kind": "amends", "source": "amender"}],
    }
    citer = m.store.get_node_by_slug("citer")
    divergence = entry_divergence(_entry(config, "citer"), citer, citer["scope"],
                                  m.store.get_outgoing_edges(citer["id"]))
    assert divergence["edges"] is not None
    sha = _sha(config)
    refused = _amend(m, "citer", {"context": "A repair."})
    assert (refused["status"], refused["reason"], refused["fields"]) == ("refused", "diverged", ["edges"])
    assert _sha(config) == sha
    assert m.store.get_node(target_id)["slug"] == "target-renamed"


def test_a_hand_added_citation_is_refused_rather_than_minted(ws) -> None:
    """Scout W1 — a pre-existing edge divergence must not ride along on a commentary edit."""
    config, m = ws
    _record(m, "other")
    _record(m, "target")
    edited = _buffer(config).replace(
        "**Context:** The target context.\n", "**Context:** The target context.\n**Cites:** other\n"
    )
    _write_buffer(config, edited)
    edges = _edge_count(config)

    result = _amend(m, "target", {"context": "A repair."})

    assert (result["status"], result["reason"], result["fields"]) == ("refused", "diverged", ["edges"])
    assert _edge_count(config) == edges and _buffer(config) == edited and _intents(config) == []


# --------------------------------------------------------------------------- #
# A9 / A10 / A11 — consequences
# --------------------------------------------------------------------------- #

def test_a_scope_reorder_moves_the_primary_and_keeps_the_entry_recent(ws) -> None:
    """A9 — the unfiltered render follows the first tag; the tick holds off rotation."""
    config, m = ws
    _record(m, "target", scope=("alpha", "beta"))
    rejected = "The target rejected reasoning."
    assert rejected in _axiom_file(config, "alpha") and rejected not in _axiom_file(config, "beta")
    _back_date(config)

    assert _amend(m, "target", {"scope": ["beta", "alpha"]})["status"] == "amended"

    alpha, beta = _axiom_file(config, "alpha"), _axiom_file(config, "beta")
    assert rejected in beta and rejected not in alpha
    assert "full entry: beta.md" in alpha
    selection = select_settled_tail(
        _buffer(config), graph=m.store, now=_utc_now_iso(), lag_days=14, threshold=1,
        window=20, archive_name="2026-Q3.md",
    )
    assert selection.blocks == [] and selection.stopped_at == ("target", RECENT)


def test_dropping_a_scopes_last_tag_sweeps_its_file(ws) -> None:
    """A10 — T6's 4a half: the verb's own unfiltered render removes the vacated file."""
    config, m = ws
    _record(m, "target", scope=("solo", "kept"))
    notes = os.path.join(config.mitos_dir, "axioms", "notes.md")
    with open(notes, "w", encoding="utf-8") as fh:
        fh.write("# my notes\n")
    assert _axiom_file(config, "solo") is not None

    assert _amend(m, "target", {"scope": ["kept"]})["status"] == "amended"

    assert _axiom_file(config, "solo") is None
    assert _axiom_file(config, "kept") is not None
    with open(notes, encoding="utf-8") as fh:
        assert fh.read() == "# my notes\n"


def test_an_unreachable_vector_store_leaves_the_amendment_standing(ws, capsys) -> None:
    """A11 — C2: the outbox holds the upsert; the warning is stderr."""
    config, m = ws
    target_id = _record(m, "target")["id"]
    capsys.readouterr()
    result = _amend(m, "target", {"context": "Repaired."})
    captured = capsys.readouterr()
    assert result["status"] == "amended" and result["embedding"] == "pending"
    assert any(row["node_id"] == target_id for row in m.store.get_pending_embeddings())
    assert "[Warning]" in captured.err and captured.out == ""


def test_an_embed_step_that_raises_does_not_fail_the_amendment(ws, capsys) -> None:
    """A11 — a post-commit fault is stderr only; the commit stands."""
    config, m = ws
    target_id = _record(m, "target")["id"]
    capsys.readouterr()
    with patch.object(m, "_best_effort_embed", side_effect=RuntimeError("boom")):
        result = _amend(m, "target", {"context": "Repaired."})
    captured = capsys.readouterr()
    assert result["status"] == "amended"
    assert m.store.get_node(target_id)["context"] == "Repaired."
    assert "Embedding step failed" in captured.err and captured.out == ""


# --------------------------------------------------------------------------- #
# A12–A15 — unchanged and faults
# --------------------------------------------------------------------------- #

def test_an_identical_request_is_unchanged_and_touches_nothing(ws) -> None:
    """A12."""
    config, m = ws
    target_id = _record(m, "target")["id"]
    sha, mtime = _sha(config), os.stat(config.decisions_file).st_mtime_ns
    updated = m.store.get_node(target_id)["updated_at"]
    result = _amend(m, "target", {"context": "  The target context.  ", "scope": ["ALPHA"]})
    assert result == {"status": "unchanged", "id": target_id, "slug": "target"}
    assert _sha(config) == sha and os.stat(config.decisions_file).st_mtime_ns == mtime
    assert _audit(config) == [] and m.store.get_node(target_id)["updated_at"] == updated


def test_an_unwritable_attribution_row_refuses_and_rolls_back(ws) -> None:
    """A13 — the attribution row is mandatory."""
    config, m = ws
    target_id = _record(m, "target")["id"]
    sha, node = _sha(config), m.store.get_node(target_id)
    with patch("mitos.telemetry.TelemetryStore.record_commentary_intent",
               side_effect=DatabaseError("disk gone")):
        result = _amend(m, "target", {"context": "Repaired."})
    assert result["code"] == "audit_unavailable" and "disk gone" in result["error"]
    assert _sha(config) == sha
    after = m.store.get_node(target_id)
    assert (after["context"], after["updated_at"]) == (node["context"], node["updated_at"])


def test_a_commit_failure_rolls_back_and_closes_the_intent_row(ws) -> None:
    """A14."""
    config, m = ws
    _record(m, "target")
    sha = _sha(config)
    with patch.object(m.store, "commit_parsed_entry", side_effect=DatabaseError("db gone")):
        result = _amend(m, "target", {"context": "Repaired."})
    assert result["code"] == "commit_failed" and "db gone" in result["error"]
    assert _sha(config) == sha
    assert len(_intents(config)) == 1
    assert _outcomes(config)[0]["correlates_to"] == _intents(config)[0]["audit_id"]


@pytest.mark.parametrize("seam", ["get_outgoing_edges", "get_node"])
def test_a_graph_read_fault_under_the_lock_is_an_error_dict_not_a_raise(ws, seam) -> None:
    """Fresh-eyes 4a — the fence is the only designed raise; a read fault is a fault.

    `get_outgoing_edges` fails in the transform (nothing written); `get_node` fails on
    its third call — classification makes two — the re-read inside `after_write`
    (written, then rolled back).
    """
    config, m = ws
    _record(m, "target")
    sha = _sha(config)
    real = getattr(m.store, seam)
    calls = {"n": 0}

    def _flaky(*args, **kwargs):
        calls["n"] += 1
        if seam == "get_outgoing_edges" or calls["n"] >= 3:
            raise DatabaseError("graph unreadable")
        return real(*args, **kwargs)

    with patch.object(m.store, seam, side_effect=_flaky):
        result = _amend(m, "target", {"context": "Repaired."})
    assert result["code"] == "commit_failed" and "graph unreadable" in result["error"]
    assert _sha(config) == sha and _intents(config) == []


def test_a_held_lock_times_out_and_writes_nothing(ws) -> None:
    """A15 — a second lock instance holds the path."""
    config, m = ws
    _record(m, "target")
    sha = _sha(config)
    m.lock.timeout = 0.1
    with FileLock(m.lock_path, timeout=1):
        result = _amend(m, "target", {"context": "Repaired."})
    assert result["code"] == "lock_timeout"
    assert _sha(config) == sha and _audit(config) == []


def test_a_failed_rollback_is_reported_as_a_fact(ws) -> None:
    """`splice_buffer`'s both-writes-failed MitosError reduced to a code, command-free."""
    config, m = ws
    _record(m, "target")
    real_write = __import__("mitos.atomic_file", fromlist=["write_source"]).write_source
    calls = {"n": 0}

    def _second_write_fails(path, content):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real_write(path, content)

    with patch.object(m.store, "commit_parsed_entry", side_effect=DatabaseError("db gone")), \
            patch("mitos.atomic_file.write_source", side_effect=_second_write_fails):
        result = _amend(m, "target", {"context": "Repaired."})
    assert result["code"] == "rollback_failed" and "disk full" in result["error"]


# --------------------------------------------------------------------------- #
# A16–A18 — stdout, the fold fence, history
# --------------------------------------------------------------------------- #

def test_stdout_stays_empty_on_every_exit(ws, capsys) -> None:
    """A16 — the MCP twin shares this path."""
    config, m = ws
    _record(m, "other")
    _record(m, "target")
    capsys.readouterr()
    _amend(m, "target", {"slug": "renamed"})
    with pytest.raises(BufferFidelityError):
        m.amend_commentary("renamed", {"context": "c\n### phantom"})
    _amend(m, "no-such-entry", {"context": "x"})
    _amend(m, "renamed", {"slug": "other"})
    _amend(m, "renamed", {"axiom": "x"})
    assert capsys.readouterr().out == ""


def test_the_verb_never_reaches_the_fold_rotation_or_the_archive_discriminator() -> None:
    """A17 — the fold self-deadlocks inside the lock; rotation is not this verb's."""
    forbidden = {"corpus_graph_divergence", "rotate", "rotate_selected", "_rotate_settled",
                 "source_path"}
    sources = [
        textwrap.dedent(inspect.getsource(MitosSyncManager.amend_commentary)),
        textwrap.dedent(inspect.getsource(MitosSyncManager._amend_divergence)),
        inspect.getsource(amend),
    ]
    names = set()
    for source in sources:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
    assert "splice_buffer" in names and "classify_target" in names, "the walk is not vacuous"
    assert names & forbidden == set()


def test_a_superseded_targets_commentary_is_amendable_and_its_state_holds(ws) -> None:
    """A18 — commentary on history is still commentary."""
    config, m = ws
    old_id = _record(m, "old")["id"]
    _record(m, "successor", supersedes="old")
    result = _amend(m, "old", {"context": "Why it was retired."})
    assert result["status"] == "amended" and result["id"] == old_id
    assert m.store.get_node(old_id)["context"] == "Why it was retired."
    assert m.store.get_node_state(old_id) == "superseded"


# --------------------------------------------------------------------------- #
# V12 / V13
# --------------------------------------------------------------------------- #

def test_record_refuses_a_phantom_entry_through_its_self_parse_count_guard(ws) -> None:
    """V12 (CC-21, D-4a-6) — `record_decision_entry` diverges from the splice fence.

    Its structural-token step does not scan the list arguments, so a newline-carrying
    scope tag holding a COMPLETE entry reaches the Phase A self-parse, whose count guard
    refuses it before any write. Removing that guard writes the phantom into the buffer.
    """
    config, m = ws
    sha = _sha(config)
    result = m.record_decision_entry(
        "The v12 axiom.", "The v12 rejected.",
        ["alpha\n### phantom\n\n**Decided:** Phantom.\n**Rejected:** Phantom rejected."],
        slug="v12", acknowledge_neighbors=True,
    )
    assert result.get("code") == "parse_failed", result
    assert _sha(config) == sha


def test_record_refuses_a_heading_in_a_prose_field_at_the_structural_token_step(ws) -> None:
    """V12's sibling — the prose fields are guarded one step earlier."""
    config, m = ws
    sha = _sha(config)
    result = m.record_decision_entry(
        "The v12 axiom.", "The v12 rejected.", ["alpha"], context="c\n### phantom",
        slug="v12", acknowledge_neighbors=True,
    )
    assert result.get("code") == "parse_failed" and _sha(config) == sha


def test_a_path_shaped_tag_through_the_verb_is_fenced_at_the_render(ws, capsys) -> None:
    """V13 (D-7) — 2e's lexical write fence holds through the verb; the gold source survives."""
    config, m = ws
    _record(m, "other")
    _record(m, "target")
    capsys.readouterr()

    result = _amend(m, "target", {"scope": ["../../decisions"]})

    captured = capsys.readouterr()
    assert result["status"] == "amended"
    text = _buffer(config)
    assert _ENTRIES_MARKER in text
    assert {e.slug for e in parse_entry_stream(text, "decision")} == {"other", "target"}
    assert repr("../../decisions") in captured.err and captured.out == ""
