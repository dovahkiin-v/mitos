"""Tests for the ``source_reencounter`` signal-write path (Phase 6a, V1b).

V1b lights up the substrate's FIRST real signal writer. When an already-stored
node is re-encountered from a DIFFERENT provenance (``source``), one
``source_reencounter`` audit row is emitted carrying the NEW source, exactly once
per ``(node, source)`` (MI-4 / V1-D14). The write fires at the four node-exists
short-circuit gates that skip ``commit_parsed_entry`` (sync, import, and both
``record`` exists-paths) — never inside the commit, which is unreachable on an
existing node. ``source`` is out-of-core, so the canonical core is identical on a
re-encounter; the signal-eval is the one pass the ``if existing:`` skip must NOT
swallow (§6.2 Lesson 13).

Signals have no v0.1 read accessor, so rows are asserted via raw SQL on the store's
own connection (the established idiom — ``test_store.py:_insert_drifted_signal``,
``test_migrations.py:578``). Deterministic + keyless: seed the prior node via
``commit_parsed_entry`` (raw axiom → stable id; NOT a first ``perform_sync``, whose
enrichment stub mints a unique refined-axiom id per call — Scout gotcha #3), then
drive each gate with an identical-canonical-core entry whose ``source`` differs.
"""

import os
import sqlite3
import shutil
import tempfile
import json
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from mitos.config import MitosConfig
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager
from mitos.importer import MitosProseImporter
from mitos.parser import ParsedEntry


# --- Builders + raw-SQL signal/node readers -----------------------------------


def _decision(
    slug: str = "d-slug",
    axiom: str = "An axiom.",
    rejected: str = "An alternative.",
    mechanisms: Optional[List[str]] = None,
    source: Optional[str] = None,
) -> ParsedEntry:
    """Builds a hand-made decision ``ParsedEntry`` on the V1a (``axiom``) surface.

    ``source`` is the provenance the seeded node is stored under (first-seen-wins,
    MI-4-fenced); the canonical core is ``{kind, axiom, mechanisms}`` only.
    """
    e = ParsedEntry("decision", slug, 1, 5)
    e.axiom = axiom
    e.rejected_paths = rejected
    e.mechanisms = list(mechanisms) if mechanisms else []
    e.scope = []
    e.source = source
    return e


def _reencounter_rows(store: GraphStore, node_id: Optional[str] = None) -> List[Dict]:
    """Reads ``source_reencounter`` signal rows via raw SQL, sorted by source."""
    conn = store._get_connection()
    try:
        if node_id is None:
            cur = conn.execute(
                "SELECT node_id, signal_type, source, created_at FROM signals "
                "WHERE signal_type = 'source_reencounter' ORDER BY source"
            )
        else:
            cur = conn.execute(
                "SELECT node_id, signal_type, source, created_at FROM signals "
                "WHERE signal_type = 'source_reencounter' AND node_id = ? "
                "ORDER BY source",
                (node_id,),
            )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _node_source(store: GraphStore, node_id: str) -> Optional[str]:
    """Reads a node's stored ``source`` via raw SQL, or None."""
    conn = store._get_connection()
    try:
        row = conn.execute(
            "SELECT source FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return row["source"] if row else None
    finally:
        conn.close()


@pytest.fixture
def store() -> GraphStore:
    """A temporary file GraphStore (boots the live V1b schema, user_version 2)."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    s = GraphStore(path)
    yield s
    if os.path.exists(path):
        os.remove(path)


# ============================================================================ #
# 1. The store policy primitive (the V1-D14 unit core + the single P10 point)
# ============================================================================ #


def test_note_source_reencounter_cross_source_writes_one_row(store: GraphStore) -> None:
    """A differing new source emits exactly one row carrying the NEW source; the
    stored node source is untouched (MI-4 fence); returns True."""
    nid = store.commit_parsed_entry(_decision(slug="p", axiom="Core.", source="user")).node_id
    assert store.note_source_reencounter(nid, "user", "import_llm") is True
    rows = _reencounter_rows(store, nid)
    assert len(rows) == 1
    assert rows[0]["source"] == "import_llm"
    assert rows[0]["created_at"]  # MI-10 app-supplied stamp present
    assert _node_source(store, nid) == "user"  # the node keeps its first source


def test_note_source_reencounter_same_source_is_noop(store: GraphStore) -> None:
    """An unchanged source writes nothing and returns False."""
    nid = store.commit_parsed_entry(_decision(slug="p", axiom="Core.", source="user")).node_id
    assert store.note_source_reencounter(nid, "user", "user") is False
    assert _reencounter_rows(store, nid) == []


def test_note_source_reencounter_pk_idempotent_then_distinct_mints_second(store: GraphStore) -> None:
    """A repeat from the same new source is a clean INSERT OR IGNORE no-op (one row);
    a third, DISTINCT source mints a second row (the per-(node, source) audit)."""
    nid = store.commit_parsed_entry(_decision(slug="p", axiom="Core.", source="user")).node_id
    assert store.note_source_reencounter(nid, "user", "import_llm") is True
    # Second emit, same new source → PK no-op, still one row (True = write attempted).
    assert store.note_source_reencounter(nid, "user", "import_llm") is True
    assert len(_reencounter_rows(store, nid)) == 1
    # A third, distinct new source → a second audit row.
    assert store.note_source_reencounter(nid, "user", "capture_llm") is True
    rows = _reencounter_rows(store, nid)
    assert len(rows) == 2
    assert {r["source"] for r in rows} == {"import_llm", "capture_llm"}


def test_note_source_reencounter_lands_in_live_schema_not_dead_prototype(store: GraphStore) -> None:
    """The row lands in the live ``migrations.py`` signals shape (``signal_type`` /
    ``source``, composite PK), NOT the dead ``store._init_db`` prototype (``type`` /
    ``actor``). A repeat from the same new source does not raise (PK honoured)."""
    nid = store.commit_parsed_entry(_decision(slug="s", axiom="Core.", source="user")).node_id
    store.note_source_reencounter(nid, "user", "import_llm")
    store.note_source_reencounter(nid, "user", "import_llm")  # must not raise
    conn = store._get_connection()
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(signals)").fetchall()}
        assert {"node_id", "signal_type", "source", "created_at"} <= cols
        assert "type" not in cols and "actor" not in cols  # dead-prototype columns
        row = conn.execute(
            "SELECT signal_type, source FROM signals WHERE node_id = ?", (nid,)
        ).fetchone()
        assert row["signal_type"] == "source_reencounter"
        assert row["source"] == "import_llm"
    finally:
        conn.close()


# ============================================================================ #
# 2. Sync gate (gate 1, perform_sync main loop) — DoD #1 primary surface
# ============================================================================ #

_DEC_HEADER = (
    "# Decisions\n"
    "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->\n"
)


@pytest.fixture
def sync_env() -> Tuple[MitosConfig, MitosSyncManager, str]:
    """A complete temp sync environment (mirrors ``test_sync.py``)."""
    tmpdir = tempfile.mkdtemp()
    config = MitosConfig(tmpdir)
    config.db_path = os.path.join(tmpdir, ".mitos", "graph.sqlite")
    config.decisions_file = os.path.join(tmpdir, "decisions.md")
    config.archive_dir = os.path.join(tmpdir, "decisions", "archive")
    os.makedirs(os.path.join(tmpdir, ".mitos"), exist_ok=True)
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write(_DEC_HEADER)
    manager = MitosSyncManager(config)
    yield config, manager, tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


def _set_enrichment_passthrough(mock_client: MagicMock) -> None:
    """Wires the mocked Gemini client to a unique refined axiom per call (mirrors
    ``test_sync.py``). On a re-encounter the gate ``continue``s BEFORE enrichment,
    so this never fires for the re-synced entry — it just satisfies the client gate."""
    counter = {"n": 0}

    def _gen(*args: object, **kwargs: object) -> MagicMock:
        counter["n"] += 1
        resp = MagicMock()
        resp.text = json.dumps({
            "refined_core_axiom": f"Refined axiom number {counter['n']}.",
            "refined_mechanisms": [],
            "refined_scope": ["core"],
            "suggested_relationships": {},
        })
        return resp

    mock_client.return_value.models.generate_content.side_effect = _gen


def _append(config: MitosConfig, text: str) -> None:
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def _reenc_entry(source_line: Optional[str]) -> str:
    """An identical-canonical-core decision buffer entry; ``source_line`` sets the
    re-encountering ``**Source:**`` (None = omit it → parser leaves source unset)."""
    src = f"**Source:** {source_line}\n" if source_line else ""
    return (
        "## 2026-05-19 — reenc — Re-encounter\n"
        "**Decided:** A re-encountered axiom.\n"
        "**Rejected:** An alternative.\n"
        f"{src}"
    )


def _seed_reenc(manager: MitosSyncManager, source: str) -> str:
    """Seeds the prior node via commit (raw axiom → stable id). Returns node_id."""
    return manager.store.commit_parsed_entry(
        _decision(slug="reenc", axiom="A re-encountered axiom.", source=source)
    ).node_id


@patch("google.genai.Client")
def test_sync_cross_source_reencounter_emits_one_signal(
    mock_client: MagicMock,
    sync_env: Tuple[MitosConfig, MitosSyncManager, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DoD #1 primary: re-sync the identical entry from a DIFFERENT source → exactly
    one ``source_reencounter`` row carrying the new source; node source stays first-seen."""
    config, manager, _ = sync_env
    _set_enrichment_passthrough(mock_client)
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")

    nid = _seed_reenc(manager, source="user")
    _append(config, _reenc_entry("import_llm"))
    manager.perform_sync(auto_accept=True)

    rows = _reencounter_rows(manager.store, nid)
    assert len(rows) == 1
    assert rows[0]["source"] == "import_llm"
    assert _node_source(manager.store, nid) == "user"  # MI-4 fence


@patch("google.genai.Client")
def test_sync_same_source_reencounter_is_clean_noop(
    mock_client: MagicMock,
    sync_env: Tuple[MitosConfig, MitosSyncManager, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DoD #1 second half: re-sync the identical entry with the SAME source (omitted
    ``**Source:**`` defaults to ``user``) → zero rows."""
    config, manager, _ = sync_env
    _set_enrichment_passthrough(mock_client)
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")

    nid = _seed_reenc(manager, source="user")
    _append(config, _reenc_entry(None))  # no Source line → "user"
    manager.perform_sync(auto_accept=True)

    assert _reencounter_rows(manager.store, nid) == []


@patch("google.genai.Client")
def test_sync_reencounter_is_pk_idempotent_across_resyncs(
    mock_client: MagicMock,
    sync_env: Tuple[MitosConfig, MitosSyncManager, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-syncing the differing-source entry TWICE (re-appended after the first
    rotates it out) still yields exactly one row (composite-PK INSERT OR IGNORE)."""
    config, manager, _ = sync_env
    _set_enrichment_passthrough(mock_client)
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")

    nid = _seed_reenc(manager, source="user")
    _append(config, _reenc_entry("import_llm"))
    manager.perform_sync(auto_accept=True)
    _append(config, _reenc_entry("import_llm"))  # the first sync archived the buffer
    manager.perform_sync(auto_accept=True)

    assert len(_reencounter_rows(manager.store, nid)) == 1


# ============================================================================ #
# 3. Import gate (gate 2) — the canonical cross-source case + the P10 proof
# ============================================================================ #


@pytest.fixture
def import_env() -> Tuple[MitosConfig, MitosProseImporter, str]:
    """A temp importer environment (mirrors ``test_importer.py``)."""
    tmpdir = tempfile.mkdtemp()
    config = MitosConfig(tmpdir)
    config.db_path = os.path.join(tmpdir, ".mitos", "graph.sqlite")
    os.makedirs(os.path.join(tmpdir, ".mitos"), exist_ok=True)
    importer = MitosProseImporter(config)
    yield config, importer, tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


def _seed_import_node(importer: MitosProseImporter, source: str) -> str:
    """Seeds a node whose canonical core matches the non-LLM import of the legacy
    file below (axiom = header title, mechanisms = []). Returns node_id."""
    return importer.store.commit_parsed_entry(
        _decision(slug="imp-reenc", axiom="Imported Again", source=source)
    ).node_id


def _write_legacy(tmpdir: str) -> str:
    path = os.path.join(tmpdir, "legacy.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "## 2026-05-19 — imp-reenc — Imported Again\n"
            "Some legacy prose body explaining the decision.\n"
        )
    return path


def test_import_cross_source_reencounter_emits_one_signal(
    import_env: Tuple[MitosConfig, MitosProseImporter, str]
) -> None:
    """The textbook cross-source case: ``mitos import`` (``import_llm``) over a
    hand-authored (``user``) node → one row carrying ``import_llm``; node unchanged."""
    config, importer, tmpdir = import_env
    nid = _seed_import_node(importer, source="user")
    importer.import_from_file(_write_legacy(tmpdir), use_llm_extract=False)

    rows = _reencounter_rows(importer.store, nid)
    assert len(rows) == 1
    assert rows[0]["source"] == "import_llm"
    assert _node_source(importer.store, nid) == "user"  # MI-4 fence


def test_p10_import_gate_blind_when_note_source_reencounter_stashed(
    import_env: Tuple[MitosConfig, MitosProseImporter, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P10 'provoke the failure': stash ``note_source_reencounter`` to a no-op → the
    import gate emits NOTHING (the RED proof the gate is genuinely reactive, not
    incidentally passing). The GREEN counterpart is the test directly above, which
    runs the SAME path with the real method and gets one row. The same-source no-op
    (which writes nothing anyway) stays GREEN under the stash."""
    config, importer, tmpdir = import_env
    nid = _seed_import_node(importer, source="user")
    monkeypatch.setattr(GraphStore, "note_source_reencounter", lambda *a, **k: False)
    importer.import_from_file(_write_legacy(tmpdir), use_llm_extract=False)

    assert _reencounter_rows(importer.store, nid) == []


# ============================================================================ #
# 4. Record gate (gate 3) — agentic-path parity (CLI ⇄ MCP)
# ============================================================================ #


def test_record_cross_source_reencounter_parity(
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """``record_decision_entry`` on an existing node returns ``status:"exists"`` AND
    emits one cross-source row. Seed source=``import_llm``; record authors no
    ``**Source:**`` line, so its new source defaults to ``user`` — a genuine delta."""
    config, manager, _ = sync_env
    nid = manager.store.commit_parsed_entry(
        _decision(slug="rec-reenc", axiom="A recorded axiom.", source="import_llm")
    ).node_id

    result = manager.record_decision_entry(
        axiom="A recorded axiom.",
        rejected_paths="An alternative.",
        scope=[],
        slug="rec-reenc",
    )

    assert result["status"] == "exists"
    rows = _reencounter_rows(manager.store, nid)
    assert len(rows) == 1
    assert rows[0]["source"] == "user"
    assert _node_source(manager.store, nid) == "import_llm"  # MI-4 fence


def test_record_same_source_reencounter_is_noop(
    sync_env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """A re-record whose new source matches the stored source writes no row (still
    returns exists). Seed source=``user``; record's new source is also ``user``."""
    config, manager, _ = sync_env
    nid = manager.store.commit_parsed_entry(
        _decision(slug="rec-same", axiom="A same-source axiom.", source="user")
    ).node_id

    result = manager.record_decision_entry(
        axiom="A same-source axiom.",
        rejected_paths="An alternative.",
        scope=[],
        slug="rec-same",
    )

    assert result["status"] == "exists"
    assert _reencounter_rows(manager.store, nid) == []
