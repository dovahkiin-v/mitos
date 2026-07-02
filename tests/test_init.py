"""Integration tests for ``mitos init`` (Phase 6b) and its ``status`` coherence.

``cmd_init`` had no direct suite before 6b — it was only exercised indirectly as a
setup fixture across other modules, so the §5.2.7 re-run-safety contract had no
home. This file is it: it inits into a ``tmp_path`` workspace and asserts the
filesystem + idempotency + pre-V1a-refusal outcomes (no external services; the
``status`` Qdrant call is monkeypatched, as ``test_status_readiness`` does).
"""

import os
import sqlite3
import tomllib

import pytest

from mitos import cli
from mitos.config import (
    MitosConfig,
    CONFIG_DEFAULTS,
    CONFIG_SCHEMA,
    default_collection_name,
)
from mitos.migrations import MIGRATION_STEPS, _pending_head
from mitos.store import GraphStore
from mitos.errors import DatabaseError


# --- helpers ---------------------------------------------------------------

def _init(path):
    """Runs ``cmd_init`` on a fresh ``MitosConfig`` for ``path``."""
    cli.cmd_init(MitosConfig(str(path)))


def _qdrant(reachable=True, collection_exists=False, points=None):
    """Builds a ``_check_qdrant`` stub (no real Qdrant in init/status tests)."""
    return lambda url, coll: {
        "reachable": reachable,
        "collection_exists": collection_exists,
        "points": points,
    }


def _plant_prototype_graph(db_path):
    """Builds a real pre-V1a (prototype ``_init_db``) graph at ``db_path``.

    Mirrors ``test_store.py::test_init_boot_guard_refuses_prototype_graph``'s
    ``__new__`` + ``_init_db`` bypass (the §16 retained-fixture pattern).
    """
    proto = GraphStore.__new__(GraphStore)
    proto.db_path = db_path
    proto.read_only = False
    proto._init_db()


def _read(path):
    return path.read_text(encoding="utf-8")


# --- from-scratch boot (Lesson 55) -----------------------------------------

def test_fresh_init_creates_v1a_workspace(tmp_path):
    """A fresh init yields every V1a artifact and boots the graph at the ladder head."""
    _init(tmp_path)

    assert (tmp_path / ".mitos").is_dir()
    for rel in (
        ".mitos/config.toml",
        ".mitos/graph.sqlite",
        ".mitos/skill.md",
        "format-spec.md",
        "decisions.md",
        "questions.md",
        ".env",
        ".gitignore",
    ):
        assert (tmp_path / rel).exists(), f"missing {rel}"

    # The graph booted via the migration ladder to the live head (read the version
    # programmatically — never hardcode a migration revision; the head moves as later
    # visions append rungs, e.g. V1b's step 2).
    conn = sqlite3.connect(str(tmp_path / ".mitos" / "graph.sqlite"))
    try:
        assert conn.execute("PRAGMA user_version;").fetchone()[0] == _pending_head(
            MIGRATION_STEPS
        )
    finally:
        conn.close()


def test_seeded_config_round_trips_clean(tmp_path, capsys):
    """The seeded config.toml re-loads through 6a's strict loader with zero warnings.

    This is the seeder↔loader single-source proof (P11 / Decision 1): every attr
    equals its default, no ``pending_threshold`` line, no unknown-key stderr warning.
    """
    _init(tmp_path)
    capsys.readouterr()  # discard cmd_init's "Initialized…" line

    config = MitosConfig(str(tmp_path))
    captured = capsys.readouterr()
    assert "unrecognized" not in captured.err
    assert captured.err == ""  # a clean nine-key file warns about nothing

    for key, default in CONFIG_DEFAULTS.items():
        assert getattr(config, key) == default
    assert config.qdrant_url == "http://localhost:7333"
    assert config.qdrant_collection == default_collection_name(str(tmp_path))

    body = _read(tmp_path / ".mitos" / "config.toml")
    assert "pending_threshold" not in body


def test_config_seeds_exactly_the_schema_keys(tmp_path):
    """The seed writes exactly CONFIG_DEFAULTS' static keys + the two dynamic qdrant keys."""
    _init(tmp_path)
    with open(tmp_path / ".mitos" / "config.toml", "rb") as f:
        data = tomllib.load(f)

    assert set(data) == set(CONFIG_SCHEMA)  # every recognized key, nothing else
    assert set(data) == set(CONFIG_DEFAULTS) | {"qdrant_url", "qdrant_collection"}
    assert "pending_threshold" not in data
    for key, default in CONFIG_DEFAULTS.items():
        assert data[key] == default


def test_config_seeds_conflict_check_as_native_bool(tmp_path):
    """A fresh init seeds ``conflict_check_on_sync = true`` (lowercase, unquoted).

    The v0.2 toggle is the first bool-typed key; ``mitos init`` must NOT crash on
    it (the ``_toml_scalar`` bool-serializer trap) and must emit a native TOML
    boolean that ``tomllib`` parses straight back to Python ``True``.
    """
    _init(tmp_path)  # must not raise — the central 1a gotcha

    body = _read(tmp_path / ".mitos" / "config.toml")
    assert "conflict_check_on_sync = true" in body  # lowercase, unquoted native bool
    assert "conflict_check_on_sync = 1" not in body
    assert 'conflict_check_on_sync = "true"' not in body

    with open(tmp_path / ".mitos" / "config.toml", "rb") as f:
        data = tomllib.load(f)
    assert data["conflict_check_on_sync"] is True


def test_questions_md_seeded_with_sentinel_and_sample(tmp_path):
    """questions.md carries the BEGIN ENTRIES sentinel and the ## 4 open-question sample."""
    _init(tmp_path)
    body = _read(tmp_path / "questions.md")
    assert "BEGIN ENTRIES" in body  # the parser's preamble-split substring
    assert "example-open-question" in body  # the ## 4 sample sits in the preamble
    # The sentinel is above the entry stream; the sample sits in the preamble.
    assert body.index("example-open-question") < body.index("BEGIN ENTRIES")


# --- re-run idempotency (§5.2.7) -------------------------------------------

def test_reinit_idempotent_on_present_files(tmp_path):
    """A second init leaves present config/buffers byte-identical and the ladder a no-op."""
    _init(tmp_path)
    targets = [".mitos/config.toml", "decisions.md", "questions.md"]
    before = {t: _read(tmp_path / t) for t in targets}

    _init(tmp_path)
    after = {t: _read(tmp_path / t) for t in targets}
    assert before == after

    conn = sqlite3.connect(str(tmp_path / ".mitos" / "graph.sqlite"))
    try:
        # Ladder no-op on re-init: still at the live head (read programmatically).
        assert conn.execute("PRAGMA user_version;").fetchone()[0] == _pending_head(
            MIGRATION_STEPS
        )
    finally:
        conn.close()


def test_reinit_reseeds_deleted_buffers(tmp_path):
    """A deleted decisions.md / questions.md is re-seeded on the next init."""
    _init(tmp_path)
    (tmp_path / "decisions.md").unlink()
    (tmp_path / "questions.md").unlink()

    _init(tmp_path)
    assert (tmp_path / "decisions.md").exists()
    assert (tmp_path / "questions.md").exists()
    assert "BEGIN ENTRIES" in _read(tmp_path / "questions.md")


def test_reinit_leaves_edited_config_untouched(tmp_path):
    """A present (user-edited) config.toml is never rewritten — present file untouched (§5.2.7)."""
    _init(tmp_path)
    config_path = tmp_path / ".mitos" / "config.toml"
    # A valid in-place value change (NOT a duplicate key — that would be malformed
    # TOML) plus a hand comment, the shape a user edit really takes.
    edited = _read(config_path).replace(
        'rotation_mode = "archive"', '# a human\'s hand-edit\nrotation_mode = "mark"'
    )
    assert edited != _read(config_path)  # the substitution actually fired
    config_path.write_text(edited, encoding="utf-8")

    _init(tmp_path)
    assert _read(config_path) == edited


# --- format-spec refresh-on-mismatch ---------------------------------------

def test_format_spec_refresh_on_mismatch(tmp_path, capsys):
    """A drifted format-spec.md is refreshed to the shipped copy with a both-hash warning."""
    _init(tmp_path)
    capsys.readouterr()

    spec_path = tmp_path / "format-spec.md"
    spec_path.write_text("CORRUPTED SPEC\n", encoding="utf-8")

    _init(tmp_path)
    out = capsys.readouterr().out
    assert "Refreshed format-spec.md" in out
    assert "→" in out  # names both short hashes (old → new)
    refreshed = _read(spec_path)
    assert "CORRUPTED" not in refreshed
    assert refreshed == cli.load_format_spec()


def test_format_spec_in_sync_is_silent_noop(tmp_path, capsys):
    """An in-sync format-spec.md is a silent no-op — no warning, no rewrite."""
    _init(tmp_path)
    capsys.readouterr()

    _init(tmp_path)  # spec already matches the shipped copy
    assert "Refreshed format-spec.md" not in capsys.readouterr().out


# --- pre-V1a refusal --------------------------------------------------------

def test_init_refuses_pre_v1a_graph_before_mutation(tmp_path):
    """init refuses a prototype graph, routing to cutover, with NO scaffolding written."""
    config = MitosConfig(str(tmp_path))
    _plant_prototype_graph(config.db_path)  # creates only .mitos/graph.sqlite

    with pytest.raises(DatabaseError) as exc:
        cli.cmd_init(config)
    assert "cutover" in str(exc.value).lower()

    # Abort-before-partial-mutation: only the planted prototype graph exists; init
    # wrote no config, buffers, format-spec, .env, or skill.
    assert not (tmp_path / ".mitos" / "config.toml").exists()
    assert not (tmp_path / ".mitos" / "skill.md").exists()
    assert not (tmp_path / "decisions.md").exists()
    assert not (tmp_path / "questions.md").exists()
    assert not (tmp_path / "format-spec.md").exists()
    assert not (tmp_path / ".env").exists()


# --- status detection -------------------------------------------------------

def test_status_reports_pre_v1a_not_ready(tmp_path, monkeypatch):
    """status reports an otherwise-healthy workspace as NOT ready when the graph is pre-V1a."""
    # Stand up an otherwise-ready workspace (config + buffer + key + reachable Qdrant)
    # so the ONLY blocker is the prototype graph — proving pre_v1a is decisive.
    config = MitosConfig(str(tmp_path))
    os.makedirs(config.mitos_dir, exist_ok=True)
    (tmp_path / ".mitos" / "config.toml").write_text('rotation_mode = "archive"\n', encoding="utf-8")
    (tmp_path / "decisions.md").write_text("# decisions\n", encoding="utf-8")
    (tmp_path / "format-spec.md").write_text(cli.load_format_spec(), encoding="utf-8")
    _plant_prototype_graph(config.db_path)

    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(reachable=True, collection_exists=True, points=1))

    import io
    import json
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.cmd_status(str(tmp_path))
    out = buf.getvalue()
    assert code == 1
    assert "READY ✓" not in out
    assert "cutover" in out.lower()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli.cmd_status(str(tmp_path), as_json=True)
    data = json.loads(buf.getvalue())
    assert data["ready"] is False
    assert data["pre_v1a"] is True


def test_empty_v1a_graph_reports_ready(tmp_path, monkeypatch):
    """A freshly-init'ed empty V1a graph is healthy, not broken (empty-is-healthy, P5)."""
    _init(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant(reachable=True, collection_exists=False))

    import io
    import json
    import contextlib

    assert cli.cmd_status(str(tmp_path)) == 0  # READY despite zero nodes / no collection

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli.cmd_status(str(tmp_path), as_json=True)
    data = json.loads(buf.getvalue())
    assert data["ready"] is True
    assert data["pre_v1a"] is False


def test_status_graceful_on_malformed_config(tmp_path, capsys):
    """A malformed config.toml makes status report not-ready GRACEFULLY — exit 1, no traceback."""
    os.makedirs(tmp_path / ".mitos", exist_ok=True)
    (tmp_path / ".mitos" / "config.toml").write_text("rotation_mode = =broken\n", encoding="utf-8")

    code = cli.cmd_status(str(tmp_path))
    captured = capsys.readouterr()
    assert code == 1
    assert "malformed" in captured.out.lower()
    assert "Traceback" not in captured.err  # Lesson 45: located, not a wall of stack

    code = cli.cmd_status(str(tmp_path), as_json=True)
    import json
    data = json.loads(capsys.readouterr().out)
    assert code == 1
    assert data["ready"] is False
    assert "config_error" in data


# --- §2 non-regression (inherited setup surface) ---------------------------

def test_init_preserves_env_gitignore_qdrant_scaffolding(tmp_path):
    """init must not regress the inherited .env + .gitignore + qdrant_* setup surface (§2)."""
    _init(tmp_path)

    env_body = _read(tmp_path / ".env")
    assert "GEMINI_API_KEY=" in env_body
    assert "ANTHROPIC_API_KEY=" in env_body

    gitignore_body = _read(tmp_path / ".gitignore")
    assert ".env" in gitignore_body.splitlines()

    with open(tmp_path / ".mitos" / "config.toml", "rb") as f:
        data = tomllib.load(f)
    assert data["qdrant_url"] == "http://localhost:7333"
    assert data["qdrant_collection"] == default_collection_name(str(tmp_path))
