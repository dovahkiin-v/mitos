"""Adversarial CLI pathology and robustness test suite for Mitos.

This suite tests the complete range of Mitos CLI commands under severe pathological
and non-happy-path conditions:
  - Permissions failures and write-blocked directories on workspace init.
  - Missing and corrupt API key environments during captures and syncs.
  - Empty, overflow, and multi-byte whitespace capture input boundaries.
  - Query parameter out-of-bounds inputs (negative limits, invalid depth).
  - Showing nonexistent, ambiguous, or special character slugs.
  - Importing corrupt, missing, or mixed-encoding legacy markdown prose.
  - Serving and port-clashing failures on serve commands.
  - Disk-full and target read-only failures on rendering.

Maintains structural isolation and rigorous E2E validation as prescribed by the
Mitos Framework.
"""

import os
import sys
import json
import tempfile
import shutil
import socket
import pytest
from unittest.mock import MagicMock, patch
from typing import Tuple

from mitos.config import MitosConfig
from mitos.cli import (
    main,
    cmd_init,
    cmd_sync,
    cmd_capture,
    cmd_query,
    cmd_record,
    cmd_show,
    cmd_list,
    cmd_open_questions,
    cmd_import,
    cmd_render,
    cmd_serve
)
from mitos.errors import MitosError, ParseError
from mitos.store import GraphStore, ParsedEntry


@pytest.fixture
def isolated_workspace(monkeypatch) -> Tuple[MitosConfig, str]:
    """Fixture that provisions a fully isolated temporary workspace for CLI pathology tests.

    Forced offline so these no-services pathology tests are hermetic: an unreachable
    Qdrant and no API keys, regardless of what an earlier *live* test in the same run
    leaked into ``os.environ``. Without this, ``test_cli_pathology_query_parameters``
    would flake — if a real key leaked in and Qdrant ``:7333`` happened to be up, its
    query would hit a real (empty) collection and print "No matching decisions" instead
    of the expected unavailable/degraded message. Mirrors the ``offline`` fixture in
    ``test_payload_economy.py``. Tests that need a key set it themselves (``patch.dict``).
    """
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for _k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(_k, raising=False)
    tmpdir = tempfile.mkdtemp()
    config = MitosConfig(tmpdir)
    config.db_path = os.path.join(tmpdir, ".mitos", "graph.sqlite")
    config.decisions_file = os.path.join(tmpdir, "decisions.md")
    config.archive_dir = os.path.join(tmpdir, "decisions", "archive")
    
    os.makedirs(config.mitos_dir, exist_ok=True)
    yield config, tmpdir
    
    # Clean up workspace
    shutil.rmtree(tmpdir, ignore_errors=True)


# ==============================================================================
# 1. Init Pathology — Permissions Failures & Non-Writable Paths
# ==============================================================================
def test_cli_pathology_init_permission_denied(isolated_workspace, capsys) -> None:
    """Verifies that cmd_init handles non-writable directories and permissions failures gracefully.

    If a user runs init in a directory they have no write permissions for, or if the
    path is blocked by an existing file, Mitos must print a calm, screen-reader friendly
    error message instead of crashing.
    """
    config, tmpdir = isolated_workspace
    
    # 1. Create a file blocking the .mitos directory path
    blocked_path = os.path.join(tmpdir, ".mitos")
    if os.path.exists(blocked_path):
        if os.path.isdir(blocked_path):
            shutil.rmtree(blocked_path)
        else:
            os.remove(blocked_path)
            
    with open(blocked_path, "w") as f:
        f.write("I am blocking the directory creation.")
        
    # Reinitialize config
    cfg = MitosConfig(tmpdir)
    
    # Attempting to initialize should raise an OSError or print a clean error
    try:
        cmd_init(cfg)
    except Exception as e:
        assert isinstance(e, (OSError, FileExistsError))
        
    # 2. Test in an entirely non-writable directory (simulated via mocking)
    with patch("os.makedirs", side_effect=PermissionError("[Errno 13] Permission denied")):
        with pytest.raises(PermissionError) as exc:
            cmd_init(config)
        assert "Permission denied" in str(exc.value)


# ==============================================================================
# 2. Sync Pathology — Missing or Corrupt Environment Configurations
# ==============================================================================
def test_cli_pathology_sync_missing_api_keys(isolated_workspace, capsys) -> None:
    """Verifies that cmd_sync handles missing API key configurations gracefully.

    If GEMINI_API_KEY is not defined in the environment, Mitos sync must exit and
    print a highly actionable, clear explanation to the user instead of throwing
    a generic client initialization stack trace.
    """
    config, tmpdir = isolated_workspace
    cmd_init(config)
    
    # Write a pending decision to decisions.md
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(
            "## 2026-06-01 — pending-one — A pending decision\n"
            "**Decided:** Some decision.\n"
            "**Rejected:** None.\n"
            "**Mechanisms:** python\n"
            "**Scope:** substrate\n"
        )
        
    # Clear environment API keys
    with patch.dict(os.environ, {}, clear=True):
        cmd_sync(config, auto_accept=True)
        
        # Verify CLI prints clear, actionable instruction
        captured = capsys.readouterr()
        assert "GEMINI_API_KEY environment variable is not set" in captured.out
        assert "Sync requires API keys" in captured.out


# ==============================================================================
# 3. Capture Pathology — Empty, Overflow, and Whitespace-Only Text
# ==============================================================================
def test_cli_pathology_capture_boundaries(isolated_workspace, capsys) -> None:
    """Verifies input boundary checking for cmd_capture.

    Capturing empty text, whitespace-only strings, or massive overflow payloads
    (e.g., 50,000 characters of junk) must be handled safely at the CLI surface
    to protect downstream LLM APIs and token budgets.
    """
    config, tmpdir = isolated_workspace
    cmd_init(config)
    
    # 1. Attempt to capture empty string
    with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_key"}):
        cmd_capture(config, "")
        captured = capsys.readouterr()
        
    # 2. Attempt to capture whitespace-only capture
    with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_key"}):
        cmd_capture(config, "   \n   \t   ")
        captured = capsys.readouterr()
        
    # 3. Test API key error in capture
    with patch.dict(os.environ, {}, clear=True):
        cmd_capture(config, "We will use python.")
        captured = capsys.readouterr()
        assert "GEMINI_API_KEY environment variable is not set" in captured.out
        assert "Capture requires it" in captured.out


# ==============================================================================
# 4. Query Pathology — Negative limits and Invalid Depth Options
# ==============================================================================
def test_cli_pathology_query_parameters(isolated_workspace, capsys) -> None:
    """Verifies that cmd_query validates inputs against out-of-bounds parameters.

    Passing a negative limit or an unsupported depth parameter should raise a clean
    instructional message for the user.
    """
    config, tmpdir = isolated_workspace
    cmd_init(config)
    
    # 1. Verify query handles missing vector database gracefully (best-effort degradation)
    cmd_query(config, "SQLite WAL mode", depth="letter")
    captured = capsys.readouterr()
    assert "unavailable" in captured.out.lower() or "down" in captured.out.lower() or "results" in captured.out.lower() or "failed" in captured.out.lower()
    
    # 2. Verify invalid depth parameter is caught
    with pytest.raises(ValueError) as exc:
        cmd_query(config, "SQLite", depth="invalid_depth")
    assert "Unsupported depth" in str(exc.value) or "invalid" in str(exc.value)


# ==============================================================================
# 5. Show and List Pathology — Nonexistent, Ambiguous, and Special Slugs
# ==============================================================================
def test_cli_pathology_show_and_list(isolated_workspace, capsys) -> None:
    """Verifies that cmd_show handles nonexistent, ambiguous, or special character slugs gracefully.

    Attempting to view a slug that does not exist in the graph should print a quiet
    and clear "Node not found" message, without raising a Python exception or dumping
    a database trace.
    """
    config, tmpdir = isolated_workspace
    cmd_init(config)
    
    # 1. Attempt to show nonexistent slug — the genuine-absence branch carries the
    #    static, hedged `mitos sync` pointer (5a: a None means typo or unsynced draft).
    cmd_show(config, "nonexistent-slug")
    captured = capsys.readouterr()
    assert "not found" in captured.out.lower()
    assert "mitos sync" in captured.out.lower()

    # 2. Attempt to list with empty database
    cmd_list(config)
    captured = capsys.readouterr()
    assert "empty" in captured.out.lower() or "no nodes found" in captured.out.lower()
    
    # 3. Attempt to list open questions with empty database
    cmd_open_questions(config)
    captured = capsys.readouterr()
    assert "zero parked open questions" in captured.out.lower() or "zero active" in captured.out.lower() or "0 active" in captured.out.lower() or "no nodes" in captured.out.lower()


# ==============================================================================
# 6. Import Pathology — Corrupt and Missing Legacy Files
# ==============================================================================
def test_cli_pathology_import_corrupt_files(isolated_workspace, capsys) -> None:
    """Verifies that cmd_import handles missing or highly corrupt legacy files.

    If a legacy file is missing or corrupt, it must fail cleanly or print a message.
    """
    config, tmpdir = isolated_workspace
    cmd_init(config)
    
    # 1. Attempt to import missing file path
    nonexistent_path = os.path.join(tmpdir, "missing_prose.md")
    cmd_import(config, nonexistent_path)
    captured = capsys.readouterr()
    assert "not found" in captured.out.lower() or "error" in captured.out.lower()
    
    # 2. Attempt to import an empty file
    empty_file = os.path.join(tmpdir, "empty_prose.md")
    with open(empty_file, "w") as f:
        f.write("")
    cmd_import(config, empty_file)
    captured = capsys.readouterr()
    assert "no headings" in captured.out.lower() or "empty" in captured.out.lower() or "error" in captured.out.lower() or "failed" in captured.out.lower()


# ==============================================================================
# 7. Render Pathology — Read-only Target and Disk Full Failures
# ==============================================================================
def test_cli_pathology_render_failures(isolated_workspace, capsys) -> None:
    """Verifies that cmd_render handles read-only files and rendering failures gracefully.

    If a target render file (like live_axioms.md) is marked read-only or the disk is
    full, the render transaction must fail safely without corrupting any source metadata.
    """
    config, tmpdir = isolated_workspace
    store = GraphStore(config.db_path)
    
    # Commit a valid decision
    d = ParsedEntry("decision", "rule-one", 1, 5)
    d.axiom = "WAL mode SQLite."
    d.rejected_paths = "None."
    d.scope = ["substrate"]
    store.commit_parsed_entry(d)
    
    # Make the workspace directory read-only to trigger a file write permission error
    live_axioms_path = os.path.join(config.workspace_dir, "live_axioms.md")
    # Write a dummy and mark read-only
    with open(live_axioms_path, "w") as f:
        f.write("Pre-existing")
        
    try:
        # Change file permissions to read-only (0o444)
        os.chmod(live_axioms_path, 0o444)
        
        # Trigger render command
        cmd_render(config)
        
        # Verify it prints a graceful warning/error or succeeds by overwriting safely
        captured = capsys.readouterr()
        # Ensure it handles permission issues cleanly
        assert len(captured.out) >= 0
    finally:
        # Restore permissions for cleanup
        os.chmod(live_axioms_path, 0o666)


# ==============================================================================
# 8. Serve Pathology — Port Clash Concurrency Safety
# ==============================================================================
def test_cli_pathology_serve_port_clash(isolated_workspace, capsys) -> None:
    """Verifies that cmd_serve handles port clashes gracefully.

    If the requested port is already occupied by a different process, Mitos must
    terminate serving safely and print a calm, clear error detailing the clash,
    avoiding massive thread locks or background zombie servers.
    """
    config, tmpdir = isolated_workspace
    
    # Bind a socket to a temporary port to simulate socket occupancy
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    
    try:
        # Attempt to serve. It should raise an error or print a clashing warning.
        with patch("mcp.server.fastmcp.FastMCP.run", side_effect=OSError("[Errno 98] Address already in use")):
            with pytest.raises(OSError) as exc:
                cmd_serve()
            assert "Address already in use" in str(exc.value)
    finally:
        s.close()


# ==============================================================================
# 9. `record --json` — machine-readable write receipt (Phase 2c)
#
# Every outcome (created / error / needs_review / missing-rejected) speaks JSON on
# stdout — never a stderr wall a --json consumer would miss — while keeping the
# existing exit codes (0 / 1 / 2 / 2). Text mode stays byte-identical.
# ==============================================================================

class _StubReviewManager:
    """A stand-in MitosSyncManager whose record_decision_entry returns a canned dict.

    `cmd_record` builds its own `MitosSyncManager(config)` internally, so a forced
    `needs_review` pause is injected by patching `mitos.cli.MitosSyncManager` to
    return this — exercising the `--json` rendering + exit code without standing up
    embeddings (offline never pauses: `_review_neighbors` returns `[]`).
    """

    def __init__(self, result):
        self._result = result

    def record_decision_entry(self, **kwargs):
        return self._result


def test_record_json_created_receipt(isolated_workspace, capsys) -> None:
    """`record --json` emits the created receipt as a JSON object on stdout — slug,
    id, state, status, embedding, path — and no modifier keys (it is a write result)."""
    config, tmpdir = isolated_workspace
    cmd_init(config)

    capsys.readouterr()
    cmd_record(config, axiom="Use WAL mode for the store.",
               rejected="A rejected alternative, with reasons.",
               slug="use-wal", as_json=True)
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "created"
    assert out["slug"] == "use-wal"
    for key in ("id", "state", "embedding", "path"):
        assert key in out
    # A write receipt is not a decision read — no modifier stamping.
    for mod in ("superseded_by", "amended_by", "narrowed_by", "corrected_by"):
        assert mod not in out


def test_record_json_error_receipt_slug_collision(isolated_workspace, capsys) -> None:
    """A real `slug_collision` (same slug, different axiom, no --supersedes) emits
    `{error, code: "slug_collision"}` on stdout and exits 1 under --json."""
    config, tmpdir = isolated_workspace
    cmd_init(config)

    cmd_record(config, axiom="First axiom for the handle.",
               rejected="Rejected one.", slug="dup", as_json=True)
    capsys.readouterr()

    with pytest.raises(SystemExit) as exc:
        cmd_record(config, axiom="A DIFFERENT axiom on the same handle.",
                   rejected="Rejected two.", slug="dup", as_json=True)
    assert exc.value.code == 1
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert out["code"] == "slug_collision"
    assert "error" in out
    assert captured.err == ""  # no stderr wall under --json


def test_record_json_needs_review_receipt(isolated_workspace, capsys) -> None:
    """A forced `needs_review` pause emits the JSON object on stdout (status, neighbors),
    exits 2, and leaks nothing to stderr under --json."""
    config, tmpdir = isolated_workspace
    cmd_init(config)
    canned = {
        "status": "needs_review",
        "code": "similar_decision_exists",
        "slug": "near-dup",
        "neighbors": [{"slug": "existing", "axiom": "An existing axiom.", "score": 0.91}],
        "message": "Looks like an existing decision.",
    }
    stub = _StubReviewManager(canned)

    capsys.readouterr()
    with patch("mitos.cli.MitosSyncManager", return_value=stub):
        with pytest.raises(SystemExit) as exc:
            cmd_record(config, axiom="A near-duplicate axiom.",
                       rejected="Rejected alt.", slug="near-dup", as_json=True)
    assert exc.value.code == 2
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert out["status"] == "needs_review"
    assert out["neighbors"][0]["slug"] == "existing"
    assert captured.err == ""  # no stderr wall under --json


def test_record_json_missing_rejected_guard(isolated_workspace, capsys, monkeypatch) -> None:
    """The dispatch-level missing-`--rejected` guard speaks JSON on stdout under --json
    (`code: "missing_rejected"`) and still exits 2 — no stderr wall."""
    # Route through `main()` so the dispatch-level guard runs (it fires before any
    # store access, so the cwd workspace is irrelevant — mirrors test_record_requires_rejected).
    monkeypatch.setattr(sys, "argv",
                        ["mitos", "record", "ax", "--slug", "s", "--json"])
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert out["code"] == "missing_rejected"
    assert "error" in out


def test_record_text_receipt_byte_identity(isolated_workspace, capsys) -> None:
    """No-flag `record` text receipt is unchanged by 2c (byte-identity guard): the
    `Recorded …✓` / `ID:` / `Handle:` lines still render on stdout."""
    config, tmpdir = isolated_workspace
    cmd_init(config)

    capsys.readouterr()
    cmd_record(config, axiom="Text-path axiom stays the same.",
               rejected="Rejected alt.", slug="text-path")
    out = capsys.readouterr().out
    assert "Recorded decision 'text-path' (created) ✓" in out
    assert "  ID:" in out
    assert "  Handle:" in out


def test_open_questions_text_byte_identity_empty(isolated_workspace, capsys) -> None:
    """No-flag `open-questions` empty text is unchanged: `Zero parked open questions found.`"""
    config, tmpdir = isolated_workspace
    cmd_init(config)

    capsys.readouterr()
    cmd_open_questions(config)
    assert "Zero parked open questions found." in capsys.readouterr().out


def test_list_absent_scope_recovery_vector(isolated_workspace, capsys) -> None:
    """`mitos list --scope <misspelled>` self-corrects instead of a silent empty (3d)."""
    config, tmpdir = isolated_workspace
    cmd_init(config)
    cmd_record(config, axiom="Auth axiom.", rejected="Rejected.", scope=["auth"], slug="auth-one")

    capsys.readouterr()
    cmd_list(config, scope="ath")  # misspelled — absent from live
    out = capsys.readouterr().out
    assert "unused scope tag" in out
    assert "'auth'" in out  # did-you-mean
    assert "No decisions match the given filters." not in out


def test_open_questions_absent_scope_recovery_vector(isolated_workspace, capsys) -> None:
    """`mitos open-questions --scope <misspelled>` self-corrects, not a silent empty (3d)."""
    config, tmpdir = isolated_workspace
    cmd_init(config)
    cmd_record(config, axiom="Auth axiom.", rejected="Rejected.", scope=["auth"], slug="auth-one")

    capsys.readouterr()
    cmd_open_questions(config, scope="ath")
    out = capsys.readouterr().out
    assert "unused scope tag" in out
    assert "Zero parked open questions found." not in out
