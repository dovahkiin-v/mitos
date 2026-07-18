"""Corpus-provenance stamping on the read surfaces.

Every recall answer names which corpus it came from (``collection`` +
``workspace``), so an empty or twilight result is never ambiguous between
"no precedent exists" and "you're standing in the wrong workspace"
(AX 2026-07-01: the reviewing cwd and a vision's decision store can diverge).
Pins the fields on the CLI JSON envelopes, the MCP twins, and the degraded
lexical fallback, plus the text-mode provenance line.
"""

import json
import os
import subprocess
import sys

import pytest

from mitos.config import MitosConfig
from mitos.recall import corpus_provenance, provenance_line


class TestHelpers:
    def test_corpus_provenance_fields(self, tmp_path):
        config = MitosConfig(workspace_dir=str(tmp_path))
        p = corpus_provenance(config)
        assert p["collection"] == config.qdrant_collection
        assert p["workspace"] == config.workspace_dir

    def test_provenance_line_contains_both(self, tmp_path):
        config = MitosConfig(workspace_dir=str(tmp_path))
        line = provenance_line(config)
        assert config.qdrant_collection in line
        assert config.workspace_dir in line


@pytest.fixture
def workspace(tmp_path):
    """A minimal initialized workspace with one synced decision."""
    ws = str(tmp_path)
    env = {**os.environ, "GEMINI_API_KEY": "", "GOOGLE_API_KEY": "",
           "QDRANT_URL": "http://localhost:1"}

    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "mitos.cli", *args],
            capture_output=True, text=True, cwd=ws, env=env,
        )

    run("init")
    with open(os.path.join(ws, "decisions.md"), "a", encoding="utf-8") as f:
        f.write(
            "\n### provenance-test-decision\n"
            "**Decided:** Provenance headers ride every read surface.\n"
            "**Rejected paths:** Silent corpus ambiguity.\n"
            "**Scope:** testing\n"
            "**Date:** 2026-07-18\n"
        )
    run("sync", "--yes")
    return ws, run


class TestCliJson:
    def test_list_json_carries_provenance(self, workspace):
        ws, run = workspace
        out = run("list", "--json")
        payload = json.loads(out.stdout)
        assert payload["collection"] == MitosConfig(workspace_dir=ws).qdrant_collection
        assert payload["workspace"] == os.path.abspath(ws)

    def test_degraded_surface_json_carries_provenance(self, workspace):
        # Qdrant points at a dead port and no key is set → the lexical fallback
        # fires, and its envelope must still name the corpus.
        ws, run = workspace
        out = run("surface", "provenance headers", "--json")
        payload = json.loads(out.stdout)
        assert payload.get("degraded") == "lexical"
        assert payload["collection"] == MitosConfig(workspace_dir=ws).qdrant_collection
        assert payload["workspace"] == os.path.abspath(ws)

    def test_degraded_surface_text_has_provenance_line(self, workspace):
        ws, run = workspace
        out = run("surface", "provenance headers")
        assert f"corpus: {MitosConfig(workspace_dir=ws).qdrant_collection}" in out.stdout

    def test_list_text_header_has_provenance(self, workspace):
        ws, run = workspace
        out = run("list")
        assert f"corpus: {MitosConfig(workspace_dir=ws).qdrant_collection}" in out.stdout


class TestMcpTwins:
    def test_list_decisions_payload_carries_provenance(self, workspace, monkeypatch):
        ws, _ = workspace
        monkeypatch.chdir(ws)
        monkeypatch.setenv("QDRANT_URL", "http://localhost:1")
        monkeypatch.setenv("GEMINI_API_KEY", "")
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        from mitos import mcp_server
        payload = json.loads(mcp_server.list_decisions())
        assert payload["collection"] == MitosConfig(workspace_dir=ws).qdrant_collection
        assert payload["workspace"] == os.path.abspath(ws)

    def test_surface_decisions_degraded_carries_provenance(self, workspace, monkeypatch):
        ws, _ = workspace
        monkeypatch.chdir(ws)
        monkeypatch.setenv("QDRANT_URL", "http://localhost:1")
        monkeypatch.setenv("GEMINI_API_KEY", "")
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        from mitos import mcp_server
        payload = json.loads(mcp_server.surface_decisions("provenance headers"))
        assert payload.get("degraded") == "lexical"
        assert payload["collection"] == MitosConfig(workspace_dir=ws).qdrant_collection
