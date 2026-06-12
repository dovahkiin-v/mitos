"""End-to-end integration tests against REAL services (Qdrant :7333 + Gemini).

Unlike the unit tests (which mock Qdrant/embeddings/routing), these wire the whole
stack together and prove the thing Mitos actually exists for: record a decision →
it's embedded + upserted → surface recalls it semantically. They also exercise the
real `mitos` binary as a subprocess (real argv, real stdin, the update/hint
side-effects), which an in-process `main()` call can't.

Skipped unless a real ``GEMINI_API_KEY`` is resolvable (env → the global
``~/.config/mitos/.env`` → the dev clone's ``.env``) AND Qdrant ``:7333`` is up.
Each test uses an isolated workspace and DELETES its Qdrant collection on teardown,
so the shared instance is never polluted.
"""

import json
import os
import subprocess
import sys

import pytest
import requests

from mitos import cli
from mitos.config import MitosConfig, default_collection_name


def _read_key_from_env_file(path: str, name: str = "GEMINI_API_KEY"):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{name}="):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if value:
                        return value
    except OSError:
        pass
    return None


def _resolve_real_gemini_key():
    """Finds a real Gemini key for the live run (NOT the hermetic test config)."""
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in (os.path.expanduser("~/.config/mitos/.env"),
                 os.path.join(repo_root, ".env")):
        key = _read_key_from_env_file(path)
        if key:
            return key
    return None


QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:7333")


def _qdrant_up():
    try:
        return requests.get(f"{QDRANT_URL}/collections", timeout=2).ok
    except Exception:
        return False


_REAL_KEY = _resolve_real_gemini_key()
HAS_SERVICES = bool(_REAL_KEY) and _qdrant_up()

pytestmark = pytest.mark.skipif(
    not HAS_SERVICES,
    reason="integration: needs a real GEMINI_API_KEY (env/global/dev .env) + Qdrant :7333",
)


def _drop_collection(name: str) -> None:
    try:
        requests.delete(f"{QDRANT_URL}/collections/{name}", timeout=5)
    except Exception:
        pass


@pytest.fixture
def live_workspace(tmp_path, monkeypatch):
    """An initialized workspace wired to real services, with Qdrant cleanup.

    The real key is forced into the env (overriding the hermetic conftest, which
    deliberately hides the global .env from the unit tests).
    """
    monkeypatch.setenv("GEMINI_API_KEY", _REAL_KEY)
    monkeypatch.setenv("QDRANT_URL", QDRANT_URL)
    ws = tmp_path / "proj"
    ws.mkdir()
    cli.cmd_init(MitosConfig(str(ws)))
    collection = default_collection_name(str(ws))
    try:
        yield ws, collection
    finally:
        _drop_collection(collection)


def test_record_then_surface_recall(live_workspace, capsys):
    """The full recall loop: a recorded decision is surfaced by a related query."""
    ws, _ = live_workspace
    cli.cmd_record(
        MitosConfig(str(ws)),
        axiom="The Portuguese tutor fails fast on missing learner data",
        rejected="Graceful-degrade rejected: a silent wrong answer is worse than a loud error",
        scope=["personas"],
        mechanisms=["prompts.py"],
    )
    capsys.readouterr()  # flush the record output

    cli.cmd_surface(
        MitosConfig(str(ws)),
        "how should the tutor handle missing data",
        scope="personas",
        as_json=True,
    )
    data = json.loads(capsys.readouterr().out)
    slugs = [d["slug"] for d in data["active_decisions"]]
    assert any("fails-fast" in s for s in slugs), f"recall missed the decision: {slugs}"


def test_status_readiness_against_real_qdrant(live_workspace, capsys):
    """The catch-22 fix holds against REAL Qdrant: empty → READY, then a real
    record auto-creates the collection and it stays READY."""
    ws, _ = live_workspace
    assert cli.cmd_status(str(ws)) == 0  # fresh init, no collection yet → READY
    capsys.readouterr()

    cli.cmd_record(
        MitosConfig(str(ws)),
        axiom="SQLite WAL mode is the graph store",
        rejected="pgvector rejected: too heavy for a local-first tool",
        scope=["substrate"],
    )
    capsys.readouterr()

    code = cli.cmd_status(str(ws), as_json=True)
    data = json.loads(capsys.readouterr().out)
    assert code == 0 and data["ready"] is True
    assert data["checks"]["collection_exists"] is True  # the record created it


def test_cli_subprocess_record_stdin_then_surface(tmp_path):
    """Real binary, real argv, real stdin pipe, real services — the AX fixes
    (MCP-name alias + --rejected-file stdin + surface recall) end-to-end."""
    mitos_bin = os.path.join(os.path.dirname(sys.executable), "mitos")
    ws = tmp_path / "proj"
    ws.mkdir()
    env = {
        **os.environ,
        "GEMINI_API_KEY": _REAL_KEY,
        "QDRANT_URL": QDRANT_URL,
        "MITOS_NO_UPDATE_CHECK": "1",
        "MITOS_NO_MCP_HINT": "1",
        "XDG_CONFIG_HOME": str(tmp_path / "cfg"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    }
    collection = default_collection_name(str(ws))
    try:
        init = subprocess.run([mitos_bin, "init"], cwd=ws, env=env,
                              capture_output=True, text=True)
        assert init.returncode == 0, init.stderr

        # MCP-name alias + prose via stdin (apostrophes must survive)
        rec = subprocess.run(
            [mitos_bin, "record_decision",
             "Camila's tutor fails fast on missing data", "--rejected-file", "-",
             "--scope", "personas"],
            cwd=ws, env=env, text=True, capture_output=True,
            input="Rejected graceful-degrade: Camila's tutor must never show a silent wrong answer",
        )
        assert rec.returncode == 0, rec.stderr
        assert "Recorded decision" in rec.stdout

        surf = subprocess.run(
            [mitos_bin, "surface", "how should the tutor handle missing data",
             "--scope", "personas"],
            cwd=ws, env=env, capture_output=True, text=True,
        )
        assert surf.returncode == 0, surf.stderr
        assert "Camila's" in surf.stdout or "fails-fast" in surf.stdout
    finally:
        _drop_collection(collection)
