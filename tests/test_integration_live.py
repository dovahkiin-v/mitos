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

import shutil
import pytest
import requests

from mitos import cli
from mitos.config import MitosConfig, default_collection_name
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager

from live_helpers import skip_if_embed_quota_exhausted, skip_if_global_mitos_stale


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
        slug="tutor-pt-fails-fast",
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
        slug="sqlite-wal-mode"
    )
    capsys.readouterr()

    code = cli.cmd_status(str(ws), as_json=True)
    data = json.loads(capsys.readouterr().out)
    assert code == 0 and data["ready"] is True
    assert data["checks"]["collection_exists"] is True  # the record created it


def test_list_decisions_complete_set_vs_capped_surface(live_workspace, capsys):
    """The exhaustive path's whole reason for being, proven against REAL recall:
    list_decisions returns EVERY decision in a scope, where semantic surface ranks
    and caps at the top few — so a completeness pass can't miss anything below the
    relevance cliff (loop-Claude's "am I seeing everything?" gap)."""
    ws, _ = live_workspace
    config = MitosConfig(str(ws))

    decisions = [
        ("Payments settle through Stripe as the single PSP",
         "Adyen rejected: heavier integration for our volume"),
        ("Idempotency keys are required on every charge request",
         "Dedup-by-amount rejected: collides on legitimate repeat purchases"),
        ("Refunds are asynchronous and webhook-driven",
         "Synchronous refunds rejected: blocks the request on PSP latency"),
        ("Currency is stored in minor units as integers",
         "Floats rejected: rounding drift accumulates"),
        ("Failed charges retry with exponential backoff, capped at three",
         "Infinite retry rejected: hammers the PSP on hard declines"),
        ("Webhook signatures are verified before any processing",
         "Trust-by-source-IP rejected: spoofable and brittle"),
        ("Payment state lives in an append-only ledger",
         "A mutable balance row rejected: loses the audit trail"),
    ]
    for i, (axiom, rejected) in enumerate(decisions):  # each is a real embed + upsert
        cli.cmd_record(config, axiom=axiom, rejected=rejected, scope=["payments"], slug=f"payment-slug-{i}")
    capsys.readouterr()

    # Semantic recall is ranked and capped at the top matches.
    cli.cmd_surface(config, "payments architecture and money handling",
                    scope="payments", as_json=True)
    surfaced = json.loads(capsys.readouterr().out)["active_decisions"]
    assert len(surfaced) <= 5  # the semantic cap

    # Exhaustive enumeration returns ALL of them — nothing hidden below the cliff.
    cli.cmd_list(config, scope="payments", as_json=True)
    listed = json.loads(capsys.readouterr().out)
    assert listed["total"] == len(decisions) == 7
    assert {d["slug"] for d in listed["decisions"]} >= {d["slug"] for d in surfaced}
    assert listed["total"] > len(surfaced), "the whole point: list sees more than capped surface"


def test_cli_subprocess_list_decisions_json(tmp_path):
    """Real binary, real argv: the `list_decisions` MCP-name alias + `--json` emit
    the complete structured set after real records (the exhaustive CLI path)."""
    mitos_bin = shutil.which("mitos") or "mitos"
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

        records = [
            ("Services deploy as containers on Kubernetes",
             "Bare VMs rejected: manual scaling toil"),
            ("Config is injected via environment, never baked into images",
             "Image-baked config rejected: a rebuild per environment"),
            ("Secrets are fetched from the vault at runtime",
             "Committed .env rejected: leaks on a public mirror"),
        ]
        for i, (axiom, rejected) in enumerate(records):
            rec = subprocess.run(
                [mitos_bin, "record", axiom, "--rejected", rejected, "--scope", "infra", "--slug", f"slug-{i}"],
                cwd=ws, env=env, capture_output=True, text=True,
            )
            assert rec.returncode == 0, rec.stderr

        listed = subprocess.run(
            [mitos_bin, "list_decisions", "--scope", "infra", "--json"],
            cwd=ws, env=env, capture_output=True, text=True,
        )
        assert listed.returncode == 0, listed.stderr
        data = json.loads(listed.stdout)
        assert data["total"] == 3
        assert data["state"] == "active"
        assert {d["slug"] for d in data["decisions"]} == {
            "slug-0",
            "slug-1",
            "slug-2"
        }
    finally:
        _drop_collection(collection)


def test_adjacency_surfaces_related_decision(live_workspace):
    """Recording a decision surfaces its nearest existing live neighbour — the
    write-time adjacency guardrail, proven against REAL embeddings (③)."""
    ws, _ = live_workspace
    m = MitosSyncManager(MitosConfig(str(ws)))
    m.record_decision_entry(
        axiom="Payments settle through Stripe as the single payment processor",
        rejected_paths="Adyen rejected: heavier integration for our volume",
        scope=["payments"], slug="stripe-single-psp",
    )
    res = m.record_decision_entry(
        axiom="Stripe webhooks are the source of truth for charge status",
        rejected_paths="Polling the Stripe API rejected: rate limits and latency",
        scope=["payments"], slug="stripe-webhooks-source-of-truth",
        # This test deliberately records a decision with a known live neighbour to
        # prove the post-commit `related` echo — acknowledge the P4 review so a high
        # real-embedding similarity doesn't pause the write (the echo still fires).
        acknowledge_neighbors=True,
    )
    assert res["status"] == "created"
    related_slugs = [r["slug"] for r in res.get("related", [])]
    if not related_slugs:
        # The adjacency echo is fail-silent (production swallows embed errors,
        # sync.py:1688); an empty echo under a spent embed quota is environmental,
        # not a code defect. Probe to distinguish: skip loudly on a 429, else let
        # the assert fire on a real regression.
        skip_if_embed_quota_exhausted(m.embed_provider)
    assert "stripe-single-psp" in related_slugs, related_slugs
    # The new decision never lists itself as its own neighbour.
    assert "stripe-webhooks-source-of-truth" not in related_slugs


def test_surface_brief_omits_rejected_paths_real_semantic(live_workspace, capsys):
    """brief drops rejected_paths on the REAL semantic surface path (not the fallback)."""
    ws, _ = live_workspace
    config = MitosConfig(str(ws))
    cli.cmd_record(
        config,
        axiom="The tutor fails fast on missing learner data",
        rejected="Graceful-degrade rejected: a silent wrong answer is worse than a loud error",
        scope=["personas"],
        slug="tutor-fails-fast",
    )
    capsys.readouterr()

    cli.cmd_surface(config, "how to handle missing data", scope="personas",
                    as_json=True, brief=True)
    brief = json.loads(capsys.readouterr().out)
    assert brief["active_decisions"], "semantic recall should find the decision"
    assert all("rejected_paths" not in d for d in brief["active_decisions"])
    assert brief["active_decisions"][0]["axiom"]  # axiom still present

    cli.cmd_surface(config, "how to handle missing data", scope="personas", as_json=True)
    full = json.loads(capsys.readouterr().out)
    assert any("rejected_paths" in d for d in full["active_decisions"])


def test_cli_subprocess_relation_flag_links_decisions(tmp_path):
    """Real binary, real argv: --depends-on links two decisions, edge lands in graph."""
    mitos_bin = shutil.which("mitos") or "mitos"
    skip_if_global_mitos_stale(mitos_bin)
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
        assert subprocess.run([mitos_bin, "init"], cwd=ws, env=env,
                              capture_output=True, text=True).returncode == 0
        a = subprocess.run(
            [mitos_bin, "record", "Adopt hexagonal architecture",
             "--rejected", "Layered rejected: leaks IO into the core", "--scope", "arch",
             "--slug", "hexagonal-arch"],
            cwd=ws, env=env, capture_output=True, text=True)
        assert a.returncode == 0, a.stderr
        b = subprocess.run(
            [mitos_bin, "record", "Adapters live at the edges",
             "--rejected", "Core importing IO rejected: violates the dependency rule",
             "--scope", "arch", "--slug", "adapters-at-edges", "--depends-on", "hexagonal-arch"],
            cwd=ws, env=env, capture_output=True, text=True)
        assert b.returncode == 0, b.stderr
        assert "Recorded decision" in b.stdout

        store = GraphStore(MitosConfig(str(ws)).db_path)
        # Both decisions committed via the real binary; both resolve as active.
        src = store.get_node_by_slug("adapters-at-edges")
        tgt = store.get_node_by_slug("hexagonal-arch")
        assert src is not None
        assert tgt is not None
        # `--depends-on` commits a non-kill depends_on edge as of V1b 2a (D→D,
        # same-kind): the flag threads through the real binary into the buffer AND
        # the reconciler commits the edge. Both endpoints stay active — a non-kill
        # edge retires nothing (unlike supersedes/corrects).
        with open(os.path.join(str(ws), "decisions.md"), encoding="utf-8") as f:
            buf = f.read()
        assert "**Depends-On:** hexagonal-arch" in buf
        edges = store.get_edges()
        assert len(edges) == 1
        assert edges[0]["edge_type"] == "depends_on"
        assert edges[0]["source_id"] == src["id"]
        assert edges[0]["target_id"] == tgt["id"]
        assert store.get_node_state(src["id"]) == "active"
        assert store.get_node_state(tgt["id"]) == "active"
    finally:
        _drop_collection(collection)


def test_cli_subprocess_record_stdin_then_surface(tmp_path):
    """Real binary, real argv, real stdin pipe, real services — the AX fixes
    (MCP-name alias + --rejected-file stdin + surface recall) end-to-end."""
    mitos_bin = shutil.which("mitos") or "mitos"
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
             "--slug", "camilas-tutor", "--scope", "personas"],
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
