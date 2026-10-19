"""T19: §1's "What it proves when complete", read off a real stdio session (9b).

Every earlier proof of this vision was taken from inside: an imported function, a
phase's own fixture, an in-process tool call. This module stands where the vision's
reader stands — an agent on the far end of an MCP pipe — against a workspace in the
state the release handoff leaves: built by the product's own verbs, its graveyard
re-seeded and its rejection index built by one ``reconcile``.

**The precondition** runs every verb through the launcher's embed-only mode
(``tests/_stdio_fake_providers.py``, ``MITOS_FAKE_PROVIDERS=embed``): ``init`` and
every ``record`` with ``QDRANT_URL`` on a dead port (the graph commits; the decision
outbox and the rejection backlog queue; no collection exists — an older build's
partly indexed workspace), then ``reconcile --json`` with Qdrant up, then ``status
--json``. No row here writes SQLite or Qdrant directly.

**Session 1** (Qdrant up) walks §1: R1, one ``surface_decisions`` call carrying the
live window and its top cluster, a rejection match with its owner's reason, a retired
decision two kill-hops from its live end, the edge, the band's evidence and a stamp
label; R2, a re-tread draft paused by its tombstone alone; R3, ``exists`` on a
word-for-word re-record of the retired decision; R4, the field-scoped lookup; R5, the
acknowledged re-send. **Session 2** (Qdrant dead): R6, a degraded read naming what it
did not compute, with a lexical rejection match; R6b, the lookup answering lexically.

The embedder is a deterministic bag of words, never a model, and it ignores
``is_query``, so the query-space floor (rejection) and the document-space floor
(tombstone) compare the same vectors here. **This walk proves the assembled surface,
not the spaces**: that a real query clears a real floor is Layer C's claim and the
dense set's (W2, W6), and 9b's recorded real-provider read of the measurement copy.
Every fixture score is computed in the row from the shipped constants, so a text that
drifts reds with its reason.

Gated like the other live modules: the live brake and a reachable Qdrant. It spends
nothing — the child's ``GEMINI_API_KEY`` is a placeholder the fake never reads (``status``
needs a key present to call the workspace ready). The workspace's basename starts
``tmp``, so its two collections fall inside the suite sweeper's prefix; the row drops
both itself. Serial only, alone (CLAUDE.md, the live tier).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import requests

from _stdio_fake_providers import (
    EMBED_BOUND_MARKER,
    EMBED_MODE,
    IDENTITY_MARKER,
    MODE_ENV,
    UNIMPLEMENTED_MARKER,
    cosine,
    fake_vector,
)
from live_helpers import live_tests_disabled
from mcp_harness import harness_env, mitos_server
from mitos import cli, stamp_labels
from mitos.config import MitosConfig
from mitos.identity import embedding_text
from mitos.models import EMBEDDING_DIM
from mitos.recall import (DEGRADED_NOT_COMPUTED, EDGE_KIND_MID_CLUSTER,
                          LOOKUP_DEGRADED_NOT_COMPUTED, LOOKUP_FIELDS, RANKED_EDGE_MARGIN,
                          RANKED_TOP_CLUSTER_MARGIN, REJECTION_MATCH_FLOOR,
                          SURFACE_STRONG_THRESHOLD, SURFACE_WEAK_THRESHOLD, WindowEdge,
                          edge_clause, retired_lead_clause)
from mitos.rejections import enumerate_rejected_paths
from mitos.sync import _NEIGHBOR_REVIEW_THRESHOLD, _TOMBSTONE_FLOOR

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:7333")
DEAD_QDRANT_URL = "http://127.0.0.1:9"
PLACEHOLDER_KEY = "placeholder-not-a-key"
LAUNCHER = Path(__file__).resolve().parent / "_stdio_fake_providers.py"
REPO_PACKAGE = Path(cli.__file__).resolve().parent


def _qdrant_reachable() -> bool:
    try:
        return requests.get(f"{QDRANT_URL}/collections", timeout=2).ok
    except Exception:
        return False


_BRAKE_ON = live_tests_disabled()
HAS_QDRANT = (not _BRAKE_ON) and _qdrant_reachable()

if _BRAKE_ON:
    _SKIP_REASON = (
        "MITOS_NO_LIVE_TESTS is set — the live tier is switched off, so the stdio recall "
        "walk did not run and nothing pretended to check."
    )
else:
    _SKIP_REASON = (
        f"Qdrant is not reachable at {QDRANT_URL} — the stdio recall walk needs a real "
        "instance. Environmental, not a code defect; start it with `docker compose up -d` "
        "in the mitos repo."
    )

pytestmark = pytest.mark.skipif(not HAS_QDRANT, reason=_SKIP_REASON)

# --------------------------------------------------------------------------- #
# The corpus. Scores below are the fake's; each is asserted in the row.
# --------------------------------------------------------------------------- #

#: The claim an agent is about to propose.
CLAIM = ("billing ledger rows replicate through one append only stream into every "
         "regional reporting replica")

#: Three live decisions that tie against the claim (the top cluster, and a window of
#: two that ends mid-cluster on the third).
CLUSTER = {
    "ledger-shard-a": ("Billing ledger rows replicate through one append only stream into "
                       "the alpha cluster tonight"),
    "ledger-shard-b": ("Billing ledger rows replicate through one append only stream into "
                       "the beta warehouse weekly"),
    "ledger-shard-c": ("Billing ledger rows replicate through one append only stream into "
                       "the gamma archive monthly"),
}

#: The retired decision closest to the claim, two kill-hops from its live end:
#: RETIRED ← superseded by MIDDLE (itself retired) ← superseded by LIVE_END.
RETIRED = ("retired-regional-stream", "Billing ledger rows replicate through one append "
           "only stream into every regional replica nightly")
MIDDLE = ("payments-journal-broker", "Payments journal entries fan out over a message "
          "broker toward a central analytics lake")
LIVE_END = ("invoice-cold-export", "Invoice records are copied by a scheduled export job "
            "to a single cold storage bucket")

#: A live decision that already REJECTED the claim, for its own question.
OWNER = ("support-rota", "Customer support tickets are triaged by a rotating on call "
         "engineer each morning")
OWNER_REJECTED = (f"(1) {CLAIM[0].upper()}{CLAIM[1:]}: REJECTED — replicas lag.\n"
                  "(2) A shared inbox: REJECTED — nobody owns it.")

#: A draft that re-treads RETIRED and sits under the live floor against every live one.
DRAFT = ("re-tread-regional-stream", "Billing ledger rows replicate through one append "
         "only stream into every regional replica hourly")

LIMIT = 2


def _decision_vector(axiom: str) -> List[float]:
    return fake_vector(embedding_text({"kind": "decision", "axiom": axiom}), EMBEDDING_DIM)


def _score(left: List[float], axiom: str) -> float:
    return cosine(left, _decision_vector(axiom))


def _parts(field: str) -> List[str]:
    """The texts a rejection point embeds: each item's parts, marker dropped."""
    return [p.text for item in enumerate_rejected_paths(field).items for p in item.parts]


def _items(field: str) -> List[str]:
    """The texts a match shows: each item whole, ``item.text.rstrip()`` (the contract
    ``mitos/rejections.py`` states for a match's ``item``)."""
    return [item.text.rstrip() for item in enumerate_rejected_paths(field).items]


# --------------------------------------------------------------------------- #
# Scaffolding
# --------------------------------------------------------------------------- #


def _env(tmp_path: Path, qdrant_url: str) -> Dict[str, str]:
    """The declared child env: embed-only mode, a placeholder key, QDRANT_URL always
    named (``init`` writes the URL it saw into config.toml, so a child left to the
    default would read the dead one)."""
    return harness_env(tmp_path, extra={
        MODE_ENV: EMBED_MODE, "GEMINI_API_KEY": PLACEHOLDER_KEY, "QDRANT_URL": qdrant_url})


def _verb(*args: str, cwd: Path, env: Dict[str, str], ok=(0,)) -> subprocess.CompletedProcess:
    """Runs one verb through the launcher (this interpreter, this checkout)."""
    done = subprocess.run([sys.executable, str(LAUNCHER), *args], cwd=str(cwd), env=env,
                          capture_output=True, text=True, timeout=180)
    assert done.returncode in ok, (
        f"`mitos {' '.join(args)}` rc={done.returncode}\nstdout:\n{done.stdout}\n"
        f"stderr:\n{done.stderr}")
    _assert_embed_child(done.stderr, f"`mitos {args[0]}`")
    return done


def _assert_embed_child(stderr: str, step: str) -> None:
    """Non-vacuity: the embed-only bind ran in this interpreter over this checkout."""
    assert EMBED_BOUND_MARKER in stderr, f"{step}: the embed-only bind never ran:\n{stderr}"
    assert UNIMPLEMENTED_MARKER not in stderr, f"{step}: a fake was asked too much:\n{stderr}"
    line = next((ln for ln in stderr.splitlines() if ln.startswith(IDENTITY_MARKER)), None)
    assert line is not None, f"{step}: no identity line:\n{stderr}"
    fields = dict(re.findall(r"(\w+)=(\S+)", line))
    assert fields.get("mode") == EMBED_MODE, f"{step}: {line}"
    assert Path(fields.get("mitos", "")).resolve().parent == REPO_PACKAGE, (
        f"{step}: the child imported another mitos: {line}")
    assert fields.get("python") == sys.executable, f"{step}: another interpreter: {line}"


def _payload(result: Any, step: str) -> Dict[str, Any]:
    assert result.isError is False, f"{step}: tool call errored: {result.content}"
    return json.loads(result.content[0].text)


def _shell_tokens(text: str) -> List[str]:
    """`mitos <parser verb>` occurrences: the no-shell rule, copied from
    `test_mcp_session_walk._shell_tokens` (a test module is not imported)."""
    verbs = sorted(cli._build_parser()._subparsers._group_actions[0].choices,
                   key=len, reverse=True)
    pattern = r"\bmitos\s+(?:" + "|".join(re.escape(v) for v in verbs) + r")\b"
    return re.findall(pattern, text)


def _drop(name: str) -> None:
    try:
        requests.delete(f"{QDRANT_URL}/collections/{name}", timeout=10)
    except Exception:
        pass


@pytest.fixture
def workspace(tmp_path: Path):
    """``tmp_path / "tmpwalk"``: its collections derive ``mitos-tmpwalk-…``."""
    ws = tmp_path / "tmpwalk"
    ws.mkdir()
    config = MitosConfig(str(ws))
    names = (config.qdrant_collection, config.rejection_collection)
    assert all(name.startswith("mitos-tmp") for name in names), names
    for name in names:
        _drop(name)
    yield ws
    for name in names:
        _drop(name)


def _record(ws: Path, env: Dict[str, str], slug: str, axiom: str, rejected: str,
            **relations: str) -> Dict[str, Any]:
    flags = [arg for rel, target in relations.items() for arg in (f"--{rel}", target)]
    done = _verb("record", "-p", str(ws), "--json", axiom, "--rejected", rejected,
                 "--scope", "ledger", "--slug", slug, *flags, cwd=ws, env=env)
    return json.loads(done.stdout)


# --------------------------------------------------------------------------- #
# The walk
# --------------------------------------------------------------------------- #


def test_the_fixture_scores_sit_where_the_walk_needs_them() -> None:
    """The geometry every step relies on, from the shipped constants. Runs offline in
    effect (no I/O), but lives here beside the rows it documents."""
    q = fake_vector(CLAIM, EMBEDDING_DIM)
    cluster = [_score(q, ax) for ax in CLUSTER.values()]
    assert max(cluster) - min(cluster) < RANKED_TOP_CLUSTER_MARGIN       # one top cluster
    assert max(cluster) - min(cluster) < RANKED_EDGE_MARGIN              # mid-cluster edge
    assert SURFACE_WEAK_THRESHOLD <= max(cluster) < SURFACE_STRONG_THRESHOLD   # weak band
    assert _score(q, RETIRED[1]) > max(cluster)                          # the closest match
    for _slug, ax in (MIDDLE, LIVE_END, OWNER):
        assert _score(q, ax) < min(cluster) - RANKED_EDGE_MARGIN         # never in the window
    claim_item = _parts(OWNER_REJECTED)[0]
    assert cosine(q, fake_vector(claim_item, EMBEDDING_DIM)) >= REJECTION_MATCH_FLOOR
    assert all(cosine(q, fake_vector(p, EMBEDDING_DIM)) < REJECTION_MATCH_FLOOR
               for p in _parts(OWNER_REJECTED)[1:])
    d = _decision_vector(DRAFT[1])
    assert _score(d, RETIRED[1]) >= _TOMBSTONE_FLOOR                     # it re-treads
    assert _score(d, MIDDLE[1]) < _TOMBSTONE_FLOOR                       # only RETIRED
    for _slug, ax in (*CLUSTER.items(), LIVE_END, OWNER):
        assert _score(d, ax) < _NEIGHBOR_REVIEW_THRESHOLD                # pauses alone


@pytest.mark.asyncio
async def test_an_mcp_agent_learns_section_one_from_the_payload(workspace, tmp_path) -> None:
    ws, project = workspace, str(workspace)
    dead, live = _env(tmp_path, DEAD_QDRANT_URL), _env(tmp_path, QDRANT_URL)
    config = MitosConfig(project)

    # -- Precondition: a workspace an older build indexed partly, healed by reconcile --
    _verb("init", cwd=ws, env=dead)
    rejected = {slug: f"(1) A {slug.replace('-', ' ')} variant: REJECTED — out of scope."
                for slug in (RETIRED[0], MIDDLE[0], LIVE_END[0], *CLUSTER)}
    written = [
        _record(ws, dead, RETIRED[0], RETIRED[1], rejected[RETIRED[0]]),
        _record(ws, dead, MIDDLE[0], MIDDLE[1], rejected[MIDDLE[0]], supersedes=RETIRED[0]),
        _record(ws, dead, LIVE_END[0], LIVE_END[1], rejected[LIVE_END[0]],
                supersedes=MIDDLE[0]),
        *(_record(ws, dead, slug, ax, rejected[slug]) for slug, ax in CLUSTER.items()),
        _record(ws, dead, OWNER[0], OWNER[1], OWNER_REJECTED),
    ]
    for receipt in written:
        assert receipt.get("status") == "created", f"precondition: {receipt}"
        assert receipt.get("embedding") == "pending", f"precondition: no outbox row: {receipt}"
    n_decisions, n_retired = len(written), 2                    # RETIRED and MIDDLE
    n_items = sum(len(_parts(f)) for f in (*rejected.values(), OWNER_REJECTED))
    assert requests.get(f"{QDRANT_URL}/collections/{config.qdrant_collection}",
                        timeout=5).status_code == 404, "precondition: a collection exists"

    receipt = json.loads(_verb("reconcile", "-p", project, "--json", cwd=ws, env=live).stdout)
    assert receipt.get("expected") == n_decisions, f"reconcile: {receipt}"
    assert receipt.get("written") == n_decisions, f"reconcile: the graveyard was not seeded: {receipt}"
    rindex, rdrain = receipt.get("rejection_index") or {}, receipt.get("rejection_drain") or {}
    assert rindex.get("collection_present") is False, f"reconcile: not built from absent: {rindex}"
    assert rindex.get("expected_items") == n_items == rindex.get("missing_items"), rindex
    assert rindex.get("seed_stamped") is True, rindex
    assert rdrain.get("items_embedded") == n_items, f"reconcile: {rdrain}"

    status = json.loads(_verb("status", "--json", project, cwd=ws, env=live).stdout)
    checks = status.get("checks") or {}
    assert checks.get("missing_active_vectors") == 0, f"status: {checks}"
    assert checks.get("retired_decisions") == n_retired, f"status: {checks}"
    assert checks.get("missing_retired_vectors") == 0, f"status: {checks}"
    assert checks.get("retired_node_points") == n_retired, f"status: {checks}"
    assert checks.get("rejection_collection_exists") is True, f"status: {checks}"
    assert checks.get("rejection_expected_items") == n_items, f"status: {checks}"
    for key in ("missing_rejection_items", "stale_rejection_items", "orphan_rejection_items"):
        assert checks.get(key) == 0, f"status: {key}: {checks}"

    q = fake_vector(CLAIM, EMBEDDING_DIM)
    scores = {slug: _score(q, ax) for slug, ax in CLUSTER.items()}
    claim_part, claim_item = _parts(OWNER_REJECTED)[0], _items(OWNER_REJECTED)[0]
    texts: List[str] = []

    # -- Session 1: Qdrant up ---------------------------------------------------------
    async with mitos_server(cwd=tmp_path, env=live, args=(str(LAUNCHER),)) as server:
        async def call(step: str, tool: str, **arguments: Any) -> Dict[str, Any]:
            result = await server.session.call_tool(tool, {**arguments, "project": project})
            texts.append(result.content[0].text)
            out = _payload(result, step)
            assert out.get("workspace") == project, f"{step}: provenance: {out}"
            return out

        # R1 — one surface_decisions call carries every §1 element.
        r1 = await call("R1", "surface_decisions", query=CLAIM, limit=LIMIT)
        assert "degraded" not in r1, f"R1: {r1}"
        window = r1.get("active_decisions") or []
        assert len(window) == LIMIT and {h["slug"] for h in window} <= set(CLUSTER), f"R1: {window}"
        for hit in window:
            assert hit["score"] == pytest.approx(scores[hit["slug"]], abs=1e-4), f"R1: {hit}"
        evidence = r1.get("band_evidence") or {}
        assert evidence.get("top_cluster") == LIMIT, f"R1: top cluster: {evidence}"
        assert evidence.get("top_score") == pytest.approx(max(scores.values()), abs=1e-4), evidence
        assert evidence.get("edge") == EDGE_KIND_MID_CLUSTER, f"R1: edge: {evidence}"
        assert evidence.get("next_score") == pytest.approx(min(scores.values()), abs=1e-4), evidence
        assert r1.get("confidence") == "weak", f"R1: band: {r1.get('confidence')}"
        assert edge_clause(WindowEdge(EDGE_KIND_MID_CLUSTER, LIMIT, LIMIT),
                           surface="mcp") in r1.get("note", ""), f"R1: edge clause: {r1.get('note')}"

        matches = [m for m in r1.get("rejection_matches") or [] if m.get("slug") == OWNER[0]]
        assert len(matches) == 1, f"R1: rejection match: {r1.get('rejection_matches')}"
        (match,) = matches
        assert match["item"] == claim_item and match["ordinal"] == 1, f"R1: {match}"
        assert match["letter"] == "here" and match["axiom"] == OWNER[1], f"R1: {match}"
        assert match["item"].endswith("REJECTED — replicas lag."), f"R1: the why: {match}"
        assert match["score"] == pytest.approx(
            cosine(q, fake_vector(claim_part, EMBEDDING_DIM)), abs=1e-4), f"R1: {match}"

        part = {h["slug"]: h for h in r1.get("retired_partition") or []}
        handle = part.get(RETIRED[0])
        assert handle is not None, f"R1: retired partition: {r1.get('retired_partition')}"
        assert handle["superseded_by"] == [MIDDLE[0]], f"R1: {handle}"
        assert handle["live_end"] == [LIVE_END[0]], f"R1: two hops to the live end: {handle}"
        assert handle["score"] == pytest.approx(_score(q, RETIRED[1]), abs=1e-4), handle
        assert len(handle["id"]) == 64, f"R1: {handle}"
        # 6b's composer opens with the rejection lead, then the retired lead.
        assert retired_lead_clause(handle, surface="mcp") in r1.get("note", ""), (
            f"R1: the retired lead: {r1.get('note')}")
        assert r1.get("stamp_labels") == {LIVE_END[0]: stamp_labels.cut_label(LIVE_END[1])}, (
            f"R1: stamp labels: {r1.get('stamp_labels')}")
        shown = await call("R1", "show_node", ident=handle["id"])
        assert shown.get("slug") == RETIRED[0] and shown.get("id") == handle["id"], f"R1: {shown}"

        # R2 — a re-tread draft pauses on its tombstone alone.
        draft = dict(axiom=DRAFT[1], rejected_paths="(1) A per-region stream: REJECTED — "
                     "too many writers.", scope=["ledger"], slug=DRAFT[0])
        r2 = await call("R2", "record_decision", **draft)
        assert r2.get("status") == "needs_review", f"R2: {r2}"
        assert r2.get("neighbors") == [], f"R2: not alone: {r2.get('neighbors')}"
        (tomb,) = r2.get("tombstones") or [None]
        assert tomb and tomb["id"] == handle["id"] and tomb["axiom"] == RETIRED[1], f"R2: {tomb}"
        assert tomb["live_end"] == [LIVE_END[0]], f"R2: {tomb}"
        assert tomb["score"] == pytest.approx(
            _score(_decision_vector(DRAFT[1]), RETIRED[1]), abs=1e-4), f"R2: {tomb}"
        assert "successor_on_pause" not in tomb, f"R2: no neighbour holds its successor: {tomb}"
        assert r2.get("draft_digest"), f"R2: {r2}"

        # R3 — a word-for-word re-record of the retired decision answers `exists`.
        r3 = await call("R3", "record_decision", axiom=RETIRED[1], rejected_paths="Anything.",
                        scope=["ledger"], slug="retired-regional-stream-again")
        assert r3.get("status") == "exists" and r3.get("id") == handle["id"], f"R3: {r3}"
        assert r3.get("superseded_by") == [MIDDLE[0]], f"R3: {r3}"
        assert r3.get("live_end") == [LIVE_END[0]], f"R3: {r3}"
        unwritten = await call("R3", "show_node", ident="retired-regional-stream-again")
        assert unwritten.get("found") is False, f"R3: the no-op wrote: {unwritten}"

        # R4 — the field-scoped lookup answers from rejections alone.
        r4 = await call("R4", "query_decisions", query=CLAIM, field=LOOKUP_FIELDS[0])
        assert "degraded" not in r4 and "matches" not in r4, f"R4: {r4}"
        owners = [m["slug"] for m in r4.get("rejection_matches") or []]
        assert owners and owners[0] == OWNER[0] and owners.count(OWNER[0]) == 1, f"R4: {r4}"
        assert r4["rejection_matches"][0]["item"] == claim_item, f"R4: {r4}"
        r4b = await call("R4", "query_decisions", query=OWNER[0], field=LOOKUP_FIELDS[0])
        assert "slug" not in r4b and "matches" not in r4b, f"R4: a slug was dereferenced: {r4b}"

        # R5 — the acknowledged re-send lands and names the tombstone it went past.
        r5 = await call("R5", "record_decision", **draft, draft_digest=r2["draft_digest"],
                        acknowledge_neighbors=True)
        assert r5.get("status") == "created", f"R5: {r5}"
        acknowledged = r5.get("acknowledged_tombstones") or []
        assert [t.get("id") for t in acknowledged] == [handle["id"]], f"R5: {r5}"
        assert acknowledged[0].get("live_end") == [LIVE_END[0]], f"R5: {acknowledged}"

        for step, out in (("R1", r1), ("R2", r2), ("R3", r3), ("R4", r4), ("R5", r5)):
            assert "degraded" not in out, f"{step}: a healthy step degraded: {out}"
        _assert_embed_child(server.stderr_text(), "session 1")

    # -- Session 2: the same workspace, Qdrant dead -----------------------------------
    async with mitos_server(cwd=tmp_path, env=dead, args=(str(LAUNCHER),)) as server:
        async def call2(step: str, tool: str, **arguments: Any) -> Dict[str, Any]:
            result = await server.session.call_tool(tool, {**arguments, "project": project})
            texts.append(result.content[0].text)
            out = _payload(result, step)
            assert out.get("workspace") == project, f"{step}: provenance: {out}"
            return out

        # R6 — a degraded read names what it did not compute, and still finds the claim.
        r6 = await call2("R6", "surface_decisions", query=CLAIM, limit=LIMIT)
        assert r6.get("degraded") == "lexical", f"R6: {r6}"
        assert r6.get("not_computed") == list(DEGRADED_NOT_COMPUTED), f"R6: {r6}"
        assert "confidence" not in r6 and "band_evidence" not in r6, f"R6: {r6}"
        assert r6.get("rejection_section") == "lexical", f"R6: {r6}"
        lexical = [m for m in r6.get("rejection_matches") or [] if m.get("slug") == OWNER[0]]
        assert lexical and lexical[0]["item"] == claim_item, f"R6: {r6.get('rejection_matches')}"
        assert "score" not in lexical[0] and lexical[0].get("matched_terms"), f"R6: {lexical}"

        # R6b — the field-scoped lookup answers lexically and says why.
        r6b = await call2("R6b", "query_decisions", query=CLAIM, field=LOOKUP_FIELDS[0])
        assert r6b.get("degraded") == "lexical", f"R6b: {r6b}"
        assert r6b.get("not_computed") == list(LOOKUP_DEGRADED_NOT_COMPUTED), f"R6b: {r6b}"
        assert r6b.get("degraded_reason") and r6b["degraded_reason"] in r6b.get("note", ""), r6b
        assert (r6b.get("rejection_matches") or [{}])[0].get("item") == claim_item, f"R6b: {r6b}"
        _assert_embed_child(server.stderr_text(), "session 2")

    shell = [token for text in texts for token in _shell_tokens(text)]
    assert shell == [], f"a payload handed the agent a shell command: {shell}"
