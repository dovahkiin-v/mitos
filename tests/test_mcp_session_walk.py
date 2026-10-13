"""T21: an MCP agent's §1 list, read off the payload of a real ``serve`` child.

The vision's promise is that an agent working only through MCP can tell, from the
payload alone, what its write did and what its read withheld. The rows here sit
where that agent sits: a client, a pipe, and a server process built from this
checkout. Every step names the §1 key it reads; a red names its step.

The child is ``tests/_stdio_fake_providers.py``, which binds deterministic fakes
at the two provider constructors the MCP path reaches and then enters the same
``cli.main()`` serve path as ``-m mitos.cli serve``. Everything else is the
branch's code. That keeps the rows offline (constraint 15): no key in the test
process, and ``QDRANT_URL`` points at a dead port, so a binding that missed fails
fast instead of reaching a live Qdrant.

Both construct sites swallow a failure and degrade to lexical recall, so a missed
binding would read as a plausible answer. Non-vacuity is asserted, not assumed:
the launcher's bind marker is on stderr, no fake reported an unimplemented call,
the writes come back ``embedding: indexed`` (the ``mitos.sync`` binding), the
pause fires (the gather ran), and no healthy read is degraded (the
``mitos.mcp_server`` binding).

The child's environment declares a placeholder ``ANTHROPIC_API_KEY``. It is not a
leak and nothing spends it: the standing check notice shows only in a workspace
with a judge key, and no MCP tool constructs an Anthropic client (only
``perform_sync`` builds the judge).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from _stdio_fake_providers import (
    BOUND_MARKER,
    FAULT_BODY_SENTINEL,
    FAULT_ENV,
    UNIMPLEMENTED_MARKER,
)
from mcp_harness import harness_env, mitos_server
from mitos import cli
from mitos.config import MitosConfig
from mitos.identity import mechanism_canonical_norm
from mitos.provider_cause import PROVIDER_CAUSES, provider_cause_phrase
from mitos.recall import WindowLever, limit_clause
from mitos.telemetry import (
    ATTEMPT_COULD_NOT_COMPLETE,
    AttemptOutcome,
    AttemptStart,
    CheckRunRow,
    TelemetryStore,
)

LAUNCHER = Path(__file__).resolve().parent / "_stdio_fake_providers.py"

#: A port nothing listens on (as in `test_mcp_stdio_harness`).
DEAD_QDRANT_URL = "http://127.0.0.1:9"

#: Makes the workspace keyed for the notice's show rule. Never spent: see above.
PLACEHOLDER_JUDGE_KEY = "placeholder-not-a-key"

#: Three decisions. A and C share almost every word (the fake scores them ~0.93,
#: over the 0.80 pause floor); B shares none with either, so no write but C's
#: first send pauses.
AXIOM_A = ("Session tokens are stored hashed with a per-user salt and are never "
           "written to any log.")
AXIOM_B = "The billing export runs nightly as one batch job against the read replica."
AXIOM_C = ("Session tokens are stored hashed with a per-user salt and are never "
           "written to a log file.")

#: Scores ~0.73 against A on the fake: the `weak` band.
WEAK_QUERY = "session tokens stored hashed with salt never written to log"

MECHANISMS_A = ["SQLite!", "Str Casefold", "argon2"]
CONTEXT_A = "Found while auditing the login flow's storage."
INVALIDATES_IF_A = "The session store moves to a managed vault that hashes for us."


# --------------------------------------------------------------------------- #
# Scaffolding
# --------------------------------------------------------------------------- #


def _run_mitos(*args, cwd, env):
    done = subprocess.run(
        [sys.executable, "-m", "mitos.cli", *args],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, (
        f"`mitos {' '.join(args)}` failed (rc={done.returncode})\n"
        f"stdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    )


def _plant_could_not_complete(ws: Path) -> None:
    """Writes a `could_not_complete` last attempt through the real writers.

    The argument construction mirrors `_seed_attempt` in `tests/test_commit_gate.py`
    and `_check_run_row` in `tests/test_check_coverage.py`; they are copied, not
    imported, because importing a test module drags its collection along.
    """
    tel = TelemetryStore(MitosConfig(str(ws)).telemetry_path)
    tel.record_attempt_start(AttemptStart(
        attempt_id="att-1", started_at="2026-09-21T00:00:00.000000+00:00",
        fingerprint="f" * 64))
    row = CheckRunRow(
        run_id="run-1", mode="corpus",
        started_at="2026-09-21T00:00:00+00:00", ended_at="2026-09-21T00:01:00+00:00",
        exit_code=1, nodes_swept=2, pairs_judged_fresh=0, pairs_reused=0,
        findings_new=0, findings_known=0, coverage_exclusions=0,
        degraded_reason=None, mitos_version="test",
    )
    tel.record_run_end(row, coverage=None, attempt=AttemptOutcome(
        attempt_id="att-1", state=ATTEMPT_COULD_NOT_COMPLETE, run_id="run-1",
        outcome_at="2026-09-21T00:01:00.000000+00:00", degradation_tokens=(),
        new_pairs=(("a" * 64, "b" * 64),), findings_known=0))


def _scaffold(tmp_path: Path) -> tuple:
    """An initialized workspace with the attempt planted, and a separate launch dir."""
    ws = tmp_path / "ws"
    ws.mkdir()
    _run_mitos("init", cwd=ws, env=harness_env(tmp_path, extra={"QDRANT_URL": DEAD_QDRANT_URL}))
    _plant_could_not_complete(ws)
    launch = tmp_path / "launch"
    launch.mkdir()
    return ws, launch


def _child_env(tmp_path: Path, **extra: str) -> dict:
    return harness_env(tmp_path, extra={
        "QDRANT_URL": DEAD_QDRANT_URL,
        "ANTHROPIC_API_KEY": PLACEHOLDER_JUDGE_KEY,
        **extra,
    })


def _payload(result, step: str) -> dict:
    assert result.isError is False, f"{step}: tool call errored: {result.content}"
    return json.loads(result.content[0].text)


def _shell_tokens(text: str) -> list:
    """`mitos <parser verb>` occurrences — 8a1's no-shell rule, mirrored.

    The same pattern `tests/test_recipe_sweep.py` asks with: the product's name in
    prose is not a command; a `mitos` followed by one of the parser's verbs is.
    """
    verbs = sorted(cli._build_parser()._subparsers._group_actions[0].choices,
                   key=len, reverse=True)
    pattern = r"\bmitos\s+(?:" + "|".join(re.escape(v) for v in verbs) + r")\b"
    return re.findall(pattern, text)


def test_the_shell_token_rule_has_teeth() -> None:
    """In-row control for the mirror: a command is found, the product's name is not."""
    assert _shell_tokens("run `mitos sync -p 'x'` first") == ["mitos sync"]
    assert _shell_tokens("so mitos never shortens one") == []


def _assert_bound(stderr: str) -> None:
    assert BOUND_MARKER in stderr, f"the launcher never bound the fakes:\n{stderr}"
    assert UNIMPLEMENTED_MARKER not in stderr, (
        f"production called a method the fakes lack:\n{stderr}")


# --------------------------------------------------------------------------- #
# S1–S9: one session, the receipts and the reads
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_an_mcp_agent_reads_every_section_one_key_from_the_payload(tmp_path):
    """S1–S9 of the §1 walk, in the order an agent meets them, on one session."""
    ws, launch = _scaffold(tmp_path)
    project = str(ws)
    texts = []  # every payload, for the no-shell scan at the end

    async with mitos_server(cwd=launch, env=_child_env(tmp_path),
                            args=(str(LAUNCHER),)) as server:
        async def call(step, tool, **arguments):
            result = await server.session.call_tool(tool, {**arguments, "project": project})
            texts.append(result.content[0].text)
            return result

        record_c = dict(
            axiom=AXIOM_C, rejected_paths="Storing the raw token for easier debugging.",
            scope=["auth"], slug="session-token-log-exclusion",
        )

        # S1 — the receipt: audit debt, mechanisms as authored and as folded, the notice.
        s1 = _payload(await call("S1", "record_decision",
                                 axiom=AXIOM_A, rejected_paths="Plaintext tokens in the table.",
                                 scope=["auth"], slug="session-tokens-hashed",
                                 mechanisms=MECHANISMS_A, context=CONTEXT_A), "S1")
        assert s1.get("status") == "created", f"S1: {s1}"
        assert s1.get("embedding") == "indexed", f"S1: the sync binding missed: {s1}"
        assert s1.get("audit_debt") == {"uncovered": 1, "total": 1, "excluded": 0}, f"S1: {s1}"
        assert s1.get("mechanisms") == MECHANISMS_A, f"S1: {s1}"
        assert s1.get("mechanisms_normalized") == {
            m: mechanism_canonical_norm(m) for m in MECHANISMS_A
            if mechanism_canonical_norm(m) != m
        } != {}, f"S1: {s1}"
        notice = s1.get("check_notice") or {}
        assert notice.get("state") == ATTEMPT_COULD_NOT_COMPLETE, f"S1: {s1}"
        assert "degraded" not in s1 and "neighbor_review_unavailable" not in s1, f"S1: {s1}"

        # S1b — the amend that gives S8 an invalidates_if to read back.
        s1b = _payload(await call("S1b", "amend_commentary", slug="session-tokens-hashed",
                                  invalidates_if=INVALIDATES_IF_A), "S1b")
        assert "error" not in s1b, f"S1b: {s1b}"

        # S2 — an acting relation's edge names its target's state, stamps and axiom.
        s2 = _payload(await call("S2", "record_decision",
                                 axiom=AXIOM_B, rejected_paths="Streaming each invoice live.",
                                 scope=["billing"], slug="billing-export-nightly",
                                 amends="session-tokens-hashed"), "S2")
        assert s2.get("status") == "created", f"S2: {s2}"
        edge = (s2.get("edges_created") or [{}])[0]
        assert edge.get("target_state") == "active", f"S2: {edge}"
        assert isinstance(edge.get("target_stamps"), dict), f"S2: {edge}"
        assert edge.get("target_axiom") == AXIOM_A, f"S2: {edge}"

        # S3 — a near-duplicate pauses, names its neighbour and hands back a digest.
        s3 = _payload(await call("S3", "record_decision", **record_c), "S3")
        assert s3.get("status") == "needs_review", f"S3: the gather never ran: {s3}"
        assert s3.get("draft_digest"), f"S3: {s3}"
        neighbors = [n.get("slug") for n in s3.get("neighbors", [])]
        assert neighbors == ["session-tokens-hashed"], f"S3: {s3}"

        # S4 — a drifted re-send is refused by field, and writes nothing.
        s4 = _payload(await call("S4", "record_decision",
                                 **{**record_c, "rejected_paths": "Something else entirely."},
                                 draft_digest=s3["draft_digest"],
                                 acknowledge_neighbors=True), "S4")
        assert s4.get("code") == "draft_drifted", f"S4: {s4}"
        assert s4.get("drifted_fields") == ["rejected_paths"], f"S4: {s4}"
        s4_read = _payload(await call("S4", "show_node", ident=record_c["slug"]), "S4")
        assert s4_read.get("found") is False, f"S4: the refused re-send wrote: {s4_read}"

        # S5 — the clean re-send lands and names the neighbour it acknowledged.
        s5 = _payload(await call("S5", "record_decision", **record_c,
                                 draft_digest=s3["draft_digest"],
                                 acknowledge_neighbors=True), "S5")
        assert s5.get("status") == "created", f"S5: {s5}"
        assert s5.get("acknowledged_neighbors") == ["session-tokens-hashed"], f"S5: {s5}"

        # S6 — full_top keeps the top hit's rejected paths and counts the rest withheld.
        s6 = _payload(await call("S6", "surface_decisions", query="session tokens",
                                 full_top=1), "S6")
        assert "degraded" not in s6, f"S6: the mcp_server binding missed: {s6}"
        hits = s6.get("active_decisions", [])
        assert len(hits) == 3, f"S6: {s6}"
        assert "rejected_paths" in hits[0], f"S6: {hits[0]}"
        assert all("rejected_paths" not in hit for hit in hits[1:]), f"S6: {hits}"
        assert s6.get("rejected_paths_withheld") == 2, f"S6: {s6}"
        notice = s6.get("check_notice") or {}
        assert notice.get("state") == ATTEMPT_COULD_NOT_COMPLETE, f"S6: {s6}"

        # S7 — a weak ranking that filled its window names limit as the lever.
        s7 = _payload(await call("S7", "surface_decisions", query=WEAK_QUERY, limit=2), "S7")
        assert "degraded" not in s7, f"S7: {s7}"
        assert s7.get("confidence") == "weak", f"S7: the fixture left the band: {s7}"
        assert limit_clause(WindowLever(limit=2, held=2), surface="mcp") in s7.get("note", ""), (
            f"S7: {s7.get('note')}")

        # S8 — both by-handle reads carry context, invalidates_if and folded mechanisms.
        folded = [mechanism_canonical_norm(m) for m in MECHANISMS_A]
        s8_show = _payload(await call("S8", "show_node", ident="session-tokens-hashed"), "S8")
        s8_query = _payload(await call("S8", "query_decisions", query="session-tokens-hashed"),
                            "S8")
        assert s8_query.get("slug") == "session-tokens-hashed", (
            f"S8: not the slug branch: {s8_query}")
        for read, label in ((s8_show, "show_node"), (s8_query, "query_decisions")):
            assert read.get("context") == CONTEXT_A, f"S8 {label}: {read}"
            assert read.get("invalidates_if") == INVALIDATES_IF_A, f"S8 {label}: {read}"
            assert sorted(read.get("mechanisms") or []) == sorted(folded), f"S8 {label}: {read}"

        # S9 — an undeclared argument is refused in mitos's voice, on the same session.
        s9 = await server.session.call_tool("list_decisions",
                                            {"project": project, "bogus_param": 1})
        texts.append(s9.content[0].text)
        assert s9.isError is True, f"S9: {s9.content}"
        assert s9.content[0].text.startswith("list_decisions was not run: "), f"S9: {s9.content}"
        assert "`bogus_param` is not an argument of list_decisions." in s9.content[0].text, (
            f"S9: {s9.content}")

        stderr = server.stderr_text()

    _assert_bound(stderr)
    assert not [t for text in texts for t in _shell_tokens(text)], (
        "a walk payload hands the agent a shell command")


# --------------------------------------------------------------------------- #
# S10: a second session, the provider faulted at launch
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("cause", PROVIDER_CAUSES)
async def test_a_degraded_read_names_its_class_and_never_the_provider_body(tmp_path, cause):
    """S10 — each provider class reaches the envelope as its phrase, body withheld."""
    ws, launch = _scaffold(tmp_path)
    async with mitos_server(cwd=launch, env=_child_env(tmp_path, **{FAULT_ENV: cause}),
                            args=(str(LAUNCHER),)) as server:
        result = await server.session.call_tool("surface_decisions", {
            "query": f"a query no cache has seen for {cause}", "project": str(ws)})
        stderr = server.stderr_text()

    _assert_bound(stderr)
    text = result.content[0].text
    payload = _payload(result, "S10")
    assert payload.get("degraded") == "lexical", f"S10: {payload}"
    assert payload.get("degraded_reason") == provider_cause_phrase(cause), f"S10: {payload}"
    assert FAULT_BODY_SENTINEL not in text, f"S10: the provider body reached the payload: {text}"
    assert _shell_tokens(text) == [], f"S10: {text}"
