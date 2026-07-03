"""Phases 5a + 5b — the sync-time Conflict surface (`_run_and_surface_conflict` + the hook).

5a wires the shipped `run_conflict_check` facade into `mitos sync`'s per-entry review:
before the accept prompt, a decision entry is judged against its undeclared close
neighbours in the active graph, and a high-confidence not-tenable finding is surfaced at
the prompt. The sensor is advisory — it prints, applies no verb, writes nothing to the
graph, and NEVER blocks a commit.

5b gives that surface its memory and its failure manners: a healthy JUDGED result persists
one `judgment_batches` row + N `conflict_checks` rows to the sibling `.mitos/telemetry.sqlite`
(threaded by a run-scoped `sync_run_id`); a typed `Unavailable` prints a loud
`[Conflict sensor unavailable]` notice and trips a per-run aggregate breaker so a single
downstream outage costs one penalty, not N; and neither persistence nor degradation ever
blocks a commit. The 5b cases read rows back through a real connection to
`config.telemetry_path` (an emission-reaches-the-sink e2e against real SQLite, not a
MagicMock roundtrip).

Discipline (scout brief / PATTERNS live-test rule): deterministic + keyless + **no SDK**.
The `_build_conflict_judge` seam is monkeypatched to return a `_RecordingJudge`, bypassing
the Anthropic client + `ANTHROPIC_API_KEY` entirely; the two-seam gotcha (⚠-1) means the
facade's real `gather_candidates` still reads `manager.embed_provider` / `manager.vector_store`,
so those are replaced with hand-rolled fakes too. The graph store is a real temp
`GraphStore` seeded via `commit_parsed_entry` (never embeds), so commit assertions key on
graph state. The telemetry store is REAL (the temp `.mitos/telemetry.sqlite`). Judge JSON is
always routed through `_execution([...])` (a bare Mock return makes 3a's
`parse_judgment_response` fail → a spurious `Unavailable`).
"""

import datetime as _datetime
import os
import shutil
import sqlite3
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

import mitos
from mitos.config import MitosConfig
from mitos.conflict import (
    CONFLICT_CANDIDATE_SOURCE,
    CONFLICT_PROMPT_VERSION,
    Candidate,
    ConflictCheckResult,
    ConflictUnavailableReason,
    JudgeInput,
    JudgedPair,
    Judgment,
    JudgmentExecution,
    Unavailable,
)
from mitos.errors import DatabaseError
from mitos.identity import embedding_text
from mitos.parser import ParsedEntry
from mitos.store import open_connection
from mitos.sync import MitosSyncManager


# --------------------------------------------------------------------------- #
# Hand-rolled fakes (the 2a / test_conflict_facade idiom — synchronous, no SDK)
# --------------------------------------------------------------------------- #

class _FakeEmbed:
    """Returns a fixed document-space vector; records every call."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, bool]] = []

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        self.calls.append((text, is_query))
        return [0.1, 0.2, 0.3]


class _CountingEmbed:
    """Models the real provider's content-hash cache to pin no-double-embed (D3).

    Counts cache MISSES (the real "API" calls) keyed on the exact text. Because both the
    conflict-check gather and the post-commit `_best_effort_embed` route the committed
    entry through the SAME `identity.embedding_text` (byte-identical text), the second is a
    cache hit — so the committed entry's text records exactly one miss across the run.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, List[float]] = {}
        self.calls: List[Tuple[str, bool]] = []
        self.miss_texts: List[str] = []

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        self.calls.append((text, is_query))
        if text not in self._cache:
            self.miss_texts.append(text)
            self._cache[text] = [0.1, 0.2, 0.3]
        return self._cache[text]


class _FakeVector:
    """Returns canned `(slug, score)` matches for the over-fetch query; no-op upsert."""

    def __init__(self, matches: Optional[List[Dict[str, Any]]] = None) -> None:
        self._matches = matches if matches is not None else []
        self.queries = 0

    def query(self, vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        self.queries += 1
        return list(self._matches)

    def upsert(self, node_id: str, vector: List[float], payload: Dict[str, Any]) -> None:
        pass


class _RecordingJudge:
    """A fake `judge` returning a canned value; records whether/how often it was called."""

    def __init__(self, ret: Any) -> None:
        self._ret = ret
        self.called = False
        self.calls = 0
        self.last_prompt: Any = None

    def __call__(self, prompt: Any) -> Any:
        self.called = True
        self.calls += 1
        self.last_prompt = prompt
        return self._ret


class _SequenceJudge:
    """A fake `judge` returning a canned value PER call — one per entry in a multi-entry run.

    `_RecordingJudge` returns one fixed value forever; a multi-entry sync that persists a row
    per entry needs a DISTINCT `batch_id` per judged batch (it is the `judgment_batches` PK),
    so this returns `rets[i]` on the i-th call.
    """

    def __init__(self, rets: List[Any]) -> None:
        self._rets = list(rets)
        self.calls = 0
        self.last_prompt: Any = None

    def __call__(self, prompt: Any) -> Any:
        self.last_prompt = prompt
        ret = self._rets[self.calls]
        self.calls += 1
        return ret


def _execution(verdicts: List[tuple], *, batch_id: str = "batch-fixed-id") -> JudgmentExecution:
    """Builds a JudgmentExecution whose `raw_text` is the judge JSON for `verdicts`.

    `verdicts`: list of `(slug, tenable_together, confidence, rationale)`. The set/count
    must match the screened batch (3a's parse realigns by casefolded slug).
    """
    import json

    raw = json.dumps(
        [
            {
                "slug": slug,
                "rationale": rationale,
                "tenable_together": tenable,
                "confidence": confidence,
            }
            for (slug, tenable, confidence, rationale) in verdicts
        ]
    )
    return JudgmentExecution(
        raw_text=raw,
        batch_id=batch_id,
        model_alias="SONNET",
        token_input=100,
        token_output=40,
        token_cache_read=0,
        token_cache_creation=0,
        elapsed_ms=12,
    )


def _match(slug: str, score: float) -> Dict[str, Any]:
    return {"slug": slug, "score": score}


# --------------------------------------------------------------------------- #
# Environment + harness (mirrors tests/test_sync.py's sync_env)
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No reachable service; GEMINI key present (the sync gate), no ANTHROPIC key.

    `_perform_sync_internal` returns early without GEMINI_API_KEY (sync.py:552), so it is
    set. ANTHROPIC_API_KEY is unset because the judge is injected via the monkeypatched
    `_build_conflict_judge` seam — no real client is ever built (keyless-deterministic).
    """
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture
def env() -> Iterator[Tuple[MitosConfig, MitosSyncManager, str]]:
    """A complete temp sync environment with a real (empty) on-disk graph store."""
    tmpdir = tempfile.mkdtemp()
    config = MitosConfig(tmpdir)
    config.db_path = os.path.join(tmpdir, ".mitos", "graph.sqlite")
    config.decisions_file = os.path.join(tmpdir, "decisions.md")
    config.archive_dir = os.path.join(tmpdir, "decisions", "archive")
    os.makedirs(os.path.join(tmpdir, ".mitos"), exist_ok=True)
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write(
            "# Decisions\n"
            "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->\n"
        )
    manager = MitosSyncManager(config)
    yield config, manager, tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


def _wire_fakes(
    manager: MitosSyncManager,
    *,
    judge: Any,
    matches: Optional[List[Dict[str, Any]]] = None,
    embed: Optional[Any] = None,
) -> _FakeVector:
    """Wires the two seams (⚠-1): the injected judge AND the embed/vector substrate.

    Monkeypatching only `_build_conflict_judge` is not enough — the facade's real
    `gather_candidates` reads `manager.embed_provider` / `manager.vector_store` directly.
    """
    manager.embed_provider = embed if embed is not None else _FakeEmbed()  # type: ignore[assignment]
    vector = _FakeVector(matches)
    manager.vector_store = vector  # type: ignore[assignment]
    manager._build_conflict_judge = lambda: judge  # type: ignore[assignment]
    return vector


def _append_decision(
    config: MitosConfig,
    slug: str,
    axiom: str,
    *,
    rejected: str = "Rejected the obvious alternative.",
    scope: Optional[str] = "api",
    mechanisms: str = "python",
    date: str = "2026-06-01",
) -> None:
    """Appends a well-formed decision entry to the decisions.md write buffer.

    ``scope=None`` omits the ``**Scope:**`` line entirely → the parsed entry is global
    (``scope == []``), the fixture the 5b MI-9 proposal_scope-IS-NULL case needs.
    """
    scope_line = f"**Scope:** {scope}\n" if scope is not None else ""
    block = (
        f"## {date} — {slug} — {slug.replace('-', ' ').title()}\n"
        f"**Decided:** {axiom}\n"
        f"**Rejected:** {rejected}\n"
        f"**Mechanisms:** {mechanisms}\n"
        f"{scope_line}"
    )
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(block + "\n")


def _seed_active(
    manager: MitosSyncManager,
    slug: str,
    axiom: str,
    *,
    scope: Optional[List[str]] = None,
    rejected: str = "Rejected the obvious alternative.",
) -> None:
    """Commits one live decision directly (keyless — commit_parsed_entry never embeds)."""
    entry = ParsedEntry("decision", slug, 1, 10)
    entry.axiom = axiom
    entry.rejected_paths = rejected
    entry.scope = scope if scope is not None else ["api"]
    manager.store.commit_parsed_entry(entry)


def _read_conflict_rows(config: MitosConfig) -> List[Dict[str, Any]]:
    """Reads back every `conflict_checks` row via a real connection (the store is write-only).

    The 5b e2e read-side: open `config.telemetry_path` read-only and `SELECT *` so each row
    is a name-keyed dict (the writer is fire-and-forget; the test is the reader). Returns []
    if the telemetry DB was never created (no judged batch fired).
    """
    if not os.path.exists(config.telemetry_path):
        return []
    conn = open_connection(config.telemetry_path, read_only=True)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM conflict_checks")
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def _read_batch_rows(config: MitosConfig) -> List[Dict[str, Any]]:
    """Reads back every `judgment_batches` row via a real read-only connection."""
    if not os.path.exists(config.telemetry_path):
        return []
    conn = open_connection(config.telemetry_path, read_only=True)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM judgment_batches")
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Case 1 — surfaced finding + accept-anyway commits
# --------------------------------------------------------------------------- #

def test_surfaced_finding_prints_and_accept_commits(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """not-tenable @ 0.9 → the finding block prints; accepting still commits the entry."""
    config, manager, _ = env
    _seed_active(manager, "endpoints-auth", "All API endpoints require authentication.",
                 scope=["api", "security"])
    judge = _RecordingJudge(
        _execution([("endpoints-auth", False, 0.9,
                     "The proposal exempts /health from auth; the active decision admits "
                     "no unauthenticated endpoint.")])
    )
    _wire_fakes(manager, judge=judge, matches=[_match("endpoints-auth", 0.9)])

    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")

    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    out = capsys.readouterr().out
    assert "[Conflict]" in out
    assert "endpoints-auth" in out
    assert "The proposal exempts /health from auth" in out  # the rationale (the "why")
    assert "All API endpoints require authentication." in out  # the candidate axiom
    # Accept-anyway committed the proposal verbatim, unlinked.
    assert manager.store.get_node_by_slug("health-public") is not None
    assert judge.calls == 1


def test_modifier_stamp_survives_into_printed_notice(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """An amended-but-active candidate's `amended_by` stamp rides into the printed block.

    2b pins the stamp at the payload level; 5a confirms it survives into the surface's
    plain-text notice (the "amended axioms read as live" trap — the finding must not read
    as the candidate's final word). `amends` is not a kill-edge, so the candidate stays
    computed-active and still surfaces.
    """
    config, manager, _ = env
    _seed_active(manager, "endpoints-auth", "All API endpoints require authentication.")
    # A later decision amends the candidate → it carries an `amended_by` stamp, still active.
    amender = ParsedEntry("decision", "auth-tokens", 1, 10)
    amender.axiom = "Auth uses bearer tokens."
    amender.rejected_paths = "Rejected session cookies."
    amender.scope = ["api"]
    amender.amends = ["endpoints-auth"]
    manager.store.commit_parsed_entry(amender)

    judge = _RecordingJudge(
        _execution([("endpoints-auth", False, 0.9, "Health is exempted; auth admits none.")])
    )
    _wire_fakes(manager, judge=judge, matches=[_match("endpoints-auth", 0.9)])

    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    out = capsys.readouterr().out
    assert "[Conflict]" in out
    assert "amended by: auth-tokens" in out  # the stamp survived into the surface notice


# --------------------------------------------------------------------------- #
# Case 2 — tenable ⇒ silent, commits
# --------------------------------------------------------------------------- #

def test_tenable_verdict_is_silent_and_commits(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """A tenable verdict surfaces nothing; the entry commits (P9 quiet success)."""
    config, manager, _ = env
    _seed_active(manager, "endpoints-auth", "All API endpoints require authentication.")
    judge = _RecordingJudge(
        _execution([("endpoints-auth", True, 0.9, "A scoped carve-out coexists fine.")])
    )
    _wire_fakes(manager, judge=judge, matches=[_match("endpoints-auth", 0.9)])

    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")

    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    out = capsys.readouterr().out
    assert "[Conflict]" not in out
    assert manager.store.get_node_by_slug("health-public") is not None
    assert judge.calls == 1  # the judge ran; the gate simply didn't surface it


# --------------------------------------------------------------------------- #
# Case 3 — clean-empty ⇒ silent, judge never called
# --------------------------------------------------------------------------- #

def test_clean_empty_is_silent_and_judge_not_called(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """No candidate clears the floor → nothing printed, the judge is never invoked."""
    config, manager, _ = env
    judge = _RecordingJudge(_execution([]))  # would be a parse-empty if ever called
    _wire_fakes(manager, judge=judge, matches=[])  # empty over-fetch → clean-empty

    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")

    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    out = capsys.readouterr().out
    assert "[Conflict]" not in out
    assert judge.calls == 0
    assert manager.store.get_node_by_slug("health-public") is not None


# --------------------------------------------------------------------------- #
# Case 4 — Unavailable never blocks the commit
# --------------------------------------------------------------------------- #

def test_unavailable_prints_loud_notice_and_never_blocks_commit(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """A judge Unavailable now prints the loud 5b notice (naming 'judgment'); commit proceeds.

    Supersedes 5a's silent case: 5b turns the swallowed typed `Unavailable` into a loud
    `[Conflict sensor unavailable]` notice + a breaker trip, while the commit still lands and
    no `conflict_checks` row is persisted for a degradation (fail-open, CONF-D10).
    """
    config, manager, _ = env
    _seed_active(manager, "endpoints-auth", "All API endpoints require authentication.")
    judge = _RecordingJudge(
        Unavailable(reason=ConflictUnavailableReason.JUDGMENT_TIMEOUT, detail="timed out")
    )
    _wire_fakes(manager, judge=judge, matches=[_match("endpoints-auth", 0.9)])

    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")

    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    out = capsys.readouterr().out
    assert "[Conflict sensor unavailable]" in out  # 5b: the loud degradation notice
    assert "judgment" in out.lower()  # names WHICH subsystem went dark (JUDGMENT_TIMEOUT)
    assert "[Conflict]" not in out  # not a finding — a degradation
    # Commit proceeds; no row persisted for a degradation.
    assert manager.store.get_node_by_slug("health-public") is not None
    assert _read_conflict_rows(config) == []


def test_graph_fault_inside_check_never_blocks_commit(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """A raise past the facade (2a D4) is caught defensively; the commit proceeds."""
    config, manager, _ = env
    judge = _RecordingJudge(_execution([]))

    class _BoomVector(_FakeVector):
        def query(self, vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
            raise RuntimeError("qdrant/graph exploded mid-check")

    manager.embed_provider = _FakeEmbed()  # type: ignore[assignment]
    manager.vector_store = _BoomVector()  # type: ignore[assignment]
    manager._build_conflict_judge = lambda: judge  # type: ignore[assignment]

    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")

    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    err = capsys.readouterr().err
    assert "Conflict check failed" in err  # logged to stderr, not stdout
    assert manager.store.get_node_by_slug("health-public") is not None


# --------------------------------------------------------------------------- #
# Case 5 — --yes skips the check entirely
# --------------------------------------------------------------------------- #

def test_auto_accept_skips_check_entirely(
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """`--yes` (auto_accept) never calls `_run_and_surface_conflict`; the entry commits."""
    config, manager, _ = env
    spy = MagicMock()
    manager._run_and_surface_conflict = spy  # type: ignore[assignment]
    # Even the builder should be untouched under --yes (the gate skips it).
    build_spy = MagicMock(return_value=_RecordingJudge(_execution([])))
    manager._build_conflict_judge = build_spy  # type: ignore[assignment]

    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")

    manager.perform_sync(auto_accept=True)

    assert spy.call_count == 0
    assert build_spy.call_count == 0
    assert manager.store.get_node_by_slug("health-public") is not None


# --------------------------------------------------------------------------- #
# Case 6 — toggle off skips the check
# --------------------------------------------------------------------------- #

def test_toggle_off_skips_check(
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """`conflict_check_on_sync = False` → the check never runs, no judge is built."""
    config, manager, _ = env
    manager.config.conflict_check_on_sync = False
    spy = MagicMock()
    manager._run_and_surface_conflict = spy  # type: ignore[assignment]
    build_spy = MagicMock(return_value=_RecordingJudge(_execution([])))
    manager._build_conflict_judge = build_spy  # type: ignore[assignment]

    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")

    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    assert spy.call_count == 0
    assert build_spy.call_count == 0
    assert manager.store.get_node_by_slug("health-public") is not None


# --------------------------------------------------------------------------- #
# Case 7 — kind gate: open questions are never checked
# --------------------------------------------------------------------------- #

def test_open_question_entry_is_never_checked(
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """An open-question entry is outside the decision branch → the hook never fires."""
    config, manager, tmpdir = env
    config.questions_file = os.path.join(tmpdir, "questions.md")
    spy = MagicMock()
    manager._run_and_surface_conflict = spy  # type: ignore[assignment]
    manager.embed_provider = _FakeEmbed()  # type: ignore[assignment]
    manager.vector_store = _FakeVector()  # type: ignore[assignment]
    manager._build_conflict_judge = lambda: _RecordingJudge(_execution([]))  # type: ignore[assignment]

    with open(config.questions_file, "w", encoding="utf-8") as f:
        f.write(
            "# Open Questions\n"
            "<!-- BEGIN ENTRIES -->\n"
            "## 2026-06-01 — auth-scope — Auth scope open question\n"
            "**Questions:** Should health checks bypass auth?\n\n"
        )

    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    assert spy.call_count == 0  # the OQ branch has no conflict check by design


# --------------------------------------------------------------------------- #
# Case 8 — activation disclosure fires exactly once
# --------------------------------------------------------------------------- #

def test_disclosure_fires_once_and_creates_sentinel(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """First gate-passing entry prints the disclosure + writes the sentinel; second is silent."""
    config, manager, _ = env
    manager.embed_provider = _FakeEmbed()  # type: ignore[assignment]
    manager.vector_store = _FakeVector(matches=[])  # clean-empty; disclosure fires regardless
    manager._build_conflict_judge = lambda: _RecordingJudge(_execution([]))  # type: ignore[assignment]
    sentinel = os.path.join(config.mitos_dir, ".conflict_disclosed")

    _append_decision(config, "first-decision", "The first decision axiom.")
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)
    first = capsys.readouterr().out
    assert "[Conflict sensor active]" in first
    assert "conflict_check_on_sync = false" in first
    assert os.path.exists(sentinel)

    _append_decision(config, "second-decision", "The second decision axiom.")
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)
    second = capsys.readouterr().out
    assert "[Conflict sensor active]" not in second  # sentinel present → no re-fire


# --------------------------------------------------------------------------- #
# Case 9 — no double-embed (D3): committed entry embeds exactly once
# --------------------------------------------------------------------------- #

def test_committed_entry_embeds_exactly_once(
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """Gather + post-commit embed hit the same content-hash text → one real embed call."""
    config, manager, _ = env
    proposal_axiom = "The /health endpoint is publicly accessible."
    proposal_text = embedding_text({"kind": "decision", "axiom": proposal_axiom})
    embed = _CountingEmbed()
    _wire_fakes(manager, judge=_RecordingJudge(_execution([])), matches=[], embed=embed)

    _append_decision(config, "health-public", proposal_axiom)
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    # The proposal's text was requested twice (gather + post-commit _best_effort_embed)…
    requested = [t for t, _q in embed.calls]
    assert requested.count(proposal_text) == 2
    # …but only ONE was a real embed call (the second is a content-hash cache hit).
    assert embed.miss_texts.count(proposal_text) == 1
    assert manager.store.get_node_by_slug("health-public") is not None


# --------------------------------------------------------------------------- #
# Case 10 — availability gate: _build_conflict_judge returns None when unavailable
# --------------------------------------------------------------------------- #

def test_build_conflict_judge_none_without_anthropic_key(
    env: Tuple[MitosConfig, MitosSyncManager, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Components present but no ANTHROPIC_API_KEY → None (sensor skipped, no error)."""
    _, manager, _ = env
    manager.embed_provider = _FakeEmbed()  # type: ignore[assignment]
    manager.vector_store = _FakeVector()  # type: ignore[assignment]
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert manager._build_conflict_judge() is None


def test_build_conflict_judge_none_when_component_down(
    env: Tuple[MitosConfig, MitosSyncManager, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing embed/vector component → None even with an ANTHROPIC key set."""
    _, manager, _ = env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    manager.embed_provider = None  # type: ignore[assignment]
    manager.vector_store = _FakeVector()  # type: ignore[assignment]
    assert manager._build_conflict_judge() is None

    manager.embed_provider = _FakeEmbed()  # type: ignore[assignment]
    manager.vector_store = None  # type: ignore[assignment]
    assert manager._build_conflict_judge() is None


def test_missing_component_makes_sync_a_noop_without_error(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """With a real (unpatched) builder and no key, the loop skips the check cleanly."""
    config, manager, _ = env
    # Real _build_conflict_judge: no ANTHROPIC key (offline fixture) → None → skipped.
    manager.embed_provider = _FakeEmbed()  # type: ignore[assignment]
    manager.vector_store = _FakeVector(matches=[_match("x", 0.9)])  # type: ignore[assignment]

    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    out = capsys.readouterr().out
    assert "[Conflict]" not in out
    assert "[Conflict sensor active]" not in out  # never disclosed — never fired
    assert manager.store.get_node_by_slug("health-public") is not None


# =========================================================================== #
# Phase 5b — telemetry persistence + degradation disposition + aggregate breaker
# =========================================================================== #

# --------------------------------------------------------------------------- #
# 5b Case 1 — a judged-surfaced row persists and reads back verbatim
# --------------------------------------------------------------------------- #

def test_judged_surfaced_row_persists_and_reads_back(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """not-tenable @ 0.9 → one conflict_checks row (surfaced=1) + one judgment_batches row.

    The emission-reaches-the-sink e2e: the row is read back through a real connection to the
    temp telemetry.sqlite (PLANNING_NOTES), not asserted against a MagicMock.
    """
    config, manager, _ = env
    _seed_active(manager, "endpoints-auth", "All API endpoints require authentication.",
                 scope=["api", "security"])
    judge = _RecordingJudge(
        _execution([("endpoints-auth", False, 0.9, "Health is exempted; auth admits none.")])
    )
    _wire_fakes(manager, judge=judge, matches=[_match("endpoints-auth", 0.9)])

    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    rows = _read_conflict_rows(config)
    assert len(rows) == 1
    row = rows[0]
    assert row["tenable"] == 0
    assert row["surfaced"] == 1
    assert abs(row["confidence"] - 0.9) < 1e-9
    assert row["judged_axiom"] == "The /health endpoint is publicly accessible."
    assert row["candidate_slug"] == "endpoints-auth"
    # candidate_hash is the M2 content hash, not the slug — joins to the committed node.
    node = manager.store.get_node_by_slug("endpoints-auth")
    assert row["candidate_hash"] == node["id"]
    assert "Health is exempted" in row["rationale"]
    assert row["model_alias"] == "SONNET"
    assert row["prompt_version"] == CONFLICT_PROMPT_VERSION
    assert row["candidate_source"] == CONFLICT_CANDIDATE_SOURCE == "embedding_topk"
    # mitos_version is read programmatically — never a hardcoded literal.
    assert row["mitos_version"] == mitos.__version__
    assert row["sync_run_id"]  # non-null
    # created_at is a real, parseable UTC ISO-8601 stamp (never CURRENT_TIMESTAMP, MI-10).
    assert _datetime.datetime.fromisoformat(row["created_at"])

    batches = _read_batch_rows(config)
    assert len(batches) == 1
    assert batches[0]["batch_id"] == "batch-fixed-id"
    assert batches[0]["token_input"] == 100
    assert batches[0]["elapsed_ms"] == 12

    # The entry committed — persistence rode alongside the commit, never blocked it.
    assert manager.store.get_node_by_slug("health-public") is not None


# --------------------------------------------------------------------------- #
# 5b Case 2 — all judged pairs persist, not only the surfaced one
# --------------------------------------------------------------------------- #

def test_all_judged_pairs_persist_not_only_surfaced(
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """Two candidates (one tenable, one not) → two rows (surfaced 1 and 0), one shared batch.

    The tenable pair is the silent-but-judged negative label the future classifier needs
    (CONF-D8) — persistence must NOT be gated on `surfaced`.
    """
    config, manager, _ = env
    _seed_active(manager, "endpoints-auth", "All API endpoints require authentication.")
    _seed_active(manager, "rate-limit", "All API traffic is rate-limited per client.")
    judge = _RecordingJudge(
        _execution([
            ("endpoints-auth", False, 0.9, "Health exemption contradicts blanket auth."),
            ("rate-limit", True, 0.9, "A public endpoint coexists with rate limiting."),
        ])
    )
    _wire_fakes(
        manager,
        judge=judge,
        matches=[_match("endpoints-auth", 0.9), _match("rate-limit", 0.88)],
    )

    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    rows = _read_conflict_rows(config)
    assert len(rows) == 2
    by_slug = {r["candidate_slug"]: r for r in rows}
    assert by_slug["endpoints-auth"]["surfaced"] == 1
    assert by_slug["rate-limit"]["surfaced"] == 0  # judged but not surfaced — still stored
    assert by_slug["rate-limit"]["tenable"] == 1
    # One batch, shared batch_id across both rows (the intra-corpus join key).
    assert len({r["batch_id"] for r in rows}) == 1
    assert len(_read_batch_rows(config)) == 1


# --------------------------------------------------------------------------- #
# 5b Case 3 — clean-empty persists nothing
# --------------------------------------------------------------------------- #

def test_clean_empty_persists_no_row(
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """No candidate clears the floor → execution is None → zero rows, entry commits."""
    config, manager, _ = env
    judge = _RecordingJudge(_execution([]))
    _wire_fakes(manager, judge=judge, matches=[])  # empty over-fetch → clean-empty

    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    assert _read_conflict_rows(config) == []
    assert _read_batch_rows(config) == []
    assert judge.calls == 0
    assert manager.store.get_node_by_slug("health-public") is not None


# --------------------------------------------------------------------------- #
# 5b Case 4 — fed-context serialization + MI-9 (empty → NULL, "" verbatim)
# --------------------------------------------------------------------------- #

def test_fed_context_serialization_and_mi9_nulls_e2e(
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """A global proposal + a scoped and a global candidate exercise MI-9 through the surface.

    proposal_scope IS NULL (global proposal); a scoped candidate → "api, security"; a global
    candidate → candidate_scope IS NULL. proposal_rejected_paths is required on a decision, so
    it rides verbatim (the "" → NULL branch is proven at the unit level below).
    """
    config, manager, _ = env
    _seed_active(manager, "scoped-cand", "Scoped candidate axiom.", scope=["api", "security"])
    _seed_active(manager, "global-cand", "Global candidate axiom.", scope=[])
    judge = _RecordingJudge(
        _execution([
            ("scoped-cand", False, 0.9, "Scoped tension."),
            ("global-cand", True, 0.9, "Global coexists."),
        ])
    )
    _wire_fakes(
        manager,
        judge=judge,
        matches=[_match("scoped-cand", 0.9), _match("global-cand", 0.88)],
    )

    # Global proposal: no Scope line → scope == [] → proposal_scope IS NULL.
    _append_decision(
        config, "health-public", "The /health endpoint is publicly accessible.",
        rejected="Rejected always-open access.", scope=None,
    )
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    rows = _read_conflict_rows(config)
    assert len(rows) == 2
    by_slug = {r["candidate_slug"]: r for r in rows}
    # Proposal side: global → scope NULL; rejected required → stored verbatim.
    assert by_slug["scoped-cand"]["proposal_scope"] is None
    assert by_slug["scoped-cand"]["proposal_rejected_paths"] == "Rejected always-open access."
    # Candidate side: scoped joins as "api, security"; global → NULL (MI-9).
    assert by_slug["scoped-cand"]["candidate_scope"] == "api, security"
    assert by_slug["global-cand"]["candidate_scope"] is None


def test_persist_mi9_empty_maps_to_null_unit(
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """Direct `_persist_conflict_batch`: empty proposal rejected_paths/scope → NULL.

    The one MI-9 branch the buffer path cannot reach (a decision requires `**Rejected:**`, so
    a synced proposal never has an empty rejected_paths): drive the mapper directly to prove
    an empty proposal `rejected_paths`/`scope` and an empty candidate `scope` become NULL,
    while the NOT-NULL candidate `rejected_paths` stores the degenerate "" verbatim.
    """
    config, manager, _ = env
    proposal_input = JudgeInput(axiom="A global axiom.", rejected_paths="", scope=[])
    candidate = Candidate(slug="cand", score=0.9, node={"id": "cand-hash"}, state="active")
    candidate_input = JudgeInput(axiom="Cand axiom.", rejected_paths="", scope=[])
    judgment = Judgment(slug="cand", rationale="why", tenable_together=False, confidence=0.9)
    pair = JudgedPair(
        candidate=candidate, candidate_input=candidate_input,
        judgment=judgment, surfaced=True,
    )
    result = ConflictCheckResult(
        proposal_input=proposal_input,
        proposed_hash_if_any="proposal-hash",
        findings=[],
        judged_pairs=[pair],
        execution=_execution([("cand", False, 0.9, "why")]),
    )
    run = manager._new_conflict_run()
    manager._persist_conflict_batch(result, run)

    rows = _read_conflict_rows(config)
    assert len(rows) == 1
    row = rows[0]
    assert row["proposal_rejected_paths"] is None  # "" → NULL (MI-9)
    assert row["proposal_scope"] is None            # [] → NULL (MI-9)
    assert row["candidate_scope"] is None           # [] → NULL (MI-9)
    assert row["candidate_rejected_paths"] == ""     # NOT NULL — degenerate "" verbatim
    assert row["proposed_hash_if_any"] == "proposal-hash"
    assert row["candidate_hash"] == "cand-hash"


# --------------------------------------------------------------------------- #
# 5b Case 6 — aggregate breaker: one penalty, not N (integration, two entries)
# --------------------------------------------------------------------------- #

def test_breaker_trips_once_and_skips_later_entries(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """First entry degrades → notice prints ONCE; the second neither gathers nor judges.

    The write-then-read breaker (shared `_ConflictSyncRun` built once per run): a single
    downstream outage costs one penalty for the run, not N. Both entries still commit.
    """
    config, manager, _ = env
    _seed_active(manager, "endpoints-auth", "All API endpoints require authentication.")
    judge = _RecordingJudge(
        Unavailable(reason=ConflictUnavailableReason.JUDGMENT_TIMEOUT, detail="timed out")
    )
    vector = _wire_fakes(manager, judge=judge, matches=[_match("endpoints-auth", 0.9)])

    _append_decision(config, "first-entry", "The first decision axiom.")
    _append_decision(config, "second-entry", "The second decision axiom.")
    with patch("builtins.input", side_effect=["a", "a"]):
        manager.perform_sync(auto_accept=False)

    out = capsys.readouterr().out
    assert out.count("[Conflict sensor unavailable]") == 1  # one penalty, not two
    # The tripping entry gathered + judged once; the second entry did neither.
    assert vector.queries == 1
    assert judge.calls == 1
    # Both entries committed — a degradation never blocks a commit.
    assert manager.store.get_node_by_slug("first-entry") is not None
    assert manager.store.get_node_by_slug("second-entry") is not None
    # No row persisted for a degradation.
    assert _read_conflict_rows(config) == []


# --------------------------------------------------------------------------- #
# 5b Case 7 — sync_run_id threads every row of one run (P16)
# --------------------------------------------------------------------------- #

def test_sync_run_id_threads_a_run(
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """Two judged entries in one sync → both rows share one non-null sync_run_id."""
    config, manager, _ = env
    _seed_active(manager, "endpoints-auth", "All API endpoints require authentication.")
    judge = _SequenceJudge([
        _execution([("endpoints-auth", False, 0.9, "why one")], batch_id="batch-1"),
        _execution([("endpoints-auth", False, 0.9, "why two")], batch_id="batch-2"),
    ])
    _wire_fakes(manager, judge=judge, matches=[_match("endpoints-auth", 0.9)])

    _append_decision(config, "first-entry", "The first decision axiom.")
    _append_decision(config, "second-entry", "The second decision axiom.")
    with patch("builtins.input", side_effect=["a", "a"]):
        manager.perform_sync(auto_accept=False)

    rows = _read_conflict_rows(config)
    assert len(rows) == 2
    run_ids = {r["sync_run_id"] for r in rows}
    assert len(run_ids) == 1           # one thread of truth across the run
    assert all(r["sync_run_id"] for r in rows)  # non-null
    # Distinct batches (distinct PKs) both landed.
    assert {r["batch_id"] for r in rows} == {"batch-1", "batch-2"}


# --------------------------------------------------------------------------- #
# 5b Case 8 — a telemetry write failure never aborts the sync
# --------------------------------------------------------------------------- #

def test_telemetry_write_failure_never_aborts_sync(
    env: Tuple[MitosConfig, MitosSyncManager, str],
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising `record_judged_batch` → the finding still prints, a stderr warning, commit lands."""
    config, manager, _ = env
    _seed_active(manager, "endpoints-auth", "All API endpoints require authentication.")
    judge = _RecordingJudge(
        _execution([("endpoints-auth", False, 0.9, "Health is exempted; auth admits none.")])
    )
    _wire_fakes(manager, judge=judge, matches=[_match("endpoints-auth", 0.9)])

    def _boom(self: Any, *args: Any, **kwargs: Any) -> None:
        raise DatabaseError("telemetry write boom")

    monkeypatch.setattr("mitos.telemetry.TelemetryStore.record_judged_batch", _boom)

    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    captured = capsys.readouterr()
    assert "[Conflict]" in captured.out  # the finding still surfaced (surfacing ⊥ persist)
    assert "Could not persist conflict telemetry" in captured.err  # best-effort warning
    assert manager.store.get_node_by_slug("health-public") is not None  # commit landed
    assert _read_conflict_rows(config) == []  # the write rolled back — no partial row
