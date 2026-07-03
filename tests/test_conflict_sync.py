"""Phase 5a — the sync-time Conflict surface (`_run_and_surface_conflict` + the hook).

5a wires the shipped `run_conflict_check` facade into `mitos sync`'s per-entry review:
before the accept prompt, a decision entry is judged against its undeclared close
neighbours in the active graph, and a high-confidence not-tenable finding is surfaced at
the prompt. The sensor is advisory — it prints, applies no verb, writes nothing, and NEVER
blocks a commit.

Discipline (scout brief / PATTERNS live-test rule): deterministic + keyless + **no SDK**.
The `_build_conflict_judge` seam is monkeypatched to return a `_RecordingJudge`, bypassing
the Anthropic client + `ANTHROPIC_API_KEY` entirely; the two-seam gotcha (⚠-1) means the
facade's real `gather_candidates` still reads `manager.embed_provider` / `manager.vector_store`,
so those are replaced with hand-rolled fakes too. The graph store is a real temp
`GraphStore` seeded via `commit_parsed_entry` (never embeds), so commit assertions key on
graph state. Judge JSON is always routed through `_execution([...])` (a bare Mock return
makes 3a's `parse_judgment_response` fail → a spurious `Unavailable`).
"""

import os
import shutil
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from mitos.config import MitosConfig
from mitos.conflict import (
    ConflictUnavailableReason,
    JudgmentExecution,
    Unavailable,
)
from mitos.identity import embedding_text
from mitos.parser import ParsedEntry
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
    scope: str = "api",
    mechanisms: str = "python",
    date: str = "2026-06-01",
) -> None:
    """Appends a well-formed decision entry to the decisions.md write buffer."""
    block = (
        f"## {date} — {slug} — {slug.replace('-', ' ').title()}\n"
        f"**Decided:** {axiom}\n"
        f"**Rejected:** {rejected}\n"
        f"**Mechanisms:** {mechanisms}\n"
        f"**Scope:** {scope}\n"
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

def test_unavailable_never_blocks_commit(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """A judge Unavailable is swallowed in 5a (silent); the entry still commits."""
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
    assert "[Conflict]" not in out  # 5b adds the loud notice; 5a is silent-but-safe
    assert manager.store.get_node_by_slug("health-public") is not None


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
