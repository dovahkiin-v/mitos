"""Shared fakes + fixtures for the Conflict-sensor sync suites (5a/5b + 6a).

Extracted in Phase 6a (DoD-3): the provoked-failure suite
(``tests/test_conflict_faults.py``) and the surface suite
(``tests/test_conflict_sync.py``) both drive the real ``perform_sync`` loop against
these hand-rolled, keyless-deterministic fakes, so the harness lives here once and both
files import it — reuse, not duplicate (PLANNING_NOTES / P17).

NOT a ``test_``-prefixed module → pytest does not collect it. Under pytest's default
``prepend`` import mode ``tests/`` is on ``sys.path``, so it imports as
``from _conflict_helpers import ...`` (never ``tests._conflict_helpers`` — there is no
package ``__init__.py``). The ``offline``/``env`` fixtures are ordinary importable
fixtures; importing them into a test module registers them there (``offline`` stays
autouse wherever it is imported).

Discipline (PATTERNS live-test rule): deterministic + keyless + no SDK. The graph store
is a real temp ``GraphStore`` seeded via ``commit_parsed_entry`` (never embeds), the
telemetry store is the real temp ``.mitos/telemetry.sqlite``, and no ``ANTHROPIC_API_KEY``
is present — the judge is injected through the monkeypatched ``_build_conflict_judge`` seam.
"""

import os
import shutil
import sqlite3
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest

from mitos.config import MitosConfig
from mitos.conflict import JudgmentExecution
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


class _KeyedEmbed:
    """Returns a DISTINCT vector per exact text — per-proposal neighbourhoods (2b).

    The shared ``_FakeEmbed`` returns one fixed vector, so every query looks identical
    to the vector fake and per-proposal neighbourhoods are impossible; the corpus sweep
    (``tests/test_check_sweep.py``) needs each swept axiom to map to its own canned
    KNN result. ``raises`` maps a text to the exception its embed call should raise —
    the per-node degradation seam (§9-6). An unknown text fails loud (``KeyError``):
    a fixture gap, never a silent default vector.
    """

    def __init__(
        self,
        vectors: Dict[str, List[float]],
        *,
        raises: Optional[Dict[str, BaseException]] = None,
    ) -> None:
        self._vectors = vectors
        self._raises = raises if raises is not None else {}
        self.calls: List[Tuple[str, bool]] = []

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        self.calls.append((text, is_query))
        if text in self._raises:
            raise self._raises[text]
        return list(self._vectors[text])


class _KeyedVector:
    """Returns canned matches keyed on the exact query VECTOR; records every query.

    The ``_KeyedEmbed`` counterpart: sentinel vector in → that node's canned matches
    out (keyed on ``tuple(vector)`` — lists aren't hashable). ``queried`` is the
    per-query log the 2b laziness test reads: an unconsumed sweep node must leave no
    entry here. ``raises`` (also vector-keyed) is the vector-substrate fault seam —
    folded in now so 2c's breaker tests don't grow a third private fork.
    """

    def __init__(
        self,
        matches: Dict[Tuple[float, ...], List[Dict[str, Any]]],
        *,
        raises: Optional[Dict[Tuple[float, ...], BaseException]] = None,
    ) -> None:
        self._matches = matches
        self._raises = raises if raises is not None else {}
        self.queried: List[Tuple[float, ...]] = []

    def query(self, vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        key = tuple(vector)
        self.queried.append(key)
        if key in self._raises:
            raise self._raises[key]
        return list(self._matches[key])

    def upsert(self, node_id: str, vector: List[float], payload: Dict[str, Any]) -> None:
        pass


def _keyed_substrate(
    neighbourhoods: Dict[str, List[Dict[str, Any]]],
    *,
    embed_raises: Optional[Dict[str, BaseException]] = None,
    vector_raises: Optional[Dict[str, BaseException]] = None,
) -> Tuple[_KeyedEmbed, _KeyedVector]:
    """Builds a coherent ``(_KeyedEmbed, _KeyedVector)`` pair from ``{axiom_text: matches}``.

    Mints one sentinel vector per text (insertion-order indexed) and wires the two
    fakes so sweeping the node with that axiom sees exactly its canned matches. Note
    the text key is what gather actually embeds — ``embedding_text(...)`` = NFC+strip
    of the axiom, so plain ASCII fixture axioms key verbatim. Both ``raises`` maps are
    TEXT-keyed (the vector fake's is translated onto its sentinel vector here), so one
    builder covers per-node embed faults and vector faults alike. A ``vector_raises``
    key absent from ``neighbourhoods`` has no sentinel to ride — that is a fixture gap
    and fails loud here, never a silently un-armed fault.
    """
    if vector_raises:
        unknown = set(vector_raises) - set(neighbourhoods)
        if unknown:
            raise ValueError(
                f"vector_raises keys missing from neighbourhoods: {sorted(unknown)}"
            )
    vectors: Dict[str, List[float]] = {}
    matches: Dict[Tuple[float, ...], List[Dict[str, Any]]] = {}
    translated: Dict[Tuple[float, ...], BaseException] = {}
    for index, (text, canned) in enumerate(neighbourhoods.items()):
        sentinel = [float(index + 1), 0.5]
        vectors[text] = sentinel
        matches[tuple(sentinel)] = canned
        if vector_raises and text in vector_raises:
            translated[tuple(sentinel)] = vector_raises[text]
    return (
        _KeyedEmbed(vectors, raises=embed_raises),
        _KeyedVector(matches, raises=translated),
    )


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
    so this returns `rets[i]` on the i-th call. An entry that IS a `BaseException` instance
    is RAISED when reached instead of returned — the 2c simulated-kill seam (a judge dying
    mid-run, distinct from a typed `Unavailable` return). Calling past the end raises
    `IndexError` — a free "judge called more often than planned" tripwire.
    """

    def __init__(self, rets: List[Any]) -> None:
        self._rets = list(rets)
        self.calls = 0
        self.last_prompt: Any = None

    def __call__(self, prompt: Any) -> Any:
        self.last_prompt = prompt
        ret = self._rets[self.calls]
        self.calls += 1
        if isinstance(ret, BaseException):
            raise ret
        return ret


def _execution(
    verdicts: List[tuple],
    *,
    batch_id: str = "batch-fixed-id",
    model_alias: str = "SONNET",
) -> JudgmentExecution:
    """Builds a JudgmentExecution whose `raw_text` is the judge JSON for `verdicts`.

    `verdicts`: list of `(slug, tenable_together, confidence, rationale)`. The set/count
    must match the screened batch (3a's parse realigns by casefolded slug).
    `model_alias` overrides the alias the execution carries — the 1a defensive-resolution
    case needs one `get_model_id` rejects (e.g. "CLAUDE_SONNET").
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
        model_alias=model_alias,
        token_input=100,
        token_output=40,
        token_cache_read=0,
        token_cache_creation=0,
        elapsed_ms=12,
    )


def _match(slug: str, score: float) -> Dict[str, Any]:
    return {"slug": slug, "score": score}


def _drain_outbox(store: Any) -> None:
    """Drains the pending-embeddings Outbox — the healthy post-sync fixture state.

    Every ``commit_parsed_entry`` unconditionally enqueues its node (MI-12), so a
    keyless test corpus carries a transient backlog by default — under the 2d stale
    probe that reads as an incomplete audit (exit 2), not a healthy run. Fixtures
    asserting exit/degradation/row outcomes state their backlog posture explicitly:
    healthy ones drain (this helper mirrors sync.py's post-upsert
    ``remove_pending_embedding``); stale ones seed deliberately.
    """
    for row in store.get_pending_embeddings():
        store.remove_pending_embedding(row["node_id"])


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
