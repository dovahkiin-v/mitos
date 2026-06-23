"""Deterministic, keyless regression gate for ``tests/live_helpers.py`` (Phase r2).

The live-suite robustness logic (embed-quota 429 → loud skip; stale-global-binary
version-lag → loud skip) is otherwise only ever exercised on Vinga's box under the
exact un-reproducible quota/version conditions it tames. P10 ("Regression Ironclad:
every fix carries a fixture") and P16 require it be verifiable in keyless CI. All
pure cores are deterministic; the IO wrappers are driven with fakes — no live keys,
no Qdrant, no LLM, no real ``mitos`` binary. This file runs in the ``-m 'not
packaging'`` gate.
"""

import subprocess

import pytest
from _pytest.outcomes import Skipped

from mitos.errors import EmbeddingError, SynthesisError

import live_helpers
from live_helpers import (
    EMBED_QUOTA_SKIP_REASON,
    ENRICHMENT_QUOTA_SKIP_REASON,
    _is_embed_quota_exhausted,
    _is_enrichment_quota_exhausted,
    _is_stale,
    _parse_semver,
    skip_if_embed_quota_exhausted,
    skip_if_enrichment_quota_exhausted,
    skip_if_global_mitos_stale,
    skip_on_embed_quota,
)


# A realistic wrapped genai 429 message — the EmbeddingError raise site wraps the
# genai error verbatim into ``f"Gemini embedding API call failed: {str(e)}"``.
_QUOTA_MSG = (
    "Gemini embedding API call failed: 429 RESOURCE_EXHAUSTED. "
    "{'error': {'code': 429, 'message': 'Quota exceeded for metric "
    "embed_content_free_tier_requests, limit 1000', 'status': 'RESOURCE_EXHAUSTED'}}"
)


class _FakeProvider:
    """A tiny stand-in for ``GeminiEmbeddingProvider`` (no ``unittest.mock`` needed).

    ``get_embedding`` either raises a supplied exception or returns a fixed vector,
    recording whether it was called so a no-skip path can be distinguished from a
    short-circuit.
    """

    def __init__(self, raises: BaseException = None, vector=None):
        self._raises = raises
        self._vector = vector if vector is not None else [0.1, 0.2, 0.3]
        self.called = False

    def get_embedding(self, text: str, is_query: bool = False):
        self.called = True
        if self._raises is not None:
            raise self._raises
        return self._vector


# ---------------------------------------------------------------------------
# _is_embed_quota_exhausted — the narrow signature matcher
# ---------------------------------------------------------------------------

def test_is_embed_quota_exhausted_matches_full_429_payload():
    assert _is_embed_quota_exhausted(EmbeddingError(_QUOTA_MSG)) is True


def test_is_embed_quota_exhausted_matches_resource_exhausted_only():
    assert _is_embed_quota_exhausted(EmbeddingError("RESOURCE_EXHAUSTED")) is True


def test_is_embed_quota_exhausted_matches_bare_429():
    assert _is_embed_quota_exhausted(EmbeddingError("HTTP 429 rate limited")) is True


def test_is_embed_quota_exhausted_rejects_non_quota_error():
    # Narrowness proof: an auth / non-429 EmbeddingError must NOT match.
    assert _is_embed_quota_exhausted(
        EmbeddingError("Gemini embedding API call failed: 401 unauthorized")
    ) is False
    assert _is_embed_quota_exhausted(
        EmbeddingError("Gemini API returned an empty embedding list")
    ) is False


# ---------------------------------------------------------------------------
# skip_on_embed_quota — the propagating boundary (s1/x1)
# ---------------------------------------------------------------------------

def test_skip_on_embed_quota_skips_on_quota_error():
    with pytest.raises(Skipped) as exc_info:
        with skip_on_embed_quota():
            raise EmbeddingError(_QUOTA_MSG)
    # The skip carries the loud, named reason — not a bare/silent skip.
    assert "RESOURCE_EXHAUSTED" in str(exc_info.value.msg)


def test_skip_on_embed_quota_propagates_non_quota_embedding_error():
    # A non-quota EmbeddingError is re-raised, not swallowed into a skip.
    with pytest.raises(EmbeddingError):
        with skip_on_embed_quota():
            raise EmbeddingError("Gemini API returned an empty embedding list")


def test_skip_on_embed_quota_propagates_unrelated_exception():
    # Only EmbeddingError is caught; everything else passes straight through.
    with pytest.raises(ValueError):
        with skip_on_embed_quota():
            raise ValueError("unrelated")


def test_skip_on_embed_quota_is_transparent_on_success():
    sentinel = []
    with skip_on_embed_quota():
        sentinel.append("ran")
    assert sentinel == ["ran"]


# ---------------------------------------------------------------------------
# skip_if_embed_quota_exhausted — the active probe (adjacency)
# ---------------------------------------------------------------------------

def test_probe_skips_when_provider_raises_quota_error():
    provider = _FakeProvider(raises=EmbeddingError(_QUOTA_MSG))
    with pytest.raises(Skipped):
        skip_if_embed_quota_exhausted(provider)
    assert provider.called is True


def test_probe_returns_when_provider_healthy():
    # A healthy probe (no 429) returns → the caller's assert fires on a real bug.
    provider = _FakeProvider(vector=[0.4, 0.5, 0.6])
    skip_if_embed_quota_exhausted(provider)  # no skip raised
    assert provider.called is True


def test_probe_propagates_non_quota_error_from_provider():
    provider = _FakeProvider(raises=EmbeddingError("401 unauthorized"))
    with pytest.raises(EmbeddingError):
        skip_if_embed_quota_exhausted(provider)


def test_probe_none_provider_returns_without_skip():
    # A None provider can't be probed — return without skipping so the caller's
    # assert fires rather than being silently masked. Must NOT raise.
    skip_if_embed_quota_exhausted(None)


# ---------------------------------------------------------------------------
# _parse_semver — name-prefix-tolerant version parse (no `packaging` dep)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("mitos 0.4.0", (0, 4, 0)),       # global probe format (name-prefixed)
        ("0.3.2", (0, 3, 2)),             # SOURCE_VERSION format (bare)
        ("mitos 0.4.0\n", (0, 4, 0)),     # trailing newline from argparse --version
        ("  0.3.2  ", (0, 3, 2)),         # surrounding whitespace
        ("", None),                       # empty
        ("mitos vX", None),               # non-integer token
        ("mitos", None),                  # name only, no version token
        ("mitos 0.x.0", None),            # partial non-integer
    ],
)
def test_parse_semver(text, expected):
    assert _parse_semver(text) == expected


# ---------------------------------------------------------------------------
# _is_stale — the version gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "global_text,source_text,expected",
    [
        ("mitos 0.3.2", "0.4.0", True),    # behind → stale (the skip case)
        ("mitos 0.4.0", "0.4.0", False),   # equal → the gate: NOT stale, test runs
        ("mitos 9.9.9", "0.4.0", False),   # ahead → NOT stale
        ("mitos 0.3.9", "0.4.0", True),    # behind on minor
        ("garbage", "0.4.0", False),       # global unparseable → uncertain, don't skip
        ("mitos 0.3.2", "garbage", False), # source unparseable → uncertain, don't skip
    ],
)
def test_is_stale(global_text, source_text, expected):
    assert _is_stale(global_text, source_text) is expected


# ---------------------------------------------------------------------------
# skip_if_global_mitos_stale — IO wrapper (driven with fakes, no real binary)
# ---------------------------------------------------------------------------

def _fake_run(stdout="", stderr="", returncode=0):
    def _run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0] if args else [], returncode=returncode,
            stdout=stdout, stderr=stderr,
        )
    return _run


def test_skip_if_global_mitos_stale_skips_when_behind(monkeypatch):
    # Global binary reports a version behind the source → loud, actionable skip.
    monkeypatch.setattr(
        live_helpers, "SOURCE_VERSION", "0.4.0", raising=True
    )
    monkeypatch.setattr(
        live_helpers.subprocess, "run", _fake_run(stdout="mitos 0.3.2\n")
    )
    with pytest.raises(Skipped) as exc_info:
        skip_if_global_mitos_stale("mitos")
    reason = str(exc_info.value.msg)
    assert "lags source 0.4.0" in reason
    assert "pipx install --force" in reason  # the skip is actionable


def test_skip_if_global_mitos_stale_runs_when_matched(monkeypatch):
    # Versions match → the gate does NOT skip (preserves packaging-drift catching).
    monkeypatch.setattr(live_helpers, "SOURCE_VERSION", "0.4.0", raising=True)
    monkeypatch.setattr(
        live_helpers.subprocess, "run", _fake_run(stdout="mitos 0.4.0\n")
    )
    skip_if_global_mitos_stale("mitos")  # no skip raised


def test_skip_if_global_mitos_stale_reads_stderr_fallback(monkeypatch):
    # Some builds emit --version on stderr; the wrapper falls back to it.
    monkeypatch.setattr(live_helpers, "SOURCE_VERSION", "0.4.0", raising=True)
    monkeypatch.setattr(
        live_helpers.subprocess, "run",
        _fake_run(stdout="", stderr="mitos 0.3.2\n"),
    )
    with pytest.raises(Skipped):
        skip_if_global_mitos_stale("mitos")


def test_skip_if_global_mitos_stale_returns_when_probe_fails(monkeypatch):
    # A binary that can't be probed (subprocess raises) → run the test, don't mask.
    def _boom(*args, **kwargs):
        raise FileNotFoundError("no such binary")

    monkeypatch.setattr(live_helpers.subprocess, "run", _boom)
    skip_if_global_mitos_stale("definitely-not-a-real-binary-xyz")  # no skip, no raise


# ---------------------------------------------------------------------------
# EMBED_QUOTA_SKIP_REASON — the loudness contract
# ---------------------------------------------------------------------------

def test_embed_quota_skip_reason_is_loud_and_actionable():
    # The reason must name the cause, that it is environmental, and that it resets —
    # never a bare/silent skip (PATTERNS.md: a silent skip is an invisible failure).
    assert "429" in EMBED_QUOTA_SKIP_REASON
    assert "RESOURCE_EXHAUSTED" in EMBED_QUOTA_SKIP_REASON
    assert "resets daily" in EMBED_QUOTA_SKIP_REASON
    assert "NOT a code defect" in EMBED_QUOTA_SKIP_REASON


# ===========================================================================
# Generative-enrichment-quota probe (Phase r3 / C4) — keyless fakes
# ===========================================================================
# The probe must be CI-verifiable (P10 "every fix carries a fixture"; P16) —
# otherwise it is only ever exercised on Vinga's box under the un-reproducible
# spent-quota condition. Drive it with a tiny fake client (no ``unittest.mock``,
# mirroring ``_FakeProvider``); no keys / Qdrant / network. ``run_sync_enrichment``
# constructs a ``types.GenerateContentConfig`` and (on the healthy path) does
# ``json.loads(resp.text)`` — both are pure, no network, on the fake path.


class _FakeResp:
    """A stand-in for the genai response object (only ``.text`` is read)."""

    def __init__(self, text):
        self.text = text


class _FakeModels:
    """``client.models`` stand-in — either raises or returns a fixed response.

    Records whether ``generate_content`` was called so a no-skip path can be
    distinguished from a short-circuit. The signature accepts the three keyword
    args ``run_sync_enrichment`` passes (``model``/``contents``/``config``).
    """

    def __init__(self, raises=None, text=None):
        self._raises, self._text, self.called = raises, text, False

    def generate_content(self, model, contents, config):
        self.called = True
        if self._raises is not None:
            raise self._raises
        return _FakeResp(self._text)


class _FakeClient:
    """A ``genai.Client`` stand-in exposing only ``.models.generate_content``."""

    def __init__(self, raises=None, text=None):
        self.models = _FakeModels(raises=raises, text=text)


# A valid JSON body for the healthy path (run_sync_enrichment does json.loads(resp.text)).
_HEALTHY_JSON = ('{"refined_core_axiom": "x", "refined_mechanisms": [], '
                 '"refined_scope": [], "suggested_relationships": {}}')


# ---------------------------------------------------------------------------
# _is_enrichment_quota_exhausted — the narrow signature matcher (wider than embed)
# ---------------------------------------------------------------------------

def test_is_enrichment_quota_exhausted_matches_full_429_payload():
    msg = (
        "LLM enrichment call failed: 429 RESOURCE_EXHAUSTED. Quota exceeded for "
        "metric generate_content_free_tier_requests, limit ..."
    )
    assert _is_enrichment_quota_exhausted(SynthesisError(msg)) is True


def test_is_enrichment_quota_exhausted_matches_resource_exhausted_only():
    assert _is_enrichment_quota_exhausted(SynthesisError("RESOURCE_EXHAUSTED")) is True


def test_is_enrichment_quota_exhausted_matches_bare_429():
    assert _is_enrichment_quota_exhausted(SynthesisError("HTTP 429 rate limited")) is True


def test_is_enrichment_quota_exhausted_matches_503():
    # Wider than the embed matcher: the generative model 503s under high demand.
    assert _is_enrichment_quota_exhausted(SynthesisError("HTTP 503 model overloaded")) is True


def test_is_enrichment_quota_exhausted_matches_unavailable():
    assert _is_enrichment_quota_exhausted(
        SynthesisError("503 UNAVAILABLE — model experiencing high demand")
    ) is True


def test_is_enrichment_quota_exhausted_rejects_non_quota_error():
    # Narrowness proof: an auth error and a malformed-JSON error must NOT match.
    assert _is_enrichment_quota_exhausted(
        SynthesisError("LLM enrichment call failed: 401 unauthorized")
    ) is False
    assert _is_enrichment_quota_exhausted(
        SynthesisError("LLM enrichment call failed: Expecting value: line 1 column 1")
    ) is False


# ---------------------------------------------------------------------------
# skip_if_enrichment_quota_exhausted — the active probe (driven with a fake client)
# ---------------------------------------------------------------------------

def test_enrichment_probe_skips_on_429():
    client = _FakeClient(raises=Exception(
        "429 RESOURCE_EXHAUSTED. Quota exceeded for metric "
        "generate_content_free_tier_requests"
    ))
    with pytest.raises(Skipped) as exc_info:
        skip_if_enrichment_quota_exhausted(client)
    # Loud, named reason — not a bare/silent skip.
    assert "RESOURCE_EXHAUSTED" in str(exc_info.value.msg)
    assert client.models.called is True


def test_enrichment_probe_skips_on_503():
    client = _FakeClient(raises=Exception("503 UNAVAILABLE model overloaded"))
    with pytest.raises(Skipped):
        skip_if_enrichment_quota_exhausted(client)
    assert client.models.called is True


def test_enrichment_probe_returns_when_healthy():
    # A healthy probe (valid JSON, no 429/503) returns → a real shortfall still
    # reaches the caller's assert (the probe never masks a genuine lost node).
    client = _FakeClient(text=_HEALTHY_JSON)
    skip_if_enrichment_quota_exhausted(client)  # no skip raised
    assert client.models.called is True


def test_enrichment_probe_propagates_non_quota_error():
    # A non-quota SynthesisError (auth) is re-raised, not swallowed into a skip —
    # a real enrichment bug fails red.
    client = _FakeClient(raises=Exception("401 unauthorized"))
    with pytest.raises(SynthesisError):
        skip_if_enrichment_quota_exhausted(client)


def test_enrichment_probe_no_key_returns_without_skip(monkeypatch):
    # No client + no GEMINI_API_KEY → can't probe → return without skipping (don't
    # mask the caller's assert). The delenv is load-bearing: a sibling live module's
    # ``load_live_env()`` may have injected the key into ``os.environ`` at import.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    skip_if_enrichment_quota_exhausted()  # no arg, no key → returns, no raise


# ---------------------------------------------------------------------------
# ENRICHMENT_QUOTA_SKIP_REASON — the loudness contract
# ---------------------------------------------------------------------------

def test_enrichment_quota_skip_reason_is_loud_and_actionable():
    assert "429" in ENRICHMENT_QUOTA_SKIP_REASON
    assert "503" in ENRICHMENT_QUOTA_SKIP_REASON
    assert "RESOURCE_EXHAUSTED" in ENRICHMENT_QUOTA_SKIP_REASON
    assert "resets daily" in ENRICHMENT_QUOTA_SKIP_REASON
    assert "NOT a code defect" in ENRICHMENT_QUOTA_SKIP_REASON
    # Names the SEPARATE generative bucket (distinct from the embed quota).
    assert "generate_content_free_tier_requests" in ENRICHMENT_QUOTA_SKIP_REASON
