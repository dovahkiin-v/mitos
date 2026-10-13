"""Tests for the closed provider-cause classifier and the three sites it serves (7a, B1).

When semantic recall degrades, three places say why: the read header
(``lexical.degraded_reason_from_error``), ``surface``'s scope-dump note
(``recall.assess_surface_recall``) and the write-path warning (``sync.py``). They
all take the cause from ``mitos.provider_cause``, so for one exception they carry
one phrase, and none of them prints the provider's body.

The provider fakes are the real SDK classes (``google.genai.errors``), raised and
re-wrapped exactly as ``embeddings.GeminiEmbeddingProvider`` wraps them. Every fake
carries :data:`SENTINEL` in its message and details, so "no provider text" is
asserted as the sentinel's absence, never as the absence of a guessed substring.
"""

import io
import json
import re
import shlex
import shutil
import tempfile
from contextlib import redirect_stdout
from typing import Iterator, Tuple

import pytest
from google.genai.errors import ClientError, ServerError
from unittest.mock import patch

from mitos import cli, mcp_server
from mitos.cli import cmd_init, cmd_query, cmd_surface
from mitos.config import MitosConfig
from mitos.errors import (CollectionMissingError, DatabaseError, EmbeddingError,
                          VectorStoreError)
from mitos.lexical import degraded_reason_from_error
from mitos.provider_cause import (AUTH, PROVIDER_CAUSES, QUOTA, TRANSIENT, UNKNOWN,
                                  classify_provider_error, describe_embed_failure,
                                  provider_cause_phrase)
from mitos.recall import (_SURFACE_POINTERS, assess_surface_recall,
                          scope_filter_recovery)
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager

SENTINEL = "B1-BLOB-SENTINEL"


class TimeoutException(Exception):
    """Stands in for httpx's timeout base, which the classifier matches by name."""


def _wrap(cause: BaseException) -> EmbeddingError:
    """Raises ``cause`` and re-wraps it the way ``get_embedding`` does."""
    try:
        raise cause
    except Exception as e:
        try:
            raise EmbeddingError(f"Gemini embedding API call failed: {str(e)}") from e
        except EmbeddingError as wrapped:
            return wrapped


def _body(code: int, status: str, reason: str = None, *, wrapped: bool = True):
    inner = {"code": code, "message": f"{SENTINEL} provider says no", "status": status}
    if reason:
        inner["details"] = [{
            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
            "reason": reason, "domain": "googleapis.com",
            "metadata": {"service": "generativelanguage.googleapis.com",
                         "note": SENTINEL},
        }]
    return {"error": inner} if wrapped else inner


# One representative per class, drives the cross-site rows (C2, C3).
_REPRESENTATIVE = {
    AUTH: lambda: _wrap(ClientError(400, _body(400, "INVALID_ARGUMENT", "API_KEY_INVALID"))),
    QUOTA: lambda: _wrap(ClientError(429, _body(429, "RESOURCE_EXHAUSTED"))),
    TRANSIENT: lambda: _wrap(ServerError(503, _body(503, "UNAVAILABLE"))),
    UNKNOWN: lambda: _wrap(ClientError(400, _body(400, "INVALID_ARGUMENT"))),
}


# --------------------------------------------------------------------------- #
# C1 — the classifier
# --------------------------------------------------------------------------- #

_CLASSIFIED = [
    ("401 unauthenticated", lambda: _wrap(ClientError(401, _body(401, "UNAUTHENTICATED"))), AUTH),
    ("403 permission denied", lambda: _wrap(ClientError(403, _body(403, "PERMISSION_DENIED"))), AUTH),
    ("400 API_KEY_INVALID", _REPRESENTATIVE[AUTH], AUTH),
    ("400 API_KEY_INVALID, replay shape",
     lambda: _wrap(ClientError(400, _body(400, "INVALID_ARGUMENT", "API_KEY_INVALID",
                                          wrapped=False))), AUTH),
    ("400 SERVICE_DISABLED",
     lambda: _wrap(ClientError(400, _body(400, "FAILED_PRECONDITION", "SERVICE_DISABLED"))), AUTH),
    ("429 typed", _REPRESENTATIVE[QUOTA], QUOTA),
    ("429 text only (the shipped arm)",
     lambda: EmbeddingError(f'429 {{"error": {{"status": "RESOURCE_EXHAUSTED", "m": "{SENTINEL}"}}}}'),
     QUOTA),
    ("503 unavailable", _REPRESENTATIVE[TRANSIENT], TRANSIENT),
    ("500 internal", lambda: _wrap(ServerError(500, _body(500, "INTERNAL"))), TRANSIENT),
    ("TimeoutError cause", lambda: _wrap(TimeoutError(f"{SENTINEL} timed out")), TRANSIENT),
    ("ConnectionError cause", lambda: _wrap(ConnectionRefusedError(SENTINEL)), TRANSIENT),
    ("MRO-named TimeoutException", lambda: _wrap(TimeoutException(SENTINEL)), TRANSIENT),
    ("empty embeddings, cause-less",
     lambda: EmbeddingError("Gemini API returned an empty embedding list"), UNKNOWN),
    ("empty embeddings, re-wrapped as production does",
     lambda: _wrap(EmbeddingError("Gemini API returned an empty embedding list")), UNKNOWN),
    ("an unrelated 400", _REPRESENTATIVE[UNKNOWN], UNKNOWN),
    ("a non-JSON 401 body", lambda: _wrap(ClientError(401, f"{SENTINEL} text body")), AUTH),
    ("'403' only in the text",
     lambda: EmbeddingError(f"upsert failed for 'error-403-handling' {SENTINEL}"), UNKNOWN),
]


@pytest.mark.parametrize("label,make,expected", _CLASSIFIED, ids=[c[0] for c in _CLASSIFIED])
def test_each_fixture_classifies_into_its_class(label, make, expected):
    cause = classify_provider_error(make())
    assert cause == expected, label
    assert cause in PROVIDER_CAUSES


def test_the_vocabulary_is_closed_at_exactly_four_tokens():
    assert PROVIDER_CAUSES == ("auth", "quota", "transient", "unknown")


def test_the_quota_and_unknown_phrases_are_the_shipped_strings():
    assert provider_cause_phrase(QUOTA) == "embedding provider rate-limited (429)"
    assert provider_cause_phrase(UNKNOWN) == "embedding provider error"


@pytest.mark.parametrize("cause", PROVIDER_CAUSES)
def test_every_phrase_is_short_names_its_class_and_no_command(cause):
    phrase = provider_cause_phrase(cause)
    assert len(phrase) <= 60
    assert "mitos " not in phrase and "`" not in phrase
    if cause in (AUTH, TRANSIENT):
        assert phrase.endswith(f"({cause})")
    if cause == AUTH:
        assert "down" not in phrase  # the provider answered, and refused


def test_quota_text_wins_over_a_typed_transient_link():
    """The shipped 429 arm stays first: a rate-limit text over a 5xx chain is quota."""
    exc = EmbeddingError(f"429 RESOURCE_EXHAUSTED {SENTINEL}")
    exc.__cause__ = ServerError(503, _body(503, "UNAVAILABLE"))
    assert classify_provider_error(exc) == QUOTA


def test_the_chain_walk_survives_a_cycle():
    a, b = EmbeddingError("a"), EmbeddingError("b")
    a.__cause__, b.__cause__ = b, a
    assert classify_provider_error(a) == UNKNOWN


def test_the_write_path_helper_leaves_a_non_provider_error_untouched():
    assert describe_embed_failure(VectorStoreError("store fault x")) == "store fault x"
    for make in _REPRESENTATIVE.values():
        assert SENTINEL not in describe_embed_failure(make())


# --------------------------------------------------------------------------- #
# Fixtures for the surfaces
# --------------------------------------------------------------------------- #

@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def ws(offline) -> Iterator[Tuple[MitosConfig, MitosSyncManager]]:
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    cmd_init(config)
    m = MitosSyncManager(config)
    for slug in ("cache-strategy", "cache-eviction"):
        res = m.record_decision_entry(f"Use a {slug} policy.", "Rejected x.", ["y"],
                                      slug=slug)
        assert "error" not in res, res
    yield config, m
    shutil.rmtree(tmp, ignore_errors=True)


class _Boom:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def get_embedding(self, text, is_query=False):
        raise self.exc


class _Upserts:
    def upsert(self, *a, **kw):
        raise AssertionError("the embedding raised first; nothing reaches upsert")


def _mcp(config, exc, tool="surface_decisions", **kw):
    comps = (GraphStore(config.db_path, read_only=True), _Boom(exc), object())
    with patch.object(mcp_server, "get_workspace_components", return_value=comps):
        return getattr(mcp_server, tool)("cache strategy", project=config.workspace_dir, **kw)


def _cli(config, exc, verb=cmd_surface, **kw):
    mgr = MitosSyncManager(config)
    mgr.embed_provider = _Boom(exc)
    mgr.vector_store = object()
    buf = io.StringIO()
    with patch.object(cli, "MitosSyncManager", return_value=mgr), redirect_stdout(buf):
        verb(config, "cache strategy", **kw)
    return buf.getvalue()


def _record_with(config, exc, capsys, slug):
    m = MitosSyncManager(config)
    m.embed_provider = _Boom(exc)
    m.vector_store = _Upserts()
    capsys.readouterr()
    res = m.record_decision_entry(f"Axiom for {slug}.", "Rejected x.", [], slug=slug)
    assert "error" not in res, res
    return m, capsys.readouterr()


# --------------------------------------------------------------------------- #
# C3 — three sites, one phrase; C2 — no provider text anywhere
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cause", PROVIDER_CAUSES)
def test_header_dump_note_and_write_warning_carry_one_phrase(ws, capsys, cause):
    config, _ = ws
    phrase = provider_cause_phrase(cause)
    exc = _REPRESENTATIVE[cause]()
    assert classify_provider_error(exc) == cause

    # (a) the lexical header, both surfaces.
    header = json.loads(_mcp(config, exc))
    assert header["degraded"] == "lexical"
    assert header["degraded_reason"] == phrase
    cli_header = json.loads(_cli(config, exc, as_json=True))
    assert cli_header["degraded_reason"] == phrase

    # (b) the scoped dump note, both surfaces.
    head = f"Semantic recall unavailable ({phrase}) — showing "
    dump = json.loads(_mcp(config, exc, scope="y"))
    assert dump["note"].startswith(head), dump["note"]
    cli_dump = json.loads(_cli(config, exc, as_json=True, scope="y"))
    assert cli_dump["note"].startswith(head), cli_dump["note"]

    # (c) the record-path warning.
    _, captured = _record_with(config, exc, capsys, f"write-{cause}")
    assert f"Embedding upsert deferred for 'write-{cause}': {phrase}\n" in captured.err


@pytest.mark.parametrize("cause", PROVIDER_CAUSES)
def test_the_partial_list_note_names_the_same_phrase(cause):
    """A fault after hits were appended keeps them; the note names the injected cause."""
    phrase = provider_cause_phrase(cause)
    for surface in ("cli", "mcp"):
        _, note = assess_surface_recall(
            semantic_ran=False, top_score=None, result_count=2, scope="x",
            surface=surface, lever=None, scope_total=None, degraded_reason=phrase)
        assert note.startswith(f"Semantic recall unavailable ({phrase}) — showing the ")


@pytest.mark.parametrize("cause", PROVIDER_CAUSES)
def test_no_encoding_carries_the_provider_body(ws, capsys, cause):
    config, _ = ws
    exc = _REPRESENTATIVE[cause]()
    assert SENTINEL in str(exc)  # the fake really does carry a body

    outputs = [
        _mcp(config, exc),
        _mcp(config, exc, scope="y"),
        _mcp(config, exc, tool="query_decisions"),
        _cli(config, exc),
        _cli(config, exc, as_json=True),
        _cli(config, exc, scope="y"),
        _cli(config, exc, verb=cmd_query),
        _cli(config, exc, verb=cmd_query, as_json=True),
    ]
    for out in outputs:
        assert SENTINEL not in out

    m, captured = _record_with(config, exc, capsys, f"blob-{cause}")
    assert SENTINEL not in captured.err + captured.out

    # The drain retries the queued row with the same provider and reports on stdout.
    m.drain_pending_embeddings()
    drained = capsys.readouterr()
    assert f"Failed to drain embedding for 'blob-{cause}': " in drained.out
    assert SENTINEL not in drained.out + drained.err


# --------------------------------------------------------------------------- #
# C4 — no shell command on MCP
# --------------------------------------------------------------------------- #

_ARM_REPRESENTATIVES = [
    None,
    DatabaseError("This graph predates the V1a schema (a prototype layout)."),
    EmbeddingError('429 {"status": "RESOURCE_EXHAUSTED"}'),
    *[make() for make in _REPRESENTATIVE.values()],
    CollectionMissingError("gone", collection="mitos-x"),
    CollectionMissingError("gone"),
    VectorStoreError("Qdrant 500"),
    RuntimeError("connection refused"),
    KeyError("x"),
]


@pytest.mark.parametrize("exc", _ARM_REPRESENTATIVES, ids=lambda e: type(e).__name__)
def test_no_mcp_reason_names_a_shell_command(exc):
    reason = degraded_reason_from_error(exc, surface="mcp", project="my proj")
    assert "mitos " not in reason
    assert "`" not in reason


def test_the_mcp_heal_clauses_state_the_fact_and_the_actor():
    for exc in (CollectionMissingError("gone", collection="mitos-x"),
                DatabaseError("This graph predates the V1a schema.")):
        reason = degraded_reason_from_error(exc, surface="mcp", project="p")
        assert "no tool on this surface performs" in reason
        assert "a person with a shell" in reason


def test_no_mcp_pointer_names_a_shell_command():
    for key, value in _SURFACE_POINTERS["mcp"].items():
        assert "mitos " not in value, key


def test_the_rendered_unused_scope_prefix_on_mcp_names_no_command():
    counts = {"auth": {"decisions": 3, "open_questions": 0}}
    _, note = assess_surface_recall(
        semantic_ran=True, top_score=None, result_count=0, scope="ghost",
        scope_counts=counts, surface="mcp", lever=None, degraded_reason=None)
    assert "unused scope tag" in note and "mitos " not in note
    recovery = scope_filter_recovery(scope="ghost", scope_counts=counts, surface="mcp")
    assert recovery is not None
    assert "mitos " not in json.dumps(recovery, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# C5 — the CLI recipes in the reason parse, a project with a space included
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("exc,verb", [
    (CollectionMissingError("gone", collection="mitos-x"), "reconcile"),
    (DatabaseError("This graph predates the V1a schema."), "cutover"),
])
def test_the_cli_reason_recipes_parse_with_their_selector(exc, verb):
    reason = degraded_reason_from_error(exc, surface="cli", project="my proj")
    (recipe,) = re.findall(r"`(mitos [^`]+)`", reason)
    args = cli._build_parser().parse_args(shlex.split(recipe)[1:])
    assert args.command == verb
    assert args.project_post == "my proj"


def test_the_mocked_cutover_arm_prints_a_recipe_for_this_project(offline, tmp_path):
    """The CLI surface over a pre-V1a graph: the header's recipe targets this workspace."""
    ws = tmp_path / "my proj"
    ws.mkdir()
    config = MitosConfig(str(ws))
    cmd_init(config)
    exc = DatabaseError("This graph predates the V1a schema (a prototype layout).")
    buf = io.StringIO()
    with patch.object(cli, "MitosSyncManager", side_effect=exc), redirect_stdout(buf):
        cmd_surface(config, "cache", as_json=True)
    (recipe,) = re.findall(r"`(mitos [^`]+)`", json.loads(buf.getvalue())["degraded_reason"])
    args = cli._build_parser().parse_args(shlex.split(recipe)[1:])
    assert args.command == "cutover"
    assert args.project_post == config.project


# --------------------------------------------------------------------------- #
# C8 — a missing reason is loud
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("scope_total", [None, 4])
def test_a_degraded_call_without_a_reason_raises(scope_total):
    with pytest.raises(ValueError, match="degraded_reason"):
        assess_surface_recall(semantic_ran=False, top_score=None, result_count=2,
                              scope="x", surface="mcp", lever=None,
                              scope_total=scope_total, degraded_reason=None)


def test_the_reason_is_a_required_keyword():
    with pytest.raises(TypeError, match="degraded_reason"):
        assess_surface_recall(semantic_ran=True, top_score=0.9, result_count=1,
                              scope=None, surface="mcp", lever=None)
    with pytest.raises(TypeError):
        degraded_reason_from_error(None)


# --------------------------------------------------------------------------- #
# Stretch — a print site added later cannot print the provider body
# --------------------------------------------------------------------------- #

_EMBED_CALLS = {"get_embedding", "get_embeddings_batch", "_best_effort_embed",
                "drain_pending_embeddings"}


def _embedding_handlers(tree):
    """Yields (handler, name) for each `except … as name` whose `try` embeds."""
    import ast
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        calls = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
                 for stmt in node.body for n in ast.walk(stmt) if isinstance(n, ast.Call)}
        if calls & _EMBED_CALLS:
            for handler in node.handlers:
                if handler.name:
                    yield handler, handler.name


def _interpolates_bare(handler, name):
    """Returns the lines of f-strings that interpolate `name` or `str(name)` directly."""
    import ast
    hits = []
    for n in ast.walk(handler):
        if not isinstance(n, ast.FormattedValue):
            continue
        v = n.value
        if isinstance(v, ast.Name) and v.id == name:
            hits.append(n.lineno)
        elif (isinstance(v, ast.Call) and getattr(v.func, "id", None) == "str"
              and v.args and isinstance(v.args[0], ast.Name) and v.args[0].id == name):
            hits.append(n.lineno)
    return hits


def test_no_sync_print_site_interpolates_an_embedding_exception_bare():
    import ast
    import inspect
    from mitos import sync
    tree = ast.parse(inspect.getsource(sync))
    handlers = list(_embedding_handlers(tree))
    # Population guard: the five shipped sites' handlers are all found, so the row
    # cannot go green over an empty set.
    routed = [h for h, name in handlers
              if any(isinstance(n, ast.Call) and getattr(n.func, "id", None) == "describe_embed_failure"
                     for n in ast.walk(h))]
    assert len(routed) >= 5, len(routed)
    offenders = [line for h, name in handlers for line in _interpolates_bare(h, name)]
    assert offenders == [], f"sync.py lines print an embedding exception bare: {offenders}"
