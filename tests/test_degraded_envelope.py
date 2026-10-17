"""Phase 2g: the degraded envelope — one clamped limit, the omission statement, the dump.

When semantic recall is down, both read verbs end in a degraded exit: the lexical
envelope (a term-match over the markdown corpus) on both verbs, or ``surface``'s
unranked scope dump. ``degraded.py`` composes both once for the two boundaries. The
rows here pin, on every driver (MCP ``surface_decisions`` / ``query_decisions``,
``mitos surface`` / ``mitos query`` on ``--json`` and text):

- row 8: the lexical match count is ``clamp_limit(limit)`` on every route into it;
- row 9: every degraded exit carries ``not_computed`` and the omission clause, then
  its rejection section's clause and ``rejection_section: "lexical"`` (5d), and no
  ranked, healthy-empty or exact-slug exit does;
- row 10: no degraded envelope carries a band, a partition or an edge clause;
- row 11: the dump is stamped by its own read, makes no modifiers read, and routes
  to the lexical exit when that read raises or comes back empty;
- row 12: CLI ``--json`` and MCP agree on every key and value but the pointers.

Every row enters the outer frame and asserts its ``degraded_reason`` names the cause,
so a row whose frame never reached the degraded exit cannot pass.

The seeding writes markdown blocks into the buffer and commits the same text through
``parse_entry_stream`` → ``commit_parsed_entry`` (the sync path), so the lexical path
reads exactly what the graph committed (plan G3).
"""

import io
import json
import re
import shutil
import tempfile
from contextlib import redirect_stdout
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from unittest.mock import patch

import pytest

from mitos import cli, mcp_server
from mitos.cli import cmd_init, cmd_query, cmd_surface
from mitos.config import MitosConfig
from mitos.errors import CollectionMissingError, DatabaseError
from mitos.parser import parse_entry_stream
from mitos.recall import (DEGRADED_NOT_COMPUTED, degraded_omission_clause,
                          degraded_section_clause, stamps_unread_clause,
                          withheld_clause)
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager

from _point_hits import CannedVector, point_hit
from conftest import resolve_like_main

_OMISSION = degraded_omission_clause()
# 5d: every degraded exit's rejection section clause, right after the omission clause.
_SECTION = degraded_section_clause(computed=True, retired_dropped=True)
_SECTION_UNFILTERED = degraded_section_clause(computed=True, retired_dropped=False)
_ABSENT = "mitos-tmp-absent-collection"
_PRE_V1A = "This graph predates the V1a schema (a prototype layout was detected)."

CAUSES = ("pre-v1a", "no-providers", "missing-collection", "mid-query")
# Causes that leave a readable store (the pre-V1a graph takes the store down with it).
STORE_CAUSES = ("no-providers", "missing-collection", "mid-query")


def reason_named(cause: str, reason: str) -> bool:
    """Says whether a degraded reason names the cause the row injected."""
    return {
        "pre-v1a": lambda r: "V1a" in r,
        "no-providers": lambda r: r == "embeddings/Qdrant unavailable",
        "missing-collection": lambda r: _ABSENT in r,
        "mid-query": lambda r: r == "Qdrant unreachable",
    }[cause](reason)


# --------------------------------------------------------------------------- #
# The workspace and its corpus
# --------------------------------------------------------------------------- #

@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def dws(offline) -> Iterator[MitosConfig]:
    """A fresh workspace, configured the way ``main()`` builds it (CLI⇄MCP echo parity)."""
    tmp = tempfile.mkdtemp()
    with redirect_stdout(io.StringIO()):
        cmd_init(MitosConfig(tmp))
    yield resolve_like_main(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


def block(slug: str, axiom: str, *, rejected: str = "Rejected the obvious alternative.",
          scope: Optional[str] = None, mechanisms: Optional[str] = None,
          **relations: str) -> str:
    """One decision block in the corpus format (relations by field name, e.g. supersedes=)."""
    lines = [f"### {slug}", "", f"**Decided:** {axiom}", f"**Rejected:** {rejected}"]
    if mechanisms:
        lines.append(f"**Mechanisms:** {mechanisms}")
    if scope:
        lines.append(f"**Scope:** {scope}")
    for field, value in relations.items():
        lines.append(f"**{field.capitalize()}:** {value}")
    return "\n".join(lines) + "\n"


def write_buffer(config: MitosConfig, blocks: List[str]) -> str:
    """Replaces the buffer's entry stream with ``blocks`` (newest first) and returns it."""
    with open(config.decisions_file, encoding="utf-8") as f:
        text = f.read()
    marker = text.index("<!-- BEGIN ENTRIES")
    head = text[:text.index("\n", marker) + 1]
    body = "\n".join(blocks)
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write(f"{head}\n{body}")
    return body


def write_archive(config: MitosConfig, name: str, blocks: List[str]) -> str:
    """Writes one archive file holding ``blocks`` (newest first) and returns its text."""
    import os
    os.makedirs(config.archive_dir, exist_ok=True)
    body = "\n".join(blocks)
    with open(os.path.join(config.archive_dir, name), "w", encoding="utf-8") as f:
        f.write(body)
    return body


def commit_text(config: MitosConfig, text: str) -> Dict[str, str]:
    """Commits every decision block in ``text`` oldest first; returns slug → node id
    of the last commit per slug (the newest block's node)."""
    store = GraphStore(config.db_path)
    ids: Dict[str, str] = {}
    for entry in reversed(parse_entry_stream(text, "decision", failures=[])):
        ids[entry.slug] = store.commit_parsed_entry(entry).node_id
    return ids


# --------------------------------------------------------------------------- #
# Drivers: (surface, verb, encoding) over each cause
# --------------------------------------------------------------------------- #

class _Embeds:
    """An embedding provider that succeeds — the fault under test is downstream."""

    def get_embedding(self, text, is_query=False):
        return [0.1, 0.2, 0.3]


class _MissingCollection:
    """Qdrant is up and says that collection does not exist."""

    def query(self, vector, limit=5, *, with_payload=True):
        raise CollectionMissingError(
            f"Qdrant collection '{_ABSENT}' does not exist.", collection=_ABSENT)

    def delete_points(self, point_ids):
        raise CollectionMissingError(
            f"Qdrant collection '{_ABSENT}' does not exist.", collection=_ABSENT)


class _Raising:
    """A vector store that fails mid-query."""

    def query(self, vector, limit=5, *, with_payload=True):
        raise RuntimeError("qdrant fell over")

    def delete_points(self, point_ids):
        raise RuntimeError("qdrant fell over")


def _providers(cause: str, vector: Any = None) -> Tuple[Any, Any]:
    if vector is not None:
        return _Embeds(), vector
    return {
        "no-providers": (None, None),
        "missing-collection": (_Embeds(), _MissingCollection()),
        "mid-query": (_Embeds(), _Raising()),
    }[cause]


Wrap = Optional[Callable[[Any], Any]]


def _kwargs(**kw: Any) -> Dict[str, Any]:
    return {k: v for k, v in kw.items() if v is not None}


def _mcp(verb: str, config: MitosConfig, query: str, *, cause: str, wrap: Wrap = None,
         vector: Any = None, limit: Optional[int] = None, scope: Optional[str] = None,
         full_top: Optional[int] = None) -> Dict[str, Any]:
    tool = mcp_server.surface_decisions if verb == "surface" else mcp_server.query_decisions
    kw = _kwargs(limit=limit, full_top=full_top,
                 **({"scope": scope} if verb == "surface" else {}))
    if cause == "pre-v1a":
        patcher = patch.object(mcp_server, "get_workspace_components",
                               side_effect=DatabaseError(_PRE_V1A))
    else:
        store = GraphStore(config.db_path, read_only=True)
        if wrap is not None:
            store = wrap(store)
        embed, vec = _providers(cause, vector)
        patcher = patch.object(mcp_server, "get_workspace_components",
                               return_value=(store, embed, vec))
    with patcher:
        return json.loads(tool(query, project=config.workspace_dir, **kw))


def _cli(verb: str, config: MitosConfig, query: str, *, cause: str, as_json: bool,
         wrap: Wrap = None, vector: Any = None, limit: Optional[int] = None,
         scope: Optional[str] = None, full_top: Optional[int] = None):
    fn = cmd_surface if verb == "surface" else cmd_query
    kw = _kwargs(limit=limit, full_top=full_top,
                 **({"scope": scope} if verb == "surface" else {}))
    if cause == "pre-v1a":
        patcher = patch.object(cli, "MitosSyncManager", side_effect=DatabaseError(_PRE_V1A))
    else:
        manager = MitosSyncManager(config)
        manager.embed_provider, manager.vector_store = _providers(cause, vector)
        if wrap is not None:
            manager.store = wrap(manager.store)
        patcher = patch.object(cli, "MitosSyncManager", return_value=manager)
    buf = io.StringIO()
    with patcher, redirect_stdout(buf):
        fn(config, query, as_json=as_json, **kw)
    out = buf.getvalue()
    return json.loads(out) if as_json else out


# name → (call, verb, surface, encoding)
DRIVERS: Dict[str, Tuple[Callable[..., Any], str, str, str]] = {
    "mcp-surface": (lambda *a, **k: _mcp("surface", *a, **k), "surface", "mcp", "json"),
    "mcp-query": (lambda *a, **k: _mcp("query", *a, **k), "query", "mcp", "json"),
    "cli-surface-json": (lambda *a, **k: _cli("surface", *a, as_json=True, **k),
                         "surface", "cli", "json"),
    "cli-query-json": (lambda *a, **k: _cli("query", *a, as_json=True, **k),
                       "query", "cli", "json"),
    "cli-surface-text": (lambda *a, **k: _cli("surface", *a, as_json=False, **k),
                         "surface", "cli", "text"),
    "cli-query-text": (lambda *a, **k: _cli("query", *a, as_json=False, **k),
                       "query", "cli", "text"),
}
JSON_DRIVERS = [d for d, spec in DRIVERS.items() if spec[3] == "json"]
SURFACE_DRIVERS = [d for d, spec in DRIVERS.items() if spec[1] == "surface"]


def drive(driver: str, config: MitosConfig, query: str, **kw: Any):
    return DRIVERS[driver][0](config, query, **kw)


def text_slugs(out: str) -> List[str]:
    """The ``N. slug`` lines of a text render, slug only."""
    return re.findall(r"^\d+\. (\S+)", out, flags=re.MULTILINE)


def listed(driver: str, resp: Any) -> List[str]:
    """The slugs an answer lists, on any encoding."""
    if DRIVERS[driver][3] == "text":
        return text_slugs(resp)
    key = "matches" if "matches" in resp else "active_decisions"
    return [d["slug"] for d in resp[key]]


def assert_cause(driver: str, cause: str, resp: Any) -> None:
    """The outer frame was entered: the answer names the injected cause."""
    if DRIVERS[driver][3] == "text":
        header = next(ln for ln in resp.splitlines() if "Semantic recall unavailable" in ln)
        reason = header.split("Semantic recall unavailable (", 1)[1].split(") —", 1)[0]
    else:
        reason = resp["degraded_reason"] if "degraded_reason" in resp else None
        if reason is None:  # the dump carries its cause on the note only
            reason = resp["note"].split("Semantic recall unavailable (", 1)[1].split(") —", 1)[0]
    assert reason_named(cause, reason), (cause, reason)


# --------------------------------------------------------------------------- #
# Row 8 — one clamped limit on every route into the lexical match
# --------------------------------------------------------------------------- #

def _sixty(config: MitosConfig) -> None:
    """60 blocks that all match "cache", one of them committed (a populated graph is
    what makes the absent collection a gap rather than a healthy-empty index)."""
    blocks = [block(f"cache-blk-{i:02d}", f"Cache rule number {i}.") for i in range(60)]
    write_buffer(config, blocks)
    commit_text(config, blocks[-1])


@pytest.mark.parametrize("cause", CAUSES)
@pytest.mark.parametrize("driver", list(DRIVERS))
@pytest.mark.parametrize("limit,shown", [(None, 5), (3, 3), (200, 50), (0, 1)])
def test_the_lexical_count_is_the_clamped_limit_on_every_route(dws, driver, cause,
                                                               limit, shown):
    """Row 8: default 5, explicit 3, 200 → 50, 0 → 1 — the ranked reads' clamp."""
    _sixty(dws)
    resp = drive(driver, dws, "cache", cause=cause, limit=limit)
    assert_cause(driver, cause, resp)
    if DRIVERS[driver][3] == "json":
        assert resp["degraded"] == "lexical"
    assert len(listed(driver, resp)) == shown


# --------------------------------------------------------------------------- #
# Rows 9 and 10 — every degraded exit says what it did not compute; none carries a band
# --------------------------------------------------------------------------- #

# 5d: `rejection_matches` / `rejection_section` left this set — every degraded exit
# carries its section, matched by words, under the `lexical` token only.
_BAND_KEYS = ("confidence", "retired_partition", "retired_partition_omitted",
              "all_superseded", "rejection_dropped")
_EDGE_STEMS = ("This ranking ended", "This ranking's edge")


def _assert_degraded_statement(driver: str, resp: Any, cause: str) -> None:
    section = _SECTION_UNFILTERED if cause == "pre-v1a" else _SECTION
    # 8a: with the graph unread nothing was stamped, so the stamp labels join the
    # omissions — the token on the key, its sentence on the clause.
    omission = degraded_omission_clause(labels_computed=cause != "pre-v1a")
    if DRIVERS[driver][3] == "text":
        assert f"{omission} {section}" in resp
        assert not any(stem in resp for stem in _EDGE_STEMS)
        assert "⚠ confidence:" not in resp
        return
    # 6b: the band's evidence object is named beside the band it rides with.
    assert list(DEGRADED_NOT_COMPUTED) == [
        "retired_partition", "edge", "band", "band_evidence"]
    assert resp["not_computed"] == (list(DEGRADED_NOT_COMPUTED)
                                    + (["stamp_labels"] if cause == "pre-v1a" else []))
    assert f"{omission} {section}" in resp["note"]
    assert resp["rejection_section"] == "lexical"
    for key in _BAND_KEYS:
        assert key not in resp, key
    assert not any(stem in resp["note"] for stem in _EDGE_STEMS)


@pytest.mark.parametrize("cause", CAUSES)
@pytest.mark.parametrize("driver", list(DRIVERS))
def test_every_lexical_exit_names_what_it_did_not_compute(dws, driver, cause):
    """Rows 9 and 10, the lexical exit: the key, the clause, and no band or edge."""
    write_buffer(dws, [block("cache-strategy", "Use a write-through cache.")])
    commit_text(dws, block("cache-strategy", "Use a write-through cache."))
    resp = drive(driver, dws, "cache", cause=cause)
    assert_cause(driver, cause, resp)
    assert listed(driver, resp) == ["cache-strategy"]
    _assert_degraded_statement(driver, resp, cause)


def _scoped(config: MitosConfig) -> Dict[str, str]:
    blocks = [block("cache-amend", "Amend the write-through cache.", scope="other",
                    amends="cache-strategy"),
              block("cache-strategy", "Use a write-through cache.", scope="caching")]
    return commit_text(config, write_buffer(config, blocks))


@pytest.mark.parametrize("cause", STORE_CAUSES)
@pytest.mark.parametrize("driver", SURFACE_DRIVERS)
def test_the_scope_dump_names_what_it_did_not_compute(dws, driver, cause):
    """Rows 9 and 10, the dump exit (no `degraded` marker: it is not the lexical one)."""
    _scoped(dws)
    resp = drive(driver, dws, "nothing lexical matches this", cause=cause, scope="caching")
    assert_cause(driver, cause, resp)
    assert listed(driver, resp) == ["cache-strategy"]
    if DRIVERS[driver][3] == "json":
        assert "degraded" not in resp
    _assert_degraded_statement(driver, resp, cause)


def _assert_no_statement(resp: Any) -> None:
    if isinstance(resp, str):
        assert _OMISSION not in resp
        assert "searched by their words" not in resp
    else:
        assert "not_computed" not in resp
        assert _OMISSION not in resp.get("note", "")
        assert "searched by their words" not in resp.get("note", "")
        assert resp.get("rejection_section") != "lexical"


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_a_ranked_answer_carries_no_omission_statement(dws, driver):
    """Row 9's absent cell: a healthy ranked envelope computed all three."""
    ids = _scoped(dws)
    vector = CannedVector([point_hit(ids["cache-strategy"], 0.9, slug="cache-strategy")])
    resp = drive(driver, dws, "a claim that is not any slug", cause="no-providers",
                 vector=vector)
    assert listed(driver, resp) == ["cache-strategy"]
    _assert_no_statement(resp)


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_the_healthy_empty_index_carries_no_omission_statement(dws, driver):
    """Row 9's absent cell: an absent collection over an empty graph is the empty index."""
    resp = drive(driver, dws, "anything at all", cause="missing-collection")
    if not isinstance(resp, str):
        assert "degraded" not in resp
    _assert_no_statement(resp)


@pytest.mark.parametrize("driver", ["mcp-query"])
def test_the_exact_slug_exit_carries_no_omission_statement(dws, driver):
    """Row 9's absent cell: an exact-slug hit is a graph read, even with recall down.
    (MCP only: `mitos query` has no exact-slug exit.)"""
    _scoped(dws)
    resp = drive(driver, dws, "cache-strategy", cause="no-providers")
    assert "cache-strategy" in (resp if isinstance(resp, str) else json.dumps(resp))
    _assert_no_statement(resp)


# --------------------------------------------------------------------------- #
# Row 11 — the dump is stamped by its own read and makes no modifiers read
# --------------------------------------------------------------------------- #

_COUNTED = ("get_modifiers", "get_modifiers_map", "get_node_states", "resolve_slug",
            "get_node_state", "get_active_decisions")


class _Spy:
    """Forwards to a real store, counting calls (not lookups: the walk checks its
    store's methods exist) to the reads that matter, optionally replacing one."""

    def __init__(self, store: Any, **replace: Callable[..., Any]) -> None:
        self._store = store
        self._replace = replace
        self.calls: List[str] = []

    def __getattr__(self, name: str) -> Any:
        target = self._replace.get(name) or getattr(self._store, name)
        if name not in _COUNTED:
            return target

        def counted(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return target(*args, **kwargs)
        return counted


@pytest.mark.parametrize("cause", STORE_CAUSES)
@pytest.mark.parametrize("driver", SURFACE_DRIVERS)
def test_the_dump_stamps_from_its_own_read(dws, driver, cause):
    """Row 11: an amended decision in the scope shows `amended_by`, and the dump made
    no modifiers read — its stamps came from `get_active_decisions`' hydration."""
    _scoped(dws)
    spies: List[_Spy] = []

    def wrap(store):
        spies.append(_Spy(store))
        return spies[-1]

    resp = drive(driver, dws, "nothing lexical matches this", cause=cause,
                 scope="caching", wrap=wrap)
    assert_cause(driver, cause, resp)
    if DRIVERS[driver][3] == "text":
        assert "cache-amend" in resp  # the modifier marker names the amender
    else:
        assert resp["active_decisions"][0]["amended_by"] == ["cache-amend"]
    calls = spies[0].calls
    assert "get_active_decisions" in calls
    assert "get_modifiers" not in calls and "get_modifiers_map" not in calls


def _raise(*_a, **_k):
    raise RuntimeError("the hydrating read fell over")


@pytest.mark.parametrize("empty", [False, True], ids=["raising", "empty"])
@pytest.mark.parametrize("driver", SURFACE_DRIVERS)
def test_a_raising_or_empty_dump_takes_the_lexical_exit(dws, driver, empty):
    """Row 11: `dump is not None` is the partition — a fault and an empty scope both
    reach the lexical exit, never an empty answer that looks undegraded."""
    _scoped(dws)
    scope = "no-such-scope" if empty else "caching"
    wrap = None if empty else (lambda store: _Spy(store, get_active_decisions=_raise))
    resp = drive(driver, dws, "write-through", cause="no-providers", scope=scope, wrap=wrap)
    assert_cause(driver, "no-providers", resp)
    if DRIVERS[driver][3] == "text":
        assert "deterministic text match" in resp
    else:
        assert resp["degraded"] == "lexical"
    assert listed(driver, resp) == ["cache-amend", "cache-strategy"]  # newest first


def test_scope_dump_is_none_exactly_when_there_is_no_dump(dws):
    """Row 11 at the leaf: None on a raise and on an empty scope, so `dump is not
    None` is the partition (the boundaries re-test emptiness too, which is why the
    surface rows above cannot see a `ScopeDump([], 0)`; this row can)."""
    from mitos.degraded import ScopeDump, scope_dump
    _scoped(dws)
    store = GraphStore(dws.db_path, read_only=True)
    assert scope_dump(store, "no-such-scope", limit=None, full_top=None) is None
    assert scope_dump(_Spy(store, get_active_decisions=_raise), "caching",
                      limit=None, full_top=None) is None
    dump = scope_dump(store, "caching", limit=None, full_top=None)
    assert isinstance(dump, ScopeDump) and dump.total == 1
    assert [d["slug"] for d in dump.decisions] == ["cache-strategy"]


@pytest.mark.parametrize("driver", ["mcp-surface", "cli-surface-json"])
def test_the_dump_note_reads_dump_sentence_then_omission_then_withheld(dws, driver):
    """Row 11: the dump's order — dump sentence, omission, section clause (5d),
    withheld (its decisions cannot be unstamped; a section owner's stamp fault has
    its own row)."""
    _scoped(dws)
    resp = drive(driver, dws, "nothing lexical matches this", cause="no-providers",
                 scope="caching", full_top=0)
    note = resp["note"]
    surface = DRIVERS[driver][2]
    withheld = withheld_clause(1, surface=surface, config=dws)
    assert note.startswith("Semantic recall unavailable (embeddings/Qdrant unavailable) "
                           "— showing the one active decision in scope 'caching'")
    assert note.endswith(f"(pure graph read). {_OMISSION} {_SECTION} {withheld}")


# --------------------------------------------------------------------------- #
# The lexical exit's stamp fault, stated on every surface; its note order
# --------------------------------------------------------------------------- #

def _modifiers_fault(*_a, **_k):
    raise DatabaseError("Failed to read modifiers: database is locked")


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_a_lexical_stamp_fault_is_stated_last(dws, driver):
    """Row 7 through the composer: the stamps cost alone, stated after the omission,
    section (5d) and withheld clauses (the ranked order's tail); the state filter
    still ran."""
    _scoped(dws)
    resp = drive(driver, dws, "write-through", cause="no-providers", full_top=0,
                 wrap=lambda s: _Spy(s, get_modifiers_map=_modifiers_fault))
    surface = DRIVERS[driver][2]
    unread = stamps_unread_clause(surface=surface, config=dws)
    withheld = withheld_clause(2, surface=surface, config=dws)
    if DRIVERS[driver][3] == "text":
        assert f"{_OMISSION} {_SECTION} {withheld} {unread}" in resp
        return
    assert resp["stamps_unread"] is True and "stamps_unavailable" not in resp
    assert resp["note"].endswith(f"{_OMISSION} {_SECTION} {withheld} {unread}")
    assert all(m["state"] == "active" for m in resp["matches"])
    assert not any("amended_by" in m for m in resp["matches"])


# --------------------------------------------------------------------------- #
# Row 12 — CLI ⇄ MCP parity on the degraded envelope
# --------------------------------------------------------------------------- #

def _without_reason(env: Dict[str, Any]) -> Dict[str, Any]:
    """The envelope with its per-surface cause phrase replaced in the note."""
    out = dict(env)
    reason = env.get("degraded_reason")
    if reason:
        out["note"] = env["note"].replace(reason, "<reason>")
        out["degraded_reason"] = "<reason>"
    return out


@pytest.mark.parametrize("cause", CAUSES)
@pytest.mark.parametrize("verb", ["surface", "query"])
@pytest.mark.parametrize("full_top", [None, 0])
def test_cli_and_mcp_degraded_envelopes_agree(dws, verb, cause, full_top):
    """Row 12: equal key sets and values on CLI `--json` and MCP, except the note's
    per-surface pointers and a cause phrase that carries a recovery recipe; MCP names
    no `mitos ` command."""
    _scoped(dws)
    cli_env = drive(f"cli-{verb}-json", dws, "write-through", cause=cause,
                    full_top=full_top)
    mcp_env = drive(f"mcp-{verb}", dws, "write-through", cause=cause, full_top=full_top)
    assert_cause(f"mcp-{verb}", cause, mcp_env)
    assert set(cli_env) == set(mcp_env)
    assert "mitos " not in mcp_env["note"] and "mitos " not in mcp_env["degraded_reason"]
    if cause in ("no-providers", "mid-query"):
        assert cli_env["degraded_reason"] == mcp_env["degraded_reason"]
    cli_n, mcp_n = _without_reason(cli_env), _without_reason(mcp_env)
    if full_top == 0:
        n = cli_env["rejected_paths_withheld"]
        cli_n["note"] = cli_n["note"].replace(withheld_clause(n, surface="cli", config=dws), "<w>")
        mcp_n["note"] = mcp_n["note"].replace(withheld_clause(n, surface="mcp", config=dws), "<w>")
    assert cli_n == mcp_n


@pytest.mark.parametrize("cause", STORE_CAUSES)
@pytest.mark.parametrize("full_top", [None, 0])
def test_cli_and_mcp_scope_dumps_agree(dws, cause, full_top):
    """Row 12, the dump exit: `degraded.dump_section` composes the dump's
    `not_computed` and section once (5d; each boundary used to set the key itself),
    so the two envelopes must agree on every key and every value but the note, whose
    complete-set hint and withheld clause are per-surface; MCP names no `mitos `."""
    _scoped(dws)
    kw = dict(cause=cause, scope="caching", full_top=full_top)
    cli_env = drive("cli-surface-json", dws, "nothing lexical matches this", **kw)
    mcp_env = drive("mcp-surface", dws, "nothing lexical matches this", **kw)
    assert_cause("mcp-surface", cause, mcp_env)
    assert set(cli_env) == set(mcp_env)
    assert "not_computed" in mcp_env and "degraded" not in mcp_env
    assert "mitos " not in mcp_env["note"]
    assert {k: v for k, v in cli_env.items() if k != "note"} == {
        k: v for k, v in mcp_env.items() if k != "note"}
    for env in (cli_env, mcp_env):
        assert _OMISSION in env["note"]
