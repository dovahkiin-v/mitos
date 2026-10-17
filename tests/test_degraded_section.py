"""Phase 5d: degraded reads find rejected options by their words.

When semantic recall is down, every degraded exit — the lexical envelope on both
verbs and ``surface``'s unranked scope dump — carries a rejection section matched
lexically over **parsed** items, beside the decisions, labelled
``rejection_section: "lexical"``, with no score and no band. The rows here pin, over
2g's drivers (MCP ``surface_decisions`` / ``query_decisions``, ``mitos surface`` /
``mitos query`` on ``--json`` and text, imported from ``test_degraded_envelope``):

- rows 1–2: the section on the lexical exit and on the dump;
- rows 3–4: a prose field is one item; items are matched, never the raw field;
- row 5: the section bound, independent of ``limit``;
- row 6: the window join by content-hash id, unsynced decisions included;
- row 7: retired owners dropped and named, unfiltered when no state was read;
- row 10: the withheld count over decisions on both exits;
- rows 11–12: the dump's and the lexical envelope's faults;
- rows 13–14: the reads and the one corpus pass;
- row 15: CLI ``--json`` ⇄ MCP parity with a non-empty section; the key order.

Rows 8 and 9 (the three tokens, the section clause in its slot, the ``lexical``
token on degraded exits only) live in ``test_degraded_envelope``; row 16 (the
partial-list arm raises) in ``test_surface_confidence``.
"""

import json
import re
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

import mitos.lexical as lexical
from mitos.errors import DatabaseError
from mitos.recall import (DEGRADED_NOT_COMPUTED, REJECTION_SECTION_BOUND,
                          degraded_omission_clause, degraded_section_clause,
                          stamps_unread_clause, withheld_clause)

from test_degraded_envelope import (CAUSES, DRIVERS, JSON_DRIVERS, STORE_CAUSES,
                                    SURFACE_DRIVERS, _Spy, _without_reason,
                                    assert_cause, block, commit_text, drive, dws,
                                    listed, offline, write_archive,  # noqa: F401
                                    write_buffer)

_OMISSION = degraded_omission_clause()
_SECTION = degraded_section_clause(computed=True, retired_dropped=True)
_HEADING = "Rejected options sharing this query's words (most words first):"
_TEXT_DRIVERS = [d for d, spec in DRIVERS.items() if spec[3] == "text"]


def _section(driver: str, resp: Any) -> List[Dict[str, Any]]:
    return resp.get("rejection_matches", [])


def _is_text(driver: str) -> bool:
    return DRIVERS[driver][3] == "text"


def _decisions(driver: str, resp: Any) -> List[str]:
    """The decision slugs an answer lists. On text, the ``N. slug`` lines whose slug is
    a decision's: a multi-line ``rejected_paths`` printed whole leaves its own
    ``2. …`` lines at column 0 (the whole-field print 5e replaces), which
    ``text_slugs`` would also count."""
    if not _is_text(driver):
        return listed(driver, resp)
    return re.findall(r"^\d+\. ([a-z][a-z0-9-]*)(?:  \(score|$)", resp, flags=re.MULTILINE)


# --------------------------------------------------------------------------- #
# Row 1 — the section on the lexical exit
# --------------------------------------------------------------------------- #

def _two_owners(config) -> Dict[str, str]:
    """A decision whose axiom matches "redis", and one whose rejected item does."""
    blocks = [block("queue-redis", "Use redis for the job queue.",
                    rejected="1. Kafka — too heavy for one consumer."),
              block("session-store", "Sessions live in Postgres.",
                    rejected="1. Redis sessions — volatile under eviction.\n"
                             "2. Cookies only — size limits.")]
    return commit_text(config, write_buffer(config, blocks))


@pytest.mark.parametrize("cause", CAUSES)
@pytest.mark.parametrize("driver", list(DRIVERS))
def test_the_lexical_exit_carries_the_section(dws, driver, cause):
    """Row 1 (T8, W29): the decision match and, beside it, the rejected item found
    by its words — `here` with its owner's Letter, no score, matched terms named."""
    ids = _two_owners(dws)
    resp = drive(driver, dws, "redis", cause=cause)
    assert_cause(driver, cause, resp)
    assert _decisions(driver, resp) == ["queue-redis"]
    if _is_text(driver):
        assert _HEADING in resp
        state = "" if cause == "pre-v1a" else "  (active)"
        assert (f"  session-store{state}  item 1: 1. Redis sessions — volatile under "
                "eviction.") in resp
        assert "    matched: redis" in resp
        assert "    Decided:  Sessions live in Postgres." in resp
        return
    assert resp["degraded"] == "lexical" and resp["rejection_section"] == "lexical"
    (match,) = _section(driver, resp)
    assert match["slug"] == "session-store" and match["id"] == ids["session-store"]
    assert match["ordinal"] == 1 and match["letter"] == "here"
    assert match["item"] == "1. Redis sessions — volatile under eviction."
    assert match["matched_terms"] == ["redis"]
    assert "score" not in match
    assert match["axiom"] == "Sessions live in Postgres."
    assert match["rejected_paths"].startswith("1. Redis sessions")
    assert ("state" in match) == (cause != "pre-v1a")


@pytest.mark.parametrize("driver", _TEXT_DRIVERS)
def test_a_degraded_answer_with_no_section_match_prints_no_section(dws, driver):
    """The lexical printer is a no-op on no matches: no heading, no extra line."""
    _two_owners(dws)
    out = drive(driver, dws, "postgres", cause="no-providers")
    assert _decisions(driver, out) == ["session-store"]
    assert _HEADING not in out
    assert out.endswith("size limits.\n\n") and not out.endswith("\n\n\n")


# --------------------------------------------------------------------------- #
# Row 2 — the section on the dump
# --------------------------------------------------------------------------- #

def _dump_world(config) -> Dict[str, str]:
    """Scope 'caching' holds cache-strategy (its item 1 matches "write-back"); a
    decision outside the scope rejects a write-back buffer."""
    blocks = [block("queue-buffer", "Buffer queue writes in memory.", scope="queue",
                    rejected="1. A write-back buffer — loses jobs on crash."),
              block("cache-strategy", "Use a write-through cache.", scope="caching",
                    rejected="1. Write-back cache — loses writes on crash.\n"
                             "2. No cache — too slow.")]
    return commit_text(config, write_buffer(config, blocks))


@pytest.mark.parametrize("cause", STORE_CAUSES)
@pytest.mark.parametrize("driver", SURFACE_DRIVERS)
def test_the_scope_dump_carries_the_section(dws, driver, cause):
    """Row 2 (W29): the dump's own decision is `window` with its rank; the owner
    outside it is `here`. No `degraded` key (the dump is a graph read), the
    section labelled lexical, three tokens."""
    ids = _dump_world(dws)
    resp = drive(driver, dws, "write-back", cause=cause, scope="caching")
    assert_cause(driver, cause, resp)
    assert _decisions(driver, resp) == ["cache-strategy"]
    if _is_text(driver):
        assert _HEADING in resp
        assert "  cache-strategy  (active)  item 1: 1. Write-back cache" in resp
        assert "    see rank 1 above" in resp
        assert "  queue-buffer  (active)  item 1: 1. A write-back buffer" in resp
        assert f"{_OMISSION} {_SECTION}" in resp
        return
    assert "degraded" not in resp
    assert resp["rejection_section"] == "lexical"
    assert resp["not_computed"] == list(DEGRADED_NOT_COMPUTED)
    by_slug = {m["slug"]: m for m in resp["rejection_matches"]}
    assert set(by_slug) == {"cache-strategy", "queue-buffer"}
    window = by_slug["cache-strategy"]
    assert window["letter"] == "window" and window["rank"] == 1
    assert "axiom" not in window and window["id"] == ids["cache-strategy"]
    assert window["state"] == "active"
    here = by_slug["queue-buffer"]
    assert here["letter"] == "here" and here["axiom"] == "Buffer queue writes in memory."
    assert all("score" not in m for m in resp["rejection_matches"])


# --------------------------------------------------------------------------- #
# Rows 3–4 — items, never the raw field
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_a_prose_field_is_matched_as_one_item(dws, driver):
    """Row 3: a field with no numbering is one item, matched and served whole."""
    prose = "A redis queue was rejected because it loses jobs on restart."
    commit_text(dws, write_buffer(dws, [block("jobs", "Jobs run from Postgres.",
                                              rejected=prose)]))
    resp = drive(driver, dws, "redis restart", cause="no-providers")
    (match,) = resp["rejection_matches"]
    assert match["ordinal"] == 1 and match["item"] == prose
    assert match["matched_terms"] == ["redis", "restart"]


@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_items_are_matched_never_the_raw_field(dws, driver):
    """Row 4: a term in item 2 only serves item 2 only; two terms in two items serve
    each item with its own term — a raw-field read would serve one item holding
    both. (A term cannot straddle two items: terms split on whitespace, items on
    lines.)"""
    field = "1. Postgres queue — too heavy.\n2. Kafka cluster — operational burden."
    commit_text(dws, write_buffer(dws, [block("jobs", "Jobs run in-process.",
                                              rejected=field)]))
    (match,) = drive(driver, dws, "kafka", cause="no-providers")["rejection_matches"]
    assert match["ordinal"] == 2 and match["item"] == "2. Kafka cluster — operational burden."
    both = drive(driver, dws, "kafka postgres", cause="no-providers")["rejection_matches"]
    assert [(m["ordinal"], m["matched_terms"]) for m in both] == [
        (1, ["postgres"]), (2, ["kafka"])]


# --------------------------------------------------------------------------- #
# Row 5 — the bound
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_the_section_holds_the_bound_whatever_the_limit(dws, driver):
    """Row 5: five matching items across three owners — exactly the section bound,
    in the leaf's order, the same at limit 1 and 50."""
    blocks = [block("owner-a", "First owner.",
                    rejected="1. Redis one.\n2. Redis two."),
              block("owner-b", "Second owner.",
                    rejected="1. Redis three.\n2. Redis four."),
              block("owner-c", "Third owner.", rejected="Redis five.")]
    commit_text(dws, write_buffer(dws, blocks))
    one = drive(driver, dws, "redis", cause="no-providers", limit=1)["rejection_matches"]
    fifty = drive(driver, dws, "redis", cause="no-providers", limit=50)["rejection_matches"]
    assert len(one) == REJECTION_SECTION_BOUND == 3
    assert one == fifty
    assert [(m["slug"], m["ordinal"]) for m in one] == [
        ("owner-a", 1), ("owner-a", 2), ("owner-b", 1)]
    assert [m["letter"] for m in one] == ["here", "section", "here"]


# --------------------------------------------------------------------------- #
# Row 6 — the window join, by id
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_an_owner_in_the_list_is_window_never_a_second_letter(dws, driver):
    """Row 6 (G4): an owner among the decision matches is `window` with its rank —
    an unsynced decision too, joined by its computed id, with no state."""
    synced = block("queue-redis", "Use redis for the job queue.",
                   rejected="1. Redis streams — immature.")
    unsynced = block("cache-redis", "Cache in redis as well.",
                     rejected="Redis cluster mode — too much to run.")
    write_buffer(dws, [unsynced, synced])
    commit_text(dws, synced)
    resp = drive(driver, dws, "redis", cause="no-providers")
    assert listed(driver, resp) == ["cache-redis", "queue-redis"]
    by_slug = {m["slug"]: m for m in resp["rejection_matches"]}
    for slug, rank in (("cache-redis", 1), ("queue-redis", 2)):
        assert by_slug[slug]["letter"] == "window" and by_slug[slug]["rank"] == rank
        assert "axiom" not in by_slug[slug]
    assert "state" not in by_slug["cache-redis"]
    assert by_slug["queue-redis"]["state"] == "active"


# --------------------------------------------------------------------------- #
# Row 7 — retired owners
# --------------------------------------------------------------------------- #

def _retired_world(config) -> None:
    blocks = [block("new-queue", "Run the queue on Postgres.", supersedes="old-queue",
                    rejected="1. Kafka — too heavy."),
              block("old-queue", "Run the queue in memory.",
                    rejected="1. Redis queue — lost jobs on restart.")]
    commit_text(config, write_buffer(config, blocks))


@pytest.mark.parametrize("cause", CAUSES)
@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_retired_owners_are_dropped_and_the_answer_says_so(dws, driver, cause):
    """Row 7: a superseded owner's items are absent and the clause names the drop;
    with no state read (pre-V1a) they stay, unfiltered, and the clause says only
    what was searched."""
    _retired_world(dws)
    resp = drive(driver, dws, "redis", cause=cause)
    assert_cause(driver, cause, resp)
    slugs = [m["slug"] for m in resp.get("rejection_matches", [])]
    if cause == "pre-v1a":
        assert slugs == ["old-queue"] and resp["stamps_unavailable"] is True
        assert degraded_section_clause(computed=True, retired_dropped=False) in resp["note"]
        assert "those of retired decisions" not in resp["note"]
    else:
        assert slugs == []
        assert f"{_OMISSION} {_SECTION}" in resp["note"]


# --------------------------------------------------------------------------- #
# Row 10 — withheld over decisions
# --------------------------------------------------------------------------- #

def _withheld_world(config) -> None:
    """Two thinned decisions in scope 'w': one whose only item matches "redis" (the
    section shows all of it), one whose field matches nothing."""
    blocks = [block("redis-covered", "Keep redis for locks.", scope="w",
                    rejected="Redis cluster mode was rejected."),
              block("redis-bare", "Keep redis for counters.", scope="w",
                    rejected="Memcached was rejected.")]
    commit_text(config, write_buffer(config, blocks))


@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_the_lexical_withheld_count_exempts_what_the_section_shows(dws, driver):
    """Row 10, lexical: `full_top=0` thins both; the covered one is exempt."""
    _withheld_world(dws)
    resp = drive(driver, dws, "redis", cause="no-providers", full_top=0)
    assert listed(driver, resp) == ["redis-covered", "redis-bare"]
    assert [m["slug"] for m in resp["rejection_matches"]] == ["redis-covered"]
    assert resp["rejected_paths_withheld"] == 1
    surface = DRIVERS[driver][2]
    assert resp["note"].endswith(
        f"{_SECTION} {withheld_clause(1, surface=surface, config=dws)}")


@pytest.mark.parametrize("driver", ["mcp-surface", "cli-surface-json"])
def test_the_dump_withheld_count_exempts_what_the_section_shows(dws, driver):
    """Row 10, the dump: the same two decisions, listed by the dump."""
    _withheld_world(dws)
    resp = drive(driver, dws, "redis", cause="no-providers", scope="w", full_top=0)
    assert "degraded" not in resp
    assert sorted(listed(driver, resp)) == ["redis-bare", "redis-covered"]
    assert resp["rejected_paths_withheld"] == 1


# --------------------------------------------------------------------------- #
# Row 11 — the dump's faults
# --------------------------------------------------------------------------- #

class _ArgSpy(_Spy):
    """`_Spy`, also recording the ids each state and stamp read was asked."""

    def __init__(self, store: Any, **replace: Any) -> None:
        super().__init__(store, **replace)
        self.asked: List[tuple] = []

    def __getattr__(self, name: str) -> Any:
        target = super().__getattr__(name)
        if name not in ("get_node_states", "get_modifiers_map"):
            return target

        def recording(ids, *args, **kwargs):
            self.asked.append((name, list(ids)))
            return target(ids, *args, **kwargs)
        return recording


def _states_fault(*_a, **_k):
    raise DatabaseError("Failed to read node states: database is locked")


def _stamps_fault(*_a, **_k):
    raise DatabaseError("Failed to read modifiers: database is locked")


def _fault_blocks() -> List[str]:
    return [block("buffer-note", "Note the buffer's limits.", scope="other",
                    amends="queue-buffer", rejected="Nothing else."),
              block("cache-note", "Note the cache's limits.", scope="other",
                    amends="cache-strategy", rejected="Nothing else."),
              block("queue-buffer", "Buffer queue writes in memory.", scope="queue",
                    rejected="1. A write-back buffer — loses jobs on crash."),
              block("cache-strategy", "Use a write-through cache.", scope="caching",
                    rejected="1. Write-back cache — loses writes on crash.")]


def _fault_world(config) -> Dict[str, str]:
    """The dump world, with the dump's decision amended (its stamps come from its
    hydration) and the outside owner amended too (its stamps cost a read)."""
    return commit_text(config, write_buffer(config, _fault_blocks()))


def _drive_spied(driver, config, query, **kw):
    spies: List[_ArgSpy] = []
    replace = kw.pop("replace", {})

    def wrap(store):
        spies.append(_ArgSpy(store, **replace))
        return spies[-1]
    resp = drive(driver, config, query, cause="no-providers", wrap=wrap, **kw)
    return resp, spies[0]


@pytest.mark.parametrize("driver", SURFACE_DRIVERS)
def test_a_dump_state_fault_leaves_the_section_not_searched(dws, driver):
    """Row 11 (D3): the outside owner's state read raises → `unavailable`, no
    matches, the fourth token back, the not-searched clause; the dump's decisions
    whole and stamped."""
    _fault_world(dws)
    resp, spy = _drive_spied(driver, dws, "write-back", scope="caching",
                             replace={"get_node_states": _states_fault})
    assert "get_node_states" in spy.calls
    not_searched = degraded_section_clause(computed=False, retired_dropped=False)
    if _is_text(driver):
        assert _HEADING not in resp and f"{_OMISSION} {not_searched}" in resp
        assert "cache-note" in resp  # the dump decision's own stamp marker
        return
    assert resp["rejection_section"] == "unavailable"
    assert "rejection_matches" not in resp and "stamps_unavailable" not in resp
    assert resp["not_computed"] == [*DEGRADED_NOT_COMPUTED, "rejection_matches"]
    assert f"{_OMISSION} {not_searched}" in resp["note"]
    (decision,) = resp["active_decisions"]
    assert decision["amended_by"] == ["cache-note"] and "rejected_paths" in decision


@pytest.mark.parametrize("driver", ["mcp-surface", "cli-surface-json"])
def test_a_dump_stamp_fault_costs_the_section_its_stamps(dws, driver):
    """Row 11: the outside owner's stamp read raises → `stamps_unread`, its clause
    last; the outside match unstamped, the dump's decision (and its window match)
    stamped from its hydration."""
    _fault_world(dws)
    resp, spy = _drive_spied(driver, dws, "write-back", scope="caching",
                             replace={"get_modifiers_map": _stamps_fault})
    assert "get_modifiers_map" in spy.calls
    surface = DRIVERS[driver][2]
    assert resp["stamps_unread"] is True
    assert resp["note"].endswith(
        f"{_SECTION} {stamps_unread_clause(surface=surface, config=dws)}")
    by_slug = {m["slug"]: m for m in resp["rejection_matches"]}
    assert "amended_by" not in by_slug["queue-buffer"]
    assert by_slug["cache-strategy"]["amended_by"] == ["cache-note"]
    assert resp["active_decisions"][0]["amended_by"] == ["cache-note"]


@pytest.mark.parametrize("driver", ["mcp-surface", "cli-surface-json"])
def test_an_unreadable_corpus_file_on_the_dump_is_named(dws, driver):
    """Row 11: the dump never read the corpus before, so its section names the file
    it could not read; the section is still computed over the rest."""
    _dump_world(dws)
    write_archive(dws, "2026-Q1.md", ["placeholder"])
    import os
    with open(os.path.join(dws.archive_dir, "2026-Q1.md"), "wb") as f:
        f.write(b"\xff\xfe not utf-8 \xff")
    resp = drive(driver, dws, "write-back", cause="no-providers", scope="caching")
    assert resp["unread_files"] == ["2026-Q1.md"]
    assert f"{_SECTION} 1 corpus file(s) could not be read: 2026-Q1.md." in resp["note"]
    assert resp["rejection_section"] == "lexical" and resp["rejection_matches"]


# --------------------------------------------------------------------------- #
# Row 12 — the lexical envelope's faults
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_a_lexical_state_fault_serves_the_section_unfiltered(dws, driver):
    """Row 12: on the lexical envelope a state fault is `stamps_unavailable` — the
    retired owner's item is served like the decisions, never `unavailable`."""
    _retired_world(dws)
    resp, _spy = _drive_spied(driver, dws, "redis",
                              replace={"get_node_states": _states_fault})
    assert resp["stamps_unavailable"] is True
    assert resp["rejection_section"] == "lexical"
    assert [m["slug"] for m in resp["rejection_matches"]] == ["old-queue"]


@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_a_lexical_stamp_fault_unstamps_both_lists_once(dws, driver):
    """Row 12: one stamp read, one `stamps_unread`, both lists unstamped."""
    blocks = [block("redis-note", "Note redis's limits.", amends="redis-locks",
                    rejected="Nothing else."),
              block("redis-locks", "Keep redis for locks.",
                    rejected="Redis cluster mode was rejected.")]
    commit_text(dws, write_buffer(dws, blocks))
    healthy = drive(driver, dws, "locks", cause="no-providers")
    assert healthy["matches"][0]["amended_by"] == ["redis-note"]
    resp, spy = _drive_spied(driver, dws, "redis cluster",
                             replace={"get_modifiers_map": _stamps_fault})
    assert spy.calls.count("get_modifiers_map") == 1
    assert resp["stamps_unread"] is True and resp["note"].count("Modifier stamps") == 1
    assert not any("amended_by" in m for m in resp["matches"] + resp["rejection_matches"])


# --------------------------------------------------------------------------- #
# Rows 13–14 — the reads and the one corpus pass
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("query", ["redis", "kafka", "nothing matches here"])
@pytest.mark.parametrize("driver", JSON_DRIVERS)
def test_the_lexical_envelope_reads_states_once_and_stamps_once(dws, driver, query):
    """Row 13: with or without a section, matched or not — one of each read; the
    state read covers decision candidates and item owners in one call."""
    ids = _two_owners(dws)
    resp, spy = _drive_spied(driver, dws, query)
    assert [c for c in spy.calls if c in ("get_node_states", "get_modifiers_map")] == [
        "get_node_states", "get_modifiers_map"]
    if query == "redis":
        (states_ids,) = [ids_ for name, ids_ in spy.asked if name == "get_node_states"]
        assert set(states_ids) == {ids["queue-redis"], ids["session-store"]}
        assert resp["rejection_matches"]


@pytest.mark.parametrize("driver", SURFACE_DRIVERS)
def test_the_dump_reads_only_what_its_section_needs(dws, driver):
    """Row 13 (G2): the dump's own read once; the section's reads once each when an
    owner lies outside the dump, carrying no dump id, and never with `[]`."""
    ids = _dump_world(dws)
    _resp, spy = _drive_spied(driver, dws, "write-back", scope="caching")
    assert spy.calls.count("get_active_decisions") == 1
    assert spy.calls.count("get_node_states") == 1
    assert spy.calls.count("get_modifiers_map") == 1
    for _name, asked in spy.asked:
        assert asked == [ids["queue-buffer"]]
        assert ids["cache-strategy"] not in asked


@pytest.mark.parametrize("query", ["write-through", "nothing matches here"])
@pytest.mark.parametrize("driver", SURFACE_DRIVERS)
def test_a_dump_whose_section_needs_no_read_makes_none(dws, driver, query):
    """Row 13: every matched owner in the dump, or no item match at all → zero
    section reads (the gate, so `_Spy` never sees an empty-list call)."""
    blocks = [block("cache-strategy", "Use a write-through cache.", scope="caching",
                    rejected="1. A write-through queue — too slow.")]
    commit_text(dws, write_buffer(dws, blocks))
    resp, spy = _drive_spied(driver, dws, query, scope="caching")
    assert spy.calls.count("get_active_decisions") == 1
    assert "get_node_states" not in spy.calls and "get_modifiers_map" not in spy.calls
    if query == "write-through" and not _is_text(driver):
        assert [m["letter"] for m in resp["rejection_matches"]] == ["window"]


@pytest.mark.parametrize("scope", [None, "caching"], ids=["lexical", "dump"])
@pytest.mark.parametrize("driver", ["mcp-surface", "cli-surface-json"])
def test_a_degraded_answer_reads_the_corpus_once(dws, driver, scope):
    """Row 14 (D1): one corpus pass per lexical envelope and per dump section."""
    _dump_world(dws)
    real = lexical._corpus_entries
    passes: List[int] = []

    def counted(*args, **kwargs):
        passes.append(1)
        return real(*args, **kwargs)
    with patch.object(lexical, "_corpus_entries", counted):
        resp = drive(driver, dws, "write-back", cause="no-providers", scope=scope)
    assert ("degraded" in resp) == (scope is None)
    assert resp["rejection_matches"]
    assert len(passes) == 1


# --------------------------------------------------------------------------- #
# Row 15 — parity with a section, and the key order
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("full_top", [None, 0])
@pytest.mark.parametrize("verb", ["surface", "query"])
def test_cli_and_mcp_agree_on_a_lexical_section(dws, verb, full_top):
    """Row 15: CLI `--json` ⇄ MCP, a non-empty section; every value but the note's
    per-surface pointers, and the same key order."""
    _two_owners(dws)
    cli_env = drive(f"cli-{verb}-json", dws, "redis", cause="no-providers",
                    full_top=full_top)
    mcp_env = drive(f"mcp-{verb}", dws, "redis", cause="no-providers", full_top=full_top)
    assert mcp_env["rejection_matches"]
    assert list(cli_env) == list(mcp_env)
    cli_n, mcp_n = _without_reason(cli_env), _without_reason(mcp_env)
    if "rejected_paths_withheld" in cli_env:
        n = cli_env["rejected_paths_withheld"]
        cli_n["note"] = cli_n["note"].replace(withheld_clause(n, surface="cli", config=dws), "<w>")
        mcp_n["note"] = mcp_n["note"].replace(withheld_clause(n, surface="mcp", config=dws), "<w>")
    assert cli_n == mcp_n
    assert "mitos " not in mcp_env["note"]


@pytest.mark.parametrize("full_top", [None, 0])
def test_cli_and_mcp_agree_on_a_dump_section(dws, full_top):
    """Row 15, the dump: the section composed once; only the note differs."""
    _dump_world(dws)
    kw = dict(cause="no-providers", scope="caching", full_top=full_top)
    cli_env = drive("cli-surface-json", dws, "write-back", **kw)
    mcp_env = drive("mcp-surface", dws, "write-back", **kw)
    assert mcp_env["rejection_matches"]
    assert list(cli_env) == list(mcp_env)
    assert {k: v for k, v in cli_env.items() if k != "note"} == {
        k: v for k, v in mcp_env.items() if k != "note"}


def test_the_lexical_envelope_key_order(dws):
    """The section's keys follow `not_computed` and precede the withheld count, the
    open questions and the check notice."""
    _two_owners(dws)
    resp = drive("mcp-surface", dws, "redis", cause="no-providers", full_top=0,
                 scope="nowhere")
    keys = list(resp)
    assert keys.index("note") < keys.index("not_computed")
    assert (keys.index("not_computed") < keys.index("rejection_matches")
            < keys.index("rejection_section") < keys.index("rejected_paths_withheld")
            < keys.index("open_questions"))


def test_the_dump_key_order(dws):
    """The dump's degraded keys land after its note, in the composer's order, then
    the withheld count (the boundary's `stamps_unread` reassignment keeps the
    composer's slot, before it — scout 6)."""
    _fault_world(dws)
    write_buffer(dws, [block("cache-ttl", "Expire cache entries hourly.", scope="caching",
                             rejected="Never expiring.")] + _fault_blocks())
    commit_text(dws, block("cache-ttl", "Expire cache entries hourly.", scope="caching",
                           rejected="Never expiring."))
    resp, _spy = _drive_spied("mcp-surface", dws, "write-back", scope="caching",
                              full_top=0, replace={"get_modifiers_map": _stamps_fault})
    assert resp["rejected_paths_withheld"] == 1  # cache-ttl: thinned, no section match
    keys = list(resp)
    assert keys[keys.index("note") + 1:] == [
        "not_computed", "rejection_matches", "rejection_section", "stamps_unread",
        "rejected_paths_withheld"]
    assert json.dumps(resp)  # encodable


def test_a_drifted_dump_decision_is_window_and_says_drifted(dws):
    """Row 2's state cell: a dump decision's state comes from its own hydration
    (`is_drifted`), never a read — drift annotates, it does not retire."""
    from mitos.store import GraphStore
    ids = _dump_world(dws)
    GraphStore(dws.db_path).write_signal(ids["cache-strategy"], "drifted")
    resp, spy = _drive_spied("mcp-surface", dws, "write-back", scope="caching")
    by_slug = {m["slug"]: m for m in resp["rejection_matches"]}
    assert by_slug["cache-strategy"]["letter"] == "window"
    assert by_slug["cache-strategy"]["state"] == "drifted"
    for _name, asked in spy.asked:
        assert ids["cache-strategy"] not in asked
