"""Phase 2g: the lexical fallback resolves each corpus block by its own content-hash id.

When semantic recall is down, ``lexical.lexical_fallback`` term-matches the markdown
corpus. A matched block is named by the id the commit minted for it — parsed with the
commit's own tokenizer (``parse_entry_stream``) and hashed by
``identity.compute_node_id`` over its axiom and mechanisms — never by its slug, which
a same-slug supersession shares and a rename moves (ADR
``lexical-fallback-resolves-corpus-entries-by-content-hash-id-never-by-slug``). The
answer makes two batched reads: ``GraphStore.get_node_states`` (filter and state) and
``get_modifiers_map`` (stamps).

Rows (plan p2g §10):

1. agreement — the id the lexical path asks about is the committed node's id, an HTML
   comment and a multi-line axiom included;
2. a same-slug lineage gives the successor, once, on every surface;
3. a renamed, unsynced block still resolves, live or retired;
4. an unsynced axiom edit has no node and no state; the older block is not consulted;
5. one call each to the two reads per answer, whatever the match count;
6. ``get_node_states`` agrees with ``get_node_state``, is absent for an unknown id,
   dedupes, chunks, and wraps its fault;
7. a state-read fault reads as no graph; a stamp fault costs the stamps alone; a
   store missing the method raises.
"""

import sqlite3
from typing import Any, Dict, List

import pytest

import mitos.store as store_module
from mitos.divergence import _corpus_files
from mitos.errors import DatabaseError
from mitos.lexical import lexical_fallback
from mitos.store import GraphStore

from test_degraded_envelope import (DRIVERS, _Spy, block, commit_text, drive,  # noqa: F401
                                    dws, listed, offline, write_archive, write_buffer)
from test_point_identity import _commit, _node_reads, _record


def _lexical(config, query: str, store: Any, **kw: Any) -> Dict[str, Any]:
    return lexical_fallback(query, corpus_paths=_corpus_files(config), reason="test",
                            store=store, **kw)


def _states_asked(spy: _Spy, store: GraphStore) -> List[List[str]]:
    """Wraps the real ``get_node_states`` so a row can read the ids it was asked."""
    asked: List[List[str]] = []

    def recording(ids):
        asked.append(list(ids))
        return store.get_node_states(ids)
    spy._replace["get_node_states"] = recording
    return asked


# --------------------------------------------------------------------------- #
# Row 1 — agreement: the lexical id is the commit's id
# --------------------------------------------------------------------------- #

_AGREEMENT_BLOCKS = [
    block("cache-commented", "The cache <!-- note --> strategy decides where results live."),
    block("cache-multiline", "The cache spans two lines\nof axiom text, continued."),
    block("cache-mechanised", "The cache is keyed by content hash.",
          mechanisms="Content-Hash, sqlite"),
    block("cache-scoped", "The cache is warmed at start.", scope="Cache, Ops"),
    block("cache-plain", "A plain cache axiom."),
]


def test_the_lexical_id_is_the_id_the_commit_minted(dws):
    """Row 1: every block's id, as the lexical path computes it, names its committed
    node — the comment-bearing one included, which the prototype parser misnames."""
    ids = commit_text(dws, write_buffer(dws, _AGREEMENT_BLOCKS))
    store = GraphStore(dws.db_path)
    spy = _Spy(store)
    asked = _states_asked(spy, store)
    env = _lexical(dws, "cache", spy, limit=50)
    assert len(asked) == 1
    assert sorted(asked[0]) == sorted(ids.values())
    assert {m["slug"] for m in env["matches"]} == set(ids)
    assert all(m["state"] == "active" for m in env["matches"])
    by_slug = {m["slug"]: m for m in env["matches"]}
    # G12: the stream parser keeps the comment literal and casefolds scope tags,
    # which is the graph's stored form.
    assert by_slug["cache-commented"]["axiom"] == (
        "The cache <!-- note --> strategy decides where results live.")
    assert by_slug["cache-scoped"]["scope"] == ["cache", "ops"]


def test_the_record_path_mints_the_same_ids(dws):
    """Row 1, the other writer: a decision recorded through `record` (which writes the
    buffer and commits) is found by the id the lexical path computes from its block."""
    from mitos.sync import MitosSyncManager
    m = MitosSyncManager(dws)
    for slug, axiom in [("rec-commented", "The cache <!-- aside --> is local."),
                        ("rec-plain", "The cache is local.")]:
        res = m.record_decision_entry(axiom, "Rejected a remote cache.", ["Cache"], slug=slug)
        assert "error" not in res, res
    env = _lexical(dws, "cache", GraphStore(dws.db_path))
    assert {mm["slug"]: mm.get("state") for mm in env["matches"]} == {
        "rec-commented": "active", "rec-plain": "active"}


# --------------------------------------------------------------------------- #
# Row 2 — a same-slug lineage gives the successor, once, on every surface
# --------------------------------------------------------------------------- #

def _same_slug(config) -> Dict[str, str]:
    """`database` superseded by a same-slug `database`, the successor amended.

    Committed predecessor first, so an unordered slug read returns the retired one
    (rowid order) — the old code's drop, deterministically."""
    pred = block("database", "We use PostgreSQL for the database.")
    succ = block("database", "We use SQLite for the database, in WAL mode.",
                 supersedes="database")
    store = GraphStore(config.db_path)
    retired = commit_text(config, pred)["database"]
    successor = commit_text(config, succ)["database"]
    amender = commit_text(config, block(
        "wal-tuning", "Tune the WAL checkpoint interval.", amends="database"))["wal-tuning"]
    write_buffer(config, [block("wal-tuning", "Tune the WAL checkpoint interval.",
                                amends="database"), succ, pred])
    assert store.get_node_state(retired) == "superseded"
    assert store.get_node(successor)["amended_by"] == ["wal-tuning"]
    return {"retired": retired, "successor": successor, "amender": amender}


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_a_same_slug_lineage_answers_the_successor(dws, driver):
    """Row 2: the live decision once, with its own axiom, state and stamps."""
    _same_slug(dws)
    resp = drive(driver, dws, "postgresql sqlite database", cause="no-providers")
    assert listed(driver, resp).count("database") == 1
    if DRIVERS[driver][3] == "text":
        assert "We use SQLite for the database, in WAL mode." in resp
        assert "PostgreSQL" not in resp
        assert "wal-tuning" in resp
        return
    (match,) = [m for m in resp["matches"] if m["slug"] == "database"]
    assert match["axiom"] == "We use SQLite for the database, in WAL mode."
    assert match["state"] == "active"
    assert match["amended_by"] == ["wal-tuning"]
    assert "superseded_by" not in match


# --------------------------------------------------------------------------- #
# Row 3 — a renamed, unsynced block still resolves by its content
# --------------------------------------------------------------------------- #

def _renamed(config) -> None:
    """`old-live` (amended) and `old-dead` (superseded) are renamed in the buffer only."""
    commit_text(config, block("old-live", "The renamed live cache rule."))
    commit_text(config, block("old-dead", "The renamed retired cache rule."))
    commit_text(config, block("live-amender", "Amend the live rule.", amends="old-live"))
    commit_text(config, block("dead-killer", "Replace the retired rule.",
                              supersedes="old-dead"))
    write_buffer(config, [
        block("dead-killer", "Replace the retired rule.", supersedes="old-dead"),
        block("live-amender", "Amend the live rule.", amends="old-live"),
        block("new-live", "The renamed live cache rule."),
        block("new-dead", "The renamed retired cache rule."),
    ])


@pytest.mark.parametrize("driver", list(DRIVERS))
def test_a_renamed_block_resolves_by_its_content(dws, driver):
    """Row 3: the live one is served stamped under its new slug; the retired one is
    filtered out, though no slug in the graph names it any more."""
    _renamed(dws)
    resp = drive(driver, dws, "renamed", cause="no-providers")
    assert listed(driver, resp) == ["new-live"]
    if DRIVERS[driver][3] == "text":
        assert "live-amender" in resp
        return
    (match,) = resp["matches"]
    assert match["state"] == "active" and match["amended_by"] == ["live-amender"]


# --------------------------------------------------------------------------- #
# Row 4 — an unsynced axiom edit has no node
# --------------------------------------------------------------------------- #

def test_an_unsynced_edit_is_served_unstated_and_the_older_block_is_not_read(dws):
    """Row 4 / G2: the buffer's edited block is the newest per slug; its id is absent,
    so it carries no state and no stamps, and the committed block in the archive (whose
    node is amended) is not consulted."""
    archived = block("cache-rule", "The cache is flushed hourly.")
    commit_text(dws, archived)
    commit_text(dws, block("cache-amender", "Amend the flush.", amends="cache-rule"))
    write_archive(dws, "2026-Q3.md", [archived])
    write_buffer(dws, [block("cache-amender", "Amend the flush.", amends="cache-rule"),
                       block("cache-rule", "The cache is flushed every minute.")])
    env = _lexical(dws, "flushed", GraphStore(dws.db_path))
    (match,) = env["matches"]
    assert match["axiom"] == "The cache is flushed every minute."
    assert "state" not in match and "amended_by" not in match
    assert "stamps_unavailable" not in env and "stamps_unread" not in env


# --------------------------------------------------------------------------- #
# Row 5 — two reads per answer, whatever the match count
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cell", ["many", "zero", "all-unsynced"])
def test_an_answer_makes_one_state_read_and_one_stamp_read(dws, cell, monkeypatch):
    """Row 5: exactly one `get_node_states` and one `get_modifiers_map` call per answer,
    no per-hit read; on an empty id list a read runs no SQL statement at all."""
    blocks = [block(f"cache-{i}", f"Cache rule {i}.") for i in range(8)]
    text = write_buffer(dws, blocks)
    if cell != "all-unsynced":
        commit_text(dws, text)
    store = GraphStore(dws.db_path)
    recorder = _record(store, monkeypatch)
    spy = _Spy(store)
    env = _lexical(dws, "nothing-matches" if cell == "zero" else "cache", spy)
    assert spy.calls == ["get_node_states", "get_modifiers_map"]
    if cell == "many":
        assert len(env["matches"]) == 5
    elif cell == "zero":
        assert env["matches"] == []
        assert recorder.statements == [] and recorder.connections == 0
    else:
        assert len(env["matches"]) == 5 and all("state" not in m for m in env["matches"])
        # The state read ran once; the stamp read had no stated id and ran nothing.
        assert recorder.connections == 1 and len(_node_reads(recorder)) == 1


def test_a_drifted_match_is_live_and_says_so(dws):
    """Row 5's sibling for B6: drift annotates, it does not retire — a drifted node's
    block is kept and carries its state."""
    ids = commit_text(dws, write_buffer(dws, [block("cache-drifting", "A drifting cache rule.")]))
    store = GraphStore(dws.db_path)
    store.write_signal(ids["cache-drifting"], "drifted")
    (match,) = _lexical(dws, "cache", store)["matches"]
    assert match["state"] == "drifted"


# --------------------------------------------------------------------------- #
# Row 6 — `get_node_states`
# --------------------------------------------------------------------------- #

@pytest.fixture
def store(tmp_path) -> GraphStore:
    return GraphStore(str(tmp_path / "graph.sqlite"))


def test_get_node_states_agrees_with_get_node_state(store):
    """Row 6 / G4: one liveness atom — every node's batched state equals its single
    read, across active, drifted, superseded and corrected; an unknown id is absent."""
    active = _commit(store, "keep-it", "Keep it simple.")
    drifted = _commit(store, "drifting", "A decision whose code moved on.")
    store.write_signal(drifted, "drifted")
    old = _commit(store, "old-way", "The old way.")
    _commit(store, "new-way", "The new way.", supersedes=["old-way"])
    typo = _commit(store, "typo-one", "A tpyo.")
    _commit(store, "typo-fixed", "A typo.", corrects=["typo-one"])
    oq = _commit(store, "open-q", "Is this open?", kind="open_question")
    ids = [n["id"] for n in store.get_all_nodes()]
    unknown = "f" * 64
    states = store.get_node_states(ids + [unknown])
    assert states == {i: store.get_node_state(i) for i in ids}
    assert unknown not in states
    assert store.get_node_state(unknown) == "active"  # why existence comes from the read
    assert {states[active], states[drifted], states[old], states[typo]} == {
        "active", "drifted", "superseded", "corrected"}
    assert oq in states


def test_get_node_states_dedupes_and_reads_one_statement_per_chunk(store, monkeypatch):
    """Row 6: duplicates are bound once; seven ids at a chunk of three are three
    SELECTs; an empty input runs nothing."""
    ids = [_commit(store, f"d-{i}", f"Decision {i}.") for i in range(7)]
    monkeypatch.setattr(store_module, "_POINT_ID_CHUNK", 3)
    recorder = _record(store, monkeypatch)
    assert set(store.get_node_states(ids + ids[:4])) == set(ids)
    reads = _node_reads(recorder)
    assert len(reads) == 3 and recorder.connections == 1
    assert sorted(p for _, params in reads for p in params) == sorted(ids)
    assert store.get_node_states([]) == {} and recorder.connections == 1


def test_get_node_states_wraps_a_sqlite_fault(store, monkeypatch):
    """Row 6: a SQLite fault surfaces typed, the one exception the lexical leaf catches."""
    node_id = _commit(store, "the-node", "The node.")
    original = store._get_connection

    class _Faulting:
        def __init__(self, conn: Any) -> None:
            self._conn = conn

        def execute(self, sql: str, params: Any = ()) -> Any:
            raise sqlite3.OperationalError("database is locked")

        def __getattr__(self, name: str) -> Any:
            return getattr(self._conn, name)

    monkeypatch.setattr(store, "_get_connection", lambda: _Faulting(original()))
    with pytest.raises(DatabaseError, match="state"):
        store.get_node_states([node_id])


def test_the_state_read_is_on_the_graph_protocol():
    """Row 6: the read is declared where 2a's resolver and 2f's live-end read are."""
    import inspect

    from mitos.protocols import GraphStoreProtocol
    assert "get_node_states" in GraphStoreProtocol.__dict__
    proto = inspect.signature(GraphStoreProtocol.get_node_states)
    real = inspect.signature(GraphStore.get_node_states)
    assert list(proto.parameters) == list(real.parameters)


# --------------------------------------------------------------------------- #
# Row 7 — faults
# --------------------------------------------------------------------------- #

def _fault(name: str):
    def raising(*_a, **_k):
        raise DatabaseError(f"{name}: database is locked")
    return raising


def _retired_and_live(config) -> None:
    commit_text(config, block("cache-old", "The old cache rule."))
    commit_text(config, block("cache-new", "The new cache rule.", supersedes="cache-old"))
    commit_text(config, block("cache-amend", "Amend the new cache rule.", amends="cache-new"))
    write_buffer(config, [
        block("cache-amend", "Amend the new cache rule.", amends="cache-new"),
        block("cache-new", "The new cache rule.", supersedes="cache-old"),
        block("cache-old", "The old cache rule."),
    ])


def test_a_state_read_fault_reads_as_no_graph(dws):
    """Row 7: liveness unknown — no filter, no state, and the store-less disclosure."""
    _retired_and_live(dws)
    store = GraphStore(dws.db_path)
    faulted = _lexical(dws, "cache", _Spy(store, get_node_states=_fault("states")))
    storeless = _lexical(dws, "cache", None)
    assert faulted == storeless
    assert faulted["stamps_unavailable"] is True and "stamps_unread" not in faulted
    assert "cache-old" in [m["slug"] for m in faulted["matches"]]
    assert "may include superseded ones" in faulted["note"]


def test_a_stamp_read_fault_costs_the_stamps_alone(dws):
    """Row 7: the filter and the states stand; `stamps_unread` says the rest."""
    _retired_and_live(dws)
    store = GraphStore(dws.db_path)
    healthy = _lexical(dws, "cache", store)
    faulted = _lexical(dws, "cache", _Spy(store, get_modifiers_map=_fault("modifiers")))
    assert [m["slug"] for m in faulted["matches"]] == ["cache-amend", "cache-new"]
    assert all(m["state"] == "active" for m in faulted["matches"])
    assert faulted["stamps_unread"] is True and "stamps_unavailable" not in faulted
    assert healthy["matches"][1]["amended_by"] == ["cache-amend"]
    assert "amended_by" not in faulted["matches"][1]
    # The leaf adds no clause for it: the wording is per-surface, the composer's.
    assert faulted["note"] == healthy["note"]


class _Bare:
    """A store double with neither batched read."""

    def get_modifiers(self, node_id):
        return {}


@pytest.mark.parametrize("cell", ["missing-method", "other-exception"])
def test_anything_but_a_database_error_is_loud(dws, cell):
    """Row 7 / G7: the catch is `DatabaseError` alone — a double missing the method, or
    a read raising anything else, propagates rather than posing as a stated fault."""
    write_buffer(dws, [block("cache-rule", "A cache rule.")])
    if cell == "missing-method":
        with pytest.raises(AttributeError):
            _lexical(dws, "nothing-matches", _Bare())
    else:
        def boom(*_a, **_k):
            raise RuntimeError("a defect, not a fault")
        with pytest.raises(RuntimeError):
            _lexical(dws, "cache", _Spy(GraphStore(dws.db_path), get_node_states=boom))
