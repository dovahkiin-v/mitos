"""The ``node_scopes`` ordinal: authored scope order in the graph (surface-entropy 2a).

A decision's primary scope is the tag its author wrote first. Ladder step 4 gives
``node_scopes`` an ``ordinal`` column, ``commit_parsed_entry`` persists the parser's
authored order, and every hydrated read returns it. The rows here pin the contract
at each seam, against real SQLite with no mocks:

- the migration: an existing graph backfills alphabetically, so it reads and renders
  byte-identically, and replaying the ladder changes nothing (MI-3);
- a read-only store over a graph that has not migrated yet still answers, in the
  order that graph holds (the MCP reads, ``status``, ``rebuild``'s reference read);
- the write: authored order through all four ``_scopes_for`` callers and the delta;
  a same-set reorder rewrites the rows and ticks ``updated_at`` without touching the
  node id or its edges (C1); an equal list writes nothing;
- version overlap: an older ladder applies nothing to a step-4 graph, and an older
  writer's column-list INSERT still commits at the column's DEFAULT.
"""

import json
import os
import sqlite3

import pytest

from mitos import mcp_server
from mitos.cli import cmd_init, cmd_status
from mitos.config import MitosConfig
from mitos.cutover import _read_current_graph_reference_cores
from mitos.migrations import MIGRATION_STEPS, _pending_head, run_migrations
from mitos.parser import ParsedEntry
from mitos.renderer import assemble_render
from mitos.store import _SCOPE_ORDER_SQL, GraphStore, open_connection

_TS = "2026-09-14T00:00:00.000000+00:00"

# Tags inserted in a non-alphabetical order, so the table's rowid order differs from
# the alphabetical rank: a backfill that ranked by rowid would read differently.
# ``ž``/``ä`` sort after ASCII under BINARY collation and Python ``sorted()`` alike.
_STEP3_NODES = [
    ("n-one", "one", ["zeta", "ax", "ž", "cli"]),
    ("n-two", "two", ["z", "ä"]),
    ("n-three", "three", ["solo"]),
    ("n-four", "four", []),
]


def _build_step3_graph(path: str) -> None:
    """Builds a populated graph at ladder step 3, the shape every pre-2a graph has.

    A writable ``GraphStore`` always runs the live ladder to its head, so the step-3
    graph is laddered by injection and filled with raw inserts (G13).
    """
    conn = open_connection(path)
    try:
        run_migrations(conn, MIGRATION_STEPS[:3])
        for node_id, slug, scopes in _STEP3_NODES:
            conn.execute(
                "INSERT INTO nodes (id, kind, slug, slug_casefold, source, axiom, "
                "mechanism_refs_json, rejected_paths_json, created_at, updated_at) "
                "VALUES (?, 'decision', ?, ?, 'user', ?, '[]', ?, ?, ?)",
                (node_id, slug, slug, f"Axiom for {slug}.", f"Not {slug}.", _TS, _TS),
            )
            for tag in scopes:
                conn.execute(
                    "INSERT INTO node_scopes (node_id, scope) VALUES (?, ?)",
                    (node_id, tag),
                )
        conn.commit()
    finally:
        conn.close()


def _user_version(path: str) -> int:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def _raw(path: str, sql: str, params=()) -> list:
    conn = sqlite3.connect(path)
    try:
        return [tuple(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


def _ordered_rows(store: GraphStore, node_id: str) -> list:
    """Reads a node's ``(scope, ordinal)`` rows in stored order (never sorted by tag)."""
    return _raw(
        store.db_path,
        "SELECT scope, ordinal FROM node_scopes WHERE node_id = ? ORDER BY ordinal, scope",
        (node_id,),
    )


def _rowids(store: GraphStore, node_id: str) -> list:
    return _raw(
        store.db_path,
        "SELECT rowid, scope, ordinal FROM node_scopes WHERE node_id = ? ORDER BY rowid",
        (node_id,),
    )


def _updated_at(store: GraphStore, node_id: str) -> str:
    return _raw(store.db_path, "SELECT updated_at FROM nodes WHERE id = ?", (node_id,))[0][0]


def _decision(slug: str, axiom: str, scope, supersedes=None) -> ParsedEntry:
    e = ParsedEntry("decision", slug, 1, 5)
    e.axiom = axiom
    e.rejected_paths = f"An alternative to {slug}."
    e.mechanisms = []
    e.scope = list(scope)
    e.supersedes = list(supersedes or [])
    return e


def _scopes_by_id(store: GraphStore) -> str:
    return json.dumps(
        {n["id"]: n["scope"] for n in store.get_all_nodes()}, sort_keys=True
    )


@pytest.fixture
def store(tmp_path) -> GraphStore:
    """A writable store at the live ladder head."""
    return GraphStore(str(tmp_path / "graph.sqlite"))


@pytest.fixture
def step3_workspace(tmp_path, monkeypatch) -> MitosConfig:
    """An initialised workspace whose graph is a populated step-3 graph."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")  # nothing listens here
    root = tmp_path / "ws"
    root.mkdir()
    config = MitosConfig(str(root))
    cmd_init(config)
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(config.db_path + suffix):
            os.remove(config.db_path + suffix)
    _build_step3_graph(config.db_path)
    return config


# --- The migration ---------------------------------------------------------------


def test_step4_backfills_alphabetically_so_a_migrated_graph_reads_and_renders_identically(
    tmp_path,
) -> None:
    path = str(tmp_path / "graph.sqlite")
    _build_step3_graph(path)
    # Non-vacuity: insertion (rowid) order is not the alphabetical rank.
    by_rowid = [
        r[0]
        for r in _raw(path, "SELECT scope FROM node_scopes WHERE node_id='n-one' ORDER BY rowid")
    ]
    assert by_rowid == ["zeta", "ax", "ž", "cli"]
    assert by_rowid != sorted(by_rowid)

    # Before: what the un-migrated graph answers (read-only, so it stays at step 3).
    before = GraphStore(path, read_only=True)
    before_scopes = _scopes_by_id(before)
    before_render = json.dumps(assemble_render(before), sort_keys=True)
    assert _user_version(path) == 3
    assert json.loads(before_scopes)["n-one"] == ["ax", "cli", "zeta", "ž"]

    after = GraphStore(path)  # the writable boot runs step 4
    assert _user_version(path) == _pending_head(MIGRATION_STEPS) == 4
    assert _ordered_rows(after, "n-one") == [("ax", 0), ("cli", 1), ("zeta", 2), ("ž", 3)]
    assert _ordered_rows(after, "n-two") == [("z", 0), ("ä", 1)]
    assert _ordered_rows(after, "n-three") == [("solo", 0)]
    assert _scopes_by_id(after) == before_scopes
    assert json.dumps(assemble_render(after), sort_keys=True) == before_render
    # The pre-ladder snapshot of the step-3 image is retained on success.
    assert os.path.exists(path + ".snapshot_v3")


def test_replaying_the_ladder_changes_no_row_and_keeps_authored_ordinals(tmp_path) -> None:
    path = str(tmp_path / "graph.sqlite")
    _build_step3_graph(path)
    store = GraphStore(path)
    node_id = store.commit_parsed_entry(
        _decision("wal", "Use WAL.", ["substrate", "database"])
    ).node_id
    rows_before = _raw(path, "SELECT rowid, node_id, scope, ordinal FROM node_scopes ORDER BY rowid")

    conn = open_connection(path)
    try:
        assert run_migrations(conn, MIGRATION_STEPS) == _pending_head(MIGRATION_STEPS)
    finally:
        conn.close()

    assert _raw(path, "SELECT rowid, node_id, scope, ordinal FROM node_scopes ORDER BY rowid") == rows_before
    assert store.get_node(node_id)["scope"] == ["substrate", "database"]


# --- Reads over a graph that has not migrated yet (read-only stores) -------------


def test_a_read_only_store_over_a_step3_graph_hydrates_through_every_caller(tmp_path) -> None:
    path = str(tmp_path / "graph.sqlite")
    _build_step3_graph(path)
    ro = GraphStore(path, read_only=True)
    alphabetical = ["ax", "cli", "zeta", "ž"]
    assert ro.get_node("n-one")["scope"] == alphabetical
    assert ro.get_node_by_slug("one")["scope"] == alphabetical
    assert {d["slug"]: d["scope"] for d in ro.get_active_decisions()}["one"] == alphabetical
    assert ro.query_letter(slug="one")[0]["scope"] == alphabetical
    assert _user_version(path) == 3


def test_an_mcp_read_answers_over_a_step3_graph(step3_workspace) -> None:
    config = step3_workspace
    resp = json.loads(mcp_server.list_decisions(project=config.workspace_dir))
    scopes = {d["slug"]: d["scope"] for d in resp["decisions"]}
    assert scopes["one"] == ["ax", "cli", "zeta", "ž"]
    assert scopes["two"] == ["z", "ä"]
    assert _user_version(config.db_path) == 3  # a read never migrates


def test_status_reports_the_true_node_count_over_a_step3_graph(step3_workspace, capsys) -> None:
    config = step3_workspace
    capsys.readouterr()
    cmd_status(config.workspace_dir, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    # Without the fallback the read-only store's hydration raises, status swallows
    # it, and the count silently reads null.
    assert payload["checks"]["graph_nodes"] == len(_STEP3_NODES)


def test_rebuilds_reference_read_answers_over_a_step3_graph(tmp_path) -> None:
    path = str(tmp_path / "graph.sqlite")
    _build_step3_graph(path)
    cores = _read_current_graph_reference_cores(path)
    assert set(cores) == {node_id for node_id, _slug, _scopes in _STEP3_NODES}
    assert _user_version(path) == 3


class _FailsOnceConn:
    """A connection whose first ``execute`` raises and whose later ones answer empty."""

    def __init__(self, message: str) -> None:
        self.message = message
        self.calls = 0

    def execute(self, sql, params=()):
        self.calls += 1
        if self.calls == 1:
            raise sqlite3.OperationalError(self.message)
        return self

    def fetchall(self) -> list:
        return []


def test_the_fallback_catches_only_the_missing_ordinal_column(store) -> None:
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        store._scopes_for(_FailsOnceConn("database is locked"), ["n-one"])
    with pytest.raises(sqlite3.OperationalError, match="no such column: slug_order"):
        store._scopes_for(_FailsOnceConn("no such column: slug_order"), ["n-one"])
    legacy = _FailsOnceConn("no such column: ordinal")
    assert store._scopes_for(legacy, ["n-one"]) == {}
    assert legacy.calls == 2


# --- The write: authored order, reorder, no-op ------------------------------------


def test_authored_order_reaches_the_delta_and_every_scopes_for_caller(store) -> None:
    delta = store.commit_parsed_entry(
        _decision("wal", "Use WAL.", ["Substrate", "database", "substrate"])
    )
    authored = ["substrate", "database"]
    assert delta.node_scope == authored
    assert delta.self_old_scope == []
    assert _ordered_rows(store, delta.node_id) == [("substrate", 0), ("database", 1)]
    assert store.get_node(delta.node_id)["scope"] == authored
    assert store.get_node_by_slug("wal")["scope"] == authored
    assert store.get_active_decisions()[0]["scope"] == authored
    assert store.query_letter(slug="wal")[0]["scope"] == authored


def test_a_reorder_only_recommit_rewrites_ordinals_and_ticks_but_keeps_identity(store) -> None:
    store.commit_parsed_entry(_decision("old-choice", "The old choice.", ["x"]))
    first = store.commit_parsed_entry(
        _decision("wal", "Use WAL.", ["substrate", "database"], supersedes=["old-choice"])
    )
    edges_before = _raw(store.db_path, "SELECT * FROM edges ORDER BY 1, 2, 3")
    nodes_before = _raw(store.db_path, "SELECT COUNT(*) FROM nodes")
    ticked_before = _updated_at(store, first.node_id)
    assert edges_before  # non-vacuous: the node carries a kill edge
    # Clear the outbox so the re-commit's enqueue is observable.
    conn = sqlite3.connect(store.db_path)
    conn.execute("DELETE FROM pending_embeddings")
    conn.commit()
    conn.close()

    second = store.commit_parsed_entry(
        _decision("wal", "Use WAL.", ["database", "substrate"], supersedes=["old-choice"])
    )

    assert second.node_id == first.node_id  # order never enters the hash (C1)
    assert second.commentary_fields_changed is True
    assert second.node_scope == ["database", "substrate"]
    assert second.self_old_scope == ["substrate", "database"]
    assert _ordered_rows(store, second.node_id) == [("database", 0), ("substrate", 1)]
    assert store.get_node(second.node_id)["scope"] == ["database", "substrate"]
    ticked_after = _updated_at(store, second.node_id)
    assert ticked_after > ticked_before
    assert _raw(store.db_path, "SELECT * FROM edges ORDER BY 1, 2, 3") == edges_before
    assert _raw(store.db_path, "SELECT COUNT(*) FROM nodes") == nodes_before
    assert _raw(
        store.db_path, "SELECT node_id, queued_at FROM pending_embeddings"
    ) == [(second.node_id, ticked_after)]


def test_an_equal_scope_list_writes_no_scope_row_and_does_not_tick(store) -> None:
    node_id = store.commit_parsed_entry(
        _decision("wal", "Use WAL.", ["substrate", "database"])
    ).node_id
    # A later node's rows sit above this node's, so a spurious delete-and-reinsert
    # would be assigned new rowids rather than reusing the old ones.
    store.commit_parsed_entry(_decision("later", "A later call.", ["zeta", "alpha"]))
    rowids = _rowids(store, node_id)
    ticked = _updated_at(store, node_id)

    for scope in (["substrate", "database"], ["Substrate", "substrate", "DATABASE"]):
        delta = store.commit_parsed_entry(_decision("wal", "Use WAL.", scope))
        assert delta.commentary_fields_changed is False
        assert delta.cascade_affected_scopes == []
        assert _rowids(store, node_id) == rowids
        assert _updated_at(store, node_id) == ticked


def test_a_mixed_add_remove_and_reorder_lands_the_exact_authored_list(store) -> None:
    node_id = store.commit_parsed_entry(_decision("wal", "Use WAL.", ["a", "b", "c"])).node_id
    delta = store.commit_parsed_entry(_decision("wal", "Use WAL.", ["d", "c", "a"]))
    assert delta.node_id == node_id
    assert _ordered_rows(store, node_id) == [("d", 0), ("c", 1), ("a", 2)]
    assert store.get_node(node_id)["scope"] == ["d", "c", "a"]
    assert delta.self_old_scope == ["a", "b", "c"]
    # The re-render set is a sorted set of names, not an order.
    assert delta.cascade_affected_scopes == ["a", "b", "c", "d"]


# --- Version overlap and ties ------------------------------------------------------


def test_an_older_ladder_and_an_older_writer_still_work_on_a_step4_graph(store) -> None:
    node_id = store.commit_parsed_entry(_decision("wal", "Use WAL.", ["zeta", "ax"])).node_id
    schema_before = _raw(store.db_path, "SELECT sql FROM sqlite_master ORDER BY name")

    conn = open_connection(store.db_path)
    try:
        # An older mitos's ladder (head 3) applies nothing to a step-4 graph.
        assert run_migrations(conn, MIGRATION_STEPS[:3]) == 4
        assert _raw(store.db_path, "SELECT sql FROM sqlite_master ORDER BY name") == schema_before
        # Its reconcile INSERT names no ordinal and still commits, at the DEFAULT.
        conn.execute(
            "INSERT INTO node_scopes (node_id, scope) VALUES (?, ?)", (node_id, "cli")
        )
        # Its read query still runs.
        legacy = conn.execute(
            "SELECT node_id, scope FROM node_scopes WHERE node_id IN (?) "
            "ORDER BY node_id, scope",
            (node_id,),
        ).fetchall()
    finally:
        conn.close()

    assert [r[1] for r in legacy] == ["ax", "cli", "zeta"]
    assert _raw(
        store.db_path,
        "SELECT ordinal FROM node_scopes WHERE node_id = ? AND scope = 'cli'",
        (node_id,),
    ) == [(0,)]
    # The stated residual: the old writer's row ties with the authored primary and
    # reads first by the tag tiebreak, until new code re-commits the node.
    assert store.get_node(node_id)["scope"] == ["cli", "zeta", "ax"]
    store.commit_parsed_entry(_decision("wal", "Use WAL.", ["zeta", "ax"]))
    assert store.get_node(node_id)["scope"] == ["zeta", "ax"]


def test_tied_ordinals_read_by_the_tag_tiebreak(store) -> None:
    node_id = store.commit_parsed_entry(_decision("tied", "Tied tags.", [])).node_id
    conn = sqlite3.connect(store.db_path)
    for tag in ("zeta", "mid", "ax"):  # all at ordinal 0, rowid order non-alphabetical
        conn.execute("INSERT INTO node_scopes (node_id, scope) VALUES (?, ?)", (node_id, tag))
    conn.commit()
    conn.close()
    assert store.get_node(node_id)["scope"] == ["ax", "mid", "zeta"]


def test_the_scope_order_is_total_without_the_indexs_delivery_order(store) -> None:
    """The shared order fragment breaks ordinal ties on its own.

    Through the store's read, the PK index ``(node_id, scope)`` already delivers a
    node's rows in tag order, so a fragment missing its ``scope`` tiebreak still
    reads alphabetically on today's plan and the row above cannot tell. Forcing a
    rowid-order scan makes the fragment carry the whole order, which is what keeps
    the read deterministic if the plan ever changes.
    """
    node_id = store.commit_parsed_entry(_decision("tied", "Tied tags.", [])).node_id
    conn = sqlite3.connect(store.db_path)
    try:
        for tag in ("zeta", "mid", "ax"):  # all at ordinal 0, rowid order non-alphabetical
            conn.execute(
                "INSERT INTO node_scopes (node_id, scope) VALUES (?, ?)", (node_id, tag)
            )
        scanned = [
            r[0]
            for r in conn.execute(
                "SELECT scope FROM node_scopes NOT INDEXED WHERE node_id = ? "
                f"ORDER BY {_SCOPE_ORDER_SQL}",
                (node_id,),
            )
        ]
    finally:
        conn.close()
    assert scanned == ["ax", "mid", "zeta"]


def test_the_scope_read_searches_the_primary_key_index(store) -> None:
    plan = _raw(
        store.db_path,
        "EXPLAIN QUERY PLAN SELECT node_id, scope FROM node_scopes "
        f"WHERE node_id IN (?, ?) ORDER BY node_id, {_SCOPE_ORDER_SQL}",
        ("a", "b"),
    )
    details = " | ".join(str(row[-1]) for row in plan)
    assert "SEARCH node_scopes USING INDEX sqlite_autoindex_node_scopes_1" in details
