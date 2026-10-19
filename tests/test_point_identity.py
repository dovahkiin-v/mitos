"""Phase 2a: ``GraphStore.resolve_point_ids`` — a hit's point id names its node.

A decision's point id is ``hash_to_uuid(node_id)``: the first 128 bits of the node's
content-hash id. The resolver maps a batch of them back to nodes in one state-agnostic
read and partitions the input into resolved, orphan and ambiguous. The rows here pin
that partition against a real temp graph, and pin the read's shape: one statement per
chunk, an index search, and the reader-facing node with ``computed_state`` and no
modifier stamps (vision §4.2; ADR
``vector-hits-resolve-to-nodes-by-point-id-prefix-never-by-payload-slug``).

Every id is derived (``commit_parsed_entry``'s delta, ``hash_to_uuid``), never
hardcoded. The same-slug fixture here is the one 2c's four-surface rows reuse.
"""

import importlib.util
import math
import os
import shutil
import tempfile
from typing import Any, List, Optional, Tuple

import pytest

import mitos.store as store_module
from mitos.parser import ParsedEntry
from mitos.store import MODIFIER_EDGE_KEYS, GraphStore, PointResolution
from mitos.vector_store import hash_to_uuid

from _point_hits import orphan_point_id

_GOLDEN_HARNESS = os.path.join(os.path.dirname(__file__), "golden", "_harness.py")


@pytest.fixture
def store(tmp_path) -> GraphStore:
    return GraphStore(str(tmp_path / "graph.sqlite"))


def _commit(store: GraphStore, slug: str, axiom: str, *, kind: str = "decision",
            supersedes: Optional[List[str]] = None, corrects: Optional[List[str]] = None,
            amends: Optional[List[str]] = None, scope: Optional[List[str]] = None) -> str:
    """Commits one entry through the parse→commit path and returns its node id."""
    entry = ParsedEntry(kind, slug, 1, 10)
    if kind == "decision":
        entry.axiom = axiom
        entry.rejected_paths = "Rejected the obvious alternative."
    else:
        entry.topic = axiom
        entry.questions_raised = [f"What about {slug}?"]
    entry.scope = scope if scope is not None else ["core"]
    if supersedes:
        entry.supersedes = supersedes
    if corrects:
        entry.corrects = corrects
    if amends:
        entry.amends = amends
    return store.commit_parsed_entry(entry).node_id


def same_slug_lineage(store: GraphStore, slug: str = "database") -> Tuple[str, str]:
    """A retired decision and its same-slug successor (MI-13): ``(retired, successor)``.

    Both nodes carry one slug, so a slug join cannot tell their points apart; the
    point id can. Built through the commit path a declared same-slug supersession takes.
    """
    retired = _commit(store, slug, "We use PostgreSQL.")
    successor = _commit(store, slug, "We use SQLite for local WAL reads.", supersedes=[slug])
    return retired, successor


class _Recorder:
    """Counts the connections a store opens and the statements run on them."""

    def __init__(self) -> None:
        self.connections = 0
        self.statements: List[Tuple[str, Any]] = []


class _RecordingConnection:
    """Forwards to a real connection, recording every ``execute``."""

    def __init__(self, conn: Any, recorder: _Recorder) -> None:
        self._conn = conn
        self._recorder = recorder

    def execute(self, sql: str, params: Any = ()) -> Any:
        self._recorder.statements.append((sql, params))
        return self._conn.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


def _record(store: GraphStore, monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """Wraps ``store._get_connection``: it opens a fresh connection per call, so the
    seam is the opener, never one connection held beforehand."""
    recorder = _Recorder()
    original = store._get_connection

    def opener() -> Any:
        recorder.connections += 1
        return _RecordingConnection(original(), recorder)

    monkeypatch.setattr(store, "_get_connection", opener)
    return recorder


def _node_reads(recorder: _Recorder) -> List[Tuple[str, Any]]:
    """The statements that read ``nodes`` (the scope fetch reads ``node_scopes``)."""
    return [(sql, params) for sql, params in recorder.statements if "node_scopes" not in sql]


def _assert_partition(result: PointResolution, inputs: List[str]) -> None:
    """Every distinct input lands in exactly one bucket, and nothing else does."""
    buckets = [set(result.nodes), set(result.orphans), set(result.ambiguous)]
    distinct = list(dict.fromkeys(inputs))
    for pid in distinct:
        assert sum(pid in b for b in buckets) == 1, f"{pid!r} is in {sum(pid in b for b in buckets)} buckets"
    assert set().union(*buckets) == set(distinct)
    assert len(result.orphans) == len(set(result.orphans))


# --------------------------------------------------------------------------- #
# Resolved, across every state — retired nodes resolve, which is the point
# --------------------------------------------------------------------------- #

def test_resolves_every_state_with_get_node_states_semantics(store: GraphStore) -> None:
    active = _commit(store, "keep-it", "Keep it simple.")
    drifted = _commit(store, "drifting-one", "A decision whose code moved on.")
    store.write_signal(drifted, "drifted")
    superseded = _commit(store, "old-cache", "Cache for an hour.")
    _commit(store, "new-cache", "Cache for a minute.", supersedes=["old-cache"])
    corrected = _commit(store, "typo-one", "Use teh cache.")
    _commit(store, "typo-fixed", "Use the cache.", corrects=["typo-one"])
    question = _commit(store, "open-thing", "How do we shard?", kind="open_question")

    ids = [active, drifted, superseded, corrected, question]
    result = store.resolve_point_ids([hash_to_uuid(i) for i in ids])

    assert result.orphans == [] and result.ambiguous == {}
    states = {}
    for node_id in ids:
        node = result.nodes[hash_to_uuid(node_id)]
        assert node["id"] == node_id
        assert node["computed_state"] == store.get_node_state(node_id)
        states[node_id] = node["computed_state"]
    # Non-vacuous: the set really spans the vocabulary.
    assert [states[i] for i in ids] == ["active", "drifted", "superseded", "corrected", "active"]
    assert result.nodes[hash_to_uuid(question)]["kind"] == "open_question"


def test_a_same_slug_lineage_resolves_each_point_to_its_own_node(store: GraphStore) -> None:
    """The slug is shared (MI-13); the point ids are not, and neither node is the other."""
    retired, successor = same_slug_lineage(store)
    assert sorted(store.resolve_slug("database")) == sorted([retired, successor])

    result = store.resolve_point_ids([hash_to_uuid(retired), hash_to_uuid(successor)])

    assert result.nodes[hash_to_uuid(retired)]["id"] == retired
    assert result.nodes[hash_to_uuid(retired)]["computed_state"] == "superseded"
    assert result.nodes[hash_to_uuid(successor)]["id"] == successor
    assert result.nodes[hash_to_uuid(successor)]["computed_state"] == "active"
    assert result.nodes[hash_to_uuid(retired)]["slug"] == result.nodes[hash_to_uuid(successor)]["slug"]


def test_a_renamed_decision_still_resolves_by_its_point_id(tmp_path, monkeypatch) -> None:
    """After an ``amend_commentary`` rename the id is unchanged (M2): the property is
    that the point id is slug-free, so a renamed decision's hit is never dropped."""
    from mitos.cli import cmd_init
    from mitos.config import MitosConfig
    from mitos.sync import MitosSyncManager

    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    workspace = tempfile.mkdtemp()
    try:
        config = MitosConfig(workspace)
        cmd_init(config)
        manager = MitosSyncManager(config)
        manager.record_decision_entry("Use SQLite for the store.", "rej", ["db"],
                                      slug="use-sqlite")
        node_id = manager.store.resolve_slug("use-sqlite")[0]
        receipt = manager.amend_commentary("use-sqlite", {"slug": "sqlite-store"})
        assert receipt["status"] == "amended", receipt
        assert manager.store.resolve_slug("use-sqlite") == []

        result = manager.store.resolve_point_ids([hash_to_uuid(node_id)])

        assert result.nodes[hash_to_uuid(node_id)]["id"] == node_id
        assert result.nodes[hash_to_uuid(node_id)]["slug"] == "sqlite-store"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Orphan and ambiguous — existence is tested, never asked of a state
# --------------------------------------------------------------------------- #

def test_an_absent_point_is_an_orphan_found_by_existence(store: GraphStore) -> None:
    _commit(store, "present", "A node that is here.")
    absent = orphan_point_id("absent")

    result = store.resolve_point_ids([absent])

    assert result.orphans == [absent]
    assert result.nodes == {} and result.ambiguous == {}


def test_get_node_state_answers_active_for_that_same_absent_id(store: GraphStore) -> None:
    """The twin: asking state would have called the orphan live. Existence-testing is
    the load-bearing half of the orphan rule, not a nicety."""
    absent = orphan_point_id("absent").replace("-", "") * 2
    assert store.get_node(absent) is None
    assert store.get_node_state(absent) == "active"
    assert store.resolve_point_ids([hash_to_uuid(absent)]).orphans == [hash_to_uuid(absent)]


def _insert_raw_decision(store: GraphStore, node_id: str, slug: str) -> None:
    """Writes one ``nodes`` row directly: content hashes cannot be made to collide.

    Every NOT NULL column of the live schema (``migrations.py``'s ``nodes`` DDL) is
    given: ``kind``, ``slug``, ``slug_casefold``, ``source``, ``created_at``,
    ``updated_at``, with ``id`` as the primary key.
    """
    conn = store._get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO nodes (id, kind, slug, slug_casefold, source, axiom, "
                "created_at, updated_at) VALUES (?, 'decision', ?, ?, 'user', ?, ?, ?)",
                (node_id, slug, slug.casefold(), f"Axiom of {slug}.",
                 "2026-10-16T00:00:00.000000Z", "2026-10-16T00:00:00.000000Z"),
            )
    finally:
        conn.close()


def test_two_nodes_sharing_a_prefix_are_ambiguous_and_resolve_to_neither(
    store: GraphStore,
) -> None:
    """A 128-bit collision is a birthday bound of 2⁶⁴ — treated as unresolvable, never
    guessed (the ADR's rejected path 4)."""
    real = _commit(store, "real-one", "A real node.")
    prefix = real[:32]
    twin_a = prefix + "a" * 32
    twin_b = prefix + "0" * 32
    other = _commit(store, "other-one", "Another node.")
    _insert_raw_decision(store, twin_a, "twin-a")
    _insert_raw_decision(store, twin_b, "twin-b")

    pid = hash_to_uuid(real)
    result = store.resolve_point_ids([pid, hash_to_uuid(other)])

    assert result.ambiguous == {pid: sorted([real, twin_a, twin_b])}
    assert pid not in result.nodes and pid not in result.orphans
    assert result.nodes[hash_to_uuid(other)]["id"] == other


# --------------------------------------------------------------------------- #
# Input handling — malformed, duplicate, dashless, uppercase, empty
# --------------------------------------------------------------------------- #

def test_malformed_and_duplicate_input(store: GraphStore) -> None:
    node_id = _commit(store, "the-node", "The node.")
    pid = hash_to_uuid(node_id)
    inputs = [pid, "not-a-point-id", pid, "", "7", pid.replace("-", "")[:31],
              "not-a-point-id"]

    result = store.resolve_point_ids(inputs)

    assert list(result.nodes) == [pid]
    assert result.orphans == ["not-a-point-id", "", "7", pid.replace("-", "")[:31]]
    assert result.ambiguous == {}
    _assert_partition(result, inputs)


def test_every_spelling_of_one_point_gets_its_own_entry_and_its_own_dict(
    store: GraphStore,
) -> None:
    node_id = _commit(store, "the-node", "The node.")
    dashed = hash_to_uuid(node_id)
    dashless = dashed.replace("-", "")
    upper = dashed.upper()

    result = store.resolve_point_ids([dashed, dashless, upper])

    assert list(result.nodes) == [dashed, dashless, upper]
    assert {n["id"] for n in result.nodes.values()} == {node_id}
    first, second, third = result.nodes.values()
    assert first is not second and second is not third and first is not third


def test_empty_input_runs_no_statement(store: GraphStore, monkeypatch) -> None:
    recorder = _record(store, monkeypatch)

    result = store.resolve_point_ids([])

    assert result == PointResolution(nodes={}, orphans=[], ambiguous={})
    assert recorder.connections == 0 and recorder.statements == []


def test_input_that_is_all_malformed_runs_no_statement(store: GraphStore, monkeypatch) -> None:
    recorder = _record(store, monkeypatch)

    result = store.resolve_point_ids(["nope", "also-nope"])

    assert result.orphans == ["nope", "also-nope"]
    assert recorder.connections == 0


def test_every_input_lands_in_exactly_one_bucket(store: GraphStore) -> None:
    real = _commit(store, "real-one", "A real node.")
    other = _commit(store, "other-one", "Another node.")
    _insert_raw_decision(store, real[:32] + "f" * 32, "real-twin")
    inputs = [hash_to_uuid(real), hash_to_uuid(other), orphan_point_id("x"), "junk",
              hash_to_uuid(other).upper(), hash_to_uuid(other)]

    result = store.resolve_point_ids(inputs)

    _assert_partition(result, inputs)
    assert set(result.ambiguous) == {hash_to_uuid(real)}
    assert set(result.nodes) == {hash_to_uuid(other), hash_to_uuid(other).upper()}
    assert result.orphans == [orphan_point_id("x"), "junk"]


# --------------------------------------------------------------------------- #
# The read's shape — batched, indexed, reader-facing, no stamps
# --------------------------------------------------------------------------- #

def test_n_ids_cost_one_node_read_per_chunk_never_one_per_id(
    store: GraphStore, monkeypatch
) -> None:
    ids = [_commit(store, f"node-{i}", f"Axiom number {i}.") for i in range(7)]
    monkeypatch.setattr(store_module, "_POINT_ID_CHUNK", 3)
    recorder = _record(store, monkeypatch)

    result = store.resolve_point_ids([hash_to_uuid(i) for i in ids])

    assert len(result.nodes) == 7
    chunks = math.ceil(7 / 3)
    assert recorder.connections == 1
    assert len(_node_reads(recorder)) == chunks
    assert len(recorder.statements) <= 2 * chunks


def test_the_read_is_an_index_search_on_nodes_never_a_scan(
    store: GraphStore, monkeypatch
) -> None:
    """Pins the property, not the plan text: SQLite picks the ``UNIQUE (id, kind)``
    index when ``nodes.*`` is selected, the primary key when only ``id`` is."""
    ids = [_commit(store, f"node-{i}", f"Axiom number {i}.") for i in range(3)]
    recorder = _record(store, monkeypatch)
    store.resolve_point_ids([hash_to_uuid(i) for i in ids])
    (sql, params), = _node_reads(recorder)

    conn = GraphStore._get_connection(store)  # the class's opener, past the recorder
    try:
        plan = [row[3] for row in conn.execute(f"EXPLAIN QUERY PLAN {sql}", params)]
    finally:
        conn.close()

    on_nodes = [line for line in plan if " nodes " in f" {line} "]
    assert any(line.startswith("SEARCH nodes") and "(id>? AND id<?)" in line
               for line in on_nodes), plan
    assert not any(line.startswith("SCAN nodes") for line in on_nodes), plan


def test_a_resolved_node_is_get_nodes_shape_plus_state_minus_stamps(store: GraphStore) -> None:
    plain = _commit(store, "plain-one", "A node nothing modifies.", scope=["core", "db"])
    amended = _commit(store, "amended-one", "A node a later decision amends.")
    _commit(store, "the-amender", "Amends the earlier one.", amends=["amended-one"])

    result = store.resolve_point_ids([hash_to_uuid(plain), hash_to_uuid(amended)])

    for node_id in (plain, amended):
        reference = store.get_node(node_id)
        expected_keys = (set(reference) - set(MODIFIER_EDGE_KEYS.values())) | {"computed_state"}
        resolved = result.nodes[hash_to_uuid(node_id)]
        assert set(resolved) == expected_keys
        for key in set(reference) - set(MODIFIER_EDGE_KEYS.values()):
            assert resolved[key] == reference[key], key
    # The stamp read is separate (D4): the resolver carries none, even where one exists.
    assert store.get_node(amended)["amended_by"] == ["the-amender"]
    assert "amended_by" not in result.nodes[hash_to_uuid(amended)]
    assert result.nodes[hash_to_uuid(plain)]["scope"] == ["core", "db"]


def test_a_sqlite_fault_raises_database_error(store: GraphStore, monkeypatch) -> None:
    """A fault inside the read surfaces typed, as ``created_at_for``'s does; the caller
    (2c's walk) owns what it costs."""
    import sqlite3

    from mitos.errors import DatabaseError

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
    with pytest.raises(DatabaseError, match="point id"):
        store.resolve_point_ids([hash_to_uuid(node_id)])


# --------------------------------------------------------------------------- #
# The agreement pin — the whole Harbor graph, retired nodes and questions included
# --------------------------------------------------------------------------- #

def _harbor(tmp_path) -> GraphStore:
    spec = importlib.util.spec_from_file_location("_golden_harness_2a", _GOLDEN_HARNESS)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    return harness.build_reference_graph(str(tmp_path / "harbor.sqlite"))


def test_resolver_agrees_with_hash_to_uuid_and_get_node_state_over_harbor(tmp_path) -> None:
    store = _harbor(tmp_path)
    conn = store._get_connection()
    try:
        node_ids = [row["id"] for row in conn.execute("SELECT id FROM nodes")]
    finally:
        conn.close()
    states = {i: store.get_node_state(i) for i in node_ids}
    # Non-vacuous: Harbor holds retired decisions and open questions.
    assert {"superseded", "active"} <= set(states.values())
    assert any(store.get_node(i)["kind"] == "open_question" for i in node_ids)

    result = store.resolve_point_ids([hash_to_uuid(i) for i in node_ids])

    assert result.orphans == [] and result.ambiguous == {}
    assert len(result.nodes) == len(node_ids)
    for node_id in node_ids:
        node = result.nodes[hash_to_uuid(node_id)]
        assert node["id"] == node_id
        assert node["computed_state"] == states[node_id]
