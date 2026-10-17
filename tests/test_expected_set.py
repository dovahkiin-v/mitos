"""The decision index's expected set: one store read, and every consumer keyed on it.

Phase 3b of the recall-confidence vision (ADRs
``graveyard-is-deliberate-every-decision-vector-lives-as-long-as-its-node`` and
``decision-index-gap-and-creation-carve-out-key-on-its-expected-set``). The decision
collection should hold a vector for every decision node, live or retired, plus the active
open questions. ``GraphStore.get_decision_index_expected_ids`` is that set, and
``decision_index_expects_node_other_than`` is its per-record ``EXISTS`` twin over the same
SQL atom. ``status`` (the sensor), ``reconcile`` (the heal), the rebuild/cutover seed,
``recall.missing_index_is_a_gap`` and the inline/import creation carve-out all key on it,
so the heal clears its own warning by construction.

The rows:

* E1 — the store read over a fixture matrix of every computed state of both kinds.
* E2 — the twin agrees with the read for every node, an absent id, and the empty graph.
* E3 — a retired-only graph (an out-of-band kill-edge cycle, the one shape that tells the
  expected-set gate from the active one): the gap predicate says gap, and a lone record
  or import defers rather than creating a one-point collection.
* E4 — the supported path's two familiar carve-out cases, unchanged.
* E5 — an AST row deriving every call site of the three reads under ``mitos/``.
* E6 — a spy row: one patched expected read answers ``status``, ``reconcile``, the prune
  and the gap predicate.
* E7 — the protocol row.
* The ``status`` rows for the three sibling keys, and the shipped-prose pins.

Run under ``./venv/bin/python -m pytest`` (PATTERNS — bare ``python`` lacks deps).
"""

import ast
import inspect
import json
import os
import pathlib
from typing import Any, Dict, FrozenSet, List, Optional, Set

import pytest

from mitos import cli
from mitos.config import MitosConfig
from mitos.importer import MitosProseImporter
from mitos.parser import ParsedEntry
from mitos.protocols import GraphStoreProtocol
from mitos.recall import missing_index_is_a_gap
from mitos.shell_quote import shell_flag
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager
from mitos.vector_store import hash_to_uuid

from test_covering_write_gate import _FakeEmbed, _RecordingVector

REPO = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

@pytest.fixture
def store(tmp_path) -> GraphStore:
    """A fresh graph, laddered to the live head."""
    db_path = str(tmp_path / ".mitos" / "graph.sqlite")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return GraphStore(db_path)


def _decision(store: GraphStore, slug: str, axiom: str, *,
              supersedes: Optional[List[str]] = None,
              corrects: Optional[List[str]] = None,
              resolves: Optional[List[str]] = None) -> str:
    """Commits a decision through parse→commit and returns its node id."""
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = axiom
    entry.rejected_paths = "None."
    entry.supersedes = list(supersedes or [])
    entry.corrects = list(corrects or [])
    entry.resolves = list(resolves or [])
    return store.commit_parsed_entry(entry).node_id


def _open_question(store: GraphStore, slug: str, question: str, *,
                   supersedes: Optional[List[str]] = None,
                   corrects: Optional[List[str]] = None) -> str:
    """Commits an open question through parse→commit and returns its node id."""
    entry = ParsedEntry("open_question", slug, 1, 5)
    entry.topic = "A topic."
    entry.questions_raised = [question]
    entry.supersedes = list(supersedes or [])
    entry.corrects = list(corrects or [])
    return store.commit_parsed_entry(entry).node_id


def _plant_edge(store: GraphStore, source: str, target: str,
                edge_type: str = "supersedes") -> None:
    """An out-of-band kill edge, the shape of ``test_retired_partition._plant_edge``.

    The write path refuses cycles, so a retired-only graph is built by raw insert.
    """
    conn = store._get_connection()
    try:
        conn.execute(
            "INSERT INTO edges (source_id, source_kind, target_id, target_kind, "
            "edge_type, created_at) VALUES (?, 'decision', ?, 'decision', ?, ?)",
            (source, target, edge_type, "2026-10-16T00:00:00+00:00"))
        conn.commit()
    finally:
        conn.close()


def _matrix(store: GraphStore) -> Dict[str, str]:
    """Every computed state of both kinds; returns slug → node id."""
    ids: Dict[str, str] = {}
    ids["d-active"] = _decision(store, "d-active", "An active axiom.")
    ids["d-superseded"] = _decision(store, "d-superseded", "A superseded axiom.")
    ids["d-successor"] = _decision(store, "d-successor", "Its successor.",
                                   supersedes=["d-superseded"])
    ids["d-corrected"] = _decision(store, "d-corrected", "A corrected axiom.")
    ids["d-correction"] = _decision(store, "d-correction", "Its correction.",
                                    corrects=["d-corrected"])
    ids["d-chain-1"] = _decision(store, "d-chain-1", "Chain link one.")
    ids["d-chain-2"] = _decision(store, "d-chain-2", "Chain link two.",
                                 supersedes=["d-chain-1"])
    ids["d-chain-3"] = _decision(store, "d-chain-3", "Chain link three.",
                                 supersedes=["d-chain-2"])
    ids["q-parked"] = _open_question(store, "q-parked", "Parked?")
    ids["q-resolved"] = _open_question(store, "q-resolved", "Resolved?")
    ids["d-resolver"] = _decision(store, "d-resolver", "The resolver.",
                                  resolves=["q-resolved"])
    ids["q-superseded"] = _open_question(store, "q-superseded", "Superseded?")
    ids["q-successor"] = _open_question(store, "q-successor", "Its successor?",
                                        supersedes=["q-superseded"])
    ids["q-corrected"] = _open_question(store, "q-corrected", "Corrected?")
    ids["q-correction"] = _open_question(store, "q-correction", "Its correction?",
                                         corrects=["q-corrected"])
    return ids


_RETIRED_DECISIONS = {"d-superseded", "d-corrected", "d-chain-1", "d-chain-2"}
_KILLED_QUESTIONS = {"q-superseded", "q-corrected"}


# --------------------------------------------------------------------------- #
# E1 — the store read
# --------------------------------------------------------------------------- #

def test_e1_every_decision_in_every_state_plus_the_surviving_open_questions(store) -> None:
    ids = _matrix(store)
    expected = store.get_decision_index_expected_ids()

    assert isinstance(expected, frozenset)
    assert expected == frozenset(
        node_id for slug, node_id in ids.items() if slug not in _KILLED_QUESTIONS
    )
    # A resolved open question stays in (``resolves`` is not a kill edge).
    assert ids["q-resolved"] in expected

    active = store.get_active_node_ids()
    assert active <= expected
    assert expected - active == frozenset(ids[slug] for slug in _RETIRED_DECISIONS)


def test_e1_a_drifted_decision_stays_expected(store) -> None:
    """Drift annotates, it never retires — the expected set keeps it like the active view does."""
    node_id = _decision(store, "drifting", "A drifting axiom.")
    store.write_signal(node_id, "drifted")
    assert node_id in store.get_decision_index_expected_ids()


def test_e1_an_empty_graph_expects_nothing(store) -> None:
    assert store.get_decision_index_expected_ids() == frozenset()


# --------------------------------------------------------------------------- #
# E2 — the twin agrees with the read
# --------------------------------------------------------------------------- #

def _agrees(store: GraphStore, node_id: str) -> bool:
    twin = store.decision_index_expects_node_other_than(node_id)
    reference = len(store.get_decision_index_expected_ids() - {node_id}) > 0
    assert twin == reference, (
        f"decision_index_expects_node_other_than({node_id[:12]}…)={twin} but the "
        f"expected set says {reference}"
    )
    return twin


def test_e2_the_twin_agrees_over_the_matrix(store) -> None:
    assert _agrees(store, "f" * 64) is False  # empty graph
    lone = _decision(store, "lone", "The only axiom.")
    assert _agrees(store, lone) is False  # one node: the fresh-project carve-out
    assert _agrees(store, "f" * 64) is True
    # Its successor is the only LIVE node, but the retired one is still expected — the
    # step that tells the expected-set twin from the pre-3b active-set one (B3).
    successor = _decision(store, "lone-successor", "Its successor.", supersedes=["lone"])
    assert store.get_active_node_ids() == {successor}
    assert _agrees(store, successor) is True
    _matrix(store)
    for node in store.get_all_nodes():
        _agrees(store, node["id"])
    _agrees(store, "f" * 64)


def test_e2_a_killed_open_question_does_not_hold_the_twin_open(store) -> None:
    """The only rows outside the set are killed open questions — the twin must agree."""
    old = _open_question(store, "q-old", "Old?")
    new = _open_question(store, "q-new", "New?", supersedes=["q-old"])
    assert store.get_decision_index_expected_ids() == {new}
    assert _agrees(store, new) is False
    assert _agrees(store, old) is True


# --------------------------------------------------------------------------- #
# E3 — the retired-only graph: the one shape that tells the two gates apart
# --------------------------------------------------------------------------- #

def _retired_only(store: GraphStore) -> FrozenSet[str]:
    a = _decision(store, "cycle-a", "Cycle axiom A.")
    b = _decision(store, "cycle-b", "Cycle axiom B.")
    _plant_edge(store, a, b)
    _plant_edge(store, b, a)
    for row in store.get_pending_embeddings():
        store.remove_pending_embedding(row["node_id"])
    return frozenset({a, b})


def test_e3_a_retired_only_graph_is_a_gap(store) -> None:
    retired = _retired_only(store)

    assert store.get_active_node_ids() == set()
    assert store.get_decision_index_expected_ids() == retired
    assert missing_index_is_a_gap(store) is True
    # What the pre-3b gate answered over the same graph: the empty index, no diagnostic.
    assert bool(store.get_active_node_ids()) is False


def _workspace(tmp_path, name: str = "ws") -> MitosConfig:
    root = tmp_path / name
    os.makedirs(root / ".mitos", exist_ok=True)
    config = MitosConfig(str(root))
    with open(config.decisions_file, "w", encoding="utf-8") as handle:
        handle.write(
            "# Decisions\n"
            "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->\n"
        )
    return config


def _entry(slug: str, axiom: str) -> ParsedEntry:
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = axiom
    entry.rejected_paths = "None."
    return entry


def test_e3_a_lone_record_into_a_retired_only_graph_defers(tmp_path, capsys) -> None:
    """ADR ``decision-index-gap-and-creation-carve-out-key-on-its-expected-set``'s rejected
    path (2), made concrete: the old gate saw one live node and created a one-point index.
    """
    config = _workspace(tmp_path)
    _retired_only(GraphStore(config.db_path))
    vector = _RecordingVector(absent=True)
    manager = MitosSyncManager(config)
    manager.embed_provider = _FakeEmbed()  # type: ignore[assignment]
    manager.vector_store = vector  # type: ignore[assignment]

    entry = _entry("newcomer", "A newly recorded axiom.")
    delta = manager.store.commit_parsed_entry(entry)
    # Under today's active gate this write would declare covering: it is the only live node.
    assert manager.store.get_active_node_ids() == {delta.node_id}

    manager._best_effort_embed(delta, entry)

    assert vector.declarations == [False]
    assert vector.created is False
    assert [r["node_id"] for r in manager.store.get_pending_embeddings()] == [delta.node_id]
    err = capsys.readouterr().err
    assert "does not cover every decision the index should hold" in err


def test_e3_an_import_into_a_retired_only_graph_defers(tmp_path, capsys) -> None:
    config = _workspace(tmp_path)
    _retired_only(GraphStore(config.db_path))
    importer = MitosProseImporter(config)
    importer.embed_provider = _FakeEmbed()  # type: ignore[assignment]
    vector = _RecordingVector(absent=True)
    importer.vector_store = vector  # type: ignore[assignment]

    entry = _entry("imported", "An imported axiom.")
    importer._best_effort_embed(importer.store.commit_parsed_entry(entry), entry)

    assert vector.declarations == [False]
    assert vector.created is False
    assert "do not cover every decision the index should hold" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# E4 — the supported path, unchanged
# --------------------------------------------------------------------------- #

def test_e4_after_a_supersession_a_new_record_still_defers(tmp_path) -> None:
    config = _workspace(tmp_path)
    store = GraphStore(config.db_path)
    _decision(store, "a", "Axiom A.")
    _decision(store, "b", "Axiom B.", supersedes=["a"])
    vector = _RecordingVector(absent=True)
    manager = MitosSyncManager(config)
    manager.embed_provider = _FakeEmbed()  # type: ignore[assignment]
    manager.vector_store = vector  # type: ignore[assignment]

    entry = _entry("c", "Axiom C.")
    manager._best_effort_embed(manager.store.commit_parsed_entry(entry), entry)

    assert vector.declarations == [False]
    assert vector.created is False


def test_e4_the_fresh_project_carve_out_still_creates(tmp_path) -> None:
    config = _workspace(tmp_path)
    vector = _RecordingVector(absent=True)
    manager = MitosSyncManager(config)
    manager.embed_provider = _FakeEmbed()  # type: ignore[assignment]
    manager.vector_store = vector  # type: ignore[assignment]

    entry = _entry("first", "The first axiom.")
    manager._best_effort_embed(manager.store.commit_parsed_entry(entry), entry)

    assert vector.declarations == [True]
    assert vector.created is True


# --------------------------------------------------------------------------- #
# E5 — who calls which read, derived from the source
# --------------------------------------------------------------------------- #

_READS = ("get_decision_index_expected_ids", "decision_index_expects_node_other_than",
          "get_active_node_ids")


def _call_sites() -> Dict[str, Set[str]]:
    """Maps each read to the ``module.qualname`` of every function under ``mitos/`` that calls it."""
    sites: Dict[str, Set[str]] = {name: set() for name in _READS}
    for path in sorted((REPO / "mitos").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))

        def _walk(node: ast.AST, scope: List[str]) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    _walk(child, scope + [child.name])
                    continue
                if (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                        and child.func.attr in sites):
                    sites[child.func.attr].add(".".join([path.stem] + scope))
                _walk(child, scope)

        _walk(tree, [])
    return sites


def test_e5_every_what_should_the_index_hold_consumer_reads_the_expected_set() -> None:
    sites = _call_sites()

    assert sites["get_decision_index_expected_ids"] == {
        "recall.missing_index_is_a_gap",
        "sync.MitosSyncManager.reconcile_embeddings",
        "cutover.rebuild_and_gate",
        "cli.cmd_status",
    }
    # The decision leg of the inline embed (4b split ``_best_effort_embed`` into
    # the decision leg and the rejection leg; the gate moved with the body).
    assert sites["decision_index_expects_node_other_than"] == {
        "sync.MitosSyncManager._embed_decision_inline",
        "importer.MitosProseImporter._best_effort_embed",
    }
    # The active read survives only as a count of the live surface.
    assert sites["get_active_node_ids"] == {
        "cli.cmd_status",
        "sync.MitosSyncManager.reconcile_embeddings",
        "cutover._reconstructed_active_ids",
    }


def test_e5_the_removed_twin_is_gone() -> None:
    assert not hasattr(GraphStore, "has_active_node_other_than")


# --------------------------------------------------------------------------- #
# E6 — one patched read answers every consumer
# --------------------------------------------------------------------------- #

# A sentinel no derivation would produce: two live decisions, without the retired one
# the real read returns and without the live open question. Built from the store being
# asked, so it holds for the build-aside graph a rebuild reads too.
_SENTINEL_SLUGS = frozenset({"d-new", "d-live"})


def _sentinel(self: GraphStore) -> FrozenSet[str]:
    return frozenset(n["id"] for n in self.get_all_nodes() if n["slug"] in _SENTINEL_SLUGS)


_SPY_DECISIONS = (
    "<!-- BEGIN ENTRIES -->\n\n"
    "### d-new\n\n**Decided:** The new axiom.\n**Rejected:** n/a\n**Supersedes:** d-old\n\n"
    "### d-old\n\n**Decided:** The old axiom.\n**Rejected:** n/a\n\n"
    "### d-live\n\n**Decided:** A live axiom.\n**Rejected:** n/a\n"
)
_SPY_QUESTIONS = (
    "<!-- BEGIN ENTRIES -->\n\n"
    "### q-new\n\n**Topic:** A topic.\n**Questions:** New?\n**Supersedes:** q-old\n\n"
    "### q-old\n\n**Topic:** A topic.\n**Questions:** Old?\n"
)


def _spy_corpus(tmp_path) -> MitosConfig:
    config = _workspace(tmp_path, "spy")
    with open(config.decisions_file, "w", encoding="utf-8") as handle:
        handle.write(_SPY_DECISIONS)
    with open(config.questions_file, "w", encoding="utf-8") as handle:
        handle.write(_SPY_QUESTIONS)
    return config


def _slugs(store: GraphStore, ids: Any) -> Set[str]:
    by_id = {n["id"]: n["slug"] for n in store.get_all_nodes()}
    return {by_id[i] for i in ids}


def test_e6_the_rebuild_prune_keeps_what_the_read_returns(tmp_path, monkeypatch) -> None:
    from mitos.cutover import rebuild_and_gate

    config = _spy_corpus(tmp_path)
    monkeypatch.setattr(GraphStore, "get_decision_index_expected_ids", _sentinel)
    result = rebuild_and_gate(config, aside_db_path=config.db_path + ".aside")

    aside = GraphStore(result.aside_db_path)
    assert result.decisions_committed == 3
    assert _slugs(aside, [r["node_id"] for r in aside.get_pending_embeddings()]) == _SENTINEL_SLUGS


def test_e6_reconcile_and_status_and_the_gap_answer_from_the_read(
    tmp_path, monkeypatch, capsys
) -> None:
    from mitos.cutover import rebuild_and_gate, perform_swap

    config = _spy_corpus(tmp_path)
    clean = rebuild_and_gate(config, aside_db_path=config.db_path + ".aside", strict=False)
    perform_swap(config, clean.aside_db_path, timestamp="20261016-000000")
    store = GraphStore(config.db_path)
    for row in store.get_pending_embeddings():
        store.remove_pending_embedding(row["node_id"])
    assert _slugs(store, store.get_decision_index_expected_ids()) == {
        "d-new", "d-old", "d-live", "q-new"}

    monkeypatch.setattr(GraphStore, "get_decision_index_expected_ids", _sentinel)

    # reconcile
    vector = _RecordingVector()
    manager = MitosSyncManager(config)
    manager.embed_provider = _FakeEmbed()  # type: ignore[assignment]
    manager.vector_store = vector  # type: ignore[assignment]
    result = manager.reconcile_embeddings()
    assert result["expected"] == 2
    assert _slugs(store, vector.landed) == _SENTINEL_SLUGS

    # status --json: retired = sentinel − active = ∅ (the real read says d-old)
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", lambda url, coll: {
        "reachable": True, "collection_exists": True, "points": 0})
    monkeypatch.setattr(cli, "scroll_point_ids", lambda *a, **k: set())
    capsys.readouterr()
    cli.cmd_status(config.workspace_dir, as_json=True)
    checks = json.loads(capsys.readouterr().out)["checks"]
    assert checks["retired_decisions"] == 0
    assert checks["missing_retired_slugs"] == []

    # the gap predicate
    monkeypatch.setattr(GraphStore, "get_decision_index_expected_ids",
                        lambda self: frozenset())
    assert missing_index_is_a_gap(store) is False


# --------------------------------------------------------------------------- #
# E7 — the protocol row
# --------------------------------------------------------------------------- #

def test_e7_the_expected_read_is_on_the_graph_protocol() -> None:
    assert "get_decision_index_expected_ids" in GraphStoreProtocol.__dict__
    proto = inspect.signature(GraphStoreProtocol.get_decision_index_expected_ids)
    real = inspect.signature(GraphStore.get_decision_index_expected_ids)
    assert list(proto.parameters) == list(real.parameters)
    assert proto.return_annotation == real.return_annotation


def test_e7_the_active_read_stays_off_the_protocol() -> None:
    """D2: no duck-typed caller reaches ``get_active_node_ids`` after 3b, so documenting
    it on the protocol would document a reach nobody makes. Pinned so it changes only
    deliberately."""
    assert "get_active_node_ids" not in GraphStoreProtocol.__dict__
    assert "decision_index_expects_node_other_than" not in GraphStoreProtocol.__dict__


# --------------------------------------------------------------------------- #
# status — retired completeness in sibling keys
# --------------------------------------------------------------------------- #

_NEW_KEYS = ("retired_decisions", "missing_retired_vectors", "missing_retired_slugs")


def _status_ws(tmp_path, monkeypatch) -> MitosConfig:
    cli.cmd_init(MitosConfig(str(tmp_path)))
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    return MitosConfig(str(tmp_path))


def _qdrant(monkeypatch, *, exists: bool, present: Optional[Set[str]] = None,
            scroll_fails: bool = False) -> None:
    from mitos.errors import VectorStoreError

    monkeypatch.setattr(cli, "_check_qdrant", lambda url, coll: {
        "reachable": True, "collection_exists": exists,
        "points": None if present is None else len(present)})

    def _scroll(base_url, collection, page_size=256):
        if scroll_fails or not exists:
            raise VectorStoreError("Qdrant scroll connection error")
        return set(present or set())

    monkeypatch.setattr(cli, "scroll_point_ids", _scroll)


def _status(config: MitosConfig, capsys, *, as_json: bool) -> Any:
    capsys.readouterr()
    rc = cli.cmd_status(config.workspace_dir, as_json=as_json)
    out = capsys.readouterr().out
    return rc, (json.loads(out) if as_json else out)


def _graveyard(config: MitosConfig) -> Dict[str, str]:
    """Two live decisions, two retired ones, and a killed open question."""
    store = GraphStore(config.db_path)
    ids = {
        "live-a": _decision(store, "live-a", "Live axiom A."),
        "old-b": _decision(store, "old-b", "Retired axiom B."),
        "old-c": _decision(store, "old-c", "Retired axiom C."),
    }
    ids["new-b"] = _decision(store, "new-b", "New axiom B.", supersedes=["old-b"])
    ids["new-c"] = _decision(store, "new-c", "New axiom C.", corrects=["old-c"])
    ids["q-old"] = _open_question(store, "q-old", "Old?")
    ids["q-new"] = _open_question(store, "q-new", "New?", supersedes=["q-old"])
    return ids


def test_status_reports_retired_decisions_missing_and_clears_after_the_heal(
    tmp_path, monkeypatch, capsys
) -> None:
    config = _status_ws(tmp_path, monkeypatch)
    ids = _graveyard(config)
    active = {ids[s] for s in ("live-a", "new-b", "new-c", "q-new")}
    _qdrant(monkeypatch, exists=True, present={hash_to_uuid(i) for i in active})

    rc, payload = _status(config, capsys, as_json=True)
    checks = payload["checks"]
    assert checks["active_nodes"] == 4
    assert checks["missing_active_vectors"] == 0
    # A killed open question is not retired-and-expected (a second derivation would count it).
    assert checks["retired_decisions"] == 2
    assert checks["missing_retired_vectors"] == 2
    assert checks["missing_retired_slugs"] == ["old-b", "old-c"]

    rc_text, out = _status(config, capsys, as_json=False)
    assert "2 retired decision(s) have no vector" in out
    assert f"mitos reconcile {shell_flag('-p', config.project)}" in out
    assert "• old-b" in out and "• old-c" in out
    assert "vector index incomplete" not in out  # the active rung stays quiet

    # The heal: points for the retired ids land; the warning clears.
    healed = active | {ids["old-b"], ids["old-c"]}
    _qdrant(monkeypatch, exists=True, present={hash_to_uuid(i) for i in healed})
    _rc, payload = _status(config, capsys, as_json=True)
    assert payload["checks"]["missing_retired_vectors"] == 0
    assert payload["checks"]["missing_retired_slugs"] == []
    _rc, out = _status(config, capsys, as_json=False)
    assert "retired decision(s) have no vector" not in out


def test_status_the_retired_keys_never_move_readiness(tmp_path, monkeypatch, capsys) -> None:
    config = _status_ws(tmp_path, monkeypatch)
    ids = _graveyard(config)
    active = {ids[s] for s in ("live-a", "new-b", "new-c", "q-new")}

    _qdrant(monkeypatch, exists=True, present={hash_to_uuid(i) for i in active})
    rc_missing, missing = _status(config, capsys, as_json=True)
    _qdrant(monkeypatch, exists=True, present={hash_to_uuid(i) for i in ids.values()})
    rc_whole, whole = _status(config, capsys, as_json=True)

    assert missing["checks"]["missing_retired_vectors"] == 2
    assert whole["checks"]["missing_retired_vectors"] == 0
    assert rc_missing == rc_whole == 0
    assert missing["ready"] is True and whole["ready"] is True


def test_status_a_scroll_failure_leaves_the_retired_keys_unknown(
    tmp_path, monkeypatch, capsys
) -> None:
    config = _status_ws(tmp_path, monkeypatch)
    _graveyard(config)
    _qdrant(monkeypatch, exists=True, scroll_fails=True)

    _rc, payload = _status(config, capsys, as_json=True)
    checks = payload["checks"]
    assert checks["retired_decisions"] == 2  # the graph was read
    assert checks["missing_retired_vectors"] is None
    assert checks["missing_retired_slugs"] is None


def test_status_a_retired_only_graph_over_an_absent_collection(
    tmp_path, monkeypatch, capsys
) -> None:
    """B11 and B13's row: both absent-collection gates key on the expected set."""
    config = _status_ws(tmp_path, monkeypatch)
    _retired_only(GraphStore(config.db_path))
    _qdrant(monkeypatch, exists=False)

    rc, payload = _status(config, capsys, as_json=True)
    checks = payload["checks"]
    assert checks["active_nodes"] == 0
    assert checks["retired_decisions"] == 2
    assert checks["missing_retired_vectors"] == 2
    assert checks["missing_retired_slugs"] == ["cycle-a", "cycle-b"]
    # G9: an absent collection has no points, verified rather than unknown.
    assert checks["orphan_points"] == 0
    # 3c: its partition is judged and empty too.
    assert checks["retired_node_points"] == 0
    assert checks["nodeless_points"] == 0
    assert checks["missing_active_vectors"] == 0

    _rc, out = _status(config, capsys, as_json=False)
    assert "none recorded yet" not in out
    assert "missing — 2 retired decision(s) have no vectors" in out
    assert f"mitos reconcile {shell_flag('-p', config.project)}" in out


def test_status_key_order_is_identical_in_every_branch(tmp_path, monkeypatch, capsys) -> None:
    """G5: the cluster checkpoints compare ``list(checks)``, order included."""
    orders = []

    fresh = _status_ws(tmp_path / "fresh", monkeypatch)
    _qdrant(monkeypatch, exists=False)
    orders.append(list(_status(fresh, capsys, as_json=True)[1]["checks"]))

    full = _status_ws(tmp_path / "full", monkeypatch)
    _graveyard(full)
    for exists, fails in ((True, False), (True, True), (False, False)):
        _qdrant(monkeypatch, exists=exists, present=set(), scroll_fails=fails)
        orders.append(list(_status(full, capsys, as_json=True)[1]["checks"]))

    assert all(order == orders[0] for order in orders)
    for key in _NEW_KEYS:
        assert key in orders[0]
    assert orders[0].index("missing_retired_slugs") < orders[0].index("orphan_points")
    # 3c: the orphan count's partition sits immediately after it.
    at = orders[0].index("orphan_points")
    assert orders[0][at + 1:at + 3] == ["retired_node_points", "nodeless_points"]


def test_status_a_fresh_project_reports_no_retired_decisions(tmp_path, monkeypatch, capsys) -> None:
    config = _status_ws(tmp_path, monkeypatch)
    _qdrant(monkeypatch, exists=False)

    rc, payload = _status(config, capsys, as_json=True)
    assert rc == 0
    assert payload["checks"]["retired_decisions"] == 0
    assert payload["checks"]["missing_retired_vectors"] is None
    _rc, out = _status(config, capsys, as_json=False)
    assert "auto-created on first record — none recorded yet" in out


# --------------------------------------------------------------------------- #
# Shipped prose (constraint 12): the active-only bound is gone where 3b ships the change
# --------------------------------------------------------------------------- #

def _collapsed(text: str) -> str:
    return " ".join(text.replace(">", " ").split())


def test_setup_runbook_names_every_decision_not_the_active_node_set() -> None:
    setup = _collapsed((REPO / "SETUP.md").read_text(encoding="utf-8"))
    assert "it diffs the active node set" not in setup
    assert "Re-embed the V1a active set" not in setup
    assert ("it diffs every decision (live or retired) plus the active open questions "
            "against Qdrant's actual points") in setup
    # 3c inverted the box's silence on deletion: reconcile now collects node-less points,
    # over a built graph only, and names `rebuild` over an unbuilt one.
    assert ("`reconcile` also deletes points whose node is no longer in the graph, over "
            "a built graph only: over a clone whose graph is unbuilt it deletes nothing "
            "and names `mitos rebuild` instead.") in setup
    # 4d rewrote the cache clause, true for both indexes: decision re-embeds hit the
    # cache, rejected items bypass it and `status` prices the build.
    assert "re-embeds hit the cache, so no embedding-API spend" not in setup
    assert ("Decision re-embeds hit the embedding cache, so they cost no embedding-API "
            "spend when text is unchanged; rejected items bypass the cache, so a first "
            "build or a lost rejection collection pays one embedding per item") in setup
    assert "It builds the rejection index in the same pass" in setup


def test_the_reconcile_help_names_every_decision() -> None:
    parser = cli._build_parser()
    action = next(a for a in parser._subparsers._group_actions)  # type: ignore[attr-defined]
    helps = {choice.dest: choice.help for choice in action._choices_actions}
    assert "Re-embed active nodes" not in helps["reconcile"]
    assert "every decision (live or retired)" in helps["reconcile"]
    # 3c: the help names the collection too.
    assert "delete points whose node has left the graph" in helps["reconcile"]
    # 4d: and the rejection index it builds and collects.
    assert "Also builds the rejection index and deletes rejected items" in helps["reconcile"]


def test_the_upgrade_note_orders_the_server_first_and_states_the_egress() -> None:
    """The upgrade note as 4d ships it: server first, one reconcile, egress stated."""
    setup = _collapsed((REPO / "SETUP.md").read_text(encoding="utf-8"))
    head = "## Upgrading a workspace written by an earlier Mitos"
    assert head in setup and setup.index(head) < setup.index("## Cutover")
    note = setup[setup.index(head):setup.index("## Cutover")]
    server = note.index("**Move every build that reads a workspace first.**")
    # `_collapsed` blanks every ">", so the placeholder reads "<project " here.
    reconcile = note.index("**Then run one `mitos reconcile -p <project ` per workspace.**")
    assert server < reconcile
    assert ("It embeds the retired decisions the decision index lacks and builds the "
            "rejection index in the same pass") in note
    assert ("Each decision's `rejected_paths` is now sent to the embedding provider "
            "beside its axiom.") in note
    # True at 4d: nothing here may claim a read finds or hides rejected items (5a).
    for word in ("surface", "query", "match", "hide"):
        assert word not in note.lower(), word


def test_the_cutover_finish_block_names_every_decision() -> None:
    source = inspect.getsource(cli.cmd_cutover)
    assert "Re-embed the V1a active set" not in source
    assert "re-embeds the active set" not in source
    assert "`mitos reconcile` embeds every missing decision in one pass." in source


def test_the_mcp_reconcile_clause_names_every_decision() -> None:
    from mitos.lexical import _RECOVERY_CLAUSES

    clause = _RECOVERY_CLAUSES["mcp"]["reconcile"]
    assert "re-embeds the active decisions" not in clause
    assert "embeds every decision it is missing" in clause
    assert "mitos reconcile" not in clause  # a fact with a person as the next actor


def test_the_vector_store_refusal_names_the_expected_set(monkeypatch) -> None:
    from unittest.mock import MagicMock

    from mitos.errors import CollectionMissingError
    from mitos.vector_store import QdrantVectorStore

    monkeypatch.setattr("mitos.vector_store.requests.put",
                        lambda *a, **k: MagicMock(status_code=404, text="Not found"))
    store = QdrantVectorStore("http://localhost:9", "mitos-tmp-x")
    with pytest.raises(CollectionMissingError) as excinfo:
        store.upsert("a" * 64, [0.1], {}, may_create=False)

    message = str(excinfo.value)
    assert "does not cover every decision the index should hold" in message
    assert "active set" not in message


# --------------------------------------------------------------------------- #
# The rebuild seed, drained: W17
# --------------------------------------------------------------------------- #

def test_rebuild_seeds_every_decision_and_the_drain_embeds_only_the_retired(tmp_path) -> None:
    """``rebuild`` leaves the outbox holding the expected set under its marker; the next
    ``sync --embed-only`` over a collection holding the active points presence-skips them
    and embeds exactly the retired decisions."""
    from mitos.cutover import perform_swap, rebuild_and_gate
    from test_outbox_drain import _Embed, _Vector, _manager, _sync_embed_only

    config = _spy_corpus(tmp_path)
    result = rebuild_and_gate(config, aside_db_path=config.db_path + ".aside", strict=False)
    aside = GraphStore(result.aside_db_path)
    assert _slugs(aside, [r["node_id"] for r in aside.get_pending_embeddings()]) == {
        "d-new", "d-old", "d-live", "q-new"}  # q-old, superseded, is pruned
    assert aside.embedding_seed()["established_by"] == "rebuild"
    perform_swap(config, result.aside_db_path, timestamp="20261016-000001")

    store = GraphStore(config.db_path)
    active = store.get_active_node_ids()
    retired = store.get_decision_index_expected_ids() - active
    assert _slugs(store, retired) == {"d-old"}

    vector, embed = _Vector(present=tuple(active)), _Embed()
    outcome = _sync_embed_only(config, _manager(config, vector, embed))

    assert outcome.skipped_present == len(active) == 3
    assert outcome.written == len(retired) == 1
    assert len(embed.calls) == 1
    assert vector.written == list(retired)
    assert store.get_pending_embeddings() == []
