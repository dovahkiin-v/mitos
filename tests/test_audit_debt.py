"""Tests for the audit-debt leaf — the one derivation of uncovered decisions.

``mitos.audit_debt.derive_audit_debt`` turns a graph path and a telemetry path into
the active decisions no undegraded check has seen (N of M, with K excluded beside
it) and a membership fingerprint, or *no graph*, or *unreadable* — never a false
zero. The rows here pin:

* the population: ``GraphStore.get_active_decision_ids`` equals the id set of
  ``get_active_decisions`` (the sweep's own population) across a fixture matrix;
* the arithmetic: set subtraction over live ids, so stale coverage rows (superseded
  or vanished nodes) never move a count;
* the three results, including absent telemetry reading N = M and either file being
  unreadable reading as a fault;
* the fingerprint: over membership, order-free, on a pinned recipe;
* read-only on both files, and the leaf's fences (import closure, no env/cwd reads).

One row drives a real ``cmd_check`` so the ids ``check`` writes are proven to be the
ids the leaf subtracts. Real SQLite throughout; no mocks.
"""

import ast
import hashlib
import os
import sqlite3
import subprocess
import sys
from typing import Tuple

import pytest

from mitos.audit_debt import (
    AuditDebt,
    DebtUnreadable,
    NoGraph,
    derive_audit_debt,
    uncovered_fingerprint,
)
from mitos.migrations import run_migrations
from mitos.store import GraphStore, open_connection
from mitos.telemetry import TELEMETRY_MIGRATION_STEPS, CoverageMarks, TelemetryStore

from _conflict_helpers import _drain_outbox
from test_check_coverage import (  # noqa: F401  (offline is autouse; workspace a fixture)
    _check_run_row,
    _judge_for,
    _pair,
    _run,
    _run_row,
    _wire_judge,
    _wire_substrate,
    offline,
    workspace,
)
from test_check_probe import _commit
from test_embedding_seed import _open_question

_LEAF_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "mitos", "audit_debt.py"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

@pytest.fixture
def paths(tmp_path) -> Tuple[GraphStore, str]:
    """A built graph in ``.mitos/`` and the (not yet created) telemetry path beside it."""
    db_path = str(tmp_path / ".mitos" / "graph.sqlite")
    return GraphStore(db_path), str(tmp_path / ".mitos" / "telemetry.sqlite")


def _mark(telemetry_path: str, run_id: str, *, covered=(), excluded=()) -> None:
    """Writes one run's coverage marks through the real run-end seam."""
    TelemetryStore(telemetry_path).record_run_end(
        _check_run_row(run_id),
        coverage=CoverageMarks(run_id=run_id, marked_at="t",
                               covered=tuple(covered), excluded=tuple(excluded)),
        attempt=None,
    )


def _derive(store: GraphStore, telemetry_path: str):
    return derive_audit_debt(store.db_path, telemetry_path)


def _counts(debt) -> Tuple[int, int, int]:
    assert isinstance(debt, AuditDebt), debt
    return debt.uncovered, debt.total, debt.excluded


def _snapshot(path: str) -> Tuple[bytes, int]:
    """The database file's bytes and ``user_version`` (siblings deliberately ignored)."""
    with open(path, "rb") as f:
        data = f.read()
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()
    return data, version


def _below_rung(path: str) -> None:
    """A rung-4 telemetry file: it predates the coverage table."""
    conn = open_connection(path)
    run_migrations(conn, TELEMETRY_MIGRATION_STEPS[:4])
    conn.close()


# --------------------------------------------------------------------------- #
# 1 — the population agrees with the sweep's
# --------------------------------------------------------------------------- #

def test_the_ids_read_agrees_with_the_active_decisions_across_the_matrix(paths) -> None:
    """Criterion 1: ids-only read == ``get_active_decisions()``' id set, at every step.

    The matrix holds each thing a wrong read would get wrong: a parked and a resolved
    open question (a widened kind test lets them in), a superseded and a corrected
    decision (an aliased ``nodes`` or a supersedes-only kill set lets them back), and
    a drifted one (it must stay in both).
    """
    store, _ = paths

    def _agrees() -> frozenset:
        ids = store.get_active_decision_ids()
        assert isinstance(ids, frozenset)
        assert ids == {n["id"] for n in store.get_active_decisions()}
        return ids

    assert _agrees() == frozenset()  # empty graph
    d1 = _commit(store, "d1", "Axiom one.")
    assert _agrees() == {d1}
    _open_question(store, "q1", "Caching", "Which cache?")
    assert _agrees() == {d1}
    d2 = _commit(store, "d2", "Axiom two.", resolves=["q1"])
    assert _agrees() == {d1, d2}
    d3 = _commit(store, "d3", "Axiom three.", supersedes=["d1"])
    assert _agrees() == {d2, d3}
    d4 = _commit(store, "d4", "Axiom four.", corrects=["d2"])
    assert _agrees() == {d3, d4}
    store.write_signal(d3, "drifted")
    assert _agrees() == {d3, d4}
    assert [n["is_drifted"] for n in store.get_active_decisions() if n["id"] == d3] == [True]


# --------------------------------------------------------------------------- #
# 2–4 — the arithmetic is set subtraction over live ids
# --------------------------------------------------------------------------- #

def test_counts_over_a_mixed_fixture(paths) -> None:
    """Criterion 2: N = never-swept actives, M = active decisions, K = active ∩ excluded.

    Two superseded decisions hold rows (one ``covered``, one ``excluded``) from when
    they were active; neither may move a count.
    """
    store, tel = paths
    covered = _commit(store, "covered", "Covered axiom.")
    excluded = _commit(store, "excluded", "Excluded axiom.")
    never = _commit(store, "never", "Never swept axiom.")
    old_cov = _commit(store, "old-cov", "Old covered axiom.")
    old_exc = _commit(store, "old-exc", "Old excluded axiom.")
    _open_question(store, "q1", "Caching", "Which cache?")
    _mark(tel, "r1", covered=[covered, old_cov], excluded=[excluded, old_exc])
    new_cov = _commit(store, "new-cov", "Newer covered axiom.", supersedes=["old-cov"])
    new_exc = _commit(store, "new-exc", "Newer excluded axiom.", supersedes=["old-exc"])
    _mark(tel, "r2", covered=[new_cov, new_exc])

    debt = _derive(store, tel)

    assert debt.uncovered_ids == {never}
    assert _counts(debt) == (1, 5, 1)


def test_one_poison_node_does_not_hold_the_count_above_zero(paths) -> None:
    """Criterion 3: everything covered but one excluded → N = 0, K = 1."""
    store, tel = paths
    a = _commit(store, "a", "Axiom a.")
    b = _commit(store, "b", "Axiom b.")
    poison = _commit(store, "poison", "Poison axiom.")
    _mark(tel, "r1", covered=[a, b], excluded=[poison])

    assert _counts(_derive(store, tel)) == (0, 3, 1)


def test_rows_for_ids_no_longer_live_are_harmless(paths) -> None:
    """Criterion 4: a row for a vanished id and one for a superseded id move nothing."""
    store, tel = paths
    _commit(store, "old", "Old axiom.")
    old = store.get_active_decision_ids()
    a = _commit(store, "a", "Axiom a.")
    b = _commit(store, "b", "Axiom b.", supersedes=["old"])
    _mark(tel, "r1", covered=[a, b, "f" * 64, *old], excluded=["e" * 64])

    assert _counts(_derive(store, tel)) == (0, 2, 0)


# --------------------------------------------------------------------------- #
# 5–8 — the three results, never a false zero
# --------------------------------------------------------------------------- #

def test_an_empty_corpus_is_a_built_graph_not_no_graph(paths) -> None:
    """Criterion 5: no decisions (an open question allowed) → ``AuditDebt(∅, 0, 0)``."""
    store, tel = paths
    _open_question(store, "q1", "Caching", "Which cache?")

    for state in ("absent", "present"):
        if state == "present":
            _mark(tel, "r1", covered=["f" * 64])
        debt = _derive(store, tel)
        assert debt == AuditDebt(frozenset(), 0, 0)
        assert debt.fingerprint == uncovered_fingerprint([])


def test_absent_telemetry_reads_every_active_decision_uncovered(paths) -> None:
    """Criterion 6: no file, and a file below the coverage rung, both read N = M."""
    store, tel = paths
    a = _commit(store, "a", "Axiom a.")
    b = _commit(store, "b", "Axiom b.")

    debt = _derive(store, tel)
    assert debt.uncovered_ids == {a, b} and _counts(debt) == (2, 2, 0)
    assert not os.path.exists(tel)

    _below_rung(tel)
    before = _snapshot(tel)
    debt = _derive(store, tel)
    assert debt.uncovered_ids == {a, b} and _counts(debt) == (2, 2, 0)
    assert _snapshot(tel) == before
    assert before[1] == 4


@pytest.mark.parametrize("fault", ["corrupt", "directory"])
def test_unreadable_telemetry_is_a_fault_not_a_zero(paths, fault) -> None:
    """Criterion 7, telemetry side: never an ``AuditDebt``, never a raise."""
    store, tel = paths
    _commit(store, "a", "Axiom a.")
    if fault == "corrupt":
        with open(tel, "wb") as f:
            f.write(b"this is not a sqlite database, not even close" * 20)
    else:
        os.mkdir(tel)

    debt = _derive(store, tel)
    assert isinstance(debt, DebtUnreadable) and debt.source == "telemetry"
    assert debt.detail


@pytest.mark.parametrize("fault", ["corrupt", "directory", "empty"])
def test_unreadable_graph_is_a_fault_not_a_zero(tmp_path, fault) -> None:
    """Criterion 7, graph side: a corrupt image, a directory or a tableless file.

    A 0-byte graph is *unreadable* on purpose, unlike telemetry's (which has a rung
    ladder defining "holds nothing yet"): a graph with no ``nodes`` table is a broken
    or unbuilt workspace.
    """
    db = tmp_path / ".mitos" / "graph.sqlite"
    db.parent.mkdir()
    tel = str(tmp_path / ".mitos" / "telemetry.sqlite")
    if fault == "corrupt":
        db.write_bytes(b"this is not a sqlite database, not even close" * 20)
    elif fault == "directory":
        db.mkdir()
    else:
        db.write_bytes(b"")

    debt = derive_audit_debt(str(db), tel)
    assert isinstance(debt, DebtUnreadable) and debt.source == "graph"
    assert debt.detail
    assert not os.path.exists(tel)  # telemetry never touched on a graph fault


def test_a_missing_graph_is_no_graph_and_creates_nothing(tmp_path) -> None:
    """Criterion 8: no graph file → ``NoGraph``; telemetry unopened; no ``.mitos/`` made.

    The telemetry path is a directory, which would read *unreadable* if it were
    opened.
    """
    telemetry_dir = tmp_path / "elsewhere" / "telemetry.sqlite"
    telemetry_dir.mkdir(parents=True)
    workspace_dir = tmp_path / "ws"
    workspace_dir.mkdir()
    db_path = str(workspace_dir / ".mitos" / "graph.sqlite")
    before = sorted(os.walk(tmp_path))

    assert derive_audit_debt(db_path, str(telemetry_dir)) == NoGraph()
    assert not (workspace_dir / ".mitos").exists()
    assert sorted(os.walk(tmp_path)) == before


# --------------------------------------------------------------------------- #
# 9 — the join key, through the outermost frame
# --------------------------------------------------------------------------- #

def test_the_ids_check_writes_are_the_ids_the_leaf_subtracts(
    workspace, monkeypatch, capsys
) -> None:
    """Criterion 9: seed three → (3,3,0); a real undegraded check → (0,3,0); a fourth → (1,4,0).

    A hand-seeded coverage fixture cannot prove this: it would use whatever ids the
    test chose. Here ``cmd_check`` writes them.
    """
    config, store, tel = workspace
    _, _, nbhds = _pair(store)
    third_axiom = "Corpus axiom gamma for the check verb."
    _commit(store, "cli-c", third_axiom)
    nbhds[third_axiom] = []
    _drain_outbox(store)

    assert _counts(derive_audit_debt(config.db_path, config.telemetry_path)) == (3, 3, 0)

    embed, vector = _wire_substrate(monkeypatch, nbhds)
    _wire_judge(monkeypatch, _judge_for(store, embed, vector, tel))
    code, obj = _run(config, capsys)
    assert code in (0, 1) and obj["degradations"] == []
    assert _run_row(config, obj["run_id"])["coverage_exclusions"] == 0

    assert _counts(derive_audit_debt(config.db_path, config.telemetry_path)) == (0, 3, 0)

    fourth = _commit(store, "cli-d", "Corpus axiom delta for the check verb.")
    debt = derive_audit_debt(config.db_path, config.telemetry_path)
    assert _counts(debt) == (1, 4, 0)
    assert debt.fingerprint == uncovered_fingerprint([fourth])


# --------------------------------------------------------------------------- #
# 10 — the fingerprint is over membership
# --------------------------------------------------------------------------- #

def test_the_fingerprint_sees_a_swap_the_counts_cannot(paths) -> None:
    """Criterion 10: one uncovered decision leaves, one arrives — same counts, new print."""
    store, tel = paths
    a = _commit(store, "a", "Axiom a.")
    _commit(store, "b", "Axiom b.")
    _mark(tel, "r1", covered=[a])
    before = _derive(store, tel)

    _commit(store, "b2", "Axiom b, revised.", supersedes=["b"])
    after = _derive(store, tel)

    assert _counts(before) == _counts(after) == (1, 2, 0)
    assert before.uncovered_ids != after.uncovered_ids
    assert before.fingerprint != after.fingerprint


def test_the_fingerprint_is_order_free_and_on_the_pinned_recipe() -> None:
    """Criterion 10 / §8: sha256 over the sorted ids joined by newline, lowercase hex."""
    ids = [hashlib.sha256(s.encode()).hexdigest() for s in ("x", "y", "z")]
    expected = hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()

    assert uncovered_fingerprint(ids) == expected
    assert uncovered_fingerprint(reversed(ids)) == expected
    assert uncovered_fingerprint(frozenset(ids)) == expected
    assert AuditDebt(frozenset(ids), 3, 0).fingerprint == expected
    assert uncovered_fingerprint([]) == hashlib.sha256(b"").hexdigest()
    assert len(expected) == 64 and expected == expected.lower()


# --------------------------------------------------------------------------- #
# 11 — read-only on both files
# --------------------------------------------------------------------------- #

def test_a_derive_leaves_both_database_files_untouched(paths) -> None:
    """Criterion 11: bytes and ``user_version`` of both files unchanged.

    Only the database files are compared: a read-only WAL open may touch the
    ``-shm``/``-wal`` siblings.
    """
    store, tel = paths
    a = _commit(store, "a", "Axiom a.")
    _commit(store, "b", "Axiom b.")
    _mark(tel, "r1", covered=[a])
    graph_before, tel_before = _snapshot(store.db_path), _snapshot(tel)

    assert _counts(_derive(store, tel)) == (1, 2, 0)

    assert _snapshot(store.db_path) == graph_before
    assert _snapshot(tel) == tel_before


# --------------------------------------------------------------------------- #
# 12 — the fences
# --------------------------------------------------------------------------- #

def _loaded_modules(statement: str) -> set:
    probe = (
        f"import sys; {statement}; "
        "print('\\n'.join(sorted(m for m in sys.modules "
        "if m == 'mitos' or m.startswith('mitos.') "
        "or m in ('anthropic', 'google.genai', 'qdrant_client', 'numpy'))))"
    )
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def test_the_leaf_adds_nothing_to_the_store_and_telemetry_closure() -> None:
    """Fence (b): the leaf's import closure is ``store`` + ``telemetry``'s, plus itself.

    The baseline is measured, not hand-typed, so a new module in ``store``'s closure
    moves both sides and only an import the leaf itself adds goes red.
    """
    baseline = _loaded_modules("import mitos.store, mitos.telemetry")
    leaf = _loaded_modules("import mitos.audit_debt")

    assert "mitos.audit_debt" in leaf
    assert leaf - baseline == {"mitos.audit_debt"}
    for forbidden in ("anthropic", "google.genai", "mitos.cli", "mitos.sync",
                      "mitos.check", "mitos.config", "mitos.mcp_server"):
        assert forbidden not in leaf


def test_the_leaf_reads_no_environment_and_no_working_directory() -> None:
    """Fence (c): callers pass both paths; the leaf never looks around for them."""
    with open(_LEAF_PATH, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    forbidden = {"environ", "getenv", "getcwd", "getcwdb", "environb"}
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in forbidden:
            hits.append(node.attr)
        elif isinstance(node, ast.Name) and node.id in forbidden:
            hits.append(node.id)
        elif isinstance(node, ast.ImportFrom):
            hits.extend(a.name for a in node.names if a.name in forbidden)
    assert hits == []
