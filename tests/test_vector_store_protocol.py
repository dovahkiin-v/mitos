"""Phase 2a: every ``VectorStore.query`` in the tree honours the whole protocol.

Once readers resolve hits by point id, a double that returned no id would empty every
window it fed, and a row asserting an absence would pass over the window its double
emptied (vision §6.2 constraint 8). A test-only implementation satisfies the protocol's
whole contract, including the fields its own rows never read. Three rows hold that:

- **Structural.** An AST walk over ``tests/`` and ``mitos/`` finds every class defining
  ``query`` (nested and function-local classes included — four of them cannot be
  imported, so ``inspect.signature`` cannot reach them) and asserts that the set whose
  parameters differ from ``VectorStore.query``'s is empty. A future double conforms by
  arriving. Parameters compare by name, kind and default, never annotation.
- **Second clause.** Every such class under ``tests/`` raises only, returns a literal
  ``[]``, wraps an inner store (``self._inner``), or builds its hits through the shared
  leaf (``tests/_point_hits.py``); a hit-returning double cannot bypass the leaf.
- **Behavioural.** The shared doubles' hits carry a string ``point_id`` that resolves,
  through ``resolve_point_ids`` on the bound graph, to the node intended, with the
  production key sets and ``limit`` honoured.

The fixture-fault teardown (conftest's ``point_hit_fixture_faults``) is pinned last, in
a pytest subprocess, because its subject is a test failing.
"""

import ast
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from _point_hits import (FULL_HIT_KEYS, IDS_ONLY_HIT_KEYS, CannedVector, orphan_hit,
                         point_hit, take_faults)
from mitos.parser import ParsedEntry
from mitos.protocols import VectorStore
from mitos.store import GraphStore
from mitos.vector_store import hash_to_uuid

TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parent
MITOS_DIR = REPO / "mitos"

#: The leaf itself defines the canned ``query``; it is the clause's reference point.
LEAF_FILE = TESTS_DIR / "_point_hits.py"
#: Names whose call inside a double's ``query`` routes its hits through the leaf.
LEAF_BUILDERS = {"point_hit", "orphan_hit", "project_hit", "derive_hits"}

#: Found today: 19 (the protocol, the client, the leaf, fifteen doubles and this
#: module's own wrapper). A floor,
#: not an equality, so a new double does not red this row — only a non-conforming one.
MIN_QUERY_CLASSES = 15

Param = Tuple[str, str, Any]


# --------------------------------------------------------------------------- #
# The structural row
# --------------------------------------------------------------------------- #

_NO_DEFAULT = "<no default>"


def _reference_params() -> List[Param]:
    sig = inspect.signature(VectorStore.query)
    return [(p.name, p.kind.name,
             _NO_DEFAULT if p.default is inspect.Parameter.empty else p.default)
            for p in sig.parameters.values()]


def _ast_params(fn: ast.FunctionDef) -> List[Param]:
    args = fn.args
    positional = [*args.posonlyargs, *args.args]
    defaults: List[Any] = [_NO_DEFAULT] * (len(positional) - len(args.defaults)) + [
        ast.literal_eval(d) for d in args.defaults]
    out: List[Param] = []
    for i, arg in enumerate(positional):
        kind = "POSITIONAL_ONLY" if i < len(args.posonlyargs) else "POSITIONAL_OR_KEYWORD"
        out.append((arg.arg, kind, defaults[i]))
    if args.vararg is not None:
        out.append((args.vararg.arg, "VAR_POSITIONAL", _NO_DEFAULT))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        out.append((arg.arg, "KEYWORD_ONLY",
                    _NO_DEFAULT if default is None else ast.literal_eval(default)))
    if args.kwarg is not None:
        out.append((args.kwarg.arg, "VAR_KEYWORD", _NO_DEFAULT))
    return out


def query_classes(source: str) -> List[Tuple[str, ast.FunctionDef]]:
    """Every class in ``source`` defining ``query``, nested ones included."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "query":
                    found.append((node.name, item))
    return found


def _tree_files() -> List[Path]:
    return sorted([*TESTS_DIR.rglob("*.py"), *MITOS_DIR.rglob("*.py")])


def nonconforming(files: List[Path], reference: List[Param]) -> List[str]:
    out = []
    for path in files:
        for cls, fn in query_classes(path.read_text(encoding="utf-8")):
            params = _ast_params(fn)
            if params != reference:
                out.append(f"{path.relative_to(REPO)}::{cls}.query{params}")
    return out


def test_every_query_in_the_tree_has_the_protocols_signature() -> None:
    reference = _reference_params()
    assert reference == [
        ("self", "POSITIONAL_OR_KEYWORD", _NO_DEFAULT),
        ("vector", "POSITIONAL_OR_KEYWORD", _NO_DEFAULT),
        ("limit", "POSITIONAL_OR_KEYWORD", 5),
        ("with_payload", "KEYWORD_ONLY", True),
    ]
    files = _tree_files()
    found = [(p, c) for p in files for c, _ in query_classes(p.read_text(encoding="utf-8"))]
    assert len(found) >= MIN_QUERY_CLASSES, found

    assert nonconforming(files, reference) == []


def test_the_structural_row_reports_a_planted_double(tmp_path) -> None:
    """Positive control: today's old signature, nested inside a test function."""
    planted = tmp_path / "test_planted.py"
    planted.write_text(textwrap.dedent('''
        def test_x():
            class _Old:
                def query(self, vector, limit=5):
                    return []
            class _Kwargs:
                def query(self, vector, limit=5, **kwargs):
                    return []
    '''), encoding="utf-8")
    reference = _reference_params()
    reported = [line.split("::")[1].split(".query")[0] for line in
                nonconforming_from(planted, reference)]
    assert reported == ["_Old", "_Kwargs"]


def nonconforming_from(path: Path, reference: List[Param]) -> List[str]:
    return [f"{path.name}::{cls}.query{_ast_params(fn)}"
            for cls, fn in query_classes(path.read_text(encoding="utf-8"))
            if _ast_params(fn) != reference]


# --------------------------------------------------------------------------- #
# The second clause — hit-returning doubles go through the leaf
# --------------------------------------------------------------------------- #

def _routes_honestly(fn: ast.FunctionDef) -> bool:
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
    if all(isinstance(r.value, ast.List) and not r.value.elts for r in returns):
        return True  # raises only, or a literal []
    for node in ast.walk(fn):
        if (isinstance(node, ast.Attribute) and node.attr == "_inner"
                and isinstance(node.value, ast.Name) and node.value.id == "self"):
            return True  # wraps an inner store
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name in LEAF_BUILDERS:
                return True  # builds its hits through the leaf
    return False


def bypassing(files: List[Path]) -> List[str]:
    return [f"{path.relative_to(REPO) if path.is_relative_to(REPO) else path.name}::{cls}"
            for path in files
            for cls, fn in query_classes(path.read_text(encoding="utf-8"))
            if not _routes_honestly(fn)]


def test_every_hit_returning_double_routes_through_the_shared_leaf() -> None:
    doubles = [p for p in TESTS_DIR.rglob("*.py") if p != LEAF_FILE]
    assert bypassing(doubles) == []


def test_the_second_clause_reports_a_bypassing_double(tmp_path) -> None:
    planted = tmp_path / "test_bypass.py"
    planted.write_text(textwrap.dedent('''
        class _Replay:
            def query(self, vector, limit=5, *, with_payload=True):
                return self._matches[:limit]
        class _Fine:
            def query(self, vector, limit=5, *, with_payload=True):
                return [project_hit(h, with_payload) for h in self._hits]
    '''), encoding="utf-8")
    assert bypassing([planted]) == ["test_bypass.py::_Replay"]


# --------------------------------------------------------------------------- #
# Behavioural conformance of the shared doubles
# --------------------------------------------------------------------------- #

def _commit(store: GraphStore, slug: str, axiom: str) -> str:
    entry = ParsedEntry("decision", slug, 1, 10)
    entry.axiom = axiom
    entry.rejected_paths = "Rejected the obvious alternative."
    entry.scope = ["core"]
    return store.commit_parsed_entry(entry).node_id


class _AskWith:
    """Wraps a double so every query asks it with one fixed vector (a sentinel the
    keyed double knows, or the stored points' query direction)."""

    def __init__(self, inner: Any, fixed: List[float]) -> None:
        self._inner = inner
        self._fixed = fixed

    def query(self, vector: List[float], limit: int = 5, *,
              with_payload: bool = True) -> List[Dict[str, Any]]:
        return self._inner.query(self._fixed, limit, with_payload=with_payload)


def _canned(store: GraphStore, matches: List[Dict[str, Any]]) -> Any:
    return CannedVector(matches, store=store)


def _conflict_fake(store: GraphStore, matches: List[Dict[str, Any]]) -> Any:
    from _conflict_helpers import _FakeVector
    return _FakeVector(matches, store=store)


def _keyed(store: GraphStore, matches: List[Dict[str, Any]]) -> Any:
    from _conflict_helpers import _keyed_substrate
    embed, vector = _keyed_substrate({"the axiom": matches}, store=store)
    sentinel = embed.get_embedding("the axiom")

    return _AskWith(vector, sentinel)


def _stdio_fake(store: GraphStore, matches: List[Dict[str, Any]]) -> Any:
    """``FakeVectorStore`` runs in the serve child in production of its rows; its
    conformance is provable only in-process (scout W4), so it is built here directly."""
    from _stdio_fake_providers import FakeVectorStore

    collection = f"conformance-{id(store)}"
    FakeVectorStore._collections.pop(collection, None)
    fake = FakeVectorStore("http://unused", collection)
    query_vector = [1.0, 0.0]
    for rank, match in enumerate(matches):
        node_id = store.resolve_slug(match["slug"])[0]
        # Each stored vector sits progressively further from the query.
        vector = [1.0, float(rank)]
        fake.upsert(node_id, vector, {"slug": match["slug"], "state": "active",
                                      "kind": "decision", "scope": ["core"],
                                      "embedding_text": match["slug"]}, may_create=True)

    return _AskWith(fake, query_vector)


@pytest.mark.parametrize("build", [_canned, _conflict_fake, _keyed, _stdio_fake],
                         ids=["CannedVector", "_conflict_helpers._FakeVector",
                              "_KeyedVector", "FakeVectorStore"])
def test_shared_doubles_honour_the_whole_contract(tmp_path, build) -> None:
    store = GraphStore(str(tmp_path / "graph.sqlite"))
    slugs = ["first-node", "second-node", "third-node"]
    ids = {slug: _commit(store, slug, f"Axiom of {slug}.") for slug in slugs}
    matches = [{"slug": s, "score": 0.9 - 0.1 * i} for i, s in enumerate(slugs)]
    double = build(store, matches)

    full = double.query([0.1, 0.2], limit=3)
    assert [set(h) for h in full] == [FULL_HIT_KEYS] * 3
    assert all(isinstance(h["point_id"], str) for h in full)
    resolved = store.resolve_point_ids([h["point_id"] for h in full])
    assert [resolved.nodes[h["point_id"]]["id"] for h in full] == [ids[s] for s in slugs]
    assert [h["slug"] for h in full] == slugs

    ids_only = double.query([0.1, 0.2], limit=3, with_payload=False)
    assert [set(h) for h in ids_only] == [IDS_ONLY_HIT_KEYS] * 3
    assert [h["point_id"] for h in ids_only] == [h["point_id"] for h in full]

    assert len(double.query([0.1, 0.2], limit=2)) == 2
    assert take_faults() == []


# --------------------------------------------------------------------------- #
# The leaf's fault and binding rules (consumed in-row; the teardown is pinned below)
# --------------------------------------------------------------------------- #

def test_a_slug_naming_no_node_is_a_recorded_fault_never_a_raise(tmp_path) -> None:
    store = GraphStore(str(tmp_path / "graph.sqlite"))
    known = _commit(store, "known", "A known node.")
    vector = CannedVector([{"slug": "nobody", "score": 0.9}, {"slug": "known", "score": 0.8}],
                          store=store)

    hits = vector.query([0.1])

    assert [h["point_id"] for h in hits] == [hash_to_uuid(known)]
    faults = take_faults()
    assert len(faults) == 1 and "'nobody' names no node" in faults[0]


def test_a_slug_held_by_two_nodes_is_a_recorded_fault(tmp_path) -> None:
    """G3: today's join silently picks the live one; the leaf makes the fixture name it."""
    from test_point_identity import same_slug_lineage

    store = GraphStore(str(tmp_path / "graph.sqlite"))
    retired, successor = same_slug_lineage(store)

    assert CannedVector([{"slug": "database", "score": 0.9}], store=store).query([0.1]) == []
    faults = take_faults()
    assert len(faults) == 1 and "names 2 nodes" in faults[0]
    # The fixture names the node it means instead.
    named = CannedVector([point_hit(successor, 0.9, slug="database")], store=store)
    assert named.query([0.1])[0]["point_id"] == hash_to_uuid(successor)


def test_binding_falls_back_only_to_one_graph(tmp_path) -> None:
    writable = GraphStore(str(tmp_path / "a.sqlite"))
    node = _commit(writable, "only-node", "The only node.")
    GraphStore(str(tmp_path / "a.sqlite"), read_only=True)  # the same graph, read-only
    assert CannedVector([{"slug": "only-node", "score": 0.9}]).query([0.1])[0]["point_id"] \
        == hash_to_uuid(node)
    assert take_faults() == []

    GraphStore(str(tmp_path / "b.sqlite"))  # a second graph: the fallback refuses to guess
    assert CannedVector([{"slug": "only-node", "score": 0.9}]).query([0.1]) == []
    faults = take_faults()
    assert len(faults) == 1 and "2 distinct graphs" in faults[0]
    # An explicit store always wins.
    assert len(CannedVector([{"slug": "only-node", "score": 0.9}], store=writable)
               .query([0.1])) == 1


def test_no_recorded_graph_is_a_fault() -> None:
    assert CannedVector([{"slug": "x", "score": 0.9}]).query([0.1]) == []
    faults = take_faults()
    assert len(faults) == 1 and "no store to bind" in faults[0]


def test_declared_orphans_and_raw_hits_pass_through_without_a_graph() -> None:
    hits = CannedVector([orphan_hit(0.7, slug="gone"),
                         point_hit("ab" * 32, 0.6, slug="raw")]).query([0.1])
    assert [h["slug"] for h in hits] == ["gone", "raw"]
    assert take_faults() == []


def test_an_unknown_keyed_vector_is_a_recorded_fault_not_a_key_error(tmp_path) -> None:
    from _conflict_helpers import _keyed_substrate

    GraphStore(str(tmp_path / "graph.sqlite"))
    _embed, vector = _keyed_substrate({"known text": []})
    assert vector.query([99.0, 0.5]) == []
    faults = take_faults()
    assert len(faults) == 1 and "no canned matches" in faults[0]


# --------------------------------------------------------------------------- #
# The teardown fails a test that leaves a fault unconsumed (in a pytest child)
# --------------------------------------------------------------------------- #

_CHILD_TEST = '''
from mitos.store import GraphStore
from _point_hits import CannedVector


def test_canned_slug_names_no_node(tmp_path):
    GraphStore(str(tmp_path / "graph.sqlite"))
    # The reader would see an empty window and pass; the teardown must not.
    assert CannedVector([{"slug": "nobody", "score": 0.9}]).query([0.1]) == []


def test_clean(tmp_path):
    GraphStore(str(tmp_path / "graph.sqlite"))
    assert CannedVector([]).query([0.1]) == []
'''


def test_an_unconsumed_fixture_fault_fails_the_test_at_teardown(tmp_path) -> None:
    child = tmp_path / "test_child_fault.py"
    child.write_text(_CHILD_TEST, encoding="utf-8")
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": os.pathsep.join([str(TESTS_DIR), env.get("PYTHONPATH", "")]),
        "MITOS_NO_LIVE_TESTS": "1",
        # The child loads conftest, whose session sweep deletes `mitos-tmp*`
        # collections; point it at nothing so a concurrent live run is untouched.
        "QDRANT_URL": "http://localhost:9",
    })
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "conftest", "-p", "no:cacheprovider",
         "-p", "no:xdist", "--rootdir", str(tmp_path), str(child)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "unconsumed canned-hit fixture fault" in out, out
    assert "ERROR at teardown of test_canned_slug_names_no_node" in out, out
    assert "'nobody' names no node" in out, out
    assert "2 passed, 1 error" in out, out


# --------------------------------------------------------------------------- #
# Phase 2c: every `VectorStore.query` call site in `mitos/` goes through the walk
# --------------------------------------------------------------------------- #

#: The direct `.query(` call sites allowed in `mitos/`: the walk's one call, alone.
#: Since 2d moved the conflict gather onto the walk, every reader resolves by identity.
ALLOWED_QUERY_CALL_SITES = {
    ("retrieval_walk.py", "walk_ranked"),
}

#: The four ranked surfaces, each of which must call the walk.
RANKED_SURFACES = {
    ("mcp_server.py", "surface_decisions"),
    ("mcp_server.py", "query_decisions"),
    ("cli.py", "cmd_surface"),
    ("cli.py", "cmd_query"),
}

#: Every walk caller: the four ranked surfaces plus the conflict gather (2d), which
#: is not a ranked read and so stays out of `RANKED_SURFACES`.
WALK_CALLERS = RANKED_SURFACES | {("conflict.py", "gather_candidates")}


def _enclosing_calls(source: str) -> List[Tuple[str, ast.Call]]:
    """Every call in ``source``, paired with the name of its outermost function.

    Total over the module: a call outside any function is paired with ``<module>``
    (or the class name, for a class body), so a module-level ``vs.query(...)`` is a
    set difference too rather than a call the row never looked at.
    """
    def calls(tree: ast.AST) -> List[ast.Call]:
        return [c for c in ast.walk(tree) if isinstance(c, ast.Call)]

    out = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.extend((node.name, c) for c in calls(node))
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.extend((f"{node.name}.{item.name}", c) for c in calls(item))
                else:
                    out.extend((node.name, c) for c in calls(item))
        else:
            out.extend(("<module>", c) for c in calls(node))
    return out


def query_call_sites(files: List[Path]) -> set:
    """``(file, function)`` for every call whose callee is an attribute named ``query``."""
    return {(path.name, fn) for path in files
            for fn, call in _enclosing_calls(path.read_text(encoding="utf-8"))
            if isinstance(call.func, ast.Attribute) and call.func.attr == "query"}


def walk_callers(files: List[Path]) -> set:
    """``(file, function)`` for every call to ``walk_ranked``, bare or qualified."""
    return {(path.name, fn) for path in files
            for fn, call in _enclosing_calls(path.read_text(encoding="utf-8"))
            if (isinstance(call.func, ast.Name) and call.func.id == "walk_ranked")
            or (isinstance(call.func, ast.Attribute) and call.func.attr == "walk_ranked")}


def test_every_vector_query_in_mitos_goes_through_the_walk() -> None:
    """Vision §6.2 constraint 8, by AST. A later reader is covered by arriving: a new
    direct call is a set difference here. The second clause catches a surface that
    kept a private copy of the loop, which would pass the first by calling nothing."""
    files = sorted(MITOS_DIR.rglob("*.py"))
    assert query_call_sites(files) == ALLOWED_QUERY_CALL_SITES
    assert WALK_CALLERS <= walk_callers(files)


def test_the_call_site_row_reports_a_planted_direct_call(tmp_path) -> None:
    """Positive control over a planted module (break K12 plants the same in `mitos/`)."""
    planted = tmp_path / "_zz_scratch_2c.py"
    planted.write_text(textwrap.dedent('''
        def reader(vs, v):
            return vs.query(v, limit=5)

        HITS = VS.query(V, limit=5)

        class Reader:
            hits = VS.query(V, limit=5)
    '''), encoding="utf-8")
    assert query_call_sites([planted]) == {("_zz_scratch_2c.py", "reader"),
                                           ("_zz_scratch_2c.py", "<module>"),
                                           ("_zz_scratch_2c.py", "Reader")}
