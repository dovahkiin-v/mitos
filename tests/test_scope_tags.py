"""Tests for the shared scope-tag rule (`mitos/scope_tags.py`).

Three sites decide what "the same scopes" means — the parser, the store's writer and
the divergence comparator. If the comparator and the writer disagree, sync reconciles
a divergence whose commit writes nothing and reports it again on every run. So the
rule has one home, and this module pins both the rule and the fact that all three
sites call it.
"""

import ast
import inspect
import subprocess
import sys
import textwrap

import pytest

from mitos import divergence, parser, scope_tags, store
from mitos.scope_tags import normalize_scope_tags


@pytest.mark.parametrize("raw, expected", [
    (["alpha", "beta"], ["alpha", "beta"]),
    (["beta", "alpha"], ["beta", "alpha"]),          # authored order is kept
    ([" Alpha ", "BETA"], ["alpha", "beta"]),        # strip + casefold
    (["alpha", "", "  ", "beta"], ["alpha", "beta"]),  # empties dropped (MI-9)
    (["beta", "alpha", "Beta"], ["beta", "alpha"]),  # first-seen dedup, after folding
    (["Straße", "STRASSE"], ["strasse"]),            # casefold, not lower()
    (["ŽEMĖLAPIS", "žemėlapis"], ["žemėlapis"]),     # Lithuanian folds too
    ([], []),
    (None, []),
])
def test_the_rule(raw, expected) -> None:
    """Strip, drop empties, `str.casefold`, first-seen dedup, authored order."""
    assert normalize_scope_tags(raw) == expected


def test_casefold_is_not_lower() -> None:
    """`ß` is the guard: `lower()` leaves it, `casefold()` folds it to `ss`."""
    assert normalize_scope_tags(["ß"]) == ["ss"]


def test_the_leaf_imports_nothing_from_mitos() -> None:
    """A stdlib leaf, proved by the import graph rather than read off the source.

    `divergence` imports it at module level, and its probe forbids `parser` and
    `identity`; a leaf that grew either import would leak them there.
    """
    probe = ("import sys; import mitos.scope_tags; "
             "print(','.join(sorted(m for m in sys.modules "
             "if m == 'mitos' or m.startswith('mitos.'))))")
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["mitos,mitos.scope_tags"]


def _iterates_over(tree: ast.AST, names: set) -> list:
    """Returns every comprehension/generator in `tree` whose iterable mentions `names`."""
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.comprehension):
            mentioned = {n.id for n in ast.walk(node.iter) if isinstance(n, ast.Name)} | {
                n.attr for n in ast.walk(node.iter) if isinstance(n, ast.Attribute)
            }
            if mentioned & names:
                hits.append(ast.unparse(node.iter))
    return hits


def _calls(tree: ast.AST, name: str) -> int:
    return sum(
        1 for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
    )


def test_parser_store_and_divergence_share_the_one_function() -> None:
    """The structural row: one function, three callers, no copy grown back.

    Identity holds for the two module namespaces that import the name. The parser
    keeps `_normalize_scope_list` by name (its test imports it), so its half is that
    the delegate calls the leaf. The no-copy half is scoped to comprehensions that
    iterate a scope list, because both methods legitimately casefold a slug.
    """
    assert divergence.normalize_scope_tags is scope_tags.normalize_scope_tags
    assert store.normalize_scope_tags is scope_tags.normalize_scope_tags
    assert parser.normalize_scope_tags is scope_tags.normalize_scope_tags

    delegate = ast.parse(inspect.getsource(parser._normalize_scope_list))
    assert _calls(delegate, "normalize_scope_tags") == 1
    assert _iterates_over(delegate, {"items"}) == []

    commit = ast.parse(textwrap.dedent(inspect.getsource(store.GraphStore.commit_parsed_entry)))
    assert _calls(commit, "normalize_scope_tags") == 1
    assert _iterates_over(commit, {"scope"}) == [], "store grew its own scope rule back"

    leaf = ast.parse(inspect.getsource(divergence.entry_divergence))
    assert _calls(leaf, "normalize_scope_tags") == 2
    assert _iterates_over(leaf, {"scope", "stored_scopes"}) == [], (
        "the comparator grew its own scope rule back"
    )


def test_the_parser_still_emits_the_same_scope_list() -> None:
    """Moving the body behind a delegate must not change what the parser emits."""
    text = ("### probe\n\n**Decided:** An axiom.\n**Rejected:** n/a\n"
            "**Mechanisms:** m1\n**Scope:** Beta, alpha, BETA, , Straße\n")
    [entry] = parser.parse_entry_stream(text, "decision")
    assert entry.scope == ["beta", "alpha", "strasse"]
