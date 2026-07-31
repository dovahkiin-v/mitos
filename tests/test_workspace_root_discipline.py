"""The workspace-root discipline: no parameter in `mitos/` defaults to the cwd.

I1's structural second leg. Phases 5a and 5b removed the working-directory fallback
from every workspace-targeting verb on both surfaces — but at the *call sites*. The
two constructors those call sites feed still declared `workspace_dir: str = "."`, so
the fallback survived as a **shape**: any future call site could re-summon it by
omitting an argument, and nothing red would say so. 5d removed both defaults, which
is what turns the rule from one the tests police into one the constructor cannot
express a violation of — a zero-argument construction is a `TypeError` at the call,
not a silent `abspath(".")` three frames down.

Two halves, and neither is the claim on its own:

* **The three `TypeError` rows** pin the constructors that hold the hazard today —
  `MitosConfig`, `MitosRenderer`, and (on a related but distinct argument)
  `GeminiEmbeddingProvider.api_key`. They are behavioural and they are exact.
* **The AST sweep** pins the *property*: no parameter anywhere in `mitos/` defaults
  to the working directory. The property is what outlives the two classes — a
  name-keyed check would miss `root: str = "."` and `directory: str = "."`, which is
  precisely the class a future author would write.

The sweep keys on the default **value**, not the parameter name, and it asserts the
collection is **empty** rather than matching a declared set: after 5d the claim is
absolute, and an empty declared dict left behind is a hook for the next exemption
(`test_env_routing.py`'s own docstring says so, about the dict 5c deleted rather
than emptied). A legitimate future `"."` default therefore reds and forces a
conscious exemption instead of sliding in beside a comment.

It is an AST sweep and not a closing `grep` for one measured reason beyond the
standing-check argument: `cli.py`'s `cmd_status_overview` holds
`cwd: Optional[str] = os.getcwd()` — an annotated *local*, spelled
character-for-character like a defaulted parameter, and a read ledger entry-005
explicitly protects as a keeper. A text-keyed check reds on it; a sweep over
`args.defaults` / `args.kw_defaults` cannot see it at all, which is correct.

Every row here is offline: three constructor calls that never reach a filesystem,
and one sweep that parses source and opens no workspace.
"""

import ast
import glob
import os
from typing import List, Tuple

import pytest

from mitos import models
from mitos.config import MitosConfig
from mitos.embeddings import GeminiEmbeddingProvider
from mitos.renderer import MitosRenderer


# The three spellings of "this parameter silently means the process's working
# directory". The third is the neighbouring one the first two cannot see, and
# folding it in here is cheaper than a second net.
#
# Keyed on the default VALUE rather than the parameter NAME: `workspace_dir` is not
# the property. A name-keyed sweep goes green on `def __init__(self, root: str =
# ".")`, which is the exact shape this check exists to catch.
_CWD_DEFAULT_SHAPES = "a `\".\"` literal, `os.curdir`, or `os.getcwd` called or bare"

# A floor, not a count: the package holds well over this many modules, and a
# floor does not churn when one is added. Its only job is to fail loudly if the
# glob ever stops finding the package at all.
_MODULE_FLOOR = 15


def _package_modules() -> List[str]:
    """Every `.py` file in the installed `mitos/` package directory, sorted.

    Globbed off a module's `__file__` rather than enumerated, so a new module is
    swept the day it lands and nothing here shifts between runs.
    """
    return sorted(glob.glob(os.path.join(os.path.dirname(models.__file__), "*.py")))


def _parameter_defaults(path: str) -> List[Tuple[str, str, ast.expr]]:
    """Every defaulted parameter in a module, as `(function, parameter, default, kind)`.

    `kind` is `"positional"` or `"keyword-only"` — carried not for the offender
    report (a `"."` default is equally wrong either way) but so the non-vacuity
    control below can assert the sweep reached **both** default lists. A visitor
    that silently stopped reading one of them would otherwise pass every emptiness
    claim in this file.

    Both default lists are read, and the two are shaped differently:

    * `args.defaults` aligns to the **tail** of `posonlyargs + args` — a naive
      `zip(args.args, args.defaults)` pairs `MitosConfig`'s default with `self`,
      which still reds an emptiness claim but names the wrong parameter in the
      failure message.
    * `args.kw_defaults` is positional against `kwonlyargs` and carries a Python
      `None` **placeholder** for a keyword-only parameter with no default — which
      is a different thing from `ast.Constant(None)`, a real `= None` default.
      Filter the placeholders out; a sweep that reads only the first list is blind
      to `*, root: str = "."` entirely.

    Functions are reported by their dotted scope (`ClassName.method`), following
    `test_env_routing.py`'s sweep.
    """
    found: List[Tuple[str, str, ast.expr, str]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: List[str] = []

        def _scoped(self, node: ast.AST) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_ClassDef = _scoped

        def _function(self, node: ast.AST) -> None:
            self.stack.append(node.name)
            where = ".".join(self.stack)
            a = node.args
            positional = a.posonlyargs + a.args
            tail = positional[len(positional) - len(a.defaults):]
            for arg, default in zip(tail, a.defaults):
                found.append((where, arg.arg, default, "positional"))
            for arg, default in zip(a.kwonlyargs, a.kw_defaults):
                if default is not None:
                    found.append((where, arg.arg, default, "keyword-only"))
            self.generic_visit(node)
            self.stack.pop()

        visit_FunctionDef = _function
        visit_AsyncFunctionDef = _function

    with open(path, encoding="utf-8") as handle:
        Visitor().visit(ast.parse(handle.read()))
    return found


def _is_cwd_default(node: ast.expr) -> bool:
    """True for the spellings of a working-directory default.

    A bare `os.getcwd` *reference* is caught alongside `os.curdir` and not only the
    `os.getcwd()` call: as a default it is a nullary factory one `()` away from the
    call, and matching the attribute name costs nothing over matching one of them.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value == ".":
        return True
    if isinstance(node, ast.Attribute) and node.attr in ("curdir", "getcwd"):
        return True
    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "getcwd"):
        return True
    return False


# --- the three constructors that held the hazard ---------------------------- #

def test_a_zero_argument_config_is_a_type_error() -> None:
    """`MitosConfig()` no longer resolves the working directory — it refuses.

    This is the claim `test_config.py`'s four-forms row carried as a *construction*
    until 5d; inverted here rather than deleted, because it is the only place in the
    tree that names the form the constructor used to define (1d's two tripwires at
    4b and 2c's four fallback rows at 5c set that discipline).

    Only the exception **type** is asserted. CPython's "missing 1 required
    positional argument" wording is interpreter-version-dependent; the parameter
    name is matched as a substring because that much is ours, not the interpreter's.
    """
    with pytest.raises(TypeError) as exc:
        MitosConfig()  # type: ignore[call-arg]

    assert "workspace_dir" in str(exc.value)


def test_a_zero_argument_renderer_is_a_type_error(tmp_path) -> None:
    """`MitosRenderer()` refuses too — and this row is the only proof of that half.

    The renderer's default had **zero** consumers: all fifteen construction sites
    already passed an explicit root, so removing it cost no caller and a green suite
    proves nothing about it. That is the layer-removal trap in miniature — a change
    with no consumers has no natural net — and this row plus the sweep below are the
    whole net.

    The surviving explicit form is exercised alongside it, so a "fix" that made the
    parameter unusable rather than merely required could not pass.
    """
    with pytest.raises(TypeError) as exc:
        MitosRenderer()  # type: ignore[call-arg]

    assert "workspace_dir" in str(exc.value)
    assert MitosRenderer(str(tmp_path)).workspace_dir == os.path.abspath(str(tmp_path))


def test_a_provider_without_the_api_key_keyword_is_a_type_error(tmp_path) -> None:
    """`api_key` is keyword-**required**, and that is a different claim from its value.

    Honest scope first: this is not I1's property. 5c already made the provider read
    no environment at all, so a bare construction refuses *loudly* today rather than
    guessing. What 5d closes is the constructor-default hole one step earlier — a
    call site that forgot the keyword gets a refusal at the call instead of three
    frames down, on the same argument 1d used to remove
    `QdrantVectorStore(collection_name: str = "mitos")`.

    So the two failures must stay distinguishable, and the second row is the load-
    bearing one: a **supplied** `None` is still a supplied answer, and it still
    raises the worded `EmbeddingError` 5c redesigned. A change that turned the
    forgotten keyword into a silent `None` would pass the first assertion's
    neighbourhood and red here.
    """
    from mitos.errors import EmbeddingError

    cache = str(tmp_path / "cache.sqlite")

    with pytest.raises(TypeError) as exc:
        GeminiEmbeddingProvider(cache)  # type: ignore[call-arg]

    assert "api_key" in str(exc.value)

    with pytest.raises(EmbeddingError) as refusal:
        GeminiEmbeddingProvider(cache, api_key=None)

    assert "GEMINI_API_KEY" in str(refusal.value)
    assert "mitos set-key" in str(refusal.value)


# --- the property, swept ---------------------------------------------------- #

def test_no_parameter_in_the_package_defaults_to_the_working_directory() -> None:
    """The standing check, and the reason the rule outlives the two classes above.

    Asserts **empty**, not a declared set: after 5d there is no legitimate member,
    and an empty dict left behind is a hook for the next one. A future `"."` default
    reds here and forces a conscious exemption.

    The sweep must also prove it is not vacuous, and that is the half of this row
    that is not optional: a check whose only assertion is *"I found nothing"* goes
    green when it visits **zero files** — a wrong glob, a moved package, an
    `__init__.py`-only match — and a standing check that can pass by looking at
    nothing is worse than none, because it reads as coverage. So two controls ride
    in-row: a floor on the module count, and a **positive control** — the sweep
    finds ordinary parameter defaults from **both** default lists, which is the
    sharper form of the same idea. `args.kw_defaults` is a separate list from
    `args.defaults`, so a visitor that read only the first would be blind to
    `*, root: str = "."` entirely while passing every emptiness claim in this file;
    requiring a hit from each is what makes that regression impossible rather than
    merely commented against. Counting instead of naming a member keeps the control
    from churning when a signature moves.

    `mitos/cli.py`'s exempt-arm `MitosConfig(".")` is out of scope by construction
    and must stay: it is `main()`'s `init`/`serve`/`projects`/`set-key --global`
    branch, the four verbs where cwd genuinely *is* the target, and 5a wrote it
    deliberately as an explicit argument. The sweep keys on parameter *defaults*, so
    it never sees a call-site argument.
    """
    modules = _package_modules()
    assert len(modules) >= _MODULE_FLOOR, (
        f"the sweep found only {len(modules)} modules in "
        f"{os.path.dirname(models.__file__)} — the glob has stopped finding the package"
    )

    offenders: List[str] = []
    seen = {"positional": 0, "keyword-only": 0}
    for path in modules:
        module = os.path.basename(path)
        for where, parameter, default, kind in _parameter_defaults(path):
            seen[kind] += 1
            if _is_cwd_default(default):
                offenders.append(f"{module}:{where}:{parameter}")

    assert seen["positional"] > 0 and seen["keyword-only"] > 0, (
        f"the sweep visited {len(modules)} modules but collected {seen} — a list it "
        "reads as empty is a list it cannot police"
    )
    assert offenders == [], (
        f"parameters in mitos/ defaulting to the working directory "
        f"({_CWD_DEFAULT_SHAPES}): {offenders}"
    )
