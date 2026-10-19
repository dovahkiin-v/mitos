"""Phase 5e: the CLI prints a whole ``rejected_paths`` through one renderer.

Every CLI text line that prints a decision's whole field goes through
``rejections.render_rejected_paths``: the field's enumeration, one item per line, a
prose field verbatim as its one item, every later line hung at the label's width so
none lands at column 0. JSON, MCP and the rendered files keep the stored string. The
skill text teaches writing one rejected option per line. The rows here pin:

- the renderer's layout rules and its properties over every trace-table and Harbor
  field (coverage, no column 0, line count, byte identity where nothing changes);
- the source sweep: every printer call in ``mitos/`` whose arguments reference the
  field (directly, or through a name its own function bound from one) reaches it
  through the renderer, except the record pause's named preview, with a positive and
  a negative control on the selector itself. A value carried in by a parameter or
  accumulated into a list first is outside what the selector reads; the one such
  site, sync's divergence report, prints a byte-diagnostic ``repr`` and is pinned by
  its own row;
- each command's text on a numbered field and on a one-line prose field, and the
  shell ``grep`` over ``mitos show`` that returns one whole option;
- the surfaces that do not change (``--json``, MCP, ``live_axioms.md``);
- the skill sentence: outside the interpolated spec, and both of its claims true.
"""

import ast
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, List, NamedTuple, Set, Tuple
from unittest.mock import patch

import pytest

import mitos.cli as cli
import mitos.mcp_server as mcp_server
from mitos.cli import cmd_list, cmd_render, cmd_show
from mitos.config import MitosConfig
from mitos.parser import ParsedEntry, load_format_spec
from mitos.rejections import (enumerate_rejected_paths, point_items, render_rejected_item,
                              render_rejected_paths)
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager

import test_rejection_lookup as lookup
import test_rejection_section as section
from _conflict_helpers import _RecordingJudge, _execution, _match, _seed_active, _wire_fakes
from _rejection_index_double import FakeRejectionIndex
from test_degraded_envelope import (DRIVERS, block, commit_text, drive, dws,  # noqa: F401
                                    listed, offline, write_buffer)
from test_rejections import TRACE, _AGREEMENT_FIELDS

REPO = Path(__file__).resolve().parent.parent
ws = section.ws  # the ranked rows' workspace fixture

P = "   Rejected: "
NUMBERED = ("(1) Redis: REJECTED — a server to run.\n"
            "(2) Memcached: REJECTED — no persistence.")
PROSE = "Rejected the obvious alternative."
# One line, two items: the shape the renderer changes even under an empty prefix, so a
# payload that took the terminal layout differs from the stored string on it.
INLINE = "(1) Redis: REJECTED — a server to run. (2) Memcached: REJECTED — no persistence."
STORED = [NUMBERED, INLINE]


def _aligned(prefix: str) -> str:
    """NUMBERED as every site must print it: item 2 at the value column."""
    pad = " " * len(prefix)
    return (f"{prefix}(1) Redis: REJECTED — a server to run.\n"
            f"{pad}(2) Memcached: REJECTED — no persistence.\n")


def _whole_line(prefix: str, field: str, out: str) -> bool:
    return f"{prefix}{field}" in out.splitlines()


def _capture(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# 10.1 — the renderer
# --------------------------------------------------------------------------- #

LAYOUT = [
    pytest.param(NUMBERED, _aligned(P)[:-1], id="numbered-by-lines"),
    pytest.param(INLINE, _aligned(P)[:-1], id="numbered-inline"),
    pytest.param("  1. first\n  2. second",
                 f"{P}  1. first\n{' ' * len(P)}  2. second", id="indented-numbered"),
    pytest.param("  - first\n    cont\n  - second",
                 f"{P}  - first\n{' ' * len(P)}    cont\n{' ' * len(P)}  - second",
                 id="indented-bulleted"),
    pytest.param("Polling: REJECTED — wastes calls. Webhooks: REJECTED — no port.",
                 f"{P}Polling: REJECTED — wastes calls.\n{' ' * len(P)}Webhooks: REJECTED — no port.",
                 id="rejected-marker"),
    pytest.param("Preamble first.\n1. Polling: wastes calls\n   over two lines.\n2. Webhooks: no inbound port.",
                 f"{P}Preamble first.\n{' ' * len(P)}1. Polling: wastes calls\n"
                 f"{' ' * len(P)}   over two lines.\n{' ' * len(P)}2. Webhooks: no inbound port.",
                 id="preamble-and-continuation"),
    pytest.param("\n\n(1) a: x\n(2) b: y", f"{P}(1) a: x\n{' ' * len(P)}(2) b: y",
                 id="leading-newlines"),
    pytest.param("1. a first\n\n   its second paragraph\n2. b",
                 f"{P}1. a first\n\n{' ' * len(P)}   its second paragraph\n{' ' * len(P)}2. b",
                 id="multi-paragraph-item"),
    pytest.param("(1) a: x\r\n(2) b: y\r\n", f"{P}(1) a: x\n{' ' * len(P)}(2) b: y", id="crlf"),
    pytest.param("(1) a still a: x\n(2) b\x0bb: y",
                 f"{P}(1) a still a: x\n{' ' * len(P)}(2) b\x0bb: y",
                 id="u2028-and-vt-stay-inside-a-line"),
    pytest.param(PROSE, f"{P}{PROSE}", id="prose-one-line"),
    pytest.param("Prose that mentions\n2. a stray line-start number.",
                 f"{P}Prose that mentions\n{' ' * len(P)}2. a stray line-start number.",
                 id="prose-stray-number"),
    pytest.param("（1）键值存储：太重。\n（2）文件：不够。",
                 f"{P}（1）键值存储：太重。\n{' ' * len(P)}（2）文件：不够。", id="cjk"),
    pytest.param("1. Хранилище — тяжело.\n2. Файлы — мало.",
                 f"{P}1. Хранилище — тяжело.\n{' ' * len(P)}2. Файлы — мало.", id="cyrillic"),
    pytest.param("", P, id="empty"),
    pytest.param("   ", P, id="spaces-only"),
    pytest.param("  \n  ", P, id="whitespace-lines-only"),
]


@pytest.mark.parametrize("field,expected", LAYOUT)
def test_the_layout_rules(field: str, expected: str) -> None:
    """R1–R4 on every worked shape."""
    out = render_rejected_paths(field, P)
    assert out == expected
    _assert_layout(field, P, out)


@pytest.mark.parametrize("field", [NUMBERED, INLINE, PROSE, "  1. a\n  2. b", "", "  "])
def test_an_empty_prefix_lays_out_the_items_and_nothing_else(field: str) -> None:
    """No label, no indent: the items one per line, so ``""`` is a fixed point only
    for a field already laid out that way."""
    lines = [line for item in enumerate_rejected_paths(field).items
             for line in item.text.rstrip().split("\n")]
    assert render_rejected_paths(field, "") == "\n".join(line.rstrip() for line in lines)
    assert render_rejected_paths(INLINE, "") != INLINE


def _non_space(text: str) -> str:
    return "".join(c for c in text if not c.isspace())


def _assert_layout(field: str, prefix: str, out: str) -> None:
    """The renderer's properties on one field (prefix non-empty)."""
    assert out.startswith(prefix) and not out.endswith("\n")
    enum = enumerate_rejected_paths(field)
    if enum.shape == "empty":
        assert out == prefix
        return
    lines = out.split("\n")
    # Coverage: no content byte dropped, reordered or rewritten (M7/P6).
    assert _non_space(out[len(prefix):]) == _non_space(field)
    # No column 0, and an empty line is empty.
    pad = " " * len(prefix)
    for line in lines[1:]:
        assert line == "" or line.startswith(pad), (field, line)
    assert all(line == line.rstrip() for line in lines)
    assert lines[0] != prefix.rstrip()
    # Line count: N single-line items print as N lines.
    if all("\n" not in item.text.strip("\r\n") for item in enum.items):
        assert len(lines) == len(enum.items), (field, out)


def _harbor_fields() -> List[str]:
    """Every Harbor field the oracle enumerates, rebuilt by the coverage contract."""
    oracle = json.loads((REPO / "tests/golden/oracle.reference.json").read_text(encoding="utf-8"))
    return ["".join(item["text"] for item in node["rejected_items"]["items"])
            for node in oracle["nodes"].values() if node.get("rejected_items")]


POPULATION = _AGREEMENT_FIELDS + _harbor_fields()


def test_the_population_is_the_trace_table_and_harbor() -> None:
    assert len(TRACE) > 10 and len(_harbor_fields()) >= 20
    assert {enumerate_rejected_paths(f).shape for f in _harbor_fields()} >= {"numbered", "prose"}


@pytest.mark.parametrize("prefix", ["   Rejected: ", "Rejected:     ", "       Rejected: "])
def test_the_properties_hold_over_the_whole_population(prefix: str) -> None:
    for field in POPULATION:
        _assert_layout(field, prefix, render_rejected_paths(field, prefix))


def test_a_one_line_one_item_field_is_byte_identical_to_the_old_print() -> None:
    fields = [f for f in POPULATION
              if "\n" not in f and f == f.rstrip() and f.strip()
              and len(enumerate_rejected_paths(f).items) == 1]
    assert len(fields) >= 10
    for field in fields + [PROSE, "(1) Only one: REJECTED — alone."]:
        assert render_rejected_paths(field, P) == f"{P}{field}"


def test_the_nth_printed_item_is_the_item_a_match_names_for_ordinal_n() -> None:
    """The join key: a match's ``item`` (``text.rstrip()``) is line N of the render."""
    owner = "0" * 64
    for field in [NUMBERED, "1. Alpha — a.\n2. Bravo — b.\n3. Charlie — c.",
                  "(1) A: REJECTED — x. (2) B: REJECTED — y."]:
        lines = render_rejected_paths(field, P).split("\n")
        by_ordinal = {i.ordinal: i.text.rstrip() for i in point_items(owner, field).values()}
        for ordinal, item in by_ordinal.items():
            assert lines[ordinal - 1][len(P):] == item


def test_a_matched_item_hangs_by_the_same_rule() -> None:
    hit = "  owner  (active)  item 1: "
    out = render_rejected_item("Preamble.\n1. Polling: wastes calls", hit)
    assert out == f"{hit}Preamble.\n{' ' * len(hit)}1. Polling: wastes calls"
    for field in POPULATION:
        for item in enumerate_rejected_paths(field).items:
            text = item.text.rstrip()
            rendered = render_rejected_item(text, hit)
            assert _non_space(rendered[len(hit):]) == _non_space(text)
            assert all(not ln or ln.startswith(" " * len(hit)) for ln in rendered.split("\n")[1:])


# --------------------------------------------------------------------------- #
# 10.2 — the sweep: every whole-field print goes through the renderer
# --------------------------------------------------------------------------- #

FIELD = "rejected_paths"
RENDERER = "render_rejected_paths"
EXEMPTION_NAME = "_PAUSE_REJECTED_PREVIEW"
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


class Member(NamedTuple):
    """One printer call that reaches the field."""

    path: str
    qualname: str
    line: int
    exempt: bool
    raw: int  # field references outside a renderer call


def _is_printer(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "print"
    return (isinstance(func, ast.Attribute) and func.attr == "write"
            and isinstance(func.value, ast.Attribute) and func.value.attr in ("stdout", "stderr")
            and isinstance(func.value.value, ast.Name) and func.value.value.id == "sys")


def _is_renderer(call: ast.Call) -> bool:
    func = call.func
    return ((isinstance(func, ast.Name) and func.id == RENDERER)
            or (isinstance(func, ast.Attribute) and func.attr == RENDERER))


def _is_ref(node: ast.AST, bound: Set[str]) -> bool:
    """A field reference: ``x["rejected_paths"]``, ``x.rejected_paths``,
    ``x.get("rejected_paths", …)``, or a name this scope bound from one."""
    if isinstance(node, ast.Subscript):
        key = node.slice
        return isinstance(key, ast.Constant) and key.value == FIELD
    if isinstance(node, ast.Attribute):
        return node.attr == FIELD
    if isinstance(node, ast.Call):
        func = node.func
        return (isinstance(func, ast.Attribute) and func.attr == "get" and bool(node.args)
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == FIELD)
    return isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in bound


def _refs(node: ast.AST, bound: Set[str], inside: bool = False) -> Iterator[bool]:
    """Yields, per field reference under ``node``, whether a renderer call wraps it.
    Only expressions are read; constant text never is."""
    if _is_ref(node, bound):
        yield inside
        return
    if isinstance(node, ast.Call) and _is_renderer(node):
        inside = True
    for child in ast.iter_child_nodes(node):
        yield from _refs(child, bound, inside)


def _own_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Every node in a scope, stopping at (but yielding) nested defs and classes."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _SCOPES):
            stack.extend(ast.iter_child_nodes(node))


def whole_field_prints(source: str, path: str = "<snippet>") -> List[Member]:
    """The selector: every printer call in ``source`` with a field reference in its
    arguments, each marked exempt when it sits inside the pause's preview chain."""
    members: List[Member] = []

    def visit(scope: ast.AST, qualname: str) -> None:
        own = list(_own_nodes(scope))
        bound = {target.id
                 for node in own if isinstance(node, (ast.Assign, ast.AnnAssign))
                 if node.value is not None and any(True for _ in _refs(node.value, set()))
                 for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                 if isinstance(target, ast.Name)}
        exempt = {id(call)
                  for node in own if isinstance(node, ast.If)
                  if any(isinstance(n, ast.Name) and n.id == EXEMPTION_NAME
                         for n in ast.walk(node.test))
                  for call in ast.walk(node) if isinstance(call, ast.Call)}
        for node in own:
            if isinstance(node, ast.Call) and _is_printer(node):
                found = [inside for arg in [*node.args, *[k.value for k in node.keywords]]
                         for inside in _refs(arg, bound)]
                if found:
                    members.append(Member(path, qualname, node.lineno, id(node) in exempt,
                                          sum(1 for inside in found if not inside)))
            if isinstance(node, _SCOPES):
                visit(node, f"{qualname}.{node.name}" if qualname else node.name)

    visit(ast.parse(source), "")
    return members


def _sweep() -> List[Member]:
    members: List[Member] = []
    for path in sorted((REPO / "mitos").glob("*.py")):
        members.extend(whole_field_prints(path.read_text(encoding="utf-8"), path.name))
    return members


def test_every_whole_field_print_goes_through_the_renderer() -> None:
    members = _sweep()
    assert members, "the selector found no print site at all"
    assert {m.path for m in members} >= {"cli.py", "sync.py"}
    exempt = [m for m in members if m.exempt]
    assert exempt, f"the exemption names {EXEMPTION_NAME}, and no print sits under it"
    assert {m.qualname for m in exempt} == {"cmd_record"}
    violations = [m for m in members if not m.exempt and m.raw]
    assert not violations, (
        "These print a whole rejected_paths without rejections.render_rejected_paths: "
        f"{violations}")


_PLANT = '''
import sys
def cmd_scratch(config, d, e):
    print(f"   Rejected: {d['rejected_paths']}")
    r = e.rejected_paths
    print(r)
    sys.stderr.write(e.get("rejected_paths"))
    print(render_rejected_paths(d["rejected_paths"], "   Rejected: "))
    print(f"{render_rejected_paths(r, 'x')} and {d['rejected_paths']}")
    print("a constant naming rejected_paths is never a reference")
    if len(r) > _PAUSE_REJECTED_PREVIEW:
        print(r)
    elif r:
        print(f"rejected: {r}")

class Manager:
    def review(self, entry):
        print(f"  Rejected:    {entry.rejected_paths}")
'''


def test_the_selector_flags_planted_raw_prints() -> None:
    """Positive control: every raw shape (subscript, name-bound, ``.get``, a stderr
    write, a method) is flagged; a rendered one is a member with no raw reference."""
    got = {(m.qualname, m.line): (m.exempt, m.raw) for m in whole_field_prints(_PLANT)}
    assert got == {
        ("cmd_scratch", 4): (False, 1),
        ("cmd_scratch", 6): (False, 1),
        ("cmd_scratch", 7): (False, 1),
        ("cmd_scratch", 8): (False, 0),
        ("cmd_scratch", 9): (False, 1),
        ("cmd_scratch", 12): (True, 1),
        ("cmd_scratch", 14): (True, 1),
        ("Manager.review", 18): (False, 1),
    }


def test_the_importer_warning_naming_the_field_is_not_a_member() -> None:
    """Negative control: constant text naming the field is not a reference."""
    source = (REPO / "mitos/importer.py").read_text(encoding="utf-8")
    warning = [node.lineno for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Call) and _is_printer(node)
               and FIELD in ast.get_source_segment(source, node)]
    assert warning, "the importer's warning print moved; re-anchor this control"
    assert not {m.line for m in whole_field_prints(source, "importer.py")} & set(warning)


def _names(tree: ast.AST) -> Set[str]:
    return ({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            | {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names})


def test_the_file_writers_never_take_the_terminal_layout() -> None:
    """D4: the rendered files, the decisions.md writers and the judge prompt keep the
    stored string, so the renderer is absent from them."""
    renderers = {RENDERER, "render_rejected_item"}
    for name in ("renderer.py", "restore.py", "amend.py", "conflict.py"):
        tree = ast.parse((REPO / "mitos" / name).read_text(encoding="utf-8"))
        assert not _names(tree) & renderers, name
    sync_tree = ast.parse((REPO / "mitos/sync.py").read_text(encoding="utf-8"))
    (writer,) = [n for n in ast.walk(sync_tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "record_decision_entry"]
    assert not _names(writer) & renderers


# --------------------------------------------------------------------------- #
# 10.3 — each command's text
# --------------------------------------------------------------------------- #

def _owner_field(field: str) -> str:
    return field.replace("Redis", "Redis sessions")


def _lexical_world(config: MitosConfig, field: str) -> Dict[str, str]:
    """``queue-redis`` matches "redis" by its axiom; ``session-store``'s item does."""
    blocks = [block("queue-redis", "Use redis for the job queue.", rejected=field,
                    scope="caching"),
              block("session-store", "Sessions live in Postgres.", scope="other",
                    rejected=_owner_field(field))]
    return commit_text(config, write_buffer(config, blocks))


@pytest.mark.parametrize("verb", ["surface", "query"])
def test_the_lexical_envelope_prints_numbered_items_aligned(dws, verb) -> None:
    _lexical_world(dws, NUMBERED)
    out = drive(f"cli-{verb}-text", dws, "redis", cause="no-providers")
    assert _aligned("   Rejected: ") in out
    owner = render_rejected_paths(_owner_field(NUMBERED), "    Rejected: ")
    assert owner + "\n" in out and owner.count("\n") == 1
    assert listed(f"cli-{verb}-text", out) == ["queue-redis"]


@pytest.mark.parametrize("verb", ["surface", "query"])
def test_the_lexical_envelope_prints_a_prose_field_as_before(dws, verb) -> None:
    _lexical_world(dws, PROSE)
    out = drive(f"cli-{verb}-text", dws, "redis", cause="no-providers")
    assert _whole_line("   Rejected: ", PROSE, out)


def test_the_scope_dump_prints_numbered_items_aligned(dws) -> None:
    _lexical_world(dws, NUMBERED)
    out = drive("cli-surface-text", dws, "redis", cause="no-providers", scope="caching")
    assert _aligned("   Rejected: ") in out


@pytest.mark.parametrize("field", [NUMBERED, PROSE])
@pytest.mark.parametrize("verb", list(section.TEXT_DRIVERS))
def test_the_ranked_hit_and_its_section_print_through_the_renderer(ws, monkeypatch, verb,
                                                                   field) -> None:
    section._commit(ws, "a", field)
    owner = section._commit(ws, "owner", field)
    index = FakeRejectionIndex(present=True)
    section._plant(index, owner, field, {1: section.ABOVE, 2: section.JUST_ABOVE})
    section._bind(monkeypatch, index)
    out = section.TEXT_DRIVERS[verb](ws, [{"slug": "a", "score": .9}])
    assert "Rejected options matching this query (closest first):" in out
    if field == NUMBERED:
        assert _aligned("   Rejected: ") in out
        assert _aligned("    Rejected: ") in out
    else:
        assert _whole_line("   Rejected: ", PROSE, out)
        assert _whole_line("    Rejected: ", PROSE, out)


def test_the_healthy_lookup_prints_the_owner_aligned(ws) -> None:
    owner = section._commit(ws, "owner", NUMBERED)
    index = FakeRejectionIndex(present=True)
    lookup._plant(index, ws, owner, {2: .99})
    out = lookup._cli(ws, index, as_json=False)
    assert "item 2: (2) Memcached: REJECTED — no persistence." in out
    assert _aligned("    Rejected: ") in out


def test_the_lexical_lookup_prints_the_owner_aligned(dws) -> None:
    _lexical_world(dws, NUMBERED)
    out = lookup._cli(dws, FakeRejectionIndex(present=False), as_json=False, query="memcached")
    assert "item 2: (2) Memcached: REJECTED — no persistence." in out
    assert _aligned("    Rejected: ") in out


def test_a_multi_line_matched_item_hangs_under_the_hit_line(dws) -> None:
    field = "Prose about redis that mentions\n2. a stray line-start number."
    _lexical_world(dws, field)
    out = drive("cli-query-text", dws, "stray", cause="no-providers")
    assert not re.search(r"^2\. a stray", out, flags=re.MULTILINE), out
    assert listed("cli-query-text", out) == []


@pytest.mark.parametrize("field,expected", [(NUMBERED, _aligned("Rejected:     ")),
                                            (PROSE, f"Rejected:     {PROSE}\n"),
                                            ("", "Rejected:     \n")])
def test_show_prints_the_field_through_the_renderer(ws, field, expected) -> None:
    """An empty field keeps its unconditional line, the label alone (R4)."""
    section._commit(ws, "one", field)
    assert _stored(ws, GraphStore(ws.db_path).get_node_by_slug("one")["id"]) == field
    out = _capture(cmd_show, ws, "one")
    assert expected in out


def test_a_shell_grep_over_show_returns_the_whole_option(dws) -> None:
    """§4.6: one option's words find exactly that option's line, and the whole of it."""
    _lexical_world(dws, NUMBERED)
    out = _capture(cmd_show, dws, "queue-redis")
    grep = subprocess.run(["grep", "-F", "Memcached"], input=out, capture_output=True,
                          text=True, check=True)
    assert grep.stdout == "              (2) Memcached: REJECTED — no persistence.\n"


def test_list_text_prints_no_whole_field(dws) -> None:
    """§7: ``list`` is an index view; the field reaches only its ``--json``."""
    _lexical_world(dws, NUMBERED)
    out = _capture(cmd_list, dws)
    assert "queue-redis" in out and "Memcached" not in out and "Redis: REJECTED" not in out


@pytest.mark.parametrize("field", [NUMBERED, PROSE])
def test_the_check_finding_sides_print_through_the_renderer(field) -> None:
    node = {"slug": "x", "core_axiom": "Ax.", "scope": ["s"], "rejected_paths": field}
    side = _capture(cli._print_check_finding_side, node)
    entry = ParsedEntry("decision", "pending", 1, 5)
    entry.axiom, entry.scope, entry.rejected_paths = "P.", ["s"], field
    finding = SimpleNamespace(payload={"slug": "active", "score": .9, "axiom": "A.",
                                       "scope": [], "rejected_paths": field},
                              rationale="why", confidence=.9)
    staged = _capture(cli._print_staged_finding, entry, finding)
    if field == NUMBERED:
        assert _aligned("      Rejected: ") in side
        assert staged.count(_aligned("      Rejected: ")) == 2
    else:
        assert _whole_line("      Rejected: ", PROSE, side)
        assert staged.splitlines().count(f"      Rejected: {PROSE}") == 2


@pytest.fixture
def sync_env(monkeypatch) -> Iterator[Tuple[MitosConfig, MitosSyncManager]]:
    """``_conflict_helpers``' sync world: the key floor met, interactive, keyless judge."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    config.db_path = os.path.join(tmp, ".mitos", "graph.sqlite")
    config.decisions_file = os.path.join(tmp, "decisions.md")
    config.archive_dir = os.path.join(tmp, "decisions", "archive")
    os.makedirs(os.path.join(tmp, ".mitos"), exist_ok=True)
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write("# Decisions\n<!-- BEGIN ENTRIES — new decisions go directly below this "
                "line, newest first -->\n")
    yield config, MitosSyncManager(config)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.parametrize("field", [NUMBERED, PROSE])
def test_sync_review_and_sensor_blocks_print_through_the_renderer(sync_env, capsys,
                                                                  field) -> None:
    config, manager = sync_env
    _seed_active(manager, "endpoints-auth", "All API endpoints require authentication.",
                 rejected=field)
    _wire_fakes(manager, judge=_RecordingJudge(_execution(
        [("endpoints-auth", False, 0.9, "They contradict.")])),
        matches=[_match("endpoints-auth", 0.9)])
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(f"## 2026-06-01 — health-public — Health\n"
                f"**Decided:** The /health endpoint is public.\n"
                f"**Rejected:** {field}\n**Mechanisms:** python\n**Scope:** api\n\n")
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)
    out = capsys.readouterr().out
    assert "[Conflict]" in out and "Proposed Decision: health-public" in out
    if field == NUMBERED:
        assert _aligned("  Rejected:    ") in out
        assert _aligned("       Rejected: ") in out
    else:
        assert _whole_line("  Rejected:    ", PROSE, out)
        assert _whole_line("       Rejected: ", PROSE, out)


# --------------------------------------------------------------------------- #
# 10.4 — the surfaces that keep the stored string
# --------------------------------------------------------------------------- #

def _stored(config: MitosConfig, node_id: str) -> str:
    return GraphStore(config.db_path, read_only=True).get_node(node_id)["rejected_paths"]


@pytest.mark.parametrize("field", STORED)
@pytest.mark.parametrize("driver", [d for d, spec in DRIVERS.items() if spec[3] == "json"])
def test_json_and_mcp_carry_the_stored_string(dws, driver, field) -> None:
    ids = _lexical_world(dws, field)
    stored = {slug: _stored(dws, node_id) for slug, node_id in ids.items()}
    assert stored["queue-redis"] == field
    resp = drive(driver, dws, "redis", cause="no-providers")
    for d in resp["matches"]:
        assert d["rejected_paths"] == stored[d["slug"]]
    here = [m for m in resp["rejection_matches"] if m["letter"] == "here"]
    assert here and all(m["rejected_paths"] == stored[m["slug"]] for m in here)


@pytest.mark.parametrize("field", STORED)
def test_show_json_and_show_node_carry_the_stored_string(dws, field) -> None:
    _lexical_world(dws, field)
    cli_json = json.loads(_capture(cmd_show, dws, "queue-redis", as_json=True))
    mcp_json = json.loads(mcp_server.show_node("queue-redis", project=dws.workspace_dir))
    assert cli_json["rejected_paths"] == mcp_json["rejected_paths"] == field


@pytest.mark.parametrize("field", STORED)
@pytest.mark.parametrize("driver", list(section.DRIVERS))
def test_the_ranked_json_carries_the_stored_string(ws, monkeypatch, driver, field) -> None:
    section._commit(ws, "a", field)
    owner = section._commit(ws, "owner", field)
    index = FakeRejectionIndex(present=True)
    section._plant(index, owner, field, {1: section.ABOVE})
    section._bind(monkeypatch, index)
    env = section._call(driver, ws, [{"slug": "a", "score": .9}])
    hits = env[section.DRIVERS[driver][1]]
    assert hits[0]["rejected_paths"] == field
    assert env["rejection_matches"][0]["rejected_paths"] == field


@pytest.mark.parametrize("field", STORED)
def test_the_rendered_files_keep_their_own_layout(dws, field) -> None:
    """``renderer.py``'s own rule, byte for byte; on the inline field the terminal
    layout would differ under any prefix, so a leak reds here, not only in D4's pin."""
    _lexical_world(dws, field)
    _capture(cmd_render, dws)
    kept = "- **Rejected:**\n  " + "\n  ".join(field.splitlines())
    root = Path(dws.workspace_dir)
    for path in (root / "live_axioms.md", root / ".mitos/axioms/caching.md"):
        text = path.read_text(encoding="utf-8")
        assert kept in text, path
        if field == INLINE:
            for prefix in ("- **Rejected:** ", "  ", ""):
                assert render_rejected_paths(field, prefix) not in text


def test_the_divergence_report_prints_a_whole_field_as_one_repr_line() -> None:
    """Outside the sweep's reach (the field arrives as a parameter), and not a member
    by design: ``--full`` shows both raw values byte for byte, whitespace included,
    which the terminal layout would strip. ``repr`` keeps each on one line."""
    from mitos.sync import _divergence_field_lines
    lines = _divergence_field_lines("rejected_paths", NUMBERED, NUMBERED + " \n", full=True)
    assert lines == ["  rejected_paths:", f"    graph:    {NUMBERED!r}",
                     f"    markdown: {NUMBERED + ' ' + chr(10)!r}"]
    assert all("\n" not in line for line in lines)


# --------------------------------------------------------------------------- #
# 10.5 — the skill sentence
# --------------------------------------------------------------------------- #

_FORM = "(N) <option>: REJECTED — <reason>"


def _record_bullet(body: str) -> str:
    return next(line for line in body.splitlines() if line.startswith("- `record_decision`"))


def test_the_skill_teaches_the_form_outside_the_interpolated_spec() -> None:
    body = cli._skill_md_text("<<FORMAT SPEC SENTINEL>>")
    line = _record_bullet(body)
    assert body.index("<<FORMAT SPEC SENTINEL>>") < body.index(line)
    assert "one rejected option per line" in line and f"`{_FORM}`" in line
    assert _FORM in mcp_server.record_decision.__doc__  # spelled as the description spells it
    assert "one rejected option per line" not in load_format_spec()
    assert "REJECTED —" not in load_format_spec()


def test_the_sentences_two_claims_are_true_through_the_enumeration() -> None:
    """The taught form, two options a line each, is two items; the spec's own sample,
    which the same sentence says parses as one option, is one item."""
    sentence = _record_bullet(cli._skill_md_text(""))
    form = re.search(r"`(\(N\) <option>: REJECTED — <reason>)`", sentence).group(1)
    field = "\n".join(form.replace("(N)", f"({n})").replace("<option>", option)
                      .replace("<reason>", reason)
                      for n, option, reason in [(1, "Redis", "a server to run."),
                                                (2, "Memcached", "no persistence.")])
    taught = enumerate_rejected_paths(field)
    assert taught.shape == "numbered" and len(taught.items) == 2
    assert "as one option" in sentence
    (sample,) = [line.split("**Rejected:**", 1)[1].strip()
                 for line in load_format_spec().splitlines()
                 if line.startswith("**Rejected:**")]
    assert "," in sample
    assert len(enumerate_rejected_paths(sample).items) == 1
