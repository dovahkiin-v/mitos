"""MCP rows for `amend_commentary` (Phase 4c, W19, T13's MCP half).

The tool is a surface over `MitosSyncManager.amend_commentary` (4a), the CLI verb's
twin (4b). It owns three things and these rows pin each: the argument → `changes`
translation (clears are a named list, an empty value is a fault), the channel (every
result class returns in-band with provenance; only the fidelity fence raises), and this
boundary's recovery clauses (one per class, none naming a shell command).

In-process rows call `mcp_server.amend_commentary(...)` against real registered
`cmd_init` workspaces seeded by `record`, keyless, with the embed provider and vector
store down. The stdio rows drive a real `mitos serve` the way
`test_mcp_stdio_harness.py` does.
"""

import ast
import asyncio
import copy
import inspect
import json
import os
from typing import Any, Dict, List

import pytest
from unittest.mock import MagicMock, patch

from mitos import amend, cli, mcp_server
from mitos.cli import cmd_amend_commentary
from mitos.config import MitosConfig
from mitos.divergence import RELATIONSHIP_FIELDS, entry_divergence
from mitos.errors import EntryFailure, FailureItem, ValidationError
from mitos.restore import (BufferFidelityError, RestoreError, verify_amended_buffer,
                           verify_whole_buffer)
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager

from mcp_harness import mitos_server
from test_amend_commentary import _intents
from test_cli_amend_commentary import (_archived, _buffer, _canned, _diverged, _draft,
                                       _entry, _plain, _record, _sha, _write_buffer)
from test_corpus_provenance import _init_workspace, _write_registry
from test_description_budget import _flat
from test_mcp_selector import FORBIDDEN_SYNTAX
from test_mcp_stdio_harness import (_raw_stdio_exchange, _rotating_workspace, _run_mitos,
                                    _scaffold_env, _tool_json)

DEAD_QDRANT_URL = "http://127.0.0.1:9"
NAME = "amend-mcp-ws"
PROVENANCE = ("project", "collection", "workspace")
PHANTOM = "c\n### phantom"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Keyless and serviceless: the embed step defers, nothing reaches a network."""
    monkeypatch.setenv("QDRANT_URL", DEAD_QDRANT_URL)
    down = MagicMock(side_effect=Exception("backend down"))
    monkeypatch.setattr("mitos.sync.GeminiEmbeddingProvider", down)
    monkeypatch.setattr("mitos.sync.QdrantVectorStore", down)


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A registered workspace addressed as `NAME`, with the cwd outside it."""
    root = _init_workspace(tmp_path / "ws")
    _write_registry(**{NAME: root})
    monkeypatch.chdir(tmp_path)
    config = MitosConfig(root, project=NAME)
    return config, MitosSyncManager(config)


@pytest.fixture
def twins(tmp_path, monkeypatch):
    """Two registered workspaces for parity: amending mutates, so each surface gets one."""
    roots = {name: _init_workspace(tmp_path / name) for name in ("mcp-side", "cli-side")}
    _write_registry(**roots)
    monkeypatch.chdir(tmp_path)
    configs = {name: MitosConfig(root, project=name) for name, root in roots.items()}
    return {name: (config, MitosSyncManager(config)) for name, config in configs.items()}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _call(**kwargs: Any) -> Dict[str, Any]:
    return json.loads(mcp_server.amend_commentary(project=kwargs.pop("project", NAME), **kwargs))


def _tools() -> Dict[str, Any]:
    return {tool.name: tool for tool in asyncio.run(mcp_server.mcp.list_tools())}


def _spy():
    return patch.object(MitosSyncManager, "amend_commentary", autospec=True,
                        side_effect=lambda self, slug, changes: {
                            "status": amend.STATUS_UNCHANGED, "id": "i", "slug": slug})


def _recording_spy(monkeypatch) -> List[Dict[str, Any]]:
    """Wraps the real core, recording each call's `changes` and a copy of its result."""
    calls: List[Dict[str, Any]] = []
    real = MitosSyncManager.amend_commentary

    def spy(self, slug, changes):
        calls.append({"changes": copy.deepcopy(dict(changes))})
        result = real(self, slug, changes)
        calls[-1]["result"] = copy.deepcopy(result)
        return result

    monkeypatch.setattr(MitosSyncManager, "amend_commentary", spy)
    return calls


def _constants(prefix: str) -> set:
    return {value for name, value in vars(amend).items() if name.startswith(prefix)}


def _cited(config, m):
    _record(m, "target")
    _record(m, "citer", cites="target")


def _normalized(payload: Dict[str, Any], config: MitosConfig) -> Dict[str, Any]:
    """Drops the boundary keys and makes `path` workspace-relative (gotcha 7)."""
    result = {k: v for k, v in payload.items() if k not in PROVENANCE and k != "recovery"}
    if "path" in result:
        result["path"] = os.path.relpath(result["path"], config.workspace_dir)
    return result


# --------------------------------------------------------------------------- #
# M1 — the argument map, and argument faults before resolution
# --------------------------------------------------------------------------- #

_CHANGE_KEYS = set(amend.EDITABLE_FIELDS) | {"axiom", "mechanisms"} | set(RELATIONSHIP_FIELDS)

_MAP_CASES = [
    ({"rejected_paths": "R."}, {"rejected_paths": "R."}),
    ({"invalidates_if": "I."}, {"invalidates_if": "I."}),
    ({"context": "C."}, {"context": "C."}),
    ({"scope": ["a", "b"]}, {"scope": ["a", "b"]}),
    ({"new_slug": "s"}, {"slug": "s"}),
    ({"clear": ["context"]}, {"context": None}),
    ({"clear": ["invalidates_if"]}, {"invalidates_if": None}),
    ({"clear": ["scope"]}, {"scope": []}),
    ({"clear": ["rejected_paths"]}, {"rejected_paths": None}),
    ({"axiom": "A."}, {"axiom": "A."}),
    ({"mechanisms": ["m"]}, {"mechanisms": ["m"]}),
    ({}, {}),
    ({"context": "C.", "scope": ["a"], "clear": ["invalidates_if"]},
     {"context": "C.", "scope": ["a"], "invalidates_if": None}),
] + [({field: "x"}, {field: "x"}) for field in RELATIONSHIP_FIELDS]


@pytest.mark.parametrize("kwargs, expected", _MAP_CASES,
                         ids=[",".join(sorted(k)) or "none" for k, _ in _MAP_CASES])
def test_each_argument_maps_to_exactly_its_change(ws, kwargs, expected) -> None:
    """M1 — an omitted argument contributes no key; a cleared name maps to its removing value."""
    with _spy() as spy:
        payload = _call(slug="h", **kwargs)
    assert payload["status"] == amend.STATUS_UNCHANGED
    _self, handle, changes = spy.call_args.args
    assert (handle, changes) == ("h", expected)
    assert set(changes) <= _CHANGE_KEYS


@pytest.mark.parametrize("kwargs, code, names", [
    ({"context": ""}, "empty_value", 'clear=["context"]'),
    ({"invalidates_if": "  "}, "empty_value", 'clear=["invalidates_if"]'),
    ({"rejected_paths": " \n"}, "empty_value", "required field"),
    ({"scope": []}, "empty_value", 'clear=["scope"]'),
    ({"scope": ["a", " "]}, "empty_value", 'clear=["scope"]'),
    ({"context": "x", "clear": ["context"]}, "conflicting_arguments", "'context'"),
    ({"scope": ["a"], "clear": ["scope"]}, "conflicting_arguments", "'scope'"),
    ({"new_slug": "s", "clear": ["slug"]}, "conflicting_arguments", "'slug'"),
])
def test_an_argument_fault_returns_before_any_project_is_resolved(kwargs, code, names) -> None:
    """M1 — returned, never raised; no provenance; neither resolution nor the core runs.

    `project` names nothing registered, so a tool that resolved first would raise the
    targeting anatomy instead of answering the argument.
    """
    with _spy() as spy, patch.object(mcp_server, "_target_config",
                                     side_effect=AssertionError("resolved")) as target:
        payload = json.loads(mcp_server.amend_commentary("h", project="nosuchproject", **kwargs))
    assert set(payload) == {"error", "code", "slug"}
    assert payload["code"] == code and payload["slug"] == "h"
    assert names in payload["error"]
    spy.assert_not_called()
    target.assert_not_called()


def test_the_shared_code_names_equal_the_clis_and_cli_is_never_imported() -> None:
    """D-4c-1 — the names are shared by value; `mcp_server` must not import `cli`."""
    assert mcp_server._AMEND_CODE_EMPTY_VALUE == cli.AMEND_CODE_EMPTY_VALUE
    assert mcp_server._AMEND_CODE_BUFFER_FIDELITY == cli.AMEND_CODE_BUFFER_FIDELITY
    tree = ast.parse(inspect.getsource(mcp_server))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert "mitos.cli" not in imported


# --------------------------------------------------------------------------- #
# M2 / M3 — every class over the real core: in-band, provenance last, parity with CLI
# --------------------------------------------------------------------------- #

# scenario → (setup, tool kwargs, the class it lands in, whether a clause rides it)
_SCENARIOS = {
    "not_found": (_plain, {"slug": "no-such-handle", "context": "A repair."},
                  amend.STATUS_NOT_FOUND, True),
    "archived": (_archived, {"slug": "old-one", "context": "A repair."},
                 amend.STATUS_ARCHIVED, True),
    "uncommitted": (_draft, {"slug": "draft-one", "context": "A repair."},
                    amend.STATUS_UNCOMMITTED, True),
    "axiom": (_plain, {"slug": "target", "axiom": "A different axiom."},
              amend.REASON_CANONICAL_CORE, True),
    "mechanisms": (_plain, {"slug": "target", "mechanisms": ["m"]},
                   amend.REASON_CANONICAL_CORE, True),
    "cites": (_plain, {"slug": "target", "cites": "other"}, amend.REASON_EDGES, True),
    "diverged": (_diverged, {"slug": "target", "context": "A repair."},
                 amend.REASON_DIVERGED, True),
    "invalid_value": (_plain, {"slug": "target", "clear": ["rejected_paths"]},
                      amend.REASON_INVALID_VALUE, True),
    "unchanged": (_plain, {"slug": "target", "context": "The target context."},
                  amend.STATUS_UNCHANGED, False),
    "amended": (_plain, {"slug": "target", "context": "A repaired context."},
                amend.STATUS_AMENDED, False),
    "renamed": (_cited, {"slug": "target", "new_slug": "target-renamed"},
                amend.STATUS_AMENDED, True),
    "slug_collision": (_plain, {"slug": "target", "new_slug": "other"},
                       "slug_collision", True),
}
_MUTATING = {"amended", "renamed"}


def _class_of(result: Dict[str, Any]) -> str:
    return result.get("code") or result.get("reason") or result["status"]


@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
def test_every_class_returns_in_band_with_provenance_last(ws, monkeypatch, capsys,
                                                          scenario) -> None:
    """M2 — parseable JSON, the core dict verbatim, `recovery` exactly where the table says."""
    config, m = ws
    setup, kwargs, klass, has_recovery = _SCENARIOS[scenario]
    setup(config, m)
    calls = _recording_spy(monkeypatch)
    sha = _sha(config)
    capsys.readouterr()

    payload = _call(**kwargs)

    assert capsys.readouterr().out == "", "stdout is the JSON-RPC channel"
    core = calls[0]["result"]
    assert _class_of(core) == klass
    assert tuple(payload)[-3:] == PROVENANCE
    assert {k: payload[k] for k in PROVENANCE} == {
        "project": NAME, "collection": config.qdrant_collection,
        "workspace": config.workspace_dir}
    assert not ({"recovery", *PROVENANCE} & set(core)), "a core key collides"
    assert {k: v for k, v in payload.items() if k not in PROVENANCE and k != "recovery"} == core
    assert ("recovery" in payload) is has_recovery
    if has_recovery:
        assert payload["recovery"] == mcp_server._amend_recovery(core)
    if scenario not in _MUTATING:
        assert _sha(config) == sha


@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
def test_each_class_equals_the_cli_json_object(twins, monkeypatch, capsys, scenario) -> None:
    """M3 — T13's MCP half: the MCP dict less `recovery` is the CLI `--json` object.

    Twin workspaces seeded identically, one per surface; both receive the `changes`
    the tool built, so the comparison is of the two boundaries over one core.
    """
    setup, kwargs, _klass, _has = _SCENARIOS[scenario]
    (mcp_config, mcp_m), (cli_config, cli_m) = twins["mcp-side"], twins["cli-side"]
    setup(mcp_config, mcp_m)
    setup(cli_config, cli_m)
    calls = _recording_spy(monkeypatch)

    payload = _call(project="mcp-side", **kwargs)
    capsys.readouterr()
    cmd_amend_commentary(cli_config, kwargs["slug"], calls[0]["changes"], as_json=True)
    cli_payload = json.loads(capsys.readouterr().out)

    assert _normalized(payload, mcp_config) == _normalized(cli_payload, cli_config)
    assert payload["project"] == "mcp-side" and cli_payload["project"] == "cli-side"
    if scenario == "not_found":
        assert mcp_server.SHOW_NOT_FOUND_HINT not in payload["recovery"]
        assert "synced" not in payload["recovery"]


def test_the_fidelity_fact_is_the_clis_on_both_surfaces(twins, capsys) -> None:
    """M3 + M7 — the raised body's first line is the CLI `--json` error, and it is text."""
    (mcp_config, mcp_m), (cli_config, cli_m) = twins["mcp-side"], twins["cli-side"]
    _plain(mcp_config, mcp_m)
    _plain(cli_config, cli_m)
    with pytest.raises(mcp_server._RenderedToolError) as excinfo:
        mcp_server.amend_commentary("target", context=PHANTOM, project="mcp-side")
    capsys.readouterr()
    assert cmd_amend_commentary(cli_config, "target", {"context": PHANTOM}, as_json=True) == 2
    cli_payload = json.loads(capsys.readouterr().out)

    body = str(excinfo.value)
    assert body.splitlines()[0] == f"[buffer_fidelity] {cli_payload['error']}"
    assert "object at 0x" not in body and "'phantom'" in body


# --------------------------------------------------------------------------- #
# M4 / M5 — the recovery tables: derived coverage, and no shell command anywhere
# --------------------------------------------------------------------------- #

_NONE_BY_DESIGN = {amend.STATUS_UNCHANGED, "commit_failed", "audit_unavailable"}


def test_the_recovery_tables_cover_the_reflected_vocabulary() -> None:
    """M4 — a vocabulary member added to `mitos.amend` reds here, never prints nothing."""
    statuses, reasons, codes = _constants("STATUS_"), _constants("REASON_"), set(amend.ERROR_FACTS)
    assert statuses and codes
    assert {amend.REASON_DIVERGED, amend.REASON_UNPARSEABLE} <= reasons
    assert set(mcp_server._AMEND_RECOVERY_BY_STATUS) == statuses
    assert set(mcp_server._AMEND_RECOVERY_BY_REASON) == reasons
    assert set(mcp_server._AMEND_RECOVERY_BY_CODE) == codes
    for table in (mcp_server._AMEND_RECOVERY_BY_STATUS, mcp_server._AMEND_RECOVERY_BY_CODE):
        for key, clause in table.items():
            assert (clause is None) is (key in _NONE_BY_DESIGN), key
    for reason in reasons:
        clause = mcp_server._amend_recovery(_canned(status=amend.STATUS_REFUSED, reason=reason))
        assert clause, reason


def _every_clause() -> Dict[str, str]:
    amended = _canned(status=amend.STATUS_AMENDED)
    results = {f"status:{s}": _canned(status=s) for s in _constants("STATUS_")
               if s != amend.STATUS_REFUSED}
    results["rename:cited"] = {**amended, "rename": {
        "from": "t", "to": "u", "incoming": [{"kind": "cites", "source": "c"}]}}
    results["rename:unread"] = {**amended, "rename": {"from": "t", "to": "u", "incoming": None}}
    results.update({f"reason:{r}": _canned(status=amend.STATUS_REFUSED, reason=r)
                    for r in _constants("REASON_")})
    results.update({f"code:{c}": _canned(code=c) for c in amend.ERROR_FACTS})
    clauses = {label: mcp_server._amend_recovery(result) for label, result in results.items()}
    clauses["fidelity"] = mcp_server._AMEND_FIDELITY_RECOVERY
    return {label: clause for label, clause in clauses.items() if clause is not None}


_CLAUSES = _every_clause()


@pytest.mark.parametrize("label", sorted(_CLAUSES))
def test_no_clause_names_a_shell_command(label) -> None:
    """M5 — scoped per clause string: the provenance `collection` starts `mitos-`."""
    clause = _CLAUSES[label]
    assert "mitos " not in clause
    for syntax in FORBIDDEN_SYNTAX:
        assert syntax not in clause


def test_the_canonical_core_clause_names_every_route_by_its_distinction() -> None:
    """M5 — `record_decision` and all three relations, each an argument that tool takes."""
    clause = _CLAUSES[f"reason:{amend.REASON_CANONICAL_CORE}"]
    record_args = _tools()["record_decision"].inputSchema["properties"]
    assert "`record_decision`" in clause and "this same project" in clause
    for intent, relation in amend.ROUTES.items():
        assert f"`{relation}='t'`" in clause
        assert mcp_server._AMEND_ROUTE_INTENTS[intent] in clause
        assert relation in record_args


def test_the_miss_and_rename_clauses_name_what_their_table_rows_fix() -> None:
    """M5 — archived names a person and no rebuild command; rename and draft name no tool."""
    tool_names = set(_tools())
    archived = _CLAUSES[f"status:{amend.STATUS_ARCHIVED}"]
    assert "a person" in archived and "decisions/archive/" in archived
    assert "mitos rebuild" not in archived
    for label in ("rename:cited", "rename:unread", f"status:{amend.STATUS_UNCOMMITTED}"):
        assert not any(name in _CLAUSES[label] for name in tool_names), label
    assert "'u'" in _CLAUSES["rename:cited"]
    not_found = _CLAUSES[f"status:{amend.STATUS_NOT_FOUND}"]
    assert "`list_decisions`" in not_found and "`show_node`" in not_found
    assert mcp_server.SHOW_NOT_FOUND_HINT not in not_found and "synced" not in not_found
    for label in ("reason:edges", "reason:diverged"):
        assert "a person" in _CLAUSES[label].lower()
    assert "`new_slug`" in _CLAUSES["code:slug_collision"]


def test_a_refused_clear_says_what_cannot_be_cleared_and_verbs_agree(ws) -> None:
    """Fresh-eyes 4c — a clear gives no value, so the clause may not blame one; lists agree."""
    config, m = ws
    _plain(config, m)
    for kwargs in ({"clear": ["rejected_paths"]}, {"clear": ["slug"]}):
        payload = _call(slug="target", **kwargs)
        assert payload["reason"] == amend.REASON_INVALID_VALUE
        assert "never cleared" in payload["recovery"]
    plural = _call(slug="target", clear=["nope", "nada"])
    assert plural["reason"] == amend.REASON_UNKNOWN_FIELD
    assert "'nada', 'nope' are not fields" in plural["recovery"]
    single = _call(slug="target", clear=["nope"])
    assert "'nope' is not a field" in single["recovery"]
    both = json.loads(mcp_server.amend_commentary(
        "target", context="x", scope=["a"], clear=["context", "scope"], project=NAME))
    assert "'context', 'scope' were both given" in both["error"]


def test_a_markup_refusal_carries_its_spans_and_a_clause_naming_no_command(ws) -> None:
    """R8 — in-band, with provenance; a new slug is reported as `slug` (G6)."""
    config, m = ws
    _plain(config, m)
    sha = _sha(config)
    payload = _call(slug="target", context="x </context>", new_slug="t</invoke>")
    assert (payload["status"], payload["reason"], payload["fields"]) == (
        "refused", amend.REASON_TOOL_CALL_MARKUP, ["context", "slug"])
    assert payload["markup_spans"] == [
        {"field": "context", "span": "</context>", "offset": 2},
        {"field": "slug", "span": "</invoke>", "offset": 1}]
    clause = payload["recovery"]
    assert "'context', 'slug' hold tool-call markup ('</context>' at character 2 of " \
           "'context', plus 1 more)" in clause
    assert "backticks" in clause and "mitos " not in clause
    assert all(payload[key] for key in PROVENANCE)
    assert _sha(config) == sha


def test_the_description_is_terse_true_and_names_no_command() -> None:
    """M5 + D-4c-6 — no shell syntax, no count, no future capability; the channel is stated."""
    desc = _flat(_tools()["amend_commentary"].description)
    assert "mitos " not in desc and "madr" not in desc.lower()
    for syntax in FORBIDDEN_SYNTAX:
        assert syntax not in desc
    for word in ("seven", "eight", "nine", "five", "four"):
        assert f" {word} " not in f" {desc.lower()} "
    for phrase in ("clear:", "archived", "RETURNS", "rename.incoming", "only to be refused",
                   "decisions.md", "unresolvable `project`"):
        assert phrase in desc, phrase
    # The front carries what a truncating client must not cut (test_description_budget).
    from test_description_budget import FRONT_WINDOW
    for phrase in ("still in decisions.md", "RETURNS", "name it in `clear`",
                   "refused with the route", "After a rename"):
        assert 0 <= desc.find(phrase) < FRONT_WINDOW, phrase


# --------------------------------------------------------------------------- #
# M6 — the fidelity raise, and the catch's width
# --------------------------------------------------------------------------- #

def test_a_value_that_would_corrupt_the_buffer_raises_and_writes_nothing(ws) -> None:
    """M6 — the only raise: `[buffer_fidelity]` body, buffer byte-identical, nothing attributed."""
    config, m = ws
    _plain(config, m)
    sha, intents = _sha(config), len(_intents(config))

    with pytest.raises(mcp_server._RenderedToolError) as excinfo:
        mcp_server.amend_commentary("target", context=PHANTOM, project=NAME)

    body = str(excinfo.value)
    assert body.startswith("[buffer_fidelity] ")
    assert mcp_server._AMEND_FIDELITY_RECOVERY in body
    assert isinstance(excinfo.value.__cause__, BufferFidelityError)
    assert _sha(config) == sha and len(_intents(config)) == intents


def test_an_invariant_breach_propagates_as_itself_not_as_a_fidelity_refusal(ws, monkeypatch) -> None:
    """M6 — catching `MitosError` would swallow MI-13's `ValidationError` into a code it is not."""
    config, m = ws
    _plain(config, m)
    monkeypatch.setattr(GraphStore, "resolve_handle",
                        MagicMock(side_effect=ValidationError("two active nodes share a slug")))
    with pytest.raises(ValidationError):
        mcp_server.amend_commentary("target", context="A repair.", project=NAME)


# --------------------------------------------------------------------------- #
# M7 — ledger entry-003: an entry failure renders as text
# --------------------------------------------------------------------------- #

def test_an_entry_failure_renders_its_entry_line_and_message() -> None:
    """M7 — slugged, pre-header and itemless envelopes; never an object repr."""
    item = FailureItem("C", "parser", "missing required field `**Decided:**`")
    slugged = str(EntryFailure("x", 42, 50, [item, item]))
    assert slugged == "entry 'x' (line 42): missing required field `**Decided:**` (and 1 more)"
    pre_header = str(EntryFailure(None, 3, 4, [item], raw_header="### bad header\n"))
    assert pre_header == ("the entry headed '### bad header' (line 3): missing required field "
                          "`**Decided:**`")
    assert str(EntryFailure("x", 1, 2)) == "entry 'x' (line 1) does not parse"


def test_the_parse_failure_branch_of_both_verifiers_reads_as_text(ws) -> None:
    """M7 — the amend fence and `verify_whole_buffer`'s RestoreError twin."""
    config, m = ws
    _plain(config, m)
    before = _buffer(config)
    assert before.count("The target context.") == 1
    after = before.replace("The target context.", PHANTOM)

    with pytest.raises(BufferFidelityError) as fidelity:
        verify_amended_buffer(before, after, target_id="unused", expected={})
    with pytest.raises(RestoreError) as splice:
        verify_whole_buffer(before, after, added=0)

    for message in (str(fidelity.value), str(splice.value)):
        assert "parse failure: entry 'phantom' (line " in message
        assert "object at 0x" not in message


# --------------------------------------------------------------------------- #
# M8 — the refusal-carrying arguments are derived-equal to record_decision's
# --------------------------------------------------------------------------- #

def test_the_refusal_arguments_equal_the_relation_set_and_records_types() -> None:
    """M8 — a tenth relation reaches both tools, or the row reds."""
    tools = _tools()
    amend_props = tools["amend_commentary"].inputSchema["properties"]
    record_props = tools["record_decision"].inputSchema["properties"]
    own = {"slug", "rejected_paths", "invalidates_if", "context", "scope", "new_slug",
           "clear", "axiom", "mechanisms", "project"}
    record_own = {"axiom", "rejected_paths", "scope", "slug", "mechanisms", "context",
                  "acknowledge_neighbors", "draft_digest", "project"}
    assert set(amend_props) - own == set(RELATIONSHIP_FIELDS) == set(record_props) - record_own
    for name in RELATIONSHIP_FIELDS:
        assert amend_props[name] == record_props[name], name
    assert amend_props["mechanisms"] == record_props["mechanisms"]
    assert {"type": "string"} in amend_props["axiom"]["anyOf"]
    assert record_props["axiom"]["type"] == "string"
    assert tools["amend_commentary"].inputSchema.get("required") == ["slug"]


# --------------------------------------------------------------------------- #
# M10 — rename, and the edit its clause describes
# --------------------------------------------------------------------------- #

def test_a_rename_lists_its_citers_and_editing_their_line_is_the_whole_repair(ws) -> None:
    """M10 — `rename.incoming` rows, a clause naming no tool, and 4b C5's proof on this surface."""
    config, m = ws
    _cited(config, m)

    payload = _call(slug="target", new_slug="target-renamed")
    assert payload["rename"] == {"from": "target", "to": "target-renamed",
                                 "incoming": [{"kind": "cites", "source": "citer"}]}
    assert "'target-renamed'" in payload["recovery"]
    assert not any(name in payload["recovery"] for name in _tools())

    assert _call(slug="citer", context="A repair.")["reason"] == amend.REASON_DIVERGED

    text = _buffer(config)
    assert text.count("**Cites:** target\n") == 1
    _write_buffer(config, text.replace("**Cites:** target\n", "**Cites:** target-renamed\n"))
    citer = m.store.get_node_by_slug("citer")
    report = entry_divergence(_entry(config, "citer"), citer, citer["scope"],
                              m.store.get_outgoing_edges(citer["id"]))
    assert not any(report.values()), report
    assert _call(slug="citer", context="A repair.")["status"] == amend.STATUS_AMENDED


# --------------------------------------------------------------------------- #
# S1 / S3 (M9) — a real `mitos serve` over stdio
# --------------------------------------------------------------------------- #

def _served_workspace(tmp_path, env):
    """A workspace holding one archived entry and one buffered one, both committed."""
    ws = _rotating_workspace(tmp_path, "ws_amend", env=env)
    _run_mitos("-p", str(ws), "record", "The fresh axiom.", "--rejected", "rej",
               "--slug", "fresh-write", "--acknowledge-neighbors", cwd=ws, env=env)
    text = (ws / "decisions.md").read_text(encoding="utf-8")
    archived = [f"seed-{i}" for i in range(3) if f"### seed-{i}\n" not in text]
    assert archived and "### fresh-write\n" in text, text
    return ws, archived[0]


@pytest.mark.asyncio
async def test_a_real_serve_returns_the_misses_and_raises_only_the_fidelity_refusal(tmp_path):
    """S1 + S3 — `isError` per class over the wire; the undeclared-argument measurement.

    M9: FastMCP registers its handler with `validate_input=False` and its argument
    model sets no `extra`, so an undeclared key was measured DROPPED silently —
    `rationale=` beside `context=` amended the context and said nothing about the
    rationale. Since AX-3 6b mitos's boundary refuses it before the tool runs: an
    `isError` naming `rationale`, and the buffer byte-unchanged. `axiom`,
    `mechanisms` and the relations stay declared so that their refusal names the
    route that does change them, which "not an argument" would not; `axiom=` beside
    `context=` refuses with `canonical_core` and writes nothing.
    """
    env = _scaffold_env(tmp_path)
    ws, archived = _served_workspace(tmp_path, env)
    buffer = ws / "decisions.md"

    async with mitos_server(cwd=ws, env=env) as server:
        async def call(**arguments):
            return await server.session.call_tool(
                "amend_commentary", {"project": str(ws), **arguments})

        not_found = _tool_json(await call(slug="no-such-handle", context="x"))
        archived_miss = _tool_json(await call(slug=archived, context="x"))
        core = _tool_json(await call(slug="fresh-write", mechanisms=["m"]))
        amended = _tool_json(await call(slug="fresh-write", context="A repaired context."))

        before = buffer.read_bytes()
        fidelity = await call(slug="fresh-write", context=PHANTOM)
        after_fidelity = buffer.read_bytes()

        dropped = await call(slug="fresh-write", context="Another context.",
                             rationale="an undeclared argument")
        before_silent = buffer.read_bytes()
        silent = _tool_json(await call(slug="fresh-write", context="Yet another.",
                                       axiom="A different axiom."))
        after_silent = buffer.read_bytes()

    assert not_found["status"] == amend.STATUS_NOT_FOUND
    assert archived_miss["status"] == amend.STATUS_ARCHIVED
    assert (core["status"], core["reason"]) == (amend.STATUS_REFUSED, amend.REASON_CANONICAL_CORE)
    for payload in (not_found, archived_miss, core):
        assert "mitos " not in payload["recovery"]
    assert amended["status"] == amend.STATUS_AMENDED and "recovery" not in amended

    assert fidelity.isError is True
    body = fidelity.content[0].text
    assert body.startswith("Error executing tool amend_commentary: [buffer_fidelity] ")
    assert "object at 0x" not in body and "mitos " not in body
    assert after_fidelity == before

    assert dropped.isError is True, "an undeclared argument is refused, not dropped"
    refusal = dropped.content[0].text
    assert refusal.startswith("amend_commentary was not run: 1 argument fault.")
    assert "`rationale`" in refusal and "Another context." not in refusal
    assert before_silent == after_fidelity, "the refused call wrote nothing"
    assert (silent["status"], silent["reason"]) == (amend.STATUS_REFUSED,
                                                    amend.REASON_CANONICAL_CORE)
    assert after_silent == before_silent


def test_an_amend_and_a_fidelity_refusal_keep_the_transport_json_rpc(tmp_path):
    """S1 — raw stdout over `_raw_stdio_exchange` (drain, then EOF): JSON-RPC lines only."""
    env = _scaffold_env(tmp_path)
    ws, _archived_slug = _served_workspace(tmp_path, env)

    def call(request_id, **arguments):
        return {"jsonrpc": "2.0", "id": request_id, "method": "tools/call",
                "params": {"name": "amend_commentary",
                           "arguments": {"project": str(ws), **arguments}}}

    exchange = "".join(json.dumps(message) + "\n" for message in (
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "amend-probe", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        call(2, slug="fresh-write", context="A repaired context."),
        call(3, slug="fresh-write", context=PHANTOM),
    ))

    stdout, stderr = _raw_stdio_exchange(exchange, expect_lines=3, cwd=ws, env=env)

    lines = [line for line in stdout.splitlines() if line.strip()]
    assert len(lines) == 3, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    messages = {}
    for line in lines:
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"non-JSON line on the protocol channel: {line!r}") from exc
        messages[message.get("id")] = message
    amended, refused = messages[2]["result"], messages[3]["result"]
    assert amended["isError"] is False
    assert json.loads(amended["content"][0]["text"])["status"] == amend.STATUS_AMENDED
    assert refused["isError"] is True and "[buffer_fidelity]" in refused["content"][0]["text"]
