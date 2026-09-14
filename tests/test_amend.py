"""Pure rows for the commentary amendment: `mitos.amend` and `restore.verify_amended_buffer`.

No workspace, no store: the graph is two plain functions over a dict, and every buffer is
text. The manager rows live in `test_amend_commentary.py`.
"""

import json
import subprocess
import sys
from typing import Dict, Optional

import pytest

from mitos.amend import (
    EDITABLE_FIELDS,
    FIELD_LABELS,
    ROUTES,
    Miss,
    Target,
    apply_changes,
    classify_target,
    entry_node_id,
    expected_fingerprint,
    validate_changes,
)
from mitos.divergence import COMMENTARY_FIELDS, RELATIONSHIP_FIELDS
from mitos.identity import SLUG_MAX_LEN
from mitos.parser import FIELD_MAP, parse_entry_stream
from mitos.restore import BufferFidelityError, _entry_fingerprint, verify_amended_buffer

_HEADER = (
    "# Decisions\n"
    "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->\n\n"
)
_TARGET = (
    "### target\n"
    "\n"
    "**Decided:** The target axiom.\n"
    "**Rejected:** First rejected line.\n"
    "  second rejected line\n"
    "**Mechanisms:** sqlite\n"
    "**Scope:** alpha, beta\n"
    "**Context:** Target context.\n"
    "**Cites:** [neighbour]\n"
    "[DECISION_TRANSCRIPT]\n"
    "User: keep it.\n"
    "**Scope:** transcript text, not a field\n"
    "[/DECISION_TRANSCRIPT]\n"
    "\n"
)
_NEIGHBOUR = (
    "### neighbour\n"
    "\n"
    "**Decided:** The neighbour axiom.\n"
    "**Rejected:** Neighbour rejected.\n"
    "**Mechanisms:** qdrant\n"
    "**Scope:** gamma\n"
    "**Context:** Neighbour context.\n"
)
BUFFER = _HEADER + _TARGET + _NEIGHBOUR


def _entry(text: str, slug: str):
    return next(e for e in parse_entry_stream(text, "decision") if e.slug == slug)


def _target(text: str = BUFFER, slug: str = "target") -> Target:
    entry = _entry(text, slug)
    node_id = entry_node_id(entry)
    return Target(entry=entry, node_id=node_id, node={"id": node_id, "slug": slug})


def _graph(nodes: Dict[str, Dict]):
    """A fake `resolve_handle` / `get_node` pair over ``{id: node}``."""
    def resolve(handle: str) -> Optional[Dict]:
        if handle in nodes:
            return nodes[handle]
        return next((n for n in nodes.values() if n["slug"].casefold() == handle.casefold()), None)

    return resolve, nodes.get


# --- P1: the field set and its labels ------------------------------------------------

def test_the_editable_set_is_the_reconciles_commentary_plus_scope() -> None:
    """P1 — derived, never restated."""
    assert EDITABLE_FIELDS == tuple(COMMENTARY_FIELDS) + ("scope",)


def test_every_label_reads_back_through_the_parsers_field_map() -> None:
    """P1b — the explicit label table cannot drift off format-spec's single source."""
    assert set(FIELD_LABELS) == set(EDITABLE_FIELDS) - {"slug"}
    for field, label in FIELD_LABELS.items():
        assert FIELD_MAP[label.lower()] == field


# --- P2: validation ------------------------------------------------------------------

_CORE = "canonical_core"
_REFUSALS = [
    ({"axiom": "x"}, _CORE, ["axiom"]),
    ({"decided": "x"}, _CORE, ["decided"]),
    ({"mechanisms": ["x"]}, _CORE, ["mechanisms"]),
    *[({field: "x"}, "edges", [field]) for field in RELATIONSHIP_FIELDS],
    ({"edges": []}, "edges", ["edges"]),
    *[({field: "x"}, "not_editable", [field])
      for field in ("source", "transcript", "confirmed_by", "confirmed_at")],
    ({"colour": "x"}, "unknown_field", ["colour"]),
    ({}, "no_changes", []),
    ({"rejected_paths": "   \n "}, "invalid_value", ["rejected_paths"]),
    ({"rejected_paths": None}, "invalid_value", ["rejected_paths"]),
    ({"context": 3}, "invalid_value", ["context"]),
    ({"slug": "x" * (SLUG_MAX_LEN + 1)}, "invalid_value", ["slug"]),
    ({"slug": "two words"}, "invalid_value", ["slug"]),
    ({"slug": "a—b"}, "invalid_value", ["slug"]),
    ({"slug": ""}, "invalid_value", ["slug"]),
    ({"scope": "alpha"}, "invalid_value", ["scope"]),
    ({"scope": ["a,b"]}, "invalid_value", ["scope"]),
    ({"scope": ["a\n### phantom"]}, "invalid_value", ["scope"]),
    ({"context": "opens <!-- a comment"}, "invalid_value", ["context"]),
    # Precedence: the canonical core outranks everything else in one request.
    ({"axiom": "x", "colour": "y", "context": 3}, _CORE, ["axiom"]),
]


@pytest.mark.parametrize("changes, reason, fields", _REFUSALS)
def test_a_request_that_cannot_be_honoured_is_refused_as_a_request(changes, reason, fields) -> None:
    """P2 — each refusal names its reason and the keys it is about; routes are data."""
    result = validate_changes(changes)
    assert result is not None
    assert (result["status"], result["reason"], result["fields"]) == ("refused", reason, fields)
    if reason == _CORE:
        assert result["routes"] == ROUTES
        assert set(result["routes"].values()) == {"supersedes", "corrects", "amends"}
    else:
        assert "routes" not in result
    assert json.loads(json.dumps(result)) == result


def test_a_well_formed_request_passes_validation() -> None:
    """P2 — clearing optional fields, an empty scope and a maximal slug are all legal."""
    assert validate_changes({
        "context": None, "invalidates_if": "", "scope": [], "rejected_paths": "r\n  r2",
        "slug": "x" * SLUG_MAX_LEN,
    }) is None
    assert validate_changes({"slug": "Target", "context": "closed <!-- ok --> comment"}) is None


# --- P3: classification --------------------------------------------------------------

def test_a_committed_buffered_block_is_the_target_by_slug_casefold_or_id() -> None:
    """P3 — the one locked read plus the graph name the block."""
    target_id = entry_node_id(_entry(BUFFER, "target"))
    resolve, lookup = _graph({target_id: {"id": target_id, "slug": "target", "kind": "decision"}})
    for handle in ("target", "TARGET", target_id):
        found = classify_target(BUFFER, slug=handle, resolve=resolve, lookup=lookup)
        assert isinstance(found, Target) and found.node_id == target_id
        assert found.entry.slug == "target"


def test_a_block_with_no_node_is_an_uncommitted_draft() -> None:
    """P3 — even when an archived node elsewhere carries the same slug (mutant d)."""
    resolve, lookup = _graph({})
    assert classify_target(BUFFER, slug="target", resolve=resolve, lookup=lookup).result == {
        "status": "uncommitted"
    }
    archived = {"archived-id": {"id": "archived-id", "slug": "target", "kind": "decision"}}
    resolve, lookup = _graph(archived)
    found = classify_target(BUFFER, slug="target", resolve=resolve, lookup=lookup)
    assert isinstance(found, Miss) and found.result == {"status": "uncommitted"}


@pytest.mark.parametrize("kind, expected", [
    ("decision", {"status": "archived", "id": "gone-id"}),
    ("open_question", {"status": "refused", "reason": "open_question", "fields": []}),
])
def test_a_node_no_buffered_block_bears_is_archived_or_an_open_question(kind, expected) -> None:
    """P3 — a graph lookup plus the read the verb already holds; no archive read."""
    resolve, lookup = _graph({"gone-id": {"id": "gone-id", "slug": "gone", "kind": kind}})
    found = classify_target(BUFFER, slug="gone", resolve=resolve, lookup=lookup)
    assert isinstance(found, Miss) and found.result == expected


def test_a_handle_nothing_bears_is_not_found() -> None:
    resolve, lookup = _graph({})
    assert classify_target(BUFFER, slug="nothing", resolve=resolve, lookup=lookup).result == {
        "status": "not_found"
    }


def test_an_unparseable_block_bearing_the_handle_is_refused_not_misreported() -> None:
    """P3 — a malformed block is not "absent": it would otherwise read as archived."""
    text = BUFFER + "\n### broken\n\n**Decided:** No rejected line.\n"
    resolve, lookup = _graph({"broken-id": {"id": "broken-id", "slug": "broken", "kind": "decision"}})
    found = classify_target(text, slug="broken", resolve=resolve, lookup=lookup)
    assert isinstance(found, Miss) and found.result["reason"] == "unparseable"


def test_two_copies_of_the_target_raise_because_the_splice_cannot_be_proven() -> None:
    """P3 — the buffer is already ambiguous."""
    text = _HEADER + _TARGET + _TARGET + _NEIGHBOUR
    target_id = entry_node_id(_entry(BUFFER, "target"))
    resolve, lookup = _graph({target_id: {"id": target_id, "slug": "target", "kind": "decision"}})
    with pytest.raises(BufferFidelityError, match="copies"):
        classify_target(text, slug="target", resolve=resolve, lookup=lookup)


# --- P4: surgery ---------------------------------------------------------------------

def test_replacing_a_multiline_field_changes_only_its_extent() -> None:
    """P4 — every other byte, the transcript's field-shaped line included, is kept."""
    after = apply_changes(BUFFER, _target(), {"rejected_paths": "New one.\n  New two."})
    assert after == BUFFER.replace(
        "**Rejected:** First rejected line.\n  second rejected line\n",
        "**Rejected:** New one.\n  New two.\n",
    )
    assert _entry(after, "target").rejected_paths == "New one.\nNew two."


def test_an_absent_field_is_inserted_in_format_spec_order() -> None:
    """P4 — Invalidates-If lands after Scope and before Context."""
    after = apply_changes(BUFFER, _target(), {"invalidates_if": "When it moves."})
    assert after == BUFFER.replace(
        "**Scope:** alpha, beta\n", "**Scope:** alpha, beta\n**Invalidates-If:** When it moves.\n"
    )


def test_clearing_and_inserting_in_one_request() -> None:
    """P4 — removal plus insertion at the same anchor, before the relation line."""
    after = apply_changes(BUFFER, _target(), {"context": None, "invalidates_if": "When."})
    assert after == BUFFER.replace(
        "**Context:** Target context.\n", ""
    ).replace("**Scope:** alpha, beta\n", "**Scope:** alpha, beta\n**Invalidates-If:** When.\n")
    assert _entry(after, "target").context is None


def test_a_scope_reorder_and_a_rename_rewrite_their_lines_only() -> None:
    """P4 — the heading keeps its prefix; the scope line is emitted normalized."""
    after = apply_changes(BUFFER, _target(), {"scope": ["Beta", "alpha"], "slug": "renamed"})
    assert after == BUFFER.replace("**Scope:** alpha, beta\n", "**Scope:** beta, alpha\n").replace(
        "### target\n", "### renamed\n"
    )


def test_a_dated_heading_keeps_its_date_and_title_on_rename() -> None:
    """P4 — only the slug token moves, even when the title repeats it."""
    text = BUFFER.replace("### target\n", "## 2026-01-01 — target — target title\n")
    after = apply_changes(text, _target(text), {"slug": "renamed"})
    assert after == text.replace(
        "## 2026-01-01 — target — target title\n", "## 2026-01-01 — renamed — target title\n"
    )


def test_crlf_line_endings_survive_an_edit() -> None:
    text = BUFFER.replace("\n", "\r\n")
    after = apply_changes(text, _target(text), {"context": "New."})
    assert after == text.replace("**Context:** Target context.\r\n", "**Context:** New.\r\n")


def test_an_insertion_after_an_unterminated_last_line_terminates_it() -> None:
    text = _HEADER + _NEIGHBOUR.rstrip("\n")
    after = apply_changes(text, _target(text, "neighbour"), {"invalidates_if": "When."})
    assert after.endswith("**Scope:** gamma\n**Invalidates-If:** When.\n**Context:** Neighbour context.")
    assert _entry(after, "neighbour").invalidates_if == "When."


def test_a_repeated_field_line_collapses_to_one_carrying_the_new_value() -> None:
    """P4 — the parser keeps the last of two lines; surgery replaces both."""
    text = BUFFER.replace("**Context:** Target context.\n",
                          "**Context:** First.\n**Context:** Target context.\n")
    after = apply_changes(text, _target(text), {"context": "Only."})
    assert after == BUFFER.replace("**Context:** Target context.\n", "**Context:** Only.\n")


def test_clearing_an_already_empty_field_line_is_no_change() -> None:
    """P4b — `""` and `None` are one absence, so no attribution row for nothing."""
    from mitos.amend import fingerprint_matches

    text = BUFFER.replace("**Scope:** alpha, beta\n", "**Scope:** alpha, beta\n**Invalidates-If:**\n")
    entry = _entry(text, "target")
    assert entry.invalidates_if == ""
    assert fingerprint_matches(expected_fingerprint(entry, {"invalidates_if": None}), entry)
    assert not fingerprint_matches(expected_fingerprint(entry, {"invalidates_if": "When."}), entry)


# --- P5: the fence -------------------------------------------------------------------

def _verify(after: str, changes: Dict) -> object:
    target = _target()
    return verify_amended_buffer(
        BUFFER, after, target_id=target.node_id,
        expected=expected_fingerprint(target.entry, changes),
    )


def test_the_fence_passes_a_real_amendment_and_returns_the_parsed_entry() -> None:
    changes = {"context": "New.\n  second line", "scope": ["beta"]}
    amended = _verify(apply_changes(BUFFER, _target(), changes), changes)
    assert amended.context == "New.\nsecond line" and amended.scope == ["beta"]


@pytest.mark.parametrize("after, changes, match", [
    (BUFFER.replace("**Context:** Target context.\n", "**Context:** c\n### phantom\n"),
     {"context": "c"}, "parse failure|entry count"),
    (BUFFER + "\n### extra\n\n**Decided:** Extra.\n**Rejected:** R.\n",
     {}, "entry count"),
    (BUFFER.replace("**Context:** Neighbour context.\n", "**Context:** Neighbour context.\nbleed\n"),
     {}, "neighbouring"),
    (BUFFER.replace("**Context:** Target context.\n", "**Context:** c\n**Decided:** other\n"),
     {"context": "c"}, "no longer hashes"),
    (BUFFER.replace("**Context:** Target context.\n", "**Context:** c\n**Scope:** evil\n"),
     {"context": "c"}, "scope"),
    (BUFFER.replace("**Cites:** [neighbour]\n", ""), {}, "edges"),
])
def test_the_fence_refuses_each_disturbance(after, changes, match) -> None:
    """P5 — one row per condition: failure/phantom, count, neighbour, core, fields."""
    with pytest.raises(BufferFidelityError, match=match):
        _verify(after, changes)


def test_the_neighbour_check_is_a_multiset_not_a_slug_keyed_map() -> None:
    """P5 — two same-slug neighbours: rewriting the first into the second must fail."""
    dup_a = _NEIGHBOUR.replace("neighbour", "dup").replace("qdrant", "m-a")
    dup_b = _NEIGHBOUR.replace("neighbour", "dup").replace("qdrant", "m-b")
    before = _HEADER + _TARGET + dup_a + "\n" + dup_b
    after = _HEADER + _TARGET + dup_b + "\n" + dup_b
    target = _target(before)
    with pytest.raises(BufferFidelityError, match="neighbouring"):
        verify_amended_buffer(before, after, target_id=target.node_id,
                              expected=_entry_fingerprint(target.entry))


# --- P6: the tier --------------------------------------------------------------------

def test_importing_amend_pulls_in_no_store_sync_lock_telemetry_or_sdk() -> None:
    """P6 — the graph arrives as callables; the leaf never imports the store."""
    probe = (
        "import sys; import mitos.amend; "
        "print(','.join(sorted(m for m in ('mitos.store', 'mitos.sync', 'mitos.cutover', "
        "'mitos.telemetry', 'filelock', 'anthropic', 'google.genai') if m in sys.modules))); "
        "print('mitos.parser' in sys.modules and 'mitos.restore' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.split("\n")[:2] == ["", "True"], out.stdout
