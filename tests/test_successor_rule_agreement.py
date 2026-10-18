"""The successor rule's three readers agree, and its key renderers print by membership.

The rule is "live end, else the immediate kill stamps" (WIRING_LEDGER entry-018). Read at
source it has three readers and no fourth:

1. ``sync._live_successors`` — the write path's (pause, ``exists``, CLI recovery lines);
2. ``recall._successor_slugs`` — the fenced twin on the read path (``recall.py`` sits
   behind the check family's no-write fence, so the two are not merged);
3. ``stamp_labels._named_slugs`` — the label leaf's, which also adds ``amended_by`` /
   ``narrowed_by`` (stamps a label names that are not the successor rule).

``cli.cmd_record``'s tombstone loop is not a reader. It prints every successor key a
handle holds, tested by membership, and two sibling printers share the shape:
``cli._print_retired_partition`` and ``cli._print_rejection_section``'s retired letter.
Membership is what keeps a ``live_end: []`` (the chain reaches no live decision) from
vanishing; a truthiness test would drop it silently.

Where the readers differ, the row states it rather than hiding it under a set:
``_successor_slugs`` answers ``None`` for "no successor key read" (mapped to ``[]``
here), and only ``_live_successors`` deduplicates across the two stamps. A ``None``
VALUE is out of scope: only ``sync._exists_handle_keys``' fault path writes one, and
that dict never reaches ``recall`` or ``stamp_labels``.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from mitos import cli, recall, stamp_labels, sync

from test_neighbor_review import offline, ws  # noqa: F401  (fixtures)
from test_stamp_labels import _chain_handles, gstore  # noqa: F401  (fixture)

_ID = "e" * 64
_KEYS = ("live_end", "superseded_by", "corrected_by")


def _handle(slug: str, **keys: List[str]) -> Dict[str, Any]:
    """A handle in ``display.retired_handle``'s shape: stamps only when non-empty,
    ``live_end`` by membership."""
    return {"slug": slug, "id": _ID, "state": "superseded", "score": 0.9, **keys}


#: The handle table: (case, handle, the rule's answer).
TABLE = [
    ("live end beside a retired intermediate's stamp",
     _handle("t1", superseded_by=["mid"], live_end=["end"]), ["end"]),
    ("live_end [] beside a dead stamp",
     _handle("t2", superseded_by=["dead"], live_end=[]), []),
    ("stamps only", _handle("t3", superseded_by=["s"], corrected_by=["c"]), ["s", "c"]),
    ("no successor key", _handle("t4"), []),
]

#: The one handle where the twins' lists differ from the rule's: a slug on both stamps.
DUPLICATE = _handle("t5", superseded_by=["b"], corrected_by=["b"])


def _readers(handle: Dict[str, Any]) -> Dict[str, List[str]]:
    slugs = recall._successor_slugs(handle)
    return {
        "_live_successors": sync._live_successors(handle),
        # None means "no successor key read"; on a handle that carries none it is [].
        "_successor_slugs": [] if slugs is None else slugs,
        "_named_slugs": stamp_labels._named_slugs(handle),
    }


# -- The readers ---------------------------------------------------------------------------

@pytest.mark.parametrize("case,handle,expected", TABLE, ids=[c for c, _, _ in TABLE])
def test_the_three_readers_give_the_rules_answer(case, handle, expected):
    assert _readers(handle) == {name: expected for name in _readers(handle)}, case


def test_successor_slugs_answers_none_only_when_no_successor_key_was_read():
    assert recall._successor_slugs(_handle("t4")) is None
    assert recall._successor_slugs(_handle("t2", live_end=[])) == []


def test_only_the_write_paths_reader_deduplicates_and_otherwise_they_agree():
    readers = _readers(DUPLICATE)
    assert readers["_live_successors"] == ["b"]
    assert readers["_successor_slugs"] == ["b", "b"] == readers["_named_slugs"]
    assert list(dict.fromkeys(readers["_successor_slugs"])) == readers["_live_successors"]


def test_named_slugs_adds_the_non_kill_stamps_beyond_the_rule():
    handle = {**_handle("t6", live_end=[]), "amended_by": ["am"], "narrowed_by": ["na"]}
    assert stamp_labels._named_slugs(handle) == ["am", "na"]
    assert sync._live_successors(handle) == [] == recall._successor_slugs(handle)


def test_the_readers_agree_on_a_real_graphs_handles(gstore):
    """The handles ``retired_handle`` builds over a real kill-chain graph, compared as
    lists (the 8a pin compares sets, which hides order and duplicates)."""
    handles = _chain_handles(gstore)
    assert any("live_end" in h for h in handles)                  # the chain case is in
    for handle in handles:
        readers = _readers(handle)
        assert readers["_successor_slugs"] == readers["_live_successors"], handle
        assert readers["_named_slugs"] == readers["_live_successors"], handle


# -- The renderers -------------------------------------------------------------------------

def _expected_lines(handle: Dict[str, Any], none: str) -> Dict[str, str]:
    return {key: (", ".join(handle[key]) if handle[key] else none)
            for key in _KEYS if key in handle}


def test_cmd_records_tombstone_lines_print_every_present_key_by_membership(ws):
    config, _ = ws
    tombs = [{**h, "axiom": f"Axiom for {h['slug']}."} for _, h, _ in TABLE]
    pause = {"status": "needs_review", "code": "similar_decision_exists", "slug": "draft",
             "neighbors": [], "tombstones": tombs, "message": "paused"}

    class _Canned:
        def __init__(self, *_a, **_k):
            pass

        def record_decision_entry(self, *_a, **_k):
            return pause

    err = io.StringIO()
    with patch("mitos.cli.MitosSyncManager", _Canned), redirect_stderr(err), \
            redirect_stdout(io.StringIO()), pytest.raises(SystemExit):
        cli.cmd_record(config, axiom="A draft.", rejected="rej", scope=["s"], slug="draft")
    blocks = err.getvalue().split("  † ")[1:]
    assert len(blocks) == len(tombs)
    for (case, handle, _), block in zip(TABLE, blocks):
        assert block.startswith(handle["slug"]), case
        for key in _KEYS:
            label = f"      {key.replace('_', ' ')}: "
            if key in handle:
                value = _expected_lines(handle, "(none reached)")[key]
                assert f"{label}{value}\n" in block, (case, key)
            else:
                assert label not in block, (case, key)
    assert "      live end: (none reached)\n" in blocks[1]


def test_the_retired_partition_printer_keeps_an_empty_live_end():
    out = io.StringIO()
    with redirect_stdout(out):
        cli._print_retired_partition([h for _, h, _ in TABLE], 0)
    lines = [ln for ln in out.getvalue().splitlines() if ln.startswith("  t")]
    assert len(lines) == len(TABLE), lines                       # zip would hide a drop
    for (case, handle, _), line in zip(TABLE, lines):
        for key in _KEYS:
            if key in handle:
                assert f"  {key}: {_expected_lines(handle, '(none)')[key]}" in line, case
            else:
                assert f"{key}:" not in line, case
    assert lines[1].endswith("live_end: (none)")


def test_the_rejection_sections_retired_letter_keeps_an_empty_live_end():
    matches = [{**h, "item": f"Item of {h['slug']}.", "ordinal": 1, "letter": "retired"}
               for _, h, _ in TABLE]
    out = io.StringIO()
    with redirect_stdout(out):
        cli._print_rejection_section(SimpleNamespace(matches=matches))
    text = out.getvalue()
    assert "live_end: (none)" in text
    for case, handle, _ in TABLE:
        block = text.split(f"  {handle['slug']}  (", 1)[1].split("\n  t", 1)[0]
        for key in _KEYS:
            if key in handle:
                assert f" {key}: {_expected_lines(handle, '(none)')[key]}" in block, case
            else:
                assert f"{key}:" not in block, case
