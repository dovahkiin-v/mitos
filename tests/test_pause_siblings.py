"""Phase 7c: the declared target's sibling — a live neighbour that descends from a
target the draft declares says so on the pause.

When a draft declares ``amends R``, the pause drops R and R's ancestors from its live
neighbours and deliberately keeps R's descendants: a co-child of R may be a genuine
duplicate of the draft. ``MitosSyncManager._review_siblings`` walks each shown live
neighbour's own mutation lineage (``GraphStore.get_lineage_ids``) and marks the
neighbour ``descends_from_declared`` with the declared targets whose own id that
lineage holds. The note shows and never suppresses. (Vision §4.9; plan p7c §3, §4,
§9 rows 1–13.)

Every row runs the record path over a real workspace, the gather armed through
``CannedVector`` in rank order, as ``test_pause_tombstones.py`` does.
"""

import json
from unittest.mock import patch

import pytest

from mitos import sync
from mitos.errors import DatabaseError
from mitos.sync import MitosSyncManager

from test_neighbor_review import _arm, _seed, offline, ws  # noqa: F401
from test_pause_tombstones import _Counting, _HIGH, _draft, _id_of, _retire, _surfaces

_KEY = "descends_from_declared"
_STEP = 0.01
# Neighbour scores, closest first, all above the live floor and below _HIGH.
_NEAR = [round(_HIGH - _STEP * i, 4) for i in range(1, 8)]
assert _NEAR[-1] >= sync._NEIGHBOR_REVIEW_THRESHOLD
_CLAUSE = ("A neighbour marked descends_from_declared descends from a decision this "
           "draft declares, named there; that declaration does not clear it, so judge "
           "it like any other neighbour.")
_TEXT_LINE = "descends from what this draft declares: "


def _notes(res):
    """The pause's neighbours as (slug, note or None), in pause order."""
    return [(n["slug"], n.get(_KEY)) for n in res["neighbors"]]


# -- Row 1: the direct co-amender, on MCP, --json and text --------------------------------

def test_a_direct_co_amender_names_the_declared_target_on_every_surface(ws):
    config, m = ws
    _seed(m, "r", "Root decision R.")
    _seed(m, "n", "N amends R.", amends="r")
    mcp, cli_json, text, code = _surfaces(config, [{"slug": "n", "score": _HIGH}],
                                          amends="R")
    assert code == 2 and mcp["status"] == "needs_review"
    # The stored spelling, never the typed one (the echo's naming rule).
    assert _notes(mcp) == [("n", ["r"])]
    assert cli_json["neighbors"] == mcp["neighbors"]
    assert f"      {_TEXT_LINE}r\n" in text


# -- Row 2: a transitive descendant -------------------------------------------------------

def test_a_transitive_descendant_and_its_parent_both_carry_the_note(ws):
    config, m = ws
    _seed(m, "r", "Root decision R.")
    _seed(m, "mid", "M amends R.", amends="r")
    _seed(m, "n", "N amends M.", amends="mid")
    _arm(m, [{"slug": "n", "score": _HIGH}, {"slug": "mid", "score": _NEAR[0]}])
    res = _draft(m, amends="r")
    # M is R's descendant, not its ancestor: shown, never suppressed.
    assert _notes(res) == [("n", ["r"]), ("mid", ["r"])]


# -- Row 3: an unrelated neighbour carries no key -----------------------------------------

def test_an_unrelated_neighbour_carries_no_key_at_all(ws):
    config, m = ws
    _seed(m, "r", "Root decision R.")
    _seed(m, "n", "N amends R.", amends="r")
    _seed(m, "u", "Unrelated U.")
    _arm(m, [{"slug": "u", "score": _HIGH}, {"slug": "n", "score": _NEAR[0]}])
    res = _draft(m, amends="r")
    assert _notes(res) == [("u", None), ("n", ["r"])]
    assert _KEY not in res["neighbors"][0]


# -- Row 4: the cousin (D1) ---------------------------------------------------------------

def test_a_cousin_of_the_declared_target_carries_no_note(ws):
    """T amends R; N amends R; the draft amends T. N shares T's ancestor and does not
    descend from T. R is suppressed as T's ancestor; N is not."""
    config, m = ws
    _seed(m, "r", "Root decision R.")
    _seed(m, "t", "T amends R.", amends="r")
    _seed(m, "n", "N amends R.", amends="r")
    _arm(m, [{"slug": "n", "score": _HIGH}, {"slug": "r", "score": _NEAR[0]}])
    res = _draft(m, amends="t")
    assert _notes(res) == [("n", None)]


# -- Row 5: the note suppresses nothing ---------------------------------------------------

def test_the_note_suppresses_nothing(ws):
    config, m = ws
    _seed(m, "r", "Root decision R.")
    _seed(m, "n", "N amends R.", amends="r")
    for i in range(6):
        _seed(m, f"u{i}", f"Unrelated decision {i}.")
    _retire(m, "old-way")
    matches = ([{"slug": "old-way", "score": _HIGH}, {"slug": "n", "score": _NEAR[0]}]
               + [{"slug": f"u{i}", "score": _NEAR[i + 1]} for i in range(6)])
    _arm(m, matches)
    noted = _draft(m, amends="r")
    with patch.object(MitosSyncManager, "_review_siblings", return_value={}):
        bare = _draft(m, amends="r")
    assert noted["neighbors_omitted"] == 2 and noted["tombstones"]
    assert [n["slug"] for n in noted["neighbors"]] == [n["slug"] for n in bare["neighbors"]]
    stripped = [{k: v for k, v in n.items() if k != _KEY} for n in noted["neighbors"]]
    assert stripped == bare["neighbors"]
    assert noted["neighbors"] != bare["neighbors"]
    assert {k: v for k, v in noted.items() if k not in ("neighbors", "message")} == \
        {k: v for k, v in bare.items() if k not in ("neighbors", "message")}
    assert noted["message"].replace(f" {_CLAUSE}", "", 1) == bare["message"]


# -- Row 6: two targets, declaration order, each once -------------------------------------

def test_two_targets_are_named_in_declaration_order_each_once(ws):
    config, m = ws
    _seed(m, "r", "Root decision R.")
    _seed(m, "s", "Root decision S.")
    _seed(m, "n", "N amends R and S.", amends="s, r")
    _arm(m, [{"slug": "n", "score": _HIGH}])
    # The relation catalogue orders the declarations (amends before cites), whatever
    # the keyword order of the call.
    assert _notes(_draft(m, cites="s", amends="r")) == [("n", ["r", "s"])]
    assert _notes(_draft(m, amends="r", depends_on="r")) == [("n", ["r"])]


# -- Row 7: any declared relation counts (D2) ---------------------------------------------

@pytest.mark.parametrize("relation", ["depends_on", "corrects", "cites"])
def test_any_declared_relation_counts(ws, relation):
    config, m = ws
    _seed(m, "r", "Root decision R.")
    _seed(m, "n", "N amends R.", amends="r")
    _arm(m, [{"slug": "n", "score": _HIGH}])
    assert _notes(_draft(m, **{relation: "r"})) == [("n", ["r"])]


# -- Row 8: the note lives on the pause only ----------------------------------------------

def test_an_acknowledged_write_carries_no_note(ws):
    config, m = ws
    _seed(m, "r", "Root decision R.")
    _seed(m, "n", "N amends R.", amends="r")
    _arm(m, [{"slug": "n", "score": _HIGH}])
    assert _notes(_draft(m, amends="r")) == [("n", ["r"])]
    res = _draft(m, amends="r", acknowledge_neighbors=True)
    assert res["status"] == "created"
    assert res["acknowledged_neighbors"] == ["n"]
    assert _KEY not in json.dumps(res)


# -- Row 9: corrects is not lineage -------------------------------------------------------

def test_a_neighbour_that_only_corrects_the_target_carries_no_note(ws):
    config, m = ws
    _seed(m, "r", "Root decision R.")
    _seed(m, "n", "N corrects R.", corrects="r")
    _arm(m, [{"slug": "n", "score": _HIGH}])
    assert _notes(_draft(m, amends="r")) == [("n", None)]


# -- Row 10: the read's cost --------------------------------------------------------------

def _counted(m):
    """Installs the counting store on the armed manager, and a spy on the seeds."""
    seeds = []
    real = m.store.get_lineage_ids

    def spy(node_ids):
        seeds.append(list(node_ids))
        return real(node_ids)
    m.store.get_lineage_ids = spy
    m.store = _Counting(m.store)
    return seeds


def test_no_declaration_reads_no_lineage(ws):
    config, m = ws
    _seed(m, "u", "Unrelated U.")
    _arm(m, [{"slug": "u", "score": _HIGH}])
    seeds = _counted(m)
    res = _draft(m)
    assert res["status"] == "needs_review"
    assert m.store.calls.get("get_lineage_ids", 0) == 0 and seeds == []


def test_one_batched_read_seeds_the_shown_neighbours_only(ws):
    config, m = ws
    _seed(m, "r", "Root decision R.")
    slugs = [f"n{i}" for i in range(7)]
    for slug in slugs:
        _seed(m, slug, f"{slug} amends R.", amends="r")
    _arm(m, [{"slug": s, "score": _NEAR[i]} for i, s in enumerate(slugs)])
    seeds = _counted(m)
    res = _draft(m, amends="r")
    shown = [n["slug"] for n in res["neighbors"]]
    assert shown == slugs[:5] and res["neighbors_omitted"] == 2
    assert m.store.calls["get_lineage_ids"] == 1
    assert seeds == [[_id_of(m, s) for s in shown]]
    assert all(note == ["r"] for _, note in _notes(res))


# -- Row 11: a lineage fault discards the whole pause -------------------------------------

def test_a_lineage_fault_fails_open_and_discards_both_partitions(ws, monkeypatch, capsys):
    config, m = ws
    _seed(m, "r", "Root decision R.")
    _seed(m, "n", "N amends R.", amends="r")
    _retire(m, "old-way")
    _arm(m, [{"slug": "old-way", "score": _HIGH}, {"slug": "n", "score": _NEAR[0]}])
    calls = []

    def broken(node_ids):
        calls.append(node_ids)
        raise DatabaseError("injected lineage fault")
    monkeypatch.setattr(m.store, "get_lineage_ids", broken)
    res = _draft(m, amends="r")
    assert calls and res["status"] == "created"
    assert "could not run (graph read failed)" in res["neighbor_review_unavailable"]
    assert "tombstones" not in res and "neighbors" not in res
    assert "[Warning] Neighbor review skipped for 'draft'" in capsys.readouterr().err


def test_the_pause_review_itself_returns_no_tombstone_only_pause_on_a_lineage_fault(
        ws, monkeypatch):
    """The fault arrives after the tombstones were read: the arm resets them."""
    config, m = ws
    _seed(m, "r", "Root decision R.")
    _seed(m, "n", "N amends R.", amends="r")
    _retire(m, "old-way")
    _arm(m, [{"slug": "old-way", "score": _HIGH}, {"slug": "n", "score": _NEAR[0]}])
    monkeypatch.setattr(m.store, "get_lineage_ids",
                        lambda ids: (_ for _ in ()).throw(DatabaseError("injected")))
    review = m._review_pause(_parsed_draft(m), None, None, {"amends": "r"},
                             canonical_slugs={"r": "r"})
    assert review.pause is None and review.unavailable


def _parsed_draft(m):
    from mitos.parser import ParsedEntry
    entry = ParsedEntry("decision", "draft", 1, 5)
    entry.axiom = "The draft walks a road the corpus already tried."
    entry.rejected_paths = "rej"
    entry.scope = ["s"]
    return entry


# -- Row 12: the message clause is gated on a carrier -------------------------------------

def test_the_message_clause_appears_iff_a_shown_neighbour_carries_the_note(ws):
    config, m = ws
    _seed(m, "r", "Root decision R.")
    _seed(m, "n", "N amends R.", amends="r")
    _seed(m, "u", "Unrelated U.")
    _arm(m, [{"slug": "n", "score": _HIGH}, {"slug": "u", "score": _NEAR[0]}])
    noted = _draft(m, amends="r")
    assert _CLAUSE in noted["message"]
    assert "acknowledge_neighbors=True" in noted["message"]
    # The same payload without the key renders the message this phase found.
    body = {k: v for k, v in noted.items() if k != "message"}
    body["neighbors"] = [{k: v for k, v in n.items() if k != _KEY}
                         for n in noted["neighbors"]]
    assert sync._pause_message(body) == noted["message"].replace(f" {_CLAUSE}", "", 1)
    # No carrier, no clause.
    _arm(m, [{"slug": "u", "score": _HIGH}])
    assert _CLAUSE not in _draft(m, amends="r")["message"]


def test_a_tombstone_only_pause_carries_no_clause(ws):
    config, m = ws
    _seed(m, "r", "Root decision R.")
    _retire(m, "old-way")
    _arm(m, [{"slug": "old-way", "score": _HIGH}])
    res = _draft(m, amends="r")
    assert res["neighbors"] == [] and res["tombstones"]
    assert _KEY not in res["message"]


# -- Row 13: parity, and one text line per carrier ----------------------------------------

def test_mcp_and_json_agree_and_the_text_prints_one_line_per_carrier(ws):
    config, m = ws
    _seed(m, "r", "Root decision R.")
    _seed(m, "mid", "M amends R.", amends="r")
    _seed(m, "n", "N amends M.", amends="mid")
    _seed(m, "u", "Unrelated U.")
    matches = [{"slug": "n", "score": _HIGH}, {"slug": "u", "score": _NEAR[0]},
               {"slug": "mid", "score": _NEAR[1]}]
    mcp, cli_json, text, code = _surfaces(config, matches, amends="r")
    assert code == 2
    assert [set(n) for n in mcp["neighbors"]] == [set(n) for n in cli_json["neighbors"]]
    assert mcp["neighbors"] == cli_json["neighbors"]
    assert mcp["message"] == cli_json["message"] and _CLAUSE in mcp["message"]
    assert _notes(mcp) == [("n", ["r"]), ("u", None), ("mid", ["r"])]
    assert text.count(_TEXT_LINE) == 2
    assert "mitos " not in mcp["message"]
