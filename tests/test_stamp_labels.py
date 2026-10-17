"""Phase 8a: the stamp label map.

Every response on the carrying list (vision §4.8; plan p8a §3.6) carries one
``stamp_labels`` map: for each live decision its stamps or live ends name and whose
axiom it does not already carry, the first clause of that axiom, cut by
``stamp_labels.cut_label``. One batched store read per response
(``GraphStore.get_live_axioms``), none when nothing is targeted; a fault in it costs
the map alone (``stamp_labels_unread``). The excluded surfaces keep their bare stamps,
each proven beside a carrying sibling over the same graph, so no exclusion row passes
over an empty map (plan G1).

Graphs are built through the write path (``commit_parsed_entry`` / ``record``), never
``sync``. Ranked reads are armed with canned vector hits in rank order.
"""

import ast
import inspect
import io
import json
import logging
import os
import shutil
import tempfile
from contextlib import redirect_stdout
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import patch

import pytest

import mitos.cli as cli
import mitos.mcp_server as mcp_server
from mitos import recall, stamp_labels, store as store_module, sync
from mitos.cli import cmd_init, cmd_list, cmd_show
from mitos.config import MitosConfig
from mitos.errors import DatabaseError
from mitos.parser import ParsedEntry
from mitos.protocols import GraphStoreProtocol
from mitos.recall import (RetiredPartition, compose_ranked, missing_graph_note,
                          rejection_lookup, stamp_labels_unread_clause)
from mitos.stamp_labels import (CLAUSE_BOUNDARIES, CUT_MARK, LABEL_CUT_BYTES,
                                NO_LABELS, StampLabels, cut_label, label_targets,
                                list_labels, match_labels, pause_labels, ranked_labels,
                                read_stamp_labels)
from mitos.store import GraphStore

from _point_hits import CannedVector, point_hit
from _rejection_index_double import FakeRejectionIndex
from conftest import resolve_like_main
from test_degraded_envelope import (DRIVERS as DEGRADED_DRIVERS, block, commit_text,
                                    write_buffer)
from test_neighbor_review import _arm, _seed
from test_point_identity import same_slug_lineage
from test_rejection_lookup import (_cli as _lookup_cli, _mcp as _lookup_mcp,
                                   _plant as _lookup_plant)
from test_rejection_section import FLOOR, _bind, _plant as _section_plant

QUERY = "a claim that names no slug"


def _clause(slug: str) -> str:
    """A two-clause axiom whose label is its first clause, marked."""
    return f"The {slug} rule holds; the rest of it explains why."


def _label(slug: str) -> str:
    return f"The {slug} rule holds{CUT_MARK}"


# --------------------------------------------------------------------------- #
# The leaf: the cut's trace table (input axiom → label)
# --------------------------------------------------------------------------- #

def _bound_text(n_words: int) -> str:
    return " ".join(f"word{i:02d}" for i in range(n_words))


TRACE = [
    # T1 one clause within the bound: whole, no mark.
    ("T1", "We use SQLite", "We use SQLite"),
    # T2 the first ; : — within the bound, cut at it, marked.
    ("T2a", "Keep the cache; expire hourly", f"Keep the cache{CUT_MARK}"),
    ("T2b", "The rule: expire hourly", f"The rule{CUT_MARK}"),
    ("T2c", "The cache stays — it expires hourly", f"The cache stays{CUT_MARK}"),
    # T5–T8 non-English clause marks.
    ("T5", "キャッシュを使う。毎時失効する", f"キャッシュを使う{CUT_MARK}"),
    ("T6", "使用缓存；每小时过期", f"使用缓存{CUT_MARK}"),
    ("T7", "نستخدم ذاكرة التخزين؛ تنتهي كل ساعة", f"نستخدم ذاكرة التخزين{CUT_MARK}"),
    ("T8", "हम कैश रखते हैं। यह हर घंटे समाप्त होता है", f"हम कैश रखते हैं{CUT_MARK}"),
    # Greek ano teleia, as NFC normalises it (U+00B7), and as typed (U+0387).
    ("T8g", "Κρατάμε την κρυφή μνήμη\u00b7 λήγει κάθε ώρα",
     f"Κρατάμε την κρυφή μνήμη{CUT_MARK}"),
    ("T8h", "Κρατάμε την κρυφή μνήμη\u0387 λήγει κάθε ώρα",
     f"Κρατάμε την κρυφή μνήμη{CUT_MARK}"),
    # T10 a full stop inside a token does not cut.
    ("T10", "Ship v0.19.1 with recall.py intact", "Ship v0.19.1 with recall.py intact"),
    # T11 a boundary at position 0 is ignored; the scan goes on.
    ("T11", "—Keep it; then more", f"—Keep it{CUT_MARK}"),
    # T12 newlines and whitespace runs collapse.
    ("T12", "  Keep\n\nthe   cache  ", "Keep the cache"),
    # T13 an empty or whitespace axiom yields "" (unreachable, but total).
    ("T13a", "", ""),
    ("T13b", " \n\t ", ""),
    # T14 a single sentence ending in "." stays whole, its full stop kept.
    ("T14", "We use SQLite.", "We use SQLite."),
    # T15 a terminal ideographic full stop: whole.
    ("T15", "キャッシュを使う。", "キャッシュを使う。"),
    # T16 a token full stop skipped, the later ; cuts.
    ("T16", "v0.19.1 ships X; rest", f"v0.19.1 ships X{CUT_MARK}"),
    # A spaced mark cuts only when whitespace follows it.
    ("T16b", "Use mech:created_at as the key; more", f"Use mech:created_at as the key{CUT_MARK}"),
    # The comma is no clause mark.
    ("T16c", "A, B and C hold", "A, B and C hold"),
]


@pytest.mark.parametrize("row, axiom, label", TRACE, ids=[r[0] for r in TRACE])
def test_the_cut_trace_table(row, axiom, label):
    assert cut_label(axiom) == label
    assert len(cut_label(axiom).encode("utf-8")) <= LABEL_CUT_BYTES


def test_t3_no_boundary_over_the_bound_cuts_at_a_word_and_marks_it():
    axiom = _bound_text(40)
    label = cut_label(axiom)
    assert label.endswith(CUT_MARK) and len(label.encode()) <= LABEL_CUT_BYTES
    kept = label[: -len(CUT_MARK)]
    assert axiom.startswith(kept) and axiom[len(kept)] == " "     # a whole word kept


def test_t4_a_boundary_past_the_bound_cuts_at_the_bound_and_marks_it():
    axiom = _bound_text(40) + "; and then a clause"
    label = cut_label(axiom)
    assert label.endswith(CUT_MARK) and len(label.encode()) <= LABEL_CUT_BYTES
    assert ";" not in label


def test_t9_cjk_with_no_mark_cuts_at_a_whole_code_point_inside_the_byte_bound():
    axiom = "漢" * 100                                # 300 bytes, no space, no mark
    label = cut_label(axiom)
    body = label[: -len(CUT_MARK)]
    assert label.endswith(CUT_MARK)
    assert len(label.encode("utf-8")) <= LABEL_CUT_BYTES
    assert set(body) == {"漢"} and len(body) == (LABEL_CUT_BYTES - 3) // 3
    label.encode("utf-8").decode("utf-8")              # never mid-sequence


def test_a_whole_axiom_that_fits_is_not_marked_and_one_that_does_not_is():
    fits = "x" * LABEL_CUT_BYTES
    assert cut_label(fits) == fits
    assert cut_label(fits + "y").endswith(CUT_MARK)


def test_the_boundary_set_names_every_script_the_plan_requires():
    floor = set(".;:!?—。；：！？．؛؟۔।॥։።፤\u037e\u0387")
    assert floor <= CLAUSE_BOUNDARIES
    assert "," not in CLAUSE_BOUNDARIES


# --------------------------------------------------------------------------- #
# The leaf: the target rule (§3.2) and the read's composition
# --------------------------------------------------------------------------- #

def test_named_is_the_amenders_plus_the_live_end_else_the_kill_stamps():
    entries = [
        {"slug": "w", "amended_by": ["a1"], "narrowed_by": ["n1"]},
        {"slug": "r", "superseded_by": ["mid"], "live_end": ["end"]},      # dead hop hidden
        {"slug": "s", "superseded_by": ["s2"], "corrected_by": ["c2"]},
        {"slug": "z", "superseded_by": ["zz"], "live_end": []},            # no live end
    ]
    assert label_targets(named=entries, carried=[]) == ["a1", "n1", "end", "s2", "c2"]


def test_carried_subtraction_and_spelling_compare_by_casefold_but_keep_every_spelling():
    entries = [{"slug": "w", "amended_by": ["Foo", "foo", "BAR"]},
               {"slug": "v", "amended_by": ["foo", "Straße"]}]
    assert label_targets(named=entries, carried=["bar", "STRASSE"]) == ["Foo", "foo"]


def test_the_declared_echo_and_the_sibling_note_are_never_read():
    """The not-target rules are keys the leaf never reads (§3.2)."""
    pause_neighbour = {"slug": "n", "descends_from_declared": ["d"],
                       "successor_on_pause": ["x"]}
    assert label_targets(named=[pause_neighbour, {"declared": ["d"]}], carried=[]) == []


class _AxiomStore:
    def __init__(self, axioms: Dict[str, Any], raises: Optional[BaseException] = None):
        self.axioms = axioms
        self.raises = raises
        self.calls: List[List[str]] = []

    def get_live_axioms(self, slugs):
        self.calls.append(list(slugs))
        if self.raises is not None:
            raise self.raises
        return {s: self.axioms[s] for s in slugs if s in self.axioms}


def test_nothing_targeted_reads_nothing_and_returns_no_labels():
    reader = _AxiomStore({})
    assert read_stamp_labels(reader, named=[{"slug": "a"}], carried=["a"]) is NO_LABELS
    assert reader.calls == []
    # A store with no reader at all is fine when nothing is targeted.
    assert read_stamp_labels(object(), named=[], carried=[]) is NO_LABELS


def test_t17_two_spellings_one_read_two_keys_one_value():
    reader = _AxiomStore({"Foo": _clause("foo"), "foo": _clause("foo")})
    out = read_stamp_labels(reader, named=[{"slug": "w", "amended_by": ["Foo", "foo"]}],
                            carried=[])
    assert reader.calls == [["Foo", "foo"]]
    assert out == StampLabels({"Foo": _label("foo"), "foo": _label("foo")}, False)


def test_the_labels_keep_target_order_and_omit_a_target_the_read_does_not_hold():
    reader = _AxiomStore({"b": _clause("b"), "a": _clause("a")})
    out = read_stamp_labels(reader, named=[{"slug": "w", "amended_by": ["a", "gone", "b"]}],
                            carried=[])
    assert list(out.labels) == ["a", "b"]


def test_a_fault_costs_the_map_alone_and_is_flagged():
    reader = _AxiomStore({}, raises=DatabaseError("boom"))
    out = read_stamp_labels(reader, named=[{"slug": "w", "amended_by": ["a"]}], carried=[])
    assert out == StampLabels({}, True)
    assert out.envelope_keys() == {"stamp_labels_unread": True}


@pytest.mark.parametrize("store, match", [
    (object(), "no get_live_axioms"),
    (type("S", (), {"get_live_axioms": lambda self, s: ["a"]})(), "not a dict"),
    (type("S", (), {"get_live_axioms": lambda self, s: {"a": 3}})(), "not str"),
])
def test_contract_faults_are_loud(store, match):
    with pytest.raises(TypeError, match=match):
        read_stamp_labels(store, named=[{"slug": "w", "amended_by": ["a"]}], carried=[])


def test_the_envelope_keys_are_absent_when_computed_and_empty():
    assert NO_LABELS.envelope_keys() == {}
    assert StampLabels({"a": "x"}, False).envelope_keys() == {"stamp_labels": {"a": "x"}}


def test_each_family_helper_spells_its_carried_rule():
    reader = _AxiomStore({s: _clause(s) for s in ("x", "y", "z", "w")})
    hit = {"slug": "h", "amended_by": ["x", "o"]}
    handle = {"slug": "r", "superseded_by": ["y"]}
    here = {"slug": "o", "letter": "here", "amended_by": ["z"]}
    retired = {"slug": "w", "letter": "retired", "superseded_by": ["w"]}
    out = ranked_labels(reader, window=[hit], handles=[handle], matches=[here, retired])
    # "o" is a `here` owner (carried); "w" is a retired owner's live successor (not).
    assert list(out.labels) == ["x", "y", "z", "w"]
    listed = list_labels(reader, decisions=[{"slug": "x", "state": "superseded",
                                             "superseded_by": ["y"]},
                                            {"slug": "y", "state": "active"}])
    assert listed.labels == {}                                # y is listed live
    pause = pause_labels(reader, neighbors=[{"slug": "x", "amended_by": ["z"]}],
                         tombstones=[{"slug": "t", "superseded_by": ["x"]}])
    assert list(pause.labels) == ["z"]                        # x is a shown neighbour
    lexical = match_labels(reader, [here], decisions=[{"slug": "z"}])
    assert lexical.labels == {}                               # z is a decision match


# --------------------------------------------------------------------------- #
# Row 12: the store read
# --------------------------------------------------------------------------- #

def _commit(store: GraphStore, slug: str, axiom: Optional[str] = None, *,
            kind: str = "decision", scope: Optional[List[str]] = None,
            **rels: List[str]) -> str:
    entry = ParsedEntry(kind, slug, 1, 5)
    if kind == "decision":
        entry.axiom = axiom if axiom is not None else _clause(slug)
        entry.rejected_paths = "Rejected the obvious alternative."
    else:
        entry.topic = f"Topic for {slug}"
        entry.questions_raised = [f"What about {slug}?"]
    entry.scope = list(scope or ["core"])
    for name, value in rels.items():
        setattr(entry, name, value)
    return store.commit_parsed_entry(entry).node_id


@pytest.fixture
def gstore(tmp_path) -> GraphStore:
    return GraphStore(str(tmp_path / "graph.sqlite"))


def test_the_read_matches_by_casefold_and_keys_by_the_input_spelling(gstore):
    _commit(gstore, "Straße-rule")
    assert gstore.get_live_axioms(["STRASSE-RULE", "straße-rule"]) == {
        "STRASSE-RULE": _clause("Straße-rule"), "straße-rule": _clause("Straße-rule")}


def test_the_read_is_live_only_drifted_counts_and_an_open_question_is_excluded(gstore):
    _commit(gstore, "old")
    _commit(gstore, "new", supersedes=["old"])
    drifted = _commit(gstore, "drifty")
    conn = gstore._get_connection()
    with conn:
        conn.execute("INSERT INTO signals (node_id, signal_type, created_at) "
                     "VALUES (?, 'drifted', '2026-10-17T00:00:00.000000+00:00')", (drifted,))
    conn.close()
    _commit(gstore, "oq-slug", kind="open_question")
    assert gstore.get_node_states([drifted]) == {drifted: "drifted"}
    assert gstore.get_live_axioms(["old", "new", "drifty", "oq-slug"]) == {
        "new": _clause("new"), "drifty": _clause("drifty")}


def test_the_read_chunks_past_the_bind_bound(gstore, monkeypatch):
    for i in range(7):
        _commit(gstore, f"d{i}")
    monkeypatch.setattr(store_module, "_POINT_ID_CHUNK", 3)
    calls = []
    original = gstore._get_connection

    def opener():
        conn = original()
        calls.append(conn)
        return conn
    monkeypatch.setattr(gstore, "_get_connection", opener)
    out = gstore.get_live_axioms([f"d{i}" for i in range(7)] + ["missing"])
    assert list(out) == [f"d{i}" for i in range(7)]
    assert len(calls) == 1                                    # one connection, three chunks


def test_an_empty_read_opens_no_connection(gstore, monkeypatch):
    monkeypatch.setattr(gstore, "_get_connection",
                        lambda: pytest.fail("a connection was opened"))
    assert gstore.get_live_axioms([]) == {}


def test_a_broken_read_raises_a_typed_database_error(tmp_path):
    path = tmp_path / "graph.sqlite"
    store = GraphStore(str(path))
    conn = store._get_connection()
    with conn:
        conn.execute("DROP TABLE edges")
    conn.close()
    with pytest.raises(DatabaseError, match="live axioms"):
        store.get_live_axioms(["x"])


def test_an_mi13_breach_omits_that_slug_and_warns_without_costing_the_rest(gstore, caplog):
    _commit(gstore, "twin")
    other = _commit(gstore, "elsewhere")
    _commit(gstore, "fine")
    conn = gstore._get_connection()
    with conn:                                                # forced out of band
        conn.execute("UPDATE nodes SET slug = 'Twin', slug_casefold = 'twin' WHERE id = ?",
                     (other,))
    conn.close()
    with caplog.at_level(logging.WARNING, logger="mitos.store"):
        out = gstore.get_live_axioms(["twin", "fine"])
    assert out == {"fine": _clause("fine")}
    assert any("'twin'" in r.getMessage() and "MI-13" in r.getMessage()
               for r in caplog.records)


# --------------------------------------------------------------------------- #
# Row 21: agreement rows
# --------------------------------------------------------------------------- #

def test_the_leafs_live_states_are_the_stores():
    assert stamp_labels.LIVE_STATES == store_module.LIVE_STATES


def test_the_read_is_on_the_graph_protocol_with_the_stores_signature():
    assert "get_live_axioms" in GraphStoreProtocol.__dict__
    proto = GraphStoreProtocol.get_live_axioms
    real = GraphStore.get_live_axioms
    assert (list(inspect.signature(proto).parameters)
            == list(inspect.signature(real).parameters))


def _chain_handles(store: GraphStore) -> List[Dict[str, Any]]:
    """Handles over 2f's chain fixtures and 7a's tombstone shapes (a two- and a
    three-deep chain, a mid hop, a correction), built by the one constructor from the
    store's own reads."""
    from mitos.display import retired_handle
    _commit(store, "c0")
    _commit(store, "c1", supersedes=["c0"])
    _commit(store, "c2", supersedes=["c1"])                   # c0 ← c1 ← c2 (live)
    _commit(store, "k0")
    _commit(store, "k1", corrects=["k0"])                     # corrected
    _commit(store, "y0")
    _commit(store, "y1", supersedes=["y0"])
    _commit(store, "y2", supersedes=["y1"])
    _commit(store, "y3", supersedes=["y2"])                   # three deep
    ids = [store.resolve_slug(s)[0] for s in ("c0", "c1", "k0", "y0", "y1")]
    nodes = {n["id"]: n for n in (store.get_node(i) for i in ids)}
    for node in nodes.values():
        node["computed_state"] = store.get_node_state(node["id"])
    stamps = store.get_modifiers_map(ids)
    live_ends = store.get_live_ends(ids)
    return [retired_handle(nodes[i], score=0.9, stamps=stamps, live_ends=live_ends)
            for i in ids]


def test_the_per_handle_targets_agree_with_the_pauses_successor_rule(gstore):
    handles = _chain_handles(gstore)
    assert any("live_end" in h for h in handles)               # the chain case is in
    for handle in handles:
        assert (set(label_targets(named=[handle], carried=[]))
                == set(sync._live_successors(handle))), handle


# --------------------------------------------------------------------------- #
# The surfaces: workspace, drivers and spies
# --------------------------------------------------------------------------- #

@pytest.fixture
def lws(monkeypatch) -> Iterator[MitosConfig]:
    """A fresh offline workspace, configured the way ``main()`` builds it."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    tmp = tempfile.mkdtemp()
    with redirect_stdout(io.StringIO()):
        cmd_init(MitosConfig(tmp))
    yield resolve_like_main(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


class _Spy:
    """Forwards to a real store; records ``get_live_axioms`` calls, or faults them."""

    def __init__(self, inner: Any, raises: Optional[BaseException] = None) -> None:
        self._inner = inner
        self._raises = raises
        self.label_reads: List[List[str]] = []

    def get_live_axioms(self, slugs):
        self.label_reads.append(list(slugs))
        if self._raises is not None:
            raise self._raises
        return self._inner.get_live_axioms(slugs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _faulting(store: Any) -> _Spy:
    return _Spy(store, raises=DatabaseError("injected label fault"))


HEADING = "Stamped slugs, labelled:"


def _text_labels(out: str) -> Dict[str, str]:
    """The pairs the CLI text label block prints, or {} when it printed none."""
    if HEADING not in out:
        return {}
    pairs: Dict[str, str] = {}
    for line in out.split(f"{HEADING}\n", 1)[1].splitlines():
        if not line.startswith("  "):
            break
        slug, label = line[2:].split(" — ", 1)
        pairs[slug] = label
    return pairs


def _ranked(driver: str, config: MitosConfig, hits: List[Dict[str, Any]], **kw: Any) -> Any:
    """A healthy ranked read on one of the six drivers (MCP/CLI × surface/query × enc)."""
    return DEGRADED_DRIVERS[driver][0](config, QUERY, cause="no-providers",
                                       vector=CannedVector(hits), **kw)


def _is_text(driver: str) -> bool:
    return DEGRADED_DRIVERS[driver][3] == "text"


def _surface_of(driver: str) -> str:
    return DEGRADED_DRIVERS[driver][2]


def _labels_of(driver: str, resp: Any) -> Dict[str, str]:
    return _text_labels(resp) if _is_text(driver) else resp.get("stamp_labels", {})


RANKED = list(DEGRADED_DRIVERS)
RANKED_JSON = [d for d in RANKED if not _is_text(d)]


def _ranked_world(config: MitosConfig) -> None:
    """a (live, amended by a-amend), r ← r-next, c ← c-mid ← c-end."""
    store = GraphStore(config.db_path)
    _commit(store, "a")
    _commit(store, "a-amend", amends=["a"])
    _commit(store, "r")
    _commit(store, "r-next", supersedes=["r"])
    _commit(store, "c")
    _commit(store, "c-mid", supersedes=["c"])
    _commit(store, "c-end", supersedes=["c-mid"])


# Retired hits ranked above the live one, so both are partition members.
_HITS = [{"slug": "r", "score": 0.9}, {"slug": "c", "score": 0.88},
         {"slug": "a", "score": 0.86}]
_RANKED_LABELS = {"a-amend": _label("a-amend"), "r-next": _label("r-next"),
                  "c-end": _label("c-end")}


# --------------------------------------------------------------------------- #
# Row 1: the map on a ranked read, both verbs, both surfaces, every encoding
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", RANKED)
def test_a_ranked_read_labels_its_amenders_and_live_ends_never_a_dead_hop(lws, driver):
    _ranked_world(lws)
    resp = _ranked(driver, lws, _HITS)
    labels = _labels_of(driver, resp)
    assert labels == _RANKED_LABELS
    assert "c-mid" not in labels and "a" not in labels
    if not _is_text(driver):
        assert list(labels) == ["a-amend", "r-next", "c-end"]     # response order


@pytest.mark.parametrize("driver", RANKED)
def test_the_blackout_labels_every_all_superseded_successor(lws, driver):
    _ranked_world(lws)
    resp = _ranked(driver, lws, _HITS[:2])
    if not _is_text(driver):
        assert [h["slug"] for h in resp["all_superseded"]] == ["r", "c"]
    assert _labels_of(driver, resp) == {"r-next": _label("r-next"),
                                        "c-end": _label("c-end")}


# --------------------------------------------------------------------------- #
# Row 6: one map at the envelope; entries carry none; stamps stay lists of slugs
# --------------------------------------------------------------------------- #

def _entries(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for key in ("active_decisions", "matches", "retired_partition", "all_superseded",
                "rejection_matches", "decisions", "neighbors", "tombstones"):
        out.extend(resp.get(key) or [])
    return out


def _assert_one_map(resp: Dict[str, Any]) -> None:
    assert isinstance(resp["stamp_labels"], dict)
    assert all(isinstance(v, str) for v in resp["stamp_labels"].values())
    for entry in _entries(resp):
        assert "stamp_labels" not in entry and "stamp_labels_unread" not in entry
        for key in ("amended_by", "narrowed_by", "superseded_by", "corrected_by",
                    "live_end"):
            if key in entry:
                assert isinstance(entry[key], list)
                assert all(isinstance(s, str) for s in entry[key])


@pytest.mark.parametrize("driver", RANKED_JSON)
def test_the_ranked_map_is_one_object_at_the_envelope(lws, driver):
    _ranked_world(lws)
    _assert_one_map(_ranked(driver, lws, _HITS))


# --------------------------------------------------------------------------- #
# Row 7: parity — MCP and --json agree on keys and labels; text lists the same pairs
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("verb", ["surface", "query"])
def test_ranked_parity_across_the_three_encodings(lws, verb):
    _ranked_world(lws)
    mcp = _ranked(f"mcp-{verb}", lws, _HITS)
    cli_json = _ranked(f"cli-{verb}-json", lws, _HITS)
    text = _ranked(f"cli-{verb}-text", lws, _HITS)
    assert set(mcp) == set(cli_json)
    assert mcp["stamp_labels"] == cli_json["stamp_labels"] == _text_labels(text)


# --------------------------------------------------------------------------- #
# Row 9: a carried decision is never labelled again
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", RANKED)
def test_a_window_hit_amended_by_another_window_hit_is_not_labelled(lws, driver):
    store = GraphStore(lws.db_path)
    _commit(store, "b")
    _commit(store, "b2", amends=["b"])
    _commit(store, "b3", amends=["b"])
    resp = _ranked(driver, lws, [{"slug": "b", "score": 0.9}, {"slug": "b2", "score": 0.85}])
    assert _labels_of(driver, resp) == {"b3": _label("b3")}


def _rejection_world(config: MitosConfig, index: FakeRejectionIndex) -> None:
    """w (window) is amended by o, which shows only as a `here` rejection owner."""
    store = GraphStore(config.db_path)
    _commit(store, "w")
    owner = _commit(store, "o", amends=["w"])
    _section_plant(index, owner, "Rejected the obvious alternative.", {1: round(FLOOR + 0.1, 4)})


def _section_call(driver: str, config: MitosConfig, hits) -> Dict[str, Any]:
    from test_rejection_section import DRIVERS as SECTION_DRIVERS
    return SECTION_DRIVERS[driver][0](config, hits)


@pytest.mark.parametrize("driver", ["mcp-surface", "mcp-query", "cli-surface", "cli-query"])
def test_a_here_owner_named_by_a_window_stamp_is_carried(lws, driver, monkeypatch):
    index = FakeRejectionIndex(present=True)
    _rejection_world(lws, index)
    _bind(monkeypatch, index)
    resp = _section_call(driver, lws, [{"slug": "w", "score": 0.9}])
    assert [m["slug"] for m in resp["rejection_matches"]] == ["o"]
    assert resp["rejection_matches"][0]["letter"] == "here"
    assert resp["active_decisions" if "surface" in driver else "matches"][0]["amended_by"] == ["o"]
    assert "stamp_labels" not in resp


# --------------------------------------------------------------------------- #
# Row 10: the same-slug case — a retired handle's successor shares its slug
# --------------------------------------------------------------------------- #

def test_a_same_slug_successor_labels_the_retired_handle_and_the_listed_retired(lws):
    store = GraphStore(lws.db_path)
    retired, successor = same_slug_lineage(store)
    live_label = cut_label("We use SQLite for local WAL reads.")
    resp = DEGRADED_DRIVERS["mcp-surface"][0](lws, QUERY, cause="no-providers",
                                              vector=CannedVector([point_hit(retired, .9)]))
    assert [h["id"] for h in resp["all_superseded"]] == [retired]
    assert resp["stamp_labels"] == {"database": live_label}
    listed = json.loads(mcp_server.list_decisions(state="superseded",
                                                  project=lws.workspace_dir))
    assert [d["slug"] for d in listed["decisions"]] == ["database"]
    assert listed["stamp_labels"] == {"database": live_label}


def test_g2_a_listed_retired_decisions_unwalked_kill_stamp_labels_what_its_slug_names(lws):
    """G2: under ``state="all"`` a retired decision's immediate killer may itself be
    retired; the list reads no live ends, so that hop is named and the live filter
    drops it — nothing is labelled for it."""
    store = GraphStore(lws.db_path)
    _commit(store, "g0")
    _commit(store, "g1", supersedes=["g0"])
    _commit(store, "g2", supersedes=["g1"])
    listed = json.loads(mcp_server.list_decisions(state="superseded",
                                                  project=lws.workspace_dir))
    assert {d["slug"]: d.get("superseded_by") for d in listed["decisions"]} == {
        "g0": ["g1"], "g1": ["g2"]}
    # g1 is named by g0's unwalked stamp and is retired: never labelled. g2 is live.
    assert listed["stamp_labels"] == {"g2": _label("g2")}


# --------------------------------------------------------------------------- #
# Rows 11 and 13: a dead hop is never asked about; one batched read per response
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", RANKED)
def test_one_label_read_per_ranked_response_and_the_dead_hop_is_never_asked(lws, driver):
    _ranked_world(lws)
    spies: List[_Spy] = []

    def wrap(store):
        spies.append(_Spy(store))
        return spies[-1]
    _ranked(driver, lws, _HITS, wrap=wrap)
    (spy,) = spies
    assert spy.label_reads == [["a-amend", "r-next", "c-end"]]


@pytest.mark.parametrize("driver", RANKED)
def test_an_unstamped_ranked_response_reads_no_labels(lws, driver):
    store = GraphStore(lws.db_path)
    _commit(store, "plain")
    spies: List[_Spy] = []

    def wrap(inner):
        spies.append(_Spy(inner))
        return spies[-1]
    resp = _ranked(driver, lws, [{"slug": "plain", "score": 0.9}], wrap=wrap)
    assert spies[0].label_reads == []
    assert _labels_of(driver, resp) == {}
    if not _is_text(driver):
        assert "stamp_labels" not in resp and "stamp_labels_unread" not in resp


# --------------------------------------------------------------------------- #
# Row 14a: a label fault costs the map alone, stated
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("driver", RANKED)
def test_a_ranked_label_fault_costs_the_map_alone(lws, driver):
    _ranked_world(lws)
    healthy = _ranked(driver, lws, _HITS)
    faulty = _ranked(driver, lws, _HITS, wrap=_faulting)
    clause = stamp_labels_unread_clause(surface=_surface_of(driver), config=lws)
    if _is_text(driver):
        assert HEADING in healthy and HEADING not in faulty
        assert clause in faulty and clause not in healthy
        return
    expected = {k: v for k, v in healthy.items() if k != "stamp_labels"}
    expected["stamp_labels_unread"] = True
    expected["note"] = f"{healthy['note']} {clause}"
    assert faulty == expected


# --------------------------------------------------------------------------- #
# Row 2: the field-scoped lookup, healthy and lexical (rows 13b, 14b)
# --------------------------------------------------------------------------- #

_LOOKUP_REJECTED = "Postgres as the webhook queue — too heavy for one consumer."


def _lookup_world(config: MitosConfig) -> Dict[str, str]:
    """ll (live, amended by ll-amend) and lr ← lr-next, both owning a rejected option."""
    blocks = [block("ll-amend", _clause("ll-amend"), amends="ll"),
              block("ll", _clause("ll"), rejected=_LOOKUP_REJECTED),
              block("lr-next", _clause("lr-next"), supersedes="lr"),
              block("lr", _clause("lr"), rejected=_LOOKUP_REJECTED)]
    return commit_text(config, write_buffer(config, blocks))


@pytest.mark.parametrize("driver", ["mcp", "cli"])
def test_the_lookup_labels_a_retired_owners_live_end_and_a_live_owners_amender(lws, driver):
    ids = _lookup_world(lws)
    index = FakeRejectionIndex(present=True)
    _lookup_plant(index, lws, ids["ll"], {1: 0.95})
    _lookup_plant(index, lws, ids["lr"], {1: 0.93})
    call = _lookup_mcp if driver == "mcp" else _lookup_cli
    resp = call(lws, index)
    assert [(m["slug"], m["letter"]) for m in resp["rejection_matches"]] == [
        ("ll", "here"), ("lr", "retired")]
    assert resp["stamp_labels"] == {"ll-amend": _label("ll-amend"),
                                    "lr-next": _label("lr-next")}
    _assert_one_map(resp)
    text = _lookup_cli(lws, index, as_json=False) if driver == "cli" else None
    if text is not None:
        assert _text_labels(text) == resp["stamp_labels"]


def test_the_lookup_reads_once_and_its_fault_costs_the_map_alone(lws):
    ids = _lookup_world(lws)
    index = FakeRejectionIndex(present=True)
    _lookup_plant(index, lws, ids["ll"], {1: 0.95})
    _lookup_plant(index, lws, ids["lr"], {1: 0.93})
    spy = _Spy(GraphStore(lws.db_path, read_only=True))
    healthy = _lookup_mcp(lws, index, store=spy)
    assert spy.label_reads == [["ll-amend", "lr-next"]]
    faulty = _lookup_mcp(lws, index, store=_faulting(GraphStore(lws.db_path, read_only=True)))
    clause = stamp_labels_unread_clause(surface="mcp", config=lws)
    expected = {k: v for k, v in healthy.items() if k != "stamp_labels"}
    expected["stamp_labels_unread"] = True
    expected["note"] = f"{healthy['note']} {clause}"
    assert faulty == expected


@pytest.mark.parametrize("driver", ["mcp", "cli"])
def test_the_lexical_lookup_labels_a_live_owners_amender(lws, driver):
    _lookup_world(lws)
    call = _lookup_mcp if driver == "mcp" else _lookup_cli
    resp = call(lws, None, query="postgres webhook queue")
    assert resp["degraded"] == "lexical"
    assert [m["slug"] for m in resp["rejection_matches"]] == ["ll"]      # retired dropped
    assert resp["stamp_labels"] == {"ll-amend": _label("ll-amend")}
    assert "stamp_labels" not in resp["not_computed"]


# --------------------------------------------------------------------------- #
# Row 3: degraded exits (rows 13c, 14c, 15)
# --------------------------------------------------------------------------- #

STORE_CAUSES = ("no-providers", "missing-collection", "mid-query")


def _degraded_world(config: MitosConfig) -> None:
    blocks = [block("cache-amend", _clause("cache-amend"), scope="other",
                    amends="cache-strategy"),
              block("cache-strategy", "Use a write-through cache.", scope="caching")]
    commit_text(config, write_buffer(config, blocks))


@pytest.mark.parametrize("cause", STORE_CAUSES)
@pytest.mark.parametrize("driver", RANKED)
def test_every_lexical_exit_labels_its_matches_stamps(lws, driver, cause):
    _degraded_world(lws)
    resp = DEGRADED_DRIVERS[driver][0](lws, "write-through", cause=cause)
    assert _labels_of(driver, resp) == {"cache-amend": _label("cache-amend")}
    if not _is_text(driver):
        assert resp["degraded"] == "lexical"
        assert "stamp_labels" not in resp["not_computed"]
        _assert_one_map(resp)


@pytest.mark.parametrize("driver", [d for d in RANKED if DEGRADED_DRIVERS[d][1] == "surface"])
def test_the_scope_dump_labels_its_decisions_stamps(lws, driver):
    _degraded_world(lws)
    resp = DEGRADED_DRIVERS[driver][0](lws, "write-through", cause="no-providers",
                                       scope="caching")
    assert _labels_of(driver, resp) == {"cache-amend": _label("cache-amend")}
    if not _is_text(driver):
        assert "degraded" not in resp                    # the dump, not the lexical exit
        assert "stamp_labels" not in resp["not_computed"]


def test_3b_the_dump_labels_even_when_its_section_could_not_be_searched(lws):
    _degraded_world(lws)
    outside = [block("queue-x", "Queue writes.", scope="queue",
                     rejected="1. A write-through queue — too slow.")]
    commit_text(lws, write_buffer(lws, outside + [
        block("cache-amend", _clause("cache-amend"), scope="other", amends="cache-strategy"),
        block("cache-strategy", "Use a write-through cache.", scope="caching")]))

    class _StatesFault(_Spy):
        def get_node_states(self, node_ids):
            raise DatabaseError("injected states fault")
    resp = DEGRADED_DRIVERS["mcp-surface"][0](lws, "write-through", cause="no-providers",
                                              scope="caching", wrap=_StatesFault)
    assert resp["rejection_section"] == "unavailable"
    assert resp["stamp_labels"] == {"cache-amend": _label("cache-amend")}


@pytest.mark.parametrize("scope", [None, "caching"])
def test_a_degraded_response_reads_its_labels_once(lws, scope):
    _degraded_world(lws)
    spies: List[_Spy] = []

    def wrap(inner):
        spies.append(_Spy(inner))
        return spies[-1]
    kw = {"scope": scope} if scope else {}
    resp = DEGRADED_DRIVERS["mcp-surface"][0](lws, "write-through", cause="no-providers",
                                              wrap=wrap, **kw)
    assert ("degraded" in resp) is (scope is None)            # lexical, else the dump
    assert spies[0].label_reads == [["cache-amend"]]


@pytest.mark.parametrize("driver", ["mcp-surface", "mcp-query"])
def test_a_degraded_label_fault_costs_the_map_alone(lws, driver):
    _degraded_world(lws)
    healthy = DEGRADED_DRIVERS[driver][0](lws, "write-through", cause="no-providers")
    faulty = DEGRADED_DRIVERS[driver][0](lws, "write-through", cause="no-providers",
                                         wrap=_faulting)
    clause = stamp_labels_unread_clause(surface="mcp", config=lws)
    expected = {k: v for k, v in healthy.items() if k != "stamp_labels"}
    expected["stamp_labels_unread"] = True
    expected["note"] = f"{healthy['note']} {clause}"
    assert faulty == expected


@pytest.mark.parametrize("driver", RANKED_JSON)
def test_15_an_unread_graph_names_the_labels_as_not_computed(lws, driver):
    _degraded_world(lws)
    resp = DEGRADED_DRIVERS[driver][0](lws, "write-through", cause="pre-v1a")
    assert resp["not_computed"][-1] == "stamp_labels"
    assert recall.degraded_omission_clause(labels_computed=False) in resp["note"]
    assert "Stamp labels were not computed" in resp["note"]
    assert "stamp_labels" not in resp and "stamp_labels_unread" not in resp


@pytest.mark.parametrize("driver", ["mcp", "cli"])
def test_15_the_lexical_lookup_over_an_unread_graph_names_the_labels(lws, driver):
    _lookup_world(lws)
    call = _lookup_mcp if driver == "mcp" else _lookup_cli
    resp = call(lws, None, query="postgres webhook queue",
                components_raise=DatabaseError("This graph predates the V1a schema."))
    assert resp["not_computed"][-1] == "stamp_labels"
    assert "Stamp labels were not computed" in resp["note"]
    assert "stamp_labels" not in resp


def _oq_world(config: MitosConfig) -> GraphStore:
    store = GraphStore(config.db_path)
    entry = ParsedEntry("open_question", "oq-one", 1, 5)
    entry.topic, entry.questions_raised, entry.scope = "Topic one", ["Q one?"], ["q"]
    store.commit_parsed_entry(entry)
    entry = ParsedEntry("open_question", "oq-two", 1, 5)
    entry.topic, entry.questions_raised, entry.scope = "Topic two", ["Q two?"], ["other"]
    entry.amends = ["oq-one"]
    store.commit_parsed_entry(entry)
    return store


def test_3c_the_degraded_surface_text_prints_each_open_questions_marker(lws):
    _oq_world(lws)
    out = DEGRADED_DRIVERS["cli-surface-text"][0](lws, "topic", cause="no-providers",
                                                  scope="q")
    lines = out.splitlines()
    at = lines.index("[open question in scope] oq-one")
    assert lines[at + 1].strip() == "⚠ amended by: oq-two"


@pytest.mark.parametrize("cause", ["healthy", "no-providers"])
@pytest.mark.parametrize("driver", ["mcp-surface", "cli-surface-json"])
def test_w1_an_open_questions_stamps_name_open_questions_and_read_nothing(lws, driver, cause):
    """Scout W1: same-kind edges, so an open question's amender is an open question,
    which no decision read can label. The scoped scan's stamps cost no read."""
    store = _oq_world(lws)
    _commit(store, "plain")
    spies: List[_Spy] = []

    def wrap(inner):
        spies.append(_Spy(inner))
        return spies[-1]
    kw: Dict[str, Any] = {"scope": "q", "wrap": wrap}
    if cause == "healthy":
        resp = _ranked(driver, lws, [{"slug": "plain", "score": 0.9}], **kw)
    else:
        resp = DEGRADED_DRIVERS[driver][0](lws, "topic", cause=cause, **kw)
    assert resp["open_questions"][0]["amended_by"] == ["oq-two"]
    assert "stamp_labels" not in resp
    assert spies[0].label_reads == []


# --------------------------------------------------------------------------- #
# Row 4 and row 8: the listing (rows 13d, 14d, 16a)
# --------------------------------------------------------------------------- #

def _list_world(config: MitosConfig) -> None:
    """la in scope core, amended by la-amend in another scope (so never listed)."""
    store = GraphStore(config.db_path)
    _commit(store, "la")
    _commit(store, "la-amend", scope=["elsewhere"], amends=["la"])


def _mcp_list(config: MitosConfig, store: Any = None, **kw: Any) -> Dict[str, Any]:
    if store is None:
        return json.loads(mcp_server.list_decisions(project=config.workspace_dir, **kw))
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, None, None)):
        return json.loads(mcp_server.list_decisions(project=config.workspace_dir, **kw))


def _cli_list(config: MitosConfig, *, as_json: bool, **kw: Any) -> Any:
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_list(config, as_json=as_json, **kw)
    return json.loads(buf.getvalue()) if as_json else buf.getvalue()


@pytest.mark.parametrize("brief", [False, True])
def test_the_listing_labels_an_unlisted_amender_on_every_encoding(lws, brief):
    _list_world(lws)
    expected = {"la-amend": _label("la-amend")}
    mcp = _mcp_list(lws, scope="core", brief=brief)
    cli_json = _cli_list(lws, as_json=True, scope="core", brief=brief)
    text = _cli_list(lws, as_json=False, scope="core", brief=brief)
    assert [d["slug"] for d in mcp["decisions"]] == ["la"]
    assert mcp["stamp_labels"] == cli_json["stamp_labels"] == _text_labels(text) == expected
    assert set(mcp) == set(cli_json)
    keys = list(mcp)
    assert keys.index("state") < keys.index("stamp_labels") < keys.index("project")
    _assert_one_map(mcp)


def test_8_a_listing_whose_members_amend_one_another_re_quotes_nothing(lws):
    store = GraphStore(lws.db_path)
    _commit(store, "ma")
    _commit(store, "mc", amends=["ma"])
    _commit(store, "mb", amends=["mc"])
    spy = _Spy(GraphStore(lws.db_path))
    listed = _mcp_list(lws, store=spy, scope="core")
    assert {d["slug"] for d in listed["decisions"]} == {"ma", "mb", "mc"}
    assert "stamp_labels" not in listed and spy.label_reads == []
    _commit(store, "md", scope=["elsewhere"], amends=["ma"])
    spy = _Spy(GraphStore(lws.db_path))
    listed = _mcp_list(lws, store=spy, scope="core")
    assert listed["stamp_labels"] == {"md": _label("md")}
    assert spy.label_reads == [["md"]]


def test_a_list_label_fault_costs_the_map_alone(lws):
    _list_world(lws)
    healthy = _mcp_list(lws, scope="core")
    faulty = _mcp_list(lws, store=_faulting(GraphStore(lws.db_path)), scope="core")
    expected = {k: v for k, v in healthy.items() if k != "stamp_labels"}
    expected["stamp_labels_unread"] = True
    assert faulty == expected


def test_a_cli_list_label_fault_prints_one_line(lws):
    _list_world(lws)
    real = GraphStore
    with patch.object(cli, "GraphStore", side_effect=lambda *a, **k: _faulting(real(*a, **k))):
        text = _cli_list(lws, as_json=False, scope="core")
        payload = _cli_list(lws, as_json=True, scope="core")
    assert payload["stamp_labels_unread"] is True and "stamp_labels" not in payload
    assert stamp_labels_unread_clause(surface="cli", config=lws) in text
    assert HEADING not in text


def test_16a_the_one_line_tier_keeps_bare_stamps_beside_the_default_that_labels(lws):
    _list_world(lws)
    assert _mcp_list(lws, scope="core")["stamp_labels"]                   # the sibling
    mcp = _mcp_list(lws, scope="core", oneline=True)
    cli_json = _cli_list(lws, as_json=True, scope="core", oneline=True)
    text = _cli_list(lws, as_json=False, scope="core", oneline=True)
    for resp in (mcp, cli_json):
        assert resp["decisions"][0]["amended_by"] == ["la-amend"]
        assert "stamp_labels" not in resp and "stamp_labels_unread" not in resp
    assert "amended by: la-amend" in text and HEADING not in text


# --------------------------------------------------------------------------- #
# Row 5: the record pause (rows 13e, 14e, 16b, 16c, 16d)
# --------------------------------------------------------------------------- #

@pytest.fixture
def rws(lws):
    from mitos.sync import MitosSyncManager
    return lws, MitosSyncManager(lws)


def _pause_world(m) -> List[Dict[str, Any]]:
    """nb (live, amended by nb-amend), old ← new, old2 ← nb. Returns the gather."""
    _seed(m, "old2", _clause("old2"))
    _seed(m, "nb", _clause("nb"), supersedes="old2")
    _seed(m, "nb-amend", _clause("nb-amend"), amends="nb")
    _seed(m, "old", _clause("old"))
    _seed(m, "new", _clause("new"), supersedes="old")
    return [{"slug": "nb", "score": 0.9}, {"slug": "old", "score": 0.92},
            {"slug": "old2", "score": 0.91}]


_PAUSE_LABELS = {"nb-amend": _label("nb-amend"), "new": _label("new")}


def _draft(m, **kw):
    return m.record_decision_entry("A draft that walks near nb and old.", "rej", ["s"],
                                   slug="draft", **kw)


def test_the_pause_labels_a_neighbours_amender_and_a_tombstones_successor(rws):
    config, m = rws
    _arm(m, _pause_world(m))
    res = _draft(m)
    assert res["status"] == "needs_review"
    assert [n["slug"] for n in res["neighbors"]] == ["nb"]
    assert [t["slug"] for t in res["tombstones"]] == ["old", "old2"]
    assert res["tombstones"][1]["successor_on_pause"]                    # nb is shown
    assert res["stamp_labels"] == _PAUSE_LABELS                          # nb is carried
    _assert_one_map(res)
    keys = list(res)
    assert keys.index("tombstones") < keys.index("stamp_labels") < keys.index("message")


def test_the_pause_carries_the_map_on_every_surface(rws):
    from test_pause_tombstones import _surfaces
    config, m = rws
    matches = _pause_world(m)
    mcp, cli_json, text, code = _surfaces(config, matches)
    assert code == 2
    assert mcp["stamp_labels"] == cli_json["stamp_labels"] == _PAUSE_LABELS
    assert _text_labels(text) == _PAUSE_LABELS
    assert text.index("Tombstones —") < text.index(HEADING)


def test_the_pause_reads_its_labels_once(rws):
    config, m = rws
    _arm(m, _pause_world(m))
    spy = _Spy(m.store)
    m.store = spy
    _draft(m)
    assert spy.label_reads == [["nb-amend", "new"]]


def test_a_pause_label_fault_costs_the_map_alone_and_keeps_the_pause(rws):
    config, m = rws
    _arm(m, _pause_world(m))
    healthy = _draft(m)
    m.store = _faulting(m.store)
    faulty = _draft(m)
    clause = stamp_labels_unread_clause(surface="mcp", config=None)
    assert faulty["neighbors"] == healthy["neighbors"]
    assert faulty["tombstones"] == healthy["tombstones"]
    assert "neighbor_review_unavailable" not in faulty
    assert faulty["stamp_labels_unread"] is True and "stamp_labels" not in faulty
    assert clause in faulty["message"]
    expected = {k: v for k, v in healthy.items() if k != "stamp_labels"}
    expected["stamp_labels_unread"] = True
    expected["message"] = faulty["message"]
    assert faulty == expected
    assert faulty["message"].replace(f" {clause}", "", 1) == healthy["message"]


def test_a_pause_label_fault_prints_one_line_on_the_cli(rws):
    from test_pause_tombstones import _surfaces
    config, m = rws
    matches = _pause_world(m)
    real = sync.pause_labels

    def faulting_labels(store, **kw):
        return real(_faulting(store), **kw)
    with patch.object(sync, "pause_labels", faulting_labels):
        mcp, cli_json, text, code = _surfaces(config, matches)
    assert cli_json["stamp_labels_unread"] is True
    assert stamp_labels_unread_clause(surface="cli", config=config) in text
    assert HEADING not in text


def test_16b_the_created_receipts_edges_keep_bare_stamps(rws):
    config, m = rws
    matches = _pause_world(m)
    receipt = m.record_decision_entry("A draft that amends nb.", "rej", ["s"],
                                      slug="amender", amends="nb")
    assert receipt["status"] == "created"
    (edge,) = receipt["edges_created"]
    assert "nb-amend" in edge["target_stamps"]["amended_by"]
    assert "stamp_labels" not in receipt and "stamp_labels_unread" not in receipt
    _arm(m, matches)                                                     # the sibling
    assert "nb-amend" in _draft(m)["stamp_labels"]


def test_16c_the_acknowledged_lists_keep_bare_stamps_beside_the_pause_that_labels(rws):
    config, m = rws
    _arm(m, _pause_world(m))
    assert _draft(m)["stamp_labels"] == _PAUSE_LABELS                    # the sibling
    receipt = _draft(m, acknowledge_neighbors=True)
    assert receipt["status"] == "created"
    assert receipt["acknowledged_neighbors"] == ["nb"]
    assert [t["slug"] for t in receipt["acknowledged_tombstones"]] == ["old", "old2"]
    assert "stamp_labels" not in receipt and "stamp_labels_unread" not in receipt


def test_16d_a_retired_exists_handle_keeps_bare_stamps_beside_the_blackout(rws):
    config, m = rws
    _pause_world(m)
    receipt = m.record_decision_entry(_clause("old"), "Rejected the obvious alternative.",
                                      ["s"], slug="old-again")
    assert receipt["status"] == "exists" and receipt["superseded_by"] == ["new"]
    assert "stamp_labels" not in receipt and "stamp_labels_unread" not in receipt
    resp = _ranked("mcp-surface", config, [{"slug": "old", "score": 0.9}])
    assert resp["stamp_labels"] == {"new": _label("new")}              # the sibling


# --------------------------------------------------------------------------- #
# Rows 16e–16g: the by-handle reads and the rendered files keep bare stamps
# --------------------------------------------------------------------------- #

def test_16e_show_keeps_bare_stamps_beside_a_ranked_read_that_labels(lws):
    _ranked_world(lws)
    assert _ranked("mcp-surface", lws, [{"slug": "a", "score": 0.9}])["stamp_labels"] == {
        "a-amend": _label("a-amend")}
    shown = json.loads(mcp_server.show_node("a", project=lws.workspace_dir))
    assert shown["amended_by"] == ["a-amend"] and "stamp_labels" not in shown
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_show(lws, "a", as_json=True)
    assert "stamp_labels" not in json.loads(buf.getvalue())
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_show(lws, "a")
    assert "a-amend" in buf.getvalue() and HEADING not in buf.getvalue()


def test_16f_the_exact_slug_hit_keeps_bare_stamps_beside_the_ranked_query(lws):
    _ranked_world(lws)
    store = GraphStore(lws.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, None, None)):
        hit = json.loads(mcp_server.query_decisions("a", project=lws.workspace_dir))
    assert hit["slug"] == "a" and hit["amended_by"] == ["a-amend"]
    assert "stamp_labels" not in hit
    ranked = _ranked("mcp-query", lws, [{"slug": "a", "score": 0.9}])
    assert ranked["stamp_labels"] == {"a-amend": _label("a-amend")}


def test_16g_the_rendered_files_never_read_a_label(lws, monkeypatch):
    """C3: a render reads no label and prints none, beside a ranked read that labels.

    ``renderer.py``'s diff is empty for the phase (checked by the break record), so a
    render over the same graph is the parent commit's render; this row proves the
    store's one new read is never reached from it.
    """
    import mitos.renderer as renderer
    _ranked_world(lws)
    assert _ranked("mcp-surface", lws, [{"slug": "a", "score": 0.9}])["stamp_labels"]
    monkeypatch.setattr(GraphStore, "get_live_axioms",
                        lambda self, slugs: pytest.fail("a render read the labels"))
    paths = renderer.MitosRenderer(lws.workspace_dir).render_all(GraphStore(lws.db_path))
    assert paths
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        assert "stamp_labels" not in text and _label("a-amend") not in text, path
    assert "stamp_labels" not in inspect.getsource(renderer)


# --------------------------------------------------------------------------- #
# Rows 17 and 18: the unbuilt graph, and the healthy note
# --------------------------------------------------------------------------- #

def test_17_the_unbuilt_note_stands_for_the_label_map_too():
    walk = type("W", (), {"window": [], "edge": None, "end": "exhausted", "limit": 5,
                          "returned": 0, "retired": [], "passed": [], "stamps": {},
                          "live_ends": {}, "stamps_unread": False})()
    labels = StampLabels({"x": "y"}, True)
    for register in ("surface", "query"):
        composed = compose_ranked(
            walk, register=register, surface="mcp", config=type("C", (), {"project": "p"})(),
            handles=[], part=RetiredPartition([], 0, False), section=None,
            unbuilt=lambda: True, top_cluster_margin=0.01, edge_margin=0.01, ceiling=50,
            full_top=None, scan_faults=(), labels=labels)
        assert list(composed.envelope_keys()) == ["confidence", "note"]


def test_17_the_envelope_places_the_map_below_the_unbuilt_return():
    """L14's row: ``compose_ranked`` also empties the map on its unbuilt return, so the
    placement inside ``envelope_keys`` is pinned on the note itself."""
    note = recall.RankedNote(confidence="none", note="n", evidence=None,
                             override=recall.OVERRIDE_UNBUILT, handles=[], omitted=0,
                             section=None, withheld=0, stamps_unread=False,
                             labels={"x": "A label."}, labels_unread=True)
    assert list(note.envelope_keys()) == ["confidence", "note"]
    healthy = note._replace(override=None)
    assert list(healthy.envelope_keys())[-2:] == ["stamp_labels", "stamp_labels_unread"]


def test_17_the_lookup_under_the_unbuilt_graph_never_reads_labels():
    asked: List[Any] = []
    config = type("C", (), {"project": "p", "qdrant_collection": "c", "workspace_dir": "w"})()
    lookup = rejection_lookup(
        None, query="q", full_top=None, shape=lambda *a, **k: [], margin=.01,
        top_cluster_margin=.01, ceiling=50, graph_gap=lambda: True, surface="mcp",
        config=config, read_labels=lambda m: asked.append(m) or StampLabels({"x": "y"}, True))
    assert asked == [] and "stamp_labels" not in lookup.envelope
    assert lookup.envelope["note"] == missing_graph_note("mcp", config)


@pytest.mark.parametrize("driver", RANKED_JSON)
def test_18_a_healthy_note_carries_no_unread_clause(lws, driver):
    _ranked_world(lws)
    resp = _ranked(driver, lws, _HITS)
    assert "Stamp labels could not be read" not in resp["note"]
    assert "stamp_labels_unread" not in resp


# --------------------------------------------------------------------------- #
# Row 19: structural — a reader that composes a stamped payload is labelled or
# exempted by name
# --------------------------------------------------------------------------- #

# The calls that compose a stamped decision payload. A function that makes one must
# reach a label helper, or be named below with its reason.
_COMPOSES = {"compose_ranked", "rejection_lookup", "get_modifiers_map", "get_modifiers",
             "letter_payload", "oneline_payload", "retired_handle",
             "rejection_matches_payload", "show_payload", "lexical_degraded_read",
             "lexical_rejection_matches", "lexical_window_section", "scope_dump",
             "_review_pause"}
# A text renderer that prints the map reaches it through `_print_stamp_labels`.
_LABELS = {"ranked_labels", "list_labels", "match_labels", "pause_labels",
           "read_stamp_labels", "_print_stamp_labels"}
_COMPOSES |= {"_review_neighbors", "_review_tombstones", "_exists_handle_keys",
              "get_outgoing_edge_targets", "get_open_questions"}
# A reader that reads a stamp key by name joins the population too.
_STAMP_KEYS = {"amended_by", "narrowed_by", "superseded_by", "corrected_by", "live_end"}

# §3.6's right column by name, `cmd_open_questions`, and the readers the vision's list
# does not name (scout W3). The oneline tier and the exact-slug hit live inside a
# labelling function, so rows 16a and 16f pin them; the renderer is row 16g's.
EXEMPT = {
    "mcp_server._decision_payload": "a window shaper; its caller labels",
    "mcp_server.show_node": "lock 2: a by-handle read composes nothing",
    "cli.cmd_show": "lock 2: a by-handle read composes nothing",
    "cli.cmd_open_questions": "not on the vision's list; open questions name open questions",
    "cli._print_retired_partition": "a section printer; its exit prints the map (pinned)",
    "cli._print_rejection_section": "a section printer; its exit prints the map (pinned)",
    "cli._exists_recovery_lines": "the retired exists handle: a receipt echo",
    "cli._exists_retired_line": "the retired exists handle: a receipt echo",
    "cli._check_finding_side": "the check report: judge-adjacent, deferred to Trace",
    "degraded.scope_dump": "a read; its composer dump_section labels",
    "sync._live_successors": "the successor rule, read by the pause and the receipts",
    "sync._retired_exists_note": "the retired exists handle: a receipt echo",
    "sync._pause_message": "composed from the pause dict _review_pause labelled",
    "sync.MitosSyncManager._print_conflict_findings": "the check report: judge-adjacent",
    "sync.MitosSyncManager._review_tombstones":
        "the pause's partition (labelled in _review_pause) and the acknowledged list",
    "sync.MitosSyncManager._acknowledged_fields": "the acknowledged receipt's echo",
    "sync.MitosSyncManager._edges_created_field": "the created receipt's echo",
    "sync.MitosSyncManager._exists_receipt": "the retired exists handle: a receipt echo",
    "sync.MitosSyncManager._exists_handle_keys": "the retired exists handle: a receipt echo",
    "sync.MitosSyncManager.record_decision_entry":
        "the write path; its pause is labelled inside _review_pause",
}

# Every CLI text exit that prints a carrying answer, and how many label blocks it
# prints (cmd_query: blackout, miss, populated; _print_surface_answer: empty, full).
_TEXT_EXITS = {"_emit_lexical_degraded": 1, "cmd_query": 3, "_print_rejection_lookup": 1,
               "_print_rejection_lookup_lexical": 1, "cmd_list": 1, "cmd_record": 1,
               "_print_surface_answer": 2}
# Functions printing a `→ ` line that is no recall note.
_NOT_A_NOTE = {"_overview_notes"}

_SWEPT = ("mcp_server", "cli", "degraded", "sync")


def _functions(module_name: str):
    import importlib
    module = importlib.import_module(f"mitos.{module_name}")
    tree = ast.parse(inspect.getsource(module))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield f"{module_name}.{node.name}", node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield f"{module_name}.{node.name}.{item.name}", item


def _called(node: ast.AST) -> set:
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def _reads_a_stamp_key(node: ast.AST) -> bool:
    return any(isinstance(sub, ast.Constant) and sub.value in _STAMP_KEYS
               for sub in ast.walk(node))


def test_19_every_reader_that_composes_a_stamped_payload_is_labelled_or_exempt():
    """The population: every function in the four boundary modules that calls a
    stamped-payload composer (``_COMPOSES``) or reads a stamp key by name. Each must
    reach a label helper (or, on the CLI text surface, ``_print_stamp_labels``) or be
    exempt by name with its reason. A new reader fails here until it is one or the
    other."""
    composing, unlabelled = [], []
    for module_name in _SWEPT:
        for name, node in _functions(module_name):
            called = _called(node)
            if not (called & _COMPOSES or _reads_a_stamp_key(node)):
                continue
            composing.append(name)
            if not called & _LABELS and name not in EXEMPT:
                unlabelled.append(name)
    assert len(composing) >= 30, composing                    # the sweep saw the readers
    assert unlabelled == []
    stale = sorted(set(EXEMPT) - set(composing))
    assert stale == [], "an exemption names a reader the sweep no longer finds"


def test_19_every_cli_text_exit_prints_its_label_blocks():
    """Each text exit that renders a carrying answer prints the map through the one
    printer, once per exit branch; a new exit printing a note line must join the pin."""
    printed: Dict[str, int] = {}
    noted = set()
    for name, node in _functions("cli"):
        short = name.split(".", 1)[1]
        count = sum(1 for sub in ast.walk(node) if isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name) and sub.func.id == "_print_stamp_labels")
        if count:
            printed[short] = count
        for sub in ast.walk(node):
            if (isinstance(sub, ast.JoinedStr) and sub.values
                    and isinstance(sub.values[0], ast.Constant)
                    and str(sub.values[0].value).lstrip("\n").startswith("→ ")):
                noted.add(short)
    assert printed == _TEXT_EXITS
    assert noted - _NOT_A_NOTE <= set(_TEXT_EXITS)


# --------------------------------------------------------------------------- #
# Row 20: the shipped prose
# --------------------------------------------------------------------------- #

def test_20_the_three_read_descriptions_name_the_map_and_record_decisions_does_not():
    import asyncio
    tools = {t.name: " ".join(t.description.split())
             for t in asyncio.run(mcp_server.mcp.list_tools())}
    for name in ("surface_decisions", "query_decisions", "list_decisions"):
        assert "`stamp_labels` maps each live slug" in tools[name], name
        assert "`stamp_labels_unread`" in tools[name], name
    assert "stamp_labels" not in tools["record_decision"]
    assert "stamp_labels" not in tools["show_node"]


def test_20_the_skill_teaches_the_map_beside_the_stamps_and_on_the_pause():
    text = cli._skill_md_text("SPEC")
    assert ("the answer's `stamp_labels` gives each live decision those stamps name its "
            "axiom's first clause") in text
    assert "The pause's `stamp_labels` gives the first clause" in text


def test_20_setup_names_the_pauses_map():
    path = os.path.join(os.path.dirname(os.path.dirname(cli.__file__)), "SETUP.md")
    with open(path, encoding="utf-8") as fh:
        text = " ".join(fh.read().split())
    assert "The pause's `stamp_labels` gives the first clause" in text
