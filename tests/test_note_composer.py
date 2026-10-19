"""Phase 6b: one composer orders every line of a ranked note, and the band carries its evidence.

The pure-function half of the composer's proof. ``recall.compose_note`` is the one place
line order is decided; ``recall.compose_ranked`` (the four ranked surfaces) and
``recall.rejection_lookup`` (the field-scoped lookup) choose the lines. D9's table drives
both over band × lead × top cluster × edge × register × surface and checks every lawful
cell against each slot's own produced text, joined in the order this module states
literally (``_ORDER``), never against a hand-copied sentence. The driven rows (every
exit, on MCP, ``--json`` and CLI text) live in ``test_surface_confidence.py``, beside the
fixtures they need.

Scores derive from the calibrated constants (``_WEAK_SCORE``'s idiom), never literals a
re-calibration would move. The table passes the shipped margins; the distinct-margin
rows pass two synthetic ones so a swapped use reds.
"""

import ast
import pathlib
import shlex
from types import SimpleNamespace
from typing import Dict, List, Optional

import pytest

from mitos.stamp_labels import NO_LABELS, StampLabels
from mitos import recall, retrieval_walk
from mitos.recall import (EVIDENCE_EDGE_KINDS, NOTE_SLOTS, RANKED_EDGE_MARGIN,
                          RANKED_TOP_CLUSTER_MARGIN, REJECTION_MATCH_FLOOR,
                          SURFACE_STRONG_THRESHOLD, SURFACE_WEAK_THRESHOLD,
                          WEAK_MEANING, RejectionSection, RetiredPartition,
                          _SURFACE_POINTERS, absence_clause, assess_query_recall,
                          assess_surface_recall, band_evidence, blackout_note,
                          compose_note, compose_ranked, edge_clause,
                          lookup_edge, lookup_edge_clause, lookup_short_clause,
                          missing_graph_note, oq_scan_fault_clause, rejection_lead,
                          rejection_lead_clause, rejection_lookup, retired_lead_clause,
                          scope_counts_fault_clause, short_window, short_window_clause,
                          top_cluster_clause, top_cluster_size, window_edge)

EDGE_FOUND = retrieval_walk.EDGE_FOUND
EDGE_UNMEASURED = retrieval_walk.EDGE_UNMEASURED
EXHAUSTED = retrieval_walk.EXHAUSTED
SHORT_BOUND = retrieval_walk.SHORT_BOUND

# D1's order, stated here literally so a reordered ``NOTE_SLOTS`` reds the table.
_ORDER = ("rejection_lead", "retired_lead", "legend", "top_cluster", "edge", "absence",
          "not_computed", "section_state", "withheld", "stamps_unread", "labels_unread",
          "scan_faults")

_CONFIG = SimpleNamespace(project="my proj")
_CEILING = 50
_LIMIT = 3

# Top scores per band, from the constants. ``strong`` sits below the rejection floor,
# so a match at the floor can outrank it (6a: the floor is above ``strong``).
_TOP = {
    "strong": round((SURFACE_STRONG_THRESHOLD + REJECTION_MATCH_FLOOR) / 2, 4),
    "weak": round((SURFACE_STRONG_THRESHOLD + SURFACE_WEAK_THRESHOLD) / 2, 2),
    "none-with-results": round(SURFACE_WEAK_THRESHOLD - 0.05, 2),
    "none-empty": None,
}
assert SURFACE_STRONG_THRESHOLD <= _TOP["strong"] < REJECTION_MATCH_FLOOR
assert SURFACE_WEAK_THRESHOLD <= _TOP["weak"] < SURFACE_STRONG_THRESHOLD
_MATCH_SCORE = round(REJECTION_MATCH_FLOOR + 0.01, 4)
_BAND = {"strong": "strong", "weak": "weak", "none-with-results": "none",
         "none-empty": "none"}

_HANDLE = {"slug": "old", "id": "r1", "state": "superseded", "score": 0.9,
           "superseded_by": ["mid"], "live_end": ["new"]}
_MATCH_HERE = {"slug": "owner", "id": "o1", "state": "active", "score": _MATCH_SCORE,
               "ordinal": 2, "item": "the option", "letter": "here", "axiom": "A."}


def _hits(scores) -> List[SimpleNamespace]:
    return [SimpleNamespace(score=s, rank=i, node={"id": f"n{i}"})
            for i, s in enumerate(scores, start=1)]


def _scores(top: Optional[float], *, cluster: int, short: bool) -> List[float]:
    """A window for the cell: ``cluster`` hits inside the shipped margin, the rest clear."""
    if top is None:
        return []
    n = _LIMIT - 1 if short else _LIMIT
    inside = [round(top - RANKED_TOP_CLUSTER_MARGIN * k / 4, 6) for k in range(cluster)]
    outside = [round(top - 0.1 - 0.01 * k, 6) for k in range(n - len(inside))]
    return (inside + outside)[:n]


def _walk(scores, *, edge_kind: str, retired: int = 0) -> SimpleNamespace:
    """A ranked walk double for one edge kind over ``scores``."""
    edge = None
    if edge_kind == "mid":
        end = EDGE_FOUND
        edge = SimpleNamespace(score=scores[-1] - RANKED_EDGE_MARGIN / 4)
    elif edge_kind == "clear":
        end = EDGE_FOUND
        edge = SimpleNamespace(score=scores[-1] - RANKED_EDGE_MARGIN * 4)
    elif edge_kind == "unmeasured":
        end = EDGE_UNMEASURED
    else:  # "none" (a full window over an index that ran out) and "short"
        end = EXHAUSTED
    returned = len(scores) + retired
    if edge_kind == "short":
        returned = max(returned, 1)
    return SimpleNamespace(window=_hits(scores), edge=edge, end=end, limit=_LIMIT,
                           returned=returned, retired=[None] * retired, passed=[],
                           stamps_unread=False)


def _section(matches) -> RejectionSection:
    return RejectionSection(matches=list(matches), state=None, dropped=None, withheld=0,
                            clause=None)


# --------------------------------------------------------------------------- #
# D9's table
# --------------------------------------------------------------------------- #

_BANDS = list(_TOP)
_LEADS = ["none", "rejection", "retired", "both", "blackout", "unbuilt"]
_CLUSTERS = [1, 2]
_EDGES = ["none", "mid", "unmeasured", "short"]
_REGISTERS = ["surface", "query", "lookup"]
_SURFACES = ["cli", "mcp"]


def _lawful(band, lead, cluster, edge, register) -> bool:
    """The table's predicate: which cells a real answer can reach."""
    empty = band == "none-empty"
    if register == "lookup" and lead not in ("none", "unbuilt"):
        return False            # the lookup has no decision ranking, so no lead line
    if lead in ("blackout", "unbuilt") and not empty:
        return False            # both need an empty window
    if empty and (cluster > 1 or edge in ("mid", "unmeasured")):
        return False            # no window: no cluster, no edge
    if empty and edge == "short" and lead != "blackout":
        return False            # an empty non-blackout window states no short clause
    if lead == "blackout" and edge != "short":
        return False            # a blackout window is always short
    if lead in ("retired", "both") and empty:
        return False            # the retired lead needs a top live hit to outrank
    return True


def _ranked_cell(band, lead, cluster, edge, register, surface):
    top = _TOP[band]
    scores = _scores(top, cluster=cluster, short=edge == "short")
    retired = 1 if lead in ("retired", "both", "blackout") else 0
    walk = _walk(scores, edge_kind=edge, retired=retired)
    handles = [_HANDLE] if retired else []
    part = RetiredPartition(shown=handles, omitted=0, leads=lead in ("retired", "both"))
    matches = [_MATCH_HERE] if lead in ("rejection", "both") else []
    composed = compose_ranked(
        walk, register=register, surface=surface, config=_CONFIG, handles=handles,
        part=part, section=_section(matches), unbuilt=lambda: lead == "unbuilt",
        top_cluster_margin=RANKED_TOP_CLUSTER_MARGIN, edge_margin=RANKED_EDGE_MARGIN,
        ceiling=_CEILING, full_top=None, scan_faults=(), labels=NO_LABELS,
    )
    if lead == "unbuilt":
        return composed, {"legend": missing_graph_note(surface, _CONFIG)}
    if register == "surface":
        _, legend = assess_surface_recall(semantic_ran=True, top_score=top,
                                          result_count=len(scores), scope=None,
                                          surface=surface, degraded_reason=None)
    else:
        _, legend = assess_query_recall(top_score=top, result_count=len(scores),
                                        config=_CONFIG, surface=surface)
    slots: Dict[str, str] = {}
    if lead in ("rejection", "both"):
        slots["rejection_lead"] = rejection_lead_clause(_MATCH_HERE, surface=surface)
    if lead in ("retired", "both"):
        slots["retired_lead"] = retired_lead_clause(_HANDLE, surface=surface)
    slots["legend"] = (blackout_note(handles, surface=surface, config=_CONFIG)
                       if lead == "blackout" else legend)
    if scores and top_cluster_size(scores, margin=RANKED_TOP_CLUSTER_MARGIN) > 1:
        slots["top_cluster"] = top_cluster_clause(cluster, register=register,
                                                  margin=RANKED_TOP_CLUSTER_MARGIN)
    if edge in ("mid", "unmeasured"):
        slots["edge"] = edge_clause(
            window_edge(walk, margin=RANKED_EDGE_MARGIN, ceiling=_CEILING, full_top=None),
            surface=surface)
    elif edge == "short":
        slots["edge"] = short_window_clause(short_window(walk), surface=surface)
    if band != "strong" and lead not in ("retired", "both", "blackout"):
        slots["absence"] = absence_clause(register=register, surface=surface,
                                          config=_CONFIG, scope=None)
    return composed, slots


def _lookup_cell(band, lead, cluster, edge, surface):
    top = _TOP[band]
    scores = _scores(top, cluster=cluster, short=edge == "short")
    walk = _walk(scores, edge_kind=edge)
    walk.short = False
    walk.dropped = {}
    walk.item_counts = {}
    walk.stamps = {}
    walk.live_ends = {}

    def shape(w, **_kw):
        return [{"id": f"o{i}", "slug": f"own{i}", "letter": "here", "ordinal": 1,
                 "score": hit.score} for i, hit in enumerate(w.window)]

    lookup = rejection_lookup(
        walk if scores or lead == "unbuilt" else None, query="q", full_top=None,
        shape=shape, margin=RANKED_EDGE_MARGIN,
        top_cluster_margin=RANKED_TOP_CLUSTER_MARGIN, ceiling=_CEILING,
        graph_gap=lambda: lead == "unbuilt", surface=surface, config=_CONFIG,
        read_labels=lambda _matches: NO_LABELS,
    )
    if lead == "unbuilt":
        return lookup, {"legend": missing_graph_note(surface, _CONFIG)}
    slots = {"legend": recall._lookup_band_note(_BAND[band], top_score=top,
                                               result_count=len(scores),
                                               surface=surface, config=_CONFIG)}
    if scores and top_cluster_size(scores, margin=RANKED_TOP_CLUSTER_MARGIN) > 1:
        slots["top_cluster"] = top_cluster_clause(cluster, register="lookup",
                                                  margin=RANKED_TOP_CLUSTER_MARGIN)
    if edge in ("mid", "unmeasured"):
        slots["edge"] = lookup_edge_clause(
            lookup_edge(walk, owners=len(scores), margin=RANKED_EDGE_MARGIN,
                        ceiling=_CEILING, full_top=None),
            ceiling=_CEILING, surface=surface)
    elif edge == "short":
        slots["edge"] = lookup_short_clause(walk, surface=surface)
    if band != "strong":
        slots["absence"] = absence_clause(register="lookup", surface=surface,
                                          config=_CONFIG, scope=None)
    return lookup, slots


def _cells():
    for band in _BANDS:
        for lead in _LEADS:
            for cluster in _CLUSTERS:
                for edge in _EDGES:
                    for register in _REGISTERS:
                        for surface in _SURFACES:
                            if _lawful(band, lead, cluster, edge, register):
                                yield band, lead, cluster, edge, register, surface


def test_the_table_enters_every_lawful_cell_and_each_note_is_its_slots_in_order():
    """D9: every lawful cell's note is exactly its present slots, in D1's order.

    Each slot's text comes from the function that words it; the expected note is those
    texts joined in ``_ORDER``. The cell floor proves the frame was entered.
    """
    count = 0
    for band, lead, cluster, edge, register, surface in _cells():
        cell = (band, lead, cluster, edge, register, surface)
        if register == "lookup":
            lookup, slots = _lookup_cell(band, lead, cluster, edge, surface)
            note, evidence = lookup.envelope["note"], lookup.envelope.get("band_evidence")
            override = lookup.override
        else:
            composed, slots = _ranked_cell(band, lead, cluster, edge, register, surface)
            note, evidence, override = composed.note, composed.evidence, composed.override
        expected = " ".join(slots[name] for name in _ORDER if name in slots)
        assert note == expected, cell
        assert (evidence is not None) == (band != "none-empty" and lead != "unbuilt"), cell
        assert override == {"blackout": "blackout", "unbuilt": "unbuilt"}.get(lead), cell
        count += 1
    assert count >= 450, count


def test_note_slots_is_d1s_order():
    assert NOTE_SLOTS == _ORDER


def test_compose_note_places_present_lines_and_skips_absent_ones():
    note = compose_note(legend="L.", absence="A.", rejection_lead="R.", withheld="W.",
                        scan_faults=["F1.", "F2."], top_cluster=None)
    assert note == "R. L. A. W. F1. F2."
    assert compose_note(legend="L.") == "L."


def test_every_slot_lands_in_order_when_all_are_present():
    texts = {name: f"<{name}>" for name in _ORDER if name != "scan_faults"}
    note = compose_note(**texts, scan_faults=["<scan_faults>"])
    assert note == " ".join(f"<{name}>" for name in _ORDER)


# --------------------------------------------------------------------------- #
# The evidence object (D3)
# --------------------------------------------------------------------------- #

def test_the_evidence_key_set_is_exactly_four_and_carries_no_admitted_input():
    """6a admitted no input: a later admitted input inverts this row deliberately."""
    walk = _walk([0.8, 0.79, 0.6], edge_kind="mid")
    evidence = band_evidence(walk, scores=[h.score for h in walk.window],
                             top_cluster_margin=0.05, edge_margin=0.05)
    assert set(evidence) == {"top_score", "top_cluster", "edge", "next_score"}


@pytest.mark.parametrize("edge_kind, token, has_next", [
    ("mid", "mid_cluster", True), ("clear", "clear", True),
    ("unmeasured", "unmeasured", False), ("none", "end", False),
])
def test_each_walk_end_gives_its_edge_token(edge_kind, token, has_next):
    walk = _walk([0.8, 0.7, 0.6], edge_kind=edge_kind)
    evidence = band_evidence(walk, scores=[0.8, 0.7, 0.6],
                             top_cluster_margin=RANKED_TOP_CLUSTER_MARGIN,
                             edge_margin=RANKED_EDGE_MARGIN)
    assert evidence["edge"] == token and token in EVIDENCE_EDGE_KINDS
    assert ("next_score" in evidence) is has_next
    assert evidence["top_score"] == 0.8


def test_a_short_bound_window_reads_unmeasured_and_an_exhausted_short_one_end():
    """SHORT_BOUND: the fetch stopped at its bound, so a further rank may exist."""
    walk = _walk([0.8, 0.7], edge_kind="short")
    walk.end = SHORT_BOUND
    kw = dict(scores=[0.8, 0.7], top_cluster_margin=0.01, edge_margin=0.01)
    assert band_evidence(walk, **kw)["edge"] == "unmeasured"
    walk.end = EXHAUSTED
    assert band_evidence(walk, **kw)["edge"] == "end"


def test_the_evidence_is_absent_on_an_empty_window_and_loud_on_an_unknown_end():
    walk = _walk([0.8], edge_kind="none")
    assert band_evidence(walk, scores=[], top_cluster_margin=0.01, edge_margin=0.01) is None
    walk.end = "sideways"
    with pytest.raises(ValueError, match="sideways"):
        band_evidence(walk, scores=[0.8], top_cluster_margin=0.01, edge_margin=0.01)


def test_a_negative_gap_is_mid_cluster_as_the_edge_rule_says():
    walk = _walk([0.7, 0.6], edge_kind="mid")
    walk.edge = SimpleNamespace(score=0.65)   # the next rank scores above the last
    evidence = band_evidence(walk, scores=[0.7, 0.6], top_cluster_margin=0.01,
                             edge_margin=0.01)
    assert evidence["edge"] == "mid_cluster" and evidence["next_score"] == 0.65


def test_the_two_margins_are_read_where_they_belong():
    """Distinct synthetic margins: the cluster reads ``top_cluster_margin``, the edge
    ``edge_margin``; a swapped use inside the leaf reds one assertion or the other."""
    scores = [0.80, 0.77, 0.70]
    walk = _walk(scores, edge_kind="mid")
    walk.edge = SimpleNamespace(score=0.69)            # gap 0.01 to the last
    wide, narrow = 0.05, 0.005
    evidence = band_evidence(walk, scores=scores, top_cluster_margin=wide,
                             edge_margin=narrow)
    assert evidence["top_cluster"] == 2                # 0.77 within 0.05; 0.70 is not
    assert evidence["edge"] == "clear"                 # 0.01 ≥ 0.005
    swapped = band_evidence(walk, scores=scores, top_cluster_margin=narrow,
                            edge_margin=wide)
    assert swapped["top_cluster"] == 1 and swapped["edge"] == "mid_cluster"
    composed = compose_ranked(
        walk, register="query", surface="mcp", config=_CONFIG, handles=[],
        part=RetiredPartition([], 0, False), section=_section([]),
        unbuilt=lambda: False, top_cluster_margin=wide, edge_margin=narrow,
        ceiling=_CEILING, full_top=None, scan_faults=(), labels=NO_LABELS,
    )
    assert composed.evidence == evidence
    assert top_cluster_clause(2, register="query", margin=wide) in composed.note


# --------------------------------------------------------------------------- #
# The rejection lead (D5)
# --------------------------------------------------------------------------- #

def test_only_a_live_owners_match_strictly_above_the_top_leads():
    retired = dict(_MATCH_HERE, letter="retired", slug="gone")
    assert rejection_lead([retired, _MATCH_HERE], top_score=0.7) is _MATCH_HERE
    assert rejection_lead([retired], top_score=0.7) is None
    assert rejection_lead([retired], top_score=None) is None


def test_an_equal_score_does_not_lead():
    assert rejection_lead([_MATCH_HERE], top_score=_MATCH_SCORE) is None
    assert rejection_lead([_MATCH_HERE], top_score=_MATCH_SCORE - 0.0001) is _MATCH_HERE


def test_a_window_owner_leads_whatever_its_own_rank_and_the_line_names_it():
    """The lead is never gated on the owner's rank: rank 3 leads when its item outranks rank 1."""
    window_owner = dict(_MATCH_HERE, letter="window", rank=3)
    assert rejection_lead([window_owner], top_score=0.75) is window_owner
    line = rejection_lead_clause(window_owner, surface="mcp")
    assert "at rank 3 of this window" in line and "owner" in line
    assert "on its match in rejection_matches" in rejection_lead_clause(_MATCH_HERE,
                                                                          surface="mcp")


def test_with_no_live_hit_a_live_owners_match_leads_the_blackout():
    walk = _walk([], edge_kind="short", retired=1)
    composed = compose_ranked(
        walk, register="surface", surface="mcp", config=_CONFIG, handles=[_HANDLE],
        part=RetiredPartition([_HANDLE], 0, False), section=_section([_MATCH_HERE]),
        unbuilt=lambda: False, top_cluster_margin=RANKED_TOP_CLUSTER_MARGIN,
        edge_margin=RANKED_EDGE_MARGIN, ceiling=_CEILING, full_top=None, scan_faults=(), labels=NO_LABELS,
    )
    lead = rejection_lead_clause(_MATCH_HERE, surface="mcp")
    blackout = blackout_note([_HANDLE], surface="mcp", config=_CONFIG)
    assert composed.note.startswith(f"{lead} {blackout}")
    assert composed.override == "blackout"


@pytest.mark.parametrize("surface", _SURFACES)
def test_the_lead_line_is_a_read_instruction_never_a_ban(surface):
    line = rejection_lead_clause(_MATCH_HERE, surface=surface).casefold()
    for banned in ("decide", "record it", "do not", "never", "banned", "must not"):
        assert banned not in line, banned
    assert "its own context" in line and "carries over" in line
    with pytest.raises(KeyError):
        rejection_lead_clause(_MATCH_HERE, surface="web")


# --------------------------------------------------------------------------- #
# The absence clause (D6)
# --------------------------------------------------------------------------- #

def _composed(lead: str, band: str = "weak", edge: str = "mid", cluster: int = 2):
    return _ranked_cell(band, lead, cluster, edge, "surface", "mcp")[0].note


def test_the_absence_clause_is_left_out_under_the_retired_lead_and_the_blackout():
    absence = absence_clause(register="surface", surface="mcp", config=_CONFIG, scope=None)
    assert absence not in _composed("retired")
    assert absence not in _composed("both")
    assert absence not in _ranked_cell("none-empty", "blackout", 1, "short", "surface",
                                       "mcp")[0].note


def test_the_absence_clause_is_kept_under_a_rejection_lead_a_cluster_and_an_edge():
    absence = absence_clause(register="surface", surface="mcp", config=_CONFIG, scope=None)
    for lead in ("none", "rejection"):
        for edge in ("mid", "unmeasured", "none"):
            assert absence in _composed(lead, edge=edge), (lead, edge)


def test_the_absence_clause_rides_weak_and_none_never_strong():
    absence = absence_clause(register="query", surface="cli", config=_CONFIG, scope=None)
    for band in ("weak", "none-with-results", "none-empty"):
        cluster = 1 if band == "none-empty" else 2
        edge = "none" if band == "none-empty" else "mid"
        assert absence in _ranked_cell(band, "none", cluster, edge, "query", "cli")[0].note
    assert absence not in _ranked_cell("strong", "none", 2, "mid", "query", "cli")[0].note


def test_the_lookup_absence_names_the_whole_depth_read_alone():
    for surface in _SURFACES:
        clause = absence_clause(register="lookup", surface=surface, config=_CONFIG,
                                scope=None)
        assert "oneline" not in clause and "brief" not in clause
        assert "rejected_paths" in clause
    assert "list_decisions(scope='<scope>')" in absence_clause(
        register="lookup", surface="mcp", config=_CONFIG, scope=None)


def test_the_absence_pointer_carries_the_selector_and_the_scope_in_hand():
    cli = absence_clause(register="surface", surface="cli", config=_CONFIG, scope="auth db")
    assert "`mitos list -p 'my proj' --scope 'auth db'`" in cli
    assert "`mitos list -p 'my proj' --scope 'auth db' --oneline`" in cli
    mcp = absence_clause(register="surface", surface="mcp", config=_CONFIG, scope="auth")
    assert "list_decisions(scope='auth')" in mcp
    assert "list_decisions(scope='auth', oneline=True)" in mcp
    assert "mitos " not in mcp
    q = absence_clause(register="query", surface="cli", config=_CONFIG, scope=None)
    assert "--scope <scope>" in q and "-p 'my proj'" in q


def test_the_exhaustive_pointers_exist_on_both_surfaces_and_the_cli_forms_parse():
    """G6: every new key under both outer keys; the CLI forms parse as printed."""
    from mitos.cli import _build_parser

    parser = _build_parser()
    for key in ("exhaustive", "exhaustive_oneline", "exhaustive_whole"):
        assert key in _SURFACE_POINTERS["cli"] and key in _SURFACE_POINTERS["mcp"]
        cli_form = _SURFACE_POINTERS["cli"][key].format(selector="-p 'x y'",
                                                        scope="'auth'")
        argv = shlex.split(cli_form.strip("`"))
        assert argv[0] == "mitos"
        args = parser.parse_args(argv[1:])
        assert args.scope == "auth"
        assert "mitos " not in _SURFACE_POINTERS["mcp"][key]


@pytest.mark.parametrize("register", ["surface", "query", "lookup"])
def test_an_unknown_register_is_loud(register):
    with pytest.raises(ValueError):
        absence_clause(register="vibes", surface="mcp", config=_CONFIG, scope=None)
    with pytest.raises(ValueError):
        top_cluster_clause(2, register="vibes", margin=0.01)
    assert top_cluster_clause(1, register=register, margin=0.01) is None


def test_compose_ranked_refuses_an_unknown_register_and_a_scoped_query():
    kw = dict(surface="mcp", config=_CONFIG, handles=[],
              part=RetiredPartition([], 0, False), section=None, unbuilt=lambda: False,
              top_cluster_margin=0.01, edge_margin=0.01, ceiling=_CEILING,
              full_top=None, scan_faults=(), labels=NO_LABELS)
    with pytest.raises(ValueError, match="register"):
        compose_ranked(None, register="lookup", **kw)
    with pytest.raises(ValueError, match="scope"):
        compose_ranked(None, register="query", scope="auth", **kw)
    with pytest.raises(ValueError, match="scope"):
        compose_ranked(None, register="query", scope_counts={}, **kw)


def test_compose_ranked_names_every_band_input_with_no_default():
    """The T7 gate idiom: a call site that forgets an input fails, never defaults."""
    import inspect

    params = inspect.signature(compose_ranked).parameters
    for name in ("register", "surface", "config", "handles", "part", "section",
                 "unbuilt", "top_cluster_margin", "edge_margin", "ceiling", "full_top",
                 "scan_faults"):
        assert params[name].default is inspect.Parameter.empty, name
    assert "top_cluster_margin" in inspect.signature(rejection_lookup).parameters


# --------------------------------------------------------------------------- #
# The legends (D4) and the stale texts (D7)
# --------------------------------------------------------------------------- #

def _legends():
    """Every legend at every reachable input, both surfaces, unused-scope variants too."""
    counts = {"auth": {"active_decisions": 2, "parked_open_questions": 0}}
    out = []
    for surface in _SURFACES:
        for band in _BANDS:
            top = _TOP[band]
            n = 0 if top is None else 2
            for scope, sc in ((None, None), ("nope", counts), ("auth", counts)):
                out.append(("surface", band, assess_surface_recall(
                    semantic_ran=True, top_score=top, result_count=n, scope=scope,
                    scope_counts=sc, surface=surface, degraded_reason=None)[1]))
            out.append(("query", band, assess_query_recall(
                top_score=top, result_count=n, config=_CONFIG, surface=surface)[1]))
            out.append(("lookup", band, recall._lookup_band_note(
                _BAND[band], top_score=top, result_count=n, surface=surface,
                config=_CONFIG)))
    return out


def test_every_weak_legend_carries_the_shared_phrase():
    weak = [(reg, legend) for reg, band, legend in _legends() if band == "weak"]
    assert {reg for reg, _ in weak} == {"surface", "query", "lookup"}
    for reg, legend in weak:
        assert WEAK_MEANING in legend, reg


def test_no_none_legend_keeps_an_unmeasured_verdict():
    for reg, band, legend in _legends():
        if not band.startswith("none"):
            continue
        low = legend.casefold()
        for verdict in ("too low to be related", "likely no settled precedent",
                        "decide fresh", "the scope is populated", "safe to decide",
                        "treat as no-precedent"):
            assert verdict not in low, (reg, band, verdict)


def test_parentheses_balance_in_every_legend():
    for reg, band, legend in _legends():
        depth = 0
        for ch in legend:
            depth += {"(": 1, ")": -1}.get(ch, 0)
            assert depth >= 0, (reg, band, legend)
        assert depth == 0, (reg, band, legend)


def test_the_nothing_ranked_surface_legend_names_no_scope():
    """Scope never filters the ranking; the absence pointer names it instead."""
    legend = assess_surface_recall(semantic_ran=True, top_score=None, result_count=0,
                                   scope="auth", surface="mcp", degraded_reason=None)[1]
    assert "auth" not in legend


_QUERY_REGISTER_BANNED = ("decide", "record it", "before deciding",
                          "the scope is populated", "no settled precedent")


def test_no_query_or_lookup_note_in_the_table_instructs_a_write_or_judges_the_corpus():
    """The register lock over every note the two registers compose, new slots included."""
    seen = 0
    for band, lead, cluster, edge, register, surface in _cells():
        if register == "surface":
            continue
        if register == "lookup":
            note = _lookup_cell(band, lead, cluster, edge, surface)[0].envelope["note"]
        else:
            note = _ranked_cell(band, lead, cluster, edge, register, surface)[0].note
        low = note.casefold()
        for banned in _QUERY_REGISTER_BANNED:
            assert banned not in low, (band, lead, cluster, edge, register, surface, banned)
        seen += 1
    assert seen >= 150, seen


def test_the_scan_fault_clauses_name_the_scope_and_say_what_was_not_read():
    assert oq_scan_fault_clause("auth") == "The open questions in scope 'auth' could not be read."
    assert "'auth' is a live tag was not checked" in scope_counts_fault_clause("auth")


def test_scan_faults_close_the_note_in_their_order():
    walk = _walk([_TOP["weak"]], edge_kind="none")
    faults = [oq_scan_fault_clause("a"), scope_counts_fault_clause("a")]
    composed = compose_ranked(
        walk, register="surface", surface="mcp", config=_CONFIG, handles=[],
        part=RetiredPartition([], 0, False), section=_section([]),
        unbuilt=lambda: False, top_cluster_margin=RANKED_TOP_CLUSTER_MARGIN,
        edge_margin=RANKED_EDGE_MARGIN, ceiling=_CEILING, full_top=None,
        scope="a", scope_counts=None, scan_faults=faults, labels=NO_LABELS,
    )
    assert composed.note.endswith(" ".join(faults))


# --------------------------------------------------------------------------- #
# Lock A under the unbuilt graph, at the composer
# --------------------------------------------------------------------------- #

def test_the_unbuilt_note_stands_for_every_key_the_graph_supplies():
    """Lock A: no lead, partition, section, evidence, withheld or stamps under the note.

    8a's stamp label map joins this list by being placed below the unbuilt return in
    ``RankedNote.envelope_keys`` (and in ``rejection_lookup``).
    """
    walk = _walk([], edge_kind="short", retired=0)
    walk.stamps_unread = True
    stale = RejectionSection(matches=[_MATCH_HERE], state="short", dropped=2, withheld=1,
                             clause="Verification dropped 2 rejection matches.")
    for register in ("surface", "query"):
        composed = compose_ranked(
            walk, register=register, surface="mcp", config=_CONFIG, handles=[],
            part=RetiredPartition([], 0, False), section=stale, unbuilt=lambda: True,
            top_cluster_margin=RANKED_TOP_CLUSTER_MARGIN, edge_margin=RANKED_EDGE_MARGIN,
            ceiling=_CEILING, full_top=None, scan_faults=(),
            labels=StampLabels({"x": "A label."}, True),
        )
        keys = composed.envelope_keys()
        assert list(keys) == ["confidence", "note"], register
        assert keys["note"] == missing_graph_note("mcp", _CONFIG)


def test_the_unbuilt_graph_is_asked_only_on_an_empty_window():
    asked = []
    walk = _walk([_TOP["weak"]], edge_kind="none")
    compose_ranked(walk, register="query", surface="mcp", config=_CONFIG, handles=[],
                   part=RetiredPartition([], 0, False), section=_section([]),
                   unbuilt=lambda: asked.append(1) or True,
                   top_cluster_margin=RANKED_TOP_CLUSTER_MARGIN,
                   edge_margin=RANKED_EDGE_MARGIN, ceiling=_CEILING, full_top=None,
                   scan_faults=(), labels=NO_LABELS)
    assert asked == []


# --------------------------------------------------------------------------- #
# Structure: the walk-end twin, the margins by name, no site concatenates a note
# --------------------------------------------------------------------------- #

def test_the_edge_found_twin_agrees_with_the_walk():
    assert recall._WALK_END_EDGE_FOUND == retrieval_walk.EDGE_FOUND


_MITOS = pathlib.Path(recall.__file__).parent


def _calls(module: str, name: str):
    tree = ast.parse((_MITOS / f"{module}.py").read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == name):
            yield node


@pytest.mark.parametrize("module", ["cli", "mcp_server"])
def test_every_production_call_passes_each_margin_by_its_own_name(module):
    """The two margins are equal at shipped values, so only names can tell them apart."""
    found = 0
    for name, wants in (("compose_ranked", {"top_cluster_margin": "RANKED_TOP_CLUSTER_MARGIN",
                                            "edge_margin": "RANKED_EDGE_MARGIN"}),
                        ("rejection_lookup", {"top_cluster_margin": "RANKED_TOP_CLUSTER_MARGIN",
                                              "margin": "RANKED_EDGE_MARGIN"})):
        for call in _calls(module, name):
            kws = {kw.arg: kw.value for kw in call.keywords}
            for arg, constant in wants.items():
                value = kws.get(arg)
                assert isinstance(value, ast.Name) and value.id == constant, (
                    module, name, call.lineno, arg)
            found += 1
    assert found >= (3 if module == "cli" else 4), found


_NOTE_EXITS = {
    "cli": ("cmd_surface", "_print_surface_answer", "cmd_query", "_cmd_query_rejections",
            "_print_rejection_lookup", "_emit_lexical_degraded"),
    "mcp_server": ("surface_decisions", "query_decisions", "_query_rejections",
                   "_lexical_degraded_response"),
    "degraded": ("lexical_envelope", "lexical_rejection_envelope", "dump_section",
                 "_lexical_legend"),
}


@pytest.mark.parametrize("module", sorted(_NOTE_EXITS))
def test_no_note_exit_concatenates_note_text(module):
    """Success criterion 1: every note is composed by ``compose_note``. Inside the
    functions that build or print a recall note, no f-string splices a note-bearing
    value beside more text (``f"{…note…} …"``); printing the finished note alone
    after the ``→`` marker is placement, not composition."""
    tree = ast.parse((_MITOS / f"{module}.py").read_text())
    functions = {node.name: node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef)}
    assert set(_NOTE_EXITS[module]) <= set(functions), module
    offences = []
    for name in _NOTE_EXITS[module]:
        for node in ast.walk(functions[name]):
            if not isinstance(node, ast.JoinedStr):
                continue
            holes = [ast.unparse(v.value) for v in node.values
                     if isinstance(v, ast.FormattedValue)]
            text = "".join(v.value for v in node.values if isinstance(v, ast.Constant))
            noted = [h for h in holes if "note" in h.casefold()
                     and "notice" not in h.casefold() and h != "scope_note"]
            if noted and (len(holes) > 1 or text.strip(" \n→")):
                offences.append((name, node.lineno, noted))
    assert not offences, offences
