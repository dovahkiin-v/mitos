"""Offline pins: every calibrated similarity threshold equals its committed record value.

Layer C (``test_calibration_live.py``) measures where the classes a threshold separates
actually sit, and the committed record (``calibration/record.json``) keeps those runs and
the boundary each constant was set from. That layer never runs in CI: its embedding cache
is local and CI holds no key. These rows do, so a moved constant, a constant the record
lacks, or a record naming another embedding model turns the suite CI runs red.

The calibrated set is held here, not read from the record, and checked against the record
in both directions: a row walking only the record's keys would never see a constant that
never joined. A second row derives from ``mitos/`` source which constants owe calibration
at all (every module-level ``*_THRESHOLD``, ``*_FLOOR`` or ``*_MARGIN``) and requires each
to be calibrated, pending its phase, or exempt with a reason.

A constant phase joins the set in one commit: its name here, its record entry, its name
out of ``PENDING``, and the comment above the constant pointing at the record.
"""

import importlib
import os
import re
import sys
from typing import Dict, FrozenSet

import mitos
from mitos.models import get_embedding_model_id

sys.path.insert(0, os.path.dirname(__file__))
import _calibration_record as R  # noqa: E402

MITOS_DIR = os.path.dirname(os.path.abspath(mitos.__file__))
RECORD_POINTER = "tests/golden/calibration/record.json"

# Dotted names whose value the record pins. 6a and 7a add theirs.
CALIBRATED_CONSTANTS: FrozenSet[str] = frozenset({
    "mitos.recall.RANKED_EDGE_MARGIN",
    "mitos.recall.REJECTION_MATCH_FLOOR",
})

# Shipped thresholds that owe calibration, with the phase that sets each.
PENDING: Dict[str, str] = {
    "mitos.recall.SURFACE_STRONG_THRESHOLD": "6a",
    "mitos.recall.SURFACE_WEAK_THRESHOLD": "6a",
    "mitos.sync._NEIGHBOR_REVIEW_THRESHOLD": "7a",
}

# Suffix matches that are not calibrated against this record, each with its reason.
EXEMPT: Dict[str, str] = {
    "mitos.conflict.CONFLICT_SIMILARITY_FLOOR": (
        "keeps its own calibration block in conflict.py until the single-provider vision "
        "(ADR calibrated-thresholds-are-measured-in-the-embedding-space-they-compare)"
    ),
    "mitos.conflict.CONFLICT_SURFACE_THRESHOLD": "judge confidence gate, not an embedding similarity",
    "mitos.display._ONELINE_AXIOM_FLOOR": "a display width in characters (24), not a similarity",
    "mitos.retrieval_walk.NO_FLOOR": (
        "-inf: the field-scoped lookup's absence of a floor, by design (5c D3); the band "
        "carries its confidence, so there is no value to calibrate"
    ),
}


def _runtime_value(dotted: str):
    module, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(module), name)


def _calibrated_values() -> Dict[str, float]:
    return {name: _runtime_value(name) for name in CALIBRATED_CONSTANTS}


def test_record_names_the_configured_embedding_model():
    record = R.load_record()
    assert record["embedding_model"] == get_embedding_model_id(), (
        f"the calibration record is for {record['embedding_model']!r}, models.py configures "
        f"{get_embedding_model_id()!r}: run set_record_model, re-run Layer C, re-derive every "
        f"constant (calibration/README.md)"
    )


def test_every_calibrated_constant_resolves_to_a_float():
    for name in CALIBRATED_CONSTANTS:
        value = _runtime_value(name)
        assert type(value) is float, f"{name} is {value!r}, not a float"


def test_every_calibrated_constant_is_in_the_record_at_its_value():
    recorded = R.load_record()["constants"]
    missing = sorted(CALIBRATED_CONSTANTS - set(recorded))
    assert not missing, f"calibrated but absent from the record: {missing}"
    moved = {
        name: (value, recorded[name].get("value"))
        for name, value in _calibrated_values().items()
        if value != recorded[name].get("value")
    }
    assert not moved, f"source value differs from the record (source, record): {moved}"


def test_the_record_holds_no_constant_outside_the_calibrated_set():
    extra = sorted(set(R.load_record()["constants"]) - CALIBRATED_CONSTANTS)
    assert not extra, f"in the record but not in CALIBRATED_CONSTANTS: {extra}"


def test_check_pins_finds_no_problem():
    record = R.load_record()
    problems = R.check_pins(record, R.load_cited_runs(record), _calibrated_values(), get_embedding_model_id())
    assert not problems, "calibration record problems:\n" + "\n".join(f"  {p}" for p in problems)


def test_every_indexed_run_matches_its_file():
    """Cited or not: a baseline no constant reads yet is checked here, not in check_pins."""
    problems = R.check_runs(R.load_record())
    assert not problems, "calibration run problems:\n" + "\n".join(f"  {p}" for p in problems)


def test_the_three_maps_are_disjoint():
    pending, exempt = set(PENDING), set(EXEMPT)
    assert not (CALIBRATED_CONSTANTS & pending), CALIBRATED_CONSTANTS & pending
    assert not (CALIBRATED_CONSTANTS & exempt), CALIBRATED_CONSTANTS & exempt
    assert not (pending & exempt), pending & exempt


def test_every_shipped_threshold_is_calibrated_pending_or_exempt():
    population = R.threshold_population(MITOS_DIR)
    accounted = set(CALIBRATED_CONSTANTS) | set(PENDING) | set(EXEMPT)
    unaccounted = sorted(set(population) - accounted)
    assert not unaccounted, (
        f"thresholds in mitos/ owing calibration that no map names: {unaccounted}. Add each "
        f"to PENDING with the phase that sets it, or to EXEMPT with the reason it is not a "
        f"similarity"
    )
    stale = sorted(accounted - set(population))
    assert not stale, f"named here but no longer a *_THRESHOLD/*_FLOOR/*_MARGIN in mitos/: {stale}"
    not_literal = sorted(n for n, v in population.items() if v is None and n not in EXEMPT)
    assert not not_literal, (
        f"not a float literal in [0, 1]: {not_literal}. Make each a literal in its own "
        f"module (an alias of another threshold is its own constant), or exempt it with a reason"
    )


def test_the_calibrated_set_is_the_edge_margin_and_the_rejection_floor():
    """Inverted from `test_no_constant_is_calibrated_yet` (landed empty in 1b): 2e's
    commit added the first record entry (`test_the_edge_margin_is_the_first_calibrated_constant`),
    and 5b's adds the rejection-match floor, which 5a shipped PENDING at a placeholder.
    The record holds exactly the pinned set."""
    assert set(R.load_record()["constants"]) == set(CALIBRATED_CONSTANTS)
    assert CALIBRATED_CONSTANTS == frozenset({"mitos.recall.RANKED_EDGE_MARGIN",
                                              "mitos.recall.REJECTION_MATCH_FLOOR"})


def test_the_edge_margins_harbor_boundary_recomputes_from_the_committed_run():
    """D6: the method is code, not a notebook. Harbor's boundary in the record is what
    `edge_margin_boundary` derives from the committed run's fixtures; the dense side
    is re-derived out of repo by the dense runner through the same function."""
    entry = R.load_record()["constants"]["mitos.recall.RANKED_EDGE_MARGIN"]
    harbor = R.load_run("harbor-20261015-a79a820")
    assert R.edge_margin_boundary(harbor["fixtures"]) == entry["boundaries"]["harbor-20261015-a79a820"]


def _rejection_floor_entry():
    return R.load_record()["constants"]["mitos.recall.REJECTION_MATCH_FLOOR"]


def test_the_rejection_floors_harbor_boundary_recomputes_from_the_committed_run():
    """5b D3: the method is code. Harbor's boundary in the record is what
    `rejection_floor_boundary` derives from the committed run's fixtures; the dense side
    is re-derived out of repo by `dense.py floor` through the same function."""
    entry = _rejection_floor_entry()
    harbor = [run_id for run_id in entry["boundaries"] if run_id.startswith("harbor-")]
    assert len(harbor) == 1, entry["boundaries"]
    run = R.load_run(harbor[0])
    assert R.rejection_floor_boundary(run["fixtures"]) == entry["boundaries"][harbor[0]]
    assert run["provenance"]["item_reader"], "the cited Harbor run read items through the shipped client"


def test_the_rejection_floor_carries_its_cry_wolf_ceiling():
    """5b D4/D5: an escalation floor's entry states its ceiling and the ratio measured on
    the copy's replay, held under it, and says whether the floor rose from its boundaries."""
    entry = _rejection_floor_entry()
    assert "mitos.recall.REJECTION_MATCH_FLOOR" in R.ESCALATION_FLOORS
    block = entry["cry_wolf"]
    assert set(block) == set(R.CRY_WOLF_KEYS)
    assert block["ratio"] <= block["ceiling"]
    top = max(entry["boundaries"].values())
    if entry["side"] == "ceiling":
        assert block["raised_from"] == top and entry["value"] > top
    else:
        assert entry["side"] == "higher" and block["raised_from"] is None and entry["value"] == top
    assert R._check_cry_wolf("mitos.recall.REJECTION_MATCH_FLOOR", entry, top) == []


def test_each_calibrated_constant_comment_points_at_the_record():
    """The comment block directly above a calibrated constant names the record, not numbers."""
    for dotted in sorted(CALIBRATED_CONSTANTS):
        module, _, name = dotted.rpartition(".")
        path = os.path.join(MITOS_DIR, *module.split(".")[1:]) + ".py"
        lines = open(path, encoding="utf-8").read().splitlines()
        at = next((i for i, line in enumerate(lines) if re.match(rf"{re.escape(name)}\s*[:=]", line)), None)
        assert at is not None, f"{dotted}: no column-0 assignment in {path} to read the comment above"
        block = []
        for line in reversed(lines[:at]):
            if not line.lstrip().startswith("#"):
                break
            block.append(line)
        assert any(RECORD_POINTER in line for line in block), (
            f"{dotted}: the comment above it should point at {RECORD_POINTER}"
        )
