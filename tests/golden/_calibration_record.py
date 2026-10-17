"""Layer C's pure core: fixture shape, per-fixture results, class statistics, the record.

Layer C measures the similarity values every calibrated threshold sits between. This
module is the part with no services: it validates fixtures, derives a fixture's result
from hits the harness has already labelled, summarizes classes, reads and writes the
committed record (``calibration/record.json`` plus one file per run under
``calibration/runs/``), checks the calibrated constants against it, derives which shipped
constants owe calibration, and diffs a run against the record's latest one.

It is stdlib only and derives every path from ``__file__``, so a runner outside the repo
can append runs with nothing but ``sys.path.insert(0, "<repo>/tests/golden")``. The
service-touching half is ``_calibration_harness.py``; the live wiring is
``test_calibration_live.py``; the offline pin rows are ``test_calibrated_constants.py``.

Three rules the shape makes structural rather than conventional:

* A fixture names its *form* and never its space. ``query`` is embedded as a query and
  read through ``surface_decisions``; ``draft`` is embedded as a document and read through
  ``conflict.gather_candidates``. The space follows from the form (``SPACE_OF_FORM``), so a
  value is always recorded in the space its threshold compares.
* A run's class statistics nest class -> space -> sub-block (schema 2). The space is the
  sub-block's key, so a constant reads its own space's values by lookup and never the
  other's; each sub-block summarizes its negatives beside its targets.
* A dense run (a private corpus) carries ``fixtures: null`` and class aggregates only.
  ``append_run`` refuses one that carries fixtures, a top-level key outside the run
  shape, or a class block holding anything but its numeric statistics, so no private
  slug, axiom or query text can enter the public repo through the run's body. Its
  ``provenance`` is the runner's to keep free of corpus text.
"""

import ast
import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

GOLDEN_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.path.join(GOLDEN_DIR, "calibration")
FIXTURES_PATH = os.path.join(RECORD_DIR, "fixtures.json")
RECORD_FILE = "record.json"
RUNS_DIRNAME = "runs"
CACHE_DIR = os.path.join(GOLDEN_DIR, ".cache")

# 2 (1c): a run's ``classes`` nest class -> space -> sub-block, each with ``negatives``.
SCHEMA = 2
CLASSES: Tuple[str, ...] = (
    "bullseye",
    "handle-words",
    "family-neighbour",
    "off-axis",
    "rejection-target",
    "retired-precedent",
)
FORMS: Tuple[str, ...] = ("query", "draft")
SPACE_OF_FORM: Dict[str, str] = {"query": "query", "draft": "document"}
SPACES: Tuple[str, ...] = ("query", "document")
MEASURES: Tuple[str, ...] = ("target", "top_live", "item")
# How a rejection-target query meets the rejected option (vision §4.1's class table).
REJECTION_FORMS: Tuple[str, ...] = ("own-question", "elsewhere-carries", "elsewhere-does-not-carry")
# A rejection-target fixture's hard negatives come in two kinds (5b): ``generic`` (do
# nothing, defer, status quo) and ``vocabulary`` (another decision's non-generic item,
# rejected for another question, naming a different mechanism while sharing the query's
# subject words). ``elsewhere-does-not-carry`` is never a negative: it stays a target
# under its form (vision §4.1).
NEGATIVE_KINDS: Tuple[str, ...] = ("generic", "vocabulary")
# The two rejection-index shapes ruling 2 compares: one point per rejected option, and one
# point per decision holding the whole field. Not ``mitos.rejections.SHAPES``.
SHAPES: Tuple[str, ...] = ("per_item", "per_decision")
# ``ceiling`` (5b): an escalation floor raised past its boundaries by its cry-wolf
# ceiling; valid only with a ``cry_wolf`` block that says from where (`_check_cry_wolf`).
SIDES: Tuple[str, ...] = ("higher", "lower", "dense", "unmoved", "ceiling")
KINDS: Tuple[str, ...] = ("threshold", "margin")
CORPORA: Tuple[str, ...] = ("harbor", "dense")

# Soft-diff tolerance per recorded value. Vectors are served from the golden cache, so a
# same-model re-run reproduces a value to float32 noise; a move past this is a corpus
# edit, a fixture edit or a model change, each worth a look and none worth a red.
VALUE_BAND: float = 0.01

# The naming convention the population row polices. A threshold named otherwise
# (``…_MIN_SCORE``) escapes it, so constant phases name theirs with one of these.
THRESHOLD_SUFFIXES: Tuple[str, ...] = ("_THRESHOLD", "_FLOOR", "_MARGIN")

_FIXTURE_REQUIRED = ("id", "class", "form", "text", "targets", "negatives", "measure", "notes")
_FIXTURE_OPTIONAL = ("exclude", "target_items", "negative_items", "rejection_form")
_SLUG_LIST_KEYS = ("targets", "negatives", "exclude")
_ITEM_REF_KEYS = ("target_items", "negative_items")
_REJECTION_ONLY_KEYS = ("target_items", "negative_items", "rejection_form")
_KEBAB_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SUMMARY_FIELDS = ("run_id", "corpus", "embedding_model", "decision_count")
# The shape a dense run is held to, key for key (its texts are private).
_RUN_KEYS = ("run_id", "corpus", "decision_count", "node_count", "provenance", "fixtures", "classes")
_STATS_KEYS = ("n", "absent", "values", "min", "p10", "median", "p90", "max")
_CLASS_BLOCK_KEYS = ("measure",) + _STATS_KEYS + ("negatives",)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def golden_cache_path(model_id: str) -> str:
    """Returns the model-keyed golden embedding cache path Layer B also reads.

    The cache key is a content hash with no model in it, so the model lives in the file
    name. A path that dropped it would hand one model's vectors to a run recorded under
    another model's name.

    Args:
        model_id: The resolved embedding model id the provider is built with.

    Returns:
        ``tests/golden/.cache/embeddings-<model_id>.sqlite``.
    """
    return os.path.join(CACHE_DIR, f"embeddings-{model_id}.sqlite")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _is_slug_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(s, str) and s for s in value)


def _is_item_ref_list(value: Any, *, kinded: bool) -> bool:
    """A list of ``{owner, option}`` refs; ``kinded`` refs carry a ``kind`` from `NEGATIVE_KINDS` too."""
    keys = {"owner", "option", "kind"} if kinded else {"owner", "option"}
    return isinstance(value, list) and all(
        isinstance(r, dict) and set(r) == keys
        and all(isinstance(r[k], str) and r[k].strip() for k in ("owner", "option"))
        and (not kinded or r["kind"] in NEGATIVE_KINDS)
        for r in value
    )


def _validate_rejection_keys(fx: Dict[str, Any], label: str) -> None:
    """The rejection-target rules: its measure, form and item keys, and nowhere else."""
    rejection = fx["class"] == "rejection-target"
    if (fx["measure"] == "item") != rejection:
        raise ValueError(
            f"fixture {label}: measure `item` belongs to class rejection-target and that "
            f"class to measure `item` (class {fx['class']!r}, measure {fx['measure']!r})"
        )
    if not rejection:
        stray = [k for k in _REJECTION_ONLY_KEYS if k in fx]
        if stray:
            raise ValueError(f"fixture {label}: {stray} belong to rejection-target fixtures only")
        return
    if fx["form"] != "query":
        raise ValueError(
            f"fixture {label}: a rejection-target fixture is a query; the draft reader reads "
            f"no rejection shapes"
        )
    if fx.get("rejection_form") not in REJECTION_FORMS:
        raise ValueError(
            f"fixture {label}: `rejection_form` {fx.get('rejection_form')!r}; one of {list(REJECTION_FORMS)}"
        )
    if not fx.get("target_items"):
        raise ValueError(f"fixture {label}: a rejection-target fixture names at least one `target_items` ref")
    if not fx.get("negative_items"):
        raise ValueError(f"fixture {label}: a rejection-target fixture names at least one `negative_items` ref")
    if fx["negatives"]:
        raise ValueError(
            f"fixture {label}: a rejection-target fixture's negatives are `negative_items`; "
            f"slug `negatives` must be empty"
        )
    owners = sorted({r["owner"] for r in fx["target_items"]} - set(fx["targets"]))
    if owners:
        raise ValueError(f"fixture {label}: target item owners {owners} must be in `targets`")


def _validate_fixture(fx: Any, require_class: bool) -> None:
    if not isinstance(fx, dict):
        raise ValueError(f"fixture {fx!r}: not an object")
    label = repr(fx.get("id", "<no id>"))
    if "space" in fx:
        raise ValueError(
            f"fixture {label}: authors `space`, which is derived from `form` "
            f"(query -> query, draft -> document); remove the key"
        )
    unknown = sorted(set(fx) - set(_FIXTURE_REQUIRED) - set(_FIXTURE_OPTIONAL))
    if unknown:
        raise ValueError(f"fixture {label}: unknown keys {unknown}")
    missing = [k for k in _FIXTURE_REQUIRED if k not in fx]
    if missing:
        raise ValueError(f"fixture {label}: missing keys {missing}")
    if not isinstance(fx["id"], str) or not _KEBAB_RE.fullmatch(fx["id"]):
        raise ValueError(f"fixture {label}: `id` must be kebab-case")
    if fx["class"] is None:
        if require_class:
            raise ValueError(f"fixture {label}: `class` is null; label it with one of {list(CLASSES)}")
    elif fx["class"] not in CLASSES:
        raise ValueError(f"fixture {label}: unknown class {fx['class']!r}; one of {list(CLASSES)}")
    if fx["form"] not in FORMS:
        raise ValueError(f"fixture {label}: unknown form {fx['form']!r}; one of {list(FORMS)}")
    if fx["measure"] not in MEASURES:
        raise ValueError(f"fixture {label}: unknown measure {fx['measure']!r}; one of {list(MEASURES)}")
    if not isinstance(fx["text"], str) or not fx["text"].strip():
        raise ValueError(f"fixture {label}: `text` must be a non-empty string")
    if not isinstance(fx["notes"], str) or not fx["notes"].strip():
        raise ValueError(f"fixture {label}: `notes` must say why this label holds")
    for key in _SLUG_LIST_KEYS:
        if key in fx and not _is_slug_list(fx[key]):
            raise ValueError(f"fixture {label}: `{key}` must be a list of slugs")
    if "target_items" in fx and not _is_item_ref_list(fx["target_items"], kinded=False):
        raise ValueError(
            f"fixture {label}: `target_items` must be a list of {{owner, option}} objects, "
            f"both non-empty strings and nothing else"
        )
    if "negative_items" in fx and not _is_item_ref_list(fx["negative_items"], kinded=True):
        raise ValueError(
            f"fixture {label}: `negative_items` must be a list of {{owner, option, kind}} objects: "
            f"owner and option non-empty strings, kind one of {list(NEGATIVE_KINDS)}, nothing else"
        )
    _validate_rejection_keys(fx, label)
    if fx["measure"] == "target" and not fx["targets"]:
        raise ValueError(f"fixture {label}: measure `target` with no targets measures nothing")
    if set(fx["targets"]) & set(fx.get("exclude", [])):
        raise ValueError(f"fixture {label}: a slug is both a target and excluded")


def validate_fixtures(fixtures: Any, *, require_class: bool = True) -> List[dict]:
    """Validates a fixture list against the §3.2 shape.

    Args:
        fixtures: The parsed fixture list.
        require_class: False only for the in-memory smoke adaptation, whose fixtures carry
            ``class: null`` so that ``append_run`` refuses them.

    Returns:
        The same list.

    Raises:
        ValueError: Naming the first malformed fixture and what is wrong with it.
    """
    if not isinstance(fixtures, list):
        raise ValueError("`fixtures` must be a list")
    seen = set()
    for fx in fixtures:
        _validate_fixture(fx, require_class)
        if fx["id"] in seen:
            raise ValueError(f"fixture {fx['id']!r}: duplicate id")
        seen.add(fx["id"])
    return fixtures


def load_fixtures(path: str = FIXTURES_PATH) -> List[dict]:
    """Loads and validates ``fixtures.json``.

    Args:
        path: The fixture file (``{"_doc": {...}, "fixtures": [...]}``).

    Returns:
        The validated fixture list, possibly empty.

    Raises:
        ValueError: Naming the malformed fixture.
    """
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or "fixtures" not in doc:
        raise ValueError(f"{path}: expected an object with a `fixtures` list")
    return validate_fixtures(doc["fixtures"])


def check_fixture_slugs(fixtures: List[dict], known_slugs: Any) -> None:
    """Refuses a fixture naming a slug the built graph does not hold, item-ref owners included.

    Args:
        fixtures: Validated fixtures.
        known_slugs: Every slug in the collection the run reads.

    Raises:
        ValueError: Naming the fixture and its unknown slugs.
    """
    known = set(known_slugs)
    for fx in fixtures:
        named = fx["targets"] + fx["negatives"] + fx.get("exclude", []) + [
            r["owner"] for r in fx.get("target_items", []) + fx.get("negative_items", [])
        ]
        unknown = sorted(s for s in named if s not in known)
        if unknown:
            raise ValueError(f"fixture {fx['id']!r}: slugs not in the graph: {unknown}")


# ---------------------------------------------------------------------------
# Per-fixture result
# ---------------------------------------------------------------------------

def _slug_entry(slug: str, raw: List[dict], live: List[dict]) -> Dict[str, Any]:
    raw_pos = next((i for i, h in enumerate(raw) if h["slug"] == slug), None)
    if raw_pos is None:
        return {"slug": slug, "value": None, "absent": True, "raw_rank": None, "live_rank": None}
    live_pos = next((i for i, h in enumerate(live) if h["slug"] == slug), None)
    return {
        "slug": slug,
        "value": raw[raw_pos]["value"],
        "absent": False,
        "raw_rank": raw_pos + 1,
        "live_rank": None if live_pos is None else live_pos + 1,
    }


# 6b: the envelope's `band_evidence` object, recorded on a query-form result as
# `envelope_evidence`. Its edge tokens are spelled here because this module is stdlib
# only (the replay worker loads it by path); a row pins them equal to
# `recall.EVIDENCE_EDGE_KINDS`. `next_score` rides only beside an edge that was found.
EVIDENCE_EDGE_KINDS = ("mid_cluster", "clear", "unmeasured", "end")
_EVIDENCE_KEYS = ("top_score", "top_cluster", "edge")
_EVIDENCE_FOUND = ("mid_cluster", "clear")


def evidence_problem(evidence: Any) -> Optional[str]:
    """Says what is wrong with a recorded `envelope_evidence`, or None when it is sound.

    The exact key set (``next_score`` present iff the edge was found), the types (scores
    numbers, ``top_cluster`` an integer of at least 1, never a bool), and ``edge`` inside
    :data:`EVIDENCE_EDGE_KINDS`.
    """
    if not isinstance(evidence, dict):
        return f"envelope_evidence {evidence!r} is not an object"
    edge = evidence.get("edge")
    if edge not in EVIDENCE_EDGE_KINDS:
        return f"envelope_evidence edge {edge!r}; one of {list(EVIDENCE_EDGE_KINDS)}"
    want = set(_EVIDENCE_KEYS) | ({"next_score"} if edge in _EVIDENCE_FOUND else set())
    if set(evidence) != want:
        return f"envelope_evidence keys {sorted(evidence)}; want {sorted(want)}"
    for key in ("top_score", "next_score"):
        if key in evidence and (isinstance(evidence[key], bool)
                                or not isinstance(evidence[key], (int, float))):
            return f"envelope_evidence {key} {evidence[key]!r} is not a number"
    size = evidence["top_cluster"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        return f"envelope_evidence top_cluster {size!r}; an integer of at least 1"
    return None


def derive_result(
    fixture: dict, hits: List[dict], envelope_band: Optional[str], pause_floor: Optional[float],
    envelope_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Derives one fixture's §3.3 result from hits the harness already labelled.

    ``hits`` is the raw ranking the shipped reader received, best first, each labelled
    ``{slug, value, kind, live, self}``. A hit counts as a live competitor only when it
    is ``live`` (a decision in a live state; an open question never is) and not ``self``
    (the draft's own node, which the pause drops). Live ranks, the top live value and
    every margin are taken over those competitors only, so an open question or the
    draft's own node near the top shifts none of them.

    Args:
        fixture: A validated fixture.
        hits: The labelled raw hits.
        envelope_band: The shipped envelope's ``confidence`` (query form), else ``None``.
        pause_floor: ``sync._NEIGHBOR_REVIEW_THRESHOLD`` at run time (draft form), else
            ``None``.
        envelope_evidence: The shipped envelope's ``band_evidence`` (query form, 6b),
            recorded as ``envelope_evidence``; ``None`` (absent) when the band read no
            score or the reader predates it.

    Returns:
        The JSON-safe result dict. A target outside the window has ``value: null`` and
        ``absent: true``; it is never dropped.

    Raises:
        ValueError: If the band, its evidence or the floor is supplied for the wrong
            form, or the evidence is malformed (:func:`evidence_problem`).
    """
    query_form = fixture["form"] == "query"
    if not query_form and envelope_band is not None:
        raise ValueError(f"fixture {fixture['id']!r}: a draft read has no envelope band")
    if not query_form and envelope_evidence is not None:
        raise ValueError(f"fixture {fixture['id']!r}: a draft read has no envelope evidence")
    if envelope_evidence is not None:
        problem = evidence_problem(envelope_evidence)
        if problem:
            raise ValueError(f"fixture {fixture['id']!r}: {problem}")
    if query_form and pause_floor is not None:
        raise ValueError(f"fixture {fixture['id']!r}: a query read has no pause floor")
    # An open question is never live, whatever the caller labelled: both shipped readers
    # drop non-decisions before they rank.
    raw = [
        {"slug": h["slug"], "value": h["value"], "kind": h["kind"],
         "live": bool(h["live"]) and h["kind"] == "decision", "self": bool(h.get("self", False))}
        for h in hits
    ]
    # 6a: a hit's lexical overlap with the query (`overlap`) rides along when the harness
    # measured it; a reader that did not carries no key.
    for out, h in zip(raw, hits):
        if "overlap" in h:
            out["overlap"] = h["overlap"]
    live = [h for h in raw if h["live"] and not h["self"]]
    target_set = set(fixture["targets"])
    best_other = max((h["value"] for h in live if h["slug"] not in target_set), default=None)

    targets = []
    for slug in fixture["targets"]:
        entry = _slug_entry(slug, raw, live)
        entry["margin_vs_best_other_live"] = (
            entry["value"] - best_other
            if entry["value"] is not None and best_other is not None else None
        )
        if not query_form:
            entry["clears_pause_floor"] = (
                entry["value"] >= pause_floor
                if entry["value"] is not None and pause_floor is not None else None
            )
        targets.append(entry)

    result = {
        "id": fixture["id"],
        "class": fixture["class"],
        "space": SPACE_OF_FORM[fixture["form"]],
        "measure": fixture["measure"],
        "hits": raw,
        "targets": targets,
        "negatives": [_slug_entry(slug, raw, live) for slug in fixture["negatives"]],
        "top_live_value": max((h["value"] for h in live), default=None),
        "envelope_band": envelope_band,
        "pause_floor": pause_floor,
    }
    if envelope_evidence is not None:
        result["envelope_evidence"] = dict(envelope_evidence)
    if "rejection_form" in fixture:
        result["rejection_form"] = fixture["rejection_form"]
    return result


def _collapse(hits: List[dict], per_item: bool) -> List[Dict[str, Any]]:
    """Collapses part hits to one value per unit, best part first, re-sorted best first.

    A unit is ``(owner, ordinal)`` in the per-item shape and the owner alone in the
    per-decision shape. Ties keep the order the search returned them in.
    """
    best: Dict[Tuple[str, Optional[int]], Dict[str, Any]] = {}
    order: List[Tuple[str, Optional[int]]] = []
    for h in hits:
        key = (h["owner"], h["ordinal"] if per_item else None)
        if key not in best:
            order.append(key)
            best[key] = {"key": key, "value": h["value"]}
        elif h["value"] > best[key]["value"]:
            best[key]["value"] = h["value"]
    units = [best[k] for k in order]
    units.sort(key=lambda u: -u["value"])  # stable: equal values keep search order
    return units


def _item_shape(fixture: dict, hits: List[dict], refs: Dict[str, List[dict]], per_item: bool) -> Dict[str, Any]:
    units = _collapse(hits, per_item)
    rank_of = {u["key"]: i + 1 for i, u in enumerate(units)}
    value_of = {u["key"]: u["value"] for u in units}

    def key_of(ref: dict) -> Tuple[str, Optional[int]]:
        # The per-decision shape values every ref at its owner's whole-field point, so a
        # target and a negative sharing an owner get one value: the dilution it costs.
        return (ref["owner"], ref["ordinal"] if per_item else None)

    target_keys = {key_of(r) for r in refs["targets"]}
    best_other = next((u["value"] for u in units if u["key"] not in target_keys), None)

    def entry(ref: dict, margin: bool) -> Dict[str, Any]:
        key = key_of(ref)
        value = value_of.get(key)
        out: Dict[str, Any] = {"owner": ref["owner"], "option": ref["option"], "value": value,
                               "rank": rank_of.get(key), "absent": value is None}
        if "kind" in ref:
            out["kind"] = ref["kind"]
        if margin:
            out["margin_vs_best_other"] = (
                value - best_other if value is not None and best_other is not None else None
            )
        return out

    return {
        "hits": [dict(h) for h in hits],
        "targets": [entry(r, True) for r in refs["targets"]],
        "negatives": [entry(r, False) for r in refs["negatives"]],
        "top_value": units[0]["value"] if units else None,
    }


def derive_item_result(
    fixture: dict, hits_by_shape: Dict[str, List[dict]], refs: Dict[str, List[dict]]
) -> Dict[str, Any]:
    """Derives a rejection-target fixture's ``items`` value in each index shape read (§3.3).

    ``hits_by_shape[shape]`` is the shape collection's whole ranking, best first, each hit
    labelled by the harness from its point id: ``{point_id, owner, ordinal, part, value,
    owner_live}`` per item part (``ordinal``/``part`` are ``None`` per decision). ``refs``
    holds the fixture's ``target_items`` and ``negative_items`` resolved to ordinals:
    ``{"targets": [{owner, option, ordinal}], "negatives": [...]}``.

    Per item, an item's parts collapse to its best part's value *before* ranks and
    margins, so a target's own second part never competes with it and a two-part
    competitor costs it one rank, not two. ``margin_vs_best_other`` is the target's value
    minus the best collapsed unit that is not a target. Per decision, every ref is valued
    at its owner's whole-field point.

    ``per_item`` is required: it is the index that ships, and the class statistics read
    it. ``per_decision`` is optional (5b): only Harbor's throwaway shapes carry it, so a
    dense run over the copy derives ``per_item`` alone. A negative ref's ``kind`` rides
    its entry.

    Args:
        fixture: A validated rejection-target fixture.
        hits_by_shape: The labelled hits per shape read, ``per_item`` among them.
        refs: The resolved refs.

    Returns:
        ``{shape: {hits, targets, negatives, top_value}}`` for each shape read, in
        ``SHAPES`` order. A ref no hit carries is kept with ``value: null`` and
        ``absent: true``.

    Raises:
        ValueError: If the fixture is not item-measured, ``per_item`` is missing, or a
            shape is not one of ``SHAPES``.
    """
    if fixture["measure"] != "item":
        raise ValueError(f"fixture {fixture['id']!r}: only an item-measured fixture has items")
    if "per_item" not in hits_by_shape:
        raise ValueError(f"fixture {fixture['id']!r}: no hits for the per_item shape")
    unknown = sorted(set(hits_by_shape) - set(SHAPES))
    if unknown:
        raise ValueError(f"fixture {fixture['id']!r}: unknown shapes {unknown}; one of {list(SHAPES)}")
    return {shape: _item_shape(fixture, hits_by_shape[shape], refs, shape == "per_item")
            for shape in SHAPES if shape in hits_by_shape}


# ---------------------------------------------------------------------------
# Class statistics
# ---------------------------------------------------------------------------

def nearest_rank(sorted_values: List[float], pct: float) -> Optional[float]:
    """Returns the nearest-rank percentile of an ascending list (no interpolation).

    The value at 1-based rank ``ceil(pct / 100 * n)``, clamped to rank 1, so a recorded
    statistic can be reproduced by hand from the class's ``values``.

    Args:
        sorted_values: Ascending values.
        pct: The percentile, 0 to 100.

    Returns:
        The percentile value, or ``None`` for an empty list.
    """
    if not sorted_values:
        return None
    rank = max(1, math.ceil(pct / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


def stats(values: List[Optional[float]]) -> Dict[str, Any]:
    """Nearest-rank statistics over values, with each ``None`` counted as ``absent``.

    Returns:
        ``{n, absent, values, min, p10, median, p90, max}``; ``n`` counts the values
        given, present and absent.
    """
    present = sorted(v for v in values if v is not None)
    return {
        "n": len(values),
        "absent": len(values) - len(present),
        "values": present,
        "min": present[0] if present else None,
        "p10": nearest_rank(present, 10),
        "median": nearest_rank(present, 50),
        "p90": nearest_rank(present, 90),
        "max": present[-1] if present else None,
    }


def _picks(r: dict) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """A result's (target-side, negative-side) values under its measure; exhaustive."""
    measure = r["measure"]
    if measure == "target":
        return [t["value"] for t in r["targets"]], [n["value"] for n in r["negatives"]]
    if measure == "top_live":
        return [r["top_live_value"]], [n["value"] for n in r["negatives"]]
    if measure == "item":
        shape = r["items"]["per_item"]  # the ruled index (Q2); per-decision is `shapes`' evidence
        return [t["value"] for t in shape["targets"]], [n["value"] for n in shape["negatives"]]
    raise ValueError(f"result {r.get('id')!r}: unknown measure {measure!r}; one of {list(MEASURES)}")


def summarize_classes(results: List[dict]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Builds the run's class block from per-fixture results: class -> space -> sub-block.

    The space is the sub-block's key, so a constant reads values from its own space by
    lookup and never from the other (ADR
    ``calibrated-thresholds-are-measured-in-the-embedding-space-they-compare``). A class
    posed both ways (retired-precedent as a query and as a draft) has two sub-blocks.

    Results whose class is ``null`` (the smoke adaptation) are left out, so a smoke-only
    run summarizes to ``{}`` and ``append_run`` refuses it. A ``target`` fixture adds
    each target's value, a ``top_live`` fixture its top live value, an ``item`` fixture
    each target item's per-item value. ``negatives`` summarizes the sub-block's negative
    values: slug negatives, or the per-item ``negative_items`` for ``item``. Missing
    values count under ``absent`` and stay out of the statistics; percentiles are
    nearest-rank (see `nearest_rank`).

    Args:
        results: `derive_result` outputs (with ``items`` on item-measured ones).

    Returns:
        ``{class: {space: {measure, n, absent, values, min, p10, median, p90, max,
        negatives: {n, absent, values, min, p10, median, p90, max}}}}``. ``n`` counts
        the sub-block's fixtures; ``values`` plus ``absent`` counts what they measured.

    Raises:
        ValueError: If one (class, space) holds results under two measures, or a result
            names an unknown measure.
    """
    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for r in results:
        if r["class"] is not None:
            grouped.setdefault((r["class"], r["space"]), []).append(r)
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for cls, space in sorted(grouped):
        members = grouped[(cls, space)]
        measures = sorted({r["measure"] for r in members})
        if len(measures) > 1:
            raise ValueError(
                f"class {cls!r} in {space!r} space mixes measures {measures}; one measure per (class, space)"
            )
        values: List[Optional[float]] = []
        negatives: List[Optional[float]] = []
        for r in members:
            v, n = _picks(r)
            values += v
            negatives += n
        block: Dict[str, Any] = {"measure": measures[0]}
        block.update(stats(values))
        block["n"] = len(members)
        block["negatives"] = stats(negatives)
        out.setdefault(cls, {})[space] = block
    return out


def summarize_shapes(results: List[dict]) -> Dict[str, Dict[str, Any]]:
    """Builds the run's ``shapes`` block from its item-measured results (ruling 2's evidence).

    Per shape: ``targets`` and ``negatives`` statistics, ``targets`` again ``by_form``,
    ``negatives`` again ``negatives_by_kind`` (5b), ``target_rank_one`` (the share of
    target refs ranked first, over every target ref, absent ones included) and ``margin``
    (statistics of ``margin_vs_best_other``).

    Harbor only: ``build_run`` calls it for a harbor run, the one corpus that reads both
    shapes. A shape some item result lacks (a per-item-only result) is left out rather
    than summarized over part of the run.

    Args:
        results: Every fixture result of the run; only ``measure: item`` ones are read.

    Returns:
        ``{shape: {targets, negatives, by_form, negatives_by_kind, target_rank_one,
        margin}}`` for each shape every item result holds, or ``{}`` when the run holds no
        item-measured result.
    """
    items = [r for r in results if r.get("measure") == "item" and r.get("class") is not None]
    if not items:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for shape in SHAPES:
        if any(shape not in r["items"] for r in items):
            continue
        targets = [t for r in items for t in r["items"][shape]["targets"]]
        by_form = {
            form: stats([t["value"] for r in items if r.get("rejection_form") == form
                         for t in r["items"][shape]["targets"]])
            for form in REJECTION_FORMS
        }
        out[shape] = {
            "targets": stats([t["value"] for t in targets]),
            "negatives": stats([n["value"] for r in items for n in r["items"][shape]["negatives"]]),
            "by_form": by_form,
            "negatives_by_kind": {
                kind: stats([n["value"] for r in items for n in r["items"][shape]["negatives"]
                             if n.get("kind") == kind])
                for kind in NEGATIVE_KINDS
            },
            "target_rank_one": (sum(1 for t in targets if t["rank"] == 1) / len(targets)) if targets else None,
            "margin": stats([t["margin_vs_best_other"] for t in targets]),
        }
    return out


# ---------------------------------------------------------------------------
# The edge margin's method (recall.RANKED_EDGE_MARGIN)
# ---------------------------------------------------------------------------

# The classes whose targets are the decisions a ranked read should return. The other two
# classes carry no targets, so they have no ranking to invert.
EDGE_MARGIN_CLASSES: Tuple[str, ...] = ("bullseye", "handle-words")


def inversion_depths(results: List[dict]) -> List[float]:
    """Measures how deep the score order inverts the relevance order, per labelled target.

    Read over query-space results of ``EDGE_MARGIN_CLASSES`` only, in the ranking a
    ranked read keeps: live, non-self decisions, best first. When a non-target ranks above
    a target, the target's depth is the best such non-target's value minus the target's.
    Another target ranked above is not an inversion (``margin_vs_best_other_live`` would
    count it). A target that is absent from the window, or not a live competitor, has no
    rank to invert and is skipped. Takes the shape both the committed Harbor run and the
    dense runner's ``results.jsonl`` hold.

    Args:
        results: Fixture results (``derive_result``'s shape).

    Returns:
        One depth per inverted target, in result order. A target ranked first gives none.
    """
    depths: List[float] = []
    for r in results:
        if r.get("class") not in EDGE_MARGIN_CLASSES or r.get("space") != "query":
            continue
        ranking = [
            h for h in r["hits"]
            if h.get("live") and not h.get("self") and h.get("kind") == "decision"
        ]
        target_slugs = {t["slug"] for t in r["targets"]}
        for target in r["targets"]:
            if target.get("absent"):
                continue
            best_other: Optional[float] = None
            value: Optional[float] = None
            for h in ranking:
                if h["slug"] == target["slug"]:
                    value = h["value"]
                    break
                if h["slug"] not in target_slugs:
                    best_other = h["value"] if best_other is None else max(best_other, h["value"])
            if value is not None and best_other is not None:
                depths.append(best_other - value)
    return depths


def edge_margin_boundary(results: List[dict]) -> Optional[float]:
    """Returns one corpus's edge-margin boundary: the median inversion depth, 4 decimals.

    The median is nearest-rank (``nearest_rank``), the record's own convention, so the
    boundary can be reproduced by hand from the depths.

    Args:
        results: Fixture results (see ``inversion_depths``).

    Returns:
        The boundary, or ``None`` when the corpus shows no inversion at all.
    """
    median = nearest_rank(sorted(inversion_depths(results)), 50)
    return None if median is None else round(median, 4)


# ---------------------------------------------------------------------------
# The rejection-match floor's method (recall.REJECTION_MATCH_FLOOR)
# ---------------------------------------------------------------------------

# The forms whose owner's reason carries over: the floor's recall is read over these.
# ``elsewhere-does-not-carry`` is a labelled case, never a negative and never recall (D1).
CARRYING_FORMS: Tuple[str, ...] = ("own-question", "elsewhere-carries")


def _rejection_items(results: List[dict]) -> List[dict]:
    """The per-item shape of every rejection-target result, item-measured and labelled."""
    return [r["items"]["per_item"] for r in results
            if r.get("class") == "rejection-target" and r.get("measure") == "item"
            and isinstance(r.get("items"), dict) and "per_item" in r["items"]]


def rejection_floor_boundary(results: List[dict]) -> Optional[float]:
    """Returns one corpus's rejection-floor boundary: just above every hard negative.

    The hard negatives are every ``negative_items`` value of the per-item shape over the
    corpus's rejection-target results, both kinds. The boundary is the smallest
    4-decimal value strictly above their maximum, so ``fill_rejections``' ``score <
    floor`` keeps the maximum negative out of a section (5b D3: P14's "better silent
    than noisy", read strictly). A maximum already at 4 decimals moves up one step.

    Args:
        results: Fixture results (``derive_result`` plus ``items``); the committed Harbor
            run's ``fixtures`` and the dense runner's ``results.jsonl`` both hold them.

    Returns:
        The boundary, or ``None`` when no negative has a value.
    """
    values = [n["value"] for shape in _rejection_items(results) for n in shape["negatives"]
              if n.get("value") is not None]
    if not values:
        return None
    return _strictly_above(max(values))


def _strictly_above(top: float) -> float:
    """The smallest 4-decimal value strictly above ``top``.

    One rounding rule for every "strictly above" boundary and for the range check's upper
    envelope. A maximum already at 4 decimals moves up one step.
    """
    boundary = round((math.floor(top * 10_000) + 1) / 10_000, 4)
    if boundary <= top:  # float noise: an exact 4-place maximum floored one step low
        boundary = round(boundary + 0.0001, 4)
    elif round(boundary - 0.0001, 4) > top:  # float noise: a value just under a step floored onto it
        boundary = round(boundary - 0.0001, 4)
    return boundary


def _at_or_below(low: float) -> float:
    """The largest 4-decimal value at or below ``low``.

    The mirror of `_strictly_above`, for every "at or below" boundary and the range
    check's lower envelope. A minimum already at 4 decimals is its own boundary.
    """
    boundary = round(math.floor(low * 10_000) / 10_000, 4)
    step = round(boundary + 0.0001, 4)
    if step <= low:  # float noise: an exact 4-place minimum floored one step low
        boundary = step
    elif boundary > low:  # float noise: a value just under a 4-place step floored onto it
        boundary = round(boundary - 0.0001, 4)
    return boundary


def rejection_floor_retention(results: List[dict], floor: float) -> Optional[float]:
    """Returns the share of carrying targets a floor keeps: the boundary's recall.

    Over the per-item targets of the ``own-question`` and ``elsewhere-carries`` results
    (``CARRYING_FORMS``); an absent target counts as not kept. ``elsewhere-does-not-carry``
    is left out (D1).

    Returns:
        The share at or above ``floor``, or ``None`` when there is no carrying target.
    """
    values = [t["value"] for r in results
              if r.get("class") == "rejection-target" and r.get("rejection_form") in CARRYING_FORMS
              and isinstance(r.get("items"), dict) and "per_item" in r["items"]
              for t in r["items"]["per_item"]["targets"]]
    if not values:
        return None
    return sum(1 for v in values if v is not None and v >= floor) / len(values)


# ---------------------------------------------------------------------------
# The band's methods (recall.SURFACE_STRONG_THRESHOLD, SURFACE_WEAK_THRESHOLD,
# RANKED_TOP_CLUSTER_MARGIN), and the item ranking's analogues (6a D4)
# ---------------------------------------------------------------------------

# The four query-form classes the band separates. A rank-1 hit can be right only for an
# answering class; a non-answering one holds nothing that answers. "Related" is every
# band class but off-axis: the only class `none` may hold.
BAND_CLASSES: Tuple[str, ...] = ("bullseye", "handle-words", "family-neighbour", "off-axis")
ANSWERING_CLASSES: Tuple[str, ...] = EDGE_MARGIN_CLASSES
NON_ANSWERING_CLASSES: Tuple[str, ...] = ("family-neighbour", "off-axis")
RELATED_CLASSES: Tuple[str, ...] = ("bullseye", "handle-words", "family-neighbour")
BANDS: Tuple[str, ...] = ("strong", "weak", "none")
# Below this many rank-1-wrong depths, the top-cluster margin pools every inversion
# depth instead: a median of a handful is not a measured distribution (2e D2).
TOP_CLUSTER_MIN_DEPTHS = 10
# 1.96: a 95% Wilson interval.
WILSON_Z = 1.96


def _band_results(results: List[dict]) -> List[dict]:
    """The results the decision band reads: the four band classes, query space only."""
    return [r for r in results if r.get("class") in BAND_CLASSES and r.get("space") == "query"]


def _live_ranking(result: dict) -> List[dict]:
    """A result's ranked-read list: live, non-self decisions, best first."""
    return [h for h in result["hits"]
            if h.get("live") and not h.get("self") and h.get("kind") == "decision"]


def strong_boundary(results: List[dict]) -> Optional[float]:
    """Returns one corpus's ``strong`` boundary: strictly above every non-answer's top.

    The smallest 4-decimal value strictly above the largest ``top_live_value`` of any
    family-neighbour or off-axis result in query space, so ``strong`` never claims a
    precedent the corpus measures as a neighbour that does not answer (6a D1).

    Returns:
        The boundary, or ``None`` when no non-answering result has a top value.
    """
    tops = [r["top_live_value"] for r in _band_results(results)
            if r["class"] in NON_ANSWERING_CLASSES and r.get("top_live_value") is not None]
    return _strictly_above(max(tops)) if tops else None


def weak_boundary(results: List[dict]) -> Optional[float]:
    """Returns one corpus's ``weak`` boundary: at or below every related read's top.

    The largest 4-decimal value at or below the smallest ``top_live_value`` of any
    bullseye, handle-words or family-neighbour result in query space. Answering classes
    read their top value, not their target's, because the classifier reads the top. So
    ``none`` never fires where the corpus still sees a near precedent (6a D1, P14).

    Returns:
        The boundary, or ``None`` when no related result has a top value.
    """
    tops = [r["top_live_value"] for r in _band_results(results)
            if r["class"] in RELATED_CLASSES and r.get("top_live_value") is not None]
    return _at_or_below(min(tops)) if tops else None


def rank_one_wrong_depths(results: List[dict]) -> List[float]:
    """Measures how far below rank 1 the best target sits, where rank 1 is not a target.

    Over answering-class results in query space whose rank-1 live hit is not one of the
    fixture's targets: rank 1's value minus the best target's value in the live ranking.
    A result with no target in the ranking has nothing to measure and gives none.

    Returns:
        One depth per such result, in result order.
    """
    depths: List[float] = []
    for r in _band_results(results):
        if r["class"] not in ANSWERING_CLASSES:
            continue
        ranking = _live_ranking(r)
        targets = {t["slug"] for t in r["targets"]}
        if not ranking or ranking[0]["slug"] in targets:
            continue
        below = [h["value"] for h in ranking if h["slug"] in targets]
        if below:
            depths.append(ranking[0]["value"] - max(below))
    return depths


def top_cluster_depths(results: List[dict]) -> Tuple[str, List[float]]:
    """The population the top-cluster margin reads, and its name (6a D3).

    The rank-1-wrong depths (`rank_one_wrong_depths`) when there are at least
    ``TOP_CLUSTER_MIN_DEPTHS`` of them; otherwise every inversion depth of the answering
    classes (`inversion_depths`, 2e's population).

    Returns:
        ``("rank-one-wrong" | "pooled-inversions", depths)``.
    """
    wrong = rank_one_wrong_depths(results)
    if len(wrong) >= TOP_CLUSTER_MIN_DEPTHS:
        return "rank-one-wrong", wrong
    return "pooled-inversions", inversion_depths(results)


def top_cluster_margin_boundary(results: List[dict]) -> Optional[float]:
    """Returns one corpus's top-cluster margin: the median depth at which rank 1 is wrong.

    The nearest-rank median of `top_cluster_depths`, 4 decimals, as `edge_margin_boundary`
    rounds. Read on the dense corpus, where clusters form; Harbor's is recorded beside it.

    Returns:
        The boundary, or ``None`` when the population is empty.
    """
    median = nearest_rank(sorted(top_cluster_depths(results)[1]), 50)
    return None if median is None else round(median, 4)


def rank_one_margin(result: dict) -> Optional[float]:
    """Rank 1's score minus the next live hit's, from a stored result (6a D5 (ii)).

    A stored result holds the window only, so with one live hit there is no next one and
    the input is ``None``: its promotion rule never fires (it fails closed).
    """
    ranking = _live_ranking(result)
    return ranking[0]["value"] - ranking[1]["value"] if len(ranking) >= 2 else None


def overlap(query: str, slug: str, mechanisms: List[str], *,
            terms: Callable[[str], List[str]], stopwords: Any) -> Optional[float]:
    """The share of the query's terms a hit's slug and mechanisms carry (6a D5 (i)).

    ``|Q ∩ H| / |Q|``: ``Q`` is the query's terms minus ``stopwords``; ``H`` is the terms
    of the slug and every mechanism, hyphens read as spaces (both are kebab handles).
    The tokenizer is injected (``lexical._query_terms`` and ``lexical.DISPLAY_STOPWORDS``
    at every caller), so this module stays stdlib.

    Returns:
        The share, 4 decimals, or ``None`` when the query has no term left.
    """
    q = [t for t in terms(query) if t not in stopwords]
    if not q:
        return None
    handles = [slug] + [m for m in mechanisms if isinstance(m, str)]
    h = set(terms(" ".join(x.replace("-", " ") for x in handles)))
    return round(sum(1 for t in q if t in h) / len(q), 4)


def wilson(k: int, n: int, z: float = WILSON_Z) -> List[float]:
    """The Wilson score interval for ``k`` of ``n``, 4 decimals; ``[0.0, 1.0]`` at ``n == 0``.

    A rate over a handful is never quoted bare: the interval says how little it pins.
    """
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    denom = 1 + z * z / n
    return [round(max(0.0, (centre - spread) / denom), 4), round(min(1.0, (centre + spread) / denom), 4)]


def auc(positives: List[float], negatives: List[float]) -> Optional[float]:
    """The probability a positive scores above a negative, ties counting half, 4 decimals.

    How well one input separates two populations, whatever its scale (6a D5).

    Returns:
        The AUC, or ``None`` when either population is empty.
    """
    if not positives or not negatives:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return round(wins / (len(positives) * len(negatives)), 4)


def band_of(top: Optional[float], *, strong: float, weak: float) -> str:
    """The band a top value lands in under a given pair, as ``recall._classify_recall`` cuts.

    A result with no live hit surfaced nothing, which the classifier reads as ``none``.
    """
    if top is None or top < weak:
        return "none"
    return "strong" if top >= strong else "weak"


def _precision_table(rows: List[Tuple[Optional[float], bool, str]], *, strong: float, weak: float,
                     groups: Tuple[str, ...]) -> Dict[str, Any]:
    """Per band: fires, right, precision and its interval, counts by group; plus the base."""
    bands: Dict[str, Any] = {}
    for band in BANDS:
        mine = [(right, group) for top, right, group in rows if band_of(top, strong=strong, weak=weak) == band]
        fires, right = len(mine), sum(1 for ok, _ in mine if ok)
        bands[band] = {
            "fires": fires, "right": right,
            "precision": round(right / fires, 4) if fires else None,
            "interval": wilson(right, fires),
            "by_group": {g: sum(1 for _, x in mine if x == g) for g in groups},
        }
    n, right = len(rows), sum(1 for _, ok, _ in rows if ok)
    return {"n": n, "right": right, "base": round(right / n, 4) if n else None, "bands": bands,
            "strong": strong, "weak": weak}


def rank_one_right(result: dict) -> bool:
    """Whether a band result's rank-1 live hit is one of its targets (answering classes only)."""
    if result.get("class") not in ANSWERING_CLASSES:
        return False
    ranking = _live_ranking(result)
    return bool(ranking) and ranking[0]["slug"] in {t["slug"] for t in result["targets"]}


def band_precision(results: List[dict], *, strong: float, weak: float) -> Dict[str, Any]:
    """States each band's measured precision on one corpus under a given pair (6a D2).

    Over the four band classes in query space: a result lands in the band its
    ``top_live_value`` reads, and is *right* when its rank-1 live hit is a target, which
    a non-answering class never is. ``base`` is rank 1 right over every band result: what
    a band calling everything ``strong`` would score.

    Returns:
        ``{n, right, base, strong, weak, bands: {band: {fires, right, precision,
        interval, by_group}}}``, ``by_group`` counting by class. ``precision`` is
        ``None`` and the interval ``[0.0, 1.0]`` for a band with no fires.
    """
    rows = [(r.get("top_live_value"), rank_one_right(r), r["class"]) for r in _band_results(results)]
    return _precision_table(rows, strong=strong, weak=weak, groups=BAND_CLASSES)


def precision_entry(table: Dict[str, Any], band: str = "strong") -> Dict[str, Any]:
    """One run's record ``precision`` block for a band, from a `band_precision` table."""
    b = table["bands"][band]
    return {"fires": b["fires"], "right": b["right"], "precision": b["precision"],
            "interval": b["interval"], "base": table["base"]}


def _rejection_results(results: List[dict]) -> List[dict]:
    """Every rejection-target result with a per-item ranking, all three forms (6a D4)."""
    return [r for r in results
            if r.get("class") == "rejection-target" and r.get("measure") == "item"
            and isinstance(r.get("items"), dict) and "per_item" in r["items"]]


def _item_rank_one_right(shape: dict) -> bool:
    return any(t.get("rank") == 1 for t in shape["targets"])


def item_strong_boundary(results: List[dict]) -> Optional[float]:
    """The item ranking's ``strong`` boundary: strictly above every hard negative.

    The band rule's item analogue, read by the floor's own method (the item classes hold
    no fixture that nothing answers, so the labelled near-misses stand in for one).
    """
    return rejection_floor_boundary(results)


def item_weak_boundary(results: List[dict]) -> Optional[float]:
    """The item ranking's ``weak`` boundary: at or below every fixture's top item value.

    Over every rejection-target result, all three forms: the lookup's question is
    option-level, so an ``elsewhere-does-not-carry`` target is a right answer to it
    (6a D4). The top value is what the lookup's classifier reads.
    """
    tops = [r["items"]["per_item"]["top_value"] for r in _rejection_results(results)
            if r["items"]["per_item"].get("top_value") is not None]
    return _at_or_below(min(tops)) if tops else None


def item_band_precision(results: List[dict], *, strong: float, weak: float) -> Dict[str, Any]:
    """`band_precision` over the item ranking: right when the rank-1 item is a target item.

    Every rejection-target result, all three forms; ``by_group`` counts by form.
    """
    rows = [(r["items"]["per_item"].get("top_value"), _item_rank_one_right(r["items"]["per_item"]),
             r.get("rejection_form")) for r in _rejection_results(results)]
    return _precision_table(rows, strong=strong, weak=weak, groups=REJECTION_FORMS)


def item_rank_one_wrong_depths(results: List[dict]) -> List[float]:
    """`rank_one_wrong_depths` over the item ranking, every form: top value minus the best target's."""
    depths: List[float] = []
    for r in _rejection_results(results):
        shape = r["items"]["per_item"]
        values = [t["value"] for t in shape["targets"] if t.get("value") is not None]
        if values and shape.get("top_value") is not None and not _item_rank_one_right(shape):
            depths.append(shape["top_value"] - max(values))
    return depths


def item_inversion_depths(results: List[dict]) -> List[float]:
    """`inversion_depths` over the item ranking, every form.

    Targets carry their collapsed rank, not their ordinal, so a unit above a target is a
    non-target when no target holds its rank.
    """
    depths: List[float] = []
    for r in _rejection_results(results):
        shape = r["items"]["per_item"]
        units = _collapse(shape["hits"], per_item=True)
        target_ranks = {t["rank"] for t in shape["targets"] if t.get("rank") is not None}
        for t in shape["targets"]:
            if t.get("rank") is None or t.get("value") is None:
                continue
            above = [units[i]["value"] for i in range(t["rank"] - 1) if i + 1 not in target_ranks]
            if above:
                depths.append(max(above) - t["value"])
    return depths


def item_top_cluster_depths(results: List[dict]) -> Tuple[str, List[float]]:
    """`top_cluster_depths` over the item ranking, with the same fallback."""
    wrong = item_rank_one_wrong_depths(results)
    if len(wrong) >= TOP_CLUSTER_MIN_DEPTHS:
        return "rank-one-wrong", wrong
    return "pooled-inversions", item_inversion_depths(results)


def item_top_cluster_margin_boundary(results: List[dict]) -> Optional[float]:
    """`top_cluster_margin_boundary` over the item ranking."""
    median = nearest_rank(sorted(item_top_cluster_depths(results)[1]), 50)
    return None if median is None else round(median, 4)


def item_edge_margin_boundary(results: List[dict]) -> Optional[float]:
    """`edge_margin_boundary` over the item ranking: the median item inversion depth."""
    median = nearest_rank(sorted(item_inversion_depths(results)), 50)
    return None if median is None else round(median, 4)


# ---------------------------------------------------------------------------
# Record and run files
# ---------------------------------------------------------------------------

def _write_json(path: str, obj: Any) -> None:
    """Writes reviewable JSON atomically: indent 2, sorted keys, UTF-8, trailing newline."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def new_record(model_id: str) -> Dict[str, Any]:
    """Returns an empty record naming the embedding model it is calibrated for."""
    return {"schema": SCHEMA, "embedding_model": model_id, "runs": [], "constants": {}}


def load_record(record_dir: str = RECORD_DIR) -> Dict[str, Any]:
    """Loads ``record.json`` from ``record_dir``."""
    with open(os.path.join(record_dir, RECORD_FILE), encoding="utf-8") as f:
        return json.load(f)


def write_record(record: Dict[str, Any], record_dir: str = RECORD_DIR) -> None:
    """Writes ``record.json`` into ``record_dir`` (created if absent)."""
    os.makedirs(record_dir, exist_ok=True)
    _write_json(os.path.join(record_dir, RECORD_FILE), record)


def _run_path(run_id: str, record_dir: str) -> str:
    if not isinstance(run_id, str) or not run_id or os.sep in run_id or run_id.startswith("."):
        raise ValueError(f"not a run id: {run_id!r}")
    return os.path.join(record_dir, RUNS_DIRNAME, f"{run_id}.json")


def load_run(run_id: str, record_dir: str = RECORD_DIR) -> Dict[str, Any]:
    """Loads ``runs/<run_id>.json`` from ``record_dir``."""
    with open(_run_path(run_id, record_dir), encoding="utf-8") as f:
        return json.load(f)


def load_cited_runs(record: Dict[str, Any], record_dir: str = RECORD_DIR) -> Dict[str, Dict[str, Any]]:
    """Loads every run file the record's index lists; a missing file is simply absent."""
    runs: Dict[str, Dict[str, Any]] = {}
    for entry in record.get("runs", []):
        try:
            runs[entry["run_id"]] = load_run(entry["run_id"], record_dir)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return runs


def set_record_model(model_id: str, record_dir: str = RECORD_DIR) -> None:
    """The model-swap step: rewrites the record's ``embedding_model`` and nothing else.

    Old-model runs stay in ``runs`` as history. Every constant citing one then fails
    `check_pins` until it is re-derived on a new-model run, and `append_run` accepts
    new-model runs from here on, so a swap is one deliberate act followed by red rows.

    Args:
        model_id: The newly configured embedding model id.
        record_dir: The record folder.
    """
    record = load_record(record_dir)
    record["embedding_model"] = model_id
    write_record(record, record_dir)


def make_run_id(corpus: str, timestamp: str, commit_sha: Optional[str]) -> str:
    """Returns ``<corpus>-<UTC yyyymmdd>-<short sha>`` (``nosha`` without a commit).

    An aware timestamp is converted to UTC first; a naive one is taken as UTC.
    """
    moment = datetime.fromisoformat(timestamp)
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)
    day = moment.strftime("%Y%m%d")
    return f"{corpus}-{day}-{(commit_sha or 'nosha')[:7]}"


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _numeric_list(values: Any) -> bool:
    return isinstance(values, list) and all(_is_number(v) for v in values)


def _non_numeric_stats(block: Dict[str, Any], where: str) -> Optional[str]:
    """Names the first statistic in ``block`` that is not a number, or returns ``None``.

    ``values`` is a list of numbers; every other statistic is a number or ``null`` (an
    empty sub-block has no minimum). A string in ``min`` is as much a leak as one in
    ``values``, so the scalars are held too.
    """
    if not _numeric_list(block.get("values", [])):
        return f"`{where}values` holds a non-number"
    for key in _STATS_KEYS:
        if key != "values" and key in block and not (block[key] is None or _is_number(block[key])):
            return f"`{where}{key}` is not a number"
    return None


def _non_numeric_block(sub: Dict[str, Any]) -> Optional[str]:
    """Names what in a dense sub-block is not a numeric statistic, or returns ``None``."""
    extra = sorted(set(sub) - set(_CLASS_BLOCK_KEYS))
    if extra:
        return f"extra keys {extra}"
    problem = _non_numeric_stats(sub, "")
    if problem:
        return problem
    negatives = sub.get("negatives", {})
    if not isinstance(negatives, dict):
        return "`negatives` is not a statistics object"
    extra = sorted(set(negatives) - set(_STATS_KEYS))
    if extra:
        return f"`negatives` has extra keys {extra}"
    return _non_numeric_stats(negatives, "negatives.")


def _refuse_run(run: Dict[str, Any], record: Dict[str, Any]) -> None:
    corpus = run.get("corpus")
    if corpus not in CORPORA:
        raise ValueError(f"run corpus {corpus!r}; one of {list(CORPORA)}")
    count = run.get("decision_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError(f"run decision_count {count!r}; must be an integer of at least 1")
    classes = run.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ValueError("run has no labelled class; smoke results never enter the record")
    for cls, block in classes.items():
        if cls not in CLASSES:
            raise ValueError(f"run class {cls!r}; one of {list(CLASSES)}")
        if not isinstance(block, dict) or not block or any(
            space not in SPACES or not isinstance(sub, dict) or sub.get("measure") not in MEASURES
            for space, sub in block.items()
        ):
            raise ValueError(
                f"run class {cls!r} must map each space in {list(SPACES)} to a sub-block "
                f"naming its measure"
            )
    if "fixtures" not in run:
        raise ValueError("run carries no `fixtures` key")
    if corpus == "dense":
        if run["fixtures"] is not None:
            raise ValueError(
                "a dense run must carry `fixtures: null`; its texts are private and never "
                "enter the public repo"
            )
        extra = sorted(set(run) - set(_RUN_KEYS))
        if extra:
            raise ValueError(f"a dense run carries keys outside the run shape: {extra}")
        for cls, block in classes.items():
            for space, sub in block.items():
                problem = _non_numeric_block(sub)
                if problem:
                    raise ValueError(
                        f"dense run class {cls!r} in {space!r} space must hold numeric "
                        f"statistics only; {problem}"
                    )
    else:
        if not isinstance(run["fixtures"], list) or not run["fixtures"]:
            raise ValueError("a harbor run must carry its per-fixture results")
        for fx in run["fixtures"]:
            if fx.get("class") not in CLASSES:
                raise ValueError(f"run fixture {fx.get('id')!r} has class {fx.get('class')!r}")
            if "envelope_evidence" in fx:
                problem = (evidence_problem(fx["envelope_evidence"])
                           if fx.get("space") == SPACE_OF_FORM["query"]
                           else "a draft-form result carries envelope_evidence")
                if problem:
                    raise ValueError(f"run fixture {fx.get('id')!r}: {problem}")
    prov = run.get("provenance") or {}
    if not isinstance(prov.get("timestamp"), str):
        raise ValueError("run provenance carries no ISO-8601 `timestamp`")
    model = prov.get("embedding_model")
    if model != record.get("embedding_model"):
        raise ValueError(
            f"run model {model!r} differs from the record's {record.get('embedding_model')!r}; "
            f"a model swap goes through set_record_model first"
        )


def append_run(run: Dict[str, Any], record_dir: str = RECORD_DIR) -> str:
    """Records a run: writes ``runs/<id>.json`` and appends its summary entry.

    Refuses a run with no labelled class, a fixture outside ``CLASSES``, a model other
    than the record's, a harbor run without fixtures, a dense run carrying fixtures or
    anything beyond the aggregate shape, and a ``decision_count`` below 1. The id is ``<corpus>-<yyyymmdd>-<sha>``, suffixed
    ``-2``, ``-3`` … on a collision; an existing run file is never overwritten. The run
    file and the record entry are written both or neither.

    Args:
        run: The run dict (§3.4); its ``run_id``, if any, is replaced.
        record_dir: The record folder.

    Returns:
        The run id written.

    Raises:
        ValueError: Naming the refusal.
    """
    record = load_record(record_dir)
    _refuse_run(run, record)
    prov = run.get("provenance") or {}
    base = make_run_id(run["corpus"], prov["timestamp"], prov.get("commit_sha"))
    taken = {e.get("run_id") for e in record.get("runs", [])}
    runs_dir = os.path.join(record_dir, RUNS_DIRNAME)
    os.makedirs(runs_dir, exist_ok=True)
    run_id, n = base, 1
    while run_id in taken or os.path.exists(_run_path(run_id, record_dir)):
        n += 1
        run_id = f"{base}-{n}"

    body = dict(run)
    body["run_id"] = run_id
    run_path = _run_path(run_id, record_dir)
    with open(run_path, "x", encoding="utf-8") as f:
        json.dump(body, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    try:
        record.setdefault("runs", []).append({
            "run_id": run_id,
            "corpus": body["corpus"],
            "embedding_model": prov["embedding_model"],
            "decision_count": body["decision_count"],
        })
        write_record(record, record_dir)
    except BaseException:
        os.remove(run_path)
        raise
    return run_id


def latest_run_id(record: Dict[str, Any], corpus: str) -> Optional[str]:
    """Returns the newest indexed run of ``corpus``, or ``None``."""
    ids = [e["run_id"] for e in record.get("runs", []) if e.get("corpus") == corpus]
    return ids[-1] if ids else None


# ---------------------------------------------------------------------------
# Pin checks
# ---------------------------------------------------------------------------

def _summary_of_file(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": run.get("run_id"),
        "corpus": run.get("corpus"),
        "embedding_model": (run.get("provenance") or {}).get("embedding_model"),
        "decision_count": run.get("decision_count"),
    }


# Floors that arm an escalation, so read on traffic: each record entry carries a
# ``cry_wolf`` block (5b D5). 7a adds the tombstone floor.
ESCALATION_FLOORS: Tuple[str, ...] = ("mitos.recall.REJECTION_MATCH_FLOOR",)
# The block's keys, exactly: the ceiling and the ratio measured against it, its
# population, and the ids that let anyone re-read it. ``earned`` is the union of the two
# signals ``earned_label`` and ``earned_proxy`` count apart. Numbers and ids only.
CRY_WOLF_KEYS: Tuple[str, ...] = (
    "ceiling", "ratio", "appearances", "earned", "earned_label", "earned_proxy", "reads",
    "replay_run", "copy_state", "labels_sha256", "raised_from",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _is_count(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _check_cry_wolf(name: str, entry: Dict[str, Any], top: Optional[float]) -> List[str]:
    """Checks an escalation floor's ``cry_wolf`` block and its side; returns problems.

    The ratio is sections shown over sections earned, rounded to 4 places, and must sit
    at or under the ceiling; nothing shown is a ratio of 0. ``raised_from`` is ``null``
    for side ``higher`` and equals ``max(boundaries)`` (``top``) for side ``ceiling``.
    """
    block = entry.get("cry_wolf")
    if not isinstance(block, dict):
        return [f"{name}: an escalation floor carries a `cry_wolf` block (its ceiling and ratio)"]
    keys = set(block)
    if keys != set(CRY_WOLF_KEYS):
        return [f"{name}: `cry_wolf` keys must be exactly {sorted(CRY_WOLF_KEYS)}; "
                f"missing {sorted(set(CRY_WOLF_KEYS) - keys)}, extra {sorted(keys - set(CRY_WOLF_KEYS))}"]
    problems: List[str] = []
    for key in ("ceiling", "ratio"):
        if not _is_number(block[key]) or block[key] < 0:
            problems.append(f"{name}: `cry_wolf.{key}` must be a non-negative number")
    for key in ("appearances", "earned", "earned_label", "earned_proxy", "reads"):
        if not _is_count(block[key]):
            problems.append(f"{name}: `cry_wolf.{key}` must be a count")
    for key in ("replay_run", "copy_state"):
        if not isinstance(block[key], str) or not block[key]:
            problems.append(f"{name}: `cry_wolf.{key}` must name its run or state")
    if not isinstance(block["labels_sha256"], str) or not _SHA256_RE.fullmatch(block["labels_sha256"]):
        problems.append(f"{name}: `cry_wolf.labels_sha256` must be a 64-hex digest")
    if problems:
        return problems
    shown, earned = block["appearances"], block["earned"]
    if earned > shown or max(block["earned_label"], block["earned_proxy"]) > earned \
            or earned > block["earned_label"] + block["earned_proxy"] or shown > block["reads"]:
        problems.append(f"{name}: `cry_wolf` counts are inconsistent (earned is the union of label and "
                        f"proxy, never above appearances; appearances never above reads)")
    elif shown and not earned:
        problems.append(f"{name}: `cry_wolf` shows {shown} sections and none earned; no ratio sits "
                        f"under a ceiling, so the floor rises until nothing appears")
    else:
        expected = round(shown / earned, 4) if shown else 0.0
        if block["ratio"] != expected:
            problems.append(f"{name}: `cry_wolf.ratio` {block['ratio']!r} is not appearances / earned "
                            f"= {expected!r}")
        if block["ratio"] > block["ceiling"]:
            problems.append(f"{name}: cry-wolf ratio {block['ratio']!r} is over its ceiling "
                            f"{block['ceiling']!r}; the floor rises before it lands")
    side, raised = entry.get("side"), block["raised_from"]
    if side == "ceiling":
        if raised is None or raised != top:
            problems.append(f"{name}: side `ceiling` needs `cry_wolf.raised_from` == max(boundaries) = {top!r}")
    elif raised is not None:
        problems.append(f"{name}: `cry_wolf.raised_from` is set but the side is {side!r}; "
                        f"a raised floor takes side `ceiling`")
    return problems


# Constants whose entry states a measured precision (6a D2/D7): a ``precision`` block
# with one key per cited run. `strong` is the one band a verdict clause rests on.
PRECISION_STATED: Tuple[str, ...] = ("mitos.recall.SURFACE_STRONG_THRESHOLD",)
# The subset that also states it on traffic, under a ``replay`` key. The harvest holds no
# field-scoped read, so an item-ranking constant can never be measured there.
REPLAY_PRECISION: Tuple[str, ...] = ("mitos.recall.SURFACE_STRONG_THRESHOLD",)
PRECISION_KEYS: Tuple[str, ...] = ("fires", "right", "precision", "interval", "base")
REPLAY_EXTRA_KEYS: Tuple[str, ...] = ("run", "proxy")
PRECISION_TOLERANCE = 1e-3


def _check_precision_part(name: str, where: str, part: Any, extra: Tuple[str, ...]) -> List[str]:
    """Checks one run's precision figures; returns problems.

    ``fires >= right >= 0``; ``precision`` within `PRECISION_TOLERANCE` of ``right /
    fires``; an interval ``[low, high]`` with ``0 <= low <= high <= 1``; ``base`` a number
    in [0, 1], or ``null`` for a corpus with no fixture. A line above every fixture is a
    stated "never fires here": ``fires == 0`` holds only ``right == 0``, ``precision:
    null`` and ``interval: [0.0, 1.0]``, never a division.
    """
    label = f"{name}: `precision.{where}"
    if not isinstance(part, dict):
        return [f"{label}` is not an object"]
    keys, want = set(part), set(PRECISION_KEYS) | set(extra)
    if keys != want:
        return [f"{label}` keys must be exactly {sorted(want)}; missing {sorted(want - keys)}, "
                f"extra {sorted(keys - want)}"]
    problems: List[str] = []
    fires, right, precision = part["fires"], part["right"], part["precision"]
    if not (_is_count(fires) and _is_count(right) and right <= fires):
        return [f"{label}` needs counts with fires >= right >= 0"]
    interval = part["interval"]
    if not (isinstance(interval, list) and len(interval) == 2 and all(_is_number(x) for x in interval)
            and 0 <= interval[0] <= interval[1] <= 1):
        problems.append(f"{label}.interval` must be [low, high] with 0 <= low <= high <= 1")
    base = part["base"]
    if not (base is None or (_is_number(base) and 0 <= base <= 1)):
        problems.append(f"{label}.base` must be a rate in [0, 1] or null")
    if fires == 0:
        if right != 0 or precision is not None or interval != [0.0, 1.0]:
            problems.append(f"{label}` fires nowhere, so it states right 0, precision null and "
                            f"interval [0.0, 1.0]")
    elif not _is_number(precision) or abs(precision - right / fires) > PRECISION_TOLERANCE:
        problems.append(f"{label}.precision` {precision!r} is not right / fires = {right / fires!r}")
    for key in extra:
        if not isinstance(part[key], str) or not part[key]:
            problems.append(f"{label}.{key}` must name its run or proxy")
    return problems


def _check_precision(name: str, entry: Dict[str, Any], cited: List[str]) -> List[str]:
    """Checks a `PRECISION_STATED` entry's ``precision`` block; returns problems.

    One key per cited run, plus ``replay`` exactly when the name is in `REPLAY_PRECISION`.
    """
    block = entry.get("precision")
    if not isinstance(block, dict):
        return [f"{name}: states its precision in a `precision` block, one key per cited run"]
    want = set(cited) | ({"replay"} if name in REPLAY_PRECISION else set())
    problems: List[str] = []
    for key in sorted(want - set(block)):
        problems.append(f"{name}: `precision` has no {key!r} key")
    for key in sorted(set(block) - want):
        if key == "replay":
            problems.append(f"{name}: carries a `precision.replay` key but is not measured on "
                            f"traffic (REPLAY_PRECISION is {list(REPLAY_PRECISION)})")
        else:
            problems.append(f"{name}: `precision` names {key!r}, which the entry does not cite")
    for key in sorted(want & set(block)):
        problems.extend(_check_precision_part(
            name, key, block[key], REPLAY_EXTRA_KEYS if key == "replay" else ()))
    return problems


def _check_constant(
    name: str, entry: Any, record: Dict[str, Any], runs: Dict[str, Dict[str, Any]]
) -> List[str]:
    if not isinstance(entry, dict):
        return [f"{name}: record entry is not an object"]
    problems: List[str] = []
    kind, side, space = entry.get("kind"), entry.get("side"), entry.get("space")
    value = entry.get("value")
    if kind not in KINDS:
        problems.append(f"{name}: unknown kind {kind!r}; one of {list(KINDS)}")
    if side not in SIDES:
        problems.append(f"{name}: unknown side {side!r}; one of {list(SIDES)}")
    if space not in SPACES:
        problems.append(f"{name}: unknown space {space!r}; one of {list(SPACES)}")
    separates = entry.get("separates")
    if not isinstance(separates, list):
        separates = []
    if not separates or any(c not in CLASSES for c in separates):
        problems.append(f"{name}: `separates` must list classes from {list(CLASSES)}")
        separates = [c for c in separates if c in CLASSES]
    boundaries = entry.get("boundaries")
    if not isinstance(boundaries, dict) or not boundaries:
        return problems + [f"{name}: `boundaries` is empty; a constant cites the runs it was set from"]

    index = {e.get("run_id"): e for e in record.get("runs", []) if isinstance(e, dict)}
    dense_cited = []
    for run_id, boundary in boundaries.items():
        if run_id not in index:
            problems.append(f"{name}: cites run {run_id!r}, which is not in the record's index")
            continue
        run = runs.get(run_id)
        if run is None:
            problems.append(f"{name}: cited run {run_id!r} has no run file")
            continue
        summary = index[run_id]
        on_file = _summary_of_file(run)
        for field in _SUMMARY_FIELDS:
            if summary.get(field) != on_file[field]:
                problems.append(
                    f"{name}: run {run_id!r} index entry says {field}={summary.get(field)!r}, "
                    f"its file says {on_file[field]!r}"
                )
        if on_file["embedding_model"] != record.get("embedding_model"):
            problems.append(
                f"{name}: cited run {run_id!r} was measured on {on_file['embedding_model']!r}, "
                f"not the record's {record.get('embedding_model')!r}; re-derive it on a new run"
            )
        if run.get("corpus") == "dense":
            dense_cited.append(run_id)
        classes = run.get("classes") or {}
        lows, highs = [], []
        for cls in separates:
            block = classes.get(cls)
            if not isinstance(block, dict):
                problems.append(f"{name}: class {cls!r} is absent from cited run {run_id!r}")
                continue
            # The space is the sub-block's key: a constant reads its own space or nothing.
            sub = block.get(space)
            if not isinstance(sub, dict):
                problems.append(
                    f"{name}: class {cls!r} has no values in {space!r} space in run "
                    f"{run_id!r}; the constant compares {space!r}"
                )
                continue
            # A floor sits between a class's negatives and its targets, so both bound it.
            for part in (sub, sub.get("negatives") or {}):
                if part.get("min") is not None:
                    lows.append(part["min"])
                if part.get("max") is not None:
                    highs.append(part["max"])
        # The range is the 4-decimal envelope of the classes' extremes, by the boundary
        # functions' own rounding: a correct `weak` floors just below its class minimum,
        # and a correct strictly-above boundary can sit one step past a class maximum.
        envelope = (_at_or_below(min(lows)), _strictly_above(max(highs))) if lows and highs else None
        if kind == "threshold" and envelope and not (envelope[0] <= boundary <= envelope[1]):
            problems.append(
                f"{name}: boundary {boundary!r} for run {run_id!r} lies outside its classes' "
                f"observed range [{envelope[0]!r}, {envelope[1]!r}] (4-decimal envelope)"
            )
        # A raised floor is decoupled from its boundaries, so its value is held to every
        # cited run's range itself: one above every target separates nothing (D3's case).
        # A non-number value is reported once, by the side check below, never raised here.
        if side == "ceiling" and _is_number(value) and envelope \
                and not (envelope[0] <= value <= envelope[1]):
            problems.append(
                f"{name}: raised value {value!r} lies outside run {run_id!r}'s observed range "
                f"[{envelope[0]!r}, {envelope[1]!r}] (4-decimal envelope)"
            )

    if side == "higher" and value != max(boundaries.values()):
        problems.append(f"{name}: side `higher` needs value == max(boundaries) = {max(boundaries.values())!r}")
    elif side == "lower" and value != min(boundaries.values()):
        problems.append(f"{name}: side `lower` needs value == min(boundaries) = {min(boundaries.values())!r}")
    elif side == "dense":
        if len(dense_cited) != 1:
            problems.append(f"{name}: side `dense` needs exactly one cited dense run, found {len(dense_cited)}")
        elif value != boundaries[dense_cited[0]]:
            problems.append(f"{name}: side `dense` needs value == boundaries[{dense_cited[0]!r}]")
    elif side == "ceiling":
        if name not in ESCALATION_FLOORS:
            problems.append(f"{name}: side `ceiling` belongs to an escalation floor ({list(ESCALATION_FLOORS)})")
        elif not (_is_number(value) and value > max(boundaries.values())):
            problems.append(f"{name}: side `ceiling` needs value > max(boundaries) = {max(boundaries.values())!r}")
    if name in ESCALATION_FLOORS:
        top = max(boundaries.values()) if all(_is_number(b) for b in boundaries.values()) else None
        problems.extend(_check_cry_wolf(name, entry, top))
    elif "cry_wolf" in entry:
        problems.append(f"{name}: carries a `cry_wolf` block but arms no escalation "
                        f"(ESCALATION_FLOORS is {list(ESCALATION_FLOORS)})")
    if name in PRECISION_STATED:
        problems.extend(_check_precision(name, entry, list(boundaries)))
    elif "precision" in entry:
        problems.append(f"{name}: carries a `precision` block but states no precision "
                        f"(PRECISION_STATED is {list(PRECISION_STATED)})")
    return problems


def check_pins(
    record: Dict[str, Any], runs: Dict[str, Dict[str, Any]], constants: Dict[str, float], model_id: str
) -> List[str]:
    """Checks the calibrated constants against the committed record, in both directions.

    Never raises: every problem comes back as one string, and an empty list means clean.
    Values compare by exact ``==``, because the record stores the literal the source holds.

    Args:
        record: The parsed record.
        runs: Run files by id (see `load_cited_runs`).
        constants: The calibrated set's runtime values, by dotted name.
        model_id: The embedding model ``models.py`` configures.

    Returns:
        Problems, one string each.
    """
    problems: List[str] = []
    try:
        if record.get("embedding_model") != model_id:
            problems.append(
                f"record names model {record.get('embedding_model')!r}; models.py configures "
                f"{model_id!r}. Run set_record_model, re-run Layer C, re-derive every constant"
            )
        recorded = record.get("constants")
        if not isinstance(recorded, dict):
            return problems + ["record has no `constants` object"]
        for name in sorted(set(constants) - set(recorded)):
            problems.append(f"{name}: in the calibrated set but absent from the record")
        for name in sorted(set(recorded) - set(constants)):
            problems.append(f"{name}: in the record but absent from the calibrated set")
        for name in sorted(set(constants) & set(recorded)):
            entry = recorded[name]
            if isinstance(entry, dict) and constants[name] != entry.get("value"):
                problems.append(
                    f"{name}: source holds {constants[name]!r}, record holds {entry.get('value')!r}"
                )
        for name in sorted(recorded):
            try:
                problems.extend(_check_constant(name, recorded[name], record, runs))
            except Exception as exc:  # one malformed entry must not hide the others
                problems.append(f"{name}: entry could not be checked: {type(exc).__name__}: {exc}")
    except Exception as exc:  # a malformed record is a problem to report, never a crash
        problems.append(f"record could not be checked: {type(exc).__name__}: {exc}")
    return problems


def check_runs(record: Dict[str, Any], record_dir: str = RECORD_DIR) -> List[str]:
    """Checks every indexed run against its file, cited by a constant or not.

    `check_pins` reaches run files only through a constant's citations, so an uncited
    run (a baseline no constant reads yet) would otherwise go unchecked. For each index
    entry this checks that ``runs/<run_id>.json`` exists and parses, passes the refusals
    `append_run` applied when it was written, and agrees with its entry on the summary
    fields. It also reports any file under ``runs/`` the index does not name.

    A run is held to its own entry's model, not the record's: after `set_record_model`
    the old-model runs stay as history, and the constants citing them are `check_pins`'s
    to report.

    Never raises: every problem comes back as one string, and an empty list means clean.

    Args:
        record: The parsed record.
        record_dir: The record folder.

    Returns:
        Problems, one string each.
    """
    problems: List[str] = []
    try:
        entries = record.get("runs")
        if not isinstance(entries, list):
            return ["record has no `runs` list"]
        named = set()
        for entry in entries:
            if not isinstance(entry, dict):
                problems.append(f"index entry {entry!r} is not an object")
                continue
            run_id = entry.get("run_id")
            try:
                path = _run_path(run_id, record_dir)
            except ValueError as exc:
                problems.append(f"index entry: {exc}")
                continue
            if run_id in named:
                problems.append(f"run {run_id!r} is indexed more than once")
                continue
            named.add(run_id)
            if not os.path.exists(path):
                problems.append(f"run {run_id!r} is indexed but has no run file")
                continue
            try:
                run = load_run(run_id, record_dir)
            except (OSError, ValueError) as exc:
                problems.append(f"run {run_id!r} has a run file that does not parse: {exc}")
                continue
            if not isinstance(run, dict):
                problems.append(f"run {run_id!r} has a run file that is not an object")
                continue
            # One malformed file (a list where an object belongs) is that run's problem;
            # it must not hide the runs after it or the orphan sweep.
            try:
                _refuse_run(run, dict(record, embedding_model=entry.get("embedding_model")))
            except ValueError as exc:
                problems.append(f"run {run_id!r} would be refused today: {exc}")
            except Exception as exc:
                problems.append(f"run {run_id!r} is malformed: {type(exc).__name__}: {exc}")
                continue
            try:
                on_file = _summary_of_file(run)
            except Exception as exc:
                problems.append(f"run {run_id!r} is malformed: {type(exc).__name__}: {exc}")
                continue
            for field in _SUMMARY_FIELDS:
                if entry.get(field) != on_file[field]:
                    problems.append(
                        f"run {run_id!r} index entry says {field}={entry.get(field)!r}, "
                        f"its file says {on_file[field]!r}"
                    )
        runs_dir = os.path.join(record_dir, RUNS_DIRNAME)
        on_disk = sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []
        for name in on_disk:
            if name.endswith(".json") and name[: -len(".json")] not in named:
                problems.append(f"run file {name!r} is not in the record's index")
    except Exception as exc:  # a malformed record is a problem to report, never a crash
        problems.append(f"runs could not be checked: {type(exc).__name__}: {exc}")
    return problems


# ---------------------------------------------------------------------------
# The population that owes calibration
# ---------------------------------------------------------------------------

def _module_level_assignments(body: List[ast.stmt]):
    """Yields module-level Assign/AnnAssign/AugAssign nodes, descending into if/try/with bodies."""
    for stmt in body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            yield stmt
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        else:
            for field in ("body", "orelse", "finalbody"):
                yield from _module_level_assignments(getattr(stmt, field, []) or [])
            for handler in getattr(stmt, "handlers", []) or []:
                yield from _module_level_assignments(handler.body)


def _target_names(target: ast.expr) -> List[Tuple[str, bool]]:
    """Returns ``(name, direct)`` pairs; ``direct`` is False inside a tuple target."""
    if isinstance(target, ast.Name):
        return [(target.id, True)]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [(n, False) for elt in target.elts for n, _ in _target_names(elt)]
    return []


def _literal_in_unit_range(value: Optional[ast.expr]) -> Optional[float]:
    if isinstance(value, ast.Constant) and type(value.value) is float and 0.0 <= value.value <= 1.0:
        return value.value
    return None


def threshold_population(mitos_dir: str) -> Dict[str, Optional[float]]:
    """Derives, from source, which shipped constants are thresholds owing calibration.

    Walks ``mitos_dir`` recursively and reads every module-level ``Assign`` or
    ``AnnAssign`` whose name ends ``_THRESHOLD``, ``_FLOOR`` or ``_MARGIN`` — by name
    first, so a non-float value cannot drop a name out of the population. Each name maps
    to its value when that value is a float literal in [0, 1], and to ``None`` otherwise
    (an alias, an int, an expression, an unannotated tuple member, or a name assigned
    more than once, an augmented assignment such as ``+=`` included). Reads the AST
    only; it never imports what it reads.

    Args:
        mitos_dir: The package directory (``…/mitos``); its basename is the dotted prefix.

    Returns:
        ``{dotted_name: float | None}``, sorted by name.
    """
    package = os.path.basename(os.path.normpath(mitos_dir))
    found: Dict[str, Optional[float]] = {}
    seen: Dict[str, int] = {}
    for root, dirs, files in os.walk(mitos_dir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, mitos_dir)[: -len(".py")].split(os.sep)
            if rel[-1] == "__init__":
                rel = rel[:-1]
            module = ".".join([package] + rel)
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=path)
            for stmt in _module_level_assignments(tree.body):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                # An augmented assignment only ever rebinds; it never holds the literal.
                augmented = isinstance(stmt, ast.AugAssign)
                for target in targets:
                    for name, direct in _target_names(target):
                        if not name.endswith(THRESHOLD_SUFFIXES):
                            continue
                        dotted = f"{module}.{name}"
                        seen[dotted] = seen.get(dotted, 0) + 1
                        found[dotted] = (
                            _literal_in_unit_range(stmt.value) if direct and not augmented else None
                        )
    for dotted, count in seen.items():
        if count > 1:
            found[dotted] = None
    return dict(sorted(found.items()))


# ---------------------------------------------------------------------------
# Soft diff
# ---------------------------------------------------------------------------

def _item_diff(fid: str, cur: Any, base: Any, band: float) -> List[Dict[str, Any]]:
    """Diffs two results' ``items``, per shape and role, keyed by ``(owner, option)``."""
    out: List[Dict[str, Any]] = []
    if not cur and not base:
        return out
    cur, base = cur or {}, base or {}
    for shape in SHAPES:
        for role in ("targets", "negatives"):
            c_refs = {(e["owner"], e["option"]): e for e in (cur.get(shape) or {}).get(role, [])}
            b_refs = {(e["owner"], e["option"]): e for e in (base.get(shape) or {}).get(role, [])}
            where = {"id": fid, "shape": shape, "role": role[:-1]}
            for owner, option in sorted(set(c_refs) - set(b_refs)):
                out.append(dict(where, kind="item_added", owner=owner, option=option, regression=False))
            for owner, option in sorted(set(b_refs) - set(c_refs)):
                out.append(dict(where, kind="item_removed", owner=owner, option=option, regression=False))
            for key in sorted(set(c_refs) & set(b_refs)):
                was, now = b_refs[key].get("value"), c_refs[key].get("value")
                kind = None
                if was is None and now is not None:
                    kind = "target_present"
                elif was is not None and now is None:
                    kind = "target_absent"
                elif was is not None and abs(now - was) > band:
                    kind = "value_moved"
                if kind:
                    out.append(dict(where, kind=kind, owner=key[0], option=key[1],
                                    regression=True, baseline=was, current=now))
    return out


def soft_diff(current: Dict[str, Any], baseline: Dict[str, Any], band: float = VALUE_BAND) -> List[Dict[str, Any]]:
    """Compares a run with a baseline run, fixture by fixture and target by target.

    Reports a target value moved past ``band`` (and, for a ``top_live`` fixture, its top
    live value), a changed envelope band, and a target that became absent or present —
    each a ``regression: true`` entry for human review. An item-measured fixture's
    ``items`` diff the same way in each shape, targets and negatives alike, keyed by
    ``(owner, option)`` and tagged with ``shape`` and ``role``. Fixtures added or removed,
    and targets or item refs added to or removed from a fixture, are listed with
    ``regression: false``. Never raises and never writes.

    Args:
        current: A run dict.
        baseline: The run it is compared with.
        band: The per-value tolerance.

    Returns:
        Diff entries, each ``{kind, id, regression, …}``.
    """
    out: List[Dict[str, Any]] = []
    try:
        cur = {f["id"]: f for f in current.get("fixtures") or []}
        base = {f["id"]: f for f in baseline.get("fixtures") or []}
        for fid in sorted(set(cur) - set(base)):
            out.append({"kind": "fixture_added", "id": fid, "regression": False})
        for fid in sorted(set(base) - set(cur)):
            out.append({"kind": "fixture_removed", "id": fid, "regression": False})
        for fid in sorted(set(cur) & set(base)):
            c, b = cur[fid], base[fid]
            if c.get("envelope_band") != b.get("envelope_band"):
                out.append({"kind": "band_changed", "id": fid, "regression": True,
                            "baseline": b.get("envelope_band"), "current": c.get("envelope_band")})
            pairs = []
            b_targets = {t["slug"]: t for t in b.get("targets", [])}
            c_targets = {t["slug"]: t for t in c.get("targets", [])}
            for slug in sorted(set(c_targets) - set(b_targets)):
                out.append({"kind": "target_added", "id": fid, "slug": slug, "regression": False})
            for slug in sorted(set(b_targets) - set(c_targets)):
                out.append({"kind": "target_removed", "id": fid, "slug": slug, "regression": False})
            for t in c.get("targets", []):
                if t["slug"] in b_targets:
                    pairs.append((t["slug"], b_targets[t["slug"]].get("value"), t.get("value")))
            if c.get("measure") == "top_live":
                pairs.append((None, b.get("top_live_value"), c.get("top_live_value")))
            for slug, was, now in pairs:
                if was is None and now is not None:
                    out.append({"kind": "target_present", "id": fid, "slug": slug,
                                "regression": True, "baseline": was, "current": now})
                elif was is not None and now is None:
                    out.append({"kind": "target_absent", "id": fid, "slug": slug,
                                "regression": True, "baseline": was, "current": now})
                elif was is not None and abs(now - was) > band:
                    out.append({"kind": "value_moved", "id": fid, "slug": slug,
                                "regression": True, "baseline": was, "current": now})
            out.extend(_item_diff(fid, c.get("items"), b.get("items"), band))
    except Exception as exc:  # a malformed run is reported, never raised
        out.append({"kind": "unreadable", "id": None, "regression": True,
                    "detail": f"{type(exc).__name__}: {exc}"})
    return out
