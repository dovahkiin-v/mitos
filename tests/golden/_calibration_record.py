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
from typing import Any, Dict, List, Optional, Tuple

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
# The two rejection-index shapes ruling 2 compares: one point per rejected option, and one
# point per decision holding the whole field. Not ``mitos.rejections.SHAPES``.
SHAPES: Tuple[str, ...] = ("per_item", "per_decision")
SIDES: Tuple[str, ...] = ("higher", "lower", "dense", "unmoved")
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


def _is_item_ref_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(r, dict) and set(r) == {"owner", "option"}
        and all(isinstance(r[k], str) and r[k].strip() for k in ("owner", "option"))
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
    for key in _ITEM_REF_KEYS:
        if key in fx and not _is_item_ref_list(fx[key]):
            raise ValueError(
                f"fixture {label}: `{key}` must be a list of {{owner, option}} objects, "
                f"both non-empty strings and nothing else"
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


def derive_result(
    fixture: dict, hits: List[dict], envelope_band: Optional[str], pause_floor: Optional[float]
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

    Returns:
        The JSON-safe result dict. A target outside the window has ``value: null`` and
        ``absent: true``; it is never dropped.

    Raises:
        ValueError: If the band or floor is supplied for the wrong form.
    """
    query_form = fixture["form"] == "query"
    if not query_form and envelope_band is not None:
        raise ValueError(f"fixture {fixture['id']!r}: a draft read has no envelope band")
    if query_form and pause_floor is not None:
        raise ValueError(f"fixture {fixture['id']!r}: a query read has no pause floor")
    # An open question is never live, whatever the caller labelled: both shipped readers
    # drop non-decisions before they rank.
    raw = [
        {"slug": h["slug"], "value": h["value"], "kind": h["kind"],
         "live": bool(h["live"]) and h["kind"] == "decision", "self": bool(h.get("self", False))}
        for h in hits
    ]
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
    """Derives a rejection-target fixture's ``items`` value in both index shapes (§3.3).

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

    Args:
        fixture: A validated rejection-target fixture.
        hits_by_shape: The labelled hits per shape in ``SHAPES``.
        refs: The resolved refs.

    Returns:
        ``{shape: {hits, targets, negatives, top_value}}``. A ref no hit carries is kept
        with ``value: null`` and ``absent: true``.

    Raises:
        ValueError: If the fixture is not item-measured or a shape is missing.
    """
    if fixture["measure"] != "item":
        raise ValueError(f"fixture {fixture['id']!r}: only an item-measured fixture has items")
    missing = [s for s in SHAPES if s not in hits_by_shape]
    if missing:
        raise ValueError(f"fixture {fixture['id']!r}: no hits for shapes {missing}")
    return {shape: _item_shape(fixture, hits_by_shape[shape], refs, shape == "per_item")
            for shape in SHAPES}


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
    ``target_rank_one`` (the share of target refs ranked first, over every target ref,
    absent ones included) and ``margin`` (statistics of ``margin_vs_best_other``).

    Args:
        results: Every fixture result of the run; only ``measure: item`` ones are read.

    Returns:
        ``{shape: {targets, negatives, by_form, target_rank_one, margin}}`` for each shape
        in ``SHAPES``, or ``{}`` when the run holds no item-measured result.
    """
    items = [r for r in results if r.get("measure") == "item" and r.get("class") is not None]
    if not items:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for shape in SHAPES:
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
            "target_rank_one": (sum(1 for t in targets if t["rank"] == 1) / len(targets)) if targets else None,
            "margin": stats([t["margin_vs_best_other"] for t in targets]),
        }
    return out


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


def _numeric_list(values: Any) -> bool:
    return isinstance(values, list) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
    )


def _non_numeric_block(sub: Dict[str, Any]) -> Optional[str]:
    """Names what in a dense sub-block is not a numeric statistic, or returns ``None``."""
    extra = sorted(set(sub) - set(_CLASS_BLOCK_KEYS))
    if extra:
        return f"extra keys {extra}"
    if not _numeric_list(sub.get("values", [])):
        return "`values` holds a non-number"
    negatives = sub.get("negatives", {})
    if not isinstance(negatives, dict):
        return "`negatives` is not a statistics object"
    extra = sorted(set(negatives) - set(_STATS_KEYS))
    if extra:
        return f"`negatives` has extra keys {extra}"
    if not _numeric_list(negatives.get("values", [])):
        return "`negatives.values` holds a non-number"
    return None


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
        if kind == "threshold" and lows and highs and not (min(lows) <= boundary <= max(highs)):
            problems.append(
                f"{name}: boundary {boundary!r} for run {run_id!r} lies outside its classes' "
                f"observed range [{min(lows)!r}, {max(highs)!r}]"
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
