"""Offline rows for Layer C's pure core (`_calibration_record`), on synthetic inputs.

No service is touched: fixtures, hits, runs and records are built here, and the record
folder is a ``tmp_path``. These rows pin the rules a later phase writes against: the
fixture shape, the open-question and self-hit exclusions, nearest-rank statistics, what
``append_run`` refuses, every problem ``check_pins`` reports, the threshold population
and the soft diff.
"""

import json
import os
import subprocess
import sys

import pytest

GOLDEN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, GOLDEN)
import _calibration_record as R  # noqa: E402

MODEL = "model-a"
TS = "2026-10-16T08:00:00+00:00"
SHA = "9dd49d4aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
STRONG = "mitos.recall.SURFACE_STRONG_THRESHOLD"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _fx(**over):
    fx = {"id": "bullseye-one", "class": "bullseye", "form": "query", "text": "a question",
          "targets": ["d2"], "negatives": [], "measure": "target", "notes": "why"}
    fx.update(over)
    return fx


def _hit(slug, value, kind="decision", live=True, self=False):
    return {"slug": slug, "value": value, "kind": kind, "live": live, "self": self}


def _class_block(space, values, measure="target"):
    values = sorted(values)
    return {"space": space, "measure": measure, "n": len(values), "absent": 0, "values": values,
            "min": values[0], "p10": values[0], "median": values[len(values) // 2],
            "p90": values[-1], "max": values[-1]}


def _run(corpus="harbor", model=MODEL, decision_count=29, classes=None, fixtures="default", ts=TS):
    if classes is None:
        classes = {"bullseye": _class_block("query", [0.78, 0.84, 0.9]),
                   "family-neighbour": _class_block("query", [0.5, 0.62, 0.72])}
    if fixtures == "default":
        fixtures = None if corpus == "dense" else [{"id": "f1", "class": "bullseye", "targets": []}]
    return {"corpus": corpus, "decision_count": decision_count, "node_count": decision_count + 2,
            "provenance": {"embedding_model": model, "timestamp": ts, "commit_sha": SHA},
            "fixtures": fixtures, "classes": classes}


@pytest.fixture
def record_dir(tmp_path):
    d = str(tmp_path / "calibration")
    R.write_record(R.new_record(MODEL), d)
    return d


def _clean(record_dir):
    """A record with a harbor and a dense run and one consistent constant."""
    harbor = R.append_run(_run(), record_dir)
    dense = R.append_run(_run(corpus="dense", decision_count=500), record_dir)
    record = R.load_record(record_dir)
    record["constants"][STRONG] = {
        "value": 0.75, "kind": "threshold", "space": "query",
        "separates": ["bullseye", "family-neighbour"],
        "boundaries": {harbor: 0.75, dense: 0.73}, "side": "higher",
        "set_by": "6a", "reason": "synthetic",
    }
    R.write_record(record, record_dir)
    return record, R.load_cited_runs(record, record_dir), {STRONG: 0.75}, harbor, dense


def _problems(record, runs, constants, model=MODEL):
    return R.check_pins(record, runs, constants, model)


def _has(problems, needle):
    return any(needle in p for p in problems)


# ---------------------------------------------------------------------------
# Paths and import closure
# ---------------------------------------------------------------------------

def test_golden_cache_path_is_model_keyed_under_the_golden_cache():
    path = R.golden_cache_path("gemini-embedding-2")
    assert os.path.basename(path) == "embeddings-gemini-embedding-2.sqlite"
    assert os.path.dirname(path) == os.path.join(GOLDEN, ".cache")


def test_the_pure_module_imports_from_outside_the_repo_and_only_the_stdlib(tmp_path):
    # What the import adds, not what interpreter startup brought (an editable finder).
    probe = (
        "import sys; before = set(sys.modules); sys.path.insert(0, %r); "
        "import _calibration_record; "
        "names = {m.split('.')[0] for m in set(sys.modules) - before}; "
        "print(sorted(n for n in names if n not in sys.stdlib_module_names "
        "and n != '_calibration_record'))" % GOLDEN
    )
    out = subprocess.run([sys.executable, "-I", "-c", probe], cwd=str(tmp_path),
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("over, needle", [
    ({"class": "neighbour"}, "unknown class"),
    ({"form": "sentence"}, "unknown form"),
    ({"measure": "best"}, "unknown measure"),
    ({"space": "query"}, "authors `space`"),
    ({"targets": "d2"}, "`targets` must be a list"),
    ({"negatives": [""]}, "`negatives` must be a list"),
    ({"id": "Not Kebab"}, "kebab-case"),
    ({"id": "kebab-case\n"}, "kebab-case"),
    ({"notes": " "}, "`notes`"),
    ({"extra": 1}, "unknown keys"),
    ({"class": None}, "`class` is null"),
    ({"measure": "target", "targets": []}, "measures nothing"),
    ({"exclude": ["d2"]}, "both a target and excluded"),
])
def test_the_validator_refuses_a_malformed_fixture_and_names_it(over, needle):
    with pytest.raises(ValueError) as err:
        R.validate_fixtures([_fx(**over)])
    assert needle in str(err.value)
    assert repr(_fx(**over)["id"]) in str(err.value)


def test_the_validator_refuses_a_duplicate_id():
    with pytest.raises(ValueError, match="'bullseye-one': duplicate id"):
        R.validate_fixtures([_fx(), _fx()])


def test_an_unlabelled_fixture_passes_only_when_classes_are_not_required():
    assert R.validate_fixtures([_fx(**{"class": None})], require_class=False)


def test_load_fixtures_reads_a_file_and_the_committed_one_validates(tmp_path):
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps({"_doc": {}, "fixtures": [_fx(), _fx(id="off-axis-one", **{
        "class": "off-axis", "targets": [], "measure": "top_live"})]}), encoding="utf-8")
    assert [f["id"] for f in R.load_fixtures(str(path))] == ["bullseye-one", "off-axis-one"]
    assert isinstance(R.load_fixtures(), list)


def test_check_fixture_slugs_refuses_a_slug_the_graph_lacks():
    R.check_fixture_slugs([_fx()], {"d2"})
    with pytest.raises(ValueError, match="bullseye-one.*d9"):
        R.check_fixture_slugs([_fx(negatives=["d9"])], {"d2"})


def test_the_space_follows_the_form():
    assert R.SPACE_OF_FORM == {"query": "query", "draft": "document"}
    assert R.derive_result(_fx(form="draft"), [_hit("d2", 0.8)], None, 0.8)["space"] == "document"


# ---------------------------------------------------------------------------
# derive_result
# ---------------------------------------------------------------------------

_BASE = [_hit("d1", 0.90), _hit("d2", 0.80), _hit("d3", 0.70)]


def _comparable(result):
    t = result["targets"][0]
    return t["live_rank"], t["margin_vs_best_other_live"], result["top_live_value"]


def test_an_open_question_at_rank_one_changes_no_live_figure():
    plain = R.derive_result(_fx(), _BASE, "strong", None)
    # Labelled live on purpose: the pure core enforces the kind rule itself.
    with_oq = R.derive_result(_fx(), [_hit("oq1", 0.99, kind="open_question", live=True)] + _BASE,
                              "strong", None)
    assert _comparable(with_oq) == _comparable(plain) == (2, 0.80 - 0.90, 0.90)
    assert with_oq["targets"][0]["raw_rank"] == 3
    assert with_oq["hits"][0]["live"] is False


def test_a_self_hit_changes_no_live_figure():
    plain = R.derive_result(_fx(form="draft"), _BASE, None, 0.8)
    with_self = R.derive_result(_fx(form="draft", exclude=["d0"]),
                                [_hit("d0", 1.0, self=True)] + _BASE, None, 0.8)
    assert _comparable(with_self) == _comparable(plain)


def test_a_retired_target_keeps_its_value_and_has_no_live_rank():
    result = R.derive_result(_fx(), [_hit("d2", 0.95, live=False)] + _BASE[:1], "strong", None)
    t = result["targets"][0]
    assert (t["value"], t["raw_rank"], t["live_rank"], t["absent"]) == (0.95, 1, None, False)
    assert t["margin_vs_best_other_live"] == pytest.approx(0.05)


def test_an_absent_target_is_kept_as_null_and_flagged():
    result = R.derive_result(_fx(targets=["d9"]), _BASE, "weak", None)
    assert result["targets"] == [{"slug": "d9", "value": None, "absent": True, "raw_rank": None,
                                  "live_rank": None, "margin_vs_best_other_live": None}]


def test_a_draft_target_records_whether_it_clears_the_pause_floor():
    result = R.derive_result(_fx(form="draft", targets=["d1", "d3"]), _BASE, None, 0.8)
    assert [t["clears_pause_floor"] for t in result["targets"]] == [True, False]
    assert result["pause_floor"] == 0.8 and result["envelope_band"] is None


def test_a_band_on_a_draft_or_a_floor_on_a_query_is_refused():
    with pytest.raises(ValueError):
        R.derive_result(_fx(form="draft"), _BASE, "strong", 0.8)
    with pytest.raises(ValueError):
        R.derive_result(_fx(), _BASE, "strong", 0.8)


def test_a_result_round_trips_through_json():
    result = R.derive_result(_fx(negatives=["d3"]), _BASE, "strong", None)
    assert json.loads(json.dumps(result)) == result


# ---------------------------------------------------------------------------
# Class statistics
# ---------------------------------------------------------------------------

def test_nearest_rank_statistics_and_the_absent_count():
    values = [0.4, 0.1, 0.9, 0.3, 0.7, 0.2, 1.0, 0.5, 0.8, 0.6]
    results = [R.derive_result(_fx(id=f"b{i}"), [_hit("d2", v)], "weak", None)
               for i, v in enumerate(values)]
    results.append(R.derive_result(_fx(id="b-absent"), [_hit("d1", 0.3)], "weak", None))
    block = R.summarize_classes(results)["bullseye"]
    assert block["values"] == sorted(values)
    assert (block["min"], block["p10"], block["median"], block["p90"], block["max"]) == (0.1, 0.1, 0.5, 0.9, 1.0)
    assert (block["n"], block["absent"], block["space"], block["measure"]) == (11, 1, "query", "target")


def test_a_top_live_class_measures_the_top_live_value():
    result = R.derive_result(_fx(**{"class": "off-axis"}, targets=[], measure="top_live"), _BASE, "weak", None)
    assert R.summarize_classes([result])["off-axis"]["values"] == [0.90]


def test_one_class_in_two_spaces_or_two_measures_is_refused():
    q = R.derive_result(_fx(id="a"), _BASE, "weak", None)
    d = R.derive_result(_fx(id="b", form="draft"), _BASE, None, 0.8)
    with pytest.raises(ValueError, match="mixes spaces"):
        R.summarize_classes([q, d])
    t = R.derive_result(_fx(id="c", measure="top_live"), _BASE, "weak", None)
    with pytest.raises(ValueError, match="mixes measures"):
        R.summarize_classes([q, t])


def test_unlabelled_results_summarize_to_nothing():
    assert R.summarize_classes([R.derive_result(_fx(**{"class": None}), _BASE, "weak", None)]) == {}


# ---------------------------------------------------------------------------
# Record and runs
# ---------------------------------------------------------------------------

def test_new_record_and_make_run_id():
    assert R.new_record("m") == {"schema": 1, "embedding_model": "m", "runs": [], "constants": {}}
    assert R.make_run_id("harbor", TS, SHA) == "harbor-20261016-9dd49d4"
    assert R.make_run_id("dense", TS, None) == "dense-20261016-nosha"
    # 01:30 in Vilnius on the 17th is still the 16th in UTC.
    assert R.make_run_id("harbor", "2026-10-17T01:30:00+03:00", SHA) == "harbor-20261016-9dd49d4"


def test_append_run_round_trips_and_indexes_the_file(record_dir):
    run = _run()
    run_id = R.append_run(run, record_dir)
    assert run_id == "harbor-20261016-9dd49d4"
    on_file = R.load_run(run_id, record_dir)
    assert on_file == dict(run, run_id=run_id)
    entry = R.load_record(record_dir)["runs"][0]
    assert entry == {"run_id": run_id, "corpus": "harbor", "embedding_model": MODEL, "decision_count": 29}
    text = open(os.path.join(record_dir, "runs", f"{run_id}.json"), encoding="utf-8").read()
    assert text.endswith("}\n") and text == json.dumps(on_file, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def test_append_run_never_overwrites_a_run(record_dir):
    first = R.append_run(_run(), record_dir)
    before = open(os.path.join(record_dir, "runs", f"{first}.json"), encoding="utf-8").read()
    second = R.append_run(_run(decision_count=30), record_dir)
    assert second == f"{first}-2"
    assert open(os.path.join(record_dir, "runs", f"{first}.json"), encoding="utf-8").read() == before
    assert [e["run_id"] for e in R.load_record(record_dir)["runs"]] == [first, second]


@pytest.mark.parametrize("run, needle", [
    (_run(classes={}), "no labelled class"),
    (_run(fixtures=[{"id": "f1", "class": None}]), "has class None"),
    (_run(model="model-b"), "differs from the record"),
    (_run(fixtures=None), "must carry its per-fixture results"),
    (_run(corpus="dense", fixtures=[{"id": "f1", "class": "bullseye"}]), "fixtures: null"),
    (_run(decision_count=0), "at least 1"),
    (_run(classes={"neighbour": _class_block("query", [0.5])}), "run class 'neighbour'"),
    (dict(_run(corpus="dense"), queries=["a private query"]), "keys outside the run shape"),
    (_run(corpus="dense", classes={"bullseye": dict(_class_block("query", [0.8]), slugs=["x"])}),
     "numeric statistics only"),
    (_run(corpus="dense", classes={"bullseye": dict(_class_block("query", [0.8]), values=["x"])}),
     "numeric statistics only"),
])
def test_append_run_refuses_and_writes_nothing(record_dir, run, needle):
    with pytest.raises(ValueError, match=None) as err:
        R.append_run(run, record_dir)
    assert needle in str(err.value)
    assert R.load_record(record_dir)["runs"] == []
    assert not os.path.exists(os.path.join(record_dir, "runs")) or not os.listdir(os.path.join(record_dir, "runs"))


def test_append_run_accepts_a_dense_run_without_fixtures(record_dir):
    run_id = R.append_run(_run(corpus="dense", decision_count=512), record_dir)
    assert R.load_run(run_id, record_dir)["fixtures"] is None
    assert R.load_record(record_dir)["runs"][0]["decision_count"] == 512


def test_set_record_model_rewrites_the_model_and_keeps_history(record_dir):
    run_id = R.append_run(_run(), record_dir)
    R.set_record_model("model-b", record_dir)
    record = R.load_record(record_dir)
    assert record["embedding_model"] == "model-b"
    assert [e["run_id"] for e in record["runs"]] == [run_id]
    assert R.append_run(_run(model="model-b", ts="2026-10-17T08:00:00+00:00"), record_dir)


# ---------------------------------------------------------------------------
# check_pins: one row per problem kind
# ---------------------------------------------------------------------------

def test_check_pins_is_silent_on_a_clean_record(record_dir):
    record, runs, constants, _, _ = _clean(record_dir)
    assert _problems(record, runs, constants) == []
    assert _problems(R.new_record(MODEL), {}, {}) == []


def test_check_pins_reports_a_set_constant_missing_from_the_record(record_dir):
    record, runs, constants, _, _ = _clean(record_dir)
    constants["mitos.recall.SURFACE_WEAK_THRESHOLD"] = 0.6
    assert _has(_problems(record, runs, constants), "absent from the record")


def test_check_pins_reports_a_record_constant_missing_from_the_set(record_dir):
    record, runs, _, _, _ = _clean(record_dir)
    assert _has(_problems(record, runs, {}), "absent from the calibrated set")


def test_check_pins_reports_a_moved_value(record_dir):
    record, runs, _, _, _ = _clean(record_dir)
    assert _has(_problems(record, runs, {STRONG: 0.76}), "source holds 0.76, record holds 0.75")


def test_check_pins_reports_another_model(record_dir):
    record, runs, constants, _, _ = _clean(record_dir)
    assert _has(_problems(record, runs, constants, model="model-b"), "record names model")


def test_check_pins_reports_empty_boundaries(record_dir):
    record, runs, constants, _, _ = _clean(record_dir)
    record["constants"][STRONG]["boundaries"] = {}
    assert _has(_problems(record, runs, constants), "`boundaries` is empty")


def test_check_pins_reports_a_citation_outside_the_index(record_dir):
    record, runs, constants, harbor, _ = _clean(record_dir)
    record["constants"][STRONG]["boundaries"]["harbor-19990101-0000000"] = 0.75
    assert _has(_problems(record, runs, constants), "not in the record's index")


def test_check_pins_reports_a_cited_run_with_no_file(record_dir):
    record, runs, constants, harbor, _ = _clean(record_dir)
    del runs[harbor]
    assert _has(_problems(record, runs, constants), "has no run file")


def test_check_pins_reports_an_index_entry_disagreeing_with_its_file(record_dir):
    record, runs, constants, _, _ = _clean(record_dir)
    record["runs"][0]["decision_count"] = 30
    assert _has(_problems(record, runs, constants), "index entry says decision_count=30")


def test_check_pins_reports_cited_runs_on_another_model_after_a_swap(record_dir):
    _clean(record_dir)
    R.set_record_model("model-b", record_dir)
    record = R.load_record(record_dir)
    problems = _problems(record, R.load_cited_runs(record, record_dir), {STRONG: 0.75}, model="model-b")
    assert _has(problems, "was measured on 'model-a'")
    assert not _has(problems, "record names model")


def test_check_pins_reports_a_class_absent_from_one_of_two_cited_runs(record_dir):
    record, runs, constants, _, dense = _clean(record_dir)
    del runs[dense]["classes"]["family-neighbour"]
    problems = _problems(record, runs, constants)
    assert _has(problems, f"'family-neighbour' is absent from cited run '{dense}'")


def test_check_pins_reports_a_class_in_another_space(record_dir):
    record, runs, constants, _, dense = _clean(record_dir)
    runs[dense]["classes"]["bullseye"]["space"] = "document"
    assert _has(_problems(record, runs, constants), "sits in 'document' space")


def test_check_pins_reports_a_threshold_boundary_outside_its_classes_range(record_dir):
    record, runs, _, harbor, dense = _clean(record_dir)
    entry = record["constants"][STRONG]
    entry.update(value=7.5, side="unmoved", boundaries={harbor: 7.5, dense: 0.73})
    problems = _problems(record, runs, {STRONG: 7.5})
    assert _has(problems, "lies outside its classes' observed range")
    entry["kind"] = "margin"
    assert _problems(record, runs, {STRONG: 7.5}) == []


@pytest.mark.parametrize("side, value, boundaries, clean", [
    ("higher", 0.73, (0.75, 0.73), False),
    ("higher", 0.75, (0.75, 0.73), True),
    ("lower", 0.75, (0.75, 0.73), False),
    ("lower", 0.73, (0.75, 0.73), True),
    ("dense", 0.75, (0.75, 0.73), False),
    ("dense", 0.73, (0.75, 0.73), True),
    ("unmoved", 0.80, (0.75, 0.73), True),
])
def test_check_pins_checks_each_side(record_dir, side, value, boundaries, clean):
    record, runs, _, harbor, dense = _clean(record_dir)
    record["constants"][STRONG].update(value=value, side=side,
                                       boundaries={harbor: boundaries[0], dense: boundaries[1]})
    problems = _problems(record, runs, {STRONG: value})
    assert (problems == []) is clean, problems
    if not clean:
        assert _has(problems, f"side `{side}`")


def test_check_pins_reports_a_dense_side_without_a_dense_citation(record_dir):
    record, runs, _, harbor, _ = _clean(record_dir)
    record["constants"][STRONG].update(side="dense", boundaries={harbor: 0.75})
    assert _has(_problems(record, runs, {STRONG: 0.75}), "exactly one cited dense run")


@pytest.mark.parametrize("key, value, needle", [
    ("side", "middle", "unknown side"),
    ("kind", "ratio", "unknown kind"),
    ("space", "both", "unknown space"),
    ("separates", ["neighbour"], "`separates` must list classes"),
])
def test_check_pins_reports_unknown_vocabulary(record_dir, key, value, needle):
    record, runs, constants, _, _ = _clean(record_dir)
    record["constants"][STRONG][key] = value
    assert _has(_problems(record, runs, constants), needle)


def test_check_pins_never_raises_on_a_malformed_record(record_dir):
    record, runs, constants, harbor, dense = _clean(record_dir)
    # Breaks the side check's max() on its own, whatever the range check does.
    record["constants"][STRONG]["boundaries"] = {harbor: "high", dense: 0.73}
    record["constants"]["mitos.x.OTHER_FLOOR"] = "not an object"
    problems = _problems(record, runs, dict(constants, **{"mitos.x.OTHER_FLOOR": 0.5}))
    assert _has(problems, "could not be checked") and _has(problems, "OTHER_FLOOR")
    assert _problems({}, {}, {STRONG: 0.75})


# ---------------------------------------------------------------------------
# The threshold population
# ---------------------------------------------------------------------------

def test_threshold_population_over_a_synthetic_tree(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("ROOT_MARGIN = 0.02\n")
    (pkg / "a.py").write_text(
        "X_THRESHOLD: float = 0.5\n"
        "Y_FLOOR = 0.3\n"
        "ALIAS_FLOOR = Y_FLOOR\n"
        "WIDTH_FLOOR = 24\n"
        "HIGH_FLOOR = 1.5\n"
        "NEG_FLOOR = -0.1\n"
        "BARE_FLOOR: float\n"
        "A_FLOOR, B_FLOOR = 0.1, 0.2\n"
        "TWICE_FLOOR = 0.1\nTWICE_FLOOR = 0.2\n"
        "BUMPED_FLOOR = 0.1\nBUMPED_FLOOR += 0.1\n"
        "ONLY_AUG_FLOOR += 0.1\n"
        "PLAIN = 0.5\n"
        "if True:\n    GUARDED_FLOOR = 0.2\n"
        "def f():\n    LOCAL_FLOOR = 0.1\n"
        "class C:\n    ATTR_FLOOR = 0.1\n"
    )
    (pkg / "sub" / "b.py").write_text("Q_MARGIN = 0.05\n")
    assert R.threshold_population(str(pkg)) == {
        "pkg.ROOT_MARGIN": 0.02,
        "pkg.a.ALIAS_FLOOR": None,
        "pkg.a.A_FLOOR": None,
        "pkg.a.BARE_FLOOR": None,
        "pkg.a.B_FLOOR": None,
        "pkg.a.BUMPED_FLOOR": None,
        "pkg.a.GUARDED_FLOOR": 0.2,
        "pkg.a.HIGH_FLOOR": None,
        "pkg.a.NEG_FLOOR": None,
        "pkg.a.ONLY_AUG_FLOOR": None,
        "pkg.a.TWICE_FLOOR": None,
        "pkg.a.WIDTH_FLOOR": None,
        "pkg.a.X_THRESHOLD": 0.5,
        "pkg.a.Y_FLOOR": 0.3,
        "pkg.sub.b.Q_MARGIN": 0.05,
    }


# ---------------------------------------------------------------------------
# Soft diff
# ---------------------------------------------------------------------------

def _diff_run(value=0.80, band="strong", fixtures=None, top=0.9):
    if fixtures is None:
        fixtures = [{"id": "f1", "measure": "target", "envelope_band": band, "top_live_value": top,
                     "targets": [{"slug": "d2", "value": value}]}]
    return {"fixtures": fixtures}


def test_soft_diff_reports_a_value_moved_past_the_band_and_not_inside_it():
    assert R.soft_diff(_diff_run(0.805), _diff_run(0.80)) == []
    moved = R.soft_diff(_diff_run(0.83), _diff_run(0.80))
    assert [(d["kind"], d["slug"], d["regression"]) for d in moved] == [("value_moved", "d2", True)]


def test_soft_diff_reports_a_band_change():
    assert [d["kind"] for d in R.soft_diff(_diff_run(band="weak"), _diff_run())] == ["band_changed"]


def test_soft_diff_reports_absent_and_present_flips():
    assert [d["kind"] for d in R.soft_diff(_diff_run(None), _diff_run())] == ["target_absent"]
    assert [d["kind"] for d in R.soft_diff(_diff_run(), _diff_run(None))] == ["target_present"]


def test_soft_diff_compares_a_top_live_fixture_by_its_top_live_value():
    def run(top):
        return {"fixtures": [{"id": "o1", "measure": "top_live", "envelope_band": "weak",
                              "top_live_value": top, "targets": []}]}
    assert [d["kind"] for d in R.soft_diff(run(0.7), run(0.6))] == ["value_moved"]


def test_soft_diff_lists_added_and_removed_fixtures_as_non_regressions():
    other = [{"id": "f2", "measure": "target", "envelope_band": "weak", "top_live_value": 0.5,
              "targets": []}]
    diffs = R.soft_diff(_diff_run(fixtures=other), _diff_run())
    assert {(d["kind"], d["id"], d["regression"]) for d in diffs} == {
        ("fixture_added", "f2", False), ("fixture_removed", "f1", False)}


def test_soft_diff_lists_a_fixture_s_added_and_removed_targets_as_non_regressions():
    def run(slugs):
        return {"fixtures": [{"id": "f1", "measure": "target", "envelope_band": "strong",
                              "top_live_value": 0.9,
                              "targets": [{"slug": s, "value": 0.8} for s in slugs]}]}
    diffs = R.soft_diff(run(["d2", "d3"]), run(["d2", "d4"]))
    assert {(d["kind"], d["slug"], d["regression"]) for d in diffs} == {
        ("target_added", "d3", False), ("target_removed", "d4", False)}
