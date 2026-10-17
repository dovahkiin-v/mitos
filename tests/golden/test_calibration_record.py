"""Offline rows for Layer C's pure core (`_calibration_record`), on synthetic inputs.

No service is touched: fixtures, hits, runs and records are built here, and the record
folder is a ``tmp_path``. These rows pin the rules a later phase writes against: the
fixture shape, the open-question and self-hit exclusions, nearest-rank statistics, what
``append_run`` refuses, every problem ``check_pins`` reports, the threshold population
and the soft diff.
"""

import json
import math
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
# A synthetic threshold name, outside `PRECISION_STATED` and `REPLAY_PRECISION`: the fixture
# record exercises the checker's shared rules; the precision block has rows of its own (6a).
STRONG = "mitos.recall.SYNTHETIC_THRESHOLD"


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


def _class_block(values, measure="target", negatives=()):
    """A (class, space) sub-block: the space is its key in the run, never a field (1c)."""
    block = {"measure": measure}
    block.update(R.stats(list(values)))
    block["negatives"] = R.stats(list(negatives))
    return block


def _run(corpus="harbor", model=MODEL, decision_count=29, classes=None, fixtures="default", ts=TS):
    if classes is None:
        classes = {"bullseye": {"query": _class_block([0.78, 0.84, 0.9])},
                   "family-neighbour": {"query": _class_block([0.5, 0.62, 0.72], "top_live")}}
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
    # 1c: the item measure and the rejection-target keys
    ({"measure": "item"}, "measure `item` belongs to class rejection-target"),
    ({"target_items": [{"owner": "d2", "option": "A"}]}, "belong to rejection-target fixtures only"),
    ({"negative_items": [{"owner": "d2", "option": "A", "kind": "generic"}]}, "belong to rejection-target fixtures only"),
    ({"rejection_form": "own-question"}, "belong to rejection-target fixtures only"),
])
def test_the_validator_refuses_a_malformed_fixture_and_names_it(over, needle):
    with pytest.raises(ValueError) as err:
        R.validate_fixtures([_fx(**over)])
    assert needle in str(err.value)
    assert repr(_fx(**over)["id"]) in str(err.value)


def _rx(**over):
    fx = {"id": "rejection-one", "class": "rejection-target", "form": "query", "text": "a proposal",
          "targets": ["d2"], "negatives": [], "measure": "item", "notes": "why",
          "target_items": [{"owner": "d2", "option": "Postgres"}],
          "negative_items": [{"owner": "d3", "option": "Do nothing", "kind": "generic"}],
          "rejection_form": "own-question"}
    fx.update(over)
    return fx


@pytest.mark.parametrize("over, needle", [
    ({"measure": "target"}, "measure `item` belongs to class rejection-target"),
    ({"form": "draft"}, "is a query"),
    ({"rejection_form": None}, "`rejection_form`"),
    ({"rejection_form": "elsewhere"}, "`rejection_form`"),
    ({"target_items": []}, "at least one `target_items`"),
    ({"negative_items": []}, "at least one `negative_items`"),
    ({"target_items": [{"owner": "d2"}]}, "{owner, option}"),
    ({"negative_items": [{"option": "Defer", "kind": "generic"}]}, "{owner, option, kind}"),
    ({"negative_items": [{"owner": "d3", "option": "Defer"}]}, "{owner, option, kind}"),
    ({"negative_items": [{"owner": "d3", "option": "Defer", "kind": "noise"}]}, "kind one of"),
    ({"target_items": [{"owner": "d2", "option": "A", "kind": "vocabulary"}]}, "{owner, option}"),
    ({"target_items": [{"owner": "d2", "option": " "}]}, "{owner, option}"),
    ({"target_items": [{"owner": "d2", "option": "A", "ordinal": 1}]}, "{owner, option}"),
    ({"target_items": ["d2"]}, "{owner, option}"),
    ({"negatives": ["d4"]}, "slug `negatives` must be empty"),
    ({"targets": ["d9"]}, "must be in `targets`"),
])
def test_the_validator_refuses_a_malformed_rejection_fixture(over, needle):
    fx = _rx(**over)
    if over.get("rejection_form", "") is None:
        del fx["rejection_form"]
    with pytest.raises(ValueError) as err:
        R.validate_fixtures([fx])
    assert needle in str(err.value) and "'rejection-one'" in str(err.value)


def test_a_well_formed_rejection_fixture_validates_and_its_owners_are_checked():
    assert R.validate_fixtures([_rx()])
    with pytest.raises(ValueError, match="rejection-one.*d3"):
        R.check_fixture_slugs([_rx()], {"d2"})


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


# -- 6b: the envelope's band evidence ----------------------------------------------------

_EVIDENCE = {"top_score": 0.9, "top_cluster": 2, "edge": "mid_cluster", "next_score": 0.85}


def test_a_query_result_records_the_envelopes_evidence_and_round_trips():
    result = R.derive_result(_fx(), _BASE, "strong", None, envelope_evidence=_EVIDENCE)
    assert result["envelope_evidence"] == _EVIDENCE
    assert json.loads(json.dumps(result)) == result
    assert "envelope_evidence" not in R.derive_result(_fx(), _BASE, "strong", None)


def test_evidence_on_a_draft_read_is_refused():
    with pytest.raises(ValueError, match="no envelope evidence"):
        R.derive_result(_fx(form="draft"), _BASE, None, 0.8, envelope_evidence=_EVIDENCE)


@pytest.mark.parametrize("bad", [
    {"top_score": 0.9, "top_cluster": 2, "edge": "mid_cluster"},             # no next_score
    {"top_score": 0.9, "top_cluster": 2, "edge": "end", "next_score": 0.8},  # next on an end
    {"top_score": 0.9, "top_cluster": 2, "edge": "sideways"},                # outside the tuple
    {"top_score": 0.9, "top_cluster": True, "edge": "end"},                  # a bool count
    {"top_score": 0.9, "top_cluster": 0, "edge": "end"},                     # below 1
    {"top_score": "0.9", "top_cluster": 1, "edge": "end"},                   # a string score
    {"top_score": 0.9, "top_cluster": 1, "edge": "end", "overlap": 0.4},     # an input key
    [0.9, 1, "end"],                                                         # not an object
])
def test_malformed_evidence_is_refused(bad):
    assert R.evidence_problem(bad)
    with pytest.raises(ValueError):
        R.derive_result(_fx(), _BASE, "strong", None, envelope_evidence=bad)


def test_the_edge_tokens_agree_with_the_shipped_tuple():
    from mitos import recall
    assert R.EVIDENCE_EDGE_KINDS == recall.EVIDENCE_EDGE_KINDS


def test_a_committed_run_with_malformed_evidence_is_refused(record_dir):
    ok = _run(fixtures=[{"id": "f1", "class": "bullseye", "targets": [], "space": "query",
                         "envelope_evidence": dict(_EVIDENCE)}])
    R.append_run(ok, record_dir)
    bad = _run(decision_count=31, fixtures=[{"id": "f1", "class": "bullseye", "targets": [],
                                             "space": "query",
                                             "envelope_evidence": {"edge": "end"}}])
    with pytest.raises(ValueError, match="envelope_evidence"):
        R.append_run(bad, record_dir)
    draft = _run(decision_count=32, fixtures=[{"id": "f1", "class": "bullseye", "targets": [],
                                               "space": "document",
                                               "envelope_evidence": dict(_EVIDENCE)}])
    with pytest.raises(ValueError, match="draft-form"):
        R.append_run(draft, record_dir)


# ---------------------------------------------------------------------------
# Class statistics
# ---------------------------------------------------------------------------

def test_nearest_rank_statistics_and_the_absent_count():
    values = [0.4, 0.1, 0.9, 0.3, 0.7, 0.2, 1.0, 0.5, 0.8, 0.6]
    results = [R.derive_result(_fx(id=f"b{i}"), [_hit("d2", v)], "weak", None)
               for i, v in enumerate(values)]
    results.append(R.derive_result(_fx(id="b-absent"), [_hit("d1", 0.3)], "weak", None))
    # 1c: the space is the sub-block's key, so the block carries no `space` field.
    block = R.summarize_classes(results)["bullseye"]["query"]
    assert block["values"] == sorted(values)
    assert (block["min"], block["p10"], block["median"], block["p90"], block["max"]) == (0.1, 0.1, 0.5, 0.9, 1.0)
    assert (block["n"], block["absent"], block["measure"]) == (11, 1, "target")
    assert "space" not in block


def test_a_top_live_class_measures_the_top_live_value():
    result = R.derive_result(_fx(**{"class": "off-axis"}, targets=[], measure="top_live"), _BASE, "weak", None)
    assert R.summarize_classes([result])["off-axis"]["query"]["values"] == [0.90]


def test_two_spaces_in_one_class_are_two_sub_blocks_and_two_measures_in_one_are_refused():
    """Inverted in 1c (was: one class in two spaces is refused).

    The vision poses retired-precedent as a query and as a draft, and needs draft-form
    bullseye and family-neighbour controls; nesting class -> space keeps the guard 1b's
    refusal held (one block never mixes two spaces' values) as structure. Two measures
    inside one (class, space) are still a labelling error.
    """
    q = R.derive_result(_fx(id="a"), [_hit("d2", 0.7)], "weak", None)
    d = R.derive_result(_fx(id="b", form="draft"), [_hit("d2", 0.9)], None, 0.8)
    classes = R.summarize_classes([q, d])
    assert set(classes["bullseye"]) == {"query", "document"}
    assert classes["bullseye"]["query"]["values"] == [0.7]
    assert classes["bullseye"]["document"]["values"] == [0.9]
    t = R.derive_result(_fx(id="c", measure="top_live"), _BASE, "weak", None)
    with pytest.raises(ValueError, match="mixes measures"):
        R.summarize_classes([q, t])
    # One measure per (class, space): the draft sub-block may measure differently.
    td = R.derive_result(_fx(id="e", form="draft", measure="top_live"), _BASE, None, 0.8)
    with pytest.raises(ValueError, match="'document' space mixes measures"):
        R.summarize_classes([d, td])


def test_a_sub_block_summarizes_its_negatives_with_absent_counted():
    results = [
        R.derive_result(_fx(id="a", negatives=["d1", "d3"]), _BASE, "weak", None),
        R.derive_result(_fx(id="b", negatives=["d9"]), _BASE, "weak", None),
    ]
    neg = R.summarize_classes(results)["bullseye"]["query"]["negatives"]
    assert (neg["n"], neg["absent"], neg["values"], neg["min"], neg["max"]) == (3, 1, [0.70, 0.90], 0.70, 0.90)


def test_an_unknown_measure_is_refused_rather_than_read_as_top_live():
    r = dict(R.derive_result(_fx(), _BASE, "weak", None), measure="ratio")
    with pytest.raises(ValueError, match="unknown measure 'ratio'"):
        R.summarize_classes([r])


# ---------------------------------------------------------------------------
# Item results and the shape comparison (1c)
# ---------------------------------------------------------------------------

def _ih(owner, value, ordinal=None, part=None, live=True):
    return {"point_id": f"{owner}-{ordinal}-{part}", "owner": owner, "ordinal": ordinal, "part": part,
            "value": value, "owner_live": live}


def _iref(owner, option, ordinal):
    return {"owner": owner, "option": option, "ordinal": ordinal}


def test_parts_collapse_to_their_best_before_rank_and_margin():
    """A two-part target's own second part never competes with it, and a two-part
    competitor above it costs one rank, not two."""
    per_item = [_ih("c", 0.95, 1, 0), _ih("c", 0.93, 1, 1), _ih("t", 0.90, 2, 0),
                _ih("t", 0.88, 2, 1), _ih("x", 0.60, 1, 0)]
    per_decision = [_ih("c", 0.8), _ih("t", 0.7), _ih("x", 0.5)]
    refs = {"targets": [_iref("t", "Postgres", 2)], "negatives": [_iref("x", "Defer", 1)]}
    items = R.derive_item_result(_rx(), {"per_item": per_item, "per_decision": per_decision}, refs)
    t = items["per_item"]["targets"][0]
    assert (t["value"], t["rank"], t["absent"]) == (0.90, 2, False)
    assert t["margin_vs_best_other"] == pytest.approx(0.90 - 0.95)
    assert items["per_item"]["top_value"] == 0.95
    assert len(items["per_item"]["hits"]) == 5  # the raw parts are kept
    n = items["per_item"]["negatives"][0]
    assert (n["owner"], n["option"], n["value"], n["rank"]) == ("x", "Defer", 0.60, 3)


def test_the_per_decision_shape_values_a_negative_at_its_owner_so_a_shared_owner_dilutes():
    per_item = [_ih("o", 0.90, 1, 0), _ih("o", 0.55, 4, 0), _ih("y", 0.70, 1, 0)]
    per_decision = [_ih("o", 0.75), _ih("y", 0.72)]
    refs = {"targets": [_iref("o", "Postgres", 1)], "negatives": [_iref("o", "Keeping", 4)]}
    items = R.derive_item_result(_rx(), {"per_item": per_item, "per_decision": per_decision}, refs)
    assert [items["per_item"][r][0]["value"] for r in ("targets", "negatives")] == [0.90, 0.55]
    pd = items["per_decision"]
    assert pd["targets"][0]["value"] == pd["negatives"][0]["value"] == 0.75
    assert pd["targets"][0]["option"] == "Postgres" and pd["negatives"][0]["option"] == "Keeping"
    assert pd["targets"][0]["margin_vs_best_other"] == pytest.approx(0.75 - 0.72)


def test_an_item_ref_no_hit_carries_is_absent_and_kept():
    refs = {"targets": [_iref("z", "Gone", 1)], "negatives": []}
    items = R.derive_item_result(_rx(), {"per_item": [_ih("o", 0.9, 1, 0)], "per_decision": [_ih("o", 0.8)]}, refs)
    for shape in R.SHAPES:
        assert items[shape]["targets"] == [{"owner": "z", "option": "Gone", "value": None, "rank": None,
                                            "absent": True, "margin_vs_best_other": None}]


def test_derive_item_result_refuses_a_non_item_fixture_and_a_missing_per_item_shape():
    with pytest.raises(ValueError, match="only an item-measured"):
        R.derive_item_result(_fx(), {"per_item": [], "per_decision": []}, {"targets": [], "negatives": []})
    with pytest.raises(ValueError, match="no hits for the per_item shape"):
        R.derive_item_result(_rx(), {"per_decision": []}, {"targets": [], "negatives": []})
    with pytest.raises(ValueError, match="unknown shapes"):
        R.derive_item_result(_rx(), {"per_item": [], "per_field": []}, {"targets": [], "negatives": []})


def test_a_per_item_only_result_derives_and_carries_each_negative_s_kind():
    """5b: the dense run over the copy has no per-decision shape; a negative's kind rides it."""
    refs = {"targets": [_iref("t", "Postgres", 1)],
            "negatives": [dict(_iref("g", "Defer", 1), kind="generic"),
                          dict(_iref("v", "Blobs", 1), kind="vocabulary")]}
    items = R.derive_item_result(
        _rx(), {"per_item": [_ih("t", 0.8, 1, 0), _ih("v", 0.7, 1, 0), _ih("g", 0.6, 1, 0)]}, refs)
    assert set(items) == {"per_item"}
    assert [(n["owner"], n["kind"]) for n in items["per_item"]["negatives"]] == [("g", "generic"), ("v", "vocabulary")]
    assert "kind" not in items["per_item"]["targets"][0]


def _item_result(fid, form, per_item_t, per_dec_t, rank_i=1, rank_d=1, neg=0.5):
    def shape(value, rank):
        return {"hits": [], "top_value": value,
                "targets": [{"owner": "o", "option": "A", "value": value, "rank": rank, "absent": value is None,
                             "margin_vs_best_other": None if value is None else value - 0.6}],
                "negatives": [{"owner": "g", "option": "Defer", "value": neg, "rank": 9, "absent": False,
                               "kind": "generic"}]}
    fx = _rx(id=fid, rejection_form=form)
    result = R.derive_result(fx, [_hit("o", 0.7)], "weak", None)
    result["items"] = {"per_item": shape(per_item_t, rank_i), "per_decision": shape(per_dec_t, rank_d)}
    return result


def test_summarize_shapes_compares_both_shapes_and_counts_absent_refs_in_rank_one():
    results = [
        _item_result("r1", "own-question", 0.90, 0.80),
        _item_result("r2", "elsewhere-carries", 0.85, 0.70, rank_d=3),
        _item_result("r3", "elsewhere-does-not-carry", None, 0.60, rank_i=None, rank_d=5),
        R.derive_result(_fx(id="b1"), _BASE, "weak", None),  # not item-measured: ignored
    ]
    shapes = R.summarize_shapes(results)
    assert set(shapes) == set(R.SHAPES)
    pi, pd = shapes["per_item"], shapes["per_decision"]
    assert (pi["targets"]["n"], pi["targets"]["absent"], pi["targets"]["values"]) == (3, 1, [0.85, 0.90])
    assert pi["target_rank_one"] == pytest.approx(2 / 3)  # absent counted in the denominator
    assert pd["target_rank_one"] == pytest.approx(1 / 3)
    assert pi["by_form"]["own-question"]["values"] == [0.90]
    assert pi["by_form"]["elsewhere-does-not-carry"]["absent"] == 1
    assert pi["negatives"]["values"] == [0.5, 0.5, 0.5]
    assert pi["negatives_by_kind"]["generic"]["values"] == [0.5, 0.5, 0.5]
    assert pi["negatives_by_kind"]["vocabulary"]["n"] == 0
    assert pd["margin"]["values"] == pytest.approx([0.0, 0.1, 0.2])
    classes = R.summarize_classes(results)
    assert classes["rejection-target"]["query"]["values"] == [0.85, 0.90]  # per item, the ruled shape
    assert classes["rejection-target"]["query"]["negatives"]["values"] == [0.5, 0.5, 0.5]
    assert R.summarize_shapes([R.derive_result(_fx(), _BASE, "weak", None)]) == {}


def test_summarize_shapes_leaves_out_a_shape_some_item_result_lacks():
    """Harbor only reads both shapes; a per-item-only result never summarizes a partial
    per-decision block."""
    full = _item_result("r1", "own-question", 0.90, 0.80)
    partial = _item_result("r2", "own-question", 0.85, 0.70)
    del partial["items"]["per_decision"]
    assert set(R.summarize_shapes([full, partial])) == {"per_item"}


def test_unlabelled_results_summarize_to_nothing():
    assert R.summarize_classes([R.derive_result(_fx(**{"class": None}), _BASE, "weak", None)]) == {}


# ---------------------------------------------------------------------------
# Record and runs
# ---------------------------------------------------------------------------

def test_new_record_and_make_run_id():
    # 1c: schema 2, a run's classes nest class -> space -> sub-block.
    assert R.new_record("m") == {"schema": 2, "embedding_model": "m", "runs": [], "constants": {}}
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
    (_run(classes={"neighbour": {"query": _class_block([0.5])}}), "run class 'neighbour'"),
    (_run(classes={"bullseye": _class_block([0.5])}), "must map each space"),
    (_run(classes={"bullseye": {"vibe": _class_block([0.5])}}), "must map each space"),
    (_run(classes={"bullseye": {"query": dict(_class_block([0.5]), measure="best")}}), "must map each space"),
    (dict(_run(corpus="dense"), queries=["a private query"]), "keys outside the run shape"),
    (_run(corpus="dense", classes={"bullseye": {"query": dict(_class_block([0.8]), slugs=["x"])}}),
     "numeric statistics only"),
    (_run(corpus="dense", classes={"bullseye": {"query": dict(_class_block([0.8]), values=["x"])}}),
     "numeric statistics only"),
    (_run(corpus="dense", classes={"bullseye": {"query": dict(_class_block([0.8]), space="query")}}),
     "numeric statistics only"),
    (_run(corpus="dense", classes={"bullseye": {"query": _class_block([0.8], negatives=["slug-x"])}}),
     "`negatives.values` holds a non-number"),
    (_run(corpus="dense", classes={"bullseye": {"query": dict(
        _class_block([0.8]), negatives=dict(R.stats([0.5]), slugs=["x"]))}}), "`negatives` has extra keys"),
    # 1c fresh eyes: a scalar statistic carries text as readily as `values` does.
    (_run(corpus="dense", classes={"bullseye": {"query": dict(_class_block([0.8]), min="a private slug")}}),
     "`min` is not a number"),
    (_run(corpus="dense", classes={"bullseye": {"query": dict(_class_block([0.8]), n=True)}}),
     "`n` is not a number"),
    (_run(corpus="dense", classes={"bullseye": {"query": dict(
        _class_block([0.8]), negatives=dict(R.stats([0.5]), median="a private query"))}}),
     "`negatives.median` is not a number"),
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


def test_check_pins_reports_a_class_with_no_sub_block_in_the_constant_s_space(record_dir):
    """Inverted in 1c (was: a class block naming another space is reported).

    The space is now the sub-block's key, so a constant reads ``classes[cls][space]`` and
    cannot read the other space's values; a class measured only in the other space is a
    lookup that finds nothing, reported as such.
    """
    record, runs, constants, _, dense = _clean(record_dir)
    block = runs[dense]["classes"]["bullseye"]
    block["document"] = block.pop("query")
    problems = _problems(record, runs, constants)
    assert _has(problems, f"class 'bullseye' has no values in 'query' space in run '{dense}'")


def _floor_record(record_dir, boundary):
    """A record whose rejection-target class has targets 0.85-0.95 and negatives 0.70-0.78."""
    run = R.append_run(_run(classes={"rejection-target": {"query": _class_block(
        [0.85, 0.90, 0.95], "item", negatives=[0.70, 0.74, 0.78])}}), record_dir)
    record = R.load_record(record_dir)
    name = "mitos.recall.REJECTION_FLOOR"
    record["constants"][name] = {"value": boundary, "kind": "threshold", "space": "query",
                                 "separates": ["rejection-target"], "boundaries": {run: boundary},
                                 "side": "higher", "set_by": "5a", "reason": "synthetic"}
    return _problems(record, R.load_cited_runs(record, record_dir), {name: boundary})


def test_a_floor_between_negatives_and_targets_is_inside_the_range(record_dir):
    assert _floor_record(record_dir, 0.80) == []


def test_a_floor_above_every_target_and_negative_is_outside_the_range(record_dir):
    assert _has(_floor_record(record_dir, 0.97), "outside its classes' observed range [0.7, 0.9501]")  # 6a: the 4-decimal envelope


def test_check_pins_reports_a_threshold_boundary_outside_its_classes_range(record_dir):
    record, runs, _, harbor, dense = _clean(record_dir)
    entry = record["constants"][STRONG]
    # 7a: side `unmoved` now holds its value at or below every boundary, so the value
    # sits under both; the out-of-range boundary is what this row is about.
    entry.update(value=0.7, side="unmoved", boundaries={harbor: 7.5, dense: 0.73})
    problems = _problems(record, runs, {STRONG: 0.7})
    assert _has(problems, "lies outside its classes' observed range")
    entry["kind"] = "margin"
    assert _problems(record, runs, {STRONG: 0.7}) == []


@pytest.mark.parametrize("side, value, boundaries, clean", [
    ("higher", 0.73, (0.75, 0.73), False),
    ("higher", 0.75, (0.75, 0.73), True),
    ("lower", 0.75, (0.75, 0.73), False),
    ("lower", 0.73, (0.75, 0.73), True),
    ("dense", 0.75, (0.75, 0.73), False),
    ("dense", 0.73, (0.75, 0.73), True),
    # 7a inverted this case (it was clean): an unmoved floor sits at or below every
    # corpus's recall line, and 0.80 is above both.
    ("unmoved", 0.80, (0.75, 0.73), False),
    ("unmoved", 0.73, (0.75, 0.73), True),
    ("unmoved", 0.70, (0.75, 0.73), True),
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
# check_runs: every indexed run, cited or not
# ---------------------------------------------------------------------------

def test_check_runs_is_silent_on_a_clean_record(record_dir):
    record, _, _, _, _ = _clean(record_dir)
    assert R.check_runs(record, record_dir) == []
    assert R.check_runs(R.new_record(MODEL), record_dir + "-empty") == []


def test_check_runs_reports_an_indexed_run_with_no_file(record_dir):
    record, _, _, harbor, _ = _clean(record_dir)
    os.remove(os.path.join(record_dir, R.RUNS_DIRNAME, f"{harbor}.json"))
    assert _has(R.check_runs(record, record_dir), f"run '{harbor}' is indexed but has no run file")


def test_check_runs_reports_a_run_file_that_does_not_parse(record_dir):
    record, _, _, harbor, _ = _clean(record_dir)
    with open(os.path.join(record_dir, R.RUNS_DIRNAME, f"{harbor}.json"), "w") as f:
        f.write("{not json")
    assert _has(R.check_runs(record, record_dir), "does not parse")


def test_check_runs_reports_an_entry_disagreeing_with_its_file(record_dir):
    record, _, _, _, dense = _clean(record_dir)
    record["runs"][1]["decision_count"] = 501
    assert _has(R.check_runs(record, record_dir),
                f"run '{dense}' index entry says decision_count=501, its file says 500")


def test_check_runs_reports_an_orphan_run_file(record_dir):
    record, _, _, _, _ = _clean(record_dir)
    with open(os.path.join(record_dir, R.RUNS_DIRNAME, "dense-19990101-0000000.json"), "w") as f:
        json.dump(_run(corpus="dense"), f)
    assert _has(R.check_runs(record, record_dir), "'dense-19990101-0000000.json' is not in the record's index")


def test_check_runs_reports_a_dense_file_carrying_fixtures(record_dir):
    record, _, _, _, dense = _clean(record_dir)
    path = os.path.join(record_dir, R.RUNS_DIRNAME, f"{dense}.json")
    with open(path) as f:
        run = json.load(f)
    run["fixtures"] = [{"id": "f1", "class": "bullseye", "text": "a private query"}]
    with open(path, "w") as f:
        json.dump(run, f)
    assert _has(R.check_runs(record, record_dir), "must carry `fixtures: null`")


def test_check_runs_stays_silent_on_old_model_history_after_a_swap(record_dir):
    """The swap reds the constants citing old runs (check_pins), not the history itself."""
    _clean(record_dir)
    R.set_record_model("model-b", record_dir)
    record = R.load_record(record_dir)
    assert R.check_runs(record, record_dir) == []
    pins = _problems(record, R.load_cited_runs(record, record_dir), {STRONG: 0.75}, model="model-b")
    assert _has(pins, "was measured on 'model-a'")


def test_check_runs_reports_a_run_indexed_twice(record_dir):
    record, _, _, harbor, _ = _clean(record_dir)
    record["runs"].append(dict(record["runs"][0]))
    problems = R.check_runs(record, record_dir)
    assert _has(problems, f"run '{harbor}' is indexed more than once")
    assert len(problems) == 1, problems


def test_check_runs_reports_a_malformed_run_and_still_checks_the_rest(record_dir):
    """A file whose provenance is a list fails that run alone; later runs and orphans are still read."""
    record, _, _, harbor, dense = _clean(record_dir)
    path = os.path.join(record_dir, R.RUNS_DIRNAME, f"{harbor}.json")
    with open(path) as f:
        run = json.load(f)
    run["provenance"] = ["not", "an", "object"]
    with open(path, "w") as f:
        json.dump(run, f)
    record["runs"][1]["decision_count"] = 501
    with open(os.path.join(record_dir, R.RUNS_DIRNAME, "dense-19990101-0000000.json"), "w") as f:
        json.dump(_run(corpus="dense"), f)
    problems = R.check_runs(record, record_dir)
    assert _has(problems, f"run '{harbor}' is malformed")
    assert _has(problems, f"run '{dense}' index entry says decision_count=501")
    assert _has(problems, "'dense-19990101-0000000.json' is not in the record's index")
    assert not _has(problems, "could not be checked")


def test_check_runs_never_raises_on_a_malformed_record(record_dir):
    assert _has(R.check_runs({}, record_dir), "no `runs` list")
    assert _has(R.check_runs({"runs": ["x", {"run_id": "../up"}]}, record_dir), "not an object")
    assert _has(R.check_runs({"runs": [{"run_id": None}]}, record_dir), "not a run id")


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


def _item_diff_run(target=0.80, negative=0.50, shape_value=None):
    def shape(t):
        return {"targets": [{"owner": "o", "option": "A", "value": t}],
                "negatives": [{"owner": "g", "option": "Defer", "value": negative}]}
    return {"fixtures": [{"id": "r1", "measure": "item", "envelope_band": "weak", "top_live_value": 0.7,
                          "targets": [{"slug": "o", "value": 0.6}],
                          "items": {"per_item": shape(target),
                                    "per_decision": shape(target if shape_value is None else shape_value)}}]}


def test_soft_diff_reads_item_values_per_shape_and_role():
    assert R.soft_diff(_item_diff_run(), _item_diff_run()) == []
    moved = R.soft_diff(_item_diff_run(0.85), _item_diff_run(0.80))
    assert {(d["kind"], d["shape"], d["role"], d["owner"], d["option"]) for d in moved} == {
        ("value_moved", "per_item", "target", "o", "A"), ("value_moved", "per_decision", "target", "o", "A")}
    assert all(d["regression"] for d in moved)
    neg = R.soft_diff(_item_diff_run(negative=0.6), _item_diff_run())
    assert {(d["kind"], d["role"]) for d in neg} == {("value_moved", "negative")}
    flips = R.soft_diff(_item_diff_run(shape_value=None, target=None), _item_diff_run())
    assert {d["kind"] for d in flips} == {"target_absent"}


def test_soft_diff_lists_added_and_removed_item_refs_as_non_regressions():
    base = _item_diff_run()
    cur = _item_diff_run()
    cur["fixtures"][0]["items"]["per_item"]["targets"][0]["option"] = "B"
    diffs = R.soft_diff(cur, base)
    assert {(d["kind"], d["option"], d["regression"]) for d in diffs} == {
        ("item_added", "B", False), ("item_removed", "A", False)}


def test_a_schema_two_run_round_trips_through_the_record(record_dir):
    results = [
        R.derive_result(_fx(id="q"), _BASE, "weak", None),
        R.derive_result(_fx(id="d", form="draft"), _BASE, None, 0.8),
        _item_result("r", "own-question", 0.9, 0.8),
    ]
    run = _run(classes=R.summarize_classes(results), fixtures=results)
    run["shapes"] = R.summarize_shapes(results)
    run_id = R.append_run(run, record_dir)
    assert R.load_run(run_id, record_dir) == dict(run, run_id=run_id)
    assert set(R.load_run(run_id, record_dir)["classes"]["bullseye"]) == {"query", "document"}


# ---------------------------------------------------------------------------
# The edge margin's method (2e): inversion depths
# ---------------------------------------------------------------------------

def _inv(hits, targets, cls="bullseye", space="query", absent=()):
    return {"class": cls, "space": space, "hits": hits,
            "targets": [{"slug": t, "absent": t in absent} for t in targets]}


def test_a_target_ranked_first_has_no_inversion():
    r = _inv([_hit("t", 0.9), _hit("n", 0.8)], ["t"])
    assert R.inversion_depths([r]) == []


def test_a_non_target_above_a_target_gives_the_best_ones_gap():
    r = _inv([_hit("n1", 0.9), _hit("n2", 0.85), _hit("t", 0.8)], ["t"])
    assert R.inversion_depths([r]) == [pytest.approx(0.1)]


def test_a_second_target_above_is_not_an_inversion():
    """Unlike `margin_vs_best_other_live`, another target outranking a target is the
    ranking agreeing with the labels."""
    r = _inv([_hit("t1", 0.9), _hit("t2", 0.85)], ["t1", "t2"])
    assert R.inversion_depths([r]) == []
    r = _inv([_hit("t1", 0.9), _hit("n", 0.88), _hit("t2", 0.85)], ["t1", "t2"])
    assert R.inversion_depths([r]) == [pytest.approx(0.03)]


def test_self_retired_and_non_decision_hits_are_not_competitors():
    r = _inv([_hit("me", 0.95, self=True), _hit("dead", 0.93, live=False),
              _hit("oq", 0.92, kind="open_question"), _hit("t", 0.9)], ["t"])
    assert R.inversion_depths([r]) == []


def test_an_absent_or_non_live_target_is_skipped():
    r = _inv([_hit("n", 0.9)], ["t"], absent=("t",))
    assert R.inversion_depths([r]) == []
    r = _inv([_hit("n", 0.9), _hit("t", 0.8, live=False)], ["t"])
    assert R.inversion_depths([r]) == []


def test_only_the_two_target_classes_in_query_space_are_read():
    hits = [_hit("n", 0.9), _hit("t", 0.8)]
    assert R.inversion_depths([_inv(hits, ["t"], cls="family-neighbour")]) == []
    assert R.inversion_depths([_inv(hits, ["t"], space="document")]) == []
    assert R.inversion_depths([_inv(hits, ["t"], cls="handle-words")]) == [pytest.approx(0.1)]


def test_the_boundary_is_the_nearest_rank_median_to_four_places_or_none():
    rows = [_inv([_hit("n", 0.9), _hit("t", 0.9 - d)], ["t"]) for d in (0.01, 0.02, 0.03, 0.04)]
    assert R.edge_margin_boundary(rows) == 0.02   # nearest rank 2 of 4, not the mean 0.025
    assert R.edge_margin_boundary([_inv([_hit("t", 0.9)], ["t"])]) is None


# ---------------------------------------------------------------------------
# The rejection-match floor (5b D3) and the escalation floor's record entry (D5)
# ---------------------------------------------------------------------------

FLOOR = "mitos.recall.REJECTION_MATCH_FLOOR"


def _rej(form, targets, negatives, cls="rejection-target"):
    """A rejection-target result holding per-item target and negative values only."""
    return {"id": f"r-{form}", "class": cls, "measure": "item", "rejection_form": form,
            "items": {"per_item": {
                "targets": [{"owner": "o", "option": "A", "value": v} for v in targets],
                "negatives": [{"owner": "g", "option": "N", "value": v, "kind": k} for v, k in negatives]}}}


@pytest.mark.parametrize("top, boundary", [
    (0.65133, 0.6514),   # the next 4-place value up
    (0.6513, 0.6514),    # an exact 4-place maximum moves up one step, never stays
    (0.70, 0.7001),
    (0.69999, 0.7),
])
def test_the_floor_boundary_is_the_smallest_four_place_value_strictly_above_every_negative(top, boundary):
    results = [_rej("own-question", [0.9], [(0.5, "generic"), (top, "vocabulary")]),
               _rej("elsewhere-carries", [0.8], [(0.6, "generic")])]
    got = R.rejection_floor_boundary(results)
    assert got == boundary and got > top


def test_the_floor_boundary_reads_both_kinds_every_form_and_only_the_rejection_class():
    results = [_rej("elsewhere-does-not-carry", [0.95], [(0.71, "vocabulary")]),
               _rej("own-question", [0.9], [(0.66, "generic")]),
               _rej("own-question", [0.9], [(0.99, "generic")], cls="bullseye")]
    assert R.rejection_floor_boundary(results) == 0.7101
    assert R.rejection_floor_boundary([_rej("own-question", [0.9], [(None, "generic")])]) is None
    assert R.rejection_floor_boundary([]) is None


def test_retention_counts_own_and_carries_targets_and_leaves_does_not_carry_out():
    results = [_rej("own-question", [0.80, 0.60, None], []),
               _rej("elsewhere-carries", [0.75], []),
               _rej("elsewhere-does-not-carry", [0.10, 0.20], [])]
    assert R.rejection_floor_retention(results, 0.70) == pytest.approx(2 / 4)  # absent is not kept
    assert R.rejection_floor_retention(results, 0.80) == pytest.approx(1 / 4)
    assert R.rejection_floor_retention([_rej("elsewhere-does-not-carry", [0.9], [])], 0.5) is None


def _cry_wolf(**over):
    block = {"ceiling": 2.0, "ratio": 1.6, "appearances": 24, "earned": 15, "earned_label": 12,
             "earned_proxy": 6, "reads": 106, "replay_run": "20261017-0000-abc1234-dogfood-copy",
             "copy_state": "s3-rejection-9c6fd89", "labels_sha256": "a" * 64, "raised_from": None}
    block.update(over)
    return block


def _escalation_record(record_dir, value=0.80, side="higher", boundary=0.80, cry_wolf="default"):
    """`_floor_record`'s classes, under the escalation floor's own name."""
    run = R.append_run(_run(classes={"rejection-target": {"query": _class_block(
        [0.85, 0.90, 0.95], "item", negatives=[0.70, 0.74, 0.78])}}), record_dir)
    record = R.load_record(record_dir)
    entry = {"value": value, "kind": "threshold", "space": "query", "separates": ["rejection-target"],
             "boundaries": {run: boundary}, "side": side, "method": "synthetic"}
    if cry_wolf is not None:
        entry["cry_wolf"] = _cry_wolf() if cry_wolf == "default" else cry_wolf
    record["constants"][FLOOR] = entry
    return _problems(record, R.load_cited_runs(record, record_dir), {FLOOR: value})


def test_an_escalation_floor_with_its_ceiling_met_is_clean(record_dir):
    assert FLOOR in R.ESCALATION_FLOORS
    assert _escalation_record(record_dir) == []


def test_an_escalation_floor_without_a_cry_wolf_block_is_refused(record_dir):
    assert _has(_escalation_record(record_dir, cry_wolf=None), "carries a `cry_wolf` block")


@pytest.mark.parametrize("over, needle", [
    ({"ratio": 2.4, "appearances": 24, "earned": 10, "earned_label": 8, "earned_proxy": 4}, "over its ceiling"),
    ({"ratio": 1.0}, "is not appearances / earned"),
    ({"earned": 0, "earned_label": 0, "earned_proxy": 0, "ratio": 0.0}, "none earned"),
    ({"earned": 25}, "inconsistent"),
    ({"earned_label": 16}, "inconsistent"),
    ({"labels_sha256": "abc"}, "64-hex"),
    ({"replay_run": ""}, "must name its run"),
    ({"reads": -1}, "must be a count"),
    ({"raised_from": 0.78}, "a raised floor takes side `ceiling`"),
])
def test_a_cry_wolf_block_that_does_not_hold_is_refused(record_dir, over, needle):
    assert _has(_escalation_record(record_dir, cry_wolf=_cry_wolf(**over)), needle)


def test_a_cry_wolf_block_with_a_key_missing_or_extra_is_refused(record_dir):
    block = _cry_wolf()
    del block["earned_proxy"]
    assert _has(_escalation_record(record_dir, cry_wolf=block), "missing ['earned_proxy']")
    assert _has(_escalation_record(record_dir, cry_wolf=_cry_wolf(slug="leak")), "extra ['slug']")


def test_nothing_shown_is_a_ratio_of_zero(record_dir):
    block = _cry_wolf(appearances=0, earned=0, earned_label=0, earned_proxy=0, ratio=0.0)
    assert _escalation_record(record_dir, cry_wolf=block) == []


@pytest.mark.parametrize("value, raised, clean, needle", [
    (0.88, 0.80, True, None),                       # raised above its boundary, inside the range
    (0.80, 0.80, False, "needs value > max(boundaries)"),
    (0.88, None, False, "needs `cry_wolf.raised_from` == max(boundaries)"),
    (0.88, 0.79, False, "needs `cry_wolf.raised_from` == max(boundaries)"),
    (0.97, 0.80, False, "raised value 0.97 lies outside run"),  # above every target: D3's escalation
])
def test_the_ceiling_side(record_dir, value, raised, clean, needle):
    problems = _escalation_record(record_dir, value=value, side="ceiling",
                                  cry_wolf=_cry_wolf(raised_from=raised))
    assert (problems == []) is clean, problems
    if needle:
        assert _has(problems, needle)


@pytest.mark.parametrize("value", [None, "0.88"])
def test_a_raised_floor_whose_value_is_not_a_number_is_reported_not_raised(record_dir, value):
    """The checker's contract: a record problem is a returned string. 5b fresh-eyes: the
    raised value's range check compared it unguarded, so a malformed value raised TypeError."""
    problems = _escalation_record(record_dir, value=value, side="ceiling",
                                  cry_wolf=_cry_wolf(raised_from=0.80))
    assert _has(problems, "needs value > max(boundaries)")


def test_the_ceiling_side_and_a_cry_wolf_block_belong_to_escalation_floors_only(record_dir):
    record, runs, _, harbor, dense = _clean(record_dir)
    record["constants"][STRONG].update(side="ceiling", value=0.76)
    assert _has(_problems(record, runs, {STRONG: 0.76}), "belongs to an escalation floor")
    record["constants"][STRONG].update(side="higher", value=0.75, cry_wolf=_cry_wolf())
    assert _has(_problems(record, runs, {STRONG: 0.75}), "arms no escalation")


# ---------------------------------------------------------------------------
# 6a: the band's methods, their item analogues, and the precision block
# ---------------------------------------------------------------------------

STRONG_BAND = "mitos.recall.SURFACE_STRONG_THRESHOLD"


def _band(cls, hits, targets=(), fid=None, space="query"):
    """A band-class result through `derive_result`, query space unless told otherwise."""
    measure = "target" if cls in R.ANSWERING_CLASSES else "top_live"
    fx = _fx(id=fid or f"{cls}-{len(hits)}", **{"class": cls}, targets=list(targets), measure=measure)
    result = R.derive_result(fx, hits, "weak", None)
    result["space"] = space
    return result


@pytest.mark.parametrize("value, above, below", [
    (0.80866736, 0.8087, 0.8086),
    (0.8087, 0.8088, 0.8087),          # an exact 4-place value: above moves up, below keeps it
    (0.6375222, 0.6376, 0.6375),       # Harbor's family-neighbour minimum floors to 0.6375
    (0.6375, 0.6376, 0.6375),
    (0.69999, 0.7, 0.6999),
    (math.nextafter(0.9998, 0), 0.9998, 0.9997),  # one ulp under a step: `x * 10_000` rounds onto it
])
def test_the_two_rounding_helpers_are_one_rule_each(value, above, below):
    assert R._strictly_above(value) == above and R._strictly_above(value) > value
    assert R._at_or_below(value) == below and R._at_or_below(value) <= value


def test_the_rounding_helpers_are_the_nearest_step_on_their_side_one_ulp_around_every_step():
    """Fresh eyes (6a): one ulp under a 4-place step, `x * 10_000` rounds onto the step, so
    `_at_or_below` once returned the step (above its input) and `_strictly_above` the step
    after it (not the smallest)."""
    for k in range(1, 10_000):
        for x in (k / 10_000, math.nextafter(k / 10_000, 0), math.nextafter(k / 10_000, 1)):
            below, above = R._at_or_below(x), R._strictly_above(x)
            assert below <= x < round(below + 0.0001, 4), x
            assert round(above - 0.0001, 4) <= x < above, x


def test_the_strong_boundary_sits_strictly_above_every_non_answering_top():
    """An exact 4-place maximum moves up a step: `strong` never fires on a measured
    non-answer (X3's tie row)."""
    results = [_band("family-neighbour", [_hit("n", 0.8)]), _band("off-axis", [_hit("o", 0.7)]),
               _band("bullseye", [_hit("t", 0.95)], ["t"])]
    assert R.strong_boundary(results) == 0.8001
    results.append(_band("off-axis", [_hit("o2", 0.80866736)]))
    assert R.strong_boundary(results) == 0.8087


def test_the_strong_boundary_reads_band_classes_in_query_space_only():
    results = [_band("family-neighbour", [_hit("n", 0.7)]),
               _band("family-neighbour", [_hit("d", 0.9)], space="document"),
               _band("bullseye", [_hit("t", 0.99)], ["t"])]
    rej = R.derive_result(_fx(id="r", **{"class": "rejection-target"}, measure="item"), [_hit("x", 0.95)], "weak", None)
    assert R.strong_boundary(results + [rej]) == 0.7001
    assert R.strong_boundary([_band("bullseye", [_hit("t", 0.9)], ["t"])]) is None


def test_the_weak_boundary_floors_the_lowest_related_top_and_reads_family_neighbours():
    """The minimum is a family-neighbour's: an answering-only population would miss it (X4)."""
    results = [_band("bullseye", [_hit("t", 0.72)], ["t"]), _band("handle-words", [_hit("t", 0.7)], ["t"]),
               _band("family-neighbour", [_hit("n", 0.6375222)]), _band("off-axis", [_hit("o", 0.5)])]
    assert R.weak_boundary(results) == 0.6375


def test_a_low_rejection_or_retired_or_document_top_does_not_move_the_weak_boundary():
    """Scout: "related" is the band classes in query space, never `class != off-axis`."""
    base = [_band("bullseye", [_hit("t", 0.72)], ["t"]), _band("family-neighbour", [_hit("n", 0.69)])]
    rej = R.derive_result(_fx(id="r", **{"class": "rejection-target"}, measure="item"), [_hit("x", 0.6163)], "weak", None)
    retired = R.derive_result(_fx(id="rp", **{"class": "retired-precedent"}), [_hit("y", 0.55)], "weak", None)
    doc = _band("bullseye", [_hit("t", 0.4)], ["t"], space="document")
    assert R.weak_boundary(base + [rej, retired, doc]) == R.weak_boundary(base) == 0.69
    assert R.weak_boundary([_band("off-axis", [_hit("o", 0.7)])]) is None


def test_a_result_with_no_live_hit_has_no_top_and_bounds_nothing():
    results = [_band("family-neighbour", [_hit("n", 0.9, live=False)]), _band("off-axis", [_hit("o", 0.7)])]
    assert R.strong_boundary(results) == 0.7001


def test_rank_one_wrong_depths_read_answering_results_whose_rank_one_is_not_a_target():
    right = _band("bullseye", [_hit("t", 0.9), _hit("n", 0.89)], ["t"])
    wrong = _band("handle-words", [_hit("n1", 0.9), _hit("n2", 0.88), _hit("t", 0.86)], ["t"])
    missing = _band("bullseye", [_hit("n", 0.9)], ["t"])  # no target in the window: nothing to measure
    neighbour = _band("family-neighbour", [_hit("n", 0.9), _hit("m", 0.5)])
    assert R.rank_one_wrong_depths([right, wrong, missing, neighbour]) == [pytest.approx(0.04)]


def _wrong(n):
    return [_band("bullseye", [_hit("n", 0.9), _hit("t", 0.9 - 0.001 * (i + 1))], ["t"], fid=f"w{i}")
            for i in range(n)]


@pytest.mark.parametrize("n, population", [(9, "pooled-inversions"), (10, "rank-one-wrong")])
def test_the_top_cluster_margin_pools_every_inversion_below_ten_rank_one_wrong_depths(n, population):
    """X6: the fallback's threshold is 10, the 2e "median of a handful" rule."""
    extra = _band("handle-words", [_hit("t1", 0.95), _hit("n", 0.9), _hit("t2", 0.5)], ["t1", "t2"])
    name, depths = R.top_cluster_depths(_wrong(n) + [extra])
    assert name == population
    assert len(depths) == (n if population == "rank-one-wrong" else n + 1)
    assert R.TOP_CLUSTER_MIN_DEPTHS == 10


def test_the_top_cluster_margin_is_the_nearest_rank_median_at_four_places():
    assert R.top_cluster_margin_boundary(_wrong(10)) == 0.005
    assert R.top_cluster_margin_boundary([_band("bullseye", [_hit("t", 0.9)], ["t"])]) is None


def test_rank_one_margin_needs_a_second_live_hit_and_fails_closed_without_one():
    r = _band("bullseye", [_hit("a", 0.9), _hit("q", 0.89, kind="open_question", live=False), _hit("b", 0.85)], ["a"])
    assert R.rank_one_margin(r) == pytest.approx(0.05)
    assert R.rank_one_margin(_band("bullseye", [_hit("a", 0.9)], ["a"])) is None


def test_derive_result_carries_a_hits_overlap_when_the_reader_measured_one():
    hits = [dict(_hit("a", 0.9), overlap=0.5), _hit("b", 0.8)]
    raw = R.derive_result(_fx(), hits, "weak", None)["hits"]
    assert raw[0]["overlap"] == 0.5 and "overlap" not in raw[1]


def _terms(text):
    return [t for t in (w.strip(".,?").casefold() for w in text.split()) if len(t) >= 3]


def test_overlap_reads_the_slug_and_the_mechanisms_hyphens_as_spaces():
    """X10's unit half: a query word that only a mechanism carries."""
    kw = {"terms": _terms, "stopwords": frozenset({"the"})}
    assert R.overlap("the harbor api rate limit", "harbor-api-rate-limit", [], **kw) == 1.0
    assert R.overlap("rate limit token bucket", "harbor-api-rate-limit", ["token-bucket"], **kw) == 1.0
    assert R.overlap("rate limit token bucket", "harbor-api-rate-limit", [], **kw) == 0.5
    assert R.overlap("the", "harbor-api-rate-limit", [], **kw) is None


@pytest.mark.parametrize("k, n, interval", [
    (0, 0, [0.0, 1.0]),
    (8, 9, [0.565, 0.9801]),
    (36, 36, [0.9036, 1.0]),
    (0, 5, [0.0, 0.4345]),
])
def test_wilson_interval(k, n, interval):
    assert R.wilson(k, n) == interval


def _precision_corpus():
    return [
        _band("bullseye", [_hit("t", 0.9)], ["t"], fid="b1"),                   # strong, right
        _band("handle-words", [_hit("n", 0.85), _hit("t", 0.84)], ["t"], fid="h1"),  # strong, wrong
        _band("bullseye", [_hit("t", 0.7)], ["t"], fid="b2"),                   # weak, right
        _band("family-neighbour", [_hit("n", 0.81)], ["n"], fid="f1"),          # strong, never right
        _band("off-axis", [_hit("o", 0.5)], fid="o1"),                          # none
    ]


def test_band_precision_states_each_band_with_counts_interval_and_base():
    table = R.band_precision(_precision_corpus(), strong=0.8, weak=0.6)
    s, w, n = (table["bands"][b] for b in R.BANDS)
    assert (s["fires"], s["right"], s["precision"]) == (3, 1, 0.3333)
    assert s["by_group"] == {"bullseye": 1, "handle-words": 1, "family-neighbour": 1, "off-axis": 0}
    assert (w["fires"], w["right"], w["precision"], w["interval"]) == (1, 1, 1.0, R.wilson(1, 1))
    assert (table["n"], table["right"], table["base"]) == (5, 2, 0.4)
    assert n["fires"] == 1 and n["by_group"]["off-axis"] == 1


def test_a_family_neighbour_rank_one_is_never_right_even_when_it_lists_the_hit():
    """X8: a non-answering class holds nothing that answers, whatever its labels say."""
    table = R.band_precision([_band("family-neighbour", [_hit("n", 0.9)], ["n"])], strong=0.8, weak=0.6)
    assert table["bands"]["strong"]["right"] == 0 and table["right"] == 0


def test_a_band_with_no_fires_states_null_precision_and_the_whole_interval():
    table = R.band_precision(_precision_corpus(), strong=0.95, weak=0.0)
    assert table["bands"]["strong"] == {"fires": 0, "right": 0, "precision": None, "interval": [0.0, 1.0],
                                        "by_group": {c: 0 for c in R.BAND_CLASSES}}
    assert R.precision_entry(table) == {"fires": 0, "right": 0, "precision": None,
                                        "interval": [0.0, 1.0], "base": 0.4}


def test_band_of_cuts_as_the_classifier_does():
    assert [R.band_of(v, strong=0.8, weak=0.6) for v in (0.8, 0.7999, 0.6, 0.5999, None)] == \
        ["strong", "weak", "weak", "none", "none"]


def _item(form, top, targets, negatives=(), hits=None):
    """A rejection-target result: `targets` as (value, rank), `negatives` as values."""
    return {"id": f"r-{form}-{top}", "class": "rejection-target", "measure": "item", "space": "query",
            "rejection_form": form,
            "items": {"per_item": {
                "hits": hits or [], "top_value": top,
                "targets": [{"owner": "o", "option": "A", "value": v, "rank": k, "absent": v is None}
                            for v, k in targets],
                "negatives": [{"owner": "g", "option": "N", "value": v, "rank": 9, "kind": "vocabulary"}
                              for v in negatives]}}}


def test_the_item_weak_boundary_reads_every_form_including_does_not_carry():
    """X17: the lookup's question is option-level, so `elsewhere-does-not-carry` counts."""
    results = [_item("own-question", 0.8, [(0.8, 1)]), _item("elsewhere-carries", 0.75, [(0.7, 2)]),
               _item("elsewhere-does-not-carry", 0.6304111, [(0.6, 3)])]
    assert R.item_weak_boundary(results) == 0.6304
    assert R.item_strong_boundary([_item("own-question", 0.9, [(0.9, 1)], [0.7366])]) == 0.7367


def test_item_band_precision_is_right_only_when_a_target_item_ranks_first():
    results = [_item("own-question", 0.9, [(0.9, 1)]), _item("elsewhere-does-not-carry", 0.85, [(0.8, 2)]),
               _item("elsewhere-carries", 0.7, [(0.7, 1)])]
    table = R.item_band_precision(results, strong=0.8, weak=0.6)
    s = table["bands"]["strong"]
    assert (s["fires"], s["right"]) == (2, 1)
    assert s["by_group"] == {"own-question": 1, "elsewhere-carries": 0, "elsewhere-does-not-carry": 1}
    assert table["base"] == 0.6667


def test_item_depths_read_collapsed_ranks_every_form():
    hits = [_ih("x", 0.9, 1, 0), _ih("o", 0.85, 1, 0), _ih("y", 0.8, 1, 0), _ih("o", 0.75, 2, 0)]
    r = _item("elsewhere-does-not-carry", 0.9, [(0.85, 2), (0.75, 4)], hits=hits)
    assert R.item_rank_one_wrong_depths([r]) == [pytest.approx(0.05)]
    assert R.item_inversion_depths([r]) == [pytest.approx(0.05), pytest.approx(0.15)]
    assert R.item_top_cluster_depths([r]) == ("pooled-inversions", R.item_inversion_depths([r]))
    assert R.item_edge_margin_boundary([r]) == 0.05


# -- the record's precision block (D7) --------------------------------------------------------

def _precision_part(fires=9, right=8, base=0.3929, **over):
    part = {"fires": fires, "right": right, "precision": round(right / fires, 4) if fires else None,
            "interval": R.wilson(right, fires), "base": base}
    part.update(over)
    return part


def _band_record(record_dir, precision="default", weak_floor=None):
    """A record with one harbor and one dense run over the four band classes, and the
    strong threshold's entry; ``weak_floor`` adds a weak entry floored at that value."""
    classes = {"bullseye": {"query": _class_block([0.71, 0.84, 0.9])},
               "handle-words": {"query": _class_block([0.8, 0.86])},
               "family-neighbour": {"query": _class_block([0.6375222, 0.7, 0.8072], "top_live")},
               "off-axis": {"query": _class_block([0.66, 0.7], "top_live")}}
    harbor = R.append_run(_run(classes=classes), record_dir)
    dense = R.append_run(_run(corpus="dense", decision_count=500, classes=classes), record_dir)
    record = R.load_record(record_dir)
    entry = {"value": 0.8073, "kind": "threshold", "space": "query", "side": "higher",
             "separates": list(R.BAND_CLASSES), "boundaries": {harbor: 0.8073, dense: 0.8073},
             "method": "synthetic"}
    if precision == "default":
        entry["precision"] = {harbor: _precision_part(), dense: _precision_part(36, 36, 0.7956),
                              "replay": _precision_part(4, 3, 0.5, run="20261017-0000-abc1234-dogfood-copy",
                                                        proxy="rank-1 USED, record-followed reads")}
    elif precision is not None:
        entry["precision"] = precision(harbor, dense)
    record["constants"][STRONG_BAND] = entry
    constants = {STRONG_BAND: 0.8073}
    if weak_floor is not None:
        name = "mitos.recall.SURFACE_WEAK_THRESHOLD"
        record["constants"][name] = {"value": weak_floor, "kind": "threshold", "space": "query",
                                     "side": "lower", "separates": list(R.BAND_CLASSES),
                                     "boundaries": {harbor: weak_floor, dense: weak_floor}, "method": "synthetic"}
        constants[name] = weak_floor
    return _problems(record, R.load_cited_runs(record, record_dir), constants), harbor, dense


def test_a_stated_precision_block_is_clean(record_dir):
    assert STRONG_BAND in R.PRECISION_STATED and STRONG_BAND in R.REPLAY_PRECISION
    assert _band_record(record_dir)[0] == []


def test_a_precision_stated_constant_without_its_block_is_refused(record_dir):
    assert _has(_band_record(record_dir, precision=None)[0], "states its precision in a `precision` block")


def test_a_cited_runs_missing_precision_key_is_refused(record_dir):
    """X7: a re-derivation that forgets to restate one corpus reds the offline tier."""
    problems, _, _ = _band_record(record_dir, precision=lambda h, d: {
        h: _precision_part(), "replay": _precision_part(run="r", proxy="p")})
    assert any("`precision` has no 'dense-" in p for p in problems)


def test_the_replay_key_is_owed_inside_replay_precision_and_refused_outside(record_dir, monkeypatch):
    problems, _, _ = _band_record(record_dir, precision=lambda h, d: {h: _precision_part(), d: _precision_part()})
    assert _has(problems, "`precision` has no 'replay' key")
    monkeypatch.setattr(R, "REPLAY_PRECISION", ())
    problems, _, _ = _band_record(record_dir)
    assert _has(problems, "is not measured on traffic")


def test_a_precision_block_on_a_constant_that_states_none_is_refused(record_dir, monkeypatch):
    monkeypatch.setattr(R, "PRECISION_STATED", ())
    assert _has(_band_record(record_dir)[0], "states no precision")


@pytest.mark.parametrize("part, needle", [
    (_precision_part(precision=0.5), "is not right / fires"),
    (dict(_precision_part(), fires=3, right=4), "fires >= right >= 0"),
    (_precision_part(interval=[0.9, 0.5]), "0 <= low <= high <= 1"),
    (_precision_part(base=1.2), "a rate in [0, 1] or null"),
    (dict(_precision_part(), extra=1), "keys must be exactly"),
    # X16: a line above every fixture states the null form, never a division
    (_precision_part(fires=0, right=0, precision=0.0), "fires nowhere"),
    (_precision_part(fires=0, right=0, interval=[0.0, 0.0]), "fires nowhere"),
])
def test_a_malformed_precision_part_is_reported(record_dir, part, needle):
    problems, _, _ = _band_record(record_dir, precision=lambda h, d: {
        h: part, d: _precision_part(), "replay": _precision_part(run="r", proxy="p")})
    assert _has(problems, needle), problems


def test_a_fires_zero_block_is_accepted_in_its_exact_null_form(record_dir):
    problems, _, _ = _band_record(record_dir, precision=lambda h, d: {
        h: _precision_part(fires=0, right=0), d: _precision_part(),
        "replay": _precision_part(run="r", proxy="p")})
    assert problems == []


@pytest.mark.parametrize("weak, clean", [
    (0.6375, True),    # floored just under the 0.6375222 class minimum: inside the envelope
    (0.6374, False),   # two steps under the raw minimum (X15's reverse): outside
])
def test_the_range_envelope_admits_a_floored_weak_and_refuses_a_step_further(record_dir, weak, clean):
    problems = _band_record(record_dir, weak_floor=weak)[0]
    assert (not any("SURFACE_WEAK_THRESHOLD" in p for p in problems)) is clean, problems


@pytest.mark.parametrize("value, clean", [(0.9001, True), (0.9002, False)])
def test_the_range_envelope_admits_one_step_past_a_maximum_and_refuses_two(record_dir, value, clean):
    record, runs, _, harbor, dense = _clean(record_dir)  # bullseye's maximum is 0.9
    record["constants"][STRONG].update(value=value, side="higher", boundaries={harbor: value, dense: 0.73})
    problems = _problems(record, runs, {STRONG: value})
    assert (not _has(problems, "observed range")) is clean, problems


# ---------------------------------------------------------------------------
# 7a: the pause's two floors — recall lines in document space
# ---------------------------------------------------------------------------

def _draft(cls, hits, targets, fid, measure="target", negatives=()):
    fx = _fx(id=fid, **{"class": cls}, form="draft", targets=list(targets),
             negatives=list(negatives), measure=measure)
    return R.derive_result(fx, hits, None, 0.8)


def test_the_tombstone_line_sits_at_or_below_every_retired_precedent_draft_target():
    results = [
        _draft("retired-precedent", [_hit("r1", 0.83456), _hit("d2", 0.9)], ["r1"], "a"),
        _draft("retired-precedent", [_hit("r2", 0.9123)], ["r2"], "b"),
        # A query-form retired-precedent read is another space: never the draft line.
        R.derive_result(_fx(id="q", **{"class": "retired-precedent"}, targets=["r3"]),
                        [_hit("r3", 0.61)], "weak", None),
        # Another class in document space is not a re-tread.
        _draft("bullseye", [_hit("b1", 0.7)], ["b1"], "c"),
    ]
    assert R.tombstone_floor_boundary(results) == 0.8345
    assert R.pause_floor_boundary(results) == 0.7


def test_the_two_lines_read_absent_targets_as_nothing_and_an_empty_class_as_none():
    results = [_draft("retired-precedent", [_hit("d2", 0.9)], ["r-missing"], "a")]
    assert results[0]["targets"][0]["absent"] is True
    assert R.tombstone_floor_boundary(results) is None
    assert R.pause_floor_boundary([]) is None


def test_a_line_already_on_a_four_place_step_is_its_own_line():
    results = [_draft("bullseye", [_hit("b1", 0.8175)], ["b1"], "a")]
    assert R.pause_floor_boundary(results) == 0.8175
