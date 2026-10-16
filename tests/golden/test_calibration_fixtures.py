"""Offline integrity rows for Layer C's committed fixtures, over the built Harbor graph.

No service is touched: Harbor is built from the reference corpus into a temp SQLite file
and the harness's service-free half (`index_graph`, `resolve_item_refs`) reads it. These
rows keep the labelled set honest between live runs: every slug and item ref names
something real, every (class, space) holds its minimum, the rejection-target class keeps
the shape its floors are set from, and the corpus edits it measures stay numbered.
"""

import os
import sys
from collections import Counter

import pytest

from mitos.rejections import enumerate_rejected_paths

GOLDEN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, GOLDEN)
from _harness import build_reference_graph  # noqa: E402
import _calibration_harness as CH  # noqa: E402
import _calibration_record as R  # noqa: E402

HUB = "harbor-storage-is-sqlite"

# Phase 1c's minimum per (class, space).
MINIMUMS = {
    ("bullseye", "query"): 6,
    ("bullseye", "document"): 3,
    ("handle-words", "query"): 5,
    ("family-neighbour", "query"): 6,
    ("family-neighbour", "document"): 4,
    ("off-axis", "query"): 6,
    ("rejection-target", "query"): 10,
    ("retired-precedent", "query"): 4,
    ("retired-precedent", "document"): 5,
}
FORM_MINIMUMS = {"own-question": 4, "elsewhere-carries": 3, "elsewhere-does-not-carry": 3}

# The generic items (do nothing, defer, status quo), named here rather than guessed from
# their text. Each is a hard negative of the rejection floor and never a target.
GENERIC_ITEMS = {
    ("harbor-storage-is-sqlite", "Keeping the status quo"),
    ("harbor-blob-encryption-at-rest", "Keeping the status quo"),
    ("harbor-backup-nightly", "Do nothing"),
    ("harbor-webhook-delivery", "Do nothing"),
    ("harbor-api-rate-limit", "Defer"),
    ("harbor-backup-nightly", "Defer"),
    ("harbor-observability-otel", "Defer"),
}

# The **Rejected:** lines 1c rewrote into the numbered form, with their item counts.
EDITED_FIELDS = {
    "harbor-storage-is-sqlite": 4,
    "harbor-api-rate-limit": 4,
    "harbor-backup-nightly": 3,
    "harbor-webhook-delivery": 4,
    "harbor-blob-encryption-at-rest": 3,
    "harbor-observability-otel": 3,
}


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    store = build_reference_graph(str(tmp_path_factory.mktemp("layerc-fixtures") / "graph.sqlite"))
    return CH.index_graph(store)


@pytest.fixture(scope="module")
def fixtures():
    return R.load_fixtures()


def _rejection(fixtures):
    return [f for f in fixtures if f["class"] == "rejection-target"]


def _refs(fx, key):
    return {(r["owner"], r["option"]) for r in fx.get(key, [])}


def test_every_slug_and_item_ref_names_the_graph(index, fixtures):
    R.check_fixture_slugs(fixtures, index.nodes_by_slug)
    for fx in _rejection(fixtures):
        refs = CH.resolve_item_refs(index, fx)
        assert len(refs["targets"]) == len(fx["target_items"])
        assert len(refs["negatives"]) == len(fx["negative_items"])
        for ref in refs["targets"] + refs["negatives"]:
            assert ref["part_ids"], f"{fx['id']}: {ref['owner']} {ref['option']!r} resolved to no point"


def test_each_class_and_space_meets_its_minimum(fixtures):
    counts = Counter((f["class"], R.SPACE_OF_FORM[f["form"]]) for f in fixtures)
    short = {k: (counts.get(k, 0), n) for k, n in MINIMUMS.items() if counts.get(k, 0) < n}
    assert not short, f"(class, space): (have, need) {short}"
    assert set(counts) == set(MINIMUMS), f"a (class, space) outside the plan: {set(counts) - set(MINIMUMS)}"


def test_the_rejection_target_class_keeps_its_shape(fixtures):
    rejection = _rejection(fixtures)
    forms = Counter(f["rejection_form"] for f in rejection)
    assert all(forms[form] >= n for form, n in FORM_MINIMUMS.items()), forms
    uses = Counter()
    for fx in rejection:
        assert fx["negative_items"], f"{fx['id']}: names no negative item"
        assert not _refs(fx, "target_items") & GENERIC_ITEMS, f"{fx['id']}: a generic item as a target"
        uses.update(_refs(fx, "negative_items") & GENERIC_ITEMS)
    thin = {item: uses[item] for item in GENERIC_ITEMS if uses[item] < 2}
    assert not thin, f"each generic item is a negative at least twice: {thin}"


def test_no_fixture_names_the_hub_as_a_negative(fixtures):
    """The hub rule covers slug ``negatives`` only (LAYER_B_FINDINGS §1).

    A ``negative_items`` ref owned by the hub names one of its rejected items, not the
    decision, and the same-owner generic negatives need exactly that; it is not a breach.
    """
    named = [f["id"] for f in fixtures if HUB in f["negatives"]]
    assert not named, f"{HUB} is a hub and never a hard negative: {named}"


def test_retired_targets_and_tombstone_negatives_are_retired(index, fixtures):
    def retired(slug):
        node = index.nodes_by_slug[slug]
        return node["kind"] == "decision" and index.store.get_node_state(node["id"]) not in CH.LIVE_STATES

    for fx in fixtures:
        if fx["class"] == "retired-precedent":
            assert all(retired(s) for s in fx["targets"]), f"{fx['id']}: a target is not retired"
        if fx["class"] == "family-neighbour" and fx["form"] == "draft":
            # 7a reads these as the tombstone floor's non-re-tread side; a live value contaminates it.
            assert all(retired(s) for s in fx["negatives"]), f"{fx['id']}: a negative is not retired"
    drafts = [f for f in fixtures if f["class"] == "family-neighbour" and f["form"] == "draft" and f["negatives"]]
    assert len(drafts) >= 2


def test_two_rejection_targets_name_a_live_adopter(index, fixtures):
    adopters = [
        fx["id"] for fx in _rejection(fixtures)
        if any(index.live_by_slug[s] for s in set(fx["targets"]) - {o for o, _ in _refs(fx, "target_items")})
    ]
    assert len(adopters) >= 2, adopters


def test_two_rejection_targets_carry_a_same_owner_negative(fixtures):
    same = [fx["id"] for fx in _rejection(fixtures)
            if {o for o, _ in _refs(fx, "target_items")} & {o for o, _ in _refs(fx, "negative_items")}]
    assert len(same) >= 2, same


def test_every_edited_field_enumerates_as_numbered(index):
    """Guards the corpus edit: a hand edit back to prose would silently unmeasure a field."""
    for slug, count in EDITED_FIELDS.items():
        enum = enumerate_rejected_paths(index.nodes_by_slug[slug]["rejected_paths"])
        assert (enum.shape, len(enum.items)) == ("numbered", count), (slug, enum.shape, len(enum.items))


def _probe(**refs):
    return {"id": "probe", "target_items": [], "negative_items": [], **refs}


@pytest.mark.parametrize("ref, needle", [
    ({"owner": "harbor-api-rate-limit", "option": "Per-"}, "matches 2 items"),
    ({"owner": "harbor-api-rate-limit", "option": "per-IP"}, "matches 0 items"),
    ({"owner": "oq-harbor-backup-cadence", "option": "Nightly"}, "is not a decision"),
    ({"owner": "harbor-nowhere", "option": "Postgres"}, "is not a decision"),
])
def test_the_resolver_refuses_a_ref_that_is_not_one_item(index, ref, needle):
    with pytest.raises(ValueError) as err:
        CH.resolve_item_refs(index, _probe(target_items=[ref]))
    assert needle in str(err.value) and "'probe'" in str(err.value)


def test_the_resolver_names_one_item_by_its_option_prefix(index):
    refs = CH.resolve_item_refs(index, _probe(target_items=[{"owner": "harbor-api-rate-limit", "option": "Per-IP"}]))
    assert [(r["ordinal"], len(r["part_ids"])) for r in refs["targets"]] == [(2, 1)]


# --------------------------------------------------------------------------- #
# Phase 2a — the harness joins hits by point id, so an ids-only read labels too
# --------------------------------------------------------------------------- #

def test_label_hits_joins_an_ids_only_read_by_point_id(index):
    """The production ids-only shape (``{point_id, score}``) carries no payload slug;
    ``label_hits`` labels it through the ``hash_to_uuid(node_id)`` map, with the
    node's slug, kind and liveness. The proxy forwards ``with_payload`` to the store
    it wraps (bound explicitly: the module-scoped graph predates the per-test record)."""
    from _point_hits import CannedVector

    matches = [{"slug": n["slug"], "score": 0.9 - 0.001 * i} for i, n in enumerate(index.nodes)]
    proxy = CH.RecordingVectorStore(CannedVector(matches, store=index.store))
    raw = proxy.query([0.1], limit=len(matches), with_payload=False)
    assert all(set(h) == {"point_id", "score"} for h in raw)
    assert proxy.calls[0][0] == len(matches)

    first = index.nodes[0]["slug"]
    hits = CH.label_hits(index, raw, [first])

    assert [h["slug"] for h in hits] == [n["slug"] for n in index.nodes]
    assert [h["kind"] for h in hits] == [n["kind"] for n in index.nodes]
    assert [h["live"] for h in hits] == [index.live_by_slug[n["slug"]] for n in index.nodes]
    assert [h["self"] for h in hits] == [s == first for s in (n["slug"] for n in index.nodes)]
    assert {False, True} <= {h["live"] for h in hits}  # retired nodes are in the window


def test_label_hits_refuses_a_point_the_collection_was_not_built_from(index):
    from _point_hits import orphan_point_id

    with pytest.raises(AssertionError, match="not a point this collection was built from"):
        CH.label_hits(index, [{"point_id": orphan_point_id("stray"), "score": 0.5}], [])


# --------------------------------------------------------------------------- #
# Phase 2c — the query form reads through the shared identity walk
# --------------------------------------------------------------------------- #

def test_the_query_form_reads_one_ids_only_call_at_the_bound_and_labels_its_window(
        index, tmp_path, monkeypatch):
    """§9.6 offline: `read_query_form` drives `surface_decisions`, which now asks the
    proxy once at the over-fetch bound with ids only, so the proxy holds every Harbor
    point; the labelled hits are the first `RECORD_LIMIT` of that ranking."""
    import dataclasses

    from _point_hits import CannedVector
    from mitos.cli import cmd_init
    from mitos.config import MitosConfig
    from mitos.retrieval_walk import RANKED_OVERFETCH_BOUND

    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    cmd_init(MitosConfig(workspace))

    class _Embed:
        def get_embedding(self, text, is_query=False):
            return [0.1, 0.2]

    matches = [{"slug": n["slug"], "score": 0.9 - 0.001 * i} for i, n in enumerate(index.nodes)]
    probe = dataclasses.replace(index, provider=_Embed(), workspace=workspace,
                                vstore=CannedVector(matches, store=index.store))
    fixture = {"id": "walk-probe", "form": "query", "measure": "node", "text": "probe",
               "class": "probe", "targets": [index.nodes[0]["slug"]], "negatives": [],
               "exclude": []}
    read = CH.read_query_form(probe, fixture)

    assert len(read.calls) == 1
    limit, raw, with_payload = read.calls[0]
    assert (limit, with_payload) == (RANKED_OVERFETCH_BOUND, False)
    assert len(raw) == len(index.nodes) > CH.RECORD_LIMIT
    assert [h["slug"] for h in read.result["hits"]] == \
        [n["slug"] for n in index.nodes[:CH.RECORD_LIMIT]]
    assert CH.READER == "mcp.surface_decisions(retrieval_walk)+conflict.gather_candidates"
