"""Offline integrity rows for Layer C's committed fixtures, over the built Harbor graph.

No service is touched: Harbor is built from the reference corpus into a temp SQLite file
and the harness's service-free half (`index_graph`, `resolve_item_refs`) reads it. These
rows keep the labelled set honest between live runs: every slug and item ref names
something real, every (class, space) holds its minimum, the rejection-target class keeps
the shape its floors are set from, and the corpus edits it measures stay numbered.
"""

import math
import os
import sys
from collections import Counter
from typing import List

import pytest

from mitos.rejections import enumerate_rejected_paths

GOLDEN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, GOLDEN)
from _harness import build_reference_graph  # noqa: E402
import _calibration_harness as CH  # noqa: E402
import _calibration_record as R  # noqa: E402

sys.path.insert(0, os.path.dirname(GOLDEN))  # tests/, for the shared rejection-index double
from _rejection_index_double import FakeRejectionIndex  # noqa: E402

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
# Raised by 5b: carries and does-not-carry each hold four, so the floor's recall
# (own + carries) and its cry-wolf reading (does-not-carry counts as not earned) both
# rest on more than three cases.
FORM_MINIMUMS = {"own-question": 4, "elsewhere-carries": 4, "elsewhere-does-not-carry": 4}
# 5b: at least this many rejection-target fixtures carry a vocabulary negative.
VOCABULARY_MINIMUM = 8

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


def test_every_negative_item_names_its_kind_and_generic_means_the_named_set(fixtures):
    """5b D1: a negative is generic (one of `GENERIC_ITEMS`, named, not guessed) or
    vocabulary; validation requires the key, this row requires the two to agree."""
    for fx in _rejection(fixtures):
        for ref in fx["negative_items"]:
            item = (ref["owner"], ref["option"])
            assert ref["kind"] in R.NEGATIVE_KINDS, (fx["id"], ref)
            assert (ref["kind"] == "generic") == (item in GENERIC_ITEMS), (
                f"{fx['id']}: {item} is kind {ref['kind']!r}; generic means one of GENERIC_ITEMS"
            )


def test_vocabulary_negatives_are_another_decisions_other_item(fixtures):
    """A vocabulary negative is a non-generic item of another decision: never the
    fixture's own target item, never owned by a target item's owner, and present on at
    least `VOCABULARY_MINIMUM` fixtures."""
    carrying = []
    for fx in _rejection(fixtures):
        vocabulary = {(r["owner"], r["option"]) for r in fx["negative_items"] if r["kind"] == "vocabulary"}
        if vocabulary:
            carrying.append(fx["id"])
        assert not vocabulary & _refs(fx, "target_items"), f"{fx['id']}: its own target as a vocabulary negative"
        target_owners = {o for o, _ in _refs(fx, "target_items")}
        assert not {o for o, _ in vocabulary} & target_owners, (
            f"{fx['id']}: a vocabulary negative owned by the target's own decision"
        )
        assert not vocabulary & GENERIC_ITEMS, f"{fx['id']}: a generic item tagged vocabulary"
    assert len(carrying) >= VOCABULARY_MINIMUM, (len(carrying), carrying)


@pytest.mark.parametrize("negative, needle", [
    ({"owner": "harbor-backup-nightly", "option": "Do nothing"}, "{owner, option, kind}"),
    ({"owner": "harbor-backup-nightly", "option": "Do nothing", "kind": "noise"}, "kind one of"),
])
def test_a_negative_item_without_a_known_kind_is_refused(negative, needle):
    fixture = {"id": "kind-probe", "class": "rejection-target", "form": "query", "text": "probe",
               "targets": ["harbor-storage-is-sqlite"], "negatives": [], "measure": "item",
               "notes": "probe", "rejection_form": "own-question",
               "target_items": [{"owner": "harbor-storage-is-sqlite", "option": "Postgres"}],
               "negative_items": [negative]}
    with pytest.raises(ValueError) as err:
        R.validate_fixtures([fixture])
    assert needle in str(err.value) and "negative_items" in str(err.value)


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
    assert CH.READER == ("mcp.surface_decisions(retrieval_walk+rejection_leg)"
                         "+conflict.gather_candidates(retrieval_walk)+sync._review_tombstones")
    # No per-item index bound: the autouse seam's empty index answers, nothing is compared.
    assert (read.result["section"], read.result["section_state"], read.item_ranking) == ([], None, None)


# --------------------------------------------------------------------------- #
# Phase 2d — the draft form's gather reads through the same walk
# --------------------------------------------------------------------------- #

def test_the_draft_form_reads_one_ids_only_call_at_the_raw_bound_and_slices_its_window(
        index, monkeypatch):
    """The draft-form twin of the row above: `read_draft_form` drives the gather, which
    asks the proxy once at `CONFLICT_RAW_BOUND` with ids only, so the proxy holds every
    Harbor point; the recorded raw window is its first `CONFLICT_OVERFETCH_LIMIT` hits,
    the size it always was, and `candidates` carries the `Gathering`."""
    import dataclasses

    from _point_hits import CannedVector
    from mitos import conflict

    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    class _Embed:
        def get_embedding(self, text, is_query=False):
            return [0.1, 0.2]

    matches = [{"slug": n["slug"], "score": 0.9 - 0.001 * i} for i, n in enumerate(index.nodes)]
    probe = dataclasses.replace(index, provider=_Embed(),
                                vstore=CannedVector(matches, store=index.store))
    fixture = {"id": "gather-probe", "form": "draft", "measure": "node", "text": "probe",
               "class": "probe", "targets": [index.nodes[0]["slug"]], "negatives": [],
               "exclude": []}
    read = CH.read_draft_form(probe, fixture)

    assert len(read.calls) == 1
    limit, raw, with_payload = read.calls[0]
    assert (limit, with_payload) == (conflict.CONFLICT_RAW_BOUND, False)
    assert len(raw) == len(index.nodes) > conflict.CONFLICT_OVERFETCH_LIMIT
    assert [h["slug"] for h in read.result["hits"]] == \
        [n["slug"] for n in index.nodes[:conflict.CONFLICT_OVERFETCH_LIMIT]]
    assert isinstance(read.candidates, conflict.Gathering)
    assert len(read.candidates.live) <= conflict.CONFLICT_OVERFETCH_LIMIT


# --------------------------------------------------------------------------- #
# Phase 7a — draft values from the whole fetch; the shipped tombstone review checked
# --------------------------------------------------------------------------- #

def _draft_probe(index, monkeypatch, matches):
    import dataclasses

    from _point_hits import CannedVector

    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    class _Embed:
        def get_embedding(self, text, is_query=False):
            return [0.1, 0.2]

    return dataclasses.replace(index, provider=_Embed(),
                               vstore=CannedVector(matches, store=index.store))


def _retired_slug(index):
    return next(n["slug"] for n in index.nodes
                if n["kind"] == "decision" and not index.live_by_slug[n["slug"]])


def test_a_draft_target_past_the_twentieth_raw_point_is_valued_where_it_sits(index, monkeypatch):
    from mitos import conflict
    matches = [{"slug": n["slug"], "score": 0.9 - 0.001 * i} for i, n in enumerate(index.nodes)]
    deep = index.nodes[conflict.CONFLICT_OVERFETCH_LIMIT + 3]["slug"]
    fixture = {"id": "deep-probe", "form": "draft", "measure": "target", "text": "probe",
               "class": "probe", "targets": [deep], "negatives": [], "exclude": []}
    read = CH.read_draft_form(_draft_probe(index, monkeypatch, matches), fixture)
    target = read.result["targets"][0]
    assert target["absent"] is False and target["raw_rank"] == conflict.CONFLICT_OVERFETCH_LIMIT + 4
    assert target["value"] == pytest.approx(0.9 - 0.001 * (conflict.CONFLICT_OVERFETCH_LIMIT + 3))
    assert len(read.result["hits"]) == conflict.CONFLICT_OVERFETCH_LIMIT


def test_a_retired_precedent_read_asks_the_shipped_tombstone_review(index, monkeypatch):
    from mitos import sync
    retired = _retired_slug(index)
    others = [n["slug"] for n in index.nodes if n["slug"] != retired]
    matches = ([{"slug": retired, "score": sync._TOMBSTONE_FLOOR + 0.05}]
               + [{"slug": s, "score": 0.5 - 0.001 * i} for i, s in enumerate(others)])
    fixture = {"id": "retired-probe", "form": "draft", "measure": "target", "text": "probe",
               "class": "retired-precedent", "targets": [retired], "negatives": [],
               "exclude": []}
    probe = _draft_probe(index, monkeypatch, matches)
    read = CH.read_draft_form(probe, fixture)
    assert read.result["pause_floor"] == sync._TOMBSTONE_FLOOR
    assert read.result["targets"][0]["clears_pause_floor"] is True

    def silent(self, gathering, declared, **kw):
        return sync.TombstoneReview(shown=[], omitted=0, cleared_by=set())
    monkeypatch.setattr(sync.MitosSyncManager, "_review_tombstones", silent)
    with pytest.raises(CH.TombstoneDisagreement, match="retired-probe"):
        CH.read_draft_form(_draft_probe(index, monkeypatch, matches), fixture)


def test_a_retired_target_below_the_floor_owes_no_tombstone(index, monkeypatch):
    from mitos import sync
    retired = _retired_slug(index)
    matches = [{"slug": retired, "score": sync._TOMBSTONE_FLOOR - 0.05}]
    fixture = {"id": "retired-low", "form": "draft", "measure": "target", "text": "probe",
               "class": "retired-precedent", "targets": [retired], "negatives": [],
               "exclude": []}
    read = CH.read_draft_form(_draft_probe(index, monkeypatch, matches), fixture)
    assert read.result["targets"][0]["clears_pause_floor"] is False


# --------------------------------------------------------------------------- #
# Phase 5b — items read through the shipped client; the shipped section checked
# --------------------------------------------------------------------------- #

def _query_angle_vector(score: float) -> List[float]:
    """A 2-d vector whose cosine with `_Embed`'s query vector is ``score``."""
    alpha = math.atan2(0.2, 0.1)
    delta = math.acos(score)
    return [math.cos(alpha + delta), math.sin(alpha + delta)]


def _item_probe(index, tmp_path, monkeypatch, client):
    """`index` with a stub embedder, a canned decision store, a workspace, and ``client``
    bound as the per-item index, seeded with every Harbor rejection point at distinct
    scores (0.99 down by 0.02 in map order, so some sit above the floor and most below)."""
    import dataclasses

    from _point_hits import CannedVector
    from mitos.cli import cmd_init
    from mitos.config import MitosConfig
    from mitos.vector_store import RejectionPoint

    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    cmd_init(MitosConfig(workspace))

    class _Embed:
        def get_embedding(self, text, is_query=False):
            return [0.1, 0.2]

    points = CH.item_points_of(index)
    client.upsert_points(
        [RejectionPoint(point_id=pid, owner_id=index.nodes_by_slug[slug]["id"],
                        vector=_query_angle_vector(0.99 - 0.02 * i))
         for i, (pid, (slug, _, _)) in enumerate(points.items())],
        model="test-embedding-model", may_create=True,
    )
    matches = [{"slug": n["slug"], "score": 0.9 - 0.001 * i} for i, n in enumerate(index.nodes)]
    return dataclasses.replace(index, provider=_Embed(), workspace=workspace,
                               vstore=CannedVector(matches, store=index.store),
                               items=CH.ItemIndex(client, points))


def _fixture(fixtures, fid):
    return next(f for f in fixtures if f["id"] == fid)


def test_the_item_reader_labels_every_point_by_the_harness_map(index, fixtures, tmp_path, monkeypatch):
    """`read_rejection_items` reads per_item through the client, every point, labelled by
    `item_points_of`; with no Harbor shapes bound it reads no per_decision shape."""
    probe = _item_probe(index, tmp_path, monkeypatch, FakeRejectionIndex(present=True))
    read = CH.read_query_form(probe, _fixture(fixtures, "rejection-own-postgres"))

    assert read.shape_searches == {"per_item": 1}
    assert set(read.result["items"]) == {"per_item"}
    hits = read.result["items"]["per_item"]["hits"]
    assert len(hits) == len(probe.items.points) == len(read.item_ranking)
    for h in hits:
        assert (h["owner"], h["ordinal"], h["part"]) == probe.items.points[h["point_id"]]
    negatives = read.result["items"]["per_item"]["negatives"]
    assert [n["kind"] for n in negatives] == [r["kind"] for r in _fixture(fixtures, "rejection-own-postgres")["negative_items"]]
    assert all(not r["absent"] for r in read.result["items"]["per_item"]["targets"] + negatives)


@pytest.mark.parametrize("fid", ["rejection-own-postgres", "bullseye-rate-limit-per-minute"])
def test_the_shipped_section_agrees_with_the_instrument(index, fixtures, tmp_path, monkeypatch, fid):
    """A faithful index: the envelope's section is the instrument's ranking cut at the
    floor and the bound, for a rejection target and for any other query fixture."""
    from mitos.recall import REJECTION_MATCH_FLOOR, REJECTION_SECTION_BOUND

    fixture = _fixture(fixtures, fid)
    probe = _item_probe(index, tmp_path, monkeypatch, FakeRejectionIndex(present=True))
    read = CH.read_query_form(probe, fixture)

    section = read.result["section"]
    assert read.result["section_state"] is None
    assert 0 < len(section) == REJECTION_SECTION_BOUND
    predicted = [p for p in read.item_ranking if p["score"] >= REJECTION_MATCH_FLOOR][:REJECTION_SECTION_BOUND]
    assert [(m["slug"], m["ordinal"]) for m in section] == [(p["slug"], p["ordinal"]) for p in predicted]


class _LegSwapsTwo(FakeRejectionIndex):
    """The shipped leg's search (its fetch, not the instrument's whole-collection read)
    answers its first two hits swapped."""

    def search(self, vector, limit, *, budget_seconds):
        answer = super().search(vector, limit, budget_seconds=budget_seconds)
        if limit != len(self.points) and len(answer.hits) > 1:
            hits = list(answer.hits)
            hits[0], hits[1] = hits[1], hits[0]
            return answer._replace(hits=hits)
        return answer


class _LegDropsOne(FakeRejectionIndex):
    """The shipped leg's search loses its best hit."""

    def search(self, vector, limit, *, budget_seconds):
        answer = super().search(vector, limit, budget_seconds=budget_seconds)
        if limit != len(self.points):
            return answer._replace(hits=list(answer.hits)[1:])
        return answer


class _LegOverruns(FakeRejectionIndex):
    """The shipped leg's search overruns its budget; the instrument's does not."""

    def search(self, vector, limit, *, budget_seconds):
        if limit != len(self.points):
            return super().search(vector, limit, budget_seconds=0.0)
        return super().search(vector, limit, budget_seconds=budget_seconds)


class _InstrumentTruncated(FakeRejectionIndex):
    """The instrument's whole-collection search comes back one point short."""

    def search(self, vector, limit, *, budget_seconds):
        answer = super().search(vector, limit, budget_seconds=budget_seconds)
        if limit == len(self.points):
            return answer._replace(hits=list(answer.hits)[:-1])
        return answer


@pytest.mark.parametrize("broken, raised, needle", [
    (_LegSwapsTwo, CH.SectionDisagreement, "!= instrument"),
    (_LegDropsOne, CH.SectionDisagreement, "!= instrument"),
    (_LegOverruns, CH.SectionDisagreement, "'not_searched'"),
    (_InstrumentTruncated, AssertionError, "of 46 points"),
])
def test_a_section_the_instrument_cannot_vouch_for_is_a_fault(
        index, fixtures, tmp_path, monkeypatch, broken, raised, needle):
    """Each broken index subclasses the shared fake and overrides one method (the
    tree-wide protocol sweep forbids any other broken double). An order swap or a lost
    match in the shipped section disagrees; an overrun section compares nothing; an
    instrument read missing a point is never a measurement."""
    probe = _item_probe(index, tmp_path, monkeypatch, broken(present=True))
    needle = needle.replace("46", str(len(probe.items.points)))
    with pytest.raises(raised) as err:
        CH.read_query_form(probe, _fixture(fixtures, "rejection-own-postgres"))
    assert needle in str(err.value) and "rejection-own-postgres" in str(err.value)


# --------------------------------------------------------------------------- #
# Phase 6a — every query-form hit carries its lexical overlap with the query
# --------------------------------------------------------------------------- #

def test_label_hits_measures_overlap_over_the_slug_and_the_nodes_mechanisms(index):
    """A handle-words query scores 1.0 on its own decision, and a query word only a
    mechanism carries counts too: the harness reads the node's ``mechanisms`` (X10)."""
    from _point_hits import CannedVector

    node = next(n for n in index.nodes if n["kind"] == "decision" and n.get("mechanisms")
                and any(m.replace("-", " ") not in n["slug"].replace("-", " ") for m in n["mechanisms"]))
    mechanism = next(m for m in node["mechanisms"] if m.replace("-", " ") not in node["slug"].replace("-", " "))
    proxy = CH.RecordingVectorStore(CannedVector([{"slug": node["slug"], "score": 0.9}], store=index.store))
    raw = proxy.query([0.1], limit=1, with_payload=False)

    handle = CH.label_hits(index, raw, [], query=node["slug"].replace("-", " "))
    assert handle[0]["overlap"] == 1.0
    words = CH.label_hits(index, raw, [], query=mechanism.replace("-", " "))
    assert words[0]["overlap"] == 1.0
    assert "overlap" not in CH.label_hits(index, raw, [])[0]  # the draft form measures none
