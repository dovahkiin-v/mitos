"""The rejection index's inline write path: a record or an amend only adds.

Recall-confidence phase 4b. ``MitosSyncManager._best_effort_embed`` runs a second leg
after the decision leg whenever the commit queued a rejection backlog row
(``CommitDelta.rejection_enqueue_seq``): a new decision embeds every item not yet
present and clears its own row against its own seq; an amend asks presence by the
ids its field determines, embeds the rest and leaves its row standing. It may create
the absent collection only when its decision is the graph's only decision, and an
unlicensed leg learns that the collection exists before it embeds anything.

The rows drive the outermost entry points where the route exists — MCP
``record_decision`` / ``amend_commentary`` in-process and CLI ``record`` /
``amend-commentary`` through ``main()`` — with the three constructors bound at
``mitos.sync``: a recording provider (never a MagicMock), a per-test decision
vector store, and ``FakeRejectionIndex`` (never a MagicMock). A real ``GraphStore``
in ``tmp_path`` throughout.

- shapes (I1–I7), the licence (L1–L4), the budget and the cache (B-one, B-bytes,
  B-cache);
- faults (F1–F8): the commit stands, the receipt returns, the row stands;
- the race (R1–R2) and the fences (X-imp, X-check, X-verb, X-ops).
"""

import copy
import hashlib
import json
import os
import sqlite3
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock

import pytest

from mitos import mcp_server
from mitos.embeddings import (
    DOCUMENT_PREFIX,
    EMBED_BATCH_MAX_BYTES,
    EMBED_BATCH_MAX_TEXTS,
    GeminiEmbeddingProvider,
    fit_batch,
)
from mitos.errors import (
    CollectionMissingError,
    DatabaseError,
    EmbeddingError,
    VectorStoreError,
)
from mitos.models import get_embedding_model_id
from mitos.parser import ParsedEntry
from mitos.rejections import rejection_points
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager, _embedding_input_text

from _rejection_index_double import FakeRejectionIndex
from _stdio_fake_providers import FAULT_FIELDS, FakeVectorStore, _SdkLikeError, fake_vector
from conftest import resolve_like_main
from test_cli_selector import _run
from test_corpus_provenance import _init_workspace

DEAD_QDRANT_URL = "http://127.0.0.1:9"
# The provider's resolved id with no override: what the manager hands its constructor.
MODEL = get_embedding_model_id({})
FIELD = "1. Keep the old — it drifted.\n2. Do nothing — it rots.\n3. Buy a tool — it locks us in."
WARNING_PREFIX = "[Warning] Rejected items for"


# --------------------------------------------------------------------------- #
# Doubles and the rig
# --------------------------------------------------------------------------- #


class RecordingProvider:
    """A provider double that records every call; not a MagicMock.

    Knobs:
        fault: ``auth`` / ``quota`` / ``transient`` / ``unknown`` makes
            ``embed_documents`` raise an ``EmbeddingError`` chained from an
            SDK-shaped error carrying the typed fields ``provider_cause`` reads.
            The decision leg's ``get_embedding`` stays healthy.
        short: ``embed_documents`` answers one vector fewer than it was sent.
    """

    def __init__(self) -> None:
        self.model_id = MODEL
        self.fault: Optional[str] = None
        self.short = False
        self.document_calls: List[List[str]] = []
        self.single_calls: List[str] = []

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        self.single_calls.append(text)
        return fake_vector(text)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        self.document_calls.append(list(texts))
        if self.fault is not None:
            body = f"planted {self.fault}"
            raise EmbeddingError(f"Gemini embedding API call failed: {body}") from _SdkLikeError(
                body, **FAULT_FIELDS[self.fault])
        vectors = [fake_vector(t) for t in texts]
        return vectors[:-1] if self.short else vectors


class Rig:
    """One workspace with its three bound doubles."""

    def __init__(self, root: str, provider: Any, index: FakeRejectionIndex,
                 vector_cls: type) -> None:
        self.root = root
        self.provider = provider
        self.index = index
        self.vector_cls = vector_cls
        self.index_names: List[str] = []

    @property
    def config(self):
        return resolve_like_main(self.root)

    def manager(self) -> MitosSyncManager:
        return MitosSyncManager(self.config)

    def store(self) -> GraphStore:
        return GraphStore(self.config.db_path)

    def row(self, node_id: str):
        return self.store().get_rejection_backlog_row(node_id)

    def ops(self) -> List[str]:
        return [name for name, _ in self.index.calls]


def _vector_store_class(*, refuse: bool = False) -> type:
    """A per-test decision vector store (its own collections dict)."""

    def upsert(self, point_id, vector, payload, *, may_create):
        if refuse:
            raise VectorStoreError("fake: decision upsert refused")
        return FakeVectorStore.upsert(self, point_id, vector, payload, may_create=may_create)

    return type("DecisionStore", (FakeVectorStore,), {"_collections": {}, "upsert": upsert})


def _bind(monkeypatch, rig: Rig) -> None:
    def provider_factory(cache_path, *, api_key=None, model_id=None):
        if isinstance(rig.provider, RecordingProvider) and model_id is not None:
            rig.provider.model_id = model_id
        return rig.provider

    def index_factory(qdrant_url, collection_name):
        rig.index_names.append(collection_name)
        return rig.index

    monkeypatch.setattr("mitos.sync.GeminiEmbeddingProvider", provider_factory)
    monkeypatch.setattr("mitos.sync.QdrantVectorStore", rig.vector_cls)
    monkeypatch.setattr("mitos.sync.QdrantRejectionIndex", index_factory)


@pytest.fixture
def rig(tmp_path, monkeypatch) -> Rig:
    monkeypatch.setenv("QDRANT_URL", DEAD_QDRANT_URL)
    monkeypatch.chdir(tmp_path)
    root = _init_workspace(tmp_path / "ws")
    made = Rig(root, RecordingProvider(), FakeRejectionIndex(), _vector_store_class())
    _bind(monkeypatch, made)
    yield made
    # X-ops, over every row: the inline leg only probes and adds.
    assert set(made.ops()) <= {"retrieve_present", "upsert_points"}, made.index.calls


def _record(m: MitosSyncManager, slug: str, field: str = FIELD, **relations: Any) -> Dict:
    result = m.record_decision_entry(
        f"The {slug} axiom stands on its own words.", field, ["alpha"],
        mechanisms=[f"{slug}-mechanism"], context=f"The {slug} context.", slug=slug,
        acknowledge_neighbors=True, **relations,
    )
    assert result["status"] == "created", result
    return result


def _mcp_record(rig: Rig, slug: str, field: str = FIELD) -> Dict:
    result = json.loads(mcp_server.record_decision(
        f"The {slug} axiom stands on its own words.", field, ["alpha"], slug,
        mechanisms=[f"{slug}-mechanism"], acknowledge_neighbors=True, project=rig.root,
    ))
    assert result["status"] == "created", result
    return result


def _cli_record(rig: Rig, slug: str, field: str = FIELD) -> Dict:
    from mitos.store import GraphStore as _Store

    code = _run(["-p", rig.root, "record", f"The {slug} axiom stands on its own words.",
                 "--rejected", field, "--scope", "alpha", "--mechanisms",
                 f"{slug}-mechanism", "--slug", slug, "--acknowledge-neighbors"])
    assert code in (0, None), code
    node = _Store(rig.config.db_path).get_node_by_slug(slug)
    assert node is not None
    return {"id": node["id"], "status": "created"}


def _mcp_amend(rig: Rig, slug: str, **changes: Any) -> Dict:
    result = json.loads(mcp_server.amend_commentary(slug, project=rig.root, **changes))
    assert result["status"] == "amended", result
    return result


def _cli_amend(rig: Rig, slug: str, field: str) -> Dict:
    code = _run(["-p", rig.root, "amend-commentary", slug, "--rejected", field])
    assert code in (0, None), code
    return {"status": "amended"}


RECORDERS = {"mcp": _mcp_record, "cli": _cli_record}


def _ids(owner: str, field: str = FIELD) -> List[str]:
    return list(rejection_points(owner, field))


def _texts(owner: str, field: str) -> Dict[str, str]:
    return {pid: ref.text for pid, ref in rejection_points(owner, field).items()}


def _warnings(err: str) -> List[str]:
    return [line for line in err.splitlines() if line.startswith(WARNING_PREFIX)]


def _record_absent_first(rig: Rig, slug: str = "first") -> Dict:
    """Records one decision while the index is unreachable: the collection stays absent.

    The shape of every upgraded workspace: decisions in the graph, no rejection
    collection, their rows standing.
    """
    rig.index.unreachable = True
    result = _record(rig.manager(), slug)
    rig.index.unreachable = False
    rig.index.calls.clear()
    rig.provider.document_calls.clear()
    assert rig.index.present is False
    return result


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("surface", ["mcp", "cli"])
def test_i1_a_first_decision_creates_writes_every_item_and_clears(rig, surface) -> None:
    result = RECORDERS[surface](rig, "first")
    owner = result["id"]

    assert rig.index_names and set(rig.index_names) == {rig.config.rejection_collection}
    assert rig.index.created == 1
    assert set(rig.index.points) == set(_ids(owner))
    for point in rig.index.points.values():
        assert point["payload"] == {"owner_id": owner, "model": MODEL}
    assert rig.row(owner) is None
    assert rig.provider.document_calls == [list(_texts(owner, FIELD).values())]
    # A licensed new owner asks nothing: the upsert is the only call.
    assert rig.index.calls == [
        ("upsert_points", {"ids": _ids(owner), "model": MODEL, "may_create": True}),
    ]


def test_i1_the_point_model_is_the_providers_resolved_override(rig, monkeypatch) -> None:
    monkeypatch.setenv("MITOS_MODEL_OVERRIDE_EMBEDDING", "an-override-embedding-model")
    owner = _mcp_record(rig, "first")["id"]
    models = {p["payload"]["model"] for p in rig.index.points.values()}
    assert models == {"an-override-embedding-model"}
    assert rig.index.calls[-1][1]["model"] == "an-override-embedding-model"
    assert rig.row(owner) is None


@pytest.mark.parametrize("surface", ["mcp", "cli"])
def test_i2_a_second_decision_probes_writes_and_clears(rig, surface) -> None:
    _record(rig.manager(), "first")
    rig.index.calls.clear()
    rig.provider.document_calls.clear()

    owner = RECORDERS[surface](rig, "second")["id"]

    assert rig.index.calls == [
        ("retrieve_present", {"ids": _ids(owner)}),
        ("upsert_points", {"ids": _ids(owner), "model": MODEL, "may_create": False}),
    ]
    assert set(_ids(owner)) <= set(rig.index.points)
    assert rig.row(owner) is None
    assert len(rig.provider.document_calls) == 1


AMENDED = "1. Keep the old — it drifted.\n2. Do nothing — it rots.\n3. Fork it — we would own it."


@pytest.mark.parametrize("surface", ["mcp", "cli"])
def test_i3_an_amend_embeds_only_its_new_item_and_leaves_its_row(rig, surface) -> None:
    _record(rig.manager(), "first")
    owner = _record(rig.manager(), "second")["id"]
    before = _texts(owner, FIELD)
    rig.index.calls.clear()
    rig.provider.document_calls.clear()

    if surface == "mcp":
        _mcp_amend(rig, "second", rejected_paths=AMENDED)
    else:
        _cli_amend(rig, "second", AMENDED)

    after = _texts(owner, AMENDED)
    added = [pid for pid in after if pid not in before]
    dropped = [pid for pid in before if pid not in after]
    assert len(added) == 1 and len(dropped) == 1
    # The probe names exactly the field's ids; the provider sees only the new text.
    assert rig.index.calls[0] == ("retrieve_present", {"ids": list(after)})
    assert rig.provider.document_calls == [[after[added[0]]]]
    assert rig.index.calls[1] == (
        "upsert_points", {"ids": added, "model": MODEL, "may_create": False})
    assert len(rig.index.calls) == 2
    # The dropped item stays in the index (only a drain removes; 5a hides it at read).
    assert dropped[0] in rig.index.points
    row = rig.row(owner)
    assert row is not None and row.retry_count == 0


def test_i3_the_standing_row_is_the_amends_own_commit(rig) -> None:
    _record(rig.manager(), "first")
    owner = _record(rig.manager(), "second")["id"]
    seen: List[Optional[int]] = []
    real = MitosSyncManager._write_rejection_items_inline

    def spy(self, delta, entry):
        seen.append(delta.rejection_enqueue_seq)
        return real(self, delta, entry)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(MitosSyncManager, "_write_rejection_items_inline", spy)
        _mcp_amend(rig, "second", rejected_paths=AMENDED)
    assert seen and seen[-1] == rig.row(owner).enqueue_seq


def test_i4_a_context_amend_touches_neither_index_nor_provider(rig) -> None:
    _record(rig.manager(), "first")
    owner = _record(rig.manager(), "second")["id"]
    _mcp_amend(rig, "second", rejected_paths=AMENDED)  # the row now stands
    standing = rig.row(owner)
    calls, documents = len(rig.index.calls), len(rig.provider.document_calls)

    _mcp_amend(rig, "second", context="A rewritten context, and nothing else.")

    assert len(rig.index.calls) == calls
    assert len(rig.provider.document_calls) == documents
    assert rig.row(owner) == standing


def test_i5_an_amend_to_an_empty_field_calls_nothing_and_the_row_stands(rig) -> None:
    # The amend route and the parser both refuse an empty Rejected, so the commit is
    # hand-built as 4a's E5 did, and handed to the leg as the route would hand it.
    _record(rig.manager(), "first")
    m = rig.manager()
    stored: List[ParsedEntry] = []
    real = MitosSyncManager._write_rejection_items_inline
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(MitosSyncManager, "_write_rejection_items_inline",
                   lambda self, delta, entry: stored.append(entry) or real(self, delta, entry))
        owner = _record(m, "second")["id"]
    entry = copy.copy(stored[-1])
    entry.rejected_paths = ""
    delta = m.store.commit_parsed_entry(entry)
    assert delta.node_id == owner and delta.rejection_enqueue_seq is not None
    rig.index.calls.clear()
    rig.provider.document_calls.clear()

    m._best_effort_embed(delta, entry)

    assert rig.index.calls == [] and rig.provider.document_calls == []
    assert rig.row(owner).enqueue_seq == delta.rejection_enqueue_seq


def test_i6_an_open_question_commit_makes_no_rejection_call(rig) -> None:
    m = rig.manager()
    entry = ParsedEntry("open_question", "q-one", 1, 5)
    entry.topic = "A topic."
    entry.questions_raised = ["A question?"]
    delta = m.store.commit_parsed_entry(entry)
    assert delta.created is True and delta.rejection_enqueue_seq is None

    m._best_effort_embed(delta, entry)

    assert rig.index.calls == [] and rig.provider.document_calls == []


def test_i7_an_amend_of_the_only_decision_may_create_and_keeps_its_row(rig) -> None:
    owner = _record_absent_first(rig)["id"]

    _mcp_amend(rig, "first", rejected_paths=AMENDED)

    assert rig.index.created == 1
    assert set(rig.index.points) == set(_ids(owner, AMENDED))
    probe, upsert = rig.index.calls
    assert probe == ("retrieve_present", {"ids": _ids(owner, AMENDED)})
    assert upsert[1]["may_create"] is True
    assert rig.row(owner) is not None


# --------------------------------------------------------------------------- #
# The licence
# --------------------------------------------------------------------------- #


def test_l1_an_absent_collection_beside_another_decision_spends_nothing(rig, capsys) -> None:
    _record_absent_first(rig)
    capsys.readouterr()

    owner = _mcp_record(rig, "second")["id"]

    assert rig.provider.document_calls == []
    assert rig.index.calls == [("retrieve_present", {"ids": _ids(owner)})]
    assert rig.index.created == 0
    assert rig.row(owner) is not None
    assert _warnings(capsys.readouterr().err) == []


def test_l2_a_retired_decision_withholds_the_licence(rig) -> None:
    old = _record_absent_first(rig, "old")["id"]
    owner = _record(rig.manager(), "successor", supersedes="old")["id"]
    assert rig.store().get_node_state(old) == "superseded"

    assert rig.index.created == 0
    assert all(c[1].get("may_create") is not True for c in rig.index.calls)
    assert rig.index.calls[0][0] == "retrieve_present"
    assert rig.row(owner) is not None


def test_l2_a_parked_open_question_does_not_withhold_it(rig) -> None:
    m = rig.manager()
    question = ParsedEntry("open_question", "q-parked", 1, 5)
    question.topic = "A parked topic."
    question.questions_raised = ["Is it parked?"]
    m.store.commit_parsed_entry(question)

    owner = _record(m, "first")["id"]

    assert rig.index.created == 1
    assert rig.index.calls[0] == (
        "upsert_points", {"ids": _ids(owner), "model": MODEL, "may_create": True})
    assert rig.row(owner) is None


def _append_entry(rig: Rig, slug: str, field: str = FIELD) -> None:
    block = (
        f"## 2026-10-17 — {slug} — {slug.title()}\n"
        f"**Decided:** The {slug} axiom stands on its own words.\n"
        f"**Rejected:**\n{field}\n"
        f"**Mechanisms:** {slug}-mechanism\n"
        f"**Scope:** alpha\n"
    )
    with open(rig.config.decisions_file, "a", encoding="utf-8") as handle:
        handle.write("\n" + block)


def _sync(rig: Rig, monkeypatch) -> MitosSyncManager:
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
    m = rig.manager()
    m.perform_sync(auto_accept=True)
    return m


def test_l3_a_multi_entry_sync_over_an_absent_collection_probes_once(rig, monkeypatch) -> None:
    _record_absent_first(rig)
    for slug in ("synced-one", "synced-two", "synced-three"):
        _append_entry(rig, slug)

    m = _sync(rig, monkeypatch)

    owners = [m.store.get_node_by_slug(s)["id"]
              for s in ("synced-one", "synced-two", "synced-three")]
    assert rig.ops() == ["retrieve_present"]
    assert rig.provider.document_calls == []
    assert all(rig.row(o) is not None for o in owners)
    assert m._rejection_collection_absent is True


def _direct_decision_count(store: GraphStore, node_id: str) -> int:
    conn = sqlite3.connect(store.db_path)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE kind = 'decision' AND id != ?", (node_id,)
        ).fetchone()[0]
    finally:
        conn.close()


@pytest.mark.parametrize("shape", ["empty", "self only", "live other", "retired other",
                                   "open questions only"])
def test_l4_the_licence_read_agrees_with_a_direct_count(tmp_path, shape) -> None:
    os.makedirs(tmp_path / ".mitos")
    store = GraphStore(str(tmp_path / ".mitos" / "graph.sqlite"))

    def decision(slug: str, axiom: str, **fields: Any) -> str:
        entry = ParsedEntry("decision", slug, 1, 5)
        entry.axiom = axiom
        entry.rejected_paths = "Something else."
        for name, value in fields.items():
            setattr(entry, name, value)
        return store.commit_parsed_entry(entry).node_id

    me = "f" * 64
    if shape == "self only":
        me = decision("me", "My axiom.")
    elif shape == "live other":
        me = decision("me", "My axiom.")
        decision("other", "Another axiom.")
    elif shape == "retired other":
        decision("old", "An old axiom.")
        me = decision("me", "My axiom.", supersedes=["old"])
    elif shape == "open questions only":
        question = ParsedEntry("open_question", "q", 1, 5)
        question.topic = "A topic."
        question.questions_raised = ["Why?"]
        store.commit_parsed_entry(question)

    expected = _direct_decision_count(store, me) > 0
    assert store.rejection_index_expects_owner_other_than(me) is expected
    assert expected is (shape in ("live other", "retired other"))


# --------------------------------------------------------------------------- #
# The budget and the cache
# --------------------------------------------------------------------------- #


def test_b_one_a_field_over_the_count_bound_spends_one_call(rig) -> None:
    field = "\n".join(f"{i}. Option {i} — reason {i}." for i in range(1, EMBED_BATCH_MAX_TEXTS + 6))
    owner = _record(rig.manager(), "first", field)["id"]
    ids = _ids(owner, field)
    assert len(ids) > EMBED_BATCH_MAX_TEXTS

    assert len(rig.provider.document_calls) == 1
    assert len(rig.provider.document_calls[0]) == EMBED_BATCH_MAX_TEXTS
    assert set(rig.index.points) == set(ids[:EMBED_BATCH_MAX_TEXTS])
    assert rig.row(owner) is not None  # the overflow waits for the drain


def test_b_one_a_field_over_the_byte_bound_spends_one_call(rig) -> None:
    field = "\n".join(f"{i}. Option {i} — " + ("x" * 7000) + f" {i}." for i in range(1, 8))
    owner = _record(rig.manager(), "first", field)["id"]
    texts = list(_texts(owner, field).values())
    n = fit_batch(texts)
    assert 0 < n < len(texts)

    assert rig.provider.document_calls == [texts[:n]]
    assert sum(len((DOCUMENT_PREFIX + t).encode("utf-8")) for t in texts[:n]) <= EMBED_BATCH_MAX_BYTES
    assert set(rig.index.points) == set(_ids(owner, field)[:n])
    assert rig.row(owner) is not None


def test_b_bytes_a_non_latin_field_splits_on_bytes_not_characters(rig) -> None:
    field = "\n".join(f"{i}. Вариант {i} — " + ("ж" * 3000) + f" {i}." for i in range(1, 9))
    owner = _record(rig.manager(), "first", field)["id"]
    texts = list(_texts(owner, field).values())
    assert sum(len(DOCUMENT_PREFIX + t) for t in texts) <= EMBED_BATCH_MAX_BYTES  # chars fit
    n = fit_batch(texts)
    assert n < len(texts)  # bytes do not

    assert len(rig.provider.document_calls) == 1
    assert rig.provider.document_calls[0] == texts[:n]


def _cache_rows(cache_path: str) -> List[str]:
    conn = sqlite3.connect(cache_path)
    try:
        return sorted(r[0] for r in conn.execute("SELECT content_hash FROM cache"))
    finally:
        conn.close()


def test_b_cache_the_leg_leaves_the_shared_cache_unchanged(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", DEAD_QDRANT_URL)
    monkeypatch.chdir(tmp_path)
    root = _init_workspace(tmp_path / "ws")
    sent: List[Any] = []

    def embed_content(*, model: str, contents):
        sent.append(contents)
        count = len(contents) if isinstance(contents, list) else 1
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5, 0.5]) for _ in range(count)])

    providers: List[GeminiEmbeddingProvider] = []

    def real_provider(cache_path, *, api_key=None, model_id=None):
        provider = GeminiEmbeddingProvider(cache_path, api_key="dummy", model_id=MODEL)
        provider.client = SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
        providers.append(provider)
        return provider

    index = FakeRejectionIndex()
    monkeypatch.setattr("mitos.sync.GeminiEmbeddingProvider", real_provider)
    monkeypatch.setattr("mitos.sync.QdrantVectorStore", _vector_store_class())
    monkeypatch.setattr("mitos.sync.QdrantRejectionIndex", lambda url, name: index)
    m = MitosSyncManager(resolve_like_main(root))
    cache_path = os.path.join(m.config.mitos_dir, "embedding_cache.sqlite")

    # Pre-seed the axiom's embedding, as a paused record leaves it.
    axiom = "The first axiom stands on its own words."
    axiom_text = _embedding_input_text(kind="decision", axiom=axiom, topic=None,
                                       questions_raised=None)
    m.embed_provider.get_embedding(axiom_text, is_query=False)
    seeded = _cache_rows(cache_path)
    sent.clear()
    m.embed_provider.reset_stats()

    owner = _record(m, "first")["id"]

    texts = list(_texts(owner, FIELD).values())
    # The rejection leg's one call carried the items (one Content each); every other
    # call was a single cached-path string. No item hash entered the cache.
    documents = [c for c in sent if isinstance(c, list)]
    assert [[content.parts[0].text for content in c] for c in documents] == [
        [DOCUMENT_PREFIX + t for t in texts]]
    assert all(isinstance(c, str) for c in sent if not isinstance(c, list))
    item_hashes = {hashlib.sha256((DOCUMENT_PREFIX + t).encode()).hexdigest() for t in texts}
    assert not item_hashes & set(_cache_rows(cache_path))
    # The axiom was served by the cache: its row survives and was a hit.
    assert set(seeded) <= set(_cache_rows(cache_path))
    assert _cache_rows(cache_path) == seeded  # the row count, and every row, unchanged
    assert m.embed_provider.hits >= 1
    assert set(index.points) == set(_ids(owner))


# --------------------------------------------------------------------------- #
# Faults: the commit stands, the receipt returns, the row stands, retry_count 0
# --------------------------------------------------------------------------- #


def _assert_row_stands(rig: Rig, owner: str, *, decision_leg_indexed: bool = True) -> None:
    """The commit stands, the backlog row stands untouched, the decision leg as today.

    With a healthy decision index the decision leg indexed the axiom and dropped its
    outbox row, exactly as before 4b: no rejection fault reaches it.
    """
    row = rig.row(owner)
    assert row is not None and row.retry_count == 0
    store = rig.store()
    assert store.get_node(owner) is not None
    queued = {r["node_id"] for r in store.get_pending_embeddings()}
    assert (owner not in queued) is decision_leg_indexed


def _two_in_one_command(rig: Rig, fault: str) -> Tuple[MitosSyncManager, str, str]:
    """A healthy first record (the collection exists), then one faulted record."""
    _record(rig.manager(), "first")
    rig.index.calls.clear()
    rig.provider.document_calls.clear()
    rig.provider.fault = fault
    m = rig.manager()
    one = _record(m, "second")["id"]
    return m, one, "third"


def test_f1_a_quota_outage_latches_the_leg_for_the_command(rig, capsys) -> None:
    m, one, _ = _two_in_one_command(rig, "quota")
    err = capsys.readouterr().err
    assert len(_warnings(err)) == 1 and "rate-limited" in _warnings(err)[0]
    _assert_row_stands(rig, one)
    calls = len(rig.index.calls)

    two = _record(m, "third")["id"]

    assert len(rig.index.calls) == calls  # latched: no probe, no write
    assert len(rig.provider.document_calls) == 1
    assert m._rejection_leg_stopped is True
    _assert_row_stands(rig, two)
    assert _warnings(capsys.readouterr().err) == []


def test_f1_a_quota_outage_in_one_sync_spends_one_call(rig, monkeypatch, capsys) -> None:
    _record(rig.manager(), "first")
    rig.index.calls.clear()
    rig.provider.document_calls.clear()
    rig.provider.fault = "quota"
    for slug in ("synced-one", "synced-two"):
        _append_entry(rig, slug)
    capsys.readouterr()

    m = _sync(rig, monkeypatch)

    assert len(rig.provider.document_calls) == 1
    assert len(_warnings(capsys.readouterr().err)) == 1
    for slug in ("synced-one", "synced-two"):
        _assert_row_stands(rig, m.store.get_node_by_slug(slug)["id"])


@pytest.mark.parametrize("cause", ["auth", "transient"])
def test_f1_every_outage_cause_latches(rig, cause) -> None:
    m, one, third = _two_in_one_command(rig, cause)
    _record(m, third)
    assert m._rejection_leg_stopped is True
    assert len(rig.provider.document_calls) == 1


def test_f2_an_unknown_cause_costs_one_owner(rig, capsys) -> None:
    m, one, third = _two_in_one_command(rig, "unknown")
    _record(m, third)
    assert len(rig.provider.document_calls) == 2  # the next entry still calls
    assert m._rejection_leg_stopped is False
    assert len(_warnings(capsys.readouterr().err)) == 2
    _assert_row_stands(rig, one)


def test_f3_an_unreachable_index_latches(rig, capsys) -> None:
    _record(rig.manager(), "first")
    rig.index.unreachable = True
    rig.index.calls.clear()
    m = rig.manager()
    capsys.readouterr()
    one = _record(m, "second")["id"]
    two = _record(m, "third")["id"]
    assert rig.ops() == ["retrieve_present"]
    assert m._rejection_leg_stopped is True
    assert len(_warnings(capsys.readouterr().err)) == 1
    _assert_row_stands(rig, one)
    _assert_row_stands(rig, two)


def test_f4_a_refused_upsert_is_not_latched(rig, capsys) -> None:
    _record(rig.manager(), "first")
    rig.index.refuse_upsert = True
    rig.index.calls.clear()
    m = rig.manager()
    capsys.readouterr()
    one = _record(m, "second")["id"]
    _record(m, "third")
    assert rig.ops().count("upsert_points") == 2
    assert m._rejection_leg_stopped is False
    assert len(_warnings(capsys.readouterr().err)) == 2
    _assert_row_stands(rig, one)


def test_f5_a_short_answer_writes_nothing(rig, capsys) -> None:
    _record(rig.manager(), "first")
    points = dict(rig.index.points)
    rig.provider.short = True
    capsys.readouterr()
    owner = _record(rig.manager(), "second")["id"]
    assert rig.index.points == points
    assert "upsert_points" not in rig.ops()[-1:]
    warned = _warnings(capsys.readouterr().err)
    assert len(warned) == 1 and "embedding provider error" in warned[0]
    _assert_row_stands(rig, owner)


@pytest.mark.parametrize("model_id", ["a-model", None])
def test_f5_a_magicmock_provider_is_an_unknown_failure(rig, monkeypatch, capsys, model_id) -> None:
    _record(rig.manager(), "first")
    points = dict(rig.index.points)
    mock = MagicMock()
    mock.get_embedding.side_effect = lambda text, is_query=False: fake_vector(text)
    if model_id is not None:
        mock.model_id = model_id
    monkeypatch.setattr("mitos.sync.GeminiEmbeddingProvider", lambda *a, **k: mock)
    capsys.readouterr()

    owner = _record(rig.manager(), "second")["id"]

    assert rig.index.points == points
    warned = _warnings(capsys.readouterr().err)
    assert len(warned) == 1 and "embedding provider error" in warned[0]
    _assert_row_stands(rig, owner)


def test_f6_a_licence_read_fault_costs_one_line(rig, monkeypatch, capsys) -> None:
    def broken(self, node_id):
        raise DatabaseError("the licence read failed")

    monkeypatch.setattr(GraphStore, "rejection_index_expects_owner_other_than", broken)
    capsys.readouterr()
    owner = _record(rig.manager(), "first")["id"]
    assert rig.index.calls == [] and rig.provider.document_calls == []
    assert len(_warnings(capsys.readouterr().err)) == 1
    _assert_row_stands(rig, owner)


def test_f6_a_clear_fault_after_the_write_says_the_items_landed(rig, monkeypatch, capsys) -> None:
    def broken(self, node_id, *, enqueue_seq):
        raise DatabaseError("the clear failed")

    monkeypatch.setattr(GraphStore, "clear_rejection_backlog_row", broken)
    capsys.readouterr()
    m = rig.manager()
    owner = _record(m, "first")["id"]

    assert set(rig.index.points) == set(_ids(owner))
    warned = _warnings(capsys.readouterr().err)
    assert len(warned) == 1
    assert "were indexed" in warned[0] and "not indexed" not in warned[0]
    for word in ("mitos ", "sync", "reconcile", "rebuild"):
        assert word not in warned[0], word
    assert m._rejection_leg_stopped is False and m._rejection_collection_absent is False
    _assert_row_stands(rig, owner)


def test_f7_a_failing_decision_leg_does_not_stop_the_rejection_leg(rig, monkeypatch) -> None:
    # The decision collection is absent and this write may not create it: the
    # decision leg latches; the rejection leg (licensed, first decision) still writes.
    _record_absent_first(rig)  # leaves the decision collection created; reset it
    rig.vector_cls._collections.clear()
    rig.index.points.clear()
    rig.index.present = True
    m = rig.manager()

    result = _record(m, "second")
    owner = result["id"]

    assert m._collection_absent is True
    assert result["embedding"] != "indexed"
    assert set(_ids(owner)) <= set(rig.index.points)
    assert rig.row(owner) is None
    # The decision latch does not gate the rejection leg on the next entry either.
    third = _record(m, "third")["id"]
    assert set(_ids(third)) <= set(rig.index.points)


def test_f7_a_refused_decision_upsert_and_the_reverse(rig, monkeypatch) -> None:
    _record(rig.manager(), "first")  # both collections exist

    # The rejection leg fails and the decision leg indexes.
    rig.index.refuse_upsert = True
    second = _record(rig.manager(), "second")
    assert second["embedding"] == "indexed"
    _assert_row_stands(rig, second["id"])

    # The decision leg's upsert is refused and the rejection leg writes.
    rig.index.refuse_upsert = False
    refusing = _vector_store_class(refuse=True)
    monkeypatch.setattr("mitos.sync.QdrantVectorStore", refusing)
    third = _record(rig.manager(), "third")
    assert third["embedding"] != "indexed"
    assert set(_ids(third["id"])) <= set(rig.index.points)
    assert rig.row(third["id"]) is None


def test_f8_a_licensed_create_whose_retry_still_misses_is_one_line_unlatched(rig, capsys) -> None:
    rig.index.create_then_missing = True
    m = rig.manager()
    owner = _record(m, "first")["id"]
    warned = _warnings(capsys.readouterr().err)
    assert len(warned) == 1
    assert m._rejection_collection_absent is False and m._rejection_leg_stopped is False
    _assert_row_stands(rig, owner)

    # Not latched: the next entry in the same command still probes.
    rig.index.calls.clear()
    _record(m, "second")
    assert rig.ops()[:1] == ["retrieve_present"]


class _DeletedAfterProbe(FakeRejectionIndex):
    """The collection answers the probe, then is deleted before the write."""

    def retrieve_present(self, point_ids):
        answer = super().retrieve_present(point_ids)
        self.present = False
        return answer


def test_f8_an_unlicensed_write_meeting_a_deleted_collection_is_silent_and_latched(
    rig, monkeypatch, capsys
) -> None:
    _record(rig.manager(), "first")
    deleting = _DeletedAfterProbe(present=True)
    deleting.points = dict(rig.index.points)
    rig.index = deleting
    _bind(monkeypatch, rig)
    m = rig.manager()
    capsys.readouterr()

    owner = _record(m, "second")["id"]

    assert _warnings(capsys.readouterr().err) == []
    assert m._rejection_collection_absent is True
    _assert_row_stands(rig, owner)
    # The batch was spent and lost (G7): one call, nothing landed.
    assert len(rig.provider.document_calls) == 2


# --------------------------------------------------------------------------- #
# The race
# --------------------------------------------------------------------------- #


def test_r1_a_requeue_between_the_write_and_the_clear_survives(rig, monkeypatch) -> None:
    _record(rig.manager(), "first")
    stash: Dict[str, Any] = {}
    real_leg = MitosSyncManager._write_rejection_items_inline
    real_clear = GraphStore.clear_rejection_backlog_row

    def leg(self, delta, entry):
        stash["entry"] = entry
        return real_leg(self, delta, entry)

    class _RequeueAfterWrite(FakeRejectionIndex):
        """A concurrent amend re-queues the owner once the write lands, before the clear."""

        def upsert_points(self, points, *, model, may_create):
            super().upsert_points(points, model=model, may_create=may_create)
            newer = copy.copy(stash["entry"])
            newer.rejected_paths = "1. A newer option — queued after the write."
            stash["newer"] = rig.store().commit_parsed_entry(newer).rejection_enqueue_seq

    def clear(self, node_id, *, enqueue_seq):
        stash["cleared"] = real_clear(self, node_id, enqueue_seq=enqueue_seq)
        return stash["cleared"]

    requeuing = _RequeueAfterWrite(present=True)
    requeuing.points = dict(rig.index.points)
    rig.index = requeuing
    _bind(monkeypatch, rig)
    monkeypatch.setattr(MitosSyncManager, "_write_rejection_items_inline", leg)
    monkeypatch.setattr(GraphStore, "clear_rejection_backlog_row", clear)

    owner = _record(rig.manager(), "second")["id"]

    assert stash["cleared"] is False
    row = rig.row(owner)
    assert row is not None and row.enqueue_seq == stash["newer"]


def test_r2_the_delta_carries_the_seq_its_commit_minted(tmp_path) -> None:
    os.makedirs(tmp_path / ".mitos")
    store = GraphStore(str(tmp_path / ".mitos" / "graph.sqlite"))
    entry = ParsedEntry("decision", "d", 1, 5)
    entry.axiom = "An axiom."
    entry.rejected_paths = "1. One — a."
    seqs = []
    for field in ("1. One — a.", "1. Two — b.", "1. Three — c."):
        entry.rejected_paths = field
        delta = store.commit_parsed_entry(entry)
        assert delta.rejection_enqueue_seq == store.get_rejection_backlog_row(
            delta.node_id).enqueue_seq
        seqs.append(delta.rejection_enqueue_seq)
    assert seqs == sorted(seqs) and len(set(seqs)) == 3
    # A commit that queues nothing carries None and leaves the row's seq alone.
    entry.context = "A context only."
    delta = store.commit_parsed_entry(entry)
    assert delta.created is False and delta.rejection_enqueue_seq is None
    assert store.get_rejection_backlog_row(delta.node_id).enqueue_seq == seqs[-1]


# --------------------------------------------------------------------------- #
# Fences
# --------------------------------------------------------------------------- #


def test_x_imp_the_importer_names_nothing_of_the_leg() -> None:
    import mitos.importer as importer

    with open(importer.__file__, encoding="utf-8") as handle:
        source = handle.read()
    for name in ("QdrantRejectionIndex", "rejection_index", "embed_documents",
                 "clear_rejection_backlog_row", "_write_rejection_items_inline"):
        assert name not in source, name


def test_x_imp_an_import_queues_its_rows_and_makes_no_rejection_call(tmp_path, monkeypatch) -> None:
    from mitos.importer import MitosProseImporter

    monkeypatch.setenv("QDRANT_URL", DEAD_QDRANT_URL)
    root = _init_workspace(tmp_path / "ws")
    legacy = tmp_path / "legacy.md"
    legacy.write_text(
        "## Use a queue\nWe chose a queue. Rejected: polling, because it wastes calls.\n\n"
        "## Keep SQLite\nSQLite stays. Rejected: Postgres, because it is a server.\n",
        encoding="utf-8",
    )
    index = FakeRejectionIndex()
    monkeypatch.setattr("mitos.sync.QdrantRejectionIndex", lambda url, name: index)
    importer = MitosProseImporter(resolve_like_main(root))
    importer.import_from_file(str(legacy), use_llm_extract=False)

    store = GraphStore(importer.config.db_path)
    decisions = [n["id"] for n in store.get_all_nodes() if n["kind"] == "decision"]
    assert decisions
    for node_id in decisions:
        assert store.get_rejection_backlog_row(node_id) is not None
    assert index.calls == []


def test_x_check_standing_rows_change_neither_the_stale_probe_nor_the_receipt(
    tmp_path, monkeypatch
) -> None:
    from mitos.check import probe_stale_index

    receipts: Dict[str, List[Dict]] = {}
    probes: Dict[str, Any] = {}
    for arm in ("leg", "no-leg"):
        monkeypatch.setenv("QDRANT_URL", DEAD_QDRANT_URL)
        root = _init_workspace(tmp_path / arm)
        made = Rig(root, RecordingProvider(), FakeRejectionIndex(), _vector_store_class())
        _bind(monkeypatch, made)
        if arm == "no-leg":
            monkeypatch.setattr("mitos.sync.QdrantRejectionIndex",
                                lambda url, name: (_ for _ in ()).throw(RuntimeError("off")))
        else:
            made.index.unreachable = True  # the collection stays absent; rows stand
        m = made.manager()
        receipts[arm] = [_record(m, s) for s in ("first", "second")]
        store = made.store()
        if arm == "leg":
            assert all(store.get_rejection_backlog_row(r["id"]) for r in receipts[arm])
        probes[arm] = probe_stale_index(store)

    keys = ("status", "embedding", "audit_debt")
    for a, b in zip(receipts["leg"], receipts["no-leg"]):
        assert {k: a.get(k) for k in keys} == {k: b.get(k) for k in keys}
    assert repr(probes["leg"]) == repr(probes["no-leg"])


def test_x_verb_the_warning_names_no_verb(rig, capsys) -> None:
    rig.provider.fault = "unknown"
    capsys.readouterr()
    _record(rig.manager(), "first")
    warned = _warnings(capsys.readouterr().err)
    assert len(warned) == 1
    for word in ("mitos ", "sync", "reconcile", "rebuild"):
        assert word not in warned[0], word


def test_x_ops_the_leg_reads_the_licence_and_never_the_row() -> None:
    """The leg's own source: it clears through its delta's seq, never a row read."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(
        MitosSyncManager._write_rejection_items_inline)))
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "get_rejection_backlog_row" not in called
    assert "decision_index_expects_node_other_than" not in called
    assert {"rejection_index_expects_owner_other_than", "retrieve_present",
            "upsert_points", "embed_documents", "clear_rejection_backlog_row"} <= called
    assert not called & {"delete_points", "list_owner_point_ids", "list_point_ids",
                         "search", "get_embedding", "get_embeddings_batch",
                         "note_rejection_backlog_failure"}


def test_y22_a_down_constructor_leaves_the_index_none(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", DEAD_QDRANT_URL)
    root = _init_workspace(tmp_path / "ws")
    built: List[str] = []
    monkeypatch.setattr("mitos.sync.QdrantRejectionIndex",
                        lambda url, name: built.append(name) or FakeRejectionIndex())
    down = MagicMock(side_effect=Exception("backend down"))
    for target in ("mitos.sync.GeminiEmbeddingProvider", "mitos.sync.QdrantVectorStore"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(target, down)
            m = MitosSyncManager(resolve_like_main(root))
            assert m.rejection_index is None
    assert built == []


def test_the_offline_autouse_keeps_an_unbound_manager_off_the_network(tmp_path, monkeypatch) -> None:
    """With a dummy key the real constructors succeed; the index alone stays None."""
    monkeypatch.setenv("QDRANT_URL", DEAD_QDRANT_URL)
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
    root = _init_workspace(tmp_path / "ws")
    m = MitosSyncManager(resolve_like_main(root))
    assert m.embed_provider is not None and m.vector_store is not None
    assert m.rejection_index is None
