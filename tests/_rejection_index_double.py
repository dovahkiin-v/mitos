"""The shared rejection-index double, and the one contract scenario both clients pass.

``FakeRejectionIndex`` implements :class:`mitos.protocols.RejectionIndex` over a dict,
offline. Every later phase that needs a rejection index in a test reuses it rather than
hand-rolling one: a ``MagicMock`` index would read every write as confirmed and absorb
any keyword it was handed (PATTERNS, "a MagicMock absorbs any new keyword").

It keeps the real client's order of answers, so a test written against it holds
against Qdrant too:

- an empty input sends nothing, and so meets no fault (``retrieve_present``,
  ``upsert_points``, ``delete_points``);
- the write guards (``ValueError``) come before any request;
- a transport fault comes before the collection's absence;
- an absent collection answers ``CollectionMissingError`` to every request, and only
  a ``may_create`` upsert creates it;
- a refusal comes from a collection that exists.

``exercise_rejection_index_contract`` is the scenario: the offline test runs it over the
double and the live lane runs the same function over ``QdrantRejectionIndex``.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from mitos.errors import (
    CollectionMissingError,
    VectorStoreError,
    VectorStoreUnreachableError,
)
from mitos.models import EMBEDDING_DIM
from mitos.rejections import item_point_id
from mitos.vector_store import (
    RejectionHit,
    RejectionPoint,
    RejectionSearch,
)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class FakeRejectionIndex:
    """An in-memory rejection index honouring ``RejectionIndex``'s contract.

    Knobs (plain attributes, flip them at any time):
        present: Whether the collection exists. Absent until a ``may_create``
            upsert, unless constructed present.
        over_budget: ``search`` answers ``searched=False``.
        unreachable: every request raises ``VectorStoreUnreachableError``.
        refuse_upsert: an upsert reaching an existing collection raises
            ``VectorStoreError``.
        refuse_delete: a delete reaching an existing collection raises
            ``VectorStoreError``.
        create_then_missing: a licensed upsert on an absent collection runs the
            create (``created`` counts it) and still answers
            ``CollectionMissingError``, as the real client does when its one retry
            after a create is answered 404 again; the collection stays absent.
        refuse_list: a listing (``list_owner_point_ids``, ``list_point_ids``,
            ``list_point_owners``) reaching an existing collection raises
            ``VectorStoreError``.
        refuse_search: a search reaching an existing collection raises
            ``VectorStoreError``, as the real client does on a non-200 or a 200 of
            the wrong shape.
        on_call: called as ``on_call(method, details)`` right after a call is
            recorded and before it answers, so a row can raise mid-walk or inject a
            commit between two of a drain's calls.

    Recording:
        calls: every call, in order, as ``(method, details)``; ``details`` names the
            ids (and, for an upsert, the model and the licence).
        points: ``{point_id: {"vector": …, "payload": {"owner_id", "model"}}}``.
        created: how many times a licensed upsert created the collection.
    """

    def __init__(self, *, present: bool = False) -> None:
        self.present = present
        self.over_budget = False
        self.unreachable = False
        self.refuse_upsert = False
        self.refuse_delete = False
        self.create_then_missing = False
        self.refuse_list = False
        self.refuse_search = False
        self.on_call: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self.points: Dict[str, Dict[str, Any]] = {}
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self.created = 0

    def _record(self, method: str, details: Dict[str, Any]) -> None:
        self.calls.append((method, details))
        if self.on_call is not None:
            self.on_call(method, details)

    def _listing(self, operation: str) -> None:
        self._reach(operation)
        if self.refuse_list:
            raise VectorStoreError(f"fake: {operation} refused")

    def _reach(self, operation: str) -> None:
        if self.unreachable:
            raise VectorStoreUnreachableError(f"fake: {operation} could not connect")
        if not self.present:
            raise CollectionMissingError(
                f"fake: the collection does not exist ({operation})",
                collection="fake-rejections",
            )

    def search(
        self, vector: List[float], limit: int, *, budget_seconds: float
    ) -> RejectionSearch:
        self._record("search", {"limit": limit, "budget_seconds": budget_seconds})
        if not budget_seconds > 0:  # a spent budget sends nothing, as the client
            return RejectionSearch(searched=False, hits=[])
        if self.unreachable:
            raise VectorStoreUnreachableError("fake: search could not connect")
        if self.over_budget:
            return RejectionSearch(searched=False, hits=[])
        self._reach("search")
        if self.refuse_search:
            raise VectorStoreError("fake: search refused")
        ranked = sorted(
            self.points.items(),
            key=lambda item: _cosine(vector, item[1]["vector"]),
            reverse=True,
        )[:limit]
        hits = []
        for point_id, point in ranked:
            owner = point["payload"].get("owner_id")
            hits.append(
                RejectionHit(
                    point_id=point_id,
                    owner_id=owner if isinstance(owner, str) and owner else None,
                    score=_cosine(vector, point["vector"]),
                )
            )
        return RejectionSearch(searched=True, hits=hits)

    def retrieve_present(self, point_ids: Sequence[str]) -> Set[str]:
        self._record("retrieve_present", {"ids": list(point_ids)})
        if not point_ids:
            return set()
        self._reach("retrieve_present")
        return {pid for pid in point_ids if pid in self.points}

    def list_owner_point_ids(self, owner_id: str, page_size: int = 256) -> Set[str]:
        self._record("list_owner_point_ids", {"owner_id": owner_id})
        self._listing("list_owner_point_ids")
        return {
            pid for pid, point in self.points.items()
            if point["payload"].get("owner_id") == owner_id
        }

    def list_point_ids(self, page_size: int = 256) -> Set[str]:
        self._record("list_point_ids", {})
        self._listing("list_point_ids")
        return set(self.points)

    def list_point_owners(self, page_size: int = 256) -> Dict[str, Optional[str]]:
        self._record("list_point_owners", {})
        self._listing("list_point_owners")
        owners: Dict[str, Optional[str]] = {}
        for pid, point in self.points.items():
            owner = point["payload"].get("owner_id")
            owners[pid] = owner if isinstance(owner, str) and owner else None
        return owners

    def upsert_points(
        self, points: Sequence[RejectionPoint], *, model: str, may_create: bool
    ) -> None:
        self._record(
            "upsert_points",
            {"ids": [p.point_id for p in points], "model": model, "may_create": may_create},
        )
        if not isinstance(model, str) or not model:
            raise ValueError("A rejection point's model must be a non-empty string.")
        for point in points:
            if not isinstance(point.owner_id, str) or not point.owner_id:
                raise ValueError(f"Rejection point {point.point_id!r} has no owner.")
        if not points:
            return
        if self.unreachable:
            raise VectorStoreUnreachableError("fake: upsert could not connect")
        if not self.present:
            if not may_create:
                raise CollectionMissingError(
                    "fake: the collection does not exist and this write may not create it",
                    collection="fake-rejections",
                )
            self.created += 1
            if self.create_then_missing:
                raise CollectionMissingError(
                    "fake: the collection was created and the retry still answered missing",
                    collection="fake-rejections",
                )
            self.present = True
        if self.refuse_upsert:
            raise VectorStoreError("fake: upsert refused")
        for point in points:
            self.points[point.point_id] = {
                "vector": list(point.vector),
                "payload": {"owner_id": point.owner_id, "model": model},
            }

    def delete_points(self, point_ids: Sequence[str]) -> None:
        self._record("delete_points", {"ids": list(point_ids)})
        if not point_ids:
            return
        self._reach("delete_points")
        if self.refuse_delete:
            raise VectorStoreError("fake: delete refused")
        for pid in point_ids:
            self.points.pop(pid, None)


# --------------------------------------------------------------------------- #
# The contract scenario
# --------------------------------------------------------------------------- #

OWNER_A = "a" * 64
OWNER_B = "b" * 64
SCENARIO_MODEL = "test-embedding-model"


def axis_vector(axis: int, *, lean: Optional[int] = None) -> List[float]:
    """A unit-ish vector on one axis, optionally leaning a little toward another."""
    vector = [0.0] * EMBEDDING_DIM
    vector[axis] = 1.0
    if lean is not None:
        vector[lean] = 0.2
    return vector


def scenario_points() -> Dict[str, RejectionPoint]:
    """Three items: two under owner A, one under owner B, on three axes."""
    a1 = item_point_id(OWNER_A, "A sentinel point — it sits in every search.")
    a2 = item_point_id(OWNER_A, "A graph row — it outlives the collection.")
    b1 = item_point_id(OWNER_B, "A prefix — it leaks past the sweep.")
    return {
        "a1": RejectionPoint(point_id=a1, owner_id=OWNER_A, vector=axis_vector(0)),
        "a2": RejectionPoint(point_id=a2, owner_id=OWNER_A, vector=axis_vector(1)),
        "b1": RejectionPoint(point_id=b1, owner_id=OWNER_B, vector=axis_vector(2)),
    }


def _raises(error: type, call: Callable[[], Any]) -> None:
    try:
        call()
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def exercise_rejection_index_contract(
    index: Any, *, overrun: Callable[[List[float]], RejectionSearch]
) -> None:
    """Walks one index through the contract every implementation must honour.

    Absent → a retrieve and a listing raise ``CollectionMissingError`` → an unlicensed
    upsert raises it too and creates nothing → a licensed upsert creates → retrieve
    answers the present subset → the owner scroll, the listing and the owner map → search ranks the
    nearest first with its owner → a delete shrinks the listing → an overrun search,
    and one under an already-spent budget, is ``searched=False``, never a raise.

    Args:
        index: A fresh index whose collection does not exist yet.
        overrun: Searches the same index's contents under a budget it overruns. The
            double flips its knob; the live lane asks a silent socket.
    """
    points = scenario_points()
    a1, a2, b1 = points["a1"], points["a2"], points["b1"]
    unknown = item_point_id(OWNER_B, "Never written.")

    _raises(CollectionMissingError, lambda: index.retrieve_present([a1.point_id]))
    _raises(CollectionMissingError, lambda: index.list_point_ids())
    _raises(CollectionMissingError, lambda: index.list_point_owners())
    assert index.retrieve_present([]) == set()  # empty input asks nothing

    _raises(
        CollectionMissingError,
        lambda: index.upsert_points([a1], model=SCENARIO_MODEL, may_create=False),
    )
    _raises(CollectionMissingError, lambda: index.list_point_ids())

    index.upsert_points([a1, a2, b1], model=SCENARIO_MODEL, may_create=True)

    assert index.retrieve_present([a1.point_id, b1.point_id, unknown]) == {
        a1.point_id, b1.point_id,
    }
    assert index.list_owner_point_ids(OWNER_A) == {a1.point_id, a2.point_id}
    assert index.list_owner_point_ids(OWNER_B) == {b1.point_id}
    assert index.list_point_ids() == {a1.point_id, a2.point_id, b1.point_id}
    assert index.list_point_owners() == {
        a1.point_id: OWNER_A, a2.point_id: OWNER_A, b1.point_id: OWNER_B,
    }

    answer = index.search(axis_vector(1, lean=0), 3, budget_seconds=5.0)
    assert answer.searched is True
    assert [hit.point_id for hit in answer.hits][:2] == [a2.point_id, a1.point_id]
    assert answer.hits[0].owner_id == OWNER_A
    assert answer.hits[0].score > answer.hits[1].score

    index.delete_points([a1.point_id])
    assert index.list_point_ids() == {a2.point_id, b1.point_id}
    assert index.list_owner_point_ids(OWNER_A) == {a2.point_id}
    assert index.list_point_owners() == {a2.point_id: OWNER_A, b1.point_id: OWNER_B}

    late = overrun(axis_vector(1))
    assert late.searched is False and late.hits == []

    # A budget already spent asks nothing and is not searched, never a raise.
    for spent in (0.0, -0.5, float("nan")):
        none = index.search(axis_vector(1), 3, budget_seconds=spent)
        assert none.searched is False and none.hits == []
