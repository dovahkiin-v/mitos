"""Canned vector hits that honour ``VectorStore.query``'s whole contract (Phase 2a).

Every hit a production ``query`` returns carries ``point_id`` — Qdrant's id for the
point, ``hash_to_uuid(node_id)`` for a mitos write — and readers map hits to nodes
through that id, never through the payload slug (a handle: mutable, MI-2, and shared
across a same-slug supersession, MI-13). A test double that returned no id would
empty every window it fed once readers resolve by identity, and a row asserting an
absence — no pause, no blackout — would pass over the window its double emptied
(vision §6.2 constraint 8). So every hit-returning double returns through here:

- :func:`point_hit` builds a production-shaped hit for a named node.
- :func:`orphan_hit` builds a hit that resolves to no node, declared as such.
- :class:`CannedVector` replays slug-authored matches (``{"slug", "score"}``, the
  historical ``_match`` shape), deriving each hit's id at query time from the
  **unique** node carrying that slug across every state (``resolve_slug``). It
  honours ``limit`` and ``with_payload`` exactly as production does.

**Fixture faults are recorded, never raised.** Every reader wraps its search in a
broad catch, so a double raising on a fixture mistake would turn into a degraded or
empty envelope and an absence-asserting row would pass (Gotcha G1). A slug naming no
node (G2) or more than one node (G3), an unbindable double, or an unknown keyed
vector is appended to a per-test record; ``tests/conftest.py``'s autouse
``point_hit_fixture_faults`` fails the test at teardown if any remain unconsumed.

**Binding.** A canned double needs a graph to resolve slugs in. An explicit
``store=`` (or :meth:`CannedVector.bind`) always wins. Otherwise it falls back to
the ``GraphStore`` instances the current test constructed, which the conftest
fixture records per test: it binds only when every recorded store shares one
``realpath(db_path)`` (a read-only and a writable instance over one file are the
same graph). Zero recorded stores, or two or more distinct paths, is a fixture
fault — "bind explicitly" — never a guess. A store built in a module- or
session-scoped fixture is constructed before the per-test record starts, so it is
invisible to the fallback: such a module must bind explicitly.

NOT a ``test_``-prefixed module → pytest does not collect it; it imports as
``from _point_hits import …`` from ``tests/`` and ``tests/golden/`` alike. It
imports ``mitos.vector_store.hash_to_uuid`` and ``mitos.store`` only, so conftest
can import it cheaply.
"""

from __future__ import annotations

import functools
import os
import re
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set

from mitos.store import GraphStore
from mitos.vector_store import hash_to_uuid

#: The payload keys a production hit carries on the default path, in the client's
#: order, with the client's defaults for a missing payload key.
PAYLOAD_DEFAULTS: Dict[str, Any] = {
    "slug": None,
    "scope": [],
    "state": None,
    "kind": None,
    "embedding_text": None,
}

#: Every key of a ``with_payload=True`` hit; ``with_payload=False`` keeps two.
FULL_HIT_KEYS = frozenset({"point_id", "score", *PAYLOAD_DEFAULTS})
IDS_ONLY_HIT_KEYS = frozenset({"point_id", "score"})

_NODE_ID_RE = re.compile(r"[0-9a-f]{64}")

#: Namespace for orphan ids: a UUIDv5 over a tag. No mitos write produces one,
#: because a mitos point id is a content-hash prefix, never a name-based UUID.
_ORPHAN_NAMESPACE = uuid.UUID("6d69746f-732d-6f72-7068-616e2d686974")

_FAULTS: List[str] = []
_RECORDED_STORES: List[GraphStore] = []
_RECORDING = False


# --------------------------------------------------------------------------- #
# Hit builders
# --------------------------------------------------------------------------- #

def point_hit(node_id: str, score: float, **payload: Any) -> Dict[str, Any]:
    """Builds a production-shaped hit for the node ``node_id`` names.

    Args:
        node_id: The node's 64-hex content-hash id (derive it, never hardcode it).
        score: The similarity score.
        **payload: Payload keys (``slug``, ``scope``, ``state``, ``kind``,
            ``embedding_text``); a missing one takes the client's default.

    Returns:
        ``{point_id, slug, scope, state, kind, embedding_text, score}``.

    Raises:
        ValueError: If ``node_id`` is not 64 lowercase hex characters, or a
            payload key is not one production carries. Called at authoring time,
            outside any reader's catch.
    """
    if not isinstance(node_id, str) or _NODE_ID_RE.fullmatch(node_id) is None:
        raise ValueError(f"point_hit needs a 64-hex node id, got {node_id!r}")
    unknown = set(payload) - set(PAYLOAD_DEFAULTS)
    if unknown:
        raise ValueError(f"point_hit: not payload keys: {sorted(unknown)}")
    hit: Dict[str, Any] = {"point_id": hash_to_uuid(node_id)}
    for key, default in PAYLOAD_DEFAULTS.items():
        hit[key] = payload.get(key, list(default) if isinstance(default, list) else default)
    hit["score"] = score
    return hit


def orphan_point_id(tag: str = "orphan") -> str:
    """Returns a well-formed point id that no node carries, derived from ``tag``."""
    return str(uuid.uuid5(_ORPHAN_NAMESPACE, tag))


def orphan_hit(score: float, *, slug: Optional[str] = None, tag: str = "orphan",
               **payload: Any) -> Dict[str, Any]:
    """Builds a hit that resolves to no node, declared as such.

    Its id is a well-formed UUID (:func:`orphan_point_id`) no content hash
    produces, so a resolver files it as an orphan; ``slug`` is whatever payload
    slug the stale point would still carry (``None`` by default).

    Args:
        score: The similarity score.
        slug: The payload slug the orphaned point carries.
        tag: Distinguishes several orphans in one window.
        **payload: Further payload keys, as :func:`point_hit` takes them.

    Returns:
        A production-shaped hit.
    """
    unknown = set(payload) - set(PAYLOAD_DEFAULTS)
    if unknown:
        raise ValueError(f"orphan_hit: not payload keys: {sorted(unknown)}")
    hit: Dict[str, Any] = {"point_id": orphan_point_id(tag)}
    for key, default in PAYLOAD_DEFAULTS.items():
        hit[key] = payload.get(key, list(default) if isinstance(default, list) else default)
    hit["slug"] = slug
    hit["score"] = score
    return hit


def project_hit(hit: Dict[str, Any], with_payload: bool) -> Dict[str, Any]:
    """Projects a full hit onto the shape ``query(…, with_payload=…)`` returns.

    Args:
        hit: A hit carrying at least ``point_id`` and ``score``.
        with_payload: The flag the caller passed.

    Returns:
        A fresh dict: the full hit (payload defaults filled) when True, exactly
        ``{point_id, score}`` when False.
    """
    if not with_payload:
        return {"point_id": hit["point_id"], "score": hit["score"]}
    out: Dict[str, Any] = {"point_id": hit["point_id"]}
    for key, default in PAYLOAD_DEFAULTS.items():
        out[key] = hit.get(key, list(default) if isinstance(default, list) else default)
    out["score"] = hit["score"]
    return out


# --------------------------------------------------------------------------- #
# The fault record and the store record (driven by tests/conftest.py)
# --------------------------------------------------------------------------- #

def record_fault(message: str) -> None:
    """Records a fixture fault; the conftest teardown fails the test on it."""
    _FAULTS.append(message)


def take_faults() -> List[str]:
    """Returns and clears every unconsumed fixture fault (a row consuming its own)."""
    faults = list(_FAULTS)
    _FAULTS.clear()
    return faults


def install_store_recorder() -> None:
    """Wraps ``GraphStore.__init__`` once so each construction is recorded per test.

    Called by ``tests/conftest.py`` at import, never by this module's import: a
    serve child that loads this leaf for :func:`point_hit` records nothing.
    """
    global _RECORDING
    if _RECORDING:
        return
    original = GraphStore.__init__

    @functools.wraps(original)
    def recording_init(self: GraphStore, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        _RECORDED_STORES.append(self)

    GraphStore.__init__ = recording_init  # type: ignore[method-assign]
    _RECORDING = True


def begin_test() -> None:
    """Starts a test's record: no stores yet, no faults carried in."""
    _RECORDED_STORES.clear()
    _FAULTS.clear()


def end_test() -> List[str]:
    """Ends a test's record and returns the faults it left unconsumed."""
    _RECORDED_STORES.clear()
    return take_faults()


def recorded_store() -> Optional[GraphStore]:
    """The one graph this test constructed, or None (recording a fault) if not one.

    Returns:
        A recorded store when every recorded store shares one
        ``realpath(db_path)``; otherwise None, with a "bind explicitly" fault
        recorded.
    """
    paths = {os.path.realpath(store.db_path) for store in _RECORDED_STORES}
    if len(paths) == 1:
        return _RECORDED_STORES[0]
    if not paths:
        record_fault(
            "canned vector double has no store to bind: this test constructed no "
            "GraphStore (or built it in a wider-scoped fixture); bind explicitly "
            "with store=… or .bind(store)"
        )
    else:
        record_fault(
            "canned vector double cannot choose a store: this test constructed "
            f"GraphStores over {len(paths)} distinct graphs {sorted(paths)}; bind "
            "explicitly with store=… or .bind(store)"
        )
    return None


# --------------------------------------------------------------------------- #
# The canned double
# --------------------------------------------------------------------------- #

def derive_hits(matches: Iterable[Dict[str, Any]], store: Optional[Any],
                *, owner: str = "canned vector double") -> List[Dict[str, Any]]:
    """Turns slug-authored matches into production-shaped hits, recording faults.

    A match already carrying ``point_id`` (authored by :func:`point_hit`,
    :func:`orphan_hit`, or raw) passes through. A slug-authored match takes the id
    of the unique node carrying its slug in any state; a slug naming no node (G2)
    or several (G3) is a fixture fault and the hit is dropped.

    Args:
        matches: The canned matches, in ranked order.
        store: The graph to resolve slugs in, or None to use the recorded one.
        owner: Names the double in a fault message.

    Returns:
        The hits that could be derived, in order.
    """
    hits: List[Dict[str, Any]] = []
    bound = store
    for match in matches:
        if "point_id" in match:
            hits.append(match)
            continue
        slug = match.get("slug")
        if bound is None:
            bound = recorded_store()
            if bound is None:
                return []
        node_ids = bound.resolve_slug(slug) if isinstance(slug, str) else []
        if len(node_ids) != 1:
            kind = "no node" if not node_ids else f"{len(node_ids)} nodes {sorted(node_ids)}"
            record_fault(
                f"{owner}: canned slug {slug!r} names {kind}; author the hit with "
                "point_hit(node_id, …) for the node the test means, or orphan_hit(…) "
                "if the point is meant to resolve to nothing"
            )
            continue
        payload = {k: v for k, v in match.items() if k in PAYLOAD_DEFAULTS}
        hits.append(point_hit(node_ids[0], match["score"], **payload))
    return hits


class CannedVector:
    """Replays canned matches through the shared leaf; records every call.

    Attributes:
        queries: How many times :meth:`query` ran.
        last_limit: The ``limit`` of the latest call (None before any).
        calls: ``(limit, with_payload)`` per call, in order.
        deleted: Every point id :meth:`delete_points` removed; a later query no
            longer returns a hit carrying one.
        delete_calls: The ids each :meth:`delete_points` call received, in order.
    """

    def __init__(self, matches: Optional[List[Dict[str, Any]]] = None, *,
                 store: Optional[Any] = None,
                 store_of: Optional[Callable[[], Any]] = None) -> None:
        # Held by reference, as the doubles this replaces held it: a row may
        # extend its list after wiring.
        self._matches = matches if matches is not None else []
        self._store = store
        # A late-bound store (e.g. ``lambda: manager.store``), read at query time,
        # for a wiring helper whose row may swap the manager's store afterwards.
        self._store_of = store_of
        self.queries = 0
        self.last_limit: Optional[int] = None
        self.calls: List[tuple] = []
        self.deleted: Set[str] = set()
        self.delete_calls: List[List[str]] = []

    def bind(self, store: Any) -> "CannedVector":
        """Binds the graph slugs resolve in; an explicit bind beats the fallback."""
        self._store = store
        return self

    def canned_for(self, vector: List[float]) -> Optional[List[Dict[str, Any]]]:
        """The canned matches for ``vector``; None records a fault (subclass hook)."""
        return self._matches

    def query(self, vector: List[float], limit: int = 5, *,
              with_payload: bool = True) -> List[Dict[str, Any]]:
        """Returns the canned hits, ``limit``-sliced and ``with_payload``-projected."""
        self.queries += 1
        self.last_limit = limit
        self.calls.append((limit, with_payload))
        canned = self.canned_for(vector)
        if canned is None:
            return []
        store = self._store
        if store is None and self._store_of is not None:
            store = self._store_of()
        hits = derive_hits(canned, store, owner=type(self).__name__)
        # A deleted point is gone from the index: filtered after the hits are derived
        # and before the limit, as a real store would rank only what it still holds.
        hits = [hit for hit in hits if hit["point_id"] not in self.deleted]
        return [project_hit(hit, with_payload) for hit in hits[:limit]]

    def delete_points(self, point_ids: Sequence[str]) -> None:
        """Records the call and removes the ids from every later query's hits."""
        self.delete_calls.append(list(point_ids))
        self.deleted.update(point_ids)
