"""Qdrant-backed vector store for Mitos.

This module implements the vector store pipeline (D) using the Qdrant REST API
directly, reducing dependency bloat and ensuring maximum interoperability.

Three contracts hold across every member here. **Reading never writes:** neither
constructing a store nor querying/scrolling one can bring a collection into
existence — creation attaches to the writes, :meth:`QdrantVectorStore.upsert` and
:meth:`QdrantRejectionIndex.upsert_points`, lazily, on a 404. **Missing is not
unreachable:** an absent collection raises the typed
:class:`~mitos.errors.CollectionMissingError`, which every surface words as the
recoverable state it is (``mitos reconcile``) rather than as an outage. And **not
every write may create:** the upsert takes a required ``may_create`` declaration, so
a write covering only itself leaves an absent collection absent rather than minting a
one-point index over a populated corpus. Absence is a signal several nets key on; a
single write must not be able to spend it.
"""

import requests
import json
from typing import (
    AbstractSet, Any, Dict, FrozenSet, List, NamedTuple, Optional, Sequence, Set,
)
from mitos.errors import (
    CollectionMissingError,
    VectorStoreError,
    VectorStoreUnreachableError,
)
from mitos.models import EMBEDDING_DIM


def _collection_is_missing(resp: "requests.Response") -> bool:
    """Decides whether a Qdrant response means "that collection does not exist".

    The single place this question is answered, so a fifth endpoint cannot drift
    from the other four. Keyed on the **status code**, never on the body text:
    measured 2026-07-30 against Qdrant 1.16.3, an absent collection answers 404 to
    ``GET /collections/{c}``, ``POST …/points/search``, ``POST …/points/scroll``
    and ``PUT …/points`` alike, while the error string carries backticks and an
    exclamation mark and is plainly version-shaped.

    Args:
        resp: The Qdrant response.

    Returns:
        True when the response says the collection is absent.
    """
    return resp.status_code == 404


def _raise_response_error(resp: "requests.Response", *, operation: str,
                          collection: str) -> None:
    """Raises the typed error for a non-200 Qdrant response — missing ≠ unreachable.

    Args:
        resp: The non-200 response.
        operation: The operation name for the message (``"query"``, ``"upsert"``, …).
        collection: The collection the operation addressed.

    Raises:
        CollectionMissingError: If the collection does not exist (HTTP 404).
        VectorStoreError: For any other non-200 response.
    """
    if _collection_is_missing(resp):
        raise CollectionMissingError(
            f"Qdrant collection '{collection}' does not exist "
            f"(Qdrant is up and answered 404 to the {operation}).",
            collection=collection,
        )
    raise VectorStoreError(f"Qdrant {operation} failed: {resp.text}")


def _is_point_id(value: Any) -> bool:
    """Reports whether a value is a Qdrant point id: a string or an integer.

    ``bool`` is an ``int`` subclass and is refused explicitly; ``None`` is refused,
    so a hit carrying ``"id": null`` is a malformed answer, never the string
    ``"None"``.

    Args:
        value: The ``id`` a search hit carried, or None when it carried none.

    Returns:
        True if the value can be a point id.
    """
    return isinstance(value, (str, int)) and not isinstance(value, bool)


def _decode_result(resp: "requests.Response", *, operation: str) -> Any:
    """Extracts Qdrant's ``result`` from a 200 body, or raises a typed error.

    A 200 carrying a non-JSON body, or a JSON body of the wrong shape, is a
    substrate fault like any other and must reach the caller as a
    :class:`VectorStoreError`. Left unguarded it escapes as a raw
    ``ValueError``/``AttributeError`` — ``requests.RequestException`` does not
    cover a JSON decode failure — slipping every net keyed on the typed class and
    landing on a generic fatal handler. On the read path that matters more than it
    sounds: a malformed 200 is the one shape that could reach a fail-closed
    consumer looking like a healthy empty answer.

    An **absent** ``result`` key raises too, rather than defaulting to empty. The
    two states must not blur: Qdrant says "nothing matched" with ``result: []``, and
    a body with no ``result`` at all is a different protocol, not an empty answer.
    Defaulting it (the shipped ``.get("result", [])``) is the precise shape that
    reaches a fail-closed consumer as a clean, certified-complete zero.

    Args:
        resp: A 200 response.
        operation: The operation name for the message.

    Returns:
        The value of the body's ``result`` key.

    Raises:
        VectorStoreError: If the body is not JSON, is not a JSON object, or carries
            no ``result`` key.
    """
    try:
        body = resp.json()
    except ValueError as exc:
        raise VectorStoreError(
            f"Qdrant {operation} returned a malformed (non-JSON) body: {exc}"
        )
    if not isinstance(body, dict):
        raise VectorStoreError(
            f"Qdrant {operation} returned an unexpected body shape "
            f"({type(body).__name__}, expected a JSON object)."
        )
    if "result" not in body:
        raise VectorStoreError(
            f"Qdrant {operation} returned a 200 with no `result` key — a body this "
            "shape is a protocol mismatch, not an empty answer."
        )
    return body["result"]


def hash_to_uuid(sha256_hex: str) -> str:
    """Converts a 64-character SHA-256 hex string deterministically into a UUID format.

    Args:
        sha256_hex: A hex string of length 64.

    Returns:
        A 36-character standard UUID string.
    """
    sha = sha256_hex.lower()
    return f"{sha[:8]}-{sha[8:12]}-{sha[12:16]}-{sha[16:20]}-{sha[20:32]}"


def _scroll_points(
    base_url: str,
    collection: str,
    *,
    page_size: int,
    payload_filter: Optional[Dict[str, Any]] = None,
    with_payload: Any = False,
) -> List[Dict[str, Any]]:
    """Enumerates a collection's points by a paginated, no-create scroll.

    The one scroll body in this module. ``_scroll_ids`` and
    :meth:`QdrantRejectionIndex.list_point_owners` read through it, so a page's
    shape guard cannot fork between them. Vectors are excluded from every page, and
    payloads unless ``with_payload`` names the keys to take back;
    ``payload_filter``, when given, is sent as Qdrant's ``filter``.

    The three failure directions are typed apart: a transport fault raises
    :class:`VectorStoreUnreachableError`, a 404 :class:`CollectionMissingError`, and
    any other non-200 or a 200 of the wrong shape plain :class:`VectorStoreError`.

    Args:
        base_url: The Qdrant base URL (trailing slash tolerated).
        collection: The collection name to scroll.
        page_size: Maximum points fetched per scroll page.
        payload_filter: A Qdrant filter object, or None for every point.
        with_payload: Qdrant's ``with_payload``: False, or a list of payload keys.

    Returns:
        Every point object the scroll returned, each carrying an ``id``.

    Raises:
        VectorStoreUnreachableError: If Qdrant did not answer.
        CollectionMissingError: If Qdrant is up but the collection does not exist.
        VectorStoreError: If Qdrant answered another non-200 status, or a 200 with a
            body of the wrong shape.
    """
    scroll_url = f"{base_url.rstrip('/')}/collections/{collection}/points/scroll"
    found: List[Dict[str, Any]] = []
    offset: Any = None
    try:
        while True:
            body: Dict[str, Any] = {
                "limit": page_size,
                "with_payload": with_payload,
                "with_vector": False,
            }
            if payload_filter is not None:
                body["filter"] = payload_filter
            if offset is not None:
                body["offset"] = offset
            resp = requests.post(
                scroll_url,
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            if resp.status_code != 200:
                _raise_response_error(
                    resp, operation="scroll", collection=collection
                )

            # No `or {}` — a JSON `null` result must reach the shape guard below
            # rather than be coerced into a clean empty page. That coercion is the
            # same certified-complete-zero the absent-key guard exists to refuse.
            result = _decode_result(resp, operation="scroll")
            if not isinstance(result, dict):
                raise VectorStoreError(
                    "Qdrant scroll returned an unexpected result shape "
                    f"({type(result).__name__}, expected an object)."
                )
            points = result.get("points") or []
            if not isinstance(points, list) or any(
                not isinstance(point, dict) or "id" not in point for point in points
            ):
                raise VectorStoreError(
                    "Qdrant scroll returned an unexpected points shape "
                    "(expected a list of objects carrying an `id`)."
                )
            found.extend(points)

            offset = result.get("next_page_offset")
            if offset is None:
                break
        return found
    except requests.RequestException as e:
        raise VectorStoreUnreachableError(
            f"Qdrant scroll connection error: {str(e)}"
        ) from e


def _scroll_ids(
    base_url: str,
    collection: str,
    *,
    page_size: int,
    payload_filter: Optional[Dict[str, Any]] = None,
) -> Set[str]:
    """Enumerates point ids in a collection by a paginated, no-create scroll.

    ``scroll_point_ids`` and both collections' clients list ids through this, over
    the one scroll body (:func:`_scroll_points`), with payloads excluded from every
    page; ``payload_filter``, when given, is sent as Qdrant's ``filter`` (the
    rejection index's owner scroll).

    The three failure directions are typed apart: a transport fault raises
    :class:`VectorStoreUnreachableError`, a 404 :class:`CollectionMissingError`, and
    any other non-200 or a 200 of the wrong shape plain :class:`VectorStoreError`.

    Args:
        base_url: The Qdrant base URL (trailing slash tolerated).
        collection: The collection name to scroll.
        page_size: Maximum points fetched per scroll page.
        payload_filter: A Qdrant filter object, or None for every point.

    Returns:
        The set of point-id strings in the collection (or matching the filter).

    Raises:
        VectorStoreUnreachableError: If Qdrant did not answer.
        CollectionMissingError: If Qdrant is up but the collection does not exist.
        VectorStoreError: If Qdrant answered another non-200 status, or a 200 with a
            body of the wrong shape.
    """
    return {
        str(point["id"])
        for point in _scroll_points(
            base_url, collection, page_size=page_size, payload_filter=payload_filter
        )
    }


def scroll_point_ids(base_url: str, collection: str, page_size: int = 256) -> Set[str]:
    """Enumerates every point id in a collection via a no-create paginated scroll.

    A module-level read path that talks to Qdrant directly without constructing a
    :class:`QdrantVectorStore` — kept so a read-only probe (e.g. ``mitos status``)
    can diff the graph against Qdrant's actual points without needing a store at
    all. Reads create nothing: an absent collection is *reported*, never
    materialized. Payloads and vectors are excluded from the response to keep each
    page cheap; the scan is bounded to one pass over the collection (no per-node
    existence probes).

    Its transport arm raises plain :class:`VectorStoreError`, unlike the shared
    :func:`_scroll_ids` it delegates to: the decision index's readers (``status``,
    ``reconcile``) were written against that contract, and it does not move here.

    Args:
        base_url: The Qdrant base URL (trailing slash tolerated).
        collection: The collection name to scroll.
        page_size: Maximum points fetched per scroll page.

    Returns:
        The set of point-id strings (the ``hash_to_uuid`` UUIDs) in the collection.

    Raises:
        CollectionMissingError: If Qdrant is up but the collection does not exist.
        VectorStoreError: If Qdrant is unreachable, returns another non-200 status,
            or answers 200 with a body of the wrong shape.
    """
    try:
        return _scroll_ids(base_url, collection, page_size=page_size)
    except VectorStoreUnreachableError as e:
        raise VectorStoreError(str(e)) from e


def list_collection_names(base_url: str, *, timeout: float) -> Set[str]:
    """Names every collection on a Qdrant instance — one call, instance-scoped.

    The bulk shape the global overview needs: whether N projects' collections
    exist is answered by **one** request to the instance plus a local set
    membership test, never by N per-collection probes (``mitos status``'
    ``_check_qdrant`` is the per-collection shape, and it stays the deep report's).
    So the call budget scales with distinct instances and never with project count.

    This endpoint addresses the **instance**, not a collection, which is why a
    non-200 is never routed through :func:`_raise_response_error`: that helper maps
    404 → :class:`CollectionMissingError`, and a 404 *here* means a wrong base URL
    or a proxy in the way — an instance fault mis-filed as a missing collection at
    exactly the boundary a consumer's tri-state depends on.

    The two failure directions are typed apart, because a caller reporting instance
    health must not read "answered with something unusable" as "did not answer":
    a transport fault raises :class:`VectorStoreUnreachableError`, everything else
    raises plain :class:`VectorStoreError`.

    Args:
        base_url: The Qdrant base URL (trailing slash tolerated).
        timeout: Per-request socket timeout in seconds.

    Returns:
        The set of collection names on the instance. Empty is a legitimate answer
        (a fresh instance) and is never confused with an unavailable listing — that
        one raises.

    Raises:
        VectorStoreUnreachableError: If the instance did not answer at all.
        VectorStoreError: If it answered with a non-200, a non-JSON or wrongly
            shaped body, or a listing that is not a list of named objects.
    """
    url = f"{base_url.rstrip('/')}/collections"
    try:
        resp = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        raise VectorStoreUnreachableError(
            f"Qdrant collection listing connection error: {str(e)}"
        ) from e
    if resp.status_code != 200:
        raise VectorStoreError(
            f"Qdrant collection listing failed (HTTP {resp.status_code}): {resp.text}"
        )

    result = _decode_result(resp, operation="collection listing")
    if not isinstance(result, dict):
        raise VectorStoreError(
            "Qdrant collection listing returned an unexpected result shape "
            f"({type(result).__name__}, expected an object)."
        )
    # No `or []` — deliberately unlike `scroll_point_ids`' `points` line one module
    # over. An unusable listing and an empty instance are the two states this
    # function's whole contract keeps apart, and coalescing a `null` into an empty
    # list collapses them into "your collection is missing" for every project on
    # the instance.
    collections = result.get("collections")
    if not isinstance(collections, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("name"), str)
        for item in collections
    ):
        raise VectorStoreError(
            "Qdrant collection listing returned an unexpected collections shape "
            "(expected a list of objects carrying a `name`)."
        )
    return {item["name"] for item in collections}


def _delete_points(base_url: str, collection: str, point_ids: Sequence[str]) -> None:
    """Deletes exactly these points by their listed ids; confirmed or raised.

    The one delete body in this module, shared by :meth:`QdrantVectorStore.delete_points`
    and :meth:`QdrantRejectionIndex.delete_points` so the two contracts cannot fork.
    The contract itself is documented on :meth:`QdrantVectorStore.delete_points`.

    Args:
        base_url: The Qdrant base URL, without a trailing slash.
        collection: The collection to delete from.
        point_ids: The listed point ids to delete. Empty sends no request.

    Raises:
        VectorStoreUnreachableError: If Qdrant did not answer.
        CollectionMissingError: If Qdrant is up but the collection does not exist.
        VectorStoreError: If Qdrant answered and refused or did not confirm.
    """
    if not point_ids:
        return
    delete_url = (
        f"{base_url}/collections/{collection}/points/delete?wait=true"
    )
    body = {
        "points": [
            int(pid) if pid.isascii() and pid.isdigit() else pid
            for pid in point_ids
        ]
    }
    try:
        resp = requests.post(
            delete_url,
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
    except requests.RequestException as e:
        raise VectorStoreUnreachableError(
            f"Qdrant connection error during delete: {str(e)}"
        ) from e
    if resp.status_code != 200:
        _raise_response_error(
            resp, operation="delete", collection=collection
        )
    result = _decode_result(resp, operation="delete")
    status = result.get("status") if isinstance(result, dict) else None
    if status != "completed":
        raise VectorStoreError(
            "Qdrant delete was not confirmed (expected result status "
            f"'completed', got {status!r}); the points may still be there."
        )


class QdrantVectorStore:
    """REST client for Qdrant vector store managing points and semantic queries."""

    def __init__(self, qdrant_url: str, collection_name: str) -> None:
        """Binds the store to a Qdrant endpoint and collection — **no network I/O**.

        Construction is pure: it dispatches zero HTTP requests and therefore cannot
        create, probe, or fail. Creation attaches to the *write* instead (see
        :meth:`upsert`), which is what makes a read structurally incapable of
        materializing the collection it was only meant to look at — one
        constructor serves every reader and every writer, so the policy cannot
        live here.

        Args:
            qdrant_url: The Qdrant REST endpoint (trailing slash tolerated).
            collection_name: The project's collection. **Required** — it used to
                default to ``"mitos"``, which is the exact name a basename-less or
                wholly-non-Latin workspace once derived, i.e. the shared collection
                the path-hash derivation exists to abolish. Every construction site
                already passes it, so the default was dead weight; what it could
                still do is land a *future* site silently in a shared namespace.
                Declared, not defaulted — the property, not the callers that hold it
                today.
        """
        self.base_url = qdrant_url.rstrip("/")
        self.collection = collection_name

    def _ensure_collection(self) -> None:
        """Verifies if the collection exists, creating it with Cosine configuration if missing.

        Reached from exactly one place: :meth:`upsert`'s 404 recovery branch. It is
        the only path in this module that may create.
        """
        check_url = f"{self.base_url}/collections/{self.collection}"
        try:
            resp = requests.get(check_url, timeout=5)
            if resp.status_code == 200:
                # Collection exists
                return

            if _collection_is_missing(resp):
                # Create collection
                create_url = f"{self.base_url}/collections/{self.collection}"
                payload = {
                    "vectors": {
                        "size": EMBEDDING_DIM,  # Size of gemini-embedding-2
                        "distance": "Cosine"
                    }
                }
                c_resp = requests.put(
                    create_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=5
                )
                if c_resp.status_code != 200:
                    raise VectorStoreError(
                        f"Failed to create Qdrant collection: {c_resp.text}"
                    )
            else:
                raise VectorStoreError(
                    f"Unexpected Qdrant response checking collection: {resp.text}"
                )
        except requests.RequestException as e:
            raise VectorStoreError(f"Qdrant connection refused: {str(e)}")

    def upsert(self, point_id: str, vector: List[float], payload: Dict[str, Any],
               *, may_create: bool) -> None:
        """Upserts a single point into Qdrant using the deterministic UUID mapping.

        Creation lives here, lazily: the write is attempted first, and only a 404 —
        the collection does not exist — reaches the creation decision. When the caller
        declared ``may_create``, that is ``_ensure_collection`` and **one** retry; when
        it did not, the absent collection is left absent and reported. The healthy case
        pays zero extra round trips either way, and a 404 that turns out to mean
        something else (a renamed endpoint in a future Qdrant) costs one wasted create
        and then raises on the second 404 — never a loop, never a silent success.

        ``may_create`` is **required and keyword-only** on purpose. A default of False
        would be safe today and would silently lose the contract tomorrow: a future
        call site that omits it defers forever and reads as working, because the outbox
        merely grows. A default of True re-arms the hazard the parameter exists to
        close. Required makes every write site declare, and turns an un-migrated one
        into a ``TypeError`` rather than a behaviour change nobody sees. The declaration
        is read **only on a 404**, so a healthy project's every write may pass False
        forever with nothing changing.

        Args:
            point_id: The SHA-256 node ID.
            vector: The embedding vector values.
            payload: Node metadata {slug, scope, state, kind, embedding_text}.
            may_create: Whether this write covers the decision index's expected set
                (every decision plus the active open questions) and may therefore
                bring an absent collection into existence. Callers derive it
                from declared intent — the graph gate for a single-node write, the
                ``embedding_seed`` marker for a drain — never from an inferred
                comparison.

        Raises:
            CollectionMissingError: If the collection is absent and this write may not
                create it, or if it is still absent after the create-and-retry.
            VectorStoreError: If Qdrant is unreachable or rejects the write.
        """
        uuid_id = hash_to_uuid(point_id)
        upsert_url = f"{self.base_url}/collections/{self.collection}/points"

        body = {
            "points": [
                {
                    "id": uuid_id,
                    "vector": vector,
                    "payload": payload
                }
            ]
        }

        def _put() -> "requests.Response":
            return requests.put(
                upsert_url,
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=5
            )

        try:
            resp = _put()
            if _collection_is_missing(resp):
                # ── The one place an absent collection is created. ──────────────
                # Narrowed to writes that COVER what the decision index should hold
                # (every decision plus the active open questions). A write
                # covering only itself defers here: creating would leave a collection
                # holding one point out of N, and from the next write on every net
                # keyed on absence would report "checked, clean" over an index that
                # was never checked. Deferral is cheap — the commit already wrote the
                # outbox row, so the work is queued, not lost.
                if not may_create:
                    raise CollectionMissingError(
                        f"Qdrant collection '{self.collection}' does not exist, and "
                        f"this write does not cover every decision the index should "
                        f"hold — "
                        f"deferring rather than creating an index holding only part "
                        f"of the corpus. Run `mitos reconcile` to build it in one pass.",
                        collection=self.collection,
                    )
                self._ensure_collection()
                resp = _put()
            if resp.status_code != 200:
                _raise_response_error(
                    resp, operation="upsert", collection=self.collection
                )
        except requests.RequestException as e:
            raise VectorStoreError(f"Qdrant connection error during upsert: {str(e)}")

    def query(
        self,
        vector: List[float],
        limit: int = 5,
        *,
        with_payload: bool = True,
    ) -> List[Dict[str, Any]]:
        """Queries Qdrant for the semantically nearest vectors.

        Semantic recall is deliberately scope-blind: a decision is the precedent on
        its subject regardless of which scope drawer it was filed under, and gating
        the search by a caller-guessed tag silently hides real precedent (the
        ``gemini-live`` vs ``live-voice`` drift). Scope is handled downstream as a
        discoverability hint in :mod:`mitos.recall`, never as a recall filter.

        Every hit carries ``point_id``: the Qdrant point id exactly as Qdrant
        returns it, as a string. For a point mitos wrote it is
        ``hash_to_uuid(node_id)`` — the first 128 bits of the node's content-hash
        id, so the hit names its node — and it is **not** the 64-hex node id that
        ``id`` means on a node dict (nor ``upsert``'s ``point_id`` argument, which
        is the node id). A reader maps a hit to its node through this id
        (:meth:`~mitos.store.GraphStore.resolve_point_ids`); the payload slug is a
        handle, mutable and shareable across a same-slug supersession.

        Args:
            vector: The query embedding vector.
            limit: Maximum matches to return.
            with_payload: When True (the default), each hit is
                ``{point_id, slug, scope, state, kind, embedding_text, score}``,
                a missing payload key reading ``[]`` for ``scope`` and ``None``
                otherwise. When False, the request asks Qdrant for no payload and
                each hit is exactly ``{point_id, score}``: no payload key appears,
                not even as ``None``.

        Returns:
            A list of hit dicts in Qdrant's ranked order.

        Raises:
            CollectionMissingError: If Qdrant is up but the collection does not
                exist. Reads never create it — absence is reported, and the
                calling surface decides whether that is a gap or simply the empty
                index of a project with nothing to index yet.
            VectorStoreError: If Qdrant is unreachable, returns another non-200
                status, or answers 200 with a body of the wrong shape — including
                a hit with no ``id``, or (when a payload was asked for) a hit whose
                payload is not an object.
        """
        search_url = f"{self.base_url}/collections/{self.collection}/points/search"

        body: Dict[str, Any] = {
            "vector": vector,
            "limit": limit,
            "with_payload": with_payload
        }

        try:
            resp = requests.post(
                search_url,
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            if resp.status_code != 200:
                _raise_response_error(
                    resp, operation="query", collection=self.collection
                )

            # No `or []` — see the scroll's twin above. On the read path a `null`
            # result coerced to empty is the one shape that reaches a fail-closed
            # consumer looking like a healthy "nothing matched".
            results = _decode_result(resp, operation="query")
            # A hit with no `id` is the scroll's "200 of the wrong shape", refused
            # the same way: coalesced to None, it would reach the resolver as an
            # orphan and a whole window would read "nothing here" over a malformed
            # answer. An `id` that is neither a string nor an integer (Qdrant's two
            # point-id types) is the same fault by another spelling — `null` would
            # otherwise become the string "None". A payload that is not an object
            # (a `null` on the default path) is refused here too, rather than
            # escaping untyped below.
            if not isinstance(results, list) or any(
                not isinstance(item, dict)
                or not _is_point_id(item.get("id"))
                or (with_payload and not isinstance(item.get("payload", {}), dict))
                for item in results
            ):
                raise VectorStoreError(
                    "Qdrant query returned an unexpected result shape "
                    "(expected a list of objects, each carrying an `id`"
                    + (" and an object payload)." if with_payload else ").")
                )

            output = []
            for item in results:
                hit: Dict[str, Any] = {"point_id": str(item["id"])}
                if with_payload:
                    payload = item.get("payload", {})
                    hit.update({
                        "slug": payload.get("slug"),
                        "scope": payload.get("scope", []),
                        "state": payload.get("state"),
                        "kind": payload.get("kind"),
                        "embedding_text": payload.get("embedding_text"),
                    })
                hit["score"] = item.get("score", 0.0)
                output.append(hit)

            return output

        except requests.RequestException as e:
            raise VectorStoreError(f"Qdrant query connection error: {str(e)}")
            
    def list_point_ids(self, page_size: int = 256) -> Set[str]:
        """Lists every point id currently in the collection via a paginated scroll.

        A thin instance-bound delegate to :func:`scroll_point_ids`, kept so callers
        holding a store instance need not thread the URL + collection by hand. It
        inherits that function's typed contract, including the missing-collection
        classification.

        Args:
            page_size: Maximum points fetched per scroll page.

        Returns:
            The set of point-id strings (the ``hash_to_uuid`` UUIDs) in the collection.

        Raises:
            CollectionMissingError: If Qdrant is up but the collection does not exist.
            VectorStoreError: If Qdrant is unreachable, returns another non-200
                status, or answers 200 with a body of the wrong shape.
        """
        return scroll_point_ids(self.base_url, self.collection, page_size=page_size)

    def delete_points(self, point_ids: Sequence[str]) -> None:
        """Deletes exactly these points, by the index's own ids; confirmed or raised.

        The ids are taken **as the index lists them** — the strings
        :meth:`list_point_ids` / :func:`scroll_point_ids` return — and are never
        re-hashed. That is the contrast with :meth:`upsert`, whose ``point_id``
        argument is a 64-hex node id: a point whose node has left the graph has no
        node id left to hash, so its listed id is the only name it has.

        The scroll stringifies every id, so an all-decimal ASCII string goes back as
        a JSON integer: a foreign integer-id point then round-trips, and the request
        is not refused whole as malformed (measured on Qdrant 1.16.3: a string
        ``"7"`` is a 400 for the entire batch). A non-ASCII digit string stays a
        string, because ``str.isdigit`` alone would admit ``"٧"``.

        The request carries ``wait=true`` and success is a 200 whose ``result``
        reads ``status: "completed"``. Without ``wait`` Qdrant answers
        ``"acknowledged"``, which confirms nothing — the next listing could still
        hold the point — so that answer raises here. Measured on 1.16.3: deleting an
        id that does not exist answers ``completed`` too, so a confirmation says
        "this id is gone", never "this id existed".

        Deletion never creates: a 404 is reported, not healed, so the
        "reading never writes" contract is held by the delete side as well.

        Args:
            point_ids: The listed point ids to delete. Empty sends no request.

        Raises:
            VectorStoreUnreachableError: If Qdrant did not answer (a transport
                fault) — a caller looping over batches stops.
            CollectionMissingError: If Qdrant is up but the collection does not
                exist (HTTP 404) — a caller stops likewise.
            VectorStoreError: If Qdrant answered and refused or did not confirm —
                any other non-200, a body of the wrong shape, or a status other
                than ``completed``. A caller may try its next batch.
        """
        _delete_points(self.base_url, self.collection, point_ids)


class RejectionPoint(NamedTuple):
    """One rejected item's point, ready to write.

    Attributes:
        point_id: The item's point id, already in UUID form
            (``rejections.item_point_id``); written exactly as given.
        owner_id: The owning decision's 64-hex node id.
        vector: The item's embedding.
    """

    point_id: str
    owner_id: str
    vector: List[float]


class RejectionHit(NamedTuple):
    """One rejection-index search hit.

    Attributes:
        point_id: The hit's point id, as Qdrant returned it, as a string.
        owner_id: The payload's ``owner_id``, or None when the point carries no
            non-empty string there (not a fault; a reader drops it).
        score: Qdrant's similarity score.
    """

    point_id: str
    owner_id: Optional[str]
    score: float


class RejectionSearch(NamedTuple):
    """A rejection-index search's answer.

    Attributes:
        searched: False when the search did not finish inside its time budget; the
            hits are then empty and say nothing about the index.
        hits: The hits in Qdrant's ranked order.
    """

    searched: bool
    hits: List[RejectionHit]


# Points per upsert request. A chunk is one ``wait=true`` request, so a failure
# costs at most this many re-sent points (their ids make a re-send idempotent).
_REJECTION_UPSERT_CHUNK: int = 64

# The payload key the owner index covers, and the only one a read takes back.
_OWNER_KEY: str = "owner_id"


class QdrantRejectionIndex:
    """REST client for the rejection index: one point per rejected item.

    It sits in a collection of its own beside the decision collection
    (``MitosConfig.rejection_collection``) and is a client apart from
    :class:`QdrantVectorStore`, not a second instance of it: item ids are already
    UUIDs (never re-hashed), the payload is the owner and the model, and its
    operations are named apart (``search``, ``upsert_points``) so a rejection hit can
    never be taken for a decision hit.

    Every point's payload is exactly ``{"owner_id": <64-hex node id>, "model":
    <embedding model id>}``. The model rides the points because it describes the
    vectors: a graph row would outlive a lost collection, and one row could not
    describe points written under two models.

    One error typing holds for every method except ``search``'s time budget: a
    transport fault raises :class:`VectorStoreUnreachableError`, a 404
    :class:`CollectionMissingError`, and any other non-200 or a 200 of the wrong
    shape plain :class:`VectorStoreError`. So a caller's first request for an owner
    tells an outage from a refusal.
    """

    def __init__(self, qdrant_url: str, collection_name: str) -> None:
        """Binds the client to a Qdrant endpoint and collection — **no network I/O**.

        Args:
            qdrant_url: The Qdrant REST endpoint (trailing slash tolerated).
            collection_name: The rejection collection's name.
        """
        self.base_url = qdrant_url.rstrip("/")
        self.collection = collection_name

    def _points_url(self, suffix: str = "") -> str:
        """Returns this collection's ``/points`` URL with ``suffix`` appended."""
        return f"{self.base_url}/collections/{self.collection}/points{suffix}"

    def search(
        self, vector: List[float], limit: int, *, budget_seconds: float
    ) -> RejectionSearch:
        """Searches the nearest items under a caller-bounded wait.

        ``budget_seconds`` is the request's socket timeout. It bounds each socket
        operation (the connect, then each read), not the wall clock, so a search can
        take up to about twice the budget. A search that runs over it returns
        ``searched=False`` and never raises: a slow index costs its matches and
        nothing else. A budget that is already spent (zero, negative or NaN, as a
        caller's remaining time can be) sends no request and answers the same way;
        ``requests`` would refuse such a timeout with a raw ``ValueError``.

        Args:
            vector: The query embedding.
            limit: Maximum hits to return.
            budget_seconds: The wait bound, in seconds.

        Returns:
            ``RejectionSearch(searched=True, hits=…)`` in Qdrant's order, or
            ``RejectionSearch(searched=False, hits=[])`` on any timeout or a spent
            budget.

        Raises:
            VectorStoreUnreachableError: On a transport fault other than a timeout.
            CollectionMissingError: If the collection does not exist.
            VectorStoreError: On another non-200 or a 200 of the wrong shape,
                including a hit with no ``id``.
        """
        # `not >` rather than `<=`, so a NaN budget counts as spent too.
        if not budget_seconds > 0:
            return RejectionSearch(searched=False, hits=[])
        body: Dict[str, Any] = {
            "vector": vector,
            "limit": limit,
            "with_payload": [_OWNER_KEY],
        }
        try:
            resp = requests.post(
                self._points_url("/search"),
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=budget_seconds,
            )
        # Timeout first: ``ConnectTimeout`` is also a ``ConnectionError``, so the
        # other order would read a slow connect as an outage.
        except requests.Timeout:
            return RejectionSearch(searched=False, hits=[])
        except requests.RequestException as e:
            raise VectorStoreUnreachableError(
                f"Qdrant rejection search connection error: {str(e)}"
            ) from e
        if resp.status_code != 200:
            _raise_response_error(
                resp, operation="rejection search", collection=self.collection
            )
        results = _decode_result(resp, operation="rejection search")
        if not isinstance(results, list) or any(
            not isinstance(item, dict)
            or not _is_point_id(item.get("id"))
            or not isinstance(item.get("score"), (int, float))
            or isinstance(item.get("score"), bool)
            for item in results
        ):
            raise VectorStoreError(
                "Qdrant rejection search returned an unexpected result shape "
                "(expected a list of objects, each carrying an `id` and a `score`)."
            )
        hits = []
        for item in results:
            payload = item.get("payload")
            owner = payload.get(_OWNER_KEY) if isinstance(payload, dict) else None
            hits.append(
                RejectionHit(
                    point_id=str(item["id"]),
                    owner_id=owner if isinstance(owner, str) and owner else None,
                    score=float(item["score"]),
                )
            )
        return RejectionSearch(searched=True, hits=hits)

    def retrieve_present(self, point_ids: Sequence[str]) -> Set[str]:
        """Answers which of these points the index holds, by their ids.

        The existence probe a writer asks before it embeds: a 404 means the
        collection is absent. Ids are sent exactly as given.

        Args:
            point_ids: The point ids to ask about. Empty sends no request.

        Returns:
            The subset present, as strings.

        Raises:
            VectorStoreUnreachableError: If Qdrant did not answer.
            CollectionMissingError: If the collection does not exist.
            VectorStoreError: On another non-200 or a 200 of the wrong shape.
        """
        if not point_ids:
            return set()
        body = {"ids": list(point_ids), "with_payload": False, "with_vector": False}
        try:
            resp = requests.post(
                self._points_url(),
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
        except requests.RequestException as e:
            raise VectorStoreUnreachableError(
                f"Qdrant rejection retrieve connection error: {str(e)}"
            ) from e
        if resp.status_code != 200:
            _raise_response_error(
                resp, operation="rejection retrieve", collection=self.collection
            )
        # No `or []` — a `null` result is a protocol fault, never "none present".
        results = _decode_result(resp, operation="rejection retrieve")
        if not isinstance(results, list) or any(
            not isinstance(item, dict) or not _is_point_id(item.get("id"))
            for item in results
        ):
            raise VectorStoreError(
                "Qdrant rejection retrieve returned an unexpected result shape "
                "(expected a list of objects, each carrying an `id`)."
            )
        return {str(item["id"]) for item in results}

    def list_owner_point_ids(self, owner_id: str, page_size: int = 256) -> Set[str]:
        """Lists every point one owner holds, by a filtered scroll on ``owner_id``.

        Args:
            owner_id: The owning decision's 64-hex node id.
            page_size: Maximum points fetched per scroll page.

        Returns:
            The owner's point-id strings.

        Raises:
            VectorStoreUnreachableError: If Qdrant did not answer.
            CollectionMissingError: If the collection does not exist.
            VectorStoreError: On another non-200 or a 200 of the wrong shape.
        """
        return _scroll_ids(
            self.base_url,
            self.collection,
            page_size=page_size,
            payload_filter={
                "must": [{"key": _OWNER_KEY, "match": {"value": owner_id}}]
            },
        )

    def list_point_ids(self, page_size: int = 256) -> Set[str]:
        """Lists every point in the collection.

        Args:
            page_size: Maximum points fetched per scroll page.

        Returns:
            Every point-id string in the collection.

        Raises:
            VectorStoreUnreachableError: If Qdrant did not answer.
            CollectionMissingError: If the collection does not exist.
            VectorStoreError: On another non-200 or a 200 of the wrong shape.
        """
        return _scroll_ids(self.base_url, self.collection, page_size=page_size)

    def list_point_owners(self, page_size: int = 256) -> Dict[str, Optional[str]]:
        """Lists every point in the collection with the owner its payload names.

        One whole scroll taking back the ``owner_id`` key only, so a mass drain
        attributes every held item to its owner without a call per owner.

        Args:
            page_size: Maximum points fetched per scroll page.

        Returns:
            ``{point_id: owner_id}``; the owner is None when the payload carries no
            non-empty string there (not a fault; such a point belongs to no owner).

        Raises:
            VectorStoreUnreachableError: If Qdrant did not answer.
            CollectionMissingError: If the collection does not exist.
            VectorStoreError: On another non-200 or a 200 of the wrong shape.
        """
        owners: Dict[str, Optional[str]] = {}
        for point in _scroll_points(
            self.base_url, self.collection, page_size=page_size,
            with_payload=[_OWNER_KEY],
        ):
            payload = point.get("payload")
            owner = payload.get(_OWNER_KEY) if isinstance(payload, dict) else None
            owners[str(point["id"])] = owner if isinstance(owner, str) and owner else None
        return owners

    def upsert_points(
        self, points: Sequence[RejectionPoint], *, model: str, may_create: bool
    ) -> None:
        """Writes items, each confirmed ``completed``, creating the collection if licensed.

        Points go in chunks of ``_REJECTION_UPSERT_CHUNK``, each a ``wait=true``
        request whose success is a 200 reading ``status: "completed"``; an
        ``acknowledged`` answer confirms nothing and raises. A chunk that fails
        raises, and chunks already confirmed stay written (their ids make a re-send
        idempotent). Ids are written exactly as given, never re-hashed.

        ``model`` and ``may_create`` are required and keyword-only: every write
        declares which model made its vectors and whether it may bring an absent
        collection into existence. On a 404, a licensed write creates the
        collection with its owner index and retries that chunk once; an unlicensed
        one raises.

        Args:
            points: The items to write. Empty sends no request.
            model: The embedding model id the vectors came from, recorded on every
                point.
            may_create: Whether this write may create an absent collection.

        Raises:
            ValueError: Before any request, for a point whose ``owner_id`` is not a
                non-empty string, or an empty ``model``.
            VectorStoreUnreachableError: If Qdrant did not answer.
            CollectionMissingError: If the collection is absent and this write may
                not create it, or is still absent after the create and retry.
            VectorStoreError: If Qdrant refused or did not confirm a chunk, or the
                creation failed.
        """
        if not isinstance(model, str) or not model:
            raise ValueError("A rejection point's model must be a non-empty string.")
        for point in points:
            if not isinstance(point.owner_id, str) or not point.owner_id:
                raise ValueError(
                    f"Rejection point {point.point_id!r} has no owner: owner_id "
                    "must be a non-empty string."
                )
        for start in range(0, len(points), _REJECTION_UPSERT_CHUNK):
            chunk = points[start:start + _REJECTION_UPSERT_CHUNK]
            body = {
                "points": [
                    {
                        "id": point.point_id,
                        "vector": point.vector,
                        "payload": {_OWNER_KEY: point.owner_id, "model": model},
                    }
                    for point in chunk
                ]
            }
            resp = self._put_points(body)
            if _collection_is_missing(resp):
                if not may_create:
                    raise CollectionMissingError(
                        f"Qdrant collection '{self.collection}' does not exist, and "
                        "this write may not create it.",
                        collection=self.collection,
                    )
                self._create_collection()
                resp = self._put_points(body)
            if resp.status_code != 200:
                _raise_response_error(
                    resp, operation="rejection upsert", collection=self.collection
                )
            result = _decode_result(resp, operation="rejection upsert")
            status = result.get("status") if isinstance(result, dict) else None
            if status != "completed":
                raise VectorStoreError(
                    "Qdrant rejection upsert was not confirmed (expected result "
                    f"status 'completed', got {status!r})."
                )

    def _put_points(self, body: Dict[str, Any]) -> "requests.Response":
        """Sends one ``wait=true`` upsert request, typing a transport fault."""
        try:
            return requests.put(
                self._points_url("?wait=true"),
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
        except requests.RequestException as e:
            raise VectorStoreUnreachableError(
                f"Qdrant connection error during rejection upsert: {str(e)}"
            ) from e

    def _create_collection(self) -> None:
        """Creates the collection, then its keyword index on ``owner_id``.

        Reached only from a licensed upsert's 404. The collection PUT must answer
        200 with ``result: true``, or 409: another writer's create won the race
        between the 404 and this PUT, so the collection exists and this goes on.
        The index PUT carries ``wait=true`` and must answer 200 with ``status:
        "completed"`` (it does over an index that already exists, so the race's
        loser re-declares it harmlessly). Without the index every owner scroll scans
        the whole collection.

        Residual, stated: if the index PUT fails, this raises and the collection
        stays, unindexed. That is still correct (owner scrolls work, only slower),
        and a later write sees no 404, so nothing here retries the index. No verb
        deletes a collection.

        Raises:
            VectorStoreUnreachableError: If Qdrant did not answer either request.
            VectorStoreError: If either request was refused or not confirmed (any
                non-200 but the collection PUT's 409).
        """
        collection_url = f"{self.base_url}/collections/{self.collection}"
        try:
            resp = requests.put(
                collection_url,
                json={"vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"}},
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
        except requests.RequestException as e:
            raise VectorStoreUnreachableError(
                f"Qdrant connection error creating the rejection collection: {str(e)}"
            ) from e
        if resp.status_code == 409:
            pass  # A racing create won: the collection exists. Declare the index.
        elif resp.status_code != 200:
            raise VectorStoreError(
                f"Failed to create Qdrant collection '{self.collection}' "
                f"(HTTP {resp.status_code}): {resp.text}"
            )
        elif _decode_result(resp, operation="rejection collection create") is not True:
            raise VectorStoreError(
                f"Qdrant did not confirm creating collection '{self.collection}'."
            )
        try:
            resp = requests.put(
                f"{collection_url}/index?wait=true",
                json={"field_name": _OWNER_KEY, "field_schema": "keyword"},
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
        except requests.RequestException as e:
            raise VectorStoreUnreachableError(
                f"Qdrant connection error creating the owner index: {str(e)}"
            ) from e
        if resp.status_code != 200:
            raise VectorStoreError(
                f"Failed to create the owner index on '{self.collection}' "
                f"(HTTP {resp.status_code}): {resp.text}"
            )
        result = _decode_result(resp, operation="owner index create")
        status = result.get("status") if isinstance(result, dict) else None
        if status != "completed":
            raise VectorStoreError(
                "Qdrant owner index creation was not confirmed (expected result "
                f"status 'completed', got {status!r})."
            )

    def delete_points(self, point_ids: Sequence[str]) -> None:
        """Deletes exactly these items by their ids, under 3c's delete contract.

        The body is shared with :meth:`QdrantVectorStore.delete_points`, which
        documents the contract: ids as listed (decimal strings sent as integers),
        ``wait=true`` and ``completed``, and the three typed errors.

        Args:
            point_ids: The point ids to delete. Empty sends no request.

        Raises:
            VectorStoreUnreachableError: If Qdrant did not answer.
            CollectionMissingError: If the collection does not exist.
            VectorStoreError: If Qdrant refused or did not confirm.
        """
        _delete_points(self.base_url, self.collection, point_ids)


class PointPartition(NamedTuple):
    """A listing's non-active points, split by whether their node is in the graph.

    Attributes:
        retired_node: Listed ids whose node is in the graph but not in its active
            set — history, kept.
        nodeless: Listed ids no graph node maps to — garbage, collected by
            ``mitos reconcile``.
    """

    retired_node: FrozenSet[str]
    nodeless: FrozenSet[str]


def partition_points(present: AbstractSet[str], *, node_ids: AbstractSet[str],
                     active_ids: AbstractSet[str]) -> PointPartition:
    """Splits a listing's non-active points into history and garbage.

    The one partition both ``mitos status`` and ``mitos reconcile`` apply, so the
    heal clears its own warning by construction. Pure set arithmetic over three id
    sets: each node id is mapped once through :func:`hash_to_uuid` and compared with
    the listed strings exactly as the index returned them (Qdrant lists its UUIDs
    lowercase and hyphenated, which is ``hash_to_uuid``'s form).

    History requires node membership, so an active id the node read lacks (a
    rebuild swapped the graph between the caller's two reads) cannot make its point
    history: that point is node-less, which is true of the graph now. The two
    buckets therefore always sum to ``|present − uuid(active ∩ nodes)|``, and a
    caller that reports that count beside them must derive it with the same
    intersection (``status`` does).

    **What "retired node" holds:** every graph node outside the active set —
    retired decisions (superseded or corrected) **and killed open questions**. A
    superseded or corrected open question is outside the decision index's expected
    set, so ``reconcile`` never re-seeds its point; but its node is still in the
    graph, so its point is history and is never deleted either. The only garbage is
    a point whose node is not in the graph (ADR
    ``reconcile-heals-every-decision-vector-and-collects-only-node-less-points``).

    Args:
        present: The listed point ids.
        node_ids: Every node id in the graph (64-hex), any kind, any state.
        active_ids: The active node ids (active decisions plus open questions).

    Returns:
        The partition of ``present`` minus the active points.
    """
    node_points = {hash_to_uuid(node_id) for node_id in node_ids}
    active_points = {hash_to_uuid(node_id) for node_id in active_ids}
    nodeless = frozenset(pid for pid in present if pid not in node_points)
    retired_node = frozenset(
        pid for pid in present if pid in node_points and pid not in active_points
    )
    return PointPartition(retired_node=retired_node, nodeless=nodeless)
