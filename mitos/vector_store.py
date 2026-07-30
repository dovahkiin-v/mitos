"""Qdrant-backed vector store for Mitos.

This module implements the vector store pipeline (D) using the Qdrant REST API
directly, reducing dependency bloat and ensuring maximum interoperability.

Three contracts hold across every member here. **Reading never writes:** neither
constructing a store nor querying/scrolling one can bring a collection into
existence — creation attaches to :meth:`QdrantVectorStore.upsert`, lazily, on a
404. **Missing is not unreachable:** an absent collection raises the typed
:class:`~mitos.errors.CollectionMissingError`, which every surface words as the
recoverable state it is (``mitos reconcile``) rather than as an outage. And **not
every write may create:** the upsert takes a required ``may_create`` declaration, so
a write covering only itself leaves an absent collection absent rather than minting a
one-point index over a populated corpus. Absence is a signal several nets key on; a
single write must not be able to spend it.
"""

import requests
import json
from typing import List, Dict, Any, Set
from mitos.errors import CollectionMissingError, VectorStoreError
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


def scroll_point_ids(base_url: str, collection: str, page_size: int = 256) -> Set[str]:
    """Enumerates every point id in a collection via a no-create paginated scroll.

    A module-level read path that talks to Qdrant directly without constructing a
    :class:`QdrantVectorStore` — kept so a read-only probe (e.g. ``mitos status``)
    can diff the graph against Qdrant's actual points without needing a store at
    all. Reads create nothing: an absent collection is *reported*, never
    materialized. Payloads and vectors are excluded from the response to keep each
    page cheap; the scan is bounded to one pass over the collection (no per-node
    existence probes).

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
    scroll_url = f"{base_url.rstrip('/')}/collections/{collection}/points/scroll"
    ids: Set[str] = set()
    offset: Any = None
    try:
        while True:
            body: Dict[str, Any] = {
                "limit": page_size,
                "with_payload": False,
                "with_vector": False,
            }
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
            for point in points:
                ids.add(str(point["id"]))

            offset = result.get("next_page_offset")
            if offset is None:
                break
        return ids
    except requests.RequestException as e:
        raise VectorStoreError(f"Qdrant scroll connection error: {str(e)}")


class QdrantVectorStore:
    """REST client for Qdrant vector store managing points and semantic queries."""

    def __init__(self, qdrant_url: str, collection_name: str = "mitos") -> None:
        """Binds the store to a Qdrant endpoint and collection — **no network I/O**.

        Construction is pure: it dispatches zero HTTP requests and therefore cannot
        create, probe, or fail. Creation attaches to the *write* instead (see
        :meth:`upsert`), which is what makes a read structurally incapable of
        materializing the collection it was only meant to look at — one
        constructor serves every reader and every writer, so the policy cannot
        live here.

        Args:
            qdrant_url: The Qdrant REST endpoint (trailing slash tolerated).
            collection_name: The project's collection.
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
            may_create: Whether this write covers the workspace's active set and may
                therefore bring an absent collection into existence. Callers derive it
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
                # Narrowed to writes that COVER the workspace's active set. A write
                # covering only itself defers here: creating would leave a collection
                # holding one point out of N, and from the next write on every net
                # keyed on absence would report "checked, clean" over an index that
                # was never checked. Deferral is cheap — the commit already wrote the
                # outbox row, so the work is queued, not lost.
                if not may_create:
                    raise CollectionMissingError(
                        f"Qdrant collection '{self.collection}' does not exist, and "
                        f"this write does not cover the workspace's active set — "
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
    ) -> List[Dict[str, Any]]:
        """Queries Qdrant for the semantically nearest vectors.

        Semantic recall is deliberately scope-blind: a decision is the precedent on
        its subject regardless of which scope drawer it was filed under, and gating
        the search by a caller-guessed tag silently hides real precedent (the
        ``gemini-live`` vs ``live-voice`` drift). Scope is handled downstream as a
        discoverability hint in :mod:`mitos.recall`, never as a recall filter.

        Args:
            vector: The query embedding vector.
            limit: Maximum matches to return.

        Returns:
            A list of dictionary results with payload and scores.

        Raises:
            CollectionMissingError: If Qdrant is up but the collection does not
                exist. Reads never create it — absence is reported, and the
                calling surface decides whether that is a gap or simply the empty
                index of a project with nothing to index yet.
            VectorStoreError: If Qdrant is unreachable, returns another non-200
                status, or answers 200 with a body of the wrong shape.
        """
        search_url = f"{self.base_url}/collections/{self.collection}/points/search"

        body: Dict[str, Any] = {
            "vector": vector,
            "limit": limit,
            "with_payload": True
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
            if not isinstance(results, list) or any(
                not isinstance(item, dict) for item in results
            ):
                raise VectorStoreError(
                    "Qdrant query returned an unexpected result shape "
                    "(expected a list of objects)."
                )

            output = []
            for item in results:
                # Format to a standard output tuple
                payload = item.get("payload", {})
                score = item.get("score", 0.0)
                output.append({
                    "slug": payload.get("slug"),
                    "scope": payload.get("scope", []),
                    "state": payload.get("state"),
                    "kind": payload.get("kind"),
                    "embedding_text": payload.get("embedding_text"),
                    "score": score
                })

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

    def delete_point(self, point_id: str) -> None:
        """Deletes a point from Qdrant by its SHA-256 node ID."""
        uuid_id = hash_to_uuid(point_id)
        delete_url = f"{self.base_url}/collections/{self.collection}/points/delete"
        body = {"points": [uuid_id]}
        try:
            requests.post(delete_url, json=body, timeout=5)
        except Exception:
            pass
