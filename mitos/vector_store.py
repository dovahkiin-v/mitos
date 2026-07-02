"""Qdrant-backed vector store for Mitos.

This module implements the vector store pipeline (D) using the Qdrant REST API
directly, reducing dependency bloat and ensuring maximum interoperability.
"""

import requests
import json
from typing import List, Dict, Any, Set
from mitos.errors import VectorStoreError
from mitos.models import EMBEDDING_DIM

def hash_to_uuid(sha256_hex: str) -> str:
    """Converts a 64-character SHA-256 hex string deterministically into a UUID format.

    Args:
        sha256_hex: A hex string of length 64.

    Returns:
        A 36-character standard UUID string.
    """
    sha = sha256_hex.lower()
    return f"{sha[:8]}-{sha[8:12]}-{sha[12:16]}-{sha[16:20]}-{sha[20:32]}"


class QdrantVectorStore:
    """REST client for Qdrant vector store managing points and semantic queries."""

    def __init__(self, qdrant_url: str, collection_name: str = "mitos") -> None:
        self.base_url = qdrant_url.rstrip("/")
        self.collection = collection_name
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Verifies if the collection exists, creating it with Cosine configuration if missing."""
        check_url = f"{self.base_url}/collections/{self.collection}"
        try:
            resp = requests.get(check_url, timeout=5)
            if resp.status_code == 200:
                # Collection exists
                return

            if resp.status_code == 404:
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

    def upsert(self, point_id: str, vector: List[float], payload: Dict[str, Any]) -> None:
        """Upserts a single point into Qdrant using the deterministic UUID mapping.

        Args:
            point_id: The SHA-256 node ID.
            vector: The embedding vector values.
            payload: Node metadata {slug, scope, state, kind, embedding_text}.
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
        
        try:
            resp = requests.put(
                upsert_url,
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            if resp.status_code != 200:
                raise VectorStoreError(f"Qdrant upsert failed: {resp.text}")
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
                raise VectorStoreError(f"Qdrant query failed: {resp.text}")

            results = resp.json().get("result", [])

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

        Enumerates the collection's actual point ids in one bounded scan (no
        per-node existence probes), so a caller can diff the graph's node set
        against what Qdrant really holds. Payloads and vectors are excluded from
        the response to keep each page cheap.

        Args:
            page_size: Maximum points fetched per scroll page.

        Returns:
            The set of point-id strings (the ``hash_to_uuid`` UUIDs) in the collection.

        Raises:
            VectorStoreError: If Qdrant is unreachable or returns a non-200 status.
        """
        scroll_url = f"{self.base_url}/collections/{self.collection}/points/scroll"
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
                    raise VectorStoreError(f"Qdrant scroll failed: {resp.text}")

                result = resp.json().get("result", {}) or {}
                for point in result.get("points", []):
                    ids.add(str(point["id"]))

                offset = result.get("next_page_offset")
                if offset is None:
                    break
            return ids
        except requests.RequestException as e:
            raise VectorStoreError(f"Qdrant scroll connection error: {str(e)}")

    def delete_point(self, point_id: str) -> None:
        """Deletes a point from Qdrant by its SHA-256 node ID."""
        uuid_id = hash_to_uuid(point_id)
        delete_url = f"{self.base_url}/collections/{self.collection}/points/delete"
        body = {"points": [uuid_id]}
        try:
            requests.post(delete_url, json=body, timeout=5)
        except Exception:
            pass
