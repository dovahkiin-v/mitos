"""Launches ``mitos serve`` with only its two provider constructors bound to fakes.

A stdio row that walks the receipts needs a server that embeds and ranks, and
constraint 15 says every row this vision adds runs offline: no key, no Qdrant. A
real ``serve`` child cannot take an in-process monkeypatch, so this file is the
child's entry point instead of ``-m mitos.cli serve``. It does three things and
gets out of the way:

1. imports ``mitos`` from this checkout (``sys.executable`` is the venv
   interpreter, exactly as ``mcp_harness.SERVE_ARGS`` relies on);
2. rebinds ``GeminiEmbeddingProvider`` and ``QdrantVectorStore`` at the two
   construct sites the MCP path reaches — ``mitos.mcp_server``
   (``get_workspace_components``) and ``mitos.sync`` (``MitosSyncManager``). Both
   modules did ``from … import`` at their own scope, so rebinding the defining
   modules would change nothing. The other two construct sites are left alone on
   purpose: ``mitos.importer`` (the import verb) and ``mitos.cli``'s
   ``_build_check_substrate`` (the check verb) are not reachable from a tool;
3. sets ``sys.argv`` to ``mitos serve`` and calls the same ``cli.main()`` that
   ``python -m mitos.cli serve`` runs. Everything above the two constructors — the
   FastMCP subclass, the tool bodies, the payload assembly — is the branch's code.

Launch it with ``mitos_server(cwd=…, env=…, args=(str(LAUNCHER),))``. ADR
``mcp-e2e-rides-a-real-serve-subprocess-with-a-declared-environment`` and its
amendment record why this shape exists and why the harness itself stays mitos-free.

Both production construct sites wrap the constructors in ``except Exception: pass``,
so a fake that fails, or a binding that misses, degrades silently to lexical
recall. The fakes therefore implement exactly the methods the MCP path calls and
answer anything else loudly: an ``UNIMPLEMENTED`` line on stderr as well as the
raise, because production may swallow the raise. A row asserts that line absent.

``MITOS_FAKE_EMBED_FAULT`` (``auth``, ``quota``, ``transient`` or ``unknown``)
makes every ``get_embedding`` call fail the way the Gemini SDK does: an
``EmbeddingError`` carrying a provider body, chained from an SDK-shaped error with
the matching typed fields. The fault fires in the call, never the constructor: a
constructor raise is swallowed at the construct site, and the read would then
report no provider at all instead of the classified cause.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

#: The environment name that selects a provider fault for the whole child.
FAULT_ENV = "MITOS_FAKE_EMBED_FAULT"

#: Written to stderr once both sites are bound; a row reads it back.
BOUND_MARKER = "[fake-providers] bound at mitos.mcp_server and mitos.sync"

#: Prefixes the line a fake writes when production calls something it lacks.
UNIMPLEMENTED_MARKER = "[fake-providers] UNIMPLEMENTED"

#: Text inside every faulted provider body, so a row can assert it never surfaces.
FAULT_BODY_SENTINEL = "FAKE-PROVIDER-BODY"

#: The typed fields each fault class carries on its SDK-shaped cause. `quota` and
#: `auth` mirror the codes Gemini returns; `unknown` carries nothing typed at all.
FAULT_FIELDS: Dict[str, Dict[str, Any]] = {
    "auth": {"code": 401, "status": "UNAUTHENTICATED"},
    "quota": {"code": 429, "status": "RESOURCE_EXHAUSTED"},
    "transient": {"code": 503, "status": "UNAVAILABLE"},
    "unknown": {},
}

#: Vector width. Wide enough that the fixture vocabularies rarely collide.
DIMENSION = 1024

_TOKEN = re.compile(r"[a-z0-9]+")


def fake_vector(text: str) -> List[float]:
    """Embeds text as a normalised, hashed bag of lower-cased tokens.

    Deterministic across processes (sha256, never ``hash()``), so the test that
    chooses fixture texts and the child that embeds them agree on every score.
    Axioms that share most of their words score high; unrelated ones score low.

    Args:
        text: The text to embed.

    Returns:
        A unit vector of :data:`DIMENSION` floats (all zeros for a text with no
        tokens).
    """
    vector = [0.0] * DIMENSION
    for token, count in Counter(_TOKEN.findall(text.lower())).items():
        index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big")
        vector[index % DIMENSION] += count
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def cosine(left: List[float], right: List[float]) -> float:
    """Returns the cosine similarity of two vectors (0.0 when either is zero)."""
    dot = sum(a * b for a, b in zip(left, right))
    norms = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norms if norms else 0.0


def _unimplemented(owner: str, name: str) -> NotImplementedError:
    print(f"{UNIMPLEMENTED_MARKER}: {owner}.{name}", file=sys.stderr, flush=True)
    return NotImplementedError(f"{owner} does not implement {name!r}")


class _SdkLikeError(Exception):
    """Stands in for ``google.genai.errors.APIError``: the fields the classifier reads."""

    def __init__(self, message: str, **fields: Any) -> None:
        super().__init__(message)
        for name, value in fields.items():
            setattr(self, name, value)


class FakeEmbeddingProvider:
    """Answers ``get_embedding`` the way the Gemini provider does, offline.

    Takes the real constructor's arguments and ignores the key: the real one
    refuses a missing key, which would make a keyless child degrade.
    """

    def __init__(self, cache_path: str, *, api_key: Optional[str] = None,
                 model_id: Optional[str] = None) -> None:
        self.cache_path = cache_path
        self.model_id = model_id
        self._fault = os.environ.get(FAULT_ENV) or None
        if self._fault is not None and self._fault not in FAULT_FIELDS:
            raise SystemExit(f"{FAULT_ENV}={self._fault!r} is not one of {sorted(FAULT_FIELDS)}")

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """Returns the fake vector for ``text``, or raises the planted fault."""
        if self._fault is not None:
            from mitos.errors import EmbeddingError

            body = f'{FAULT_BODY_SENTINEL} {{"error": {{"message": "planted {self._fault}"}}}}'
            raise EmbeddingError(f"Gemini embedding API call failed: {body}") from _SdkLikeError(
                body, **FAULT_FIELDS[self._fault])
        return fake_vector(text)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        raise _unimplemented("FakeEmbeddingProvider", name)


class FakeVectorStore:
    """Keeps points in a process-lifetime dict, one per collection.

    Class-level on purpose: every tool call constructs a fresh store, so the
    points must outlive the instance, as they outlive a real ``QdrantVectorStore``.
    A collection that has never been written is absent, and reads and uncovered
    writes on it raise ``CollectionMissingError``, as the real one does on a 404.
    """

    _collections: Dict[str, Dict[str, Tuple[List[float], Dict[str, Any]]]] = {}

    def __init__(self, qdrant_url: str, collection_name: str) -> None:
        self.qdrant_url = qdrant_url
        self.collection = collection_name

    def upsert(self, point_id: str, vector: List[float], payload: Dict[str, Any],
               *, may_create: bool) -> None:
        """Stores one point; creates the collection only when the write may."""
        from mitos.errors import CollectionMissingError

        if self.collection not in self._collections:
            if not may_create:
                raise CollectionMissingError(
                    f"collection {self.collection!r} is missing", collection=self.collection)
            self._collections[self.collection] = {}
        self._collections[self.collection][point_id] = (list(vector), dict(payload))

    def query(self, vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Returns the nearest points in the real client's shape, best first."""
        from mitos.errors import CollectionMissingError

        points = self._collections.get(self.collection)
        if points is None:
            raise CollectionMissingError(
                f"collection {self.collection!r} is missing", collection=self.collection)
        ranked = sorted(
            ((cosine(vector, stored), payload) for stored, payload in points.values()),
            key=lambda pair: pair[0], reverse=True,
        )
        return [
            {
                "slug": payload.get("slug"),
                "scope": payload.get("scope", []),
                "state": payload.get("state"),
                "kind": payload.get("kind"),
                "embedding_text": payload.get("embedding_text"),
                "score": score,
            }
            for score, payload in ranked[:limit]
        ]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        raise _unimplemented("FakeVectorStore", name)


def bind() -> None:
    """Rebinds both constructors at both MCP-reachable construct sites."""
    import mitos.mcp_server
    import mitos.sync

    for module in (mitos.mcp_server, mitos.sync):
        module.GeminiEmbeddingProvider = FakeEmbeddingProvider
        module.QdrantVectorStore = FakeVectorStore
    print(BOUND_MARKER, file=sys.stderr, flush=True)


def main() -> None:
    """Binds the fakes, then enters the ``mitos serve`` path ``-m mitos.cli`` takes."""
    bind()
    from mitos.cli import main as cli_main

    sys.argv = ["mitos", "serve"]
    cli_main()


if __name__ == "__main__":
    main()
