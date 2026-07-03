"""Surface-specific tests for the Conflict finding renderer (Phase 2b, ``candidate_payload``).

``candidate_payload`` is a new decision-read surface, so it must obey the project rule
"every decision-read surface stamps modifiers" (CLAUDE.md): an amended-but-active
candidate must carry its ``amended_by`` stamp onto the surfaced finding, or the graph's
edge knowledge is lost and the axiom reads as the final word (the "amended axioms read as
live" trap). Two layers:

* A pure shape test (minimal in-memory ``node``) pins the Letter core + ``score`` + the
  ``brief`` contract.
* A real-temp-``GraphStore`` test proves the stamp actually rides through, mirroring
  ``tests/test_modifier_surfacing.py``'s ``offline`` + ``commit_parsed_entry`` idiom
  (keyless, deterministic — ``commit_parsed_entry`` never embeds).
"""

import shutil
import tempfile
from typing import Any, Dict, Iterator

import pytest

from mitos.cli import cmd_init
from mitos.config import MitosConfig
from mitos.conflict import Candidate, candidate_payload
from mitos.parser import ParsedEntry
from mitos.store import GraphStore


# --------------------------------------------------------------------------- #
# Test 10 — candidate_payload shape + score (pure, minimal node)
# --------------------------------------------------------------------------- #

def _node(**overrides: Any) -> Dict[str, Any]:
    """A minimal hydrated reader dict (letter_payload reads these four keys)."""
    node = {
        "slug": "cache-ttl-fixed",
        "core_axiom": "TTL is fixed at 60s.",
        "scope": ["cache"],
        "rejected_paths": "Rejected a per-key TTL.",
    }
    node.update(overrides)
    return node


def test_candidate_payload_shape_carries_letter_core_and_score() -> None:
    """A survivor renders to {slug, axiom, scope, score, rejected_paths}; axiom from core_axiom."""
    cand = Candidate(slug="cache-ttl-fixed", score=0.78, node=_node(), state="active")
    payload = candidate_payload(cand)
    assert payload["slug"] == "cache-ttl-fixed"
    assert payload["axiom"] == "TTL is fixed at 60s."   # from node["core_axiom"]
    assert payload["scope"] == ["cache"]
    assert payload["score"] == 0.78
    assert payload["rejected_paths"] == "Rejected a per-key TTL."


def test_candidate_payload_brief_drops_rejected_paths() -> None:
    """brief=True omits rejected_paths (the M4 opt-out) and nothing else."""
    cand = Candidate(slug="cache-ttl-fixed", score=0.78, node=_node(), state="active")
    payload = candidate_payload(cand, brief=True)
    assert "rejected_paths" not in payload
    assert payload["slug"] == "cache-ttl-fixed"
    assert payload["score"] == 0.78


def test_candidate_payload_unmodified_node_has_no_stamp_keys() -> None:
    """An unmodified candidate's node carries no modifier keys → payload has none.

    ``_stamp_modifiers`` adds a reverse key only when a modifier exists; the copy is
    conditional (``if key in node``), so blind indexing never KeyErrors on this common case.
    """
    cand = Candidate(slug="cache-ttl-fixed", score=0.78, node=_node(), state="active")
    payload = candidate_payload(cand)
    for key in ("superseded_by", "amended_by", "narrowed_by", "corrected_by"):
        assert key not in payload


# --------------------------------------------------------------------------- #
# Fixtures for the real-store modifier test (mirror test_conflict_gather.py)
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key, no reachable service — commit_parsed_entry stays keyless/offline."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def store() -> Iterator[GraphStore]:
    """A fresh, initialized, empty on-disk graph store (no network, no keys)."""
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    cmd_init(config)
    yield GraphStore(config.db_path)
    shutil.rmtree(tmp, ignore_errors=True)


def _seed(store: GraphStore, slug: str, axiom: str, **relationships: Any) -> None:
    """Commits one live decision (keyless — commit_parsed_entry never embeds)."""
    entry = ParsedEntry("decision", slug, 1, 10)
    entry.axiom = axiom
    entry.rejected_paths = "Rejected the obvious alternative."
    entry.scope = ["cache"]
    for field, targets in relationships.items():
        setattr(entry, field, targets)
    store.commit_parsed_entry(entry)


# --------------------------------------------------------------------------- #
# Test 11 — surface-specific modifier stamp rides onto the finding (real store)
# --------------------------------------------------------------------------- #

def test_amended_but_active_candidate_surfaces_amended_by(store: GraphStore) -> None:
    """Commit A, then B amends A → candidate_payload for A carries amended_by == ['b'].

    ``amends`` (not ``supersedes``/``corrects``) is deliberate: it is NOT a kill-edge, so
    A stays computed-``active`` and ``get_node_by_slug('a')`` still resolves — yet the node
    carries the ``amended_by`` stamp. This proves an amended-but-active candidate never
    reads as the final word on the surfaced finding (the project's oldest AX lesson).
    """
    _seed(store, "a", "Cache uses an LRU eviction policy.")
    _seed(store, "b", "Cache LRU eviction is bounded to 10k entries.", amends=["a"])

    node = store.get_node_by_slug("a")  # active-scoped; carries amended_by via _stamp_modifiers
    assert node is not None
    cand = Candidate(slug="a", score=0.7, node=node, state="active")

    payload = candidate_payload(cand)
    assert payload["amended_by"] == ["b"]
    # The Letter core still renders alongside the stamp.
    assert payload["slug"] == "a"
    assert payload["score"] == 0.7

    # brief governs only rejected_paths; the stamp is copied AFTER letter_payload, so
    # brief=True still carries amended_by (a brief finding never reads as the final word).
    brief_payload = candidate_payload(cand, brief=True)
    assert brief_payload["amended_by"] == ["b"]
    assert "rejected_paths" not in brief_payload
