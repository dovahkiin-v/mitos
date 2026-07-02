"""Shared machinery for the mitos golden-dataset harness (Layer A — deterministic).

Layer A commits the frozen reference corpus into a throwaway graph and asserts the
*emergent* graph facts — computed state, modifier stamps, lineage, active view, and
content-hash stability — against a frozen oracle. No embeddings, no Qdrant, no LLM:
it runs in bare CI and never flakes. See MITOS_GOLDEN_DATASET_SPEC for the design.
"""

import json
import os
import tempfile
from typing import Any, Dict, List

from mitos.parser import parse_entry_stream
from mitos.store import GraphStore

GOLDEN_DIR = os.path.dirname(__file__)
CORPUS_PATH = os.path.join(GOLDEN_DIR, "decisions.reference.md")
ORACLE_PATH = os.path.join(GOLDEN_DIR, "oracle.reference.json")


def build_reference_graph(db_path: str) -> GraphStore:
    """Parses the reference corpus and commits it into a fresh graph at db_path.

    Commits oldest-first (the corpus is authored newest-first, the human convention),
    mirroring `rebuild` replay so an edge's target is committed before the entry that
    references it — the corpus is authored with no forward references on purpose.

    Args:
        db_path: Filesystem path for the throwaway SQLite graph.

    Returns:
        The populated GraphStore.
    """
    text = open(CORPUS_PATH, encoding="utf-8").read()
    failures: List[Any] = []
    entries = parse_entry_stream(text, "decision", failures=failures)
    if failures:
        raise AssertionError(f"reference corpus failed to parse cleanly: {failures}")
    store = GraphStore(db_path)
    for entry in reversed(entries):  # oldest-first
        store.commit_parsed_entry(entry)
    return store


def snapshot(store: GraphStore) -> Dict[str, Any]:
    """Captures the deterministic graph facts the oracle compares against.

    Args:
        store: A populated GraphStore.

    Returns:
        A dict with per-node state/modifiers/lineage/id and the sorted active view.
    """
    nodes: Dict[str, Any] = {}
    id_to_slug: Dict[str, str] = {}
    for node in store.get_all_nodes():
        id_to_slug[node["id"]] = node["slug"]
    for node in store.get_all_nodes():
        nid = node["id"]
        nodes[node["slug"]] = {
            "id": nid,
            "state": store.get_node_state(nid),
            "scope": sorted(node.get("scope") or []),
            "modifiers": store.get_modifiers(nid),
            "lineage": [n.get("slug") for n in store.get_lineage(nid)],
        }
    # Typed edge set, id→slug mapped and sorted; created_at is dropped (it is a
    # non-deterministic application-supplied timestamp, MI-10).
    edges = sorted(
        [id_to_slug.get(e["source_id"], e["source_id"]),
         e["edge_type"],
         id_to_slug.get(e["target_id"], e["target_id"])]
        for e in store.get_edges()
    )
    active_view = sorted(d["slug"] for d in store.get_active_decisions())
    return {"nodes": nodes, "edges": edges, "active_view": active_view}


def build_snapshot_in_tmp() -> Dict[str, Any]:
    """Builds the reference graph in a temp dir and returns its snapshot."""
    tmp = tempfile.mkdtemp(prefix="mitos-golden-")
    store = build_reference_graph(os.path.join(tmp, "graph.sqlite"))
    return snapshot(store)


def load_oracle() -> Dict[str, Any]:
    """Loads the frozen expected-outcomes oracle."""
    with open(ORACLE_PATH, encoding="utf-8") as f:
        return json.load(f)


def write_oracle(snap: Dict[str, Any]) -> None:
    """Freezes a verified snapshot as the oracle (the --update-golden path).

    Only call this after eyeballing the snapshot for correctness — the oracle is the
    hand-verified ground truth, never a blind capture of whatever the tool emitted.
    """
    with open(ORACLE_PATH, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    # Regeneration entry point: python -m tests.golden._harness  (review the diff!)
    write_oracle(build_snapshot_in_tmp())
    print(f"wrote {ORACLE_PATH}")
