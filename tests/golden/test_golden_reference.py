"""Golden-dataset harness — Layer A (deterministic, no LLM/Qdrant, bare-CI safe).

Commits the frozen `decisions.reference.md` corpus and asserts the emergent graph
facts against `oracle.reference.json`. This is the regression gate for behaviour that
unit tests only cover piecemeal on toy inputs: computed state across a real edge web,
kill vs non-kill edge semantics, modifier stamping, lineage, active view, and
content-hash stability.

Regenerate the oracle after an INTENTIONAL corpus change (review the diff!):
    python tests/golden/_harness.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _harness import build_reference_graph, build_snapshot_in_tmp, load_oracle, snapshot  # noqa: E402


def test_reference_snapshot_matches_oracle(tmp_path):
    """The whole computed snapshot equals the frozen oracle (per-node diff on failure)."""
    store = build_reference_graph(str(tmp_path / "graph.sqlite"))
    got = snapshot(store)
    oracle = load_oracle()

    # Per-node diff first — a failure names the exact slug + field, not a wall of JSON.
    got_nodes, exp_nodes = got["nodes"], oracle["nodes"]
    assert set(got_nodes) == set(exp_nodes), (
        f"node set drift: missing={set(exp_nodes) - set(got_nodes)}, "
        f"extra={set(got_nodes) - set(exp_nodes)}"
    )
    for slug, exp in exp_nodes.items():
        assert got_nodes[slug] == exp, f"'{slug}' drifted from oracle:\n  got={got_nodes[slug]}\n  exp={exp}"

    assert got["active_view"] == oracle["active_view"]


def test_reference_build_is_deterministic():
    """Two independent builds yield identical content-hash ids (hash + rebuild determinism)."""
    a = build_snapshot_in_tmp()
    b = build_snapshot_in_tmp()
    assert a == b


def test_kill_edges_retire_targets_non_kill_edges_do_not(tmp_path):
    """Marquee invariant, spelled out so a regression reads clearly:

    - supersedes / corrects are KILL edges → target leaves the active view.
    - amends / narrows are NOT kill edges → target stays active but carries a stamp
      (the 'amended axiom reads as live' trap).
    """
    store = build_reference_graph(str(tmp_path / "graph.sqlite"))
    snap = snapshot(store)
    n = snap["nodes"]

    # Killed by a kill-edge → not active, stamped by the killer.
    assert n["harbor-auth-jwt-v1"]["state"] == "superseded"
    assert n["harbor-auth-jwt-v2"]["state"] == "superseded"
    assert n["harbor-legacy-ftp-gateway"]["state"] == "corrected"
    for dead in ("harbor-auth-jwt-v1", "harbor-auth-jwt-v2", "harbor-legacy-ftp-gateway"):
        assert dead not in snap["active_view"]

    # Amended / narrowed but STILL ACTIVE — the trap. Active AND stamped.
    enc = n["harbor-blob-encryption-at-rest"]
    assert enc["state"] == "active"
    assert enc["modifiers"].get("amended_by") == ["harbor-blob-key-rotation-quarterly"]
    assert "harbor-blob-encryption-at-rest" in snap["active_view"]

    rl = n["harbor-api-rate-limit"]
    assert rl["state"] == "active"
    assert rl["modifiers"].get("narrowed_by") == ["harbor-premium-exempt-rate-limit"]
    assert "harbor-api-rate-limit" in snap["active_view"]


def test_non_kill_edges_and_scope_semantics(tmp_path):
    """contradicts / amends(multi) / narrows(global↔scoped) / weak edges — all non-kill."""
    store = build_reference_graph(str(tmp_path / "graph.sqlite"))
    snap = snapshot(store)
    n, edges = snap["nodes"], snap["edges"]

    # contradicts is non-kill: BOTH endpoints stay active, edge is recorded.
    assert n["harbor-sync-last-write-wins"]["state"] == "active"
    assert n["harbor-sync-crdt-merge"]["state"] == "active"
    assert ["harbor-sync-crdt-merge", "contradicts", "harbor-sync-last-write-wins"] in edges

    # Multi-valued amends: one entry amends two targets → two edges, both targets stamped.
    assert ["harbor-observability-otel", "amends", "harbor-structured-logging"] in edges
    assert ["harbor-observability-otel", "amends", "harbor-prometheus-metrics"] in edges
    assert n["harbor-structured-logging"]["modifiers"].get("amended_by") == ["harbor-observability-otel"]
    assert n["harbor-prometheus-metrics"]["modifiers"].get("amended_by") == ["harbor-observability-otel"]

    # Global↔scoped narrows: a scoped exception narrows an unscoped global rule; both active.
    glob = n["harbor-all-endpoints-authenticated"]
    assert glob["scope"] == []  # global = zero scope tags (MI-9: absent, not "")
    assert glob["state"] == "active"
    assert glob["modifiers"].get("narrowed_by") == ["harbor-health-endpoint-public"]
    assert n["harbor-health-endpoint-public"]["scope"] == ["api"]

    # Weak edges commit and do not retire either endpoint.
    assert ["harbor-api-versioning", "cites", "harbor-storage-is-sqlite"] in edges
    assert ["harbor-api-versioning", "depends_on", "harbor-auth-sessions-v3"] in edges


def test_non_ascii_axiom_round_trips(tmp_path):
    """P9 language sovereignty: a Lithuanian axiom parses, commits, and hashes intact."""
    store = build_reference_graph(str(tmp_path / "graph.sqlite"))
    node = store.get_node_by_slug("harbor-duomenu-saugojimas-lietuvoje")
    assert node is not None
    assert "Lietuvoje" in node["core_axiom"]  # non-ASCII content preserved verbatim
