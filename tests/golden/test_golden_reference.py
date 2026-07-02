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

import pytest

from mitos.errors import CommitError
from mitos.parser import parse_entry_stream
from mitos.store import GraphStore

sys.path.insert(0, os.path.dirname(__file__))
from _harness import build_reference_graph, build_snapshot_in_tmp, load_oracle, snapshot  # noqa: E402


def _commit_all(store, text, kind="decision"):
    for entry in parse_entry_stream(text, kind):
        store.commit_parsed_entry(entry)


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _expect_rejection(tmp_path, text, code, pre=None):
    """Commits `pre` (if any), then `text`; asserts a CommitError with FailureItem `code`."""
    store = GraphStore(str(tmp_path / "graph.sqlite"))
    if pre:
        _commit_all(store, pre)
    with pytest.raises(CommitError) as exc:
        _commit_all(store, text)
    codes = [it.code for it in exc.value.failure.items]
    assert code in codes, f"expected {code}, got {codes}"


_HDR = "#\n<!-- DO NOT MODIFY ABOVE THIS LINE -->\n<!-- BEGIN ENTRIES -->\n"


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


def test_cross_kind_resolves_and_oq_state(tmp_path):
    """OQ Stage-2 state: an OQ is 'resolved' iff a `resolves` edge points at it from an
    active decision; otherwise 'parked'. This is derived by oq_state_view, separate from
    node liveness (get_node_state)."""
    store = build_reference_graph(str(tmp_path / "graph.sqlite"))
    snap = snapshot(store)

    assert snap["oq_state"]["oq-harbor-backup-cadence"] == "resolved"
    assert snap["oq_state"]["oq-harbor-multiregion"] == "parked"
    assert ["harbor-backup-nightly", "resolves", "oq-harbor-backup-cadence"] in snap["edges"]
    # The resolving decision is an ordinary active decision.
    assert snap["nodes"]["harbor-backup-nightly"]["state"] == "active"


def test_non_ascii_axiom_round_trips(tmp_path):
    """P9 language sovereignty: a Lithuanian axiom parses, commits, and hashes intact."""
    store = build_reference_graph(str(tmp_path / "graph.sqlite"))
    node = store.get_node_by_slug("harbor-duomenu-saugojimas-lietuvoje")
    assert node is not None
    assert "Lietuvoje" in node["core_axiom"]  # non-ASCII content preserved verbatim


# --- Cluster 8: adversarial — the write path must REJECT these (commit-layer) ---

def test_reject_dangling_edge_to_uncommitted_target(tmp_path):
    """An edge to a target not yet in the graph is rejected (missing_target): edges point
    newer→older, so the cited entry must already exist. Guards the acyclic newer→older
    ordering the graph depends on."""
    text = (
        _HDR
        + "### harbor-bad-b\n**Decided:** B.\n**Rejected:** n/a.\n**Depends-On:** harbor-bad-a\n"
        + "### harbor-bad-a\n**Decided:** A.\n**Rejected:** n/a.\n**Depends-On:** harbor-bad-b\n"
    )
    _expect_rejection(tmp_path, text, "missing_target")


def test_reject_cross_kind_resolves(tmp_path):
    """`resolves` is decision→open_question only; pointing it at a decision is rejected
    (kind_constraint_violation)."""
    pre = _HDR + "### harbor-target-decision\n**Decided:** A.\n**Rejected:** n/a.\n"
    bad = _HDR + "### harbor-bad-resolver\n**Decided:** X.\n**Rejected:** n/a.\n**Resolves:** harbor-target-decision\n"
    _expect_rejection(tmp_path, bad, "kind_constraint_violation", pre=pre)


# --- Polish: every read surface stamps modifiers; the real rebuild reproduces the golden ---

def test_all_read_surfaces_stamp_modifiers(tmp_path):
    """CLAUDE.md rule: every decision-read surface stamps modifiers. An amended/narrowed-
    but-active node must carry its stamp through get_node_by_slug, get_active_decisions,
    get_decisions, and the Letter-mode query_letter alike — no surface reads it as final."""
    store = build_reference_graph(str(tmp_path / "graph.sqlite"))
    cases = [
        ("harbor-blob-encryption-at-rest", "amended_by", ["harbor-blob-key-rotation-quarterly"]),
        ("harbor-api-rate-limit", "narrowed_by", ["harbor-premium-exempt-rate-limit"]),
        ("harbor-all-endpoints-authenticated", "narrowed_by", ["harbor-health-endpoint-public"]),
    ]
    for slug, key, expected in cases:
        surfaces = {
            "get_node_by_slug": store.get_node_by_slug(slug),
            "get_active_decisions": next(d for d in store.get_active_decisions() if d["slug"] == slug),
            "get_decisions": next(d for d in store.get_decisions() if d["slug"] == slug),
            "query_letter": store.query_letter(slug=slug)[0],
        }
        for name, payload in surfaces.items():
            assert payload.get(key) == expected, f"{name} did not stamp {key} on {slug}: {payload.get(key)}"


def test_real_rebuild_reproduces_the_golden(tmp_path):
    """`mitos rebuild` — the real oldest-first replay + forward-ref fixpoint — reproduces
    the golden graph exactly (content-hash ids, state, edges, active view, OQ state). This
    cross-validates the harness's linear commit order AND pins rebuild determinism: a change
    that silently reshaped a rebuilt corpus would fail here."""
    from mitos.config import MitosConfig
    from mitos.cutover import default_aside_db_path, rebuild_and_gate

    from _harness import CORPUS_PATH, QUESTIONS_PATH, load_oracle

    config = MitosConfig(str(tmp_path))
    os.makedirs(config.mitos_dir, exist_ok=True)
    _write(config.decisions_file, open(CORPUS_PATH, encoding="utf-8").read())
    _write(config.questions_file, open(QUESTIONS_PATH, encoding="utf-8").read())

    result = rebuild_and_gate(config, aside_db_path=default_aside_db_path(config), strict=False)
    assert result.residual_casualties == []

    store = GraphStore(result.aside_db_path, read_only=True)
    assert snapshot(store) == load_oracle()
