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

from mitos.errors import CommitError, PARSER_SLUG_TOO_LONG, ValidationError
from mitos.identity import SLUG_MAX_LEN
from mitos.parser import ParsedEntry, parse_entry_stream
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


def test_reject_overlong_slug_on_file_route(tmp_path):
    """A >100-char slug is rejected on the file route (mitos sync/rebuild/import), which
    used to bypass the guard that only the record path enforced. Two layers:
    the parser quarantines the entry (slug_too_long), and the store's identity-fence
    backstop raises on any direct commit that bypasses the parser."""
    long_slug = "harbor-" + "x" * (SLUG_MAX_LEN + 10)  # 117 chars
    text = _HDR + f"### {long_slug}\n**Decided:** X.\n**Rejected:** n/a.\n"

    # Parser (collector mode): the over-length entry is quarantined, not returned.
    failures = []
    returned = parse_entry_stream(text, "decision", failures=failures)
    assert long_slug not in [e.slug for e in returned]
    assert PARSER_SLUG_TOO_LONG in [item.code for f in failures for item in f.items]

    # Store backstop: a direct commit bypassing the parser is fenced.
    store = GraphStore(str(tmp_path / "graph.sqlite"))
    entry = ParsedEntry("decision", long_slug, 1, 3)
    entry.axiom, entry.rejected_paths = "X.", "n/a"
    with pytest.raises(ValidationError):
        store.commit_parsed_entry(entry)


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


def test_record_pause_surface_stamps_modifiers(tmp_path):
    """The record pause is a decision-read surface too: `_review_neighbors` output
    (candidate_payload dicts the authoring agent judges from) must stamp an
    amended-but-active Harbor node exactly as the store-level surfaces above do.
    Deterministic and keyless: tiny local fakes stand in for embed/vector; the
    neighbour itself is hydrated from the real reference graph, so the stamp
    comes from the true `get_node_by_slug` path."""
    from mitos.cli import cmd_init
    from mitos.config import MitosConfig
    from mitos.sync import MitosSyncManager

    class _Embed:
        def get_embedding(self, text, is_query=False):
            return [0.1, 0.2, 0.3]

    class _Vector:
        def query(self, vector, limit=5):
            return [{"slug": "harbor-blob-encryption-at-rest", "score": 0.9}]

        def upsert(self, *a, **k):
            pass

    config = MitosConfig(str(tmp_path / "ws"))
    cmd_init(config)
    manager = MitosSyncManager(config)
    manager.store = build_reference_graph(str(tmp_path / "graph.sqlite"))
    manager.embed_provider = _Embed()
    manager.vector_store = _Vector()

    entry = ParsedEntry("decision", "probe-blob-encryption", 1, 2)
    entry.axiom = "Encrypt every stored blob at rest."
    neighbors = manager._review_neighbors(entry, declared_targets=set())
    assert isinstance(neighbors, list) and neighbors, neighbors
    payload = neighbors[0]
    assert payload["slug"] == "harbor-blob-encryption-at-rest"
    assert payload["amended_by"] == ["harbor-blob-key-rotation-quarterly"]


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


# --- Layer A: the store transitions the commentary reconcile is built on ----------
#
# The reconcile routes a caller to the store's in-place commentary UPDATE, which was
# reachable only from unit tests before it — so the three tests below pin the STORE
# transitions it depends on (in-place update, MI-3's no-tick, declarative edge
# deletion) as deterministic emergent behaviour, and the fourth drives the real
# `perform_sync` path end to end. Stated precisely because the distinction matters:
# the first three would pass against a build with no reconcile at all, which makes
# them a foundation rather than a regression guard for it.
#
# Deliberately NOT added to `decisions.reference.md` + a re-frozen oracle: the corpus
# is a set of committed FACTS, and a reconcile is a TRANSITION between two states of
# one. Baking a diverged entry into the reference corpus would change what the corpus
# *is* and make every other oracle assertion read against a corpus that disagrees with
# its own graph.

def _harbor_entry(slug):
    """Returns the reference corpus's parsed entry for `slug`."""
    from mitos.parser import parse_entry_stream

    from _harness import CORPUS_PATH

    text = open(CORPUS_PATH, encoding="utf-8").read()
    matches = [e for e in parse_entry_stream(text, "decision") if e.slug == slug]
    assert len(matches) == 1, f"{slug} must appear exactly once in the reference corpus"
    return matches[0]


def test_a_commentary_reconcile_updates_in_place_and_touches_nothing_else(tmp_path):
    """A corrected `rejected_paths` lands on the SAME node, leaving the core untouched.

    Pins the whole shape of the reconcile in one place: same id, same axiom, same
    mechanisms, same scope, same edges, same node count — one field changed and
    `updated_at` ticked.
    """
    store = build_reference_graph(str(tmp_path / "golden.sqlite"))
    before = {n["slug"]: n for n in store.get_all_nodes()}
    count_before = len(before)
    target = before["harbor-backup-nightly"]
    edges_before = sorted(
        (e["source_id"], e["target_id"], e["edge_type"]) for e in store.get_edges()
    )

    # The golden fixture commits with a NULL confirmation pair, so stamp a real one —
    # otherwise the carry-forward assertion below compares None to None and holds
    # against a build that destroys the stamps.
    with store._get_connection() as conn:
        conn.execute("UPDATE nodes SET confirmed_by = 'agent', confirmed_at = ? WHERE id = ?",
                     ("2026-06-23T13:04:17.147973+00:00", target["id"]))
    target = {n["slug"]: n for n in store.get_all_nodes()}["harbor-backup-nightly"]
    before["harbor-backup-nightly"] = target

    entry = _harbor_entry("harbor-backup-nightly")
    entry.rejected_paths = "Continuous backup — CORRECTED: revisited at pilot scale."
    # The reconcile carries the stored confirmation pair forward rather than restamping.
    entry.confirmed_by = target["confirmed_by"]
    entry.confirmed_at = target["confirmed_at"]
    store.commit_parsed_entry(entry)

    after = {n["slug"]: n for n in store.get_all_nodes()}
    node = after["harbor-backup-nightly"]

    assert len(after) == count_before, "a commentary correction mints no new node"
    assert node["id"] == target["id"], "the canonical core is immutable (M1)"
    assert node["core_axiom"] == target["core_axiom"]
    assert node["mechanisms"] == target["mechanisms"]
    assert node["rejected_paths"] == "Continuous backup — CORRECTED: revisited at pilot scale."
    assert sorted(node["scope"]) == sorted(target["scope"]), "scope untouched by an S1 fix"
    assert node["created_at"] == target["created_at"], "created_at never re-mints"
    assert node["confirmed_by"] == "agent", "the stored confirmation pair carries forward"
    assert node["confirmed_at"] == "2026-06-23T13:04:17.147973+00:00"
    assert node["updated_at"] != target["updated_at"], "a real change ticks updated_at"
    assert sorted(
        (e["source_id"], e["target_id"], e["edge_type"]) for e in store.get_edges()
    ) == edges_before, "an unchanged declaration must not disturb any edge (MI-5)"
    # Every other node is byte-identical.
    for slug, row in after.items():
        if slug != "harbor-backup-nightly":
            assert row == before[slug], f"the reconcile disturbed '{slug}'"


def test_a_byte_identical_recommit_does_not_tick_updated_at(tmp_path):
    """MI-3, which the divergence gate exists to preserve.

    If the gate ever stopped gating, every sync would re-commit every entry — and this
    is the property that would break first and most visibly, since `updated_at` is half
    the divergence cache's fingerprint.
    """
    store = build_reference_graph(str(tmp_path / "golden.sqlite"))
    before = {n["slug"]: n for n in store.get_all_nodes()}
    target = before["harbor-cache-is-process-singleton"]

    entry = _harbor_entry("harbor-cache-is-process-singleton")
    entry.confirmed_by = target["confirmed_by"]
    entry.confirmed_at = target["confirmed_at"]
    store.commit_parsed_entry(entry)

    node = [n for n in store.get_all_nodes() if n["slug"] == target["slug"]][0]
    assert node == target, "a byte-identical re-commit is a true no-op (MI-3)"


def test_dropping_a_declared_edge_deletes_exactly_that_edge(tmp_path):
    """Declarative mirroring: a removed markdown line DELETES the stored edge.

    This is why the reconcile splits edge additions from deletions and why `--yes`
    refuses the deletions — pinned here so the mirroring semantics cannot drift out
    from under that policy.
    """
    store = build_reference_graph(str(tmp_path / "golden.sqlite"))
    before = {n["slug"]: n for n in store.get_all_nodes()}
    target = before["harbor-backup-nightly"]
    resolved_id = before["oq-harbor-backup-cadence"]["id"]
    assert any(e["source_id"] == target["id"] and e["target_id"] == resolved_id
               for e in store.get_edges()), "the fixture must start with the edge"

    entry = _harbor_entry("harbor-backup-nightly")
    entry.resolves = []
    entry.confirmed_by = target["confirmed_by"]
    entry.confirmed_at = target["confirmed_at"]
    store.commit_parsed_entry(entry)

    assert not any(e["source_id"] == target["id"] and e["target_id"] == resolved_id
                   for e in store.get_edges()), "the declaration's removal deletes it"
    assert len(store.get_all_nodes()) == len(before), "and mints no node"


def test_the_real_sync_path_reconciles_a_harbor_entry_end_to_end(tmp_path, monkeypatch):
    """The Harbor fixture driven through `perform_sync`, not the store directly.

    The three tests above pin the store transitions the reconcile stands on; this one
    pins the reconcile itself — it fails against a build whose gate still skips every
    already-committed entry. Deterministic and service-free: the mock key satisfies the
    key gate, and no reconcile path makes a model call (the conflict judge fires below
    the gate and the reconcile never reaches it).
    """
    import os

    from mitos.config import MitosConfig
    from mitos.sync import MitosSyncManager

    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")

    config = MitosConfig(str(tmp_path))
    os.makedirs(config.mitos_dir, exist_ok=True)
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write("# Decisions\n<!-- BEGIN ENTRIES — new decisions go directly below "
                 "this line, newest first -->\n")
    manager = MitosSyncManager(config)

    # `record` rather than `sync` to seed: a sync-committed entry ROTATES out of the
    # buffer, and sync reads only the buffer — so a rotated entry is not reconcilable
    # at all. Record-authored entries never rotate, which is why they are the ones the
    # reconcile actually meets on a live corpus.
    seeded = manager.record_decision_entry(
        "Harbor backs up the metadata database nightly and retains 14 days of history.",
        "Continuous backup — operational cost unjustified at pilot scale.",
        ["storage"], mechanisms=["postgres"], slug="harbor-backup-nightly",
        acknowledge_neighbors=True,
    )
    assert seeded.get("state") == "active", seeded

    with open(config.decisions_file, "r", encoding="utf-8") as fh:
        text = fh.read()
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write(text.replace(
            "Continuous backup — operational cost unjustified at pilot scale.",
            "Continuous backup — CORRECTED: revisited at pilot scale.",
        ))

    manager.perform_sync(auto_accept=True)

    node = GraphStore(config.db_path).get_node_by_slug("harbor-backup-nightly")
    assert node["rejected_paths"] == "Continuous backup — CORRECTED: revisited at pilot scale."
    assert node["core_axiom"].startswith("Harbor backs up"), "the core is untouched (M1)"
