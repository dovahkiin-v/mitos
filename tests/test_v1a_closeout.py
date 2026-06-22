"""V1a Definition-of-Done closeout suite (Phase 8b — the §1.2 DoD gate).

The deterministic, **keyless, CI-gated** integration proofs that V1a works *as a
whole*, not just as a pile of green substrate units. Each test is a closeout
proof of one acceptance scenario against V1a's surface, mapped to a contract row:

- **T1** S1 cold-start round-trip — ``init`` → record → active view (W5/W6/W7/W8/W2/W3)
- **T2** S4 edit-in-place correction + the V1-D16 in-place commentary UPDATE (W1/W2)
- **T3** S5 idempotent re-record no-op — MI-3 (W1/W2)
- **T4** S2 bulk N≥200 distinct cores + the "Long" ≥40-deep chains — P10 (W1/W2/W3)
- **T5** S3 structured-filter Letter payload shape over a write-path graph (W3/W12)
- **T6** F2 idempotent ``pending_embeddings`` enqueue over a write-path graph (W4)

**Keyless is non-negotiable (G1).** The DoD CI gate (``ci.yml``) runs with no
secrets and no Qdrant — so every test here must run green *without*
``GEMINI_API_KEY``/``ANTHROPIC_API_KEY`` or the gate silently proves nothing
(the OD1 silent-skip the vision forbids). This dev box carries a global ``.env``
(``hermetic_mitos_env`` does NOT unset the keys — scout W1), so the ``_keyless``
autouse fixture below strips them: with no Gemini key the
``GeminiEmbeddingProvider`` raises at construction, the manager degrades to
graph-only, and ``record_decision_entry`` commits the graph with a best-effort
embed that is a pure no-op — byte-for-byte the CI posture, deterministic
regardless of quota. Every assertion keys on **graph state**, never on the
``embedding`` field (W1).

Consumer-surface tests (T1–T3) drive the real ``cmd_init`` + the sacred
``record_decision_entry`` write path (the ``ws`` clone of
``tests/test_record_decision.py``); substrate-surface tests (T4–T6) drive the
parse→commit pipeline (``parse_entry_stream`` → ``commit_parsed_entry``) — the
e2e flip of the 5c/5d substrate proofs, asserted now over a graph built by the
real parser, not hand-made ``ParsedEntry`` objects. ids are asserted by
recompute via ``identity.compute_node_id`` (never a hardcoded 64-hex — G4).
"""

import inspect
import os
import shutil
import tempfile
from typing import Iterator, List, Tuple

import pytest

from mitos.cli import cmd_init
from mitos.config import MitosConfig
from mitos.identity import compute_node_id
from mitos.migrations import MIGRATION_STEPS, _pending_head, is_pre_v1a_schema
from mitos.parser import parse_entry_stream
from mitos.store import GraphStore, open_connection
from mitos.sync import MitosSyncManager


# --------------------------------------------------------------------------- #
# Keyless posture + fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _keyless(monkeypatch) -> None:
    """Strips live API keys so the dev box matches the keyless CI gate (G1/W1).

    ``hermetic_mitos_env`` (conftest, autouse) redirects the XDG dirs but leaves
    ``GEMINI_API_KEY``/``QDRANT_URL`` alone, and this box has a global ``.env`` —
    so without this the "keyless" DoD tests would make real embedding calls
    (slow, quota-flaky, non-deterministic). Unsetting the keys makes
    ``GeminiEmbeddingProvider`` raise at construction → the manager boots in
    graph-only mode → the best-effort embed is a no-op and ``_review_neighbors``
    returns ``[]`` (offline-safe, no pause). The CI posture, deterministically.
    """
    for var in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "QDRANT_URL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def ws(_keyless) -> Iterator[Tuple[MitosConfig, MitosSyncManager]]:
    """A fully initialised keyless temp workspace + a manager bound to it (T1–T3).

    The consumer-surface fixture: clones ``tests/test_record_decision.py::ws`` —
    ``MitosConfig`` over a ``tempfile.mkdtemp()`` + the real ``cmd_init`` (W5/W6/
    W7/W8 scaffolding) — but depends on ``_keyless`` so the bound manager degrades
    to graph-only. Read back via ``GraphStore(config.db_path)``.
    """
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    cmd_init(config)
    try:
        yield config, MitosSyncManager(config)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def store(_keyless) -> Iterator[GraphStore]:
    """A bare keyless file ``GraphStore`` booted at the V1a schema (T4–T6).

    The substrate-surface fixture: clones ``tests/test_store.py::temp_store``. The
    parse→commit pipeline writes here with no consumer overhead — fast at bulk
    scale, and ``commit_parsed_entry`` never embeds (sidesteps W1 entirely).
    """
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    s = GraphStore(path)
    try:
        yield s
    finally:
        if os.path.exists(path):
            os.remove(path)


def _commit_md(store: GraphStore, md: str, kind: str = "decision") -> List:
    """Parses a markdown stream and commits every entry through the write path.

    The e2e seam under test for T4–T6: real ``parse_entry_stream`` tokenization →
    real ``commit_parsed_entry`` graph mutation (strict mode — a malformed stream
    raises, which a closeout fixture should never produce).

    Args:
        store: The destination graph store.
        md: The decisions/questions markdown stream (no sentinel needed — the
            whole string is the entry stream).
        kind: ``"decision"`` or ``"open_question"`` (caller-declared, V1-D8).

    Returns:
        The list of ``CommitDelta`` objects, one per committed entry, in stream
        order — so callers building a chain commit oldest-first (G7).
    """
    return [store.commit_parsed_entry(e) for e in parse_entry_stream(md, kind)]


def _user_version(db_path: str) -> int:
    """Reads ``PRAGMA user_version`` through the V1a connection chokepoint."""
    conn = open_connection(db_path, read_only=True)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def _is_v1a_not_prototype(db_path: str) -> bool:
    """True iff the booted graph is a real V1a graph, NOT a refused pre-V1a one."""
    conn = open_connection(db_path, read_only=True)
    try:
        return not is_pre_v1a_schema(conn)
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# T1 — S1 cold-start: init → record → active view (proves W5/W6/W7/W8/W2/W3)
# --------------------------------------------------------------------------- #


def test_t1_s1_cold_start_round_trip(ws) -> None:
    """A decision recorded cold round-trips through init → record → active view.

    The only DoD scenario spanning init + config + spec + scaffolding + commit +
    read together, so it runs the real ``cmd_init`` (the ``ws`` fixture) + the
    real ``record_decision_entry`` write path + ``get_active_decisions``.
    """
    config, m = ws

    # Cold-start scaffolding landed (W5 ladder boot at the live head, W6 config seed,
    # W7 format-spec install, W8 .mitos/ + buffers) — a healthy, ready workspace. The
    # head is read programmatically (it advances as later visions append rungs, e.g.
    # V1b's step 2); ``_is_v1a_not_prototype`` proves the V1a schema is present.
    assert _user_version(config.db_path) == _pending_head(MIGRATION_STEPS)
    assert _is_v1a_not_prototype(config.db_path)
    assert os.path.exists(config.decisions_file)
    assert os.path.exists(config.questions_file)

    res = m.record_decision_entry(
        axiom="Use SQLite in WAL mode for the graph store.",
        rejected_paths="pgvector (too heavy for local-first); a server DB (breaks offline).",
        scope=["substrate", "database"],
        mechanisms=["sqlite", "wal-mode"],
        context="Local-first concurrent reads and writes.",
        slug="use-sqlite-wal",
    )
    assert "error" not in res
    assert res["status"] == "created"
    assert res["state"] == "active"

    # Read back through the active view (W3) — the committed node has the V1a
    # reader-facing shape, hydrated from the STRICT schema.
    actives = GraphStore(config.db_path).get_active_decisions()
    assert len(actives) == 1
    node = actives[0]
    assert node["slug"] == "use-sqlite-wal"
    assert node["core_axiom"] == "Use SQLite in WAL mode for the graph store."
    assert node["mechanisms"] == ["sqlite", "wal-mode"]
    assert node["scope"] == ["database", "substrate"]  # casefold + sorted scopes
    assert node["rejected_paths"].startswith("pgvector")

    # The write path's id IS the slug-free canonical-core hash (W2 identity) —
    # recompute over the read-back core and assert equality (G4, never a literal).
    assert res["id"] == compute_node_id(
        kind="decision",
        axiom=node["core_axiom"],
        mechanism_refs=node["mechanisms"],
    )


# --------------------------------------------------------------------------- #
# T2 — S4 edit-in-place correction + the V1-D16 in-place UPDATE (proves W1/W2)
# --------------------------------------------------------------------------- #


def test_t2_s4_correction_via_corrects(ws) -> None:
    """A ``--corrects`` correction mints a NEW node + a kill-edge; the target retires.

    The correction is a *different* canonical core (the axiom is fixed), so it is a
    genuinely new node + a ``corrects`` kill-edge — never an in-place UPDATE. The
    corrected predecessor leaves the active view (``get_node_state`` →
    ``"corrected"``, the V1a vocabulary, ≠ ``"superseded"`` — G6) and carries the
    stamped ``corrected_by`` modifier so a reader of the moved-on node still sees
    who moved on.
    """
    config, m = ws
    store = GraphStore(config.db_path)

    buggy = m.record_decision_entry(
        axiom="Store timestamps as local time.",
        rejected_paths="UTC (deemed unnecessary at the time — wrongly).",
        scope=["substrate"],
        slug="local-timestamps",
    )
    assert buggy["status"] == "created"

    fixed = m.record_decision_entry(
        axiom="Store timestamps as UTC ISO-8601 microseconds.",
        rejected_paths="Local time (drifts across hosts; the bug being corrected).",
        scope=["substrate"],
        slug="utc-timestamps",
        corrects="local-timestamps",
    )
    assert fixed["status"] == "created"
    assert fixed["id"] != buggy["id"]  # a corrected core is a new node, not an UPDATE

    # The corrected predecessor retired; only the corrector is active.
    assert store.get_node_state(buggy["id"]) == "corrected"
    assert store.get_node_state(fixed["id"]) == "active"
    active_slugs = {n["slug"] for n in store.get_active_decisions()}
    assert active_slugs == {"utc-timestamps"}

    # corrected_by stamped on the inactive predecessor (read by id — a retired node
    # is NOT resolvable by slug, only by content-hash id; scout 8a note).
    retired = store.get_node(buggy["id"])
    assert retired["corrected_by"] == ["utc-timestamps"]


def test_t2_s4_inplace_commentary_update_is_not_a_new_node(store) -> None:
    """A same-core commentary edit (new slug + changed scope) is one node UPDATE (V1-D16).

    The distinguishing half of S4: a correction is a new node (above), but a
    *commentary* edit on the same canonical core is an in-place UPDATE — same id,
    one node, the slug renamed in place, ``updated_at`` ticks on the footprint
    change. Driven through ``commit_parsed_entry`` (the agentic
    ``record_decision_entry`` short-circuits a same-core re-record to ``exists``
    before any commentary write — T3 proves that path; this proves the store-level
    UPDATE the cutover/in-file edit relies on).
    """
    (d1,) = _commit_md(
        store,
        "### caching-policy\n"
        "**Decided:** Cache embeddings on the content hash.\n"
        "**Rejected:** Re-embed every sync (wasteful).\n"
        "**Scope:** vectors\n",
    )
    row1 = _node_row(store, d1.node_id)

    (d2,) = _commit_md(
        store,
        "### caching-policy-renamed\n"
        "**Decided:** Cache embeddings on the content hash.\n"
        "**Rejected:** Re-embed every sync (wasteful).\n"
        "**Scope:** vectors, performance\n",
    )
    assert d2.node_id == d1.node_id  # slug excluded from identity (Q5)
    assert d2.commentary_fields_changed is True
    assert _count(store, "nodes") == 1  # a rename + scope edit, not a second node

    row2 = _node_row(store, d1.node_id)
    assert row2["slug"] == "caching-policy-renamed"
    assert row2["axiom"] == "Cache embeddings on the content hash."  # core fenced
    assert row2["created_at"] == row1["created_at"]
    assert row2["updated_at"] > row1["updated_at"]  # ticked on the footprint change


# --------------------------------------------------------------------------- #
# T3 — S5 idempotent re-record no-op (proves W1/W2, MI-3)
# --------------------------------------------------------------------------- #


def test_t3_s5_idempotent_recommit_noop(ws) -> None:
    """A byte-identical re-record is a true no-op — same id, one node, no tick (MI-3).

    The consumer-path proof: the second ``record_decision_entry`` of the same
    canonical core returns ``status: "exists"`` (the agentic idempotency
    short-circuit, V1-D16) — not a duplicate node and not a spurious
    ``slug_collision``. The node row is untouched (``updated_at`` stable).
    """
    config, m = ws
    store = GraphStore(config.db_path)
    kwargs = dict(
        axiom="Markdown is the source of truth; the graph is derivative.",
        rejected_paths="A binary store (M7 — never lock content behind a proprietary format).",
        scope=["substrate"],
        slug="markdown-is-truth",
    )

    first = m.record_decision_entry(**kwargs)
    assert first["status"] == "created"
    node_before = store.get_node(first["id"])

    second = m.record_decision_entry(**kwargs)
    assert second["status"] == "exists"  # idempotent — no second node minted
    assert second["id"] == first["id"]

    assert len(store.get_active_decisions()) == 1
    node_after = store.get_node(first["id"])
    assert node_after["updated_at"] == node_before["updated_at"]  # no tick (MI-3)


# --------------------------------------------------------------------------- #
# T4 — S2 bulk (N≥200 distinct cores) + the "Long" depth fixture (proves W1/W2/W3)
# --------------------------------------------------------------------------- #

BULK_N = 256  # ≥200 floor (§1.2); a round power of two


def test_t4_s2_bulk_distinct_cores(store) -> None:
    """N≥200 distinct canonical cores → N distinct ids (MI-1 determinism at scale).

    The breadth half of P10: parse + commit ``BULK_N`` entries with distinct
    axioms (distinct cores) through the write path and assert the id set has
    exactly ``BULK_N`` members — no collision, no convergence — and every one is
    active. (Two entries sharing a core but differing only in slug would converge;
    the fixture uses distinct cores deliberately, §6.2.)
    """
    md = "".join(
        f"### axiom-{i}\n"
        f"**Decided:** Distinct architectural axiom number {i}.\n"
        f"**Rejected:** The alternative rejected for axiom {i}.\n\n"
        for i in range(BULK_N)
    )
    deltas = _commit_md(store, md)

    assert len(deltas) == BULK_N
    ids = {d.node_id for d in deltas}
    assert len(ids) == BULK_N  # determinism + uniqueness at bulk (MI-1)
    assert len(store.get_active_decisions()) == BULK_N

    # Recompute one id from the read-back core to pin the hash is the real thing.
    sample = store.get_node_by_slug("axiom-7")
    assert sample["id"] == compute_node_id(
        kind="decision", axiom=sample["core_axiom"], mechanism_refs=sample["mechanisms"]
    )


CHAIN_DEPTH = 45  # ≥40 floor (P10 depth)


def test_t4_long_supersedes_chain(store) -> None:
    """A ≥40-deep ``supersedes`` chain resolves the active view to exactly one node.

    The depth half of P10 (depth and breadth fail differently): build the chain
    oldest-first (G7 — each kill-edge target must pre-exist) so link k supersedes
    link k-1. All ``CHAIN_DEPTH`` nodes are retained (append-only), but the
    kill-edge anti-join excludes the 39+ predecessors — the active view for the
    chain's scope is the single head. Referential integrity (every edge target
    resolves) and the kill-edge ancestry (a single path, 1 hop per link) hold
    end-to-end.
    """
    prev = None
    for i in range(CHAIN_DEPTH):
        md = (
            f"### chain-{i}\n"
            f"**Decided:** Chain link {i}.\n"
            f"**Rejected:** Stopping at link {i}.\n"
            f"**Scope:** longchain\n"
        )
        if prev is not None:
            md += f"**Supersedes:** {prev}\n"
        _commit_md(store, md)
        prev = f"chain-{i}"

    actives = store.get_active_decisions(scope="longchain")
    assert len(actives) == 1
    assert actives[0]["slug"] == f"chain-{CHAIN_DEPTH - 1}"  # only the head survives

    # Referential integrity + kill-edge ancestry: exactly CHAIN_DEPTH-1 supersedes
    # edges, each target an existing node, forming one path (every link but the
    # head is superseded exactly once).
    edges = [e for e in store.get_edges() if e["edge_type"] == "supersedes"]
    assert len(edges) == CHAIN_DEPTH - 1
    assert all(store.get_node(e["target_id"]) is not None for e in edges)
    superseded_targets = {e["target_id"] for e in edges}
    assert len(superseded_targets) == CHAIN_DEPTH - 1  # each predecessor killed once
    head_id = actives[0]["id"]
    assert head_id not in superseded_targets  # the head is nobody's target


def test_t4_long_corrects_chain(store) -> None:
    """A ≥40-deep ``corrects`` chain likewise resolves to exactly one active node.

    The corrects-edge variant of the Long test (V1a's second kill-edge): the same
    depth invariant holds through the other kill-edge type — the anti-join is over
    ``('supersedes', 'corrects')`` as one set.
    """
    prev = None
    for i in range(CHAIN_DEPTH):
        md = (
            f"### fix-{i}\n"
            f"**Decided:** Correction revision {i}.\n"
            f"**Rejected:** Leaving revision {i} as the final word.\n"
            f"**Scope:** longfix\n"
        )
        if prev is not None:
            md += f"**Corrects:** {prev}\n"
        _commit_md(store, md)
        prev = f"fix-{i}"

    actives = store.get_active_decisions(scope="longfix")
    assert len(actives) == 1
    assert actives[0]["slug"] == f"fix-{CHAIN_DEPTH - 1}"
    assert store.get_node_state(actives[0]["id"]) == "active"


# --------------------------------------------------------------------------- #
# T5 — S3 structured-filter Letter payload shape over a write-path graph (W3/W12)
# --------------------------------------------------------------------------- #


def test_t5_s3_letter_shape_e2e(store) -> None:
    """``query_letter`` over a write-path graph returns the contracted C4 payload.

    The e2e flip of the 5d substrate proof: build the graph through parse→commit
    (oldest-first so the superseding entry's target pre-exists), then assert the
    Letter projection ``{slug, axiom, rejected_paths, scope}`` (note ``axiom``, the
    C4 projection name, NOT ``core_axiom``), that ``brief=True`` drops only
    ``rejected_paths`` (keeps ``axiom``), that the view is active-only (a
    superseded node never appears), and that there is no semantic path.
    """
    _commit_md(
        store,
        "### keep-me\n"
        "**Decided:** The standing axiom.\n"
        "**Rejected:** The path not taken.\n"
        "**Scope:** letter\n\n"
        "### retire-me\n"
        "**Decided:** The old axiom.\n"
        "**Rejected:** Old reasoning.\n"
        "**Scope:** letter\n\n"
        "### replace-it\n"
        "**Decided:** The replacement axiom.\n"
        "**Rejected:** New reasoning.\n"
        "**Scope:** letter\n"
        "**Supersedes:** retire-me\n",
    )

    payloads = store.query_letter(scope="letter", kind="decision")
    assert {p["slug"] for p in payloads} == {"keep-me", "replace-it"}  # active-view only

    keep = next(p for p in payloads if p["slug"] == "keep-me")
    assert set(keep) == {"slug", "axiom", "rejected_paths", "scope"}
    assert keep["axiom"] == "The standing axiom."  # C4 projection name
    assert keep["rejected_paths"] == "The path not taken."
    assert keep["scope"] == ["letter"]

    (brief,) = store.query_letter(slug="keep-me", brief=True)
    assert "rejected_paths" not in brief
    assert brief["axiom"] == "The standing axiom."

    # The superseded node is unreachable by the Letter query (active-view only).
    assert store.query_letter(slug="retire-me") == []
    # Deterministic structured filter — no vector/embedding path in V1a.
    assert "semantic" not in inspect.getsource(GraphStore.query_letter).lower()


# --------------------------------------------------------------------------- #
# T6 — F2 idempotent pending_embeddings enqueue over a write-path graph (W4)
# --------------------------------------------------------------------------- #


def test_t6_f2_enqueue_idempotent_e2e(store) -> None:
    """Committing one node twice keeps exactly one ``pending_embeddings`` row (F2).

    The e2e flip of the 5c substrate proof, observed via the live drain surface
    (``get_pending_embeddings`` / ``claim_pending_embeddings``): the enqueue is an
    UPSERT on the ``node_id`` PK, so a re-commit of the same core never duplicates
    the Outbox row; and a re-enqueue RESETS drain state (MI-12 — a dead-letter on a
    transient outage gets a fresh drain attempt), proven here entirely through the
    public drain surface (``increment_pending_attempts`` → re-commit → reset).
    """
    md = (
        "### enqueue-once\n"
        "**Decided:** A node committed twice enqueues once.\n"
        "**Rejected:** A per-commit duplicate row.\n"
    )
    (d1,) = _commit_md(store, md)
    (d2,) = _commit_md(store, md)
    assert d2.node_id == d1.node_id

    pending = store.get_pending_embeddings()
    assert len(pending) == 1
    assert pending[0]["node_id"] == d1.node_id

    claimed = store.claim_pending_embeddings("drainer-1", limit=10)
    assert [r["node_id"] for r in claimed] == [d1.node_id]

    # MI-12 re-enqueue resets drain state: advance retry_count via the public
    # surface, re-commit byte-identically, and assert the row is revived to 0.
    store.increment_pending_attempts(d1.node_id)
    store.increment_pending_attempts(d1.node_id)
    assert store.get_pending_embeddings()[0]["retry_count"] == 2

    _commit_md(store, md)
    pending = store.get_pending_embeddings()
    assert len(pending) == 1  # still exactly one row
    assert pending[0]["retry_count"] == 0  # drain state reset (MI-12)


# --------------------------------------------------------------------------- #
# Raw-SQL read helpers (T2 store-path UPDATE asserts on the nodes table directly)
# --------------------------------------------------------------------------- #


def _node_row(store: GraphStore, node_id: str):
    """Reads a single ``nodes`` row as a dict via raw SQL, or None."""
    conn = store._get_connection()
    try:
        row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _count(store: GraphStore, table: str) -> int:
    """Counts rows in a table via raw SQL (table name is a code-internal literal)."""
    conn = store._get_connection()
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()
