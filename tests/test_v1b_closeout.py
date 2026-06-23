"""V1b Definition-of-Done closeout suite (Phase 7b — the §1.2 DoD gate).

The deterministic, **keyless, CI-gated** integration proofs that V1b works *as a
whole* — the full v0.1 edge catalog, ``questions.md`` steady-state ingestion, the
intra-sync fixpoint, and OQ Stage-2 resolution all flowing through the real sync
path and landing in the graph as a coherent whole. Each test is a closeout proof
of one of the three DoD gates 7b owns (the ones no single behaviour phase could
close alone, because they only exist at the whole-substrate level):

- **DoD #4** Dogfood self-parse — a dogfood-shaped corpus (OQ ``Derives-From:``,
  decision ``Resolves:``, multi-edge-type entries, a cross-file forward-ref) syncs
  cleanly through ``perform_sync`` + ``questions.md`` ingestion (4a) + the fixpoint
  (4b); every OQ reaches the graph, every declared edge commits, OQ ``state``
  computes, the forward-ref chain converges in **one** sync, and no entry is ever
  silently dropped (OD1 — a deliberately-malformed OQ quarantines *loudly* while
  the batch proceeds; branch (b) of the V1a §6.2 (a)/(b) protocol). Contract row T4.
- **DoD #5** Successor-less retirement — a deletion decision carrying
  ``**Supersedes:**`` drops its target from the active view via the existing
  ``supersedes`` kill-edge **while the deletion decision itself stays active and
  surfaceable with its reasoning** (M5 anti-knowledge — a re-proposal meets *why*
  the feature was dropped); removing the line resurrects the target. Contract row T5.
- **DoD #13** V1a-suite-green-with-7-folded — the seven newly-committing non-kill
  edge types commit *without* retiring either endpoint while the two kill-edges
  still retire, so the V1a active-view contract holds under the widened nine-type
  catalog. The MI verifications (MI-4/MI-5/MI-12) are **run** green in the full
  keyless lane and cited as evidence (IMPLEMENTATION_NOTES), not re-authored here.
  Contract row T13 (full-suite slice).

**Keyless is non-negotiable.** Two postures coexist (the established discipline):

1. **parse→commit** (DoD #5/#13, deterministic substrate): the ``store`` fixture
   strips keys (``_keyless``) so ``commit_parsed_entry`` never embeds; every
   assertion keys on **graph state**, never on the ``embedding`` field.
2. **``perform_sync`` ingestion** (DoD #4, the one gate that exercises the 4a/4b
   ``questions.md`` ingestion + fixpoint): ``perform_sync`` early-returns without
   ``GEMINI_API_KEY`` (4a), so the gate sets a *mock* key + patches
   ``google.genai.Client`` (the ``tests/test_sync.py`` idiom), forces the manager
   into graph-only mode (no embedding/Qdrant network — the P14 degraded fallback),
   and patches ``run_sync_enrichment`` to an **identity passthrough** so each
   distinct authored axiom stays a distinct canonical core (a single fixed mock
   enrichment response would collapse every decision to one node). It asserts only
   on graph state (``get_edges``/``get_open_questions``/``get_active_decisions``/
   ``get_node_state``), never on embedding.

ids are read dynamically (``CommitDelta.node_id`` / ``get_node_by_slug``); the
shipped version is read from ``mitos.__version__`` — never a hardcoded content-hash.
"""

import os
import shutil
import tempfile
from typing import Dict, Iterator, List, Tuple
from unittest.mock import patch

import pytest

from mitos import __version__
from mitos.config import MitosConfig
from mitos.parser import parse_entry_stream
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager


# --------------------------------------------------------------------------- #
# Keyless posture + fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def _keyless(monkeypatch) -> None:
    """Strips live API keys so the dev box matches the keyless CI gate.

    Mirrors ``tests/test_v1a_closeout.py::_keyless``: with no Gemini key the
    embedding provider raises at construction → the store degrades to graph-only →
    ``commit_parsed_entry`` is a pure graph mutation (never embeds), deterministic
    regardless of quota. Used by the parse→commit gates (DoD #5/#13). The DoD #4
    ``perform_sync`` gate sets its own mock key per-test (the two postures coexist).
    """
    for var in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "QDRANT_URL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def store(_keyless) -> Iterator[GraphStore]:
    """A bare keyless file ``GraphStore`` for the parse→commit gates (DoD #5/#13).

    Clones ``tests/test_v1a_closeout.py::store``: the parse→commit pipeline writes
    here with no consumer overhead, and ``commit_parsed_entry`` never embeds.
    """
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    s = GraphStore(path)
    try:
        yield s
    finally:
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture
def sync_ws(_keyless) -> Iterator[Tuple[MitosConfig, MitosSyncManager]]:
    """A temp workspace + a graph-only manager for the DoD #4 ``perform_sync`` gate.

    Mirrors ``tests/test_sync.py::sync_env`` (manual ``.mitos/`` + explicit config
    paths) and adds a sibling ``questions.md`` target. The manager is forced into
    graph-only mode (``embed_provider``/``vector_store`` = ``None``) so the sync runs
    fully offline and deterministic — no embedding call, no Qdrant write — exactly
    the P14 degraded fallback the keyless DoD posture relies on. DoD #4 asserts only
    on graph state, so embeddings are out of scope.
    """
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    config.db_path = os.path.join(tmp, ".mitos", "graph.sqlite")
    config.decisions_file = os.path.join(tmp, "decisions.md")
    config.questions_file = os.path.join(tmp, "questions.md")
    config.archive_dir = os.path.join(tmp, "decisions", "archive")
    os.makedirs(os.path.join(tmp, ".mitos"), exist_ok=True)

    manager = MitosSyncManager(config)
    # Force graph-only: guarantee no embedding/Qdrant network regardless of the
    # mock key the DoD #4 test sets to clear the sync's key gate.
    manager.embed_provider = None
    manager.vector_store = None
    try:
        yield config, manager
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _commit_md(store: GraphStore, md: str, kind: str = "decision") -> List:
    """Parses a markdown stream and commits every entry through the write path.

    The DoD #5/#13 seam: real ``parse_entry_stream`` tokenization → real
    ``commit_parsed_entry`` graph mutation, in stream (top-to-bottom) order — so a
    fixture authoring a kill/non-kill edge places its target *above* the referrer.

    Args:
        store: The destination graph store.
        md: The decisions/questions markdown stream (no sentinel needed — the whole
            string is the entry stream).
        kind: ``"decision"`` or ``"open_question"`` (caller-declared, V1-D8).

    Returns:
        The list of ``CommitDelta`` objects, one per committed entry, in stream order.
    """
    return [store.commit_parsed_entry(e) for e in parse_entry_stream(md, kind)]


# The canonical sentinel'd headers (format-spec.md §5). The preamble (everything up
# to and including the sentinel) yields zero graph state; entries go below it.
_DEC_HEADER = (
    "# Decisions\n"
    "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->\n"
)
_OQ_HEADER = (
    "# Open Questions\n"
    "<!-- BEGIN ENTRIES — new open questions go directly below this line, newest first -->\n"
)


def _write(path: str, header: str, body: str) -> None:
    """Writes a sentinel'd buffer file (header preamble + entry-stream body)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + body)


def _identity_enrichment(_client, entry, _active):
    """Identity passthrough for ``run_sync_enrichment`` — keeps cores distinct.

    The real enrichment refines the axiom via an LLM; under ``auto_accept`` a
    *failed* enrichment skips the entry (4a's F1 fold-in), and a single fixed mock
    response would refine every decision to the *same* axiom → one converged node.
    Echoing the entry's own fields keeps each authored core distinct and suggests no
    relationships (so the test's *authored* edges are the only ones committed).
    """
    return {
        "refined_core_axiom": entry.axiom,
        "refined_mechanisms": entry.mechanisms,
        "refined_scope": entry.scope,
        "suggested_relationships": {},
    }


def _slug_id_map(store: GraphStore, slugs: List[str]) -> Dict[str, str]:
    """Maps each (active) slug to its content-hash node id via the read surface."""
    out: Dict[str, str] = {}
    for slug in slugs:
        node = store.get_node_by_slug(slug)
        assert node is not None, f"expected an active node for slug {slug!r}"
        out[slug] = node["id"]
    return out


def _edge_triples(store: GraphStore) -> set:
    """All committed edges as ``(edge_type, source_id, target_id)`` triples."""
    return {
        (e["edge_type"], e["source_id"], e["target_id"]) for e in store.get_edges()
    }


# --------------------------------------------------------------------------- #
# DoD #4 — Dogfood self-parse (the end-to-end spine proof, contract row T4)
# --------------------------------------------------------------------------- #


@patch("mitos.sync.run_sync_enrichment", side_effect=_identity_enrichment)
@patch("google.genai.Client")
def test_dod4_dogfood_corpus_syncs_in_one_pass(
    _mock_client, _mock_enrich, sync_ws, monkeypatch
) -> None:
    """A dogfood corpus syncs cleanly in ONE ``perform_sync`` — every edge family lands.

    The whole-substrate spine proof. The corpus exercises, in a single sync:

    - a host decision ``host-d`` everything hangs off;
    - a resolver decision ``resolver-d`` carrying ``Resolves: oq-resolved`` — a
      **cross-file forward-ref** (the OQ is in ``questions.md``, parsed after all
      decisions), so it quarantines on the main pass and converges via 4b's
      intra-sync fixpoint in THIS sync;
    - two multi-edge decisions (``multi-a`` Amends+Cites ``host-d``; ``multi-b``
      Depends-On+Narrows ``host-d``) — each commits **both** its edges from one entry;
    - three OQs: ``oq-open`` (Derives-From ``host-d`` — OQ→D), ``oq-resolved`` (the
      one ``resolver-d`` resolves), ``oq-cite`` (Cites ``host-d`` — OQ→any).

    Asserts on graph state only: node presence (no silent drop, OD1), every declared
    edge with correct type+endpoints, and OQ Stage-2 ``state`` (the resolved OQ reads
    ``resolved``, the parked ones ``parked``).
    """
    config, manager = sync_ws
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")  # clear the 4a sync key gate

    # Entries newest-first in the file; parse_file_reversed commits oldest-first, so
    # host-d (bottom) lands before its same-file referrers. The cross-file
    # resolver-d → oq-resolved ref converges via the fixpoint.
    _write(
        config.decisions_file,
        _DEC_HEADER,
        "### multi-b\n"
        "**Decided:** Multi-edge B both depends on and narrows the host.\n"
        "**Rejected:** Folding both relations into a single edge.\n"
        "**Depends-On:** host-d\n"
        "**Narrows:** host-d\n\n"
        "### multi-a\n"
        "**Decided:** Multi-edge A both amends and cites the host.\n"
        "**Rejected:** Splitting it into two separate entries.\n"
        "**Amends:** host-d\n"
        "**Cites:** host-d\n\n"
        "### resolver-d\n"
        "**Decided:** The resolver decision settles the resolved question.\n"
        "**Rejected:** Leaving the question open indefinitely.\n"
        "**Resolves:** oq-resolved\n\n"
        "### host-d\n"
        "**Decided:** The host decision the rest of the corpus hangs off.\n"
        "**Rejected:** A corpus with no shared host.\n",
    )
    _write(
        config.questions_file,
        _OQ_HEADER,
        "### oq-cite\n"
        "**Topic:** A question that cites the host decision.\n"
        "**Questions:** Does an OQ-sourced cite edge commit?\n"
        "**Cites:** host-d\n\n"
        "### oq-resolved\n"
        "**Topic:** The question the resolver decision answers.\n"
        "**Questions:** What does resolver-d settle?\n\n"
        "### oq-open\n"
        "**Topic:** The open question deriving from the host.\n"
        "**Questions:** What hangs off host-d while still open?\n"
        "**Derives-From:** host-d\n",
    )

    manager.perform_sync(auto_accept=True)

    read = GraphStore(config.db_path)

    # No silent drop (OD1): every well-formed entry committed — 4 decisions + 3 OQs.
    dec_slugs = {d["slug"] for d in read.get_active_decisions()}
    assert dec_slugs == {"host-d", "resolver-d", "multi-a", "multi-b"}
    oqs = {q["slug"]: q for q in read.get_open_questions()}
    assert set(oqs) == {"oq-open", "oq-resolved", "oq-cite"}

    # OQ Stage-2 state computes (4c): the active resolver makes oq-resolved resolved;
    # the others have no active resolver → parked.
    assert oqs["oq-resolved"]["state"] == "resolved"
    assert oqs["oq-open"]["state"] == "parked"
    assert oqs["oq-cite"]["state"] == "parked"

    # Every declared edge committed with the right type + endpoints (multi-edge
    # entries commit BOTH edges; the cross-file resolver edge proves the forward-ref
    # chain converged in this one sync, asserted via graph presence not timing).
    ids = _slug_id_map(
        read, ["host-d", "resolver-d", "multi-a", "multi-b", "oq-open", "oq-resolved", "oq-cite"]
    )
    expected = {
        ("derives_from", ids["oq-open"], ids["host-d"]),
        ("resolves", ids["resolver-d"], ids["oq-resolved"]),
        ("amends", ids["multi-a"], ids["host-d"]),
        ("cites", ids["multi-a"], ids["host-d"]),
        ("depends_on", ids["multi-b"], ids["host-d"]),
        ("narrows", ids["multi-b"], ids["host-d"]),
        ("cites", ids["oq-cite"], ids["host-d"]),
    }
    actual = _edge_triples(read)
    assert expected <= actual
    assert len(actual) == len(expected)  # exactly the authored edges, nothing spurious


@patch("mitos.sync.run_sync_enrichment", side_effect=_identity_enrichment)
@patch("google.genai.Client")
def test_dod4_malformed_oq_quarantines_loudly_batch_proceeds(
    _mock_client, _mock_enrich, sync_ws, monkeypatch, capsys
) -> None:
    """OD1 branch (b): a bad OQ quarantines as a LOUD vector while the batch commits.

    The V1a §6.2 protocol forbids a silent third branch. An OQ referencing a target
    that is authored nowhere in the corpus fails *loudly* (the post-fixpoint
    ``missing_target`` guiding vector) and stays out of the graph, while the
    well-formed host decision and OQ commit untouched — the per-entry bulkhead (P5
    dead-letter / P7) isolating the one poison record, never amputating the batch.
    """
    config, manager = sync_ws
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")

    _write(
        config.decisions_file,
        _DEC_HEADER,
        "### good-host\n"
        "**Decided:** A well-formed decision that must commit despite a sibling's failure.\n"
        "**Rejected:** Aborting the whole batch on one bad entry.\n",
    )
    _write(
        config.questions_file,
        _OQ_HEADER,
        "### bad-oq\n"
        "**Topic:** An OQ pointing at a target authored nowhere in the corpus.\n"
        "**Questions:** Does this quarantine loudly instead of dropping silently?\n"
        "**Derives-From:** ghost-target-never-authored\n\n"
        "### good-oq\n"
        "**Topic:** A well-formed open question.\n"
        "**Questions:** Does the healthy entry still commit?\n",
    )

    manager.perform_sync(auto_accept=True)

    read = GraphStore(config.db_path)
    # The batch proceeded: both well-formed entries committed.
    assert {d["slug"] for d in read.get_active_decisions()} == {"good-host"}
    assert {q["slug"] for q in read.get_open_questions()} == {"good-oq"}
    # The bad entry was NOT silently dropped — it quarantined as a loud guiding vector.
    out = capsys.readouterr().out
    assert "[Quarantined]" in out
    assert "bad-oq" in out
    assert "not present anywhere in this corpus" in out


# --------------------------------------------------------------------------- #
# DoD #5 — Successor-less retirement (deletion decision, contract row T5)
# --------------------------------------------------------------------------- #


# The deletion decision's reasoning — the M5 anti-knowledge that a [RETIRED] marker
# would have destroyed. Pinned as substrings so the surfaceability axis is meaningful.
_DROP_AXIOM = "We are dropping the speculative cache because it never paid for its complexity."
_DROP_REJECTED = "Keeping the speculative cache (it added surface for no measured win)."


def _drop_x_md(with_supersedes: bool) -> str:
    """The deletion decision ``drop-x``, optionally carrying its kill-edge line.

    Same canonical core either way (the axiom is fixed), so re-committing toggles
    only the ``Supersedes:`` commentary edge — the declarative-mirror reversal.
    """
    md = (
        "### drop-x\n"
        f"**Decided:** {_DROP_AXIOM}\n"
        f"**Rejected:** {_DROP_REJECTED}\n"
    )
    if with_supersedes:
        md += "**Supersedes:** old-x\n"
    return md


def test_dod5_successor_less_death_drops_target_and_stays_surfaceable(store) -> None:
    """A deletion decision retires its target yet stays active + surfaceable (M5).

    The heart of DoD #5 — the difference between a tombstone that erases and a
    decision that *remembers*. Three axes:

    1. **Drop** — ``old-x`` leaves the active view (incoming ``supersedes`` kill-edge;
       ``get_node_state`` → ``"superseded"``).
    2. **Surfaceable successor (M5)** — ``drop-x`` itself stays active and readable
       with its ``core_axiom`` / ``rejected_paths`` intact, so a future re-proposal
       of the dropped idea surfaces *why* it died (the axis the existing
       ``test_declarative_mirror_*`` test does not cover).
    3. **Resurrection** — re-committing ``drop-x`` WITHOUT the ``Supersedes:`` line
       deletes the edge (declarative mirror, V1-D21) and ``old-x`` is active again;
       re-adding the line retires it once more.
    """
    (old,) = _commit_md(
        store,
        "### old-x\n"
        "**Decided:** The speculative cache we are about to drop.\n"
        "**Rejected:** Shipping with no cache at all.\n",
    )
    (drop,) = _commit_md(store, _drop_x_md(with_supersedes=True))

    # Axis 1 — drop.
    active = {d["slug"]: d for d in store.get_active_decisions()}
    assert "old-x" not in active
    assert store.get_node_state(old.node_id) == "superseded"

    # Axis 2 — the deletion decision stays active AND carries its reasoning (M5).
    assert "drop-x" in active
    assert store.get_node_state(drop.node_id) == "active"
    surfaced = active["drop-x"]
    assert surfaced["core_axiom"] == _DROP_AXIOM
    assert surfaced["rejected_paths"] == _DROP_REJECTED

    # Axis 3 — resurrection on edge removal, re-retirement on re-add (provoke the
    # reversal, P10 — not just the happy drop).
    again = _commit_md(store, _drop_x_md(with_supersedes=False))
    assert again[0].node_id == drop.node_id  # same core → in-place commentary edit
    assert store.get_node_state(old.node_id) == "active"  # target resurrected
    assert {d["slug"] for d in store.get_active_decisions()} == {"old-x", "drop-x"}

    _commit_md(store, _drop_x_md(with_supersedes=True))
    assert store.get_node_state(old.node_id) == "superseded"  # retired again
    assert "old-x" not in {d["slug"] for d in store.get_active_decisions()}


# --------------------------------------------------------------------------- #
# DoD #13 — V1a suite green with the 7 new types folded in (contract row T13)
# --------------------------------------------------------------------------- #


def test_dod13_seven_non_kill_types_commit_without_retiring_endpoints(store) -> None:
    """The seven non-kill edge types commit without retiring endpoints; the two kill-edges still do.

    The "7 folded into the must-not-regress set" proof: a single committed decision
    plus one OQ author a set spanning kill + non-kill + cross-kind. The V1a
    active-view anti-join keys ONLY on ``('supersedes', 'corrects')``, so the catalog
    widening to nine must leave that contract untouched — the two kill-edges retire
    their targets while the seven non-kill types (``amends`` / ``narrows`` /
    ``depends_on`` / ``contradicts`` / ``derives_from`` / ``cites`` / ``resolves``)
    commit as edges with both endpoints still active. (All targets are distinct, so
    the kill+non-kill ``dangling_edge`` trap — a kill and a non-kill edge to the same
    target in one entry — is avoided.)
    """
    # Targets first (stream order = commit order via _commit_md): two kill targets,
    # five non-kill decision targets, h6 for the cross-kind derives_from, and an OQ
    # target for the cross-kind resolves.
    _commit_md(
        store,
        "### kill-a\n**Decided:** A decision to be superseded.\n**Rejected:** None.\n\n"
        "### kill-b\n**Decided:** A decision to be corrected.\n**Rejected:** None.\n\n"
        "### h1\n**Decided:** Amend target.\n**Rejected:** None.\n\n"
        "### h2\n**Decided:** Narrow target.\n**Rejected:** None.\n\n"
        "### h3\n**Decided:** Depends-on target.\n**Rejected:** None.\n\n"
        "### h4\n**Decided:** Contradicts target.\n**Rejected:** None.\n\n"
        "### h5\n**Decided:** Cites target.\n**Rejected:** None.\n\n"
        "### h6\n**Decided:** Derives-from target.\n**Rejected:** None.\n",
    )
    _commit_md(store, "### oq-target\n**Topic:** The question a decision resolves.\n"
               "**Questions:** What does folder-d resolve?\n", kind="open_question")

    # One decision authoring both kill-edges + five non-kill edges + the cross-kind
    # resolves; then one OQ authoring the cross-kind derives_from.
    (folder,) = _commit_md(
        store,
        "### folder-d\n"
        "**Decided:** The folding decision spanning kill and non-kill edges.\n"
        "**Rejected:** Authoring each relation on its own entry.\n"
        "**Supersedes:** kill-a\n"
        "**Corrects:** kill-b\n"
        "**Amends:** h1\n"
        "**Narrows:** h2\n"
        "**Depends-On:** h3\n"
        "**Contradicts:** h4\n"
        "**Cites:** h5\n"
        "**Resolves:** oq-target\n",
    )
    (oq_src,) = _commit_md(
        store,
        "### oq-src\n**Topic:** An OQ deriving from a decision.\n"
        "**Questions:** What does oq-src derive from?\n**Derives-From:** h6\n",
        kind="open_question",
    )

    active_dec = {d["slug"] for d in store.get_active_decisions()}

    # The two kill-edges retired their targets (active view shrank).
    assert "kill-a" not in active_dec
    assert "kill-b" not in active_dec
    assert store.get_node_state(_slug_id_map(store, ["folder-d"])["folder-d"]) == "active"

    # The seven non-kill types committed WITHOUT retiring either endpoint: every
    # non-kill decision target stays active, and folder-d itself stays active.
    assert {"h1", "h2", "h3", "h4", "h5", "h6", "folder-d"} <= active_dec
    # The OQ endpoints are not killed either (resolves sets state, it does not kill):
    oqs = {q["slug"]: q for q in store.get_open_questions()}
    assert set(oqs) == {"oq-target", "oq-src"}
    assert oqs["oq-target"]["state"] == "resolved"  # active resolver folder-d

    # All nine edges present: two kill + seven non-kill. (The kill targets are
    # retired → not slug-resolvable, so their edges are checked off the source below.)
    ids = _slug_id_map(store, ["folder-d", "h1", "h2", "h3", "h4", "h5", "h6", "oq-src"])
    triples = _edge_triples(store)
    non_kill_expected = {
        ("amends", ids["folder-d"], ids["h1"]),
        ("narrows", ids["folder-d"], ids["h2"]),
        ("depends_on", ids["folder-d"], ids["h3"]),
        ("contradicts", ids["folder-d"], ids["h4"]),
        ("cites", ids["folder-d"], ids["h5"]),
        ("derives_from", ids["oq-src"], ids["h6"]),
    }
    assert non_kill_expected <= triples
    # The resolves edge (cross-kind, non-kill) committed; oq-target id via the edge.
    resolves_edges = [e for e in store.get_edges() if e["edge_type"] == "resolves"]
    assert len(resolves_edges) == 1
    assert resolves_edges[0]["source_id"] == ids["folder-d"]
    # The two kill-edges committed from folder-d (targets retired, read off the edge).
    kill_types = {
        e["edge_type"] for e in store.get_edges()
        if e["source_id"] == ids["folder-d"] and e["edge_type"] in ("supersedes", "corrects")
    }
    assert kill_types == {"supersedes", "corrects"}


# --------------------------------------------------------------------------- #
# The V1b shipping act — the single __version__ minor bump (Decision 5)
# --------------------------------------------------------------------------- #


def test_version_is_v1b_minor_bump() -> None:
    """``mitos.__version__`` reads the V1b minor bump (0.3.3 → 0.4.0).

    The one user-facing act of the whole vision: a whole-substrate catalog
    completion is a minor bump (PATTERNS / CLAUDE.md release ritual). Read from the
    single source of truth (``mitos/__init__.py``), pinned once here.
    """
    assert __version__ == "0.4.0"
