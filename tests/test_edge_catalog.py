"""The nine-type edge catalog — per-edge-type commit matrix (V1b Phase 2a).

The warn-defer→commit flip: V1a's reconciler committed only the two kill-edges
(``supersedes`` / ``corrects``) and warn-deferred the other seven; as of Phase 2a
``_reconcile_edges`` commits all nine, from comma-separated multi-valued
relationship fields. These fixtures pin that catalog end-to-end against real temp
SQLite (no async, no LLM, no embeddings, no mocks — PATTERNS): the per-type commit,
the three cross-kind validity shapes, the ``kind_constraint_violation`` negative
case (DoD #8b — the one adversarial gate the layer removal makes load-bearing), the
multi-valued split, the referential guards on the new types, and the
declarative-mirror DELETE.

Driven via ``parse_entry_stream`` → ``commit_parsed_entry`` for the parse-coupled
cases (multi-valued / bracketed), and via hand-built ``ParsedEntry`` (List[str]
fields) for the matrix precision. The edge-type catalog is sourced from the store
constants, never a retyped literal (Lesson 33).
"""

import os
import tempfile

import pytest

from mitos.errors import CommitError
from mitos.parser import ParsedEntry, parse_entry_stream
from mitos.store import (
    GraphStore,
    _KILL_EDGE_FIELDS,
    _DEFERRED_EDGE_FIELDS,
    _RELATION_TOKENS,
)


@pytest.fixture
def temp_store() -> GraphStore:
    """A temporary file GraphStore booted to the live ladder head."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    store = GraphStore(path)
    yield store
    if os.path.exists(path):
        os.remove(path)


# --- Builders + read helpers ---------------------------------------------------


def _decision(slug: str, axiom: str, **rels) -> ParsedEntry:
    """A hand-built decision; ``rels`` values are List[str] (V1b shape)."""
    e = ParsedEntry("decision", slug, 1, 5)
    e.axiom = axiom
    e.rejected_paths = "An alternative."
    for name, value in rels.items():
        setattr(e, name, value)
    return e


def _oq(slug: str, topic: str, questions=None, **rels) -> ParsedEntry:
    """A hand-built open_question; ``rels`` values are List[str] (V1b shape)."""
    e = ParsedEntry("open_question", slug, 1, 5)
    e.topic = topic
    e.questions_raised = questions or ["A question?"]
    for name, value in rels.items():
        setattr(e, name, value)
    return e


def _edges_between(store: GraphStore, src_id: str, tgt_id: str):
    return [
        e for e in store.get_edges()
        if e["source_id"] == src_id and e["target_id"] == tgt_id
    ]


def _edge_count(store: GraphStore, edge_type: str) -> int:
    return sum(1 for e in store.get_edges() if e["edge_type"] == edge_type)


# The seven non-kill types that the flip lights up. Sourced from the store constant
# so a catalog change can never silently desync this test (Lesson 33).
_NON_KILL = list(_DEFERRED_EDGE_FIELDS)
# The same-kind non-kill types (valid D→D). `cites` is any→any; `resolves`/
# `derives_from` are cross-kind, exercised separately.
_NEW_SAME_KIND = [t for t in _NON_KILL if t not in ("cites", "resolves", "derives_from")]


# --- 1. Per-edge-type commit (the flip) ----------------------------------------


@pytest.mark.parametrize("edge_type", _NEW_SAME_KIND)
def test_new_same_kind_relation_commits(temp_store: GraphStore, edge_type: str) -> None:
    """Each new same-kind (D→D) non-kill type commits exactly one edge, target stays active."""
    target = temp_store.commit_parsed_entry(_decision("target", "Target axiom."))
    source = temp_store.commit_parsed_entry(
        _decision("source", "Source axiom.", **{edge_type: ["target"]})
    )
    rows = _edges_between(temp_store, source.node_id, target.node_id)
    assert [r["edge_type"] for r in rows] == [edge_type]
    assert rows[0]["source_kind"] == "decision"
    assert rows[0]["target_kind"] == "decision"
    # Non-kill: neither endpoint is retired.
    assert temp_store.get_node_state(target.node_id) == "active"
    assert temp_store.get_node_state(source.node_id) == "active"


def test_catalog_covers_exactly_nine_types() -> None:
    """The reconciler's iterated catalog is exactly the nine the widened CHECK enforces."""
    catalog = set(_KILL_EDGE_FIELDS) | set(_DEFERRED_EDGE_FIELDS)
    assert catalog == {
        "supersedes", "corrects", "amends", "narrows", "depends_on",
        "contradicts", "derives_from", "cites", "resolves",
    }
    # Every type has a display token for its FailureItem.field (P3 vector error).
    assert set(_RELATION_TOKENS) == catalog


def test_relation_tokens_match_canonical_field_labels() -> None:
    """``_RELATION_TOKENS`` never drifts from ``sync._EXTRA_RELATIONS`` field labels.

    The store is a lower tier than ``sync`` and keeps its OWN field-label copy
    rather than importing one (the import edge would invert the tier order). This
    cross-checks the two retyped copies so a relabel on either side fails loudly —
    the "pinned by the test suite so the two can never drift" guarantee the store's
    ``_RELATION_TOKENS`` comment promises (Lesson 33 join-key discipline).
    """
    from mitos.sync import _EXTRA_RELATIONS

    for name, label in _EXTRA_RELATIONS:
        assert _RELATION_TOKENS[name] == f"**{label}:**"
    # The two kill-edges complete the nine (their labels are the buffer anchors the
    # ``record_decision_entry`` serializer writes — ``**Supersedes:**``/``**Corrects:**``).
    assert _RELATION_TOKENS["supersedes"] == "**Supersedes:**"
    assert _RELATION_TOKENS["corrects"] == "**Corrects:**"


# --- 2. Cross-kind validity (the three) ----------------------------------------


def test_cites_any_to_any_all_accepted(temp_store: GraphStore) -> None:
    """`cites` connects any→any: D→D, D→OQ, OQ→D, OQ→OQ all commit."""
    d1 = temp_store.commit_parsed_entry(_decision("d1", "Decision one."))
    oq1 = temp_store.commit_parsed_entry(_oq("oq1", "Topic one."))
    # D→D and D→OQ from one decision source.
    d_src = temp_store.commit_parsed_entry(
        _decision("d-src", "Citing decision.", cites=["d1", "oq1"])
    )
    assert {r["edge_type"] for r in _edges_between(temp_store, d_src.node_id, d1.node_id)} == {"cites"}
    assert {r["edge_type"] for r in _edges_between(temp_store, d_src.node_id, oq1.node_id)} == {"cites"}
    # OQ→D and OQ→OQ from one open-question source.
    oq_src = temp_store.commit_parsed_entry(
        _oq("oq-src", "Citing question.", cites=["d1", "oq1"])
    )
    assert {r["edge_type"] for r in _edges_between(temp_store, oq_src.node_id, d1.node_id)} == {"cites"}
    assert {r["edge_type"] for r in _edges_between(temp_store, oq_src.node_id, oq1.node_id)} == {"cites"}


def test_derives_from_open_question_to_decision_commits(temp_store: GraphStore) -> None:
    """`derives_from` is OQ→D: an open question deriving from a decision commits."""
    host = temp_store.commit_parsed_entry(_decision("host", "Host decision."))
    oq = temp_store.commit_parsed_entry(
        _oq("derived-q", "Derived topic.", derives_from=["host"])
    )
    rows = _edges_between(temp_store, oq.node_id, host.node_id)
    assert [r["edge_type"] for r in rows] == ["derives_from"]
    assert rows[0]["source_kind"] == "open_question"
    assert rows[0]["target_kind"] == "decision"


def test_resolves_decision_to_open_question_commits(temp_store: GraphStore) -> None:
    """`resolves` is D→OQ: a decision resolving an open question commits."""
    oq = temp_store.commit_parsed_entry(_oq("open-q", "An open topic."))
    resolver = temp_store.commit_parsed_entry(
        _decision("resolver", "Resolving decision.", resolves=["open-q"])
    )
    rows = _edges_between(temp_store, resolver.node_id, oq.node_id)
    assert [r["edge_type"] for r in rows] == ["resolves"]
    assert rows[0]["source_kind"] == "decision"
    assert rows[0]["target_kind"] == "open_question"


# --- 3. Negative kind-constraint — DoD #8b (the load-bearing adversarial gate) --


def test_resolves_decision_to_decision_is_kind_violation(temp_store: GraphStore) -> None:
    """A `resolves` D→D trips the widened CHECK → kind_constraint_violation, full rollback.

    This is the canonical #8b case the warn-defer removal makes load-bearing: the
    code maps the DDL rejection to the right vector, names the field, and the entry
    rolls back (no partial edge, no orphan node).
    """
    temp_store.commit_parsed_entry(_decision("a-decision", "A decision."))
    bad = _decision("bad", "Bad resolver.", resolves=["a-decision"])
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(bad)
    items = exc.value.failure.items
    assert items[0].code == "kind_constraint_violation"
    assert items[0].field == "**Resolves:**"
    # Whole-entry rollback: no edge, no orphan 'bad' node.
    assert _edge_count(temp_store, "resolves") == 0
    assert temp_store.resolve_slug("bad") == []


def test_decision_source_derives_from_is_kind_violation(temp_store: GraphStore) -> None:
    """`derives_from` is OQ→D, so a decision-source `derives_from` is a kind violation."""
    temp_store.commit_parsed_entry(_decision("host", "Host decision."))
    bad = _decision("bad", "Bad deriver.", derives_from=["host"])
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(bad)
    assert exc.value.failure.items[0].code == "kind_constraint_violation"
    assert exc.value.failure.items[0].field == "**Derives-From:**"
    assert temp_store.resolve_slug("bad") == []


# --- 4. Multi-valued (comma-separated) -----------------------------------------


def test_cites_multivalued_commits_one_edge_per_slug(temp_store: GraphStore) -> None:
    """`Cites: a, b` commits two cites edges; a lone slug commits one."""
    temp_store.commit_parsed_entry(_decision("a", "Axiom A."))
    temp_store.commit_parsed_entry(_decision("b", "Axiom B."))
    text = (
        "### hub\n"
        "**Decided:** Hub axiom.\n"
        "**Rejected:** None.\n"
        "**Cites:** a, b\n"
    )
    hub = temp_store.commit_parsed_entry(parse_entry_stream(text, "decision")[0])
    targets = {r["target_id"] for r in temp_store.get_edges() if r["edge_type"] == "cites"}
    assert len(targets) == 2
    assert all(r["source_id"] == hub.node_id for r in temp_store.get_edges() if r["edge_type"] == "cites")


def test_supersedes_multivalued_commits_two_kill_edges(temp_store: GraphStore) -> None:
    """`Supersedes: x, y` commits two kill-edges (the lineage-cluster case), both retired."""
    x = temp_store.commit_parsed_entry(_decision("x", "Axiom X."))
    y = temp_store.commit_parsed_entry(_decision("y", "Axiom Y."))
    text = (
        "### merger\n"
        "**Decided:** Merger axiom.\n"
        "**Rejected:** None.\n"
        "**Supersedes:** x, y\n"
    )
    temp_store.commit_parsed_entry(parse_entry_stream(text, "decision")[0])
    assert _edge_count(temp_store, "supersedes") == 2
    assert temp_store.get_node_state(x.node_id) == "superseded"
    assert temp_store.get_node_state(y.node_id) == "superseded"


def test_bracketed_multivalued_citations_resolve(temp_store: GraphStore) -> None:
    """`Cites: [a], [b]` (individually bracketed) → two cites edges (bracket stripped per slug)."""
    temp_store.commit_parsed_entry(_decision("a", "Axiom A."))
    temp_store.commit_parsed_entry(_decision("b", "Axiom B."))
    text = (
        "### hub\n"
        "**Decided:** Hub axiom.\n"
        "**Rejected:** None.\n"
        "**Cites:** [a], [b]\n"
    )
    temp_store.commit_parsed_entry(parse_entry_stream(text, "decision")[0])
    assert _edge_count(temp_store, "cites") == 2


def test_repeated_citation_collapses_to_one_edge(temp_store: GraphStore) -> None:
    """`Cites: a, a` is a within-field repeat → exactly one edge (no spurious PK dup)."""
    temp_store.commit_parsed_entry(_decision("a", "Axiom A."))
    text = (
        "### dup\n"
        "**Decided:** Dup axiom.\n"
        "**Rejected:** None.\n"
        "**Cites:** a, a\n"
    )
    temp_store.commit_parsed_entry(parse_entry_stream(text, "decision")[0])
    assert _edge_count(temp_store, "cites") == 1


# --- 5. Referential guards on the new types ------------------------------------


def test_new_type_missing_target_rejects(temp_store: GraphStore) -> None:
    """A `cites` to a nonexistent slug → missing_target, whole-entry rollback."""
    bad = _decision("citer", "Citing axiom.", cites=["ghost-slug"])
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(bad)
    item = next(i for i in exc.value.failure.items if i.field == "**Cites:**")
    assert item.code == "missing_target"
    assert temp_store.resolve_slug("citer") == []


def test_new_type_dangling_edge_rejects(temp_store: GraphStore) -> None:
    """A `cites` to a superseded-only slug → dangling_edge (cite the live version)."""
    temp_store.commit_parsed_entry(_decision("old", "Old axiom."))
    temp_store.commit_parsed_entry(_decision("new", "New axiom.", supersedes=["old"]))
    # "old" now has only an inactive carrier.
    bad = _decision("citer", "Citing axiom.", cites=["old"])
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(bad)
    item = next(i for i in exc.value.failure.items if i.field == "**Cites:**")
    assert item.code == "dangling_edge"


def test_new_type_self_edge_is_cycle_violation(temp_store: GraphStore) -> None:
    """An entry citing its own slug → cycle_violation (an entry cannot relate to itself)."""
    bad = _decision("selfie", "Self axiom.", cites=["selfie"])
    with pytest.raises(CommitError) as exc:
        temp_store.commit_parsed_entry(bad)
    item = next(i for i in exc.value.failure.items if i.field == "**Cites:**")
    assert item.code == "cycle_violation"


def test_non_kill_edge_mints_from_superseded_source(temp_store: GraphStore) -> None:
    """A superseded source may still add a non-kill edge (the source-active reject is kill-only).

    The kill-edge "itself superseded" guard (V1-D6) does NOT apply to the seven
    non-kill types — a commentary update on a superseded node adding a `Cites:` is
    legal (Decision 2). It re-commits the SAME core (so the same node), now also
    citing a live target.
    """
    target = temp_store.commit_parsed_entry(_decision("target", "Target axiom."))
    old = temp_store.commit_parsed_entry(_decision("old", "Old axiom."))
    temp_store.commit_parsed_entry(_decision("new", "New axiom.", supersedes=["old"]))
    assert temp_store.get_node_state(old.node_id) == "superseded"
    # Re-commit 'old' (same canonical core → same node) now citing the live target.
    re_old = temp_store.commit_parsed_entry(_decision("old", "Old axiom.", cites=["target"]))
    assert re_old.node_id == old.node_id  # same node (core unchanged)
    rows = _edges_between(temp_store, old.node_id, target.node_id)
    assert [r["edge_type"] for r in rows] == ["cites"]


# --- 6. Authoring matrix — every edge type authors from a valid source ----------


def test_every_edge_type_authors_from_a_valid_source(temp_store: GraphStore) -> None:
    """All nine types author an edge from a kind-valid source (DoD #13 store-level slice)."""
    d1 = temp_store.commit_parsed_entry(_decision("d1", "Decision one."))
    d2 = temp_store.commit_parsed_entry(_decision("d2", "Decision two."))
    oq1 = temp_store.commit_parsed_entry(_oq("oq1", "Topic one."))
    # The six same-kind types (D→D) + cites (D→OQ) + resolves (D→OQ) from a decision.
    temp_store.commit_parsed_entry(_decision(
        "rich", "Rich decision.",
        supersedes=["d1"], amends=["d2"], narrows=["d2"],
        depends_on=["d2"], contradicts=["d2"], cites=["oq1"], resolves=["oq1"],
    ))
    # derives_from (OQ→D) authors from an open-question source.
    temp_store.commit_parsed_entry(_oq("derived", "Derived topic.", derives_from=["d2"]))
    committed = {e["edge_type"] for e in temp_store.get_edges()}
    assert committed == {
        "supersedes", "amends", "narrows", "depends_on",
        "contradicts", "cites", "resolves", "derives_from",
    }
    # 'corrects' is the ninth — authored separately (it would conflict with supersedes
    # on the same source in one entry); prove it commits on its own.
    temp_store.commit_parsed_entry(_decision("fixer", "Fixer decision.", corrects=["d2"]))
    assert "corrects" in {e["edge_type"] for e in temp_store.get_edges()}


# --- 7. Declarative-mirror for the new types -----------------------------------


def test_removing_cites_line_deletes_edge_no_resurrection(temp_store: GraphStore) -> None:
    """Re-committing without a previously-declared `Cites:` DELETEs the edge (declarative mirror).

    A removed non-kill edge resurrects nothing (it retired no target), so the
    declarative-mirror DELETE does not feed the resurrection slug-collision re-check
    (Decision 5) — a clean re-commit, no spurious collision error.
    """
    temp_store.commit_parsed_entry(_decision("a", "Axiom A."))
    src = temp_store.commit_parsed_entry(_decision("src", "Source axiom.", cites=["a"]))
    assert _edge_count(temp_store, "cites") == 1
    # Re-commit the SAME core (same axiom → same node) with the cites line removed.
    re_src = temp_store.commit_parsed_entry(_decision("src", "Source axiom."))
    assert re_src.node_id == src.node_id
    assert _edge_count(temp_store, "cites") == 0


# --- 8. Symmetric `contradicts` bidirectional accessor (Phase 2c, DoD #9) -------
#
# `contradicts` is the catalog's one semantically-symmetric edge: stored ONCE as
# authored (A as source_id, B as target_id, never mirrored), yet "A contradicts B"
# means the same as "B contradicts A". `get_contradictions(node_id)` is the single
# safe bidirectional read so no future consumer hand-rolls
# `WHERE source_id = X OR target_id = X` and silently under-reports by forgetting
# the `OR` (the OD1 hazard). These gates prove the read from BOTH ends (#9), the
# cross-direction dedup, and the v0.1 no-active-view-action pin — the three
# negatives that make the seam trustworthy (a happy-path "it returns something"
# proves nothing).


def test_contradicts_bidirectional_read_dod9(temp_store: GraphStore) -> None:
    """DoD #9: a `contradicts` authored A→B is returned from EITHER endpoint."""
    b = temp_store.commit_parsed_entry(_decision("b", "B axiom."))
    a = temp_store.commit_parsed_entry(_decision("a", "A axiom.", contradicts=["b"]))
    # Returned when queried from A (the authoring source)...
    assert [c["node_id"] for c in temp_store.get_contradictions(a.node_id)] == [b.node_id]
    # ...AND from B (the target). Neither call hand-rolls a directional query.
    assert [c["node_id"] for c in temp_store.get_contradictions(b.node_id)] == [a.node_id]


def test_contradicts_oq_to_oq_bidirectional(temp_store: GraphStore) -> None:
    """OQ→OQ `contradicts` is a real authored shape (same-kind); reads from both ends."""
    oq2 = temp_store.commit_parsed_entry(_oq("oq2", "Topic two."))
    oq1 = temp_store.commit_parsed_entry(_oq("oq1", "Topic one.", contradicts=["oq2"]))
    from_oq1 = temp_store.get_contradictions(oq1.node_id)
    assert [c["node_id"] for c in from_oq1] == [oq2.node_id]
    assert [c["node_id"] for c in temp_store.get_contradictions(oq2.node_id)] == [oq1.node_id]
    # Counterpart kind is the queried kind (same-kind only) — informational, not a discriminator.
    assert from_oq1[0]["kind"] == "open_question"


def test_contradicts_multiple_counterparts_each_once(temp_store: GraphStore) -> None:
    """A node contradicting B and C returns both, each exactly once."""
    b = temp_store.commit_parsed_entry(_decision("b", "B axiom."))
    c = temp_store.commit_parsed_entry(_decision("c", "C axiom."))
    a = temp_store.commit_parsed_entry(_decision("a", "A axiom.", contradicts=["b", "c"]))
    result = temp_store.get_contradictions(a.node_id)
    assert {r["node_id"] for r in result} == {b.node_id, c.node_id}
    assert len(result) == 2  # no duplicates


def test_contradicts_dedup_across_directions(temp_store: GraphStore) -> None:
    """Both `A→B` and `B→A` authored → the counterpart appears ONCE (UNION, not UNION ALL).

    The cross-direction dedup tripwire: querying A finds B via the source_id=A
    direction AND via the target_id=A direction — B must collapse to one entry.
    """
    a = temp_store.commit_parsed_entry(_decision("a", "A axiom."))
    # b→a ...
    b = temp_store.commit_parsed_entry(_decision("b", "B axiom.", contradicts=["a"]))
    # ...and a→b (re-commit 'a' on the SAME core → same node, now also contradicting b).
    re_a = temp_store.commit_parsed_entry(_decision("a", "A axiom.", contradicts=["b"]))
    assert re_a.node_id == a.node_id
    # Both directional edges exist...
    assert _edge_count(temp_store, "contradicts") == 2
    # ...but each endpoint reports its counterpart exactly once.
    assert [r["node_id"] for r in temp_store.get_contradictions(a.node_id)] == [b.node_id]
    assert [r["node_id"] for r in temp_store.get_contradictions(b.node_id)] == [a.node_id]


def test_contradicts_returned_regardless_of_endpoint_liveness(temp_store: GraphStore) -> None:
    """No-liveness pin (Decision 3): superseding an endpoint does NOT drop the contradiction.

    v0.1 takes no active-view action on `contradicts` — the liveness treatment is
    v0.2 Conflict's. This gate would regress loudly if someone later bolts a
    liveness filter onto the accessor.
    """
    b = temp_store.commit_parsed_entry(_decision("b", "B axiom."))
    a = temp_store.commit_parsed_entry(_decision("a", "A axiom.", contradicts=["b"]))
    # Supersede A via a kill-edge.
    temp_store.commit_parsed_entry(_decision("killer", "Killer axiom.", supersedes=["a"]))
    assert temp_store.get_node_state(a.node_id) == "superseded"
    # The contradiction is still returned from the live endpoint B...
    assert [r["node_id"] for r in temp_store.get_contradictions(b.node_id)] == [a.node_id]
    # ...and from the superseded endpoint A (the edge itself is untouched).
    assert [r["node_id"] for r in temp_store.get_contradictions(a.node_id)] == [b.node_id]


def test_contradicts_empty_and_absent_return_empty_list(temp_store: GraphStore) -> None:
    """A node with no `contradicts` edges, and a non-existent id, both return `[]`."""
    lonely = temp_store.commit_parsed_entry(_decision("lonely", "Lonely axiom."))
    assert temp_store.get_contradictions(lonely.node_id) == []
    assert temp_store.get_contradictions("0" * 64) == []  # absent id is not an error


def test_contradicts_return_shape_is_identity_only(temp_store: GraphStore) -> None:
    """Each entry carries exactly {node_id, slug, kind} — no axiom/payload, nothing stamped."""
    b = temp_store.commit_parsed_entry(_decision("b", "B axiom."))
    a = temp_store.commit_parsed_entry(_decision("a", "A axiom.", contradicts=["b"]))
    result = temp_store.get_contradictions(a.node_id)
    assert result == [{"node_id": b.node_id, "slug": "b", "kind": "decision"}]


def test_contradicts_isolation_other_edge_types_do_not_leak(temp_store: GraphStore) -> None:
    """Only `contradicts` edges are returned — `cites`/`amends`/`depends_on` don't leak."""
    b = temp_store.commit_parsed_entry(_decision("b", "B axiom."))
    c = temp_store.commit_parsed_entry(_decision("c", "C axiom."))
    d = temp_store.commit_parsed_entry(_decision("d", "D axiom."))
    # 'hub' contradicts b, but also cites c and amends d (distinct targets, no kill edge).
    hub = temp_store.commit_parsed_entry(_decision(
        "hub", "Hub axiom.", contradicts=["b"], cites=["c"], amends=["d"],
    ))
    assert [r["node_id"] for r in temp_store.get_contradictions(hub.node_id)] == [b.node_id]
    # The cited/amended nodes report no contradiction with the hub.
    assert temp_store.get_contradictions(c.node_id) == []
    assert temp_store.get_contradictions(d.node_id) == []


def test_contradicts_self_edge_defense_never_returns_self(temp_store: GraphStore) -> None:
    """A corrupt out-of-band self-edge (`A contradicts A`) never makes A return itself.

    The supported write path rejects `A contradicts A` as `cycle_violation`, so a
    self-edge can never be authored. But the graph is a rebuildable derivative
    (M7/P6) that out-of-band corruption could leave holding one, so the accessor
    carries `!= ?` guards. This pins them load-bearing: inject a raw self-edge,
    bypassing the reconciler, and assert it never surfaces as A's own contradiction
    while a legitimate counterpart still does.
    """
    b = temp_store.commit_parsed_entry(_decision("b", "B axiom."))
    a = temp_store.commit_parsed_entry(_decision("a", "A axiom.", contradicts=["b"]))
    # Inject a raw self-edge directly, bypassing the reconciler that would reject it.
    conn = temp_store._get_connection()
    try:
        conn.execute(
            "INSERT INTO edges "
            "(source_id, source_kind, target_id, target_kind, edge_type, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?);",
            (a.node_id, "decision", a.node_id, "decision", "contradicts",
             "2026-06-23T00:00:00.000000+00:00"),
        )
        conn.commit()
    finally:
        conn.close()
    # The corrupt self-edge IS present in the raw edge set (the injection took)...
    assert any(
        e["source_id"] == a.node_id and e["target_id"] == a.node_id
        and e["edge_type"] == "contradicts"
        for e in temp_store.get_edges()
    )
    # ...but the accessor never returns A as its own contradiction — only B.
    assert [r["node_id"] for r in temp_store.get_contradictions(a.node_id)] == [b.node_id]
