"""Tests for the Phase 5a Mechanism Registry — auto-registration writer,
first-seen-wins identity, and the pre-commit feedback read.

Phase 5a gives a decision's cited mechanism tokens (``sqlite``, ``wal-mode``,
``ReactRouter``) first-class existence: the first time anyone cites a mechanism,
``commit_parsed_entry`` auto-registers it as a ``mechanisms`` row INSIDE the
per-entry transaction (MI-5), keyed by its slugified ``canonical_name``,
recording its first-seen presentation casing (``authored_name``) and provenance
(``source``) — the lost-if-not-captured datum v0.2's Drift sensor reads (V1-D5 /
V1-D15). A read-only ``get_unregistered_mechanisms`` ships gated for V3a.

Deterministic, keyless, parse→commit (``commit_parsed_entry`` never embeds — no
LLM/async/embed mocks). Read methods over the ``mechanisms`` table go through raw
SQL on the store's own connection (the ``_node_row`` idiom from ``test_store``).

The casing/first-seen-wins gates are driven through ``parse_entry_stream`` ON
PURPOSE: a hand-built ``ParsedEntry`` does not populate ``mechanisms_authored``
(defaulting it to ``[]``), so the authored-name resolution would silently fall
back to the ref itself rather than exercising the real canonical-MATCH path. Only
the real parse path proves the canonical-match logic that recovers first-seen
casing from the folded authoritative set.
"""

import os
import sqlite3
import tempfile

import pytest

from mitos.store import GraphStore
from mitos.identity import mechanism_canonical_norm
from mitos.errors import CommitError
from mitos.parser import ParsedEntry, parse_entry_stream


@pytest.fixture
def temp_store() -> GraphStore:
    """Initializes a temporary file GraphStore (boots the V1b schema ladder)."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    store = GraphStore(path)
    yield store
    if os.path.exists(path):
        os.remove(path)


# --- Raw-SQL read helpers (read methods over ``mechanisms`` land later) --------


def _mechanisms(store: GraphStore):
    """Reads all ``mechanisms`` rows as dicts via raw SQL, ordered by canonical_name."""
    conn = store._get_connection()
    try:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT canonical_name, authored_name, source, created_at "
                "FROM mechanisms ORDER BY canonical_name"
            )
        ]
    finally:
        conn.close()


def _count(store: GraphStore, table: str) -> int:
    """Counts rows in a table via raw SQL (table name is a code-internal literal)."""
    conn = store._get_connection()
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def _decision(
    slug: str = "d-slug",
    axiom: str = "An axiom.",
    rejected: str = "An alternative.",
    mechanisms=None,
    source=None,
    context=None,
) -> ParsedEntry:
    """Builds a hand-made decision ``ParsedEntry`` (mirrors ``test_store._decision``).

    NOTE: like ``test_store``'s helper, this sets ``mechanisms`` to the RAW unfolded
    value and leaves ``mechanisms_authored`` at its ``[]`` default — fine for the
    atomicity / empty-fold gates that do not depend on authored-casing resolution,
    but the casing/first-seen-wins gates use ``parse_entry_stream`` instead (see the
    module docstring).
    """
    e = ParsedEntry("decision", slug, 1, 5)
    e.axiom = axiom
    e.rejected_paths = rejected
    e.mechanisms = list(mechanisms) if mechanisms else []
    e.source = source
    e.context = context
    return e


def _parse_decision(slug: str, axiom: str, mechanisms: str, extra: str = "") -> ParsedEntry:
    """Parses one decision through the REAL fold path (populates ``mechanisms_authored``).

    ``mechanisms`` is the raw comma-separated ``**Mechanisms:**`` value as a human
    would author it (e.g. ``"ReactRouter"`` or ``"sqlite, wal-mode"``).
    """
    text = (
        f"### {slug}\n"
        f"**Decided:** {axiom}\n"
        "**Rejected:** an alternative\n"
        f"**Mechanisms:** {mechanisms}\n"
        f"{extra}"
    )
    entries = parse_entry_stream(text, "decision")
    assert len(entries) == 1
    return entries[0]


# --- Gate 1: Auto-registration (W13 / W1) -------------------------------------


def test_novel_mechanism_mints_exactly_one_row(temp_store: GraphStore) -> None:
    """A decision citing a novel mechanism mints exactly one row, keyed by canonical_name."""
    entry = _parse_decision("d1", "an axiom", "sqlite")
    temp_store.commit_parsed_entry(entry)

    rows = _mechanisms(temp_store)
    assert len(rows) == 1
    assert rows[0]["canonical_name"] == mechanism_canonical_norm("sqlite") == "sqlite"
    assert rows[0]["authored_name"] == "sqlite"


def test_multiple_distinct_mechanisms_each_register(temp_store: GraphStore) -> None:
    """Every distinct cited mechanism in one entry registers its own row."""
    entry = _parse_decision("d1", "an axiom", "sqlite, qdrant, gemini")
    temp_store.commit_parsed_entry(entry)

    assert {r["canonical_name"] for r in _mechanisms(temp_store)} == {
        "sqlite",
        "qdrant",
        "gemini",
    }


# --- Gate 2: First-seen-wins identity + authored casing (V1-D15) ---------------


def test_first_seen_casing_survives_across_entries(temp_store: GraphStore) -> None:
    """Two entries citing the same mechanism in different casing → one row, FIRST casing.

    Drives both commits through ``parse_entry_stream`` so the authored-name resolution
    exercises the real canonical-MATCH path (not the hand-built ref-fallback).
    """
    first = _parse_decision("d1", "axiom one", "ReactRouter")
    temp_store.commit_parsed_entry(first)
    after_first = _mechanisms(temp_store)
    assert len(after_first) == 1
    assert after_first[0]["canonical_name"] == "reactrouter"
    assert after_first[0]["authored_name"] == "ReactRouter"

    # A DIFFERENT decision (different axiom → different node id) cites the same
    # mechanism in folded casing. INSERT OR IGNORE no-ops on the PK collision.
    second = _parse_decision("d2", "axiom two", "reactrouter")
    temp_store.commit_parsed_entry(second)

    after_second = _mechanisms(temp_store)
    assert len(after_second) == 1
    # The whole row is the FIRST entry's — casing, source, and created_at all stick.
    assert after_second == after_first


def test_canonical_convergence_punct_and_case(temp_store: GraphStore) -> None:
    """``LINT:wal`` then ``lint-wal`` converge to one canonical, first casing wins."""
    temp_store.commit_parsed_entry(_parse_decision("d1", "axiom one", "LINT:wal"))
    temp_store.commit_parsed_entry(_parse_decision("d2", "axiom two", "lint-wal"))

    rows = _mechanisms(temp_store)
    assert len(rows) == 1
    assert rows[0]["canonical_name"] == "lint-wal"
    assert rows[0]["authored_name"] == "LINT:wal"  # first-seen presentation casing


def test_authored_name_first_token_wins_within_entry(temp_store: GraphStore) -> None:
    """Within one entry, the FIRST authored token folding to a canonical wins its casing."""
    # Folded set dedups to {sqlite, wal-mode}; authored order is preserved raw, so the
    # first token folding to each canonical is the one recorded.
    entry = _parse_decision("d1", "an axiom", "WAL Mode, wal-mode, SQLite, sqlite")
    temp_store.commit_parsed_entry(entry)

    by_canonical = {r["canonical_name"]: r["authored_name"] for r in _mechanisms(temp_store)}
    assert by_canonical == {"wal-mode": "WAL Mode", "sqlite": "SQLite"}


# --- Gate 3: MI-5 / DoD #2 — commentary re-commit mints/mutates nothing --------


def test_commentary_recommit_mints_no_row_mutates_none(temp_store: GraphStore) -> None:
    """A same-core commentary update mints no new mechanism row and mutates no existing one."""
    first = _parse_decision("d1", "an axiom", "redis", extra="**Context:** first context\n")
    temp_store.commit_parsed_entry(first)
    before = _mechanisms(temp_store)
    assert len(before) == 1

    # Same canonical core (axiom + mechanisms unchanged), changed Context → same node
    # id → is_new is False → the registration path is never entered.
    second = _parse_decision("d1", "an axiom", "redis", extra="**Context:** SECOND context\n")
    temp_store.commit_parsed_entry(second)

    assert _count(temp_store, "nodes") == 1  # confirms a same-id commit, not a new node
    assert _mechanisms(temp_store) == before  # no new row, no mutation


def test_byte_identical_recommit_is_idempotent(temp_store: GraphStore) -> None:
    """Replaying a byte-identical decision no-ops the registry (MI-3, §14 idempotency CC).

    The first commit registers; the replay is a same-id (not ``is_new``) commit, so
    the registration path is never entered — and even if it were, ``INSERT OR IGNORE``
    on the PK would no-op. Either way the registry is unchanged.
    """
    temp_store.commit_parsed_entry(_parse_decision("d1", "an axiom", "elasticsearch"))
    before = _mechanisms(temp_store)
    temp_store.commit_parsed_entry(_parse_decision("d1", "an axiom", "elasticsearch"))
    assert _mechanisms(temp_store) == before
    assert len(before) == 1


# --- Gate 4: source first-seen-wins -------------------------------------------


def test_source_first_seen_wins(temp_store: GraphStore) -> None:
    """A ``capture_llm`` first-coin then a ``user`` re-cite → registry source stays capture_llm."""
    temp_store.commit_parsed_entry(
        _parse_decision("d1", "axiom one", "kafka", extra="**Source:** capture_llm\n")
    )
    temp_store.commit_parsed_entry(
        _parse_decision("d2", "axiom two", "kafka", extra="**Source:** user\n")
    )

    rows = _mechanisms(temp_store)
    assert len(rows) == 1
    assert rows[0]["source"] == "capture_llm"  # earliest-authoring actor's provenance


# --- Gate 5: ``mechanisms.source`` enum coverage (§6.2 Lesson 13) -------------


@pytest.mark.parametrize(
    "source_line, expected",
    [
        ("**Source:** user\n", "user"),
        ("**Source:** capture_llm\n", "capture_llm"),
        ("**Source:** import_llm\n", "import_llm"),
        ("", "user"),  # absent -> "user"
    ],
)
def test_mechanism_source_enum_coverage_through_parse(
    temp_store: GraphStore, source_line: str, expected: str
) -> None:
    """All three mechanisms.source enum values (plus absent→user) flow through parse→commit."""
    entry = _parse_decision("m-src", "an axiom for source coverage", "datadog", extra=source_line)
    temp_store.commit_parsed_entry(entry)

    rows = _mechanisms(temp_store)
    assert len(rows) == 1
    assert rows[0]["source"] == expected


# --- Gate 6: ``mechanisms.source`` DDL-CHECK rejection (raw insert) -----------


def test_mechanism_source_out_of_enum_rejected_by_ddl_check(temp_store: GraphStore) -> None:
    """A raw ``mechanisms`` row whose source is outside the enum is rejected by the DDL CHECK.

    Raw insert (not parse→commit): the nodes-table CHECK already rejects a bad
    ``source`` at the node INSERT before ``_register_mechanisms`` runs, so the
    mechanisms CHECK can only be reached out-of-band — mirror ``test_migrations``.
    """
    conn = temp_store._get_connection()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO mechanisms (canonical_name, authored_name, source, created_at) "
                "VALUES (?, ?, ?, ?);",
                ("lint:wal", "LINT:wal", "banana", "2026-06-23T00:00:00.000000+00:00"),
            )
    finally:
        conn.close()


# --- Gate 7: Feedback read (W14 / T2 second clause) ---------------------------


def test_feedback_read_returns_exact_unregistered_subset(temp_store: GraphStore) -> None:
    """``get_unregistered_mechanisms`` returns EXACTLY the not-yet-registered subset."""
    temp_store.commit_parsed_entry(_parse_decision("d1", "an axiom", "sqlite"))

    probe = _parse_decision("d2", "another axiom", "sqlite, redis, qdrant")
    result = temp_store.get_unregistered_mechanisms(probe)

    assert set(result) == {"redis", "qdrant"}  # sqlite already registered, excluded


def test_feedback_read_empty_when_all_registered(temp_store: GraphStore) -> None:
    """``get_unregistered_mechanisms`` returns [] when every cited mechanism is registered."""
    temp_store.commit_parsed_entry(_parse_decision("d1", "an axiom", "sqlite, redis"))

    probe = _parse_decision("d2", "another axiom", "redis, sqlite")
    assert temp_store.get_unregistered_mechanisms(probe) == []


def test_feedback_read_all_when_registry_empty(temp_store: GraphStore) -> None:
    """With an empty registry, every cited mechanism is reported unregistered (authored form)."""
    probe = _parse_decision("d1", "an axiom", "Redis, Kafka")
    assert set(temp_store.get_unregistered_mechanisms(probe)) == {"Redis", "Kafka"}


def test_feedback_read_empty_for_open_question(temp_store: GraphStore) -> None:
    """An open question carries no mechanisms → the feedback read returns [] (decision-gated)."""
    oq = ParsedEntry("open_question", "oq-slug", 1, 5)
    oq.topic = "a topic"
    oq.questions_raised = ["A question?"]
    assert temp_store.get_unregistered_mechanisms(oq) == []


def test_feedback_read_registers_nothing(temp_store: GraphStore) -> None:
    """The feedback read is read-only: calling it never writes a registry row."""
    probe = _parse_decision("d1", "an axiom", "redis, qdrant")
    temp_store.get_unregistered_mechanisms(probe)
    assert _count(temp_store, "mechanisms") == 0


# --- Gate 8: Empty-fold skip --------------------------------------------------


def test_empty_fold_token_registers_no_row(temp_store: GraphStore) -> None:
    """A mechanism that folds to '' (``!!!``) mints no junk ``canonical_name=''`` row."""
    entry = _parse_decision("d1", "an axiom", "!!!")
    temp_store.commit_parsed_entry(entry)
    assert _count(temp_store, "mechanisms") == 0


def test_empty_fold_skipped_valid_still_registers(temp_store: GraphStore) -> None:
    """A junk empty-fold token is skipped while a valid sibling mechanism still registers."""
    entry = _parse_decision("d1", "an axiom", "sqlite, !!!")
    temp_store.commit_parsed_entry(entry)

    rows = _mechanisms(temp_store)
    assert len(rows) == 1
    assert rows[0]["canonical_name"] == "sqlite"


def test_feedback_read_skips_empty_fold(temp_store: GraphStore) -> None:
    """The feedback read never reports an empty-fold token as unregistered."""
    probe = _parse_decision("d1", "an axiom", "redis, !!!")
    assert temp_store.get_unregistered_mechanisms(probe) == ["redis"]


# --- Gate 9: Atomicity — registration rolls back with a failed entry (MI-5) ----


def test_mechanism_registration_rolls_back_with_failed_entry(temp_store: GraphStore) -> None:
    """A novel mechanism on an entry that later trips a store failure rolls back too.

    Provoke-the-failure (P10): a control decision registers ``keepmech`` (persists),
    then a second decision cites a NOVEL ``novelmech`` AND a ``Supersedes:`` to a
    nonexistent slug — the ``missing_target`` failure raises ``CommitError`` inside
    the single ``with conn:`` AFTER the registration write, so the whole entry
    (mechanism row included) rolls back. The control row proves the rollback is
    scoped to the failed entry, not a blanket "nothing registered".
    """
    control = _decision(slug="ok", axiom="ax ok", mechanisms=["keepmech"])
    temp_store.commit_parsed_entry(control)
    assert _count(temp_store, "mechanisms") == 1

    bad = _decision(slug="bad", axiom="ax bad", mechanisms=["novelmech"])
    bad.supersedes = ["does-not-exist"]  # forward ref → missing_target store failure
    with pytest.raises(CommitError):
        temp_store.commit_parsed_entry(bad)

    rows = _mechanisms(temp_store)
    assert len(rows) == 1
    assert rows[0]["canonical_name"] == "keepmech"  # novelmech rolled back; control intact
    assert _count(temp_store, "nodes") == 1  # the failed node rolled back too


# --- Open question carries no mechanisms (decision-gated writer) ---------------


def test_open_question_registers_no_mechanism(temp_store: GraphStore) -> None:
    """An open_question commit registers nothing (the writer is decision-gated)."""
    oq = ParsedEntry("open_question", "oq-slug", 1, 5)
    oq.topic = "a topic"
    oq.questions_raised = ["A question?"]
    temp_store.commit_parsed_entry(oq)
    assert _count(temp_store, "mechanisms") == 0
