"""Integration/fixture suite for the Phase 7a build-aside rebuilder & gate.

``[integration/fixture]`` — these drive the **real** parse→commit path against
temp-dir SQLite files (no mocks, no external services). The old (reference)
prototype graph is built via the retained ``_init_db`` fixture (store.py §16); the
new (aside) graph boots the V1a ladder. Node ids are never hardcoded — they are
recomputed via ``identity.compute_node_id`` or asserted by slug.

Covers §12 SC1–SC11 plus the two gate vacuous-pass guards (G7) and the
serialization roundtrip (§10).
"""

import hashlib
import json
import os
import shutil
import sqlite3
import sys

import pytest

from mitos.cli import cmd_cutover, main as cli_main
from mitos.config import MitosConfig
from mitos.cutover import (
    MissingCore,
    RebuildResult,
    check_reconstruction_completeness,
    default_aside_db_path,
    perform_swap,
    rebuild_and_gate,
)
from mitos.errors import CutoverError, EntryFailure
from mitos.identity import compute_node_id
from mitos.migrations import is_pre_v1a_schema
from mitos.store import GraphStore, compute_hash, open_connection

SENTINEL = "<!-- BEGIN ENTRIES — newest first -->"


# --- corpus authoring helpers --------------------------------------------------


def _decision(
    slug,
    decided,
    *,
    rejected="n/a",
    mechanisms=None,
    supersedes=None,
    corrects=None,
    scope=None,
    omit_rejected=False,
):
    """Builds one decision entry block (fields in spec order)."""
    lines = [f"### {slug}", "", f"**Decided:** {decided}"]
    if not omit_rejected:
        lines.append(f"**Rejected:** {rejected}")
    if mechanisms:
        lines.append(f"**Mechanisms:** {', '.join(mechanisms)}")
    if scope:
        lines.append(f"**Scope:** {', '.join(scope)}")
    if supersedes:
        lines.append(f"**Supersedes:** [{supersedes}]")
    if corrects:
        lines.append(f"**Corrects:** [{corrects}]")
    return "\n".join(lines)


def _oq(slug, topic, questions, *, scope=None):
    """Builds one open-question entry block."""
    lines = [f"### {slug}", "", f"**Topic:** {topic}", f"**Questions:** {questions}"]
    if scope:
        lines.append(f"**Scope:** {', '.join(scope)}")
    return "\n".join(lines)


def _stream(*entries):
    """Joins entry blocks newest-first under the sentinel (as authored on disk)."""
    return SENTINEL + "\n\n" + "\n\n".join(entries) + "\n"


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _config(tmp_path):
    return MitosConfig(str(tmp_path))


def _aside(config):
    return default_aside_db_path(config)


# --- reference (old prototype) graph helper ------------------------------------


def _plant_prototype(db_path, nodes, edges=()):
    """Builds a real pre-V1a prototype graph at ``db_path`` (store.py §16 fixture).

    ``nodes``: dicts with ``slug`` / ``kind`` / ``core_axiom`` / optional
    ``mechanisms`` / ``questions_raised``. The prototype id is minted via the
    retained slug-inclusive ``compute_hash`` (realistic, distinct per slug).
    ``edges``: ``(from_slug, to_slug, type)`` tuples, wired by recomputed id.
    """
    proto = GraphStore.__new__(GraphStore)
    proto.db_path = str(db_path)
    proto.read_only = False
    proto._init_db()

    def _pid(n):
        return compute_hash(
            n["kind"],
            n["slug"],
            n.get("core_axiom", "") or "",
            n.get("mechanisms", []),
            n.get("questions_raised", []),
        )

    by_slug = {n["slug"]: _pid(n) for n in nodes}
    conn = sqlite3.connect(str(db_path))
    try:
        for n in nodes:
            conn.execute(
                "INSERT INTO nodes (id, slug, kind, core_axiom, mechanisms, "
                "questions_raised, source) VALUES (?, ?, ?, ?, ?, ?, 'user')",
                (
                    by_slug[n["slug"]],
                    n["slug"],
                    n["kind"],
                    n.get("core_axiom"),
                    json.dumps(n.get("mechanisms", [])),
                    json.dumps(n.get("questions_raised", [])),
                ),
            )
        for from_slug, to_slug, etype in edges:
            conn.execute(
                "INSERT INTO edges (from_id, to_id, type) VALUES (?, ?, ?)",
                (by_slug[from_slug], by_slug[to_slug], etype),
            )
        conn.commit()
    finally:
        conn.close()


# --- aside-graph readback helpers ----------------------------------------------


def _active_slugs(aside_db_path):
    """Returns ``(active_decision_slugs, active_oq_slugs)`` from the aside graph."""
    store = GraphStore(aside_db_path)
    decs = {n["slug"] for n in store.get_active_decisions()}
    oqs = {n["slug"] for n in store.get_open_questions()}
    return decs, oqs


def _all_slugs(aside_db_path):
    conn = sqlite3.connect(aside_db_path)
    try:
        return {r[0] for r in conn.execute("SELECT slug FROM nodes").fetchall()}
    finally:
        conn.close()


def _pending_slugs(aside_db_path):
    """Returns the slug set behind the aside graph's ``pending_embeddings`` rows."""
    conn = sqlite3.connect(aside_db_path)
    try:
        rows = conn.execute(
            "SELECT n.slug FROM pending_embeddings p JOIN nodes n ON n.id = p.node_id"
        ).fetchall()
        return {r[0] for r in rows}
    finally:
        conn.close()


# --- SC1: happy path -----------------------------------------------------------


def test_sc1_happy_path_rebuilds_and_gate_passes(tmp_path):
    config = _config(tmp_path)
    _write(
        os.path.join(config.archive_dir, "2026-Q1.md"),
        _stream(_decision("alpha", "Alpha axiom.")),
    )
    _write(config.decisions_file, _stream(_decision("beta", "Beta axiom.")))
    # A matching prototype graph so the gate actually compares (not a vacuous pass).
    _plant_prototype(
        config.db_path,
        [
            {"slug": "alpha", "kind": "decision", "core_axiom": "Alpha axiom."},
            {"slug": "beta", "kind": "decision", "core_axiom": "Beta axiom."},
        ],
    )

    result = rebuild_and_gate(config, aside_db_path=_aside(config))

    assert result.decisions_committed == 2
    assert result.open_questions_committed == 0
    assert result.gate_passed is True
    assert result.missing_cores == []
    assert result.reference_active_count == 2
    assert result.reconstructed_active_count == 2
    decs, oqs = _active_slugs(result.aside_db_path)
    assert decs == {"alpha", "beta"}
    assert oqs == set()


# --- SC2: oldest-first ordering resolves a cross-file kill-edge -----------------


def test_sc2_oldest_first_resolves_archive_to_buffer_supersedes(tmp_path):
    config = _config(tmp_path)
    # Archive holds the older (superseded) decision; the buffer supersedes it.
    _write(
        os.path.join(config.archive_dir, "2026-Q1.md"),
        _stream(_decision("old-decision", "Old axiom.")),
    )
    _write(
        config.decisions_file,
        _stream(_decision("new-decision", "New axiom.", supersedes="old-decision")),
    )

    # No missing_target raised → ordering put the target before its superseder.
    result = rebuild_and_gate(config, aside_db_path=_aside(config))

    assert result.decisions_committed == 2
    decs, _ = _active_slugs(result.aside_db_path)
    assert decs == {"new-decision"}  # old-decision is superseded (inactive)
    assert _all_slugs(result.aside_db_path) == {"old-decision", "new-decision"}


# --- SC3: Q5 convergence passes silently ---------------------------------------


def test_sc3_q5_convergence_passes_silently(tmp_path):
    config = _config(tmp_path)
    # Two slugs, one canonical core, NO kill-edge between them → converge to one node.
    _write(
        config.decisions_file,
        _stream(
            _decision("conv-b", "Shared axiom.", mechanisms=["sqlite"]),
            _decision("conv-a", "Shared axiom.", mechanisms=["sqlite"]),
        ),
    )
    # The old graph has BOTH active slugs (distinct prototype ids, same core).
    _plant_prototype(
        config.db_path,
        [
            {
                "slug": "conv-a",
                "kind": "decision",
                "core_axiom": "Shared axiom.",
                "mechanisms": ["sqlite"],
            },
            {
                "slug": "conv-b",
                "kind": "decision",
                "core_axiom": "Shared axiom.",
                "mechanisms": ["sqlite"],
            },
        ],
    )

    result = rebuild_and_gate(config, aside_db_path=_aside(config))

    # Both old slugs recompute to one core id → reference deduped to 1, present in
    # the (single-node) reconstruction → no offender.
    assert result.reference_active_count == 1
    assert result.reconstructed_active_count == 1
    assert result.gate_passed is True
    assert result.missing_cores == []


# --- SC4: Q5 self-edge aborts with cleanup guidance ----------------------------


def test_sc4_q5_self_edge_aborts_with_guidance(tmp_path):
    config = _config(tmp_path)
    # Two same-core entries WITH a kill-edge between them → the edge degenerates to
    # a self-reference once they converge.
    _write(
        config.decisions_file,
        _stream(
            _decision("self-b", "Dup axiom.", supersedes="self-a"),
            _decision("self-a", "Dup axiom."),
        ),
    )
    db_before = config.db_path
    # No live graph planted: the abort must happen on the build-aside copy regardless.

    with pytest.raises(CutoverError) as excinfo:
        rebuild_and_gate(config, aside_db_path=_aside(config))

    msg = str(excinfo.value)
    assert "self-a" in msg and "self-b" in msg
    assert "Drop" in msg  # the G4 drop-the-degenerate-line guidance
    # Live graph untouched (it never existed — the abort did not create it).
    assert not os.path.exists(db_before)


# --- SC5: gate shortfall surfaces, does not raise ------------------------------


def test_sc5_gate_shortfall_surfaces_without_raising(tmp_path):
    config = _config(tmp_path)
    # The buffer rebuilds only alpha; gamma's archive was "dropped" from the corpus.
    _write(config.decisions_file, _stream(_decision("alpha", "Alpha axiom.")))
    # The still-live old graph knows alpha AND gamma are active.
    _plant_prototype(
        config.db_path,
        [
            {"slug": "alpha", "kind": "decision", "core_axiom": "Alpha axiom."},
            {"slug": "gamma", "kind": "decision", "core_axiom": "Gamma axiom."},
        ],
    )

    # A shortfall is a verdict, not an exception.
    result = rebuild_and_gate(config, aside_db_path=_aside(config))

    assert result.gate_passed is False
    assert result.reference_active_count == 2
    assert result.reconstructed_active_count == 1
    assert len(result.missing_cores) == 1
    offender = result.missing_cores[0]
    assert offender.slug == "gamma"
    assert offender.kind == "decision"
    assert "Gamma axiom." in offender.axiom_excerpt
    expected_id = compute_node_id(kind="decision", axiom="Gamma axiom.", mechanism_refs=[])
    assert offender.core_id == expected_id


# --- SC6: missing_target aborts ------------------------------------------------


def test_sc6_missing_target_aborts(tmp_path):
    config = _config(tmp_path)
    _write(
        config.decisions_file,
        _stream(_decision("orphan", "Orphan axiom.", supersedes="ghost")),
    )

    with pytest.raises(CutoverError) as excinfo:
        rebuild_and_gate(config, aside_db_path=_aside(config))

    # The store's referential code rides through onto the envelope.
    failure = excinfo.value.failure
    assert isinstance(failure, EntryFailure)
    assert any(item.code == "missing_target" for item in failure.items)


# --- SC7: parse-stage aggregate abort ------------------------------------------


def test_sc7_parse_stage_aggregate_aborts_with_nothing_committed(tmp_path):
    config = _config(tmp_path)
    # A decision missing the required **Rejected:** field is a format defect.
    _write(
        config.decisions_file,
        _stream(_decision("malformed", "Has axiom.", omit_rejected=True)),
    )
    aside = _aside(config)

    with pytest.raises(CutoverError) as excinfo:
        rebuild_and_gate(config, aside_db_path=aside)

    # The aggregate is a LIST of parse envelopes; nothing was committed (no replay
    # was attempted, so the aside file was never created).
    failure = excinfo.value.failure
    assert isinstance(failure, list)
    assert len(failure) == 1
    assert any(
        item.code == "missing_required_field" for item in failure[0].items
    )
    assert not os.path.exists(aside)


# --- SC8: live graph untouched (passing + aborting runs) -----------------------


def _fingerprint(path):
    st = os.stat(path)
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return digest, st.st_mtime_ns


def test_sc8_live_graph_untouched_on_pass_and_abort(tmp_path):
    config = _config(tmp_path)
    _plant_prototype(
        config.db_path,
        [{"slug": "alpha", "kind": "decision", "core_axiom": "Alpha axiom."}],
    )
    before = _fingerprint(config.db_path)

    # Passing run.
    _write(config.decisions_file, _stream(_decision("alpha", "Alpha axiom.")))
    rebuild_and_gate(config, aside_db_path=_aside(config))
    assert _fingerprint(config.db_path) == before

    # Aborting run (missing_target).
    _write(
        config.decisions_file,
        _stream(_decision("orphan", "Orphan axiom.", supersedes="ghost")),
    )
    with pytest.raises(CutoverError):
        rebuild_and_gate(config, aside_db_path=_aside(config))
    assert _fingerprint(config.db_path) == before


# --- SC9: embedding seed bounded to active (mixed corpus, both kinds) ----------


def test_sc9_embedding_seed_bounded_to_active(tmp_path):
    config = _config(tmp_path)
    # A 3-deep supersedes chain (2 dead, 1 active) + one active open question.
    _write(
        config.decisions_file,
        _stream(
            _decision("d3", "Axiom three.", supersedes="d2"),
            _decision("d2", "Axiom two.", supersedes="d1"),
            _decision("d1", "Axiom one."),
        ),
    )
    _write(config.questions_file, _stream(_oq("q1", "A topic.", "Q one?")))

    result = rebuild_and_gate(config, aside_db_path=_aside(config))

    assert result.decisions_committed == 3
    assert result.open_questions_committed == 1
    # Only the live tip of the chain + the active OQ keep an embedding seed.
    assert _pending_slugs(result.aside_db_path) == {"d3", "q1"}
    decs, oqs = _active_slugs(result.aside_db_path)
    assert decs == {"d3"}
    assert oqs == {"q1"}


# --- SC10: idempotent retry discards a stale/garbage aside ---------------------


def test_sc10_idempotent_retry_discards_stale_aside(tmp_path):
    config = _config(tmp_path)
    _write(
        config.decisions_file,
        _stream(_decision("alpha", "Alpha axiom."), _decision("beta", "Beta axiom.")),
    )
    aside = _aside(config)

    first = rebuild_and_gate(config, aside_db_path=aside)
    first_active = _active_slugs(first.aside_db_path)

    # Simulate a prior crashed run: garbage main file + junk WAL/SHM sidecars.
    _write(aside, "not a sqlite database at all")
    _write(aside + "-wal", "junk")
    _write(aside + "-shm", "junk")

    second = rebuild_and_gate(config, aside_db_path=aside)

    # The garbage was discarded cleanly; the rebuild is identical.
    assert _active_slugs(second.aside_db_path) == first_active == ({"alpha", "beta"}, set())


# --- SC11: empty OQ stream no-op -----------------------------------------------


def test_sc11_absent_questions_file_is_noop(tmp_path):
    config = _config(tmp_path)
    _write(config.decisions_file, _stream(_decision("alpha", "Alpha axiom.")))
    assert not os.path.exists(config.questions_file)

    result = rebuild_and_gate(config, aside_db_path=_aside(config))

    assert result.open_questions_committed == 0
    decs, oqs = _active_slugs(result.aside_db_path)
    assert decs == {"alpha"}
    assert oqs == set()


# --- G7 guards: gate vacuous-passes with no prototype reference -----------------


def test_gate_vacuous_pass_when_old_graph_absent(tmp_path):
    config = _config(tmp_path)
    _write(config.decisions_file, _stream(_decision("alpha", "Alpha axiom.")))
    assert not os.path.exists(config.db_path)

    result = rebuild_and_gate(config, aside_db_path=_aside(config))

    assert result.gate_passed is True
    assert result.reference_active_count == 0


def test_gate_vacuous_pass_when_old_graph_already_v1a(tmp_path):
    config = _config(tmp_path)
    # A fresh V1a (non-prototype) graph at the live path → no prototype reference.
    GraphStore(config.db_path)
    _write(config.decisions_file, _stream(_decision("alpha", "Alpha axiom.")))

    result = rebuild_and_gate(config, aside_db_path=_aside(config))

    assert result.gate_passed is True
    assert result.reference_active_count == 0


def test_gate_old_graph_kill_edge_excludes_superseded(tmp_path):
    """The old-graph reference filter excludes a superseded prototype node (G2/G5).

    Pins the one hand-written anti-join: a prototype node that is the ``to_id`` of a
    ``supersedes`` edge is inactive, so it is NOT part of the reference baseline (no
    false shortfall for an intentionally-superseded core).
    """
    config = _config(tmp_path)
    # beta supersedes alpha in the OLD graph → alpha inactive, beta active.
    _plant_prototype(
        config.db_path,
        [
            {"slug": "alpha", "kind": "decision", "core_axiom": "Alpha axiom."},
            {"slug": "beta", "kind": "decision", "core_axiom": "Beta axiom."},
        ],
        edges=[("beta", "alpha", "supersedes")],
    )
    # The corpus rebuilds only beta; alpha is gone (correctly — it was superseded).
    _write(config.decisions_file, _stream(_decision("beta", "Beta axiom.")))

    result = rebuild_and_gate(config, aside_db_path=_aside(config))

    # alpha is excluded from the reference set, so its absence is NOT a shortfall.
    assert result.reference_active_count == 1
    assert result.gate_passed is True
    assert result.missing_cores == []


def test_check_completeness_is_pure_verdict_no_raise(tmp_path):
    """The gate helper returns a verdict directly (callable in isolation, no raise)."""
    config = _config(tmp_path)
    _plant_prototype(
        config.db_path,
        [{"slug": "alpha", "kind": "decision", "core_axiom": "Alpha axiom."}],
    )
    # An empty reconstruction → alpha is missing, surfaced (not raised).
    missing, ref_count = check_reconstruction_completeness(config, set())
    assert ref_count == 1
    assert [m.slug for m in missing] == ["alpha"]


def test_gate_recomputes_prototype_open_question_core(tmp_path):
    """The gate's OQ reference-recompute branch maps ``topic`` ← prototype ``core_axiom`` (G6).

    The prototype ``nodes`` table has no ``topic`` column — an open question's core
    text lives in the general ``core_axiom`` column — so the gate recomputes an OQ's
    slug-free id via ``compute_node_id(topic=core_axiom, questions_raised=…)``. The
    live corpus has zero open questions, so this branch is fixture-only; this pins it.

    Plant an active prototype OQ + a decision, rebuild only the decision (the OQ is
    "dropped"): the OQ must surface as an ``open_question`` offender whose
    ``core_id`` equals the OQ recompute — proving the branch reads the right column
    and discriminates kind correctly (a decision-branch recompute over the same text
    would yield a different id and never match).
    """
    config = _config(tmp_path)
    _plant_prototype(
        config.db_path,
        [
            {"slug": "alpha", "kind": "decision", "core_axiom": "Alpha axiom."},
            {
                "slug": "oq1",
                "kind": "open_question",
                "core_axiom": "OQ topic.",
                "questions_raised": ["First question?"],
            },
        ],
    )
    # The rebuild keeps only the decision; the OQ corpus is absent (dropped).
    _write(config.decisions_file, _stream(_decision("alpha", "Alpha axiom.")))
    assert not os.path.exists(config.questions_file)

    result = rebuild_and_gate(config, aside_db_path=_aside(config))

    assert result.reference_active_count == 2  # alpha + the recomputed OQ core
    assert result.gate_passed is False
    assert len(result.missing_cores) == 1
    offender = result.missing_cores[0]
    assert offender.slug == "oq1"
    assert offender.kind == "open_question"
    assert "OQ topic." in offender.axiom_excerpt
    # The OQ id is recomputed via the open_question branch (topic ← core_axiom),
    # NOT the decision branch — pin the exact slug-free id.
    expected_id = compute_node_id(
        kind="open_question",
        topic="OQ topic.",
        questions_raised=["First question?"],
    )
    assert offender.core_id == expected_id


# --- serialization roundtrip (§10) ---------------------------------------------


def test_rebuild_result_to_dict_is_json_safe(tmp_path):
    config = _config(tmp_path)
    _write(config.decisions_file, _stream(_decision("alpha", "Alpha axiom.")))
    _plant_prototype(
        config.db_path,
        [
            {"slug": "alpha", "kind": "decision", "core_axiom": "Alpha axiom."},
            {"slug": "gamma", "kind": "decision", "core_axiom": "Gamma axiom."},
        ],
    )

    result = rebuild_and_gate(config, aside_db_path=_aside(config))
    payload = result.to_dict()

    # JSON roundtrips with no tuples and the computed gate flag present.
    restored = json.loads(json.dumps(payload))
    assert restored["gate_passed"] is False
    assert isinstance(restored["missing_cores"], list)
    assert restored["missing_cores"][0]["slug"] == "gamma"
    assert restored["reference_active_count"] == 2


def test_missing_core_to_dict_shape():
    mc = MissingCore(core_id="abc", kind="decision", slug="s", axiom_excerpt="x")
    assert mc.to_dict() == {
        "core_id": "abc",
        "kind": "decision",
        "slug": "s",
        "axiom_excerpt": "x",
    }


def test_rebuild_result_gate_passed_is_computed():
    """``gate_passed`` derives from ``missing_cores`` (M3 — never independently set)."""
    passed = RebuildResult("p", 1, 0, 1, 1, [])
    assert passed.gate_passed is True
    shortfall = RebuildResult(
        "p", 1, 0, 2, 1, [MissingCore("id", "decision", "s", "x")]
    )
    assert shortfall.gate_passed is False


# ==============================================================================
# Phase 7b — atomic swap, WAL-sidecar safety & the `mitos cutover` verb
# ==============================================================================
#
# [integration/fixture] — real temp-dir SQLite, no mocks of the store/file layer.
# Crash injection is deterministic via monkeypatch (no real process kills);
# `timestamp` is always a fixed fixture string so `.bak` paths assert exactly.


def _is_prototype(db_path):
    """Returns True iff ``db_path`` holds a pre-V1a (prototype) graph (RO probe)."""
    conn = open_connection(db_path, read_only=True)
    try:
        return is_pre_v1a_schema(conn)
    finally:
        conn.close()


# A valid standalone V1a node (no FK to satisfy) used to strand a committed frame
# in the aside's -wal — its survival after the swap proves the checkpoint ran.
_SENTINEL_NODE_SQL = (
    "INSERT INTO nodes (id, kind, slug, slug_casefold, source, axiom, "
    "created_at, updated_at) VALUES "
    "('wal-sentinel-id', 'decision', 'wal-sentinel', 'wal-sentinel', 'user', "
    "'WAL sentinel axiom.', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')"
)


def _strand_write_in_wal(db_path, sql, params=()):
    """Leaves a committed write in ``db_path``'s -wal but OUT of the main file.

    Reproduces a pre-checkpoint 'crash' deterministically: with auto-checkpoint
    disabled, commit ``sql`` (the frame lands in -wal, not the main file), snapshot
    the main+wal pair WHILE the writer is open, let the close-time checkpoint fold
    the frame, then restore the pre-checkpoint pair — so the frame again lives only
    in the -wal. A stale -shm is dropped (SQLite rebuilds it from the -wal on open).
    """
    conn = open_connection(db_path)
    try:
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute(sql, params)
        conn.commit()
        shutil.copy2(db_path, db_path + ".mainsnap")
        shutil.copy2(db_path + "-wal", db_path + "-wal.snap")
    finally:
        conn.close()  # close-checkpoint folds the frame into the main file
    os.replace(db_path + ".mainsnap", db_path)
    os.replace(db_path + "-wal.snap", db_path + "-wal")
    try:
        os.remove(db_path + "-shm")
    except FileNotFoundError:
        pass


# --- perform_swap: SC1–SC5 + the missing-aside guard ---------------------------


def test_perform_swap_happy_path(tmp_path):
    """SC1: a passing rebuild swaps into place; the old graph is backed up."""
    config = _config(tmp_path)
    _plant_prototype(
        config.db_path,
        [{"slug": "alpha", "kind": "decision", "core_axiom": "Alpha axiom."}],
    )
    _write(
        config.decisions_file,
        _stream(_decision("alpha", "Alpha axiom."), _decision("beta", "Beta axiom.")),
    )
    aside = _aside(config)
    rebuild_and_gate(config, aside_db_path=aside)

    bak = perform_swap(config, aside, timestamp="20260618-120000")

    # Immediately after the swap (before any open re-creates them): the aside is
    # gone and the new graph carries NO sidecars. Assert this FIRST — opening a
    # write GraphStore for the readback below legitimately re-creates a -wal.
    assert not os.path.exists(aside)
    assert not os.path.exists(config.db_path + "-wal")
    assert not os.path.exists(config.db_path + "-shm")
    # The .bak is the old prototype graph, recognizably a prototype.
    assert bak == config.db_path + ".bak_20260618-120000"
    assert os.path.exists(bak)
    assert _is_prototype(bak)
    # The live graph is now the rebuilt V1a graph with the full active set.
    assert not _is_prototype(config.db_path)
    decs, oqs = _active_slugs(config.db_path)
    assert decs == {"alpha", "beta"}


def test_perform_swap_clears_destination_wal_orphan(tmp_path):
    """SC2: a stale destination -wal/-shm is cleared before the rename (R11/K3)."""
    config = _config(tmp_path)
    _plant_prototype(
        config.db_path,
        [{"slug": "alpha", "kind": "decision", "core_axiom": "Alpha axiom."}],
    )
    _write(config.decisions_file, _stream(_decision("alpha", "Alpha axiom.")))
    aside = _aside(config)
    rebuild_and_gate(config, aside_db_path=aside)

    # The R11 hazard: a stale orphan -wal beside the destination would be applied
    # to the swapped-in graph on next open → SQLITE_CORRUPT.
    _write(config.db_path + "-wal", "stale orphan wal frames")
    _write(config.db_path + "-shm", "stale shm")

    perform_swap(config, aside, timestamp="20260618-000000")

    assert not os.path.exists(config.db_path + "-wal")
    assert not os.path.exists(config.db_path + "-shm")
    decs, oqs = _active_slugs(config.db_path)
    assert decs == {"alpha"}


def test_perform_swap_folds_aside_wal_checkpoint(tmp_path):
    """SC3: the aside's un-checkpointed -wal frames are folded in before the rename."""
    config = _config(tmp_path)
    _write(
        config.decisions_file,
        _stream(_decision("alpha", "Alpha axiom."), _decision("beta", "Beta axiom.")),
    )
    aside = _aside(config)
    rebuild_and_gate(config, aside_db_path=aside)

    # Strand a committed node only in the aside's -wal (a pre-checkpoint crash):
    # if perform_swap clears the -wal without checkpointing first, it is lost.
    _strand_write_in_wal(aside, _SENTINEL_NODE_SQL)
    assert os.path.exists(aside + "-wal")

    perform_swap(config, aside, timestamp="20260618-000000")

    # No -wal beside the swapped-in graph, yet the stranded node survived → the
    # TRUNCATE checkpoint folded it into the main file BEFORE the rename (K3/G2).
    assert not os.path.exists(config.db_path + "-wal")
    all_slugs = _all_slugs(config.db_path)
    assert "wal-sentinel" in all_slugs
    assert {"alpha", "beta"} <= all_slugs


def test_perform_swap_crash_mid_rename_leaves_prototype_valid(tmp_path, monkeypatch):
    """SC4: a crash at the atomic rename leaves the prototype intact; re-run recovers."""
    config = _config(tmp_path)
    _plant_prototype(
        config.db_path,
        [{"slug": "alpha", "kind": "decision", "core_axiom": "Alpha axiom."}],
    )
    _write(config.decisions_file, _stream(_decision("alpha", "Alpha axiom.")))
    before = _fingerprint(config.db_path)
    result = rebuild_and_gate(config, aside_db_path=_aside(config))

    def _raise_oserror(*a, **k):
        raise OSError("simulated crash at the atomic rename")

    monkeypatch.setattr(os, "rename", _raise_oserror)
    with pytest.raises(OSError):
        perform_swap(config, result.aside_db_path, timestamp="20260618-000000")
    monkeypatch.undo()

    # Copy-not-move (K2): config.db_path is never absent — it is still the intact
    # prototype, byte-for-byte (the rename is the only destructive step).
    assert _fingerprint(config.db_path) == before
    assert _is_prototype(config.db_path)

    # A clean re-run recovers with no manual file surgery (P5 Unplugged).
    result2 = rebuild_and_gate(config, aside_db_path=_aside(config))
    perform_swap(config, result2.aside_db_path, timestamp="20260618-000001")
    assert not _is_prototype(config.db_path)
    decs, oqs = _active_slugs(config.db_path)
    assert decs == {"alpha"}


def test_perform_swap_crash_discards_orphan_aside_and_sidecars(tmp_path):
    """SC5: an orphan aside (+ stale sidecars) from a crashed run is discarded, then swap."""
    config = _config(tmp_path)
    _plant_prototype(
        config.db_path,
        [
            {"slug": "alpha", "kind": "decision", "core_axiom": "Alpha axiom."},
            {"slug": "beta", "kind": "decision", "core_axiom": "Beta axiom."},
        ],
    )
    _write(
        config.decisions_file,
        _stream(_decision("alpha", "Alpha axiom."), _decision("beta", "Beta axiom.")),
    )
    aside = _aside(config)
    # A prior 'crashed' run left an orphan aside main + stale -wal/-shm sidecars.
    _write(aside, "not a sqlite database")
    _write(aside + "-wal", "junk")
    _write(aside + "-shm", "junk")

    # rebuild_and_gate discards the orphan (+ its sidecars) and rebuilds cleanly...
    result = rebuild_and_gate(config, aside_db_path=aside)
    # ...then a clean swap follows with no manual step.
    bak = perform_swap(config, result.aside_db_path, timestamp="20260618-000000")

    assert not os.path.exists(aside)
    assert not os.path.exists(config.db_path + "-wal")
    assert not os.path.exists(config.db_path + "-shm")
    assert _is_prototype(bak)
    decs, oqs = _active_slugs(config.db_path)
    assert decs == {"alpha", "beta"}


def test_perform_swap_missing_aside_raises(tmp_path):
    """The defensive guard: a missing build-aside file raises CutoverError."""
    config = _config(tmp_path)
    with pytest.raises(CutoverError):
        perform_swap(config, _aside(config), timestamp="20260618-000000")


# --- cmd_cutover: SC6–SC10 + the --json payload --------------------------------


def test_cmd_cutover_gate_pass_swaps(tmp_path, capsys):
    """SC6: a gate-passing cutover swaps and prints the post-swap runbook (exit 0)."""
    config = _config(tmp_path)
    _plant_prototype(
        config.db_path,
        [{"slug": "alpha", "kind": "decision", "core_axiom": "Alpha axiom."}],
    )
    _write(config.decisions_file, _stream(_decision("alpha", "Alpha axiom.")))

    rc = cmd_cutover(config, allow_drops=False, assume_yes=True, as_json=False)

    assert rc == 0
    out = capsys.readouterr().out
    assert ".bak_" in out                 # names the backup
    assert "curl -X DELETE" in out        # the Qdrant-wipe guidance
    assert "mitos sync" in out            # the re-sync guidance
    assert not _is_prototype(config.db_path)
    decs, oqs = _active_slugs(config.db_path)
    assert decs == {"alpha"}


def test_cmd_cutover_shortfall_refuses_then_allow_drops_overrides(tmp_path, capsys):
    """SC7: a shortfall refuses without --allow-drops; --allow-drops (P6) overrides."""
    config = _config(tmp_path)
    _plant_prototype(
        config.db_path,
        [
            {"slug": "alpha", "kind": "decision", "core_axiom": "Alpha axiom."},
            {"slug": "gamma", "kind": "decision", "core_axiom": "Gamma axiom."},
        ],
    )
    # The corpus rebuilds only alpha → gamma would be dropped (a shortfall).
    _write(config.decisions_file, _stream(_decision("alpha", "Alpha axiom.")))

    rc = cmd_cutover(config, allow_drops=False, assume_yes=True, as_json=False)
    assert rc == 1
    out = capsys.readouterr().out
    assert "gamma" in out                 # the offender is surfaced
    assert _is_prototype(config.db_path)  # no swap — still the prototype

    # With --allow-drops (+ --yes): the P6 override swaps.
    rc2 = cmd_cutover(config, allow_drops=True, assume_yes=True, as_json=False)
    assert rc2 == 0
    assert not _is_prototype(config.db_path)
    decs, oqs = _active_slugs(config.db_path)
    assert decs == {"alpha"}


def test_cmd_cutover_corpus_defect_one_line_error_via_main(tmp_path, monkeypatch, capsys):
    """SC8: a corpus defect → one-line `Error:` via main()'s boundary, no swap."""
    config = _config(tmp_path)
    _plant_prototype(
        config.db_path,
        [{"slug": "alpha", "kind": "decision", "core_axiom": "Alpha axiom."}],
    )
    before = _fingerprint(config.db_path)
    # Malformed corpus: a kill-edge to a non-existent target (missing_target).
    _write(
        config.decisions_file,
        _stream(_decision("orphan", "Orphan axiom.", supersedes="ghost")),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MITOS_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(sys, "argv", ["mitos", "cutover", "--yes"])

    with pytest.raises(SystemExit) as exc:
        cli_main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "Error:" in err
    # No swap — the live graph is the untouched prototype.
    assert _fingerprint(config.db_path) == before


def test_cmd_cutover_noop_on_non_prototype(tmp_path, capsys):
    """SC9: an already-V1a graph is a no-op (no rebuild, no swap, exit 0)."""
    config = _config(tmp_path)
    GraphStore(config.db_path)  # a fresh V1a (non-prototype) graph
    _write(config.decisions_file, _stream(_decision("alpha", "Alpha axiom.")))
    before = _fingerprint(config.db_path)

    rc = cmd_cutover(config, allow_drops=False, assume_yes=True, as_json=False)

    assert rc == 0
    out = capsys.readouterr().out
    assert "nothing to cut over" in out.lower()
    # No rebuild, no swap — the graph is byte-for-byte untouched, no aside built.
    assert _fingerprint(config.db_path) == before
    assert not os.path.exists(_aside(config))


def test_cmd_cutover_no_tty_without_yes_refuses(tmp_path, monkeypatch, capsys):
    """SC10: a no-TTY stdin without --yes refuses calmly, never calling input()."""
    config = _config(tmp_path)
    _plant_prototype(
        config.db_path,
        [{"slug": "alpha", "kind": "decision", "core_axiom": "Alpha axiom."}],
    )
    _write(config.decisions_file, _stream(_decision("alpha", "Alpha axiom.")))

    class _FakeStdin:
        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdin", _FakeStdin())

    def _no_input(*a, **k):
        raise AssertionError("input() must never be called on a no-TTY path")

    monkeypatch.setattr("builtins.input", _no_input)

    rc = cmd_cutover(config, allow_drops=False, assume_yes=False, as_json=False)

    assert rc == 1
    out = capsys.readouterr().out
    assert "--yes" in out
    assert _is_prototype(config.db_path)  # no swap


def test_cmd_cutover_json_success_payload(tmp_path, capsys):
    """The --json success payload is a single parseable object (no human text)."""
    config = _config(tmp_path)
    _plant_prototype(
        config.db_path,
        [{"slug": "alpha", "kind": "decision", "core_axiom": "Alpha axiom."}],
    )
    _write(config.decisions_file, _stream(_decision("alpha", "Alpha axiom.")))

    rc = cmd_cutover(config, allow_drops=False, assume_yes=True, as_json=True)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["swapped"] is True
    assert payload["gate_passed"] is True
    assert payload["bak_path"] and ".bak_" in payload["bak_path"]
    assert payload["qdrant_wipe_cmd"].startswith("curl -X DELETE")
