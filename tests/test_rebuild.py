"""Integration suite for ``mitos rebuild`` (corpus rebuild & upgrade UX).

``[integration]`` — drives the real parse→commit→gate→swap path against temp-dir
SQLite files (no mocks, no external services). Covers the shared replay+fixpoint
engine, the resilient-vs-strict per-caller policy, the current-graph completeness
gate, the ``cmd_rebuild`` verb (swap / refusal / prototype redirect), CLI routing,
and the ``mitos status`` graph-behind-buffer nudge.
"""

import os
import sys

import pytest

from mitos import cli
from mitos.cli import cmd_rebuild, main as cli_main
from mitos.config import MitosConfig
from mitos.cutover import default_aside_db_path, rebuild_and_gate
from mitos.errors import CutoverError
from mitos.parser import parse_entry_stream
from mitos.store import GraphStore, open_connection

SENTINEL = "<!-- BEGIN ENTRIES — newest first -->"


# --- corpus authoring helpers (local; tests/ is not a package) -----------------


def _decision(
    slug,
    decided,
    *,
    rejected="n/a",
    mechanisms=None,
    supersedes=None,
    cites=None,
    amends=None,
    depends_on=None,
):
    """Builds one decision entry block (fields in spec order)."""
    lines = [f"### {slug}", "", f"**Decided:** {decided}", f"**Rejected:** {rejected}"]
    if mechanisms:
        lines.append(f"**Mechanisms:** {', '.join(mechanisms)}")
    if supersedes:
        lines.append(f"**Supersedes:** [{supersedes}]")
    if amends:
        lines.append(f"**Amends:** [{amends}]")
    if cites:
        lines.append(f"**Cites:** [{cites}]")
    if depends_on:
        lines.append(f"**Depends-On:** [{depends_on}]")
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


def _commit(store, *blocks):
    """Commits each raw entry block into ``store`` via the real parse→commit path."""
    for block in blocks:
        for entry in parse_entry_stream(block, "decision"):
            store.commit_parsed_entry(entry)


def _edge_counts(db_path):
    conn = open_connection(db_path, read_only=True)
    try:
        return dict(conn.execute("SELECT edge_type, COUNT(*) FROM edges GROUP BY edge_type"))
    finally:
        conn.close()


def _mechanism_count(db_path):
    conn = open_connection(db_path, read_only=True)
    try:
        return conn.execute("SELECT COUNT(*) FROM mechanisms").fetchone()[0]
    finally:
        conn.close()


def _qdrant(reachable=True, collection_exists=True, points=1):
    return lambda url, coll: {
        "reachable": reachable,
        "collection_exists": collection_exists,
        "points": points,
    }


# --- the headline: a rebuild populates the full catalog ------------------------


def test_rebuild_populates_full_catalog_from_corpus(tmp_path):
    """A corpus authoring amends/cites/Mechanisms rebuilds into the full catalog.

    The upgrade-path win: a graph that warn-deferred the V1b edges (so it holds only
    nodes + kill-edges, 0 mechanisms) gains the catalog by re-committing the corpus
    through the current path. Asserted on the build-aside graph (no live graph → the
    gate vacuous-passes; this isolates the catalog-population behaviour).
    """
    config = _config(tmp_path)
    _write(
        config.decisions_file,
        _stream(
            _decision("ref", "References base.", cites="base"),
            _decision("ext", "Extends base.", mechanisms=["m1", "m2"], amends="base"),
            _decision("base", "Base axiom.", mechanisms=["m1"]),
        ),
    )

    result = rebuild_and_gate(config, aside_db_path=_aside(config), strict=False)

    assert result.decisions_committed == 3
    assert result.residual_casualties == []
    edges = _edge_counts(result.aside_db_path)
    assert edges.get("amends", 0) == 1
    assert edges.get("cites", 0) == 1
    # First-seen-wins: m1 (from base) + m2 (from ext) → 2 distinct registry rows.
    assert _mechanism_count(result.aside_db_path) == 2


# --- the demo's casualty: a cites-to-superseded target -------------------------


def test_rebuild_quarantines_dangling_cites_and_flags_shortfall(tmp_path):
    """The demo case: an entry citing a since-superseded node is a casualty.

    The live graph holds ``citer`` active (the warn-defer artifact — it committed
    under 0.3.x WITHOUT its cites edge). The buffer authors ``citer`` WITH the cites
    edge to a node that is now superseded, so a clean rebuild rejects it
    (``dangling_edge``). It surfaces both as a residual casualty (why) AND a gate
    shortfall (impact — a live-active node would be dropped); the live graph is
    untouched (rebuild_and_gate never swaps).
    """
    config = _config(tmp_path)
    # Live graph: citer committed WITHOUT its cites edge → it is active.
    live = GraphStore(config.db_path)
    _commit(
        live,
        _decision("old", "Old axiom."),
        _decision("newer", "Newer axiom.", supersedes="old"),
        _decision("citer", "Cites old."),
    )
    live_active = {n["slug"] for n in GraphStore(config.db_path, read_only=True).get_active_decisions()}
    assert {"newer", "citer"} <= live_active

    # Buffer: citer NOW authors the cites edge to old (which is superseded).
    _write(
        config.decisions_file,
        _stream(
            _decision("citer", "Cites old.", cites="old"),
            _decision("newer", "Newer axiom.", supersedes="old"),
            _decision("old", "Old axiom."),
        ),
    )

    result = rebuild_and_gate(config, aside_db_path=_aside(config), strict=False)

    # Why: citer is a dangling_edge casualty.
    casualty = next((c for c in result.residual_casualties if c.slug == "citer"), None)
    assert casualty is not None
    assert "dangling_edge" in casualty.codes
    # Impact: the gate (baselined on the live graph) flags citer as dropped.
    assert result.gate_passed is False
    assert "citer" in {mc.slug for mc in result.missing_cores}
    # The live graph is untouched — rebuild_and_gate only writes the aside.
    still_active = {n["slug"] for n in GraphStore(config.db_path, read_only=True).get_active_decisions()}
    assert "citer" in still_active


# --- forward-ref at scale converges in one rebuild -----------------------------


def test_rebuild_converges_deep_chain_in_one_pass(tmp_path):
    """A deep cites chain authored newest-first converges in ONE rebuild.

    Oldest-first replay + the fixpoint commit the whole chain regardless of authoring
    order (the 7/129 → full-corpus difference): every target lands before its citer.
    """
    config = _config(tmp_path)
    # d10 cites d9, …, d2 cites d1; authored newest-first (d10 on top).
    chain = [
        _decision(f"d{i}", f"Axiom {i}.", cites=(f"d{i-1}" if i > 1 else None))
        for i in range(10, 0, -1)
    ]
    _write(config.decisions_file, _stream(*chain))

    result = rebuild_and_gate(config, aside_db_path=_aside(config), strict=False)

    assert result.decisions_committed == 10
    assert result.residual_casualties == []
    assert _edge_counts(result.aside_db_path).get("cites", 0) == 9


# --- per-caller policy over the SAME defect corpus -----------------------------


def test_per_caller_policy_strict_raises_resilient_surfaces(tmp_path):
    """The same corpus: strict (cutover) raises; resilient (rebuild) surfaces."""
    config = _config(tmp_path)
    _write(
        config.decisions_file,
        _stream(_decision("orphan", "Orphan axiom.", supersedes="ghost")),
    )

    # strict=True (the cutover contract) — a casualty aborts.
    with pytest.raises(CutoverError) as excinfo:
        rebuild_and_gate(config, aside_db_path=_aside(config), strict=True)
    assert "orphan" in str(excinfo.value)

    # strict=False (the rebuild contract) — the same casualty is surfaced.
    result = rebuild_and_gate(config, aside_db_path=_aside(config), strict=False)
    casualty = next((c for c in result.residual_casualties if c.slug == "orphan"), None)
    assert casualty is not None
    assert "missing_target" in casualty.codes


# --- the gate baselines against the current (non-prototype) graph --------------


def test_gate_baselines_against_current_v1b_graph(tmp_path):
    """The generalized gate: a current-graph-active node dropped by the rebuild is flagged.

    Today the gate vacuous-passes on a non-prototype graph; generalized, it reads the
    live graph's active set as the baseline (reference_active_count == 2, not 0) so a
    dropped decision surfaces as a shortfall.
    """
    config = _config(tmp_path)
    live = GraphStore(config.db_path)
    _commit(live, _decision("alpha", "Alpha axiom."), _decision("beta", "Beta axiom."))
    # The buffer rebuilds only alpha; beta was dropped from the corpus.
    _write(config.decisions_file, _stream(_decision("alpha", "Alpha axiom.")))

    result = rebuild_and_gate(config, aside_db_path=_aside(config), strict=False)

    assert result.reference_active_count == 2  # current graph baseline, NOT a vacuous 0
    assert result.gate_passed is False
    assert [mc.slug for mc in result.missing_cores] == ["beta"]


# --- the cmd_rebuild verb ------------------------------------------------------


def test_cmd_rebuild_clean_swaps_and_backs_up(tmp_path, monkeypatch):
    """A clean rebuild swaps the graph in and leaves a .bak reversal."""
    config = _config(tmp_path)
    _commit(GraphStore(config.db_path), _decision("alpha", "Alpha axiom.", mechanisms=["m1"]))
    _write(
        config.decisions_file,
        _stream(_decision("alpha", "Alpha axiom.", mechanisms=["m1"])),
    )

    rc = cmd_rebuild(config, allow_drops=False, assume_yes=True, as_json=False)

    assert rc == 0
    baks = [f for f in os.listdir(config.mitos_dir) if f.startswith("graph.sqlite.bak_")]
    assert baks, "perform_swap should leave a timestamped backup (binary reversal)"
    # The swapped-in graph is intact and still active.
    active = {n["slug"] for n in GraphStore(config.db_path, read_only=True).get_active_decisions()}
    assert "alpha" in active


def test_cmd_rebuild_refuses_casualty_without_allow_drops(tmp_path):
    """A casualty blocks the swap (no --allow-drops): live graph untouched, no .bak."""
    config = _config(tmp_path)
    live = GraphStore(config.db_path)
    _commit(
        live,
        _decision("old", "Old axiom."),
        _decision("newer", "Newer axiom.", supersedes="old"),
        _decision("citer", "Cites old."),
    )
    _write(
        config.decisions_file,
        _stream(
            _decision("citer", "Cites old.", cites="old"),
            _decision("newer", "Newer axiom.", supersedes="old"),
            _decision("old", "Old axiom."),
        ),
    )

    rc = cmd_rebuild(config, allow_drops=False, assume_yes=True, as_json=False)

    assert rc == 1
    assert not [f for f in os.listdir(config.mitos_dir) if f.startswith("graph.sqlite.bak_")]
    # Live graph untouched — citer still active.
    active = {n["slug"] for n in GraphStore(config.db_path, read_only=True).get_active_decisions()}
    assert "citer" in active


def test_cmd_rebuild_allow_drops_proceeds(tmp_path):
    """--allow-drops accepts the casualty and swaps (the drop stays in the markdown)."""
    config = _config(tmp_path)
    live = GraphStore(config.db_path)
    _commit(
        live,
        _decision("old", "Old axiom."),
        _decision("newer", "Newer axiom.", supersedes="old"),
        _decision("citer", "Cites old."),
    )
    _write(
        config.decisions_file,
        _stream(
            _decision("citer", "Cites old.", cites="old"),
            _decision("newer", "Newer axiom.", supersedes="old"),
            _decision("old", "Old axiom."),
        ),
    )

    rc = cmd_rebuild(config, allow_drops=True, assume_yes=True, as_json=False)

    assert rc == 0
    assert [f for f in os.listdir(config.mitos_dir) if f.startswith("graph.sqlite.bak_")]
    # citer was dropped from the rebuilt graph (still in the markdown buffer).
    active = {n["slug"] for n in GraphStore(config.db_path, read_only=True).get_active_decisions()}
    assert "citer" not in active


def test_cmd_rebuild_redirects_prototype_to_cutover(tmp_path, monkeypatch):
    """A pre-V1a prototype graph is cutover's job — rebuild refuses and redirects."""
    config = _config(tmp_path)
    GraphStore(config.db_path)  # a real (V1b) graph so db_path exists
    monkeypatch.setattr(cli, "is_pre_v1a_schema", lambda conn: True)

    rc = cmd_rebuild(config, allow_drops=False, assume_yes=True, as_json=False)

    assert rc == 1


def test_cmd_rebuild_no_graph_returns_error(tmp_path):
    """No graph at the workspace → an actionable error (run init), exit 1."""
    config = _config(tmp_path)
    rc = cmd_rebuild(config, allow_drops=False, assume_yes=True, as_json=False)
    assert rc == 1


# --- CLI routing ---------------------------------------------------------------


def test_cli_routes_rebuild_to_cmd_rebuild(tmp_path, monkeypatch):
    """`mitos rebuild` dispatches to cmd_rebuild with its flags bound."""
    captured = {}

    def _fake_rebuild(config, *, allow_drops, assume_yes, as_json):
        captured.update(allow_drops=allow_drops, assume_yes=assume_yes, as_json=as_json)
        return 0

    monkeypatch.setattr(cli, "cmd_rebuild", _fake_rebuild)
    monkeypatch.setattr(sys, "argv", ["mitos", "rebuild", "--allow-drops", "--yes", "--json"])
    with pytest.raises(SystemExit) as exc:
        cli_main()
    assert exc.value.code == 0
    assert captured == {"allow_drops": True, "assume_yes": True, "as_json": True}


# --- the mitos status graph-behind-buffer nudge --------------------------------


def test_status_nudges_when_graph_behind_buffer(tmp_path, monkeypatch, capsys):
    """A V1b-schema graph with an empty registry but mechanism-bearing nodes nudges."""
    cli.cmd_init(MitosConfig(str(tmp_path)))
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant())
    config = MitosConfig(str(tmp_path))
    _commit(GraphStore(config.db_path), _decision("d1", "Axiom.", mechanisms=["m1"]))
    # The migrated-but-not-rebuilt signature: schema present, registry never committed.
    conn = open_connection(config.db_path)
    conn.execute("DELETE FROM mechanisms")
    conn.commit()
    conn.close()

    assert cli.cmd_status(str(tmp_path)) == 0  # informational — still READY
    out = capsys.readouterr().out
    assert "behind your buffer" in out
    assert "mitos rebuild" in out


def test_status_no_nudge_when_registry_populated(tmp_path, monkeypatch, capsys):
    """A graph whose registry IS populated does not nudge (no false positive)."""
    cli.cmd_init(MitosConfig(str(tmp_path)))
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", _qdrant())
    config = MitosConfig(str(tmp_path))
    _commit(GraphStore(config.db_path), _decision("d1", "Axiom.", mechanisms=["m1"]))

    assert cli.cmd_status(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "behind your buffer" not in out


# --- the refused-rebuild remediation guidance (upgrade-path UX) ----------------


def test_cmd_rebuild_refusal_prints_actionable_remediation(tmp_path, capsys):
    """A refused rebuild guides the user: safe + per-class fix + --allow-drops escape.

    A stranger upgrading must not hit a bare ``refused`` wall — they need to learn
    their decisions are safe, exactly what to do for a dangling_edge casualty, and
    that --allow-drops is a safe escape.
    """
    config = _config(tmp_path)
    live = GraphStore(config.db_path)
    _commit(
        live,
        _decision("old", "Old axiom."),
        _decision("newer", "Newer axiom.", supersedes="old"),
        _decision("citer", "Cites old."),
    )
    _write(
        config.decisions_file,
        _stream(
            _decision("citer", "Cites old.", cites="old"),
            _decision("newer", "Newer axiom.", supersedes="old"),
            _decision("old", "Old axiom."),
        ),
    )

    rc = cmd_rebuild(config, allow_drops=False, assume_yes=True, as_json=False)
    out = capsys.readouterr().out

    assert rc == 1
    assert "nothing is lost" in out          # reassurance
    assert "source of truth" in out
    assert "dangling_edge" in out            # per-class fix
    assert "active successor" in out
    assert "--allow-drops" in out            # the safe escape
