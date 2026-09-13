"""Adversarial test suite for the Mitos stateless renderer.

Verifies stateless rendering from primary sources (M8), atomic-write tempfile
swapping, and global vs scope-specific tag segregation.
"""

import tempfile
import os
import pytest
from typing import Tuple
import mitos.renderer as R
from mitos.store import GraphStore
from mitos.parser import ParsedEntry
from mitos.renderer import (
    MitosRenderer, atomic_write, assemble_render, overflow_report,
    summarize_overflows, estimate_tokens,
)

@pytest.fixture
def temp_workspace() -> Tuple[GraphStore, str]:
    """Fixture initializing temporary workspace and GraphStore."""
    workspace_dir = tempfile.mkdtemp()
    db_path = os.path.join(workspace_dir, ".mitos", "graph.sqlite")
    store = GraphStore(db_path)
    yield store, workspace_dir
    # Cleanup
    shutil_rm = True
    if shutil_rm:
        import shutil
        shutil.rmtree(workspace_dir, ignore_errors=True)


def test_atomic_write_safety() -> None:
    """Verifies that atomic_write prevents partial files and works safely."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "dest.txt")
        content = "Secure stateless data."
        atomic_write(filepath, content)
        
        assert os.path.exists(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            assert f.read() == content


def test_render_never_reaches_the_durable_path(temp_workspace: Tuple[GraphStore, str]) -> None:
    """A render is derivative: no fsync, and never the replayed-from entry point.

    SQLite syncs in C and never calls `os.fsync`, so a zero count here is meaningful.
    Nodes are committed before the spies start.
    """
    from unittest.mock import patch
    from mitos import atomic_file

    store, workspace = temp_workspace
    entry = ParsedEntry("decision", "render-only", 1, 5)
    entry.axiom = "Renders regenerate."
    entry.rejected_paths = "Durable renders."
    entry.scope = ["backend"]
    store.commit_parsed_entry(entry)

    with patch("os.fsync", side_effect=os.fsync) as fsync, \
            patch("mitos.atomic_file.write_source",
                  side_effect=atomic_file.write_source) as durable:
        written = MitosRenderer(workspace).render_all(store)

    assert written, "the render wrote nothing, so the zero counts would be vacuous"
    assert fsync.call_count == 0
    assert durable.call_count == 0


def test_renderer_stateless_outputs(temp_workspace: Tuple[GraphStore, str]) -> None:
    """Tests global and per-scope renders against active nodes."""
    store, workspace = temp_workspace
    renderer = MitosRenderer(workspace)

    # Commit active node in scope 'backend'
    entry1 = ParsedEntry("decision", "be-choice", 1, 5)
    entry1.axiom = "We use Python 3.12."
    entry1.rejected_paths = "Older versions."
    entry1.scope = ["backend"]
    store.commit_parsed_entry(entry1)

    # Commit superseded node in scope 'frontend' (should be excluded)
    entry2 = ParsedEntry("decision", "fe-old", 1, 5)
    entry2.axiom = "Vanilla JS."
    entry2.rejected_paths = "React."
    entry2.scope = ["frontend"]
    d2 = store.commit_parsed_entry(entry2)

    entry3 = ParsedEntry("decision", "fe-new", 1, 5)
    entry3.axiom = "Vite + TS."
    entry3.rejected_paths = "Vanilla JS."
    entry3.supersedes = ["fe-old"]
    entry3.scope = ["frontend"]
    store.commit_parsed_entry(entry3)

    # Trigger renders
    renderer.render_all(store)

    # 1. Verify global live_axioms.md
    global_path = os.path.join(workspace, "live_axioms.md")
    assert os.path.exists(global_path)
    with open(global_path, "r", encoding="utf-8") as f:
        global_content = f.read()
        
    assert "be-choice" in global_content
    assert "fe-new" in global_content
    # M3/M8: Superseded nodes must be excluded from active renders
    assert "fe-old" not in global_content

    # 2. Verify per-scope Tag rendering
    be_scope_path = os.path.join(workspace, ".mitos", "axioms", "backend.md")
    fe_scope_path = os.path.join(workspace, ".mitos", "axioms", "frontend.md")
    
    assert os.path.exists(be_scope_path)
    assert os.path.exists(fe_scope_path)

    with open(be_scope_path, "r", encoding="utf-8") as f:
        be_content = f.read()
    assert "be-choice" in be_content
    assert "fe-new" not in be_content

    with open(fe_scope_path, "r", encoding="utf-8") as f:
        fe_content = f.read()
    assert "fe-new" in fe_content
    assert "be-choice" not in fe_content


# --------------------------------------------------------------------------- #
# Size-ceiling overflow: recorded as data, never printed (so it can't bury a receipt)
# --------------------------------------------------------------------------- #

def test_estimate_tokens_heuristic() -> None:
    """estimate_tokens uses the ~4-chars/token floor heuristic."""
    assert estimate_tokens(0) == 0
    assert estimate_tokens(4) == 1
    assert estimate_tokens(401) == 100  # floor division


def test_summarize_overflows_none_singular_plural() -> None:
    """summarize_overflows is None when clean, and pluralises + points at `mitos status`."""
    assert summarize_overflows([]) is None
    one = summarize_overflows([{"name": "substrate.md"}])
    assert one is not None and "1 rendered axiom file " in one and "mitos status" in one
    two = summarize_overflows([{"name": "a.md"}, {"name": "b.md"}])
    assert "2 rendered axiom files " in two


def test_assemble_render_matches_disk(temp_workspace: Tuple[GraphStore, str]) -> None:
    """assemble_render's content is byte-identical to what render_all writes (no drift)."""
    store, workspace = temp_workspace
    e = ParsedEntry("decision", "use-sqlite", 1, 5)
    e.axiom = "We use SQLite in WAL mode."
    e.rejected_paths = "Postgres (too heavy)."
    e.scope = ["substrate"]
    store.commit_parsed_entry(e)

    assembled = assemble_render(store)
    MitosRenderer(workspace).render_all(store)

    with open(os.path.join(workspace, "live_axioms.md"), encoding="utf-8") as f:
        assert f.read() == assembled["global"]["content"]
    with open(os.path.join(workspace, ".mitos", "axioms", "substrate.md"), encoding="utf-8") as f:
        assert f.read() == assembled["scopes"]["substrate"]["content"]


def test_render_all_is_silent_and_records_overflow(
    temp_workspace: Tuple[GraphStore, str], capsys, monkeypatch
) -> None:
    """render_all writes the files, prints nothing, and records the overflow on .overflows."""
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", 150)
    store, workspace = temp_workspace
    e = ParsedEntry("decision", "over-one", 1, 5)
    e.axiom = "Rationale that is comfortably long. " * 12
    e.rejected_paths = "n/a"
    e.scope = ["substrate"]
    store.commit_parsed_entry(e)

    renderer = MitosRenderer(workspace)
    renderer.render_all(store)

    captured = capsys.readouterr()
    assert captured.out == "" and "exceeds" not in captured.err
    names = [o["name"] for o in renderer.overflows]
    assert "substrate.md" in names


def test_overflow_report_ranks_largest_decision_first(
    temp_workspace: Tuple[GraphStore, str], monkeypatch
) -> None:
    """overflow_report flags an over-ceiling scope and ranks its biggest decision first."""
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", 200)
    monkeypatch.setattr(R, "GLOBAL_OVERFLOW_WARN_CHARS", 10_000_000)  # keep the global file out
    store, workspace = temp_workspace

    small = ParsedEntry("decision", "small-one", 1, 5)
    small.axiom = "Tiny axiom."
    small.rejected_paths = "n/a"
    small.scope = ["substrate"]
    store.commit_parsed_entry(small)

    big = ParsedEntry("decision", "big-one", 1, 5)
    big.axiom = "A much larger rationale block. " * 40
    big.rejected_paths = "n/a"
    big.scope = ["substrate"]
    store.commit_parsed_entry(big)

    report = overflow_report(store)
    sub = [o for o in report if o["name"] == "substrate.md"]
    assert len(sub) == 1
    o = sub[0]
    assert o["scope"] == "substrate"
    assert o["chars"] > 200 and o["threshold_chars"] == 200
    assert o["est_tokens"] == o["chars"] // 4
    # Largest decision is ranked first, so an author knows what to re-scope.
    assert o["top_decisions"][0]["slug"] == "big-one"
    assert o["top_decisions"][0]["chars"] >= o["top_decisions"][-1]["chars"]


# --------------------------------------------------------------------------- #
# Primary-tag dedupe (the render-dedupe ADR): full body once, pointers elsewhere
# --------------------------------------------------------------------------- #

def _commit(store: GraphStore, slug: str, scope, axiom: str = None) -> None:
    e = ParsedEntry("decision", slug, 1, 5)
    e.axiom = axiom or f"Axiom for {slug} with enough words to truncate cleanly at a boundary."
    e.rejected_paths = f"Rejected for {slug}."
    e.scope = scope
    store.commit_parsed_entry(e)


def test_multi_tag_full_body_only_under_primary(temp_workspace) -> None:
    """A multi-tag decision renders its full body under its FIRST tag only;
    every secondary tag's file carries a one-line pointer to the primary file."""
    store, workspace = temp_workspace
    _commit(store, "multi-call", ["alpha", "beta", "gamma"])

    assembled = assemble_render(store)
    alpha = assembled["scopes"]["alpha"]["content"]
    beta = assembled["scopes"]["beta"]["content"]
    gamma = assembled["scopes"]["gamma"]["content"]

    # Full body (with its Rejected block) only under the primary tag.
    assert "## multi-call" in alpha and "Rejected for multi-call." in alpha
    for secondary in (beta, gamma):
        assert "## multi-call" not in secondary
        assert "Rejected for multi-call." not in secondary
        assert R.POINTER_SECTION_HEADING in secondary
        assert "multi-call" in secondary
        assert "→ full entry: alpha.md" in secondary
    # The primary file carries no pointer section for this decision.
    assert R.POINTER_SECTION_HEADING not in alpha


def test_single_tag_scope_file_unchanged(temp_workspace) -> None:
    """A single-tag decision's scope file renders exactly as before (no pointers)."""
    store, workspace = temp_workspace
    _commit(store, "solo-call", ["solo"])

    assembled = assemble_render(store)
    content = assembled["scopes"]["solo"]["content"]
    assert "## solo-call" in content and "Rejected for solo-call." in content
    assert R.POINTER_SECTION_HEADING not in content
    assert "→ full entry" not in content


def test_global_file_unaffected_by_dedupe(temp_workspace) -> None:
    """live_axioms.md keeps one full body per decision — no pointers."""
    store, workspace = temp_workspace
    _commit(store, "multi-call", ["alpha", "beta"])
    assembled = assemble_render(store)
    g = assembled["global"]["content"]
    assert g.count("## multi-call") == 1
    assert "Rejected for multi-call." in g
    assert R.POINTER_SECTION_HEADING not in g


def test_pointer_line_truncates_at_word_boundary(temp_workspace) -> None:
    """The pointer's axiom is word-boundary-truncated with an ellipsis."""
    store, workspace = temp_workspace
    long_axiom = "This deliberately long axiom keeps going with many words " * 4
    _commit(store, "long-call", ["prime", "second"], axiom=long_axiom.strip())
    assembled = assemble_render(store)
    second = assembled["scopes"]["second"]["content"]
    pointer = next(l for l in second.splitlines() if l.startswith("- **long-call**"))
    assert "…" in pointer and "→ full entry: prime.md" in pointer
    # No mid-word cut: the char before the ellipsis ends a whole word.
    snippet = pointer.split("— ", 1)[1].split(" → full entry", 1)[0]
    assert snippet.endswith("…")
    assert long_axiom.startswith(snippet[:-1])
    assert long_axiom[len(snippet) - 1] == " "


def test_overflow_accounting_reflects_pointer_weight(temp_workspace, monkeypatch) -> None:
    """A secondary scope's size-contributor list carries the decision at pointer
    weight (one line), not full-body weight — the accounting matches the content."""
    store, workspace = temp_workspace
    big_axiom = "A very heavy rationale block indeed. " * 30
    _commit(store, "heavy-call", ["main", "side"], axiom=big_axiom.strip())
    _commit(store, "side-own", ["side"])

    assembled = assemble_render(store)
    side = assembled["scopes"]["side"]
    sizes = dict(side["decisions"])
    main_sizes = dict(assembled["scopes"]["main"]["decisions"])
    # Pointer weight is a single line — far below the full-body weight.
    assert sizes["heavy-call"] < 200 < main_sizes["heavy-call"]
    # The per-decision sizes sum to less than the file (header + section heading).
    assert sum(sizes.values()) < len(side["content"])
    # And render_all's disk write matches the assembled accounting source.
    MitosRenderer(workspace).render_all(store)
    with open(os.path.join(workspace, ".mitos", "axioms", "side.md"), encoding="utf-8") as f:
        assert f.read() == side["content"]


# --------------------------------------------------------------------------- #
# Global degradation (the global-render-degrades ADR): full under the ceiling,
# oneline index over it — a pure deterministic function of rendered size.
# --------------------------------------------------------------------------- #

def test_under_ceiling_global_is_full_and_bannerless(temp_workspace) -> None:
    """A corpus under the global ceiling renders the unchanged full global file."""
    store, workspace = temp_workspace
    _commit(store, "small-one", ["alpha"])
    _commit(store, "small-two", ["beta"])

    assembled = assemble_render(store)
    g = assembled["global"]
    assert g["mode"] == "full"
    assert g["content"].startswith("# Live Axioms\n")
    assert "Index" not in g["content"]
    assert "## small-one" in g["content"] and "Rejected for small-one." in g["content"]


def test_over_ceiling_global_degrades_to_index(temp_workspace, monkeypatch) -> None:
    """Over the ceiling, the global file is a banner + grouped oneline index."""
    monkeypatch.setattr(R, "GLOBAL_OVERFLOW_WARN_CHARS", 400)
    store, workspace = temp_workspace
    _commit(store, "alpha-one", ["alpha", "beta"])
    _commit(store, "alpha-two", ["alpha"])
    _commit(store, "beta-one", ["beta"])
    untagged = ParsedEntry("decision", "no-scope-one", 1, 5)
    untagged.axiom = "An untagged decision with a perfectly reasonable axiom sentence."
    untagged.rejected_paths = "Rejected for no-scope-one."
    store.commit_parsed_entry(untagged)

    assembled = assemble_render(store)
    g = assembled["global"]
    assert g["mode"] == "index"
    content = g["content"]
    # Banner states plainly what happened and where the full renders live.
    assert content.startswith("# Live Axioms — Index")
    assert "exceeds the global size ceiling" in content
    assert "canonical full renders" in content
    # Grouped by PRIMARY scope tag, each heading pointing at the per-scope file.
    assert "## alpha — full entries: .mitos/axioms/alpha.md" in content
    assert "## beta — full entries: .mitos/axioms/beta.md" in content
    # Multi-tag decision indexes once, under its primary tag's group only.
    assert content.count("**alpha-one**") == 1
    # One row per decision; no full bodies, no rejected_paths.
    for slug in ("alpha-one", "alpha-two", "beta-one", "no-scope-one"):
        assert f"- **{slug}** — " in content
    assert "Rejected for" not in content
    assert "## alpha-one" not in content
    # Untagged decisions gather in the final unscoped group.
    assert "## (unscoped)" in content
    assert content.index("## (unscoped)") > content.index("## beta")
    # render_all writes exactly the assembled index (no drift between the seams).
    MitosRenderer(workspace).render_all(store)
    with open(os.path.join(workspace, "live_axioms.md"), encoding="utf-8") as f:
        assert f.read() == content


def test_index_rows_carry_modifier_markers(temp_workspace, monkeypatch) -> None:
    """An amended-but-active decision's index row carries the compact ⚠ marker."""
    monkeypatch.setattr(R, "GLOBAL_OVERFLOW_WARN_CHARS", 300)
    store, workspace = temp_workspace
    _commit(store, "base-call", ["alpha"])
    amender = ParsedEntry("decision", "amend-call", 1, 5)
    amender.axiom = "We refine the base call with a narrower rule."
    amender.rejected_paths = "n/a"
    amender.scope = ["alpha"]
    amender.amends = ["base-call"]
    store.commit_parsed_entry(amender)

    content = assemble_render(store)["global"]["content"]
    base_row = next(l for l in content.splitlines() if l.startswith("- **base-call**"))
    assert "⚠ amended by: amend-call" in base_row
    amend_row = next(l for l in content.splitlines() if l.startswith("- **amend-call**"))
    assert "⚠" not in amend_row


def test_threshold_boundary_is_deterministic(temp_workspace, monkeypatch) -> None:
    """Exactly-at-ceiling stays full; one char over flips to the index."""
    store, workspace = temp_workspace
    _commit(store, "boundary-call", ["alpha"])
    full_len = len(assemble_render(store)["global"]["content"])

    monkeypatch.setattr(R, "GLOBAL_OVERFLOW_WARN_CHARS", full_len)
    assert assemble_render(store)["global"]["mode"] == "full"
    monkeypatch.setattr(R, "GLOBAL_OVERFLOW_WARN_CHARS", full_len - 1)
    assert assemble_render(store)["global"]["mode"] == "index"


def test_overflow_accounting_in_index_mode(temp_workspace, monkeypatch) -> None:
    """In index mode the global file drops out of overflows (the index fits) —
    but an index that itself breaches the ceiling is still reported honestly."""
    monkeypatch.setattr(R, "GLOBAL_OVERFLOW_WARN_CHARS", 1200)
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", 10_000_000)
    store, workspace = temp_workspace
    for i in range(6):
        # Distinct axioms — identical content hashes to the same node id.
        _commit(store, f"bulk-{i}", ["alpha"],
                axiom=f"A comfortably verbose axiom sentence for overflow test {i}. " * 3)

    # Full render > 1200 → index mode; the index is small → no global overflow entry.
    assembled = assemble_render(store)
    assert assembled["global"]["mode"] == "index"
    assert len(assembled["global"]["content"]) <= 1200
    assert overflow_report(store) == []
    # Accounting reflects index-row weight, not full-body weight.
    assert all(size < 200 for _, size in assembled["global"]["decisions"])

    # Squeeze the ceiling below even the index: the index reports itself honestly.
    monkeypatch.setattr(R, "GLOBAL_OVERFLOW_WARN_CHARS", 200)
    report = overflow_report(store)
    entries = [o for o in report if o["name"] == "live_axioms.md"]
    assert len(entries) == 1
    assert entries[0]["threshold_chars"] == 200
    assert entries[0]["chars"] > 200


# --------------------------------------------------------------------------- #
# Per-scope degradation (the over-ceiling-scope-render ADR, phase 2c): the degrade
# set is decided once from maximum-form sizes; a member renders as a stamped
# oneline index under a self-declaring header, every other file byte-identical.
# --------------------------------------------------------------------------- #

import json
import re
import shlex
import inspect

from mitos.display import truncate_words

_HUGE = 10_000_000
_HEAVY = "A deliberately heavy rationale sentence for the degrade fixture. "


def _modes(store) -> dict:
    return {s: f["mode"] for s, f in assemble_render(store)["scopes"].items()}


def _full_forms(store, monkeypatch) -> dict:
    """Every scope's maximum-form record, read at a ceiling nothing crosses."""
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", _HUGE)
    return assemble_render(store)["scopes"]


def _cli_recipe(content: str) -> str:
    return re.search(r"`(mitos list [^`]*)`", content).group(1)


def _mcp_recipe(content: str) -> str:
    return re.search(r"`(list_decisions\(.*?\))`", content).group(1)


def test_over_ceiling_scope_degrades_to_index(temp_workspace, monkeypatch) -> None:
    """S1: an over-ceiling scope is one stamped row per tagged decision, no bodies,
    no pointer section, accounting at index-row weight."""
    store, _ = temp_workspace
    for i in range(3):
        _commit(store, f"big-{i}", ["big"], axiom=f"{_HEAVY * 8}Variant {i}.")
    _commit(store, "big-guest", ["home", "big"])
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", 300)

    record = assemble_render(store)["scopes"]["big"]
    content = record["content"]
    assert record["mode"] == "index"
    assert content.startswith("# Active Axioms for Scope: big — Index\n")
    assert "Rejected" not in content and "## big-" not in content
    assert R.POINTER_SECTION_HEADING not in content and "→ full entry" not in content
    tagged = ["big-0", "big-1", "big-2", "big-guest"]
    for slug in tagged:
        assert content.count(f"- **{slug}** — ") == 1
    assert [slug for slug, _ in record["decisions"]] == tagged
    assert all(size < 200 for _, size in record["decisions"])


def test_under_ceiling_scope_stays_full_and_byte_identical(temp_workspace, monkeypatch) -> None:
    """S2: a sibling under the ceiling is the pre-degrade file, byte for byte."""
    store, _ = temp_workspace
    for i in range(3):
        _commit(store, f"big-{i}", ["big"], axiom=f"{_HEAVY * 8}Variant {i}.")
    _commit(store, "small-one", ["small"])
    _commit(store, "shared-one", ["big", "small"])

    shared_axiom = "Axiom for shared-one with enough words to truncate cleanly at a boundary."
    expected_small = (
        "# Active Axioms for Scope: small\n"
        "*Generated automatically by Mitos. Derived statelessly from primary sources (M8).*\n\n"
        "## small-one\n"
        "- **Decided:** Axiom for small-one with enough words to truncate cleanly at a boundary.\n"
        "- **Scope:** small\n"
        "- **Rejected:**\n  Rejected for small-one.\n"
        "\n## Also scoped here (full entries elsewhere)\n"
        f"- **shared-one** — {truncate_words(shared_axiom, 70)} → full entry: big.md\n"
    )
    full = _full_forms(store, monkeypatch)
    assert full["small"]["content"] == expected_small
    ceiling = len(expected_small)
    assert len(full["big"]["content"]) > ceiling
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", ceiling)

    scopes = assemble_render(store)["scopes"]
    assert scopes["big"]["mode"] == "index"
    assert scopes["small"]["mode"] == "full"
    assert scopes["small"]["content"] == expected_small
    assert scopes["small"]["decisions"] == full["small"]["decisions"]


def test_scope_crossing_only_through_its_pointer_section_degrades(
    temp_workspace, monkeypatch
) -> None:
    """S3 (the `substrate` shape): one primary, many secondary rows — the pointer
    section alone carries the file over, and that is enough to degrade."""
    store, _ = temp_workspace
    _commit(store, "sub-own", ["sub"])
    for i in range(20):
        _commit(store, f"visitor-{i}", [f"home-{i}", "sub"])
    full = _full_forms(store, monkeypatch)["sub"]["content"]
    bodies_only = len(full[:full.index(R.POINTER_SECTION_HEADING)])
    ceiling = (bodies_only + len(full)) // 2
    assert bodies_only < ceiling < len(full)
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", ceiling)

    modes = _modes(store)
    assert modes["sub"] == "index"
    assert all(modes[f"home-{i}"] == "full" for i in range(20))


def test_scope_reverts_to_full_when_a_supersede_shrinks_it(temp_workspace, monkeypatch) -> None:
    """S4: the same ceiling across two renders; retiring the heavy decision brings
    the file back under, and it renders as a fresh full-form file."""
    store, _ = temp_workspace
    _commit(store, "heavy-call", ["rev"], axiom=f"{_HEAVY * 12}Heavy.")
    _commit(store, "light-call", ["rev"])
    ceiling = len(_full_forms(store, monkeypatch)["rev"]["content"]) - 1
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", ceiling)
    assert assemble_render(store)["scopes"]["rev"]["mode"] == "index"

    successor = ParsedEntry("decision", "heavy-successor", 1, 5)
    successor.axiom = "A short successor axiom."
    successor.rejected_paths = "The heavy call."
    successor.scope = ["rev"]
    successor.supersedes = ["heavy-call"]
    store.commit_parsed_entry(successor)

    after = assemble_render(store)["scopes"]["rev"]
    fresh = _full_forms(store, monkeypatch)["rev"]
    assert len(fresh["content"]) <= ceiling
    assert after["mode"] == "full"
    assert after["content"] == fresh["content"]


def _degrade_corpus(store) -> None:
    for i in range(3):
        _commit(store, f"same-{i}", ["same", "peer"], axiom=f"{_HEAVY * 6}Variant {i}.")
    _commit(store, "peer-own", ["peer"])


def test_degraded_render_is_stateless_and_path_free(monkeypatch) -> None:
    """S6: the same active set renders the same bytes twice and in two workspaces."""
    import shutil
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", 400)
    renders, paths = [], []
    for _ in range(2):
        workspace = tempfile.mkdtemp()
        paths.append(workspace)
        try:
            store = GraphStore(os.path.join(workspace, ".mitos", "graph.sqlite"))
            _degrade_corpus(store)
            first = assemble_render(store)
            assert assemble_render(store) == first
            renders.append(first["scopes"])
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    assert {s: f["mode"] for s, f in renders[0].items()} == {"same": "index", "peer": "index"}
    assert renders[0] == renders[1]
    for scope_record in renders[0].values():
        for path in paths:
            assert path not in scope_record["content"]
            assert os.path.realpath(path) not in scope_record["content"]


def test_degraded_header_both_registers(temp_workspace, monkeypatch) -> None:
    """S7: under the ceiling the header names the file, stamps, the corpus and both
    tool forms with no size; over it, it adds the rows' size and the applied ceiling."""
    store, _ = temp_workspace
    for i in range(3):
        _commit(store, f"reg-{i}", ["reg"], axiom=f"{_HEAVY * 8}Variant {i}.")
    full_len = len(_full_forms(store, monkeypatch)["reg"]["content"])

    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", full_len - 1)
    under = assemble_render(store)["scopes"]["reg"]
    under_len = len(under["content"])
    assert under["mode"] == "index" and under_len < full_len - 1
    head = under["content"].split("\n- **reg-0**", 1)[0]
    for needle in ("is an index of the 3 active decisions tagged `reg`", "modifier stamps",
                   "full bodies are not in this file", "`decisions.md`",
                   "`decisions/archive/`", "`grep`",
                   "`mitos list --scope=reg --oneline -p .`",
                   "from the workspace root, or `-p <that absolute path>` from anywhere",
                   'list_decisions(scope="reg", oneline=True, project='):
        assert needle in head
    assert "chars" not in head

    # At exactly the ceiling the short register stands; one below, the long one.
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", under_len)
    assert assemble_render(store)["scopes"]["reg"]["content"] == under["content"]
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", under_len - 1)
    over = assemble_render(store)["scopes"]["reg"]
    rows_chars = sum(size for _, size in over["decisions"])
    assert rows_chars < under_len - 1
    assert over["mode"] == "index"
    assert f"({R.SCOPE_OVERFLOW_WARN_CHARS:,} chars)" in over["content"]
    assert f"rows alone come to {rows_chars:,} chars" in over["content"]
    assert over["content"].replace(
        over["content"].split("\n")[4] + "\n", "", 1) == under["content"]


def test_degraded_header_carries_no_machine_identity(temp_workspace, monkeypatch) -> None:
    """S8: the header states the addressing form, never a path, name or config key,
    and names no semantic verb, no row tier and no later format."""
    store, workspace = temp_workspace
    for i in range(3):
        _commit(store, f"plain-{i}", ["plain"], axiom=f"{_HEAVY * 6}Variant {i}.")
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", 300)
    content = assemble_render(store)["scopes"]["plain"]["content"]
    assert "-p ." in content
    assert ("project=<absolute path of the workspace directory this file's .mitos/ "
            "sits in>)") in content
    assert workspace not in content and os.path.realpath(workspace) not in content
    assert os.getcwd() not in content
    assert re.search(r"surface|query|show", content) is None
    assert "madr" not in content.lower()
    assert "render_scope_overflow_warn_chars" not in content
    assert f"({R.SCOPE_OVERFLOW_WARN_CHARS:,} chars)" in content


@pytest.mark.parametrize("tag", ["plain", "foo bar", "-x", "--oneline", "ž tag"])
def test_degraded_header_recipes_parse(temp_workspace, monkeypatch, tag) -> None:
    """S9: the CLI recipe parses to this scope's oneline listing with `-p .`, and the
    MCP form's keywords are real `list_decisions` parameters, for awkward tags too."""
    from mitos import cli, mcp_server
    store, _ = temp_workspace
    for i in range(2):
        _commit(store, f"tagged-{i}", [tag], axiom=f"{_HEAVY * 6}Variant {i}.")
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", 300)
    record = assemble_render(store)["scopes"][tag]
    assert record["mode"] == "index"

    args = cli._build_parser().parse_args(shlex.split(_cli_recipe(record["content"]))[1:])
    assert args.command == "list"
    assert args.scope == tag
    assert args.oneline is True
    assert args.project_post == "."

    mcp = _mcp_recipe(record["content"])
    params = inspect.signature(mcp_server.list_decisions).parameters
    keywords = re.findall(r"(?:\(|, )(\w+)=", mcp)
    assert keywords == ["scope", "oneline", "project"]
    assert set(keywords) <= set(params)
    literal = re.match(r'list_decisions\(scope=(".*?"), oneline=', mcp).group(1)
    assert json.loads(literal) == tag


def test_render_all_writes_the_degraded_file(temp_workspace, monkeypatch) -> None:
    """S10: disk equals the assembled index, every scope path is returned, and the
    overflow record holds a degraded file iff the index itself is over."""
    store, workspace = temp_workspace
    for i in range(3):
        _commit(store, f"fits-{i}", ["fits"], axiom=f"{_HEAVY * 10}Variant {i}.")
    for i in range(40):
        _commit(store, f"many-{i}", ["many"], axiom=f"Many rows decision number {i} here.")
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", 1500)

    assembled = assemble_render(store)
    fits, many = assembled["scopes"]["fits"], assembled["scopes"]["many"]
    assert fits["mode"] == "index" and len(fits["content"]) <= 1500
    assert many["mode"] == "index" and len(many["content"]) > 1500

    renderer = MitosRenderer(workspace)
    written = renderer.render_all(store)
    axioms = os.path.join(workspace, ".mitos", "axioms")
    for name, record in (("fits", fits), ("many", many)):
        path = os.path.join(axioms, f"{name}.md")
        assert path in written
        with open(path, encoding="utf-8") as f:
            assert f.read() == record["content"]
    names = [o["name"] for o in renderer.overflows]
    assert "many.md" in names and "fits.md" not in names


def test_degrade_set_is_one_pass_and_order_free(temp_workspace, monkeypatch) -> None:
    """S11: two scopes sized by each other's secondary rows; at every threshold the
    degrade set is the pass-one set, and reversing the store's order leaves it."""
    store, _ = temp_workspace
    for i in range(4):
        _commit(store, f"xy-{i}", ["x", "y"], axiom=f"{_HEAVY * (i + 2)}Variant {i}.")
        _commit(store, f"yx-{i}", ["y", "x"], axiom=f"{_HEAVY * (5 - i)}Other {i}.")
    full = _full_forms(store, monkeypatch)
    lengths = sorted(len(f["content"]) for f in full.values())
    original = store.get_active_decisions

    for ceiling in sorted({n + d for n in lengths for d in (-1, 0)}):
        expected = {s: ("index" if len(f["content"]) > ceiling else "full")
                    for s, f in full.items()}
        monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", ceiling)
        monkeypatch.setattr(store, "get_active_decisions", original)
        assert _modes(store) == expected
        monkeypatch.setattr(store, "get_active_decisions", lambda: list(reversed(original())))
        assert _modes(store) == expected
