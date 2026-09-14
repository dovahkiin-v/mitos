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
    # The largest entry is ranked first, so the report says what makes the file long.
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
    # Banner states plainly what happened, and calls no file canonical (2d: a named
    # file may be an index, so the banner routes instead of vouching).
    assert content.startswith("# Live Axioms — Index")
    assert "exceeds the global size ceiling" in content
    assert "one-line index of every active decision, with modifier stamps" in content
    assert "canonical" not in content
    # Grouped by PRIMARY scope tag; both scope files are full at the default scope
    # ceiling, so each heading still names its file.
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
    but an index that itself breaches the ceiling is still reported honestly.

    The ceiling is derived from the full render's length (2d: the banner grew, so a
    hand-picked number no longer sits between the index and the full render)."""
    monkeypatch.setattr(R, "GLOBAL_OVERFLOW_WARN_CHARS", 10_000_000)
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", 10_000_000)
    store, workspace = temp_workspace
    for i in range(6):
        # Distinct axioms — identical content hashes to the same node id.
        _commit(store, f"bulk-{i}", ["alpha"],
                axiom=f"A comfortably verbose axiom sentence for overflow test {i}. " * 8)
    ceiling = len(assemble_render(store)["global"]["content"]) - 1
    monkeypatch.setattr(R, "GLOBAL_OVERFLOW_WARN_CHARS", ceiling)

    # Full render > ceiling → index mode; the index fits → no global overflow entry.
    assembled = assemble_render(store)
    assert assembled["global"]["mode"] == "index"
    assert len(assembled["global"]["content"]) <= ceiling
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
    """S2: a sibling under the ceiling stays full; its bytes are the pass-one measure
    except that its row into the degraded `big` now ends in the marker (2d), which
    never makes the file longer than the measure that kept it full."""
    store, _ = temp_workspace
    for i in range(3):
        _commit(store, f"big-{i}", ["big"], axiom=f"{_HEAVY * 8}Variant {i}.")
    _commit(store, "small-one", ["small"])
    _commit(store, "shared-one", ["big", "small"])

    shared_axiom = "Axiom for shared-one with enough words to truncate cleanly at a boundary."
    row_lead = f"- **shared-one** — {truncate_words(shared_axiom, 70)}"
    measured_small = (
        "# Active Axioms for Scope: small\n"
        "*Generated automatically by Mitos. Derived statelessly from primary sources (M8).*\n\n"
        "## small-one\n"
        "- **Decided:** Axiom for small-one with enough words to truncate cleanly at a boundary.\n"
        "- **Scope:** small\n"
        "- **Rejected:**\n  Rejected for small-one.\n"
        "\n" + R.POINTER_SECTION_HEADING + "\n"
        f"{row_lead} → full entry: big.md\n"
    )
    full = _full_forms(store, monkeypatch)
    assert full["small"]["content"] == measured_small
    ceiling = len(measured_small)
    assert len(full["big"]["content"]) > ceiling
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", ceiling)

    scopes = assemble_render(store)["scopes"]
    assert scopes["big"]["mode"] == "index"
    assert scopes["small"]["mode"] == "full"
    emitted_small = measured_small.replace(
        f"{row_lead} → full entry: big.md\n", f"{row_lead}{R.POINTER_INDEX_TARGET_MARKER}\n")
    assert emitted_small != measured_small
    assert scopes["small"]["content"] == emitted_small
    assert len(emitted_small) <= len(measured_small)
    assert ([slug for slug, _ in scopes["small"]["decisions"]]
            == [slug for slug, _ in full["small"]["decisions"]])


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


# --------------------------------------------------------------------------- #
# The pointer half of D5 (phase 2d): every surface that names a destination names
# a full file, and anything else routes to the bounded tool tier its surface asks.
# --------------------------------------------------------------------------- #

from render_sweep import sweep_destinations


def _pointer_corpus(store) -> None:
    """`big` degrades; `home` holds rows into `big` and into full `lite` (the mix);
    `degonly` / `fullonly` hold rows into one kind only; one decision is untagged."""
    for i in range(3):
        _commit(store, f"big-{i}", ["big"], axiom=f"{_HEAVY * 8}Variant {i}.")
    for i in range(3):
        _commit(store, f"big-guest-{i}", ["big", "home"], axiom=f"{_HEAVY * 3}Guest {i}.")
    _commit(store, "big-visitor", ["big", "degonly"], axiom=f"{_HEAVY * 3}Visitor.")
    _commit(store, "home-own", ["home"])
    _commit(store, "lite-one", ["lite"])
    _commit(store, "lite-guest", ["lite", "home"])
    _commit(store, "lite-visitor", ["lite", "fullonly"])
    untagged = ParsedEntry("decision", "loose-one", 1, 5)
    untagged.axiom = f"{_HEAVY * 4}An untagged decision."
    untagged.rejected_paths = "Rejected for loose-one."
    store.commit_parsed_entry(untagged)


def _pointer_tree(store, monkeypatch):
    """The G1 near-ceiling fixture: the scope ceiling is exactly `home`'s pass-one
    length, and the global file is forced into its index. Returns (measure, tree)."""
    _pointer_corpus(store)
    monkeypatch.setattr(R, "GLOBAL_OVERFLOW_WARN_CHARS", _HUGE)
    full = _full_forms(store, monkeypatch)
    global_full = len(assemble_render(store)["global"]["content"])
    ceiling = len(full["home"]["content"])
    assert len(full["big"]["content"]) > ceiling
    assert all(len(full[s]["content"]) <= ceiling for s in ("lite", "degonly", "fullonly"))
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", ceiling)
    monkeypatch.setattr(R, "GLOBAL_OVERFLOW_WARN_CHARS", global_full - 1)
    return full, assemble_render(store)


def _lines_starting(content: str, lead: str) -> list:
    return [line for line in content.splitlines() if line.startswith(lead)]


def test_index_target_marker_never_longer_than_shortest_file_pointer() -> None:
    """S1: the guard the one-pass exactness rests on, derived from the renderer: a
    marker row for a one-character primary is no longer than its file-form row."""
    node = {"slug": "s", "core_axiom": ""}
    file_row = R.render_pointer_line(node, "x")
    marker_row = R.render_pointer_line(node, "x", True)
    assert file_row != marker_row
    assert marker_row.endswith(R.POINTER_INDEX_TARGET_MARKER + "\n")
    assert len(marker_row) <= len(file_row)


def test_row_form_follows_its_target_mode(temp_workspace, monkeypatch) -> None:
    """S2: in one undegraded file a row into a full primary keeps its file-form bytes
    and a row into a degraded primary ends in the marker; the file never outgrows its
    pass-one measure, and a file with no such row is the pass-one file itself."""
    store, _ = temp_workspace
    full, tree = _pointer_tree(store, monkeypatch)
    home, measured = tree["scopes"]["home"], full["home"]
    assert tree["scopes"]["big"]["mode"] == "index" and home["mode"] == "full"

    lite_row = _lines_starting(measured["content"], "- **lite-guest**")
    assert lite_row and lite_row[0].endswith(" → full entry: lite.md")
    assert _lines_starting(home["content"], "- **lite-guest**") == lite_row
    for i in range(3):
        measured_row = _lines_starting(measured["content"], f"- **big-guest-{i}**")[0]
        assert measured_row.endswith(" → full entry: big.md")
        row = _lines_starting(home["content"], f"- **big-guest-{i}**")
        assert len(row) == 1 and row[0].endswith(R.POINTER_INDEX_TARGET_MARKER)
        assert ".md" not in row[0] and row[0].count(f"big-guest-{i}") == 1
    assert home["content"] != measured["content"]
    assert len(home["content"]) <= len(measured["content"])
    assert tree["scopes"]["lite"]["content"] == full["lite"]["content"]
    assert tree["scopes"]["fullonly"]["content"] == full["fullonly"]["content"]


def test_pointer_section_block_is_constant_and_true_in_every_mix(
        temp_workspace, monkeypatch) -> None:
    """S3: rows all into full files, all into indexes, or mixed — one constant block,
    carrying the decision tier in slot form, and no `(full entries elsewhere)`."""
    store, workspace = temp_workspace
    _, tree = _pointer_tree(store, monkeypatch)
    kinds = {"fullonly": (1, 0), "degonly": (0, 1), "home": (1, 3)}
    for s, (file_rows, marker_rows) in kinds.items():
        content = tree["scopes"][s]["content"]
        assert tree["scopes"][s]["mode"] == "full"
        assert content.count(R.POINTER_SECTION_HEADING) == 1
        assert content.count(" → full entry: ") == file_rows
        assert content.count(R.POINTER_INDEX_TARGET_MARKER + "\n") == marker_rows
        assert "(full entries elsewhere)" not in content
    block = R.POINTER_SECTION_HEADING
    assert "<slug>" in block and "mitos show -p . -- <slug>" in block
    assert ("show_node(ident=\"<slug>\", project=<absolute path of the workspace "
            "directory this file's .mitos/ sits in>)") in block
    assert workspace not in block and ".md" not in block


def _banner(content: str) -> str:
    return content.split("\n## ", 1)[0]


def _banner_corpus(store, kind: str) -> None:
    if kind == "all-full":
        for s in ("a", "b"):
            for i in range(2):
                _commit(store, f"{s}-{i}", [s], axiom=f"{_HEAVY * 6}{s} {i}.")
    elif kind == "all-index":
        for s in ("a", "b"):
            for i in range(2):
                _commit(store, f"{s}-{i}", [s], axiom=f"{_HEAVY * 6}{s} {i}.")
    else:
        _pointer_corpus(store)


@pytest.mark.parametrize("kind", ["all-full", "all-index", "mixed"])
def test_global_banner_routes_honestly_in_both_registers(monkeypatch, kind) -> None:
    """S4: over its ceiling the banner calls nothing canonical, states the full
    render's size and the ceiling, says what each heading kind present means, names
    both tiers in slot form and the corpus; the rows' size appears only when the index
    is itself over, chosen on the whole file."""
    workspace = tempfile.mkdtemp()
    try:
        store = GraphStore(os.path.join(workspace, ".mitos", "graph.sqlite"))
        _banner_corpus(store, kind)
        monkeypatch.setattr(R, "GLOBAL_OVERFLOW_WARN_CHARS", _HUGE)
        full = _full_forms(store, monkeypatch)
        full_len = len(assemble_render(store)["global"]["content"])
        if kind == "all-index":
            scope_ceiling = min(len(f["content"]) for f in full.values()) - 1
        elif kind == "mixed":
            scope_ceiling = len(full["home"]["content"])
        else:
            scope_ceiling = _HUGE
        monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", scope_ceiling)
        modes = _modes(store)

        monkeypatch.setattr(R, "GLOBAL_OVERFLOW_WARN_CHARS", full_len - 1)
        short = assemble_render(store)["global"]
        assert short["mode"] == "index"
        assert "rows alone" not in short["content"]
        assert len(short["content"]) <= full_len - 1
        # The banner states the ceiling, so the bytes move with it; the register does
        # not until the whole file is over.
        monkeypatch.setattr(R, "GLOBAL_OVERFLOW_WARN_CHARS", len(short["content"]))
        content = assemble_render(store)["global"]["content"]
        assert "rows alone" not in content and len(content) <= len(short["content"])
        banner = _banner(content)
        assert "canonical" not in content
        assert f"({full_len:,} chars, ~{full_len // 4:,} tokens)" in banner
        assert f"({R.GLOBAL_OVERFLOW_WARN_CHARS:,} chars)" in banner
        has_file = any(" — full entries: .mitos/axioms/" in l for l in content.splitlines())
        has_index = any(l.endswith(R.INDEX_GROUP_CLAUSE) for l in content.splitlines())
        has_unscoped = "\n## (unscoped)\n" in content
        assert has_file == any(m == "full" for m in modes.values() if m)
        assert has_index == (kind != "all-full")
        assert ("A heading that names a file points at" in banner) == has_file
        assert (f"`{R.INDEX_GROUP_CLAUSE}` names no file" in banner) == has_index
        assert ("`(unscoped)` gathers" in banner) == has_unscoped
        for needle in ("`mitos list --scope=<scope> --oneline -p .`",
                       'list_decisions(scope="<scope>", oneline=True, project=',
                       "`mitos show -p . -- <slug>`",
                       'show_node(ident="<slug>", project=',
                       R._corpus_pointer()):
            assert needle in banner

        squeezed = len(content) - 1
        monkeypatch.setattr(R, "GLOBAL_OVERFLOW_WARN_CHARS", squeezed)
        over = assemble_render(store)["global"]
        rows_chars = sum(size for _, size in over["decisions"])
        assert rows_chars < squeezed
        assert f"its rows alone come to {rows_chars:,} chars" in _banner(over["content"])
        over_line = next(l for l in over["content"].splitlines() if "rows alone" in l)
        at_squeezed_short = content.replace(
            f"({len(content):,} chars)", f"({squeezed:,} chars)", 1)
        assert over["content"].replace(over_line + "\n", "", 1) == at_squeezed_short
    finally:
        import shutil
        shutil.rmtree(workspace, ignore_errors=True)


def test_global_headings_name_a_file_only_while_it_is_full(temp_workspace, monkeypatch) -> None:
    """S5: a full group's heading keeps its file clause byte for byte; an index
    group's names no file; the unscoped heading is exactly `## (unscoped)`."""
    store, _ = temp_workspace
    _, tree = _pointer_tree(store, monkeypatch)
    lines = tree["global"]["content"].splitlines()
    assert tree["global"]["mode"] == "index"
    assert "## home — full entries: .mitos/axioms/home.md" in lines
    assert "## lite — full entries: .mitos/axioms/lite.md" in lines
    big = _lines_starting(tree["global"]["content"], "## big")
    assert big == [f"## big — {R.INDEX_GROUP_CLAUSE}"] and ".md" not in big[0]
    assert _lines_starting(tree["global"]["content"], "## (unscoped)") == ["## (unscoped)"]


def test_no_surface_names_a_non_full_destination(temp_workspace, monkeypatch) -> None:
    """S6 (T5, 2d half): the derived sweep finds no violation over a ceiling-crossing
    tree, parses every destination kind at least once, and flags a planted 2c row."""
    store, _ = temp_workspace
    _, tree = _pointer_tree(store, monkeypatch)
    violations, counts = sweep_destinations(tree)
    assert violations == []
    for kind in ("file_heading", "index_heading", "unscoped_heading", "section_block",
                 "file_row", "marker_row"):
        assert counts[kind] > 0, kind

    import copy
    planted = copy.deepcopy(tree)
    home = planted["scopes"]["home"]
    home["content"] = home["content"].replace(
        R.POINTER_INDEX_TARGET_MARKER + "\n", " → full entry: big.md\n", 1)
    planted_violations, _ = sweep_destinations(planted)
    assert len(planted_violations) == 1 and "big.md" in planted_violations[0]


@pytest.mark.parametrize("token", ["plain", "foo bar", "-x"])
def test_slot_recipes_parse_with_awkward_tokens(temp_workspace, monkeypatch, token) -> None:
    """S7: the banner's and the section block's recipes, with a tag or slug put in
    the slot shell-quoted, parse to the right verb, value and `-p .`; the MCP forms'
    keywords are real tool parameters."""
    from mitos import cli, mcp_server
    store, _ = temp_workspace
    _, tree = _pointer_tree(store, monkeypatch)
    banner = _banner(tree["global"]["content"])
    parser = cli._build_parser()
    quoted = shlex.quote(token)

    assert "--scope=<scope> " in banner
    args = parser.parse_args(
        shlex.split(_cli_recipe(banner).replace("<scope>", quoted))[1:])
    assert (args.command, args.scope, args.oneline, args.project_post) == (
        "list", token, True, ".")
    for surface in (banner, R.POINTER_SECTION_HEADING):
        recipe = re.search(r"`(mitos show [^`]*)`", surface).group(1)
        assert recipe.endswith("-- <slug>")
        args = parser.parse_args(shlex.split(recipe.replace("<slug>", quoted))[1:])
        assert (args.command, args.ident, args.project_post) == ("show", token, ".")
        mcp = re.search(r"`(show_node\(.*?\))`", surface).group(1)
        keywords = re.findall(r"(?:\(|, )(\w+)=", mcp)
        assert keywords == ["ident", "project"]
        assert set(keywords) <= set(inspect.signature(mcp_server.show_node).parameters)
        assert json.loads(re.match(r'show_node\(ident=(".*?"), ', mcp).group(1)) == "<slug>"
    mcp_list = _mcp_recipe(banner)
    assert re.findall(r"(?:\(|, )(\w+)=", mcp_list) == ["scope", "oneline", "project"]


def test_pointer_surfaces_carry_no_machine_identity(monkeypatch) -> None:
    """S8: banner and section block state forms, never a path or a semantic verb;
    one corpus in two workspaces renders every file byte for byte."""
    import shutil
    trees, paths = [], []
    for _ in range(2):
        workspace = tempfile.mkdtemp()
        paths.append(workspace)
        try:
            store = GraphStore(os.path.join(workspace, ".mitos", "graph.sqlite"))
            trees.append(_pointer_tree(store, monkeypatch)[1])
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
    assert trees[0] == trees[1]
    banner = _banner(trees[0]["global"]["content"])
    for text in (banner, R.POINTER_SECTION_HEADING):
        assert re.search(r"surface|query", text) is None
        for path in paths:
            assert path not in text and os.path.realpath(path) not in text
    assert "project=<absolute path of the workspace directory" in banner


def test_every_over_ceiling_file_is_an_index(temp_workspace, monkeypatch) -> None:
    """S9: the invariant `mitos status` states — at every boundary ceiling of the
    pointer fixture (G1 included), a file over its ceiling has `mode == "index"`,
    so the overflow report names only index files."""
    store, _ = temp_workspace
    full, _ = _pointer_tree(store, monkeypatch)
    global_ceiling = R.GLOBAL_OVERFLOW_WARN_CHARS
    lengths = sorted(len(f["content"]) for f in full.values())
    for ceiling in sorted({n + d for n in lengths for d in (-1, 0)}):
        monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", ceiling)
        for g in (global_ceiling, _HUGE):
            monkeypatch.setattr(R, "GLOBAL_OVERFLOW_WARN_CHARS", g)
            tree = assemble_render(store)
            records = [tree["global"]] + list(tree["scopes"].values())
            for record in records:
                if len(record["content"]) > R._ceiling_for(record):
                    assert record["mode"] == "index", (ceiling, record["name"])
            by_name = {r["name"]: r for r in records}
            assert all(by_name[o["name"]]["mode"] == "index" for o in overflow_report(store))


def test_row_rewrite_keeps_the_degrade_set_order_free(temp_workspace, monkeypatch) -> None:
    """S10: with rows into a degraded primary, reversing the store's order leaves the
    degrade set and every record's mode unchanged."""
    store, _ = temp_workspace
    _, tree = _pointer_tree(store, monkeypatch)
    expected = {s: f["mode"] for s, f in tree["scopes"].items()}
    assert "index" in expected.values() and "full" in expected.values()
    original = store.get_active_decisions
    monkeypatch.setattr(store, "get_active_decisions", lambda: list(reversed(original())))
    reversed_tree = assemble_render(store)
    assert {s: f["mode"] for s, f in reversed_tree["scopes"].items()} == expected
    assert reversed_tree["global"]["mode"] == tree["global"]["mode"]


# --------------------------------------------------------------------------- #
# Phase 2e: the vacated-scope sweep (the directory is a projection of the active
# set) and the write fence (no scope tag renders outside .mitos/axioms/)
# --------------------------------------------------------------------------- #

def _legacy_title(s: str) -> str:
    """The scope title as typed by hand, byte-identical since v0.1 (`a84f0dd`).

    Pinned independently of the renderer's title helper, so the rows below cannot
    pass by deriving their expectation from the code under test.
    """
    return f"# Active Axioms for Scope: {s}"


def _axioms(workspace: str) -> str:
    return os.path.join(workspace, ".mitos", "axioms")


def _plant(directory: str, name: str, first_line, body: bytes = b"Body line.\n") -> str:
    """Writes a file byte-exactly (a CRLF or undecodable plant stays as planted)."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    head = first_line if isinstance(first_line, bytes) else (first_line + "\n").encode("utf-8")
    with open(path, "wb") as f:
        f.write(head + body)
    return path


def _tree(directory: str) -> dict:
    """Every entry under ``directory``: bytes for files, the link text for symlinks."""
    found = {}
    for dirpath, dirnames, filenames in os.walk(directory):
        for name in dirnames + filenames:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, directory)
            if os.path.islink(path):
                found[rel] = ("link", os.readlink(path))
            elif os.path.isdir(path):
                found[rel] = ("dir",)
            else:
                with open(path, "rb") as f:
                    found[rel] = f.read()
    return found


def _warnings(err: str) -> list:
    return [ln for ln in err.splitlines() if ln.startswith("[Warning]")]


def _vacated_tree(store, workspace) -> None:
    """Scopes `a` and `b`, rendered; then `b` vacated by a same-core scope edit."""
    _commit(store, "x", ["a", "b"])
    MitosRenderer(workspace).render_all(store)
    nodes = len(store.get_all_nodes())
    _commit(store, "x", ["a"])
    assert len(store.get_all_nodes()) == nodes, "the edit must re-commit one node, not add one"
    assert os.path.exists(os.path.join(_axioms(workspace), "b.md"))


def test_filtered_render_never_sweeps(temp_workspace, capsys) -> None:
    """E1: a filtered render removes nothing; every other entry stays byte-identical."""
    store, workspace = temp_workspace
    _vacated_tree(store, workspace)
    ax = _axioms(workspace)
    _plant(ax, "notes.md", "# My notes")
    before = _tree(ax)

    renderer = MitosRenderer(workspace)
    renderer.render_all(store, scope="a")

    after = _tree(ax)
    assert {k: v for k, v in after.items() if k != "a.md"} == \
        {k: v for k, v in before.items() if k != "a.md"}
    assert renderer.swept == [] and renderer.sweep_failures == []
    assert _warnings(capsys.readouterr().err) == []

    # Non-vacuity: the same tree under an unfiltered render loses b.md.
    renderer.render_all(store)
    assert renderer.swept == ["b.md"]
    assert not os.path.exists(os.path.join(ax, "b.md"))


def test_empty_scope_string_is_an_unfiltered_render(temp_workspace) -> None:
    """E1b: `scope=""` renders every scope, so it sweeps too (one predicate for both)."""
    store, workspace = temp_workspace
    _vacated_tree(store, workspace)
    renderer = MitosRenderer(workspace)
    renderer.render_all(store, scope="")
    assert renderer.swept == ["b.md"]
    assert not os.path.exists(os.path.join(_axioms(workspace), "b.md"))


def test_scope_vacated_by_an_edit_loses_its_file(temp_workspace, capsys) -> None:
    """E2: the vacated file is removed, never returned, never an overflow; idempotent."""
    store, workspace = temp_workspace
    _vacated_tree(store, workspace)
    ax = _axioms(workspace)
    renderer = MitosRenderer(workspace)

    written = renderer.render_all(store)

    assert not os.path.exists(os.path.join(ax, "b.md"))
    assert os.path.exists(os.path.join(ax, "a.md"))
    assert all(os.path.basename(p) != "b.md" for p in written)
    assert all(o["name"] != "b.md" for o in renderer.overflows)
    assert renderer.swept == ["b.md"] and renderer.sweep_failures == []

    again = renderer.render_all(store)
    assert again == written
    assert renderer.swept == [] and renderer.sweep_failures == []
    assert capsys.readouterr().out == ""


def test_scope_vacated_by_a_supersede_loses_its_file(temp_workspace) -> None:
    """E3: a successor outside the old scope retires its last decision; the file goes."""
    store, workspace = temp_workspace
    _commit(store, "old-call", ["legacy"])
    renderer = MitosRenderer(workspace)
    renderer.render_all(store)
    legacy = os.path.join(_axioms(workspace), "legacy.md")
    assert os.path.exists(legacy)

    successor = ParsedEntry("decision", "new-call", 1, 5)
    successor.axiom = "The modern call replaces the legacy one."
    successor.rejected_paths = "Keeping the legacy call."
    successor.scope = ["modern"]
    successor.supersedes = ["old-call"]
    store.commit_parsed_entry(successor)

    renderer.render_all(store)
    assert not os.path.exists(legacy)
    assert renderer.swept == ["legacy.md"]


def test_foreign_entries_survive_the_sweep_byte_identical(temp_workspace, capsys) -> None:
    """E4: only a file whose name AND first line are the renderer's own for an
    unclaimed scope is removed; everything else survives unreported."""
    store, workspace = temp_workspace
    _commit(store, "ax-call", ["ax"])
    ax = _axioms(workspace)
    _plant(ax, "notes.md", "Arbitrary first line")
    _plant(ax, "README.md", _legacy_title("readme"))
    _plant(ax, "Upper.md", _legacy_title("upper"))
    _plant(ax, "b.md", "# My notes on b")
    _plant(ax, "ax-backup.md", _legacy_title("ax"))
    _plant(ax, ".gone.md.0123456789ab.tmp", _legacy_title("gone"))
    _plant(os.path.join(ax, "old"), "c.md", _legacy_title("c"))
    outside = _plant(workspace, "outside-d.md", _legacy_title("d"))
    os.symlink(outside, os.path.join(ax, "d.md"))
    _plant(ax, "e.md", _legacy_title("e").encode("utf-8") + b"\xff\xfe\n")
    with open(os.path.join(os.fsencode(ax), b"\xff.md"), "wb") as f:  # an undecodable name
        f.write(b"# Active Axioms for Scope: \xff\n")
    # Non-vacuity, in the same tree: two genuine orphans, one checked out with CRLF.
    _plant(ax, "gone.md", _legacy_title("gone"))
    _plant(ax, "crlf.md", (_legacy_title("crlf") + "\r\n").encode("utf-8"), b"Body.\r\n")
    with open(outside, "rb") as f:
        outside_bytes = f.read()
    before = _tree(ax)

    renderer = MitosRenderer(workspace)
    renderer.render_all(store)

    after = _tree(ax)
    kept = {k: v for k, v in before.items() if k not in ("gone.md", "crlf.md", "ax.md")}
    assert {k: v for k, v in after.items() if k != "ax.md"} == kept
    assert os.path.islink(os.path.join(ax, "d.md"))
    with open(outside, "rb") as f:
        assert f.read() == outside_bytes
    assert sorted(renderer.swept) == ["crlf.md", "gone.md"]
    assert renderer.sweep_failures == []
    assert _warnings(capsys.readouterr().err) == []


def test_every_form_the_renderer_writes_is_recognised(temp_workspace, monkeypatch) -> None:
    """E5: full, index and empty-state records all classify as the renderer's own,
    and so does the hand-typed title every shipped version wrote."""
    store, workspace = temp_workspace
    _commit(store, "full-call", ["fullscope"])
    for i in range(3):
        _commit(store, f"heavy-{i}", ["indexscope"], axiom=f"{_HEAVY * 6}Variant {i}.")
    full_len = len(_full_forms(store, monkeypatch)["indexscope"]["content"])
    monkeypatch.setattr(R, "SCOPE_OVERFLOW_WARN_CHARS", full_len - 1)
    scopes = assemble_render(store)["scopes"]
    records = [scopes["fullscope"], scopes["indexscope"], R._empty_scope_file("z")]
    assert [r["mode"] for r in records] == ["full", "index", "full"]

    probe = os.path.join(workspace, "probe")
    os.makedirs(probe)
    for record in records:
        with open(os.path.join(probe, record["name"]), "w", encoding="utf-8") as f:
            f.write(record["content"])
    _plant(probe, "legacy.md", _legacy_title("legacy"))
    with os.scandir(probe) as entries:
        verdicts = {e.name: R._is_vacated_scope_render(e, frozenset()) for e in entries}
    assert verdicts == {"fullscope.md": True, "indexscope.md": True, "z.md": True,
                        "legacy.md": True}


def test_a_removal_failure_is_reported_and_never_raised(temp_workspace, capsys, monkeypatch) -> None:
    """E6: one refused removal leaves the render's result intact and prints one stderr line."""
    store, workspace = temp_workspace
    _commit(store, "ok-call", ["ok"])
    renderer = MitosRenderer(workspace)
    clean = renderer.render_all(store)
    clean_overflows = list(renderer.overflows)
    ax = _axioms(workspace)
    stuck = _plant(ax, "p.md", _legacy_title("p"))
    _plant(ax, "q.md", _legacy_title("q"))
    capsys.readouterr()

    real_remove, fired = os.remove, []

    def refuse_p(path, *args, **kwargs):
        if path == stuck:
            fired.append(path)
            raise PermissionError(13, "Permission denied", path)
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(os, "remove", refuse_p)
    written = renderer.render_all(store)
    monkeypatch.setattr(os, "remove", real_remove)

    assert fired == [stuck]
    assert written == clean and renderer.overflows == clean_overflows
    assert os.path.exists(stuck) and not os.path.exists(os.path.join(ax, "q.md"))
    assert renderer.swept == ["q.md"]
    assert [f["name"] for f in renderer.sweep_failures] == ["p.md"]
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = _warnings(captured.err)
    assert len(lines) == 1 and "p.md" in lines[0] and "mitos " not in lines[0]


def test_a_listing_failure_is_reported_and_the_render_stands(temp_workspace, capsys, monkeypatch) -> None:
    """E8: the directory cannot be listed; one failure, one stderr line, normal return."""
    store, workspace = temp_workspace
    _commit(store, "ok-call", ["ok"])
    renderer = MitosRenderer(workspace)
    real_scandir, fired = os.scandir, []

    def refuse_axioms(path=".", *args, **kwargs):
        if path == renderer.axioms_dir:
            fired.append(path)
            raise PermissionError(13, "Permission denied", path)
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", refuse_axioms)
    written = renderer.render_all(store)
    monkeypatch.setattr(os, "scandir", real_scandir)

    assert fired == [renderer.axioms_dir]
    assert os.path.join(renderer.axioms_dir, "ok.md") in written
    assert len(renderer.sweep_failures) == 1
    captured = capsys.readouterr()
    lines = _warnings(captured.err)
    assert captured.out == "" and len(lines) == 1
    # The line names the directory it could not check, never a "removal" of it.
    assert "Could not check" in lines[0] and "remove" not in lines[0]


def test_a_concurrent_removal_counts_as_success(temp_workspace, capsys, monkeypatch) -> None:
    """E9: another render removed the candidate first; nothing is reported."""
    store, workspace = temp_workspace
    _commit(store, "ok-call", ["ok"])
    gone = _plant(_axioms(workspace), "gone.md", _legacy_title("gone"))
    real_remove, fired = os.remove, []

    def already_gone(path, *args, **kwargs):
        if path == gone:
            fired.append(path)
            real_remove(path)
            raise FileNotFoundError(2, "No such file or directory", path)
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(os, "remove", already_gone)
    renderer = MitosRenderer(workspace)
    renderer.render_all(store)
    monkeypatch.setattr(os, "remove", real_remove)

    assert fired == [gone]
    assert renderer.sweep_failures == []
    assert _warnings(capsys.readouterr().err) == []


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="environmental: permission bits do not bind root, so the "
                           "unreadable-candidate state cannot be built")
def test_an_unreadable_candidate_is_kept_and_reported(temp_workspace, capsys) -> None:
    """E10: ownership cannot be proven without reading the title, so the file stays."""
    store, workspace = temp_workspace
    _commit(store, "ok-call", ["ok"])
    locked = _plant(_axioms(workspace), "locked.md", _legacy_title("locked"))
    os.chmod(locked, 0)
    try:
        renderer = MitosRenderer(workspace)
        renderer.render_all(store)
        assert os.path.exists(locked)
        assert [f["name"] for f in renderer.sweep_failures] == ["locked.md"]
        assert len(_warnings(capsys.readouterr().err)) == 1
    finally:
        os.chmod(locked, 0o644)


def test_the_flat_listing_reaches_nothing_outside_the_tree(temp_workspace) -> None:
    """E11: a titled orphan one directory up and one in a subdirectory both survive."""
    store, workspace = temp_workspace
    _commit(store, "ok-call", ["ok"])
    above = _plant(os.path.join(workspace, ".mitos"), "x.md", _legacy_title("x"))
    nested = _plant(os.path.join(_axioms(workspace), "sub"), "x.md", _legacy_title("x"))
    renderer = MitosRenderer(workspace)
    renderer.render_all(store)
    assert os.path.exists(above) and os.path.exists(nested)
    assert renderer.swept == [] and renderer.sweep_failures == []


def test_no_scope_tag_renders_outside_the_axioms_tree(temp_workspace, capsys) -> None:
    """E12: an escaping tag's file is skipped and reported; every other file renders,
    the gold source is untouched, and a user's symlink is still written through."""
    import shutil
    store, workspace = temp_workspace
    outside_dir = tempfile.mkdtemp()
    try:
        gold = os.path.join(workspace, "decisions.md")
        with open(gold, "wb") as f:
            f.write(b"GOLD SOURCE\n")
        abs_tag = os.path.join(outside_dir, "abs-x")
        escaping = ["../../decisions", abs_tag, "a/../../x"]
        for i, tag in enumerate(escaping):
            _commit(store, f"escape-{i}", [tag])
        _commit(store, "nested-call", ["a/b"])
        _commit(store, "ok-call", ["ok"])
        ax = _axioms(workspace)
        target = _plant(workspace, "ok-target.md", "user content")
        os.makedirs(ax, exist_ok=True)
        os.symlink(target, os.path.join(ax, "ok.md"))

        renderer = MitosRenderer(workspace)
        written = renderer.render_all(store)

        with open(gold, "rb") as f:
            assert f.read() == b"GOLD SOURCE\n"
        assert not os.path.exists(abs_tag + ".md")
        assert not os.path.exists(os.path.join(workspace, ".mitos", "x.md"))
        assert os.path.exists(os.path.join(ax, "a", "b.md"))
        assert os.path.islink(os.path.join(ax, "ok.md"))
        with open(target, encoding="utf-8") as f:
            assert f.read() == assemble_render(store)["scopes"]["ok"]["content"]
        escaped = {os.path.normpath(os.path.join(ax, f"{t}.md")) for t in escaping}
        assert not escaped & {os.path.normpath(p) for p in written}
        assert sorted(r["scope"] for r in renderer.write_refusals) == sorted(escaping)
        captured = capsys.readouterr()
        assert captured.out == ""
        lines = _warnings(captured.err)
        assert len(lines) == 3
        assert all(any(repr(t) in ln for ln in lines) for t in escaping)

        renderer.render_all(store, scope="../../decisions")
        with open(gold, "rb") as f:
            assert f.read() == b"GOLD SOURCE\n"
        assert [r["scope"] for r in renderer.write_refusals] == ["../../decisions"]
    finally:
        shutil.rmtree(outside_dir, ignore_errors=True)


def test_an_unwritable_escaping_tag_no_longer_stalls_the_render(temp_workspace) -> None:
    """E12b (scout W2): before the fence, an absolute tag into an unwritable directory
    raised mid-loop and every later scope went unwritten."""
    store, workspace = temp_workspace
    _commit(store, "unwritable-call", ["/proc/mitos-nope-x"])
    _commit(store, "later-call", ["later"])
    renderer = MitosRenderer(workspace)
    renderer.render_all(store)
    assert os.path.exists(os.path.join(_axioms(workspace), "later.md"))
    assert [r["scope"] for r in renderer.write_refusals] == ["/proc/mitos-nope-x"]


def test_a_refusal_is_still_reported_when_a_later_write_raises(temp_workspace, capsys, monkeypatch) -> None:
    """A refusal recorded before an ordinary write failure is reported, not lost with the raise;
    and the render that raised removes nothing."""
    store, workspace = temp_workspace
    _commit(store, "escape-call", ["../../decisions"])
    _commit(store, "broken-call", ["broken"])
    gone = _plant(_axioms(workspace), "gone.md", _legacy_title("gone"))
    real_write = R.atomic_write

    def refuse_broken(path, content):
        if os.path.basename(path) == "broken.md":
            raise IOError("disk full")
        return real_write(path, content)

    monkeypatch.setattr(R, "atomic_write", refuse_broken)
    with pytest.raises(IOError):
        MitosRenderer(workspace).render_all(store)
    assert any("'../../decisions'" in ln for ln in _warnings(capsys.readouterr().err))
    assert os.path.exists(gone)


def test_a_call_that_raises_early_shows_none_of_the_last_calls_results(temp_workspace, monkeypatch) -> None:
    """The three runtime attributes reset before assembly, so a call that raises there
    never leaves the previous call's refusals and removals readable as its own."""
    store, workspace = temp_workspace
    _commit(store, "escape-call", ["../../decisions"])
    _plant(_axioms(workspace), "gone.md", _legacy_title("gone"))
    renderer = MitosRenderer(workspace)
    renderer.render_all(store)
    assert renderer.write_refusals and renderer.swept == ["gone.md"]

    def broken_assembly(_store):
        raise RuntimeError("graph unreadable")

    monkeypatch.setattr(R, "assemble_render", broken_assembly)
    with pytest.raises(RuntimeError):
        renderer.render_all(store)
    assert renderer.swept == [] and renderer.sweep_failures == [] and renderer.write_refusals == []
