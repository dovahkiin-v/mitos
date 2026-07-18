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
