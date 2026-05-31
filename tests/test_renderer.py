"""Adversarial test suite for the Mitos stateless renderer.

Verifies stateless rendering from primary sources (M8), atomic-write tempfile
swapping, and global vs scope-specific tag segregation.
"""

import tempfile
import os
import pytest
from typing import Tuple
from mitos.store import GraphStore
from mitos.parser import ParsedEntry
from mitos.renderer import MitosRenderer, atomic_write

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
    entry1.core_axiom = "We use Python 3.12."
    entry1.rejected_paths = "Older versions."
    entry1.scope = ["backend"]
    store.commit_parsed_entry(entry1)

    # Commit superseded node in scope 'frontend' (should be excluded)
    entry2 = ParsedEntry("decision", "fe-old", 1, 5)
    entry2.core_axiom = "Vanilla JS."
    entry2.rejected_paths = "React."
    entry2.scope = ["frontend"]
    d2 = store.commit_parsed_entry(entry2)

    entry3 = ParsedEntry("decision", "fe-new", 1, 5)
    entry3.core_axiom = "Vite + TS."
    entry3.rejected_paths = "Vanilla JS."
    entry3.supersedes = "fe-old"
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
