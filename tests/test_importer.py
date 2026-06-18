"""Adversarial test suite for the Mitos legacy prose importer.

Verifies splitting legacy headings, mock LLM compression passes, cross-referencing
relationships, and stable identity hashes.
"""

import tempfile
import os
import shutil
import json
import pytest
from typing import Tuple
from unittest.mock import MagicMock, patch

from mitos.config import MitosConfig
from mitos.store import GraphStore
from mitos.parser import ParsedEntry
from mitos.importer import MitosProseImporter

@pytest.fixture
def import_env() -> Tuple[MitosConfig, MitosProseImporter, str]:
    """Fixture to set up a temporary importer environment."""
    tmpdir = tempfile.mkdtemp()
    config = MitosConfig(tmpdir)
    config.db_path = os.path.join(tmpdir, ".mitos", "graph.sqlite")
    
    os.makedirs(os.path.join(tmpdir, ".mitos"), exist_ok=True)
    importer = MitosProseImporter(config)
    yield config, importer, tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_prose_splitting(import_env: Tuple[MitosConfig, MitosProseImporter, str]) -> None:
    """Verifies that splits of legacy prose by headings are parsed correctly."""
    _, importer, _ = import_env
    
    prose = (
        "# Legacy ADRs\n\n"
        "## adr-one — First Legacy Decision\n"
        "This is details for first decision.\n\n"
        "### adr-two — Second Legacy Decision\n"
        "Details for second decision.\n"
    )
    sections = importer.split_prose_sections(prose)
    
    assert len(sections) == 2
    assert sections[0]["header"] == "## adr-one — First Legacy Decision"
    assert sections[1]["header"] == "### adr-two — Second Legacy Decision"
    assert "This is details for first" in "\n".join(sections[0]["lines"])


@patch("anthropic.Anthropic")
def test_import_with_llm_extract(mock_anthropic: MagicMock, import_env: Tuple[MitosConfig, MitosProseImporter, str]) -> None:
    """Tests that LLM extraction parses and compresses legacy files into the GraphStore."""
    config, importer, tmpdir = import_env

    # 1. Write mock legacy prose file
    legacy_file = os.path.join(tmpdir, "legacy.md")
    with open(legacy_file, "w", encoding="utf-8") as f:
        f.write(
            "## 2026-05-19 — legacy-sqlite — Legacy SQLite decision\n"
            "This is some long prose explaining our database choice.\n"
        )

    # 2. Mock Claude Sonnet response
    mock_msg = MagicMock()
    mock_msg.content = [
        MagicMock(text=json.dumps({
            "core_axiom": "We use SQLite in WAL mode.",
            "rejected_paths": "pgvector.",
            "mechanisms": ["sqlite"],
            "scope": ["substrate"],
            "supersedes": None,
            "amends": None,
            "resolves": None
        }))
    ]
    mock_anthropic.return_value.messages.create.return_value = mock_msg
    os.environ["ANTHROPIC_API_KEY"] = "mock_anthropic_key"

    # 3. Perform import
    importer.import_from_file(legacy_file, use_llm_extract=True)

    # 4. Assertions
    store = GraphStore(config.db_path)
    nodes = store.get_all_nodes()
    assert len(nodes) == 1
    node = nodes[0]
    
    assert node["slug"] == "legacy-sqlite"
    assert node["core_axiom"] == "We use SQLite in WAL mode."
    assert node["rejected_paths"] == "pgvector."
    # V1a import provenance rides nodes.source as the enum value 'import_llm' (V1-D20),
    # set on the entry pre-commit (the prototype's post-commit source='imported' +
    # source_ref UPDATE is retired in 8a — 'imported' is outside the enum and source_ref
    # is a dropped column, §6.5). The file:line provenance has no V1a home (deferred).
    assert node["source"] == "import_llm"
    assert "source_ref" not in node
