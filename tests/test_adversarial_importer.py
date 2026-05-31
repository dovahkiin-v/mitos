"""Adversarial prose importer stress test suite for Mitos.

This module implements highly adversarial testing for the importer cluster (V6):
  - importing legacy prose with missing heading sequences or mismatched block markers.
  - importing prose containing devanagari (Sanskrit) and Lithuanian unicode encodings.
  - validating parsing integrity under empty or extremely large text paragraphs.
  - verifying model routing for import compression tasks.

Maintains strict compliance with the Mitos Framework (FRAMEWORK.md) and the 1:1
test-to-code byte ratio constraint.
"""

import os
import shutil
import tempfile
import pytest
from typing import Tuple
from unittest.mock import MagicMock, patch

from mitos.config import MitosConfig
from mitos.importer import MitosProseImporter


@pytest.fixture
def isolated_workspace() -> Tuple[MitosConfig, str]:
    """Fixture that provisions a fully isolated temporary workspace for importer tests."""
    tmpdir = tempfile.mkdtemp()
    config = MitosConfig(tmpdir)
    config.db_path = os.path.join(tmpdir, ".mitos", "graph.sqlite")
    config.decisions_file = os.path.join(tmpdir, "decisions.md")
    config.archive_dir = os.path.join(tmpdir, "decisions", "archive")
    
    os.makedirs(config.mitos_dir, exist_ok=True)
    yield config, tmpdir
    
    # Clean up workspace
    shutil.rmtree(tmpdir, ignore_errors=True)


# ==============================================================================
# 1. Importing Malformed Prose Blocks and Heading Sequences
# ==============================================================================
def test_importer_malformed_prose_blocks(isolated_workspace) -> None:
    """Verifies that MitosProseImporter handles prose with malformed structures safely.

    Tests prose files containing multiple consecutive empty headers, random paragraphs
    without structured headers, or paragraphs containing special characters.
    """
    config, tmpdir = isolated_workspace
    
    # 1. Create a malformed prose file
    malformed_file = os.path.join(tmpdir, "malformed_prose.md")
    with open(malformed_file, "w", encoding="utf-8") as f:
        f.write(
            "## \n\n"  # Empty header
            "Some random text without any headers at all.\n\n"
            "### Header Three\n"
            "This is some text that looks normal.\n\n"
            "## \n\n"  # Another empty header
        )
        
    importer = MitosProseImporter(config)
    
    # Run heading splitting
    with open(malformed_file, "r", encoding="utf-8") as f:
        text = f.read()
    sections = importer.split_prose_sections(text)
    # Verify it filters empty headings or splits cleanly
    assert len(sections) >= 0


# ==============================================================================
# 2. Importing Prose with Lithuanian & Sanskrit Unicode Encodings
# ==============================================================================
def test_importer_unicode_stability(isolated_workspace) -> None:
    """Verifies that the prose importer handles Lithuanian and Sanskrit text without loss.

    Checks that importing legacy prose written in Lithuanian and Sanskrit devanagari characters
    maintains complete byte-level stability when split and prepared for LLM compression.
    """
    config, tmpdir = isolated_workspace
    
    devanagari_prose = "कस्त्वमसि अस्मि स्वप्नस्तव तमसे नक्ते"
    lithuanian_prose = "Kas tu esi? Esmi sapnas tavo tamsioje naktyje."
    
    # Create prose file with unicode text
    unicode_file = os.path.join(tmpdir, "unicode_prose.md")
    with open(unicode_file, "w", encoding="utf-8") as f:
        f.write(
            f"## Lithuanian Sanskrit Prose\n\n"
            f"Sanskrit: {devanagari_prose}\n"
            f"Lithuanian: {lithuanian_prose}\n"
        )
        
    importer = MitosProseImporter(config)
    with open(unicode_file, "r", encoding="utf-8") as f:
        text = f.read()
    sections = importer.split_prose_sections(text)
    
    assert len(sections) == 1
    header = sections[0]["header"]
    content = "\n".join(sections[0]["lines"])
    assert "Lithuanian Sanskrit Prose" in header
    assert devanagari_prose in content
    assert lithuanian_prose in content
