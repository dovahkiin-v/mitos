"""Adversarial test suite for the Mitos configuration loader.

Tests defaults, manual attributes settings, file path resolutions,
and TOML-style config loading logic.
"""

import os
import tempfile
import pytest
from mitos.config import MitosConfig

def test_config_defaults() -> None:
    """Verifies that MitosConfig initializes with standard default values."""
    config = MitosConfig()
    assert config.rotation_mode == "archive"
    assert config.pending_threshold == 30
    assert "graph.sqlite" in config.db_path
    assert config.qdrant_collection == "mitos"
    assert "decisions.md" in config.decisions_file


def test_config_set_attributes() -> None:
    """Verifies manual setting of configuration attributes and bounds checking."""
    config = MitosConfig()
    
    # Valid setting overrides
    config.set_attribute("rotation_mode", "prune")
    assert config.rotation_mode == "prune"
    
    config.set_attribute("pending_threshold", "50")
    assert config.pending_threshold == 50
    
    # Invalid setting overrides (ignored)
    config.set_attribute("rotation_mode", "invalid_mode")
    assert config.rotation_mode == "prune"


def test_config_file_loading() -> None:
    """Tests loading config overrides from an on-disk config.toml file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MitosConfig(tmpdir)
        
        # Write config.toml with manual overrides
        os.makedirs(os.path.join(tmpdir, ".mitos"), exist_ok=True)
        config_path = os.path.join(tmpdir, ".mitos", "config.toml")
        
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(
                'rotation_mode = "mark"\n'
                'pending_threshold = 42\n'
                'qdrant_collection = "custom_collection"\n'
            )
            
        # Re-initialize config in same workspace
        loaded_config = MitosConfig(tmpdir)
        
        assert loaded_config.rotation_mode == "mark"
        assert loaded_config.pending_threshold == 42
        assert loaded_config.qdrant_collection == "custom_collection"
