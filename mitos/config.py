"""Configuration management for Mitos.

This module handles loading and validating Mitos configuration from `.mitos/config.toml`
and defines system-wide defaults.
"""

import os
from typing import Dict, Any

class MitosConfig:
    """Represents the configuration state for the active Mitos workspace."""

    def __init__(self, workspace_dir: str = ".") -> None:
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.mitos_dir = os.path.join(self.workspace_dir, ".mitos")
        
        # Default configuration values
        self.db_path = os.path.join(self.mitos_dir, "graph.sqlite")
        self.qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        self.qdrant_collection = "mitos"
        self.rotation_mode = "archive"  # "archive" | "mark" | "prune"
        self.pending_threshold = 30
        self.decisions_file = os.path.join(self.workspace_dir, "decisions.md")
        self.archive_dir = os.path.join(self.workspace_dir, "decisions", "archive")

        self._load_config_file()

    def _load_config_file(self) -> None:
        """Loads configuration overrides from .mitos/config.toml if present."""
        config_path = os.path.join(self.mitos_dir, "config.toml")
        if not os.path.exists(config_path):
            return

        try:
            # Since tomlkit or similar might not be installed or standard,
            # we do a simple manual key-value parse to prevent dependencies,
            # adhering to P19 (Dependency Skepticism).
            with open(config_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        
                        if val.isdigit():
                            parsed_val: Any = int(val)
                        elif val.lower() in ("true", "yes", "1"):
                            parsed_val = True
                        elif val.lower() in ("false", "no", "0"):
                            parsed_val = False
                        else:
                            parsed_val = val
                            
                        self.set_attribute(key, parsed_val)
        except Exception:
            # Fail silently and fallback to defaults
            pass

    def set_attribute(self, key: str, val: Any) -> None:
        """Sets a configuration attribute if it matches a known setting."""
        if key == "db_path":
            self.db_path = os.path.abspath(val) if os.path.isabs(val) else os.path.join(self.mitos_dir, val)
        elif key == "qdrant_url":
            self.qdrant_url = val
        elif key == "qdrant_collection":
            self.qdrant_collection = val
        elif key == "rotation_mode":
            if val in ("archive", "mark", "prune"):
                self.rotation_mode = val
        elif key == "pending_threshold":
            self.pending_threshold = int(val)
        elif key == "decisions_file":
            self.decisions_file = os.path.abspath(val) if os.path.isabs(val) else os.path.join(self.workspace_dir, val)
        elif key == "archive_dir":
            self.archive_dir = os.path.abspath(val) if os.path.isabs(val) else os.path.join(self.workspace_dir, val)

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to dictionary form.

        Returns:
            A dictionary containing configuration fields.
        """
        return {
            "workspace_dir": self.workspace_dir,
            "mitos_dir": self.mitos_dir,
            "db_path": self.db_path,
            "qdrant_url": self.qdrant_url,
            "qdrant_collection": self.qdrant_collection,
            "rotation_mode": self.rotation_mode,
            "pending_threshold": self.pending_threshold,
            "decisions_file": self.decisions_file,
            "archive_dir": self.archive_dir,
        }
