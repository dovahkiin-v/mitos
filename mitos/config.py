"""Configuration management for Mitos.

This module handles loading and validating Mitos configuration from `.mitos/config.toml`
and defines system-wide defaults.
"""

import os
import re
from typing import Dict, Any


def _hint_cache_path(cache_name: str) -> str:
    """Returns the path to a debounce cache file under the user cache dir.

    Honors ``XDG_CACHE_HOME`` (so tests redirect it into a tmp dir) and falls back
    to ``~/.cache``. The file need not exist.

    Args:
        cache_name: The cache file's basename (e.g. ``"mcp_hint.json"``).

    Returns:
        Absolute path to ``<cache>/mitos/<cache_name>``.
    """
    cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(cache_home, "mitos", cache_name)


def hint_due(cache_name: str, key: str, window_seconds: float) -> bool:
    """Fail-silent once-per-window gate for a debounced nudge.

    Backs the recurring-nudge surfaces (the MCP-server hint, the render-overflow
    summary) so they fire at most once per ``window_seconds`` per ``key`` instead of
    on every call. Reads a small JSON cache keyed by ``key``; if that key has not
    fired within the window it stamps the current time and returns True, otherwise
    returns False. Never raises — a missing/corrupt cache or an unwritable cache dir
    degrades to "due" (the nudge shows) rather than crashing the caller.

    Args:
        cache_name: The cache file's basename, namespacing one nudge from another.
        key: The per-subject key to debounce on (typically a workspace path).
        window_seconds: Minimum seconds between two firings for the same key.

    Returns:
        True if the nudge is due now (and the firing was just stamped), else False.
    """
    import json
    import time

    now = time.time()
    path = _hint_cache_path(cache_name)
    shown: Dict[str, Any] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            shown = json.load(f)
    except (OSError, ValueError):
        shown = {}
    if not isinstance(shown, dict):
        shown = {}
    if now - shown.get(key, 0) < window_seconds:
        return False
    shown[key] = now
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(shown, f)
    except OSError:
        pass
    return True


def global_env_path() -> str:
    """Returns the path to Mitos's global ``.env`` (shared across all projects).

    A single-user machine usually wants one set of API keys for every project,
    not a key re-entered per workspace. Mitos reads this global ``.env`` as a
    fallback BELOW any project ``.env`` (and below an explicit environment
    variable), so a key set here once serves every project; a project ``.env``
    still overrides it locally. Honors ``XDG_CONFIG_HOME``.

    Returns:
        Absolute path to ``<config>/mitos/.env`` (``~/.config/mitos/.env`` by
        default). The file need not exist.
    """
    config_home = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config"
    )
    return os.path.join(config_home, "mitos", ".env")


def default_collection_name(workspace_dir: str) -> str:
    """Derives a per-project Qdrant collection name from the workspace path.

    Each Mitos workspace gets its OWN collection so a single shared Qdrant
    instance never mixes decisions across projects. Without this, every project
    would default to the same ``"mitos"`` collection and cross-contaminate
    semantic queries — and, because a point's id is ``hash_to_uuid`` of the
    content hash (M2), two projects recording the same axiom would collide on
    one Qdrant point. The name is ``mitos-<sanitized-basename>`` of the
    workspace dir; set ``qdrant_collection`` in ``.mitos/config.toml`` to
    override explicitly.

    Args:
        workspace_dir: The workspace directory (the project root holding
            ``.mitos/``).

    Returns:
        A Qdrant-safe, project-unique collection name.
    """
    base = os.path.basename(os.path.normpath(workspace_dir)).lower()
    safe = re.sub(r"[^a-z0-9_-]+", "-", base).strip("-")
    return f"mitos-{safe}" if safe else "mitos"


class MitosConfig:
    """Represents the configuration state for the active Mitos workspace."""

    def __init__(self, workspace_dir: str = ".") -> None:
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.mitos_dir = os.path.join(self.workspace_dir, ".mitos")
        
        # Default configuration values
        self.db_path = os.path.join(self.mitos_dir, "graph.sqlite")
        # Mitos defaults to its OWN dedicated port (:7333), NOT the standard
        # Qdrant :6333 — a user's :6333 is usually running for something else, so
        # defaulting there would co-locate Mitos's collections in their instance
        # and share its wipe/contamination risk. :7333 fails safe (semantic just
        # degrades if Mitos's Qdrant isn't up). `docker compose up` starts it.
        # QDRANT_URL overrides for anyone pointing at a different instance.
        self.qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:7333")
        # Per-project by default so a shared Qdrant never mixes projects' decisions.
        # An explicit qdrant_collection in .mitos/config.toml overrides this.
        self.qdrant_collection = default_collection_name(self.workspace_dir)
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
