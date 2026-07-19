"""Configuration management for Mitos.

This module handles loading and validating Mitos configuration from `.mitos/config.toml`
and defines system-wide defaults.
"""

import os
import re
import sys
import tomllib
from typing import Dict, Any

from mitos.errors import ConfigError

# ---------------------------------------------------------------------------
# v0.1 config schema (§5.2.6) — the SINGLE source of the static defaults.
#
# `CONFIG_DEFAULTS` holds the seven STATIC-default schema keys: `mitos init` (6b)
# seeds `config.toml` from this exact map, and the loader's missing-key fallback
# reads it — so a seeded file and a deleted-key fallback can never diverge (P11).
# The two QDRANT keys are recognized + type-validated (in `CONFIG_SCHEMA`) but NOT
# here: their defaults are DYNAMIC (env- / workspace-derived) and computed in
# `__init__` from their existing single-source helpers, then file-overridable.
# ---------------------------------------------------------------------------
CONFIG_DEFAULTS: Dict[str, Any] = {
    "rotation_mode": "archive",
    "rotation_archive_path_template": "decisions/archive/{year}-Q{quarter}.md",
    "rotation_volume_threshold_entries": 50,
    "stale_entry_window_days": 30,
    "embedding_cache_max_entries": 10_000,
    # Pinned to renderer.py's GLOBAL/SCOPE_OVERFLOW_WARN_CHARS via a cross-check
    # test (config.py is a lower-tier leaf; importing renderer would invert tiers).
    # V4 wires the renderer to read these keys, making config the runtime source.
    "render_global_overflow_warn_chars": 50_000,
    "render_scope_overflow_warn_chars": 20_000,
    # The Conflict sensor's licence toggle (v0.2). Read by the sync hook in Phase
    # 5a; dormant until then. The first bool-typed key across this machinery.
    "conflict_check_on_sync": True,
}

# The recognized file keys → expected (TOML scalar) type, for strict validation.
# The eight static keys above PLUS the two dynamic-default qdrant keys = the §5.2.6
# ten-key schema. A file key NOT in this map is tolerated and skipped — split into
# two buckets by `_load_config_file`: a RECOGNIZED-but-retired key (`RETIRED_CONFIG_KEYS`
# below) is tolerated SILENTLY, while a genuinely unknown key (a typo) earns one
# calm stderr line.
CONFIG_SCHEMA: Dict[str, type] = {
    "rotation_mode": str,
    "rotation_archive_path_template": str,
    "rotation_volume_threshold_entries": int,
    "stale_entry_window_days": int,
    "embedding_cache_max_entries": int,
    "render_global_overflow_warn_chars": int,
    "render_scope_overflow_warn_chars": int,
    "qdrant_url": str,
    "qdrant_collection": str,
    "conflict_check_on_sync": bool,
}

# Keys the code DELIBERATELY dropped from the file schema but still recognizes —
# their ATTRIBUTES survive at a default (R12); only the file-override capability is
# gone. These are NOT typos, so the per-invocation "unrecognized config key" warning
# is a false alarm: the `mitos init`-seeded `pending_threshold` line tripped it on
# every single call. They are tolerated SILENTLY. The warning is reserved for keys
# the code does not know at all — where it is the useful signal that a setting will
# silently not take effect.
RETIRED_CONFIG_KEYS: frozenset = frozenset(
    {"pending_threshold", "db_path", "decisions_file", "archive_dir"}
)

# The `rotation_mode` enum: correct type (str) but a value outside this set is a
# hard ConfigError — a typo'd `rotation_mode` silently defaulting to "archive" and
# then archiving when the author meant "mark" is exactly the silent-coerce OD1
# forbids (a deliberate behavior change from the prototype's silent-ignore).
ROTATION_MODES = frozenset({"archive", "mark", "prune"})


def _value_matches_type(value: Any, expected: type) -> bool:
    """Returns True if a parsed TOML value matches a schema key's expected type.

    Treats ``bool`` as distinct from ``int`` even though ``bool`` subclasses
    ``int``: a TOML ``true`` must NOT satisfy an int-typed key (the silent-coerce
    the strict loader exists to kill). Symmetrically, a bool-typed key (the v0.2
    ``conflict_check_on_sync``) only accepts a native TOML boolean — a ``1`` or a
    quoted ``"true"`` is a loud mismatch, not a coercion.

    Args:
        value: The value ``tomllib`` parsed for the key.
        expected: The type the key's ``CONFIG_SCHEMA`` entry requires.

    Returns:
        True if ``value`` is acceptably typed for ``expected``, else False.
    """
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected)


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

        # Convention-path attributes — derived from the workspace, NOT file-schema
        # keys in v0.1 (a file occurrence is warn-tolerated). Consumers
        # (store/sync/importer/cli/mcp_server) bind these by name (R12), so they
        # stay real instance attributes even though the file can no longer set them.
        self.db_path = os.path.join(self.mitos_dir, "graph.sqlite")
        # The Conflict sensor's non-rebuildable telemetry store (v0.2), a sibling of
        # the graph deliberately fenced OUTSIDE the rebuild/cutover swap set so it
        # survives ``rm graph.sqlite`` / ``mitos rebuild`` (CONF-D8, the T8
        # guarantee). A derived attribute, NOT a user-overridable file-schema key —
        # deriving it here gives the store + the 5b sync surface one canonical path
        # expression sitting next to ``db_path``, instead of reassembling "sibling of
        # the graph" at each call site.
        self.telemetry_path = os.path.join(self.mitos_dir, "telemetry.sqlite")
        self.decisions_file = os.path.join(self.workspace_dir, "decisions.md")
        # The open-question authoring buffer, a fixed v0.1 convention path
        # paralleling ``decisions_file`` (ADR
        # ``open-questions-authored-in-separate-questions-md-file``). ``mitos init``
        # (6b) seeds it; the V3a sync / V6 importer consumers read it later
        # (forward-provided — no in-vision reader yet, like 5d's protocol seams).
        self.questions_file = os.path.join(self.workspace_dir, "questions.md")
        self.archive_dir = os.path.join(self.workspace_dir, "decisions", "archive")

        # `pending_threshold` LEFT the v0.1 file schema (its migration to
        # `rotation_volume_threshold_entries` is V3a's, not V1a's) but stays a
        # default-valued attribute — `sync.py`'s rotation-prompt gate reads it. A
        # `pending_threshold` file key is now silently tolerated (a recognized
        # retired key — see RETIRED_CONFIG_KEYS), not applied.
        self.pending_threshold = 30

        # Dynamic-default schema keys: recognized + type-validated by CONFIG_SCHEMA,
        # file-overridable, but defaulted from their single-source helpers (not from
        # CONFIG_DEFAULTS, which holds only the STATIC defaults).
        #
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

        # Static-default schema keys — seeded from the single CONFIG_DEFAULTS map
        # (P11), the same map `mitos init` (6b) serializes. The keys are exactly the
        # attribute names, so a plain setattr keeps the surface in lockstep.
        for key, default in CONFIG_DEFAULTS.items():
            setattr(self, key, default)

        self._load_config_file()

        # Env wins over the config file for the Qdrant URL — matching the key
        # resolution order (env → project .env → global .env) and the documented
        # contract above ("QDRANT_URL overrides for anyone pointing at a different
        # instance"). Before this re-assert, a toml-pinned qdrant_url silently
        # shadowed the env var (AX 2026-07-18): the caller's override did nothing
        # and nothing said so.
        if os.environ.get("QDRANT_URL"):
            self.qdrant_url = os.environ["QDRANT_URL"]

    def _load_config_file(self) -> None:
        """Overlays `.mitos/config.toml` onto the defaults under the strict policy.

        Replaces the prototype's hand-rolled ``key=val`` parser (which swallowed
        every error back to defaults) with a ``tomllib`` loader enforcing the
        §5.2.6 failure-mode policy, symmetric with OD1: a broken config is loud and
        located, never silently defaulted.

        Policy:
            - Malformed TOML → ``ConfigError`` carrying the path + the decoder's
              line/column message. No fallback.
            - A known key with the wrong type → ``ConfigError`` naming the key,
              expected type, and got type.
            - ``rotation_mode`` with a valid-string-but-out-of-enum value →
              ``ConfigError`` (the silent-coerce OD1 forbids).
            - A missing known key → keeps the already-seeded default.
            - A recognized-but-retired key (``RETIRED_CONFIG_KEYS``) → tolerated and
              skipped SILENTLY (not a typo; a per-call warning on it is just noise).
            - A genuinely unknown key (a typo) → one calm stderr line, tolerated,
              skipped.

        Raises:
            ConfigError: On malformed TOML, a type mismatch, or an out-of-enum
                ``rotation_mode``.
        """
        config_path = os.path.join(self.mitos_dir, "config.toml")
        if not os.path.exists(config_path):
            return

        try:
            # tomllib requires BINARY mode — a text-mode handle raises TypeError.
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(
                f"Malformed config at {config_path}: {e}. "
                f"Fix the offending line or remove it."
            ) from e
        except OSError as e:
            # The file existed at the os.path.exists check but can't be read now
            # (permissions, a TOCTOU vanish, a directory). Keep the error vector
            # uniform: every failure to LOAD the config is a located ConfigError,
            # never a raw "Fatal Unexpected Error" — and never a silent default.
            raise ConfigError(f"Cannot read config at {config_path}: {e}.") from e

        for key, val in data.items():
            if key not in CONFIG_SCHEMA:
                # A recognized-but-retired key (deliberately dropped from the file
                # schema; its attribute still lives at a default, R12) is tolerated
                # SILENTLY — it is not a typo, so warning on it every call is pure
                # noise. A genuinely unknown key (a typo whose setting silently won't
                # take effect) still earns one calm, terse, screen-reader-clean line
                # to stderr (P9, no emoji) — that warning is the useful signal.
                if key not in RETIRED_CONFIG_KEYS:
                    print(
                        f"Warning: ignoring unrecognized config key "
                        f"'{key}' in {config_path}",
                        file=sys.stderr,
                    )
                continue

            expected = CONFIG_SCHEMA[key]
            if not _value_matches_type(val, expected):
                raise ConfigError(
                    f"Config key '{key}' in {config_path} expects "
                    f"{expected.__name__}, got {type(val).__name__} ({val!r}). "
                    f"Fix the value's type."
                )

            if key == "rotation_mode" and val not in ROTATION_MODES:
                allowed = ", ".join(sorted(ROTATION_MODES))
                raise ConfigError(
                    f"Config key 'rotation_mode' in {config_path} must be one of "
                    f"{{{allowed}}}, got {val!r}."
                )

            # Schema keys are exactly the attribute names (R12 surface).
            setattr(self, key, val)

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to dictionary form.

        Includes the convention-path attributes, the two dynamic-default qdrant
        keys, the kept-but-de-schema'd ``pending_threshold``, and the eight static
        schema keys (sourced from ``CONFIG_DEFAULTS`` so the set can't drift). No
        consumer binds this today; it exists for a future ``--json``/debug surface.

        Returns:
            A dictionary containing every configuration field.
        """
        result: Dict[str, Any] = {
            "workspace_dir": self.workspace_dir,
            "mitos_dir": self.mitos_dir,
            "db_path": self.db_path,
            "telemetry_path": self.telemetry_path,
            "qdrant_url": self.qdrant_url,
            "qdrant_collection": self.qdrant_collection,
            "pending_threshold": self.pending_threshold,
            "decisions_file": self.decisions_file,
            "questions_file": self.questions_file,
            "archive_dir": self.archive_dir,
        }
        # The eight static schema keys (incl. rotation_mode) from their one source.
        for key in CONFIG_DEFAULTS:
            result[key] = getattr(self, key)
        return result
