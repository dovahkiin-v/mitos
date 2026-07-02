"""Adversarial test suite for the Mitos configuration loader.

Covers the v0.1 nine-key schema, the single-source `CONFIG_DEFAULTS` map, and the
strict `tomllib`-based failure-mode policy (§5.2.6, OD1-symmetric): a malformed or
mistyped config is a loud, located `ConfigError`, never a silent fallback. Also
pins the R12 attribute surface every live consumer binds, and the cross-check that
keeps the render ceilings in lockstep with `renderer.py`'s constants.
"""

import os
import tempfile
import pytest
from mitos import renderer
from mitos.config import (
    MitosConfig,
    CONFIG_DEFAULTS,
    CONFIG_SCHEMA,
    ROTATION_MODES,
    default_collection_name,
    hint_due,
)
from mitos.errors import ConfigError, MitosError


def _write_config(workspace_dir: str, body: str) -> str:
    """Writes a `.mitos/config.toml` under workspace_dir and returns its path.

    Args:
        workspace_dir: The workspace root (the dir holding `.mitos/`).
        body: The raw TOML text to write.

    Returns:
        Absolute path to the written `config.toml`.
    """
    mitos_dir = os.path.join(workspace_dir, ".mitos")
    os.makedirs(mitos_dir, exist_ok=True)
    config_path = os.path.join(mitos_dir, "config.toml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(body)
    return config_path


# ---------------------------------------------------------------------------
# Defaults & the dynamic-default helper
# ---------------------------------------------------------------------------

def test_config_defaults() -> None:
    """A fresh workspace yields every documented default (static + dynamic)."""
    # Use a clean temp workspace (no .mitos/config.toml) so we test the DEFAULTS,
    # not whatever config.toml happens to live in the test runner's cwd.
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MitosConfig(tmpdir)
        # Static schema defaults — must equal CONFIG_DEFAULTS exactly (P11 source).
        for key, expected in CONFIG_DEFAULTS.items():
            assert getattr(config, key) == expected, key
        assert config.rotation_mode == "archive"
        assert config.rotation_volume_threshold_entries == 50
        assert config.stale_entry_window_days == 30
        assert config.embedding_cache_max_entries == 10_000
        # Dynamic defaults.
        assert config.qdrant_url == os.environ.get("QDRANT_URL", "http://localhost:7333")
        assert config.qdrant_collection == default_collection_name(tmpdir)
        assert config.qdrant_collection.startswith("mitos")
        # Kept-but-de-schema'd attribute + convention paths.
        assert config.pending_threshold == 30
        assert "graph.sqlite" in config.db_path
        assert "decisions.md" in config.decisions_file
        assert config.archive_dir.endswith(os.path.join("decisions", "archive"))


def test_default_collection_name_is_per_project() -> None:
    """Verifies the per-project Qdrant collection derivation + sanitization."""
    assert default_collection_name("/home/vinga/Forge/Blacksmith") == "mitos-blacksmith"
    assert default_collection_name("/x/workshop_mcp") == "mitos-workshop_mcp"
    assert default_collection_name("/x/My Project!") == "mitos-my-project"
    # Distinct projects -> distinct collections (the anti-contamination property).
    assert default_collection_name("/a/proj-one") != default_collection_name("/a/proj-two")
    # Degenerate path falls back to the bare "mitos" collection.
    assert default_collection_name("/") == "mitos"


# ---------------------------------------------------------------------------
# Valid file overrides
# ---------------------------------------------------------------------------

def test_config_file_loading_applies_valid_overrides() -> None:
    """A well-formed config.toml overlays recognized keys onto the defaults."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(
            tmpdir,
            'rotation_mode = "mark"\n'
            'rotation_volume_threshold_entries = 99\n'
            'qdrant_collection = "custom_collection"\n'
            'qdrant_url = "http://example:7333"\n',
        )
        config = MitosConfig(tmpdir)
        assert config.rotation_mode == "mark"
        assert config.rotation_volume_threshold_entries == 99
        assert config.qdrant_collection == "custom_collection"
        assert config.qdrant_url == "http://example:7333"
        # Untouched keys keep their defaults.
        assert config.stale_entry_window_days == CONFIG_DEFAULTS["stale_entry_window_days"]


def test_config_all_rotation_modes_accepted() -> None:
    """Every enum member of rotation_mode loads without raising."""
    for mode in sorted(ROTATION_MODES):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_config(tmpdir, f'rotation_mode = "{mode}"\n')
            assert MitosConfig(tmpdir).rotation_mode == mode


# ---------------------------------------------------------------------------
# Strict failure-mode policy (§5.2.6, OD1-symmetric)
# ---------------------------------------------------------------------------

def test_malformed_toml_raises_located_config_error() -> None:
    """Malformed TOML hard-fails with a ConfigError naming the file + line/col."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Unterminated string mid-file → tomllib reports a line/column.
        path = _write_config(tmpdir, 'rotation_mode = "archive\nqdrant_url = "x"\n')
        with pytest.raises(ConfigError) as exc:
            MitosConfig(tmpdir)
        msg = str(exc.value)
        assert path in msg
        assert "line" in msg  # the decoder's located message is carried through
        # No silent fallback: the error is raised, not swallowed to defaults.


def test_type_mismatch_raises_config_error() -> None:
    """A known int key given a quoted string hard-fails (TOML native typing)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Quoted "50" is a TOML string, not an int — the prototype would have
        # string-munged it; the strict loader refuses it.
        _write_config(tmpdir, 'rotation_volume_threshold_entries = "50"\n')
        with pytest.raises(ConfigError) as exc:
            MitosConfig(tmpdir)
        msg = str(exc.value)
        assert "rotation_volume_threshold_entries" in msg
        assert "int" in msg
        assert "str" in msg


def test_bool_rejected_for_int_key() -> None:
    """A TOML boolean never satisfies an int key (bool subclasses int)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(tmpdir, "embedding_cache_max_entries = true\n")
        with pytest.raises(ConfigError) as exc:
            MitosConfig(tmpdir)
        assert "bool" in str(exc.value)


# ---------------------------------------------------------------------------
# The first bool-typed key: conflict_check_on_sync (v0.2 Conflict sensor toggle)
# ---------------------------------------------------------------------------

def test_conflict_check_on_sync_defaults_true() -> None:
    """A workspace with no override falls back to the seeded default True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # No config.toml at all → the CONFIG_DEFAULTS value takes effect.
        assert MitosConfig(tmpdir).conflict_check_on_sync is True


def test_conflict_check_on_sync_missing_key_falls_back_to_default() -> None:
    """A config.toml present but WITHOUT the key still yields the True default."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(tmpdir, 'rotation_mode = "mark"\n')
        assert MitosConfig(tmpdir).conflict_check_on_sync is True


def test_conflict_check_on_sync_false_round_trips() -> None:
    """A native TOML ``false`` loads to the Python boolean False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(tmpdir, "conflict_check_on_sync = false\n")
        assert MitosConfig(tmpdir).conflict_check_on_sync is False


def test_conflict_check_on_sync_true_round_trips() -> None:
    """A native TOML ``true`` loads to the Python boolean True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(tmpdir, "conflict_check_on_sync = true\n")
        assert MitosConfig(tmpdir).conflict_check_on_sync is True


def test_conflict_check_on_sync_int_rejected() -> None:
    """A bare ``1`` never satisfies the bool key (no silent int→bool coerce)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(tmpdir, "conflict_check_on_sync = 1\n")
        with pytest.raises(ConfigError) as exc:
            MitosConfig(tmpdir)
        msg = str(exc.value)
        assert "conflict_check_on_sync" in msg
        assert "bool" in msg
        assert "int" in msg  # names the got-type too


def test_conflict_check_on_sync_quoted_string_rejected() -> None:
    """A quoted ``"true"`` (TOML string) never satisfies the bool key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(tmpdir, 'conflict_check_on_sync = "true"\n')
        with pytest.raises(ConfigError) as exc:
            MitosConfig(tmpdir)
        msg = str(exc.value)
        assert "conflict_check_on_sync" in msg
        assert "bool" in msg
        assert "str" in msg


def test_rotation_mode_out_of_enum_raises_config_error() -> None:
    """A correctly-typed but out-of-enum rotation_mode hard-fails (no silent coerce)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(tmpdir, 'rotation_mode = "delete"\n')
        with pytest.raises(ConfigError) as exc:
            MitosConfig(tmpdir)
        msg = str(exc.value)
        assert "rotation_mode" in msg
        assert "delete" in msg


def test_config_error_is_mitos_error() -> None:
    """ConfigError is a MitosError so the CLI's except-MitosError boundary catches it."""
    assert issubclass(ConfigError, MitosError)


def test_unreadable_config_raises_config_error() -> None:
    """A config path that exists but can't be read is a located ConfigError, not a raw OSError.

    Keeps the error vector uniform: every failure to LOAD the config is a clean
    ConfigError (never a silent default, never a 'Fatal Unexpected Error'). A
    directory at the config path triggers IsADirectoryError (an OSError) on the
    binary open — deterministic regardless of the test user.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Place a DIRECTORY where config.toml should be a file.
        os.makedirs(os.path.join(tmpdir, ".mitos", "config.toml"))
        with pytest.raises(ConfigError) as exc:
            MitosConfig(tmpdir)
        assert os.path.join(tmpdir, ".mitos", "config.toml") in str(exc.value)


# ---------------------------------------------------------------------------
# Missing & unknown keys
# ---------------------------------------------------------------------------

def test_missing_known_key_falls_back_to_default() -> None:
    """Deleting a key from a written file re-loads to the CONFIG_DEFAULTS value."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # File sets only rotation_mode; everything else must default.
        _write_config(tmpdir, 'rotation_mode = "prune"\n')
        config = MitosConfig(tmpdir)
        assert config.rotation_mode == "prune"
        assert config.rotation_volume_threshold_entries == CONFIG_DEFAULTS[
            "rotation_volume_threshold_entries"
        ]
        assert config.stale_entry_window_days == CONFIG_DEFAULTS["stale_entry_window_days"]


def test_retired_keys_silent_unknown_keys_warn(capsys: pytest.CaptureFixture) -> None:
    """Retired keys are tolerated SILENTLY; only genuinely-unknown keys warn.

    A recognized-but-retired key (`RETIRED_CONFIG_KEYS`: `pending_threshold`,
    `db_path`, `decisions_file`, `archive_dir`) was deliberately dropped from the
    file schema but is still recognized — its ATTRIBUTE survives at its default
    (R12) and its file occurrence is skipped with NO warning (it is not a typo, so
    warning on it every call is noise). A genuinely unknown key (a typo) still earns
    one calm stderr line — that warning is the signal the setting won't take effect.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(
            tmpdir,
            'pending_threshold = 99\n'
            'db_path = "/somewhere/else.sqlite"\n'
            'frobnicate = 1\n'
            'rotation_mode = "mark"\n',
        )
        config = MitosConfig(tmpdir)
        # Recognized key still applies.
        assert config.rotation_mode == "mark"
        # Retired file keys are ignored — the attributes keep their defaults.
        assert config.pending_threshold == 30
        assert "graph.sqlite" in config.db_path
        err = capsys.readouterr().err
        # Retired keys are tolerated silently — no per-invocation noise.
        assert "pending_threshold" not in err
        assert "db_path" not in err
        # A genuine typo still earns one calm stderr line (P9: terse, no emoji).
        assert "frobnicate" in err
        assert "Traceback" not in err


# ---------------------------------------------------------------------------
# R12 attribute surface
# ---------------------------------------------------------------------------

def test_r12_attribute_surface_preserved() -> None:
    """Every consumer-bound MitosConfig attribute exists after construction (R12)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MitosConfig(tmpdir)
        # The nine prototype consumer-bound attributes (§3 / §11) ...
        for attr in (
            "workspace_dir",
            "mitos_dir",
            "db_path",
            "decisions_file",
            "archive_dir",
            "qdrant_url",
            "qdrant_collection",
            "rotation_mode",
            "pending_threshold",
        ):
            assert hasattr(config, attr), attr
        # ... plus the seven new static schema attributes.
        for attr in CONFIG_DEFAULTS:
            assert hasattr(config, attr), attr


def test_post_construction_attribute_assignment_untouched() -> None:
    """Tests/consumers set config attributes directly after construction — still works.

    The strict loader governs only the file→config path; plain attribute assignment
    (the pattern many consumer tests use: `config.db_path = ...`) is unaffected.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MitosConfig(tmpdir)
        config.db_path = "/custom/graph.sqlite"
        config.pending_threshold = 1
        assert config.db_path == "/custom/graph.sqlite"
        assert config.pending_threshold == 1


def test_to_dict_carries_full_surface() -> None:
    """to_dict exposes the convention paths, dynamic keys, and all eight schema keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        d = MitosConfig(tmpdir).to_dict()
        for key in CONFIG_DEFAULTS:
            assert key in d, key
        for key in ("db_path", "qdrant_url", "qdrant_collection", "pending_threshold",
                    "decisions_file", "archive_dir", "workspace_dir", "mitos_dir"):
            assert key in d, key


# ---------------------------------------------------------------------------
# Existing-file coherence (§5.2.7) — the live prototype-shaped file
# ---------------------------------------------------------------------------

def test_prototype_shaped_config_loads_clean(capsys: pytest.CaptureFixture) -> None:
    """The real `mitos init`-seeded file (incl. an inline comment) loads without raising.

    Mirrors the live `.mitos/config.toml`: `rotation_mode` carries a trailing inline
    comment (which the hand-rolled parser mangled and silently defaulted; tomllib
    parses it cleanly), `pending_threshold` is now silently tolerated (a recognized
    retired key), and the `qdrant_*` keys apply.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(
            tmpdir,
            "# Mitos Workspace Configuration\n"
            'rotation_mode = "archive" # "archive" | "mark" | "prune"\n'
            "pending_threshold = 30\n"
            'qdrant_url = "http://localhost:7333"\n'
            'qdrant_collection = "mitos-mitos-pub"\n',
        )
        config = MitosConfig(tmpdir)  # must NOT raise
        # Inline comment stripped by tomllib → the clean value applies.
        assert config.rotation_mode == "archive"
        assert config.qdrant_url == "http://localhost:7333"
        assert config.qdrant_collection == "mitos-mitos-pub"
        # pending_threshold file key silently tolerated; attribute keeps its default.
        assert config.pending_threshold == 30
        # The real seeded file now loads with a CLEAN stderr — no per-invocation
        # noise on the recognized-but-retired `pending_threshold` key.
        err = capsys.readouterr().err
        assert "pending_threshold" not in err
        assert err.strip() == ""


# ---------------------------------------------------------------------------
# Single-source render-ceiling cross-check (Decision 5)
# ---------------------------------------------------------------------------

def test_render_defaults_match_renderer_constants() -> None:
    """The render ceilings in CONFIG_DEFAULTS equal renderer.py's constants.

    config.py is a lower-tier leaf (importing renderer would invert tiers), so the
    literals are pinned here by a test instead of a runtime import — they cannot
    silently drift. V4 wires the renderer to read the config key (single runtime
    source then).
    """
    assert (
        CONFIG_DEFAULTS["render_global_overflow_warn_chars"]
        == renderer.GLOBAL_OVERFLOW_WARN_CHARS
    )
    assert (
        CONFIG_DEFAULTS["render_scope_overflow_warn_chars"]
        == renderer.SCOPE_OVERFLOW_WARN_CHARS
    )


def test_schema_covers_ten_keys_and_defaults_are_the_static_eight() -> None:
    """CONFIG_SCHEMA recognizes ten file keys; CONFIG_DEFAULTS holds the static eight.

    The two dynamic qdrant keys are recognized + validated but defaulted in
    __init__, so they are in CONFIG_SCHEMA but not CONFIG_DEFAULTS. v0.2's
    ``conflict_check_on_sync`` is in BOTH (static default True), so it lifts the
    counts to ten/eight but leaves the qdrant-only difference unchanged.
    """
    assert len(CONFIG_SCHEMA) == 10
    assert len(CONFIG_DEFAULTS) == 8
    assert set(CONFIG_DEFAULTS) < set(CONFIG_SCHEMA)
    assert set(CONFIG_SCHEMA) - set(CONFIG_DEFAULTS) == {"qdrant_url", "qdrant_collection"}


# ---------------------------------------------------------------------------
# Debounce helper (unchanged from the prototype suite)
# ---------------------------------------------------------------------------

def test_hint_due_debounces_within_window(tmp_path) -> None:
    """hint_due fires once per window per key, and never raises (fail-silent debounce).

    Backs both the MCP-server hint and the render-overflow summary, so a recurring
    nudge fires at most once per window instead of on every call. (The autouse
    hermetic fixture redirects XDG_CACHE_HOME into a tmp dir, so this never touches
    the real ~/.cache.)
    """
    key = str(tmp_path / "proj")
    # First call in the window is due (and stamps); the next is debounced.
    assert hint_due("overflow_test.json", key, 10_000) is True
    assert hint_due("overflow_test.json", key, 10_000) is False
    # A different key is tracked independently.
    assert hint_due("overflow_test.json", key + "-other", 10_000) is True
    # A different cache file is a separate namespace, so it fires again for the same key.
    assert hint_due("other_test.json", key, 10_000) is True
    # A zero-second window always re-fires (the elapsed time is never < 0).
    assert hint_due("overflow_test.json", key, 0) is True
