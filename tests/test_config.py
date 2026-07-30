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


# The derivation itself — its step order, its digest, its non-injective basename
# classes, and the retirement of the `qdrant_collection` file override — is pinned by
# `tests/test_collection_derivation.py`. This module keeps only the attribute-level
# rows (above): that `MitosConfig` binds whatever the function returns.


# ---------------------------------------------------------------------------
# Valid file overrides
# ---------------------------------------------------------------------------

def test_config_file_loading_applies_valid_overrides() -> None:
    """A well-formed config.toml overlays recognized keys onto the defaults.

    `rotation_mode` deliberately does NOT appear: its only non-deprecated value is
    `archive`, which is also its default, so it can no longer demonstrate that an
    override was APPLIED — the assertion would pass even against a loader that
    ignored the file. Its own contract is pinned by the deprecation tests below.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(
            tmpdir,
            'rotation_volume_threshold_entries = 99\n'
            'qdrant_url = "http://example:7333"\n',
        )
        config = MitosConfig(tmpdir)
        assert config.rotation_volume_threshold_entries == 99
        assert config.qdrant_url == "http://example:7333"
        # Untouched keys keep their defaults.
        assert config.stale_entry_window_days == CONFIG_DEFAULTS["stale_entry_window_days"]


def test_config_every_enum_member_still_loads_without_raising() -> None:
    """No value in `ROTATION_MODES` bricks the workspace at load.

    Retargeted from "each mode yields itself", and stated as the bare no-raise
    property it now is: every member resolves to `archive`, so any per-mode assertion
    would read as discriminating while checking nothing. That resolution is a real
    contract, pinned by the deprecation tests below; THIS test exists for the reason
    the epoch-1 shape exists at all — a `ConfigError` here fires inside
    `MitosConfig()` construction, which `cli.py` performs before verb dispatch, so it
    would take `mitos status` down with it.
    """
    for mode in sorted(ROTATION_MODES):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_config(tmpdir, f'rotation_mode = "{mode}"\n')
            MitosConfig(tmpdir)  # must not raise


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
        # File sets only stale_entry_window_days; everything else must default.
        # (This used `rotation_mode = "prune"` as its one set key, which no longer
        # demonstrates an applied override — `prune` is deprecated and pins to
        # `archive`, which is also the default, so the assertion would pass either
        # way. A key whose set value differs from its default is the real test.)
        _write_config(tmpdir, 'stale_entry_window_days = 7\n')
        config = MitosConfig(tmpdir)
        assert config.stale_entry_window_days == 7
        assert config.rotation_volume_threshold_entries == CONFIG_DEFAULTS[
            "rotation_volume_threshold_entries"
        ]
        assert config.rotation_mode == CONFIG_DEFAULTS["rotation_mode"]


def test_retired_keys_silent_unknown_keys_warn(capsys: pytest.CaptureFixture) -> None:
    """Retired keys are tolerated SILENTLY; only genuinely-unknown keys warn.

    A recognized-but-retired key (`RETIRED_CONFIG_KEYS`: `pending_threshold`,
    `db_path`, `decisions_file`, `archive_dir`, `qdrant_collection`) was deliberately
    dropped from the file schema but is still recognized — its ATTRIBUTE survives at
    its default (R12) and its file occurrence is skipped with NO warning (it is not a
    typo, so warning on it every call is noise). A genuinely unknown key (a typo)
    still earns one calm stderr line — that warning is the signal the setting won't
    take effect.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(
            tmpdir,
            'pending_threshold = 99\n'
            'db_path = "/somewhere/else.sqlite"\n'
            'frobnicate = 1\n'
            'stale_entry_window_days = 7\n',
        )
        config = MitosConfig(tmpdir)
        # Recognized key still applies. (Was `rotation_mode = "mark"`, retargeted:
        # `mark` is deprecated and pins to `archive`, so it no longer demonstrates
        # an applied override.)
        assert config.stale_entry_window_days == 7
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
        # ... plus the eight new static schema attributes.
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
    parses it cleanly), `qdrant_url` applies, and BOTH `pending_threshold` and
    `qdrant_collection` are now silently tolerated retired keys — the second one
    inert rather than applied, which is the whole point: this file's pin is exactly
    the shape `mitos init` used to write, and it no longer decides anything.
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
        # The pin is INERT: the collection stays the one derived from this path.
        assert config.qdrant_collection == default_collection_name(tmpdir)
        assert config.qdrant_collection != "mitos-mitos-pub"
        assert config.inert_file_keys["qdrant_collection"] == "mitos-mitos-pub"
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


def test_schema_covers_nine_keys_and_defaults_are_the_static_eight() -> None:
    """CONFIG_SCHEMA recognizes nine file keys; CONFIG_DEFAULTS holds the static eight.

    ``qdrant_url`` is recognized + validated but defaulted in __init__, so it is in
    CONFIG_SCHEMA and not CONFIG_DEFAULTS — and it is now the ONLY such key.
    ``qdrant_collection`` was the other one until it was retired from the file schema
    entirely (its value is derived from the workspace path and not overridable), which
    is what takes the count from ten to nine. v0.2's ``conflict_check_on_sync`` is in
    BOTH (static default True).
    """
    assert len(CONFIG_SCHEMA) == 9
    assert len(CONFIG_DEFAULTS) == 8
    assert set(CONFIG_DEFAULTS) < set(CONFIG_SCHEMA)
    assert set(CONFIG_SCHEMA) - set(CONFIG_DEFAULTS) == {"qdrant_url"}


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


# ---------------------------------------------------------------------------
# rotation_mode narrows to `archive` — epoch-1 accept-and-warn deprecation
# ---------------------------------------------------------------------------

def test_deprecated_rotation_modes_are_accepted_and_pinned_to_archive() -> None:
    """`mark` and `prune` still LOAD, and both behave as `archive`.

    Removing them from the enum would raise ``ConfigError`` at load — which fires
    before verb dispatch, bricking the whole workspace including ``mitos status``,
    the one command needed to diagnose it. A wall, not a vector, and a breaking
    change with no epoch. So the values stay accepted and the attribute is pinned.
    """
    from mitos.config import DEPRECATED_ROTATION_MODES

    assert DEPRECATED_ROTATION_MODES == {"mark", "prune"}
    for mode in sorted(DEPRECATED_ROTATION_MODES):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_config(tmpdir, f'rotation_mode = "{mode}"\n')
            config = MitosConfig(tmpdir)
            assert config.rotation_mode == "archive", (
                f"{mode!r} must behave as archive, not as itself"
            )
            assert config.deprecated_rotation_mode == mode, (
                "the original value must survive so the CLI can name it in the warning"
            )


def test_archive_is_not_flagged_as_deprecated() -> None:
    """The surviving mode carries no deprecation marker — a clean project stays quiet."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(tmpdir, 'rotation_mode = "archive"\n')
        config = MitosConfig(tmpdir)
        assert config.rotation_mode == "archive"
        assert config.deprecated_rotation_mode is None


def test_absent_rotation_mode_carries_no_deprecation_marker() -> None:
    """A workspace with no config.toml at all is healthy and quiet (empty states first-class)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MitosConfig(tmpdir)
        assert config.rotation_mode == "archive"
        assert config.deprecated_rotation_mode is None


def test_unknown_rotation_mode_still_hard_fails() -> None:
    """Deprecation is not a loosening: a typo is still a loud, located ConfigError.

    ``config-loader-rotation-mode-enum-hard-fail`` commits to *loud validation*, not
    to three modes. Silent coercion is what it forbids — and the epoch-1 warning is
    exactly what keeps the coercion from being silent.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(tmpdir, 'rotation_mode = "archiv"\n')
        with pytest.raises(ConfigError) as exc:
            MitosConfig(tmpdir)
        assert "rotation_mode" in str(exc.value)


def test_config_loading_never_writes_to_stdout(capsys) -> None:
    """A deprecated mode must not print to stdout — it would corrupt MCP's JSON-RPC.

    ``mcp_server.py`` constructs ``MitosConfig()`` per tool call over a stdio
    JSON-RPC channel, so anything on stdout there is protocol corruption rather than
    noise. The warning is the CLI dispatcher's job, on stderr; the loader stays mute.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(tmpdir, 'rotation_mode = "mark"\n')
        MitosConfig(tmpdir)
    captured = capsys.readouterr()
    assert captured.out == "", f"the config loader wrote to stdout: {captured.out!r}"
    assert captured.err == "", (
        "the loader must not warn either — per-construction warnings fire twice on "
        "`mitos status` and once per MCP tool call"
    )


def test_deprecated_rotation_mode_warns_once_at_dispatch(tmp_path, monkeypatch, capsys) -> None:
    """The CLI warns exactly once per invocation, on stderr, naming the mode.

    "Once per invocation" is load-bearing: `cmd_status` builds a SECOND MitosConfig
    of its own, so a warning inside the loader prints twice on `mitos status`. It is
    emitted at verb dispatch instead, which needs no once-flag — a fresh process per
    CLI invocation — and structurally cannot double-fire.
    """
    import sys as _sys
    from mitos.cli import main

    mitos_dir = tmp_path / ".mitos"
    mitos_dir.mkdir()
    (mitos_dir / "config.toml").write_text('rotation_mode = "mark"\n', encoding="utf-8")

    monkeypatch.setattr(_sys, "argv", ["mitos", "-C", str(tmp_path), "status"])
    try:
        main()
    except SystemExit:
        pass

    err = capsys.readouterr().err
    assert err.count("rotation_mode = 'mark' is deprecated") == 1, (
        f"expected exactly one deprecation warning, got:\n{err}"
    )
    assert "archive" in err, "the warning must name what happens instead"


def test_clean_rotation_mode_warns_nothing_at_dispatch(tmp_path, monkeypatch, capsys) -> None:
    """A healthy workspace prints no deprecation line — clean projects read clean."""
    import sys as _sys
    from mitos.cli import main

    mitos_dir = tmp_path / ".mitos"
    mitos_dir.mkdir()
    (mitos_dir / "config.toml").write_text('rotation_mode = "archive"\n', encoding="utf-8")

    monkeypatch.setattr(_sys, "argv", ["mitos", "-C", str(tmp_path), "status"])
    try:
        main()
    except SystemExit:
        pass

    assert "deprecated" not in capsys.readouterr().err


def test_status_with_a_path_warns_about_that_workspace(tmp_path, monkeypatch, capsys) -> None:
    """`mitos status <path>` warns for the workspace it INSPECTS, not the CWD.

    `cmd_status` builds its own config for the path argument, so warning on the
    dispatcher's CWD config stayed silent about exactly the workspace the operator
    asked to diagnose — while `mitos status` is the command this deprecation's own
    rationale names as the one that must keep working.
    """
    import os as _os
    import sys as _sys
    from mitos.cli import main

    target = tmp_path / "target"
    (target / ".mitos").mkdir(parents=True)
    (target / ".mitos" / "config.toml").write_text('rotation_mode = "prune"\n', encoding="utf-8")
    # From phase 3b the positional is a project selector, so the target must carry
    # the whole validity triple — `.mitos/config.toml` AND `decisions.md` — or the
    # call lands on the targeting error before any warning can fire.
    (target / "decisions.md").write_text("# Decisions\n", encoding="utf-8")

    clean_cwd = tmp_path / "clean"
    (clean_cwd / ".mitos").mkdir(parents=True)
    monkeypatch.chdir(clean_cwd)

    monkeypatch.setattr(_sys, "argv", ["mitos", "status", str(target)])
    try:
        main()
    except SystemExit:
        pass

    err = capsys.readouterr().err
    assert err.count("rotation_mode = 'prune' is deprecated") == 1, (
        f"expected one warning for the inspected workspace, got:\n{err}"
    )


def test_status_with_a_path_stays_quiet_about_a_deprecated_cwd(tmp_path, monkeypatch, capsys) -> None:
    """The converse: a deprecated CWD must not warn when reporting on a clean path."""
    import sys as _sys
    from mitos.cli import main

    clean_target = tmp_path / "target"
    (clean_target / ".mitos").mkdir(parents=True)
    # The validity triple, as above — and the config.toml stays CLEAN (no
    # rotation_mode), which is what this row is about.
    (clean_target / ".mitos" / "config.toml").write_text("", encoding="utf-8")
    (clean_target / "decisions.md").write_text("# Decisions\n", encoding="utf-8")

    dirty_cwd = tmp_path / "dirty"
    (dirty_cwd / ".mitos").mkdir(parents=True)
    (dirty_cwd / ".mitos" / "config.toml").write_text('rotation_mode = "mark"\n', encoding="utf-8")
    monkeypatch.chdir(dirty_cwd)

    monkeypatch.setattr(_sys, "argv", ["mitos", "status", str(clean_target)])
    try:
        main()
    except SystemExit:
        pass

    assert "deprecated" not in capsys.readouterr().err
