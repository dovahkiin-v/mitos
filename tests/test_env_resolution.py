"""Tests for the functional env resolver (`mitos/env.py`) and `MitosConfig.env`.

Phase 2b of the global-MCP-registry vision. The leaf answers *"with which
keys?"* for a **named target** without writing to ``os.environ``; the carrier
rides the per-call ``MitosConfig`` so the orchestrators that already hold a
config need nothing new. Nothing consumes a *key* from the carrier yet (that is
2c) — its one live consumer here is ``MitosConfig``'s own ``QDRANT_URL``
resolution, which is why the carrier has to exist before the key sites can be
routed onto it.

Every row drives real files under ``tmp_path`` with ``monkeypatch.setenv`` /
``delenv``. No mocks: the thing under test is a file parse and a three-tier
lookup, and a mocked one proves nothing about either.

**The module-autouse key strip is load-bearing, not hygiene.** Six test modules
write the repo's real ``.env`` into ``os.environ`` at *import* time — and two of
them are not ``*_live.py`` (``test_check_hook_recipe.py``,
``test_adversarial_invariants.py``, the second an unconditional overwrite).
Conftest's ``hermetic_mitos_env`` now strips the three LLM credential names from
every non-live test (the importer CI-red incident, 2026-08), so those import-time
writes no longer reach a test body — but it strips only the credentials.
``_keyless`` below stays for the wider set this module resolves (``QDRANT_URL``
plus every ``RESOLVED_ENV_KEYS`` name) and for ``_unset``'s guaranteed-teardown
semantics, which the rows that raw-write a name depend on. Each row still opts
*in* to the tier it means.

Consequently: assert on **specific keys**, never on a whole-dict equality that
could print a live credential into a pytest report (and, on a ``vision/**`` push,
a CI log).
"""

import ast
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from mitos import cli, env, models
from mitos.config import MitosConfig, RESOLVED_ENV_KEYS, global_env_path
from mitos.env import (
    ENV_TIERS,
    TIER_ENVIRONMENT,
    TIER_GLOBAL_ENV,
    TIER_PROJECT_ENV,
    ResolvedValue,
    parse_env_file,
    resolve_key,
    resolve_values,
)
from mitos.store import GraphStore


def _unset(monkeypatch, name: str) -> None:
    """Removes `name` for the test AND guarantees it stays removed at teardown.

    ``monkeypatch.delenv(name, raising=False)`` on an **already-absent** name
    records nothing, so monkeypatch's undo has nothing to undo — and a raw
    ``os.environ`` write is unavoidable wherever a row means to exercise a leak
    (group 5's two fixture-net rows do exactly that; until 5c the whole of group
    5 did, because the entry-time load it pinned wrote the environment itself).
    The two failure shapes, both measured: a raw write with no later ``delenv``
    survives teardown outright, and a raw write followed by ``monkeypatch.delenv``
    is *worse* — the delete records the leaked value and undo faithfully **puts it
    back**. Either way ``GEMINI_API_KEY`` escapes into every module collected
    after this one, which is precisely the order-dependent class the module
    docstring above warns about.

    Setting the name before deleting it forces monkeypatch to record the absence
    (or the real prior value), so its undo removes the raw write too. Correct in
    both directions: a name that was genuinely exported is restored to its own
    value, because the records unwind in reverse.
    """
    monkeypatch.setenv(name, "")
    monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def _keyless(monkeypatch) -> None:
    """Strips every name this module resolves, so each row builds its own tiers.

    ``hermetic_mitos_env`` (conftest, autouse) redirects the XDG dirs — so
    ``global_env_path()`` already lands inside ``tmp_path`` and tier 3 is empty by
    default — and since the importer CI-red incident it also strips the three LLM
    credential names. This module resolves more names than that (``QDRANT_URL``,
    the whole of ``RESOLVED_ENV_KEYS``), and its rows raw-write names that need
    ``_unset``'s guaranteed teardown, so the module-level strip stays.

    (``NEW`` used to ride here too — the third key group 5's entry-load row
    wrote. That row went with the mechanism in 5c; no row writes the name now.)
    """
    for name in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
                 "QDRANT_URL", *RESOLVED_ENV_KEYS):
        _unset(monkeypatch, name)


def _write(path, text: str) -> str:
    """Writes a `.env` (creating parents) and returns its path as a string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


# --- group 1: parse_env_file ------------------------------------------------

def test_comments_blanks_and_non_assignments_are_skipped(tmp_path):
    """The three shapes a real `.env` carries beside its assignments."""
    path = _write(tmp_path / ".env",
                  "# a comment\n\n   \nnot an assignment\nK=v\n")
    assert parse_env_file(path) == {"K": "v"}


def test_whitespace_around_the_key_and_the_equals_is_tolerated(tmp_path):
    """`GEMINI_API_KEY = x` is a shape humans write and must resolve.

    (The ``cli`` reader 2c retired — the tree's third parse — did *not* tolerate
    it, which is one reason it was folded onto this one. ``cli._upsert_env_var``
    was the last holdout of the class, a *writer*; 5c brought its key match onto
    the same shape — see group 5.)
    """
    path = _write(tmp_path / ".env", "  GEMINI_API_KEY = x  \n")
    assert parse_env_file(path) == {"GEMINI_API_KEY": "x"}


@pytest.mark.parametrize(
    "line, expected",
    [
        ('K="v"', "v"),
        ("K='v'", "v"),
        ("K=\"'v'\"", "v"),
        # The MIRROR of the row above does NOT strip, because the strip is
        # ordered: double quotes come off first, so a single-outer wrapping
        # survives. Pinned deliberately — spelling it `.strip("'").strip('"')`,
        # or `.strip("\"'")`, silently changes behaviour for every `.env` that
        # uses single-outer quoting.
        ("K='\"v\"'", '"v"'),
        ('K="a"b"', 'a"b'),
        ("K=http://h:7333/?a=b", "http://h:7333/?a=b"),
    ],
)
def test_the_quote_strip_matches_the_shipped_parse_character_for_character(
    tmp_path, line, expected
):
    """Quote handling is inherited, order-dependent, and asymmetric."""
    path = _write(tmp_path / ".env", line + "\n")
    assert parse_env_file(path) == {"K": expected}


def test_an_empty_assignment_is_skipped(tmp_path):
    """`init` scaffolds `GEMINI_API_KEY=`; it must not read as a resolved value."""
    path = _write(tmp_path / ".env", "GEMINI_API_KEY=\nOTHER=v\n")
    assert parse_env_file(path) == {"OTHER": "v"}


def test_the_first_non_empty_assignment_wins_within_a_file(tmp_path):
    """First-wins, not last — the retired entry load's `key not in os.environ`.

    A plain ``dict[key] = value`` loop is last-wins and inverts this silently. The
    scaffolded-empty-slot-then-real-value shape agrees either way; two real values
    do not, which is why both rows are here.
    """
    empty_then_real = _write(tmp_path / "a" / ".env",
                             "K=\nK=real\n")
    assert parse_env_file(empty_then_real) == {"K": "real"}

    two_real = _write(tmp_path / "b" / ".env", "K=a\nK=b\n")
    assert parse_env_file(two_real) == {"K": "a"}


def test_an_absent_file_parses_to_empty(tmp_path):
    """No `.env` is the overwhelmingly common case, and it is not an error."""
    assert parse_env_file(str(tmp_path / "nope" / ".env")) == {}


def test_a_directory_named_env_parses_to_empty(tmp_path):
    """`IsADirectoryError` is an `OSError`; inherited behaviour, pinned."""
    d = tmp_path / ".env"
    d.mkdir()
    assert parse_env_file(str(d)) == {}


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores mode bits, so a 0o000 file stays readable",
)
def test_an_unreadable_file_parses_to_empty(tmp_path):
    """`PermissionError` is an `OSError`; inherited behaviour, pinned."""
    path = _write(tmp_path / ".env", "K=v\n")
    os.chmod(path, 0o000)
    try:
        assert parse_env_file(path) == {}
    finally:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def test_a_non_utf8_file_parses_to_empty_instead_of_raising(tmp_path):
    """The one row here that is NEW coverage rather than an inherited pin.

    Before this leaf, a non-UTF-8 `.env` raised ``UnicodeDecodeError`` out of
    the entry-time load 5c deleted — which is **not** an ``OSError``, so its
    ``except`` missed it and ``main()``'s generic arm rendered ``Fatal
    Unexpected Error``
    for *every verb*, ``mitos status`` included. Whole-file, not partial: the
    first line parses cleanly and is still discarded, because a file that failed
    mid-decode is not a source of truth for the bytes before the bad one.
    """
    path = tmp_path / ".env"
    path.write_bytes(b"K=v\n\xff\xfe\n")
    assert parse_env_file(str(path)) == {}


# --- group 2: the three tiers ----------------------------------------------

def _tiers(tmp_path, *, project: str = None, glob: str = None) -> str:
    """Builds the two file tiers and returns the target dir.

    ``global_env_path()`` already points inside ``tmp_path`` (conftest's XDG
    redirect), so tier 3 is built by writing that exact path — never a
    hand-rolled ``~/.config``.
    """
    target = tmp_path / "proj"
    target.mkdir(exist_ok=True)
    if project is not None:
        _write(target / ".env", project)
    if glob is not None:
        _write(Path(global_env_path()), glob)
    return str(target)


def test_real_env_beats_both_files(tmp_path, monkeypatch):
    """Tier 1 wins whatever the files say — an explicit export is an override."""
    monkeypatch.setenv("GEMINI_API_KEY", "ENVKEY")
    target = _tiers(tmp_path, project="GEMINI_API_KEY=PROJKEY\n",
                    glob="GEMINI_API_KEY=GLOBALKEY\n")
    assert resolve_key("GEMINI_API_KEY", target, global_env_path()) == ResolvedValue(
        "ENVKEY", TIER_ENVIRONMENT
    )


def test_an_exported_empty_variable_masks_both_files(tmp_path, monkeypatch):
    """Tier 1 tests PRESENCE, not a non-empty value — and this is why.

    ``env GEMINI_API_KEY= ANTHROPIC_API_KEY= mitos …`` is how mitos's own
    PATTERNS produces a keyless run on a key-bearing dev box; a bare unset does
    not work, because the global ``.env`` refills it. Under a uniform non-empty
    test the empty export stops masking, falls through to the files, resolves the
    real global key and fires real billed LLM calls. Money and hermeticity, not
    style.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "")
    target = _tiers(tmp_path, project="GEMINI_API_KEY=PROJKEY\n",
                    glob="GEMINI_API_KEY=GLOBALKEY\n")
    assert resolve_key("GEMINI_API_KEY", target, global_env_path()) == ResolvedValue(
        "", TIER_ENVIRONMENT
    )


def test_an_empty_project_slot_falls_through_to_the_global_key(tmp_path):
    """Tiers 2 and 3 test a NON-EMPTY value — the setup the tool recommends.

    ``mitos init`` scaffolds an empty ``GEMINI_API_KEY=`` line under a comment
    telling the user to set the key once globally. A resolver testing key
    *presence* at tier 2 finds ``""`` there, stops, and every project that
    followed that advice loses its key.
    """
    target = _tiers(tmp_path, project="GEMINI_API_KEY=\n",
                    glob="GEMINI_API_KEY=GLOBALKEY\n")
    assert resolve_key("GEMINI_API_KEY", target, global_env_path()) == ResolvedValue(
        "GLOBALKEY", TIER_GLOBAL_ENV
    )


def test_a_project_key_beats_a_global_key(tmp_path):
    """Tier 2 above tier 3: a workspace may override the machine's shared key."""
    target = _tiers(tmp_path, project="GEMINI_API_KEY=PROJKEY\n",
                    glob="GEMINI_API_KEY=GLOBALKEY\n")
    assert resolve_key("GEMINI_API_KEY", target, global_env_path()) == ResolvedValue(
        "PROJKEY", TIER_PROJECT_ENV
    )


@pytest.mark.parametrize(
    "project, glob",
    [(None, None), ("OTHER=v\n", None), (None, "OTHER=v\n"), ("GEMINI_API_KEY=\n", None)],
)
def test_nothing_anywhere_resolves_to_a_null_report(tmp_path, project, glob):
    """Absent project file, absent global file, absent both, empty slot only."""
    target = _tiers(tmp_path, project=project, glob=glob)
    assert resolve_key("GEMINI_API_KEY", target, global_env_path()) == ResolvedValue(
        None, None
    )


def test_the_tier_vocabulary_is_pinned_to_its_literals():
    """The three strings `mitos status` already prints, byte-for-byte.

    Per-constant literals *and* the set, on the routing/parser failure-code
    idiom: the set alone would stay green through two constants swapping values.
    They are join keys — 2c routes ``_gemini_key_source`` onto this report, and a
    reworded tier would move ``status``'s output as a side effect.
    """
    assert TIER_ENVIRONMENT == "environment"
    assert TIER_PROJECT_ENV == "project .env"
    assert TIER_GLOBAL_ENV == "global .env"
    assert ENV_TIERS == frozenset({"environment", "project .env", "global .env"})


def test_the_tier_strings_match_the_shipped_key_source_report(tmp_path, monkeypatch):
    """The second net on the same three literals, driven through `cli`.

    Written in 2b against the *pre-routing* ``_gemini_key_source``, as independent
    evidence that the leaf's vocabulary was the shipped one rather than a
    plausible re-spelling. 2c routed that function onto ``resolve_key``, so the
    two are no longer independent — the row survives as the end-to-end pin that
    the reroute moved no string, and the tier-order rows in
    ``tests/test_env_routing.py`` are where the new body's precedence is proven.
    """
    proj = tmp_path / "proj"
    proj.mkdir()
    # The environment rung first, while no file carries the key. The order was
    # load-bearing against the old file-first body (a global `.env` written first
    # would have shadowed this rung); under the env-first resolver it is merely
    # the honest sequence — each rung asserted while the ones above it are empty.
    monkeypatch.setenv("GEMINI_API_KEY", "ENVKEY")
    assert cli._gemini_key_source(str(proj)) == TIER_ENVIRONMENT

    monkeypatch.delenv("GEMINI_API_KEY")
    cli.cmd_set_key("GLOBALKEY", workspace_dir=None, is_global=True)
    assert cli._gemini_key_source(str(proj)) == TIER_GLOBAL_ENV
    (proj / ".env").write_text("GEMINI_API_KEY=PROJKEY\n", encoding="utf-8")
    assert cli._gemini_key_source(str(proj)) == TIER_PROJECT_ENV


def test_resolve_values_agrees_with_resolve_key_name_by_name(tmp_path, monkeypatch):
    """One layering, two entry points — never two answers.

    The fixture spans all four outcomes at once: an exported value, an exported
    EMPTY one, a project value, a global value, and a name nothing carries.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ENVKEY")
    monkeypatch.setenv("MITOS_MODEL_OVERRIDE_FLASH", "")
    target = _tiers(
        tmp_path,
        project="ANTHROPIC_API_KEY=PROJ\nQDRANT_URL=http://proj:7333\n"
                "MITOS_MODEL_OVERRIDE_FLASH=proj-model\n",
        glob="MITOS_MODEL_OVERRIDE_SONNET=global-model\n",
    )
    values = resolve_values(RESOLVED_ENV_KEYS, target, global_env_path())

    for name in RESOLVED_ENV_KEYS:
        report = resolve_key(name, target, global_env_path())
        if report.tier is None:
            assert name not in values, name
        else:
            assert values[name] == report.value, name

    # …and the outcomes themselves, so the agreement above is not vacuous.
    assert values["ANTHROPIC_API_KEY"] == "ENVKEY"
    assert values["MITOS_MODEL_OVERRIDE_FLASH"] == ""       # exported empty: resolved
    assert values["QDRANT_URL"] == "http://proj:7333"
    assert values["MITOS_MODEL_OVERRIDE_SONNET"] == "global-model"
    assert "GEMINI_API_KEY" not in values                    # unresolved: ABSENT


def test_resolution_never_writes_to_the_process_environment(tmp_path):
    """The whole point of the leaf, asserted as a property rather than assumed.

    A resolver that quietly promoted its answer into ``os.environ`` would make
    every later call for a *different* project resolve this one's key — the exact
    launch-directory leak the always-on server topology cannot tolerate.
    """
    target = _tiers(tmp_path, project="GEMINI_API_KEY=PROJKEY\n",
                    glob="ANTHROPIC_API_KEY=GLOBALKEY\n")
    before = dict(os.environ)
    resolve_key("GEMINI_API_KEY", target, global_env_path())
    resolve_values(RESOLVED_ENV_KEYS, target, global_env_path())
    assert dict(os.environ) == before


def test_two_targets_resolve_independently_from_one_process(tmp_path):
    """Asked twice about two projects, the leaf answers honestly both times.

    I7's mechanism (its end-to-end proof is 2c/5c, once a key consumer exists).
    """
    a = _write(tmp_path / "a" / ".env", "GEMINI_API_KEY=AKEY\n")
    b = _write(tmp_path / "b" / ".env", "GEMINI_API_KEY=BKEY\n")
    assert resolve_key("GEMINI_API_KEY", os.path.dirname(a),
                       global_env_path()).value == "AKEY"
    assert resolve_key("GEMINI_API_KEY", os.path.dirname(b),
                       global_env_path()).value == "BKEY"


# --- group 3: the carrier on MitosConfig ------------------------------------

def test_the_carrier_holds_only_declared_keys_that_resolved(tmp_path):
    """A stray `.env` variable is absent — the carrier does not hoover the file.

    P8: the map exists to answer mitos's own questions, and a user's ``.env`` is
    their file. An unresolved declared name is absent too (not present-and-empty),
    so a consumer's ``.get(name, default)`` behaves.
    """
    ws = tmp_path / "proj"
    _write(ws / ".env", "GEMINI_API_KEY=PROJKEY\nAWS_SECRET=nope\n")
    config = MitosConfig(str(ws))

    assert config.env["GEMINI_API_KEY"] == "PROJKEY"
    assert "AWS_SECRET" not in config.env
    assert "ANTHROPIC_API_KEY" not in config.env
    assert set(config.env) <= set(RESOLVED_ENV_KEYS)


def test_resolved_env_keys_stay_in_lockstep_with_the_model_registry():
    """A fifth model alias cannot land without a carrier slot.

    Derived from ``MODEL_IDS``, whose keys are already upper-case — building the
    set from ``MODEL_ALIASES`` instead silently drops ``EMBEDDING``, and that is
    the costliest one to lose: the embedding cache keys on content hash with no
    model id in it, so a mis-routed embedding override reads as working while
    cached prior-generation vectors flow into a new-generation collection.
    """
    overrides = {k for k in RESOLVED_ENV_KEYS if k.startswith("MITOS_MODEL_OVERRIDE_")}
    assert overrides == {f"MITOS_MODEL_OVERRIDE_{alias}" for alias in models.MODEL_IDS}
    assert "MITOS_MODEL_OVERRIDE_EMBEDDING" in overrides
    assert set(RESOLVED_ENV_KEYS) - overrides == {
        "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "QDRANT_URL"
    }


def test_the_carrier_is_absent_from_every_serialization_surface(tmp_path):
    """`config.env` holds real API keys and must reach no persisted surface.

    This row is NEW coverage, not an inherited pin: ``to_dict``'s existing test
    asserts each expected key *is in* the dict — a membership check, so it would
    stay green while ``to_dict()`` quietly grew a map of live credentials.
    ``inert_file_keys`` sits in the same unpinned position and is asserted here
    beside it.
    """
    ws = tmp_path / "proj"
    _write(ws / ".env", "GEMINI_API_KEY=PROJKEY\n")
    d = MitosConfig(str(ws)).to_dict()
    assert "env" not in d
    assert "inert_file_keys" not in d
    assert "PROJKEY" not in repr(d)


def test_qdrant_url_resolves_env_over_project_over_global_over_toml(tmp_path,
                                                                   monkeypatch):
    """The full precedence ladder, one construction per rung.

    The bottom two rungs are what this phase actually moves: before it, a
    ``QDRANT_URL`` in the *target* workspace's ``.env`` was invisible unless the
    process happened to have been launched from that directory.
    """
    ws = tmp_path / "proj"
    mitos_dir = ws / ".mitos"
    mitos_dir.mkdir(parents=True)

    # 5. nothing anywhere → the dedicated default port
    assert MitosConfig(str(ws)).qdrant_url == "http://localhost:7333"

    # 4. config.toml pins it
    (mitos_dir / "config.toml").write_text('qdrant_url = "http://toml:7333"\n',
                                           encoding="utf-8")
    assert MitosConfig(str(ws)).qdrant_url == "http://toml:7333"

    # 3. the global .env beats the toml pin
    _write(Path(global_env_path()),
           "QDRANT_URL=http://global:7333\n")
    assert MitosConfig(str(ws)).qdrant_url == "http://global:7333"

    # 2. the TARGET workspace's .env beats the global one
    _write(ws / ".env", "QDRANT_URL=http://proj:7333\n")
    assert MitosConfig(str(ws)).qdrant_url == "http://proj:7333"

    # 1. a real export beats everything
    monkeypatch.setenv("QDRANT_URL", "http://exported:7333")
    assert MitosConfig(str(ws)).qdrant_url == "http://exported:7333"


def test_an_exported_empty_qdrant_url_stays_empty(tmp_path, monkeypatch):
    """Byte-identical to the pre-carrier behaviour, and the second D2 argument.

    ``os.environ.get("QDRANT_URL", default)`` returned ``""`` for an
    exported-empty variable — not the default — and the post-load re-assert then
    skipped, leaving ``qdrant_url == ""``. Presence-at-tier-1 reproduces that
    exactly. A non-empty test at tier 1 (or ``.get(k) or default`` at either read)
    silently restores the default instead.
    """
    monkeypatch.setenv("QDRANT_URL", "")
    ws = tmp_path / "proj"
    _write(ws / ".env", "QDRANT_URL=http://proj:7333\n")
    assert MitosConfig(str(ws)).qdrant_url == ""


def test_the_carrier_is_populated_before_the_first_read_that_needs_it(tmp_path):
    """Ordering inside `__init__` is load-bearing; this is the row that bites.

    ``self.env`` must exist before the ``qdrant_url`` assignment reads it, and a
    move *past* the post-load re-assert reds this row: a workspace ``.env`` must
    beat a ``config.toml`` pin, and with an empty map at the first read the pin
    survives.

    Measured, because the plan predicted otherwise: moving the population to
    *between* the two reads is **not** a total non-result. The re-assert rescues
    every **truthy** value — so this row and the whole precedence ladder stay
    green — but it cannot rescue an exported **empty** ``QDRANT_URL``, because
    ``""`` is falsy and the re-assert skips; the first read then returns its
    default instead of ``""``. So the rescued-assertion caveat holds for the
    non-empty rungs only, and ``…_an_exported_empty_qdrant_url_stays_empty`` is
    the row that bites in that window.
    """
    ws = tmp_path / "proj"
    (ws / ".mitos").mkdir(parents=True)
    (ws / ".mitos" / "config.toml").write_text('qdrant_url = "http://toml:7333"\n',
                                               encoding="utf-8")
    _write(ws / ".env", "QDRANT_URL=http://proj:7333\n")
    assert MitosConfig(str(ws)).qdrant_url == "http://proj:7333"


def _workspace_with_url(tmp_path, url: str = "http://from-target:7333") -> str:
    """A minimal workspace whose own `.env` carries a QDRANT_URL, nothing exported.

    The four W20 rows below read ``store.base_url`` — ``QdrantVectorStore``
    stores its endpoint under that name (trailing slash stripped), not under the
    constructor's ``qdrant_url`` parameter name.
    """
    ws = tmp_path / "proj"
    (ws / ".mitos").mkdir(parents=True)
    _write(ws / ".env", f"QDRANT_URL={url}\n")
    GraphStore(str(ws / ".mitos" / "graph.sqlite"))  # materialize the graph file
    return str(ws)


def test_the_sync_manager_builds_its_store_on_the_targets_url(tmp_path, monkeypatch):
    """W20 at `sync.py`'s construction site — the value arrives, unedited.

    All four ``QdrantVectorStore(`` sites already pass ``config.qdrant_url``
    positionally, so 2c has no per-site edit to make: the URL arrives the moment
    ``MitosConfig`` resolves it. A dummy ``GEMINI_API_KEY`` is required —
    ``GeminiEmbeddingProvider`` raises without one and this site swallows that
    into ``vector_store = None``, so the row would fail on the wrong thing.
    ``genai.Client`` does no network I/O at construction.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    from mitos.sync import MitosSyncManager

    manager = MitosSyncManager(MitosConfig(_workspace_with_url(tmp_path)))
    assert manager.vector_store is not None
    assert manager.vector_store.base_url == "http://from-target:7333"


def test_the_importer_builds_its_store_on_the_targets_url(tmp_path, monkeypatch):
    """W20 at `importer.py`'s construction site."""
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    from mitos.importer import MitosProseImporter

    importer = MitosProseImporter(MitosConfig(_workspace_with_url(tmp_path)))
    assert importer.vector_store is not None
    assert importer.vector_store.base_url == "http://from-target:7333"


def test_the_check_substrate_builds_its_store_on_the_targets_url(tmp_path):
    """W20 at `cli.py`'s construction site — the one needing no key.

    1b moved the store construction *outside* the ``try``, so it is returned
    unconditionally and an absent ``GEMINI_API_KEY`` degrades only the embedding
    provider.
    """
    _, vector, _ = cli._build_check_substrate(
        MitosConfig(_workspace_with_url(tmp_path))
    )
    assert vector.base_url == "http://from-target:7333"


def test_the_mcp_server_builds_its_store_on_the_targets_url(tmp_path, monkeypatch):
    """W20 at `mcp_server.py`'s construction site.

    It takes the workspace config as an argument (phase 3c), so the cwd read that
    used to live inside it now lives at the call site — which is what leaves this
    row proving what it always proved: the URL of the workspace *given* is the URL
    the vector store is built on. Phase 5d removed the constructor's ``"."``
    default, so the target is named outright and the ``chdir`` that used to supply
    it is gone; a callee reading the working directory instead of the config would
    resolve pytest's cwd (the repo — a real workspace) and this assertion would
    still red.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    from mitos import mcp_server

    ws = _workspace_with_url(tmp_path)
    _, _, vector_store = mcp_server.get_workspace_components(MitosConfig(ws))
    assert vector_store is not None
    assert vector_store.base_url == "http://from-target:7333"


# --- group 4: tier discipline ----------------------------------------------

def test_importing_the_env_leaf_pulls_in_no_other_mitos_module():
    """`mitos.env` imports NOTHING from `mitos` — the strongest tier claim here.

    It is what lets 5c call the resolver from ``init`` and ``set-key`` without
    dragging anything, and it is what keeps ``config`` → ``env`` acyclic while
    ``config`` owns the global-path derivation. The exact closure, not a
    blacklist: the rule this module states is "nothing else, ever".
    """
    probe = (
        "import sys; import mitos.env; "
        "print(','.join(sorted(m for m in sys.modules if m.startswith('mitos'))))"
    )
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip().split(",") == ["mitos", "mitos.env"]


# --- group 5: the fixture's own net, and the last hand-rolled `.env` writer --

def test_a_raw_environ_write_does_not_survive_the_key_strip():
    """The fixture's own net: this module writes `os.environ` raw, and it must not leak.

    Verified to red against the plain ``monkeypatch.delenv(name, raising=False)``
    this replaced — on an already-absent name that records nothing, so the raw
    write below survives teardown and every module collected after this one sees
    a ``GEMINI_API_KEY``. The module docstring names that class; this is the row
    that keeps this module out of it.
    """
    with pytest.MonkeyPatch.context() as mp:
        _unset(mp, "MITOS_TEST_RAW_WRITE")
        os.environ["MITOS_TEST_RAW_WRITE"] = "leaked"
    assert "MITOS_TEST_RAW_WRITE" not in os.environ


def test_the_key_strip_still_restores_a_genuinely_exported_value(monkeypatch):
    """The other direction: stripping a real export is a loan, not a theft.

    A shell that really did export the name gets it back — including over a raw
    write made while it was stripped, because the records unwind in reverse.
    """
    monkeypatch.setenv("MITOS_TEST_EXPORTED", "REAL")
    with pytest.MonkeyPatch.context() as mp:
        _unset(mp, "MITOS_TEST_EXPORTED")
        assert "MITOS_TEST_EXPORTED" not in os.environ
        os.environ["MITOS_TEST_EXPORTED"] = "leaked"
    assert os.environ["MITOS_TEST_EXPORTED"] == "REAL"

def test_the_writer_and_the_reader_agree_on_a_hand_spaced_key(tmp_path):
    """D6/entry-004a: `set-key` must REPLACE `GEMINI_API_KEY = old`, not shadow it.

    The row this replaces pinned the entry load and the resolver reading one
    parse; the entry load is gone, and the surviving half of its subject is the
    clause it named in passing — *"the tree's third parse still gets this
    wrong"*. That third parse is ``cli._upsert_env_var``, the last hand-rolled
    ``.env`` handler and the only one that **writes**. Until 5c it matched
    ``line.strip().startswith(f"{name}=")``, so a hand-spaced assignment was
    invisible to it and ``set-key`` appended a *second* line — leaving a file
    whose writer and reader disagreed about which line was the key, with
    ``parse_env_file``'s first-wins rule handing the reader the stale one.

    Asserted end to end rather than on the writer alone: the write lands on the
    existing line (no duplicate), and the resolver then reads the **new** value.
    """
    target = tmp_path / "proj"
    path = _write(target / ".env",
                  "# scaffold\n  GEMINI_API_KEY = old  \nOTHER=keep\n")

    cli._upsert_env_var(path, "GEMINI_API_KEY", "new")

    lines = open(path, encoding="utf-8").read().splitlines()
    assert [ln for ln in lines if "GEMINI_API_KEY" in ln] == ["GEMINI_API_KEY=new"]
    assert "# scaffold" in lines and "OTHER=keep" in lines
    assert parse_env_file(path) == {"GEMINI_API_KEY": "new", "OTHER": "keep"}
    assert resolve_key("GEMINI_API_KEY", str(target),
                       global_env_path()).value == "new"


def test_the_writer_leaves_a_line_the_reader_does_not_see_as_an_assignment(tmp_path):
    """The other half of the agreement, and the edge the fix introduces.

    ``line.split("=", 1)[0]`` on a line carrying no ``=`` returns the whole line,
    so a bare ``GEMINI_API_KEY`` would match the name and be rewritten — while
    ``parse_env_file`` skips it. The writer therefore tests for an ``=`` first,
    which is the reader's own predicate: a non-assignment is not the key line for
    either of them, and a commented-out assignment parses its key as ``#NAME``
    and matches neither.
    """
    path = _write(tmp_path / ".env",
                  "GEMINI_API_KEY\n#GEMINI_API_KEY=commented\n")

    cli._upsert_env_var(path, "GEMINI_API_KEY", "new")

    lines = open(path, encoding="utf-8").read().splitlines()
    assert lines == ["GEMINI_API_KEY", "#GEMINI_API_KEY=commented",
                     "GEMINI_API_KEY=new"]
    assert parse_env_file(path) == {"GEMINI_API_KEY": "new"}


def test_the_env_module_writes_to_the_process_environment_nowhere(tmp_path):
    """A source sweep, because the invariant is about the module, not one call.

    The behavioural row above proves one resolution leaves ``os.environ``
    untouched; this forbids the *write* growing back anywhere in the leaf while
    every behavioural row stays green. Swept over the parsed code rather than the
    raw text — the module's prose names ``os.environ`` repeatedly, since it is the
    thing this design stops writing to.
    """
    tree = ast.parse(open(env.__file__, encoding="utf-8").read())
    writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete))
        for target in (node.targets if isinstance(node, (ast.Assign, ast.Delete))
                       else [node.target])
        if isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == "environ"
    ]
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("setdefault", "update", "pop", "clear")
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "environ"
    ]
    assert writes == [], "the env leaf writes to os.environ"
    assert calls == [], f"the env leaf mutates os.environ: {calls}"


def test_the_resolver_is_callable_without_a_config_object(tmp_path):
    """A broken `config.toml` must not brick the answer to "where is my key?".

    ``MitosConfig.__init__`` raises ``ConfigError`` on malformed TOML — *after*
    the carrier is built but before any caller can read it — so resolution
    reachable only through a config object would be unavailable for exactly the
    verbs (``init``, ``set-key``, ``status``) a user reaches for while a workspace
    is half-set-up. The leaf takes a directory and a path, so it answers anyway.
    """
    from mitos.errors import ConfigError

    ws = tmp_path / "proj"
    (ws / ".mitos").mkdir(parents=True)
    (ws / ".mitos" / "config.toml").write_text("this is not = = toml\n",
                                               encoding="utf-8")
    _write(ws / ".env", "GEMINI_API_KEY=PROJKEY\n")

    with pytest.raises(ConfigError):
        MitosConfig(str(ws))
    assert resolve_key("GEMINI_API_KEY", str(ws), global_env_path()) == ResolvedValue(
        "PROJKEY", TIER_PROJECT_ENV
    )


def test_a_second_workspace_is_unaffected_by_the_first_construction(tmp_path):
    """No cross-call cache: two configs in one process carry their own answers.

    I6 forbids caching resolution across calls, and ``mcp_server`` builds a fresh
    ``MitosConfig`` per tool call by design — a module-level cache would make the
    second call inherit the first project's URL.
    """
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write(a / ".env", "QDRANT_URL=http://a:7333\n")
    _write(b / ".env", "QDRANT_URL=http://b:7333\n")
    assert MitosConfig(str(a)).qdrant_url == "http://a:7333"
    assert MitosConfig(str(b)).qdrant_url == "http://b:7333"
    assert MitosConfig(str(a)).qdrant_url == "http://a:7333"


def test_a_workspace_built_from_a_bare_tempdir_resolves_nothing(tmp_path):
    """The overwhelmingly common shape: no `.env` anywhere, and it is healthy.

    Empty/fresh is first-class in this tree — a workspace with no credentials
    file must construct cleanly and simply carry an empty map.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MitosConfig(tmpdir)
        assert config.env == {}
        assert config.qdrant_url == "http://localhost:7333"
