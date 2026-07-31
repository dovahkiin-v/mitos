"""Tests for global API-key resolution and `mitos set-key` (config.py + cli.py)."""

import os

import pytest
from unittest.mock import MagicMock

from mitos import cli, embeddings
from mitos import config as mitos_config
from mitos.config import MitosConfig
from mitos.env import (
    TIER_GLOBAL_ENV,
    TIER_PROJECT_ENV,
    ResolvedValue,
    parse_env_file,
    resolve_key,
)
from mitos.errors import MitosError


def test_global_env_path_honors_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert mitos_config.global_env_path() == str(tmp_path / "mitos" / ".env")


def test_set_key_global_writes_xdg_env_mode_600(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cli.cmd_set_key("ABC123", workspace_dir=None, is_global=True)
    gpath = mitos_config.global_env_path()
    assert os.path.exists(gpath)
    assert "GEMINI_API_KEY=ABC123" in open(gpath, encoding="utf-8").read()
    assert oct(os.stat(gpath).st_mode)[-3:] == "600"


def test_set_key_custom_name(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cli.cmd_set_key("SECRET", workspace_dir=None, name="ANTHROPIC_API_KEY", is_global=True)
    assert "ANTHROPIC_API_KEY=SECRET" in open(mitos_config.global_env_path(), encoding="utf-8").read()


def test_upsert_replaces_a_hand_spaced_slot_no_duplicate(tmp_path):
    """D6/entry-004a: the shape the pre-5c `startswith(f"{name}=")` match could not see.

    Extends the shipped row below rather than replacing it. A hand-edited
    ``GEMINI_API_KEY = old`` was invisible to the writer, so ``set-key`` appended a
    *second* assignment and ``env.parse_env_file``'s first-non-empty-wins rule then
    handed the reader the stale one — writer and reader disagreeing about which
    line is the key, in a file holding a credential.
    """
    env = tmp_path / ".env"
    env.write_text("# header\n  GEMINI_API_KEY = old  \nOTHER=keep\n")
    cli._upsert_env_var(str(env), "GEMINI_API_KEY", "NEWKEY")
    content = env.read_text()
    assert content.count("GEMINI_API_KEY") == 1
    assert "GEMINI_API_KEY=NEWKEY" in content
    assert "# header" in content and "OTHER=keep" in content


def test_upsert_replaces_empty_slot_no_duplicate(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# header\nGEMINI_API_KEY=\nOTHER=keep\n")
    cli._upsert_env_var(str(env), "GEMINI_API_KEY", "NEWKEY")
    content = env.read_text()
    assert content.count("GEMINI_API_KEY=") == 1
    assert "GEMINI_API_KEY=NEWKEY" in content
    assert "OTHER=keep" in content  # other lines preserved


# The empty-slot row that lived here drove `cli`'s own second hand-rolled `.env`
# parse, which phase 2c retired: `_gemini_key_source` now reads `env.resolve_key`,
# the tree's one layering implementation. That behaviour (an empty scaffolded slot
# skipped, a real value on a later line still found) is pinned on
# `env.parse_env_file` in `tests/test_env_resolution.py`; retargeting the row here
# would duplicate a shipped one.


def test_key_source_global_then_project_override(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cli.cmd_set_key("GLOBALKEY", workspace_dir=None, is_global=True)
    proj = tmp_path / "proj"
    proj.mkdir()
    assert cli._gemini_key_source(str(proj)) == "global .env"
    (proj / ".env").write_text("GEMINI_API_KEY=PROJKEY\n")
    assert cli._gemini_key_source(str(proj)) == "project .env"


def test_key_source_environment_only(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg-empty"))
    monkeypatch.setenv("GEMINI_API_KEY", "ENVKEY")
    proj = tmp_path / "p"
    proj.mkdir()
    assert cli._gemini_key_source(str(proj)) == "environment"


def test_key_source_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg-empty2"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    proj = tmp_path / "p2"
    proj.mkdir()
    assert cli._gemini_key_source(str(proj)) is None


def test_the_project_form_is_unconstructible_without_a_workspace(tmp_path, monkeypatch):
    """5a: `workspace_dir` is required, and `None` means "no project", never "use cwd".

    The boundary refuses a selectorless `set-key` before the handler is reached, so
    this row is what keeps the handler's own guard non-vacuous — it is the direct
    call, the shape a future caller could make. Fault injection: restore the
    `workspace_dir=None` default and its `os.getcwd()` fallback, and this is the only
    row that reds; every boundary-driven row stays green.
    """
    monkeypatch.chdir(tmp_path)

    with pytest.raises(MitosError, match="--project"):
        cli.cmd_set_key("SECRET", workspace_dir=None)

    assert not os.path.exists(os.path.join(str(tmp_path), ".env"))
    assert not os.path.exists(mitos_config.global_env_path())


def test_a_key_dropped_into_the_scaffolded_env_still_just_works(tmp_path, monkeypatch):
    """5c criterion 4: only the DELIVERY changed — auto-discovery is intact.

    The ADR ``init-scaffolds-gitignored-env`` promises that a key dropped into the
    ``.env`` ``mitos init`` scaffolds is found with no further configuration. 5c
    deleted the mechanism that used to *deliver* it (the entry-time load into
    ``os.environ``), so the promise has to be re-asserted against the replacement
    rather than assumed to have survived it — the scaffolded file is read by
    ``env.resolve_values`` for the workspace the call named, at the same tier, in
    the same precedence order.

    Driven end to end through the real verbs: `init` writes the scaffold with its
    empty slot, `set-key` fills it, the resolver reads it back, and a consumer
    (`cmd_capture`) is handed it. The global half is the setup SETUP.md actually
    recommends — one key for the machine, an untouched empty slot per project —
    and it is the half a surviving *global* load would have made resolve from the
    wrong tier.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("GEMINI_API_KEY", "")   # force the absence into the record
    monkeypatch.delenv("GEMINI_API_KEY")
    ws = tmp_path / "proj"
    ws.mkdir()
    cli.cmd_init(MitosConfig(str(ws)))

    scaffold = (ws / ".env").read_text(encoding="utf-8")
    assert "GEMINI_API_KEY=" in scaffold
    assert parse_env_file(str(ws / ".env")) == {}   # the slot is empty, not a value

    # 1. A global key, project slot untouched → the global tier answers.
    cli.cmd_set_key("GLOBALKEY", workspace_dir=None, is_global=True)
    assert resolve_key("GEMINI_API_KEY", str(ws),
                       mitos_config.global_env_path()) == ResolvedValue(
        "GLOBALKEY", TIER_GLOBAL_ENV)

    # 2. The project's own slot filled by `set-key` → it wins, in place.
    cli.cmd_set_key("PROJKEY", workspace_dir=str(ws))
    assert (ws / ".env").read_text(encoding="utf-8").count("GEMINI_API_KEY") == 1
    assert resolve_key("GEMINI_API_KEY", str(ws),
                       mitos_config.global_env_path()) == ResolvedValue(
        "PROJKEY", TIER_PROJECT_ENV)

    # 3. And a consumer is handed it — the whole point of the delivery change.
    seen = []

    def _client(*, api_key=None):
        seen.append(api_key)
        return MagicMock()

    monkeypatch.setattr(embeddings.genai, "Client", _client)
    config = MitosConfig(str(ws))
    assert config.env["GEMINI_API_KEY"] == "PROJKEY"
    cli.cmd_capture(config, "We will use python.")
    # A set, not a list: `capture` builds a client for synthesis and another on
    # the way to the buffer, and the claim is about *which key* every one of them
    # was handed, not how many there are.
    assert seen and set(seen) == {"PROJKEY"}
