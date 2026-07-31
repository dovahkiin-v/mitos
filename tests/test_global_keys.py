"""Tests for global API-key resolution and `mitos set-key` (config.py + cli.py)."""

import os

import pytest

from mitos import cli
from mitos import config as mitos_config
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
