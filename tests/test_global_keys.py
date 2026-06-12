"""Tests for global API-key resolution and `mitos set-key` (config.py + cli.py)."""

import os

from mitos import cli
from mitos import config as mitos_config


def test_global_env_path_honors_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert mitos_config.global_env_path() == str(tmp_path / "mitos" / ".env")


def test_set_key_global_writes_xdg_env_mode_600(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cli.cmd_set_key("ABC123", is_global=True)
    gpath = mitos_config.global_env_path()
    assert os.path.exists(gpath)
    assert "GEMINI_API_KEY=ABC123" in open(gpath, encoding="utf-8").read()
    assert oct(os.stat(gpath).st_mode)[-3:] == "600"


def test_set_key_custom_name(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cli.cmd_set_key("SECRET", name="ANTHROPIC_API_KEY", is_global=True)
    assert "ANTHROPIC_API_KEY=SECRET" in open(mitos_config.global_env_path(), encoding="utf-8").read()


def test_upsert_replaces_empty_slot_no_duplicate(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# header\nGEMINI_API_KEY=\nOTHER=keep\n")
    cli._upsert_env_var(str(env), "GEMINI_API_KEY", "NEWKEY")
    content = env.read_text()
    assert content.count("GEMINI_API_KEY=") == 1
    assert "GEMINI_API_KEY=NEWKEY" in content
    assert "OTHER=keep" in content  # other lines preserved


def test_env_file_has_key_skips_empty_slot(tmp_path):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=\n")
    assert cli._env_file_has_key(str(env), "GEMINI_API_KEY") is False
    # a real value on a LATER line is still found (past the empty scaffolded slot)
    env.write_text("GEMINI_API_KEY=\nGEMINI_API_KEY=real\n")
    assert cli._env_file_has_key(str(env), "GEMINI_API_KEY") is True


def test_key_source_global_then_project_override(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cli.cmd_set_key("GLOBALKEY", is_global=True)
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
