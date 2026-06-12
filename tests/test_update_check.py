"""Tests for the once-a-day 'new version available' check (mitos/_update.py)."""

import pytest

from mitos import _update


@pytest.fixture
def enable_update_check(monkeypatch):
    """Undoes the autouse opt-out so the check actually runs."""
    monkeypatch.delenv("MITOS_NO_UPDATE_CHECK", raising=False)


def test_version_tuple_ordering():
    vt = _update._version_tuple
    assert vt("0.1.2") > vt("0.1.1")
    assert vt("0.2.0") > vt("0.1.9")
    assert vt("1.0.0") > vt("0.9.9")
    assert vt("0.1.1") == vt("0.1.1")


def test_parse_version():
    assert _update._parse_version('__version__ = "0.1.3"') == "0.1.3"
    assert _update._parse_version("__version__ = '1.2.3'") == "1.2.3"
    assert _update._parse_version("no version anywhere") is None


def test_notice_when_newer(enable_update_check, monkeypatch):
    monkeypatch.setattr(_update, "_fetch_remote_version", lambda: "9.9.9")
    notice = _update.update_notice("0.1.3")
    assert notice is not None
    assert "0.1.3" in notice and "9.9.9" in notice
    assert "pipx install --force" in notice


def test_no_notice_when_current(enable_update_check, monkeypatch):
    monkeypatch.setattr(_update, "_fetch_remote_version", lambda: "0.1.3")
    assert _update.update_notice("0.1.3") is None


def test_no_notice_when_local_is_ahead(enable_update_check, monkeypatch):
    monkeypatch.setattr(_update, "_fetch_remote_version", lambda: "0.1.0")
    assert _update.update_notice("0.2.0") is None


def test_opt_out_suppresses_notice(monkeypatch):
    monkeypatch.setenv("MITOS_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(_update, "_fetch_remote_version", lambda: "9.9.9")
    assert _update.update_notice("0.1.3") is None


def test_network_failure_is_silent(enable_update_check, monkeypatch):
    monkeypatch.setattr(_update, "_fetch_remote_version", lambda: None)
    assert _update.update_notice("0.1.3") is None


def test_cache_avoids_refetch_within_ttl(enable_update_check, monkeypatch):
    calls = {"n": 0}

    def fake_fetch():
        calls["n"] += 1
        return "9.9.9"

    monkeypatch.setattr(_update, "_fetch_remote_version", fake_fetch)
    assert _update.update_notice("0.1.3") is not None  # fetch #1, caches
    assert _update.update_notice("0.1.3") is not None  # served from cache
    assert calls["n"] == 1
