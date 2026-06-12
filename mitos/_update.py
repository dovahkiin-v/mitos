"""Best-effort 'a new version is available' check for the pipx/git install.

Mitos ships from a git repo via pipx, so the only "is there something newer?"
signal is the ``__version__`` on ``main``. This module fetches that at most once
per day (cached under ``~/.cache/mitos/``), compares it to the running version,
and returns a one-line notice the CLI prints to stderr. Everything here is
best-effort and fail-silent: no network, a timeout, or a parse error simply
yields no notice — it must never disrupt a real command.

Opt out entirely with ``MITOS_NO_UPDATE_CHECK=1``.
"""

import json
import os
import re
import time
from typing import Optional

_CACHE_TTL_SECONDS = 24 * 60 * 60
_REMOTE_VERSION_URL = (
    "https://raw.githubusercontent.com/dovahkiin-v/mitos/main/mitos/__init__.py"
)
_UPDATE_COMMAND = "pipx install --force git+https://github.com/dovahkiin-v/mitos"
_HTTP_TIMEOUT_SECONDS = 2.0


def _cache_path() -> str:
    """Returns the update-check cache file path (honors ``XDG_CACHE_HOME``)."""
    cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(cache_home, "mitos", "update_check.json")


def _parse_version(text: str) -> Optional[str]:
    """Extracts a ``__version__ = "x.y.z"`` value from module source text."""
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    return match.group(1) if match else None


def _version_tuple(version: str) -> tuple:
    """Parses a dotted version into an int tuple for ordering (non-ints → 0)."""
    parts = []
    for piece in version.strip().split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _read_cache() -> Optional[dict]:
    """Returns the cached check payload, or None if absent/unreadable."""
    try:
        with open(_cache_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_cache(latest: Optional[str], now: float) -> None:
    """Persists the last-checked timestamp and latest seen version."""
    try:
        path = _cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"checked_at": now, "latest": latest}, f)
    except OSError:
        pass


def _fetch_remote_version() -> Optional[str]:
    """Fetches ``__version__`` from ``main`` on GitHub (fail-silent)."""
    try:
        import requests

        resp = requests.get(_REMOTE_VERSION_URL, timeout=_HTTP_TIMEOUT_SECONDS)
        if resp.status_code == 200:
            return _parse_version(resp.text)
    except Exception:
        pass
    return None


def _latest_version(now: float) -> Optional[str]:
    """Returns the latest known remote version, hitting the network ≤ once/day.

    Within the cache TTL the cached value is returned with no network call. When
    stale, one fetch is attempted; on failure the previous cached value (if any)
    is reused rather than thrashing the network on every invocation.
    """
    cache = _read_cache()
    if cache and (now - cache.get("checked_at", 0)) < _CACHE_TTL_SECONDS:
        return cache.get("latest")
    latest = _fetch_remote_version()
    if latest:
        _write_cache(latest, now)
        return latest
    return cache.get("latest") if cache else None


def update_notice(current_version: str, now: Optional[float] = None) -> Optional[str]:
    """Returns a one-line 'update available' notice, or None.

    Args:
        current_version: The running ``__version__``.
        now: Override for the current epoch seconds (testing); defaults to
            ``time.time()``.

    Returns:
        A short stderr-ready notice when a newer version exists, else None.
    """
    if os.environ.get("MITOS_NO_UPDATE_CHECK"):
        return None
    now = time.time() if now is None else now
    latest = _latest_version(now)
    if not latest:
        return None
    try:
        if _version_tuple(latest) > _version_tuple(current_version):
            return (
                f"📦 A new version of mitos is available ({current_version} → {latest}).\n"
                f"   Update: {_UPDATE_COMMAND}"
            )
    except Exception:
        pass
    return None
