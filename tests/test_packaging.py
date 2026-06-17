"""Packaging integration tests — proves the wheel ships what a real install needs.

These are deliberately heavy: they build a real wheel and install it into a
*fresh, throwaway* virtualenv (non-editable). That is the whole point — an
editable install reads ``format-spec.md`` from the source tree and would hide a
missing-``package-data`` bug. ``mitos/format-spec.md`` is read from the installed
package dir in two places (``mitos.cli.load_format_spec`` at ``mitos init`` time
and ``mitos.parser`` at *import* time), so a wheel missing it doesn't just break
``init`` — it breaks ``import mitos.parser`` outright. Only a non-editable install
exercises that real read path (vision V1-D7 / §6.2).

Both tests carry the ``packaging`` marker: they are slow (venv + network build
isolation pulls setuptools/wheel) and testmon will not select them on its own, so
CI (Phase 1b) runs them explicitly with ``-m packaging`` while the fast suite
skips them with ``-m 'not packaging'``.

Offline-tolerant: build isolation needs PyPI for setuptools/wheel and the wheel
install needs it for the five runtime deps. When PyPI is unreachable the tests
skip rather than fail, so offline dev stays clean. CI has network and always runs.
"""

import os
import socket
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import mitos as _mitos

# Single source of truth, read at test time — never hardcode "0.1.13" (the release
# ritual bumps it on every shipped change; a literal here would rot silently).
EXPECTED_VERSION = _mitos.__version__

# Project root = the source tree pip builds from (tests/ -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Reuse the host's real pip cache so repeat runs (and warm-cache offline runs) are
# fast and don't re-download setuptools/wheel/deps every time. The hermetic
# autouse fixture points XDG_CACHE_HOME at a tmp dir; PIP_CACHE_DIR overrides that
# for pip specifically without touching the update-check hermeticity.
HOST_PIP_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "pip")

pytestmark = pytest.mark.packaging


def _pypi_reachable() -> bool:
    """Returns True if pypi.org:443 accepts a TCP connection within a short timeout."""
    try:
        with socket.create_connection(("pypi.org", 443), timeout=3):
            return True
    except OSError:
        return False


def _pip_env() -> dict:
    """Builds a subprocess env for pip with a warm, host-backed cache."""
    return {**os.environ, "PIP_CACHE_DIR": HOST_PIP_CACHE}


def _mitos_env(tmp_path) -> dict:
    """Builds a hermetic env for invoking the installed `mitos` binary.

    The autouse ``hermetic_mitos_env`` fixture only patches the *parent* process;
    a subprocess that the install test spawns must carry these keys explicitly so
    it never hits the network update path or pollutes the real ``~/.config/mitos``.
    """
    return {
        **os.environ,
        "MITOS_NO_UPDATE_CHECK": "1",
        "MITOS_NO_MCP_HINT": "1",
        "XDG_CONFIG_HOME": str(tmp_path / "xdg_config"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg_cache"),
    }


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory) -> Path:
    """Builds the project wheel once (non-editable), shared by the packaging tests.

    Skips the whole packaging suite when PyPI is unreachable — build isolation
    needs it for setuptools/wheel. A build failure while *online* is a real
    failure (asserted), not a skip.
    """
    if not _pypi_reachable():
        pytest.skip("PyPI unreachable — build isolation needs setuptools/wheel from PyPI")

    wheelhouse = tmp_path_factory.mktemp("wheelhouse")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps",
         str(PROJECT_ROOT), "-w", str(wheelhouse)],
        env=_pip_env(), capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"wheel build failed:\n{result.stdout}\n{result.stderr}"

    wheels = list(wheelhouse.glob("mitos_adr-*.whl"))
    assert len(wheels) == 1, f"expected exactly one mitos_adr wheel, got {wheels}"
    return wheels[0]


def test_format_spec_ships_in_wheel(built_wheel):
    """The wheel carries ``mitos/format-spec.md`` as package data (fast namelist check).

    The cheap complement to the full install test: a missing-``package-data``
    regression shows up here in milliseconds, before paying for a venv install.
    """
    with zipfile.ZipFile(built_wheel) as zf:
        names = zf.namelist()
    assert "mitos/format-spec.md" in names, (
        f"format-spec.md missing from the wheel — package-data is broken. Wheel contents: {names}"
    )


def test_non_editable_install_real_read_path(built_wheel, tmp_path):
    """A fresh venv + non-editable install exercises the real package-dir read path.

    Proves the load-bearing chain: the installed wheel reports the single-source
    version (no static drift), ``import mitos.parser`` succeeds (its import-time
    ``FIELD_MAP`` reads the bundled spec), the installed ``format-spec.md`` matches
    the source byte-for-byte, and ``mitos init`` reads the spec from the *installed*
    package dir.
    """
    venv_dir = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)],
                   check=True, capture_output=True, text=True, timeout=120)
    venv_python = venv_dir / "bin" / "python"
    venv_mitos = venv_dir / "bin" / "mitos"

    # Non-editable install of the built wheel (deps resolved from PyPI).
    install = subprocess.run(
        [str(venv_python), "-m", "pip", "install", str(built_wheel)],
        env=_pip_env(), capture_output=True, text=True, timeout=600,
    )
    assert install.returncode == 0, f"wheel install failed:\n{install.stdout}\n{install.stderr}"
    assert venv_mitos.exists(), "console script `mitos` was not installed"

    # `mitos --version` reports the single-source version (no static pyproject drift).
    version = subprocess.run([str(venv_mitos), "--version"], env=_mitos_env(tmp_path),
                             capture_output=True, text=True, timeout=60)
    assert version.returncode == 0, version.stderr
    assert version.stdout.strip() == f"mitos {EXPECTED_VERSION}", version.stdout

    # `import mitos.parser` runs FIELD_MAP = load_dynamic_field_map() at import time,
    # which reads the bundled spec — a wheel missing it fails here outright.
    parser_import = subprocess.run([str(venv_python), "-c", "import mitos.parser"],
                                   env=_mitos_env(tmp_path), capture_output=True, text=True, timeout=60)
    assert parser_import.returncode == 0, (
        f"`import mitos.parser` failed in the installed venv — "
        f"format-spec.md likely missing from the wheel:\n{parser_import.stderr}"
    )

    # The spec ships in the installed package dir and matches the source exactly.
    installed_spec = subprocess.run(
        [str(venv_python), "-c",
         "import os, mitos; print(os.path.join(os.path.dirname(mitos.__file__), 'format-spec.md'))"],
        env=_mitos_env(tmp_path), capture_output=True, text=True, timeout=60,
    )
    assert installed_spec.returncode == 0, installed_spec.stderr
    installed_spec_path = Path(installed_spec.stdout.strip())
    assert installed_spec_path.is_file(), f"installed spec not found at {installed_spec_path}"
    assert installed_spec_path.read_bytes() == (PROJECT_ROOT / "mitos" / "format-spec.md").read_bytes(), (
        "installed format-spec.md differs from the package source"
    )

    # `mitos init` reads the spec from the installed package dir and scaffolds a workspace.
    workspace = tmp_path / "proj"
    workspace.mkdir()
    init = subprocess.run([str(venv_mitos), "init"], cwd=workspace, env=_mitos_env(tmp_path),
                          capture_output=True, text=True, timeout=120)
    assert init.returncode == 0, f"`mitos init` failed:\n{init.stdout}\n{init.stderr}"
    assert (workspace / "format-spec.md").is_file(), "`mitos init` did not seed format-spec.md"
