"""Pins that importing the CLI loads no LLM SDK.

`google.genai` and `anthropic` together cost ~1s of import (measured 2026-09-12:
`import mitos.cli` went from 1.6s to well under half that once they moved into the
functions that call them). Every `mitos` verb pays the CLI's import, including the
ones that never reach a model, so a module-scope SDK import anywhere in the CLI's
closure is a regression on every invocation. A subprocess, because this test
process has long since imported both.
"""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["mitos.cli", "mitos.mcp_server"])
def test_importing_the_entry_point_loads_no_llm_sdk(module: str) -> None:
    probe = (
        f"import sys; import {module}; "
        "print(sorted(m for m in ('anthropic', 'google.genai') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", f"{module} imported: {out.stdout.strip()}"
