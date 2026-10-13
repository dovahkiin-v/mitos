"""Pins that importing the CLI, and the commit gate's closure, loads no LLM SDK.

`google.genai` and `anthropic` together cost ~1s of import (measured 2026-09-12:
`import mitos.cli` went from 1.6s to well under half that once they moved into the
functions that call them). Every `mitos` verb pays the CLI's import, including the
ones that never reach a model, so a module-scope SDK import anywhere in the CLI's
closure is a regression on every invocation. A subprocess, because this test
process has long since imported both.

The commit gate's modules (`commit_gate`, the `audit_debt` leaf it reads, and
`config`, the key test's home) run on every commit, so they are pinned by name as
well as through `mitos.cli`: the gate must stay importable without an SDK by the
status row and the record notice, which will not come through the CLI.
"""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["mitos.cli", "mitos.mcp_server",
                                    "mitos.commit_gate", "mitos.audit_debt", "mitos.config"])
def test_importing_the_entry_point_loads_no_llm_sdk(module: str) -> None:
    probe = (
        f"import sys; import {module}; "
        "print(sorted(m for m in ('anthropic', 'google.genai') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", f"{module} imported: {out.stdout.strip()}"
