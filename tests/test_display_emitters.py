"""Phase 1b — every CLI + MCP display-JSON emitter routes through ``dumps_display``.

Phase 1a shipped the serializer (``mitos/display.py``) and pinned its unit-level
contracts (``tests/test_display_encoding.py``). Phase 1b flips the 14 CLI +
8 MCP display ``json.dumps`` sites onto it, so ``ensure_ascii=False`` lands once
and CLI⇄MCP drift becomes structurally impossible. These tests close the
emission e2e (WIRING_LEDGER entry-001) that 1a deferred:

* **T1/W1** — a glyph-bearing payload emitted by a real CLI ``--json`` verb and
  its MCP twin contains the **raw glyphs** (``—``/``§``/Lithuanian), not
  ``\\uXXXX`` noise, and ``json.loads`` round-trips it.
* **T3** — the CLI verb and its MCP twin treat the same glyphs **identically**
  (both unescaped on a UTF-8 stdout).
* **R6 through the emitter** — on a non-UTF-8 stdout the CLI ``--json`` path
  falls back to pure-ASCII ``\\uXXXX`` that still round-trips (never a
  ``UnicodeEncodeError``).
* **Zero text-path change** — a verb's no-flag text output is untouched by the
  JSON flip.
* **§3-(1) nudge** — the 💡 "wire the MCP" nudge lands on its own **stderr** line
  and is **absent** from stdout (pinning the already-satisfied state so a future
  regression that re-concatenates it onto stdout is caught).

Driven fully offline (unreachable Qdrant + no keys) so they exercise the pure
graph read/write path and never touch the machine's running services. Assertions
are via ``json.loads`` round-trip and glyph presence, never hardcoded escape bytes.
"""

import io
import json
import shutil
import sys
import tempfile
from typing import Iterator, Tuple

import pytest
from unittest.mock import patch

from mitos.config import MitosConfig
from mitos.cli import cmd_init, cmd_list, main
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager

# §-dense, em-dash- and Lithuanian-bearing text — the exact glyphs that today get
# escaped into \uXXXX noise. The slug stays ASCII (slugs are casefold handles);
# the glyphs live in the axiom + rejected_paths, which the emitters carry verbatim.
GLYPH_AXIOM = "Naudoti SQLite — ne PostgreSQL (§4 ąčęėįšųūž)"
GLYPH_REJECTED = "Atmesta — PostgreSQL § kabutė: per sunku."


@pytest.fixture
def offline(monkeypatch):
    """Forces degraded graph-only mode: unreachable Qdrant, no embedding keys."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")  # nothing listens here
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def ws(offline) -> Iterator[Tuple[MitosConfig, MitosSyncManager]]:
    """An initialised temp workspace + a manager, in offline graph-only mode."""
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    cmd_init(config)
    yield config, MitosSyncManager(config)
    shutil.rmtree(tmp, ignore_errors=True)


def _record_glyph(m: MitosSyncManager, slug: str) -> None:
    """Records one glyph-bearing decision into the offline graph."""
    res = m.record_decision_entry(
        axiom=GLYPH_AXIOM,
        rejected_paths=GLYPH_REJECTED,
        scope=["enc"],
        slug=slug,
    )
    assert "error" not in res, res


# --------------------------------------------------------------------------- #
# T1/W1 — raw-glyph JSON emission e2e (CLI verb + MCP twin)
# --------------------------------------------------------------------------- #

def test_cli_list_json_emits_raw_glyphs(ws, capsys) -> None:
    """`mitos list --json` over glyph content emits raw glyphs that round-trip."""
    config, m = ws
    _record_glyph(m, "enc-one")
    capsys.readouterr()  # drain the init banner

    cmd_list(config, scope="enc", as_json=True)
    out = capsys.readouterr().out

    assert "—" in out and "§" in out and "ąčęėįšųūž" in out
    assert "\\u" not in out  # no escape noise on a UTF-8 capture stream
    payload = json.loads(out)
    assert payload["decisions"][0]["axiom"] == GLYPH_AXIOM
    assert payload["decisions"][0]["rejected_paths"] == GLYPH_REJECTED


def test_mcp_list_decisions_emits_raw_glyphs(ws) -> None:
    """The MCP `list_decisions` twin returns raw-glyph JSON that round-trips."""
    from mitos import mcp_server
    config, m = ws
    _record_glyph(m, "enc-one")
    store = GraphStore(config.db_path, read_only=True)

    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        out = mcp_server.list_decisions(scope="enc")

    assert "—" in out and "§" in out and "ąčęėįšųūž" in out
    assert "\\u" not in out
    payload = json.loads(out)
    assert payload["decisions"][0]["axiom"] == GLYPH_AXIOM


# --------------------------------------------------------------------------- #
# T3 — CLI⇄MCP ensure_ascii parity (the structural anti-drift proof)
# --------------------------------------------------------------------------- #

def test_cli_mcp_glyph_parity(ws, capsys) -> None:
    """`list --json` and `list_decisions` treat the same glyphs identically.

    With no parked open questions the two surfaces build a byte-identical
    `decisions` section — pin that they agree, and that both leave the glyphs
    raw (unescaped) on a UTF-8 stdout.
    """
    from mitos import mcp_server
    config, m = ws
    _record_glyph(m, "enc-one")
    store = GraphStore(config.db_path, read_only=True)

    capsys.readouterr()
    cmd_list(config, scope="enc", as_json=True)
    cli_payload = json.loads(capsys.readouterr().out)

    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        mcp_out = mcp_server.list_decisions(scope="enc")
    mcp_payload = json.loads(mcp_out)

    # Same glyph treatment: both unescaped, and the same decision content.
    assert cli_payload["decisions"] == mcp_payload["decisions"]
    assert "\\u" not in mcp_out
    assert "—" in mcp_out and "§" in mcp_out


# --------------------------------------------------------------------------- #
# R6 re-exercised through the emitter — non-UTF-8 stdout falls back to \uXXXX
# --------------------------------------------------------------------------- #

def test_cli_list_json_ascii_fallback(ws, monkeypatch) -> None:
    """On a non-UTF-8 stdout the CLI `--json` emit stays valid pure-ASCII JSON.

    The adaptive resolution must be wired through the emitter (`_emit_json`),
    not just present in the 1a unit: a real ascii stdout drives `ensure_ascii`
    to True, the glyphs escape to `\\uXXXX`, and nothing raises.
    """
    config, m = ws
    _record_glyph(m, "enc-one")

    ascii_stream = io.TextIOWrapper(io.BytesIO(), encoding="ascii", newline="")
    monkeypatch.setattr(sys, "stdout", ascii_stream)
    cmd_list(config, scope="enc", as_json=True)  # must not raise UnicodeEncodeError
    ascii_stream.flush()
    raw = ascii_stream.buffer.getvalue()

    assert raw.isascii()  # backslashreplace/escape fallback kept it pure-ASCII
    payload = json.loads(raw.decode("ascii"))  # still valid JSON, round-trips
    assert payload["decisions"][0]["axiom"] == GLYPH_AXIOM


# --------------------------------------------------------------------------- #
# MCP single-line error returns stay single-line (indent=None preserved)
# --------------------------------------------------------------------------- #

def test_mcp_error_return_stays_single_line() -> None:
    """A non-letter depth error return is single-line JSON (indent=None kept)."""
    from mitos.mcp_server import query_decisions
    out = query_decisions(query="anything", depth="trace")
    assert "\n" not in out  # single-line — widening to indent=2 would be a text change
    assert "error" in json.loads(out)


# --------------------------------------------------------------------------- #
# Zero text-path change — the no-flag text path is untouched by the JSON flip
# --------------------------------------------------------------------------- #

def test_cli_list_text_path_unperturbed(ws, capsys) -> None:
    """`mitos list` (no --json) still renders human text, not the JSON wrapper.

    The flip only touches the `--json` branch; the text branch's `print(f"…")`
    lines are untouched. The glyphs render as themselves on a UTF-8 capture and
    the output is the human table, never a JSON document.
    """
    config, m = ws
    _record_glyph(m, "enc-one")
    capsys.readouterr()

    cmd_list(config, scope="enc")  # no as_json
    out = capsys.readouterr().out

    assert "enc-one" in out  # the human listing
    assert not out.lstrip().startswith("{")  # not the JSON document
    assert '"decisions"' not in out  # the JSON wrapper key never leaks into text


# --------------------------------------------------------------------------- #
# §3-(1) nudge — own stderr line, absent from stdout (stale-premise pin)
# --------------------------------------------------------------------------- #

def test_mcp_nudge_on_stderr_absent_from_stdout(ws, capsys, monkeypatch) -> None:
    """The 💡 MCP nudge fires on its own stderr line and never touches stdout.

    Pins the already-satisfied state (the nudge has gone to a separate stderr
    line since commit 8b3c90f, 2026-06-12 — the vision's "concatenates onto the
    axiom" premise was stale at authoring). A regression that re-routes it onto
    the stdout JSON/text body is caught here.
    """
    config, _ = ws
    # Enable the nudge: it is suppressed by the autouse hermetic fixture.
    monkeypatch.delenv("MITOS_NO_MCP_HINT", raising=False)
    monkeypatch.chdir(config.workspace_dir)  # main() builds MitosConfig(".")
    capsys.readouterr()

    with patch.object(sys, "argv", ["mitos", "list", "--json"]):
        main()

    captured = capsys.readouterr()
    assert "💡" in captured.err  # the nudge fired on stderr
    assert "💡" not in captured.out  # never on the stdout JSON body
    json.loads(captured.out)  # stdout is clean JSON (the nudge didn't corrupt it)
