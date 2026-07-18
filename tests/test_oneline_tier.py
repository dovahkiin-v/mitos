"""Tests for the oneline output tier and the shared word-boundary truncation
helper (the render-dedupe-primary-tag-full-body-oneline-tier ADR).

Covers:
* ``display.truncate_words`` — the single truncation seam every preview cut
  routes through (no mid-word cuts, ellipsis only when truncated).
* ``mitos list --oneline`` — CLI text rows, ``--json`` minimal objects, modifier
  stamps surviving the thinner tier, and the ``--brief`` mutual exclusion.
* MCP ``list_decisions(oneline=True)`` — payload parity with the CLI JSON twin
  and the brief+oneline error.

Driven fully offline (unreachable Qdrant + no keys) — pure graph reads.
"""

import json
import shutil
import sys
import tempfile
from typing import Iterator, Tuple

import pytest
from unittest.mock import patch

from mitos.config import MitosConfig
from mitos.cli import cmd_init, cmd_list, main
from mitos.display import oneline_axiom, oneline_payload, truncate_words
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager


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


def _record(m: MitosSyncManager, slug: str, axiom: str, scope, amends=None) -> None:
    res = m.record_decision_entry(
        axiom=axiom,
        rejected_paths=f"Rejected alternative for {slug}.",
        scope=scope,
        slug=slug,
        amends=amends,
        acknowledge_neighbors=True,
    )
    assert "error" not in res, res


# --------------------------------------------------------------------------- #
# truncate_words — the shared word-boundary truncation seam
# --------------------------------------------------------------------------- #

def test_truncate_words_short_text_unchanged() -> None:
    """Text at or under the limit is returned unchanged — no ellipsis."""
    assert truncate_words("short", 60) == "short"
    exact = "x" * 60
    assert truncate_words(exact, 60) == exact


def test_truncate_words_cuts_at_word_boundary_with_ellipsis() -> None:
    """A truncated result ends with … at a word boundary, within the limit."""
    text = "We use SQLite in WAL mode for the graph store because it is local-first."
    out = truncate_words(text, 40)
    assert out.endswith("…")
    assert len(out) <= 40
    kept = out[:-1]
    # No mid-word cut: the kept prefix is a whole-word prefix of the original.
    assert text.startswith(kept)
    assert text[len(kept)] == " "


def test_truncate_words_single_long_token_hard_cuts() -> None:
    """One unbroken token longer than the limit is hard-cut (still ellipsised)."""
    token = "a" * 100
    out = truncate_words(token, 30)
    assert out.endswith("…") and len(out) == 30
    assert out[:-1] == token[:29]


def test_truncate_words_keeps_whole_words_only() -> None:
    """The kept prefix is always whole words — never a fragment of a cut word."""
    text = "ab supercalifragilisticexpialidocious continues"
    out = truncate_words(text, 20)
    # Backing to the boundary keeps the whole first word, not "supercal…".
    assert out == "ab…"


# --------------------------------------------------------------------------- #
# CLI: mitos list --oneline
# --------------------------------------------------------------------------- #

LONG_AXIOM = ("The renderer dedupes scope files by primary tag so render weight "
              "stops converging toward tags times corpus while every decision "
              "keeps exactly one full Letter-complete body somewhere.")


def test_cmd_list_oneline_text_rows(ws, capsys) -> None:
    """--oneline emits one row per decision: slug + word-boundary-truncated axiom."""
    config, m = ws
    _record(m, "row-one", LONG_AXIOM, scope=["tier"])
    _record(m, "row-two", "Short axiom.", scope=["tier"])
    capsys.readouterr()

    cmd_list(config, scope="tier", oneline=True)
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if l.startswith("row-")]
    assert len(lines) == 2
    long_row = next(l for l in lines if l.startswith("row-one"))
    assert "…" in long_row  # the long axiom was truncated at a word boundary
    assert len(long_row) <= 104  # ~100-char row budget (slug + axiom)
    short_row = next(l for l in lines if l.startswith("row-two"))
    assert "Short axiom." in short_row and "…" not in short_row


def test_cmd_list_oneline_json_minimal_shape_and_stamps(ws, capsys) -> None:
    """--oneline --json emits {slug, axiom_oneline, state} + modifier keys."""
    config, m = ws
    _record(m, "base-call", LONG_AXIOM, scope=["tier"])
    _record(m, "later-call", "We refined the base call.", scope=["tier"],
            amends="base-call")
    capsys.readouterr()

    cmd_list(config, scope="tier", as_json=True, oneline=True)
    out = json.loads(capsys.readouterr().out)
    by_slug = {d["slug"]: d for d in out["decisions"]}
    plain = by_slug["later-call"]
    assert set(plain) == {"slug", "axiom_oneline", "state"}
    assert plain["state"] == "active"
    # Stamps survive the thinner tier: the amended decision carries amended_by.
    amended = by_slug["base-call"]
    assert amended["amended_by"] == ["later-call"]
    assert amended["axiom_oneline"].endswith("…")
    assert "rejected_paths" not in amended and "axiom" not in amended


def test_cmd_list_oneline_text_keeps_modifier_marker(ws, capsys) -> None:
    """The text row carries the compact ⚠ marker for a modified decision."""
    config, m = ws
    _record(m, "old-shape", "The original shape.", scope=["tier"])
    _record(m, "new-shape", "The refined shape.", scope=["tier"], amends="old-shape")
    capsys.readouterr()

    cmd_list(config, scope="tier", oneline=True)
    out = capsys.readouterr().out
    marker_line = next(l for l in out.splitlines() if l.startswith("old-shape"))
    assert "⚠" in marker_line and "amended by: new-shape" in marker_line


def test_cli_brief_and_oneline_mutually_exclusive(ws, monkeypatch, capsys) -> None:
    """`mitos list --brief --oneline` is an argparse error (exit 2)."""
    monkeypatch.setattr(sys, "argv", ["mitos", "list", "--brief", "--oneline"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# MCP: list_decisions(oneline=True) — CLI⇄MCP parity
# --------------------------------------------------------------------------- #

def test_mcp_oneline_matches_cli_json(ws, capsys) -> None:
    """MCP oneline decisions are byte-shape-identical to the CLI --json twin."""
    from mitos import mcp_server
    config, m = ws
    _record(m, "parity-long", LONG_AXIOM, scope=["tier"])
    _record(m, "parity-amend", "Refines the long one.", scope=["tier"],
            amends="parity-long")
    capsys.readouterr()

    cmd_list(config, scope="tier", as_json=True, oneline=True)
    cli_out = json.loads(capsys.readouterr().out)

    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, None, None)):
        mcp_out = json.loads(mcp_server.list_decisions(scope="tier", oneline=True))

    assert mcp_out["decisions"] == cli_out["decisions"]
    assert mcp_out["total"] == cli_out["total"] == 2


def test_mcp_brief_and_oneline_error(ws) -> None:
    """list_decisions(brief=True, oneline=True) returns an in-band error object."""
    from mitos import mcp_server
    config, _ = ws
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components",
                      return_value=(store, None, None)):
        resp = json.loads(mcp_server.list_decisions(brief=True, oneline=True))
    assert "error" in resp and "mutually exclusive" in resp["error"]


def test_oneline_payload_unit_shape() -> None:
    """oneline_payload/oneline_axiom shape the minimal un-stamped core."""
    node = {"slug": "s" * 60, "core_axiom": "word " * 40, "computed_state": "active"}
    p = oneline_payload(node)
    assert set(p) == {"slug", "axiom_oneline", "state"}
    # A very long slug can't starve the axiom below the floor.
    assert len(p["axiom_oneline"]) >= 20
    assert p["axiom_oneline"] == oneline_axiom(node)
