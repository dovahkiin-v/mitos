"""Canonical, versioned agent-file block + drift detection.

The snippet a project pastes into its agent-instruction files (``AGENTS.md`` /
``CLAUDE.md`` / ``GEMINI.md`` / ``.cursorrules``) so the next agent that opens the
project knows mitos is set up here and how to use it.

It is a **copy**, and copies drift. Two design choices keep that manageable:

- **The block is deliberately thin** — durable pointers only (run ``mitos status``,
  read SETUP.md, the tools are self-describing). The volatile detail (required
  fields, the slug rule and its length cap, when-to-record) lives in the always-fresh
  surfaces: the MCP tool schemas (which ship with the code and *enforce* the
  required args) and SETUP.md-on-GitHub. If the copy holds no detail that can change,
  there is little to go stale.
- **The block carries a version marker** (``<!-- mitos-agent-guide: vN -->``) so the
  rare time the pointer block itself changes, a pasted copy that predates it can be
  *detected* (``mitos status``) and *refreshed* (``mitos agent-block``).

This module is the single source of truth for both the emitted block and the drift
check, so the two can never disagree.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

# Bump ONLY when the canonical block's wording changes in a way worth nudging pasted
# copies to refresh. The marker embeds this number; `mitos status` compares a project's
# pasted marker against it. (Not tied to the package __version__ — most releases don't
# touch the block.)
AGENT_GUIDE_VERSION = 2

# The agent-instruction filenames mitos knows about (the SETUP.md §6 list), checked at
# the project root only — these are where the block is pasted.
AGENT_FILENAMES = ("AGENTS.md", "CLAUDE.md", "GEMINI.md", ".cursorrules")

_MARKER_RE = re.compile(r"<!--\s*mitos-agent-guide:\s*v(\d+)", re.IGNORECASE)

# A file "carries the block" only if it has the marker OR the block's distinctive
# *heading* — a markdown heading line naming both "Architectural Decisions" and
# "Mitos" (true of every version of the block, including pre-marker pastes). Keyed on
# the heading, not a loose "mentions mitos", so a file that merely *discusses* mitos
# (e.g. mitos's own dev guide, or a doc that references `record_decision`) is NOT
# mistaken for a stale paste — precision over recall, so the drift signal never cries
# wolf. A custom-worded note without the heading is the author's own; once refreshed
# via `mitos agent-block` it gains the marker and is tracked from then on.
_BLOCK_HEADING_RE = re.compile(
    r"^#{1,6}\s+Architectural Decisions\b.*\bMitos\b", re.IGNORECASE | re.MULTILINE
)


def agent_block(version: int = AGENT_GUIDE_VERSION) -> str:
    """Builds the canonical agent-file block (paste-ready markdown).

    Args:
        version: The guide version to stamp into the marker. Defaults to the
            current :data:`AGENT_GUIDE_VERSION`; overridable only for tests.

    Returns:
        The full block, opened and closed by ``mitos-agent-guide`` HTML-comment
        markers (invisible in rendered markdown, greppable for drift detection).
    """
    return (
        f"<!-- mitos-agent-guide: v{version} — managed block, refresh with "
        f"`mitos agent-block`. -->\n"
        "## Architectural Decisions — Mitos (per-project)\n"
        "This project uses **mitos** for architectural decision memory — markdown for "
        "humans, a graph for the agents. It is **per-project** (its own `.mitos/` "
        "workspace + Qdrant collection).\n"
        "\n"
        "- **Check it's ready:** run `mitos status`. If it isn't `READY ✓`, follow the "
        "setup guide — https://github.com/dovahkiin-v/mitos/blob/main/SETUP.md (or the "
        "`mitos-setup` skill). If `mitos` is `command not found`, it was uninstalled "
        "after setup — reinstall it (pipx); don't silently drop decision-recording.\n"
        "- **Record** ADR-worthy decisions with `record_decision`; **check precedents** "
        "with `surface_decisions` / `query_decisions` *before* deciding. The tools are "
        "self-describing — their schemas state the required fields and constraints "
        "(including the slug handle) — and SETUP.md is the full guide: what's worth "
        "recording, how to choose a slug, and how to link related decisions.\n"
        "- **Keep the graph honest:** after a burst of `record_decision` writes — or "
        "when starting a session — run `mitos check` to audit the live corpus for "
        "undeclared contradictions (read-only; it never edits anything). Resolve any "
        "finding the normal way: declare the relationship in `decisions.md`.\n"
        "<!-- /mitos-agent-guide -->"
    )


def marker_version(text: str) -> Optional[int]:
    """Returns the guide version declared by a block marker in ``text``, or None."""
    m = _MARKER_RE.search(text)
    return int(m.group(1)) if m else None


def _carries_block(text: str) -> bool:
    """True if ``text`` looks like it carries the mitos block (marker or its heading)."""
    return _MARKER_RE.search(text) is not None or _BLOCK_HEADING_RE.search(text) is not None


def scan_agent_files(workspace_dir: str) -> List[Dict[str, object]]:
    """Inspects the project's agent files for the mitos block and its version.

    Only files that actually *carry the block* are reported — detected by the version
    marker or the block's distinctive heading (see :data:`_BLOCK_HEADING_RE`). A file
    that merely discusses mitos, or has no mitos note at all, is not "stale" — it is
    simply uninvolved, and omitted.

    Args:
        workspace_dir: The project root to scan.

    Returns:
        One dict per existing, block-bearing agent file:
        ``{"file", "path", "marker_version", "status"}`` where ``status`` is
        ``"current"`` (marker ≥ the running guide version), ``"outdated"`` (marker
        behind), or ``"unversioned"`` (the block heading is present but carries no
        marker — a pre-versioning paste). In ``AGENT_FILENAMES`` order.
    """
    out: List[Dict[str, object]] = []
    for name in AGENT_FILENAMES:
        path = os.path.join(workspace_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue
        if not _carries_block(text):
            continue
        v = marker_version(text)
        if v is None:
            status = "unversioned"
        elif v >= AGENT_GUIDE_VERSION:
            status = "current"
        else:
            status = "outdated"
        out.append({"file": name, "path": path, "marker_version": v, "status": status})
    return out


def agent_block_drift(workspace_dir: str) -> Dict[str, object]:
    """Summarises agent-file block staleness for ``mitos status`` / ``agent-block --check``.

    Args:
        workspace_dir: The project root to scan.

    Returns:
        ``{"stale", "files"}`` — ``stale`` is True when at least one
        mitos-referencing agent file is ``outdated`` or ``unversioned``; ``files``
        is the per-file scan from :func:`scan_agent_files`.
    """
    files = scan_agent_files(workspace_dir)
    stale = any(f["status"] in ("outdated", "unversioned") for f in files)
    return {"stale": stale, "files": files}
