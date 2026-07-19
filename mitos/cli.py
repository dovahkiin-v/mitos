"""CLI entry point for Mitos.

This module implements the command-line interface for Mitos, coordinating
initialization, sync, ambient capture, querying, list, render, import, and
MCP serving.
"""

import sys
import os
import re
import time
import json
import uuid
import sqlite3
import hashlib
import argparse
from datetime import datetime, timezone
from typing import Callable, List, Optional, Dict, Any, Set, Tuple
from google import genai

from mitos import __version__
from mitos import check
from mitos.display import (
    apply_stdout_text_safety,
    blackout_note,
    clamp_limit,
    dumps_display,
    letter_payload,
    oneline_axiom,
    oneline_payload,
    order_scope_counts,
    truncate_words,
    resolve_display_ensure_ascii,
    show_payload,
    SHOW_NOT_FOUND_HINT,
)
from mitos.config import (
    MitosConfig,
    CONFIG_DEFAULTS,
    default_collection_name,
    global_env_path,
    hint_due,
)
from mitos.errors import (
    MitosError, ParseError, ValidationError, DatabaseError, ConfigError,
    VectorStoreError, EmbeddingError,
)
from mitos.vector_store import scroll_point_ids, hash_to_uuid, QdrantVectorStore
from mitos.embeddings import GeminiEmbeddingProvider
from mitos.telemetry import TelemetryStore, ConflictCheckRow, JudgmentBatch
from mitos.identity import compute_node_id
from mitos.models import get_model_id
from mitos.parser import ParsedEntry, parse_entry_stream, read_text_or_none
from mitos.conflict import run_conflict_check, ConflictUnavailableReason
from mitos.migrations import is_pre_v1a_schema
from mitos.store import GraphStore, MODIFIER_EDGE_KEYS, open_connection
from mitos.cutover import default_aside_db_path, perform_swap, rebuild_and_gate
from mitos.lexical import degraded_reason_from_error, lexical_fallback
from mitos.recall import (assess_surface_recall, corpus_provenance,
                          provenance_line, scope_filter_recovery)
from mitos.sync import MitosSyncManager, run_ambient_capture, _SLUG_MAX_LEN
from mitos._agent_block import agent_block, agent_block_drift, AGENT_GUIDE_VERSION
from mitos.renderer import MitosRenderer, overflow_report
from mitos.importer import MitosProseImporter


# Worked-examples block rendered at the foot of `mitos --help` / `mitos -h`.
# Lazy by design (§6 description economy): zero context cost until --help is
# invoked, so it can afford the expansive examples the eager MCP descriptions
# can't. Teaches the surface→record reflex, the scope vocabulary, workspace
# targeting (-C), and the relation-edge guidance. Every flag/verb shown here is
# real CLI surface — keep it runnable, never invent a flag (there is no
# --retired edge). Rendered verbatim via RawDescriptionHelpFormatter.
_EPILOG = """\
Examples:
  # Before deciding: surface precedent, then record the outcome you chose
  mitos surface "cache invalidation strategy"
  mitos record "Write-through cache for session data" \\
    --rejected "write-back: data loss on crash" --scope cache --slug write-through-sessions

  # Discover the scope vocabulary before you invent a near-duplicate tag
  mitos scopes

  # Operate on another workspace without cd (git's -C; must precede the verb)
  mitos -C /path/to/repo list --scope auth

Relating a decision to a prior (pass the prior's EXACT slug):
  --supersedes a,b   priors you've outgrown / evolved past (comma-separated for several)
  --corrects   slug  a prior that was WRONG (not merely outgrown)
  (no "retired" edge type exists — a decision dies by being superseded or corrected)
"""


def _emit_json(obj: Any, *, indent: Optional[int] = 2) -> None:
    """Prints a display payload as adaptive-``ensure_ascii`` JSON to stdout.

    The single CLI display-JSON emit path: it resolves ``ensure_ascii`` against
    the *live* ``sys.stdout`` at call time (a pytest capture, a pipe, or a real
    terminal), so raw glyphs emit on a UTF-8 stdout and fall back to ``\\uXXXX``
    escapes on a non-UTF-8 one — never a ``UnicodeEncodeError``. Centralizing the
    resolution here keeps all CLI sites uniform and the CLI⇄MCP seam single.

    Args:
        obj: A JSON-native display payload.
        indent: Pretty-print indent; ``None`` for single-line output.

    Returns:
        None.
    """
    print(dumps_display(obj, ensure_ascii=resolve_display_ensure_ascii(sys.stdout), indent=indent))


# Shared route-to-cutover guidance. `mitos init` raises it (DatabaseError),
# `mitos status` reports it (both the pre-V1a check-line and the next-steps line)
# when a prototype graph is detected, so every operator surface points the same
# direction and names the same verb. Mirrors the substance of the
# GraphStore.__init__ boot-guard message (store.py) + vision §2.1; the one-time
# `mitos cutover` verb itself is implemented below (cmd_cutover).
_CUTOVER_GUIDANCE = (
    "This graph predates the V1a schema (a prototype layout was detected). "
    "Mitos will not migrate it in place — run the one-time cutover (`mitos "
    "cutover`) to rebuild it into the V1a store (see SETUP.md → Cutover). "
    "Meanwhile the markdown gold source still answers: `mitos surface`/`query` "
    "fall back to a text match over decisions.md, and `grep decisions.md` "
    "always works — nothing is lost."
)


def _modifier_marker(payload: Dict[str, Any]) -> str:
    """Builds a one-line staleness marker from a payload's modifier keys.

    Reads the reverse-relation keys (``superseded_by``/``amended_by``/… set by
    :meth:`GraphStore.get_modifiers`) off an already-shaped payload and renders a
    compact ``⚠ amended by: <slug>`` marker so a human scanning text output sees
    that a still-live axiom has been moved on from. Empty when the node is unmodified.

    Args:
        payload: A decision payload that may carry reverse-relation modifier keys.

    Returns:
        A ``⚠ …`` marker string, or ``""`` when there are no modifiers.
    """
    parts = []
    for key in MODIFIER_EDGE_KEYS.values():
        slugs = payload.get(key)
        if slugs:
            parts.append(f"{key.replace('_', ' ')}: {', '.join(slugs)}")
    return ("⚠ " + "; ".join(parts)) if parts else ""


def _oq_modifiers(oq: Dict[str, Any]) -> Dict[str, List[str]]:
    """Lifts the reverse-relation modifier keys already stamped on an OQ dict.

    ``GraphStore.get_open_questions`` routes through the 2b modifier chokepoint, so
    a still-active OQ that a later ``amends`` / ``narrows`` has moved on from already
    carries ``amended_by`` / ``narrowed_by``. This returns the present (non-empty)
    modifier keys so the user-facing OQ output carries them too — the OQ analogue of
    the decision-side ``item.update(modifiers.get(d["id"], {}))``, read straight off
    the stamped payload (no separate ``get_modifiers_map`` call), so an amended OQ
    never reads as the final word.

    Args:
        oq: A hydrated, modifier-stamped open-question dict from
            ``get_open_questions``.

    Returns:
        A dict of the present reverse-relation keys to their slug lists (empty when
        the OQ is unmodified).
    """
    return {key: oq[key] for key in MODIFIER_EDGE_KEYS.values() if oq.get(key)}


def _oq_payload(oq: Dict[str, Any]) -> Dict[str, Any]:
    """Builds the machine-readable per-OQ dict shared by every OQ ``--json`` surface.

    The single source of the open-question JSON shape: ``cmd_list``'s
    ``open_questions[]`` array and ``cmd_open_questions --json`` both emit this so an
    agent sees one OQ schema across both verbs. The present modifier keys ride via
    ``_oq_modifiers`` (an amended-but-active OQ carries ``amended_by``/``narrowed_by``
    so it never reads as the final word); the decision-only keys
    (``superseded_by``/``corrected_by``) never appear because ``get_open_questions``
    never stamps them on an OQ — the subset is structural, not filtered here.

    Args:
        oq: A hydrated, modifier-stamped open-question dict from
            ``get_open_questions``.

    Returns:
        A JSON-native dict with ``topic``, ``questions_raised``, ``park_reason``, and
        any present reverse-relation modifier keys.
    """
    return {"topic": oq["slug"], "questions_raised": oq["questions_raised"],
            "park_reason": oq.get("park_reason"), **_oq_modifiers(oq)}


def load_format_spec() -> str:
    """Loads the canonical format specification from the package's single source of truth."""
    spec_path = os.path.join(os.path.dirname(__file__), "format-spec.md")
    with open(spec_path, "r", encoding="utf-8") as f:
        return f.read()


def _ensure_gitignore_entry(gitignore_path: str, entry: str) -> None:
    """Ensures ``entry`` is present in ``.gitignore``, creating the file if needed.

    Keeps a scaffolded ``.env`` (which will hold real API keys) out of version
    control. Idempotent — a no-op when the entry is already a line in the file.

    Args:
        gitignore_path: Path to the workspace ``.gitignore``.
        entry: The line to ensure is present (e.g. ``".env"``).
    """
    existing = ""
    if os.path.exists(gitignore_path):
        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                existing = f.read()
        except OSError:
            return
        if entry in existing.splitlines():
            return
    sep = "" if (not existing or existing.endswith("\n")) else "\n"
    try:
        with open(gitignore_path, "a", encoding="utf-8") as f:
            f.write(f"{sep}{entry}\n")
    except OSError:
        pass


def _extract_sample_block(spec: str, header: str) -> str:
    """Extracts the fenced markdown sample under a ``## N`` header from format-spec.md.

    The spec carries one worked sample per kind inside a ```` ```markdown ```` fence:
    ``## 3. Sample Entry`` (decisions) and ``## 4. Open Question Sample`` (questions).
    ``mitos init`` lifts each into the matching buffer's preamble so a fresh
    ``decisions.md`` / ``questions.md`` shows the author the canonical shape. Only
    these two sections carry a fenced sample; the ``## 1`` / ``## 2`` field-definition
    sections do not, so this helper serves exactly those two callers.

    Args:
        spec: The full ``format-spec.md`` content.
        header: The section header to match (e.g. ``"## 3. Sample Entry"``).

    Returns:
        The sample block's inner text (stripped), or ``""`` if no fenced sample
        follows the header.
    """
    match = re.search(
        rf"{re.escape(header)}.*?\n```markdown\n(.*?)\n```",
        spec,
        re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _toml_scalar(value: Any) -> str:
    """Serializes a v0.1 config scalar to its TOML right-hand-side literal.

    A deliberately tiny serializer for exactly the value set the schema uses —
    plain strings (no embedded ``"`` or newline), integers, and booleans — NOT a
    general TOML writer. The stdlib ``tomllib`` is read-only and P19 forbids pulling
    ``tomli-w`` for a handful of flat scalars, so ``mitos init`` seeds ``config.toml``
    through this (mirrors the project's hand-rolled ``.env``/config readers).

    The ``bool`` branch MUST stay above the ``int`` branch: ``bool`` subclasses
    ``int``, so an int-first order would emit ``True`` as ``1`` instead of ``true``.

    Args:
        value: The config value to serialize (``str``, ``int``, or ``bool``).

    Returns:
        The TOML literal — e.g. ``'"archive"'`` for a string, ``'50'`` for an int,
        ``'true'``/``'false'`` for a bool.

    Raises:
        TypeError: If the value is not a plain ``str``/``int``/``bool``, or is a
            ``str`` containing a ``"`` or newline (beyond this serializer's scope).
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        if '"' in value or "\n" in value:
            raise TypeError(
                f"_toml_scalar only handles simple strings without quotes or "
                f"newlines (got {value!r})"
            )
        return f'"{value}"'
    raise TypeError(
        f"_toml_scalar cannot serialize {type(value).__name__}: {value!r}"
    )


def cmd_init(config: MitosConfig) -> None:
    """Initializes (or idempotently re-initializes) the Mitos workspace.

    Scaffolds the V1a ``.mitos/`` layout: the graph boots at the migration-ladder
    head, ``config.toml`` is seeded from the single-source ``CONFIG_DEFAULTS``,
    ``format-spec.md`` is installed from the package (refresh-on-mismatch), and the
    ``decisions.md`` / ``questions.md`` buffers are seeded only when absent. A
    re-run is idempotent: present config/buffers are left untouched, a deleted
    buffer is re-seeded, the ladder re-runs as a no-op (§5.2.7). A pre-V1a
    (prototype) graph is refused **before any file mutation** with route-to-cutover
    guidance, never ladder-advanced into a hybrid.

    Args:
        config: The workspace configuration to initialize.

    Raises:
        DatabaseError: If a pre-V1a (prototype) graph is detected — the workspace
            is left in its pre-init state; route the operator to the cutover.
    """
    # 0. Refuse a pre-V1a (prototype) graph BEFORE any file mutation. The RW
    #    GraphStore boot guard would also refuse it, but only at the very end —
    #    after config/.env/skill/buffers were written. §5.2.7 requires
    #    abort-before-partial-mutation, so probe explicitly up front (read-only;
    #    open_connection's mode=ro needs the file to exist, hence the guard) and
    #    raise with route-to-cutover guidance, leaving the directory untouched.
    if os.path.exists(config.db_path):
        probe_conn = open_connection(config.db_path, read_only=True)
        try:
            if is_pre_v1a_schema(probe_conn):
                raise DatabaseError(_CUTOVER_GUIDANCE)
        finally:
            probe_conn.close()

    os.makedirs(config.mitos_dir, exist_ok=True)

    # 1. Install format-spec.md from the package — the C5 single source of truth.
    #    Refresh-on-mismatch (V1-D7 / §5.2.7): absent -> install; present but drifted
    #    from the shipped copy -> overwrite with a calm one-line warning naming both
    #    short hashes (never a silent overwrite, never a silent stale-skip). skill.md
    #    embeds the spec, so it is regenerated every init (below) and stays in lockstep.
    format_spec_path = os.path.join(config.workspace_dir, "format-spec.md")
    format_spec_content = load_format_spec()
    if not os.path.exists(format_spec_path):
        with open(format_spec_path, "w", encoding="utf-8") as f:
            f.write(format_spec_content)
    else:
        with open(format_spec_path, "r", encoding="utf-8") as f:
            on_disk_spec = f.read()
        if on_disk_spec != format_spec_content:
            old_hash = hashlib.sha256(on_disk_spec.encode("utf-8")).hexdigest()[:12]
            new_hash = hashlib.sha256(format_spec_content.encode("utf-8")).hexdigest()[:12]
            with open(format_spec_path, "w", encoding="utf-8") as f:
                f.write(format_spec_content)
            print(
                f"Refreshed format-spec.md to match the installed Mitos package "
                f"({old_hash} → {new_hash})."
            )

    # Extract the canonical sample for each buffer from the spec (one helper, both
    # kinds): the ## 3 decision sample and the ## 4 open-question sample.
    decision_sample = _extract_sample_block(format_spec_content, "## 3. Sample Entry")
    question_sample = _extract_sample_block(format_spec_content, "## 4. Open Question Sample")

    # 1a. Seed config.toml when absent — from the single-source CONFIG_DEFAULTS map
    #     (P11 / WIRING_LEDGER entry-004), NOT hand-copied literals, so a seeded file
    #     and the loader's deleted-key fallback can never diverge. The seven static
    #     keys serialize in CONFIG_DEFAULTS order; the two dynamic qdrant_* lines
    #     follow (env-/workspace-derived defaults, computed in MitosConfig.__init__).
    #     NO pending_threshold line — it left the v0.1 file schema (the loader would
    #     warn-tolerate it on every command).
    config_path = os.path.join(config.mitos_dir, "config.toml")
    if not os.path.exists(config_path):
        lines = ["# Mitos Workspace Configuration"]
        for key, default in CONFIG_DEFAULTS.items():
            lines.append(f"{key} = {_toml_scalar(default)}")
        lines += [
            "# Qdrant REST endpoint. Defaults to Mitos's dedicated :7333 (not the",
            "# standard :6333) so Mitos never co-locates its collections in another",
            "# Qdrant you run. Set QDRANT_URL before `init` or edit this line.",
            f"qdrant_url = {_toml_scalar(config.qdrant_url)}",
            "# Per-project collection: keeps this project's vectors isolated",
            "# from other Mitos workspaces sharing the same Qdrant instance.",
            f"qdrant_collection = {_toml_scalar(config.qdrant_collection)}",
        ]
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # 1b. Scaffold a gitignored .env with the credential slots, so a human or
    #     LLM setting Mitos up knows exactly where to drop keys (empty by default).
    env_path = os.path.join(config.workspace_dir, ".env")
    if not os.path.exists(env_path):
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(
                "# ============================================================\n"
                "# Mitos API keys — fill in the value(s), then run `mitos sync`.\n"
                "# This file is gitignored; never commit real keys.\n"
                "# ============================================================\n\n"
                "# Google Gemini API key — REQUIRED (unless already set globally).\n"
                "# One key covers BOTH embeddings (semantic surface/query) AND\n"
                "# decision synthesis (sync/capture).\n"
                "# Tip: set it ONCE for every project with\n"
                "#   mitos set-key --global <KEY>   (stored in ~/.config/mitos/.env)\n"
                "# or drop a project-specific key on the line below to override it.\n"
                "# Get one: https://aistudio.google.com/app/apikey\n"
                "GEMINI_API_KEY=\n\n"
                "# Anthropic (Claude) API key — OPTIONAL. Only used by\n"
                "# `mitos import --llm-extract` to convert legacy prose ADRs.\n"
                "# Get one: https://console.anthropic.com/settings/keys\n"
                "ANTHROPIC_API_KEY=\n"
            )
    # Never let the .env (real keys) get committed.
    _ensure_gitignore_entry(os.path.join(config.workspace_dir, ".gitignore"), ".env")

    # 2. Always write/overwrite skill.md (by inclusion of format-spec.md)
    skill_path = os.path.join(config.mitos_dir, "skill.md")
    with open(skill_path, "w", encoding="utf-8") as f:
        f.write(
            "# Mitos Architecture Skill\n\n"
            "You are operating in a workspace governed by Mitos, an architectural decision graph.\n"
            "When you make an architectural decision or change a foundational pattern, you MUST record it in `decisions.md`.\n\n"
            "(If `mitos` itself is ever `command not found`, it was uninstalled after setup — reinstall it (pipx) or flag it to the human; don't silently drop decision-recording.)\n\n"
            "## Canonical Format Specification\n"
            "Your entries MUST adhere EXACTLY to the following markdown format (loaded from format-spec.md):\n\n"
            f"{format_spec_content}\n\n"
            "## Setup — API Keys\n"
            "Mitos reads keys from a `.env` file at the workspace root (`mitos init` scaffolds it with empty slots; it is gitignored). Set exactly one required key:\n"
            "- **`GEMINI_API_KEY`** (Google Gemini) — REQUIRED for semantic `surface_decisions`/`query_decisions` and for `mitos sync`. One key covers both embeddings and synthesis.\n"
            "- `ANTHROPIC_API_KEY` — OPTIONAL, only for `mitos import --llm-extract` (legacy prose import).\n"
            "Without `GEMINI_API_KEY`, `record_decision` still works (it commits to the local graph; the embedding is queued and drains on the next `mitos sync` once the key is set), but semantic surface/query are unavailable. If a tool reports a missing key, tell the human to put it in `.env`.\n\n"
            "Mitos uses its own Qdrant on **:7333** (not the standard :6333), started with `docker compose up -d`. If semantic tools report Qdrant unreachable, tell the human to start it; `record_decision` still works meanwhile (embeddings queue and drain once it's up).\n\n"
            "## Recording & recall — MCP tools (preferred) or CLI fallback\n"
            "All decisions you record, surface, and query are scoped to THIS project's decision graph and its own Qdrant collection — you will not see, and cannot contaminate, other projects' decisions.\n"
            "If the Mitos MCP server is wired into your agent, call these tools directly — best experience: structured args, no shell-quoting. If it is NOT wired, each maps to a CLI verb (and the CLI also accepts the long names as aliases, e.g. `mitos record_decision`):\n"
            "- `record_decision`  (CLI: `mitos record`) — the moment you commit to a foundational choice (a schema, a library, a pattern, a path you're abandoning), persist it WITH the alternatives you rejected and why, so future sessions inherit it instead of relitigating. Recording rich prose via the CLI? Use `--axiom-file -` / `--rejected-file -` / `--context-file -` to read from stdin and avoid shell-quoting.\n"
            "- `surface_decisions` (CLI: `mitos surface`) — surface active precedents for a claim/scope BEFORE you decide, so you don't relitigate a settled call. This is the recall loop — use it first. Every hit carries its full `rejected_paths`; pass `brief=True` (CLI `--brief`) for an axiom-only scan.\n"
            "- `query_decisions`   (CLI: `mitos query`) — semantic or slug lookup when unsure whether a precedent exists.\n"
            "- `list_decisions`    (CLI: `mitos list`) — the EXHAUSTIVE recall path. surface/query are semantic and capped at the top few matches; this returns EVERY decision in a scope, deterministically, so a completeness pass or audit doesn't miss anything below the relevance cliff. Needs no key or Qdrant.\n\n"
            "## When to record — the capture trigger (YOUR judgement; Mitos stores, it does not decide what is worth storing)\n"
            "Recall is easy to ask for; knowing WHAT is worth recording is the real call, and it falls to you. Record a decision when it:\n"
            "- sets a pattern future work must follow, or\n"
            "- forecloses a real alternative you weighed and rejected (capture WHY in `rejected_paths` — that is what stops the next agent re-proposing it), or\n"
            "- is structural or costly to reverse, or\n"
            "- reverses or supersedes a prior decision, or\n"
            "- has cross-cutting blast radius (touches many areas).\n"
            "Skip the local, easily-reversible, or already-settled choice. A quick self-test at any fork: *would the next agent waste time re-deriving or re-litigating this?* If yes, record it. When unsure, `surface_decisions` first — if nothing is there and it clears the bar, record it.\n\n"
            "## Linking decisions\n"
            "When a decision relates to an existing one, pass that one's EXACT slug to the matching relation arg so the graph stays connected instead of accumulating silent tension: `supersedes` (replaces it), `amends`, `narrows`, `depends_on`, `resolves`, `contradicts`, `cites`. On `record_decision` these are args; on the CLI they are flags (`--supersedes`, `--depends-on`, …). Look the target up first to get its exact slug. After you record, the result may list nearby existing decisions (`related`) — if one is genuinely connected, link it.\n"
        )

    # 3. Seed the decisions.md buffer when absent (with the extracted ## 3 sample).
    if not os.path.exists(config.decisions_file):
        with open(config.decisions_file, "w", encoding="utf-8") as f:
            f.write(
                "# Decisions for Mitos\n\n"
                "<!-- This file is managed by mitos. LLM integration: see .mitos/skill.md once V5 ships. -->\n"
                "<!-- DO NOT MODIFY ABOVE THIS LINE -->\n\n"
                "## SAMPLE FORMAT — auto-restored by mitos sync, do not modify or delete\n\n"
                f"{decision_sample}\n\n"
                "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->\n"
            )

    # 4. Seed the questions.md buffer when absent — the open-question authoring
    #    file (ADR open-questions-authored-in-separate-questions-md-file), parallel
    #    to decisions.md. The load-bearing parts are the BEGIN ENTRIES sentinel (the
    #    parser splits the preamble on that substring) and the ## 4 sample sitting in
    #    the preamble (it yields zero graph state on the first sync).
    if not os.path.exists(config.questions_file):
        with open(config.questions_file, "w", encoding="utf-8") as f:
            f.write(
                "# Open Questions for Mitos\n\n"
                "<!-- This file is managed by mitos. LLM integration: see .mitos/skill.md once V5 ships. -->\n"
                "<!-- DO NOT MODIFY ABOVE THIS LINE -->\n\n"
                "## SAMPLE FORMAT — auto-restored by mitos sync, do not modify or delete\n\n"
                f"{question_sample}\n\n"
                "<!-- BEGIN ENTRIES — new questions go directly below this line, newest first -->\n"
            )

    # Touch database to initialize — boots the V1a STRICT schema via the migration
    # ladder (fresh -> user_version=1; an existing V1a graph re-runs as a no-op). A
    # pre-V1a graph was already refused by the early probe above, so this never
    # ladder-advances a prototype into a hybrid.
    GraphStore(config.db_path)
    print(f"Initialized Mitos workspace at {config.workspace_dir} ✓")


def cmd_sync(config: MitosConfig, auto_accept: bool = False, embed_only: bool = False, verbose: bool = False) -> None:
    """Synchronizes the decisions write buffer with the graph store."""
    manager = MitosSyncManager(config)
    if embed_only:
        manager.drain_pending_embeddings()
    else:
        try:
            manager.perform_sync(auto_accept=auto_accept, verbose=verbose)
        except ParseError as e:
            print(f"Sync Aborted: Parse error in write-buffer.\n{str(e)}", file=sys.stderr)
            sys.exit(1)
        except ValidationError as e:
            print(f"Sync Aborted: Validation error.\n{str(e)}", file=sys.stderr)
            sys.exit(1)


def cmd_reconcile(config: MitosConfig, as_json: bool = False) -> int:
    """Re-embeds active nodes missing from Qdrant, healing a direct vector wipe.

    The one-command heal for the gap ``sync`` cannot reach: a bare Qdrant wipe
    (``curl -X DELETE`` of the collection, no ``rebuild``/``cutover``) leaves the
    graph populated, Qdrant empty, and the outbox empty — so ``sync`` drains
    nothing. Reconcile diffs the ACTIVE node set against Qdrant's actual point
    ids, enqueues the missing nodes, and drains. Idempotent.

    Args:
        config: The active workspace config.
        as_json: Whether to emit the result as a JSON object.

    Returns:
        Process exit code (0 on success, 1 if Qdrant/embedding provider is down).
    """
    manager = MitosSyncManager(config)
    try:
        result = manager.reconcile_embeddings()
    except VectorStoreError as e:
        msg = f"Reconcile unavailable — Qdrant or embedding provider down: {str(e)}"
        if as_json:
            print(json.dumps({"error": msg}))
        else:
            print(msg, file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps(result))
    else:
        print(
            f"Reconciled: {result['active']} active node(s), "
            f"{result['present']} point(s) already indexed, "
            f"{result['enqueued']} re-embedded."
        )
    return 0


def cmd_capture(config: MitosConfig, text: str) -> None:
    """Captures a raw architectural thought and appends it to decisions.md."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY environment variable is not set. Capture requires it.")
        return
        
    client = genai.Client(api_key=api_key)
    print("Synthesizing canonical decision entry ...")
    
    try:
        entry_text = run_ambient_capture(client, text)
    except Exception as e:
        print(f"Ambient capture failed: {str(e)}")
        return

    # Append below BEGIN ENTRIES line under advisory lock
    manager = MitosSyncManager(config)
    try:
        with manager.lock:
            with open(config.decisions_file, "r", encoding="utf-8") as f:
                content = f.read()

            marker = "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->"
            if marker in content:
                content = content.replace(marker, f"{marker}\n\n{entry_text}\n")
            else:
                content += f"\n\n{entry_text}\n"

            with open(config.decisions_file, "w", encoding="utf-8") as f:
                f.write(content)
        print(f"Appended synthesized decision to decisions.md buffer ✓")
    except Exception as e:
        print(f"Failed to append captured entry: {str(e)}")


def _retired_handle(store: GraphStore, slug: str) -> Optional[Dict[str, Any]]:
    """Builds a retired-handle pointer for a superseded-filtered ranked match.

    A match dropped by the active-view filter (``get_node_by_slug`` → ``None`` or a
    non-``active``/``drifted`` computed state) is not noise — it is a genuine retired
    handle the agent can chase (V1-D16: a vector-store slug always resolves to *some*
    node; nodes are never deleted). This returns ``{"slug", "state"}`` — and, when the
    graph knows it, the live successor under ``superseded_by`` — so the blackout vector
    hands the agent a pointer, not a payload. The state is read authoritatively from the
    *computed* ``get_node_state`` (the vector payload's ``state`` is stale-at-embed-time
    and absent under test), via the state-agnostic ``resolve_slug``.

    Calm degradation (P9): if the slug fails to resolve at all, returns ``None`` (the
    caller omits it) rather than crash; if the state read fails, falls back to
    ``"superseded"``.

    Args:
        store: The graph store to resolve the slug and read state/modifiers from.
        slug: The slug of the superseded-filtered match.

    Returns:
        The retired-handle dict, or ``None`` if the slug does not resolve.
    """
    try:
        node_ids = store.resolve_slug(slug)
    except Exception:
        return None
    if not node_ids:
        return None
    node_id = node_ids[0]
    try:
        state = store.get_node_state(node_id)
    except Exception:
        state = "superseded"
    handle: Dict[str, Any] = {"slug": slug, "state": state}
    try:
        successors = store.get_modifiers(node_id).get("superseded_by")
        if successors:
            handle["superseded_by"] = successors
    except Exception:
        pass
    return handle


def _emit_lexical_degraded(config: MitosConfig, query: str, *, reason: str,
                           store: Optional[GraphStore], as_json: bool,
                           brief: bool, limit: Optional[int],
                           open_questions: Optional[List[Dict[str, Any]]] = None) -> None:
    """Runs the deterministic lexical fallback and renders it on the CLI.

    The shared degraded exit for ``surface``/``query`` (ADR
    ``read-verbs-degrade-to-lexical-decisions-md-fallback``): one calm header
    naming the cause, then a term-match over decisions.md — never the raw
    provider blob, never the clean-empty header. Exit code stays 0 (deliberate:
    the JSON ``degraded`` marker + changed header already disambiguate).

    Args:
        config: The active workspace configuration (supplies decisions.md path).
        query: The claim/topic the caller was trying to recall.
        reason: One-line cause phrase (see ``degraded_reason_from_error``).
        store: A readable graph store for active-filtering + modifier stamps,
            or None when the graph itself is down (pre-V1a).
        as_json: Emit the degraded JSON envelope instead of text.
        brief: Omit ``rejected_paths`` from each match.
        limit: Max matches; None ⇒ the lexical default.
        open_questions: An already-computed scoped parked-OQ list to carry on
            the envelope (present-if-scanned semantics — None means omitted).
    """
    envelope = lexical_fallback(
        query, config.decisions_file, reason=reason, store=store,
        limit=limit, brief=brief,
    )
    envelope["query"] = query
    envelope.update(corpus_provenance(config))
    if open_questions is not None:
        envelope["open_questions"] = open_questions
    if as_json:
        _emit_json(envelope)
        return
    print(provenance_line(config))
    print(envelope["note"])
    for i, d in enumerate(envelope["matches"], start=1):
        print(f"{i}. {d['slug']}")
        print(f"   Decided:  {d['axiom']}")
        marker = _modifier_marker(d)
        if marker:
            print(f"   {marker}")
        if "rejected_paths" in d:
            print(f"   Rejected: {d['rejected_paths']}")
        if d["scope"]:
            print(f"   Scope:    {', '.join(d['scope'])}")
        print()
    for oq in envelope.get("open_questions", []):
        print(f"[open question in scope] {oq['topic']}")


def cmd_query(config: MitosConfig, query_text: str, depth: str = "letter",
              as_json: bool = False, brief: bool = False,
              limit: Optional[int] = None) -> None:
    """Queries the vector store semantically for similar decisions — the CLI twin
    of the MCP ``query_decisions`` tool's *ranked* branch.

    Brings the CLI verb up to its MCP twin's bar: it filters superseded matches
    (state not in ``active``/``drifted``), carries a modifier-stamped,
    Letter-complete per-match payload (``core_axiom`` + ``rejected_paths`` fence)
    built via the shared :func:`letter_payload`, and emits either text or, with
    ``as_json``, the same ranked envelope ``query_decisions`` returns. The text
    render is a *renderer over the same payload list* the ``--json`` path emits, so
    the two can never disagree on what was filtered or stamped (kernel M5 + M3).

    Unlike its MCP twin, the CLI verb stays semantic-only — there is no exact-slug
    dereference branch (that is ``show``'s job, ADR
    ``cli-query-stays-semantic-not-dereference-twin``).

    Args:
        config: The active workspace configuration.
        query_text: The assertion or subsystem claim to find precedents for.
        depth: The retrieval depth; v0.1 enforces ``letter``.
        as_json: Emit the machine-readable ranked JSON envelope instead of text.
        brief: Omit ``rejected_paths`` (axiom-only) — never sheds a modifier stamp.
        limit: Ranked top-k to retrieve; ``None`` ⇒ the default 5. SETS the count
            (raises or lowers it), clamped to ``[1, RANKED_LIMIT_CEILING]`` — not a
            ``min(default, N)`` truncation.
    """
    if depth != "letter":
        msg = f"Depth mode '{depth}' is not yet implemented in v0.1 (Letter-only retrieval)."
        if as_json:
            _emit_json({"error": msg}, indent=None)
            return
        raise ValueError(msg)

    # A pre-V1a graph raises at store construction — the SQLite graph is unusable,
    # so the fallback parses decisions.md directly and must not touch the graph.
    try:
        manager = MitosSyncManager(config)
    except Exception as e:
        _emit_lexical_degraded(
            config, query_text, reason=degraded_reason_from_error(e),
            store=None, as_json=as_json, brief=brief, limit=limit,
        )
        return

    if not manager.embed_provider or not manager.vector_store:
        _emit_lexical_degraded(
            config, query_text, reason=degraded_reason_from_error(None),
            store=manager.store, as_json=as_json, brief=brief, limit=limit,
        )
        return

    store = manager.store
    top_k = clamp_limit(limit)
    try:
        q_vector = manager.embed_provider.get_embedding(query_text, is_query=True)
        raw_matches = manager.vector_store.query(q_vector, limit=top_k)

        # Filter superseded first, then stamp + Letter — mirrors the ranked loop in
        # mcp_server.query_decisions byte-for-byte (T4 parity). A superseded-not-reused
        # slug is dropped at the active-view get_node_by_slug → None step, closing the
        # M3 leak where a superseded node would otherwise read as live. Each dropped
        # match is a retired handle the blackout vector points the agent at.
        matches = []
        retired: List[Dict[str, Any]] = []
        for m in raw_matches:
            node = store.get_node_by_slug(m["slug"])
            if not node:
                handle = _retired_handle(store, m["slug"])
                if handle:
                    retired.append(handle)
                continue
            node_state = store.get_node_state(node["id"])
            if node_state not in ("active", "drifted"):
                handle = _retired_handle(store, m["slug"])
                if handle:
                    retired.append(handle)
                continue
            match = letter_payload(
                node,
                brief=brief,
                extras={"state": node_state, "score": m["score"], "depth_mode": "letter"},
            )
            match.update(store.get_modifiers(node["id"]))
            matches.append(match)
    except Exception as e:
        # Embedding/Qdrant failure mid-query (e.g. a 429): never the raw
        # provider blob — one calm cause line + the deterministic fallback.
        _emit_lexical_degraded(
            config, query_text, reason=degraded_reason_from_error(e),
            store=store, as_json=as_json, brief=brief, limit=limit,
        )
        return

    # Blackout: retrieval returned matches but every one was superseded-filtered
    # (displayed == 0, retrieved > 0). That is NOT a true miss — surfacing it as one
    # makes the agent assume novelty and re-derive a settled contradiction. Emit the
    # retired handles + a distinct note instead. `retired` is non-empty only when the
    # filter dropped something, so `not matches and retired` is exactly the blackout.
    blackout = not matches and bool(retired)

    # Build the per-match list once, then branch the two renderings over it.
    if as_json:
        envelope: Dict[str, Any] = {"query": query_text, "depth_mode": "letter", "matches": matches}
        envelope.update(corpus_provenance(config))
        if blackout:
            envelope["all_superseded"] = retired
        _emit_json(envelope)
        return

    if blackout:
        print(blackout_note(retired))
        return

    # Genuine miss — nothing was retrieved (or nothing resolved). The provenance
    # line disambiguates "no precedent" from "wrong workspace" — the miss is
    # exactly where that ambiguity bites.
    if not matches:
        print(provenance_line(config))
        print("No matching decisions found.")
        return

    print(f"\nQuery matches for: '{query_text}'  [{provenance_line(config)}]")
    print("-" * 60)
    for i, d in enumerate(matches, start=1):
        print(f"{i}. {d['slug']}  (score {d['score']:.3f})")
        print(f"   Decided:  {d['axiom']}")
        marker = _modifier_marker(d)
        if marker:
            print(f"   {marker}")
        if "rejected_paths" in d:
            print(f"   Rejected: {d['rejected_paths']}")
        if d["scope"]:
            print(f"   Scope:    {', '.join(d['scope'])}")
        print()


def cmd_show(config: MitosConfig, ident: str, as_json: bool = False) -> None:
    """Shows full details of a specific node by ID or slug.

    Dereferences a single handle state-agnostically via ``GraphStore.resolve_handle``
    — active-first, else the most-recent superseded node in the casefolded-slug
    lineage (marked superseded) — so a moved-on node still answers to its own slug
    instead of 404-ing. Only a genuinely-absent identifier reaches the not-found
    branch, whose static, hedged ``mitos sync`` pointer reads no buffer (truthful for
    both a typo and an authored-but-unsynced draft). With ``as_json`` it emits a
    Letter-complete, modifier-stamped JSON object (the not-found case a JSON object
    too, never a bare text print).

    Args:
        config: The active workspace config (supplies the graph db path).
        ident: A content-hash id or a slug (case-insensitive).
        as_json: When True, emit a machine-readable JSON object instead of text.

    Returns:
        None.
    """
    store = GraphStore(config.db_path)

    # State-agnostic resolution: id → active slug → most-recent in lineage → None.
    # The one seam 5b's `show_node` reuses, so resolution parity is structural.
    node = store.resolve_handle(ident)

    if not node:
        # Genuine absence: a typo, or an authored-but-unsynced draft. The hint is
        # static and hedged — it reads no buffer, never asserts presence to a typo.
        # Single-sourced in display.py so the `show_node` MCP twin emits the same
        # not-found object byte-for-byte (parity is structural).
        hint = SHOW_NOT_FOUND_HINT
        if as_json:
            _emit_json({"found": False, "ident": ident, "hint": hint})
            return
        print(f"Node with ID or Slug '{ident}' not found — {hint}.")
        return

    # Compute current active/superseded state (single-node V1a derivation, 8a)
    state = store.get_node_state(node["id"])

    # One stamp source for both the text and the --json branch (kind-agnostic — an OQ
    # carries only amended_by/narrowed_by). A superseded show that omits its modifier
    # keys reads as the final word ("amended axioms read as live" trap), and surfacing
    # superseded nodes is exactly this verb's new job — so stamping is load-bearing.
    modifiers = store.get_modifiers(node["id"])

    if as_json:
        # The dereference payload shape is single-sourced in display.show_payload
        # so the `show_node` MCP twin produces a byte-identical dict (parity is
        # structural, not test-enforced). The 5a --json regression pins prove this
        # extraction is byte-identical to the prior inline builder.
        payload = show_payload(node, state=state, modifiers=modifiers)
        _emit_json(payload)
        return

    print(f"\n[{node['kind'].upper()}] {node['slug']}")
    print(f"ID:           {node['id']}")
    print(f"State:        {state}")
    for key in MODIFIER_EDGE_KEYS.values():
        if modifiers.get(key):
            print(f"{(key.replace('_', ' ').capitalize() + ':'):14}{', '.join(modifiers[key])}")
    if node.get("date"):
        print(f"Date:         {node['date']}")
    if node.get("title"):
        print(f"Title:        {node['title']}")
        
    if node["kind"] == "decision":
        print(f"Decided:      {node['core_axiom']}")
        print(f"Rejected:     {node['rejected_paths']}")
        print(f"Mechanisms:   {', '.join(node['mechanisms'])}")
        print(f"Scope:        {', '.join(node['scope'])}")
        if node.get("invalidates_if"):
            print(f"Invalidates:  {node['invalidates_if']}")
        if node.get("context"):
            print(f"Context:      {node['context']}")
    else:
        print(f"Park Reason:  {node.get('park_reason') or 'None'}")
        print("Questions Raised:")
        for q in node["questions_raised"]:
            print(f"  - {q}")

    if node.get("transcript"):
        print("\n[Transcript]")
        print(node["transcript"])
    print()


def cmd_list(config: MitosConfig, scope: Optional[str] = None,
             state_filter: Optional[str] = None, as_json: bool = False,
             brief: bool = False, oneline: bool = False) -> None:
    """Enumerates the complete set of decisions (+ parked open questions) for a scope.

    The CLI twin of the MCP ``list_decisions`` tool — the exhaustive, deterministic
    counterpart to the ranked, capped ``surface``/``query`` recall path. Use it for a
    completeness pass: every settled call in a scope, nothing hidden below a relevance
    cliff. Needs no API key or Qdrant (it is a pure graph read).

    Args:
        config: The active workspace configuration.
        scope: Optional scope tag filter; omit for the whole project.
        state_filter: ``"active"`` (the default view) = the live set (active +
            drifted); ``"all"`` = every decision regardless of state; any other value
            = an exact computed-state match (e.g. "superseded").
        as_json: Emit a machine-readable JSON report (for agents) instead of text.
        brief: Axiom-only (omit ``rejected_paths``) — the M4 opt-out. Mutually
            exclusive with ``oneline`` (argparse enforces it on the CLI surface).
        oneline: One row per decision (slug + word-boundary-truncated axiom) — the
            orientation/table-of-contents tier below ``brief`` for big scopes.
            Modifier markers survive (stamps ride every thinner tier).
    """
    store = GraphStore(config.db_path)
    # Default the view to the live set; an absent filter must not dump superseded
    # decisions into what an agent reads as a completeness pass.
    effective_state = state_filter or "active"
    decisions = store.get_decisions(scope=scope, state=effective_state)
    modifiers = store.get_modifiers_map([d["id"] for d in decisions])
    parked = [oq for oq in store.get_open_questions(scope=scope)
              if oq["state"] == "parked"]

    def _list_item(d):
        # The oneline tier swaps the Letter core for the minimal {slug,
        # axiom_oneline, state} object; modifier stamps ride either shape
        # (stamps survive every thinner tier).
        if oneline:
            item = oneline_payload(d)
        else:
            item = letter_payload(d, brief=brief, extras={"state": d["computed_state"]})
        item.update(modifiers.get(d["id"], {}))
        return item

    # On an empty scoped read, distinguish a genuinely-fresh scope from a misspelled
    # one: an absent-from-live scope gets the same bounded self-correction vector the
    # surface verbs use (3d). Computed once, before the as_json split, so the text and
    # JSON emit points don't drift — and only on the miss path (guarded on emptiness),
    # so the hot non-empty path never pays the get_scope_counts() read. The recovery
    # payload carries no node id, so there is nothing to modifier-stamp here.
    recovery = None
    if scope and not decisions and not parked:
        scope_counts: Optional[Dict[str, Dict[str, int]]] = None
        try:
            scope_counts = order_scope_counts(store.get_scope_counts())
        except Exception:
            pass
        recovery = scope_filter_recovery(
            scope=scope, scope_counts=scope_counts, surface="cli"
        )

    if as_json:
        payload = {
            "decisions": [_list_item(d) for d in decisions],
            "open_questions": [_oq_payload(oq) for oq in parked],
            "total": len(decisions),
            "scope": scope,
            "state": effective_state,
            **corpus_provenance(config),
        }
        if recovery:
            payload["scope_known"] = False
            payload["scope_recovery"] = recovery["note"]
        _emit_json(payload)
        return

    if not decisions and not parked:
        print(provenance_line(config))
        if not store.get_all_nodes():
            # Empty-graph precedence wins over the unused-scope vector: a graph with no
            # nodes has an empty vocabulary, but "run sync" is the truer nudge.
            print("Graph database is empty. Run 'mitos sync' to ingest entries.")
        elif recovery:
            print(recovery["note"])
        else:
            print("No decisions match the given filters.")
        return

    scope_note = f"  (scope: {scope})" if scope else ""
    print(f"\nDecisions ({len(decisions)} found, state={effective_state}){scope_note}  "
          f"[{provenance_line(config)}]:")
    print("-" * 80)
    for d in decisions:
        marker = _modifier_marker(modifiers.get(d["id"], {}))
        if oneline:
            # One row per decision: slug + word-boundary-truncated axiom (the
            # orientation tier); a compact modifier marker rides the same row.
            row = f"{d['slug']}  {oneline_axiom(d)}"
            print(f"{row}  {marker}" if marker else row)
            continue
        scopes = f"[{', '.join(d['scope'])}]" if d["scope"] else ""
        print(f"{d['computed_state']:11} | {d['slug']:30} {scopes}")
        axiom_snip = truncate_words(d.get("core_axiom", ""), 66)
        if axiom_snip:
            print(f"              {axiom_snip}")
        if marker:
            print(f"              {marker}")
    if parked:
        print(f"\nParked open questions ({len(parked)}):")
        for oq in parked:
            print(f"  ? {oq['slug']}")
            marker = _modifier_marker(oq)
            if marker:
                print(f"        {marker}")
    print()


def cmd_open_questions(config: MitosConfig, scope: Optional[str] = None,
                       as_json: bool = False) -> None:
    """Lists all parked open questions.

    Args:
        config: The active workspace configuration.
        scope: Optional scope tag filter; omit for the whole project.
        as_json: Emit a machine-readable JSON map (the parked OQ set, each carrying
            its ``amended_by``/``narrowed_by`` modifier subset) instead of text.
    """
    store = GraphStore(config.db_path)
    oqs = store.get_open_questions(scope=scope)

    parked = [q for q in oqs if q["state"] == "parked"]

    # On an empty scoped read, an absent-from-live scope gets the bounded self-correction
    # vector (3d) instead of a silent "zero parked" line. Only the miss path pays the
    # get_scope_counts() read. No empty-graph precedence here (CLI asymmetry vs cmd_list):
    # on an empty graph a scoped OQ read trips the vector whose static `mitos sync` hedge
    # already covers the "just authored" case. The payload carries no node id — nothing
    # to modifier-stamp.
    recovery = None
    if scope and not parked:
        scope_counts: Optional[Dict[str, Dict[str, int]]] = None
        try:
            scope_counts = order_scope_counts(store.get_scope_counts())
        except Exception:
            pass
        recovery = scope_filter_recovery(
            scope=scope, scope_counts=scope_counts, surface="cli"
        )

    if as_json:
        # Honest-empty envelope on an empty/unmatched scope (never an error — empty is
        # first-class). An absent-from-live scope rides the additive recovery fields (3d).
        payload = {
            "open_questions": [_oq_payload(q) for q in parked],
            "total": len(parked),
            "scope": scope,
        }
        if recovery:
            payload["scope_known"] = False
            payload["scope_recovery"] = recovery["note"]
        _emit_json(payload)
        return

    if not parked:
        if recovery:
            print(recovery["note"])
        else:
            print("Zero parked open questions found.")
        return

    print(f"\nParked Open Questions ({len(parked)} found):")
    print("-" * 80)
    for q in parked:
        reason = f"({q['park_reason']})" if q.get("park_reason") else ""
        print(f"Topic: {q['slug']} {reason}")
        for question in q["questions_raised"]:
            print(f"  - {question}")
        marker = _modifier_marker(q)
        if marker:
            print(f"  {marker}")
    print()


def cmd_scopes(config: MitosConfig, as_json: bool = False, archived: bool = False) -> None:
    """Enumerates the scope-tag vocabulary with each domain's live-node counts.

    The discovery surface for the project's scope vocabulary — the CLI twin of the
    MCP ``list_scopes`` tool. An agent landing in a project can already *record*
    into a scope and *recall* from one, but this is how it *sees the map*: every
    scope tag that carries a live node, ranked busiest-domain-first (total active
    decisions + parked open questions, descending; ties alphabetical), so the
    domains that matter most read first. Use it before recording or recalling, to
    learn the project's vocabulary instead of guessing it. A pure graph read — no
    API key or Qdrant needed.

    This returns a tag→counts *aggregate*, not a decision payload: there is no node
    ``id`` to stamp, so the "every decision-read surface stamps modifiers" rule does
    **not** apply here (no modifier seam — that is correct, not a missing stamp).

    Args:
        config: The active workspace configuration.
        as_json: Emit the machine-readable ordered ``{scope: {active_decisions,
            parked_open_questions}}`` map (for agents) instead of the text table.
        archived: Include fully-dead domains (every scope present in the graph at a
            ``0/0`` floor) — the scope-level parallel of ``list --state all``.
            Omit for the live vocabulary only.

    Returns:
        None.
    """
    store = GraphStore(config.db_path)
    counts = order_scope_counts(store.get_scope_counts(include_archived=archived))

    if as_json:
        _emit_json(counts)
        return

    if not counts:
        # Empty/fresh is first-class: an empty vocabulary IS the healthy empty state,
        # never an error. A just-initialised project simply has no scopes yet.
        print("No scopes yet — record a decision with --scope to start the vocabulary.")
        return

    name_w = max(len("scope"), max(len(s) for s in counts))
    print(f"\nScopes ({len(counts)} found, busiest first):")
    print("-" * (name_w + 30))
    print(f"{'scope':{name_w}}   {'active':>6}  {'parked':>6}  {'total':>6}")
    for scope, c in counts.items():
        active = c["active_decisions"]
        parked = c["parked_open_questions"]
        print(f"{scope:{name_w}}   {active:>6}  {parked:>6}  {active + parked:>6}")
    print()


def cmd_import(config: MitosConfig, filepath: str, use_llm_extract: bool = False) -> None:
    """Imports legacy prose ADR file."""
    importer = MitosProseImporter(config)
    importer.import_from_file(filepath, use_llm_extract)


def cmd_render(config: MitosConfig, scope: Optional[str] = None, render_format: str = "live-axioms") -> None:
    """Statelessly regenerates live_axioms.md and scope axioms."""
    if render_format != "live-axioms":
        print(f"Warning: format '{render_format}' is not supported in v0.1. Falling back to live-axioms.")
    store = GraphStore(config.db_path)
    renderer = MitosRenderer(config.workspace_dir)
    rendered = renderer.render_all(store, scope)
    print("Render complete. Generated files:")
    for path in rendered:
        print(f"  - {path}")


def cmd_record(
    config: MitosConfig,
    axiom: str,
    rejected: str,
    scope: Optional[List[str]] = None,
    mechanisms: Optional[List[str]] = None,
    context: Optional[str] = None,
    supersedes: Optional[str] = None,
    corrects: Optional[str] = None,
    amends: Optional[str] = None,
    narrows: Optional[str] = None,
    depends_on: Optional[str] = None,
    resolves: Optional[str] = None,
    contradicts: Optional[str] = None,
    derives_from: Optional[str] = None,
    cites: Optional[str] = None,
    *,
    slug: str,
    acknowledge_neighbors: bool = False,
    as_json: bool = False,
) -> None:
    """Records a decision directly to the write-buffer and graph (thin wrapper).

    Under ``as_json``, every outcome — created/exists, the ``needs_review`` pause, and
    error — is emitted as the raw ``record_decision_entry`` receipt dict (the same shape
    the MCP ``record_decision`` tool serializes) on **stdout**, never a stderr wall a
    ``--json`` consumer would miss. The existing exit codes are preserved (0
    created/exists, 2 needs_review, 1 error): exit code is the shell's signal, the JSON
    object is the agent's.
    """
    manager = MitosSyncManager(config)
    result = manager.record_decision_entry(
        axiom=axiom,
        rejected_paths=rejected,
        scope=scope or [],
        mechanisms=mechanisms,
        context=context,
        supersedes=supersedes,
        corrects=corrects,
        amends=amends,
        narrows=narrows,
        depends_on=depends_on,
        resolves=resolves,
        contradicts=contradicts,
        derives_from=derives_from,
        cites=cites,
        slug=slug,
        acknowledge_neighbors=acknowledge_neighbors,
    )

    if as_json:
        # Every outcome speaks JSON on stdout (no stderr walls); exit codes ride along.
        # The receipt is already the structured dict — emit it verbatim, no reshaping
        # (the record receipt is a write result, NOT a decision read: no modifier
        # stamping; its related/neighbors are recall pointers the agent dereferences
        # by slug). scope_overflow, when present, is already inside `result`.
        _emit_json(result)
        if "error" in result:
            sys.exit(1)
        if result.get("status") == "needs_review":
            sys.exit(2)
        return

    if "error" in result:
        print(f"Record failed [{result['code']}]: {result['error']}", file=sys.stderr)
        sys.exit(1)

    if result.get("status") == "needs_review":
        # P4 pause — nothing was written. Show the neighbours and how to proceed.
        print(f"⚠ Paused — '{result['slug']}' looks like an existing decision. Nothing written.",
              file=sys.stderr)
        for n in result.get("neighbors", []):
            score = n.get("score")
            score_s = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
            tension = "  [possible tension]" if n.get("possible_tension") else ""
            print(f"  ↔ {n['slug']}  ({score_s}){tension}  "
                  f"{truncate_words(n.get('axiom') or '', 60)}",
                  file=sys.stderr)
        print("  → Re-record with --supersedes/--amends/--contradicts/--cites <slug> to link "
              "it, or --acknowledge-neighbors to record as independent.", file=sys.stderr)
        sys.exit(2)

    print(f"Recorded decision '{result['slug']}' ({result['status']}) ✓")
    print(f"  ID:        {result['id']}")
    print(f"  State:     {result['state']}")
    print(f"  Embedding: {result['embedding']}")
    if result.get("path"):
        print(f"  Written:   {result['path']}  (the human-readable entry — eyeball it)")
    print(f"  Handle:    '{result['slug']}' — pass this to --supersedes/--amends/--depends-on/… to link future decisions.")
    # Write facts read back from the committed node (NOT an echo of the flags):
    # the edges the commit actually wired, and scope/mechanisms as stored. Lines
    # are omitted when empty — a bare decision keeps a bare receipt.
    edges = result.get("edges_created")
    if edges:
        edges_s = ", ".join(f"{e['kind']} → {e['target']}" for e in edges)
        print(f"  Edges:     {edges_s}")
    if result.get("scope"):
        print(f"  Scope:     {', '.join(result['scope'])}")
    if result.get("mechanisms"):
        print(f"  Mechanisms: {', '.join(result['mechanisms'])}")
    related = result.get("related")
    if related:
        print("  ↔ Nearest existing decisions (an intended neighbour, or a tension to reconcile?):")
        for r in related:
            score = r.get("score")
            score_s = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
            axiom_snip = truncate_words(r.get("axiom") or "", 60)
            print(f"     - {r['slug']}  ({score_s})  {axiom_snip}")
    # Debounced size-ceiling nudge — AFTER the receipt, on stderr (an ancillary health
    # hint, never the receipt itself), so a healthy growing corpus can't bury "Recorded ✓".
    # Flush stdout first so the receipt lands before the nudge even when stdout is piped
    # (block-buffered) while stderr is unbuffered — otherwise the streams can interleave.
    overflow = result.get("scope_overflow")
    if overflow:
        sys.stdout.flush()
        print(f"\n{overflow}", file=sys.stderr)


def _read_text_arg(inline: Optional[str], file_path: Optional[str]) -> Optional[str]:
    """Resolves a text argument from an inline value or a file.

    Lets agents pass multi-sentence prose without fighting shell quoting: a
    ``--*-file`` path (or ``-`` for stdin) sidesteps apostrophe/quote escaping
    that would otherwise force the prose to be degraded to satisfy bash.

    Args:
        inline: The value passed directly on the command line, if any.
        file_path: A file path to read instead, or ``"-"`` for stdin.

    Returns:
        The resolved text, or None if neither source was provided.
    """
    if file_path is not None:
        if file_path == "-":
            return sys.stdin.read()
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return inline


def cmd_surface(config: MitosConfig, query: str, scope: Optional[str] = None,
                as_json: bool = False, brief: bool = False,
                limit: Optional[int] = None) -> None:
    """Surfaces active decisions relevant to a query — the CLI twin of the MCP
    ``surface_decisions`` tool (the precedent-recall half of Mitos).

    Mirrors ``mcp_server.surface_decisions`` so a CLI-only agent (or a human) can
    run the recall loop without the MCP wired. The semantic match is scope-blind;
    ``scope`` only narrows the parked open questions and the recall note (plus the
    degraded fallback when semantic recall is down). For scope-RESTRICTED retrieval
    use ``mitos list --scope`` — the only surface that hard-filters by scope. (Both
    surfaces return full ``rejected_paths``; pass ``--brief`` for a lighter scan.)

    Args:
        config: The active workspace configuration.
        query: The claim or topic to find precedents for.
        scope: Optional scope hint — does NOT filter the semantic search; scopes the
            open-questions scan and recall note only. Use ``mitos list --scope`` to
            hard-filter by scope.
        as_json: Emit a machine-readable JSON report (for agents) instead of text.
        brief: Omit ``rejected_paths`` (axiom-only — a quick "anything nearby?" scan).
        limit: Ranked top-k to retrieve; ``None`` ⇒ the default 5. SETS the count,
            clamped to ``[1, RANKED_LIMIT_CEILING]`` — not a ``min(default, N)`` clamp.
    """
    # A pre-V1a graph raises at store construction — the graph is unusable, so
    # the lexical fallback parses decisions.md directly (no graph access).
    try:
        manager = MitosSyncManager(config)
    except Exception as e:
        _emit_lexical_degraded(
            config, query, reason=degraded_reason_from_error(e),
            store=None, as_json=as_json, brief=brief, limit=limit,
        )
        return

    store = manager.store
    top_k = clamp_limit(limit)

    def _shape(node, score):
        d = letter_payload(node, brief=brief, extras={"score": score})
        d.update(store.get_modifiers(node["id"]))
        return d

    results: Dict[str, Any] = {"active_decisions": []}
    results.update(corpus_provenance(config))
    semantic_ran = False
    top_score: Optional[float] = None
    retired: List[Dict[str, Any]] = []
    degraded_error: Optional[Exception] = None

    if manager.embed_provider and manager.vector_store:
        try:
            q_vector = manager.embed_provider.get_embedding(query, is_query=True)
            matches = manager.vector_store.query(q_vector, limit=top_k)
            semantic_ran = True
            for m in matches:
                node = store.get_node_by_slug(m["slug"])
                if not node:
                    handle = _retired_handle(store, m["slug"])
                    if handle:
                        retired.append(handle)
                    continue
                state = store.get_node_state(node["id"])
                if state not in ("active", "drifted"):
                    handle = _retired_handle(store, m["slug"])
                    if handle:
                        retired.append(handle)
                    continue
                results["active_decisions"].append(_shape(node, m["score"]))
                if top_score is None or m["score"] > top_score:
                    top_score = m["score"]
        except Exception as e:
            semantic_ran = False
            degraded_error = e

    # Scope listing fallback ONLY in degraded mode (mirrors the MCP tool, P5): a
    # semantic run that found nothing must not masquerade as an unranked scope dump.
    if not semantic_ran and not results["active_decisions"] and scope:
        try:
            for d in store.get_active_decisions(scope=scope)[:5]:
                results["active_decisions"].append(_shape(d, 1.0))
        except Exception:
            pass

    # Open questions only when a scope was given (absent = not scanned, [] = none here).
    if scope:
        open_questions = []
        try:
            for oq in store.get_open_questions(scope=scope):
                if oq["state"] == "parked":
                    open_questions.append({
                        "topic": oq["slug"], "questions_raised": oq["questions_raised"],
                        "park_reason": oq.get("park_reason"), **_oq_modifiers(oq),
                    })
        except Exception:
            pass
        results["open_questions"] = open_questions

    # Degraded and empty-handed on decisions: route into the deterministic
    # lexical fallback (ADR read-verbs-degrade-to-lexical-decisions-md-fallback)
    # instead of the self-contradicting "No active precedents found" +
    # unavailable note. The scoped parked-OQ scan (a pure graph read that
    # survived) rides along on the degraded output.
    if not semantic_ran and not results["active_decisions"]:
        _emit_lexical_degraded(
            config, query, reason=degraded_reason_from_error(degraded_error),
            store=store, as_json=as_json, brief=brief, limit=limit,
            open_questions=results.get("open_questions"),
        )
        return

    # Confidence signal — distinguish a settled precedent from loose neighbours / no
    # match (AX P5). Shared policy with the MCP tool via mitos.recall. The live
    # scope-count map (busiest-first) is the unused-scope oracle + did-you-mean / top-K
    # source; calm-degrade to None on error.
    scope_counts: Optional[Dict[str, Dict[str, int]]] = None
    if scope:
        try:
            scope_counts = order_scope_counts(store.get_scope_counts())
        except Exception:
            pass
    confidence, note = assess_surface_recall(
        semantic_ran=semantic_ran,
        top_score=top_score,
        result_count=len(results["active_decisions"]),
        scope=scope,
        scope_counts=scope_counts,
        surface="cli",
    )
    if confidence is not None:
        results["confidence"] = confidence
    results["note"] = note

    # Blackout: semantic ranking ran and retrieved precedents, but every one was
    # superseded-filtered (no active match). Override the note with the recovery
    # vector and attach the retired handles — distinct from a true miss (where
    # `retired` is empty). Fires regardless of any parked open questions (the
    # all_superseded vector must not be suppressed by a non-empty open_questions).
    blackout = semantic_ran and not results["active_decisions"] and bool(retired)
    if blackout:
        results["note"] = blackout_note(retired)
        results["all_superseded"] = retired
        note = results["note"]

    if as_json:
        _emit_json(results)
        return

    ad, oqs = results["active_decisions"], results.get("open_questions", [])
    conf = results.get("confidence")
    if not ad and not oqs:
        # The clean-empty header asserts "checked, none found" — it must never
        # co-occur with a degraded/unavailable note ("couldn't check"). With the
        # lexical fallback routing above this branch only fires when semantic
        # ran (confidence is set); the guard is belt-and-braces against any
        # future path that reaches here degraded.
        if conf is not None:
            scope_note = f" (scope: {scope})" if scope else ""
            print(provenance_line(config))
            print(f"No active precedents found for: '{query}'{scope_note}")
        print(f"→ {note}")
        return
    print(f"\nPrecedents for: '{query}'" + (f"  (scope: {scope})" if scope else "")
          + f"  [{provenance_line(config)}]")
    if conf == "weak":
        print("⚠ confidence: weak — twilight zone: matches are close but may not settle this.")
    elif conf == "none":
        print("⚠ confidence: likely off-axis — the scope is populated, but nothing matches your query.")
    print("-" * 60)
    for i, d in enumerate(ad, start=1):
        print(f"{i}. {d['slug']}  (score {d['score']:.3f})")
        print(f"   Decided:  {d['axiom']}")
        marker = _modifier_marker(d)
        if marker:
            print(f"   {marker}")
        if "rejected_paths" in d:
            print(f"   Rejected: {d['rejected_paths']}")
        if d["scope"]:
            print(f"   Scope:    {', '.join(d['scope'])}")
        print()
    for oq in oqs:
        print(f"[open question in scope] {oq['topic']}")
        marker = _modifier_marker(oq)
        if marker:
            print(f"   {marker}")
    print(f"\n→ {note}")


def cmd_serve() -> None:
    """Starts the FastMCP server over standard stdio."""
    # Importing mcp instance from mcp_server inside the function prevents early execution issues
    from mitos.mcp_server import mcp
    print("Starting Mitos MCP Server on stdio ...")
    mcp.run()


def _check_qdrant(qdrant_url: str, collection: str) -> Dict[str, Any]:
    """Probes Qdrant reachability and the project's collection (best-effort).

    Args:
        qdrant_url: The configured Qdrant REST endpoint.
        collection: The project's collection name.

    Returns:
        ``{reachable, collection_exists, points}`` — ``collection_exists`` and
        ``points`` are ``None`` when Qdrant is unreachable.
    """
    import requests
    out: Dict[str, Any] = {"reachable": False, "collection_exists": None, "points": None}
    try:
        r = requests.get(
            f"{qdrant_url.rstrip('/')}/collections/{collection}", timeout=3
        )
        out["reachable"] = True
        if r.status_code == 200:
            out["collection_exists"] = True
            out["points"] = r.json().get("result", {}).get("points_count")
        elif r.status_code == 404:
            out["collection_exists"] = False
    except Exception:
        pass
    return out


def _env_file_has_key(env_path: str, name: str) -> bool:
    """True if ``env_path`` assigns ``name`` a non-empty value on ANY line.

    Skips empty assignments (the scaffolded ``GEMINI_API_KEY=`` slot) and keeps
    scanning, so a key added on a later line is still found — matching
    ``load_dotenv_file``'s "first non-empty value wins" semantics.
    """
    if not os.path.exists(env_path):
        return False
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{name}="):
                    if line.split("=", 1)[1].strip().strip('"').strip("'"):
                        return True
    except OSError:
        pass
    return False


def _gemini_key_source(workspace_dir: str) -> Optional[str]:
    """Reports where GEMINI_API_KEY comes from, in precedence order.

    Files are checked before the live environment so the report attributes the
    key to its durable home (``main()`` also loads both files into the
    environment, which would otherwise mask the distinction).

    Args:
        workspace_dir: The project directory to inspect.

    Returns:
        ``"project .env"``, ``"global .env"``, ``"environment"``, or None.
    """
    if _env_file_has_key(os.path.join(workspace_dir, ".env"), "GEMINI_API_KEY"):
        return "project .env"
    if _env_file_has_key(global_env_path(), "GEMINI_API_KEY"):
        return "global .env"
    if os.environ.get("GEMINI_API_KEY"):
        return "environment"
    return None


def _gemini_key_present(workspace_dir: str) -> bool:
    """True if GEMINI_API_KEY is available — env, project .env, or global .env."""
    return _gemini_key_source(workspace_dir) is not None


def _mcp_wired(workspace_dir: str) -> bool:
    """True if a project-scoped ``.mcp.json`` wires the mitos MCP server.

    This is the Claude Code convention (a ``mitos`` entry under ``mcpServers``).
    It's a *recommendation* signal for agents, never a readiness blocker — other
    harnesses wire the MCP elsewhere, and humans don't need it at all.
    """
    import json as _json
    path = os.path.join(workspace_dir, ".mcp.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        return "mitos" in (data.get("mcpServers") or {})
    except (OSError, ValueError, AttributeError):
        return False


# The decision-loop verbs (+ their MCP-name aliases). Only these get the
# "consider wiring the MCP" nudge — setup/ops/inspection verbs are CLI-native,
# and `serve` IS the MCP, so nudging there would be nonsense.
_DECISION_LOOP_COMMANDS = frozenset({
    "record", "record_decision", "surface", "surface_decisions", "query", "query_decisions",
    "list", "list_decisions",
})


def _mcp_hint(workspace_dir: str) -> Optional[str]:
    """Returns a gentle 'wire the MCP for the best experience' nudge, or None.

    Fires only when this project has no MCP wired, at most once per 24h per
    workspace (so it's a nudge, not a nag), and never when ``MITOS_NO_MCP_HINT``
    is set or the MCP is already wired. Fully fail-silent.

    Args:
        workspace_dir: The project directory the CLI command acted on.

    Returns:
        A one-line stderr-ready nudge, or None.
    """
    if os.environ.get("MITOS_NO_MCP_HINT") or _mcp_wired(workspace_dir):
        return None
    if not hint_due("mcp_hint.json", workspace_dir, 24 * 60 * 60):
        return None
    return (
        "💡 You're using the mitos CLI directly. For the best experience — ambient "
        "recall and structured recording (no shell-quoting) — wire the MCP server: "
        "see SETUP.md §3.\n   (Silence with MITOS_NO_MCP_HINT=1.)"
    )


def _upsert_env_var(env_path: str, name: str, value: str) -> None:
    """Inserts or replaces ``name=value`` in a ``.env`` file, preserving the rest.

    Replaces an existing (possibly empty) ``name=`` line in place; otherwise
    appends one. Creates the file (and parent dirs) if absent, and tightens the
    file to ``0600`` since it holds secrets.

    Args:
        env_path: Path to the ``.env`` file to write.
        name: The variable name (e.g. ``GEMINI_API_KEY``).
        value: The value to store.
    """
    os.makedirs(os.path.dirname(os.path.abspath(env_path)), exist_ok=True)
    lines: List[str] = []
    found = False
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith(f"{name}="):
                    lines.append(f"{name}={value}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        if lines and not lines[-1].endswith("\n"):
            lines.append("\n")
        lines.append(f"{name}={value}\n")
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    try:
        os.chmod(env_path, 0o600)
    except OSError:
        pass


def cmd_set_key(value: str, name: str = "GEMINI_API_KEY", is_global: bool = False) -> None:
    """Stores an API key in the global or project ``.env``.

    Args:
        value: The API key value to store.
        name: The env var name (default ``GEMINI_API_KEY``).
        is_global: If True, write the shared ``~/.config/mitos/.env`` (serves
            every project); otherwise write ``./.env`` for the current project.
    """
    if is_global:
        env_path = global_env_path()
    else:
        env_path = os.path.join(os.getcwd(), ".env")
    _upsert_env_var(env_path, name, value)
    scope = "globally (all projects)" if is_global else "for this project"
    print(f"Stored {name} {scope} → {env_path}")
    if not is_global:
        _ensure_gitignore_entry(os.path.join(os.getcwd(), ".gitignore"), ".env")


def _fmt_k(n: int) -> str:
    """Formats a token count with a 'k' suffix for readability (e.g. 14237 → '~14k')."""
    return f"~{round(n / 1000)}k" if n >= 1000 else f"~{n}"


def _print_overflow_detail(overflows: List[Dict[str, Any]]) -> None:
    """Prints the size-ceiling breakdown for over-budget context files (status surface).

    The detailed counterpart to the one-line nudge the write path shows: per file, its
    char/estimated-token size and the largest decisions in it, so an author knows what
    to re-scope. Informational only — never a readiness blocker.

    Args:
        overflows: Overflow records from ``overflow_report`` (largest file first).
    """
    n = len(overflows)
    noun = "file" if n == 1 else "files"
    print(f"\n  ⚠ {n} rendered axiom {noun} over the size ceiling "
          f"(informational — not a readiness blocker):")
    for o in overflows:
        print(f"      - {o['name']}: {o['chars']:,} chars "
              f"({_fmt_k(o['est_tokens'])} tokens, ceiling {o['threshold_chars']:,})")
        top = o.get("top_decisions", [])
        if top:
            print("          largest decisions:")
            for d in top:
                print(f"            • {d['slug']}  ({d['chars']:,} chars)")
    print("    These context files grow with the corpus — re-scope the largest decisions "
          "above, or split a broad scope.")


def _graph_behind_buffer(db_path: str) -> bool:
    """Detects a graph migrated to the V1b schema in place but never re-committed.

    Cheap, **graph-only** signal (no buffer parse, no false positives): ``True`` iff
    the ``mechanisms`` registry is empty while decision nodes still carry mechanism
    refs — the signature of a corpus whose V1b catalog (the seven non-kill edge types
    + the first-seen-wins mechanism registry) was never committed because the schema
    migration only widened the DDL. A ``mitos rebuild`` populates them. Any read
    failure (a pre-mechanisms V1a-schema graph, an absent/locked DB) is a safe
    ``False`` — never a spurious nudge.

    Args:
        db_path: The live graph path.

    Returns:
        ``True`` if the graph is behind its buffer's catalog, else ``False``.
    """
    try:
        conn = open_connection(db_path, read_only=True)
    except Exception:
        return False
    try:
        if conn.execute("SELECT COUNT(*) FROM mechanisms").fetchone()[0] > 0:
            return False
        carries_refs = conn.execute(
            "SELECT 1 FROM nodes WHERE mechanism_refs_json IS NOT NULL "
            "AND mechanism_refs_json NOT IN ('', '[]') LIMIT 1"
        ).fetchone()
        return carries_refs is not None
    except Exception:
        return False
    finally:
        conn.close()


def cmd_status(workspace_dir: str, as_json: bool = False) -> int:
    """Reports whether Mitos is set up for a project, and what (if anything) is missing.

    Designed to be run by a human OR an LLM in a new project: it answers "is Mitos
    ready here?" with a clear ✓/⚠/✗ report, a one-line verdict, and an exit code
    (0 = fully ready, 1 = needs attention / not set up). When not ready it prints
    concise next steps and points at the full SETUP walkthrough.

    Args:
        workspace_dir: The project directory to inspect.
        as_json: Emit a machine-readable JSON report instead of the text report.

    Returns:
        ``0`` if fully ready, ``1`` otherwise.
    """
    workspace_dir = os.path.abspath(workspace_dir)

    # `status` is the "is this set up?" probe, so a malformed config.toml is exactly
    # what it should surface — calmly, as not-ready, never a traceback. The main()
    # boundary would render only a generic `Error: …`; status owes its caller the
    # contextual "config malformed → not ready" report (Lesson 45 / entry-004).
    try:
        config = MitosConfig(workspace_dir)
    except ConfigError as e:
        if as_json:
            _emit_json({
                "workspace": workspace_dir,
                "ready": False,
                "initialized": False,
                "config_error": str(e),
            })
        else:
            print(f"\nMITOS STATUS for {workspace_dir} — NOT SET UP ✗\n")
            print(f"  ✗ config.toml malformed: {e}")
            print("      → fix it or re-run `mitos init`")
            print()
        return 1

    # Pre-V1a (prototype) graph detection — mirrors `init`'s early probe so the two
    # surfaces stay coherent (§5.2.7). A read-only GraphStore SKIPS the boot guard
    # (RO can't migrate), so a prototype graph would open fine and only fail deep in
    # get_all_nodes() (swallowed below) → status must run its OWN probe and force
    # not-ready. is_pre_v1a_schema is False for an absent/empty or V1a-or-later DB,
    # so a freshly-init'ed empty graph stays healthy (empty-is-healthy, P5).
    pre_v1a = False
    if os.path.exists(config.db_path):
        try:
            probe_conn = open_connection(config.db_path, read_only=True)
            try:
                pre_v1a = is_pre_v1a_schema(probe_conn)
            finally:
                probe_conn.close()
        except Exception:
            pass  # best-effort, like the RO read below: a probe failure leaves it False

    mitos_dir_ok = os.path.isdir(config.mitos_dir) and os.path.exists(
        os.path.join(config.mitos_dir, "config.toml")
    )
    decisions_ok = os.path.exists(config.decisions_file)
    spec_ok = os.path.exists(os.path.join(workspace_dir, "format-spec.md"))
    key_source = _gemini_key_source(workspace_dir)
    key_ok = key_source is not None
    q = _check_qdrant(config.qdrant_url, config.qdrant_collection)
    mcp_wired = _mcp_wired(workspace_dir)
    # Best-effort: is the pasted agent-file mitos note out of date? A recommendation,
    # never a readiness blocker — like the MCP-wired check.
    agent_drift = agent_block_drift(workspace_dir)

    graph_nodes = None
    active_nodes = None
    active_ids: Set[str] = set()
    id_to_slug: Dict[str, str] = {}
    # Read-only size-ceiling report for the generated context files. This is the
    # health surface the write-path overflow nudge points at — the detailed breakdown
    # (which files, which decisions to re-scope) lives here, not on every `record`.
    overflows: List[Dict[str, Any]] = []
    graph_behind = False
    if os.path.exists(config.db_path) and not pre_v1a:
        try:
            ro_store = GraphStore(config.db_path, read_only=True)
            all_nodes = ro_store.get_all_nodes()
            graph_nodes = len(all_nodes)
            id_to_slug = {n["id"]: n["slug"] for n in all_nodes}
            active_ids = ro_store.get_active_node_ids()
            active_nodes = len(active_ids)
            overflows = overflow_report(ro_store)
        except Exception:
            pass  # both reads are best-effort; a failure leaves the safe defaults
        graph_behind = _graph_behind_buffer(config.db_path)

    # Vector-index completeness by EXACT id-diff, not a count proxy. `mitos status`
    # is the SENSOR (reconcile is the heal), so it must catch a shortfall the count
    # `points >= active` structurally can't — dead-vector slack in the graveyard
    # (superseded vectors that linger, never GC'd) inflates the point total past the
    # active threshold and hides genuinely-missing active vectors (the live-corpus
    # incident: 181 >= 178 read healthy over 12 invisible active nodes). We scroll
    # Qdrant's actual point ids via the NO-CREATE read path (never constructing a
    # store that would `_ensure_collection` a missing collection) and diff.
    #
    # `missing_active` = active nodes with no vector (invisible to semantic search —
    #   the warned state). `orphan_points` = points with no active node — the
    #   graveyard substrate the all-superseded blackout vector consumes, per
    #   graveyard-vectors-now-consumed-by-blackout-best-effort: reported neutrally,
    #   never a warning, never deleted here. `None` means "could not verify" (scroll
    #   failed) — distinct from `0` ("verified complete"); we never fall back to the
    #   count proxy we just declared structurally blind.
    missing_active_slugs: Optional[List[str]] = None
    orphan_points: Optional[int] = None
    scroll_failed = False
    if q["reachable"] and q["collection_exists"] and active_nodes is not None:
        try:
            present = scroll_point_ids(config.qdrant_url, config.qdrant_collection)
            missing_active_slugs = sorted(
                id_to_slug.get(nid, nid) for nid in active_ids if hash_to_uuid(nid) not in present
            )
            active_uuids = {hash_to_uuid(nid) for nid in active_ids}
            orphan_points = len(present - active_uuids)
        except VectorStoreError:
            scroll_failed = True  # unknown — do not fabricate "complete"
    elif q["reachable"] and not q["collection_exists"] and active_ids:
        # Collection deleted OUTRIGHT (not just its points) while the graph holds
        # active nodes — the whole active surface is missing (the diff against an
        # empty collection). This is a full wipe, NOT the healthy fresh-project empty
        # state; absence of the collection must not read as calm health next to a
        # populated graph. `mitos reconcile` re-creates the collection and re-embeds.
        missing_active_slugs = sorted(id_to_slug.get(nid, nid) for nid in active_ids)
        orphan_points = 0

    initialized = mitos_dir_ok and decisions_ok
    # A fresh, initialized project has NO Qdrant collection yet — it auto-creates
    # on the first `record_decision`. So an absent (or empty) collection is a
    # normal ready state, NOT a blocker: a project with .mitos/, a key, and a
    # reachable Qdrant is ready to record its first decision. Only an unreachable
    # Qdrant degrades semantic surface/query. A pre-V1a (prototype) graph is never
    # ready — it must be routed through the one-time cutover first (§5.2.7).
    ready = initialized and key_ok and q["reachable"] and not pre_v1a

    if as_json:
        _emit_json({
            "workspace": workspace_dir,
            "ready": ready,
            "initialized": initialized,
            "pre_v1a": pre_v1a,
            "qdrant_url": config.qdrant_url,
            "collection": config.qdrant_collection,
            "checks": {
                "mitos_workspace": mitos_dir_ok,
                "decisions_buffer": decisions_ok,
                "format_spec": spec_ok,
                "gemini_api_key": key_ok,
                "qdrant_reachable": q["reachable"],
                "collection_exists": q["collection_exists"],
                "collection_points": q["points"],
                "graph_nodes": graph_nodes,
                "active_nodes": active_nodes,
                "missing_active_vectors": (
                    None if missing_active_slugs is None else len(missing_active_slugs)
                ),
                "missing_active_slugs": missing_active_slugs,
                "orphan_points": orphan_points,
                "mcp_wired": mcp_wired,
            },
            "graph_behind_buffer": graph_behind,
            "scope_overflow": overflows,
            "agent_guide_version": AGENT_GUIDE_VERSION,
            "agent_files": agent_drift["files"],
        })
        return 0 if ready else 1

    verdict = "READY ✓" if ready else ("NEEDS ATTENTION ⚠" if initialized else "NOT SET UP ✗")
    mark = lambda ok: "✓" if ok is True else ("✗" if ok is False else "—")
    # An absent collection on a reachable Qdrant is normal (auto-creates on the
    # first record), so show it as a neutral "—" with a note — never a ✗ that
    # would contradict an otherwise-READY verdict.
    if not q["reachable"]:
        coll_mark, coll_hint = None, "needs Qdrant up (see above)"
    elif q["collection_exists"]:
        coll_mark, coll_hint = True, None
    elif active_ids:
        # Absent collection but a populated graph = a full wipe, not a fresh project.
        # Say so accurately rather than the calm "none recorded yet" (the warning
        # below carries the detail); still neutral, never a readiness ✗.
        coll_mark, coll_hint = (
            None,
            f"missing — {len(active_ids)} active node(s) have no vectors; run `mitos reconcile`",
        )
    else:
        coll_mark, coll_hint = None, "auto-created on first record — none recorded yet"
    checks = [
        ("workspace (.mitos/ + config.toml)", mitos_dir_ok, "run `mitos init`"),
        ("decisions.md buffer", decisions_ok, "created by `mitos init`"),
        # Reference copy for humans/agents — the parser reads the spec from the
        # installed package, so a missing workspace copy never gates readiness:
        # neutral "—", never a ✗ under a READY ✓ verdict (✗ is for real blockers).
        ("format-spec.md", True if spec_ok else None,
         "restore the reference copy: re-run `mitos init` (non-destructive)"),
        ("GEMINI_API_KEY" + (f" (from {key_source})" if key_source else ""), key_ok,
         "set it once for all projects: `mitos set-key --global <KEY>`"),
        (f"Qdrant reachable ({config.qdrant_url})", q["reachable"],
         "start it: `docker compose up -d` in the mitos repo"),
        (f"collection '{config.qdrant_collection}'", coll_mark, coll_hint),
        # Recommendation, not a requirement — never a ✗. Agents get the best AX
        # (ambient surface/record, structured args, no shell-quoting) via the MCP.
        ("MCP wired (recommended for agents)", True if mcp_wired else None,
         "agents: wire `mitos serve` — see SETUP.md §3 (CLI works without it)"),
    ]
    # A pre-V1a (prototype) graph is the dominant blocker — surface it prominently,
    # right after the workspace line, with the same route-to-cutover guidance `init`
    # raises. Never `READY ✓` for a graph `init` would refuse (§5.2.7).
    if pre_v1a:
        # Route through the shared constant (single source) so this check-line hint
        # and the next-steps line below can never re-diverge — both name `mitos
        # cutover`. (The store.py boot-guard message stays its own deeper-internal
        # phrasing; it is not an operator-primary surface.)
        checks.insert(1, ("graph schema (V1a)", False, _CUTOVER_GUIDANCE))
    print(f"\nMITOS STATUS for {workspace_dir} — {verdict}\n")
    for label, ok, hint in checks:
        line = f"  {mark(ok)} {label}"
        if ok is not True and hint:
            line += f"   → {hint}"
        print(line)
    if q["reachable"] and q["collection_exists"] and q["points"] is not None:
        print(f"      ({q['points']} vector(s) indexed)")
    if graph_nodes is not None:
        print(f"  • graph holds {graph_nodes} node(s)")
    # Vector-completeness verdict from the exact id-diff computed above (not a
    # count). Three outcomes:
    #   • scroll failed (missing_active_slugs is None) → we could not verify; say so
    #     and never fall back to the count proxy (structurally blind, per
    #     status-vector-completeness-by-id-diff-not-count-proxy).
    #   • missing_active_slugs non-empty → the warned state: active nodes with no
    #     vector, invisible to semantic surface/query (names slugs at small N).
    #   • empty → verified complete; stay quiet.
    # Orphan (graveyard) points are reported neutrally, never as a warning — they
    # are the blackout vector's substrate (graveyard-vectors-now-consumed-by-blackout).
    # Each branch is self-guarding on the diff outcome above — no collection_exists
    # gate here, so a full collection wipe with a populated graph (which sets
    # missing_active_slugs to the whole active set) warns just like a points wipe.
    if scroll_failed:
        print(
            "\n  ⚠ could not verify vector completeness — Qdrant scroll failed; "
            "run `mitos status` again when Qdrant is reachable."
        )
    elif missing_active_slugs:
        n = len(missing_active_slugs)
        print(
            f"\n  ⚠ vector index incomplete — {n} active node(s) have no vector "
            f"and are invisible to semantic surface/query. Run `mitos reconcile` "
            f"to re-embed them (or `mitos sync` if the outbox is non-empty) — "
            f"informational, not a readiness blocker."
        )
        if n <= 5:
            for slug in missing_active_slugs:
                print(f"      • {slug}")
    if orphan_points:
        print(
            f"  • {orphan_points} graveyard point(s) belong to inactive/removed "
            f"nodes — retained (they power all-superseded blackout recovery), "
            f"not an error."
        )
    if overflows:
        _print_overflow_detail(overflows)
    if graph_behind:
        print(
            "\n  ⚠ graph is behind your buffer — the V1b edge catalog + mechanism "
            "registry were never committed for this corpus (a schema upgrade widens "
            "the DDL but does not re-commit). Run `mitos rebuild` to populate them "
            "(informational — not a readiness blocker; no decisions are at risk)."
        )
    if agent_drift["stale"]:
        stale_files = ", ".join(
            f["file"] for f in agent_drift["files"]
            if f["status"] in ("outdated", "unversioned")
        )
        print(f"  ⚠ agent-file mitos note out of date ({stale_files}) "
              f"— refresh with `mitos agent-block`")
    print()
    if not ready:
        print("Next steps:")
        n = 1
        if pre_v1a:
            print(f"  {n}. {_CUTOVER_GUIDANCE}"); n += 1
        if not initialized:
            print(f"  {n}. `mitos init` here (creates .mitos/, decisions.md, scaffolds .env)"); n += 1
        if not key_ok:
            print(f"  {n}. Set your GEMINI_API_KEY once for all projects: "
                  f"`mitos set-key --global <KEY>` (or per-project: `mitos set-key <KEY>`)"); n += 1
        if not q["reachable"]:
            print(f"  {n}. Start Mitos's Qdrant: `docker compose up -d` from the mitos repo"); n += 1
        print("  Full walkthrough → SETUP.md "
              "(https://github.com/dovahkiin-v/mitos/blob/main/SETUP.md)")
        print()
    return 0 if ready else 1


def cmd_agent_block(workspace_dir: str, check: bool = False) -> int:
    """Prints the canonical agent-file block, or checks pasted copies for drift.

    The block is the thin, versioned pointer a project pastes into its agent files
    (``AGENTS.md`` / ``CLAUDE.md`` / ``GEMINI.md`` / ``.cursorrules``) so the next
    agent knows mitos is set up here. Without ``--check`` it prints the current block
    to stdout, paste-ready; with ``--check`` it scans the project's agent files and
    reports which carry an out-of-date or unversioned mitos note.

    Args:
        workspace_dir: The project root (only used by ``--check``).
        check: Report drift in the project's agent files instead of printing the block.

    Returns:
        ``0`` on a plain print, or when ``--check`` finds no stale copy; ``1`` when
        ``--check`` finds an outdated/unversioned mitos note to refresh.
    """
    if not check:
        print(agent_block())
        return 0

    workspace_dir = os.path.abspath(workspace_dir)
    report = agent_block_drift(workspace_dir)
    files = report["files"]
    print(f"\nAgent-file mitos note (current guide: v{AGENT_GUIDE_VERSION}) for {workspace_dir}\n")
    if not files:
        print("  — no agent file references mitos yet.")
        print("    Paste `mitos agent-block` into your AGENTS.md / CLAUDE.md / GEMINI.md so")
        print("    the next agent knows mitos is set up here.\n")
        return 0
    for f in files:
        if f["status"] == "current":
            print(f"  ✓ {f['file']}  (guide v{f['marker_version']})")
        elif f["status"] == "outdated":
            print(f"  ⚠ {f['file']}  (guide v{f['marker_version']} → v{AGENT_GUIDE_VERSION}) "
                  f"— refresh with `mitos agent-block`")
        else:  # unversioned
            print(f"  ⚠ {f['file']}  (mitos note with no version marker) "
                  f"— refresh with `mitos agent-block`")
    print()
    return 1 if report["stale"] else 0


def cmd_cutover(
    config: MitosConfig, *, allow_drops: bool, assume_yes: bool, as_json: bool
) -> int:
    """Runs the one-time prototype→V1a cutover (the destructive migration).

    Orchestrates 7a's verdict surface into an operator-runnable verb: probe →
    rebuild + gate → present the verdict → confirm (or override a shortfall with
    ``--allow-drops``) → atomic swap → print the post-swap runbook. The load-bearing
    correctness lives in :func:`~mitos.cutover.perform_swap`; this is the thin
    interactive orchestrator (K1).

    Only a genuine **prototype** graph proceeds — an already-V1a, empty, or absent
    graph is a cheap no-op (G7), which also makes a post-success or post-crash
    re-run idempotent. A **corpus defect** raises ``CutoverError`` (propagated to
    ``main()``'s boundary, rendered one-line, exit 1) and is never overridable; a
    completeness **shortfall** is overridable with ``--allow-drops`` (P6 — the
    markdown is authoritative, a drop may be a deliberate purge).

    Args:
        config: The active workspace config.
        allow_drops: Proceed past a completeness shortfall (active cores absent from
            the rebuild). Never overrides a corpus defect.
        assume_yes: Skip the interactive swap confirmation (automation / non-TTY).
        as_json: Emit a machine-readable JSON report instead of the human runbook.

    Returns:
        ``0`` on a successful swap (or a no-op non-prototype graph), ``1``
        otherwise (absent graph, refused shortfall, declined/missing confirmation).

    Raises:
        CutoverError: On a corpus defect during the rebuild (caught at the
            ``main()`` boundary).
    """
    # 1. Up-front prototype probe (G7) — mirrors the cmd_init / cmd_status RO-probe
    #    shape. An absent / already-V1a / empty graph is a cheap no-op: no rebuild,
    #    no swap, no Qdrant churn (and a post-success re-run is idempotent).
    if not os.path.exists(config.db_path):
        if as_json:
            _emit_json({"workspace": config.workspace_dir,
                        "swapped": False, "reason": "no_graph"})
        else:
            print("No graph found at this workspace — run `mitos init` for a fresh "
                  "V1a workspace (nothing to cut over).")
        return 1
    probe_conn = open_connection(config.db_path, read_only=True)
    try:
        is_prototype = is_pre_v1a_schema(probe_conn)
    finally:
        probe_conn.close()
    if not is_prototype:
        if as_json:
            _emit_json({"workspace": config.workspace_dir,
                        "swapped": False, "reason": "not_a_prototype"})
        else:
            print("Graph is already on the V1a schema (or empty) — nothing to "
                  "cut over.")
        return 0

    # 2. Rebuild + gate (7a). A corpus defect raises CutoverError, which propagates
    #    to main()'s `except MitosError` boundary (one-line error, exit 1) — never
    #    overridable here, it is malformed markdown the operator must fix.
    aside_db_path = default_aside_db_path(config)
    result = rebuild_and_gate(config, aside_db_path=aside_db_path)

    qdrant_wipe_cmd = (
        f"curl -X DELETE {config.qdrant_url}/collections/{config.qdrant_collection}"
    )

    # 3. Present the verdict.
    if not as_json:
        print("\nCutover rebuild verdict:")
        print(f"  decisions committed:       {result.decisions_committed}")
        print(f"  open questions committed:  {result.open_questions_committed}")
        print(f"  active cores (old graph):  {result.reference_active_count}")
        print(f"  active cores (rebuild):    {result.reconstructed_active_count}")

    if not result.gate_passed:
        n = len(result.missing_cores)
        if not as_json:
            print(f"\n⚠ {n} active core(s) from the prototype are ABSENT from the "
                  f"rebuild:")
            for mc in result.missing_cores:
                print(f"    - '{mc.slug}' [{mc.kind}]: {mc.axiom_excerpt}")
        if not allow_drops:
            if as_json:
                _emit_json({**result.to_dict(), "swapped": False,
                            "reason": "shortfall_refused",
                            "qdrant_wipe_cmd": qdrant_wipe_cmd})
            else:
                print(f"\nRefusing to swap: {n} active core(s) would be dropped. "
                      f"Review the offenders above. If this purge is intentional "
                      f"(they were deliberately removed from the corpus), re-run "
                      f"with --allow-drops. Otherwise restore them in "
                      f"{os.path.basename(config.decisions_file)} and re-run.")
            return 1
        if not as_json:
            print(f"\n--allow-drops set: proceeding despite the {n} dropped "
                  f"core(s), treating the corpus as authoritative (P6).")

    # 4. Confirm the destructive swap (K5/G5 — never call input() on a no-TTY).
    if not assume_yes:
        if as_json:
            # JSON mode is for automation: never prompt; require an explicit --yes.
            _emit_json({**result.to_dict(), "swapped": False,
                        "reason": "confirmation_required",
                        "qdrant_wipe_cmd": qdrant_wipe_cmd})
            return 1
        if sys.stdin.isatty():
            answer = input("\nProceed with the cutover swap? This replaces the "
                           "live graph. [y/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                print("Aborted — no changes made.")
                return 1
        else:
            print("\nRefusing to prompt: this is a destructive operation and stdin "
                  "is not a TTY. Re-run with --yes to proceed non-interactively.")
            return 1

    # 5. Swap — the single atomic instant. The timestamp is pinned by the caller
    #    (G8) so perform_swap stays wall-clock-free and fixture-deterministic.
    bak_path = perform_swap(
        config, result.aside_db_path, timestamp=time.strftime("%Y%m%d-%H%M%S")
    )

    # 6. Print the post-swap runbook (the operator must not have to remember it).
    if as_json:
        _emit_json({**result.to_dict(), "swapped": True,
                    "bak_path": bak_path,
                    "qdrant_wipe_cmd": qdrant_wipe_cmd})
        return 0

    print(f"\n✓ Cutover complete — the V1a graph is live at {config.db_path}.")
    if bak_path:
        print(f"  Old prototype graph backed up to: {bak_path}")
    print("\nFinish the cutover (it is not fully done until these run):")
    print("  1. Wipe the stale Qdrant collection (its vectors are keyed on the old")
    print("     prototype ids — it auto-recreates on the next sync):")
    print(f"       {qdrant_wipe_cmd}")
    print("  2. Re-embed the V1a active set:  mitos sync   (or: mitos sync --embed-only)")
    print("     Semantic surface/query stay degraded until the queue drains;")
    print("     graph-only `mitos list` works throughout.")
    print("     (If you ever wipe Qdrant later with the outbox already empty,")
    print("      `mitos reconcile` re-embeds the active set in one pass.)")
    print("  3. If `mitos serve` was running, restart it.")
    print("  4. Verify:  mitos status   → expect READY ✓")
    if bak_path:
        print(f"  5. Once satisfied, remove the backup:  rm {bak_path}")
    print("  Full runbook → SETUP.md → Cutover.")
    return 0


def _print_rebuild_remediation(casualties, missing_cores, decisions_basename: str) -> None:
    """Prints reassuring, per-class remediation when a rebuild is refused.

    The upgrade-path UX (no stranger's experience is broken): a user who hits a stale
    citation must learn three things at once — their decisions are SAFE, exactly WHAT
    to do per failure class, and that ``--allow-drops`` is a safe escape — never a
    bare ``refused`` wall.

    Args:
        casualties: The :class:`~mitos.cutover.Casualty` punch-list (each carries
            ``codes`` + a ``detail`` that already names any superseding successor).
        missing_cores: Active decisions absent from the rebuild (a corpus removal,
            not a citation defect) — guided separately.
        decisions_basename: The buffer filename to point edits at (e.g.
            ``decisions.md``).
    """
    print(
        f"\nRefusing to swap — the live graph is untouched and nothing is lost: "
        f"{decisions_basename} (plus the archives) is the source of truth, and every "
        f"entry below stays there. Here is how to clear each one:"
    )
    codes = {code for c in casualties for code in c.codes}
    if "dangling_edge" in codes:
        print(
            "  • dangling_edge — the entry cites a decision that has since been "
            "superseded. Re-point that citation to the active successor named in the "
            "detail above (or delete the citation line), then re-run `mitos rebuild`."
        )
    if "missing_target" in codes:
        print(
            "  • missing_target — the entry cites a slug that no longer exists "
            "(renamed away, or a typo). Fix or remove the citation, then re-run."
        )
    other = sorted(codes - {"dangling_edge", "missing_target"})
    if other:
        print(
            f"  • {', '.join(other)} — see the detail above; fix the entry in "
            f"{decisions_basename} and re-run."
        )
    if missing_cores:
        print(
            "  • Some active decisions are absent from the corpus entirely (a removal, "
            "not a citation defect). If that is intentional, --allow-drops accepts it; "
            "otherwise restore them in the buffer."
        )
    print(
        "\nOr re-run `mitos rebuild --allow-drops` to proceed now — the listed entries "
        "remain in your markdown and re-enter the graph the moment you fix the "
        "citation and rebuild again."
    )


def cmd_rebuild(
    config: MitosConfig, *, allow_drops: bool, assume_yes: bool, as_json: bool
) -> int:
    """Rebuilds the graph from the full corpus through the current catalog.

    The recurring twin of :func:`cmd_cutover`: re-commits every decision and open
    question oldest-first (archives then buffer) into a build-aside graph and
    atomically swaps it in, so a graph upgraded in place (the V1b schema on pre-V1b
    data — the catalog flip's edges and the mechanism registry never re-committed)
    gains the full catalog. Unlike cutover it runs on a **current** (V1a/V1b) graph
    and is **resilient**: an entry the catalog now rejects (a citation to a since-
    superseded or never-authored node) is a surfaced casualty, not an abort. No ADRs
    are at risk — the markdown (buffer + archives) is the source of truth (M7/P6) and
    the swap backs up the old graph.

    A graph **format** defect still raises ``CutoverError`` (propagated to ``main()``).
    A **casualty** (an entry that cannot commit) or a completeness **shortfall** (an
    active decision the rebuild would drop) blocks the swap unless ``--allow-drops``.

    Args:
        config: The active workspace config.
        allow_drops: Proceed past casualties / a shortfall (the dropped entries stay
            in the markdown; fix their citations and re-run to re-include them).
        assume_yes: Skip the interactive swap confirmation (automation / non-TTY).
        as_json: Emit a machine-readable JSON report instead of the human summary.

    Returns:
        ``0`` on a successful swap, ``1`` otherwise (absent/prototype graph, refused
        casualties/shortfall, declined/missing confirmation).

    Raises:
        CutoverError: On a corpus format defect during the rebuild (caught at the
            ``main()`` boundary).
    """
    # 1. Probe: rebuild runs on a CURRENT graph. Absent → init; prototype → the
    #    one-time cutover owns it (don't double-handle).
    if not os.path.exists(config.db_path):
        if as_json:
            _emit_json({"workspace": config.workspace_dir,
                        "swapped": False, "reason": "no_graph"})
        else:
            print("No graph found at this workspace — run `mitos init` first "
                  "(nothing to rebuild).")
        return 1
    probe_conn = open_connection(config.db_path, read_only=True)
    try:
        is_prototype = is_pre_v1a_schema(probe_conn)
    finally:
        probe_conn.close()
    if is_prototype:
        if as_json:
            _emit_json({"workspace": config.workspace_dir,
                        "swapped": False, "reason": "prototype_graph"})
        else:
            print("Graph is a pre-V1a prototype — run `mitos cutover` (the one-time "
                  "migration) instead of `mitos rebuild`.")
        return 1

    # 2. Rebuild + gate (resilient: casualties are surfaced, not raised). A corpus
    #    FORMAT defect still raises CutoverError → main()'s boundary (exit 1).
    aside_db_path = default_aside_db_path(config)
    result = rebuild_and_gate(config, aside_db_path=aside_db_path, strict=False)

    # 3. Present the verdict.
    if not as_json:
        print("\nRebuild verdict:")
        print(f"  decisions committed:       {result.decisions_committed}")
        print(f"  open questions committed:  {result.open_questions_committed}")
        print(f"  active cores (live graph): {result.reference_active_count}")
        print(f"  active cores (rebuild):    {result.reconstructed_active_count}")

    casualties = result.residual_casualties
    if casualties and not as_json:
        noun = "entry" if len(casualties) == 1 else "entries"
        print(f"\n⚠ {len(casualties)} {noun} could not be rebuilt (left in the buffer "
              f"— fix the citation to re-include):")
        for c in casualties:
            code_str = ", ".join(c.codes) if c.codes else "rejected"
            print(f"    - '{c.slug}' (lines {c.line_start}-{c.line_end}) "
                  f"[{code_str}]: {c.detail}")

    if not result.gate_passed and not as_json:
        n = len(result.missing_cores)
        print(f"\n⚠ {n} active decision(s) in the live graph would be DROPPED by this "
              f"rebuild:")
        for mc in result.missing_cores:
            print(f"    - '{mc.slug}' [{mc.kind}]: {mc.axiom_excerpt}")

    blocked = bool(casualties) or not result.gate_passed
    if blocked and not allow_drops:
        if as_json:
            _emit_json({**result.to_dict(), "swapped": False,
                        "reason": "casualties_or_shortfall_refused"})
        else:
            _print_rebuild_remediation(
                casualties, result.missing_cores, os.path.basename(config.decisions_file)
            )
        return 1
    if blocked and not as_json:
        print("\n--allow-drops set: proceeding despite the dropped content, treating "
              "the corpus as authoritative (P6). Dropped entries remain in the markdown.")

    # 4. Confirm the destructive swap (never call input() on a no-TTY).
    if not assume_yes:
        if as_json:
            _emit_json({**result.to_dict(), "swapped": False,
                        "reason": "confirmation_required"})
            return 1
        if sys.stdin.isatty():
            answer = input("\nProceed with the rebuild swap? This replaces the live "
                           "graph (a backup is kept). [y/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                print("Aborted — no changes made.")
                return 1
        else:
            print("\nRefusing to prompt: this replaces the live graph and stdin is "
                  "not a TTY. Re-run with --yes to proceed non-interactively.")
            return 1

    # 5. Swap — the single atomic instant (timestamp pinned by the caller, G8).
    bak_path = perform_swap(
        config, result.aside_db_path, timestamp=time.strftime("%Y%m%d-%H%M%S")
    )

    # 6. Post-swap guidance.
    if as_json:
        _emit_json({**result.to_dict(), "swapped": True,
                    "bak_path": bak_path})
        return 0

    print(f"\n✓ Rebuild complete — the graph at {config.db_path} now reflects the "
          f"full catalog from your corpus.")
    if bak_path:
        print(f"  Old graph backed up to: {bak_path}")
    print("\nNext:")
    print("  - Re-embed so semantic surface/query reflect the rebuild:  mitos sync")
    print("    (Or, if Qdrant was wiped directly and the outbox is empty:  mitos reconcile)")
    print("  - Verify:  mitos status   → expect READY ✓ (the rebuild nudge clears)")
    if bak_path:
        print(f"  - Once satisfied, remove the backup:  rm {bak_path}")
    return 0


def load_dotenv_file(path: str = ".env") -> None:
    """Loads ``KEY=value`` pairs from a ``.env`` file into the environment.

    Mitos reads its credentials (``GEMINI_API_KEY``, ``ANTHROPIC_API_KEY``) and
    ``QDRANT_URL`` straight from ``os.environ``. This loads them from a workspace
    ``.env`` so a key dropped in that file takes effect without a manual
    ``export`` — the same manual parse the live test-suite already uses, with no
    new dependency (P19 — Dependency Skepticism). An empty value is skipped, and
    an existing environment value is never overridden (an explicit ``export``
    wins over the file).

    Args:
        path: Path to the ``.env`` file (default: ``.env`` in the cwd, i.e. the
            workspace root where ``mitos`` is invoked).
    """
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val and key not in os.environ:
                    os.environ[key] = val
    except OSError:
        pass


# =========================================================================== #
# Phase 3a — `mitos check`: the read-only corpus conflict audit / CI gate.
#
# Presentation + disposition only: the engine (mitos/check.py) computes every
# partition and count; `cmd_check` maps a typed CheckRunResult to human/JSON
# output and the shipped 0/1/2 exit contract (CHK-C2). The load-bearing rules —
# the exit table, the no-row-on-refusal rule (KD4), the plan→confirm→execute→
# row seam order (KD5), the `_emit_json`-only discipline — live in the phase plan.
# =========================================================================== #

# The parent's P15 per-check token budget estimate, per judged batch — used only
# to size the TTY confirm's disclosure (a rough figure, not a billed number).
_CHECK_TOKENS_PER_BATCH_ESTIMATE = 3000

# The four reverse-relation modifier stamp keys copied off a hydrated finding node
# (the `candidate_payload` manner, conflict.py). Single-sourced from the store's
# canonical map so a new modifier edge type never drifts this surface.
_CHECK_MODIFIER_STAMP_KEYS: Tuple[str, ...] = tuple(MODIFIER_EDGE_KEYS.values())


def _confirm_spend(n: int, *, assume_yes: bool, as_json: bool) -> Optional[int]:
    """The shared CHK-D5 spend confirm — corpus + staged (KD4).

    The single gate both ``check`` modes pass ``n`` (corpus: fresh judgment groups;
    staged: pending decision entries) through before any judge call, so the ``>``
    comparison and the three refusal surfaces can never fork between the two. Fires
    strictly ``n > check.CHECK_CONFIRM_BATCHES`` (read as a module attribute so a test
    monkeypatch is seen); at/below the threshold or with ``assume_yes`` it returns
    ``None`` (proceed) without prompting. All three refusals return exit ``2``, zero
    spend: ``--json`` emits an error object (automation never prompts), a non-TTY
    prints the vector message, an interactive decline prints "nothing spent".

    Args:
        n: The disclosure unit — the count of pending judgment batches.
        assume_yes: Waive the confirm (the ``--yes`` opt-in).
        as_json: Automation surface — emit an error object instead of prompting.

    Returns:
        A refusal exit code (``2``) when the spend is declined, or ``None`` to proceed.
    """
    if n <= check.CHECK_CONFIRM_BATCHES or assume_yes:
        return None
    if as_json:
        # Automation never prompts (a prompt would also corrupt the object).
        _emit_json({
            "error": (f"{n} judgment batches pending — re-run with --yes to "
                      f"authorize the spend."),
            "code": "confirmation_required",
            "batches_planned": n,
        })
        return 2
    if not sys.stdin.isatty():
        print(f"{n} judgment batches pending — re-run with --yes to authorize "
              f"the spend.", file=sys.stderr)
        return 2
    estimate = n * _CHECK_TOKENS_PER_BATCH_ESTIMATE
    print(f"{n} judgment batches pending (≈{estimate:,} tokens) — this run "
          f"will call the judge model.")
    if input("Proceed with the spend? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Aborted — nothing spent.")
        return 2
    return None


def _build_check_substrate(
    config: MitosConfig,
) -> Tuple[Optional[GeminiEmbeddingProvider], Optional[QdrantVectorStore],
           Optional[str], Optional[str]]:
    """Constructs the two best-effort external substrate providers (KD2).

    Both providers RAISE at construction — ``GeminiEmbeddingProvider`` when
    ``GEMINI_API_KEY`` is unset, ``QdrantVectorStore`` when Qdrant is unreachable
    (its ``__init__`` contacts the network). Each is caught NARROWLY (its own typed
    error, never a blanket ``except``) and degraded to ``None`` + a kept detail
    string, so an unexpected error still propagates. The disposition (refuse iff the
    run has sweep work) is the caller's — this only reports availability. Separated
    into a module-level helper so tests inject keyed fakes at this seam.

    Args:
        config: The active workspace config (paths + Qdrant coordinates).

    Returns:
        ``(embed, vector, embed_detail, vector_detail)`` — a provider or ``None``,
        and the failure message string (``None`` on success) for each.
    """
    embed: Optional[GeminiEmbeddingProvider] = None
    embed_detail: Optional[str] = None
    try:
        embed = GeminiEmbeddingProvider(
            os.path.join(config.mitos_dir, "embedding_cache.sqlite")
        )
    except EmbeddingError as exc:
        embed_detail = str(exc)
    vector: Optional[QdrantVectorStore] = None
    vector_detail: Optional[str] = None
    try:
        vector = QdrantVectorStore(config.qdrant_url, config.qdrant_collection)
    except VectorStoreError as exc:
        vector_detail = str(exc)
    return embed, vector, embed_detail, vector_detail


def _build_check_telemetry(config: MitosConfig) -> Optional[TelemetryStore]:
    """Constructs the sibling telemetry store best-effort (KD2), or ``None``.

    A telemetry-construction failure is the engine's documented ``reuse_read``
    degradation (the run proceeds all-fresh, reports unpartitioned, and the KD5
    seam records no row) — it must never crash a read-only audit. Mirrors
    ``_new_conflict_run``'s best-effort posture (sync.py). A module-level seam so
    tests inject ``None`` or a failing-write wrapper here.

    Args:
        config: The active workspace config.

    Returns:
        The :class:`TelemetryStore`, or ``None`` when it could not be constructed.
    """
    try:
        return TelemetryStore(config.telemetry_path)
    except (sqlite3.Error, DatabaseError, MitosError):
        return None


def _build_check_judge() -> Optional[Callable]:
    """Builds the bound conflict-judgment executor, or ``None`` when keyless (KD6).

    The ``_build_conflict_judge`` shape with the OPPOSITE disposition: it does NOT
    couple to embed/vector presence (check's candidate gather already ran at plan
    time), and ``None`` means "let the engine degrade typed" (``judge=None`` + fresh
    groups → a typed judgment degradation, exit 2, zero spend), not "skip the
    surface". The Anthropic SDK import is lazy so no other verb drags ``anthropic``
    onto its import path (Tier discipline). Built only after the confirm passes and
    only when fresh groups exist, so a reuse-only/clean run never constructs a client.

    Returns:
        The bound one-arg ``judge`` callable, or ``None`` when ``ANTHROPIC_API_KEY``
        is absent.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    import anthropic
    from mitos.conflict_judgment import make_judgment_executor

    return make_judgment_executor(anthropic.Anthropic(api_key=api_key))


def _check_finding_side(node: Dict[str, Any]) -> Dict[str, Any]:
    """Shapes one finding side as ``id`` + the Letter core + non-empty stamps (C4).

    The node is already hydrated + modifier-stamped (2b snapshot / ``Candidate.node``)
    — slugs and stamps ride free, no per-finding store read. Reuses
    :func:`display.letter_payload` for the Letter core (never a raw node), then copies
    only the present modifier stamps (the ``candidate_payload`` conditional-copy
    manner — blind indexing would KeyError on the common unmodified node).

    Args:
        node: A finding's hydrated ``proposal_node`` / ``partner_node`` dict.

    Returns:
        The JSON-native finding-side object (``id``, ``slug``, ``axiom``, ``scope``,
        ``rejected_paths``, plus any non-empty modifier stamps).
    """
    side: Dict[str, Any] = {"id": node["id"]}
    side.update(letter_payload(node, brief=False))
    for key in _CHECK_MODIFIER_STAMP_KEYS:
        if key in node:
            side[key] = node[key]
    return side


def _check_finding_json(finding: "check.CheckFinding") -> Dict[str, Any]:
    """Renders one :class:`~mitos.check.CheckFinding` as its flat JSON object (§8/KD7)."""
    return {
        "novelty": finding.novelty,
        "confidence": finding.confidence,
        "rationale": finding.rationale,
        "score": finding.score,
        "reused": finding.reused,
        "source_batch_id": finding.source_batch_id,
        "source_created_at": finding.source_created_at,
        "proposal": _check_finding_side(finding.proposal_node),
        "partner": _check_finding_side(finding.partner_node),
    }


def _resolve_exclusion_display(
    store: GraphStore, ids: Tuple[str, ...]
) -> List[Dict[str, Any]]:
    """Resolves display slugs for coverage-exclusion node ids, best-effort live (MI-2).

    ``coverage_exclusion_ids`` returns content hashes; ``get_node`` is
    state-agnostic (a since-superseded node still resolves), so the raw-id fallback
    covers only a genuinely absent node (``None`` → ``slug`` ``None``). These reads
    sit in the display-model build, before the write seam.
    """
    out: List[Dict[str, Any]] = []
    for node_id in ids:
        node = store.get_node(node_id)
        out.append({"id": node_id, "slug": node.get("slug") if node else None})
    return out


def _check_transient_count(result: "check.CheckRunResult") -> int:
    """Distinct transient-backlog node count across the readable probes (§8)."""
    ids = {row.node_id for row in result.start_probe.transient}
    if isinstance(result.end_probe, check.StaleProbe):
        ids |= {row.node_id for row in result.end_probe.transient}
    return len(ids)


def _check_json_object(
    result: "check.CheckRunResult",
    row: "check.CheckRunRow",
    *,
    exclusions: List[Dict[str, Any]],
    exit_code: int,
    row_written: bool,
    scope: Optional[str],
    fresh: bool,
    transient_count: int,
) -> Dict[str, Any]:
    """Assembles the single §8 ``--json`` object (a shipped API — additive only).

    ``findings_new`` / ``findings_known`` are read off ``row`` (built via
    :func:`check.check_run_row_from_result`) so the JSON and the ``check_runs`` row
    can never disagree on the NULL-when-unpartitioned rule. ``scope`` is ABSENT when
    unset (MI-9: never ``""``). Emission is deferred to :func:`_emit_json` by the
    caller (the ONLY JSON path).
    """
    obj: Dict[str, Any] = {
        "run_id": result.run_id,
        "mode": "corpus",
        "exit_code": exit_code,
        "started_at": result.started_at,
        "ended_at": result.ended_at,
        "fresh": fresh,
    }
    if scope is not None:
        obj["scope"] = scope
    obj.update({
        "nodes_total": result.nodes_total,
        "nodes_swept": result.nodes_swept,
        "pairs_judged_fresh": result.pairs_judged_fresh,
        "pairs_reused": result.pairs_reused,
        "batches_planned": result.batches_planned,
        "batches_executed": result.batches_executed,
        "batches_skipped": result.batches_skipped,
        "findings": [_check_finding_json(f) for f in result.findings],
        "findings_new": row.findings_new,
        "findings_known": row.findings_known,
        "degradations": list(check.run_degradations(result)),
        "coverage_exclusions": exclusions,
        "index_backlog_transient": transient_count,
        "summary_row_written": row_written,
    })
    return obj


def _print_check_finding_side(node: Dict[str, Any]) -> None:
    """Prints one finding side's Letter fields as a calm plain-text block (P9)."""
    print(f"    {node['slug']}")
    print(f"      Axiom:    {node['core_axiom']}")
    scope = node.get("scope") or []
    scope_text = ", ".join(scope) if scope else "(global — no scope declared)"
    print(f"      Scope:    {scope_text}")
    rejected = node.get("rejected_paths")
    if rejected:
        print(f"      Rejected: {rejected}")
    for key in _CHECK_MODIFIER_STAMP_KEYS:
        if key in node:
            print(f"      ({key.replace('_', ' ')}: {', '.join(node[key])})")


def _print_full_finding(finding: "check.CheckFinding") -> None:
    """Prints both sides of a finding plus its rationale (the full new-finding block)."""
    _print_check_finding_side(finding.proposal_node)
    _print_check_finding_side(finding.partner_node)
    print("      Why they may not both stand:")
    print(f"        {finding.rationale}   (confidence {finding.confidence:.2f})")


def _check_degradation_summary(degradations: Tuple[str, ...]) -> str:
    """Renders the degradation tokens as calm human wording (KD4 — from tokens, never
    by re-parsing ``degraded_reason``)."""
    words = {
        "sweep": "the corpus sweep degraded mid-run",
        "judgment": "the judgment stage could not complete",
        "reuse_read": "prior-verdict history was unreadable (findings shown unpartitioned)",
        "telemetry_write": "some per-batch results could not be recorded",
        "stale_index": "the vector index is behind (recall may be thinned)",
        "probe_read": "completeness could not be certified (the index probe was unreadable)",
    }
    return "; ".join(words[token] for token in degradations)


def _print_check_report(
    result: "check.CheckRunResult",
    *,
    exclusions: List[Dict[str, Any]],
    denominator: Optional[int],
    scope: Optional[str],
    row_written: bool,
    transient_count: int,
) -> None:
    """Renders the human report to stdout (findings + disposition), calm ASCII (P9).

    All wording lives HERE (the surface); the engine renders nothing. Findings are
    partitioned by the already-derived ``novelty`` (never re-derived): new findings
    print in full with the resolution pointer, standing (known) findings ride a
    compact section under the index-pinned ``standing (previously reported)`` label,
    and unpartitioned findings (novelty unknown — only under a reuse-read failure)
    get their own labeled section rather than reading as new.
    """
    degradations = check.run_degradations(result)
    new = [f for f in result.findings if f.novelty == "new"]
    known = [f for f in result.findings if f.novelty == "known"]
    unpartitioned = [f for f in result.findings if f.novelty is None]

    if new:
        noun = "contradiction" if len(new) == 1 else "contradictions"
        print(f"\n[Conflict] {len(new)} new {noun} — these decisions may not both stand:")
        for finding in new:
            _print_full_finding(finding)
        print("  Resolve by declaring a relationship in decisions.md "
              "(Supersedes: / Amends: / Narrows: / Contradicts:), then re-sync.")

    if unpartitioned:
        print("\nfindings (history unavailable — unpartitioned):")
        for finding in unpartitioned:
            _print_full_finding(finding)

    if known:
        print("\nstanding (previously reported):")
        for finding in known:
            a, b = finding.proposal_node["slug"], finding.partner_node["slug"]
            print(f"  {a} — {b}   (confidence {finding.confidence:.2f}, "
                  f"first reported {finding.source_created_at})")

    if degradations:
        print(f"\n[partial] This check could not fully run "
              f"({_check_degradation_summary(degradations)}).")
        print(f"  Swept {result.nodes_swept} of {result.nodes_total} decisions; any "
              f"findings above are labeled partial, not certified complete.")

    if not row_written:
        print("  Note: this run was not recorded to check history "
              "(the summary row could not be written).")

    if exclusions:
        print("\nCoverage exclusions (chronically un-embedded — NOT audited):")
        for item in exclusions:
            print(f"  - {item['slug'] or item['id']}")
        print("  These keep failing to embed; the durable fix is outbox quarantine "
              "(substrate-owned). Re-run `mitos sync` to retry.")

    if transient_count:
        print(f"\n{transient_count} decision(s) are behind the vector index — recall "
              f"may be thinned. Run `mitos sync` to catch up.")

    if not result.findings and not degradations:
        if scope is not None and result.nodes_total == 0:
            print(f"0 of {denominator} live decisions match scope '{scope}' — "
                  f"nothing audited.")
        elif result.nodes_total == 0:
            print("No decisions to audit — the corpus is empty.")
        else:
            noun = "decision" if result.nodes_swept == 1 else "decisions"
            print(f"Corpus coherent — {result.nodes_swept} {noun} audited, "
                  f"no contradictions found.")


def cmd_check(
    config: MitosConfig,
    *,
    staged: bool = False,
    scope: Optional[str],
    fresh: bool,
    assume_yes: bool,
    as_json: bool,
) -> int:
    """Audits the live corpus for undeclared contradictions (read-only) → exit 0/1/2.

    The one sequence (each step's contract in the phase plan §4): build substrate →
    provider-absent disposition (KD2) → ``plan_corpus_check`` → CHK-D5 confirm (KD3)
    → build judge iff fresh groups (KD6) → ``execute_corpus_check`` → build the full
    display model → the run-end seam (``exit_code_for`` → row → ``record_check_run``
    LAST, KD5) → emit. ``cmd_check`` owns its error boundary (KD1a): store faults
    around plan/execute/display map to a calm exit-2 vector message, never a traceback
    read by CI as "new findings".

    Args:
        config: The active workspace config.
        staged: Gate the pending buffer instead of sweeping the live corpus (the
            pre-commit / CI gate mode — a self-contained sequence, Phase 3b).
        scope: Optional scope tag filtering the audited (proposal) set (candidate
            recall stays scope-blind, CONF-D2).
        fresh: Re-judge every pair, bypassing verdict reuse (never the novelty read).
        assume_yes: Waive the CHK-D5 spend confirm (the opt-in on every surface).
        as_json: Emit one machine-readable object via :func:`_emit_json` (never prompts).

    Returns:
        ``0`` clean or known-only, ``1`` a NEW contradiction, ``2`` degraded, refused,
        or could-not-run.
    """
    # Flag-combo guard (staged §4 step 1) — pure, pre-store: staged never reuses
    # (nothing to bypass) and always gates the whole pending buffer (no proposal-set
    # filter), so `--scope`/`--fresh` are invocation errors, rejected before any store
    # contact. argparse can't express this (both flags are valid alone).
    if staged and (scope is not None or fresh):
        msg = ("check --staged cannot combine with --scope or --fresh — the gate always "
               "checks the whole pending buffer and never reuses verdicts.")
        if as_json:
            _emit_json({"error": msg, "code": "invalid_flags"})
        else:
            print(msg, file=sys.stderr)
        return 2
    # The gate is a self-contained sequence (its own error boundary); 3a's corpus body
    # below is untouched so its exit contract cannot regress.
    if staged:
        return _run_staged_check(config, assume_yes=assume_yes, as_json=as_json)

    # Lazy at entry (KD6): the alias is needed at PLAN time, but importing
    # `conflict_judgment` module-scope would drag `anthropic` onto every other verb.
    from mitos.conflict_judgment import _JUDGMENT_MODEL_ALIAS

    try:
        store = GraphStore(config.db_path)
        embed, vector, embed_detail, vector_detail = _build_check_substrate(config)
        telemetry = _build_check_telemetry(config)

        # KD2 — provider-absent disposition keys on whether the run has sweep work.
        if embed is None or vector is None:
            active = store.get_active_decisions(scope)
            if active:
                parts: List[str] = []
                if embed is None:
                    parts.append(f"embeddings ({embed_detail})")
                if vector is None:
                    parts.append(f"vector store ({vector_detail})")
                msg = (f"check could not run: cannot audit {len(active)} live "
                       f"decision(s) — {' and '.join(parts)} unavailable.")
                if as_json:
                    _emit_json({"error": msg, "code": "substrate_unavailable"})
                else:
                    print(msg, file=sys.stderr)
                return 2
            # Empty snapshot → the providers are never touched (iter_sweep is lazy
            # over zero nodes); fall through to the one healthy-empty engine path.

        plan = check.plan_corpus_check(
            store=store,
            embed_provider=embed,
            vector_store=vector,
            telemetry=telemetry,
            model_alias=_JUDGMENT_MODEL_ALIAS,
            scope=scope,
            fresh=fresh,
        )

        # CHK-D5 confirm (KD3) — strictly above the threshold; all refusals exit 2,
        # zero spend, no row. The shared `_confirm_spend` helper (KD4) is byte-identical
        # for corpus and staged, so the `>` comparison can never fork between the modes.
        refusal = _confirm_spend(
            len(plan.fresh_groups), assume_yes=assume_yes, as_json=as_json
        )
        if refusal is not None:
            return refusal

        # Build the judge only after the confirm passes and only when there is fresh
        # work (KD6): a reuse-only/clean run never constructs a client.
        judge = _build_check_judge() if plan.fresh_groups else None
        result = check.execute_corpus_check(
            plan, judge=judge, telemetry=telemetry, store=store
        )

        # The full display model — every remaining store read happens HERE, before
        # the write seam (KD5).
        exclusions = _resolve_exclusion_display(
            store, check.coverage_exclusion_ids(result)
        )
        transient_count = _check_transient_count(result)
        denominator: Optional[int] = None
        if scope is not None and plan.nodes_total == 0:
            # The zero-match denominator needs the unscoped live count the plan does
            # not carry (plan.nodes_total is already scope-filtered).
            denominator = len(store.get_active_decisions())

        # Run-end seam (KD5): exit → row → write LAST. Build the row unconditionally
        # (pure) so the JSON scalars derive from the one source; write it only when
        # telemetry exists.
        exit_code = check.exit_code_for(result)
        row = check.check_run_row_from_result(result, mode="corpus", exit_code=exit_code)
        row_written = False
        if telemetry is not None:
            try:
                telemetry.record_check_run(row)
                row_written = True
            except DatabaseError:
                # The write is the last fallible act: a failure only moves toward 2.
                exit_code = 2
        else:
            # telemetry None is already exit 2 via reuse_read; the no-row disclosure
            # is additive, not a second exit driver.
            exit_code = 2
    except (sqlite3.Error, DatabaseError, MitosError) as exc:
        # KD1a — the verb owns its boundary: a store fault is exit 2 with a calm
        # vector message, never a traceback CI would read as "new findings".
        msg = f"check could not run: {exc}"
        if as_json:
            _emit_json({"error": msg, "code": "check_faulted"})
        else:
            print(msg, file=sys.stderr)
        return 2

    # Emission is pure (out of the write contract): one JSON object or the report.
    if as_json:
        _emit_json(_check_json_object(
            result, row, exclusions=exclusions, exit_code=exit_code,
            row_written=row_written, scope=scope, fresh=fresh,
            transient_count=transient_count,
        ))
    else:
        _print_check_report(
            result, exclusions=exclusions, denominator=denominator, scope=scope,
            row_written=row_written, transient_count=transient_count,
        )
    return exit_code


# =========================================================================== #
# Phase 3b — `mitos check --staged`: the pre-commit / CI gate mode.
#
# The proactive half of the `check` verb: it gates the PENDING (not-yet-committed)
# decision entries of the working-tree `decisions.md` and fails CLOSED — a pending
# undeclared contradiction blocks the commit (exit 1), a clean buffer passes (0),
# a gate that cannot run says so (exit 2) rather than a silent pass. Self-contained
# (its own sequence + error boundary) so 3a's corpus contract cannot regress. The
# load-bearing rules — the exit table (§3), the pure-read predicate (KD1: no graph
# write), the no-row-unless-judged rule (KD2/KD8), the `surface='check'` attribution
# (KD7), `_emit_json`-only — live in the phase plan.
# =========================================================================== #


def _pending_decision_entries(
    store: GraphStore, entries: List[ParsedEntry]
) -> List[ParsedEntry]:
    """Selects the pending decision entries via the sync idempotency predicate (KD1).

    A pure READ: an entry is pending iff its slug-free canonical-core content hash is
    not yet a committed node. Replicates sync.py:650-656's ``compute_node_id`` call and
    the ``get_node`` test — but deliberately NOT the ``note_source_reencounter`` write
    two lines below it (sync.py:668): a gate that mutates the graph while gating it is a
    contradiction of its own. OQ entries are skipped (the facade is decision-only).

    Args:
        store: The graph store — touched only through ``get_node`` (read).
        entries: The parsed working-tree entries (already all ``decision`` kind when
            they come from ``parse_entry_stream(text, "decision")``; the filter is a
            harmless safety belt).

    Returns:
        The pending decision entries, in parse order.
    """
    pending: List[ParsedEntry] = []
    for entry in entries:
        if entry.kind != "decision":
            continue
        node_id = compute_node_id(
            kind=entry.kind,
            axiom=entry.axiom,
            mechanism_refs=entry.mechanisms,
            topic=entry.topic,
            questions_raised=entry.questions_raised,
        )
        if store.get_node(node_id) is None:
            pending.append(entry)
    return pending


def _persist_staged_batch(
    telemetry: Optional[TelemetryStore],
    result: "Any",
    *,
    run_id: str,
) -> Optional[str]:
    """Persists one judged staged batch with ``surface='check'`` (KD7), best-effort.

    Mirrors ``sync._persist_conflict_batch``'s ``ConflictCheckResult`` → ``(JudgmentBatch,
    [ConflictCheckRow])`` mapping (sync.py:1196-1239) verbatim EXCEPT for two data-level
    values: ``surface='check'`` (this is the check surface, not sync) and ``sync_run_id``
    = this run's id. Every fed-context field is read off the result's ``JudgeInput``\\ s
    (what the judge saw), never a node re-read. The MI-9 ``""→None`` proposal/candidate
    scope + rejected coercions are load-bearing.

    Args:
        telemetry: The run's telemetry store, or ``None`` (a judged run needs it — a
            ``None`` store is a write failure the caller degrades on, KD7).
        result: A judged :class:`~mitos.conflict.ConflictCheckResult` (``execution`` set;
            the caller guards ``execution is not None``).
        run_id: This run's id, stamped as ``sync_run_id`` on every row (the one-thread-of-
            truth join to the ``check_runs`` PK).

    Returns:
        ``None`` on a clean write, or a write-failure detail string (the caller marks the
        run degraded and reports it) — never raising, so one bad batch never crashes the gate.
    """
    if telemetry is None:
        return "telemetry store unavailable"
    try:
        execution = result.execution
        # CHK-D3: resolve the versioned model id here (same process/env, moments after
        # the call); an unknown alias degrades to NULL (provenance-only), never a lost row.
        try:
            model_id: Optional[str] = get_model_id(execution.model_alias)
        except ValueError:
            model_id = None
        batch = JudgmentBatch(
            batch_id=execution.batch_id,
            model_id=model_id,
            token_input=execution.token_input,
            token_output=execution.token_output,
            token_cache_read=execution.token_cache_read,
            token_cache_creation=execution.token_cache_creation,
            elapsed_ms=execution.elapsed_ms,
        )
        proposal = result.proposal_input
        rows: List[ConflictCheckRow] = []
        for pair in result.judged_pairs:
            candidate_input = pair.candidate_input
            rows.append(
                ConflictCheckRow(
                    batch_id=execution.batch_id,
                    sync_run_id=run_id,
                    # The staged difference from the sync mapper: this IS the check
                    # surface, stamped explicitly (CHK-D7), never the schema DEFAULT.
                    surface="check",
                    judged_axiom=proposal.axiom,
                    proposal_rejected_paths=proposal.rejected_paths or None,
                    proposal_scope=", ".join(proposal.scope) or None,
                    proposed_hash_if_any=result.proposed_hash_if_any,
                    candidate_slug=pair.candidate.slug,
                    candidate_hash=pair.candidate.node["id"],
                    candidate_rejected_paths=candidate_input.rejected_paths,
                    candidate_scope=", ".join(candidate_input.scope) or None,
                    tenable=pair.judgment.tenable_together,
                    confidence=pair.judgment.confidence,
                    surfaced=pair.surfaced,
                    candidate_source=check.CONFLICT_CANDIDATE_SOURCE,
                    model_alias=execution.model_alias,
                    prompt_version=check.CONFLICT_PROMPT_VERSION,
                    mitos_version=__version__,
                    rationale=pair.judgment.rationale,
                )
            )
        telemetry.record_judged_batch(
            batch, rows, datetime.now(timezone.utc).isoformat()
        )
        return None
    except (sqlite3.Error, DatabaseError, TypeError, ValueError) as exc:
        # Best-effort (KD7): a mapping/write failure degrades the run, never crashes it.
        return str(exc)


# The staged degradation vocabulary — a subset of the corpus tokens that a gate can
# reach (KD8). Rendered to calm human wording; the raw tokens ride the `--json`.
_STAGED_DEGRADATION_WORDS = {
    "stale_index": "the vector index is behind (recall may be thinned)",
    "sweep": "the semantic substrate went dark mid-run (findings shown are partial)",
    "judgment": "the judge became unavailable mid-run (findings shown are partial)",
    "telemetry_write": "some results could not be recorded",
}


def _staged_finding_json(
    entry: ParsedEntry, proposed_hash: str, finding: "Any", partner_hash: Optional[str]
) -> Dict[str, Any]:
    """Renders one staged finding as its §8 JSON object — both sides named (KD9).

    Every staged finding is ``novelty:"new"`` (the gate never partitions, CHK-D10). The
    proposal side is the pending entry (not yet a node — no id-in-graph, no modifier
    stamps); the partner side is the facade's candidate ``payload`` (a Letter render) plus
    the candidate content hash resolved from ``judged_pairs``.
    """
    payload = finding.payload
    partner: Dict[str, Any] = {"id": partner_hash}
    for key in ("slug", "axiom", "scope", "rejected_paths"):
        if key in payload:
            partner[key] = payload[key]
    for key in _CHECK_MODIFIER_STAMP_KEYS:
        if key in payload:
            partner[key] = payload[key]
    return {
        "novelty": "new",
        "confidence": finding.confidence,
        "rationale": finding.rationale,
        "score": payload.get("score"),
        "proposal": {
            "id": proposed_hash,
            "slug": entry.slug,
            "axiom": entry.axiom,
            "scope": list(entry.scope),
            "rejected_paths": entry.rejected_paths,
        },
        "partner": partner,
    }


def _print_staged_finding(entry: ParsedEntry, finding: "Any") -> None:
    """Prints both sides of one staged finding plus its rationale (calm ASCII, P9)."""
    print(f"  Pending entry '{entry.slug}':")
    print(f"      Axiom:    {entry.axiom}")
    scope_text = ", ".join(entry.scope) if entry.scope else "(global — no scope declared)"
    print(f"      Scope:    {scope_text}")
    if entry.rejected_paths:
        print(f"      Rejected: {entry.rejected_paths}")
    payload = finding.payload
    print(f"  conflicts with active decision '{payload['slug']}'   "
          f"(similarity {payload['score']:.2f}):")
    print(f"      Axiom:    {payload['axiom']}")
    p_scope = payload.get("scope") or []
    p_scope_text = ", ".join(p_scope) if p_scope else "(global — no scope declared)"
    print(f"      Scope:    {p_scope_text}")
    if "rejected_paths" in payload:
        print(f"      Rejected: {payload['rejected_paths']}")
    for key in _CHECK_MODIFIER_STAMP_KEYS:
        if key in payload:
            print(f"      ({key.replace('_', ' ')}: {', '.join(payload[key])})")
    print("      Why they may not both stand:")
    print(f"        {finding.rationale}   (confidence {finding.confidence:.2f})")


def _print_staged_report(
    findings: List[Tuple[ParsedEntry, str, "Any", Optional[str]]],
    *,
    nodes_swept: int,
    nodes_total: int,
    degraded: "Set[str]",
    exclusions: List[Dict[str, Any]],
    transient_count: int,
) -> None:
    """Renders the human gate report to stdout — calm, both sides named (P9)."""
    if findings:
        noun = "contradiction" if len(findings) == 1 else "contradictions"
        print(f"\n[Conflict] {len(findings)} pending {noun} — these decisions may "
              f"not both stand:")
        for entry, _hash, finding, _pid in findings:
            _print_staged_finding(entry, finding)
        print("  Resolve by declaring a relationship in decisions.md "
              "(Supersedes: / Amends: / Narrows: / Contradicts:) before committing, "
              "or `git commit --no-verify` to bypass the gate deliberately.")

    if degraded:
        summary = "; ".join(
            _STAGED_DEGRADATION_WORDS[t] for t in sorted(degraded)
            if t in _STAGED_DEGRADATION_WORDS
        )
        print(f"\n[partial] This gate could not fully run ({summary}).")
        print(f"  Checked {nodes_swept} of {nodes_total} pending decision(s); any "
              f"findings above are partial, not certified complete.")

    if exclusions:
        print("\nCoverage exclusions (chronically un-embedded — NOT audited):")
        for item in exclusions:
            print(f"  - {item['slug'] or item['id']}")
        print("  These keep failing to embed; run `mitos sync` to retry (the durable "
              "fix is outbox quarantine, substrate-owned).")

    if transient_count:
        print(f"\n{transient_count} decision(s) are behind the vector index — recall "
              f"may be thinned. Run `mitos sync` to catch up.")

    if not findings and not degraded:
        noun = "decision" if nodes_swept == 1 else "decisions"
        print(f"Gate clear — {nodes_swept} pending {noun} checked, "
              f"no contradictions found.")


def _run_staged_check(
    config: MitosConfig, *, assume_yes: bool, as_json: bool
) -> int:
    """Gates the pending decision buffer, fail-closed → exit 0/1/2 (Phase 3b).

    The one sequence (§4): parse the working-tree ``decisions.md`` → select pending via
    the pure-read predicate (KD1) → no-pending short-circuit (exit 0, no probe/substrate/
    row, KD2) → build substrate (absent + pending ⇒ fail-closed exit 2) → start probe
    (KD3) → CHK-D5 confirm (KD4) → build judge (absent + pending ⇒ fail-closed exit 2,
    KD5) → per-entry facade loop with the aggregate breaker (KD6) → exit derivation +
    the hand-built ``mode='staged'`` row written LAST, only when a judgment fired (KD8).
    Owns its own error boundary: a store/parse fault is exit 2 with a calm vector message,
    never a traceback CI reads as "new findings".

    Args:
        config: The active workspace config (paths).
        assume_yes: Waive the CHK-D5 spend confirm.
        as_json: Emit one machine-readable object via :func:`_emit_json` (never prompts).

    Returns:
        ``0`` clean / no pending, ``1`` a pending contradiction, ``2`` degraded, refused,
        or could-not-gate.
    """
    run_id = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        store = GraphStore(config.db_path)
        # Parse the WORKING-TREE decisions.md (git-agnostic; absent file → no pending).
        # `parse_entry_stream` STRICT mode raises ParseError(MitosError) on a malformed
        # buffer → the boundary below maps it to exit 2 (fail-closed). Never pass a
        # `failures=` collector (that would silently isolate a bad entry).
        text = read_text_or_none(config.decisions_file)
        entries = parse_entry_stream(text, "decision") if text else []
        pending = _pending_decision_entries(store, entries)

        # KD2 — no-pending short-circuit: exit 0 with zero LLM contact, no probe, no
        # substrate build, no row. The overwhelmingly common commit is free.
        if not pending:
            ended_at = datetime.now(timezone.utc).isoformat()
            if as_json:
                _emit_json(_staged_json_object(
                    run_id=run_id, started_at=started_at, ended_at=ended_at,
                    exit_code=0, nodes_total=0, nodes_swept=0, batches_executed=0,
                    pairs_judged_fresh=0, finding_objs=[], degradations=[],
                    exclusions=[], transient_count=0, row_written=False,
                ))
            else:
                print("Gate clear — no pending decisions to check.")
            return 0

        embed, vector, embed_detail, vector_detail = _build_check_substrate(config)
        # Providers absent WITH pending work → fail-closed (the hook precondition), no row.
        if embed is None or vector is None:
            parts: List[str] = []
            if embed is None:
                parts.append(f"embeddings ({embed_detail})")
            if vector is None:
                parts.append(f"vector store ({vector_detail})")
            msg = (f"check --staged could not gate {len(pending)} pending decision(s) — "
                   f"{' and '.join(parts)} unavailable.")
            if as_json:
                _emit_json({"error": msg, "code": "substrate_unavailable"})
            else:
                print(msg, file=sys.stderr)
            return 2

        telemetry = _build_check_telemetry(config)

        # KD3 — start probe only (no end probe: staged judges a fixed buffer, not a
        # sweep). A probe fault propagates to the boundary. A transient backlog gates
        # partial (exit 2) but does NOT skip the judgment; over-tolerance rows are
        # disclosed coverage exclusions that never gate (the poison-row escape).
        start_probe = check.probe_stale_index(store)
        transient_count = len({row.node_id for row in start_probe.transient})
        degraded: Set[str] = set()
        if transient_count:
            degraded.add("stale_index")

        # KD4 — the shared confirm on the pending count. All refusals exit 2, no row.
        refusal = _confirm_spend(len(pending), assume_yes=assume_yes, as_json=as_json)
        if refusal is not None:
            return refusal

        # KD5 — the judge is required to gate real work: `run_conflict_check` CALLS it
        # (unlike corpus, which absorbs `judge=None` as a typed degradation), so a
        # missing key with pending entries is fail-closed exit 2, no row. `--no-verify`
        # is the deliberate human bypass (documented in 4b).
        judge = _build_check_judge()
        if judge is None:
            msg = (f"check --staged could not gate {len(pending)} pending decision(s) — "
                   f"ANTHROPIC_API_KEY is not set (the judge is required to gate).")
            if as_json:
                _emit_json({"error": msg, "code": "judge_unavailable"})
            else:
                print(msg, file=sys.stderr)
            return 2

        # KD6 — per-entry facade loop, verbatim + sequential, aggregate breaker on the
        # first typed `Unavailable` (one penalty, not N). A genuine local store fault
        # RAISES past the facade → the boundary below (prior findings lost, the rare case).
        findings: List[Tuple[ParsedEntry, str, "Any", Optional[str]]] = []
        nodes_swept = 0
        pairs_judged_fresh = 0
        batches_executed = 0
        for entry in pending:
            result = run_conflict_check(
                entry, embed_provider=embed, vector_store=vector, store=store, judge=judge
            )
            if isinstance(result, check.Unavailable):
                # Trip the breaker: stop calling the facade for the remaining entries.
                # The token is faithful to WHICH downstream went dark (aligning with the
                # corpus P18 vocabulary): the semantic substrate (embedding/vector) reads
                # as `sweep`, the judge as `judgment`.
                degraded.add(
                    "sweep" if result.reason in (
                        ConflictUnavailableReason.EMBEDDING,
                        ConflictUnavailableReason.VECTOR_STORE,
                    ) else "judgment"
                )
                break
            nodes_swept += 1
            pairs_judged_fresh += len(result.judged_pairs)
            if result.execution is not None:
                batches_executed += 1
                detail = _persist_staged_batch(telemetry, result, run_id=run_id)
                if detail is not None:
                    degraded.add("telemetry_write")
            # Map each surfaced finding to its candidate content hash (from judged_pairs).
            hash_by_slug = {
                pair.candidate.slug: pair.candidate.node["id"]
                for pair in result.judged_pairs
            }
            for finding in result.findings:
                findings.append((
                    entry, result.proposed_hash_if_any, finding,
                    hash_by_slug.get(finding.slug),
                ))

        exclusions = _resolve_exclusion_display(
            store, tuple(row.node_id for row in start_probe.excluded)
        )

        # KD8 — exit derivation (degraded 2 dominates finding 1 dominates clean 0), then
        # the hand-built row, written LAST and only when a judgment actually fired.
        exit_code = 2 if degraded else (1 if findings else 0)
        ended_at = datetime.now(timezone.utc).isoformat()
        row_written = False
        if batches_executed > 0:
            row = check.CheckRunRow(
                run_id=run_id,
                mode="staged",
                started_at=started_at,
                ended_at=ended_at,
                exit_code=exit_code,
                nodes_swept=nodes_swept,
                pairs_judged_fresh=pairs_judged_fresh,
                pairs_reused=0,
                findings_new=len(findings),
                findings_known=0,
                coverage_exclusions=len(start_probe.excluded),
                degraded_reason=",".join(sorted(degraded)) or None,
                mitos_version=__version__,
            )
            if telemetry is not None:
                try:
                    telemetry.record_check_run(row)
                    row_written = True
                except DatabaseError:
                    # The write is the last fallible act: a failure only moves toward 2.
                    degraded.add("telemetry_write")
                    exit_code = 2
            else:
                # A judged run whose telemetry could not be built cannot record — degrade.
                degraded.add("telemetry_write")
                exit_code = 2
    except (sqlite3.Error, DatabaseError, MitosError) as exc:
        # The gate's error boundary (mirrors 3a's): a store/parse/probe/facade fault is
        # exit 2 with a calm vector message, never a traceback CI reads as "new findings".
        msg = f"check --staged could not run: {exc}"
        if as_json:
            _emit_json({"error": msg, "code": "check_faulted"})
        else:
            print(msg, file=sys.stderr)
        return 2

    # Emission is pure (out of the write contract): one JSON object or the report.
    if as_json:
        _emit_json(_staged_json_object(
            run_id=run_id, started_at=started_at, ended_at=ended_at,
            exit_code=exit_code, nodes_total=len(pending), nodes_swept=nodes_swept,
            batches_executed=batches_executed, pairs_judged_fresh=pairs_judged_fresh,
            finding_objs=[_staged_finding_json(e, h, f, p) for e, h, f, p in findings],
            degradations=sorted(degraded), exclusions=exclusions,
            transient_count=transient_count, row_written=row_written,
        ))
    else:
        _print_staged_report(
            findings, nodes_swept=nodes_swept, nodes_total=len(pending),
            degraded=degraded, exclusions=exclusions, transient_count=transient_count,
        )
    return exit_code


def _staged_json_object(
    *,
    run_id: str,
    started_at: str,
    ended_at: str,
    exit_code: int,
    nodes_total: int,
    nodes_swept: int,
    batches_executed: int,
    pairs_judged_fresh: int,
    finding_objs: List[Dict[str, Any]],
    degradations: List[str],
    exclusions: List[Dict[str, Any]],
    transient_count: int,
    row_written: bool,
) -> Dict[str, Any]:
    """Assembles the single §8 staged ``--json`` object (a shipped API — additive only).

    The same key set as 3a §8 with staged values (KD9): ``mode:"staged"``, ``scope`` key
    ABSENT (staged never scopes), ``fresh:false``, ``pairs_reused:0``, every finding
    ``novelty:"new"``, ``findings_known:0`` — so a CI consumer's cross-surface invariant
    ``exit_code == 1 ⟺ findings_new > 0`` holds on staged exactly as on corpus. The
    ``batches_*`` accounting reads: planned = one potential batch per pending entry;
    executed = entries that fired the judge; skipped = clean-empty or breaker-skipped
    (so ``planned == executed + skipped`` holds).
    """
    return {
        "run_id": run_id,
        "mode": "staged",
        "exit_code": exit_code,
        "started_at": started_at,
        "ended_at": ended_at,
        "fresh": False,
        "nodes_total": nodes_total,
        "nodes_swept": nodes_swept,
        "pairs_judged_fresh": pairs_judged_fresh,
        "pairs_reused": 0,
        "batches_planned": nodes_total,
        "batches_executed": batches_executed,
        "batches_skipped": nodes_total - batches_executed,
        "findings": finding_objs,
        "findings_new": len(finding_objs),
        "findings_known": 0,
        "degradations": degradations,
        "coverage_exclusions": exclusions,
        "index_backlog_transient": transient_count,
        "summary_row_written": row_written,
    }


def _enter_target_directory(directory: Optional[str]) -> None:
    """chdir into a -C/--directory target, or no-op when none was given.

    Git's ``-C`` semantics at the CLI boundary: the chdir runs once at process
    entry, before any env load, config construction, or arg-driven file open, so
    the whole workspace (graph, collection, ``.env``/keys, relative path args)
    retargets at once — each downstream site derives from the process CWD.

    Args:
        directory: The ``-C``/``--directory`` value, or None when the flag was
            absent (then this is a no-op and the launch CWD is unchanged).

    Raises:
        MitosError: When ``directory`` is given but is not an existing directory
            (a clean P3 error — never a raw OSError traceback). Existence is
            checked against the launch CWD, so a relative ``-C ./sub`` resolves
            where mitos was started.
    """
    if directory is None:
        return
    if not os.path.isdir(directory):
        raise MitosError(f"directory not found: {directory}")
    os.chdir(directory)


def main() -> None:
    """Main CLI execution router."""
    # Make raw-text print()s crash-safe on a non-UTF-8 stdout before any verb
    # can print (R6). Inert on a UTF-8 terminal; CLI-only — the MCP transport
    # has no terminal stdout to harden (P7 bulkhead).
    apply_stdout_text_safety(sys.stdout)
    parser = argparse.ArgumentParser(
        description="Mitos: Architectural Decision Substrate for LLM-native workflows.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"mitos {__version__}")
    parser.add_argument(
        "-C", "--directory", dest="directory", default=None, metavar="DIR",
        help="Run as if mitos were started in DIR (git's -C). Retargets the whole "
             "workspace — graph, collection, .env/keys, and relative path args. "
             "Must appear BEFORE the verb: `mitos -C /ws list`.",
    )
    # metavar collapses the width-doubling {init,sync,query,query_decisions,…}
    # brace-list in the usage banner to a single COMMAND token (R5). This is a
    # render-only hint — it structurally cannot unregister an alias, so every
    # `aliases=[...]` verb below stays callable while absent from the banner.
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # init
    subparsers.add_parser("init", help="Initialize Mitos in current workspace.")

    # sync
    sync_p = subparsers.add_parser("sync", help="Sync buffer decisions to graph database.")
    sync_p.add_argument("--yes", action="store_true", help="Auto-accept all parsed changes.")
    sync_p.add_argument("--embed-only", action="store_true", help="Drain the pending embeddings outbox queue only.")
    sync_p.add_argument("--verbose", action="store_true", help="Show verbose cache statistics.")

    # reconcile
    rec_p = subparsers.add_parser(
        "reconcile",
        help="Re-embed active nodes missing from Qdrant (heal a direct vector wipe).",
    )
    rec_p.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON.")

    # capture
    cap_p = subparsers.add_parser("capture", help="Synthesize and append a decision.")
    cap_p.add_argument("text", help="Raw decision description.")

    # query (alias: query_decisions — MCP tool name)
    q_p = subparsers.add_parser("query", aliases=["query_decisions"], help="Semantic lookup for precedents.")
    q_p.add_argument("claim", help="Assertion or subsystem query.")
    q_p.add_argument("--depth", default="letter", help="Depth (default: letter).")
    q_p.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON.")
    q_p.add_argument("--brief", action="store_true", help="Axiom-only (omit rejected_paths) — a quick scan.")
    q_p.add_argument("--limit", type=int, default=None,
                     help="Set ranked top-k to retrieve (1–50; default 5). Raises or lowers the count — a context-budget dial.")

    # surface (alias: surface_decisions — MCP tool name) — the precedent-recall loop
    surf_p = subparsers.add_parser("surface", aliases=["surface_decisions"],
                                   help="Surface active decisions relevant to a query (precedent check before deciding).")
    surf_p.add_argument("query", help="The claim or topic to find precedents for.")
    surf_p.add_argument("--scope", default=None, help="Optional scope hint (does NOT filter semantic recall — scopes open-questions + note only). Use `list --scope` to hard-filter by scope.")
    surf_p.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON.")
    surf_p.add_argument("--brief", action="store_true", help="Axiom-only (omit rejected_paths) — a quick scan.")
    surf_p.add_argument("--limit", type=int, default=None,
                        help="Set ranked top-k to retrieve (1–50; default 5). Raises or lowers the count — a context-budget dial.")

    # show
    show_p = subparsers.add_parser("show", help="Display details of a specific node.")
    show_p.add_argument("ident", help="Slug or ID of node.")
    show_p.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON (for agents).")

    # list (alias: list_decisions — the MCP tool name, so an agent's first instinct works)
    list_p = subparsers.add_parser("list", aliases=["list_decisions"],
                                   help="Enumerate the complete set of decisions in a scope (exhaustive recall).")
    list_p.add_argument("--scope", help="Filter by scope tag.")
    list_p.add_argument("--state", help="Computed state filter: 'active' (default, live set), 'all', or an exact state.")
    list_p.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON (for agents).")
    # brief and oneline are depth tiers below the Letter-complete default — pick one.
    list_depth = list_p.add_mutually_exclusive_group()
    list_depth.add_argument("--brief", action="store_true", help="Axiom-only (omit rejected_paths) — lighter over a big scope.")
    list_depth.add_argument("--oneline", action="store_true",
                            help="One row per decision: slug + truncated axiom — the orientation tier for big scopes (modifier markers kept).")

    # open-questions
    oq_p = subparsers.add_parser("open-questions", help="List active open questions.")
    oq_p.add_argument("--scope", help="Filter by scope tag.")
    oq_p.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON.")

    # scopes (alias: list_scopes — the MCP tool name, so an agent's first instinct works)
    scopes_p = subparsers.add_parser("scopes", aliases=["list_scopes"],
                                     help="Enumerate the scope vocabulary with live-node counts (busiest first).")
    scopes_p.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON (for agents).")
    scopes_p.add_argument("--archived", action="store_true", dest="archived",
                          help="Include fully-dead domains at a 0/0 floor (scope-level 'list --state all').")

    # import
    imp_p = subparsers.add_parser("import", help="Import legacy prose ADR.")
    imp_p.add_argument("path", help="Path to markdown prose file.")
    imp_p.add_argument("--from", dest="import_from", default="prose", help="Import source kind.")
    imp_p.add_argument("--llm-extract", action="store_true", help="Use Sonnet compression pass.")

    # render
    ren_p = subparsers.add_parser("render", help="Regenerate rendered outputs.")
    ren_p.add_argument("--format", default="live-axioms", help="Target format.")
    ren_p.add_argument("--scope", help="Optional scope filter.")

    # record (alias: record_decision — the MCP tool name, so an agent's first instinct works)
    rec_p = subparsers.add_parser("record", aliases=["record_decision"], help="Record a decision directly to buffer and graph.")
    rec_p.add_argument("axiom", nargs="?", default=None,
                       help="The decision as a single clear sentence true going forward "
                            "(or use --axiom-file; exactly one of the two).")
    rec_p.add_argument("--axiom-file", default=None, dest="axiom_file",
                       help="Read the axiom from a file ('-' = stdin) to avoid shell-quoting; "
                            "replaces the positional axiom (supply one, not both).")
    rec_p.add_argument("--rejected", default=None, help="Alternatives considered and rejected, and why (REQUIRED — or use --rejected-file).")
    rec_p.add_argument("--rejected-file", default=None, dest="rejected_file",
                       help="Read --rejected from a file ('-' = stdin) to avoid shell-quoting long prose.")
    rec_p.add_argument("--scope", nargs="*", default=[], help="Area tags, e.g. --scope database auth.")
    rec_p.add_argument("--mechanisms", nargs="*", default=None, help="Concrete technologies/entities, e.g. --mechanisms sqlite wal-mode.")
    rec_p.add_argument("--context", default=None, help="Optional background on why this was decided.")
    rec_p.add_argument("--context-file", default=None, dest="context_file",
                       help="Read --context from a file ('-' = stdin).")
    rec_p.add_argument("--supersedes", default=None, help="Exact slug(s) of prior decision(s) this one replaces — comma-separated for several (e.g. 'a, b').")
    rec_p.add_argument("--corrects", default=None, help="Exact slug(s) of prior decision(s) this one corrects (kill-edge twin of --supersedes) — comma-separated for several.")
    rec_p.add_argument("--amends", default=None, help="Exact slug(s) of decision(s) this one amends — comma-separated for several.")
    rec_p.add_argument("--narrows", default=None, help="Exact slug(s) of decision(s) this one narrows — comma-separated for several.")
    rec_p.add_argument("--depends-on", default=None, dest="depends_on", help="Exact slug(s) of decision(s) this one depends on — comma-separated for several.")
    rec_p.add_argument("--resolves", default=None, help="Exact slug(s) of open question(s) this one resolves (resolves is decision→open-question only) — comma-separated for several.")
    rec_p.add_argument("--contradicts", default=None, help="Exact slug(s) of decision(s) this one contradicts — comma-separated for several.")
    rec_p.add_argument("--derives-from", default=None, dest="derives_from", help="Not valid when recording a decision — a derives_from edge originates from an open question (open_question -> decision), so a decision cannot be its source. Use --cites to link a decision this one builds on.")
    rec_p.add_argument("--cites", default=None, help="Exact slug(s) of decision(s) this one cites — comma-separated for several.")
    rec_p.add_argument("--slug", required=True,
                       help=f"Explicit slug (handle) for the decision, required "
                            f"(≤{_SLUG_MAX_LEN} chars; an over-length slug is rejected, not truncated).")
    rec_p.add_argument("--acknowledge-neighbors", action="store_true", dest="acknowledge_neighbors",
                       help="Record past the near-duplicate review (the decision is genuinely independent).")
    rec_p.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON.")

    # serve
    subparsers.add_parser("serve", help="Launch Mitos FastMCP server on stdio.")

    # status — is Mitos set up for this project? (human- and LLM-friendly check)
    status_p = subparsers.add_parser("status", help="Check whether Mitos is set up for a project.")
    status_p.add_argument("path", nargs="?", default=None, help="Project directory to check (default: current directory).")
    status_p.add_argument("--json", action="store_true", dest="as_json", help="Emit a machine-readable JSON report.")

    # set-key — store an API key globally (all projects) or for this project
    sk_p = subparsers.add_parser("set-key", help="Store an API key globally (all projects) or for this project.")
    sk_p.add_argument("value", help="The API key value.")
    sk_p.add_argument("--name", default="GEMINI_API_KEY", help="Env var name to store (default: GEMINI_API_KEY).")
    sk_p.add_argument("--global", action="store_true", dest="is_global",
                      help="Write the global ~/.config/mitos/.env (shared by ALL projects) instead of this project's .env.")

    # cutover — the one-time prototype→V1a migration (destructive; operator-run).
    cut_p = subparsers.add_parser(
        "cutover",
        help="One-time migration of a prototype graph to the V1a store (destructive).")
    cut_p.add_argument("--allow-drops", action="store_true", dest="allow_drops",
                       help="Proceed even if active decisions would be dropped from the "
                            "rebuild (P6: a drop may be a deliberate purge).")
    cut_p.add_argument("--yes", action="store_true",
                       help="Skip the interactive confirmation (automation / non-TTY).")
    cut_p.add_argument("--json", action="store_true", dest="as_json",
                       help="Emit a machine-readable JSON report.")

    rebuild_p = subparsers.add_parser(
        "rebuild",
        help="Rebuild the graph from the full corpus through the current catalog "
             "(e.g. after a 0.3.x→0.4.0 upgrade to populate the new edges + mechanisms).")
    rebuild_p.add_argument("--allow-drops", action="store_true", dest="allow_drops",
                           help="Proceed even if entries cannot be rebuilt or active "
                                "decisions would be dropped (the markdown stays the "
                                "source of truth; a drop may be deliberate).")
    rebuild_p.add_argument("--yes", action="store_true",
                           help="Skip the interactive confirmation (automation / non-TTY).")
    rebuild_p.add_argument("--json", action="store_true", dest="as_json",
                           help="Emit a machine-readable JSON report.")

    # check — read-only corpus conflict audit / CI gate (exit 0/1/2).
    check_p = subparsers.add_parser(
        "check",
        help="Audit the live corpus for undeclared contradictions (read-only). "
             "Exit 0 = clean or known-only, 1 = a NEW contradiction, 2 = degraded, "
             "refused, or could not run.")
    check_p.add_argument("--scope", default=None,
                         help="Restrict the audited (proposal) set to one scope tag "
                              "(candidate recall stays scope-blind).")
    check_p.add_argument("--fresh", action="store_true",
                         help="Re-judge every pair, bypassing verdict reuse.")
    check_p.add_argument("--yes", action="store_true",
                         help="Authorize the LLM spend without prompting (the opt-in "
                              "on every non-interactive surface).")
    check_p.add_argument("--json", action="store_true", dest="as_json",
                         help="Emit one machine-readable JSON object (never prompts).")
    check_p.add_argument("--staged", action="store_true",
                         help="Gate the PENDING buffer of decisions.md (the pre-commit / "
                              "CI gate) instead of sweeping the live corpus. Fails closed: "
                              "exit 2 when it cannot run. Not git's staging — reads the "
                              "working-tree decisions.md. Rejects --scope/--fresh.")

    # agent-block — print the canonical agent-file block to paste, or --check pasted copies.
    ab_p = subparsers.add_parser(
        "agent-block",
        help="Print the agent-file block to paste into AGENTS.md/CLAUDE.md/…, or --check for drift.")
    ab_p.add_argument("path", nargs="?", default=None,
                      help="Project directory (default: current directory) — used by --check.")
    ab_p.add_argument("--check", action="store_true",
                      help="Scan the project's agent files and report stale/unversioned mitos notes.")

    args = parser.parse_args()

    try:
        # Constructed INSIDE the try so a strict-loader ConfigError on a malformed
        # `.mitos/config.toml` is caught by `except MitosError` below and rendered
        # as a one-line `Error: …` — not a raw traceback (the 6a raising-loader owns
        # this boundary). The `finally`'s only config read (config.workspace_dir for
        # the MCP hint) is already wrapped in its own `except Exception: pass`, so an
        # unbound `config` after a construction failure stays silent.
        #
        # -C/--directory runs FIRST (before the project .env load + config) so the
        # whole workspace retargets together: chdir into the target, THEN load the
        # CWD-relative project .env, THEN the fixed-path global .env (CWD-independent
        # — it stays global). An absent -C target raises MitosError here and renders
        # through the `except MitosError` boundary as a clean one-line error. The
        # project .env load also sits inside the try now (strictly safer — a .env
        # read failure is caught, not a bare traceback). Precedence is unchanged:
        # load_dotenv_file never overwrites an already-set key, so env > project >
        # global still holds.
        _enter_target_directory(args.directory)
        load_dotenv_file()
        load_dotenv_file(global_env_path())
        config = MitosConfig()
        if args.command == "init":
            cmd_init(config)
        elif args.command == "sync":
            cmd_sync(config, auto_accept=args.yes, embed_only=args.embed_only, verbose=args.verbose)
        elif args.command == "reconcile":
            sys.exit(cmd_reconcile(config, as_json=args.as_json))
        elif args.command == "capture":
            cmd_capture(config, args.text)
        elif args.command in ("query", "query_decisions"):
            cmd_query(config, args.claim, depth=args.depth, as_json=args.as_json, brief=args.brief, limit=args.limit)
        elif args.command in ("surface", "surface_decisions"):
            cmd_surface(config, args.query, scope=args.scope, as_json=args.as_json, brief=args.brief, limit=args.limit)
        elif args.command == "show":
            cmd_show(config, args.ident, as_json=args.as_json)
        elif args.command in ("list", "list_decisions"):
            cmd_list(config, scope=args.scope, state_filter=args.state, as_json=args.as_json,
                     brief=args.brief, oneline=args.oneline)
        elif args.command == "open-questions":
            cmd_open_questions(config, scope=args.scope, as_json=args.as_json)
        elif args.command in ("scopes", "list_scopes"):
            cmd_scopes(config, as_json=args.as_json, archived=args.archived)
        elif args.command == "import":
            cmd_import(config, args.path, use_llm_extract=args.llm_extract)
        elif args.command == "render":
            cmd_render(config, scope=args.scope, render_format=args.format)
        elif args.command in ("record", "record_decision"):
            # Exactly one axiom source: the positional or --axiom-file (the
            # quoting-safe twin of --rejected-file). Same JSON-aware dead-end
            # shape as the missing-rejected check below.
            if (args.axiom is None) == (args.axiom_file is None):
                msg = ("record requires exactly one axiom source: the positional "
                       "axiom OR --axiom-file ('-' = stdin), not both and not neither.")
                if args.as_json:
                    _emit_json({"error": msg, "code": "ambiguous_axiom_source"
                                if args.axiom is not None else "missing_axiom"})
                else:
                    print(msg, file=sys.stderr)
                sys.exit(2)
            axiom = _read_text_arg(args.axiom, args.axiom_file)
            if args.axiom_file is not None and axiom.endswith("\n"):
                axiom = axiom[:-1]  # strip the single trailing newline files/heredocs add
            rejected = _read_text_arg(args.rejected, args.rejected_file)
            if not (rejected and rejected.strip()):
                msg = ("record requires --rejected or --rejected-file "
                       "(the rejected alternatives are mandatory).")
                if args.as_json:
                    # No stderr walls under --json: the dead-end speaks a structured
                    # object on stdout, with the exit code preserved (2).
                    _emit_json({"error": msg, "code": "missing_rejected"})
                else:
                    print(msg, file=sys.stderr)
                sys.exit(2)
            context = _read_text_arg(args.context, args.context_file)
            cmd_record(
                config,
                axiom=axiom,
                rejected=rejected,
                scope=args.scope,
                mechanisms=args.mechanisms,
                context=context,
                supersedes=args.supersedes,
                corrects=args.corrects,
                amends=args.amends,
                narrows=args.narrows,
                depends_on=args.depends_on,
                resolves=args.resolves,
                contradicts=args.contradicts,
                derives_from=args.derives_from,
                cites=args.cites,
                slug=args.slug,
                acknowledge_neighbors=args.acknowledge_neighbors,
                as_json=args.as_json,
            )
        elif args.command == "serve":
            cmd_serve()
        elif args.command == "status":
            sys.exit(cmd_status(args.path or os.getcwd(), as_json=args.as_json))
        elif args.command == "agent-block":
            sys.exit(cmd_agent_block(args.path or os.getcwd(), check=args.check))
        elif args.command == "set-key":
            cmd_set_key(args.value, name=args.name, is_global=args.is_global)
        elif args.command == "cutover":
            sys.exit(cmd_cutover(config, allow_drops=args.allow_drops,
                                 assume_yes=args.yes, as_json=args.as_json))
        elif args.command == "rebuild":
            sys.exit(cmd_rebuild(config, allow_drops=args.allow_drops,
                                 assume_yes=args.yes, as_json=args.as_json))
        elif args.command == "check":
            sys.exit(cmd_check(config, staged=args.staged, scope=args.scope,
                               fresh=args.fresh, assume_yes=args.yes,
                               as_json=args.as_json))
    except MitosError as e:
        # KD1: `check` maps every pre-verb/boundary failure (bad -C, ConfigError, an
        # escaped store fault) to exit 2 — for CI, "could not run" is one routing
        # class with the verb's own exit-2 refusals; no other verb's contract moves.
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(2 if args.command == "check" else 1)
    except Exception as e:
        # The generic arm needs the same conditional: an unexpected crash under
        # `check` must not read as exit-1 "findings" to a CI consumer.
        print(f"Fatal Unexpected Error: {str(e)}", file=sys.stderr)
        sys.exit(2 if args.command == "check" else 1)
    finally:
        # Best-effort 'new version available' nudge, AFTER the command's own
        # output (the finally runs even on the sys.exit paths above). Skipped for
        # the long-running MCP server; fully fail-silent so it never disrupts work.
        if args.command != "serve":
            try:
                from mitos._update import update_notice
                _notice = update_notice(__version__)
                if _notice:
                    print(_notice, file=sys.stderr)
            except Exception:
                pass
            # Nudge CLI-only agents toward the MCP, but only on the decision-loop
            # verbs where it actually helps (and rate-limited inside _mcp_hint).
            if args.command in _DECISION_LOOP_COMMANDS:
                try:
                    _hint = _mcp_hint(config.workspace_dir)
                    if _hint:
                        print(_hint, file=sys.stderr)
                except Exception:
                    pass


if __name__ == "__main__":
    main()
