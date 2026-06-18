"""Sync pipeline for Mitos.

This module implements the core V3a and V3b sync loops, managing snapshotting,
concurrency file locks, LLM capture enrichment, user reviews, and content-aware
archive rotation.
"""

import os
import sys
import shutil
import re
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from filelock import FileLock, Timeout
from google import genai
from google.genai import types

from mitos.config import MitosConfig, hint_due
from mitos.errors import SynthesisError, ParseError, ValidationError, DatabaseError, EntryFailure
from mitos.models import get_model_id
from mitos.parser import ParsedEntry, parse_entry_stream
from mitos.store import GraphStore, CommitDelta
from mitos.identity import compute_node_id, embedding_text
from mitos.embeddings import GeminiEmbeddingProvider
from mitos.vector_store import QdrantVectorStore
from mitos.renderer import MitosRenderer, summarize_overflows

def run_sync_enrichment(
    client: genai.Client,
    entry: ParsedEntry,
    active_decisions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Calls Gemini to refine the decision, infer scopes, and suggest relationships."""
    active_summary = ""
    for d in active_decisions[:20]:  # Limit to top 20 active decisions for prompt budget
        active_summary += f"- slug: {d['slug']}\n  axiom: {d['core_axiom']}\n  scope: {','.join(d['scope'])}\n\n"

    prompt = f"""
You are the Mitos v0.1 capture enrichment agent. Your task is to refine and enrich a newly captured architectural decision.

Here are some currently ACTIVE decisions in the workspace:
{active_summary}

Here is the proposed decision entry:
Slug: {entry.slug}
Decided: {entry.axiom}
Rejected: {entry.rejected_paths}
Mechanisms: {','.join(entry.mechanisms)}
Scope: {','.join(entry.scope)}
Context: {entry.context}

Please enrich this entry. You must:
1. Verify and refine the `core_axiom` into a single, extremely precise, clear, and unambiguous sentence. If the raw axiom is already high quality, keep it verbatim or make minimal grammatical corrections.
2. Verify and refine the list of mechanism tags.
3. Suggest appropriate scope tags based on the content and existing active decisions.
4. Detect if this decision should supersede, amend, narrow, or depend on any of the active decisions listed above. If so, return their slugs in the suggested relationships.

Respond strictly in valid JSON format with the following keys:
- refined_core_axiom (string)
- refined_mechanisms (list of strings)
- refined_scope (list of strings)
- suggested_relationships (object with keys: supersedes, amends, narrows, depends_on, resolves)
"""
    model_id = get_model_id("FLASH_LITE")
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        return json.loads(response.text)
    except Exception as e:
        raise SynthesisError(f"LLM enrichment call failed: {str(e)}")


def run_ambient_capture(client: genai.Client, raw_text: str) -> str:
    """Uses FLASH to convert raw conversational text into a canonical Markdown entry."""
    prompt = f"""
You are the Mitos v0.1 capture scribe. Convert the following developer conversation or thought into a canonical Mitos Decision Entry.

Input text:
"{raw_text}"

Please generate a canonical Markdown entry. Use exactly this format (do not include markdown block quotes):

### [slug]

**Decided:** [Single-sentence axiom that is true going forward]
**Rejected:**
- [alternative] — [specific reason why it was rejected, be precise and adversarial]
**Mechanisms:** [comma-separated mechanisms, or none]
**Scope:** [comma-separated scope tags, or none]
**Context:** [brief background context explaining why this decision was made]

[DECISION_TRANSCRIPT]
User: {raw_text}
[/DECISION_TRANSCRIPT]

Make sure the slug is a clean, lowercase hyphenated string that matches the decision topic.
"""
    model_id = get_model_id("FLASH")
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2
            )
        )
        return response.text.strip()
    except Exception as e:
        raise SynthesisError(f"Ambient capture synthesis failed: {str(e)}")


# --- record_decision helpers (write-half of the MCP server) ---

# The exact buffer marker, byte-for-byte identical to cmd_capture (cli.py). The
# `—` is an em dash; do not retype it.
_ENTRIES_MARKER = "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->"

# A content line that looks like a Mitos field header (e.g. `**Decided:**`); the
# parser would treat it as a new field and corrupt the entry (parser.py:400-409).
_FIELD_LINE_RE = re.compile(r'^\s*\*\*[A-Za-z -]+:\*\*')

# A column-0 H2/H3 heading opens a NEW entry in the parser (parser.py:308). Note
# this is deliberately narrow: a single `#` (H1), `####`+ headings, and any
# *indented* heading are all SAFE and must NOT be rejected.
_SECTION_HEADER_RE = re.compile(r'^#{2,3}(?!#)')

# Inline markers the parser reacts to: section/transcript/buffer boundaries and
# the [NOTE:]/[PARKED:] scanners that siphon content into entry.notes (parser.py:419-426).
_STRUCTURAL_MARKERS = (
    "[DECISION_PARKED:",
    "[DECISION_TRANSCRIPT]",
    "[/DECISION_TRANSCRIPT]",
    "BEGIN ENTRIES",
    "[NOTE:",
    "[PARKED:",
)

# The exact agent-facing error messages (spec §5). Each says what went wrong AND
# how to recover, interpolating the offending value.
_ERROR_MESSAGES: Dict[str, str] = {
    "not_initialized": "No Mitos workspace found here. Run 'mitos init' before recording decisions.",
    "empty_axiom": "'axiom' is empty. Provide the decision as a single clear sentence that is true going forward.",
    "empty_slug": "'slug' is empty. Provide a short, explicit, hyphenated handle (e.g. 'sqlite-wal-mode').",
    "missing_rejected_paths": "'rejected_paths' is required: state the alternatives you considered and why you ruled them out — this is what stops you or another agent from re-proposing them later.",
    "parse_failed": "The decision could not be serialised into a valid entry — most likely a structural token in axiom/rejected_paths/context: a line beginning with '##' or '###' (indent it or use '#'/'####' instead), a line shaped like '**Something:**', or a '[DECISION_TRANSCRIPT]' / '[DECISION_PARKED:' / 'BEGIN ENTRIES' / '[NOTE:' / '[PARKED:' marker. Remove or rephrase that line and retry.",
    "slug_collision": "A different decision already uses the slug '{slug}'. Give this one a distinct 'slug'; and if it is meant to replace the existing decision, also set supersedes='{slug}' (the new decision must still have its own slug — two decisions cannot share one).",
    "supersedes_not_found": "supersedes='{supersedes}' does not match any existing decision. Look it up first with query_decisions to get the exact slug, or omit 'supersedes' if this is a brand-new decision.",
    "supersedes_ambiguous": "supersedes='{supersedes}' matches more than one decision. Use query_decisions to find the exact, full slug and pass that.",
    "corrects_not_found": "corrects='{corrects}' does not match any existing decision. Look it up first with query_decisions to get the exact slug, or omit 'corrects' if this is a brand-new decision.",
    "corrects_ambiguous": "corrects='{corrects}' matches more than one decision. Use query_decisions to find the exact, full slug and pass that.",
    "relation_target_not_found": "{relation}='{target}' does not match any existing decision. Look it up first with surface_decisions/query_decisions to get the exact slug, or omit '{relation}' if no such link applies.",
    "relation_target_ambiguous": "{relation}='{target}' matches more than one decision. Use query_decisions to find the exact, full slug and pass that.",
    "commit_failed": "The decision validated but the commit failed and nothing was written: {reason}. Retry; if it persists, the workspace store may be locked or corrupt.",
}

# The user-facing typed relations beyond `supersedes` (which is special: it changes
# computed state and has its own error codes). Each maps an agent-facing kwarg name to
# its canonical decisions.md field label. The parser and the store's commit path
# already understand all of these (format-spec.md §"Relationship Fields"); the agentic
# write path just had to serialize + validate them the way it already does supersedes.
_EXTRA_RELATIONS = (
    ("amends", "Amends"),
    ("narrows", "Narrows"),
    ("depends_on", "Depends-On"),
    ("resolves", "Resolves"),
    ("contradicts", "Contradicts"),
    ("derives_from", "Derives-From"),
    ("cites", "Cites"),
)


def _record_error(code: str, **fields: Any) -> Dict[str, str]:
    """Builds a structured {error, code} dict using the canonical message for ``code``."""
    return {"error": _ERROR_MESSAGES[code].format(**fields), "code": code}


def _embedding_input_text(
    kind: str,
    axiom: Optional[str] = None,
    topic: Optional[str] = None,
    questions_raised: Optional[List[str]] = None,
) -> str:
    """Derives the V1a embedding-input string for an entry or node (C2/M8 single source).

    Routes through :func:`identity.embedding_text` so the record-time and drain-time
    embedding text are byte-identical — a node embedded inline at record time and the
    same node re-derived from the Outbox at drain time yield the same vector (the
    ``embedding_text`` column is gone in V1a, so drain re-derives from the immutable
    core, M8). Bridges the one reader-key gap: a decision's axiom is exposed as
    ``axiom`` on a :class:`ParsedEntry` but ``core_axiom`` on a store node dict —
    callers pass whichever they hold under ``axiom``.

    Args:
        kind: ``"decision"`` or ``"open_question"``.
        axiom: The decision axiom (for a decision entry/node).
        topic: The open_question topic.
        questions_raised: The open_question's questions, in authored order.

    Returns:
        The embedding-input string (normalized, M8-consistent with the hashed core).
    """
    return embedding_text({
        "kind": kind,
        "axiom": axiom,
        "topic": topic,
        "questions_raised": questions_raised or [],
    })


def _contains_structural_token(text: str) -> bool:
    """Returns True if any line of ``text`` would corrupt the flat-file parser.

    Matches the parser's own raw-line semantics (no leading-whitespace strip for
    headers, parser.py:308): only column-0 ``##``/``###`` headings, field-shaped
    lines, and the inline markers trigger.
    """
    for line in text.split("\n"):
        if _SECTION_HEADER_RE.match(line):
            return True
        if _FIELD_LINE_RE.match(line):
            return True
        if any(marker in line for marker in _STRUCTURAL_MARKERS):
            return True
    return False


_SLUG_MAX_LEN = 64
_SLUG_MIN_LEN = 32  # don't trim a word boundary back past here — hard-cap instead

# A new decision at/above this document-document similarity to an existing one the
# author did NOT reference is paused for review (AX P4): the neighbour was invisible
# until the post-commit `related` echo, one step too late to point an
# amends/supersedes/contradicts at it (you can't relink after commit — re-record is a
# no-op). Same score scale as the `related` echo. Tune here.
_NEIGHBOR_REVIEW_THRESHOLD = 0.85

# Cheap polarity cues for the `possible_tension` hint — a high-similarity pair where
# one axiom negates and the other doesn't ("never a per-persona field" vs "is a
# per-persona field") is a likely contradiction, not a duplicate. Heuristic only: it
# flags a neighbour for the author's eye, never changes the pause decision.
_NEGATION_CUES = (" not ", " never ", " no ", "n't", " without ", " cannot ",
                  " neither ", " nor ", " none ", " non-")


def _has_negation(text: str) -> bool:
    """Reports whether ``text`` carries a surface negation cue (whitespace-padded scan)."""
    padded = f" {text.lower()} "
    return any(cue in padded for cue in _NEGATION_CUES)


def _polarity_mismatch(a: str, b: str) -> bool:
    """True when exactly one of two axioms negates — a possible-tension signal."""
    return _has_negation(a) != _has_negation(b)


def _slugify(text: str) -> str:
    """Derives a deterministic, lowercase-hyphenated slug from free text.

    Determinism keeps the human-readable handle stable: the slug is NOT part of the
    node id (V1a identity is the slug-free canonical core — ``compute_node_id``), but
    a stable auto-derived slug means the same decision presents the same handle and
    the casefold slug-collision check (V1-D4) behaves predictably.

    When the slug exceeds the length cap it is trimmed back to the last word
    boundary (hyphen) rather than sliced mid-word, so the handle an agent carries
    into ``supersedes``/relations stays readable (``…brazilian-portuguese``, not
    ``…brazilian-portug``). Still a pure function of the text, so determinism holds.
    """
    if not text:
        return ""
    s = re.sub(r'[^a-z0-9]+', '-', text.lower())
    s = re.sub(r'-+', '-', s).strip('-')
    if len(s) > _SLUG_MAX_LEN:
        cut = s[:_SLUG_MAX_LEN]
        boundary = cut.rfind('-')
        # Trim to the last whole word, unless that would gut the slug (one very
        # long leading token) — then fall back to the hard cap.
        if boundary >= _SLUG_MIN_LEN:
            cut = cut[:boundary]
        s = cut.rstrip('-')
    return s


def _slug_is_truncated(text: str) -> bool:
    """Reports whether deriving a slug from ``text`` would hit the length cap.

    A truncated auto-slug is still valid but makes a lossy handle to carry into
    ``supersedes``/relations — so the write path nudges for an explicit ``slug=``.
    """
    s = re.sub(r'[^a-z0-9]+', '-', text.lower())
    s = re.sub(r'-+', '-', s).strip('-')
    return len(s) > _SLUG_MAX_LEN


class MitosSyncManager:
    """Manages the full parse-enrich-commit sync flow and side effects."""

    def __init__(self, config: MitosConfig) -> None:
        self.config = config
        self.lock_path = self.config.decisions_file + ".lock"
        self.lock = FileLock(self.lock_path, timeout=60)
        self.store = GraphStore(self.config.db_path)
        
        # Lazy initialize vector / embedding dependencies as best-effort (C2/P14)
        self.embed_provider: Optional[GeminiEmbeddingProvider] = None
        self.vector_store: Optional[QdrantVectorStore] = None
        
        try:
            cache_path = os.path.join(self.config.mitos_dir, "embedding_cache.sqlite")
            self.embed_provider = GeminiEmbeddingProvider(cache_path)
            self.vector_store = QdrantVectorStore(
                self.config.qdrant_url,
                self.config.qdrant_collection
            )
        except Exception as e:
            # Let operations continue in degraded graph-only mode per S1/F2
            pass

    def auto_heal_decisions_file(self) -> None:
        """Auto-restores the decisions.md header and sample format block if modified or missing."""
        filepath = self.config.decisions_file
        if not os.path.exists(filepath):
            return

        # Load canonical format spec from package single source of truth
        from mitos.cli import load_format_spec
        try:
            format_spec_content = load_format_spec()
        except Exception:
            return

        # Extract sample block
        import re
        match = re.search(r'## 3\.\s+Sample Entry.*?\n```markdown\n(.*?)\n```', format_spec_content, re.DOTALL | re.IGNORECASE)
        sample_block = match.group(1).strip() if match else ""
        if not sample_block:
            return

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        canonical_header = (
            "# Decisions for Mitos\n\n"
            "<!-- This file is managed by mitos. LLM integration: see .mitos/skill.md once V5 ships. -->\n"
            "<!-- DO NOT MODIFY ABOVE THIS LINE -->\n\n"
            "## SAMPLE FORMAT — auto-restored by mitos sync, do not modify or delete\n\n"
            f"{sample_block}\n\n"
        )

        marker = "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->"
        if marker in content:
            parts = content.split(marker, 1)
            entries_content = parts[1]
            current_header = parts[0]
            if current_header.strip() != canonical_header.strip():
                new_content = canonical_header + marker + entries_content
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print("Auto-restored decisions.md sample format header block ✓")
        else:
            if "## SAMPLE FORMAT" not in content:
                new_content = canonical_header + marker + "\n\n" + content
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print("Auto-restored missing sample format header and BEGIN ENTRIES marker ✓")

    def perform_sync(self, auto_accept: bool = False, verbose: bool = False) -> None:
        """Executes the complete transactional sync flow."""
        snapshot_path = os.path.join(self.config.mitos_dir, "sync_snapshot.md")
        try:
            self._perform_sync_internal(snapshot_path, auto_accept, verbose)
        finally:
            if os.path.exists(snapshot_path):
                try:
                    os.remove(snapshot_path)
                except Exception:
                    pass

    def _perform_sync_internal(self, snapshot_path: str, auto_accept: bool = False, verbose: bool = False) -> None:
        """Executes the internal transactional sync flow."""
        # 1. Snapshot-at-sync-start under brief file lock
        try:
            with self.lock:
                if not os.path.exists(self.config.decisions_file):
                    print("No decisions.md file found. Run 'mitos init' first.")
                    return
                # Auto-heal the decisions file header/sample block under lock
                self.auto_heal_decisions_file()
                shutil.copy(self.config.decisions_file, snapshot_path)
        except Timeout:
            print("Another Mitos process holds the lock; check for stuck 'mitos sync'.")
            return

        # 2. Parse from the snapshot. Per-entry isolation (§7.2-A degradation
        #    contract): a malformed entry is reported with its line range and
        #    skipped, so the remaining well-formed entries still sync. The skipped
        #    entries are never committed, so content-aware rotation leaves them in
        #    decisions.md for the user to fix and re-sync.
        with open(snapshot_path, "r", encoding="utf-8") as f:
            snapshot_text = f.read()
        # decisions.md is a decisions-only stream in V1a (OQ authoring lives in a
        # separate questions.md — 1c). Collector mode isolates a malformed entry
        # (reported + skipped) so the rest still sync (§5.2.2 per-entry isolation).
        parse_failures: List[EntryFailure] = []
        entries = parse_entry_stream(snapshot_text, "decision", failures=parse_failures)

        for fail in parse_failures:
            msgs = "; ".join(item.message for item in fail.items) or "malformed entry"
            print(
                f"[Parse error] {msgs} (lines {fail.line_start}-{fail.line_end}). "
                f"Entry skipped — fix it and re-run sync."
            )

        if not entries:
            if parse_failures:
                print("No parseable entries to commit. Fix the reported entries above and re-run sync.")
            else:
                print("Zero pending entries found in decisions.md write-buffer.")
            return

        # Stale-entry detection (>14 days unprocessed)
        for entry in entries:
            if entry.date:
                try:
                    entry_dt = datetime.strptime(entry.date, "%Y-%m-%d")
                    diff = datetime.now() - entry_dt
                    if diff.days > 14:
                        print(f"[Warning] Entry '{entry.slug}' was drafted on {entry.date} (>14 days ago) and remains unsynced.")
                except Exception:
                    pass

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("GEMINI_API_KEY environment variable is not set. Sync requires API keys.")
            return
            
        genai_client = genai.Client(api_key=api_key)
        renderer = MitosRenderer(self.config.workspace_dir)

        synced_blocks: List[Tuple[ParsedEntry, str]] = []

        # 3. Process each parsed entry
        for entry in entries:
            # Read exact raw text block of this entry from snapshot for content-aware rotation
            with open(snapshot_path, "r", encoding="utf-8") as f:
                snap_lines = f.readlines()
            entry_raw_text = "".join(snap_lines[entry.line_start - 1 : entry.line_end])

            # Check if this node is already in the database (slug-free V1a id — V1-D2).
            node_id = compute_node_id(
                kind=entry.kind,
                axiom=entry.axiom,
                mechanism_refs=entry.mechanisms,
                topic=entry.topic,
                questions_raised=entry.questions_raised,
            )

            existing = self.store.get_node(node_id)
            if existing:
                # Idempotency short-circuit (S5)
                continue

            # Slug collision check
            collision = self.store.get_node_by_slug(entry.slug)
            edge_relationship: Optional[str] = None
            
            if collision:
                print(f"\n[Collision] Slug '{entry.slug}' already exists in graph.")
                print(f"  Existing Axiom: {collision.get('core_axiom')}")
                print(f"  New Axiom:      {entry.axiom}")
                
                if auto_accept:
                    # Default to correction in auto-mode
                    edge_relationship = "corrects"
                else:
                    while True:
                        choice = input("Is this a [c]orrection, [s]upersession, or [a]bort? ").strip().lower()
                        if choice == 'c':
                            edge_relationship = "corrects"
                            break
                        elif choice == 's':
                            edge_relationship = "supersedes"
                            break
                        elif choice == 'a':
                            print("Sync aborted by user.")
                            return
                        else:
                            print("Invalid choice.")

            if entry.kind == "decision":
                # LLM capture enrichment (FLASH_LITE)
                active_decs = self.store.get_active_decisions()
                
                try:
                    enrichment = run_sync_enrichment(genai_client, entry, active_decs)
                except Exception as e:
                    # Degradation F1: Pause and let user decide
                    print(f"\n[Error] LLM enrichment failed for '{entry.slug}': {str(e)}")
                    choice = input("Would you like to [r]etry, [s]kip this entry, or [a]bort sync? ").strip().lower()
                    if choice == 'r':
                        # Retry once
                        enrichment = run_sync_enrichment(genai_client, entry, active_decs)
                    elif choice == 's':
                        continue
                    else:
                        print("Sync aborted.")
                        return

                # Apply refinements
                refined_axiom = enrichment.get("refined_core_axiom", entry.axiom)
                refined_mechs = enrichment.get("refined_mechanisms", entry.mechanisms)
                refined_scopes = enrichment.get("refined_scope", entry.scope)
                sugg_rels = enrichment.get("suggested_relationships", {})

                print(f"\nProposed Capture: {entry.slug}")
                print(f"  Core Axiom:  {refined_axiom}")
                print(f"  Rejected:    {entry.rejected_paths}")
                print(f"  Mechanisms:  {', '.join(refined_mechs)}")
                print(f"  Scope:       {', '.join(refined_scopes)}")
                
                for rel_type, rel_slug in sugg_rels.items():
                    if rel_slug:
                        print(f"  Suggested Relationship: {rel_type} -> {rel_slug}")

                if not auto_accept:
                    u_choice = input("Accept this decision? [a]ccept / [e]dit / [s]kip / [q]uit: ").strip().lower()
                    if u_choice == 's':
                        continue
                    elif u_choice == 'q':
                        print("Sync paused by user.")
                        break
                    elif u_choice == 'e':
                        # Simple inline editing loop
                        refined_axiom = input(f"Enter core axiom [{refined_axiom}]: ").strip() or refined_axiom

                # Commit refined values back onto the entry (commit reads .axiom — V1a).
                entry.axiom = refined_axiom
                entry.mechanisms = refined_mechs
                entry.scope = refined_scopes
                
                # Apply slug-collision override if present
                if edge_relationship == "corrects":
                    entry.corrects = entry.slug
                    entry.supersedes = None
                elif edge_relationship == "supersedes":
                    entry.supersedes = entry.slug
                    entry.corrects = None
                else:
                    def norm_rel(v: Any) -> Optional[str]:
                        if not v:
                            return None
                        if isinstance(v, list):
                            return str(v[0]).strip() if v else None
                        return str(v).strip()

                    entry.supersedes = entry.supersedes or norm_rel(sugg_rels.get("supersedes"))
                    entry.amends = entry.amends or norm_rel(sugg_rels.get("amends"))
                    entry.narrows = entry.narrows or norm_rel(sugg_rels.get("narrows"))
                    entry.depends_on = entry.depends_on or norm_rel(sugg_rels.get("depends_on"))
                    entry.resolves = entry.resolves or norm_rel(sugg_rels.get("resolves"))

            else:
                # Open Question Sync
                print(f"\nProposed Open Question: {entry.slug}")
                print(f"  Questions: {', '.join(entry.questions_raised)}")
                if not auto_accept:
                    u_choice = input("Accept this open question? [a]ccept / [s]kip / [q]uit: ").strip().lower()
                    if u_choice == 's':
                        continue
                    elif u_choice == 'q':
                        print("Sync paused by user.")
                        break

            # Populate OD3 confirmation metadata
            entry.confirmed_by = get_model_id("FLASH_LITE") if entry.kind == "decision" else "user"
            entry.confirmed_at = datetime.now().isoformat()

            # Commit to graph database atomically per entry (C1 atomicity)
            delta = self.store.commit_parsed_entry(entry)
            print(f"Committed node: {entry.slug} ✓")

            # best-effort embedding upsert (C2)
            self._best_effort_embed(delta, entry)

            # Record successfully committed block for rotation
            synced_blocks.append((entry, entry_raw_text))

        # 4. Content-aware archive rotation under brief lock (V3b)
        if synced_blocks:
            if len(synced_blocks) >= self.config.pending_threshold and not auto_accept:
                print(f"\n[Lifecycle] Sync volume threshold reached ({len(synced_blocks)} entries pending rotation).")
                choice = input("Would you like to rotate the write-buffer to quarterly archive now? [y/n]: ").strip().lower()
                if choice != 'y':
                    synced_blocks.clear()
                    print("Archive rotation deferred. Entries remain in write-buffer.")

        if synced_blocks:
            try:
                with self.lock:
                    with open(self.config.decisions_file, "r", encoding="utf-8") as f:
                        live_content = f.read()

                    rotated_text = ""
                    for entry, raw_block in synced_blocks:
                        # Match by content block exactly and remove/modify in live file
                        if raw_block in live_content:
                            if self.config.rotation_mode == "mark":
                                # Mark mode: wrap the raw block in an HTML comment so it's ignored but preserved
                                commented_block = f"<!-- ROTATED START\n{raw_block}\nROTATED END -->"
                                live_content = live_content.replace(raw_block, commented_block)
                            else:
                                # Archive/Prune mode: remove from live buffer
                                live_content = live_content.replace(raw_block, "")
                            rotated_text += raw_block + "\n"

                    # Write back live buffer (non-destructive)
                    with open(self.config.decisions_file, "w", encoding="utf-8") as f:
                        f.write(live_content)

                    # Only write to archive directory if in archive mode!
                    if self.config.rotation_mode == "archive":
                        quarter_file = f"{datetime.now().year}-Q{(datetime.now().month-1)//3 + 1}.md"
                        os.makedirs(self.config.archive_dir, exist_ok=True)
                        archive_path = os.path.join(self.config.archive_dir, quarter_file)
                        
                        with open(archive_path, "a", encoding="utf-8") as f:
                            f.write(rotated_text)
                        print(f"Rotated {len(synced_blocks)} entries to {archive_path} ✓")
                    elif self.config.rotation_mode == "prune":
                        print(f"Pruned {len(synced_blocks)} entries from buffer (rotation_mode=prune) ✓")
                    elif self.config.rotation_mode == "mark":
                        print(f"Marked {len(synced_blocks)} entries as rotated in buffer (rotation_mode=mark) ✓")
            except Exception as e:
                print(f"[Warning] Archive rotation failed: {str(e)}")

        # 5. Trigger renderer to statelessly regenerate files (C3)
        try:
            renderer.render_all(self.store)
            print("Regenerated live_axioms.md ✓")
        except Exception as e:
            # Degradation F4b: render failure doesn't affect graph commits
            print(f"[Warning] Failed to render active axioms: {str(e)}")

        # Temporary snapshot cleanup is handled by the perform_sync finally block

        # 6. Best-effort outbox queue drain attempt (C2)
        try:
            self.drain_pending_embeddings()
        except Exception as e:
            print(f"[Warning] Outbox queue drain failed: {str(e)}")

        # 7. Surplus hit/miss stats observability (4.D)
        if verbose and self.embed_provider:
            hits, misses, rate = self.embed_provider.get_stats()
            print(f"\n[Observability] Cache Stats: Hits: {hits}, Misses: {misses}, Hit Rate: {rate*100:.1f}%")

    def _best_effort_embed(self, delta: CommitDelta, entry: ParsedEntry) -> Optional[List[float]]:
        """Best-effort async embedding upsert pipeline (C2).

        The committing node is already enqueued on the ``pending_embeddings`` Outbox
        by ``commit_parsed_entry`` (``_enqueue_outbox``, 5c), so this is the inline
        fast path: with the provider up we embed + upsert immediately and DROP the
        now-redundant Outbox row (the node is indexed); with it down or the call
        failing we leave the row for the next ``sync`` drain — never a second enqueue
        (the commit already wrote one; the prototype's deferred ``add_pending_embedding``
        is retired here, 8a). Returns the document vector it computed and upserted (so
        a caller can reuse it for a neighbour query), or None if embedding was
        deferred/failed.
        """
        embed_text = _embedding_input_text(
            kind=entry.kind, axiom=entry.axiom,
            topic=entry.topic, questions_raised=entry.questions_raised,
        )

        if not self.embed_provider or not self.vector_store:
            # Already enqueued by the commit; just note the deferral. stderr — this
            # path is shared with the MCP write tool, whose stdout is the JSON-RPC
            # channel (a stray stdout line there corrupts the protocol).
            print(f"[Warning] Embedding upsert deferred for '{entry.slug}': Embedding provider down.",
                  file=sys.stderr)
            return None

        # Prepare payload
        payload = {
            "slug": entry.slug,
            "scope": entry.scope,
            "state": "active",
            "kind": entry.kind,
            "embedding_text": embed_text
        }

        try:
            # Check embedding provider and generate vector
            vector = self.embed_provider.get_embedding(payload["embedding_text"], is_query=False)
            self.vector_store.upsert(delta.node_id, vector, payload)
            # Indexed now — drop the Outbox row the commit enqueued.
            try:
                self.store.remove_pending_embedding(delta.node_id)
            except Exception as dbe:
                print(f"[Warning] Failed to clear outbox row: {str(dbe)}", file=sys.stderr)
            return vector
        except Exception as e:
            # The commit already enqueued this node (C2); leave the row for the next
            # drain. stderr — shared with the MCP write tool's JSON-RPC stdout channel.
            print(f"[Warning] Embedding upsert deferred for '{entry.slug}': {str(e)}", file=sys.stderr)
            return None

    def drain_pending_embeddings(self) -> None:
        """Drains the pending embeddings outbox queue (C2).

        Claims a batch of pending embeddings atomically to prevent concurrent
        drainers from double-processing rows, processes them, and removes resolved entries.
        """
        if not self.embed_provider or not self.vector_store:
            print("Cannot drain outbox: Embedding provider or vector store down.")
            return

        import uuid
        drainer_id = f"drainer-{uuid.uuid4()}"

        try:
            # Claim up to 10 pending embeddings atomically
            pending = self.store.claim_pending_embeddings(drainer_id, limit=10)
        except Exception as e:
            print(f"[Warning] Failed to claim outbox queue: {str(e)}")
            return

        if not pending:
            return

        print(f"Draining pending embeddings queue ({len(pending)} items) ...")
        
        try:
            for item in pending:
                node_id = item["node_id"]

                # Fetch node details from graph for Qdrant payload
                node = self.store.get_node(node_id)
                if not node:
                    # Node has been deleted from graph; remove from queue
                    try:
                        self.store.remove_pending_embedding(node_id)
                    except Exception:
                        pass
                    continue

                # Re-derive the embedding text from the node's immutable core (the
                # Outbox row no longer carries it — C2/M8); byte-identical to what the
                # inline record-time embed used for the same node.
                embed_text = _embedding_input_text(
                    kind=node["kind"], axiom=node.get("core_axiom"),
                    topic=node.get("topic"), questions_raised=node.get("questions_raised"),
                )

                payload = {
                    "slug": node["slug"],
                    "scope": node["scope"],
                    "state": "active",
                    "kind": node["kind"],
                    "embedding_text": embed_text
                }

                try:
                    # 1. Fetch embedding vector
                    vector = self.embed_provider.get_embedding(embed_text, is_query=False)
                    # 2. Upsert to Qdrant
                    self.vector_store.upsert(node_id, vector, payload)
                    # 3. Clean up queue row on success
                    self.store.remove_pending_embedding(node_id)
                    print(f"Successfully drained embedding for '{node['slug']}' ✓")
                except Exception as e:
                    # Increment retry count on failure (which also releases this row)
                    try:
                        self.store.increment_pending_attempts(node_id)
                    except Exception:
                        pass
                    print(f"[Warning] Failed to drain embedding for '{node['slug']}': {str(e)}")
        finally:
            # Clean up: release any remaining locks held by this specific drainer
            try:
                self.store.release_pending_embeddings(drainer_id)
            except Exception:
                pass

    # --- record_decision: the write half of the MCP server (Fork A) ---

    def _exact_slug_node(self, slug: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Resolves a slug to an EXACT-match node.

        ``resolve_slug`` is now single-tier casefold-exact (no fuzzy prefix tier), so
        every id it returns already shares the casefolded slug; the per-node re-filter
        below is a defensive guard on that contract (and folds with ``str.casefold()``,
        never ``str.lower()`` — the two diverge on ``ß``/Greek, MI-9).

        Returns:
            A (node_id, node) tuple for the exact-slug node, or (None, None).
        """
        for node_id in self.store.resolve_slug(slug):
            node = self.store.get_node(node_id)
            if node and node.get("slug", "").casefold() == slug.casefold():
                return node_id, node
        return None, None

    def _validate_relation_target(self, relation: str, target: str) -> Optional[Dict[str, str]]:
        """Validates a typed relation's target is a unique, EXACT-match decision.

        Mirrors the supersedes check (``resolve_slug`` is casefold-exact; the re-filter
        defends its contract), keeping every recorded edge pointed at a real, unambiguous
        node. Runs in Phase A — a failure returns a structured error and writes nothing.

        Args:
            relation: The relation kwarg name (for the error message), e.g. "amends".
            target: The slug the agent passed as that relation's target.

        Returns:
            None if valid, else a structured ``{error, code}`` dict.
        """
        ids = self.store.resolve_slug(target)
        if not ids:
            return _record_error("relation_target_not_found", relation=relation, target=target)
        if len(ids) > 1:
            return _record_error("relation_target_ambiguous", relation=relation, target=target)
        node = self.store.get_node(ids[0])
        if not node or node.get("slug", "").casefold() != target.casefold():
            return _record_error("relation_target_not_found", relation=relation, target=target)
        return None

    def _adjacent_decisions(self, vector: Optional[List[float]], exclude_slug: str,
                            limit: int = 3) -> List[Dict[str, Any]]:
        """Best-effort: the nearest OTHER live decisions to a just-recorded one.

        A write-time guardrail — surfaces semantic neighbours so an agent notices an
        adjacent or contradictory prior decision instead of silently accumulating
        tension in the graph. Needs embeddings (it is semantic), so it is empty when
        offline. A pure read that runs AFTER the commit and is fully fail-silent: it
        never touches the buffer-first + rollback write contract.

        Args:
            vector: The just-recorded decision's document embedding (reused), or None.
            exclude_slug: The new decision's own slug, filtered out of the neighbours.
            limit: Maximum neighbours to return.

        Returns:
            Up to ``limit`` dicts ``{slug, axiom, score}`` for live neighbours, most
            similar first; empty if offline or nothing comparable exists.
        """
        if vector is None or not self.vector_store:
            return []
        out: List[Dict[str, Any]] = []
        try:
            for m in self.vector_store.query(vector, limit=limit + 3):
                slug = m.get("slug")
                if not slug or slug == exclude_slug:
                    continue
                node = self.store.get_node_by_slug(slug)
                if not node or self.store.get_node_state(node["id"]) not in ("active", "drifted"):
                    continue
                out.append({"slug": slug, "axiom": node["core_axiom"], "score": m.get("score")})
                if len(out) >= limit:
                    break
        except Exception:
            return []
        return out

    def _review_neighbors(self, entry: ParsedEntry,
                          declared_targets: set) -> List[Dict[str, Any]]:
        """Pre-commit: existing live decisions too similar to ``entry`` to ignore (P4).

        Embeds the about-to-be-recorded axiom (same document vector and score scale as
        the post-commit ``related`` echo), finds its nearest live neighbours, and keeps
        those at/above ``_NEIGHBOR_REVIEW_THRESHOLD`` that the author did NOT already
        reference via a relation. Surfacing these BEFORE the write is the whole point:
        after the commit the author can no longer point an amends/supersedes at them (a
        re-record is a no-op). Fully fail-silent and offline-safe — no embeddings/vector
        store, or any error, means an empty list (never block a write we can't check).

        Args:
            entry: The parsed entry about to be committed.
            declared_targets: Casefolded slugs the entry already links to (its declared
                supersedes/amends/… targets), excluded so a linked neighbour is not re-flagged.

        Returns:
            A list of ``{slug, axiom, score, possible_tension}`` for unreferenced
            high-similarity neighbours, most similar first; empty when there is nothing
            to flag or the check could not run.
        """
        if not self.embed_provider or not self.vector_store:
            return []
        try:
            text = _embedding_input_text(
                kind=entry.kind, axiom=entry.axiom,
                topic=entry.topic, questions_raised=entry.questions_raised,
            )
            vector = self.embed_provider.get_embedding(text, is_query=False)
        except Exception:
            return []
        flagged: List[Dict[str, Any]] = []
        for n in self._adjacent_decisions(vector, exclude_slug=entry.slug, limit=5):
            score = n.get("score")
            if score is None or score < _NEIGHBOR_REVIEW_THRESHOLD:
                continue
            if n["slug"].casefold() in declared_targets:
                continue
            flagged.append({
                "slug": n["slug"],
                "axiom": n["axiom"],
                "score": score,
                "possible_tension": _polarity_mismatch(entry.axiom, n["axiom"]),
            })
        return flagged

    def _node_state(self, node_id: str) -> str:
        """Returns the computed state of a node ('active'/'superseded'/'corrected'/'drifted')."""
        return self.store.get_node_state(node_id)

    def _embedding_status(self, node_id: str) -> str:
        """Reports whether a node's embedding is queued in the outbox ('pending') or done."""
        try:
            for row in self.store.get_pending_embeddings():
                if row.get("node_id") == node_id:
                    return "pending"
        except Exception:
            pass
        return "indexed"

    def record_decision_entry(
        self,
        axiom: str,
        rejected_paths: str,
        scope: List[str],
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
        slug: Optional[str] = None,
        actor: str = "agent",
        acknowledge_neighbors: bool = False,
    ) -> Dict[str, Any]:
        """Records a single decision into the buffer and graph, non-interactively.

        The agentic write half of Mitos: persists one deliberate, pre-structured
        decision (and the alternatives it rejected) the moment an agent makes it,
        without LLM enrichment and without calling ``perform_sync``. Composes the
        existing primitives only — ``commit_parsed_entry`` for the graph and
        ``_best_effort_embed`` for the vector (preserves M7).

        Validation runs entirely in memory FIRST (Phase A); the buffer append and
        the graph commit happen together, last, under a single lock, with the
        buffer rolled back if the commit fails (Phase B). The contract: on any
        error code, ``decisions.md`` is byte-for-byte unchanged and nothing is
        committed.

        Args:
            axiom: The decision as a single clear sentence true going forward (M1).
            rejected_paths: The alternatives considered and rejected, and why (M5, required).
            scope: Area tags (may be empty).
            mechanisms: Concrete technologies/entities involved (M6).
            context: Optional background on why this was decided.
            supersedes: Optional exact slug of a prior decision this one replaces.
            corrects: Optional exact slug of a prior decision this one corrects (a
                kill-edge twin of supersedes — the target leaves the active view).
            amends: Optional exact slug of a decision this one amends.
            narrows: Optional exact slug of a decision this one narrows.
            depends_on: Optional exact slug of a decision this one depends on.
            resolves: Optional exact slug of an open question/decision this resolves.
            contradicts: Optional exact slug of a decision this one contradicts.
            derives_from: Optional exact slug of a decision this one derives from.
            cites: Optional exact slug of a decision this one cites.
            slug: Optional explicit slug; derived deterministically from axiom if None.
            actor: Provenance, stored in ``confirmed_by``.
            acknowledge_neighbors: Skip the pre-commit near-duplicate review and record
                even when a highly-similar unreferenced decision exists (P4). Pass True
                to commit a genuinely independent decision past the pause.

        Returns:
            A success dict ``{slug, id, state, embedding, status}`` (status
            "created"|"exists"), plus an optional ``related`` list of the nearest
            existing live decisions (a write-time adjacency hint on the "created"
            path); OR, when a highly-similar unreferenced decision exists and
            ``acknowledge_neighbors`` is False, a ``{status: "needs_review", code:
            "similar_decision_exists", neighbors, message}`` pause that wrote NOTHING;
            OR a structured ``{error, code}`` failure (see spec §5).
        """
        # === Phase A — validate everything in memory (no writes) ===

        # 1. Preconditions. os.path.exists, NOT manager construction (which creates the db).
        if not os.path.exists(self.config.decisions_file):
            return _record_error("not_initialized")

        # 2. Normalise CRLF, then validate. A stray \r perturbs the field regex and
        #    the canonical-core hash (same decision would hash differently across
        #    environments — compute_node_id normalizes, but normalize at the boundary).
        axiom = (axiom or "").replace("\r\n", "\n").replace("\r", "\n")
        rejected_paths = (rejected_paths or "").replace("\r\n", "\n").replace("\r", "\n")
        if context is not None:
            context = context.replace("\r\n", "\n").replace("\r", "\n")

        if not axiom.strip():
            return _record_error("empty_axiom")
        if not rejected_paths.strip():
            return _record_error("missing_rejected_paths")

        # 3. Reject (do NOT sanitise) content fields carrying structural tokens.
        #    Check the NON-stripped values: a leading-whitespace `  ## heading` is
        #    safe (the parser only splits on column-0 `##`), and stripping it first
        #    would falsely promote it to column 0. Storage stripping happens after.
        for field_text in (axiom, rejected_paths, context):
            if field_text and _contains_structural_token(field_text):
                return _record_error("parse_failed")

        axiom = axiom.strip()
        rejected_paths = rejected_paths.strip()
        context = context.strip() if context and context.strip() else None
        mechanisms = [m.strip() for m in mechanisms if m and m.strip()] if mechanisms else []
        scope = [s.strip() for s in scope if s and s.strip()] if scope else []
        if supersedes is not None:
            supersedes = supersedes.strip() or None
        if corrects is not None:
            corrects = corrects.strip() or None

        # Normalise the other typed relations into a stable-ordered map (supersedes and
        # corrects are handled separately — both are kill-edges that change computed
        # state and carry bespoke error codes).
        _provided = {
            "amends": amends, "narrows": narrows, "depends_on": depends_on,
            "resolves": resolves, "contradicts": contradicts,
            "derives_from": derives_from, "cites": cites,
        }
        extra_relations: Dict[str, str] = {}
        for _name, _label in _EXTRA_RELATIONS:
            _val = _provided.get(_name)
            if _val and _val.strip():
                extra_relations[_name] = _val.strip()

        # 4. Deterministic slug.
        slug = _slugify(slug)
        if not slug:
            return _record_error("empty_slug")

        # 5. Serialise to the canonical format (in memory only).
        lines = [f"### {slug}", "", f"**Decided:** {axiom}", f"**Rejected:** {rejected_paths}"]
        if mechanisms:
            lines.append(f"**Mechanisms:** {', '.join(mechanisms)}")
        if scope:
            lines.append(f"**Scope:** {', '.join(scope)}")
        if context:
            lines.append(f"**Context:** {context}")
        if supersedes:
            lines.append(f"**Supersedes:** {supersedes}")
        if corrects:
            lines.append(f"**Corrects:** {corrects}")
        for _name, _label in _EXTRA_RELATIONS:
            if _name in extra_relations:
                lines.append(f"**{_label}:** {extra_relations[_name]}")
        entry_text = "\n".join(lines) + "\n"

        # 6. Parse our entry back through the V1a tokenizer (sets .axiom/.topic +
        #    the relationship attrs), then run the graph-level checks as a read-only
        #    fast-fail. STRICT mode (no collector): a malformed self-serialized entry
        #    raises ParseError, which we map to the structured parse_failed code (G2).
        try:
            parsed = parse_entry_stream(entry_text, "decision")
        except ParseError:
            return _record_error("parse_failed")
        if len(parsed) != 1:
            return _record_error("parse_failed")
        entry = parsed[0]

        # Pre-validate supersedes with an EXACT match.
        if supersedes:
            ids = self.store.resolve_slug(supersedes)
            if not ids:
                return _record_error("supersedes_not_found", supersedes=supersedes)
            if len(ids) > 1:
                return _record_error("supersedes_ambiguous", supersedes=supersedes)
            target = self.store.get_node(ids[0])
            if not target or target.get("slug", "").casefold() != supersedes.casefold():
                return _record_error("supersedes_not_found", supersedes=supersedes)
            entry.supersedes = supersedes

        # Pre-validate corrects with an EXACT match — the kill-edge twin of supersedes
        # (V1a's second kill-edge; the target leaves the active view). Same Phase-A
        # read-only fast-fail shape, so a miss writes nothing.
        if corrects:
            ids = self.store.resolve_slug(corrects)
            if not ids:
                return _record_error("corrects_not_found", corrects=corrects)
            if len(ids) > 1:
                return _record_error("corrects_ambiguous", corrects=corrects)
            target = self.store.get_node(ids[0])
            if not target or target.get("slug", "").casefold() != corrects.casefold():
                return _record_error("corrects_not_found", corrects=corrects)
            entry.corrects = corrects

        # Validate every other typed relation EXACTLY like supersedes — each must
        # point at a real, unambiguous decision. Still Phase A: a miss returns an
        # error and writes nothing, so the buffer stays byte-for-byte unchanged.
        for _name, _target in extra_relations.items():
            err = self._validate_relation_target(_name, _target)
            if err:
                return err
            setattr(entry, _name, _target)

        # Identity (slug-free canonical-core hash — V1-D2). Computed over the SAME
        # fields commit_parsed_entry hashes, so this pre-commit idempotency id equals
        # the commit id: a same-core re-record with a new --slug is an in-place UPDATE
        # (slug rename), never a spurious slug_collision (G3, V1-D16).
        node_id = compute_node_id(
            kind=entry.kind,
            axiom=entry.axiom,
            mechanism_refs=entry.mechanisms,
            topic=entry.topic,
            questions_raised=entry.questions_raised,
        )

        # Idempotency (M2) fast-fail.
        existing = self.store.get_node(node_id)
        if existing:
            return {
                "slug": existing["slug"],
                "id": node_id,
                "state": self._node_state(node_id),
                "embedding": self._embedding_status(node_id),
                "status": "exists",
                "path": self.config.decisions_file,
            }

        # Slug-collision fast-fail (exact match only).
        coll_id, _coll_node = self._exact_slug_node(entry.slug)
        if coll_id and coll_id != node_id:
            return _record_error("slug_collision", slug=entry.slug)

        # Near-duplicate / possible-tension review (P4) — still Phase A, so a pause
        # writes NOTHING (buffer byte-for-byte unchanged). Surfacing the neighbour now,
        # not in the post-commit `related` echo, is the point: after commit the author
        # can no longer point a relation at it (a re-record is a no-op). Offline-safe
        # (no embeddings → no pause) and bypassable with acknowledge_neighbors=True.
        if not acknowledge_neighbors:
            declared_targets = {
                t.casefold() for t in (
                    ([supersedes] if supersedes else [])
                    + ([corrects] if corrects else [])
                    + list(extra_relations.values())
                )
            }
            neighbors = self._review_neighbors(entry, declared_targets)
            if neighbors:
                return {
                    "status": "needs_review",
                    "code": "similar_decision_exists",
                    "slug": entry.slug,
                    "neighbors": neighbors,
                    "message": (
                        f"Paused: '{entry.slug}' is ≥{_NEIGHBOR_REVIEW_THRESHOLD:.2f} "
                        f"similar to {len(neighbors)} existing decision(s) you did not "
                        "reference. If it amends/supersedes/contradicts/cites one, "
                        "re-record with that relation pointing at the neighbour's slug; "
                        "if it is genuinely independent, re-record with "
                        "acknowledge_neighbors=True. Nothing was written."
                    ),
                }

        # === Phase B — the only writes, fully serialised under one lock ===
        try:
            with self.lock:
                # Re-run the authoritative state checks INSIDE the lock, BEFORE any
                # buffer write (closes the TOCTOU window: a racer that committed
                # since Phase A is now seen — and a rejection here still leaves the
                # buffer byte-for-byte unchanged, since auto-heal hasn't run yet).
                if self.store.get_node(node_id):
                    return {
                        "slug": entry.slug,
                        "id": node_id,
                        "state": self._node_state(node_id),
                        "embedding": self._embedding_status(node_id),
                        "status": "exists",
                        "path": self.config.decisions_file,
                    }
                coll_id, _coll_node = self._exact_slug_node(entry.slug)
                if coll_id and coll_id != node_id:
                    return _record_error("slug_collision", slug=entry.slug)

                # Guarantee header + marker, then anchor the buffer for rollback.
                self.auto_heal_decisions_file()
                with open(self.config.decisions_file, "r", encoding="utf-8") as f:
                    original_content = f.read()

                # Compute the new buffer (newest-first, replacing ONLY the first marker).
                if _ENTRIES_MARKER in original_content:
                    new_content = original_content.replace(
                        _ENTRIES_MARKER, f"{_ENTRIES_MARKER}\n\n{entry_text}", 1
                    )
                else:
                    new_content = original_content.rstrip("\n") + f"\n\n{entry_text}"

                # Provenance (mirror perform_sync).
                entry.confirmed_by = actor
                entry.confirmed_at = datetime.now().isoformat()

                # Write the buffer, then commit the graph. On ANY failure of either
                # — including an OSError on the write itself — roll the buffer back
                # so a failure leaves NO orphan entry, and return JSON (never raise).
                try:
                    with open(self.config.decisions_file, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    delta = self.store.commit_parsed_entry(entry)
                except (ValidationError, DatabaseError, OSError) as commit_exc:
                    # Roll the buffer back so a failed write/commit leaves NO orphan.
                    try:
                        with open(self.config.decisions_file, "w", encoding="utf-8") as f:
                            f.write(original_content)
                    except Exception as restore_exc:
                        return {
                            "error": (
                                "The commit failed AND decisions.md may still hold the "
                                f"uncommitted entry (rollback error: {restore_exc}) — run "
                                f"'mitos sync' to reconcile. Underlying commit error: {commit_exc}."
                            ),
                            "code": "commit_failed",
                        }
                    return _record_error("commit_failed", reason=str(commit_exc))
        except Timeout:
            return _record_error(
                "commit_failed", reason="another Mitos process holds the decisions.md lock"
            )

        # 8. Embed best-effort (queues to the outbox if Gemini/Qdrant are down).
        vector: Optional[List[float]] = None
        try:
            vector = self._best_effort_embed(delta, entry)
        except Exception as e:
            # stderr: the MCP write tool shares this path and uses stdout for JSON-RPC.
            print(f"[Warning] Embedding step failed for '{entry.slug}': {str(e)}", file=sys.stderr)

        # 9. Re-render live_axioms.md (a render failure must not fail the commit).
        #    The renderer records size-ceiling overflows on `.overflows` instead of
        #    printing them, so we can attach ONE debounced summary to the result below
        #    — after the success receipt — rather than burying it under a per-file wall.
        overflow_summary: Optional[str] = None
        try:
            renderer = MitosRenderer(self.config.workspace_dir)
            renderer.render_all(self.store)
            if renderer.overflows and hint_due(
                "scope_overflow_hint.json", self.config.workspace_dir, 24 * 60 * 60
            ):
                overflow_summary = summarize_overflows(renderer.overflows)
        except Exception as e:
            # stderr: the MCP write tool shares this path and uses stdout for JSON-RPC.
            print(f"[Warning] Failed to render active axioms: {str(e)}", file=sys.stderr)

        # 10. Return. A freshly recorded decision is always active. Adjacency is a
        #     post-commit, fail-silent guardrail — surfacing it here never affects the
        #     write contract (the commit already succeeded above).
        result: Dict[str, Any] = {
            "slug": entry.slug,
            "id": node_id,
            "state": "active",
            "embedding": self._embedding_status(node_id),
            "status": "created",
            "path": self.config.decisions_file,
        }
        related = self._adjacent_decisions(vector, exclude_slug=entry.slug)
        if related:
            result["related"] = related
        # Debounced, presentation-only: a one-line "N files over the size ceiling — run
        # `mitos status`" nudge, never on the burying-the-receipt critical path. Shared
        # by both surfaces (CLI prints it after the receipt; MCP returns it structured).
        if overflow_summary:
            result["scope_overflow"] = overflow_summary
        return result
