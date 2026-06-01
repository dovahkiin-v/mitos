"""Sync pipeline for Mitos.

This module implements the core V3a and V3b sync loops, managing snapshotting,
concurrency file locks, LLM capture enrichment, user reviews, and content-aware
archive rotation.
"""

import os
import shutil
import re
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from filelock import FileLock, Timeout
from google import genai
from google.genai import types

from mitos.config import MitosConfig
from mitos.errors import SynthesisError, ParseError, ValidationError
from mitos.models import get_model_id
from mitos.parser import ParsedEntry, parse_decisions_file
from mitos.store import GraphStore, CommitDelta, compute_hash
from mitos.embeddings import GeminiEmbeddingProvider
from mitos.vector_store import QdrantVectorStore
from mitos.renderer import MitosRenderer

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
Decided: {entry.core_axiom}
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

        # 2. Parse from the snapshot
        with open(snapshot_path, "r", encoding="utf-8") as f:
            snapshot_text = f.read()
        entries = parse_decisions_file(snapshot_text)

        if not entries:
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

            # Check if this node is already in the database
            node_id = compute_hash(
                entry.kind,
                entry.slug,
                entry.core_axiom,
                entry.mechanisms,
                entry.questions_raised
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
                print(f"  New Axiom:      {entry.core_axiom}")
                
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
                refined_axiom = enrichment.get("refined_core_axiom", entry.core_axiom)
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

                # Commit to local variables for database insert
                entry.core_axiom = refined_axiom
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

    def _best_effort_embed(self, delta: CommitDelta, entry: ParsedEntry) -> None:
        """Best-effort async embedding upsert pipeline (C2)."""
        embedding_text = entry.core_axiom if entry.kind == "decision" else f"{entry.slug}: " + " ".join(entry.questions_raised)
        
        if not self.embed_provider or not self.vector_store:
            try:
                self.store.add_pending_embedding(delta.node_id, embedding_text)
                print(f"[Warning] Embedding upsert deferred for '{entry.slug}': Embedding provider down.")
            except Exception as e:
                print(f"[Warning] Failed to write outbox queue: {str(e)}")
            return

        # Prepare payload
        payload = {
            "slug": entry.slug,
            "scope": entry.scope,
            "state": "active",
            "kind": entry.kind,
            "embedding_text": embedding_text
        }

        try:
            # Check embedding provider and generate vector
            vector = self.embed_provider.get_embedding(payload["embedding_text"], is_query=False)
            self.vector_store.upsert(delta.node_id, vector, payload)
        except Exception as e:
            # Failed embeddings land in graph Outbox queue (C2)
            print(f"[Warning] Embedding upsert deferred for '{entry.slug}': {str(e)}")
            try:
                self.store.add_pending_embedding(delta.node_id, embedding_text)
            except Exception as dbe:
                print(f"[Warning] Failed to write outbox queue: {str(dbe)}")

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
                embedding_text = item["embedding_text"]
                
                # Fetch node details from graph for Qdrant payload
                node = self.store.get_node(node_id)
                if not node:
                    # Node has been deleted from graph; remove from queue
                    try:
                        self.store.remove_pending_embedding(node_id)
                    except Exception:
                        pass
                    continue

                payload = {
                    "slug": node["slug"],
                    "scope": node["scope"],
                    "state": "active",
                    "kind": node["kind"],
                    "embedding_text": embedding_text
                }

                try:
                    # 1. Fetch embedding vector
                    vector = self.embed_provider.get_embedding(embedding_text, is_query=False)
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
