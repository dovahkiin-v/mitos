"""Legacy ADR prose importer for Mitos.

This module implements the Importer capability (V6) and the import CLI surface,
using Anthropic Sonnet (complex tier) to deterministically compress prose into
canonical Mitos entries.
"""

import os
import json
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime
import anthropic

from mitos.config import MitosConfig
from mitos.errors import SynthesisError, ValidationError
from mitos.models import get_model_id
from mitos.parser import ParsedEntry, parse_header
from mitos.store import GraphStore, CommitDelta
from mitos.identity import compute_node_id, embedding_text
from mitos.embeddings import GeminiEmbeddingProvider
from mitos.vector_store import QdrantVectorStore
from mitos.renderer import MitosRenderer

def run_llm_prose_compression(client: anthropic.Anthropic, title: str, prose_content: str) -> Dict[str, Any]:
    """Uses Claude Sonnet to faithfully compress a legacy prose ADR into canonical fields."""
    prompt = f"""
You are the Mitos v0.1 import compression scribe. Your task is to compress a legacy architectural decision record (ADR) into standard, highly precise Mitos fields.

Do NOT elaborate, speculate, or introduce any concepts not present in the source text. Be completely faithful.

Title of ADR: {title}
Legacy Content:
{prose_content}

You must extract:
1. `core_axiom`: A single concise, punchy sentence representing the exact decision made.
2. `rejected_paths`: A detailed list/summary of alternative paths considered and rejected in the prose, and why. MUST be present and non-empty. If the prose doesn't explicitly name alternatives, identify what this decision blocks or overrides based solely on the text.
3. `mechanisms`: A list of specific technologies or modules mentioned.
4. `scope`: Tags classifying the subsystem (e.g., auth, backend, UI).
5. `supersedes`: The slug of any prior decision this entry replaces, if mentioned.
6. `amends`: The slug of any decision this entry amends, if mentioned.
7. `resolves`: The slug of any open question this entry resolves, if mentioned.

Respond strictly in valid JSON format with the following keys:
- core_axiom (string)
- rejected_paths (string)
- mechanisms (list of strings)
- scope (list of strings)
- supersedes (string or null)
- amends (string or null)
- resolves (string or null)
"""
    model_id = get_model_id("SONNET")
    try:
        message = client.messages.create(
            model=model_id,
            max_tokens=2000,
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        # Parse the JSON response
        text_resp = message.content[0].text.strip()
        # Find JSON boundaries in case of wrapper prose
        json_match = re.search(r'\{.*\}', text_resp, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return json.loads(text_resp)
    except Exception as e:
        raise SynthesisError(f"Claude import compression call failed: {str(e)}")


class MitosProseImporter:
    """Manages parsing legacy prose ADR files and importing them into the GraphStore."""

    def __init__(self, config: MitosConfig) -> None:
        self.config = config
        self.store = GraphStore(self.config.db_path)
        
        # Lazy load embeddings/vectors
        self.embed_provider = None
        self.vector_store = None
        try:
            cache_path = os.path.join(self.config.mitos_dir, "embedding_cache.sqlite")
            self.embed_provider = GeminiEmbeddingProvider(cache_path)
            self.vector_store = QdrantVectorStore(
                self.config.qdrant_url,
                self.config.qdrant_collection
            )
        except Exception:
            pass

    def split_prose_sections(self, text: str) -> List[Dict[str, Any]]:
        """Splits a legacy markdown prose file into heading-bounded sections."""
        lines = text.splitlines()
        sections = []
        current_section = None
        
        for idx, line in enumerate(lines, start=1):
            is_header = (line.startswith("##") or line.startswith("###")) and not line.startswith("####")
            if is_header:
                if current_section:
                    current_section["line_end"] = idx - 1
                    sections.append(current_section)
                current_section = {
                    "header": line,
                    "lines": [],
                    "line_start": idx,
                    "line_end": idx
                }
            if current_section:
                current_section["lines"].append(line)
                
        if current_section:
            current_section["line_end"] = len(lines)
            sections.append(current_section)
            
        return sections

    def import_from_file(self, filepath: str, use_llm_extract: bool = False) -> None:
        """Imports legacy prose ADRs from file into the graph store.

        Args:
            filepath: Path to the markdown file to import.
            use_llm_extract: True to use the Claude Sonnet compression pipeline.
        """
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        sections = self.split_prose_sections(raw_text)
        if not sections:
            print("No headings starting with ## or ### found in the import file.")
            return

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if use_llm_extract and not api_key:
            print("ANTHROPIC_API_KEY environment variable is not set. Import --llm-extract requires it.")
            return

        client = anthropic.Anthropic(api_key=api_key) if use_llm_extract else None
        renderer = MitosRenderer(self.config.workspace_dir)

        imported_count = 0

        for sec in sections:
            header_line = sec["header"]
            raw_content = "\n".join(sec["lines"][1:]).strip()
            line_start = sec["line_start"]
            line_end = sec["line_end"]

            try:
                slug, date, title = parse_header(header_line)
            except Exception:
                # Fallback slugify
                slug = re.sub(r'[^a-zA-Z0-9-]+', '-', header_line.lstrip("#").strip().lower()).strip("-")
                date = None
                title = header_line.lstrip("#").strip()

            entry = ParsedEntry("decision", slug, line_start, line_end)
            entry.date = date
            entry.title = title

            if use_llm_extract and client:
                print(f"Compressing: {slug} ...")
                try:
                    compressed = run_llm_prose_compression(client, title or slug, raw_content)
                except Exception as e:
                    print(f"[Warning] Failed to compress entry '{slug}': {str(e)}. Skipping.")
                    continue

                entry.axiom = compressed.get("core_axiom", "")
                entry.rejected_paths = compressed.get("rejected_paths", "")
                entry.mechanisms = compressed.get("mechanisms", [])
                entry.scope = compressed.get("scope", [])
                # Relationship fields are List[str] (V1b multi-valued); wrap the
                # single compressed slug into a 1-element list, [] when absent.
                entry.supersedes = [v] if (v := compressed.get("supersedes")) else []
                entry.amends = [v] if (v := compressed.get("amends")) else []
                entry.resolves = [v] if (v := compressed.get("resolves")) else []
            else:
                # If not using LLM, populate core fields with raw content as best effort
                entry.axiom = title or slug
                entry.rejected_paths = "No rejected paths specified in raw import."
                entry.context = raw_content

            # Standardize invariants check
            if not entry.axiom or not entry.rejected_paths:
                print(f"[Warning] Skipping '{slug}': missing core_axiom or rejected_paths.")
                continue

            # Assign imported metadata fields
            # GraphStore requires parsed entries
            try:
                # Populate OD3 confirmation metadata + V1a import provenance. The
                # import provenance rides ``nodes.source`` (V1-D20 enum: import_llm),
                # set on the entry BEFORE commit so it passes the DDL CHECK and is
                # fenced as canonical core (MI-4). The prototype's post-commit
                # ``UPDATE … SET source='imported', source_ref=?`` is retired (8a):
                # 'imported' is outside the V1a enum and ``source_ref`` is a dropped
                # column (§6.5) — the file:line provenance has no V1a home and is
                # deferred (a V1b importer concern), not silently crash-on-write.
                entry.source = "import_llm"
                entry.confirmed_by = get_model_id("SONNET") if use_llm_extract else "user"
                entry.confirmed_at = datetime.now().isoformat()

                # Compute the stable slug-free V1a id (V1-D2) — matches commit_parsed_entry.
                node_id = compute_node_id(
                    kind=entry.kind,
                    axiom=entry.axiom,
                    mechanism_refs=entry.mechanisms,
                )

                # Check duplication
                existing = self.store.get_node(node_id)
                if existing:
                    # Stable identity: re-running import on same content is a no-op
                    # (S2/S5). But a corpus re-imported over a hand-authored graph is
                    # the textbook cross-source case (existing 'user' vs new
                    # 'import_llm') — emit one source_reencounter audit signal before
                    # the skip (MI-4 / V1-D14). entry.source is "import_llm" (set
                    # above); the `or "user"` mirrors the other gates for uniformity.
                    # Best-effort, and additionally inside this per-entry try/except.
                    self.store.note_source_reencounter(
                        node_id, existing["source"], entry.source or "user"
                    )
                    continue

                # Add to DB via GraphStore commit helper
                delta = self.store.commit_parsed_entry(entry)
                imported_count += 1

                # Best-effort embedding upsert
                self._best_effort_embed(delta, entry)
            except Exception as e:
                print(f"[Warning] Failed to save imported node '{slug}': {str(e)}")

        print(f"Successfully imported {imported_count} nodes from {filepath} ✓")
        
        # Regenerate live_axioms
        try:
            renderer.render_all(self.store)
        except Exception:
            pass

    def _best_effort_embed(self, delta: CommitDelta, entry: ParsedEntry) -> None:
        """Best-effort embedding upsert pipeline for imported nodes."""
        if not self.embed_provider or not self.vector_store:
            return

        payload = {
            "slug": entry.slug,
            "scope": entry.scope,
            "state": "active",
            "kind": entry.kind,
            # V1a embedding text re-derived from the immutable core (M8) so it is
            # byte-consistent with the drain-time re-derivation (sync).
            "embedding_text": embedding_text({"kind": entry.kind, "axiom": entry.axiom})
        }

        try:
            vector = self.embed_provider.get_embedding(payload["embedding_text"], is_query=False)
            self.vector_store.upsert(delta.node_id, vector, payload)
        except Exception:
            pass
