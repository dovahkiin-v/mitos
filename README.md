# Mitos (v0.1)

![Status: Alpha](https://img.shields.io/badge/status-alpha-orange) ![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue) ![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)

> 🔧 **Early release** — actively developed

Mitos is a strict, deterministic, local-first architectural decision graph system designed to prevent citation rot and establish a bidirectional LLM-integration boundary. It maps decisions (`decisions.md` write-buffer) into a structured graph stored in SQLite and semantically indexed in Qdrant, dynamically deriving state from typed relations.

---

## 🌌 Core Principles (The Mitos Kernel)

*   **M1. Axiom Immutability**: The canonical core of a node (axiom + mechanisms for decisions; topic + questions for open questions) is strictly immutable. Changes generate a new node linked via `corrects` or `supersedes` edges. Commentary fields outside the core remain mutable in place.
*   **M2. Content-Hash Identity (with Human-Readable Slugs)**: Node IDs are cryptographic SHA-256 hashes of their canonical core to prevent parallel work collisions. Human-readable slugs are used for citations. In Qdrant, the 256-bit hash is mapped to a 128-bit RFC-4122 UUID (`hash_to_uuid`) to preserve re-upsert idempotency.
*   **M3. State is Computed, Not Stored**: Active states (`active`, `superseded`, `drifted` for decisions; `parked`, `resolved` for open questions) are derived dynamically from edges and signals at query time. No static `status` field exists, making status drift impossible within the graph.
*   **M4. Three Retrieval Modes (Consumer Chooses)**: The system supports Letter, Trace, and Vibe modes. v0.1 ships with **Letter-only** retrieval, exposing only the axiom and rejected paths (~200–500 tokens) to keep LLM context thin and precise.
*   **M5. Anti-Knowledge as First-Class**: Mandates `rejected_paths` as a required field. This acts as a critical constraint boundary, preventing LLM partners from defaulting to common but incorrect patterns that have already been discarded.
*   **M6. Typed Mechanism Entities (and Verification Anchors)**: Mechanisms (e.g. `sqlite`, `wal-mode`) are first-class registry entities rather than plain text strings. This enables dependency auditing and serves as the substrate for future automated drift sensing.
*   **M7. Markdown is a Transient Render**: The SQLite graph is the database of record; Markdown files like `decisions.md` are write-buffers, and output files like `live_axioms.md` are transient projects generated or parsed on demand.
*   **M8. Always Derive From Primary Sources**: Summary projections are always regenerated fresh from primary active sets (not from summaries of summaries), preventing semantic loss and accumulation of LLM drift over time.

---

## 📂 Project Directory Structure

```
mitos_2/
├── .mitos/                     # Mitos configuration and state
│   ├── graph.sqlite            # The SQLite local-first graph store (Gitignored)
│   ├── embedding_cache.sqlite  # Token-aware embedding cache database (Gitignored)
│   ├── config.toml             # Configuration variables
│   └── skill.md                # Generated single-source LLM prompt
├── decisions/                  # Sync archives
│   └── archive/
│       └── 2026-Q2.md          # Quarterly rotated write-buffer entries
├── mitos/                      # Mitos python package source
│   ├── cli.py                  # CLI commands and entry points
│   ├── store.py                # GraphStore engine, state derivation, and transaction protocols
│   ├── sync.py                 # File locking, snapshotting, and quarterly rotation
│   ├── parser.py               # Deterministic comment stripper and Markdown parser
│   ├── importer.py             # Legacy prose converter (Claude Sonnet-driven)
│   ├── format-spec.md          # Packaged single-source format specifier
│   └── ...
├── decisions.md                # Write-buffer decisions file (Auto-healed)
├── format-spec.md              # Global single-source format specification file
├── live_axioms.md              # Compiled stateless active axioms list
├── tests/                      # 1:1 test-to-code byte ratio adversarial test package
└── .gitignore                  # Local repository exclusion list
```

---

## 🛠️ CLI Operations

### 1. Workspace Initialization
Sets up `.mitos/`, initializes the SQLite schema, copies the single-source `format-spec.md` to the workspace, and generates `.mitos/skill.md` cleanly.
```bash
python3 mitos/cli.py init
```

### 2. Transactional Synchronization
Parses `decisions.md` under file lock, runs LLM enrichment on new entries, resolves outgoing edges case-insensitively (using prefix-fallback for legacy ADR formats), commits changes atomically, updates Qdrant, and rotates successfully synced entries to quarterly archives once the volume threshold is met.
```bash
# Requires environment credentials (GEMINI_API_KEY / GOOGLE_API_KEY)
PYTHONPATH=. python3 mitos/cli.py sync --yes
```

### 3. Legacy Prose Import
Imports legacy Markdown ADR files, compressing their prose into canonical Mitos structures using Claude Sonnet.
```bash
# Requires ANTHROPIC_API_KEY
PYTHONPATH=. python3 mitos/cli.py import /path/to/legacy.md --llm-extract
```

### 4. Axiom Rendering
Manually compiles active axioms into `live_axioms.md`.
```bash
PYTHONPATH=. python3 mitos/cli.py render
```

---

## 🧪 Testing

Run the test suite sequentially — SQLite transaction locking can make parallel runs flaky:

```bash
# Run the adversarial, pathological, and live scenario suites
PYTHONPATH=. pytest
```
