# Mitos (v0.1)

![Status: Alpha](https://img.shields.io/badge/status-alpha-orange) ![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue) ![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-green)

> ⚠️ **Alpha** — Core functionality works and passes its test suite, but has not been validated in real-world sustained use. Expect rough edges. The API surface may shift before v1.

Mitos is a strict, deterministic, local-first architectural decision graph system designed to prevent citation rot and establish a bidirectional LLM-integration boundary. It maps decisions (`decisions.md` write-buffer) into a structured graph stored in SQLite and semantically indexed in Qdrant, dynamically deriving state from typed relations.

---

## 🌌 Core Principles (The M-System)

*   **M1: Canonical Core Hash**: Node identities are computed as an immutable SHA-256 hash of their kind, slug, core axiom, mechanisms, and questions. Changing the core content generates a new node identity linked via a typed transition edge (`corrects` or `supersedes`).
*   **M2: Cross-Store Identity**: Identifiers are mapped deterministically. To satisfy Qdrant's schema constraints, the 256-bit SQLite node ID is truncated to a 128-bit RFC-4122 UUID (`hash_to_uuid`), preserving the idempotency of re-upserts.
*   **M3: Dynamic State Derivation**: Node active states are computed dynamically at runtime based on the fixed-point resolution of the typed-edge graph:
    *   **Decision States**: `active` (default), `superseded` (target of a `supersedes` or `corrects` edge), or `amended` (target of an `amends` edge).
    *   **Open Question States**: `parked` (default) or `resolved` (target of a `resolves` edge originating from an *active* decision).
*   **M4: Auto-Healing Write Buffer**: If the write-buffer (`decisions.md`) is modified, deleted, or missing its header or sample format blocks, the sync pipeline automatically restores them under a brief file lock using the packaged single-source specification.
*   **M5: Rejected Alternatives Constraint**: For architectural integrity, every committed decision must document at least one alternative design and the reasons it was rejected (`**Rejected:**` block).
*   **M6: Mechanism Registry**: Automatically registers third-party dependencies, models, and technologies (e.g. `sqlite`, `gemini-3.5-flash`) into a dedicated registry table on commit.
*   **M7: Strict Parser & Skill**: High-fidelity transcript boundaries (`[DECISION_TRANSCRIPT]` / `[/DECISION_TRANSCRIPT]`) are parsed strictly using stripped exact-line matching, ensuring that textual references inside decision prose do not cause false positives.
*   **M8: Stateless Asset Rendering**: Axioms are compiled statelessly from the active graph into a single consolidated `live_axioms.md` asset, along with scope-level files.

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

Mitos enforces a strict **1:1 test-to-code byte ratio** constraint to ensure complete test coverage. Run the test suite sequentially to prevent SQLite transaction locking:

```bash
# Run the adversarial, pathological, and live scenario suites
PYTHONPATH=. pytest
```
