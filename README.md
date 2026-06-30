# Mitos (v0.1)

![Status: Alpha](https://img.shields.io/badge/status-alpha-orange) ![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue) ![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)

> 🔧 **Early release** — actively developed

**In plain terms:** When you build software with AI assistants over months, the *reasoning* behind your decisions gets lost. The AI forgets why you chose one approach, re-suggests options you already rejected, and your design notes drift out of sync with what was actually decided. Mitos is a memory layer for those decisions — it records each decision, the alternatives you ruled out, and how later decisions replace earlier ones, then feeds that history back to your AI assistant in a compact, trustworthy form.

**The result:** your AI collaborator stays consistent with the calls you've actually made — it stops contradicting a past decision or re-opening a settled question, and your decision record never silently rots.

Mitos is a strict, deterministic, local-first architectural decision graph system designed to prevent citation rot and establish a bidirectional LLM-integration boundary. It maps decisions (`decisions.md` write-buffer) into a structured graph stored in SQLite and semantically indexed in Qdrant, dynamically deriving state from typed relations.

---

## Why mitos exists

Intense work with agents, lots of ADR's, too much to track.

I have a project being built through intensive, structured design reviews — and most of that reviewing happens between LLM's  (Claude and Gemini). Every review surfaces architectural decisions, and each one gets written down as an ADR (Architectural Decision Record). I started with a single plain `DECISIONS.md` file.

Within about a month of working this way, that file held **close to 900 ADRs**. On a very large codebase — the file was no longer greppable, readable in one sitting, or manageable by hand, and Claude was running **five separate linters** just to keep it honest. So I started building an LLM-native ADR tool: that tool became mitos.

Established ADR tools do exist — I only found that out afterwards. But they're built for human teams logging the occasional decision; none of them fit a solo developer working with LLMs that generate decisions faster than a person can track them.

## Who it's for

Mitos fits a specific way of working:

- a **solo developer** — not a team;
- working with **heavy LLM automation**, where AI assistants are actively making and recording architectural decisions;
- where those **decisions accumulate faster than you can track them by hand**.

Project size isn't a gate — mitos is useful on small projects too. But the larger the codebase and the higher the decision volume, the faster it crosses from *comfort* to *necessity*. A very small team might get mileage out of it as well — but it isn't a team-coordination tool.

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
├── live_axioms.md              # Compiled stateless active axioms list
├── tests/                      # 1:1 test-to-code byte ratio adversarial test package
└── .gitignore                  # Local repository exclusion list
```

---

## 🔑 API Keys & Credentials

Mitos loads keys from a **`.env` file at the workspace root**. `mitos init` scaffolds this file with empty, clearly-labelled slots (and adds it to `.gitignore`), so a human or LLM just drops the value in — no guessing variable names.

| Variable | Required? | Used for | Get one |
|---|---|---|---|
| **`GEMINI_API_KEY`** | **Required** for semantic search + sync | Embeddings (`surface`/`query`) **and** decision synthesis (`sync`/`capture`). **One key covers both.** | <https://aistudio.google.com/app/apikey> |
| `ANTHROPIC_API_KEY` | Optional | `mitos import --llm-extract` only (legacy prose ADR conversion). | <https://console.anthropic.com/settings/keys> |

- The single key to set is **`GEMINI_API_KEY`**. Without it, `mitos record` still works (it commits to the graph and queues the embedding to drain on the next `sync`), but semantic `surface`/`query` are unavailable.
- Keys are read from `.env` automatically on every `mitos` invocation; an explicit `export` in your shell takes precedence over the file.

## 🐳 Running Qdrant (its own instance)

Mitos keeps its vectors in **its own dedicated Qdrant on `:7333`** — deliberately *not* the standard `:6333`. If you already run Qdrant on `:6333` for other work, defaulting there would drop Mitos's collections into your instance and put them at risk of an unrelated "wipe all collections". `:7333` keeps Mitos isolated and **fails safe**: if its Qdrant isn't running, `record` still commits to the graph and queues embeddings — only semantic search pauses.

Start it once (shared across all your Mitos projects):
```bash
docker compose up -d        # from the mitos repo root → mitos-qdrant on :7333
```
One Qdrant instance, **one collection per project** (`mitos-<project>`), so projects never mix. To point Mitos at a different instance, set **`QDRANT_URL`** before `mitos init` (it's written into `config.toml`) or edit `qdrant_url` there. *(v0.2's `sqlite-vec` substrate will remove the separate-Qdrant requirement entirely.)*

## 🛠️ CLI Operations

### 1. Workspace Initialization
Sets up `.mitos/`, initializes the SQLite schema, copies the single-source `format-spec.md` to the workspace, and generates `.mitos/skill.md` cleanly.
```bash
python3 mitos/cli.py init
```
> **New project?** See **[SETUP.md](SETUP.md)** for the full per-project walkthrough (init → key → Qdrant → MCP wiring), and run `mitos status` to verify.

### 1b. Setup Check
Reports whether Mitos is set up for a project — `.mitos/` workspace, `decisions.md`, `GEMINI_API_KEY`, Qdrant reachability, the project's collection, and graph size — with a clear `READY ✓` / `NOT SET UP ✗` verdict, next-step hints, and an exit code (`0` ready / `1` not). Built for both humans and LLMs (`--json` for scripts).
```bash
mitos status [project-path]   # defaults to the current directory
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

### 5. Rebuild (upgrade path)
Re-commits the full corpus (`decisions.md` + the quarterly archives) through the **current** edge catalog and mechanism registry, into a build-aside graph that is swapped in atomically (the old graph is backed up to `graph.sqlite.bak_<timestamp>`). Use it after upgrading across a schema change — e.g. **0.3.x → 0.4.0**, where the in-place migration widens the schema but does *not* re-commit, so the newer edge types (`amends`/`narrows`/`depends_on`/`cites`/…) and the mechanism registry stay empty until you rebuild. **No decisions are ever at risk** — the markdown is the source of truth; an entry the current catalog rejects (e.g. a citation to a since-superseded decision) is surfaced as a punch-list, never silently dropped. `mitos status` nudges when a rebuild is due.
```bash
mitos rebuild                # refuses to swap if it would drop content; shows why
mitos rebuild --allow-drops  # proceed, accepting the drops (they stay in the markdown)
```

---

## 🧪 Testing

Run the test suite sequentially — SQLite transaction locking can make parallel runs flaky:

```bash
# Run the adversarial, pathological, and live scenario suites
PYTHONPATH=. pytest
```
