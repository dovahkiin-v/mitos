# Decisions for Mitos

<!-- This file is managed by mitos. LLM integration: see .mitos/skill.md once V5 ships. -->
<!-- DO NOT MODIFY ABOVE THIS LINE -->

## SAMPLE FORMAT — auto-restored by mitos sync, do not modify or delete

### example-slug

**Decided:** We will use SQLite in WAL mode for the graph store.
**Rejected:** pgvector (too heavy for local-first portfolio audience), sqlite-vec (defer to v0.2 to preserve V1 ship date).
**Mechanisms:** sqlite, wal-mode
**Scope:** substrate
**Context:** We need a local-first graph that supports concurrent reads and writes gracefully.

[DECISION_TRANSCRIPT]
User: Let's use Postgres.
Claude: That breaks the local-first requirement in P10. Let's use SQLite.
[/DECISION_TRANSCRIPT]

<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->

### mitos-status-and-per-project-setup

**Decided:** Mitos ships a 'mitos status [path]' command (exit 0 ready / 1 not, plus --json) and a per-project SETUP.md walkthrough, so any human or LLM can check whether Mitos is set up for a project and follow a clear path to set it up; the setup skill also adds a SETUP.md pointer to the project's agent files (AGENTS.md/CLAUDE.md/GEMINI.md) noting it is per-project.
**Rejected:** Make the check an MCP tool only — the MCP server only runs once Mitos is already wired, but the whole point is checking projects where it is NOT, so a CLI command works regardless. Leave setup discovery to prose docs alone — humans and LLMs must find the right doc and guess the steps.
**Mechanisms:** cli, mcp
**Scope:** setup, cli, llm-integration
**Context:** Mitos is per-project; onboarding a new project needs init + key + Qdrant + MCP wiring. 'mitos status' makes that state machine-checkable; SETUP.md + a setup skill make the walkthrough findable; the AGENTS.md pointer propagates Mitos-awareness to every agent that later opens the project.


### init-scaffolds-gitignored-env

**Decided:** mitos init scaffolds a gitignored .env at the workspace root with explicit empty credential slots (GEMINI_API_KEY required; ANTHROPIC_API_KEY optional), and the CLI auto-loads .env, so credential setup is unambiguous for any human or LLM.
**Rejected:** Leave credential discovery to docs only — humans and LLMs must guess env-var names and where to put them. Commit a non-gitignored .env template — risks leaking real keys once filled.
**Mechanisms:** dotenv, gemini-api, anthropic-api
**Scope:** setup, config, llm-integration
**Context:** Mitos reads keys straight from os.environ; previously the CLI loaded no .env (only tests did) and init created none, so a setting-up human/LLM had no signposted place for keys. init now scaffolds + gitignores .env and the CLI loads it on startup (dependency-free, P19).


### mitos-dedicated-qdrant-port

**Decided:** mitos defaults its Qdrant endpoint to a dedicated port (http://localhost:7333) and ships a docker-compose for it, so mitos never co-locates its collections inside whatever Qdrant the user already runs on the standard :6333; QDRANT_URL still overrides.
**Rejected:** Default to the standard :6333 — fails dangerous: it silently joins the user's existing general-purpose Qdrant (the standard port is the MOST-likely-occupied one) and shares its wipe/contamination risk. 'Keep :6333 because it is standard' is backwards: standard = most-likely-occupied = highest risk.
**Mechanisms:** qdrant, docker-compose
**Scope:** vector-store, config, setup
**Context:** Per-project collections prevent mitos-internal mixing but do NOT protect against landing in a DIFFERENT app's Qdrant on :6333. A mitos-dedicated :7333 fails safe (semantic degrades if mitos's Qdrant is down) instead of dangerous. v0.2 sqlite-vec removes the separate-Qdrant need entirely.


### per-project-qdrant-collection

**Decided:** Each Mitos workspace derives its own Qdrant collection name (mitos-<project>) by default, so a single shared Qdrant instance never mixes decisions across projects.
**Rejected:** Shared single 'mitos' collection across all projects — causes cross-project semantic contamination and content-hash point_id collisions when two projects record the same axiom. Separate Qdrant instance per project — operational overhead with no benefit, since collections already isolate within one instance.
**Mechanisms:** qdrant, content-hash-identity
**Scope:** vector-store, config
**Context:** config.qdrant_collection defaulted to a hardcoded 'mitos' and 'mitos init' wrote the same literal into every config.toml; point ids are hash_to_uuid(content_hash) per M2, so identical axioms across projects would collide. The fix derives the collection from the workspace basename, overridable via .mitos/config.toml.



