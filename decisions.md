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


