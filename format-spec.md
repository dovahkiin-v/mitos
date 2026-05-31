# Mitos Canonical Format Specification
*The single source of truth for the C5 contract (Skill ↔ Parser).*

This specification defines the strict, deterministic format for entries in `decisions.md`. The Parser (V1) hard-fails on variants. The Skill (V5) instructs LLMs to write exactly this format.

## 1. Decision Entry

A Decision entry represents an architectural commitment. It must begin with a markdown header (`###` or `##`) followed by the slug.

### Canonical Core (Immutable)
These fields define the identity of the decision. Changing them creates a new node (with a `corrects` or `supersedes` edge).
- `**Decided:**` (Required) The core axiom. Must be present and non-empty.
- `**Mechanisms:**` (Optional) Comma-separated list of mechanism entities (e.g., `sqlite`, `gemini-3.1-flash-lite`).

### Commentary Fields (Mutable)
These fields can be edited in place without creating a new node identity.
- `**Rejected:**` (Required - per M5) Paths considered and rejected, and why.
- `**Invalidates-If:**` (Optional) Conditions under which this decision should be re-evaluated.
- `**Scope:**` (Optional) Comma-separated tags (e.g., `backend`, `auth`).
- `**Context:**` (Optional) Free-form prose providing background.

### Relationship Fields (Edges)
These create typed edges in the graph. Values must be valid slugs.
- `**Supersedes:**` [slug]
- `**Amends:**` [slug]
- `**Narrows:**` [slug]
- `**Depends-On:**` [slug]
- `**Resolves:**` [slug]
- `**Contradicts:**` [slug]
- `**Derives-From:**` [slug]
- `**Cites:**` [slug]

### Transcripts
A verbatim transcript of the LLM conversation that led to the decision.
Must be bounded exactly by:
`[DECISION_TRANSCRIPT]`
... text ...
`[/DECISION_TRANSCRIPT]`

## 2. Open Question (Parked)

An open question represents a paused architectural thread.
Must use the exact inline marker:
`[DECISION_PARKED: topic]`

Following the marker, the entry must contain the `**Questions:**` field.
- `**Questions:**` (Required) The explicit questions raised. Must be non-empty.

## 3. Sample Entry
This sample is included at the top of `decisions.md` to guide LLMs and humans.

```markdown
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
```
