# Mitos Canonical Format Specification
*The single source of truth for the C5 contract (Skill ↔ Parser).*

This specification defines the strict, deterministic format for entries in `decisions.md`
(decisions) and `questions.md` (open questions). The Parser hard-fails on variants; the
Skill (V5) instructs LLMs to write exactly this format. **Kind is determined by file** —
there is no inline kind marker: an entry in `decisions.md` is a `decision`, an entry in
`questions.md` is an `open_question`. Both files share one preamble/entry-stream layout (§5).

## 1. Decision Entry

A Decision entry represents an architectural commitment, authored in `decisions.md`. It must
begin with a markdown header (`###` or `##`) followed by the slug.

### Canonical Core (Immutable)
These fields define the identity of the decision. Changing them creates a new node (with a
`corrects` or `supersedes` edge).
- `**Decided:**` (Required) The core axiom. Must be present and non-empty.
- `**Mechanisms:**` (Optional) Comma-separated list of mechanism entities (e.g., `sqlite`, `wal-mode`).

### Commentary Fields (Mutable)
These fields can be edited in place without creating a new node identity.
- `**Rejected:**` (Required — per M5) Paths considered and rejected, and why.
- `**Invalidates-If:**` (Optional) Conditions under which this decision should be re-evaluated.
- `**Scope:**` (Optional) Comma-separated tags (e.g., `backend`, `auth`).
- `**Context:**` (Optional) Free-form prose providing background.

### Provenance (Tool-Only, Optional)
Provenance of the entry. **Humans never author this field** — it is omitted from the
human-authored sample in §3.
- `**Source:**` (Optional) One of `user`, `capture_llm`, `import_llm`. Absent ⇒ `user`. Emitted by `mitos capture` (`capture_llm`) and `mitos import` (`import_llm`); it lives in the markdown so it survives a wipe-and-rebuild.

### Relationship Fields (Edges)
These create typed edges in the graph. Values must be valid slugs.
- `**Supersedes:**` [slug]
- `**Corrects:**` [slug]
- `**Amends:**` [slug]
- `**Narrows:**` [slug]
- `**Depends-On:**` [slug]
- `**Resolves:**` [slug]
- `**Contradicts:**` [slug]
- `**Derives-From:**` [slug]
- `**Cites:**` [slug]

In V1a the graph actively commits the two kill-edges, `Supersedes` and `Corrects`. The
remaining edges are recognized and reserved (warn-deferred until V1b).

### Transcripts
A verbatim transcript of the LLM conversation that led to the decision. `[DECISION_TRANSCRIPT]`
is the **sole** inline marker. Must be bounded exactly by:
`[DECISION_TRANSCRIPT]`
... text ...
`[/DECISION_TRANSCRIPT]`

## 2. Open Question Entry

An Open Question represents an unresolved architectural thread, authored in `questions.md`.
**Kind is determined by file** — there is no inline marker; an entry in `questions.md` is an
`open_question`. It must begin with a markdown header (`###` or `##`) followed by the slug
(identical slug tokenization to decisions).

### Canonical Core (Immutable)
Identity is `{kind, topic, questions}`. The questions are kept in **authored order** — prose
order is identity-significant and is never sorted.
- `**Topic:**` (Required) The subject of the open question. Must be present and non-empty.
- `**Questions:**` (Required) The explicit questions raised, in authored order. Must be non-empty.

### Commentary Fields (Mutable)
- `**Scope:**` (Optional) Comma-separated tags (cross-kind).

The `**Rejected:**` field is **not** permitted on an open question — it is decision-only (M5).
An open question carries no inline marker of its own; kind is the file, not a tag.

## 3. Sample Entry
This decision sample is included at the top of `decisions.md` to guide LLMs and humans
(`Source` is omitted — it is tool-only).

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

## 4. Open Question Sample
This open-question sample is included at the top of `questions.md` to guide LLMs and humans.

```markdown
### example-open-question

**Topic:** Embedding model selection for v0.2 semantic surface
**Questions:** Do we pin one embedding model or allow per-project choice? What is the re-embed cost if we switch after the corpus grows past ~1k nodes?
**Scope:** substrate, embeddings
```

## 5. Entry-Stream Sentinel & File Layout

Both `decisions.md` and `questions.md` share one layout. A file is split into a **preamble**
and an **entry stream** by a sentinel comment:

`<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->`

- **Preamble** — everything up to **and including** the sentinel line (file-header comments plus the canonical sample). The preamble yields **zero** graph state.
- **Entry stream** — the `##` / `### slug` blocks after the sentinel, newest first.
- A file with **no sentinel** is treated as wholly an entry stream (no preamble).
- Inside the entry stream, **HTML comments are literal field text** — they are not skipped or stripped. Only preamble comments are non-content.
