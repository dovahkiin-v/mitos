# Working on Mitos

**Mitos** is an architectural-decision substrate for LLM-native workflows: markdown
for humans, a graph for the agents that work alongside them. Per-project `.mitos/`
workspace + a per-project Qdrant collection on a shared instance (`:7333`).

## ⚠️ Release ritual — bump the version on every meaningful change

**When you commit a change worth flagging to users, bump `__version__` in
`mitos/__init__.py`** (e.g. `0.1.2` → `0.1.3`).

- It is the **single source of truth** — `setup.py` reads it, and the CLI's
  once-a-day update check compares the installed `__version__` against the one on
  `main`. If you don't bump it, users on the old build get **no "update available"
  nudge** and `pipx`/`pip` may treat the package as unchanged.
- Bump it **in the same commit** as the change. Patch bump for fixes/small
  features; minor bump for larger ones. There's no PyPI release — the git `main`
  HEAD *is* the release.
- Skip the bump only for non-shipping changes (docs typos, this file, tests).

## Architecture orientation

- `cli.py` — all CLI verbs (`init`, `status`, `set-key`, `sync`, `record`,
  `surface`, `query`, `import`, `render`, `serve`, …) + the argparse router. The
  MCP tool names (`record_decision`/`surface_decisions`/`query_decisions`) are
  accepted as **CLI aliases**, and `record`/`surface`/`query` mirror the MCP tools.
- `mcp_server.py` — the FastMCP server (`mitos serve`). Three agent tools:
  `record_decision`, `surface_decisions`, `query_decisions`. This is the
  recommended agent interface (structured args, no shell-quoting). The CLI is the
  substrate + fallback. **Keep the CLI and MCP implementations behaviourally in
  sync** — they are parallel surfaces over the same store.
- `sync.py` — `MitosSyncManager`, including `record_decision_entry`: the agentic
  write path. It has a **strict buffer-first + rollback contract** (validate fully
  in memory → append buffer + commit graph under one lock → roll back the buffer
  byte-for-byte on any failure). Treat this contract as sacred; never weaken it for
  convenience.
- `store.py` — SQLite graph (nodes + typed edges: `supersedes`, `amends`,
  `narrows`, `depends_on`, `resolves`, …) and computed decision state.
- `vector_store.py` / `embeddings.py` — Qdrant REST + Gemini embeddings. Fail safe:
  if Qdrant/Gemini is down, `record` still commits to the graph and queues the
  embedding for the next `sync`; only semantic surface/query pause.
- `config.py` — `MitosConfig`, per-project collection naming, and
  `global_env_path()` (`~/.config/mitos/.env`). Keys resolve **env → project
  `.env` → global `.env`**.
- `format-spec.md` — the canonical decision format. It is bundled as package data
  (`MANIFEST.in` + `package_data`); a real install reads it from the installed
  package dir, so it MUST ship in the wheel.

## Dev workflow

- **Edit + test against the editable dev install**: `./venv/bin/mitos …` and
  `./venv/bin/python -m pytest`.
- **Refresh the global (pipx) install** after pushing:
  `pipx install --force git+https://github.com/dovahkiin-v/mitos` (use `--force` —
  a git install can no-op on `pipx upgrade`).
- **Before pushing**, secret-scan the diff (the repo is public; `.env` files are
  gitignored and must never be committed).
- **Dogfood the real install path** for packaging-sensitive changes: a fresh
  non-editable `pip install` into a throwaway venv, then `mitos init` — an editable
  install hides missing package-data bugs.

## Principles

- **P19 — Dependency Skepticism.** Prefer a small hand-rolled parse over a new
  dependency (see the manual `.env` and `config.toml` readers). Justify every new
  `install_requires` entry.
- **Google-style docstrings, type hints in signatures.** Match the surrounding
  code's density and idiom.
- **Empty/fresh states are first-class.** A just-initialized project (no decisions,
  no collection yet) is healthy, not broken — never make "healthy and empty" look
  like "broken."
