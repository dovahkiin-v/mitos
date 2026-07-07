"""T4 (5b half) — the pre-commit hook recipe, end-to-end in a temp git repo.

The `SETUP.md` pre-commit recipe (4b) is the artifact under test: the
`git diff --quiet -- decisions.md` divergence guard + `mitos check --staged`,
reproduced VERBATIM here (KD4 — a reword in either place diverges the test; the
`test_setup_recipe_is_in_lockstep` tripwire catches drift). Two disciplines:

* **Keyless deterministic half (always runs)** — drives the recipe's SHELL LOGIC
  through a real `git commit` with a FAKE `mitos` on PATH: the divergence guard
  fires (verbatim message) BEFORE any check runs, and the recipe wires the check's
  exit code straight to the commit's success/failure. Hermetic — no keys, no judge,
  no Qdrant; it tests the RECIPE, not the engine (the staged engine's bad-buffer→1
  behaviour is W9's `test_check_staged.py`).
* **Live half (`HAS_LIVE_KEYS`-gated)** — one real end-to-end with the shipped
  binary: a real workspace, a pending undeclared contradiction against an indexed
  decision → the hook blocks the commit; a clean (no-pending) buffer → the commit
  passes.

The divergence message's em-dash is U+2014 (byte-confirmed against SETUP.md:204).
"""

import os
import subprocess
import uuid

import pytest
import requests

# --- The recipe under test (VERBATIM from SETUP.md — keep in lockstep, KD4) ----
# The shebang is hook scaffolding (implementer's latitude); the load-bearing
# verbatim parts are the guard command and the divergence message.
RECIPE = """#!/bin/sh
# Fail loudly if decisions.md differs between the index and the working tree —
# `mitos check --staged` reads the WORKING TREE, git commits the INDEX; a divergence
# would gate the wrong bytes (a bad entry fixed-but-not-restaged slips the gate).
if ! git diff --quiet -- decisions.md; then
    echo "decisions.md has unstaged changes — stage or stash them before committing" >&2
    exit 1
fi
mitos check --staged
"""

# The verbatim divergence message the guard emits (em-dash U+2014).
DIVERGENCE_MESSAGE = (
    "decisions.md has unstaged changes — stage or stash them before committing"
)
# The verbatim guard command (the recipe's git plumbing).
GUARD_COMMAND = "git diff --quiet -- decisions.md"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SETUP_PATH = os.path.join(_REPO_ROOT, "SETUP.md")
_VENV_BIN = os.path.join(_REPO_ROOT, "venv", "bin")

# --- Reference-corpus delete pair (a known-good undeclared contradiction) ------
_HARD_DELETE = (
    "### harbor-delete-is-immediate-hard\n"
    "**Decided:** Harbor deletes are immediate and irreversible — the blob and its "
    "metadata are purged at once.\n"
    "**Rejected:** A grace period — regulated tenants require provable immediate "
    "erasure on request.\n"
    "**Scope:** storage\n"
)
_SOFT_DELETE = (
    "\n### harbor-delete-is-soft-30d\n"
    "**Decided:** Harbor deletes are soft: a deleted file is recoverable for 30 days "
    "before purge.\n"
    "**Rejected:** Immediate hard delete — one fat-fingered call loses a customer's "
    "data with no recourse.\n"
    "**Scope:** storage\n"
)


def _load_live_env() -> None:
    """Loads keys from the repo-root .env into os.environ (mirrors the live suites)."""
    env_path = os.path.join(_REPO_ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


_load_live_env()
HAS_LIVE_KEYS = bool(
    os.environ.get("GEMINI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")
)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:7333")


# --------------------------------------------------------------------------- #
# Scaffolding
# --------------------------------------------------------------------------- #

def _run(args, cwd, env=None):
    return subprocess.run(
        args, cwd=str(cwd), env=env, capture_output=True, text=True
    )


def _init_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "-q"], repo)
    _run(["git", "config", "user.email", "hook@test"], repo)
    _run(["git", "config", "user.name", "hook test"], repo)
    _run(["git", "config", "commit.gpgsign", "false"], repo)
    return repo


def _install_hook(repo):
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text(RECIPE, encoding="utf-8")
    hook.chmod(0o755)


def _fake_mitos(bin_dir, marker, exit_code):
    """Writes a fake `mitos` that records its invocation and exits a fixed code."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "mitos"
    script.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{marker}"\nexit {exit_code}\n',
        encoding="utf-8",
    )
    script.chmod(0o755)


def _env_with_path(bin_dir, extra=None):
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env.get("PATH", "")
    if extra:
        env.update(extra)
    return env


# --------------------------------------------------------------------------- #
# Lockstep tripwire
# --------------------------------------------------------------------------- #

def test_setup_recipe_is_in_lockstep():
    """The verbatim guard + message this test asserts still appear in SETUP.md (KD4).

    If 4b's recipe is reworded, this fails — the signal to update BOTH in lockstep.
    """
    setup = open(_SETUP_PATH, encoding="utf-8").read()
    assert GUARD_COMMAND in setup, (
        "the divergence guard command drifted from SETUP.md — update in lockstep"
    )
    assert DIVERGENCE_MESSAGE in setup, (
        "the divergence message drifted from SETUP.md — update in lockstep "
        "(check the em-dash is U+2014)"
    )
    # The em-dash is U+2014, not a hyphen or U+2013 (byte-confirmed).
    assert "—" in DIVERGENCE_MESSAGE


# --------------------------------------------------------------------------- #
# Keyless half — the recipe's shell logic through a real git commit
# --------------------------------------------------------------------------- #

def test_divergence_guard_fires_before_check(tmp_path):
    """A staged/worktree divergence fires the guard with the verbatim message, before check.

    The guard is pure git plumbing — no keys. Stage decisions.md, edit it without
    re-staging, then commit: the pre-commit hook detects the divergence and aborts
    with the verbatim message BEFORE `mitos check --staged` ever runs (the fake
    mitos's marker stays absent — the check never fired).
    """
    repo = _init_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    marker = tmp_path / "mitos_called"
    _fake_mitos(bin_dir, marker, 0)

    (repo / "decisions.md").write_text("### a\n**Decided:** v1.\n", encoding="utf-8")
    _run(["git", "add", "decisions.md"], repo)
    _run(["git", "commit", "--no-verify", "-m", "base"], repo)
    _install_hook(repo)

    # Diverge: worktree != index for decisions.md; stage an UNRELATED change to commit.
    (repo / "decisions.md").write_text("### a\n**Decided:** v2.\n", encoding="utf-8")
    (repo / "other.txt").write_text("x\n", encoding="utf-8")
    _run(["git", "add", "other.txt"], repo)
    r = _run(["git", "commit", "-m", "should-abort"], repo, env=_env_with_path(bin_dir))

    assert r.returncode != 0, "the commit should have been aborted by the guard"
    assert DIVERGENCE_MESSAGE in (r.stdout + r.stderr), (
        f"guard did not emit the verbatim message. output:\n{r.stdout}\n{r.stderr}"
    )
    assert not marker.exists(), (
        "the fake `mitos` was invoked — the guard did NOT precede the check"
    )


@pytest.mark.parametrize("exit_code,should_commit", [(0, True), (1, False)])
def test_recipe_wires_check_exit_to_commit(tmp_path, exit_code, should_commit):
    """With no divergence, the recipe wires `mitos check --staged`'s exit to the commit.

    A clean staged/worktree state passes the guard and runs `mitos check --staged`
    (the fake): exit 0 → the commit succeeds; exit 1 → the commit is blocked. This
    is the recipe's contract — the gate's verdict is the commit's verdict —
    independent of the (separately-tested) engine.
    """
    repo = _init_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    marker = tmp_path / "mitos_called"
    _fake_mitos(bin_dir, marker, exit_code)

    (repo / "decisions.md").write_text("### a\n**Decided:** v1.\n", encoding="utf-8")
    _run(["git", "add", "decisions.md"], repo)
    _run(["git", "commit", "--no-verify", "-m", "base"], repo)
    _install_hook(repo)

    # No divergence (decisions.md unchanged); stage an unrelated change to commit.
    (repo / "other.txt").write_text("x\n", encoding="utf-8")
    _run(["git", "add", "other.txt"], repo)
    r = _run(["git", "commit", "-m", "gated"], repo, env=_env_with_path(bin_dir))

    assert marker.exists(), "the guard passed but `mitos check --staged` never ran"
    if should_commit:
        assert r.returncode == 0, (
            f"exit-0 check should let the commit through. output:\n{r.stdout}\n{r.stderr}"
        )
    else:
        assert r.returncode != 0, "exit-1 check should block the commit"


# --------------------------------------------------------------------------- #
# Live half — the real binary, one full end-to-end
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not HAS_LIVE_KEYS,
    reason="GEMINI_API_KEY and ANTHROPIC_API_KEY both required — the live hook half "
    "drives the real `mitos check --staged` (embeddings + SONNET judge).",
)
def test_live_hook_blocks_bad_buffer_passes_clean(tmp_path):
    """The shipped `mitos check --staged`, wired into a real pre-commit hook, gates a commit.

    Seed an indexed hard-delete decision; a pending soft-delete buffer (the undeclared
    contradiction) → the hook blocks the commit (fail-closed: exit 1 on the finding, or
    exit 2 if a live substrate degrades — either blocks). Then a clean (no-pending)
    buffer → the no-pending short-circuit exits 0 and the commit passes. `mitos` is the
    real venv binary on PATH.
    """
    try:
        if requests.get(f"{QDRANT_URL.rstrip('/')}/collections", timeout=5).status_code != 200:
            pytest.skip(f"Qdrant unreachable at {QDRANT_URL} — environmental.")
    except requests.RequestException:
        pytest.skip(f"Qdrant unreachable at {QDRANT_URL} — environmental.")

    # A workspace dir named so its derived collection (mitos-<basename>) is swept by
    # conftest's mitos-tmp-* backstop even if teardown misses.
    ws = tmp_path / f"tmp-golden-hook-{uuid.uuid4().hex[:8]}"
    ws.mkdir()
    collection = f"mitos-{ws.name}"
    env = _env_with_path(_VENV_BIN)

    try:
        _run(["git", "init", "-q"], ws)
        _run(["git", "config", "user.email", "hook@test"], ws)
        _run(["git", "config", "user.name", "hook test"], ws)
        _run(["git", "config", "commit.gpgsign", "false"], ws)

        init = _run(["mitos", "init"], ws, env=env)
        assert init.returncode == 0, f"mitos init failed:\n{init.stdout}\n{init.stderr}"

        # Seed + index the hard-delete decision (non-interactive).
        (ws / "decisions.md").write_text(_HARD_DELETE, encoding="utf-8")
        sync = _run(["mitos", "sync", "--yes"], ws, env=env)
        if sync.returncode != 0:
            # A quota/service outage during seeding is environmental, not a defect.
            pytest.skip(
                f"seed `mitos sync --yes` failed (likely quota/service):\n"
                f"{sync.stdout}\n{sync.stderr}"
            )

        # Baseline commit (hook not installed yet), then install the hook.
        _run(["git", "add", "-A"], ws)
        _run(["git", "commit", "--no-verify", "-m", "base"], ws)
        _install_hook(ws)

        # BAD buffer: a pending undeclared contradiction; staged (no divergence).
        with open(ws / "decisions.md", "a", encoding="utf-8") as f:
            f.write(_SOFT_DELETE)
        _run(["git", "add", "decisions.md"], ws)
        bad = _run(["git", "commit", "-m", "bad-buffer"], ws, env=env)
        assert bad.returncode != 0, (
            f"the hook should BLOCK a pending undeclared contradiction. "
            f"output:\n{bad.stdout}\n{bad.stderr}"
        )

        # CLEAN buffer: revert to the baseline (no pending); stage an unrelated change.
        _run(["git", "reset", "--hard", "HEAD"], ws)
        (ws / "other.txt").write_text("x\n", encoding="utf-8")
        _run(["git", "add", "other.txt"], ws)
        clean = _run(["git", "commit", "-m", "clean-buffer"], ws, env=env)
        assert clean.returncode == 0, (
            f"the hook should PASS a no-pending buffer (short-circuit exit 0). "
            f"output:\n{clean.stdout}\n{clean.stderr}"
        )
    finally:
        try:
            requests.delete(
                f"{QDRANT_URL.rstrip('/')}/collections/{collection}", timeout=5
            )
        except requests.RequestException:
            pass
