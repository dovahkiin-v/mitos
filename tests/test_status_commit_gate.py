"""Phase 3f: the commit gate row on ``mitos status <project>``, and ``init``'s line.

The row reads what 3a–3e left in the world (git's answer, the key, the
``pre-commit`` file, the attempt record, the audit debt) and says it in one
place. It has to be honest in both directions: never "no hook" over a file that
exists, never ``⚠`` at a keyless project that could not use a gate, never a ``✓``
it did not verify, and never the cost of a project's ``READY ✓``.

Every row runs real ``git`` under isolation, with the cwd moved to ``tmp_path``
so a regression that asks git from the cwd cannot reach the checkout's own
``.git/hooks``. Hooks reach the fixtures through 3e's real verb.
"""

import json
import os
import shlex
import sqlite3
from typing import Any, Dict, List, Tuple

import pytest

from mitos import cli
from mitos.audit_debt import AuditDebt, derive_audit_debt
from mitos.commit_gate import HOOK_BLOCK_MARKER, HOOK_FILE_MARKER, render_hook_block
from mitos.config import MitosConfig
from mitos.store import GraphStore
from mitos.telemetry import (
    ATTEMPT_COULD_NOT_COMPLETE, ATTEMPT_NEW_FINDINGS, AttemptOutcome, AttemptRefusal,
    AttemptStart, CoverageMarks, TelemetryStore,
)
from test_check_coverage import _check_run_row
from test_check_probe import _commit
from test_hook_install import _install, _repo_with_workspace, _stub, _workspace
from test_process_fence import _needs_git, isolate_git, make_scratch_repo


@pytest.fixture(autouse=True)
def _git_isolation(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    isolate_git(monkeypatch, str(tmp_path))
    # pytest's cwd is the checkout: a cwd-leaning regression lands where git sees
    # no repository, never in the developer's own `.git/hooks`.
    monkeypatch.chdir(tmp_path)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_DUMMY_KEY = "sk-dummy-never-sent"

_KEYS = {"state", "hooks_dir", "hooks_dir_shared", "hook_file", "serves",
         "serves_this_workspace", "last_attempt_status", "last_attempt", "debt_status",
         "uncovered", "excluded"}

_STARTED_AT = "2026-09-21T00:00:00.000000+00:00"


def _keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", _DUMMY_KEY)


def _ready_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Gemini key and a reachable (stubbed) Qdrant: a fresh workspace reads READY."""
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", lambda url, coll: {
        "reachable": True, "collection_exists": False, "points": None})


def _status(ws: str, capsys: Any) -> Tuple[int, str, Dict[str, Any]]:
    """Runs text then JSON status on ``ws``; returns ``(rc, text, payload)``."""
    capsys.readouterr()
    rc = cli.cmd_status(ws)
    text = capsys.readouterr().out
    rc_json = cli.cmd_status(ws, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert rc == rc_json
    return rc, text, payload


def _row_lines(text: str) -> List[str]:
    """The gate row's block: its headline and the indented lines under it."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if " commit gate: " in line]
    assert len(starts) == 1, text
    block = [lines[starts[0]]]
    for line in lines[starts[0] + 1:]:
        if not line.startswith("      "):
            break
        block.append(line)
    return block


def _gate(ws: str, capsys: Any) -> Tuple[List[str], Dict[str, Any]]:
    """The row's text lines and ``commit_gate`` object, from one fixture."""
    _rc, text, payload = _status(ws, capsys)
    row = payload["commit_gate"]
    assert set(row) == _KEYS
    return _row_lines(text), row


def _glyph(lines: List[str]) -> str:
    return lines[0].split()[0]


def _recipe(line: str) -> str:
    """The backticked recipe after the row's ``→``."""
    return line.split("→", 1)[1].strip().strip("`")


def _assert_recipe_resolves(recipe: str, verb: str, ws: str) -> None:
    """Parses a printed recipe and resolves its selector back to ``ws``."""
    tokens = shlex.split(recipe)
    assert tokens[0] == "mitos"
    args = cli._build_parser().parse_args(tokens[1:])
    assert args.command == verb
    assert args.project_post is not None
    resolved = cli._resolve_selector(cli._selector_from_args(args), args.command)
    assert os.path.realpath(resolved.root) == os.path.realpath(ws)


def _installed(tmp_path, capsys: Any, sub: str = "") -> Tuple[str, str, str]:
    """A repository with a workspace and 3e's real hook; returns (repo, ws, hook_file)."""
    repo, ws = _repo_with_workspace(tmp_path, sub)
    code, _out, err = _install(ws, _stub(tmp_path / "bin"), capsys)
    assert code == 0, err
    return repo, ws, os.path.join(repo, ".git", "hooks", "pre-commit")


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class _Raises:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"the keyless row must not call {self.name}")


# --------------------------------------------------------------------------- #
# 1–3: no row, and the rows git decides
# --------------------------------------------------------------------------- #

def test_an_uninitialized_directory_has_no_gate_row(tmp_path, capsys) -> None:
    """Criterion 1: ``null``, never ``{}``, and no text line."""
    target = tmp_path / "plain"
    target.mkdir()
    _rc, text, payload = _status(str(target), capsys)
    assert "commit_gate" in payload and payload["commit_gate"] is None
    assert "commit gate" not in text


@_needs_git
def test_a_workspace_outside_any_repository(tmp_path, capsys) -> None:
    """Criterion 2."""
    ws = _workspace(tmp_path / "ws")
    lines, row = _gate(ws, capsys)
    assert row == dict.fromkeys(_KEYS) | {"state": "outside_work_tree"}
    assert _glyph(lines) == "—"
    assert "not inside a git work tree" in lines[0]


@_needs_git
def test_no_git_on_path_is_git_unavailable_and_exits_as_before(
        tmp_path, capsys, monkeypatch) -> None:
    """Criterion 3: the exit code is the one status gives with git present."""
    _repo, ws = _repo_with_workspace(tmp_path)
    rc_with_git, _text, _payload = _status(ws, capsys)
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    rc, text, payload = _status(ws, capsys)
    assert rc == rc_with_git
    assert payload["commit_gate"] == dict.fromkeys(_KEYS) | {"state": "git_unavailable"}
    lines = _row_lines(text)
    assert _glyph(lines) == "—"
    assert "git was not found" in lines[0]


# --------------------------------------------------------------------------- #
# 4: keyless reads the same over any hook, and reads nothing
# --------------------------------------------------------------------------- #

@_needs_git
def test_keyless_reads_identically_with_and_without_a_hook(
        tmp_path, capsys, monkeypatch) -> None:
    """Criterion 4: no hook → installed → foreign, one repository, one row."""
    repo, ws = _repo_with_workspace(tmp_path)
    hook_file = os.path.join(repo, ".git", "hooks", "pre-commit")

    def keyless_status() -> Tuple[List[str], Dict[str, Any]]:
        with pytest.MonkeyPatch.context() as mp:
            for name in ("read_last_attempt", "derive_audit_debt", "classify_hook_file"):
                mp.setattr(cli, name, _Raises(name))
            return _gate(ws, capsys)

    before = keyless_status()
    code, _out, err = _install(ws, _stub(tmp_path / "bin"), capsys)
    assert code == 0, err
    installed = keyless_status()
    _write(hook_file, "#!/bin/sh\necho someone else's hook\n")
    foreign = keyless_status()

    assert before == installed == foreign
    lines, row = before
    assert row == dict.fromkeys(_KEYS) | {"state": "keyless"}
    assert _glyph(lines) == "—"
    assert "no judge key" in lines[0] and "hook" not in " ".join(lines)
    assert not os.path.exists(os.path.join(ws, ".mitos", "telemetry.sqlite"))


# --------------------------------------------------------------------------- #
# 5–8: the hook file's kinds
# --------------------------------------------------------------------------- #

@_needs_git
def test_keyed_with_no_pre_commit_names_the_install_recipe(
        tmp_path, capsys, monkeypatch) -> None:
    """Criterion 5: ⚠, the recipe parses as hook-install and resolves to this workspace."""
    _keyed(monkeypatch)
    repo, ws = _repo_with_workspace(tmp_path, "sub")
    lines, row = _gate(ws, capsys)
    hooks_dir = os.path.join(repo, ".git", "hooks")
    assert row == dict.fromkeys(_KEYS) | {
        "state": "no_hook", "hooks_dir": hooks_dir, "hooks_dir_shared": False,
        "hook_file": os.path.join(hooks_dir, "pre-commit")}
    assert _glyph(lines) == "⚠"
    assert f"no pre-commit in {hooks_dir}" in lines[0]
    assert "shared" not in " ".join(lines)
    _assert_recipe_resolves(_recipe(lines[0]), "hook-install", ws)


@_needs_git
def test_a_shared_hooks_directory_is_read_and_says_a_block_is_printed(
        tmp_path, capsys, monkeypatch) -> None:
    """Criterion 6: an in-tree ``core.hooksPath`` → no_hook (shared), then the pasted block."""
    _keyed(monkeypatch)
    repo = make_scratch_repo(str(tmp_path / "repo"), hooks_path="hooks")
    ws = _workspace(repo)
    shared = os.path.join(repo, "hooks")
    lines, row = _gate(ws, capsys)
    assert row["state"] == "no_hook"
    assert row["hooks_dir"] == shared and row["hooks_dir_shared"] is True
    assert _glyph(lines) == "⚠"
    assert "shared" in lines[1] and "block to paste" in lines[1]
    _assert_recipe_resolves(_recipe(lines[0]), "hook-install", ws)

    block = render_hook_block(command=None, selector=".", guard_dir="./.mitos")
    _write(os.path.join(shared, "pre-commit"), "#!/bin/sh\nset -eu\n" + block)
    lines, row = _gate(ws, capsys)
    assert row["state"] == "foreign_with_block" and row["hooks_dir_shared"] is True
    assert _glyph(lines) == "•"


@_needs_git
def test_one_file_with_and_without_the_block_marker(tmp_path, capsys, monkeypatch) -> None:
    """Criterion 7: foreign ⚠ "does not carry" vs foreign_with_block • "carries"."""
    _keyed(monkeypatch)
    repo, ws = _repo_with_workspace(tmp_path)
    hook_file = os.path.join(repo, ".git", "hooks", "pre-commit")
    _write(hook_file, "#!/bin/sh\necho lint\n")
    foreign, row = _gate(ws, capsys)
    assert row["state"] == "foreign" and row["hook_file"] == hook_file
    assert _glyph(foreign) == "⚠"
    assert "does not carry" in foreign[0]
    assert "hook manager" in " ".join(foreign)
    _assert_recipe_resolves(_recipe(foreign[0]), "hook-install", ws)

    _write(hook_file, f"#!/bin/sh\necho lint\n{HOOK_BLOCK_MARKER}\n")
    with_block, row = _gate(ws, capsys)
    assert row["state"] == "foreign_with_block"
    assert _glyph(with_block) == "•"
    assert "carries" in with_block[0] and "→" not in with_block[0]
    for lines in (foreign, with_block):
        assert "no pre-commit" not in " ".join(lines)


@_needs_git
@pytest.mark.parametrize("shape", ["symlink", "dangling"])
def test_a_symlinked_pre_commit_is_foreign(tmp_path, capsys, monkeypatch, shape) -> None:
    """Criterion 8: a symlink, live or dangling, is never ours and never absent."""
    _keyed(monkeypatch)
    repo, ws = _repo_with_workspace(tmp_path)
    hook_file = os.path.join(repo, ".git", "hooks", "pre-commit")
    target = str(tmp_path / "elsewhere")
    if shape == "symlink":
        _write(target, f"#!/bin/sh\n{HOOK_FILE_MARKER}\n")
    os.symlink(target, hook_file)
    lines, row = _gate(ws, capsys)
    assert row["state"] == "foreign"
    assert _glyph(lines) == "⚠"


@_needs_git
def test_an_unreadable_pre_commit_is_never_no_hook(tmp_path, capsys, monkeypatch) -> None:
    """Criterion 8: ``chmod 000`` → unreadable, ⚠, "could not be read"."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root reads a mode-000 file")
    _keyed(monkeypatch)
    repo, ws = _repo_with_workspace(tmp_path)
    hook_file = os.path.join(repo, ".git", "hooks", "pre-commit")
    _write(hook_file, "#!/bin/sh\n")
    os.chmod(hook_file, 0)
    try:
        lines, row = _gate(ws, capsys)
    finally:
        os.chmod(hook_file, 0o644)
    assert row["state"] == "unreadable" and row["hook_file"] == hook_file
    assert _glyph(lines) == "⚠"
    assert "could not be read" in lines[0] and "no pre-commit" not in lines[0]


@_needs_git
def test_a_hooks_path_that_is_a_file_is_unreadable(tmp_path, capsys, monkeypatch) -> None:
    """Criterion 8 / Gotcha 8: ``core.hooksPath`` naming a regular file."""
    _keyed(monkeypatch)
    repo = make_scratch_repo(str(tmp_path / "repo"), hooks_path="hookfile")
    _write(os.path.join(repo, "hookfile"), "not a directory\n")
    ws = _workspace(repo)
    lines, row = _gate(ws, capsys)
    assert row["state"] == "unreadable"
    assert _glyph(lines) == "⚠"


# --------------------------------------------------------------------------- #
# 9–11: an installed hook and the workspace it serves
# --------------------------------------------------------------------------- #

@_needs_git
def test_installed_serving_this_workspace_through_a_symlinked_route(
        tmp_path, capsys, monkeypatch) -> None:
    """Criterion 9: realpath on both sides, so a symlinked route is still this workspace."""
    _keyed(monkeypatch)
    repo, ws, hook_file = _installed(tmp_path, capsys, "sub")
    link = str(tmp_path / "link")
    os.symlink(repo, link)
    via_link = os.path.join(link, "sub")
    assert via_link != ws and os.path.realpath(via_link) == ws
    lines, row = _gate(via_link, capsys)
    assert row["state"] == "installed" and row["serves_this_workspace"] is True
    assert row["hook_file"] == hook_file and row["serves"] == ws
    assert _glyph(lines) == "✓"
    assert "serves: this workspace" in lines[1]
    assert lines[-1].strip() == f"to remove the gate, delete {hook_file}"


@_needs_git
def test_installed_serving_another_workspace_warns_with_no_recipe(
        tmp_path, capsys, monkeypatch) -> None:
    """Criterion 10: installed from B, status on A."""
    _keyed(monkeypatch)
    repo = make_scratch_repo(str(tmp_path / "repo"))
    ws_a = _workspace(os.path.join(repo, "a"))
    ws_b = _workspace(os.path.join(repo, "b"))
    code, _out, err = _install(ws_b, _stub(tmp_path / "bin"), capsys)
    assert code == 0, err
    lines, row = _gate(ws_a, capsys)
    assert row["state"] == "installed"
    assert row["serves_this_workspace"] is False
    assert os.path.realpath(row["serves"]) == ws_b
    assert _glyph(lines) == "⚠"
    assert f"another workspace, {row['serves']}" in lines[1]
    assert not any("→" in line or "hook-install" in line for line in lines)


@_needs_git
def test_installed_with_an_unreadable_served_workspace(tmp_path, capsys, monkeypatch) -> None:
    """Criterion 11 / Gotcha 6: line 2 kept, the hook-run line removed."""
    _keyed(monkeypatch)
    _repo, ws, hook_file = _installed(tmp_path, capsys)
    with open(hook_file, encoding="utf-8") as f:
        text = f.read()
    kept = [line for line in text.splitlines(keepends=True) if "hook-run -p" not in line]
    assert len(kept) < len(text.splitlines())
    _write(hook_file, "".join(kept))
    lines, row = _gate(ws, capsys)
    assert row["state"] == "installed"
    assert row["serves"] is None and row["serves_this_workspace"] is False
    assert _glyph(lines) == "⚠"
    assert "could not be read" in lines[1]
    assert "None" not in " ".join(lines)


# --------------------------------------------------------------------------- #
# 12: the attempt line
# --------------------------------------------------------------------------- #

def _telemetry(ws: str) -> TelemetryStore:
    return TelemetryStore(MitosConfig(ws).telemetry_path)


def _start(tel: TelemetryStore) -> None:
    tel.record_attempt_start(AttemptStart(attempt_id="att-1", started_at=_STARTED_AT,
                                          fingerprint="f" * 64))


def _end(tel: TelemetryStore, **overrides: Any) -> None:
    _start(tel)
    base = dict(attempt_id="att-1", state="no_new_findings", run_id="run-1",
                outcome_at="2026-09-21T00:01:00.000000+00:00", degradation_tokens=(),
                new_pairs=(), findings_known=0)
    base.update(overrides)
    tel.record_run_end(_check_run_row("run-1"), coverage=None,
                       attempt=AttemptOutcome(**base))


def _raw(ws: str, sql: str) -> None:
    conn = sqlite3.connect(MitosConfig(ws).telemetry_path)
    with conn:
        conn.execute(sql)
    conn.close()


def _seed_none_file(ws: str) -> None:
    path = MitosConfig(ws).telemetry_path
    if os.path.exists(path):
        os.remove(path)


def _seed_refusal(ws: str) -> None:
    tel = _telemetry(ws)
    _start(tel)
    tel.record_attempt_refusal(AttemptRefusal(
        attempt_id="att-1", refused_at="2026-09-21T00:02:00.000000+00:00",
        batches_planned=5))


def _seed_unknown(ws: str) -> None:
    _start(_telemetry(ws))
    _raw(ws, "UPDATE check_attempt SET state = 'from_a_newer_build'")


def _seed_damaged(ws: str) -> None:
    _end(_telemetry(ws), state=ATTEMPT_NEW_FINDINGS, new_pairs=(("a", "b"),))
    _raw(ws, "UPDATE check_attempt SET new_pairs = 'not json'")


_PAIRS = (("a", "b"), ("c", "d"))

# (id, seeder, last_attempt_status, expected attempt words, words that must not appear)
_ATTEMPT_CASES = [
    ("no-telemetry", _seed_none_file, "none", "no attempt on record", None),
    ("empty-table", lambda ws: _telemetry(ws), "none", "no attempt on record", None),
    ("started", lambda ws: _start(_telemetry(ws)), "on_record",
     "started, no outcome recorded", None),
    ("no-new-findings", lambda ws: _end(_telemetry(ws), findings_known=4), "on_record",
     "no new findings (4 known finding(s))", None),
    ("no-new-findings-unknown", lambda ws: _end(_telemetry(ws), findings_known=None),
     "on_record", "no new findings (known findings: unknown)", "0 known"),
    ("new-findings", lambda ws: _end(_telemetry(ws), state=ATTEMPT_NEW_FINDINGS,
                                     new_pairs=_PAIRS, findings_known=2),
     "on_record", "new findings (2 new pair(s))", None),
    ("could-not-complete-pairs",
     lambda ws: _end(_telemetry(ws), state=ATTEMPT_COULD_NOT_COMPLETE,
                     new_pairs=_PAIRS[:1], degradation_tokens=("x",)),
     "on_record", "could not complete (1 new pair(s))", None),
    ("could-not-complete-none",
     lambda ws: _end(_telemetry(ws), state=ATTEMPT_COULD_NOT_COMPLETE,
                     degradation_tokens=("x",)),
     "on_record", "could not complete", "new pair"),
    ("spend-not-authorized", _seed_refusal, "on_record",
     "spend not authorized (5 planned batch(es); a person authorizes that spend)", None),
    ("spend-not-authorized-null-batches",
     lambda ws: (_seed_refusal(ws),
                 _raw(ws, "UPDATE check_attempt SET batches_planned = NULL")),
     "on_record", "spend not authorized (planned batches: unknown;", "None"),
    ("unknown-state", _seed_unknown, "on_record", "from_a_newer_build", None),
    ("damaged", _seed_damaged, "unreadable", "the last attempt could not be read",
     "no attempt on record"),
]


@_needs_git
@pytest.mark.parametrize("case", _ATTEMPT_CASES, ids=[c[0] for c in _ATTEMPT_CASES])
def test_the_attempt_line(tmp_path, capsys, monkeypatch, case) -> None:
    """Criterion 12: one row per outcome, text and payload from one fixture."""
    _id, seed, status, words, absent = case
    _keyed(monkeypatch)
    _repo, ws, _hook = _installed(tmp_path, capsys)
    seed(ws)
    lines, row = _gate(ws, capsys)
    assert row["last_attempt_status"] == status
    (attempt_line,) = [line for line in lines if "last check attempt:" in line]
    assert words in attempt_line
    if absent:
        assert absent not in attempt_line
    if status == "on_record":
        attempt = row["last_attempt"]
        assert set(attempt) == {"started_at", "state", "outcome_at", "findings_known",
                                "new_pairs", "batches_planned"}
        assert attempt["started_at"] == _STARTED_AT
        assert _STARTED_AT in attempt_line
    else:
        assert row["last_attempt"] is None


@_needs_git
def test_the_attempt_object_carries_counts_not_ids(tmp_path, capsys, monkeypatch) -> None:
    """§3.2: ``new_pairs`` is the count; every value is a JSON scalar."""
    _keyed(monkeypatch)
    _repo, ws, _hook = _installed(tmp_path, capsys)
    _end(_telemetry(ws), state=ATTEMPT_NEW_FINDINGS, new_pairs=_PAIRS, findings_known=2)
    _lines, row = _gate(ws, capsys)
    assert row["last_attempt"] == {
        "started_at": _STARTED_AT, "state": ATTEMPT_NEW_FINDINGS,
        "outcome_at": "2026-09-21T00:01:00.000000+00:00", "findings_known": 2,
        "new_pairs": 2, "batches_planned": None}


# --------------------------------------------------------------------------- #
# 13: the counts line
# --------------------------------------------------------------------------- #

@_needs_git
def test_the_counts_line_matches_the_debt_and_shows_exclusions(
        tmp_path, capsys, monkeypatch) -> None:
    """Criterion 13: the leaf's own numbers, an exclusion included."""
    _keyed(monkeypatch)
    _repo, ws, _hook = _installed(tmp_path, capsys)
    config = MitosConfig(ws)
    store = GraphStore(config.db_path)
    a = _commit(store, "gate-row-a", "Alpha holds.")
    b = _commit(store, "gate-row-b", "Beta holds.")
    _commit(store, "gate-row-c", "Gamma holds.")
    TelemetryStore(config.telemetry_path).record_run_end(
        _check_run_row("run-1"),
        coverage=CoverageMarks(run_id="run-1", marked_at="2026-09-21T00:01:00+00:00",
                               covered=(a,), excluded=(b,)),
        attempt=None)
    debt = derive_audit_debt(config.db_path, config.telemetry_path)
    assert isinstance(debt, AuditDebt) and debt.excluded == 1 and debt.uncovered == 1
    lines, row = _gate(ws, capsys)
    assert (row["debt_status"], row["uncovered"], row["excluded"]) == (
        "read", debt.uncovered, debt.excluded)
    assert f"uncovered decisions: {debt.uncovered}, excluded: {debt.excluded}" in (
        " ".join(lines))


@_needs_git
def test_the_counts_line_with_no_graph_and_with_damaged_telemetry(
        tmp_path, capsys, monkeypatch) -> None:
    """Criterion 13: no graph yet; a damaged ``telemetry.sqlite`` is unreadable, not 0."""
    _keyed(monkeypatch)
    _repo, ws, _hook = _installed(tmp_path, capsys)
    config = MitosConfig(ws)
    TelemetryStore(config.telemetry_path)
    for path in (config.telemetry_path + "-wal", config.telemetry_path + "-shm"):
        if os.path.exists(path):
            os.remove(path)
    with open(config.telemetry_path, "wb") as f:
        f.write(b"this is not a sqlite database" * 64)
    lines, row = _gate(ws, capsys)
    assert row["debt_status"] == "unreadable"
    assert row["uncovered"] is None and row["excluded"] is None
    assert row["last_attempt_status"] == "unreadable"
    assert "the count could not be read" in " ".join(lines)

    os.remove(config.telemetry_path)
    if os.path.exists(config.db_path):
        os.remove(config.db_path)
    lines, row = _gate(ws, capsys)
    assert row["debt_status"] == "no_graph"
    assert "no graph yet" in " ".join(lines)


# --------------------------------------------------------------------------- #
# 14–15: READY and the key set, over every state
# --------------------------------------------------------------------------- #

def _scenario(name: str, tmp_path, capsys, monkeypatch) -> str:
    """Builds the named state on a READY-capable workspace; returns the path status reads."""
    if name == "outside_work_tree":
        return _workspace(tmp_path / "ws")
    if name == "git_unavailable":
        _repo, ws = _repo_with_workspace(tmp_path)
        empty = tmp_path / "empty-bin"
        empty.mkdir()
        monkeypatch.setenv("PATH", str(empty))
        return ws
    if name == "keyless":
        _repo, ws, _hook = _installed(tmp_path, capsys)
        return ws
    _keyed(monkeypatch)
    if name in ("installed_this", "installed_other"):
        repo = make_scratch_repo(str(tmp_path / "repo"))
        ws_a = _workspace(os.path.join(repo, "a"))
        ws_b = _workspace(os.path.join(repo, "b"))
        code, _out, err = _install(ws_b if name == "installed_other" else ws_a,
                                   _stub(tmp_path / "bin"), capsys)
        assert code == 0, err
        return ws_a
    repo, ws = _repo_with_workspace(tmp_path)
    hook_file = os.path.join(repo, ".git", "hooks", "pre-commit")
    if name == "unreadable":
        scratch = os.path.join(repo, ".git", "hooks")
        for entry in os.listdir(scratch):
            os.remove(os.path.join(scratch, entry))
        os.rmdir(scratch)
        _write(scratch, "a file where the hooks directory belongs\n")
    elif name == "foreign":
        _write(hook_file, "#!/bin/sh\necho lint\n")
    elif name == "foreign_with_block":
        _write(hook_file, f"#!/bin/sh\n{HOOK_BLOCK_MARKER}\n")
    return ws


_SCENARIOS = {
    "outside_work_tree": "outside_work_tree",
    "git_unavailable": "git_unavailable",
    "keyless": "keyless",
    "no_hook": "no_hook",
    "unreadable": "unreadable",
    "foreign": "foreign",
    "foreign_with_block": "foreign_with_block",
    "installed_this": "installed",
    "installed_other": "installed",
}


@_needs_git
@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_ready_and_the_key_set_hold_in_every_state(tmp_path, capsys, monkeypatch,
                                                   name) -> None:
    """Criteria 14 and 15: READY ✓, ``ready: true``, exit 0, and the same keys."""
    _ready_env(monkeypatch)
    ws = _scenario(name, tmp_path, capsys, monkeypatch)
    rc, text, payload = _status(ws, capsys)
    assert payload["commit_gate"]["state"] == _SCENARIOS[name]
    assert set(payload["commit_gate"]) == _KEYS
    assert payload["ready"] is True and rc == 0
    assert "READY ✓" in text.splitlines()[1]
    assert "✗" not in " ".join(_row_lines(text))


def test_the_state_tuple_and_its_renderer_agree() -> None:
    """Criterion 18 (D3): the payload's closed tokens, and words for exactly those."""
    assert cli._COMMIT_GATE_STATES == (
        "git_unavailable", "outside_work_tree", "keyless", "no_hook", "unreadable",
        "foreign", "foreign_with_block", "installed")
    assert set(cli._COMMIT_GATE_WORDS) == set(cli._COMMIT_GATE_STATES)
    assert set(cli._COMMIT_GATE_KEYS) == _KEYS
    assert set(cli._COMMIT_GATE_STATE_OF_KIND.values()) <= set(cli._COMMIT_GATE_STATES)


# --------------------------------------------------------------------------- #
# 16–17: git is asked from the workspace; nothing is created
# --------------------------------------------------------------------------- #

@_needs_git
def test_status_asks_git_from_the_workspace_not_the_cwd(tmp_path, capsys,
                                                        monkeypatch) -> None:
    """Criterion 16: a cwd inside another repository with its own hook changes nothing."""
    _keyed(monkeypatch)
    repo, ws = _repo_with_workspace(tmp_path)
    other = make_scratch_repo(str(tmp_path / "other"))
    _write(os.path.join(other, ".git", "hooks", "pre-commit"), "#!/bin/sh\necho other\n")
    monkeypatch.chdir(other)
    _lines, row = _gate(ws, capsys)
    assert row["state"] == "no_hook"
    assert row["hooks_dir"] == os.path.join(repo, ".git", "hooks")


@_needs_git
def test_a_keyed_status_leaves_no_telemetry_behind(tmp_path, capsys, monkeypatch) -> None:
    """Criterion 17: state 7 reads telemetry read-only and creates nothing."""
    _keyed(monkeypatch)
    _repo, ws, _hook = _installed(tmp_path, capsys)
    telemetry = MitosConfig(ws).telemetry_path
    _seed_none_file(ws)
    _lines, row = _gate(ws, capsys)
    assert row["state"] == "installed" and row["last_attempt_status"] == "none"
    assert not os.path.exists(telemetry)


# --------------------------------------------------------------------------- #
# 19–22: init's line
# --------------------------------------------------------------------------- #

def _init_out(directory, capsys: Any) -> str:
    os.makedirs(str(directory), exist_ok=True)
    capsys.readouterr()
    cli.cmd_init(MitosConfig(str(directory)))
    return capsys.readouterr().out


def _gate_lines(out: str) -> List[str]:
    return [line for line in out.splitlines() if "hook-install" in line]


@_needs_git
@pytest.mark.parametrize("sub", ["", "nested/ws"])
def test_init_in_a_work_tree_names_hook_install_once(tmp_path, capsys, sub) -> None:
    """Criterion 19: after the collection line; the recipe resolves to this workspace."""
    repo = make_scratch_repo(str(tmp_path / "repo"))
    target = os.path.join(repo, sub) if sub else repo
    out = _init_out(target, capsys)
    (line,) = _gate_lines(out)
    lines = out.splitlines()
    collection = [i for i, text in enumerate(lines) if text.lstrip().startswith("collection:")]
    assert collection and lines.index(line) > collection[-1]
    recipe = line.split("`")[1]
    _assert_recipe_resolves(recipe, "hook-install", os.path.realpath(target))


def _setup_init_sample() -> List[str]:
    """Returns the lines of the untagged fence under SETUP.md's init step."""
    setup_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "SETUP.md")
    with open(setup_path, encoding="utf-8") as fh:
        text = fh.read()
    step = text.split("\n### 1. Initialize the workspace\n", 1)[1].split("\n### ", 1)[0]
    # Fences are paired in order: a closing fence also reads as an untagged opener.
    samples, body, tag = [], None, None
    for ln in step.splitlines():
        if ln.startswith("```"):
            if body is None:
                body, tag = [], ln[3:].strip()
            else:
                if not tag:
                    samples.append(body)
                body = None
        elif body is not None:
            body.append(ln)
    assert len(samples) == 1, "the init step should hold exactly one untagged sample fence"
    return samples[0]


@_needs_git
def test_setup_init_sample_ends_with_the_line_init_prints(tmp_path, capsys) -> None:
    """3i R9: SETUP.md's sample ends with the exact hook line a real `init` prints.

    The sample's workspace is `harbor`, so the scratch repository is named that too;
    `init` names the project after the basename.
    """
    repo = make_scratch_repo(str(tmp_path / "x" / "harbor"))
    (line,) = _gate_lines(_init_out(repo, capsys))
    assert _setup_init_sample()[-1] == line


@_needs_git
def test_init_outside_a_work_tree_prints_no_line(tmp_path, capsys) -> None:
    """Criterion 20: outside a work tree."""
    out = _init_out(tmp_path / "ws", capsys)
    assert _gate_lines(out) == []
    assert "commit gate" not in out


@_needs_git
def test_init_without_git_prints_no_line_and_completes(tmp_path, capsys,
                                                       monkeypatch) -> None:
    """Criterion 20: git absent — the same output as outside a work tree, bar the paths."""
    repo = make_scratch_repo(str(tmp_path / "repo"))
    outside = _init_out(tmp_path / "ws", capsys)
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    gitless = _init_out(repo, capsys)
    assert _gate_lines(gitless) == []
    assert os.path.isfile(os.path.join(repo, ".mitos", "config.toml"))
    assert len(gitless.splitlines()) == len(outside.splitlines())


@_needs_git
def test_init_asks_git_about_the_directory_not_the_cwd(tmp_path, capsys,
                                                       monkeypatch) -> None:
    """Criterion 21, both directions."""
    repo = make_scratch_repo(str(tmp_path / "repo"))
    monkeypatch.chdir(repo)
    assert _gate_lines(_init_out(tmp_path / "outside", capsys)) == []
    monkeypatch.chdir(tmp_path)
    assert len(_gate_lines(_init_out(os.path.join(repo, "ws"), capsys))) == 1


@_needs_git
def test_init_installs_nothing(tmp_path, capsys) -> None:
    """Criterion 22: the line names the verb; the hooks directory holds only samples."""
    repo = make_scratch_repo(str(tmp_path / "repo"))
    _init_out(repo, capsys)
    hooks = os.path.join(repo, ".git", "hooks")
    assert not os.path.lexists(os.path.join(hooks, "pre-commit"))
    assert all(name.endswith(".sample") for name in os.listdir(hooks))
