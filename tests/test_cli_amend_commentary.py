"""CLI rows for `mitos amend-commentary` (Phase 4b, W18, T13's CLI half).

The verb is a surface over `MitosSyncManager.amend_commentary` (4a): it maps flags to
`changes`, renders every result class with this boundary's recovery clause, and carries
the class in the exit code — 0 applied, 1 not applied, 2 a value the fence refused.

Every workspace is a real, registered `cmd_init` one, driven through `main()` with a
selector, seeded through `record` (a keyless `sync` commits nothing), with the embed
provider and vector store down. The subprocess frame (C7) isolates its registry, keys and
Qdrant the way `test_cluster_b_checkpoint.py` does.
"""

import copy
import hashlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest
from unittest.mock import MagicMock, patch

from mitos import amend, cli
from mitos.cli import _build_parser, _render_amend_result, cmd_amend_commentary
from mitos.config import MitosConfig
from mitos.divergence import RELATIONSHIP_FIELDS, entry_divergence
from mitos.parser import parse_entry_stream
from mitos.recall import corpus_provenance, provenance_line
from mitos.restore import BufferFidelityError
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager, _ENTRIES_MARKER

from test_cli_selector import _ALIASES, _run, _subparsers
from test_corpus_provenance import _init_workspace, _write_registry

DEAD_QDRANT_URL = "http://127.0.0.1:9"
NAME = "amend-ws"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Keyless and serviceless: the embed step defers, nothing reaches a network."""
    monkeypatch.setenv("QDRANT_URL", DEAD_QDRANT_URL)
    down = MagicMock(side_effect=Exception("backend down"))
    monkeypatch.setattr("mitos.sync.GeminiEmbeddingProvider", down)
    monkeypatch.setattr("mitos.sync.QdrantVectorStore", down)


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A registered workspace addressed as `NAME`, with the cwd outside it."""
    root = _init_workspace(tmp_path / "ws")
    _write_registry(**{NAME: root})
    monkeypatch.chdir(tmp_path)
    config = MitosConfig(root, project=NAME)
    return config, MitosSyncManager(config)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _record(m: MitosSyncManager, slug: str, **relations: Any) -> Dict:
    result = m.record_decision_entry(
        f"The {slug} axiom.", f"The {slug} rejected reasoning.", ["alpha"],
        mechanisms=[f"{slug}-mechanism"], context=f"The {slug} context.", slug=slug,
        acknowledge_neighbors=True, **relations,
    )
    assert result["status"] == "created", result
    return result


def _amend(capsys, *argv: str):
    """Runs `mitos -p NAME amend-commentary …` through `main()`: (exit, stdout, stderr)."""
    capsys.readouterr()
    code = _run(["-p", NAME, "amend-commentary", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _buffer(config: MitosConfig) -> str:
    with open(config.decisions_file, encoding="utf-8") as fh:
        return fh.read()


def _write_buffer(config: MitosConfig, text: str) -> None:
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write(text)


def _sha(config: MitosConfig) -> str:
    with open(config.decisions_file, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _entry(config: MitosConfig, slug: str):
    return next(e for e in parse_entry_stream(_buffer(config), "decision") if e.slug == slug)


def _answer_lines(text: str) -> List[str]:
    """The lines a channel carries, less the core's `[Warning]` lines."""
    return [line for line in text.splitlines() if not line.startswith("[Warning]")]


def _leads_with_echo(text: str, config: MitosConfig) -> bool:
    lines = _answer_lines(text)
    return bool(lines) and lines[0] == provenance_line(config)


def _clause_lines(text: str) -> List[str]:
    """The answer's own lines: no echo (its collection starts `mitos-`), no warnings."""
    return [line for line in _answer_lines(text)
            if line.strip() and not line.startswith("corpus: ")]


def _archive_one(config: MitosConfig, m: MitosSyncManager) -> None:
    """Records `old-one`, settles it, then lets a real record-path rotation archive it."""
    _record(m, "old-one")
    stamp = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    with GraphStore(config.db_path)._get_connection() as conn:
        conn.execute("UPDATE nodes SET updated_at = ?", (stamp,))
    config.rotation_volume_threshold_entries = 1
    config.rotation_lag_days = 1
    fresh = _record(m, "fresh-one")
    assert fresh.get("rotation", {}).get("outcome") == "rotated", fresh
    assert "### old-one" not in _buffer(config)


def _record_flags() -> set:
    return {opt for action in _subparsers(_build_parser())["record"]._actions
            for opt in action.option_strings}


# --------------------------------------------------------------------------- #
# C1 — the flag map
# --------------------------------------------------------------------------- #

_CHANGE_KEYS = set(amend.EDITABLE_FIELDS) | {"axiom", "mechanisms"} | set(RELATIONSHIP_FIELDS)
_SPY_RESULT = {"status": amend.STATUS_UNCHANGED, "id": "an-id", "slug": "h"}


def _spy():
    return patch.object(MitosSyncManager, "amend_commentary", autospec=True,
                        side_effect=lambda self, slug, changes: dict(_SPY_RESULT))


@pytest.mark.parametrize("flags, expected", [
    (["--rejected", "R."], {"rejected_paths": "R."}),
    (["--invalidates-if", "I."], {"invalidates_if": "I."}),
    (["--clear-invalidates-if"], {"invalidates_if": None}),
    (["--context", "C."], {"context": "C."}),
    (["--clear-context"], {"context": None}),
    (["--scope", "a", "b", "--scope", "c"], {"scope": ["a", "b", "c"]}),
    (["--clear-scope"], {"scope": []}),
    (["--new-slug", "s"], {"slug": "s"}),
    (["--axiom", "A."], {"axiom": "A."}),
    (["--mechanisms", "m", "n"], {"mechanisms": ["m", "n"]}),
    (["--mechanisms"], {"mechanisms": []}),
    (["--cites", "a", "--cites", "b"], {"cites": "a, b"}),
    (["--depends-on", "a"], {"depends_on": "a"}),
    ([], {}),
    (["--context", "C.", "--scope", "a", "--clear-invalidates-if"],
     {"context": "C.", "scope": ["a"], "invalidates_if": None}),
])
def test_each_flag_maps_to_exactly_its_change(ws, capsys, flags, expected) -> None:
    """C1 — an absent flag contributes no key; a clear flag maps to the removing value."""
    with _spy() as spy:
        code, _out, _err = _amend(capsys, "h", *flags)
    assert code == 0
    _self, handle, changes = spy.call_args.args
    assert (handle, changes) == ("h", expected)
    assert set(changes) <= _CHANGE_KEYS


@pytest.mark.parametrize("stem, key", [
    ("rejected", "rejected_paths"), ("invalidates-if", "invalidates_if"), ("context", "context"),
])
def test_a_file_or_stdin_value_equals_the_inline_one(ws, capsys, tmp_path, monkeypatch,
                                                     stem, key) -> None:
    """C1 — the single trailing newline a file or heredoc adds is stripped, as --axiom-file."""
    path = tmp_path / f"{stem}.txt"
    path.write_text("Line one.\nLine two.\n", encoding="utf-8")
    with _spy() as spy:
        _amend(capsys, "h", f"--{stem}", "Line one.\nLine two.")
        _amend(capsys, "h", f"--{stem}-file", str(path))
        monkeypatch.setattr(sys, "stdin", io.StringIO("Line one.\nLine two.\n"))
        _amend(capsys, "h", f"--{stem}-file", "-")
    inline, from_file, from_stdin = (call.args[2] for call in spy.call_args_list)
    assert inline == from_file == from_stdin == {key: "Line one.\nLine two."}


@pytest.mark.parametrize("flags", [
    ["--context", "x", "--clear-context"],
    ["--context", "x", "--context-file", "f"],
    ["--invalidates-if", "x", "--clear-invalidates-if"],
    ["--rejected", "x", "--rejected-file", "f"],
    ["--scope", "a", "--clear-scope"],
    ["--scope"],
])
def test_two_values_for_one_field_or_a_bare_scope_is_argparses_refusal(ws, capsys, flags) -> None:
    """C1 — exit 2 before any mitos code; a bare `--scope` is never a clear."""
    with _spy() as spy:
        code, _out, _err = _amend(capsys, "h", *flags)
    assert code == 2
    spy.assert_not_called()


@pytest.mark.parametrize("flags, names", [
    (["--context", ""], "--clear-context"),
    (["--invalidates-if", "   "], "--clear-invalidates-if"),
    (["--rejected", " "], "required"),
    (["--scope", "a", ""], "--clear-scope"),
])
def test_an_empty_setter_is_a_usage_refusal_naming_the_clear(ws, capsys, flags, names) -> None:
    """C1 — `empty_value`: exit 2, stderr, no echo, nothing dispatched."""
    with _spy() as spy:
        code, out, err = _amend(capsys, "h", *flags)
    assert code == 2 and out == ""
    assert names in err and "corpus: " not in err
    spy.assert_not_called()


def test_an_empty_file_value_is_the_same_refusal(ws, capsys, tmp_path) -> None:
    path = tmp_path / "blank.txt"
    path.write_text("\n", encoding="utf-8")
    with _spy() as spy:
        code, _out, err = _amend(capsys, "h", "--context-file", str(path))
    assert code == 2 and "--clear-context" in err
    spy.assert_not_called()


def test_an_unreadable_file_is_a_usage_refusal(ws, capsys, tmp_path) -> None:
    """C1 — a missing or undecodable `-file` exits 2 naming its flag, not main()'s crash line."""
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"\xff\xfe\xfa")
    for path in (str(tmp_path / "missing.txt"), str(binary)):
        with _spy() as spy:
            code, out, err = _amend(capsys, "h", "--context-file", path)
        assert code == 2 and out == ""
        assert "--context-file could not be read" in err and "Fatal" not in err
        spy.assert_not_called()


@pytest.mark.parametrize("flags, code_name", [
    (["--context", ""], cli.AMEND_CODE_EMPTY_VALUE),
    (["--context-file", "-", "--rejected-file", "-"], cli.AMEND_CODE_MULTIPLE_STDIN),
    (["--invalidates-if-file", "/no/such/amend/file"], cli.AMEND_CODE_UNREADABLE_FILE),
])
def test_a_usage_refusal_under_json_is_one_object(ws, capsys, flags, code_name) -> None:
    """C1 — `{"error", "code"}` on stdout, exit 2, no provenance (no handler answered)."""
    with _spy() as spy:
        code, out, err = _amend(capsys, "h", *flags, "--json")
    assert code == 2
    payload = json.loads(out)
    assert set(payload) == {"error", "code"} and payload["code"] == code_name
    assert err == ""
    spy.assert_not_called()


# --------------------------------------------------------------------------- #
# C2 — every result class renders, on the right channel, with the right exit
# --------------------------------------------------------------------------- #

def _constants(prefix: str) -> set:
    return {value for name, value in vars(amend).items() if name.startswith(prefix)}


_STATUSES = sorted(_constants("STATUS_"))
_REASONS = sorted(_constants("REASON_"))
_CODES = sorted(amend.ERROR_FACTS)


def _canned(status: str = None, reason: str = None, code: str = None) -> Dict[str, Any]:
    if code is not None:
        return amend.error_result(code, slug="t", reason="a cause", requested="u")
    result: Dict[str, Any] = {"status": status, "slug": "t"}
    if status == amend.STATUS_AMENDED:
        result.update(id="i", fields_changed=["context"], embedding="pending", path="/p")
    elif status in (amend.STATUS_UNCHANGED, amend.STATUS_ARCHIVED):
        result["id"] = "i"
    elif status == amend.STATUS_REFUSED:
        result.update(reason=reason, fields=[])
        if reason == amend.REASON_CANONICAL_CORE:
            result["routes"] = dict(amend.ROUTES)
    return result


_CLASSES = (
    [(f"status:{s}", _canned(status=s), 0 if s in (amend.STATUS_AMENDED,
                                                   amend.STATUS_UNCHANGED) else 1)
     for s in _STATUSES if s != amend.STATUS_REFUSED]
    + [(f"reason:{r}", _canned(status=amend.STATUS_REFUSED, reason=r), 1) for r in _REASONS]
    + [(f"code:{c}", _canned(code=c), 1) for c in _CODES]
)


def test_the_renderer_tables_cover_the_reflected_vocabulary() -> None:
    """C2 — derived: a vocabulary member added to `mitos.amend` reds here."""
    assert _STATUSES and _CODES and amend.REASON_DIVERGED in _REASONS
    assert set(cli._AMEND_STATUS_RENDERERS) == set(_STATUSES)
    assert set(cli._AMEND_REFUSAL_RENDERERS) == set(_REASONS)
    assert set(cli._AMEND_ERROR_ACTIONS) == set(_CODES)


@pytest.mark.parametrize("label, result, exit_code", _CLASSES, ids=[c[0] for c in _CLASSES])
def test_every_result_class_renders_on_its_channel(tmp_path, capsys, label, result,
                                                   exit_code) -> None:
    """C2 + C11 — the echo leads the answer's channel; a refusal leaves stdout empty."""
    config = MitosConfig(str(tmp_path), project="p")
    with patch("mitos.cli.MitosSyncManager") as manager:
        manager.return_value.amend_commentary.return_value = copy.deepcopy(result)
        assert cmd_amend_commentary(config, "t", {"context": "x"}) == exit_code
    captured = capsys.readouterr()
    channel, other = (captured.out, captured.err) if exit_code == 0 else (captured.err, captured.out)
    assert other == ""
    assert _leads_with_echo(channel, config)
    clause = _clause_lines(channel)
    assert clause and "unrecognized" not in channel


def test_a_fidelity_raise_renders_as_exit_two(tmp_path, capsys) -> None:
    config = MitosConfig(str(tmp_path), project="p")
    with patch("mitos.cli.MitosSyncManager") as manager:
        manager.return_value.amend_commentary.side_effect = BufferFidelityError("a fact")
        assert cmd_amend_commentary(config, "t", {"context": "x"}) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and _leads_with_echo(captured.err, config)
    assert "Amend refused [buffer_fidelity]: a fact" in captured.err


def test_a_rename_whose_citers_could_not_be_read_says_so(tmp_path) -> None:
    config = MitosConfig(str(tmp_path), project="p")
    base = _canned(status=amend.STATUS_AMENDED)
    unread = _render_amend_result({**base, "rename": {"from": "a", "to": "b", "incoming": None}},
                                  config)
    assert any("could not be read" in line for line in unread)
    uncited = _render_amend_result({**base, "rename": {"from": "a", "to": "b", "incoming": []}},
                                   config)
    assert not any("Cited by" in line or "diverged" in line for line in uncited)


# --------------------------------------------------------------------------- #
# C3 / C4 / C11 — end to end through main() and the real core
# --------------------------------------------------------------------------- #

def _plain(config, m):
    _record(m, "other")
    _record(m, "target")


def _archived(config, m):
    _archive_one(config, m)


def _draft(config, m):
    _record(m, "target")
    draft = "### draft-one\n\n**Decided:** A draft axiom.\n**Rejected:** Draft rejected.\n"
    _write_buffer(config, _buffer(config).replace(_ENTRIES_MARKER,
                                                  f"{_ENTRIES_MARKER}\n\n{draft}", 1))


def _diverged(config, m):
    _plain(config, m)
    text = _buffer(config)
    assert text.count("The target context.") == 1
    _write_buffer(config, text.replace("The target context.", "A hand edit, not reconciled."))


def _check_amended(config, m, out, err):
    assert "Amended 'target' ✓" in out
    assert m.store.get_node_by_slug("target")["context"] == "A repaired context."


def _check_unchanged(config, m, out, err):
    assert "already holds those values" in out


def _check_core(config, m, out, err):
    assert f"mitos record -p {NAME!r}" in err
    for relation in amend.ROUTES.values():
        assert f"--{relation} 'target'" in err
        assert f"--{relation}" in _record_flags()   # a renamed record flag reds here


def _check_edges(config, m, out, err):
    assert "[edges]" in err
    assert f"mitos sync -p {NAME!r} --reconcile-entry 'target'" in err


def _check_archived(config, m, out, err):
    assert "decisions/archive/" in err and f"mitos rebuild -p {NAME!r}" in err


def _check_uncommitted(config, m, out, err):
    assert "not committed yet" in err
    assert all("mitos " not in line for line in _clause_lines(err))


def _check_not_found(config, m, out, err):
    assert f"mitos list -p {NAME!r} --oneline" in err
    assert "mitos sync" not in err   # D-4b-5: this verb read the buffer


def _check_diverged(config, m, out, err):
    assert "[diverged]" in err and "'context'" in err
    assert f"mitos sync -p {NAME!r} --reconcile-entry 'target'" in err


_SCENARIOS = {
    "amended": (_plain, ["target", "--context", "A repaired context."], 0, _check_amended),
    "unchanged": (_plain, ["target", "--context", "The target context."], 0, _check_unchanged),
    "axiom": (_plain, ["target", "--axiom", "A different axiom."], 1, _check_core),
    "mechanisms": (_plain, ["target", "--mechanisms", "m"], 1, _check_core),
    "cites": (_plain, ["target", "--cites", "other"], 1, _check_edges),
    "archived": (_archived, ["old-one", "--context", "A repair."], 1, _check_archived),
    "uncommitted": (_draft, ["draft-one", "--context", "A repair."], 1, _check_uncommitted),
    "not_found": (_plain, ["no-such-handle", "--context", "A repair."], 1, _check_not_found),
    "diverged": (_diverged, ["target", "--context", "A repair."], 1, _check_diverged),
}


@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
def test_end_to_end_text(ws, capsys, scenario) -> None:
    """C3 + C11 — exit, channel, echo on that channel, and nothing written on a refusal."""
    config, m = ws
    setup, argv, exit_code, check = _SCENARIOS[scenario]
    setup(config, m)
    sha = _sha(config)

    code, out, err = _amend(capsys, *argv)

    assert code == exit_code, (out, err)
    assert _leads_with_echo(out if exit_code == 0 else err, config), (out, err)
    if exit_code != 0:
        assert out == ""
    if scenario != "amended":
        assert _sha(config) == sha
    check(config, m, out, err)


@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
def test_end_to_end_json_is_the_result_plus_provenance(ws, capsys, monkeypatch, scenario) -> None:
    """C4 — one object, the core's dict verbatim plus the three keys, the same exit."""
    config, m = ws
    setup, argv, exit_code, _check = _SCENARIOS[scenario]
    setup(config, m)
    returned: List[Dict] = []
    real = MitosSyncManager.amend_commentary

    def spy(self, slug, changes):
        result = real(self, slug, changes)
        returned.append(copy.deepcopy(result))
        return result

    monkeypatch.setattr(MitosSyncManager, "amend_commentary", spy)
    code, out, _err = _amend(capsys, *argv, "--json")

    assert code == exit_code
    assert json.loads(out) == {**returned[0], **corpus_provenance(config)}


# --------------------------------------------------------------------------- #
# C5 — rename
# --------------------------------------------------------------------------- #

def test_a_rename_lists_its_citers_and_editing_their_line_is_the_whole_repair(ws, capsys) -> None:
    """C5 — the clause names no command; the edit it describes clears the divergence."""
    config, m = ws
    _record(m, "target")
    _record(m, "citer", cites="target")

    code, out, _err = _amend(capsys, "target", "--new-slug", "target-renamed")
    assert code == 0
    assert "'target' → 'target-renamed'" in out and "'citer' (cites)" in out
    assert all("mitos " not in line for line in _clause_lines(out))

    code, _out, err = _amend(capsys, "citer", "--context", "A repair.")
    assert code == 1 and "[diverged]" in err

    text = _buffer(config)
    assert text.count("**Cites:** target\n") == 1
    _write_buffer(config, text.replace("**Cites:** target\n", "**Cites:** target-renamed\n"))
    citer = m.store.get_node_by_slug("citer")
    report = entry_divergence(_entry(config, "citer"), citer, citer["scope"],
                              m.store.get_outgoing_edges(citer["id"]))
    assert not any(report.values()), report

    code, out, _err = _amend(capsys, "citer", "--context", "A repair.")
    assert code == 0 and "Amended 'citer' ✓" in out


def test_a_rename_under_json_carries_the_incoming_rows(ws, capsys) -> None:
    config, m = ws
    _record(m, "target")
    _record(m, "citer", cites="target")
    code, out, _err = _amend(capsys, "target", "--new-slug", "target-renamed", "--json")
    assert code == 0
    assert json.loads(out)["rename"] == {
        "from": "target", "to": "target-renamed",
        "incoming": [{"kind": "cites", "source": "citer"}],
    }


def test_a_rename_onto_an_active_slug_rolls_back(ws, capsys) -> None:
    config, m = ws
    _record(m, "first")
    _record(m, "second")
    sha = _sha(config)
    code, out, err = _amend(capsys, "first", "--new-slug", "second")
    assert code == 1 and out == ""
    assert "Amend failed [slug_collision]" in err and "--new-slug" in err
    assert _sha(config) == sha


# --------------------------------------------------------------------------- #
# C6 — fidelity
# --------------------------------------------------------------------------- #

def test_a_phantom_heading_is_exit_two_and_writes_nothing(ws, capsys) -> None:
    """C6 + C11."""
    config, m = ws
    _plain(config, m)
    sha = _sha(config)
    code, out, err = _amend(capsys, "target", "--context", "c\n### phantom")
    assert code == 2 and out == ""
    assert _leads_with_echo(err, config) and "[buffer_fidelity]" in err
    # Ledger entry-003 (4c): the parse failure reads as its entry, never an object repr.
    assert "parse failure: entry 'phantom'" in err and "object at 0x" not in err
    assert _sha(config) == sha


def test_a_phantom_heading_under_json_is_one_object(ws, capsys) -> None:
    config, m = ws
    _plain(config, m)
    sha = _sha(config)
    code, out, _err = _amend(capsys, "target", "--context", "c\n### phantom", "--json")
    assert code == 2
    payload = json.loads(out)
    assert payload["code"] == cli.AMEND_CODE_BUFFER_FIDELITY and payload["slug"] == "target"
    assert "parse failure: entry 'phantom'" in payload["error"]
    assert "object at 0x" not in payload["error"]
    assert {k: payload[k] for k in ("project", "collection", "workspace")} == corpus_provenance(config)
    assert _sha(config) == sha


# --------------------------------------------------------------------------- #
# C7 — the real frame: `python -m mitos.cli`
# --------------------------------------------------------------------------- #

def _env(base: str) -> Dict[str, str]:
    return {**os.environ, "MITOS_NO_UPDATE_CHECK": "1",
            "XDG_CONFIG_HOME": os.path.join(base, "xdg_config"),
            "XDG_CACHE_HOME": os.path.join(base, "xdg_cache"),
            "GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", "ANTHROPIC_API_KEY": "",
            "QDRANT_URL": DEAD_QDRANT_URL}


def _mitos(env: Dict[str, str], cwd: str, *argv: str, **kwargs) -> "subprocess.CompletedProcess[str]":
    options = {"capture_output": True, **kwargs}
    return subprocess.run([sys.executable, "-m", "mitos.cli", *argv], cwd=cwd, env=env,
                          text=True, timeout=300, **options)


def test_the_real_frame_carries_the_three_exits_and_the_echo_order(tmp_path) -> None:
    """C7 — W18 / T13's CLI half through the entry point, keyless, dead Qdrant."""
    base = str(tmp_path)
    root = os.path.join(base, "frame")
    os.makedirs(root)
    env = _env(base)
    assert _mitos(env, root, "init", "--name", "frame").returncode == 0
    recorded = _mitos(env, base, "-p", "frame", "record", "The frame axiom.", "--slug", "target",
                      "--rejected", "The frame alternative.", "--scope", "core")
    assert recorded.returncode == 0, recorded.stdout + recorded.stderr

    amended = _mitos(env, base, "-p", "frame", "amend-commentary", "target",
                     "--context", "A context set through the frame.")
    assert amended.returncode == 0, amended.stderr
    assert "Amended 'target' ✓" in amended.stdout and amended.stdout.startswith("corpus: ")

    fidelity = _mitos(env, base, "-p", "frame", "amend-commentary", "target",
                      "--context", "c\n### phantom")
    assert fidelity.returncode == 2 and "[buffer_fidelity]" in fidelity.stderr
    assert fidelity.stdout == ""

    missing = _mitos(env, base, "-p", "frame", "amend-commentary", "no-such-handle",
                     "--context", "x", "--json")
    assert missing.returncode == 1
    assert json.loads(missing.stdout)["status"] == amend.STATUS_NOT_FOUND

    combined = _mitos(env, base, "-p", "frame", "amend-commentary", "no-such-handle",
                      "--context", "x", capture_output=False,
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = combined.stdout.splitlines()
    echo = next(i for i, line in enumerate(lines) if line.startswith("corpus: "))
    answer = next(i for i, line in enumerate(lines) if "names no decision" in line)
    assert combined.returncode == 1 and echo < answer

    # Archive one entry for real, in-process over the same files, then ask the frame.
    config = MitosConfig(root)
    with patch("mitos.sync.GeminiEmbeddingProvider", MagicMock(side_effect=Exception("down"))), \
            patch("mitos.sync.QdrantVectorStore", MagicMock(side_effect=Exception("down"))):
        _archive_one(config, MitosSyncManager(config))
    archived = _mitos(env, base, "-p", "frame", "amend-commentary", "old-one", "--context", "x")
    assert archived.returncode == 1 and archived.stdout == ""
    assert "mitos rebuild -p 'frame'" in archived.stderr


# --------------------------------------------------------------------------- #
# C8 / C9 / C10 — the parser
# --------------------------------------------------------------------------- #

def _relation_flags(sub) -> set:
    return {opt for action in sub._actions if action.dest in RELATIONSHIP_FIELDS
            for opt in action.option_strings}


def test_the_refused_relation_flags_are_derived_and_match_records() -> None:
    """C8 — a tenth relation reaches both verbs, or neither."""
    choices = _subparsers(_build_parser())
    derived = {"--" + field.replace("_", "-") for field in RELATIONSHIP_FIELDS}
    assert len(derived) == len(RELATIONSHIP_FIELDS) and "--cites" in derived
    assert _relation_flags(choices["amend-commentary"]) == derived
    assert _relation_flags(choices["record"]) == derived


def test_the_verb_has_no_mcp_name_alias() -> None:
    """C9 — D-4b-1: the signed set of five stays five."""
    choices = _subparsers(_build_parser())
    assert "amend_commentary" not in choices
    assert sum(sub is choices["amend-commentary"] for sub in choices.values()) == 1
    assert len(_ALIASES) == 5 and "amend_commentary" not in _ALIASES


def test_the_help_states_the_reach() -> None:
    """C10 — entries still in decisions.md; archived ones through `mitos rebuild`."""
    parser = _build_parser()
    subaction = parser._subparsers._group_actions[0]
    listed = next(a.help for a in subaction._choices_actions if a.dest == "amend-commentary")
    description = subaction.choices["amend-commentary"].description
    for text in (listed, description):
        flat = " ".join(text.split())
        assert "still in decisions.md" in flat and "mitos rebuild" in flat
        assert "madr" not in flat.lower() and "amend_commentary" not in flat
