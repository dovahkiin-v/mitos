"""Cluster B checkpoint: rotation reach through the real CLI and the real MCP stdio server (3e).

Phases 3a–3d proved their properties in process, or through one frame for one verb. Every
row here enters the process a person or an agent runs — ``python -m mitos.cli`` and
``mitos serve`` over JSON-RPC — against a workspace that rotation has actually drained,
and asserts what B promised as one thing: the receipt names the file, the file holds what
moved, ``status`` shows the buffer, the render stays honest, the corpus readers answer from
the archives, and ``rebuild`` returns the same graph.

**The frame contract.** Each row is a self-contained journey over its own workspace, built
from markdown below the sentinel and committed with ``rebuild --yes --json`` (the 2g idiom:
a keyless ``sync`` commits nothing). Rotation mutates state, so a shared fixture would make
rows depend on scheduling under ``-n auto``. Every expected number comes from an oracle
over the files the child wrote — ``settledness.buffered_entries``, the imported
``_ARCHIVE_FILENAME_RE``, ``parse_file_reversed`` in ``rebuild``'s order, the renderer's
constants — never a figure typed here.

**Exactly one substitution crosses a process boundary.** A keyless ``sync`` returns before
its rotation step (``Sync requires API keys``), so a sync that rotates runs through
``LAUNCHER``: a fresh interpreter that patches the provider SDK class
``google.genai.Client`` and then runs the real ``mitos.cli.main()`` with real argv, selector
resolution, config and streams. The key it sees is a dummy, Qdrant is a dead port and no
judge key exists, so nothing is spent. ``record``, ``status``, ``rebuild``, ``serve`` and
F2b's bare ``sync`` run unpatched and keyless.
"""

import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mitos.renderer as R
from mitos import lexical, rotation, settledness
from mitos.cutover import _ARCHIVE_FILENAME_RE, _archive_files_oldest_first
from mitos.divergence import divergence_total
from mitos.parser import corpus_has_entries, parse_file_reversed
from render_sweep import sweep_destinations, tree_from_disk
from test_mcp_stdio_harness import _raw_stdio_exchange

W = settledness.ROTATION_WINDOW_ENTRIES
DEAD_QDRANT_URL = "http://127.0.0.1:9"
FAILURE_PREFIX = "[Warning] Archive rotation failed"
KEY_LINE_PREFIX = "GEMINI_API_KEY environment variable is not set"

#: The sync frame (D-3e-2). ``argv[1:]`` are mitos's own arguments; nothing else is
#: substituted.
LAUNCHER = (
    "import sys\n"
    "from unittest.mock import patch\n"
    "with patch('google.genai.Client'):\n"
    "    import mitos.cli\n"
    "    sys.argv = ['mitos', *sys.argv[1:]]\n"
    "    mitos.cli.main()\n"
)

_SYNC_ROTATED = re.compile(r"^Rotated (\d+) entries to (.+) ✓$")
_RECORD_ROTATED = re.compile(r"^  Rotated:   (\d+) older settled entr(?:y|ies) to (.+)$")
_RECIPE = re.compile(r"`(mitos rebuild -p [^`]+)`")

# `status` keys that legitimately move across a write: the two buffer measurements this
# cluster added, and the node counts a `record` raises (the 2g/3d count-shaped idiom).
_MOVING_CHECKS = {"decisions_buffer_entries", "decisions_buffer_chars",
                  "graph_nodes", "active_nodes"}


# --------------------------------------------------------------------------- #
# Builders — copied from the 2g/3c idioms, deliberately not generalised
# --------------------------------------------------------------------------- #


def _env(base: str) -> Dict[str, str]:
    return {**os.environ, "MITOS_NO_UPDATE_CHECK": "1",
            "XDG_CONFIG_HOME": os.path.join(base, "xdg_config"),
            "XDG_CACHE_HOME": os.path.join(base, "xdg_cache"),
            "GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", "ANTHROPIC_API_KEY": "",
            "QDRANT_URL": DEAD_QDRANT_URL}


def _mitos(env: Dict[str, str], cwd: str, *argv: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run([sys.executable, "-m", "mitos.cli", *argv], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)


def _launcher_sync(env: Dict[str, str], ws: str) -> "subprocess.CompletedProcess[str]":
    done = subprocess.run([sys.executable, "-c", LAUNCHER, "-p", ws, "sync"], cwd=ws,
                          env={**env, "GEMINI_API_KEY": "frame-dummy"},
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    return done


def _entry(slug: str, axiom: str, *, rejected: str = "The obvious alternative.",
           scope: Optional[Sequence[str]] = ("core",),
           relations: Sequence[Tuple[str, str]] = ()) -> str:
    lines = [f"### {slug}", "", f"**Decided:** {axiom}", f"**Rejected:** {rejected}"]
    if scope:
        lines.append(f"**Scope:** {', '.join(scope)}")
    lines += [f"**{field}:** [{target}]" for field, target in relations]
    return "\n".join(lines) + "\n\n"


def _plain(slugs: Sequence[str]) -> List[Tuple[str, str]]:
    return [(s, _entry(s, f"The {s} axiom holds for its own reason.")) for s in slugs]


def _newest_first(entries: Sequence[Tuple[str, str]]) -> str:
    """A corpus file's entry stream from oldest-first entries: the oldest goes LAST."""
    return "".join(text for _slug, text in reversed(entries))


def _build(base: str, env: Dict[str, str], name: str,
           entries: Sequence[Tuple[str, str]], *, threshold: int, lag: int = 1,
           archives: Optional[Dict[str, Sequence[Tuple[str, str]]]] = None) -> str:
    """Initialises, writes the corpus, commits it with ``rebuild``, then arms rotation.

    ``entries`` are oldest-first (commit order) and land newest-first below the
    sentinel, so the oldest is the buffer's tail. ``archives`` pre-writes archive files,
    which ``rebuild`` replays first. Nodes are left freshly stamped: arming the clock
    (``_back_date``) is each row's own step, after its last build step.
    """
    ws = os.path.join(base, name)
    os.makedirs(ws)
    init = _mitos(env, ws, "init", "--name", name)
    assert init.returncode == 0, init.stdout + init.stderr
    with open(os.path.join(ws, "decisions.md"), "a", encoding="utf-8") as fh:
        fh.write("\n" + _newest_first(entries))
    for archive_name, archived in (archives or {}).items():
        os.makedirs(_archive_dir(ws), exist_ok=True)
        with open(os.path.join(_archive_dir(ws), archive_name), "w", encoding="utf-8") as fh:
            fh.write(_newest_first(archived))
    rebuilt = _mitos(env, ws, "-p", ws, "rebuild", "--yes", "--json")
    report = json.loads(rebuilt.stdout)
    assert rebuilt.returncode == 0 and report["swapped"] is True, rebuilt.stdout + rebuilt.stderr
    assert not report["residual_casualties"], report

    config_path = os.path.join(ws, ".mitos", "config.toml")
    with open(config_path, encoding="utf-8") as fh:
        text = fh.read()
    # Replace, never append: tomllib refuses a duplicate key.
    text, n_threshold = re.subn(r"^rotation_volume_threshold_entries = .*$",
                                f"rotation_volume_threshold_entries = {threshold}", text,
                                flags=re.M)
    text, n_lag = re.subn(r"^rotation_lag_days = .*$", f"rotation_lag_days = {lag}", text,
                          flags=re.M)
    assert (n_threshold, n_lag) == (1, 1), text
    with open(config_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return ws


def _back_date(ws: str, days: int = 2) -> None:
    """Makes every node quiet: rotation's clock reads ``updated_at``."""
    stamp = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    conn = sqlite3.connect(_db(ws))
    with conn:
        conn.execute("UPDATE nodes SET updated_at = ?", (stamp,))
    conn.close()


# --------------------------------------------------------------------------- #
# Oracles over the files a child wrote
# --------------------------------------------------------------------------- #


def _db(ws: str) -> str:
    return os.path.join(ws, ".mitos", "graph.sqlite")


def _buffer(ws: str) -> str:
    return os.path.join(ws, "decisions.md")


def _archive_dir(ws: str) -> str:
    return os.path.join(ws, "decisions", "archive")


def _read(path: str) -> str:
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def _buffered(ws: str) -> int:
    return settledness.buffered_entries(_read(_buffer(ws)))


def _slugs_oldest_first(path: str) -> List[str]:
    failures: list = []
    slugs = [e.slug for e in parse_file_reversed(path, "decision", failures)]
    assert failures == [], failures
    return slugs


def _buffer_file_order(ws: str) -> List[str]:
    """The buffer's slugs head first — the newest entry is element 0."""
    return list(reversed(_slugs_oldest_first(_buffer(ws))))


def _stream(ws: str) -> List[str]:
    """The replay stream, in ``rebuild``'s order: archives oldest file first, then the buffer."""
    out: List[str] = []
    for path in _archive_files_oldest_first(_archive_dir(ws)) + [_buffer(ws)]:
        out += _slugs_oldest_first(path)
    return out


def _archive_snapshot(ws: str) -> Optional[Dict[str, bytes]]:
    directory = _archive_dir(ws)
    if not os.path.isdir(directory):
        return None
    snapshot = {}
    for name in sorted(os.listdir(directory)):
        with open(os.path.join(directory, name), "rb") as fh:
            snapshot[name] = fh.read()
    return snapshot


def _ids(ws: str) -> set:
    conn = sqlite3.connect(f"file:{_db(ws)}?mode=ro", uri=True)
    try:
        return {row[0] for row in conn.execute("SELECT id FROM nodes")}
    finally:
        conn.close()


def _status_json(env: Dict[str, str], ws: str) -> Tuple[int, Dict[str, Any]]:
    done = _mitos(env, ws, "-p", ws, "status", "--json")
    return done.returncode, json.loads(done.stdout)


def _verdict_line(env: Dict[str, str], ws: str) -> Tuple[int, str]:
    done = _mitos(env, ws, "-p", ws, "status")
    [line] = [l for l in done.stdout.splitlines() if l.startswith("MITOS STATUS for ")]
    return done.returncode, line


def _strings(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _drain(env: Dict[str, str], ws: str, *, limit: int = 12) -> List[Dict[str, Any]]:
    """Runs launcher syncs until one moves nothing, then one more; returns every run.

    Each run records the buffered count after it, the moved total its stderr reports,
    and both streams. The last two runs are the fixed-point pair.
    """
    runs: List[Dict[str, Any]] = []
    for _ in range(limit):
        before = (_read(_buffer(ws)), _archive_snapshot(ws))
        done = _launcher_sync(env, ws)
        after = (_read(_buffer(ws)), _archive_snapshot(ws))
        reported = [_SYNC_ROTATED.match(l) for l in done.stderr.splitlines()]
        runs.append({"buffered": _buffered(ws), "moved": after != before,
                     "reported": [(int(m.group(1)), m.group(2)) for m in reported if m],
                     "stdout": done.stdout, "stderr": done.stderr, "bytes": after})
        if len(runs) >= 2 and not runs[-2]["moved"]:
            return runs
    raise AssertionError(f"no fixed point in {limit} syncs: "
                         f"{[r['buffered'] for r in runs]}")


# --------------------------------------------------------------------------- #
# F1 — the CLI record journey
# --------------------------------------------------------------------------- #


def test_f1_a_cli_record_rotates_the_settled_tail_and_status_follows(tmp_path) -> None:
    """F1: ``status`` → ``record`` → ``status`` over an armed, settled workspace.

    The buffer holds ``T − 1`` settled entries, so the record's own write crosses the
    threshold and rotation moves every older entry (``T − 1 ≤ W``); the new entry is
    recent and stays at the head. ``status`` must read the buffer the child left through
    the shipped count, before and after, while its verdict does not move.
    """
    threshold = 4
    base = str(tmp_path)
    env = _env(base)
    seeds = _plain([f"f1-seed-{i}" for i in range(threshold - 1)])
    ws = _build(base, env, "f1ws", seeds, threshold=threshold)
    _back_date(ws)

    rc_before, before = _status_json(env, ws)
    text_rc_before, verdict_before = _verdict_line(env, ws)
    buffered_before = _buffered(ws)
    assert before["checks"]["decisions_buffer_entries"] == buffered_before == threshold - 1
    assert before["checks"]["decisions_buffer_chars"] == len(_read(_buffer(ws)))

    done = _mitos(env, ws, "-p", ws, "record", "The f1 crossing write holds.",
                  "--rejected", "rej", "--slug", "f1-crossing", "--acknowledge-neighbors")

    assert done.returncode == 0, done.stdout + done.stderr
    [receipt] = [m for m in map(_RECORD_ROTATED.match, done.stdout.splitlines()) if m]
    [archive_name] = os.listdir(_archive_dir(ws))
    archive_path = os.path.join(_archive_dir(ws), archive_name)
    assert _ARCHIVE_FILENAME_RE.match(archive_name), archive_name
    assert receipt.group(2) == archive_path
    assert receipt.group(2).startswith(_archive_dir(ws) + os.sep)
    moved = _slugs_oldest_first(archive_path)
    assert moved, "nothing moved: the receipt row would compare an empty set"
    assert int(receipt.group(1)) == len(moved)
    assert moved == [slug for slug, _t in seeds]
    assert _buffer_file_order(ws)[0] == "f1-crossing"
    assert not any(l.startswith((FAILURE_PREFIX, "Rotated ")) for l in done.stderr.splitlines())

    rc_after, after = _status_json(env, ws)
    text_rc_after, verdict_after = _verdict_line(env, ws)
    buffered_after = _buffered(ws)
    assert buffered_after == buffered_before + 1 - len(moved)
    assert after["checks"]["decisions_buffer_entries"] == buffered_after
    assert after["checks"]["decisions_buffer_chars"] == len(_read(_buffer(ws)))
    assert (rc_after, after["ready"], text_rc_after, verdict_after) == \
        (rc_before, before["ready"], text_rc_before, verdict_before)
    assert list(after["checks"]) == list(before["checks"])
    assert {k: v for k, v in after["checks"].items() if k not in _MOVING_CHECKS} == \
        {k: v for k, v in before["checks"].items() if k not in _MOVING_CHECKS}


# --------------------------------------------------------------------------- #
# F2 / F2b — the sync frame to its fixed point, and the keyless floor
# --------------------------------------------------------------------------- #


def test_f2_repeated_cli_syncs_drain_to_a_fixed_point_below_threshold(tmp_path) -> None:
    """F2: launcher syncs over ``3·W + (T − 2)`` quiet entries until one moves nothing.

    Sized from the imported window so the batch — capped by ``W``, never by ``T`` —
    leaves a non-empty resting buffer below the threshold: the fixed point is then a
    keyed sync that evaluates and selects nothing, not an empty buffer's early return.
    """
    threshold = 6
    base = str(tmp_path)
    env = _env(base)
    count = 3 * W + (threshold - 2)
    assert count >= 2 * W + threshold
    ws = _build(base, env, "f2ws", _plain([f"f2-entry-{i:03d}" for i in range(count)]),
                threshold=threshold)
    _back_date(ws)
    stream_before = _stream(ws)
    assert _archive_snapshot(ws) is None

    runs = _drain(env, ws)

    counts = [count] + [run["buffered"] for run in runs]
    assert all(a >= b for a, b in zip(counts, counts[1:])), counts
    rotating = [run for run in runs if run["moved"]]
    assert len(rotating) >= 3, counts
    for previous, run in zip(counts, counts[1:]):
        assert previous - run <= W, counts
    for run, (previous, now) in zip(runs, zip(counts, counts[1:])):
        assert sum(n for n, _p in run["reported"]) == previous - now, run["stderr"]
        for _n, path in run["reported"]:
            assert _ARCHIVE_FILENAME_RE.match(os.path.basename(path)), path
            assert os.path.dirname(path) == _archive_dir(ws)
        assert not any(l.startswith("Rotated ") for l in run["stdout"].splitlines())
        assert not any(l.startswith(FAILURE_PREFIX)
                       for l in (run["stdout"] + run["stderr"]).splitlines())
    assert all(run["reported"] for run in rotating)
    assert 0 < counts[-1] < threshold, counts
    assert runs[-1]["bytes"] == runs[-2]["bytes"], "the last sync changed bytes at rest"
    assert _stream(ws) == stream_before


def test_f2b_a_keyless_sync_evaluates_no_rotation(tmp_path) -> None:
    """F2b: the keyless floor, pinned as current truth (D-3e-5).

    A keyless ``sync`` returns at ``Sync requires API keys`` before its rotation step, so
    a workspace with no ``GEMINI_API_KEY`` drains only through ``record``. The vision's
    D2 says the predicate is evaluated identically on both paths; this row states what
    ships, so a change inverts a test rather than a comment. Handed to 5b's ROADMAP as:
    *keyless `mitos sync` evaluates no rotation — a keyless workspace drains only through
    `record` (trigger: an operator reports a keyless sync did not drain the buffer).*
    """
    threshold = 3
    base = str(tmp_path)
    env = _env(base)
    ws = _build(base, env, "f2bws", _plain([f"f2b-entry-{i}" for i in range(threshold + 2)]),
                threshold=threshold)
    _back_date(ws)
    buffer_before = _read(_buffer(ws))
    assert settledness.buffered_entries(buffer_before) >= threshold
    assert _archive_snapshot(ws) is None

    done = _mitos(env, ws, "-p", ws, "sync")

    assert done.returncode == 0, done.stdout + done.stderr
    assert any(l.startswith(KEY_LINE_PREFIX) for l in done.stdout.splitlines()), done.stdout
    assert _read(_buffer(ws)) == buffer_before
    assert _archive_snapshot(ws) is None
    assert not any(l.startswith("Rotated ") for l in (done.stdout + done.stderr).splitlines())


# --------------------------------------------------------------------------- #
# F3 — the MCP stdio journey
# --------------------------------------------------------------------------- #


F3_SLUG = "quixotic-lantern-ledger"


def test_f3_an_mcp_record_rotates_and_the_query_reaches_the_archive(tmp_path) -> None:
    """F3: one ``serve``, ``record_decision`` then ``query_decisions``, raw stdio.

    Keyless, so ``query_decisions`` has no provider and answers from the lexical arm over
    ``_corpus_files`` — buffer plus archives. The scout found that this envelope carries no
    missing-graph note even over a deleted graph, so gotcha 6 resolves to the lexical arm
    only; the unbuilt property stays with 3a's in-process rows and F4's CLI rung.
    """
    threshold = 4
    base = str(tmp_path)
    env = _env(base)
    seeds = [(F3_SLUG, _entry(F3_SLUG, "Quixotic lanterns guard the ledger archive."))]
    seeds += _plain([f"f3-seed-{i}" for i in range(threshold - 2)])
    ws = _build(base, env, "f3ws", seeds, threshold=threshold)
    _back_date(ws)
    terms = ["quixotic", "lanterns"]
    assert all(len(t) >= lexical.LEXICAL_MIN_TERM_LEN for t in terms)

    exchange = "".join(json.dumps(message) + "\n" for message in (
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "cluster-b-probe", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "record_decision", "arguments": {
             "project": ws, "axiom": "The f3 crossing write holds.", "rejected_paths": "rej",
             "scope": [], "slug": "f3-crossing", "acknowledge_neighbors": True}}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "query_decisions", "arguments": {
             "project": ws, "query": " ".join(terms)}}},
    ))

    stdout, stderr = _raw_stdio_exchange(exchange, expect_lines=3, cwd=ws, env=env)

    lines = [line for line in stdout.splitlines() if line.strip()]
    assert len(lines) == 3, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    messages = []
    for line in lines:
        try:
            messages.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AssertionError(f"non-JSON line on the protocol channel: {line!r}") from exc
    by_id = {m.get("id"): m for m in messages}
    record, query = (json.loads(by_id[i]["result"]["content"][0]["text"]) for i in (2, 3))

    assert record["status"] == "created", record
    report = record["rotation"]
    assert report["outcome"] == "rotated" and "recovery" not in report, report
    [archive] = report["archives"]
    assert os.path.dirname(archive["path"]) == _archive_dir(ws)
    assert _ARCHIVE_FILENAME_RE.match(os.path.basename(archive["path"]))
    assert sorted(_slugs_oldest_first(archive["path"])) == sorted(archive["slugs"])
    assert F3_SLUG in archive["slugs"]
    assert F3_SLUG not in _buffer_file_order(ws), "the slug is not archive-only"

    assert query["degraded"] == "lexical", query
    assert any(match.get("slug") == F3_SLUG for match in query["matches"]), query
    assert "unread_files" not in query
    for body in (record, query):
        assert not [s for s in _strings(body) if "mitos " in s], body


# --------------------------------------------------------------------------- #
# F4 — T9 through a real drain, healed by the printed recipe
# --------------------------------------------------------------------------- #


def test_f4_a_drained_corpus_heals_through_the_rung_it_prints(tmp_path) -> None:
    """F4: drain to zero through the sync frame, delete the graph, run the rung's recipe.

    The corpus carries 1d's dependency shape across two batches of one quarter file:
    ``base`` rotates in batch 1, while ``amender`` (amends it) and ``successor``
    (supersedes it) rotate in batch 2. An archive batch written in the wrong order, or a
    reader that scans only the buffer, then shows up as a casualty, an id mismatch or a
    wrong rung rather than as a pass.

    The recipe is the rung's own backticked span, run with ``python -m mitos.cli`` in place
    of ``mitos`` and with ``--yes`` and ``--json`` added: it is printed for a person at a
    terminal, and off a TTY ``rebuild`` refuses to prompt (3a's precedent).
    """
    base = str(tmp_path)
    env = _env(base)
    entries = [("base", _entry("base", "The base axiom that will be refined and replaced."))]
    entries += _plain([f"f4-filler-{i:02d}" for i in range(W - 1)])
    entries += [
        ("amender", _entry("amender", "A refinement of the base axiom.",
                           relations=[("Amends", "base")])),
        ("successor", _entry("successor", "The axiom that replaced the base.",
                             relations=[("Supersedes", "base")])),
    ]
    entries += _plain([f"f4-late-{i}" for i in range(3)])
    ws = _build(base, env, "f4ws", entries, threshold=1)
    _back_date(ws)

    runs = _drain(env, ws)

    assert [len(run["reported"]) > 0 for run in runs].count(True) >= 2, \
        [run["buffered"] for run in runs]
    assert _buffered(ws) == 0
    assert corpus_has_entries(_buffer(ws)) is False
    assert _archive_files_oldest_first(_archive_dir(ws))
    assert _stream(ws) == [slug for slug, _t in entries]
    ids_before = _ids(ws)

    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(_db(ws) + suffix):
            os.remove(_db(ws) + suffix)
    rc, unbuilt = _status_json(env, ws)
    assert unbuilt["checks"]["graph_unbuilt"] is True, unbuilt
    assert not os.path.exists(_db(ws))
    status = _mitos(env, ws, "-p", ws, "status")
    recipe = _RECIPE.search(status.stdout)
    assert recipe, status.stdout
    argv = shlex.split(recipe.group(1))
    assert argv[:3] == ["mitos", "rebuild", "-p"]

    healed = _mitos(env, ws, *argv[1:], "--yes", "--json")

    assert healed.returncode == 0, healed.stdout + healed.stderr
    report = json.loads(healed.stdout)
    assert report["swapped"] is True, report
    assert not report["residual_casualties"], report
    assert _ids(ws) == ids_before


# --------------------------------------------------------------------------- #
# F5 — the render stays honest across rotation
# --------------------------------------------------------------------------- #


F5_UNSCOPED = "f5-unscoped-tail"


def _rendered_bytes(ws: str) -> Dict[str, bytes]:
    paths = [os.path.join(ws, "live_axioms.md")]
    axioms = os.path.join(ws, ".mitos", "axioms")
    paths += [os.path.join(axioms, n) for n in sorted(os.listdir(axioms)) if n.endswith(".md")]
    out = {}
    for path in paths:
        with open(path, "rb") as fh:
            out[os.path.relpath(path, ws)] = fh.read()
    return out


def test_f5_rotation_leaves_every_render_byte_identical_and_honest(tmp_path) -> None:
    """F5: over the global ceiling, with an unscoped entry at the tail, a sync rotates it.

    Rotation writes no graph and every render is a function of the graph, so the render
    after a rotating sync must equal the render after a sync that rotated nothing. The
    first sync runs over freshly stamped nodes (nothing is quiet), the clock is then armed,
    and the second sync rotates. The corpus is sized over ``GLOBAL_OVERFLOW_WARN_CHARS``
    from a measured block weight, and the index title is asserted, so the unscoped heading
    check cannot pass vacuously.
    """
    base = str(tmp_path)
    env = _env(base)
    rejected = "A rejected path spelled out at length so the global render grows. " * 16
    weight = len(_entry("f5-bulk-000", "The f5 bulk axiom 000 holds.", rejected=rejected,
                        scope=["bulk"]))
    count = R.GLOBAL_OVERFLOW_WARN_CHARS // weight + 10
    entries = [(F5_UNSCOPED, _entry(F5_UNSCOPED, "An unscoped decision at the tail.",
                                    scope=None))]
    entries += [(f"f5-bulk-{i:03d}", _entry(f"f5-bulk-{i:03d}", f"The f5 bulk axiom {i:03d} holds.",
                                            rejected=rejected, scope=["bulk"]))
                for i in range(count)]
    ws = _build(base, env, "f5ws", entries, threshold=1)

    _rc, status = _status_json(env, ws)
    divergence = status["corpus_divergence"]
    assert divergence is not None and divergence.get("skipped") is None, divergence
    assert divergence_total(divergence) == 0, divergence

    quiet = _launcher_sync(env, ws)
    assert not any(l.startswith("Rotated ") for l in quiet.stderr.splitlines()), quiet.stderr
    assert _archive_snapshot(ws) is None
    rendered_before = _rendered_bytes(ws)

    _back_date(ws)
    rotating = _launcher_sync(env, ws)

    assert any(_SYNC_ROTATED.match(l) for l in rotating.stderr.splitlines()), rotating.stderr
    archived = [s for p in _archive_files_oldest_first(_archive_dir(ws))
                for s in _slugs_oldest_first(p)]
    assert F5_UNSCOPED in archived and F5_UNSCOPED not in _buffer_file_order(ws)
    assert _rendered_bytes(ws) == rendered_before

    content = _read(os.path.join(ws, "live_axioms.md"))
    assert content.split("\n", 1)[0] == "# Live Axioms" + R._INDEX_TITLE_SUFFIX
    assert [l for l in content.splitlines() if l.startswith("## (unscoped")] == ["## (unscoped)"]
    violations, counts = sweep_destinations(tree_from_disk(ws))
    assert violations == [], violations
    assert counts["unscoped_heading"] == 1, counts
    assert any("`decisions.md`" in l and "`decisions/archive/`" in l
               for l in content.splitlines()), content


# --------------------------------------------------------------------------- #
# F6 — the record-path failure on a drained corpus
# --------------------------------------------------------------------------- #


def test_f6_a_failing_record_rotation_on_a_drained_corpus_loses_nothing(tmp_path) -> None:
    """F6: a drained-corpus twin of 3c's R14 failure row.

    The history sits in two archives — an older quarter written before the build and the
    current quarter the drain wrote. After a committed, settled refill to ``T − 1``, the
    current rotation target is moved aside and a directory of its name takes its place,
    so the record's archive read raises a non-``FileNotFoundError`` ``OSError``.

    The test and the child each name the current quarter from their own clock; they could
    disagree only across a UTC quarter turnover mid-row.
    """
    threshold = 4
    base = str(tmp_path)
    env = _env(base)
    older = _plain([f"f6-old-{i}" for i in range(3)])
    ws = _build(base, env, "f6ws", _plain([f"f6-entry-{i}" for i in range(threshold + 2)]),
                threshold=threshold, archives={"2001-Q1.md": older})
    _back_date(ws)

    _drain(env, ws)
    assert _buffered(ws) < threshold
    target = rotation.archive_name_for(datetime.now(timezone.utc).isoformat())
    target_path = os.path.join(_archive_dir(ws), target)
    assert os.path.isfile(target_path)

    for i in range(threshold - 1 - _buffered(ws)):
        refill = _mitos(env, ws, "-p", ws, "record", f"The f6 refill-{i} axiom holds.",
                        "--rejected", "rej", "--slug", f"f6-refill-{i}",
                        "--acknowledge-neighbors")
        assert refill.returncode == 0, refill.stdout + refill.stderr
    assert _buffered(ws) == threshold - 1
    _back_date(ws)
    stream_before = _stream(ws)
    prior_buffer = _buffer_file_order(ws)
    older_bytes = _read(os.path.join(_archive_dir(ws), "2001-Q1.md"))

    aside = os.path.join(base, target + ".aside")
    shutil.move(target_path, aside)
    os.mkdir(target_path)
    try:
        done = _mitos(env, ws, "-p", ws, "record", "The f6 crossing write holds.",
                      "--rejected", "rej", "--slug", "f6-crossing", "--acknowledge-neighbors")
        buffer_after = _buffer_file_order(ws)
        older_after = _read(os.path.join(_archive_dir(ws), "2001-Q1.md"))
    finally:
        os.rmdir(target_path)
        shutil.move(aside, target_path)

    assert done.returncode == 0, done.stdout + done.stderr
    assert any(l.startswith("  Handle:") for l in done.stdout.splitlines()), done.stdout
    assert not any(_RECORD_ROTATED.match(l) for l in done.stdout.splitlines())
    failures = [l for l in done.stderr.splitlines() if l.startswith(FAILURE_PREFIX)]
    assert len(failures) == 1 and failures[0].startswith(f"{FAILURE_PREFIX} (file):"), done.stderr
    assert buffer_after == ["f6-crossing"] + prior_buffer
    assert older_after == older_bytes
    assert _stream(ws) == stream_before + ["f6-crossing"]
