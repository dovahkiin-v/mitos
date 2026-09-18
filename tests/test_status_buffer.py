"""T12 — `mitos status <project>` shows the buffer's size, and the format-spec row
stops reading as damage (surface-entropy 3d).

Since rotation runs on both write paths, `decisions.md` is a managed working set, and
its standing size belongs on the health report rather than on every write's receipt.
The size is a fact on the decisions row, never a verdict:

* **Entries is the trigger's own count** (`settledness.buffered_entries`), so a
  fresh `init` reads 0 — its sample block sits above the sentinel. A heading grep
  reads 1 there, which is why the fresh row exists.
* **The line is unconditional and carries no glyph.** The buffer reaches the rotation
  threshold on every cycle by design, so a threshold-conditioned warning would fire in
  the healthy, managed state.
* **Size never moves the mark, readiness, the verdict or the exit code** — proven on
  committed twins (a full buffer and its drained shape), because uncommitted entries
  over `init`'s empty graph are the unbuilt-graph rung and would read NOT READY for a
  reason unrelated to this phase.
* **`None` is "not measured"**, distinct from a measured `0`, in the payload's two
  sibling keys; the shipped `decisions_buffer` bool keeps its type.

Every assertion on text is scoped to the decisions row's follow-up line (or the
format-spec line): the overflow detail also says `chars` and `tokens`, so a
whole-output `in` could be satisfied by the wrong line. The offline `status` seam is
re-spelled here rather than imported from another test module.
"""

import json
import os
import re
import subprocess
import sys

import pytest

from mitos import cli
from mitos.config import MitosConfig
from mitos.parser import parse_entry_stream
from mitos.store import GraphStore


_THRESHOLD = 7   # not the default 50, so a literal threshold reds
_K = 9           # above _THRESHOLD


def _entry(i: int) -> str:
    # Non-ASCII on purpose: chars ≠ bytes, so a `getsize` count reds.
    return (f"\n### buffer-entry-{i:02d}\n\n"
            f"**Decided:** Žalgiris buffer entry {i} keeps its authored order — newest first.\n"
            f"**Rejected:** Reordering on write — it breaks every diff.\n"
            f"**Scope:** buffer\n")


_BLOCKS = "".join(_entry(i) for i in range(_K))


def _fresh(tmp_path, name="ws"):
    """`mitos init` into a new directory, with the rotation threshold set to 7."""
    ws = tmp_path / name
    ws.mkdir()
    cli.cmd_init(MitosConfig(str(ws)))
    _set_threshold(ws)
    return ws


def _set_threshold(ws, value=_THRESHOLD):
    path = os.path.join(str(ws), ".mitos", "config.toml")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # Replace, never append: tomllib refuses a duplicate key.
    text, n = re.subn(r"^rotation_volume_threshold_entries = .*$",
                      f"rotation_volume_threshold_entries = {value}", text, flags=re.M)
    assert n == 1, text
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _append(ws, text):
    with open(os.path.join(str(ws), "decisions.md"), "a", encoding="utf-8") as f:
        f.write(text)


def _buffer_text(ws) -> str:
    with open(os.path.join(str(ws), "decisions.md"), encoding="utf-8") as f:
        return f.read()


def _healthy(monkeypatch):
    """Everything else reads ready: a key, a reachable Qdrant, an empty scroll."""
    monkeypatch.setenv("GEMINI_API_KEY", "testkey")
    monkeypatch.setattr(cli, "_check_qdrant", lambda url, coll: {
        "reachable": True, "collection_exists": False, "points": 0})
    monkeypatch.setattr(cli, "scroll_point_ids",
                        lambda base_url, collection, page_size=256: set())


def _text(capsys, ws):
    capsys.readouterr()
    rc = cli.cmd_status(str(ws))
    return rc, capsys.readouterr().out


def _json(capsys, ws):
    capsys.readouterr()
    rc = cli.cmd_status(str(ws), as_json=True)
    return rc, json.loads(capsys.readouterr().out)


def _row_and_follow(out):
    """The decisions row and the line after it — never located by index."""
    lines = out.splitlines()
    i = next(i for i, ln in enumerate(lines) if "decisions.md buffer" in ln)
    return lines[i], lines[i + 1]


def _verdict(out):
    header = next(ln for ln in out.splitlines() if "MITOS STATUS for" in ln)
    return header.rsplit(" — ", 1)[1]


def _commit(ws, blocks):
    store = GraphStore(MitosConfig(str(ws)).db_path)
    entries = parse_entry_stream(blocks, "decision")
    assert len(entries) == _K
    for entry in entries:
        store.commit_parsed_entry(entry)


class TestTheBufferLine:
    def test_a_large_buffer_states_its_entries_chars_and_the_config_threshold(
            self, tmp_path, capsys, monkeypatch):
        """B1."""
        ws = _fresh(tmp_path)
        _append(ws, _BLOCKS)
        _healthy(monkeypatch)
        _, out = _text(capsys, ws)
        _, follow = _row_and_follow(out)
        chars = len(_buffer_text(ws))
        assert follow.startswith("      ") and follow.strip()
        assert follow.lstrip().startswith(f"{_K} entries, {chars:,} chars ")
        assert f"once {_THRESHOLD} or more" in follow
        assert "once 50 " not in follow  # the default, never a literal
        assert "⚠" not in follow
        assert "decisions.md buffer" not in follow

    def test_a_fresh_buffer_reads_zero_entries_not_its_sample_block(
            self, tmp_path, capsys, monkeypatch):
        """B2 — the sample `### example-slug` sits above the sentinel."""
        ws = _fresh(tmp_path)
        _healthy(monkeypatch)
        chars = len(_buffer_text(ws))
        assert "### " in _buffer_text(ws)  # non-vacuity: a heading grep would count it
        _, out = _text(capsys, ws)
        _, follow = _row_and_follow(out)
        assert follow.lstrip().startswith(f"0 entries, {chars:,} chars ")
        _, payload = _json(capsys, ws)
        assert payload["checks"]["decisions_buffer_entries"] == 0
        assert payload["checks"]["decisions_buffer_chars"] == chars

    def test_json_carries_two_int_siblings_beside_the_unchanged_bool(
            self, tmp_path, capsys, monkeypatch):
        """B3."""
        ws = _fresh(tmp_path)
        _append(ws, _BLOCKS)
        _healthy(monkeypatch)
        _, payload = _json(capsys, ws)
        checks = payload["checks"]
        assert checks["decisions_buffer"] is True
        assert type(checks["decisions_buffer_entries"]) is int
        assert type(checks["decisions_buffer_chars"]) is int
        assert checks["decisions_buffer_entries"] == _K
        assert checks["decisions_buffer_chars"] == len(_buffer_text(ws))
        assert "decisions_buffer_entries" not in payload
        assert "decisions_buffer_chars" not in payload


class TestSizeNeverMovesTheVerdict:
    def test_a_full_buffer_and_its_drained_twin_read_the_same_verdict(
            self, tmp_path, capsys, monkeypatch):
        """B4 — committed twins sharing one graph: all K buffered vs one buffered."""
        _healthy(monkeypatch)
        full = _fresh(tmp_path, "full")
        _append(full, _BLOCKS)
        _commit(full, _BLOCKS)

        drained = _fresh(tmp_path, "drained")
        first, *rest = [_entry(i) for i in range(_K)]
        _append(drained, first)
        archive_dir = MitosConfig(str(drained)).archive_dir
        os.makedirs(archive_dir, exist_ok=True)
        with open(os.path.join(archive_dir, "2026-Q3.md"), "w", encoding="utf-8") as f:
            f.write("".join(rest).lstrip("\n"))
        _commit(drained, _BLOCKS)

        rc_full, out_full = _text(capsys, full)
        rc_drained, out_drained = _text(capsys, drained)
        assert rc_full == rc_drained == 0
        assert _verdict(out_full) == _verdict(out_drained) == "READY ✓"

        _, a = _json(capsys, full)
        _, b = _json(capsys, drained)
        new = {"decisions_buffer_entries", "decisions_buffer_chars"}
        # In-row non-vacuity: the twins really sit on either side of the threshold.
        assert a["checks"]["decisions_buffer_entries"] > _THRESHOLD
        assert _THRESHOLD >= b["checks"]["decisions_buffer_entries"]
        for key in ("ready", "initialized"):
            assert a[key] == b[key]
        assert a["ready"] is True and a["checks"]["graph_unbuilt"] is False
        assert ({k: v for k, v in a["checks"].items() if k not in new}
                == {k: v for k, v in b["checks"].items() if k not in new})

    def test_an_absent_buffer_is_unmeasured_and_prints_no_follow_up(
            self, tmp_path, capsys, monkeypatch):
        """B5."""
        ws = _fresh(tmp_path)
        os.remove(os.path.join(str(ws), "decisions.md"))
        _healthy(monkeypatch)
        rc, payload = _json(capsys, ws)
        assert rc == 1
        assert payload["checks"]["decisions_buffer_entries"] is None
        assert payload["checks"]["decisions_buffer_chars"] is None
        rc, out = _text(capsys, ws)
        assert rc == 1
        row, follow = _row_and_follow(out)
        assert row.lstrip().startswith("✗") and "created by `mitos init`" in row
        assert not follow.startswith("      ")

    def test_an_unreadable_buffer_is_unmeasured_and_moves_nothing(
            self, tmp_path, capsys, monkeypatch):
        """B6 — the readable twin is sample-only, so both read READY (scout W1)."""
        _healthy(monkeypatch)
        readable = _fresh(tmp_path, "readable")
        broken = _fresh(tmp_path, "broken")
        path = os.path.join(str(broken), "decisions.md")
        with open(path, "rb") as f:
            seed = f.read()
        with open(path, "wb") as f:
            f.write(b"\xff\xfe" + seed)

        rc_ok, payload_ok = _json(capsys, readable)
        rc_bad, payload_bad = _json(capsys, broken)
        assert rc_ok == rc_bad
        assert payload_ok["ready"] == payload_bad["ready"]
        assert payload_bad["checks"]["decisions_buffer"] is True
        assert payload_bad["checks"]["decisions_buffer_entries"] is None
        assert payload_bad["checks"]["decisions_buffer_chars"] is None

        _, out_ok = _text(capsys, readable)
        rc_text, out_bad = _text(capsys, broken)
        assert rc_text == rc_ok
        assert _verdict(out_ok) == _verdict(out_bad)
        row, follow = _row_and_follow(out_bad)
        assert row.lstrip().startswith("✓")
        assert follow.startswith("      ")
        assert "could not be read" in follow and "UnicodeDecodeError" in follow

    def test_a_counting_defect_surfaces_rather_than_reading_as_unreadable(
            self, tmp_path, capsys, monkeypatch):
        """B6b — the catch covers the read, never the count."""
        ws = _fresh(tmp_path)
        _healthy(monkeypatch)

        def broken_count(text):
            raise ValueError("counting defect")

        monkeypatch.setattr(cli.settledness, "buffered_entries", broken_count)
        capsys.readouterr()
        with pytest.raises(ValueError, match="counting defect"):
            cli.cmd_status(str(ws))
        with pytest.raises(ValueError, match="counting defect"):
            cli.cmd_status(str(ws), as_json=True)
        out = capsys.readouterr().out
        assert "could not be read" not in out
        assert "decisions_buffer_entries" not in out


class TestTheFrame:
    def test_the_real_cli_emits_both_keys(self, tmp_path):
        """B7 — the one row that enters through `main()`."""
        base = str(tmp_path)
        env = {**os.environ, "MITOS_NO_UPDATE_CHECK": "1",
               "XDG_CONFIG_HOME": os.path.join(base, "xdg_config"),
               "XDG_CACHE_HOME": os.path.join(base, "xdg_cache"),
               "GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", "ANTHROPIC_API_KEY": "",
               "QDRANT_URL": "http://localhost:1"}
        ws = tmp_path / "ws"
        ws.mkdir()

        def run(*argv):
            return subprocess.run([sys.executable, "-m", "mitos.cli", *argv],
                                  cwd=str(ws), env=env, capture_output=True,
                                  text=True, timeout=300)

        assert run("init").returncode == 0
        _set_threshold(ws)
        _append(ws, _BLOCKS)
        done = run("-p", str(ws), "status", "--json")
        payload = json.loads(done.stdout)
        assert payload["checks"]["decisions_buffer_entries"] == _K
        assert payload["checks"]["decisions_buffer_chars"] == len(_buffer_text(ws))
        assert "Traceback" not in done.stderr


class TestTheFormatSpecRider:
    def test_an_absent_reference_copy_reads_as_a_fact_not_damage(
            self, tmp_path, capsys, monkeypatch):
        """B8."""
        _healthy(monkeypatch)
        ws = _fresh(tmp_path, "absent")
        os.remove(os.path.join(str(ws), "format-spec.md"))
        _, out = _text(capsys, ws)
        line = next(ln for ln in out.splitlines() if "format-spec.md" in ln)
        assert line.lstrip().startswith("—") and "✗" not in line
        assert "restore" not in line and "missing" not in line
        assert "optional reference" in line and "bundled" in line

        present = _fresh(tmp_path, "present")
        _, out = _text(capsys, present)
        line = next(ln for ln in out.splitlines() if "format-spec.md" in ln)
        assert line.lstrip().startswith("✓") and "→" not in line
