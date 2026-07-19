"""The 2026-07-19 polish pass: code-span marker exemption, status glyph, env precedence.

Three small "tool never misleads" fixes:
- inline-code spans are exempt from structural-marker scanning (parser + sync
  guard move together, so a quoted ``[NOTE: …]`` or BEGIN-ENTRIES sentinel is
  recordable prose);
- ``mitos status`` never prints ``✗`` for the non-blocking format-spec.md line;
- ``QDRANT_URL`` env overrides a toml-pinned ``qdrant_url`` (documented order).
"""

import os
import subprocess
import sys

from mitos.parser import mask_inline_code
from mitos.sync import _contains_structural_token


class TestMaskInlineCode:
    def test_span_contents_blanked_same_length(self):
        line = "see `[NOTE: quoted]` here"
        masked = mask_inline_code(line)
        assert len(masked) == len(line)
        assert "[NOTE:" not in masked
        assert masked.startswith("see ") and masked.endswith(" here")

    def test_unquoted_marker_untouched(self):
        assert "[NOTE:" in mask_inline_code("a real [NOTE: x] marker")

    def test_no_span_is_identity(self):
        assert mask_inline_code("plain prose line") == "plain prose line"

    def test_unclosed_backtick_untouched(self):
        line = "a stray ` backtick with [NOTE: x]"
        assert mask_inline_code(line) == line


class TestGuardCodeSpanExemption:
    def test_quoted_markers_pass_the_guard(self):
        assert not _contains_structural_token(
            "the parser anchors on the `BEGIN ENTRIES` sentinel"
        )
        assert not _contains_structural_token("wrap it as `[NOTE: like this]`")
        assert not _contains_structural_token("a quoted `[DECISION_PARKED: topic]`")

    def test_bare_markers_still_rejected(self):
        assert _contains_structural_token("contains BEGIN ENTRIES bare")
        assert _contains_structural_token("[NOTE: siphons me]")
        assert _contains_structural_token("## a column-zero heading")

    def test_record_accepts_quoted_marker(self, tmp_path):
        env = {**os.environ, "GEMINI_API_KEY": "", "GOOGLE_API_KEY": "",
               "QDRANT_URL": "http://localhost:1"}

        def run(*args):
            return subprocess.run([sys.executable, "-m", "mitos.cli", *args],
                                  capture_output=True, text=True,
                                  cwd=str(tmp_path), env=env)

        run("init")
        rec = run("record",
                  "The parser anchors on the `BEGIN ENTRIES` sentinel comment.",
                  "--rejected", "scanning for `[NOTE: markers]` naively",
                  "--slug", "quoted-marker-prose")
        assert rec.returncode == 0, rec.stderr
        # And the entry round-trips through a sync-driven reparse.
        sync = run("sync", "--yes")
        assert sync.returncode == 0, sync.stderr


class TestStatusGlyph:
    def test_missing_format_spec_is_neutral_not_cross(self, tmp_path):
        env = {**os.environ, "GEMINI_API_KEY": "", "GOOGLE_API_KEY": "",
               "QDRANT_URL": "http://localhost:1"}

        def run(*args):
            return subprocess.run([sys.executable, "-m", "mitos.cli", *args],
                                  capture_output=True, text=True,
                                  cwd=str(tmp_path), env=env)

        run("init")
        os.remove(os.path.join(str(tmp_path), "format-spec.md"))
        out = run("status").stdout
        spec_line = next(l for l in out.splitlines() if "format-spec.md" in l)
        assert "✗" not in spec_line
        assert "—" in spec_line and "non-destructive" in spec_line


class TestQdrantUrlPrecedence:
    def test_env_wins_over_toml(self, tmp_path, monkeypatch):
        from mitos.config import MitosConfig
        mitos_dir = tmp_path / ".mitos"
        mitos_dir.mkdir()
        (mitos_dir / "config.toml").write_text(
            'qdrant_url = "http://toml-pinned:7333"\n', encoding="utf-8"
        )
        monkeypatch.setenv("QDRANT_URL", "http://env-wins:9999")
        assert MitosConfig(workspace_dir=str(tmp_path)).qdrant_url == "http://env-wins:9999"

    def test_toml_applies_without_env(self, tmp_path, monkeypatch):
        from mitos.config import MitosConfig
        mitos_dir = tmp_path / ".mitos"
        mitos_dir.mkdir()
        (mitos_dir / "config.toml").write_text(
            'qdrant_url = "http://toml-pinned:7333"\n', encoding="utf-8"
        )
        monkeypatch.delenv("QDRANT_URL", raising=False)
        assert MitosConfig(workspace_dir=str(tmp_path)).qdrant_url == "http://toml-pinned:7333"
