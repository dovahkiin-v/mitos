"""Pins that one sync opens the decisions snapshot a fixed number of times, not once per entry.

Re-reading the whole snapshot for every entry made a large buffer's sync quadratic
(measured 2026-09-12: 47% of a 3,704-entry cold sync). Rotation no longer slices
the snapshot at all — it reads its blocks from the live buffer under the lock
(surface-entropy 3b) — so the snapshot is opened once, to parse.
"""

import builtins
import os
import shutil
import tempfile
from unittest.mock import patch

from mitos.config import MitosConfig
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager


def test_a_multi_entry_sync_reads_the_snapshot_once() -> None:
    tmpdir = tempfile.mkdtemp()
    try:
        config = MitosConfig(tmpdir)
        os.makedirs(os.path.join(tmpdir, ".mitos"), exist_ok=True)
        entries = "".join(
            f"### snapshot-read-once-{i}\n\n"
            f"**Decided:** Entry number {i} commits from a single snapshot read.\n"
            "**Rejected:** Re-reading the snapshot per entry.\n"
            "**Mechanisms:** python\n"
            "**Scope:** sync\n\n"
            for i in range(5)
        )
        with open(config.decisions_file, "w", encoding="utf-8") as f:
            f.write(
                "# Decisions\n"
                "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->\n\n"
                + entries
            )
        config.env["GEMINI_API_KEY"] = "mock_key"

        snapshot_reads = []
        real_open = builtins.open

        def counting_open(file, mode="r", *args, **kwargs):
            if str(file).endswith("sync_snapshot.md") and "r" in mode:
                snapshot_reads.append(file)
            return real_open(file, mode, *args, **kwargs)

        with patch("google.genai.Client"), patch("builtins.open", counting_open):
            MitosSyncManager(config).perform_sync(auto_accept=True)

        assert len(GraphStore(config.db_path).get_all_nodes()) == 5
        # One open, independent of entry count: the parse. The raw-block slice left
        # with snapshot-based rotation; the per-entry re-read once made this 1 + 5.
        assert len(snapshot_reads) == 1, f"snapshot opened {len(snapshot_reads)} times"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
