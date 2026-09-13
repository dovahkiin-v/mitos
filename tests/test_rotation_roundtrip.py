"""CC-10's gate: rotate, then `mitos rebuild`, and the graph comes back identical.

A per-defect assertion is silent against a whole failure class this row is loud
against: an unbounded removal that took a second copy, a torn archive, a quarter
file the reader never looks in, a heading form the parser cannot read back, or a
removal that shifted a neighbour's canonical core. Each of those changes the rebuilt
node id set, so the id set is compared across all kinds and all states — the
completeness gate reads active cores only, and a superseded predecessor lost from
the corpus would be invisible to it.

The corpus is committed through parse→commit, so the row is keyless and
deterministic. The rebuild is the in-process `cmd_rebuild --json`, swap included.
"""

import json
import os

from mitos.cli import cmd_rebuild
from mitos.config import MitosConfig
from mitos.cutover import _ARCHIVE_FILENAME_RE
from mitos.parser import parse_file_reversed
from mitos.rotation import RotationBlock, archive_name_for, rotate
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager

_SENTINEL = "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->"

_LEGACY = (
    "## 2026-05-21 — legacy-dated — Legacy Dated\n"
    "**Decided:** A decision archived in the legacy dated form.\n"
    "**Rejected:** Rewriting old archives into the new form.\n"
    "**Mechanisms:** archive\n"
    "**Scope:** core\n"
    "\n"
)

_Q2 = "2026-05-02T10:00:00+00:00"
_Q3 = "2026-08-01T00:00:00+00:00"
_Q4_PREVIOUS_YEAR = "2025-12-20T08:00:00+00:00"


def _block(slug, decided, *, scope, relations=()):
    lines = [
        f"### {slug}",
        "",
        f"**Decided:** {decided}",
        "**Rejected:** The obvious alternative, for a stated reason.",
        f"**Mechanisms:** {slug}-mechanism",
        f"**Scope:** {', '.join(scope)}",
    ]
    lines += [f"**{field}:** [{target}]" for field, target in relations]
    return "\n".join(lines) + "\n\n"


# Oldest first; the buffer holds them newest-first below its sentinel.
_ENTRIES = [
    ("base-one", _block("base-one", "The first base axiom.", scope=["core"]), _Q2),
    ("base-two", _block("base-two", "The second base axiom.", scope=["core"]), _Q2),
    ("multi-a", _block("multi-a", "A multi-scoped axiom.", scope=["zeta", "alpha"]), _Q2),
    ("multi-b", _block("multi-b", "Another multi-scoped axiom.", scope=["ops", "core"]),
     _Q4_PREVIOUS_YEAR),
    ("successor", _block("successor", "The axiom that replaced the first base.",
                         scope=["core"], relations=[("Supersedes", "base-one")]),
     _Q4_PREVIOUS_YEAR),
    ("amender", _block("amender", "A refinement of the second base.",
                       scope=["core"], relations=[("Amends", "base-two")]), _Q3),
    ("keeper", _block("keeper", "An axiom that stays in the buffer.", scope=["core"]), _Q3),
]


def _workspace(tmp_path):
    config = MitosConfig(str(tmp_path))
    os.makedirs(config.mitos_dir, exist_ok=True)
    os.makedirs(config.archive_dir)
    with open(os.path.join(config.archive_dir, "2026-Q2.md"), "w", encoding="utf-8") as fh:
        fh.write(_LEGACY)
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write(_SENTINEL + "\n\n" + "".join(text for _s, text, _c in reversed(_ENTRIES)))

    store = GraphStore(config.db_path)
    failures = []
    for path in (os.path.join(config.archive_dir, "2026-Q2.md"), config.decisions_file):
        for entry in parse_file_reversed(path, "decision", failures):
            store.commit_parsed_entry(entry)
    assert failures == []

    # Every state: `base-one` is superseded, so the active-view slug lookup misses it.
    ids = {n["slug"]: n["id"] for n in store.get_all_nodes()}
    conn = store._get_connection()
    try:
        with conn:
            for slug, _text, stamp in _ENTRIES:
                conn.execute("UPDATE nodes SET created_at = ? WHERE id = ?", (stamp, ids[slug]))
    finally:
        conn.close()
    return config, store


def _slugs(path):
    failures = []
    slugs = [e.slug for e in parse_file_reversed(path, "decision", failures)]
    assert failures == []
    return slugs


def test_rotate_then_rebuild_returns_the_identical_node_id_set(tmp_path, capsys):
    config, store = _workspace(tmp_path)
    ids_before = {n["id"] for n in store.get_all_nodes()}
    assert len(ids_before) == len(_ENTRIES) + 1

    with open(config.decisions_file, encoding="utf-8") as fh:
        lines = fh.readlines()
    parsed = {e.slug: e for e in parse_file_reversed(config.decisions_file, "decision", [])}
    ids = {n["slug"]: n["id"] for n in store.get_all_nodes()}
    stamps = store.created_at_for(list(ids.values()))

    def _rotation_block(slug):
        entry = parsed[slug]
        raw = "".join(lines[entry.line_start - 1:entry.line_end])
        return RotationBlock(slug, raw, archive_name_for(stamps[ids[slug]]))

    lock = MitosSyncManager(config).lock
    batches = [["base-one", "base-two", "multi-b"], ["multi-a", "successor", "amender"]]
    written = []
    for batch in batches:
        outcome = rotate(lock, config.decisions_file, config.archive_dir,
                         [_rotation_block(slug) for slug in batch])
        assert [b.label for b in outcome.rotated] == batch
        written += [os.path.basename(p) for p in outcome.archive_paths]

    # Non-vacuity: three quarter files, one of them the legacy file, and a file
    # rewritten by both batches.
    archives = sorted(os.listdir(config.archive_dir))
    assert archives == ["2025-Q4.md", "2026-Q2.md", "2026-Q3.md"]
    assert all(_ARCHIVE_FILENAME_RE.match(name) for name in archives)
    assert written.count("2026-Q2.md") == 2 and written.count("2025-Q4.md") == 2

    q2 = os.path.join(config.archive_dir, "2026-Q2.md")
    with open(q2, encoding="utf-8") as fh:
        q2_text = fh.read()
    assert "## 2026-05-21 — legacy-dated — " in q2_text and "### base-one\n" in q2_text
    assert sorted(_slugs(q2)) == ["base-one", "base-two", "legacy-dated", "multi-a"]
    assert _slugs(config.decisions_file) == ["keeper"]

    assert cmd_rebuild(config, allow_drops=False, assume_yes=True, as_json=True) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["swapped"] is True
    assert report["gate_passed"] is True
    assert report["residual_casualties"] == [] and report["missing_cores"] == []

    rebuilt = GraphStore(config.db_path)
    assert {n["id"] for n in rebuilt.get_all_nodes()} == ids_before
    # Carry-forward keeps each stamp, so every rotated entry still names its own file.
    rebuilt_stamps = rebuilt.created_at_for(list(ids.values()))
    for batch in batches:
        for slug in batch:
            name = archive_name_for(rebuilt_stamps[ids[slug]])
            assert slug in _slugs(os.path.join(config.archive_dir, name))
