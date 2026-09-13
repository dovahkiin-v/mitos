"""CC-10's gate: rotate, then `mitos rebuild`, and the graph comes back identical.

A per-defect assertion is silent against a whole failure class this row is loud
against: an unbounded removal that took a second copy, a torn archive, a quarter
file the reader never looks in, a heading form the parser cannot read back, a
removal that shifted a neighbour's canonical core — or a batch written in an order
the reversing reader replays inverted. Each of those changes the rebuilt node id
set, so the id set is compared across all kinds and all states — the completeness
gate reads active cores only, and a superseded predecessor lost from the corpus
would be invisible to it.

The fixture carries the two dependency shapes that made the real corpus's round
trip red (fix brief, 2026-09-13), because a fixture with no citation between two
rotated entries cannot see an ordering bug at all:

* **Same file, different batches, with a kill edge between.** ``base-two`` rotates in
  batch 1; ``amender`` (amends it) and ``successor-two`` (supersedes it) rotate in
  batch 2 into the same file. Written oldest-first, the file replays as
  ``successor-two, amender, base-two``: both are quarantined, ``successor-two``
  retires the target first, and ``amender`` then fails ``dangling_edge``.
* **Across files.** ``base-one`` rotates into an earlier quarter; ``amender-one``
  (amends it) and ``successor-one`` (supersedes it) rotate into the later quarter in
  two further batches, so the later file's own order has to hold too.

The corpus is committed through parse→commit, so the row is keyless and
deterministic. The rebuild is the in-process `cmd_rebuild --json`, swap included, and
``swapped`` is asserted before the id set is — an identical id set read off a graph
the gate refused to swap is vacuous.
"""

import json
import os

from mitos.cli import cmd_rebuild
from mitos.config import MitosConfig
from mitos.cutover import _ARCHIVE_FILENAME_RE
from mitos.parser import parse_file_reversed
from mitos.rotation import RotationBlock, rotate
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


# Oldest first (commit order); the buffer holds them newest-first below its sentinel.
_ENTRIES = [
    ("base-one", _block("base-one", "The first base axiom.", scope=["core"])),
    ("base-two", _block("base-two", "The second base axiom.", scope=["core"])),
    ("multi-a", _block("multi-a", "A multi-scoped axiom.", scope=["zeta", "alpha"])),
    ("amender-one", _block("amender-one", "A refinement of the first base.",
                           scope=["core"], relations=[("Amends", "base-one")])),
    ("amender", _block("amender", "A refinement of the second base.",
                       scope=["core"], relations=[("Amends", "base-two")])),
    ("successor-two", _block("successor-two", "The axiom that replaced the second base.",
                             scope=["core"], relations=[("Supersedes", "base-two")])),
    ("successor-one", _block("successor-one", "The axiom that replaced the first base.",
                             scope=["core"], relations=[("Supersedes", "base-one")])),
    ("keeper", _block("keeper", "An axiom that stays in the buffer.", scope=["core"])),
]

# Each batch is a slice of commit order, named for the quarter it is rotated in, as
# sync names them: the quarters never decrease across batches.
_BATCHES = [
    (["base-one"], "2026-Q2.md"),
    (["base-two", "multi-a"], "2026-Q3.md"),
    (["amender-one", "amender", "successor-two"], "2026-Q3.md"),
    (["successor-one"], "2026-Q4.md"),
]


def _workspace(tmp_path):
    config = MitosConfig(str(tmp_path))
    os.makedirs(config.mitos_dir, exist_ok=True)
    os.makedirs(config.archive_dir)
    with open(os.path.join(config.archive_dir, "2026-Q2.md"), "w", encoding="utf-8") as fh:
        fh.write(_LEGACY)
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write(_SENTINEL + "\n\n" + "".join(text for _s, text in reversed(_ENTRIES)))

    store = GraphStore(config.db_path)
    failures = []
    for path in (os.path.join(config.archive_dir, "2026-Q2.md"), config.decisions_file):
        for entry in parse_file_reversed(path, "decision", failures):
            store.commit_parsed_entry(entry)
    assert failures == []
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
    # Non-vacuity of the dependency shapes: both amenders cite a target that a later
    # entry retires, and the amend committed first (the state the rebuild must recreate).
    assert store.get_node_state(store.get_node_by_slug("amender")["id"]) == "active"
    assert store.get_node_by_slug("base-two") is None, "base-two is superseded"

    with open(config.decisions_file, encoding="utf-8") as fh:
        lines = fh.readlines()
    parsed = {e.slug: e for e in parse_file_reversed(config.decisions_file, "decision", [])}

    def _rotation_block(slug, archive_name):
        entry = parsed[slug]
        return RotationBlock(slug, "".join(lines[entry.line_start - 1:entry.line_end]),
                             archive_name)

    lock = MitosSyncManager(config).lock
    written = []
    for batch, archive_name in _BATCHES:
        outcome = rotate(lock, config.decisions_file, config.archive_dir,
                         [_rotation_block(slug, archive_name) for slug in batch])
        assert [b.label for b in outcome.rotated] == batch
        written += [os.path.basename(p) for p in outcome.archive_paths]

    # Non-vacuity: three quarter files, one of them the legacy file, and a file
    # rewritten by two batches.
    archives = sorted(os.listdir(config.archive_dir))
    assert archives == ["2026-Q2.md", "2026-Q3.md", "2026-Q4.md"]
    assert all(_ARCHIVE_FILENAME_RE.match(name) for name in archives)
    assert written.count("2026-Q3.md") == 2

    q2 = os.path.join(config.archive_dir, "2026-Q2.md")
    with open(q2, encoding="utf-8") as fh:
        q2_text = fh.read()
    assert "## 2026-05-21 — legacy-dated — " in q2_text and "### base-one\n" in q2_text
    assert q2_text.index("### base-one") < q2_text.index("legacy-dated"), "newest on top"
    assert _slugs(q2) == ["legacy-dated", "base-one"]
    # The Q3 file replays in commit order: batch 1's blocks, then batch 2's.
    assert _slugs(os.path.join(config.archive_dir, "2026-Q3.md")) == [
        "base-two", "multi-a", "amender-one", "amender", "successor-two"]
    assert _slugs(config.decisions_file) == ["keeper"]

    assert cmd_rebuild(config, allow_drops=False, assume_yes=True, as_json=True) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["swapped"] is True, report
    assert report["gate_passed"] is True
    assert report["residual_casualties"] == [] and report["missing_cores"] == []

    rebuilt = GraphStore(config.db_path)
    assert {n["id"] for n in rebuilt.get_all_nodes()} == ids_before
    assert rebuilt.get_node_state(rebuilt.get_node_by_slug("amender")["id"]) == "active"
    assert rebuilt.get_node_state(rebuilt.get_node_by_slug("amender-one")["id"]) == "active"
