"""Tests for `mitos restore-source` — re-materializing a graph-only node's block.

The verb writes into the gold source, so its whole design is refusal-first: it must
never write a block it cannot prove round-trips, and never a splice that disturbs a
neighbour. Each test below corresponds to a way a plausible implementation writes
something wrong and looks like it worked.
"""

import json
import os
import sys

import pytest

from mitos.cli import cmd_restore_source
from mitos.config import MitosConfig
from mitos.parser import parse_entry_stream
from mitos.restore import (
    RestoreError,
    render_source_block,
    verify_block_in_isolation,
    verify_whole_buffer,
)
from mitos.store import GraphStore

_SENTINEL = "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->"


def _block(slug, decided, *, rejected="n/a", mechanisms=("m1",), scope=(),
           context=None, invalidates_if=None, source=None, cites=None):
    lines = [f"### {slug}", "", f"**Decided:** {decided}", f"**Rejected:** {rejected}"]
    if mechanisms:
        lines.append(f"**Mechanisms:** {', '.join(mechanisms)}")
    if scope:
        lines.append(f"**Scope:** {', '.join(scope)}")
    if invalidates_if:
        lines.append(f"**Invalidates-If:** {invalidates_if}")
    if context:
        lines.append(f"**Context:** {context}")
    if source:
        lines.append(f"**Source:** {source}")
    if cites:
        lines.append(f"**Cites:** [{cites}]")
    return "\n".join(lines)


def _workspace(tmp_path, committed, *, kept=()):
    """Commits every block, then leaves only ``kept`` in the buffer (the rest orphan)."""
    config = MitosConfig(str(tmp_path))
    os.makedirs(config.mitos_dir, exist_ok=True)
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write(_SENTINEL + "\n\n" + "\n\n".join(committed) + "\n")

    store = GraphStore(config.db_path)
    for block in committed:
        for entry in parse_entry_stream(block, "decision"):
            store.commit_parsed_entry(entry)

    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        body = ("\n\n".join(kept) + "\n") if kept else ""
        fh.write(_SENTINEL + "\n\n" + body)
    return config


def _buffer(config):
    with open(config.decisions_file, encoding="utf-8") as fh:
        return fh.read()


# --- the happy path --------------------------------------------------------------

def test_restoring_a_node_makes_the_corpus_reconstruct_it_again(tmp_path, capsys):
    """The point of the verb: a restored block is rebuild source again.

    Asserted through `mitos rebuild --json`'s own gate rather than by inspecting the
    markdown, because that gate is what Phase 5 actually reads — and it is the only
    check that proves the block is not merely present but *replayable*.
    """
    from mitos.cli import cmd_rebuild

    kept = _block("kept", "Kept axiom.", mechanisms=("m1",))
    orphan = _block("orphaned", "Orphan axiom.", mechanisms=("m2",))
    config = _workspace(tmp_path, [kept, orphan], kept=[kept])

    before = json.loads(_rebuild_json(config, capsys))
    assert "orphaned" in [row["slug"] for row in before["missing_cores"]]
    assert before["gate_passed"] is False

    assert cmd_restore_source(config, all_graph_only=True) == 0
    capsys.readouterr()

    after = json.loads(_rebuild_json(config, capsys))
    assert [row["slug"] for row in after["missing_cores"]] == []
    assert after["gate_passed"] is True


def _rebuild_json(config, capsys):
    from mitos.cli import cmd_rebuild

    cmd_rebuild(config, allow_drops=False, assume_yes=True, as_json=True)
    return capsys.readouterr().out


def test_dry_run_writes_nothing(tmp_path, capsys):
    """`--dry-run` must be a pure read — it is how an operator reviews 43 blocks."""
    kept = _block("kept", "Kept axiom.")
    orphan = _block("orphaned", "Orphan axiom.", mechanisms=("m2",))
    config = _workspace(tmp_path, [kept, orphan], kept=[kept])
    before = _buffer(config)

    assert cmd_restore_source(config, all_graph_only=True, dry_run=True) == 0
    assert _buffer(config) == before
    assert "orphaned" in capsys.readouterr().out


def test_a_restored_node_with_edges_keeps_them_and_hashes_the_same(tmp_path, capsys):
    """Relationship lines are regenerated from the STORED edges, not re-derived.

    A restore that dropped them would leave the entry declaring nothing — and because
    edges mirror declaratively, the next commit of that entry would DELETE the real
    edges from the graph, turning a repair into data loss.
    """
    target = _block("target", "Target axiom.", mechanisms=("m1",))
    citer = _block("citer", "Citer axiom.", mechanisms=("m2",), cites="target")
    config = _workspace(tmp_path, [target, citer], kept=[target])

    assert cmd_restore_source(config, slug="citer") == 0
    capsys.readouterr()

    restored = [e for e in parse_entry_stream(_buffer(config), "decision")
                if e.slug == "citer"][0]
    assert restored.cites == ["[target]"]

    store = GraphStore(config.db_path, read_only=True)
    stored_id = [n["id"] for n in store.get_all_nodes() if n["slug"] == "citer"][0]
    from mitos.identity import compute_node_id
    assert compute_node_id(
        kind=restored.kind, axiom=restored.axiom,
        mechanism_refs=restored.mechanisms, topic=restored.topic,
        questions_raised=restored.questions_raised,
    ) == stored_id


def test_a_non_user_source_round_trips(tmp_path, capsys):
    """`**Source:**` must be re-emitted when it is not the default.

    Omitting it is silent provenance loss: a later rebuild replays the entry, the
    parser defaults the absent field to `user`, and a machine-authored decision is
    permanently restamped as human-confirmed.
    """
    kept = _block("kept", "Kept axiom.")
    imported = _block("imported-one", "Imported axiom.", mechanisms=("m2",),
                      source="import_llm")
    config = _workspace(tmp_path, [kept, imported], kept=[kept])

    assert cmd_restore_source(config, slug="imported-one") == 0
    capsys.readouterr()

    restored = [e for e in parse_entry_stream(_buffer(config), "decision")
                if e.slug == "imported-one"][0]
    assert restored.source == "import_llm"


def test_a_user_source_is_not_emitted(tmp_path, capsys):
    """Absent means `user`, so emitting `**Source:** user` would be noise in the corpus."""
    kept = _block("kept", "Kept axiom.")
    orphan = _block("orphaned", "Orphan axiom.", mechanisms=("m2",))
    config = _workspace(tmp_path, [kept, orphan], kept=[kept])

    assert cmd_restore_source(config, slug="orphaned") == 0
    capsys.readouterr()
    assert "**Source:**" not in _buffer(config)


def test_commentary_fields_survive_the_round_trip(tmp_path, capsys):
    """Every optional commentary field is re-emitted, including multi-line prose.

    Multi-line `rejected_paths` is the realistic case — 12 of the live corpus's 43
    orphans carry one — and it is exactly what a naive single-line renderer truncates.
    """
    kept = _block("kept", "Kept axiom.")
    rich = _block(
        "rich-one", "Rich axiom.", mechanisms=("m2",), scope=("alpha", "beta"),
        rejected="First rejected path.\nSecond rejected path, on its own line.",
        invalidates_if="A better substrate appears.",
        context="Background prose.",
    )
    config = _workspace(tmp_path, [kept, rich], kept=[kept])

    assert cmd_restore_source(config, slug="rich-one") == 0
    capsys.readouterr()

    restored = [e for e in parse_entry_stream(_buffer(config), "decision")
                if e.slug == "rich-one"][0]
    assert restored.rejected_paths == (
        "First rejected path.\nSecond rejected path, on its own line."
    )
    assert restored.invalidates_if == "A better substrate appears."
    assert restored.context == "Background prose."
    assert sorted(restored.scope) == ["alpha", "beta"]


# --- refusal ---------------------------------------------------------------------

def test_a_block_whose_commentary_contains_a_header_line_is_refused():
    """An id-only check would pass this while minting a PHANTOM entry.

    The node id is computed from the canonical core alone, so a `### ` line hiding in
    the commentary hashes correctly AND splits into a second entry — seeding instant
    divergence that a later reconcile would "heal" backwards into the graph.
    """
    node = {
        "id": "deadbeef", "slug": "sneaky", "core_axiom": "An axiom.",
        "mechanisms": ["m1"], "scope": [],
        "rejected_paths": "First reason.\n### injected-entry\n\n**Decided:** Phantom.",
        "invalidates_if": None, "context": None, "source": "user",
    }
    block = render_source_block(node, [])
    with pytest.raises(RestoreError, match="parses as 2 entries|does not parse"):
        verify_block_in_isolation(block, node)


def test_a_node_missing_a_required_field_is_refused():
    """M5 makes `**Rejected:**` mandatory, so a block without it would not parse back.

    Refusing here trades one broken state for nothing, rather than for another.
    """
    node = {"id": "x", "slug": "no-rejected", "core_axiom": "An axiom.",
            "mechanisms": ["m1"], "scope": [], "rejected_paths": "",
            "invalidates_if": None, "context": None, "source": "user"}
    with pytest.raises(RestoreError, match="rejected_paths"):
        render_source_block(node, [])

    node["rejected_paths"] = "A reason."
    node["core_axiom"] = ""
    with pytest.raises(RestoreError, match="axiom"):
        render_source_block(node, [])


def test_a_splice_that_would_alter_a_neighbour_is_refused():
    """Whole-buffer verification — isolation cannot prove neighbour safety.

    Continuation-line bleed crosses entry boundaries; the retired `mark` mode is a
    live demonstration, where one stray line shifted the canonical core of the entry
    ABOVE it. One extra parse closes the seam class permanently.
    """
    before = (_SENTINEL + "\n\n" + _block("neighbour", "Neighbour axiom.") + "\n")
    # A splice that swallows the neighbour's `**Mechanisms:**` line into its own field.
    after = (_SENTINEL + "\n\n### spliced\n\n**Decided:** Spliced axiom.\n"
             "**Rejected:** A reason.\n\n"
             + _block("neighbour", "Neighbour axiom.", mechanisms=()) + "\n")
    with pytest.raises(RestoreError, match="altered or removed the pre-existing entry"):
        verify_whole_buffer(before, after, added=1)


def test_a_splice_that_changes_the_entry_count_is_refused():
    """A count mismatch means the splice merged or split an entry, whatever the cause."""
    before = _SENTINEL + "\n\n" + _block("one", "One axiom.") + "\n"
    after = before  # added=1 promised, zero delivered
    with pytest.raises(RestoreError, match="changed the entry count"):
        verify_whole_buffer(before, after, added=1)


def test_verify_whole_buffer_accepts_an_honest_splice():
    """The guard must not refuse the case it exists to permit."""
    before = _SENTINEL + "\n\n" + _block("one", "One axiom.") + "\n"
    after = (_SENTINEL + "\n\n" + _block("two", "Two axiom.", mechanisms=("m2",))
             + "\n\n" + _block("one", "One axiom.") + "\n")
    verify_whole_buffer(before, after, added=1)


# --- CLI surface -----------------------------------------------------------------

def test_restoring_a_non_orphan_is_refused(tmp_path, capsys):
    """A node that already has a block is not a repair target — say so, write nothing."""
    kept = _block("kept", "Kept axiom.")
    config = _workspace(tmp_path, [kept], kept=[kept])
    before = _buffer(config)

    assert cmd_restore_source(config, slug="kept", as_json=True) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "not_graph_only"
    assert _buffer(config) == before


def test_exactly_one_target_is_required(tmp_path, capsys):
    """Neither or both is a dead end with a named reason, not a silent default."""
    config = _workspace(tmp_path, [_block("kept", "Kept axiom.")], kept=[])
    assert cmd_restore_source(config, as_json=True) == 1
    assert json.loads(capsys.readouterr().out)["code"] == "ambiguous_target"

    assert cmd_restore_source(config, slug="kept", all_graph_only=True, as_json=True) == 1
    assert json.loads(capsys.readouterr().out)["code"] == "ambiguous_target"


def test_every_refusal_names_the_corpus_it_refused_for(tmp_path, capsys):
    """The refusals echo too — this verb rewrites the user-authored gold source.

    A mis-aimed `--all-graph-only` re-materializes THIS project's graph into
    ANOTHER project's `decisions.md`, in bulk, so which corpus a refusal is
    speaking about is not decoration. Under the locus rule (a response echoes iff
    it is emitted inside a `cmd_*` handler) `ambiguous_target` qualifies — but
    `--slug`/`--all-graph-only` is a *required mutually-exclusive* argparse group,
    so argparse refuses first and that branch is reachable **only** by calling the
    handler directly. Hence this row is a direct call, not a `main()`-driven one:
    driven through `main()` it would exit 2 at argparse with nothing on stdout and
    prove nothing.
    """
    config = _workspace(tmp_path, [_block("kept", "Kept axiom.")], kept=[])

    assert cmd_restore_source(config) == 1          # neither target — text mode
    err = capsys.readouterr().err
    assert f"corpus: {config.project}" in err
    assert config.qdrant_collection in err and config.workspace_dir in err

    assert cmd_restore_source(config, as_json=True) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["project"] == config.project
    assert payload["collection"] == config.qdrant_collection
    assert payload["workspace"] == config.workspace_dir


def test_a_clean_corpus_is_a_quiet_success(tmp_path, capsys):
    """Zero orphans is healthy, not an error — empty states are first-class."""
    kept = _block("kept", "Kept axiom.")
    config = _workspace(tmp_path, [kept], kept=[kept])

    assert cmd_restore_source(config, all_graph_only=True, as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["restored"] == [] and payload["written"] is False


def test_the_verb_routes_through_the_cli(monkeypatch, tmp_path):
    """`mitos restore-source` is registered and threads its flags."""
    from unittest.mock import patch
    from mitos.cli import main

    monkeypatch.setattr(sys, "argv",
                        ["mitos", "restore-source", "--all-graph-only", "--dry-run", "--json"])
    with patch("mitos.cli.cmd_restore_source", return_value=0) as mock:
        with pytest.raises(SystemExit):
            main()
    kwargs = mock.call_args.kwargs
    assert kwargs["all_graph_only"] is True
    assert kwargs["dry_run"] is True and kwargs["as_json"] is True
    assert kwargs["slug"] is None


def test_slug_and_all_graph_only_are_mutually_exclusive_in_the_grammar():
    """argparse refuses both spellings before any workspace is touched."""
    from mitos.cli import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["restore-source", "--slug", "s", "--all-graph-only"])
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["restore-source"])  # required=True


def test_a_splice_that_rewrites_the_FIRST_of_two_same_slug_entries_is_refused():
    """Duplicate slugs must not create a blind spot.

    A slug-keyed dict silently keeps only the LAST entry per slug, so a splice that
    rewrote the first of two `dup` entries passed every check — the count matched and
    the surviving fingerprint matched. A hand-pasted duplicate slug is precisely the
    untrusted input this whole-buffer pass exists to survive (P13).
    """
    dup_a = _block("dup", "First duplicate axiom.", mechanisms=("m1",))
    dup_b = _block("dup", "Second duplicate axiom.", mechanisms=("m2",))
    before = _SENTINEL + "\n\n" + dup_a + "\n\n" + dup_b + "\n"
    tampered_a = _block("dup", "TAMPERED first axiom.", mechanisms=("m1",))
    after = (_SENTINEL + "\n\n" + _block("added", "Added axiom.", mechanisms=("m3",))
             + "\n\n" + tampered_a + "\n\n" + dup_b + "\n")
    with pytest.raises(RestoreError, match="altered or removed the pre-existing entry"):
        verify_whole_buffer(before, after, added=1)


def test_two_same_slug_entries_survive_an_honest_splice():
    """The multiset comparison must not refuse a legitimate splice past a duplicate."""
    dup_a = _block("dup", "First duplicate axiom.", mechanisms=("m1",))
    dup_b = _block("dup", "Second duplicate axiom.", mechanisms=("m2",))
    before = _SENTINEL + "\n\n" + dup_a + "\n\n" + dup_b + "\n"
    after = (_SENTINEL + "\n\n" + _block("added", "Added axiom.", mechanisms=("m3",))
             + "\n\n" + dup_a + "\n\n" + dup_b + "\n")
    verify_whole_buffer(before, after, added=1)


# --- the buffer-surgery primitive (P20's binding output) --------------------------
#
# `splice_buffer` exists so a future `amend-commentary` verb consumes a proven seam
# rather than re-implementing `record_decision_entry`'s discipline. Two implementations
# of a sacred contract can drift, so its properties are pinned directly here rather
# than only transitively through `restore-source`.

def _manager(tmp_path):
    from mitos.sync import MitosSyncManager

    config = MitosConfig(str(tmp_path))
    os.makedirs(config.mitos_dir, exist_ok=True)
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write(_SENTINEL + "\n\n" + _block("existing", "Existing axiom.") + "\n")
    manager = MitosSyncManager(config)
    # Run one identity splice so `auto_heal_decisions_file` has already restored the
    # canonical header. That write is idempotent and deliberately OUTSIDE the rollback
    # (the transform needs the healed marker to splice against), so letting it happen
    # first is what makes "the rollback is byte-for-byte" the property under test rather
    # than "auto-heal never writes".
    manager.splice_buffer(lambda original: original)
    return config, manager


def test_splice_buffer_writes_the_transformed_text(tmp_path):
    """The happy path: the transform's output is what lands on disk."""
    config, manager = _manager(tmp_path)
    written = manager.splice_buffer(lambda original: original + "\n<!-- tail -->\n")
    assert _buffer(config) == written
    assert written.endswith("<!-- tail -->\n")


def test_splice_buffer_rolls_back_byte_for_byte_when_verification_fails(tmp_path):
    """An `after_write` raise restores the file EXACTLY — the sacred rollback property.

    This is the whole reason the write is allowed to happen before the fidelity check:
    a splice that fails verification must leave no trace, not a "mostly fine" buffer.
    """
    config, manager = _manager(tmp_path)
    before = _buffer(config)

    def _reject(_after_text):
        raise RestoreError("refused by the fidelity check")

    with pytest.raises(RestoreError):
        manager.splice_buffer(lambda original: "TOTALLY DIFFERENT CONTENT\n",
                              after_write=_reject)
    assert _buffer(config) == before, "the rollback must be byte-for-byte"


def test_splice_buffer_writes_nothing_when_the_transform_raises(tmp_path):
    """A transform raise never reaches the write, so there is nothing to roll back."""
    config, manager = _manager(tmp_path)
    before = _buffer(config)

    def _explode(_original):
        raise ValueError("cannot compute the splice")

    with pytest.raises(ValueError):
        manager.splice_buffer(_explode)
    assert _buffer(config) == before


def test_splice_buffer_holds_the_same_lock_record_and_sync_use(tmp_path):
    """It must serialize against `sync`/`record`, not run beside them.

    Asserted on the lock PATH rather than by racing processes: the property that
    matters is that all three writers contend for one lock, and a second lock file
    would give the appearance of mutual exclusion with none of it.
    """
    config, manager = _manager(tmp_path)
    assert manager.lock_path == config.decisions_file + ".lock"

    from filelock import FileLock, Timeout

    manager.lock.timeout = 0.2
    with FileLock(manager.lock_path, timeout=5):
        with pytest.raises(Timeout):
            manager.splice_buffer(lambda original: original + "x")


def test_splice_buffer_reports_loudly_when_the_rollback_itself_fails(tmp_path):
    """The one case silence would be unforgivable: the file is in an unknown state.

    A rollback failure means the buffer may hold a partial edit, so the error has to
    name that rather than surfacing the original exception as if nothing was written.
    """
    from unittest.mock import patch
    from mitos.errors import MitosError

    config, manager = _manager(tmp_path)
    real_open = open
    calls = {"n": 0}

    def _fail_on_the_rollback_write(*args, **kwargs):
        # 1st write = the splice, 2nd write = the rollback.
        if len(args) > 1 and args[1] == "w":
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("disk went away")
        return real_open(*args, **kwargs)

    def _reject(_after_text):
        raise RestoreError("refused")

    with patch("builtins.open", side_effect=_fail_on_the_rollback_write):
        with pytest.raises(MitosError, match="could not be rolled back"):
            manager.splice_buffer(lambda original: "NEW\n", after_write=_reject)


def test_graph_fingerprint_moves_with_every_write_path(tmp_path):
    """The cache key's graph half must change on a commit, an edit, and a wipe."""
    config = MitosConfig(str(tmp_path))
    os.makedirs(config.mitos_dir, exist_ok=True)
    store = GraphStore(config.db_path)
    assert store.graph_fingerprint() == (0, ""), "an empty graph is (0, '')"

    for entry in parse_entry_stream(_block("alpha", "Alpha axiom."), "decision"):
        store.commit_parsed_entry(entry)
    after_first = store.graph_fingerprint()
    assert after_first[0] == 1 and after_first[1] != ""

    for entry in parse_entry_stream(_block("beta", "Beta axiom.", mechanisms=("m2",)),
                                    "decision"):
        store.commit_parsed_entry(entry)
    after_second = store.graph_fingerprint()
    assert after_second[0] == 2, "a new node changes the count"

    # A commentary edit ticks `updated_at` without changing the count — the case a
    # count-only fingerprint would miss entirely.
    with store._get_connection() as conn:
        conn.execute("UPDATE nodes SET updated_at = ? WHERE slug = 'beta'",
                     ("2026-12-31T00:00:00.000000+00:00",))
    after_edit = store.graph_fingerprint()
    assert after_edit[0] == 2 and after_edit[1] != after_second[1]


# --- emit ORDER, not just content (found running Phase 5 on the live corpus) -------

def test_restored_blocks_replay_legally_when_one_supersedes_another(tmp_path, capsys):
    """A restored SET must be replayable, not merely individually faithful.

    Every citation resolves against the ACTIVE view (`store.py`'s
    `STORE_DANGLING_EDGE`: "a {edge_type} edge must target an active entry"), which
    makes it a COMMIT-TIME rule rather than an invariant on the final state — a
    completed supersession permanently points at a retired node, and that is fine
    because the target was active at the moment the citer committed.

    So emit order decides whether the corpus replays. Restoring in the detector's
    report order (actives alphabetically, then retireds) can put a supersession BEFORE
    the entry that amends its victim: the target is retired early, the amend is
    rejected, and `rebuild` refuses. Measured on the live corpus: 24 missing cores and
    31 casualties from three such roots, all of which vanished when the set was emitted
    in the order the nodes were originally committed.

    That order is `rowid` — provably legal, because it is the sequence in which these
    commits actually SUCCEEDED in this graph, so each one's constraints held in turn.
    """
    from mitos.cli import cmd_rebuild

    older = _block("older-decision", "The original axiom.", mechanisms=("m1",))
    amender = _block("amends-the-older", "An axiom that amends the original.",
                     mechanisms=("m2",))
    # Authored LAST, and it retires `older-decision`.
    superseder = _block("supersedes-the-older", "The replacement axiom.",
                        mechanisms=("m3",))

    config = MitosConfig(str(tmp_path))
    os.makedirs(config.mitos_dir, exist_ok=True)
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write(_SENTINEL + "\n\n")
    store = GraphStore(config.db_path)

    # Commit in true authoring order, so the amend lands while its target is active.
    for block, extra in ((older, ""),
                         (amender, "**Amends:** [older-decision]\n"),
                         (superseder, "**Supersedes:** [older-decision]\n")):
        for entry in parse_entry_stream(block + "\n" + extra, "decision"):
            store.commit_parsed_entry(entry)

    # Now orphan all three — this is the state the live corpus was in.
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write(_SENTINEL + "\n\n")

    assert cmd_restore_source(config, all_graph_only=True) == 0
    capsys.readouterr()

    cmd_rebuild(config, allow_drops=False, assume_yes=True, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["residual_casualties"] == [], payload["residual_casualties"]
    assert payload["missing_cores"] == [], payload["missing_cores"]
    assert payload["gate_passed"] is True, "the restored set must replay legally"


def test_restore_emits_in_commit_order(tmp_path, capsys):
    """The buffer is newest-first, so the OLDEST-committed block must land lowest.

    Asserted on position rather than only on the gate, because the gate can pass for
    the wrong reason on a corpus with no interdependencies — and this is the property
    that makes it pass for the right one.
    """
    config = MitosConfig(str(tmp_path))
    os.makedirs(config.mitos_dir, exist_ok=True)
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write(_SENTINEL + "\n\n")
    store = GraphStore(config.db_path)
    # Commit in an order that is NOT alphabetical, so report order cannot pass by luck.
    for slug in ("zulu-first", "alpha-second", "mike-third"):
        for entry in parse_entry_stream(_block(slug, f"Axiom for {slug}.",
                                               mechanisms=(slug[:4],)), "decision"):
            store.commit_parsed_entry(entry)
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write(_SENTINEL + "\n\n")

    assert cmd_restore_source(config, all_graph_only=True) == 0
    capsys.readouterr()

    text = _buffer(config)
    pos = {s: text.index(f"### {s}") for s in ("zulu-first", "alpha-second", "mike-third")}
    assert pos["mike-third"] < pos["alpha-second"] < pos["zulu-first"], (
        f"newest-committed must sit highest in the buffer, got {pos}"
    )
