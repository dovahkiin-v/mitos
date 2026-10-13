"""Tests for the corpus↔graph divergence comparator (`mitos/divergence.py`).

The leaf is the shared unit both surfaces call — `mitos status`'s whole-corpus rung
and `sync`'s per-entry reconcile gate — so its field contract is pinned here rather
than at either surface. Every assertion below corresponds to a specific way the
comparator can be silently wrong: a field set that includes something the reconcile
cannot repair, a normalization gap that reports absence as divergence, or an
omission that leaves a species permanently invisible.
"""

import pytest

from mitos import divergence
from mitos.divergence import (
    COMMENTARY_FIELDS,
    RELATIONSHIP_FIELDS,
    SPAN_CODEPOINT_CAP,
    code_point_names,
    differing_span,
    entry_divergence,
    has_divergence,
    is_reconcilable,
    judged_text,
    needs_code_points,
    strip_citation,
)
from mitos.parser import ParsedEntry


def _entry(**overrides) -> ParsedEntry:
    """Builds a fully-populated ParsedEntry, overridable per test."""
    e = ParsedEntry("decision", "probe-slug", 1, 10)
    e.axiom = "We use SQLite in WAL mode."
    e.mechanisms = ["sqlite", "wal-mode"]
    e.rejected_paths = "pgvector — too heavy."
    e.invalidates_if = "A server database becomes acceptable."
    e.context = "Local-first, no daemon."
    e.scope = ["substrate", "store"]
    e.source = None
    for k, v in overrides.items():
        setattr(e, k, v)
    return e


def _node(**overrides) -> dict:
    """Builds the matching committed-node dict, overridable per test."""
    node = {
        "id": "abc123",
        "slug": "probe-slug",
        "core_axiom": "We use SQLite in WAL mode.",
        "mechanisms": ["sqlite", "wal-mode"],
        "rejected_paths": "pgvector — too heavy.",
        "invalidates_if": "A server database becomes acceptable.",
        "context": "Local-first, no daemon.",
        "source": "user",
        # Deliberately present, and deliberately NOT diffed — see the field-set test.
        "confirmed_by": "agent",
        "confirmed_at": "2026-06-23T13:04:17.147973+00:00",
    }
    node.update(overrides)
    return node


_SCOPES = ["substrate", "store"]
_EDGES: list = []


# --- the clean case, which everything else is measured against -------------------

def test_an_agreeing_entry_reports_nothing() -> None:
    """A corpus that matches its graph reports all-zero — the healthy steady state."""
    report = entry_divergence(_entry(), _node(), _SCOPES, _EDGES)
    assert report == {"commentary": [], "scope": None, "edges": None, "source": None}
    assert has_divergence(report) is False
    assert is_reconcilable(report) is False


def test_confirmation_metadata_never_counts_as_divergence() -> None:
    """`confirmed_by`/`confirmed_at` are excluded permanently — the §4.2 trap.

    The obvious implementation reuses the store's own `commentary` dict, which
    carries both, and yields a false positive on EVERY confirmed node — 114 of them
    on the live dogfood corpus. Their primary source is the graph; markdown has no
    field to compare against, so there is nothing to diff. This fixture deliberately
    stamps `confirmed_by='agent'` so the test catches the field-set trap rather than
    passing by luck.
    """
    node = _node(confirmed_by="agent", confirmed_at="2026-06-23T13:04:17.147973+00:00")
    assert has_divergence(entry_divergence(_entry(), node, _SCOPES, _EDGES)) is False

    # And the pinned set names neither of them.
    assert set(COMMENTARY_FIELDS).isdisjoint({"confirmed_by", "confirmed_at"})


def test_the_commentary_field_set_is_exactly_the_mutable_update_set() -> None:
    """S1 is exactly `slug|rejected_paths|invalidates_if|context`.

    Widening it without widening the store's commentary `UPDATE SET` would report a
    divergence the reconcile provably cannot repair — a signal with no repair is
    worse than no signal, because it teaches the reader to ignore the rung.
    """
    assert COMMENTARY_FIELDS == ("slug", "rejected_paths", "invalidates_if", "context")


# --- S1: commentary text ---------------------------------------------------------

@pytest.mark.parametrize("field, corpus_value", [
    ("slug", "renamed-slug"),
    ("rejected_paths", "pgvector — CORRECTED reasoning."),
    ("invalidates_if", "A different trigger entirely."),
    ("context", "Rewritten background."),
])
def test_each_commentary_field_surfaces_on_its_own(field, corpus_value) -> None:
    """Each S1 field is detected independently, and nothing else fires with it."""
    report = entry_divergence(_entry(**{field: corpus_value}), _node(), _SCOPES, _EDGES)
    assert report["commentary"] == [field]
    assert report["scope"] is None and report["edges"] is None and report["source"] is None
    assert is_reconcilable(report) is True


def test_absent_and_empty_optional_prose_are_the_same_absence() -> None:
    """`None` vs `""` vs `"  "` must not read as divergence.

    The parser yields `None` for an omitted field while the graph stores `NULL` or
    `''` by column, so a naive `!=` would report every entry that simply has no
    `**Context:**` — noise on a majority of a real corpus.
    """
    entry = _entry(context=None, invalidates_if=None)
    node = _node(context="", invalidates_if="   ")
    assert entry_divergence(entry, node, _SCOPES, _EDGES)["commentary"] == []


def test_slug_case_alone_is_not_divergence() -> None:
    """MI-13's identity is the casefolded slug, so case-only differs are not repairable."""
    report = entry_divergence(_entry(slug="Probe-Slug"), _node(slug="probe-slug"), _SCOPES, _EDGES)
    assert report["commentary"] == []


# --- the span the report prints (B5) ----------------------------------------------

# (a, b) pairs the comparator judges; each runs for every commentary field on both
# sides, so a field added to COMMENTARY_FIELDS later is covered without a new row.
_JUDGED_PAIRS = [
    ("same-value", "same-value"),
    ("same-value", "  same-value \n"),
    (None, ""),
    (None, "   "),
    (None, "some text"),
    ("Same-Value", "same-value"),
    ("stra\u00dfe", "strasse"),
    ("same-value", "s\u0430me-value"),  # Cyrillic а
    ("the old middle text", "the new middle text"),
]


@pytest.mark.parametrize("field", COMMENTARY_FIELDS)
@pytest.mark.parametrize("a, b", _JUDGED_PAIRS)
def test_judged_text_agrees_with_the_comparator(field: str, a, b) -> None:
    """A field is listed exactly when its judged values differ.

    The span report restates the comparator's rule rather than sharing it (the
    comparator is not edited to call `judged_text`), so this row is the only thing
    holding the two together. The `ß`/`ss` and case-only cells are the slug's: equal
    under casefold there, different everywhere else.
    """
    listed = field in entry_divergence(
        _entry(**{field: a}), _node(**{field: b}), _SCOPES, _EDGES)["commentary"]
    assert listed == (judged_text(field, a) != judged_text(field, b))


def _reassembles(a: str, b: str, span: dict) -> None:
    """Asserts the prefix, span and suffix put both inputs back together."""
    o, k = span["offset"], span["shared_after"]
    assert a[:o] == b[:o]
    assert a[len(a) - k:] == b[len(b) - k:]
    assert a[:o] + span["graph"] + a[len(a) - k:] == a
    assert b[:o] + span["markdown"] + b[len(b) - k:] == b
    assert (span["graph_len"], span["markdown_len"]) == (len(a), len(b))


@pytest.mark.parametrize("a, b, offset, shared_after, g, m", [
    ("the quick fox", "the quack fox", 6, 6, "i", "a"),
    ("abcdef", "abcXYdef", 3, 3, "", "XY"),
    ("abcdef", "abc", 3, 0, "def", ""),
    ("abc", "abcabc", 3, 0, "", "abc"),
    ("abcabc", "abc", 3, 0, "abc", ""),
    ("aaa", "aaaa", 3, 0, "", "a"),
    ("xyz", "pqr", 0, 0, "xyz", "pqr"),
])
def test_differing_span_arithmetic(a, b, offset, shared_after, g, m) -> None:
    """The prefix is counted first and bounds the suffix, so the two never overlap.

    Unbounded, `"abc"` against `"abcabc"` counts the shared `"abc"` at both ends and
    slices a negative-width span.
    """
    span = differing_span(a, b)
    assert (span["offset"], span["shared_after"], span["graph"], span["markdown"]) == (
        offset, shared_after, g, m)
    _reassembles(a, b, span)


def test_equal_values_have_no_span() -> None:
    assert differing_span("same", "same") is None
    assert differing_span("", "") is None


def test_offsets_count_characters_not_bytes() -> None:
    """Nine Lithuanian letters, two UTF-8 bytes each, then a hyphen: offset 10, not 19."""
    span = differing_span("ąčęėįšųūž-x", "ąčęėįšųūž-y")
    assert span["offset"] == 10
    assert (span["graph"], span["markdown"]) == ("x", "y")


def test_a_homoglyph_names_both_code_points() -> None:
    """A Cyrillic `а` renders like a Latin `a`; naming one side alone leaves a guess."""
    span = differing_span(judged_text("context", "same-value"),
                          judged_text("context", "s\u0430me-value"))
    assert needs_code_points(span["graph"], span["markdown"])
    assert code_point_names(span["graph"]) == "U+0061 LATIN SMALL LETTER A"
    assert code_point_names(span["markdown"]) == "U+0430 CYRILLIC SMALL LETTER A"


def test_a_printable_ascii_span_names_no_code_points() -> None:
    assert not needs_code_points("quick", "quack")
    assert not needs_code_points("", "~ !")


def test_whitespace_and_control_characters_are_named() -> None:
    """A tab and a space render alike, and a control character has no Unicode name."""
    assert needs_code_points("\t", " ")
    assert code_point_names("\t") == "U+0009"
    assert code_point_names("") == "(none)"


def test_code_point_names_are_capped() -> None:
    span = "ą" * (SPAN_CODEPOINT_CAP + 5)
    names = code_point_names(span)
    assert names.endswith(" (+5 more)")
    assert names.count("U+0105") == SPAN_CODEPOINT_CAP
    assert "more" not in code_point_names("ą" * SPAN_CODEPOINT_CAP)


# --- S2: scope -------------------------------------------------------------------

def test_scope_divergence_reports_both_sides() -> None:
    """Scope is a RETRIEVAL defect — a wrong value hides the decision from scoped reads."""
    report = entry_divergence(_entry(scope=["substrate", "config"]), _node(), _SCOPES, _EDGES)
    assert report["scope"] == {"graph": ["substrate", "store"],
                               "markdown": ["substrate", "config"],
                               "order_only": False}
    assert report["commentary"] == []
    assert is_reconcilable(report) is True


def test_scope_duplicates_case_and_whitespace_are_not_divergence() -> None:
    """A repeat, a case variant or padding is the same tag after normalization.

    Both sides go through the store's own scope-tag rule, so what the comparator calls
    equal is exactly what `commit_parsed_entry` would write no row for. A comparator
    that saw a difference here would reconcile forever: every sync would re-commit an
    entry whose commit changes nothing.
    """
    entry = _entry(scope=[" Substrate ", "", "substrate", "STORE", "store"])
    assert entry_divergence(entry, _node(), _SCOPES, _EDGES)["scope"] is None


def test_a_scope_reorder_is_a_reconcilable_divergence() -> None:
    """The store keeps authored order, so a same-set reorder IS a divergence.

    The first tag is the primary scope, which decides the rendered file carrying the
    full body. If a reorder did not diverge, sync would never reconcile a hand-edit
    moving a tag to the front, and the author would see nothing happen. Both lists are
    reported in authored order, and the leaf classifies the row as order-only.
    """
    report = entry_divergence(_entry(scope=["store", "substrate"]), _node(), _SCOPES, _EDGES)
    assert report["scope"] == {"graph": ["substrate", "store"],
                               "markdown": ["store", "substrate"],
                               "order_only": True}
    assert is_reconcilable(report) is True


# --- S5: edges, additions and deletions split ------------------------------------

def test_edge_additions_and_deletions_are_reported_separately() -> None:
    """The split is load-bearing: a removed markdown line DELETES the stored edge.

    `commit_parsed_entry` mirrors edges declaratively, so a reconcile is not
    commentary-only in effect. `--yes` applies additions but skips deletions, which it
    can only do if the two are distinguishable here.
    """
    entry = _entry(amends=["[kept-one]"], cites=["[newly-added]"])
    stored = [{"kind": "amends", "target": "kept-one"},
              {"kind": "supersedes", "target": "dropped-one"}]
    report = entry_divergence(entry, _node(), _SCOPES, stored)
    assert report["edges"] == {"added": ["cites:newly-added"],
                               "removed": ["supersedes:dropped-one"]}


def test_bracketed_and_bare_citations_are_the_same_declaration() -> None:
    """MI-7: the parser keeps `[slug]`, the agentic path stores a bare slug.

    Without this normalization every corpus-authored edge would report as both an
    addition and a deletion — the detector would find divergence on a clean corpus.
    """
    entry = _entry(supersedes=["[legacy-store]"])
    stored = [{"kind": "supersedes", "target": "legacy-store"}]
    assert entry_divergence(entry, _node(), _SCOPES, stored)["edges"] is None


def test_edge_target_case_is_normalized() -> None:
    """Casefolded, matching the store's own edge resolution (MI-7)."""
    entry = _entry(cites=["[Legacy-Store]"])
    stored = [{"kind": "cites", "target": "legacy-store"}]
    assert entry_divergence(entry, _node(), _SCOPES, stored)["edges"] is None


def test_every_relationship_field_is_read() -> None:
    """All nine, not just the kill-edges — an unread field is a permanently blind spot."""
    for field in RELATIONSHIP_FIELDS:
        entry = _entry(**{field: ["[some-target]"]})
        report = entry_divergence(entry, _node(), _SCOPES, [])
        assert report["edges"] == {"added": [f"{field}:some-target"], "removed": []}, field


# --- S6: source, the species the field contract forgot ---------------------------

def test_source_divergence_is_its_own_species() -> None:
    """`**Source:**` is a markdown field AND mutation-fenced graph-side (MI-4).

    That pair makes it invisible in a way no other field is: a hand-edit keeps the
    same node id (source is out-of-core), raises no S1 row, and is never reconciled —
    then a rebuild replays the markdown value and silently flips stored provenance.
    """
    report = entry_divergence(_entry(source="import_llm"), _node(source="user"), _SCOPES, _EDGES)
    assert report["source"] == {"graph": "user", "markdown": "import_llm"}
    assert report["commentary"] == []


def test_absent_source_means_user() -> None:
    """Absent ⇒ `user` per format-spec, so an omitted line is not divergence."""
    assert entry_divergence(_entry(source=None), _node(source="user"), _SCOPES, _EDGES)["source"] is None


def test_source_divergence_is_reported_but_not_reconcilable() -> None:
    """Its repair direction is markdown-conforms-to-GRAPH — the opposite of S1/S2/S5.

    `Source` is tool-only for authors, so the graph's stamped value is the authority.
    Re-committing the entry provably cannot change it (MI-4 fences it out of the
    `UPDATE SET`), so calling it reconcilable would promise a repair that does nothing.
    """
    report = entry_divergence(_entry(source="capture_llm"), _node(source="user"), _SCOPES, _EDGES)
    assert has_divergence(report) is True
    assert is_reconcilable(report) is False


# --- leaf discipline -------------------------------------------------------------

def test_importing_the_module_pulls_in_no_mitos_dependency() -> None:
    """`import mitos.divergence` must not drag in the parser, store, or cutover.

    `sync` imports this module for `entry_divergence` alone. The parser reads
    `format-spec.md` from package data at import time, and cutover pulls the whole
    replay machinery — the leaf must cost none of that, and must not be able to form
    an import cycle with the modules that call it. The whole-corpus fold DOES need
    them, which is exactly why its `mitos` imports are function-local; this asserts
    the resulting import graph rather than the source text, since the source-level
    check would forbid the fold outright.
    """
    import subprocess
    import sys as _sys

    probe = (
        "import sys; import mitos.divergence; "
        "leaked = sorted(m for m in sys.modules "
        "if m.startswith('mitos.') and m.split('.')[1] in "
        "{'parser', 'store', 'cutover', 'sync', 'cli', 'replay', 'identity'}); "
        "print(','.join(leaked))"
    )
    out = subprocess.run([_sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"leaked imports: {out.stdout.strip()}"


def test_the_pure_half_does_no_io() -> None:
    """`entry_divergence` and its helpers touch no filesystem — sync calls them in a loop.

    Sync iterates a LOCKED SNAPSHOT; a comparator that read the live file would
    compare against a different document than the one being synced, and would
    re-acquire the lock mid-pass.
    """
    import ast
    import inspect

    pure = ("entry_divergence", "declared_edges", "strip_citation",
            "has_divergence", "is_reconcilable", "_normalized_text", "_edge_key_set",
            "judged_text", "differing_span", "_is_printable_ascii",
            "needs_code_points", "code_point_names")
    forbidden = {"open", "read_text", "listdir", "makedirs", "stat", "remove"}
    for name in pure:
        tree = ast.parse(inspect.getsource(getattr(divergence, name)))
        called = {
            n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        } | {
            n.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert not (called & forbidden), f"{name} does I/O: {called & forbidden}"


def test_relationship_fields_match_the_parser() -> None:
    """The literal here must not drift from `parser._RELATIONSHIP_FIELDS`.

    Duplicated deliberately (the parser loads `format-spec.md` at import time, which
    the leaf must not), so the drift guard is this test rather than the import.
    """
    from mitos.parser import _RELATIONSHIP_FIELDS

    assert RELATIONSHIP_FIELDS == _RELATIONSHIP_FIELDS


def test_strip_citation_matches_the_stores_own() -> None:
    """Byte-identical to `store._strip_citation`, for the same leaf reason."""
    from mitos.store import _strip_citation

    for raw in ("[foo]", "foo", "  [ bar ] ", "[[nested]]", "", "   ", "[unbalanced"):
        assert strip_citation(raw) == _strip_citation(raw), raw


# ===========================================================================
# The whole-corpus fold — integration over a real workspace
# ===========================================================================

import os
import tempfile

from mitos.config import MitosConfig
from mitos.divergence import corpus_graph_divergence, divergence_total
from mitos.store import GraphStore

_SENTINEL = "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->"


def _block(slug, decided, *, rejected="n/a", mechanisms=("m1",), scope=(), context=None,
           cites=None, source=None):
    """Builds one decision entry block."""
    lines = [f"### {slug}", "", f"**Decided:** {decided}", f"**Rejected:** {rejected}"]
    if mechanisms:
        lines.append(f"**Mechanisms:** {', '.join(mechanisms)}")
    if scope:
        lines.append(f"**Scope:** {', '.join(scope)}")
    if context:
        lines.append(f"**Context:** {context}")
    if source:
        lines.append(f"**Source:** {source}")
    if cites:
        lines.append(f"**Cites:** [{cites}]")
    return "\n".join(lines)


def _workspace(tmp_path, *blocks, archive_blocks=()):
    """Builds a workspace whose corpus is committed to the graph, then returns its config."""
    from mitos.parser import parse_entry_stream

    config = MitosConfig(str(tmp_path))
    os.makedirs(config.mitos_dir, exist_ok=True)
    buffer_text = _SENTINEL + "\n\n" + "\n\n".join(blocks) + "\n"
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write(buffer_text)
    if archive_blocks:
        os.makedirs(config.archive_dir, exist_ok=True)
        with open(os.path.join(config.archive_dir, "2026-Q1.md"), "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(archive_blocks) + "\n")

    store = GraphStore(config.db_path)
    for path_text in (list(archive_blocks) + list(blocks)):
        for entry in parse_entry_stream(path_text, "decision"):
            store.commit_parsed_entry(entry)
    return config


def _report(config):
    return corpus_graph_divergence(GraphStore(config.db_path, read_only=True), config)


def test_a_byte_identical_corpus_reports_all_zero(tmp_path) -> None:
    """The healthy steady state — and the fixture is built to catch the field-set trap.

    Nodes are deliberately stamped `confirmed_by='agent'`, because the obvious
    implementation reuses the store's commentary dict, which carries the confirmation
    pair, and would report a false positive on every one of them. Without the stamp
    this test would pass by luck.
    """
    config = _workspace(tmp_path, _block("alpha", "Alpha axiom."), _block("beta", "Beta axiom."))
    store = GraphStore(config.db_path)
    with store._get_connection() as conn:
        conn.execute("UPDATE nodes SET confirmed_by = 'agent', "
                     "confirmed_at = '2026-06-23T13:04:17.147973+00:00'")

    report = _report(config)
    assert report["skipped"] is None
    assert report["checked"] == 2
    assert divergence_total(report) == 0
    assert report["reconcilable"] == 0 and report["archived_drift"] == 0


def test_a_fresh_workspace_reports_nothing_rather_than_a_verdict(tmp_path) -> None:
    """Empty/fresh states are first-class: no corpus and no graph is healthy, not broken."""
    config = MitosConfig(str(tmp_path))
    os.makedirs(config.mitos_dir, exist_ok=True)
    report = corpus_graph_divergence(None, config)
    assert report["skipped"] == "no corpus"
    assert divergence_total(report) == 0


def test_a_corpus_without_a_graph_is_skipped_not_reported_as_divergence(tmp_path) -> None:
    """A corpus with no graph yet is pending sync, not 209 phantom orphans."""
    config = MitosConfig(str(tmp_path))
    os.makedirs(config.mitos_dir, exist_ok=True)
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write(_SENTINEL + "\n\n" + _block("alpha", "Alpha axiom.") + "\n")
    report = corpus_graph_divergence(None, config)
    assert report["skipped"] == "no graph"


def test_one_hand_edited_buffer_entry_surfaces_exactly_one_commentary_row(tmp_path) -> None:
    """The brief's original case: an edit to a committed entry, invisible to `sync`."""
    config = _workspace(tmp_path, _block("alpha", "Alpha axiom."), _block("beta", "Beta axiom."))
    text = open(config.decisions_file, encoding="utf-8").read()
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write(text.replace("**Rejected:** n/a\n**Mechanisms:** m1\n\n### beta",
                              "**Rejected:** CORRECTED reasoning.\n**Mechanisms:** m1\n\n### beta"))

    report = _report(config)
    assert len(report["commentary"]) == 1
    assert report["commentary"][0]["slug"] == "alpha"
    assert report["commentary"][0]["fields"] == ["rejected_paths"]
    assert report["reconcilable"] == 1 and report["archived_drift"] == 0


def test_an_archive_resident_edit_is_archived_drift_not_reconcilable(tmp_path) -> None:
    """`sync` reads only the buffer, so an archived entry's reconciler is `rebuild`.

    Counting it as reconcilable would advertise a repair that silently does nothing.
    """
    config = _workspace(tmp_path, _block("live-one", "Live axiom."),
                        archive_blocks=(_block("archived-one", "Archived axiom."),))
    archive = os.path.join(config.archive_dir, "2026-Q1.md")
    text = open(archive, encoding="utf-8").read()
    with open(archive, "w", encoding="utf-8") as fh:
        fh.write(text.replace("**Rejected:** n/a", "**Rejected:** EDITED in the archive."))

    report = _report(config)
    assert [row["slug"] for row in report["commentary"]] == ["archived-one"]
    assert report["archived_drift"] == 1
    assert report["reconcilable"] == 0


def test_a_graph_only_node_surfaces_with_its_active_flag(tmp_path) -> None:
    """S3: a node whose `### ` block has left the corpus, with active/retired split."""
    config = _workspace(tmp_path, _block("kept", "Kept axiom."), _block("orphaned", "Orphan axiom."))
    text = open(config.decisions_file, encoding="utf-8").read()
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write(text.split("### orphaned")[0].rstrip() + "\n")

    report = _report(config)
    assert [row["slug"] for row in report["graph_only"]] == ["orphaned"]
    assert report["graph_only"][0]["active"] is True
    assert report["checked"] == 1, "a node with no block cannot be 'checked'"


def test_a_legacy_mark_wrapped_buffer_reports_clean_and_still_counts(tmp_path) -> None:
    """A legacy mark buffer must read CLEAN — and its entries must still be counted.

    Two failure directions in one assertion. If the sentinel bled into a field, the
    node id would shift and both entries would report as graph-only orphans plus
    unsynced strangers. If the tolerance skipped the whole span instead, the wrapped
    entry would vanish from `checked` and read as a genuine orphan — the rebuild-lossy
    `prune` behaviour the deprecation exists to remove.
    """
    config = _workspace(tmp_path, _block("above", "Above axiom.", mechanisms=("sqlite",)),
                        _block("wrapped", "Wrapped axiom.", mechanisms=("qdrant",)))
    text = open(config.decisions_file, encoding="utf-8").read()
    head, _, tail = text.partition("### wrapped")
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write(head + "<!-- ROTATED START\n### wrapped" + tail.rstrip("\n") + "\nROTATED END -->\n")

    report = _report(config)
    assert divergence_total(report) == 0, report
    assert report["checked"] == 2, "the enclosed block must still be corpus content"


def test_a_busy_corpus_reports_skipped_and_caches_nothing(tmp_path) -> None:
    """A torn read cached is a sticky lie — so a locked corpus yields no verdict at all.

    `status` runs at agent session start, concurrently with any sync. An unlocked read
    that caught the rotation window would see a truncated buffer and report mass
    phantom orphans; cached, that lie would stick until the next real edit.
    """
    from filelock import FileLock

    config = _workspace(tmp_path, _block("alpha", "Alpha axiom."))
    cache_path = os.path.join(config.mitos_dir, "divergence_cache.json")

    holder = FileLock(config.decisions_file + ".lock", timeout=5)
    with holder:
        report = _report(config)

    assert report["skipped"] == "corpus busy"
    assert divergence_total(report) == 0
    assert not os.path.exists(cache_path), "a skipped run must never write the cache"


def test_the_cache_is_invalidated_by_a_graph_side_change(tmp_path) -> None:
    """The key includes a GRAPH fingerprint, not just the corpus hash.

    Divergence is a property of the corpus VERSUS the graph, so keying on corpus
    content alone returns a stale verdict for any graph-side change under a static
    corpus. The sharp case is restoring a `.bak` after a rebuild swap: the graph
    reverts while the corpus keeps its restored blocks, the corpus hash is unchanged,
    the cache hits, and `status` reports clean over a corpus that has diverged the
    OTHER way. P6 invites this directly — "delete the graph and Mitos rebuilds" is a
    sanctioned operation that changes the graph without touching a byte of corpus.
    """
    config = _workspace(tmp_path, _block("alpha", "Alpha axiom."), _block("beta", "Beta axiom."))
    first = _report(config)
    assert first["cache_hit"] is False and divergence_total(first) == 0
    assert _report(config)["cache_hit"] is True, "an unchanged pair must hit the cache"

    # A graph-side edit only — not one byte of corpus moves. This is the shape a
    # `.bak` restore or a rebuild swap leaves behind.
    store = GraphStore(config.db_path)
    with store._get_connection() as conn:
        conn.execute(
            "UPDATE nodes SET rejected_paths_json = ?, updated_at = ? WHERE slug = 'beta'",
            ('"reverted by a .bak restore"', "2026-07-27T00:00:00.000000+00:00"),
        )

    after = _report(config)
    assert after["cache_hit"] is False, "a graph-side change must invalidate the cache"
    assert [row["slug"] for row in after["commentary"]] == ["beta"], (
        "a stale cache hit would have reported this corpus as clean"
    )


def test_the_report_carries_nothing_derived_from_signals(tmp_path) -> None:
    """The fingerprint reads only `nodes`, so a signals-derived field would go stale.

    `is_drifted` is an EXISTS over `signals` and rides along on every node read; a
    report echoing it would be cached against a fingerprint that cannot see it change.
    (`computed_state` is safe — it derives from the kill-edge anti-join over `edges`,
    and an edge change ticks the committing node's `updated_at`.)
    """
    config = _workspace(tmp_path, _block("alpha", "Alpha axiom."))
    report = _report(config)
    assert "is_drifted" not in repr(report)


def test_open_question_nodes_are_never_reported_as_source_block_orphans(tmp_path) -> None:
    """Open questions are out of scope on BOTH sides, or every one is a phantom orphan.

    `get_all_nodes` returns every kind while the corpus read covers `decisions.md`
    only, so folding one against the other made each open-question node a `graph_only`
    row — pointing the reader at `restore-source`, which can never repair one (an open
    question has no axiom, so no block can be rendered). A permanent, unfixable ⚠ on
    any workspace that actually uses `questions.md`. Excluding them from one side only
    is strictly worse than excluding them from both.
    """
    from mitos.parser import parse_entry_stream

    config = MitosConfig(str(tmp_path))
    os.makedirs(config.mitos_dir, exist_ok=True)
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write(_SENTINEL + "\n\n" + _block("alpha", "Alpha axiom.") + "\n")
    oq_text = (_SENTINEL + "\n\n### why-sqlite\n\n"
               "**Topic:** storage engine choice\n**Questions:**\n- Is WAL enough?\n")
    with open(config.questions_file, "w", encoding="utf-8") as fh:
        fh.write(oq_text)

    store = GraphStore(config.db_path)
    for entry in parse_entry_stream(open(config.decisions_file, encoding="utf-8").read(),
                                    "decision"):
        store.commit_parsed_entry(entry)
    oq_entries = parse_entry_stream(oq_text, "open_question")
    assert oq_entries and oq_entries[0].kind == "open_question", "fixture must commit an OQ"
    for entry in oq_entries:
        store.commit_parsed_entry(entry)

    report = _report(config)
    assert report["graph_only"] == [], "an open-question node is not a source-block orphan"
    assert divergence_total(report) == 0
    assert report["checked"] == 1, "only the decision was compared"


def test_a_cache_entry_of_an_unexpected_shape_cannot_crash_status(tmp_path, capsys) -> None:
    """A cache written by a build with a different species set must not KeyError.

    The rung indexes the species it knows, and it is called OUTSIDE `cmd_status`'s
    best-effort guard — so a stale sidecar from another version would take down the one
    command an operator runs to find out what is wrong. Two defences, both cheap: the
    key carries a schema version so the stale entry misses, and the rung reads with
    `.get` so it survives even if one ever matches.
    """
    import json as _json
    from mitos.cli import _print_divergence_rung

    # Shape from a hypothetical other build: a species this one indexes is absent.
    truncated = {"checked": 3, "skipped": None, "cache_hit": True,
                 "commentary": [{"slug": "alpha", "fields": ["context"]}],
                 "scope": [], "graph_only": [{"slug": "gone"}]}
    # `project` is required and keyword-only (3b), matching cmd_status's idiom; the
    # row is about `.get` robustness, not arity, so it loses nothing by naming one.
    _print_divergence_rung(truncated, project="demo")  # must not raise
    out = capsys.readouterr().out
    assert "disagree in 2 place(s)" in out
    assert "alpha" in out, "the species it DOES understand must still be reported"
    assert "1 node(s) have NO" in out, "and so must the one whose rows lack a key"

    # And the version prefix means such an entry is not served in the first place.
    config = _workspace(tmp_path, _block("alpha", "Alpha axiom."))
    _report(config)
    cache_path = os.path.join(config.mitos_dir, "divergence_cache.json")
    payload = _json.loads(open(cache_path, encoding="utf-8").read())
    # Bound to the constant, not a literal: the property under test is that the key
    # CARRIES a schema version, and every legitimate shape change bumps it.
    assert payload["key"].startswith(f"{divergence._CACHE_VERSION}:"), (
        "the cache key must carry a schema version"
    )


def test_every_mitos_recipe_the_rung_prints_names_its_project(capsys) -> None:
    """Every backticked `mitos …` span in the rung carries `-p <repr(project)>`.

    Since the selector flip a bare `mitos rebuild` has no target, so a reader copying a
    span out of the rung gets a second wall instead of a command. The subject is derived
    from the output — every span that starts `mitos ` — never a hand list of route
    clauses, so a clause added later is covered the day it prints. The report sets every
    species key, because a clause that does not print cannot fail the selector check,
    and each clause's lead phrase is asserted so the population cannot shrink silently.
    """
    import re
    from mitos.cli import _print_divergence_rung

    report = {
        "checked": 9, "skipped": None, "cache_hit": False,
        "commentary": [{"slug": "alpha", "fields": ["context"]}],
        "scope": [
            {"slug": "beta", "file": "decisions.md", "graph": ["x"], "markdown": ["y"],
             "order_only": False},
            {"slug": "gamma", "file": "decisions.md", "graph": ["x", "y"],
             "markdown": ["y", "x"], "order_only": True},
        ],
        "edges": [{"slug": "delta"}],
        "edge_verdicts": {"repairable": 1, "target_retired": 1, "unresolvable": 1,
                          "illegal": 1},
        "illegal_edge_types": ["resolves"],
        "source": [{"slug": "epsilon"}],
        "graph_only": [{"slug": "gone", "active": True}],
        "reconcilable": 2,
        "archived_drift": 1,
    }
    _print_divergence_rung(report, project="demo")
    out = capsys.readouterr().out

    for lead in ("commentary text differs", "scope tags differ", "ordered differently",
                 "declared relations", "replays them", "since-retired", "if its block went",
                 "can NEVER commit", "`**Source:**` line", "have NO", "can be repaired now",
                 "sit in an ARCHIVE file"):
        assert lead in out, f"the report sets this clause's species, so it must print: {lead!r}"

    spans = re.findall(r"`(mitos [^`]*)`", out)
    assert len(spans) >= 5, spans
    bare = [span for span in spans if "-p 'demo'" not in span]
    assert not bare, f"recipes printed without a selector: {bare}"


def test_rotation_mode_is_not_served_stale_from_the_cache(tmp_path) -> None:
    """`rotation_mode` is live config, not a property of the corpus/graph pair.

    Baked into a cached report it would keep reporting yesterday's mode indefinitely —
    and it reaches consumers through `status --json`.
    """
    config = _workspace(tmp_path, _block("alpha", "Alpha axiom."))
    first = _report(config)
    assert first["cache_hit"] is False

    config.rotation_mode = "archive-changed-under-us"
    second = _report(config)
    assert second["cache_hit"] is True, "the fixture must exercise the cache path"
    assert second["rotation_mode"] == "archive-changed-under-us"


def test_the_graph_is_read_under_the_same_lock_as_the_corpus(tmp_path) -> None:
    """Both reads happen inside one lock, or the diff compares mismatched snapshots.

    Reading the graph after releasing the lock would diff a corpus snapshot against a
    graph another process had since written — a node committed in that window reads as
    a phantom orphan, and the sidecar would store that false verdict under the PRE-race
    fingerprint, so returning the graph to that state later replays the lie as a cache
    hit. Asserted by making the graph read observable: a store whose `get_all_nodes`
    checks whether the lock is held.
    """
    from filelock import FileLock

    config = _workspace(tmp_path, _block("alpha", "Alpha axiom."))
    real = GraphStore(config.db_path, read_only=True)
    probe = FileLock(config.decisions_file + ".lock", timeout=0)
    observed = {}

    class _Watcher:
        """Delegates to the real store, recording whether the lock was free."""

        def graph_fingerprint(self):
            return real.graph_fingerprint()

        def get_all_nodes(self):
            observed["nodes_lock_free"] = _lock_is_free(probe)
            return real.get_all_nodes()

        def get_edges(self):
            observed["edges_lock_free"] = _lock_is_free(probe)
            return real.get_edges()

    corpus_graph_divergence(_Watcher(), config)
    assert observed["nodes_lock_free"] is False, "get_all_nodes ran outside the lock"
    assert observed["edges_lock_free"] is False, "get_edges ran outside the lock"


def _lock_is_free(probe) -> bool:
    """Returns True if the corpus lock could be acquired right now."""
    from filelock import Timeout

    try:
        with probe:
            return True
    except Timeout:
        return False


# --- The repairability verdict (§5.3.2) ---------------------------------------


def _graph_node(kind: str = "decision", state: str = "active"):
    """A minimal graph-node dict as `classify_absent_edge` reads it."""
    return {"kind": kind, "computed_state": state}


def test_absent_edge_verdicts_cover_the_four_repair_paths() -> None:
    """Each verdict must name a DIFFERENT repair, because each has a different verb.

    The rung already detected all 169 of cartolina's absent edges; what it could not
    say was what to do about them, and its one suggestion (`mitos sync`) is wrong for
    19. A count is a symptom — the reader needs the verb.
    """
    from mitos.divergence import EDGE_VERDICTS, classify_absent_edge

    # Legal, resolvable, active target — a rebuild simply replays it.
    assert classify_absent_edge(
        "cites", "decision", "t", [_graph_node()]
    ) == "repairable"

    # Legal and resolvable, but every candidate is retired: a replay has to reach it
    # in commit order, because citations resolve against the active view (§4.1).
    assert classify_absent_edge(
        "amends", "decision", "t", [_graph_node(state="superseded")]
    ) == "target_retired"

    # Names nothing at all.
    assert classify_absent_edge("cites", "decision", "t", []) == "unresolvable"

    # Cartolina's 19: `derives_from` must originate from an open question, so a
    # decision declaring it can never commit no matter what it points at.
    assert classify_absent_edge(
        "derives_from", "decision", "t", [_graph_node()]
    ) == "illegal"
    assert classify_absent_edge(
        "derives_from", "decision", "t", [_graph_node(kind="open_question")]
    ) == "illegal"

    assert set(EDGE_VERDICTS) == {
        "illegal", "unresolvable", "target_retired", "repairable"
    }


def test_a_lineage_slug_is_repairable_if_any_candidate_is_active() -> None:
    """One active legal candidate is enough — the verdict must not be first-wins.

    A slug can name a whole same-slug supersession lineage (MI-13). Judging on the
    first candidate would report a repairable edge as needing re-ordering, sending the
    reader to the harder verb for no reason.
    """
    from mitos.divergence import classify_absent_edge

    lineage = [_graph_node(state="superseded"), _graph_node(state="active")]
    assert classify_absent_edge("cites", "decision", "t", lineage) == "repairable"


def test_an_illegal_edge_is_legal_from_an_open_question_source() -> None:
    """The verdict reads the SOURCE kind, not just the edge type.

    `derives_from` is the correct relation from an open question to a decision — the
    illegality is entirely about where it is declared. A verdict that condemned the
    edge type outright would tell an author to remove a perfectly good relation.
    """
    from mitos.divergence import classify_absent_edge

    assert classify_absent_edge(
        "derives_from", "open_question", "t", [_graph_node(kind="decision")]
    ) == "repairable"


# ===========================================================================
# Phase 2b — authored order through the comparison
# ===========================================================================

_ADVERSARIAL_SCOPE_PAIRS = [
    # (stored, markdown)
    (["alpha", "beta"], ["beta", "alpha"]),                   # reorder
    (["b", "a", "c"], ["b", "c", "a"]),                       # reorder, primary kept
    (["alpha", "beta"], [" Alpha ", "BETA", "alpha", ""]),    # case + pad + dup + empty
    (["strasse"], ["Straße"]),                                # casefold, not lower()
    (["alpha", "beta"], ["alpha", "beta", "beta"]),           # dup only
    ([], [" ", ""]),                                          # empties only
    (["alpha"], ["alpha", "beta"]),                           # membership add
    (["alpha", "beta"], ["alpha"]),                           # membership drop
]


@pytest.mark.parametrize("stored, markdown", _ADVERSARIAL_SCOPE_PAIRS)
def test_the_comparator_agrees_with_the_stores_writer(tmp_path, stored, markdown) -> None:
    """"Divergence says equal" ⇔ "a real commit writes nothing and ticks nothing".

    If the two disagreed, sync would either reconcile forever (diverged, but the
    commit writes no row) or never reconcile a real change (equal, but the commit
    would have rewritten the rows). Driven through a real `GraphStore`, so it is the
    writer's actual behaviour being compared, not a restatement of it.
    """
    import sqlite3

    def entry_with(scope):
        e = ParsedEntry("decision", "probe-slug", 1, 10)
        e.axiom = "We use SQLite in WAL mode."
        e.mechanisms = ["sqlite"]
        e.rejected_paths = "pgvector — too heavy."
        e.scope = list(scope)
        return e

    db = str(tmp_path / "graph.sqlite")
    store = GraphStore(db)
    store.commit_parsed_entry(entry_with(stored))
    node = store.get_all_nodes()[0]

    def rows():
        with sqlite3.connect(db) as conn:
            return conn.execute(
                "SELECT scope, ordinal FROM node_scopes ORDER BY ordinal, scope"
            ).fetchall()

    rows_before, stamp_before = rows(), node["updated_at"]
    verdict = entry_divergence(entry_with(markdown), node, node["scope"], [])["scope"]

    store.commit_parsed_entry(entry_with(markdown))
    after = GraphStore(db).get_all_nodes()[0]
    wrote_nothing = rows() == rows_before and after["updated_at"] == stamp_before

    assert (verdict is None) == wrote_nothing, (verdict, rows_before, rows())


def test_a_previous_cache_version_is_not_served_for_a_reordered_corpus(tmp_path) -> None:
    """A sidecar from the set-comparing build must miss, not answer "clean".

    The key hashes corpus bytes and the graph fingerprint, and neither moves when only
    the comparator changed — so without the version bump an old "no scope divergence"
    for this exact corpus would be served verbatim. The row plants exactly that stale
    sidecar under the previous version's prefix and asserts the fold recomputes.
    """
    import json as _json

    config = _workspace(tmp_path, _block("alpha", "Alpha axiom.", scope=("alpha", "beta")))
    with open(config.decisions_file, encoding="utf-8") as fh:
        text = fh.read()
    assert "**Scope:** alpha, beta" in text
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write(text.replace("**Scope:** alpha, beta", "**Scope:** beta, alpha"))

    first = _report(config)
    assert first["cache_hit"] is False
    cache_path = os.path.join(config.mitos_dir, "divergence_cache.json")
    payload = _json.loads(open(cache_path, encoding="utf-8").read())
    prefix = f"{divergence._CACHE_VERSION}:"
    assert payload["key"].startswith(prefix)

    # "2" is the version the set-comparing build shipped — the build whose sidecar says
    # "clean" about a reorder. A literal, not `_CACHE_VERSION - 1`: derived, the row
    # would stay green under a build that never bumped the constant.
    previous = "2"
    stale_report = {**first, "scope": [], "reconcilable": 0}
    with open(cache_path, "w", encoding="utf-8") as fh:
        _json.dump({"key": previous + ":" + payload["key"][len(prefix):],
                    "report": stale_report}, fh)

    second = _report(config)
    assert second["cache_hit"] is False, "a previous-version sidecar must never be served"
    assert second["scope"] == [{"slug": "alpha", "file": "decisions.md",
                                "graph": ["alpha", "beta"], "markdown": ["beta", "alpha"],
                                "order_only": True}]


def test_the_reconcile_audit_row_records_values_never_the_verdict() -> None:
    """`order_only` is a classification, not a value the reconcile applies.

    The telemetry pair records what the graph held and what the markdown declares, in
    authored order, as plain JSON lists. The flag stays out of both halves: the
    documented read rule matches an intent row's `new_values` against the graph, and
    a verdict key there would never match anything.
    """
    import json as _json

    from mitos.sync import MitosSyncManager

    entry = _entry(scope=["store", "substrate"])
    report = entry_divergence(entry, _node(), _SCOPES, _EDGES)
    assert report["scope"]["order_only"] is True

    prior, new_values = MitosSyncManager._reconcile_value_pair(entry, _node(), report, [])
    assert prior == {"scope": ["substrate", "store"]}
    assert new_values == {"scope": ["store", "substrate"]}
    assert _json.loads(_json.dumps(new_values, sort_keys=True)) == new_values
    assert "order_only" not in _json.dumps([prior, new_values])
