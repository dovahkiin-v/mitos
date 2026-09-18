"""Tests for `mitos.settledness` — the predicate and the bounded tail window rotation reads.

Pure rows: the graph is a hand-written fake that counts its reads (never a MagicMock,
which would absorb a wrong call shape), the clock is a stamp, and the buffer is text.
Each fake node is built from the parsed block exactly as the store would hydrate it,
so "settled" means the same comparison sync's reconcile branch makes.
"""

import os
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pytest

from mitos import markers
from mitos.config import MitosConfig
from mitos.divergence import declared_edges
from mitos.identity import compute_node_id
from mitos.parser import parse_entry_stream
from mitos.rotation import plan_rotation
from mitos.settledness import (
    DIVERGED,
    RECENT,
    UNCOMMITTED,
    UNPARSEABLE,
    UNSTAMPED,
    WINDOW_MISMATCH,
    buffered_entries,
    select_settled_tail,
)

_HEADER = "# Decisions\n<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->\n\n"
NOW = "2026-09-14T12:00:00+00:00"
OLD = "2026-08-01T00:00:00+00:00"
FRESH = "2026-09-13T12:00:00+00:00"
_GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "decisions.reference.md")


def _block(slug: str, *, rejected: str = "The alternative, for a reason.",
           scope: str = "core", relations=(), extra: str = "") -> str:
    lines = [
        f"### {slug}",
        "",
        f"**Decided:** The {slug} axiom.",
        f"**Rejected:** {rejected}",
        f"**Mechanisms:** {slug}-mechanism",
        f"**Scope:** {scope}",
    ]
    lines += [f"**{field}:** [{target}]" for field, target in relations]
    return "\n".join(lines) + "\n" + extra + "\n"


def _buffer(*blocks: str) -> str:
    """The buffer as a writer leaves it: ``blocks`` top to bottom (newest first)."""
    return _HEADER + "".join(blocks)


class FakeGraph:
    """The two reads settledness makes, counted, over nodes built from parsed blocks."""

    def __init__(self) -> None:
        self.nodes: Dict[str, dict] = {}
        self.edges: Dict[str, List[dict]] = {}
        self.node_calls = 0
        self.edge_calls = 0

    def commit(self, block: str, *, updated_at: Optional[str] = OLD,
               edges: Optional[List[dict]] = None, **node_overrides) -> str:
        # Parsed under a header, as it sits in a buffer: a body line naming the
        # sentinel must not become this lone block's first sentinel.
        entry = parse_entry_stream(_HEADER + block, "decision")[0]
        node_id = compute_node_id(kind="decision", axiom=entry.axiom,
                                  mechanism_refs=entry.mechanisms)
        node = {
            "slug": entry.slug,
            "rejected_paths": entry.rejected_paths,
            "invalidates_if": entry.invalidates_if,
            "context": entry.context,
            "source": entry.source or "user",
            "scope": list(entry.scope),
            "updated_at": updated_at,
        }
        node.update(node_overrides)
        self.nodes[node_id] = node
        self.edges[node_id] = edges if edges is not None else [
            dict(edge) for edge in declared_edges(entry)]
        return node_id

    def get_node(self, node_id: str) -> Optional[dict]:
        self.node_calls += 1
        node = self.nodes.get(node_id)
        return dict(node) if node else None

    def get_outgoing_edges(self, node_id: str) -> List[dict]:
        self.edge_calls += 1
        return list(self.edges.get(node_id, []))


def _select(buffer: str, graph: FakeGraph, *, window: int = 10, threshold: int = 1,
            lag: int = 14, now: str = NOW):
    return select_settled_tail(buffer, graph=graph, now=now, lag_days=lag,
                               threshold=threshold, window=window,
                               archive_name="2026-Q3.md")


# --- T10-1: the truth table -----------------------------------------------------------

def _case_uncommitted(graph: FakeGraph) -> str:
    return _block("subject")


def _case_recent(graph: FakeGraph) -> str:
    block = _block("subject")
    graph.commit(block, updated_at=FRESH)
    return block


def _case_diverged_commentary(graph: FakeGraph) -> str:
    block = _block("subject")
    graph.commit(block, rejected_paths="What the graph still says.")
    return block


def _case_diverged_scope_order(graph: FakeGraph) -> str:
    block = _block("subject", scope="core, beta")
    graph.commit(block, scope=["beta", "core"])
    return block


def _case_diverged_edge(graph: FakeGraph) -> str:
    block = _block("subject")
    graph.commit(block, edges=[{"kind": "amends", "target": "gone"}])
    return block


def _case_diverged_source_only(graph: FakeGraph) -> str:
    # `source` is not reconcilable by sync (MI-4), so this entry stalls the drain until
    # hand-fixed — the sharpest member of the contiguous stop's residual.
    block = _block("subject")
    graph.commit(block, source="import_llm")
    return block


def _case_unstamped_missing(graph: FakeGraph) -> str:
    block = _block("subject")
    graph.commit(block, updated_at=None)
    return block


def _case_unstamped_naive(graph: FakeGraph) -> str:
    block = _block("subject")
    graph.commit(block, updated_at="2026-08-01T00:00:00")
    return block


def _case_unstamped_garbage(graph: FakeGraph) -> str:
    block = _block("subject")
    graph.commit(block, updated_at="last tuesday")
    return block


def _case_unparseable(graph: FakeGraph) -> str:
    return "### subject\n\n**Rejected:** A block with no decision line.\n\n"


@pytest.mark.parametrize("case, reason", [
    (_case_uncommitted, UNCOMMITTED),
    (_case_recent, RECENT),
    (_case_diverged_commentary, DIVERGED),
    (_case_diverged_scope_order, DIVERGED),
    (_case_diverged_edge, DIVERGED),
    (_case_diverged_source_only, DIVERGED),
    (_case_unstamped_missing, UNSTAMPED),
    (_case_unstamped_naive, UNSTAMPED),
    (_case_unstamped_garbage, UNSTAMPED),
    (_case_unparseable, UNPARSEABLE),
])
def test_each_unsettled_state_stops_the_batch_with_its_reason(case, reason) -> None:
    graph = FakeGraph()
    newer = _block("newer-settled")
    graph.commit(newer)
    subject = case(graph)

    selection = _select(_buffer(newer, subject), graph)

    assert selection.blocks == []
    assert selection.stopped_at == ("subject", reason)
    assert selection.examined == 1, "the walk stops at the first unsettled tail entry"


def test_a_settled_entry_is_selected() -> None:
    graph = FakeGraph()
    block = _block("subject", relations=[("Cites", "elsewhere")])
    graph.commit(block)

    selection = _select(_buffer(block), graph)

    assert [b.label for b in selection.blocks] == ["subject"]
    assert selection.blocks[0].raw_text == block
    assert selection.blocks[0].archive_name == "2026-Q3.md"
    assert selection.stopped_at is None
    assert graph.edge_calls == 1, "not diverged reads the node's own edges"


# --- T10-2 and T10-3: the tail, contiguously ------------------------------------------

def test_the_window_is_the_tail_and_the_batch_is_oldest_first() -> None:
    graph = FakeGraph()
    blocks = {f"s{i}": _block(f"s{i}") for i in range(6)}
    for block in blocks.values():
        graph.commit(block)
    # Prepend order: s5 was written last, so it sits at the top; s0 is the file bottom.
    buffer = _buffer(*(blocks[f"s{i}"] for i in reversed(range(6))))

    selection = _select(buffer, graph, window=3)

    assert [b.label for b in selection.blocks] == ["s0", "s1", "s2"]
    assert selection.examined == 3
    assert selection.buffered == 6


def test_the_batch_stops_at_the_first_unsettled_entry_from_the_tail() -> None:
    graph = FakeGraph()
    settled_top, diverged, settled_bottom = _block("c"), _block("b"), _block("a")
    graph.commit(settled_top)
    graph.commit(diverged, rejected_paths="The graph disagrees.")
    graph.commit(settled_bottom)

    selection = _select(_buffer(settled_top, diverged, settled_bottom), graph)

    assert [b.label for b in selection.blocks] == ["a"], "never the settled subset"
    assert selection.stopped_at == ("b", DIVERGED)


# --- T10-4 and T10-5: the bound is on entries examined ----------------------------------

def test_an_all_settled_buffer_examines_the_window_and_no_more() -> None:
    graph = FakeGraph()
    blocks = [_block(f"many-{i:03d}") for i in range(200)]
    for block in blocks:
        graph.commit(block)
    graph.node_calls = 0

    selection = _select(_buffer(*reversed(blocks)), graph, window=10, threshold=50)

    assert selection.buffered == 200
    assert selection.examined == 10
    assert len(selection.blocks) == 10
    assert graph.node_calls == 10, "never one read per buffered entry"


def test_a_recent_tail_examines_one_entry() -> None:
    graph = FakeGraph()
    blocks = [_block(f"many-{i:03d}") for i in range(60)]
    for block in blocks[1:]:
        graph.commit(block)
    graph.commit(blocks[0], updated_at=FRESH)

    selection = _select(_buffer(*reversed(blocks)), graph, window=10, threshold=50)

    assert selection.examined == 1 and graph.node_calls == 1
    assert selection.blocks == [] and selection.stopped_at == ("many-000", RECENT)


def test_below_the_threshold_nothing_is_parsed_and_the_graph_is_not_read(monkeypatch) -> None:
    graph = FakeGraph()
    blocks = [_block(f"few-{i:02d}") for i in range(49)]
    for block in blocks:
        graph.commit(block)
    graph.node_calls = 0
    import mitos.settledness as settledness_module
    monkeypatch.setattr(settledness_module, "parse_entry_stream",
                        lambda *a, **k: pytest.fail("parsed below the threshold"))

    selection = _select(_buffer(*reversed(blocks)), graph, threshold=50)

    assert (selection.buffered, selection.examined, selection.blocks) == (49, 0, [])
    assert graph.node_calls == 0 and graph.edge_calls == 0


def test_an_empty_buffer_selects_nothing_whatever_the_threshold() -> None:
    """A caller passing threshold 0 (3c chooses its own values) meets no empty window."""
    graph = FakeGraph()

    selection = _select(_HEADER, graph, threshold=0)

    assert (selection.buffered, selection.examined, selection.blocks) == (0, 0, [])
    assert graph.node_calls == 0


def test_the_threshold_is_inclusive() -> None:
    graph = FakeGraph()
    blocks = [_block(f"edge-{i}") for i in range(3)]
    for block in blocks:
        graph.commit(block)
    buffer = _buffer(*reversed(blocks))

    assert _select(buffer, graph, threshold=4).blocks == []
    assert graph.node_calls == 0
    assert len(_select(buffer, graph, threshold=3).blocks) == 3


# --- T10-6 and T10-7: the count is the parser's section rule ----------------------------

def test_an_init_seeded_buffer_counts_zero(tmp_path) -> None:
    from mitos import cli
    cli.cmd_init(MitosConfig(str(tmp_path)))
    with open(tmp_path / "decisions.md", encoding="utf-8") as fh:
        text = fh.read()
    assert "### " in text, "non-vacuity: the sample block is in the file"

    selection = _select(text, FakeGraph(), threshold=1)

    assert selection.buffered == 0
    assert markers.entry_heading_indices(text.splitlines(keepends=True)) == []


_TRANSCRIPT_BLOCK = _block(
    "with-transcript",
    extra=("\n[DECISION_TRANSCRIPT]\n"
           "### a heading-shaped transcript line\n"
           "## another one\n"
           "[/DECISION_TRANSCRIPT]\n"),
)


def test_a_heading_inside_a_transcript_is_not_counted() -> None:
    buffer = _buffer(_block("above"), _TRANSCRIPT_BLOCK, _block("below"))

    assert _select(buffer, FakeGraph(), threshold=99).buffered == 3


def test_the_record_paths_pre_gate_counts_what_the_selector_counts(tmp_path) -> None:
    """R12 (3c): `buffered_entries` is the selector's own count, sample block excluded."""
    from mitos import cli
    cli.cmd_init(MitosConfig(str(tmp_path)))
    with open(tmp_path / "decisions.md", encoding="utf-8") as fh:
        seeded = fh.read()
    texts = {
        "init-seed": seeded,
        "golden-reference": open(_GOLDEN, encoding="utf-8").read(),
        "transcript": _buffer(_block("above"), _TRANSCRIPT_BLOCK, _block("below")),
    }
    counts = {name: buffered_entries(text) for name, text in texts.items()}

    assert counts == {
        name: _select(text, FakeGraph(), threshold=10**9).buffered
        for name, text in texts.items()
    }
    assert counts["init-seed"] == 0 and counts["transcript"] == 3
    assert counts["golden-reference"] > 0, "non-vacuity"


@pytest.mark.parametrize("text", [
    pytest.param(open(_GOLDEN, encoding="utf-8").read(), id="golden-reference"),
    pytest.param(_buffer(_block("above"), _TRANSCRIPT_BLOCK, _block("below")),
                 id="transcript"),
])
def test_the_heading_count_equals_the_parse(text: str) -> None:
    failures: list = []
    entries = parse_entry_stream(text, "decision", failures=failures)
    indices = markers.entry_heading_indices(text.splitlines(keepends=True))

    assert len(indices) > 0, "non-vacuity"
    assert len(indices) == len(entries) + len(failures)


# --- T10-8: the raw slice is what rotation matches --------------------------------------

def test_every_selected_block_occurs_once_line_anchored_and_plans_as_rotated() -> None:
    graph = FakeGraph()
    # A body line naming the sentinel un-backticked (G6), a transcript, and a last block
    # with no final newline.
    mentions = _block("mentions-the-sentinel",
                      rejected="Treating a BEGIN ENTRIES line in a body as a sentinel.")
    last = _block("no-final-newline").rstrip("\n")
    blocks = [_block("top"), mentions, _TRANSCRIPT_BLOCK, last]
    for block in blocks:
        graph.commit(block)
    buffer = _buffer(*blocks)

    selection = _select(buffer, graph)

    assert [b.label for b in selection.blocks] == [
        "no-final-newline", "with-transcript", "mentions-the-sentinel", "top"]
    for block in selection.blocks:
        starts = [0] + [i + 1 for i, ch in enumerate(buffer) if ch == "\n"]
        assert sum(1 for s in starts if buffer.startswith(block.raw_text, s)) == 1
    plan = plan_rotation(buffer, selection.blocks)
    assert plan.rotated == selection.blocks
    assert plan.new_buffer == _HEADER


def test_the_window_mismatch_guard_rotates_nothing(monkeypatch) -> None:
    graph = FakeGraph()
    block = _block("guarded")
    graph.commit(block)
    real = markers.entry_heading_indices
    monkeypatch.setattr(markers, "entry_heading_indices",
                        lambda lines: real(lines) + [len(lines)])

    selection = _select(_buffer(block), graph)

    assert selection.blocks == []
    assert selection.stopped_at == ("", WINDOW_MISMATCH)
    assert graph.node_calls == 0


# --- T10-9: the clock ---------------------------------------------------------------------

def test_exactly_lag_old_is_not_quiet_and_a_zero_lag_needs_an_earlier_stamp() -> None:
    now = datetime.fromisoformat(NOW)
    exactly = (now - timedelta(days=14)).isoformat()
    just_over = (now - timedelta(days=14, microseconds=1)).isoformat()

    for stamp, lag, expected in [
        (exactly, 14, RECENT),
        (just_over, 14, None),
        (NOW, 0, RECENT),
        (OLD, 0, None),
    ]:
        graph = FakeGraph()
        block = _block("clocked")
        graph.commit(block, updated_at=stamp)
        selection = _select(_buffer(block), graph, lag=lag)
        if expected is None:
            assert [b.label for b in selection.blocks] == ["clocked"], (stamp, lag)
        else:
            assert selection.stopped_at == ("clocked", expected), (stamp, lag)


def test_a_rotation_instant_without_an_offset_is_refused() -> None:
    graph = FakeGraph()
    block = _block("clocked")
    graph.commit(block)
    with pytest.raises(ValueError):
        _select(_buffer(block), graph, now="2026-09-14T12:00:00")


# --- T10-10: the tier -----------------------------------------------------------------------

def test_importing_settledness_pulls_in_no_store_sync_lock_or_sdk() -> None:
    """Tier 2: the predicate reads the graph through an injected object, never a store."""
    probe = (
        "import sys; import mitos.settledness; "
        "print(','.join(sorted(m for m in ('mitos.store', 'mitos.sync', 'mitos.cutover', "
        "'filelock', 'anthropic', 'google.genai') if m in sys.modules))); "
        "print('mitos.divergence' in sys.modules and 'mitos.parser' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True)
    assert out.stdout.split("\n")[:2] == ["", "True"], out.stdout
