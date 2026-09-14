"""A scope reorder, end to end: hand-edit → divergence → reconcile → render → rebuild.

The author's first `**Scope:**` tag is the primary scope, and the full entry renders
under that scope's file. The order is lost at three seams — the write, the comparison
and the render — and fixing only some of them is worse than fixing none: an author who
moves a tag to the front sees nothing happen. This module proves the whole chain:

* **R1** — the hand-edit path. A named reconcile (`perform_sync(repair_targets=…)`,
  which `mitos sync --reconcile-entry` passes through) moves the full body to the new
  first tag's file and leaves identity alone.
* **R2** — `mitos rebuild` from the same markdown produces the same primary.
* **R3** — the control. Without authorization a non-interactive sync reconciles
  nothing, so R1's move came from the authorization and not from an unconditional path.
* **R4** — the `mitos status` rung routes each scope row on the report's `order_only`
  flag, and only a membership row is called a findability defect.

Seeding goes through `record`, which commits and leaves the entry in the buffer
without needing an embedding key; a keyless `sync` commits nothing. The helpers are copied
in shape from `test_repair_door.py` rather than imported across test modules.
"""

import os
import sys
from typing import List, Tuple

import pytest

from mitos.config import MitosConfig
from mitos.sync import MitosSyncManager

from _conflict_helpers import _RecordingJudge, _wire_fakes, env, offline  # noqa: F401

_SLUG = "reordered"
_REJECTED = "The original rejected reasoning for the reorder row."


def _seed(config: MitosConfig, manager: MitosSyncManager) -> None:
    """Records one entry scoped `alpha, beta`, leaving its block in the buffer."""
    # Set on `config.env`, the seam the sync reads: the key floor returns before the
    # per-entry loop AND before `render_all`, so a keyless run would pass the graph
    # half of these rows while rendering nothing.
    config.env["GEMINI_API_KEY"] = "mock_key"
    _wire_fakes(manager, judge=_RecordingJudge([]))
    result = manager.record_decision_entry(
        "The primary scope follows the author's first tag.", _REJECTED,
        ["alpha", "beta"], mechanisms=["sqlite"], slug=_SLUG, acknowledge_neighbors=True,
    )
    assert result.get("state") == "active", result


def _reorder_buffer(config: MitosConfig) -> None:
    """Moves `beta` to the front of the entry's scope line, refusing a no-op edit."""
    with open(config.decisions_file, encoding="utf-8") as fh:
        text = fh.read()
    assert "**Scope:** alpha, beta" in text, "fixture edit target missing"
    with open(config.decisions_file, "w", encoding="utf-8") as fh:
        fh.write(text.replace("**Scope:** alpha, beta", "**Scope:** beta, alpha"))


def _axiom_file(config: MitosConfig, scope: str) -> str:
    with open(os.path.join(config.mitos_dir, "axioms", f"{scope}.md"), encoding="utf-8") as fh:
        return fh.read()


def _state(manager: MitosSyncManager) -> Tuple[str, List[str], int]:
    node = manager.store.get_node_by_slug(_SLUG)
    return node["id"], node["scope"], len(manager.store.get_all_nodes())


def test_a_reorder_reconciled_by_name_moves_the_rendered_primary(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """R1 — T4's hand-edit path: the body leaves `alpha.md` for `beta.md`.

    The seed's starting state is asserted too, so the row cannot pass on a graph that
    was never alphabetical-first (a build whose store sorted would red on the move).
    """
    config, manager, _ = env
    assert not sys.stdin.isatty()
    _seed(config, manager)
    node_id, scope, count = _state(manager)
    assert scope == ["alpha", "beta"]
    assert _REJECTED in _axiom_file(config, "alpha"), "the full body starts under alpha"
    assert _REJECTED not in _axiom_file(config, "beta")
    assert "full entry: alpha.md" in _axiom_file(config, "beta")

    _reorder_buffer(config)
    shortfall = manager.perform_sync(auto_accept=False, repair_targets=[_SLUG])

    assert shortfall == [], "a landed repair is not a shortfall"
    assert _state(manager) == (node_id, ["beta", "alpha"], count), (
        "order moves; id and node count do not (C1)"
    )
    beta, alpha = _axiom_file(config, "beta"), _axiom_file(config, "alpha")
    assert _REJECTED in beta, "the full body now renders under the new first tag"
    assert _REJECTED not in alpha, "and it left the old primary's file"
    assert "full entry: beta.md" in alpha, "which now holds only the pointer"
    assert "full entry: alpha.md" not in beta


def test_a_rebuild_from_the_reordered_markdown_reproduces_the_primary(
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """R2 — T4's rebuild half: replay writes authored ordinals, so the primary matches."""
    from mitos.cutover import default_aside_db_path, rebuild_and_gate
    from mitos.renderer import assemble_render
    from mitos.store import GraphStore

    config, manager, _ = env
    _seed(config, manager)
    _reorder_buffer(config)

    result = rebuild_and_gate(config, aside_db_path=default_aside_db_path(config),
                              strict=False)
    assert result.residual_casualties == []

    aside = GraphStore(result.aside_db_path, read_only=True)
    [node] = [n for n in aside.get_all_nodes() if n["slug"] == _SLUG]
    assert node["scope"] == ["beta", "alpha"]
    scopes = assemble_render(aside)["scopes"]
    assert _REJECTED in scopes["beta"]["content"]
    assert _REJECTED not in scopes["alpha"]["content"]
    assert "full entry: beta.md" in scopes["alpha"]["content"]


def test_without_authorization_a_reorder_is_reported_and_left_alone(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """R3 — the day-after control: non-TTY, no `--yes`, no named target.

    A reorder is a reconcilable divergence, so it enters the reconcile gate — and that
    gate's authorization rules are unchanged: it prints the diff, skips visibly, and
    touches nothing.
    """
    config, manager, _ = env
    assert not sys.stdin.isatty()
    _seed(config, manager)
    before = _state(manager)
    _reorder_buffer(config)
    capsys.readouterr()

    manager.perform_sync(auto_accept=False)

    out = capsys.readouterr().out
    assert _state(manager) == before, "nothing reconciled without authorization"
    assert any("no terminal to confirm on" in line for line in out.splitlines()), out
    assert _REJECTED in _axiom_file(config, "alpha")


def test_the_status_rung_routes_scope_rows_on_the_order_only_flag(capsys) -> None:
    """R4 — membership is a findability defect; an order-only row is not.

    The fixture's flags deliberately DISAGREE with their lists: the row whose lists
    differ in membership is flagged order-only, and the reordered one is not. So the
    routing below can only come from reading the flag — a rung that recomputed the
    class from the lists would put each row under the other bullet.
    """
    from mitos.cli import _print_divergence_rung

    report = {
        "checked": 2, "skipped": None, "cache_hit": False,
        "commentary": [], "graph_only": [], "edges": [], "source": [],
        "reconcilable": 2, "archived_drift": 0,
        "scope": [
            {"slug": "routed-as-membership", "file": "decisions.md",
             "graph": ["alpha", "beta"], "markdown": ["beta", "alpha"], "order_only": False},
            {"slug": "routed-as-order", "file": "decisions.md",
             "graph": ["alpha"], "markdown": ["gamma"], "order_only": True},
        ],
    }
    _print_divergence_rung(report, project="demo")
    lines = capsys.readouterr().out.splitlines()

    bullets = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("• ")]
    membership = [i for i in bullets if "scope tags differ" in lines[i]]
    order = [i for i in bullets if "ordered differently" in lines[i]]
    assert len(membership) == 1 and len(order) == 1, lines

    assert "1 entry(s)" in lines[membership[0]]
    assert "FINDABILITY" in lines[membership[0]]
    assert "routed-as-membership" in lines[membership[0] + 1]

    assert "1 entry(s)" in lines[order[0]]
    assert "FINDABILITY" not in lines[order[0]], "an order-only row hides nothing"
    assert "still finds them" in lines[order[0]]
    assert "routed-as-order" in lines[order[0] + 1]
