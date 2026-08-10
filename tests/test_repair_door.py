"""The targeted repair door: the guard that makes it reachable, and the flag that opens it.

Two phases, in that order — **3a's guard** below the first divider, **3b's
`--reconcile-entry` flag** below the second. The split is the deliverable's own
seam: reachability before authorization. Read the 3b banner for that half's own
posture note (every row there that commits must wire the embed/vector fakes; 3a's
nine never commit and never needed to).

Phase 3a — the accept-prompt guard that makes the targeted repair door reachable.

`mitos sync` held three `input()` calls on the per-entry path and exactly one `isatty()`
guard — the commentary reconcile's. The two accept prompts above it were gated by
`not auto_accept` alone, so a non-interactive run carrying any *pending* buffer entry died
at `input()` with `EOF when reading a line`, rendered as `Fatal Unexpected Error`, exit 1 —
**before the reconcile was ever reached**. The narrow, surgical repair therefore could not
be taken by anyone who could not answer a prompt, while the wholesale `mitos rebuild --yes`
was not gated at all: the shape of the guard routed agents around the careful door and
toward the sledgehammer.

This module pins the guard that replaces that crash with a report-and-skip. Three
properties are the contract, not incidental:

* **One refusal, above the kind split** — so it dominates BOTH accept prompts. A guard
  written inside either branch is one chance in two to ship a door still dead on the other
  kind, which is why there is a row here per kind and a two-entry row over both.
* **Above the entry's conflict sensor, by construction.** The crash is what bounded the
  sync-time sensor's per-entry loop at one entry; removing the crash removes the bound, so
  a guard placed at the `input()` would let a run that can accept *nothing* sweep a paid
  judgment across the whole pending buffer. That is a spend contract, and only the
  judge-injected row below reds on the wrong placement.
* **Exit stays 0.** A skipped pending entry stays in `decisions.md`, which sync never
  deletes from, so `--yes` commits it next run. A skip costs a turn, not an entry — which
  is what makes exit 0 correct rather than merely convenient.

Home for the door's rows generally, and **phase 3b did extend it** with the named-target
repair flag's set (below). Fixtures come from `_conflict_helpers` — `offline` for the key injection
(the key floor returns above the per-entry loop, so a keyless fixture never reaches the
accept prompt at all and every must-fail-first row here would pass green on the pre-fix
build) — and pointedly **not** `interactive_stdin`: non-TTY is this module's subject, and
under pytest's capture it is already the default posture. Every non-TTY row nonetheless
*states* it with an `assert not sys.stdin.isatty()`, so a run that quietly acquired a
terminal (`-s` from a real shell) fails on the posture instead of blocking at a prompt.
The one TTY row opts in locally.
"""

import os
import sys
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from filelock import Timeout

from mitos.config import MitosConfig
from mitos.sync import MitosSyncManager

from conftest import make_workspace
from test_cli_selector import _run

from _conflict_helpers import (
    _RecordingJudge,
    _append_decision,
    _execution,
    _wire_fakes,
    env,
    offline,
)


#: The guard's own stem. Every text assertion here is scoped to the lines carrying it,
#: never to the whole capture: the sync body legitimately prints `Proposed Decision:`,
#: `[Collision]`, `[Divergence]` and `Skipped` on neighbouring paths, so a whole-capture
#: `in` check passes for the wrong reason. Deliberately distinct from the reconcile
#: refusal's `"no terminal to confirm on"` — a shared phrase would make
#: `test_sync.py::test_sync_without_a_tty_and_without_yes_skips_instead_of_dying`
#: ambiguous about which of the two gates fired.
_GUARD_STEM = "stdin is not a terminal"


def _refusal_lines(out: str) -> List[str]:
    """The guard's refusal lines, lifted out of a capture the sync body also writes to."""
    return [line for line in out.splitlines() if _GUARD_STEM in line]


def _write_open_question(config: MitosConfig, slug: str, topic: str, question: str) -> None:
    """Authors one pending open question into `questions.md`.

    `_append_decision` writes decisions only, and an open question's block format differs
    (`### <slug>` / `**Topic:**` / `**Questions:**` with a `-` list). The shape is copied
    from `test_sync.py::test_an_open_question_divergence_does_not_reconcile_forever`.
    """
    with open(config.questions_file, "w", encoding="utf-8") as f:
        f.write(
            "# Questions\n<!-- BEGIN ENTRIES -->\n\n"
            f"### {slug}\n\n"
            f"**Topic:** {topic}\n**Questions:**\n- {question}\n"
        )


# --------------------------------------------------------------------------- #
# G1/G2 — the crash becomes a sentence, on BOTH kinds
# --------------------------------------------------------------------------- #

def test_a_pending_decision_is_reported_and_skipped_without_a_terminal(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """The shipped-behaviour flip: a run that used to die at `input()` reports and skips."""
    config, manager, _ = env
    # The subject, stated rather than assumed — pytest's capture already gives a non-TTY
    # stdin, and a row that silently started running interactively would prove nothing.
    assert not sys.stdin.isatty()
    _append_decision(config, "door-decision", "The pending axiom nobody could accept.")

    manager.perform_sync(auto_accept=False)  # must not raise

    lines = _refusal_lines(capsys.readouterr().out)
    assert len(lines) == 1, lines
    assert "door-decision" in lines[0], "the refusal must name the entry it skipped"
    assert manager.store.get_node_by_slug("door-decision") is None, "nothing committed"
    with open(config.decisions_file, encoding="utf-8") as f:
        assert "door-decision" in f.read(), (
            "the skipped entry stays in the buffer — that is what makes exit 0 honest"
        )


def test_a_pending_open_question_is_reported_and_skipped_without_a_terminal(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """The row that reds a guard written for the decision branch alone.

    Both accept prompts are branches of one `if entry.kind == "decision":`, so the only
    position covering both is above the split. A guard told to fix "the accept prompt"
    guards the decision branch, passes every row about decisions, and ships a door still
    dead on any buffer holding a pending open question — uniformly short by one.

    The refusal-count assertion is also this row's non-vacuity proof, and that is not
    theoretical here: `test_conflict_sync.py::test_open_question_entry_is_never_checked`
    authors its open question in the *decision* header shape, so the parser drops it with
    `Missing required field **Topic:**` and the row asserts an absence over a run holding
    zero entries. A row whose fixture must PRODUCE a line cannot fail that way — if this
    block stopped parsing, `len(lines) == 1` reds immediately.
    """
    config, manager, _ = env
    assert not sys.stdin.isatty()
    _write_open_question(config, "door-question", "which embedding model",
                         "Which dimension?")

    manager.perform_sync(auto_accept=False)  # must not raise

    lines = _refusal_lines(capsys.readouterr().out)
    assert len(lines) == 1, lines
    assert "door-question" in lines[0]
    assert manager.store.get_node_by_slug("door-question") is None


# --------------------------------------------------------------------------- #
# G3 — the position row: the sensor never fires for a skipped entry
# --------------------------------------------------------------------------- #

def test_a_skipped_entry_never_reaches_the_conflict_sensor(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """The guard's POSITION, which is a spend contract rather than a placement detail.

    The sync-time conflict sensor fires one statement above the decision accept prompt. The
    crash is what bounded its per-entry loop at one entry; the guard removes that bound, so
    a guard placed at the `input()` would let a run that can accept nothing sweep an
    Anthropic judgment across the whole pending buffer — a buffer whose size is the
    corpus's, not one entry's. Placed above the kind split, the sensor is dominated by
    construction and the deferred judgment lands on `mitos check`'s batched, verdict-reusing,
    spend-ringed tier instead.

    The judge MUST be injected, or this row passes on the build it exists to catch:
    `_build_conflict_judge` returns `None` without `ANTHROPIC_API_KEY` (which `offline`
    deletes) and without live embed/vector components, so the sensor could not fire whatever
    the guard's placement. `builtins.input` is deliberately NOT patched — a patched `input`
    would make a mis-placed guard survive rather than crash, and non-TTY is the subject.
    """
    config, manager, _ = env
    assert not sys.stdin.isatty()
    judge = _RecordingJudge(_execution([]))
    vector = _wire_fakes(manager, judge=judge)
    spy = MagicMock()
    manager._run_and_surface_conflict = spy  # type: ignore[assignment]

    _append_decision(config, "door-sensor", "The axiom no judge should ever see.")

    manager.perform_sync(auto_accept=False)

    assert len(_refusal_lines(capsys.readouterr().out)) == 1, "the guard must have fired"
    assert spy.call_count == 0, "the sensor ran for an entry that could never be accepted"
    # Two further independent statements that no paid work fired, so the row does not rest
    # on the spy alone: no vector round trip, no judgment.
    assert vector.queries == 0
    assert judge.calls == 0


# --------------------------------------------------------------------------- #
# G4/G5 — both conjuncts of the predicate are load-bearing
# --------------------------------------------------------------------------- #

def test_auto_accept_is_untouched_by_the_guard(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """`--yes` on a non-TTY run commits exactly as before — the `not auto_accept` conjunct.

    Green on both sides of the fix, and that is the point: a guard written on
    `not sys.stdin.isatty()` alone kills every `mitos sync --yes` in CI and cron, which
    every non-TTY row above is structurally blind to.
    """
    config, manager, _ = env
    assert not sys.stdin.isatty()
    _append_decision(config, "door-yes", "The axiom `--yes` still commits.")

    manager.perform_sync(auto_accept=True)

    assert _refusal_lines(capsys.readouterr().out) == []
    assert manager.store.get_node_by_slug("door-yes") is not None


@patch("builtins.input", side_effect=["a"])
@patch("sys.stdin.isatty", return_value=True)
def test_a_real_terminal_still_reaches_the_accept_prompt(
    mock_isatty: MagicMock, mock_input: MagicMock,
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """At a terminal the prompt appears and answers as before — the `isatty` conjunct.

    Opts into the TTY posture locally rather than importing `interactive_stdin`: that
    fixture is autouse in whichever module imports it, and a TTY default here would flip
    every non-TTY row in this module into passing for the wrong reason.
    """
    config, manager, _ = env
    _append_decision(config, "door-tty", "The axiom a human accepted.")

    manager.perform_sync(auto_accept=False)

    assert _refusal_lines(capsys.readouterr().out) == []
    assert manager.store.get_node_by_slug("door-tty") is not None
    assert mock_input.call_count == 1, "the accept prompt must still be the thing that ran"


# --------------------------------------------------------------------------- #
# G6 — the shipped exit flip, at the process boundary
# --------------------------------------------------------------------------- #

def test_the_verb_exits_zero_instead_of_dying_on_the_prompt(
    tmp_path, capsys: pytest.CaptureFixture
) -> None:
    """`mitos sync` over a pending buffer with no terminal: exit 0, not `Fatal Unexpected Error`.

    The one row driven through `cli.main()` rather than the manager, because the contract
    it pins is the process's — today this run exits 1 with a crash banner that names no
    entry, and an agent meeting it has learned only that the safe path is broken.

    The workspace is built by `conftest.make_workspace`, not by the `env` fixture's tmpdir:
    the selector is mandatory since 0.15.0 and `routing.is_workspace` requires BOTH
    `.mitos/config.toml` and `decisions.md`, while `env` writes only the second — so a
    `-p <env tmpdir>` run would resolve to a targeting error and assert exit 0 against a run
    that never reached the sync loop at all. `make_workspace`'s `decisions.md` carries no
    `BEGIN ENTRIES` sentinel, which is safe: the parser defaults to the file head when the
    sentinel is absent, so an appended block still parses. The `offline` fixture's
    `GEMINI_API_KEY` reaches the config `main()` builds through the resolver's real-env
    tier, clearing the key floor.
    """
    workspace = make_workspace(tmp_path / "ws")
    with open(os.path.join(workspace, "decisions.md"), "a", encoding="utf-8") as f:
        f.write(
            "\n## 2026-06-01 — door-exit — Door Exit\n"
            "**Decided:** The axiom that used to kill the process.\n"
            "**Rejected:** Rejected the obvious alternative.\n"
            "**Mechanisms:** python\n"
            "**Scope:** api\n\n"
        )
    assert not sys.stdin.isatty()

    code = _run(["sync", "-p", workspace])

    captured = capsys.readouterr()
    assert code in (0, None), f"exit {code!r}; stderr was:\n{captured.err}"
    assert "Fatal Unexpected Error" not in captured.err
    assert len(_refusal_lines(captured.out)) == 1, captured.out


# --------------------------------------------------------------------------- #
# G7 — the naming obligation at N > 1
# --------------------------------------------------------------------------- #

def test_two_pending_entries_produce_two_refusals_each_naming_its_own_slug(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """One line per skipped entry, and each names ITS entry.

    Copied verbatim from the reconcile's refusal — which is indented under its own
    `[Divergence] '<slug>'` header and so needs no slug of its own — the line would report
    N anonymous skips over an imported buffer: strictly less diagnosable than the crash it
    replaces, which printed the slug one line before dying. The two slugs are visibly
    distinct so a per-entry naming bug cannot hide behind a shared string, and the entries
    are one of each kind so the row also covers both branches at once.
    """
    config, manager, _ = env
    assert not sys.stdin.isatty()
    _append_decision(config, "door-alpha-decision", "The first pending axiom.")
    _write_open_question(config, "door-omega-question", "the second pending topic",
                         "Which dimension?")

    manager.perform_sync(auto_accept=False)

    lines = _refusal_lines(capsys.readouterr().out)
    assert len(lines) == 2, lines
    # Each slug on exactly one line — a plain membership check over the pair would be
    # satisfied by two lines that both named the first entry.
    for slug in ("door-alpha-decision", "door-omega-question"):
        assert sum(slug in line for line in lines) == 1, lines


# --------------------------------------------------------------------------- #
# G8 — the refusal's own text (the recipe + register floor)
# --------------------------------------------------------------------------- #

def test_the_refusal_names_the_flag_and_no_command(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """Flag and no command; calm register.

    Asserted against the emitted line rather than a re-declared literal, so the row cannot
    drift away from what ships. The recipe rule reaches this line — it is the one sentence
    in this phase authored from nothing — and the shipped sibling discharges it by shape:
    a flag is a flag on any surface, while a spelled-out `mitos …` command would owe a
    selector since 0.15.0 and would be a state-changing shell command handed to an agent.

    The no-command assertion pins the CURRENT truth as a negative. It stays true after 3b:
    the named-target repair flag satisfies the *reconcile* gate, not the accept prompt, so
    a pending named target is still guard-skipped — and a later phase that wanted to name a
    command here would have to consciously invert this row.
    """
    config, manager, _ = env
    assert not sys.stdin.isatty()
    _append_decision(config, "door-text", "The axiom whose refusal we read.")

    manager.perform_sync(auto_accept=False)

    lines = _refusal_lines(capsys.readouterr().out)
    assert len(lines) == 1, lines
    line = lines[0]
    assert "`--yes`" in line, "the refusal must name the flag that gets past it"
    assert "mitos " not in line, "a refusal that spells a command owes a selector"
    for shout in ("CRITICAL", "NEVER", "WARNING", "⚠"):
        assert shout not in line, f"prompt-style discipline: {shout!r} in {line!r}"
    assert line == line.strip(), "one line, no block, no header plus body"


# --------------------------------------------------------------------------- #
# G9 — the rotation prompt stays unreachable, pinned rather than trusted
# --------------------------------------------------------------------------- #

def test_a_guard_skipped_run_never_reaches_the_rotation_prompt(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """The third `input()` in the loop stays unreachable after the guard lands.

    `pending_threshold` is lowered to 1 so the rotation gate would open on a single commit
    — the cheapest way to prove the guard, not the threshold, is what closes it.

    The mechanic is two steps, and the second is the load-bearing one. `synced_blocks` has
    two append sites: the in-loop one is below the accept prompt (and gated on
    decision-kind), so a guard-skipped entry never reaches it; the other lives inside
    `_commit_quarantine_fixpoint`, which runs AFTER the loop and BEFORE the rotation gate —
    *not* below the prompt at all. It is closed by a second step instead: the fixpoint only
    ever re-tries entries in `quarantined`, whose sole append also sits below the prompt, so
    a guard-skipped entry can never enter the set the fixpoint replays. Under `--yes` the
    guard never fires and `not auto_accept` closes the gate from the other side.
    """
    config, manager, _ = env
    assert not sys.stdin.isatty()
    config.pending_threshold = 1
    _append_decision(config, "door-rotate", "The axiom that must not rotate.")

    manager.perform_sync(auto_accept=False)  # must not raise at the rotation prompt

    out = capsys.readouterr().out
    assert len(_refusal_lines(out)) == 1
    assert "[Lifecycle]" not in out, "the rotation prompt's own announcement"
    with open(config.decisions_file, encoding="utf-8") as f:
        buffer_after = f.read()
    assert "door-rotate" in buffer_after, "the entry is still pending, in the buffer"
    assert not os.path.exists(config.archive_dir), "nothing was archived"

    # And the buffer has reached its fixpoint: the header auto-heal ran on the first sync,
    # so a second run over the same still-pending entry leaves the file byte-identical.
    manager.perform_sync(auto_accept=False)
    with open(config.decisions_file, encoding="utf-8") as f:
        assert f.read() == buffer_after


# =========================================================================== #
# Phase 3b — the named-target door: `mitos sync --reconcile-entry <SLUG>`
# =========================================================================== #
#
# 3a made the command reachable; these rows pin the authorization. The flag is an
# additional way to satisfy the SHIPPED reconcile gate for the entries it names —
# never a second apply path — so almost every assertion below is about what the
# flag does NOT change: the unnamed entry, the refusal for it, the exit for a bare
# run, and the state a fault leaves behind.
#
# **Posture for every 3b row that reconciles or commits: wire the fakes.** 3a's
# nine rows never commit, so none of them needed it. A reconcile's commit enqueues
# to the outbox and `_perform_sync_internal` ends in `drain_pending_embeddings()`,
# which reaches `generativelanguage.googleapis.com` for real with the `offline`
# fixture's `mock_key` — a live round trip per committed node inside a tier that
# claims to make none. `_no_network()` below is this module's standing answer.

#: The repair report's own stem — a sibling of `_GUARD_STEM`, deliberately NOT a
#: widening of `_refusal_lines`. The two gates print different sentences on
#: neighbouring paths and 3a's rows assert exact line counts against theirs.
_REPAIR_STEM = "[Repair]"

#: The rewritten reconcile refusals' stem. Kept distinct from `_GUARD_STEM` for the
#: same reason 3a made them distinct: `test_sync.py`'s non-TTY row asserts on
#: `"no terminal"` and must stay unambiguous about which of the two gates fired.
_RECONCILE_STEM = "no terminal to confirm on"


def _repair_lines(out: str) -> List[str]:
    """The repair report's lines, lifted out of a capture the sync body also writes."""
    return [line for line in out.splitlines() if _REPAIR_STEM in line]


def _no_network(manager: MitosSyncManager) -> None:
    """Replaces the embed + vector seams so a committing row makes no real call."""
    _wire_fakes(manager, judge=_RecordingJudge([]))


def _seed_committed_buffer(
    config: MitosConfig, manager: MitosSyncManager, *, slug: str = "reconcile-me",
    supersedes: Optional[str] = None, amends: Optional[str] = None,
    cites: Optional[str] = None,
    rejected: str = "The original rejected reasoning.",
    scope: Optional[List[str]] = None,
) -> None:
    """Commits one entry through `record`, leaving it IN the buffer for a re-sync.

    Lifted in shape from `test_sync.py::_seed_committed_buffer` (the C′ section)
    rather than imported across modules, and widened to the relations these rows
    need. Deliberately `record_decision_entry` and never `perform_sync`: rotation
    is tied to a first sync commit, so a sync-authored entry leaves the buffer
    immediately and is no longer reconcilable (its reconciler is `rebuild`).
    `acknowledge_neighbors=True` skips the neighbour-review pause.

    The block it writes uses the `### <slug>` header shape with `**Decided:** /
    **Rejected:** / **Mechanisms:** / **Scope:**` lines, with any relation line
    LAST — different from `_append_decision`'s `## date — slug — Title`, so a
    `_edit_buffer` literal must match whichever one actually wrote the entry.

    The axiom carries the slug, and that is load-bearing rather than cosmetic: a
    node id is the hash of `{kind, axiom, mechanism_refs}` and the slug is NOT in
    it, so two seeds sharing an axiom converge to ONE node (M2) — the second
    `record` becomes a no-op re-encounter that writes no block and mints no edge,
    and every assertion downstream then fails somewhere far from the cause.
    """
    result = manager.record_decision_entry(
        f"The reconcilable axiom for {slug}.", rejected,
        scope if scope is not None else ["alpha"],
        mechanisms=["sqlite"], slug=slug,
        supersedes=supersedes, amends=amends, cites=cites,
        acknowledge_neighbors=True,
    )
    assert result.get("state") == "active", result


def _edit_buffer(config: MitosConfig, old: str, new: str) -> None:
    """Rewrites one literal in the buffer, refusing a no-op edit.

    The assert is copied deliberately: a silent no-op edit makes a whole row
    vacuous — the sync finds nothing diverged and every assertion about the
    refusal passes for the wrong reason.
    """
    with open(config.decisions_file, "r", encoding="utf-8") as f:
        text = f.read()
    assert old in text, f"fixture edit target missing: {old!r}"
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write(text.replace(old, new))


def _edges_of(manager: MitosSyncManager, slug: str) -> List[Tuple[str, str]]:
    """The outgoing (kind, target-slug) pairs for one node, by slug."""
    node = manager.store.get_node_by_slug(slug)
    assert node is not None, f"{slug!r} is not in the active view"
    return sorted(
        (e["kind"], e.get("target") or e.get("target_slug"))
        for e in manager.store.get_outgoing_edges(node["id"])
    )


# --------------------------------------------------------------------------- #
# R1 — the happy path, and the MI-5 + MI-13 walks that ride on it
# --------------------------------------------------------------------------- #

def test_a_named_target_applies_its_whole_reconcile_without_a_terminal(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """The flip this phase exists for: the deletion lands, unattended, exit-clean.

    Today the same run reports and skips. Three properties ride here because they
    are properties of this one transaction:

    * **MI-5** — an accepted reconcile commits exactly the edges the markdown
      declares and no others. The entry carries BOTH an addition and a removal, so
      the assertion is a set equality rather than a count.
    * **MI-13's second half** — a removed kill-edge line does not merely drop an
      edge, it RESURRECTS its target into the active view, with no human present,
      for the first time. `victim` is superseded before and active after. Its id is
      captured BEFORE the supersession because `get_node_by_slug` is active-view
      only and returns `None` for a superseded node.
    * The shortfall is empty, which is what `cmd_sync` turns into exit 0.
    """
    config, manager, _ = env
    assert not sys.stdin.isatty()
    _no_network(manager)
    _seed_committed_buffer(config, manager, slug="victim")
    _seed_committed_buffer(config, manager, slug="bystander")
    victim_id = manager.store.get_node_by_slug("victim")["id"]
    _seed_committed_buffer(config, manager, slug="reconcile-me", supersedes="victim")
    assert manager.store.get_node_state(victim_id) == "superseded"

    # The mixed repair: drop the kill edge, add a citation, correct the commentary.
    _edit_buffer(config, "**Supersedes:** victim\n", "**Cites:** bystander\n")
    _edit_buffer(config, "The original rejected reasoning.\n**Mechanisms:** sqlite",
                 "The CORRECTED reasoning.\n**Mechanisms:** sqlite")

    shortfall = manager.perform_sync(
        auto_accept=False, repair_targets=["reconcile-me"]
    )

    out = capsys.readouterr().out
    assert shortfall == [], "a landed repair is not a shortfall"
    assert "Reconciled 'reconcile-me' ✓" in out
    assert _repair_lines(out) == [], "a satisfied target owes no extra line"
    assert _edges_of(manager, "reconcile-me") == [("cites", "bystander")], (
        "MI-5: exactly the declared edge set, neither more nor less"
    )
    node = manager.store.get_node_by_slug("reconcile-me")
    assert node["rejected_paths"] == "The CORRECTED reasoning."
    assert manager.store.get_node_state(victim_id) == "active", (
        "MI-13: removing the kill-edge line resurrects its target"
    )


def test_a_named_target_whose_only_divergence_is_a_dropped_scope_tag(
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """MI-9's walk: the scope rows go to zero, and no `''` sentinel row appears.

    `commit_parsed_entry` runs an insert-missing/delete-absent pass over
    `node_scopes`, and `entry_divergence` reports scope as its own species — so a
    hand-edit dropping a `**Scope:**` tag DELETES that row with no human present.
    MI-9's floor is zero rows, never an empty-string sentinel.
    """
    config, manager, _ = env
    _no_network(manager)
    _seed_committed_buffer(config, manager, slug="scoped", scope=["alpha", "beta"])
    assert sorted(manager.store.get_node_by_slug("scoped")["scope"]) == ["alpha", "beta"]

    _edit_buffer(config, "**Scope:** alpha, beta", "**Scope:** alpha")

    assert manager.perform_sync(auto_accept=False, repair_targets=["scoped"]) == []

    scopes = manager.store.get_node_by_slug("scoped")["scope"]
    assert scopes == ["alpha"], scopes
    assert "" not in scopes, "MI-9: zero rows, never an empty-string sentinel"


# --------------------------------------------------------------------------- #
# R2 — narrowness: the flag reaches the entry it names and stops there
# --------------------------------------------------------------------------- #

def test_the_flag_reaches_only_the_entry_it_names(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """Two diverged entries with removals, one named: the other behaves as today.

    The property a phase can break without noticing, because every happy-path row
    still passes on the build that leaks: the authorization must not survive into
    the next iteration of the loop.
    """
    config, manager, _ = env
    assert not sys.stdin.isatty()
    _no_network(manager)
    _seed_committed_buffer(config, manager, slug="target-a")
    _seed_committed_buffer(config, manager, slug="target-b")
    _seed_committed_buffer(config, manager, slug="named", cites="target-a")
    _seed_committed_buffer(config, manager, slug="unnamed", cites="target-b")
    _edit_buffer(config, "**Cites:** target-a\n", "")
    _edit_buffer(config, "**Cites:** target-b\n", "")

    shortfall = manager.perform_sync(
        auto_accept=False, repair_targets=["named"]
    )

    out = capsys.readouterr().out
    assert shortfall == [], "the unnamed entry is not this run's business"
    assert _edges_of(manager, "named") == [], "the named target's edge is gone"
    assert _edges_of(manager, "unnamed") == [("cites", "target-b")], (
        "the unnamed entry is refused exactly as it is today"
    )
    refusals = [line for line in out.splitlines() if _RECONCILE_STEM in line]
    assert len(refusals) == 1, refusals
    assert "Reconciled 'named' ✓" in out
    assert "Reconciled 'unnamed' ✓" not in out


# --------------------------------------------------------------------------- #
# R3 — it satisfies BOTH gate branches, and stands alone
# --------------------------------------------------------------------------- #

def test_the_flag_satisfies_the_auto_accept_branch(
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """`--yes` beside the flag changes nothing: the same named target reconciles."""
    config, manager, _ = env
    _no_network(manager)
    _seed_committed_buffer(config, manager, slug="anchor")
    _seed_committed_buffer(config, manager, slug="reconcile-me", cites="anchor")
    _edit_buffer(config, "**Cites:** anchor\n", "")

    assert manager.perform_sync(
        auto_accept=True, repair_targets=["reconcile-me"]
    ) == []
    assert _edges_of(manager, "reconcile-me") == []


@patch("builtins.input", side_effect=AssertionError("the prompt must not be reached"))
@patch("sys.stdin.isatty", return_value=True)
def test_the_flag_suppresses_the_prompt_at_a_real_terminal(
    mock_isatty: MagicMock, mock_input: MagicMock,
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """At a TTY the named target reconciles WITHOUT being asked.

    The prompt is suppressed, not answered — so `builtins.input` is never called.
    An `input` fake that returned `"r"` would pass on a build that still prompts,
    which is the whole reason this one raises instead.
    """
    config, manager, _ = env
    assert sys.stdin.isatty()
    _no_network(manager)
    _seed_committed_buffer(config, manager, slug="anchor")
    _seed_committed_buffer(config, manager, slug="reconcile-me", cites="anchor")
    _edit_buffer(config, "**Cites:** anchor\n", "")

    assert manager.perform_sync(
        auto_accept=False, repair_targets=["reconcile-me"]
    ) == []
    assert mock_input.call_count == 0, "the prompt was reached, not suppressed"
    assert _edges_of(manager, "reconcile-me") == []


# --------------------------------------------------------------------------- #
# R4 — the three refusal classes, each naming its own state
# --------------------------------------------------------------------------- #

def test_an_unparseable_named_target_names_the_parse_failure(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """A named entry the parser rejected: non-zero, and never read as *not diverged*."""
    config, manager, _ = env
    _append_decision(config, "broken-entry", "An axiom with no rejected paths.")
    _edit_buffer(config, "**Rejected:** Rejected the obvious alternative.\n", "")

    shortfall = manager.perform_sync(
        auto_accept=False, repair_targets=["broken-entry"]
    )

    lines = _repair_lines(capsys.readouterr().err)
    assert shortfall == ["broken-entry"]
    assert len(lines) == 1, lines
    assert "failed to parse" in lines[0], lines[0]
    assert "already agree" not in lines[0], "it must never read as *not diverged*"


def test_an_archived_named_target_names_rebuild_as_its_heal(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """A target `sync` cannot reach: the never-seen class, healed by `mitos rebuild`.

    `sync` snapshots the buffer alone, so an entry that has rotated into an archive
    is out of its read-set entirely — indistinguishable, from inside the loop, from
    one that was never authored. The class is stated as a property covering both,
    which is why one row proves it: the refusal names the archive AND `rebuild`,
    and `status`'s own rung prints the same heal one statement below the sentence
    this phase rewrites, so a refusal saying only "absent" would contradict the
    report that handed the caller the slug.
    """
    config, manager, _ = env
    _no_network(manager)
    _seed_committed_buffer(config, manager, slug="present-one")

    shortfall = manager.perform_sync(
        auto_accept=False, repair_targets=["long-since-archived"]
    )

    lines = _repair_lines(capsys.readouterr().err)
    assert shortfall == ["long-since-archived"]
    assert len(lines) == 1, lines
    assert "mitos rebuild" in lines[0], lines[0]
    assert "archive" in lines[0], lines[0]


def test_a_named_open_question_names_the_kind_gate(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """A committed OPEN QUESTION named as a repair target: non-zero, not exit 0.

    It is present and parsed, so the literal predicate sends it to 0 — but
    divergence was never COMPUTED for it, because the kind gate fires above
    `entry_divergence`. Exit 0 on a corpus that provably cannot converge is the
    silent no-op the fail-loud property forbids.
    """
    config, manager, _ = env
    _no_network(manager)
    _write_open_question(config, "oq-target", "The topic.", "The question?")
    manager.perform_sync(auto_accept=True)
    assert manager.store.get_node_by_slug("oq-target") is not None, (
        "the fixture must COMMIT the open question — an unparsed one would land in "
        "the never-seen class instead and prove nothing about the kind gate"
    )
    capsys.readouterr()

    shortfall = manager.perform_sync(
        auto_accept=True, repair_targets=["oq-target"]
    )

    lines = _repair_lines(capsys.readouterr().err)
    assert shortfall == ["oq-target"]
    assert len(lines) == 1, lines
    assert "open question" in lines[0], lines[0]
    assert "decisions-only" in lines[0], lines[0]


# --------------------------------------------------------------------------- #
# R5 — refused at the commit: the reconcile is named as the gate
# --------------------------------------------------------------------------- #

def test_a_named_target_whose_commit_is_foreclosed_exits_non_zero(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """Authorization satisfies the gate; it does not make a foreclosed edge legal.

    The entry declares an ADDED citation naming nothing in the graph, so
    `_uncommittable_edges` refuses after the authorization branch — one of the
    three `False` returns that survive for an authorized target. The located cause
    is printed above (on stderr, like all three), so the post-loop line names the
    reconcile as the gate rather than re-deriving why.
    """
    config, manager, _ = env
    _no_network(manager)
    _seed_committed_buffer(config, manager, slug="reconcile-me")
    _edit_buffer(config, "**Mechanisms:** sqlite",
                 "**Cites:** no-such-decision\n**Mechanisms:** sqlite")

    shortfall = manager.perform_sync(
        auto_accept=False, repair_targets=["reconcile-me"]
    )

    err = capsys.readouterr().err
    assert shortfall == ["reconcile-me"]
    assert "does not name any entry in the graph" in err, "the located cause"
    lines = _repair_lines(err)
    assert len(lines) == 1, lines
    assert "reconcile" in lines[0], lines[0]
    assert _edges_of(manager, "reconcile-me") == [], "nothing was committed"


# --------------------------------------------------------------------------- #
# R6 — the re-run, which the literal predicate fails
# --------------------------------------------------------------------------- #

def test_the_same_invocation_a_second_time_exits_zero(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """A landed repair must not redden forever.

    The reconcile `continue`s past rotation, so the repaired entry STAYS in the
    buffer and re-parses next run as committed-and-clean — the satisfied
    not-diverged state, exit 0. A build that read the predicate without its bound
    would fail the second run for the rest of the entry's life in the buffer.
    """
    config, manager, _ = env
    _no_network(manager)
    _seed_committed_buffer(config, manager, slug="anchor")
    _seed_committed_buffer(config, manager, slug="reconcile-me", cites="anchor")
    _edit_buffer(config, "**Cites:** anchor\n", "")

    assert manager.perform_sync(
        auto_accept=False, repair_targets=["reconcile-me"]) == []
    capsys.readouterr()

    second = manager.perform_sync(
        auto_accept=False, repair_targets=["reconcile-me"])

    lines = _repair_lines(capsys.readouterr().err)
    assert second == [], "the second run must exit clean"
    assert len(lines) == 1 and "already agree" in lines[0], lines
    assert "[Repair]" in lines[0], (
        "the clean case still carries its own in-band line, so a human sees why "
        "their target was not touched"
    )


# --------------------------------------------------------------------------- #
# R7 — the pending pair: the flag authorizes the reconcile, not the prompt
# --------------------------------------------------------------------------- #

def test_a_pending_named_target_is_guard_skipped_and_exits_non_zero(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """Naming a PENDING entry does not accept it — 3a's guard still skips it.

    The refusal 3a wrote names `--yes` and no command, and stays true: this flag
    satisfies the *reconcile* gate, not the accept prompt. The exit is non-zero
    because the caller asked for a state the corpus did not reach.
    """
    config, manager, _ = env
    assert not sys.stdin.isatty()
    _append_decision(config, "still-pending", "The axiom nobody accepted.")

    shortfall = manager.perform_sync(
        auto_accept=False, repair_targets=["still-pending"]
    )

    captured = capsys.readouterr()
    assert shortfall == ["still-pending"]
    assert len(_refusal_lines(captured.out)) == 1, "3a's guard fired, unchanged"
    lines = _repair_lines(captured.err)
    assert len(lines) == 1 and "still pending" in lines[0], lines
    assert manager.store.get_node_by_slug("still-pending") is None


def test_a_pending_named_target_under_yes_commits_and_exits_zero(
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """The other half of the pending pair: committed IS a satisfied state."""
    config, manager, _ = env
    _no_network(manager)
    _append_decision(config, "still-pending", "The axiom nobody accepted.")

    shortfall = manager.perform_sync(
        auto_accept=True, repair_targets=["still-pending"]
    )

    assert shortfall == []
    assert manager.store.get_node_by_slug("still-pending") is not None


def test_a_named_target_committed_by_the_fixpoint_is_satisfied(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """The satisfied-commit state is TWO sites, and this is the one a stamp misses.

    A forward-ref quarantines in the main pass and commits in the intra-sync
    fixpoint. A satisfied stamp placed only after the main pass's `Committed node:`
    would exit non-zero on a run that committed the entry — the exact inverse of
    the failure this flag exists to prevent, arriving through the flag itself.

    The ordering is the whole fixture and it is counter-intuitive twice over:
    `_append_decision` APPENDS, and the loop reads the buffer BOTTOM-UP, so the
    entry appended LAST is reached FIRST. `forward-ref` therefore has to be
    appended second to meet its citation before the target exists. Written the
    natural way round it commits in the main pass, the fixpoint never runs, and
    every assertion below passes over the wrong mechanism — measured, not feared.
    """
    config, manager, _ = env
    _no_network(manager)
    _append_decision(config, "authored-later", "The axiom cited from below.")
    # Written out rather than appended-then-edited: `_edit_buffer`'s `str.replace`
    # would match the FIRST block's field lines, which are byte-identical, and
    # silently give `authored-later` a self-citation instead (`cycle_violation`).
    with open(config.decisions_file, "a", encoding="utf-8") as f:
        f.write(
            "## 2026-06-01 — forward-ref — Forward Ref\n"
            "**Decided:** The axiom citing an earlier entry.\n"
            "**Rejected:** Rejected the obvious alternative.\n"
            "**Cites:** authored-later\n"
            "**Mechanisms:** python\n"
            "**Scope:** api\n\n"
        )

    shortfall = manager.perform_sync(
        auto_accept=True, repair_targets=["forward-ref"]
    )

    out = capsys.readouterr().out
    assert "[Fixpoint] converged 1 quarantined entry" in out, (
        "non-vacuity: this row means nothing unless the entry actually took the "
        f"quarantine → fixpoint route. Output was:\n{out}"
    )
    assert shortfall == [], "the fixpoint committed it, so it is satisfied"
    assert manager.store.get_node_by_slug("forward-ref") is not None


# --------------------------------------------------------------------------- #
# R8 — `--embed-only` may not discard the flag silently
# --------------------------------------------------------------------------- #

def test_reconcile_entry_with_embed_only_is_refused_at_the_process_boundary(
    tmp_path, capsys: pytest.CaptureFixture
) -> None:
    """`--embed-only` short-circuits above `perform_sync`, so the flag would vanish.

    Driven through `cli.main()` because the subject lives in `cmd_sync`, one frame
    above the file the rest of this phase's diff opens. Today the composition would
    drain the outbox and exit 0, having neither reconciled nor looked for the
    target — the silent no-op the fail-loud property forbids.
    """
    workspace = make_workspace(tmp_path / "ws")

    code = _run(["sync", "-p", workspace, "--reconcile-entry", "whatever",
                 "--embed-only"])

    captured = capsys.readouterr()
    assert code == 1, f"exit {code!r}; stderr was:\n{captured.err}"
    assert "--reconcile-entry" in captured.err and "--embed-only" in captured.err
    assert "Draining pending embeddings" not in captured.out, (
        "refused above the drain, not after it"
    )


def test_reconcile_entry_naming_nothing_is_refused_rather_than_ignored(
    tmp_path, capsys: pytest.CaptureFixture
) -> None:
    """`--reconcile-entry ""` is a supplied selector naming nothing, not an absent flag.

    The tree's rule since the selector flip: `-p ""` renders, it does not fall back
    to cwd. A flag present with zero usable handles is a refusal, because the
    alternative is a run that silently authorizes nothing and exits 0.
    """
    workspace = make_workspace(tmp_path / "ws")

    code = _run(["sync", "-p", workspace, "--reconcile-entry", " , "])

    captured = capsys.readouterr()
    assert code == 1, f"exit {code!r}; stderr was:\n{captured.err}"
    assert "named no entry" in captured.err, captured.err


def test_the_repeat_and_comma_spellings_accumulate_to_the_same_handles(
    tmp_path, capsys: pytest.CaptureFixture
) -> None:
    """1a's arity, pinned at the boundary that actually parses it.

    R9 below drives the manager directly, so it passes a Python list and proves
    nothing about `action="append"` or the comma split — the two spellings only
    exist above `cmd_sync`. Both failure modes 1a measured on this tree are silent:
    a bare `extend` iterates the string into CHARACTERS, and `nargs="*"` makes the
    space form parse and swallow the next token. Either ships a door that authorizes
    nothing while every manager-level row stays green.
    """
    workspace = make_workspace(tmp_path / "ws")

    with patch("mitos.cli.cmd_sync") as mock_sync:
        _run(["sync", "-p", workspace, "--reconcile-entry", "alpha, beta"])
        comma = mock_sync.call_args.kwargs["repair_targets"]
        mock_sync.reset_mock()
        _run(["sync", "-p", workspace,
              "--reconcile-entry", "alpha", "--reconcile-entry", "beta"])
        repeated = mock_sync.call_args.kwargs["repair_targets"]
        mock_sync.reset_mock()
        _run(["sync", "-p", workspace])
        absent = mock_sync.call_args.kwargs["repair_targets"]

    assert comma == ["alpha", "beta"], comma
    assert repeated == comma, "the two spellings are one contract"
    assert absent is None, (
        "an absent flag stays None — `[]` is a supplied flag naming nothing, and "
        "the handler tells them apart"
    )


# --------------------------------------------------------------------------- #
# R9 — many targets: document order, comma arity, one lands and one refuses
# --------------------------------------------------------------------------- #

def test_named_targets_apply_in_document_order_not_the_order_named(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """Order follows the loop's, which is `decisions.md`'s — never the caller's.

    Measured rather than reasoned: `record` PREPENDS, and `parse_file_reversed`
    reverses the newest-first buffer to oldest-first, so the loop reads the file
    BOTTOM-UP and the entry recorded FIRST is applied first. The fixture names them
    in the reverse of that, and the row asserts the reports came out in the loop's
    order. No surface may promise caller-order semantics.

    This row drives the manager, so the handles arrive as a list and the flag's
    arity is NOT what it proves — the row above owns that, at the boundary that
    parses it.
    """
    config, manager, _ = env
    _no_network(manager)
    _seed_committed_buffer(config, manager, slug="anchor-a")
    _seed_committed_buffer(config, manager, slug="anchor-b")
    _seed_committed_buffer(config, manager, slug="applies-first", cites="anchor-a")
    _seed_committed_buffer(config, manager, slug="applies-second", cites="anchor-b")
    # `applies-second` was recorded last, so it sits ABOVE `applies-first` in the
    # file and is reached LAST by the bottom-up loop. Asserted rather than assumed:
    # "document order" reads as top-to-bottom to a human and is the opposite here,
    # which is exactly the direction that produces a green-looking wrong fixture.
    text = open(config.decisions_file, encoding="utf-8").read()
    assert text.index("### applies-second") < text.index("### applies-first"), (
        "the fixture must put the named-first entry at the BOTTOM of the file"
    )
    # The first one lands: a plain edge removal.
    _edit_buffer(config, "**Cites:** anchor-a\n", "")
    # The second is refused at the commit: its replacement citation names nothing,
    # so `_uncommittable_edges` forecloses it AFTER the authorization branch.
    _edit_buffer(config, "**Cites:** anchor-b", "**Cites:** no-such-decision")

    shortfall = manager.perform_sync(
        auto_accept=False,
        repair_targets=["applies-second", "applies-first"],
    )

    out = capsys.readouterr().out
    assert shortfall == ["applies-second"], "one landed, one refused"
    assert _edges_of(manager, "applies-first") == [], "the first one landed"
    order = [line for line in out.splitlines() if line.startswith("[Divergence]")]
    assert len(order) == 2, order
    assert "applies-first" in order[0] and "applies-second" in order[1], (
        f"document order, not the order named: {order}"
    )


# --------------------------------------------------------------------------- #
# R10 — matching is casefold-exact, one tier
# --------------------------------------------------------------------------- #

def test_a_handle_differing_only_in_case_matches(
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """Case-sensitive membership would make this the one slug surface that refuses
    a spelling the graph accepts — and it would fail as a REFUSAL, where nothing
    looks wrong."""
    config, manager, _ = env
    _no_network(manager)
    _seed_committed_buffer(config, manager, slug="anchor")
    _seed_committed_buffer(config, manager, slug="reconcile-me", cites="anchor")
    _edit_buffer(config, "**Cites:** anchor\n", "")

    assert manager.perform_sync(
        auto_accept=False, repair_targets=["ReCoNcIlE-Me"]) == []
    assert _edges_of(manager, "reconcile-me") == []


def test_a_handle_differing_by_one_character_does_not_match(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """The twin: one tier, no prefix and no did-you-mean resolution."""
    config, manager, _ = env
    _no_network(manager)
    _seed_committed_buffer(config, manager, slug="anchor")
    _seed_committed_buffer(config, manager, slug="reconcile-me", cites="anchor")
    _edit_buffer(config, "**Cites:** anchor\n", "")

    shortfall = manager.perform_sync(
        auto_accept=False, repair_targets=["reconcile-mx"])

    assert shortfall == ["reconcile-mx"]
    assert _edges_of(manager, "reconcile-me") == [("cites", "anchor")], (
        "an unmatched handle authorizes nothing"
    )
    assert "no entry with this slug" in _repair_lines(capsys.readouterr().err)[0]


# --------------------------------------------------------------------------- #
# R11 — the handle is untrusted text; `--reconcile` is not a spelling
# --------------------------------------------------------------------------- #

def test_an_unmatched_handle_is_rendered_as_bounded_quoted_text(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """A handle carrying a quote and an ESC must not break the surface's lines.

    The safe and unsafe spellings differ by three characters (`{h!r}` vs `{h}`),
    which is PATTERNS' own tell that it needs a regression row: a future "tidy"
    reverts it silently. An unmatched handle is by construction the string that
    matched nothing, so nothing upstream vouched for it.
    """
    config, manager, _ = env
    _no_network(manager)
    _seed_committed_buffer(config, manager, slug="present-one")
    hostile = "ev'il\x1b[31m\nsecond-line"

    shortfall = manager.perform_sync(auto_accept=False, repair_targets=[hostile])

    err = capsys.readouterr().err
    assert shortfall == [hostile]
    assert len(_repair_lines(err)) == 1, "the newline must not split the report"
    assert repr(hostile) in err, "rendered through repr, never bare"
    assert "\x1b[31m" not in err, "the raw ESC never reaches the terminal"


def test_the_forbidden_reconcile_spelling_is_not_an_abbreviation(
    capsys: pytest.CaptureFixture
) -> None:
    """`--reconcile <slug>` would shadow the shipped `mitos reconcile` verb.

    argparse defaults `allow_abbrev` ON, which would make `--reconcile foo` an
    accepted abbreviation of `--reconcile-entry` — re-minting the forbidden
    spelling by accident. `_build_parser` turns it off on every registered
    subparser, and this pins that rather than trusting the loop stays.
    """
    from mitos.cli import _build_parser

    with pytest.raises(SystemExit) as excinfo:
        _build_parser().parse_args(["sync", "--reconcile", "foo"])

    assert excinfo.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# R12 — the ninth state: `source`-only divergence, and its clean twin
# --------------------------------------------------------------------------- #

def test_a_source_only_divergence_is_not_read_as_not_diverged(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """`is_reconcilable` is `source`-blind by design, so one branch is two states.

    MI-4 fences `source` out of the commentary UPDATE, so a reconcile provably
    cannot change it — the entry is diverged and permanently unreconcilable. Exit 0
    there stamps a target as satisfied on a corpus that cannot converge.
    """
    config, manager, _ = env
    _no_network(manager)
    _seed_committed_buffer(config, manager, slug="source-drift")
    _edit_buffer(config, "**Mechanisms:** sqlite",
                 "**Source:** capture_llm\n**Mechanisms:** sqlite")

    shortfall = manager.perform_sync(
        auto_accept=False, repair_targets=["source-drift"])

    lines = _repair_lines(capsys.readouterr().err)
    assert shortfall == ["source-drift"]
    assert len(lines) == 1, lines
    assert "**Source:**" in lines[0], lines[0]
    assert "already agree" not in lines[0], "never the clean sentence"


def test_a_genuinely_clean_named_target_exits_zero_and_still_says_so(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """The twin, and the one satisfied state that owes a line.

    Nothing else in the run says anything about an entry the loop skipped as
    already-agreeing, so without this line a caller reads silence as "not applied".
    """
    config, manager, _ = env
    _no_network(manager)
    _seed_committed_buffer(config, manager, slug="already-clean")

    shortfall = manager.perform_sync(
        auto_accept=False, repair_targets=["already-clean"])

    lines = _repair_lines(capsys.readouterr().err)
    assert shortfall == [], "clean is a satisfied state"
    assert len(lines) == 1 and "already agree" in lines[0], lines


# --------------------------------------------------------------------------- #
# R13 — the two rewritten refusals (constraint 3 + 4)
# --------------------------------------------------------------------------- #

def test_the_auto_accept_refusal_names_the_flag_with_this_entrys_slug(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """An UNNAMED diverged entry under `--yes`: the refusal hands over the command.

    Nothing in the suite pinned this sentence before — the two rows that look like
    they would assert a slug and a lower-cased `"skip"` — so this row is the pin
    this phase adds, not a duplicate. The command is composed with a selector and
    with this entry's own slug already in it: copy-the-line, not parse-the-report,
    which is what keeps the narrow door competitive with `rebuild --yes`.
    """
    config, manager, _ = env
    _no_network(manager)
    _seed_committed_buffer(config, manager, slug="anchor")
    _seed_committed_buffer(config, manager, slug="unauthorized", cites="anchor")
    _edit_buffer(config, "**Cites:** anchor\n", "")

    manager.perform_sync(auto_accept=True)

    out = capsys.readouterr().out
    assert f"mitos sync -p {config.project!r} --reconcile-entry 'unauthorized'" in out
    assert "always skipped" not in out, "the falsified claim is gone"
    for shout in ("CRITICAL", "NEVER", "WARNING", "⚠", "**"):
        assert shout not in out.split("[Divergence]")[1], (
            f"prompt-style discipline: {shout!r} in the refusal"
        )
    assert _edges_of(manager, "unauthorized") == [("cites", "anchor")]


def test_the_no_tty_refusal_names_the_flag_and_keeps_its_own_stem(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """The other rewritten refusal — and the stem `test_sync.py` reads stays put.

    3a deliberately gave its guard a DISTINCT stem so the non-TTY reconcile row
    stays unambiguous about which of the two gates fired; changing this one here
    would undo that. The two stems must also stay disjoint in the other direction,
    or `_refusal_lines` starts collecting this sentence and 3a's exact line counts
    red.
    """
    config, manager, _ = env
    assert not sys.stdin.isatty()
    _no_network(manager)
    _seed_committed_buffer(config, manager, slug="anchor")
    _seed_committed_buffer(config, manager, slug="unauthorized", cites="anchor")
    _edit_buffer(config, "**Cites:** anchor\n", "")

    manager.perform_sync(auto_accept=False)

    out = capsys.readouterr().out
    assert _RECONCILE_STEM in out, "the stem test_sync.py asserts on"
    assert f"mitos sync -p {config.project!r} --reconcile-entry 'unauthorized'" in out
    assert _refusal_lines(out) == [], (
        "the two gates' stems must stay disjoint — a shared phrase reds 3a's rows"
    )


# --------------------------------------------------------------------------- #
# R14 — the key-floor bound, pinned as current truth
# --------------------------------------------------------------------------- #

def test_a_keyless_workspace_leaves_the_flag_inert_and_exits_zero(
    env: Tuple[MitosConfig, MitosSyncManager, str], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """The door ships keyed-workspace-only, and the deferral does not license silence.

    `mitos sync`'s `GEMINI_API_KEY` refusal returns ABOVE the per-entry loop, so a
    named target is neither reconciled nor looked for. Fail-loud is scoped to runs
    that clear the floor; flipping that exit code is itself the contract break,
    under every CI job that reads it. Pinned as CURRENT truth so a later pass that
    owns the deprecation must consciously invert this row rather than notice a
    comment.
    """
    config, manager, _ = env
    _seed_committed_buffer(config, manager, slug="reconcile-me")
    _edit_buffer(config, "The original rejected reasoning.", "The CORRECTED reasoning.")
    capsys.readouterr()
    monkeypatch.setitem(config.env, "GEMINI_API_KEY", "")

    shortfall = manager.perform_sync(
        auto_accept=False, repair_targets=["reconcile-me"])

    captured = capsys.readouterr()
    assert shortfall == [], "the flag is inert below the key floor"
    assert "Sync requires API keys." in captured.out
    assert _repair_lines(captured.err) == [], "nothing to be loud about"
    node = manager.store.get_node_by_slug("reconcile-me")
    assert node["rejected_paths"] == "The original rejected reasoning.", (
        "the run neither reconciled nor looked for the target"
    )


# --------------------------------------------------------------------------- #
# Fresh-eyes additions (3b review) — two claims the phase's own rows could not
# see, because every fixture above names an entry whose markdown slug and graph
# slug agree, and none of them stops the loop.
# --------------------------------------------------------------------------- #

def test_the_refusal_recipe_names_the_markdown_slug_after_a_rename(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """A hand-edited RENAME is the one case where the two slugs differ.

    `entry_divergence` lists `slug` first among its commentary fields, so renaming
    an entry's `### ` header IS a reconcilable divergence — and the door matches
    handles against the MARKDOWN slug, because that is what the loop iterates. A
    recipe composed from the graph's slug therefore hands the caller a command that
    reaches the never-seen class on a target sitting in front of them, and it fails
    where nothing looks wrong: the line is present, quoted, selectored, and wrong by
    one field. Both rewritten refusals compose the same recipe, so both are pinned.
    """
    config, manager, _ = env
    _no_network(manager)
    _seed_committed_buffer(config, manager, slug="anchor")
    _seed_committed_buffer(config, manager, slug="old-name", cites="anchor")
    _edit_buffer(config, "### old-name", "### new-name")
    _edit_buffer(config, "**Cites:** anchor\n", "")  # a removal, for the --yes branch

    assert not sys.stdin.isatty()
    manager.perform_sync(auto_accept=False)
    no_tty = capsys.readouterr().out
    manager.perform_sync(auto_accept=True)
    under_yes = capsys.readouterr().out

    recipe = f"--reconcile-entry {'new-name'!r}"
    assert recipe in no_tty, "the no-TTY refusal must name the slug the door matches"
    assert recipe in under_yes, "and so must the --yes-with-removal one"
    assert "--reconcile-entry 'old-name'" not in no_tty + under_yes

    # And the printed command is the one that actually works.
    assert manager.perform_sync(auto_accept=False, repair_targets=["new-name"]) == []
    assert _edges_of(manager, "new-name") == []


@patch("sys.stdin.isatty", return_value=True)
def test_a_target_the_loop_never_reached_is_not_called_absent(
    _isatty: MagicMock, env: Tuple[MitosConfig, MitosSyncManager, str],
    capsys: pytest.CaptureFixture,
) -> None:
    """`[q]uit` ends the loop, so the entries below it went unread — not missing.

    The never-seen line names `mitos rebuild` as the archived member's heal, and
    that is a positive claim about a corpus the run stopped reading. Answering it
    for a target still sitting in the buffer is the wrong recovery for the one
    caller who was there to type `q`. The shortfall and the non-zero exit are
    unchanged — only what the report is willing to assert.
    """
    config, manager, _ = env
    _no_network(manager)
    # `_append_decision` appends while the loop reads bottom-up, so the entry
    # appended LAST is the one reached FIRST: quitting at it leaves `later-one`
    # unread.
    _append_decision(config, "later-one", "The axiom for later-one.")
    _append_decision(config, "quit-here", "The axiom for quit-here.")

    with patch("builtins.input", side_effect=["q"]):
        shortfall = manager.perform_sync(
            auto_accept=False, repair_targets=["later-one"])

    line = _repair_lines(capsys.readouterr().err)[0]
    assert shortfall == ["later-one"], "still a shortfall, still a non-zero exit"
    assert "stopped at its accept prompt" in line
    assert "absent from" not in line and "mitos rebuild" not in line, (
        "a loop that stopped early holds no evidence for an absence claim"
    )


def test_a_run_that_never_got_the_lock_does_not_call_its_target_absent(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """The lock refusal returns above the loop, so the buffer was never read.

    The sibling of the `[q]uit` row: same class, other end of the run. Exit stays
    non-zero (the repair did not land), but naming `mitos rebuild` as the heal for
    an entry sitting untouched in the buffer, while another process is mid-sync, is
    the one recovery that could make things worse.
    """
    config, manager, _ = env
    _seed_committed_buffer(config, manager, slug="reconcile-me")
    capsys.readouterr()
    locked = MagicMock()
    locked.__enter__.side_effect = Timeout("busy")

    with patch.object(manager, "lock", locked):
        shortfall = manager.perform_sync(
            auto_accept=False, repair_targets=["reconcile-me"])

    captured = capsys.readouterr()
    line = _repair_lines(captured.err)[0]
    assert shortfall == ["reconcile-me"], "still a shortfall, still a non-zero exit"
    assert "holds the corpus lock" in line
    assert "absent from" not in line and "mitos rebuild" not in line
