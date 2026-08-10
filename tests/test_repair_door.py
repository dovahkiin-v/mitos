"""Phase 3a — the accept-prompt guard that makes the targeted repair door reachable.

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

Home for the door's rows generally: **phase 3b extends this module** with the named-target
repair flag's set. Fixtures come from `_conflict_helpers` — `offline` for the key injection
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
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import pytest

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
