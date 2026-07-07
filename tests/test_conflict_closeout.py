"""Phase 6b — the §1.2 Definition-of-Done gate suite (keyless, deterministic, e2e).

The closeout proofs that name each remaining §1.2 DoD gate against the shipped
Conflict-sensor pipeline. 5a/5b's 21 e2e tests in ``test_conflict_sync.py`` already
carry most legs (surface + rationale + accept-anyway, tenable-silent, clean-empty,
verbatim row, all-pairs, ``sync_run_id`` thread, breaker); this file fills only the
*named gaps* the flip needs — one ``Tn`` / ``DoD-n``-greppable test per gate (the
un-skip audit greps this vision):

- **T1 / DoD-1** — the two missing dispositions: a surfaced conflict the author
  **skips** (not committed this run) and the **resolve-and-re-sync** round trip (re-author
  the buffer entry WITH the relationship → the declared candidate is dropped from the
  check and the entry commits with the edge). Plus the §1.2 strengthening: the candidate's
  ``rejected_paths`` (Letter M5) rides the surfaced finding.
- **T2 / DoD-2** — a candidate Qdrant *returns* but that sits **below**
  ``CONFLICT_SIMILARITY_FLOOR`` is screened out before judgment (distinct from
  ``test_clean_empty``'s empty over-fetch).
- **T4 / DoD-4** — the **static lint**: the conflict library leaf modules
  (``conflict.py`` / ``conflict_judgment.py``) issue no write to ``nodes`` / ``edges`` /
  ``decisions.md`` (MI-11) — the fence a careless future edit can't climb.
- **T5 / DoD-5** — a judged entry with a **long axiom** round-trips to ``conflict_checks``
  **uncapped and byte-identical**, and the telemetry ladder is **replay-idempotent**
  (re-boot the store → the migration re-runs as a no-op, rows intact — MI-3).
- **T6 / DoD-6** — a judged-then-**committed** entry's persisted ``proposed_hash_if_any``
  byte-equals the committed node's content-hash id (the CONF-C1 join resolves).
- **T8** — a populated ``telemetry.sqlite`` and its ``conflict_checks`` rows survive a real
  ``rebuild_and_gate`` + ``perform_swap`` (the sibling store sits OUTSIDE the graph swap set).

Discipline (scout brief / PATTERNS live-test rule): deterministic + **keyless** + no SDK.
The whole harness is imported from ``_conflict_helpers`` — its ``offline`` fixture is
autouse here (GEMINI mock key so ``perform_sync`` does not early-return, no ANTHROPIC key,
a dead Qdrant URL), the judge is a plain injected callable, the graph + ``telemetry.sqlite``
are real temp SQLite. Every dynamic value (the version, the floor, the committed node id,
row counts) is read programmatically — never a hardcoded literal a later re-calibration
would rot.
"""

import ast
import os
from typing import List, Optional, Set, Tuple
from unittest.mock import patch

import pytest

import mitos.conflict
from mitos.conflict import CONFLICT_SIMILARITY_FLOOR
from mitos.config import MitosConfig
from mitos.sync import MitosSyncManager
from mitos.telemetry import TelemetryStore

from _conflict_helpers import (
    _RecordingJudge,
    _append_decision,
    _execution,
    _match,
    _read_batch_rows,
    _read_conflict_rows,
    _seed_active,
    _wire_fakes,
    env,
    offline,
)

# The env fixture's decisions.md preamble (mirrors ``_conflict_helpers.env``). Used by the
# resolve round-trip to re-author the buffer with a declaration line ``_append_decision``
# cannot emit (its signature has no relationship arg — scout Discrepancy #3).
_BUFFER_HEADER = (
    "# Decisions\n"
    "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->\n"
)


def _write_declared_decision(
    config: MitosConfig,
    slug: str,
    axiom: str,
    *,
    narrows: str,
    rejected: str = "Rejected always-open access.",
    scope: str = "api",
    mechanisms: str = "python",
    date: str = "2026-06-01",
) -> None:
    """Rewrites the decisions.md buffer to a single entry that DECLARES a ``Narrows:`` target.

    ``_append_decision`` emits a bare decision (no relationship field), so the
    resolve-and-re-sync leg authors the second-sync entry directly with a ``**Narrows:**``
    line (verified to parse to ``entry.narrows == [target]``). ``narrows`` is a non-kill
    edge (RF: unlike ``supersedes``/``corrects`` it does not retire the candidate), so the
    candidate stays active for the post-commit edge assertion. The whole buffer is rewritten
    (not appended) so the round-trip fully controls it regardless of prior-sync rotation.

    Args:
        config: The active workspace config (supplies ``decisions_file``).
        slug: The decision slug.
        axiom: The decision axiom (verbatim under strict-deterministic sync).
        narrows: The slug the entry declares it ``Narrows:`` (the declared-drop target).
        rejected: The ``**Rejected:**`` anti-knowledge (required on a decision).
        scope: The single ``**Scope:**`` tag.
        mechanisms: The single ``**Mechanisms:**`` token.
        date: The entry date (archive-quarter neutral).
    """
    block = (
        f"## {date} — {slug} — {slug.replace('-', ' ').title()}\n"
        f"**Decided:** {axiom}\n"
        f"**Rejected:** {rejected}\n"
        f"**Mechanisms:** {mechanisms}\n"
        f"**Scope:** {scope}\n"
        f"**Narrows:** {narrows}\n"
    )
    with open(config.decisions_file, "w", encoding="utf-8") as f:
        f.write(_BUFFER_HEADER + "\n" + block + "\n")


# --------------------------------------------------------------------------- #
# T1 / DoD-1 — the two missing dispositions (skip; resolve-and-re-sync) + the
#              candidate rejected_paths riding the surfaced finding.
# --------------------------------------------------------------------------- #


def test_t1_dod1_skip_disposition_does_not_commit(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """A surfaced conflict the author SKIPS ('s') surfaces, then the entry does NOT commit.

    The disposition the 5a/5b suite never drove (it only ever fed ``input('a')``): a
    not-tenable finding prints, the author skips, and the proposal is absent from the graph
    for this run — the sensor advised, the human declined, nothing was forced (P6).
    """
    config, manager, _ = env
    _seed_active(manager, "endpoints-auth", "All API endpoints require authentication.",
                 scope=["api", "security"])
    judge = _RecordingJudge(
        _execution([("endpoints-auth", False, 0.9,
                     "The proposal exempts /health from auth; the active decision admits "
                     "no unauthenticated endpoint.")])
    )
    _wire_fakes(manager, judge=judge, matches=[_match("endpoints-auth", 0.9)])

    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")
    with patch("builtins.input", side_effect=["s"]):
        manager.perform_sync(auto_accept=False)

    out = capsys.readouterr().out
    assert "[Conflict]" in out                       # the tension was surfaced…
    assert "endpoints-auth" in out
    assert judge.calls == 1                           # …the undeclared candidate reached the judge…
    assert manager.store.get_node_by_slug("health-public") is None  # …and skip did not commit it.


def test_t1_dod1_resolve_and_resync_drops_declared_and_lands_edge(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """The resolve round trip: surface undeclared → declare → the candidate is dropped, edge lands.

    The CONF-D7 "declared → dropped, not the whole entry" contract proven e2e. Sync #1
    surfaces an undeclared contradiction and the author skips; sync #2 re-authors the SAME
    entry WITH ``Narrows: endpoints-auth`` — so ``screen_candidates``' declared-target drop
    removes the candidate BEFORE judgment (the judge is not called again), and the entry
    commits carrying the declared ``narrows`` edge. RF-1: the drop keys on the parsed
    declaration read at check time, not on any collision verb applied later.
    """
    config, manager, _ = env
    _seed_active(manager, "endpoints-auth", "All API endpoints require authentication.",
                 scope=["api", "security"])
    judge = _RecordingJudge(
        _execution([("endpoints-auth", False, 0.9, "Health is exempted; auth admits none.")])
    )
    _wire_fakes(manager, judge=judge, matches=[_match("endpoints-auth", 0.9)])

    # Sync #1 — undeclared contradiction surfaces; the author SKIPS to resolve it.
    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")
    with patch("builtins.input", side_effect=["s"]):
        manager.perform_sync(auto_accept=False)
    out1 = capsys.readouterr().out
    assert "[Conflict]" in out1
    assert judge.calls == 1
    assert manager.store.get_node_by_slug("health-public") is None

    # Sync #2 — the author resolves by declaring the relationship, then re-syncs.
    _write_declared_decision(
        config, "health-public", "The /health endpoint is publicly accessible.",
        narrows="endpoints-auth",
    )
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)
    out2 = capsys.readouterr().out

    assert "[Conflict]" not in out2               # declared → dropped → nothing surfaced
    assert judge.calls == 1                        # the judge was NOT called again (declared-drop → clean-empty)

    committed = manager.store.get_node_by_slug("health-public")
    assert committed is not None                   # the resolved entry commits
    target = manager.store.get_node_by_slug("endpoints-auth")
    narrows_edges = [
        e for e in manager.store.get_edges()
        if e["edge_type"] == "narrows" and e["source_id"] == committed["id"]
    ]
    assert len(narrows_edges) == 1                 # the declared edge landed…
    assert narrows_edges[0]["target_id"] == target["id"]  # …pointing at the candidate.


def test_t1_dod1_candidate_rejected_paths_ride_surfaced_finding(
    env: Tuple[MitosConfig, MitosSyncManager, str], capsys: pytest.CaptureFixture
) -> None:
    """§1.2 completeness: the candidate's ``rejected_paths`` (Letter M5) rides the finding.

    The 5a surface test asserts the candidate axiom + rationale; §1.2 also promises the
    candidate's ``rejected_paths`` (Letter mode) surfaces so the author sees why the standing
    decision fenced off the very path the proposal now takes. Seed a distinctive
    ``**Rejected:**`` and assert its text appears in the surfaced block (accept-anyway still
    commits — the advisory posture is untouched).
    """
    config, manager, _ = env
    marker = "Rejected leaving any endpoint unauthenticated (the load-bearing carve-out fence)."
    _seed_active(manager, "endpoints-auth", "All API endpoints require authentication.",
                 rejected=marker)
    judge = _RecordingJudge(
        _execution([("endpoints-auth", False, 0.9,
                     "The proposal exempts /health; the active decision admits no "
                     "unauthenticated endpoint.")])
    )
    _wire_fakes(manager, judge=judge, matches=[_match("endpoints-auth", 0.9)])

    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    out = capsys.readouterr().out
    assert "[Conflict]" in out
    assert marker in out                           # the candidate's M5 anti-knowledge rode the finding
    assert manager.store.get_node_by_slug("health-public") is not None  # accept-anyway committed


# --------------------------------------------------------------------------- #
# T2 / DoD-2 — a returned-but-below-floor candidate is screened out (the W7 e2e).
# --------------------------------------------------------------------------- #


def test_t2_dod2_below_floor_candidate_short_circuits_before_judgment(
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """A candidate Qdrant returns BELOW the calibrated floor is screened → no judge, no row.

    Distinct from ``test_clean_empty_is_silent_and_judge_not_called`` (which uses an EMPTY
    over-fetch): here Qdrant *returns* a match, but its score sits below
    ``CONFLICT_SIMILARITY_FLOOR`` — so ``screen_candidates`` drops it, the judge is never
    called, no ``conflict_checks`` row is written, and the entry commits. This is the gate
    that makes W7's calibrated constant load-bearing at runtime. The below-floor score is
    read off the constant (``FLOOR - epsilon``), never a literal 4b may re-calibrate.
    """
    config, manager, _ = env
    _seed_active(manager, "endpoints-auth", "All API endpoints require authentication.")
    judge = _RecordingJudge(_execution([]))  # a parse-empty if it were ever (wrongly) called
    below_floor = CONFLICT_SIMILARITY_FLOOR - 0.05
    _wire_fakes(manager, judge=judge, matches=[_match("endpoints-auth", below_floor)])

    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    assert judge.calls == 0                         # the floor short-circuited before judgment
    assert _read_conflict_rows(config) == []        # nothing judged → nothing persisted
    assert manager.store.get_node_by_slug("health-public") is not None  # the entry commits


# --------------------------------------------------------------------------- #
# T4 / DoD-4 — static lint: the conflict library leaf modules never write graph/markdown.
# --------------------------------------------------------------------------- #

# The write-vector denylist (MI-11 / CONF-D1). The conflict LIBRARY leaf modules read the
# graph freely (``store.get_node_by_slug`` / ``get_node_state``, ``compute_node_id``,
# ``embedding_text``, ``letter_payload``) but must NEVER mutate ``nodes`` / ``edges`` /
# ``decisions.md`` — so the fence targets WRITE call names, not any ``store.`` touch (a
# blanket store-denylist would false-positive on the legitimate reads). The sensor's own
# ``conflict_checks`` telemetry write is out of scope by construction: it lives in
# ``sync.py`` (the legit committer) + ``telemetry.py``, never in these two leaves.
_WRITE_CALL_DENYLIST = frozenset({
    "commit_parsed_entry",     # the graph write path
    "record_decision_entry",   # the agentic graph write path
    "write_signal",            # a signals-table mutator
    "note_source_reencounter", # a signals-table mutator
    "execute",                 # raw SQL — any SQL here is a red flag (reads route through store)
    "executemany",
    "executescript",
    "commit",                  # a connection-level commit
})

# The two — and only two — "conflict library" module SOURCE files (the facade/executor
# split). The lint scope is deliberately these leaves only (scout Pattern Siblings: no
# hidden third). Located by PATH off the (dep-free) ``mitos.conflict`` leaf rather than by
# importing ``mitos.conflict_judgment`` — importing the executor would pull the ``anthropic``
# SDK into this keyless process, breaking the suite's no-SDK discipline. The lint reads
# source text; it never needs the module object.
_MITOS_PKG_DIR = os.path.dirname(mitos.conflict.__file__)
_CONFLICT_LEAF_SOURCES = (
    os.path.join(_MITOS_PKG_DIR, "conflict.py"),
    os.path.join(_MITOS_PKG_DIR, "conflict_judgment.py"),
)


def _open_write_mode(call: ast.Call) -> Optional[str]:
    """Returns the mode string of an ``open(...)`` call if it names a WRITE mode, else None.

    Reads the 2nd positional arg or the ``mode=`` keyword (only when a string literal). A
    write mode contains any of ``w`` / ``a`` / ``x`` / ``+``. A read (or dynamically-moded)
    ``open`` returns None — neither leaf opens files today, so this only fences a future edit.
    """
    mode_node: Optional[ast.expr] = None
    if len(call.args) >= 2:
        mode_node = call.args[1]
    for kw in call.keywords:
        if kw.arg == "mode":
            mode_node = kw.value
    if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
        if any(c in mode_node.value for c in ("w", "a", "x", "+")):
            return mode_node.value
    return None


def _write_call_violations(source: str, filename: str) -> List[Tuple[str, int, str]]:
    """Walks a module's AST for any graph/markdown write call (the DoD-4 denylist).

    Args:
        source: The module source text.
        filename: A label for the offender report (the module basename).

    Returns:
        A list of ``(filename, lineno, offender)`` — empty when the module is write-clean.
    """
    tree = ast.parse(source, filename=filename)
    violations: List[Tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name: Optional[str] = None
        if isinstance(func, ast.Attribute):
            name = func.attr
        elif isinstance(func, ast.Name):
            name = func.id
        if name in _WRITE_CALL_DENYLIST:
            violations.append((filename, node.lineno, name))
        if isinstance(func, ast.Name) and func.id == "open":
            mode = _open_write_mode(node)
            if mode is not None:
                violations.append((filename, node.lineno, f"open(mode={mode!r})"))
    return violations


def test_t4_dod4_conflict_leaf_modules_issue_no_graph_or_markdown_write() -> None:
    """DoD-4: the conflict library leaf modules statically contain zero graph/markdown writes.

    The static twin of 6a's behavioural commit-integrity proof — 6a showed the check does not
    mutate the graph at *runtime*; this proves a future edit *cannot* introduce a write path
    without tripping the fence. AST-parse both leaves and assert no denylisted write call (and
    no ``open`` in a write mode) survives. The telemetry write is out of scope (it lives in
    ``sync.py`` / ``telemetry.py``, not these leaves).
    """
    violations: List[Tuple[str, int, str]] = []
    for path in _CONFLICT_LEAF_SOURCES:
        assert os.path.exists(path), f"conflict leaf source not found to lint: {path}"
        with open(path, encoding="utf-8") as f:
            source = f.read()
        violations.extend(_write_call_violations(source, os.path.basename(path)))

    assert violations == [], (
        "Conflict library leaf modules must issue no graph/markdown write (MI-11 / DoD-4). "
        "Offending write calls: "
        + "; ".join(f"{fn}:{ln} → {name}" for fn, ln, name in violations)
    )


# --------------------------------------------------------------------------- #
# T5 / DoD-5 — uncapped long-axiom round-trip + replay-idempotent ladder.
# --------------------------------------------------------------------------- #


def _sync_one_judged_conflict(
    config: MitosConfig,
    manager: MitosSyncManager,
    *,
    proposal_axiom: str,
    proposal_slug: str = "health-public",
) -> None:
    """Drives one judged, surfaced, accepted conflict sync (the shared DoD-5/6 setup).

    Seeds ``endpoints-auth`` active, wires a not-tenable @0.9 judge over a single match, and
    runs ``perform_sync`` with ``input('a')`` so exactly one judged ``conflict_checks`` row
    (surfaced) is persisted and the proposal commits.
    """
    _seed_active(manager, "endpoints-auth", "All API endpoints require authentication.")
    judge = _RecordingJudge(
        _execution([("endpoints-auth", False, 0.9, "Health is exempted; auth admits none.")])
    )
    _wire_fakes(manager, judge=judge, matches=[_match("endpoints-auth", 0.9)])
    _append_decision(config, proposal_slug, proposal_axiom)
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)


def test_t5_dod5_long_axiom_persists_uncapped_and_verbatim(
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """A judged entry with a long axiom round-trips to ``conflict_checks`` uncapped + byte-identical.

    CONF-D8's verbatim-and-uncapped rule proven end-to-end through the real sync surface + a
    real ``telemetry.sqlite`` read-back: the persisted ``judged_axiom`` byte-equals the
    authored axiom (which is byte-equal to the committed node's ``core_axiom`` — the same
    parsed string), and its length far exceeds any incidental cap. (The multi-line-with-
    newlines variant is pinned at the telemetry-store unit level in ``test_telemetry.py``;
    the parser-safe e2e path here uses a long single-line axiom — see IMPLEMENTATION_NOTES.)
    """
    config, manager, _ = env
    long_axiom = (
        "The /health endpoint is publicly accessible without authentication, and this "
        "exemption is deliberate and load-bearing for uptime probes; "
        + ("it must never be silently narrowed by a blanket auth rule. " * 40)
    ).strip()
    assert len(long_axiom) > 1000  # a genuinely long axiom (the uncapped premise)

    _sync_one_judged_conflict(config, manager, proposal_axiom=long_axiom)

    rows = _read_conflict_rows(config)
    assert len(rows) == 1
    row = rows[0]
    assert row["judged_axiom"] == long_axiom               # byte-identical, uncapped
    assert len(row["judged_axiom"]) == len(long_axiom)     # no truncation cap
    committed = manager.store.get_node_by_slug("health-public")
    assert committed is not None
    assert row["judged_axiom"] == committed["core_axiom"]  # the persisted context IS the committed core


def test_t5_dod5_telemetry_ladder_replay_is_idempotent(
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """Re-booting the telemetry store re-runs its migration ladder as a no-op; rows intact (MI-3).

    The e2e twin of 1b's unit replay proof: after a judged sync has created
    ``telemetry.sqlite`` and its rows, constructing a fresh ``TelemetryStore`` on the same
    path re-runs the ``TELEMETRY_MIGRATION_STEPS`` ladder — which is ``IF NOT EXISTS`` /
    ``PRAGMA user_version``-guarded, so it neither errors nor disturbs the existing rows.
    """
    config, manager, _ = env
    _sync_one_judged_conflict(config, manager, proposal_axiom="The /health endpoint is public.")

    rows_before = _read_conflict_rows(config)
    batches_before = _read_batch_rows(config)
    assert len(rows_before) == 1
    assert len(batches_before) == 1

    # Re-boot the store on the SAME path → the ladder replays. Must not raise.
    TelemetryStore(config.telemetry_path)

    assert _read_conflict_rows(config) == rows_before      # rows survive the replay intact
    assert _read_batch_rows(config) == batches_before


# --------------------------------------------------------------------------- #
# T6 / DoD-6 — the join key: persisted proposed_hash_if_any == committed node id.
# --------------------------------------------------------------------------- #


def test_t6_dod6_proposed_hash_joins_to_committed_node(
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """For a judged-then-committed entry, the row's ``proposed_hash_if_any`` == the node's id.

    The CONF-C1 join proven e2e through the real sync surface: the persisted
    ``proposed_hash_if_any`` byte-equals the committed node's stored content-hash ``id``
    (itself ``compute_node_id`` over the canonical core — the honest join read off the graph,
    never a re-hash). Equality is by construction (strict-deterministic sync leaves no
    parse-vs-commit gap), so this gate is the tripwire that keeps a future sync change from
    silently orphaning the corpus (DoD-6's raison d'être, §1.2).
    """
    config, manager, _ = env
    _sync_one_judged_conflict(
        config, manager, proposal_axiom="The /health endpoint is publicly accessible."
    )

    rows = _read_conflict_rows(config)
    assert len(rows) == 1
    committed = manager.store.get_node_by_slug("health-public")
    assert committed is not None
    assert rows[0]["proposed_hash_if_any"] == committed["id"]  # the join resolves


# --------------------------------------------------------------------------- #
# T8 — telemetry survives a real rebuild + swap (it sits outside the graph swap set).
# --------------------------------------------------------------------------- #


def test_t8_telemetry_survives_rebuild_swap_e2e(
    env: Tuple[MitosConfig, MitosSyncManager, str]
) -> None:
    """Telemetry populated by a real judged ``perform_sync`` survives ``rebuild`` + ``perform_swap``.

    The e2e twin of 1b's unit T8 (which populated telemetry by a direct
    ``record_judged_batch``): here the ``conflict_checks`` rows are written by the SHIPPED
    sync writer path, then a real ``rebuild_and_gate`` + ``perform_swap`` swaps the graph.
    Because the swap set is ``graph.sqlite`` (+ ``.bak_<ts>`` / ``-wal`` / ``-shm``) ONLY,
    the sibling ``telemetry.sqlite`` and its rows are untouched. Both decisions are authored
    through the buffer (so rotation archives them) → the rebuild reproduces the graph from
    the archive and the completeness gate passes.
    """
    from mitos.cutover import default_aside_db_path, perform_swap, rebuild_and_gate

    config, manager, _ = env

    # Seed the candidate through the buffer (auto_accept skips the check) so it rotates to
    # the archive and the rebuild can reproduce it — a directly-committed node would be
    # absent from the corpus and trip the completeness gate.
    _append_decision(config, "endpoints-auth", "All API endpoints require authentication.")
    manager.perform_sync(auto_accept=True)

    # A judged sync writes the telemetry rows AND commits (+ archives) the proposal.
    judge = _RecordingJudge(
        _execution([("endpoints-auth", False, 0.9, "Health is exempted; auth admits none.")])
    )
    _wire_fakes(manager, judge=judge, matches=[_match("endpoints-auth", 0.9)])
    _append_decision(config, "health-public", "The /health endpoint is publicly accessible.")
    with patch("builtins.input", side_effect=["a"]):
        manager.perform_sync(auto_accept=False)

    rows_before = _read_conflict_rows(config)
    batches_before = _read_batch_rows(config)
    assert len(rows_before) == 1
    assert len(batches_before) == 1
    assert os.path.exists(config.telemetry_path)

    # Drive the REAL swap (deterministic timestamp — never wall-clocked, PLANNING_NOTES).
    aside = default_aside_db_path(config)
    report = rebuild_and_gate(config, aside_db_path=aside)
    assert report.gate_passed, f"rebuild completeness gate failed: {report.missing_cores}"
    perform_swap(config, aside, timestamp="20260703-120000")

    # The sibling telemetry store and its rows survived the graph swap untouched.
    assert os.path.exists(config.telemetry_path)
    assert _read_conflict_rows(config) == rows_before
    assert _read_batch_rows(config) == batches_before


# --------------------------------------------------------------------------- #
# DoD-6 (2c extension) — the no-write lint over the CHECK family, discovered by
# runtime-import closure (KD7): no hand-maintained allowlist, one sanctioned
# boundary. Tomorrow's additions (3b's staged predicate, 2d's probe consumption)
# join mechanically the moment their imports land in check.py.
# --------------------------------------------------------------------------- #

# The one CHK-C1-sanctioned write sink: ``mitos/telemetry.py`` writes its own
# sibling DB (``conn.execute`` against telemetry.sqlite) — a write path the
# denylist would red instantly yet the vision explicitly sanctions. It is a
# BOUNDARY: neither linted nor traversed (its runtime imports would drag
# ``store.py``, the legitimate graph committer, plus ``migrations.py`` into a
# lint that is nonsense over them). One named constant, not an allowlist — it
# never grows as the check family grows (5a's T6 audits exactly that).
_LINT_BOUNDARY_BASENAMES = frozenset({"telemetry.py"})


def _module_source_path(module_name: str) -> Optional[str]:
    """Maps a dotted ``mitos``-family module name to its source file, or ``None``.

    The bare package ``mitos`` maps to ``mitos/__init__.py`` (the ``from mitos
    import __version__`` edge must enter the closure, not silently vanish); a
    non-``mitos`` name or a name that is an attribute rather than a module (no
    matching file) returns ``None``. Path arithmetic off ``_MITOS_PKG_DIR`` — the
    same locate-by-path idiom as ``_CONFLICT_LEAF_SOURCES`` (never an import).
    """
    if module_name == "mitos":
        return os.path.join(_MITOS_PKG_DIR, "__init__.py")
    if not module_name.startswith("mitos."):
        return None
    candidate = os.path.join(
        _MITOS_PKG_DIR, *module_name[len("mitos."):].split(".")
    ) + ".py"
    return candidate if os.path.exists(candidate) else None


def _runtime_mitos_imports(source: str, filename: str) -> List[str]:
    """The module names a source imports at RUNTIME — ``TYPE_CHECKING`` blocks skipped.

    Walks the AST manually so any ``if TYPE_CHECKING:`` body is pruned (annotation
    imports cannot execute a write; traversing them would drag ``protocols`` →
    ``parser``/``store`` into the lint — an instant false red). Both guard shapes
    are recognized (bare ``TYPE_CHECKING`` and ``typing.TYPE_CHECKING``); the
    ``orelse`` of such an ``If`` DOES run at runtime and is walked. For a
    ``from X import Y`` the walker records ``X`` and, additively, ``X.Y`` — so a
    ``from mitos import conflict``-style module import still resolves (a plain
    attribute name simply maps to no file and drops out).
    """
    tree = ast.parse(source, filename=filename)
    imports: List[str] = []

    def _is_type_checking_guard(test: ast.expr) -> bool:
        return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )

    def _walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and _is_type_checking_guard(child.test):
                for runtime_branch in child.orelse:
                    _walk(runtime_branch)
                continue
            if isinstance(child, ast.Import):
                for alias in child.names:
                    imports.append(alias.name)
            elif isinstance(child, ast.ImportFrom):
                if child.level == 0 and child.module:
                    imports.append(child.module)
                    for alias in child.names:
                        imports.append(f"{child.module}.{alias.name}")
            _walk(child)

    _walk(tree)
    return imports


def _check_family_closure() -> List[str]:
    """KD7's discovery: the check family's source files, by runtime-import closure.

    Starts at ``mitos/check.py``, resolves every runtime ``mitos``-family import to
    its source file, and recurses — except the sanctioned-sink boundary, which is
    neither returned (linted) nor traversed. Returns sorted absolute paths.
    """
    entry = os.path.join(_MITOS_PKG_DIR, "check.py")
    discovered: List[str] = []
    seen: Set[str] = set()
    queue = [entry]
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        if os.path.basename(path) in _LINT_BOUNDARY_BASENAMES:
            continue  # the sanctioned sink: neither linted nor traversed
        discovered.append(path)
        with open(path, encoding="utf-8") as f:
            source = f.read()
        for module_name in _runtime_mitos_imports(source, os.path.basename(path)):
            resolved = _module_source_path(module_name)
            if resolved is not None:
                queue.append(resolved)
    return sorted(discovered)


def test_dod6_discovery_covers_the_check_family_and_respects_the_boundary() -> None:
    """KD7 mechanics + 5a's W12 drift-pin: discovery yields the check family EXACTLY.

    2c/2d asserted ``⊇``; 5a tightens it to ``==`` against the final runtime closure, so a
    future import silently dragging a NEW module into the fence — or dropping one out of it
    — trips this test rather than passing unnoticed. The closure is the seven source files
    ``check.py`` runtime-reaches (``conflict.py`` → ``display.py``/``identity.py``;
    ``models.py``; ``errors.py`` — 2d's twin-catch; ``mitos/__init__.py`` via the
    ``__version__`` edge), and it excludes BOTH ``telemetry.py`` (the sanctioned sink) AND
    everything only reachable through it (``store.py``/``migrations.py`` — the graph
    committer must never enter this lint). The TYPE_CHECKING-guarded ``protocols`` import is
    skipped, so neither ``protocols.py`` nor its ``parser.py``/``store.py`` pulls appear."""
    names = {os.path.basename(path) for path in _check_family_closure()}
    assert names == {
        "check.py", "conflict.py", "display.py", "errors.py",
        "identity.py", "models.py", "__init__.py",
    }
    # The boundary + everything only reachable through it stay out (belt-and-suspenders
    # over the exact-set pin: a regression names the offender, not just a set diff).
    assert "telemetry.py" not in names
    assert "store.py" not in names
    assert "migrations.py" not in names
    assert "protocols.py" not in names
    assert "parser.py" not in names
    assert "sync.py" not in names


def test_dod6_no_write_lint_over_the_discovered_check_family() -> None:
    """DoD-6: every discovered check-family source, unioned with the parent's two
    static conflict leaves, passes the shared write-call walker — the fence
    extends mechanically as the family grows (no allowlist to forget)."""
    paths = sorted(set(_check_family_closure()) | set(_CONFLICT_LEAF_SOURCES))
    assert len(paths) >= 3  # the family is discovered, not an empty vacuous pass
    violations: List[Tuple[str, int, str]] = []
    for path in paths:
        assert os.path.exists(path), f"check-family source not found to lint: {path}"
        with open(path, encoding="utf-8") as f:
            violations.extend(_write_call_violations(f.read(), os.path.basename(path)))
    assert violations == [], (
        "Check-family modules must issue no graph/markdown write (MI-11 / DoD-6). "
        "Offending write calls: "
        + "; ".join(f"{fn}:{ln} → {name}" for fn, ln, name in violations)
    )


def test_dod6_walker_trips_on_a_synthetic_write_call() -> None:
    """§9-18's mechanism proof: a synthetic module holding a denylisted write call
    DOES trip the shared walker — the fence is proven live, not just
    currently-green over clean sources."""
    synthetic = (
        "def sneaky(store, entry):\n"
        "    return store.commit_parsed_entry(entry)\n"
    )
    violations = _write_call_violations(synthetic, "synthetic.py")
    assert [name for (_, _, name) in violations] == ["commit_parsed_entry"]
