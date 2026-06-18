"""Dogfood self-parse closeout guard (Phase 8b — §1.2 DoD #3, T11).

The vision's namesake gate: *"the crane that builds the crane" parses its own
decisions.* V1a's first real parse input is the corpus the vision loop itself
wrote — ``decisions.md`` + its archives + ``questions.md`` — read back through
the hardened parser those phases built, against the as-shipped ``format-spec.md``.
If the corpus the tool recorded can't parse clean against the parser the tool
shipped, V1a isn't done.

This module is the **durable regression guard**. The binding one-time closeout
RUN (the §1.2 DoD #3 gate) was performed by the implementer over the as-shipped
working-tree corpus and its resolutions logged in ``IMPLEMENTATION_NOTES`` (the
(a)/(b)/(c) branch protocol, OD1: every failure resolved, silent skip/coerce
forbidden). The RUN was clean — zero (a)/(b)-class failures; the 50+ recognized
V1b relationship-field instances (``Cites``/``Amends``/``Depends-On``/``Narrows``/
``Supersedes``) tokenized clean (branch (c) — recognized, warn-deferred at
*commit*, never a parse failure). This test keeps that dogfood-clean property
CI-guarded going forward.

**Keyless + checkout-robust (G1/G11).** Pure parse, no store/keys/Qdrant. The
hard CI gate is the always-committed ``format-spec.md`` §3/§4 sample blocks
(bundled package data — present in every checkout). The richer corpus
(``decisions.md`` is historically left uncommitted — Vinga's bookkeeping; the
archives are committed) is parsed **if present**, skip-with-reason if absent, so
the guard survives a checkout without the rich corpus while still pinning it on a
dev box. The OQ pass is attempted even though ``questions.md`` is absent today
(G10) so a future OQ corpus is never silently skipped (§2.1 "names both files").
"""

import glob
import os
from typing import List, Tuple

import pytest

from mitos.cli import _extract_sample_block, load_format_spec
from mitos.errors import EntryFailure
from mitos.parser import parse_entry_stream

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse_collect(text: str, kind: str, source: str) -> Tuple[List, List[EntryFailure]]:
    """Parses a stream in collector mode and returns ``(entries, failures)``.

    Collector mode (``failures=[]``) never raises on the first issue — it surfaces
    the *whole* stream's failure set at once, so the closeout reviews them together
    (the per-entry isolation §5.2.2 guarantees). Recognized V1b relationship fields
    tokenize clean and produce NO ``EntryFailure``; an empty accumulator therefore
    means a clean (a)/(b)-class parse.

    Args:
        text: The markdown stream.
        kind: ``"decision"`` or ``"open_question"`` (caller-declared, V1-D8).
        source: A label threaded onto each failure envelope for diagnostics.

    Returns:
        The parsed entries and the accumulated failures.
    """
    failures: List[EntryFailure] = []
    entries = parse_entry_stream(text, kind, source_path=source, failures=failures)
    return entries, failures


def _describe(failures: List[EntryFailure]) -> str:
    """Renders an accumulator into a readable assertion message (P3 vector errors)."""
    lines = []
    for f in failures:
        d = f.to_dict()
        for item in d.get("items", []):
            lines.append(
                f"{d.get('source_path')}:{d.get('line_start')} [{d.get('slug')}] "
                f"{item.get('code')} ({item.get('source')}): {item.get('message')}"
            )
    return "\n".join(lines)


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# --------------------------------------------------------------------------- #
# Hard CI gate — the always-committed format-spec samples
# --------------------------------------------------------------------------- #


def test_format_spec_samples_self_parse_clean() -> None:
    """The bundled ``format-spec.md`` §3/§4 worked samples parse clean (hard CI gate).

    The format spec's own canonical sample-per-kind is the one corpus artifact
    guaranteed in every checkout (it is package data ``mitos init`` installs). If
    the spec's own example doesn't parse against the parser the spec drives, the
    dogfood thesis is broken at the root — so this is the hard gate that runs on
    every push regardless of whether the rich corpus is checked out.
    """
    spec = load_format_spec()
    decision_sample = _extract_sample_block(spec, "## 3. Sample Entry")
    question_sample = _extract_sample_block(spec, "## 4. Open Question Sample")
    assert decision_sample, "format-spec.md §3 decision sample is missing"
    assert question_sample, "format-spec.md §4 open-question sample is missing"

    d_entries, d_failures = _parse_collect(decision_sample, "decision", "format-spec.md#3")
    assert d_failures == [], _describe(d_failures)
    assert len(d_entries) >= 1

    q_entries, q_failures = _parse_collect(
        question_sample, "open_question", "format-spec.md#4"
    )
    assert q_failures == [], _describe(q_failures)
    assert len(q_entries) >= 1


# --------------------------------------------------------------------------- #
# Present-if-available — the recorded corpus (archives committed; decisions.md
# historically uncommitted, G11)
# --------------------------------------------------------------------------- #


def test_committed_archives_self_parse_clean() -> None:
    """Every committed ``decisions/archive/*.md`` parses clean as decisions."""
    archives = sorted(glob.glob(os.path.join(REPO_ROOT, "decisions", "archive", "*.md")))
    if not archives:
        pytest.skip("no decisions/archive/*.md present in this checkout")
    for arc in archives:
        entries, failures = _parse_collect(_read(arc), "decision", arc)
        assert failures == [], _describe(failures)
        assert len(entries) >= 1


def test_working_tree_decisions_self_parse_clean() -> None:
    """The as-shipped ``decisions.md`` parses clean against the as-shipped spec.

    Parses whatever ``decisions.md`` the checkout holds — the rich 74-entry corpus
    on a dev box, the thin committed seed in CI — and asserts zero (a)/(b)-class
    failures. The recognized V1b relationship fields the rich corpus carries
    (``Cites``/``Amends``/``Depends-On``/``Narrows``/``Supersedes``) are branch (c):
    they tokenize clean and warn-defer only at *commit*, so the empty accumulator
    IS the proof the parse-stage tolerates them (it never flags a recognized field
    as ``malformed_entry``). Skip-with-reason if absent (G11 — Vinga's uncommitted
    corpus).
    """
    path = os.path.join(REPO_ROOT, "decisions.md")
    if not os.path.exists(path):
        pytest.skip("decisions.md absent in this checkout (uncommitted corpus, G11)")
    text = _read(path)
    _entries, failures = _parse_collect(text, "decision", path)
    # NOTE: no >=1 entry assertion — the committed CI seed may hold zero entries
    # below the sentinel; cleanliness is the property under guard, not corpus size.
    assert failures == [], _describe(failures)

    # When the rich corpus is checked out, prove the V1b-tolerance path actually ran
    # (the clean parse above is non-vacuous), not merely that an empty seed parsed.
    if any(m in text for m in ("**Cites:**", "**Amends:**", "**Narrows:**", "**Depends-On:**")):
        assert len(_entries) >= 1


def test_open_questions_self_parse_clean_or_absent() -> None:
    """The OQ self-parse pass is attempted even when ``questions.md`` is absent (G10).

    ``questions.md`` is absent/empty today (zero OQs in the corpus), so this is a
    no-op pass now — but the closeout MUST still *attempt* it (parse as
    ``open_question`` when present and non-empty) so a future OQ corpus is never
    silently skipped (§2.1 "names both files so it never silently skips"). Guard:
    file-absent / empty-stream → skip-with-reason, never an error.
    """
    path = os.path.join(REPO_ROOT, "questions.md")
    if not os.path.exists(path) or not _read(path).strip():
        pytest.skip("questions.md absent/empty — no-op OQ pass (G10), attempted not skipped")
    entries, failures = _parse_collect(_read(path), "open_question", path)
    assert failures == [], _describe(failures)
    assert len(entries) >= 1
