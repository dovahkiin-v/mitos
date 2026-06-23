"""Phase 7a doc/comment reconciliation regression pin.

The V1b closeout (Phase 7a) is a documented *grep gate*: it re-greps every
behaviour-phase doc fix and confirms each now reads accurately. That gate is a
judgement pass — most surviving phrase-occurrences are *accurate* corrected or
historical text (e.g. "two kill-edges" as a count, "no transitive walker — that
is V5"), so a mechanical phrase-absence assert over the whole tree would
false-fail and tempt scrubbing accurate history. It is therefore captured as
evidence in IMPLEMENTATION_NOTES, not as a brittle test suite (plan Decision 4).

The **one** exception is the orphan this phase owns: the ``slug_aliases``
forward-ref clauses in ``resolve_slug``'s two docstrings (``protocols.py`` +
``store.py``). The alias subsystem was *stripped* at vision design (§1.1/§7
negative space) — a renamed-away citation breaks loud as ``missing_target``,
never silently repaired. Those clauses promised a future that will not exist and
had **no** other test pinning their removal (the behaviour phases' fixes are
pinned by their own suites). This single keyless assert guards against a future
phase reintroducing a forward-ref to the stripped subsystem.

Mirrors the regression-assertion idiom of ``test_store.py``'s
``assert "deferred to V1b" not in caplog.text`` — pin a stale claim's *absence*.
"""

from pathlib import Path

import mitos.protocols
import mitos.store


def test_slug_aliases_forward_ref_absent_from_source() -> None:
    """Asserts the stripped ``slug_aliases`` subsystem is referenced nowhere in source.

    The alias-fallback subsystem was never built (vision §1.1/§7 negative space);
    ``resolve_slug``'s docstrings must not promise it. Reads the two source files
    that carried the forward-refs and asserts the token is gone — locking 7a's
    only actionable edit against future reintroduction.
    """
    targets = [
        Path(mitos.protocols.__file__),
        Path(mitos.store.__file__),
    ]
    for source_path in targets:
        text = source_path.read_text(encoding="utf-8")
        assert "slug_aliases" not in text, (
            f"{source_path.name} reintroduced a `slug_aliases` forward-ref to the "
            "stripped alias subsystem (vision §1.1/§7 negative space — a stale "
            "citation breaks loud as `missing_target`, never silently repaired)."
        )
