"""Shared precedent-recall policy for the surface tools.

The confidence threshold and the response ``note`` that distinguishes a settled
precedent from loose neighbours / no match at all live here, so the MCP
``surface_decisions`` tool and its CLI twin ``mitos surface`` stay behaviourally
identical (AX P5: a capped list of mid-score neighbours looked identical to a real
hit, and an empty result looked identical to "precedent hiding below the cap").

Pure stdlib — no graph or network access.
"""

from typing import Optional, Tuple

# Top semantic score at/above which a match is treated as a real precedent rather than
# a loose neighbour. Calibrated to observed Gemini-embedding scores: settled precedents
# land >0.82, adjacent-but-unrelated neighbours <0.72, so 0.80 is the strong/weak
# boundary. It gates a *hint* to the agent, never correctness — the agent always has the
# raw scores and ``list_decisions`` for certainty. Tune here only.
SURFACE_CONFIDENCE_THRESHOLD: float = 0.80


def assess_surface_recall(
    *,
    semantic_ran: bool,
    top_score: Optional[float],
    result_count: int,
    scope: Optional[str],
    scope_decision_count: Optional[int],
) -> Tuple[Optional[str], str]:
    """Classifies a surface result and builds the agent-facing note.

    Args:
        semantic_ran: Whether semantic ranking actually executed (embeddings + vector
            store available and the query succeeded). False means degraded mode.
        top_score: The highest score among the surfaced active matches, or None.
        result_count: How many active decisions are being returned.
        scope: The scope filter the caller passed, if any.
        scope_decision_count: Count of active decisions in ``scope`` when known (used to
            tell "this scope tag is unused" from "populated but nothing matched"); None
            when not computed.

    Returns:
        A ``(confidence, note)`` pair. ``confidence`` is ``"strong"`` / ``"weak"`` /
        ``"none"`` when semantic ranking ran, else ``None`` (degraded). ``note`` is a
        one-line, action-oriented string — never the old boilerplate that printed
        identically on every response.
    """
    list_hint = "list_decisions(scope=...)" if scope else "list_decisions()"
    scope_phrase = f"scope '{scope}'" if scope else "this query"

    # Degraded — no semantic ranking happened.
    if not semantic_ran:
        if result_count:
            return None, (
                f"Semantic recall unavailable (embeddings/Qdrant down) — showing the "
                f"active decisions in {scope_phrase} as a fallback, NOT a relevance "
                f"ranking. For the authoritative set use {list_hint} (pure graph read)."
            )
        if scope and scope_decision_count == 0:
            return None, (
                f"Semantic recall unavailable, and {scope_phrase} contains 0 decisions "
                f"(tag unused) — no precedent here. Safe to decide, then record it."
            )
        return None, (
            f"Semantic recall unavailable and nothing to fall back on — cannot confirm "
            f"precedent. Use {list_hint} (pure graph read) to check."
        )

    # Semantic ran with a real, confident hit.
    if result_count and (top_score is None or top_score >= SURFACE_CONFIDENCE_THRESHOLD):
        return "strong", (
            "Ranked top matches only (semantic, capped). For the COMPLETE set of "
            f"decisions in a scope — a completeness pass, not just the most relevant "
            f"few — call {list_hint}."
        )

    # Semantic ran but every match is below the confidence bar.
    if result_count:
        shown = f"{top_score:.2f}" if top_score is not None else "?"
        return "weak", (
            f"No strong precedent: top semantic score {shown} is below "
            f"{SURFACE_CONFIDENCE_THRESHOLD:.2f}, so the matches below are loose "
            f"neighbours, not a settled decision on {scope_phrase}. Treat as "
            f"no-precedent and decide (then record), or call {list_hint} to be certain."
        )

    # Semantic ran and returned nothing surfaceable.
    if scope and scope_decision_count == 0:
        return "none", (
            f"No semantic match, and {scope_phrase} contains 0 decisions (tag unused) "
            f"— no precedent. Safe to decide, then record it."
        )
    return "none", (
        f"No semantic match for {scope_phrase} — likely no settled precedent. Decide "
        f"and record it, or call {list_hint} for a certain completeness check."
    )
