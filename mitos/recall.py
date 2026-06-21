"""Shared precedent-recall policy for the surface tools.

The confidence threshold and the response ``note`` that distinguishes a settled
precedent from loose neighbours / no match at all live here, so the MCP
``surface_decisions`` tool and its CLI twin ``mitos surface`` stay behaviourally
identical (AX P5: a capped list of mid-score neighbours looked identical to a real
hit, and an empty result looked identical to "precedent hiding below the cap").

Pure stdlib — no graph or network access.
"""

from typing import Optional, Tuple, List

# Top semantic score at/above which a match is treated as a real precedent rather than
# a loose neighbour. Calibrated to observed Gemini-embedding scores: settled precedents
# land >0.82, adjacent-but-unrelated neighbours <0.72, so 0.80 is the strong/weak
# boundary. It gates a *hint* to the agent, never correctness — the agent always has the
# raw scores and ``list_decisions`` for certainty. Tune here only.
SURFACE_STRONG_THRESHOLD: float = 0.75
SURFACE_WEAK_THRESHOLD: float = 0.60


def assess_surface_recall(
    *,
    semantic_ran: bool,
    top_score: Optional[float],
    result_count: int,
    scope: Optional[str],
    scope_decision_count: Optional[int],
    all_scopes: Optional[List[str]] = None,
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

    prefix = ""
    if scope and scope_decision_count == 0:
        prefix = f"Note: '{scope}' is an unused scope tag. "
        if all_scopes:
            prefix += f"Valid scopes are: {', '.join(all_scopes)}. "

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
                f"{prefix}Semantic recall unavailable — no precedent here. Safe to decide, then record it."
            )
        return None, (
            f"Semantic recall unavailable and nothing to fall back on — cannot confirm "
            f"precedent. Use {list_hint} (pure graph read) to check."
        )

    # Semantic ran with a real, confident hit.
    if result_count and (top_score is None or top_score >= SURFACE_STRONG_THRESHOLD):
        matches_phrase = "Here are results that matched semantically." if prefix else "Ranked top matches."
        return "strong", (
            f"{prefix}{matches_phrase} For the COMPLETE set of decisions in a scope — a "
            f"completeness pass — call {list_hint}."
        )

    # Semantic ran but it's in the Twilight Zone (loose neighbour / phrased differently)
    if result_count and top_score >= SURFACE_WEAK_THRESHOLD:
        shown = f"{top_score:.2f}" if top_score is not None else "?"
        matches_phrase = "Here are results that matched semantically (twilight zone" if prefix else "Twilight zone"
        return "weak", (
            f"{prefix}{matches_phrase}: top score {shown} is close. They might be family neighbours or "
            f"exact precedent phrased differently. Check carefully before deciding."
        )

    # Semantic ran but every match is garbage (off-axis)
    if result_count:
        shown = f"{top_score:.2f}" if top_score is not None else "?"
        if scope and scope_decision_count == 0:
            msg = f"{prefix}Top score {shown} is too low to be related. Treat as no-precedent and decide fresh."
        else:
            msg = f"Very likely off-axis: top score {shown} is too low to be related. The scope is populated, but nothing matches your query. Treat as no-precedent and decide fresh."
        return "none", msg

    # Semantic ran and returned nothing surfaceable.
    if scope and scope_decision_count == 0:
        return "none", (
            f"{prefix}No semantic match. Safe to decide, then record it."
        )
    return "none", (
        f"No semantic match for {scope_phrase} — likely no settled precedent. Decide "
        f"and record it, or call {list_hint} for a certain completeness check."
    )
