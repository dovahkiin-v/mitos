"""Shared precedent-recall policy for the surface tools.

The confidence threshold and the response ``note`` that distinguishes a settled
precedent from loose neighbours / no match at all live here, so the MCP
``surface_decisions`` tool and its CLI twin ``mitos surface`` stay behaviourally
identical (AX P5: a capped list of mid-score neighbours looked identical to a real
hit, and an empty result looked identical to "precedent hiding below the cap").

The policy is **surface-agnostic**: it emits the recall *signal* (which confidence
branch, whether the scope tag is unused) and words each pointer from a per-surface
table (CLI verbs vs MCP tool call-forms), so no CLI/MCP presentation knowledge
leaks into the policy itself (P7). The unused-scope vector is **bounded** — a
did-you-mean + a top-K busiest-first candidate slice + an overflow pointer to the
dedicated scope-discovery surface — never the full tag vocabulary on the hot path
(P11).

Pure stdlib — no graph or network access (``difflib`` for the did-you-mean).
"""

import difflib
from typing import Dict, Optional, Tuple

# Top semantic score at/above which a match is treated as a real precedent rather than
# a loose neighbour. Calibrated to observed Gemini-embedding scores: settled precedents
# land >0.82, adjacent-but-unrelated neighbours <0.72, so 0.80 is the strong/weak
# boundary. It gates a *hint* to the agent, never correctness — the agent always has the
# raw scores and ``list_decisions`` for certainty. Tune here only.
SURFACE_STRONG_THRESHOLD: float = 0.75
SURFACE_WEAK_THRESHOLD: float = 0.60

# How many live scope tags the bounded unused-scope vector lists (busiest first) before
# it stops and points at the dedicated discovery surface. Mirrors clamp_limit's working
# top-k. Keeps the vector bounded at P11 scale — never the full vocabulary on a response.
SURFACE_TOP_SCOPES: int = 5

# Did-you-mean similarity cutoff (difflib's stdlib default). Raise it if it suggests
# noise; below ~0.6 unrelated tags start matching.
SURFACE_DIDYOUMEAN_CUTOFF: float = 0.6

# Per-surface pointer wording. The policy never names a literal call-form; it references
# a key here and the surface supplies its own verb (CLI shell command) or tool call-form
# (MCP). ``complete_scope`` carries a ``{scope}`` placeholder. ``sync`` is the CLI
# ``mitos sync`` on *both* surfaces — there is no MCP sync tool, so the literal shell
# command is the only truthful pointer (a shell command is not the MCP-tool leak the T7
# gate forbids).
_SURFACE_POINTERS: Dict[str, Dict[str, str]] = {
    "cli": {
        "complete": "mitos list",
        "complete_scope": "mitos list --scope '{scope}'",
        "discovery": "mitos scopes",
        "sync": "mitos sync",
    },
    "mcp": {
        "complete": "list_decisions()",
        "complete_scope": "list_decisions(scope='{scope}')",
        "discovery": "list_scopes",
        "sync": "mitos sync",
    },
}


def _unused_scope_prefix(
    scope: str,
    scope_counts: Dict[str, Dict[str, int]],
    pointers: Dict[str, str],
) -> str:
    """Builds the bounded self-correction vector for an unused scope tag.

    The vector is, in order: the unused-tag statement, an optional did-you-mean (the
    nearest live tag by string similarity), a top-K busiest-first slice of the live
    vocabulary with an overflow pointer when it is truncated, and a static
    authored-but-unsynced hedge. Bounded to at most ``SURFACE_TOP_SCOPES`` tags + one
    overflow pointer — never the full vocabulary, regardless of corpus size (P11).

    Args:
        scope: The (unused) scope tag the caller passed.
        scope_counts: The live ``get_scope_counts`` map, already busiest-first ordered
            by ``order_scope_counts`` at the callsite. Keys are canonical casefolded
            scope tags.
        pointers: The active surface's pointer table (``_SURFACE_POINTERS[surface]``).

    Returns:
        The vector as a single trailing-spaced string, ready to prepend to whichever
        confidence branch fires.
    """
    parts = [f"'{scope}' is an unused scope tag."]

    live_tags = list(scope_counts.keys())  # already count-desc ordered at the callsite

    match = difflib.get_close_matches(
        scope.casefold(), live_tags, n=1, cutoff=SURFACE_DIDYOUMEAN_CUTOFF
    )
    if match:
        parts.append(f"Did you mean '{match[0]}'?")

    if live_tags:
        top = live_tags[:SURFACE_TOP_SCOPES]
        parts.append(f"Live scopes (busiest first): {', '.join(top)}.")
        if len(live_tags) > SURFACE_TOP_SCOPES:
            parts.append(f"Full map: {pointers['discovery']}.")

    parts.append(
        f"(or {pointers['sync']} if you just authored decisions in this scope.)"
    )
    return " ".join(parts) + " "


def assess_surface_recall(
    *,
    semantic_ran: bool,
    top_score: Optional[float],
    result_count: int,
    scope: Optional[str],
    scope_counts: Optional[Dict[str, Dict[str, int]]] = None,
    surface: str,
) -> Tuple[Optional[str], str]:
    """Classifies a surface result and builds the agent-facing note.

    Args:
        semantic_ran: Whether semantic ranking actually executed (embeddings + vector
            store available and the query succeeded). False means degraded mode.
        top_score: The highest score among the surfaced active matches, or None.
        result_count: How many active decisions are being returned.
        scope: The scope filter the caller passed, if any.
        scope_counts: The live ``get_scope_counts`` map (busiest-first via
            ``order_scope_counts``), used both as the unused-scope oracle (a scope
            absent from this map is unused) and as the did-you-mean / top-K candidate
            source. None when the callsite could not compute it (degraded calmly — never
            fabricate a typo hint).
        surface: ``"cli"`` or ``"mcp"`` — selects the pointer wording. Required keyword:
            no callsite may silently emit the wrong surface's call-forms (the T7 gate).

    Returns:
        A ``(confidence, note)`` pair. ``confidence`` is ``"strong"`` / ``"weak"`` /
        ``"none"`` when semantic ranking ran, else ``None`` (degraded). ``note`` is a
        one-line, action-oriented string — never the old boilerplate that printed
        identically on every response.
    """
    pointers = _SURFACE_POINTERS[surface]
    scope_phrase = f"scope '{scope}'" if scope else "this query"
    if scope:
        complete_hint = pointers["complete_scope"].format(scope=scope)
    else:
        complete_hint = pointers["complete"]

    # The unused-scope signal keys on live-vocabulary membership (≥1 active decision OR
    # ≥1 parked OQ — 3a's get_scope_counts definition), NOT on an active-decision count:
    # a scope live only via a parked open question is a real tag, not a typo. MI-9
    # casefold; scope_counts keys are already canonical casefolded forms (fold only the
    # incoming scope). None scope_counts → treat as not-unused (calm degradation).
    scope_unused = (
        bool(scope)
        and scope_counts is not None
        and scope.casefold() not in scope_counts
    )
    # The bounded vector is orthogonal to confidence — a prefix prepended to whichever
    # confidence branch fires, exactly as the old prefix was.
    scope_prefix = (
        _unused_scope_prefix(scope, scope_counts, pointers) if scope_unused else ""
    )

    # Degraded — no semantic ranking happened.
    if not semantic_ran:
        if result_count:
            return None, (
                f"Semantic recall unavailable (embeddings/Qdrant down) — showing the "
                f"active decisions in {scope_phrase} as a fallback, NOT a relevance "
                f"ranking. For the authoritative set use {complete_hint} (pure graph read)."
            )
        if scope_unused:
            return None, (
                f"{scope_prefix}Semantic recall unavailable — no precedent here. Safe to decide, then record it."
            )
        return None, (
            f"Semantic recall unavailable and nothing to fall back on — cannot confirm "
            f"precedent. Use {complete_hint} (pure graph read) to check."
        )

    # Semantic ran with a real, confident hit.
    if result_count and (top_score is None or top_score >= SURFACE_STRONG_THRESHOLD):
        matches_phrase = "Here are results that matched semantically." if scope_prefix else "Ranked top matches."
        return "strong", (
            f"{scope_prefix}{matches_phrase} For the COMPLETE set of decisions in a scope — a "
            f"completeness pass — call {complete_hint}."
        )

    # Semantic ran but it's in the Twilight Zone (loose neighbour / phrased differently)
    if result_count and top_score >= SURFACE_WEAK_THRESHOLD:
        shown = f"{top_score:.2f}" if top_score is not None else "?"
        matches_phrase = "Here are results that matched semantically (twilight zone" if scope_prefix else "Twilight zone"
        return "weak", (
            f"{scope_prefix}{matches_phrase}: top score {shown} is close. They might be family neighbours or "
            f"exact precedent phrased differently. Check carefully before deciding."
        )

    # Semantic ran but every match is garbage (off-axis)
    if result_count:
        shown = f"{top_score:.2f}" if top_score is not None else "?"
        if scope_unused:
            msg = f"{scope_prefix}Top score {shown} is too low to be related. Treat as no-precedent and decide fresh."
        else:
            msg = f"Very likely off-axis: top score {shown} is too low to be related. The scope is populated, but nothing matches your query. Treat as no-precedent and decide fresh."
        return "none", msg

    # Semantic ran and returned nothing surfaceable.
    if scope_unused:
        return "none", (
            f"{scope_prefix}No semantic match. Safe to decide, then record it."
        )
    return "none", (
        f"No semantic match for {scope_phrase} — likely no settled precedent. Decide "
        f"and record it, or call {complete_hint} for a certain completeness check."
    )
