"""Shared precedent-recall policy for the recall and scope hard-filter tools.

The confidence threshold and the response ``note`` that distinguishes a settled
precedent from loose neighbours / no match at all live here, so the MCP
``surface_decisions`` tool and its CLI twin ``mitos surface`` stay behaviourally
identical (AX P5: a capped list of mid-score neighbours looked identical to a real
hit, and an empty result looked identical to "precedent hiding below the cap").
The **targeted** recall verbs — the MCP ``query_decisions`` tool and its CLI twin
``mitos query`` — take the same band from :func:`_classify_recall` and their own
register from :func:`assess_query_recall`: one classification, four surfaces. The
register is separate because the question is: ``surface`` answers *is there
precedent near this?* and may tell a caller to decide off a miss, while ``query``
answers *did this lookup find the thing I named?* and says nothing about the
corpus — the same wording on both is how an agent that reached for the wrong verb
mints a duplicate of a decision the corpus already holds.

The same unused-scope self-correction also serves the **scope hard-filter** read
verbs — ``mitos list`` / ``mitos open-questions`` and the MCP ``list_decisions``
tool — via ``scope_filter_recovery``: when one of those returns an empty result
because the scope tag is absent from the live vocabulary, the caller surfaces the
bounded vector instead of a silent empty (the false-negative an agent reads as a
real absence rather than its own typo).

The **rejection section** of the four ranked reads is composed here too
(:func:`rejection_section`): its floor and bound, the state token naming a section
that was not computed whole (gap, not searched, unavailable, short) with its note
clause and the gap's per-surface heal, and the withheld rule counted over decisions.
The walk that found the matches and the shaper that words them are handed in. A
degraded answer's section is matched by words and carries a fifth token, ``lexical``
(``degraded`` composes it; its clause is :func:`degraded_section_clause`).

The **field-scoped lookup** (``query``'s ``field`` selector, :func:`rejection_lookup`)
is composed here as well: the rejection ranking as the whole answer, its band legend,
its edge counted in owners, its short clause, and the lexical answer's not-computed
tokens and reasons.

**One composer orders every line of a note** (6b). :func:`compose_note` is the one place
line order is decided (:data:`NOTE_SLOTS`: rejection lead, retired lead, legend, top
cluster, edge or short, absence, then the omissions and scan faults); each line arrives
worded by the function that owns it. The four ranked surfaces call
:func:`compose_ranked`, which also builds ``band_evidence`` (:func:`band_evidence`: the
top score, the top cluster, the edge, beside ``confidence``) and applies the blackout
and unbuilt-graph overrides at their recorded precedence; the lookup composes through
it in :func:`rejection_lookup`, and ``degraded`` through it for the three degraded
exits. A band's note describes the ranking's shape in measured quantities; a verdict
stays only where the band's measured precision supports it, and every band that does
not settle the question says how an absence is concluded (:func:`absence_clause`).

The policy is **surface-agnostic**: it emits the recall *signal* (which confidence
branch, whether the scope tag is unused) and words each pointer from a per-surface
table (CLI verbs vs MCP tool call-forms), so no CLI/MCP presentation knowledge
leaks into the policy itself (P7). The unused-scope vector is **bounded** — a
did-you-mean + a top-K busiest-first candidate slice + an overflow pointer to the
dedicated scope-discovery surface — never the full tag vocabulary on the hot path
(P11).

Pure stdlib — no graph or network access — plus **one admitted ``mitos`` import**,
the import-free ``shell_quote`` leaf (``difflib`` for the did-you-mean; pinned by
``test_recall_stays_a_zero_mitos_import_leaf``). Where a policy here needs a fact only another
module can compute, the fact is *injected* rather than imported:
:func:`missing_graph_is_a_gap` takes the corpus scan as a required keyword. That
is not ceremony — ``conflict.py`` imports this module, so an import added here
lands inside the check family's discovered no-write fence
(``test_conflict_closeout.py``), and a predicate the sweep never calls has no
business widening it.
"""

import difflib
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from mitos.shell_quote import shell_arg, shell_flag

# The ``strong`` line of the confidence band over the top surfaced score. Calibrated,
# recorded in tests/golden/calibration/record.json with its measured precision on both
# corpora and the replay: strictly above every top score a query nothing answers reaches
# (family-neighbour and off-axis, query space), the higher corpus's boundary taken, so
# ``strong`` is rare and right rather than common and vague. The band gates a hint, never
# correctness. Method: tests/golden/_calibration_record.py ``strong_boundary``.
SURFACE_STRONG_THRESHOLD: float = 0.8087

# The ``weak``/``none`` line. Calibrated, recorded in tests/golden/calibration/record.json:
# at or below every top score a related query reaches (bullseye, handle-words,
# family-neighbour), the lower corpus's boundary taken, so ``none`` fires only where no
# corpus sees a near precedent. Method: tests/golden/_calibration_record.py ``weak_boundary``.
SURFACE_WEAK_THRESHOLD: float = 0.6375

# The top-cluster margin: a hit within this much of rank 1's score sits in the top cluster
# (``top_cluster_size``), where the score order stops saying which hit answers. Calibrated,
# recorded in tests/golden/calibration/record.json: the median depth at which a non-target
# outranks a labelled target, read on the dense run where clusters form. Its own constant,
# measured apart from the edge margin even where the two values agree. Method:
# tests/golden/_calibration_record.py ``top_cluster_margin_boundary``.
RANKED_TOP_CLUSTER_MARGIN: float = 0.0156

# The edge margin: a ranked window ends mid-cluster when its next live rank scores
# within this much of its last returned one (``window_edge``). Calibrated, recorded in
# tests/golden/calibration/record.json: the median depth at which a non-target outranks
# a labelled target (bullseye and handle-words, query space), read on the dense run
# beside Harbor's. Method: tests/golden/_calibration_record.py ``edge_margin_boundary``.
RANKED_EDGE_MARGIN: float = 0.0156

# The rejection-match floor: a rejection hit scoring below it is not a match, and the
# leg's walk stops at it. Calibrated, recorded in tests/golden/calibration/record.json:
# just above every hard negative (generic and vocabulary items, per-item shape, query
# space) on both corpora, the higher boundary taken, with its cry-wolf ratio read on the
# copy's replay and held to its ceiling in the same entry. Method:
# tests/golden/_calibration_record.py ``rejection_floor_boundary``.
REJECTION_MATCH_FLOOR: float = 0.8147

# The rejection section's bound, in items (not points): how many verified matches a
# ranked answer carries at most. A budget constant (vision §6.3's default), kept at 3 by
# 5b at the calibrated floor: on the copy's replay (run 20261017-0759-3c17335-dogfood-copy)
# rank 1 showed 6 matches and rank 2 one, every one earned, and rank 3 none — under the 10
# a lower bound needs; the section cost 0 tokens p50 and p90 per read (132-869 when shown).
REJECTION_SECTION_BOUND: int = 3

# How many live scope tags the bounded unused-scope vector lists (busiest first) before
# it stops and points at the dedicated discovery surface. Mirrors clamp_limit's working
# top-k. Keeps the vector bounded at P11 scale — never the full vocabulary on a response.
SURFACE_TOP_SCOPES: int = 5

# Did-you-mean similarity cutoff (difflib's stdlib default). Raise it if it suggests
# noise; below ~0.6 unrelated tags start matching.
SURFACE_DIDYOUMEAN_CUTOFF: float = 0.6

# Per-surface pointer wording. The policy never names a literal call-form; it references
# a key here and the surface supplies its own verb (CLI shell command) or tool call-form
# (MCP). ``complete_scope`` carries a ``{scope}`` placeholder and ``precedent_scan``'s CLI
# form a ``{project}`` one. ``sync`` is the unused-scope hedge's pointer. The CLI names
# the command. There is no MCP sync tool, and an agent handed a shell command runs it,
# so the MCP form names none: it says what would help and that a person with a shell
# performs it (``MISSING_GRAPH_POINTERS["mcp"]``' register). The CLI form stays bare —
# ``_unused_scope_prefix`` has no project in hand to selector it.
#
# ``precedent_scan`` is the ``query`` register's redirect: the verb that answers the
# precedent question a targeted lookup did not. It must exist under BOTH outer keys — a
# one-sided entry ships the redirect on one boundary and raises ``KeyError`` on the other.
# The CLI form carries a selector because a response note is read wherever the caller was
# standing and cwd stopped being a targeting channel at the flip; the ``{selector}`` slot
# is ``shell_quote.shell_flag("-p", project)`` at composition (shell-quoted, ``=``-bound
# when dash-led; it was ``repr`` until 0.19.1) because a legal
# registered name may carry a space and an unregistered target's ``project`` is a path —
# rendered bare, ``mitos surface -p /home/me/my projects/x`` fails misleadingly rather
# than loudly. The MCP form keeps the shipped bare call-form: ``_example_call`` is where
# addressing gets taught, and a recall note is not it.
#
# ``dereference`` is the by-handle read that restores what a thinned ranked answer
# withheld (:func:`withheld_clause`). Same selector and quoting rule as
# ``precedent_scan``; the CLI form puts ``--`` before the slug because a hand-authored
# slug may start with ``-`` (``renderer._show_forms``' idiom). The MCP form names a
# read tool and no shell command.
#
# ``dereference_id`` and ``history`` are the blackout's heals (:func:`blackout_note`,
# their one consumer): the by-handle read of one retired decision by its id — the id,
# because a retired decision may share its slug with the live one that replaced it —
# and the read of the retired history. The CLI forms carry the selector, on
# ``dereference``'s quoting rule; the MCP forms name read tools, never a shell
# command (a tool call is not one, so MCP may name it).
#
# ``limit_arg`` and ``full_top_arg`` spell the two knobs :func:`edge_clause` names,
# each with an ``{n}`` placeholder. They are argument spellings, not commands: the
# clause rides ``surface`` and ``query`` alike, so it cannot name a verb, and the CLI
# fragments are flags that parse on either verb as printed. Unwrapped on purpose —
# the backticks above mark a whole command a reader can run.
#
# ``exhaustive``, ``exhaustive_oneline`` and ``exhaustive_whole`` are the absence
# clause's reads (:func:`absence_clause`): the enumeration that can show an absence a
# ranking cannot. ``exhaustive_whole`` is the lookup's, at the default depth, which keeps
# ``rejected_paths`` (the one-line tier and ``brief`` carry none). The CLI forms carry
# the selector because the clause rides ``query`` notes, where a bare recipe is refused
# (G9). ``{scope}`` is the caller's scope when one was given (the ``complete_scope``
# idiom; shell-quoted on the CLI, where the form is a command to run), else the literal
# ``<scope>`` for the reader to fill.
_SURFACE_POINTERS: Dict[str, Dict[str, str]] = {
    "cli": {
        "complete": "mitos list",
        "complete_scope": "mitos list --scope '{scope}'",
        "dereference": "`mitos show {selector} -- <slug>`",
        "dereference_id": "`mitos show {selector} -- <id>`",
        "discovery": "mitos scopes",
        "exhaustive": "`mitos list {selector} --scope {scope}`",
        "exhaustive_oneline": "`mitos list {selector} --scope {scope} --oneline`",
        "exhaustive_whole": "`mitos list {selector} --scope {scope}`",
        "full_top_arg": "--full-top {n}",
        "history": "`mitos list {selector} --state all`",
        "limit_arg": "--limit {n}",
        "precedent_scan": "`mitos surface {selector}`",
        "state_all": "mitos list --scope '{scope}' --state all",
        "sync": "mitos sync",
    },
    "mcp": {
        "complete": "list_decisions()",
        "complete_scope": "list_decisions(scope='{scope}')",
        "dereference": "show_node(ident='<slug>')",
        "dereference_id": "show_node(ident='<id>')",
        "discovery": "list_scopes",
        "exhaustive": "list_decisions(scope='{scope}')",
        "exhaustive_oneline": "list_decisions(scope='{scope}', oneline=True)",
        "exhaustive_whole": "list_decisions(scope='{scope}')",
        "full_top_arg": "full_top={n}",
        "history": "list_decisions(state='all')",
        "limit_arg": "limit={n}",
        "precedent_scan": "surface_decisions()",
        "state_all": "list_decisions(scope='{scope}', state='all')",
        "sync": "a sync, which a person with a shell in that project runs,",
    },
}


# Per-surface wording for the unbuilt-graph signal, on the ``_SURFACE_POINTERS``
# idiom: the policy emits the signal and composes the shared sentence, each surface
# supplies its own closing clause. The heal is a full rebuild on both, because the
# corpus is buffer plus archives and only ``mitos rebuild`` replays the archives —
# ``mitos sync`` reads the buffer alone and reconstructs nothing from a rotated
# corpus. The register differs by boundary. The CLI names the command with its
# selector, ``{selector}`` filled by ``shell_flag`` at composition (the
# ``precedent_scan`` idiom above). The MCP clause names no command and no tool: it
# states the fact and a person as the next actor, because an agent handed a shell
# command runs it and there is no rebuild tool to hunt for.
#
# `mitos reconcile` is NOT a pointer here and must never become one: over an unbuilt
# graph it judges no point, deletes nothing, refuses naming `mitos rebuild` and exits
# 1 — so pointing at it sends the reader through a refusal to the heal this clause
# names directly.
MISSING_GRAPH_POINTERS: Dict[str, Dict[str, str]] = {
    "cli": {
        "build": "run `mitos rebuild {selector}` to build it from the corpus",
    },
    "mcp": {
        "build": (
            "only a full rebuild from the corpus builds it, which no tool on this "
            "surface performs — a person with a shell in that project runs it"
        ),
    },
}


# Per-surface wording for the rejection index's gap, on ``MISSING_GRAPH_POINTERS``'
# idiom: the shared sentence states the fact, each surface supplies its heal. The heal
# is ``mitos reconcile``, which builds an absent rejection collection from the graph
# (the gate, ``missing_rejection_index_is_a_gap``, keys on the same expected set it
# fills). The CLI names the command with its selector (``shell_flag`` at composition).
# The MCP clause names no command and no tool: it states the fact and a person as the
# next actor, because an agent handed a shell command runs it and there is no tool
# that builds a collection (ADR ``collection-heal-stays-operator-invoked-never-implicit``).
REJECTION_GAP_POINTERS: Dict[str, Dict[str, str]] = {
    "cli": {
        "build": "`mitos reconcile {selector}` builds it",
    },
    "mcp": {
        "build": (
            "no tool on this surface builds it — a person with a shell in that "
            "project runs the collection heal"
        ),
    },
}


def _unused_scope_prefix(
    scope: str,
    scope_counts: Dict[str, Dict[str, int]],
    pointers: Dict[str, str],
    *,
    include_state_all: bool = False,
) -> str:
    """Builds the bounded self-correction vector for an unused scope tag.

    The vector is, in order: the unused-tag statement, an optional did-you-mean (the
    nearest live tag by string similarity), a top-K busiest-first slice of the live
    vocabulary with an overflow pointer when it is truncated, an optional ``--state
    all`` pointer (the hard-filter verbs only — a dead/archived-only scope whose
    decisions are all superseded is absent-from-live, so this is its recovery), and a
    static authored-but-unsynced hedge. Bounded to at most ``SURFACE_TOP_SCOPES`` tags
    + one overflow pointer — never the full vocabulary, regardless of corpus size (P11).

    Args:
        scope: The (unused) scope tag the caller passed.
        scope_counts: The live ``get_scope_counts`` map, already busiest-first ordered
            by ``order_scope_counts`` at the callsite. Keys are canonical casefolded
            scope tags.
        pointers: The active surface's pointer table (``_SURFACE_POINTERS[surface]``).
        include_state_all: When True, append a ``--state all`` pointer — the hard-filter
            verbs (``list`` / ``open-questions`` / ``list_decisions``) offer this extra
            affordance for a dead scope whose decisions are all superseded. The scope-blind
            ``surface`` path leaves it False, so its output stays byte-identical (3c).

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

    if include_state_all:
        parts.append(
            f"To see superseded decisions in this scope: "
            f"{pointers['state_all'].format(scope=scope)}."
        )

    parts.append(
        f"(or {pointers['sync']} if you just authored decisions in this scope.)"
    )
    return " ".join(parts) + " "


def _scope_is_unused(
    scope: Optional[str],
    scope_counts: Optional[Dict[str, Dict[str, int]]],
) -> bool:
    """Tests whether a scope tag is absent from the live vocabulary.

    The unused-scope signal keys on live-vocabulary membership (≥1 active decision OR
    ≥1 parked OQ — 3a's ``get_scope_counts`` definition), NOT on an active-decision
    count: a scope live only via a parked open question is a real tag, not a typo.
    MI-9 casefold — ``scope_counts`` keys are already canonical casefolded forms, so
    only the incoming scope is folded.

    Args:
        scope: The scope filter the caller passed, if any.
        scope_counts: The live ``get_scope_counts`` map, or None when the callsite
            could not compute it.

    Returns:
        True iff a non-empty scope is given, the map was computed, and the scope is
        absent from it. A None scope or None map degrades calmly to False (never
        fabricate a typo hint).
    """
    return (
        bool(scope)
        and scope_counts is not None
        and scope.casefold() not in scope_counts
    )


def scope_filter_recovery(
    *,
    scope: Optional[str],
    scope_counts: Optional[Dict[str, Dict[str, int]]],
    surface: str,
) -> Optional[Dict[str, str]]:
    """Recovery payload for an absent-from-live scope hard-filter read, else None.

    The scope hard-filter verbs (``mitos list`` / ``mitos open-questions`` / the MCP
    ``list_decisions`` tool) consult this on their **empty-result** branch: when a
    scoped read came back empty AND the scope is absent from the live vocabulary, the
    bounded self-correction vector replaces the silent empty. A scope that is live but
    empty for this particular read, or a no-scope read, recovers nothing — the caller
    keeps its honest-empty line/envelope.

    Args:
        scope: The scope filter the caller passed, if any.
        scope_counts: The live ``get_scope_counts`` map (busiest-first via
            ``order_scope_counts``), or None when the callsite could not compute it
            (degraded calmly — never fabricate a typo hint).
        surface: ``"cli"`` or ``"mcp"`` — selects the pointer wording. Required keyword:
            no callsite may silently emit the wrong surface's call-forms (the T7 gate).

    Returns:
        ``{"note": <bounded vector string>}`` when the scope is absent from the live
        vocabulary; None when there is nothing to recover from (scope is None, the
        scope is live, or ``scope_counts`` could not be computed).
    """
    if not _scope_is_unused(scope, scope_counts):
        return None
    pointers = _SURFACE_POINTERS[surface]
    note = _unused_scope_prefix(
        scope, scope_counts, pointers, include_state_all=True
    ).rstrip()
    return {"note": note}


# The two kinds of edge a ranked answer states (``WindowEdge.kind``).
EDGE_KIND_MID_CLUSTER = "mid_cluster"
EDGE_KIND_UNMEASURED = "unmeasured"

# The end values of the shared walk this module reads. Spelled here because this
# module may not import ``retrieval_walk``; rows pin them equal to
# ``retrieval_walk.EDGE_UNMEASURED`` / ``retrieval_walk.EDGE_FOUND``.
_WALK_END_UNMEASURED = "edge_unmeasured"
_WALK_END_EDGE_FOUND = "edge_found"


def _ends_mid_cluster(last: float, next_score: float, *, margin: float) -> bool:
    """The edge rule's one comparison: the next rank sits within ``margin`` of the last.

    Strict, so a gap equal to the margin is a clear drop; a negative gap (the next rank
    scoring higher, which an unordered list can give) is mid-cluster. Shared by
    :func:`window_edge` and :func:`band_evidence`, so the clause and the evidence cannot
    disagree about one walk.
    """
    return last - next_score < margin


class WindowEdge(NamedTuple):
    """How a ranked answer's window ended, when that is worth a sentence.

    Attributes:
        kind: ``EDGE_KIND_MID_CLUSTER`` or ``EDGE_KIND_UNMEASURED``.
        limit: The clamped window this ranking filled.
        held: How many ranks stay whole if the caller widens the window (0 means
            the answer is axiom-only throughout); None when no lever is named — at
            the ceiling, or on an unmeasured edge.
    """

    kind: str
    limit: int
    held: Optional[int]


def window_edge(
    walk: "object", *, margin: float, ceiling: int, full_top: Optional[int]
) -> Optional[WindowEdge]:
    """Decides what a ranked answer says about the rank just past its window.

    The ONE place the edge rule is made, for every ranked surface. ``walk`` is the
    shared walk's result (``retrieval_walk.RankedWalk``), duck-typed: ``window``,
    ``edge``, ``end`` and ``limit`` are read. The rule:

    * an empty window says nothing;
    * with an edge, the window ended **mid-cluster** when the last returned score
      minus the edge's is below ``margin`` (strict, so a gap equal to it is a clear
      drop). A negative gap — the next rank scoring higher, which an unordered list
      can give — is mid-cluster too, on purpose;
    * with no edge because the fetch reached its bound holding exactly ``limit`` live
      decisions, the edge is **unmeasured**, and it is said, because left unsaid it
      would read as a clear drop;
    * otherwise (a short window, or a fetch that ran out, full or not) nothing.

    The lever — a higher ``limit`` with ``full_top`` held — is named only on a
    mid-cluster edge below ``ceiling``. Not on an unmeasured one: the walk fetches its
    fixed bound whatever ``limit`` is, so a higher ``limit`` re-reads the same fetch
    and comes back short. ``margin`` and ``ceiling`` are required and injected by the
    boundary (``RANKED_EDGE_MARGIN``, ``display.RANKED_LIMIT_CEILING``): a later
    ranking with a margin of its own passes it here, and a default would let it read
    the wrong one silently.

    Args:
        walk: The ranked walk the answer was composed from.
        margin: The edge margin of the ranking the walk came from.
        ceiling: The clamp ceiling a higher ``limit`` would be cut back to.
        full_top: The effective ``full_top`` after the boundary normalised ``brief``
            to 0; None when the caller set no cutoff.

    Returns:
        A :class:`WindowEdge`, or None when the clause must stay absent.
    """
    window = walk.window
    if not window:
        return None
    limit = walk.limit
    if walk.edge is not None:
        if not _ends_mid_cluster(window[-1].score, walk.edge.score, margin=margin):
            return None
        if limit >= ceiling:
            return WindowEdge(kind=EDGE_KIND_MID_CLUSTER, limit=limit, held=None)
        held = limit if full_top is None else min(full_top, limit)
        return WindowEdge(kind=EDGE_KIND_MID_CLUSTER, limit=limit, held=held)
    if walk.end == _WALK_END_UNMEASURED:
        return WindowEdge(kind=EDGE_KIND_UNMEASURED, limit=limit, held=None)
    return None


def top_cluster_size(scores: Sequence[float], *, margin: float) -> int:
    """Counts the hits that sit in the top cluster: within ``margin`` of rank 1's score.

    The ONE place top-cluster membership is decided. ``scores`` is a surfaced list's
    scores, best first; rank 1 is always a member, so a non-empty list counts at least
    one. Membership is strict, as :func:`window_edge`'s is: a hit whose score sits exactly
    ``margin`` below rank 1's is a clear drop, not a member. A hit scoring above rank 1 (an
    unordered list) is a member. ``margin`` is a required keyword because each ranking
    calibrates its own (``RANKED_TOP_CLUSTER_MARGIN`` for the decision ranking), and a
    default would let a second ranking read the wrong one silently.

    Args:
        scores: The surfaced hits' scores, rank 1 first.
        margin: The top-cluster margin of the ranking the scores came from.

    Returns:
        How many hits are in the top cluster; 0 for an empty list.
    """
    if not scores:
        return 0
    top = scores[0]
    return sum(1 for score in scores if top - score < margin)


# The ``edge`` tokens of ``band_evidence``, one per walk end over a non-empty window:
# ``mid_cluster`` / ``clear`` (an edge was found, and the next rank's gap to the last is
# below / at least the edge margin), ``unmeasured`` (the fetch stopped at its bound, so a
# further rank may exist) and ``end`` (the index holds no further rank).
EVIDENCE_EDGE_KINDS: Tuple[str, ...] = ("mid_cluster", "clear", "unmeasured", "end")


def band_evidence(walk: "object", *, scores: Sequence[float], top_cluster_margin: float,
                  edge_margin: float) -> Optional[Dict[str, Any]]:
    """Builds the measured quantities behind a band: the envelope's ``band_evidence``.

    The ONE place the object is built, for both rankings (the decision window and the
    lookup's item window). Present iff the band read a top score, which is iff the
    window is non-empty; every member that does not apply is absent, never null:

    * ``top_score`` — the window's maximum, the score :func:`_classify_recall` reads;
    * ``top_cluster`` — :func:`top_cluster_size` over the window at
      ``top_cluster_margin``, at least 1;
    * ``edge`` — one of :data:`EVIDENCE_EDGE_KINDS`, read off ``walk.end``;
    * ``next_score`` — the next rank's score, only where an edge was found. Absent
      means no next rank was measured, and ``edge`` says why.

    No admitted input rides it (6a admitted none), so the key set is exactly these
    four; a later admitted input adds its key here and inverts the row that pins them.

    Args:
        walk: The walk the window came from (duck-typed: ``end``, ``edge``).
        scores: The window's scores in rank order (rank 1 first; never a sorted copy).
        top_cluster_margin: The ranking's top-cluster margin.
        edge_margin: The ranking's edge margin.

    Returns:
        The evidence dict, or None for an empty window.

    Raises:
        ValueError: On a walk end outside the four the walk can reach — a call-site
            defect, loud rather than an evidence object naming no reason.
    """
    if not scores:
        return None
    end = walk.end
    evidence: Dict[str, Any] = {
        "top_score": max(scores),
        "top_cluster": top_cluster_size(scores, margin=top_cluster_margin),
    }
    if end == _WALK_END_EDGE_FOUND:
        next_score = walk.edge.score
        mid = _ends_mid_cluster(scores[-1], next_score, margin=edge_margin)
        evidence["edge"] = "mid_cluster" if mid else "clear"
        evidence["next_score"] = next_score
    elif end in (_WALK_END_UNMEASURED, _WALK_END_SHORT_BOUND):
        evidence["edge"] = "unmeasured"
    elif end == _WALK_END_EXHAUSTED:
        evidence["edge"] = "end"
    else:
        raise ValueError(f"band_evidence: unknown walk end {end!r}")
    return evidence


def top_cluster_clause(size: int, *, register: str, margin: float) -> Optional[str]:
    """Words the line a note carries when the top of its ranking is a cluster.

    A ranking fact and a read instruction, nothing more: the score order inside the
    cluster does not say which hit answers (6a: rank 1 right 10 of 15 times inside the
    margin, 99 of 101 outside), so the reader reads them all. No verdict and no write,
    so it is lawful in every register. The margin printed is the constant passed in,
    never a literal. Pure string formatting.

    Args:
        size: :func:`top_cluster_size` of the window.
        register: ``"surface"`` / ``"query"`` (decision hits) or ``"lookup"`` (rejected
            options).
        margin: The top-cluster margin the size was counted at.

    Returns:
        One sentence, or None when the top is a single hit.

    Raises:
        ValueError: On an unknown register.
    """
    if register in ("surface", "query"):
        noun = "hits"
    elif register == "lookup":
        noun = "rejected options"
    else:
        raise ValueError(f"top_cluster_clause: unknown register {register!r}")
    if size < 2:
        return None
    return (f"The top {size} {noun} score within {margin:g} of rank 1, so their order "
            f"does not say which one answers: read them all.")


def edge_clause(edge: WindowEdge, *, surface: str) -> str:
    """Words the sentence a ranked note ends with when its window's edge is stated.

    Verb-independent on purpose, as :func:`withheld_clause` is: ``surface`` and
    ``query`` gain the same characters on one boundary, so the query note stays no
    longer than the surface note it displaces. It speaks about this ranking's depth
    only — no verdict on the corpus, no write instruction, no claim that the next
    ranks score higher or matter more — so it is lawful in both registers. A named
    lever carries the held rank count as a number, so the re-send works in one turn;
    with nothing held it names ``limit`` alone, since naming ``full_top`` would hand
    a ``brief`` caller a re-send the boundary refuses. Pure string formatting, so it
    cannot raise inside a ranked path and pose as degraded recall.

    Args:
        edge: The edge from :func:`window_edge`.
        surface: ``"cli"`` or ``"mcp"`` — selects the argument spellings.

    Returns:
        One sentence, e.g. ``"This ranking ended mid-cluster at limit=5: the next rank
        scores as close as the last; a higher limit reaches it, and with full_top=5
        the added ranks come back axiom-only."``.
    """
    pointers = _SURFACE_POINTERS[surface]
    at = pointers["limit_arg"].format(n=edge.limit)
    if edge.kind == EDGE_KIND_UNMEASURED:
        return (
            f"This ranking's edge at {at} is unmeasured: the read fetched no further "
            f"live rank, so it cannot say whether the window ended mid-cluster."
        )
    stem = f"This ranking ended mid-cluster at {at}: the next rank scores as close as the last"
    if edge.held is None:
        return f"{stem}."
    if edge.held == 0:
        return f"{stem}; a higher limit reaches it, axiom-only like these."
    held = pointers["full_top_arg"].format(n=edge.held)
    return (
        f"{stem}; a higher limit reaches it, and with {held} the added ranks come "
        f"back axiom-only."
    )


# The two end values of the shared walk a short window can carry. Spelled here for
# the reason ``_WALK_END_UNMEASURED`` is; rows pin them equal to
# ``retrieval_walk.EXHAUSTED`` / ``retrieval_walk.SHORT_BOUND``.
_WALK_END_EXHAUSTED = "exhausted"
_WALK_END_SHORT_BOUND = "short_bound"

# The two kinds of short window (``ShortWindow.kind``).
SHORT_KIND_RAN_OUT = "ran_out"
SHORT_KIND_BOUND_RETIRED = "bound_retired"


class RetiredPartition(NamedTuple):
    """The retired decisions a ranked answer shows, as the walk passed them.

    Attributes:
        shown: The walk hits to render as handles, closest first, at most
            ``limit`` of them.
        omitted: How many further members were left out (the count an envelope
            states as ``retired_partition_omitted``).
        leads: True when the closest shown member outranks the top live hit —
            the case the lead line states.
    """

    shown: List[Any]
    omitted: int
    leads: bool


def retired_partition(walk: "object") -> RetiredPartition:
    """Bounds the walk's retired partition for presentation. The ONE place it is bounded.

    Membership is the walk's (``walk.passed``: the retired hits a full window
    outranked with its last live decision, or every retired hit when the window is
    short). The bound is a presentation budget and lives here, beside the edge
    rule: at most ``limit`` shown, closest first, the rest counted. ``walk`` is
    duck-typed (``passed``, ``window``, ``limit``).

    Args:
        walk: The shared walk's result (``retrieval_walk.RankedWalk``).

    Returns:
        A :class:`RetiredPartition`.
    """
    passed = walk.passed
    shown = list(passed[:walk.limit])
    window = walk.window
    leads = bool(window) and bool(shown) and shown[0].rank < window[0].rank
    return RetiredPartition(shown=shown, omitted=len(passed) - len(shown), leads=leads)


class ShortWindow(NamedTuple):
    """Why a ranked window holds fewer live decisions than ``limit``.

    Attributes:
        kind: ``SHORT_KIND_RAN_OUT`` (the fetch returned fewer points than its
            bound: the whole index was read) or ``SHORT_KIND_BOUND_RETIRED`` (the
            fetch reached its bound with the window still short).
        held: How many live decisions the window holds.
        limit: The clamped window size.
        returned: How many points the fetch returned.
        retired: How many of those points were retired decisions (a short walk
            never stops early, so this is every one in the fetch).
    """

    kind: str
    held: int
    limit: int
    returned: int
    retired: int


def short_window(walk: "object") -> Optional[ShortWindow]:
    """Decides whether a ranked answer says why its window is short.

    A window under ``limit`` over a fetch that returned at least one point says
    why, at every ``limit`` the ceiling included: the index ran out, or the bound
    was reached with the neighbourhood mostly retired. A full window, an empty
    fetch, or any other end says nothing (on a full window the edge rule speaks
    instead, so the two never co-occur). ``walk`` is duck-typed (``window``,
    ``limit``, ``returned``, ``end``, ``retired``).

    Args:
        walk: The shared walk's result (``retrieval_walk.RankedWalk``).

    Returns:
        A :class:`ShortWindow`, or None when the clause must stay absent.
    """
    held = len(walk.window)
    if held >= walk.limit or walk.returned < 1:
        return None
    if walk.end == _WALK_END_EXHAUSTED:
        kind = SHORT_KIND_RAN_OUT
    elif walk.end == _WALK_END_SHORT_BOUND:
        kind = SHORT_KIND_BOUND_RETIRED
    else:
        return None
    return ShortWindow(kind=kind, held=held, limit=walk.limit,
                       returned=walk.returned, retired=len(walk.retired))


def short_window_clause(short: ShortWindow, *, surface: str) -> str:
    """Words the sentence a ranked note carries when its window is short.

    Facts only, and no lever: on ``ran_out`` the whole index was read, and on
    ``bound_retired`` a higher ``limit`` re-reads the same fixed fetch, so naming
    either would be a false line. No write instruction and no verdict on the corpus,
    so it is lawful in both registers. Pure string formatting.

    Args:
        short: The short window from :func:`short_window`.
        surface: ``"cli"`` or ``"mcp"`` — selects the ``limit`` spelling.

    Returns:
        One sentence.
    """
    at = _SURFACE_POINTERS[surface]["limit_arg"].format(n=short.limit)
    noun = "decision" if short.held == 1 else "decisions"
    stem = f"This window is short, {short.held} live {noun} at {at}:"
    if short.kind == SHORT_KIND_RAN_OUT:
        points = "point" if short.returned == 1 else "points"
        return (f"{stem} the read reached the end of the index after "
                f"{short.returned} {points}.")
    return (
        f"{stem} the read stopped at its {short.returned}-point bound, and "
        f"{short.retired} of those points were retired decisions."
    )


def _successor_slugs(handle: Mapping[str, Any]) -> Optional[List[str]]:
    """Where a retired handle's current word lives, or None when unread.

    ``live_end`` when present (``[]`` stays ``[]``: the chain reaches no live
    decision); otherwise the immediate successor stamps, which are live by
    construction when no ``live_end`` was written; None when the handle carries no
    successor key at all (the stamps went unread).
    """
    if "live_end" in handle:
        return list(handle["live_end"])
    successors = list(handle.get("superseded_by", [])) + list(handle.get("corrected_by", []))
    return successors or None


def retired_lead_clause(handle: Mapping[str, Any], *, surface: str) -> str:
    """Words the line that opens a ranked note when a retired decision outranks rank 1.

    The read-path guard against re-proposing an abandoned approach: the closest
    match was retired, and here is where its current word lives. Names handle slugs
    only — no pointer, no selector — so it needs no config. When the stamps went
    unread it says the successor could not be read, and the closing
    ``stamps_unread`` clause gives the reason. Pure string formatting.

    Args:
        handle: The leading handle (``display.retired_handle``).
        surface: ``"cli"`` or ``"mcp"``; the wording is the same on both.

    Returns:
        One sentence.
    """
    if surface not in _SURFACE_POINTERS:
        raise KeyError(surface)
    stem = f"The closest match was retired: {handle['slug']} ({handle['state']})"
    successors = _successor_slugs(handle)
    if successors is None:
        return f"{stem}; its successor could not be read."
    if not successors:
        return f"{stem}; its kill chain reaches no live decision."
    return f"{stem}; its current word lives in {', '.join(successors)}."


def blackout_note(handles: List[Mapping[str, Any]], *, surface: str,
                  config: "object") -> str:
    """Builds the blackout note: the lead's zero-live form.

    Ranked recall reached precedents but every one is retired, which would
    otherwise read as a true miss — a false-novelty signal that costs the agent a
    re-derivation of a call the graveyard already settled. The note names each
    shown handle and where its current word lives (its live end, never a dead hop),
    then the reads that heal: one retired decision whole by its id, and the
    retired history. Each boundary words its own heal (``dereference_id`` /
    ``history``): the CLI names commands with the selector, MCP names read tools
    and no shell command. A blackout window is always short; the short-window
    clause follows it in :func:`compose_note`'s ``edge`` slot. Calm, one short
    block (P9). Pure string formatting.

    Args:
        handles: The shown handles (``display.retired_handle``), closest first.
        surface: ``"cli"`` or ``"mcp"``.
        config: The active ``MitosConfig`` (duck-typed). Read only for
            ``project``, which the CLI heals name as their selector.

    Returns:
        The blackout note string.
    """
    pointers = _SURFACE_POINTERS[surface]
    selector = shell_flag("-p", getattr(config, "project", ""))
    n = len(handles)
    parts: List[str] = []
    for h in handles:
        successors = _successor_slugs(h)
        if successors is None:
            parts.append(f"{h['slug']} ({h['state']})")
        elif not successors:
            parts.append(f"{h['slug']} ({h['state']}, no live successor)")
        else:
            parts.append(f"{h['slug']} ({h['state']} → {', '.join(successors)})")
    noun, verb = ("match", "is") if n == 1 else ("matches", "are")
    return (
        f"All {n} nearest {noun} {verb} retired — no live precedent on this claim, "
        f"but it was settled before. Retired: {'; '.join(parts)}. Read one whole "
        f"with {pointers['dereference_id'].format(selector=selector)}, or the retired "
        f"history with {pointers['history'].format(selector=selector)}."
    )


def _classify_recall(*, top_score: Optional[float], result_count: int) -> str:
    """Decides the confidence band. The ONE place that band choice is made.

    Reached only when ranking actually ran — a degraded read has no band, and the
    two legend producers keep that fence in different ways (``assess_surface_recall``
    returns before it; ``assess_query_recall`` has no degraded arm at all).

    It is not top-score arithmetic alone: ``result_count`` is read too, which is
    what separates a with-results ``"none"`` (everything ranked off-axis) from an
    empty one (nothing ranked at all) at the composers.

    The ``top_score is None`` clause fails **open** — a non-empty result set whose
    scores never reached the caller reads ``"strong"``. That is deliberate and
    inherited byte-for-byte: it keeps a surface that legitimately has no scores
    (the degraded scope-listing fallback) from being labelled off-axis. The
    consequence is that a caller which forgets to pass ``top_score`` ships an
    all-``strong`` band over every populated answer with nothing red, so the repair
    lives in the two composers that read it off the *surfaced* window themselves
    (:func:`compose_ranked`, :func:`rejection_lookup`; since 6b no site accumulates
    it) and in the tests that pin a superseded-filtered ranking.

    Args:
        top_score: The highest score among the surfaced matches, or None.
        result_count: How many matches are being returned.

    Returns:
        ``"strong"`` / ``"weak"`` / ``"none"``.
    """
    if result_count and (top_score is None or top_score >= SURFACE_STRONG_THRESHOLD):
        return "strong"
    if result_count and top_score >= SURFACE_WEAK_THRESHOLD:
        return "weak"
    return "none"


# The phrase every ``weak`` legend carries, on all three registers (D4 of 6b): ``weak``
# holds answers and non-answers of every class on both corpora (Harbor 3 of 19 right,
# dense 73 of 101), so what a score in it supports is exactly this and nothing narrower.
# One constant because the ranked-read descriptions are pinned to say the same thing.
WEAK_MEANING = "a score in this band does not tell the answer from a neighbour"


def assess_surface_recall(
    *,
    semantic_ran: bool,
    top_score: Optional[float],
    result_count: int,
    scope: Optional[str],
    scope_counts: Optional[Dict[str, Dict[str, int]]] = None,
    surface: str,
    scope_total: Optional[int] = None,
    degraded_reason: Optional[str],
) -> Tuple[Optional[str], str]:
    """Classifies a surface result and words its legend: the band's own sentence.

    The legend only. The lines around it — the leads, the top cluster, the edge, the
    absence clause and the omissions — are :func:`compose_ranked`'s, which calls this
    for the ``legend`` slot; on ``surface`` that slot holds the unused-scope prefix and
    the band sentence as one string, so both leads precede the prefix. A verdict stays
    only where 6a measured the band's precision supporting it: ``strong`` makes none,
    ``weak`` says :data:`WEAK_MEANING`, and ``none`` states the score and the ranking
    fact, because ``none`` fired on no fixture of either corpus and on no replayed read.

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
        scope_total: How many active decisions the scope holds, passed only by the
            degraded scope dump, whose sentence names how many of them it shows.
            A degraded call with results must pass it: the dump is the only
            degraded answer that has results.
        degraded_reason: The cause phrase the caller computed once with
            ``lexical.degraded_reason_from_error`` and also put on its degraded
            header, so the header and this note cannot disagree. Required keyword,
            never defaulted; read only when ``semantic_ran`` is False, so a ranked
            call passes None. Injected as a string: this module never classifies.

    Raises:
        ValueError: ``semantic_ran`` is False and ``degraded_reason`` is None — a
            call-site defect, loud on purpose rather than a note with no cause —
            or ``semantic_ran`` is False with results and no ``scope_total``: a
            degraded answer with results is the scope dump's.

    Returns:
        A ``(confidence, legend)`` pair. ``confidence`` is ``"strong"`` / ``"weak"`` /
        ``"none"`` when semantic ranking ran, else ``None`` (degraded). The dump's
        legend carries no omission clause: ``degraded.dump_section`` places
        :func:`degraded_omission_clause` in :func:`compose_note`'s ``not_computed``
        slot, beside the matching ``not_computed`` key.
    """
    pointers = _SURFACE_POINTERS[surface]
    scope_phrase = f"scope '{scope}'" if scope else "this query"
    if scope:
        complete_hint = pointers["complete_scope"].format(scope=scope)
    else:
        complete_hint = pointers["complete"]

    # The unused-scope signal keys on live-vocabulary membership (≥1 active decision OR
    # ≥1 parked OQ — 3a's get_scope_counts definition), NOT on an active-decision count:
    # a scope live only via a parked open question is a real tag, not a typo. Shared with
    # scope_filter_recovery via _scope_is_unused (the hard-filter verbs key on the same
    # oracle). None scope_counts → treat as not-unused (calm degradation).
    scope_unused = _scope_is_unused(scope, scope_counts)
    # The bounded vector is orthogonal to confidence — a prefix prepended to whichever
    # confidence branch fires, exactly as the old prefix was.
    scope_prefix = (
        _unused_scope_prefix(scope, scope_counts, pointers) if scope_unused else ""
    )

    # Degraded — no semantic ranking happened.
    if not semantic_ran:
        if degraded_reason is None:
            raise ValueError(
                "assess_surface_recall: a degraded call must pass degraded_reason — "
                "the cause the caller's degraded header names"
            )
        # A degraded answer with results is the scope dump's, and only the dump's:
        # with recall down no walk ran, so nothing else fills the list. Its
        # composer (``degraded.dump_section``) sets the matching ``not_computed``,
        # so "clause placed" and "key set" are one fact (WIRING_LEDGER entry-014).
        if result_count and scope_total is None:
            raise ValueError(
                "assess_surface_recall: a degraded answer with results is the scope "
                "dump's; pass scope_total"
            )
        # The no-results branches carry no omission clause — no surface reaches
        # them, since both route an empty degraded answer to the lexical exit, which
        # states its own.
        if result_count:
            # The dump's own branch: it names how many of the scope's decisions it
            # shows. ``result_count`` is what the envelope returns; the dump is one
            # read, so it is either whole (up to ``limit``) or not shown at all.
            if result_count < scope_total:
                shown_part = (
                    f"{result_count} of the {scope_total} active decisions in {scope_phrase}"
                )
            elif scope_total == 1:
                shown_part = f"the one active decision in {scope_phrase}"
            else:
                shown_part = f"all {scope_total} active decisions in {scope_phrase}"
            return None, (
                f"Semantic recall unavailable ({degraded_reason}) — showing "
                f"{shown_part} as a fallback, NOT a relevance ranking. For the "
                f"authoritative set use {complete_hint} (pure graph read)."
            )
        if scope_unused:
            return None, (
                f"{scope_prefix}Semantic recall unavailable — no precedent here. Safe to decide, then record it."
            )
        return None, (
            f"Semantic recall unavailable and nothing to fall back on — cannot confirm "
            f"precedent. Use {complete_hint} (pure graph read) to check."
        )

    band = _classify_recall(top_score=top_score, result_count=result_count)
    shown = f"{top_score:.2f}" if top_score is not None else "?"

    # Semantic ran with a real, confident hit. No verdict: it names the ranking and
    # the completeness read.
    if band == "strong":
        matches_phrase = "Here are results that matched semantically." if scope_prefix else "Ranked top matches."
        return "strong", (
            f"{scope_prefix}{matches_phrase} For the COMPLETE set of decisions in a scope — a "
            f"completeness pass — call {complete_hint}."
        )

    # The twilight band: answers and non-answers of every class sit here, so the
    # legend says the score cannot tell them apart, and asks for a read.
    if band == "weak":
        matches_phrase = "Here are results that matched semantically. Twilight zone" if scope_prefix else "Twilight zone"
        return "weak", (
            f"{scope_prefix}{matches_phrase}: top score {shown} is close, but "
            f"{WEAK_MEANING}. Read the hits before relying on rank 1, and check "
            f"carefully before deciding."
        )

    # The lowest band with results: the score and the ranking fact, no verdict on
    # the corpus — ``none`` has no measured support, and it reads on queries aimed at
    # a real rejection. The absence clause follows it (``compose_ranked``).
    if result_count:
        stem = f"{scope_prefix}Top score" if scope_unused else "Off-axis: top score"
        return "none", (
            f"{stem} {shown} sits in the lowest band — nothing ranked close to this "
            f"query."
        )

    # Nothing ranked at all. Scope never filters the ranking, so the legend names
    # none; the absence clause that follows names the scope's exhaustive read.
    return "none", f"{scope_prefix}No semantic match — no decision ranked for this query."


def assess_query_recall(
    *,
    top_score: Optional[float],
    result_count: int,
    config: "object",
    surface: str,
) -> Tuple[str, str]:
    """Classifies a targeted-lookup result and words its ``query``-register legend.

    The band is :func:`_classify_recall`'s — the same three tokens
    ``assess_surface_recall`` emits, because a band that meant different things on
    two verbs would be worse than no band. The **legend** is not shared, and none of
    it may be inherited: ``surface``'s wording may route a caller to decide, and every
    such clause is a claim a targeted lookup did not measure. On this verb the band
    describes **this lookup's ranking** and nothing else, and the corpus question is
    routed to the verb that answers it. No legend here tells the caller to decide or to
    record: a precedent that is in the graph and absent from the ranking is an
    ordinary state, and answering it with a write instruction is how a duplicate gets
    minted. The lines around the legend are :func:`compose_ranked`'s.

    Two parameters are absent on purpose. There is no ``semantic_ran``: every
    degraded route on both call sites returns before assessment, so a composer with
    no degraded arm cannot author a degraded-register sentence at all. And there is
    no ``scope``: it arrives structurally ``None`` on both surfaces, and a parameter
    that can only ever be ``None`` grows branches nobody can reach.

    Args:
        top_score: The highest score among the **surfaced** matches — the list the
            superseded filter already thinned, never the raw vector-store return.
            None when nothing was surfaced.
        result_count: How many matches are being returned.
        config: The active ``MitosConfig`` (duck-typed to avoid an import cycle —
            recall is a leaf module, as in :func:`corpus_provenance`). Read only
            for ``project``, which the CLI redirect names as its selector.
        surface: ``"cli"`` or ``"mcp"`` — selects the pointer wording. Required
            keyword: no call site may silently emit the other surface's call-forms.

    Returns:
        A ``(confidence, legend)`` pair. ``confidence`` is ``"strong"`` / ``"weak"``
        / ``"none"`` and is **never** None — see the degraded fence above.
    """
    redirect = _precedent_redirect(surface=surface, config=config)

    band = _classify_recall(top_score=top_score, result_count=result_count)

    # Legend only, and it names no verb: the caller asked for a thing and got it, so
    # a redirect here is a per-answer turn tax wearing a legend. It still carries a
    # note — this is the band a caller meets most often, and dropping it leaves the
    # common answer with no legend at all. Kept against 6a's measurement: ``strong``
    # is rare and right on both instruments.
    if band == "strong":
        return band, ("Top match cleared the strong threshold — a real hit for this "
                      "lookup, not a loose neighbour.")

    shown = f"{top_score:.2f}" if top_score is not None else "?"

    if band == "weak":
        return band, (f"Twilight zone: top score {shown} — close, maybe not your "
                      f"handle; {WEAK_MEANING}. {redirect}")

    # The ``"none"`` band forks the same way its sibling does: matches that all
    # ranked low, versus nothing ranked at all. Neither states a verdict on the
    # corpus, and neither instructs a write.
    if result_count:
        return band, (f"Off-axis: top score {shown} sits in the lowest band — nothing "
                      f"ranked close. {redirect}")
    return band, f"No semantic match — this ranking returned nothing. {redirect}"


def _precedent_redirect(*, surface: str, config: "object") -> str:
    """The ``query`` and lookup registers' redirect to the verb that answers precedent.

    Names a question, not a re-run: re-issuing this same string on the other verb buys
    the same embedding, the same ranking and the same matches for the price of a turn.
    What earns the turn is the precedent question stated as the claim the caller is
    about to commit to — a different string.
    """
    pointer = _SURFACE_POINTERS[surface]["precedent_scan"].format(
        selector=shell_flag("-p", getattr(config, "project", ""))
    )
    return f"For whether precedent exists, restate the claim and ask {pointer}."


def rejection_lead(matches: Sequence[Mapping[str, Any]], *,
                   top_score: Optional[float]) -> Optional[Mapping[str, Any]]:
    """Picks the rejection match that leads a ranked note, or None.

    The first shaped match (section order is score order) whose owner is live — its
    ``letter`` is ``here``, ``section`` or ``window``, never ``retired`` — and whose
    score is strictly above the top live hit's. With no live hit (``top_score`` None)
    the first live owner's match leads, ahead of a blackout. A retired owner's match
    stays in the section as its handle and never leads (ADR
    ``ranked-read-note-leads-with-a-live-owners-rejection-then-a-closer-retired-decision``).
    The owner's own rank is never read: an owner at window rank 4 leads when its item
    outranks rank 1 (ADR
    ``rejection-lead-binds-to-its-owners-question-and-is-never-gated-on-the-owners-rank``).

    The frame is raw scores because 6a recorded one threshold pair for both rankings
    (item and decision thresholds within a top-cluster margin of each other). Should a
    re-calibration record a second pair for the rejection ranking, a match leads only
    when it clears that ranking's own ``strong`` and the decision ranking is not
    ``strong``, and this function gains that branch (ADR
    ``rejection-match-leads-the-note-only-in-a-frame-the-calibration-layer-validates``).

    Args:
        matches: The section's shaped matches (``display.rejection_matches_payload``).
        top_score: The top live hit's score, or None with no live hit.

    Returns:
        The leading match, or None.
    """
    for match in matches:
        if match["letter"] == "retired":
            continue
        if top_score is None or match["score"] > top_score:
            return match
    return None


def rejection_lead_clause(match: Mapping[str, Any], *, surface: str) -> str:
    """Words the line that opens a ranked note when a live owner's rejection leads.

    It binds the rejection to its owner's question: the option was rejected in that
    decision's own context, so the line asks the reader to weigh whether that reason
    carries over, and never reads as a ban on the option or as an answer that
    overrides the live hits. A read instruction only, so it is lawful in ``query``'s
    register. It names the envelope key where the owner's axiom rides, so one sentence
    is true on JSON and text alike. The wording is the same on both surfaces; ``surface``
    is validated so a composer passes it uniformly (:func:`retired_lead_clause`'s
    idiom). Pure string formatting.

    Args:
        match: The leading match (:func:`rejection_lead`).
        surface: ``"cli"`` or ``"mcp"``.

    Returns:
        One sentence.
    """
    if surface not in _SURFACE_POINTERS:
        raise KeyError(surface)
    if match["letter"] == "window":
        where = f"at rank {match['rank']} of this window"
    else:
        where = "on its match in rejection_matches"
    return (
        f"A rejected option ranks above every live decision here: {match['slug']} "
        f"rejected its item {match['ordinal']} in its own context, settling its own "
        f"question. Its axiom is {where}; weigh whether that reason carries over to "
        f"this claim."
    )


def absence_clause(*, register: str, surface: str, config: "object",
                   scope: Optional[str]) -> str:
    """Words how an absence is concluded, on every band that does not settle the question.

    A ranking cannot show that nothing covers a claim; the exhaustive read can. On
    ``surface`` and ``query`` it names the enumeration and its one-line tier; on the
    lookup it names the whole-depth read alone, because the one-line tier and
    ``brief`` carry no ``rejected_paths``. One wording for ``surface`` and ``query``,
    with no write instruction, so the ``query`` register lock polices it on both. The
    CLI forms carry the selector; ``scope`` fills the scope slot when the caller gave
    one. Pure string formatting.

    Args:
        register: ``"surface"``, ``"query"`` or ``"lookup"``.
        surface: ``"cli"`` or ``"mcp"`` — selects the read's spelling.
        config: The active ``MitosConfig`` (duck-typed). Read only for ``project``.
        scope: The caller's scope, or None (always None off ``surface``).

    Returns:
        One sentence.

    Raises:
        ValueError: On an unknown register.
    """
    pointers = _SURFACE_POINTERS[surface]
    selector = shell_flag("-p", getattr(config, "project", ""))
    if scope:
        slot = shell_arg(scope) if surface == "cli" else scope
    else:
        slot = "<scope>"

    def read(key: str) -> str:
        return pointers[key].format(selector=selector, scope=slot)

    if register in ("surface", "query"):
        return (f"A ranking cannot show that nothing covers this claim; the exhaustive "
                f"read can: {read('exhaustive')}, or {read('exhaustive_oneline')} for "
                f"one line per decision.")
    if register == "lookup":
        return (f"A ranking cannot show that no rejected option covers this claim; the "
                f"exhaustive read can: {read('exhaustive_whole')}, which keeps each "
                f"decision's rejected_paths.")
    raise ValueError(f"absence_clause: unknown register {register!r}")


def oq_scan_fault_clause(scope: str) -> str:
    """Words a faulted scan of the scope's parked open questions (WIRING_LEDGER entry-005).

    Said instead of an empty list: ``open_questions: []`` reads "none parked here",
    which a scan that raised cannot claim, so the key is absent and this names why.
    Pure string formatting.
    """
    return f"The open questions in scope '{scope}' could not be read."


def scope_counts_fault_clause(scope: str) -> str:
    """Words a faulted read of the scope vocabulary (WIRING_LEDGER entry-005).

    Without the counts the unused-scope check cannot run, so a typo'd scope reads like
    a live one; the clause says the check was not made. Pure string formatting.
    """
    return (f"The scope vocabulary could not be read, so whether '{scope}' is a live "
            f"tag was not checked.")


def count_withheld(hits: List[Dict[str, Any]]) -> int:
    """Counts the hits in one answer that were returned without ``rejected_paths``.

    Derived from the returned list, never counted in the loops that thinned it, so
    the count cannot drift from the thinning rule: ``letter_payload`` and the lexical
    fallback omit exactly this key on a thinned hit, and ``rejected_paths`` is a
    required field (M5), so on a decision hit its absence always means withheld. An
    open question skipped by a ranked loop is not in the list and counts as nothing.

    Args:
        hits: The decision payloads the answer returns.

    Returns:
        How many of them lack ``rejected_paths``; 0 on a default (whole) answer.
    """
    return sum(1 for hit in hits if "rejected_paths" not in hit)


def count_withheld_decisions(*, thinned_ids: List[str],
                             shown_ordinals: Mapping[str, Any],
                             item_counts: Mapping[str, int]) -> int:
    """Counts the decisions a ranked answer withheld reasoning from.

    The ranked reads' withheld rule, over **decisions** (ADR
    ``withheld-count-names-only-reasoning-the-answer-left-out``). The population is
    every thinned window hit plus every ``letter: "here"`` rejection owner whose
    ``rejected_paths`` the depth withheld, each once. A decision is exempt when the
    rejection section's matches already show every item its field can serve
    (``len(shown_ordinals[id]) == item_counts[id]``) — a one-item field's match is
    the common case — because then nothing of its reasoning was left out.
    :func:`count_withheld` keeps serving the paths that carry no section
    (``list``); on an answer with no matches the two give the same number.

    Args:
        thinned_ids: The population's node ids, repeats allowed.
        shown_ordinals: Node id → the ordinals its rejection matches show.
        item_counts: Node id → how many distinct items its field can serve
            (``RejectionWalk.item_counts``).

    Returns:
        How many distinct decisions in the population are not covered.
    """
    counted = set()
    for node_id in thinned_ids:
        if node_id in counted:
            continue
        shown = shown_ordinals.get(node_id)
        if shown and len(shown) == item_counts.get(node_id):
            continue
        counted.add(node_id)
    return len(counted)


def rejection_owners_whole(full_top: Optional[int], window_len: int) -> bool:
    """Says whether a ``here`` rejection owner keeps its ``rejected_paths``.

    The depth the caller set for the window's **last** rank (ADR
    ``rejection-section-owners-follow-the-depth-of-the-windows-last-rank``): an owner
    outside the window ranks below all of it, so it gets that rank's depth. With an
    empty window (a blackout or a miss) only ``brief`` (``full_top == 0``) withholds.

    Args:
        full_top: The effective depth (None ⇒ every rank whole, 0 ⇒ brief).
        window_len: How many decisions the window holds.

    Returns:
        True when the owner's Letter is whole.
    """
    return not (full_top is not None and (full_top == 0 or full_top < window_len))


# The rejection leg's states, as `retrieval_walk.RejectionWalk.state` spells them.
# Agreement-pinned to `retrieval_walk.REJECTIONS_*` by a row; spelled here because
# this leaf imports no `mitos` module but `shell_quote`.
_LEG_ABSENT = "absent"
_LEG_OVER_BUDGET = "over_budget"
_LEG_ERROR = "error"
_LEG_SEARCHED = "searched"


def rejection_section_state(rejections: Optional["object"], *,
                            gap: Callable[[], bool]) -> Optional[str]:
    """Maps the walk's rejection result to the envelope's state token.

    Called only when the ranked walk ran (``walk is not None``); the healthy-empty
    ``CollectionMissingError`` arm composes no section. The token is ``None`` when
    the section was computed whole, which leaves ``rejection_section`` absent
    (constraint 5: absent means computed).

    Args:
        rejections: ``RankedWalk.rejections`` (duck-typed), or None when the read
            seam built no leg while ranking ran.
        gap: Asked only on ``absent``: ``missing_rejection_index_is_a_gap(store)``
            bound by the caller, so a healthy read makes no graph read for it.

    Returns:
        ``"gap"`` (absent over a non-empty expected set), ``"not_searched"`` (over
        budget), ``"unavailable"`` (the index errored, or no leg ran), ``"short"``
        (verification left the section short), or None.

    Raises:
        ValueError: On a state outside the walk's vocabulary.
    """
    if rejections is None:
        return "unavailable"
    state = rejections.state
    if state == _LEG_ABSENT:
        # An empty expected set: the absent collection IS the empty index.
        return "gap" if gap() else None
    if state == _LEG_OVER_BUDGET:
        return "not_searched"
    if state == _LEG_ERROR:
        return "unavailable"
    if state == _LEG_SEARCHED:
        return "short" if rejections.short else None
    raise ValueError(f"unknown rejection leg state {state!r}")


def rejection_section_clause(state: str, *, dropped: Optional[int], surface: str,
                             config: "object") -> str:
    """Words the note clause for a rejection section that was not computed whole.

    Each says what was not computed and why, and that the decision answer stands.
    The gap names its heal through :data:`REJECTION_GAP_POINTERS` (the MCP form names
    no command and no tool). Pure string formatting; it never raises for a token
    :func:`rejection_section_state` returns.

    Args:
        state: The state token (``gap``, ``not_searched``, ``unavailable``,
            ``short``).
        dropped: How many fetched hits verification dropped (``short`` only).
        surface: ``"cli"`` or ``"mcp"`` — selects the gap's heal wording.
        config: The active ``MitosConfig`` (duck-typed). Read only for
            ``project``, which the CLI recipe names as its selector.

    Returns:
        One sentence.
    """
    if state == "gap":
        heal = REJECTION_GAP_POINTERS[surface]["build"].format(
            selector=shell_flag("-p", getattr(config, "project", ""))
        )
        return (
            "The rejection index is missing while decisions here state rejected "
            "options, so no rejected option was searched for this answer; "
            f"{heal}."
        )
    if state == "not_searched":
        return (
            "The rejection search ran past its time budget, so rejected options were "
            "not searched for this answer; the decision answer is whole."
        )
    if state == "short":
        n = dropped or 0
        if n == 0:
            # Short with nothing dropped: the full fetch's hits collapsed into fewer
            # items than the bound (parts of one item), so verification is not why.
            return (
                "The rejection search's fetch filled before the rejection section "
                "did, so the section may hold fewer than it could."
            )
        noun = "match" if n == 1 else "matches"
        return (
            f"Verification dropped {n} rejection {noun} whose decision no longer "
            "states the option or is gone, so the rejection section may hold fewer "
            "than it could."
        )
    return (
        "The rejection index could not be read, so rejected options were not "
        "searched for this answer; the decision answer is whole."
    )


class RejectionSection(NamedTuple):
    """One ranked answer's rejection section, composed once for every surface.

    Attributes:
        matches: The shaped matches (``display.rejection_matches_payload``), ``[]``
            when none.
        state: The state token, or None when the section was computed whole.
        dropped: How many fetched hits verification dropped, beside ``short`` only.
        withheld: :func:`count_withheld_decisions` over the window and the section.
        clause: The state's note clause, or None.
    """

    matches: List[Dict[str, Any]]
    state: Optional[str]
    dropped: Optional[int]
    withheld: int
    clause: Optional[str]

    def envelope_keys(self) -> Dict[str, Any]:
        """The section's envelope keys, in order; empty when computed and empty.

        ``rejection_matches`` when at least one match, ``rejection_section`` when
        not computed whole, ``rejection_dropped`` only beside ``"short"``. Never a
        ``[]`` and never a ``null``: absent means computed and empty.
        """
        keys: Dict[str, Any] = {}
        if self.matches:
            keys["rejection_matches"] = self.matches
        if self.state is not None:
            keys["rejection_section"] = self.state
        if self.state == "short":
            keys["rejection_dropped"] = self.dropped
        return keys


def rejection_section(walk: "object", *, full_top: Optional[int],
                      gap: Callable[[], bool], shape: Callable[..., List[Dict[str, Any]]],
                      surface: str, config: "object") -> RejectionSection:
    """Composes a ranked answer's rejection section: matches, state, clause, withheld.

    The one composition the four ranked surfaces call, after the walk and outside
    their ranked ``try``, only when the walk ran. The shaper is injected
    (``display.rejection_matches_payload``) because this leaf imports no display.

    Args:
        walk: The ``RankedWalk`` (duck-typed): ``window``, ``rejections``,
            ``stamps``, ``live_ends``.
        full_top: The effective depth (None ⇒ whole, 0 ⇒ brief); the window's
            thinning rule is the surfaces' (rank > ``full_top``).
        gap: See :func:`rejection_section_state`.
        shape: ``display.rejection_matches_payload``.
        surface: ``"cli"`` or ``"mcp"``.
        config: The active ``MitosConfig`` (duck-typed).

    Returns:
        A :class:`RejectionSection`.
    """
    window_ids = [hit.node["id"] for hit in walk.window]
    owners_whole = rejection_owners_whole(full_top, len(window_ids))
    rejections = walk.rejections
    matches: List[Dict[str, Any]] = []
    if rejections is not None and rejections.matches:
        matches = shape(rejections, window_ids=window_ids, owners_whole=owners_whole,
                        stamps=walk.stamps, live_ends=walk.live_ends)
    state = rejection_section_state(rejections, gap=gap)
    dropped = sum(rejections.dropped.values()) if state == "short" else None
    shown: Dict[str, set] = {}
    for match in matches:
        shown.setdefault(match["id"], set()).add(match["ordinal"])
    thinned = [node_id for rank, node_id in enumerate(window_ids, start=1)
               if full_top is not None and rank > full_top]
    if not owners_whole:
        thinned += [match["id"] for match in matches if match["letter"] == "here"]
    withheld = count_withheld_decisions(
        thinned_ids=thinned, shown_ordinals=shown,
        item_counts=rejections.item_counts if rejections is not None else {},
    )
    clause = (rejection_section_clause(state, dropped=dropped, surface=surface,
                                       config=config)
              if state is not None else None)
    return RejectionSection(matches=matches, state=state, dropped=dropped,
                            withheld=withheld, clause=clause)


# The note's line order, the one statement of it (D1 of 6b): every exit that composes
# a note joins its present lines in this order through :func:`compose_note`, and no call
# site concatenates note text. ``edge`` holds the edge clause or the short-window clause
# (they never co-occur); ``scan_faults`` holds entry-005's clauses.
NOTE_SLOTS: Tuple[str, ...] = (
    "rejection_lead", "retired_lead", "legend", "top_cluster", "edge", "absence",
    "not_computed", "section_state", "withheld", "stamps_unread", "scan_faults",
)


def compose_note(*, rejection_lead: Optional[str] = None,
                 retired_lead: Optional[str] = None, legend: str,
                 top_cluster: Optional[str] = None, edge: Optional[str] = None,
                 absence: Optional[str] = None, not_computed: Optional[str] = None,
                 section_state: Optional[str] = None, withheld: Optional[str] = None,
                 stamps_unread: Optional[str] = None,
                 scan_faults: Sequence[str] = ()) -> str:
    """Joins a note's lines in :data:`NOTE_SLOTS` order. The ONE place order is decided.

    Each line arrives already worded by the function that owns it; this only places
    the present ones, joined by single spaces. The ranked surfaces reach it through
    :func:`compose_ranked`, the lookup through :func:`rejection_lookup`, and the three
    degraded composers in ``degraded`` directly (passing only ``legend``,
    ``not_computed``, ``section_state``, ``withheld``, ``stamps_unread`` and
    ``scan_faults``).

    Args:
        rejection_lead: :func:`rejection_lead_clause`, or None.
        retired_lead: :func:`retired_lead_clause`, or None.
        legend: The band's sentence (or the blackout, or a degraded header) — always.
        top_cluster: :func:`top_cluster_clause`, or None.
        edge: The edge or short-window clause, or None.
        absence: :func:`absence_clause`, or None.
        not_computed: A degraded exit's omission clause, or None.
        section_state: The rejection section's clause, or None.
        withheld: :func:`withheld_clause`, or None.
        stamps_unread: :func:`stamps_unread_clause`, or None.
        scan_faults: Entry-005's clauses, in order; empty when none.

    Returns:
        The note.
    """
    slots = {"rejection_lead": rejection_lead, "retired_lead": retired_lead,
             "legend": legend, "top_cluster": top_cluster, "edge": edge,
             "absence": absence, "not_computed": not_computed,
             "section_state": section_state, "withheld": withheld,
             "stamps_unread": stamps_unread}
    lines: List[str] = []
    for name in NOTE_SLOTS:
        if name == "scan_faults":
            lines.extend(line for line in scan_faults if line)
        elif slots[name]:
            lines.append(slots[name])
    return " ".join(lines)


# The overrides a composed note can carry (``RankedNote.override``, ``LookupNote``): the
# blackout replaces the legend; the unbuilt-graph note replaces the whole note and stands
# for every key the graph supplies (lock A).
OVERRIDE_BLACKOUT = "blackout"
OVERRIDE_UNBUILT = "unbuilt"


class RankedNote(NamedTuple):
    """One ranked answer's composed band, note and graph-supplied keys.

    ``band_evidence`` is present iff the band read a top score, which means the window
    is non-empty: absent on a miss, a blackout, the healthy-empty arms and the unbuilt
    graph, and never composed on a degraded or exact-slug exit (those name it in
    ``not_computed`` or carry no band at all).

    Attributes:
        confidence: The band, on every exit (an override about the graph does not
            contradict a fact about the ranking).
        note: The composed note.
        evidence: :func:`band_evidence`, or None (absent).
        override: :data:`OVERRIDE_BLACKOUT`, :data:`OVERRIDE_UNBUILT` or None — the
            CLI text gate for the band line and the printed sections.
        handles: The retired partition's shown handles.
        omitted: How many members the partition bound left out.
        section: The rejection section, or None when the walk did not run.
        withheld: The withheld count over decisions.
        stamps_unread: Whether the walk's stamp read faulted.
    """

    confidence: str
    note: str
    evidence: Optional[Dict[str, Any]]
    override: Optional[str]
    handles: List[Dict[str, Any]]
    omitted: int
    section: Optional[RejectionSection]
    withheld: int
    stamps_unread: bool

    def envelope_keys(self) -> Dict[str, Any]:
        """The keys every ranked envelope places after its hit list, in order.

        ``confidence``, ``band_evidence``, ``note``, ``retired_partition`` (+
        ``retired_partition_omitted``), the section's keys, ``all_superseded`` (the
        blackout), ``rejected_paths_withheld``, ``stamps_unread``. Under the unbuilt
        graph only ``confidence`` and ``note``: with no graph to read, its note stands
        for every key the graph supplies (lock A), and a later graph-supplied key joins
        the suppressed list by being placed below that return.
        """
        keys: Dict[str, Any] = {"confidence": self.confidence}
        if self.override == OVERRIDE_UNBUILT:
            keys["note"] = self.note
            return keys
        if self.evidence is not None:
            keys["band_evidence"] = self.evidence
        keys["note"] = self.note
        if self.handles:
            keys["retired_partition"] = self.handles
            if self.omitted:
                keys["retired_partition_omitted"] = self.omitted
        if self.section is not None:
            keys.update(self.section.envelope_keys())
        if self.override == OVERRIDE_BLACKOUT:
            keys["all_superseded"] = self.handles
        if self.withheld:
            keys["rejected_paths_withheld"] = self.withheld
        if self.stamps_unread:
            keys["stamps_unread"] = True
        return keys


_REGISTERS: Tuple[str, ...] = ("surface", "query")


def compose_ranked(walk: Optional["object"], *, register: str, surface: str,
                   config: "object", handles: List[Dict[str, Any]],
                   part: RetiredPartition, section: Optional[RejectionSection],
                   unbuilt: Callable[[], bool], top_cluster_margin: float,
                   edge_margin: float, ceiling: int, full_top: Optional[int],
                   scope: Optional[str] = None,
                   scope_counts: Optional[Dict[str, Dict[str, int]]] = None,
                   scan_faults: Sequence[str]) -> RankedNote:
    """Composes a ranked answer's band, evidence and note: the four ranked surfaces' one call.

    Computed once, from the walk: the window's scores and top score, the band and its
    legend (:func:`assess_surface_recall` / :func:`assess_query_recall`), the edge or
    short window, the top cluster, the rejection lead, the absence clause and the
    evidence; then :func:`compose_note` orders them. The overrides keep their recorded
    precedence (ADR ``override-precedence-is-quantified-over-the-response-not-the-note-key``):

    * **unbuilt** (an empty window over a graph holding no nodes while the corpus holds
      entries): the note is :func:`missing_graph_note` alone and no graph-supplied key
      rides — no lead, partition, section, evidence, withheld or stamps (lock A). It
      beats the blackout, which cannot co-occur (a retired handle is a node), coded
      rather than left to that;
    * **blackout** (ranking ran, zero live, handles passed): the legend is
      :func:`blackout_note`, the short clause follows it, the retired lead is empty (the
      blackout is its zero-live form) and so is the absence clause; a live owner's
      rejection match still leads it.

    The absence clause rides ``weak`` and ``none`` and is left out whenever the retired
    lead or the blackout is in the note: that note has named where the current word
    lives. The handles are shaped by the site (``display.retired_handle``) because this
    leaf imports no display; ``unbuilt`` is asked only on an empty window, so the graph
    read stays off the healthy path.

    Args:
        walk: The ``RankedWalk`` (duck-typed), or None on a healthy-empty arm (an
            absent collection over an empty expected set).
        register: ``"surface"`` or ``"query"``.
        surface: ``"cli"`` or ``"mcp"``.
        config: The active ``MitosConfig`` (duck-typed).
        handles: The partition's shown handles, shaped by the site.
        part: :func:`retired_partition` of the walk (empty with no walk).
        section: :func:`rejection_section` of the walk, or None with no walk.
        unbuilt: ``missing_graph_is_a_gap(store, config, corpus_scan=…)`` bound by the
            caller.
        top_cluster_margin: ``RANKED_TOP_CLUSTER_MARGIN``.
        edge_margin: ``RANKED_EDGE_MARGIN``.
        ceiling: The clamp ceiling (``display.RANKED_LIMIT_CEILING``).
        full_top: The effective ``full_top`` (0 under ``brief``), or None.
        scope: The caller's scope (``surface`` only).
        scope_counts: The ordered live scope counts (``surface`` only), or None.
        scan_faults: Entry-005's clauses (``oq_scan_fault_clause``,
            ``scope_counts_fault_clause``), in that order; empty when none.

    Returns:
        A :class:`RankedNote`.

    Raises:
        ValueError: On an unknown register, on ``scope``/``scope_counts`` passed with
            ``register="query"``, or on a walk end outside the four (from
            :func:`band_evidence`).
    """
    if register not in _REGISTERS:
        raise ValueError(f"compose_ranked: unknown register {register!r}")
    if register == "query" and (scope is not None or scope_counts is not None):
        raise ValueError("compose_ranked: the query register takes no scope")
    window = list(walk.window) if walk is not None else []
    scores = [hit.score for hit in window]
    top_score = max(scores) if scores else None
    if register == "surface":
        confidence, legend = assess_surface_recall(
            semantic_ran=True, top_score=top_score, result_count=len(window),
            scope=scope, scope_counts=scope_counts, surface=surface,
            degraded_reason=None,
        )
    else:
        confidence, legend = assess_query_recall(
            top_score=top_score, result_count=len(window), config=config,
            surface=surface,
        )
    if not window and unbuilt():
        return RankedNote(confidence=confidence, note=missing_graph_note(surface, config),
                          evidence=None, override=OVERRIDE_UNBUILT, handles=[],
                          omitted=0, section=None, withheld=0, stamps_unread=False)
    evidence = (band_evidence(walk, scores=scores, top_cluster_margin=top_cluster_margin,
                              edge_margin=edge_margin) if walk is not None else None)
    blackout = not window and bool(handles)
    lead = rejection_lead(section.matches if section is not None else [],
                          top_score=top_score)
    edge = (window_edge(walk, margin=edge_margin, ceiling=ceiling, full_top=full_top)
            if window else None)
    short = short_window(walk) if (window or blackout) else None
    if edge is not None:
        edge_line: Optional[str] = edge_clause(edge, surface=surface)
    elif short is not None:
        edge_line = short_window_clause(short, surface=surface)
    else:
        edge_line = None
    retired_line = (retired_lead_clause(handles[0], surface=surface)
                    if part.leads and handles else None)
    if blackout:
        legend = blackout_note(handles, surface=surface, config=config)
    absence = None
    if confidence != "strong" and retired_line is None and not blackout:
        # A scope the vocabulary says is unused is not handed to the exhaustive read:
        # the unused-scope vector already names the live tags, so the slot stays open.
        absence = absence_clause(
            register=register, surface=surface, config=config,
            scope=None if _scope_is_unused(scope, scope_counts) else scope)
    withheld = section.withheld if section is not None else 0
    stamps_unread = walk is not None and walk.stamps_unread
    note = compose_note(
        rejection_lead=(rejection_lead_clause(lead, surface=surface)
                        if lead is not None else None),
        retired_lead=retired_line,
        legend=legend,
        top_cluster=(top_cluster_clause(evidence["top_cluster"], register=register,
                                        margin=top_cluster_margin)
                     if evidence is not None else None),
        edge=edge_line,
        absence=absence,
        section_state=section.clause if section is not None else None,
        withheld=(withheld_clause(withheld, surface=surface, config=config)
                  if withheld else None),
        stamps_unread=(stamps_unread_clause(surface=surface, config=config)
                       if stamps_unread else None),
        scan_faults=scan_faults,
    )
    return RankedNote(confidence=confidence, note=note, evidence=evidence,
                      override=OVERRIDE_BLACKOUT if blackout else None,
                      handles=handles, omitted=part.omitted, section=section,
                      withheld=withheld, stamps_unread=stamps_unread)


# The field-scoped lookup (``query``'s ``field`` selector): the stored fields a caller
# may search alone. One today. A later index over another field adds a value here and
# renames nothing — the argument's name is a contract (AX Hardening 3, lock 3). Both
# boundaries refuse any other value through :func:`lookup_field_fault`, and the CLI's
# ``--field`` choices are this tuple.
REJECTED_PATHS_FIELD = "rejected_paths"
LOOKUP_FIELDS: Tuple[str, ...] = (REJECTED_PATHS_FIELD,)


def lookup_field_fault(value: Optional[str]) -> Optional[str]:
    """Words the refusal of a ``field`` value outside :data:`LOOKUP_FIELDS`, or None.

    An argument fault, answered before any project is resolved on both surfaces, as
    ``depth``'s is. Names no command and no call-form: it says what is accepted.

    Args:
        value: The caller's ``field``; None (the selector unset) is never a fault.

    Returns:
        The refusal message, or None when the value is accepted.
    """
    if value is None or value in LOOKUP_FIELDS:
        return None
    accepted = ", ".join(f"'{name}'" for name in LOOKUP_FIELDS)
    return (f"Field '{value}' cannot be searched alone; the accepted value is "
            f"{accepted}, which ranks rejected options alone.")


# The lookup's degraded reason when its search ran past its time budget. A reason
# phrase for the lexical header, on ``lexical.degraded_reason_from_error``'s register.
REJECTION_LOOKUP_OVER_BUDGET_REASON = "the rejection search ran past its time budget"


def rejection_lookup_gap_reason(*, surface: str, config: "object") -> str:
    """Words the lookup's degraded reason when the rejection index is a gap.

    Its own phrase, not :func:`rejection_section_clause`'s ``gap`` sentence: that one
    says no rejected option was searched, which beside a lexical match over rejected
    options would contradict the header it sits in. The heal is
    :data:`REJECTION_GAP_POINTERS`' (the MCP form names no command and no tool), and
    the note does not state it a second time.

    Args:
        surface: ``"cli"`` or ``"mcp"`` — selects the heal's wording.
        config: The active ``MitosConfig`` (duck-typed). Read only for ``project``.

    Returns:
        A short cause phrase for the degraded header.
    """
    heal = REJECTION_GAP_POINTERS[surface]["build"].format(
        selector=shell_flag("-p", getattr(config, "project", ""))
    )
    return ("the rejection index is missing while decisions here state rejected "
            f"options — {heal}")


# What the lookup's lexical answer could not compute: the band, its evidence and the
# edge, which need scores. Not :data:`DEGRADED_NOT_COMPUTED`: its ``retired_partition``
# token would be false here, because the lookup has no retired partition to compute.
LOOKUP_DEGRADED_NOT_COMPUTED: Tuple[str, ...] = ("band", "band_evidence", "edge")


def lookup_omission_clause(*, retired_dropped: bool) -> str:
    """Words what the lookup's lexical answer did not compute or list.

    The note twin of ``not_computed`` (:data:`LOOKUP_DEGRADED_NOT_COMPUTED`). A
    retired decision's rejected options are dropped only when the graph's states
    were read; otherwise the stamps-unavailable disclosure says the answer may
    include superseded decisions, and this clause says nothing about them.

    Args:
        retired_dropped: Whether the lexical match read the graph's states.

    Returns:
        One or two sentences.
    """
    clause = ("Not computed without similarity scores: the confidence band and its "
              "evidence, and the ranking's edge.")
    if retired_dropped:
        clause = f"{clause} Rejected options of retired decisions are not listed."
    return clause


def _lookup_band_note(band: str, *, top_score: Optional[float], result_count: int,
                      surface: str, config: "object") -> str:
    """Words the lookup's legend: a rating of its item ranking, never a verdict.

    Its own legend, not :func:`assess_query_recall`'s: on item scores a ``strong``
    top often belongs to an option that shares the claim's words and names a
    different mechanism (5b measured vocabulary near-misses up to 0.815 against an
    own-question median of 0.750), so "a real hit" would be false here, and 6a found
    no item lift for ``strong`` either. The legends say how close the ranking came and
    leave the judgement to the reader of the item; ``weak`` says :data:`WEAK_MEANING`.
    """
    redirect = _precedent_redirect(surface=surface, config=config)
    shown = f"{top_score:.2f}" if top_score is not None else "?"
    if band == "strong":
        return ("Top rejected option cleared the strong threshold — close to the "
                "claim; read the item and its decision's reason to judge whether "
                "that rejection covers your option.")
    if band == "weak":
        return (f"Twilight zone: top score {shown} — a rejected option close to the "
                f"claim, maybe not the one you mean; {WEAK_MEANING}. {redirect}")
    if result_count:
        return (f"Off-axis: top score {shown} sits in the lowest band — no rejected "
                f"option ranked close to the claim. {redirect}")
    return f"No rejected option ranked — this ranking returned nothing. {redirect}"


def lookup_edge(walk: "object", *, owners: int, margin: float, ceiling: int,
                full_top: Optional[int]) -> Optional[WindowEdge]:
    """Decides the lookup's edge, its lever counted in owners, never in ranks.

    :func:`window_edge`'s rule, unchanged, over the item ranking; then the held count
    is re-counted, because the lookup's ``full_top`` counts **live owners** (the
    ``here`` Letters), not matched items: ``owners`` at the default, ``min(full_top,
    owners)`` under ``full_top``, and None with no live owner in the window (every
    item a retired handle), under ``brief`` too, since a ``full_top`` lever means
    nothing with no Letter to thin. At the ceiling ``window_edge``'s None stands.

    Args:
        walk: The ``RejectionLookupWalk`` (duck-typed: ``window``, ``edge``,
            ``end``, ``limit``).
        owners: How many ``here`` owners the shaped window letters.
        margin: The edge margin of the ranking (``RANKED_EDGE_MARGIN`` today).
        ceiling: The clamp ceiling.
        full_top: The effective ``full_top`` (0 under ``brief``), or None.

    Returns:
        A :class:`WindowEdge`, or None when the clause stays absent.
    """
    edge = window_edge(walk, margin=margin, ceiling=ceiling, full_top=full_top)
    if edge is None or edge.held is None:
        return edge
    if owners == 0:
        return edge._replace(held=None)
    return edge._replace(held=owners if full_top is None else min(full_top, owners))


def lookup_edge_clause(edge: WindowEdge, *, ceiling: int, surface: str) -> str:
    """Words the lookup's edge sentence, in items and owners.

    :func:`edge_clause`'s shape over the rejection ranking, with its own nouns: the
    ranks are rejected options and ``full_top`` thins the decisions they belong to,
    so "live rank" and "the added ranks come back axiom-only" would be false here (an
    added item always carries its whole text). With no Letter in the window the lever
    names ``limit`` alone; at the ceiling it names none. Pure string formatting.

    Args:
        edge: The edge from :func:`lookup_edge`.
        ceiling: The clamp ceiling a higher ``limit`` would be cut back to.
        surface: ``"cli"`` or ``"mcp"`` — selects the argument spellings.

    Returns:
        One sentence.
    """
    pointers = _SURFACE_POINTERS[surface]
    at = pointers["limit_arg"].format(n=edge.limit)
    if edge.kind == EDGE_KIND_UNMEASURED:
        return (
            f"This ranking's edge at {at} is unmeasured: the read fetched no further "
            f"rejected option, so it cannot say whether the window ended mid-cluster."
        )
    stem = (f"This ranking ended mid-cluster at {at}: the next rejected option scores "
            f"as close as the last")
    if edge.held is None:
        if edge.limit >= ceiling:
            return f"{stem}."
        return f"{stem}; a higher limit reaches it."
    if edge.held == 0:
        return (f"{stem}; a higher limit reaches it, its decision axiom-only like "
                f"these.")
    held = pointers["full_top_arg"].format(n=edge.held)
    return (f"{stem}; a higher limit reaches it, and with {held} the decisions it "
            f"adds come back axiom-only.")


def lookup_short_clause(walk: "object", *, surface: str) -> Optional[str]:
    """Words why the lookup's window holds fewer than ``limit`` items, or None.

    Two reasons a window comes back short, each said (constraint 5): the index ran
    out (``exhausted`` with at least one hit), or the fetch filled before the ranking
    did (``short_bound``) — verification dropped items whose decision no longer
    states them or is gone, or, with nothing dropped, the fetch's hits were parts of
    fewer items. No lever: a higher ``limit`` re-reads the same fetch. A full window
    says nothing here; its edge is :func:`lookup_edge_clause`'s. Pure string
    formatting.

    Args:
        walk: The ``RejectionLookupWalk`` (duck-typed: ``window``, ``limit``,
            ``end``, ``returned``, ``dropped``).
        surface: ``"cli"`` or ``"mcp"`` — selects the ``limit`` spelling.

    Returns:
        One sentence, or None.
    """
    if len(walk.window) >= walk.limit:
        return None
    if walk.end == _WALK_END_EXHAUSTED and walk.returned:
        at = _SURFACE_POINTERS[surface]["limit_arg"].format(n=walk.limit)
        return (f"The ranking reached the end of the rejection index before it filled "
                f"{at}.")
    if walk.end != _WALK_END_SHORT_BOUND:
        return None
    n = sum(walk.dropped.values())
    if n == 0:
        return ("The rejection search's fetch filled before the ranking did — its "
                "hits were parts of fewer rejected options — so the ranking may hold "
                "fewer than it could.")
    noun = "rejected option" if n == 1 else "rejected options"
    return (f"Verification dropped {n} {noun} whose decision no longer states them or "
            "is gone, so the ranking may hold fewer than it could.")


class LookupNote(NamedTuple):
    """The field-scoped lookup's composed answer.

    Attributes:
        envelope: The serialized envelope (its key set carries no override marker).
        override: :data:`OVERRIDE_UNBUILT` or None — the CLI text gate for the band
            line, kept off the dict so the envelope's keys do not change.
    """

    envelope: Dict[str, Any]
    override: Optional[str]


def rejection_lookup(walk: Optional["object"], *, query: str, full_top: Optional[int],
                     shape: Callable[..., List[Dict[str, Any]]], margin: float,
                     top_cluster_margin: float, ceiling: int,
                     graph_gap: Callable[[], bool], surface: str,
                     config: "object") -> LookupNote:
    """Composes the field-scoped lookup's healthy envelope, once for both surfaces.

    The rejection ranking is the answer: ``rejection_matches`` shaped by the
    section's own shaper (no decision window, so a match's ``letter`` is ``here``,
    ``section`` or ``retired``), matched items whole at every depth and owners thinned
    by ``full_top`` in first-appearance order. The band rates that ranking, and
    ``band_evidence`` carries its measured quantities over items. The note goes through
    :func:`compose_note`: legend → top cluster → edge or short → absence → withheld →
    stamps-unread. The lookup has no decision ranking, so it has no lead line. An empty
    window over an unbuilt graph answers with :func:`missing_graph_note` alone and none
    of the graph-supplied keys (W31, lock A). Each boundary encodes or renders the
    envelope and adds nothing.

    Args:
        walk: The ``RejectionLookupWalk`` (duck-typed), or None for the healthy empty
            answer (an absent collection over an empty expected set).
        query: The caller's query, echoed.
        full_top: The effective depth in owners (None ⇒ every owner whole, 0 ⇒
            ``brief``).
        shape: ``display.rejection_matches_payload``.
        margin: The edge margin (``RANKED_EDGE_MARGIN``).
        top_cluster_margin: The top-cluster margin (``RANKED_TOP_CLUSTER_MARGIN``; one
            pair serves both rankings, 6a). Its own keyword, never ``margin`` reused.
        ceiling: The clamp ceiling (``display.RANKED_LIMIT_CEILING``).
        graph_gap: Asked only on an empty window:
            ``missing_graph_is_a_gap(store, config, corpus_scan=…)`` bound by the
            caller.
        surface: ``"cli"`` or ``"mcp"``.
        config: The active ``MitosConfig`` (duck-typed).

    Returns:
        A :class:`LookupNote`. The envelope: ``query``, ``field``, ``depth_mode``,
        ``rejection_matches`` (absent when empty), provenance, ``confidence``,
        ``band_evidence`` (absent on an empty window), ``note``, then
        ``rejection_section: "short"`` with ``rejection_dropped``,
        ``rejected_paths_withheld`` and ``stamps_unread`` when they apply.
    """
    window = list(walk.window) if walk is not None else []
    matches: List[Dict[str, Any]] = []
    if window:
        matches = shape(walk, window_ids=[], owners_whole=True, stamps=walk.stamps,
                        live_ends=walk.live_ends, whole_owners=full_top)
    scores = [match.score for match in window]
    top_score = max(scores) if scores else None
    band = _classify_recall(top_score=top_score, result_count=len(window))
    envelope: Dict[str, Any] = {"query": query, "field": REJECTED_PATHS_FIELD,
                                "depth_mode": "letter"}
    if matches:
        envelope["rejection_matches"] = matches
    envelope.update(corpus_provenance(config))
    envelope["confidence"] = band
    if not window and graph_gap():
        envelope["note"] = missing_graph_note(surface, config)
        return LookupNote(envelope=envelope, override=OVERRIDE_UNBUILT)
    evidence = (band_evidence(walk, scores=scores, top_cluster_margin=top_cluster_margin,
                              edge_margin=margin) if walk is not None else None)
    here = [match["id"] for match in matches if match["letter"] == "here"]
    edge = (lookup_edge(walk, owners=len(here), margin=margin, ceiling=ceiling,
                        full_top=full_top) if walk is not None else None)
    if edge is not None:
        edge_line: Optional[str] = lookup_edge_clause(edge, ceiling=ceiling,
                                                      surface=surface)
    else:
        edge_line = (lookup_short_clause(walk, surface=surface) if walk is not None
                     else None)
    shown: Dict[str, set] = {}
    for match in matches:
        shown.setdefault(match["id"], set()).add(match["ordinal"])
    withheld = count_withheld_decisions(
        thinned_ids=here[full_top:] if full_top is not None else [],
        shown_ordinals=shown,
        item_counts=walk.item_counts if walk is not None else {},
    )
    stamps_unread = walk is not None and walk.stamps_unread
    note = compose_note(
        legend=_lookup_band_note(band, top_score=top_score, result_count=len(window),
                                 surface=surface, config=config),
        top_cluster=(top_cluster_clause(evidence["top_cluster"], register="lookup",
                                        margin=top_cluster_margin)
                     if evidence is not None else None),
        edge=edge_line,
        absence=(absence_clause(register="lookup", surface=surface, config=config,
                                scope=None) if band != "strong" else None),
        withheld=(withheld_clause(withheld, surface=surface, config=config)
                  if withheld else None),
        stamps_unread=(stamps_unread_clause(surface=surface, config=config)
                       if stamps_unread else None),
    )
    if evidence is not None:
        envelope["band_evidence"] = evidence
    envelope["note"] = note
    if walk is not None and walk.short:
        envelope["rejection_section"] = "short"
        envelope["rejection_dropped"] = sum(walk.dropped.values())
    if withheld:
        envelope["rejected_paths_withheld"] = withheld
    if stamps_unread:
        envelope["stamps_unread"] = True
    return LookupNote(envelope=envelope, override=None)


# What a degraded answer could not compute, as concept tokens: ``retired_partition``
# names the ranked envelopes' key, ``edge`` the window-edge clause, ``band`` the
# ``confidence`` band and ``band_evidence`` its evidence object. Each needs similarity
# scores, which a degraded read has none of. Carried on every degraded envelope as ``not_computed`` (serialized as a list)
# and never on a ranked one, so on a ranked envelope an absent ``retired_partition``
# still means computed and empty. The rejection section is not among them: a
# degraded read matches rejected options by their words (``rejection_section:
# "lexical"``); only the scope dump whose state read failed adds
# :data:`DEGRADED_SECTION_TOKEN` back. A tuple here only so no caller can mutate it.
DEGRADED_NOT_COMPUTED: Tuple[str, ...] = ("retired_partition", "edge", "band",
                                          "band_evidence")

# The ``not_computed`` token for a degraded rejection section that was not searched.
DEGRADED_SECTION_TOKEN = "rejection_matches"


def degraded_omission_clause() -> str:
    """Words the note clause every degraded answer carries: what it did not compute.

    The note twin of ``not_computed`` (:data:`DEGRADED_NOT_COMPUTED`), for the CLI
    text surface, where there is no key to read. A fact with no pointer and no
    write, so it is the same sentence on both surfaces. Pure string formatting.

    Returns:
        One sentence naming the omissions.
    """
    return (
        "Not computed without similarity scores: the retired partition, the "
        "window's edge, and the confidence band and its evidence."
    )


def degraded_section_clause(*, computed: bool, retired_dropped: bool) -> str:
    """Words the note clause of a degraded answer's rejection section.

    Follows :func:`degraded_omission_clause` on every degraded exit, as a ranked
    section's state clause follows its band note. True whether or not anything
    matched ("searched", never "found"). A fact with no pointer, so the same on both
    surfaces. Pure string formatting.

    Args:
        computed: Whether the section was searched (False only on the scope dump
            whose state read failed: its list is filtered, so an unfiltered section
            beside it would be false).
        retired_dropped: Whether the graph's states were read, so retired owners'
            items were dropped. Under the stamps-unavailable disclosure they were
            not, and the disclosure already says entries may include superseded
            ones.

    Returns:
        One sentence.
    """
    if not computed:
        return ("Rejected options were not searched: the graph's decision states "
                "could not be read.")
    if retired_dropped:
        return ("Rejected options were searched by their words, not their meaning; "
                "those of retired decisions are not listed.")
    return "Rejected options were searched by their words, not their meaning."


def withheld_clause(count: int, *, surface: str, config: "object") -> str:
    """Words the note clause a thinned ranked answer ends with.

    Verb-independent on purpose: ``surface`` and ``query`` gain the same characters
    on one surface, so the query note stays no longer than the surface note it
    displaces. It names the count, the field and the read that restores it — a read,
    never a write, and no verdict on the corpus, so it is lawful in both registers.
    Pure string formatting, so it cannot raise inside ``query_decisions``' ranked
    ``try`` and pose as degraded recall.

    Args:
        count: How many hits omit ``rejected_paths`` (≥ 1; the caller only asks then).
        surface: ``"cli"`` or ``"mcp"`` — selects the dereference wording.
        config: The active ``MitosConfig`` (duck-typed, as in
            :func:`assess_query_recall`). Read only for ``project``, which the CLI
            recipe names as its selector.

    Returns:
        One sentence, e.g. ``"2 hits omit rejected_paths; read any one whole with
        show_node(ident='<slug>')."``.
    """
    pointer = _SURFACE_POINTERS[surface]["dereference"].format(
        selector=shell_flag("-p", getattr(config, "project", ""))
    )
    if count == 1:
        return f"1 hit omits rejected_paths; read it whole with {pointer}."
    return f"{count} hits omit rejected_paths; read any one whole with {pointer}."


def stamps_unread_clause(*, surface: str, config: "object") -> str:
    """Words the note clause a ranked answer ends with when its stamps went unread.

    The walk read the ranking but not the modifier stamps, so any hit may have been
    amended, narrowed or retired since and its payload cannot say so. The clause
    states that and names the by-handle read, which carries the stamps. A read,
    never a write, and no verdict on the corpus, so it is lawful in both registers.
    Appended after every note override and after the withheld clause; absent on a
    healthy answer. Pure string formatting.

    Args:
        surface: ``"cli"`` or ``"mcp"`` — selects the dereference wording.
        config: The active ``MitosConfig`` (duck-typed). Read only for ``project``,
            which the CLI recipe names as its selector.

    Returns:
        One sentence.
    """
    pointer = _SURFACE_POINTERS[surface]["dereference"].format(
        selector=shell_flag("-p", getattr(config, "project", ""))
    )
    return (
        "Modifier stamps could not be read, so any hit may have been amended or "
        f"superseded; {pointer} reads one with its stamps."
    )


def corpus_provenance(config: "object") -> Dict[str, str]:
    """Builds the corpus-provenance fields every recall answer should carry.

    Names which corpus an answer came from (AX: an empty or twilight result is
    ambiguous between "no precedent exists" and "you're standing in the wrong
    workspace" — the reviewing cwd and a vision's decision store can diverge).
    Shared by the CLI verbs and their MCP twins so the field names can't drift.

    ``project`` answers that ambiguity in the *caller's own* vocabulary: a call
    that named a registered project gets that name back, so an agent holding
    several projects can see at a glance that the answer came from the one it
    asked for. The value is read straight off the config the resolution site
    built and is **never re-derived here** — a reverse lookup against the
    registry would break this module's leaf status, read the registry on every
    stamped answer (including the degraded path, where the filesystem may be the
    fault), and miss on a symlinked route, quietly echoing a path for a
    registered project.

    Args:
        config: The active ``MitosConfig`` (duck-typed to avoid an import cycle
            — recall is a leaf module).

    Returns:
        ``{"project": <registered name or absolute path>,
        "collection": <qdrant collection>, "workspace": <workspace dir>}``.
        Field order is contractual — the identity a reader scans for first, then
        the derived collection, then the location.
    """
    return {
        "project": getattr(config, "project", ""),
        "collection": getattr(config, "qdrant_collection", ""),
        "workspace": getattr(config, "workspace_dir", ""),
    }


def missing_index_is_a_gap(store: "object") -> bool:
    """Decides whether an absent vector collection is a gap worth reporting (I8).

    The one predicate behind every consumer of an absent collection: the four
    semantic read surfaces (``mitos query`` / ``mitos surface`` and their MCP
    twins) and the conflict sweep behind ``mitos check --staged`` and the
    sync-time sensor, so the shapes they take cannot become several behaviours.
    Over a graph whose decision index expects **nothing** an absent collection
    *is* the empty index — a just-initialized project is healthy, not broken —
    and the caller renders its ordinary nothing-found result with no diagnostic.
    Over a populated graph it is a real hole in recall, and the caller says so
    and names the heal.

    Gated on ``get_decision_index_expected_ids`` — every decision, live or
    retired, plus the surviving open questions — because that set is exactly
    what ``reconcile_embeddings`` enqueues, so the diagnostic fires if and only
    if the ``mitos reconcile`` it points at would actually have something to
    index. The gate and the heal agree by construction: one store read behind
    both (ADR ``decision-index-gap-and-creation-carve-out-key-on-its-expected-set``).
    A graph holding only retired decisions is therefore a gap, not the empty
    index.

    An unreadable graph answers **True**, the loud branch: "I could not check"
    must never render as the healthy-empty "I checked, there is nothing to index".

    Args:
        store: The graph store (duck-typed to avoid an import cycle — recall is a
            leaf module, as in :func:`corpus_provenance`).

    Returns:
        True when the absence should be surfaced as a degradation.
    """
    try:
        return bool(store.get_decision_index_expected_ids())
    except Exception:
        return True


def missing_rejection_index_is_a_gap(store: "object") -> bool:
    """Decides whether an absent rejection collection is a gap worth reporting.

    The rejection index's twin of :func:`missing_index_is_a_gap`. Over a graph
    whose decisions state no rejected item, an absent rejection collection *is*
    the empty index, and nothing is wrong. Over a graph where some decision's
    ``rejected_paths`` holds an item, it is a hole the caller names with its heal.

    Gated on ``rejection_index_expects_items`` — "the rejection index's own
    expected set is non-empty" — so it fires exactly when ``reconcile``'s
    collection pass would find something missing in an absent collection: the
    gate and the heal agree. The decision index's predicate is deliberately not
    borrowed: it counts open questions, and a questions-only workspace would be
    sent to a ``reconcile`` with nothing to write.

    An unreadable graph answers **True**, the loud branch, as its twin does.

    Args:
        store: The graph store (duck-typed; recall is a leaf module).

    Returns:
        True when the absence should be surfaced as a gap.
    """
    try:
        return bool(store.rejection_index_expects_items())
    except Exception:
        return True


def missing_graph_is_a_gap(
    store: Optional["object"],
    config: "object",
    *,
    corpus_scan: Callable[[object], bool],
) -> bool:
    """Decides whether an unbuilt graph is a gap worth reporting (I8, W31).

    The mirror of :func:`missing_index_is_a_gap`, and it is meant to be read
    beside it: that predicate asks *"is an absent collection a gap?"* and answers
    on the graph; this one asks *"is an empty graph a gap?"* and answers on the
    corpus. Both exist so the surfaces that consult them cannot become several
    behaviours.

    The state it names was made routine by the absolute-path escape hatch: a clone
    carries the committed ``.mitos/config.toml`` and a markdown corpus
    (``decisions.md`` and ``decisions/archive/``) holding hundreds of decisions,
    but not the gitignored ``*.sqlite`` — nobody commits a binary graph on purpose.
    Every semantic read over that workspace answers cleanly empty, and the agent
    reads *no precedents* for a project that has hundreds. That is the *"could not
    check"* → *"checked, it's clean"* inversion, arriving through the **graph** door
    instead of the collection one. The corpus is buffer **plus archives** because
    rotation drains the buffer: a scan of ``decisions.md`` alone answers the wrong
    file and fails in exactly that silent direction.

    **The gate is the TOTAL node count, never the active set**, and the reasoning
    is the sibling's own applied to a different heal: that predicate gates on
    ``get_decision_index_expected_ids`` because that set is exactly what ``mitos
    reconcile`` enqueues, while this one gates on the node count because ``mitos rebuild``
    commits every corpus *entry* to nodes regardless of computed state. Gate and heal agree by
    construction, and they are different heals — asking a **computed-state view**
    whether the graph was ever *built* is asking the wrong question, however the
    answer happens to come out.

    Worth knowing, because it is the thing that makes the wrong build hard to
    catch: today the two gates never actually disagree. Kill edges are acyclic by
    write-path guard, so a non-empty graph always retains at least one node with no
    incoming kill edge (measured — a supersession chain always leaves a live tip,
    and a self-supersede raises). "Every decision superseded, active set empty" is
    therefore **unconstructible** end to end, which means an active-set build ships
    green and stays green until something — a corrupted graph, a future kill-edge
    type, a rebuild — makes the two diverge, at which point it renders "the graph is
    missing, run ``mitos rebuild``" over a populated one. The difference lives in the
    contract, so that is where the tests pin it.

    ``store`` is ``Optional`` because ``status`` genuinely has none — it guards
    every graph read with ``os.path.exists(config.db_path)``, and a clone's
    ``db_path`` does not exist. The four read surfaces always hold a store (their
    ``GraphStore`` construction creates the file), so they pass one and it answers
    ``(0, "")``.

    Args:
        store: The graph store, or ``None`` when there is no graph file at all
            (duck-typed to avoid an import cycle — recall is a leaf module, as in
            :func:`corpus_provenance`).
        config: The active ``MitosConfig`` (duck-typed; handed to ``corpus_scan``
            unread).
        corpus_scan: The corpus scan, called with ``config`` —
            ``divergence.corpus_holds_entries`` at every call site, which reads
            buffer plus archives. **Required**, never defaulted: this module admits
            no ``mitos`` import beyond ``shell_quote`` (see the module docstring for
            why an import is not free), and a defaulted no-op would let a forgotten call site answer
            a silent ``False`` instead of raising.

    Returns:
        True when an unbuilt graph should be surfaced as a degradation — which is
        exactly when the corpus has something for ``mitos rebuild`` to commit. An
        unreadable graph answers on the corpus too, the loud branch: "I could not
        check" must never render as the healthy "I checked, it is empty".
    """
    try:
        node_count = store.graph_fingerprint()[0] if store is not None else 0
    except Exception:
        node_count = 0
    if node_count:
        return False
    return bool(corpus_scan(config))


def missing_graph_note(surface: str, config: "object") -> str:
    """Words the unbuilt-graph signal for one surface.

    Composed here rather than at each call site for the reason the whole module
    exists: four read surfaces spelling one diagnosis is four diagnoses waiting to
    diverge. The shared sentence says what the emptiness *means* — the distinction
    the reader is about to get wrong — and the per-surface clause names the heal in
    that surface's register (:data:`MISSING_GRAPH_POINTERS`).

    Args:
        surface: ``"cli"`` or ``"mcp"``. Required and unkeyed by design, as in
            :func:`assess_surface_recall`: no call site may silently emit the other
            surface's register.
        config: The active ``MitosConfig`` (duck-typed). Read only for
            ``project``, which the CLI recipe names as its selector.

    Returns:
        The one-line note.
    """
    recovery = MISSING_GRAPH_POINTERS[surface]["build"].format(
        selector=shell_flag("-p", getattr(config, "project", ""))
    )
    return (
        "The graph is unbuilt: the markdown corpus (`decisions.md` and "
        "`decisions/archive/`) holds entries but the graph has no nodes, so this "
        "empty answer means 'nothing is indexed yet', not 'no precedent exists' — "
        f"{recovery}."
    )


def provenance_line(config: "object") -> str:
    """Formats ``corpus_provenance`` as the one-line text header.

    Same field order as the dict, for the same reason: the project the call
    resolved reads first. The ``·`` separator is used at every site, including
    the bracketed inline forms.
    """
    p = corpus_provenance(config)
    return f"corpus: {p['project']} · {p['collection']} · {p['workspace']}"
