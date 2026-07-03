"""The Conflict sensor's core — constants + the candidate-gathering stage (2a).

This module is the seed of the sync-time Conflict sensor: a safety net inside
``mitos sync`` that judges each parsed decision entry against its undeclared close
neighbours and, at high confidence, surfaces the tension at the accept prompt. The
sensor is advisory — it applies no verb, mutates nothing, and never blocks a commit.

Phase 1a landed the numeric dials (the §8 catalog). Phase 2a adds the first pipeline
stage — :func:`gather_candidates` (§6.5 S1–S3) — plus the shared typed-degradation
shape (:class:`Unavailable` / :class:`ConflictUnavailableReason`) the whole pipeline
reuses. Later phases add the filter/rank stage (2b) and the Anthropic judgment (3b).

**Tier-1 leaf, permanently.** This module must never import a higher-tier ``mitos``
module or a heavy dependency (``anthropic``, the Qdrant/genai clients) at module
scope — ``from mitos.conflict import CONFLICT_TOP_K`` must stay cheap forever. When
2a/3b need a client, inject it as a parameter and guard the type annotation behind
``if TYPE_CHECKING:`` (the ``importer.py`` shape). The dep-free import test pins this.
The module-scope imports below (``mitos.errors``, ``mitos.identity``, and 2b's
``mitos.display``) are pure-stdlib Tier-1 leaves; the injected clients arrive as
params, typed only under ``TYPE_CHECKING``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List

from mitos.display import letter_payload
from mitos.errors import EmbeddingError, VectorStoreError
from mitos.identity import embedding_text

if TYPE_CHECKING:
    # Runtime-injected, duck-typed clients — annotated only for the type checker.
    # Importing ``mitos.protocols`` at runtime pulls ``parser`` + ``store`` (not a
    # leaf-cheap import), so these stay behind the guard. See §6 of the 2a plan.
    from mitos.parser import ParsedEntry
    from mitos.protocols import EmbeddingProvider, GraphStoreProtocol, VectorStore

# The §8 constants catalog — the sensor's honesty made numeric. Each value is the
# dial one later phase reads instead of a magic number buried in prose.

CONFLICT_SURFACE_THRESHOLD = 0.85       # CONF-D4 — surface a not-tenable finding only at ≥ this confidence (high precision over recall; a sensor that cries wolf gets muted).
CONFLICT_TOP_K = 5                      # CONF-D2/D7 — cap on the FINAL post-filter batch the LLM judge sees.
CONFLICT_JUDGMENT_TEMPERATURE = 0.3     # CONF-D5 — nuance task; temp-0 over-literalizes the contradiction judgment.
CONFLICT_LLM_TIMEOUT_S = 15             # CONF-D5/D10 — hard cap on the judgment call, 3× the P95 budget ("slow AI is failed AI", P14).
CONFLICT_SIMILARITY_FLOOR = 0.55        # ⚠️ PROVISIONAL — corpus-empirical; calibrated against the §6.3 golden fixtures in Phase 4b (CONF-D2). NOT first-principles-derivable — recall-first, so err low. Do not treat this number as final.

# CONF-D3/D7 — the single bounded over-fetch width for candidate gathering (S2). The
# raw KNN window must be wide enough that S3's non-live drops AND 2b's S4 declared/
# own-slug drops cannot shadow an undeclared neighbour out of the final top-K. A
# generous fixed margin above K (4× → 20), bounded to ONE Qdrant call (never an
# iterative re-fetch loop, P11-safe at 50K nodes). Unlike CONFLICT_SIMILARITY_FLOOR
# this is an operational tuning value, not a 4b-calibrated corpus-empirical number.
CONFLICT_OVERFETCH_LIMIT = 4 * CONFLICT_TOP_K   # = 20

# The two computed states that count as "live" for candidate gathering. Mirrors the
# proven recall idiom (surface_decisions, _adjacent_decisions): keep active ∪ drifted,
# drop superseded/corrected. Re-derived per-node via get_node_state (M3), never trusted
# from the Qdrant payload's stale ``state`` field.
_LIVE_STATES = ("active", "drifted")


class ConflictUnavailableReason(Enum):
    """Why the Conflict pipeline could not produce a result (the typed-degradation reason).

    Defined here in 2a and shared across the pipeline: 2a raises the two
    semantic-substrate reasons; 3b adds ``JUDGMENT`` / ``JUDGMENT_TIMEOUT`` members
    (additive — no edit to the two below). The reason is the machine-readable
    discriminator a surface (5a) switches on to word its user-facing notice; the
    core never formats UX text (core/surface bulkhead, CONF-D10).
    """

    EMBEDDING = "embedding_unavailable"        # Gemini embed raised (S1).
    VECTOR_STORE = "vector_store_unavailable"  # Qdrant query raised (S2).


@dataclass(frozen=True)
class Unavailable:
    """A typed degradation — the pipeline surfaced a substrate failure, did not eat it.

    Returned (never raised) by :func:`gather_candidates` when the embedding call or
    the Qdrant query fails. It is the loud, typed inverse of the fail-silent
    ``except Exception: return []`` the shipped recall helpers use: a core that
    swallowed the exception into a silent empty would make every downstream consumer
    lie about "no conflict found" (CONF-D10). ``[]`` means healthy-but-empty;
    ``Unavailable`` means degraded — the two must never blur.

    Attributes:
        reason: The typed degradation reason (the surface switches on this).
        detail: The underlying exception message, for logging/telemetry ONLY —
            never rendered to a user (the surface owns UX wording).
    """

    reason: ConflictUnavailableReason
    detail: str


@dataclass(frozen=True)
class Candidate:
    """One gathered live neighbour — a proposal's potential conflict, pre-filter.

    Carried forward from 2a (gather) to 2b (filter/rank/render). 2a returns *every*
    live over-fetched neighbour un-filtered, un-floored, un-truncated; 2b applies the
    declared-target/own-slug drop, the similarity floor, the ranking, and the top-K
    truncation, then renders ``node`` via :func:`candidate_payload`.

    Attributes:
        slug: The neighbour's slug (as returned by the vector store).
        score: The raw similarity score from the KNN query (similarity-descending).
        node: The hydrated, modifier-stamped store reader dict from
            ``get_node_by_slug`` (2b renders it; 2a carries it opaquely).
        state: The re-verified computed state — ``"active"`` or ``"drifted"``.
    """

    slug: str
    score: float
    node: Dict[str, Any]
    state: str


def gather_candidates(
    axiom: str,
    *,
    embed_provider: "EmbeddingProvider",
    vector_store: "VectorStore",
    store: "GraphStoreProtocol",
) -> "List[Candidate] | Unavailable":
    """Gathers the live decisions a proposed axiom might fight (§6.5 S1–S3).

    The first pipeline stage of the Conflict sensor: embed the proposal in *document*
    space, one bounded scope-blind KNN over-fetch, then re-verify each match's computed
    state against the graph and keep only the live ones (``active ∪ drifted``). Returns
    the live candidate set **un-filtered, un-ranked, un-truncated** (S4–S6 is 2b's job)
    — or a typed :class:`Unavailable` when the semantic substrate is unreachable.

    Three terminal states the vision forbids conflating:

    * :class:`Unavailable` — the embedding call **or** the Qdrant query raised (degraded).
    * ``[]`` — substrate healthy, but no live neighbour survived S3 (a legitimate empty:
      an empty corpus, or every top match retired). **Never** an :class:`Unavailable`.
    * ``[Candidate, ...]`` — one or more live neighbours, in query (similarity-descending)
      order.

    Degradation is narrow (CONF-D10 / plan D4): only ``EmbeddingError`` and
    ``VectorStoreError`` — the two *semantic-substrate* faults — become
    :class:`Unavailable`. A ``get_node_by_slug`` / ``get_node_state`` failure is a
    **local graph-store fault** (the same store the commit uses); masking it as
    "semantic recall unavailable" would lie, so it **propagates** (5a's surface-level
    fail-open catches it and never blocks the commit).

    Args:
        axiom: The proposed decision's raw axiom text (S1 normalizes it).
        embed_provider: The injected embedding provider (Gemini). Kept keyword-only
            so call sites read self-documenting.
        vector_store: The injected vector store (Qdrant). Scope-blind by contract.
        store: The injected graph store — the S3 source-of-truth for computed state.

    Returns:
        The list of live :class:`Candidate`\\ s (possibly empty) in similarity-descending
        order, or an :class:`Unavailable` when the embedding or vector-store call failed.

    Raises:
        DatabaseError: If a graph-store read fails (propagated, never masked — D4).
        ValidationError: If ``get_node_by_slug`` finds >1 active node for a slug
            (an MI-13 breach — a real invariant fault, not semantic unavailability).
    """
    # S1 — embed the NORMALIZED axiom in document space (is_query=False, CONF-D2/D2).
    # Route through identity.embedding_text so the S1 vector is byte-identical to what
    # the outbox embeds for the same node: the corpus-comparable text (a decision's
    # normalized axiom, mechanism_refs excluded), never the raw string. Content-hash
    # caching then makes 5a's post-commit re-embed a free cache hit (P17, no double embed).
    text = embedding_text({"kind": "decision", "axiom": axiom})
    try:
        vector = embed_provider.get_embedding(text, is_query=False)
    except EmbeddingError as exc:
        return Unavailable(reason=ConflictUnavailableReason.EMBEDDING, detail=str(exc))

    # S2 — a single bounded, scope-blind over-fetch (CONF-D3/D7). One Qdrant call, never
    # iterative: the wide window absorbs S3's non-live drops and 2b's S4 declared/self drops.
    try:
        matches = vector_store.query(vector, limit=CONFLICT_OVERFETCH_LIMIT)
    except VectorStoreError as exc:
        return Unavailable(reason=ConflictUnavailableReason.VECTOR_STORE, detail=str(exc))

    # S3 — per-match computed-state re-verification (M3, the race-guard). Order-preserved.
    # Resolve FIRST (get_node_by_slug is active-scoped → None for a retired OR absent node),
    # THEN re-derive state on the survivor — never probe get_node_state on an unresolved id
    # (it defaults an absent node to "active", store.py:1055, and would mislabel a stale
    # vector as live). Store faults here propagate (D4) — no try around the graph reads.
    candidates: List[Candidate] = []
    for match in matches:
        slug = match.get("slug")
        if not slug:  # a payload missing its slug (vector_store.py emits slug=None) — drop.
            continue
        node = store.get_node_by_slug(slug)
        if not node:  # retired (active-scoped miss) or genuinely absent — the primary filter.
            continue
        state = store.get_node_state(node["id"])
        if state not in _LIVE_STATES:  # superseded/corrected slipped past under a race — drop.
            continue
        candidates.append(
            Candidate(slug=slug, score=match.get("score", 0.0), node=node, state=state)
        )
    return candidates


# The CONF-D7 "strong relationship" set — the resolution-bearing fields the author has
# already reasoned about, so a declared target is dropped (not re-litigated). Weak edges
# (``cites``/``depends_on``/``derives_from``/``resolves``) are deliberately absent: they
# express dependence, not a resolved tension, so they must NOT shield an undeclared
# conflict from judgment. Read off ``ParsedEntry`` by attribute name at runtime (no
# import — duck-typed).
_STRONG_RELATIONSHIP_FIELDS = ("supersedes", "amends", "narrows", "contradicts", "corrects")

# The four reverse-relation modifier stamp keys ``candidate_payload`` copies from a
# hydrated node onto the surfaced finding. String-identical to
# ``store.MODIFIER_EDGE_KEYS.values()`` (store.py:66) — mirrored here as a local constant
# to keep ``conflict.py`` a leaf (no ``store`` import). Copied conditionally: a node only
# carries the non-empty ones (``_stamp_modifiers`` adds a key only when a modifier exists).
_MODIFIER_STAMP_KEYS = ("superseded_by", "amended_by", "narrowed_by", "corrected_by")


def declared_strong_targets(entry: "ParsedEntry") -> "set[str]":
    """The casefolded slugs the entry declares a STRONG relationship with.

    Union of ``entry.{supersedes, amends, narrows, contradicts, corrects}`` — the
    resolution-bearing fields the author has already reasoned about (CONF-D7). Weak
    edges (``cites`` / ``depends_on`` / ``derives_from`` / ``resolves``) are
    deliberately excluded: they express dependence, not a resolved tension, so an
    undeclared conflict hiding behind a mere ``Cites:`` still reaches judgment. The
    result is casefolded (P9 / Lesson 22 — fold at the boundary so callers can't pass
    raw case) and deduped by the ``set`` (a multi-valued field's within-field
    duplicates collapse for free). An empty declaration set is the common case.

    Args:
        entry: The parsed decision entry whose strong-relationship targets to collect.

    Returns:
        The set of casefolded declared strong-target slugs (possibly empty).
    """
    targets: set[str] = set()
    for field in _STRONG_RELATIONSHIP_FIELDS:
        for slug in getattr(entry, field, ()) or ():
            targets.add(slug.casefold())
    return targets


def screen_candidates(
    candidates: "List[Candidate]",
    *,
    declared_targets: "set[str]",
    own_slug: str,
    floor: float = CONFLICT_SIMILARITY_FLOOR,
    top_k: int = CONFLICT_TOP_K,
) -> "List[Candidate]":
    """Filters, floors, ranks and truncates 2a's gathered neighbours (§6.5 S4–S6).

    The second and final candidate-pipeline stage. Given 2a's raw ``list[Candidate]``
    (every live over-fetched neighbour, un-filtered), produce the judged batch:

    * **S4 — drop the already-reasoned.** Remove any candidate whose slug is a declared
      strong-relationship target (``declared_targets``) or the proposal's own slug
      (``own_slug`` — the false-self-conflict guard, RF-1). Both compared casefolded.
    * **S5 — floor gate.** Keep only candidates with ``score >= floor`` (inclusive).
    * **S6 — rank + truncate.** Sort the survivors similarity-descending and keep at
      most ``top_k``.

    The S4 → S5 → S6 order is load-bearing (CONF-D7 "Shadowing"): dropping declared/self
    *before* the floor and *before* truncation means a high-similarity declared neighbour
    can never consume a ``top_k`` slot and shadow a genuine undeclared conflict out of the
    window. 2a's over-fetch sizes the raw list so the margin is spent on real candidates.

    Storeless and pure — no I/O, no embedding, no state re-verify (2a did that). Returns
    ``[]`` (a *clean* short-circuit) when the input is empty, everything was
    declared/self, or every survivor fell below the floor. This ``[]`` is never
    :class:`Unavailable` — 2b has no degradation type; the facade (3b) handles 2a's
    degradation upstream ("Degraded ≠ empty", §6.5).

    ``floor`` and ``top_k`` default to the module constants but are injectable so tests
    pin behaviour with an explicit floor rather than chasing the PROVISIONAL
    ``CONFLICT_SIMILARITY_FLOOR`` (4b recalibrates it — CONF-D2).

    Args:
        candidates: 2a's gathered live neighbours (similarity-descending; may be empty).
        declared_targets: The casefolded declared strong-target slugs, from
            :func:`declared_strong_targets`.
        own_slug: The proposal's own slug at check time (RF-1); folded here.
        floor: The inclusive similarity floor (default ``CONFLICT_SIMILARITY_FLOOR``).
        top_k: The cap on the returned batch (default ``CONFLICT_TOP_K``).

    Returns:
        The judged batch — undeclared live neighbours at or above ``floor``, ranked
        similarity-descending, truncated to ``top_k``. Possibly empty.
    """
    # S4 — build the drop set once (declared_targets is already folded; add the folded
    # own slug) and drop by casefolded membership. casefold on BOTH sides (Lesson 22, P9).
    drop = declared_targets | {own_slug.casefold()}
    kept = [c for c in candidates if c.slug.casefold() not in drop]
    # S5 — inclusive floor gate.
    above = [c for c in kept if c.score >= floor]
    # S6 — rank similarity-descending, truncate to top_k.
    ranked = sorted(above, key=lambda c: c.score, reverse=True)
    return ranked[:top_k]


def candidate_payload(candidate: "Candidate", *, brief: bool = False) -> "Dict[str, Any]":
    """Renders a surfaced candidate into its Letter-mode finding (modifier stamps ride along).

    The finding shape 5a surfaces at the accept prompt: the shared Letter core
    (``slug`` / ``axiom`` / ``scope`` / — unless ``brief`` — ``rejected_paths``) from
    :func:`~mitos.display.letter_payload`, the raw similarity under ``score``, plus the
    reverse-relation modifier stamps (``amended_by`` / ``narrowed_by`` / ``superseded_by``
    / ``corrected_by``).

    Storeless (plan D4-primary): the stamps are copied straight off ``candidate.node``,
    which 2a hydrated via ``get_node_by_slug`` — already ``_stamp_modifiers``-run
    (store.py:1037), so the node carries the *non-empty* stamps from the same
    ``get_modifiers`` source, one hop earlier. No redundant store read, no fault surface.
    ``letter_payload`` drops modifier keys (a store-free leaf), so 2b re-copies them here —
    the third caller mirroring the ``cli.py`` / ``mcp_server._decision_payload`` pattern
    (each caller stamps around the shared leaf, D1). Stamping happens *after*
    ``letter_payload`` returns, so ``brief`` (which governs only ``rejected_paths``) never
    drops a stamp — an amended-but-active candidate never reads as the final word (the
    "amended axioms read as live" trap).

    Args:
        candidate: A survivor of :func:`screen_candidates`; its ``node`` is the hydrated,
            modifier-stamped ``get_node_by_slug`` reader dict.
        brief: When True, omit ``rejected_paths`` (keeps stamps). Default False — the
            surfaced finding wants the candidate's M5 anti-knowledge.

    Returns:
        The Letter-mode finding dict: ``slug``, ``axiom``, ``scope``, ``score``, any
        present modifier stamps, and (unless ``brief``) ``rejected_paths``.
    """
    payload = letter_payload(
        candidate.node, brief=brief, extras={"score": candidate.score}
    )
    # Copy the reverse-relation stamps already on the hydrated node (D4-primary). Only the
    # non-empty ones are present (``_stamp_modifiers`` adds a key only when a modifier
    # exists), so copy conditionally — blind indexing would KeyError on the common
    # unmodified case.
    for key in _MODIFIER_STAMP_KEYS:
        if key in candidate.node:
            payload[key] = candidate.node[key]
    return payload
