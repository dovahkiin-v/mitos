"""Sync pipeline for Mitos.

This module implements the core V3a and V3b sync loops, managing snapshotting,
concurrency file locks, LLM capture enrichment, user reviews, and content-aware
archive rotation.
"""

import dataclasses
import functools
import hashlib
import math
import os
import sys
import shutil
import re
import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_FLOOR
from typing import (
    TYPE_CHECKING, AbstractSet, List, Dict, Mapping, NamedTuple, Optional, Any, Sequence, Set,
    Tuple, Callable,
)
from filelock import FileLock, Timeout

if TYPE_CHECKING:
    # The SDK is imported inside the two functions that call it: `google.genai`
    # costs ~0.7s, and `cli` imports this module for every verb.
    from google import genai

from mitos import __version__ as MITOS_VERSION
from mitos import amend
from mitos import atomic_file
from mitos import rotation
from mitos import settledness
from mitos.config import MitosConfig, hint_due, judge_api_key
from mitos.conflict import (
    CONFLICT_CANDIDATE_SOURCE,
    CONFLICT_OVERFETCH_LIMIT,
    CONFLICT_PROMPT_VERSION,
    CONFLICT_TOP_K,
    ConflictCheckResult,
    ConflictFinding,
    ConflictUnavailableReason,
    Gathering,
    SEMANTIC_SUBSTRATE_REASONS,
    Unavailable,
    candidate_payload,
    gather_candidates,
    run_conflict_check,
    screen_candidates,
)
from mitos.telemetry import (CommentaryAuditRow, ConflictCheckRow, JudgmentBatch,
                            TelemetryStore, read_last_attempt)
from mitos.check_notice import compose_check_notice
from mitos.audit_debt import AuditDebt, derive_audit_debt
from mitos.divergence import (
    code_point_names, corpus_holds_entries, declared_edges, differing_span, entry_divergence,
    is_reconcilable, judged_text, needs_code_points,
)
from mitos.errors import (
    CollectionMissingError,
    EmbeddingError,
    MitosError,
    SynthesisError,
    ParseError,
    ValidationError,
    DatabaseError,
    EntryFailure,
    VectorStoreError,
    VectorStoreUnreachableError,
    CommitError,
    STORE_CYCLE_VIOLATION,
    STORE_DANGLING_EDGE,
    STORE_KIND_CONSTRAINT_VIOLATION,
    STORE_MISSING_TARGET,
    STORE_SLUG_COLLISION,
)
from mitos.models import get_embedding_model_id, get_model_id
from mitos.parser import (ParsedEntry, load_format_spec, mask_inline_code,
                          parse_entry_stream, parse_file_reversed)
from mitos.replay import commit_quarantine_fixpoint
from mitos.store import (
    GraphStore,
    CommitDelta,
    _utc_now_iso,
    _strip_citation,
    _EDGE_KIND_REQUIREMENT,
    _KILL_EDGE_FIELDS,
    MODIFIER_EDGE_KEYS,
    REJECTION_BACKLOG_FAILURE_TOLERANCE,
    RejectionBacklogRow,
    edge_kind_is_legal,
    live_ends_from_chains,
    LIVE_STATES,
)
from mitos.display import retired_handle
from mitos.retrieval_walk import EDGE_FOUND, EDGE_UNMEASURED
from mitos.identity import (
    SLUG_LENGTH_REASON,
    SLUG_MAX_LEN,
    compute_node_id,
    embedding_text,
    mechanism_canonical_norm,
)
from mitos.embeddings import GeminiEmbeddingProvider, fit_batch
from mitos.provider_cause import (AUTH, QUOTA, TRANSIENT, classify_provider_error,
                                  describe_embed_failure)
from mitos.shell_quote import shell_flag
from mitos.tool_markup import field_values, find_tool_call_markup
from mitos.recall import missing_graph_is_a_gap
from mitos.rejections import (diff_rejection_index, expected_rejection_points,
                              rejection_points, render_rejected_paths)
from mitos.vector_store import (QdrantRejectionIndex, QdrantVectorStore, RejectionPoint,
                                hash_to_uuid, partition_points)
from mitos.renderer import MitosRenderer, OVERFLOW_HINT_WINDOW_SECONDS, summarize_overflows
from mitos.restore import BufferFidelityError, verify_amended_buffer


@dataclass
class _ConflictSyncRun:
    """The per-run context the sync-time Conflict sensor threads through its checks (5b).

    The one piece of mutable per-run state the sensor needs, built ONCE per
    ``perform_sync`` at the judge-build site and passed to every
    :meth:`MitosSyncManager._run_and_surface_conflict` call of the loop. Threading it
    explicitly (rather than hiding it on ``self``) makes the write-then-read aggregate
    breaker legible: ``breaker_tripped`` is set on the entry that degrades and read at
    the top of the next entry's check, so a single downstream outage costs one penalty,
    not N (P7 active-bulkhead, vision §4). Reset is structural — a fresh run means a
    fresh instance (``breaker_tripped=False``), so a prior sync's trip can never leak
    into the next; there is no explicit un-trip step to forget.

    Attributes:
        sync_run_id: The run-correlation id minted per sync (a ``uuid4().hex``), stamped
            on every ``conflict_checks`` row so a run's judgments read as one thread
            (P16 One Thread of Truth).
        telemetry: The best-effort telemetry store, or ``None`` when it could not be
            constructed — the sensor still surfaces findings, it just can't persist them.
        breaker_tripped: Set ``True`` on the first typed degradation of the run; every
            later entry's check then short-circuits to a true no-op (no gather, no
            query, no judge).
    """

    sync_run_id: str
    telemetry: Optional[TelemetryStore]
    breaker_tripped: bool = False


def run_sync_enrichment(
    client: "genai.Client",
    entry: ParsedEntry,
    active_decisions: List[Dict[str, Any]],
    *,
    model_id: Optional[str] = None
) -> Dict[str, Any]:
    """Calls Gemini to refine a decision, infer scopes, and suggest relationships.

    NOTE: no longer invoked by ``mitos sync`` — the sync runtime path is strict-
    deterministic and commits the human-authored buffer verbatim (ADR
    ``sync-strict-deterministic-no-llm-enrichment``). LLM refinement of genuinely-raw
    input lives in ``capture`` / ``import --llm-extract``. This function is retained
    only as the live test-suite's generative-quota probe target
    (``tests/live_helpers.py``); it is dead in the production path and a candidate for
    removal once that probe is repointed to a still-live generative call.

    Args:
        client: The Gemini client the caller constructed.
        entry: The parsed entry to enrich.
        active_decisions: The active decisions to summarise into the prompt.
        model_id: The resolved id for ``FLASH_LITE``, taken off the calling
            workspace's ``config.env``, or ``None`` for the baseline (2c).
    """
    active_summary = ""
    for d in active_decisions[:20]:  # Limit to top 20 active decisions for prompt budget
        active_summary += f"- slug: {d['slug']}\n  axiom: {d['core_axiom']}\n  scope: {','.join(d['scope'])}\n\n"

    prompt = f"""
You are the Mitos v0.1 capture enrichment agent. Your task is to refine and enrich a newly captured architectural decision.

Here are some currently ACTIVE decisions in the workspace:
{active_summary}

Here is the proposed decision entry:
Slug: {entry.slug}
Decided: {entry.axiom}
Rejected: {entry.rejected_paths}
Mechanisms: {','.join(entry.mechanisms)}
Scope: {','.join(entry.scope)}
Context: {entry.context}

Please enrich this entry. You must:
1. Verify and refine the `core_axiom` into a single, extremely precise, clear, and unambiguous sentence. If the raw axiom is already high quality, keep it verbatim or make minimal grammatical corrections.
2. Verify and refine the list of mechanism tags.
3. Suggest appropriate scope tags based on the content and existing active decisions.
4. Detect if this decision should supersede, amend, narrow, or depend on any of the active decisions listed above. If so, return their slugs in the suggested relationships.

Respond strictly in valid JSON format with the following keys:
- refined_core_axiom (string)
- refined_mechanisms (list of strings)
- refined_scope (list of strings)
- suggested_relationships (object with keys: supersedes, amends, narrows, depends_on, resolves)
"""
    resolved_model = (
        model_id if model_id is not None else get_model_id("FLASH_LITE")
    )
    from google.genai import types

    try:
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3
            )
        )
        return json.loads(response.text)
    except Exception as e:
        raise SynthesisError(f"LLM enrichment call failed: {str(e)}")


def run_ambient_capture(
    client: "genai.Client", raw_text: str, *, model_id: Optional[str] = None
) -> str:
    """Uses FLASH to convert raw conversational text into a canonical Markdown entry.

    Args:
        client: The Gemini client the caller constructed.
        raw_text: The developer's raw conversational input.
        model_id: The resolved id for ``FLASH``, taken off the calling
            workspace's ``config.env``, or ``None`` for the baseline (2c).
    """
    prompt = f"""
You are the Mitos v0.1 capture scribe. Convert the following developer conversation or thought into a canonical Mitos Decision Entry.

Input text:
"{raw_text}"

Please generate a canonical Markdown entry. Use exactly this format (do not include markdown block quotes):

### [slug]

**Decided:** [Single-sentence axiom that is true going forward]
**Rejected:**
- [alternative] — [specific reason why it was rejected, be precise and adversarial]
**Mechanisms:** [comma-separated mechanisms, or none]
**Scope:** [comma-separated scope tags, or none]
**Context:** [brief background context explaining why this decision was made]

[DECISION_TRANSCRIPT]
User: {raw_text}
[/DECISION_TRANSCRIPT]

Make sure the slug is a clean, lowercase hyphenated string that matches the decision topic.
"""
    resolved_model = model_id if model_id is not None else get_model_id("FLASH")
    from google.genai import types

    try:
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3
            )
        )
        return response.text.strip()
    except Exception as e:
        raise SynthesisError(f"Ambient capture synthesis failed: {str(e)}")


# --- record_decision helpers (write-half of the MCP server) ---

# The exact buffer marker. Its consumers are record_decision_entry here and
# cmd_capture / cmd_restore_source in cli.py; auto_heal_decisions_file and the
# `mitos init` seed keep their own byte-identical copies. The `—` is an em dash;
# do not retype it.
_ENTRIES_MARKER = "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->"

# A content line that looks like a Mitos field header (e.g. `**Decided:**`); the
# parser would treat it as a new field and corrupt the entry (parser.py:400-409).
_FIELD_LINE_RE = re.compile(r'^\s*\*\*[A-Za-z -]+:\*\*')

# A column-0 H2/H3 heading opens a NEW entry in the parser (parser.py:308). Note
# this is deliberately narrow: a single `#` (H1), `####`+ headings, and any
# *indented* heading are all SAFE and must NOT be rejected.
_SECTION_HEADER_RE = re.compile(r'^#{2,3}(?!#)')

# Inline markers the parser reacts to: section/transcript/buffer boundaries and
# the [NOTE:]/[PARKED:] scanners that siphon content into entry.notes (parser.py:419-426).
_STRUCTURAL_MARKERS = (
    "[DECISION_PARKED:",
    "[DECISION_TRANSCRIPT]",
    "[/DECISION_TRANSCRIPT]",
    "BEGIN ENTRIES",
    "[NOTE:",
    "[PARKED:",
)

# The exact agent-facing error messages (spec §5). Each says what went wrong AND
# how to recover, interpolating the offending value.
_ERROR_MESSAGES: Dict[str, str] = {
    "not_initialized": "No Mitos workspace found here. Run 'mitos init' before recording decisions.",
    "empty_axiom": "'axiom' is empty. Provide the decision as a single clear sentence that is true going forward.",
    "empty_slug": "'slug' is empty. Provide a short, explicit, hyphenated handle (e.g. 'sqlite-wal-mode').",
    "slug_too_long": "slug '{slug}' is {length} characters — {over} over the {max}-character limit. " + SLUG_LENGTH_REASON + ". Pass a shorter 'slug' of at most {max} characters.",
    "missing_rejected_paths": "'rejected_paths' is required: state the alternatives you considered and why you ruled them out — this is what stops you or another agent from re-proposing them later.",
    "parse_failed": "The decision could not be serialised into a valid entry — most likely a structural token in axiom/rejected_paths/context: a line beginning with '##' or '###' (indent it or use '#'/'####' instead), a line shaped like '**Something:**', or a '[DECISION_TRANSCRIPT]' / '[DECISION_PARKED:' / 'BEGIN ENTRIES' / '[NOTE:' / '[PARKED:' marker. Remove or rephrase that line and retry — or, to mention a marker as prose, wrap it in backticks (`...`): inline-code spans are exempt.",
    "slug_collision": "A different decision already uses the slug '{slug}'. Give this one a distinct 'slug'; and if it is meant to replace the existing decision, also set supersedes='{slug}' (the new decision must still have its own slug — two decisions cannot share one).",
    "supersedes_not_found": "supersedes='{supersedes}' does not match any existing decision. Look it up first with query_decisions to get the exact slug, or omit 'supersedes' if this is a brand-new decision.",
    "supersedes_ambiguous": "supersedes='{supersedes}' matches more than one decision. Use query_decisions to find the exact, full slug and pass that.",
    "corrects_not_found": "corrects='{corrects}' does not match any existing decision. Look it up first with query_decisions to get the exact slug, or omit 'corrects' if this is a brand-new decision.",
    "corrects_ambiguous": "corrects='{corrects}' matches more than one decision. Use query_decisions to find the exact, full slug and pass that.",
    "relation_target_not_found": "{relation}='{target}' does not match any existing decision. Look it up first with surface_decisions/query_decisions to get the exact slug, or omit '{relation}' if no such link applies.",
    "relation_target_ambiguous": "{relation}='{target}' matches more than one decision. Use query_decisions to find the exact, full slug and pass that.",
    "commit_failed": "The decision validated but the commit failed and nothing was written: {reason}. Retry; if it persists, the workspace store may be locked or corrupt.",
    "derives_from_on_decision": "derives_from is not valid when recording a decision: a derives_from edge originates from an open question (open_question -> decision), so a decision can never be its source. If you mean 'this decision builds on that one', use cites instead.",
    "draft_drifted": "This re-send's {fields} changed from the draft its draft_digest came from — the values of the call that paused, not your latest send. Restore those values and re-send with the same draft_digest, or omit draft_digest (which always clears this comparison) to record the draft as it is now.",
    "tool_call_markup": "{field!r} holds tool-call markup: {span!r} at character {offset}{more}. That is syntax from the tool call itself, not part of the decision, so nothing was recorded. Remove it and send the call again — or, to mention such a tag as prose, wrap it in backticks (`...`): inline-code spans are exempt.",
    "draft_digest_unreadable": "draft_digest is not a value a pause returned, so this re-send cannot be compared with its draft. Pass the pause's draft_digest exactly as it came, or omit draft_digest (which always clears this comparison).",
}

# The user-facing typed relations beyond `supersedes` (which is special: it changes
# computed state and has its own error codes). Each maps an agent-facing kwarg name to
# its canonical decisions.md field label. The parser and the store's commit path
# already understand all of these (format-spec.md §"Relationship Fields"); the agentic
# write path just had to serialize + validate them the way it already does supersedes.
_EXTRA_RELATIONS = (
    ("amends", "Amends"),
    ("narrows", "Narrows"),
    ("depends_on", "Depends-On"),
    ("resolves", "Resolves"),
    ("contradicts", "Contradicts"),
    ("derives_from", "Derives-From"),
    ("cites", "Cites"),
)

# The `created` receipt's identification echo (B4): which relations carry the
# target's identifying text beside its state and stamps. Named as the sets of
# relations that ACT on their target — a wrong-but-existing slug there changes the
# wrong decision, and the receipt is the only read-back of what was hit — never as
# "all but cites/depends_on", so a future relation carries no text until someone
# decides it acts. ADR
# `record-receipt-edges-are-a-stamped-read-surface-axiom-rides-acting-relations-only`.
_ECHO_AXIOM_RELATIONS: frozenset = frozenset(
    {"supersedes", "corrects", "amends", "narrows", "contradicts"}
)
# `resolves` targets an open question, whose identifying text is its Topic field.
_ECHO_TOPIC_RELATIONS: frozenset = frozenset({"resolves"})


def _split_relation_slugs(raw: Optional[str]) -> List[str]:
    """Split a comma-separated relation argument into a list of stripped slugs.

    The decisions.md relationship fields are comma-separated multi-valued (V1b,
    format-spec.md "Relationship Fields"), and the markdown parser already tokenizes
    them so; this mirrors that split for the CLI/MCP ``record`` edge args, so a single
    ``--supersedes "a, b, c"`` (or ``supersedes="a, b, c"`` on the MCP twin) commits one
    edge per slug. A lone slug is a 1-element list, so single-valued authoring stays
    byte-for-byte unchanged — purely additive, no caller breaks.

    Args:
        raw: The relation argument as the caller passed it — ``None`` or a possibly
            comma-separated string of exact slugs.

    Returns:
        The stripped, non-empty slugs in author order (``[]`` when ``raw`` is falsy).
    """
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def _record_error(code: str, **fields: Any) -> Dict[str, str]:
    """Builds a structured {error, code} dict using the canonical message for ``code``."""
    return {"error": _ERROR_MESSAGES[code].format(**fields), "code": code}


# The pause's draft digest (A3). mitos holds nothing between calls, so the digest
# travels with the caller: a version tag, then one short hash per field in this fixed
# order, so a re-send that passes it back can be told WHICH fields drifted. The names
# are the record_decision argument names, so a drifted field maps straight back to the
# argument the caller sent. Never stored, never an identity input (MI-1/MI-7).
DRAFT_DIGEST_FIELDS = ("slug", "axiom", "rejected_paths", "context", "scope", "mechanisms")
_DRAFT_DIGEST_TAG = "d1"
_DRAFT_DIGEST_WIDTH = 10  # hex chars per field: 40 bits against accidental drift, not a MAC
_DRAFT_DIGEST_RE = re.compile(
    rf"{_DRAFT_DIGEST_TAG}(\.[0-9a-f]{{{_DRAFT_DIGEST_WIDTH}}}){{{len(DRAFT_DIGEST_FIELDS)}}}")

# The pause's "nothing is held" clause, single-sourced: the message appends the digest
# clause to it; the CLI text pause prints it alone (its render carries no digest).
PAUSE_HELD_CLAUSE = ("Nothing is held between calls: re-send the same values, "
                     "declarations included, with the relation or acknowledgement added")
_PAUSE_DIGEST_CLAUSE = ("; pass this pause's draft_digest with them to have the re-send "
                        "compared with this draft, or omit it if you changed the draft "
                        "on purpose.")


def _draft_digest_segment(value: Any) -> str:
    """Hashes one draft field's value to its short digest segment.

    ``ensure_ascii=True`` is deliberate and the opposite of
    ``identity.canonical_core_json_form``: a lone surrogate becomes a ``\\udc80``
    escape rather than a UTF-8 encode, so this can never raise. Don't harmonise the two.

    Args:
        value: The field as the built entry holds it (a string, ``None``, or a list).

    Returns:
        The first ``_DRAFT_DIGEST_WIDTH`` hex characters of the value's SHA-256.
    """
    blob = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(blob).hexdigest()[:_DRAFT_DIGEST_WIDTH]


def _draft_digest(entry: ParsedEntry) -> str:
    """Computes the draft digest a pause returns, over the built entry.

    Reads the entry ``parse_entry_stream`` built, never the raw arguments, so the digest
    canonicalises exactly as the write does and adds no normalisation of its own
    (case-folded scope compares equal; reordered scope is drift, MI-9).

    Args:
        entry: The built entry of the call being paused or compared.

    Returns:
        ``d1.<slug>.<axiom>.<rejected_paths>.<context>.<scope>.<mechanisms>`` segments.
    """
    return ".".join([_DRAFT_DIGEST_TAG] + [
        _draft_digest_segment(getattr(entry, field)) for field in DRAFT_DIGEST_FIELDS])


def _readable_fields(fields: List[str]) -> str:
    """Joins field names as prose: ``a``, ``a and b``, ``a, b and c``."""
    return fields[0] if len(fields) == 1 else f"{', '.join(fields[:-1])} and {fields[-1]}"


def _draft_drift(entry: ParsedEntry, draft_digest: Optional[str]) -> Optional[Dict[str, Any]]:
    """Compares a re-send's built entry with the draft its digest came from.

    Pure and in memory: no store read, no raise. An absent or blank digest is omitted
    (the call behaves exactly as without one). A value this build cannot read is refused
    with its own code, never read as drift of all six fields.

    Args:
        entry: The re-send's built entry.
        draft_digest: The digest the caller passed back, or ``None``.

    Returns:
        ``None`` when there is nothing to refuse; otherwise a ``draft_digest_unreadable``
        refusal, or a ``draft_drifted`` refusal carrying ``drifted_fields`` (a list, in
        :data:`DRAFT_DIGEST_FIELDS` order).
    """
    if draft_digest is None:
        return None
    if not isinstance(draft_digest, str):
        return _record_error("draft_digest_unreadable")
    given = draft_digest.strip().lower()
    if not given:
        return None  # blank is omitted, as for the relation arguments
    if not _DRAFT_DIGEST_RE.fullmatch(given):
        return _record_error("draft_digest_unreadable")
    current = _draft_digest(entry).split(".")[1:]
    drifted = [field for field, now, then in zip(DRAFT_DIGEST_FIELDS, current,
                                                  given.split(".")[1:])
               if now != then]
    if not drifted:
        return None
    return {**_record_error("draft_drifted", fields=_readable_fields(drifted)),
            "drifted_fields": drifted}


def _tool_call_markup_refusal(arguments: List[Tuple[str, Any]]) -> Optional[Dict[str, Any]]:
    """Refuses a record call that would write tool-call markup.

    Pure and in memory: no store read, no raise (the scanner skips non-strings).

    Args:
        arguments: ``(name, value)`` for every argument the call writes; a list value
            is scanned element by element as ``name[i]``.

    Returns:
        ``None`` when nothing is found; otherwise a ``tool_call_markup`` refusal
        carrying ``markup_spans`` (every hit, ``{field, span, offset}``).
    """
    hits = find_tool_call_markup(
        pair for name, value in arguments for pair in field_values(name, value))
    if not hits:
        return None
    first, rest = hits[0], len(hits) - 1
    more = f", plus {rest} more span{'s' if rest > 1 else ''}" if rest else ""
    return {**_record_error("tool_call_markup", field=first["field"], span=first["span"],
                            offset=first["offset"], more=more),
            "markup_spans": hits}


def _divergence_field_lines(field: str, graph_raw: Optional[str],
                            markdown_raw: Optional[str], *, full: bool) -> List[str]:
    """Renders one diverged commentary field for the reconcile report.

    The span is located on the judged values (``judged_text``), never on the raw
    reprs: the comparator strips and casefolds a slug, so a raw diff can point at a
    difference it ignored. The span is printed whole and with ``repr``, which shows
    whitespace and invisible characters but not a homoglyph, so a span holding
    anything outside printable ASCII also names its code points, on both sides.

    Args:
        field: The ``COMMENTARY_FIELDS`` name ``entry_divergence`` listed.
        graph_raw: The stored value, as read from the node.
        markdown_raw: The parsed value, as read from the entry.
        full: ``--full``: the two whole raw values, byte for byte as before the span
            report existed.

    Returns:
        The field's lines, unterminated.
    """
    span = None if full else differing_span(judged_text(field, graph_raw),
                                            judged_text(field, markdown_raw))
    if span is None:
        # `--full`, or a listed field whose judged values agree. The second cannot
        # happen while the comparator and `judged_text` agree (a test pins that);
        # if they ever drift, both whole sides is honest and "no difference" under a
        # divergence header is not.
        return [f"  {field}:",
                f"    graph:    {graph_raw!r}",
                f"    markdown: {markdown_raw!r}"]
    label = f"{field} (compared casefolded)" if field == "slug" else field
    lines = [
        f"  {label}: parts after {span['offset']} shared characters "
        f"(graph {span['graph_len']}, markdown {span['markdown_len']}; "
        f"{span['shared_after']} shared at the end)",
        f"    graph:       {span['graph']!r}",
        f"    markdown:    {span['markdown']!r}",
    ]
    if needs_code_points(span["graph"], span["markdown"]):
        lines.append(f"    code points: graph {code_point_names(span['graph'])}; "
                     f"markdown {code_point_names(span['markdown'])}")
    return lines


class _AmendAnswer(Exception):
    """Carries an in-band amendment answer out of ``splice_buffer``.

    Raised, never returned as text: a transform that returned the original buffer would
    still write it, run ``after_write``, commit a no-op and attribute it. Raised from the
    transform it writes nothing; raised from ``after_write`` it rolls the buffer back.
    """

    def __init__(self, result: Dict[str, Any]) -> None:
        super().__init__(result.get("status"))
        self.result = result


class _AmendFault(Exception):
    """Carries an amendment's environment or graph fault out of ``after_write``.

    A private type, so the rollback path can tell it apart from ``splice_buffer``'s own
    rollback-failure ``MitosError`` — nearly every mitos exception is a ``MitosError``.
    """

    def __init__(self, code: str, **fields: Any) -> None:
        super().__init__(code)
        self.code = code
        self.fields = fields


def _embedding_input_text(
    kind: str,
    axiom: Optional[str] = None,
    topic: Optional[str] = None,
    questions_raised: Optional[List[str]] = None,
) -> str:
    """Derives the V1a embedding-input string for an entry or node (C2/M8 single source).

    Routes through :func:`identity.embedding_text` so the record-time and drain-time
    embedding text are byte-identical — a node embedded inline at record time and the
    same node re-derived from the Outbox at drain time yield the same vector (the
    ``embedding_text`` column is gone in V1a, so drain re-derives from the immutable
    core, M8). Bridges the one reader-key gap: a decision's axiom is exposed as
    ``axiom`` on a :class:`ParsedEntry` but ``core_axiom`` on a store node dict —
    callers pass whichever they hold under ``axiom``.

    Args:
        kind: ``"decision"`` or ``"open_question"``.
        axiom: The decision axiom (for a decision entry/node).
        topic: The open_question topic.
        questions_raised: The open_question's questions, in authored order.

    Returns:
        The embedding-input string (normalized, M8-consistent with the hashed core).
    """
    return embedding_text({
        "kind": kind,
        "axiom": axiom,
        "topic": topic,
        "questions_raised": questions_raised or [],
    })


def _contains_structural_token(text: str) -> bool:
    """Returns True if any line of ``text`` would corrupt the flat-file parser.

    Matches the parser's own raw-line semantics (no leading-whitespace strip for
    headers, parser.py:308): only column-0 ``##``/``###`` headings, field-shaped
    lines, and the inline markers trigger.
    """
    for line in text.split("\n"):
        if _SECTION_HEADER_RE.match(line):
            return True
        if _FIELD_LINE_RE.match(line):
            return True
        # Code-span-masked, matching the parser's own scanners: a backtick-quoted
        # marker (`BEGIN ENTRIES`, `[NOTE: …]`) is prose about the marker, and the
        # parser no longer reacts to it — so the guard must not reject it either.
        masked = mask_inline_code(line)
        if any(marker in masked for marker in _STRUCTURAL_MARKERS):
            return True
    return False


# The citation-handle length cap. The slug is the handle other decisions cite (it is
# not part of the canonical-core identity, V1-D2, and can be renamed) — generous
# enough for a descriptive multi-word handle (real corpus handles top out ~68 chars), firm enough to keep one
# from running away. An *explicit* slug over this is REJECTED with an exact char count
# (never silently truncated — a silent trim diverges the stored handle from the one the
# author already cited: self-inflicted citation rot). Auto-derived slugs still trim to it.
# NOTE: this cap is advertised up-front to callers — the CLI `--slug` help imports this
# constant; the MCP `record_decision` slug docstring carries the number as a literal. If
# you change it, update that docstring (mcp_server.py) too.
_SLUG_MAX_LEN = SLUG_MAX_LEN  # shared source of truth in identity.py (re-exported here for cli.py + the record path)
_SLUG_MIN_LEN = 32  # auto-derive only: don't trim a word boundary back past here — hard-cap instead


def _mechanisms_fold_map(authored: List[str]) -> Dict[str, str]:
    """Maps each authored mechanism token the identity fold changes to its folded form.

    Tokens the fold leaves byte-identical are omitted, so the map is ``{}`` when
    nothing changed. Keys keep first-occurrence authored order. Pure and total: it
    only reads :func:`~mitos.identity.mechanism_canonical_norm`, the fold that keys
    the mechanism registry and feeds the node id.

    Args:
        authored: The mechanism tokens as ``decisions.md`` holds them.

    Returns:
        ``{authored: folded}`` for every token whose folded form differs.
    """
    folded: Dict[str, str] = {}
    for token in authored:
        canon = mechanism_canonical_norm(token)
        if canon != token:
            folded.setdefault(token, canon)
    return folded


def _edge_state_labels(pairs: List[Dict[str, str]]) -> List[str]:
    """Renders edges as a sorted ``["kind:target", ...]`` state list for the audit row.

    Args:
        pairs: Edge dicts carrying ``kind`` and ``target``.

    Returns:
        The sorted labels, deduplicated.
    """
    return sorted({f"{p['kind']}:{p['target'].strip().casefold()}" for p in pairs
                   if p.get("target")})


# Returned on the two "exists" short-circuits, which write nothing at all. The
# no-op is deliberate (a committed canonical core is immutable, M1); what was not
# deliberate is that the receipt once reported the buffer path as a write, so a
# re-record aimed at correcting commentary read as success while changing nothing
# (#5b). The fact is shared: the CLI imports it for its headline. The note is the
# dict's, and the dict is an MCP boundary (`record --json` emits it verbatim), so its
# recovery names the `amend_commentary` tool and no shell command; the CLI text
# composes its own selectored recipes (ADR
# receipt-dict-strings-are-mcp-boundary-so-recovery-splits-per-renderer). The two
# states the tool cannot reach both answer `archived` there, so the note states
# them as a pair and claims neither.
EXISTS_NO_OP_FACT = (
    "already recorded — this call wrote nothing, to the buffer or the graph. "
    "A committed decision's axiom and mechanisms are immutable (M1): re-recording "
    "them is a no-op."
)
# The note's state-blind middle, shared by the live note and the retired one (7b):
# relations are not commentary, and the two states the tool cannot reach.
_EXISTS_RELATIONS_AND_ARCHIVED = (
    "Relations are not "
    "commentary and that tool refuses them: relation lines are edited in decisions.md "
    "and applied by a sync reconcile, which no tool here performs — a person with a "
    "shell in that project runs it. The tool reaches an entry only while its `### ` "
    "block is in decisions.md. A node with no source block and an entry already "
    "rotated into decisions/archive/ both answer `archived` from it, and the tool cannot "
    "tell the two apart; either one is repaired by a person."
)
_EXISTS_COMMENTARY_FIELDS = (
    "(rejected_paths, scope, invalidates_if, context, or its slug with new_slug)"
)
_EXISTS_NO_OP_NOTE = (
    EXISTS_NO_OP_FACT + " "
    f"To change its commentary {_EXISTS_COMMENTARY_FIELDS}, call the `amend_commentary` "
    "tool on this same project with "
    "this slug; it edits the entry in place and keeps the id. "
    + _EXISTS_RELATIONS_AND_ARCHIVED + " To record a changed "
    "decision, record a new one with supersedes or amends naming this slug."
)
# The handle keys a retired node's `exists` receipt adds (7b, vision §4.9), and the
# four that go `null` together when the handle read faults.
_EXISTS_HANDLE_KEYS: Tuple[str, ...] = (
    "superseded_by", "corrected_by", "live_end", "successor_ambiguous",
)

# The standing coherence debt, stated without a number: the `created` receipt's
# fallback when the audit-debt count cannot be read (`_audit_debt_line` states it with
# one; `MitosSyncManager._audit_debt_fields` picks). The REGISTER is the mechanism,
# not the presence: `mitos check` reuses prior verdicts (a reused pair never enters a
# batch), so one deferred run after N writes covers the same ground as N runs for less
# — and `_confirm_spend` only fires above CHECK_CONFIRM_BATCHES fresh groups, so a
# caller auditing per write presents ~1 fresh group forever and the tree's only spend
# ring never fires. A line reading "audit this write" therefore converts one owed run
# into N and fragments the amortization (ADR
# `record-receipt-states-cumulative-audit-debt-not-per-write-work`). So: cumulative
# and corpus-wide, no imperative, no per-entry referent — and no command, because this
# string is an MCP-visible payload field, not a CLI line (ADR
# `receipt-dict-strings-are-mcp-boundary-so-recovery-splits-per-renderer`). The
# recovery clause is each renderer's own; `cli._coherence_audit_hint` composes the
# CLI's. Enforced by test rather than by emphasis (tests/test_record_decision.py).
_COHERENCE_AUDIT_NOTE = (
    "Coherence audit is cumulative and corpus-wide: this corpus holds recorded "
    "decisions that no contradiction check has covered yet."
)


def _audit_debt_line(uncovered: int, total: int, excluded: int) -> str:
    """Composes the `created` receipt's ``coherence_audit`` from the three counts.

    The one composer, so the MCP payload, ``record --json`` and the CLI text tail
    read the same string off the receipt. Its keyword names are the ``audit_debt``
    key names, so ``_audit_debt_line(**receipt["audit_debt"])`` rebuilds the field
    from the payload's own dict. The verb agrees with ``uncovered`` and the noun
    with ``total``. Total over non-negative counts: 0 uncovered is true when a
    check lands between the commit and this read, and it is rendered, not clamped.
    The register rules above apply, plus two more words it never uses: "full" (a
    check row cannot tell a scoped sweep from a whole one) and "since" (after an
    upgrade the count includes decisions older than any check).

    Args:
        uncovered: Active decisions no completed check has covered (N).
        total: The corpus's active decisions (M).
        excluded: Active decisions a check saw but could not audit (K).

    Returns:
        One sentence ending in a period, with the corpus as its referent.
    """
    noun = "decision" if total == 1 else "decisions"
    verb = "has" if uncovered == 1 else "have"
    line = (f"{uncovered} of this corpus's {total} {noun} {verb} not been covered "
            f"by a completed contradiction check")
    if excluded > 0:
        line += f"; {excluded} more could not be audited (not embedded)"
    return line + "."


# The `rotation` field of a `created` receipt: what this write's one bounded rotation
# did. Runtime-only and never persisted; it crosses the JSON boundary on MCP
# `record_decision` and `mitos record --json`, so it holds lists and dicts only. The
# vocabulary lives here so the renderers and the tests import it rather than retype it.
ROTATION_ROTATED = "rotated"
ROTATION_SKIPPED = "skipped"
ROTATION_FAILED = "failed"

ROTATION_STAGE_LOCK = "lock"
ROTATION_STAGE_GRAPH = "graph"
ROTATION_STAGE_FILE = "file"
ROTATION_STAGE_UNEXPECTED = "unexpected"

ROTATION_REASON_DUPLICATED = "duplicated"
ROTATION_REASON_OVERLAPPING = "overlapping"

# A fact, not a recovery: like every receipt string it reaches MCP, so it names no
# command. Each renderer composes its own recovery clause beside it.
_ROTATION_FAILED_NOTE = (
    "The decision was recorded. Rotation left decisions.md unchanged; an archive may "
    "already hold a copy of the entries it was moving."
)


# The decision outbox's drain (`drain_pending_embeddings`): why a walk stopped and what
# it did with the seed marker it read. Two closed vocabularies, named once here; the
# tests import them rather than retype them.
DRAIN_STOP_OUTAGE = "outage"            # auth/quota/transient on the embed, or the store failed the write
DRAIN_STOP_REFUSED = "refused"          # absent collection and no covering intent
DRAIN_STOP_UNAVAILABLE = "unavailable"  # no provider or no vector store; nothing read
DRAIN_STOP_CLAIM_FAILED = "claim_failed"  # the outbox read itself failed
DRAIN_STOPS = (DRAIN_STOP_OUTAGE, DRAIN_STOP_REFUSED, DRAIN_STOP_UNAVAILABLE,
               DRAIN_STOP_CLAIM_FAILED)

DRAIN_MARKER_SPENT = "spent"
DRAIN_MARKER_KEPT = "kept"
DRAIN_MARKERS = (DRAIN_MARKER_SPENT, DRAIN_MARKER_KEPT)

# The provider causes that are outages, not properties of a row: the drain stops at the
# first one instead of charging it to every row behind it (ADR
# `drain-outage-is-a-provider-cause-auth-quota-or-transient-or-an-unreachable-store`).
_DRAIN_OUTAGE_CAUSES = frozenset({AUTH, QUOTA, TRANSIENT})

# Rows per outbox read. The walk is keyset-paged, so this sizes each read, not the walk.
_DRAIN_CLAIM_SIZE = 10

# Node-less points per delete request in `reconcile`'s collection — the scroll's page
# size. Read at call time, so a row can force several chunks.
_ORPHAN_DELETE_CHUNK = 256


@dataclass(frozen=True)
class DrainOutcome:
    """What one drain of the decision outbox did. Runtime-only, never persisted.

    Attributes:
        written: Rows embedded and upserted, then removed.
        skipped_present: Rows removed without embedding, because the seed marker stood
            and the node's point was already in the collection.
        pruned: Rows removed because their node is no longer in the graph.
        failed: Rows attempted and left queued with their retry count moved, the
            outage row included.
        stop: None when the walk reached the end of the backlog, else one of
            :data:`DRAIN_STOPS`.
        marker: None when no seed marker stood at entry (or nothing was read), else
            one of :data:`DRAIN_MARKERS`.
    """

    written: int
    skipped_present: int
    pruned: int
    failed: int
    stop: Optional[str]
    marker: Optional[str]


# The rejection backlog's drain (`drain_rejection_backlog`) reuses the decision drain's
# stops and markers and adds one stop of its own. `DRAIN_STOPS` stays the decision
# drain's closed vocabulary: that drain never stops `held`.
REJECTION_DRAIN_STOP_HELD = "held"  # another drain holds the backlog's lock; nothing read
REJECTION_DRAIN_STOPS = DRAIN_STOPS + (REJECTION_DRAIN_STOP_HELD,)

# What `reconcile`'s collection of the rejection index did (`rejection_index.collected`),
# a closed vocabulary beside the drain's stops; the tests import it rather than retype it.
REJECTION_COLLECT_JUDGED = "judged"            # the listing was judged (an empty one included)
REJECTION_COLLECT_HELD = "held"                # a drain held the backlog's lock; nothing deleted
REJECTION_COLLECT_UNBUILT = "unbuilt"          # the graph holds no node; nothing judged
REJECTION_COLLECT_UNREAD = "unread"            # the graph read under the lock failed
REJECTION_COLLECT_UNAVAILABLE = "unavailable"  # no provider or index, a fault, or no lock
REJECTION_COLLECT_OUTCOMES = (
    REJECTION_COLLECT_JUDGED, REJECTION_COLLECT_HELD, REJECTION_COLLECT_UNBUILT,
    REJECTION_COLLECT_UNREAD, REJECTION_COLLECT_UNAVAILABLE,
)

# Rows per rejection backlog read: the owners one page's cross-owner batches are cut
# from. Read at call time, so a row can force several pages.
_REJECTION_DRAIN_CLAIM_SIZE = 100


def rejection_drain_lock_path(config: MitosConfig) -> str:
    """Returns the path of the rejection drain's operating-system lock.

    It is never the ``decisions.md`` lock, which is held only briefly around the
    buffer; this one is held for a whole drain, embedding calls included.

    Args:
        config: The workspace config.

    Returns:
        ``<workspace>/.mitos/rejection_drain.lock``.
    """
    return os.path.join(config.mitos_dir, "rejection_drain.lock")


@dataclass(frozen=True)
class RejectionDrainOutcome:
    """What one drain of the rejection backlog did. Runtime-only, never persisted.

    Attributes:
        owners_cleared: Rows cleared: their owner's items replaced whole, or their
            node gone from the graph.
        items_embedded: Items embedded and written.
        items_skipped_present: Items the owner's field states that the index already
            held, so nothing was embedded for them.
        items_removed: Items deleted because their owner's field no longer states
            them.
        owners_failed: Owners taken and left queued for a reason of their own.
        owners_past_tolerance: Rows standing at exit whose retry count is at least
            ``REJECTION_BACKLOG_FAILURE_TOLERANCE``; None when not read.
        calls: ``embed_documents`` calls attempted.
        left_queued: Rows standing at exit, read; None when not read.
        stop: None when the walk reached the end of the backlog, else one of
            :data:`REJECTION_DRAIN_STOPS`.
        marker: None when no seed marker stood at entry (or nothing was read), else
            one of :data:`DRAIN_MARKERS`.
    """

    owners_cleared: int
    items_embedded: int
    items_skipped_present: int
    items_removed: int
    owners_failed: int
    owners_past_tolerance: Optional[int]
    calls: int
    left_queued: Optional[int]
    stop: Optional[str]
    marker: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Returns the outcome as ``reconcile --json``'s ``rejection_drain`` object."""
        return asdict(self)


def rejection_drain_report(drain: Dict[str, Any]) -> List[str]:
    """Words a rejection drain's summary from its ``to_dict()``; names no command.

    The one renderer of the summary: the drain prints it on ``sync``, and
    ``cmd_reconcile`` prints it from ``rejection_drain``, so neither prints it twice.

    Args:
        drain: A :meth:`RejectionDrainOutcome.to_dict` value.

    Returns:
        The lines to print; empty when the drain did nothing worth a line.
    """
    lines: List[str] = []
    left = drain["left_queued"]
    worked = any(drain[key] for key in (
        "owners_cleared", "items_embedded", "items_skipped_present", "items_removed",
        "owners_failed",
    ))
    if worked:
        line = (
            f"Rejection index: {drain['owners_cleared']} owner(s) replaced — "
            f"{drain['items_embedded']} item(s) embedded, "
            f"{drain['items_skipped_present']} already present, "
            f"{drain['items_removed']} removed"
        )
        if drain["owners_failed"]:
            line += f"; {drain['owners_failed']} owner(s) failed"
        if left is not None:
            line += f"; {left} row(s) left queued"
        lines.append(line + ".")
    elif drain["stop"] == DRAIN_STOP_UNAVAILABLE and left:
        lines.append(
            f"Rejection index: not drained this run; {left} backlog row(s) stay queued."
        )
    past = drain["owners_past_tolerance"]
    if past:
        lines.append(
            f"Rejection index: {past} backlog row(s) have failed "
            f"{REJECTION_BACKLOG_FAILURE_TOLERANCE} or more times. They stay queued "
            f"and every drain attempts them again."
        )
    return lines


def _rotation_failure_stage(exc: BaseException) -> str:
    """Classifies where a record-path rotation failed, by exception type.

    ``filelock.Timeout`` subclasses ``TimeoutError`` and so ``OSError``: it is tested
    first, or a busy lock would read as a file fault.
    """
    if isinstance(exc, Timeout):
        return ROTATION_STAGE_LOCK
    if isinstance(exc, (sqlite3.Error, DatabaseError)):
        return ROTATION_STAGE_GRAPH
    if isinstance(exc, (OSError, UnicodeError)):
        return ROTATION_STAGE_FILE
    return ROTATION_STAGE_UNEXPECTED


def _rotation_failure(exc: Exception) -> Dict[str, Any]:
    """Builds the receipt's ``rotation`` field for a rotation that raised."""
    return {
        "outcome": ROTATION_FAILED,
        "stage": _rotation_failure_stage(exc),
        "error": f"{type(exc).__name__}: {exc}",
        "note": _ROTATION_FAILED_NOTE,
    }


def _rotation_report(outcome: "rotation.RotationOutcome") -> Optional[Dict[str, Any]]:
    """Builds the receipt's ``rotation`` field from an outcome, or ``None`` for silence.

    Archives are counted by basename exactly as sync's ``_report_rotation`` counts
    them. A block left unmatched is silent here as it is on sync, so an outcome that
    moved nothing and skipped nothing reports nothing.
    """
    skipped: List[Dict[str, Any]] = [
        {"slug": label, "reason": ROTATION_REASON_DUPLICATED, "occurrences": count}
        for label, count in outcome.duplicated
    ]
    skipped += [
        {"slug": label, "reason": ROTATION_REASON_OVERLAPPING}
        for label in outcome.overlapping
    ]
    if outcome.rotated:
        archives = []
        for archive_path in outcome.archive_paths:
            name = os.path.basename(archive_path)
            slugs = [
                block.label for block in outcome.rotated
                if os.path.basename(block.archive_name) == name
            ]
            archives.append({"path": archive_path, "entries": len(slugs), "slugs": slugs})
        report: Dict[str, Any] = {"outcome": ROTATION_ROTATED, "archives": archives}
        if skipped:
            report["skipped"] = skipped
        return report
    if skipped:
        return {"outcome": ROTATION_SKIPPED, "archives": [], "skipped": skipped}
    return None

# A new decision at/above this similarity to an existing live one the author did NOT
# reference is paused for review (AX P4): the neighbour must surface BEFORE commit, while
# the author can still point an amends/supersedes/contradicts at it (you can't relink
# after commit — re-record is a no-op; the retired post-commit `related` echo was exactly
# one step too late). An unlinked near-duplicate must force an explicit confirm-or-link,
# because prose-only obsoletion is invisible to state-filtered retrieval — only edges
# retire a node (ADR `record-pause-floor-lowered-to-strong-match-band`; the routine-case
# pause tax stays low via --acknowledge-neighbors, the declared-target exemption, and
# transitive lineage suppression).
# It is a document-space floor: the gather embeds the draft as a document and compares it
# with documents. The confidence band rates a query against documents and shares no scale
# with it, so this is not the band's strong line. Recall first: it stays where it is while
# every measured duplicate sits at or above it, and never rises. The measured placement,
# its rule and its runs: tests/golden/calibration/record.json.
_NEIGHBOR_REVIEW_THRESHOLD = 0.80

# A retired decision the draft sits at or above this document-space similarity to is a
# tombstone on the pause: the draft may re-tread ground the corpus already tried and
# abandoned (vision §4.9). It separates a re-tread of a retired decision from proximity to
# a retired family. Its own constant, never an alias of the live floor above: the two share
# a space and a population of vectors but not a class boundary, so either may move without
# the other (§6.3). Recall first: at or below every measured re-tread, never above the live
# floor's start, and raised only by its cry-wolf ceiling (a ratio of 2.0 on the copy's
# replay), which it is. Rule, runs and ceiling: tests/golden/calibration/record.json.
_TOMBSTONE_FLOOR = 0.8265

# The relations the pause offers for resolving a flagged neighbour — a fourth
# edge-field set beside store.py's kill/deferred/mutation trio, derivable from none
# of them (those are graph semantics; this is authoring guidance). Every surface
# that teaches the recovery renders from here: sync's needs_review message and the
# CLI's pause render join it directly, and the MCP record_decision docstring is
# pinned to it by test (a docstring can't render dynamically — @mcp.tool captures
# it at decoration). Three hand-kept copies let `narrows` go missing across three
# review rounds; one named set is the fix.
#
# Omissions are decisions, not gaps:
#   resolves     — decision→open_question only, and the pause gather keeps only
#                  live decisions, so a paused neighbour is structurally never an
#                  open question.
#   derives_from — originates from an open question; invalid on record and
#                  early-rejected before the pause can matter.
#   depends_on   — a ≥0.80 near-restatement is not a dependency; the "builds on"
#                  reading of a near-twin is what `cites` is for.
# corrects is included deliberately: it suppresses like the rest, and steering an
# author whose near-twin was WRONG (not outgrown) to `supersedes` stamps the
# target superseded_by instead of corrected_by on every future read — the same
# durable mislabel that let the missing `narrows` coerce false `amends` edges.
_PAUSE_RESOLVING_RELATIONS = (
    "amends", "narrows", "supersedes", "corrects", "contradicts", "cites",
)

# How many declared targets each pause-echo group renders before it collapses to a
# slice plus a sibling count (A2). Applied per group, independently.
#
# The bound's FIRST job is safety, not readability. The count is an exception marker,
# so a bound low enough to fire on an ordinary declaration would reconstruct the
# aggregate claim the echo is designed NOT to make ("N of your edges resolved a
# neighbour") on exactly the path meant to stay quiet. Set it where a collapse is
# pathological: the largest real declaration measured in this corpus is four, an
# ordinary six-slug declaration must not collapse, and forty repeated `--cites` still
# does. `routing.REGISTERED_NAMES_BOUND = 10` and `recall.SURFACE_TOP_SCOPES = 5`
# transfer their SHAPE (count beside slice, shown only on collapse) and deliberately
# not their magnitude — both are small because for those surfaces a collapse is the
# designed common case, which is the opposite of this one.
_DECLARED_ECHO_BOUND = 20

# Surface wording for the record receipt's degraded-check causes. The record surface
# owns this text (the core returns only the typed reason — the same core/surface split
# as _notice_conflict_unavailable, the sync-surface sibling). ``Unavailable.detail`` is
# logging-only by contract and never lands in the sentence.
_REVIEW_UNAVAILABLE_CAUSES = {
    ConflictUnavailableReason.EMBEDDING: "embedding service unavailable",
    ConflictUnavailableReason.VECTOR_STORE: "vector store unavailable",
    ConflictUnavailableReason.COLLECTION_MISSING: (
        "the vector collection is missing — run `mitos reconcile`"
    ),
}

# The sync sensor's per-entry line when the conflict check's gather was cut above its
# floor (``ConflictCheckResult.gather_cut``). Printed on stdout beside the finding
# block; it names no command and trips nothing.
_CONFLICT_GATHER_CUT_LINE = (
    "[Conflict] The neighbour search for this decision read its whole bound and held "
    f"fewer than {CONFLICT_OVERFETCH_LIMIT} live decisions while its last point still "
    "scored at or above the floor; a conflicting decision may lie past what it read."
)

# What the reconcile pre-flight (``_uncommittable_edges``) does about each store
# failure code — the whole class, declared, so none can be silently missed.
#
# Phase 3 shipped the pre-flight against `missing_target` alone and called it done;
# `kind_constraint_violation` then sailed through and wrote two audit rows per sync
# forever. The lesson was not "add one more check" but "the class must be enumerated
# somewhere a test can read." ``test_preflight_covers_store_codes`` asserts this table
# has an entry for every member of ``STORE_FAILURE_CODES``, so adding a code to the
# store forces a deliberate decision here rather than a silent gap.
#
# A code is CAUGHT (refused before the intent row) or NOT_CAUGHT with a reason.
# "Not caught" is a real position, not a TODO: the commit still rejects it correctly
# and the correlated outcome row still records the failure — the pre-flight's only job
# is to stop an UNBOUNDED trail behind a defect that repeats every sync.
_PREFLIGHT_CAUGHT = "caught"
_PREFLIGHT_DISPOSITIONS: Dict[str, str] = {
    STORE_MISSING_TARGET: _PREFLIGHT_CAUGHT,
    STORE_DANGLING_EDGE: _PREFLIGHT_CAUGHT,  # same probe: the citation resolves to nothing
    STORE_KIND_CONSTRAINT_VIOLATION: _PREFLIGHT_CAUGHT,
    STORE_SLUG_COLLISION: (
        "unreachable from a reconcile — the branch is entered only on a canonical-core "
        "hash MATCH against an existing node, so the committing node already owns the "
        "slug and cannot collide with itself"
    ),
    STORE_CYCLE_VIOLATION: (
        "not pre-flighted deliberately — deciding it needs the store's mutation-lineage "
        "walk, and re-implementing that in a pre-flight is exactly the duplication that "
        "drifts. It is also self-limiting rather than unbounded in practice: it needs a "
        "hand-edit that adds a kill/mutation edge closing a cycle, and the printed "
        "per-entry failure names it. Revisit if one is ever observed repeating."
    ),
}


def _ambiguous_target(ids: Sequence[str]) -> bool:
    """Whether a relation target's resolved ids name more than one decision.

    The one predicate behind every ambiguity refusal ``record_decision`` makes
    (``supersedes_ambiguous``, ``corrects_ambiguous``, ``relation_target_ambiguous``)
    and behind the pause's ``successor_ambiguous`` fact, so the pause can never call a
    successor relatable that the validator would refuse. The ids come from
    ``GraphStore.resolve_slug``, which is casefold-exact and state-agnostic: a
    retired decision that shares its slug with its live successor makes that slug
    ambiguous for every relation.

    Args:
        ids: ``resolve_slug``'s answer for the target.

    Returns:
        True when the target names two or more nodes.
    """
    return len(ids) > 1


def _live_successors(handle: Mapping[str, Any]) -> List[str]:
    """Where a retired handle's current word lives: the one successor rule (7a, 7b).

    ``live_end`` when the handle holds one (an immediate successor is itself retired;
    ``[]`` when the chain reaches no live decision), else ``superseded_by`` ∪
    ``corrected_by`` in order, deduplicated — the live-end rule guarantees those are
    live then. The pause's tombstone loop, the retired ``exists`` receipt and the CLI's
    retired lines all read it. A ``null`` key (a faulted ``exists`` handle read)
    contributes nothing, so a caller that must tell a fault from "no live successor"
    tests the keys itself.

    ``recall._successor_slugs`` is a deliberate twin on the read path: it answers
    ``None`` for "no successor key read" (lock 3 there) and does not deduplicate, and
    ``recall.py`` sits behind the check family's fence, so the two are not merged.

    Args:
        handle: A :func:`~mitos.display.retired_handle` dict, or a receipt carrying its
            successor keys.

    Returns:
        The live successor slugs, possibly empty.
    """
    if handle.get("live_end") is not None:
        return list(handle["live_end"])
    return list(dict.fromkeys(list(handle.get("superseded_by") or [])
                              + list(handle.get("corrected_by") or [])))


def _quoted_slugs(slugs: Sequence[str]) -> str:
    """Backticks each slug and joins them, for a receipt sentence."""
    return ", ".join(f"`{s}`" for s in slugs)


def _retired_exists_note(slug: str, node_id: str, state: str,
                         keys: Mapping[str, Any]) -> str:
    """The ``no_op_reason`` of an ``exists`` receipt whose node is retired (7b, §4.9).

    MCP-worded, like :data:`_EXISTS_NO_OP_NOTE` (the dict reaches MCP and ``record
    --json`` verbatim): it names tools and argument names, never a shell command. It
    opens with :data:`EXISTS_NO_OP_FACT`, then says the decision is retired and where
    its current word lives, then how its commentary is still repaired — by this node's
    **id**, because ``amend_commentary`` resolves a handle as ``show_node`` does and a
    slug a live decision also carries names that live decision — then keeps the live
    note's state-blind relations and archived sentences, and closes with exactly one
    of three forms, from the keys alone:

    * **relatable** (a live successor, none ambiguous): record a new decision relating
      to the live successor, a deliberate reversal being a ``supersedes`` of it; never
      this node's slug, a relation the commit refuses because a relation resolves only
      to a live target (P3);
    * **ambiguous** (``successor_ambiguous``): the successor's slug names two
      decisions, so ``record_decision`` refuses a relation to it; no relation is named.
      When the successor's slug is this node's own (a same-slug supersession), the note
      says the slug reads the live decision and the id reads this one;
    * **no live end, or a fault**: no relation is named.

    Args:
        slug: The retired node's stored slug.
        node_id: Its id.
        state: Its computed state (``superseded`` / ``corrected``).
        keys: The receipt's handle keys (:data:`_EXISTS_HANDLE_KEYS`; all ``None`` on
            a faulted read).

    Returns:
        The note.
    """
    faulted = keys.get("superseded_by", []) is None
    successors = [] if faulted else _live_successors(keys)
    if faulted:
        where = ("its successors could not be read; `show_node` with this decision's id "
                 "reads them.")
    elif successors:
        where = f"its current word lives in {_quoted_slugs(successors)}."
    else:
        where = "its chain of successors reaches no live decision."
    parts = [
        EXISTS_NO_OP_FACT,
        f"This decision is {state}: {where}",
        f"To change its commentary {_EXISTS_COMMENTARY_FIELDS}, call the "
        "`amend_commentary` tool on this same project and pass this decision's id, "
        f"{node_id}, as `slug`: the tool reads a handle the way `show_node` does, and a "
        "slug that a live decision also carries names the live one. It edits the entry "
        "in place and keeps the id.",
        _EXISTS_RELATIONS_AND_ARCHIVED,
    ]
    if not faulted and successors:
        if keys.get("successor_ambiguous"):
            # A fan-out (out-of-band only) may hold one ambiguous successor among
            # several, and the key does not say which: claim no more than that.
            parts.append(
                f"`record_decision` refuses a relation to {_quoted_slugs(successors)}: "
                "two decisions carry that slug." if len(successors) == 1 else
                "`record_decision` refuses a relation to a slug that two decisions "
                f"carry, and at least one of {_quoted_slugs(successors)} is one.")
            if slug.casefold() in {s.casefold() for s in successors}:
                parts.append(
                    f"The slug `{slug}` reads the live decision through `show_node`; "
                    "this decision's id reads this one.")
        else:
            parts.append(
                "To record a changed decision, record a new one with a relation to "
                f"{_quoted_slugs(successors)}, where a deliberate reversal is a "
                "supersedes of it, and never to this decision's slug: a relation can "
                "name only a live decision.")
    return " ".join(parts)


class DeclaredReview(NamedTuple):
    """What a draft's declarations exempt on the pause, by slug and by node id (7a).

    One lineage walk fills the first three fields
    (:meth:`MitosSyncManager._review_declared`); its resolution loop fills the fourth.

    Attributes:
        slugs: Casefolded declared targets of all nine relations, plus the casefolded
            slugs of the transitive mutation lineage of the ``supersedes`` /
            ``amends`` / ``narrows`` targets. The live screen's suppression set.
        ids_by_target: Each declared target's casefolded slug → its node id, plus, for
            a mutation target, every lineage ancestor's id. A target that does not
            resolve to exactly one node contributes nothing (Phase A refuses it before
            the pause; a replay shell may meet one).
        ids: The union of ``ids_by_target``'s values.
        direct_ids: Each declared target's casefolded slug → its own node id, with no
            lineage (7c). The sibling note matches it: a neighbour descends from a
            target only when its lineage holds that target's own id, never one of the
            target's ancestors (``ids_by_target`` would claim a cousin descends).
    """

    slugs: Set[str]
    ids_by_target: Dict[str, Set[str]]
    ids: Set[str]
    direct_ids: Dict[str, str]


class TombstoneReview(NamedTuple):
    """The pause's tombstone partition (:meth:`MitosSyncManager._review_tombstones`).

    Attributes:
        shown: The tombstone entries, closest first, at most the bound.
        omitted: Non-exempt retired hits at or above the floor that were not shown.
        cleared_by: The casefolded declared targets that exempted at least one
            retired hit at or above the floor (the declared echo reads it).
    """

    shown: List[Dict[str, Any]]
    omitted: int
    cleared_by: Set[str]


class AcknowledgedReview(NamedTuple):
    """What an acknowledged write went past (:meth:`MitosSyncManager._acknowledged_fields`).

    Each field is its own lock-3 statement about a write that has landed (7b, D4): a
    list when read, ``[]`` when read and empty, ``None`` when it could not be read.

    Attributes:
        neighbors: The ``acknowledged_neighbors`` slugs, most similar first.
        tombstones: The ``acknowledged_tombstones`` retired handles, closest first,
            uncut.
    """

    neighbors: Optional[List[str]]
    tombstones: Optional[List[Dict[str, Any]]]


class PauseReview(NamedTuple):
    """What the pre-commit near-duplicate review found (:meth:`MitosSyncManager._review_pause`).

    Attributes:
        pause: The ``needs_review`` dict to return (nothing written), or None when no
            partition holds anything (or the review could not run).
        unavailable: The fail-open notice when the review could not run, else None.
        cut: The cut sentence when the review ran over less than it should have and no
            pause carries it, else None (the created receipt carries it).
    """

    pause: Optional[Dict[str, Any]]
    unavailable: Optional[str]
    cut: Optional[str]


def _review_unavailable_notice(cause: str) -> str:
    """One calm receipt sentence for a near-dup check that could not run (fail-open).

    Structure: name the cause and state that the decision committed without the
    check. "Couldn't check" must never read as "checked, clean".

    It names **no command**, and that is the boundary rule rather than a stylistic
    trim: this sentence is returned into ``result["neighbor_review_unavailable"]``,
    which ``record --json`` emits verbatim and MCP ``record_decision`` returns — so
    a selectored ``mitos check`` here would put a shell command carrying a CLI flag
    onto an agent's response. The recovery is composed per renderer instead, and on
    the CLI it rides the *unconditional* coherence line one paragraph below (which
    lands on this same ``created`` exit by construction), so the receipt names the
    verb exactly once rather than twice in two registers. ADRs
    ``receipt-dict-strings-are-mcp-boundary-so-recovery-splits-per-renderer`` and
    ``created-receipt-names-its-recovery-once-on-the-unconditional-line``.

    One cause spelling is the deliberate exception: ``COLLECTION_MISSING`` keeps its
    ``mitos reconcile`` pointer in :data:`_REVIEW_UNAVAILABLE_CAUSES`, a separate
    string this function does not compose — a local re-embed rather than a judged
    audit, and test-pinned because the unmapped fallback cannot produce it.

    Args:
        cause: Short prose naming what failed (e.g. "embedding service unavailable").

    Returns:
        The receipt notice sentence.
    """
    return (
        f"Near-duplicate review could not run ({cause}); this decision committed "
        "without a neighbour check."
    )


def _floor_text(floor: float) -> str:
    """Renders a floor exactly: two places when that is the whole value (``0.80``), else
    every place it holds (``0.8265``, ``0.82649``: the shortest spelling that reads back
    as the value). A floor rounded up would state a bound the partition does not keep
    (a tombstone at 0.827 under "≥0.83")."""
    two = f"{floor:.2f}"
    return two if float(two) == floor else repr(float(floor))


def _past_what(score: float, floor: float, tombstone_floor: Optional[float]) -> str:
    """Names what may lie past a gather whose last score is ``score``: a live neighbour
    when it sits at or above the live floor, a tombstone when at or above the tombstone
    floor (scores fall down the fetch, so nothing past it scores higher)."""
    parts = []
    if score >= floor:
        parts.append("a live neighbour")
    if tombstone_floor is not None and score >= tombstone_floor:
        parts.append("a tombstone")
    return " or ".join(parts)


def _review_cut_notice(gathering: Optional[Gathering], floor: float, *,
                       tombstone_floor: Optional[float] = None) -> Optional[str]:
    """One calm sentence for a near-dup review whose gather was cut above its floor.

    The sibling of :func:`_review_unavailable_notice`: that one says the review could
    not run, this one says it ran over less than it should have. The gather read its
    whole raw bound, held fewer live decisions than it fills, and its last point still
    scored at or above the lower of the pause's floors — so a live neighbour, or a
    tombstone when the score also clears the tombstone floor, may lie past what it read
    (7a D7: a cut above a lower tombstone floor can hide a tombstone too). It names **no
    command**, for the same boundary reason its sibling documents (this string is a
    receipt dict value, so it reaches an MCP caller verbatim), and with no command in it
    there is nothing to word per surface.

    Args:
        gathering: The review's healthy gathering, or None when the gather never ran
            healthy.
        floor: The live floor (the pause's ``_NEIGHBOR_REVIEW_THRESHOLD``).
        tombstone_floor: The tombstone floor (``_TOMBSTONE_FLOOR``) when the review's
            consumer reads tombstones; the cut is judged at the lower of the two.

    Returns:
        The sentence when the gathering was cut above the lower floor; otherwise None.
    """
    judged = floor if tombstone_floor is None else min(floor, tombstone_floor)
    if gathering is None or not gathering.cut_above(judged):
        return None
    return (
        f"Near-duplicate review read its whole bound of {gathering.bound} points and "
        f"held {len(gathering.live)} live decision(s), fewer than the "
        f"{CONFLICT_OVERFETCH_LIMIT} it fills; its last point still scored "
        f"{gathering.last_score:.2f}, at or above the {_floor_text(judged)} floor, so "
        f"{_past_what(gathering.last_score, floor, tombstone_floor)} may lie past what it read."
    )


def _review_full_notice(gathering: Optional[Gathering], floor: float,
                        tombstone_floor: float) -> Optional[str]:
    """One calm sentence for a review whose live fill ended full of above-floor candidates (7a D7).

    The gather fills ``CONFLICT_OVERFETCH_LIMIT`` live candidates; when it holds all of
    them, the last still at or above the lower of the two floors, and the walk ended at
    an edge rather than running out, live neighbours and tombstones above their floors
    may lie past what it walked — so the pause's ``neighbors_omitted`` and
    ``tombstones_omitted`` count what it read, not what exists. The sentence names what
    filled (the gather's candidates, never what the pause shows, which is at most
    ``CONFLICT_TOP_K``) and no command; it is :func:`_review_cut_notice`'s sibling and
    never rides beside it (a cut gather holds fewer than the limit, a full one does not).

    Args:
        gathering: The review's healthy gathering, or None.
        floor: The live floor.
        tombstone_floor: The tombstone floor.

    Returns:
        The sentence when the review ended full; otherwise None.
    """
    if gathering is None or len(gathering.live) < CONFLICT_OVERFETCH_LIMIT \
            or gathering.end not in (EDGE_FOUND, EDGE_UNMEASURED):
        return None
    last = gathering.live[-1].score
    if last < min(floor, tombstone_floor):
        return None
    return (
        f"Near-duplicate review filled all {CONFLICT_OVERFETCH_LIMIT} of its live candidates "
        f"and the last still scored {last:.2f}, at or above the "
        f"{_floor_text(min(floor, tombstone_floor))} floor, so "
        f"{_past_what(last, floor, tombstone_floor)} may lie past them; the omitted counts "
        "cover what it read."
    )


def _declared_echo(
    declared_by_relation: Dict[str, Optional[str]],
    canonical_slugs: Dict[str, str],
    gathered_index: Dict[str, Tuple[str, float]],
    floor: float,
    cleared_by: AbstractSet[str] = frozenset(),
) -> Dict[str, Any]:
    """Partitions the caller's own declared targets for the pause body (A2).

    The predicate, stated once here so no reader re-derives it from three call sites.
    Let ``V`` be the targets the caller typed across all nine relation flags, ``G``
    the slugs this call's KNN sweep gathered, and ``PR``
    :data:`_PAUSE_RESOLVING_RELATIONS`:

    * **Group two** (``declared_no_near_match``) — targets in ``V`` declared through at
      least one ``PR`` relation that fall outside ``V ∩ G ∩ (score >= floor)`` and did
      not clear a tombstone on this call (``cleared_by``, 7a D9).
    * **Group one** (``declared``) — the complement over ``V``, keyed on all nine
      relations.

    The tombstone clause keeps group two's sentence true: a declaration whose id set
    meets a retired hit's ``{T.id} ∪ KillChain(T).reached`` at or above the tombstone
    floor did resolve part of the pause, so it is not reported as one that "does not
    resolve this pause". It adds no tombstone to the echo and no key; ``cleared_by`` is
    computed from the primary sets (the declarations and the exemption decision, M8).

    Four properties are contract, not implementation taste:

    * The partition is over **targets**, not flag occurrences — a target declared
      through several relations appears once, and lands in group two if *any* of those
      relations is pause-resolving. So an ordinary cross-domain ``depends_on`` is
      acknowledged in group one and can never be reported as having moved nothing.
    * The inputs are the **primary sets** — the declarations, the gathered candidates
      and the floor (M8). Never ``screen_candidates``' filtered return: its S4 stage
      drops a declared target *before* the floor, so a declaration that resolved a
      strong neighbour and one that did nothing are both absent from the survivors,
      for unrelated reasons.
    * Only the caller's own half of ``declared_targets`` is echoed. The transitive
      lineage ancestors merged in for suppression are in **neither** group — the
      discriminator is *did the caller type it*, never *is it also an ancestor*.
    * An empty group renders **no key at all**, never an empty list: group one is
      genuinely reachable-empty (a lone distant ``cites``), and a ``declared: []``
      beside a populated group two would deny a declaration the call did receive.

    Group two carries ``score`` only when the target *was* gathered and fell below the
    floor; a target the sweep never saw has no candidate to read one off and renders
    scoreless. The absent key — rather than a ``null`` — is what carries that shape.

    Args:
        declared_by_relation: Relation name -> the caller's raw, unsplit argument, in
            the fixed render order (``supersedes``, ``corrects``, then
            ``_EXTRA_RELATIONS``' own order). Values may be ``None``.
        canonical_slugs: Casefolded declared target -> its stored slug, retained by the
            Phase-A validation loops. The echo prints the stored spelling for every
            target, gathered or not (a caller spelling and the casefold are both wrong
            here — ``_normalize_slug`` runs on the record path alone, so the stored
            handle genuinely differs from both).
        gathered_index: Casefolded gathered slug -> ``(stored slug, score)`` for every
            candidate this call's sweep returned, pre-screen.
        floor: The similarity floor the screen applied (``_NEIGHBOR_REVIEW_THRESHOLD``).
        cleared_by: Casefolded declared targets that exempted a retired hit at or above
            the tombstone floor (``TombstoneReview.cleared_by``); never in group two.

    Returns:
        The echo's keys, ready to merge into the pause dict — any of ``declared``,
        ``declared_total``, ``declared_no_near_match``,
        ``declared_no_near_match_total``. Empty when the caller declared nothing.
    """
    order: List[Tuple[str, str]] = []       # (casefolded, verbatim), first-seen order
    seen: set = set()
    resolving: set = set()
    for relation, raw in declared_by_relation.items():
        for target in _split_relation_slugs(raw):
            folded = target.casefold()
            if folded not in seen:
                seen.add(folded)
                order.append((folded, target))
            if relation in _PAUSE_RESOLVING_RELATIONS:
                resolving.add(folded)

    group_one: List[str] = []
    group_two: List[Dict[str, Any]] = []
    for folded, verbatim in order:
        # The stored spelling, from the validation loops. Every declared target
        # resolved there before the pause could compose, so the fallback is
        # unreachable — it exists so a future edit cannot turn the echo into a crash
        # on the write path.
        handle = canonical_slugs.get(folded, verbatim)
        gathered = gathered_index.get(folded)
        if folded in resolving and folded not in cleared_by \
                and not (gathered is not None and gathered[1] >= floor):
            item: Dict[str, Any] = {"slug": handle}
            if gathered is not None:
                item["score"] = gathered[1]
            group_two.append(item)
        else:
            group_one.append(handle)

    echo: Dict[str, Any] = {}
    for key, items in (("declared", group_one),
                       ("declared_no_near_match", group_two)):
        if not items:
            continue
        if len(items) > _DECLARED_ECHO_BOUND:
            # The elision takes the TAIL: a score-keyed truncation is undefined over
            # group two's common member (the scoreless one) and, defaulted to zero,
            # would elide exactly the declarations carrying no other signal.
            echo[key] = items[:_DECLARED_ECHO_BOUND]
            echo[f"{key}_total"] = len(items)
        else:
            echo[key] = items
    return echo


def _declared_echo_lines(payload: Dict[str, Any]) -> List[str]:
    """The echo's prose, composed from the pause payload's own keys.

    Both prose renderers read this — sync's ``message`` and ``cmd_record``'s composed
    body — so what a human reads and what the two machine encodings carry cannot
    drift: the sentences are a render of the keys, not a second computation beside
    them. Returns ``[]`` when the payload carries no echo, which is the whole render
    for a call that declared nothing.

    Both lines report in the **declared** register and say nothing about whether an
    edge committed, in either direction. The pause returns above Phase B, so nothing
    was written; a first group reading as a receipt of landed edges would invite
    dropping exactly those flags from the re-record the pause exists to force. The
    count renders only on a collapse — rendered unconditionally it is the aggregate
    claim this surface is designed not to make.

    The second group says what a declaration there is — valid, and not the relation
    that resolves this pause — and states the floor its score sits below. A third
    sentence, rendered whenever either group rendered, asks for every declaration
    again on the re-record. It is one sentence for both groups on purpose: a clause
    on the second group alone would make the pair say the first group need not be
    sent again, which is the landed-edge reading the register floor forbids.

    Args:
        payload: The pause dict (or the echo keys alone).

    Returns:
        Zero to three sentences without trailing punctuation: group one, group two,
        then the re-send sentence.
    """
    lines: List[str] = []

    declared = payload.get("declared")
    if declared:
        line = "Declared: " + ", ".join(declared)
        total = payload.get("declared_total")
        if total is not None:
            line += f" ({total} total)"
        lines.append(line)

    no_match = payload.get("declared_no_near_match")
    if no_match:
        rendered = []
        for item in no_match:
            score = item.get("score")
            rendered.append(
                item["slug"] if score is None
                else f"{item['slug']} ({_floored_score_text(score)}, below the "
                     f"{_floor_text(_NEIGHBOR_REVIEW_THRESHOLD)} floor)"
            )
        line = "Declared, not a near neighbour here: " + ", ".join(rendered)
        total = payload.get("declared_no_near_match_total")
        if total is not None:
            line += f" ({total} total)"
        # Singular or plural by every member the line is about, elided ones included.
        members = total if total is not None else len(no_match)
        line += (" — a valid declaration that does not resolve this pause"
                 if members == 1
                 else " — valid declarations that do not resolve this pause")
        lines.append(line)

    if lines:
        lines.append("Pass every declaration again when you re-record")

    return lines


def _pause_message(payload: Dict[str, Any]) -> str:
    """The pause's ``message``, composed from the pause dict's own keys (7a D6).

    Every clause is gated on a key the payload carries, so the MCP message, the CLI
    text (which composes its own lines from the same keys) and ``--json`` cannot
    disagree. It is in the MCP register: it names ``acknowledge_neighbors=True`` and
    relation arguments by their bare names, never a shell command. A tombstone-only
    pause opens on its tombstones and never says it is similar to zero decisions.

    Args:
        payload: The pause dict, without its ``message``.

    Returns:
        The message.
    """
    slug = payload["slug"]
    neighbors = payload.get("neighbors") or []
    tombstones = payload.get("tombstones") or []
    parts: List[str] = []
    if neighbors:
        omitted = payload.get("neighbors_omitted")
        parts.append(
            f"Paused: '{slug}' is ≥{_floor_text(_NEIGHBOR_REVIEW_THRESHOLD)} similar to "
            f"{len(neighbors)} existing decision(s) you did not reference"
            + (f", and {omitted} more above the floor are not shown." if omitted else ".")
            + " Judge each neighbour from its axiom, rejected_paths, scope, and modifier "
            "stamps, then re-record with that judgment: a relation arg "
            f"({'/'.join(_PAUSE_RESOLVING_RELATIONS)}) pointing at any neighbour this "
            "decision genuinely relates to, acknowledge_neighbors=True for neighbours that "
            "stand independently alongside it — or both at once for a mixed set.")
        if any(n.get("amended_by") or n.get("narrowed_by") for n in neighbors):
            # B9's dereference sentence only where it can apply: a neighbour that
            # carries a stamp.
            parts.append("If you go on to supersede or amend a neighbour, an "
                         "amended_by/narrowed_by stamp on it means it has moved on — "
                         "dereference that slug before linking.")
        if any(n.get("descends_from_declared") for n in neighbors):
            parts.append("A neighbour marked descends_from_declared descends from a "
                         "decision this draft declares, named there; that declaration "
                         "does not clear it, so judge it like any other neighbour.")
    if tombstones:
        omitted = payload.get("tombstones_omitted")
        lead = f"Paused: '{slug}' is" if not neighbors else "It is also"
        parts.append(
            f"{lead} ≥{_floor_text(_TOMBSTONE_FLOOR)} similar to {len(tombstones)} decision(s) this "
            "corpus retired (tombstones): ground it already tried and abandoned, which the "
            "draft sits closer to than the floor"
            + (f"; {omitted} more tombstone(s) above the floor are not shown." if omitted else ".")
            + " Each names where its current word lives (live_end, else superseded_by or "
            "corrected_by). If the draft does not re-tread it, declare a relation to that "
            "live successor, which clears it (a deliberate reversal is a supersedes of the "
            "successor), or pass acknowledge_neighbors=True.")
        if any(t.get("successor_on_pause") for t in tombstones):
            parts.append("A tombstone marked successor_on_pause has that successor among "
                         "the neighbours above, so one relation to it clears both.")
        ambiguous = [t for t in tombstones if t.get("successor_ambiguous")]
        if any(len(_live_successors(t)) == 1 for t in ambiguous):
            parts.append("A tombstone marked successor_ambiguous clears only with "
                         "acknowledge_neighbors=True: its successor's slug names two "
                         "decisions, and record_decision refuses a relation target that two "
                         "decisions carry.")
        if any(len(_live_successors(t)) > 1 for t in ambiguous):
            # A fan-out (out-of-band only) may hold one ambiguous successor among
            # several, and the key does not say which: claim no more than that (7b's
            # `exists` rule).
            parts.append("A tombstone marked successor_ambiguous with several successors "
                         "has at least one whose slug two decisions carry, a relation target "
                         "record_decision refuses; acknowledge_neighbors=True clears it.")
    echo_prose = "".join(f"{line}. " for line in _declared_echo_lines(payload))
    return (" ".join(parts) + f" {PAUSE_HELD_CLAUSE}{_PAUSE_DIGEST_CLAUSE} "
            f"{echo_prose}Nothing was written.")


def _floored_score_text(score: float) -> str:
    """Renders a below-floor score to two places, rounded down.

    Rounding to nearest would print 0.7967 as ``0.80`` beside a stated 0.80 floor it
    is below. ``Decimal`` over the shortest repr floors exactly, where float
    arithmetic turns 0.29 into 0.28. Display only: the payload keeps the raw float.

    Args:
        score: The cosine score of a declared target that was gathered.

    Returns:
        The score floored to two decimal places, e.g. ``"0.79"``.
    """
    score = float(score)
    if not math.isfinite(score):
        return str(score)       # quantize raises on inf; no rounding to fix there
    return str(Decimal(str(score)).quantize(Decimal("0.01"), rounding=ROUND_FLOOR))


def _normalize_slug(text: str) -> str:
    """Lowercases and hyphenates free text into slug characters — with no length cap.

    The character-normalisation half of :func:`_slugify`, factored out so the write
    path can validate an explicit slug's *length* without silently truncating it: an
    over-length explicit slug is the handle the author will cite, so the right move is
    to reject (and ask for a shorter one), not to mangle it down to fit.

    Args:
        text: Free text — an explicit slug, or an axiom for auto-derivation.

    Returns:
        The lowercase-hyphenated form, stripped of leading/trailing hyphens (``""``
        for empty/whitespace-only input). Uncapped.
    """
    if not text:
        return ""
    s = re.sub(r'[^a-z0-9]+', '-', text.lower())
    return re.sub(r'-+', '-', s).strip('-')


def _slugify(text: str) -> str:
    """Derives a deterministic, lowercase-hyphenated slug from free text, capped in length.

    Determinism keeps the human-readable handle stable: the slug is NOT part of the
    node id (V1a identity is the slug-free canonical core — ``compute_node_id``), but
    a stable auto-derived slug means the same decision presents the same handle and
    the casefold slug-collision check (V1-D4) behaves predictably.

    This is the auto-derive path: when the slug exceeds the length cap it is trimmed
    back to the last word boundary (hyphen) rather than sliced mid-word, so the handle
    stays readable (``…brazilian-portuguese``, not ``…brazilian-portug``). Still a pure
    function of the text, so determinism holds. The agentic write path does NOT trim an
    *explicit* slug — it validates length via :func:`_normalize_slug` and rejects an
    over-length one (see ``slug_too_long``).
    """
    s = _normalize_slug(text)
    if len(s) > _SLUG_MAX_LEN:
        cut = s[:_SLUG_MAX_LEN]
        boundary = cut.rfind('-')
        # Trim to the last whole word, unless that would gut the slug (one very
        # long leading token) — then fall back to the hard cap.
        if boundary >= _SLUG_MIN_LEN:
            cut = cut[:boundary]
        s = cut.rstrip('-')
    return s


# The three outcomes that SATISFY a `--reconcile-entry` target. Every other
# end-state the per-entry loop can reach is a shortfall by default, and that
# direction is the decision rather than the spelling: the loop ends an entry in
# more states than any table names, and the two directions are not symmetric. A
# failure state left unstamped exits 0 on a repair that never landed — the silent
# no-op this whole flag exists to remove — while a satisfied state left unstamped
# exits non-zero, loudly, and is caught by the re-run row. Enumerating the closed
# set is also the only direction that survives a control-flow change in a later
# vision.
_REPAIR_SATISFIED = frozenset({"reconciled", "committed", "clean"})

# The located cause each shortfall outcome renders as. Keys are the outcome tokens
# `_perform_sync_internal` stamps at its own `continue`s; an entry that reached
# none of them (a site added later and not stamped) falls through to the generic
# line below, which is a shortfall too.
_REPAIR_CAUSES = {
    "open_question": (
        "this entry is an open question, and the commentary reconcile is "
        "decisions-only — a hand-edit to it can never be applied by `sync`"
    ),
    "source_only": (
        "its only divergence is the `**Source:**` line, which a reconcile "
        "provably cannot change — restore the line to match the graph"
    ),
    "reconcile_refused": (
        "the commentary reconcile refused it; its reason is printed above"
    ),
    "collision": (
        "its slug already names another node and the entry declares no relation "
        "at it, so it was skipped before any reconcile — the collision report "
        "above says what to author"
    ),
    "pending_skipped": (
        "it is still pending, and a run with no terminal and no `--yes` skips a "
        "pending entry before it can be committed. This flag authorizes the "
        "reconcile, not the accept prompt"
    ),
    "operator_skipped": "it was skipped at the accept prompt",
    "operator_quit": "the run was stopped at its accept prompt",
    "quarantined": (
        "its commit was refused and the retry never drained it; the report above "
        "names why"
    ),
    "store_error": (
        "an unexpected store error stopped its commit; the report above names it"
    ),
}

#: The two causes that stand in for the never-seen class's absence claim when the
#: run stopped reading the buffer. Neither is a new state: the target is still a
#: shortfall and the exit is still non-zero — what changes is that a run holding no
#: evidence of absence does not assert one, and does not name `mitos rebuild` as the
#: heal for an entry that may be sitting in the buffer it stopped reading.
_REPAIR_UNREAD_LOCKED = (
    "another Mitos process holds the corpus lock, so this run never read the "
    "buffer — re-run once it is free"
)
_REPAIR_UNREAD_QUIT = (
    "the run was stopped at its accept prompt before this entry was reached, so "
    "nothing was ruled on it — re-run to reach it"
)

_REPAIR_GENERIC_CAUSE = (
    "it did not end in any of the states this flag can satisfy; the run's own "
    "report above says what happened to it"
)


class _RepairLedger:
    """Observes what the sync loop did with each entry `--reconcile-entry` named.

    Constructed once per run and inert when no targets were named, so every `note`
    is a no-op on the ordinary path. It holds no policy of its own: the loop stamps
    what it did, and the ledger renders the shortfall afterwards. Deliberately
    module-private and un-generalized — nothing else consumes it, and there is no
    second consumer to extract a shared refusal renderer for.

    Attributes:
        _handles: casefold handle → the caller's verbatim spelling. Verbatim is
            what the report renders, because the caller has to recognise what they
            typed; casefold is what matches, because every other slug lookup in
            the tree folds on both sides.
    """

    def __init__(self, targets: Optional[List[str]] = None) -> None:
        self._handles: Dict[str, str] = {}
        for raw in targets or []:
            self._handles.setdefault(raw.casefold(), raw)
        self._outcomes: Dict[str, str] = {}
        self._order: List[str] = []
        self._failure_slugs: Set[str] = set()
        self._unattributable_failures = False
        self._unread_reason: Optional[str] = None
        self._inert = False

    def authorizes(self, entry: ParsedEntry) -> bool:
        """Whether this entry was named, casefold-exactly, by the caller.

        One tier, no alias/prefix/did-you-mean resolution — the discipline every
        other slug lookup uses. Neither side is ``_normalize_slug``'d: that helper
        runs on the ``record`` path alone, and the parser keeps a header slug
        verbatim.
        """
        slug = getattr(entry, "slug", None)
        return bool(slug) and slug.casefold() in self._handles

    def note(self, entry: ParsedEntry, outcome: str) -> None:
        """Records the loop's own verdict for a named entry. A no-op otherwise.

        Later notes win, deliberately: the quarantine stamps a shortfall at the
        commit and the fixpoint upgrades the entries it drains, so the pessimistic
        stamp is the one that survives a missed correction.
        """
        if not self.authorizes(entry):
            return
        key = entry.slug.casefold()
        if key not in self._outcomes:
            self._order.append(key)
        self._outcomes[key] = outcome

    def note_parse_failures(self, failures: List[EntryFailure]) -> None:
        """Records this run's parse failures, for the never-seen classification."""
        if not self._handles:
            return
        for fail in failures:
            if fail.slug:
                self._failure_slugs.add(fail.slug.casefold())
            else:
                # A pre-header failure carries no slug, so it can be reported but
                # not attributed to a handle.
                self._unattributable_failures = True

    def mark_buffer_unread(self, reason: str) -> None:
        """Records that the run did not read the whole buffer, and why.

        Two sites reach it: a lock held by another process (nothing was read at
        all) and ``[q]uit`` at an accept prompt (everything below that entry went
        unread). It changes no verdict and no exit — an unreached handle is still a
        shortfall — but it changes what the report may CLAIM about one. "absent
        from the buffer, or already rotated into an archive" is an absence claim
        whose heal is `mitos rebuild`, and a run that stopped reading has no
        evidence for either half.

        Args:
            reason: The located cause, rendered in place of the absence claim.
        """
        self._unread_reason = reason

    def mark_keyless(self) -> None:
        """Marks the run as below the key floor — the report's one carve-out.

        ``mitos sync``'s ``GEMINI_API_KEY`` refusal returns above the per-entry
        loop, so on a keyless workspace the flag is inert: nothing was reconciled
        and nothing was looked for. Fail-loud is scoped to runs that clear the
        floor, and flipping that exit code is a separate contract break owned by
        a pass that deprecates it.
        """
        self._inert = True

    def report(self) -> List[str]:
        """Prints one line per named target that needs one; returns the shortfall.

        The channel is stderr and is not overridable: the block accompanies a
        non-zero exit on its common path, and the tree's non-zero-exit refusals
        answer there. One channel for the whole report, exit-0 line included.

        Returns:
            The caller's verbatim spellings for every target that did NOT end in
            the state its markdown describes. Empty means nothing was named, the
            run was below the key floor, or every named target is satisfied.
        """
        if self._inert or not self._handles:
            return []

        # Document order — the loop's, which is `decisions.md`'s, never the order
        # the caller named them. Handles the loop never reached have no position of
        # their own, so they follow in naming order.
        ordered = self._order + [k for k in self._handles if k not in self._outcomes]

        lines: List[str] = []
        shortfall: List[str] = []
        for key in ordered:
            handle = self._handles[key]
            outcome = self._outcomes.get(key)
            if outcome in _REPAIR_SATISFIED:
                if outcome == "clean":
                    # The one satisfied state that owes a line: nothing else in the
                    # run said anything about this entry, so without it a caller
                    # reads silence as "not applied".
                    lines.append(
                        f"[Repair] {handle!r} — nothing to reconcile: the corpus "
                        f"and graph already agree for this entry."
                    )
                continue
            shortfall.append(handle)
            lines.append(f"[Repair] {handle!r} — {self._cause(key, outcome)}.")

        if lines:
            # stdout is block-buffered under a pipe while stderr is not, so without
            # this flush the block lands above the run's own report and inverts the
            # reading. One channel and one flush for the whole block, so it stays
            # together and accompanies the non-zero exit on its common path.
            sys.stdout.flush()
            for line in lines:
                print(line, file=sys.stderr)
        return shortfall

    def _cause(self, key: str, outcome: Optional[str]) -> str:
        """Renders the located cause for one unsatisfied handle."""
        if outcome is not None:
            return _REPAIR_CAUSES.get(outcome, _REPAIR_GENERIC_CAUSE)
        # The never-seen class — stated as a property, never as a two-member list.
        if key in self._failure_slugs:
            return ("the buffer holds this entry but it failed to parse, so the "
                    "loop never reached it — fix the parse error reported above")
        if self._unread_reason is not None:
            # The run stopped reading the buffer, so it holds no evidence for the
            # absence claim below and may not make it.
            return self._unread_reason
        line = ("no entry with this slug was reached this run: it is absent from "
                "the buffer, or already rotated into an archive (`sync` reads the "
                "buffer alone, so an archived entry's reconciler is `mitos rebuild`)")
        if self._unattributable_failures:
            line += (". The buffer also holds unparsed entries this run could not "
                     "attribute to a slug")
        return line


class _OwnerTake:
    """One owner, taken once by a rejection drain: the row read, then its plan.

    Attributes:
        row: The backlog row as claimed; its seq is the only one the take may clear
            or note.
        slug: The owner's slug, for its lines.
        expected: ``rejection_points`` of the field read after the claim.
        relist: Whether the owner's holdings must be listed afresh rather than taken
            from the entry listing (a retake, or a row queued after that listing).
        to_write: ``(point_id, text)`` of the expected items not held, in
            enumeration order.
        to_remove: Held point ids the field no longer states.
        landed: How many of ``to_write`` are written.
        aborted: Set once the take failed; it is then never finalized and noted once.
    """

    def __init__(self, row: RejectionBacklogRow, slug: str,
                 expected: Dict[str, Any], relist: bool) -> None:
        self.row = row
        self.slug = slug
        self.expected = expected
        self.relist = relist
        self.to_write: List[Tuple[str, str]] = []
        self.to_remove: Set[str] = set()
        self.landed = 0
        self.aborted = False

    @property
    def owner(self) -> str:
        return self.row.node_id

    def plan(self, held: AbstractSet[str]) -> int:
        """Splits the expected items against what the owner holds.

        Returns:
            How many expected items are already held.
        """
        self.to_write = [(pid, ref.text) for pid, ref in self.expected.items()
                         if pid not in held]
        self.to_remove = set(held) - set(self.expected)
        return len(self.expected) - len(self.to_write)


# One queued item of a page: its owner's take, its point id, its text.
_QueuedItem = Tuple[_OwnerTake, str, str]


class _RejectionDrainWalk:
    """One walk of the rejection backlog, run under the drain lock its caller holds.

    The steps, in order, are :meth:`MitosSyncManager.drain_rejection_backlog`'s
    contract. Every exit returns ``(stop, marker)``; the counters are read off the
    walk afterwards, also when an exception escaped it.
    """

    def __init__(self, manager: "MitosSyncManager") -> None:
        self.store = manager.store
        self.index = manager.rejection_index
        self.provider = manager.embed_provider
        self.selector = shell_flag("-p", manager.config.project)
        self.owners_cleared = 0
        self.items_embedded = 0
        self.items_skipped_present = 0
        self.items_removed = 0
        self.owners_failed = 0
        self.calls = 0
        self.model = ""  # the provider's model id, validated once at entry
        self.seed: Optional[Dict[str, str]] = None
        self.may_create = False
        self.collection_present = False
        # Under the marker: the entry listing, by owner, and the largest seq queued
        # before it was taken. A row above that seq re-lists, as a retake does.
        self.entry_holdings: Optional[Dict[str, Set[str]]] = None
        self.watermark = 0
        self.taken: Set[str] = set()

    def kept(self) -> Optional[str]:
        return DRAIN_MARKER_KEPT if self.seed is not None else None

    # -- the walk ---------------------------------------------------------- #

    def run(self) -> Tuple[Optional[str], Optional[str]]:
        # A provider naming no model is its own fault, not any row's: checked once,
        # before the claim, so it never reads as every owner's poison.
        model = self.provider.model_id
        if not isinstance(model, str) or not model:
            print(
                "[Warning] The rejection backlog was not drained: the embedding "
                "provider reported no model id. Its rows stay queued."
            )
            return DRAIN_STOP_UNAVAILABLE, None
        self.model = model
        # Declared intent, from the state a covering verb established (read once).
        self.seed = self.store.rejection_seed()
        self.may_create = self.seed is not None
        try:
            page = self.store.claim_rejection_backlog(_REJECTION_DRAIN_CLAIM_SIZE)
        except Exception as e:
            self._claim_failed(e)
            return DRAIN_STOP_CLAIM_FAILED, self.kept()
        if not page:
            # Nothing to replace: whatever the marker claimed is moot. Lists nothing.
            if self.seed is None:
                return None, None
            return None, self._spend()

        if self.seed is not None:
            stop = self._list_entry_holdings(page)
            if stop is not None:
                return stop, self.kept()

        while page:
            stop = self._walk_page(page)
            if stop is not None:
                return stop, self.kept()
            try:
                page = self.store.claim_rejection_backlog(
                    _REJECTION_DRAIN_CLAIM_SIZE, after_seq=page[-1].enqueue_seq
                )
            except Exception as e:
                self._claim_failed(e)
                return DRAIN_STOP_CLAIM_FAILED, self.kept()

        if self.seed is None:
            return None, None
        if self.collection_present:
            return None, self._spend()
        # Still absent (nothing written). Spend only if nothing is left to cover:
        # read, never inferred — a row queued behind the cursor is invisible to it.
        try:
            empty = self.store.claim_rejection_backlog(1) == []
        except Exception as e:
            self._claim_failed(e)
            return None, DRAIN_MARKER_KEPT
        return None, (self._spend() if empty else DRAIN_MARKER_KEPT)

    def _list_entry_holdings(self, page: List[RejectionBacklogRow]) -> Optional[str]:
        """Under the marker: one listing of the whole index, with owners."""
        seqs = [row.enqueue_seq for row in page]
        try:
            last = self.store.rejection_backlog_last_seq()
        except Exception:
            last = None  # the head page's seqs still bound what predates the listing
        self.watermark = max(seqs + ([last] if last is not None else []))
        try:
            owners = self.index.list_point_owners()
            self.collection_present = True
        except CollectionMissingError:
            owners = {}  # absent: this licensed drain creates it on its first write
        except VectorStoreError as e:
            self._stop_store(e, "could not list the rejection index")
            return DRAIN_STOP_OUTAGE
        holdings: Dict[str, Set[str]] = {}
        for point_id, owner in owners.items():
            if owner is not None:
                holdings.setdefault(owner, set()).add(point_id)
        self.entry_holdings = holdings
        return None

    def _walk_page(self, page: List[RejectionBacklogRow]) -> Optional[str]:
        # Seq first, field after (the claim above): a commit between the two mints
        # a larger seq, so this take's clear fails and the retake reads that field.
        try:
            fields = self.store.get_rejection_owner_fields(
                [row.node_id for row in page]
            )
        except Exception as e:
            # One read for the page, so one fault for the page: no row is charged
            # (the fault is the store's), and the next drain retries every one.
            print(
                f"[Warning] Could not read {len(page)} rejection backlog owner(s) "
                f"from the graph: {describe_embed_failure(e)}. Their rows stay queued."
            )
            return None

        takes: List[_OwnerTake] = []
        for row in page:
            retake = row.node_id in self.taken
            self.taken.add(row.node_id)
            if row.node_id not in fields:
                # The node left the graph (only with foreign keys off). Its points
                # are orphans for `reconcile`'s collection, not for this drain.
                self._clear(row, None)
                continue
            slug, field = fields[row.node_id]
            try:
                expected = rejection_points(row.node_id, field)
            except Exception as e:
                self._fail_row(row, slug, e)
                continue
            relist = (retake or self.entry_holdings is None
                      or row.enqueue_seq > self.watermark)
            takes.append(_OwnerTake(row, slug, expected, relist))

        # Holdings for every owner of the page, before anything of the page embeds.
        for take in takes:
            if take.relist:
                try:
                    held: AbstractSet[str] = self.index.list_owner_point_ids(take.owner)
                    self.collection_present = True
                except CollectionMissingError:
                    if self.seed is None:
                        self._stop_refused()
                        return DRAIN_STOP_REFUSED
                    held = set()  # licensed: the first write creates it
                except VectorStoreError as e:
                    self._stop_store(e, "could not list an owner's items")
                    return DRAIN_STOP_OUTAGE
            else:
                held = self.entry_holdings.get(take.owner, set())
            self.items_skipped_present += take.plan(held)

        for take in takes:
            if not take.to_write:
                stop = self._finalize(take)
                if stop is not None:
                    return stop

        queue: List[_QueuedItem] = [
            (take, pid, text) for take in takes for pid, text in take.to_write
        ]
        while queue:
            n = fit_batch([text for _, _, text in queue])
            batch, queue = queue[:n], queue[n:]
            stop, queue = self._send_batch(batch, queue)
            if stop is not None:
                return stop
        return None

    # -- batches ----------------------------------------------------------- #

    def _send_batch(
        self, batch: List[_QueuedItem], queue: List[_QueuedItem]
    ) -> Tuple[Optional[str], List[_QueuedItem]]:
        owners = list(dict.fromkeys(take for take, _, _ in batch))
        try:
            vectors = self._embed_batch([text for _, _, text in batch])
        except Exception as e:
            cause = classify_provider_error(e)
            if cause in _DRAIN_OUTAGE_CAUSES:
                self._stop_provider(cause, e)
                return DRAIN_STOP_OUTAGE, queue
            if len(owners) == 1:
                self._abort(owners[0], e)
                return None, self._drop(owners[0], queue)
            # A batch failing for a reason of its own: find its owner by retrying
            # owner by owner, each with every text it has not landed.
            for take in owners:
                stop, queue = self._solo(take, batch, queue)
                if stop is not None:
                    return stop, queue
            return None, queue

        try:
            self._upsert(batch, vectors)
        except VectorStoreUnreachableError as e:
            self._stop_store(e, "did not answer the write")
            return DRAIN_STOP_OUTAGE, queue
        except CollectionMissingError:
            self._stop_missing_on_write()
            return DRAIN_STOP_REFUSED, queue
        except Exception:
            # Refused or unconfirmed: retry each owner's points alone, with the
            # vectors in hand (no second embed).
            for take in owners:
                mine = [(item, vector) for item, vector in zip(batch, vectors)
                        if item[0] is take]
                try:
                    self._upsert([item for item, _ in mine], [v for _, v in mine])
                except VectorStoreUnreachableError as e:
                    self._stop_store(e, "did not answer the write")
                    return DRAIN_STOP_OUTAGE, queue
                except CollectionMissingError:
                    self._stop_missing_on_write()
                    return DRAIN_STOP_REFUSED, queue
                except Exception as e:
                    self._abort(take, e)
                    queue = self._drop(take, queue)
                    continue
                stop = self._landed(take, len(mine))
                if stop is not None:
                    return stop, queue
            return None, queue

        for take in owners:
            stop = self._landed(take, sum(1 for item in batch if item[0] is take))
            if stop is not None:
                return stop, queue
        return None, queue

    def _solo(
        self, take: _OwnerTake, batch: List[_QueuedItem], queue: List[_QueuedItem]
    ) -> Tuple[Optional[str], List[_QueuedItem]]:
        """Retries one owner alone, over every text of it not landed, in its own chunks.

        Its texts leave the queue first, so its tail never rides a later batch. It
        stops at its first failing chunk.
        """
        items = [item for item in batch if item[0] is take]
        items += [item for item in queue if item[0] is take]
        queue = self._drop(take, queue)
        while items:
            n = fit_batch([text for _, _, text in items])
            chunk, items = items[:n], items[n:]
            try:
                vectors = self._embed_batch([text for _, _, text in chunk])
            except Exception as e:
                cause = classify_provider_error(e)
                if cause in _DRAIN_OUTAGE_CAUSES:
                    self._stop_provider(cause, e)
                    return DRAIN_STOP_OUTAGE, queue
                self._abort(take, e)
                return None, queue
            try:
                self._upsert(chunk, vectors)
            except VectorStoreUnreachableError as e:
                self._stop_store(e, "did not answer the write")
                return DRAIN_STOP_OUTAGE, queue
            except CollectionMissingError:
                self._stop_missing_on_write()
                return DRAIN_STOP_REFUSED, queue
            except Exception as e:
                self._abort(take, e)
                return None, queue
            stop = self._landed(take, len(chunk))
            if stop is not None:
                return stop, queue
        return None, queue

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """One ``embed_documents`` call, its answer validated (the model was, at entry)."""
        self.calls += 1
        vectors = self.provider.embed_documents(texts)
        if not (
            isinstance(vectors, list)
            and len(vectors) == len(texts)
            and all(isinstance(v, list) and v for v in vectors)
        ):
            raise EmbeddingError(
                "the embedding provider answered a document batch with the wrong shape"
            )
        return vectors

    def _upsert(self, items: List[_QueuedItem], vectors: List[List[float]]) -> None:
        self.index.upsert_points(
            [RejectionPoint(pid, take.owner, vector)
             for (take, pid, _), vector in zip(items, vectors)],
            model=self.model,
            may_create=self.may_create,
        )
        # Written: the collection exists now, whatever the listing said.
        self.collection_present = True

    def _landed(self, take: _OwnerTake, count: int) -> Optional[str]:
        self.items_embedded += count
        take.landed += count
        if take.landed == len(take.to_write) and not take.aborted:
            return self._finalize(take)
        return None

    @staticmethod
    def _drop(take: _OwnerTake, queue: List[_QueuedItem]) -> List[_QueuedItem]:
        return [item for item in queue if item[0] is not take]

    # -- finalizing an owner ----------------------------------------------- #

    def _finalize(self, take: _OwnerTake) -> Optional[str]:
        """Removes what the field no longer states, then clears against the seq read.

        Runs only once every item the owner lacked is written, so an interrupted
        drain leaves the old items beside the new ones, never none.
        """
        if take.to_remove:
            try:
                self.index.delete_points(sorted(take.to_remove))
            except VectorStoreUnreachableError as e:
                self._stop_store(e, "did not answer a delete")
                return DRAIN_STOP_OUTAGE
            except CollectionMissingError:
                self._stop_missing_on_write()
                return DRAIN_STOP_REFUSED
            except Exception as e:
                # Reported and left queued, never counted as done.
                self._abort(take, e, removing=True)
                return None
            self.items_removed += len(take.to_remove)
        self._clear(take.row, take.slug)
        return None

    def _clear(self, row: RejectionBacklogRow, slug: Optional[str]) -> None:
        try:
            cleared = self.store.clear_rejection_backlog_row(
                row.node_id, enqueue_seq=row.enqueue_seq
            )
        except Exception as e:
            self.owners_failed += 1
            print(
                f"[Warning] The rejection backlog row for '{slug or row.node_id}' "
                f"could not be cleared: {describe_embed_failure(e)}. The row stays "
                f"queued; the next drain finds its items in place."
            )
            return
        if cleared:
            self.owners_cleared += 1
        # False: a commit re-queued the owner since the claim. Its new row carries a
        # larger seq, ahead of the cursor, so this walk takes it again.

    def _abort(self, take: _OwnerTake, exc: BaseException, *, removing: bool = False) -> None:
        if take.aborted:
            return
        take.aborted = True
        self._fail_row(take.row, take.slug, exc, removing=removing)

    def _fail_row(self, row: RejectionBacklogRow, slug: str, exc: BaseException,
                  *, removing: bool = False) -> None:
        self.owners_failed += 1
        what = ("its old items were not removed" if removing
                else "its items were not replaced")
        print(
            f"[Warning] Rejected items for '{slug}': {what} — "
            f"{describe_embed_failure(exc)}. Its rejection backlog row stays queued."
        )
        try:
            self.store.note_rejection_backlog_failure(
                row.node_id, enqueue_seq=row.enqueue_seq
            )
        except Exception:
            pass  # the row stands either way; only its count did not move

    # -- stops and the marker ---------------------------------------------- #

    def _stop_provider(self, cause: str, exc: BaseException) -> None:
        print(
            f"Rejection index: stopping the drain — the embedding provider is "
            f"unavailable ({cause}): {describe_embed_failure(exc)}. No row was "
            f"charged; the rows stay queued for the next `mitos sync {self.selector}`."
        )

    def _stop_store(self, exc: BaseException, what: str) -> None:
        print(
            f"Rejection index: stopping the drain — the vector store {what}: "
            f"{describe_embed_failure(exc)}. No row was charged; the rows stay "
            f"queued for the next `mitos sync {self.selector}`."
        )

    def _stop_refused(self) -> None:
        # Names the heal: `reconcile` queues every owner the index lacks and stamps the
        # rejection seed marker, which licenses the create this drain refused. An
        # ordinary `sync` over a populated graph never builds the index (the build is
        # operator-invoked, never implicit).
        try:
            waiting: Optional[int] = self.store.count_rejection_backlog()
        except Exception:
            waiting = None
        count = f"{waiting} " if waiting is not None else ""
        print(
            f"Rejection index: {count}rejection backlog row(s) wait until the "
            f"rejection index is built; nothing was created or embedded. "
            f"`mitos reconcile {self.selector}` builds it."
        )

    def _stop_missing_on_write(self) -> None:
        if self.may_create:
            print(
                "Rejection index: stopping the drain — the rejection collection was "
                "created and still answered missing to the write. No row was "
                "charged; the rows stay queued."
            )
        else:
            self._stop_refused()

    def _claim_failed(self, exc: BaseException) -> None:
        print(
            f"[Warning] Failed to read the rejection backlog: "
            f"{describe_embed_failure(exc)}"
        )

    def _spend(self) -> str:
        """Spends the marker this drain read; one re-stamped meanwhile survives."""
        try:
            self.store.clear_rejection_seed(established_at=self.seed["established_at"])
        except Exception as e:
            print(
                f"[Warning] Failed to clear the rejection seed marker: "
                f"{describe_embed_failure(e)}"
            )
            return DRAIN_MARKER_KEPT
        return DRAIN_MARKER_SPENT


class MitosSyncManager:
    """Manages the full parse-enrich-commit sync flow and side effects."""

    def __init__(self, config: MitosConfig) -> None:
        self.config = config
        self.lock_path = self.config.decisions_file + ".lock"
        self.lock = FileLock(self.lock_path, timeout=60)
        self.store = GraphStore(self.config.db_path)
        
        # Lazy initialize vector / embedding dependencies as best-effort (C2/P14)
        self.embed_provider: Optional[GeminiEmbeddingProvider] = None
        self.vector_store: Optional[QdrantVectorStore] = None
        # The sibling rejection-item collection's client (recall-confidence 4b).
        # Written inline by ``_write_rejection_items_inline``; None whenever either
        # constructor above failed, so a keyless or "down" workspace runs no leg.
        self.rejection_index: Optional[QdrantRejectionIndex] = None

        # Set once an inline embed has met an absent collection it may not create, so
        # a multi-entry sync prints ONE deferral line and skips N−1 wasted Qdrant round
        # trips (and N−1 embedding calls). Per manager instance — one per command — and
        # read by the decision leg (``_embed_decision_inline``) and the post-commit
        # acknowledged-neighbour gather (``_acknowledged_fields``), which would
        # meet the same absence. Neither reads the rejection latches below, and the
        # rejection leg never reads this one. A covering drain later in the same
        # ``perform_sync`` must not consult it, or a rebuild-then-sync would refuse to
        # heal because a record earlier in the run found the collection missing.
        self._collection_absent = False

        # The rejection leg's two per-command latches, read ONLY by
        # ``_write_rejection_items_inline``. ``_rejection_collection_absent``: an
        # unlicensed leg found the rejection collection absent, so later entries skip
        # the probe and spend nothing (the absence is silent — every upgraded
        # workspace sits there until its one rebuild of the index, and the backlog
        # row is the durable record). ``_rejection_leg_stopped``: an outage (an
        # auth, quota or transient provider cause, or an unreachable store), so a
        # sync committing forty entries during a 429 spends one failed call, not
        # forty. An ``unknown`` cause or a refusal latches neither: it costs one owner.
        self._rejection_collection_absent = False
        self._rejection_leg_stopped = False

        try:
            cache_path = os.path.join(self.config.mitos_dir, "embedding_cache.sqlite")
            self.embed_provider = GeminiEmbeddingProvider(
                cache_path,
                api_key=self.config.env.get("GEMINI_API_KEY"),
                model_id=get_embedding_model_id(self.config.env),
            )
            self.vector_store = QdrantVectorStore(
                self.config.qdrant_url,
                self.config.qdrant_collection
            )
            # Last, inside the same try: no network I/O at construct, and any
            # earlier failure leaves it None exactly as it leaves the others.
            self.rejection_index = QdrantRejectionIndex(
                self.config.qdrant_url,
                self.config.rejection_collection,
            )
        except Exception as e:
            # Let operations continue in degraded graph-only mode per S1/F2
            pass

    def auto_heal_decisions_file(self) -> None:
        """Auto-restores the decisions.md header and sample format block if modified or missing."""
        filepath = self.config.decisions_file
        if not os.path.exists(filepath):
            return

        # Load canonical format spec from package single source of truth
        try:
            format_spec_content = load_format_spec()
        except Exception:
            return

        # Extract sample block
        import re
        match = re.search(r'## 3\.\s+Sample Entry.*?\n```markdown\n(.*?)\n```', format_spec_content, re.DOTALL | re.IGNORECASE)
        sample_block = match.group(1).strip() if match else ""
        if not sample_block:
            return

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        canonical_header = (
            "# Decisions for Mitos\n\n"
            "<!-- This file is managed by mitos. LLM integration: see .mitos/skill.md once V5 ships. -->\n"
            "<!-- DO NOT MODIFY ABOVE THIS LINE -->\n\n"
            "## SAMPLE FORMAT — auto-restored by mitos sync, do not modify or delete\n\n"
            f"{sample_block}\n\n"
        )

        marker = "<!-- BEGIN ENTRIES — new decisions go directly below this line, newest first -->"
        if marker in content:
            parts = content.split(marker, 1)
            entries_content = parts[1]
            current_header = parts[0]
            if current_header.strip() != canonical_header.strip():
                new_content = canonical_header + marker + entries_content
                atomic_file.write_source(filepath, new_content)
                # stderr, not stdout: this method is on `record_decision_entry`'s path,
                # which the MCP write tool shares — and that transport uses stdout for
                # JSON-RPC, so a stray line here is protocol corruption, not noise. It
                # is also ahead of `restore-source --json`'s object, which a caller
                # parses. Matches this file's existing stderr discipline on the
                # embedding/render warnings below.
                print("Auto-restored decisions.md sample format header block ✓",
                      file=sys.stderr)
        else:
            if "## SAMPLE FORMAT" not in content:
                new_content = canonical_header + marker + "\n\n" + content
                atomic_file.write_source(filepath, new_content)
                print("Auto-restored missing sample format header and BEGIN ENTRIES "
                      "marker ✓", file=sys.stderr)

    def perform_sync(self, auto_accept: bool = False, verbose: bool = False,
                     repair_targets: Optional[List[str]] = None,
                     full: bool = False) -> List[str]:
        """Executes the complete transactional sync flow.

        The repair ledger is built and reported HERE rather than inside
        ``_perform_sync_internal``, which has four returns plus a fall-through and
        two of them pull in opposite directions: the no-parseable-entries return
        carries two of the exit table's own rows (an unparseable target and one
        absent from the buffer), while the key-floor return must stay silent. One
        report site after every path, and one explicit carve-out marked at the
        floor, beats four call sites and four chances to miss the one that matters.

        The two drains, the decision outbox's and then the rejection backlog's, run
        at the end of a run that read the buffer, each in a try of its own. The
        early returns (the buffer lock busy, no entry parsed, the key floor) skip
        both; ``--embed-only`` runs both without a sync.

        Args:
            auto_accept: Whether ``--yes`` is in force.
            verbose: Emit cache statistics at the end of the run.
            repair_targets: Handles named by ``--reconcile-entry``, in the caller's
                verbatim spelling. ``None`` (no flag) and ``[]`` mean the same
                thing here — nothing was named, so the ledger is inert.
            full: ``--full``: the divergence report prints each diverged commentary
                field whole on both sides instead of its differing span.

        Returns:
            The named targets that did NOT end in the state their markdown
            describes, for the caller to exit on. Empty on every ordinary run.
        """
        ledger = _RepairLedger(repair_targets)
        snapshot_path = os.path.join(self.config.mitos_dir, "sync_snapshot.md")
        # Second snapshot for steady-state questions.md ingestion (Phase 4a): taken
        # under the same lock as the decisions snapshot for read-consistency, and
        # cleaned in the same finally. Absent when questions.md is absent (healthy).
        questions_snapshot_path = os.path.join(self.config.mitos_dir, "questions_snapshot.md")
        try:
            self._perform_sync_internal(
                snapshot_path, questions_snapshot_path, auto_accept, verbose, ledger,
                full=full,
            )
        finally:
            for path in (snapshot_path, questions_snapshot_path):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
        return ledger.report()

    def _perform_sync_internal(self, snapshot_path: str, questions_snapshot_path: str, auto_accept: bool = False, verbose: bool = False, ledger: Optional["_RepairLedger"] = None, full: bool = False) -> None:
        """Executes the internal transactional sync flow.

        ``ledger`` is defaulted so the sole production caller stays the only site
        that has to know about it and a direct call keeps working; an inert ledger
        makes every ``note`` below a no-op. ``is None`` rather than a truthiness
        coalesce, because an inert ledger is a real object with a real contract.
        """
        if ledger is None:
            ledger = _RepairLedger()

        # 1. Snapshot-at-sync-start under brief file lock
        questions_snapshotted = False
        try:
            with self.lock:
                if not os.path.exists(self.config.decisions_file):
                    print("No decisions.md file found. Run 'mitos init' first.")
                    return
                # Auto-heal the decisions file header/sample block under lock
                self.auto_heal_decisions_file()
                shutil.copy(self.config.decisions_file, snapshot_path)
                # Snapshot questions.md under the SAME lock (4a steady-state OQ
                # ingestion). An absent questions.md is healthy-empty (no current
                # corpus ships one). File-level bulkhead (D4/P7): a copy error in
                # the OQ buffer warns and yields zero OQ entries — it must never
                # abort decisions ingestion.
                if os.path.exists(self.config.questions_file):
                    try:
                        shutil.copy(self.config.questions_file, questions_snapshot_path)
                        questions_snapshotted = True
                    except OSError as exc:
                        print(
                            f"[Warning] Could not snapshot questions.md ({exc}); "
                            f"skipping open-question ingestion this sync."
                        )
        except Timeout:
            print("Another Mitos process holds the lock; check for stuck 'mitos sync'.")
            # Nothing was read, so a named target's line may not say it is absent.
            ledger.mark_buffer_unread(_REPAIR_UNREAD_LOCKED)
            return

        # 2. Parse from the snapshots, oldest-first within each file. Steady-state
        #    sync now ingests BOTH the decisions buffer and the questions.md
        #    open-question buffer (4a) — each parsed kind-by-file (V1-D8) into its
        #    OWN failure collector so a malformed entry in one buffer never aborts
        #    the other (D4 bulkhead). Oldest-first (parse_file_reversed) lands an
        #    older in-buffer entry before a newer one that references it — a flow
        #    heuristic, NOT correctness; the per-entry quarantine below (and 4b's
        #    fixpoint) are correctness. Collector mode isolates a malformed entry
        #    (reported + skipped) so the rest still sync (§5.2.2 per-entry isolation).
        dec_failures: List[EntryFailure] = []
        decision_entries = parse_file_reversed(snapshot_path, "decision", dec_failures)

        oq_failures: List[EntryFailure] = []
        oq_entries: List[ParsedEntry] = []
        if questions_snapshotted:
            # File-level bulkhead (D4/P7): the OQ snapshot read+parse is wrapped in
            # its OWN try/except so a file-level fault in questions.md — e.g. invalid
            # UTF-8 bytes the binary snapshot copy passed straight through, which
            # read_text_or_none then hits decoding as UTF-8 — warns and yields ZERO
            # OQ entries while decisions ingestion proceeds. The secondary buffer must
            # never abort the primary. (Per-ENTRY malformed OQs are already collector-
            # isolated into oq_failures, not raised; this catches the file-level fault
            # the collector cannot reach.)
            try:
                oq_entries = parse_file_reversed(
                    questions_snapshot_path, "open_question", oq_failures
                )
            except Exception as exc:
                oq_entries = []
                print(
                    f"[Warning] Could not parse questions.md ({exc}); "
                    f"skipping open-question ingestion this sync."
                )

        all_failures = dec_failures + oq_failures
        # Recorded before the `if not entries:` return below, because that return
        # carries two of the exit table's own rows: a named target whose block
        # failed to parse, and one genuinely absent from the buffer.
        ledger.note_parse_failures(all_failures)
        for fail in all_failures:
            msgs = "; ".join(item.message for item in fail.items) or "malformed entry"
            print(
                f"[Parse error] {msgs} (lines {fail.line_start}-{fail.line_end}). "
                f"Entry skipped — fix it and re-run sync."
            )

        # Decisions first, then open questions (D1): the host decision of an OQ's
        # typical Derives-From: forward-ref commits before the OQ, landing that
        # common case on the first pass. The opposite direction (a decision that
        # Resolves: an OQ authored above it) still quarantines on the first pass in
        # 4a and converges on the next sync (4b's fixpoint converges it in one).
        entries = decision_entries + oq_entries

        if not entries:
            if all_failures:
                print("No parseable entries to commit. Fix the reported entries above and re-run sync.")
            else:
                print("Zero pending entries found in the decisions.md / questions.md write-buffers.")
            return

        api_key = self.config.env.get("GEMINI_API_KEY")
        if not api_key:
            print("GEMINI_API_KEY environment variable is not set. Sync requires API keys.")
            # The report's ONE carve-out. This refusal returns above the per-entry
            # loop, so a named target was neither reconciled nor looked for — there
            # is nothing to be loud about, and the run's exit stays 0 because
            # flipping it is a contract break under every CI job that reads it.
            # A floor on loudness, never a ceiling: a keyless workspace whose buffer
            # is empty reaches the return above this one and DOES report, correctly,
            # because on that corpus the named target genuinely is absent.
            ledger.mark_keyless()
            # The keyless floor commits nothing, but what rotates is a property of the
            # buffer, not of this run's commits — so it evaluates rotation before it
            # returns, exactly as the keyed path does after its commits (D2; Vinga's
            # D-3e-5 ruling, 2026-09-18). Without it a keyless workspace drained only
            # through `record`.
            self._rotate_after_sync()
            return
            
        renderer = MitosRenderer(self.config.workspace_dir)

        # 4b intra-sync fixpoint: the main pass below collects every entry the store
        # rejects with a CommitError (a forward-ref whose in-corpus target has not
        # committed yet, a slug collision, a kind/cycle violation) into this set
        # instead of reporting it immediately. After the main pass, the fixpoint
        # re-attempts the set until a pass makes no further progress, so any acyclic
        # cross-file forward-ref chain converges in THIS single sync. Each tuple
        # carries the fully-prepared entry, a raw-text slot, and its latest failure
        # (for the post-fixpoint residual report). The slot is the shape `replay`
        # shares with `rebuild`; sync slices no raw text, so it carries "" — rotation
        # reads its blocks from the live buffer instead.
        quarantined: List[Tuple[ParsedEntry, str, CommitError]] = []

        # Conflict sensor (5a): build the judgment executor ONCE per run (CONF-D4/D7) —
        # one Anthropic client, one availability decision — and reference it per entry
        # inside the loop. Skipped (left None) when the sensor is inactive: `--yes`
        # (auto_accept runs no per-entry review at all) or the `conflict_check_on_sync`
        # toggle is off. `_build_conflict_judge` folds in the remaining availability gate
        # (ANTHROPIC_API_KEY + live embed/vector components) and lazy-imports `anthropic`,
        # so the SDK never rides a sync when the sensor is off.
        # 5b builds the per-run conflict context (sync_run_id + telemetry + breaker)
        # alongside the judge, ONCE per run, under the SAME availability guard — so no
        # TelemetryStore is constructed when the sensor is off, and the single
        # `_ConflictSyncRun` instance is what the write-then-read aggregate breaker
        # depends on (build it per entry and the breaker never trips). `conflict_run`
        # is non-None exactly when `conflict_judge` is.
        conflict_judge: Optional[Callable] = None
        conflict_run: Optional[_ConflictSyncRun] = None
        if not auto_accept and self.config.conflict_check_on_sync:
            conflict_judge = self._build_conflict_judge()
            if conflict_judge is not None:
                conflict_run = self._new_conflict_run()

        # 3. Process each parsed entry
        for entry in entries:
            # Check if this node is already in the database (slug-free V1a id — V1-D2).
            node_id = compute_node_id(
                kind=entry.kind,
                axiom=entry.axiom,
                mechanism_refs=entry.mechanisms,
                topic=entry.topic,
                questions_raised=entry.questions_raised,
            )

            existing = self.store.get_node(node_id)
            if existing:
                # Idempotency short-circuit (S5). The node commit is deliberately
                # skipped — but a re-encounter from a NEW source must still emit one
                # source_reencounter audit signal (MI-4 / V1-D14). source is
                # out-of-core, so the canonical core is identical here; the
                # signal-eval is the one pass the skip must NOT swallow (§6.2
                # Lesson 13). This gate sits above the per-entry commit quarantine,
                # but note_source_reencounter is best-effort, so a signal failure
                # can never abort a sync of real decisions.
                self.store.note_source_reencounter(
                    node_id, existing["source"], entry.source or "user"
                )

                # C′ — the commentary reconcile, DIVERGENCE-GATED. Without the gate
                # this branch would reach `commit_parsed_entry` for every
                # already-committed entry: on the dogfood corpus that is 203 accept
                # prompts and 203 SONNET conflict judgments per sync. Gated, a clean
                # corpus takes the same `continue` as before — zero behavioural delta, and MI-3's
                # no-tick-on-a-byte-identical-recommit property stays intact.
                if entry.kind != "decision":
                    # Decisions only, matching the detector, which excludes open
                    # questions on both sides. The parser assigns `context` and
                    # `invalidates_if` kind-agnostically while the store NULLs them for
                    # an open question, so a reconcile of one can never converge: it
                    # would print "Reconciled ✓" and append a fresh attribution row on
                    # every single sync, for a mutation that provably cannot land.
                    ledger.note(entry, "open_question")
                    continue

                divergence = entry_divergence(
                    entry,
                    existing,
                    existing.get("scope") or [],
                    self.store.get_outgoing_edges(node_id),
                )
                if not is_reconcilable(divergence):
                    # `source` divergence is reported by `status` but is NOT
                    # reconcilable — MI-4 fences it out of the commentary UPDATE, so
                    # re-committing provably cannot change it.
                    #
                    # So this ONE branch covers TWO states for a named target, and
                    # the repair flag has to tell them apart: a genuinely clean
                    # entry is satisfied (the corpus and graph agree — exit 0),
                    # while one diverging ONLY in `source` is permanently
                    # unreconcilable and must not read as "not diverged". Exit 0
                    # there is the silent no-op the fail-loud property forbids, and
                    # `status`'s own rung already names the heal for it.
                    ledger.note(
                        entry,
                        "source_only" if divergence.get("source") else "clean",
                    )
                    continue

                if not self._apply_commentary_reconcile(
                    entry, existing, node_id, divergence, auto_accept,
                    authorized=ledger.authorizes(entry), full=full,
                ):
                    ledger.note(entry, "reconcile_refused")
                    continue
                # Reconciled. Deliberately falls through to `continue` rather than the
                # commit path below: no conflict judge (the canonical core is unchanged,
                # so there is no new claim to judge), no accept prompt, no confirmation
                # re-stamp. The reconcile ticks `updated_at`, so the entry is recently
                # touched and step 4's settledness keeps it in the buffer: an entry that
                # leaves the buffer leaves sync's read-set, and its future divergence
                # would become invisible again.
                ledger.note(entry, "reconciled")
                continue

            # Slug collision check
            collision = self.store.get_node_by_slug(entry.slug)

            if collision:
                # A kill-edge the entry ALREADY declares at the colliding slug is the
                # supported same-slug supersession pattern (MI-13 rationale / FM1): the
                # author is evolving an axiom while keeping the citation handle. Commit
                # it exactly as declared — the override below would rewrite a declared
                # `supersedes` into `corrects` and silently contradict the author.
                # Normalize exactly as the edge resolver does (``_strip_citation`` +
                # casefold, MI-7): the deterministic parser stores `[slug]` with the
                # brackets on, the agentic write path stores a bare slug, and both are
                # the same declaration. Driving off ``_KILL_EDGE_FIELDS`` rather than a
                # re-hand-rolled pair keeps this carve-out correct if the kill-edge set
                # ever widens.
                declared_kill = {
                    _strip_citation(raw).casefold()
                    for field in _KILL_EDGE_FIELDS
                    for raw in (getattr(entry, field, None) or [])
                }
                if entry.slug.casefold() in declared_kill:
                    print(
                        f"\n[Collision] Slug '{entry.slug}' already exists in graph — "
                        "the entry declares a relation at it; committing as declared."
                    )
                else:
                    # An UNDECLARED collision is reported and skipped in BOTH modes,
                    # never auto-retired and never prompted for. Defaulting to `corrects`
                    # mints a killer node that retires a real decision on nothing but a
                    # slug match — and a canonical core can shift from any hand-edit to a
                    # `**Mechanisms:**` line, so the collision is as likely to be an
                    # accident as an intent (P5 Ironclad: automated recovery never
                    # destroys user data). A vector, not a wall: the entry stays in the
                    # buffer and the fix is one authored line.
                    #
                    # The interactive `[c]orrection / [s]upersession` prompt used to sit
                    # here, and it is retired rather than fixed. Its answer was only ever
                    # applied in memory — nothing spliced the chosen line into the buffer,
                    # and rotation archived the raw unmodified block — so every
                    # interactively-resolved collision committed a kill-edge the gold
                    # source does not declare. Against P6/M7 (the markdown must remain the
                    # rebuildable truth): the entry replays at rebuild without its
                    # declaration, collides at the store, and becomes a permanent
                    # casualty. Sending the author to the markdown puts the declaration
                    # where a rebuild can find it again, and deleting the prompt also
                    # deletes the override below it, which wholesale-replaced both kill
                    # lists and so discarded any kill-edge the entry authored at a
                    # *different* slug.
                    print(f"\n[Collision] Slug '{entry.slug}' already exists in graph.")
                    print(f"  Existing Axiom: {collision.get('core_axiom')}")
                    print(f"  New Axiom:      {entry.axiom}")
                    print(
                        "  Skipped — nothing written. This entry declares no relation at "
                        f"'{entry.slug}', and sync will not retire a decision on a slug "
                        "match alone."
                    )
                    print(
                        f"  To supersede it, add `**Supersedes:** [{entry.slug}]` to the "
                        f"entry; to correct it, add `**Corrects:** [{entry.slug}]`. "
                        "Then re-run sync."
                    )
                    ledger.note(entry, "collision")
                    continue

            # No terminal and no `--yes`: report this pending entry and skip it, never
            # prompt. Position is the contract, not a placement detail. ABOVE the kind
            # split below, so ONE refusal dominates BOTH accept prompts — a guard
            # written inside either branch is one chance in two to ship a door still
            # dead on the other kind. Above the split also puts it above the per-entry
            # conflict sensor by construction, which is a spend contract: the crash
            # this replaces bounded the sensor's loop at one entry, so a guard placed
            # at the `input()` would let a run that can accept *nothing* sweep a paid
            # judgment across the whole pending buffer. And BELOW the collision block,
            # so every shipped per-entry diagnostic above it still prints.
            #
            # Report-and-skip, exit unchanged at 0: the entry stays in decisions.md
            # (sync never deletes from it), so `--yes` commits it next run — a skip
            # costs a turn, not an entry. It replaces an unguarded `input()` that died
            # with `EOF when reading a line` → `Fatal Unexpected Error`, exit 1, naming
            # nothing; the refusal names the entry it skipped.
            if not auto_accept and not sys.stdin.isatty():
                print(f"\n[Pending] '{entry.slug}' — skipped, stdin is not a terminal. "
                      "Re-run with `--yes` to accept pending entries non-interactively.")
                # A named target still pending is NOT satisfied: the repair flag
                # authorizes the reconcile gate, not this accept prompt, so the
                # refusal above stays true and stays unchanged.
                ledger.note(entry, "pending_skipped")
                continue

            if entry.kind == "decision":
                # Strict-deterministic sync (A): the decision commits EXACTLY as
                # authored — no LLM enrichment in the sync runtime path. Refining a
                # rough axiom, inferring scopes, or suggesting relationships lives only
                # where the input is genuinely raw (`capture`, `import --llm-extract`);
                # `sync` reads the human-authored buffer and must never silently rewrite
                # it (M7/P6 — markdown is the source of truth; ROADMAP — runtime parsing
                # is strict-deterministic; ADR sync-strict-deterministic-no-llm-enrichment).
                print(f"\nProposed Decision: {entry.slug}")
                print(f"  Core Axiom:  {entry.axiom}")
                print(render_rejected_paths(entry.rejected_paths, "  Rejected:    "))
                print(f"  Mechanisms:  {', '.join(entry.mechanisms)}")
                print(f"  Scope:       {', '.join(entry.scope)}")

                # Conflict sensor (5a): judge this decision against its undeclared close
                # neighbours and surface any high-confidence contradiction BEFORE the
                # accept prompt (CONF-D7), so the tension is named while the author can
                # still choose. `conflict_judge is not None` plus the guard above imply the
                # full gate (decision-kind here, not auto_accept, a terminal on stdin,
                # toggle on, judge available). RF-1:
                # `entry` holds the parsed declarations and nothing else — the slug-collision
                # override that used to rewrite them below is retired, so the property RF-1
                # relied on is now structural. Advisory — it prints, never blocks; the accept
                # prompt below is untouched.
                if conflict_judge is not None:
                    self._run_and_surface_conflict(entry, conflict_judge, conflict_run)

                if not auto_accept:
                    u_choice = input("Accept this decision? [a]ccept / [s]kip / [q]uit: ").strip().lower()
                    if u_choice == 's':
                        ledger.note(entry, "operator_skipped")
                        continue
                    elif u_choice == 'q':
                        print("Sync paused by user.")
                        ledger.note(entry, "operator_quit")
                        # The loop ends here, so every entry BELOW this one in the
                        # document went unread. A named target among them is still a
                        # shortfall, but the report may not call it absent.
                        ledger.mark_buffer_unread(_REPAIR_UNREAD_QUIT)
                        break

            else:
                # Open Question Sync
                print(f"\nProposed Open Question: {entry.slug}")
                print(f"  Questions: {', '.join(entry.questions_raised)}")
                if not auto_accept:
                    u_choice = input("Accept this open question? [a]ccept / [s]kip / [q]uit: ").strip().lower()
                    if u_choice == 's':
                        ledger.note(entry, "operator_skipped")
                        continue
                    elif u_choice == 'q':
                        print("Sync paused by user.")
                        ledger.note(entry, "operator_quit")
                        ledger.mark_buffer_unread(_REPAIR_UNREAD_QUIT)  # as above
                        break

            # Populate OD3 confirmation metadata
            # Strict-deterministic sync commits the authored buffer verbatim — no model
            # touches the decision here, so its provenance is the user/author, not an
            # enrichment model (A: sync-strict-deterministic-no-llm-enrichment).
            entry.confirmed_by = "user"
            # MI-10: application-supplied UTC ISO-8601 with an explicit offset — the
            # same helper `created_at` uses on the very same row. `datetime.now()` is
            # naive local time, which is what put 114 offset-less stamps beside 114
            # offset-aware ones in the live graph.
            entry.confirmed_at = _utc_now_iso()

            # Commit to graph database atomically per entry (C1 atomicity), now with
            # the per-entry commit-stage quarantine (4a floor — P5 dead-letter / P7
            # bulkhead): a store-stage rejection isolates to THIS entry while the rest
            # of the batch (decisions AND open questions) still commits — never the
            # whole-sync abort it was before. The catch keys on the CommitError CLASS,
            # not a code allowlist, so every §5.2.2 structural rejection (missing_target
            # ∪ slug_collision ∪ cycle_violation ∪ kind_constraint_violation ∪
            # dangling_edge) funnels through it uniformly (D3). 4b lifts the floor to a
            # ceiling: a quarantined entry is COLLECTED (not reported here) and re-tried
            # by the intra-sync fixpoint after this loop, so an acyclic forward-ref
            # converges in this sync rather than the next.
            try:
                delta = self.store.commit_parsed_entry(entry)
            except CommitError as exc:
                # 4b: collect (don't report yet) for the intra-sync fixpoint retry
                # after this loop. The entry is fully prepared (enriched, collision-
                # resolved, confirmed-stamped) — only the store-stage commit failed,
                # and its in-corpus target may yet commit in this same sync.
                quarantined.append((entry, "", exc))
                # Pessimistic stamp: the fixpoint below upgrades whichever of these
                # it drains, so a missed correction leaves a shortfall rather than a
                # false satisfaction.
                ledger.note(entry, "quarantined")
                continue
            except (ValidationError, DatabaseError) as exc:
                # Defensive secondary bulkhead (Gotcha): a bypassed-parser empty core
                # or a raw SQLite failure carries NO §5.2.2 envelope, so the CommitError
                # quarantine above would miss it and it would abort the batch. Isolate
                # it per-entry too — report + skip (a genuine defect, never retry-
                # eligible). CommitError-only is the firm contract; this is the
                # defensive half.
                print(
                    f"\n[Quarantined] '{entry.slug}' (lines {entry.line_start}-"
                    f"{entry.line_end}): unexpected store error — {exc}. Entry left in "
                    f"its buffer; fix and re-sync."
                )
                ledger.note(entry, "store_error")
                continue
            print(f"Committed node: {entry.slug} ✓")
            ledger.note(entry, "committed")

            # best-effort embedding upsert (C2) — applies to OQ nodes too. The
            # rejection leg is left to this sync's own drain (step 6b).
            self._best_effort_embed(delta, entry, rejection_leg=False)

        # 3b. Intra-sync fixpoint retry (4b). The main pass committed every entry
        # whose targets were already present; re-attempt the quarantined set until a
        # pass makes no further progress, so any acyclic cross-file forward-ref chain
        # (D resolves Q, Q derives_from D', … however deep, in any authoring order)
        # converges in THIS single sync — order-independently, while every retry stays
        # an isolated per-entry commit_parsed_entry transaction (no batching, no
        # ordering — D5/MI-12). A genuinely-unresolvable reference (a never-authored
        # target, or a true A↔B mutual-reference cycle) makes zero progress, terminates
        # after one no-progress pass, and surfaces below as a loud per-entry vector —
        # never a hang, never a whole-sync abort.
        residual = self._commit_quarantine_fixpoint(quarantined)
        # The satisfied "committed" state is TWO sites, not one. A stamp placed only
        # after the main pass's `Committed node:` misses every entry the fixpoint
        # drains, so a named forward-ref target would exit non-zero on a run that
        # committed it. Keyed on identity because the residual tuples are the
        # quarantined ones.
        residual_ids = {id(entry) for entry, _raw, _exc in residual}
        for entry, _raw, _exc in quarantined:
            if id(entry) not in residual_ids:
                ledger.note(entry, "committed")
        for entry, _raw, exc in residual:
            self._report_commit_quarantine(entry, exc)

        # 4. Archive rotation of the buffer's settled tail.
        self._rotate_after_sync()

        # 5. Trigger renderer to statelessly regenerate files (C3)
        try:
            renderer.render_all(self.store)
            print("Regenerated live_axioms.md ✓")
        except Exception as e:
            # Degradation F4b: render failure doesn't affect graph commits
            print(f"[Warning] Failed to render active axioms: {str(e)}")

        # Temporary snapshot cleanup is handled by the perform_sync finally block

        # 6. Best-effort outbox queue drain attempt (C2)
        try:
            self.drain_pending_embeddings()
        except Exception as e:
            print(f"[Warning] Outbox queue drain failed: {describe_embed_failure(e)}")

        # 6b. The rejection backlog's drain, after the decision drain and in a try of
        # its own: a fault here never touches that drain, its marker or the exit. It
        # takes every row the commit loop above left (that loop runs no inline
        # rejection leg), cross-owner batched.
        try:
            self.drain_rejection_backlog()
        except Exception as e:
            print(
                f"[Warning] Rejection backlog drain failed: {describe_embed_failure(e)}",
                file=sys.stderr,
            )

        # 7. Surplus hit/miss stats observability (4.D)
        if verbose and self.embed_provider:
            hits, misses, rate = self.embed_provider.get_stats()
            print(f"\n[Observability] Cache Stats: Hits: {hits}, Misses: {misses}, Hit Rate: {rate*100:.1f}%")

    def _rotate_after_sync(self) -> None:
        """Rotates the buffer's settled tail on the sync path and words the outcome.

        `_rotate_settled` reads the live buffer once under the lock and moves the
        contiguous settled run at its oldest end, bounded per acquisition. A rotation
        failure — a graph read included — is reported here and never fails the sync:
        the commits stand. Every line goes to stderr, and nothing rotated is silence.
        """
        try:
            outcome = self._rotate_settled(window=settledness.ROTATION_WINDOW_ENTRIES)
        except Exception as e:
            sys.stdout.flush()
            print(
                f"[Warning] Archive rotation failed: {e}. decisions.md is unchanged; an "
                f"archive may already hold a copy of the entries it was moving.",
                file=sys.stderr,
            )
        else:
            if outcome is not None:
                self._report_rotation(outcome)

    def _rotate_settled(self, *, window: int) -> Optional[rotation.RotationOutcome]:
        """Rotates the buffer's settled tail, at most ``window`` entries, archive first.

        Settledness (committed ∧ quiet ∧ not diverged) is evaluated by
        ``settledness.select_settled_tail`` inside ``rotation.rotate_selected``'s one
        read of the live buffer, under this manager's own lock — never a second read,
        and never the sync snapshot, whose whole-file replace would discard every
        capture since. The clock is read once: the same instant is the quiet test's
        ``now`` and names the archive (ADR
        ``rotation-names-the-archive-for-the-rotation-instant-not-created-at``). Only
        ``config.decisions_file`` is read, and it is parsed as decisions, so an open
        question never rotates.

        On the sync path this runs after the commits and their quarantine fixpoint, and
        on the keyless floor in their place (that floor commits nothing, but the buffer's
        settled tail does not depend on this run's commits). A sync that returns earlier
        — an empty buffer or a lock timeout — evaluates nothing. A run whose entries were skipped for review
        does evaluate: an uncommitted entry is simply never settled.

        On the record path ``record_decision_entry`` calls it once per ``created``
        write, after that write's own lock hold has released, and only when the bytes
        it just wrote hold at least ``rotation_volume_threshold_entries`` entries
        (``settledness.buffered_entries``). Every other record exit — ``exists``,
        ``needs_review``, an error, a lock timeout — evaluates nothing. The record
        path turns a raise into its receipt's ``rotation`` field.

        Args:
            window: The most entries to evaluate, and so to move, in this acquisition.

        Returns:
            The rotation outcome, or ``None`` when nothing was selected and nothing was
            written.

        Raises:
            Exception: Whatever the clock stamp, a graph read or the rotation raised.
                The buffer is unchanged; an archive may already hold a copy of the
                blocks. Prints nothing — each caller words the failure.
        """
        now = _utc_now_iso()
        archive_name = rotation.archive_name_for(now)
        chosen: List[settledness.Selection] = []

        def _select(buffer_text: str) -> List[rotation.RotationBlock]:
            selection = settledness.select_settled_tail(
                buffer_text,
                graph=self.store,
                now=now,
                lag_days=self.config.rotation_lag_days,
                threshold=self.config.rotation_volume_threshold_entries,
                window=window,
                archive_name=archive_name,
            )
            chosen.append(selection)
            return selection.blocks

        outcome = rotation.rotate_selected(
            self.lock, self.config.decisions_file, self.config.archive_dir, _select
        )
        if not chosen[0].blocks:
            return None
        return outcome

    @staticmethod
    def _report_rotation(outcome: "rotation.RotationOutcome") -> None:
        """Words a rotation outcome on stderr; a block left unmatched stays silent."""
        sys.stdout.flush()
        for archive_path in outcome.archive_paths:
            name = os.path.basename(archive_path)
            moved = sum(
                1 for block in outcome.rotated if os.path.basename(block.archive_name) == name
            )
            print(f"Rotated {moved} entries to {archive_path} ✓", file=sys.stderr)
        for label, count in outcome.duplicated:
            print(
                f"[Warning] Rotation skipped {label!r}: its block occurs {count} times in "
                f"decisions.md, so nothing was removed.",
                file=sys.stderr,
            )
        for label in outcome.overlapping:
            print(
                f"[Warning] Rotation skipped {label!r}: its block overlaps another block in "
                f"this batch, so neither was rotated.",
                file=sys.stderr,
            )

    def _commit_quarantine_fixpoint(
        self,
        quarantined: List[Tuple[ParsedEntry, str, CommitError]],
    ) -> List[Tuple[ParsedEntry, str, CommitError]]:
        """Drains the per-entry quarantine set to a fixpoint (4b).

        Sits on 4a's per-entry quarantine *floor*: the main pass collected every
        entry the store rejected with a :class:`CommitError` (a forward-ref whose
        in-corpus target had not committed yet, plus the structural rejections that
        never self-heal). This re-attempts that set until a pass commits nothing new,
        so any acyclic cross-file forward-ref chain converges in a **single** sync,
        order-independently.

        The convergence loop is the shared :func:`mitos.replay.commit_quarantine_fixpoint`
        primitive (the same engine the ``mitos rebuild`` corpus replay uses). This
        wrapper supplies the sync-specific embed callback and the loud
        convergence-observability line.

        Args:
            quarantined: The fully-prepared entries the main pass quarantined, each
                with an empty raw-text slot and its latest ``CommitError``.

        Returns:
            The residual entries that never committed, each still carrying its latest
            ``CommitError`` — ``[]`` when everything converged.
        """
        committed, passes, residual = commit_quarantine_fixpoint(
            self.store,
            quarantined,
            embed_fn=functools.partial(self._best_effort_embed, rejection_leg=False),
        )

        # Convergence observability: make the fixpoint's work visible (the vision
        # values loud diagnostics) without any timing assertion — a structural signal
        # a test can read. Only printed when the quarantine set was non-empty.
        if quarantined:
            entry_word = "entry" if committed == 1 else "entries"
            pass_word = "pass" if passes == 1 else "passes"
            print(
                f"\n[Fixpoint] converged {committed} quarantined {entry_word} over "
                f"{passes} retry {pass_word}; {len(residual)} unresolved."
            )
        return residual

    def _report_commit_quarantine(self, entry: ParsedEntry, exc: CommitError) -> None:
        """Reports a residual per-entry commit failure as a guiding vector.

        The per-entry commit-stage bulkhead (P5 dead-letter / P7 Bulkhead): a single
        entry's store-stage rejection — any member of the §5.2.2 ``CommitError``
        class — is isolated to *that* entry. The entry is left in its buffer (never
        recorded for rotation), so a later sync can commit it once the operator fixes
        the reference or authors the missing target; the rest of this batch proceeds
        untouched.

        Called **post-fixpoint** (4b): by the time this fires, the intra-sync fixpoint
        has already exhausted every in-corpus retry, so a residual ``missing_target``
        no longer means "authored later in this corpus, settles next sync" (4a's
        optimistic framing) — it means the referenced target is **not present anywhere
        in this corpus**: a forward-ref to a never-authored / renamed-away target, or a
        true mutual-reference cycle where neither member can commit first. The vector
        says so honestly (D4), still a *guiding* vector rather than the bare "does not
        match any entry" wall that reads as a typo (D6). Every other code gets a calm,
        located generic surface. This mirrors the code-aware message builder in
        ``cutover._cutover_error_for_commit`` (the sync-side analogue — that one is
        cutover-specific and returns a ``CutoverError``, so it is not reused here).

        Args:
            entry: The entry the store rejected.
            exc: The ``CommitError`` carrying the §5.2.2 failure envelope.
        """
        items = exc.failure.items if exc.failure else []
        codes = {item.code for item in items}
        detail = "; ".join(item.message for item in items) or str(exc)
        if STORE_MISSING_TARGET in codes:
            print(
                f"\n[Quarantined] '{entry.slug}' (lines {entry.line_start}-{entry.line_end}): "
                f"references a target that is not present anywhere in this corpus. The "
                f"intra-sync fixpoint already retried every in-corpus dependency, so this is "
                f"not a settles-on-the-next-sync forward-ref — author the missing target, fix "
                f"the reference, or break the mutual-reference cycle (in an A↔B cycle neither "
                f"member can commit first). Entry left in its buffer. ({detail})"
            )
        else:
            code_str = ", ".join(sorted(codes)) if codes else "referential violation"
            print(
                f"\n[Quarantined] '{entry.slug}' (lines {entry.line_start}-{entry.line_end}): "
                f"the store rejected it ({code_str}). Entry left in its buffer for a later "
                f"sync once fixed. ({detail})"
            )

    def _build_conflict_judge(self) -> Optional[Callable]:
        """Builds the bound conflict-judgment executor for this sync run, or None (5a).

        The availability gate for the sync-time conflict sensor: the judgment needs an
        Anthropic client (``ANTHROPIC_API_KEY``, resolved for *this workspace* off
        ``config.env`` — as in ``importer.py``) AND
        the live ``embed_provider`` + ``vector_store`` the facade's candidate gather reads
        (both ``Optional``, ``None`` when Qdrant/Gemini were down at manager init). If any
        is absent the sensor cannot run, so this returns ``None`` and the per-entry hook is
        skipped for the whole run — the sensor is advisory and its absence must be invisible
        to a commit (P14/CONF-D7). The caller's kind/toggle/``--yes`` gates sit above this;
        this method owns only component + key availability.

        Built once per run (CONF-D4): one client, one availability decision — never per
        entry (that would re-read env + re-import ``anthropic`` N times). The SDK import is
        lazy so ``import anthropic`` never rides a ``mitos sync`` when the sensor is off (the
        caller skips this builder when the toggle is off; a missing key returns before the
        import here). The executor caps retries/timeout itself via ``with_options``, so the
        client needs no retry knobs (IMPL_NOTES 3b).

        Returns:
            The bound one-arg ``judge`` callable
            ``(RenderedPrompt) -> JudgmentExecution | Unavailable``, or ``None`` when the
            sensor is unavailable.
        """
        if self.embed_provider is None or self.vector_store is None:
            return None
        api_key = judge_api_key(self.config)
        if api_key is None:
            return None
        # Lazy import (CONF-D4/§8, load-bearing): `conflict_judgment` is the sole
        # module-scope `import anthropic` in the conflict pipeline. Importing it here keeps
        # the SDK off the `mitos sync` import path whenever the sensor is inactive.
        import anthropic
        from mitos.conflict_judgment import (_JUDGMENT_MODEL_ALIAS,
                                             make_judgment_executor)

        client = anthropic.Anthropic(api_key=api_key)
        return make_judgment_executor(
            client, model_id=get_model_id(_JUDGMENT_MODEL_ALIAS, self.config.env)
        )

    def _run_and_surface_conflict(
        self, entry: ParsedEntry, judge: Callable, run: "_ConflictSyncRun"
    ) -> None:
        """Runs the conflict check for one decision entry, surfaces + persists it (5a/5b).

        The sync-time surface of the Conflict sensor. Called under the caller's gates
        (decision-kind, ``not auto_accept``, a terminal on stdin, toggle on, judge
        available) immediately before the accept prompt, so a high-confidence contradiction
        is named at the moment the author can still choose (CONF-D7). The TTY gate is the
        loop's own report-and-skip refusal, which sits above the kind split and therefore
        above this call: a run with no terminal can accept nothing, so it judges nothing.
        Advisory only: it prints, applies no verb, writes nothing to the graph, and
        **never blocks the commit**.

        Contract (load-bearing, §8): no conflict-path outcome may abort a real decision
        commit. 5b gives the surface its memory and its failure manners:

        * **Aggregate breaker (P7, vision §4).** If a prior entry's check tripped the run's
          breaker, this returns immediately — a true no-op (no gather, no query, no judge),
          so a single downstream outage costs ONE penalty for the run, not N.
        * **Loud degradation notice.** A typed :class:`~mitos.conflict.Unavailable` (the
          sensor's downstream deps went dark) now prints one calm ``[Conflict sensor
          unavailable]`` notice and trips the breaker for the rest of the run (write-then-
          read: set here, read at the top of the next entry's check). The commit proceeds
          untouched (fail-open, CONF-D10).
        * **Best-effort telemetry.** A healthy JUDGED result (``execution is not None``)
          persists one ``judgment_batches`` row + N ``conflict_checks`` rows; the write is
          decoupled from surfacing and never aborts the sync (D5).

        A genuine local graph-store fault propagates past the facade (2a's D4
        ``DatabaseError``/``ValidationError``), so the whole call is wrapped defensively and
        any exception is logged to stderr, commit proceeds — matching this file's best-effort
        bulkhead idiom. This generic seam does NOT trip the breaker (it is a rare local fault,
        not a downstream-dep outage — IMPL_NOTES 5a, D3).

        Args:
            entry: The proposed decision entry — passed raw, so the facade reads the parsed
                declarations only (RF-1; structural since the slug-collision override that
                rewrote them was retired).
            judge: The bound judgment executor from :meth:`_build_conflict_judge`.
            run: The per-run :class:`_ConflictSyncRun` context (sync_run_id + telemetry +
                breaker), built once per sync and threaded through every entry's check.

        Returns:
            None. Output is the printed finding block or a degradation notice; nothing on a
            clean / tenable check (P9 quiet success — only a gated finding or a degradation
            speaks).
        """
        # Aggregate breaker gate (5b, §8): a prior entry already saw the sensor's downstream
        # go dark, so skip the whole check — including the disclosure below (harmless: it
        # already fired on the tripping entry or earlier). One penalty, not N (P3/P7).
        if run.breaker_tripped:
            return

        # First-fire activation disclosure (CONF-D5/P15): reached only after every gate has
        # passed, so the notice never prints when the sensor is inactive (no key, toggle
        # off, `--yes`).
        self._disclose_conflict_sensor_once()

        try:
            result = run_conflict_check(
                entry,
                embed_provider=self.embed_provider,
                vector_store=self.vector_store,
                store=self.store,
                judge=judge,
            )
        except Exception as exc:
            # Bulkhead (§8/D3): a graph-store fault propagates past the facade (2a D4). Log
            # and never block the commit — but do NOT trip the breaker (this is a local
            # fault, not a downstream-dep outage; the breaker is for the typed Unavailable).
            print(
                f"[Warning] Conflict check failed for '{entry.slug}': {exc}",
                file=sys.stderr,
            )
            return

        if isinstance(result, Unavailable):
            # Degraded check (5b): a loud one-time notice + trip the aggregate breaker so
            # every later entry skips the check. The commit proceeds exactly as if the
            # sensor were off (fail-open, CONF-D10); no row is persisted for a degradation.
            self._notice_conflict_unavailable(result.reason)
            run.breaker_tripped = True
            return

        # Healthy result: only surfaced (gated) findings speak (P9). Clean-empty or tenable
        # → `findings == []` → print nothing.
        if result.findings:
            self._print_conflict_findings(result.findings)
        # The gather was cut above the check's floor: say so beside the findings, on
        # the same channel. Data, never a degradation — the breaker stays untouched.
        if result.gather_cut:
            print(_CONFLICT_GATHER_CUT_LINE)

        # 5b persistence: a JUDGED result (the LLM fired) writes its judged pairs to the
        # telemetry corpus — the surfaced findings AND the silent-but-judged negative labels
        # (CONF-D8). Clean-empty (`execution is None`) is a healthy novel entry, not a
        # judged batch — it writes nothing (the canonical DoD-2 discriminator). Best-effort:
        # a telemetry failure never aborts the sync (D5).
        if result.execution is not None:
            self._persist_conflict_batch(result, run)

    def _new_conflict_run(self) -> "_ConflictSyncRun":
        """Mints the per-run conflict context for one sync (5b): id + telemetry + breaker.

        Called ONCE per ``perform_sync`` at the judge-build site, under the same
        availability guard as :meth:`_build_conflict_judge` — so no telemetry store is
        constructed when the sensor is off, and the single returned instance is what the
        write-then-read aggregate breaker threads through the loop (§8). The
        :class:`~mitos.telemetry.TelemetryStore` construction is best-effort: it boots its
        own ladder against ``config.telemetry_path`` and, on any failure, the run keeps a
        ``None`` telemetry handle — the sensor still surfaces findings, it simply can't
        persist them (a persistence outage must never disable the advisory surface, P7).

        Returns:
            A fresh :class:`_ConflictSyncRun` with a minted ``sync_run_id``, a (possibly
            ``None``) telemetry store, and ``breaker_tripped=False``.
        """
        sync_run_id = uuid.uuid4().hex
        telemetry: Optional[TelemetryStore] = None
        try:
            telemetry = TelemetryStore(self.config.telemetry_path)
        except Exception as exc:
            # Best-effort: a telemetry store that can't boot must not disable the surface.
            print(
                f"[Warning] Conflict telemetry unavailable ({exc}); "
                f"judged checks will not be persisted this sync.",
                file=sys.stderr,
            )
            telemetry = None
        return _ConflictSyncRun(sync_run_id=sync_run_id, telemetry=telemetry)

    def _persist_conflict_batch(
        self, result: ConflictCheckResult, run: "_ConflictSyncRun"
    ) -> None:
        """Persists one judged conflict batch to the telemetry corpus (5b, best-effort).

        Maps a healthy JUDGED :class:`~mitos.conflict.ConflictCheckResult` to its
        ``(JudgmentBatch, [ConflictCheckRow])`` and hands it to 1b's atomic writer. Stamps
        ``surface='sync'`` on every row (this writer IS the sync surface, CHK-D7) and the
        batch's ``model_id`` resolved from ``execution.model_alias`` at call time. Every
        fed-context field is read off the result's :class:`~mitos.conflict.JudgeInput`\\ s
        (``result.proposal_input`` / ``pair.candidate_input``) — exactly what the judge saw,
        VERBATIM — never a re-read of the node (which would risk drift and hit the 2a
        ``core_axiom`` key-name gotcha). Empty proposal ``rejected_paths``/scope and empty
        candidate scope serialize to ``NULL`` (MI-9); the NOT-NULL ``candidate_rejected_paths``
        is stored raw (even the degenerate ``""``).

        Best-effort bulkhead (D5): the write is a separate step after the print, wrapped so a
        telemetry failure (a wrapped ``DatabaseError`` from the writer, or a raw ``TypeError``
        a bad Python value dies on in ``to_params`` before SQLite sees it — IMPL_NOTES 1b)
        logs a stderr ``[Warning]`` and returns. It never aborts the sync and never trips the
        breaker (the breaker is for the *sensor's* downstream, not the corpus store).

        Args:
            result: A healthy judged result — the caller guards ``execution is not None``, so
                ``result.execution`` and ``result.judged_pairs`` are populated here.
            run: The per-run context carrying ``sync_run_id`` (stamped on every row) and the
                (possibly ``None``) telemetry store.
        """
        if run.telemetry is None:
            return
        try:
            execution = result.execution
            # CHK-D3: resolve the versioned model id HERE — moments after the call,
            # against the same resolved environment the call itself used
            # (``self.config.env``, 2c) — without widening the frozen executor
            # boundary. The map, not a pre-resolved id: ``execution.model_alias``
            # exists only inside this loop.
            # Deliberately narrower than the batch's best-effort wrap: an unknown
            # alias degrades to NULL (the column is provenance-only), never to a
            # lost batch whose rationale is non-regenerable (M8).
            try:
                model_id: Optional[str] = get_model_id(
                    execution.model_alias, self.config.env
                )
            except ValueError:
                model_id = None
            batch = JudgmentBatch(
                batch_id=execution.batch_id,
                model_id=model_id,
                token_input=execution.token_input,
                token_output=execution.token_output,
                token_cache_read=execution.token_cache_read,
                token_cache_creation=execution.token_cache_creation,
                elapsed_ms=execution.elapsed_ms,
                stop_reason=execution.stop_reason,
            )
            proposal = result.proposal_input
            rows: List[ConflictCheckRow] = []
            for pair in result.judged_pairs:
                candidate_input = pair.candidate_input
                rows.append(
                    ConflictCheckRow(
                        batch_id=execution.batch_id,
                        sync_run_id=run.sync_run_id,
                        # This writer IS the sync surface — stamped explicitly on
                        # every row, never left to the schema DEFAULT (CHK-D7).
                        surface="sync",
                        judged_axiom=proposal.axiom,
                        # MI-9: an empty proposal rejected_paths ("") / scope ([]) is NULL,
                        # never "" — the NULL column already expresses "nothing here".
                        proposal_rejected_paths=proposal.rejected_paths or None,
                        proposal_scope=", ".join(proposal.scope) or None,
                        proposed_hash_if_any=result.proposed_hash_if_any,
                        candidate_slug=pair.candidate.slug,  # verbatim, no casefold
                        candidate_hash=pair.candidate.node["id"],  # M2 content hash, not slug
                        # NOT NULL (M5 guarantees a committed decision has it): store the raw
                        # str, even the degenerate "" — never coerce to None here.
                        candidate_rejected_paths=candidate_input.rejected_paths,
                        candidate_scope=", ".join(candidate_input.scope) or None,
                        tenable=pair.judgment.tenable_together,
                        confidence=pair.judgment.confidence,
                        surfaced=pair.surfaced,
                        candidate_source=CONFLICT_CANDIDATE_SOURCE,
                        model_alias=execution.model_alias,
                        prompt_version=CONFLICT_PROMPT_VERSION,
                        mitos_version=MITOS_VERSION,
                        rationale=pair.judgment.rationale,
                    )
                )
            # One MI-10 UTC stamp per batch, shared by every row (D5).
            self._record_conflict_batch(run.telemetry, batch, rows)
        except Exception as exc:
            # Best-effort (D5/§8): a mapping or write failure never aborts the sync. Mirror
            # the file's `[Warning] ...: {exc}` stderr idiom (the embed/drain/rotation
            # siblings). No breaker trip — the corpus store is not the sensor's downstream.
            print(
                f"[Warning] Could not persist conflict telemetry: {exc}",
                file=sys.stderr,
            )

    def _record_conflict_batch(
        self,
        telemetry: TelemetryStore,
        batch: JudgmentBatch,
        rows: List[ConflictCheckRow],
    ) -> None:
        """Stamps one MI-10 UTC timestamp and writes the batch (the sole telemetry write).

        A thin seam over :meth:`~mitos.telemetry.TelemetryStore.record_judged_batch` that
        takes the single ``created_at`` stamp (MI-10: application-supplied UTC ISO-8601, one
        per batch, shared by every row — never ``CURRENT_TIMESTAMP``). Kept separate from
        :meth:`_persist_conflict_batch`'s mapping so a test can monkeypatch the write in
        isolation and so the one wall-clock read has a single home.

        Args:
            telemetry: The run's telemetry store (non-``None`` — the caller guards it).
            batch: The per-batch metrics row.
            rows: The judged candidate rows.
        """
        telemetry.record_judged_batch(batch, rows, _utc_now_iso())

    def _notice_conflict_unavailable(self, reason: ConflictUnavailableReason) -> None:
        """Prints the loud one-time degradation notice for a typed Unavailable (5b, §7).

        The surface's disposition of a typed :class:`~mitos.conflict.Unavailable`: a calm,
        plain-text notice that (a) names WHICH subsystem went dark — switching on ``reason``
        (embedding/vector-store → semantic recall; judgment/timeout → the judge) — (b) states
        the breaker consequence (conflict checking is off for the rest of this sync), and
        (c) reassures that the commit is unaffected (fail-open, CONF-D10). All conflict UX
        wording lives HERE in the surface (P7 core/surface bulkhead); ``mitos.conflict``
        returns the typed reason and ``result.detail`` is logging-only, NEVER rendered.

        A tag distinct from 5a's ``[Conflict]`` (finding) and ``[Conflict sensor active]``
        (disclosure) — ``[Conflict sensor unavailable]`` — so the three surfaces don't blur.
        Plain ASCII, no emoji, no required colour (P9 A11y).

        Args:
            reason: The typed :class:`~mitos.conflict.ConflictUnavailableReason` from the
                degraded result — the machine-readable discriminator the surface words.
        """
        if reason is ConflictUnavailableReason.COLLECTION_MISSING:
            # Inside the semantic-substrate bucket, but worded apart: the vector store
            # DID respond, so "did not respond" would be the missing-as-unreachable
            # conflation, on the surface an operator meets most often.
            what = (
                "Semantic recall is unavailable (the vector collection is missing — run "
                "`mitos reconcile`)"
            )
        elif reason in SEMANTIC_SUBSTRATE_REASONS:
            what = (
                "Semantic recall is unavailable (the vector store or embedding service did "
                "not respond)"
            )
        elif reason is ConflictUnavailableReason.JUDGMENT_TRUNCATED:
            what = (
                "Conflict judgment is unavailable (the judge's response was truncated at "
                "max_tokens)"
            )
        elif reason is ConflictUnavailableReason.JUDGMENT_TIMEOUT:
            what = (
                "Conflict judgment is unavailable (the judgment model did not respond in "
                "time)"
            )
        elif reason is ConflictUnavailableReason.JUDGMENT_REJECTED:
            what = (
                "Conflict judgment is unavailable (the judgment API rejected the request — "
                "check the Anthropic key and any MITOS_MODEL_OVERRIDE_SONNET)"
            )
        else:
            what = (
                "Conflict judgment is unavailable (the judgment batch was malformed)"
            )
        print(
            f"\n[Conflict sensor unavailable] {what}. Conflict checking is skipped for the\n"
            "  rest of this sync; your decisions still commit normally."
        )

    def _disclose_conflict_sensor_once(self) -> None:
        """Prints the conflict-sensor activation notice once per project (CONF-D5).

        The ``conflict_check_on_sync`` toggle defaults on, so an upgrade silently enables a
        feature that can spend Claude-tier tokens. Per P15 / progressive activation, disclose
        it the first time it actually fires: if the ``.mitos/.conflict_disclosed`` sentinel
        is absent, print a calm one-time notice (what it does, the tier + rough cost, how to
        turn it off) and create the sentinel. A **presence-only** marker, zero content —
        derivative state, safe to delete (the notice simply re-fires, harmlessly).
        """
        sentinel = os.path.join(self.config.mitos_dir, ".conflict_disclosed")
        if os.path.exists(sentinel):
            return
        print(
            "\n[Conflict sensor active] mitos now checks each synced decision against its\n"
            "  close neighbours in the graph and, before you accept, names any active one it\n"
            "  may contradict without linking to. It uses the Claude judgment tier (roughly a\n"
            "  few cents only for an entry with a contradiction-suspect neighbour; most cost\n"
            "  nothing — the similarity floor screens them out first). It only advises: it\n"
            "  never blocks a commit and writes nothing. To turn it off, set\n"
            "  `conflict_check_on_sync = false` in .mitos/config.toml."
        )
        try:
            with open(sentinel, "w", encoding="utf-8") as f:
                f.write("")
        except Exception as exc:
            # A derivative marker; if it can't be written the notice re-fires next run
            # (harmless). Never let it block the sensor or the commit.
            print(
                f"[Warning] Could not write conflict-disclosure sentinel: {exc}",
                file=sys.stderr,
            )

    def _print_conflict_findings(self, findings: List[ConflictFinding]) -> None:
        """Renders surfaced conflict findings as a calm plain-text block (5a, §7).

        For each surfaced finding, prints the conflicting decision's slug + similarity, its
        Letter fields (axiom, rejected_paths, scope) and any modifier stamps, then the
        judgment rationale — the "why" the two may not both stand (the payload describes the
        *candidate*; the rationale describes the *tension*). The existing accept prompt runs
        unchanged afterwards.

        All conflict/notice wording lives HERE, in the surface (P7 core/surface bulkhead):
        :mod:`mitos.conflict` returns structured findings and never prints. Plain text, ASCII
        markers, no emoji, no required colour (P9 A11y). Wording is calm and non-accusatory
        ("may contradict"): the calibrated floor cannot screen a legitimate scoped narrows
        carve-out (4b), so a false positive is possible — and accept-anyway (the unchanged
        prompt) is the one-keystroke backstop.

        Args:
            findings: The gated (surfaced) findings from a healthy ``ConflictCheckResult``.
        """
        plural = "decisions it does not link to" if len(findings) > 1 else \
            "decision it does not link to"
        print(f"\n[Conflict] This decision may contradict an active {plural}:")
        for finding in findings:
            payload = finding.payload
            print(f"  -> {payload['slug']}   (similarity {payload['score']:.2f})")
            print(f"       Axiom:    {payload['axiom']}")
            # rejected_paths rides a non-brief finding payload; guard anyway (a brief
            # payload would omit it). Modifier stamps ride CONDITIONALLY — a key is present
            # only when non-empty (blind indexing KeyErrors on the common unmodified
            # candidate), so guard every optional key (mirrors 2b's own copy guard).
            if "rejected_paths" in payload:
                print(render_rejected_paths(payload["rejected_paths"], "       Rejected: "))
            scope = payload.get("scope") or []
            scope_text = ", ".join(scope) if scope else "(global — no scope declared)"
            print(f"       Scope:    {scope_text}")
            for key in ("superseded_by", "amended_by", "narrowed_by", "corrected_by"):
                if key in payload:
                    print(f"       ({key.replace('_', ' ')}: {', '.join(payload[key])})")
            print("     Why they may not both stand:")
            print(f"       {finding.rationale}")
        print(
            "  Accept commits this decision as-is (unlinked). To resolve, skip and add a\n"
            "  relationship (Supersedes: / Amends: / Narrows: / Contradicts:) in\n"
            "  decisions.md, then re-sync. Quit to abort."
        )

    def _best_effort_embed(
        self, delta: CommitDelta, entry: ParsedEntry, *, rejection_leg: bool = True
    ) -> Optional[List[float]]:
        """Best-effort inline embedding of one committed node into both indexes.

        Every path that indexes a committed node calls this: ``record_decision``,
        ``amend_commentary``, and ``mitos sync``'s commit loop and quarantine
        fixpoint. Two legs run in order, each a bulkhead for the other:

        1. :meth:`_embed_decision_inline`, the decision index (the axiom's point,
           through the shared embedding cache). Gated by ``_collection_absent``.
        2. :meth:`_write_rejection_items_inline`, the rejection index (the items
           the commit's ``rejected_paths`` states, uncached). Gated by
           ``_rejection_collection_absent`` and ``_rejection_leg_stopped``.

        The rejection leg runs whatever the decision leg returned: the decision
        collection being absent or its write failing costs that leg alone. The
        axiom's leg goes first, so the post-commit gather reads exactly what it
        read before the rejection leg existed. Neither leg raises.

        ``mitos sync``'s commit loop and fixpoint pass ``rejection_leg=False``: their
        rows wait for the same command's :meth:`drain_rejection_backlog`, which
        batches across owners rather than spending a call per entry (ADR
        ``sync-commit-loop-leaves-rejection-rows-to-its-own-drain-record-and-amend-write-inline``).
        ``record_decision`` and ``amend_commentary`` keep the default, because a
        workspace written only over MCP has no drain until its next ``sync``.

        Args:
            delta: The commit's delta.
            entry: The ``ParsedEntry`` that commit stored.
            rejection_leg: Whether to run the rejection leg inline.

        Returns:
            The decision leg's document vector (a caller may reuse it for a
            neighbour query), or None if that leg deferred or failed.
        """
        vector = self._embed_decision_inline(delta, entry)
        if rejection_leg:
            self._write_rejection_items_inline(delta, entry)
        return vector

    def _embed_decision_inline(
        self, delta: CommitDelta, entry: ParsedEntry
    ) -> Optional[List[float]]:
        """The decision leg of the inline embed: the axiom's point (C2).

        The committing node is already enqueued on the ``pending_embeddings`` Outbox
        by ``commit_parsed_entry`` (``_enqueue_outbox``, 5c), so this is the inline
        fast path: with the provider up we embed + upsert immediately and DROP the
        now-redundant Outbox row (the node is indexed); with it down or the call
        failing we leave the row for the next ``sync`` drain — never a second enqueue
        (the commit already wrote one; the prototype's deferred ``add_pending_embedding``
        is retired here, 8a). Returns the document vector it computed and upserted (so
        a caller can reuse it for a neighbour query), or None if embedding was
        deferred/failed.

        This is a **single-node** write, so it may create an absent collection only when
        that one node *is* the whole of what the decision index should hold — the
        fresh-project carve-out. The declaration comes from the graph gate
        (:meth:`~mitos.store.GraphStore.decision_index_expects_node_other_than`), the
        twin of the expected-set read the read-side gate uses; a populated graph — one
        holding only retired decisions included — defers instead,
        keeping the collection absent so the honest "couldn't check" notice keeps firing
        rather than quietly becoming "checked, clean" over a one-point index.
        """
        if self._collection_absent:
            # Already reported once for this command: say nothing, spend nothing, and
            # skip the round trip. The outbox row the commit wrote is the durable
            # record of the deferral either way.
            return None

        embed_text = _embedding_input_text(
            kind=entry.kind, axiom=entry.axiom,
            topic=entry.topic, questions_raised=entry.questions_raised,
        )

        if not self.embed_provider or not self.vector_store:
            # Already enqueued by the commit; just note the deferral. stderr — this
            # path is shared with the MCP write tool, whose stdout is the JSON-RPC
            # channel (a stray stdout line there corrupts the protocol).
            print(f"[Warning] Embedding upsert deferred for '{entry.slug}': Embedding provider down.",
                  file=sys.stderr)
            return None

        # Prepare payload. ``state`` is an inert literal no reader trusts: state is
        # computed (M3), a retired decision's point carries "active" too, and a stored
        # state would go stale on the next kill edge. Do not "fix" it.
        payload = {
            "slug": entry.slug,
            "scope": entry.scope,
            "state": "active",
            "kind": entry.kind,
            "embedding_text": embed_text
        }

        try:
            # Declared intent, from the graph gate: this write covers the decision index's
            # expected set iff the graph expects no other node. Cheap enough to run
            # unconditionally (an EXISTS … LIMIT 1), and it must be computed before the
            # write because only the 404 reveals whether it matters. Deliberately INSIDE
            # this try: the
            # helper's contract is that it swallows its own faults and leaves the outbox
            # row (that is what makes deferral cheap), and a graph read added outside it
            # would let a store fault escape a path that has never raised.
            may_create = not self.store.decision_index_expects_node_other_than(delta.node_id)
            # Check embedding provider and generate vector
            vector = self.embed_provider.get_embedding(payload["embedding_text"], is_query=False)
            self.vector_store.upsert(delta.node_id, vector, payload, may_create=may_create)
            # Indexed now — drop the Outbox row the commit enqueued.
            try:
                self.store.remove_pending_embedding(delta.node_id)
            except Exception as dbe:
                print(f"[Warning] Failed to clear outbox row: {str(dbe)}", file=sys.stderr)
            return vector
        except CollectionMissingError:
            # Named as itself, not as an outage: Qdrant is up, the index is not there,
            # and this write is not the one that may build it. One line per command
            # (the latch), and the outbox row the commit wrote stays put — so the work
            # is queued and `mitos reconcile` completes it in one pass. stderr, for the
            # same JSON-RPC reason as the provider-down twin above.
            self._collection_absent = True
            print(
                f"[Warning] Embedding upsert deferred for '{entry.slug}': the Qdrant "
                f"collection is missing and this write does not cover every decision "
                f"the index should hold. "
                f"Queued — run `mitos reconcile` to rebuild the index.",
                file=sys.stderr,
            )
            return None
        except Exception as e:
            # The commit already enqueued this node (C2); leave the row for the next
            # drain. stderr — shared with the MCP write tool's JSON-RPC stdout channel.
            print(f"[Warning] Embedding upsert deferred for '{entry.slug}': {describe_embed_failure(e)}", file=sys.stderr)
            return None

    def _write_rejection_items_inline(
        self, delta: CommitDelta, entry: ParsedEntry
    ) -> None:
        """Adds the items a commit's ``rejected_paths`` states to the rejection index.

        The inline half of the rejection index's lifecycle (recall-confidence 4b). It
        runs only when the commit queued a rejection backlog row, and it only
        **adds**: it never removes, never retries and never notes the row (ADR
        ``rejection-index-inline-writes-only-add-and-only-one-drain-at-a-time-removes``).

        - The items are read from ``entry``, the field this commit stored, and the
          only seq it may clear is ``delta.rejection_enqueue_seq``, the one this
          commit minted. That (seq, field) pair is one snapshot; a row read back
          after the commit could hold a newer seq a concurrent re-queue minted.
        - **Licence**: it may create the absent collection only when its decision is
          the graph's only decision, live or retired
          (``rejection_index_expects_owner_other_than``).
        - **Presence**: a new owner holding the licence asks nothing (its ids derive
          from an owner that is new). Every other write asks once, by the ids its
          field determines: an unlicensed new owner learns that the collection exists
          before it embeds anything, and an amend embeds only what is not present
          (ADR ``inline-rejection-write-checks-presence-by-item-id-and-a-new-owner-asks-nothing``).
          An unlicensed write that finds the collection absent latches and returns
          silently, having embedded nothing.
        - **One call**: at most one ``embed_documents`` call, holding ``fit_batch``'s
          prefix of the items not present, in enumeration order. It never touches
          the shared embedding cache. Any overflow waits in the row for the drain.
        - **Clear**: a new owner whose every item landed clears its own row against
          its seq (a ``False`` answer means a newer commit re-queued it, which the
          drain converges). An amend never clears: the items its field dropped are
          still in the index, and only a drain removes.

        Every fault leaves the row and costs this leg alone: the commit, the
        receipt and the decision leg are untouched. An outage (``auth``, ``quota``
        or ``transient`` provider cause, or an unreachable store) latches the leg
        off for the rest of the command; any other fault costs one owner. A failed
        write prints one stderr line naming the selectored ``sync`` whose drain
        retries the row (stdout is the MCP JSON-RPC channel); the unlicensed
        absent-collection probe stays silent (``status`` states that gap).

        Args:
            delta: The commit's delta (``created``, ``rejection_enqueue_seq``).
            entry: The ``ParsedEntry`` that commit stored.
        """
        seq = getattr(delta, "rejection_enqueue_seq", None)
        if not isinstance(seq, int) or isinstance(seq, bool):
            return
        if self.embed_provider is None or self.rejection_index is None:
            return
        if self._rejection_collection_absent or self._rejection_leg_stopped:
            return
        may_create = False
        try:
            points = rejection_points(delta.node_id, entry.rejected_paths or "")
            if not points:
                # An amend to an empty field: nothing to add. The row stays for a
                # drain to remove the old items.
                return
            ids = list(points)
            may_create = not self.store.rejection_index_expects_owner_other_than(
                delta.node_id
            )
            if delta.created and may_create:
                present: AbstractSet[str] = frozenset()
            else:
                try:
                    present = self.rejection_index.retrieve_present(ids)
                except CollectionMissingError:
                    if not may_create:
                        self._rejection_collection_absent = True
                        return
                    present = frozenset()
            missing = [pid for pid in ids if pid not in present]
            if missing:
                model = self.embed_provider.model_id
                if not isinstance(model, str) or not model:
                    raise EmbeddingError(
                        "the embedding provider reported no model id"
                    )
                texts = [points[pid].text for pid in missing]
                n = fit_batch(texts)
                vectors = self.embed_provider.embed_documents(texts[:n])
                if not (
                    isinstance(vectors, list)
                    and len(vectors) == n
                    and all(isinstance(v, list) and v for v in vectors)
                ):
                    raise EmbeddingError(
                        "the embedding provider answered a document batch with the "
                        "wrong shape"
                    )
                self.rejection_index.upsert_points(
                    [
                        RejectionPoint(pid, delta.node_id, vector)
                        for pid, vector in zip(missing[:n], vectors)
                    ],
                    model=model,
                    may_create=may_create,
                )
                landed = n == len(missing)
            else:
                landed = True
            if delta.created and landed:
                try:
                    self.store.clear_rejection_backlog_row(
                        delta.node_id, enqueue_seq=seq
                    )
                except Exception as exc:
                    # The items landed; only the clear failed. Say so, rather than
                    # the not-indexed line, and leave the row (a drain finds every
                    # item present and clears it). One owner, no latch.
                    self._warn_rejection_row_not_cleared(entry, exc)
        except CollectionMissingError as exc:
            if not may_create:
                # Deleted between the probe and the write: the steady-state absence,
                # silent as the probe's. The batch is spent and lost (bounded to one).
                self._rejection_collection_absent = True
                return
            # The create ran and the retry still answered missing: the endpoint is
            # broken for this write, a refusal of one owner, not a latch.
            self._warn_rejection_items_not_indexed(entry, exc)
        except VectorStoreUnreachableError as exc:
            self._rejection_leg_stopped = True
            self._warn_rejection_items_not_indexed(entry, exc)
        except EmbeddingError as exc:
            if classify_provider_error(exc) in (AUTH, QUOTA, TRANSIENT):
                self._rejection_leg_stopped = True
            self._warn_rejection_items_not_indexed(entry, exc)
        except Exception as exc:
            # A refusal (``VectorStoreError``), a store fault, a caller's
            # ValueError: one owner's leg, no latch. A racing create's 409 is not
            # among them: the client reads it as "exists" and writes.
            self._warn_rejection_items_not_indexed(entry, exc)

    def _warn_rejection_items_not_indexed(
        self, entry: ParsedEntry, exc: BaseException
    ) -> None:
        """Prints the rejection leg's one failure line on stderr.

        It names the selectored ``sync``: a failure arises only where a write was
        attempted, so the collection exists or the write was licensed, and the next
        ``sync``'s rejection drain does act on the row. stderr only, never an MCP
        payload — the channel the decision leg's ``reconcile`` line already uses.
        """
        try:
            print(
                f"[Warning] Rejected items for '{entry.slug}' were not indexed: "
                f"{describe_embed_failure(exc)}. Its rejection backlog row stays queued; "
                f"the next `mitos sync {shell_flag('-p', self.config.project)}` "
                f"retries it.",
                file=sys.stderr,
            )
        except Exception:
            pass

    def _warn_rejection_row_not_cleared(
        self, entry: ParsedEntry, exc: BaseException
    ) -> None:
        """Prints the line for a write that landed and a clear that failed.

        Like :meth:`_warn_rejection_items_not_indexed`, it names the selectored
        ``sync``, whose drain finds every item present and clears the row.
        """
        try:
            print(
                f"[Warning] Rejected items for '{entry.slug}' were indexed, but its "
                f"rejection backlog row could not be cleared: "
                f"{describe_embed_failure(exc)}. The row stays queued; the next "
                f"`mitos sync {shell_flag('-p', self.config.project)}` clears it.",
                file=sys.stderr,
            )
        except Exception:
            pass

    def drain_pending_embeddings(
        self, *, listing: Optional[AbstractSet[str]] = None
    ) -> DrainOutcome:
        """Drains the decision outbox (``pending_embeddings``, C2) in one walk.

        This is the decision drain; "the drain" below means it. The rejection
        backlog has a drain of its own, :meth:`drain_rejection_backlog`, with its
        own lock and its own seed marker.

        **The walk.** The outbox is read in queue order, ``(queued_at, node_id)``, a
        claim-sized page at a time, each page keyed after the last row walked. The
        read is a plain ordered ``SELECT``, not a claim: two concurrent drainers can
        both process a row, which costs an idempotent upsert and at worst a second
        embed. Each row standing at entry is attempted once per walk. A row that
        fails for a reason of its own (the provider classifier's ``unknown``, or a
        fault after the embed) keeps its place and gains one retry, and the walk
        moves on to the rows behind it, so no head of failing rows can stop a later
        drain (ADR ``backlog-drains-never-let-a-failing-or-held-row-block-the-rows-behind-it``).
        A row enqueued mid-walk is re-stamped ahead of the cursor and taken in the same
        walk. An **outage** stops the walk at once: the embed call failing ``auth``,
        ``quota`` or ``transient``, or the vector store failing the write with anything
        but a missing collection. The row the outage hit still gains its retry, exactly
        as every failure always has; the gain is that an outage charges one row, not
        every row behind it. There is no backoff: the next drain is the retry.

        **The seed marker.** Whether this drain may create an absent collection is not a
        property of the call site: ``--embed-only`` is a drain with no buffer at all, so
        its covering-ness is outbox state a prior command established. The
        ``embedding_seed`` marker carries it — stamped by ``rebuild``/``cutover``'s
        prune and by ``reconcile``'s enqueue. It has two readers: this drain, which acts
        on it, and ``mitos status``, which prints it. It is read **once at entry**, so a
        concurrent drainer cannot flip it mid-walk.

        Under a standing marker the drain lists the collection's point ids once, and a
        row whose point is already present is removed without embedding or reading its
        node: a content-derived point id means presence is the right vector under the
        same model (ADR ``mass-seed-drain-skips-present-points-under-the-seed-marker-an-ordinary-drain-re-upserts``).
        The residual is known: a per-commit row queued for a present node while a
        marker stands is skipped too, so a renamed decision's payload slug waits for
        that node's next commit. With no marker nothing is listed and every row is
        embedded and upserted, which is what heals such a payload.

        The marker is **spent** when the drain finds its backlog empty, or when its walk
        completes with the collection present (the listing found it, or a write in this
        drain created it). A walk that completes with the collection still absent spends
        it only if the backlog reads empty afterwards. An outage, a refusal, a failed
        read or an exception escaping the walk **keeps** it: spending is never in a
        ``finally``. Only the marker this drain read is spent; one re-stamped while it
        ran survives (ADR ``seed-marker-is-spent-once-a-drain-has-attempted-every-row-standing-at-its-entry``).

        It is emphatically *not* an inferred comparison of outbox size against the
        decision index's expected set — that is not a completeness measure (the outbox
        can hold rows for nodes that have left the graph), it is more expensive than the
        signals it would replace, and it decays silently once a migration hands every
        project a clean collection.

        Args:
            listing: A point-id listing of a **present** collection that the caller has
                already taken (``reconcile_embeddings``, which just listed it to diff).
                Consulted only when a seed marker stands at entry, and then it replaces
                the walk's own listing, exactly as a normal listing return would — so a
                reconcile scrolls the collection once, not twice. With no marker it is
                ignored. A caller that met ``CollectionMissingError`` passes nothing, so
                this walk lists for itself and the absent-collection spend rule applies
                untouched; a handed listing must never fake a present collection.

        Returns:
            The walk's :class:`DrainOutcome`. ``reconcile_embeddings`` reports from it;
            ``sync`` ignores it.
        """
        if not self.embed_provider or not self.vector_store:
            print("Cannot drain outbox: Embedding provider or vector store down.")
            return DrainOutcome(0, 0, 0, 0, stop=DRAIN_STOP_UNAVAILABLE, marker=None)

        drainer_id = f"drainer-{uuid.uuid4()}"

        # Declared intent, from the state a prior command established (read once).
        seed = self.store.embedding_seed()
        may_create = seed is not None

        written = skipped = pruned = failed = 0

        def outcome(stop: Optional[str], marker: Optional[str]) -> DrainOutcome:
            # Every exit after a skip says how many rows it skipped: a skip prints no
            # per-row line, so an outage after hundreds of skips would otherwise hide them.
            if skipped:
                print(
                    f"Skipped {skipped} already-indexed row(s): the embedding queue was "
                    f"seeded by `{seed['established_by']}`."
                )
            return DrainOutcome(written, skipped, pruned, failed, stop=stop, marker=marker)

        kept = DRAIN_MARKER_KEPT if seed is not None else None

        try:
            try:
                pending = self.store.claim_pending_embeddings(
                    drainer_id, limit=_DRAIN_CLAIM_SIZE
                )
            except Exception as e:
                print(f"[Warning] Failed to claim outbox queue: {str(e)}")
                return outcome(DRAIN_STOP_CLAIM_FAILED, kept)

            if not pending:
                # Nothing to deliver: whatever coverage the marker claimed is moot, so
                # it is spent. Never lists the collection.
                if seed is None:
                    return outcome(None, None)
                return outcome(None, self._spend_embedding_seed(seed))

            # Under the marker, one listing for the whole walk — the caller's, when it
            # handed one (always of a present collection), else our own. A normal return
            # means the collection is present, whatever it holds — an empty set included.
            present: AbstractSet[str] = set()
            collection_present = False
            if seed is not None:
                try:
                    present = self.vector_store.list_point_ids() if listing is None else listing
                    collection_present = True
                except CollectionMissingError:
                    present = set()
                except VectorStoreError as e:
                    print(
                        f"Stopping the drain before any row: the vector store could not "
                        f"list the collection ({str(e)}). The queue is unchanged; the "
                        f"next `mitos sync` retries."
                    )
                    return outcome(DRAIN_STOP_OUTAGE, kept)

            print("Draining pending embeddings queue ...")

            while pending:
                for item in pending:
                    node_id = item["node_id"]

                    if seed is not None and hash_to_uuid(node_id) in present:
                        # Presence-skip: before `get_node`, so a skipped row costs no
                        # graph read and no provider call. A present point whose node
                        # has left the graph is skipped too; the point is an orphan
                        # for the collector, not for this drain.
                        try:
                            self.store.remove_pending_embedding(node_id)
                            skipped += 1
                        except Exception as e:
                            failed += 1
                            self._increment_pending_best_effort(node_id)
                            print(f"[Warning] Failed to clear an already-indexed outbox row: {str(e)}")
                        continue

                    # Fetch node details from graph for Qdrant payload
                    node = self.store.get_node(node_id)
                    if not node:
                        # The node left the graph. Close to unreachable in production:
                        # `pending_embeddings.node_id` is a foreign key to `nodes`, so
                        # only a store opened with foreign keys off can leave this row.
                        try:
                            self.store.remove_pending_embedding(node_id)
                            pruned += 1
                        except Exception:
                            pass
                        continue

                    # Re-derive the embedding text from the node's immutable core (the
                    # Outbox row no longer carries it — C2/M8); byte-identical to what the
                    # inline record-time embed used for the same node.
                    embed_text = _embedding_input_text(
                        kind=node["kind"], axiom=node.get("core_axiom"),
                        topic=node.get("topic"), questions_raised=node.get("questions_raised"),
                    )

                    # ``state`` is an inert literal no reader trusts (M3): a retired
                    # decision's point carries "active" too. Do not "fix" it.
                    payload = {
                        "slug": node["slug"],
                        "scope": node["scope"],
                        "state": "active",
                        "kind": node["kind"],
                        "embedding_text": embed_text
                    }

                    # The embed call has a try of its own: only ITS exception is
                    # classified (a store or graph error's text can carry "429").
                    try:
                        vector = self.embed_provider.get_embedding(embed_text, is_query=False)
                    except Exception as e:
                        cause = classify_provider_error(e)
                        failed += 1
                        self._increment_pending_best_effort(node_id)
                        print(f"[Warning] Failed to drain embedding for '{node['slug']}': {describe_embed_failure(e)}")
                        if cause in _DRAIN_OUTAGE_CAUSES:
                            print(
                                f"Stopping the drain: the embedding provider is unavailable "
                                f"({cause}). The rows behind this one stay queued for the "
                                f"next `mitos sync`."
                            )
                            return outcome(DRAIN_STOP_OUTAGE, kept)
                        continue

                    try:
                        self.vector_store.upsert(
                            node_id, vector, payload, may_create=may_create
                        )
                    except CollectionMissingError:
                        # Ahead of the VectorStoreError arm (it is a subclass), and
                        # deliberately quiet: a refusal is a property of the whole drain,
                        # not of this row, so it costs ONE calm line and stops
                        # immediately. And NO `increment_pending_attempts` — the row was
                        # not rejected, the drain declined; inflating a counter that
                        # exists to describe genuine failure would poison it.
                        print(
                            "Stopping the drain: the Qdrant collection is missing and no "
                            "covering re-embed has been requested, so writing now would "
                            "index only part of the corpus. Run `mitos reconcile` to "
                            "rebuild it in one pass."
                        )
                        return outcome(DRAIN_STOP_REFUSED, kept)
                    except VectorStoreError as e:
                        # A store outage: the point id, vector and payload are well-formed
                        # by construction, so a fault on the write belongs to the store,
                        # and trying the next row would only spend another embedding.
                        failed += 1
                        self._increment_pending_best_effort(node_id)
                        print(f"[Warning] Failed to drain embedding for '{node['slug']}': {describe_embed_failure(e)}")
                        print(
                            "Stopping the drain: the vector store failed the write. The "
                            "rows behind this one stay queued for the next `mitos sync`."
                        )
                        return outcome(DRAIN_STOP_OUTAGE, kept)
                    except Exception as e:
                        failed += 1
                        self._increment_pending_best_effort(node_id)
                        print(f"[Warning] Failed to drain embedding for '{node['slug']}': {describe_embed_failure(e)}")
                        continue

                    # Written: the collection exists now, whatever the listing said.
                    collection_present = True
                    try:
                        self.store.remove_pending_embedding(node_id)
                    except Exception as e:
                        # The point is written; the row stays and the next drain
                        # re-upserts it (a cache hit).
                        failed += 1
                        self._increment_pending_best_effort(node_id)
                        print(f"[Warning] Failed to drain embedding for '{node['slug']}': {describe_embed_failure(e)}")
                        continue
                    written += 1
                    print(f"Successfully drained embedding for '{node['slug']}' ✓")

                cursor = (pending[-1]["queued_at"], pending[-1]["node_id"])
                try:
                    pending = self.store.claim_pending_embeddings(
                        drainer_id, limit=_DRAIN_CLAIM_SIZE, after=cursor
                    )
                except Exception as e:
                    print(f"[Warning] Failed to claim outbox queue: {str(e)}")
                    return outcome(DRAIN_STOP_CLAIM_FAILED, kept)
        finally:
            # Clean up: release any remaining locks held by this specific drainer
            try:
                self.store.release_pending_embeddings(drainer_id)
            except Exception:
                pass

        if seed is None:
            return outcome(None, None)
        if collection_present:
            return outcome(None, self._spend_embedding_seed(seed))
        # The collection is still absent (nothing written). Spend only if nothing is
        # left to cover — read, never inferred from `failed == 0`: a row queued behind
        # the cursor mid-walk is invisible to the walk.
        try:
            empty = self.store.claim_pending_embeddings(drainer_id, limit=1) == []
        except Exception as e:
            print(f"[Warning] Failed to claim outbox queue: {str(e)}")
            return outcome(None, DRAIN_MARKER_KEPT)
        if empty:
            return outcome(None, self._spend_embedding_seed(seed))
        return outcome(None, DRAIN_MARKER_KEPT)

    def _increment_pending_best_effort(self, node_id: str) -> None:
        """Moves a failed row's retry count, swallowing a store fault (as always)."""
        try:
            self.store.increment_pending_attempts(node_id)
        except Exception:
            pass

    def _spend_embedding_seed(self, seed: Dict[str, str]) -> str:
        """Spends the marker the drain read, and reports which way it went.

        Args:
            seed: The marker as the drain read it at entry.

        Returns:
            ``DRAIN_MARKER_SPENT``, or ``DRAIN_MARKER_KEPT`` when the clear failed.
        """
        if self._clear_embedding_seed_best_effort(established_at=seed["established_at"]):
            return DRAIN_MARKER_SPENT
        return DRAIN_MARKER_KEPT

    def _clear_embedding_seed_best_effort(self, *, established_at: Optional[str] = None) -> bool:
        """Clears the coverage marker, reporting rather than raising on failure.

        The clear is bookkeeping on a delivered claim, not part of the drain's product,
        so a graph fault here must not turn a fully-drained outbox into a failed
        ``sync``. Failing to clear is also the *safe* direction — the marker only ever
        authorizes a covering drain, and the next one re-reads the outbox and spends it
        again.

        Args:
            established_at: The stamp of the marker read at the drain's entry; the
                delete spends only that marker, so a re-stamp made meanwhile survives.

        Returns:
            True when the delete ran, False when it failed.
        """
        try:
            self.store.clear_embedding_seed(established_at=established_at)
        except Exception as e:
            print(f"[Warning] Failed to clear the embedding seed marker: {str(e)}")
            return False
        return True

    def drain_rejection_backlog(self, *, report: bool = True) -> RejectionDrainOutcome:
        """Drains the rejection backlog in one walk: each owner's items replaced whole.

        The only writer that removes from the rejection index (ADR
        ``rejection-index-inline-writes-only-add-and-only-one-drain-at-a-time-removes``).
        It never raises: every exit returns an outcome, an escaped fault included
        (worded, ``stop="unavailable"``, the marker kept). It is silent and returns
        ``stop="unavailable"`` when the provider or the rejection index is None: the
        decision drain has already said so on the same output.

        **One at a time.** The walk runs under a non-blocking operating-system lock of
        its own, :func:`rejection_drain_lock_path`, taken before any read and released
        when the walk ends or its process dies: never a lease in the graph, never the
        ``decisions.md`` lock, never taken by an inline write. A drain that finds it
        held returns ``held`` having read and written nothing, leaving the backlog to
        the holder (ADR ``rejection-backlog-hold-dies-with-its-writer-and-the-record-path-never-waits-on-it``).
        A held lock file a killed holder left behind is acquired over, never unlinked.

        **The walk.** Rows are read in ``enqueue_seq`` order, a page at a time, keyed
        after the last seq walked, so each row standing at entry is attempted once.
        Per row: the seq is read (the claim), **then** the owner's raw field; the
        expected items are the field's ``rejection_points``; what the owner holds is
        listed; the items it lacks are written (those held are skipped and counted),
        and only once every one has landed are the items its field no longer states
        deleted by point id, and the row cleared against the seq read. A clear that
        answers False means a commit re-queued the owner; its new row lies ahead of
        the cursor, so this walk takes it again, listing afresh, and converges it on
        the latest field (ADR ``rejection-drain-retakes-an-owner-it-left-standing-only-overlapping-inline-writes-wait``).
        An interrupted owner keeps old and new items, never none.

        **Batches and failure.** The page's texts are cut into cross-owner batches by
        ``fit_batch``, one ``embed_documents`` call and one upsert each, and an owner
        is finalized as soon as its last item lands. An ``auth``, ``quota`` or
        ``transient`` embed cause, an unreachable store, or a listing the store
        refused stops the drain at once (``outage``) and charges no row. An absent
        collection with no marker stops it (``refused``), creating and embedding
        nothing. A batch failing for any other reason is retried owner by owner, and
        a solo failure notes that owner's row once against its seq and aborts the
        owner for this take (its tail is dropped, nothing of it is removed or
        cleared), while the walk goes on (ADRs ``backlog-drains-never-let-a-failing-or-held-row-block-the-rows-behind-it``,
        ``rejection-drain-charges-a-row-only-for-its-own-failure-never-for-an-outage-or-refusal``).

        **The marker** (``rejection_seed``, never the decision index's) is read once
        at entry. Only a standing marker licenses creating the collection, and it
        makes the drain list the whole index once, with owners, instead of each
        owner; a row queued after that listing is listed afresh. It is spent by 3a's
        rule: on an empty backlog, or when the walk completes with the collection
        present, or completes with it absent and a fresh read finds no row; an
        outage, a refusal, a read fault or an escaped exception keeps it, and only
        the marker this drain read is spent (ADR
        ``seed-marker-is-spent-once-a-drain-has-attempted-every-row-standing-at-its-entry``).
        Its four producers are the operator's covering acts, never a read, a record or
        a check (``migrations.REJECTION_SEED_ESTABLISHED_BY``): ``reconcile``, when it
        queued an owner the index lacked; ``rebuild`` and ``cutover``, in the aside
        transaction beside the decision marker; and an import into a graph that held no
        decision. Without one, an absent collection makes the drain refuse, naming
        ``mitos reconcile``.

        Args:
            report: Print the summary lines (:func:`rejection_drain_report`) after
                the walk. ``reconcile`` passes False and its renderer prints them.

        Returns:
            The walk's :class:`RejectionDrainOutcome`, with ``left_queued`` and
            ``owners_past_tolerance`` read at exit.
        """
        if self.embed_provider is None or self.rejection_index is None:
            return self._rejection_drain_unavailable()

        lock = FileLock(rejection_drain_lock_path(self.config), timeout=0)
        try:
            lock.acquire()
        except Timeout:
            # Ahead of OSError, which Timeout subclasses.
            print(
                "Rejection index: another drain holds the rejection backlog; this one "
                "leaves it to that drain."
            )
            return RejectionDrainOutcome(
                0, 0, 0, 0, 0, None, 0, None,
                stop=REJECTION_DRAIN_STOP_HELD, marker=None,
            )
        except OSError as e:
            # Not another drain's hold: the lock could not be taken at all.
            print(
                f"[Warning] The rejection backlog was not drained: its lock could not "
                f"be taken ({e})."
            )
            return self._rejection_drain_unavailable()

        walk = _RejectionDrainWalk(self)
        try:
            try:
                stop, marker = walk.run()
            except Exception as e:
                print(
                    f"[Warning] The rejection drain stopped: {describe_embed_failure(e)}. "
                    f"Its rows stay queued."
                )
                stop, marker = DRAIN_STOP_UNAVAILABLE, walk.kept()
            left, past = self._rejection_backlog_counts()
        finally:
            lock.release()

        outcome = RejectionDrainOutcome(
            walk.owners_cleared, walk.items_embedded, walk.items_skipped_present,
            walk.items_removed, walk.owners_failed, past, walk.calls, left,
            stop=stop, marker=marker,
        )
        if report:
            for line in rejection_drain_report(outcome.to_dict()):
                print(line)
        return outcome

    def _rejection_backlog_counts(self) -> Tuple[Optional[int], Optional[int]]:
        """Reads the standing rows and those past the tolerance; None when unreadable."""
        try:
            left: Optional[int] = self.store.count_rejection_backlog()
            past: Optional[int] = self.store.count_rejection_backlog(
                min_retry=REJECTION_BACKLOG_FAILURE_TOLERANCE
            )
        except Exception:
            left = past = None
        return left, past

    def _rejection_drain_unavailable(self) -> RejectionDrainOutcome:
        """The outcome of a drain that did not run, its two counts read."""
        left, past = self._rejection_backlog_counts()
        return RejectionDrainOutcome(
            0, 0, 0, 0, 0, past, 0, left, stop=DRAIN_STOP_UNAVAILABLE, marker=None,
        )

    def reconcile_embeddings(self) -> Dict[str, Any]:
        """Heals both indexes: embeds what each is missing, drains, then collects what is dead.

        Heals the gap an ordinary drain cannot reach: when Qdrant is wiped directly
        (a bare ``curl -X DELETE`` of the collection) WITHOUT a ``rebuild``/
        ``cutover`` — which re-seed the outbox — the graph has nodes, Qdrant has
        none, and the outbox is empty, so ``sync`` drains nothing. This pass diffs
        the decision index's **expected set**
        (:meth:`~mitos.store.GraphStore.get_decision_index_expected_ids`: every
        decision, live or retired, plus the active open questions) against Qdrant's
        actual point ids, enqueues every missing node, and drains. It is the same
        read ``mitos status`` reports completeness against, so the heal clears its
        own warning by construction.

        A retired decision is seeded like a live one: its vector lives exactly as
        long as its node, and the read surfaces partition it by computed state (ADR
        ``graveyard-is-deliberate-every-decision-vector-lives-as-long-as-its-node``).
        Bounded to one paginated scroll + a set difference (no per-node probes), so
        it stays cheap at the P11 horizon, and the drain is handed that same listing
        rather than scrolling again. Idempotent: a second run finds nothing missing
        and enqueues nothing. Re-queued decision nodes hit the embedding cache, so
        the decision half costs no embedding-API calls when the text is unchanged;
        the rejection index keeps no cache (below).

        An **absent collection reads as empty**, not as an error: it is the very
        state this heal exists for (a wipe, or a rename that left the old name
        behind), so every expected node is missing and every one gets enqueued —
        the drain's first upsert then creates the collection. Only genuine
        unreachability raises.

        **The collection** (the index's death story, P4). After the drain the graph
        is read — active ids, then every node id — and the listing is partitioned
        (:func:`~mitos.vector_store.partition_points`, the one partition ``status``
        applies too): points of retired nodes are history and kept, killed open
        questions included; points whose node is not in the graph are garbage and
        are deleted by their listed id (ADR
        ``reconcile-heals-every-decision-vector-and-collects-only-node-less-points``).
        The graph read comes **after** the listing: a write commits its node before
        it upserts its point, so every listed point's node is committed by then, and
        a record landing mid-collection keeps its point. The delete runs in chunks of
        ``_ORPHAN_DELETE_CHUNK``; a refused chunk is counted in ``delete_failed`` and
        the next is tried, while an unreachable Qdrant or a missing collection stops
        the loop and counts every unconfirmed id as failed. A foreign point written
        into this workspace's collection by another workspace sharing it is node-less
        here and is deleted too — accepted, because nothing deleted is beyond a
        re-seed.

        **Over a graph holding no nodes, nothing is judged and nothing is deleted.**
        Every listed point would look node-less there (a clone without its gitignored
        graph file), so a heal that trusted that would wipe the collection of every
        such workspace at once — P5's cascade. The guard keys on the **node count**,
        never on :func:`~mitos.recall.missing_graph_is_a_gap`: that predicate also
        asks the corpus, and ``corpus_holds_entries`` reads an unreadable file as
        empty, so a zero-node graph beside an unreadable ``decisions.md`` would read
        as "built and empty" and lose every point. The predicate answers
        ``graph_unbuilt`` instead, so ``cmd_reconcile`` names ``mitos rebuild``
        exactly when ``status``' unbuilt rung does.

        **The rejection index** (R1–R5, after the decision heal, each step in a
        bulkhead of its own). Its expected set is computable from the graph alone —
        every decision's ``rejected_paths``, enumerated by the one leaf
        (:func:`~mitos.rejections.expected_rejection_points`) — and it is diffed by
        point id (:func:`~mitos.rejections.diff_rejection_index`), the same diff
        ``status`` words, so the heal clears its own warning. R1 reads the owners'
        fields, R2 lists the whole index once with owners (absent reads as empty),
        R3 queues every owner lacking an item (a standing row is never touched) and
        stamps ``rejection_seed("reconcile")`` when anything is missing, R4 is the
        rejection drain (it lists for itself and, under that marker, creates an
        absent collection and writes every item), and R5 deletes the listing's
        stale items (an owner that no longer states them) and orphaned ones (an
        owner not in the graph, or unnamed) — judged against a graph read taken
        after the listing, under the drain's lock, and never over a graph holding
        no node (:meth:`_collect_rejection_index`). The items bypass the embedding
        cache: a first build or a lost collection pays one embedding per item, the
        missing count ``status`` prices. With no provider or no rejection index,
        nothing of it runs, silently.

        Precedence for the partition keys, first match wins (the same rule
        ``status`` applies): the graph pair could not be read, or the provider-down
        return → ``None``/``None``; an empty listing (absent collection, or no
        point) → ``0``/``0`` on any graph; a non-empty listing over a graph holding
        no nodes → ``None``/``None``; otherwise the partition's counts.

        Returns:
            The shipped keys ``active`` (active-node count), ``present`` (the
            collection's point count, retired and orphan points included) and
            ``enqueued`` (nodes this reconcile enqueued), plus ``expected`` (the
            expected set's size), ``expected_present`` (``expected - enqueued``:
            the part already indexed), ``written`` and ``drain_stop`` (the drain's
            :class:`DrainOutcome` ``written`` / ``stop``; ``0`` / ``None`` when
            nothing was missing and no drain ran) and ``left_queued`` (the outbox
            rows standing when this returns, read — never inferred from the drain's
            counters, which count only rows attempted). Then the collection's:
            ``retired_node_points`` and ``nodeless_points`` (the listing's history
            and garbage buckets, ``None`` when not judged — see the precedence
            above); ``deleted`` (node-less points whose delete Qdrant confirmed);
            ``delete_failed`` (node-less points refused or unconfirmed — still
            there, and the next reconcile retries them); and ``graph_unbuilt``
            (``missing_graph_is_a_gap``'s answer, ``status --json``'s
            ``checks.graph_unbuilt``). The partition counts describe the listing
            this call judged, not the collection after its writes, so
            ``retired_node_points`` plus the retired points ``written`` here is
            roughly ``status``' next count. Last, ``rejection_drain``: the
            rejection backlog's drain, :meth:`RejectionDrainOutcome.to_dict`
            (``stop="unavailable"`` with its counts read when the drain did not
            run or raised), and beside it ``rejection_index``: ``expected_items``,
            ``expected_owners``, ``present_items``, ``collection_present``,
            ``missing_items`` (R3's count: what this reconcile set out to embed),
            ``owners_queued``, ``seed_stamped``, ``stale_items`` and
            ``orphan_items`` (the listing as R5 judged it; None unless
            ``judged``), ``deleted``, ``delete_failed`` and ``collected`` (one of
            :data:`REJECTION_COLLECT_OUTCOMES`); a count not computed is None.
            Every return path carries the same key set.

        Raises:
            VectorStoreError: If Qdrant is unreachable while scrolling point ids.
                A *missing collection* does not raise (see above), and no delete
                fault escapes: each is counted in ``delete_failed``. That raise
                carries no result, so no ``rejection_drain`` either: it happens
                before either drain runs.
        """
        if not self.embed_provider or not self.vector_store:
            # stderr, not stdout: `mitos reconcile --json` promises stdout carries
            # exactly one JSON object, and this line is `as_json`-blind — it broke
            # `json.loads(stdout)` on every provider-down workspace. Threading a
            # presentation flag down into the manager was the alternative; a
            # diagnostic on the diagnostic channel keeps the routing decision at
            # the boundary, where the rest of it already lives.
            print(
                "Cannot reconcile: Embedding provider or vector store down.",
                file=sys.stderr,
            )
            # Nothing was read or done, so the counts are 0 — except ``left_queued``,
            # which is a read of the standing outbox on every path: rows queued before
            # this call still stand, and 0 would tell the operator nothing is waiting.
            return {
                "active": 0, "present": 0, "enqueued": 0,
                "expected": 0, "expected_present": 0,
                "written": 0, "left_queued": len(self.store.get_pending_embeddings()),
                "drain_stop": None,
                # Nothing was listed, so nothing was judged or collected. The unbuilt
                # answer is still computed: it is a graph-and-corpus read, and a
                # keyless clone is still a clone.
                "retired_node_points": None, "nodeless_points": None,
                "deleted": 0, "delete_failed": 0,
                "graph_unbuilt": self.is_graph_unbuilt(),
                # Not drained (no provider); its two counts are still read.
                "rejection_drain": self._rejection_drain_unavailable().to_dict(),
                "rejection_index": self._rejection_index_unavailable(),
            }

        expected_ids = self.store.get_decision_index_expected_ids()
        # The live surface, for the shipped ``active`` count only — never what to seed.
        active_count = len(self.store.get_active_node_ids())
        listed = False
        try:
            present = self.vector_store.list_point_ids()
            listed = True
        except CollectionMissingError:
            # Narrow by design: a connection-level VectorStoreError must keep
            # propagating so `mitos reconcile` can still report Qdrant down.
            present = set()
        missing = [nid for nid in expected_ids if hash_to_uuid(nid) not in present]

        for node_id in missing:
            self.store.add_pending_embedding(node_id)

        written = 0
        drain_stop: Optional[str] = None
        if missing:
            # The enqueue above is the act that makes the outbox cover the decision
            # index's expected set, so record it before draining: the drain reads the
            # marker, not a flag, and stamping it durably buys crash-safety for free —
            # a reconcile that dies mid-drain leaves the marker standing, and the
            # operator's next `sync` completes the heal instead of refusing it. This is
            # one of only three producers, all operator-invoked: nothing implicit ever
            # stamps it, which is what keeps the heal operator-invoked rather than
            # automatic.
            self.store.stamp_embedding_seed("reconcile")
            # Hand over the listing only when it is one of a PRESENT collection. Over
            # a 404 the drain lists for itself (a cheap 404), so 3a's absent-collection
            # spend rule sees the truth rather than a faked present collection.
            drained = self.drain_pending_embeddings(listing=present if listed else None)
            written = drained.written
            drain_stop = drained.stop

        # The rejection index, R1–R3: measure it against the graph and queue every
        # owner lacking an item (stamping the marker), in a bulkhead of its own — a
        # fault leaves nothing queued or stamped and the decision heal untouched.
        # Silent with no index (the drain is silent there too).
        rejection_index = self._rejection_index_unavailable()
        rejection_listing: Optional[Dict[str, Optional[str]]] = None
        enumerated: Dict[Tuple[str, str], Tuple[str, ...]] = {}
        if self.rejection_index is not None:
            try:
                rejection_listing = self._measure_rejection_index(rejection_index, enumerated)
            except Exception as e:
                print(
                    f"[Warning] The rejection index could not be measured against "
                    f"the graph: {describe_embed_failure(e)}",
                    file=sys.stderr,
                )
                rejection_index["collected"] = REJECTION_COLLECT_UNAVAILABLE

        # R4, the rejection backlog's drain, in a bulkhead of its own: a fault in it
        # leaves the decision heal and the collection below untouched, and the key
        # still lands (an unavailable outcome, its counts read). Before the graph
        # read below, which judges only the decision collection's points. It lists
        # for itself: its watermark needs its own read before its own listing.
        try:
            rejection = self.drain_rejection_backlog(report=False)
        except Exception as e:
            print(
                f"[Warning] Rejection backlog drain failed: {describe_embed_failure(e)}",
                file=sys.stderr,
            )
            rejection = self._rejection_drain_unavailable()

        # R5, the rejection index's collection, after the drain and in its own
        # bulkhead: only over a listing R2 actually made.
        if rejection_listing is not None:
            try:
                self._collect_rejection_index(rejection_index, rejection_listing, enumerated)
            except Exception as e:
                print(
                    f"[Warning] The rejection index's collection stopped: {e}",
                    file=sys.stderr,
                )
                rejection_index["collected"] = REJECTION_COLLECT_UNAVAILABLE

        # The collection: the graph pair is read AFTER the listing (and the drain),
        # so every listed point's node is committed by now — a write commits its node
        # before it upserts — and a record landing mid-collection keeps its point.
        # Active first, then every node; the partition intersects them, so a rebuild
        # swap between the two reads cannot make an active node look node-less.
        node_ids: Optional[AbstractSet[str]] = None
        try:
            active_ids = self.store.get_active_node_ids()
            node_ids = self.store.get_all_node_ids()
        except Exception as e:
            # The seed above already ran; a graph fault here leaves the collection
            # undone (not judged), it does not fail the heal.
            print(
                f"[Warning] Could not read the graph to judge the index's points; "
                f"nothing was collected: {e}",
                file=sys.stderr,
            )
        retired_node_points: Optional[int] = None
        nodeless_points: Optional[int] = None
        deleted = 0
        delete_failed = 0
        if node_ids is None:
            pass  # not judged: the graph pair could not be read
        elif not present:
            # An empty listing is judged and empty on any graph: no point to judge.
            retired_node_points = nodeless_points = 0
        elif node_ids:
            # The guard: only over a graph holding nodes. Over a zero-node graph every
            # listed point would look node-less, so it is left not judged instead.
            partition = partition_points(present, node_ids=node_ids, active_ids=active_ids)
            retired_node_points = len(partition.retired_node)
            nodeless_points = len(partition.nodeless)
            deleted, delete_failed = self._delete_nodeless_points(partition.nodeless)

        return {
            "active": active_count,
            "present": len(present),
            "enqueued": len(missing),
            "expected": len(expected_ids),
            "expected_present": len(expected_ids) - len(missing),
            "written": written,
            # Read, never inferred: ``failed`` counts only rows attempted (an outage on
            # row 1 of 33 reports 1 while 33 stand), and ``enqueued - written`` misses
            # per-commit rows that were queued before this reconcile.
            "left_queued": len(self.store.get_pending_embeddings()),
            "drain_stop": drain_stop,
            "retired_node_points": retired_node_points,
            "nodeless_points": nodeless_points,
            "deleted": deleted,
            "delete_failed": delete_failed,
            "graph_unbuilt": self.is_graph_unbuilt(),
            "rejection_drain": rejection.to_dict(),
            "rejection_index": rejection_index,
        }

    def is_graph_unbuilt(self) -> bool:
        """Answers ``status``' unbuilt-graph question, with ``status``' own predicate.

        Returns:
            True when the graph holds no node while the corpus holds entries.
        """
        return missing_graph_is_a_gap(
            self.store, self.config, corpus_scan=corpus_holds_entries
        )

    def _delete_nodeless_points(self, nodeless: AbstractSet[str]) -> Tuple[int, int]:
        """Deletes the decision collection's node-less points; counts confirmed and failed.

        Args:
            nodeless: The listed ids whose node is not in the graph.

        Returns:
            ``(deleted, delete_failed)``.
        """
        return self._delete_points_chunked(
            self.vector_store, nodeless, noun="node-less point"
        )

    @staticmethod
    def _delete_points_chunked(
        client: Any, point_ids: AbstractSet[str], *, noun: str
    ) -> Tuple[int, int]:
        """Deletes listed points by id, chunk by chunk; counts confirmed and failed.

        The one stop/continue rule for both indexes' collections. A chunk refused or
        left unconfirmed (plain ``VectorStoreError``) is counted failed and the next
        chunk is tried. An unreachable Qdrant or a missing collection stops the loop,
        because every later chunk would meet the same fault; every id not yet
        confirmed is counted failed. Nothing else is caught.

        Args:
            client: The collection's client (``delete_points(ids)``).
            point_ids: The listed ids to delete.
            noun: What a point is, singular, for the warnings.

        Returns:
            ``(deleted, delete_failed)``.
        """
        ids = sorted(point_ids)
        size = _ORPHAN_DELETE_CHUNK
        deleted = 0
        failed = 0
        for start in range(0, len(ids), size):
            chunk = ids[start:start + size]
            try:
                client.delete_points(chunk)
            except (VectorStoreUnreachableError, CollectionMissingError) as e:
                failed += len(ids) - start
                print(
                    f"[Warning] Stopped deleting {noun}s; "
                    f"{len(ids) - start} not deleted: {e}",
                    file=sys.stderr,
                )
                break
            except VectorStoreError as e:
                failed += len(chunk)
                print(
                    f"[Warning] {len(chunk)} {noun}(s) not deleted: {e}",
                    file=sys.stderr,
                )
                continue
            deleted += len(chunk)
        return deleted, failed

    @staticmethod
    def _rejection_index_unavailable() -> Dict[str, Any]:
        """``reconcile --json``'s ``rejection_index`` when nothing was measured.

        Counts not computed are None; nothing was queued, stamped or deleted.
        """
        return {
            "expected_items": None, "expected_owners": None, "present_items": None,
            "collection_present": None, "missing_items": None, "owners_queued": 0,
            "seed_stamped": False, "stale_items": None, "orphan_items": None,
            "deleted": 0, "delete_failed": 0,
            "collected": REJECTION_COLLECT_UNAVAILABLE,
        }

    def _measure_rejection_index(
        self, index: Dict[str, Any], memo: Dict[Tuple[str, str], Tuple[str, ...]]
    ) -> Dict[str, Optional[str]]:
        """Reconcile's R1–R3: reads the owners, lists the index, queues the missing.

        Fills ``index`` in place as each count is computed, so a fault part-way
        leaves only what was really read. The owner fields are read **before** the
        listing, for the missing count only: an item written between the two is
        listed, so it is not queued, and an owner that changes between them is
        queued anyway and the drain converges it. Every owner lacking an item is
        queued (never a standing row touched), and the rejection seed marker is
        stamped (``reconcile``) only when something is missing — the act that
        makes the backlog cover the expected set, and so the one that licenses the
        drain to create an absent collection.

        Args:
            index: The ``rejection_index`` dict being built.
            memo: The command's enumeration memo, shared with R5.

        Returns:
            The listing (``{}`` for an absent collection).

        Raises:
            Exception: Any graph, listing or write fault; the caller reports it.
        """
        owner_fields = self.store.get_rejection_index_owner_fields()
        expected = expected_rejection_points(owner_fields, memo)
        index["expected_items"] = len(expected)
        index["expected_owners"] = len(set(expected.values()))
        try:
            listing = self.rejection_index.list_point_owners()
            index["collection_present"] = True
        except CollectionMissingError:
            listing = {}
            index["collection_present"] = False
        index["present_items"] = len(listing)
        diff = diff_rejection_index(expected, listing, owners=owner_fields.keys())
        index["missing_items"] = len(diff.missing)
        if diff.missing:
            index["owners_queued"] = self.store.enqueue_rejection_backlog_owners(
                sorted(diff.missing_owners)
            )
            self.store.stamp_rejection_seed("reconcile")
            index["seed_stamped"] = True
        return listing

    def _collect_rejection_index(
        self,
        index: Dict[str, Any],
        listing: Dict[str, Optional[str]],
        memo: Dict[Tuple[str, str], Tuple[str, ...]],
    ) -> None:
        """Reconcile's R5: deletes the listing's stale and orphaned items, under the drain lock.

        The listing is judged against a graph read taken **after** it (and after
        the drain), so an item a record or amend writes mid-collection was never
        listed and is never judged, and a listed item its owner still states at
        this read is expected and kept. Precedence, first match wins: an empty
        listing is judged and empty on any graph (no lock, no second read — every
        first build); otherwise the rejection drain's operating-system lock is
        taken non-blocking (held → ``held``, nothing deleted; not takeable →
        ``unavailable``); under it a graph holding no node is ``unbuilt`` (every
        item would look ownerless — the decision collection's guard, keyed on the
        node count); otherwise the listing is judged and its stale and orphaned
        items are deleted by id. Every remover takes that one lock (ADR
        ``rejection-index-removers-share-the-drain-lock-reconcile-collects-under-it``),
        so a concurrent ``sync`` drain cannot interleave between this read and its
        deletes. A graph fault under the lock is ``unread``. Re-deleting an item
        the drain already removed is harmless: a delete of an absent id completes.

        Args:
            index: The ``rejection_index`` dict being built, filled in place.
            listing: R2's listing (``listing0``).
            memo: The command's enumeration memo, shared with R1.
        """
        if not listing:
            index.update(stale_items=0, orphan_items=0, collected=REJECTION_COLLECT_JUDGED)
            return
        collect_lock = FileLock(rejection_drain_lock_path(self.config), timeout=0)
        try:
            collect_lock.acquire()
        except Timeout:
            # Ahead of OSError, which Timeout subclasses.
            index["collected"] = REJECTION_COLLECT_HELD
            return
        except OSError as e:
            print(
                f"[Warning] The rejection index was not collected: the rejection "
                f"backlog's lock could not be taken ({e}).",
                file=sys.stderr,
            )
            index["collected"] = REJECTION_COLLECT_UNAVAILABLE
            return
        try:
            try:
                node_count = len(self.store.get_all_node_ids())
                owner_fields = (
                    self.store.get_rejection_index_owner_fields() if node_count else {}
                )
                expected = expected_rejection_points(owner_fields, memo)
            except Exception as e:
                print(
                    f"[Warning] Could not read the graph to judge the rejection index's "
                    f"items; nothing was deleted from it: {e}",
                    file=sys.stderr,
                )
                index["collected"] = REJECTION_COLLECT_UNREAD
                return
            if node_count == 0:
                index["collected"] = REJECTION_COLLECT_UNBUILT
                return
            diff = diff_rejection_index(expected, listing, owners=owner_fields.keys())
            index["stale_items"] = len(diff.stale)
            index["orphan_items"] = len(diff.orphaned)
            index["deleted"], index["delete_failed"] = self._delete_points_chunked(
                self.rejection_index, diff.stale | diff.orphaned, noun="rejected item"
            )
            index["collected"] = REJECTION_COLLECT_JUDGED
        finally:
            collect_lock.release()

    # --- record_decision: the write half of the MCP server (Fork A) ---

    def _exact_slug_node(self, slug: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Resolves a slug to an EXACT-match node.

        ``resolve_slug`` is now single-tier casefold-exact (no fuzzy prefix tier), so
        every id it returns already shares the casefolded slug; the per-node re-filter
        below is a defensive guard on that contract (and folds with ``str.casefold()``,
        never ``str.lower()`` — the two diverge on ``ß``/Greek, MI-9).

        Returns:
            A (node_id, node) tuple for the exact-slug node, or (None, None).
        """
        for node_id in self.store.resolve_slug(slug):
            node = self.store.get_node(node_id)
            if node and node.get("slug", "").casefold() == slug.casefold():
                return node_id, node
        return None, None

    def _validate_relation_target(
        self, relation: str, target: str, *,
        canonical_slugs: Optional[Dict[str, str]] = None,
        canonical_ids: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, str]]:
        """Validates a typed relation's target is a unique, EXACT-match decision.

        Mirrors the supersedes check (``resolve_slug`` is casefold-exact; the re-filter
        defends its contract), keeping every recorded edge pointed at a real, unambiguous
        node. Runs in Phase A — a failure returns a structured error and writes nothing.

        Args:
            relation: The relation kwarg name (for the error message), e.g. "amends".
            target: The slug the agent passed as that relation's target.
            canonical_slugs: Optional casefolded-target -> stored-slug map to record the
                resolved node's own spelling into. This is the ONLY place an
                *ungathered* declared target's canonical handle is ever computed — the
                node is already fetched here and its slug discarded — and the pause echo
                prints stored spellings, never caller ones. Absent on the callers that
                do not compose an echo.
            canonical_ids: Optional casefolded-target -> node-id map, filled from the
                same node: the id Phase A proved unique before the commit, which the
                acknowledged path's post-commit review exempts by (7b, D3).

        Returns:
            None if valid, else a structured ``{error, code}`` dict.
        """
        ids = self.store.resolve_slug(target)
        if not ids:
            return _record_error("relation_target_not_found", relation=relation, target=target)
        if _ambiguous_target(ids):
            return _record_error("relation_target_ambiguous", relation=relation, target=target)
        node = self.store.get_node(ids[0])
        if not node or node.get("slug", "").casefold() != target.casefold():
            return _record_error("relation_target_not_found", relation=relation, target=target)
        if canonical_slugs is not None:
            canonical_slugs[target.casefold()] = node["slug"]
        if canonical_ids is not None:
            canonical_ids[target.casefold()] = node["id"]
        return None

    def _review_neighbors(self, entry: ParsedEntry, declared_targets: set, *,
                          gathered_index: Optional[Dict[str, Tuple[str, float]]] = None,
                          top_k: int = CONFLICT_TOP_K,
                          gathering_sink: Optional[List[Gathering]] = None,
                          ) -> "List[Dict[str, Any]] | Unavailable":
        """Pre-commit: existing live decisions too similar to ``entry`` to ignore (P4).

        Composes the Conflict sensor's candidate stages (the same discovery `mitos
        check` uses): :func:`gather_candidates` embeds the axiom in document space,
        fills ``CONFLICT_OVERFETCH_LIMIT`` live *decisions* from one bounded KNN call
        resolved by point id — an open question or a retired point in the fetch takes
        no slot — then
        :func:`screen_candidates` drops declared/self, floors at
        ``_NEIGHBOR_REVIEW_THRESHOLD``, and truncates to ``top_k``.
        Surfacing survivors BEFORE the write is the whole point: after the commit the
        author can no longer point an amends/supersedes at them (a re-record is a no-op).

        Degradation is typed, never silent: a failed embed or KNN query returns the
        gather's :class:`Unavailable` untouched (the caller words the receipt notice
        from its reason), and graph-store faults (``DatabaseError``/``ValidationError``)
        propagate — the record call site owns the fail-open disposition. Only the
        *structural* no-providers state (a graph-only workspace, first-class healthy)
        returns a clean ``[]`` here.

        Args:
            entry: The parsed entry about to be committed.
            declared_targets: Casefolded slugs the entry already links to — its declared
                relation targets plus their transitive mutation lineage — excluded so a
                linked neighbour is not re-flagged.
            gathered_index: Optional sink the caller passes to keep this call's
                gathered live set, ``{casefolded slug: (stored slug, score)}``. The pause
                echo partitions the caller's declarations against the gathered
                candidates and the floor — the primary sets (M8) — and the screened
                return cannot answer for them: S4 drops a declared target *before* the
                floor, so a declaration that resolved a strong neighbour and one that
                fell short are indistinguishable there. A sink rather than a widened
                return type, so the eight ``patch.object`` sites binding this method by
                string stay green.
            top_k: How many screened survivors to keep. The pause uses the default,
                ``CONFLICT_TOP_K``, which is what it displays. The acknowledged path
                (:meth:`_acknowledged_fields`) passes
                ``CONFLICT_OVERFETCH_LIMIT``, the gather's 20 live candidates,
                because an acknowledgement covers every neighbour the gate would have
                held the write for, not only the ones a pause would show.
            gathering_sink: Optional sink that receives the one
                :class:`~mitos.conflict.Gathering` when the gather ran healthy, and
                nothing otherwise (no providers, a typed ``Unavailable``, or a
                raise). The caller words a cut from it; the record path's tombstones
                read its retired hits. A sink for the same reason as
                ``gathered_index``.

        Returns:
            A list of :func:`~mitos.conflict.candidate_payload` dicts for
            unreferenced high-similarity live neighbours, most similar first
            (possibly empty) — the enriched Letter shape (``slug`` / ``axiom`` /
            ``scope`` / ``score`` / ``rejected_paths``) plus any present
            ``amended_by``/``narrowed_by`` modifier stamps (this is a
            decision-read surface; an amended-but-active neighbour must not read
            as the final word) — or the :class:`Unavailable` the gather returned
            when the semantic substrate was unreachable.
        """
        if not self.embed_provider or not self.vector_store:
            return []
        gathered = gather_candidates(
            entry.axiom,
            embed_provider=self.embed_provider,
            vector_store=self.vector_store,
            store=self.store,
        )
        if isinstance(gathered, Unavailable):
            return gathered
        # Filled only once the gather is known healthy, so a cut can never ride beside
        # a review that could not run.
        if gathering_sink is not None:
            gathering_sink.append(gathered)
        if gathered_index is not None:
            # The gather resolves by point id, so the node's slug is the one handle.
            gathered_index.update(
                {c.node["slug"].casefold(): (c.node["slug"], c.score)
                 for c in gathered.live}
            )
        screened = screen_candidates(
            gathered.live,
            declared_targets=declared_targets,
            own_slug=entry.slug,
            floor=_NEIGHBOR_REVIEW_THRESHOLD,
            top_k=top_k,
        )
        return [candidate_payload(c) for c in screened]

    def _resolve_slug_typed(self, slug: str) -> List[str]:
        """``GraphStore.resolve_slug`` with its raw ``sqlite3.Error`` typed (7a).

        The pause's fail-open arm catches exactly ``DatabaseError``/``ValidationError``,
        and ``resolve_slug`` raises the raw error, so every slug read the pause's review
        makes goes through here.

        Raises:
            DatabaseError: If the read fails.
        """
        try:
            return self.store.resolve_slug(slug)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to resolve slug '{slug}': {exc}")

    def _lineage_ancestors(
        self, mutation_target_slugs: List[Optional[str]], *,
        target_ids: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, List[Dict[str, str]]]:
        """The transitive mutation ancestors of declared targets, per target (3b, 7a).

        Transitive-lineage near-duplicate suppression. When an author declares an
        ``amends``/``narrows``/``supersedes`` edge to the HEAD of a multi-link mutation
        chain, the near-dup gate (:meth:`_review_neighbors`) must not pause on a near-dup
        OLDER member reachable *through* that chain — by naming one node in a lineage the
        author has acknowledged the whole lineage, not just the node they happened to
        declare. This helper resolves each declared mutation target to its node id and
        walks :meth:`GraphStore.get_lineage` (the transitive ``supersedes`` ∪ ``amends``
        ∪ ``narrows`` ancestor walk, Phase 3a) from it. :meth:`_review_declared` reads
        the one walk twice: every ancestor's casefolded slug merges into
        ``declared_targets`` — the suppression set :meth:`_review_neighbors` passes to
        ``screen_candidates``, whose S4 stage drops candidates by casefolded membership
        (both sides casefold, matching that filter exactly) — and every ancestor's
        ``node_id`` joins the target's exemption ids for the tombstone partition, which
        compares ids because a retired decision may share its slug with a live one
        (7a). Merging
        into the set only ever *grows* suppression, so V1a's DIRECT suppression of all
        nine relation types is preserved by construction (must-not-regress, DoD #13b).

        Seed ONLY the mutation three — ``supersedes`` ∪ ``amends`` ∪ ``narrows``, which
        equals ``store._MUTATION_EDGE_FIELDS`` (the exact edge set ``get_lineage``
        walks). ``corrects`` and the other five declared relations still get DIRECT
        suppression via ``declared_targets`` but NO transitive extension: walking
        ``get_lineage`` from, say, a ``cites`` target would traverse a mutation graph
        that edge has nothing to do with — semantically wrong and over-suppressing.

        Sibling-cluster shape — design-AWARE, deliberately NOT auto-suppressed
        (Decision 4, vision §6.2). When N decisions each amend one shared root R, a new
        sibling that also amends R is near-dup to its CO-CHILDREN (R's other amenders),
        not only to R. Those co-children are R's *descendants*, NOT in ``get_lineage(R)``
        (R is *their* ancestor), so this walk does not — and must not — suppress them.
        Auto-suppressing them would mean "declaring ``amends R`` silences the gate for
        R's entire descendant cluster", hiding a GENUINE duplicate of one sibling behind
        an unrelated declaration. The suppression stays exactly that. The symmetric
        check is built as a note instead (7c): :meth:`_review_siblings` walks each
        shown live neighbour's own lineage (:meth:`GraphStore.get_lineage_ids`) and
        marks the neighbour ``descends_from_declared`` when that lineage holds a
        declared target's own id. It differs from the seam once named here in two
        ways, both because a note hides nothing: it matches the targets of all nine
        declared relations, not only the mutation three, and it is shown, never
        suppressing. It costs no read when nothing is declared. The per-neighbour
        acknowledgement that seam was paired with stays unbuilt:
        ``acknowledge_neighbors`` still acknowledges the whole draft's neighbours.

        Consumes ``get_lineage``'s loud-but-non-fatal partial-on-cycle tolerance as-is
        (Decision 5): over a corrupt graph the suppression degrades gracefully (it
        suppresses what was walked before the cycle bound fired, and ``get_lineage``
        emits the loud ``logger.warning``) and never hangs the hot ``record`` path. This
        helper adds no cycle handling of its own — that is the read-side "tolerate" half
        of 3a's one-walk-two-call-sites split.

        ``get_lineage`` and ``resolve_slug`` raise a raw ``sqlite3.Error`` on a failed
        read; both are typed as ``DatabaseError`` here, so the pause's fail-open arm
        (which catches exactly ``DatabaseError``/``ValidationError``) owns them like
        every other pause read.

        Args:
            mutation_target_slugs: The declared ``supersedes``/``amends``/``narrows``
                target slugs; any element may be ``None`` or empty (skipped).
            target_ids: Optional casefolded target → node id that Phase A validated
                (7b). A target found here seeds its walk from that id, unresolved; the
                rest resolve as before. The acknowledged path passes it, because after
                the commit a slug may name more nodes than it did before.

        Returns:
            Casefolded target slug → its ancestors (``get_lineage``'s ``{node_id, slug,
            kind}`` dicts), for every target that resolves to exactly one node; empty
            when no mutation edge was declared.

        Raises:
            DatabaseError: If the lineage read fails.
        """
        ancestors: Dict[str, List[Dict[str, str]]] = {}
        for slug in mutation_target_slugs:
            if not slug or not slug.strip():
                continue
            pinned = (target_ids or {}).get(slug.casefold())
            ids = [pinned] if pinned is not None else self._resolve_slug_typed(slug)
            # Each declared target was pre-validated unique-and-existing before the gate
            # runs (supersedes at the supersedes fast-fail, amends/narrows via
            # _validate_relation_target), so resolve_slug returns exactly one id on the
            # supported path. Defensive skip on 0/>1 keeps the helper robust if reused:
            # a non-unique seed cannot meaningfully seed a single lineage walk.
            if len(ids) != 1:
                continue
            try:
                walked = self.store.get_lineage(ids[0])
            except sqlite3.Error as exc:
                raise DatabaseError(f"Failed to read the lineage of '{slug}': {exc}")
            ancestors.setdefault(slug.casefold(), []).extend(walked)
        return ancestors

    def _review_declared(self, supersedes: Optional[str], corrects: Optional[str],
                         extra_relations: Dict[str, str], *,
                         target_ids: Optional[Mapping[str, str]] = None,
                         ) -> DeclaredReview:
        """What a draft's declarations exempt on the pause, by slug and by id (7a).

        One lineage walk (:meth:`_lineage_ancestors`) feeds the :class:`DeclaredReview`:
        the live screen's suppression slugs, and each declared target's exemption ids
        for the tombstone partition. The resolution loop before the walk also keeps
        each target's own id (``direct_ids``, 7c), which the sibling note matches. It
        reads ``self.store`` unconditionally and nothing else on the manager, because
        the replay worker and Layer C's draft reader call it on a store-only shell; the
        provider gate that keeps the graph-only fast path free of the walk is
        :meth:`_review_declared_targets`'.

        Ids, not slugs, carry the exemption: a retired decision may share its slug with
        its live bearer, so a declaration of the live ``x`` must not clear the retired
        ``x`` (constraint 8). Phase A has already refused any target that names no node
        or several, so each target resolves to exactly one id here — possibly a
        retired one (a draft may ``amends`` a uniquely-slugged retired decision), which
        then exempts that decision itself. A target resolving otherwise (a replay shell
        skips Phase A) contributes its slug and no id.

        ``target_ids`` pins targets to the ids Phase A validated (7b, D3). The pause
        passes none: before the commit each target resolves to exactly the id Phase A
        saw. The acknowledged path runs after the commit, where a slug may name more
        nodes than before (a reconcile that landed a same-slug node between this
        write's commit and its read; ``record_decision`` itself never mints a slug a
        node already carries), so it passes Phase A's ids and its exemptions cannot
        move with the commit.

        Args:
            supersedes: The raw ``supersedes`` argument.
            corrects: The raw ``corrects`` argument.
            extra_relations: The seven non-kill relations, stripped, non-empty only.
            target_ids: Optional casefolded target → validated node id; a target found
                here takes that id unresolved, and seeds its lineage walk from it.

        Returns:
            The :class:`DeclaredReview`.

        Raises:
            DatabaseError: If a slug or lineage read fails.
        """
        direct = [t for raw in ([supersedes, corrects] + list(extra_relations.values()))
                  for t in _split_relation_slugs(raw)]
        slugs = {t.casefold() for t in direct}
        ids_by_target: Dict[str, Set[str]] = {}
        direct_ids: Dict[str, str] = {}
        for target in direct:
            pinned = (target_ids or {}).get(target.casefold())
            ids = [pinned] if pinned is not None else self._resolve_slug_typed(target)
            if len(ids) == 1:
                ids_by_target.setdefault(target.casefold(), set()).add(ids[0])
                direct_ids[target.casefold()] = ids[0]
        lineage = self._lineage_ancestors(
            _split_relation_slugs(supersedes)
            + _split_relation_slugs(extra_relations.get("amends"))
            + _split_relation_slugs(extra_relations.get("narrows")),
            target_ids=target_ids,
        )
        for folded, ancestors in lineage.items():
            for ancestor in ancestors:
                slugs.add(ancestor["slug"].casefold())
                ids_by_target.setdefault(folded, set()).add(ancestor["node_id"])
        return DeclaredReview(
            slugs=slugs,
            ids_by_target=ids_by_target,
            ids=set().union(*ids_by_target.values()) if ids_by_target else set(),
            direct_ids=direct_ids,
        )

    def _review_declared_targets(self, supersedes: Optional[str],
                                 corrects: Optional[str],
                                 extra_relations: Dict[str, str]) -> set[str]:
        """Builds the neighbour review's suppression set from the raw relation args.

        The pause's no-provider branch reads it, and so does the replay worker's legacy
        build. The acknowledged path's post-commit gather (:meth:`_acknowledged_fields`)
        reads :meth:`_review_declared` directly since 7b, seeded with Phase A's target
        ids, and its own early ``None`` answers the provider gate below. Every copy is
        the one walk: two would drift, and the first drift would name a neighbour the
        caller had linked.
        The set is every declared relation target, casefolded, plus the transitive
        mutation lineage of the ``supersedes``/``amends``/``narrows`` targets (3b).
        The lineage walk is skipped without providers: the gate is then a no-op
        (:meth:`_review_neighbors` returns ``[]``), and the walk only ever grows the
        set, so direct suppression of all nine types holds either way (DoD #13b). It
        is suppression-only and re-declares nothing, so a bridged predecessor still
        reads un-amended (§6.2).

        It reads the raw args and writes nothing back: Phase B writes edges from the
        same raw args, and normalising them here would drop declared edges under
        ``acknowledge_neighbors`` (``test_mixed_neighbors_declared_edge_survives_acknowledge_bypass``).

        Args:
            supersedes: The raw ``supersedes`` argument.
            corrects: The raw ``corrects`` argument.
            extra_relations: The seven non-kill relations, stripped, non-empty only.

        Returns:
            The casefolded slugs the review must not flag. Graph-store faults from the
            lineage walk propagate; each caller owns its disposition.
        """
        if self.embed_provider and self.vector_store:
            return self._review_declared(supersedes, corrects, extra_relations).slugs
        return {
            t.casefold()
            for raw in ([supersedes, corrects] + list(extra_relations.values()))
            for t in _split_relation_slugs(raw)
        }

    def _review_tombstones(self, gathering: Gathering, declared: DeclaredReview, *,
                           shown_neighbors: Sequence[Mapping[str, Any]],
                           bound: Optional[int] = CONFLICT_TOP_K,
                           floor: Optional[float] = None,
                           bare: bool = False) -> TombstoneReview:
        """The pause's tombstone partition from one healthy gathering (7a, vision §4.9).

        A pure consumer of the gather's retired partition plus at most three reads, all
        in Phase A: it runs no gather and embeds nothing.

        * **Membership.** ``gathering.retired`` (rank order, so closest first) at or
          above the floor. When none is, nothing is read and the review is empty.
        * **Exemption, before the bound, by node id** (D4). One
          :meth:`~mitos.store.GraphStore.get_kill_chains` over every such hit. A hit
          ``T`` is exempt when some declared target's ids (``declared.ids_by_target``)
          meet ``{T.id} ∪ KillChain(T).reached``: the declared target is ``T`` or its
          lineage holds ``T``, or ``T``'s forward kill chain reaches a declared target
          or its lineage — the walk that covers ``corrects``, which ``get_lineage``
          leaves out. Those targets are ``cleared_by``.
        * **The bound.** The first ``bound`` non-exempt hits are shown, the rest
          counted in ``omitted``. ``bound=None`` shows them all.
        * **The entries.** Each is :func:`~mitos.display.retired_handle`'s handle (from
          one :meth:`~mitos.store.GraphStore.get_modifiers_map` over the shown ids,
          and the live ends of the walk already read, through
          :func:`~mitos.store.live_ends_from_chains`), plus the hit node's whole
          ``axiom`` (the agent must recognise what it re-treads), plus two facts over
          its live successors (:func:`_live_successors`: its ``live_end`` when present,
          else its ``superseded_by`` ∪ ``corrected_by``, which the live-end rule
          guarantees are live then): ``successor_on_pause``, the successors among
          ``shown_neighbors`` by casefolded slug (both sides live, and a live slug
          names its one live bearer), present only when non-empty; and
          ``successor_ambiguous: True`` when some successor's slug names several
          nodes, by the validator's own predicate (:func:`_ambiguous_target`), so only
          acknowledgement clears the tombstone. Present only when true.
        * **Bare** (7b, D2). ``bare=True`` makes each entry the handle alone — no axiom,
          no successor facts — and skips the per-successor slug reads with them. The
          acknowledged receipt names what a landed write went past; it does not
          re-open it. Membership, exemption and order do not read the flag.

        Faults are not caught: a ``DatabaseError`` from any read propagates to the
        pause's fail-open arm, which discards the live neighbours too (D3). A partly
        read tombstone partition would pass for a whole one.

        It reads ``self.store`` and nothing else on the manager, so the replay worker
        and Layer C's draft reader call it on a store-only shell.

        Args:
            gathering: The review's healthy gathering.
            declared: The draft's :class:`DeclaredReview`.
            shown_neighbors: The live neighbours the pause shows (their ``slug``).
            bound: How many tombstones to show; ``None`` for all.
            floor: The tombstone floor; ``None`` reads ``_TOMBSTONE_FLOOR`` at call
                time. The candidate sweep passes a lower floor to price a lowered one.
            bare: Entries are the handle alone, and no slug is read.

        Returns:
            The :class:`TombstoneReview`.

        Raises:
            DatabaseError: If a kill-chain, stamp or slug read fails.
        """
        if floor is None:
            floor = _TOMBSTONE_FLOOR
        above = [hit for hit in gathering.retired if hit.score >= floor]
        if not above:
            return TombstoneReview(shown=[], omitted=0, cleared_by=set())
        chains, chain_slugs = self.store.get_kill_chains([hit.node["id"] for hit in above])
        kept = []
        cleared_by: Set[str] = set()
        for hit in above:
            node_id = hit.node["id"]
            reach = {node_id, *chains[node_id].reached}
            clearing = {target for target, ids in declared.ids_by_target.items()
                        if ids & reach}
            if clearing:
                cleared_by |= clearing
                continue
            kept.append(hit)
        shown_hits = kept if bound is None else kept[:bound]
        if not shown_hits:
            return TombstoneReview(shown=[], omitted=0, cleared_by=cleared_by)
        shown_ids = [hit.node["id"] for hit in shown_hits]
        live_ends = live_ends_from_chains({i: chains[i] for i in shown_ids}, chain_slugs)
        stamps = self.store.get_modifiers_map(shown_ids)
        on_pause = {n["slug"].casefold() for n in shown_neighbors}
        ambiguity: Dict[str, bool] = {}
        shown: List[Dict[str, Any]] = []
        for hit in shown_hits:
            entry = retired_handle(hit.node, score=hit.score, stamps=stamps,
                                   live_ends=live_ends)
            if bare:
                shown.append(entry)
                continue
            entry["axiom"] = hit.node["core_axiom"]
            successors = _live_successors(entry)
            beside = [s for s in successors if s.casefold() in on_pause]
            if beside:
                entry["successor_on_pause"] = beside
            for slug in successors:
                if slug not in ambiguity:
                    ambiguity[slug] = _ambiguous_target(self._resolve_slug_typed(slug))
            if any(ambiguity[slug] for slug in successors):
                entry["successor_ambiguous"] = True
            shown.append(entry)
        return TombstoneReview(shown=shown, omitted=len(kept) - len(shown_hits),
                               cleared_by=cleared_by)

    def _review_siblings(self, gathering: Gathering, declared: DeclaredReview, *,
                         shown_neighbors: Sequence[Mapping[str, Any]],
                         declared_by_relation: Mapping[str, Optional[str]],
                         canonical_slugs: Mapping[str, str]) -> Dict[str, List[str]]:
        """Which shown neighbours descend from a declared target: the sibling note (7c).

        A live neighbour whose own mutation lineage (``supersedes`` ∪ ``amends`` ∪
        ``narrows``, walked from the neighbour by
        :meth:`~mitos.store.GraphStore.get_lineage_ids`) holds a declared target's own
        id (``declared.direct_ids``) descends from that target; the pause says so
        (vision §4.9). The targets of all nine declared relations count. The match is
        the target's own id, never its lineage: a neighbour that shares an ancestor
        with the target is its cousin, not its descendant. The note shows and never
        suppresses: the neighbour still pauses the draft.

        One batched read over the shown neighbours' ids, and none when the draft
        declares no target that resolved or no neighbour is shown. A neighbour's id is
        its gathered live candidate's, joined by casefolded slug (a live slug names
        its one live bearer; ``record_decision`` refuses a second, so the first in
        gather order is kept if one ever appears out of band).

        Faults are not caught: a ``DatabaseError`` propagates to the pause's fail-open
        arm, which discards the whole pause (one fault domain, 7a D3), so an absent
        note never stands beside an unread lineage. It reads ``self.store`` and
        nothing else on the manager, because the replay worker calls the pause on a
        store-and-providers shell.

        Args:
            gathering: The review's healthy gathering (its ``live`` candidates).
            declared: The draft's :class:`DeclaredReview`.
            shown_neighbors: The live neighbours the pause shows (their ``slug``).
            declared_by_relation: Relation name → the caller's raw argument, in the
                echo's render order; it orders and spells the names.
            canonical_slugs: Casefolded declared target → stored slug (the echo's
                naming rule, falling back to the declared spelling).

        Returns:
            Casefolded neighbour slug → the declared targets it descends from, in
            declaration order, each once; only neighbours with at least one.

        Raises:
            DatabaseError: If the lineage read fails.
        """
        targets: List[Tuple[str, str]] = []     # (target's own id, its name)
        named: Set[str] = set()
        for raw in declared_by_relation.values():
            for target in _split_relation_slugs(raw):
                folded = target.casefold()
                target_id = declared.direct_ids.get(folded)
                if target_id is None or target_id in named:
                    continue
                named.add(target_id)
                targets.append((target_id, canonical_slugs.get(folded, target)))
        if not targets or not shown_neighbors:
            return {}
        live_ids: Dict[str, str] = {}
        for candidate in gathering.live:
            live_ids.setdefault(candidate.node["slug"].casefold(), candidate.node["id"])
        seeds = [live_ids[n["slug"].casefold()] for n in shown_neighbors
                 if n["slug"].casefold() in live_ids]
        if not seeds:
            return {}
        lineage = self.store.get_lineage_ids(seeds)
        notes: Dict[str, List[str]] = {}
        for neighbor in shown_neighbors:
            folded = neighbor["slug"].casefold()
            if folded not in live_ids:
                continue
            reach = set(lineage[live_ids[folded]])
            names = [name for target_id, name in targets if target_id in reach]
            if names:
                notes[folded] = names
        return notes

    def _review_pause(self, entry: ParsedEntry, supersedes: Optional[str],
                      corrects: Optional[str], extra_relations: Dict[str, str], *,
                      canonical_slugs: Dict[str, str],
                      own_id: Optional[str] = None) -> PauseReview:
        """The pre-commit near-duplicate review, both partitions (P4; 7a).

        Still Phase A, so a pause writes nothing. One gather
        (:meth:`_review_neighbors`, screened to the gather's whole live window so both
        counts come from one list, D5) feeds the live partition, cut to
        ``CONFLICT_TOP_K``, and the tombstone partition (:meth:`_review_tombstones`, over
        the same gathering's retired hits); one lineage walk (:meth:`_review_declared`)
        feeds the live screen's suppression and the tombstones' exemption. A third
        read (:meth:`_review_siblings`, 7c) marks each shown live neighbour that
        descends from a declared target with ``descends_from_declared``; it shows and
        never suppresses. The pause returns when either partition holds something.

        Fail-open, one fault domain (D3): a ``DatabaseError``/``ValidationError`` from
        any read — the gather's, the lineage walk's, the kill-chain, stamp or slug reads
        of the tombstones, the sibling note's lineage read — discards both partitions
        and yields the ``neighbor_review_unavailable`` notice, so an absent tombstone
        never reads as checked-clean; a typed embed or vector-store ``Unavailable``
        words its cause the same way. Anything else is a real bug and propagates.

        The record path and the replay worker (a store-and-providers shell) call it, so
        the replay prices the shipped pause and not a copy of it.

        Args:
            entry: The parsed draft.
            supersedes: The raw ``supersedes`` argument.
            corrects: The raw ``corrects`` argument.
            extra_relations: The seven non-kill relations, stripped, non-empty only.
            canonical_slugs: Casefolded declared target → stored slug, from Phase A's
                validation loops (the echo prints stored spellings).
            own_id: The draft's own node id. Never in the graph on the record path (the
                ``exists`` gate answers first), so dropping it from the retired hits is
                a no-op there; a replayed draft's node may exist, retired, and is never
                its own tombstone.

        Returns:
            The :class:`PauseReview`.
        """
        review_unavailable: Optional[str] = None
        neighbors: List[Dict[str, Any]] = []
        tombs = TombstoneReview(shown=[], omitted=0, cleared_by=set())
        # Casefolded shown-neighbour slug → the declared targets it descends from.
        siblings: Dict[str, List[str]] = {}
        # The caller's own declarations, per relation, for the pause echo — built
        # BESIDE the declared review below, which folds the relation name away. Purely
        # additive: nothing that Phase B reads moves in here.
        # `test_mixed_neighbors_declared_edge_survives_acknowledge_bypass` documents
        # why that matters — this review is skipped under acknowledge_neighbors while
        # Phase B still writes edges from the raw relation args, so relocating any
        # normalization into it drops declared edges when both flags travel together,
        # with every other row green.
        declared_by_relation: Dict[str, Optional[str]] = {
            "supersedes": supersedes, "corrects": corrects, **extra_relations,
        }
        # This call's raw gathered set, filled by _review_neighbors. Primary-set input to
        # the partition (M8); empty when the sweep never ran.
        gathered_index: Dict[str, Tuple[str, float]] = {}
        # The gathering itself, when the review ran healthy: the cut, the full-review
        # sentence and the tombstones read it. Empty when it never ran or could not.
        gatherings: List[Gathering] = []
        try:
            # Declared targets plus their transitive mutation lineage (3b): by naming a
            # chain head the author has acknowledged the whole lineage. Without
            # providers the gate cannot run, and the declared set stays the direct
            # targets with no walk (the graph-only fast path).
            declared: Optional[DeclaredReview] = None
            if self.embed_provider and self.vector_store:
                declared = self._review_declared(supersedes, corrects, extra_relations)
                declared_targets = declared.slugs
            else:
                declared_targets = self._review_declared_targets(
                    supersedes, corrects, extra_relations)
            reviewed = self._review_neighbors(entry, declared_targets,
                                              gathered_index=gathered_index,
                                              top_k=CONFLICT_OVERFETCH_LIMIT,
                                              gathering_sink=gatherings)
            if not isinstance(reviewed, Unavailable) and gatherings and declared is not None:
                gathering = gatherings[0]
                if own_id is not None:
                    gathering = dataclasses.replace(
                        gathering,
                        retired=[h for h in gathering.retired if h.node["id"] != own_id])
                tombs = self._review_tombstones(
                    gathering, declared, shown_neighbors=reviewed[:CONFLICT_TOP_K])
                siblings = self._review_siblings(
                    gathering, declared, shown_neighbors=reviewed[:CONFLICT_TOP_K],
                    declared_by_relation=declared_by_relation,
                    canonical_slugs=canonical_slugs)
        except (DatabaseError, ValidationError) as exc:
            # A graph-store fault during the pause read (the gather's node reads, the
            # lineage walk, a tombstone read). Fail open — a fault this severe fails
            # Phase B anyway, where commit-arbitration belongs; a pre-commit crash never
            # does. Both partitions are discarded: a partly read pause would pass for a
            # whole one.
            review_unavailable = _review_unavailable_notice("graph read failed")
            gatherings = []
            # A read after the tombstones (the sibling note) can fault with the
            # tombstones already read; reset them, or a tombstone-only pause would
            # pass for a whole one.
            tombs = TombstoneReview(shown=[], omitted=0, cleared_by=set())
            siblings = {}
            # stderr: the MCP write tool shares this path and uses stdout for JSON-RPC.
            print(f"[Warning] Neighbor review skipped for '{entry.slug}': {exc}",
                  file=sys.stderr)
        else:
            if isinstance(reviewed, Unavailable):
                # Embed/vector-store degradation, typed by the gather. The reason words
                # the notice; `.detail` is logging-only, never rendered.
                review_unavailable = _review_unavailable_notice(
                    _REVIEW_UNAVAILABLE_CAUSES.get(
                        reviewed.reason, reviewed.reason.value.replace("_", " ")
                    )
                )
            else:
                neighbors = reviewed
        gathering = gatherings[0] if gatherings else None
        # The cut statement and the full-review sentence, only beside a review that ran:
        # a review that could not run already says so, and the two never co-occur.
        cut = full = None
        if review_unavailable is None and gathering is not None:
            cut = _review_cut_notice(gathering, _NEIGHBOR_REVIEW_THRESHOLD,
                                     tombstone_floor=_TOMBSTONE_FLOOR)
            full = _review_full_notice(gathering, _NEIGHBOR_REVIEW_THRESHOLD, _TOMBSTONE_FLOOR)
        # The sibling note rides a copy of its neighbour, never the entry in place.
        shown = [{**n, "descends_from_declared": siblings[n["slug"].casefold()]}
                 if n["slug"].casefold() in siblings else n
                 for n in neighbors[:CONFLICT_TOP_K]]
        if not shown and not tombs.shown:
            return PauseReview(pause=None, unavailable=review_unavailable, cut=cut)
        # The declared-edge echo (A2). mitos is stateless across calls, so a pause on a
        # different node cannot say "since your last attempt" — and a caller that
        # cannot tell whether its declaration registered mints a second, false edge
        # into the gold source. The one thing that answers it is what this call already
        # computed and threw away. A declaration that cleared a tombstone resolved part
        # of the pause, so it is never group two (D9).
        echo = _declared_echo(declared_by_relation, canonical_slugs, gathered_index,
                              _NEIGHBOR_REVIEW_THRESHOLD, cleared_by=tombs.cleared_by)
        pause: Dict[str, Any] = {
            "status": "needs_review",
            "code": "similar_decision_exists",
            "slug": entry.slug,
            # Present on every pause, `[]` on a tombstone-only one: consumers index it.
            "neighbors": shown,
        }
        if len(neighbors) > len(shown):
            pause["neighbors_omitted"] = len(neighbors) - len(shown)
        if tombs.shown:
            pause["tombstones"] = tombs.shown
        if tombs.omitted:
            pause["tombstones_omitted"] = tombs.omitted
        pause.update(echo)
        pause = {
            **{k: pause[k] for k in ("status", "code", "slug", "neighbors")},
            **{k: pause[k] for k in ("neighbors_omitted", "tombstones", "tombstones_omitted")
               if k in pause},
            "message": _pause_message(pause),
            "draft_digest": _draft_digest(entry),
            **echo,
            **({"neighbor_review_full": full} if full else {}),
            **({"neighbor_review_cut": cut} if cut else {}),
        }
        return PauseReview(pause=pause, unavailable=None, cut=None)

    def _acknowledged_fields(self, entry: ParsedEntry,
                             supersedes: Optional[str],
                             corrects: Optional[str],
                             extra_relations: Dict[str, str],
                             *,
                             target_ids: Optional[Mapping[str, str]] = None,
                             gathering_sink: Optional[List[Gathering]] = None,
                             ) -> AcknowledgedReview:
        """Names what an acknowledged write went past, never raising (F10; 7b).

        ``acknowledge_neighbors=True`` skips the pause, and mitos holds nothing between
        calls, so a pause cleared on a re-send and a flag passed blind on the first
        call look the same. This lists what the gate would have held the write for, for
        both: the live neighbours (``acknowledged_neighbors``) and the tombstones
        (``acknowledged_tombstones``). It re-runs the pause's own gather after the
        commit and the embed, so the axiom's vector is already in the embedding cache
        and the gather adds one vector query, no embedding call. Neither list is cut to
        the pause's ``CONFLICT_TOP_K``: an acknowledgement disarms the gate for the
        whole draft, so it covers everything the gather read — 20 live candidates
        (``CONFLICT_OVERFETCH_LIMIT``) and the retired hits beside them. The new node's
        own vector is in that fetch and holds one of the 20, and the own-slug screen
        drops it.

        Both lists read one :meth:`_review_declared` (one lineage walk), seeded with
        the target ids Phase A validated (``target_ids``), so neither list's
        exemptions move with the commit: the decision this write superseded or
        corrected, now retired and the closest hit by construction, and its lineage stay
        exempt as they were on the pause (D3). The tombstones are 7a's own
        :meth:`_review_tombstones` over the gathering's retired partition, bare and
        uncut (D1, D2).

        Called after the commit and outside the lock, like :meth:`_audit_debt_fields`.
        Two fault domains, one per key (D4). Both are ``None`` when the gather could not
        be trusted: no embedding provider or vector store (where
        :meth:`_review_neighbors` would answer a healthy ``[]`` for a gather that never
        ran), the collection-absent latch, a typed :class:`Unavailable`, or any
        exception from the declared review or the gather. A fault in the tombstone
        reads alone, after a healthy gather, makes only ``tombstones`` ``None``. Each
        fault writes one stderr warning. The write has landed, so nothing here may
        escape (§7).

        Args:
            entry: The committed entry.
            supersedes: The raw ``supersedes`` argument.
            corrects: The raw ``corrects`` argument.
            extra_relations: The seven non-kill relations, as the pause reads them.
            target_ids: Casefolded declared target → the node id Phase A validated.
            gathering_sink: Receives the re-run gathering, so the caller can word its
                cut; a local list is used when ``None``.

        Returns:
            The :class:`AcknowledgedReview`.
        """
        if not self.embed_provider or not self.vector_store:
            return AcknowledgedReview(neighbors=None, tombstones=None)
        # Step 8 met a missing collection it could not create: the gather would meet
        # it too, and with the new node live `missing_index_is_a_gap` says True, so it
        # would answer Unavailable. Same None, one round trip fewer. A first write
        # (may_create) never sets the latch, so a fresh workspace still gathers.
        if self._collection_absent:
            return AcknowledgedReview(neighbors=None, tombstones=None)
        sink: List[Gathering] = gathering_sink if gathering_sink is not None else []
        try:
            declared = self._review_declared(supersedes, corrects, extra_relations,
                                             target_ids=target_ids)
            reviewed = self._review_neighbors(entry, declared.slugs,
                                              top_k=CONFLICT_OVERFETCH_LIMIT,
                                              gathering_sink=sink)
        except Exception as e:
            # stderr: the MCP write tool shares this path and uses stdout for JSON-RPC.
            print(f"[Warning] Acknowledged-neighbour read failed for '{entry.slug}': "
                  f"{str(e)}", file=sys.stderr)
            return AcknowledgedReview(neighbors=None, tombstones=None)
        if isinstance(reviewed, Unavailable):
            return AcknowledgedReview(neighbors=None, tombstones=None)
        neighbors = [n["slug"] for n in reviewed]
        if not sink:
            # A healthy answer with no gathering: nothing was gathered to read.
            return AcknowledgedReview(neighbors=neighbors, tombstones=None)
        try:
            tombs = self._review_tombstones(sink[0], declared, shown_neighbors=[],
                                            bound=None, bare=True)
        except Exception as e:
            print(f"[Warning] Acknowledged-tombstone read failed for '{entry.slug}': "
                  f"{str(e)}", file=sys.stderr)
            return AcknowledgedReview(neighbors=neighbors, tombstones=None)
        return AcknowledgedReview(neighbors=neighbors, tombstones=tombs.shown)

    def _node_state(self, node_id: str) -> str:
        """Returns the computed state of a node ('active'/'superseded'/'corrected'/'drifted')."""
        return self.store.get_node_state(node_id)

    def _embedding_status(self, node_id: str) -> str:
        """Reports whether a node's embedding is queued in the outbox ('pending') or done."""
        try:
            for row in self.store.get_pending_embeddings():
                if row.get("node_id") == node_id:
                    return "pending"
        except Exception:
            pass
        return "indexed"

    def _audit_debt_fields(self, slug: str) -> Tuple[str, Optional[Dict[str, int]]]:
        """Reads the corpus's audit debt for a `created` receipt, never raising.

        Called after the commit and after the lock is released, so a telemetry
        fault can neither lengthen the lock hold nor touch the write. Unreadable
        telemetry (or no graph) gives the numberless note and ``None``, silently:
        that is an honest state, and `mitos status` owns its diagnosis. An
        unexpected exception gives the same pair plus one stderr warning, because
        a bug hidden behind a healthy-looking sentence would stay hidden. ADRs
        `record-receipt-states-cumulative-audit-debt-not-per-write-work`,
        `coverage-table-ends-the-coherence-gate-deferral` and
        `telemetry-readers-outside-check-open-read-only-absent-is-empty-not-unreadable`.

        Args:
            slug: The recorded entry's slug, named in the warning.

        Returns:
            ``(coherence_audit, audit_debt)``: the line and its three counts, or
            ``_COHERENCE_AUDIT_NOTE`` and ``None``.
        """
        try:
            debt = derive_audit_debt(self.config.db_path, self.config.telemetry_path)
        except Exception as e:
            # stderr: the MCP write tool shares this path and uses stdout for JSON-RPC.
            print(f"[Warning] Audit-debt read failed for '{slug}': {str(e)}",
                  file=sys.stderr)
            return _COHERENCE_AUDIT_NOTE, None
        if not isinstance(debt, AuditDebt):  # DebtUnreadable, or NoGraph
            return _COHERENCE_AUDIT_NOTE, None
        counts = {"uncovered": debt.uncovered, "total": debt.total,
                  "excluded": debt.excluded}
        return _audit_debt_line(**counts), counts

    def _check_notice_field(self) -> Optional[Dict[str, Any]]:
        """Composes the `created` receipt's standing check notice, never raising.

        Called after the commit and outside the lock, like
        :meth:`_audit_debt_fields`. The composer is total: it returns None for a
        keyless workspace, an absent or unreadable record, or a healthy outcome,
        and writes its own stderr warning on an unexpected fault.

        Returns:
            The ``check_notice`` payload, or None.
        """
        return compose_check_notice(
            self.config, read_attempt=read_last_attempt, get_node=self.store.get_node
        )

    def _edges_created_field(
        self, node_id: str, slug: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Composes the `created` receipt's `edges_created` echo, never raising.

        Each committed edge names its target's state computed now, its modifier
        stamps, and — on an acting relation — the target's whole axiom (the open
        question's topic on ``resolves``). The key set is decided by the relation,
        never by the value. Called after the commit and outside the lock, like
        :meth:`_audit_debt_fields`: a raise here would report a committed write as
        failed, and the caller's retry would answer ``exists``. So a failed echo
        read falls back to the base edge read with the target keys ``null``, and a
        failure of that read too gives ``None``. Each failed read writes one stderr
        warning. ADRs
        `record-receipt-edges-are-a-stamped-read-surface-axiom-rides-acting-relations-only`
        and `identification-echo-is-stamped-but-sub-letter-letter-default-governs-precedent-reads`.

        Args:
            node_id: The committed node's id.
            slug: The recorded entry's slug, named in the warnings.

        Returns:
            One entry per edge in insertion order, ``[]`` for a node with no edges,
            or None when the edges could not be read back.
        """
        try:
            entries: List[Dict[str, Any]] = []
            for t in self.store.get_outgoing_edge_targets(node_id):
                node = t["node"]
                entry: Dict[str, Any] = {
                    "kind": t["kind"],
                    "target": t["target"],
                    "target_state": t["state"],
                    "target_stamps": {
                        key: node[key]
                        for key in MODIFIER_EDGE_KEYS.values() if key in node
                    },
                }
                if t["kind"] in _ECHO_AXIOM_RELATIONS:
                    entry["target_axiom"] = node["core_axiom"]
                elif t["kind"] in _ECHO_TOPIC_RELATIONS:
                    entry["target_topic"] = node["topic"]
                entries.append(entry)
            return entries
        except Exception as e:
            # stderr: the MCP write tool shares this path and uses stdout for JSON-RPC.
            print(f"[Warning] Edge echo read failed for '{slug}': {str(e)}",
                  file=sys.stderr)
        # The fallback keeps the edges when only the echo failed: the base read,
        # with every target key the relation carries set to null (not absent).
        try:
            base = self.store.get_outgoing_edges(node_id)
        except Exception as e:
            print(f"[Warning] Edge read failed for '{slug}': {str(e)}",
                  file=sys.stderr)
            return None
        degraded: List[Dict[str, Any]] = []
        for b in base:
            entry = {"kind": b["kind"], "target": b["target"],
                     "target_state": None, "target_stamps": None}
            if b["kind"] in _ECHO_AXIOM_RELATIONS:
                entry["target_axiom"] = None
            elif b["kind"] in _ECHO_TOPIC_RELATIONS:
                entry["target_topic"] = None
            degraded.append(entry)
        return degraded

    def _exists_receipt(self, entry: ParsedEntry, existing: Dict[str, Any],
                        node_id: str, *, slug: str, state: str) -> Dict[str, Any]:
        """Composes the ``exists`` receipt, for gate 3 and its TOCTOU twin alike (7b).

        One composer, so the two returns cannot drift. ``state`` is read once by the
        caller and passed in; nothing here reads it again, so the top-level ``state``
        and the note come from one snapshot. A live node's receipt (``active`` /
        ``drifted``) is the receipt as it always was: the same keys and
        :data:`_EXISTS_NO_OP_NOTE`. A retired node's adds its handle keys after
        ``state`` (:meth:`_exists_handle_keys`) and carries a note composed from them
        (:func:`_retired_exists_note`), which names where the current word lives and
        never a relation the commit would refuse (vision §4.9, P3).

        Args:
            entry: The parsed entry this call built.
            existing: The committed node (``get_node``).
            node_id: The node's id.
            slug: The receipt's ``slug`` (gate 3 echoes the stored slug, the twin the
                entry's; a pre-existing difference, left as it was).
            state: The node's computed state, read once by the caller.

        Returns:
            The receipt dict.
        """
        receipt: Dict[str, Any] = {"slug": slug, "id": node_id, "state": state}
        note = _EXISTS_NO_OP_NOTE
        if state not in LIVE_STATES:
            keys = self._exists_handle_keys(existing, node_id, state)
            receipt.update(keys)
            note = _retired_exists_note(existing["slug"], node_id, state, keys)
        receipt.update({
            "embedding": self._embedding_status(node_id),
            "status": "exists",
            # `path` stays: it points at where the existing entry LIVES, which is
            # the useful answer to "already recorded — so where is it?" (#5b).
            # What was wrong is that the renderer labelled it `Written:` under a
            # `Recorded ✓` headline, so a re-record aimed at correcting commentary
            # — or at restoring a source block on a graph-only node — read as a
            # successful write while nothing had changed. The claim moves to
            # no_op_reason; the pointer stays.
            "path": self.config.decisions_file,
            "no_op_reason": note,
            **self._exists_receipt_extras(entry, existing, node_id),
        })
        return receipt

    def _exists_handle_keys(self, existing: Dict[str, Any], node_id: str,
                            state: str) -> Dict[str, Any]:
        """A retired node's handle keys for its ``exists`` receipt, never raising (7b).

        The §4.4 handle (:func:`~mitos.display.retired_handle`, with no score: an
        ``exists`` is not a ranked hit) from one kill-chain walk and one stamp read, then
        ``successor_ambiguous`` over its live successors (:func:`_live_successors`) by
        the validator's own predicate. Flat on the receipt, because they are the stamps
        every decision-read surface carries. One fault domain: any exception from those
        reads makes all four keys ``None``, ``superseded_by`` included — a handle that
        kept its stamps but lost its live end could name a dead hop as the place to go.

        Args:
            existing: The committed node.
            node_id: Its id.
            state: Its computed state.

        Returns:
            ``superseded_by`` / ``corrected_by`` when held, ``live_end`` when an
            immediate successor is retired, ``successor_ambiguous: True`` when some live
            successor's slug names several nodes; or all four ``None`` on a fault.
        """
        try:
            chains, chain_slugs = self.store.get_kill_chains([node_id])
            live_ends = live_ends_from_chains(chains, chain_slugs)
            stamps = self.store.get_modifiers_map([node_id])
            handle = retired_handle({**existing, "computed_state": state}, score=None,
                                    stamps=stamps, live_ends=live_ends)
            keys = {k: handle[k] for k in ("superseded_by", "corrected_by", "live_end")
                    if k in handle}
            if any(_ambiguous_target(self._resolve_slug_typed(s))
                   for s in _live_successors(handle)):
                keys["successor_ambiguous"] = True
            return keys
        except Exception as e:
            # stderr: the MCP write tool shares this path and uses stdout for JSON-RPC.
            print(f"[Warning] Retired-handle read failed for '{existing.get('slug')}': "
                  f"{str(e)}", file=sys.stderr)
            return dict.fromkeys(_EXISTS_HANDLE_KEYS)

    def _exists_receipt_extras(
        self, entry: ParsedEntry, existing: Dict[str, Any], node_id: str
    ) -> Dict[str, Any]:
        """Builds the `exists` receipt's divergence hint.

        AX round 10's actual ask, verbatim: *say what it ignored*. A re-record aimed at
        correcting commentary got a clean `(exists) ✓` and no indication that the values
        it carried differed from the stored ones — so the caller had to go and check by
        hand to learn that nothing had changed.

        Args:
            entry: The parsed entry this call built.
            existing: The committed node.
            node_id: The node's id.

        Returns:
            ``{"differs": [field, ...]}`` when the call carried different commentary
            than the graph holds, else ``{}`` — an empty list would read as a
            measurement rather than an absence.
        """
        try:
            report = entry_divergence(
                entry,
                existing,
                existing.get("scope") or [],
                self.store.get_outgoing_edges(node_id),
            )
        except Exception:
            return {}  # a receipt hint must never fail the receipt
        differs = list(report.get("commentary") or [])
        if report.get("scope"):
            differs.append("scope")
        if report.get("edges"):
            differs.append("edges")
        return {"differs": sorted(differs)} if differs else {}

    def _uncommittable_edges(self, entry: ParsedEntry,
                             divergence: Dict[str, Any]) -> List[str]:
        """Returns why each newly-declared edge provably cannot commit, if any.

        The reconcile pre-flight. A reconcile whose commit is FORECLOSED must be
        refused before the write-ahead intent row, because the commit failure then
        writes a correlated outcome row too — so every sync of an uncommittable entry
        appends two audit rows, forever, and retention is deliberately deferred.

        **Scoped to the class, not an instance.** Phase 3 shipped this as a
        `missing_target`-only check and was too narrow: a kind-illegal edge has a
        perfectly resolvable target, sailed through, and repeated indefinitely
        (measured on cartolina: 19 such edges, 6 audit rows after 3 syncs). The
        disposition of EVERY store failure code is now declared in
        ``_PREFLIGHT_DISPOSITIONS`` and pinned by ``test_preflight_covers_store_codes``,
        so a code added later cannot be silently missed the same way again.

        Refusal is deliberately conservative: it fires only when the commit is
        foreclosed for EVERY node the citation could bind to. A false positive here
        silently skips a legitimate reconcile, which is worse than an extra audit row.

        Args:
            entry: The parsed corpus entry (its ``kind`` is the edge source kind).
            divergence: The ``entry_divergence`` report.

        Returns:
            Human-readable reasons, in report order. Empty means nothing forecloses
            the commit — not that it will necessarily succeed.
        """
        added = (divergence.get("edges") or {}).get("added") or []
        reasons: List[str] = []
        for spec in added:
            edge_type, _, target = spec.partition(":")
            if not target:
                continue
            # `missing_target` / `dangling_edge` — the citation names nothing.
            target_kinds = self.store.resolve_slug_kinds(target)
            if not target_kinds:
                reasons.append(
                    f"'{target}' does not name any entry in the graph — fix the "
                    "citation in decisions.md, or restore the cited entry with "
                    "`mitos restore-source`"
                )
                continue
            # `kind_constraint_violation` — the citation resolves, but the `edges`
            # CHECK can never admit the edge. Refuse only if EVERY candidate kind is
            # illegal (a slug can name a whole supersession lineage, MI-13).
            if not any(
                edge_kind_is_legal(edge_type, entry.kind, target_kind)
                for target_kind in target_kinds
            ):
                requirement = _EDGE_KIND_REQUIREMENT.get(
                    edge_type, "satisfy the kind matrix"
                )
                reasons.append(
                    f"'{target}' has an incompatible kind for a {edge_type} edge — "
                    f"{edge_type} edges must {requirement}. Re-author the relation "
                    "(a decision citing a precedent wants `Cites:`) or remove the line"
                )
        return reasons

    def _apply_commentary_reconcile(
        self,
        entry: ParsedEntry,
        existing: Dict[str, Any],
        node_id: str,
        divergence: Dict[str, Any],
        auto_accept: bool,
        *,
        authorized: bool = False,
        full: bool = False,
    ) -> bool:
        """Applies one commentary reconcile, or reports why it was skipped.

        The store has always been able to update commentary in place on a matching
        canonical core; all four callers gated the branch away, so it was reachable only
        from unit tests. This routes one caller to it — which is why Phase 3 is the
        first production consumer of MI-4 and MI-5.

        Args:
            entry: The parsed corpus entry.
            existing: The committed node, as stored.
            node_id: The node's id.
            divergence: The ``entry_divergence`` report.
            auto_accept: Whether ``--yes`` is in force.
            authorized: Whether the caller named THIS entry with
                ``--reconcile-entry``. It is an additional way to satisfy the gate
                below, never a second apply path: an authorized entry skips both
                authorization branches and falls through to the same foreclosure
                check, the same write-ahead attribution row and the same
                ``commit_parsed_entry`` a TTY ``[r]`` already reaches, so a fault
                mid-reconcile leaves exactly the state a confirmed reconcile leaves
                today. The named-target test lives at the call site, so this
                function stays ignorant of the flag's vocabulary.
            full: ``--full``: print each diverged commentary field whole on both
                sides, and no pointer to the flag.

        Returns:
            True if the reconcile was applied.
        """
        edges = divergence.get("edges") or {}
        removals = edges.get("removed") or []

        print(f"\n[Divergence] '{existing.get('slug')}' — the corpus and graph disagree.")
        commentary = divergence.get("commentary") or []
        for field in commentary:
            for line in _divergence_field_lines(
                field, existing.get(field), getattr(entry, field, None), full=full,
            ):
                print(line)
        if divergence.get("scope"):
            print(f"  scope:      graph {divergence['scope']['graph']} → "
                  f"markdown {divergence['scope']['markdown']}")
        for added in edges.get("added") or []:
            print(f"  edge ADDED:   {added}")
        for removed in removals:
            # Printed explicitly, and named as a deletion, because a re-commit mirrors
            # edges declaratively: a line removed from the markdown DELETES that edge.
            print(f"  edge DELETED: {removed}")
        if commentary and not full:
            # After the edge lines and before any refusal, so this recipe never sits
            # beside a `--reconcile-entry` one where the two could be mistaken.
            print("  Only the differing spans are shown. For the whole values:")
            print(f"    mitos sync {shell_flag('-p', self.config.project)} --full")

        # `authorized` satisfies BOTH branches below for the entry the caller named,
        # and nothing else: the fall-through is the shipped one, not a second route.
        if not authorized:
            if auto_accept and removals:
                # `--yes` widens sync from append-only to mutate, but never to delete
                # an edge unattended. And because `commit_parsed_entry` mirrors edges
                # declaratively, an entry's commentary cannot be applied while
                # withholding its edge state — so a MIXED entry is skipped whole. That
                # holds for every entry a run was NOT authorized for; naming the entry
                # is the other way to satisfy this gate, and it applies the whole
                # reconcile rather than the deletion alone.
                print("  Skipped — this run authorized no repair for this entry, and "
                      "an edge DELETION cannot be applied under --yes alone (an "
                      "entry's commentary cannot be applied while withholding its "
                      "edge state, so a mixed entry is skipped whole). To apply it, "
                      "name the entry:")
                # `entry.slug`, never `existing['slug']`: the door matches the handle
                # against the MARKDOWN slug the loop is iterating, so on a hand-edited
                # RENAME the graph's slug is the one string this recipe must not carry
                # — a caller copying it would name an entry no parsed block matches and
                # get the never-seen refusal on a target sitting in front of them.
                print(f"    mitos sync {shell_flag('-p', self.config.project)} "
                      f"{shell_flag('--reconcile-entry', entry.slug)}")
                print("  Or restore the relation line in decisions.md to reconcile "
                      "the commentary alone.")
                return False

            if not auto_accept:
                if not sys.stdin.isatty():
                    # No TTY, no `--yes` and no authorization for this entry: report
                    # and skip, never prompt. This gate fires on a corpus state that
                    # used to produce ZERO prompts (the old code `continue`d
                    # unconditionally), so prompting here turned every non-interactive
                    # `mitos sync` — an agent, a CI job, a cron, a piped invocation —
                    # into a fatal `EOF when reading a line`. Skipping is the
                    # fail-closed choice: the absence of a terminal is not an
                    # authorization to mutate. What IS one is `--yes` (for an entry
                    # carrying no edge deletion) or naming this entry.
                    print("  Skipped — no terminal to confirm on, and this run "
                          "authorized no repair for this entry. To apply it "
                          "non-interactively, name it:")
                    # The markdown slug, for the reason the sibling refusal states.
                    print(f"    mitos sync {shell_flag('-p', self.config.project)} "
                          f"{shell_flag('--reconcile-entry', entry.slug)}")
                    return False
                choice = input("Reconcile the graph to the markdown? "
                               "[r]econcile / [s]kip: ").strip().lower()
                if choice != "r":
                    print("  Skipped.")
                    return False

        uncommittable = self._uncommittable_edges(entry, divergence)
        if uncommittable:
            print("  Skipped — cannot reconcile; this commit is foreclosed, so no "
                  "attribution row was written:", file=sys.stderr)
            for reason in uncommittable:
                print(f"    - {reason}.", file=sys.stderr)
            return False

        # Write-AHEAD: the attribution row lands BEFORE the mutation. A best-effort
        # write-after can leave a mutation with no attribution, which is undetectable
        # and exactly what P8 forbids; write-ahead leaves at worst a phantom row for a
        # change that never landed, and a phantom never violates P8 — it forbids
        # unattributed mutations, not attributed non-mutations.
        stored_edges = self.store.get_outgoing_edges(node_id)
        prior, new_values = self._reconcile_value_pair(
            entry, existing, divergence, stored_edges
        )
        audit_id = uuid.uuid4().hex
        try:
            telemetry = TelemetryStore(self.config.telemetry_path)
            telemetry.record_commentary_intent(
                CommentaryAuditRow(
                    audit_id=audit_id,
                    node_id=node_id,
                    slug=existing.get("slug"),
                    fields_changed=sorted(new_values),
                    prior_values=prior,
                    new_values=new_values,
                    mitos_version=MITOS_VERSION,
                ),
                created_at=_utc_now_iso(),
            )
        except Exception as exc:
            # MANDATORY writer, inverting this store's ambient best-effort posture: a
            # telemetry failure must never abort a sync of real decisions, but a
            # P8-mandatory row cannot inherit that. An unauditable reconcile is REFUSED,
            # not applied unaudited — otherwise P8 holds only when the disk cooperates.
            print(f"  Skipped — cannot write the attribution row ({exc}); "
                  "refusing to reconcile unaudited.", file=sys.stderr)
            return False

        # Carry the STORED confirmation pair forward. Both columns sit in the store's
        # commentary UPDATE SET, and the parsed entry carries `confirmed_by=None` here
        # (sync's "user" stamp happens further down the commit path) — so a naive
        # reconcile NULLs them and one reusing sync's stamping rewrites them to "user".
        # Measured: either destroys 114 stamps on a change that touched only prose.
        entry.confirmed_by = existing.get("confirmed_by")
        entry.confirmed_at = existing.get("confirmed_at")

        # Withhold the transcript. `commit_parsed_entry` UPDATEs `transcripts` whenever
        # the incoming text is non-None and differs — but the transcript is NOT in the
        # divergence set, so it appears in no printed diff, no `fields_changed`, and no
        # `prior_values`. Left attached it RIDES ALONG on any reconcile triggered by
        # another species: an agent that edits `**Rejected:**` and rewrites the
        # transcript block gets both applied, the operator pressing `[r]` sees only
        # `rejected_paths` named, and the prior transcript then exists nowhere — markdown
        # rewritten, graph updated, audit row silent. That is exactly the unattributed
        # graph mutation P8 forbids, inside the feature that adds the attribution row.
        # `None` means no-change to the store, which restores the documented
        # write-once-preserved semantics. `amend_commentary` shipped without transcript
        # editing (it withholds the transcript too), so transcript reconciliation is
        # still unhomed.
        entry.transcript = None

        try:
            self.store.commit_parsed_entry(entry)
        except CommitError as exc:
            # Reported per-entry and moved past — NEVER routed to the quarantine
            # fixpoint. A retry cannot help: nothing later in this pass changes the
            # outcome, and MI-13's FM2 (dropping a `Supersedes:` line resurrects a
            # predecessor into a slug_collision rollback) would re-fail every pass.
            reason = "; ".join(i.message for i in exc.failure.items) if exc.failure else str(exc)
            print(f"  Reconcile FAILED for '{existing.get('slug')}': {reason}",
                  file=sys.stderr)
            # Append-only closure: the intent row stays and a correlated outcome row
            # records the failure. Telemetry never UPDATEs or DELETEs, and this write
            # IS best-effort — the mutation did not happen, so there is no unattributed
            # mutation to hide.
            try:
                telemetry.record_commentary_outcome(
                    audit_id=uuid.uuid4().hex,
                    correlates_to=audit_id,
                    outcome=f"failed: {reason}",
                    created_at=_utc_now_iso(),
                    mitos_version=MITOS_VERSION,
                )
            except Exception:
                pass
            return False

        print(f"  Reconciled '{existing.get('slug')}' ✓")
        return True

    @staticmethod
    def _reconcile_value_pair(
        entry: ParsedEntry,
        existing: Dict[str, Any],
        divergence: Dict[str, Any],
        stored_edges: List[Dict[str, str]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Builds the audit row's ``(prior, new)`` value pair.

        Both sides, not just the prior: self-detection of a crash-phantom compares the
        NEW values against the graph, and chain consistency compares a successor's
        ``prior`` against its predecessor's ``new``.

        Args:
            entry: The parsed corpus entry.
            existing: The committed node.
            divergence: The ``entry_divergence`` report.
            stored_edges: The node's outgoing edges as stored, so the edge columns can
                record full state rather than a delta.

        Returns:
            The ``(prior_values, new_values)`` pair, keyed identically.
        """
        prior: Dict[str, Any] = {}
        new_values: Dict[str, Any] = {}
        for field in divergence.get("commentary") or []:
            prior[field] = existing.get(field)
            new_values[field] = getattr(entry, field, None)
        if divergence.get("scope"):
            prior["scope"] = divergence["scope"]["graph"]
            new_values["scope"] = divergence["scope"]["markdown"]
        if divergence.get("edges"):
            # FULL STATE on both sides, not the delta. `prior["edges"] = removed` reads
            # as "the graph held exactly these", which is false whenever the node keeps
            # an edge — and the documented read rule (an intent row over a graph
            # matching its `new_values` means applied) would then mismatch and report a
            # successfully-applied reconcile as NOT applied, inverting the P8 verdict.
            stored = _edge_state_labels(stored_edges)
            declared = _edge_state_labels(declared_edges(entry))
            prior["edges"] = stored
            new_values["edges"] = declared
        return prior, new_values

    def splice_buffer(
        self,
        transform: Callable[[str], str],
        *,
        after_write: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Rewrites `decisions.md` under the lock, rolling back on any failure.

        The buffer-surgery primitive: **lock → auto-heal → read → splice → write →
        verify → roll back on failure.** Extracted rather than inlined at its first
        caller (``cli.cmd_restore_source``) so the commentary amendment could consume
        it without a retrofit; ``amend_commentary`` is its second consumer, committing
        the graph inside ``after_write`` so the buffer edit and the commit roll back
        together.

        Modelled on ``record_decision_entry``'s buffer-first + rollback contract,
        which this project treats as sacred. That method is deliberately NOT refactored
        to route through here: it interleaves the graph commit with the buffer write in
        a way this generic seam would have to special-case, and rewriting a sacred
        contract to serve a new caller is the wrong direction of dependency. The
        equivalence is pinned by test instead.

        Args:
            transform: Maps the current buffer text to its replacement. Runs before the
                splice write, so a raise here writes no entry — though note the
                auto-heal above it may already have restored a drifted header, which is
                an idempotent repair and deliberately outside the rollback. This is the
                one ordering difference from ``record_decision_entry``, which runs its
                gates BEFORE auto-heal precisely so a rejected record leaves the buffer
                byte-for-byte untouched; here the transform NEEDS the healed marker to
                splice against.
            after_write: Optional verification invoked with the written text. A raise
                rolls the buffer back — this is where a fidelity check belongs, so a
                splice that would corrupt a neighbour never survives.

        Returns:
            The buffer text as written.

        Raises:
            MitosError: If the rollback ITSELF fails, naming the state the file is in —
                the one case where silence would leave an operator guessing.
            Exception: Whatever ``transform`` or ``after_write`` raised, re-raised after
                a successful rollback.
        """
        with self.lock:
            self.auto_heal_decisions_file()
            with open(self.config.decisions_file, "r", encoding="utf-8") as fh:
                original = fh.read()

            # Computed before the write, so a transform failure is a pure no-op.
            new_content = transform(original)

            try:
                atomic_file.write_source(self.config.decisions_file, new_content)
                if after_write is not None:
                    after_write(new_content)
            except Exception:
                try:
                    atomic_file.write_source(self.config.decisions_file, original)
                except Exception as restore_exc:
                    raise MitosError(
                        "The splice failed AND decisions.md could not be rolled back "
                        f"(rollback error: {restore_exc}). Each write replaces the file "
                        "whole, so it holds whole content: either its text from before "
                        "the splice or the unverified splice. Check which before running "
                        "`mitos sync`."
                    ) from restore_exc
                raise
            return new_content

    def _amend_divergence(self, entry: ParsedEntry, node: Dict[str, Any],
                          stored_edges: List[Dict[str, str]]) -> Tuple[List[str], Dict[str, Any]]:
        """Names every way a buffered block and its node already disagree.

        Args:
            entry: The parsed buffer block.
            node: Its committed node.
            stored_edges: The node's outgoing edges.

        Returns:
            ``(species, report)``: the diverged field names — commentary fields,
            ``scope``, ``edges``, ``source``, and ``slug`` for a case-only difference the
            casefold comparison does not report — and the ``entry_divergence`` report.
        """
        report = entry_divergence(entry, node, node.get("scope") or [], stored_edges)
        species = set(report.get("commentary") or [])
        for key in ("scope", "edges", "source"):
            if report.get(key):
                species.add(key)
        if (entry.slug or "") != (node.get("slug") or ""):
            species.add("slug")
        return sorted(species), report

    def amend_commentary(self, slug: str, changes: Mapping[str, Any]) -> Dict[str, Any]:
        """Edits one committed buffered entry's mutable fields and re-commits it.

        The tool-side replacement for hand-editing ``decisions.md`` and running a
        reconcile. It repairs commentary — ``amend.EDITABLE_FIELDS``, the reconcile's
        set less edges — and never what a decision says: the canonical core and the
        edges are refused as requests, with the kill-edge routes as data.

        One lock hold, through ``splice_buffer``. The transform classifies the target on
        the live locked read and applies field-line surgery; ``after_write`` runs the
        whole-buffer fence (``restore.verify_amended_buffer``), writes the write-ahead
        attribution row, and commits — so a refused fence, an unwritable audit row or a
        failed commit rolls the buffer back byte for byte. Embedding and rendering run
        after the lock, best-effort, warnings on stderr only; stdout stays empty because
        an MCP twin shares this path. No rotation is evaluated: the entry count never
        changes, and the target's ``updated_at`` tick keeps it recent anyway.

        A target whose buffered block already disagrees with its node (a hand edit not
        yet reconciled, or a citation left stale by a rename elsewhere) is refused as
        ``diverged``: committing it would also apply edge or field changes the request
        did not name.

        CC-21: ``record_decision_entry`` diverges from this fence and is not edited. Its
        structural-token check refuses heading- and field-shaped lines in the prose
        fields, and its self-parse count guard refuses a phantom entry from the list and
        relation arguments, both before any write; a new block is prepended below the
        entries marker, so continuation bleed cannot reach a neighbour.

        Args:
            slug: The target handle (a slug, or an id), resolved as ``show`` resolves it.
            changes: Field → new value; keys from ``amend.EDITABLE_FIELDS``. ``slug`` is
                the new slug; ``invalidates_if``/``context`` take ``None`` or ``""`` to
                remove the field; ``scope`` takes a list, ``[]`` removing the line.

        Returns:
            A JSON-safe dict whose ``status`` is ``amended`` (``id``, ``fields_changed``,
            ``embedding``, ``path``, and ``rename`` = ``{"from", "to", "incoming"}`` on a
            rename), ``unchanged`` (``id``), ``refused`` (``reason``, ``fields``,
            ``routes`` for ``canonical_core``, ``markup_spans`` for
            ``tool_call_markup``), ``not_found``, ``archived`` (``id``) or
            ``uncommitted``; or a fault ``{"error", "code", "slug"}`` with code
            ``slug_collision``, ``commit_failed`` (also a graph read that failed under
            the lock), ``audit_unavailable``,
            ``lock_timeout`` or ``rollback_failed``. Every string states a cause and
            names no command.

        Raises:
            BufferFidelityError: If the edit would change the buffer beyond the request
                (a phantom entry, neighbour bleed, a moved core or an unrequested field);
                the buffer is rolled back first. The only designed raise.
            ValidationError: If the handle's slug resolves to more than one active node
                (an MI-13 breach is not "not found").
        """
        refusal = amend.validate_changes(changes)
        if refusal is not None:
            return {**refusal, "slug": slug}

        state: Dict[str, Any] = {}

        def _transform(original: str) -> str:
            target = amend.classify_target(
                original, slug=slug,
                resolve=self.store.resolve_handle, lookup=self.store.get_node,
            )
            if isinstance(target, amend.Miss):
                raise _AmendAnswer(target.result)
            species, _ = self._amend_divergence(
                target.entry, target.node, self.store.get_outgoing_edges(target.node_id)
            )
            if species:
                raise _AmendAnswer(amend.refused(amend.REASON_DIVERGED, species))
            expected = amend.expected_fingerprint(target.entry, changes)
            if amend.fingerprint_matches(expected, target.entry):
                raise _AmendAnswer({"status": amend.STATUS_UNCHANGED, "id": target.node_id})
            # The baseline is the text this transform saw, never a pre-lock snapshot.
            state.update(before=original, target=target, expected=expected)
            return amend.apply_changes(original, target, changes)

        def _verify_and_commit(after_text: str) -> None:
            target = state["target"]
            entry = verify_amended_buffer(
                state["before"], after_text,
                target_id=target.node_id, expected=state["expected"],
            )
            node = self.store.get_node(target.node_id)
            if node is None:
                raise _AmendAnswer({"status": amend.STATUS_UNCOMMITTED})
            stored_edges = self.store.get_outgoing_edges(target.node_id)
            # Re-checked against the graph as it is now: a concurrent sync reconciles
            # outside this lock. Only the requested fields may differ from the node.
            species, divergence = self._amend_divergence(entry, node, stored_edges)
            unrequested = sorted(set(species) - set(changes))
            if unrequested:
                raise _AmendAnswer(amend.refused(amend.REASON_DIVERGED, unrequested))
            prior, new_values = self._reconcile_value_pair(entry, node, divergence, stored_edges)
            if entry.slug != node.get("slug") and "slug" not in new_values:
                # A case-only rename: the divergence leaf compares casefold.
                prior["slug"], new_values["slug"] = node.get("slug"), entry.slug

            audit_id = uuid.uuid4().hex
            try:
                telemetry = TelemetryStore(self.config.telemetry_path)
                telemetry.record_commentary_intent(
                    CommentaryAuditRow(
                        audit_id=audit_id,
                        node_id=target.node_id,
                        slug=node.get("slug"),
                        fields_changed=sorted(new_values),
                        prior_values=prior,
                        new_values=new_values,
                        mitos_version=MITOS_VERSION,
                    ),
                    created_at=_utc_now_iso(),
                )
            except Exception as exc:
                raise _AmendFault("audit_unavailable", reason=str(exc)) from exc

            # Graph-primary confirmation pair carried forward, transcript withheld: the
            # reconcile's two rules, for the same reasons.
            entry.confirmed_by = node.get("confirmed_by")
            entry.confirmed_at = node.get("confirmed_at")
            entry.transcript = None
            try:
                delta = self.store.commit_parsed_entry(entry)
            except (CommitError, DatabaseError, ValidationError, OSError) as exc:
                reason = str(exc)
                fault: Dict[str, Any] = {"code": "commit_failed"}
                if isinstance(exc, CommitError) and exc.failure:
                    reason = "; ".join(item.message for item in exc.failure.items) or reason
                    if any(item.code == STORE_SLUG_COLLISION for item in exc.failure.items):
                        fault = {"code": "slug_collision", "requested": entry.slug}
                if fault["code"] == "commit_failed":
                    fault["reason"] = reason
                # Append-only closure, best-effort: the mutation did not happen.
                try:
                    telemetry.record_commentary_outcome(
                        audit_id=uuid.uuid4().hex,
                        correlates_to=audit_id,
                        outcome=f"failed: {reason}",
                        created_at=_utc_now_iso(),
                        mitos_version=MITOS_VERSION,
                    )
                except Exception:
                    pass
                raise _AmendFault(**fault) from exc
            state.update(delta=delta, entry=entry, old_slug=node.get("slug"),
                         fields_changed=sorted(new_values))

        try:
            self.splice_buffer(_transform, after_write=_verify_and_commit)
        except _AmendAnswer as answer:
            return {**answer.result, "slug": slug}
        except _AmendFault as fault:
            return amend.error_result(fault.code, slug=slug, **fault.fields)
        except BufferFidelityError:
            raise
        except Timeout:
            return amend.error_result("lock_timeout", slug=slug)
        except (DatabaseError, sqlite3.Error) as exc:
            # A graph read under the lock failed (classification, the divergence checks
            # or the re-read before commit): an environment fault, not an answer about
            # the target, and the buffer was never written or has been rolled back.
            return amend.error_result("commit_failed", slug=slug, reason=str(exc))
        except MitosError as exc:
            # `splice_buffer` raises a bare MitosError, chained, only when its rollback
            # write failed; every other MitosError is a subclass.
            if type(exc) is MitosError and exc.__cause__ is not None:
                return amend.error_result("rollback_failed", slug=slug, reason=str(exc.__cause__))
            raise
        except OSError as exc:
            return amend.error_result(
                "commit_failed", slug=slug, reason=f"the buffer could not be read or written: {exc}"
            )

        target, entry, delta = state["target"], state["entry"], state["delta"]
        result: Dict[str, Any] = {
            "status": amend.STATUS_AMENDED,
            "slug": entry.slug,
            "id": target.node_id,
            "fields_changed": state["fields_changed"],
        }
        if entry.slug != state["old_slug"]:
            # Read after the commit: the id is unchanged, so the edges survive; what goes
            # stale is the citing entries' markdown.
            try:
                incoming: Optional[List[Dict[str, str]]] = self.store.get_incoming_edges(
                    target.node_id
                )
            except Exception as e:
                print(f"[Warning] Could not read the entries citing '{entry.slug}': {e}",
                      file=sys.stderr)
                incoming = None
            result["rename"] = {"from": state["old_slug"], "to": entry.slug, "incoming": incoming}

        try:
            self._best_effort_embed(delta, entry)
        except Exception as e:
            print(f"[Warning] Embedding step failed for '{entry.slug}': {describe_embed_failure(e)}", file=sys.stderr)
        try:
            # Unfiltered, so a vacated scope's file is swept and a moved primary follows.
            MitosRenderer(self.config.workspace_dir).render_all(self.store)
        except Exception as e:
            print(f"[Warning] Failed to render active axioms: {e}", file=sys.stderr)

        result["embedding"] = self._embedding_status(target.node_id)
        result["path"] = self.config.decisions_file
        return result

    def record_decision_entry(
        self,
        axiom: str,
        rejected_paths: str,
        scope: List[str],
        mechanisms: Optional[List[str]] = None,
        context: Optional[str] = None,
        supersedes: Optional[str] = None,
        corrects: Optional[str] = None,
        amends: Optional[str] = None,
        narrows: Optional[str] = None,
        depends_on: Optional[str] = None,
        resolves: Optional[str] = None,
        contradicts: Optional[str] = None,
        derives_from: Optional[str] = None,
        cites: Optional[str] = None,
        slug: Optional[str] = None,
        actor: str = "agent",
        acknowledge_neighbors: bool = False,
        draft_digest: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Records a single decision into the buffer and graph, non-interactively.

        The agentic write half of Mitos: persists one deliberate, pre-structured
        decision (and the alternatives it rejected) the moment an agent makes it,
        without LLM enrichment and without calling ``perform_sync``. Composes the
        existing primitives only — ``commit_parsed_entry`` for the graph and
        ``_best_effort_embed`` for the vector (preserves M7).

        Validation runs entirely in memory FIRST (Phase A); the buffer append and
        the graph commit happen together, last, under a single lock, with the
        buffer rolled back if the commit fails (Phase B). The contract: on any
        error code, ``decisions.md`` is byte-for-byte unchanged and nothing is
        committed.

        Args:
            axiom: The decision as a single clear sentence true going forward (M1).
            rejected_paths: The alternatives considered and rejected, and why (M5, required).
            scope: Area tags (may be empty).
            mechanisms: Concrete technologies/entities involved (M6).
            context: Optional background on why this was decided.
            supersedes: Optional exact slug of a prior decision this one replaces.
            corrects: Optional exact slug of a prior decision this one corrects (a
                kill-edge twin of supersedes — the target leaves the active view).
            amends: Optional exact slug of a decision this one amends.
            narrows: Optional exact slug of a decision this one narrows.
            depends_on: Optional exact slug of a decision this one depends on.
            resolves: Optional exact slug of an open question this resolves (the
                ``resolves`` edge is decision→open_question only).
            contradicts: Optional exact slug of a decision this one contradicts.
            derives_from: Optional exact slug of a decision this one derives from.
            cites: Optional exact slug of a decision this one cites.
            slug: Optional explicit slug; derived deterministically from axiom if None.
            actor: Provenance, stored in ``confirmed_by``.
            acknowledge_neighbors: Skip the pre-commit near-duplicate review and record
                even when a highly-similar unreferenced decision exists (P4). Pass True
                to commit a genuinely independent decision past the pause.
            draft_digest: Optional; the ``draft_digest`` a pause returned. The built
                draft is compared with it, field by field, before any store read — a
                drifted field refuses the call, under ``acknowledge_neighbors`` too.
                ``None`` or blank skips the comparison.

        Returns:
            A success dict ``{slug, id, state, embedding, status}`` (status
            "created"|"exists"); an "exists" receipt whose node is retired also carries
            that node's handle keys after ``state`` (:meth:`_exists_receipt`, 7b):
            ``superseded_by`` / ``corrected_by`` when held, ``live_end`` when an
            immediate successor is retired, ``successor_ambiguous: true`` when a live
            successor's slug names two decisions — all four ``None`` when the handle
            could not be read — and a ``no_op_reason`` naming where its current word
            lives and never a relation the commit would refuse (a live node's receipt
            is unchanged); on the "created" path it also carries
            ``edges_created`` (the edges the commit actually wired, read back
            from the store and not echoed from the input args, each ``{kind,
            target, target_state, target_stamps}`` plus the target's whole
            ``target_axiom`` on an acting relation or its ``target_topic`` on
            ``resolves`` — an identification echo, never the target's reasoning;
            the target keys are ``null`` when the echo read failed, and the list
            is ``None`` when the edges could not be read back), the resolved ``scope`` as committed (the
            graph's casefolded form), ``mechanisms`` as authored — the spelling
            and order ``decisions.md`` now holds, read back from the parsed
            entry — and ``mechanisms_normalized``, mapping each authored token
            the fold changed to its folded form (the form the graph and the node
            id hold; ``{}`` when none changed), plus an optional
            ``neighbor_review_unavailable`` notice when the pre-commit near-dup
            check could not run (the record fails open — the commit proceeds
            unchecked; the notice names the cause and no command, each surface
            composing its own recovery), plus an always-present
            ``coherence_audit`` statement of the corpus's standing, cumulative
            contradiction-check debt ("N of this corpus's M decisions have not
            been covered…", or the numberless ``_COHERENCE_AUDIT_NOTE`` when the
            count cannot be read) and an always-present ``audit_debt``
            (``{uncovered, total, excluded}``, or ``None`` when unreadable), plus
            an optional ``rotation`` report of the one bounded rotation this
            write ran once its buffer reached
            ``rotation_volume_threshold_entries``: ``outcome`` "rotated" (the
            ``archives`` written, each ``{path, entries, slugs}``, and any
            ``skipped``), "skipped" (nothing moved; each ``skipped`` element names
            its slug and reason) or "failed" (``stage``/``error``/``note``; the write
            still stands, and the field names no command); absent when nothing was
            eligible; plus an optional ``check_notice`` (the last contradiction
            check's outcome, :func:`~mitos.check_notice.compose_check_notice`),
            present only while a judge key is set and that outcome is not
            ``no_new_findings``; plus, only when ``acknowledge_neighbors`` was
            passed, ``acknowledged_neighbors`` (:meth:`_acknowledged_fields`):
            the stored slugs of every unlinked live decision at or above the floor
            that the write went past, most similar first, ``[]`` when there were
            none, or ``None`` when the gather could not run; and beside it
            ``acknowledged_tombstones``: the bare retired handles (``slug`` / ``id`` /
            ``state`` / ``score`` / successor stamps / ``live_end``; no axiom) of every
            non-exempt retired decision at or above ``_TOMBSTONE_FLOOR`` the write went
            past, closest first and uncut, ``[]`` when there were none, or ``None`` when
            the gather or the tombstone reads could not run; OR, when a highly-similar
            unreferenced live decision or retired decision exists and
            ``acknowledge_neighbors`` is False, a ``{status: "needs_review", code:
            "similar_decision_exists", slug, neighbors, message, draft_digest}`` pause
            that wrote NOTHING (:meth:`_review_pause`) — ``draft_digest``
            (:func:`_draft_digest`) is what a re-send passes back to be compared with
            this draft;
            each ``neighbors`` element is an enriched, modifier-stamped decision-read
            payload (:func:`~mitos.conflict.candidate_payload`: ``slug`` / ``axiom`` /
            ``scope`` / ``score`` / ``rejected_paths`` plus any ``amended_by``/
            ``narrowed_by`` stamps) the authoring agent judges tenability from, at most
            ``CONFLICT_TOP_K`` and ``[]`` on a tombstone-only pause; a neighbour whose
            own mutation lineage reaches a target this draft declares also carries
            ``descends_from_declared`` (7c, :meth:`_review_siblings`): those targets'
            slugs, in declaration order. The pause's second
            partition, ``tombstones`` (7a), lists the retired decisions the draft sits at
            or above ``_TOMBSTONE_FLOOR`` to, closest first, at most ``CONFLICT_TOP_K``,
            exempt ones dropped first (:meth:`_review_tombstones`): each a retired handle
            (:func:`~mitos.display.retired_handle`: ``slug`` / ``id`` / ``state`` /
            ``score`` / successor stamps / ``live_end``) plus its whole ``axiom`` and, when
            they hold, ``successor_on_pause`` and ``successor_ambiguous: true``. Counts
            ride beside them only when non-zero: ``neighbors_omitted`` and
            ``tombstones_omitted``; ``neighbor_review_full`` says, when the gather's live
            fill ended full above the lower floor, that both counts cover what it read;
            ``neighbor_review_cut`` says the gather was cut. Every one of these keys,
            ``descends_from_declared`` included, is absent when computed and empty. That
            pause also echoes the caller's own declared relation targets, partitioned
            by :func:`_declared_echo` and present only when non-empty: ``declared``
            (every target it typed, canonical spellings) and
            ``declared_no_near_match`` (``{slug[, score]}`` for the pause-resolving
            declarations that moved nothing on this call), each with a
            ``*_total`` sibling count when the group collapsed at
            :data:`_DECLARED_ECHO_BOUND`;
            OR a structured ``{error, code}`` failure (see spec §5) — a
            ``draft_drifted`` one also carries ``drifted_fields``, and a
            ``tool_call_markup`` one carries ``markup_spans`` (every hit, ``{field,
            span, offset}``; a list element's field reads ``name[i]``, and a prose
            field's offset counts characters of its CRLF-normalised value).
        """
        # === Phase A — validate everything in memory (no writes) ===

        # 1. Preconditions. os.path.exists, NOT manager construction (which creates the db).
        if not os.path.exists(self.config.decisions_file):
            return _record_error("not_initialized")

        # 2. Normalise CRLF, then validate. A stray \r perturbs the field regex and
        #    the canonical-core hash (same decision would hash differently across
        #    environments — compute_node_id normalizes, but normalize at the boundary).
        axiom = (axiom or "").replace("\r\n", "\n").replace("\r", "\n")
        rejected_paths = (rejected_paths or "").replace("\r\n", "\n").replace("\r", "\n")
        if context is not None:
            context = context.replace("\r\n", "\n").replace("\r", "\n")

        if not axiom.strip():
            return _record_error("empty_axiom")
        if not rejected_paths.strip():
            return _record_error("missing_rejected_paths")

        # 2b. Refuse (never strip) tool-call markup in every value this call writes,
        #     before any strip, slug fold, mechanism fold or hash can hide it. The prose
        #     fields are scanned CRLF-normalised; everything else raw.
        markup = _tool_call_markup_refusal([
            ("axiom", axiom), ("rejected_paths", rejected_paths), ("context", context),
            ("slug", slug), ("scope", scope), ("mechanisms", mechanisms),
            ("supersedes", supersedes), ("corrects", corrects), ("amends", amends),
            ("narrows", narrows), ("depends_on", depends_on), ("resolves", resolves),
            ("contradicts", contradicts), ("derives_from", derives_from), ("cites", cites),
        ])
        if markup is not None:
            return markup

        # 3. Reject (do NOT sanitise) content fields carrying structural tokens.
        #    Check the NON-stripped values: a leading-whitespace `  ## heading` is
        #    safe (the parser only splits on column-0 `##`), and stripping it first
        #    would falsely promote it to column 0. Storage stripping happens after.
        for field_text in (axiom, rejected_paths, context):
            if field_text and _contains_structural_token(field_text):
                return _record_error("parse_failed")

        axiom = axiom.strip()
        rejected_paths = rejected_paths.strip()
        context = context.strip() if context and context.strip() else None
        mechanisms = [m.strip() for m in mechanisms if m and m.strip()] if mechanisms else []
        scope = [s.strip() for s in scope if s and s.strip()] if scope else []
        if supersedes is not None:
            supersedes = supersedes.strip() or None
        if corrects is not None:
            corrects = corrects.strip() or None

        # Normalise the other typed relations into a stable-ordered map (supersedes and
        # corrects are handled separately — both are kill-edges that change computed
        # state and carry bespoke error codes).
        _provided = {
            "amends": amends, "narrows": narrows, "depends_on": depends_on,
            "resolves": resolves, "contradicts": contradicts,
            "derives_from": derives_from, "cites": cites,
        }
        extra_relations: Dict[str, str] = {}
        for _name, _label in _EXTRA_RELATIONS:
            _val = _provided.get(_name)
            if _val and _val.strip():
                extra_relations[_name] = _val.strip()

        # record_decision always mints a `decision`, but a `derives_from` edge must
        # ORIGINATE from an open question (open_question -> decision) — a decision can
        # never be its source, so this relation is always invalid here. Reject it in
        # the validate phase with a redirect, rather than letting it pass validation
        # and fail only at the store's kind CHECK ("validated but the commit failed").
        if "derives_from" in extra_relations:
            return _record_error("derives_from_on_decision")

        # 4. Slug — validate, don't mangle. The slug is now mandatory and explicit, and
        #    it is the handle other decisions cite (not part of the canonical-core
        #    identity, V1-D2). Normalise case/separators, but REJECT an over-length slug
        #    with an exact char count rather than silently truncating it — a silent trim
        #    would diverge the stored handle from the one the author already cited
        #    (self-inflicted citation rot, the exact failure the handle subsystem exists
        #    to prevent).
        slug = _normalize_slug(slug)
        if not slug:
            return _record_error("empty_slug")
        if len(slug) > _SLUG_MAX_LEN:
            return _record_error(
                "slug_too_long",
                slug=slug,
                length=len(slug),
                over=len(slug) - _SLUG_MAX_LEN,
                max=_SLUG_MAX_LEN,
            )

        # 5. Serialise to the canonical format (in memory only).
        lines = [f"### {slug}", "", f"**Decided:** {axiom}", f"**Rejected:** {rejected_paths}"]
        if mechanisms:
            lines.append(f"**Mechanisms:** {', '.join(mechanisms)}")
        if scope:
            lines.append(f"**Scope:** {', '.join(scope)}")
        if context:
            lines.append(f"**Context:** {context}")
        if supersedes:
            lines.append(f"**Supersedes:** {supersedes}")
        if corrects:
            lines.append(f"**Corrects:** {corrects}")
        for _name, _label in _EXTRA_RELATIONS:
            if _name in extra_relations:
                lines.append(f"**{_label}:** {extra_relations[_name]}")
        entry_text = "\n".join(lines) + "\n"

        # 6. Parse our entry back through the V1a tokenizer (sets .axiom/.topic +
        #    the relationship attrs), then run the graph-level checks as a read-only
        #    fast-fail. STRICT mode (no collector): a malformed self-serialized entry
        #    raises ParseError, which we map to the structured parse_failed code (G2).
        try:
            parsed = parse_entry_stream(entry_text, "decision")
        except ParseError:
            return _record_error("parse_failed")
        if len(parsed) != 1:
            return _record_error("parse_failed")
        entry = parsed[0]

        # A re-send carrying the pause's draft_digest is compared here: after the
        # entry is built (so it canonicalises exactly as the write does) and before any
        # store read, so a drifted draft is refused ahead of the exists short-circuit
        # (which would hide it) and under acknowledge_neighbors (which skips the review
        # a drifted draft would otherwise commit past). Pure; it can never raise.
        drift = _draft_drift(entry, draft_digest)
        if drift:
            return drift

        # Pre-validate supersedes with an EXACT match — comma-separated for a
        # multi-target supersede (each slug resolved independently; a lone slug is the
        # 1-element common case). Phase A read-only fast-fail: a miss on ANY target
        # returns an error naming that slug and writes nothing.
        # Each loop below also retains the resolved node's OWN slug spelling, keyed on
        # the casefolded caller spelling. That is the pause echo's canonical handle for
        # every declared target — including the ones the neighbour sweep never gathers,
        # whose stored spelling is computed nowhere else — and it costs a dict write on
        # a node already fetched.
        canonical_slugs: Dict[str, str] = {}
        # Beside it, the resolved node's id under the same key: what this validation
        # proved unique before the commit, which the acknowledged path's post-commit
        # review exempts by, so a slug that names more nodes after the commit cannot
        # move its exemptions (7b, D3).
        canonical_ids: Dict[str, str] = {}
        supersedes_slugs = _split_relation_slugs(supersedes)
        for _sup in supersedes_slugs:
            ids = self.store.resolve_slug(_sup)
            if not ids:
                return _record_error("supersedes_not_found", supersedes=_sup)
            if _ambiguous_target(ids):
                return _record_error("supersedes_ambiguous", supersedes=_sup)
            target = self.store.get_node(ids[0])
            if not target or target.get("slug", "").casefold() != _sup.casefold():
                return _record_error("supersedes_not_found", supersedes=_sup)
            canonical_slugs[_sup.casefold()] = target["slug"]
            canonical_ids[_sup.casefold()] = target["id"]
        if supersedes_slugs:
            entry.supersedes = supersedes_slugs  # List[str] shape (V1b multi-valued)

        # Pre-validate corrects with an EXACT match — the kill-edge twin of supersedes
        # (V1a's second kill-edge; the target leaves the active view), comma-separated
        # for multiple. Same Phase-A read-only fast-fail shape, so a miss writes nothing.
        corrects_slugs = _split_relation_slugs(corrects)
        for _cor in corrects_slugs:
            ids = self.store.resolve_slug(_cor)
            if not ids:
                return _record_error("corrects_not_found", corrects=_cor)
            if _ambiguous_target(ids):
                return _record_error("corrects_ambiguous", corrects=_cor)
            target = self.store.get_node(ids[0])
            if not target or target.get("slug", "").casefold() != _cor.casefold():
                return _record_error("corrects_not_found", corrects=_cor)
            canonical_slugs[_cor.casefold()] = target["slug"]
            canonical_ids[_cor.casefold()] = target["id"]
        if corrects_slugs:
            entry.corrects = corrects_slugs  # List[str] shape (V1b multi-valued)

        # Validate every other typed relation EXACTLY like supersedes — each must point
        # at a real, unambiguous decision; each is comma-separated multi-valued too.
        # Still Phase A: a miss on any target returns an error and writes nothing, so
        # the buffer stays byte-for-byte unchanged.
        for _name, _raw in extra_relations.items():
            _targets = _split_relation_slugs(_raw)
            for _t in _targets:
                err = self._validate_relation_target(
                    _name, _t, canonical_slugs=canonical_slugs,
                    canonical_ids=canonical_ids)
                if err:
                    return err
            if _targets:
                setattr(entry, _name, _targets)  # List[str] shape (V1b multi-valued)

        # Identity (slug-free canonical-core hash — V1-D2). Computed over the SAME
        # fields commit_parsed_entry hashes, so this pre-commit idempotency id equals
        # the commit id: a same-core re-record with a new --slug answers `exists` below
        # and renames nothing, never a spurious slug_collision (G3, V1-D16). A rename
        # is amend_commentary's `new_slug`.
        node_id = compute_node_id(
            kind=entry.kind,
            axiom=entry.axiom,
            mechanism_refs=entry.mechanisms,
            topic=entry.topic,
            questions_raised=entry.questions_raised,
        )

        # Idempotency (M2) fast-fail.
        existing = self.store.get_node(node_id)
        if existing:
            # Cross-source re-encounter audit (MI-4 / V1-D14) before the exists
            # short-circuit returns. record authors no **Source:** line, so on this
            # path entry.source is None → "user". Best-effort (note_source_reencounter
            # never raises), so the agentic exists-path can never crash on the audit.
            self.store.note_source_reencounter(
                node_id, existing["source"], entry.source or "user"
            )
            state = self._node_state(node_id)
            return self._exists_receipt(entry, existing, node_id,
                                        slug=existing["slug"], state=state)

        # Slug-collision fast-fail (exact match only).
        coll_id, _coll_node = self._exact_slug_node(entry.slug)
        if coll_id and coll_id != node_id:
            return _record_error("slug_collision", slug=entry.slug)

        # Near-duplicate review (P4) — still Phase A, so a pause writes NOTHING
        # (buffer byte-for-byte unchanged). Two partitions: the live neighbours, a
        # stamped decision-read surface (candidate_payload) the agent judges from, and
        # the tombstones, retired decisions the draft sits closer to than their floor
        # (7a). Surfacing BEFORE commit is the point: after commit the author can no
        # longer point a relation at either (a re-record is a no-op). Offline-safe (no
        # embeddings → no pause) and bypassable with acknowledge_neighbors=True. The
        # record surface fails OPEN on any pause-read fault: the check degrading must
        # never block the commit, but "couldn't check" must not read as "checked,
        # clean" either — the threaded notice lands on the created receipt (step 10).
        review_unavailable: Optional[str] = None
        # The near-dup review's cut statement, threaded to the pause or the created
        # receipt; absent when not cut.
        review_cut: Optional[str] = None
        if not acknowledge_neighbors:
            review = self._review_pause(entry, supersedes, corrects, extra_relations,
                                        canonical_slugs=canonical_slugs, own_id=node_id)
            if review.pause is not None:
                return review.pause
            review_unavailable, review_cut = review.unavailable, review.cut

        # === Phase B — the only writes, fully serialised under one lock ===
        try:
            with self.lock:
                # Re-run the authoritative state checks INSIDE the lock, BEFORE any
                # buffer write (closes the TOCTOU window: a racer that committed
                # since Phase A is now seen — and a rejection here still leaves the
                # buffer byte-for-byte unchanged, since auto-heal hasn't run yet).
                existing = self.store.get_node(node_id)
                if existing:
                    # TOCTOU re-check: a racer committed this node between Phase A
                    # and here. Same cross-source audit as gate 3 (rare; INSERT OR
                    # IGNORE makes any Phase-A/Phase-B overlap a harmless no-op).
                    # Best-effort, and the threading lock here is not a DB txn, so
                    # write_signal's own connection is fine.
                    self.store.note_source_reencounter(
                        node_id, existing["source"], entry.source or "user"
                    )
                    # One composer with gate 3, and one state read per return.
                    state = self._node_state(node_id)
                    return self._exists_receipt(entry, existing, node_id,
                                                slug=entry.slug, state=state)
                coll_id, _coll_node = self._exact_slug_node(entry.slug)
                if coll_id and coll_id != node_id:
                    return _record_error("slug_collision", slug=entry.slug)

                # Guarantee header + marker, then anchor the buffer for rollback.
                self.auto_heal_decisions_file()
                with open(self.config.decisions_file, "r", encoding="utf-8") as f:
                    original_content = f.read()

                # Compute the new buffer (newest-first, replacing ONLY the first marker).
                if _ENTRIES_MARKER in original_content:
                    new_content = original_content.replace(
                        _ENTRIES_MARKER, f"{_ENTRIES_MARKER}\n\n{entry_text}", 1
                    )
                else:
                    new_content = original_content.rstrip("\n") + f"\n\n{entry_text}"

                # Provenance (mirror perform_sync).
                entry.confirmed_by = actor
                entry.confirmed_at = _utc_now_iso()  # MI-10, as in perform_sync above

                # Write the buffer, then commit the graph. On ANY failure of either
                # — including an OSError on the write itself — roll the buffer back
                # so a failure leaves NO orphan entry, and return JSON (never raise).
                try:
                    atomic_file.write_source(self.config.decisions_file, new_content)
                    delta = self.store.commit_parsed_entry(entry)
                except (ValidationError, DatabaseError, OSError, CommitError) as commit_exc:
                    # Roll the buffer back so a failed write/commit leaves NO orphan
                    # (the sacred buffer-first + rollback contract). ``CommitError``
                    # carries the store-stage failure envelope — as of V1b a declared
                    # relation can fail at commit (e.g. a decision-source ``resolves``/
                    # ``derives_from`` violates the edge kind matrix → a loud
                    # ``kind_constraint_violation``, where V1a silently warn-deferred
                    # it). Surface the per-item messages so the agent sees the actionable
                    # field (P3 vector error), not a generic wall.
                    try:
                        atomic_file.write_source(self.config.decisions_file, original_content)
                    except Exception as restore_exc:
                        return {
                            "error": (
                                "The commit failed AND decisions.md may still hold the "
                                f"uncommitted entry (rollback error: {restore_exc}) — run "
                                f"'mitos sync' to reconcile. Underlying commit error: {commit_exc}."
                            ),
                            "code": "commit_failed",
                        }
                    reason = str(commit_exc)
                    if isinstance(commit_exc, CommitError) and commit_exc.failure:
                        reason = "; ".join(
                            i.message for i in commit_exc.failure.items
                        ) or reason
                    return _record_error("commit_failed", reason=reason)
        except Timeout:
            return _record_error(
                "commit_failed", reason="another Mitos process holds the decisions.md lock"
            )

        # 7b. One bounded rotation of the buffer's settled tail — the write landed, so
        #     it is `created` and only `created` reaches here. It runs after the write's
        #     lock hold releases, never woven into it, and takes its own acquisition,
        #     reading the live buffer there: that one read holds the entry just written
        #     plus anything a concurrent writer added since. `new_content` is exactly
        #     the file just written, so it pre-gates the count without a second read —
        #     below the threshold there is no acquisition at all. A stale gate either
        #     way is harmless: the selector re-counts over the live read. One call, no
        #     loop — the agent is blocked on this call, and the residual drains across
        #     later writes. Unlike the embed and render steps around it, a failure here
        #     prints nothing: it rides the receipt as data, and each renderer words it
        #     once (a print here as well would say it twice on the CLI text surface).
        rotation_report: Optional[Dict[str, Any]] = None
        if (settledness.buffered_entries(new_content)
                >= self.config.rotation_volume_threshold_entries):
            try:
                outcome = self._rotate_settled(window=settledness.ROTATION_WINDOW_ENTRIES)
            except Exception as e:
                rotation_report = _rotation_failure(e)
            else:
                if outcome is not None:
                    rotation_report = _rotation_report(outcome)

        # 8. Embed best-effort (queues to the outbox if Gemini/Qdrant are down).
        try:
            self._best_effort_embed(delta, entry)
        except Exception as e:
            # stderr: the MCP write tool shares this path and uses stdout for JSON-RPC.
            print(f"[Warning] Embedding step failed for '{entry.slug}': {describe_embed_failure(e)}", file=sys.stderr)

        # 9. Re-render live_axioms.md (a render failure must not fail the commit).
        #    The renderer records size-ceiling overflows on `.overflows` instead of
        #    printing them, so we can attach ONE debounced summary to the result below
        #    — after the success receipt — rather than burying it under a per-file wall.
        overflow_summary: Optional[str] = None
        try:
            renderer = MitosRenderer(self.config.workspace_dir)
            renderer.render_all(self.store)
            if renderer.overflows and hint_due(
                "scope_overflow_hint.json", self.config.workspace_dir,
                OVERFLOW_HINT_WINDOW_SECONDS,
            ):
                overflow_summary = summarize_overflows(renderer.overflows)
        except Exception as e:
            # stderr: the MCP write tool shares this path and uses stdout for JSON-RPC.
            print(f"[Warning] Failed to render active axioms: {str(e)}", file=sys.stderr)

        # 9b. The corpus's audit debt, read after the commit and outside the lock
        #     (it adds nothing to the §7.3 hold), and never able to fail the write.
        coherence_audit, audit_debt = self._audit_debt_fields(entry.slug)

        # 9c. The standing check notice: the last contradiction check's outcome, for
        #     an agent that was not there. Same place and same never-raising shape
        #     as 9b; None on a keyless workspace or a healthy record.
        check_notice = self._check_notice_field()

        # 9d. Under acknowledge_neighbors, the neighbours and the tombstones this write
        #     went past (F10; 7b): the pause's gather re-run after the embed, so it hits
        #     the embedding cache, exempting by the target ids Phase A validated. Same
        #     never-raising shape as 9b/9c; each list None when it could not be read.
        acknowledged = AcknowledgedReview(neighbors=None, tombstones=None)
        if acknowledge_neighbors:
            acknowledged_gatherings: List[Gathering] = []
            acknowledged = self._acknowledged_fields(
                entry, supersedes, corrects, extra_relations,
                target_ids=canonical_ids, gathering_sink=acknowledged_gatherings)
            # The re-run gather's cut, only when the live list it fills was produced: a
            # `None` list already says the read could not be trusted. A tombstone-only
            # fault keeps it, because the gather that was cut did run.
            if acknowledged.neighbors is not None and acknowledged_gatherings:
                review_cut = _review_cut_notice(acknowledged_gatherings[0],
                                                _NEIGHBOR_REVIEW_THRESHOLD,
                                                tombstone_floor=_TOMBSTONE_FLOOR)

        # 9e. The edges this write wired, each naming what it hit (B4). Same
        #     never-raising shape as 9b/9c/9d, so the literal below holds a value
        #     and no live store read.
        edges_created = self._edges_created_field(node_id, entry.slug)

        # 10. Return. A freshly recorded decision is always active. Everything below is
        #     post-commit read-back — the write contract is untouched (the commit
        #     already succeeded above).
        result: Dict[str, Any] = {
            "slug": entry.slug,
            "id": node_id,
            "state": "active",
            "embedding": self._embedding_status(node_id),
            "status": "created",
            "path": self.config.decisions_file,
            # A stamped identification echo, read back from the committed graph and
            # never re-derived from the author's input args: each edge names its
            # target's state, computed at receipt time (M3), and its modifier stamps,
            # plus the whole axiom on a relation that acts on its target (the topic on
            # `resolves`). It identifies what the write hit and is not a precedent
            # read, so no reasoning rides — a reader who needs the target's reasoning
            # dereferences it by slug. `null` target keys mean the read failed; `{}`
            # stamps mean it was read and held none. ADRs
            # `record-receipt-edges-are-a-stamped-read-surface-axiom-rides-acting-relations-only`
            # and `identification-echo-is-stamped-but-sub-letter-letter-default-governs-precedent-reads`.
            "edges_created": edges_created,
            "scope": entry.scope,
            # `mechanisms` is the gold source's authored form, read from the entry
            # step 6 parsed back from the serialised text (so an element carrying a
            # comma echoes as the split the markdown holds), never from the argument.
            # `mechanisms_normalized` names the fold, unconditionally — `{}` when
            # nothing changed, so an absent key never has to be interpreted.
            "mechanisms": list(entry.mechanisms_authored),
            "mechanisms_normalized": _mechanisms_fold_map(entry.mechanisms_authored),
            # Unconditional, and only here: a write that landed incurred coherence
            # debt, while `exists`/`needs_review`/every error wrote nothing and owe
            # nothing. Carrying the FACT on the field (rather than leaving the CLI
            # renderer to decide) is what keeps the shared text tail from leaking
            # the line onto a no-op — the renderer gates on the key's presence.
            # `audit_debt` carries the same counts so no agent parses the sentence;
            # None means unreadable, never zero.
            "coherence_audit": coherence_audit,
            "audit_debt": audit_debt,
        }
        # Honest degradation (KDD-4): when the pre-commit near-dup check could not run,
        # say so on the receipt — one calm sentence, only on a genuinely failed check
        # (never on clean-empty, unconfigured, or acknowledged-past records).
        if review_unavailable:
            result["neighbor_review_unavailable"] = review_unavailable
        # The review ran over less than it should have: its gather was cut above the
        # pause's floor (pre-commit, or the acknowledged path's re-run). Absent when the
        # case did not arise; never beside `neighbor_review_unavailable`.
        if review_cut:
            result["neighbor_review_cut"] = review_cut
        # Debounced, presentation-only: a one-line nudge naming the over-ceiling files,
        # their ceilings and its own once-a-day cadence, and no command — it is shared
        # by both surfaces (MCP returns it verbatim; CLI text prints it after the
        # receipt and adds its own selectored `mitos status -p` recipe line).
        if overflow_summary:
            result["scope_overflow"] = overflow_summary
        # What this call's rotation did, only when it moved, skipped or failed — never
        # the buffer's standing size, which is `mitos status`'s to report.
        if rotation_report:
            result["rotation"] = rotation_report
        # Present only when shown, so a healthy corpus pays zero bytes. The fact
        # alone: each renderer decides whether a recovery clause follows it.
        if check_notice:
            result["check_notice"] = check_notice
        # Keyed on the flag, not the value: None ("could not look") and [] ("looked,
        # none") are both answers, and the key's absence means the flag was not passed.
        if acknowledge_neighbors:
            result["acknowledged_neighbors"] = acknowledged.neighbors
            result["acknowledged_tombstones"] = acknowledged.tombstones
        return result
