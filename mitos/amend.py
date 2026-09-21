"""The pure half of the commentary amendment: validation, targeting, surgery, expectation.

``MitosSyncManager.amend_commentary`` edits one committed entry's mutable fields in the
``decisions.md`` buffer and re-commits it through parse→commit. Everything here is the
part of that verb that needs no lock, no store and no file: it decides whether a request
is honourable as a request, which buffered block it names, what bytes the edit writes,
and what the edited entry must parse back as. The manager supplies the graph through two
callables, so this module never imports the store.

Four properties are decisions, not details:

* **The field set is closed and derived.** ``EDITABLE_FIELDS`` is
  ``divergence.COMMENTARY_FIELDS`` plus ``scope`` — what the reconcile can carry, less
  edges (an edge edit changes computed state). A canonical-core key is refused with the
  kill-edge distinction as data (``ROUTES``), never one door.
* **The edit is surgery, never regeneration.** Only the edited fields' extents change;
  every other byte of the block, and of the buffer, stays as the author wrote it.
* **The surgery may be simple because it is fenced.** ``restore.verify_amended_buffer``
  re-parses the whole buffer and compares against ``expected_fingerprint``; any shape
  the surgery gets wrong is refused and rolled back, never written.
* **Result strings state causes and name no command.** Each surface composes its own
  recovery clause.

Tier 2: imports ``divergence`` (constants), ``identity``, ``markers``, ``parser``,
``restore``, ``scope_tags``, ``tool_markup`` — never ``store``, ``sync``, ``telemetry`` or a lock.
"""

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from mitos.divergence import COMMENTARY_FIELDS, RELATIONSHIP_FIELDS
from mitos.identity import SLUG_MAX_LEN, compute_node_id
from mitos.markers import TRANSCRIPT_CLOSE, TRANSCRIPT_OPEN
# The parser's own field-line regex and field map, imported by their private names: the
# surgery must find extents exactly where the tokenizer reads them (sync.py carries a
# narrower regex of the same name that lacks the `**Field**:` form).
from mitos.parser import FIELD_MAP, _FIELD_LINE_RE, parse_entry_stream, parse_header
from mitos.restore import BufferFidelityError, _entry_fingerprint
from mitos.scope_tags import normalize_scope_tags
from mitos.tool_markup import field_values, find_tool_call_markup

EDITABLE_FIELDS: Tuple[str, ...] = tuple(COMMENTARY_FIELDS) + ("scope",)

# The markdown label each editable field is written under. An explicit table rather than
# a reversal of FIELD_MAP, whose keys are lowercased and carry space-form aliases; a test
# pins every label back to FIELD_MAP.
FIELD_LABELS: Dict[str, str] = {
    "rejected_paths": "Rejected",
    "scope": "Scope",
    "invalidates_if": "Invalidates-If",
    "context": "Context",
}

# format-spec.md §1 order of the fields an insertion can land among (FIELD_MAP values).
_SPEC_ORDER: Tuple[str, ...] = (
    "core_axiom", "rejected_paths", "mechanisms", "scope", "invalidates_if", "context",
)

# The canonical-core refusal's routes: which relation serves which intent.
ROUTES: Dict[str, str] = {"wrong": "corrects", "outgrown": "supersedes", "partial": "amends"}

_CANONICAL_CORE_KEYS = frozenset({"axiom", "decided", "core_axiom", "mechanisms", "mechanism_refs"})
_EDGE_KEYS = frozenset(RELATIONSHIP_FIELDS) | {"edges"}
_NOT_EDITABLE_KEYS = frozenset(
    {"source", "transcript", "confirmed_by", "confirmed_at", "id", "kind"}
)

STATUS_AMENDED = "amended"
STATUS_UNCHANGED = "unchanged"
STATUS_REFUSED = "refused"
STATUS_NOT_FOUND = "not_found"
STATUS_ARCHIVED = "archived"
STATUS_UNCOMMITTED = "uncommitted"

REASON_CANONICAL_CORE = "canonical_core"
REASON_EDGES = "edges"
REASON_NOT_EDITABLE = "not_editable"
REASON_UNKNOWN_FIELD = "unknown_field"
REASON_INVALID_VALUE = "invalid_value"
REASON_NO_CHANGES = "no_changes"
REASON_OPEN_QUESTION = "open_question"
REASON_UNPARSEABLE = "unparseable"
REASON_DIVERGED = "diverged"
REASON_TOOL_CALL_MARKUP = "tool_call_markup"

# Error facts: the cause, never a recovery command (the renderers own recovery).
ERROR_FACTS: Dict[str, str] = {
    "slug_collision": (
        "Another active decision already uses the slug {requested!r}; the rename of "
        "{slug!r} was rolled back and decisions.md is unchanged."
    ),
    "commit_failed": (
        "The amendment could not be committed ({reason}); decisions.md was rolled back "
        "and is unchanged."
    ),
    "audit_unavailable": (
        "The attribution row could not be written ({reason}), so the amendment was not "
        "applied; decisions.md is unchanged."
    ),
    "lock_timeout": (
        "Another process held the decisions.md lock; nothing was written."
    ),
    "rollback_failed": (
        "The amendment failed and decisions.md could not be restored ({reason}); the "
        "file holds whole content, either its previous text or the unverified amendment."
    ),
}


@dataclass
class Target:
    """The one committed buffered block an amendment edits.

    Attributes:
        entry: The block as parsed from the locked read.
        node_id: The block's content-hash id, which is its committed node's id.
        node: The committed node's reader-facing dict.
    """

    entry: Any
    node_id: str
    node: Dict[str, Any]


@dataclass
class Miss:
    """A handle that names no amendable block; ``result`` is the in-band answer."""

    result: Dict[str, Any]


def refused(reason: str, fields: List[str]) -> Dict[str, Any]:
    """Builds a refusal result.

    Args:
        reason: One of the ``REASON_*`` values.
        fields: The request keys (or divergence species) the refusal is about.

    Returns:
        ``{"status": "refused", "reason", "fields"}``, plus ``routes`` for a
        canonical-core refusal.
    """
    result: Dict[str, Any] = {
        "status": STATUS_REFUSED, "reason": reason, "fields": sorted(fields),
    }
    if reason == REASON_CANONICAL_CORE:
        result["routes"] = dict(ROUTES)
    return result


def error_result(code: str, *, slug: str, **fields: Any) -> Dict[str, Any]:
    """Builds an environment-or-graph fault result in ``record``'s ``{error, code}`` shape.

    Args:
        code: A key of ``ERROR_FACTS``.
        slug: The handle the call named.
        **fields: The values the fact interpolates.

    Returns:
        ``{"error": <fact>, "code": code, "slug": slug}``.
    """
    return {"error": ERROR_FACTS[code].format(slug=slug, **fields), "code": code, "slug": slug}


def _has_line_break(text: str) -> bool:
    """Reports whether ``text`` holds any boundary ``str.splitlines`` splits on."""
    return bool(text) and text.splitlines() != [text]


def _leaves_comment_open(text: str) -> bool:
    """Reports whether ``text`` ends inside an HTML comment it opened.

    The lexical fallback's reader strips comments with state carried across lines, so
    an unterminated ``<!--`` in the buffer blanks every entry below it there. The fence
    parses with ``parse_entry_stream``, which keeps comments verbatim and cannot see it.
    """
    position, inside = 0, False
    while True:
        token = "-->" if inside else "<!--"
        found = text.find(token, position)
        if found < 0:
            return inside
        inside = not inside
        position = found + len(token)


def normalize_prose(field: str, value: Optional[str]) -> Optional[str]:
    """Returns the value a prose field parses back as, or ``None`` for an absent field.

    Mirrors ``parser._tokenize_entry``'s join discipline: every line stripped, blank
    lines dropped, then newline-joined (``rejected_paths``, ``context``) or space-joined
    (``invalidates_if``).

    Args:
        field: The field name.
        value: The requested value.

    Returns:
        The parsed form, or ``None`` when nothing non-blank remains.
    """
    if value is None:
        return None
    parts = [line.strip() for line in value.splitlines() if line.strip()]
    if not parts:
        return None
    return (" " if field == "invalidates_if" else "\n").join(parts)


def validate_changes(changes: Any) -> Optional[Dict[str, Any]]:
    """Refuses a request that cannot be honoured as a request. Pure; no I/O.

    Precedence when several keys are wrong: canonical core, edges, not editable,
    unknown, then invalid values, then tool-call markup. The markup scan reads only
    the values sent — never the target entry's untouched fields — so an entry that
    already holds markup stays repairable through this verb.

    Args:
        changes: The field → new-value mapping.

    Returns:
        A refusal result, or ``None`` when the request is well-formed. A
        ``tool_call_markup`` refusal names the base fields in ``fields`` and carries
        ``markup_spans`` (every hit, ``{field, span, offset}``, a scope tag as
        ``scope[i]``).
    """
    if not isinstance(changes, Mapping):
        return refused(REASON_INVALID_VALUE, [])
    if not changes:
        return refused(REASON_NO_CHANGES, [])

    buckets: Dict[str, List[str]] = {
        REASON_CANONICAL_CORE: [], REASON_EDGES: [], REASON_NOT_EDITABLE: [],
        REASON_UNKNOWN_FIELD: [],
    }
    for key in changes:
        name = str(key)
        folded = name.strip().casefold().replace("-", "_")
        if isinstance(key, str) and key in EDITABLE_FIELDS:
            continue
        if folded in _CANONICAL_CORE_KEYS:
            buckets[REASON_CANONICAL_CORE].append(name)
        elif folded in _EDGE_KEYS:
            buckets[REASON_EDGES].append(name)
        elif folded in _NOT_EDITABLE_KEYS:
            buckets[REASON_NOT_EDITABLE].append(name)
        else:
            buckets[REASON_UNKNOWN_FIELD].append(name)
    for reason, names in buckets.items():
        if names:
            return refused(reason, names)

    invalid = [field for field, value in changes.items() if not _value_is_valid(field, value)]
    if invalid:
        return refused(REASON_INVALID_VALUE, invalid)

    hits = find_tool_call_markup(
        pair for field, value in changes.items() for pair in field_values(field, value))
    if hits:
        result = refused(REASON_TOOL_CALL_MARKUP,
                         list({hit["field"].split("[", 1)[0] for hit in hits}))
        result["markup_spans"] = hits
        return result
    return None


def _value_is_valid(field: str, value: Any) -> bool:
    """Checks one editable field's requested value."""
    if field == "scope":
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            return False
        for tag in value:
            if not isinstance(tag, str):
                return False
            # A comma splits the tag on re-parse; a line break mints lines of its own.
            if "," in tag or _has_line_break(tag) or _leaves_comment_open(tag):
                return False
        return True
    if field == "slug":
        if not isinstance(value, str) or not value or len(value) > SLUG_MAX_LEN:
            return False
        if any(ch.isspace() for ch in value) or _leaves_comment_open(value):
            return False
        # The heading split must read it back as the whole slug, no date, no title.
        try:
            return parse_header(f"### {value}") == (value, None, None)
        except ValueError:
            return False
    if field == "rejected_paths":
        # M5: a decision's rejected paths are required, so they cannot be cleared.
        return (
            isinstance(value, str)
            and normalize_prose(field, value) is not None
            and not _leaves_comment_open(value)
        )
    # invalidates_if / context: None or "" removes the field.
    if value is None:
        return True
    return isinstance(value, str) and not _leaves_comment_open(value)


def entry_node_id(entry: Any) -> str:
    """Returns a parsed decision entry's content-hash id, the id its commit would mint."""
    return compute_node_id(kind="decision", axiom=entry.axiom, mechanism_refs=entry.mechanisms)


def classify_target(
    buffer_text: str,
    *,
    slug: str,
    resolve: Callable[[str], Optional[Dict[str, Any]]],
    lookup: Callable[[str], Optional[Dict[str, Any]]],
) -> Union[Target, Miss]:
    """Names the committed buffered block a handle refers to, or the miss it is.

    Decided by block facts over the one locked read: a buffered block bears the handle
    (by slug, or by id when the handle resolves to a node no buffered slug names), and
    that block hashes to a committed node or it does not. Never the divergence fold,
    never an archive read.

    Args:
        buffer_text: The buffer as read under the lock.
        slug: The handle the caller named.
        resolve: ``GraphStore.resolve_handle``.
        lookup: ``GraphStore.get_node``.

    Returns:
        The ``Target``, or a ``Miss`` carrying ``not_found`` / ``archived`` /
        ``uncommitted`` / a refusal (``open_question``, ``unparseable``).

    Raises:
        BufferFidelityError: When the buffer already holds the target ambiguously
            (two copies of one id, or two committed blocks the handle cannot choose
            between) — a splice there cannot be proven.
    """
    failures: List[Any] = []
    entries = parse_entry_stream(buffer_text, "decision", failures=failures)
    key = slug.casefold()
    node = resolve(slug)
    ids = [entry_node_id(entry) for entry in entries]

    bearing = [i for i, entry in enumerate(entries) if (entry.slug or "").casefold() == key]
    if not bearing and node is not None and node.get("kind") == "decision":
        bearing = [i for i, node_id in enumerate(ids) if node_id == node.get("id")]

    if bearing:
        committed = [i for i in bearing if lookup(ids[i]) is not None]
        if not committed:
            return Miss({"status": STATUS_UNCOMMITTED})
        committed_ids = {ids[i] for i in committed}
        if node is not None and node.get("id") in committed_ids:
            chosen = node["id"]
        elif len(committed_ids) == 1:
            chosen = next(iter(committed_ids))
        else:
            raise BufferFidelityError(
                f"the buffer holds {len(committed_ids)} committed entries bearing {slug!r} "
                "and none is the node the handle resolves to"
            )
        copies = [i for i, node_id in enumerate(ids) if node_id == chosen]
        if len(copies) != 1:
            raise BufferFidelityError(
                f"the buffer holds {len(copies)} copies of the entry {slug!r}; an edit to "
                "one of them cannot be proven"
            )
        target_node = lookup(chosen)
        return Target(entry=entries[copies[0]], node_id=chosen, node=target_node or {})

    if any((failure.slug or "").casefold() == key for failure in failures):
        return Miss(refused(REASON_UNPARSEABLE, []))
    if node is None:
        return Miss({"status": STATUS_NOT_FOUND})
    if node.get("kind") != "decision":
        return Miss(refused(REASON_OPEN_QUESTION, []))
    return Miss({"status": STATUS_ARCHIVED, "id": node.get("id")})


def expected_fingerprint(before_entry: Any, changes: Mapping[str, Any]) -> Dict[str, Any]:
    """Returns the fingerprint the amended entry must parse back as.

    The before-fingerprint with each requested value in its parsed form: prose joined
    as the tokenizer joins it, scope through ``normalize_scope_tags``, the slug exact
    (a case-only rename is a change).

    Args:
        before_entry: The target as parsed before the edit.
        changes: The validated request.

    Returns:
        A ``restore._entry_fingerprint``-shaped dict.
    """
    fingerprint = _entry_fingerprint(before_entry)
    for field, value in changes.items():
        if field == "slug":
            fingerprint["slug"] = value
        elif field == "scope":
            fingerprint["scope"] = normalize_scope_tags(value)
        else:
            fingerprint[field] = normalize_prose(field, value)
    return fingerprint


def fingerprint_matches(fingerprint: Dict[str, Any], entry: Any) -> bool:
    """Reports whether a request would leave ``entry`` as it is — the ``unchanged`` test.

    An empty prose field line (``""``) and an absent one (``None``) are the same absence,
    as the divergence comparison and the graph treat them; clearing an already-empty
    field is no change, and must not write a row whose ``fields_changed`` is empty.

    Args:
        fingerprint: The expected fingerprint.
        entry: The target as parsed before the edit.

    Returns:
        True when the two agree.
    """
    def _key(value: Dict[str, Any]) -> str:
        folded = dict(value)
        for field in ("rejected_paths", "invalidates_if", "context"):
            folded[field] = folded.get(field) or None
        return json.dumps(folded, sort_keys=True, default=str)

    return _key(fingerprint) == _key(_entry_fingerprint(entry))


def _line_body(line: str) -> str:
    """Returns a ``splitlines(keepends=True)`` element without its terminator."""
    parts = line.splitlines()
    return parts[0] if parts else ""


def _field_extents(block: List[str]) -> List[Tuple[Optional[str], int, int]]:
    """Locates every field's extent in a block, as the tokenizer reads them.

    An extent is a field line plus its continuation lines, up to the next field line of
    any name, a transcript marker, or the block's end; trailing blank lines stay outside
    it. Lines inside a transcript span belong to no field.

    Args:
        block: The block's lines with terminators, heading first.

    Returns:
        ``(attribute or None for an unknown field, start, end)`` triples, end exclusive.
    """
    extents: List[Tuple[Optional[str], int, int]] = []
    current: Optional[List[Any]] = None
    in_transcript = False
    for index in range(1, len(block)):
        body = _line_body(block[index])
        stripped = body.strip()
        if not in_transcript and stripped == TRANSCRIPT_OPEN:
            in_transcript, current = True, None
            continue
        if in_transcript:
            if stripped == TRANSCRIPT_CLOSE:
                in_transcript = False
            continue
        if stripped == TRANSCRIPT_CLOSE:
            current = None
            continue
        match = _FIELD_LINE_RE.match(body)
        if match:
            current = [FIELD_MAP.get(match.group("field").strip().lower()), index, index + 1]
            extents.append(current)  # type: ignore[arg-type]
        elif current is not None and stripped:
            current[2] = index + 1
    return [(attr, start, end) for attr, start, end in extents]


def _render_field(field: str, value: Any, newline: str) -> List[str]:
    """Renders one field's lines, or ``[]`` when the value clears the field."""
    label = FIELD_LABELS[field]
    if field == "scope":
        tags = normalize_scope_tags(value)
        return [f"**{label}:** {', '.join(tags)}{newline}"] if tags else []
    if value is None:
        return []
    lines = [line.rstrip() for line in value.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return []
    rendered = [f"**{label}:** {lines[0].lstrip()}{newline}"]
    rendered.extend(f"{line}{newline}" for line in lines[1:])
    return rendered


def apply_changes(buffer_text: str, target: Target, changes: Mapping[str, Any]) -> str:
    """Applies a validated request to the target block by field-line surgery.

    Each edited field's extents are replaced (by one extent, at the last one's place) or
    removed; an absent field is inserted after the latest present field that precedes it
    in ``format-spec.md`` order; a rename rewrites only the heading's slug token. Every
    byte outside the edited extents is kept.

    Args:
        buffer_text: The buffer as read under the lock.
        target: The classified target.
        changes: The validated request.

    Returns:
        The buffer text with the amendment applied.

    Raises:
        BufferFidelityError: If the heading's slug token cannot be located for a rename.
    """
    lines = buffer_text.splitlines(keepends=True)
    start, end = target.entry.line_start - 1, target.entry.line_end
    block = lines[start:end]
    heading_ending = block[0][len(_line_body(block[0])):]
    newline = heading_ending if heading_ending in ("\n", "\r\n") else "\n"

    extents = _field_extents(block)
    operations: List[Tuple[int, int, int, List[str]]] = []  # (start, rank, end, lines)
    for field, value in changes.items():
        if field == "slug":
            continue
        rendered = _render_field(field, value, newline)
        rank = _SPEC_ORDER.index(field)
        own = [extent for extent in extents if extent[0] == field]
        if own:
            for _, ext_start, ext_end in own[:-1]:
                operations.append((ext_start, rank, ext_end, []))
            operations.append((own[-1][1], rank, own[-1][2], rendered))
        elif rendered:
            preceding = [
                extent for extent in extents
                if extent[0] in _SPEC_ORDER[:rank]
            ]
            if preceding:
                latest_rank = max(_SPEC_ORDER.index(extent[0]) for extent in preceding)
                anchor = [e for e in preceding if _SPEC_ORDER.index(e[0]) == latest_rank][-1]
                position = anchor[2]
            else:
                position = 1
            operations.append((position, rank, position, rendered))

    # Apply from the bottom up; at one insertion point the later spec field goes first,
    # so the earlier one lands above it.
    for op_start, _, op_end, rendered in sorted(operations, key=lambda op: (op[0], op[1]), reverse=True):
        if rendered and op_start == op_end and op_start > 0:
            previous = block[op_start - 1]
            if previous == _line_body(previous):
                block[op_start - 1] = previous + newline
        block[op_start:op_end] = rendered

    if "slug" in changes:
        body = _line_body(block[0])
        ending = block[0][len(body):]
        hashes = len(body) - len(body.lstrip("#"))
        _, date, _ = parse_header(body)
        search_from = body.find(date, hashes) + len(date) if date else hashes
        position = body.find(target.entry.slug, search_from)
        if position < 0:
            raise BufferFidelityError(
                f"the heading of {target.entry.slug!r} does not carry its slug where the "
                "parser reads it"
            )
        block[0] = (
            body[:position] + changes["slug"] + body[position + len(target.entry.slug):]
            + ending
        )

    return "".join(lines[:start] + block + lines[end:])
