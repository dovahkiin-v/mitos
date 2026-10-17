"""The ``rejected_paths`` enumeration: one parse from a free-text field to items (M5).

``rejected_paths`` is free text, written in many shapes. This leaf turns any such
string into ordered **items** (one rejected option plus its reason each), each item
into one or more **parts** that fit the embedding model's input, and each part into
a **point id** under its owner node. It is the single parser that the rejection
index and every display use, so there is exactly one answer to "what is one
rejected option" inside a field.

**These rules are part of the rejection index's identity.** Item ids depend on the
grammar, the merge rule, the hash input and the part bound, across builds and
platforms. Once anything is embedded through them, a change re-splits fields and
re-ids their items, which is an announced re-seed, never a silent edit. The rules
are pinned by the trace table in ``tests/test_rejections.py`` and by Harbor's
enumerations in Layer A's oracle (``tests/golden/oracle.reference.json``), so a
change arrives as a red row or a reviewed oracle diff. The reasoning lives in the
ADRs ``rejection-item-identity-text-drops-its-marker-and-shares-the-canonical-core-normalization``,
``rejection-item-split-is-total-and-a-pure-function-of-the-text-in-any-script``,
``rejected-paths-enumeration-lets-the-list-form-that-begins-first-decide-and-never-falls-through``
(the grammar) and
``rejection-item-point-id-hashes-kind-tagged-canonical-json-of-owner-id-and-part-text``
(the point id).

The contracts consumers rely on:

* **Totality.** :func:`enumerate_rejected_paths` never raises for any ``str``. A
  field with no non-whitespace character gives ``shape="empty"`` and no items; any
  other field gives at least one item. A field in a language that writes no
  ``REJECTED`` falls to the language-neutral forms (numbering, bullets) or to one
  prose item, never to none.
* **Coverage.** Item spans are contiguous and exactly partition ``[0, len(field))``,
  so ``"".join(i.text for i in items) == field``. Text before the first marker
  belongs to item 1; whitespace between items belongs to the earlier item. So
  ``item.text`` carries the inter-item separator (item 1 of ``(1) A⏎(2) B`` ends with
  ``"\\n"``), and a renderer prints ``item.text.rstrip()``. ``item.text`` is the
  verbatim slice and never changes; the markdown stays as authored (M7/P6).
* **Parts.** Every part is non-empty, fits in :data:`ITEM_PART_BOUND_BYTES` UTF-8
  bytes, and is a fixpoint of ``identity.canonical_core_string_norm``. Joined, the
  parts' non-whitespace characters equal the identity text's.
* **Identity.** ``item_point_id(owner, text)`` depends only on ``(owner, text)``.
  Ordinal, part index and position never enter it, so inserting, removing or
  renumbering options re-ids only the items whose own text changed. Composed and
  decomposed (NFC-equivalent) fields give identical ids. :func:`rejection_points`
  collapses byte-identical items or parts under one owner to one point, keeping the
  first occurrence; :func:`enumerate_rejected_paths` never dedupes, because coverage
  and display need both.

**The data join key.** The owner id fed to :func:`item_point_id` is the node's
``nodes.id`` (the 64-hex canonical-core digest), never its UUID form and never a
slug. The text fed is ``ItemPart.text``, exactly as this leaf returns it. The
hashed text and the embedded text are one projection, normalized by
``identity.py``'s canonical-core discipline (NFC, UTF-8, the pinned UCD), as a
node's id and embedding text are (M2).

**The format epoch inherits this leaf.** An epoch that changes what
``rejected_paths`` stores re-splits those fields and re-ids their items, as an
announced re-seed. The ``(N) … REJECTED —`` form the skill teaches sits outside
``format-spec.md``; if a later epoch takes it into the spec, this grammar must
recognise what the spec then says.

This module is a pure **Tier-1** leaf: the stdlib plus ``mitos.identity`` (itself a
stdlib-only leaf), and nothing else from ``mitos``. It reads no environment, no
working directory and no config, and it defines no normalization of its own.
``recall.py`` receives it **injected**, never imported, because ``recall`` sits in
the check family's fenced import closure.
"""

import hashlib
import re
import unicodedata
import uuid
from dataclasses import dataclass
from typing import AbstractSet, Any, Dict, FrozenSet, List, Mapping, NamedTuple, Optional, Tuple

from mitos.identity import canonical_core_json_form, canonical_core_string_norm

# The part bound is an identity rule, so it is a constant and never read from config:
# a per-workspace MITOS_MODEL_OVERRIDE_EMBEDDING must not change an item's id.
#
# Derivation (2026-10-15). models.MODEL_IDS["EMBEDDING"] is gemini-embedding-2, whose
# input limit is 8,192 tokens (the Gemini API model page and the Vertex AI page agree).
# embeddings.py prepends "search_document: " (17 bytes) on both document paths. A
# SentencePiece-style tokenizer with byte fallback emits at most one token per input
# byte, plus special tokens, so 8192 - 17 = 8175 bytes always fit. 8,000 keeps 175
# tokens of headroom for special or control tokens. tests/test_rejections.py
# tripwires the model id, the prefix and this inequality, so a model swap reds offline
# until someone re-derives the bound (a re-seed decision).
ITEM_PART_BOUND_BYTES = 8000
BOUND_DERIVED_FOR_MODEL = "gemini-embedding-2"
BOUND_MODEL_TOKEN_LIMIT = 8192
DOCUMENT_PREFIX_BYTES = 17

SHAPES = ("numbered", "bulleted", "rejected_marker", "prose", "empty")

# Digits are [0-9], never \d: \d matches Arabic-Indic and other digits, and int()
# accepts them. Leading zeros are refused.
_PAREN_MARKER_RE = re.compile(r"\(([1-9][0-9]{0,2})\)")
# Line-start `N.` / `N)` followed by a space or tab. `^` with re.M matches after
# "\n" only (never after U+2028 or \x85), which is the line rule here.
_LINE_NUMBER_MARKER_RE = re.compile(r"^[ \t]*([1-9][0-9]{0,2})[.)](?=[ \t])", re.M)
_BULLET_RE = re.compile(r"^([ \t]*)([-*\u2022])[ \t]", re.M)
_REJECTED_RE = re.compile(r":[ \t]*REJECTED\b")
_SENTENCE_STOP_RE = re.compile(r"[.;!?][ \t]+")
_HSPACE_RE = re.compile(r"[ \t]*")
_PARAGRAPH_RE = re.compile(r"(?=(\n[ \t]*\n))")

# What may precede a mid-line `(N)` (after at least one space or tab) for it to count
# as a marker rather than a reference (`path (2)`) or notation (`O(1)`).
_NUMBERED_BOUNDARY_CHARS = frozenset(".;:!?)\u3002\uff1b\uff1a\uff01\uff1f")
_SENTENCE_END_CHARS = frozenset(".!?;")
_CJK_SENTENCE_END_CHARS = frozenset("\u3002\uff01\uff1f\uff1b")
_OWNER_ID_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ItemPart:
    """One embeddable part of an item; most items have exactly one."""

    index: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the part."""
        return {"index": self.index, "text": self.text}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ItemPart":
        """Restores a part, ignoring keys it does not own (such as ``point_id``)."""
        return cls(index=data["index"], text=data["text"])


@dataclass(frozen=True)
class RejectedItem:
    """One rejected option and its reason, as a span of the raw field.

    ``start``/``end`` are ``str`` indices into the raw field, ``text`` is
    ``field[start:end]`` verbatim, and ``identity_text`` is the span with the item's
    own marker removed, then ``canonical_core_string_norm``-ed.
    """

    ordinal: int
    start: int
    end: int
    text: str
    identity_text: str
    parts: List[ItemPart]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the item; the span is a list, never a tuple."""
        return {
            "ordinal": self.ordinal,
            "span": [self.start, self.end],
            "text": self.text,
            "identity_text": self.identity_text,
            "parts": [part.to_dict() for part in self.parts],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RejectedItem":
        """Restores an item from :meth:`to_dict`'s shape."""
        start, end = data["span"]
        return cls(
            ordinal=data["ordinal"],
            start=start,
            end=end,
            text=data["text"],
            identity_text=data["identity_text"],
            parts=[ItemPart.from_dict(part) for part in data["parts"]],
        )


@dataclass(frozen=True)
class Enumeration:
    """A field's items and the list form that produced them (one of :data:`SHAPES`)."""

    shape: str
    items: List[RejectedItem]

    @property
    def parsed(self) -> bool:
        """Whether a list form was recognised (not ``prose`` or ``empty``)."""
        return self.shape not in ("prose", "empty")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the enumeration."""
        return {"shape": self.shape, "items": [item.to_dict() for item in self.items]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Enumeration":
        """Restores an enumeration from :meth:`to_dict`'s shape."""
        return cls(
            shape=data["shape"],
            items=[RejectedItem.from_dict(item) for item in data["items"]],
        )


@dataclass(frozen=True)
class PointRef:
    """The first occurrence (in enumeration order) that produced a point id."""

    ordinal: int
    part_index: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the reference."""
        return {"ordinal": self.ordinal, "part_index": self.part_index, "text": self.text}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PointRef":
        """Restores a reference from :meth:`to_dict`'s shape."""
        return cls(ordinal=data["ordinal"], part_index=data["part_index"], text=data["text"])


@dataclass(frozen=True)
class _Marker:
    """A recognised list marker.

    ``position`` is where its item starts: the line start when only spaces or tabs
    precede the marker on its line, otherwise the marker's own index (for
    rejected_marker, the option start). ``body_start`` is where the item's text begins
    once the marker is removed; it equals ``position`` when nothing is removed.
    """

    position: int
    body_start: int


def _line_start(field: str, index: int) -> int:
    return field.rfind("\n", 0, index) + 1


def _only_hspace_before(field: str, index: int) -> bool:
    return _HSPACE_RE.fullmatch(field, _line_start(field, index), index) is not None


def _position(field: str, index: int) -> int:
    return _line_start(field, index) if _only_hspace_before(field, index) else index


def _after_hspace(field: str, index: int) -> int:
    return _HSPACE_RE.match(field, index).end()


def _at_numbered_boundary(field: str, index: int) -> bool:
    """Whether a numbered token at ``index`` sits where a list marker can sit."""
    if _only_hspace_before(field, index):
        return True
    j = index
    while j > 0 and field[j - 1] in " \t":
        j -= 1
    return j < index and j > 0 and field[j - 1] in _NUMBERED_BOUNDARY_CHARS


def _numbered_markers(field: str) -> List[Tuple[int, int]]:
    """Returns ``(token_index, token_end)`` for each accepted numbered marker.

    A token counts only if its number is the next expected one (1, then 2, …) and it
    sits at a numbered boundary, so a skipped or restarted number stays inside the
    previous item and a reference or notation is never a marker.
    """
    tokens = [(m.start(), m.end(), m.group(1)) for m in _PAREN_MARKER_RE.finditer(field)]
    tokens += [(m.start(1), m.end(), m.group(1)) for m in _LINE_NUMBER_MARKER_RE.finditer(field)]
    tokens.sort()
    accepted: List[Tuple[int, int]] = []
    for start, end, digits in tokens:
        if int(digits) == len(accepted) + 1 and _at_numbered_boundary(field, start):
            accepted.append((start, end))
    return accepted


def _bullets(field: str) -> List[Tuple[int, int, int]]:
    """Returns ``(line_start, bullet_index, bullet_end)`` for every line-start bullet."""
    return [(m.start(), m.start(2), m.end(2)) for m in _BULLET_RE.finditer(field)]


def _option_start(field: str, colon: int, stops: List[int]) -> int:
    """The latest option boundary at or before a ``: REJECTED`` colon, on its line.

    A boundary is the line start, the end of a ``[.;!?][ \\t]+`` run, or the end of
    another list's marker token plus the spaces after it (``stops``). A bare ``:`` is
    deliberately not a boundary: an option may hold a colon before its ``: REJECTED``.
    """
    line_start = _line_start(field, colon)
    best = line_start
    for m in _SENTENCE_STOP_RE.finditer(field, line_start, colon + 1):
        if m.end() <= colon:
            best = max(best, m.end())
    for stop in stops:
        if line_start <= stop <= colon:
            best = max(best, stop)
    return best


def _form_markers(field: str) -> Dict[str, List[_Marker]]:
    """Finds each list form's markers, in field order."""
    numbered_tokens = _numbered_markers(field)
    numbered = [
        _Marker(_position(field, start), _after_hspace(field, end))
        for start, end in numbered_tokens
    ]

    all_bullets = _bullets(field)
    bulleted: List[_Marker] = []
    if all_bullets:
        indent = field[all_bullets[0][0]:all_bullets[0][1]]
        for line_start, index, end in all_bullets:
            if field[line_start:index] == indent:
                bulleted.append(_Marker(line_start, _after_hspace(field, end)))

    # The marker stop: an accepted numbered marker or any line-start bullet ends an
    # option boundary, so the scan never leaps past `(1)` onto a preamble.
    stops = [_after_hspace(field, end) for _, end in numbered_tokens]
    stops += [_after_hspace(field, end) for _, _, end in all_bullets]
    rejected: List[_Marker] = []
    for m in _REJECTED_RE.finditer(field):
        # `REJECTED` + a combining mark is not a marker: NFC would fuse the mark into
        # the `D` (`D` + U+0307 -> U+1E0A), so the marker would exist in one of two
        # NFC-equivalent fields only and their ids would differ.
        if m.end() < len(field) and unicodedata.category(field[m.end()]).startswith("M"):
            continue
        start = _option_start(field, m.start(), stops)
        if not rejected or start > rejected[-1].position:
            rejected.append(_Marker(start, start))

    return {"numbered": numbered, "bulleted": bulleted, "rejected_marker": rejected}


def _select(field: str) -> Tuple[str, List[_Marker]]:
    """Picks the list form that begins first; no fall-through if it does not qualify."""
    forms = _form_markers(field)
    found = [(markers[0].position, rank, name)
             for rank, (name, markers) in enumerate(forms.items()) if markers]
    if not found:
        return "prose", []
    _, _, name = min(found)
    markers = forms[name]
    if len(markers) >= 2:
        return name, markers
    if name != "rejected_marker" and not field[:markers[0].position].strip():
        return name, markers
    return "prose", []


def _cut_candidates(text: str, level: str) -> List[int]:
    """Every cut position ``p`` a level offers, ascending (the boundary ends at ``p``)."""
    if level == "paragraph":
        return [m.start() + len(m.group(1)) for m in _PARAGRAPH_RE.finditer(text)]
    if level == "line":
        return [i + 1 for i, ch in enumerate(text) if ch == "\n"]
    if level == "sentence":
        out = []
        for i, ch in enumerate(text):
            if ch in _CJK_SENTENCE_END_CHARS:
                out.append(i + 1)
            elif ch in _SENTENCE_END_CHARS and i + 1 < len(text) and text[i + 1].isspace():
                out.append(i + 1)
        return out
    return [i + 1 for i, ch in enumerate(text) if ch.isspace()]


_SPLIT_LEVELS = ("paragraph", "line", "sentence", "word")


def _utf8_len(s: str) -> int:
    # surrogatepass: a lone surrogate (a valid str, never a decoded file) counts as
    # its 3-byte form instead of raising, so the enumeration stays total.
    return len(s.encode("utf-8", "surrogatepass"))


def _split(identity_text: str) -> List[str]:
    """Greedily cuts an identity text into parts of at most the bound (total, pure).

    Per part: the largest cut at the first level (paragraph, line, sentence, word)
    that offers one within the bound, else the largest code-point cut within it. The
    part is ``T[:p].strip()`` and the remainder ``T[p:].lstrip()`` on every level.
    """
    parts: List[str] = []
    rest = identity_text
    while rest:
        if _utf8_len(rest) <= ITEM_PART_BOUND_BYTES:
            parts.append(rest)
            break
        # cum[i] is the UTF-8 length of rest[:i], built only as far as a cut can reach:
        # past the first over-bound index and the whitespace after it, every p strips
        # to more than the bound. Scanning the whole remainder per part would make a
        # long field quadratic.
        cum = [0]
        window = 0
        while window < len(rest) and cum[-1] <= ITEM_PART_BOUND_BYTES:
            cum.append(cum[-1] + _utf8_len(rest[window]))
            window += 1
        while window < len(rest) and rest[window].isspace():
            cum.append(cum[-1] + _utf8_len(rest[window]))
            window += 1
        # One character past the window lets the sentence level see its whitespace.
        scan = rest[:window + 1]
        cut: Optional[int] = None
        for level in _SPLIT_LEVELS:
            for p in reversed(_cut_candidates(scan, level)):
                if p > window:
                    continue
                e = p
                while e > 0 and rest[e - 1].isspace():
                    e -= 1
                if e > 0 and cum[e] <= ITEM_PART_BOUND_BYTES:
                    cut = p
                    break
            if cut is not None:
                break
        if cut is None:
            cut = max(p for p in range(1, window + 1) if cum[p] <= ITEM_PART_BOUND_BYTES)
        parts.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
    return parts


def _make_item(ordinal: int, field: str, start: int, end: int, body: str) -> RejectedItem:
    identity_text = canonical_core_string_norm(body)
    parts = [ItemPart(index=i, text=t) for i, t in enumerate(_split(identity_text))]
    return RejectedItem(ordinal=ordinal, start=start, end=end, text=field[start:end],
                        identity_text=identity_text, parts=parts)


def enumerate_rejected_paths(field: str) -> Enumeration:
    """Turns a ``rejected_paths`` string into ordered items (total, never raises).

    The grammar: three list forms (``numbered`` ``(N)`` / line-start ``N.`` ``N)``,
    ``bulleted`` ``-`` ``*`` ``•`` at the first bullet's indentation, and
    ``rejected_marker`` ``X: REJECTED``); the form that begins first decides, and if
    it does not qualify the field is one ``prose`` item. Items whose post-marker body
    is empty merge into the next item (the last into the previous); the preamble
    joins the first surviving item. Ordinals are list positions after merging.

    Args:
        field: The raw ``rejected_paths`` text, exactly as stored.

    Returns:
        The enumeration; ``shape="empty"`` with no items only for a field with no
        non-whitespace character.
    """
    if not field.strip():
        return Enumeration(shape="empty", items=[])
    shape, markers = _select(field)
    if shape != "prose":
        # One segment per marker; the marker and its spaces are removed from the body.
        segments = []
        for k, marker in enumerate(markers):
            seg_end = markers[k + 1].position if k + 1 < len(markers) else len(field)
            body = field[min(marker.body_start, seg_end):seg_end]
            segments.append((marker.position, body))
        # Empty bodies merge into the next surviving item; a trailing run of empties
        # merges into the previous one. An empty body contributes no text.
        groups: List[Tuple[int, str]] = []
        pending: Optional[int] = None
        for position, body in segments:
            if not body.strip():
                if pending is None:
                    pending = position
                continue
            groups.append((position if pending is None else pending, body))
            pending = None
        if groups:
            items = []
            for k, (start, body) in enumerate(groups):
                if k == 0:
                    body = field[:start] + body
                    start = 0
                end = groups[k + 1][0] if k + 1 < len(groups) else len(field)
                items.append(_make_item(k + 1, field, start, end, body))
            return Enumeration(shape=shape, items=items)
    return Enumeration(shape="prose", items=[_make_item(1, field, 0, len(field), field)])


def _check_owner(owner_id: str) -> None:
    if not isinstance(owner_id, str) or _OWNER_ID_RE.fullmatch(owner_id) is None:
        raise ValueError(
            f"owner_id must be a node id (64 lowercase hex characters), got {owner_id!r}"
        )


def item_point_id(owner_id: str, part_text: str) -> str:
    """Computes a part's point id under its owner node.

    ``sha256`` of the kind-tagged canonical JSON (MI-7) of ``{kind, owner, text}``,
    formatted as a lowercase 8-4-4-4-12 UUID of its first 128 bits (the same form
    ``vector_store.hash_to_uuid`` gives). The ``kind`` tag keeps item hashes in their
    own domain, apart from node digests.

    Args:
        owner_id: The owner node's ``nodes.id``: 64 lowercase hex characters.
        part_text: ``ItemPart.text`` exactly as :func:`enumerate_rejected_paths`
            returns it.

    Returns:
        The point id.

    Raises:
        ValueError: If ``owner_id`` is not 64 lowercase hex characters (a slug or a
            UUID passed by mistake would otherwise mint plausible wrong ids), or
            (as ``UnicodeEncodeError``) if ``part_text`` holds a lone surrogate,
            which no stored field can.
    """
    _check_owner(owner_id)
    payload = canonical_core_json_form({"kind": "rejected_item", "owner": owner_id, "text": part_text})
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hex=digest[:32]))


def rejection_points(owner_id: str, field: str) -> Dict[str, PointRef]:
    """Maps each distinct point id of a field to the first part that produced it.

    Ordered by enumeration order; byte-identical items or parts under one owner
    collapse to one point.

    Args:
        owner_id: The owner node's ``nodes.id`` (see :func:`item_point_id`).
        field: The raw ``rejected_paths`` text.

    Returns:
        ``{point_id: PointRef}``, empty for an empty field.

    Raises:
        ValueError: If ``owner_id`` is not 64 lowercase hex characters.
    """
    _check_owner(owner_id)
    points: Dict[str, PointRef] = {}
    for item in enumerate_rejected_paths(field).items:
        for part in item.parts:
            pid = item_point_id(owner_id, part.text)
            if pid not in points:
                points[pid] = PointRef(ordinal=item.ordinal, part_index=part.index, text=part.text)
    return points


def expected_rejection_points(
    owner_fields: Mapping[str, str],
    memo: Optional[Dict[Tuple[str, str], Tuple[str, ...]]] = None,
) -> Dict[str, str]:
    """Derives the rejection index's expected contents from the graph's owner fields.

    The one expected-set derivation that ``status`` and ``reconcile`` both call, so
    the heal clears its own warning by construction.

    Args:
        owner_fields: ``{decision id: raw rejected_paths}``, as
            ``GraphStore.get_rejection_index_owner_fields`` returns it.
        memo: An optional per-command cache keyed on ``(owner_id, field)``. A
            caller that reads the graph twice passes the same dict to both calls,
            so an owner whose raw string is unchanged is enumerated once.

    Returns:
        ``{point_id: owner_id}`` over every owner's :func:`rejection_points`.

    Raises:
        ValueError: If an owner id is not 64 lowercase hex characters (a store
            fault: the store hands over node ids).
    """
    expected: Dict[str, str] = {}
    for owner_id, field in owner_fields.items():
        key = (owner_id, field)
        ids = memo.get(key) if memo is not None else None
        if ids is None:
            ids = tuple(rejection_points(owner_id, field))
            if memo is not None:
                memo[key] = ids
        for pid in ids:
            expected[pid] = owner_id
    return expected


class RejectionIndexDiff(NamedTuple):
    """The rejection index measured against its expected contents.

    A point's id is its identity: a listed id that is expected is present,
    whatever owner its payload names.

    Attributes:
        missing: Expected point id -> owner, for every expected id not listed.
        stale: Listed ids not expected whose payload owner is a graph decision
            (its field no longer states the item).
        orphaned: Listed ids not expected whose payload owner is not a graph
            decision, or is unnamed.
    """

    missing: Dict[str, str]
    stale: FrozenSet[str]
    orphaned: FrozenSet[str]

    @property
    def missing_owners(self) -> FrozenSet[str]:
        """The distinct owners of the missing items."""
        return frozenset(self.missing.values())


def diff_rejection_index(
    expected: Mapping[str, str],
    listing: Mapping[str, Optional[str]],
    *,
    owners: AbstractSet[str],
) -> RejectionIndexDiff:
    """Diffs a listing of the rejection index against its expected contents.

    Args:
        expected: ``{point_id: owner_id}`` from :func:`expected_rejection_points`.
        listing: ``{point_id: owner_id or None}`` as
            ``RejectionIndex.list_point_owners`` returns it.
        owners: Every graph decision's id (the owner-fields read's keys).

    Returns:
        The missing, stale and orphaned split.
    """
    missing = {pid: owner for pid, owner in expected.items() if pid not in listing}
    stale = set()
    orphaned = set()
    for pid, owner in listing.items():
        if pid in expected:
            continue
        if owner is not None and owner in owners:
            stale.add(pid)
        else:
            orphaned.add(pid)
    return RejectionIndexDiff(missing=missing, stale=frozenset(stale), orphaned=frozenset(orphaned))
