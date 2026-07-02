"""Slug-free canonical-core node identity for Mitos (M2, V1-D2, §11/§12).

This is the gravitational center of the graph: the function that turns a
decision's ``{kind, axiom, mechanism_refs}`` (or an open_question's
``{kind, topic, questions_raised}``) into the SHA-256 hex that *is* the node.
The slug is **not** an input — two captures of the same canonical core converge
to one id regardless of slug, and a rename is an in-place commentary edit that
leaves the id unchanged (Q5 / ``slug-removed-from-canonical-core-hash``).

This module also exports :func:`embedding_text` — the **second** projection of
the same immutable canonical core. Where :func:`compute_node_id` renders the
core to the SHA-256 that *names* the node, :func:`embedding_text` renders it to
the per-kind string V2's embedding pipeline embeds against — the *meaning* a
semantic search reads. Both are pure functions of the canonical core through the
same normalization rules, so the node's name and its meaning can never drift
apart (M8): re-deriving the embedding string for an unchanged node yields the
identical bytes, and V2's content-hash cache returns the cached vector for free.

This module is a **pure stdlib Tier-1 leaf**: it imports only ``hashlib``,
``json``, ``re``, ``string`` and ``unicodedata``, and nothing from ``mitos/``.
The eventual edges are inbound only — Phase 5a points ``commit_parsed_entry``
at :func:`compute_node_id`; Phase 8a moves ``sync.py``/``importer.py`` onto it
and retires the prototype ``store.compute_hash``. 3a wires none of them: it
authors the unit and golden-proves it, leaving the prototype hash live.

Two asymmetries are the spine of the catalog (§12) — collapsing either silently
corrupts node identity, so they are two distinct functions, never one shared
helper:

* **Case axis.** :func:`canonical_core_string_norm` (axiom, topic, OQ question
  items) is case-**preserved** — case carries meaning in natural language
  ("Use SQLite" != "use sqlite"). :func:`mechanism_canonical_norm` (mechanism
  items) **casefolds** — short LLM-authored identifier tokens churn in casing,
  so folding keeps mechanism identity stable ("SQLite" == "sqlite").
* **Order axis.** ``mechanism_refs`` is an **unordered tag set**
  (``sorted(set(...))``) — reordering a ref never changes identity.
  ``questions_raised`` is an **ordered prose sequence** — order-preserving
  dedup, **never** sorted: authored order is part of the inquiry's meaning, so
  reordering mints a new node (M1).

**MI-7 (the trap):** the hash input is
``json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",",":")).encode("utf-8")``.
All four ``json.dumps`` arguments are load-bearing. On a first run you may hit
``TypeError: Unicode-objects must be encoded before hashing`` and be tempted to
"fix" it by flipping ``ensure_ascii=True`` — **do not**. That produces
escaped-ASCII JSON, type-checks, passes ASCII tests, and silently destroys
byte-stability across runtimes that escape Unicode differently. The correct fix
is the ``.encode("utf-8")``. The Lithuanian (D5/Q5) and CJK (D6) golden rows are
the canary.

**Pinning:** the §11 golden digests are frozen against **Python 3.13.5 /
unicodedata UCD 15.1.0**. NFC normalization depends on the UCD version; a bump
is a deliberate one-time golden regeneration recorded as an amendment, never a
silent regeneration (see ``tests/test_identity.py``).
"""

import hashlib
import json
import re
import string
import unicodedata
from typing import Dict, List, Mapping, Optional

# The slug is the permanent citation handle (M3 active-slug uniqueness; folded into
# the decision's identity, V1-D2), so an over-length slug is never silently truncated
# — it is rejected at every write path. Home here (the identity leaf) so the parser,
# the store's commit fence, and the record write path share one source of truth
# without a dependency-tier inversion. The MCP `record_decision` slug docstring
# carries this number as a literal — update it (mcp_server.py) if this changes.
SLUG_MAX_LEN = 100

# Maximal run of ASCII whitespace OR ASCII punctuation -> a single hyphen. The
# ``re.ASCII`` flag restricts ``\s`` to ASCII whitespace ([ \t\n\r\f\v]) so the
# fold is ASCII-only by design (V1-D3): a non-ASCII character (CJK, an accented
# letter, Unicode punctuation, a no-break space) passes through NFC+casefold
# untouched and never collides with a different non-ASCII token.
# ``string.punctuation`` is the 32 ASCII marks; ``re.escape`` makes them literal.
_MECHANISM_FOLD_RE = re.compile(r"[\s" + re.escape(string.punctuation) + r"]+", re.ASCII)


def canonical_core_string_norm(s: str) -> str:
    """Normalizes a case-preserved canonical-core string field.

    Applied to the ``axiom``, the ``topic``, and each ``questions_raised`` item
    — the fields where case and internal whitespace are content. NFC first
    (so a composed and a decomposed Lithuanian "ė" hash identically), then strip
    leading/trailing whitespace. **No case fold and no internal-whitespace
    collapse** — internal whitespace is part of the prose.

    Args:
        s: The raw field text.

    Returns:
        The NFC-normalized, end-stripped string with case preserved.
    """
    return unicodedata.normalize("NFC", s).strip()


def mechanism_canonical_norm(s: str) -> str:
    """Normalizes a single mechanism token to its canonical, casefolded form.

    Mechanism refs are short LLM-authored identifier tokens that churn in casing
    and punctuation, so the fold is aggressive: NFC, then ``casefold``, then
    collapse every maximal run of ASCII punctuation or ASCII whitespace to a
    single hyphen and strip leading/trailing hyphens. So ``"SQLite!"`` and
    ``"sqlite"`` both fold to ``"sqlite"``; ``"str_casefold"``,
    ``"str-casefold"`` and ``"Str Casefold"`` all fold to ``"str-casefold"``;
    ``"node-scopes"`` stays ``"node-scopes"``. Non-ASCII characters survive
    NFC+casefold untouched (the fold is ASCII-only — V1-D3).

    This exact byte form is reused as V1b's ``mechanisms.canonical_name`` PK, so
    it is pinned cross-vision.

    Args:
        s: The raw mechanism token.

    Returns:
        The casefolded, punctuation-folded canonical token.
    """
    folded = unicodedata.normalize("NFC", s).casefold()
    return _MECHANISM_FOLD_RE.sub("-", folded).strip("-")


def dedup_preserve_order(items: List[str]) -> List[str]:
    """Deduplicates a list, keeping the first occurrence of each item, no sort.

    ``dict.fromkeys`` preserves insertion order (guaranteed since Python 3.7),
    so this is an order-preserving dedup. Used by
    :func:`questions_raised_list_norm` where authored order is identity (M1).

    Args:
        items: The items to dedup.

    Returns:
        A new list with duplicates removed, original order preserved.
    """
    return list(dict.fromkeys(items))


def mechanism_refs_list_norm(items: List[str]) -> List[str]:
    """Normalizes a mechanism-refs list into a sorted, deduped tag set.

    Filters empty/whitespace-only raw items (``if m.strip()`` on the *raw* item,
    before folding), maps each through :func:`mechanism_canonical_norm`, then
    set-dedups and sorts by code point. ``mechanism_refs`` is an **unordered
    set** — reordering or duplicating a ref never changes the node id. Returns a
    ``list`` (a JSON array; never a tuple — M2).

    Args:
        items: The raw mechanism tokens.

    Returns:
        The folded, deduped, code-point-sorted mechanism list.
    """
    return sorted({mechanism_canonical_norm(m) for m in items if m.strip()})


def questions_raised_list_norm(items: List[str]) -> List[str]:
    """Normalizes an open-question ``questions_raised`` list, order preserved.

    Filters empty/whitespace-only raw items (``if q.strip()`` on the *raw* item),
    maps each through :func:`canonical_core_string_norm` (case-preserved), then
    applies an **order-preserving dedup — never a sort**. Authored order is part
    of the inquiry's meaning, so reordering the questions mints a new node (M1);
    sorting would scramble the V2 embedding and wrongly converge two distinct
    inquiries. This is the order-axis counterpart to
    :func:`mechanism_refs_list_norm`. Returns a ``list`` (JSON array; never a
    tuple — M2).

    Args:
        items: The raw question strings.

    Returns:
        The normalized questions in authored order, duplicates removed.
    """
    return dedup_preserve_order(
        [canonical_core_string_norm(q) for q in items if q.strip()]
    )


def canonical_core_json_form(obj: Dict[str, object]) -> str:
    """Serializes a canonical-core dict to the pinned, byte-stable JSON form.

    The four ``json.dumps`` arguments are all load-bearing (MI-7):
    ``sort_keys=True`` makes the key order alphabetical regardless of dict
    construction order; ``ensure_ascii=False`` keeps non-ASCII characters as
    literal UTF-8 (the ``.encode("utf-8")`` in :func:`compute_node_id` is the
    correct way to feed the hasher — **never** flip this to ``True``);
    ``separators=(",",":")`` removes incidental whitespace.

    Args:
        obj: The canonical-core dict (decision or open_question shape).

    Returns:
        The compact, key-sorted JSON string (still ``str`` — caller encodes it).
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compute_node_id(
    *,
    kind: str,
    axiom: Optional[str] = None,
    mechanism_refs: Optional[List[str]] = None,
    topic: Optional[str] = None,
    questions_raised: Optional[List[str]] = None,
) -> str:
    """Computes the slug-free canonical-core SHA-256 id for a node (M2, V1-D2).

    Identity is the *content*, not the slug: a ``decision`` hashes over
    ``{kind, axiom, mechanism_refs}`` and an ``open_question`` over
    ``{kind, topic, questions_raised}``. ``kind`` lives inside the hashed object
    so a decision and an open_question with text-identical canonical strings can
    never collide. The id is the value that flows into ``nodes.id`` (TEXT PK)
    once Phase 5a wires this in.

    **Keyword-only by design** (the leading ``*``): the prototype
    ``store.compute_hash(kind, slug, ...)`` was positional with the slug second,
    so a stray old-style positional call would silently mis-hash. Making this
    signature keyword-only and slug-less means such a call fails loudly at the
    boundary. It takes **raw fields, not a ``ParsedEntry``**, so ``identity.py``
    stays a pure leaf and V2 (``point_id`` reuse) / V6 (re-hashing), which have
    no ``ParsedEntry``, share this one identity path.

    Args:
        kind: ``"decision"`` or ``"open_question"``. Any other value raises.
        axiom: The decision axiom (required for ``decision``).
        mechanism_refs: The decision's mechanism tokens (optional → ``[]``).
        topic: The open_question topic (required for ``open_question``).
        questions_raised: The open_question's questions (optional → ``[]``).

    Returns:
        The 64-character lowercase SHA-256 hex digest.

    Raises:
        ValueError: If ``kind`` is neither ``"decision"`` nor ``"open_question"``.
    """
    if kind == "decision":
        obj: Dict[str, object] = {
            "kind": "decision",
            "axiom": canonical_core_string_norm(axiom or ""),
            "mechanism_refs": mechanism_refs_list_norm(mechanism_refs or []),
        }
    elif kind == "open_question":
        obj = {
            "kind": "open_question",
            "topic": canonical_core_string_norm(topic or ""),
            "questions_raised": questions_raised_list_norm(questions_raised or []),
        }
    else:
        raise ValueError(
            f"compute_node_id: unknown kind {kind!r}; "
            "expected 'decision' or 'open_question'"
        )

    serialized = canonical_core_json_form(obj)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def embedding_text(node: Mapping[str, object]) -> str:
    """Builds the single embedding-input string for a node (§5.1 C2, M8).

    The companion to :func:`compute_node_id`: the *second* projection of the
    immutable canonical core. The hash names the node; this string is what V2's
    embedding pipeline embeds and caches against. Both run the same core through
    the same 3a normalization rules (:func:`canonical_core_string_norm`,
    :func:`questions_raised_list_norm`), so the embedding string can never drift
    from the hashed identity — re-deriving it for an unchanged node yields the
    identical bytes and V2's content-hash cache returns the cached vector for
    free (P15).

    Per-kind shape — the pinned cross-vision cache-key byte form:

    * **decision** → the normalized ``axiom`` and **nothing else**.
      ``mechanism_refs`` is deliberately excluded, and the reason is
      **recall-cleanliness, NOT cache-stability**: mechanism refs *are*
      immutable canonical core (so they would not churn the cache), but they are
      short casefolded identifier tokens whose folded form dilutes a
      natural-language recall vector. Do **not** "fix" this by appending them
      [ADR ``embedding-text-excludes-mechanism-refs``].
    * **open_question** → ``topic`` + a blank line + the ``questions_raised``
      joined by single newlines, **in authored order** (order-preserving dedup,
      **never sorted**). Authored order is identity-significant (§11/§12), and
      the list joined here is byte-identical to the one :func:`compute_node_id`
      hashes — both call :func:`questions_raised_list_norm`
      [ADR ``questions-raised-order-significant-prose-not-sorted-set``].

    ``node`` is a native-typed mapping (in practice a ``dict`` from a store read
    method) carrying the canonical-core fields under reader-facing keys in native
    Python types: ``axiom``/``topic`` as ``str``, ``questions_raised`` as an
    already-JSON-decoded ``list[str]``. JSON decoding of the stored
    ``questions_raised_json`` column is the store read layer's job, not this pure
    leaf's — this function never sees a ``ParsedEntry`` or a ``*_json`` string.

    A degenerate open_question with no surviving questions renders as
    ``topic + "\\n\\n"`` (a trailing blank line) — deterministic and total, with
    no special-case branch (4b validation makes it unreachable in practice).

    Args:
        node: The canonical-core mapping. ``kind`` is ``"decision"`` or
            ``"open_question"``; a decision carries ``axiom`` (``mechanism_refs``
            may be present but is ignored); an open_question carries ``topic`` and
            ``questions_raised``.

    Returns:
        The embedding-input string for the node.

    Raises:
        ValueError: If ``node["kind"]`` is neither ``"decision"`` nor
            ``"open_question"`` (mirrors :func:`compute_node_id`'s guard).
    """
    kind = node.get("kind")
    if kind == "decision":
        return canonical_core_string_norm(node.get("axiom") or "")
    if kind == "open_question":
        topic = canonical_core_string_norm(node.get("topic") or "")
        questions = questions_raised_list_norm(node.get("questions_raised") or [])
        return topic + "\n\n" + "\n".join(questions)
    raise ValueError(
        f"embedding_text: unknown kind {kind!r}; "
        "expected 'decision' or 'open_question'"
    )
