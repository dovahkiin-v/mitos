"""Display-output primitives for the CLI and MCP surfaces.

Tier-1 leaf module (stdlib only). It carries three small, load-bearing pieces
of display hygiene so they live in exactly one place and CLI⇄MCP drift becomes
structurally impossible:

* :func:`dumps_display` — the single shared display-JSON serializer (the
  *encoding* seam). Both ``cli.py`` and ``mcp_server.py`` route their display
  ``json.dumps`` through it (wired in Phase 1b). It takes ``ensure_ascii`` as a
  *parameter* so it is surface-neutral.
* :func:`letter_payload` — the single shared Letter-payload shaper (the *shape*
  seam). Both ``cli.py`` and ``mcp_server.py`` route their per-decision
  Letter-payload assembly through it (wired in Phase 2a) so the key set and the
  M5 ``rejected_paths``-unless-``brief`` rule live in exactly one place. It is
  the *sibling* of :func:`dumps_display`, never an extension: shape and encoding
  are distinct seams and neither calls the other.
* :func:`resolve_display_ensure_ascii` — **CLI-internal.** Decides
  ``ensure_ascii``'s value by sniffing the live stdout encoding.
* :func:`apply_stdout_text_safety` — **CLI-internal.** Makes raw-text
  ``print()``s crash-safe on a non-UTF-8 stdout.

The P7 bulkhead is a *call-graph* rule, not a file rule: :func:`dumps_display`
must never call the two CLI-internal helpers, and ``mcp_server.py`` imports only
:func:`dumps_display`. The MCP transport has no terminal stdout to sniff, so the
stdout-encoding adaptation stays strictly on the CLI side.

This module is **not** the hash-input serializer. ``identity.py`` is fenced
(MI-7): its ``json.dumps`` has its own deliberate ``sort_keys`` + compact
``separators`` for byte-stable hashing and must never route through here.
"""

import codecs
import json
from typing import Any, Dict, List, Mapping, Optional, TextIO


# The sane upper bound on the ranked-recall top-k (`--limit` / the MCP `limit`
# arg). `--limit` is the agent's context-budget lever (P15): it SETS the
# top-k — raising it past the working default of 5 or trimming below it — so it
# must NOT be clamped to the default (that would make any `--limit` above 5 a
# silent no-op). It is bounded only by this ceiling — well above the default
# working range, below any context-bomb. Lives here, in the one leaf both `cli.py`
# and `mcp_server.py` already import, so the literal exists in exactly one place
# and CLI⇄MCP can never disagree on it.
RANKED_LIMIT_CEILING = 50


def clamp_limit(limit: Optional[int]) -> int:
    """Resolves a caller's ranked-recall ``limit`` to a sane top-k, calmly.

    ``None`` (the flag was omitted) resolves to the default working top-k of 5.
    Any explicit value is clamped to ``[1, RANKED_LIMIT_CEILING]`` — silently and
    calmly (P9), never an error wall: a request below 1 clamps up to 1, a request
    above the ceiling clamps down to it. The resolved value SETS the top-k passed
    to ``vector_store.query(limit=…)`` — it is not a ``min(default, N)`` truncation,
    so ``limit=20`` genuinely deepens recall past the default.

    Args:
        limit: The caller's requested top-k, or ``None`` to take the default.

    Returns:
        The clamped top-k in ``[1, RANKED_LIMIT_CEILING]``.
    """
    if limit is None:
        return 5
    return max(1, min(limit, RANKED_LIMIT_CEILING))


def blackout_note(retired_handles: List[Mapping[str, Any]]) -> str:
    """Builds the all-superseded blackout recovery note (the P3 vector text).

    When ranked recall retrieved precedents but every one was superseded-filtered,
    the surface would otherwise read as a true semantic *miss* — a false-novelty
    signal that costs the agent an expensive re-derivation of a contradiction the
    graveyard already settled. This note turns that dead-end into a vector: it names
    the retired handles (and their live successors, when known) and points at the
    command that reads the retired history. Calm, one short block (P9).

    Args:
        retired_handles: The filtered-out handles, each a dict with ``slug`` and
            optionally ``superseded_by`` (the live successor slugs).

    Returns:
        The blackout note string.
    """
    n = len(retired_handles)
    parts: List[str] = []
    for h in retired_handles:
        successors = h.get("superseded_by")
        if successors:
            parts.append(f"{h['slug']} (→ {', '.join(successors)})")
        else:
            parts.append(str(h["slug"]))
    noun, verb = ("match", "is") if n == 1 else ("matches", "are")
    return (
        f"All {n} nearest {noun} {verb} superseded — no active precedent on "
        f"this claim, but it was settled before. Retired: {'; '.join(parts)}. "
        f"Read the retired history with: mitos list --state all "
        f"(or list_decisions(state=\"all\"))."
    )


def order_scope_counts(counts: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """Re-orders the scope→counts map by liveness, busiest domain first (the sort seam).

    ``GraphStore.get_scope_counts`` returns its tag→counts map *alphabetically*
    (deterministic, but presentation-neutral by design — 3a does not own display
    order). This is the single place that imposes the discovery surface's order:
    total live-node count (``active_decisions + parked_open_questions``)
    **descending**, ties broken **alphabetically** by scope tag. The busiest
    domains read first — the map an agent scans before recording or recalling.

    Both surfaces (the ``scopes`` CLI verb and the ``list_scopes`` MCP tool) call
    this on the same ``get_scope_counts`` result, so the rendered table and the
    ``--json`` / MCP map share one ordered dict — CLI⇄MCP order parity is
    structural, not coincidental. A Python ``dict`` preserves insertion order and
    ``json.dumps`` honors it, so ordering once here orders every downstream render.

    This is a tag→counts *aggregate*, not a decision-read payload: there is no node
    ``id`` to stamp, so the "every decision-read surface stamps modifiers" rule does
    **not** apply (no modifier seam). It only re-keys an existing dict into a new
    insertion order — it never transforms keys or values.

    Args:
        counts: The ``{scope: {"active_decisions": int, "parked_open_questions":
            int}}`` map from ``get_scope_counts`` (alphabetical, casefolded keys).

    Returns:
        The same map, re-inserted in total-live-count-descending, ties-alphabetical
        order. Empty input returns ``{}``.
    """
    return dict(
        sorted(
            counts.items(),
            key=lambda kv: (
                -(kv[1]["active_decisions"] + kv[1]["parked_open_questions"]),
                kv[0],
            ),
        )
    )


def letter_payload(
    node: Mapping[str, Any], *, brief: bool, extras: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Shapes the Letter-complete decision-read core, shared CLI⇄MCP (the shape seam).

    The single place the per-decision Letter payload key set lives: ``slug``,
    ``axiom`` (from the node's ``core_axiom``), ``scope``, any caller ``extras``,
    then ``rejected_paths`` *unless* ``brief``. ``extras`` land in a deterministic
    slot — between ``scope`` and ``rejected_paths`` — reproducing each caller's
    shipped key order byte-identically (the verb-envelope fields ``score`` /
    ``state`` / ``depth_mode`` already occupy that slot at every routed site).

    Returns an **un-stamped** core: modifier-stamping
    (``superseded_by``/``amended_by``/…) is the caller's job via the
    ``GraphStore.get_modifiers(...)`` seam — never folded in here. This keeps the
    helper a pure dict→dict Tier-1 leaf (no ``store`` coupling) and preserves the
    callers' batch (``get_modifiers_map``) vs per-node stamping paths. ``brief``
    governs only the ``rejected_paths`` key inside this helper; because stamping
    happens in the caller *after* this returns, ``brief`` can never drop a
    modifier stamp.

    This is display-only and shapes the *decision* dict; it is the sibling of
    :func:`dumps_display` (which *encodes* any display dict) and never calls it.
    NOT the hash-input serializer — ``identity.py`` is fenced (MI-7); the Letter
    payload must never reach the hash/persistence path.

    Args:
        node: A decision node dict; reads ``slug``, ``core_axiom``, ``scope`` and
            (unless ``brief``) ``rejected_paths``.
        brief: When True, omit ``rejected_paths`` (the M4 opt-out) and nothing
            else; when False, include it (the M5 anti-knowledge fence).
        extras: Ordered verb-envelope fields to interleave between ``scope`` and
            ``rejected_paths``, in the caller's order. ``None`` adds nothing.

    Returns:
        The un-stamped Letter-payload dict.
    """
    payload: Dict[str, Any] = {
        "slug": node["slug"],
        "axiom": node["core_axiom"],
        "scope": node["scope"],
    }
    if extras:
        payload.update(extras)
    if not brief:
        payload["rejected_paths"] = node["rejected_paths"]
    return payload


def dumps_display(obj: Any, *, ensure_ascii: bool, indent: Optional[int] = 2) -> str:
    """Serializes a display payload to JSON — the single CLI⇄MCP display seam.

    A thin passthrough over :func:`json.dumps` exposing only ``ensure_ascii``
    and ``indent``. The caller owns the dict shape; this serializer owns only
    the two display knobs. It is display-only (terminal text / JSON-RPC string):
    no persistence, no hashing.

    NOT for hash/persistence input — ``identity.py`` is fenced (MI-7). Do not
    route the hash-input serializer through this function.

    Args:
        obj: A JSON-native object (the caller already passes display-ready dicts).
        ensure_ascii: When True, non-ASCII characters are escaped to ``\\uXXXX``
            (valid JSON, never a crash on a non-UTF-8 stdout); when False, raw
            UTF-8 glyphs are emitted (the MCP-mode contract).
        indent: Pretty-print indent; ``None`` for single-line output.

    Returns:
        The serialized JSON string.
    """
    return json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent)


def resolve_display_ensure_ascii(stream: TextIO) -> bool:
    """Decides ``ensure_ascii``'s value for a stdout stream (CLI-internal).

    Returns ``False`` (emit raw glyphs) only when the stream genuinely encodes
    UTF-8 — so raw ``§``/Lithuanian text is safe. For any other case (a ``None``
    or absent encoding, an unresolvable encoding, or any non-UTF-8 encoding) it
    returns ``True`` so the JSON path falls back to ``\\uXXXX`` escapes, which
    stay valid JSON instead of crashing with ``UnicodeEncodeError``.

    Comparison is on the codec's normalized name (``codecs.lookup(enc).name``),
    so spelling variants like ``"UTF8"`` / ``"utf_8"`` all resolve correctly.

    Args:
        stream: The stdout stream to inspect.

    Returns:
        True to escape non-ASCII (safe fallback); False only for a real UTF-8
        stream.
    """
    enc = getattr(stream, "encoding", None)
    if not enc:
        return True
    try:
        return codecs.lookup(enc).name != "utf-8"
    except (LookupError, TypeError):
        return True


def apply_stdout_text_safety(stream: TextIO) -> None:
    """Makes raw-text ``print()``s crash-safe on a non-UTF-8 stdout (CLI-internal).

    Sets the stream's *error handler* to ``backslashreplace`` so an unencodable
    glyph becomes a readable escape instead of raising ``UnicodeEncodeError``.
    Only the error handler is changed — ``encoding``, ``line_buffering`` and
    ``newline`` are left untouched, so buffering, flushing and piping behave
    exactly as before. On a UTF-8 stdout the handler never fires (encodable
    content is emitted byte-identically), so output stays unchanged.

    Best-effort and fail-silent: a no-op when the stream lacks ``reconfigure``
    (a captured / ``StringIO`` stdout) or when reconfiguring raises on a
    detached stream. Safe to call more than once.

    Args:
        stream: The stdout stream to harden (never ``sys.stderr``).

    Returns:
        None.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(errors="backslashreplace")
    except (ValueError, OSError):
        return
