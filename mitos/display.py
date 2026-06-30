"""Display-output primitives for the CLI and MCP surfaces.

Tier-1 leaf module (stdlib only). It carries three small, load-bearing pieces
of display hygiene so they live in exactly one place and CLI⇄MCP drift becomes
structurally impossible:

* :func:`dumps_display` — the single shared display-JSON serializer. Both
  ``cli.py`` and ``mcp_server.py`` route their display ``json.dumps`` through it
  (wired in Phase 1b). It takes ``ensure_ascii`` as a *parameter* so it is
  surface-neutral.
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
from typing import Any, Optional, TextIO


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
