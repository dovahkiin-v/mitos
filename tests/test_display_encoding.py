"""R6 crash-safety + display-JSON validity matrix for ``mitos.display``.

Phase 1a builds three display primitives before any emitter site is flipped
(that is Phase 1b). These tests pin their behavioral contracts:

* :func:`dumps_display` emits **valid JSON at either ``ensure_ascii`` value** —
  raw glyphs when False, ``\\uXXXX`` escapes when True — and both round-trip.
* :func:`resolve_display_ensure_ascii` returns False only for a genuine UTF-8
  stream; True for ascii / None-encoding / unresolvable-encoding streams.
* :func:`apply_stdout_text_safety` makes raw-text ``print()`` crash-safe on a
  non-UTF-8 stdout, is byte-transparent on UTF-8, and is a silent no-op on a
  stream lacking ``reconfigure``.

Pure stdlib unit tests — no graph fixture, no network, no keys. Assertions are
via ``json.loads`` round-trip and glyph presence, never hardcoded escape bytes
(mirrors ``test_identity.py``'s "assert via relation" discipline).
"""

import io
import json

import pytest

from mitos.display import (
    apply_stdout_text_safety,
    dumps_display,
    resolve_display_ensure_ascii,
)

# A §-dense, Lithuanian-bearing, em-dash-bearing payload — the real content shape
# mitos holds. The exact glyphs that today get escaped into \uXXXX noise.
SAMPLE = {
    "axiom": "Naudoti SQLite — ne PostgreSQL",
    "scope": "§4 kabutė",
    "note": "em—dash and § and ąčęėįšųūž",
}


def _utf8_stream() -> io.TextIOWrapper:
    """Returns a fresh UTF-8 TextIOWrapper over a byte buffer (has reconfigure)."""
    return io.TextIOWrapper(io.BytesIO(), encoding="utf-8", newline="")


def _ascii_stream() -> io.TextIOWrapper:
    """Returns a fresh ascii TextIOWrapper over a byte buffer (has reconfigure)."""
    return io.TextIOWrapper(io.BytesIO(), encoding="ascii", newline="")


# --- dumps_display: valid JSON at either ensure_ascii value -------------------

def test_dumps_display_raw_glyphs_when_not_ascii():
    """ensure_ascii=False emits raw glyphs AND round-trips back to the object."""
    out = dumps_display(SAMPLE, ensure_ascii=False)
    assert "§" in out
    assert "—" in out
    assert "kabutė" in out
    assert "\\u" not in out  # no escape noise
    assert json.loads(out) == SAMPLE


def test_dumps_display_escaped_when_ascii():
    """ensure_ascii=True emits pure-ASCII \\uXXXX that STILL round-trips."""
    out = dumps_display(SAMPLE, ensure_ascii=True)
    assert out.isascii()
    assert "§" not in out
    assert "\\u" in out  # escapes present
    assert json.loads(out) == SAMPLE


def test_dumps_display_indent_default_is_pretty():
    """Default indent=2 pretty-prints (multi-line)."""
    out = dumps_display(SAMPLE, ensure_ascii=False)
    assert "\n" in out


def test_dumps_display_indent_none_is_single_line():
    """indent=None yields single-line JSON (the MCP error-return shape, 1b)."""
    out = dumps_display(SAMPLE, ensure_ascii=False, indent=None)
    assert "\n" not in out
    assert json.loads(out) == SAMPLE


# --- resolve_display_ensure_ascii: only False for a real UTF-8 stream ---------

def test_resolve_false_for_utf8_stream():
    """A genuine UTF-8 stream resolves to False (raw glyphs safe)."""
    assert resolve_display_ensure_ascii(_utf8_stream()) is False


def test_resolve_true_for_ascii_stream():
    """An ascii stream resolves to True (escape — would otherwise crash)."""
    assert resolve_display_ensure_ascii(_ascii_stream()) is True


@pytest.mark.parametrize("spelling", ["utf-8", "UTF8", "utf_8", "UTF-8"])
def test_resolve_normalizes_utf8_spellings(spelling):
    """Spelling variants of UTF-8 all normalize to False via codecs.lookup."""
    stream = io.TextIOWrapper(io.BytesIO(), encoding=spelling, newline="")
    assert resolve_display_ensure_ascii(stream) is False


def test_resolve_true_for_none_encoding():
    """A None .encoding (e.g. a bare object) resolves to True."""

    class _NoEncoding:
        encoding = None

    assert resolve_display_ensure_ascii(_NoEncoding()) is True


def test_resolve_true_for_missing_encoding_attr():
    """A stream with no encoding attribute at all resolves to True."""
    assert resolve_display_ensure_ascii(object()) is True


def test_resolve_true_for_unresolvable_encoding():
    """A bogus encoding name (codecs.lookup raises LookupError) resolves True."""

    class _BogusEncoding:
        encoding = "not-a-real-codec-xyz"

    assert resolve_display_ensure_ascii(_BogusEncoding()) is True


def test_resolve_true_for_non_utf8_encoding():
    """A real but non-UTF-8 encoding (latin-1) resolves to True."""

    class _Latin1:
        encoding = "latin-1"

    assert resolve_display_ensure_ascii(_Latin1()) is True


# --- apply_stdout_text_safety: crash-safe, transparent, no-op-safe ------------

def test_text_safety_makes_ascii_print_not_raise():
    """After hardening, printing § to an ascii stream does not raise; emits bytes."""
    stream = _ascii_stream()
    apply_stdout_text_safety(stream)
    print("§4 kabutė —", file=stream)  # would raise UnicodeEncodeError unhardened
    stream.flush()
    raw = stream.buffer.getvalue()
    assert raw  # something was written
    assert raw.isascii()  # backslashreplace keeps it pure-ASCII


def test_text_safety_utf8_emits_raw_glyphs_unchanged():
    """On a UTF-8 stream the handler is inert: raw glyphs emitted byte-identically."""
    plain = _utf8_stream()
    print("§4 kabutė —", file=plain)
    plain.flush()
    expected = plain.buffer.getvalue()

    hardened = _utf8_stream()
    apply_stdout_text_safety(hardened)
    print("§4 kabutė —", file=hardened)
    hardened.flush()
    assert hardened.buffer.getvalue() == expected
    assert "§".encode("utf-8") in expected


def test_text_safety_noop_on_stream_without_reconfigure():
    """A StringIO (no reconfigure) is a silent no-op — no raise."""
    apply_stdout_text_safety(io.StringIO())  # must not raise


def test_text_safety_idempotent():
    """Calling twice is safe and still crash-safe."""
    stream = _ascii_stream()
    apply_stdout_text_safety(stream)
    apply_stdout_text_safety(stream)
    print("§§", file=stream)
    stream.flush()
    assert stream.buffer.getvalue()


def test_text_safety_leaves_encoding_unchanged():
    """Only the error handler changes — encoding/newline stay put (piping safe)."""
    stream = _ascii_stream()
    apply_stdout_text_safety(stream)
    assert stream.encoding.lower().replace("-", "") == "ascii"
    assert stream.errors == "backslashreplace"


# --- the two crash-safe paths never collide (Decision 3) ----------------------

def test_json_path_inert_to_text_handler_on_non_utf8():
    """On a non-UTF-8 stdout the JSON path resolves to ascii escapes — pure ASCII,
    so the text error-handler has nothing to replace. The two paths don't collide."""
    stream = _ascii_stream()
    apply_stdout_text_safety(stream)
    ensure_ascii = resolve_display_ensure_ascii(stream)
    assert ensure_ascii is True
    payload = dumps_display(SAMPLE, ensure_ascii=ensure_ascii)
    assert payload.isascii()
    print(payload, file=stream)  # no unencodable bytes for the handler to touch
    stream.flush()
    assert json.loads(stream.buffer.getvalue().decode("ascii")) == SAMPLE
