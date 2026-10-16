"""The ``rejected_paths`` enumeration leaf: trace table, property row, guards, tripwire.

The rules in ``mitos/rejections.py`` are part of the rejection index's identity, so
they are pinned here before anything embeds through them. The discipline matches
``tests/test_identity.py``:

* **Expected identity texts are hand-traced** from the grammar (the phase plan's §5),
  never captured from what the leaf emits. A disagreement between a row and the leaf
  is investigated as a possible transcription error in the row first, then as a rule
  defect; the leaf is never loosened to fit a row.
* **Expected point ids are computed out of band.** T1 and T12 carry literals computed
  outside Python (``printf '%s' '<json>' | sha256sum``); every other row computes its
  expected ids with :func:`_oob_point_id`, which shares no code with the leaf.
* The out-of-band literals inherit ``identity.py``'s UCD 15.1.0 pin (the T12 row NFCs
  a decomposed field), so a UCD bump reds both suites deliberately.

Every ``U+2028``/``U+2029``/NBSP/combining character in this file is written as an
escape, never as a literal, so it cannot silently become a plain space.
"""

import ast
import hashlib
import json
import random
import subprocess
import sys
import unicodedata
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

from mitos import rejections
from mitos.identity import canonical_core_string_norm
from mitos.rejections import (
    BOUND_DERIVED_FOR_MODEL,
    BOUND_MODEL_TOKEN_LIMIT,
    DOCUMENT_PREFIX_BYTES,
    ITEM_PART_BOUND_BYTES,
    SHAPES,
    Enumeration,
    PointRef,
    enumerate_rejected_paths,
    item_point_id,
    rejection_points,
)

# Harbor's `harbor-storage-is-sqlite` node id, as a literal: the out-of-band ids below
# were computed against it, so it must never be derived from the oracle at test time.
OWNER = "031f7eb70553044daf35f5ffcb0dd300bc3f1f7f1f751e61e007a37dabc4050f"

T12_COMPOSED = (
    "Tik Lietuva — brangesn\u0117 infrastrukt\u016bra ir prastesnis "
    "v\u0117lavimas kit\u0173 ES \u0161ali\u0173 naudotojams."
)
T12_DECOMPOSED = T12_COMPOSED.replace("\u0117", "e\u0307")


def _oob_point_id(owner: str, text: str) -> str:
    """The point id, computed with no code shared with the leaf."""
    payload = json.dumps(
        {"kind": "rejected_item", "owner": owner, "text": text},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _nonspace(s: str) -> str:
    return "".join(ch for ch in s if not ch.isspace())


def _assert_contracts(field: str, enum: Enumeration) -> None:
    """The four §4.2 contracts plus determinism, for any field."""
    assert enum.shape in SHAPES
    if not field.strip():
        assert enum.shape == "empty" and enum.items == []
        return
    assert enum.items, "a non-empty field must give at least one item"
    assert enum.shape != "empty"
    # Coverage: contiguous spans that exactly partition the field.
    assert enum.items[0].start == 0
    assert enum.items[-1].end == len(field)
    for prev, nxt in zip(enum.items, enum.items[1:]):
        assert prev.end == nxt.start
    for k, item in enumerate(enum.items, start=1):
        assert item.ordinal == k
        assert item.start < item.end
        assert item.text == field[item.start:item.end]
        assert item.identity_text, "an item's identity text is never empty"
        assert canonical_core_string_norm(item.identity_text) == item.identity_text
        assert [p.index for p in item.parts] == list(range(len(item.parts)))
        for part in item.parts:
            assert part.text
            assert len(part.text.encode("utf-8", "surrogatepass")) <= ITEM_PART_BOUND_BYTES
            assert canonical_core_string_norm(part.text) == part.text
        assert _nonspace("".join(p.text for p in item.parts)) == _nonspace(item.identity_text)
        if len(item.identity_text.encode("utf-8", "surrogatepass")) <= ITEM_PART_BOUND_BYTES:
            assert [p.text for p in item.parts] == [item.identity_text]
    assert "".join(i.text for i in enum.items) == field
    assert enumerate_rejected_paths(field) == enum, "the enumeration must be deterministic"


# ---------------------------------------------------------------------------
# The trace table (constraint 15). Identity texts hand-traced from §5.
# ---------------------------------------------------------------------------

TRACE = [
    pytest.param("(1) A: REJECTED — x.\n(2) B: REJECTED — y.", "numbered",
                 ["A: REJECTED — x.", "B: REJECTED — y."], id="T1"),
    pytest.param("Two paths:\n(1) A — x\n(2) B — y", "numbered",
                 ["Two paths:\nA — x", "B — y"], id="T2"),
    pytest.param("(1) A - rejected: x. (2) B - rejected: y", "numbered",
                 ["A - rejected: x.", "B - rejected: y"], id="T3"),
    pytest.param("(1) A — unlike path (2) below. (2) B — y", "numbered",
                 ["A — unlike path (2) below.", "B — y"], id="T4"),
    pytest.param("(1) An O(1) check — x\n(2) B — y", "numbered",
                 ["An O(1) check — x", "B — y"], id="T5"),
    # T5 refuses the inner (1) by sequence alone. Here the notation IS the next
    # number, so only the boundary rule keeps `O(1)` from becoming marker 1.
    pytest.param("Use an O(1) map.\n(1) A — x\n(2) B — y", "numbered",
                 ["Use an O(1) map.\nA — x", "B — y"], id="T5b"),
    pytest.param("A: REJECTED — x. B: REJECTED — y.", "rejected_marker",
                 ["A: REJECTED — x.", "B: REJECTED — y."], id="T6"),
    pytest.param("X: REJECTED on two grounds. (1) a. (2) b.", "prose",
                 ["X: REJECTED on two grounds. (1) a. (2) b."], id="T7"),
    pytest.param("- A — x\n  more\n  - sub\n- B — y", "bulleted",
                 ["A — x\n  more\n  - sub", "B — y"], id="T8"),
    pytest.param("- A: (1) r1 (2) r2\n- B", "bulleted",
                 ["A: (1) r1 (2) r2", "B"], id="T9"),
    pytest.param("(1) A — x", "numbered", ["A — x"], id="T10"),
    pytest.param("(1) A\n(3) B", "numbered", ["A\n(3) B"], id="T11"),
    pytest.param(T12_COMPOSED, "prose", [T12_COMPOSED], id="T12"),
    pytest.param(T12_DECOMPOSED, "prose", [T12_COMPOSED], id="T12-nfd"),
    pytest.param("(1) A — x\n(2) A — x", "numbered", ["A — x", "A — x"], id="T13"),
    pytest.param("(1)\n(2) B — y", "numbered", ["B — y"], id="T14"),
    pytest.param("(\u0661) A\n(\u0662) B", "prose", ["(\u0661) A\n(\u0662) B"], id="T15"),
    pytest.param("1. A — x\n2. B — y", "numbered", ["A — x", "B — y"], id="T16"),
    pytest.param("", "empty", [], id="T21"),
    pytest.param("  \n ", "empty", [], id="T21-ws"),
    pytest.param("\xa0\u2028\t", "empty", [], id="T21-unicode-ws"),
    pytest.param("A — x\u2028B — y", "prose", ["A — x\u2028B — y"], id="T22"),
    pytest.param("A — x\u2029(2) B — y", "prose", ["A — x\u2029(2) B — y"], id="T22-u2029"),
    # A `\r` inside an item stays (strip only trims the ends) ...
    pytest.param("A\n\r\nB", "prose", ["A\n\r\nB"], id="T22-cr"),
    # ... and a `\r` before a line break falls to the item's end-strip.
    pytest.param("(1) A — x\r\n(2) B — y", "numbered", ["A — x", "B — y"], id="T22-crlf"),
    pytest.param("Two paths: (1) A: REJECTED — x. (2) B: REJECTED — y.", "numbered",
                 ["Two paths: A: REJECTED — x.", "B: REJECTED — y."], id="T23"),
    pytest.param("Two paths. (1) A: REJECTED — x. (2) B: REJECTED — y.", "numbered",
                 ["Two paths. A: REJECTED — x.", "B: REJECTED — y."], id="T24"),
    pytest.param("Two paths:\n(1)\n(2) B — y", "numbered", ["Two paths:\nB — y"], id="T25"),
    pytest.param("Option A: the fold: REJECTED — x.\nOption B: the cache: REJECTED — y.",
                 "rejected_marker",
                 ["Option A: the fold: REJECTED — x.", "Option B: the cache: REJECTED — y."],
                 id="T26"),
    # Live shape: the parser strips every stored line, so a nested list reaches the
    # leaf flattened to column 0 and its sub-bullets become sibling items.
    pytest.param("- A — x\n- sub\n- B — y", "bulleted", ["A — x", "sub", "B — y"], id="T27"),
    # `REJECTED` + U+0307 is not a marker (NFC would fuse it into the D), so only one
    # marker remains; a lone rejected_marker does not qualify.
    pytest.param("A: REJECTED\u0307 — x. B: REJECTED — y.", "prose",
                 ["A: REJECTE\u1e0a — x. B: REJECTED — y."], id="T29"),
    pytest.param("(1) A\n(2)", "numbered", ["A"], id="T30-trailing-empty"),
    pytest.param("(1)\n(2)", "prose", ["(1)\n(2)"], id="T31-only-markers"),
    pytest.param("Two:\n(1)\n(2)", "prose", ["Two:\n(1)\n(2)"], id="T32-preamble-only"),
    pytest.param("(1) A (2) B", "numbered", ["A (2) B"], id="T33-no-boundary"),
    pytest.param("(1) A. (2) B", "numbered", ["A.", "B"], id="T34-dot-boundary"),
    # A CJK stop is a boundary only with a space or tab after it, so (2) stays inside.
    pytest.param("(1) A\u3002(2) B", "numbered", ["A\u3002(2) B"], id="T35-cjk-needs-space"),
    pytest.param("(1) A\u3002 (2) B", "numbered", ["A\u3002", "B"], id="T35b-cjk-boundary"),
    # One numbered marker qualifies only with nothing but whitespace before it.
    pytest.param("Note: (1) A — x", "prose", ["Note: (1) A — x"], id="T36-single-with-preamble"),
]


@pytest.mark.parametrize("field, shape, identity_texts", TRACE)
def test_trace_table(field: str, shape: str, identity_texts: List[str]) -> None:
    """Each hand-traced row: shape, identity texts, one part each, out-of-band ids."""
    enum = enumerate_rejected_paths(field)
    assert enum.shape == shape
    assert [i.identity_text for i in enum.items] == identity_texts
    for item in enum.items:
        assert [p.text for p in item.parts] == [item.identity_text]
        assert item_point_id(OWNER, item.parts[0].text) == _oob_point_id(OWNER, item.identity_text)
    _assert_contracts(field, enum)


def test_t1_point_ids_match_an_out_of_band_digest() -> None:
    """T1's ids, computed outside Python:

        O=031f7eb70553044daf35f5ffcb0dd300bc3f1f7f1f751e61e007a37dabc4050f
        printf '%s' "{\\"kind\\":\\"rejected_item\\",\\"owner\\":\\"$O\\",\\"text\\":\\"A: REJECTED — x.\\"}" | sha256sum
          -> d8c9303ce10848dfbecc160b2d1e3a223ffd953a0737174965986a807094e12e
        (same for "B: REJECTED — y.")
          -> 12de9e68b8f310cbacf8301011428528c339d94e46376e6ea817f9875088026a
    """
    enum = enumerate_rejected_paths("(1) A: REJECTED — x.\n(2) B: REJECTED — y.")
    ids = [item_point_id(OWNER, i.parts[0].text) for i in enum.items]
    assert ids == ["d8c9303c-e108-48df-becc-160b2d1e3a22", "12de9e68-b8f3-10cb-acf8-301011428528"]


def test_t12_lithuanian_id_matches_an_out_of_band_digest_in_both_forms() -> None:
    """T12's id, computed outside Python over the composed text (``ė`` = C4 97):

        printf '%s' "{\\"kind\\":\\"rejected_item\\",\\"owner\\":\\"$O\\",\\"text\\":\\"Tik Lietuva — brangesnė …\\"}" | sha256sum
          -> 2b84ba7b6639dc012ee20546e8230508b2b6feb958abc7031de10cb49a2ab5e2

    The decomposed field enumerates to the same identity text and the same id.
    """
    assert unicodedata.is_normalized("NFC", T12_COMPOSED)
    assert T12_DECOMPOSED != T12_COMPOSED
    expected = "2b84ba7b-6639-dc01-2ee2-0546e8230508"
    for field in (T12_COMPOSED, T12_DECOMPOSED):
        (item,) = enumerate_rejected_paths(field).items
        assert item_point_id(OWNER, item.parts[0].text) == expected
    assert list(rejection_points(OWNER, T12_DECOMPOSED)) == [expected]


def test_t13_duplicates_stay_two_items_and_collapse_to_one_point() -> None:
    """Both ordinals display; ``rejection_points`` keeps the first occurrence only."""
    field = "(1) A — x\n(2) A — x"
    assert [i.ordinal for i in enumerate_rejected_paths(field).items] == [1, 2]
    points = rejection_points(OWNER, field)
    assert points == {_oob_point_id(OWNER, "A — x"): PointRef(ordinal=1, part_index=0, text="A — x")}


def test_renumbering_and_inserting_re_ids_only_the_changed_items() -> None:
    """Position never enters the id: a new option 1 adds exactly one point id."""
    before = set(rejection_points(OWNER, "(1) A: REJECTED — x.\n(2) B: REJECTED — y."))
    inserted = set(rejection_points(
        OWNER, "(1) N: REJECTED — z.\n(2) A: REJECTED — x.\n(3) B: REJECTED — y."))
    assert before < inserted
    assert inserted - before == {_oob_point_id(OWNER, "N: REJECTED — z.")}
    removed = set(rejection_points(OWNER, "(1) B: REJECTED — y."))
    assert removed == before - {_oob_point_id(OWNER, "A: REJECTED — x.")}
    # A single-marker field drops its marker, so adding (2) later keeps item 1's id.
    single = set(rejection_points(OWNER, "(1) A — x"))
    assert single < set(rejection_points(OWNER, "(1) A — x\n(2) B — y"))


def test_a_bullet_is_a_marker_stop_so_bulleted_begins_first() -> None:
    """The bullet stop moves rejected_marker's option start past ``- `` (to 2), so the
    bulleted form (0) begins first outright and each option keeps its own item."""
    enum = enumerate_rejected_paths("- A: REJECTED — x.\n- B: REJECTED — y.")
    assert enum.shape == "bulleted"
    assert [i.identity_text for i in enum.items] == ["A: REJECTED — x.", "B: REJECTED — y."]


def test_selection_ties_go_numbered_then_bulleted_then_rejected_marker(monkeypatch) -> None:
    """The tie order, pinned on ``_select`` directly.

    With the marker stop in place no field ties two forms: a rejected_marker option
    start on a marker's line lands after that marker, and numbered and bulleted
    positions are distinct line starts (a 60,000-field fuzz found no tie). A field
    cannot reach the tie rule, so it is pinned here on fabricated marker positions;
    without the stop (plant B13), T24 is the field that reaches it.
    """
    def tied(_field):
        marker = rejections._Marker(position=0, body_start=0)
        return {"numbered": [marker, marker], "bulleted": [marker, marker],
                "rejected_marker": [marker, marker]}

    monkeypatch.setattr(rejections, "_form_markers", tied)
    assert rejections._select("any")[0] == "numbered"
    monkeypatch.setattr(rejections, "_form_markers",
                        lambda f: {**tied(f), "numbered": []})
    assert rejections._select("any")[0] == "bulleted"


# ---------------------------------------------------------------------------
# The split (T17–T20, T28): long fixtures generated deterministically, byte
# lengths asserted so a row cannot silently fall under the bound.
# ---------------------------------------------------------------------------

def test_t17_one_paragraph_splits_at_the_last_sentence_within_the_bound() -> None:
    """360 sentences of exactly 25 bytes (with their space): 320 fit (7,999 bytes)."""
    sentences = [f"Sentence {i:04d} ends here." for i in range(360)]
    assert {len(s) for s in sentences} == {24}
    field = " ".join(sentences)
    assert len(field.encode("utf-8")) == 8999
    (item,) = enumerate_rejected_paths(field).items
    assert [p.text for p in item.parts] == [" ".join(sentences[:320]), " ".join(sentences[320:])]
    assert len(item.parts[0].text.encode("utf-8")) == 7999
    _assert_contracts(field, enumerate_rejected_paths(field))


def test_t18_the_paragraph_level_is_preferred_over_the_line_level() -> None:
    """Two 4,999-byte paragraphs of 49-byte lines: the cut is the paragraph break,
    not the last line break within the bound (which falls inside paragraph 2)."""
    def paragraph(tag: str) -> str:
        return "\n".join(f"Line {i:03d} of paragraph {tag}".ljust(49, ".") for i in range(100))

    one, two = paragraph("one"), paragraph("two")
    assert len(one.encode("utf-8")) == len(two.encode("utf-8")) == 4999
    field = one + "\n\n" + two
    (item,) = enumerate_rejected_paths(field).items
    assert [p.text for p in item.parts] == [one, two]
    _assert_contracts(field, enumerate_rejected_paths(field))


def test_t19_the_code_point_cut_counts_bytes_not_characters() -> None:
    """3,000 CJK characters (9,000 bytes), no space or punctuation: 2,666 per part."""
    chars = "".join(chr(0x4E00 + (i % 500)) for i in range(3000))
    assert len(chars.encode("utf-8")) == 9000
    (item,) = enumerate_rejected_paths(chars).items
    assert [p.text for p in item.parts] == [chars[:2666], chars[2666:]]
    _assert_contracts(chars, enumerate_rejected_paths(chars))


def test_t20_a_multi_token_script_cuts_at_whitespace() -> None:
    """500 Devanagari words (18 bytes, with a virama and a vowel sign) plus spaces:
    421 words fit (421 × 19 − 1 = 7,998 bytes)."""
    word = "\u0928\u092e\u0938\u094d\u0924\u0947"  # "namaste": a virama and a vowel sign
    assert len(word.encode("utf-8")) == 18
    field = " ".join([word] * 500)
    assert len(field.encode("utf-8")) == 9499
    (item,) = enumerate_rejected_paths(field).items
    assert [p.text for p in item.parts] == [" ".join([word] * 421), " ".join([word] * 79)]
    _assert_contracts(field, enumerate_rejected_paths(field))


def test_t28_the_code_point_cut_may_split_a_grapheme_cluster() -> None:
    """The rule, pinned rather than left to chance: ``a`` + 1,999 four-byte code points
    fit, so the cut lands between a thumbs-up and its skin-tone modifier."""
    field = "a" + "\U0001F44D\U0001F3FD" * 1500
    (item,) = enumerate_rejected_paths(field).items
    assert len(item.parts) == 2
    assert item.parts[0].text == field[:2000]
    assert item.parts[0].text.endswith("\U0001F44D")
    assert item.parts[1].text.startswith("\U0001F3FD")
    _assert_contracts(field, enumerate_rejected_paths(field))


# ---------------------------------------------------------------------------
# The property row (seeded, stdlib only — hypothesis is not a dependency).
# ---------------------------------------------------------------------------

PROPERTY_SEED = 20261015
_FRAGMENTS = (
    ["(1)", "(2)", "(3)", "(4)", "1.", "2.", "1)", "- ", "* ", "\u2022 ", ": REJECTED",
     "\n", "\n\n", ". ", "; ", ": ", " ", "  ", "\t", "\u0117", "e\u0307", "\u0307",
     "\u4e2d", "\u3002", "\U0001F600", "\u2028", "\u2029", "\xa0", "\x85", "\r\n",
     "(\u0661)", "\ud55c", "O(1)"]
    + ["alpha", "beta", "gamma", "option", "reason", "cache", "x", "B", "REJECTED"] * 3
)
# Long fields draw mostly from text, so their items are long enough to split; markers
# stay possible, so a long field can still be a list of long items.
_LONG_FRAGMENTS = _FRAGMENTS + ["alpha", "beta", "gamma", "option", "reason", " ", " ", ". "] * 40


def _property_fields() -> List[str]:
    rng = random.Random(PROPERTY_SEED)
    fields = []
    for n in range(2000):
        if n % 20 == 0:
            # About 5% long enough to split: mostly words, so every level gets used.
            pieces: List[str] = []
            size = 0
            target = rng.randint(8200, 20000)
            while size < target:
                frag = rng.choice(_LONG_FRAGMENTS)
                pieces.append(frag)
                size += len(frag.encode("utf-8"))
            fields.append("".join(pieces))
        else:
            fields.append("".join(rng.choice(_FRAGMENTS) for _ in range(rng.randint(0, 40))))
    return fields


def test_property_row_contracts_hold_on_generated_fields() -> None:
    """~2,000 seeded fields: totality, coverage, parts, NFC/NFD id equality, determinism."""
    long_seen = 0
    for n, field in enumerate(_property_fields()):
        context = f"seed={PROPERTY_SEED} n={n} field={field!r}"
        try:
            enum = enumerate_rejected_paths(field)
            _assert_contracts(field, enum)
            ids = [item_point_id(OWNER, p.text) for i in enum.items for p in i.parts]
            for form in ("NFC", "NFD"):
                other = enumerate_rejected_paths(unicodedata.normalize(form, field))
                assert [item_point_id(OWNER, p.text) for i in other.items for p in i.parts] == ids
        except AssertionError as exc:
            raise AssertionError(f"{context}: {exc}") from exc
        except Exception as exc:  # totality: the leaf never raises on a str
            raise AssertionError(f"{context}: raised {exc!r}") from exc
        long_seen += any(len(i.parts) > 1 for i in enum.items)
    assert long_seen >= 50, "the generator must exercise the split"


# ---------------------------------------------------------------------------
# Guards, the tier probe, the bound tripwire, the UUID agreement, the round trip.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("owner", [
    "harbor-storage-is-sqlite",
    OWNER[:63],
    OWNER.upper(),
    str(uuid.UUID(hex=OWNER[:32])),
    OWNER + "\n",
    "",
])
def test_owner_id_guard_refuses_anything_but_64_lowercase_hex(owner: str) -> None:
    """A slug or UUID passed by mistake would mint plausible wrong ids; both entry
    points refuse, even for an empty field."""
    with pytest.raises(ValueError):
        item_point_id(owner, "A — x")
    with pytest.raises(ValueError):
        rejection_points(owner, "")


def test_enumeration_takes_no_owner_and_never_raises_on_odd_input() -> None:
    """The guard lives on the id only; enumeration stays total."""
    for field in ["\x00", "\ud800", "(" * 5000, "\n" * 100, ": REJECTED" * 3, ")" * 10]:
        _assert_contracts(field, enumerate_rejected_paths(field))


def test_importing_the_leaf_pulls_in_only_identity() -> None:
    """Tier 1: stdlib plus ``mitos.identity``, probed in a fresh interpreter."""
    probe = ("import sys; import mitos.rejections; "
             "print(','.join(sorted(m for m in sys.modules if m.startswith('mitos'))))")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip().split(",") == ["mitos", "mitos.identity", "mitos.rejections"]


def test_the_leaf_source_imports_only_stdlib_and_identity() -> None:
    """The AST twin of the probe: a function-local import shows up here too."""
    source = Path(rejections.__file__).read_text(encoding="utf-8")
    modules = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
    assert {m for m in modules if m.startswith("mitos")} == {"mitos.identity"}
    assert modules - {"mitos.identity"} <= set(sys.stdlib_module_names)


def test_the_part_bound_is_tripwired_to_the_embedding_model_and_prefix(tmp_path) -> None:
    """A model swap or a prefix change reds offline until the bound is re-derived."""
    from mitos import models
    from mitos.embeddings import GeminiEmbeddingProvider

    assert models.MODEL_IDS["EMBEDDING"] == BOUND_DERIVED_FOR_MODEL
    assert ITEM_PART_BOUND_BYTES + DOCUMENT_PREFIX_BYTES <= BOUND_MODEL_TOKEN_LIMIT

    calls: list = []

    def embed_content(*, model: str, contents):
        calls.append(contents)
        count = len(contents) if isinstance(contents, list) else 1
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.0]) for _ in range(count)])

    provider = GeminiEmbeddingProvider(str(tmp_path / "c.sqlite"), api_key="dummy")
    provider.client = SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
    provider.get_embedding("probe-single", is_query=False)
    provider.get_embeddings_batch(["probe-batch"])
    provider.embed_documents(["probe-documents"])  # the path rejection items take
    assert len(calls) == 3
    single, (batched,), (documents,) = calls
    documents = documents.parts[0].text  # the uncached path sends one Content per text
    for sent, text in ((single, "probe-single"), (batched, "probe-batch"),
                       (documents, "probe-documents")):
        assert sent.endswith(text)
        assert len(sent[: -len(text)].encode("utf-8")) == DOCUMENT_PREFIX_BYTES


def test_point_id_formatting_agrees_with_vector_store_hash_to_uuid() -> None:
    """The leaf formats the UUID itself (it may not import a Tier-2 service)."""
    from mitos.vector_store import hash_to_uuid

    for text in ["A — x", T12_COMPOSED, "\u4e2d\u6587", ""]:
        payload = json.dumps({"kind": "rejected_item", "owner": OWNER, "text": text},
                             sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        assert item_point_id(OWNER, text) == hash_to_uuid(digest)


def test_enumeration_round_trips_through_json() -> None:
    """Lists, never tuples: to_dict → json → from_dict gives the equal enumeration."""
    for field in ["Two paths:\n(1) A — x\n(2) B — y", T12_DECOMPOSED, "",
                  " ".join(f"Sentence {i:04d} ends here." for i in range(360))]:
        enum = enumerate_rejected_paths(field)
        data = json.loads(json.dumps(enum.to_dict(), ensure_ascii=False))
        assert data == enum.to_dict()
        assert Enumeration.from_dict(data) == enum
    ref = PointRef(ordinal=2, part_index=1, text="x")
    assert PointRef.from_dict(json.loads(json.dumps(ref.to_dict()))) == ref


def test_the_parsed_flag_is_false_for_prose_and_empty_only() -> None:
    """``.parsed`` is what a renderer gates its per-item lines on."""
    assert enumerate_rejected_paths("(1) A\n(2) B").parsed
    assert not enumerate_rejected_paths("just prose").parsed
    assert not enumerate_rejected_paths("").parsed


def test_ucd_pin_guard() -> None:
    """The out-of-band ids NFC a decomposed field, so they share identity's UCD pin."""
    assert unicodedata.unidata_version == "15.1.0"
