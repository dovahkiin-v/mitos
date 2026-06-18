"""Golden trace table + per-rule fixtures for the slug-free canonical-core hash.

This is the MI-1 byte-stability gate for ``mitos.identity`` (V1-D2, §11/§12). It
runs on every push to the 3.13 CI floor; a digest divergence is an MI-1
violation surfaced at the gate, never a silent regeneration.

The §11 golden trace table (``DECISION_GOLDENS`` + ``OQ_GOLDENS``) is authored
upfront as the fixture source. Its frozen 64-hex digests were **generated once**
by running ``compute_node_id`` on the pinned interpreter (**Python 3.13.5 / UCD
15.1.0**) — never hand-typed. Two layers of assertion:

* **Frozen digests** (``test_golden_digest_*``) — the byte-stability lock. These
  shift only if the serialized bytes change (a normalization-rule regression or
  a UCD bump). A UCD bump is a deliberate one-time regen recorded as an
  amendment (see ``test_ucd_pin_guard``).
* **Convergence / divergence relations** (``test_*_relation``) — the durable
  behavioral lock. They survive a UCD bump (which shifts absolute digests, not
  *which* inputs converge), so they pin V1-D2/V1-D3 and the two §12 asymmetries
  regardless of interpreter.

Pure functions only — no DB, no services, no async, no mocks.

Green-field-test note (testmon cold cache): under ``-n auto`` the first run of a
brand-new test file can hit "Different tests collected between gw0 and gw1".
Warm once with ``pytest tests/test_identity.py -n 0 --testmon-forceselect`` (or
run the file directly) before the parallel suite.
"""

import sys
import unicodedata

import pytest

from mitos.identity import (
    canonical_core_json_form,
    canonical_core_string_norm,
    compute_node_id,
    dedup_preserve_order,
    mechanism_canonical_norm,
    mechanism_refs_list_norm,
    questions_raised_list_norm,
)

# --- Unicode building blocks (explicit code points so the NFC/NFD pairing is
#     unambiguous and durable, independent of how this file's bytes were saved) ---
LT_AXIOM_NFC = "Sprendimas naudoti kabut\u0117"      # "ė" composed (U+0117)
LT_AXIOM_NFD = "Sprendimas naudoti kabute\u0307"     # "e" + COMBINING DOT ABOVE
OQ_TOPIC_NFC = "Ar naudoti kabut\u0117?"
OQ_TOPIC_NFD = "Ar naudoti kabute\u0307?"
OQ_Q_NFC = "Kod\u0117l?"
OQ_Q_NFD = "Kode\u0307l?"
CJK = "使用 SQLite 存储决策图"   # CJK ideographs: NFC is identity, exercises UTF-8 byte form

AX = "Use SQLite for the graph store"


# ---------------------------------------------------------------------------
# §11 Golden trace table — (name, kind, kwargs, frozen sha256 hex)
# Frozen on Python 3.13.5 / UCD 15.1.0. Generated, never hand-authored.
# ---------------------------------------------------------------------------

DECISION_GOLDENS = [
    ("D1", {"axiom": AX, "mechanism_refs": ["sqlite", "wal"]},
     "22d27886398cfa8d8d7c607a7ccfc1fe3eb13ccfcd111b4064ae37e6c77954fa"),
    ("D2", {"axiom": AX, "mechanism_refs": ["wal", "sqlite"]},
     "22d27886398cfa8d8d7c607a7ccfc1fe3eb13ccfcd111b4064ae37e6c77954fa"),
    ("D3", {"axiom": "use sqlite for the graph store", "mechanism_refs": ["sqlite", "wal"]},
     "9d4deaef159d7d0adc47b976989ea7632d62b701d4545e6aafb549ec3d095ad7"),
    ("D4", {"axiom": "  Use SQLite for the graph store  ", "mechanism_refs": ["sqlite", "wal"]},
     "22d27886398cfa8d8d7c607a7ccfc1fe3eb13ccfcd111b4064ae37e6c77954fa"),
    ("D5a", {"axiom": LT_AXIOM_NFC, "mechanism_refs": []},
     "0495d6fbced536f8eb817f190b3571e5c86aba63283f41910fdc1cbf58bcd423"),
    ("D5b", {"axiom": LT_AXIOM_NFD, "mechanism_refs": []},
     "0495d6fbced536f8eb817f190b3571e5c86aba63283f41910fdc1cbf58bcd423"),
    ("D6", {"axiom": CJK, "mechanism_refs": ["sqlite"]},
     "bf327f65e9d165d6027c854340625966b29b9cbe7e1544899376e3b6cb9c4fa3"),
    ("D7", {"axiom": AX, "mechanism_refs": ["SQLite", "WAL"]},
     "22d27886398cfa8d8d7c607a7ccfc1fe3eb13ccfcd111b4064ae37e6c77954fa"),
    ("D8", {"axiom": AX, "mechanism_refs": ["str_casefold", "str-casefold", "Str Casefold"]},
     "3c09a195fc4cb2ca2bcd2daff2e96d5013ae9c35bf9ac59316708ae110cf8f5f"),
    ("D9", {"axiom": AX, "mechanism_refs": ["sqlite", "SQLite", "sqlite!"]},
     "730d449aece1ba24bd449858fbaf519bd978561559e1d857d764bc35bf49a538"),
    ("D10", {"axiom": AX, "mechanism_refs": ["sqlite", "", "   ", "wal"]},
     "22d27886398cfa8d8d7c607a7ccfc1fe3eb13ccfcd111b4064ae37e6c77954fa"),
    ("D11", {"axiom": AX, "mechanism_refs": []},
     "7176159ec095a334a836d7d6663b7571ee4b1f7ba4692e379b4ab94e0c4ce5f9"),
    ("D12", {"axiom": "When should we shard?", "mechanism_refs": []},
     "a768cff943c333232601aeb563bae6dfb6e1835e663e0c48262920dd0f841c80"),
]

OQ_GOLDENS = [
    ("Q1", {"topic": "SQLite vs Postgres for the store",
            "questions_raised": ["Which scales for our write pattern?", "What is the ops cost?"]},
     "f96ea6c26511e0e4626a4311e2f8798156ba98b93bcefe5193bfb6efe6db967f"),
    ("Q2", {"topic": "SQLite vs Postgres for the store",
            "questions_raised": ["What is the ops cost?", "Which scales for our write pattern?"]},
     "9d1971dbbbb5f732f187a5625f5ca332014cf953dd14dad55c298f48baf85fe0"),
    ("Q3", {"topic": "SQLite vs Postgres for the store",
            "questions_raised": ["What is the ops cost?", "Which scales for our write pattern?",
                                 "What is the ops cost?"]},
     "9d1971dbbbb5f732f187a5625f5ca332014cf953dd14dad55c298f48baf85fe0"),
    ("Q4", {"topic": "SQLite vs Postgres for the store",
            "questions_raised": ["which scales for our write pattern?", "What is the ops cost?"]},
     "edc705e97b137c7e8612309a2219227084fb5f3b21ee661026ab6ffc167294dd"),
    ("Q5a", {"topic": OQ_TOPIC_NFC, "questions_raised": [OQ_Q_NFC]},
     "a1e8ccd0f7514678c3df67290f2febdb93735ff9f74db11f38ddb00fca89feb9"),
    ("Q5b", {"topic": OQ_TOPIC_NFD, "questions_raised": [OQ_Q_NFD]},
     "a1e8ccd0f7514678c3df67290f2febdb93735ff9f74db11f38ddb00fca89feb9"),
    ("Q6", {"topic": "SQLite vs Postgres for the store",
            "questions_raised": ["Which scales for our write pattern?", "", "   "]},
     "788d178420e242b50b4527b58d61e71917453baa9e2ef8db0b3aedc19d8b527a"),
    ("Q7", {"topic": "When should we shard?", "questions_raised": ["When should we shard?"]},
     "c77554d8b2c895e889185323b307913cafeb4567649cc508d23b8e81b0ac3ca6"),
]

# name -> id, for cross-row relation assertions.
DEC_ID = {name: digest for name, _kw, digest in DECISION_GOLDENS}
OQ_ID = {name: digest for name, _kw, digest in OQ_GOLDENS}


def _decision_id(**kw: object) -> str:
    return compute_node_id(kind="decision", **kw)


def _oq_id(**kw: object) -> str:
    return compute_node_id(kind="open_question", **kw)


# ---------------------------------------------------------------------------
# Layer 1: frozen-digest byte-stability gate (MI-1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, kwargs, expected", DECISION_GOLDENS,
                         ids=[r[0] for r in DECISION_GOLDENS])
def test_golden_digest_decision(name: str, kwargs: dict, expected: str) -> None:
    """Each decision row hashes to its frozen §11 digest (byte-stability lock)."""
    assert _decision_id(**kwargs) == expected


@pytest.mark.parametrize("name, kwargs, expected", OQ_GOLDENS,
                         ids=[r[0] for r in OQ_GOLDENS])
def test_golden_digest_open_question(name: str, kwargs: dict, expected: str) -> None:
    """Each open_question row hashes to its frozen §11 digest (byte-stability lock)."""
    assert _oq_id(**kwargs) == expected


def test_all_golden_ids_are_64_hex() -> None:
    """Every minted id is a 64-char lowercase hex string (nodes.id PK shape)."""
    for _name, digest in {**DEC_ID, **OQ_ID}.items():
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


# ---------------------------------------------------------------------------
# Layer 2: convergence / divergence relations (durable behavioral lock — these
# survive a UCD bump; they pin V1-D2/V1-D3 + the two §12 asymmetries).
# ---------------------------------------------------------------------------

def test_mechanism_order_independence_relation() -> None:
    """D1 == D2 — mechanism_refs is an unordered set (sorted(set)). Order axis."""
    assert DEC_ID["D1"] == DEC_ID["D2"]


def test_axiom_strip_relation() -> None:
    """D1 == D4 — canonical_core_string_norm strips leading/trailing whitespace."""
    assert DEC_ID["D1"] == DEC_ID["D4"]


def test_mechanism_casefold_relation() -> None:
    """D1 == D7 — mechanism_canonical_norm casefolds (vs D3 axiom case-preserve)."""
    assert DEC_ID["D1"] == DEC_ID["D7"]


def test_axiom_case_preserved_relation() -> None:
    """D3 != D1 — axiom case carries meaning (case axis; asymmetry vs D7)."""
    assert DEC_ID["D3"] != DEC_ID["D1"]


def test_axiom_nfc_convergence_relation() -> None:
    """D5a == D5b — composed and decomposed Lithuanian "ė" converge under NFC."""
    assert DEC_ID["D5a"] == DEC_ID["D5b"]


def test_mechanism_punct_ws_fold_relation() -> None:
    """D8 == same-axiom decision with mechanism_refs=['str-casefold'].

    'str_casefold' / 'str-casefold' / 'Str Casefold' all fold to one token.
    """
    assert mechanism_refs_list_norm(["str_casefold", "str-casefold", "Str Casefold"]) == ["str-casefold"]
    assert DEC_ID["D8"] == _decision_id(axiom=AX, mechanism_refs=["str-casefold"])


def test_mechanism_set_dedup_relation() -> None:
    """D9 == same-axiom decision with mechanism_refs=['sqlite'] (post-fold dedup); != D1."""
    assert mechanism_refs_list_norm(["sqlite", "SQLite", "sqlite!"]) == ["sqlite"]
    assert DEC_ID["D9"] == _decision_id(axiom=AX, mechanism_refs=["sqlite"])
    assert DEC_ID["D9"] != DEC_ID["D1"]  # D1's list is ['sqlite','wal']


def test_mechanism_empty_filter_relation() -> None:
    """D10 == D1 — empty/whitespace-only mechanism items dropped (raw `if m.strip()`)."""
    assert DEC_ID["D10"] == DEC_ID["D1"]


def test_question_order_significant_relation() -> None:
    """Q1 != Q2 — question ORDER is identity-significant (order axis; vs D2)."""
    assert OQ_ID["Q1"] != OQ_ID["Q2"]


def test_question_order_preserving_dedup_relation() -> None:
    """Q2 == Q3 — order-preserving dedup keeps first occurrence, never sorts."""
    assert OQ_ID["Q2"] == OQ_ID["Q3"]


def test_question_case_preserved_relation() -> None:
    """Q4 != Q1 — question case is preserved (case axis; asymmetry vs D7)."""
    assert OQ_ID["Q4"] != OQ_ID["Q1"]


def test_topic_and_question_nfc_convergence_relation() -> None:
    """Q5a == Q5b — topic and question items both NFC-normalize (case-preserved fields)."""
    assert OQ_ID["Q5a"] == OQ_ID["Q5b"]


def test_question_empty_filter_relation() -> None:
    """Q6 == same-topic OQ with the single non-empty question (raw `if q.strip()`)."""
    assert OQ_ID["Q6"] == _oq_id(
        topic="SQLite vs Postgres for the store",
        questions_raised=["Which scales for our write pattern?"],
    )


def test_kind_discriminator_no_collision() -> None:
    """D12 != Q7 — a decision and an OQ with text-identical strings don't collide.

    ``kind`` lives inside the hashed object, so identical canonical text under
    two kinds yields two distinct ids.
    """
    assert DEC_ID["D12"] != OQ_ID["Q7"]


# ---------------------------------------------------------------------------
# Serialization / byte-form rows (S1/S2/S3) — assert on canonical_core_json_form.
# These are the more diagnostic half of the MI-7 proof (human-readable JSON).
# ---------------------------------------------------------------------------

def test_s1_separators_no_incidental_whitespace() -> None:
    """S1 — separators=(',',':'): the serialized form carries no ', ' or ': '."""
    serialized = canonical_core_json_form(
        {"kind": "decision", "axiom": AX, "mechanism_refs": ["sqlite", "wal"]}
    )
    assert serialized == '{"axiom":"Use SQLite for the graph store","kind":"decision","mechanism_refs":["sqlite","wal"]}'
    assert ", " not in serialized
    assert ": " not in serialized


def test_s2_ensure_ascii_false_literal_utf8_bytes() -> None:
    """S2 — ensure_ascii=False: non-ASCII serializes as literal UTF-8, not \\uXXXX (MI-7)."""
    # CJK
    cjk_serialized = canonical_core_json_form(
        {"kind": "decision", "axiom": CJK, "mechanism_refs": ["sqlite"]}
    )
    assert "\\u" not in cjk_serialized
    assert "使用 SQLite 存储决策图" in cjk_serialized
    assert "使用".encode("utf-8") in cjk_serialized.encode("utf-8")

    # Lithuanian (NFC ė)
    lt_serialized = canonical_core_json_form(
        {"kind": "decision", "axiom": LT_AXIOM_NFC, "mechanism_refs": []}
    )
    assert "\\u" not in lt_serialized
    assert "kabut\u0117" in lt_serialized
    # the NFC "ė" is the two-byte UTF-8 sequence 0xC4 0x97, not an escape.
    assert b"\xc4\x97" in lt_serialized.encode("utf-8")
    assert b"\\u0117" not in lt_serialized.encode("utf-8")


def test_s3_sort_keys_robustness() -> None:
    """S3 — sort_keys=True: dict built in non-alphabetical order still serializes
    alphabetically (axiom, kind, mechanism_refs) and yields the D1 id."""
    scrambled = {"mechanism_refs": ["sqlite", "wal"], "kind": "decision", "axiom": AX}
    serialized = canonical_core_json_form(scrambled)
    assert serialized == '{"axiom":"Use SQLite for the graph store","kind":"decision","mechanism_refs":["sqlite","wal"]}'
    # and the id is stable regardless of construction order
    assert compute_node_id(kind="decision", axiom=AX, mechanism_refs=["sqlite", "wal"]) == DEC_ID["D1"]


# ---------------------------------------------------------------------------
# Per-rule unit fixtures — so a regression localizes to the offending rule.
# ---------------------------------------------------------------------------

def test_canonical_core_string_norm_nfc_convergence() -> None:
    """NFC: a decomposed string normalizes to its composed form."""
    assert LT_AXIOM_NFC != LT_AXIOM_NFD  # genuinely different code points
    assert canonical_core_string_norm(LT_AXIOM_NFD) == canonical_core_string_norm(LT_AXIOM_NFC)
    assert canonical_core_string_norm(LT_AXIOM_NFD) == LT_AXIOM_NFC


def test_canonical_core_string_norm_strips_ends_only() -> None:
    """Strips leading/trailing whitespace; preserves internal whitespace and case."""
    assert canonical_core_string_norm("  Use SQLite  ") == "Use SQLite"
    assert canonical_core_string_norm("a  b\tc") == "a  b\tc"   # internal ws is content
    assert canonical_core_string_norm("Use SQLite") == "Use SQLite"   # case preserved


def test_mechanism_canonical_norm_casefolds() -> None:
    """Mechanism tokens casefold (the case-axis counterpart to string norm)."""
    assert mechanism_canonical_norm("SQLite") == "sqlite"
    assert mechanism_canonical_norm("WAL") == "wal"


def test_mechanism_canonical_norm_punct_ws_fold() -> None:
    """Maximal runs of ASCII punctuation/whitespace fold to a single hyphen, ends stripped."""
    assert mechanism_canonical_norm("str_casefold") == "str-casefold"
    assert mechanism_canonical_norm("str-casefold") == "str-casefold"
    assert mechanism_canonical_norm("Str Casefold") == "str-casefold"
    assert mechanism_canonical_norm("node-scopes") == "node-scopes"
    assert mechanism_canonical_norm("sqlite!") == "sqlite"
    assert mechanism_canonical_norm("!!!sqlite!!!") == "sqlite"
    assert mechanism_canonical_norm("a___b   c") == "a-b-c"


def test_mechanism_canonical_norm_nfc_then_casefold() -> None:
    """NFC runs before casefold: a decomposed accented mechanism token converges."""
    composed = "kabut\u0117"        # ė
    decomposed = "kabute\u0307"     # e + combining dot
    assert mechanism_canonical_norm(composed) == mechanism_canonical_norm(decomposed)


def test_mechanism_canonical_norm_non_ascii_passes_through() -> None:
    """Non-ASCII punctuation/whitespace is NOT folded (the fold is ASCII-only)."""
    # A no-break space (U+00A0) is non-ASCII whitespace → survives the fold.
    assert " " in mechanism_canonical_norm("a\u00a0b")
    assert mechanism_canonical_norm("a\u00a0b") == "a\u00a0b"


def test_mechanism_refs_list_norm_unordered_set() -> None:
    """sorted(set): order-independent, deduped, code-point sorted, returns a list."""
    out = mechanism_refs_list_norm(["wal", "sqlite"])
    assert out == ["sqlite", "wal"]
    assert isinstance(out, list)
    assert mechanism_refs_list_norm(["sqlite", "wal"]) == out
    assert mechanism_refs_list_norm(["sqlite", "SQLite", "sqlite!"]) == ["sqlite"]


def test_mechanism_refs_list_norm_raw_empty_filter() -> None:
    """Empty / whitespace-only RAW items are filtered before folding."""
    assert mechanism_refs_list_norm(["sqlite", "", "   ", "wal"]) == ["sqlite", "wal"]
    assert mechanism_refs_list_norm([]) == []


def test_questions_raised_list_norm_order_preserving() -> None:
    """Order preserved (NEVER sorted); case preserved; returns a list."""
    out = questions_raised_list_norm(["B question?", "A question?"])
    assert out == ["B question?", "A question?"]   # NOT sorted to A, B
    assert isinstance(out, list)
    assert questions_raised_list_norm(["Why?"]) != questions_raised_list_norm(["why?"])


def test_questions_raised_list_norm_order_preserving_dedup() -> None:
    """Order-preserving dedup keeps first occurrence."""
    assert questions_raised_list_norm(["A?", "B?", "A?"]) == ["A?", "B?"]


def test_questions_raised_list_norm_raw_empty_filter() -> None:
    """Empty / whitespace-only RAW question items are filtered before norming."""
    assert questions_raised_list_norm(["X?", "", "   "]) == ["X?"]
    assert questions_raised_list_norm([]) == []


def test_dedup_preserve_order_helper() -> None:
    """dedup_preserve_order keeps first occurrence, preserves order, no sort."""
    assert dedup_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert dedup_preserve_order([]) == []


def test_two_list_norms_are_distinct_functions() -> None:
    """The order asymmetry: the same reordered input diverges for questions,
    converges for mechanisms — proving they are NOT one shared list-norm."""
    reordered = ["beta", "alpha"]
    original = ["alpha", "beta"]
    # mechanisms: order-independent → equal
    assert mechanism_refs_list_norm(reordered) == mechanism_refs_list_norm(original)
    # questions: order-significant → different
    assert questions_raised_list_norm(reordered + ["?"]) != questions_raised_list_norm(original + ["?"])
    assert questions_raised_list_norm(["beta", "alpha"]) == ["beta", "alpha"]


# ---------------------------------------------------------------------------
# compute_node_id contract
# ---------------------------------------------------------------------------

def test_compute_node_id_rejects_unknown_kind() -> None:
    """A third kind is a ValueError, not an extension point (defensive)."""
    with pytest.raises(ValueError, match="unknown kind"):
        compute_node_id(kind="mechanism", axiom="x")
    with pytest.raises(ValueError):
        compute_node_id(kind="", axiom="x")


def test_compute_node_id_is_keyword_only() -> None:
    """The signature is keyword-only — a positional call fails loudly (slug-safety)."""
    with pytest.raises(TypeError):
        compute_node_id("decision", "some axiom")  # type: ignore[misc]


def test_compute_node_id_optional_lists_default_empty() -> None:
    """Omitting mechanism_refs / questions_raised is equivalent to passing []."""
    assert compute_node_id(kind="decision", axiom=AX) == \
        compute_node_id(kind="decision", axiom=AX, mechanism_refs=[])
    assert compute_node_id(kind="open_question", topic="T") == \
        compute_node_id(kind="open_question", topic="T", questions_raised=[])


# ---------------------------------------------------------------------------
# UCD-pin guard — the goldens are frozen against UCD 15.1.0. A future
# interpreter on a newer UCD fails this loudly: the intended signal that the
# digests need a deliberate, recorded one-time regeneration (never silent).
# ---------------------------------------------------------------------------

def test_ucd_pin_guard() -> None:
    """The frozen §11 digests are pinned to Python 3.13.x / UCD 15.1.0."""
    assert sys.version_info >= (3, 13)
    assert unicodedata.unidata_version == "15.1.0", (
        f"UCD version is {unicodedata.unidata_version!r}, not '15.1.0'. The §11 "
        "golden digests are frozen against UCD 15.1.0; a bump requires a "
        "deliberate one-time regeneration recorded as an amendment (vision "
        "UCD-bump policy), never a silent edit of the frozen constants."
    )
