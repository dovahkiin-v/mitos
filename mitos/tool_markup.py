"""Tool-call markup — the one pattern set the write verbs refuse, and its scanner.

When an agent's tool call is mis-serialised, part of the call's own syntax lands inside
an argument's value: a closing ``</parameter>``, a bare ``</invoke>``, the next
argument's opening ``<parameter name="context">``. Stored, it reads back as reasoning,
and in a mechanism it folds into the identity hash (``wal-mode</parameter>`` becomes
``wal-mode-parameter``). Source and graph then agree about the damage, so no health
surface can see it; the only clean point to stop it is the write.

``record_decision`` and ``amend_commentary`` scan every value they write with
:func:`find_tool_call_markup` and refuse on any hit, before anything is parsed,
normalised or hashed. The write verbs refuse this markup, but the parser still accepts
it from hand-authored markdown: the refusal narrows two producers and leaves the
consumer as it is, and a producer narrower than its consumer cannot drift the format.

Matching is literal and case-sensitive, because a leaked tag is verbatim model output.
Text inside an inline-code span is exempt (``markers.mask_inline_code``, the exemption
the structural-token guard teaches); fenced blocks are not.

Tier 1: stdlib and ``mitos.markers`` only.
"""

import re
from typing import Any, Dict, Iterable, List, Pattern, Tuple

from mitos.markers import mask_inline_code

# Every argument name of `record_decision` and `amend_commentary`, whose closing tag is
# the leak shape. Hand-listed at Tier 1 and pinned against both tools' live input
# schemas by a reflection row, so a new argument reds there until it is added. Closers
# of arguments that are never written (`</project>`) are included on purpose: the shape
# is the call's own argument names, whichever argument leaked.
FIELD_CLOSER_NAMES: Tuple[str, ...] = (
    "axiom", "rejected_paths", "scope", "slug", "mechanisms", "context",
    "supersedes", "corrects", "amends", "narrows", "depends_on", "resolves",
    "contradicts", "derives_from", "cites", "acknowledge_neighbors", "draft_digest",
    "project", "new_slug", "clear", "invalidates_if",
)

# The namespace prefix is built by concatenation: the tooling that writes this code
# parses the literal form.
_NAMESPACE = "ant" + "ml:"

# The pattern set, labelled. A generic `</word>` is not a member (it would refuse a
# `</div>` in an ADR about a template), and an opening `<parameter`/`<invoke` counts
# only with ` name=` after it (a bare `<parameter>` is prose about generics).
TOOL_CALL_MARKUP: Tuple[Tuple[str, Pattern[str]], ...] = (
    ("function_calls", re.compile(r"</?function_calls>")),
    ("invoke", re.compile(r"<invoke\s+name=[^<>\n]*>?|</invoke>")),
    ("parameter", re.compile(r"<parameter\s+name=[^<>\n]*>?|</parameter>")),
    ("antml", re.compile("</?" + re.escape(_NAMESPACE) + r"[A-Za-z][^<>\n]*>?")),
    ("field_closer", re.compile(
        "</(?:" + "|".join(re.escape(name) for name in FIELD_CLOSER_NAMES) + ")>")),
)


def field_values(name: str, value: Any) -> List[Tuple[str, Any]]:
    """Expands one argument into the ``(label, value)`` pairs the scanner reads.

    Args:
        name: The argument name.
        value: Its value; a list or tuple is expanded element by element.

    Returns:
        ``[(name, value)]``, or ``[("name[0]", v0), ...]`` for a list or tuple.
    """
    if isinstance(value, (list, tuple)):
        return [(f"{name}[{index}]", item) for index, item in enumerate(value)]
    return [(name, value)]


def find_tool_call_markup(values: Iterable[Tuple[str, Any]]) -> List[Dict[str, Any]]:
    """Finds every tool-call markup span in the values a write verb is about to write.

    The write verbs refuse this markup, but the parser still accepts it from
    hand-authored markdown: this narrows two producers and leaves the consumer as it
    is, and a producer narrower than its consumer cannot drift the format. The match
    runs on an inline-code-masked copy (same length), so a backticked tag is exempt
    and every offset is an offset into the original.

    Args:
        values: ``(field, value)`` pairs in the caller's order. A value that is not a
            string (``None``, junk) is skipped: this judges markup, not types.

    Returns:
        A list of ``{"field", "span", "offset"}`` hits, in the caller's field order and
        then by offset; ``span`` is verbatim from the value and ``offset`` is its
        0-based character offset. Every hit is returned; ``[]`` when there are none.
    """
    hits: List[Dict[str, Any]] = []
    for field, value in values:
        if not isinstance(value, str):
            continue
        masked = mask_inline_code(value)
        found = [
            (match.start(), match.end())
            for _label, pattern in TOOL_CALL_MARKUP
            for match in pattern.finditer(masked)
        ]
        for start, end in sorted(found):
            hits.append({"field": field, "span": value[start:end], "offset": start})
    return hits
