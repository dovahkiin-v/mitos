"""Re-materializes a `### slug` source block for a graph-only node.

A node whose entry has left `decisions.md` is invisible to the corpus: `mitos rebuild`
replays only what the markdown holds, so the node is dropped, and the completeness
gate then refuses the swap — the tool's own repair story stops working, on exactly the
corpus that needs repairing. The store never acts on whole-entry buffer-absence, so
the repair is to restore the block, never to let the graph drop the node.

The graph already holds every field the parser reads, so the block is a *derivation*,
not an authoring act: the round-trip is hash-exact by construction, and this module
refuses to write anything it cannot prove so.

**Refusal is at two scopes (P13 — decision text is untrusted input).** Isolated
fidelity is not enough: a block whose commentary happens to contain a `### ` line, a
stray `**Decided:**`, or a `[DECISION_TRANSCRIPT]` marker would re-parse to the right
node id while ALSO minting a phantom entry — or would bleed into its neighbour, which
the retired `mark` mode is a live demonstration of. So the whole buffer is re-parsed
after the splice and every pre-existing entry must be byte-unchanged.

**Restored into the buffer, never an archive.** Archives are quarter-partitioned and
`nodes.created_at` is stamped at COMMIT time, not authoring time, so it cannot date an
entry honestly — most of this corpus stamps one month because it was re-committed
wholesale then. Choosing a quarter would put a fabricated date in the gold source.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from mitos.errors import MitosError

# Emission order. The canonical core first, then commentary, then provenance, then
# relations, then the transcript — matching `format-spec.md`'s own section order and
# its sample entry, so a restored block is indistinguishable from a hand-authored one.
# Field ORDER does not affect the node id (the hash reads named fields, not position);
# it is a readability contract, not a correctness one.
_RELATIONSHIP_EMISSION: Tuple[Tuple[str, str], ...] = (
    ("supersedes", "Supersedes"),
    ("corrects", "Corrects"),
    ("amends", "Amends"),
    ("narrows", "Narrows"),
    ("depends_on", "Depends-On"),
    ("resolves", "Resolves"),
    ("contradicts", "Contradicts"),
    ("derives_from", "Derives-From"),
    ("cites", "Cites"),
)


class RestoreError(MitosError):
    """A block could not be regenerated at full fidelity, so nothing was written."""


def render_source_block(
    node: Dict[str, Any],
    outgoing_edges: List[Dict[str, str]],
    transcript: Optional[str] = None,
) -> str:
    """Renders a graph node back into its `### slug` markdown entry.

    Args:
        node: The node's reader-facing dict.
        outgoing_edges: Its outgoing edges as ``{"kind", "target"}`` dicts.
        transcript: The node's transcript text, when it has one.

    Returns:
        The entry block, newline-terminated, with no surrounding blank lines.

    Raises:
        RestoreError: If the node lacks a field the parser requires, so a caller can
            never write a block that would come back as a parse failure.
    """
    slug = (node.get("slug") or "").strip()
    axiom = (node.get("core_axiom") or "").strip()
    rejected = (node.get("rejected_paths") or "").strip()
    if not slug:
        raise RestoreError("node has no slug")
    if not axiom:
        raise RestoreError(f"'{slug}' has no axiom — the canonical core is required")
    if not rejected:
        # M5: `**Rejected:**` is mandatory on a decision. Emitting a block without it
        # would restore an entry that fails to parse — trading one broken state for
        # another, and silently.
        raise RestoreError(f"'{slug}' has no rejected_paths — required on a decision (M5)")

    lines: List[str] = [f"### {slug}", "", f"**Decided:** {axiom}", f"**Rejected:** {rejected}"]

    mechanisms = node.get("mechanisms") or []
    if mechanisms:
        lines.append(f"**Mechanisms:** {', '.join(mechanisms)}")
    scope = node.get("scope") or []
    if scope:
        lines.append(f"**Scope:** {', '.join(scope)}")

    invalidates_if = (node.get("invalidates_if") or "").strip()
    if invalidates_if:
        lines.append(f"**Invalidates-If:** {invalidates_if}")
    context = (node.get("context") or "").strip()
    if context:
        lines.append(f"**Context:** {context}")

    # `Source` is emitted ONLY when it is not the default. Absent means `user`
    # (format-spec), so emitting `**Source:** user` is noise — but OMITTING a non-user
    # value is a defect: a later rebuild would replay the entry with the default and
    # silently flip stored provenance, which is precisely the invisible species the
    # divergence detector reports.
    source = (node.get("source") or "user").strip()
    if source and source != "user":
        lines.append(f"**Source:** {source}")

    by_kind: Dict[str, List[str]] = {}
    for edge in outgoing_edges or []:
        by_kind.setdefault(edge["kind"], []).append(edge["target"])
    for kind, label in _RELATIONSHIP_EMISSION:
        targets = by_kind.get(kind)
        if targets:
            # Bracketed, matching how `format-spec.md` authors a citation and how the
            # deterministic parser stores one.
            lines.append(f"**{label}:** {', '.join(f'[{t}]' for t in targets)}")

    if transcript:
        lines.append("[DECISION_TRANSCRIPT]")
        lines.extend(transcript.splitlines())
        lines.append("[/DECISION_TRANSCRIPT]")

    return "\n".join(lines) + "\n"


def _entry_fingerprint(entry: Any) -> Dict[str, Any]:
    """Reduces a parsed entry to the fields a fidelity check compares.

    Args:
        entry: A ``ParsedEntry``.

    Returns:
        A comparable dict of every field a restore could disturb.
    """
    from mitos.divergence import declared_edges

    return {
        "slug": entry.slug,
        "kind": entry.kind,
        "axiom": entry.axiom,
        "mechanisms": list(entry.mechanisms or []),
        "topic": entry.topic,
        "questions_raised": list(entry.questions_raised or []),
        "rejected_paths": entry.rejected_paths,
        "invalidates_if": entry.invalidates_if,
        "scope": list(entry.scope or []),
        "context": entry.context,
        "source": entry.source,
        "transcript": entry.transcript,
        "edges": [(e["kind"], e["target"]) for e in declared_edges(entry)],
    }


def verify_block_in_isolation(block: str, node: Dict[str, Any]) -> None:
    """Asserts a rendered block re-parses to EXACTLY the node it came from.

    Args:
        block: The rendered entry block.
        node: The node it was rendered from.

    Raises:
        RestoreError: On any parse failure, entry-count mismatch, node-id shift, or
            commentary field that did not survive the round trip.
    """
    from mitos.identity import compute_node_id
    from mitos.parser import parse_entry_stream

    failures: List[Any] = []
    entries = parse_entry_stream(block, "decision", failures=failures)
    slug = node.get("slug")
    if failures:
        raise RestoreError(f"'{slug}': regenerated block does not parse ({failures[0]})")
    if len(entries) != 1:
        # The guard an id-only check would miss: a `### ` line inside the commentary
        # mints a PHANTOM entry while the first one still hashes correctly.
        raise RestoreError(
            f"'{slug}': regenerated block parses as {len(entries)} entries, expected 1 — "
            "its commentary probably contains a `### ` line or a stray field marker"
        )

    entry = entries[0]
    rebuilt_id = compute_node_id(
        kind=entry.kind,
        axiom=entry.axiom,
        mechanism_refs=entry.mechanisms,
        topic=entry.topic,
        questions_raised=entry.questions_raised,
    )
    if rebuilt_id != node.get("id"):
        raise RestoreError(
            f"'{slug}': regenerated block hashes to a DIFFERENT node "
            f"({rebuilt_id[:12]}… vs {str(node.get('id'))[:12]}…)"
        )

    for field, stored in (
        ("rejected_paths", node.get("rejected_paths")),
        ("invalidates_if", node.get("invalidates_if")),
        ("context", node.get("context")),
    ):
        parsed = getattr(entry, field, None)
        if (parsed or "").strip() != (stored or "").strip():
            raise RestoreError(
                f"'{slug}': commentary field '{field}' did not survive the round trip"
            )
    if sorted(entry.scope or []) != sorted(node.get("scope") or []):
        raise RestoreError(f"'{slug}': scope did not survive the round trip")


def verify_whole_buffer(before_text: str, after_text: str, added: int) -> None:
    """Asserts the spliced buffer gained exactly ``added`` entries and disturbed none.

    Isolation cannot prove neighbour safety: continuation-line bleed crosses entry
    boundaries, which the retired `mark` mode demonstrated on live data — one stray
    line shifted the canonical core of the entry ABOVE it. One extra parse closes that
    seam class permanently, and it is cheap next to a corrupted gold source.

    Args:
        before_text: The buffer as it was.
        after_text: The buffer as it would be written.
        added: How many entries the splice is expected to add.

    Raises:
        RestoreError: On a parse failure, a count mismatch, or any pre-existing entry
            whose fields changed.
    """
    from mitos.parser import parse_entry_stream

    before_failures: List[Any] = []
    after_failures: List[Any] = []
    before = parse_entry_stream(before_text, "decision", failures=before_failures)
    after = parse_entry_stream(after_text, "decision", failures=after_failures)

    if len(after_failures) > len(before_failures):
        raise RestoreError(
            f"the splice introduced a parse failure: {after_failures[-1]}"
        )
    if len(after) != len(before) + added:
        raise RestoreError(
            f"the splice changed the entry count to {len(after)}, expected "
            f"{len(before) + added} — it disturbed a neighbouring entry"
        )

    # Compared as MULTISETS, not slug-keyed dicts. A dict silently keeps only the last
    # entry per slug, so with two entries sharing a slug — a duplicate is exactly the
    # hand-pasted input this check exists to survive (P13) — a splice could rewrite the
    # FIRST one and pass. Every pre-existing fingerprint must still be present, with
    # its multiplicity.
    before_counts: Dict[str, int] = {}
    for entry in before:
        key = json.dumps(_entry_fingerprint(entry), sort_keys=True, default=str)
        before_counts[key] = before_counts.get(key, 0) + 1
    after_counts: Dict[str, int] = {}
    for entry in after:
        key = json.dumps(_entry_fingerprint(entry), sort_keys=True, default=str)
        after_counts[key] = after_counts.get(key, 0) + 1

    for key, count in before_counts.items():
        if after_counts.get(key, 0) < count:
            slug = json.loads(key).get("slug")
            raise RestoreError(
                f"the splice altered or removed the pre-existing entry '{slug}' — "
                "continuation-line bleed across an entry boundary"
            )
