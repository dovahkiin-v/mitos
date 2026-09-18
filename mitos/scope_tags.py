"""The scope-tag rule, in one place.

Three sites must agree on what "the same scopes" means: the parser at the C1 parse
boundary, ``GraphStore.commit_parsed_entry`` when it writes ``node_scopes``, and
``divergence.entry_divergence`` when it decides whether a hand-edit diverged. If the
comparator and the writer ever disagreed, sync would reconcile a "divergence" whose
commit writes nothing, and report it again on every run. So all three call this
function rather than restating it.

A Tier-1 leaf: stdlib only, and it imports nothing from ``mitos``. ``divergence``
imports it at module level, and that module's import probe forbids ``parser`` (which
reads ``format-spec.md`` at import time) and ``identity`` (the canonical-core home,
where a commentary-tier rule does not belong).
"""

from typing import Iterable, List, Optional


def normalize_scope_tags(items: Optional[Iterable[str]]) -> List[str]:
    """Normalizes raw scope tags: strip, drop empties, casefold, first-seen dedup.

    Scope is a cross-kind tag list. Each tag is stripped and casefolded (Python
    ``str.casefold`` — never SQLite ``NOCASE``/``LOWER``, MI-7/P9), empties are
    dropped, and duplicates are removed keeping the first occurrence, so no
    empty/NULL scope row can ever reach the store (MI-9). The author's order is
    kept: the first tag is the node's primary scope, persisted as the
    ``node_scopes`` ordinal. Scope has **no** ``identity.py`` counterpart (it is
    commentary, not hashed) — its byte-form is pinned by its own golden, not by
    the identity cross-check.

    Args:
        items: The raw scope tags (already comma-split); ``None`` reads as none.

    Returns:
        The casefolded, deduped scope list in authored order.
    """
    return list(dict.fromkeys(s.strip().casefold() for s in (items or []) if s.strip()))
