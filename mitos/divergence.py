"""Corpus↔graph divergence detection — a pure, read-only diff.

The markdown corpus is the gold source (P6/M7) and the graph is its derivative, but
nothing has ever *checked* that they still agree. They can drift apart silently: a
hand-edit to a committed entry's commentary is skipped by ``sync``'s idempotency
short-circuit, and a node whose ``### `` block has left the corpus is invisible until
``mitos rebuild`` refuses to reconstruct it.

**Naming.** "Drift" is taken — ``signals.signal_type='drifted'`` and ``is_drifted``
are the v0.2 code-drift sensor's channel, and M3 says explicitly *"Drift between the
graph and surrounding code is a separate matter."* This module's subject is
**corpus↔graph divergence**; the two must never blur on a shared surface.

Five species are reported:

======  ======================================================================
``S1``  Commentary **text** — ``slug``, ``rejected_paths``, ``invalidates_if``,
        ``context``. Exactly the mutable half of the store's commentary
        ``UPDATE SET``, minus the confirmation pair (see below).
``S2``  **Scope** — a *retrieval* defect rather than a reading one: a wrong value
        makes the decision miss every scope-filtered read and ``mitos scopes``.
``S3``  **Absent source block** — a node with no ``### `` entry anywhere in the
        corpus. ``rebuild`` drops it, the completeness gate refuses, and the
        tool's own repair story stops working.
``S5``  **Edge** — declared in the markdown vs stored in the graph, additions and
        deletions reported separately, because ``commit_parsed_entry`` mirrors
        edges *declaratively*: a removed line DELETES that edge.
``S6``  **Source** — ``**Source:**`` is a markdown field AND is mutation-fenced
        graph-side (MI-4), which makes it invisible in a way no other field is: a
        hand-edit keeps the same node id (source is out-of-core), raises no S1
        row, and is never reconciled — then a rebuild replays the markdown value
        and silently flips stored provenance.
======  ======================================================================

``S4`` (unreconstructable citations) is deliberately **not** recomputed here — it is
a rebuild-time projection, and ``mitos rebuild --json`` is its authority.

Two things are excluded on purpose, and both would look like bugs to a reader who
did not know:

* **``confirmed_by`` / ``confirmed_at``.** The obvious implementation reuses the
  store's own ``commentary`` dict, which includes them — and yields a false positive
  on every confirmed node (114 of them on the dogfood corpus), because the markdown
  has no field to compare against. Their primary source **is** the graph, so there is
  nothing to diff. Permanent.
* **Transcripts.** ``transcripts`` is a separate table with no reconcile path yet;
  a transcript-only edit is not merely invisible but never reconciled. Stated rather
  than silently omitted, and homed in the future ``amend-commentary`` verb.

The module has two halves, and the split is the point:

* ``entry_divergence`` and its helpers are a **pure leaf** — dict in, dict out, no
  I/O. It is shaped as the unit both surfaces call, so that ``status`` and ``sync``
  cannot grow two comparators that drift apart. Today the fold below is its only
  caller: ``sync``'s reconcile is a later release, and this docstring will not claim
  the seam is live before it is.
* ``corpus_graph_divergence`` is the whole-corpus **fold** — it owns the read, the
  advisory lock, and the sidecar cache, and it is ``status``'s half alone.

Importing this module stays cheap and cycle-free either way: every ``mitos`` import
the fold needs is function-local, so ``sync`` can import the leaf without dragging in
the parser (which reads ``format-spec.md`` from package data at import time), the
store, or the cutover replay machinery. A test pins that import graph.

**Open questions are out of scope, on both sides.** ``questions.md`` is not read, and
open-question nodes are filtered out of the graph side to match — excluding them from
one side only would report every one as a source-block orphan whose named repair
cannot touch it.
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The S1 diff set, pinned. Exactly the mutable-commentary columns the store's
# `UPDATE SET` can reach, minus the confirmation pair (graph-primary, see the module
# docstring). Widening this without widening that `UPDATE SET` would report a
# divergence the reconcile cannot repair.
COMMENTARY_FIELDS: tuple = ("slug", "rejected_paths", "invalidates_if", "context")

# The nine relationship fields, in `parser._RELATIONSHIP_FIELDS` order. Duplicated as
# a literal rather than imported: this module is a leaf by design, and the parser
# pulls in `format-spec.md` at import time. A drift guard test pins the two together.
RELATIONSHIP_FIELDS: tuple = (
    "supersedes", "corrects", "amends", "narrows", "depends_on",
    "resolves", "contradicts", "derives_from", "cites",
)


def strip_citation(raw: str) -> str:
    """Strips one surrounding ``[ ... ]`` and whitespace from an edge citation.

    ``format-spec.md`` authors kill-edges as ``**Supersedes:** [slug]`` and the
    deterministic parser keeps the brackets, while the agentic write path supplies a
    bare slug. Both are the same declaration (MI-7), so a comparison that does not
    normalize reports every corpus-authored edge as diverged. Byte-identical to
    ``store._strip_citation``; kept here to preserve the leaf property, and pinned to
    it by a cross-check test.

    Args:
        raw: The relationship value as stored on a ``ParsedEntry``.

    Returns:
        The citation with one bracket layer and surrounding whitespace removed.
    """
    value = raw.strip()
    if len(value) >= 2 and value.startswith("[") and value.endswith("]"):
        value = value[1:-1].strip()
    return value


def _normalized_text(value: Optional[str]) -> str:
    """Normalizes an optional prose field for comparison.

    ``None`` and ``""`` are the same absence — the parser yields ``None`` for a field
    the author omitted while the graph stores ``NULL`` or ``''`` depending on the
    column, and treating those as different would report divergence on every entry
    that simply has no ``**Context:**``.

    Args:
        value: The field value from either side.

    Returns:
        The value stripped, or ``""`` when absent.
    """
    return (value or "").strip()


def _edge_key_set(pairs: Sequence[Dict[str, str]]) -> set:
    """Folds edge dicts to a comparable ``{(kind, target_casefold)}`` set.

    Args:
        pairs: Edge dicts carrying ``kind`` and ``target``.

    Returns:
        A set of normalized ``(kind, target)`` tuples.
    """
    return {
        (p["kind"], strip_citation(p["target"]).casefold())
        for p in pairs
        if strip_citation(p["target"])
    }


def declared_edges(entry: Any) -> List[Dict[str, str]]:
    """Reads the edges an entry DECLARES in the markdown.

    Args:
        entry: A ``ParsedEntry``.

    Returns:
        ``{"kind", "target"}`` dicts, one per declared citation, in field order.
    """
    out: List[Dict[str, str]] = []
    for field in RELATIONSHIP_FIELDS:
        for raw in (getattr(entry, field, None) or []):
            target = strip_citation(raw)
            if target:
                out.append({"kind": field, "target": target})
    return out


def entry_divergence(
    entry: Any,
    node: Dict[str, Any],
    stored_scopes: Sequence[str],
    stored_edges: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    """Diffs one parsed entry against its committed node.

    Pure and I/O-free, which is what makes it safe for ``sync``'s per-entry loop to
    call once the reconcile lands: sync iterates a locked snapshot, and a comparator
    that re-read the live file would compare against a different document than the one
    being synced. (Not yet wired — the fold below is currently the only caller.)

    The caller supplies ``stored_scopes`` and ``stored_edges`` rather than having
    them looked up here, which is what lets the whole-corpus surface batch every read
    into one query instead of paying a lookup per entry (P11).

    Args:
        entry: The ``ParsedEntry`` parsed from the corpus.
        node: The committed node's reader-facing dict.
        stored_scopes: The node's scope tags as stored.
        stored_edges: The node's outgoing edges as ``{"kind", "target"}`` dicts.

    Returns:
        A dict with ``commentary`` (list of diverged field names), ``scope``
        (``{"graph", "markdown"}`` or ``None``), ``edges``
        (``{"added", "removed"}`` or ``None``), and ``source``
        (``{"graph", "markdown"}`` or ``None``). Every value is falsy when the entry
        and its node agree, so ``any(result.values())`` is the "diverged" predicate.
    """
    commentary: List[str] = []

    # `slug` is compared casefold — MI-13's identity is the casefolded slug, so a
    # case-only difference is not divergence the reconcile could meaningfully repair.
    if _normalized_text(entry.slug).casefold() != _normalized_text(node.get("slug")).casefold():
        commentary.append("slug")
    if _normalized_text(entry.rejected_paths) != _normalized_text(node.get("rejected_paths")):
        commentary.append("rejected_paths")
    if _normalized_text(entry.invalidates_if) != _normalized_text(node.get("invalidates_if")):
        commentary.append("invalidates_if")
    if _normalized_text(entry.context) != _normalized_text(node.get("context")):
        commentary.append("context")

    markdown_scopes = sorted({s.strip() for s in (entry.scope or []) if s.strip()})
    graph_scopes = sorted({s.strip() for s in (stored_scopes or []) if s.strip()})
    scope = (
        None if markdown_scopes == graph_scopes
        else {"graph": graph_scopes, "markdown": markdown_scopes}
    )

    declared = _edge_key_set(declared_edges(entry))
    stored = _edge_key_set(list(stored_edges or []))
    added = sorted(f"{kind}:{target}" for kind, target in declared - stored)
    removed = sorted(f"{kind}:{target}" for kind, target in stored - declared)
    edges = {"added": added, "removed": removed} if (added or removed) else None

    # S6. `Source` is tool-only for authors (`format-spec.md`), so the GRAPH's stamped
    # value is the authority and the repair direction is markdown-conforms-to-graph —
    # the opposite of every other species here. Absent in markdown means `user`, which
    # is why the comparison defaults rather than treating absence as a difference.
    markdown_source = _normalized_text(entry.source) or "user"
    graph_source = _normalized_text(node.get("source")) or "user"
    source = (
        None if markdown_source == graph_source
        else {"graph": graph_source, "markdown": markdown_source}
    )

    return {
        "commentary": commentary,
        "scope": scope,
        "edges": edges,
        "source": source,
    }


def has_divergence(report: Dict[str, Any]) -> bool:
    """Returns True when an ``entry_divergence`` report carries any species.

    Args:
        report: An ``entry_divergence`` result.

    Returns:
        True if anything diverged.
    """
    return bool(
        report.get("commentary")
        or report.get("scope")
        or report.get("edges")
        or report.get("source")
    )


def is_reconcilable(report: Dict[str, Any]) -> bool:
    """Returns True when ``sync``'s reconcile could repair this divergence.

    ``source`` is excluded deliberately: it is mutation-fenced graph-side (MI-4), so
    re-committing the entry cannot change it, and reporting it as reconcilable would
    promise a repair that provably does nothing.

    Args:
        report: An ``entry_divergence`` result.

    Returns:
        True if the reconcile path would change something.
    """
    return bool(report.get("commentary") or report.get("scope") or report.get("edges"))


# ---------------------------------------------------------------------------
# The whole-corpus fold
#
# Everything below owns I/O — the corpus read, the advisory lock, and the sidecar
# cache — and is the `mitos status` half. Its `mitos` imports are deliberately
# FUNCTION-LOCAL: importing `mitos.divergence` must not drag in the parser (which
# reads `format-spec.md` from package data at import time), the store, or the cutover
# replay machinery. `sync`'s per-entry loop imports this module for `entry_divergence`
# alone and must not pay for any of that; the import-graph property is pinned by test.
#
# When sync's reconcile lands it must call `entry_divergence` and never this: calling
# the fold inside sync's per-entry loop would re-read the LIVE corpus against the
# locked snapshot sync is iterating, and re-acquire the lock mid-pass — inconsistent
# by construction.
# ---------------------------------------------------------------------------

# The advisory-lock acquire budget, seconds. Deliberately NOT the sync manager's
# shared 60s lock: `status` runs at agent session start, concurrently with any sync,
# and a session-start command that can block for a minute behind a running sync is a
# worse failure than not answering. On timeout we report "skipped" and cache nothing —
# never a verdict, because a torn read cached is a sticky lie.
_LOCK_TIMEOUT_SECONDS: float = 1.0

_CACHE_BASENAME: str = "divergence_cache.json"

# Bumped whenever the report SHAPE changes. Without it a cache written by a build with
# a different species set is served verbatim to a reader that indexes the keys it
# expects — a `KeyError` out of `mitos status`, from a stale file, for a command whose
# entire job is to be the thing that still works.
_CACHE_VERSION: str = "1"


def _empty_report(**overrides: Any) -> Dict[str, Any]:
    """Builds the zero report, so every early return has the same shape."""
    report: Dict[str, Any] = {
        "checked": 0,
        "rotation_mode": None,
        "cache_hit": False,
        "skipped": None,
        "commentary": [],
        "scope": [],
        "graph_only": [],
        "edges": [],
        "source": [],
        "reconcilable": 0,
        "archived_drift": 0,
    }
    report.update(overrides)
    return report


def _corpus_files(config: Any) -> List[str]:
    """Lists the corpus files whose bytes the cache key hashes, oldest archive first.

    Args:
        config: The workspace config.

    Returns:
        Existing archive paths followed by the buffer, absent files omitted.
    """
    paths: List[str] = []
    # The SAME helper the replay uses, so the cache key can never cover a file the
    # diff ignores (or miss one it reads) — a stray `notes.md` in the archive dir
    # would otherwise invalidate the cache on every edit while never being parsed.
    from mitos.cutover import _archive_files_oldest_first

    archive_dir = getattr(config, "archive_dir", None)
    if archive_dir:
        paths.extend(_archive_files_oldest_first(archive_dir))
    buffer_path = getattr(config, "decisions_file", None)
    if buffer_path and os.path.exists(buffer_path):
        paths.append(buffer_path)
    return paths


def _corpus_hash(paths: Sequence[str]) -> str:
    """Hashes the corpus bytes, cheaply, without parsing.

    Hashing raw bytes rather than parsed entries is the point: the cache exists to
    skip the parse, so the key cannot depend on it.

    Args:
        paths: Corpus file paths in a stable order.

    Returns:
        A hex digest over each file's path and contents.
    """
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(os.path.basename(path).encode("utf-8"))
        try:
            with open(path, "rb") as fh:
                hasher.update(fh.read())
        except OSError:
            hasher.update(b"<unreadable>")
    return hasher.hexdigest()


def _read_cache(cache_path: str, key: str) -> Optional[Dict[str, Any]]:
    """Returns the cached report when its key matches, else ``None``.

    Any fault — missing, unreadable, corrupt, wrong shape — is a cache MISS, never an
    error and never a fabricated verdict.

    Args:
        cache_path: The sidecar file.
        key: The composite corpus+graph key the entry must carry.

    Returns:
        The cached report dict, or ``None``.
    """
    try:
        with open(cache_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("key") == key and isinstance(payload.get("report"), dict):
            return payload["report"]
    except (OSError, ValueError, TypeError):
        pass
    return None


def _write_cache(cache_path: str, key: str, report: Dict[str, Any]) -> None:
    """Writes the sidecar cache, best-effort.

    The cache lives in `.mitos/`, NOT the graph: `status` opens the graph read-only,
    and MI-11's lint bans any graph-side writer outside parse→commit. A read-only
    filesystem must degrade to "no cache", never crash a read-only command.

    Args:
        cache_path: The sidecar file.
        key: The composite corpus+graph key.
        report: The report to store.
    """
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump({"key": key, "report": report}, fh)
    except OSError:
        pass


def corpus_graph_divergence(store: Any, config: Any) -> Dict[str, Any]:
    """Diffs the WHOLE corpus (buffer + archives) against the graph, read-only.

    Reads through `cutover._load_decision_stream` so the detector and `mitos rebuild`
    can never disagree about what "the corpus" is — `perform_sync` reads only the
    buffer, which is exactly why a rotated corpus is invisible to it.

    Scale (P11): `status` did zero corpus parsing before this, so the cost is real —
    O(corpus parse + diff), measured at ~57ms over 209 entries and linear from there,
    roughly 13s at 50,000. Node and edge reads are batched into two queries rather
    than one lookup per entry, and the sidecar cache skips parse+diff entirely when
    neither the corpus bytes nor the graph fingerprint have moved.

    Args:
        store: An open (read-only is fine) ``GraphStore``.
        config: The workspace config.

    Returns:
        A report dict. ``skipped`` is a human-readable reason string when no verdict
        could be reached — a fresh workspace, an absent graph, or a busy corpus — and
        every count is zero in that case. A skipped run NEVER writes the cache.
    """
    from mitos.cutover import _load_decision_stream
    from mitos.identity import compute_node_id

    rotation_mode = getattr(config, "rotation_mode", None)

    corpus_paths = _corpus_files(config)
    if not corpus_paths:
        # A just-initialized project is healthy, not broken — and must not parse a
        # corpus that isn't there.
        return _empty_report(rotation_mode=rotation_mode, skipped="no corpus")

    db_path = getattr(config, "db_path", None)
    if not db_path or not os.path.exists(db_path):
        return _empty_report(rotation_mode=rotation_mode, skipped="no graph")

    cache_path = os.path.join(getattr(config, "mitos_dir", "."), _CACHE_BASENAME)

    # A FRESH lock object, not the sync manager's shared 60s one — see the constant.
    from filelock import FileLock, Timeout

    lock = FileLock(getattr(config, "decisions_file") + ".lock",
                    timeout=_LOCK_TIMEOUT_SECONDS)
    try:
        with lock:
            corpus_key = _corpus_hash(corpus_paths)
            node_count, max_updated = store.graph_fingerprint()
            key = f"{_CACHE_VERSION}:{corpus_key}:{node_count}:{max_updated}"

            cached = _read_cache(cache_path, key)
            if cached is not None:
                cached = dict(cached)
                cached["cache_hit"] = True
                # NOT cached: `rotation_mode` is a live config value, not a property of
                # the corpus/graph pair the key describes, so a cache hit must not
                # serve yesterday's mode.
                cached["rotation_mode"] = rotation_mode
                return cached

            # EVERY read — corpus and graph alike — happens inside this one lock.
            # Reading the graph after releasing it would diff a corpus snapshot against
            # a graph another process had since written: a node committed in that window
            # reads as a phantom graph-only orphan, and the sidecar would store that
            # false verdict under the PRE-race fingerprint — so returning the graph to
            # that state later (a `.bak` restore, exactly what the fingerprint exists to
            # catch) replays the lie as a cache hit. A torn read cached is a sticky lie
            # whichever side is torn.
            failures: List[Any] = []
            entries = _load_decision_stream(config, failures)
            # Decisions only, on BOTH sides. `get_all_nodes` returns every kind, so
            # folding it against the decision stream alone made each open-question node
            # a phantom `graph_only` orphan — pointing the reader at
            # `restore-source`, which can never repair one (an open question has no
            # axiom, so the block cannot be rendered). Open questions are deliberately
            # out of scope for this detector; excluding them from one side only is
            # strictly worse than excluding them from both.
            all_nodes = [n for n in store.get_all_nodes() if n.get("kind") == "decision"]
            all_edges = store.get_edges()
    except Timeout:
        # Another process holds the corpus lock. Report the skip, never a verdict:
        # a read that catches the rotation window sees a truncated buffer and would
        # report mass phantom graph-only nodes — and cached, that lie would stick
        # until the next real edit.
        return _empty_report(rotation_mode=rotation_mode, skipped="corpus busy")

    nodes_by_id = {n["id"]: n for n in all_nodes}
    slug_by_id = {nid: n.get("slug") for nid, n in nodes_by_id.items()}
    edges_by_source: Dict[str, List[Dict[str, str]]] = {}
    for edge in all_edges:
        target_slug = slug_by_id.get(edge["target_id"])
        if target_slug is None:
            continue
        edges_by_source.setdefault(edge["source_id"], []).append(
            {"kind": edge["edge_type"], "target": target_slug}
        )

    buffer_path = getattr(config, "decisions_file", "")
    report = _empty_report(rotation_mode=rotation_mode)

    # Buffer-wins on a duplicate canonical core. After a `restore-source` or a manual
    # re-add the same id can sit in BOTH an archive and the buffer; the stream is
    # oldest-first (archives, then buffer), so a plain last-wins fold matches
    # rebuild's own replay order. Without this rule the detector reports permanent
    # phantom `archived_drift` on every restored entry.
    seen_ids: Dict[str, Any] = {}
    for entry in entries:
        node_id = compute_node_id(
            kind=entry.kind,
            axiom=entry.axiom,
            mechanism_refs=entry.mechanisms,
            topic=entry.topic,
            questions_raised=entry.questions_raised,
        )
        seen_ids[node_id] = entry

    for node_id, entry in seen_ids.items():
        node = nodes_by_id.get(node_id)
        if node is None:
            # In the corpus but not the graph: an unsynced entry, which is the normal
            # pending state, not divergence. `sync` is its verb. Deliberately NOT
            # counted in `checked` — nothing was compared for it, and a count that
            # includes uncompared entries reads as coverage it does not have.
            continue
        report["checked"] += 1

        source_file = getattr(entry, "source_path", None) or buffer_path
        archived = os.path.abspath(source_file) != os.path.abspath(buffer_path)

        per_entry = entry_divergence(
            entry, node, node.get("scope") or [], edges_by_source.get(node_id, [])
        )
        if not has_divergence(per_entry):
            continue

        label = {"slug": node.get("slug"), "file": os.path.basename(source_file)}
        if per_entry["commentary"]:
            report["commentary"].append({**label, "fields": per_entry["commentary"]})
        if per_entry["scope"]:
            report["scope"].append({**label, **per_entry["scope"]})
        if per_entry["edges"]:
            report["edges"].append({**label, **per_entry["edges"]})
        if per_entry["source"]:
            report["source"].append({**label, **per_entry["source"]})

        if archived:
            # An archived entry's divergence is REPORTED, never reconciled: `sync`
            # reads the buffer only, and reconciling archives would turn a settled
            # partition into a live authoring surface. Its reconciler is `rebuild`.
            report["archived_drift"] += 1
        elif is_reconcilable(per_entry):
            report["reconcilable"] += 1

    # S3 — a node with no `### ` block anywhere in the corpus. `rebuild` drops it and
    # the completeness gate refuses, so the tool's own repair story stops working.
    # The repair is to re-materialize the block (`mitos restore-source`), never to let
    # the graph drop the node: the store never acts on whole-entry buffer-absence.
    for node_id, node in nodes_by_id.items():
        if node_id not in seen_ids:
            report["graph_only"].append({
                "slug": node.get("slug"),
                "active": node.get("computed_state") == "active",
            })
    report["graph_only"].sort(key=lambda row: (not row["active"], row["slug"] or ""))

    _write_cache(cache_path, key, report)
    return report


def divergence_total(report: Dict[str, Any]) -> int:
    """Counts every reported row across species.

    Args:
        report: A ``corpus_graph_divergence`` result.

    Returns:
        The number of diverged rows; ``0`` on a clean or skipped corpus.
    """
    return sum(
        len(report.get(species) or [])
        for species in ("commentary", "scope", "graph_only", "edges", "source")
    )
