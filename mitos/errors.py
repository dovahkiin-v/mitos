"""Custom exception definitions for Mitos.

This module contains the hierarchical exception architecture used across
the Mitos codebase to ensure precise error vectors and graceful degradation.

It also holds the §5.2.2 structured failure envelope (``FailureItem`` /
``EntryFailure``) — the C1-neutral surface both the parser (4b) and the store
(5b) emit. The structs live here, in the import-free leaf, so neither stage has
to import the other (that would reverse the C1 direction).
"""

from typing import Any, Dict, List, Optional, Union


class MitosError(Exception):
    """Base exception class for all Mitos errors."""
    pass


class ParseError(MitosError):
    """Raised when the deterministic markdown parser encounters malformed input.

    Attributes:
        line_start: The line number where the offending entry starts.
        line_end: The line number where the offending entry ends.
        message: Detailed explanation of the parsing failure.
        failure: The structured §5.2.2 envelope for the offending entry, when
            the error was raised in strict mode by ``parse_entry_stream``. A
            ``None`` value preserves the prototype path's plain ``ParseError``.
    """

    def __init__(
        self,
        message: str,
        line_start: int = 1,
        line_end: int = 1,
        failure: Optional["EntryFailure"] = None,
    ) -> None:
        self.line_start = line_start
        self.line_end = line_end
        self.message = message
        self.failure = failure
        super().__init__(f"Lines {line_start}-{line_end}: {message}")


class ValidationError(MitosError):
    """Raised when a node or relationship violates architectural invariants."""
    pass


class DatabaseError(MitosError):
    """Raised when SQLite database operations fail or encounter lock contention."""
    pass


class ConfigError(MitosError):
    """Raised when `.mitos/config.toml` is malformed or carries an invalid value.

    The strict config loader (``MitosConfig._load_config_file``) refuses to swallow
    a broken config: malformed TOML, a known key with the wrong type, or an
    out-of-enum ``rotation_mode`` all raise this rather than silently falling back
    to defaults (the OD1-symmetric failure-mode policy, §5.2.6). A ``MitosError``
    subclass so the CLI's ``except MitosError`` boundary renders it as a one-line
    ``Error: …`` message instead of a raw traceback. The message names the file
    path and the located cause (the ``tomllib`` line/column, or the offending key +
    expected type).
    """
    pass


class RegistryError(MitosError):
    """Raised when the global project registry is unusable or a registration is refused.

    The registry (``<config-home>/mitos/registry.toml``) is the machine-local
    ``name → workspace path`` routing map ``mitos init`` writes and every
    name-targeted command reads. This covers both classes of fault in it: the
    **file** being unusable (unparseable TOML, a value that is not a path string,
    an unwritable config dir) and a **registration attempt** being refused (an
    illegal name, a name already registered at another path, a path already
    registered under another name).

    A ``MitosError`` subclass so the CLI's ``except MitosError`` boundary renders
    it as a one-line ``Error: …`` message instead of a raw traceback. Every
    message names the resolved registry path (never the literal ``~/.config/…``)
    plus the located cause, **and a recovery** — ``--name``/``--force``, or the
    hand-edit the file invites (P3: a vector, not a wall).

    Deliberately distinct from the *targeting* errors a later phase's resolver
    raises when a selector cannot be resolved: those are faults of a lookup, these
    are faults of the file and of a write to it.
    """
    pass


# ---------------------------------------------------------------------------
# Project-targeting discriminators.
#
# The NAMES are the cross-surface contract: the CLI boundary renderer and the MCP
# boundary renderer both switch on these exact strings to choose their wording,
# so a typo is a silent break on one surface only. Six classes, none falling
# through to another, because a wrong class teaches the wrong lesson — answering
# a relative path with "unknown project — did you mean 'mitos'?" invites a retry
# with another relative spelling, and answering an uninitialized directory the
# same way says the *name* is wrong when the *directory* is the problem.
# ---------------------------------------------------------------------------

TARGET_MISSING = "missing"
TARGET_UNKNOWN_NAME = "unknown_name"
TARGET_RELATIVE_PATH = "relative_path"
TARGET_PATH_NOT_A_WORKSPACE = "path_not_a_workspace"
TARGET_REGISTERED_UNREACHABLE = "registered_unreachable"
TARGET_EXEMPT_VERB = "exempt_verb"

TARGETING_DISCRIMINATORS = frozenset(
    {
        TARGET_MISSING,
        TARGET_UNKNOWN_NAME,
        TARGET_RELATIVE_PATH,
        TARGET_PATH_NOT_A_WORKSPACE,
        TARGET_REGISTERED_UNREACHABLE,
        TARGET_EXEMPT_VERB,
    }
)

# Why a verb takes no project selector — a key, never prose, so both surfaces
# word it themselves. The verb→reason mapping belongs to whichever surface owns
# its verb table, not here.
EXEMPT_CREATES_REGISTRATION = "creates_registration"   # `init` — it creates one
EXEMPT_NO_WORKSPACE = "no_workspace"                   # `serve` — it targets none
EXEMPT_EXPLICITLY_GLOBAL = "explicitly_global"         # `projects`, the zero-arg
                                                       # overview, a global key set

EXEMPT_REASONS = frozenset(
    {
        EXEMPT_CREATES_REGISTRATION,
        EXEMPT_NO_WORKSPACE,
        EXEMPT_EXPLICITLY_GLOBAL,
    }
)

# What each class must carry to be renderable. A discriminator whose required
# data is absent is a half-filled error, which the constructor refuses.
_TARGETING_REQUIRED_FIELDS: Dict[str, tuple] = {
    TARGET_MISSING: (),
    TARGET_UNKNOWN_NAME: ("selector",),
    TARGET_RELATIVE_PATH: ("selector",),
    TARGET_PATH_NOT_A_WORKSPACE: ("selector", "path"),
    TARGET_REGISTERED_UNREACHABLE: ("selector", "name", "path"),
    TARGET_EXEMPT_VERB: ("verb", "exempt_reason"),
}


class ProjectTargetingError(MitosError):
    """Raised when a project selector cannot be resolved to a workspace.

    The routing resolver's failure type (``mitos.routing.resolve_project``). It
    carries **structured data only** — a discriminator, the registered-name
    vocabulary, close matches, the paths involved — and no finished presentation
    string: each surface's boundary renderer composes its own example and
    discovery pointer from these fields, so no CLI wording reaches an MCP caller
    and no MCP call-form reaches a terminal.

    A ``MitosError`` subclass so the CLI's ``except MitosError`` boundary renders
    it as one calm ``Error: …`` line even on a path no renderer has reached yet.
    Deliberately distinct from :class:`RegistryError`: that is a fault of the
    registry *file*, this is a fault of a *lookup* through it.

    ``str(err)`` is a **terse fallback for an unrendered path** — a
    discriminator-level statement and nothing more. It is not the teaching-error
    anatomy, it names no CLI flag and no MCP call form, and every value it echoes
    goes through ``repr`` (a selector is untrusted text and can carry a newline or
    an ANSI escape).

    The constructor validates, raising ``ValueError`` — a programming fault in
    mitos, not a user fault, so deliberately **not** a ``MitosError`` — when the
    discriminator is unknown, a required field for it is absent, or an
    ``exempt_reason`` is not one of ``EXEMPT_REASONS``. It exists so a boundary
    renderer never meets a half-filled error and never has to defend against one.

    Attributes:
        discriminator: One of ``TARGETING_DISCRIMINATORS`` — the failure class.
            Renderers switch on this; no class ever falls through to another,
            because a wrong class teaches the wrong lesson.
        selector: The selector as the caller supplied it.
        path: The path involved — the probed workspace root for
            ``path_not_a_workspace``, the registry's **recorded** path (the value
            a repoint edits) for ``registered_unreachable``.
        name: The registered name, for ``registered_unreachable``.
        registered_names: Every registered name, in document order. The full
            list; bounding it for display is the renderer's policy
            (``routing.bounded_registered_names``).
        close_matches: Did-you-mean suggestions, non-empty only where a *name*
            was claimed — did-you-mean over registered names is noise for a path.
        cwd_hint_name: The nearest registered ancestor of the caller's working
            directory. Always ``None`` here: the resolver has no cwd branch (the
            invariant's structural leg), and the boundary fills this field.
        verb: The global verb that was given a selector, for ``exempt_verb``.
        exempt_reason: Why that verb takes none — a key from ``EXEMPT_REASONS``,
            never prose.
    """

    def __init__(
        self,
        discriminator: str,
        *,
        selector: Optional[str] = None,
        path: Optional[str] = None,
        name: Optional[str] = None,
        registered_names: Optional[List[str]] = None,
        close_matches: Optional[List[str]] = None,
        cwd_hint_name: Optional[str] = None,
        verb: Optional[str] = None,
        exempt_reason: Optional[str] = None,
    ) -> None:
        if discriminator not in TARGETING_DISCRIMINATORS:
            raise ValueError(
                f"unknown targeting discriminator {discriminator!r}; expected one "
                f"of {sorted(TARGETING_DISCRIMINATORS)}"
            )
        self.discriminator = discriminator
        self.selector = selector
        self.path = path
        self.name = name
        # Copied, and spelled without a truthiness coalesce: `x or []` would fold
        # a deliberately-empty list into the default, which is the shape that
        # quietly undoes a guard one line below itself elsewhere in this tree.
        self.registered_names: List[str] = (
            [] if registered_names is None else list(registered_names)
        )
        self.close_matches: List[str] = (
            [] if close_matches is None else list(close_matches)
        )
        self.cwd_hint_name = cwd_hint_name
        self.verb = verb
        self.exempt_reason = exempt_reason

        for field in _TARGETING_REQUIRED_FIELDS[discriminator]:
            if getattr(self, field) is None:
                raise ValueError(
                    f"targeting discriminator {discriminator!r} requires {field!r}"
                )
        if exempt_reason is not None and exempt_reason not in EXEMPT_REASONS:
            raise ValueError(
                f"unknown exempt reason {exempt_reason!r}; expected one of "
                f"{sorted(EXEMPT_REASONS)}"
            )
        super().__init__(self._fallback_message())

    def _fallback_message(self) -> str:
        """Builds the terse discriminator-level fallback (see the class docstring)."""
        if self.discriminator == TARGET_MISSING:
            return "no project selector was supplied"
        if self.discriminator == TARGET_UNKNOWN_NAME:
            return f"unknown project {self.selector!r}"
        if self.discriminator == TARGET_RELATIVE_PATH:
            return (
                f"the project selector {self.selector!r} is a relative path; a "
                f"registered name or an absolute path is required"
            )
        if self.discriminator == TARGET_PATH_NOT_A_WORKSPACE:
            return f"no Mitos workspace at {self.path!r}"
        if self.discriminator == TARGET_REGISTERED_UNREACHABLE:
            return (
                f"the project {self.name!r} is registered at {self.path!r}, which "
                f"no longer holds a Mitos workspace"
            )
        # TARGET_EXEMPT_VERB — the discriminator was whitelisted above, so this is
        # the remaining class rather than a fall-through default.
        return f"the {self.verb!r} verb takes no project selector"


class VectorStoreError(MitosError):
    """Raised when Qdrant vector store operations fail."""
    pass


class CollectionMissingError(VectorStoreError):
    """Qdrant answered: I am up, that collection does not exist (HTTP 404).

    The typed distinction between *missing* and *unreachable*. Absence is a
    recoverable state with a one-command heal (``mitos reconcile``); an
    unreachable Qdrant is an infrastructure fault. Reporting the first as the
    second sends the operator to fix a service that is running.

    A :class:`VectorStoreError` **subclass** on purpose: every shipped
    ``except VectorStoreError`` net keeps catching it unchanged (``mitos status``'
    scroll guard, ``cmd_reconcile``'s boundary, ``gather_candidates``' fail-open),
    so nothing silently changes class and each consumer opts into the narrower
    catch deliberately.

    Attributes:
        collection: The absent collection's name, so a surface can name it
            without re-deriving it from config or scraping the message.
    """

    def __init__(self, message: str, collection: Optional[str] = None) -> None:
        super().__init__(message)
        self.collection = collection


class VectorStoreUnreachableError(VectorStoreError):
    """Qdrant did not answer at all — the transport failed before any status code.

    :class:`CollectionMissingError`'s sibling from the other direction, and the
    reason it lives here beside it rather than in the module that raises it: that
    class is documented as *"the typed distinction between missing and
    unreachable"*, and the unreachable half was, until now, indistinguishable from
    every other fault. A refused connection, a DNS failure and a socket timeout say
    **nothing about the service's state**; a 500, an auth wall or a malformed body
    say the service is up and answering something unusable. Both raise
    ``VectorStoreError`` today, so a caller that must report the two apart — the
    ``mitos status`` overview's per-instance tri-state — could otherwise only sniff
    the message text.

    A :class:`VectorStoreError` **subclass**, on the same reasoning as its sibling:
    every shipped ``except VectorStoreError`` net keeps catching it unchanged, and a
    consumer opts into the narrower catch deliberately.

    Raised today only by :func:`mitos.vector_store.list_collection_names`.
    ``scroll_point_ids``' identical transport arm is deliberately left alone — its
    consumers collapse the two states and narrowing that error's class is a change
    to a shipped read path, not this one's work.
    """


class EmbeddingError(MitosError):
    """Raised when embedding provider API calls fail."""
    pass


class SynthesisError(MitosError):
    """Raised when the LLM synthesis or enrichment call fails."""
    pass


class CommitError(MitosError):
    """Raised when the store rejects an entry on referential/graph grounds (§5.2.2).

    The store-stage analogue of :class:`ParseError`: a referential violation found
    while committing (a missing/dangling kill-edge target, a cross-kind edge, a
    cycle, or a slug collision) raises this carrying the structured ``EntryFailure``
    envelope (``source="store"`` items), so the caller gets the same
    machine-readable payload the parser produces.

    It is deliberately a :class:`MitosError` and **not** a ``sqlite3.Error``
    subclass: raised inside ``commit_parsed_entry``'s ``with conn:`` block it rolls
    the whole entry back (V1-D10), then propagates *past* the SQLite exception
    handlers with its envelope intact.

    Attributes:
        failure: The structured §5.2.2 envelope for the rejected entry.
    """

    def __init__(self, message: str, failure: "EntryFailure") -> None:
        self.failure = failure
        super().__init__(message)


class CutoverError(MitosError):
    """Raised when the V1a cutover rebuild hits a genuine corpus defect (§2.1, R11).

    The cutover (Phase 7a) re-parses the corpus and replays it oldest-first into a
    fresh *build-aside* graph, leaving the live graph untouched. A **corpus
    defect** — a parse-stage format failure, a missing kill-edge target, a
    Q5-convergence self-edge (``cycle_violation``), an empty canonical core
    reaching the store — aborts the rebuild and raises this, carrying the
    offending failure(s) so the Phase 7b CLI boundary can render one located line
    and the operator can fix the markdown and re-run.

    It is deliberately distinct from a **completeness shortfall** (an active
    reference core absent from the reconstruction). A shortfall is NOT raised: it
    is a verdict on :class:`~mitos.cutover.RebuildResult` the operator may override
    (P6) — the markdown is authoritative, so a drop may be intentional. Abort is a
    loud exception; shortfall is a returned verdict. Keeping the two on separate
    channels is the §2.1 contract.

    A :class:`MitosError` subclass so the CLI's ``except MitosError`` boundary
    renders it as a one-line ``Error: …`` message instead of a raw traceback.

    Attributes:
        failure: The offending :class:`EntryFailure` (a single replay/commit
            reject) or a ``list`` of them (the parse-stage aggregate, where every
            format defect is collected before aborting so the operator fixes them
            all at once), or ``None`` when the defect carried no structured
            envelope (a bypassed-parser ``ValidationError``/``DatabaseError``
            surfacing mid-replay).
    """

    def __init__(
        self,
        message: str,
        failure: Optional[Union["EntryFailure", List["EntryFailure"]]] = None,
    ) -> None:
        self.failure = failure
        super().__init__(message)


# ---------------------------------------------------------------------------
# §5.2.2 Structured Failure Envelope (Phase 4b)
#
# When parsing or committing an entry fails, V1a reports a structured payload
# anchored to the parsed slug (or ``None`` for a pre-header failure): a line
# range, the raw header bytes, the source path, and a list of structured
# ``FailureItem``s. Each item carries a stable ``code`` from a per-stage
# whitelist, a ``source`` discriminator ("parser" | "store" — the C1 boundary
# marker; mixing the lists is a C1 breach), a calm message, and optional
# field/line localization.
#
# These structs are the C1-NEUTRAL shared surface: ``parser.py`` (4b) emits
# ``source="parser"`` items; ``store.py`` (5b) REUSES the same structs for
# ``source="store"`` items. They live in this import-free leaf so neither stage
# imports the other.
#
# The failure-code NAMES are the cross-vision §5.2.2 contract: V3a's interactive
# review UX and V5's MCP error surface switch on these exact strings, so a typo
# is a silent cross-vision break (pinned by test).
# ---------------------------------------------------------------------------

# Parser-stage code names (emitted with ``source="parser"``).
PARSER_MALFORMED_ENTRY = "malformed_entry"
PARSER_MISSING_REQUIRED_FIELD = "missing_required_field"
PARSER_MALFORMED_MARKER = "malformed_marker"
PARSER_SLUG_TOO_LONG = "slug_too_long"

# The parser-stage whitelist. The store-stage codes (``slug_collision`` /
# ``missing_target`` / ``dangling_edge`` / ``kind_constraint_violation`` /
# ``cycle_violation``) are ADDED BY 5b as their own ``STORE_FAILURE_CODES`` set,
# emitted with ``source="store"`` — they are NOT parser codes and must never
# appear in a parser-produced envelope (the stage-purity invariant).
PARSER_FAILURE_CODES = frozenset(
    {
        PARSER_MALFORMED_ENTRY,
        PARSER_MISSING_REQUIRED_FIELD,
        PARSER_MALFORMED_MARKER,
        PARSER_SLUG_TOO_LONG,
    }
)

# Store-stage code names (emitted with ``source="store"``, Phase 5b) — the five
# referential codes reserved in the comment above. They are NOT parser codes and
# must never appear in a parser-produced envelope (the stage-purity invariant).
STORE_SLUG_COLLISION = "slug_collision"
STORE_MISSING_TARGET = "missing_target"
STORE_DANGLING_EDGE = "dangling_edge"
STORE_KIND_CONSTRAINT_VIOLATION = "kind_constraint_violation"
STORE_CYCLE_VIOLATION = "cycle_violation"

# The store-stage whitelist (mirrors ``PARSER_FAILURE_CODES``). The NAMES are the
# cross-vision §5.2.2 contract — V3a's interactive review UX and V5's MCP error
# surface switch on these exact strings, so a typo is a silent cross-vision break
# (pinned by ``test_store_failure_codes_pin``).
STORE_FAILURE_CODES = frozenset(
    {
        STORE_SLUG_COLLISION,
        STORE_MISSING_TARGET,
        STORE_DANGLING_EDGE,
        STORE_KIND_CONSTRAINT_VIOLATION,
        STORE_CYCLE_VIOLATION,
    }
)


class FailureItem:
    """A single format- or referential-level violation within one entry.

    One ``FailureItem`` is one thing that is wrong. Items accumulate within a
    stage — a decision missing both required fields yields two items — and the
    ``source`` discriminator records which stage produced the item (the C1
    boundary marker, §5.2.2). A parser-produced envelope contains only
    ``source="parser"`` items; the store appends ``source="store"`` items to its
    own envelopes. Mixing them in one envelope is a C1 breach.

    Attributes:
        code: A stable failure code from a per-stage whitelist
            (``PARSER_FAILURE_CODES`` for the parser stage).
        source: The producing stage — ``"parser"`` or ``"store"``.
        message: A calm, terse, screen-reader-clean explanation (P9).
        field: The offending field token (e.g. ``"**Decided:**"``), if any.
        line_start: 1-based start line of the violation, if localizable.
        line_end: 1-based end line of the violation, if localizable.
    """

    def __init__(
        self,
        code: str,
        source: str,
        message: str,
        field: Optional[str] = None,
        line_start: Optional[int] = None,
        line_end: Optional[int] = None,
    ) -> None:
        self.code = code
        self.source = source
        self.message = message
        self.field = field
        self.line_start = line_start
        self.line_end = line_end

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the item into a JSON-compatible dictionary.

        Returns:
            A dict with plain string/int/None values (JSON-roundtrip-safe).
        """
        return {
            "code": self.code,
            "source": self.source,
            "message": self.message,
            "field": self.field,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }


class EntryFailure:
    """The per-entry failure envelope (the §5.2.2 payload).

    Anchors a list of :class:`FailureItem` to one entry. ``slug`` is the parsed
    slug, or ``None`` when tokenization failed before a slug could be read (a
    pre-header failure) — in which case ``raw_header`` carries the raw header
    bytes so the offending entry stays identifiable. A malformed entry is
    isolated: it is reported and skipped, never aborting the whole batch
    (§5.2.2 per-entry isolation).

    Attributes:
        slug: The parsed slug, or ``None`` for a pre-header failure.
        line_start: 1-based start line of the entry's section span.
        line_end: 1-based end line of the entry's section span.
        source_path: The originating file path, if known.
        raw_header: The raw header line, for a pre-header (slug-less) failure.
        items: The accumulated failures for this entry (always at least one).
    """

    def __init__(
        self,
        slug: Optional[str],
        line_start: int,
        line_end: int,
        items: Optional[List["FailureItem"]] = None,
        source_path: Optional[str] = None,
        raw_header: Optional[str] = None,
    ) -> None:
        self.slug = slug
        self.line_start = line_start
        self.line_end = line_end
        self.items: List["FailureItem"] = items if items is not None else []
        self.source_path = source_path
        self.raw_header = raw_header

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the envelope into a JSON-compatible dictionary.

        Returns:
            A dict with plain values and an ``items`` list of item dicts
            (JSON-roundtrip-safe; lists, never tuples).
        """
        return {
            "slug": self.slug,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "source_path": self.source_path,
            "raw_header": self.raw_header,
            "items": [item.to_dict() for item in self.items],
        }
