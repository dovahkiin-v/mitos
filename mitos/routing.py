"""Selector → workspace routing: the thing that reads the registry's map.

Every workspace-targeting call in mitos answers *"which project?"* by looking at
where the process happens to be standing. On a CLI that is a defensible guess; on
an always-on server it is not a guess at all — the launch directory is fixed for
the server's whole life, so a call meaning *project B* lands in whatever project
the server started in, and the resulting failure is not merely wrong but
**unnameable** (measured: every tool loads, and every call fails with "unable to
open database file").

:func:`resolve_project` replaces the guess with an answer. It admits exactly two
selector forms — a **registered name** or an **absolute path** (the escape hatch
for a valid-but-unregistered workspace: a fresh clone, a mid-setup project) — and
tells them apart by **shape**, never by trying one and falling through to the
other. The obvious build, "absolutize anything that is not already absolute",
turns the registered name ``mitos`` into ``<cwd>/mitos``, so the registry lookup
never fires and every test that passes an absolute path still goes green; and the
mirror-image build, "try the registry first, fall through to a path", resolves a
mistyped name to a same-named directory that happens to sit in cwd. Shape-first
is therefore a contract, not a convenience, and it is total because a registry
name may not carry the path syntax a path claims (``registry.validate_name``).

The other half of the module is the failure vocabulary. A resolver that returns
the right root on the happy path and strands its caller on the unhappy one has
moved the dead end rather than removed it — and the caller is often an agent that
cannot read a stack trace or retry thoughtfully. So every way resolution can fail
raises :class:`~mitos.errors.ProjectTargetingError` carrying **structured data
only**: the discriminator, the registered-name vocabulary, close matches, the
paths involved. Wording — the example, the discovery pointer, the recovery — is
each surface's, composed from that data, so no CLI syntax reaches an MCP caller
and no tool call-form reaches a terminal.

**Canonicalization happens once, here.** Both admitted forms reduce through
``registry.canonicalize`` (``os.path.realpath``) and nothing downstream
re-canonicalizes; a second spelling anywhere (``abspath``, ``normpath``, a
trailing-slash strip) splits identity silently.

Tier 1, permanently: stdlib plus ``mitos.registry`` and ``mitos.errors``, and
nothing else, ever. This leaf sits below ``cli``, ``store``, ``sync`` and their
neighbours — the CLI and MCP boundaries are what consume it — so an import back
from any of them would cycle, and the leaf must never construct a ``MitosConfig``
(see :func:`is_workspace`). ``mitos.config`` arrives transitively through
``registry``, which is tier-legal and is why nothing here imports it directly.
"""

import difflib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from mitos import registry
from mitos.errors import (
    TARGET_EXEMPT_VERB,
    TARGET_MISSING,
    TARGET_PATH_NOT_A_WORKSPACE,
    TARGET_REGISTERED_UNREACHABLE,
    TARGET_RELATIVE_PATH,
    TARGET_UNKNOWN_NAME,
    ProjectTargetingError,
)

# How many registered names a renderer may enumerate before the list collapses to
# the close matches plus a count. At or below it the full list is the better
# answer while it is short; above it, an enumeration is a wall in place of a
# diagnostic — on the MCP side it is input tokens charged to the calling agent's
# turn, on every targeting failure. Ten names is about one terminal line. The
# policy lives in this leaf rather than in either renderer because it is a rule
# both surfaces must agree on, and agreement is worth more as a fact of the call
# graph than as a convention (``recall.SURFACE_TOP_SCOPES`` sits here for the
# same reason).
REGISTERED_NAMES_BOUND: int = 10

# Did-you-mean tuning for *project names*. Deliberately its own constants rather
# than a share of ``recall.SURFACE_DIDYOUMEAN_CUTOFF``: project names and scope
# tags are different vocabularies, and tuning one must not move the other.
PROJECT_DIDYOUMEAN_CUTOFF: float = 0.6
PROJECT_DIDYOUMEAN_MAX: int = 3


@dataclass
class ResolvedProject:
    """One successful resolution — a runtime value, never persisted.

    A struct rather than a bare root because the resolver already knows the rest:
    it has the registry loaded, so ``name`` and ``via`` are free here and a second
    registry read at every consumer otherwise.

    Attributes:
        root: The canonical workspace root — ``os.path.realpath``, via
            ``registry.canonicalize``. The one canonical spelling; nothing
            downstream re-canonicalizes it.
        name: The registered name when one is known — the name the caller passed
            for a name-form selector, the reverse lookup for a path-form one, and
            ``None`` for an unregistered path, which is a correct steady state
            and draws no warning of any kind.
        via: ``"name"`` or ``"path"`` — how the selector was admitted. A plain
            ``str``, not an enum, on ``RegistrationOutcome``'s precedent: nothing
            serializes it, so an enum would buy a persistence contract no boundary
            needs.
    """

    root: str
    name: Optional[str]
    via: str


@dataclass
class BoundedNames:
    """What a renderer may enumerate of the registered-name vocabulary.

    Attributes:
        names: The names to render — the whole list, or the close matches when
            collapsed. Collapsing is deliberately **not** a "first 10 of 40"
            prefix: registry order is *document* order, which carries no relevance
            ranking, so a prefix of it is arbitrary and arbitrary is worse than a
            count.
        total: How many are registered, so a collapsed rendering stays truthful.
        collapsed: True when ``names`` is not the whole list. Each renderer adds
            the discovery pointer its surface owns.
    """

    names: List[str]
    total: int
    collapsed: bool


def is_path_shaped(selector: str) -> bool:
    """Reports whether a selector carries explicit path syntax.

    The discrimination rule: a separator **anywhere**, or a leading ``.`` or
    ``~``. Everything else is a name and reaches the registry untouched.

    This is the exact complement of ``registry.validate_name``'s legality, which
    is what makes shape-first discrimination total — a name may not contain a
    separator and may not begin with ``.`` or ``~``, so no string is ever both.
    "Separator anywhere" is wider than a leading-or-trailing reading on purpose:
    ``src/proj`` can never be a legal name, so under the narrower rule it would be
    neither a path nor a name and would land on the unknown-name anatomy —
    "did you mean 'mitos'?" for something obviously a path.

    Two strings are neither, and both are handled elsewhere rather than left
    implicit: the empty string (its own missing class) and a control-character
    string with no separator (a registry miss — an unknown name, echoed escaped).

    Args:
        selector: The selector as the caller supplied it.

    Returns:
        True if the selector is path-shaped.
    """
    separators = [os.sep] + ([os.altsep] if os.altsep else [])
    if any(sep in selector for sep in separators):
        return True
    return selector[:1] in (".", "~")


def is_workspace(root: str) -> bool:
    """Reports whether a directory holds a Mitos workspace.

    The shipped validity triple — ``.mitos/`` is a directory, it holds a
    ``config.toml``, and ``decisions.md`` sits beside it — re-spelled here rather
    than imported, because the only shipped spelling is three local variables
    inside ``mitos status``, and reaching them by constructing a ``MitosConfig``
    would make this leaf import a heavier module **and** raise ``ConfigError``
    from a validity probe: a workspace whose shape is perfectly valid but whose
    ``config.toml`` is malformed would become *unresolvable* instead of
    resolvable-and-then-diagnosed — the wrong failure at the wrong altitude, on a
    state ``status`` exists to report. The duplication is deliberate and netted by
    a test asserting the two verdicts agree file-by-file.

    It proves a **workspace**, not a built graph, and must not be strengthened: a
    clone that commits ``.mitos/`` but gitignores the SQLite file validates here
    and answers reads from an empty graph. That is what makes the escape hatch
    work on a fresh clone; the empty-graph state is named by the read surfaces and
    by ``status``, not by refusing to resolve.

    A permission fault reads as ``False`` rather than raising —
    ``os.path.isdir`` returns False on an unreadable parent — so an unreadable
    directory renders as "no workspace there". Accepted: the message names the
    path, so the operator sees the subject, and the alternative is a bespoke
    permission class no consumer asked for.

    Args:
        root: An absolute workspace root.

    Returns:
        True if all three parts of the triple are present.
    """
    mitos_dir = os.path.join(root, ".mitos")
    return (
        os.path.isdir(mitos_dir)
        and os.path.exists(os.path.join(mitos_dir, "config.toml"))
        and os.path.exists(os.path.join(root, "decisions.md"))
    )


def nearest_registered_ancestor(start_dir: str, reg: Dict[str, str]) -> Optional[str]:
    """Names the registered project a directory sits inside, if any.

    The cwd hint's predicate — and it takes the directory as an **argument**,
    never reading the process's working directory itself. (Spelled that way
    rather than naming the call, so the tree-wide grep for that call keeps
    returning only real reads — a prose hit in the one module that must never
    make one reads as a finding.) The cwd read lives exclusively in the
    error-rendering layer, so the resolution path structurally has no cwd branch
    while the hint still ships, and this stays a testable function needing no
    ``chdir``.

    Containment is by whole path segment: ``P`` is an ancestor of ``C`` when
    ``C == P`` or ``C`` starts with ``P`` plus a separator. A bare ``startswith``
    is the trap, and the shape it fails on is worth naming precisely, because the
    obvious example does not show it: with ``mitos`` and ``mitos-pub`` both
    registered, a prefix test from inside ``mitos-pub`` matches *both* entries —
    and longest-wins below then picks the right one anyway. What it gets wrong is
    a directory whose name merely **extends** a registration:
    ``…/mitos-pub-sandbox`` is not inside ``…/mitos-pub``, and a prefix test says
    it is. That shape is ordinary (a ``cp -r``, a second clone), and the answer is
    confidently wrong rather than absent. The ``rstrip`` is the one legitimate
    trailing-separator strip in this module: it is a *comparison* helper that
    never produces a stored or returned path, and it is what makes a workspace
    registered at ``/`` behave.

    Longest registered path wins; a tie (two names for one path — a tolerated
    hand-edit) resolves to the first in document order, ``reverse_lookup``'s rule.

    Args:
        start_dir: The directory to hint from, in any spelling.
        reg: An already-loaded registry.

    Returns:
        The nearest registered ancestor's name, or None.
    """
    canonical = registry.canonicalize(start_dir)
    best: Optional[str] = None
    best_length = -1
    for name, path in reg.items():
        if canonical == path or canonical.startswith(path.rstrip(os.sep) + os.sep):
            if len(path) > best_length:
                best, best_length = name, len(path)
    return best


def close_project_matches(selector: str, names: List[str]) -> List[str]:
    """Finds the registered names a selector plausibly meant.

    Matching runs over **casefolded** names on both sides and maps back to the
    originals, because the lookup itself is exact and every near-miss — case
    variants included — is supposed to route through did-you-mean. The naive
    one-sided spelling only half-honours that: a lightly-cased typo (``Mitos``)
    matches either way, but an **all-caps** selector (``MITOS``, ``ĄŽUOLAS``) —
    what a human produces from a shouted config value or a copied heading —
    silently suggests nothing. ``casefold()``, never ``lower()``: Lithuanian is
    load-bearing in this project, and ``lower()`` does not fold ``ß`` at all.

    Folding exists for *matching* only. No normalization enters resolution — the
    registry comparison stays exact, which is what forecloses the whole
    ``lower()``-vs-``casefold()`` drift class at the root.

    Args:
        selector: The unresolvable selector.
        names: Registered names, in document order.

    Returns:
        Up to ``PROJECT_DIDYOUMEAN_MAX`` folded matches, best first, each expanded
        to the originals that fold onto it in document order (two names differing
        only in case is a legal hand-edit, so both are named rather than one
        being silently dropped).
    """
    folded: Dict[str, List[str]] = {}
    for name in names:
        folded.setdefault(name.casefold(), []).append(name)

    matches = difflib.get_close_matches(
        selector.casefold(),
        list(folded),
        n=PROJECT_DIDYOUMEAN_MAX,
        cutoff=PROJECT_DIDYOUMEAN_CUTOFF,
    )
    suggestions: List[str] = []
    for match in matches:
        suggestions.extend(folded[match])
    return suggestions


def bounded_registered_names(
    names: List[str], close_matches: List[str]
) -> BoundedNames:
    """Applies the enumeration bound a renderer must respect.

    Args:
        names: Every registered name, in document order.
        close_matches: The did-you-mean suggestions for this failure, used as the
            collapsed rendering.

    Returns:
        The :class:`BoundedNames` policy verdict. Above the bound with no close
        matches, ``names`` is empty and ``total`` is intact — the honest answer is
        the count plus the surface's discovery pointer, not an arbitrary slice.
    """
    total = len(names)
    if total <= REGISTERED_NAMES_BOUND:
        return BoundedNames(names=list(names), total=total, collapsed=False)
    return BoundedNames(names=list(close_matches), total=total, collapsed=True)


def exempt_verb_error(verb: str, reason: str) -> ProjectTargetingError:
    """Builds the error for a selector handed to a verb that targets no project.

    Returned rather than raised, so the caller raises it at its own boundary. The
    verb→reason mapping belongs to whichever surface owns its verb table, not
    here — this leaf only fixes the vocabulary the two surfaces share.

    Args:
        verb: The verb that was given a selector.
        reason: A key from ``EXEMPT_REASONS``, never prose.

    Returns:
        The typed error, ready to raise.

    Raises:
        ValueError: If ``reason`` is not a known exempt reason — a programming
            fault in mitos, deliberately not a ``MitosError``.
    """
    return ProjectTargetingError(
        TARGET_EXEMPT_VERB, verb=verb, exempt_reason=reason
    )


def resolve_project(selector: Optional[str]) -> ResolvedProject:
    """Resolves a selector to a validated, canonical workspace root.

    The selector is a **required argument with no default**, which is the
    structural half of "no call resolves its workspace from the process's working
    directory": the function cannot express a cwd fallback. It is ``Optional``-
    typed so an *absent* selector goes through the same single raise site as every
    other failure, rather than letting either boundary invent a fifth wording. An
    empty string is the missing class too, not an unknown project named ``''``: it
    carries no target — and both boundaries gate on ``is not None``, never on
    truthiness, so ``-p ""`` / ``project=""`` reach that raise site rather than
    falling back.

    **Mid-vision, the absent case does not reach here at all.** Each boundary
    coalesces its own spelling first — ``cli._selector_from_args`` over
    ``project_pre``/``project_post`` plus the ``status``/``agent-block``
    positional, ``mcp_server._target_config`` over the tool's ``project``
    argument — and a *selector-less* call still keeps today's working-directory
    behaviour instead of calling this function, which is why the ``Optional``
    branch is currently reachable only through an empty string. Phases 5a and 5b
    remove those fallbacks and route the absent case here too; until then the
    branch is built, tested, and deliberately not yet the only path.

    The registry read is **not** optional and its faults propagate unwrapped —
    including on the path form, which needs routing for nothing but the echo name.
    A best-effort ``except RegistryError: {}`` there would hide a corrupt routing
    table on the dominant agent path, and it is the exact shape of a guard undone
    by a coalescing default one line below itself. A genuinely *absent* registry
    stays healthy: it loads as ``{}``, so a fresh machine's escape-hatch call is
    unaffected.

    Args:
        selector: A registered name, or an absolute path. A path-shaped selector
            that is not absolute is refused rather than resolved against cwd; a
            boundary that means an explicitly-typed relative path absolutizes it
            before calling.

    Returns:
        The :class:`ResolvedProject`.

    Raises:
        ProjectTargetingError: On any of the five resolution failure classes.
        RegistryError: If the registry file itself is unusable — a fault of the
            file, not of this lookup.
    """
    reg = registry.load()
    names = list(reg)

    if not selector:
        raise ProjectTargetingError(TARGET_MISSING, registered_names=names)

    if is_path_shaped(selector):
        if not os.path.isabs(selector):
            # No did-you-mean here: the caller claimed a path, and suggesting a
            # name for it says the wrong thing is the spelling of the name. The
            # registered list still rides along, because a name is usually what
            # the caller wanted.
            raise ProjectTargetingError(
                TARGET_RELATIVE_PATH, selector=selector, registered_names=names
            )
        root = registry.canonicalize(selector)
        if not is_workspace(root):
            raise ProjectTargetingError(
                TARGET_PATH_NOT_A_WORKSPACE,
                selector=selector,
                path=root,
                registered_names=names,
            )
        # An unregistered path is a correct steady state — a clone that never runs
        # `init` — so `name` is simply None and no warning fires anywhere.
        return ResolvedProject(
            root=root, name=registry.reverse_lookup(root, reg), via="path"
        )

    recorded = registry.lookup(selector, reg)
    if recorded is None:
        raise ProjectTargetingError(
            TARGET_UNKNOWN_NAME,
            selector=selector,
            registered_names=names,
            close_matches=close_project_matches(selector, names),
        )

    root = registry.canonicalize(recorded)
    if not is_workspace(root):
        # Its own class, and it fires on the name form only: a path-form selector
        # resolves *through* nothing, so a path that fails validation is always
        # "no workspace there" even when that path happens to be registered.
        # Keeping the class to one meaning is what makes its recovery — repoint
        # the registration — the true diagnosis every time it fires. `path` is the
        # registry's recorded value, because that is the string a repoint edits.
        raise ProjectTargetingError(
            TARGET_REGISTERED_UNREACHABLE,
            selector=selector,
            name=selector,
            path=recorded,
            registered_names=names,
        )
    return ResolvedProject(root=root, name=selector, via="name")
