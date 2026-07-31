"""Layered ``.env`` resolution: the thing that answers *with which keys?*

Mitos used to learn its secrets by **mutating the process it lived in** —
``cli.main()`` poured a project ``.env`` and the global ``.env`` into
``os.environ`` at startup, and from then on every question about "which key?"
was answered by the environment the process happened to be standing in. For a
CLI whose working directory *is* its project that was a defensible guess. For an
always-on server that serves every project on the machine from one launch
directory it was not a guess at all: the launch dir's ``.env`` was baked in for
the process's whole life, so a call meaning project B resolved A's key — and the
thing that leaked was the user's credential.

This module replaced that mutation with a function, and since phase 5c the
mutation is gone: **mitos writes to** ``os.environ`` **nowhere.**
:func:`resolve_key` and :func:`resolve_values` compute a value **for a named
target**, reading real env first, then ``<target_dir>/.env``, then the injected
global ``.env``. That is what makes per-call statelessness a property that can be
stated and tested rather than hoped for: a program that never writes to its own
environment can be asked the same question twice, about two different projects,
and answer honestly both times.

**The resolution matrix.** The two file tiers and the process environment do not
test the same thing, and the asymmetry is load-bearing rather than cosmetic
(``absent`` = the name is not there at all; ``empty`` = present with an empty
value; ``set`` = present and non-empty):

    ================  ===============  ==============  ==================
    ``os.environ``    project ``.env``  global ``.env``  resolves to
    ================  ===============  ==============  ==================
    set               (any)            (any)           env value
    **empty**         (any)            (any)           ``""`` @ environment
    absent            set              (any)           project value
    absent            **empty**        set             global value
    absent            absent           set             global value
    absent            absent           absent/empty    ``None``
    ================  ===============  ==============  ==================

Tier 1 tests **presence**, tiers 2 and 3 test a **non-empty value**. Both halves
are earned:

* An **empty project slot with a global key must resolve to the global key** —
  ``mitos init`` scaffolds an empty ``GEMINI_API_KEY=`` line under a comment
  telling the user to set the key once globally, so a resolver testing key
  *presence* at tier 2 would find ``""`` and every project that followed the
  tool's own advice would lose its key.
* An **exported empty variable must mask both files.** Mitos ships an operator
  idiom that depends on it: ``env GEMINI_API_KEY= ANTHROPIC_API_KEY= mitos …``
  is how a keyless run is produced on a key-bearing dev box (a bare unset does
  not work — a global ``.env`` refills it). Under a uniform non-empty test the
  empty export stops masking, falls through to the files, and the "keyless" run
  fires real, billed LLM calls. The same rule keeps ``QDRANT_URL`` byte-
  identical to its pre-resolver behaviour, where ``os.environ.get(name,
  default)`` returned ``""`` rather than the default.

Failure is silent by construction: a ``.env`` that is absent, unreadable, a
directory, or not valid UTF-8 parses to ``{}``. It is a credentials file read on
every construction, so the only calm behaviour is to have nothing to say about
a file that cannot be read — and until this module existed a non-UTF-8 ``.env``
raised ``UnicodeDecodeError`` out of the CLI's entry load and rendered as
``Fatal Unexpected Error`` for every verb, ``mitos status`` included.

Tier 1, permanently: **stdlib only, and nothing from ``mitos`` at all** — not
``config``, not ``errors``, not ``models``. That is the strongest tier statement
in the tree and it is pinned by a subprocess import probe. It is also why the
global ``.env`` path arrives as an **argument** rather than being derived here
(see :func:`resolve_key`): ``global_env_path`` lives in ``config``, which must
import *this* module to build its carrier, so deriving it here would close a
cycle — and a second hand-rolled ``~/.config`` resolution would slip the test
suite's per-test ``XDG_CONFIG_HOME`` redirect and read the developer's real
keys from ~1900 tests.
"""

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# The winning-tier vocabulary. Byte-identical to the three strings `mitos status`
# already prints for its key-attribution line, so routing that report through
# this leaf is a swap and not a rewording.
TIER_ENVIRONMENT = "environment"      # the process environment
TIER_PROJECT_ENV = "project .env"     # <target_dir>/.env
TIER_GLOBAL_ENV = "global .env"       # the injected global path
ENV_TIERS = frozenset({TIER_ENVIRONMENT, TIER_PROJECT_ENV, TIER_GLOBAL_ENV})

# The workspace-relative `.env` convention, in one place. Its other two writers
# are `mitos init`'s scaffold and `mitos set-key`; nothing here writes.
ENV_BASENAME = ".env"


@dataclass
class ResolvedValue:
    """One resolution, with the tier that answered — a runtime value, never persisted.

    Attributes:
        value: The resolved value, or ``None`` iff no tier answered. ``""`` is a
            real answer, not an absence: it is what an exported-empty variable
            resolves to, and collapsing the two is exactly the masking bug the
            module docstring's matrix exists to prevent.
        tier: One of :data:`ENV_TIERS`, or ``None`` alongside a ``None`` value.
            A plain ``str`` rather than an enum, on ``ResolvedProject.via``'s
            precedent: nothing serializes it, so an enum would buy a persistence
            contract no boundary needs.

    Note:
        The report answers *where this value came from*, never *is there a usable
        key*. An exported-empty variable resolves to ``ResolvedValue("",
        "environment")`` — a consumer reporting key presence must key on the
        **value** being truthy, not on the tier being non-``None``, or ``env
        GEMINI_API_KEY= mitos status`` starts claiming a key is present.
    """

    value: Optional[str]
    tier: Optional[str]


def parse_env_file(path: str) -> Dict[str, str]:
    """Parses a ``.env`` file into its ``KEY=value`` pairs.

    The single ``.env`` **read** for the whole tree: two hand-rolled parses of
    one file that must agree is a drift this module can simply not create. (The
    one remaining hand-rolled sibling is ``cli._upsert_env_var``, a *writer*,
    which matches on the parsed key so writer and reader agree about which line
    is which.) Comments, blank lines and lines carrying no ``=`` are skipped;
    whitespace around the key and the value is stripped; a value keeps everything
    right of its first ``=``, so a query string survives.

    **First non-empty assignment wins within a file**, not last. It is the
    behaviour the deleted entry-time load had — it guarded on ``key not in
    os.environ``, so once a key was set later lines for it were skipped — and a
    plain ``dict[key] = value`` loop is last-wins and would have inverted it
    silently on any file carrying a key twice. The common shape (a scaffolded
    empty slot followed by a real value) happens to agree either way; two real
    values do not.

    Quote stripping is ``.strip('"').strip("'")`` — double quotes first — copied
    from the shipped parse character for character, which makes it deliberately
    asymmetric: ``"'v'"`` strips to ``v`` while ``'"v"'`` strips to ``"v"``.
    Reordering it (or spelling it ``.strip("\\"'")``) moves behaviour for any
    ``.env`` using single-outer quoting.

    Args:
        path: Path to the ``.env`` file. Need not exist.

    Returns:
        The parsed pairs, empty values excluded. ``{}`` for a file that is
        absent, unreadable, a directory, or not valid UTF-8 — a credentials file
        read on every config construction has nothing useful to say by raising,
        and the non-UTF-8 case used to reach the CLI's generic error arm.
    """
    values: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val and key not in values:
                    values[key] = val
    except (OSError, UnicodeDecodeError):
        # Whole-file, not partial: a file that failed mid-decode is not a source
        # of truth for the lines that happened to parse before the bad byte.
        return {}
    return values


def _file_tiers(
    target_dir: str, global_env_path: str
) -> List[Tuple[str, Dict[str, str]]]:
    """Reads the two file tiers once each, in precedence order.

    Read once per resolution call rather than once per name: the carrier asks for
    seven names, and a per-name read would be fourteen file reads on every
    ``MitosConfig`` construction — one per MCP tool call, by design.

    Args:
        target_dir: The workspace whose ``.env`` is tier 2.
        global_env_path: Absolute path to the machine-global ``.env`` (tier 3).

    Returns:
        ``[(tier, parsed pairs), …]``, project before global.
    """
    return [
        (TIER_PROJECT_ENV, parse_env_file(os.path.join(target_dir, ENV_BASENAME))),
        (TIER_GLOBAL_ENV, parse_env_file(global_env_path)),
    ]


def _resolve(name: str, file_tiers: List[Tuple[str, Dict[str, str]]]) -> ResolvedValue:
    """Applies the layering to one name over already-read file tiers.

    The one layering implementation. :func:`resolve_key` and
    :func:`resolve_values` are two entry points onto it, never two answers.

    Args:
        name: The variable name.
        file_tiers: The output of :func:`_file_tiers`.

    Returns:
        The :class:`ResolvedValue`.
    """
    # Presence, not truthiness — see the module docstring's matrix. `in` is the
    # whole point of this line; `os.environ.get(name)` would break the masking
    # idiom and change `QDRANT_URL`'s empty-export behaviour.
    if name in os.environ:
        return ResolvedValue(os.environ[name], TIER_ENVIRONMENT)
    for tier, values in file_tiers:
        # `parse_env_file` already drops empty assignments, so a hit here is
        # non-empty by construction and the empty-slot case falls through.
        value = values.get(name)
        if value:
            return ResolvedValue(value, tier)
    return ResolvedValue(None, None)


def resolve_key(name: str, target_dir: str, global_env_path: str) -> ResolvedValue:
    """Resolves one variable for a target workspace, with the tier that answered.

    Callable without a ``MitosConfig`` on purpose. A malformed
    ``.mitos/config.toml`` raises ``ConfigError`` from ``MitosConfig.__init__``,
    so if key resolution were reachable *only* through a config object a broken
    config would also brick the answer to "where is my key?" — for exactly the
    verbs (``init``, ``set-key``, ``status``) a user reaches for while a
    workspace is half-set-up.

    Args:
        name: The variable name.
        target_dir: The workspace to resolve *for* — never the process's working
            directory, which is the guess this module exists to remove.
        global_env_path: Absolute path to the machine-global ``.env``. Injected
            rather than derived here; the asymmetry with ``target_dir`` is the
            dependency-injection boundary that keeps this leaf ``mitos``-free
            (see the module docstring).

    Returns:
        The :class:`ResolvedValue` — ``ResolvedValue(None, None)`` when no tier
        answered.
    """
    return _resolve(name, _file_tiers(target_dir, global_env_path))


def resolve_values(
    names: Iterable[str], target_dir: str, global_env_path: str
) -> Dict[str, str]:
    """Resolves a set of variables for a target workspace.

    The carrier builder. Returns **values**, not reports: every consumer wants
    the value, and a ``Dict[str, ResolvedValue]`` would make each of them write
    ``.value`` for a diagnostic none of them read. The one consumer that wants
    the tier calls :func:`resolve_key`.

    Args:
        names: The variable names to resolve.
        target_dir: The workspace to resolve for.
        global_env_path: Absolute path to the machine-global ``.env``.

    Returns:
        ``{name: value}`` for the names that resolved, and only those — an
        unresolved name is **absent** rather than present-and-``None``, so a
        consumer's ``.get(name, <default>)`` behaves. A ``""`` from an exported
        empty variable **is** resolved and is present in the map.
    """
    file_tiers = _file_tiers(target_dir, global_env_path)
    resolved: Dict[str, str] = {}
    for name in names:
        result = _resolve(name, file_tiers)
        # Key on the tier, not on the value: `""` from tier 1 is a real answer
        # whose absence from the map would silently restore the default it is
        # there to suppress. A tier that answered always carries a value, so no
        # coalescing default belongs on the next line — `result.value or ""`
        # reads as harmless and is the same shape that has already produced two
        # defects in this vision.
        if result.tier is not None:
            resolved[name] = result.value
    return resolved
