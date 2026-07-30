"""Configuration management for Mitos.

This module handles loading and validating Mitos configuration from `.mitos/config.toml`
and defines system-wide defaults.
"""

import hashlib
import os
import re
import sys
import tomllib
from typing import Dict, Any, Optional, Tuple

from mitos import env, models
from mitos.errors import ConfigError

# The variables `MitosConfig` resolves per construction and carries on `.env`.
# Two groups: the credentials + the Qdrant URL that every substrate consumer
# reads, and the four model overrides.
#
# The override names are DERIVED from `models.MODEL_IDS` rather than re-declared,
# because `models` is a sibling leaf (stdlib only, no `config` import) — the
# renderer precedent in `CONFIG_DEFAULTS` below, which re-declares constants with
# a lockstep cross-check test, exists to avoid a tier INVERSION between `config`
# and the higher-tier `renderer`, and there is none here. Deriving them is also
# what keeps `MITOS_MODEL_OVERRIDE_EMBEDDING` in the set: `MODEL_ALIASES` omits
# `EMBEDDING`, and a set built from it looks correct while dropping the one
# override that is costliest to lose — the embedding cache keys on content hash
# alone (`embeddings.py`), so a mis-routed embedding override reads as working
# while cached prior-generation vectors flow into a new-generation collection.
# `MODEL_IDS`' keys are already upper-case; no `.upper()` belongs here.
RESOLVED_ENV_KEYS: Tuple[str, ...] = (
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "QDRANT_URL",
) + tuple(f"MITOS_MODEL_OVERRIDE_{alias}" for alias in models.MODEL_IDS)

# ---------------------------------------------------------------------------
# v0.1 config schema (§5.2.6) — the SINGLE source of the static defaults.
#
# `CONFIG_DEFAULTS` holds the eight STATIC-default schema keys: `mitos init` (6b)
# seeds `config.toml` from this exact map, and the loader's missing-key fallback
# reads it — so a seeded file and a deleted-key fallback can never diverge (P11).
# `qdrant_url` is recognized + type-validated (in `CONFIG_SCHEMA`) but NOT here:
# its default is DYNAMIC (env-derived) and computed in `__init__` from its
# existing single-source helper, then file-overridable. `qdrant_collection` is
# dynamic too but no longer file-overridable at all — it is derived from the
# workspace path on every construction and retired from the file schema below.
# ---------------------------------------------------------------------------
CONFIG_DEFAULTS: Dict[str, Any] = {
    "rotation_mode": "archive",
    "rotation_archive_path_template": "decisions/archive/{year}-Q{quarter}.md",
    "rotation_volume_threshold_entries": 50,
    "stale_entry_window_days": 30,
    "embedding_cache_max_entries": 10_000,
    # Pinned to renderer.py's GLOBAL/SCOPE_OVERFLOW_WARN_CHARS via a cross-check
    # test (config.py is a lower-tier leaf; importing renderer would invert tiers).
    # V4 wires the renderer to read these keys, making config the runtime source.
    "render_global_overflow_warn_chars": 50_000,
    "render_scope_overflow_warn_chars": 20_000,
    # The Conflict sensor's licence toggle (v0.2). Read by the sync hook in Phase
    # 5a; dormant until then. The first bool-typed key across this machinery.
    "conflict_check_on_sync": True,
}

# The recognized file keys → expected (TOML scalar) type, for strict validation.
# The eight static keys above PLUS the dynamic-default `qdrant_url` = the nine-key
# schema. A file key NOT in this map is tolerated and skipped — split into two
# buckets by `_load_config_file`: a RECOGNIZED-but-retired key (`RETIRED_CONFIG_KEYS`
# below) is tolerated SILENTLY, while a genuinely unknown key (a typo) earns one
# calm stderr line.
CONFIG_SCHEMA: Dict[str, type] = {
    "rotation_mode": str,
    "rotation_archive_path_template": str,
    "rotation_volume_threshold_entries": int,
    "stale_entry_window_days": int,
    "embedding_cache_max_entries": int,
    "render_global_overflow_warn_chars": int,
    "render_scope_overflow_warn_chars": int,
    "qdrant_url": str,
    "conflict_check_on_sync": bool,
}

# Keys the code DELIBERATELY dropped from the file schema but still recognizes —
# their ATTRIBUTES survive at a default (R12); only the file-override capability is
# gone. These are NOT typos, so the per-invocation "unrecognized config key" warning
# is a false alarm: the `mitos init`-seeded `pending_threshold` line tripped it on
# every single call. They are tolerated SILENTLY. The warning is reserved for keys
# the code does not know at all — where it is the useful signal that a setting will
# silently not take effect.
#
# `qdrant_collection` joins them for a stronger reason than the other four: its
# file override was a SAFETY hole, not just dead weight. `mitos init` materialized
# the derived name into the file, so a `cp -r` sandbox or a `git clone` of a repo
# that committed `.mitos/` carried the ORIGINAL project's collection and every
# write in the copy overwrote the original's vectors. Retiring it at resolution
# time (rather than stripping the line at `init` time) is what makes the fix total:
# a workspace `init` never runs on is exactly the clonable state, so no `init`-time
# mechanism can reach it. A surviving line of any type is inert wherever it is met,
# and `_load_config_file` records it in `inert_file_keys` so the two CLI surfaces
# that print the resolved collection can name it as legacy.
RETIRED_CONFIG_KEYS: frozenset = frozenset(
    {
        "pending_threshold",
        "db_path",
        "decisions_file",
        "archive_dir",
        "qdrant_collection",
    }
)

# The `rotation_mode` enum: correct type (str) but a value outside this set is a
# hard ConfigError — a typo'd `rotation_mode` silently defaulting to "archive" and
# then archiving when the author meant "mark" is exactly the silent-coerce OD1
# forbids (a deliberate behavior change from the prototype's silent-ignore).
ROTATION_MODES = frozenset({"archive", "mark", "prune"})

# Epoch 1 of narrowing `rotation_mode` to `archive`. Both values stay ACCEPTED and
# both behave as `archive`; the CLI dispatcher prints one calm stderr line naming the
# mode. Grounds are corpus preservation (P6/M7 rebuild-faithfulness) plus ROADMAP
# Vision 7's "never an in-place purge" — deliberately NOT P5's literal "Sync never
# deletes entries from decisions.md", which read alone would also indict `archive`.
#
#   `mark`  wraps a synced block as `<!-- ROTATED START … ROTATED END -->`, and the
#           entry-stream tokenizer keeps comments as literal field text (V1-D7), so
#           both sentinel lines are absorbed as CONTINUATION lines of the adjacent
#           field. When that field is `**Mechanisms:**` the node id SHIFTS — for the
#           rotated entry AND for the un-rotated one above it. The next sync then
#           sees a new node colliding on slug. What shipped is also not what was
#           designed: the design specifies a single-line `<!-- synced: … -->`
#           annotation on an entry that stays live content, and two ADRs require any
#           parser-skipped annotation to sit in a reserved namespace — `parser.py`
#           has no namespace predicate at all. Deprecating it loses nothing that was
#           ever built.
#   `prune` removes the block from the buffer and writes it NOWHERE (the archive
#           write is gated on `rotation_mode == "archive"`), so the node has no source
#           block and `rebuild` — the tool's own repair story — permanently cannot
#           reconstruct it. Its "for users who fully trust the graph as source"
#           rationale belongs to the pre-M7/P6 "storage is the graph, markdown is a
#           render target" direction, which was later reversed.
#
# The values are NOT removed from `ROTATION_MODES`. Removing them raises ConfigError
# at load, which fires BEFORE verb dispatch and bricks the whole workspace — including
# `mitos status`, the one command needed to diagnose it. Epoch 2 (moving the key into
# `RETIRED_CONFIG_KEYS`) is tracked in ROADMAP and needs several releases of warning
# first. The coercion is safe both ways: for `mark` it is a rescue (today's behaviour
# is corruption); for `prune` it is strictly preserving (both remove from the buffer,
# archive additionally keeps the text), the only user-visible delta being new files
# under `decisions/archive/`, which the warning names.
DEPRECATED_ROTATION_MODES = frozenset({"mark", "prune"})


def _value_matches_type(value: Any, expected: type) -> bool:
    """Returns True if a parsed TOML value matches a schema key's expected type.

    Treats ``bool`` as distinct from ``int`` even though ``bool`` subclasses
    ``int``: a TOML ``true`` must NOT satisfy an int-typed key (the silent-coerce
    the strict loader exists to kill). Symmetrically, a bool-typed key (the v0.2
    ``conflict_check_on_sync``) only accepts a native TOML boolean — a ``1`` or a
    quoted ``"true"`` is a loud mismatch, not a coercion.

    Args:
        value: The value ``tomllib`` parsed for the key.
        expected: The type the key's ``CONFIG_SCHEMA`` entry requires.

    Returns:
        True if ``value`` is acceptably typed for ``expected``, else False.
    """
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected)


def _hint_cache_path(cache_name: str) -> str:
    """Returns the path to a debounce cache file under the user cache dir.

    Honors ``XDG_CACHE_HOME`` (so tests redirect it into a tmp dir) and falls back
    to ``~/.cache``. The file need not exist.

    Args:
        cache_name: The cache file's basename (e.g. ``"mcp_hint.json"``).

    Returns:
        Absolute path to ``<cache>/mitos/<cache_name>``.
    """
    cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(cache_home, "mitos", cache_name)


def hint_due(cache_name: str, key: str, window_seconds: float) -> bool:
    """Fail-silent once-per-window gate for a debounced nudge.

    Backs the recurring-nudge surfaces (the MCP-server hint, the render-overflow
    summary) so they fire at most once per ``window_seconds`` per ``key`` instead of
    on every call. Reads a small JSON cache keyed by ``key``; if that key has not
    fired within the window it stamps the current time and returns True, otherwise
    returns False. Never raises — a missing/corrupt cache or an unwritable cache dir
    degrades to "due" (the nudge shows) rather than crashing the caller.

    Args:
        cache_name: The cache file's basename, namespacing one nudge from another.
        key: The per-subject key to debounce on (typically a workspace path).
        window_seconds: Minimum seconds between two firings for the same key.

    Returns:
        True if the nudge is due now (and the firing was just stamped), else False.
    """
    import json
    import time

    now = time.time()
    path = _hint_cache_path(cache_name)
    shown: Dict[str, Any] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            shown = json.load(f)
    except (OSError, ValueError):
        shown = {}
    if not isinstance(shown, dict):
        shown = {}
    if now - shown.get(key, 0) < window_seconds:
        return False
    shown[key] = now
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(shown, f)
    except OSError:
        pass
    return True


def config_home() -> str:
    """Returns the machine-local config root Mitos keeps its global state under.

    The single XDG resolution for every global (non-workspace) Mitos file — the
    shared ``.env`` and the project registry both hang off it. It is one mechanism
    on purpose: the test suite redirects ``XDG_CONFIG_HOME`` per test, so a second
    hand-rolled ``expanduser("~/.config")`` anywhere would slip that isolation and
    write into the real user config.

    Returns:
        Absolute path to the config root (``$XDG_CONFIG_HOME``, else
        ``~/.config``). The directory need not exist.
    """
    return os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config"
    )


def global_env_path() -> str:
    """Returns the path to Mitos's global ``.env`` (shared across all projects).

    A single-user machine usually wants one set of API keys for every project,
    not a key re-entered per workspace. Mitos reads this global ``.env`` as a
    fallback BELOW any project ``.env`` (and below an explicit environment
    variable), so a key set here once serves every project; a project ``.env``
    still overrides it locally. Honors ``XDG_CONFIG_HOME``.

    Returns:
        Absolute path to ``<config>/mitos/.env`` (``~/.config/mitos/.env`` by
        default). The file need not exist.
    """
    return os.path.join(config_home(), "mitos", ".env")


def global_registry_path() -> str:
    """Returns the path to Mitos's global project registry.

    The registry is the machine-local ``name → absolute workspace path`` routing
    map: ``mitos init`` registers the workspace it scaffolds, and name-targeted
    commands resolve a selector through it instead of inferring a workspace from
    the process's working directory. It lives beside the global ``.env`` (one
    config root, one XDG mechanism) and deliberately **outside** every workspace —
    it routes between projects, so no project owns it.

    It stores routing only. Nothing about a project's identity (its collection,
    its graph, its corpus) is derived from this file, so editing, removing, or
    repointing an entry can never change what a workspace *is*.

    Returns:
        Absolute path to ``<config>/mitos/registry.toml``
        (``~/.config/mitos/registry.toml`` by default). The file need not exist —
        an absent registry is the healthy "no projects registered yet" state.
    """
    return os.path.join(config_home(), "mitos", "registry.toml")


def toml_scalar(value: Any) -> str:
    """Serializes a scalar to its TOML right-hand-side literal (or a quoted key).

    A deliberately tiny serializer — NOT a general TOML writer. The stdlib
    ``tomllib`` is read-only and P19 forbids pulling ``tomli-w`` for a handful of
    flat scalars, so both hand-rolled writers use this one: ``mitos init`` seeds
    ``.mitos/config.toml`` through it, and the registry writes both its keys
    (project names) and its values (workspace paths) through it.

    The string form is chosen by *value shape*, because the registry's real domain
    includes paths and names a config value never carries:

    * clean (no ``"``, no ``\\``, no control character) → today's **basic**
      string, byte-identical to what the config seeder has always emitted;
    * carrying a ``\\`` or a ``"`` but no ``'`` or control character → a TOML
      **literal** string, which processes no escapes at all. This is the
      load-bearing case: a path written into a *basic* string has its escapes
      interpreted on the next read (``"/x/a\\tb"`` comes back with a TAB), which
      would silently register a path that does not exist;
    * anything left (both quote kinds, or a newline/control character) → a basic
      string with ``\\\\`` / ``\\"`` / ``\\n`` / ``\\uXXXX`` escapes.

    The ``bool`` branch MUST stay above the ``int`` branch: ``bool`` subclasses
    ``int``, so an int-first order would emit ``True`` as ``1`` instead of ``true``.

    Args:
        value: The value to serialize (``str``, ``int``, or ``bool``).

    Returns:
        The TOML literal — e.g. ``'"archive"'`` for a string, ``'50'`` for an int,
        ``'true'``/``'false'`` for a bool. Every string form round-trips back to
        the original value through ``tomllib``.

    Raises:
        TypeError: If the value is not a plain ``str``/``int``/``bool``. Callers
            at a user-facing boundary convert this to their own calm error rather
            than letting it surface (I5).
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _toml_string(value)
    raise TypeError(
        f"toml_scalar cannot serialize {type(value).__name__}: {value!r}"
    )


def _is_toml_control(char: str) -> bool:
    """Reports whether a character may not appear raw inside a TOML string.

    TOML forbids raw control characters in both string kinds, with the single
    exception of a tab. Anything this returns True for has to be escaped, which
    also rules the value out of the literal-string form (literal strings have no
    escape mechanism at all).
    """
    return (char < "\x20" and char != "\t") or char == "\x7f"


#: Control characters TOML gives a short escape to; everything else goes \uXXXX.
_TOML_SHORT_ESCAPES = {
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _toml_string(value: str) -> str:
    """Serializes a string to the TOML form that round-trips it byte-identically.

    See :func:`toml_scalar` for the three-way choice and why it is value-driven.
    The same function serves keys and values: a basic- or literal-quoted key is
    legal TOML, and the registry always quotes its keys — a bare key would parse a
    dotted name (``example.com``) as a nested table and a non-ASCII name (P9:
    ``ąžuolas``) not at all.
    """
    has_control = any(_is_toml_control(c) for c in value)
    needs_escaping = '"' in value or "\\" in value

    if not needs_escaping and not has_control:
        # The shipped form. Every existing config.toml line keeps these bytes.
        return f'"{value}"'
    if "'" not in value and not has_control:
        # A literal string processes no escapes — the faithful form for a path.
        return f"'{value}'"

    # Both quote kinds, or a control character: a fully escaped basic string.
    # `\\` MUST be escaped before `"`, or the backslash just added is doubled.
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    out = []
    for char in escaped:
        if _is_toml_control(char) or char == "\t":
            out.append(_TOML_SHORT_ESCAPES.get(char) or f"\\u{ord(char):04X}")
        else:
            out.append(char)
    return '"' + "".join(out) + '"'


def default_collection_name(workspace_dir: str) -> str:
    """Derives a per-project Qdrant collection name from the workspace path.

    Each Mitos workspace gets its OWN collection so a single shared Qdrant
    instance never mixes decisions across projects. Without this, every project
    would default to the same ``"mitos"`` collection and cross-contaminate
    semantic queries — and, because a point's id is ``hash_to_uuid`` of the
    content hash (M2), two projects recording the same axiom would collide on
    one Qdrant point.

    The name is ``mitos-<safe basename>-<8 hex of sha256(canonical path)>`` — a
    pure function of *where the workspace is*, with no opt-out. Nothing persists
    it (not ``config.toml``, not the registry, not the graph), so a copied or
    cloned workspace cannot inherit the original's collection; a
    ``qdrant_collection`` line in ``.mitos/config.toml`` is inert legacy config
    (see ``RETIRED_CONFIG_KEYS``). The basename alone would not do: two
    same-named sibling projects, and a ``cp -r`` of one workspace, share it.

    The four steps and their order are **contract** and cannot change after
    release. The name is the address of live data, so a changed derivation
    renames every collection in the wild and strands its vectors (P1,
    interoperability across time):

    1. ``os.path.realpath`` the input — collapses ``.``/``..``/a trailing slash
       **and** resolves symlinks, so every route to one directory lands on one
       collection. It never raises and never requires the path to exist, which
       keeps this function total (and is why no ``exists()`` probe belongs here —
       whether a path is a workspace is the caller's question).
    2. ``safe`` = the basename **of that canonical path**, lowercased, each run of
       non-``[a-z0-9_-]`` collapsed to ``-``, outer ``-`` stripped.
    3. ``digest`` = the first 8 hex of ``sha256`` over the **full canonical path**
       — never over the basename and never over ``safe``. Hashing either would
       give a ``cp -r`` sibling the same digest, which is the entire failure this
       derivation exists to close.
    4. Join, omitting an empty ``safe``.

    Two details a reader will otherwise assume wrongly:

    * **The digest's input is ``os.fsencode(canonical)``, not
      ``canonical.encode("utf-8")``.** A POSIX path is bytes, and ``realpath``
      hands back a ``str`` carrying ``surrogateescape`` code points for any byte
      that is not valid UTF-8 — which ``str.encode("utf-8")`` refuses. So the
      obvious spelling would raise ``UnicodeEncodeError`` from ``MitosConfig``'s
      constructor for a workspace whose path holds one Latin-1 byte, taking down
      every verb including the ``mitos status`` you would run to diagnose it.
      ``fsencode`` agrees with UTF-8 byte-for-byte on every valid path, so it
      costs nothing and hashes the path's actual bytes — its true identity.
    * **An empty ``safe`` yields two segments (``mitos-<digest>``), never a bare
      ``mitos``.** A basename that sanitizes away entirely (``проект``, ``日本語``,
      ``/``) must not land every such project on one shared collection — P9's own
      case should not be the one that cross-contaminates. The digest is never
      omitted, so a directory literally *named* like a digest still derives three
      segments and cannot collide with the two-segment form.

    Residual, named rather than fixed: ``realpath`` does not case-fold, so on a
    case-insensitive filesystem two spellings of one directory derive two
    collections. That degrades benignly (a duplicate empty collection, healed by
    one ``mitos reconcile``); case-folding would trade it for a genuine
    cross-project collision on a case-sensitive one.

    Args:
        workspace_dir: The workspace directory (the project root holding
            ``.mitos/``). Need not exist.

    Returns:
        A Qdrant-safe collection name unique to the workspace's canonical path.
    """
    canonical = os.path.realpath(workspace_dir)
    base = os.path.basename(canonical).lower()
    safe = re.sub(r"[^a-z0-9_-]+", "-", base).strip("-")
    digest = hashlib.sha256(os.fsencode(canonical)).hexdigest()[:8]
    return f"mitos-{safe}-{digest}" if safe else f"mitos-{digest}"


class MitosConfig:
    """Represents the configuration state for the active Mitos workspace."""

    def __init__(self, workspace_dir: str = ".", *, project: Optional[str] = None) -> None:
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.mitos_dir = os.path.join(self.workspace_dir, ".mitos")

        # The name the CALLER used for this workspace, carried so every answer
        # can echo the target back in the caller's own vocabulary. Filled at the
        # two resolution sites (`cli.main`, `mcp_server._target_config`) from
        # `ResolvedProject.name`, which is already the registered name for both
        # selector forms — a name-form selector by construction, a path-form one
        # via the registry's reverse lookup — and `None` for an unregistered
        # path. So the fallback below covers the escape hatch and the
        # transitional selector-less call in one expression, and "never empty"
        # is a property of construction rather than of every call site
        # remembering.
        #
        # `or` rather than `if project is None` (2c's rule elsewhere) precisely
        # here: `registry.validate_name` forbids an empty name, so `""` is not a
        # supplied answer on this field the way it is for an env value — the two
        # spellings are behaviourally identical and `or` also absorbs a future
        # caller passing `""`. Do not "fix" it into the other form.
        #
        # Runtime-only, like `self.env` below and `inert_file_keys`: absent from
        # `to_dict()`, never persisted, and not a `CONFIG_SCHEMA` key — a
        # `project = "…"` line in `config.toml` takes the unknown-key branch
        # (one calm warning, no `setattr`) and cannot become the echo.
        self.project = project or self.workspace_dir

        # The resolved environment for THIS workspace — real env, then the
        # workspace's own `.env`, then the machine-global one, computed rather
        # than read off a process `os.environ` some launch directory filled in.
        # Runtime-only: it holds real API keys, so it is never persisted, never
        # in `to_dict()`, and never cached across calls (`mcp_server` builds a
        # fresh config per tool call by design, which is the point).
        #
        # Position is load-bearing and belongs HERE rather than beside the
        # qdrant block: `self.qdrant_url` reads this map below, and a later edit
        # that drifts the population downward past the post-load re-assert would
        # break that read. Everything between here and the read is pure
        # `os.path.join` derivation — no env read, no file read — so this is the
        # earliest slot that has `self.workspace_dir` to resolve for. (The
        # `self.project` line above is a plain assignment that reads neither, so
        # it does not move this boundary.)
        self.env: Dict[str, str] = env.resolve_values(
            RESOLVED_ENV_KEYS, self.workspace_dir, global_env_path()
        )

        # Convention-path attributes — derived from the workspace, NOT file-schema
        # keys in v0.1 (a file occurrence is warn-tolerated). Consumers
        # (store/sync/importer/cli/mcp_server) bind these by name (R12), so they
        # stay real instance attributes even though the file can no longer set them.
        self.db_path = os.path.join(self.mitos_dir, "graph.sqlite")
        # The Conflict sensor's non-rebuildable telemetry store (v0.2), a sibling of
        # the graph deliberately fenced OUTSIDE the rebuild/cutover swap set so it
        # survives ``rm graph.sqlite`` / ``mitos rebuild`` (CONF-D8, the T8
        # guarantee). A derived attribute, NOT a user-overridable file-schema key —
        # deriving it here gives the store + the 5b sync surface one canonical path
        # expression sitting next to ``db_path``, instead of reassembling "sibling of
        # the graph" at each call site.
        self.telemetry_path = os.path.join(self.mitos_dir, "telemetry.sqlite")
        self.decisions_file = os.path.join(self.workspace_dir, "decisions.md")
        # The open-question authoring buffer, a fixed v0.1 convention path
        # paralleling ``decisions_file`` (ADR
        # ``open-questions-authored-in-separate-questions-md-file``). ``mitos init``
        # (6b) seeds it; the V3a sync / V6 importer consumers read it later
        # (forward-provided — no in-vision reader yet, like 5d's protocol seams).
        self.questions_file = os.path.join(self.workspace_dir, "questions.md")
        self.archive_dir = os.path.join(self.workspace_dir, "decisions", "archive")

        # `pending_threshold` LEFT the v0.1 file schema (its migration to
        # `rotation_volume_threshold_entries` is V3a's, not V1a's) but stays a
        # default-valued attribute — `sync.py`'s rotation-prompt gate reads it. A
        # `pending_threshold` file key is now silently tolerated (a recognized
        # retired key — see RETIRED_CONFIG_KEYS), not applied.
        self.pending_threshold = 30

        # Dynamic-default schema keys: recognized + type-validated by CONFIG_SCHEMA,
        # file-overridable, but defaulted from their single-source helpers (not from
        # CONFIG_DEFAULTS, which holds only the STATIC defaults).
        #
        # Mitos defaults to its OWN dedicated port (:7333), NOT the standard
        # Qdrant :6333 — a user's :6333 is usually running for something else, so
        # defaulting there would co-locate Mitos's collections in their instance
        # and share its wipe/contamination risk. :7333 fails safe (semantic just
        # degrades if Mitos's Qdrant isn't up). `docker compose up` starts it.
        # QDRANT_URL overrides for anyone pointing at a different instance —
        # resolved through the carrier above, so a URL living in the TARGET
        # workspace's `.env` reaches every store construction no matter which
        # directory the process was launched from. `.get(k, default)`, never
        # `.get(k) or default`: an exported-empty QDRANT_URL resolves to `""`
        # and must stay `""` (the second spelling silently restores the default).
        self.qdrant_url = self.env.get("QDRANT_URL", "http://localhost:7333")
        # Per-project so a shared Qdrant never mixes projects' decisions — and
        # per-PATH, not per-name, so a copy of a workspace cannot address the
        # original's vectors. Re-derived on every construction and NEVER persisted;
        # a `qdrant_collection` line in the file cannot override it (it is retired).
        self.qdrant_collection = default_collection_name(self.workspace_dir)

        # Retired file keys the workspace's config.toml actually carries:
        # {key: the value found in the file}. Populated by `_load_config_file`,
        # runtime-only, never persisted and absent from `to_dict()`. It exists so
        # the CLI surfaces that PRINT a resolved value can name a surviving line as
        # inert legacy config — the loader itself must stay mute (it runs once per
        # MCP tool call over a stdio JSON-RPC channel where stdout is protocol, and
        # `status` builds a second config of its own). General over the whole retired
        # set rather than `qdrant_collection`-shaped: no key-specific branch in the
        # loader, and a future retirement gets its data for free.
        self.inert_file_keys: Dict[str, Any] = {}

        # Static-default schema keys — seeded from the single CONFIG_DEFAULTS map
        # (P11), the same map `mitos init` (6b) serializes. The keys are exactly the
        # attribute names, so a plain setattr keeps the surface in lockstep.
        for key, default in CONFIG_DEFAULTS.items():
            setattr(self, key, default)

        # The configured-but-deprecated rotation mode, kept so the CLI dispatcher can
        # name it in its one stderr line. `None` on every healthy workspace, so a
        # clean project reads clean. The loader itself stays MUTE: `mitos status`
        # builds a second MitosConfig of its own, and `mcp_server` builds one per tool
        # call over a stdio JSON-RPC channel — a warning here would print twice on
        # status, once per MCP call, and anything on stdout there is protocol
        # corruption rather than noise. Warning at dispatch needs no once-flag (a
        # fresh process per CLI invocation) and means MCP never warns, which is right:
        # no config author is present on that surface.
        self.deprecated_rotation_mode: Optional[str] = None

        self._load_config_file()

        # The resolved env wins over the config file for the Qdrant URL — the same
        # layering as the keys (env → project .env → global .env), now literally
        # the same mechanism, and the documented contract above ("QDRANT_URL
        # overrides for anyone pointing at a different instance"). Before this
        # re-assert, a toml-pinned qdrant_url silently shadowed the env var
        # (AX 2026-07-18): the caller's override did nothing and nothing said so.
        #
        # `QDRANT_URL` is deliberately NOT migrated out of `.env` into
        # `config.toml`, where the schema key already lives: `mitos init`
        # force-gitignores `.env` and never `config.toml`, so a project's
        # `config.toml` is committable by intent and a user who put a
        # secret-bearing remote URL in the gitignored file did so deliberately.
        # The global tier is load-bearing for the same variable — a
        # shared-instance URL is per-machine and naturally lives in the global
        # `.env`.
        if self.env.get("QDRANT_URL"):
            self.qdrant_url = self.env["QDRANT_URL"]

    def _load_config_file(self) -> None:
        """Overlays `.mitos/config.toml` onto the defaults under the strict policy.

        Replaces the prototype's hand-rolled ``key=val`` parser (which swallowed
        every error back to defaults) with a ``tomllib`` loader enforcing the
        §5.2.6 failure-mode policy, symmetric with OD1: a broken config is loud and
        located, never silently defaulted.

        Policy:
            - Malformed TOML → ``ConfigError`` carrying the path + the decoder's
              line/column message. No fallback.
            - A known key with the wrong type → ``ConfigError`` naming the key,
              expected type, and got type.
            - ``rotation_mode`` with a valid-string-but-out-of-enum value →
              ``ConfigError`` (the silent-coerce OD1 forbids).
            - A missing known key → keeps the already-seeded default.
            - A recognized-but-retired key (``RETIRED_CONFIG_KEYS``) → tolerated and
              skipped SILENTLY (not a typo; a per-call warning on it is just noise),
              before the type check, so a mistyped retired key is inert rather than
              fatal. It is recorded in ``inert_file_keys`` for the CLI surfaces that
              report a surviving line; this loader prints nothing.
            - A genuinely unknown key (a typo) → one calm stderr line, tolerated,
              skipped.

        Raises:
            ConfigError: On malformed TOML, a type mismatch, or an out-of-enum
                ``rotation_mode``.
        """
        config_path = os.path.join(self.mitos_dir, "config.toml")
        if not os.path.exists(config_path):
            return

        try:
            # tomllib requires BINARY mode — a text-mode handle raises TypeError.
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(
                f"Malformed config at {config_path}: {e}. "
                f"Fix the offending line or remove it."
            ) from e
        except OSError as e:
            # The file existed at the os.path.exists check but can't be read now
            # (permissions, a TOCTOU vanish, a directory). Keep the error vector
            # uniform: every failure to LOAD the config is a located ConfigError,
            # never a raw "Fatal Unexpected Error" — and never a silent default.
            raise ConfigError(f"Cannot read config at {config_path}: {e}.") from e

        for key, val in data.items():
            if key not in CONFIG_SCHEMA:
                # A recognized-but-retired key (deliberately dropped from the file
                # schema; its attribute still lives at a default, R12) is tolerated
                # SILENTLY — it is not a typo, so warning on it every call is pure
                # noise. A genuinely unknown key (a typo whose setting silently won't
                # take effect) still earns one calm, terse, screen-reader-clean line
                # to stderr (P9, no emoji) — that warning is the useful signal.
                if key in RETIRED_CONFIG_KEYS:
                    # Remembered, not applied: a printing surface can name it as
                    # inert legacy config beside the value it claims to set.
                    self.inert_file_keys[key] = val
                else:
                    print(
                        f"Warning: ignoring unrecognized config key "
                        f"'{key}' in {config_path}",
                        file=sys.stderr,
                    )
                continue

            expected = CONFIG_SCHEMA[key]
            if not _value_matches_type(val, expected):
                raise ConfigError(
                    f"Config key '{key}' in {config_path} expects "
                    f"{expected.__name__}, got {type(val).__name__} ({val!r}). "
                    f"Fix the value's type."
                )

            if key == "rotation_mode" and val not in ROTATION_MODES:
                allowed = ", ".join(sorted(ROTATION_MODES))
                raise ConfigError(
                    f"Config key 'rotation_mode' in {config_path} must be one of "
                    f"{{{allowed}}}, got {val!r}."
                )

            if key == "rotation_mode" and val in DEPRECATED_ROTATION_MODES:
                # Epoch 1: accepted, pinned to `archive`, and remembered so the CLI
                # dispatcher can name it. Not silent coercion — which is the thing
                # `config-loader-rotation-mode-enum-hard-fail` forbids — because the
                # dispatcher warns every run. That ADR commits to loud validation,
                # not to three modes.
                self.deprecated_rotation_mode = val
                self.rotation_mode = "archive"
                continue

            # Schema keys are exactly the attribute names (R12 surface).
            setattr(self, key, val)

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to dictionary form.

        Includes the convention-path attributes, the two dynamically-defaulted
        qdrant attributes, the kept-but-de-schema'd ``pending_threshold``, and the
        eight static schema keys (sourced from ``CONFIG_DEFAULTS`` so the set can't
        drift). ``qdrant_collection`` stays here even though it left the file
        schema — that is the retirement pattern's promise: the attribute survives
        at its computed default and every consumer binding it is unaffected.
        ``inert_file_keys`` is deliberately absent (runtime-only, never persisted).
        No consumer binds this today; it exists for a future ``--json``/debug
        surface.

        Returns:
            A dictionary containing every configuration field.
        """
        result: Dict[str, Any] = {
            "workspace_dir": self.workspace_dir,
            "mitos_dir": self.mitos_dir,
            "db_path": self.db_path,
            "telemetry_path": self.telemetry_path,
            "qdrant_url": self.qdrant_url,
            "qdrant_collection": self.qdrant_collection,
            "pending_threshold": self.pending_threshold,
            "decisions_file": self.decisions_file,
            "questions_file": self.questions_file,
            "archive_dir": self.archive_dir,
        }
        # The eight static schema keys (incl. rotation_mode) from their one source.
        for key in CONFIG_DEFAULTS:
            result[key] = getattr(self, key)
        return result
