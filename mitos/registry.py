"""The global project registry — Mitos's ``name → workspace`` routing map.

Mitos resolves its workspace from the process's working directory, which is
intent-blind on an always-on server: the launch directory is fixed for the
server's whole life, so a call meaning *project B* lands in whatever project the
server started in. This module is the fix's substrate — a machine-local map from
a project *name* to an absolute workspace *path*, held in a flat TOML file at
``<config-home>/mitos/registry.toml``, outside every workspace.

**The registry is routing; the workspace is identity.** It stores no collection,
no graph pointer, nothing derived. A project's identity stays entirely
workspace-local, so editing, removing, or repointing a registration can never
change what a workspace *is* — only which name reaches it. Nothing downstream may
treat this file as a source of record: it is a derivative, rebuildable by
re-running ``mitos init`` in each project.

The file is deliberately hand-editable, and two things follow from that. A write
is **structure-preserving**: one line changes and every other byte — comments,
blank lines, ordering, a human's organization — survives. And both write-time
guards (name collision, path uniqueness) are **guidance, not safety properties**:
a hand-edit can always produce two names for one path, which is a tolerated state
resolved by an explicit first-match rule, never rendered as a fault. Nothing
downstream may be built on their uniqueness.

Tier 1, permanently: stdlib plus ``mitos.config`` and ``mitos.errors``, and
nothing else. ``cli`` imports this module, so an import back would cycle; a
filesystem probe ("does that path still hold a workspace?") is a *reachability*
question belonging to the caller that asks it, not to a routing leaf.
"""

import os
import tempfile
import tomllib
from dataclasses import dataclass
from typing import Dict, List, Optional

from mitos.config import global_registry_path, toml_scalar
from mitos.errors import RegistryError


@dataclass
class RegistrationOutcome:
    """What one :func:`register` call did — a runtime value, never persisted.

    Attributes:
        name: The name the workspace is registered under.
        path: The canonical absolute path stored for it.
        action: ``"created"`` (the name is new), ``"reasserted"`` (the exact
            name→path pair was already there — an idempotent re-init), or
            ``"repointed"`` (``--force`` moved an existing name to this path). A
            plain ``str``, not an enum: nothing serializes it, so an enum would
            buy a persistence contract no boundary needs.
        previous_path: The path the name pointed at before, set only when
            ``action == "repointed"``.
    """

    name: str
    path: str
    action: str
    previous_path: Optional[str] = None


def registry_path() -> str:
    """Returns the resolved absolute path of the registry file.

    Delegates to :func:`mitos.config.global_registry_path` rather than resolving
    XDG a second time — one mechanism, so the suite's per-test config-root
    redirect covers every reader and writer.

    Returns:
        Absolute path to ``registry.toml``. The file need not exist.
    """
    return global_registry_path()


def load() -> Dict[str, str]:
    """Reads the registry into a ``name → path`` map, in document order.

    Document order is part of the contract, not an implementation detail: it is
    the order ``mitos projects`` renders and the order :func:`reverse_lookup`
    resolves its first match in, so a human reading the list sees the order that
    actually decides.

    Returns:
        The registered ``name → absolute path`` pairs, in the order they appear in
        the file. An **absent** registry (or an absent config directory) returns
        ``{}`` — no projects registered yet is the healthy fresh state, never an
        error.

    Raises:
        RegistryError: If the file is not readable, is not valid TOML, carries a
            value that is not a string, or carries a value that is not an
            absolute path (the two shapes the flat schema can be violated with).
    """
    path = registry_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        return {}
    except OSError as e:
        raise RegistryError(
            f"cannot read the Mitos project registry at {path}: {e}. "
            f"Fix its permissions or remove the file — `mitos init` re-registers "
            f"a project."
        ) from e
    except UnicodeDecodeError as e:
        raise RegistryError(
            f"the Mitos project registry at {path} is not valid UTF-8 text: {e}. "
            f"Fix or remove {path}."
        ) from e

    try:
        data = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as e:
        # A duplicate name arrives here for free: tomllib refuses a repeated key,
        # so a hand-edited collision surfaces as a malformed registry rather than
        # silently keeping one of the two.
        raise RegistryError(
            f"the Mitos project registry at {path} is not valid TOML: {e}. "
            f"Fix or remove {path} — an absent registry is healthy, and "
            f"`mitos init` re-registers a project."
        ) from e

    for name, value in data.items():
        if not isinstance(value, str):
            raise RegistryError(
                f"registry entry {name!r} in {path} holds a "
                f"{type(value).__name__}, not a workspace path string. The "
                f"registry is a flat name-to-path map; a name containing dots "
                f"must be quoted (`\"example.com\" = \"/path/to/it\"`) or TOML "
                f"reads it as a nested table. Fix or remove {path}."
            )
        # The schema is `name → ABSOLUTE path`, and this is where it is enforced —
        # one gate for every reader, on a pure string test that keeps the leaf's
        # tier untouched (no filesystem probe here, by design). A hand-edited
        # relative value would otherwise resolve against whatever directory the
        # process happens to stand in, which is the cwd-dependence the routing map
        # exists to remove — defeated *through* the map, silently. `~` is refused
        # rather than expanded: expanding it here would mint a second
        # canonicalization spelling beside `canonicalize`, and two spellings split
        # identity silently.
        if not os.path.isabs(value):
            raise RegistryError(
                f"registry entry {name!r} in {path} holds {value!r}, which is not "
                f"an absolute path. The registry stores absolute paths — write it "
                f"out in full (`~` is not expanded). Fix or remove {path}."
            )
    return data


def lookup(name: str, reg: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Resolves a project name to its registered workspace path.

    Args:
        name: The project name to look up, compared **exactly** (no folding, no
            normalization — the registry stores what was registered).
        reg: An already-loaded registry, to avoid re-reading the file when a
            caller resolves several names. Loaded on demand when omitted.

    Returns:
        The registered absolute path, or ``None`` if the name is not registered.

    Raises:
        RegistryError: If ``reg`` is omitted and the registry file is unusable.
    """
    return (load() if reg is None else reg).get(name)


def reverse_lookup(
    canonical_path: str, reg: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """Resolves a canonical workspace path back to the name registered for it.

    Two names may point at one path (a tolerated hand-edit — both reach the same
    workspace, so nothing can be corrupted). This answers with the **first** match
    in document order, an explicit rule rather than whichever one a
    path-keyed dict happened to keep.

    Args:
        canonical_path: A path already through :func:`canonicalize` — comparison
            is exact string equality, so a second spelling silently misses.
        reg: An already-loaded registry; loaded on demand when omitted.

    Returns:
        The first registered name pointing at that path, or ``None``.

    Raises:
        RegistryError: If ``reg`` is omitted and the registry file is unusable.
    """
    for name, path in (load() if reg is None else reg).items():
        if path == canonical_path:
            return name
    return None


def canonicalize(workspace_dir: str) -> str:
    """Reduces a workspace directory to the one canonical form the registry stores.

    ``os.path.realpath`` — absolute *and* symlink-resolved, which
    ``MitosConfig.workspace_dir`` (``abspath``) is not. This is the single
    canonicalization spelling for the value that crosses systems: registration,
    the path-uniqueness guard, and reverse lookup all compare paths produced here.
    A second spelling anywhere (``abspath``, ``normpath``, a trailing-slash strip)
    splits identity silently.

    Args:
        workspace_dir: Any spelling of the workspace directory.

    Returns:
        The canonical absolute path. The directory need not exist.
    """
    return os.path.realpath(workspace_dir)


def default_name(canonical_path: str) -> str:
    """Derives a project's default registry name from its canonical path.

    The basename of the **canonical** path, so a trailing slash and a symlinked
    route both register the real directory's name. Two ordinary inputs yield a
    name :func:`validate_name` refuses — a workspace at ``/`` (empty basename) and
    one in a dot-directory (``~/.config/x`` → ``.config``) — which is why the
    result is validated rather than trusted.

    Args:
        canonical_path: A path already through :func:`canonicalize`.

    Returns:
        The candidate name, unvalidated and un-normalized.
    """
    return os.path.basename(canonical_path)


def validate_name(name: str) -> None:
    """Checks that a name is a legal registry key, raising if it is not.

    Deliberately permissive: any non-empty name without a path separator, a
    leading ``.`` or ``~``, or a control character is legal — **including
    non-ASCII** (a ``ąžuolas/`` basename registers unchanged, P9), and nothing is
    normalized. The forbidden shapes are exactly the ones a *path* claims, so no
    string is ever both a legal name and a path — that disjointness is what lets a
    selector be told apart from a path by shape alone.

    Args:
        name: The candidate project name.

    Returns:
        None.

    Raises:
        RegistryError: If the name is empty, contains a path separator or a
            control character, or begins with ``.`` or ``~``. The message names
            the rule that failed and points at ``--name``.
    """
    recovery = (
        "pass `mitos init --name <other-name>` to register this workspace under a "
        "name of your choosing"
    )
    if not name:
        raise RegistryError(
            f"a project name cannot be empty (a workspace at a filesystem root "
            f"has no directory name to take) — {recovery}."
        )
    separators = [os.sep] + ([os.altsep] if os.altsep else [])
    for sep in separators:
        if sep in name:
            raise RegistryError(
                f"the project name {name!r} contains a path separator "
                f"({sep!r}) — a registry name is a plain label, never a path. "
                f"{recovery.capitalize()}."
            )
    if name[0] in (".", "~"):
        raise RegistryError(
            f"the project name {name!r} begins with {name[0]!r}, which a path "
            f"selector claims (a name and a path must stay tellable apart by "
            f"shape) — {recovery}."
        )
    for char in name:
        # Stricter than TOML, which tolerates a raw tab inside a string: a name is
        # a label a human types and reads back, so every control character is a
        # mistake rather than a spelling to preserve.
        if char < "\x20" or char == "\x7f":
            raise RegistryError(
                f"the project name {name!r} contains a control character "
                f"({char!r}) — {recovery}."
            )


def register(
    workspace_dir: str, name: Optional[str] = None, *, force: bool = False
) -> RegistrationOutcome:
    """Registers a workspace under a name, the registry's only writer.

    Canonicalizes the path, derives-or-validates the name, runs both guards, then
    performs one structure-preserving atomic write.

    **Guard order is load-bearing.** The path-uniqueness guard (unwaivable) is
    checked before the name-collision guard (waivable by ``force``). If both
    conditions hold and the waivable one were reported first, the caller would be
    sent to ``--force``, re-run, and hit the unwaivable one — a two-round-trip
    dead end. Reporting the unwaivable condition first names the real recovery on
    the first try. An exact re-assert (this name already points at this path)
    short-circuits ahead of both: it is neither collision, and in a registry
    carrying the tolerated two-names-one-path shape it must stay the no-op it is
    rather than reading as a fault.

    Args:
        workspace_dir: The workspace to register, in any spelling.
        name: The name to register it under; defaults to the canonical path's
            basename.
        force: Repoint ``name`` at this workspace when it is already registered
            elsewhere. It waives **only** the name collision — it never widens to
            path uniqueness, so a re-init in the wrong directory still stops.

    Returns:
        The :class:`RegistrationOutcome` describing what changed.

    Raises:
        RegistryError: On an unusable registry file (see :func:`load`), an illegal
            name, either guard firing, or an unwritable registry.
    """
    path = canonicalize(workspace_dir)
    # Validate the whole file BEFORE the write's line scan. A hand-written
    # `[table]` header decodes as one key (`tomllib.loads("[t]") == {"t": {}}`),
    # so a per-line key scan would read it as the definition of key `t` and
    # overwrite it. Loading first makes that state unreachable: a table is a
    # non-string value, so it raises before the scan is ever entered.
    reg = load()

    resolved = name if name is not None else default_name(path)
    validate_name(resolved)

    existing = reg.get(resolved)
    if existing == path:
        # Idempotent re-init. Rewrite the (identical) line anyway rather than
        # branching: one write path, and a hand-edited spelling of the same pair
        # converges on the canonical form.
        _write_entry(resolved, path)
        return RegistrationOutcome(name=resolved, path=path, action="reasserted")

    holder = reverse_lookup(path, reg)
    if holder is not None:
        raise RegistryError(
            f"this workspace ({path}) is already registered as {holder!r} in "
            f"{registry_path()}, so it will not also be registered as "
            f"{resolved!r}. Renaming a registration is a hand-edit of "
            f"{registry_path()} — `--force` deliberately does not do it, because "
            f"the shape it would silently accept is usually a re-init in the "
            f"wrong directory."
        )

    if existing is not None and not force:
        raise RegistryError(
            f"the project name {resolved!r} is already registered at {existing} "
            f"in {registry_path()}. Either `mitos init --force` to repoint "
            f"{resolved!r} at this workspace ({path}), or `mitos init --name "
            f"<other-name>` to register this workspace separately."
        )

    _write_entry(resolved, path)
    if existing is not None:
        return RegistrationOutcome(
            name=resolved, path=path, action="repointed", previous_path=existing
        )
    return RegistrationOutcome(name=resolved, path=path, action="created")


def _line_defines_key(line: str, name: str) -> bool:
    """Reports whether one raw registry line is the definition of ``name``.

    Key identification is delegated to ``tomllib`` — the line is handed to the
    same decoder that will read it back, so a quoted, escaped, or non-ASCII key
    matches without a hand-rolled unescaper of our own. Anything that is not a
    single-key assignment (a comment, a blank line, a fragment of a multi-line
    value) simply fails to decode and is left alone.

    The caller must have validated the whole file first; see :func:`register`.
    """
    candidate = line.strip()
    if not candidate or candidate.startswith("#") or "=" not in candidate:
        return False
    try:
        decoded = tomllib.loads(candidate)
    except tomllib.TOMLDecodeError:
        return False
    return list(decoded) == [name]


def _write_entry(name: str, path: str) -> None:
    """Writes one ``name = path`` entry, preserving every other byte of the file.

    Single-line surgery: the line defining ``name`` is replaced in place (keeping
    its own line ending), or the entry is appended when the name is new. Comments,
    blank lines, ordering, and a hand-editor's organization all survive — the file
    is meant to be edited by a human, so a full re-serialization would quietly
    destroy their work on every ``mitos init``.

    Raises:
        RegistryError: If the registry cannot be written.
    """
    target = registry_path()
    try:
        # Both sides go through the shared serializer: the key because a bare name
        # with a dot would parse as a nested table (and a bare non-ASCII name not
        # at all), the value because a path's backslashes must survive the read.
        entry = f"{toml_scalar(name)} = {toml_scalar(path)}"
    except TypeError as e:  # pragma: no cover - unreachable for two str inputs
        raise RegistryError(
            f"cannot write the registry entry for {name!r} → {path!r}: the value "
            f"has no TOML representation ({e}). Register this workspace under a "
            f"different name with `mitos init --name <other-name>`."
        ) from e

    try:
        with open(target, "r", encoding="utf-8") as f:
            existing = f.read()
    except FileNotFoundError:
        existing = ""
    except OSError as e:
        raise RegistryError(_unwritable_message(target, e)) from e

    lines: List[str] = existing.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if _line_defines_key(line, name):
            ending = line[len(line.rstrip("\r\n")):] or "\n"
            lines[index] = entry + ending
            break
    else:
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines[-1] += "\n"
        lines.append(entry + "\n")

    content = "".join(lines)
    _verify_reads_back(target, content, name, path)
    _atomic_write(target, content)


def _verify_reads_back(target: str, content: str, name: str, path: str) -> None:
    """Refuses to write a file that would not read back as the entry just composed.

    A postcondition on the line surgery, checked *before* the replace so a failure
    leaves the existing registry untouched. Two states reach it, both from a
    hand-edit the per-line scan cannot safely amend:

    * a value written as a **multi-line** TOML string — the scan sees neither half
      as a key definition, so the entry would be *appended*, leaving two
      definitions of one name and a registry that raises on its very next read;
    * a workspace path carrying bytes that are not valid UTF-8 (legal on Linux),
      which has no representation in the file at all.

    Without this, both end the same way: ``init`` reports success and the registry
    is broken afterwards. With it, the failure is named at the moment it happens
    and the file is as it was.

    Raises:
        RegistryError: If the composed file is not encodable, does not parse, or
            does not yield ``name → path``.
    """
    try:
        content.encode("utf-8")
    except UnicodeEncodeError as e:
        raise RegistryError(
            f"the workspace path {path!r} contains bytes that are not valid UTF-8, "
            f"so it cannot be written to {target} ({e}). Rename the directory, or "
            f"register a parent directory instead."
        ) from e

    try:
        parsed = tomllib.loads(content)
    except tomllib.TOMLDecodeError as e:
        raise RegistryError(
            f"updating the entry for {name!r} would leave {target} unparseable "
            f"({e}) — the existing entry is written in a shape this single-line "
            f"update cannot amend (a multi-line value, most likely). Nothing was "
            f"written; simplify that entry to one `\"name\" = \"/path\"` line by "
            f"hand and re-run."
        ) from e

    if parsed.get(name) != path:
        raise RegistryError(
            f"updating the entry for {name!r} in {target} would not read back as "
            f"the path it was given ({path}) — nothing was written. Simplify that "
            f"entry to one `\"name\" = \"/path\"` line by hand and re-run."
        )


def _unwritable_message(target: str, cause: OSError) -> str:
    """Builds the R6 message: what failed, and that the workspace itself is fine."""
    return (
        f"cannot write the Mitos project registry at {target}: {cause}. The "
        f"workspace is initialized but unregistered — it stays fully usable by "
        f"its path; re-run `mitos init` once the registry is writable to register "
        f"it."
    )


def _atomic_write(target: str, content: str) -> None:
    """Replaces the registry file atomically, so no reader ever sees a partial one.

    Writes a temp file in the **same directory** (same filesystem, so the replace
    is atomic) and ``os.replace``s it over the target: a concurrent reader sees
    either the whole old file or the whole new one. Mirrors
    ``renderer.atomic_write``, reimplemented rather than imported — that module
    imports ``mitos.store``, and pulling it in here would break this leaf's tier
    on day one. The ``fsync`` is the one addition: cheap durability on a
    once-per-``init`` write.

    Raises:
        RegistryError: If the directory cannot be created or the write/replace
            fails. Any non-``OSError`` propagates unchanged, but the temp file is
            cleaned up either way.
    """
    directory = os.path.dirname(target)
    try:
        # The config dir may not exist yet — the global `.env` path only ever
        # reads, so `init` on a fresh machine is the first thing to need it.
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(dir=directory, prefix=".registry-", suffix=".tmp")
    except OSError as e:
        raise RegistryError(_unwritable_message(target, e)) from e

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target)
    except OSError as e:
        _remove_quietly(temp_path)
        raise RegistryError(_unwritable_message(target, e)) from e
    except BaseException:
        _remove_quietly(temp_path)
        raise


def _remove_quietly(path: str) -> None:
    """Deletes a temp file, ignoring its absence — used on the failure path only."""
    try:
        os.remove(path)
    except OSError:
        pass
