"""The global project overview — a bounded, isolated sweep of the registry.

``mitos status`` answers *"is this one workspace ready?"* about whatever directory
the process is standing in. This module builds the other answer: **what does this
machine have, and is any of it broken?** — every registered project, its local
reality check, and whether its derived Qdrant collection actually exists. One
payload, JSON-native, composed in exactly one place (:func:`build_overview`'s tail)
so the ``--json`` shape and the text table can never disagree about what was found.

Three properties are load-bearing, and each is a decision rather than a detail.

**Per-entry isolation.** No single registry entry can end the sweep. An entry whose
probe *raises* — a malformed target ``config.toml``, an ``OSError`` from a symlink
loop — renders as that entry's own state and the iteration continues; an entry that
never answers renders ``unresponsive`` and the rest are still fully reported. The
value of this surface is the *set*, so an exception that takes the table down costs
more than the entry that caused it. ``unresponsive`` is deliberately **not**
``missing``: a hung mount is not a deleted directory, and telling someone their work
is gone when the disk is merely asleep is the kind of small cruelty good
infrastructure does not commit.

**A wall-clock bound that bounds the *process*, not just the sweep.** One
``threading.Thread(daemon=True)`` per probe, all started, then joined against a
single shared deadline. The obvious primitive is the trap: ``ThreadPoolExecutor``
with ``future.result(timeout=…)`` bounds the *sweep* and not the process, because
``concurrent.futures.thread`` registers an atexit hook that joins every worker
thread — measured on python 3.13.5, a 6-second parked probe cost **6.03 s** of total
process wall behind a 0.5 s bound, versus **0.53 s** for the daemon thread. On a CLI
that is the same hang wearing a hat: the table renders and the shell does not get its
prompt back, and it surfaces only against the unresponsive mount the bound exists for.
The deadline is **shared, not per-entry**: per-entry × N is O(N × budget), so twenty
projects behind a dead NAS would take twenty times too long. Two consequences follow
and must survive later edits — a probe body must stay **read-only and lock-free** (a
daemon thread can be killed mid-call at interpreter finalization, so nothing it does
may need to *complete*), and the ceiling is one thread per entry. At the realistic
scale — tens — that is free in a one-shot process; past a few hundred the fix is a
bounded number of *rounds*, each keeping its own share of the budget, which preserves
per-entry isolation because a parked entry only ever costs its own batch's remainder.

**The Qdrant check is instance-shaped, never per-project.** One
``GET /collections`` per **distinct** resolved URL answers both the reachability
verdict and the collection listing; each project's presence is then a local set
membership test. The load-bearing reason is the **call budget** (and the wall-clock
bound on a from-anywhere habit), *not* the create-on-construct hazard that reads
already lost — a reader meeting this after that hazard was removed could reasonably
conclude the constraint lapsed, and it has not. Nor is a single instance assumed:
skipping non-localhost URLs would hand the least honest answer to exactly the instance
most likely to be down.

What this module deliberately does **not** do: read a graph, open SQLite, or count
points. Presence never implies population, and the durable completeness check is the
deep report's active-node/point-id diff. Keeping the graph out is what holds the
per-entry cost at three stats plus a config read, and it is why a flagged project gets
a pointer at ``mitos reconcile`` rather than a priced *"K of N will re-embed"* — that
prescription needs per-project local reads this sweep refuses to do.

Composition locus: **typed data here, wording at the surface.** Every user-facing
sentence — including the one that words a timeout — lives in ``cli.py``.

Tier 2, permanently: stdlib plus ``mitos.registry``, ``mitos.routing``,
``mitos.config``, ``mitos.errors`` and ``mitos.vector_store``, and nothing else, ever.
``cli`` imports this module (so an import back would cycle *and* drag the Anthropic
SDK, which ``import mitos.cli`` carries at module scope), and ``store``/``sync`` are
the graph reads above. Enforced by a subprocess import-closure probe, not by prose.
It never reads the process's working directory: the cwd marker's directory arrives as
an **argument**, so the sweep structurally has no cwd branch.
"""

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

from mitos import registry, routing
from mitos.config import MitosConfig
from mitos.errors import VectorStoreError, VectorStoreUnreachableError
from mitos.vector_store import list_collection_names

#: Seconds for the **whole** local phase, not per entry — every entry's probe runs
#: concurrently against this one deadline, so the total is the bound no matter how
#: many entries park. A healthy probe (three stats plus a config read) measures
#: 0.08 ms on this box, so this is roughly four orders of magnitude of headroom for
#: a loaded machine or a cold page cache, while still reading as instant.
LOCAL_PROBE_BUDGET = 1.5

#: Seconds for one ``requests``-level instance probe. Reuses the number
#: ``cli._check_qdrant`` already ships rather than minting a second one.
INSTANCE_SOCKET_TIMEOUT = 3.0

#: Seconds for the whole instance phase. **Strictly greater than the socket
#: timeout**, and pinned as an inequality: the outer budget is the backstop for what
#: a per-socket-op timeout does not bound (a slow-trickle response), so an outer
#: budget at or below it would make the socket timeout dead code.
INSTANCE_PROBE_BUDGET = 4.0

STATE_OK = "ok"
STATE_MISSING = "missing"
STATE_NOT_A_WORKSPACE = "not_a_workspace"
STATE_UNRESPONSIVE = "unresponsive"
STATE_ERROR = "error"

#: The closed vocabulary a registry entry's ``state`` is drawn from. A single
#: discriminator rather than the two booleans "path exists" + "workspace valid",
#: because ``unresponsive`` and ``error`` are *unknown* rather than ``False`` and two
#: booleans cannot say so without inventing a null-means-what convention.
#:
#: **Precedence, highest first:** ``unresponsive`` (the probe never answered, so no
#: verdict of its own exists) → ``error`` (it raised) → ``missing`` (nothing at the
#: path) → ``not_a_workspace`` (something is there and the validity triple fails) →
#: ``ok``. The first two are decided by the harness, the last three inside
#: :func:`probe_entry`, and the order is what keeps *unknown* from ever being
#: rendered as a finding.
ENTRY_STATES = frozenset(
    {STATE_OK, STATE_MISSING, STATE_NOT_A_WORKSPACE, STATE_UNRESPONSIVE, STATE_ERROR}
)


@dataclass
class _Outcome:
    """What one bounded call produced — internal, never crosses the JSON boundary.

    Attributes:
        finished: True once the worker ran to completion (with a value or an
            exception). False means the deadline passed first.
        value: The call's return value.
        error: The exception it raised, if any.
    """

    finished: bool = False
    value: Any = None
    error: Optional[BaseException] = None


def probe_entry(name: str, path: str) -> Dict[str, Any]:
    """Checks one registered project locally, and reports what it found.

    A **module-level named function**, not a closure inside
    :func:`build_overview`: it is the seam the isolation and bounding rows patch
    (raise from one, park one), and a nested closure is unpatchable — which is how a
    bound ships unproven.

    It may raise, and that is the contract: the harness classifies any escape as
    this entry's own ``error`` state. Raising here rather than catching inside keeps
    exactly one catch site for the property, so the fault-injection row that lets an
    exception escape reds the isolation row rather than nothing.

    **Read-only and lock-free**, permanently — it runs on a daemon thread that may be
    killed mid-call at interpreter finalization, so nothing it does may need to
    complete. No write, no lock, no SQLite open belongs here.

    A permission fault reads as ``not_a_workspace`` rather than an error, because
    ``os.path.isdir`` returns False under an unreadable parent instead of raising.
    That is deliberate and shipped (``routing.is_workspace`` documents it); the
    message names the path, so the operator still sees the subject.

    Args:
        name: The registered name. Unused by the body — carried so a test can raise
            or park for one *named* entry, which is what makes the isolation rows
            per-entry rather than global.
        path: The registered absolute workspace path.

    Returns:
        ``{"state", "qdrant_url", "collection"}`` — the last two are ``None`` for
        every state but ``ok``, because nothing was constructed to resolve them.

    Raises:
        ConfigError: If the target's ``config.toml`` is malformed or invalid.
        OSError: If a filesystem probe fails in a way the stdlib does not swallow.
    """
    if not os.path.exists(path):
        return {"state": STATE_MISSING, "qdrant_url": None, "collection": None}
    if not routing.is_workspace(path):
        return {"state": STATE_NOT_A_WORKSPACE, "qdrant_url": None, "collection": None}
    # A real MitosConfig, never a re-spelling: the resolved QDRANT_URL is a five-rung
    # ladder with exactly one implementation, and the collection name is derived from
    # the canonical path — a name assembled here from the basename would silently
    # disagree with the derivation for every project, which is the entire point of it.
    # (Spelled without the literal so the tree-wide grep for hand-built collection
    # names keeps returning only the derivation itself, `config.default_collection_name`
    # — a prose hit in a module that must never build one reads as a finding.) `project=`
    # is deliberately not passed: this response answers for the machine and echoes no
    # single corpus.
    config = MitosConfig(path)
    return {
        "state": STATE_OK,
        "qdrant_url": config.qdrant_url,
        "collection": config.qdrant_collection,
    }


def probe_instance(url: str) -> Dict[str, Any]:
    """Probes one Qdrant instance once, for both reachability and its listing.

    Module-level for the same reason as :func:`probe_entry` — it is a patch seam.

    The tri-state is the point, and its middle row is not decoration: *answered with
    something unusable* must not be read as "your collection is missing", so an
    unavailable listing makes every project on this instance ``collection_present:
    null``, never ``false``.

    ==================================================  =========  =================
    observation                                         reachable  listing_available
    ==================================================  =========  =================
    no answer (refused, DNS, socket timeout)            False      False
    answered, but not usably (500, auth wall, garbage)  True       False
    answered with the listing                           True       True
    ==================================================  =========  =================

    Args:
        url: The resolved Qdrant base URL.

    Returns:
        ``{"reachable", "listing_available", "names", "error"}``. ``names`` is a
        ``set`` and is **internal** — it is another project's data and can be
        hundreds of entries, so only a count crosses the payload boundary.
    """
    try:
        names = list_collection_names(url, timeout=INSTANCE_SOCKET_TIMEOUT)
    except VectorStoreUnreachableError as e:
        # Ordered above its base class on purpose: this is the arm that means the
        # instance said nothing at all.
        return {
            "reachable": False,
            "listing_available": False,
            "names": None,
            "error": str(e),
        }
    except VectorStoreError as e:
        return {
            "reachable": True,
            "listing_available": False,
            "names": None,
            "error": str(e),
        }
    return {
        "reachable": True,
        "listing_available": True,
        "names": names,
        "error": None,
    }


def _run_bounded(call: Callable[[], Any], outcome: _Outcome) -> None:
    """Runs one call on its own thread, recording whatever it did.

    ``BaseException``, not ``Exception``: an escape from a worker thread is otherwise
    printed to stderr and lost, and the entry would then read ``unresponsive`` — a
    lie about a probe that answered. ``finished`` is set **last** so a reader seeing
    it True also sees the value or the error.
    """
    try:
        outcome.value = call()
    except BaseException as exc:
        # Deliberately the widest catch in the tree, because isolation IS the
        # property: anything narrower lets a surprise escape into a thread, where it
        # is printed to stderr and lost, and the entry then reads as unresponsive.
        outcome.error = exc
    finally:
        outcome.finished = True


def _fan_out(calls: List[Callable[[], Any]], budget: float) -> List[_Outcome]:
    """Runs every call concurrently against ONE shared deadline.

    Daemon threads, deliberately — see the module docstring for the measurement that
    rules out ``ThreadPoolExecutor``. A thread still alive when the deadline passes
    is abandoned and dies with the process; its outcome stays unfinished, which the
    caller renders as *unknown*.

    Args:
        calls: Zero or more no-argument callables. An empty list does nothing at
            all — a registry of nothing but broken entries issues zero probes.
        budget: Seconds for the whole phase, not per call.

    Returns:
        One :class:`_Outcome` per call, positionally.
    """
    outcomes = [_Outcome() for _ in calls]
    threads = [
        threading.Thread(target=_run_bounded, args=(call, outcome), daemon=True)
        for call, outcome in zip(calls, outcomes)
    ]
    for thread in threads:
        thread.start()
    # Computed AFTER the starts, once: the deadline bounds the phase from the moment
    # the work is in flight, and a per-thread deadline would make the total O(N).
    deadline = time.monotonic() + budget
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    return outcomes


def _error_text(exc: BaseException) -> str:
    """The cause as one line — never a traceback, never an empty string."""
    return str(exc) or exc.__class__.__name__


def _classify_entry(outcome: _Outcome) -> Dict[str, Any]:
    """Turns one local outcome into an entry's ``state`` / ``error`` pair."""
    if not outcome.finished:
        # No prose here: the leaf carries typed data and each surface words the
        # timeout itself.
        return {
            "state": STATE_UNRESPONSIVE,
            "error": None,
            "qdrant_url": None,
            "collection": None,
        }
    if outcome.error is not None:
        return {
            "state": STATE_ERROR,
            "error": _error_text(outcome.error),
            "qdrant_url": None,
            "collection": None,
        }
    return dict(outcome.value, error=None)


def _classify_instance(outcome: _Outcome) -> Dict[str, Any]:
    """Turns one instance outcome into the tri-state, timeouts included."""
    if not outcome.finished:
        return {
            "reachable": False,
            "listing_available": False,
            "names": None,
            "error": None,
        }
    if outcome.error is not None:
        # An unexpected escape says nothing about the instance's state, so it
        # degrades to the least-claiming row rather than to "up but unusable".
        return {
            "reachable": False,
            "listing_available": False,
            "names": None,
            "error": _error_text(outcome.error),
        }
    return outcome.value


def build_overview(*, cwd: Optional[str] = None) -> Dict[str, Any]:
    """Sweeps the project registry and returns the JSON-native overview payload.

    The whole sweep in three phases: the registry read, one bounded round of local
    per-entry probes, then one bounded round of instance probes over the **distinct**
    resolved URLs those probes reported. Everything after that is local computation.

    ``registry.load()`` is deliberately **outside** the bounded phase. It is one small
    file under the config home, and if *that* read fails there is nothing to render —
    it is the exit contract's only non-zero path, not an entry whose failure the sweep
    should survive.

    Args:
        cwd: The caller's working directory, read at the boundary and passed in, so
            this module structurally has no cwd branch. ``None`` — including when the
            boundary's own read failed — simply means no ``cwd_project`` marker.

    Returns:
        ``{"report": "overview", "registry_path", "count", "cwd_project",
        "projects": [...], "instances": [...]}``. ``report`` announces which payload
        this is, because one ``--json`` flag will return two of them and a consumer
        must not have to sniff a shape. Per project:
        ``{"name", "path", "state", "error", "shares_path_with", "qdrant_url",
        "collection", "collection_present"}`` in registry **document order** — the
        order a reverse lookup resolves its first match in, so the listing shows what
        actually decides. Per instance:
        ``{"url", "reachable", "listing_available", "collection_count", "error"}`` in
        first-appearance order. Tri-states are ``null``, never a sentinel string; no
        set and no tuple crosses the boundary.

    Raises:
        RegistryError: If the registry file exists and is unusable.
    """
    reg = registry.load()

    # A lambda, never `functools.partial`: partial binds the function object at build
    # time, so a test patching `overview.probe_entry` would silently drive the
    # original and the isolation/bounding rows would prove nothing. The global lookup
    # happens when the thread calls it.
    probes = [
        _classify_entry(outcome)
        for outcome in _fan_out(
            [lambda n=name, p=path: probe_entry(n, p) for name, path in reg.items()],
            LOCAL_PROBE_BUDGET,
        )
    ]

    # The distinct set, in first-appearance order. An entry that yielded no URL —
    # anything but `ok` — contributes nothing here, so a registry of nothing but
    # broken entries issues zero network calls.
    urls: List[str] = []
    for probe in probes:
        if probe["qdrant_url"] is not None and probe["qdrant_url"] not in urls:
            urls.append(probe["qdrant_url"])
    instance_outcomes = _fan_out(
        [lambda u=url: probe_instance(u) for url in urls], INSTANCE_PROBE_BUDGET
    )
    instances = {
        url: _classify_instance(outcome)
        for url, outcome in zip(urls, instance_outcomes)
    }

    # Two names for one path is a tolerated hand-edit, never a fault: both reach the
    # same workspace, so nothing can be corrupted, and it must not change either
    # entry's state. Grouping the loaded map is pure local computation — no second
    # registry read, and it gives the same first-in-document-order answer a reverse
    # lookup would. Grouped on the **stored string**, deliberately not canonicalized:
    # `reverse_lookup` compares exact strings too, so this reports a finding about
    # what actually decides. Canonicalizing here would group `/a` with `/a/` and then
    # name a deciding entry the reverse lookup does not honour — a truer-looking
    # finding that is wrong about the only thing it is for.
    names_by_path: Dict[str, List[str]] = {}
    for name, path in reg.items():
        names_by_path.setdefault(path, []).append(name)

    # The payload is composed HERE and only here, so the two branches of the surface
    # cannot fork its shape.
    projects: List[Dict[str, Any]] = []
    for probe, (name, path) in zip(probes, reg.items()):
        instance = (
            instances.get(probe["qdrant_url"])
            if probe["qdrant_url"] is not None
            else None
        )
        present: Optional[bool] = None
        if (
            probe["collection"] is not None
            and instance is not None
            and instance["listing_available"]
        ):
            names: Set[str] = instance["names"]
            present = probe["collection"] in names
        projects.append(
            {
                "name": name,
                "path": path,
                "state": probe["state"],
                "error": probe["error"],
                "shares_path_with": [
                    other for other in names_by_path[path] if other != name
                ],
                "qdrant_url": probe["qdrant_url"],
                "collection": probe["collection"],
                "collection_present": present,
            }
        )

    return {
        "report": "overview",
        "registry_path": registry.registry_path(),
        "count": len(projects),
        "cwd_project": (
            routing.nearest_registered_ancestor(cwd, reg) if cwd is not None else None
        ),
        "projects": projects,
        "instances": [
            {
                "url": url,
                "reachable": instance["reachable"],
                "listing_available": instance["listing_available"],
                "collection_count": (
                    len(instance["names"]) if instance["names"] is not None else None
                ),
                "error": instance["error"],
            }
            for url, instance in instances.items()
        ],
    }
