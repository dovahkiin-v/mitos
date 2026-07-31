"""Drives a real ``mitos serve`` subprocess over JSON-RPC through its own pipes.

Every other test in this suite talks to mitos by *importing* it. That is right for
a graph, a parser, a resolver — and structurally blind for two hazards, because
both live outside the function call:

1. An in-process tool call never enters ``cli.main()``, so it cannot observe what
   the **entry path** does to the process (the two ``load_dotenv_file`` calls that
   put a project's keys into ``os.environ``).
2. An in-process tool call shares pytest's ``os.environ``, so it cannot observe
   **process-owned environment** at all.

This module is the instrument that watches mitos from the outside, as a process,
the way a real agent's MCP client does. Import it by bare name — ``tests/`` is on
``sys.path`` under pytest's default prepend mode (there is no ``tests/__init__.py``);
``live_helpers`` and ``conftest`` are the in-tree precedents.

Usage — one session, many calls, a filesystem mutation permitted between them::

    async with mitos_server(cwd=workspace, env=harness_env(tmp_path)) as server:
        result = await server.session.call_tool(
            "list_scopes", {"project": str(workspace)})
        payload = json.loads(result.content[0].text)

Two properties are not negotiable, and each exists because a measurement said so:

**The child's environment is declared, never inherited.** ``stdio_client`` does
*not* hand the child ``os.environ``. It hands it
``{**get_default_environment(), **params.env}`` — and ``get_default_environment()``
is exactly six names on POSIX (``HOME``, ``LOGNAME``, ``PATH``, ``SHELL``, ``TERM``,
``USER``). ``XDG_CONFIG_HOME`` is **not** among them while ``HOME`` **is**, so
``tests/conftest.py``'s autouse redirect does not reach the child and
``config_home()`` falls back to the *developer's real* ``~/.config/mitos``. A
harness written like the tree's other subprocess-driving modules — which build
``env`` from ``os.environ`` and get the redirect for free — would read the real
global ``.env`` and write the real ``registry.toml``, and would look identical in
review. ``harness_env`` is the answer; ``env`` is a required keyword with no
inherit-the-parent default. Note the merge is one-way: the six inherited names can
be *overridden* by naming them, but never *removed*, so no child launched through
this transport has a truly empty environment.

**A server that cannot start renders its stderr.** Left alone, a child that dies
before the handshake surfaces as a nested anyio ``ExceptionGroup`` — "unhandled
errors in a TaskGroup (1 sub-exception)" wrapping another of the same — and the
actual cause appears at *no* depth of that chain. It exists only in the child's
stderr. So the child's stderr goes to a file, the handshake is wrapped, and the
failure is a ``ServerStartupError`` carrying the command, the cwd, and that text.
A red row here is a vector, not a wall (P3), including for our own tools. The same
wrapper sits on the *common* path — anyio wraps a caller's own failed assertion in
one ``BaseExceptionGroup`` per task group crossed, twice — so a single-error group
is collapsed back to its cause on the way out (``_unwrap_transport_group``), and a
cancellation or an interrupt is never re-labelled as a startup failure.

This module imports ``mcp`` and the standard library and **nothing from mitos**.
The server under test is a separate interpreter that imports mitos for real; a
driver that imported it would re-create in miniature the in-process coupling this
harness exists to escape.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import AsyncIterator, Dict, Optional, Sequence, Tuple, Union

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

#: The source under test, launched through the venv interpreter running pytest.
#: Never ``shutil.which("mitos")`` — that resolves the *pipx* build, which is a
#: different tree at a different commit, and which cannot start the server at all
#: from a fresh install (WIRING_LEDGER entry-001: ``mcp>=1.26.0`` is unceilinged
#: and ``mcp`` 2.0.0 removed ``mcp.server.fastmcp``). The dev venv is healthy only
#: because it is pinned by age. A green run here says nothing about a real install.
SERVE_ARGS: Tuple[str, ...] = ("-m", "mitos.cli", "serve")

#: Failures the harness must never re-label as a startup diagnosis. An interrupt
#: stops a run; a cancellation belongs to whoever asked for it (an outer
#: ``asyncio.wait_for``, a pytest timeout, a sibling task) and must reach them as
#: itself, or their timeout reports this harness's error instead of their own.
_NEVER_A_STARTUP_FAILURE = (KeyboardInterrupt, SystemExit, asyncio.CancelledError)


class ServerStartupError(RuntimeError):
    """Raised when the child never completed the MCP handshake.

    Attributes:
        command: The command line the harness launched, as a single string.
        cwd: The directory the child was launched from.
        stderr: Whatever the child wrote to stderr before dying — the only place
            the real cause appears.
    """

    def __init__(self, message: str, *, command: str, cwd: str, stderr: str) -> None:
        super().__init__(message)
        self.command = command
        self.cwd = cwd
        self.stderr = stderr


def harness_env(
    root: Union[str, Path], *, extra: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Builds the DECLARED environment for a harness child. Nothing is implicit.

    Every name the child needs is stated here. The config root, the cache root and
    ``HOME`` are all redirected under ``root`` (give it a ``tmp_path``), which is
    what keeps a subprocess out of the developer's real ``~/.config/mitos``.
    Redirecting ``HOME`` as well as the two ``XDG_*`` names is belt-and-braces:
    ``config_home()`` reads ``XDG_CONFIG_HOME`` *or* falls back to
    ``expanduser("~")``, so either name alone would do, and missing both is the
    escape.

    No API key is declared, and ``QDRANT_URL`` is left unset — a keyless child on
    the default Qdrant URL is the honest default posture, since the tools that
    need neither must keep working without them. Pass ``extra`` to add a key, to
    point the child at a dead port, or to plant a sentinel a test then looks for.

    Args:
        root: Directory to root the child's ``HOME``/config/cache under.
        extra: Additional names, merged last so a caller can override any default.

    Returns:
        A fresh dict; callers may mutate it without affecting later calls.
    """
    home = Path(root) / "home"
    env = {
        # PATH is one of the six the transport inherits anyway; naming it keeps
        # this dict a complete statement of the child's environment rather than a
        # patch over an invisible base.
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "MITOS_NO_UPDATE_CHECK": "1",
        "MITOS_NO_MCP_HINT": "1",
    }
    if extra:
        env.update(extra)
    return env


@dataclass
class MitosServer:
    """A live, initialized session over a running ``mitos serve`` child.

    Attributes:
        session: The initialized ``ClientSession``. Call tools through it; never
            write to the child's stdin directly.
        stderr_path: The file the child's stderr is piped to. Readable while the
            session is live — FastMCP logs one line per tools request there, so a
            healthy run's stderr is **not** empty and must never back an emptiness
            assertion.
    """

    session: ClientSession
    stderr_path: Path

    def stderr_text(self) -> str:
        """Returns whatever the child has written to stderr so far.

        Read from disk rather than through a second open handle, and decoded with
        ``errors="replace"`` so rendering a failure can never itself raise on a
        child that emitted non-UTF-8.
        """
        return _read_stderr(self.stderr_path)


@asynccontextmanager
async def mitos_server(
    *,
    cwd: Union[str, Path],
    env: Dict[str, str],
    args: Sequence[str] = SERVE_ARGS,
    deadline: float = 30.0,
) -> AsyncIterator[MitosServer]:
    """Yields an INITIALIZED session over a real ``mitos serve`` subprocess.

    The context yields only after ``session.initialize()`` returns: the handshake
    *is* the readiness gate, so there is no sleep and no polling loop. Teardown is
    the SDK's own — close stdin, wait, escalate SIGTERM→SIGKILL — under an
    ``async with … process``, which makes a zombie structurally impossible. This
    function's job is to not defeat that: it swallows nothing around the exit and
    holds no stream past the context.

    Args:
        cwd: Directory to launch the child from. Required, and never implicit: the
            launch directory *is* the workspace a server binds today, so it is
            part of the topology under test, not an incidental detail.
        env: The child's declared environment — see :func:`harness_env`. Required;
            there is deliberately no inherit-the-parent default.
        args: Interpreter arguments. Defaults to :data:`SERVE_ARGS`; override it
            to drive some other MCP server (or a deliberately dead one).
        deadline: Per-request read timeout in seconds, handshake included. Sized
            for a loaded CI runner, not for this box: a whole local smoke —
            interpreter start, ``mitos init``, handshake and one call — measures
            about 4s, and ``import mitos.cli`` drags the ``anthropic`` SDK at
            module scope, so a cold, contended runner has real headroom to use.

    Yields:
        A :class:`MitosServer` handle.

    Raises:
        ServerStartupError: If the child failed to start, died, or hung before
            completing the handshake. The message carries the command, the cwd and
            the child's stderr. Anything raised *after* a successful handshake —
            including from the caller's own body and from teardown — arrives as
            **itself**, not as the transport's wrapper: see
            :func:`_unwrap_transport_group`. An interrupt or a cancellation is
            never re-labelled as a startup failure, at any point.
    """
    params = StdioServerParameters(
        command=sys.executable,
        args=list(args),
        cwd=str(cwd),
        env=dict(env),
    )
    stderr_path = _new_stderr_log(env=env, cwd=cwd)

    started = False
    # The errlog handle is opened OUTSIDE the transport context and closed after
    # it, because `stdio_client` takes an already-open TextIO and never owns its
    # lifetime. A handle leaked here surfaces as an `unclosed file`
    # ResourceWarning — which is exactly the warning class the teardown row
    # watches for, so a mismanaged log reds the row written to protect it.
    with stderr_path.open("w", encoding="utf-8") as errlog:
        try:
            async with stdio_client(params, errlog=errlog) as (read, write):
                async with ClientSession(
                    read, write, read_timeout_seconds=timedelta(seconds=deadline)
                ) as session:
                    await session.initialize()
                    started = True
                    yield MitosServer(session=session, stderr_path=stderr_path)
        except BaseException as exc:
            cause = _unwrap_transport_group(exc)
            # `started` is the whole guard: once the handshake has landed, every
            # later failure is the caller's or the transport's and must arrive
            # unmodified. Without it, a plain assertion error in the caller's body
            # would be re-labelled as a startup failure. The second clause covers
            # the failures that are never a startup diagnosis even *before* the
            # handshake lands — a Ctrl-C during it, or an outer timeout cancelling
            # the task that is waiting on it.
            if not started and not isinstance(cause, _NEVER_A_STARTUP_FAILURE):
                raise _startup_error(params, stderr_path, deadline) from exc
            if cause is exc:
                raise
            context, suppress = cause.__context__, cause.__suppress_context__
            try:
                raise cause
            finally:
                # Raising from inside an active handler chains the transport's
                # wrapper onto the cause as its `__context__`, which puts the
                # ExceptionGroup back in the rendered traceback by the back door.
                # Restore the pair the caller's own exception actually carried.
                cause.__context__ = context
                cause.__suppress_context__ = suppress


def _unwrap_transport_group(exc: BaseException) -> BaseException:
    """Collapses the transport's single-error exception groups back to the cause.

    ``stdio_client`` and ``ClientSession`` each run an anyio task group, and anyio
    wraps whatever crossed one in a ``BaseExceptionGroup`` — twice, once per group.
    So a plain ``assert`` in a caller's body arrives as
    ``ExceptionGroup([ExceptionGroup([AssertionError])])``: ``pytest.raises`` stops
    matching it, and a caller's own failed assertion reads as a transport crash.
    That is the same opaque wall this harness replaces on the startup path, met on
    the *common* path, so it is collapsed here too.

    Only a group holding exactly one exception is unwrapped, at any depth. A
    genuinely multi-error group is left intact: there the grouping *is* the
    information, and a harness that discarded it would be hiding a second failure.

    Args:
        exc: The exception the transport raised.

    Returns:
        The single leaf of a chain of one-error groups, or ``exc`` unchanged.
    """
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


def _new_stderr_log(*, env: Dict[str, str], cwd: Union[str, Path]) -> Path:
    """Creates an empty, uniquely-named file for one child's stderr.

    Homed under the child's declared ``HOME``, which :func:`harness_env` puts
    inside the caller's ``tmp_path``, so the log never lands in the workspace a
    test is asserting about. A caller who declares no ``HOME`` gets it under the
    launch directory instead — which *is* that workspace, so declare one.
    Unique per launch, so several sessions in one test do not overwrite each
    other's diagnostics.
    """
    log_dir = Path(env.get("HOME") or cwd) / ".mitos-harness"
    log_dir.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix="serve-stderr-", suffix=".log", dir=str(log_dir))
    os.close(handle)
    return Path(name)


def _read_stderr(path: Path) -> str:
    """Reads a child's stderr log, tolerating a missing file and bad bytes."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _startup_error(
    params: StdioServerParameters, stderr_path: Path, deadline: float
) -> ServerStartupError:
    """Builds the named startup failure, with the child's stderr as the body."""
    command = " ".join([params.command, *params.args])
    cwd = str(params.cwd)
    stderr = _read_stderr(stderr_path)
    body = stderr.strip() or "(the child wrote nothing to stderr)"
    message = (
        f"the MCP handshake never completed within {deadline}s — the server "
        "process failed to start, died, or hung before answering.\n"
        f"  command: {command}\n"
        f"  cwd:     {cwd}\n"
        f"  stderr:  {body}\n"
        "The stderr above is the cause; the anyio ExceptionGroup this replaces "
        "carries it at no depth."
    )
    return ServerStartupError(message, command=command, cwd=cwd, stderr=stderr)
