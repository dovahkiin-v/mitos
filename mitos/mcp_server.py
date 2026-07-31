"""MCP Server implementation for Mitos.

This module implements the MCP Server (F) and the C4 integration contract,
exposing surface_decisions and query_decisions tools to LLM clients.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
from mcp.server.fastmcp import FastMCP

from mitos import registry, routing
from mitos.display import blackout_note, clamp_limit, dumps_display, letter_payload, oneline_payload, order_scope_counts, projects_payload, show_payload, SHOW_NOT_FOUND_HINT
from mitos.config import MitosConfig
from mitos.store import GraphStore, MODIFIER_EDGE_KEYS
from mitos.embeddings import GeminiEmbeddingProvider
from mitos.models import get_embedding_model_id
from mitos.vector_store import QdrantVectorStore
from mitos.errors import (
    CollectionMissingError,
    ProjectTargetingError,
    TARGET_EXEMPT_VERB,
    TARGET_MISSING,
    TARGET_PATH_NOT_A_WORKSPACE,
    TARGET_RELATIVE_PATH,
    TARGET_UNKNOWN_NAME,
)
from mitos.lexical import degraded_reason_from_error, lexical_fallback
from mitos.parser import corpus_has_entries
from mitos.recall import (assess_surface_recall, corpus_provenance,
                          missing_graph_is_a_gap, missing_graph_note,
                          missing_index_is_a_gap, scope_filter_recovery)

# Create FastMCP server instance
mcp = FastMCP("Mitos")

# No cross-call "seen" dedup state — deliberately. An earlier design cached
# already-surfaced slugs in a process-global set and trimmed `rejected_paths` (the
# relitigation-stopping field) from re-hits, flagging them `seen`. But `mitos serve`
# outlives a single agent session — the orchestrator `/clear`s and respawns the agent
# against the SAME long-running server — so the set leaked across session resets: a
# brand-new fresh-eyes session was handed `seen: true` with `rejected_paths` withheld,
# exactly the field it needed, with no way to tell it was being short-changed. No
# connection/session key is fully correct either (a bare `/clear` keeps the connection
# while resetting the agent's context), so the only correct shape is to hold no
# cross-call state at all. A caller who wants a lightweight scan passes `brief=True` —
# explicit, per-call, stateless. (V5 owns the rebuilt MCP server; carry this forward.)


def _attach_modifiers(payload: Dict[str, Any], node: Dict[str, Any], store: GraphStore) -> Dict[str, Any]:
    """Stamps reverse-relation modifier keys onto a decision payload, in place.

    Adds ``superseded_by`` / ``amended_by`` / ``narrowed_by`` / ``corrected_by``
    (only the non-empty ones) so a reader knows a later decision has moved on from
    this axiom and which one to chase — the fix for amended/narrowed nodes that stay
    ``active`` with their original (now-stale) mechanism text. Always applied,
    independent of ``brief``: the staleness flag matters even on an axiom-only scan,
    where ``rejected_paths`` is trimmed but the trap remains.
    Fail-silent — a modifier lookup error never breaks the recall response.

    Args:
        payload: The decision payload to augment.
        node: The store node dict (must carry ``id``).
        store: The graph store to read reverse edges from.

    Returns:
        The same payload dict, with any modifier keys added.
    """
    try:
        for key, slugs in store.get_modifiers(node["id"]).items():
            payload[key] = slugs
    except Exception:
        pass
    return payload


def _retired_handle(store: GraphStore, slug: str) -> Optional[Dict[str, Any]]:
    """Builds a retired-handle pointer for a superseded-filtered ranked match.

    The MCP twin of ``cli._retired_handle`` — kept independent per surface (the
    ranked loops collect retired handles separately, mirroring the deliberate
    payload-shaper asymmetry), but emitting the identical ``{"slug", "state"}`` (+
    ``superseded_by`` successor) shape so the blackout ``all_superseded`` field is
    byte-equal CLI⇄MCP (T5 parity). State is read authoritatively from the computed
    ``get_node_state`` via the state-agnostic ``resolve_slug`` (the vector payload's
    ``state`` is stale-at-embed-time). Calm degradation (P9): an unresolvable slug
    returns ``None`` (omitted by the caller), a failed state read falls back to
    ``"superseded"``.

    Args:
        store: The graph store to resolve the slug and read state/modifiers from.
        slug: The slug of the superseded-filtered match.

    Returns:
        The retired-handle dict, or ``None`` if the slug does not resolve.
    """
    try:
        node_ids = store.resolve_slug(slug)
    except Exception:
        return None
    if not node_ids:
        return None
    node_id = node_ids[0]
    try:
        state = store.get_node_state(node_id)
    except Exception:
        state = "superseded"
    handle: Dict[str, Any] = {"slug": slug, "state": state}
    try:
        successors = store.get_modifiers(node_id).get("superseded_by")
        if successors:
            handle["superseded_by"] = successors
    except Exception:
        pass
    return handle


def _decision_payload(node: Dict[str, Any], score: float, *, brief: bool,
                      store: Optional[GraphStore] = None) -> Dict[str, Any]:
    """Shapes a Letter-mode decision payload.

    ``rejected_paths`` — the heavy, high-value field whose reasoning stops
    relitigation — is always included unless ``brief`` (the caller explicitly asked
    for an axiom-only scan). There is deliberately no cross-call "seen" trimming; see
    the module-level note on why that state was removed.

    Args:
        node: A store node dict (``slug``, ``core_axiom``, ``rejected_paths``, ``scope``).
        score: The relevance score to attach.
        brief: Drop ``rejected_paths`` for an axiom-only scan.
        store: When given, reverse-relation modifier keys are stamped on (always,
            even for brief payloads).

    Returns:
        A Letter-mode decision dict.
    """
    payload = letter_payload(node, brief=brief, extras={"score": score})
    if store is not None:
        _attach_modifiers(payload, node, store)
    return payload

class _RenderedToolError(Exception):
    """A targeting failure already rendered for delivery. ``str()`` IS the body.

    Module-private, and it never leaves this file. Its whole purpose is to be the
    thing a tool **raises** once the anatomy has been composed, because that is
    the only delivery shape that carries it — measured against this tree's own
    ``mcp`` over real stdio:

    * ``raise ProjectTargetingError(...)`` → ``isError: True``, and the delivered
      text is that error's *terse discriminator-level fallback*. The anatomy is
      gone, and the result looks entirely correct while carrying nothing an agent
      can act on. This is the trap.
    * ``raise _RenderedToolError(<anatomy>)`` → ``isError: True`` and the full
      anatomy.
    * ``return <anatomy>`` → an ordinary **success**, which an addressing-class
      failure is not.

    ``Tool.run`` wraps whatever a tool raises as ``ToolError(f"Error executing
    tool {name}: {e}")``, and the low-level server renders *that* through
    ``_make_error_result(str(err))`` — so the delivered body is ``str()`` of what
    was raised. FastMCP's prefix is kept rather than fought: it names the failing
    tool, which the body would otherwise have to repeat.

    Deliberately **not** a ``MitosError`` and deliberately not a field on
    ``ProjectTargetingError``: a finished presentation string living on a shared
    error type is exactly the leak the composition locus forbids — it would put
    MCP call syntax one attribute away from the CLI boundary.
    """


def _example_project_name(err: ProjectTargetingError) -> Optional[str]:
    """Picks the project name a rendered example should use, or None for none.

    The closest match first when the caller mistyped a name — an example naming
    what they probably meant is worth more than an arbitrary one — then the first
    registered name. **Read from ``err.registered_names``, never from the bounded
    view**: above the enumeration bound with no close matches, ``BoundedNames``
    is deliberately empty, and the bound governs *enumeration*, not the existence
    of one example.

    Args:
        err: The typed targeting error.

    Returns:
        A registered name, or ``None`` on a machine with none registered (the
        caller then renders the absolute-path form, which is the only recovery
        that exists there).
    """
    if err.close_matches:
        return err.close_matches[0]
    if err.registered_names:
        return err.registered_names[0]
    return None


def _example_call(tool: str, err: ProjectTargetingError,
                  *, prefer_path: bool = False) -> str:
    """Renders the concrete call form the caller should have made.

    An ellipsis stands in for the tool's other arguments rather than a per-tool
    argument list: the list would have to be maintained beside six signatures and
    would drift from them silently, and the caller already holds the arguments —
    what they are missing is the ``project``.

    Args:
        tool: The tool the failing call named.
        err: The typed targeting error.
        prefer_path: Render the absolute-path form even when a registered name is
            available. Load-bearing for ``registered_unreachable``, where the
            registered names *include the one that just failed*: an example
            naming a project by name there says retry the thing that did not
            work, which is the dead end the class exists to avoid.

    Returns:
        One backticked call form.
    """
    name = None if prefer_path else _example_project_name(err)
    target = name if name is not None else "/absolute/path/to/the/workspace"
    return f"`{tool}(project='{target}', …)`"


def _registered_projects_line(bounded: routing.BoundedNames) -> str:
    """Renders the registered-name vocabulary, respecting the enumeration bound.

    The MCP twin of ``cli._registered_projects_line`` — same policy, different
    words, and it must stay different: this one may not name ``mitos init``,
    because registration is a human setup act and an agent handed a
    state-creating shell command is invited to run it.

    Above ``routing.REGISTERED_NAMES_BOUND`` the enumeration collapses to the
    close matches plus a count — and when there are no close matches, to the
    count **alone**. An empty ``names`` with ``collapsed=True`` is the honest
    answer, so this must never be spelled ``bounded.names or [...]``: that undoes
    the distinction the policy exists to make, in the one place nobody looks. The
    discovery pointer is a separate line here (the caller appends it to every
    body), so this one stays pure vocabulary.

    All four variants open with the same ``Registered projects:`` token — the CLI
    twin gives each variant its own shape, and this one deliberately does not.
    Every other line of every body also contains the word *registered*, so a test
    asserting "no names were enumerated" by scanning the whole body is satisfied
    by a neighbouring line and goes green against the bug (3b met exactly that
    trap on the did-you-mean line). One stable token means the vocabulary line can
    be parsed out and asserted **exactly**, which is the only assertion shape that
    can see a bound regression.

    Args:
        bounded: The policy verdict from ``routing.bounded_registered_names``.

    Returns:
        One indented line for the error body.
    """
    if bounded.total == 0:
        # Empty is first-class. The CLI answers this state by prescribing the
        # setup act; this surface answers it with the escape hatch, which is the
        # only recovery an agent is permitted to take.
        return ("  Registered projects: none on this machine yet — a workspace is "
                "still reachable by its absolute path.")
    if not bounded.collapsed:
        return f"  Registered projects: {', '.join(bounded.names)}."
    if bounded.names:
        return (f"  Registered projects: {bounded.total}, closest to what you "
                f"passed: {', '.join(bounded.names)}.")
    return f"  Registered projects: {bounded.total} — too many to enumerate here."


#: The discovery pointer, on every rendered body. An agent that meets a targeting
#: failure has, by construction, no project vocabulary — so the one thing every
#: body must carry is where to go and get it. Worded as a capability statement
#: rather than an instruction so it reads sanely under an empty registry too.
_DISCOVERY_POINTER = ("  `list_projects()` returns every registered project name "
                      "with its workspace path.")


def _render_targeting_error(err: ProjectTargetingError, tool: str) -> str:
    """Composes the MCP surface's teaching anatomy for a targeting failure.

    The §4.5 parts, in this surface's own vocabulary: what is wrong, a concrete
    **tool call** to copy, the ``list_projects()`` discovery pointer, and the
    registered-name line in its bound-appropriate form. It is a second renderer,
    not a shared body — ``cli._render_targeting_error`` says the same things in
    terms of ``mitos`` commands, and a message that blended the two would hand
    one surface's syntax to the other's caller.

    Two rules make this renderer's wording different in kind, not merely in
    phrasing, and both are enforced by tripwire:

    * **It never names a state-creating shell command.** No ``mitos init``, no
      ``mitos projects``, no CLI flag. An agent that meets an error naming a
      shell command is invited to run it — an autonomous ``init`` scaffolding a
      workspace and claiming a registration nobody asked for. Where the CLI's
      recovery is a setup act, this surface's is the absolute-path escape hatch,
      or naming a **human** as the next actor. Surfacing a repoint to the
      operator *is* a terminating action; running one is not.
    * **It reads nothing.** No ``os.getcwd()``, no second ``registry.load()`` —
      it is a pure function of the typed error, so it cannot fail. The CLI's
      renderer needs a ``try/except (RegistryError, OSError)`` guard precisely
      because it re-reads both *inside* an ``except`` arm; porting that guard
      here would imply a read that is not there.

    There is deliberately **no cwd line at all**, though §4.5 permits framing one
    as launch-dir context. On an always-on server ``os.getcwd()`` is fixed for
    the process's whole life, so a hint derived from it is constant across every
    call — always noise, or always wrong in the same way. And since the
    working-directory fallback was deleted there is nothing true left for such a
    line to say: "your launch dir sits inside project X" would name a project this
    surface will not use for anything, which is worse than silence.

    It never calls ``str(err)``: that is the terse discriminator-level fallback
    for an unrendered path, fenced by a tripwire forbidding exactly the strings
    this function emits.

    Args:
        err: The typed error, carrying structured data only.
        tool: The tool whose call failed — it appears in the example, so the
            caller can copy the fix rather than translate it.

    Returns:
        The message body. Multi-line; every line after the first is indented as a
        recovery, not as a new failure. FastMCP prefixes it with
        ``Error executing tool <tool>: `` on the way out.
    """
    if err.discriminator == TARGET_EXEMPT_VERB:
        # Unreachable from this surface — no MCP tool is exempt, and
        # `list_projects` takes no `project` parameter, so there is nothing to
        # refuse. It gets a terse honest branch anyway rather than falling into
        # a neighbour's `else`, which would render it in `registered_unreachable`'s
        # words. If you find yourself reaching for `routing.exempt_verb_error`
        # here, the design has drifted.
        return (f"the `{err.verb}` tool takes no `project` argument — it answers "
                f"for the machine, not for one workspace.")

    bounded = routing.bounded_registered_names(err.registered_names, err.close_matches)
    example = _example_call(tool, err)

    if err.discriminator == TARGET_MISSING:
        lines = [
            "no project was named, so there is no workspace to answer for.",
            f"  Name one on the call — {example} — passing either a registered "
            f"project name or the absolute path of a workspace directory.",
            _registered_projects_line(bounded),
        ]
    elif err.discriminator == TARGET_UNKNOWN_NAME:
        lines = [f"no project named {err.selector!r} is registered on this machine."]
        if err.close_matches:
            # Never truncated: `close_project_matches` expands each folded match
            # to every original that folds onto it, so a registry holding several
            # case variants of one name legitimately returns more than
            # `PROJECT_DIDYOUMEAN_MAX` — and dropping one would hide the very
            # distinction the caller needs to see.
            lines.append(f"  Did you mean: {', '.join(err.close_matches)}")
        # The recovery leads with whichever option actually exists. On a machine
        # with an empty registry a name selector still reaches this class — an
        # agent guessing a plausible project name is the ordinary way in — and
        # "retry with a registered name" would there be a recovery naming an
        # option that does not exist, which is the dead end the anatomy removes.
        lines.append(
            f"  Retry with "
            f"{'a registered name, or with ' if err.registered_names else ''}"
            f"the absolute path of the workspace you mean — {example}.")
        lines.append(_registered_projects_line(bounded))
    elif err.discriminator == TARGET_RELATIVE_PATH:
        # The realistic mistake on this surface: agents reach for relative paths
        # by habit, from a world where a working directory means something. This
        # surface canonicalizes nothing, so the habit lands here — and the CLI's
        # line about shell quoting is wrong here, because there is no shell.
        lines = [
            f"the project selector {err.selector!r} is a relative path, and this "
            f"surface resolves nothing against a working directory.",
            f"  Pass a registered project name or an absolute path — {example}.",
        ]
        if err.selector.startswith("~"):
            lines.append("  (A leading `~` is not expanded here — pass the "
                         "absolute path it stands for.)")
        lines.append(_registered_projects_line(bounded))
    elif err.discriminator == TARGET_PATH_NOT_A_WORKSPACE:
        # Naming the triple is a description of what was looked for, which is
        # fair to say; prescribing the command that creates it is not.
        lines = [
            f"there is no Mitos workspace at {err.path!r}.",
            "  A workspace is a directory holding both .mitos/config.toml and "
            "decisions.md; that path does not.",
            f"  Name a registered project, or the absolute path of a workspace "
            f"that already exists — {example}.",
            _registered_projects_line(bounded),
        ]
    else:
        # TARGET_REGISTERED_UNREACHABLE — the constructor whitelists the
        # discriminator and the other five are handled above, so this is the
        # remaining class rather than a fall-through default (`errors.
        # _fallback_message` and the CLI renderer are spelled the same way).
        #
        # The CLI's recovery for this class is `mitos init --force`, which is
        # closed to this surface. Left unspecified, the class would hand an agent
        # a failure with no action it is permitted to take — and an agent with no
        # named recovery retries or improvises. So the recovery is named as what
        # it actually is: work for a human, which the agent's job is to surface.
        lines = [
            f"the project {err.name!r} is registered at {err.path!r}, which no "
            f"longer holds a Mitos workspace.",
            "  The registration points somewhere that has moved or been removed, "
            "and repointing it needs a human — report that to whoever is "
            "operating this session.",
            f"  If you already know where the workspace lives now, name that "
            f"absolute path instead — {_example_call(tool, err, prefer_path=True)}.",
            _registered_projects_line(bounded),
        ]

    lines.append(_DISCOVERY_POINTER)
    return "\n".join(lines)


def _target_config(project: Optional[str], tool: str) -> MitosConfig:
    """Resolves a tool call's ``project`` to the one config every read below uses.

    The MCP analogue of ``cli.main()``'s single resolution block. It is per-tool
    rather than per-process because this surface has no shared entry point, but
    the discipline is identical and is the point of the phase: **one resolution
    site, one config, and no second construction anywhere in the call.** A tool
    that resolved project B and then rebuilt a cwd-derived config for its
    provenance stamp would return B's decisions labelled with A's collection —
    not a fallback surviving, but a second, disagreeing resolution inside one
    call.

    There is no gate on ``project`` at all, and that is the finished shape rather
    than a simplification: an omitted selector and an empty one are the *same*
    class, so both go straight to ``routing.resolve_project``'s single raise site
    and come back as the missing anatomy. The parameter stays ``Optional`` because
    ``None`` is what an omitted argument delivers — the type describes the call,
    not a default this function supplies.

    Args:
        project: The selector as the caller passed it, or ``None`` when the
            argument was omitted. Both reach the resolver; neither resolves a
            working directory.
        tool: The calling tool's name, for the rendered example.

    Returns:
        The config for the resolved workspace, carrying the resolved
        ``project`` name for the provenance echo. A targeting failure raises
        *before* any config exists, which is why an error response carries the
        teaching anatomy in the echo's place rather than a stamp.

    Raises:
        _RenderedToolError: On any targeting failure — carrying the finished
            anatomy, because the typed error's own ``str()`` cannot.
        RegistryError: If the registry file itself is unusable. Deliberately
            unwrapped: there is no registered vocabulary to teach when the file
            holding it cannot be read, so it arrives as one calm line.
        ConfigError: If the *resolved* workspace's ``config.toml`` is malformed.
            Resolution proves a workspace's shape, not that its config parses —
            a malformed config must be resolvable-then-diagnosed. No carve-out
            here; it is the same pre-existing gap the CLI has.
    """
    try:
        target = routing.resolve_project(project)
    except ProjectTargetingError as err:
        raise _RenderedToolError(_render_targeting_error(err, tool)) from err
    # `project=` carries the caller's own vocabulary onto the config so every
    # answer below can echo the target back. `target.name` is already the
    # registered name for both selector forms and `None` for an unregistered
    # path, which the constructor resolves to the canonical path — so the echo
    # is defined for every form without a branch here.
    return MitosConfig(target.root, project=target.name)


def get_workspace_components(config: MitosConfig) -> Tuple[GraphStore, Optional[GeminiEmbeddingProvider], Optional[QdrantVectorStore]]:
    """Loads and returns the graph store (read-only), embedding provider, and vector store.

    Args:
        config: The already-resolved workspace config — the caller resolves once
            at the tool boundary (``_target_config``) and hands the same object
            to every read in the call. It used to build a zero-argument
            ``MitosConfig()`` here, which made the target the process's working
            directory: on an always-on server, the launch directory, fixed for
            the process's whole life and independent of what any call meant.
    """
    store = GraphStore(config.db_path, read_only=True)
    
    embed_provider = None
    vector_store = None
    try:
        cache_path = os.path.join(config.mitos_dir, "embedding_cache.sqlite")
        embed_provider = GeminiEmbeddingProvider(
            cache_path,
            api_key=config.env.get("GEMINI_API_KEY"),
            model_id=get_embedding_model_id(config.env),
        )
        vector_store = QdrantVectorStore(config.qdrant_url, config.qdrant_collection)
    except Exception:
        pass
        
    return store, embed_provider, vector_store


def _lexical_degraded_response(query: str, *, config: MitosConfig, reason: str,
                               store: Optional[GraphStore], brief: bool,
                               limit: int,
                               open_questions: Optional[List[Dict[str, Any]]] = None) -> str:
    """Builds the degraded lexical-fallback JSON for the MCP read tools.

    The MCP twin of ``cli._emit_lexical_degraded`` (ADR
    ``read-verbs-degrade-to-lexical-decisions-md-fallback``): the shared
    ``lexical_fallback`` runs the term-match over decisions.md, so the two
    surfaces cannot drift. The envelope carries ``degraded: "lexical"`` and a
    ``degraded_reason`` — never an ``{error}`` object or raw provider text.

    Args:
        query: The claim/topic the caller was trying to recall.
        config: The call's one resolved workspace config. The degraded envelope
            must name the workspace the caller *asked* for — it reads that
            workspace's ``decisions.md`` and stamps its provenance — so this is
            threaded in rather than rebuilt: a second construction here would
            answer a targeted call out of whichever directory the server started
            in, and label it as such.
        reason: One-line cause phrase (see ``degraded_reason_from_error``).
        store: A readable graph store for active-filtering + modifier stamps,
            or None when the graph itself is down (pre-V1a).
        brief: Omit ``rejected_paths`` from each match.
        limit: Max matches to return.
        open_questions: An already-computed scoped parked-OQ list to carry on
            the envelope (present-if-scanned semantics — None means omitted).

    Returns:
        The degraded envelope as a JSON string.
    """
    envelope = lexical_fallback(
        query, config.decisions_file, reason=reason, store=store,
        limit=limit, brief=brief,
    )
    envelope["query"] = query
    envelope.update(corpus_provenance(config))
    if open_questions is not None:
        envelope["open_questions"] = open_questions
    return dumps_display(envelope, ensure_ascii=False, indent=2)


def _oq_payload(oq: Dict[str, Any]) -> Dict[str, Any]:
    """Builds the open-question output sub-dict for the MCP visibility tools.

    Mirrors the CLI twins (``cmd_surface`` / ``cmd_list``): the
    ``{topic, questions_raised, park_reason}`` shape PLUS any reverse-relation
    modifier keys (``amended_by`` / ``narrowed_by``) already stamped on the OQ by
    ``get_open_questions``' 2b modifier chokepoint, read straight off the payload —
    so an amended-but-active OQ never reads as the final word, and the MCP surface
    stays behaviourally in sync with its CLI mirror (CLI⇄MCP parity).

    Args:
        oq: A hydrated, modifier-stamped open-question dict from
            ``get_open_questions``.

    Returns:
        The OQ output sub-dict, carrying present modifier keys when non-empty.
    """
    payload: Dict[str, Any] = {
        "topic": oq["slug"],
        "questions_raised": oq["questions_raised"],
        "park_reason": oq.get("park_reason"),
    }
    payload.update({key: oq[key] for key in MODIFIER_EDGE_KEYS.values() if oq.get(key)})
    return payload


@mcp.tool()
def surface_decisions(query: str, scope: Optional[str] = None, brief: bool = False, limit: int = 5,
                      project: Optional[str] = None) -> str:
    """Surface active precedents for a CLAIM before you decide — the recall loop, use first.

    The broad "is there a settled decision near this?" scan: a ranked, capped (top
    few) semantic match. Reach for this when deciding something; reach for
    query_decisions to look up a SPECIFIC slug or claim, and list_decisions for the
    EXHAUSTIVE set in a scope. Each returned precedent carries its `rejected_paths`
    (why alternatives were ruled out) — the field that actually stops relitigation.
    Every hit carries its full `rejected_paths` unless you pass `brief=True`. Closing
    the loop: after you decide, `record_decision` the outcome so the next agent
    inherits it instead of relitigating.

    Args:
        query: The semantic claim or topic string (e.g. 'cache strategy').
        scope: Optional scope hint — does NOT filter the semantic search. Recall is
            scope-blind by design, so a mis-guessed tag can't hide cross-scope
            precedent; scope only narrows the `open_questions` scan and shapes the
            recall `note` (incl. the "unused tag → valid scopes" redirect). For
            scope-RESTRICTED retrieval use list_decisions(scope=...) — the only
            surface that hard-filters by scope.
        brief: If True, omit `rejected_paths` from every result (axiom-only — a quick
            "is there anything nearby?" scan). Default False keeps the full reasoning.
        limit: Ranked top-k to retrieve (default 5; clamped to 1–50). Raise it to dig
            deeper, lower it to save context — a context-budget dial, not a cap at 5.
        project: Which project this call is about — REQUIRED on every call: a
            registered project name (e.g. 'mitos') or the absolute path of a
            workspace. Call `list_projects()` if you do not know the names.
            Distinct from `scope`: `project` picks the corpus, `scope` filters
            within it.

    Returns:
        A JSON string with `active_decisions` (ranked, Letter-mode), plus
        `open_questions` ONLY when a scope was given (absent = not scanned, [] = none
        parked in that scope). Each decision: slug, axiom, scope, score, and
        rejected_paths unless brief. A precedent a later decision has moved on from
        also carries the modifying slugs under
        `superseded_by`/`amended_by`/`narrowed_by`/`corrected_by` (always present when
        they apply, even on a brief scan) — chase those before treating its axiom as
        the current mechanism. Also includes `confidence` (`strong`/`weak`/`none` when
        semantic ranking ran) and a `note`: `weak` or `none` means no settled precedent
        on this claim — treat it as no-precedent and decide, or call
        list_decisions(scope=...) for a certain check (don't read weak neighbours as a
        settled decision). When ranked recall retrieved precedents but every one is
        superseded (a blackout), `active_decisions` stays empty and a sibling
        `all_superseded` list carries the retired handles (`slug`, `state`, and the live
        `superseded_by` successor when known) with the `note` naming them — that is a
        recoverable "it was settled before, go read the history", not a true miss.
    """
    top_k = clamp_limit(limit)
    # Resolve BEFORE the `try` below, never inside it. That `except Exception`
    # exists to degrade a broken *graph* to the lexical fallback; a targeting
    # failure caught by it would come back as "semantic recall is degraded" with
    # a nonsense reason and isError: False — the anatomy silently destroyed on
    # the highest-traffic tool, with every existing row still green.
    config = _target_config(project, "surface_decisions")
    # A pre-V1a graph raises at store construction — the graph is unusable, so
    # the lexical fallback parses decisions.md directly (no graph access).
    try:
        store, embed_provider, vector_store = get_workspace_components(config)
    except Exception as e:
        return _lexical_degraded_response(
            query, config=config, reason=degraded_reason_from_error(e), store=None,
            brief=brief, limit=top_k,
        )

    results: Dict[str, Any] = {"active_decisions": []}
    results.update(corpus_provenance(config))
    semantic_ran = False
    top_score: Optional[float] = None
    retired: List[Dict[str, Any]] = []
    degraded_error: Optional[Exception] = None

    # 1. Semantic search if embeddings and vector store are active
    if embed_provider and vector_store:
        try:
            # Generate query vector
            q_vector = embed_provider.get_embedding(query, is_query=True)
            matches = vector_store.query(q_vector, limit=top_k)
            semantic_ran = True

            for m in matches:
                slug = m["slug"]
                node = store.get_node_by_slug(slug)
                if not node:
                    handle = _retired_handle(store, slug)
                    if handle:
                        retired.append(handle)
                    continue

                # Verify computed active status in SQLite (M3 computed state is source-of-truth)
                node_state = store.get_node_state(node["id"])
                if node_state not in ("active", "drifted"):
                    # Stale vector reference — a retired handle for the blackout vector.
                    handle = _retired_handle(store, slug)
                    if handle:
                        retired.append(handle)
                    continue

                results["active_decisions"].append(
                    _decision_payload(node, m["score"], brief=brief, store=store)
                )
                if top_score is None or m["score"] > top_score:
                    top_score = m["score"]
        except CollectionMissingError as e:
            # I8 — an absent collection over an EMPTY active set IS the empty index,
            # and a just-initialized project must not read as broken: leave
            # `semantic_ran` True so this renders "ran and found nothing" (which is
            # also what suppresses the degraded-only unranked scope dump below).
            # Over a populated graph it is a real hole in recall and degrades, with
            # the header naming the collection and `mitos reconcile`.
            if missing_index_is_a_gap(store):
                semantic_ran = False
                degraded_error = e
            else:
                semantic_ran = True
        except Exception as e:
            # Degrade to exact/scope filtering only
            semantic_ran = False
            degraded_error = e

    # 2. Scope pre-filtering fallback — ONLY when semantic recall is down (degraded).
    #    When semantic ran and simply found nothing, do NOT dump an unranked scope
    #    listing dressed as matches — that's the false-precedent ambiguity P5 closes.
    if not semantic_ran and not results["active_decisions"] and scope:
        try:
            active_decs = store.get_active_decisions(scope=scope)
            for d in active_decs[:5]:
                results["active_decisions"].append(
                    _decision_payload(d, 1.0, brief=brief, store=store)
                )
        except Exception:
            pass

    # 3. Append Open Questions ONLY when a scope was given (C4 resolves clause).
    #    Omitting the key when no scope disambiguates "not scanned" from "none here".
    if scope:
        open_questions = []
        try:
            for q in store.get_open_questions(scope=scope):
                if q["state"] == "parked":
                    open_questions.append(_oq_payload(q))
        except Exception:
            pass
        results["open_questions"] = open_questions

    # Degraded and empty-handed on decisions: route into the deterministic
    # lexical fallback (ADR read-verbs-degrade-to-lexical-decisions-md-fallback)
    # instead of the self-contradicting clean-empty result + unavailable note.
    # The scoped open-questions scan (a pure graph read that survived) rides
    # along on the degraded envelope.
    if not semantic_ran and not results["active_decisions"]:
        return _lexical_degraded_response(
            query, config=config, reason=degraded_reason_from_error(degraded_error),
            store=store, brief=brief, limit=top_k,
            open_questions=results.get("open_questions"),
        )

    # Confidence signal — let the agent tell a settled precedent from loose neighbours
    # or genuine absence, instead of a boilerplate note that read the same every time
    # (AX P5). Pass the live scope-count map (busiest-first) when a scope is given: it is
    # the unused-scope oracle (a tag absent from it gets a bounded self-correction
    # vector) and the did-you-mean / top-K source. Calm-degrade to None on error.
    scope_counts: Optional[Dict[str, Dict[str, int]]] = None
    if scope:
        try:
            scope_counts = order_scope_counts(store.get_scope_counts())
        except Exception:
            pass

    confidence, note = assess_surface_recall(
        semantic_ran=semantic_ran,
        top_score=top_score,
        result_count=len(results["active_decisions"]),
        scope=scope,
        scope_counts=scope_counts,
        surface="mcp",
    )
    if confidence is not None:
        results["confidence"] = confidence
    results["note"] = note

    # Blackout: semantic ranking ran and retrieved precedents, but every one was
    # superseded-filtered. Override the note with the recovery vector and attach the
    # retired handles (CLI⇄MCP-identical shape, T5 parity). Distinct from a true miss
    # (where `retired` is empty); fires regardless of any parked open questions.
    if semantic_ran and not results["active_decisions"] and retired:
        results["note"] = blackout_note(retired)
        results["all_superseded"] = retired

    # W31 — the unbuilt graph (a clone carrying the corpus but not the gitignored
    # *.sqlite). Consulted on the ORDINARY empty path, not in an `except` arm: an
    # empty graph raises nothing, it answers empty, and that answer is what an agent
    # reads as "no precedent" for a project holding hundreds. After the blackout
    # override so the graph note wins if both applied — with no graph, a pointer at
    # the graveyard names the wrong heal; in fact they exclude each other, since a
    # retired handle is a node. CLI⇄MCP parity is structural: one predicate, one
    # composer, each surface's own register.
    if not results["active_decisions"] and missing_graph_is_a_gap(
        store, config, corpus_has_entries=corpus_has_entries
    ):
        results["note"] = missing_graph_note("mcp")

    return dumps_display(results, ensure_ascii=False, indent=2)


@mcp.tool()
def list_decisions(scope: Optional[str] = None, state: str = "active", brief: bool = False,
                   oneline: bool = False, project: Optional[str] = None) -> str:
    """Enumerate the COMPLETE set of decisions (optionally scope-filtered) — no ranking, no top-k.

    surface_decisions / query_decisions are SEMANTIC and capped at the top few
    matches: ideal for "is there precedent for this claim?", but they cannot tell
    you whether you have seen *everything*. When you need certainty — a completeness
    pass over a scope, an audit, "show me every settled call in `auth`" — use this.
    It returns every matching decision deterministically, straight from the graph,
    so nothing hides below a relevance cliff. Needs no API key or Qdrant (pure graph
    read), so it works even when semantic recall is degraded. This is also the ONLY
    retrieval surface that hard-filters by scope — surface/query are scope-blind, so
    when you want results restricted to a scope, this is the verb.

    Args:
        scope: Optional scope tag filter (e.g. 'auth') — a true hard filter (this is
            the only retrieval surface that restricts by scope). Omit for the whole
            project.
        state: 'active' (default) returns the live set (active + drifted); 'all'
            returns every decision regardless of state (including superseded); any
            other value is an exact computed-state match (e.g. 'superseded').
        brief: If True, omit `rejected_paths` from every decision (axiom-only). Useful
            here — an exhaustive scope can otherwise return many full reasoning walls.
        oneline: If True, return the orientation/table-of-contents tier: one minimal
            object per decision — {slug, axiom_oneline (word-boundary-truncated),
            state} plus modifier slugs when present. For big scopes where even
            brief=True blows the result ceiling (measured: a 45-decision scope did) —
            scan the map here, then dereference the few that matter with query/show.
            Letter-complete stays the default depth; this is an explicit opt-down,
            never a default. Mutually exclusive with brief.
        project: Which project this call is about — REQUIRED on every call: a
            registered project name (e.g. 'mitos') or the absolute path of a
            workspace. Call `list_projects()` if you do not know the names.
            Distinct from `scope`: `project` picks the corpus, `scope` filters
            within it.

    Returns:
        A JSON string: {decisions, open_questions, total, scope, state}. Each
        decision carries the same Letter-mode shape as surface_decisions (slug,
        axiom, rejected_paths, scope) plus its computed `state`, and — when a later
        decision modifies it — `superseded_by`/`amended_by`/`narrowed_by`/
        `corrected_by` modifier slugs (the stamps survive every thinner tier,
        including oneline). UNBOUNDED.
    """
    # An argument fault is answered by naming the argument, before any project is
    # resolved: the caller's mistake is in the depth tier, not in the target, and
    # resolving first would answer a different question than the one they got wrong.
    if brief and oneline:
        return dumps_display(
            {"error": "brief and oneline are mutually exclusive — pick one depth tier."},
            ensure_ascii=False, indent=None)

    config = _target_config(project, "list_decisions")
    store, _embed, _vec = get_workspace_components(config)

    nodes = store.get_decisions(scope=scope, state=state)
    modifiers = store.get_modifiers_map([n["id"] for n in nodes])
    decisions = []
    for n in nodes:
        # oneline swaps the Letter core for the minimal {slug, axiom_oneline, state}
        # object (same shape as the CLI's `list --oneline --json` — parity seam in
        # display.oneline_payload); modifier stamps ride either shape.
        if oneline:
            d = oneline_payload(n)
        else:
            d = letter_payload(n, brief=brief, extras={"state": n["computed_state"]})
        d.update(modifiers.get(n["id"], {}))
        decisions.append(d)

    open_questions = []
    try:
        for oq in store.get_open_questions(scope=scope):
            if oq["state"] == "parked":
                open_questions.append(_oq_payload(oq))
    except Exception:
        pass

    payload = {
        "decisions": decisions,
        "open_questions": open_questions,
        "total": len(decisions),
        "scope": scope,
        "state": state,
        **corpus_provenance(config),
    }

    # On an empty scoped read, distinguish a genuinely-fresh scope from a misspelled one:
    # an absent-from-live scope rides two additive, in-band fields (never an error object
    # or non-zero exit — an LLM agent reads those as a call-syntax fault and thrashes).
    # Only the miss path pays the get_scope_counts() read. The recovery payload carries no
    # node id, so there is nothing to modifier-stamp here.
    if scope and not decisions and not open_questions:
        scope_counts: Optional[Dict[str, Dict[str, int]]] = None
        try:
            scope_counts = order_scope_counts(store.get_scope_counts())
        except Exception:
            pass
        recovery = scope_filter_recovery(
            scope=scope, scope_counts=scope_counts, surface="mcp"
        )
        if recovery:
            payload["scope_known"] = False
            payload["scope_recovery"] = recovery["note"]

    return dumps_display(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def list_scopes(include_archived: bool = False, project: Optional[str] = None) -> str:
    """List the project's scope-tag vocabulary with each domain's live-node counts.

    The map an agent reads BEFORE recording or recalling: every scope tag that
    carries a live node, ranked busiest-domain-first (total active decisions +
    parked open questions, descending; ties alphabetical). record_decision /
    surface_decisions / query_decisions / list_decisions all let you write into a
    scope or read from one — but only this tells you *what scopes exist and how
    alive each is*, so you can pick the project's real vocabulary instead of
    inventing a near-duplicate tag. A pure graph read — no API key or Qdrant needed,
    so it works even when semantic recall is degraded.

    This returns a tag→counts AGGREGATE, not decision payloads: there is no node id
    to stamp, so — unlike surface/query/list_decisions — it carries no
    `superseded_by`/`amended_by`/… modifier keys (that is correct, not a missing
    stamp). An empty/fresh project returns `{"scopes": {}, …}` — a valid empty
    vocabulary, never an error, and the provenance says which project was empty.

    Args:
        include_archived: When False (default), returns only live domains (≥1 active
            decision OR ≥1 parked open question). When True, additionally includes
            every other scope tag present in the graph at a `{active_decisions: 0,
            parked_open_questions: 0}` floor — the scope-level parallel of
            list_decisions(state="all").
        project: Which project this call is about — REQUIRED on every call: a
            registered project name (e.g. 'mitos') or the absolute path of a
            workspace. Call `list_projects()` if you do not know the names.

    Returns:
        A JSON string: `{scopes, project, collection, workspace}`. `scopes` is an
        ordered map `{scope: {active_decisions, parked_open_questions}}`, busiest
        domain first — the key order of THAT map IS the deliverable, so iterate it
        as-is. The other three name the corpus the vocabulary came from: the
        project as you addressed it, its derived collection, and its path.
    """
    config = _target_config(project, "list_scopes")
    store, _embed, _vec = get_workspace_components(config)
    # The vocabulary nests under `scopes` rather than sharing the top level with
    # the stamp: scope tags are user-authored strings, and a project holding a
    # tag literally named `project`/`collection`/`workspace` would otherwise have
    # its own vocabulary silently overwritten by the provenance.
    envelope = {
        "scopes": order_scope_counts(store.get_scope_counts(include_archived=include_archived)),
    }
    envelope.update(corpus_provenance(config))
    return dumps_display(envelope, ensure_ascii=False, indent=2)


@mcp.tool()
def show_node(ident: str, project: Optional[str] = None) -> str:
    """Dereference ONE decision or open question by exact handle — slug OR content-hash id.

    The exact-handle lookup that reaches the graveyard: it resolves a node
    state-agnostically (active-first, else the most-recent superseded node in the
    casefolded-slug lineage), so it answers for a SUPERSEDED node that
    query_decisions' slug branch — active-view-only — cannot reach. Use it to
    reconstruct *why* a now-retired call was made (don't relitigate a settled
    rejection). Not a search: pass the precise slug or id you already hold, not a
    claim (for ranked recall use query_decisions / surface_decisions).

    Args:
        ident: A content-hash id or a slug (case-insensitive) — the exact handle.
        project: Which project this call is about — REQUIRED on every call: a
            registered project name (e.g. 'mitos') or the absolute path of a
            workspace. Call `list_projects()` if you do not know the names.

    Returns:
        A JSON string. A found **decision** is a Letter-complete object (`axiom` +
        `rejected_paths`) with `kind`/`id`/`slug`/`scope`/`state`; a found **open
        question** carries `topic`/`questions_raised`/`park_reason`. Both stamp the
        present reverse-relation modifier keys — a superseded node names its
        `superseded_by`, an amended one its `amended_by`/`narrowed_by` — so a
        moved-on node never reads as the final word. A genuinely-absent handle
        returns `{found: false, ident, hint}` (never an error), the hint pointing
        at `mitos sync` for an authored-but-unsynced draft. Both shapes carry the
        trailing `project`/`collection`/`workspace` provenance — most valuable on
        the absent one, which is otherwise ambiguous between "no such handle
        here" and "you are asking the wrong project".
    """
    config = _target_config(project, "show_node")
    store, _embed, _vec = get_workspace_components(config)

    # State-agnostic resolution via the SHARED 5a seam — the identical method
    # cmd_show calls, so the resolution selection cannot drift between surfaces.
    # A genuine MI-13 breach raises ValidationError out of resolve_handle; we do
    # NOT swallow it into not-found (a breach is not "not found").
    node = store.resolve_handle(ident)
    if not node:
        # Stamped too, and it is the more valuable half: "not found" is exactly
        # the answer that is ambiguous between a missing handle and a mis-aimed
        # call. Provenance last, on both branches, so the CLI twin's dict stays
        # equal key-for-key.
        missing = {"found": False, "ident": ident, "hint": SHOW_NOT_FOUND_HINT}
        missing.update(corpus_provenance(config))
        return dumps_display(missing, ensure_ascii=False, indent=2)

    # state from the separate computed-state read (never node.get("state") —
    # absent on the resolved dict); modifiers are the one kind-agnostic stamp
    # source. Stamping is LOAD-BEARING: surfacing the superseded is this tool's
    # whole job, so the superseded_by stamp is not decoration.
    state = store.get_node_state(node["id"])
    modifiers = store.get_modifiers(node["id"])
    payload = show_payload(node, state=state, modifiers=modifiers)
    payload.update(corpus_provenance(config))
    return dumps_display(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def query_decisions(query: str, depth: str = "letter", brief: bool = False, limit: int = 5,
                    project: Optional[str] = None) -> str:
    """Look up a SPECIFIC decision by slug or claim — the targeted lookup.

    Use this when you know roughly what you're after (a slug you're carrying, or a
    pointed claim). For the broad "is there precedent near this?" scan before
    deciding, use surface_decisions; for the EXHAUSTIVE set in a scope, list_decisions.
    If query matches a unique slug exactly, returns that one decision (full); otherwise
    a ranked semantic search for the claim. Its slug branch is active-view-only — to
    dereference an EXACT handle including a superseded node it can't reach, use show_node.
    Once you decide, `record_decision` the outcome so the next agent inherits it.

    Args:
        query: Unique decision slug identifier OR a semantic claim search query.
        depth: The retrieval depth (e.g. 'letter', 'trace', 'vibe'). v0.1 enforces Letter mode.
        brief: If True, omit `rejected_paths` from ranked semantic matches (axiom-only).
            An exact-slug hit is always returned in full (you asked for that one).
        limit: Ranked top-k for the SEMANTIC branch (default 5; clamped to 1–50). Raise
            it to dig deeper, lower it to save context. Ignored by an exact-slug hit
            (that returns the one decision you named).
        project: Which project this call is about — REQUIRED on every call: a
            registered project name (e.g. 'mitos') or the absolute path of a
            workspace. Call `list_projects()` if you do not know the names.

    Returns:
        A JSON string containing the ranked results in Letter-mode payload shape.
        A decision a later one has moved on from also carries `superseded_by`/
        `amended_by`/`narrowed_by`/`corrected_by` (the modifying slugs) — an exact-slug
        hit on an amended decision still reads `state: "active"`, so chase these before
        trusting its axiom's mechanism. When the semantic branch retrieved precedents but
        every one is superseded (a blackout), `matches` stays empty and a sibling
        `all_superseded` list carries the retired handles (`slug`, `state`, live
        `superseded_by` when known) — settled before, not a true miss; read the history
        with list_decisions(state="all").
    """
    # The argument fault is answered before any project is resolved — see
    # list_decisions for why the ordering is deliberate.
    if depth != "letter":
        return dumps_display({"error": f"Depth mode '{depth}' is not yet implemented in v0.1 (Letter-only retrieval)."}, ensure_ascii=False, indent=None)

    # Resolved BEFORE the `try` — see surface_decisions: that `except Exception`
    # would swallow a targeting failure into the lexical-degraded envelope.
    config = _target_config(project, "query_decisions")
    # A pre-V1a graph raises at store construction — the graph is unusable, so
    # the lexical fallback parses decisions.md directly (no graph access).
    try:
        store, embed_provider, vector_store = get_workspace_components(config)
    except Exception as e:
        return _lexical_degraded_response(
            query, config=config, reason=degraded_reason_from_error(e), store=None,
            brief=brief, limit=clamp_limit(limit),
        )

    # 1. Try resolving query as direct slug first
    try:
        node = store.get_node_by_slug(query)
        if node:
            state = store.get_node_state(node["id"])

            output = {
                "slug": node["slug"],
                "axiom": node["core_axiom"],
                "rejected_paths": node["rejected_paths"],
                "scope": node["scope"],
                "state": state,
                "depth_mode": "letter"
            }
            output.update(store.get_modifiers(node["id"]))
            return dumps_display(output, ensure_ascii=False, indent=2)
    except Exception:
        # Not a slug collision or lookup failed; proceed to semantic claim lookup
        pass

    # 2. Perform ranked semantic claim search
    if embed_provider and vector_store:
        try:
            top_k = clamp_limit(limit)
            q_vector = embed_provider.get_embedding(query, is_query=True)
            matches = vector_store.query(q_vector, limit=top_k)

            output_list = []
            retired: List[Dict[str, Any]] = []
            for m in matches:
                slug = m["slug"]
                node = store.get_node_by_slug(slug)
                if not node:
                    handle = _retired_handle(store, slug)
                    if handle:
                        retired.append(handle)
                    continue

                node_state = store.get_node_state(node["id"])
                if node_state not in ("active", "drifted"):
                    handle = _retired_handle(store, slug)
                    if handle:
                        retired.append(handle)
                    continue

                match = letter_payload(
                    node,
                    brief=brief,
                    extras={"state": node_state, "score": m["score"], "depth_mode": "letter"},
                )
                match.update(store.get_modifiers(node["id"]))
                output_list.append(match)

            # Blackout: retrieved precedents but every one superseded-filtered.
            # Add the retired handles so the agent gets a pointer, not a false miss
            # (CLI⇄MCP-identical `all_superseded` shape, T5 parity).
            envelope: Dict[str, Any] = {"query": query, "depth_mode": "letter", "matches": output_list}
            envelope.update(corpus_provenance(config))
            if not output_list and retired:
                envelope["all_superseded"] = retired
            # W31 — see surface_decisions. Both of this tool's empty envelopes carry
            # it: this one (semantic ran and matched nothing) and the healthy-empty
            # one built in the CollectionMissingError arm below. An unbuilt clone
            # typically reaches the second — its collection is absent too — but the
            # first is reachable whenever the collection exists and the graph does
            # not, and a diagnosis present on one exit only is a verb that reads as
            # done (3e's per-EXIT lesson).
            elif not output_list and missing_graph_is_a_gap(
                store, config, corpus_has_entries=corpus_has_entries
            ):
                envelope["note"] = missing_graph_note("mcp")
            return dumps_display(envelope, ensure_ascii=False, indent=2)
        except CollectionMissingError as e:
            # I8 — see surface_decisions. The healthy-empty envelope is BUILT here
            # rather than fallen through to: the envelope above (and its `return`)
            # live inside this `try`, after the query that raised, so there is no
            # empty path to fall through to at all.
            if missing_index_is_a_gap(store):
                return _lexical_degraded_response(
                    query, config=config, reason=degraded_reason_from_error(e),
                    store=store, brief=brief, limit=clamp_limit(limit),
                )
            empty: Dict[str, Any] = {
                "query": query, "depth_mode": "letter", "matches": [],
            }
            empty.update(corpus_provenance(config))
            # W31 — the state this envelope was built for and the unbuilt graph are
            # the SAME workspace on a clone: no *.sqlite, so no collection was ever
            # created either. `missing_index_is_a_gap` said the absence is healthy
            # (an empty active set has nothing to index); this says why the active
            # set is empty, and names the heal that fixes both. Without it the clone
            # gets the cleanest possible empty answer over hundreds of decisions.
            if missing_graph_is_a_gap(
                store, config, corpus_has_entries=corpus_has_entries
            ):
                empty["note"] = missing_graph_note("mcp")
            return dumps_display(empty, ensure_ascii=False, indent=2)
        except Exception as e:
            # Embedding/Qdrant failure mid-query (e.g. a 429): never the raw
            # provider blob — the deterministic lexical fallback instead.
            return _lexical_degraded_response(
                query, config=config, reason=degraded_reason_from_error(e),
                store=store, brief=brief, limit=clamp_limit(limit),
            )

    # No embedding provider / vector store wired at all — degrade lexically.
    return _lexical_degraded_response(
        query, config=config, reason=degraded_reason_from_error(None), store=store,
        brief=brief, limit=clamp_limit(limit),
    )


@mcp.tool()
def record_decision(axiom: str, rejected_paths: str, scope: List[str], slug: str,
                    mechanisms: Optional[List[str]] = None, context: Optional[str] = None,
                    supersedes: Optional[str] = None, corrects: Optional[str] = None,
                    amends: Optional[str] = None,
                    narrows: Optional[str] = None, depends_on: Optional[str] = None,
                    resolves: Optional[str] = None, contradicts: Optional[str] = None,
                    derives_from: Optional[str] = None, cites: Optional[str] = None,
                    acknowledge_neighbors: bool = False,
                    project: Optional[str] = None) -> str:
    """Record a decision you just made, with the alternatives you rejected and why,
    so future sessions and other agents inherit it instead of relitigating it.

    Call this the moment you commit to a foundational choice — a schema, a library,
    a pattern, or a path you've decided to abandon. `rejected_paths` is required:
    recording WHY you ruled options out is what stops you (or the next agent) from
    re-proposing them. If this decision relates to an earlier one, look the earlier
    one up first with query_decisions/surface_decisions and pass its EXACT slug to the
    matching relation arg below (each is validated to point at a real decision). Each
    relation arg also accepts a comma-separated list to link several at once
    (e.g. supersedes="a, b").
    Returns the decision's slug; look it up afterwards with query_decisions.

    Args:
        axiom: The decision as a single clear sentence true going forward.
        rejected_paths: The alternatives considered and rejected, and why. REQUIRED.
        scope: Area tags, e.g. ["database", "auth"].
        mechanisms: Concrete technologies/entities involved, e.g. ["sqlite", "wal-mode"].
        context: Optional background on why this was decided.
        supersedes: Exact slug of a prior decision this one REPLACES (the old one
            becomes superseded). Use this for decision evolution.
        corrects: Exact slug of a prior decision this one CORRECTS (an in-buffer
            correction — the old one leaves the active view, like supersedes; use
            this when the earlier decision was wrong rather than outgrown).
        amends: Exact slug of a decision this one amends (modifies without replacing).
        narrows: Exact slug of a decision this one narrows the scope of.
        depends_on: Exact slug of a decision this one depends on.
        resolves: Exact slug of an open question this one resolves (the resolves edge is decision→open_question only).
        contradicts: Exact slug of a decision this one is in tension with.
        derives_from: Not valid when recording a decision — a derives_from edge
            originates from an open question (open_question -> decision), so a
            decision cannot be its source. Use cites to link a decision this one
            builds on.
        cites: Exact slug of a decision this one cites.
        slug: The short, descriptive handle for the decision (e.g. 'sqlite-wal-mode').
            Keep it to at most 100 characters — the slug is the permanent citation
            handle, so an over-length one is rejected (not silently truncated).
        acknowledge_neighbors: Record past the near-duplicate review after inspecting
            the flagged neighbours and judging this decision genuinely independent.
            Leave False (default) on the first attempt. Combines with the relation
            args — declared edges are still written.
        project: Which project this decision belongs to — REQUIRED on every
            call: a registered project name (e.g. 'mitos') or the absolute path of
            a workspace. Call `list_projects()` if you do not know the names.
            Distinct from `scope`: `project` picks the corpus, `scope` filters
            within it. This is the write — a mis-aimed call lands a real entry in
            another project's corpus.

    Returns:
        A JSON string: {slug, id, state, embedding, status} or {error, code},
        every outcome additionally carrying the trailing {project, collection,
        workspace} naming the corpus this write landed in — check it, since a
        mis-aimed write is the one mistake here that is unpleasant to unwind.
        status="created" means newly recorded; status="exists" is a SUCCESS — the
        identical decision was already recorded and is now confirmed present, not an
        error and not something to retry. Only a top-level {error, code} is a failure.
        status="needs_review" (code "similar_decision_exists") is a PAUSE, not a failure
        and not a write: this decision is ≥0.80 similar to existing `neighbors` it does
        not reference. Each neighbour carries its axiom, rejected_paths, scope, score,
        and an amended_by/narrowed_by stamp when a later decision has moved it on
        (dereference that slug before linking). Judge each neighbour, then re-record
        with that judgment: a relation arg
        (amends/narrows/supersedes/corrects/contradicts/cites) pointing at any
        neighbour this decision genuinely relates to, acknowledge_neighbors=True for
        neighbours that stand independently alongside it — or both at once for a
        mixed set. Nothing was written,
        so a re-record is the right move (unlike an "exists" no-op).
        The "created" result also carries `edges_created` — the relation edges this
        record actually wired, each `{kind, target}` (write facts read back from the
        committed graph, so an empty list means no edge landed) — and the resolved
        `scope`/`mechanisms` as committed.
        NOTE: identity is (slug + axiom + mechanisms). Re-recording an
        existing decision is a no-op — a changed `context`/`rejected_paths`/`scope` or
        relation on a re-record is NOT saved. To record different reasoning or a new
        relationship, make a NEW decision (a distinct axiom), don't resubmit the old one.
        A "created" result MAY carry `neighbor_review_unavailable`: the commit
        succeeded but the near-duplicate review could not run — absent neighbours are
        not checked-clean; a later `check` pass over this project covers the gap
        retroactively.
        The result MAY also carry `scope_overflow`: a one-line, debounced (≤once/24h)
        health nudge that the generated context files have grown past their size ceiling
        — not an error and not about this decision; a status report on this project
        has the breakdown.
    """
    config = _target_config(project, "record_decision")
    # Build our own writable manager — do NOT reuse get_workspace_components()
    # (it opens a read_only=True store). The workspace is the one the call named,
    # resolved once above like the read tools. The import stays LAZY: mcp_server →
    # sync → cli → mcp_server is a real cycle, broken only by all three edges
    # being deferred to call time, and this is one of the three.
    from mitos.sync import MitosSyncManager
    manager = MitosSyncManager(config)
    result = manager.record_decision_entry(
        axiom=axiom,
        rejected_paths=rejected_paths,
        scope=scope,
        mechanisms=mechanisms,
        context=context,
        supersedes=supersedes,
        corrects=corrects,
        amends=amends,
        narrows=narrows,
        depends_on=depends_on,
        resolves=resolves,
        contradicts=contradicts,
        derives_from=derives_from,
        cites=cites,
        slug=slug,
        acknowledge_neighbors=acknowledge_neighbors,
    )
    # The receipt names the corpus it just wrote to — the highest-value stamp in
    # the set precisely because it is the write: a mis-aimed read wastes a turn,
    # a mis-aimed write lands a real entry in another project's gold source.
    # Stamped HERE, at the boundary that serializes it, never inside
    # `record_decision_entry`: the buffer-first + rollback contract is not the
    # place for a routing concern, and the receipt keys it emits (slug/id/state/
    # embedding/status/code/neighbors/message/error/edges_created/scope/
    # mechanisms/…) contain none of these three, so the update adds and never
    # overwrites.
    result.update(corpus_provenance(config))
    return dumps_display(result, ensure_ascii=False, indent=None)


@mcp.tool()
def list_projects() -> str:
    """List every Mitos project registered on this machine, with its workspace path.

    The discovery primitive behind every other tool's `project` argument: this is
    where the vocabulary comes from. Call it when you do not know a project's
    registered name, when a call came back saying the name you passed is not
    registered, or once at the start of a session to learn what this machine
    holds. One round trip, and every later call can name its target.

    It reports registrations, not health — whether a registered path still holds
    a usable workspace is a question for the tool you actually want to call.

    An empty result is a healthy state, not a failure: nothing is registered on
    this machine yet. A workspace can still be targeted by passing its absolute
    path as `project`, which is also the escape hatch for a workspace that exists
    but was never registered.

    It takes no `project` argument, deliberately: it answers for the machine, not
    for one workspace.

    Returns:
        A JSON string: {registry_path, count, projects}, where each entry in
        `projects` is {name, path} — `name` is what you pass as `project`, `path`
        is the workspace it reaches. The order is the registry's own document
        order, never sorted: it is the order a path lookup resolves its first
        match in, so it is the order that actually decides. `registry_path` names
        the file the registrations live in, and is reported whether or not that
        file exists yet.
    """
    return dumps_display(
        projects_payload(registry.load(), registry.registry_path()),
        ensure_ascii=False,
        indent=2,
    )
