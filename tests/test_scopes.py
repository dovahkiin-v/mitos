"""Tests for the scope-discovery surface (Phase 3b): the `mitos scopes` CLI verb,
its `list_scopes` MCP twin, and the `display.order_scope_counts` sort seam.

3a built the data primitive (`GraphStore.get_scope_counts`, with its own exhaustive
store-layer counts==verbs gate); 3b reveals it through two thin surfaces. So these
tests are the *surface* legs of T6 — CLI⇄MCP map/order parity, the busiest-first
ordering, `--archived` adding the dead 0/0 domains, empty-healthy, the casefold key
flowing through, and a single counts==verbs spot-check through the verb (not a
re-proof of 3a's gate).

Forced fully offline (unreachable Qdrant + no keys) so they exercise the pure graph
read and never depend on the machine's running services.
"""

import json
import os
import shutil
import sys
import tempfile
from typing import Iterator, Tuple

import pytest
from unittest.mock import patch

from mitos import routing
from mitos.config import MitosConfig
from mitos.cli import cmd_init, cmd_list, cmd_open_questions, cmd_scopes, main
from mitos.display import order_scope_counts
from mitos.store import GraphStore
from mitos.sync import MitosSyncManager
from mitos.parser import ParsedEntry


@pytest.fixture
def offline(monkeypatch):
    """Forces degraded graph-only mode: unreachable Qdrant, no embedding keys."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9")  # nothing listens here
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def ws(offline) -> Iterator[Tuple[MitosConfig, MitosSyncManager]]:
    """An initialised temp workspace + a manager, in offline graph-only mode.

    `realpath` rather than the raw mkdtemp path: the MCP twin resolves its
    path-form selector through `registry.canonicalize` (= realpath), so on a
    platform whose temp root is a symlink the two surfaces would stamp different
    `workspace` strings and the parity rows would red on the environment rather
    than on a regression.
    """
    tmp = os.path.realpath(tempfile.mkdtemp())
    config = MitosConfig(tmp)
    cmd_init(config)
    config = _resolve_like_main(tmp)
    yield config, MitosSyncManager(config)
    shutil.rmtree(tmp, ignore_errors=True)


def _resolve_like_main(root: str) -> MitosConfig:
    """The config ``cli.main()`` builds for a path-form selector.

    A CLI verb never receives a hand-built config in production — ``main()``
    resolves the selector and passes what it got, so the config carries the
    resolved *name*. The echo is exactly the field where that matters: ``mitos
    init`` registers the workspace, so a path-form selector reverse-looks-up to
    the registered name on both surfaces. Hand-building the config here would
    have the CLI echo a path while the MCP twin echoed the name, and the parity
    rows would red on the test's shortcut rather than on a drift.
    """
    target = routing.resolve_project(root)
    return MitosConfig(target.root, project=target.name)


def _record(m: MitosSyncManager, slug: str, scope, supersedes=None, resolves=None) -> None:
    """Seeds one decision via the agentic write path."""
    res = m.record_decision_entry(
        axiom=f"Axiom for {slug}.",
        rejected_paths=f"Rejected alternative for {slug}.",
        scope=scope,
        slug=slug,
        supersedes=supersedes,
        resolves=resolves,
    )
    assert "error" not in res, res


def _commit_oq(store: GraphStore, slug: str, scope) -> None:
    """Commits a hand-built parked open_question (in the given scope) through the write path."""
    e = ParsedEntry("open_question", slug, 1, 5)
    e.topic = f"Topic for {slug}"
    e.questions_raised = [f"What about {slug}?"]
    e.scope = list(scope)
    store.commit_parsed_entry(e)  # returns a CommitDelta; raises CommitError on failure


def _seed(config) -> None:
    """A representative multi-scope graph: several live domains (one OQ-only and one
    tie), a resolved OQ that must NOT inflate the parked count, and a fully-dead 0/0
    domain that only `--archived` should surface.

    Resulting live map (include_archived=False):
        substrate {3, 0}  store {2, 0}  auth {0, 1}  schema {1, 0}
    Busiest-first, ties alpha → substrate, store, auth, schema.
    With --archived, `dead` {0, 0} joins at the tail.
    """
    m = MitosSyncManager(config)
    store = GraphStore(config.db_path)

    _record(m, "sub-a", scope=["substrate"])
    _record(m, "sub-b", scope=["substrate"])
    _record(m, "sub-c", scope=["substrate"])
    _record(m, "store-a", scope=["store"])
    _record(m, "store-b", scope=["store"])
    _record(m, "schema-a", scope=["schema"])

    # auth: live via a parked OQ only (0 active decisions).
    _commit_oq(store, "q-auth", scope=["auth"])

    # A resolved OQ in `store`: its node carries the `store` scope tag, but being
    # resolved it must NOT count toward store's parked total (the 3a gotcha).
    _commit_oq(store, "q-store-done", scope=["store"])
    _record(m, "store-resolver", scope=[], resolves="q-store-done")

    # dead: a decision superseded by a SCOPELESS superseder → the domain computes to
    # 0/0 (tagging the superseder into `dead` would keep it live and defeat the intent).
    _record(m, "dead-v1", scope=["dead"])
    _record(m, "dead-v2", scope=[], supersedes="dead-v1")


def _mcp_scopes(config, **kwargs) -> str:
    """Calls the MCP `list_scopes` tool against a read-only store on this workspace.

    The workspace is named in **path form**, and that is load-bearing rather than
    tidy. The `ws` fixture never chdirs, so a `project`-less call would resolve
    pytest's cwd — the mitos-pub repo, which is itself a valid workspace — and the
    tool would stamp the repo's collection while the CLI twin stamped the temp
    workspace's. Every parity row below would red, reading as a broken stamp when
    the truth is that the helper never named a target. The path form needs no
    registry and resolves to exactly this config's root.

    `_target_config` is deliberately left to run (only `get_workspace_components`
    is patched): a row that mocked the resolution could not observe the echo at
    all.
    """
    from mitos import mcp_server
    store = GraphStore(config.db_path, read_only=True)
    with patch.object(mcp_server, "get_workspace_components", return_value=(store, None, None)):
        return mcp_server.list_scopes(project=config.workspace_dir, **kwargs)


# --------------------------------------------------------------------------- #
# display.order_scope_counts — the sort seam (unit)
# --------------------------------------------------------------------------- #

def test_order_scope_counts_busiest_first_ties_alpha() -> None:
    """Total live-count descending; equal totals break alphabetically by tag."""
    raw = {  # alpha-ordered, as get_scope_counts returns it
        "auth": {"active_decisions": 0, "parked_open_questions": 1},
        "schema": {"active_decisions": 1, "parked_open_questions": 0},
        "store": {"active_decisions": 2, "parked_open_questions": 0},
        "substrate": {"active_decisions": 3, "parked_open_questions": 0},
    }
    assert list(order_scope_counts(raw)) == ["substrate", "store", "auth", "schema"]


def test_order_scope_counts_empty() -> None:
    """An empty vocabulary orders to an empty dict (never an error)."""
    assert order_scope_counts({}) == {}


# --------------------------------------------------------------------------- #
# CLI cmd_scopes --json
# --------------------------------------------------------------------------- #

def test_cmd_scopes_json_ordering(ws, capsys) -> None:
    """`mitos scopes --json` emits the ordered map, busiest domain first, ties alpha.

    The map now sits under `scopes` inside the provenance envelope; the ordering
    contract moved down with it and is asserted where it now lives.
    """
    config, _ = ws
    _seed(config)
    capsys.readouterr()  # drain the init banner
    cmd_scopes(config, as_json=True)
    envelope = json.loads(capsys.readouterr().out)
    out = envelope["scopes"]
    assert list(out) == ["substrate", "store", "auth", "schema"]
    assert out["substrate"] == {"active_decisions": 3, "parked_open_questions": 0}
    assert out["auth"] == {"active_decisions": 0, "parked_open_questions": 1}
    # The envelope names the corpus the vocabulary came from.
    assert envelope["project"] == config.project
    assert envelope["collection"] == config.qdrant_collection
    assert envelope["workspace"] == config.workspace_dir


def test_cmd_scopes_archived_adds_dead_domain(ws, capsys) -> None:
    """The fully-dead 0/0 domain is absent by default, present at 0/0 under --archived."""
    config, _ = ws
    _seed(config)
    capsys.readouterr()

    cmd_scopes(config, as_json=True)
    live = json.loads(capsys.readouterr().out)["scopes"]
    assert "dead" not in live

    cmd_scopes(config, as_json=True, archived=True)
    archived = json.loads(capsys.readouterr().out)["scopes"]
    assert archived["dead"] == {"active_decisions": 0, "parked_open_questions": 0}
    # The dead 0/0 domain sorts to the tail (lowest total).
    assert list(archived)[-1] == "dead"


def test_cmd_scopes_text_table(ws, capsys) -> None:
    """The text table lists the busiest domain first and is calm (no error wording)."""
    config, _ = ws
    _seed(config)
    capsys.readouterr()
    cmd_scopes(config)
    out = capsys.readouterr().out
    assert "substrate" in out
    # Busiest domain appears before the lighter ones in the rendered order.
    assert out.index("substrate") < out.index("schema")


def test_cmd_scopes_empty_is_healthy(ws, capsys) -> None:
    """A just-init'd workspace: an empty vocabulary under --json, a calm message in
    text, exit 0 — and the empty answer now says WHICH project was empty.

    That attribution is the feature, not a dilution of the empty-is-healthy rule:
    `{}` from a fresh project and `{}` from the wrong project used to be the same
    answer.
    """
    config, _ = ws
    capsys.readouterr()
    cmd_scopes(config, as_json=True)
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["scopes"] == {}
    assert envelope["project"] == config.project
    assert envelope["collection"] == config.qdrant_collection
    assert envelope["workspace"] == config.workspace_dir

    cmd_scopes(config)
    text = capsys.readouterr().out
    assert "No scopes yet" in text
    assert "error" not in text.lower()


def test_cmd_scopes_casefold_key(ws, capsys) -> None:
    """A decision recorded with scope ['Auth'] surfaces under the casefolded key 'auth'."""
    config, m = ws
    _record(m, "cap-one", scope=["Auth"])
    capsys.readouterr()
    cmd_scopes(config, as_json=True)
    # Read the nested map, not the envelope: against the top level `"Auth" not in`
    # would be trivially true and would stop testing the casefold.
    out = json.loads(capsys.readouterr().out)["scopes"]
    assert "auth" in out
    assert "Auth" not in out


# --------------------------------------------------------------------------- #
# MCP list_scopes
# --------------------------------------------------------------------------- #

def test_mcp_list_scopes_ordering(ws) -> None:
    """`list_scopes` returns the same ordered map JSON, busiest first."""
    config, _ = ws
    _seed(config)
    envelope = json.loads(_mcp_scopes(config))
    assert list(envelope["scopes"]) == ["substrate", "store", "auth", "schema"]
    assert envelope["project"] == config.project
    assert envelope["collection"] == config.qdrant_collection
    assert envelope["workspace"] == config.workspace_dir


def test_mcp_list_scopes_archived(ws) -> None:
    """`include_archived=True` adds the dead 0/0 domain on the MCP surface too."""
    config, _ = ws
    _seed(config)
    assert "dead" not in json.loads(_mcp_scopes(config))["scopes"]
    archived = json.loads(_mcp_scopes(config, include_archived=True))["scopes"]
    assert archived["dead"] == {"active_decisions": 0, "parked_open_questions": 0}


def test_mcp_list_scopes_empty_is_healthy(ws) -> None:
    """An empty/fresh project returns an empty vocabulary — never an error — and
    names the project it was empty for."""
    config, _ = ws
    envelope = json.loads(_mcp_scopes(config))
    assert envelope["scopes"] == {}
    assert envelope["project"] == config.project
    assert envelope["collection"] == config.qdrant_collection
    assert envelope["workspace"] == config.workspace_dir


def test_mcp_list_scopes_envelope_nests_the_ordering_contract(ws) -> None:
    """The ordering contract lives on the nested map, and the stamp cannot reach it.

    The nesting is not cosmetic: scope tags are user-authored strings, so a project
    whose vocabulary literally holds `project`/`collection`/`workspace` must keep
    those tags — and their counts — rather than have them overwritten by the
    provenance. A merged-into-one-map build reds here.
    """
    config, m = ws
    for tag in ("project", "collection", "workspace"):
        _record(m, f"{tag}-probe", scope=[tag])

    envelope = json.loads(_mcp_scopes(config))

    assert list(envelope) == ["scopes", "project", "collection", "workspace"]
    assert list(envelope["scopes"]) == ["collection", "project", "workspace"]  # ties alpha
    for tag in ("project", "collection", "workspace"):
        assert envelope["scopes"][tag] == {"active_decisions": 1,
                                           "parked_open_questions": 0}
    assert envelope["workspace"] == config.workspace_dir  # the path, not the count


def test_mcp_list_scopes_registered() -> None:
    """list_scopes is the 5th registered MCP tool, alongside surface/query/list/record."""
    import asyncio
    from mitos.mcp_server import mcp
    names = [t.name for t in asyncio.run(mcp.list_tools())]
    assert "list_scopes" in names


# --------------------------------------------------------------------------- #
# T6 — CLI⇄MCP parity (the definition of done)
# --------------------------------------------------------------------------- #

def test_cli_mcp_map_parity(ws, capsys) -> None:
    """The `scopes --json` map and `list_scopes` map are the SAME ordered dict —
    equal parsed maps with identical key order, and serialized bodies equal modulo
    the CLI `print` newline (T6). Run under default UTF-8 capsys."""
    config, _ = ws
    _seed(config)
    capsys.readouterr()
    cmd_scopes(config, as_json=True)
    cli_out = capsys.readouterr().out
    mcp_out = _mcp_scopes(config)

    assert json.loads(cli_out) == json.loads(mcp_out)
    # key order IS the deliverable — at both levels of the envelope
    assert list(json.loads(cli_out)) == list(json.loads(mcp_out))
    assert list(json.loads(cli_out)["scopes"]) == list(json.loads(mcp_out)["scopes"])
    assert cli_out.rstrip("\n") == mcp_out  # only the CLI print newline differs


def test_cli_mcp_parity_with_archived(ws, capsys) -> None:
    """Parity holds under --archived / include_archived=True (dead domains included)."""
    config, _ = ws
    _seed(config)
    capsys.readouterr()
    cmd_scopes(config, as_json=True, archived=True)
    cli_out = capsys.readouterr().out
    mcp_out = _mcp_scopes(config, include_archived=True)
    assert json.loads(cli_out) == json.loads(mcp_out)
    assert list(json.loads(cli_out)["scopes"]) == list(json.loads(mcp_out)["scopes"])


# --------------------------------------------------------------------------- #
# Counts==verbs surface spot-check — the map reaches the surface intact
# --------------------------------------------------------------------------- #

def test_counts_match_read_verbs_through_surface(ws, capsys) -> None:
    """One scope's surfaced counts equal the read verbs' sizes — confirms 3a's map
    carries through the verb faithfully (not a re-proof of 3a's exhaustive gate)."""
    config, _ = ws
    _seed(config)
    capsys.readouterr()
    cmd_scopes(config, as_json=True)
    scopes = json.loads(capsys.readouterr().out)["scopes"]

    # active_decisions for `substrate` == len(list --scope substrate --json decisions)
    cmd_list(config, scope="substrate", as_json=True)
    listed = json.loads(capsys.readouterr().out)
    assert scopes["substrate"]["active_decisions"] == len(listed["decisions"])

    # parked_open_questions for `auth` == the parked OQ subset for that scope
    cmd_open_questions(config, scope="auth", as_json=True)
    oqs = json.loads(capsys.readouterr().out)
    assert scopes["auth"]["parked_open_questions"] == oqs["total"]


# --------------------------------------------------------------------------- #
# Alias routing
# --------------------------------------------------------------------------- #

@patch("mitos.cli.cmd_scopes")
def test_list_scopes_alias_routes(mock_scopes, monkeypatch, workspace) -> None:
    """The MCP-name alias `list_scopes` routes to cmd_scopes with the flags plumbed."""
    monkeypatch.setattr(sys, "argv", ["mitos", "-p", workspace, "list_scopes", "--json", "--archived"])
    main()
    mock_scopes.assert_called_once()
    _, kwargs = mock_scopes.call_args
    assert kwargs["as_json"] is True
    assert kwargs["archived"] is True
