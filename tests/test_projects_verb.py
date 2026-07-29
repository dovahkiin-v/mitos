"""Tests for the ``mitos projects`` verb — the registry's discovery read.

Driven through ``main()`` (patched ``sys.argv``) rather than by calling
``cmd_projects`` directly, because half of what this verb promises is boundary
behaviour: it must work with no workspace anywhere, exit 0 on an empty registry,
and render a hand-broken registry as one calm line at exit 1 instead of a
traceback. The autouse ``hermetic_mitos_env`` fixture redirects the config root
per test, so each row gets its own registry file.
"""

import json
import os
import sys
from unittest.mock import patch

import pytest

from mitos import registry
from mitos.cli import main


def _run_projects(*flags):
    """Runs ``mitos projects`` through the real entry point."""
    with patch.object(sys, "argv", ["mitos", "projects", *flags]):
        main()


def _write_registry(text: str) -> str:
    path = registry.registry_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# --- the populated read ----------------------------------------------------

def test_text_form_lists_every_project_in_registry_order(capsys):
    """The listing shows name + path in FILE order and names the resolved registry.

    File order, never sorted: it is the order a reverse lookup resolves its first
    match in, so a human reading this must see the order that actually decides.
    The resolved path is printed rather than a literal ``~/.config/…`` so the line
    stays true under an ``XDG_CONFIG_HOME`` that points elsewhere.
    """
    _write_registry('"zulu" = "/ws/zulu"\n"alpha" = "/ws/alpha"\n')

    _run_projects()

    out = capsys.readouterr().out
    assert "zulu" in out and "/ws/zulu" in out
    assert "alpha" in out and "/ws/alpha" in out
    assert out.index("zulu") < out.index("alpha")   # registry order, not alphabetical
    assert registry.registry_path() in out
    assert "2 registered" in out


def test_text_form_carries_no_health_verdict(capsys):
    """The listing reports registrations, not reachability.

    Whether a recorded path still holds a valid workspace is ``mitos status``'s
    question. A flag here would be a second source of that answer, and two sources
    drift — so this verb must not probe the filesystem at all.
    """
    _write_registry('"gone" = "/ws/deleted-long-ago"\n')

    _run_projects()

    out = capsys.readouterr().out
    assert "/ws/deleted-long-ago" in out  # listed exactly as recorded, unjudged
    for verdict in ("missing", "not found", "invalid", "unreachable", "✗", "✓"):
        assert verdict not in out.lower(), (
            f"health verdict {verdict!r} leaked into `projects` — health is `status`'s"
        )


def test_json_form_emits_the_documented_payload(capsys):
    """``--json`` returns exactly ``registry_path`` / ``count`` / ``projects``.

    This payload is the shape a later phase lifts into a shared leaf for the MCP
    twin, so it stays JSON-native: plain dicts, no ``None``s, and no presentation
    strings an agent surface could inherit.
    """
    _write_registry('"alpha" = "/ws/alpha"\n"beta" = "/ws/beta"\n')

    _run_projects("--json")

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "registry_path": registry.registry_path(),
        "count": 2,
        "projects": [
            {"name": "alpha", "path": "/ws/alpha"},
            {"name": "beta", "path": "/ws/beta"},
        ],
    }
    assert payload["count"] == len(payload["projects"])


def test_json_form_round_trips_a_non_ascii_name(capsys):
    """A non-ASCII project name survives to the JSON payload unchanged (P9)."""
    _write_registry('"ąžuolas" = "/ws/oak"\n')

    _run_projects("--json")

    payload = json.loads(capsys.readouterr().out)
    assert payload["projects"] == [{"name": "ąžuolas", "path": "/ws/oak"}]


# --- healthy and empty -----------------------------------------------------

@pytest.mark.parametrize(
    "seed",
    [None, "", "# nothing registered yet, just my notes\n"],
    ids=["missing-file", "empty-file", "comments-only"],
)
def test_an_empty_or_missing_registry_is_the_healthy_empty_state(capsys, seed):
    """Nothing registered reads as "no projects yet" at exit 0 — never an error.

    A fresh machine has no registry file at all, and this is the verb most likely
    to be met in that state. It must not raise, must not print a bare
    ``Projects:`` header with nothing under it, and must not exit non-zero — an
    agent reads a non-zero exit as a call-syntax mistake and retries.
    """
    if seed is not None:
        _write_registry(seed)

    _run_projects()  # no SystemExit: exit 0 is the fall-through

    out = capsys.readouterr().out
    assert "No projects registered yet" in out
    assert "mitos init" in out                 # the vector out of the empty state
    assert registry.registry_path() in out
    assert "registered, in registry order" not in out  # no empty table header


def test_json_on_an_empty_registry_returns_the_empty_payload_not_the_prose(capsys):
    """``--json`` emits ``{"projects": [], "count": 0, …}`` even when empty.

    The JSON branch precedes the healthy-empty branch on purpose: a machine
    consumer asking for JSON must always get parseable JSON, never a prose line it
    has to sniff for.
    """
    _run_projects("--json")

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "registry_path": registry.registry_path(),
        "count": 0,
        "projects": [],
    }


def test_projects_works_where_there_is_no_workspace_at_all(capsys, tmp_path, monkeypatch):
    """A global verb run from a directory with no ``.mitos/`` still answers.

    ``main()`` builds a working-directory ``MitosConfig`` before dispatch, which
    looks like it would break a workspace-less verb — it does not, because the
    config loader only reads ``config.toml`` when the file exists. Pinned so a
    later change to that construction cannot quietly make the fresh-machine case
    fail.
    """
    bare = tmp_path / "not-a-workspace"
    bare.mkdir()
    monkeypatch.chdir(bare)
    _write_registry('"alpha" = "/ws/alpha"\n')

    _run_projects()

    assert "alpha" in capsys.readouterr().out


# --- the malformed-registry boundary --------------------------------------

@pytest.mark.parametrize(
    "broken, cause",
    [
        ('"alpha" = "/ws/alpha"\n"beta" =\n', "a value-less key"),
        ('"alpha" = { path = "/ws/alpha" }\n', "a table where a path belongs"),
        ('"alpha" = "/a"\n"alpha" = "/b"\n', "a duplicated name"),
    ],
)
def test_a_hand_broken_registry_renders_one_calm_line_at_exit_1(capsys, broken, cause):
    """Every unusable-registry shape exits 1 with one ``Error:`` line and no traceback.

    The file is hand-editable, so a broken one is an ordinary state to be met in —
    and the operator needs the path, the cause, and the fix, not a stack dump. No
    new boundary carries this: ``RegistryError`` is a ``MitosError``, so the
    shipped ``except MitosError`` arm renders it.
    """
    path = _write_registry(broken)

    with pytest.raises(SystemExit) as exc:
        _run_projects()

    assert exc.value.code == 1, cause
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert captured.err.count("Error:") == 1
    assert path in captured.err
    assert captured.out == ""  # no half-rendered table before the refusal


def test_json_on_a_broken_registry_still_refuses_rather_than_emitting_a_lie(capsys):
    """``--json`` over an unusable registry errors out instead of reporting zero projects.

    Rendering ``{"projects": [], "count": 0}`` for a file that could not be read
    would tell an agent the machine has no projects, which is a different — and
    silently wrong — claim from "the registry is broken".
    """
    _write_registry('"alpha" =\n')

    with pytest.raises(SystemExit) as exc:
        _run_projects("--json")

    assert exc.value.code == 1
    assert capsys.readouterr().out == ""


# --- grammar ---------------------------------------------------------------

def test_projects_takes_no_abbreviated_flag():
    """``--js`` is not accepted: abbreviation stays off on this verb like every other.

    The subparser has to be registered above ``_build_parser``'s ``allow_abbrev``
    sweep; one added after it silently abbreviates, and a prefix bug on a verb that
    writes is how a 470-character axiom once went to a file reader.
    """
    with patch.object(sys, "argv", ["mitos", "projects", "--js"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2  # argparse usage error, not a silent acceptance
