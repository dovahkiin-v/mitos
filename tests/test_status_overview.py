"""The global overview: what this machine has, and is any of it broken.

Everything provable without a service lives here — the state vocabulary, per-entry
isolation, the wall-clock bound, the local join against a *faked* listing, the payload
shape, the call budget and the exit contract. The rows that need a real Qdrant are in
``tests/test_status_overview_live.py``; neither module's claims overlap the other's.

Two rows here drive a **subprocess** rather than ``capsys``, and in both cases that is
the whole point rather than an inconvenience:

* the bound's claim is *"the shell gets its prompt back"*, and only total process wall
  measured from outside can see it — a ``ThreadPoolExecutor`` bound looks correct from
  inside the process and still costs the full park at interpreter exit;
* the flush's claim is about **stream interleaving on one pipe**, and ``capsys`` keeps
  the streams apart, so it structurally cannot observe the inversion.
"""

import ast
import json
import os
import subprocess
import sys
import textwrap
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import requests

from mitos import cli, overview, registry
from mitos.config import default_collection_name
from mitos.errors import RegistryError, VectorStoreError


# --- fixtures & helpers ----------------------------------------------------

def _write_registry(text: str) -> str:
    """Hand-writes the registry file (the hand-editable states we must tolerate)."""
    path = registry.registry_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _register(**projects: str) -> str:
    """Writes a registry from ``name=path`` pairs, in the order given."""
    return _write_registry(
        "".join(f'"{name}" = "{path}"\n' for name, path in projects.items()))


def _workspace(root) -> str:
    """Builds the shipped validity triple at ``root`` and returns its canonical path.

    Canonical (``realpath``), never the raw fixture path: the registry stores what
    ``registry.canonicalize`` produced, and the two happen to be identical on this
    box — so a row that gets it wrong passes anyway and rots elsewhere.
    """
    root = str(root)
    os.makedirs(os.path.join(root, ".mitos"), exist_ok=True)
    with open(os.path.join(root, ".mitos", "config.toml"), "w") as f:
        f.write("")
    with open(os.path.join(root, "decisions.md"), "w") as f:
        f.write("")
    return os.path.realpath(root)


def _listing_response(names: List[str]) -> MagicMock:
    """A healthy ``GET /collections`` answer carrying ``names``."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "result": {"collections": [{"name": name} for name in names]}
    }
    return resp


@pytest.fixture
def qdrant(monkeypatch):
    """Fakes the instance boundary at ``requests``, recording every URL asked for.

    Patched in ``mitos.vector_store``'s namespace — the only place the overview's
    network I/O happens — so the recorded calls are the real request shapes, which is
    what makes the "never a per-collection probe" assertion mean something.
    """
    class _Fake:
        def __init__(self) -> None:
            self.urls: List[str] = []
            self.names: Dict[str, List[str]] = {}
            self.faults: Dict[str, Any] = {}

        def get(self, url: str, timeout: float = 0) -> MagicMock:
            self.urls.append(url)
            base = url[: -len("/collections")]
            fault = self.faults.get(base)
            if fault is not None:
                raise fault
            return _listing_response(self.names.get(base, []))

    fake = _Fake()
    monkeypatch.setattr("mitos.vector_store.requests.get", fake.get)
    return fake


def _by_name(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {project["name"]: project for project in payload["projects"]}


# --- the closed vocabulary and the constants -------------------------------

def test_the_entry_state_vocabulary_is_closed_and_pinned():
    """Pinned per-constant AND as a set, on ``TARGETING_DISCRIMINATORS``' idiom.

    The names are what a ``--json`` consumer switches on, so a typo is a silent
    break; the set is what keeps a sixth state from being added without a decision.
    """
    assert overview.STATE_OK == "ok"
    assert overview.STATE_MISSING == "missing"
    assert overview.STATE_NOT_A_WORKSPACE == "not_a_workspace"
    assert overview.STATE_UNRESPONSIVE == "unresponsive"
    assert overview.STATE_ERROR == "error"
    assert overview.ENTRY_STATES == frozenset(
        {"ok", "missing", "not_a_workspace", "unresponsive", "error"})


def test_the_instance_budget_strictly_exceeds_the_socket_timeout():
    """The outer budget backstops what a per-socket-op timeout does not bound.

    A slow-trickle response resets the socket timeout on every byte, so the phase
    budget is the only real bound on it. At or below the socket timeout, the socket
    timeout would be dead code — an inequality, pinned rather than assumed.
    """
    assert overview.INSTANCE_PROBE_BUDGET > overview.INSTANCE_SOCKET_TIMEOUT
    assert overview.LOCAL_PROBE_BUDGET > 0


def test_every_rendered_mark_covers_the_whole_vocabulary():
    """The CLI's mark table and the leaf's vocabulary stay in lockstep.

    A state added to the leaf without a mark would render through the ``.get``
    fallback — honest and useless, the fourth instance of this vision's most-repeated
    defect shape.
    """
    assert set(cli._OVERVIEW_MARKS) == overview.ENTRY_STATES


# --- states, one per outcome, each with a surviving sibling ----------------

def test_every_entry_is_reported_in_document_order_with_a_legal_state(tmp_path, qdrant):
    """Success criterion 1: one entry per registration, in file order."""
    healthy = _workspace(tmp_path / "healthy")
    _register(zeta=healthy, alpha=str(tmp_path / "nowhere"))

    payload = overview.build_overview()

    assert [p["name"] for p in payload["projects"]] == ["zeta", "alpha"]
    assert payload["count"] == 2
    assert payload["registry_path"] == registry.registry_path()
    assert payload["report"] == "overview"
    for project in payload["projects"]:
        assert project["state"] in overview.ENTRY_STATES


def test_a_vanished_path_is_missing_and_a_present_non_workspace_is_not(tmp_path, qdrant):
    """The two halves of "path exists + workspace valid", as one discriminator."""
    healthy = _workspace(tmp_path / "healthy")
    bare = tmp_path / "bare"
    bare.mkdir()
    _register(gone=str(tmp_path / "nowhere"), bare=str(bare), healthy=healthy)

    projects = _by_name(overview.build_overview())

    assert projects["gone"]["state"] == "missing"
    assert projects["bare"]["state"] == "not_a_workspace"
    assert projects["healthy"]["state"] == "ok"
    # Nothing was constructed for the two broken ones, so they carry no URL and no
    # derived name — which is also why they cost no network call.
    for name in ("gone", "bare"):
        assert projects[name]["qdrant_url"] is None
        assert projects[name]["collection"] is None
        assert projects[name]["collection_present"] is None


def test_a_malformed_target_config_renders_as_that_entry_s_own_error(tmp_path, qdrant):
    """A genuine raise from a real target — not a patched seam.

    ``MitosConfig`` raises ``ConfigError`` on malformed TOML, and the validity triple
    passes (the file exists), so this is the ordinary shape of a probe that raises.
    """
    broken = _workspace(tmp_path / "broken")
    with open(os.path.join(broken, ".mitos", "config.toml"), "w") as f:
        f.write("this is not = = toml")
    healthy = _workspace(tmp_path / "healthy")
    _register(broken=broken, healthy=healthy)

    projects = _by_name(overview.build_overview())

    assert projects["broken"]["state"] == "error"
    assert "config.toml" in projects["broken"]["error"]
    assert "Traceback" not in projects["broken"]["error"]
    # The isolation property: the OTHER entry is fully reported.
    assert projects["healthy"]["state"] == "ok"
    assert projects["healthy"]["collection"] == default_collection_name(healthy)


def test_one_raising_probe_never_ends_the_sweep(tmp_path, monkeypatch, qdrant):
    """The isolation row. Fault injection: let the exception escape and this reds.

    Raised for **one named entry**, which is why ``probe_entry`` takes the name: a
    global raise would prove only that the sweep dies, not that it isolates.
    """
    first = _workspace(tmp_path / "first")
    second = _workspace(tmp_path / "second")
    third = _workspace(tmp_path / "third")
    _register(first=first, second=second, third=third)

    real = overview.probe_entry

    def exploding(name: str, path: str) -> Dict[str, Any]:
        if name == "second":
            raise OSError("[Errno 40] Too many levels of symbolic links")
        return real(name, path)

    monkeypatch.setattr(overview, "probe_entry", exploding)
    projects = _by_name(overview.build_overview())

    assert projects["second"]["state"] == "error"
    assert "symbolic links" in projects["second"]["error"]
    assert [projects[n]["state"] for n in ("first", "third")] == ["ok", "ok"]


def test_an_unresponsive_entry_is_not_a_missing_one(tmp_path, monkeypatch, qdrant):
    """A contract, not a nicety: a hung mount is not a deleted directory.

    Telling a human their project is gone when the disk is merely asleep is a wall,
    so the two states are distinguishable in the JSON **and** in the text.
    """
    parked = _workspace(tmp_path / "parked")
    _register(parked=parked, gone=str(tmp_path / "nowhere"))

    real = overview.probe_entry

    def slow(name: str, path: str) -> Dict[str, Any]:
        if name == "parked":
            time.sleep(30)
        return real(name, path)

    monkeypatch.setattr(overview, "probe_entry", slow)
    monkeypatch.setattr(overview, "LOCAL_PROBE_BUDGET", 0.2)

    started = time.monotonic()
    payload = overview.build_overview()
    elapsed = time.monotonic() - started

    projects = _by_name(payload)
    assert projects["parked"]["state"] == "unresponsive"
    assert projects["gone"]["state"] == "missing"
    assert projects["parked"]["error"] is None  # nothing was concluded about it
    assert elapsed < 10, "the shared deadline did not bound the local phase"

    rendered = _render_to_text(payload)
    assert "unresponsive" in rendered and "missing" in rendered
    assert "unresponsive mount, not a missing project" in rendered


def test_the_local_phase_is_bounded_by_ONE_shared_deadline(tmp_path, monkeypatch, qdrant):
    """Three parked entries cost one budget, not three.

    Per-entry × N is O(N × budget) — twenty projects behind a dead NAS would take
    twenty times too long, which fails the from-anywhere habit the bound protects.
    """
    _register(**{f"parked{i}": str(tmp_path / f"ws{i}") for i in range(3)})
    monkeypatch.setattr(overview, "probe_entry", lambda name, path: time.sleep(30))
    monkeypatch.setattr(overview, "LOCAL_PROBE_BUDGET", 0.4)

    started = time.monotonic()
    payload = overview.build_overview()
    elapsed = time.monotonic() - started

    assert {p["state"] for p in payload["projects"]} == {"unresponsive"}
    assert elapsed < 1.2, f"three parked entries cost {elapsed:.2f}s — not one budget"


# --- the instance phase: bulk-shaped, tri-stated ---------------------------

def test_two_projects_on_one_url_cause_exactly_one_probe(tmp_path, qdrant):
    """Success criterion 5, first half — the budget scales with instances.

    Fault injection: probe per project instead of per distinct URL and this reds.
    """
    first = _workspace(tmp_path / "first")
    second = _workspace(tmp_path / "second")
    _register(first=first, second=second)

    overview.build_overview()

    assert qdrant.urls == ["http://localhost:7333/collections"]


def test_distinct_urls_are_each_probed_and_attributed(tmp_path, monkeypatch, qdrant):
    """Two instances, two probes, each project joined to its own.

    The overview must not assume a single instance: skipping a non-localhost URL
    would hand the least honest answer to the instance most likely to be down.
    """
    # An exported QDRANT_URL beats every file tier, so a distinct-URL fixture
    # silently collapses to one URL on a box that exports it. A bare
    # `delenv(raising=False)` on an already-absent name records nothing, so force it.
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.delenv("QDRANT_URL")

    here = _workspace(tmp_path / "here")
    there = _workspace(tmp_path / "there")
    with open(os.path.join(there, ".env"), "w") as f:
        f.write("QDRANT_URL=http://127.0.0.1:7333\n")
    _register(here=here, there=there)
    qdrant.names["http://localhost:7333"] = [default_collection_name(here)]
    qdrant.names["http://127.0.0.1:7333"] = []

    payload = overview.build_overview()
    projects = _by_name(payload)

    assert sorted(qdrant.urls) == [
        "http://127.0.0.1:7333/collections",
        "http://localhost:7333/collections",
    ]
    assert projects["here"]["collection_present"] is True
    assert projects["there"]["collection_present"] is False
    assert [i["url"] for i in payload["instances"]] == [
        "http://localhost:7333", "http://127.0.0.1:7333"]


def test_no_request_is_ever_made_to_a_per_collection_path(tmp_path, qdrant):
    """The shape §4.9 forbids — one call *per project* — is absent by construction."""
    _register(**{
        f"p{i}": _workspace(tmp_path / f"ws{i}") for i in range(4)
    })

    overview.build_overview()

    assert len(qdrant.urls) == 1
    assert all(url.endswith("/collections") for url in qdrant.urls)


def test_a_registry_of_nothing_but_broken_entries_issues_zero_network_calls(
        tmp_path, qdrant):
    """No entry yielded a URL, so there is no instance to ask about."""
    bare = tmp_path / "bare"
    bare.mkdir()
    _register(gone=str(tmp_path / "nowhere"), bare=str(bare))

    payload = overview.build_overview()

    assert qdrant.urls == []
    assert payload["instances"] == []


def test_an_absent_collection_is_false_and_an_unavailable_listing_is_null(
        tmp_path, qdrant):
    """The tri-state's middle row — the reason it is not decoration.

    Fault injection: collapse ``collection_present`` to ``False`` when the listing is
    unavailable and this reds. That collapse is the one that would tell every project
    on a sick instance that its vectors are gone.
    """
    present = _workspace(tmp_path / "present")
    absent = _workspace(tmp_path / "absent")
    _register(present=present, absent=absent)
    qdrant.names["http://localhost:7333"] = [default_collection_name(present)]

    projects = _by_name(overview.build_overview())
    assert projects["present"]["collection_present"] is True
    assert projects["absent"]["collection_present"] is False

    # …and now the same instance answers with something unusable.
    qdrant.faults["http://localhost:7333"] = VectorStoreError("boom")
    payload = overview.build_overview()
    projects = _by_name(payload)
    assert projects["present"]["collection_present"] is None
    assert projects["absent"]["collection_present"] is None
    assert payload["instances"][0]["reachable"] is True
    assert payload["instances"][0]["listing_available"] is False
    assert payload["instances"][0]["collection_count"] is None


def test_an_unreachable_instance_reports_no_answer_and_the_sweep_completes(
        tmp_path, qdrant):
    """Row one of the tri-state, and the isolation property at the instance level."""
    project = _workspace(tmp_path / "project")
    _register(project=project)
    qdrant.faults["http://localhost:7333"] = requests.exceptions.ConnectionError("refused")

    payload = overview.build_overview()

    assert payload["projects"][0]["state"] == "ok"
    assert payload["projects"][0]["collection_present"] is None
    assert payload["instances"] == [{
        "url": "http://localhost:7333",
        "reachable": False,
        "listing_available": False,
        "collection_count": None,
        "error": payload["instances"][0]["error"],
    }]
    assert "connection error" in payload["instances"][0]["error"]


def test_an_exported_empty_qdrant_url_renders_as_unreachable_rather_than_crashing(
        tmp_path, monkeypatch, qdrant):
    """``""`` resolves to ``""`` and stays ``""`` — a supplied answer, not a default.

    ``requests.get("/collections")`` raises ``MissingSchema``, a ``RequestException``
    subclass, so it lands on the transport arm and renders as an instance that did
    not answer.
    """
    monkeypatch.setenv("QDRANT_URL", "")
    project = _workspace(tmp_path / "project")
    _register(project=project)
    qdrant.faults[""] = requests.exceptions.MissingSchema("no scheme")

    payload = overview.build_overview()

    assert payload["projects"][0]["qdrant_url"] == ""
    assert payload["projects"][0]["collection_present"] is None
    assert payload["instances"][0]["reachable"] is False


def test_a_parked_instance_probe_is_bounded_and_renders_as_no_answer(
        tmp_path, monkeypatch, qdrant):
    """The instance phase carries its own shared deadline."""
    _register(project=_workspace(tmp_path / "project"))
    monkeypatch.setattr(overview, "probe_instance", lambda url: time.sleep(30))
    monkeypatch.setattr(overview, "INSTANCE_PROBE_BUDGET", 0.2)

    started = time.monotonic()
    payload = overview.build_overview()

    assert time.monotonic() - started < 10
    assert payload["instances"][0]["reachable"] is False
    assert payload["projects"][0]["collection_present"] is None


def test_the_join_key_is_the_derivation_and_not_a_hand_built_name(tmp_path, qdrant):
    """Success criterion 12's behavioural twin.

    The workspace is deliberately **not** named after its own basename-derived
    collection: a fixture whose hand-built name happens to match the derivation
    proves nothing. Fault injection: join on ``f"mitos-{basename}"`` and this reds.
    """
    project = _workspace(tmp_path / "project")
    _register(project=project)
    qdrant.names["http://localhost:7333"] = ["mitos-project"]  # the pre-1d shape

    payload = overview.build_overview()

    assert payload["projects"][0]["collection"] == default_collection_name(project)
    assert payload["projects"][0]["collection"] != "mitos-project"
    assert payload["projects"][0]["collection_present"] is False


def test_the_overview_opens_no_sqlite_connection(tmp_path, qdrant):
    """D6 as a fact rather than a statement — presence never implies population."""
    _register(project=_workspace(tmp_path / "project"))

    with patch("sqlite3.connect") as connect:
        overview.build_overview()

    connect.assert_not_called()


# --- the registry-level findings -------------------------------------------

def test_two_names_for_one_path_are_a_finding_and_never_a_fault(tmp_path, qdrant):
    """Both entries keep the state their own probe found; the finding is additive.

    A tolerated hand-edit: both names reach the same workspace, so nothing can be
    corrupted. The actionable half is which one a reverse lookup resolves to.
    """
    shared = _workspace(tmp_path / "shared")
    other = _workspace(tmp_path / "other")
    _register(first=shared, second=shared, other=other)

    payload = overview.build_overview()
    projects = _by_name(payload)

    assert projects["first"]["shares_path_with"] == ["second"]
    assert projects["second"]["shares_path_with"] == ["first"]
    assert projects["other"]["shares_path_with"] == []   # a list, never None
    assert projects["first"]["state"] == projects["second"]["state"] == "ok"
    assert registry.reverse_lookup(shared) == "first"

    rendered = _render_to_text(payload)
    assert "also registered as second" in rendered
    assert "every echo names first" in rendered


def test_an_empty_registry_and_an_absent_one_both_render_the_healthy_overview(
        tmp_path, capsys):
    """I8: empty is first-class, and an absent file is the same healthy state."""
    for build in (lambda: _write_registry(""), lambda: None):
        build()
        assert cli.cmd_status_overview() == 0
        text = capsys.readouterr().out
        assert "No projects registered yet" in text
        assert registry.registry_path() in text

        assert cli.cmd_status_overview(as_json=True) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload == {
            "report": "overview",
            "registry_path": registry.registry_path(),
            "count": 0,
            "cwd_project": None,
            "projects": [],
            "instances": [],
        }
        if os.path.exists(registry.registry_path()):
            os.remove(registry.registry_path())


def test_a_flagged_entry_and_a_dead_instance_both_still_exit_zero(
        tmp_path, qdrant, capsys):
    """The exit contract's other half: **0 whenever it can render.**

    A stale entry is flagged-not-fatal and an unreachable Qdrant is a timed-out cell;
    neither is an error. This matters because the shipped ``0``/``1`` readiness
    mapping is sold in SETUP.md as *the* machine-readable signal, and ``ready`` is
    undefined for a report about N projects — so a scripted caller that keeps reading
    the exit code must not meet a report that reds on a finding.
    """
    healthy = _workspace(tmp_path / "healthy")
    _register(healthy=healthy, stale=str(tmp_path / "nowhere"))
    qdrant.faults["http://localhost:7333"] = requests.exceptions.ConnectionError("no")

    assert cli.cmd_status_overview() == 0
    text = capsys.readouterr().out
    assert "missing" in text and "no answer" in text

    assert cli.cmd_status_overview(as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["report"] == "overview"
    assert _by_name(payload)["stale"]["state"] == "missing"

    # Criterion 6's sharp half: an instance that did not answer leaves the collection
    # **unknown**, and unknown must never wear the flag that means absent. A pointer
    # at `mitos reconcile` here would send an operator to re-embed a project whose
    # vectors are, for all anyone knows, perfectly intact.
    notes = "\n".join(cli._overview_notes(_by_name(payload)["healthy"], payload))
    assert "unknown" in notes
    assert "reconcile" not in notes


def test_a_registry_name_carrying_a_control_character_renders_raw(tmp_path, qdrant):
    """Pins the CURRENT disposition so a later audit inverts a row, not a comment.

    ``registry.load()`` validates only *is a string* and *is absolute*, and only on
    the **value** — so a hand edit can leave a name holding a newline, which breaks
    the table's line structure. The disposition here is deliberate and is **not**
    that this is fine: the shipped ``mitos projects`` table renders the same data
    raw, and a unilateral divergence would give the machine two spellings of one
    listing. Fixing it means fixing **both** surfaces in one edit — recorded for the
    later ADR-edge audit. If that audit lands ``{value!r}``, this row is the one that
    must be inverted.
    """
    _write_registry(f'"line\\nbreak" = "{_workspace(tmp_path / "ws")}"\n')

    payload = overview.build_overview()

    assert payload["projects"][0]["name"] == "line\nbreak"
    assert "line\nbreak" in _render_to_text(payload)   # raw, exactly like `projects`


def test_a_malformed_registry_is_the_only_non_zero_path(tmp_path):
    """The exit contract's single failure: nothing can be rendered at all.

    ``registry.load()`` is deliberately outside the bounded phase for this reason —
    it is not an entry whose failure the sweep should survive.
    """
    _write_registry("this is not [ valid toml")

    with pytest.raises(RegistryError):
        overview.build_overview()
    with pytest.raises(RegistryError):
        cli.cmd_status_overview()


# --- the cwd marker ---------------------------------------------------------

def test_the_overview_marks_the_project_containing_cwd_and_teaches_the_form(
        tmp_path, monkeypatch, qdrant, capsys):
    """Success criterion 11 — the overview closes its own discovery loop."""
    project = _workspace(tmp_path / "project")
    nested = os.path.join(project, "src", "deep")
    os.makedirs(nested)
    _register(project=project, other=_workspace(tmp_path / "other"))

    payload = overview.build_overview(cwd=nested)
    assert payload["cwd_project"] == "project"

    cli._render_overview(payload)
    text = capsys.readouterr().out
    assert "`mitos status project` for this project's full report" in text
    # …and exactly one row is marked.
    assert text.count("you are here") == 1


def test_run_from_outside_every_registration_nothing_is_marked(tmp_path, qdrant):
    """No marker is a state, not a failure."""
    _register(project=_workspace(tmp_path / "project"))
    outside = tmp_path / "elsewhere"
    outside.mkdir()

    payload = overview.build_overview(cwd=str(outside))

    assert payload["cwd_project"] is None
    assert "you are here" not in _render_to_text(payload)


def test_a_cwd_that_cannot_be_read_still_renders_the_whole_table(
        tmp_path, monkeypatch, qdrant, capsys):
    """A deleted working directory raises ``OSError`` — the boundary degrades.

    Losing an optional marker is the right degradation; losing the rendered answer
    is not. The guard is narrower than the precedent it copies on purpose: a
    ``RegistryError`` here must NOT be swallowed into an empty-looking table.
    """
    _register(project=_workspace(tmp_path / "project"))
    monkeypatch.setattr(os, "getcwd", MagicMock(side_effect=OSError("no such dir")))

    assert cli.cmd_status_overview(as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["cwd_project"] is None
    assert [p["name"] for p in payload["projects"]] == ["project"]


def test_the_leaf_never_reads_the_working_directory_itself():
    """A source sweep over the parsed code, because the invariant is structural.

    Swept as AST rather than text: the module's prose names the call (it is the thing
    this design removes), so a string sweep could only ever fail for the wrong reason.
    """
    tree = ast.parse(open(overview.__file__, encoding="utf-8").read())
    reads = [
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in ("getcwd", "chdir")
    ]
    assert reads == [], f"the overview leaf touches the working directory: {reads}"


# --- payload shape ----------------------------------------------------------

def test_the_payload_is_json_native_and_carries_no_set_or_tuple(tmp_path, qdrant):
    """No collection listing crosses the boundary — it is another project's data.

    The round trip is the check that catches a tuple (which survives in-process and
    becomes a list across JSON, so two surfaces would disagree by how each was
    tested) and a set (which does not survive at all).
    """
    project = _workspace(tmp_path / "project")
    _register(project=project, gone=str(tmp_path / "nowhere"))
    qdrant.names["http://localhost:7333"] = [default_collection_name(project), "other"]

    payload = overview.build_overview(cwd=str(tmp_path))

    assert json.loads(json.dumps(payload)) == payload
    assert set(payload["projects"][0]) == {
        "name", "path", "state", "error", "shares_path_with", "qdrant_url",
        "collection", "collection_present"}
    assert set(payload["instances"][0]) == {
        "url", "reachable", "listing_available", "collection_count", "error"}
    assert payload["instances"][0]["collection_count"] == 2
    # `registry_path` and `count` are spelled exactly as `mitos projects` spells
    # them, so a reader moves between the two listings without translation.
    assert set(payload) >= {"registry_path", "count"}


def test_the_status_project_payload_announces_which_report_it_is(tmp_path, capsys):
    """D7, on both of ``cmd_status``' emission sites.

    One flag will return two payloads, so each announces itself — a discriminator
    present on only one side is detectable by absence, which is sniffing.
    """
    workspace = _workspace(tmp_path / "project")
    with patch.object(cli, "_check_qdrant", return_value={
            "reachable": False, "collection_exists": None, "points": None}):
        cli.cmd_status(workspace, as_json=True)
    assert json.loads(capsys.readouterr().out)["report"] == "project"

    with open(os.path.join(workspace, ".mitos", "config.toml"), "w") as f:
        f.write("not = = toml")
    cli.cmd_status(workspace, as_json=True)
    assert json.loads(capsys.readouterr().out)["report"] == "project"


# --- the dispatch flip (5a) -------------------------------------------------

def _run(argv: List[str]) -> Any:
    """Drives `cli.main()` through argv, returning the exit code."""
    with patch.object(sys, "argv", ["mitos"] + list(argv)):
        try:
            cli.main()
        except SystemExit as exc:
            return exc.code
    return 0


def test_the_zero_arg_status_routes_to_the_overview_and_the_named_form_does_not(
        tmp_path, qdrant, capsys):
    """W29's named consumer: the report 4a built is what a selectorless `status` is.

    Both halves in one row on purpose — the flip is a *fork*, and a row that only
    proved the new branch would stay green under a build that routed **every**
    `status` to the overview, which would silently delete the deep report.
    """
    project = _workspace(tmp_path / "project")
    _register(project=project)
    capsys.readouterr()

    assert _run(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["report"] == "overview"

    assert _run(["status", "project", "--json"]) in (0, 1)
    assert json.loads(capsys.readouterr().out)["report"] == "project"


def test_the_one_non_zero_path_reaches_the_user_as_a_calm_line(tmp_path, capsys):
    """The exit contract's failure half, asserted where a user now meets it.

    `test_a_malformed_registry_is_the_only_non_zero_path` pins the raise at the
    direct call — the only reach 4a had. The flip gave that raise a route through
    `main()`, and `RegistryError` subclasses `MitosError`, so it lands on the
    boundary's generic arm *below* the `ProjectTargetingError` one: a one-line
    `Error:` on stderr and exit 1. Nothing asserted that until the route existed,
    and an uncaught class here would surface as a raw traceback on the report a
    caller reaches for when their machine is already confusing them.

    `--json` too, deliberately: the shipped boundary answers on stderr for every
    fault regardless of the flag, so this emits **no** JSON — the same asymmetry
    3b signed for the targeting errors, restated on the one report that has a
    machine-readable twin.
    """
    _write_registry("this is not [ valid toml")

    for argv in (["status"], ["status", "--json"]):
        assert _run(argv) == 1, argv
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("Error: ")
        # Named by the fault it actually is, not merely "something failed": only
        # `RegistryError` speaks this, so a build that answered exit 1 for some
        # other reason (a targeting error, say) cannot satisfy the row.
        assert "not valid TOML" in captured.err
        assert "Traceback" not in captured.err


def test_the_overview_is_built_before_any_workspace_config_is(tmp_path, qdrant,
                                                              monkeypatch, capsys):
    """Ordering is contract: a broken workspace underfoot is not a global failure.

    The caller stands in a directory whose `.mitos/config.toml` is malformed. Build
    the boundary config first and 4b's `ConfigError` carve-out catches it and routes
    to `_answer_workspace_optional_verb` → the **deep** report about that directory:
    exit 1, the wrong answer, and nothing red anywhere. Fault injection: move the
    branch below the config construction and this row reds while every other row in
    the module stays green.
    """
    broken = tmp_path / "broken"
    os.makedirs(str(broken / ".mitos"))
    with open(str(broken / ".mitos" / "config.toml"), "w") as f:
        f.write("not = = toml")
    with open(str(broken / "decisions.md"), "w") as f:
        f.write("")
    _register(elsewhere=_workspace(tmp_path / "elsewhere"))
    monkeypatch.chdir(str(broken))
    capsys.readouterr()

    assert _run(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["report"] == "overview"


# --- the collection flag's corpus gate (entry-010) --------------------------

def _populate(workspace: str) -> None:
    """Writes a corpus holding one entry BELOW the sentinel.

    By hand rather than through `record`/`sync`: the flag under test is about the
    *corpus*, and both of those would also build a graph this module never opens.
    """
    with open(os.path.join(workspace, "decisions.md"), "w", encoding="utf-8") as f:
        f.write("# Decisions\n\n<!-- BEGIN ENTRIES -->\n\n### an-entry\n\n"
                "**Decided:** Something was decided here.\n"
                "**Rejected:** Deciding nothing.\n")


def test_a_fresh_project_with_no_collection_yet_is_not_flagged(tmp_path, qdrant):
    """I8 at the overview: healthy-and-empty is first-class, so it wears no ⚠.

    Post-1c `may_create` binds creation to the upsert, so **every** project between
    `mitos init` and its first `record` has no collection — the single most common
    state on a machine with a fresh registration. The deep report calls that state
    READY ✓ and its collection row reads *"auto-created on first record"*; an
    unconditional flag here made the two halves of one command contradict each other
    on the healthiest state there is.
    """
    project = _workspace(tmp_path / "project")     # `_workspace` leaves the corpus empty
    _register(project=project)

    payload = overview.build_overview()

    assert _by_name(payload)["project"]["collection_present"] is False
    assert cli._overview_notes(_by_name(payload)["project"], payload) == []


def test_a_populated_project_with_no_collection_is_flagged_and_names_no_heal(
        tmp_path, qdrant):
    """The other side of the gate — and the flag prescribes a *report*, not a command.

    The overview reads no graph, so it cannot tell the clone whose graph was never
    built (heal: `mitos sync`) from the project whose collection was swept (heal:
    `mitos reconcile`) — and for the clone `reconcile` is the heal 4b calls *"one
    word away and worse than silence"*: it diffs an empty active set against an
    absent collection, enqueues nothing, and reports success on a workspace it did
    not touch. So the note hands the reader to `mitos status <name>`, which reads the
    graph and names the right one.
    """
    project = _workspace(tmp_path / "project")
    _populate(project)
    _register(project=project)

    payload = overview.build_overview()
    notes = "\n".join(cli._overview_notes(_by_name(payload)["project"], payload))

    assert "no vector collection" in notes
    assert default_collection_name(project) in notes
    assert "mitos status project" in notes
    assert "reconcile" not in notes
    assert "sync" not in notes


def test_the_corpus_gate_is_injected_and_not_re_derived(tmp_path, qdrant):
    """The scan is a parameter, so a row can drive both verdicts without a filesystem.

    It also pins *which* file is scanned: `decisions.md` beside `.mitos/` — the
    shipped validity triple `is_workspace` just proved — rather than anything a
    second config construction might derive.
    """
    project = _workspace(tmp_path / "project")
    _register(project=project)
    payload = overview.build_overview()
    entry = _by_name(payload)["project"]

    scanned: List[str] = []

    def _scan(path: str) -> bool:
        scanned.append(path)
        return True

    notes = cli._overview_notes(entry, payload, corpus_scan=_scan)

    assert scanned == [os.path.join(project, "decisions.md")]
    assert any("no vector collection" in note for note in notes)
    assert cli._overview_notes(entry, payload, corpus_scan=lambda _: False) == []


# --- tier ------------------------------------------------------------------

def test_importing_the_overview_pulls_in_no_higher_tier_module():
    """The tier rule is a subprocess probe, not prose.

    The sharper half is ``anthropic``: ``import mitos.cli`` drags the SDK at module
    scope, so homing the sweep there would make every test of it — and any second
    consumer — pay an SDK import to enumerate a TOML file.
    """
    probe = (
        "import sys; import mitos.overview; "
        "leaked = sorted(m for m in sys.modules "
        "if m.startswith('mitos.') and m.split('.')[1] in "
        "{'cli', 'store', 'sync', 'renderer', 'importer', 'parser', 'cutover', "
        "'recall', 'display'}); "
        "print(','.join(leaked)); "
        "print('anthropic' in sys.modules); "
        "print(','.join(sorted(m for m in sys.modules if m.startswith('mitos'))))"
    )
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True)
    leaked, anthropic, loaded = out.stdout.splitlines()
    assert leaked == "", f"leaked imports: {leaked}"
    assert anthropic == "False"

    # The exact closure, because the rule is "nothing else, **ever**" and the
    # blacklist above names nine modules. `vector_store` is deliberately ABSENT from
    # that blacklist — the two existing instances of this idiom list it, and copying
    # either verbatim would give a row that fails for the right reason about the
    # wrong module: this leaf imports it by design (the instance listing). A new
    # member below is a tier decision to adjudicate, not a set to widen reflexively.
    assert loaded.split(",") == [
        "mitos", "mitos.config", "mitos.env", "mitos.errors", "mitos.models",
        "mitos.overview", "mitos.registry", "mitos.routing", "mitos.vector_store",
    ]


# --- the two subprocess rows ------------------------------------------------

def _child_env(tmp_path) -> Dict[str, str]:
    """The child's environment: hermetic, and with the update check silenced.

    ``main()``'s ``finally`` runs ``update_notice`` on every non-``serve`` command —
    fail-silent, but it can make a **network call**, which would land inside a
    measured wall clock and make the bounding row flaky for a reason that has nothing
    to do with the bound. The autouse fixture sets these in-process only, so a
    subprocess inherits them exactly by building its env from ``os.environ`` — which
    is also what redirects ``XDG_CONFIG_HOME`` away from the developer's real
    registry.
    """
    env = dict(os.environ)
    assert env["MITOS_NO_UPDATE_CHECK"] == "1"
    assert registry.registry_path().startswith(env["XDG_CONFIG_HOME"])
    return env


def test_the_process_exits_promptly_when_a_probe_parks(tmp_path):
    """The bound must bound the PROCESS, not merely the sweep.

    Measured 2026-07-31 on this box: a 6 s park bounded by a ``ThreadPoolExecutor``
    with ``future.result(timeout=0.5)`` and ``shutdown(wait=False,
    cancel_futures=True)`` costs **6.03 s of total process wall**, because
    ``concurrent.futures.thread`` registers an atexit hook that joins every worker.
    The same park on a daemon thread costs **0.53 s**. Both look correct from inside
    the process — which is exactly why this row measures a child from outside and
    ``capsys`` cannot replace it.
    """
    parked = _workspace(tmp_path / "parked")
    _register(parked=parked, gone=str(tmp_path / "nowhere"))

    script = textwrap.dedent("""
        import json, time
        from mitos import overview
        real = overview.probe_entry
        def slow(name, path):
            if name == "parked":
                time.sleep(30)
            return real(name, path)
        overview.probe_entry = slow
        overview.LOCAL_PROBE_BUDGET = 0.5
        payload = overview.build_overview()
        print(json.dumps({p["name"]: p["state"] for p in payload["projects"]}))
    """)

    started = time.monotonic()
    out = subprocess.run([sys.executable, "-c", script], env=_child_env(tmp_path),
                         capture_output=True, text=True, timeout=60)
    elapsed = time.monotonic() - started

    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == {"parked": "unresponsive", "gone": "missing"}
    # Generous headroom for a loaded box; the park is 30 s and the ThreadPoolExecutor
    # build would cost all of it.
    assert elapsed < 10, (
        f"the child took {elapsed:.2f}s to exit behind a 0.5s bound — the sweep was "
        f"bounded but the process was not")


@pytest.mark.parametrize("registered", [True, False])
def test_the_table_reaches_a_combined_pipe_above_a_later_stderr_write(
        tmp_path, registered):
    """Gotcha 7: the rendered table must not arrive *under* the error it precedes.

    ``main()``'s ``except MitosError`` boundary writes unbuffered stderr while a
    piped stdout is block-buffered, so without the flush the refusal lands above the
    output it annotates — with the whole suite green, because ``capsys`` keeps the
    streams apart and cannot see it. Both branches of the renderer print, so both owe
    the flush.
    """
    if registered:
        _register(project=_workspace(tmp_path / "project"))
    else:
        _write_registry("")

    script = textwrap.dedent("""
        import sys
        from mitos import cli, overview
        cli._render_overview(overview.build_overview())
        print("Error: simulated boundary fault", file=sys.stderr)
    """)
    out = subprocess.run([sys.executable, "-c", script], env=_child_env(tmp_path),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         timeout=120)

    assert out.returncode == 0, out.stdout
    lines = out.stdout.splitlines()
    error_at = next(i for i, line in enumerate(lines) if line.startswith("Error:"))
    body_at = next(i for i, line in enumerate(lines) if line.strip())
    assert body_at < error_at, (
        f"the table landed below the error it precedes:\n{out.stdout}")


# --- shared render helper ---------------------------------------------------

def _render_to_text(payload: Dict[str, Any]) -> str:
    """Renders a payload and returns the text, for rows asserting on wording."""
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        cli._render_overview(payload)
    return buffer.getvalue()
