"""Live overview rows: the sweep against a real Qdrant instance.

The offline module proves everything provable without a service. What only a real
instance can answer is here, and it is the half the vision leans on hardest:

* two projects on **distinct** Qdrant URLs are each probed and each attributed —
  ``http://localhost:7333`` and ``http://127.0.0.1:7333`` are distinct strings and
  both genuinely reachable, so the row proves *attribution* rather than proving one
  failure;
* one live URL beside a dead one: the dead instance renders as no answer, its
  project's ``collection_present`` is ``null`` (never ``false``), and **the sweep
  still completes** with the live project fully reported;
* one shared URL across two projects issues exactly **one** request, and no request
  is ever addressed to ``/collections/<name>``;
* a project whose derived collection is genuinely absent is flagged — with its twin,
  a project whose collection the fixture created, which is not. Both halves of that
  pair carry a **populated corpus**, because 5a gated the flag on one: a project
  between ``mitos init`` and its first ``record`` legitimately has no collection and
  must wear no ⚠;
* a raising local probe on the same run as a healthy project renders its own state
  while everything else is fully reported.

Discipline: every fixture workspace is built at ``tmp_path / "tmp-…"`` so the derived
collection name starts with ``mitos-tmp`` and ``conftest``'s
``sweep_leaked_qdrant_collections`` can reclaim it — the derivation is
``mitos-<basename>-<hash>``, so the **basename** is what decides, and a workspace at a
bare ``tmp_path`` would leak a collection onto the shared instance on every run. The
fixture asserts its own sweepability so a future rename cannot leak silently. No
Gemini/Anthropic spend: the one collection this module creates is created empty,
through Qdrant's own REST API.

``-n auto`` is forbidden here, and no second pytest session may run while this module
does: ``sweep_leaked_qdrant_collections`` is session-scoped and deletes by prefix with
no attribution, so a concurrent session reclaims a collection this one is still
asserting against — and the failure mode is a **skip**, not a red.
"""

import os
from typing import Any, Dict, List

import pytest
import requests

from mitos import overview, registry
from mitos.config import default_collection_name
from mitos.models import EMBEDDING_DIM

from live_helpers import live_tests_disabled

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:7333")

#: The same instance under a second, genuinely distinct URL string. Both resolve to
#: the same Qdrant, which is what makes the distinct-URL row about attribution rather
#: than about one of them failing.
ALT_QDRANT_URL = QDRANT_URL.replace("localhost", "127.0.0.1")

#: A port nothing listens on — refused in about a millisecond, so the dead-instance
#: row is fast and deterministic rather than a timeout.
DEAD_QDRANT_URL = "http://localhost:7999"


def _qdrant_reachable() -> bool:
    """Best-effort probe: this tier needs a service, not a key."""
    try:
        return requests.get(f"{QDRANT_URL}/collections", timeout=2).ok
    except Exception:
        return False


#: The two skip causes stay apart, and only one carries the live-floor's
#: ``not a code defect`` marker. The brake being on means the tier was switched off
#: wholesale — nothing pretended to check. An unreachable Qdrant while the tier is
#: live is the environmental degradation the floor exists to count.
_BRAKE_ON = live_tests_disabled()
HAS_QDRANT = (not _BRAKE_ON) and _qdrant_reachable()

if _BRAKE_ON:
    _SKIP_REASON = (
        "MITOS_NO_LIVE_TESTS is set — the live tier is switched off, so these rows "
        "did not run and nothing pretended to check."
    )
else:
    _SKIP_REASON = (
        f"Qdrant is not reachable at {QDRANT_URL} — the overview's instance rows "
        "need a real instance. Environmental, not a code defect; start it with "
        "`docker compose up -d` in the mitos repo."
    )

pytestmark = pytest.mark.skipif(not HAS_QDRANT, reason=_SKIP_REASON)


# --- helpers ----------------------------------------------------------------

def _write_registry(text: str) -> str:
    path = registry.registry_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _register(**projects: str) -> str:
    return _write_registry(
        "".join(f'"{name}" = "{path}"\n' for name, path in projects.items()))


_POPULATED_CORPUS = ("# Decisions\n\n<!-- BEGIN ENTRIES -->\n\n### an-entry\n\n"
                     "**Decided:** Something was decided here.\n"
                     "**Rejected:** Deciding nothing.\n")


def _workspace(root, *, qdrant_url: str = None, populated: bool = False) -> str:
    """Builds a sweepable workspace and returns its canonical path.

    The caller passes a ``tmp-``-prefixed directory; this asserts the derived
    collection name is reclaimable by the conftest sweep, so a future rename of the
    fixture cannot start leaking collections silently.

    ``populated`` writes one entry below the sentinel — by hand, since the corpus is
    all the collection flag's gate reads and neither ``record`` nor ``sync`` belongs
    in a fixture this module never opens a graph for.
    """
    root = str(root)
    os.makedirs(os.path.join(root, ".mitos"), exist_ok=True)
    with open(os.path.join(root, ".mitos", "config.toml"), "w") as f:
        f.write("")
    with open(os.path.join(root, "decisions.md"), "w") as f:
        f.write(_POPULATED_CORPUS if populated else "")
    if qdrant_url is not None:
        with open(os.path.join(root, ".env"), "w") as f:
            f.write(f"QDRANT_URL={qdrant_url}\n")
    canonical = os.path.realpath(root)
    derived = default_collection_name(canonical)
    assert derived.startswith("mitos-tmp"), (
        f"{derived} is not reclaimable by conftest's sweep — build the fixture "
        f"workspace under a `tmp-` basename or this leaks a collection per run")
    return canonical


def _listing() -> List[str]:
    resp = requests.get(f"{QDRANT_URL}/collections", timeout=5)
    resp.raise_for_status()
    return [c["name"] for c in resp.json()["result"]["collections"]]


@pytest.fixture
def created_collection():
    """Creates an empty collection for a workspace, and removes it afterwards.

    Empty on purpose: the overview asks only whether the collection *exists* — it
    reads no graph and counts no points, because presence never implies population
    and the durable completeness check is the deep report's id-diff. So no vectors,
    and no embedding spend.
    """
    created: List[str] = []

    def create(name: str) -> str:
        resp = requests.put(
            f"{QDRANT_URL}/collections/{name}",
            json={"vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"}},
            headers={"Content-Type": "application/json"}, timeout=10)
        assert resp.status_code == 200, resp.text
        created.append(name)
        return name

    yield create
    for name in created:
        requests.delete(f"{QDRANT_URL}/collections/{name}", timeout=5)


@pytest.fixture
def spy(monkeypatch):
    """Records every instance request while still talking to the real Qdrant.

    A spy, not a fake: the call *count* is the claim (the budget scales with distinct
    instances, never with project count) and the answers must stay real, so the row
    proves the shape rather than proving the fake.

    **It records this module's own requests too**, and that is not a wrinkle to route
    around silently: ``mitos.vector_store.requests`` *is* the ``requests`` module, so
    setting an attribute on it is a global patch, and a helper here that reads the
    listing goes through the spy as well. Every row asserting on the count therefore
    snapshots it immediately after the sweep and does its own reads outside that
    window.
    """
    calls: List[str] = []
    real = requests.get

    def recording(url: str, *args: Any, **kwargs: Any) -> Any:
        calls.append(url)
        return real(url, *args, **kwargs)

    monkeypatch.setattr("mitos.vector_store.requests.get", recording)
    return calls


@pytest.fixture(autouse=True)
def no_exported_qdrant_url(monkeypatch):
    """Forces ``QDRANT_URL``'s absence so the per-workspace ``.env`` tiers decide.

    An exported ``QDRANT_URL`` beats every file tier, so a distinct-URL fixture
    silently collapses to one URL on a box that exports it. A bare
    ``delenv(raising=False)`` on an already-absent name **records nothing** and
    protects nothing, which is why this is the two-line forced-absence form.
    """
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.delenv("QDRANT_URL")


def _by_name(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {project["name"]: project for project in payload["projects"]}


# --- rows -------------------------------------------------------------------

def test_two_projects_on_distinct_urls_are_each_probed_and_attributed(tmp_path, spy):
    """T15: the overview must not assume a single instance.

    Both URLs are live, so a failure here is about attribution rather than about
    reachability — the two are told apart, each project joins its own instance's
    listing, and both rows report ``up``.
    """
    here = _workspace(tmp_path / "tmp-here", qdrant_url=QDRANT_URL)
    there = _workspace(tmp_path / "tmp-there", qdrant_url=ALT_QDRANT_URL)
    _register(here=here, there=there)
    live_count = len(_listing())
    spy.clear()

    payload = overview.build_overview()
    calls = list(spy)          # snapshot: the spy is global (see the fixture)
    projects = _by_name(payload)

    assert projects["here"]["qdrant_url"] == QDRANT_URL
    assert projects["there"]["qdrant_url"] == ALT_QDRANT_URL
    assert [i["url"] for i in payload["instances"]] == [QDRANT_URL, ALT_QDRANT_URL]
    for instance in payload["instances"]:
        assert instance["reachable"] is True
        assert instance["listing_available"] is True
        # Both URLs address the same instance, so both must report the same listing —
        # which is the point: the row proves attribution, not one of them failing.
        assert instance["collection_count"] == live_count
    assert sorted(calls) == sorted(
        [f"{QDRANT_URL}/collections", f"{ALT_QDRANT_URL}/collections"])


def test_a_dead_instance_never_takes_the_sweep_down_with_it(tmp_path):
    """T15: one live URL, one dead one — the live project is still fully reported.

    And the dead instance's project is ``collection_present: null``, never ``false``:
    an instance that did not answer has said nothing about anyone's vectors, and
    rendering silence as absence would send an operator to heal a healthy project.
    """
    live = _workspace(tmp_path / "tmp-live", qdrant_url=QDRANT_URL)
    stranded = _workspace(tmp_path / "tmp-stranded", qdrant_url=DEAD_QDRANT_URL)
    _register(live=live, stranded=stranded)

    payload = overview.build_overview()
    projects = _by_name(payload)
    instances = {i["url"]: i for i in payload["instances"]}

    assert projects["live"]["state"] == "ok"
    assert projects["live"]["collection"] == default_collection_name(live)
    assert projects["live"]["collection_present"] is False  # genuinely absent
    assert projects["stranded"]["state"] == "ok"            # the workspace is fine
    assert projects["stranded"]["collection_present"] is None

    assert instances[QDRANT_URL]["reachable"] is True
    assert instances[DEAD_QDRANT_URL]["reachable"] is False
    assert instances[DEAD_QDRANT_URL]["listing_available"] is False
    assert instances[DEAD_QDRANT_URL]["collection_count"] is None


def test_two_projects_sharing_one_url_cost_exactly_one_request(tmp_path, spy):
    """T15: the call budget scales with distinct instances, never with projects.

    The second assertion is the shape §4.9 forbids: not one request in this whole
    sweep addresses a single collection. ``cli._check_qdrant`` is that shape and
    stays the deep report's.
    """
    first = _workspace(tmp_path / "tmp-first")
    second = _workspace(tmp_path / "tmp-second")
    _register(first=first, second=second)
    spy.clear()

    overview.build_overview()
    calls = list(spy)

    assert calls == [f"{QDRANT_URL}/collections"]
    assert not any("/collections/" in url for url in calls)


def test_an_absent_collection_is_flagged_and_its_present_twin_is_not(
        tmp_path, created_collection):
    """T15 + W30: the post-migration net, with the fixture pair that makes it bite.

    The two workspaces differ **only** in whether their collection exists — a fixture
    that varied along a second axis would pass under either behaviour. Both are
    **populated**, which is the axis 5a added: the flag is gated on the corpus, so an
    empty-corpus pair would leave both unflagged and the row would prove nothing.

    The pointer inverted with that gate. It used to name ``mitos reconcile``, which
    is wrong for a clone whose graph was never built; it now hands the reader to
    ``mitos status <name>``, the surface that reads the graph and can tell the two
    heals apart.
    """
    from mitos import cli

    swept = _workspace(tmp_path / "tmp-swept", populated=True)
    reconciled = _workspace(tmp_path / "tmp-reconciled", populated=True)
    created_collection(default_collection_name(reconciled))
    _register(swept=swept, reconciled=reconciled)

    payload = overview.build_overview()
    projects = _by_name(payload)

    assert projects["swept"]["collection_present"] is False
    assert projects["reconciled"]["collection_present"] is True

    notes = "\n".join(cli._overview_notes(projects["swept"], payload))
    assert "mitos status swept" in notes
    assert "reconcile" not in notes
    assert default_collection_name(swept) in notes
    assert cli._overview_notes(projects["reconciled"], payload) == []


def test_a_fresh_projects_absent_collection_is_not_a_finding(tmp_path):
    """Entry-010's headline state, proven against the real instance.

    Same shape as the swept project above and the same genuinely-absent collection —
    the corpus is the only difference. A machine full of freshly-registered projects
    must render a calm table, not a column of ⚠.
    """
    from mitos import cli

    fresh = _workspace(tmp_path / "tmp-fresh")           # empty corpus
    _register(fresh=fresh)

    payload = overview.build_overview()
    projects = _by_name(payload)

    assert projects["fresh"]["collection_present"] is False
    assert cli._overview_notes(projects["fresh"], payload) == []


def test_a_raising_entry_renders_its_own_state_while_the_sweep_completes(
        tmp_path, monkeypatch, created_collection):
    """T15: per-entry isolation, proven on a run that also does real network work.

    The offline twin proves the same property against a faked instance; this one
    proves it does not evaporate once the instance phase is real.
    """
    healthy = _workspace(tmp_path / "tmp-healthy")
    broken = _workspace(tmp_path / "tmp-broken")
    created_collection(default_collection_name(healthy))
    _register(healthy=healthy, broken=broken)

    real = overview.probe_entry

    def exploding(name: str, path: str) -> Dict[str, Any]:
        if name == "broken":
            raise OSError("[Errno 5] Input/output error")
        return real(name, path)

    monkeypatch.setattr(overview, "probe_entry", exploding)
    payload = overview.build_overview()
    projects = _by_name(payload)

    assert projects["broken"]["state"] == "error"
    assert "Input/output error" in projects["broken"]["error"]
    assert projects["broken"]["qdrant_url"] is None
    # The healthy project is FULLY reported — including the instance join, which
    # happens after the phase the broken entry failed in.
    assert projects["healthy"]["state"] == "ok"
    assert projects["healthy"]["collection_present"] is True
    assert payload["instances"][0]["listing_available"] is True
