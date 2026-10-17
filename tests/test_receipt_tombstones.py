"""Phase 7b: the receipts around the record pause name the graveyard too.

Two receipts said nothing about retired decisions after 7a put them on the pause
(vision §4.9). The ``created`` receipt under ``acknowledge_neighbors`` now carries
``acknowledged_tombstones``: the bare retired handles the write went past, uncut, under
lock 3, exempted by the target ids Phase A validated so the commit cannot move an
exemption. An exact re-tread of a retired decision answers ``exists`` before any pause;
its receipt now carries that node's handle keys and a note naming where the current
word lives, never a relation the commit would refuse. (Plan p7b §3, §4 D1–D5, §9.)

The D3 rows that need a same-slug lineage call the acknowledged field directly over a
graph built through ``GraphStore.commit_parsed_entry``: ``record_decision`` itself
refuses a slug any node carries (``slug_collision``), so its own path cannot build the
shape the post-commit exemption defends against (scout warning 1).
"""

import inspect
import io
import json
import re
import shutil
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import pytest

from mitos import sync
from mitos.cli import cmd_init, cmd_record
from mitos.config import MitosConfig
from mitos.conflict import CONFLICT_TOP_K
from mitos.display import retired_handle
from mitos.errors import DatabaseError
from mitos.parser import ParsedEntry
from mitos.store import live_ends_from_chains
from mitos.sync import (AcknowledgedReview, MitosSyncManager, _EXISTS_NO_OP_NOTE,
                        _PAUSE_RESOLVING_RELATIONS)

from _point_hits import point_hit
from test_neighbor_review import (_arm, _armed_real_manager_factory, _inject_raw_edge,  # noqa: F401
                                  _seed, offline, ws)
from test_pause_tombstones import (_BETWEEN, _DRAFT, _HIGH, _Counting, _commit, _draft,
                                   _gathering, _id_of, _retire, _shell)
from test_point_identity import same_slug_lineage
from test_record_decision import _shell_tokens

_FLOOR = sync._TOMBSTONE_FLOOR
# Every relation an agent can declare: none may be named by a note that must name none.
_RELATIONS = ("supersedes", "corrects", "amends", "narrows", "depends_on", "resolves",
              "contradicts", "derives_from", "cites")
_LIVE_EXISTS_KEYS = {"slug", "id", "state", "embedding", "status", "path", "no_op_reason"}
_ECHO = {"project", "collection", "workspace"}


def _names_a_relation(note: str) -> list:
    return [r for r in _RELATIONS if re.search(rf"\b{r}\b", note)]


def _entry(slug: str, axiom: str) -> ParsedEntry:
    entry = ParsedEntry("decision", slug, 1, 5)
    entry.axiom = axiom
    entry.rejected_paths = "An alternative."
    return entry


# =========================================================================== #
# The acknowledged receipt: `acknowledged_tombstones`
# =========================================================================== #

# -- Row 1: a non-exempt tombstone above the floor, as a bare handle -----------------

def test_acknowledging_past_a_tombstone_lists_its_bare_handle(ws):
    config, m = ws
    old = _retire(m, "old-way", "We store sessions in signed JWTs.", by="new-way")
    _arm(m, [{"slug": "old-way", "score": _HIGH}])
    res = _draft(m, acknowledge_neighbors=True)
    assert res["status"] == "created"
    assert res["acknowledged_neighbors"] == []
    assert res["acknowledged_tombstones"] == [{
        "slug": "old-way", "id": old, "state": "superseded", "score": _HIGH,
        "superseded_by": ["new-way"]}]
    keys = list(res)
    assert keys.index("acknowledged_tombstones") == keys.index("acknowledged_neighbors") + 1


def test_an_acknowledged_tombstone_through_a_retired_successor_names_its_live_end(ws):
    config, m = ws
    old = _retire(m, "old-way", by="mid-way")
    _seed(m, "new-way", "Axiom for new-way.", supersedes="mid-way")
    _arm(m, [{"slug": "old-way", "score": _HIGH}])
    (handle,) = _draft(m, acknowledge_neighbors=True)["acknowledged_tombstones"]
    assert handle == {"slug": "old-way", "id": old, "state": "superseded",
                      "score": _HIGH, "superseded_by": ["mid-way"], "live_end": ["new-way"]}


# -- Row 2: a healthy gather with nothing to name ------------------------------------

def test_nothing_at_or_above_the_tombstone_floor_is_an_empty_list(ws):
    config, m = ws
    _retire(m, "old-way")
    _arm(m, [{"slug": "old-way", "score": _BETWEEN}])
    res = _draft(m, acknowledge_neighbors=True)
    assert res["acknowledged_tombstones"] == []


def test_only_exempt_tombstones_above_the_floor_is_an_empty_list(ws):
    config, m = ws
    _retire(m, "old-way", by="new-way")
    _arm(m, [{"slug": "old-way", "score": _HIGH}])
    res = _draft(m, cites="new-way", acknowledge_neighbors=True)
    assert res["acknowledged_tombstones"] == [] and res["acknowledged_neighbors"] == []


# -- Row 3: lock 3, two fault domains -----------------------------------------------

def test_a_gather_fault_nulls_both_lists(ws, capsys):
    config, m = ws
    _retire(m, "old-way")
    _arm(m, [{"slug": "old-way", "score": _HIGH}])
    with patch.object(MitosSyncManager, "_review_neighbors",
                      side_effect=RuntimeError("injected gather fault")):
        res = _draft(m, acknowledge_neighbors=True)
    assert res["status"] == "created"
    assert res["acknowledged_neighbors"] is None and res["acknowledged_tombstones"] is None
    assert "injected gather fault" in capsys.readouterr().err


def test_a_declared_review_fault_nulls_both_lists(ws):
    config, m = ws
    _retire(m, "old-way", by="new-way")
    _arm(m, [{"slug": "old-way", "score": _HIGH}])
    with patch.object(m.store, "get_lineage", side_effect=DatabaseError("injected")):
        res = _draft(m, amends="new-way", acknowledge_neighbors=True)
    assert res["status"] == "created"
    assert res["acknowledged_neighbors"] is None and res["acknowledged_tombstones"] is None


def test_a_kill_chain_fault_nulls_the_tombstones_alone(ws, capsys):
    config, m = ws
    _retire(m, "old-way")
    _seed(m, "live-twin", "Axiom for live twin.")
    _arm(m, [{"slug": "old-way", "score": _HIGH}, {"slug": "live-twin", "score": 0.88}])
    capsys.readouterr()
    with patch.object(m.store, "get_kill_chains",
                      side_effect=DatabaseError("injected kill-chain fault")):
        res = _draft(m, acknowledge_neighbors=True)
    assert res["status"] == "created"
    assert res["acknowledged_neighbors"] == ["live-twin"]
    assert res["acknowledged_tombstones"] is None
    err = capsys.readouterr().err
    assert err.count("[Warning]") == 1 and "Acknowledged-tombstone read failed" in err


def test_a_typed_unavailable_gather_nulls_both_lists(ws):
    from test_neighbor_review import _RaisingVector
    config, m = ws
    _retire(m, "old-way")
    _arm(m, [])
    m.vector_store = _RaisingVector(ConnectionError("qdrant down"))
    review = m._acknowledged_fields(_entry("draft", _DRAFT), None, None, {})
    assert review == AcknowledgedReview(neighbors=None, tombstones=None)


def test_without_providers_both_lists_are_null(ws):
    config, m = ws
    _retire(m, "old-way")
    res = _draft(m, acknowledge_neighbors=True)
    assert res["status"] == "created"
    assert res["acknowledged_neighbors"] is None and res["acknowledged_tombstones"] is None


def test_the_collection_latch_nulls_both_lists(ws):
    config, m = ws
    _arm(m, [])
    m._collection_absent = True
    review = m._acknowledged_fields(_entry("draft", _DRAFT), None, None, {})
    assert review == AcknowledgedReview(neighbors=None, tombstones=None)


# -- Row 4: the key rides the flag alone --------------------------------------------

def test_without_the_flag_the_key_is_absent_on_every_exit(ws):
    config, m = ws
    _retire(m, "old-way")
    created = m.record_decision_entry("A quiet decision.", "rej", ["s"], slug="quiet")
    _arm(m, [{"slug": "old-way", "score": _HIGH}])
    paused = _draft(m)
    exists = m.record_decision_entry("A quiet decision.", "rej", ["s"], slug="quiet")
    error = _draft(m, supersedes="no-such-decision")
    assert (created["status"], paused["status"], exists["status"]) == \
        ("created", "needs_review", "exists")
    assert "code" in error and "status" not in error
    for res in (created, paused, exists, error):
        assert "acknowledged_tombstones" not in res and "acknowledged_neighbors" not in res


def test_with_the_flag_an_exists_or_an_error_carries_no_key(ws):
    config, m = ws
    _retire(m, "old-way")
    _arm(m, [{"slug": "old-way", "score": _HIGH}])
    exists = m.record_decision_entry("Axiom for old-way.", "rej", ["s"], slug="old-way",
                                     acknowledge_neighbors=True)
    error = _draft(m, supersedes="no-such-decision", acknowledge_neighbors=True)
    assert exists["status"] == "exists"
    for res in (exists, error):
        assert "acknowledged_tombstones" not in res


# -- Row 5: uncut, closest first --------------------------------------------------------

def test_more_tombstones_than_the_pause_shows_are_all_listed_closest_first(ws):
    config, m = ws
    count = CONFLICT_TOP_K + 3
    slugs = [f"old-{i}" for i in range(count)]
    for slug in slugs:
        _retire(m, slug)
    scores = [round(0.99 - i * 0.01, 4) for i in range(count)]
    assert scores[-1] >= _FLOOR
    # Rank order, as the vector store returns it; old-0 is the closest.
    _arm(m, [{"slug": s, "score": sc} for s, sc in zip(slugs, scores)])
    res = _draft(m, acknowledge_neighbors=True)
    listed = res["acknowledged_tombstones"]
    assert len(listed) == count > CONFLICT_TOP_K
    assert [t["slug"] for t in listed] == slugs
    assert [t["score"] for t in listed] == sorted(scores, reverse=True)


# -- Row 6: D3, exemption by the ids Phase A validated ---------------------------------

def _same_slug_supersession(store, *, how="supersedes"):
    """X = ``database``: amends a live A, amends M (M later superseded by Z), and
    supersedes P; then N, also ``database``, kills X. Returns the ids by name."""
    ids = {"A": _commit(store, "live-anchor", "A live anchor decision.")}
    ids["M"] = _commit(store, "mid-rule", "A mid rule decision.")
    ids["P"] = _commit(store, "legacy-db", "We use MySQL.")
    ids["X"] = _commit(store, "database", "We use PostgreSQL.",
                       amends=["live-anchor", "mid-rule"], supersedes=["legacy-db"])
    ids["Z"] = _commit(store, "mid-rule-v2", "A newer mid rule.", supersedes=["mid-rule"])
    ids["N"] = _commit(store, "database", "We use SQLite for local WAL reads.",
                       **{how: ["database"]})
    return ids


def _gathered_hits(ids, live=("A", "N"), retired=("X", "P", "M")):
    scores = iter([0.97, 0.96, 0.95, 0.94, 0.93, 0.92])
    return [point_hit(ids[k], next(scores)) for k in (*retired, *live)]


def _acknowledged_after(m, ids, target, how="supersedes", target_ids="validated"):
    if target_ids == "validated":
        target_ids = {target: ids["X"]}
    _arm(m, _gathered_hits(ids, live=tuple(k for k in ("A", "N") if k in ids),
                           retired=tuple(k for k in ("X", "P", "M") if k in ids)))
    entry = _entry("database", "We use SQLite for local WAL reads.")
    rels = {"supersedes": target if how == "supersedes" else None,
            "corrects": target if how == "corrects" else None}
    return m._acknowledged_fields(entry, rels["supersedes"], rels["corrects"], {},
                                  target_ids=target_ids)


def test_a_same_slug_supersession_lists_neither_its_target_nor_its_lineage(ws):
    config, m = ws
    ids = _same_slug_supersession(m.store)
    assert len(m.store.resolve_slug("database")) == 2
    review = _acknowledged_after(m, ids, "database")
    # (a) the live anchor X amends is X's lineage, so it was exempt on the pause.
    assert review.neighbors == []
    # X, its superseded predecessor P, and (b) M, whose own chain ends at Z.
    assert review.tombstones == []


def test_re_resolving_after_the_commit_would_list_the_target_and_its_lineage(ws):
    """The trap the ids defend against is real on this graph: the control."""
    config, m = ws
    ids = _same_slug_supersession(m.store)
    review = _acknowledged_after(m, ids, "database", target_ids=None)
    assert review.neighbors == ["live-anchor"]
    assert {t["id"] for t in review.tombstones} == {ids["X"], ids["P"], ids["M"]}


def test_a_same_slug_correction_lists_neither_its_target_nor_its_predecessor(ws):
    config, m = ws
    ids = _same_slug_supersession(m.store, how="corrects")
    review = _acknowledged_after(m, ids, "database", how="corrects")
    # `corrects` walks no lineage: X by id, and P because its chain reaches X. M's
    # chain ends at Z and X's amends lineage is not walked, so M stays named.
    assert [t["id"] for t in review.tombstones] == [ids["M"]]


def test_the_pause_and_the_receipt_exempt_alike_across_the_commit(ws):
    """Pre-commit (the pause) and post-commit (the receipt) agree on one fixture."""
    config, m = ws
    ids = {"P": _commit(m.store, "legacy-db", "We use MySQL.")}
    ids["M"] = _commit(m.store, "mid-rule", "A mid rule decision.")
    ids["X"] = _commit(m.store, "database", "We use PostgreSQL.",
                       amends=["mid-rule"], supersedes=["legacy-db"])
    ids["Z"] = _commit(m.store, "mid-rule-v2", "A newer mid rule.", supersedes=["mid-rule"])
    shell = _shell(m.store)
    before = shell._review_tombstones(
        _gathering(m.store, [(ids["P"], 0.95), (ids["M"], 0.94)]),
        shell._review_declared("database", None, {}), shown_neighbors=[], bound=None)
    assert before.shown == [] and before.cleared_by == {"database"}
    ids["N"] = _commit(m.store, "database", "We use SQLite for local WAL reads.",
                       supersedes=["database"])
    after = _acknowledged_after(m, ids, "database")
    assert after.tombstones == []


def test_a_distinct_slug_supersession_on_the_real_path_lists_no_predecessor(ws):
    config, m = ws
    _retire(m, "p", by="x")
    _arm(m, [{"slug": "x", "score": 0.95}, {"slug": "p", "score": _HIGH}])
    res = _draft(m, supersedes="x", acknowledge_neighbors=True)
    assert res["status"] == "created"
    assert res["acknowledged_tombstones"] == [] and res["acknowledged_neighbors"] == []


def test_a_distinct_slug_correction_on_the_real_path_lists_no_predecessor(ws):
    config, m = ws
    _retire(m, "p", by="x")
    _arm(m, [{"slug": "x", "score": 0.95}, {"slug": "p", "score": _HIGH}])
    res = _draft(m, corrects="x", acknowledge_neighbors=True)
    assert res["acknowledged_tombstones"] == [] and res["acknowledged_neighbors"] == []


# -- Row 7: bare selects the same, and reads no slug ----------------------------------

def test_bare_and_full_reviews_select_the_same_ids_in_the_same_order(tmp_path):
    from mitos.store import GraphStore
    store = GraphStore(str(tmp_path / "graph.sqlite"))
    scored = []
    for i in range(CONFLICT_TOP_K + 2):
        old = _commit(store, f"old-{i}")
        _commit(store, f"new-{i}", supersedes=[f"old-{i}"])
        scored.append((old, round(0.97 - i * 0.01, 4)))
    counting = _Counting(store)
    shell = _shell(counting)
    gathering = _gathering(store, scored)
    none = sync.DeclaredReview(slugs=set(), ids_by_target={}, ids=set())
    bare = shell._review_tombstones(gathering, none, shown_neighbors=[], bound=None,
                                    bare=True)
    assert "resolve_slug" not in counting.calls
    full = shell._review_tombstones(gathering, none, shown_neighbors=[], bound=None)
    assert counting.calls.get("resolve_slug")
    assert [t["id"] for t in bare.shown] == [t["id"] for t in full.shown]
    assert [t["id"] for t in bare.shown] == [i for i, _ in scored]
    assert bare.omitted == full.omitted == 0
    for t in bare.shown:
        assert not {"axiom", "successor_on_pause", "successor_ambiguous"} & set(t)


def test_a_bare_review_survives_a_raising_slug_read(ws):
    config, m = ws
    _retire(m, "old-way")
    _arm(m, [{"slug": "old-way", "score": _HIGH}])
    with patch.object(MitosSyncManager, "_resolve_slug_typed",
                      side_effect=DatabaseError("injected slug fault")):
        review = m._acknowledged_fields(_entry("draft", _DRAFT), None, None, {})
    assert [t["slug"] for t in review.tombstones] == ["old-way"]


# =========================================================================== #
# The retired node's `exists` receipt
# =========================================================================== #

def _retread(m, slug, axiom=None, rejected="rej"):
    return m.record_decision_entry(axiom or f"Axiom for {slug}.", rejected, ["s"],
                                   slug=slug)


def _cited_snake(note: str) -> set:
    return set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", note))


def _tool_and_arg_names() -> set:
    from mitos import mcp_server
    names = set()
    for tool in ("amend_commentary", "record_decision", "show_node"):
        names |= set(inspect.signature(getattr(mcp_server, tool)).parameters) | {tool}
    return names


# -- Row 8: retired with a live successor ---------------------------------------------

def test_a_retired_re_tread_carries_its_handle_and_names_its_successor(ws):
    config, m = ws
    old = _retire(m, "pin", by="pin-v2")
    res = _retread(m, "pin")
    assert res["status"] == "exists" and res["state"] == "superseded"
    assert res["superseded_by"] == ["pin-v2"]
    assert not {"corrected_by", "live_end", "successor_ambiguous", "score"} & set(res)
    keys = list(res)
    assert keys[:4] == ["slug", "id", "state", "superseded_by"]
    assert keys.index("superseded_by") < keys.index("embedding")
    note = res["no_op_reason"]
    assert note.startswith(sync.EXISTS_NO_OP_FACT) and note != _EXISTS_NO_OP_NOTE
    assert "This decision is superseded" in note and "`pin-v2`" in note
    assert "`amend_commentary`" in note and old in note and "as `slug`" in note
    assert "supersedes or amends naming this slug" not in note
    assert "`pin`" not in note, "the retired slug is never a relation target"
    assert _names_a_relation(note) == ["supersedes"]
    assert _shell_tokens(note) == [], note
    assert _cited_snake(note) <= _tool_and_arg_names(), _cited_snake(note)


def test_the_merged_handle_keys_agree_with_the_one_constructor(ws):
    config, m = ws
    old = _retire(m, "pin", by="pin-v2")
    res = _retread(m, "pin")
    node = {**m.store.get_node(old), "computed_state": m.store.get_node_state(old)}
    chains, slugs = m.store.get_kill_chains([old])
    handle = retired_handle(node, score=None, stamps=m.store.get_modifiers_map([old]),
                            live_ends=live_ends_from_chains(chains, slugs))
    assert "score" not in handle
    assert {k: res[k] for k in handle} == handle


def test_a_corrected_re_tread_names_its_corrector(ws):
    config, m = ws
    _retire(m, "pin", by="pin-fix", how="corrects")
    res = _retread(m, "pin")
    assert res["state"] == "corrected" and res["corrected_by"] == ["pin-fix"]
    assert "superseded_by" not in res
    assert "`pin-fix`" in res["no_op_reason"]


# -- Row 9: through a retired successor -------------------------------------------------

def test_a_re_tread_through_a_retired_successor_points_at_the_live_end(ws):
    config, m = ws
    _retire(m, "pin", by="pin-v2")
    _seed(m, "pin-v3", "Axiom for pin-v3.", supersedes="pin-v2")
    res = _retread(m, "pin")
    assert res["superseded_by"] == ["pin-v2"] and res["live_end"] == ["pin-v3"]
    note = res["no_op_reason"]
    assert "lives in `pin-v3`" in note and "`pin-v2`" not in note
    assert "relation to `pin-v3`" in note


# -- Row 10: the successor's slug names two decisions ----------------------------------

def test_a_same_slug_re_tread_names_no_relation_and_tells_the_two_apart(ws):
    config, m = ws
    retired, successor = same_slug_lineage(m.store)
    res = m.record_decision_entry("We use PostgreSQL.", "rej", ["core"], slug="database")
    assert res["status"] == "exists" and res["id"] == retired
    assert res["superseded_by"] == ["database"] and res["successor_ambiguous"] is True
    note = res["no_op_reason"]
    assert _names_a_relation(note) == [], _names_a_relation(note)
    assert "two decisions carry that slug" in note
    assert "The slug `database` reads the live decision" in note
    assert "this decision's id reads this one" in note and retired in note
    assert _shell_tokens(note) == []
    assert _cited_snake(note) <= _tool_and_arg_names()


# -- Row 11: the chain reaches no live decision ------------------------------------------

def test_a_re_tread_whose_chain_reaches_no_live_decision_names_no_relation(ws):
    config, m = ws
    old = _retire(m, "pin", by="pin-v2")
    # A cycle with no exit is authorable only out of band.
    _inject_raw_edge(m.store, old, _id_of(m, "pin-v2"), "supersedes")
    res = _retread(m, "pin")
    assert res["live_end"] == [] and "successor_ambiguous" not in res
    note = res["no_op_reason"]
    assert "reaches no live decision" in note
    assert _names_a_relation(note) == []


# -- Row 12: a faulted handle read --------------------------------------------------------

def test_a_faulted_handle_read_nulls_all_four_keys_and_still_answers_exists(ws, capsys):
    config, m = ws
    _retire(m, "pin", by="pin-v2")
    with patch.object(m.store, "get_kill_chains", side_effect=DatabaseError("injected")):
        res = _retread(m, "pin", rejected="CORRECTED rej")
    assert res["status"] == "exists" and res["state"] == "superseded"
    for key in sync._EXISTS_HANDLE_KEYS:
        assert key in res and res[key] is None, key
    assert res["differs"] == ["rejected_paths"]
    note = res["no_op_reason"]
    assert "could not be read" in note and "`show_node`" in note
    assert _names_a_relation(note) == []
    assert "Retired-handle read failed" in capsys.readouterr().err


# -- Row 13: a live node's receipt does not move -----------------------------------------

@pytest.mark.parametrize("drifted", [False, True])
def test_a_live_re_tread_receipt_is_unchanged(ws, drifted):
    config, m = ws
    _seed(m, "pin", "Axiom for pin.")
    if drifted:
        m.store.write_signal(_id_of(m, "pin"), "drifted")
    res = _retread(m, "pin")
    assert res["state"] == ("drifted" if drifted else "active")
    assert set(res) == _LIVE_EXISTS_KEYS
    assert res["no_op_reason"] == _EXISTS_NO_OP_NOTE


# -- Row 14: the TOCTOU twin, one state read per return ----------------------------------

def test_the_toctou_twin_carries_the_same_retired_receipt_as_gate_3(ws):
    config, m = ws
    other = MitosSyncManager(config)
    real_lock = m.lock
    raced = []

    class InjectingLock:
        def __enter__(self):
            if not raced:
                entry = _entry("raced", "Raced axiom.")
                other.store.commit_parsed_entry(entry)
                other.store.commit_parsed_entry(_successor_entry("raced"))
                raced.append(True)
            return real_lock.__enter__()

        def __exit__(self, *exc):
            return real_lock.__exit__(*exc)

    m.lock = InjectingLock()
    reads = []
    real_state = MitosSyncManager._node_state

    def counted(self, node_id):
        reads.append(node_id)
        return real_state(self, node_id)

    with patch.object(MitosSyncManager, "_node_state", counted):
        twin = m.record_decision_entry("Raced axiom.", "An alternative.", [], slug="raced")
        assert raced and twin["status"] == "exists" and len(reads) == 1, (twin, reads)
        gate3 = m.record_decision_entry("Raced axiom.", "An alternative.", [], slug="raced")
        assert len(reads) == 2
    assert twin["state"] == "superseded" and twin["superseded_by"] == ["raced-v2"]
    assert twin == gate3


def _successor_entry(slug):
    entry = _entry(f"{slug}-v2", f"A newer {slug}.")
    entry.supersedes = [slug]
    return entry


# -- Row 16: parity, once per receipt -----------------------------------------------------

def _fresh_ws():
    tmp = tempfile.mkdtemp()
    config = MitosConfig(tmp)
    cmd_init(config)
    return config, MitosSyncManager(config)


def test_mcp_and_cli_json_carry_one_acknowledged_receipt(ws):
    from mitos import mcp_server
    config, m = ws
    config_b, m_b = _fresh_ws()
    try:
        for manager in (m, m_b):
            _retire(manager, "old-way", by="new-way")
            _seed(manager, "live-twin", "Axiom for live twin.")
        matches = [{"slug": "old-way", "score": _HIGH}, {"slug": "live-twin", "score": 0.88}]
        factory = _armed_real_manager_factory(matches)
        with patch("mitos.sync.MitosSyncManager", side_effect=factory), \
             patch("mitos.cli.MitosSyncManager", side_effect=factory), \
             patch("mitos.mcp_server.MitosConfig", return_value=config):
            mcp = json.loads(mcp_server.record_decision(
                _DRAFT, "rej", ["s"], slug="draft", acknowledge_neighbors=True,
                project=config.workspace_dir))
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(io.StringIO()):
                try:
                    cmd_record(config_b, axiom=_DRAFT, rejected="rej", scope=["s"],
                               slug="draft", acknowledge_neighbors=True, as_json=True)
                except SystemExit:
                    pass
            cli_json = json.loads(out.getvalue())
    finally:
        shutil.rmtree(config_b.workspace_dir, ignore_errors=True)
    assert mcp["status"] == cli_json["status"] == "created"
    assert set(mcp) == set(cli_json)
    differ = _ECHO | {"path"}  # two workspaces: each names its own
    assert {k: v for k, v in mcp.items() if k not in differ} == \
        {k: v for k, v in cli_json.items() if k not in differ}
    assert [t["slug"] for t in mcp["acknowledged_tombstones"]] == ["old-way"]
    assert mcp["acknowledged_neighbors"] == ["live-twin"]


def test_mcp_and_cli_json_carry_one_retired_exists_receipt(ws):
    from mitos import mcp_server
    config, m = ws
    _retire(m, "pin", by="pin-v2")
    with patch("mitos.mcp_server.MitosConfig", return_value=config):
        mcp = json.loads(mcp_server.record_decision(
            "Axiom for pin.", "rej", ["s"], slug="pin", project=config.workspace_dir))
    out = io.StringIO()
    with redirect_stdout(out):
        try:
            cmd_record(config, axiom="Axiom for pin.", rejected="rej", scope=["s"],
                       slug="pin", as_json=True)
        except SystemExit:
            pass
    cli_json = json.loads(out.getvalue())
    assert mcp["status"] == cli_json["status"] == "exists"
    assert set(mcp) == set(cli_json)
    assert {k: v for k, v in mcp.items() if k not in _ECHO} == \
        {k: v for k, v in cli_json.items() if k not in _ECHO}
    assert mcp["superseded_by"] == ["pin-v2"]


# =========================================================================== #
# The CLI text of both receipts
# =========================================================================== #

def _text(config, **kw):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            cmd_record(config, **kw)
        except SystemExit:
            pass
    return out.getvalue(), err.getvalue()


def test_the_text_receipt_names_each_acknowledged_tombstone_at_the_exact_floor(ws):
    config, m = ws
    old = _retire(m, "old-way", by="new-way")
    with patch("mitos.cli.MitosSyncManager",
               side_effect=_armed_real_manager_factory([{"slug": "old-way",
                                                         "score": _HIGH}])):
        out, _ = _text(config, axiom=_DRAFT, rejected="rej", scope=["s"], slug="draft",
                       acknowledge_neighbors=True)
    floor = sync._floor_text(_FLOOR)
    assert f"at or above {floor} similarity this record went past" in out
    assert f"† old-way  (superseded, score {_HIGH:.3f})  id {old}" in out
    assert "current word: new-way" in out
    assert "mitos show" in out


@pytest.mark.parametrize("value, phrase", [
    (None, "Acknowledged tombstones: unknown"),
    ([], "Acknowledged tombstones: no retired decision at or above"),
])
def test_the_text_line_tells_unknown_from_none(ws, value, phrase):
    from mitos.cli import _acknowledged_tombstones_lines
    config, _ = ws
    (line,) = _acknowledged_tombstones_lines(value, config)
    assert phrase in line
    if value == []:
        assert sync._floor_text(_FLOOR) in line


def test_the_retired_exists_text_names_the_id_and_advises_no_relation_to_the_slug(ws):
    config, m = ws
    old = _retire(m, "pin", by="pin-v2")
    out, err = _text(config, axiom="Axiom for pin.", rejected="rej", scope=["s"], slug="pin")
    assert "naming this slug" not in out
    assert "pass this to --supersedes" not in out
    assert f"Handle:    {old}" in out
    assert "current word lives in pin-v2" in out
    assert "relating to pin-v2" in out


def test_the_ambiguous_retired_exists_text_names_no_relation_flag(ws):
    config, m = ws
    retired, _ = same_slug_lineage(m.store)
    out, _ = _text(config, axiom="We use PostgreSQL.", rejected="rej", scope=["core"],
                   slug="database")
    tail = out.split("already recorded", 1)[1]
    for relation in _PAUSE_RESOLVING_RELATIONS:
        assert f"--{relation} " not in tail and f"--{relation} at" not in tail, relation
    assert "two decisions carry that slug" in out
    assert f"-- {retired}" in out or retired in out
