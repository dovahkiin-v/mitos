"""The live-tier coverage floor: its predicate, and the list it depends on.

The floor exists because the live suites degrade every environmental fault to a
loud *skip*, so a run that silently executed half its live tests still exits 0.
See the note above ``live_floor_verdict`` in conftest.py.
"""

import pathlib

import pytest

from conftest import LIVE_MODULES, _is_live, live_floor_verdict


# --- the predicate ------------------------------------------------------------

@pytest.mark.parametrize("passed, env_skipped, expect_hole", [
    (5, 0, False),   # clean full run
    (0, 0, False),   # tier off (MITOS_NO_LIVE_TESTS) or keyless fork — nothing pretended
    (0, 8, False),   # wholly unavailable (Qdrant down): honest, nothing ran
    (5, 1, True),    # the real hazard: mostly ran, one silently didn't
    (2, 6, True),    # the observed xdist collapse
])
def test_floor_fires_only_on_a_partial_run(passed, env_skipped, expect_hole):
    """Zero passes is honest; a partial run is the hole the floor exists to catch."""
    assert (live_floor_verdict(passed, env_skipped) is not None) is expect_hole


def test_verdict_names_the_numbers_and_the_escape_hatch():
    """The message has to be actionable, not just red."""
    msg = live_floor_verdict(2, 6)
    assert "6 of 8" in msg and "MITOS_STRICT_LIVE" in msg


# --- the module list ----------------------------------------------------------

def test_live_module_list_matches_the_modules_consulting_the_brake():
    """LIVE_MODULES must equal the set wired to live_tests_disabled().

    Without this, a new live module added later falls outside the floor silently —
    which is the very failure mode the floor exists to end.
    """
    tests_dir = pathlib.Path(__file__).parent
    wired = {
        p.name
        for p in list(tests_dir.glob("test_*.py")) + list((tests_dir / "golden").glob("test_*.py"))
        if p.name != pathlib.Path(__file__).name  # this file names the symbol it scans for
        and "live_tests_disabled" in p.read_text()
    }
    assert wired == set(LIVE_MODULES), (
        f"live modules and the floor's list have drifted.\n"
        f"  wired but not in LIVE_MODULES: {sorted(wired - set(LIVE_MODULES))}\n"
        f"  in LIVE_MODULES but not wired: {sorted(set(LIVE_MODULES) - wired)}"
    )


def test_is_live_matches_on_path_not_substring():
    """A nodeid is classified by its module file, not a loose substring match."""
    assert _is_live("tests/golden/test_conflict_eval_live.py::test_x")
    assert _is_live("tests/test_scenarios_live.py::TestC::test_y")
    assert not _is_live("tests/test_record_decision.py::test_z")
    assert not _is_live("tests/test_sync.py::test_not_test_scenarios_live_py")


# --- the hook's exit behaviour -------------------------------------------------

class _Rep:
    """Minimal terminalreporter stand-in: the stats dict and a write sink."""
    def __init__(self, stats):
        self.stats, self.lines = stats, []
    def write_line(self, line, **kw):
        self.lines.append(line)


class _Session:
    def __init__(self, reporter, worker=False):
        rep = reporter
        class _PM:
            def get_plugin(self, name): return rep
        class _Cfg:
            pluginmanager = _PM()
        self.config = _Cfg()
        if worker:
            self.config.workerinput = {}
        self.exitstatus = 0


def _partial_stats():
    """5 live passed, 2 live env-skipped, 1 legitimate always-skip."""
    mk = lambda nid, rep=None: type("R", (), {"nodeid": nid, "longrepr": rep})()
    return {
        "passed": [mk(f"tests/golden/test_conflict_eval_live.py::t{i}") for i in range(5)],
        "skipped": [
            mk("tests/golden/test_conflict_eval_live.py::s1", "judgment timed out — NOT a code defect"),
            mk("tests/golden/test_conflict_eval_live.py::s2", "Qdrant query failed — not a code defect"),
            mk("tests/golden/test_conflict_eval_live.py::s3", "baseline seeding is explicit-only"),
        ],
    }


def test_hook_reports_but_does_not_fail_by_default(monkeypatch):
    """The default is loud-and-green: a transient judge timeout must not red the run."""
    from conftest import pytest_sessionfinish
    monkeypatch.delenv("MITOS_STRICT_LIVE", raising=False)
    s = _Session(_Rep(_partial_stats()))
    pytest_sessionfinish(s, 0)
    assert s.exitstatus == 0
    assert any("live-floor" in ln and "2 of 7" in ln for ln in s.config.pluginmanager.get_plugin("").lines)


def test_hook_fails_under_strict(monkeypatch):
    """MITOS_STRICT_LIVE turns the same state into a hard failure for a release gate."""
    from conftest import pytest_sessionfinish
    monkeypatch.setenv("MITOS_STRICT_LIVE", "1")
    s = _Session(_Rep(_partial_stats()))
    pytest_sessionfinish(s, 0)
    assert s.exitstatus == 1


def test_hook_is_silent_on_a_clean_or_wholly_skipped_tier(monkeypatch):
    """No passes means nothing pretended to check — brake on, keyless fork, service down."""
    from conftest import pytest_sessionfinish
    monkeypatch.setenv("MITOS_STRICT_LIVE", "1")
    mk = lambda nid, rep=None: type("R", (), {"nodeid": nid, "longrepr": rep})()
    stats = {"passed": [], "skipped": [
        mk("tests/golden/test_conflict_eval_live.py::s", "Qdrant unreachable — not a code defect")]}
    s = _Session(_Rep(stats))
    pytest_sessionfinish(s, 0)
    assert s.exitstatus == 0 and not s.config.pluginmanager.get_plugin("").lines


def test_hook_no_ops_on_an_xdist_worker(monkeypatch):
    """Only the controller aggregates and owns the exit status."""
    from conftest import pytest_sessionfinish
    monkeypatch.setenv("MITOS_STRICT_LIVE", "1")
    s = _Session(_Rep(_partial_stats()), worker=True)
    pytest_sessionfinish(s, 0)
    assert s.exitstatus == 0
