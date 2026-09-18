"""Cluster C checkpoint: scope order and the render ceiling through the real CLI (2g, T8).

Phases 2a–2f proved their properties in process. Every row here enters the process a
person runs, ``python -m mitos.cli``, over a workspace built from markdown and
committed with ``rebuild --yes`` (a keyless ``sync`` commits nothing). The corpus
crosses the **shipped** ceilings with real bytes — a subprocess cannot be monkeypatched —
and every size is derived from the renderer's constants at run time, so a ceiling or row
width change resizes the fixture rather than silently un-crossing it.

No row imports a ``cmd_*`` handler to produce what it asserts. ``mitos.renderer`` and a
read-only ``GraphStore`` appear only as oracles over the graph a subprocess left behind,
and ``render_sweep`` reads the files a subprocess wrote.
"""

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set

import pytest

import mitos.renderer as R
from mitos.scope_tags import normalize_scope_tags
from mitos.store import GraphStore
from render_sweep import sweep_destinations, tree_from_disk
from test_renderer import _cli_recipe

CROSSING_NAME = "ckpt-crossing"
CONTROL_NAME = "ckpt-control"
REORDER_SLUG = "ord-zeta-alpha"
SOLO_SLUG = "solo-entry"
_SCOPES_VALUE_KEYS = ["active_decisions", "parked_open_questions",
                      "authored_first_decisions", "co_tagged_scopes"]

# The small, multi-scoped vocabulary both workspaces carry. Authored order is
# deliberately not alphabetical: `zeta, alpha` makes `zeta` primary where a sorted read
# would pick `alpha`, the pair below is a same-set reorder across two slugs, `big` is a
# degraded primary with `alpha` as its full secondary (a marker row), and `mid, zeta`
# gives `zeta.md` a row naming a full file.
_SHARED = [
    (REORDER_SLUG, ["zeta", "alpha"]),
    ("ord-alpha-zeta", ["alpha", "zeta"]),
    ("zeta-only", ["zeta"]),
    ("marker-big-alpha", ["big", "alpha"]),
    ("mid-zeta", ["mid", "zeta"]),
    (SOLO_SLUG, ["solo"]),
    ("unscoped-entry", None),
]


def _env(base: str) -> Dict[str, str]:
    """The complete subprocess env: this fixture runs outside the function-scoped
    ``hermetic_mitos_env``, so every isolation it needs is stated here."""
    return {**os.environ, "MITOS_NO_UPDATE_CHECK": "1",
            "XDG_CONFIG_HOME": os.path.join(base, "xdg_config"),
            "XDG_CACHE_HOME": os.path.join(base, "xdg_cache"),
            "GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", "ANTHROPIC_API_KEY": "",
            "QDRANT_URL": "http://localhost:1"}


def _mitos(env: Dict[str, str], cwd: str, *argv: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run([sys.executable, "-m", "mitos.cli", *argv], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)


def _entry(slug: str, axiom: str, rejected: str, tags: Optional[List[str]]) -> str:
    lines = [f"### {slug}", f"**Decided:** {axiom}", f"**Rejected:** {rejected}"]
    if tags:
        lines.append(f"**Scope:** {', '.join(tags)}")
    return "\n" + "\n".join(lines) + "\n"


def _crossing_entries() -> List[str]:
    """Sizes both over-ceiling populations from the constants the renderer applies.

    `big` has few entries with long rejected paths, so its full form is over the
    per-scope ceiling while its index is small. `huge` has many short entries, so even
    its index — rows capped near ``ONELINE_ROW_WIDTH`` — stays over. The row weight is
    measured with the renderer's own row function over this fixture's own slug shape.
    """
    ceiling = R.SCOPE_OVERFLOW_WARN_CHARS
    reject_len = ceiling // 8
    n_big = ceiling // reject_len + 4
    huge_slug = "huge-population-entry-{:03d}"
    huge_axiom = ("Huge-scope decision {} keeps its index row wide, so the per-scope "
                  "index of this population stays over its ceiling by design.")
    row_len = len(R.render_index_row({"slug": huge_slug.format(0),
                                      "core_axiom": huge_axiom.format(0)}))
    n_huge = ceiling // row_len + 10
    filler = ("A rejected alternative, spelled out at length so the big scope's full "
              "render crosses its ceiling. ")
    long_rejected = (filler * (reject_len // len(filler) + 1))[:reject_len]

    entries = [_entry(f"big-entry-{i:02d}", f"Big-scope decision {i} is load-bearing.",
                      long_rejected, ["big"]) for i in range(n_big)]
    entries += [_entry(huge_slug.format(i), huge_axiom.format(i), "Short.", ["huge"])
                for i in range(n_huge)]
    entries += [_entry(slug, f"Shared decision {slug} holds.", "Short.", tags)
                for slug, tags in _SHARED]
    return entries


def _control_entries() -> List[str]:
    entries = [_entry("big-entry-00", "Big-scope decision 0 is load-bearing.", "Short.", ["big"]),
               _entry("huge-population-entry-000", "Huge-scope decision 0.", "Short.", ["huge"])]
    entries += [_entry(slug, f"Shared decision {slug} holds.", "Short.", tags)
                for slug, tags in _SHARED]
    return entries


def _build(base: str, env: Dict[str, str], name: str, entries: List[str]) -> Dict[str, Any]:
    ws = os.path.join(base, name)
    os.makedirs(ws)
    init = _mitos(env, ws, "init", "--name", name)
    assert init.returncode == 0, init.stdout + init.stderr
    with open(os.path.join(ws, "decisions.md"), "a", encoding="utf-8") as f:
        f.write("".join(entries))
    rebuild = _mitos(env, ws, "-p", ws, "rebuild", "--yes", "--json")
    render = _mitos(env, ws, "-p", ws, "render")
    return {"ws": ws, "rebuild": rebuild, "render": render}


@pytest.fixture(scope="module")
def frame(tmp_path_factory):
    """The crossing workspace and its control, each initialised, rebuilt and rendered.

    The control carries the same init, key posture and tag vocabulary (multi-scoped
    entries included) but few short entries, so nothing crosses a ceiling. It differs
    from the crossing workspace in size, not only in overflow — an entry count equal to
    the crossing one would make `huge` cross on bodies alone. That is sound for F7
    because the property compared there is the readiness verdict and exit code, and
    neither reads entry count or size except through overflow; count-shaped values are
    named and excluded (PATTERNS' one-dimension rule, applied knowingly).
    """
    base = str(tmp_path_factory.mktemp("cluster_c"))
    env = _env(base)
    return {"env": env, "base": base,
            "crossing": _build(base, env, CROSSING_NAME, _crossing_entries()),
            "control": _build(base, env, CONTROL_NAME, _control_entries())}


def _authored(ws: str) -> Dict[str, List[str]]:
    """Every entry below the sentinel, slug → its normalised tags in authored order."""
    with open(os.path.join(ws, "decisions.md"), encoding="utf-8") as f:
        text = f.read()
    body = text[text.index("BEGIN ENTRIES"):]
    out: Dict[str, List[str]] = {}
    for block in re.split(r"^### ", body, flags=re.M)[1:]:
        slug = block.split("\n", 1)[0].strip()
        scope = re.search(r"^\*\*Scope:\*\* (.*)$", block, re.M)
        out[slug] = normalize_scope_tags(scope.group(1).split(",")) if scope else []
    return out


def _oracle(ws: str) -> Dict[str, Any]:
    """Recomputes, over the graph a subprocess left, which files must be over a ceiling.

    Measured at each scope file's maximum form (``_full_scope_file`` with no degrade
    set, the pass-one measure) and the full global render, never through
    ``assemble_render``: a degrade predicate forced off would also be forced off in this
    process, and an oracle sharing the defect cannot see it.
    """
    store = GraphStore(os.path.join(ws, ".mitos", "graph.sqlite"), read_only=True)
    decs = store.get_active_decisions()
    mods = store.get_modifiers_map([d["id"] for d in decs])
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for d in decs:
        for s in d.get("scope") or []:
            groups.setdefault(s, []).append(d)
    over = {s for s, ds in groups.items()
            if len(R._full_scope_file(s, ds, mods)["content"]) > R.SCOPE_OVERFLOW_WARN_CHARS}
    global_full = sum(len(R.render_node_markdown(d, mods.get(d["id"]))) for d in decs)
    return {"scopes": set(groups), "over": over, "global_full_chars": global_full}


def _body_lines(content: str) -> Set[str]:
    return {line[len("## "):] for line in content.splitlines() if line.startswith("## ")}


def _index_slugs(content: str) -> List[str]:
    return re.findall(r"^- \*\*(.+?)\*\* — ", content, re.M)


def _json_stdout(proc: "subprocess.CompletedProcess[str]") -> Dict[str, Any]:
    return json.loads(proc.stdout)


def _copy(frame: Dict[str, Any], label: str) -> str:
    dst = os.path.join(frame["base"], label)
    shutil.copytree(frame["crossing"]["ws"], dst)
    return dst


def _edit_block(ws: str, slug: str, edit) -> None:
    path = os.path.join(ws, "decisions.md")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    start = text.index(f"\n### {slug}\n") + 1
    end = text.find("\n### ", start)
    end = len(text) if end < 0 else end + 1
    block = text[start:end]
    new = edit(block)
    assert new != block, slug
    with open(path, "w", encoding="utf-8") as f:
        f.write(text[:start] + new + text[end:])


def _hashes(directory: str) -> Dict[str, str]:
    out = {}
    for name in sorted(os.listdir(directory)):
        with open(os.path.join(directory, name), "rb") as f:
            out[name] = hashlib.sha256(f.read()).hexdigest()
    return out


def _ceiling_for(record: Dict[str, Any]) -> int:
    return R.GLOBAL_OVERFLOW_WARN_CHARS if record["scope"] is None else R.SCOPE_OVERFLOW_WARN_CHARS


# --------------------------------------------------------------------------- #
# F1 — the fixture reached the graph
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("which", ["crossing", "control"])
def test_f1_rebuild_commits_every_authored_entry(frame, which) -> None:
    """F1: `rebuild --yes --json` swapped, with every entry the markdown holds committed
    and no casualty — the cheap guard that every later row reads a built graph."""
    built = frame[which]
    assert built["rebuild"].returncode == 0, built["rebuild"].stderr
    report = _json_stdout(built["rebuild"])
    assert report["swapped"] is True
    assert report["gate_passed"] is True
    assert report["residual_casualties"] == []
    assert report["decisions_committed"] == len(_authored(built["ws"]))


# --------------------------------------------------------------------------- #
# F2 — degrade and pointer honesty, read off disk
# --------------------------------------------------------------------------- #

def test_f2_the_degraded_set_on_disk_is_the_over_ceiling_set(frame) -> None:
    """F2: the scope files that call themselves indexes are exactly those whose maximum
    form is over the ceiling, and the tree on disk names no non-full destination.

    A row asserting only "big.md is an index" passes under a renderer that indexes
    everything, so the set is compared whole against an oracle recomputed from the graph.
    """
    ws, render = frame["crossing"]["ws"], frame["crossing"]["render"]
    assert render.returncode == 0, render.stderr
    assert "[Warning]" not in render.stderr
    oracle = _oracle(ws)
    tree = tree_from_disk(ws)
    scope_ceiling = R.SCOPE_OVERFLOW_WARN_CHARS

    # Each crossing first, or nothing below means anything.
    assert {"big", "huge"} <= oracle["over"]
    assert oracle["global_full_chars"] > R.GLOBAL_OVERFLOW_WARN_CHARS
    assert set(tree["scopes"]) == oracle["scopes"]

    indexed = {s for s, f in tree["scopes"].items() if f["mode"] == "index"}
    assert indexed == oracle["over"]
    assert len(tree["scopes"]["huge"]["content"]) > scope_ceiling
    assert len(tree["scopes"]["big"]["content"]) <= scope_ceiling
    for s, f in tree["scopes"].items():
        if f["mode"] == "full":
            assert len(f["content"]) <= scope_ceiling, s
    assert tree["global"]["mode"] == "index"

    violations, counts = sweep_destinations(tree)
    assert violations == []
    for kind in ("file_heading", "index_heading", "unscoped_heading", "marker_row"):
        assert counts[kind] > 0, kind


# --------------------------------------------------------------------------- #
# F3 — the authored first tag is the primary
# --------------------------------------------------------------------------- #

def test_f3_every_multi_scoped_body_renders_under_its_first_authored_tag(frame) -> None:
    """F3: a multi-scoped entry's body is in its first authored tag's file (or its index
    row, when that file is an index) and is a body in no other scope file. Expected tags
    come from the markdown, not from the graph."""
    ws = frame["crossing"]["ws"]
    tree = tree_from_disk(ws)
    multi = {slug: tags for slug, tags in _authored(ws).items() if len(tags) > 1}
    # Non-vacuity: a sorted read must pick a different primary for a full file here.
    assert any(tags[0] != sorted(tags)[0] and tree["scopes"][tags[0]]["mode"] == "full"
               for tags in multi.values())
    assert any(tree["scopes"][tags[0]]["mode"] == "index" for tags in multi.values())
    for slug, tags in multi.items():
        primary = tree["scopes"][tags[0]]
        if primary["mode"] == "full":
            assert slug in _body_lines(primary["content"]), (slug, tags)
        else:
            assert slug in _index_slugs(primary["content"]), (slug, tags)
        for s, f in tree["scopes"].items():
            if s != tags[0]:
                assert slug not in _body_lines(f["content"]), (slug, s)


# --------------------------------------------------------------------------- #
# F4 — a reorder through the frame
# --------------------------------------------------------------------------- #

def test_f4_a_scope_reorder_moves_the_body_and_keeps_the_node(frame, tmp_path) -> None:
    """F4: reversing an entry's two tags, then `rebuild` and `render`, moves its body to
    the new first tag's file; `show --json` returns the same id and `status --json` the
    same node count.

    The reconcile half (`sync --reconcile-entry`) needs a key — a keyless sync returns
    at the key floor above the per-entry loop — so it stays 2b's in-process R1.
    """
    env = frame["env"]
    ws = _copy(frame, "f4")
    old_first, new_first = _authored(ws)[REORDER_SLUG]
    before = tree_from_disk(ws)
    assert before["scopes"][old_first]["mode"] == before["scopes"][new_first]["mode"] == "full"
    assert REORDER_SLUG in _body_lines(before["scopes"][old_first]["content"])

    show_before = _mitos(env, ws, "-p", ws, "show", "--json", "--", REORDER_SLUG)
    status_before = _mitos(env, ws, "-p", ws, "status", "--json")
    _edit_block(ws, REORDER_SLUG, lambda b: b.replace(
        f"**Scope:** {old_first}, {new_first}", f"**Scope:** {new_first}, {old_first}"))
    assert _authored(ws)[REORDER_SLUG] == [new_first, old_first]

    rebuild = _mitos(env, ws, "-p", ws, "rebuild", "--yes", "--json")
    assert _json_stdout(rebuild)["swapped"] is True, rebuild.stderr
    render = _mitos(env, ws, "-p", ws, "render")
    assert render.returncode == 0 and "[Warning]" not in render.stderr, render.stderr

    after = tree_from_disk(ws)
    assert REORDER_SLUG in _body_lines(after["scopes"][new_first]["content"])
    assert REORDER_SLUG not in _body_lines(after["scopes"][old_first]["content"])
    show_after = _mitos(env, ws, "-p", ws, "show", "--json", "--", REORDER_SLUG)
    assert _json_stdout(show_after)["id"] == _json_stdout(show_before)["id"]
    status_after = _mitos(env, ws, "-p", ws, "status", "--json")
    assert (_json_stdout(status_after)["checks"]["graph_nodes"]
            == _json_stdout(status_before)["checks"]["graph_nodes"])


# --------------------------------------------------------------------------- #
# F5 — the scope-health report
# --------------------------------------------------------------------------- #

def test_f5_scopes_json_counts_what_the_markdown_authored(frame) -> None:
    """F5: `scopes --json` is one object with the shipped envelope and value key order,
    busiest-first, and each tag's counts equal what the markdown authored."""
    ws = frame["crossing"]["ws"]
    proc = _mitos(frame["env"], ws, "-p", ws, "scopes", "--json")
    assert proc.returncode == 0, proc.stderr
    payload = _json_stdout(proc)
    assert list(payload) == ["scopes", "project", "collection", "workspace"]
    scopes = payload["scopes"]
    for tag, value in scopes.items():
        assert list(value) == _SCOPES_VALUE_KEYS, tag
        assert all(type(v) is int for v in value.values()), tag
    assert list(scopes) == sorted(scopes, key=lambda t: (
        -(scopes[t]["active_decisions"] + scopes[t]["parked_open_questions"]), t))
    assert "superseded_by" not in proc.stdout and "amended_by" not in proc.stdout

    authored = [tags for tags in _authored(ws).values() if tags]
    tags_all = {t for tags in authored for t in tags}
    active = {t: sum(t in tags for tags in authored) for t in tags_all}
    first = {t: sum(tags[0] == t for tags in authored) for t in tags_all}
    partners = {t: len({p for tags in authored if t in tags for p in tags} - {t})
                for t in tags_all}
    # Non-vacuity: alphabetical primacy would count a different first.
    assert first != {t: sum(sorted(tags)[0] == t for tags in authored) for t in tags_all}
    assert {t: v["active_decisions"] for t, v in scopes.items()} == active
    assert {t: v["authored_first_decisions"] for t, v in scopes.items()} == first
    assert {t: v["co_tagged_scopes"] for t, v in scopes.items()} == partners
    assert all(v["parked_open_questions"] == 0 for v in scopes.values())


# --------------------------------------------------------------------------- #
# F6 — the vacated-scope sweep and both fences
# --------------------------------------------------------------------------- #

def test_f6_the_sweep_runs_only_unfiltered_and_takes_only_its_own_files(frame) -> None:
    """F6: a scope vacated by a rebuild leaves a stale titled file. A filtered render
    removes nothing and rewrites no other scope file; an unfiltered one removes it and
    leaves a person's `notes.md` byte-identical.

    `live_axioms.md` is outside the byte-identity claim: `render_all` rewrites the global
    file on every call, filtered or not, and the vacate changed its content.
    """
    env = frame["env"]
    ws = _copy(frame, "f6")
    axioms = os.path.join(ws, ".mitos", "axioms")
    with open(os.path.join(axioms, "notes.md"), "w", encoding="utf-8") as f:
        f.write("# Notes\nKept by hand, not by the renderer.\n")
    _edit_block(ws, SOLO_SLUG, lambda b: b.replace("**Scope:** solo\n", ""))
    rebuild = _mitos(env, ws, "-p", ws, "rebuild", "--yes", "--json")
    assert _json_stdout(rebuild)["swapped"] is True, rebuild.stderr
    with open(os.path.join(axioms, "solo.md"), encoding="utf-8") as f:
        assert f.readline().rstrip("\n") == R._scope_title("solo")

    target = "zeta"
    before = _hashes(axioms)
    filtered = _mitos(env, ws, "-p", ws, "render", "--scope", target)
    assert filtered.returncode == 0 and "[Warning]" not in filtered.stderr, filtered.stderr
    after = _hashes(axioms)
    assert "solo.md" in after
    others = f"{target}.md"
    assert {k: v for k, v in after.items() if k != others} == \
        {k: v for k, v in before.items() if k != others}

    full = _mitos(env, ws, "-p", ws, "render")
    assert full.returncode == 0 and "[Warning]" not in full.stderr, full.stderr
    swept = _hashes(axioms)
    assert "solo.md" not in swept
    assert swept["notes.md"] == before["notes.md"]


# --------------------------------------------------------------------------- #
# F7 — status: the verdict is unchanged, the channel is honest
# --------------------------------------------------------------------------- #

# Values that differ between the two workspaces because they count or name their
# entries, not because of overflow. The buffer's two size counts (surface-entropy 3d)
# are the same class: the two corpora hold different entries.
_COUNT_SHAPED_CHECKS = {"graph_nodes", "active_nodes",
                        "decisions_buffer_entries", "decisions_buffer_chars"}


def test_f7_status_reports_overflow_without_moving_the_verdict(frame) -> None:
    """F7: crossing and control agree on `ready`, the exit code and every non-count
    check, while `scope_overflow` names exactly the over-ceiling files on disk — each an
    index — and is empty on the control. The text footer routes to the bounded tier.

    Limit, stated: both keyless workspaces are already NEEDS ATTENTION, so this frame
    cannot see overflow flip a READY verdict. That property is pinned in process by
    `tests/test_status_readiness.py::test_status_reports_scope_overflow_detail`.
    """
    env = frame["env"]
    crossing, control = frame["crossing"]["ws"], frame["control"]["ws"]
    c_proc = _mitos(env, crossing, "-p", crossing, "status", "--json")
    t_proc = _mitos(env, control, "-p", control, "status", "--json")
    c, t = _json_stdout(c_proc), _json_stdout(t_proc)

    control_tree = tree_from_disk(control)
    assert control_tree["global"]["mode"] == "full"
    assert all(f["mode"] == "full" for f in control_tree["scopes"].values())
    assert t["scope_overflow"] == []

    assert c_proc.returncode == t_proc.returncode
    assert c["ready"] == t["ready"]
    assert list(c["checks"]) == list(t["checks"])
    differing = {k for k in c["checks"] if c["checks"][k] != t["checks"][k]}
    assert differing <= _COUNT_SHAPED_CHECKS, differing

    tree = tree_from_disk(crossing)
    records = [tree["global"], *tree["scopes"].values()]
    expected = {f["name"]: len(f["content"]) for f in records
                if len(f["content"]) > _ceiling_for(f)}
    assert expected
    by_name = {f["name"]: f for f in records}
    assert {o["name"]: o["chars"] for o in c["scope_overflow"]} == expected
    for o in c["scope_overflow"]:
        record = by_name[o["name"]]
        assert o["threshold_chars"] == _ceiling_for(record)
        assert record["mode"] == "index", o["name"]
        assert isinstance(o["top_decisions"], list) and o["top_decisions"]

    text = _mitos(env, crossing, "-p", crossing, "status", "-v")
    assert "mitos list --scope=" in text.stdout
    assert f"-p {c['project']!r}" in text.stdout
    assert "re-scope" not in text.stdout


# --------------------------------------------------------------------------- #
# F8 — the recipes the rendered files print actually run
# --------------------------------------------------------------------------- #

def _cli_oneline_slugs(stdout: str) -> List[str]:
    lines = stdout.splitlines()
    start = lines.index("-" * 80) + 1
    rows = []
    for line in lines[start:]:
        if not line.strip():
            break
        rows.append(line.split("  ", 1)[0])
    return rows


def test_f8_the_printed_recipes_run_from_the_workspace_root(frame) -> None:
    """F8: the `list` recipe in `big.md`'s header runs verbatim from the workspace root
    and lists exactly the index's rows; the `show` recipe runs for a marker row's slug.

    Both recipes come from ``_list_forms`` / ``_show_forms``, which every emitter shares
    (scope headers, the global banner, the pointer section), so one run of each form
    covers them all. The show recipe is printed once, in slot form, in the pointer
    section heading; running it means putting the marker row's slug in place of
    `<slug>` — the one substitution, and the one the file tells its reader to make.
    """
    env, ws = frame["env"], frame["crossing"]["ws"]
    tree = tree_from_disk(ws)
    big = tree["scopes"]["big"]
    assert big["mode"] == "index"

    argv = shlex.split(_cli_recipe(big["content"]))
    assert argv[0] == "mitos"
    listed = _mitos(env, ws, *argv[1:])
    assert listed.returncode == 0, listed.stdout + listed.stderr
    rows = _cli_oneline_slugs(listed.stdout)
    file_rows = _index_slugs(big["content"])
    assert file_rows and len(rows) == len(file_rows)
    assert sorted(rows) == sorted(file_rows)

    content, row = next((f["content"], line) for f in tree["scopes"].values()
                        if f["mode"] == "full" for line in f["content"].splitlines()
                        if line.endswith(R.POINTER_INDEX_TARGET_MARKER))
    slug = re.match(r"^- \*\*(.+?)\*\* — ", row).group(1)
    slot = re.search(r"`(mitos show [^`]*)`", content).group(1)
    assert "<slug>" in slot
    shown = _mitos(env, ws, *shlex.split(slot.replace("<slug>", shlex.quote(slug)))[1:])
    assert shown.returncode == 0, shown.stdout + shown.stderr
    assert slug in shown.stdout
