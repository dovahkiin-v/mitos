"""The destination sweep over an assembled render tree (T5's pointer half, phase 2d).

D5's pointer invariant: no render surface names, as a destination for full entries, a
file that is an index or over its ceiling — nor the markdown corpus. This parses every
destination by the clause shapes the renderer emits (a global group heading's
``.mitos/axioms/<name>`` and a secondary row's ``→ full entry: <name>``), never by bare
``.md`` tokens, because a real corpus has axiom text that names ``decisions.md``.

It takes any tree ``assemble_render`` returns, so a checkpoint can run it over a real
corpus as well as over a fixture. It returns counts alongside the violations so every
caller can assert non-vacuity: a parser that matches nothing reports no violations.
``tree_from_disk`` builds the same shape from the files a render wrote, so a caller
outside the rendering process runs this sweep over bytes on disk (phase 2g).
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import mitos.renderer as R

_HEADING_FILE = " — full entries: .mitos/axioms/"
_ROW_FILE = " → full entry: "
_UNSCOPED = "## (unscoped)"
# The global file's title; its index form carries the same suffix a scope index does.
_GLOBAL_TITLE = "# Live Axioms"


def _read_rendered(path: str) -> str:
    """Reads a rendered file as the exact string the renderer wrote (no newline translation)."""
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def tree_from_disk(workspace_dir: str) -> Dict[str, Any]:
    """Reads a rendered tree on disk into the dict ``sweep_destinations`` takes.

    Each file's ``mode`` comes from the file's own self-declaration, never from a size
    this reader measures: a scope file is an index when its first line is the scope
    title plus ``_INDEX_TITLE_SUFFIX``, and ``live_axioms.md`` when its first line is
    the index title. So a render that wrote the wrong form is read as the form it
    claims, and the sweep holds that claim to its ceiling.

    Only titled files are admitted as scope records: a ``.md`` whose first line is not
    ``_scope_title(stem)`` or its index form (a person's ``notes.md``) is not a render,
    and admitting it would let it answer a pointer as a destination.

    Args:
        workspace_dir: The workspace root holding ``live_axioms.md`` and
            ``.mitos/axioms/``.

    Returns:
        ``{"global": <file>, "scopes": {stem: <file>}}``, each ``<file>`` being
        ``{"name", "scope", "mode", "content"}`` with ``scope`` ``None`` for the global
        file.

    Raises:
        ValueError: ``live_axioms.md`` does not open with either global title.
    """
    content = _read_rendered(os.path.join(workspace_dir, "live_axioms.md"))
    first = content.split("\n", 1)[0]
    if first == _GLOBAL_TITLE + R._INDEX_TITLE_SUFFIX:
        mode = "index"
    elif first == _GLOBAL_TITLE:
        mode = "full"
    else:
        raise ValueError(f"live_axioms.md opens {first!r}, which is no global title")
    tree: Dict[str, Any] = {
        "global": {"name": "live_axioms.md", "scope": None, "mode": mode, "content": content},
        "scopes": {},
    }
    axioms_dir = os.path.join(workspace_dir, ".mitos", "axioms")
    for name in sorted(os.listdir(axioms_dir)):
        if not name.endswith(".md"):
            continue
        stem = name[:-len(".md")]
        content = _read_rendered(os.path.join(axioms_dir, name))
        first = content.split("\n", 1)[0]
        if first == R._scope_title(stem) + R._INDEX_TITLE_SUFFIX:
            mode = "index"
        elif first == R._scope_title(stem):
            mode = "full"
        else:
            continue
        tree["scopes"][stem] = {"name": name, "scope": stem, "mode": mode, "content": content}
    return tree


def sweep_destinations(assembled: Dict[str, Any],
                       scope_ceiling: Optional[int] = None
                       ) -> Tuple[List[str], Dict[str, int]]:
    """Sweeps an assembled tree for pointer surfaces that name a non-full destination.

    Args:
        assembled: The dict ``assemble_render`` returns.
        scope_ceiling: The per-scope ceiling to hold destinations to; defaults to
            ``SCOPE_OVERFLOW_WARN_CHARS`` read now.

    Returns:
        ``(violations, counts)``: one message per violation, and how many surfaces of
        each kind the sweep parsed.
    """
    ceiling = R.SCOPE_OVERFLOW_WARN_CHARS if scope_ceiling is None else scope_ceiling
    records = {r["name"]: r for r in assembled["scopes"].values()}
    files = [assembled["global"]] + list(assembled["scopes"].values())
    violations: List[str] = []
    counts = {"file_heading": 0, "index_heading": 0, "unscoped_heading": 0,
              "section_block": 0, "file_row": 0, "marker_row": 0}

    def resolve(where: str, name: str) -> None:
        record = records.get(name)
        if record is None:
            violations.append(f"{where}: names {name!r}, which is no scope file")
        elif record["mode"] != "full":
            violations.append(f"{where}: names {name!r}, which is an index")
        elif len(record["content"]) > ceiling:
            violations.append(f"{where}: names {name!r}, which is over its ceiling")

    for f in files:
        content, name = f["content"], f["name"]
        for phrase in ("canonical full render", "(full entries elsewhere)"):
            if phrase in content:
                violations.append(f"{name}: contains {phrase!r}")

        if f["scope"] is None and f["mode"] == "index":
            banner = content.split("\n## ", 1)[0]
            if ".mitos/axioms/" in banner:
                violations.append(f"{name}: banner names a scope file path")
            for line in content.splitlines():
                if not line.startswith("## "):
                    continue
                where = f"{name} heading {line!r}"
                if _HEADING_FILE in line:
                    counts["file_heading"] += 1
                    resolve(where, line.rsplit(_HEADING_FILE, 1)[1])
                elif line.startswith(_UNSCOPED):
                    if line == _UNSCOPED:
                        counts["unscoped_heading"] += 1
                    else:
                        violations.append(f"{where}: unscoped heading carries a clause")
                elif line.endswith(" — " + R.INDEX_GROUP_CLAUSE):
                    counts["index_heading"] += 1
                    scope = line[len("## "):-len(" — " + R.INDEX_GROUP_CLAUSE)]
                    record = records.get(f"{scope}.md")
                    if record is None or record["mode"] != "index":
                        violations.append(f"{where}: says index over a file that is not one")
                else:
                    violations.append(f"{where}: unrecognised group heading shape")

        elif f["scope"] is not None and f["mode"] == "full":
            opener = "\n" + R.POINTER_SECTION_HEADING + "\n"
            at = content.find(opener)
            if at < 0:
                continue
            counts["section_block"] += 1
            if ".md" in R.POINTER_SECTION_HEADING or "decisions/archive" in R.POINTER_SECTION_HEADING:
                violations.append(f"{name}: pointer section block names a file")
            for line in content[at + len(opener):].splitlines():
                where = f"{name} row {line!r}"
                if not line.startswith("- **"):
                    violations.append(f"{where}: unexpected line in the pointer section")
                elif line.endswith(R.POINTER_INDEX_TARGET_MARKER):
                    counts["marker_row"] += 1
                elif _ROW_FILE in line:
                    counts["file_row"] += 1
                    resolve(where, line.rsplit(_ROW_FILE, 1)[1])
                else:
                    violations.append(f"{where}: row names no destination and no marker")
    return violations, counts
