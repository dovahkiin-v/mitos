"""Unit tests for the shared Letter-payload shaper (`mitos.display.letter_payload`).

Phase 2a (W3): one helper owns the per-decision Letter key set, the M5
``rejected_paths``-unless-``brief`` rule, and the deterministic ``extras`` slot.
These tests pin the contract; the byte-identity of the five routed CLI⇄MCP sites
is proven by the existing shape/parity/economy regression nets staying green.
"""

from mitos.display import letter_payload


def _node():
    """Returns a minimal synthetic decision node dict (no services needed)."""
    return {
        "slug": "use-sqlite",
        "core_axiom": "Persist the graph in SQLite.",
        "scope": ["storage"],
        "rejected_paths": "Postgres: heavier ops burden for a single-user CLI.",
    }


def test_core_keys_and_order():
    """The core is exactly slug, axiom, scope in that order (no extras, full)."""
    payload = letter_payload(_node(), brief=True)
    assert list(payload.keys()) == ["slug", "axiom", "scope"]
    assert payload["slug"] == "use-sqlite"
    assert payload["axiom"] == "Persist the graph in SQLite."
    assert payload["scope"] == ["storage"]


def test_axiom_reads_from_core_axiom():
    """The ``axiom`` key is populated from the node's ``core_axiom`` field."""
    payload = letter_payload(_node(), brief=True)
    assert "core_axiom" not in payload
    assert payload["axiom"] == "Persist the graph in SQLite."


def test_brief_omits_rejected_paths():
    """``brief=True`` drops ``rejected_paths`` and nothing else."""
    payload = letter_payload(_node(), brief=True)
    assert "rejected_paths" not in payload


def test_full_includes_rejected_paths():
    """``brief=False`` includes the M5 ``rejected_paths`` fence."""
    payload = letter_payload(_node(), brief=False)
    assert payload["rejected_paths"] == "Postgres: heavier ops burden for a single-user CLI."


def test_extras_land_between_scope_and_rejected_paths():
    """``extras`` interleave between ``scope`` and ``rejected_paths``, in order."""
    payload = letter_payload(
        _node(),
        brief=False,
        extras={"state": "active", "score": 0.91, "depth_mode": "letter"},
    )
    assert list(payload.keys()) == [
        "slug",
        "axiom",
        "scope",
        "state",
        "score",
        "depth_mode",
        "rejected_paths",
    ]


def test_extras_preserve_insertion_order():
    """The helper preserves the caller's ``extras`` dict order verbatim."""
    payload = letter_payload(_node(), brief=True, extras={"score": 0.5, "state": "drifted"})
    assert list(payload.keys()) == ["slug", "axiom", "scope", "score", "state"]


def test_extras_before_rejected_paths_under_brief():
    """Under ``brief``, ``extras`` are present but ``rejected_paths`` is not."""
    payload = letter_payload(_node(), brief=True, extras={"score": 0.5})
    assert list(payload.keys()) == ["slug", "axiom", "scope", "score"]


def test_no_modifier_keys_emitted():
    """Stamping is the caller's job — the helper emits no modifier keys."""
    payload = letter_payload(_node(), brief=False, extras={"score": 0.5})
    for mod_key in ("superseded_by", "amended_by", "narrowed_by", "corrected_by"):
        assert mod_key not in payload
