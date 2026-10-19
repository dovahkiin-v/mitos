"""Phase 6a: `recall.top_cluster_size`, the one place top-cluster membership is decided.

The margin here is binary-exact (0.25) so the at-margin rows compare exact values, as
`test_ranked_edge`'s policy table does; the shipped margin is injected by the caller.
6b wires the count onto the envelope; until then the instruments are its readers.
"""

import inspect

import pytest

from mitos import recall
from mitos.recall import top_cluster_size

_M = 0.25


@pytest.mark.parametrize("scores, expected", [
    ([], 0),                              # an empty list has no cluster
    ([0.9], 1),                           # rank 1 alone is a cluster of one
    ([1.0, 0.75], 1),                     # exactly the margin below: a clear drop (strict)
    ([1.0, 0.75 + 2 ** -10], 2),          # just inside the margin
    ([1.0, 1.0, 1.0], 3),                 # ties all sit in the cluster
    ([1.0, 0.875, 0.5, 0.875], 3),        # every hit within the margin counts, not a prefix
    ([0.5, 0.625, 0.25], 2),              # a hit above rank 1 (negative gap) is a member
    ([1.0, 0.5, 0.25], 1),                # a drop after rank 1
])
def test_top_cluster_size_counts_hits_strictly_within_the_margin(scores, expected):
    assert top_cluster_size(scores, margin=_M) == expected


def test_the_margin_is_a_required_keyword():
    """A second ranking passes its own margin; a default would read the wrong one silently."""
    param = inspect.signature(top_cluster_size).parameters["margin"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        top_cluster_size([1.0, 0.9])  # type: ignore[call-arg]


def test_membership_matches_window_edges_strictness():
    """A gap equal to the margin is a clear drop in both rules (2e's `window_edge`)."""
    assert "a gap equal to it is a clear drop" in " ".join(inspect.getdoc(recall.window_edge).split())
    assert top_cluster_size([0.5, 0.25], margin=0.25) == 1
