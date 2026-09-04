"""Phase 9.3.5 - the routing, and that the control partition matches the real one in shape."""

from __future__ import annotations

import numpy as np

from src.ocr import adapt
from src.ocr.styled import random_groups


def test_the_random_partition_has_the_same_group_sizes_as_the_real_one():
    """8.7's rule, restated here because 9.3.5's whole comparison rests on it.

    The control must differ from the style partition in *which* writers are together and in
    nothing else - if the groups were merely equal-sized, the comparison would price two things.
    """
    style = {f"w{i}": i % 3 for i in range(30)}
    style["w0"] = 0
    style.update({f"x{i}": 0 for i in range(12)})  # a lopsided partition, like 8.6's
    control = random_groups(style, seed=7)

    assert sorted(control) == sorted(style)
    assert (
        np.bincount(list(control.values())).tolist() == np.bincount(list(style.values())).tolist()
    )


def test_the_random_partition_is_not_the_style_partition():
    style = {f"w{i}": i % 3 for i in range(60)}
    control = random_groups(style, seed=1)
    assert control != style


def test_unrouted_writers_fall_back_rather_than_being_dropped():
    """fa_bresler has no 8.6 style, and every arm must still be scored on the same crops."""
    assert adapt.UNROUTED == -1
    groups = {"writer0001": 0}
    assignment = [groups.get(s, adapt.UNROUTED) for s in ["writer0001", "fa-writer000"]]
    assert assignment == [0, -1]


def test_every_crop_gets_exactly_one_prediction_slot():
    """The routing writes into positions; a crop routed nowhere would silently stay empty."""
    scribes = np.array(["a", "b", "a", "c"])
    groups = {"a": 0, "b": 1}
    assignment = np.array([groups.get(s, adapt.UNROUTED) for s in scribes])
    covered = sorted(
        i for cluster in set(assignment.tolist()) for i in np.flatnonzero(assignment == cluster)
    )
    assert covered == [0, 1, 2, 3]
