"""Cross-cutting - the parallel helper must not change any answer."""

from __future__ import annotations

import os

from src.utils import parallel


def square(x):
    return x * x


def test_results_come_back_in_submission_order():
    """The whole point: a parallel run and a serial run must be byte-identical."""
    items = list(range(200))
    assert parallel.pmap(square, items) == [square(i) for i in items]


def test_small_inputs_stay_serial():
    """Starting a pool for four items costs more than it saves."""
    assert parallel.pmap(square, [1, 2, 3]) == [1, 4, 9]


def test_one_worker_is_a_plain_loop():
    assert parallel.pmap(square, list(range(50)), n_jobs=1) == [i * i for i in range(50)]


def test_worker_count_leaves_a_core_free():
    assert parallel.workers() == max(1, (os.cpu_count() or 2) - 1)
    assert parallel.workers(4) == 4


def test_pstarmap_passes_several_arguments():
    pairs = [(i, i + 1) for i in range(30)]
    assert parallel.pstarmap(lambda a, b: a * b, pairs) == [a * b for a, b in pairs]


def test_threads_backend_gives_the_same_answers():
    items = list(range(100))
    assert parallel.pmap(square, items, prefer="threads") == parallel.pmap(square, items)
