"""Phase 10.1.7 - crossing vs junction on synthetic geometry with an exactly known answer."""

from __future__ import annotations

import math

import pytest

from src.assemble import crossings as C
from src.assemble.crossings import Branch, best_matching, classify_junction


def deg(*angles: float, curvature: tuple[float, ...] | None = None) -> list[Branch]:
    curves = curvature or (0.0,) * len(angles)
    return [Branch(angle=math.radians(a), curvature=k) for a, k in zip(angles, curves, strict=True)]


# -- the five named shapes, each a sentence -------------------------------------------------


def test_a_clean_x_is_a_crossing_with_opposite_arms_paired():
    verdict = classify_junction(deg(0, 90, 180, 270))
    assert verdict.is_crossing
    assert set(verdict.pairing) == {(0, 2), (1, 3)}


def test_a_clean_t_is_a_junction_not_a_crossing():
    verdict = classify_junction(deg(0, 180, 270))
    assert not verdict.is_crossing
    assert verdict.kind == "junction"


def test_a_shallow_angle_crossing_is_still_a_crossing():
    """Two strokes 20 degrees off a right angle - well inside ANGLE_TOL - still cross."""
    verdict = classify_junction(deg(0, 20, 180, 200))
    assert verdict.is_crossing
    assert set(verdict.pairing) == {(0, 2), (1, 3)}


def test_a_curved_crossing_pairs_by_continuity_not_by_raw_angle():
    """5 degrees off antiparallel plus matching curvature on each half of one stroke."""
    verdict = classify_junction(deg(0, 90, 175, 265, curvature=(0.3, 0.3, -0.3, -0.3)))
    assert verdict.is_crossing
    assert set(verdict.pairing) == {(0, 2), (1, 3)}


def test_a_y_branch_is_a_junction():
    verdict = classify_junction(deg(90, 210, 330))
    assert not verdict.is_crossing


# -- pairing is optimal, not greedy ----------------------------------------------------------


def test_best_matching_beats_the_greedy_first_pair():
    """Branch 0's single nearest-to-antiparallel partner is branch 2 (1 deg of deviation), but
    taking that pair forces the leftovers (1, 3) into an 8-degree pair, for a worse total than
    pairing (0, 3) + (1, 2) (3 deg + 4 deg). `best_matching` must find the second, better,
    matching rather than greedily locking in branch 0's single best partner first."""
    branches = deg(0, 5, 181, 177)
    pairing, unpaired, total = best_matching(branches)
    assert not unpaired
    assert set(pairing) == {(0, 3), (1, 2)}
    # The greedy choice - pair 0 with its single closest-to-antiparallel partner - is branch 2
    # (1 degree of deviation, the smallest of any pair), which the optimal matching rejects.
    greedy_first_pair = (0, 2)
    assert greedy_first_pair not in pairing
    # And the greedy total (using 0-2 and stuck with 1-3) is worse than what was found.
    greedy_total = (
        C.pair_score(branches[0], branches[2])[0] + C.pair_score(branches[1], branches[3])[0]
    )
    assert total > greedy_total


def test_best_matching_is_the_true_maximum_over_all_perfect_matchings():
    """Brute-force every perfect matching by hand and check `best_matching` finds the same max."""
    branches = deg(0, 5, 182, 178, curvature=(0.1, -0.1, -0.1, 0.1))
    pairing, _, total = best_matching(branches)

    def matchings(items):
        if not items:
            yield []
            return
        head, rest = items[0], items[1:]
        for k in range(len(rest)):
            for tail in matchings(rest[:k] + rest[k + 1 :]):
                yield [(head, rest[k]), *tail]

    best_brute = max(
        sum(C.pair_score(branches[i], branches[j])[0] for i, j in m)
        for m in matchings(list(range(4)))
    )
    assert total == pytest.approx(best_brute)


# -- odd number of branches ------------------------------------------------------------------


def test_three_branches_is_always_a_junction_regardless_of_angles():
    """Fewer than four branches cannot supply the two continuing pairs a crossing needs, so a
    three-armed site is a junction even when two of its arms are a perfect antiparallel pair
    (0 and 180 here, like two arms of a clean X) - one continuing pair is never enough, and the
    third arm is left unpaired rather than forcing a spurious second pair."""
    verdict = classify_junction(deg(0, 180, 90))
    assert not verdict.is_crossing
    assert verdict.kind == "junction"
    assert len(verdict.pairing) <= 1
    assert len(verdict.unpaired) >= 1


def test_five_branches_leaves_exactly_one_unpaired():
    verdict = classify_junction(deg(0, 90, 180, 270, 45))
    assert len(verdict.pairing) == 2
    assert len(verdict.unpaired) == 1
    paired = {i for pair in verdict.pairing for i in pair}
    assert paired | set(verdict.unpaired) == {0, 1, 2, 3, 4}


def test_a_four_way_merge_with_only_one_antiparallel_pair_is_a_junction():
    """Four branches where exactly one pair (0, 180) lines up and the other (60, 120) does
    not - a real 4-way merge that happens to have two co-linear arms, not two strokes
    crossing. A crossing needs *two* continuing pairs; this site supplies only one."""
    verdict = classify_junction(deg(0, 60, 120, 180))
    assert not verdict.is_crossing
    assert (0, 3) in verdict.pairing or (3, 0) in verdict.pairing


# -- deviation and pair_score in isolation ---------------------------------------------------


def test_deviation_is_zero_for_perfectly_antiparallel_branches():
    assert C.deviation(Branch(angle=0.0), Branch(angle=math.pi)) == pytest.approx(0.0, abs=1e-9)


def test_deviation_is_maximal_for_parallel_branches():
    assert C.deviation(Branch(angle=0.0), Branch(angle=0.0)) == pytest.approx(math.pi)


def test_pair_score_penalises_mismatched_stroke_width():
    thin = Branch(angle=0.0, width=2.0)
    thick_partner = Branch(angle=math.pi, width=2.0)
    mismatched_partner = Branch(angle=math.pi, width=20.0)
    matched_score, _ = C.pair_score(thin, thick_partner)
    mismatched_score, _ = C.pair_score(thin, mismatched_partner)
    assert mismatched_score < matched_score


def test_synthetic_cases_all_pass_as_defined_in_the_module():
    rows = []
    for name, (branches, expected) in C.synthetic().items():
        verdict = classify_junction(branches)
        rows.append((name, verdict.kind == expected))
    failed = [name for name, ok in rows if not ok]
    assert not failed, f"synthetic cases failed: {failed}"
