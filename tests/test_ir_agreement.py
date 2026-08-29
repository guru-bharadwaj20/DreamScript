"""Phase 2.2.4 - agreement measurement.

The numbers themselves live in `reports/annotator_agreement.md`; these tests protect the
machinery that produces them, and one factual claim the report rests on.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ir import agreement


def test_kappa_of_perfect_agreement_is_one():
    labels = ["circle", "diamond", "rectangle", "circle"]
    assert agreement.kappa(labels, labels)["kappa"] == pytest.approx(1.0)


def test_kappa_of_chance_agreement_is_about_zero():
    rng = np.random.default_rng(0)
    pool = ["circle", "diamond", "rectangle"]
    a = list(rng.choice(pool, 600))
    b = list(rng.choice(pool, 600))
    assert abs(agreement.kappa(a, b)["kappa"]) < 0.1


def test_kappa_handles_a_single_label_without_dividing_by_zero():
    result = agreement.kappa(["circle"] * 5, ["circle"] * 5)
    assert result["agreement"] == 1.0


def test_every_shape_maps_to_a_role_for_labeller_b():
    from src.ir.vocab import ROLES, SHAPES

    assert set(agreement.ROLE_FROM_SHAPE) == set(SHAPES)
    assert set(agreement.ROLE_FROM_SHAPE.values()) <= set(ROLES)


def test_sampling_is_deterministic():
    first = [d.id for d in agreement.sample_diagrams(6)]
    second = [d.id for d in agreement.sample_diagrams(6)]
    if not first:
        pytest.skip("hdBPMN not present")
    assert first == second


def test_the_sample_is_split_into_calibration_and_evaluation():
    """B's thresholds were tuned on one half; the reported kappa must come from the other."""
    diagrams = agreement.sample_diagrams(6)
    if not diagrams:
        pytest.skip("hdBPMN not present")
    rows = agreement.crops(diagrams)
    halves = {r["half"] for r in rows}
    assert halves == {"calibration", "evaluation"}


def test_geometry_labeller_never_claims_certainty():
    diagrams = agreement.sample_diagrams(4)
    if not diagrams:
        pytest.skip("hdBPMN not present")
    rows = agreement.crops(diagrams)
    agreement.label_with_geometry(rows)
    assert rows
    assert all(r["b_confidence"] < 1.0 for r in rows)
