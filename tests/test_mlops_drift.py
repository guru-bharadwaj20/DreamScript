"""Phase 15.5 - a drift alarm is only worth having if it is silent on nothing.

Anyone can write a detector that fires. The properties worth pinning are the ones that make a
firing mean something: zero on identical inputs, quiet on two halves of the same distribution,
loud on a shift that really happened, and not fireable by a single noisy column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mlops import drift


def table(values: dict[str, np.ndarray], **identity) -> pd.DataFrame:
    frame = pd.DataFrame(values)
    frame.insert(0, "id", [f"p/{i}" for i in range(len(frame))])
    frame["source"] = identity.get("source", "hdbpmn")
    frame["diagram_type"] = identity.get("diagram_type", "flowchart")
    frame["split"] = identity.get("split", "train")
    frame["scribe_id"] = None
    frame["adverse"] = False
    frame["synthetic"] = False
    return frame


def test_psi_of_a_distribution_against_itself_is_zero():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=2000)
    assert drift.psi(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=4000)
    small = drift.psi(reference, rng.normal(loc=0.2, size=4000))
    large = drift.psi(reference, rng.normal(loc=2.0, size=4000))
    assert 0 <= small < large
    assert large >= drift.PSI_SIGNIFICANT


def test_psi_is_binned_on_the_reference_not_the_combined_sample():
    """Combined-sample bins move toward the shift and understate it; this pins that they don't."""
    rng = np.random.default_rng(1)
    reference = rng.normal(size=4000)
    shifted = rng.normal(loc=3.0, size=4000)
    # Reference-binned PSI must see a large shift. A combined-quantile implementation would put
    # bin edges between the two modes and report far less.
    assert drift.psi(reference, shifted) > 1.0


def test_a_constant_feature_reports_no_drift_rather_than_a_degenerate_number():
    constant = np.ones(500)
    assert drift.psi(constant, np.ones(500)) == 0.0
    # Even when the constant itself moves, the reference has no distribution to bin.
    assert drift.psi(constant, np.full(500, 7.0)) == 0.0


def test_psi_is_finite_when_a_bin_is_empty_on_one_side():
    """Without the epsilon floor this is a division by zero and a log of zero."""
    rng = np.random.default_rng(2)
    reference = rng.normal(size=1000)
    disjoint = rng.normal(loc=50.0, scale=0.01, size=1000)
    value = drift.psi(reference, disjoint)
    assert np.isfinite(value)
    assert value > drift.PSI_SIGNIFICANT


def test_bands_follow_the_stated_thresholds():
    assert drift.band(0.0) == "none"
    assert drift.band(drift.PSI_MODERATE - 1e-9) == "none"
    assert drift.band(drift.PSI_MODERATE) == "moderate"
    assert drift.band(drift.PSI_SIGNIFICANT - 1e-9) == "moderate"
    assert drift.band(drift.PSI_SIGNIFICANT) == "significant"
    assert drift.band(float("nan")) == "undefined"


def test_one_noisy_feature_cannot_fire_the_alarm():
    """The whole point of MIN_DRIFTED_FEATURES: a distribution is not one column."""
    rng = np.random.default_rng(3)
    n = 1500
    reference = table({f"f{i}": rng.normal(size=n) for i in range(6)})
    current = table({f"f{i}": rng.normal(size=n) for i in range(6)})
    current["f0"] = rng.normal(loc=6.0, size=n)  # one column moved hard
    result = drift.compare(reference, current)
    assert result["drifted"] < drift.MIN_DRIFTED_FEATURES
    assert result["alarm"] is False


def test_enough_drifted_features_do_fire_it():
    rng = np.random.default_rng(4)
    n = 1500
    reference = table({f"f{i}": rng.normal(size=n) for i in range(6)})
    current = table({f"f{i}": rng.normal(loc=6.0, size=n) for i in range(6)})
    result = drift.compare(reference, current)
    assert result["drifted"] >= drift.MIN_DRIFTED_FEATURES
    assert result["alarm"] is True


def test_identity_columns_are_never_scored_for_drift():
    """`source` moving between batches is not drift, it is bookkeeping."""
    rng = np.random.default_rng(5)
    frame = table({"f0": rng.normal(size=100)})
    columns = drift.feature_columns(frame)
    for banned in drift.IDENTITY:
        assert banned not in columns
    assert columns == ["f0"]


def test_ks_agrees_with_psi_about_an_obvious_shift():
    rng = np.random.default_rng(6)
    reference = rng.normal(size=2000)
    statistic, pvalue = drift.ks(reference, rng.normal(loc=3.0, size=2000))
    assert statistic > 0.5
    assert pvalue < 1e-10
    same_stat, same_p = drift.ks(reference, rng.normal(size=2000))
    assert same_stat < statistic
    assert same_p > 1e-10


def test_the_null_control_is_quiet_on_real_data():
    """Two random halves of the reference must not alarm, or nothing else here is readable."""
    if not drift.TABLE.is_file():
        pytest.skip("needs data/features/handcrafted.parquet")
    result = drift.arm_null(drift.load())
    assert result["alarm"] is False, result["worst"][:3]
    assert result["passes"] is True


def test_the_self_control_reports_zero_on_real_data():
    if not drift.TABLE.is_file():
        pytest.skip("needs data/features/handcrafted.parquet")
    result = drift.arm_self(drift.load())
    assert result["max_psi"] == pytest.approx(0.0, abs=1e-9)
    assert result["passes"] is True


def test_a_shift_the_corpus_really_contains_is_detected():
    """A new diagram type is the plan's own example, and this corpus has one."""
    if not drift.TABLE.is_file():
        pytest.skip("needs data/features/handcrafted.parquet")
    result = drift.arm_new_diagram_type(drift.load())
    assert result["alarm"] is True
    assert result["passes"] is True


def test_empty_input_is_undefined_rather_than_zero():
    """An empty batch must not read as 'no drift' - that would silence the alarm on no data."""
    assert not np.isfinite(drift.psi(np.array([]), np.array([1.0, 2.0])))
    assert not np.isfinite(drift.psi(np.array([1.0, 2.0]), np.array([])))
