"""Phase 8.6 - clustering scribes by how they write."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cluster import styles

# -- the declared design ---------------------------------------------------------------------------


def test_the_sweep_reaches_the_three_clusters_the_plan_requires():
    assert max(styles.KS) >= 3


def test_all_four_statistics_the_plan_names_are_computed():
    for name in ("slant", "thickness", "curvature", "spacing"):
        assert name in styles.NAMES


def test_nothing_in_the_vector_carries_the_size_of_the_photograph():
    """hdbpmn's writers did not share a camera; an absolute length would cluster the cameras."""
    assert "width" not in styles.NAMES
    assert "ink_pixels" not in styles.NAMES


# -- slant -----------------------------------------------------------------------------------------


def _bars(angle_degrees: float, size: int = 120) -> np.ndarray:
    """A field of parallel strokes leaning at a known angle."""
    mask = np.zeros((size, size), dtype=bool)
    shift = np.tan(np.radians(angle_degrees))
    for x0 in range(10, size - 10, 15):
        for y in range(15, size - 15):
            x = int(x0 + shift * (size / 2 - y))
            if 0 <= x < size:
                mask[y, max(0, x - 1) : x + 2] = True
    return mask


def test_upright_strokes_measure_near_zero_slant():
    assert abs(styles.slant(_bars(0.0))) <= 3


def test_a_leaning_hand_measures_a_slant_of_the_right_sign():
    right = styles.slant(_bars(20.0))
    left = styles.slant(_bars(-20.0))
    assert right > 5
    assert left < -5
    assert right > left


def test_slant_is_insensitive_to_stroke_thickness():
    """The reason this estimator was chosen over a gradient histogram."""
    import cv2

    thin = _bars(15.0)
    thick = cv2.dilate(thin.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    assert abs(styles.slant(thin) - styles.slant(thick)) <= 6


# -- thickness -------------------------------------------------------------------------------------


def test_a_thicker_pen_measures_thicker():
    thin = np.zeros((60, 60), dtype=bool)
    thin[20:40, 29:31] = True
    thick = np.zeros((60, 60), dtype=bool)
    thick[20:40, 25:35] = True
    assert styles.thickness(thick) > styles.thickness(thin)


def test_thickness_of_an_empty_mask_is_zero_rather_than_a_crash():
    assert styles.thickness(np.zeros((20, 20), dtype=bool)) == 0.0


# -- curvature -------------------------------------------------------------------------------------


def test_a_circle_turns_more_per_unit_length_than_a_long_rectangle():
    import cv2

    circle = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(circle, (60, 60), 20, 1, 2)
    box = np.zeros((120, 120), dtype=np.uint8)
    cv2.rectangle(box, (10, 55), (110, 65), 1, 2)
    assert styles.curvature(circle.astype(bool)) > styles.curvature(box.astype(bool))


def test_curvature_of_an_empty_mask_is_zero():
    assert styles.curvature(np.zeros((30, 30), dtype=bool)) == 0.0


# -- component statistics --------------------------------------------------------------------------


def _blobs(positions, size=(200, 200), box=6):
    mask = np.zeros(size, dtype=bool)
    for y, x in positions:
        mask[y : y + box, x : x + box] = True
    return mask


def test_too_few_components_returns_nothing_rather_than_a_number():
    """Three blobs cannot support a median spacing; a value here would be noise with a name."""
    assert styles.component_stats(_blobs([(10, 10), (10, 40)])) == {}


def test_wider_spacing_measures_wider():
    close = _blobs([(10, x) for x in range(10, 90, 12)])
    far = _blobs([(10, x) for x in range(10, 180, 30)])
    assert styles.component_stats(far)["spacing"] > styles.component_stats(close)["spacing"]


def test_spacing_is_relative_to_component_height_not_to_the_page():
    """A photograph taken twice as close must not read as twice the spacing."""
    import cv2

    small = _blobs([(10, x) for x in range(10, 90, 12)], size=(200, 200), box=6)
    big = cv2.resize(small.astype(np.uint8), (400, 400), interpolation=cv2.INTER_NEAREST).astype(
        bool
    )
    assert styles.component_stats(small)["spacing"] == pytest.approx(
        styles.component_stats(big)["spacing"], rel=0.25
    )


def test_density_is_not_identically_one():
    """Dividing ink by the components' own pixel counts would be 1.0 for every page."""
    mask = _blobs([(10, x) for x in range(10, 90, 12)])
    assert 0.0 < styles.component_stats(mask)["density"] <= 1.0


# -- the controls ----------------------------------------------------------------------------------


def _table(scribes, exercises, n=8):
    rng = np.random.default_rng(0)
    rows = []
    for scribe, exercise in zip(scribes, exercises, strict=True):
        rows.append(
            {
                "page": f"{exercise}_{scribe}",
                "scribe": scribe,
                "exercise": exercise,
                **{name: float(rng.normal(hash(scribe) % 7, 0.05)) for name in styles.NAMES},
            }
        )
    return pd.DataFrame(rows)


def test_the_controls_report_both_readings_of_page_agreement():
    """All-or-nothing agreement fails a ten-page writer on one stray; the modal share does not."""
    scribes = [f"w{i // 4}" for i in range(24)]
    table = _table(scribes, ["ex0"] * 24)
    X, names, _ = styles.per_scribe(table)
    assignment = styles.cluster(X, 3).named_steps["kmeans"].labels_
    result = styles.controls(table, names, assignment)
    assert "pages_agree_share" in result
    assert "pages_modal_share" in result


def test_both_page_agreement_readings_carry_their_chance_rate():
    """0.28 is a failure against a chance rate of 0.33 and a success against one of 0.016."""
    scribes = [f"w{i // 4}" for i in range(24)]
    table = _table(scribes, ["ex0"] * 24)
    X, names, _ = styles.per_scribe(table)
    assignment = styles.cluster(X, 3).named_steps["kmeans"].labels_
    result = styles.controls(table, names, assignment)
    assert result["pages_modal_share_at_chance"] == pytest.approx(1 / 3, abs=1e-4)
    assert 0.0 < result["pages_agree_share_at_chance"] < result["pages_modal_share_at_chance"]


def test_a_scribe_row_is_the_median_of_their_pages():
    """One badly lit photograph must not move a writer; the mean would let it."""
    table = pd.DataFrame(
        [
            {"page": f"p{i}", "scribe": "w0", "exercise": "ex0", **dict.fromkeys(styles.NAMES, 1.0)}
            for i in range(4)
        ]
        + [{"page": "p9", "scribe": "w0", "exercise": "ex0", **dict.fromkeys(styles.NAMES, 99.0)}]
    )
    X, names, counts = styles.per_scribe(table)
    assert list(names) == ["w0"]
    assert X[0, 0] == 1.0
    assert counts[0] == 5


# -- the clustering --------------------------------------------------------------------------------


def test_the_scaler_travels_with_the_style_model():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(30, len(styles.NAMES)))
    model = styles.cluster(X, 3)
    assert "scale" in model.named_steps


def test_the_sweep_carries_the_smallest_cluster():
    """8.7 has to fit a model inside each cluster; a one-scribe cluster cannot support one."""
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, size=(40, len(styles.NAMES)))
    for row in styles.sweep(X, ks=(2, 3)):
        assert "smallest" in row
        assert sum(row["sizes"]) == len(X)
