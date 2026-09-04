"""Phase 7.1.6 - four damage models, and the point in the pipeline they are applied at."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import occlusion


@pytest.fixture
def page():
    """A light page with dark strokes on it - enough structure for the masks to destroy."""
    rng = np.random.default_rng(68)
    image = np.full((600, 800), 235, dtype=np.uint8)
    image += rng.normal(0, 4, size=image.shape).astype(np.int16).clip(-20, 20).astype(np.uint8)
    for y in (150, 300, 450):
        image[y : y + 6, 100:700] = 30
    for x in (200, 400, 600):
        image[100:500, x : x + 6] = 30
    return image


# -- the four the plan names ---------------------------------------------------------------------


def test_the_four_damage_models_the_plan_names_are_implemented():
    assert set(occlusion.KINDS) == {"coffee_stain", "torn_corner", "finger", "crop"}


def test_the_clean_control_is_in_the_severity_grid():
    """A robustness curve without its own baseline cannot be read."""
    assert 0.0 in occlusion.SEVERITIES


def test_severity_zero_returns_the_page_unchanged(page):
    for kind in occlusion.KINDS:
        assert np.array_equal(occlusion.occlude(page, kind, 0.0), page)


def test_an_unknown_kind_is_refused_by_name(page):
    with pytest.raises(ValueError, match="kind must be one of"):
        occlusion.occlude(page, "scribble", 0.2)


@pytest.mark.parametrize("kind", occlusion.KINDS)
def test_every_damage_model_changes_the_page(page, kind):
    damaged = occlusion.occlude(page, kind, 0.3)
    assert damaged.shape != page.shape or not np.array_equal(damaged, page)


@pytest.mark.parametrize("kind", occlusion.KINDS)
def test_every_damage_model_is_monotone_in_severity(page, kind):
    """More severity must destroy more; a mask that saturates would flatten the curve."""

    def changed(severity):
        damaged = occlusion.occlude(page, kind, severity)
        if damaged.shape != page.shape:
            return page.size - damaged.size
        return int((damaged != page).sum())

    assert changed(0.4) > changed(0.05)


# -- each model damages in its own way, which is why there are four --------------------------------------


def test_a_coffee_stain_darkens_without_removing_the_page(page):
    damaged = occlusion.occlude(page, "coffee_stain", 0.3)
    assert damaged.shape == page.shape
    assert damaged.mean() < page.mean()


def test_a_stain_leaves_ink_visible_underneath(page):
    """It is multiplicative, which is the whole difference from the opaque finger."""
    stained = occlusion.occlude(page, "coffee_stain", 0.5)
    finger = occlusion.occlude(page, "finger", 0.5)
    assert stained.std() > finger.std() * 0.5


def test_a_torn_corner_is_page_white_not_black(page):
    """Filling it dark would add ink where the tear removed content."""
    damaged = occlusion.occlude(page, "torn_corner", 0.3)
    assert damaged.mean() > page.mean()


def test_a_torn_corner_touches_a_corner(page):
    damaged = occlusion.occlude(page, "torn_corner", 0.3)
    corners = [damaged[0, 0], damaged[0, -1], damaged[-1, 0], damaged[-1, -1]]
    assert max(corners) >= np.percentile(page, 95) - 1


def test_a_finger_is_opaque_and_dark(page):
    damaged = occlusion.occlude(page, "finger", 0.4)
    assert damaged.min() <= 60
    assert damaged.mean() < page.mean()


def test_a_crop_shrinks_the_page(page):
    damaged = occlusion.occlude(page, "crop", 0.5)
    assert damaged.size < page.size
    assert damaged.shape[0] < page.shape[0]


def test_a_crop_keeps_roughly_the_requested_area(page):
    damaged = occlusion.occlude(page, "crop", 0.5)
    assert 0.4 < damaged.size / page.size < 0.6


def test_the_damage_is_deterministic_for_a_given_seed(page):
    a = occlusion.occlude(page, "finger", 0.3, seed=7)
    b = occlusion.occlude(page, "finger", 0.3, seed=7)
    assert np.array_equal(a, b)


def test_different_seeds_place_the_damage_differently(page):
    a = occlusion.occlude(page, "coffee_stain", 0.3, seed=1)
    b = occlusion.occlude(page, "coffee_stain", 0.3, seed=2)
    assert not np.array_equal(a, b)


# -- where in the pipeline it is applied -------------------------------------------------------------------


def test_the_occlusion_runs_before_illumination_correction(page):
    """Occluding the normalised array would measure a perturbation the pipeline never saw."""
    import inspect

    source = inspect.getsource(occlusion.page_array_from)
    assert "illumination.correct" in source
    run_source = inspect.getsource(occlusion.embed_variants)
    assert "page_array_from(occlude(" in run_source


def test_the_page_array_matches_6_1_1s_geometry(page):
    from src.embed.backbone import SIZE

    array = occlusion.page_array_from(page, "gray")
    assert array.shape == (SIZE, SIZE)
    assert array.dtype == np.float32
    assert array.min() >= 0.0 and array.max() <= 1.0


def test_an_unknown_mode_is_refused(page):
    with pytest.raises(ValueError, match="input mode must be"):
        occlusion.page_array_from(page, "rgb")


# -- reading the curve ------------------------------------------------------------------------------------


def test_the_half_life_is_the_first_severity_below_half_the_clean_score():
    series = [
        {"severity": 0.0, "macro_f1": 0.90},
        {"severity": 0.1, "macro_f1": 0.80},
        {"severity": 0.2, "macro_f1": 0.40},
        {"severity": 0.5, "macro_f1": 0.20},
    ]
    assert occlusion.half_life(series, 0.90) == 0.2


def test_a_model_that_never_halves_reports_none():
    """Which is itself a result, and better than an invented ceiling."""
    series = [{"severity": s, "macro_f1": 0.9} for s in (0.0, 0.1, 0.5)]
    assert occlusion.half_life(series, 0.9) is None


def test_the_sample_is_stratified(monkeypatch):
    from src.classify.data import Dataset

    y = np.array(["a"] * 100 + ["b"] * 100 + ["c"] * 20, dtype=object)
    data = Dataset(
        X=np.zeros((220, 3)),
        y=y,
        groups=np.arange(220).astype(object),
        ids=np.arange(220).astype(object),
        feature_names=["a", "b", "c"],
        corpus="test",
    )
    index = occlusion.sample_pages(data, limit=110)
    assert len(index) == 110
    assert set(y[index]) == {"a", "b", "c"}


def test_a_limit_above_the_corpus_returns_every_row():
    from src.classify.data import Dataset

    data = Dataset(
        X=np.zeros((10, 2)),
        y=np.array(["a"] * 10, dtype=object),
        groups=np.arange(10).astype(object),
        ids=np.arange(10).astype(object),
        feature_names=["a", "b"],
        corpus="test",
    )
    assert len(occlusion.sample_pages(data, limit=999)) == 10
