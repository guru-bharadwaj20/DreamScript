"""Phase 6.3.5 - which pages are support vectors, and whether that means what the plan expects."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import supportvectors as sv
from src.classify.data import Dataset


@pytest.fixture
def toy():
    """Imbalanced the way the corpus is, so the class-size effect is reproducible here."""
    rng = np.random.default_rng(60)
    y = np.array(["flowchart"] * 90 + ["wireframe"] * 90 + ["circuit"] * 15, dtype=object)
    centres = {"flowchart": (0.0, 0.0), "wireframe": (3.5, 0.0), "circuit": (1.75, 2.5)}
    X = np.array([centres[label] for label in y]) + rng.normal(0, 0.8, size=(195, 2))
    X = np.hstack([X, rng.normal(size=(195, 2))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(195)], dtype=object),
        ids=np.arange(195).astype(object),
        feature_names=["f0", "f1", "f2", "f3"],
        corpus="test",
    )


# -- the count ---------------------------------------------------------------------------------


def test_support_flags_mark_exactly_the_estimators_support_rows(toy):
    fitted = sv.fit_svc(toy, "linear")
    flags = sv.support_flags(fitted, toy)
    assert flags.sum() == len(fitted.named_steps["model"].support_)
    assert flags.dtype == bool
    assert len(flags) == len(toy.y)


def test_the_per_class_table_covers_every_class_and_sums_to_the_total(toy):
    fitted = sv.fit_svc(toy, "linear")
    shares = sv.by_class(fitted, toy)
    assert set(shares) == set(toy.classes)
    assert (
        sum(row["support_vectors"] for row in shares.values())
        == sv.support_flags(fitted, toy).sum()
    )
    for row in shares.values():
        assert row["share"] == pytest.approx(row["support_vectors"] / row["rows"], abs=1e-4)


def test_the_smallest_class_has_the_largest_support_vector_share(toy):
    """The class-size account: a 15-row class has almost no interior."""
    shares = sv.by_class(sv.fit_svc(toy, "linear"), toy)
    assert shares["circuit"]["share"] > shares["flowchart"]["share"]


# -- the margin -------------------------------------------------------------------------------------


def test_the_margin_width_is_two_over_the_weight_norm(toy):
    fitted = sv.fit_svc(toy, "linear")
    result = sv.margin_width(fitted)
    model = fitted.named_steps["model"]
    expected = 2.0 / np.linalg.norm(model.coef_, axis=1)
    assert result["problems"] == len(expected)
    assert result["mean_margin_width"] == pytest.approx(expected.mean(), abs=1e-4)
    assert result["min_margin_width"] <= result["max_margin_width"]


def test_the_margin_width_is_none_for_a_kernel_model(toy):
    """`w` is never formed in an RBF feature space; a substitute would be a different quantity."""
    assert sv.margin_width(sv.fit_svc(toy, "rbf")) is None


# -- the two competing accounts -------------------------------------------------------------------------


def test_no_row_outside_the_margin_is_misclassified_by_the_fitted_model(toy):
    """The geometric fact the whole finding rests on, checked on the fit rather than assumed."""
    fitted = sv.fit_svc(toy, "linear")
    flags = sv.support_flags(fitted, toy)
    wrong = fitted.predict(toy.X) != toy.y
    assert not (wrong & ~flags).any()


def test_the_ambiguity_check_controls_for_class(toy):
    """Support vectors and errors both concentrate in small classes, so the pooled number alone
    would largely be measuring class size twice."""
    result = sv.ambiguity(sv.fit_svc(toy, "linear"), toy, folds=3)
    assert set(result["within_class"]) == set(toy.classes)
    assert "pooled_error_rate_if_support_vector" in result
    for row in result["within_class"].values():
        assert row["support_vectors"] + row["others"] > 0


def test_the_ambiguity_check_counts_both_directions(toy):
    """Errors that are not support vectors, and support vectors that are not errors."""
    result = sv.ambiguity(sv.fit_svc(toy, "linear"), toy, folds=3)
    assert result["errors_that_are_support_vectors"] <= result["errors_total"]
    assert result["support_vectors_classified_correctly"] <= result["support_vectors_total"]


def test_being_a_support_vector_is_not_sufficient_for_being_an_error(toy):
    """The measured version on the corpus: 342 of 353 support vectors are classified correctly."""
    result = sv.ambiguity(sv.fit_svc(toy, "linear"), toy, folds=3)
    assert result["support_vectors_classified_correctly"] > result["errors_total"]


# -- the gallery --------------------------------------------------------------------------------------------


def test_the_gallery_returns_none_when_no_images_are_on_disk(toy, tmp_path, monkeypatch):
    """The analysis must not fail on a checkout without the raw corpus pulled."""
    monkeypatch.setattr(sv, "MANIFEST", tmp_path / "absent.parquet")
    assert sv.gallery(sv.fit_svc(toy, "linear"), toy, tmp_path / "g.png") is None


def test_the_image_paths_lookup_is_empty_without_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "MANIFEST", tmp_path / "absent.parquet")
    assert sv.image_paths(["anything"]) == {}


def test_the_gallery_contrasts_support_vectors_against_non_support_vectors():
    """A grid of support vectors alone cannot test the hypothesis it would be used to confirm."""
    import inspect

    source = inspect.getsource(sv.gallery)
    assert "not SV" in source
    assert "2 * len(classes)" in source


def test_a_missing_image_file_becomes_a_blank_rather_than_a_crash(tmp_path):
    thumbnail = sv._thumbnail(tmp_path / "nothing.png")
    assert thumbnail.shape[0] > 0
    assert not thumbnail.any()
