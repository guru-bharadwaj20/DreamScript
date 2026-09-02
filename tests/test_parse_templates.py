"""Phase 7.4.8 - the learned vocabulary against the hand-written one, and against the ceiling."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.descriptors import NAMES
from src.parse import templates

# -- the declared design -----------------------------------------------------------------------------


def test_the_three_predictors_include_a_ceiling():
    """Without the supervised row, 'learned beats template' could be noise between two failures."""
    assert set(templates.PREDICTORS) == {"template", "learned", "supervised"}


def test_the_collapse_map_only_merges_names_the_corpus_cannot_express():
    """hdbpmn has zero `ellipse` labels, so every such prediction is wrong by construction."""
    assert templates.COLLAPSE == {"ellipse": "circle"}


# -- scoring -----------------------------------------------------------------------------------------


def test_a_perfect_prediction_scores_one():
    truth = np.array(["circle", "diamond", "rectangle"], dtype=object)
    assert templates.score(truth.copy(), truth)["macro_f1"] == 1.0


def test_the_strict_score_punishes_an_unreachable_name():
    truth = np.array(["circle", "circle"], dtype=object)
    predicted = np.array(["ellipse", "ellipse"], dtype=object)
    assert templates.score(predicted, truth)["accuracy"] == 0.0


def test_collapsing_rescues_exactly_that_case():
    truth = np.array(["circle", "circle"], dtype=object)
    predicted = np.array(["ellipse", "ellipse"], dtype=object)
    assert templates.score(predicted, truth, collapsed=True)["accuracy"] == 1.0


def test_collapsing_does_not_change_a_prediction_that_never_says_ellipse():
    truth = np.array(["circle", "rectangle"], dtype=object)
    predicted = np.array(["circle", "circle"], dtype=object)
    strict = templates.score(predicted, truth)
    assert strict == templates.score(predicted, truth, collapsed=True)


def test_the_names_actually_predicted_are_reported():
    """A predictor that never emits a class is a finding, not a detail."""
    truth = np.array(["circle", "rectangle"], dtype=object)
    predicted = np.array(["circle", "circle"], dtype=object)
    assert templates.score(predicted, truth)["predicted_names"] == ["circle"]


def test_every_true_class_gets_an_f1_even_if_never_predicted():
    truth = np.array(["circle", "rectangle"], dtype=object)
    predicted = np.array(["circle", "circle"], dtype=object)
    per_class = templates.score(predicted, truth)["per_class_f1"]
    assert set(per_class) == {"circle", "rectangle"}
    assert per_class["rectangle"] == 0.0


def test_macro_f1_weights_a_rare_class_as_much_as_a_common_one():
    truth = np.array(["rectangle"] * 9 + ["circle"], dtype=object)
    predicted = np.array(["rectangle"] * 10, dtype=object)
    result = templates.score(predicted, truth)
    assert result["accuracy"] == 0.9
    assert result["macro_f1"] < 0.55


# -- the learned predictor ---------------------------------------------------------------------------


@pytest.fixture
def separable():
    """Three classes, well separated, several pages each."""
    rng = np.random.default_rng(0)
    X, labels, pages = [], [], []
    for i, name in enumerate(["circle", "diamond", "rectangle"]):
        centre = rng.normal(0, 10, size=len(NAMES)) + i * 40
        X.append(rng.normal(centre, 0.5, size=(60, len(NAMES))))
        labels += [name] * 60
        pages += [f"{name}_p{j // 10}" for j in range(60)]
    return np.vstack(X), np.array(labels, dtype=object), np.array(pages, dtype=object)


def test_the_learned_predictor_labels_every_row(separable):
    X, labels, pages = separable
    predicted = templates.learned_predictions(X, labels, pages, k=3, folds=3)
    assert len(predicted) == len(labels)
    assert all(p is not None for p in predicted)


def test_the_learned_predictor_only_emits_real_class_names(separable):
    X, labels, pages = separable
    predicted = templates.learned_predictions(X, labels, pages, k=3, folds=3)
    assert set(predicted) <= set(labels)


def test_a_separable_problem_is_recovered_by_the_learned_vocabulary(separable):
    """A floor: if naming cannot work on obvious clusters, the corpus number means nothing."""
    X, labels, pages = separable
    predicted = templates.learned_predictions(X, labels, pages, k=3, folds=3)
    assert templates.score(predicted, labels)["macro_f1"] > 0.95


def test_more_components_than_classes_still_produces_valid_names(separable):
    """With k > number of classes the surplus components must still be named."""
    X, labels, pages = separable
    predicted = templates.learned_predictions(X, labels, pages, k=6, folds=3)
    assert set(predicted) <= set(labels)
    assert all(p is not None for p in predicted)


def test_the_naming_never_sees_the_held_out_labels(separable):
    """The mapping is built on the training fold only.

    Corrupting one fold's labels must leave the predictions made *for that fold* unchanged: an
    implementation that named its components using test labels would track the corruption.
    """
    from sklearn.model_selection import GroupKFold

    X, labels, pages = separable
    train, test = next(iter(GroupKFold(3).split(X, labels, groups=pages)))

    corrupted = labels.copy()
    rng = np.random.default_rng(1)
    corrupted[test] = rng.permutation(corrupted[test])

    clean = templates.learned_predictions(X, labels, pages, k=3, folds=3)
    dirty = templates.learned_predictions(X, corrupted, pages, k=3, folds=3)
    assert list(clean[test]) == list(dirty[test])


def test_folds_are_grouped_so_a_page_never_spans_the_split(separable):
    """7.4.7 showed a scribe is recoverable from geometry; a node-level split would leak."""
    from sklearn.model_selection import GroupKFold

    X, labels, pages = separable
    for train, test in GroupKFold(3).split(X, labels, groups=pages):
        assert not (set(pages[train]) & set(pages[test]))
