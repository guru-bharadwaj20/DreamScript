"""Phase 7.2 - the string-label adapter, and the column order it must not scramble."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.classify.labels import LabelSafe


@pytest.fixture
def toy():
    rng = np.random.default_rng(7)
    X = np.vstack([rng.normal(0, 1, size=(30, 3)), rng.normal(3, 1, size=(30, 3))])
    y = np.array(["flowchart"] * 30 + ["state-machine"] * 30, dtype=object)
    return X, y


def test_it_returns_the_strings_it_was_given(toy):
    X, y = toy
    predicted = LabelSafe(LogisticRegression(max_iter=500)).fit(X, y).predict(X)
    assert set(predicted) <= {"flowchart", "state-machine"}


def test_classes_are_the_original_labels_not_codes(toy):
    X, y = toy
    assert list(LabelSafe(LogisticRegression(max_iter=500)).fit(X, y).classes_) == [
        "flowchart",
        "state-machine",
    ]


def test_the_inner_estimator_actually_saw_integers(toy):
    """The whole point of the wrapper; if it forwarded strings xgboost would still raise."""
    X, y = toy
    wrapper = LabelSafe(LogisticRegression(max_iter=500)).fit(X, y)
    assert np.issubdtype(wrapper.estimator_.classes_.dtype, np.integer)


def test_the_wrapped_instance_is_never_mutated(toy):
    """`clone` in fit is what makes the wrapper safe to reuse across CV folds."""
    X, y = toy
    inner = LogisticRegression(max_iter=500)
    LabelSafe(inner).fit(X, y)
    assert not hasattr(inner, "classes_")


def test_probability_columns_line_up_with_classes(toy):
    """A silent mismatch here would mislabel every probability 5.2.1 reads."""
    X, y = toy
    wrapper = LabelSafe(LogisticRegression(max_iter=500)).fit(X, y)
    proba = wrapper.predict_proba(X)
    assert proba.shape[1] == len(wrapper.classes_)
    argmax = wrapper.classes_[proba.argmax(axis=1)]
    assert list(argmax) == list(wrapper.predict(X))


def test_probabilities_sum_to_one(toy):
    X, y = toy
    proba = LabelSafe(LogisticRegression(max_iter=500)).fit(X, y).predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_decision_function_is_forwarded(toy):
    X, y = toy
    assert LabelSafe(LogisticRegression(max_iter=500)).fit(X, y).decision_function(X).shape == (60,)


def test_feature_importances_are_forwarded(toy):
    X, y = toy
    wrapper = LabelSafe(RandomForestClassifier(n_estimators=8, random_state=0)).fit(X, y)
    assert wrapper.feature_importances_.shape == (3,)


def test_three_classes_round_trip(toy):
    X, _ = toy
    y = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20, dtype=object)
    wrapper = LabelSafe(LogisticRegression(max_iter=500)).fit(X, y)
    assert list(wrapper.classes_) == ["a", "b", "c"]
    assert wrapper.predict_proba(X).shape == (60, 3)
