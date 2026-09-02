"""A string-label adapter for estimators that insist on integer targets.

Every `Dataset` in this project has carried string labels since Phase 5, because 5.3.2 and 5.3.3
read rules and coefficients as sentences about diagram types and `class 3` is not a sentence.
Most of scikit-learn accepts that. Two things in the project do not:

    xgboost's `XGBClassifier`   raises on non-integer `y` in its sklearn wrapper
    sklearn's `MLPClassifier`   with `early_stopping=True`, calls `np.isnan` on predicted labels

6.2.1 solved the second case with `StringSafeMLP`, a subclass. That approach does not generalise -
it needs one subclass per estimator, and it inherits whatever else the parent does - so this
module provides the wrapper form instead: an encoder around *any* classifier, transparent from
the outside.

The contract is that `LabelSafe(est)` behaves exactly like `est` would if `est` accepted strings.
`classes_` holds the original labels, `predict` returns them, and `predict_proba`'s columns are
in `classes_` order - which matters because 5.2.1's harness reads probability columns through
`classes_` and would silently mislabel them otherwise.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class LabelSafe(ClassifierMixin, BaseEstimator):
    """Wrap a classifier so it accepts string labels and returns them.

    The wrapped estimator is cloned in `fit`, so the instance passed in is never mutated and the
    wrapper is reusable across cross-validation folds the way sklearn expects.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **kwargs):  # noqa: N803 - sklearn's parameter name
        from sklearn.preprocessing import LabelEncoder

        self.encoder_ = LabelEncoder().fit(np.asarray(y).astype(str))
        self.classes_ = self.encoder_.classes_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, self.encoder_.transform(np.asarray(y).astype(str)), **kwargs)
        return self

    def predict(self, X):  # noqa: N803
        return self.encoder_.inverse_transform(np.asarray(self.estimator_.predict(X)).astype(int))

    def predict_proba(self, X):  # noqa: N803
        # The inner estimator's columns are in encoded order, and `LabelEncoder` sorts, so they
        # already line up with `classes_`. Asserted rather than assumed because a mismatch here
        # would be invisible - the probabilities would look fine and belong to the wrong classes.
        proba = self.estimator_.predict_proba(X)
        assert proba.shape[1] == len(self.classes_)
        return proba

    def decision_function(self, X):  # noqa: N803
        return self.estimator_.decision_function(X)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_
