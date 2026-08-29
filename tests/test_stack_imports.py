"""Phase 0.1.4 acceptance test — the classical ML stack imports *and* runs.

An import alone does not prove a wheel works (xgboost/lightgbm ship native binaries that
fail at first call, not at import), so each library is exercised on a tiny synthetic task.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    return x, y


def test_sklearn_trains(toy):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    x, y = toy
    scores = cross_val_score(RandomForestClassifier(n_estimators=25, random_state=0), x, y, cv=3)
    assert scores.mean() > 0.7


def test_xgboost_trains(toy):
    import xgboost as xgb

    x, y = toy
    clf = xgb.XGBClassifier(n_estimators=20, max_depth=3, verbosity=0)
    clf.fit(x, y)
    assert clf.score(x, y) > 0.8


def test_lightgbm_trains(toy):
    import lightgbm as lgb

    x, y = toy
    clf = lgb.LGBMClassifier(n_estimators=20, verbose=-1)
    clf.fit(x, y)
    assert clf.score(x, y) > 0.8


def test_hmmlearn_fits_and_decodes():
    """Phase 7.3 depends on Viterbi decoding, so decode a 2-state sequence here."""
    from hmmlearn import hmm

    model = hmm.CategoricalHMM(n_components=2, random_state=0, n_iter=20)
    obs = np.array([[0], [1], [0], [1], [2], [2], [1], [0]] * 8)
    model.fit(obs)
    logprob, states = model.decode(obs, algorithm="viterbi")
    assert np.isfinite(logprob)
    assert len(states) == len(obs)
    assert set(np.unique(states)) <= {0, 1}


def test_imblearn_resamples(toy):
    from imblearn.over_sampling import SMOTE

    x, y = toy
    x_imb, y_imb = x[:150], y[:150]
    x_res, y_res = SMOTE(random_state=0).fit_resample(x_imb, y_imb)
    counts = np.bincount(y_res)
    assert counts[0] == counts[1]
    assert len(x_res) >= len(x_imb)
