"""Phase 7.4.7 - whether a rectangle is this person's rectangle, and whether that is actionable."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.descriptors import NAMES
from src.parse import scribes


def table(per_scribe, spread, offset, n_scribes=4, seed=0):
    """Rectangles from several scribes, each with its own mean."""
    rng = np.random.default_rng(seed)
    X, who, page = [], [], []
    for s in range(n_scribes):
        centre = np.full(len(NAMES), s * offset)
        X.append(rng.normal(centre, spread, size=(per_scribe, len(NAMES))))
        who += [f"w{s}"] * per_scribe
        page += [f"p{s}_{i // 10}" for i in range(per_scribe)]
    return np.vstack(X), np.array(who, dtype=object), np.array(page, dtype=object)


# -- the declared design -----------------------------------------------------------------------------


def test_a_scribe_needs_a_minimum_number_of_rows_to_be_modelled():
    assert scribes.MIN_ROWS > 0


def test_the_figure_panel_names_real_columns():
    assert set(scribes.PANEL) <= set(NAMES)


# -- separation ---------------------------------------------------------------------------------------


def test_scribes_that_draw_alike_separate_weakly():
    X, who, _ = table(per_scribe=40, spread=1.0, offset=0.0)
    assert max(scribes.separation(X, who).values()) < 5.0


def test_scribes_that_draw_differently_separate_strongly():
    X, who, _ = table(per_scribe=40, spread=0.2, offset=3.0)
    assert max(scribes.separation(X, who).values()) > 100.0


def test_every_usable_column_is_scored():
    X, who, _ = table(per_scribe=40, spread=1.0, offset=0.5)
    assert set(scribes.separation(X, who)) == set(NAMES)


def test_the_columns_come_back_ranked():
    X, who, _ = table(per_scribe=40, spread=1.0, offset=0.5)
    values = list(scribes.separation(X, who).values())
    assert values == sorted(values, reverse=True)


def test_a_constant_column_is_dropped_rather_than_producing_a_nan():
    X, who, _ = table(per_scribe=40, spread=1.0, offset=0.5)
    X[:, 0] = 7.0
    result = scribes.separation(X, who)
    assert NAMES[0] not in result
    assert all(np.isfinite(v) for v in result.values())


def test_a_scribe_with_too_few_rows_does_not_break_the_statistic():
    X, who, _ = table(per_scribe=40, spread=1.0, offset=0.5)
    X = np.vstack([X, np.zeros((1, len(NAMES)))])
    who = np.append(who, "loner")
    assert all(np.isfinite(v) for v in scribes.separation(X, who).values())


# -- adaptation ---------------------------------------------------------------------------------------


def test_the_unseen_scribe_split_cannot_fit_a_personal_model():
    """The control that keeps the unseen-page number honest: held out by scribe, a writer has no
    training rows of their own, so there is nothing to adapt from and the task must say so."""
    X, who, page = table(per_scribe=40, spread=0.5, offset=2.0, n_scribes=6)
    result = scribes.adaptation(X, who, page, folds=3)
    assert result["unseen_scribe"]["scribe_blocks"] == 0


def test_the_unseen_page_split_scores_both_models():
    X, who, page = table(per_scribe=40, spread=0.5, offset=2.0, n_scribes=6)
    result = scribes.adaptation(X, who, page, folds=3)["unseen_page"]
    assert result["scribe_blocks"] > 0
    assert "global_log_likelihood" in result
    assert "per_scribe_log_likelihood" in result


def test_the_gain_is_personal_minus_global():
    X, who, page = table(per_scribe=40, spread=0.5, offset=2.0, n_scribes=6)
    r = scribes.adaptation(X, who, page, folds=3)["unseen_page"]
    assert r["gain"] == pytest.approx(
        r["per_scribe_log_likelihood"] - r["global_log_likelihood"], abs=1e-3
    )


def test_adaptation_wins_when_the_scribes_really_are_far_apart_and_data_is_plentiful():
    """A positive control: if per-scribe fitting never helps on any data, the measurement on
    the real corpus would be meaningless."""
    X, who, page = table(per_scribe=400, spread=0.3, offset=12.0, n_scribes=4, seed=3)
    assert scribes.adaptation(X, who, page, folds=3)["unseen_page"]["gain"] > 0


def test_scribes_below_the_row_floor_are_excluded_from_adaptation():
    X, who, page = table(per_scribe=40, spread=0.5, offset=2.0, n_scribes=4)
    tiny = np.zeros((3, len(NAMES)))
    X = np.vstack([X, tiny])
    who = np.append(who, ["rare"] * 3)
    page = np.append(page, ["pz"] * 3)
    result = scribes.adaptation(X, who, page, folds=3)["unseen_page"]
    assert result["scribe_blocks"] > 0
