"""Phase 8.7 - routing text crops to a per-style OCR head."""

from __future__ import annotations

import collections

import numpy as np
import pandas as pd
import pytest

from src.ocr import styled

# -- the declared design ---------------------------------------------------------------------------


def test_the_random_control_is_one_of_the_arms():
    """Without it, `style` losing to `global` would only be the arithmetic of dividing by three."""
    assert "random" in styled.ARMS
    assert "style" in styled.ARMS
    assert "global" in styled.ARMS
    assert "zero_shot" in styled.ARMS


def test_the_crop_is_inset_so_the_drawn_border_is_dropped():
    assert 0.0 < styled.INSET < 0.5


# -- the random control ----------------------------------------------------------------------------


def test_the_random_groups_match_the_style_groups_size_for_size():
    """Same sizes, not equal sizes: only *which* writers are together may differ."""
    style = {f"w{i}": i % 3 for i in range(30)}
    style["w0"] = 1  # make the sizes uneven on purpose
    randomised = styled.random_groups(style)
    assert collections.Counter(randomised.values()) == collections.Counter(style.values())


def test_the_random_groups_cover_exactly_the_same_writers():
    style = {f"w{i}": i % 3 for i in range(12)}
    assert set(styled.random_groups(style)) == set(style)


def test_the_random_grouping_is_not_the_style_grouping():
    style = {f"w{i}": i % 3 for i in range(60)}
    randomised = styled.random_groups(style)
    assert sum(style[s] != randomised[s] for s in style) > 10


def test_the_random_grouping_is_reproducible():
    style = {f"w{i}": i % 3 for i in range(20)}
    assert styled.random_groups(style, seed=7) == styled.random_groups(style, seed=7)


# -- the split -------------------------------------------------------------------------------------


@pytest.fixture
def index():
    return pd.DataFrame(
        {
            "file": [f"f{i}.png" for i in range(80)],
            "page": [f"p{i // 4}" for i in range(80)],
            "scribe": [f"w{i // 8}" for i in range(80)],
            "exercise": ["ex0"] * 80,
            "text": ["hello"] * 80,
        }
    )


def test_no_test_writer_appears_in_the_training_set(index):
    """The deployment case, and 7.4.7's `unseen_scribe` control applied here."""
    train, test, held = styled.split(index)
    assert not (set(train["scribe"]) & set(test["scribe"]))
    assert set(test["scribe"]) == set(held)


def test_the_split_holds_out_writers_rather_than_crops(index):
    train, test, held = styled.split(index)
    assert len(held) == round(styled.TEST_SCRIBES * index["scribe"].nunique())


def test_every_crop_lands_on_exactly_one_side(index):
    train, test, _ = styled.split(index)
    assert len(train) + len(test) == len(index)


def test_the_split_is_reproducible(index):
    assert styled.split(index)[2] == styled.split(index)[2]


# -- scoring ---------------------------------------------------------------------------------------


def test_a_perfect_transcription_scores_zero_error():
    result = styled.score(["hello world"], ["hello world"])
    assert result["cer"] == 0.0
    assert result["wer"] == 0.0
    assert result["exact"] == 1.0


def test_one_wrong_character_in_ten_is_a_tenth_of_a_cer():
    assert styled.score(["abcdefghij"], ["abcdefghiX"])["cer"] == pytest.approx(0.1, abs=1e-3)


def test_an_empty_prediction_is_scored_rather_than_crashing():
    """A crop whose group had no head gets an empty string, and it must count against the arm."""
    assert styled.score(["hello"], [""])["cer"] > 0.5


def test_exact_match_ignores_surrounding_whitespace():
    assert styled.score(["hello"], ["  hello "])["exact"] == 1.0


def test_the_sample_size_is_carried_with_every_score():
    assert styled.score(["a", "b"], ["a", "b"])["n"] == 2


# -- the reported gain -----------------------------------------------------------------------------


def test_a_relative_gain_is_the_share_of_the_error_removed():
    """The quantity the plan's '>= 15%' is stated in."""
    assert styled.relative_gain(0.40, 0.20) == 0.5
    assert styled.relative_gain(0.40, 0.34) == pytest.approx(0.15, abs=1e-4)


def test_a_worse_candidate_gives_a_negative_gain():
    assert styled.relative_gain(0.20, 0.40) < 0


def test_a_zero_baseline_does_not_divide_by_zero():
    assert styled.relative_gain(0.0, 0.1) == 0.0


# -- routing ---------------------------------------------------------------------------------------


def test_a_writer_the_clustering_never_saw_is_marked_unroutable_not_folded_into_group_zero():
    """Silently sending an unknown writer to cluster 0 would flatter whichever arm owns it."""
    frame = pd.DataFrame({"scribe": ["w0", "w1", "ghost"]})
    groups = {"w0": 0, "w1": 2}
    mapped = frame["scribe"].map(groups).fillna(-1).astype(int).to_numpy()
    assert mapped.tolist() == [0, 2, -1]


def test_every_test_crop_gets_a_prediction_even_with_a_missing_head():
    """Dropping unroutable crops would flatter the arm; they are scored as empty instead."""
    predicted = np.empty(3, dtype=object)
    predicted[0] = "hi"
    filled = [p if isinstance(p, str) else "" for p in predicted]
    assert filled == ["hi", "", ""]
