"""One split vocabulary, two spellings, and a typo that fails instead of filtering to nothing.

The manifest and the feature tables say `validation`; `data/processed/detect/index.json` and
`data.yaml` say `val`, because ultralytics walks a directory with that name. Both are right where
they are. What was wrong is that asking for the other one matched no rows and returned an empty
result, so `corpus.pages(("validation",))` was indistinguishable from "this split is empty".
"""

from __future__ import annotations

import pytest

from src.utils.splits import ALIASES, CANONICAL, DETECT, detect_name, normalise, same


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("train", "train"),
        ("training", "train"),
        ("val", "validation"),
        ("valid", "validation"),
        ("validation", "validation"),
        ("dev", "validation"),
        ("VAL", "validation"),
        ("  test  ", "test"),
        ("eval", "test"),
    ],
)
def test_every_spelling_in_the_repo_normalises(spelling, expected):
    assert normalise(spelling) == expected


def test_an_unknown_spelling_raises_rather_than_filtering_to_nothing():
    with pytest.raises(ValueError, match="unknown split"):
        normalise("vall")
    with pytest.raises(ValueError, match="unknown split"):
        normalise("")


def test_detect_name_is_the_on_disk_spelling():
    assert detect_name("validation") == "val"
    assert detect_name("val") == "val"
    assert [detect_name(name) for name in CANONICAL] == list(DETECT)


def test_same_sees_through_the_two_vocabularies():
    assert same("val", "validation")
    assert not same("val", "test")


def test_the_two_modules_that_had_half_this_table_now_share_it():
    from src.codegen import schema, splits
    from src.detect import dataset

    assert splits.SPLITS == CANONICAL
    assert schema.SPLITS == CANONICAL
    assert dataset.SPLITS == DETECT
    assert dataset.SPLIT_ALIAS["validation"] == "val"
    assert dataset.SPLIT_ALIAS["val"] == "val"


def test_codegen_splits_still_answers_none_for_an_unlabelled_row():
    """Deliberately not the strict form: that one filters manifest rows, which may carry
    anything, and an unlabelled row is skipped rather than fatal. The table is shared."""
    from src.codegen.splits import _normalise

    assert _normalise(None) is None
    assert _normalise("nonsense") is None
    assert _normalise("val") == "validation"
    assert set(ALIASES) >= {"val", "validation", "train", "test"}
