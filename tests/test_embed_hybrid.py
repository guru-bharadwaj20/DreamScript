"""Phase 6.1.4 - handcrafted, embedding, and the two concatenated."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.classify.data import Dataset
from src.embed import hybrid


@pytest.fixture
def handcrafted():
    rng = np.random.default_rng(40)
    y = np.array(["a"] * 60 + ["b"] * 60, dtype=object)
    X = np.array([[0.0, 0.0] if label == "a" else [2.0, 2.0] for label in y])
    X = X + rng.normal(0, 0.9, size=X.shape)
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(120)], dtype=object),
        ids=np.array([f"page/{i:03d}" for i in range(120)], dtype=object),
        feature_names=["f0", "f1"],
        corpus="real",
        sources=np.array(["src"] * 120, dtype=object),
    )


@pytest.fixture
def cached(handcrafted, tmp_path, monkeypatch):
    """A reduced-embedding file whose rows are deliberately in a *different* order."""
    rng = np.random.default_rng(41)
    order = rng.permutation(len(handcrafted.ids))
    frame = pd.DataFrame(
        {
            "row": np.arange(len(order)),
            "id": handcrafted.ids[order],
            "synthetic": [False] * len(order),
            "diagram_type": handcrafted.y[order],
        }
    )
    # Column 0 encodes the label, so a correct join is separable and a wrong one is not.
    matrix = np.zeros((len(order), 4), np.float32)
    matrix[:, 0] = np.where(handcrafted.y[order] == "a", 0.0, 3.0)
    matrix[:, 1:] = rng.normal(size=(len(order), 3))

    path = tmp_path / "pca.npy"
    np.save(path, matrix)
    monkeypatch.setattr(hybrid, "load_handcrafted", lambda corpus: handcrafted)
    monkeypatch.setattr("src.embed.reduce.REDUCED", path)
    monkeypatch.setattr("src.embed.cache.load", lambda *a, **k: (matrix, frame, {}))
    return matrix, frame


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="kind must be one of"):
        hybrid.dataset("embeddings")


def test_handcrafted_is_the_phase_5_table_untouched(handcrafted, monkeypatch):
    monkeypatch.setattr(hybrid, "load_handcrafted", lambda corpus: handcrafted)
    assert hybrid.dataset("handcrafted").n_features == 2


def test_the_embedding_table_has_the_embedding_width(cached, handcrafted):
    data = hybrid.dataset("embedding")
    assert data.n_features == 4
    assert len(data.y) == len(handcrafted.y)


def test_the_hybrid_is_both_tables_side_by_side(cached, handcrafted):
    data = hybrid.dataset("hybrid")
    assert data.n_features == 2 + 4
    assert data.feature_names[:2] == handcrafted.feature_names
    assert data.feature_names[2].startswith("pca_")


def test_the_join_is_by_id_and_not_by_position(cached, handcrafted):
    """The cache fixture is deliberately shuffled; a positional join would misalign it."""
    data = hybrid.dataset("embedding")
    # Column 0 of the cache encodes the label, so after a correct join it must still do so.
    expected = np.where(handcrafted.y == "a", 0.0, 3.0)
    assert np.allclose(data.X[:, 0], expected)


def test_the_labels_and_ids_come_from_the_handcrafted_table(cached, handcrafted):
    data = hybrid.dataset("hybrid")
    assert np.array_equal(data.y, handcrafted.y)
    assert np.array_equal(data.ids, handcrafted.ids)
    assert np.array_equal(data.groups, handcrafted.groups)


def test_a_missing_id_is_refused_rather_than_dropped(handcrafted, tmp_path, monkeypatch):
    frame = pd.DataFrame(
        {"row": [0, 1], "id": ["page/000", "page/001"], "synthetic": [False, False]}
    )
    matrix = np.zeros((2, 3), np.float32)
    path = tmp_path / "pca.npy"
    np.save(path, matrix)
    monkeypatch.setattr(hybrid, "load_handcrafted", lambda corpus: handcrafted)
    monkeypatch.setattr("src.embed.reduce.REDUCED", path)
    monkeypatch.setattr("src.embed.cache.load", lambda *a, **k: (matrix, frame, {}))
    with pytest.raises(KeyError, match="not in the embedding cache"):
        hybrid.dataset("hybrid")


def test_a_missing_reduction_names_the_command_that_builds_it(handcrafted, tmp_path, monkeypatch):
    monkeypatch.setattr(hybrid, "load_handcrafted", lambda corpus: handcrafted)
    monkeypatch.setattr("src.embed.reduce.REDUCED", tmp_path / "absent.npy")
    with pytest.raises(FileNotFoundError, match="src.embed.reduce"):
        hybrid.dataset("hybrid")


def test_the_score_is_averaged_over_the_seeds(handcrafted):
    result = hybrid.score(handcrafted, seeds=(42, 43))
    assert len(result["macro_f1_by_seed"]) == 2
    assert result["macro_f1"] == pytest.approx(np.mean(result["macro_f1_by_seed"]), abs=1e-3)
    assert result["features"] == 2


def test_a_separable_table_scores_highly(handcrafted):
    assert hybrid.score(handcrafted, seeds=(42,))["macro_f1"] > 0.8


def test_the_run_ranks_all_three(cached):
    result = hybrid.run("real")
    assert set(result["scores"]) == set(hybrid.KINDS)
    assert len(result["ranking"]) == 3
    assert result["best"] in result["ranking"]
    assert isinstance(result["hybrid_over_handcrafted"], float)


def test_mcnemar_finds_no_difference_between_identical_predictions():
    """The 6.1.4 result rests on this test saying 'not significant' when it should."""
    from src.classify.significance import mcnemar

    truth = np.array(["a", "b"] * 50, dtype=object)
    identical = truth.copy()
    result = mcnemar(identical == truth, identical == truth)
    assert result["discordant"] == 0
    assert result["p_value"] == 1.0


def test_significance_compares_every_pair_on_the_same_rows(monkeypatch):
    """McNemar is only meaningful if the partition is shared; the tables share y, so it is."""
    rng = np.random.default_rng(70)
    y = np.array(["a"] * 60 + ["b"] * 60, dtype=object)

    def table(separation):
        X = np.array([[0.0] if label == "a" else [separation] for label in y])
        return hybrid.Dataset(
            X=np.hstack([X + rng.normal(0, 0.5, size=(120, 1)), rng.normal(size=(120, 2))]),
            y=y,
            groups=np.arange(120).astype(object),
            ids=np.arange(120).astype(object),
            feature_names=["f0", "f1", "f2"],
            corpus="test",
        )

    rows = hybrid.significance({"weak": table(0.3), "strong": table(6.0)})
    assert len(rows) == 1
    assert rows[0]["discordant"] > 0
    assert set(rows[0]) >= {"a", "b", "p_value", "significant_at_05", "discordant"}
