"""Phase 6.1.1 - frozen backbone selection."""

from __future__ import annotations

import numpy as np
import pytest

from src.embed import backbone
from src.utils.config import ROOT

FIXTURES = sorted((ROOT / "tests" / "fixtures").glob("*.png"))


def test_a_page_comes_back_at_the_backbones_input_size():
    array = backbone.page_array(FIXTURES[0], "binary")
    assert array.shape == (backbone.SIZE, backbone.SIZE)
    assert array.dtype == np.float32


def test_a_binarized_page_is_two_toned_and_a_grey_one_is_not():
    """The comparison this task exists to make is only meaningful if the inputs differ."""
    path = FIXTURES[0]
    binary = backbone.page_array(path, "binary")
    grey = backbone.page_array(path, "gray")

    # Counting distinct values is the wrong test: downsampling a two-tone mask interpolates
    # along every stroke edge and can produce more distinct values than a flat greyscale. What
    # binarizing actually does is push the mass to the ends, so that is what is measured.
    def extreme_share(array):
        return float(((array < 0.05) | (array > 0.95)).mean())

    assert extreme_share(binary) > extreme_share(grey)
    assert not np.allclose(binary, grey)


def test_ink_is_dark_on_both_paths():
    for mode in ("binary", "gray"):
        array = backbone.page_array(FIXTURES[0], mode)
        assert array.min() >= 0.0 and array.max() <= 1.0
        # A page is mostly paper, so the mean must sit well above the middle.
        assert array.mean() > 0.5


def test_an_unknown_input_mode_is_refused():
    with pytest.raises(ValueError, match="binary or gray"):
        backbone.page_array(FIXTURES[0], "sepia")


def test_an_unreadable_page_raises_rather_than_returning_noise(tmp_path):
    empty = tmp_path / "not_an_image.png"
    empty.write_bytes(b"")
    with pytest.raises(OSError):
        backbone.page_array(empty, "binary")


def test_the_batch_is_three_channel_and_normalised():
    arrays = [np.full((backbone.SIZE, backbone.SIZE), 0.5, np.float32) for _ in range(3)]
    statistics = ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    tensor = backbone.batch_tensor(arrays, statistics)
    assert tuple(tensor.shape) == (3, 3, backbone.SIZE, backbone.SIZE)
    # 0.5 with mean 0.5 and std 0.25 must come out as exactly zero.
    assert float(tensor.abs().max()) == pytest.approx(0.0, abs=1e-6)


def test_the_grey_plane_is_repeated_not_padded():
    arrays = [np.linspace(0, 1, backbone.SIZE**2, dtype=np.float32).reshape(224, 224)]
    tensor = backbone.batch_tensor(arrays, ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    assert np.allclose(tensor[0, 0].numpy(), tensor[0, 1].numpy())
    assert np.allclose(tensor[0, 0].numpy(), tensor[0, 2].numpy())


def test_an_unknown_backbone_is_refused():
    with pytest.raises(ValueError, match="unknown backbone"):
        backbone.load("resnet50")


def test_resnet18_returns_one_512_wide_row_per_page():
    result = backbone.embed(FIXTURES[:3], "resnet18", "binary", batch_size=2)
    assert result["embeddings"].shape == (3, 512)
    assert result["readable"] == 3
    assert result["dimension"] == 512


def test_an_unreadable_page_becomes_a_row_of_nan(tmp_path):
    """A dead page must not kill the batch - 4.2.1 made the same choice for the same reason."""
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a png")
    result = backbone.embed([FIXTURES[0], broken], "resnet18", "binary", batch_size=4)
    assert result["readable"] == 1
    assert not np.isnan(result["embeddings"][0]).any()
    assert np.isnan(result["embeddings"][1]).all()


def test_the_backbone_is_frozen():
    """Nothing here trains, so every parameter must come back with gradients switched off."""
    import torch

    model, _, _ = backbone.load("resnet18")
    assert not model.training
    with torch.no_grad():
        tensor = backbone.batch_tensor(
            [backbone.page_array(FIXTURES[0], "binary")],
            (backbone.IMAGENET_MEAN, backbone.IMAGENET_STD),
        ).to(backbone.device())
        first = model(tensor)
        second = model(tensor)
    assert torch.allclose(first, second)


def test_the_probe_scores_a_separable_embedding_highly():
    rng = np.random.default_rng(20)
    y = np.array(["a"] * 60 + ["b"] * 60, dtype=object)
    X = np.array([[0.0] * 4 if label == "a" else [4.0] * 4 for label in y])
    X = X + rng.normal(0, 0.4, size=X.shape)
    assert backbone.probe(X, y)["macro_f1"] > 0.9


def test_the_probe_ignores_unreadable_rows():
    rng = np.random.default_rng(21)
    y = np.array(["a"] * 40 + ["b"] * 40, dtype=object)
    X = np.array([[0.0, 0.0] if label == "a" else [3.0, 3.0] for label in y])
    X = X + rng.normal(0, 0.3, size=X.shape)
    X[0] = np.nan
    assert backbone.probe(X, y)["rows"] == len(y) - 1


def test_provenance_catches_an_embedding_that_only_knows_the_source():
    """The failure mode this corpus invites: source determines class for 1,200 of 1,340 rows."""
    rng = np.random.default_rng(60)
    sources = np.array(["hdbpmn"] * 60 + ["sketch2code"] * 60 + ["chaos"] * 60, dtype=object)
    labels = np.array(
        ["flowchart"] * 60 + ["wireframe"] * 60 + ["circuit"] * 30 + ["er_diagram"] * 30,
        dtype=object,
    )
    # Column 0 encodes the source and nothing else. Within chaos it is constant, so the
    # embedding cannot tell a circuit from an ER diagram.
    codes = {"hdbpmn": 0.0, "sketch2code": 5.0, "chaos": 10.0}
    X = np.array([[codes[s]] for s in sources]) + rng.normal(0, 0.2, size=(180, 1))
    X = np.hstack([X, rng.normal(size=(180, 3))])

    result = backbone.provenance(X, labels, sources)
    assert result["source_accuracy"] > 0.95
    assert result["within_chaos"]["macro_f1"] < 0.7
    assert result["label_determined_by_source"] == pytest.approx(60 / 180 * 2 + 30 / 180, 0.01)


def test_provenance_reports_a_representation_that_reads_the_diagram():
    rng = np.random.default_rng(61)
    sources = np.array(["chaos"] * 120, dtype=object)
    labels = np.array(["circuit"] * 60 + ["er_diagram"] * 60, dtype=object)
    X = np.array([[0.0] if label == "circuit" else [4.0] for label in labels])
    X = np.hstack([X + rng.normal(0, 0.5, size=(120, 1)), rng.normal(size=(120, 2))])
    result = backbone.provenance(X, labels, sources)
    assert result["within_chaos"]["macro_f1"] > 0.9


def test_provenance_skips_the_within_source_probe_when_there_is_nothing_to_ask():
    labels = np.array(["flowchart"] * 40, dtype=object)
    sources = np.array(["hdbpmn"] * 40, dtype=object)
    X = np.random.default_rng(62).normal(size=(40, 4))
    assert "within_chaos" not in backbone.provenance(X, labels, sources)


def test_the_corpus_sample_is_stratified():
    paths, labels, ids = backbone.corpus(limit=50)
    assert len(paths) == len(labels) == len(ids)
    counts = {name: int((labels == name).sum()) for name in set(labels.tolist())}
    assert len(counts) > 1
    assert max(counts.values()) - min(counts.values()) <= max(counts.values())


def test_the_corpus_is_exactly_the_handcrafted_table():
    """Walking the disk returns 1,695 real pages; 4.2.2's table holds 1,340 of them. The probe
    is only comparable with 5.1.1 if it is scored on the table's rows, in the table's order."""
    import pandas as pd

    from src.classify.data import TABLE

    frame = pd.read_parquet(TABLE)
    expected = frame[~frame["synthetic"].astype(bool)]["id"].tolist()
    _, labels, ids = backbone.corpus()
    assert list(ids) == expected
    assert len(labels) == len(expected)


def test_the_corpus_is_not_just_whatever_is_on_disk():
    from src.features.build import collect

    on_disk = len([row for row in collect(None) if not row["synthetic"]])
    _, _, ids = backbone.corpus()
    assert len(ids) < on_disk
