"""S3's target, its writer-disjointness guard, and the label masking the loss depends on."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.ocr import s3
from src.ocr.metrics import TARGET_CER


def test_s3_target_is_the_plans_own_number():
    """S3 scores against 9.3.6's constant rather than a target redefined here."""
    assert s3.TARGET_CER == TARGET_CER == 0.15


def test_check_disjoint_accepts_a_writer_disjoint_split():
    train = pd.DataFrame({"scribe": ["w1", "w1", "w2"]})
    evaluation = pd.DataFrame({"scribe": ["w3", "w4"]})
    assert s3.check_disjoint(train, evaluation) == {
        "train_writers": 2,
        "eval_writers": 2,
        "writer_overlap": 0,
    }


def test_check_disjoint_refuses_a_shared_writer():
    """The one error that would make a low CER meaningless has to raise, not warn."""
    train = pd.DataFrame({"scribe": ["w1", "w2"]})
    evaluation = pd.DataFrame({"scribe": ["w2", "w3"]})
    with pytest.raises(AssertionError, match="w2"):
        s3.check_disjoint(train, evaluation)


def test_targets_mask_padding_out_of_the_loss():
    """Padded positions must be -100 or short transcripts train the model to emit padding."""
    torch = pytest.importorskip("torch")
    proc = s3.processor()
    labels = s3.targets(proc, ["a much longer transcript here", "hi"], torch.device("cpu"))
    assert (labels == -100).any(), "no padding was masked"
    assert not (labels[0] == proc.tokenizer.pad_token_id).all()
    # every masked position is padding in the original tokenisation
    assert labels.shape[0] == 2


def test_targets_normalise_before_tokenising():
    """Scoring lowercases and collapses whitespace, so training must too."""
    pytest.importorskip("torch")
    import torch

    proc = s3.processor()
    a = s3.targets(proc, ["Send  Card"], torch.device("cpu"))
    b = s3.targets(proc, ["send card"], torch.device("cpu"))
    assert a.tolist() == b.tolist()


def test_train_dev_and_val_writers_are_mutually_disjoint():
    """The epoch is chosen on dev, so dev must not share a writer with train or with val."""
    (_, _, train), (_, _, dev) = s3.train_dev_split()
    _, _, val = s3.split("val")
    a = set(train["scribe"].dropna())
    b = set(dev["scribe"].dropna())
    c = set(val["scribe"].dropna())
    assert len(b) == s3.DEV_WRITERS
    assert not (a & b), "a dev writer is also a training writer"
    assert not (a & c) and not (b & c), "a val writer leaked into model selection"


def test_train_dev_split_keeps_every_crop():
    (train_files, _, _), (dev_files, _, _) = s3.train_dev_split()
    files, _, _ = s3.split("train")
    assert len(train_files) + len(dev_files) == len(files)
    assert not (set(train_files) & set(dev_files))


def test_augment_returns_a_usable_image():
    """Augmentation must not produce a degenerate crop the processor cannot normalise."""
    Image = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(0)
    base = Image.new("RGB", (99, 64), color=255)
    for _ in range(25):
        out = s3.augment(base, rng)
        assert out.size[0] >= 8 and out.size[1] >= 8


def test_evaluate_without_a_checkpoint_is_a_clean_failure(tmp_path):
    with pytest.raises(FileNotFoundError, match="--train"):
        s3.evaluate("val", checkpoint=tmp_path / "absent")


def test_write_report_round_trips(tmp_path):
    path = s3.write_report({"criterion": "S3", "cer": 0.1, "passes": True}, tmp_path / "s3.json")
    assert json.loads(path.read_text(encoding="utf-8"))["passes"] is True
