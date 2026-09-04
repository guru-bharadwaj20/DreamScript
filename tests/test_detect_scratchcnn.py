"""Phase 9.2.1 / 9.2.4 / 9.2.5 / 9.2.6 / 9.2.7 - the pieces a test can pin without a GPU."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.detect import ablations, crops, transfer, visualise
from src.detect.scratchcnn import Spec, build, class_weights, evaluate

# -- the architecture ------------------------------------------------------------------------


def test_the_reference_model_maps_a_crop_to_one_logit_per_class():
    model = build(Spec())
    out = model(torch.zeros(2, 1, crops.SIZE, crops.SIZE))
    assert out.shape == (2, len(crops.SHAPE_CLASSES))


@pytest.mark.parametrize(
    "spec",
    [
        Spec(kernel=5),
        Spec(kernel=7),
        Spec(pool="avg"),
        Spec(batchnorm=False),
        Spec(stages=((16, 2), (32, 2))),
        Spec(stages=((16, 2), (32, 2), (64, 1))),
    ],
)
def test_every_ablation_configuration_builds_and_runs(spec):
    out = build(spec)(torch.zeros(1, 1, spec.size, spec.size))
    assert out.shape == (1, len(spec.classes))


def test_dropping_batchnorm_removes_the_norm_layers_and_restores_the_bias():
    with_bn = build(Spec())
    without = build(Spec(batchnorm=False))
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in with_bn)
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in without)
    assert next(m for m in with_bn if isinstance(m, torch.nn.Conv2d)).bias is None
    assert next(m for m in without if isinstance(m, torch.nn.Conv2d)).bias is not None


def test_average_pooling_is_actually_swapped_in():
    model = build(Spec(pool="avg"))
    assert any(isinstance(m, torch.nn.AvgPool2d) for m in model)
    assert not any(isinstance(m, torch.nn.MaxPool2d) for m in model)


# -- the imbalance handling ------------------------------------------------------------------


def test_the_rare_class_gets_the_larger_weight():
    """24,373 rectangles beside 195 parallelograms is why the loss is weighted at all."""
    y = np.array([0] * 900 + [1] * 100, dtype=np.int64)
    weights = class_weights(y, ("a", "b"))
    assert weights[1] > weights[0]
    assert weights[0] * 900 == pytest.approx(weights[1] * 100)


def test_an_absent_class_gets_zero_weight_rather_than_infinity():
    y = np.array([0, 0, 0], dtype=np.int64)
    assert class_weights(y, ("a", "b", "c"))[2] == 0.0


# -- the metric ------------------------------------------------------------------------------


def test_a_majority_only_predictor_scores_high_accuracy_and_low_macro_f1():
    """The reason 9.2 reports macro F1: this model is useless and 59% accurate."""

    class Majority(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros(len(x), len(crops.SHAPE_CLASSES))
            out[:, 0] = 1.0
            return out

    y = np.array([0] * 59 + [1] * 20 + [2] * 21, dtype=np.int64)
    x = np.zeros((100, 1, 8, 8), dtype=np.float32)
    result = evaluate(Majority(), x, y, torch.device("cpu"))
    assert result["accuracy"] == pytest.approx(0.59)
    assert result["macro_f1"] < 0.30


# -- the ablation grid -----------------------------------------------------------------------


def test_every_arm_differs_from_the_reference_in_exactly_one_respect():
    """A grid that moves two things at once cannot attribute anything."""
    rows = ablations.grid()
    reference = next(spec for name, _, spec in rows if name == "reference")
    for name, _, spec in rows:
        if name == "reference":
            continue
        differences = sum(
            [
                spec.kernel != reference.kernel,
                spec.stages != reference.stages,
                spec.pool != reference.pool,
                spec.batchnorm != reference.batchnorm,
            ]
        )
        assert differences == 1, name


def test_the_grid_covers_all_four_axes_the_plan_names():
    names = {name for name, _, _ in ablations.grid()}
    assert {"kernel_5", "kernel_7", "depth_3", "depth_2", "pool_avg", "no_batchnorm"} <= names


# -- transfer --------------------------------------------------------------------------------


def test_the_frozen_arm_trains_only_its_head():
    model = transfer.build_resnet(pretrained=False, freeze=True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all(n.startswith("fc.") for n in trainable), trainable


def test_the_finetuned_arm_trains_everything():
    model = transfer.build_resnet(pretrained=False, freeze=False)
    assert all(p.requires_grad for p in model.parameters())


def test_the_head_is_resized_to_the_shape_vocabulary():
    model = transfer.build_resnet(pretrained=False, freeze=False)
    assert model.fc.out_features == len(crops.SHAPE_CLASSES)


def test_grey_crops_are_replicated_to_three_channels():
    x = np.zeros((2, 1, 8, 8), dtype=np.float32)
    assert transfer.to_three_channel(x).shape == (2, 3, 8, 8)


# -- grad-cam --------------------------------------------------------------------------------


def test_the_cam_is_the_size_of_the_crop_and_normalised():
    model = build(Spec())
    cam = visualise.gradcam(model, np.random.rand(1, crops.SIZE, crops.SIZE).astype(np.float32), 0)
    assert cam.shape == (crops.SIZE, crops.SIZE)
    assert cam.min() >= 0.0 and cam.max() <= 1.0


def test_the_cam_is_taken_at_the_last_convolution():
    model = build(Spec())
    index = visualise.last_conv_index(model)
    assert isinstance(model[index], torch.nn.Conv2d)
    assert not any(isinstance(m, torch.nn.Conv2d) for m in list(model)[index + 1 :])


def test_identical_kernels_would_be_reported_as_such():
    """The check the filter grid cannot make by eye: sixteen small squares all look alike."""
    model = build(Spec())
    conv = visualise.first_conv(model)
    with torch.no_grad():
        conv.weight.copy_(conv.weight[0].expand_as(conv.weight))
    assert visualise.filter_stats(model)["mean_abs_cosine_between_kernels"] == pytest.approx(1.0)


# -- the crop corpus -------------------------------------------------------------------------


def test_the_arrowhead_is_not_a_shape_class():
    """9.1.2's derived class has a convention for a box, not a shape."""
    assert "arrowhead" not in crops.SHAPE_CLASSES


def test_a_crop_is_padded_squared_and_greyscale():
    image = np.full((200, 300), 240, np.uint8)
    image[50:100, 80:180] = 20
    patch = crops.crop_one(image, (80, 50, 100, 50))
    assert patch.shape == (crops.SIZE, crops.SIZE)


def test_a_box_smaller_than_the_floor_is_refused():
    image = np.full((200, 300), 240, np.uint8)
    assert crops.crop_one(image, (10, 10, 5, 5)) is None


def test_the_margin_reaches_outside_the_annotated_box():
    """A tight crop of a double-circle's inner ring is a circle; the context is the label."""
    image = np.zeros((200, 200), np.uint8)
    image[90:110, 90:110] = 255
    tight = crops.crop_one(image, (90, 90, 20, 20), margin=0.0)
    padded = crops.crop_one(image, (90, 90, 20, 20), margin=0.5)
    assert tight.mean() > padded.mean()
