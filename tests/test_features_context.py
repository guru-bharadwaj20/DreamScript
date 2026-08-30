"""Phase 4 - the shared page context."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.features import context as ctx


def canvas(size=(500, 700)) -> np.ndarray:
    return np.zeros(size, np.uint8)


def shape_only(kind: str, size: int = 400) -> np.ndarray:
    image = np.zeros((size, size), np.uint8)
    if kind == "box":
        cv2.rectangle(image, (60, 60), (340, 300), 255, 3)
    elif kind == "diamond":
        cv2.polylines(
            image, [np.array([[200, 40], [360, 200], [200, 360], [40, 200]])], True, 255, 3
        )
    elif kind == "circle":
        cv2.circle(image, (200, 200), 150, 255, 3)
    elif kind == "ellipse":
        cv2.ellipse(image, (200, 200), (170, 90), 0, 0, 360, 255, 3)
    return image > 0


def two_boxes_joined() -> np.ndarray:
    image = canvas()
    cv2.rectangle(image, (50, 50), (250, 180), 255, 3)
    cv2.rectangle(image, (400, 50), (600, 180), 255, 3)
    cv2.line(image, (250, 115), (400, 115), 255, 3)
    return image > 0


def nested_panel() -> np.ndarray:
    image = canvas((600, 600))
    cv2.rectangle(image, (30, 30), (570, 570), 255, 3)
    cv2.rectangle(image, (60, 60), (280, 200), 255, 3)
    cv2.rectangle(image, (320, 60), (540, 200), 255, 3)
    return image > 0


@pytest.mark.parametrize("kind", ["box", "diamond", "circle", "ellipse"])
def test_one_drawn_shape_is_one_region(kind):
    """The inner and outer edge of a pen stroke must not both count."""
    assert len(ctx.build_regions(shape_only(kind))) == 1


def test_a_connector_does_not_invent_a_third_node():
    """Two boxes joined by a line trace as one closed outline; that shell is not a node."""
    assert len(ctx.build_regions(two_boxes_joined())) == 2


def test_genuine_nesting_survives_the_fusion_test():
    regions = ctx.build_regions(nested_panel())
    assert len(regions) == 3
    assert sorted(r.depth for r in regions) == [0, 1, 1]


def test_nesting_is_recounted_not_inherited():
    """Depth comes from the surviving regions, so a dropped shell cannot leave a phantom parent."""
    for region in ctx.build_regions(two_boxes_joined()):
        assert region.depth == 0
        assert region.parent == -1


def test_connector_ink_is_what_is_left_outside_the_shapes():
    context = ctx.from_masks(two_boxes_joined(), page_id="joined")
    assert context.connector_mask.sum() > 0
    for region in context.regions:
        x, y, w, h = region.bbox
        assert not context.connector_mask[y : y + h, x : x + w].any()


def test_no_connector_ink_when_nothing_joins_the_shapes():
    context = ctx.from_masks(nested_panel(), page_id="nested")
    assert context.connector_mask.sum() == 0


def test_densify_puts_back_the_points_chain_approx_removed():
    """Four corner points cannot carry curvature; the resampled outline can."""
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    assert len(ctx.densify(square, spacing=2.0)) == 200


def test_straightness_separates_polygons_from_curves():
    """Reproduces 3.2.5's numbers through the region path, which is what 4.1.3 will use."""
    box = ctx.build_regions(shape_only("box"))[0]
    circle = ctx.build_regions(shape_only("circle"))[0]
    assert box.straightness > 0.6 > circle.straightness
    assert box.corners == 4
    assert circle.corners == 0


def test_extent_separates_a_box_from_a_diamond():
    """Both are four-vertex polygons; only the fill of their own box tells them apart."""
    box = ctx.build_regions(shape_only("box"))[0]
    diamond = ctx.build_regions(shape_only("diamond"))[0]
    assert box.vertices == diamond.vertices == 4
    assert box.extent > 0.9 > diamond.extent
    assert diamond.extent == pytest.approx(0.5, abs=0.08)


def test_specks_are_not_regions():
    image = canvas()
    cv2.rectangle(image, (50, 50), (250, 180), 255, 3)
    cv2.circle(image, (600, 400), 3, 255, -1)
    assert len(ctx.build_regions(image > 0)) == 1


def test_context_is_built_from_the_two_layer_partition(flowchart_image):
    context = ctx.from_masks(flowchart_image < 128, flowchart_image, page_id="fixture")
    assert np.array_equal(context.text_mask | context.shape_mask, context.mask)
    assert not (context.text_mask & context.shape_mask).any()


def test_from_image_reads_a_real_fixture(fixtures_dir):
    context = ctx.from_image(fixtures_dir / "flowchart.png")
    assert context.id == "flowchart"
    assert context.summary()["regions"] >= 1
    assert 0.0 < context.summary()["ink_frac"] < 1.0
