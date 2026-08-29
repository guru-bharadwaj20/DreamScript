"""Phase 3.1.3 - perspective rectification.

The strong test is a round trip: take a known page, warp it *out* of true with a homography we
choose, then let the pipeline warp it back and check the content lands where it should.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocess import rectify as rect


def flat_page(width=520, height=380) -> np.ndarray:
    image = np.full((height, width, 3), 235, np.uint8)
    cv2.rectangle(image, (60, 60), (200, 150), (20, 20, 20), 4)
    cv2.circle(image, (400, 280), 45, (20, 20, 20), 4)
    cv2.line(image, (200, 105), (355, 265), (20, 20, 20), 3)
    return image


def skewed(image, corners) -> np.ndarray:
    h, w = image.shape[:2]
    source = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    homography = cv2.getPerspectiveTransform(source, corners.astype(np.float32))
    canvas = np.full((700, 900, 3), 60, np.uint8)
    return cv2.warpPerspective(
        image, homography, (900, 700), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT
    )


def test_rectifying_a_skewed_page_recovers_its_aspect_ratio():
    corners = np.array([[150, 90], [770, 150], [740, 600], [180, 540]], np.float32)
    result = rect.rectify(skewed(flat_page(), corners))
    assert result.applied
    width, height = result.size
    # The source page is 520x380; the recovered aspect should be close, allowing for the
    # outward margin page detection adds and for foreshortening.
    assert 1.1 < width / height < 1.7


def test_no_page_means_the_image_passes_through_untouched():
    image = np.full((300, 400, 3), 235, np.uint8)
    cv2.rectangle(image, (100, 100), (250, 200), (20, 20, 20), 3)
    result = rect.rectify(image)
    assert not result.applied
    assert result.homography is None
    assert np.array_equal(result.image, image)


def test_points_map_through_the_same_homography():
    """Anything holding coordinates in the old frame has to move with the pixels."""
    corners = np.array([[150, 90], [770, 150], [740, 600], [180, 540]], np.float32)
    photo = skewed(flat_page(), corners)
    result = rect.rectify(photo)
    assert result.applied
    moved = result.map_points(result.quad)
    width, height = result.size
    # The page's own corners must land on the corners of the output.
    assert np.allclose(moved[0], [0, 0], atol=2)
    assert np.allclose(moved[2], [width - 1, height - 1], atol=2)


def test_map_points_is_identity_when_nothing_was_applied():
    result = rect.rectify(np.full((200, 200, 3), 235, np.uint8))
    points = np.array([[10, 20], [30, 40]], np.float32)
    assert np.array_equal(result.map_points(points), points)


def test_map_box_returns_the_bounding_box_of_the_warped_corners():
    corners = np.array([[150, 90], [770, 150], [740, 600], [180, 540]], np.float32)
    result = rect.rectify(skewed(flat_page(), corners))
    box = result.map_box([200, 150, 100, 80])
    assert len(box) == 4
    assert box[2] > 0 and box[3] > 0


def test_target_size_comes_from_the_quad_not_the_frame():
    quad = np.array([[0, 0], [400, 0], [400, 200], [0, 200]], np.float32)
    assert rect.target_size(quad) == (400, 200)


def test_target_size_is_capped():
    quad = np.array([[0, 0], [90000, 0], [90000, 45000], [0, 45000]], np.float32)
    width, height = rect.target_size(quad)
    assert max(width, height) <= rect.MAX_SIDE
    assert abs(width / height - 2.0) < 0.01, "the aspect ratio must survive the cap"


def test_qa_grid_is_produced():
    figure = rect.qa_grid(8)
    if figure is None:
        pytest.skip("hdBPMN IR not present")
    assert figure.is_file()
