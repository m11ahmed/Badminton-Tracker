"""Heatmap checks use known pixels; they do not measure model accuracy."""
import numpy as np
import pytest

from badminton_tracker.tracknet import decode_heatmap


def test_missing_heatmap_retains_score_without_fabricating_origin():
    heatmap = np.full((12, 16), 0.2, dtype=np.float32)
    x, y, score = decode_heatmap(heatmap, 320, 120)
    assert x is None and y is None
    assert score == pytest.approx(0.2)


def test_bounding_rectangle_center_scales_to_source_pixels():
    heatmap = np.zeros((12, 16), dtype=np.float32)
    heatmap[2:5, 4:7] = 0.8
    x, y, score = decode_heatmap(heatmap, 320, 120)
    # The upstream decoder truncates (5.5, 3.5) to (5, 3) before scaling.
    assert (x, y) == (100.0, 30.0)
    assert score == pytest.approx(0.8)


def test_largest_component_wins_over_one_bright_distractor():
    heatmap = np.zeros((12, 16), dtype=np.float32)
    heatmap[1, 1] = 0.99
    heatmap[6:9, 8:12] = 0.7
    assert decode_heatmap(heatmap, 160, 120) == pytest.approx((100.0, 70.0, 0.7))


def test_threshold_is_strict_and_zero_coordinates_can_be_real():
    heatmap = np.zeros((12, 16), dtype=np.float32)
    heatmap[0, 0] = 0.5
    assert decode_heatmap(heatmap, 160, 120, threshold=0.5)[:2] == (None, None)
    heatmap[0, 0] = 0.6
    assert decode_heatmap(heatmap, 160, 120, threshold=0.5)[:2] == (0.0, 0.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_output_is_rejected(value):
    heatmap = np.zeros((12, 16), dtype=np.float32)
    heatmap[1, 1] = value
    with pytest.raises(ValueError, match="nonfinite"):
        decode_heatmap(heatmap, 160, 120)


@pytest.mark.parametrize("heatmap", [np.array([]), np.zeros((12,)), np.zeros((1, 12, 16))])
def test_invalid_heatmap_shapes_are_rejected(heatmap):
    with pytest.raises(ValueError, match="two dimensional"):
        decode_heatmap(heatmap, 160, 120)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan")])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        decode_heatmap(np.zeros((12, 16)), 160, 120, threshold=threshold)
