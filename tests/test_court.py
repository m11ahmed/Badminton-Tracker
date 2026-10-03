"""M3 homography tests use an independent analytic camera with held-out points.

A zero error on four fitted corners is not an independent calibration accuracy
claim. These tests check the math against other known court locations.
"""
import json

import numpy as np
import pytest

from badminton_tracker.court import create_calibration, load_calibration


# Synthetic camera: (u,v) = (80*x+140, 35*y+90)/(1-0.035*y).
# It expands the near side and produces a real perspective, not an affine map.
def synthetic_image_points(court_points):
    points = np.asarray(court_points, dtype=float).reshape(-1, 2)
    divisor = 1.0 - 0.035 * points[:, 1]
    return np.column_stack(((80.0 * points[:, 0] + 140.0) / divisor,
                            (35.0 * points[:, 1] + 90.0) / divisor))


def perspective_calibration():
    corners = synthetic_image_points([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]])
    return create_calibration(corners.tolist(), 1500, 1300,
                              method="manual", reference_frame=0)


def test_known_perspective_recovers_held_out_net_and_service_points():
    calibration = perspective_calibration()
    # Net midpoint, singles left short-service intersection, near right
    # singles service intersection and doubles left net line, all unfitted.
    known = np.array([[3.05, 6.7], [0.46, 4.72], [5.64, 8.68], [0, 6.7]])
    camera_pixels = synthetic_image_points(known)
    recovered = calibration.image_to_court(camera_pixels)
    assert recovered.shape == (4, 2)
    assert recovered == pytest.approx(known, abs=2e-5)
    assert calibration.court_to_image(known) == pytest.approx(camera_pixels, abs=2e-3)


def test_round_trip_preserves_external_points_without_clipping():
    calibration = perspective_calibration()
    known = np.array([[-0.25, 2.0], [6.4, 12.0], [3.05, 14.0], [3.05, -0.5]])
    pixels = synthetic_image_points(known)
    mapped = calibration.image_to_court(pixels)
    assert mapped == pytest.approx(known, abs=3e-5)
    assert mapped[0, 0] < 0 and mapped[1, 0] > 6.1
    assert mapped[2, 1] > 13.4 and mapped[3, 1] < 0
    assert calibration.court_to_image(mapped) == pytest.approx(pixels, abs=2e-3)


def test_perspective_horizon_is_rejected_instead_of_infinite_coordinates():
    calibration = perspective_calibration()
    # Inverse y=(v-90)/(35+0.035*v) has its projective horizon at v=-1000.
    with pytest.raises(ValueError):
        calibration.image_to_court([[100.0, -1000.0]])


def test_empty_transform_retains_empty_two_column_shape():
    calibration = perspective_calibration()
    assert calibration.image_to_court(np.empty((0, 2))).shape == (0, 2)
    assert calibration.court_to_image(np.empty((0, 2))).shape == (0, 2)


@pytest.mark.parametrize("points", [
    [[float("nan"), 100], [300, 100], [350, 500], [50, 500]],
    [[100, 100], [300, float("inf")], [350, 500], [50, 500]],
    [[100, 100], [200, 100], [300, 100], [400, 100]],
    [[100, 100], [300, 100], [100, 100], [50, 500]],
    [[100, 100], [350, 500], [300, 100], [50, 500]],
    [[100, 100], [300, 100], [190, 200], [50, 500]],
    [[-10, 100], [300, 100], [350, 500], [50, 500]],
    [[100, 100], [700, 100], [350, 500], [50, 500]],
    [[100, 100], [300, 100], [350, 700], [50, 500]],
])
def test_invalid_crossed_concave_or_out_of_image_corners_are_rejected(points):
    with pytest.raises(ValueError):
        create_calibration(points, 640, 600)


@pytest.mark.parametrize("order", [[3, 2, 1, 0], [1, 2, 3, 0], [1, 0, 3, 2]])
def test_reversed_or_wrong_start_order_is_rejected(order):
    corners = np.array([[100, 100], [300, 100], [350, 500], [50, 500]])
    with pytest.raises(ValueError):
        create_calibration(corners[order].tolist(), 640, 600)


@pytest.mark.parametrize("points", [
    [[100, 100], [300, 100], [300, 100.0001], [100, 100.0001]],
    [[100, 100], [100.000001, 100], [100.000001, 500], [100, 500]],
])
def test_nearly_degenerate_corner_geometry_is_rejected(points):
    with pytest.raises(ValueError):
        create_calibration(points, 640, 600)


@pytest.mark.parametrize("points", [
    [[100.0, float("nan")]], [[float("inf"), 100.0]], [100.0, 200.0, 300.0],
])
def test_transform_rejects_invalid_points(points):
    with pytest.raises(ValueError):
        perspective_calibration().image_to_court(points)


def test_serialized_calibration_round_trip_binds_source_dimensions(project_tmp_path):
    calibration = perspective_calibration()
    path = project_tmp_path / "calibration.json"
    payload = calibration.to_dict()
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    restored = load_calibration(path, 1500, 1300)
    assert restored.image_to_court(synthetic_image_points([[3.05, 6.7]])) == pytest.approx(
        np.array([[3.05, 6.7]]), abs=2e-5)
    assert restored.source_dimensions == calibration.source_dimensions
    with pytest.raises(ValueError):
        load_calibration(path, 1502, 1300)
    with pytest.raises(ValueError):
        load_calibration(path, 1500, 1302)
