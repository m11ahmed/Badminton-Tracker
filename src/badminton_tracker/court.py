"""Validated plane homography for a fixed rear camera badminton court.

Numerical corner fit checks describe the transform, not physical accuracy. All
four supplied corners must be the OUTER doubles court line intersections.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4
SINGLE_WIDTH_M = 5.18
CORNER_ORDER = ("far_left", "far_right", "near_right", "near_left")
COURT_CORNERS_M = np.asarray([[0.0, 0.0], [COURT_WIDTH_M, 0.0],
                             [COURT_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M]], dtype=np.float64)


def _dimensions(width, height):
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 2
           for value in (width, height)):
        raise ValueError("Source image dimensions must be integers of at least two pixels.")
    return int(width), int(height)


def _points(points):
    try:
        values = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Coordinates must be an N by 2 array of finite numbers.") from error
    if values.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Coordinates must be an N by 2 array of finite numbers.")
    return values


def _project(matrix, points):
    values = _points(points)
    if len(values) == 0:
        return values.copy()
    homogeneous = np.column_stack((values, np.ones(len(values))))
    projected = homogeneous @ matrix.T
    denominators = projected[:, 2]
    scale = np.maximum(1.0, np.linalg.norm(homogeneous, axis=1) * np.linalg.norm(matrix[2]))
    if np.any(np.abs(denominators) <= 1e-12 * scale):
        raise ValueError("A point lies at or too close to the homography horizon; projection is undefined.")
    output = projected[:, :2] / denominators[:, None]
    if not np.isfinite(output).all():
        raise ValueError("Homography projection produced nonfinite coordinates.")
    return output


def _normalize(points):
    center = points.mean(axis=0)
    distance = np.linalg.norm(points - center, axis=1).mean()
    if distance <= 1e-12:
        raise ValueError("Calibration corners are degenerate.")
    scale = math.sqrt(2.0) / distance
    transform = np.asarray([[scale, 0.0, -scale * center[0]],
                            [0.0, scale, -scale * center[1]], [0.0, 0.0, 1.0]])
    return _project(transform, points), transform


def _homography(source, destination):
    src, src_transform = _normalize(source)
    dst, dst_transform = _normalize(destination)
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.extend([[-x, -y, -1, 0, 0, 0, u*x, u*y, u],
                     [0, 0, 0, -x, -y, -1, v*x, v*y, v]])
    design = np.asarray(rows, dtype=np.float64)
    _, singular_values, vectors = np.linalg.svd(design, full_matrices=True)
    condition = float(singular_values[0] / singular_values[-1])
    if not math.isfinite(condition) or condition > 1e8:
        raise ValueError("Calibration geometry is ill conditioned; select clearly separated court corners.")
    normalized = vectors[-1].reshape(3, 3)
    normalized_condition = float(np.linalg.cond(normalized))
    if not math.isfinite(normalized_condition) or normalized_condition > 1e10:
        raise ValueError("Calibration homography is nearly singular.")
    matrix = np.linalg.inv(dst_transform) @ normalized @ src_transform
    if abs(matrix[2, 2]) <= 1e-14 * np.linalg.norm(matrix):
        raise ValueError("Calibration places the image origin on the homography horizon.")
    matrix /= matrix[2, 2]
    inverse = np.linalg.inv(matrix)
    inverse /= inverse[2, 2]
    return matrix, inverse, condition, normalized_condition


def _validate_corners(corners, width, height):
    values = _points(corners)
    if values.shape != (4, 2):
        raise ValueError("Calibration requires exactly four corners in far-left, far-right, near-right, near-left order.")
    if np.any(values[:, 0] < 0) or np.any(values[:, 0] > width - 1) or np.any(values[:, 1] < 0) or np.any(values[:, 1] > height - 1):
        raise ValueError("Calibration corners must lie within source image pixel bounds.")
    edges = np.roll(values, -1, axis=0) - values
    next_edges = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    tolerance = max(1e-8, width * height * 1e-10)
    if np.any(crosses <= tolerance):
        raise ValueError("Corners must form a strictly convex noncrossing polygon in the stated clockwise image order.")
    area = 0.5 * np.sum(values[:, 0] * np.roll(values[:, 1], -1) - values[:, 1] * np.roll(values[:, 0], -1))
    if area <= max(1e-6, width * height * 1e-7):
        raise ValueError("Court corner polygon has a degenerate area.")
    if not (values[0, 0] < values[1, 0] and values[3, 0] < values[2, 0]
            and values[:2, 1].mean() < values[2:, 1].mean()):
        raise ValueError("Corner order must be far-left, far-right, near-right, near-left for a rear court view.")
    return values.copy()


@dataclass
class CourtCalibration:
    corners_px: np.ndarray
    source_dimensions: tuple[int, int]
    homography: np.ndarray
    inverse_homography: np.ndarray
    method: str
    reference_frame: int
    metadata: dict
    diagnostics: dict

    def image_to_court(self, points):
        return _project(self.homography, points)

    def court_to_image(self, points):
        return _project(self.inverse_homography, points)

    def to_dict(self):
        return {
            "schema_version": 1, "source_dimensions": list(self.source_dimensions),
            "corners_px": self.corners_px.tolist(), "corner_order": list(CORNER_ORDER),
            "court_dimensions_m": [COURT_WIDTH_M, COURT_LENGTH_M],
            "singles_width_m": SINGLE_WIDTH_M, "method": self.method,
            "reference_frame": self.reference_frame, "metadata": self.metadata.copy(),
            "homography": self.homography.tolist(), "inverse_homography": self.inverse_homography.tolist(),
            "diagnostics": self.diagnostics.copy(),
            "coordinate_convention": "origin at far-left outer doubles corner; x across court; y towards near baseline; meters",
        }


def create_calibration(corners, image_width, image_height, *, method="manual", reference_frame=0, metadata=None):
    width, height = _dimensions(image_width, image_height)
    source = _validate_corners(corners, width, height)
    if not isinstance(method, str) or not method.strip():
        raise ValueError("Calibration method must be a nonempty string.")
    if isinstance(reference_frame, bool) or not isinstance(reference_frame, (int, np.integer)) or reference_frame < 0:
        raise ValueError("Calibration reference frame must be a nonnegative integer.")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("Calibration metadata must be a JSON object.")
    try:
        metadata = json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Calibration metadata must be serializable without nonfinite values.") from error
    matrix, inverse, design_condition, normalized_condition = _homography(source, COURT_CORNERS_M)
    fit_error_m = float(np.max(np.linalg.norm(_project(matrix, source) - COURT_CORNERS_M, axis=1)))
    samples = np.vstack([source, (source + np.roll(source, -1, axis=0)) / 2, source.mean(axis=0)])
    roundtrip_error_px = float(np.max(np.linalg.norm(_project(inverse, _project(matrix, samples)) - samples, axis=1)))
    if fit_error_m > 1e-7 or roundtrip_error_px > 1e-5:
        raise ValueError("Calibration failed numerical fit or roundtrip checks.")
    diagnostics = {
        "solver": "normalized direct linear transform with SVD",
        "normalized_dlt_condition": design_condition,
        "normalized_homography_condition": normalized_condition,
        "max_corner_fit_error_m": fit_error_m,
        "max_roundtrip_error_px": roundtrip_error_px,
        "physical_accuracy_status": "not measured; an exact fit of four supplied corners is not independent calibration validation",
    }
    return CourtCalibration(source, (width, height), matrix, inverse, method.strip(), int(reference_frame), metadata, diagnostics)


def load_calibration(path, image_width, image_height):
    try:
        with Path(path).open("r", encoding="utf-8-sig") as source:
            record = json.load(source)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read court calibration {path}: {error}") from error
    if not isinstance(record, dict) or "source_dimensions" not in record or "corners_px" not in record:
        raise ValueError("Calibration file requires source_dimensions and corners_px.")
    dimensions = record["source_dimensions"]
    if not isinstance(dimensions, (list, tuple)) or len(dimensions) != 2:
        raise ValueError("Calibration source_dimensions must contain width and height.")
    saved_dimensions = _dimensions(*dimensions)
    requested_dimensions = _dimensions(image_width, image_height)
    if saved_dimensions != requested_dimensions:
        raise ValueError(f"Calibration dimensions {saved_dimensions} do not match this video's {requested_dimensions}; recalibrate the camera.")
    if record.get("corner_order", list(CORNER_ORDER)) != list(CORNER_ORDER):
        raise ValueError("Calibration file has an unsupported court corner order.")
    return create_calibration(record["corners_px"], *requested_dimensions,
                              method=record.get("method", "manual"),
                              reference_frame=record.get("reference_frame", 0), metadata=record.get("metadata", {}))
