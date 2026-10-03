"""Conservative white-line proposal for a green court in a fixed rear view.

This heuristic is intentionally limited. Its score is line evidence, not a
calibrated accuracy probability. Every returned candidate remains unconfirmed.
"""
from __future__ import annotations

import cv2
import numpy as np

from .court import create_calibration


def _line(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    vx, vy, x, y = np.asarray(cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)).reshape(-1)
    coefficients = np.asarray([-vy, vx, vy*x - vx*y], dtype=np.float64)
    return coefficients / np.linalg.norm(coefficients[:2])


def _intersection(first, second):
    point = np.cross(first, second)
    if abs(point[2]) < 1e-9:
        raise ValueError("Court line candidates are nearly parallel; use manual corner calibration.")
    return point[:2] / point[2]


def _x_at_y(line, y):
    if abs(line[0]) < 1e-9:
        raise ValueError("A court side candidate is horizontal.")
    return -(line[1]*y + line[2]) / line[0]


def _white_support(mask, first, second):
    samples = np.linspace(first, second, max(40, round(np.linalg.norm(second-first))))
    x = np.clip(np.rint(samples[:, 0]).astype(int), 0, mask.shape[1]-1)
    y = np.clip(np.rint(samples[:, 1]).astype(int), 0, mask.shape[0]-1)
    return float(np.mean(mask[y, x] > 0))


def detect_court(rgb):
    """Return {corners_px, score, evidence, warnings}, or reject ambiguous evidence."""
    values = np.asarray(rgb)
    if values.ndim != 3 or values.shape[2] != 3 or values.dtype != np.uint8 or min(values.shape[:2]) < 100:
        raise ValueError("Automatic court detection requires a uint8 RGB image at least 100 pixels in each dimension.")
    height, width = values.shape[:2]
    hsv = cv2.cvtColor(values, cv2.COLOR_RGB2HSV)
    green = cv2.inRange(hsv, np.asarray([30, 45, 30]), np.asarray([95, 255, 255]))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(green)
    components = sorted(range(1, count), key=lambda index: int(stats[index, cv2.CC_STAT_AREA]), reverse=True)
    if not components or stats[components[0], cv2.CC_STAT_AREA] < width*height*0.10:
        raise ValueError("No large green playing area was found; use manual court corners for this camera or floor.")
    selected = components[0]
    area = int(stats[selected, cv2.CC_STAT_AREA])
    if len(components) > 1 and stats[components[1], cv2.CC_STAT_AREA] > area*0.45:
        raise ValueError("Several green playing areas are plausible; select the court manually.")
    gx, gy, gw, gh, _ = stats[selected]
    if gw < width*0.35 or gh < height*0.25 or centers[selected, 1] < height*0.40:
        raise ValueError("Green playing area does not resemble a full rear court view; use manual calibration.")
    component = (labels == selected).astype(np.uint8)*255
    neighborhood = cv2.dilate(component, np.ones((17, 17), np.uint8))
    white = cv2.inRange(hsv, np.asarray([0, 0, 145]), np.asarray([179, 100, 255]))
    white = cv2.bitwise_and(white, neighborhood)
    segments = cv2.HoughLinesP(white, 1, np.pi/1800, threshold=max(35, round(height*0.07)),
                               minLineLength=max(80, round(width*0.12)), maxLineGap=max(8, round(width*0.01)))
    if segments is None:
        raise ValueError("Insufficient white court line evidence; use manual corner calibration.")
    segments = np.asarray(segments).reshape(-1, 4)
    horizontal = []
    for x1, y1, x2, y2 in segments:
        if abs(y2-y1) <= max(2, abs(x2-x1)*0.06) and abs(x2-x1) >= width*0.18:
            horizontal.append((float((y1+y2)/2), [x1, y1, x2, y2]))
    horizontal.sort(key=lambda entry: entry[0])
    groups = []
    for y, segment in horizontal:
        if y < gy + height*0.006 or y > gy+gh-height*0.006:
            continue
        if not groups or y - np.mean([entry[0] for entry in groups[-1]]) > max(4, height*0.008):
            groups.append([])
        groups[-1].append((y, segment))
    if len(groups) < 2:
        raise ValueError("Both outer baseline candidates are not visible; select all four corners manually.")
    far_group, near_group = groups[0], groups[-1]
    far_y = float(np.mean([entry[0] for entry in far_group]))
    near_y = float(np.mean([entry[0] for entry in near_group]))
    if near_y-far_y < height*0.25:
        raise ValueError("Baseline candidates are too close or ambiguous; use manual calibration.")
    far_line = _line([segment[j:j+2] for _, segment in far_group for j in (0, 2)])
    near_line = _line([segment[j:j+2] for _, segment in near_group for j in (0, 2)])
    middle_y = (far_y + near_y)/2
    left_candidates, right_candidates = [], []
    for x1, y1, x2, y2 in segments:
        if abs(y2-y1) < max(height*0.18, (near_y-far_y)*0.35):
            continue
        slope = (x2-x1)/(y2-y1)
        if not 0.05 < abs(slope) < 1.0:
            continue
        line = _line([[x1, y1], [x2, y2]])
        middle_x = _x_at_y(line, middle_y)
        if slope < 0 and width*0.08 < middle_x < width*0.49:
            left_candidates.append((middle_x, line))
        elif slope > 0 and width*0.51 < middle_x < width*0.92:
            right_candidates.append((middle_x, line))
    if not left_candidates or not right_candidates:
        raise ValueError("Outer left and right sidelines are not sufficiently supported; use manual calibration.")
    left_line = min(left_candidates, key=lambda entry: entry[0])[1]
    right_line = max(right_candidates, key=lambda entry: entry[0])[1]
    corners = np.asarray([_intersection(far_line, left_line), _intersection(far_line, right_line),
                          _intersection(near_line, right_line), _intersection(near_line, left_line)])
    calibration = create_calibration(corners, width, height, method="auto_proposal")
    supported_mask = cv2.dilate(white, np.ones((7, 7), np.uint8))
    supports = {name: _white_support(supported_mask, corners[index], corners[(index+1)%4])
                for index, name in enumerate(("far_baseline", "right_sideline", "near_baseline", "left_sideline"))}
    if min(supports.values()) < 0.55:
        raise ValueError("Proposed outer court lines have weak white line support; use manual calibration.")
    score = float(min(0.95, 0.10 + 0.85*min(supports.values())))
    return {
        "corners_px": corners.tolist(), "score": score,
        "evidence": {"green_component_area_fraction": area/(width*height),
                     "green_component_bbox_px": [int(gx), int(gy), int(gw), int(gh)],
                     "white_line_support": supports, "hough_segment_count": len(segments),
                     "horizontal_line_group_count": len(groups),
                     "score_semantics": "uncalibrated geometric line support; not a probability of correct court geometry",
                     "numerical_diagnostics": calibration.diagnostics},
        "warnings": ["Automatic geometry is an unconfirmed proposal. Inspect outer doubles line intersections and independent service lines.",
                     "This heuristic assumes one dominant green court, visible white boundaries and a fixed rear camera; other floors or views require manual calibration.",
                     "Four fitted corners alone do not establish physical distance accuracy."],
    }
