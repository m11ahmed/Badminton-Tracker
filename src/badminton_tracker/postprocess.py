"""Conservative short gap filling and transparent tracking quality summaries."""
from dataclasses import replace
import math

from .types import Detection


def validate_detections(detections: list[Detection]) -> None:
    previous_time = None
    for index, row in enumerate(detections):
        if row.frame_index != index:
            raise ValueError("Detection indices must be consecutive and start at zero.")
        if not math.isfinite(row.timestamp_s) or row.timestamp_s < 0:
            raise ValueError("Detection timestamps must be finite and nonnegative.")
        if previous_time is not None and row.timestamp_s <= previous_time:
            raise ValueError("Detection timestamps must strictly increase.")
        previous_time = row.timestamp_s
        if (row.x is None) != (row.y is None):
            raise ValueError("A detection must have both x and y coordinates or neither.")
        for name in ("x", "y", "confidence", "raw_x", "raw_y", "raw_confidence"):
            value = getattr(row, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"Detection {name} must be finite or null.")
        if row.source == "detected" and row.x is None:
            raise ValueError("Detected rows require coordinates.")
        if row.source == "missing" and row.x is not None:
            raise ValueError("Missing rows cannot contain accepted coordinates; use raw_x and raw_y for rejected peaks.")


def interpolate_short_gaps(detections: list[Detection], max_gap_frames: int = 2) -> list[Detection]:
    """Fill short gaps between reliable endpoints using elapsed source time.

    There is no extrapolation. Raw detector coordinates and score are preserved.
    An interpolated coordinate has no detector confidence, and remains flagged.
    """
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be zero or greater.")
    validate_detections(detections)
    rows = [replace(row) for row in detections]
    if not max_gap_frames:
        return rows
    index = 0
    while index < len(rows):
        if rows[index].source != "missing":
            index += 1
            continue
        start = index
        while index < len(rows) and rows[index].source == "missing":
            index += 1
        length = index - start
        if start == 0 or index == len(rows) or length > max_gap_frames:
            continue
        left, right = rows[start - 1], rows[index]
        if left.source != "detected" or right.source != "detected" or left.low_confidence or right.low_confidence:
            continue
        elapsed = right.timestamp_s - left.timestamp_s
        for gap_index in range(start, index):
            fraction = (rows[gap_index].timestamp_s - left.timestamp_s) / elapsed
            rows[gap_index] = replace(
                rows[gap_index],
                x=left.x + fraction * (right.x - left.x),
                y=left.y + fraction * (right.y - left.y),
                confidence=None,
                source="interpolated",
                low_confidence=True,
            )
    return rows


def tracking_summary(detections: list[Detection]) -> dict:
    validate_detections(detections)
    total = len(detections)
    counts = {source: sum(row.source == source for row in detections) for source in ("detected", "missing", "interpolated")}
    scores = [row.raw_confidence for row in detections if row.raw_confidence is not None]
    raw_gap_lengths = []
    current = 0
    for row in detections:
        if row.source != "detected":
            current += 1
        elif current:
            raw_gap_lengths.append(current)
            current = 0
    if current:
        raw_gap_lengths.append(current)
    return {
        "frames_analyzed": total,
        "source_counts": counts,
        "observed_fraction": counts["detected"] / total if total else 0.0,
        "reliable_observed_fraction": sum(row.source == "detected" and not row.low_confidence for row in detections) / total if total else 0.0,
        "available_fraction_after_interpolation": (counts["detected"] + counts["interpolated"]) / total if total else 0.0,
        "low_confidence_frames": sum(row.low_confidence for row in detections),
        "raw_gap_count": len(raw_gap_lengths),
        "longest_raw_gap_frames": max(raw_gap_lengths, default=0),
        "mean_raw_heatmap_score": sum(scores) / len(scores) if scores else None,
        "time_first_frame_s": detections[0].timestamp_s if total else None,
        "time_last_frame_s": detections[-1].timestamp_s if total else None,
        "score_interpretation": "uncalibrated model heatmap peak score; not a probability or validated accuracy measure",
        "evaluation": "coverage and model scores only; accuracy requires independent labeled video",
        "coordinate_system": "source image pixels; no court calibration or physical speed estimate in M1",
    }