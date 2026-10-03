"""Project observed ankle references onto a calibrated floor without filling gaps."""
from copy import deepcopy
import math

import numpy as np

from .court import COURT_WIDTH_M, COURT_LENGTH_M, SINGLE_WIDTH_M


def project_players_on_court(frames, calibration, *, mode="singles"):
    if mode not in {"singles", "doubles"}:
        raise ValueError("Court mode must be singles or doubles.")
    projected = deepcopy(frames)
    unconfirmed = calibration.method.startswith("auto") and not calibration.metadata.get("manually_reviewed", False)
    observations = available = missing = outside = low_pose = 0
    x_margin = (COURT_WIDTH_M - SINGLE_WIDTH_M) / 2
    for frame in projected:
        frame_flags = []
        for player in frame["players"]:
            observations += 1
            player["court_position"] = None
            flags = []
            anchor = player.get("ankle_midpoint")
            if anchor is None:
                flags.append("ankle_midpoint_unavailable")
            else:
                try:
                    point = np.asarray(anchor, dtype=np.float64)
                    if point.shape != (2,) or not np.isfinite(point).all():
                        raise ValueError("Invalid ankle reference.")
                    x, y = calibration.image_to_court(point.reshape(1, 2))[0]
                    if not math.isfinite(x) or not math.isfinite(y):
                        raise ValueError("Invalid projection.")
                    in_doubles = bool(-1e-9 <= x <= COURT_WIDTH_M + 1e-9 and -1e-9 <= y <= COURT_LENGTH_M + 1e-9)
                    in_singles = bool(x_margin - 1e-9 <= x <= COURT_WIDTH_M - x_margin + 1e-9 and -1e-9 <= y <= COURT_LENGTH_M + 1e-9)
                    flags.append("approximate_floor_projection")
                    if player.get("low_confidence"):
                        flags.append("low_confidence_pose")
                        low_pose += 1
                    if unconfirmed:
                        flags.append("unconfirmed_court_calibration")
                    if not (in_singles if mode == "singles" else in_doubles):
                        flags.append("outside_active_court")
                        outside += 1
                    player["court_position"] = {
                        "x_m": float(x), "y_m": float(y),
                        "in_doubles_court": in_doubles, "in_singles_court": in_singles,
                        "quality_flags": list(flags),
                    }
                    available += 1
                except (ValueError, ArithmeticError, np.linalg.LinAlgError):
                    flags.append("court_projection_unavailable")
            player["court_flags"] = flags
            if player["court_position"] is None:
                missing += 1
                frame_flags.append("court_position_unavailable")
        if unconfirmed:
            frame_flags.append("unconfirmed_court_calibration")
        frame["court_quality_flags"] = sorted(set(frame_flags))
    summary = {
        "frames_analyzed": len(projected), "player_observations": observations,
        "positions_available": available, "positions_missing": missing,
        "positions_outside_active_court": outside,
        "positions_with_low_confidence_pose": low_pose,
        "mode": mode, "court_width_m": COURT_WIDTH_M, "court_length_m": COURT_LENGTH_M,
        "origin": "far-left doubles corner; x increases toward far-right, y toward near baseline",
        "position_source": "midpoint of two observed ankle keypoints projected onto a planar floor",
        "missing_policy": "no player interpolation or bounding box fallback",
        "calibration_status": "unconfirmed_automatic_proposal" if unconfirmed else "manual_corner_calibration",
        "accuracy_status": "not measured against independent floor positions or surveyed camera calibration",
        "limitations": [
            "Ankle midpoint is approximate; jumping, perspective, pose error and lens distortion can move it away from a true floor reference.",
            "Four corner fit residuals describe mathematical consistency, not real-world measurement accuracy.",
            "Fixed camera and correct outer doubles corners are required; cuts or camera movement require recalibration.",
            "Shuttle positions are not projected onto the floor because the shuttle is airborne.",
            "Outside-court coordinates remain signed estimates and are never clamped into the court.",
        ],
    }
    return projected, summary
