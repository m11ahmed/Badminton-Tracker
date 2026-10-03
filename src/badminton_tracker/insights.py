"""Quality-aware observed movement estimates and explicit human review context.

Player metres require a reviewed floor calibration. Shuttle speed is IMAGE
motion in pixels per second, never an airborne shuttle's physical speed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import math

import numpy as np

from .court import COURT_WIDTH_M, COURT_LENGTH_M

POLICY = {
    "smoothing_window_s": 0.20,
    "maximum_player_speed_m_s": 12.0,
    "minimum_shuttle_confidence": 0.5,
    "stationary_segment_extent_m": 0.08,
    "maximum_interval_s_minimum": 0.25,
    "maximum_interval_nominal_multiplier": 3.0,
    "heatmap_shape": [54, 24],
    "position_margin_m": 1.0,
    "shuttle_jump_review_fraction_of_image_diagonal": 0.10,
}
CUT_FLAGS = {"camera_cut", "scene_cut", "cut", "camera_motion", "calibration_invalid"}
INVALID_POSITION_FLAGS = CUT_FLAGS | {
    "low_confidence_pose", "unconfirmed_court_calibration", "court_projection_unavailable",
    "ankle_midpoint_unavailable", "player_selection_unverified",
}


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def _record(value):
    return asdict(value) if is_dataclass(value) else value


def _validate_timeline(records, name):
    previous = None
    seen = set()
    for item in records:
        index, timestamp = item.get("frame_index"), item.get("timestamp_s")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in seen:
            raise ValueError(f"{name} frame indices must be unique nonnegative integers.")
        if not _finite(timestamp) or timestamp < 0:
            raise ValueError(f"{name} timestamps must be finite nonnegative seconds.")
        if previous is not None and (index <= previous[0] or timestamp <= previous[1]):
            raise ValueError(f"{name} frames and source timestamps must strictly increase.")
        previous = (index, timestamp)
        seen.add(index)


def validate_review(review, player_frames):
    """Validate optional JSON annotations; absence never invents identity or hits."""
    if review is None:
        return {"schema_version": 1, "identity_map": {}, "rallies": [], "events": [], "recovery_targets": []}
    if not isinstance(review, dict):
        raise ValueError("Review must be a JSON object.")
    allowed = {"schema_version", "source_video_sha256", "identity_map", "rallies", "events", "recovery_targets"}
    if set(review) - allowed:
        raise ValueError(f"Unknown review fields: {sorted(set(review) - allowed)}")
    if isinstance(review.get("schema_version", 1), bool) or review.get("schema_version", 1) != 1:
        raise ValueError("Review schema_version must be 1.")
    output = deepcopy(review)
    output["schema_version"] = 1
    if "source_video_sha256" in output:
        digest = output["source_video_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ValueError("Review source_video_sha256 must contain 64 hexadecimal characters.")
        output["source_video_sha256"] = digest.lower()
    for field, default in (("identity_map", {}), ("rallies", []), ("events", []), ("recovery_targets", [])):
        output.setdefault(field, default)
        if not isinstance(output[field], type(default)):
            raise ValueError(f"Review {field} has the wrong type.")
    frame_indices = {frame["frame_index"] for frame in player_frames}
    observed_ids = {str(player["track_id"]) for frame in player_frames for player in frame["players"]}
    labels = set()
    for track_id, identity in output["identity_map"].items():
        if not isinstance(track_id, str) or track_id not in observed_ids:
            raise ValueError("Identity mapping references an unknown raw tracker ID.")
        if not isinstance(identity, dict) or set(identity) - {"player_label", "side", "team"}:
            raise ValueError("Identity mapping values require player_label and optional side/team.")
        label = identity.get("player_label")
        if not isinstance(label, str) or not label.strip() or len(label) > 80 or any(ord(c) < 32 for c in label):
            raise ValueError("Player labels must be nonempty printable strings of at most 80 characters.")
        identity["player_label"] = label.strip()
        labels.add(label.strip())
        if "side" in identity and identity["side"] not in {"near", "far"}:
            raise ValueError("Reviewed side must be near or far.")
        if "team" in identity and (not isinstance(identity["team"], str) or not identity["team"].strip()):
            raise ValueError("Reviewed team must be a nonempty string.")
    identities_by_label = defaultdict(list)
    for raw_id, identity in output["identity_map"].items():
        identities_by_label[identity["player_label"]].append(identity)
    for identities in identities_by_label.values():
        if len({i.get("side") for i in identities if i.get("side") is not None}) > 1 or len({i.get("team") for i in identities if i.get("team") is not None}) > 1:
            raise ValueError("Fragments mapped to one athlete must agree on side and team.")
    for frame in player_frames:
        mapped = [output["identity_map"][str(p["track_id"])]["player_label"] for p in frame["players"] if str(p["track_id"]) in output["identity_map"]]
        if len(set(mapped)) != len(mapped):
            raise ValueError("Two simultaneous tracker IDs cannot map to the same player label.")
    def frame_exists(value):
        return isinstance(value, int) and not isinstance(value, bool) and value in frame_indices
    rally_ids, previous_end = set(), -1
    for rally in output["rallies"]:
        if not isinstance(rally, dict) or set(rally) - {"id", "start_frame", "end_frame", "notes", "outcome", "winner_label"}:
            raise ValueError("A rally requires id/start_frame/end_frame and optional notes/outcome/winner_label.")
        if not isinstance(rally.get("id"), str) or not rally["id"].strip() or rally["id"] in rally_ids:
            raise ValueError("Reviewed rally IDs must be unique nonempty strings.")
        start, end = rally.get("start_frame"), rally.get("end_frame")
        if not frame_exists(start) or not frame_exists(end) or start > end or start <= previous_end:
            raise ValueError("Reviewed rallies need existing, ordered, nonoverlapping frame boundaries.")
        previous_end = end
        rally_ids.add(rally["id"])
        for field in ("notes", "outcome", "winner_label"):
            if field in rally and not isinstance(rally[field], str):
                raise ValueError(f"Rally {field} must be text.")
        if "winner_label" in rally and rally["winner_label"] not in labels:
            raise ValueError("Rally winner must reference a reviewed player label.")
    for event in output["events"]:
        if not isinstance(event, dict) or set(event) - {"frame_index", "kind", "player_label", "note"}:
            raise ValueError("Review events require frame_index/kind and optional player_label/note.")
        if not frame_exists(event.get("frame_index")) or event.get("kind") not in {"contact", "outcome", "note"}:
            raise ValueError("Review event must name an existing frame and contact/outcome/note kind.")
        if "player_label" in event and (not isinstance(event["player_label"], str) or event["player_label"] not in labels):
            raise ValueError("Review event player_label is unknown.")
        if "note" in event and not isinstance(event["note"], str):
            raise ValueError("Review event note must be text.")
    for target in output["recovery_targets"]:
        if not isinstance(target, dict) or set(target) - {"player_label", "contact_frame", "target_x_m", "target_y_m", "radius_m", "until_frame"}:
            raise ValueError("Recovery target has unsupported fields.")
        if not isinstance(target.get("player_label"), str) or target["player_label"] not in labels or not frame_exists(target.get("contact_frame")):
            raise ValueError("Recovery target needs a reviewed player and an existing contact frame.")
        target.setdefault("radius_m", 0.5)
        target.setdefault("until_frame", max(frame_indices))
        if not frame_exists(target["until_frame"]) or target["until_frame"] < target["contact_frame"]:
            raise ValueError("Recovery end frame must follow its contact frame.")
        if not all(_finite(target.get(key)) for key in ("target_x_m", "target_y_m", "radius_m")):
            raise ValueError("Recovery target coordinates and radius must be finite numbers.")
        if not 0 <= target["target_x_m"] <= COURT_WIDTH_M or not 0 <= target["target_y_m"] <= COURT_LENGTH_M or not 0 < target["radius_m"] <= 3:
            raise ValueError("Recovery target must be on court with radius greater than zero and at most three metres.")
    return output


def _zone(x, y):
    if not 0 <= x <= COURT_WIDTH_M or not 0 <= y <= COURT_LENGTH_M:
        return "outside_court"
    side = "far" if y < COURT_LENGTH_M / 2 else "near"
    depth = y if side == "far" else COURT_LENGTH_M - y
    band = "back" if depth < COURT_LENGTH_M / 6 else "mid" if depth < COURT_LENGTH_M / 3 else "front"
    lateral = "left" if x < COURT_WIDTH_M / 3 else "center" if x < 2 * COURT_WIDTH_M / 3 else "right"
    return f"{side}_{band}_{lateral}"


def _smooth_segment(rows):
    points = np.array([[row["x_m"], row["y_m"]] for row in rows], dtype=float)
    if len(rows) < 3:
        return points
    if np.max(np.linalg.norm(points - np.median(points, axis=0), axis=1)) * 2 <= POLICY["stationary_segment_extent_m"]:
        return np.tile(np.median(points, axis=0), (len(rows), 1))
    times = np.array([row["timestamp_s"] for row in rows])
    smoothed = points.copy()
    half_window = POLICY["smoothing_window_s"] / 2
    for index, center in enumerate(times):
        # Include three nearby observations when the source cadence is sparse.
        neighbors = np.flatnonzero(np.abs(times - center) <= half_window + 1e-12)
        if len(neighbors) < 3:
            neighbors = np.argsort(np.abs(times - center))[:min(3, len(rows))]
        offsets = times[neighbors] - center
        design = np.column_stack((np.ones(len(neighbors)), offsets))
        weights = 1 / (1 + (offsets / max(half_window, 1e-9)) ** 2)
        fit = np.linalg.lstsq(design * np.sqrt(weights)[:, None], points[neighbors] * np.sqrt(weights)[:, None], rcond=None)[0]
        smoothed[index] = fit[0]
    return smoothed


def _max_interval(records):
    intervals = [b["timestamp_s"] - a["timestamp_s"] for a, b in zip(records, records[1:]) if b["frame_index"] == a["frame_index"] + 1]
    nominal = float(np.median(intervals)) if intervals else 0
    return max(POLICY["maximum_interval_s_minimum"], POLICY["maximum_interval_nominal_multiplier"] * nominal)


def _player_motion(frames, calibration, review):
    raw_rows = []
    groups = defaultdict(list)
    metadata = getattr(calibration, "metadata", {}) if calibration is not None else {}
    method = getattr(calibration, "method", "") if calibration is not None else ""
    confirmed = calibration is not None and not (method.startswith("auto") and not metadata.get("manually_reviewed", False))
    max_interval = _max_interval(frames)
    for frame in frames:
        frame_flags = set(frame.get("flags", [])) | set(frame.get("court_quality_flags", []))
        for player in frame["players"]:
            position = player.get("court_position")
            flags = set(player.get("court_flags", [])) | frame_flags
            if position:
                flags.update(position.get("quality_flags", []))
            if player.get("low_confidence", False):
                flags.add("low_confidence_pose")
            if not confirmed:
                flags.add("unconfirmed_court_calibration" if calibration else "court_calibration_unavailable")
            x, y = (position.get("x_m"), position.get("y_m")) if position else (None, None)
            valid = bool(confirmed and _finite(x) and _finite(y) and not flags.intersection(INVALID_POSITION_FLAGS))
            if not (_finite(x) and _finite(y)):
                flags.add("court_position_unavailable")
            margin = POLICY["position_margin_m"]
            if valid and not (-margin <= x <= COURT_WIDTH_M + margin and -margin <= y <= COURT_LENGTH_M + margin):
                valid = False
                flags.add("position_outside_reasonable_floor_bounds")
            identity = review["identity_map"].get(str(player["track_id"]), {})
            row = {
                "frame_index": frame["frame_index"], "timestamp_s": float(frame["timestamp_s"]),
                "track_id": player["track_id"], "player_label": identity.get("player_label"),
                "side": identity.get("side"), "team": identity.get("team"),
                "x_m": float(x) if _finite(x) else None, "y_m": float(y) if _finite(y) else None,
                "valid_position": valid, "smoothed_x_m": None, "smoothed_y_m": None,
                "interval_start_frame": None, "interval_start_timestamp_s": None,
                "interval_duration_s": None, "distance_m": None, "speed_m_s": None,
                "zone": _zone(x, y) if valid else None,
                "quality_flags": sorted(flags),
            }
            groups[player["track_id"]].append(row)
            raw_rows.append(row)
    summaries = []
    for track_id, rows in sorted(groups.items()):
        segments, segment = [], []
        for row in rows:
            if not row["valid_position"]:
                if segment:
                    segments.append(segment)
                    segment = []
                continue
            if segment:
                previous = segment[-1]
                dt = row["timestamp_s"] - previous["timestamp_s"]
                raw_speed = math.hypot(row["x_m"] - previous["x_m"], row["y_m"] - previous["y_m"]) / dt
                reason = ("observation_gap" if row["frame_index"] != previous["frame_index"] + 1 or dt > max_interval
                          else "implausible_position_jump" if raw_speed > POLICY["maximum_player_speed_m_s"] else None)
                if reason:
                    row["quality_flags"].append(reason)
                    segments.append(segment)
                    segment = []
            segment.append(row)
        if segment:
            segments.append(segment)
        heatmap = np.zeros(POLICY["heatmap_shape"], dtype=float)
        zone_seconds = defaultdict(float)
        accepted = []
        for segment in segments:
            smoothed = _smooth_segment(segment)
            for row, point in zip(segment, smoothed):
                row["smoothed_x_m"], row["smoothed_y_m"] = map(float, point)
            for previous, current in zip(segment, segment[1:]):
                dt = current["timestamp_s"] - previous["timestamp_s"]
                distance = math.hypot(current["smoothed_x_m"] - previous["smoothed_x_m"], current["smoothed_y_m"] - previous["smoothed_y_m"])
                speed = distance / dt
                if speed > POLICY["maximum_player_speed_m_s"]:
                    current["quality_flags"].append("implausible_smoothed_speed")
                    continue
                current.update(interval_start_frame=previous["frame_index"], interval_start_timestamp_s=previous["timestamp_s"], interval_duration_s=dt, distance_m=distance, speed_m_s=speed)
                accepted.append(current)
                for row in (previous, current):
                    x, y = row["x_m"], row["y_m"]
                    zone_seconds[_zone(x, y)] += dt / 2
                    if 0 <= x <= COURT_WIDTH_M and 0 <= y <= COURT_LENGTH_M:
                        iy = min(heatmap.shape[0] - 1, int(y / COURT_LENGTH_M * heatmap.shape[0]))
                        ix = min(heatmap.shape[1] - 1, int(x / COURT_WIDTH_M * heatmap.shape[1]))
                        heatmap[iy, ix] += dt / 2
        total_time = sum(row["interval_duration_s"] for row in accepted)
        distance = sum(row["distance_m"] for row in accepted) if accepted else None
        identity = review["identity_map"].get(str(track_id), {})
        summaries.append({
            "track_id": track_id, "player_label": identity.get("player_label"),
            "side": identity.get("side"), "team": identity.get("team"),
            "first_frame": rows[0]["frame_index"], "last_frame": rows[-1]["frame_index"],
            "observed_frames": len(rows), "valid_position_frames": sum(row["valid_position"] for row in rows),
            "motion_intervals": len(accepted), "excluded_motion_intervals": max(0, len(rows) - 1 - len(accepted)),
            "distance_m": distance, "valid_motion_time_s": total_time,
            "average_speed_m_s": distance / total_time if total_time else None,
            "max_speed_m_s": max((row["speed_m_s"] for row in accepted), default=None),
            "coverage": len(accepted) / max(1, len(frames) - 1),
            "zones": dict(sorted(zone_seconds.items())),
            "heatmap": {"values_s": heatmap.tolist(), "grid_shape": list(heatmap.shape), "bounds_m": [0, 0, COURT_WIDTH_M, COURT_LENGTH_M], "unit": "observed seconds"},
            "quality_flags": dict(Counter(flag for row in rows for flag in row["quality_flags"])),
            "identity_status": "manually_mapped_fragment" if identity else "unverified_tracker_fragment",
        })
    return raw_rows, summaries, confirmed, max_interval


def _shuttle_motion(detections, frames, calibration):
    frame_lookup = {frame["frame_index"]: frame for frame in frames}
    max_interval = _max_interval(detections)
    source_dimensions = getattr(calibration, "source_dimensions", None)
    rows, previous = [], None
    for detection in detections:
        flags = set()
        frame = frame_lookup.get(detection["frame_index"])
        if frame is not None:
            if abs(detection["timestamp_s"] - frame["timestamp_s"]) > 1e-6:
                raise ValueError("Shuttle and player source timestamps must align.")
            flags.update(set(frame.get("flags", [])).intersection(CUT_FLAGS))
        x, y = detection.get("x"), detection.get("y")
        confidence = detection.get("confidence")
        valid = detection.get("source") == "detected" and not detection.get("low_confidence", False) and _finite(confidence) and confidence >= POLICY["minimum_shuttle_confidence"] and _finite(x) and _finite(y) and not flags
        if detection.get("source") != "detected":
            flags.add("shuttle_not_observed")
        if detection.get("low_confidence", False) or (_finite(confidence) and confidence < POLICY["minimum_shuttle_confidence"]):
            flags.add("low_confidence_shuttle")
        if not _finite(confidence):
            flags.add("shuttle_confidence_unavailable")
        if not (_finite(x) and _finite(y)):
            flags.add("shuttle_position_unavailable")
        if valid and source_dimensions and not (0 <= x < source_dimensions[0] and 0 <= y < source_dimensions[1]):
            valid = False
            flags.add("shuttle_outside_image")
        row = {
            "frame_index": detection["frame_index"], "timestamp_s": float(detection["timestamp_s"]),
            "x_px": float(x) if _finite(x) else None, "y_px": float(y) if _finite(y) else None,
            "source": detection.get("source"), "valid_observation": bool(valid),
            "interval_start_frame": None, "interval_duration_s": None,
            "displacement_px": None, "image_speed_px_s": None,
            "physical_speed_km_h": None, "quality_flags": sorted(flags),
        }
        if valid and previous and previous["valid_observation"]:
            dt = row["timestamp_s"] - previous["timestamp_s"]
            if row["frame_index"] == previous["frame_index"] + 1 and dt <= max_interval:
                displacement = math.hypot(x - previous["x_px"], y - previous["y_px"])
                if source_dimensions:
                    diagonal = math.hypot(*source_dimensions)
                    if diagonal > 0 and displacement / diagonal >= POLICY["shuttle_jump_review_fraction_of_image_diagonal"]:
                        flags.add("large_image_jump_review")
                        row["quality_flags"] = sorted(flags)
                row.update(interval_start_frame=previous["frame_index"], interval_duration_s=dt, displacement_px=displacement, image_speed_px_s=displacement / dt)
            else:
                row["quality_flags"].append("observation_gap")
        rows.append(row)
        previous = row
    accepted = [row for row in rows if row["image_speed_px_s"] is not None]
    duration = sum(row["interval_duration_s"] for row in accepted)
    summary = {
        "observed_frames": sum(row["valid_observation"] for row in rows),
        "total_frames": len(rows), "valid_motion_intervals": len(accepted),
        "excluded_motion_intervals": max(0, len(rows) - 1 - len(accepted)),
        "valid_motion_time_s": duration,
        "average_image_speed_px_s": sum(row["displacement_px"] for row in accepted) / duration if duration else None,
        "max_image_speed_px_s": max((row["image_speed_px_s"] for row in accepted), default=None),
        "physical_speed_km_h": None,
        "physical_speed_status": "unavailable_from_single_camera_without_airborne_depth_and_validated_3d_calibration",
        "image_speed_unit": "source-image pixels per second",
        "quality_flags": dict(Counter(flag for row in rows for flag in row["quality_flags"])),
    }
    return rows, summary


def _doubles_spacing(frames, rows, mode):
    output = {"status": "unavailable", "reason": "Only applicable to doubles.", "frames": [], "pairs": []}
    if mode != "doubles":
        return output
    lookup = defaultdict(list)
    for row in rows:
        lookup[row["frame_index"]].append(row)
    for frame in frames:
        observed = lookup[frame["frame_index"]]
        if len(observed) != 4 or not all(row["valid_position"] and row["player_label"] and row["side"] and not {"implausible_position_jump", "implausible_smoothed_speed"}.intersection(row["quality_flags"]) for row in observed):
            continue
        sides = {side: [row for row in observed if row["side"] == side] for side in ("near", "far")}
        if any(len(players) != 2 for players in sides.values()):
            continue
        for side, partners in sides.items():
            first, second = partners
            if first["team"] and second["team"] and first["team"] != second["team"]:
                continue
            output["frames"].append({
                "frame_index": frame["frame_index"], "timestamp_s": frame["timestamp_s"], "side": side,
                "players": sorted([first["player_label"], second["player_label"]]),
                "spacing_m": math.hypot(first["x_m"] - second["x_m"], first["y_m"] - second["y_m"]),
                "lateral_spacing_m": abs(first["x_m"] - second["x_m"]),
                "depth_spacing_m": abs(first["y_m"] - second["y_m"]),
            })
    if output["frames"]:
        output.update(status="available", reason=None)
        groups = defaultdict(list)
        for row in output["frames"]:
            groups[(row["side"], tuple(row["players"]))].append(row)
        for (side, players), observations in groups.items():
            output["pairs"].append({"side": side, "players": list(players), "valid_frames": len(observations),
                                    "median_spacing_m": float(np.median([r["spacing_m"] for r in observations])),
                                    "minimum_spacing_m": min(r["spacing_m"] for r in observations),
                                    "maximum_spacing_m": max(r["spacing_m"] for r in observations)})
    else:
        output["reason"] = "Requires exactly four valid floor positions with reviewed identity and near/far side mapping; no inferred partnerships."
    return output


def _rally_summaries(review, frames, motion):
    if not frames:
        return []
    rallies = review["rallies"] or [{"id": "unsegmented_clip", "start_frame": frames[0]["frame_index"], "end_frame": frames[-1]["frame_index"]}]
    times = {frame["frame_index"]: frame["timestamp_s"] for frame in frames}
    output = []
    for rally in rallies:
        record = deepcopy(rally)
        record["boundary_source"] = "manual_review" if review["rallies"] else "whole_clip_not_automatically_classified_as_rally"
        record["duration_s"] = times[rally["end_frame"]] - times[rally["start_frame"]]
        groups = defaultdict(list)
        for row in motion:
            if row["interval_start_frame"] is not None and rally["start_frame"] <= row["interval_start_frame"] and row["frame_index"] <= rally["end_frame"]:
                groups[row["track_id"]].append(row)
        record["players"] = []
        for track_id, rows in sorted(groups.items()):
            duration = sum(row["interval_duration_s"] for row in rows)
            distance = sum(row["distance_m"] for row in rows)
            record["players"].append({"track_id": track_id, "player_label": rows[0]["player_label"],
                                       "distance_m": distance, "valid_motion_time_s": duration,
                                       "average_speed_m_s": distance / duration if duration else None})
        record["events"] = [deepcopy(event) for event in review["events"] if rally["start_frame"] <= event["frame_index"] <= rally["end_frame"]]
        output.append(record)
    return output


def _recovery(review, frames, rows):
    by_label_frame = {(row["player_label"], row["frame_index"]): row for row in rows if row["player_label"]}
    times = {frame["frame_index"]: frame["timestamp_s"] for frame in frames}
    results = []
    for target in review["recovery_targets"]:
        result = deepcopy(target)
        result.update(recovery_time_s=None, reached_frame=None, status="unavailable", reason=None)
        previous = None
        for index in range(target["contact_frame"], target["until_frame"] + 1):
            row = by_label_frame.get((target["player_label"], index))
            if row is None or not row["valid_position"] or {"implausible_position_jump", "implausible_smoothed_speed"}.intersection(row["quality_flags"]) or (previous and (row["track_id"] != previous["track_id"] or row["interval_start_frame"] != previous["frame_index"])):
                result["reason"] = "Position gap, excluded interval or tracker fragment boundary prevents continuous recovery timing."
                break
            distance = math.hypot(row["x_m"] - target["target_x_m"], row["y_m"] - target["target_y_m"])
            if distance <= target["radius_m"]:
                result.update(recovery_time_s=row["timestamp_s"] - times[target["contact_frame"]], reached_frame=index, status="available")
                break
            previous = row
        else:
            result.update(status="not_reached", reason="Target was not reached in the continuously observed review interval.")
        results.append(result)
    return results


def analyze_insights(shuttle_detections, player_frames, calibration, *, mode="singles", review=None):
    """Return JSON-safe estimates without mutating tracking or calibration inputs."""
    if mode not in {"singles", "doubles"}:
        raise ValueError("Insights mode must be singles or doubles.")
    frames = deepcopy(list(player_frames))
    detections = [deepcopy(_record(item)) for item in shuttle_detections]
    _validate_timeline(frames, "Player")
    _validate_timeline(detections, "Shuttle")
    for frame in frames:
        ids = [p.get("track_id") for p in frame.get("players", [])]
        if any(isinstance(raw_id, bool) or not isinstance(raw_id, int) or raw_id < 1 for raw_id in ids) or len(set(ids)) != len(ids):
            raise ValueError("Players require unique positive raw tracker IDs in each frame.")
    reviewed = validate_review(review, frames)
    review_digest = reviewed.get("source_video_sha256")
    calibration_digest = getattr(calibration, "metadata", {}).get("input_video_sha256") if calibration is not None else None
    if review_digest and calibration_digest and review_digest != calibration_digest:
        raise ValueError("Review source SHA256 differs from this calibrated video's source.")
    motion, players, confirmed, max_interval = _player_motion(frames, calibration, reviewed)
    shuttle_motion, shuttle = _shuttle_motion(detections, frames, calibration)
    identified = []
    labels = sorted({player["player_label"] for player in players if player["player_label"]})
    for label in labels:
        fragments = [player for player in players if player["player_label"] == label]
        valid = [p for p in fragments if p["distance_m"] is not None]
        duration = sum(p["valid_motion_time_s"] for p in valid)
        distance = sum(p["distance_m"] for p in valid) if valid else None
        identified.append({"player_label": label, "track_ids": [p["track_id"] for p in fragments],
                           "distance_m": distance, "valid_motion_time_s": duration,
                           "average_speed_m_s": distance / duration if duration else None,
                           "max_speed_m_s": max((p["max_speed_m_s"] for p in valid), default=None),
                           "aggregation_policy": "manual identity labels; sum valid intervals only, never bridge fragments"})
    warnings = []
    if any(reviewed[field] for field in ("identity_map", "rallies", "events", "recovery_targets")) and not reviewed.get("source_video_sha256"):
        warnings.append("The supplied annotations have no source_video_sha256 binding; confirm they belong to this exact video before trusting identities or targets.")
    if not confirmed:
        warnings.append("Court calibration is unavailable or an unconfirmed automatic proposal; metre movement metrics are unavailable.")
    mismatch = sum(frame.get("count_mismatch", False) for frame in frames)
    if mismatch:
        warnings.append(f"{mismatch} frames have unexpected player counts; review identities and court ROI.")
    if any(p["excluded_motion_intervals"] for p in players):
        warnings.append("Some movement intervals were excluded due to gaps, confidence, calibration or implausible jumps; distance is a partial observed estimate.")
    expected = 2 if mode == "singles" else 4
    if len(players) > expected:
        warnings.append("More tracker fragments than expected athletes; raw tracker IDs do not establish athlete identity.")
    if not reviewed["identity_map"]:
        warnings.append("No athlete identity mapping was reviewed; all player summaries describe raw tracker fragments.")
    if not reviewed["rallies"]:
        warnings.append("No rally boundaries were reviewed; whole-clip totals may include waiting, replay or between-rally movement.")
    shuttle_jump_count = sum("large_image_jump_review" in row["quality_flags"] for row in shuttle_motion)
    if shuttle_jump_count:
        threshold = POLICY["shuttle_jump_review_fraction_of_image_diagonal"]
        warnings.append(
            f"{shuttle_jump_count} shuttle intervals moved at least {threshold:.0%} of the source image diagonal in one frame. "
            "This review cue does not distinguish a fast shuttle from a false detection. Review the video before interpreting mean or largest detected image motion."
        )
    return {
        "schema_version": "1.0", "mode": mode,
        "coordinate_convention": "metres from far-left outer doubles corner; x across court, y toward near baseline",
        "players": players, "identified_players": identified, "player_motion": motion,
        "shuttle": shuttle, "shuttle_motion": shuttle_motion,
        "rallies": _rally_summaries(reviewed, frames, motion),
        "doubles_pair_spacing": _doubles_spacing(frames, motion, mode),
        "recovery": _recovery(reviewed, frames, motion), "review": reviewed,
        "reviewed_events": [{**event, "positions": [{key: row[key] for key in ("track_id", "player_label", "x_m", "y_m", "valid_position", "quality_flags")} for row in motion if row["frame_index"] == event["frame_index"] and (not event.get("player_label") or row["player_label"] == event["player_label"])]} for event in reviewed["events"]],
        "quality": {"calibration_reviewed": confirmed, "frame_count": len(frames), "count_mismatch_frames": mismatch,
                    "valid_player_positions": sum(row["valid_position"] for row in motion),
                    "player_observations": len(motion), "warnings": warnings,
                    "policy": {**POLICY, "maximum_interval_s": max_interval}},
        "limitations": [
            "Player positions and distances are approximate floor projections of ankle midpoint, not independently validated athlete measurements.",
            "Local linear smoothing and a stationary noise guard reduce jitter but can attenuate brief accelerations and undercount short movements.",
            "Distances and time averages use only consecutive accepted observations; gaps, camera cuts, low confidence and fragment boundaries are never bridged.",
            "Fixed camera and correct court corners are required. Unflagged cuts, zooms or camera motion invalidate estimates and require a separate clip/calibration.",
            "Shuttle image speed is pixels per second and depends on perspective, camera framing and detection noise; it is not comparable to measured smash speed.",
            "Physical shuttle speed in km/h is unavailable because a single camera and floor homography do not identify airborne height and depth.",
            "Zones are observed occupancy, not judgments about correct tactical positioning. Contact, outcome, partnerships and recovery targets require explicit review.",
            "These descriptive reports support coach video review; improved player performance has not been established.",
        ],
    }

def export_insights(outdir, insights):
    from .report import export_insights as export
    return export(outdir, insights)
