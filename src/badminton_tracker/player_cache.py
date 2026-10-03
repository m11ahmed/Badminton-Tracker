"""Validate complete, input-bound player observations before reusing inference."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .video import iter_rgb_frames

NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_cached_players(path, input_video, metadata, settings, fps_hint, roi=None):
    cache_path = Path(path) / "players.json" if Path(path).is_dir() else Path(path)
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load cached player observations: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("Unsupported player cache schema.")
    context, rows = payload.get("metadata"), payload.get("frames")
    if not isinstance(context, dict) or not isinstance(rows, list) or not rows:
        raise ValueError("Player cache needs metadata and nonempty frames.")
    if context.get("input_video_sha256") != digest(input_video):
        raise ValueError("Player cache input SHA256 does not match this video.")
    for key in ("width", "height", "fps", "frame_count", "duration_s", "time_base", "codec"):
        if context.get("video", {}).get(key) != metadata[key]:
            raise ValueError(f"Player cache video metadata mismatch: {key}.")
    cached_settings = context.get("settings", {})
    for key in ("players_mode", "pose_image_size", "player_confidence", "keypoint_confidence"):
        if cached_settings.get(key) != settings[key]:
            raise ValueError(f"Player cache inference parameter mismatch: {key}.")
    model = context.get("player_model")
    if not isinstance(model, dict) or model.get("tracker") != "ByteTrack" or not isinstance(model.get("weights_sha256"), str) or len(model["weights_sha256"]) != 64:
        raise ValueError("Player cache lacks pose model and tracker provenance.")
    cached_iou = cached_settings.get("pose_nms_iou", 0.7)
    if cached_iou != settings.get("pose_nms_iou", 0.7) or model.get("box_nms_iou", 0.7) != cached_iou:
        raise ValueError("Player cache inference parameter mismatch: pose_nms_iou.")
    import yaml
    current_tracker = yaml.safe_load((ROOT / "configs" / "bytetrack.yaml").read_text(encoding="utf-8-sig"))
    current_tracker["track_high_thresh"] = max(settings["player_confidence"], current_tracker["track_high_thresh"])
    current_tracker["new_track_thresh"] = max(settings["player_confidence"], current_tracker["new_track_thresh"])
    if model.get("tracker_settings") != current_tracker:
        raise ValueError("Player cache tracker settings mismatch.")
    cached_roi = context.get("player_roi")
    if roi is not None:
        try:
            matches = np.asarray(cached_roi).shape == np.asarray(roi).shape and np.allclose(cached_roi, roi, atol=1e-6, rtol=0)
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError("Player cache ROI mismatch; changed selection requires fresh player inference.")
    if cached_roi is not None:
        from .players import validate_player_roi
        validate_player_roi(cached_roi)
    expected = 2 if settings["players_mode"] == "singles" else 4
    width, height = metadata["width"], metadata["height"]
    count = 0
    for index, timestamp, _ in iter_rgb_frames(input_video, fps_hint=fps_hint):
        if index >= len(rows):
            raise ValueError("Player cache does not cover the complete source frame count.")
        frame = rows[index]
        if not isinstance(frame, dict) or frame.get("frame_index") != index or not finite(frame.get("timestamp_s")) or abs(frame["timestamp_s"] - timestamp) > 1e-5:
            raise ValueError("Player cache frame indices or timestamps do not match source PTS.")
        players = frame.get("players")
        if not isinstance(players, list) or frame.get("observed_player_count") != len(players) or frame.get("expected_player_count") != expected or frame.get("count_mismatch") is not (len(players) != expected):
            raise ValueError("Player cache has inconsistent player counts.")
        if not isinstance(frame.get("flags"), list) or not all(isinstance(flag, str) for flag in frame["flags"]):
            raise ValueError("Player cache flags must be strings.")
        ids = set()
        for player in players:
            if not isinstance(player, dict):
                raise ValueError("Player cache observation must be an object.")
            track_id = player.get("track_id")
            if type(track_id) is not int or track_id < 1 or track_id in ids or player.get("identity_source") != "bytetrack":
                raise ValueError("Player cache IDs must be unique positive ByteTrack integers per frame.")
            ids.add(track_id)
            box = player.get("bbox")
            if not isinstance(box, list) or len(box) != 4 or not all(finite(value) for value in box) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Player cache box is invalid.")
            score = player.get("confidence")
            if not finite(score) or not 0 <= score <= 1 or type(player.get("low_confidence")) is not bool:
                raise ValueError("Player cache confidence is invalid.")
            points = player.get("keypoints")
            if not isinstance(points, list) or len(points) != 17 or [point.get("name") if isinstance(point, dict) else None for point in points] != list(NAMES):
                raise ValueError("Player cache needs all seventeen ordered COCO keypoints.")
            for point in points:
                if type(point.get("visible")) is not bool or not finite(point.get("confidence")) or not 0 <= point["confidence"] <= 1:
                    raise ValueError("Player cache keypoint confidence is invalid.")
                if point["visible"]:
                    if not finite(point.get("x")) or not finite(point.get("y")) or not (0 <= point["x"] <= width and 0 <= point["y"] <= height) or point["confidence"] < settings["keypoint_confidence"] or (point["x"] == 0 and point["y"] == 0):
                        raise ValueError("Player cache visible keypoint is invalid.")
                elif point.get("x") is not None or point.get("y") is not None:
                    raise ValueError("Player cache missing keypoints must have null coordinates.")
            ankles_visible = all(point["visible"] for point in points[15:17])
            midpoint = player.get("ankle_midpoint")
            if ankles_visible:
                actual = [(points[15][coordinate] + points[16][coordinate]) / 2 for coordinate in ("x", "y")]
                if not isinstance(midpoint, list) or len(midpoint) != 2 or not all(finite(value) for value in midpoint) or not np.allclose(midpoint, actual, rtol=0, atol=1e-5):
                    raise ValueError("Player cache ankle midpoint is inconsistent with observed ankles.")
            elif midpoint is not None:
                raise ValueError("Player cache must not invent a midpoint with a missing ankle.")
        count += 1
    if count != len(rows):
        raise ValueError("Player cache frame count does not match complete decoded source.")
    selected = deepcopy(rows if settings["max_frames"] is None else rows[:settings["max_frames"]])
    for frame in selected:
        frame.pop("court_quality_flags", None)
        for player in frame["players"]:
            player.pop("court_position", None)
            player.pop("court_flags", None)
    return selected, {
        "cache_path": str(cache_path.resolve()), "cache_sha256": digest(cache_path),
        "input_sha256_verified": True, "decoded_frames_verified": count, "source_timestamps_verified": True,
        "court_positions_discarded": True, "cached_parameters": cached_settings,
        "cached_model": model, "cached_roi": cached_roi,
        "cached_pose_weights": context.get("pose_weights"), "cached_pose_weights_sha256": context.get("pose_weights_sha256"),
    }
