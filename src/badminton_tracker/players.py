"""YOLO11 pose inference and ByteTrack association for M2.

A track ID describes a tracker fragment, not a confirmed athlete identity. Court
ROI filtering happens before association; no person counts are silently capped.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from .video import iter_rgb_frames, probe_video

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COCO17_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def validate_player_roi(player_roi):
    """Validate a normalized polygon; None leaves selection explicitly unverified."""
    if player_roi is None:
        return None
    try:
        polygon = np.asarray(player_roi, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("Player ROI must be a list of normalized [x, y] points.") from error
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("Player ROI needs at least three normalized [x, y] corners.")
    if not np.isfinite(polygon).all() or np.any(polygon < 0) or np.any(polygon > 1):
        raise ValueError("Player ROI coordinates must be finite values between zero and one.")
    if abs(cv2.contourArea(polygon)) < 1e-6:
        raise ValueError("Player ROI must enclose a nonzero area with ordered corners.")
    if len(np.unique(polygon, axis=0)) != len(polygon):
        raise ValueError("Player ROI corners must not repeat, including the closing corner.")
    return polygon


def format_keypoints(raw_keypoints, threshold=0.3, image_shape=None):
    """Suppress uncertain or invalid coordinates while retaining confidence values."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Keypoint threshold must be between zero and one.")
    raw = np.asarray(raw_keypoints, dtype=np.float64)
    if raw.shape != (17, 3):
        raise ValueError("A COCO pose must contain seventeen [x, y, confidence] keypoints.")
    points = []
    for name, (x, y, score) in zip(COCO17_KEYPOINT_NAMES, raw):
        confidence = float(np.clip(score, 0.0, 1.0)) if math.isfinite(score) else 0.0
        visible = bool(math.isfinite(x) and math.isfinite(y) and math.isfinite(score) and confidence >= threshold)
        # A zeroed coordinate pair is a missing keypoint, even if its score is malformed.
        visible = visible and not (x == 0.0 and y == 0.0)
        if image_shape is not None:
            height, width = image_shape
            visible = visible and 0 <= x <= width and 0 <= y <= height
        points.append({
            "name": name, "x": float(x) if visible else None,
            "y": float(y) if visible else None, "confidence": confidence,
            "visible": bool(visible),
        })
    return points


def ankle_midpoint(keypoints):
    """Return an image point only when both ankles have trustworthy coordinates."""
    if len(keypoints) != 17:
        raise ValueError("Ankle extraction requires seventeen COCO keypoints.")
    left, right = keypoints[15], keypoints[16]
    if not all(point.get("visible") and point.get("x") is not None and point.get("y") is not None
               for point in (left, right)):
        return None
    return [float((left["x"] + right["x"]) / 2), float((left["y"] + right["y"]) / 2)]


def select_player_indices(boxes_xyxy, keypoints, roi, width, height, keypoint_threshold=0.3):
    """Return original detection indices whose floor anchor lies in the supplied ROI.

    Either bounding-box bottom center or a trusted ankle midpoint may pass. The
    model still sees the original full frame, including court and background.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    poses = np.asarray(keypoints, dtype=np.float32)
    if poses.shape != (len(boxes), 17, 3):
        raise ValueError("Every person box must have its corresponding seventeen-keypoint pose.")
    polygon = validate_player_roi(roi)
    if polygon is None:
        return list(range(len(boxes)))
    polygon_px = polygon * np.asarray([width, height], dtype=np.float32)
    indices = []
    for index, box in enumerate(boxes):
        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            continue
        candidates = [[float((box[0] + box[2]) / 2), float(box[3])]]
        midpoint = ankle_midpoint(format_keypoints(poses[index], keypoint_threshold, (height, width)))
        if midpoint is not None:
            candidates.append(midpoint)
        if any(cv2.pointPolygonTest(polygon_px, (float(x), float(y)), False) >= 0
               for x, y in candidates):
            indices.append(index)
    return indices


class _IndexedDetections:
    """The small Boxes-compatible interface ByteTrack uses, with stable source indices.

    Ultralytics splits high and low confidence detections inside update(). Keeping
    the original index on every slice prevents a pose from being assigned to the
    wrong box when those confidence subsets are enumerated independently.
    """
    def __init__(self, xyxy, conf, classes, source_indices=None):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(classes, dtype=np.float32).reshape(-1)
        self.source_indices = (np.arange(len(self.xyxy), dtype=np.int64) if source_indices is None
                               else np.asarray(source_indices, dtype=np.int64).reshape(-1))
        if not len(self.xyxy) == len(self.conf) == len(self.cls) == len(self.source_indices):
            raise ValueError("Box, confidence, class and original-index arrays must align.")

    @property
    def xywh(self):
        output = self.xyxy.copy()
        output[:, :2] = (self.xyxy[:, :2] + self.xyxy[:, 2:]) / 2
        output[:, 2:] = self.xyxy[:, 2:] - self.xyxy[:, :2]
        return output

    def __len__(self):
        return len(self.xyxy)

    def __getitem__(self, item):
        return _IndexedDetections(self.xyxy[item], self.conf[item], self.cls[item], self.source_indices[item])


def _create_tracker(settings, frame_rate):
    from ultralytics.trackers.byte_tracker import BYTETracker, STrack

    class IndexedBYTETracker(BYTETracker):
        def init_track(self, results, img=None):
            if len(results) == 0:
                return []
            boxes_with_indices = np.column_stack((results.xywh, results.source_indices))
            return [STrack(box, score, category)
                    for box, score, category in zip(boxes_with_indices, results.conf, results.cls)]

    # BYTETracker initialization resets its ID counter and state for this video.
    return IndexedBYTETracker(SimpleNamespace(**settings), frame_rate=max(1, round(frame_rate)))


def make_frame_record(frame_index, timestamp_s, players, expected_count, selection_warning=False):
    """Expose counts and uncertainty without inventing or reassigning observations."""
    observed = len(players)
    flags = []
    if selection_warning:
        flags.append("player_selection_unverified")
    if observed != expected_count:
        flags.append("player_count_mismatch")
    if not observed:
        flags.append("no_players_observed")
    if any(player["low_confidence"] for player in players):
        flags.append("low_confidence_player_or_pose")
    if any(player["ankle_midpoint"] is None for player in players):
        flags.append("ankle_midpoint_unavailable")
    return {
        "frame_index": int(frame_index), "timestamp_s": float(timestamp_s), "players": players,
        "expected_player_count": int(expected_count), "observed_player_count": observed,
        "count_mismatch": observed != expected_count, "flags": flags,
    }


def summarize_player_tracks(frames, expected_count):
    """Report observed fragments and gaps, without asserting athlete continuity."""
    observations = {}
    for frame in frames:
        for player in frame["players"]:
            observations.setdefault(player["track_id"], []).append(frame["frame_index"])
    fragments = []
    for track_id, indices in sorted(observations.items()):
        gaps = [{"after_frame": first, "before_frame": second, "missing_frames": second - first - 1}
                for first, second in zip(indices, indices[1:]) if second - first > 1]
        fragments.append({
            "track_id": int(track_id), "observed_frames": len(indices),
            "first_frame": indices[0], "last_frame": indices[-1], "gaps": gaps,
            "missing_frames_within_span": sum(gap["missing_frames"] for gap in gaps),
        })
    mismatch_frames = [frame["frame_index"] for frame in frames if frame["count_mismatch"]]
    warnings = []
    if mismatch_frames:
        warnings.append(f"{len(mismatch_frames)} frames differ from the expected {expected_count} players; review the overlay and ROI.")
    if len(observations) > expected_count:
        warnings.append("More track IDs than expected players were observed; fragmentation, ID switches or ROI contamination need review.")
    if any(fragment["gaps"] for fragment in fragments):
        warnings.append("Some tracks contain observation gaps; missing poses were not interpolated or assigned replacement IDs.")
    return {
        "total_frames": len(frames), "expected_player_count": expected_count,
        "unique_track_ids": len(observations), "track_fragments": fragments,
        "count_mismatch_frames": len(mismatch_frames), "count_mismatch_frame_indices": mismatch_frames,
        "frames_without_players": sum(not frame["players"] for frame in frames),
        "frames_with_low_confidence_players": sum(any(player["low_confidence"] for player in frame["players"]) for frame in frames),
        "warnings": warnings,
        "identity_policy": "raw ByteTrack IDs are track fragments, not verified athlete identities; no identity reassignment heuristic",
        "identity_switch_rate": None,
        "identity_switch_rate_status": "not measured against athlete identity annotations",
    }


def _load_tracker_settings(path, confidence_threshold):
    import yaml
    with Path(path).open("r", encoding="utf-8") as source:
        settings = yaml.safe_load(source)
    required = {"tracker_type", "track_high_thresh", "track_low_thresh", "new_track_thresh", "track_buffer", "match_thresh", "fuse_score"}
    if not isinstance(settings, dict) or not required <= settings.keys():
        raise ValueError("ByteTrack configuration is missing required fields.")
    if settings["tracker_type"] != "bytetrack":
        raise ValueError("M2 requires a ByteTrack configuration.")
    for name in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"ByteTrack {name} must be a number greater than zero and at most one.")
    if isinstance(settings["track_buffer"], bool) or not isinstance(settings["track_buffer"], int) or settings["track_buffer"] < 1:
        raise ValueError("ByteTrack track_buffer must be a positive integer.")
    if not isinstance(settings["fuse_score"], bool):
        raise ValueError("ByteTrack fuse_score must be boolean.")
    settings["track_high_thresh"] = max(confidence_threshold, settings["track_high_thresh"])
    settings["new_track_thresh"] = max(confidence_threshold, settings["new_track_thresh"])
    if settings["track_low_thresh"] > settings["track_high_thresh"]:
        raise ValueError("ByteTrack low confidence threshold cannot exceed its high confidence threshold.")
    return settings


def validate_box_nms_iou(value):
    """Validate the detector NMS threshold without adding a person-count cap."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("Player box NMS IoU must be finite, greater than zero and at most one.")
    return float(value)


def track_players(video_path: Path, weights_path: Path, *, mode="singles", player_roi=None,
                  device="auto", image_size=1280, confidence_threshold=0.25,
                  keypoint_threshold=0.3, box_nms_iou=0.7, max_frames=None, fps_hint=None,
                  tracker_config: Path | None = None) -> tuple[list[dict], dict]:
    """Run actual pretrained pose inference and per-video ByteTrack association."""
    if mode not in {"singles", "doubles"}:
        raise ValueError("Player mode must be singles or doubles.")
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 32:
        raise ValueError("Player image size must be an integer of at least 32 pixels.")
    if not math.isfinite(confidence_threshold) or not 0 < confidence_threshold <= 1:
        raise ValueError("Player confidence threshold must be greater than zero and at most one.")
    if not math.isfinite(keypoint_threshold) or not 0 <= keypoint_threshold <= 1:
        raise ValueError("Player keypoint threshold must be between zero and one.")
    box_nms_iou = validate_box_nms_iou(box_nms_iou)
    roi = validate_player_roi(player_roi)
    weights_path = Path(weights_path).resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Pose checkpoint does not exist: {weights_path}. Run the model download script first.")
    config_path = Path(tracker_config or PROJECT_ROOT / "configs" / "bytetrack.yaml").resolve()
    settings = _load_tracker_settings(config_path, confidence_threshold)
    video_metadata = probe_video(Path(video_path), fps_hint)

    # These variables must be set before the first Ultralytics import. Every path
    # is local to this project; no implicit weight download is requested below.
    config_dir = PROJECT_ROOT / "working" / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("M2 dependencies are missing. Install the project's requirements first.") from error
    selected_device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if selected_device == "cuda":
        selected_device = "cuda:0"
    if selected_device != "cpu" and not selected_device.startswith("cuda:"):
        raise ValueError("M2 supports cpu, cuda, cuda:N or auto devices.")
    if selected_device.startswith("cuda:") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu or install CUDA enabled PyTorch.")
    expected_count = 2 if mode == "singles" else 4
    selection_warning = roi is None
    try:
        model = YOLO(str(weights_path), task="pose")
        tracker = _create_tracker(settings, video_metadata["fps"])
    except Exception as error:
        raise RuntimeError(f"Could not load pose model or ByteTrack: {error}") from error
    settings_dir = PROJECT_ROOT / "working" / "ultralytics"
    # Disable integrations and put supported cache/output paths on G as well.
    ultralytics.settings.update({"datasets_dir": str(PROJECT_ROOT / "data"),
                                 "weights_dir": str(PROJECT_ROOT / "models"),
                                 "runs_dir": str(PROJECT_ROOT / "results"),
                                 "sync": False})
    frames = []
    rejected_detections = 0
    person_detections = 0
    inference_threshold = min(confidence_threshold, settings["track_low_thresh"])
    started = time.perf_counter()
    for index, timestamp, rgb in iter_rgb_frames(Path(video_path), max_frames=max_frames, fps_hint=fps_hint):
        # NumPy image inputs to Ultralytics use BGR; exports retain source pixels.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        results = model.predict(source=bgr, imgsz=image_size, conf=inference_threshold, iou=box_nms_iou,
                                device=selected_device, classes=[0], verbose=False,
                                save=False, project=str(PROJECT_ROOT / "results"),
                                max_det=300)
        result = results[0]
        boxes = result.boxes.cpu().numpy()
        if result.keypoints is None:
            raise RuntimeError("The supplied checkpoint did not produce pose keypoints; use YOLO11n-pose weights.")
        poses = result.keypoints.data.detach().cpu().numpy()
        if poses.shape != (len(boxes), 17, 3):
            raise RuntimeError("M2 requires a COCO17 pose checkpoint with confidence scores.")
        accepted = select_player_indices(boxes.xyxy, poses, roi, rgb.shape[1], rgb.shape[0], keypoint_threshold)
        person_detections += len(boxes)
        rejected_detections += len(boxes) - len(accepted)
        selected = _IndexedDetections(boxes.xyxy[accepted], boxes.conf[accepted], boxes.cls[accepted], accepted)
        tracks = tracker.update(selected, img=bgr)
        players = []
        for tracked in tracks:
            source_index = int(tracked[-1])
            if not 0 <= source_index < len(poses) or source_index not in accepted:
                raise RuntimeError("ByteTrack returned an invalid pose association index.")
            points = format_keypoints(poses[source_index], keypoint_threshold, rgb.shape[:2])
            midpoint = ankle_midpoint(points)
            score = float(tracked[5])
            low = score < confidence_threshold or sum(point["visible"] for point in points) < 6 or midpoint is None
            players.append({
                "track_id": int(tracked[4]), "bbox": [float(value) for value in tracked[:4]],
                "confidence": score, "keypoints": points, "ankle_midpoint": midpoint,
                "low_confidence": bool(low), "identity_source": "bytetrack",
            })
        players.sort(key=lambda player: player["track_id"])
        frames.append(make_frame_record(index, timestamp, players, expected_count, selection_warning))
        if (index + 1) % 25 == 0:
            available_total = video_metadata.get("frame_count")
            if max_frames is not None and available_total is not None:
                available_total = min(max_frames, available_total)
            total_text = f"/{available_total}" if available_total else ""
            print(f"Player tracking: {index + 1}{total_text} frames processed", flush=True)
    if not frames:
        raise ValueError("The input video has no decodable frames for player tracking.")
    elapsed = time.perf_counter() - started
    summary = summarize_player_tracks(frames, expected_count)
    warnings = list(summary["warnings"])
    if selection_warning:
        warnings.insert(0, "No player ROI was supplied: all detected people can be tracked, including officials and spectators. Manual player selection is required.")
    digest = hashlib.sha256()
    with weights_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = {
        "milestone": "M2", "model": "YOLO11 pose", "tracker": "ByteTrack",
        "ultralytics_version": ultralytics.__version__, "torch_version": torch.__version__,
        "weights_path": str(weights_path), "weights_sha256": digest.hexdigest(),
        "device": selected_device, "image_size": image_size,
        "confidence_threshold": confidence_threshold, "detector_threshold": inference_threshold,
        "box_nms_iou": box_nms_iou,
        "keypoint_threshold": keypoint_threshold, "mode": mode,
        "player_roi": None if roi is None else roi.tolist(),
        "selection_warning": selection_warning, "selection_policy": "ROI anchor gate before tracking; no cap to expected count",
        "tracker_config": str(config_path), "tracker_settings": settings,
        "pose_association_policy": "original detection index retained through ROI filtering and ByteTrack confidence subsets",
        "box_nms_policy": "Ultralytics person box suppression before ROI and ByteTrack; configurable IoU, no player-count cap or pose merging",
        "keypoint_names": list(COCO17_KEYPOINT_NAMES), "keypoint_coordinates": "source image pixels; low confidence coordinates are null",
        "ankle_midpoint_policy": "both ankle keypoints must pass confidence and validity checks; no floor projection in M2",
        "low_confidence_policy": "box score below requested threshold, fewer than six visible keypoints, or unavailable ankle midpoint",
        "missing_data_policy": "no synthetic poses or interpolated players; tracker may recover an existing ID across a short gap",
        "video": video_metadata, "frames_analyzed": len(frames),
        "person_detections_before_roi": person_detections, "detections_rejected_by_roi": rejected_detections,
        "processing_seconds": elapsed, "processing_fps": len(frames) / elapsed if elapsed else None,
        "settings_directory": str(settings_dir), "summary": summary, "warnings": warnings,
        "limitations": ["COCO pose model was not trained specifically for badminton.",
                        "ByteTrack can switch or fragment IDs during overlap, occlusion or camera cuts.",
                        "An ROI restricts where people stand; it does not establish who is an athlete.",
                        "M2 provides image coordinates only, without court distance or physical speed.",
                        "ID switch rate and accuracy on other videos have not been measured."],
    }
    return frames, metadata
