"""Command line orchestration and transactional shuttle and player exports."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from . import __version__
from .postprocess import interpolate_short_gaps, tracking_summary, validate_detections
from .types import Detection


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "device": "auto", "batch_size": 1, "threads": 2, "max_frames": None,
    "confidence_threshold": 0.5, "heatmap_threshold": 0.5,
    "background_samples": 32, "max_gap_frames": 2, "trail_frames": 12,
    "players_mode": "singles", "pose_image_size": 1280,
    "player_confidence": 0.25, "keypoint_confidence": 0.3, "pose_nms_iou": 0.7,
}
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def _configure_project_runtime() -> None:
    """Use supported project directories before loading optional model packages."""
    locations = {
        "YOLO_CONFIG_DIR": PROJECT_ROOT / "working" / "ultralytics",
        "TEMP": PROJECT_ROOT / "working" / "tmp",
        "TMP": PROJECT_ROOT / "working" / "tmp",
        "TORCH_HOME": PROJECT_ROOT / "working" / "torch",
        "MPLCONFIGDIR": PROJECT_ROOT / "working" / "matplotlib",
        "XDG_CACHE_HOME": PROJECT_ROOT / "working" / "cache",
    }
    for name, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    # Python may have discovered the temporary directory before CLI startup.
    import tempfile
    tempfile.tempdir = str(locations["TEMP"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Badminton neural shuttle tracking, player pose and calibrated court review using source timestamps.")
    parser.add_argument("--video", type=Path, required=True, help="Input match video. Its presentation timestamps take precedence.")
    parser.add_argument("--milestone", choices=("m1", "m2", "m3", "m5"), default="m1", help="m1: shuttle; m2: player IDs and pose; m3: calibrated court and minimap; m5: movement and coaching insights. Default m1.")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "models" / "TrackNet_best.pt", help="Trusted TrackNetV3 checkpoint for fresh shuttle inference.")
    parser.add_argument("--pose-weights", type=Path, default=PROJECT_ROOT / "models" / "yolo11n-pose.pt", help="Trusted, local YOLO pose checkpoint for M2.")
    parser.add_argument("--shuttle-results", type=Path, help="M2/M3/M5: existing shuttle.json or its result directory. Requires recorded input SHA256 and matching source timing; no unsafe legacy reuse.")
    parser.add_argument("--player-results", type=Path, help="M2/M3/M5: verified complete players.json or its result directory; reuses image observations before court projection.")
    court_options = parser.add_mutually_exclusive_group()
    court_options.add_argument("--court-calibration", type=Path, help="M3/M5: four outer doubles corners from scripts/calibrate_court.py.")
    court_options.add_argument("--auto-court", action="store_true", help="M3/M5: propose court corners from lines/playing area; automatic geometry remains flagged as unconfirmed.")
    parser.add_argument("--out", type=Path, help="New or empty result directory inside this G drive project. Default: results/<UTC timestamp>.")
    parser.add_argument("--config", type=Path, help="JSON settings; defaults to configs/<milestone>.json.")
    parser.add_argument("--fps-hint", type=float, help="Fallback rate for missing source metadata; never retimes valid source PTS.")
    parser.add_argument("--device", help="auto, cpu, cuda or a specific device such as cuda:0.")
    parser.add_argument("--batch-size", type=int, help="Number of checkpoint length temporal windows per shuttle batch.")
    parser.add_argument("--threads", type=int, help="CPU inference thread count. Default 2 for this machine.")
    parser.add_argument("--max-frames", type=int, help="Analyze only the first N frames for a quick smoke test.")
    parser.add_argument("--confidence-threshold", type=float, help="Flag low shuttle heatmap scores.")
    parser.add_argument("--heatmap-threshold", type=float, help="Heatmap level used to isolate the predicted shuttle region.")
    parser.add_argument("--background-samples", type=int, help="Sample count for median shuttle background extraction.")
    parser.add_argument("--max-gap-frames", type=int, help="Fill bounded shuttle gaps up to N frames; zero disables filling.")
    parser.add_argument("--trail-frames", type=int, help="Maximum number of observed shuttle points in an overlay trail.")
    parser.add_argument("--players-mode", choices=("singles", "doubles"), help="Expected count is 2 or 4; detections are never capped to this number.")
    parser.add_argument("--player-roi", type=Path, help="Explicit normalized polygon JSON for this camera view. No ROI is applied by default.")
    parser.add_argument("--pose-image-size", type=int, help="YOLO inference image size; default 1280.")
    parser.add_argument("--player-confidence", type=float, help="Player detection confidence threshold; default 0.25.")
    parser.add_argument("--keypoint-confidence", type=float, help="Minimum confidence for visible pose points; default 0.3.")
    parser.add_argument("--pose-nms-iou", type=float, help="Person box overlap threshold; default 0.7. Lower values suppress duplicate boxes more strongly but may suppress overlapping doubles players.")
    parser.add_argument("--review-file", type=Path, help="M5: optional reviewed player labels, rally/contact notes and recovery targets JSON.")
    return parser


def _load_settings(args) -> dict:
    settings = dict(DEFAULTS)
    config = args.config or PROJECT_ROOT / "configs" / f"{args.milestone}.json"
    if not config.is_file():
        raise ValueError(f"Configuration does not exist: {config}")
    try:
        loaded = json.loads(config.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load configuration: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("Configuration must be a JSON object.")
    unknown = set(loaded) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    settings.update(loaded)
    for name in DEFAULTS:
        value = getattr(args, name, None)
        if value is not None:
            settings[name] = value
    for name in ("batch_size", "threads", "background_samples", "trail_frames", "pose_image_size"):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if settings["max_frames"] is not None and (type(settings["max_frames"]) is not int or settings["max_frames"] < 1):
        raise ValueError("max_frames must be a positive integer.")
    if type(settings["max_gap_frames"]) is not int or settings["max_gap_frames"] < 0:
        raise ValueError("max_gap_frames must be zero or greater.")
    for name in ("confidence_threshold", "heatmap_threshold", "player_confidence", "keypoint_confidence", "pose_nms_iou"):
        value = settings[name]
        if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be finite and greater than zero, up to one.")
    if settings["pose_image_size"] < 32:
        raise ValueError("pose_image_size must be at least 32 pixels.")
    if settings["players_mode"] not in ("singles", "doubles"):
        raise ValueError("players_mode must be singles or doubles.")
    if not isinstance(settings["device"], str) or not settings["device"]:
        raise ValueError("device must be a nonempty string.")
    if args.fps_hint is not None and (not math.isfinite(args.fps_hint) or args.fps_hint <= 0):
        raise ValueError("--fps-hint must be a positive finite number.")
    if args.milestone == "m1" and (args.shuttle_results is not None or args.player_roi is not None or args.player_results is not None):
        raise ValueError("Observation caches and player ROI require --milestone m2 or m3.")
    if args.milestone not in {"m3", "m5"} and (args.court_calibration is not None or args.auto_court):
        raise ValueError("Court calibration requires --milestone m3 or m5.")
    if args.milestone in {"m3", "m5"} and args.court_calibration is None and not args.auto_court:
        raise ValueError("M3/M5 needs --court-calibration or --auto-court. Use scripts/calibrate_court.py to click the four outer doubles corners.")
    if getattr(args, "review_file", None) is not None and args.milestone != "m5":
        raise ValueError("Review notes require --milestone m5.")
    return settings


def _contained_path(path: Path, name: str) -> Path:
    path = path.resolve()
    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"{name} must be inside {PROJECT_ROOT} to keep project work on G drive.") from exc
    if not relative.parts:
        raise ValueError(f"{name} cannot be the project root.")
    return path


def _validate_output(output: Path, input_video: Path, weights: Path, *other_sources: Path) -> None:
    if output.is_file():
        raise ValueError(f"Output path is a file: {output}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Choose a new --out directory; previous results are never overwritten.")
    for protected in (input_video, weights, *other_sources):
        if protected == output or output in protected.parents:
            raise ValueError("Output directory would contain a source video, model or cache. Choose a separate result directory.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_player_roi(path: Path | None):
    if path is None:
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load player ROI: {exc}") from exc
    points = loaded.get("points") if isinstance(loaded, dict) else loaded
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("Player ROI must have at least three normalized [x, y] polygon points.")
    polygon = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each player ROI point must be [x, y].")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1 for value in point):
            raise ValueError("Player ROI coordinates must be finite numbers between 0 and 1.")
        polygon.append(tuple(float(value) for value in point))
    if polygon[0] == polygon[-1]:
        polygon.pop()
    if len(set(polygon)) != len(polygon) or len(polygon) < 3:
        raise ValueError("Player ROI must contain distinct polygon vertices.")
    area = abs(sum(polygon[i][0] * polygon[(i + 1) % len(polygon)][1] - polygon[(i + 1) % len(polygon)][0] * polygon[i][1] for i in range(len(polygon)))) / 2
    if area <= 1e-8:
        raise ValueError("Player ROI polygon has zero area.")
    def orientation(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    def intersects(a, b, c, d):
        o1, o2, o3, o4 = orientation(a, b, c), orientation(a, b, d), orientation(c, d, a), orientation(c, d, b)
        if o1 * o2 < 0 and o3 * o4 < 0:
            return True
        for value, point, left, right in ((o1, c, a, b), (o2, d, a, b), (o3, a, c, d), (o4, b, c, d)):
            if abs(value) <= 1e-12 and min(left[0], right[0]) <= point[0] <= max(left[0], right[0]) and min(left[1], right[1]) <= point[1] <= max(left[1], right[1]):
                return True
        return False
    for i in range(len(polygon)):
        for j in range(i + 1, len(polygon)):
            if j == i + 1 or (i == 0 and j == len(polygon) - 1):
                continue
            if intersects(polygon[i], polygon[(i + 1) % len(polygon)], polygon[j], polygon[(j + 1) % len(polygon)]):
                raise ValueError("Player ROI polygon must not self intersect.")
    return polygon


def _load_cached_shuttle(path: Path, input_video: Path, metadata: dict, settings: dict, fps_hint):
    """Validate cache provenance and decode timing, then recover raw observations."""
    cache_path = path / "shuttle.json" if path.is_dir() else path
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load cached shuttle observations: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("Unsupported shuttle cache schema.")
    context, rows = payload.get("metadata"), payload.get("frames")
    if not isinstance(context, dict) or not isinstance(rows, list) or not rows:
        raise ValueError("Shuttle cache must contain metadata and nonempty per-frame records.")
    digest = context.get("input_video_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Shuttle cache has no recorded input SHA256. Legacy caches cannot prove input content; rerun shuttle inference once to create a verified cache.")
    if digest != _sha256(input_video):
        raise ValueError("Shuttle cache input SHA256 does not match this video.")
    cached_video = context.get("video", {})
    for name in ("width", "height", "fps", "frame_count", "duration_s", "time_base", "codec"):
        if name not in cached_video or cached_video[name] != metadata[name]:
            raise ValueError(f"Shuttle cache video metadata mismatch: {name}.")
    cached_settings = context.get("settings", {})
    for name in ("confidence_threshold", "heatmap_threshold", "background_samples"):
        if cached_settings.get(name) != settings[name]:
            raise ValueError(f"Shuttle cache inference parameter mismatch: {name}. Use matching settings or fresh inference.")
    # Decode the entire video: frame_count metadata can be missing or inaccurate.
    from .video import iter_rgb_frames
    count = 0
    for index, timestamp, _ in iter_rgb_frames(input_video, fps_hint=fps_hint):
        if index >= len(rows):
            raise ValueError("Shuttle cache does not cover the complete source frame count.")
        row = rows[index]
        if not isinstance(row, dict) or row.get("frame_index") != index:
            raise ValueError("Shuttle cache indices must match consecutive source frames.")
        recorded_time = row.get("timestamp_s")
        if not isinstance(recorded_time, (int, float)) or isinstance(recorded_time, bool) or not math.isfinite(recorded_time) or abs(recorded_time - timestamp) > 1e-5:
            raise ValueError("Shuttle cache frame timestamps do not match source PTS.")
        count += 1
    if count != len(rows):
        raise ValueError("Shuttle cache frame count does not match decoded video.")
    selected = rows if settings["max_frames"] is None else rows[:settings["max_frames"]]
    raw = []
    for row in selected:
        if not all(name in row for name in ("raw_x", "raw_y", "raw_confidence")):
            raise ValueError("Shuttle cache lacks raw detector fields.")
        x, y, score = row["raw_x"], row["raw_y"], row["raw_confidence"]
        if (x is None) != (y is None):
            raise ValueError("Shuttle cache raw coordinates must both be present or absent.")
        for value in (x, y, score):
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError("Shuttle cache raw values must be finite numbers or null.")
        missing = x is None
        raw.append(Detection(row["frame_index"], row["timestamp_s"], x, y, score, "missing" if missing else "detected", missing or score is None or score < settings["confidence_threshold"], x, y, score))
    validate_detections(raw)
    return raw, {
        "cache_path": str(cache_path.resolve()), "cache_sha256": _sha256(cache_path),
        "input_sha256_verified": True, "decoded_frames_verified": count,
        "source_timestamps_verified": True, "raw_detector_fields_reused": True,
        "interpolation_recomputed": True, "cached_parameters": cached_settings,
        "cached_model": context.get("model"), "cached_weights": context.get("weights"),
        "cached_weights_sha256": context.get("weights_sha256"),
    }


def _load_cached_players(path, input_video, metadata, settings, fps_hint, roi=None):
    from .player_cache import load_cached_players
    return load_cached_players(path, input_video, metadata, settings, fps_hint, roi)


def _export_observations(stage: Path, detections: list, context: dict) -> None:
    rows = [row.to_dict() for row in detections]
    with (stage / "shuttle.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(stage / "shuttle.json", {"schema_version": "1.0", "metadata": context, "frames": rows})


def _export_players(stage: Path, frames: list[dict], context: dict) -> None:
    """JSON retains empty frames; CSV contains one row per observed player."""
    _write_json(stage / "players.json", {"schema_version": "1.0", "metadata": context, "frames": frames})
    fields = ["frame_index", "timestamp_s", "expected_player_count", "observed_player_count", "count_mismatch", "frame_flags", "track_id", "identity_source", "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "ankle_midpoint_x", "ankle_midpoint_y", "low_confidence"]
    fields += [f"{name}_{component}" for name in KEYPOINT_NAMES for component in ("x", "y", "confidence", "visible")]
    is_court = context.get("milestone") in {"M3", "M5"}
    if is_court:
        fields += ["court_x_m", "court_y_m", "in_doubles_court", "in_singles_court", "court_flags"]
    with (stage / "players.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for frame in frames:
            for player in frame["players"]:
                bbox = player["bbox"]
                ankle = player.get("ankle_midpoint")
                row = {name: frame[name] for name in ("frame_index", "timestamp_s", "expected_player_count", "observed_player_count", "count_mismatch")}
                row.update({"frame_flags": ";".join(frame.get("flags", [])), "track_id": player.get("track_id"), "identity_source": player.get("identity_source"), "confidence": player.get("confidence"), "low_confidence": player.get("low_confidence"), "bbox_x1": bbox[0], "bbox_y1": bbox[1], "bbox_x2": bbox[2], "bbox_y2": bbox[3], "ankle_midpoint_x": ankle[0] if ankle else None, "ankle_midpoint_y": ankle[1] if ankle else None})
                for keypoint in player.get("keypoints", []):
                    name = keypoint["name"]
                    if name not in KEYPOINT_NAMES:
                        raise ValueError(f"Unknown player keypoint name: {name}")
                    row.update({f"{name}_{component}": keypoint.get(component) for component in ("x", "y", "confidence", "visible")})
                if is_court:
                    position = player.get("court_position") or {}
                    row.update({"court_x_m": position.get("x_m"), "court_y_m": position.get("y_m"), "in_doubles_court": position.get("in_doubles_court"), "in_singles_court": position.get("in_singles_court"), "court_flags": ";".join(player.get("court_flags", []))})
                writer.writerow(row)


def _export_court_positions(stage, frames, context):
    rows = []
    per_frame = []
    for frame in frames:
        players = []
        for player in frame["players"]:
            position = player.get("court_position") or {}
            record = {"track_id": player["track_id"], "x_m": position.get("x_m"), "y_m": position.get("y_m"), "in_doubles_court": position.get("in_doubles_court"), "in_singles_court": position.get("in_singles_court"), "quality_flags": player.get("court_flags", [])}
            players.append(record)
            rows.append({"frame_index": frame["frame_index"], "timestamp_s": frame["timestamp_s"], **record})
        per_frame.append({"frame_index": frame["frame_index"], "timestamp_s": frame["timestamp_s"], "players": players, "quality_flags": frame.get("court_quality_flags", [])})
    _write_json(stage / "court_positions.json", {"schema_version": "1.0", "metadata": {**context, "coordinate_units": "court meters; approximate floor references"}, "frames": per_frame})
    fields = ["frame_index", "timestamp_s", "track_id", "x_m", "y_m", "in_doubles_court", "in_singles_court", "quality_flags"]
    with (stage / "court_positions.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "quality_flags": ";".join(row["quality_flags"])})


def _player_summary(frames: list[dict]) -> dict:
    histogram, ids, fragments = {}, {}, []
    no_id = 0
    for frame in frames:
        count = str(frame["observed_player_count"])
        histogram[count] = histogram.get(count, 0) + 1
        for player in frame["players"]:
            track_id = player.get("track_id")
            if track_id is None:
                no_id += 1
                continue
            ids.setdefault(track_id, []).append(frame["frame_index"])
    for track_id, indices in sorted(ids.items()):
        segments = 1 + sum(right != left + 1 for left, right in zip(indices, indices[1:]))
        fragments.append({"track_id": track_id, "frames_observed": len(indices), "first_frame": indices[0], "last_frame": indices[-1], "contiguous_segments": segments})
    return {
        "frames_analyzed": len(frames), "expected_player_count": frames[0]["expected_player_count"] if frames else None,
        "observed_count_histogram": histogram,
        "count_mismatch_frames": sum(frame["count_mismatch"] for frame in frames),
        "empty_frames": sum(not frame["players"] for frame in frames),
        "players_with_missing_ankle_midpoint": sum(player.get("ankle_midpoint") is None for frame in frames for player in frame["players"]),
        "observations_without_track_id": no_id, "unique_tracker_ids": len(ids),
        "track_fragments": fragments,
        "identity_interpretation": "ByteTrack IDs describe tracker trajectories, not verified athlete identities. Unique IDs and gaps do not measure actual identity switches without labeled review.",
        "evaluation": "Detection counts and pose scores are coverage only; tracking and keypoint accuracy need labeled evaluation.",
    }


def execute(args, settings: dict) -> Path:
    input_video, weights = args.video.resolve(), args.weights.resolve()
    milestone = args.milestone
    has_players = milestone in {"m2", "m3", "m5"}
    if not input_video.is_file():
        raise ValueError(f"Video does not exist: {input_video}. Pass an existing file to --video.")
    if args.shuttle_results is None and not weights.is_file():
        raise ValueError(f"Checkpoint does not exist: {weights}. Run scripts/download_models.py or supply --weights.")
    pose_weights = args.pose_weights.resolve()
    if has_players and args.player_results is None and not pose_weights.is_file():
        raise ValueError(f"Pose checkpoint does not exist: {pose_weights}. Download the model or supply --pose-weights.")
    roi = _load_player_roi(args.player_roi) if has_players else None
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = _contained_path(args.out or PROJECT_ROOT / "results" / now, "Output directory")
    protected = [pose_weights] if has_players else []
    for path in (args.shuttle_results, args.player_results, args.player_roi, args.court_calibration):
        if path is not None:
            protected.append(path.resolve())
    review = None
    review_digest = None
    review_path = getattr(args, "review_file", None)
    if review_path is not None:
        review_path = review_path.resolve()
        protected.append(review_path)
        try:
            review_bytes = review_path.read_bytes()
            review_digest = hashlib.sha256(review_bytes).hexdigest()
            review = json.loads(review_bytes.decode("utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load review JSON: {exc}") from exc
        if not isinstance(review, dict):
            raise ValueError("Review JSON must be an object.")
    _validate_output(output, input_video, weights, *protected)
    from .video import probe_video, write_overlay, iter_rgb_frames
    metadata = probe_video(input_video, args.fps_hint)
    for warning in metadata["timing_warnings"]:
        print(f"Timing: {warning}", flush=True)
    input_digest = _sha256(input_video)
    if review and review.get("source_video_sha256") is not None and review["source_video_sha256"] != input_digest:
        raise ValueError("Review source SHA256 differs from this video; use notes reviewed for this clip.")
    _configure_project_runtime()
    stage = PROJECT_ROOT / "working" / f"{milestone}_{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        print(f"Analyzing {input_video.name} ({metadata['width']} x {metadata['height']}, source rate {metadata['fps']:.3f} fps)...", flush=True)
        cache = player_cache = calibration = court_quality = None
        if args.shuttle_results is not None:
            detections, cache = _load_cached_shuttle(args.shuttle_results, input_video, metadata, settings, args.fps_hint)
        if args.player_results is not None:
            player_frames, player_cache = _load_cached_players(args.player_results, input_video, metadata, settings, args.fps_hint, roi)
            if roi is None:
                roi = player_cache["cached_roi"]
        if milestone in {"m3", "m5"}:
            from .court import load_calibration, create_calibration
            if args.court_calibration is not None:
                calibration = load_calibration(args.court_calibration, metadata["width"], metadata["height"])
                recorded_digest = calibration.metadata.get("input_video_sha256")
                if recorded_digest is not None and recorded_digest != input_digest:
                    raise ValueError("Court calibration source SHA256 differs from this video; create a calibration for this clip.")
            else:
                from .court_detection import detect_court
                _, _, rgb = next(iter_rgb_frames(input_video, max_frames=1, fps_hint=args.fps_hint))
                candidate = detect_court(rgb)
                calibration = create_calibration(candidate["corners_px"], metadata["width"], metadata["height"], method="auto_proposal", metadata={"input_video_sha256": input_digest, "proposal": candidate})
                print("Automatic court proposal remains unconfirmed; inspect the projected lines and use manual corner calibration if they do not match.", flush=True)
        if cache is None or (has_players and player_cache is None):
            import torch
            torch.set_num_threads(settings["threads"])
        inference_start = time.perf_counter()
        if cache is None:
            from .tracknet import track_video
            detections, model_metadata = track_video(input_video, weights, device=settings["device"], batch_size=settings["batch_size"], max_frames=settings["max_frames"], confidence_threshold=settings["confidence_threshold"], heatmap_threshold=settings["heatmap_threshold"], background_samples=settings["background_samples"], fps_hint=args.fps_hint)
            checkpoint_sha, checkpoint_path = _sha256(weights), str(weights)
        else:
            print(f"Reusing {len(detections)} raw shuttle records after input SHA256 and complete source timing checks.", flush=True)
            model_metadata = cache["cached_model"]
            checkpoint_sha, checkpoint_path = cache["cached_weights_sha256"], cache["cached_weights"]
        shuttle_seconds = time.perf_counter() - inference_start
        if not detections:
            raise ValueError("Input video produced no decoded frames.")
        detections = interpolate_short_gaps(detections, settings["max_gap_frames"])
        player_metadata = None
        player_seconds = None
        if has_players:
            player_start = time.perf_counter()
            if player_cache is None:
                from .players import track_players
                print("Tracking players and estimating pose with YOLO and ByteTrack...", flush=True)
                player_frames, player_metadata = track_players(input_video, pose_weights, mode=settings["players_mode"], player_roi=roi, device=settings["device"], image_size=settings["pose_image_size"], confidence_threshold=settings["player_confidence"], keypoint_threshold=settings["keypoint_confidence"], box_nms_iou=settings["pose_nms_iou"], max_frames=settings["max_frames"], fps_hint=args.fps_hint, tracker_config=PROJECT_ROOT / "configs" / "bytetrack.yaml")
                pose_path, pose_sha = str(pose_weights), _sha256(pose_weights)
            else:
                print(f"Reusing {len(player_frames)} image-space player frames after full source verification; prior court positions discarded.", flush=True)
                player_metadata = player_cache["cached_model"]
                pose_path, pose_sha = player_cache["cached_pose_weights"], player_cache["cached_pose_weights_sha256"]
            player_seconds = time.perf_counter() - player_start
        else:
            player_frames = None
        if milestone in {"m3", "m5"}:
            from .court_tracking import project_players_on_court
            player_frames, court_quality = project_players_on_court(player_frames, calibration, mode=settings["players_mode"])
        if _sha256(input_video) != input_digest:
            raise ValueError("Input video changed during analysis; no result can be published.")
        insights = None
        if milestone == "m5":
            from .insights import analyze_insights, export_insights
            print("Computing quality-aware movement and shuttle image motion insights...", flush=True)
            insights = analyze_insights(detections, player_frames, calibration,
                                        mode=settings["players_mode"], review=review)
        print(f"Rendering {len(detections)} analyzed frames...", flush=True)
        overlay_arguments = {"fps_hint": args.fps_hint, "trail_frames": settings["trail_frames"], "player_frames": player_frames, "player_roi": roi}
        if calibration is not None:
            overlay_arguments["court_calibration"] = calibration
        if insights is not None:
            overlay_arguments["motion_insights"] = insights
        overlay = write_overlay(input_video, stage / "annotated.mp4", detections, **overlay_arguments)
        context = {
            "project_version": __version__, "milestone": milestone.upper(),
            "input_video": str(input_video), "input_video_sha256": input_digest,
            "weights": checkpoint_path, "weights_sha256": checkpoint_sha,
            "settings": settings, "fps_hint": args.fps_hint, "video": metadata,
            "model": model_metadata, "coordinate_units": "source image pixels",
            "confidence_definition": "uncalibrated heatmap peak score, not probability",
            "raw_fields": "raw_x, raw_y and raw_confidence retain detector output; interpolation is always flagged",
        }
        if cache is not None:
            context["shuttle_cache"] = cache
        if has_players:
            context.update({"player_model": player_metadata, "pose_weights": pose_path, "pose_weights_sha256": pose_sha, "player_roi": roi, "player_roi_path": str(args.player_roi.resolve()) if args.player_roi else None, "player_coordinate_units": "source image pixels; ankle midpoint is approximate", "identity_definition": "ByteTrack tracker ID; not a verified athlete identity"})
            if player_cache is not None:
                context["player_cache"] = player_cache
        if calibration is not None:
            context.update({"court_calibration": calibration.to_dict(), "court_calibration_path": str(args.court_calibration.resolve()) if args.court_calibration else None, "court_coordinate_units": "meters on approximate floor projection; source image and shuttle fields stay in pixels"})
        _export_observations(stage, detections, context)
        outputs = ["annotated.mp4", "shuttle.csv", "shuttle.json", "summary.json"]
        limitations = ["Tracking accuracy requires independent labeled evaluation; scores and coverage are not accuracy.", "Occlusion, blur, camera changes and unfamiliar footage may reduce tracking quality.", "Interpolated shuttle points are estimates, never observations.", "Overlay is silent. Shuttle coordinates remain source image pixels.", "No physical movement speed, racket measurement or coaching score is included."]
        summary = {"status": "complete", "provenance": context, "tracking_quality": tracking_summary(detections), "overlay": overlay, "inference_seconds": shuttle_seconds, "player_inference_seconds": player_seconds, "outputs": outputs, "limitations": limitations}
        if has_players:
            _export_players(stage, player_frames, context)
            outputs.extend(["players.csv", "players.json"])
            summary["player_tracking_quality"] = _player_summary(player_frames)
            limitations.extend(["ByteTrack IDs can fragment or switch and are not verified athlete identities.", "Camera-specific player ROI is needed to exclude background people.", "Player counts are not capped; mismatches remain flagged.", "Missing player positions are not interpolated and missing ankles do not produce a midpoint."])
        if calibration is not None:
            from .minimap import write_minimap_video
            _export_court_positions(stage, player_frames, context)
            _write_json(stage / "court_calibration.json", calibration.to_dict())
            minimap = write_minimap_video(stage / "minimap.mp4", player_frames, fps=metadata["fps"], calibration=calibration)
            outputs.extend(["court_positions.csv", "court_positions.json", "court_calibration.json", "minimap.mp4"])
            summary["court_tracking_quality"] = court_quality
            summary["minimap"] = minimap
            limitations.extend(court_quality["limitations"])
        else:
            limitations.append("No court calibration or floor coordinates in this milestone.")
        if milestone == "m5":
            insights["provenance"] = {"input_video_sha256": input_digest,
                "video": metadata, "court_calibration": calibration.to_dict(),
                "milestone": "M5", "review_file": str(review_path) if review_path else None,
                "review_file_sha256": review_digest}
            insight_files = export_insights(stage, insights)
            outputs.extend(insight_files.values() if isinstance(insight_files, dict) else insight_files)
            if review is not None:
                _write_json(stage / "review.json", review)
                outputs.append("review.json")
            summary["insights"] = {"file": "insights.json", "player_fragments": len(insights["players"]),
                "shuttle_speed_units": "source image pixels per second",
                "physical_shuttle_speed_km_h": None,
                "physical_speed_status": "unavailable_from_this_single_view"}
            limitations.remove("No physical movement speed, racket measurement or coaching score is included.")
            limitations.extend(insights.get("limitations", []))
        summary["processing_seconds"] = time.perf_counter() - start
        _write_json(stage / "summary.json", summary)
        if _sha256(input_video) != input_digest:
            raise ValueError("Input video changed during rendering; no result can be published.")
        _validate_output(output, input_video, weights, *protected)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.rmdir()
        stage.rename(output)
    except Exception as exc:
        _write_json(stage / "FAILED.json", {"status": "failed", "error": str(exc), "requested_output": str(output)})
        raise ValueError(f"Run failed: {exc}. Partial diagnostics remain in {stage}; no completed result was published.") from exc
    quality = summary["tracking_quality"]
    print(f"Completed: {output}", flush=True)
    print(f"Shuttle observed {quality['source_counts']['detected']}/{len(detections)} frames; interpolated {quality['source_counts']['interpolated']}; still missing {quality['source_counts']['missing']}.", flush=True)
    if has_players:
        player_quality = summary["player_tracking_quality"]
        print(f"Player counts: {player_quality['observed_count_histogram']}; count mismatches: {player_quality['count_mismatch_frames']} frames; unique tracker IDs: {player_quality['unique_tracker_ids']}.", flush=True)
    if calibration is not None:
        print(f"Court positions: {court_quality['positions_available']}/{court_quality['player_observations']} observations; calibration {court_quality['calibration_status']}.", flush=True)
    print("Open annotated.mp4 for review. Scores, coverage and tracker IDs do not establish accuracy.", flush=True)
    return output


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = _load_settings(args)
        execute(args, settings)
        return 0
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Install this project's requirements in its G drive environment, then run again.", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
