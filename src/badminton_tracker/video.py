"""Video decoding and an H.264 overlay with source presentation timing."""
from collections import deque
from fractions import Fraction
from pathlib import Path
import math

from .postprocess import validate_detections
from .minimap import court_line_segments, render_minimap, track_color

import av
import cv2
import numpy as np


class VideoError(ValueError):
    """An input cannot be decoded safely or output cannot be encoded."""


def _valid_rate(value):
    try:
        rate = float(value)
        return rate if math.isfinite(rate) and rate > 0 else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def probe_video(path: Path, fps_hint: float | None = None) -> dict:
    path = Path(path)
    if not path.is_file():
        raise VideoError(f"Video does not exist: {path}. Pass an existing file to --video.")
    if fps_hint is not None and _valid_rate(fps_hint) is None:
        raise VideoError("--fps-hint must be a positive finite number.")
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise VideoError(f"No video stream found in {path}.")
            stream = container.streams.video[0]
            width, height = stream.codec_context.width, stream.codec_context.height
            if width <= 0 or height <= 0:
                raise VideoError("Video metadata has invalid image dimensions.")
            metadata_fps = _valid_rate(stream.average_rate)
            fps_source = "average_rate"
            if metadata_fps is None:
                metadata_fps = _valid_rate(stream.guessed_rate)
                fps_source = "guessed_rate"
            if metadata_fps is None:
                metadata_fps = _valid_rate(stream.base_rate)
                fps_source = "base_rate"
            fps = metadata_fps or fps_hint
            if fps is None:
                raise VideoError("Video has no usable frame rate. Supply --fps-hint for missing timestamp fallback.")
            if metadata_fps is None:
                fps_source = "fps_hint"
            warnings = []
            if metadata_fps and fps_hint and abs(metadata_fps - fps_hint) / metadata_fps > 0.01:
                warnings.append(
                    f"FPS hint {fps_hint:g} differs from metadata {metadata_fps:g}; source timestamps and metadata take precedence."
                )
            duration = None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = container.duration / av.time_base
            if duration is not None and (not math.isfinite(duration) or duration < 0):
                raise VideoError("Video metadata contains an invalid duration.")
            return {
                "width": width,
                "height": height,
                "fps": float(fps),
                "metadata_fps": metadata_fps,
                "fps_source": fps_source,
                "frame_count": int(stream.frames) if stream.frames > 0 else None,
                "duration_s": duration,
                "time_base": str(stream.time_base),
                "codec": stream.codec_context.name,
                "timing_policy": "source presentation timestamps relative to first frame; rate fallback only when PTS is missing",
                "timing_warnings": warnings,
            }
    except VideoError:
        raise
    except (av.FFmpegError, OSError, ValueError) as exc:
        raise VideoError(f"Cannot read video {path}: {exc}") from exc


def iter_rgb_frames(path: Path, *, max_frames: int | None = None, fps_hint: float | None = None):
    """Yield (zero based index, relative timestamp in seconds, uint8 RGB array).

    Presentation timestamps are authoritative. Non increasing timestamps are an
    actionable error instead of silently manufacturing movement timing.
    """
    if max_frames is not None and max_frames < 1:
        raise VideoError("max_frames must be a positive integer.")
    metadata = probe_video(path, fps_hint)
    fallback_step = 1.0 / metadata["fps"]
    origin = None
    previous = None
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for index, frame in enumerate(container.decode(stream)):
                if max_frames is not None and index >= max_frames:
                    break
                absolute = None
                if frame.pts is not None and frame.time_base is not None:
                    absolute = float(frame.pts * frame.time_base)
                    if not math.isfinite(absolute):
                        raise VideoError(f"Frame {index} has a non finite presentation timestamp.")
                if absolute is not None:
                    if origin is None:
                        origin = absolute - index * fallback_step
                    timestamp = absolute - origin
                else:
                    timestamp = 0.0 if previous is None else previous + fallback_step
                if timestamp < -1e-6 or (previous is not None and timestamp <= previous):
                    raise VideoError(
                        f"Frame {index} has invalid or non increasing source timing ({timestamp:.6f}s). "
                        "Use a clean export of the source video with valid presentation timestamps."
                    )
                previous = timestamp
                yield index, max(0.0, timestamp), frame.to_ndarray(format="rgb24")
    except VideoError:
        raise
    except (av.FFmpegError, OSError, ValueError) as exc:
        raise VideoError(f"Failed while decoding video {path}: {exc}") from exc


SKELETON_EDGES = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
)


def _track_color(track_id):
    """Deterministic BGR color shared between source overlay and court map."""
    return track_color(track_id)


def _draw_court_lines(bgr, calibration):
    """Project known floor markings onto source pixels without changing scale."""
    for a, b, kind in court_line_segments():
        projected = np.asarray(calibration.court_to_image(np.asarray([a, b], dtype=np.float64)), dtype=np.float64)
        if projected.shape != (2, 2) or not np.isfinite(projected).all():
            raise VideoError("Court projection must return two finite source image points.")
        if np.any(np.abs(projected) > 2 ** 30):
            raise VideoError("Court projection is too far outside the source image to draw safely.")
        left, right = (tuple(round(float(value)) for value in point) for point in projected)
        color = (55, 185, 245) if kind == "net" else (180, 220, 200)
        cv2.line(bgr, left, right, color, 2 if kind in ("boundary", "net") else 1, cv2.LINE_AA)


def _visible_keypoint(keypoint, width, height):
    x, y = keypoint.get("x"), keypoint.get("y")
    return bool(keypoint.get("visible")) and isinstance(x, (int, float)) and isinstance(y, (int, float)) and math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height and not (x == 0 and y == 0)


def _draw_players(bgr, frame_record, player_roi=None):
    height, width = bgr.shape[:2]
    if player_roi is not None:
        vertices = np.asarray([(round(x * (width - 1)), round(y * (height - 1))) for x, y in player_roi], dtype=np.int32)
        cv2.polylines(bgr, [vertices], True, (160, 160, 160), 1, cv2.LINE_AA)
    for player in frame_record["players"]:
        color = _track_color(player.get("track_id"))
        x1, y1, x2, y2 = (round(value) for value in player["bbox"])
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        track_id = player.get("track_id")
        identity = f"ID {track_id}" if track_id is not None else "ID unassigned"
        suffix = " | LOW" if player.get("low_confidence") else ""
        label = identity + suffix
        font_scale = max(0.4, min(0.7, width / 1800))
        cv2.putText(bgr, label, (max(0, x1), max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (15, 15, 15), 3, cv2.LINE_AA)
        cv2.putText(bgr, label, (max(0, x1), max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
        keypoints = {point["name"]: point for point in player.get("keypoints", [])}
        for left, right in SKELETON_EDGES:
            a, b = keypoints.get(left), keypoints.get(right)
            if a and b and _visible_keypoint(a, width, height) and _visible_keypoint(b, width, height):
                cv2.line(bgr, (round(a["x"]), round(a["y"])), (round(b["x"]), round(b["y"])), color, 2, cv2.LINE_AA)
        for point in keypoints.values():
            if _visible_keypoint(point, width, height):
                cv2.circle(bgr, (round(point["x"]), round(point["y"])), 3, color, -1, cv2.LINE_AA)
        ankle = player.get("ankle_midpoint")
        if ankle is not None and all(math.isfinite(value) for value in ankle):
            cv2.drawMarker(bgr, (round(ankle[0]), round(ankle[1])), color, cv2.MARKER_CROSS, 10, 2, cv2.LINE_AA)


def _validate_player_frames(player_frames, detections):
    if len(player_frames) != len(detections):
        raise VideoError("Player records and shuttle records must have the same frame count.")
    for index, (record, detection) in enumerate(zip(player_frames, detections)):
        timestamp = record.get("timestamp_s")
        if record.get("frame_index") != index or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or abs(timestamp - detection.timestamp_s) > 1e-5:
            raise VideoError("Player records do not match shuttle/source frame indices and timestamps.")
        players = record.get("players")
        if not isinstance(players, list) or record.get("observed_player_count") != len(players):
            raise VideoError("Player record observed count must equal the number of player observations.")
        for player in players:
            bbox = player.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in bbox):
                raise VideoError("Player boxes must contain four finite coordinates.")


def write_overlay(video_path: Path, output_path: Path, detections: list, *, fps_hint=None, trail_frames=12, player_frames=None, player_roi=None, court_calibration=None, motion_insights=None) -> dict:
    """Draw observations on source frames and encode a silent, timed H.264 MP4."""
    if not detections:
        raise VideoError("No video frames were analyzed; no overlay can be written.")
    validate_detections(detections)
    if player_frames is not None:
        _validate_player_frames(player_frames, detections)
    elif player_roi is not None:
        raise VideoError("A player ROI overlay requires player records.")
    if court_calibration is not None and player_frames is None:
        raise VideoError("A calibrated court overlay requires player frame records.")
    shuttle_motion = {}
    player_motion = {}
    if motion_insights is not None:
        if not isinstance(motion_insights, dict) or player_frames is None or court_calibration is None:
            raise VideoError("Movement overlay requires insights, player records and court calibration.")
        rows = motion_insights.get("shuttle_motion", [])
        if len(rows) != len(detections):
            raise VideoError("Shuttle motion rows must match all overlay frames.")
        for row, detection in zip(rows, detections):
            if row.get("frame_index") != detection.frame_index or abs(row.get("timestamp_s", -1) - detection.timestamp_s) > 1e-5:
                raise VideoError("Shuttle motion indices and timestamps must match the source.")
            shuttle_motion[row["frame_index"]] = row
        for row in motion_insights.get("player_motion", []):
            player_motion.setdefault(row["frame_index"], []).append(row)
    metadata = probe_video(video_path, fps_hint)
    width, height = metadata["width"], metadata["height"]
    minimap_width = 320 if court_calibration is not None else 0
    canvas_width = width + minimap_width
    minimap_trails = {}
    mode = "doubles" if player_frames and player_frames[0].get("expected_player_count") == 4 else "singles"
    # yuv420p requires even dimensions. Padding preserves source pixel positions.
    encoded_width, encoded_height = canvas_width + canvas_width % 2, height + height % 2
    time_base = Fraction(1, 90000)
    rate = Fraction(str(metadata["fps"])).limit_denominator(100000)
    trail = deque(maxlen=trail_frames)
    frame_count = 0
    try:
        with av.open(str(output_path), mode="w", format="mp4", options={"movflags": "+faststart"}) as output:
            stream = output.add_stream("libx264", rate=rate)
            stream.width, stream.height = encoded_width, encoded_height
            stream.pix_fmt = "yuv420p"
            stream.time_base = time_base
            stream.codec_context.time_base = time_base
            stream.codec_context.max_b_frames = 0
            stream.options = {"crf": "22", "preset": "fast"}
            last_pts = None
            for index, timestamp, rgb in iter_rgb_frames(video_path, max_frames=len(detections), fps_hint=fps_hint):
                detection = detections[index]
                if detection.frame_index != index or abs(detection.timestamp_s - timestamp) > 1e-5:
                    raise VideoError("Detection records do not match decoded source frame indices and timestamps.")
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if court_calibration is not None:
                    _draw_court_lines(bgr, court_calibration)
                if player_frames is not None:
                    _draw_players(bgr, player_frames[index], player_roi)
                valid = detection.x is not None and detection.y is not None
                if valid:
                    point = (round(detection.x), round(detection.y))
                    if detection.source == "interpolated":
                        trail.clear()
                    else:
                        trail.append(point)
                    color = (0, 195, 255) if detection.low_confidence or detection.source == "interpolated" else (80, 255, 80)
                    if len(trail) > 1:
                        cv2.polylines(bgr, [np.asarray(trail, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
                    cv2.circle(bgr, point, 7, color, 2, cv2.LINE_AA)
                    cv2.circle(bgr, point, 2, color, -1, cv2.LINE_AA)
                else:
                    # An unfilled gap always starts a new trail.
                    trail.clear()
                score = "n/a" if detection.confidence is None else f"{detection.confidence:.3f}"
                status = detection.source + (" | LOW CONFIDENCE" if detection.low_confidence else "")
                title = "M1 SHUTTLE" if player_frames is None else "M2 SHUTTLE + PLAYERS"
                if court_calibration is not None:
                    proposal = "unconfirmed_court_calibration" in player_frames[index].get("court_quality_flags", [])
                    title = "M3 COURT PROPOSAL | UNCONFIRMED" if proposal else "M3 SHUTTLE + COURT"
                lines = [f"{title} | frame {index} | {timestamp:.3f}s", f"{status} | heatmap score {score}"]
                if player_frames is not None:
                    record = player_frames[index]
                    count_status = " | COUNT MISMATCH" if record["count_mismatch"] else ""
                    lines.append(f"Players {record['observed_player_count']}/{record['expected_player_count']}{count_status}")
                    lines.append("IDs: tracker trajectories | pose: visible points only")
                if motion_insights is not None:
                    lines[0] = lines[0].replace("M3", "M5")
                    motion = shuttle_motion[index].get("image_speed_px_s")
                    motion_text = "unavailable" if motion is None else f"{motion:.0f} px/s"
                    lines.append(f"Shuttle image motion: {motion_text} | physical km/h unavailable")
                    values = []
                    for row in player_motion.get(index, []):
                        speed = row.get("speed_m_s")
                        value = "n/a" if speed is None else f"{speed:.2f} m/s"
                        values.append(f"ID {row['track_id']}: {value}")
                    lines.append("Estimated player motion | " + " | ".join(values))
                font_scale = max(0.4, min(0.65, width / 1800))
                text_height = max(18, round(34 * font_scale))
                cv2.rectangle(bgr, (0, 0), (min(width, 1000 if motion_insights is not None else 630), text_height * len(lines) + 12), (20, 20, 20), -1)
                for row, line in enumerate(lines):
                    cv2.putText(bgr, line, (8, 8 + text_height * (row + 1)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (240, 240, 240), 1, cv2.LINE_AA)
                if court_calibration is not None:
                    panel = render_minimap(player_frames[index], width=minimap_width, height=height, trail_state=minimap_trails, mode=mode)
                    bgr = np.concatenate((bgr, panel), axis=1)
                if encoded_height != height or encoded_width != canvas_width:
                    bgr = cv2.copyMakeBorder(bgr, 0, encoded_height - height, 0, encoded_width - canvas_width, cv2.BORDER_CONSTANT)
                frame = av.VideoFrame.from_ndarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), format="rgb24")
                pts = round(timestamp / float(time_base))
                if last_pts is not None and pts <= last_pts:
                    raise VideoError("Source frame timing exceeds the 90 kHz output timestamp resolution.")
                frame.pts = pts
                frame.time_base = time_base
                last_pts = pts
                for packet in stream.encode(frame):
                    output.mux(packet)
                frame_count += 1
            for packet in stream.encode():
                output.mux(packet)
        if frame_count != len(detections):
            raise VideoError("Overlay decode ended before all analyzed frames were rendered.")
        return {
            "frames_written": frame_count,
            "codec": "h264",
            "pixel_format": "yuv420p",
            "source_width": width,
            "source_height": height,
            "encoded_width": encoded_width,
            "encoded_height": encoded_height,
            "audio": "none; this overlay is silent",
            "player_layer": player_frames is not None,
            "player_roi_drawn": player_roi is not None,
            "timestamp_time_base": "1/90000",
            "court_layer": court_calibration is not None,
            "movement_layer": motion_insights is not None,
            "minimap_panel_width": minimap_width,
            "source_pixel_policy": "original source pixels remain at original coordinates; map panel is appended to the right",
        }
    except VideoError:
        raise
    except (av.FFmpegError, OSError, ValueError) as exc:
        raise VideoError(f"Could not encode H.264 overlay: {exc}. Ensure PyAV includes libx264.") from exc