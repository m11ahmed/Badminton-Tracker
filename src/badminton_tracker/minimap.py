"""Court-coordinate review panels and silent H.264 minimap video export.

The map contains player floor references only. Raw tracker IDs remain fragments,
source coordinates are never clamped onto the court, and gaps break every trail.
"""
from collections import deque
from fractions import Fraction
import math
from pathlib import Path

import av
import cv2
import numpy as np

COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4
SINGLES_WIDTH_M = 5.18
TRAIL_FRAMES = 60
APRON_M = 0.8


def track_color(track_id):
    """Deterministic BGR trajectory color shared with the source-video overlay."""
    if track_id is None:
        return (180, 180, 180)
    hue = (int(track_id) * 67) % 180
    color = cv2.cvtColor(np.array([[[hue, 220, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(value) for value in color)


def court_line_segments():
    """Return physical doubles, singles, net and service segments in metres."""
    side = (COURT_WIDTH_M - SINGLES_WIDTH_M) / 2
    net = COURT_LENGTH_M / 2
    segments = [
        ((0, 0), (COURT_WIDTH_M, 0), "boundary"),
        ((COURT_WIDTH_M, 0), (COURT_WIDTH_M, COURT_LENGTH_M), "boundary"),
        ((COURT_WIDTH_M, COURT_LENGTH_M), (0, COURT_LENGTH_M), "boundary"),
        ((0, COURT_LENGTH_M), (0, 0), "boundary"),
        ((side, 0), (side, COURT_LENGTH_M), "singles"),
        ((COURT_WIDTH_M - side, 0), (COURT_WIDTH_M - side, COURT_LENGTH_M), "singles"),
        ((0, net), (COURT_WIDTH_M, net), "net"),
        ((0, net - 1.98), (COURT_WIDTH_M, net - 1.98), "service"),
        ((0, net + 1.98), (COURT_WIDTH_M, net + 1.98), "service"),
        ((0, 0.76), (COURT_WIDTH_M, 0.76), "service"),
        ((0, COURT_LENGTH_M - 0.76), (COURT_WIDTH_M, COURT_LENGTH_M - 0.76), "service"),
        ((COURT_WIDTH_M / 2, 0), (COURT_WIDTH_M / 2, net - 1.98), "service"),
        ((COURT_WIDTH_M / 2, net + 1.98), (COURT_WIDTH_M / 2, COURT_LENGTH_M), "service"),
    ]
    return segments


def _court_layout(width, height):
    margin = min(24.0, width * 0.075)
    header, footer = min(72.0, height * 0.22), min(90.0, height * 0.28)
    usable_height = height - header - footer
    scale = min((width - 2 * margin) / (COURT_WIDTH_M + 2 * APRON_M), usable_height / (COURT_LENGTH_M + 2 * APRON_M))
    left = (width - COURT_WIDTH_M * scale) / 2
    top = header + (usable_height - COURT_LENGTH_M * scale) / 2
    return left, top, scale


def _text(image, text, origin, *, scale=0.42, color=(220, 230, 230)):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _position(player):
    position = player.get("court_position")
    if not isinstance(position, dict):
        return None
    x, y = position.get("x_m"), position.get("y_m")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in (x, y)):
        return None
    return float(x), float(y), position


def render_minimap(frame_record, *, width=320, height=720, trail_state=None, mode="singles"):
    """Render a BGR uint8 review map, optionally retaining short contiguous trails.

    ``trail_state`` is a caller-owned dict keyed by raw track ID. Its values hold
    ``last_frame_index`` and ``points`` (a bounded deque of metre coordinates).
    Invalid, missing or off-panel positions clear that trajectory. IDs absent
    from the current frame are removed, preventing a trail across an occlusion.
    """
    if type(width) is not int or type(height) is not int or width < 160 or height < 120:
        raise ValueError("Minimap width must be at least 160 and height at least 120 pixels.")
    if mode not in ("singles", "doubles"):
        raise ValueError("Minimap mode must be singles or doubles.")
    if not isinstance(frame_record, dict) or not isinstance(frame_record.get("players"), list):
        raise ValueError("Minimap requires a frame record with a players list.")
    index, timestamp = frame_record.get("frame_index"), frame_record.get("timestamp_s")
    if type(index) is not int or index < 0 or not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool) or not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("Minimap frame index and timestamp must be valid source values.")
    if trail_state is not None and not isinstance(trail_state, dict):
        raise ValueError("Minimap trail_state must be a dictionary or None.")
    proposal = "unconfirmed_court_calibration" in frame_record.get("court_quality_flags", [])
    state = {} if trail_state is None else trail_state
    image = np.full((height, width, 3), (20, 28, 30), dtype=np.uint8)
    left, top, scale = _court_layout(width, height)
    def pixel(x, y):
        return round(left + x * scale), round(top + y * scale)
    def on_panel(x, y):
        return -APRON_M <= x <= COURT_WIDTH_M + APRON_M and -APRON_M <= y <= COURT_LENGTH_M + APRON_M
    board_left, board_top = pixel(-APRON_M, -APRON_M)
    board_right, board_bottom = pixel(COURT_WIDTH_M + APRON_M, COURT_LENGTH_M + APRON_M)
    cv2.rectangle(image, (board_left, board_top), (board_right, board_bottom), (30, 43, 34), -1)
    cv2.rectangle(image, pixel(0, 0), pixel(COURT_WIDTH_M, COURT_LENGTH_M), (48, 75, 48), -1)
    for a, b, kind in court_line_segments():
        color = (85, 195, 240) if kind == "net" else ((185, 215, 225) if kind == "singles" else (190, 205, 195))
        thickness = 2 if kind in ("boundary", "net") else 1
        cv2.line(image, pixel(*a), pixel(*b), color, thickness, cv2.LINE_AA)
    label_scale = max(0.28, min(0.43, width / 760))
    if height >= 300:
        _text(image, "COURT PROPOSAL" if proposal else "COURT POSITION", (14, 22), scale=0.5, color=(100, 185, 245) if proposal else (220, 230, 230))
        _text(image, f"Frame {index} | {timestamp:.3f}s", (14, 43), scale=label_scale)
        _text(image, "FAR  y=0 m", (round(left), max(58, round(top - 10))), scale=label_scale)
        _text(image, "NEAR  y=13.4 m", (round(left), round(top + COURT_LENGTH_M * scale + 19)), scale=label_scale)
        _text(image, "NET", (min(width - 35, round(left + COURT_WIDTH_M * scale + 5)), round(top + COURT_LENGTH_M * scale / 2 + 4)), scale=0.3)
    else:
        title = "COURT PROPOSAL" if proposal else "COURT"
        _text(image, f"{title} | {timestamp:.3f}s", (8, 16), scale=label_scale)
        _text(image, "FAR", (round(left), max(29, round(top - 5))), scale=0.28)
        _text(image, "NEAR", (round(left), round(top + COURT_LENGTH_M * scale + 12)), scale=0.28)
    current_ids = set()
    missing, outside, off_panel = 0, 0, 0
    for player in frame_record["players"]:
        track_id = player.get("track_id")
        if track_id is not None:
            current_ids.add(track_id)
        position = _position(player)
        if position is None:
            missing += 1
            state.pop(track_id, None)
            continue
        x, y, details = position
        inside = details.get("in_singles_court") if mode == "singles" else details.get("in_doubles_court")
        if inside is None:
            side = (COURT_WIDTH_M - SINGLES_WIDTH_M) / 2 if mode == "singles" else 0
            inside = side <= x <= COURT_WIDTH_M - side and 0 <= y <= COURT_LENGTH_M
        outside += not bool(inside)
        if not on_panel(x, y):
            # Preserve exported metres; report this point instead of clamping it.
            off_panel += 1
            state.pop(track_id, None)
            continue
        color = track_color(track_id)
        if track_id is not None:
            previous = state.get(track_id)
            if previous is None or previous.get("last_frame_index") != index - 1:
                previous = {"last_frame_index": index, "points": deque(maxlen=TRAIL_FRAMES)}
                state[track_id] = previous
            previous["points"].append((x, y))
            previous["last_frame_index"] = index
            if len(previous["points"]) > 1:
                points = np.asarray([pixel(*point) for point in previous["points"]], dtype=np.int32)
                cv2.polylines(image, [points], False, color, 1, cv2.LINE_AA)
        center = pixel(x, y)
        cv2.circle(image, center, 6, (12, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(image, center, 5, color, -1 if inside else 2, cv2.LINE_AA)
        label = f"{track_id}" if track_id is not None else "?"
        _text(image, label, (center[0] + 7, center[1] - 5), scale=0.37, color=color)
    for track_id in list(state):
        if track_id not in current_ids:
            del state[track_id]
    if height >= 300:
        baseline = max(board_bottom + 27, height - 65)
        _text(image, "x: 0 to 6.1 m | y: 0 to 13.4 m", (14, baseline), scale=0.36)
        _text(image, f"{mode.title()} width {SINGLES_WIDTH_M if mode == 'singles' else COURT_WIDTH_M:g} m", (14, baseline + 17), scale=0.38)
        warning = f"Outside: {outside} | no anchor: {missing}"
        if off_panel:
            warning += f" | off map: {off_panel}"
        _text(image, warning, (14, baseline + 34), scale=0.34, color=(135, 195, 235) if outside or missing else (175, 200, 185))
        footer = "Check court corners | raw IDs" if proposal else "Approximate floor points | raw IDs"
        _text(image, footer, (14, min(height - 5, baseline + 51)), scale=0.34, color=(100, 185, 245) if proposal else (170, 185, 185))
    else:
        footer = "Check court corners" if proposal else f"6.1 x 13.4 m | outside {outside}"
        _text(image, footer, (8, height - 7), scale=0.28)
    return image


def _validate_minimap_frames(player_frames, fps):
    if not isinstance(fps, (float, int)) or isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Minimap video fps must be a positive finite number.")
    if not isinstance(player_frames, list) or not player_frames:
        raise ValueError("Minimap video requires nonempty player frame records.")
    previous = None
    previous_pts = None
    for index, frame in enumerate(player_frames):
        if not isinstance(frame, dict) or frame.get("frame_index") != index or not isinstance(frame.get("players"), list):
            raise ValueError("Minimap video frame indices must be consecutive and start at zero.")
        timestamp = frame.get("timestamp_s")
        if not isinstance(timestamp, (float, int)) or isinstance(timestamp, bool) or not math.isfinite(timestamp) or timestamp < 0 or (previous is not None and timestamp <= previous):
            raise ValueError("Minimap timestamps must be finite, nonnegative and strictly increasing.")
        pts = round(timestamp * 90000)
        if previous_pts is not None and pts <= previous_pts:
            raise ValueError("Minimap source timing exceeds the 90 kHz timestamp resolution.")
        previous, previous_pts = timestamp, pts


def write_minimap_video(output, player_frames, *, fps, calibration=None):
    """Encode a standalone 320 by 720 map with each record's source timestamp."""
    _validate_minimap_frames(player_frames, fps)
    output = Path(output)
    width, height = 320, 720
    time_base = Fraction(1, 90000)
    rate = Fraction(str(fps)).limit_denominator(100000)
    state = {}
    mode = "doubles" if player_frames[0].get("expected_player_count") == 4 else "singles"
    try:
        with av.open(str(output), mode="w", format="mp4", options={"movflags": "+faststart"}) as container:
            stream = container.add_stream("libx264", rate=rate)
            stream.width, stream.height = width, height
            stream.pix_fmt = "yuv420p"
            stream.time_base = time_base
            stream.codec_context.time_base = time_base
            stream.codec_context.max_b_frames = 0
            stream.options = {"crf": "22", "preset": "fast"}
            for record in player_frames:
                bgr = render_minimap(record, width=width, height=height, trail_state=state, mode=mode)
                frame = av.VideoFrame.from_ndarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), format="rgb24")
                frame.pts = round(record["timestamp_s"] / float(time_base))
                frame.time_base = time_base
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except (av.FFmpegError, OSError, ValueError) as exc:
        raise ValueError(f"Could not encode H.264 minimap: {exc}. Ensure PyAV includes libx264.") from exc
    return {
        "frames_written": len(player_frames), "codec": "h264", "pixel_format": "yuv420p",
        "encoded_width": width, "encoded_height": height,
        "audio": "none; minimap video is silent", "timestamp_time_base": "1/90000",
        "timing_policy": "player records retain source presentation timestamps",
        "coordinate_units": "court metres; approximate player floor reference only",
        "orientation": "far-left (0, 0); near-right (6.1, 13.4)",
        "outside_policy": "nearby apron positions drawn without clamping; off-panel coordinates reported",
        "trail_policy": f"at most {TRAIL_FRAMES} valid consecutive frames per raw tracker ID; gaps break trails",
        "calibration_supplied": calibration is not None,
    }
