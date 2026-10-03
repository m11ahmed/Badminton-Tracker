"""Inspect actual M3 videos, timing and unchanged cached player observations.

This checks real encoded artifacts. Reprojection consistency is a software/math
check, not independent physical accuracy. Optional manual line checks measure
visible intersections that were not used to fit the four calibration corners.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from badminton_tracker.cli import _contained_path
from badminton_tracker.court import load_calibration
from badminton_tracker.video import iter_rgb_frames, probe_video


RAW_PLAYER_FIELDS = ("track_id", "bbox", "confidence", "keypoints", "ankle_midpoint",
                     "low_confidence", "identity_source")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def decode_for_review(path, selected):
    times, images, dimensions = [], {}, None
    for index, timestamp, rgb in iter_rgb_frames(path):
        times.append(timestamp)
        dimensions = [rgb.shape[1], rgb.shape[0]]
        if index in selected:
            images[index] = Image.fromarray(rgb)
    if not times:
        raise RuntimeError(f"No decoded frames in {path}.")
    return {"frames": len(times), "timestamps": times, "dimensions": dimensions,
            "images": images, "probe": probe_video(path)}


def require_aligned(left, right, label):
    if len(left) != len(right):
        raise RuntimeError(f"{label}: frame counts differ ({len(left)} versus {len(right)}).")
    error = max(abs(a - b) for a, b in zip(left, right))
    if error > 1 / 90000 + 1e-8:
        raise RuntimeError(f"{label}: timestamps differ by {error}s.")
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/m3-sample")
    parser.add_argument("--upstream", type=Path, default=ROOT / "results/m2-verified-sample")
    parser.add_argument("--expected-frames", type=int, default=81)
    parser.add_argument("--line-checks", type=Path,
                        help="JSON list or {points: [...]} of independent name,image_px,court_m checks.")
    args = parser.parse_args()
    results = _contained_path(args.results, "M3 result directory")
    upstream = _contained_path(args.upstream, "Upstream player result directory")
    players = read_json(results / "players.json")
    frames = players["frames"]
    context = players["metadata"]
    source_path = Path(context["input_video"])
    if len(frames) != args.expected_frames:
        raise RuntimeError(f"Expected {args.expected_frames} player frames, got {len(frames)}.")
    selected = sorted(set(index for index in [0, 16, 36, 48, 64, len(frames) - 1]
                          if 0 <= index < len(frames)))
    source = decode_for_review(source_path, selected)
    annotated = decode_for_review(results / "annotated.mp4", selected)
    minimap = decode_for_review(results / "minimap.mp4", selected)
    record_times = [frame["timestamp_s"] for frame in frames]
    errors = {
        "source_to_player_records_s": require_aligned(source["timestamps"], record_times, "Player records"),
        "source_to_annotated_s": require_aligned(source["timestamps"], annotated["timestamps"], "Annotated video"),
        "source_to_minimap_s": require_aligned(source["timestamps"], minimap["timestamps"], "Minimap video"),
    }
    if [frame["frame_index"] for frame in frames] != list(range(len(frames))):
        raise RuntimeError("Player frame indices are not consecutive.")
    w, h = source["dimensions"]
    if annotated["dimensions"] != [w + 320 + (w + 320) % 2, h + h % 2]:
        raise RuntimeError("Combined video changed the expected source plus 320 pixel map dimensions.")
    if minimap["dimensions"] != [320, 720]:
        raise RuntimeError("Standalone minimap dimensions differ from 320 by 720.")
    if annotated["probe"]["codec"] != "h264" or minimap["probe"]["codec"] != "h264":
        raise RuntimeError("Both exported videos must be H.264.")
    calibration = load_calibration(results / "court_calibration.json", w, h)
    old_frames = read_json(upstream / "players.json")["frames"]
    if len(old_frames) != len(frames):
        raise RuntimeError("Upstream and M3 player counts differ.")
    unchanged = 0
    for old, new in zip(old_frames, frames):
        if len(old["players"]) != len(new["players"]):
            raise RuntimeError("Player observations changed while reusing inference.")
        for a, b in zip(old["players"], new["players"]):
            if any(a.get(field) != b.get(field) for field in RAW_PLAYER_FIELDS):
                raise RuntimeError(f"Raw player observation changed at frame {new['frame_index']}.")
            unchanged += 1
    positions, missing, reprojection_errors = [], 0, []
    for frame in frames:
        for player in frame["players"]:
            position = player.get("court_position")
            if position is None:
                missing += 1
                if player.get("ankle_midpoint") is not None and "court_projection_unavailable" not in player.get("court_flags", []):
                    raise RuntimeError("An available anchor was lost without a projection flag.")
                continue
            x, y = position["x_m"], position["y_m"]
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (x, y)):
                raise RuntimeError("Court positions contain invalid coordinates.")
            if player.get("ankle_midpoint") is None:
                raise RuntimeError("Court position was manufactured without an ankle midpoint.")
            positions.append((x, y))
            back = calibration.court_to_image([[x, y]])[0]
            reprojection_errors.append(float(np.linalg.norm(back - np.asarray(player["ankle_midpoint"]))))
    line_checks = []
    if args.line_checks:
        data = read_json(args.line_checks)
        checks = data["points"] if isinstance(data, dict) else data
        for point in checks:
            measured_px, known_m = np.asarray(point["image_px"], float), np.asarray(point["court_m"], float)
            if measured_px.shape != (2,) or known_m.shape != (2,) or not np.isfinite([measured_px, known_m]).all():
                raise RuntimeError("Independent line checks must contain finite 2D image and court coordinates.")
            predicted_px = calibration.court_to_image([known_m.tolist()])[0]
            measured_m = calibration.image_to_court([measured_px.tolist()])[0]
            line_checks.append({"name": point["name"], "observed_image_px": measured_px.tolist(),
                "known_court_m": known_m.tolist(), "predicted_image_px": predicted_px.tolist(),
                "line_intersection_error_px": float(np.linalg.norm(measured_px - predicted_px)),
                "projected_court_m": measured_m.tolist(),
                "line_intersection_error_m": float(np.linalg.norm(measured_m - known_m))})
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    visualizations = ROOT / "visualizations"
    visualizations.mkdir(exist_ok=True)
    sheet = Image.new("RGB", (1600, 3 * 405 + 78), "#111923")
    drawing = ImageDraw.Draw(sheet)
    drawing.text((16, 12), "M3 actual encoded output: source video plus court minimap", font=font, fill="white")
    drawing.text((16, 40), "Court points are approximate ankle floor projections; raw tracker IDs remain fragments", font=font, fill="#d3dfec")
    map_sheet = Image.new("RGB", (960, 2 * 758 + 70), "#111923")
    map_drawing = ImageDraw.Draw(map_sheet)
    map_drawing.text((16, 12), "M3 standalone minimap: actual decoded output frames", font=font, fill="white")
    map_drawing.text((16, 40), "Empty or missing observations retain gaps. Source presentation timestamps are preserved.", font=font, fill="#d3dfec")
    for offset, index in enumerate(selected):
        x0, y0 = (offset % 2) * 800, (offset // 2) * 405 + 78
        sheet.paste(annotated["images"][index].resize((800, 360)), (x0, y0))
        available = sum(player.get("court_position") is not None for player in frames[index]["players"])
        drawing.text((x0 + 12, y0 + 366), f"Frame {index} | {record_times[index]:.3f}s | court anchors {available}/{len(frames[index]['players'])}", font=font, fill="white")
        mx, my = (offset % 3) * 320, (offset // 3) * 758 + 70
        map_sheet.paste(minimap["images"][index], (mx, my))
        map_drawing.text((mx + 12, my + 724), f"Frame {index} | {record_times[index]:.3f}s", font=font, fill="white")
    sheet_path, map_path = visualizations / f"{results.name}-contact-sheet.jpg", visualizations / f"{results.name}-minimap-contact-sheet.jpg"
    sheet.save(sheet_path, quality=94)
    map_sheet.save(map_path, quality=94)
    report = {
        "inspection_scope": "Actual decoded H.264 outputs, source timing, cache preservation and projection consistency; visual inspection of generated sheets is a separate step.",
        "source_frames": source["frames"], "annotated_frames": annotated["frames"], "minimap_frames": minimap["frames"],
        "source_dimensions": source["dimensions"], "annotated_dimensions": annotated["dimensions"], "minimap_dimensions": minimap["dimensions"],
        "timestamp_errors": errors, "unchanged_raw_player_observations": unchanged,
        "observed_player_count_histogram": dict(Counter(str(len(frame["players"])) for frame in frames)),
        "court_positions_available": len(positions), "court_positions_missing": missing,
        "court_x_range_m": [min(point[0] for point in positions), max(point[0] for point in positions)] if positions else None,
        "court_y_range_m": [min(point[1] for point in positions), max(point[1] for point in positions)] if positions else None,
        "maximum_inverse_projection_consistency_error_px": max(reprojection_errors, default=None),
        "independent_visible_line_checks": line_checks,
        "annotated_probe": annotated["probe"], "minimap_probe": minimap["probe"],
        "contact_sheet": str(sheet_path), "minimap_contact_sheet": str(map_path),
        "limitations": ["Inverse projection consistency and four-corner fit residuals do not measure independent physical accuracy.",
            "Optional visible line intersections are approximate manual pixel measurements, not surveyed court ground truth.",
            "Jumping, occluded ankles, pose error, lens distortion and camera movement can change the floor estimate.",
            "Browser playback and full-match identity accuracy are not established by these decoded frame checks."],
    }
    (results / "video_verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
