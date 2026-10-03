"""Portable CSV, JSON, heatmap and coaching-review exports for M5."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _heatmap(path, player):
    values = np.asarray(player["heatmap"]["values_s"], dtype=float)
    if values.shape != tuple(player["heatmap"]["grid_shape"]) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Heatmap must contain finite nonnegative observed seconds.")
    maximum = float(values.max()) if values.size else 0
    intensity = np.log1p(values) / math.log1p(maximum) if maximum else np.zeros_like(values)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = 15 + 240 * intensity
    rgb[:, :, 1] = 45 + 165 * np.minimum(1, intensity * 1.6)
    rgb[:, :, 2] = 40 + 30 * (1 - intensity)
    image = Image.new("RGB", (440, 880), "#101820")
    court = Image.fromarray(rgb).resize((360, 792), Image.Resampling.NEAREST)
    image.paste(court, (40, 45))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 45, 399, 836), outline="white", width=2)
    draw.line((40, 441, 399, 441), fill="white", width=2)
    # Center and short-service lines, preserving the 6.1 by 13.4 court aspect.
    for y_m in (4.72, 8.68):
        y = 45 + int(y_m / 13.4 * 792)
        draw.line((40, y, 399, y), fill="#d8e5e3", width=1)
    draw.line((220, 45, 220, 324), fill="#d8e5e3", width=1)
    draw.line((220, 558, 220, 837), fill="#d8e5e3", width=1)
    label = player.get("player_label") or f"Tracker fragment {player['track_id']}"
    # Pillow's default font is portable; unicode names are safe in current Pillow.
    draw.text((15, 10), label[:55], fill="white")
    draw.text((15, 26), "Far baseline above | Near baseline below", fill="#becacb")
    draw.text((15, 847), f"Observed occupancy: {player['valid_motion_time_s']:.2f} s", fill="white")
    draw.text((15, 863), f"Peak cell: {maximum:.3f} s | excluded time has no colour", fill="#becacb")
    image.save(path)


def _number(value, unit="", digits=2):
    return "unavailable" if value is None else f"{value:.{digits}f}{unit}"


def coaching_review(insights):
    """Describe observations and concrete review questions without tactical verdicts."""
    lines = ["# Badminton coaching review", "",
             "Review these estimates alongside the annotated video. Tracker IDs are fragments unless identities were manually mapped. Measurements describe accepted observations, not a full physical performance assessment.", "", "## Player movement", "",
             "| Fragment | Reviewed player | Distance | Accepted time | Average speed | Peak estimated speed |", "| --- | --- | --- | --- | --- | --- |"]
    for player in insights["players"]:
        label = (player.get("player_label") or "unmapped").replace("|", "/").replace("\n", " ")
        lines.append(f"| {player['track_id']} | {label} | {_number(player['distance_m'], ' m')} | {_number(player['valid_motion_time_s'], ' s')} | {_number(player['average_speed_m_s'], ' m/s')} | {_number(player['max_speed_m_s'], ' m/s')} |")
    if not insights["players"]:
        lines.append("| unavailable | unmapped | unavailable | 0 s | unavailable | unavailable |")
    lines.extend(["", "Distance uses smoothed consecutive accepted floor positions. Gaps and tracker fragments are never bridged. Average speed is distance divided by accepted interval time, including accepted stationary time.", "", "## Shuttle movement", "",
                  f"Average observed image motion: {_number(insights['shuttle']['average_image_speed_px_s'], ' px/s', 1)}. Largest detected image motion: {_number(insights['shuttle']['max_image_speed_px_s'], ' px/s', 1)}.", "",
                  "Abrupt consecutive jumps in the detector output receive a review flag. The flagged image speeds stay visible in the CSV and statistics so a coach can inspect the video; they do not distinguish a genuine shuttle movement from a false detection. Do not call the largest detected image motion the fastest shot.", "",
                  "Physical shuttle speed in km/h is unavailable. The airborne shuttle's depth and height cannot be recovered from a floor homography. Pixel speed changes with perspective and camera framing and must not be compared to official smash-speed records.", "", "## Positioning and rally review", ""])
    for player in insights["players"]:
        zones = player["zones"]
        top = sorted(zones.items(), key=lambda item: item[1], reverse=True)[:3]
        descriptions = ", ".join(f"{zone.replace('_', ' ')} ({seconds:.2f} s)" for zone, seconds in top) or "no accepted occupancy intervals"
        lines.append(f"Tracker fragment {player['track_id']}: {descriptions}. Heatmaps show accepted observed seconds, with far baseline at the top.")
    lines.extend(["", "For each reviewed contact, pause the overlay and discuss court coverage, recovery choice and the next shot with the coach. Occupancy alone does not establish whether a tactical position was correct.", ""])
    for rally in insights["rallies"]:
        if insights.get("review", {}).get("rallies"):
            lines.append(f"Reviewed rally {rally['id']}: frames {rally['start_frame']} through {rally['end_frame']}, {rally['duration_s']:.2f} s.")
        else:
            lines.append(f"Clip span: frames {rally['start_frame']} through {rally['end_frame']}, {rally['duration_s']:.2f} s. No rally boundaries were reviewed.")
        if rally.get("notes"):
            lines.append(f"Reviewed note: {rally['notes']}")
        if rally.get("outcome"):
            lines.append(f"Reviewed outcome: {rally['outcome']}")
    lines.extend(["", "## Doubles spacing", ""])
    spacing = insights["doubles_pair_spacing"]
    if spacing["status"] == "available":
        for pair in spacing["pairs"]:
            lines.append(f"Reviewed {pair['side']} pair {', '.join(pair['players'])}: median spacing {pair['median_spacing_m']:.2f} m across {pair['valid_frames']} valid observations.")
        lines.append("Spacing describes the reviewed pair; it does not classify an attacking or defensive formation.")
    else:
        lines.append(spacing["reason"])
    lines.extend(["", "## Recovery targets", ""])
    if insights["recovery"]:
        for recovery in insights["recovery"]:
            lines.append(f"{recovery['player_label']} after reviewed contact frame {recovery['contact_frame']}: {_number(recovery['recovery_time_s'], ' s')} ({recovery['status']}). {recovery.get('reason') or ''}")
    else:
        lines.append("No explicit contact and recovery targets were supplied; automatic recovery advice was not inferred.")
    lines.extend(["", "## Quality and limits", ""])
    for message in insights["quality"]["warnings"] + insights["limitations"]:
        lines.append(f"1. {message}")
    lines.extend(["", "Check IDs against the video before linking a report to an athlete. Coaching decisions should use the video, training context and player feedback together.", ""])
    return "\n".join(lines)


def export_insights(outdir, insights):
    """Export into an existing run stage; return all filenames relative to outdir."""
    destination = Path(outdir).resolve()
    if not destination.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Insight exports must stay inside the G drive project.")
    outputs = {
        "insights_json": "insights.json", "player_motion_csv": "player_motion.csv",
        "player_motion_json": "player_motion.json", "shuttle_motion_csv": "shuttle_motion.csv",
        "shuttle_motion_json": "shuttle_motion.json", "coaching_review": "coaching_review.md",
    }
    for player in insights["players"]:
        outputs[f"heatmap_track_{player['track_id']}"] = f"heatmaps/track_{player['track_id']}.png"
    # Validate serialization and conflicts before writing any output.
    json.dumps(insights, allow_nan=False)
    if any((destination / filename).exists() for filename in outputs.values()):
        raise ValueError("Insight export refuses to overwrite existing output files.")
    destination.mkdir(parents=True, exist_ok=True)
    if insights["players"]:
        (destination / "heatmaps").mkdir(exist_ok=True)
    _json(destination / outputs["insights_json"], insights)
    _json(destination / outputs["player_motion_json"], {"schema_version": "1.0", "units": "metres, seconds", "frames": insights["player_motion"]})
    _json(destination / outputs["shuttle_motion_json"], {"schema_version": "1.0", "units": "pixels, seconds", "physical_speed_km_h": None, "frames": insights["shuttle_motion"]})
    player_columns = ["frame_index", "timestamp_s", "track_id", "player_label", "side", "team", "x_m", "y_m", "valid_position", "smoothed_x_m", "smoothed_y_m", "interval_start_frame", "interval_start_timestamp_s", "interval_duration_s", "distance_m", "speed_m_s", "zone", "quality_flags"]
    shuttle_columns = ["frame_index", "timestamp_s", "x_px", "y_px", "source", "valid_observation", "interval_start_frame", "interval_duration_s", "displacement_px", "image_speed_px_s", "physical_speed_km_h", "quality_flags"]
    _csv(destination / outputs["player_motion_csv"], insights["player_motion"], player_columns)
    _csv(destination / outputs["shuttle_motion_csv"], insights["shuttle_motion"], shuttle_columns)
    (destination / outputs["coaching_review"]).write_text(coaching_review(insights), encoding="utf-8")
    for player in insights["players"]:
        _heatmap(destination / outputs[f"heatmap_track_{player['track_id']}"], player)
    return outputs