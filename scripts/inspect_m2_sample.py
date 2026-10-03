"""Inspect a real M2 export without rerunning model inference."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from badminton_tracker.video import iter_rgb_frames, probe_video


def decode_summary(path, selected):
    times, images = [], {}
    for index, timestamp, rgb in iter_rgb_frames(path):
        times.append(timestamp)
        if index in selected:
            images[index] = Image.fromarray(rgb)
    return times, images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "m2-verified-sample")
    args = parser.parse_args()
    results = args.results.resolve()
    if not results.is_relative_to(ROOT):
        parser.error("Keep inspection output inside the project.")
    players_payload = json.loads((results / "players.json").read_text(encoding="utf-8"))
    shuttle_payload = json.loads((results / "shuttle.json").read_text(encoding="utf-8"))
    frames = players_payload["frames"]
    selected = sorted(set([0, len(frames) // 5, len(frames) * 2 // 5, len(frames) * 3 // 5,
                           len(frames) * 4 // 5, len(frames) - 1]))
    source_path = Path(players_payload["metadata"]["input_video"])
    source_times, _ = decode_summary(source_path, set())
    output_times, images = decode_summary(results / "annotated.mp4", set(selected))
    shuttle_times = [row["timestamp_s"] for row in shuttle_payload["frames"]]
    player_times = [row["timestamp_s"] for row in frames]
    if len({len(source_times), len(output_times), len(shuttle_times), len(player_times)}) != 1:
        raise RuntimeError("Source, player records, shuttle records and output frame counts differ.")
    errors = {
        "encoded": max(abs(a - b) for a, b in zip(source_times, output_times)),
        "players": max(abs(a - b) for a, b in zip(source_times, player_times)),
        "shuttle": max(abs(a - b) for a, b in zip(source_times, shuttle_times)),
    }
    if max(errors.values()) > 1 / 90000 + 1e-8:
        raise RuntimeError("Frame timestamps are inconsistent.")
    track_frames = defaultdict(list)
    visible_counts = defaultdict(list)
    missing_ankles = Counter()
    count_histogram = Counter()
    null_keypoints = 0
    ordering = []
    for index, frame in enumerate(frames):
        if frame["frame_index"] != index:
            raise RuntimeError("Player frame indices are discontinuous.")
        count_histogram[len(frame["players"])] += 1
        ids = [player["track_id"] for player in frame["players"]]
        if len(ids) != len(set(ids)):
            raise RuntimeError("Duplicate tracker ID in a frame.")
        for player in frame["players"]:
            track_id = player["track_id"]
            track_frames[track_id].append(index)
            points = player["keypoints"]
            if len(points) != 17:
                raise RuntimeError("An emitted pose does not have 17 COCO keypoints.")
            visible_counts[track_id].append(sum(point["visible"] for point in points))
            for point in points:
                if not point["visible"]:
                    null_keypoints += 1
                    if point["x"] is not None or point["y"] is not None:
                        raise RuntimeError("An invisible point has coordinates.")
            if player["ankle_midpoint"] is None:
                missing_ankles[track_id] += 1
        if len(frame["players"]) == 2 and all(player["ankle_midpoint"] for player in frame["players"]):
            ordering.append(frame["players"][0]["ankle_midpoint"][1] > frame["players"][1]["ankle_midpoint"][1])
    with (results / "players.csv").open(encoding="utf-8", newline="") as file:
        csv_count = sum(1 for _ in csv.DictReader(file))
    if csv_count != sum(len(frame["players"]) for frame in frames):
        raise RuntimeError("Player CSV does not match JSON observations.")
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 19)
    except OSError:
        font = ImageFont.load_default()
    visualization_dir = ROOT / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (1280, 1320), "#111923")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 12), "M2 actual decoded output: player IDs, pose and shuttle together", font=font, fill="white")
    draw.text((16, 40), "Review across the clip; coverage does not measure keypoint or identity accuracy", font=font, fill="#cccccc")
    for position, index in enumerate(selected):
        x0, y0 = (position % 2) * 640, 80 + (position // 2) * 410
        sheet.paste(images[index].resize((640, 360)), (x0, y0))
        ids = [player["track_id"] for player in frames[index]["players"]]
        draw.text((x0 + 12, y0 + 365), f"Frame {index} | {source_times[index]:.3f}s | IDs {ids}", font=font, fill="white")
    contact_sheet = visualization_dir / (results.name + "-contact-sheet.jpg")
    sheet.save(contact_sheet, quality=94)
    # Crop each actual encoded player, keeping its pixels and surrounding context.
    crop_sheet = Image.new("RGB", (1280, len(selected) * 270 + 70), "#111923")
    crop_draw = ImageDraw.Draw(crop_sheet)
    crop_draw.text((16, 16), "M2 encoded pose details, enlarged for inspection", font=font, fill="white")
    for row_index, index in enumerate(selected):
        for column, player in enumerate(frames[index]["players"][:2]):
            x1, y1, x2, y2 = player["bbox"]
            crop = images[index].crop((max(0, int(x1) - 25), max(0, int(y1) - 35),
                                       min(images[index].width, int(x2) + 25), min(images[index].height, int(y2) + 25)))
            crop.thumbnail((620, 220))
            x0, y0 = column * 640, 70 + row_index * 270
            crop_sheet.paste(crop, (x0 + 10, y0 + 35))
            crop_draw.text((x0 + 12, y0 + 4), f"Frame {index} | ID {player['track_id']}", font=font, fill="white")
    crop_path = visualization_dir / (results.name + "-pose-details.jpg")
    crop_sheet.save(crop_path, quality=94)
    report = {
        "source_frames": len(source_times), "output_frames": len(output_times),
        "maximum_source_timestamp_difference_s": errors,
        "output_probe": probe_video(results / "annotated.mp4"),
        "player_observations_json": sum(len(frame["players"]) for frame in frames),
        "player_observations_csv": csv_count,
        "observed_count_histogram": dict(count_histogram),
        "tracks": {str(track_id): {"frames_observed": len(indices), "first": indices[0], "last": indices[-1],
                    "visible_keypoint_count_min": min(visible_counts[track_id]),
                    "visible_keypoint_count_max": max(visible_counts[track_id]),
                    "visible_keypoint_count_mean": sum(visible_counts[track_id]) / len(indices),
                    "missing_ankle_midpoints": missing_ankles[track_id]}
                   for track_id, indices in track_frames.items()},
        "null_keypoint_count": null_keypoints,
        "two_player_first_id_nearer_image_bottom_all_frames": all(ordering) if ordering else None,
        "identity_accuracy": "No athlete identity annotations; image ordering is an invariant, not an ID-switch metric.",
        "keypoint_accuracy": "No labeled pose annotations; visible counts are coverage only.",
        "selected_frames": selected,
        "contact_sheet": str(contact_sheet), "pose_details": str(crop_path),
        "inspection_scope": "Actual decoded output and timestamps. Human visual review is separate. Browser playback was not tested.",
    }
    (results / "video_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
