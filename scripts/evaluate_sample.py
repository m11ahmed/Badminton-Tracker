"""Evaluate raw observations against the bundled clip's normalized annotations."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "m1-sample")
    parser.add_argument("--tolerance-px", type=float, default=8.0, help="Maximum matching distance in original image pixels.")
    args = parser.parse_args()
    if not math.isfinite(args.tolerance_px) or args.tolerance_px <= 0:
        parser.error("tolerance must be a positive finite number")
    results = args.results.resolve()
    if not results.is_relative_to(ROOT):
        parser.error("results must be inside this project on G")
    payload = json.loads((results / "shuttle.json").read_text(encoding="utf-8-sig"))
    width = payload["metadata"]["video"]["width"]
    height = payload["metadata"]["video"]["height"]
    with (ROOT / "data" / "sample_ground_truth.csv").open(encoding="utf-8-sig", newline="") as source:
        truth = {int(row["Frame"]): row for row in csv.DictReader(source)}
    frames = payload["frames"]
    if len(frames) != len(truth) or set(row["frame_index"] for row in frames) != set(truth):
        raise RuntimeError("This evaluator requires the complete bundled sample and every matching label frame.")
    provenance = json.loads((ROOT / "data" / "sample_provenance.json").read_text(encoding="utf-8-sig"))
    import hashlib
    input_path = Path(payload["metadata"]["input_video"])
    actual_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if actual_sha != provenance["video_sha256"]:
        raise RuntimeError("Input video is not the verified bundled sample; its labels cannot be applied.")
    matches = false_positives = false_negatives = true_negatives = 0
    errors = []
    missed_frames = []
    for frame in frames:
        label = truth[frame["frame_index"]]
        visible = int(label["Ball"]) == 1
        # Interpolation never counts as a neural detection.
        detected = frame["raw_x"] is not None and frame["raw_y"] is not None
        if visible and detected:
            distance = math.hypot(frame["raw_x"] - float(label["x"]) * width, frame["raw_y"] - float(label["y"]) * height)
            errors.append(distance)
            if distance <= args.tolerance_px:
                matches += 1
            else:
                false_positives += 1
                false_negatives += 1
                missed_frames.append(frame["frame_index"])
        elif visible:
            false_negatives += 1
            missed_frames.append(frame["frame_index"])
        elif detected:
            false_positives += 1
        else:
            true_negatives += 1
    sorted_errors = sorted(errors)
    median_error = (sorted_errors[(len(sorted_errors)-1)//2] + sorted_errors[len(sorted_errors)//2])/2 if errors else None
    report = {
        "scope": "sample check, not an independent held-out benchmark",
        "training_overlap": "unknown",
        "frames": len(frames), "visible_ground_truth_frames": sum(int(row["Ball"]) == 1 for row in truth.values()),
        "tolerance_original_pixels": args.tolerance_px,
        "matched_detections": matches, "false_positives": false_positives, "false_negatives": false_negatives, "true_negatives": true_negatives,
        "precision": matches/(matches+false_positives) if matches+false_positives else None,
        "recall": matches/(matches+false_negatives) if matches+false_negatives else None,
        "median_localization_error_on_visible_detected_frames_px": median_error,
        "mean_localization_error_on_visible_detected_frames_px": sum(errors)/len(errors) if errors else None,
        "missed_or_mislocalized_frame_indices": missed_frames,
        "note": "Wrong visible locations count as both a false positive and a false negative. Error statistics omit frames without a detection. Annotation coordinates are rounded normalized values.",
    }
    path = results / "sample_evaluation.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
