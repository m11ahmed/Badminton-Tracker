"""Create a project-local court calibration with manual clicks or a proposal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from badminton_tracker.cli import _configure_project_runtime
_configure_project_runtime()

import cv2
import numpy as np

from badminton_tracker.court import CORNER_ORDER, COURT_LENGTH_M, COURT_WIDTH_M, SINGLE_WIDTH_M, create_calibration
from badminton_tracker.court_detection import detect_court
from badminton_tracker.video import iter_rgb_frames, probe_video


def _parse_corners(text):
    try:
        numbers = [float(number.strip()) for number in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Corners must be eight comma separated pixel numbers.") from error
    if len(numbers) != 8:
        raise argparse.ArgumentTypeError("Corners must contain exactly eight numbers: far-left, far-right, near-right, near-left.")
    return np.asarray(numbers, dtype=np.float64).reshape(4, 2)


def _output_path(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Calibration output must remain inside the G drive project: {ROOT}")
    preview = ROOT / "visualizations" / (resolved.stem + ".preview.jpg")
    if resolved.exists() or preview.exists():
        raise FileExistsError("Calibration or preview already exists. Choose a new output path; files are not overwritten.")
    return resolved, preview


def _preview(rgb, calibration):
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    corners = calibration.corners_px
    cv2.polylines(image, [np.rint(corners).astype(np.int32)], True, (40, 230, 255), 2, cv2.LINE_AA)
    margin = (COURT_WIDTH_M-SINGLE_WIDTH_M)/2
    # Service boundaries and the NET FLOOR POSITION provide independent visual
    # checks. The net tape in a camera image is above this projected floor line.
    world_lines = [([margin, 0], [margin, COURT_LENGTH_M]),
                   ([COURT_WIDTH_M-margin, 0], [COURT_WIDTH_M-margin, COURT_LENGTH_M]),
                   ([0, 0.76], [COURT_WIDTH_M, 0.76]),
                   ([0, COURT_LENGTH_M-0.76], [COURT_WIDTH_M, COURT_LENGTH_M-0.76]),
                   ([0, 4.72], [COURT_WIDTH_M, 4.72]),
                   ([0, 8.68], [COURT_WIDTH_M, 8.68]),
                   ([COURT_WIDTH_M/2, 0], [COURT_WIDTH_M/2, 4.72]),
                   ([COURT_WIDTH_M/2, 8.68], [COURT_WIDTH_M/2, COURT_LENGTH_M])]
    for first, second in world_lines:
        line = np.rint(calibration.court_to_image([first, second])).astype(int)
        cv2.line(image, tuple(line[0]), tuple(line[1]), (255, 210, 40), 1, cv2.LINE_AA)
    net = np.rint(calibration.court_to_image([[0, 6.7], [COURT_WIDTH_M, 6.7]])).astype(int)
    cv2.line(image, tuple(net[0]), tuple(net[1]), (220, 80, 220), 1, cv2.LINE_AA)
    for index, (x, y) in enumerate(corners):
        cv2.circle(image, (round(x), round(y)), 5, (0, 60, 255), -1, cv2.LINE_AA)
        cv2.putText(image, f"{index+1}: {CORNER_ORDER[index]}", (round(x)+7, round(y)-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (245, 245, 245), 1, cv2.LINE_AA)
    title = "UNCONFIRMED AUTOMATIC PROPOSAL" if calibration.method == "auto_proposal" else "MANUAL COURT CORNERS"
    cv2.rectangle(image, (0, 0), (image.shape[1], 49), (20, 20, 20), -1)
    cv2.putText(image, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(image, "Yellow: outer court | cyan: service lines | purple: net FLOOR position", (10, 41),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (240, 240, 240), 1, cv2.LINE_AA)
    return image


def _manual_clicks(rgb):
    height, width = rgb.shape[:2]
    scale = min(1.0, 1280/width, 720/height)
    window = "Court corners: far-left, far-right, near-right, near-left"
    display = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (round(width*scale), round(height*scale)))
    corners = []
    status = "Click OUTER doubles corners in the listed order."

    def on_mouse(event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append([float(x/scale), float(y/scale)])

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            canvas = display.copy()
            if corners:
                cv2.polylines(canvas, [np.rint(np.asarray(corners)*scale).astype(np.int32)], len(corners)==4,
                              (0, 230, 255), 2, cv2.LINE_AA)
            for index, point in enumerate(corners):
                x, y = np.rint(np.asarray(point)*scale).astype(int)
                cv2.circle(canvas, (int(x), int(y)), 5, (20, 60, 255), -1)
                cv2.putText(canvas, f"{index+1}: {CORNER_ORDER[index]}", (int(x)+8, int(y)-7),
                            cv2.FONT_HERSHEY_SIMPLEX, .43, (255,255,255), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 52), (20,20,20), -1)
            cv2.putText(canvas, "R reset | Backspace undo | S/Enter save | Q/Esc quit", (8,20),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (240,240,240),1,cv2.LINE_AA)
            cv2.putText(canvas, status, (8,42), cv2.FONT_HERSHEY_SIMPLEX, .43,(240,240,240),1,cv2.LINE_AA)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30)&0xff
            if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key == ord("r"):
                corners.clear()
                status = "Click OUTER doubles corners in the listed order."
            elif key in (8, 127) and corners:
                corners.pop()
            elif key in (ord("s"), 10, 13):
                if len(corners) != 4:
                    status = "Four corners required before saving."
                    continue
                try:
                    create_calibration(corners, width, height)
                    return corners
                except ValueError as error:
                    status = str(error)
                    print(f"Cannot save: {error}. Reset or undo the incorrect point.")
    finally:
        cv2.destroyWindow(window)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0, help="Zero based source reference frame")
    parser.add_argument("--out", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--corners", type=_parse_corners, help="Eight pixel numbers; outer doubles corners in far-left,far-right,near-right,near-left order")
    selection.add_argument("--auto-propose", action="store_true", help="Save an unconfirmed green court/white line proposal and visual preview")
    args = parser.parse_args()
    try:
        if args.frame < 0:
            raise ValueError("Reference frame must be nonnegative.")
        output, preview_path = _output_path(args.out)
        video = probe_video(args.video)
        selected = next(((i,t,rgb) for i,t,rgb in iter_rgb_frames(args.video, max_frames=args.frame+1) if i==args.frame), None)
        if selected is None:
            raise ValueError(f"Source video has no frame {args.frame}.")
        index, timestamp, rgb = selected
        digest = hashlib.sha256()
        with args.video.open("rb") as source:
            for chunk in iter(lambda: source.read(1024*1024), b""):
                digest.update(chunk)
        metadata = {"input_video": str(args.video.resolve()), "input_video_sha256": digest.hexdigest(),
                    "reference_timestamp_s": timestamp, "video_fps": video["fps"],
                    "timing_policy": video["timing_policy"], "fixed_camera_assumption": True}
        method = "manual"
        if args.auto_propose:
            proposal = detect_court(rgb)
            corners = proposal["corners_px"]
            method = "auto_proposal"
            metadata["automatic_candidate"] = proposal
            metadata["confirmation_status"] = "unconfirmed"
        elif args.corners is not None:
            corners = args.corners
            metadata["selection_interface"] = "explicit pixel coordinates"
            metadata["confirmation_status"] = "manually supplied corners; inspect the independent line preview"
        else:
            corners = _manual_clicks(rgb)
            if corners is None:
                print("Calibration cancelled; no calibration or preview was written.")
                return 0
            metadata["selection_interface"] = "four source frame mouse clicks"
            metadata["confirmation_status"] = "manual corner selection; inspect the independent line preview"
        calibration = create_calibration(corners, video["width"], video["height"],
                                         method=method, reference_frame=index, metadata=metadata)
        preview = _preview(rgb, calibration)
        success, encoded = cv2.imencode(".jpg", preview)
        if not success:
            raise RuntimeError("Could not encode calibration preview.")
        output.parent.mkdir(parents=True, exist_ok=True)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        with preview_path.open("xb") as preview_file:
            preview_file.write(encoded.tobytes())
        with output.open("x", encoding="utf-8") as record:
            json.dump(calibration.to_dict(), record, indent=2, allow_nan=False)
        print(f"Calibration: {output}\nPreview: {preview_path}\nMethod: {method}")
        if method == "auto_proposal":
            print("Automatic geometry remains an unconfirmed estimate; inspect its outer boundary and service lines.")
        return 0
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(2, f"Calibration failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
