"""Small, project-contained helpers for the local rally review dashboard."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PALETTE = ((34, 211, 153), (251, 191, 36), (96, 165, 250), (244, 114, 182), (167, 139, 250))


def contained_path(root: Path, path: str | Path, area: str | None = None) -> Path:
    root = Path(root).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    allowed = (root / area).resolve() if area else root
    if candidate == allowed or allowed not in candidate.parents:
        raise ValueError(f"Choose a file inside {allowed}.")
    return candidate


def read_json(path: Path, *, optional=False):
    if optional and not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_run_summary(run: Path) -> dict:
    if (run/"summary.json").is_file():
        return read_json(run/"summary.json")
    validation = read_json(run/"validation.json")
    players = read_json(run/"players.json")
    frames = frames_of(players)
    if not frames or validation.get("frames_analyzed") != len(frames):
        raise ValueError("Player validation is incomplete.")
    metadata = players.get("metadata",{})
    model = metadata.get("model",{})
    return {"status":"complete","provenance":{**metadata,"milestone":model.get("milestone","M2"),"video":model.get("video",{}),"settings":{"players_mode":model.get("mode","doubles")}},"analysis_scope":validation.get("analysis_scope")}


def list_runs(root: Path = PROJECT_ROOT) -> list[Path]:
    result_root = Path(root) / "results"
    if not result_root.is_dir():
        return []
    runs = []
    for folder in result_root.iterdir():
        if not folder.is_dir() or folder.is_symlink() or not ((folder / "summary.json").is_file() or (folder/"validation.json").is_file()):
            continue
        try:
            summary = read_run_summary(folder)
            if summary.get("status") == "complete":
                milestone = str(summary.get("provenance", {}).get("milestone", ""))
                runs.append((milestone == "M5" or (folder / "insights.json").is_file(), folder.stat().st_mtime, folder))
        except (OSError, ValueError, TypeError):
            continue
    return [item[2] for item in sorted(runs, key=lambda item: (item[0], item[1]), reverse=True)]


def list_videos(root: Path = PROJECT_ROOT) -> list[Path]:
    data = Path(root) / "data"
    if not data.is_dir():
        return []
    videos = [p for p in data.rglob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"} and data.resolve() in p.resolve().parents]
    return sorted(videos, key=lambda p: ("rally" not in p.name.lower(), p.name.lower()))


def load_run(run: Path, root: Path = PROJECT_ROOT) -> dict:
    run = contained_path(root, run, "results")
    summary = read_run_summary(run)
    if summary.get("status") != "complete":
        raise ValueError("This run is not complete yet. Select another result.")
    payload = {"path": run, "summary": summary}
    for name in ("players", "shuttle", "court_positions", "insights", "court_calibration"):
        value = read_json(run / f"{name}.json", optional=True)
        payload[name] = value
    return payload


def frames_of(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    return payload.get("frames", []) if isinstance(payload, dict) else []


def track_ids(player_frames: list[dict]) -> list[int]:
    return sorted({int(player["track_id"]) for frame in player_frames for player in frame.get("players", []) if player.get("track_id") is not None})


def position_rows(player_frames: list[dict]) -> list[dict]:
    rows = []
    for frame in player_frames:
        for player in frame.get("players", []):
            position = player.get("court_position") or player
            x, y = position.get("x_m"), position.get("y_m")
            if x is None or y is None or not math.isfinite(float(x)) or not math.isfinite(float(y)):
                continue
            rows.append({"frame_index": int(frame["frame_index"]), "timestamp_s": float(frame["timestamp_s"]),
                         "track_id": int(player["track_id"]), "x_m": float(x), "y_m": float(y)})
    return rows


def inspect_video(path: Path) -> dict:
    import av
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("The file has no readable video stream.")
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 0)
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else (float(container.duration / 1_000_000) if container.duration else None)
        return {"width": stream.width, "height": stream.height, "fps": fps, "frame_count": stream.frames or None, "duration_s": duration}


def source_frame(path: Path, frame_index: int) -> bytes:
    import av
    if frame_index < 0:
        raise ValueError("Frame number must be nonnegative.")
    with av.open(str(path)) as container:
        container.streams.video[0].codec_context.thread_count=2
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                image = frame.to_image()
                result = io.BytesIO()
                image.save(result, format="JPEG", quality=88)
                return result.getvalue()
    raise ValueError("This frame is outside the source video.")


def calibration_preview(frame_jpeg: bytes, corners: list[list[float]]) -> bytes:
    image = Image.open(io.BytesIO(frame_jpeg)).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.line([tuple(p) for p in corners] + [tuple(corners[0])], fill=(34, 211, 153), width=3)
    for label, point in zip(("Far left", "Far right", "Near right", "Near left"), corners):
        x, y = point
        draw.ellipse((x-5, y-5, x+5, y+5), fill=(251, 191, 36))
        draw.text((x+8, y-8), label, fill="white", stroke_fill="black", stroke_width=2)
    result = io.BytesIO()
    image.save(result, format="JPEG", quality=90)
    return result.getvalue()


def court_image(rows: list[dict], *, frame_index: int | None = None, selected_id: int | None = None) -> bytes:
    width, height = 360, 720
    image = Image.new("RGB", (width, height), (10, 33, 31))
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 35, 32, 325, 672
    def point(x, y):
        return left + x / 6.1 * (right-left), top + y / 13.4 * (bottom-top)
    def line(a, b, color=(193, 219, 213), thickness=2):
        draw.line((point(*a), point(*b)), fill=color, width=thickness)
    line((0, 0), (6.1, 0)); line((6.1, 0), (6.1, 13.4)); line((6.1, 13.4), (0, 13.4)); line((0, 13.4), (0, 0))
    for x in (.46, 5.64): line((x, 0), (x, 13.4))
    for y in (.76, 4.72, 8.68, 12.64): line((0, y), (6.1, y))
    line((0, 6.7), (6.1, 6.7), (34, 211, 153), 3)
    line((3.05, 0), (3.05, 4.72)); line((3.05, 8.68), (3.05, 13.4))
    ids = sorted({r["track_id"] for r in rows})
    for track_id in ids:
        if selected_id is not None and track_id != selected_id:
            continue
        color = PALETTE[ids.index(track_id) % len(PALETTE)]
        subset = [r for r in rows if r["track_id"] == track_id and (frame_index is None or frame_index-45 <= r["frame_index"] <= frame_index)]
        previous = None
        for row in subset:
            if not (0 <= row["x_m"] <= 6.1 and 0 <= row["y_m"] <= 13.4):
                previous = None
                continue
            current = point(row["x_m"], row["y_m"])
            if previous and row["frame_index"] == previous[0]+1:
                draw.line((previous[1], current), fill=color, width=2)
            previous = (row["frame_index"], current)
            if frame_index is None:
                draw.ellipse((current[0]-2, current[1]-2, current[0]+2, current[1]+2), fill=color)
            elif row["frame_index"] == frame_index:
                draw.ellipse((current[0]-7, current[1]-7, current[0]+7, current[1]+7), fill=color)
                draw.text((current[0]+10, current[1]-10), f"ID {track_id}", fill=color)
    draw.text((35, 10), "FAR BASELINE", fill=(159, 187, 179))
    draw.text((35, 686), "NEAR BASELINE  /  METERS", fill=(159, 187, 179))
    result = io.BytesIO(); image.save(result, format="PNG")
    return result.getvalue()


def save_upload(root: Path, name: str, contents: bytes) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.]", "_", Path(name).name)
    if Path(safe_name).suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi"}:
        raise ValueError("Upload an MP4, MOV, MKV or AVI video.")
    destination = contained_path(root, Path("data") / "uploads" / f"{uuid.uuid4().hex[:10]}_{safe_name}", "data")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(contents)
    try:
        inspect_video(destination)
    except Exception:
        destination.unlink()
        raise ValueError("The upload could not be decoded as video.")
    return destination


def validate_review_for_run(review: dict, player_frames: list[dict]) -> dict:
    from .insights import validate_review
    return validate_review(review, player_frames)


def save_review(root: Path, run: Path, review: dict, player_frames: list[dict]) -> Path:
    run = contained_path(root, run, "results")
    review = dict(review)
    source_hash = read_run_summary(run).get("provenance",{}).get("input_video_sha256")
    if source_hash:
        previous_hash = review.get("source_video_sha256")
        if previous_hash and previous_hash != source_hash:
            raise ValueError("The review belongs to a different source video.")
        review["source_video_sha256"] = source_hash
    validated = validate_review_for_run(review, player_frames)
    # Some validators return None after checking an object.
    document = validated if isinstance(validated, dict) else review
    folder = contained_path(root, Path("results") / "reviews" / run.name / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]), "results")
    folder.mkdir(parents=True, exist_ok=False)
    output = folder / "review.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
    return output


def prepared_pose_nms_iou(video: Path, mode: str) -> float:
    """Use the prepared singles calibration only for its measured camera view."""
    return 0.5 if Path(video).name == "lee_axelsen_rally.mp4" and mode == "singles" else 0.7



def analysis_command(root: Path, video: Path, mode: str, calibration: Path | None, *, automatic=False, max_frames=0, roi: Path | None = None, pose_nms_iou: float = 0.7) -> tuple[list[str], Path]:
    root = Path(root).resolve()
    video = contained_path(root, video, "data")
    if not video.is_file():
        raise ValueError("Choose an existing video in this project's data folder.")
    if mode not in {"singles", "doubles"}:
        raise ValueError("Choose singles or doubles.")
    if not isinstance(pose_nms_iou,(float,int)) or isinstance(pose_nms_iou,bool) or not math.isfinite(pose_nms_iou) or not 0 < pose_nms_iou <= 1:
        raise ValueError("The overlap suppression threshold must be between 0 and 1.")
    if not automatic and calibration is None:
        raise ValueError("Set the four court corners or choose a matching calibration.")
    python = root / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        raise ValueError("The project environment is missing. Run scripts/setup.ps1 first.")
    output = root / "results" / ("m5_demo_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    command = [str(python), "-u", str(root / "run.py"), "--milestone", "m5", "--video", str(video), "--players-mode", mode, "--device", "cpu", "--threads", "2", "--pose-nms-iou", str(pose_nms_iou), "--out", str(output)]
    if automatic:
        command.append("--auto-court")
    else:
        command += ["--court-calibration", str(contained_path(root, calibration))]
    if roi is not None:
        command += ["--player-roi", str(contained_path(root, roi, "configs"))]
    if max_frames:
        if max_frames < 1:
            raise ValueError("The frame limit must be positive, or zero for the full clip.")
        command += ["--max-frames", str(int(max_frames))]
    return command, output


def project_environment(root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    working = Path(root) / "working"
    for variable, folder in {"TEMP":"tmp", "TMP":"tmp", "YOLO_CONFIG_DIR":"ultralytics", "TORCH_HOME":"torch", "MPLCONFIGDIR":"matplotlib", "XDG_CACHE_HOME":"cache", "PIP_CACHE_DIR":"pip-cache"}.items():
        path = working / folder
        path.mkdir(parents=True, exist_ok=True)
        environment[variable] = str(path)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def run_analysis(root: Path, command: list[str], on_line=None) -> int:
    # No shell interpretation: video paths and widget values stay separate arguments.
    with subprocess.Popen(command, cwd=str(root), env=project_environment(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
        assert process.stdout is not None
        for line in process.stdout:
            if on_line:
                on_line(line.rstrip())
        return process.wait()


def write_manual_calibration(root: Path, video: Path, corners: list[list[float]]) -> Path:
    from .court import create_calibration
    video = contained_path(root, video, "data")
    metadata = inspect_video(video)
    digest = hashlib.sha256()
    with video.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    calibration = create_calibration(corners, metadata["width"], metadata["height"], metadata={"input_video":str(video), "input_video_sha256":digest.hexdigest(), "selection_interface":"Streamlit numeric corner entry", "fixed_camera_assumption":True, "confirmation_status":"manually supplied corners; inspect court lines"})
    output = contained_path(root, Path("configs") / ("demo_court_"+uuid.uuid4().hex[:10]+".json"), "configs")
    with output.open("x", encoding="utf-8") as handle:
        json.dump(calibration.to_dict(), handle, indent=2, allow_nan=False)
    return output

def extract_clip(root: Path, video: Path, start_s: float, end_s: float, on_line=None) -> Path:
    video = contained_path(root, video, "data")
    if not (math.isfinite(start_s) and math.isfinite(end_s) and 0 <= start_s < end_s and end_s-start_s <= 300):
        raise ValueError("Choose 0 <= start < end, with at most 300 seconds per clip.")
    output = contained_path(root, Path("data") / (video.stem + "_rally_" + uuid.uuid4().hex[:8] + ".mp4"), "data")
    command = [str(Path(root)/".venv"/"Scripts"/"python.exe"), "-u", str(Path(root)/"scripts"/"extract_rally.py"), "--video", str(video), "--start", str(start_s), "--end", str(end_s), "--out", str(output)]
    result = run_analysis(root, command, on_line)
    if result or not output.is_file():
        raise ValueError("Rally extraction failed. Check the log and selected start/end times.")
    inspect_video(output)
    return output


def recompute_reviewed_insights(root: Path, run: Path, review: dict, player_frames: list[dict]) -> tuple[Path, dict]:
    from .court import load_calibration
    from .insights import analyze_insights, export_insights
    run = contained_path(root, run, "results")
    summary = read_run_summary(run)
    provenance = summary.get("provenance", {})
    video = provenance.get("video", {})
    calibration = load_calibration(run/"court_calibration.json", video['width'], video['height']) if (run/"court_calibration.json").is_file() else None
    review_path = save_review(root, run, review, player_frames)
    bound_review = read_json(review_path)
    insights = analyze_insights(frames_of(read_json(run/"shuttle.json",optional=True)), player_frames, calibration, mode=provenance.get("settings",{}).get("players_mode","singles"), review=bound_review)
    insights["provenance"] = {**provenance,"source_run":str(run),"review_file":str(review_path),"review_action":"recomputed from existing observations; no neural inference rerun"}
    export_insights(review_path.parent, insights)
    return review_path, insights


def write_roi(root: Path, points: list[list[float]]) -> Path:
    from .court import create_calibration
    # Reuse the geometric check for a convex, ordered four-corner polygon.
    create_calibration([[x*1000,y*1000] for x,y in points],1001,1001,method="manual_roi")
    if any(not 0<=coordinate<=1 for point in points for coordinate in point):
        raise ValueError("Player region coordinates must lie between 0 and 1.")
    output=contained_path(root,Path("configs")/("demo_player_roi_"+uuid.uuid4().hex[:10]+".json"),"configs")
    with output.open("x",encoding="utf-8") as handle:
        json.dump({"points":points,"description":"Player selection region explicitly entered in the local dashboard; inspect against source footage."},handle,indent=2,allow_nan=False)
    return output
