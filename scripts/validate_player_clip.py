"""Run a real player-only validation clip; no shuttle inference is implied."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from badminton_tracker.cli import _configure_project_runtime, _contained_path, _load_player_roi

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mode", choices=["singles", "doubles"], required=True)
    parser.add_argument("--roi", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--image-size", type=int, default=1280)
    args = parser.parse_args()
    output = _contained_path(args.out, "Validation output")
    if output.exists():
        parser.error("Choose a new output directory; validation records are never overwritten.")
    _configure_project_runtime()
    import torch
    from badminton_tracker.players import track_players
    torch.set_num_threads(2)
    frames, metadata = track_players(args.video, ROOT / "models" / "yolo11n-pose.pt", mode=args.mode,
        player_roi=_load_player_roi(args.roi), device="cpu", image_size=args.image_size, max_frames=args.max_frames)
    output.mkdir(parents=True)
    context = {
        "analysis_scope": "real player-only inference; shuttle was not analyzed",
        "input_video": str(args.video.resolve()),
        "input_video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
        "model": metadata,
        "timing_caveat": ("GIF playback timestamps do not establish original camera capture timing." if args.video.suffix.lower() == ".gif" else "Source presentation timestamps describe the published clip; original capture timing is not independently verified."),
    }
    (output/"players.json").write_text(json.dumps({"metadata":context,"frames":frames},indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(metadata["summary"],indent=2))

if __name__ == "__main__":
    main()
