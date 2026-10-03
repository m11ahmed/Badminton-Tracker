"""Download and verify the pinned raw singles rally used for longer clip checks."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "f53e0b8f7bbeda10ba7848cba7a19fa91406861c"
SOURCE = f"https://raw.githubusercontent.com/dybtom/Good-Badminton/{COMMIT}/videos/demo.mp4"
SHA256 = "b60cc98c5c5066e31eed777c6cf92bc2d4899d210a5f7285d7358ee7c63d7b26"


def verify(payload):
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError("Rally checksum mismatch; no existing video will be overwritten.")


def main():
    destination = ROOT / "data" / "candidate_match.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify(destination.read_bytes())
        print("Existing rally checksum verified.")
    else:
        print("Downloading the 2.5 MB rally from the pinned public source...")
        request = urllib.request.Request(SOURCE, headers={"User-Agent": "badminton-tracker-rally-download"})
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read(32 * 1024 * 1024 + 1)
        verify(payload)
        with destination.open("xb") as output:
            output.write(payload)
        print("Download checksum verified.")
    import av
    with av.open(str(destination)) as container:
        stream = container.streams.video[0]
        width, height = stream.codec_context.width, stream.codec_context.height
        fps = float(stream.average_rate)
        frames = sum(1 for _ in container.decode(video=0))
    if (width, height, frames) != (1280, 720, 426) or abs(fps - 30) > 1e-6:
        raise ValueError("Unexpected rally metadata; review this source before analysis.")
    provenance = {
        "repository": "https://github.com/dybtom/Good-Badminton",
        "commit": COMMIT, "source_video": "videos/demo.mp4", "download_url": SOURCE,
        "video_sha256": SHA256, "width": width, "height": height,
        "frames": frames, "fps": fps, "duration_s": frames / fps,
        "description": "Raw broadcast singles rally, scoreboard Axelsen and Ginting.",
        "review": "Sampled source frames show both players and the full court. New calibration and tracking verification are still required.",
        "evaluation_scope": "Convenience input; independence from pretrained model training is not established.",
    }
    manifest = destination.with_name("candidate_match_provenance.json")
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != provenance:
            raise ValueError("Existing provenance differs; it will not be overwritten.")
    else:
        with manifest.open("x", encoding="utf-8") as output:
            output.write(json.dumps(provenance, indent=2) + "\n")
    print(f"Ready: {destination}")
    print(f"Verified all {frames} frames: {width}x{height}, {fps:g} fps, {frames/fps:g} seconds.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Rally download error: {error}")