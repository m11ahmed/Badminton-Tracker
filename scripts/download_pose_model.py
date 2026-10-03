"""Download the pinned official YOLO11 nano pose weights into this project."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "yolo11n-pose.pt"
URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
EXPECTED_SHA = "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0"

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    if MODEL.exists():
        if sha(MODEL) != EXPECTED_SHA:
            raise RuntimeError("Existing pose checkpoint differs from the verified official model. It was not overwritten.")
    else:
        partial = MODEL.with_suffix(".pt.partial")
        with urllib.request.urlopen(URL, timeout=120) as source, partial.open("wb") as destination:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)
        if sha(partial) != EXPECTED_SHA:
            raise RuntimeError("Pose checkpoint checksum mismatch. No verified model was published.")
        partial.replace(MODEL)
    manifest = {
        "filename": MODEL.name, "source": URL, "release": "v8.3.0",
        "sha256": EXPECTED_SHA, "bytes": MODEL.stat().st_size,
        "ultralytics_version": "8.3.228", "keypoint_layout": "COCO17",
        "license_reference": "https://github.com/ultralytics/ultralytics/blob/v8.3.228/LICENSE",
    }
    (MODEL.parent / "pose_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Verified official pose model: {MODEL}")

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Pose model download error: {error}")
