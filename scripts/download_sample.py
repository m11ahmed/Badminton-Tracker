"""Fetch the pinned small public example; this is not an independent benchmark."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "d1d96f39d275970d3597a45926f4f46e6ee85e2c"
BASE = f"https://raw.githubusercontent.com/alenzenx/TrackNetV3/{COMMIT}/raw_data2"
FILES = {
    "sample.mp4": ("00013.mp4", "b45178f29dcff7cd12f5a61f364b55a058ada532dd6110cd59d85ed9d64a4f24"),
    "sample_ground_truth.csv": ("00013.csv", "48a3ef35fcd7c6b6bf7d148ab8eb391af63c15fd7ccc637e9bd3241cc1745d7d"),
}

def main():
    destination = ROOT / "data"
    destination.mkdir(parents=True, exist_ok=True)
    for local_name, (remote_name, expected_sha) in FILES.items():
        path = destination / local_name
        if path.exists():
            payload = path.read_bytes()
        else:
            with urllib.request.urlopen(f"{BASE}/{remote_name}", timeout=90) as response:
                payload = response.read()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected_sha:
            raise RuntimeError(f"Sample checksum mismatch: {local_name}; existing data will not be overwritten.")
        if not path.exists():
            path.write_bytes(payload)
        print(f"Verified: {path}")
    provenance = {
        "repository": "https://github.com/alenzenx/TrackNetV3", "commit": COMMIT,
        "source_video": "raw_data2/00013.mp4", "source_labels": "raw_data2/00013.csv",
        "video_sha256": FILES["sample.mp4"][1], "labels_sha256": FILES["sample_ground_truth.csv"][1],
        "label_coordinates": "normalized fractions of original video dimensions; invisible positions -1",
        "evaluation_scope": "sample check only; independence from selected checkpoint training has not been established",
    }
    (destination / "sample_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Sample download error: {error}")
