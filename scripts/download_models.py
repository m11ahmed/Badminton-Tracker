"""Download and verify the authors' public checkpoint inside this project."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "TrackNet_best.pt"
ARCHIVE = ROOT / "working" / "TrackNetV3_ckpts.zip"
FILE_ID = "1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"
EXPECTED_SHA = "df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a"
EXPECTED_ARCHIVE_SHA = "8de68d34ea2457e368e22947f1309143d60a6a7f9ddccd31cd526d4c4571bcc4"

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if MODEL.exists():
        if sha(MODEL) != EXPECTED_SHA:
            raise RuntimeError("Existing checkpoint differs from the verified public download. Keep it and supply a different --weights path, or rename it before downloading.")
        print(f"Verified existing model: {MODEL}")
        return
    if not ARCHIVE.exists():
        import gdown
        result = gdown.download(id=FILE_ID, output=str(ARCHIVE), quiet=False, use_cookies=False)
        if result is None:
            raise RuntimeError("Public model download failed. See docs/RUNNING.md for the official checkpoint URL.")
    if sha(ARCHIVE) != EXPECTED_ARCHIVE_SHA:
        raise RuntimeError("Archive checksum differs from the verified public download. The upstream file may have changed; it was not extracted.")
    if not zipfile.is_zipfile(ARCHIVE):
        raise RuntimeError("Downloaded file is not a ZIP archive.")
    temporary = MODEL.with_suffix(".pt.partial")
    with zipfile.ZipFile(ARCHIVE) as archive:
        candidates = [entry for entry in archive.namelist() if Path(entry).name == MODEL.name]
        if len(candidates) != 1:
            raise RuntimeError("Archive does not contain exactly one TrackNet_best.pt checkpoint.")
        with archive.open(candidates[0]) as source, temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    if sha(temporary) != EXPECTED_SHA:
        raise RuntimeError("Extracted checkpoint failed checksum verification.")
    temporary.replace(MODEL)
    manifest = {
        "source": "https://github.com/qaz812345/TrackNetV3",
        "upstream_commit": "6eda442ada1740573f200f836d93edc9a541ee86",
        "checkpoint_url": f"https://drive.google.com/file/d/{FILE_ID}/view",
        "archive_sha256": EXPECTED_ARCHIVE_SHA,
        "filename": MODEL.name, "sha256": EXPECTED_SHA, "bytes": MODEL.stat().st_size,
        "module_used": "TrackNet only; no InpaintNet rectification",
    }
    (MODEL.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Downloaded and verified: {MODEL}")

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Model download error: {error}")
