# Run Badminton Tracker on Windows

Open PowerShell in the project root. The commands below use the project local Python environment and keep new code, data, model weights, results and supported caches in this folder.

## Setup

Open PowerShell and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\.venv\Scripts\python.exe .\scripts\download_models.py
.\.venv\Scripts\python.exe .\scripts\download_pose_model.py
.\.venv\Scripts\python.exe .\scripts\download_sample.py
```

Setup installs matching CPU builds of PyTorch 2.8.0 and torchvision 0.23.0, then the project requirements. M2 uses Ultralytics 8.3.228, ByteTrack and the official YOLO11 nano pose checkpoint. The pose downloader verifies the model checksum and writes its source manifest under `models`.

The requirements deliberately use `polars[rtcompat]`. Its compatibility runtime supports this older CPU; replacing it with the usual Polars runtime can fail on processors without the necessary AVX instructions. Keep the matching PyTorch and torchvision builds so that YOLO's image operations are available.

The CPU configuration is tested on this computer. For a supported NVIDIA GPU, run `setup.ps1 -Device cuda` to install the matching PyTorch 2.8.0 and torchvision 0.23.0 CUDA 12.8 builds, then check `torch.cuda.is_available()`. The GPU configuration has not been tested on this machine. Official PyTorch wheels do not require a separate local CUDA toolkit; a compatible NVIDIA driver is still required. See the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v280).

If Python is not found, pass `-PythonExecutable 'G:\path\to\python.exe'` or another existing Python 3.10 to 3.13 interpreter to the setup script. The project environment itself is created on G. A bundled or existing interpreter may still have its original installation on C.

## Review the completed M2 sample

Open `results\m2-verified-sample\annotated.mp4`. This corrected singles run combines the shuttle trail with player boxes, raw tracker IDs, reliable pose points and ankle midpoint markers.

It contains 81 frames at 1280 by 720 and 30 fps. Two players were observed in all 81 frames. Raw ByteTrack IDs 1 and 2 each have 81 observations with no recorded gaps. These counts describe this short sample; they do not establish an identity switch rate or keypoint accuracy on other footage.

The shuttle has 66 detected frames, 4 flagged interpolations and 11 missing frames. A missing shuttle frame remains missing unless a short bounded gap can be filled from reliable observations.

Generate a fresh M2 sample run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_m2_sample.ps1
```

To avoid repeating shuttle inference when reviewing pose settings on this same sample, reuse the verified raw shuttle cache:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_m2_sample.ps1 -ShuttleResults '.\results\m2-verified-sample'
```

Both commands create a fresh timestamped result folder. Previous outputs are never overwritten. The script explicitly uses `configs\sample_player_roi.json`; that polygon applies only to the included camera view.

The corrected sample used a pose image size of 1280 and two CPU threads. With verified shuttle reuse, its complete M2 processing took approximately 121 seconds, including approximately 118 seconds of player inference. A fresh run also pays the shuttle inference cost. This CPU therefore processes a 2.7 second clip much more slowly than real time. Start with short clips.

## Analyze your own recording

Use a short fixed camera clip with the whole court and active players visible. Save the recording and its camera-specific ROI under this project's `data` and `configs` folders.

A singles example is:

```powershell
.\.venv\Scripts\python.exe .\run.py --milestone m2 --video .\data\my-rally.mp4 --players-mode singles --player-roi .\configs\my-camera-roi.json --device cpu --threads 2 --out .\results\my-first-m2-rally
```

Use `--players-mode doubles` for four expected players. The software does not cap detections at two or four; counts that differ from the expected number are exported and shown as warnings. The combined verification above is singles. A separate real rear-view doubles check tracked all four players while visible in its first 56 frames, then exposed a camera-edge gap and a new ID on return. Overhead footage still has a missed visible athlete. These small checks do not establish general doubles identity reliability; see `docs\M2_VERIFICATION.md`.

The ROI file can contain either a direct list of points or an object with `points`:

```json
{
  "points": [[0.25, 0.20], [0.75, 0.20], [0.95, 0.98], [0.05, 0.98]]
}
```

Those numbers illustrate the format; replace them with a polygon suited to your camera. Divide each horizontal pixel coordinate by image width and each vertical coordinate by image height. Order the vertices around the polygon without crossing edges. Include the active players' floor positions and immediate court apron while excluding officials and spectators where possible. A camera change requires a new polygon.

The ROI gates detections using their ankle midpoint or box bottom center before tracking. It reduces background people entering the analysis; it cannot prove that every person inside the polygon is an athlete. No ROI is applied by default. Without `--player-roi`, all detected people may be tracked and `player_selection_unverified` is flagged. Do not reuse the public sample polygon as a universal court boundary.

For a quicker initial check:

```powershell
.\.venv\Scripts\python.exe .\run.py --milestone m2 --video .\data\my-rally.mp4 --players-mode singles --player-roi .\configs\my-camera-roi.json --max-frames 16 --pose-image-size 1280 --device cpu --threads 2 --out .\results\my-m2-smoke-test
```

A smaller pose input is faster but can lose detail on the far player. The M2 configuration currently uses 1280. A partial smoke run cannot serve as a complete-source shuttle cache.

## Timing and reusable shuttle results

`--fps-hint 60` supplies timing only when source metadata is missing. Source presentation timestamps and usable frame rate take precedence. A hint does not turn 30 fps footage into a 60 fps recording. Slow motion playback does not automatically reveal the original capture rate.

New M1 and M2 outputs record the input video's SHA256. `--shuttle-results` accepts an existing `shuttle.json` or its containing result directory. Before reuse, it checks the input digest, video metadata, complete decoded frame count, every source timestamp and shuttle inference settings. It reuses raw detector records and recomputes gap interpolation.

The original `results\m1-sample` cache predates input SHA recording and is intentionally rejected for reuse. Its completed video and data remain available. Generate one new complete run before using its shuttle observations across later pose trials. A verified complete cache can support a shorter `--max-frames` prefix, but a partial cache cannot replace a complete-source cache.

For your own video, reuse its previously generated complete M2 result:

```powershell
.\.venv\Scripts\python.exe .\run.py --milestone m2 --video .\data\my-rally.mp4 --players-mode singles --player-roi .\configs\my-camera-roi.json --shuttle-results .\results\my-first-m2-rally --pose-image-size 1280 --out .\results\my-m2-pose-review
```

Shuttle confidence threshold, heatmap threshold and background sample count must match the cached inference settings. Pose settings and the current ROI can change. Mismatches are rejected before completed results are published.

## Outputs and interpretation

1. `annotated.mp4` is a silent H.264 video retaining source frame timing. M2 adds player boxes, ID colors, visible pose points and count warnings to the shuttle overlay.
2. `shuttle.csv` and `shuttle.json` contain one record per analyzed frame, source image coordinates, raw detector values, scores and interpolation flags.
3. `players.json` contains every analyzed frame, including empty frames, expected and observed counts, flags and each player's pose.
4. `players.csv` contains one row per observed player with its timestamp, raw tracker ID, box, ankle midpoint and all 17 keypoint coordinates, scores and visibility fields. Empty frames are retained in JSON and summary counts rather than represented as fictional CSV players.
5. `summary.json` records settings, model and input provenance, count mismatches, tracker fragments, missing coordinates and processing time.

Stable colors follow the raw ByteTrack ID; they do not identify a named athlete. Tracking can fragment or switch IDs during overlap, occlusion or camera cuts. Missing player poses are not interpolated in M2. Low confidence keypoint coordinates are null in JSON and empty in CSV, and are omitted from drawing. An ankle midpoint is available only when both ankles pass the confidence and validity checks.

Shuttle scores and pose confidence values are model outputs, not probabilities of correctness. Interpolated shuttle coordinates have no detector score and remain flagged. Coverage, matching expected counts and continuous tracker IDs do not replace accuracy annotations.

M2 exports image pixels only. M3 adds court coordinates; movement summaries follow in M5. Racket angular velocity is deferred from the current release.

Defaults live in `configs\m1.json` and `configs\m2.json`. Explicit CLI values take precedence. Run `run.py --help` for options. `--max-gap-frames 0` disables shuttle interpolation.

## Reproduce the player-only doubles check

The researcher-supplied rear-view clip is stored locally under `data`. Its exact source and archive member are recorded in `data\doubles_clementi_provenance.json`.

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_player_clip.py --video .\data\doubles_candidate_clementi.mp4 --mode doubles --roi .\configs\clementi_player_roi.json --max-frames 90 --out .\results\my-doubles-player-check
```

This validation helper uses actual pose inference and ByteTrack but does not analyze the shuttle or produce a combined match video. It writes player records into a new output folder. Review the delivered comparison images under `results\m2-rear-doubles-check` for the existing check. Camera-edge gaps and missing ankles remain flagged; returning after a gap can create a new ID.
## M1 remains available

The CLI defaults to M1 when `--milestone` is omitted:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_sample.ps1
.\.venv\Scripts\python.exe .\run.py --video .\data\my-rally.mp4 --out .\results\my-new-m1-rally
```

The original completed example is in `results\m1-sample`. M1 uses the official TrackNetV3 checkpoint with eight-frame temporal sequences and background channels. It uses nonoverlapping sequence inference without InpaintNet; metadata records these choices.

## Checks and evidence

```powershell
.\.venv\Scripts\python.exe -m pytest .\tests -q -p no:cacheprovider --basetemp .\working\pytest
.\.venv\Scripts\python.exe -m pip check
```

The M2 delivery suite passed 79 tests in 7.65 seconds, and `pip check` passed. Tests cover meaningful timing, cache, export and player association behavior; they do not establish accuracy on unseen matches. Read `docs\VERIFICATION.md` for M1 evidence and `docs\M2_VERIFICATION.md` for M2 artifact checks and remaining limits.

The public sample has normalized labels from a separate repository. Its exclusion from the shuttle model's training data is unconfirmed. Use independent complete videos or matches for future model evaluation, rather than splitting neighboring frames between training and evaluation.

## Common issues

A missing video error means the supplied path does not exist. Quote paths containing spaces. A missing shuttle checkpoint is resolved by `scripts\download_models.py`; a missing pose checkpoint by `scripts\download_pose_model.py`.

For a CUDA error, use `--device cpu` or prepare a compatible GPU environment. For a nonempty output directory, choose a new folder. A legacy shuttle cache without an input digest must be regenerated once. If player counts differ, inspect the overlay and adjust the ROI for that camera before trusting per-player exports.

A failed run remains under `working` with diagnostics and never appears as a completed result. Setup and CLI entrypoints redirect supported processing and model package state to project paths. Existing application installations and state on C are not migrated.

For court coordinates and the minimap, follow [the M3 guide](M3_RUNNING.md).
