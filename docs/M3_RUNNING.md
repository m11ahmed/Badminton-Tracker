# Run M3 court calibration and minimap

M3 adds court coordinates and a top down player map to the shuttle and pose review. Open PowerShell in the project root before using these commands. Keep new inputs and results under the project folder.

## Included sample

Open PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_m3_sample.ps1
```

The helper uses `data\sample.mp4`, the sample-specific player ROI and `configs\sample_court.json`. It verifies existing model and sample digests through the asset scripts; assets are downloaded only when absent. Each run creates a new timestamped output directory and never overwrites a completed run.

By default, the helper reuses the complete observations in `results\m2-verified-sample` when both shuttle and player caches are present. The CLI verifies their source content, frame counts, timestamps, inference settings, tracker settings and ROI before reuse. If that cache pair is absent, the helper runs both models from the source video instead. Existing but invalid caches are rejected rather than accepted on their filenames alone.

To request fresh neural inference explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_m3_sample.ps1 -Fresh
```

You can select another complete observation cache for this exact sample:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_m3_sample.ps1 -ShuttleResults '.\results\m2-verified-sample' -PlayerResults '.\results\m2-verified-sample'
```

Court projection and video rendering with observation reuse are much faster than rerunning the neural models. Measured M3 timing and artifact checks are recorded in `docs\M3_VERIFICATION.md`. Fresh shuttle and player inference can take several minutes for the 2.7 second sample on this CPU. Start with short clips.

The direct sample command is:

```powershell
.\.venv\Scripts\python.exe .\run.py --milestone m3 --video .\data\sample.mp4 --court-calibration .\configs\sample_court.json --player-roi .\configs\sample_player_roi.json --shuttle-results .\results\m2-verified-sample --player-results .\results\m2-verified-sample --players-mode singles --device cpu --threads 2 --out .\results\my-m3-sample
```

Choose a different `--out` folder if that directory already contains a result. For setup, CPU/GPU dependencies and the matched PyTorch, torchvision and Polars compatibility runtime, read `docs\RUNNING.md`.

## Calibrate your own fixed camera

Save a short full-court recording under `data`, then create a calibration file under `configs`:

```powershell
.\.venv\Scripts\python.exe .\scripts\calibrate_court.py --video .\data\my-rally.mp4 --frame 0 --out .\configs\my-court.json
```

The native desktop window shows the selected source frame. Left-click the four outer doubles court corner intersections in this order:

1. Far left.
2. Far right.
3. Near right.
4. Near left.

Press S or Enter to validate and save. R resets the corners, Backspace undoes the last click, and Q or Escape closes the window without saving. Saving creates the JSON calibration and a preview JPEG under `visualizations` so the clicked points and projected service lines can be reviewed. The net marker represents its floor position beneath the raised net; it does not need to overlap the elevated tape in the source image. Existing calibration files are not overwritten; choose a new output filename when revising a camera view.

Use the outer doubles rectangle even for singles. Its dimensions are 6.1 metres across and 13.4 metres along the court. Singles sidelines are inside that rectangle and define a 5.18 metre playing width. Clicking singles corners while using doubles dimensions would stretch the mapping incorrectly.

The coordinate origin is the far-left doubles corner. Positive x points right across the court; positive y points toward the near baseline. The near-right corner is therefore `(6.1, 13.4)` metres. Corners must form an ordered, noncrossing rectangle in perspective.

Choose a frame where the boundary intersections are visible. A fixed camera calibration applies to that camera view. Camera cuts, zoom, movement or a different crop require recalibration. A precise mathematical fit to the clicked corners does not prove physical accuracy away from them; pose error and lens distortion still matter.

The court calibration and player ROI serve different purposes. Calibration converts image floor references into metres. The ROI selects people around the active court and apron while reducing officials and spectators entering the tracks. Supply your own normalized ROI as described in `docs\RUNNING.md`; the public sample ROI belongs only to its camera.

Run fresh M3 analysis on your recording:

```powershell
.\.venv\Scripts\python.exe .\run.py --milestone m3 --video .\data\my-rally.mp4 --court-calibration .\configs\my-court.json --player-roi .\configs\my-camera-roi.json --players-mode singles --device cpu --threads 2 --out .\results\my-first-m3-rally
```

Use `--players-mode doubles` to expect four players. That changes count and active-boundary interpretation; it does not guarantee doubles identity accuracy. No player observations are capped to the expected count.

If you already have complete observations for this recording, add both caches:

```powershell
.\.venv\Scripts\python.exe .\run.py --milestone m3 --video .\data\my-rally.mp4 --court-calibration .\configs\my-court.json --player-roi .\configs\my-camera-roi.json --players-mode singles --shuttle-results .\results\my-first-m2-rally --player-results .\results\my-first-m2-rally --out .\results\my-first-m3-from-cache
```

Caches must cover the complete source and use compatible settings. If no explicit ROI is supplied for player reuse, the CLI inherits the cached ROI. A supplied ROI must match the cached selection. To change the ROI or pose inference settings, run fresh player inference by omitting `--player-results`. Old M3 court fields are discarded when observations are reused, then recomputed from the current calibration. Cached floor positions never substitute for raw image observations.

## Automatic court candidate

As an alternative to the clicked calibration, use `--auto-court`:

```powershell
.\.venv\Scripts\python.exe .\run.py --milestone m3 --video .\data\sample.mp4 --auto-court --player-roi .\configs\sample_player_roi.json --shuttle-results .\results\m2-verified-sample --player-results .\results\m2-verified-sample --out .\results\my-m3-auto-candidate
```

Automatic detection proposes corners from the court lines or playing area. Analysis can complete with that proposal, but the calibration and projected positions retain `unconfirmed_court_calibration`. The video labels it as an unconfirmed court proposal and the map says to check its corners. Compare the projected boundaries with the actual outer court lines before interpreting its metre coordinates. Manual corner calibration is the fallback when the proposal is inaccurate or detection fails. Use either `--auto-court` or `--court-calibration` in a run.

## Outputs

Each completed M3 run contains ten output files:

1. `annotated.mp4` combines the original source-coordinate overlay with a separate map panel appended on the right. Source pixels retain their original positions and presentation timing. The video has no audio.
2. `minimap.mp4` is a standalone timed player map with court boundaries, singles sidelines, service lines, net and raw tracker ID colors.
3. `court_positions.csv` and `court_positions.json` contain the approximate player floor coordinates in metres, boundary membership and quality flags. JSON retains every frame, including frames without available positions.
4. `court_calibration.json` records the current corners, transform, origin, dimensions and calibration provenance.
5. `players.csv` and `players.json` retain image-space boxes, pose points, ankle references, counts and raw tracker IDs. M3 also records current projected positions and their flags.
6. `shuttle.csv` and `shuttle.json` retain shuttle observations in source image pixels, with missing and interpolation flags.
7. `summary.json` records model and input provenance, observation reuse, timing, tracking coverage and court projection quality.

The annotated video's width is larger because the minimap is appended, rather than placed over the players or shuttle. Both videos preserve source timestamps. `--fps-hint` supplies timing only when source metadata is missing; it never converts valid 30 fps footage into 60 fps footage or recovers a slow motion capture rate.

## Interpret the map carefully

A court position uses the midpoint of two observed ankle keypoints projected onto an approximate planar floor. Missing ankles produce null coordinates; there is no box-center fallback or invented player position. Jumping, perspective, low confidence poses and occlusion can displace the ankle midpoint from the actual floor reference.

Coordinates remain signed estimates. A player outside the active court is flagged and is not clamped back inside it. Nearby outside positions appear in a small map apron; positions beyond the displayed apron produce an off-map warning. Trails retain only short contiguous observations and break across missing anchors or track gaps.

Raw ByteTrack IDs and matching colors describe tracker trajectories, not verified athlete identities. An apparent path can still contain an identity switch during an overlap. Review the source clip alongside the map before making player-specific judgments.

The airborne shuttle remains in image coordinates. A floor homography cannot recover its physical floor location or height. M3 does not calculate distance covered, physical speed, swing angular velocity, power or coaching scores. Those need later measurement work and validation.

Read `docs\M3_VERIFICATION.md` for completed M3 sample checks and their limits. Native corner clicking, automatic candidate detection and new camera reliability must be described according to what was actually inspected. M3 is implemented. Racket analysis is outside this release. Longer rally validation, M5 movement summaries and the M6 Streamlit app are complete; see `docs\FINAL_VERIFICATION.md` for evidence.
