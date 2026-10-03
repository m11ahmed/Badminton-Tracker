# Movement and coaching review

Open PowerShell in the project root and use the project environment for these commands.

## Included 50 fps rally

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rally.ps1
```

The helper checks the selected rally, camera calibration and player ROI, then runs M5 into a new timestamped directory. It reuses the corrected complete neural observations in `results/m5-lee-rally-final` when available. The prepared singles rally uses `--pose-nms-iou 0.5`, verified to suppress duplicate boxes on this view; other recordings retain the default 0.7 unless you choose a camera-specific setting. A lower overlap threshold can suppress overlapping doubles athletes and is not a universal improvement. Changing this setting invalidates a player cache. `-Fresh` runs the neural models again. CPU analysis is slower than playback on this machine; cached reports and review are much quicker.

The downloaded highlights video remains untouched. A separate H264 court-view clip preserves its 50 fps presentation times. The broadcast uses an initial service closeup, which is excluded from the calibrated view. This clip is one selected court-view rally section, not the entire highlights compilation.

## Original short sample

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_m5_sample.ps1
```

## Your own singles or doubles rally

Prepare a continuous rear court view, four outer doubles court corners, and a player selection polygon for that camera. Use `docs/M3_RUNNING.md` for calibration and `scripts/extract_rally.py --help` for timestamp-preserving trimming. M5 accepts the same tracking and calibration settings as M3.

```powershell
.\.venv\Scripts\python.exe run.py --milestone m5 --video data\my_rally.mp4 --court-calibration configs\my_court.json --player-roi configs\my_roi.json --players-mode doubles --device cpu --threads 2 --out results\my_doubles_review
```

Select `singles` for two expected players and `doubles` for four. The tracker never forces an incorrect count. Occlusion, players leaving the image, and identity switches remain visible in the records. Labels and same-team spacing can be reviewed in the dashboard. Raw fragment IDs are not automatically confirmed athlete identities.

## Outputs

M5 exports the annotated H264 video, separate minimap video, neural shuttle and player observations, court coordinates, player movement CSV/JSON, shuttle image motion CSV/JSON, combined insights, occupancy heatmap PNG files, and a coaching review document. Existing result directories are never overwritten.

Player movement is an approximate floor-position estimate. The analysis uses actual source timestamps, rejects uncertain positions and discontinuities, smooths short contiguous trajectories, and reports the amount of usable observed time. Average speed is distance divided by accepted time. Missing intervals do not count as stationary or get silently joined.

Shuttle image motion is in pixels per second and uses consecutive observed points only. Abrupt jumps receive a review flag and stay visible; they can be wrong detections and the largest displayed value does not measure the fastest shot. Physical shuttle speed in km/h is unavailable for this single-camera pipeline: the airborne shuttle has unknown depth and height. It is not projected onto the floor to fabricate a physical speed. Motion blur, occlusion, detector mistakes and camera movement still affect image motion estimates.

Use the dashboard to review player labels, rally boundaries, contact times, outcomes and recovery targets. Notes are human annotations and remain separate from model detections. No technique grade, injury judgment, automatic winner assertion or guaranteed improvement is inferred from movement alone. Racket tracking, swing speed and its training workflow remain deferred.