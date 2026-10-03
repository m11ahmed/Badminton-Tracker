# Badminton Tracker

Badminton Tracker turns a fixed court video into shuttle observations, player movement estimates, court maps and coach review notes. Rally Lab is a local Streamlit app for watching the annotated video and checking the measurements.

## What you get

1. Shuttle positions for each frame, with missing observations and short interpolated gaps identified.
2. Player detections, pose keypoints and persistent tracking fragments for singles or doubles.
3. Approximate court positions in meters, mapped from the midpoint of the two ankles onto the calibrated floor.
4. Distance, average speed, peak retained speed, usable observation time, court zones and player occupancy heatmaps.
5. Shuttle image motion in pixels per second, calculated from consecutive observed positions and source video timing.
6. Annotated video with an embedded court map, a separate map video, per frame CSV and JSON, movement files, shuttle files, summary statistics and a coach review document.
7. Coach entered player names, team and court side, rally boundaries, contacts, outcomes, observations and recovery targets.

## Understand the measurements

Player movement uses accepted ankle positions projected onto the court floor. Distance and speed use continuous observations that pass confidence, gap, calibration and jump checks. Excluded intervals do not count as movement. Each summary reports its usable observation time, so treat the distance as a partial estimate when parts of a player trail are missing.

The heatmap shows observed time in court regions. Its square cells summarize occupancy and do not claim an exact path or a good tactical position. Player tracking IDs identify video fragments, not confirmed athlete identities. A player can receive a new ID after an occlusion or gap. Review and name fragments before combining them.

Shuttle motion is reported in image pixels per second. A single camera does not provide the airborne shuttle height and depth needed for a measured speed in kilometers per hour. Large jumps are flagged for review and may be false detections. They are not automatically removed, and the largest image motion is not a fastest shot estimate.

The tool supports the singles setting with two expected players and the doubles setting with four. It shows detection count warnings instead of inventing missing players. Doubles tracking has only limited sample checks and has not been validated across a full doubles match.

These measurements can help a coach discuss court coverage and recovery after checking the video and confirming player identities and rally boundaries. They do not grade technique, diagnose injury or guarantee improvement. Racket tracking and swing speed are not part of this release.

## Set up on Windows

Use PowerShell and open it in this project folder. Python 3.10 through 3.13 is supported. The setup script creates a project local environment and installs the CPU version of PyTorch by default.

```powershell
.\scripts\setup.ps1
.\.venv\Scripts\python.exe .\scripts\download_models.py
.\.venv\Scripts\python.exe .\scripts\download_pose_model.py
.\.venv\Scripts\python.exe .\scripts\download_sample.py
.\.venv\Scripts\python.exe .\scripts\download_rally.py
```

The download scripts verify checksums and keep model weights and videos in the local project folders. They do not add these large files to GitHub. The small public sample and the rally download are convenience examples, not independent evidence of model accuracy.

For NVIDIA setup, alternate Python choices, full dependency details and troubleshooting, read [the Windows setup guide](docs/RUNNING.md). GPU inference has not been verified on the computer used for this project.

## Launch Rally Lab

```powershell
.\scripts\start_demo.ps1
```

Open `http://127.0.0.1:8501` in your browser and keep the PowerShell window open. The sample downloads alone do not create a completed M5 analysis. Open **Analyze a clip** to select a source video, review its court calibration and player selection region, choose singles or doubles, and start the analysis. The prepared Lee and Axelsen result described in the verification documents uses a separate local video that is not included in this source repository.

For your own recording, use a steady view with the complete court and both or all four players visible. Select one continuous rally. Highlights can contain camera cuts, replays and closeups that need a different calibration. The frame limit of zero analyzes the complete clip. A positive limit is useful for a short test before a full run. CPU tracking can take several minutes for a short rally.

## Review a result

1. Play the annotated video. Its court map inset follows the same video timeline. A download button provides the standalone court map video.
2. In **Positioning**, move the frame control to compare the original image and the matching court diagram.
3. In **Movement and heatmaps**, inspect each tracker fragment, observed distance, average speed, usable time, court zones and heatmap.
4. In **Shuttle motion**, inspect detections, gaps, image motion and review flags.
5. In **Coach review**, check athlete identity, add rally boundaries or contact notes, then save a separate reviewed summary.

## Files and guides

M5 writes an annotated video, standalone map video, per frame shuttle and player observations, court coordinates, movement and shuttle motion CSV and JSON files, summary statistics, occupancy heatmaps and a coach review document. Results are written to a new folder under `results` so previous runs remain available.

Read [the insight definitions](docs/INSIGHTS.md) for metric meanings and quality flags. Follow [the M5 guide](docs/M5_RUNNING.md) for a full movement analysis, [the M3 guide](docs/M3_RUNNING.md) for court calibration, and [the demo guide](docs/DEMO.md) for review controls. The [final verification](docs/FINAL_VERIFICATION.md) documents what was tested and where results may be unreliable. Earlier stage evidence is in [M1 verification](docs/VERIFICATION.md), [M2 verification](docs/M2_VERIFICATION.md) and [M3 verification](docs/M3_VERIFICATION.md).

The Python test suite is in `tests`. Model versions and source checksums are listed in `models/manifest.json`, `models/pose_manifest.json` and `vendor/tracknetv3/provenance.json`. Check upstream terms before redistributing downloaded checkpoints or sample media. No project license is included, so public visibility does not grant permission to reuse this project code.

## Local storage

Videos, model weights, environments, caches, result files and screenshots are kept in local folders excluded from the source commit. To keep all project files together, use a project folder on a drive with enough free space. The sample video and model weights can be downloaded again with the commands above.
