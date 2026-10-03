# Rally Lab local demo

## Start the application

Open PowerShell in the project folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1
```

Open http://127.0.0.1:8501 in a browser. Keep the terminal open. Ctrl+C stops the application. If the port is occupied, append `-Port 8502` and use that port in the browser.

The application listens on localhost. It does not publish footage or results. New source clips, uploads, calibration files, temporary work and review exports remain inside this G drive project. Uploads are limited to 1 GB; placing larger existing files in the data folder avoids a browser upload.

## Review the finished example

1. Select the completed result in the sidebar. M5 analyses are preferred by default, with the latest result first.
2. Watch the annotated video, which already includes the court map inset. Use the Positioning frame control to compare an original source frame with its matching court diagram. Download the standalone court map video when you need it separately.
3. In Positioning, choose a frame. The original frame and the court diagram refer to the same frame index and timestamp.
4. In Movement and heatmaps, select a tracker fragment to inspect retained distance, measured movement time, speed and court zones.
5. In Shuttle motion, inspect the image speed timeline and the numbers of detected, interpolated and missing observations.
6. In Coach review, assign reviewed player names and court sides, add contact or outcome annotations, and optionally set recovery targets and rally boundaries.
7. Save the review or choose Recalculate insights with this review. Recalculation uses existing observations; it does not rerun the neural models. Exports live under a new `results/reviews/<run>/<timestamp>/` folder, while original measurements remain available.

Tracker IDs are fragments. The same athlete can acquire a different ID after a gap. Manual labels may combine nonoverlapping fragments into an athlete summary. Simultaneous IDs cannot be assigned to the same athlete, and movement across fragment gaps is never invented. Review files are bound to the source video digest to prevent transferring identities to another clip by accident.

The editable JSON section allows correcting or removing annotations. Invalid athlete labels, frame numbers, overlapping rallies and invalid target coordinates are rejected before saving. Download buttons export the measurement files and a review JSON.

## Analyze your own rally

Choose a source in the project's data folder, or save an uploaded video there. The prepared `lee_axelsen_rally.mp4` example is a 50 fps uninterrupted court view. Its preset uses a stricter overlapping-person suppression threshold of 0.5 to remove duplicate detections observed in this footage. Other sources retain the general 0.7 threshold; the stricter setting is not applied globally to doubles, where partners can overlap. The original highlights video is a long source and cannot be submitted directly; use Extract one rally to choose start and end times in seconds first. Each extraction creates a fresh clip with preserved source intervals and separate provenance. Court calibration is specific to the camera view and source dimensions.

Choose Singles for two expected players or Doubles for four. Use a matching saved calibration, enter the four outer doubles court corners numerically, or request an automatic proposal. Corner order is far left, far right, near right, near left. Original source pixels are used in the input controls; the image preview shows the selected corners. Singles still calibrates against the outer doubles court geometry and then uses inner sidelines for the active court.

Select a saved player region, or enter a new normalized four-corner region. A region should contain active players and their feet, including the immediate court apron, while excluding officials and spectators. The prepared sample and rally select matching saved regions. Without an explicit selection region, the app allows flagged tracking but movement totals are withheld because player selection is unverified.

Press Analyze rally to start the M5 command. The app launches the project environment directly with separate arguments and no shell interpretation. It displays a live command log and checks a complete summary before reporting success. New output folders never overwrite previous runs. A frame limit of zero analyzes the full prepared clip; a positive limit is useful for a short trial.

CPU inference can take several minutes for a short rally on this computer. Keep the browser tab and terminal open. The application does not provide an estimated completion time or claim a result is complete when the process fails.

## Interpret the measurements

Court positions estimate the ankle midpoint on the floor plane. Jumps, occlusions, camera movement, lens distortion and inaccurate calibration can affect them. Movement smoothing reduces small jitter; gaps, low confidence, unreviewed calibration and implausible jumps exclude intervals. Reported distance is a retained observed estimate, not a complete physical measurement of every step. Analyzed duration refers to the analyzed frames rather than the full source duration when a frame limit was used.

Shuttle image motion uses source pixels per second. Actual speed in km/h is unavailable because an airborne shuttle cannot be located in 3D with the floor homography alone. Do not compare pixel speed across cameras or resolutions. Interpolated and missing positions remain visibly flagged, and image speed alone cannot measure shot quality.

Doubles tracking supports four expected players. Partnership spacing requires reviewed identities and near/far court sides, with all four usable positions present. Tracking gaps and ID changes need human review. Racket tracking and angular velocity are deferred from this release.

Use rally outcomes, contacts and coach-defined recovery targets to interpret movement. More distance or higher average speed does not by itself indicate better badminton.

## Verification

Dashboard tests use the existing M3 and M5 sample outputs and Streamlit's official AppTest API. They check review rendering, partial doubles results, source duration reporting, player labels, invalid contact rejection, recovery targets, source-bound review exports, explicit process launch, error display and blocking full highlights until trimming. AppTest verifies widget behavior; real browser video playback and tracking accuracy are checked separately.

Run the focused checks:

```powershell
.\.venv\Scripts\python.exe -m pytest .\tests\test_dashboard.py -q --basetemp .\working\dashboard-test-tmp
```

Streamlit AppTest reference: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest
