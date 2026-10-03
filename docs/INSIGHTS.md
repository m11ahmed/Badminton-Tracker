# Movement and coaching insights

M5 describes observed player movement, court positioning and shuttle image motion. Racket and swing analysis are outside this release. All exports are written under the project results folder.

## API and output schema

```python
from badminton_tracker.insights import analyze_insights, export_insights, validate_review

insights = analyze_insights(
    shuttle_detections, player_frames, calibration,
    mode="singles", review=None,
)
filenames = export_insights(output_directory, insights)
```

`shuttle_detections` contains `Detection` objects or their dictionaries. `player_frames` contains the M3 frame records with `court_position`. `calibration` is a `CourtCalibration` or `None`. Inputs are copied and never modified. Source indices and timestamps must be unique and strictly increasing. Shuttle and player timestamps must agree wherever their frame indices overlap. `mode` accepts `singles` or `doubles`.

The returned JSON-safe dictionary has these fields:

| Field | Contents |
| --- | --- |
| `schema_version` | `1.0` |
| `players` | One summary per raw tracker fragment, with distance, accepted time, time-weighted average, estimated peak speed, occupancy zones, heatmap and flags |
| `identified_players` | Optional human-mapped fragment aggregates; no movement is added across fragment boundaries |
| `player_motion` | Observation rows including source PTS, court position, local smoothed position, valid position, interval distance/speed or null, zone and flags |
| `shuttle` | Observed coverage, accepted interval coverage, time-weighted mean and maximum image speed; physical km/h is null |
| `shuttle_motion` | Source-frame rows, observed coordinates, pixel displacement and px/s or null, source provenance and flags |
| `rallies` | Explicit reviewed boundaries and accepted player movement within each; absent review yields an explicitly unsegmented whole clip |
| `doubles_pair_spacing` | Status, reason, valid per-frame partner distances and summaries; requires reviewed identities and sides |
| `recovery` | Timing to explicit reviewed recovery targets, or unavailable/not reached |
| `review` | Validated user-supplied context |
| `reviewed_events` | Contact/outcome/note annotations enriched with matching observed court positions and flags |
| `quality` | Reviewed calibration status, counts, warnings and recorded algorithm policy |
| `limitations` | Measurement and interpretation limits |

`export_insights` returns a dictionary mapping artifact names to filenames relative to its output directory. It writes `insights.json`, `player_motion.csv`, `player_motion.json`, `shuttle_motion.csv`, `shuttle_motion.json`, `coaching_review.md`, and a `heatmaps/track_ID.png` for each observed tracker fragment. JSON frame exports have a `frames` array and explicit units. CSV unavailable values are empty cells; JSON unavailable values are null. CSV flag arrays are JSON strings. Existing export files are never overwritten.

## Player distance and speed

The floor reference is the midpoint of the two observed ankle keypoints projected through the court homography. Coordinate origin is the far-left outer doubles corner. X increases across the court, and Y increases toward the near baseline. Coordinates outside the court remain signed and are never clamped. Court dimensions remain 6.1 by 13.4 metres for either mode; singles has narrower active sidelines.

Only finite observed floor positions from a manual calibration, or an automatic proposal marked `metadata.manually_reviewed=true`, contribute to metre metrics. Low-confidence pose, missing anchors, unverified player selection, invalid projection and explicitly flagged cuts/motion are excluded. Positions over one metre beyond the outer court are excluded from movement. Values just outside the court remain valid observed floor estimates and contribute to an `outside_court` occupancy zone.

Each raw tracker ID has separate continuous segments. A missing frame, invalid position, source time gap or implausible raw jump breaks the segment. A time gap means more than the larger of 0.25 seconds and three times the median consecutive source interval. The default raw and smoothed speed guard is 12 m/s. This guard rejects suspicious estimates; it is not a validated biomechanical limit or proof that retained values are accurate. Smoothing cannot hide an already rejected jump.

Within each accepted segment, weighted local linear regression uses actual source timestamps and a nominal 0.20 second window. At sparse cadence, the nearest three points are used. A segment with very small spatial extent, at most 0.08 metres around its median, is treated as stationary to reduce pose jitter. These choices can suppress small movements or brief accelerations. The full policy is saved in the results.

Per-interval speed is smoothed displacement divided by source elapsed time. Distance is the sum of accepted interval displacement. Average speed is distance divided by accepted elapsed time, including accepted stationary intervals. It is not an unweighted average of frame speeds. No distance is added across gaps, cuts or tracker fragments, including fragments manually mapped to one athlete. Consequently, coverage gaps undercount total movement. No accepted intervals yields null distance and speed rather than a misleading zero.

Occupancy adds half of each accepted interval's elapsed time to its two observed endpoint positions. Zone seconds sum to accepted movement time. The 54 by 24 heatmap stores observed seconds inside the outer court; outside time appears only in the zones. Front/mid/back bands are equal thirds of each half court, with front closest to the net. Left/center/right describe X in the fixed court coordinate system, not an athlete's handedness. Heatmaps are descriptive and do not classify a tactical position as correct.

## Shuttle speed

The available speed is source-image displacement in **pixels per second** between two consecutive high-confidence observed detections using their actual presentation timestamps. Both observations must have numeric confidence of at least 0.5, must be within source image bounds when calibration dimensions are available, and must avoid cut flags. Missing, interpolated and low-confidence detections never contribute. The time-weighted image average is total accepted pixel displacement divided by accepted elapsed time. The largest image motion is a two-observation detector estimate, not a fastest-shot estimate. A consecutive jump of at least 10% of the source image diagonal receives a large_image_jump_review flag and a quality warning. That threshold is only a prompt to inspect the footage. The interval remains visible in the CSV and summary because this rule cannot tell a true shuttle movement from a wrong detection; review the annotated video before interpreting the value.

Physical shuttle speed in km/h is deliberately **unavailable**. A floor homography applies to points on the floor. An airborne shuttle can have different heights and depths that produce the same image point. Applying the floor transform would create a false physical speed. Pixel speed also varies with perspective, camera resolution and framing, so it cannot be compared between videos or to measured smash speed. Multiple synchronized calibrated views, a validated 3D reconstruction, or a physical sensor would be needed for a defensible physical measurement.

## Optional reviewed coaching context

Supply a JSON object through the application or CLI review option. These example labels are placeholders and do not identify anyone in the included videos:

```json
{
  "schema_version": 1,
  "identity_map": {
    "1": {"player_label": "Near player", "side": "near", "team": "A"},
    "2": {"player_label": "Far player", "side": "far", "team": "B"}
  },
  "rallies": [
    {"id": "rally1", "start_frame": 0, "end_frame": 80,
     "notes": "Review movement following the lift",
     "outcome": "Outcome entered after video review"}
  ],
  "events": [
    {"frame_index": 20, "kind": "contact", "player_label": "Near player",
     "note": "Contact checked in the video"}
  ],
  "recovery_targets": [
    {"player_label": "Near player", "contact_frame": 20,
     "target_x_m": 3.05, "target_y_m": 10.05,
     "radius_m": 0.5, "until_frame": 80}
  ]
}
```

Optional `source_video_sha256` binds annotations to the exact source video. It must be a 64-character hexadecimal digest; the application supplies it when saving reviews. The CLI rejects a different source digest before inference, and the pure analysis API compares it with calibration provenance when both are present. Unbound manual annotations remain usable with an explicit quality warning. Empty default review context does not trigger that warning.

Every referenced tracker ID and frame must exist. Rally boundaries must be ordered and nonoverlapping. Unknown fields, unknown labels, nonfinite numbers and conflicting simultaneous identity mappings are rejected. A human may map two nonoverlapping tracker fragments to the same label, but their movement remains separate. A reviewed winner label must reference a mapped athlete.

Contact annotations and outcomes are human observations, not a hit detector. Recovery time is the interval from an explicit contact frame to the first observed position within the specified target radius. It is unavailable when the intervening observation sequence has a gap, excluded interval or tracker boundary. Recovery targets reflect the coach's choice; the system does not invent a universal base position.

For doubles, four valid simultaneous floor positions and explicit reviewed identities with two `near` and two `far` side assignments are required for partner spacing. If explicit team labels disagree within a side, that pairing is excluded. Observations flagged as implausible position jumps or smoothed speeds are excluded from spacing and recovery timing even when their floor projection is available for visual review. The report provides observed spacing and its median/minimum/maximum; it does not infer partnerships or classify attack and defence.

## Verification and limits

Run the synthetic suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_insights.py -q --basetemp working\tests\insights_pytest
```

Known-answer tests cover constant velocity with variable timestamps, time-weighted averages, stationary jitter, fast footwork, teleports, missing observations, low confidence, camera cuts, automatic calibration status, raw tracker boundaries, manual mappings, doubles pair spacing, rally boundaries, recovery timing, malformed context, image-speed provenance and real JSON/CSV/PNG exports. These validate the calculation and exclusion policies, not physical measurement accuracy.

The module does not discover camera cuts. Explicit flags break metrics, but unflagged cuts, zooms or camera movement invalidate the transform. Select an uninterrupted fixed-view rally and calibrate it separately. Source-frame count, pose/model confidence and four-corner residuals do not establish accuracy. Improved player performance has not been measured; a coach should review the video and training context before using these estimates.