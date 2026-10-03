# M3 verification

The reviewed example is `results/m3-sample`. It combines the existing neural shuttle and player observations with an approximate court floor projection and a top-down map. It uses the included short sample and the previously reviewed M2 observations.

## Real sample output

The source contains 81 frames at 1280 by 720 and 30 fps. Both players have available court positions in all 81 frames, giving 162 projected observations. Their original boxes, keypoints, tracker IDs, confidence values and ankle references exactly match the verified M2 records. The shuttle retains 66 detected frames, 4 flagged interpolations and 11 missing frames.

The combined H.264 video is 1600 by 720, with the original source canvas on the left and a 320 pixel map on the right. The standalone map is 320 by 720. Both exported videos decode to 81 frames and have exactly matching source presentation timestamps. They are silent.

Actual encoded frames 0, 16, 36, 48, 64 and 80 were visually reviewed in both videos. Player ID colors and the near/far court movement agree with the source. No shuttle floor position is drawn. Map trajectories break at missing observations, unavailable anchors or tracker gaps.

Projection and rendering took approximately 9.19 seconds using verified existing shuttle and player observations. This excludes fresh neural inference, which takes substantially longer on the tested CPU.

## Calibration checks

The manual source-pixel corners are far-left [454,263], far-right [827,264], near-right [985,653] and near-left [297,654]. These are the outer doubles boundary intersections, also used when reviewing singles. The resulting coordinates have origin at the far-left corner, x across the 6.1 meter width and y toward the near baseline at 13.4 meters. Singles uses the centered 5.18 meter width. Dimensions follow Diagram A in the [BWF Laws of Badminton](https://extranet.bwf.sport/docs/document-system/81/1466/1470/Section%204.1%20-%20Laws%20of%20Badminton%20-%2026%20April%202025%20V5.0%20(2)%20.pdf).

Three visible service-line intersections were independently read from the source frame and were not used to fit the four corners. Their projected pixel discrepancies are approximately 1.14, 0.53 and 1.07 pixels. This supports alignment with these markings in this sample. It does not establish player-position accuracy in centimeters: the pixel readings are approximate, and the ankle observations, floor assumption and lens distortion introduce separate uncertainty.

The court preview was visually inspected against the outer boundaries, singles sidelines and service lines. The plotted net line marks the floor beneath the net, so it appears below the raised net tape in the source camera image. Four-corner fit and inverse projection residuals are numerical consistency checks, not independent physical accuracy measurements.

The automatic detector also completed a separate sample run at `results/m3-auto-sample`. It uses white-line support inside one dominant green playing area and produces an explicitly unconfirmed proposal. The actual encoded first frame visibly labels COURT PROPOSAL, UNCONFIRMED and Check court corners. This detector is a limited fixed-rear-camera heuristic; non-green floors, multiple courts and obscured boundaries can require manual selection.

## Software checks

All 134 tests passed in 9.60 seconds, and dependency checking found no broken requirements. The 55 new tests include a known synthetic perspective camera with held-out court points, degeneracy and horizon handling, singles boundaries, missing ankles without box fallback, outside coordinates without clamping, cache mismatch rejection, stale projection removal and H.264 timing.

The calibration helper ran successfully with explicit pixel corners and with automatic proposals. The corner-clicking interface is implemented with reset, undo, save and cancel controls. Native clicking was not interactively tested in this session. The explicit-coordinate fallback was tested again after redirecting generated previews into `visualizations`. Browser playback was not tested; the actual encoded outputs were decoded and visually inspected.

## Evidence and scope

Measurements and independently checked line points are in `results/m3-sample/video_verification.json`. Inspection images are `visualizations/m3-sample-contact-sheet.jpg`, `visualizations/m3-sample-minimap-contact-sheet.jpg` and `visualizations/sample_court.preview.jpg`. Calibration source dimensions and input digest are recorded in `configs/sample_court.json` and each completed run.

M3 supports approximate player floor coordinates and map review. Missing anchors remain null; outside-court coordinates remain signed and flagged. Camera cuts, movement, jumping and pose uncertainty need review before interpreting the positions. Athlete identity accuracy, full-match robustness and physical movement speed are not established here. Distance summaries, coaching judgments and racket measurements remain later milestones.

Follow `docs/M3_RUNNING.md` to reproduce the sample or calibrate another camera. Racket and swing analysis are outside this release. The full rally validation and M5 coaching review are documented in `docs/FINAL_VERIFICATION.md`.
