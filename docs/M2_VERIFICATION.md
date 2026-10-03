# M2 verification

The delivered combined run is `results/m2-verified-sample`. It uses the real 1280 by 720, 30 fps, 81 frame sample from M1 and the pretrained YOLO11n pose checkpoint with ByteTrack. Input provenance remains in `data/sample_provenance.json`; pose weights and their verified hash are recorded in `models/pose_manifest.json`.

## Singles result

Both players were observed in all 81 frames. Tracker IDs 1 and 2 each span frames 0 through 80 without observation gaps. All 162 player observations include an ankle midpoint; visible pose keypoints range from 13 to 16 for ID 1 and 12 to 17 for ID 2. There are 250 suppressed keypoints across the exported poses. Suppressed points remain null rather than being drawn at zero.

Actual decoded output was visually inspected at frames 0, 16, 32, 48, 64 and 80, including cropped player details. ID 1 follows the near player and ID 2 follows the far player in these views. The same image ordering also holds in every frame's ankle records. This is a small continuity check, not a measured identity switch rate or keypoint accuracy benchmark: no athlete identity or pose reference labels are available.

The sample selection polygon rejects officials and spectators before tracking. The full frame still goes to the neural model. Of 954 person candidates across the clip, 792 lie outside the selected region and are rejected; the tracker is not capped to two people. Selection regions must be changed for another camera.

The initial pose input size of 960 missed the far player in frames 42 through 72. Diagnostic frames at 1280 recover that player with the original confidence settings. The corrected whole clip then retained both IDs in all frames. The default pose input size is therefore 1280; smaller inputs can trade recall for faster processing.

Shuttle output remains 66 detections, 4 explicitly marked interpolations and 11 missing frames. This run reused the raw shuttle predictions from `results/m2-sample`, after verifying the input SHA256, complete source frame count, timestamps and inference settings. Interpolation was recomputed. The earlier M1 accuracy caveat still applies: training overlap with the public sample is unconfirmed.

## Artifact and regression checks

The H.264 output decodes to 81 frames at 1280 by 720. Source, output, player records and shuttle records have exactly matching timestamps. CSV and JSON both contain 162 player observations. Verification measurements are in `results/m2-verified-sample/video_verification.json`; actual output frames are in `visualizations/m2-verified-sample-contact-sheet.jpg` and `visualizations/m2-verified-sample-pose-details.jpg`. Native video opening was requested, but interactive browser playback was not tested.

All 79 tests passed in 7.65 seconds, and `pip check` found no broken requirements. Tests include real ByteTrack association through confidence and ROI subsets, gap recovery and tracker reset, null keypoint handling, complete per-frame JSON, CSV alignment, malformed shuttle cache rejection and video timestamp preservation. Synthetic fixtures validate implementation behavior, not empirical detector accuracy.

On this CPU with two inference threads, the corrected run took approximately 121 seconds, including 118 seconds for the player stage. Shuttle inference was cached, so this timing does not represent a fresh combined analysis or GPU performance.

## Rear-view doubles check

A separate author-published amateur doubles rally was tested with actual player inference on its first 90 frames, approximately three seconds. Four players were tracked in every frame from 0 through 55, with consistent IDs 1 through 4 in the inspected views. In frames 56 through 87, the near black-shirt player exits or becomes substantially clipped at the right edge; only three players are tracked, and missing-player warnings remain explicit. Arm or racket fragments remain visible in some gap frames. The player starts partially returning around frame 87 and is tracked again in frames 88 and 89 with a new ID 10. The tracker does not silently merge these fragments into a confirmed athlete identity.

The observed count histogram is four players in 58 frames and three in 32 frames. IDs 1, 3 and 4 each have 90 observations, ID 2 has 56 and ID 10 has two. There are 328 player observations; 23 frames contain low-confidence pose flags. Of 5,576 keypoint observations, 4,885 pass confidence and validity checks, and 306 player observations have an ankle midpoint. These figures describe coverage. The initial clipped ankle is one concrete reason for leaving ankle positions unavailable despite a high box score.

Actual source and tracking layers were visually reviewed at frames 0, 30, 55, 60, 75 and 89. Comparison sheets and detailed records are in `results/m2-rear-doubles-check`; `validation.json` records a zero maximum source timestamp error. An additional sheet shows every source frame from 50 through 89 around the camera-edge gap. This check shows real four-player tracking and an off-camera gap, not general doubles accuracy or long-gap athlete reidentification. It uses a low rear-oblique view, while a complete elevated court view remains the initial analysis target.

Source is the pinned [badminton hit detection research repository](https://github.com/kwyoke/Badminton-hit-detection/blob/7dad97e2da86332158319451ec5c2ab1a1f4a289/README.md), which supplies the raw amateur doubles archive. Archive member, input hash and publication links are in `data/doubles_clementi_provenance.json`. Source presentation timestamps are retained. Only the first 90 of 299 source frames were tested; no shuttle analysis, physical speed or labeled pose accuracy is implied.
## Doubles domain check

The first eight frames of a genuine overhead doubles research GIF were also tested with four expected players. The model emitted three players in seven frames and all four in the eighth frame. Count mismatch flags correctly expose the missing turquoise-shirt player. Accepted poses still had high coverage, showing why confidence and visible keypoint counts alone cannot establish accuracy.

This overhead check is a known failure case, not a successful all-player validation. Results and limits are in `results/m2-doubles-check/validation.json`; the actual source and player layers were inspected in `visualizations/m2-doubles-contact-sheet.jpg`. Source is the pinned [drone doubles research repository](https://github.com/Ning-D/Drone_BD_ControlArea/tree/14a6a74969f54adb57e013faa4fb24298794e985), with URL and hash in `data/doubles_research_provenance.json`. GIF playback timing does not establish original camera capture timing. No shuttle inference or physical speed was measured on this input.

## Scope

M2 supports singles and doubles expected counts, arbitrary observed player counts, persistent tracker IDs and generic COCO pose observations. Its initial target is a fixed elevated rear camera with the complete court visible. Other camera angles, overlapping doubles players, occlusion and cuts need their own labeled checks. No court coordinates in meters, movement speed, racket measurements or coaching scores are included yet. Missing player poses remain missing; downstream movement estimates must not bridge them without a separate, flagged quality policy.

To reproduce the sample or use another clip, follow `docs/RUNNING.md`. M3 follows review of this milestone.
