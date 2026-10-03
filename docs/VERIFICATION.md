# M1 verification

Verified on 3 October 2026 on this computer using Python 3.12.14 and PyTorch 2.8.0+cpu. Hardware reports Intel Core i3-2350M at 2.30 GHz; CUDA is unavailable in the installed CPU environment.

## Actual run

```powershell
.\.venv\Scripts\python.exe -m pytest .\tests -q -p no:cacheprovider --basetemp .\working\pytest
.\.venv\Scripts\python.exe .\run.py --video .\data\sample.mp4 --out .\results\m1-sample --device cpu --threads 2
.\.venv\Scripts\python.exe .\scripts\evaluate_sample.py --results .\results\m1-sample
.\.venv\Scripts\python.exe .\scripts\inspect_sample.py
```

Tests: 34 passed. These cover known-pixel heatmap decoding, empty/null observations, timestamp-weighted short-gap interpolation, preserved raw detector fields, no extrapolation or long-gap filling, weak endpoint protection, final sequence padding, source timestamp precedence and real H.264 decode/encode round trips including variable frame timing and odd source dimensions. The sequence padding test uses a stub network and is not model-quality evidence. The real sample run below uses actual pretrained weights.

Dependency check: pip check reports no broken requirements. Python source parsing and the project README dash check pass. A nonexistent input produces an actionable error and exit code 2. Exact installed packages are recorded in requirements-lock.txt.

## Public example and provenance

Source: [alenzenx/TrackNetV3](https://github.com/alenzenx/TrackNetV3), commit d1d96f39d275970d3597a45926f4f46e6ee85e2c, raw_data2/00013.mp4 and raw_data2/00013.csv.

The repository calls raw_data2 a test directory, but exclusion from the selected qaz812345 pretrained checkpoint's training data has not been established. This is a sample check, not an independent held-out benchmark.

Input: 81 frames, 1280 x 720, 30 fps, 2.7 seconds. All source frame presentation timestamps decode correctly. The source annotations contain 75 visible and 6 invisible shuttle frames, with normalized coordinates rounded to three decimal places.

Model: authors' unmodified TrackNet architecture at commit 6eda442ada1740573f200f836d93edc9a541ee86. Checkpoint SHA256 df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a. The checkpoint specifies eight-frame sequences with concatenated background RGB channels.

## Measured results

1. Raw neural detections: 66 of 81 frames.
2. Short interpolated gaps: 4 frames, always flagged as estimates with no detector confidence.
3. Still missing: 11 frames. The initial eight-frame gap remains missing rather than being extrapolated.
4. Matching detections within 8 original image pixels of visible annotations: 62 of 75 visible frames.
5. False detections on annotated invisible frames: 4. No detected visible positions exceed the chosen 8-pixel tolerance.
6. Raw-detection precision under this matching rule: 93.94 percent. Recall: 82.67 percent.
7. Median localization error on visible frames that have a raw detection: 2.91 pixels. This omits misses and must not be read as overall tracking accuracy.
8. Inference including model load: approximately 97 seconds. Complete export: approximately 105 seconds. This CPU configuration is unsuitable for real-time match processing.

Machine-readable measurements are in results/m1-sample/sample_evaluation.json and summary.json.

## Actual output inspection

The encoded overlay decodes to all 81 frames, H.264, yuv420p, 1280 x 720, 30 fps and 2.7 seconds. Source and output presentation timestamps have zero measured difference on this clip. There is no audio.

The contact sheet was generated from the actual encoded output and visually inspected. It includes missing, interpolated and detected frames, with reference crosses added only to the verification sheet. The overlay banner, marker and trail render correctly. The misses near the beginning and around a shot remain visible as missing-data flags.

Output video: results/m1-sample/annotated.mp4. Contact sheet: visualizations/m1-contact-sheet.jpg. Timing verification: results/m1-sample/video_verification.json. Native video opening was requested in Codex and returned queued; interactive browser playback was not tested.

## Remaining limits

Only one short public broadcast clip has been evaluated. There is no established accuracy on the user's own videos, doubles, amateur footage or different camera positions. Scores are uncalibrated. This implementation uses nonoverlapping sequences, omits learned InpaintNet rectification and computes its bounded background median after resizing. These choices are recorded in every run's metadata.

M1 contains no player, court, physical speed, swing measurement or coaching recommendation. Those remain later milestone work.
