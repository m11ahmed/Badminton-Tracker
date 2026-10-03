# Longer rally input

The raw example at `data/candidate_match.mp4` is a 14.2 second singles broadcast clip showing Axelsen and Ginting. It has 426 frames, 1280 by 720 pixels and a recorded frame rate of 30 fps. Source previews show both players, a rear court view, rally play and the players stopping near the end. It is a useful longer test input; full inference and a new camera calibration remain to be verified.

## Download or verify the included clip

```powershell
.\.venv\Scripts\python.exe .\scripts\download_rally.py
```

The script downloads the pinned source only when the file is absent, verifies its SHA256, decodes all frames and writes a provenance manifest beside it. An existing different file is never overwritten. No full match download, account or additional model is required for this input.

Source: [Good Badminton raw input video](https://github.com/dybtom/Good-Badminton/blob/f53e0b8f7bbeda10ba7848cba7a19fa91406861c/videos/demo.mp4).

Direct MP4: [download the raw rally](https://raw.githubusercontent.com/dybtom/Good-Badminton/f53e0b8f7bbeda10ba7848cba7a19fa91406861c/videos/demo.mp4). A browser may play the file; use its download button or Save video as and save inside this project's data folder.

This is the raw `videos/demo.mp4` input, not the separately annotated `assets/demo.mp4` output. The checksum is `b60cc98c5c5066e31eed777c6cf92bc2d4899d210a5f7285d7358ee7c63d7b26`. The source is convenience footage and is not an independent accuracy benchmark.

## Analysis preparation

Use a new court calibration and player selection region for this camera. Do not reuse `configs/sample_court.json` or the original sample's cached observations. See `M3_RUNNING.md` for the calibration workflow. Preserve the real 30 fps timestamps; an FPS hint of 60 cannot add detail to this recording. Movement summaries and the Streamlit interface are implemented; their completed 50 fps example uses a separate source described below. Racket and swing analysis are deferred from the current release.

## Selected 50 fps court view

The user downloaded [Lee Zii Jia vs Viktor Axelsen, 2021 All England final highlights](https://www.youtube.com/watch?v=2B2_wt96Q5w) to `data/lee_axelsen_50fps.mp4`. Actual local metadata records 1920 by 1080, 50 encoded frames per second, 34624 frames and 692.48 seconds. This verifies the downloaded file timing, not the broadcaster's original capture hardware or any earlier frame interpolation.

The selected court view is `data/lee_axelsen_rally.mp4`: 536 H264 frames, 50 fps, 10.72 seconds, from source times 82.12 through just before 92.84 seconds. The broadcaster's initial service closeup is excluded; this is the visible court-view rally section. The source highlights remain untouched. `data/lee_axelsen_rally_provenance.json` records extraction and source binding. Matching calibration and player selection live in `configs/lee_axelsen_rally_court.json` and `configs/lee_axelsen_rally_roi.json`.

Use `scripts/run_rally.ps1` for the final M5 analysis, or `scripts/start_demo.ps1` to review it. Physical shuttle speed cannot be recovered simply by increasing FPS or projecting an airborne shuttle onto the court floor. See `docs/INSIGHTS.md` for the measurement policy.
