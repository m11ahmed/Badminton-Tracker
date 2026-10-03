# Final verification

Verification date: 4 October 2026. Project version 0.5.0. Racket and swing tracking are outside this release.

## Automated checks

The latest complete regression suite passed 219 tests in 45.54 seconds. Package dependencies pass pip check. All project Python sources parse, PowerShell helpers parse, and the root README contains zero dash characters with all local documentation links resolving.

This PC uses Python 3.12.14, PyTorch 2.8.0 CPU, torchvision 0.23.0 CPU, Ultralytics 8.3.228 and Streamlit 1.65.0. CUDA inference has not been tested here.

## Full court rally

The locally supplied match highlights video is encoded at 1080p and 50 fps. The prepared court view is a separate 536 frame clip spanning 10.72 seconds. Its initial service closeup is excluded. The original highlights remain unmodified and are not distributed with the source repository.

The final M5 result is in results/m5_lee_rally_20261004_004704_911. It includes 18 nonempty exports: the source aligned annotated video, minimap, raw observations, court positions, player movement and shuttle motion CSV/JSON, insights, occupancy heatmaps, calibration and coach report.

Decoded source, annotated and minimap videos contain the same 536 frames at 50 fps. Their per frame timestamps match exactly. The source view is 1920 by 1080, the annotated view is 2240 by 1080 with the map at right, and the standalone map is 320 by 720. CSV and JSON movement intervals reconcile with the reported player distances.

For this prepared singles view, independent 13 frame neural inference tests justified a box overlap threshold of 0.5. All 536 frames produced two tracker IDs. One ID was observed in all frames; the other in 533. Three frames had only one player detection and carry an explicit mismatch flag. Every one of the 1069 detected players has a projected ankle position. IDs remain unconfirmed video track labels. A human should identify the athletes before using the summaries as personal coaching records.

The report contains observed player floor movement, average and peak estimated speed, occupied court zones and heatmaps, with excluded intervals flagged. Estimates are not an independently validated full performance measurement.

## Shuttle motion

TrackNet reported 488 detected positions, 7 short gap interpolations and 41 missing frames. Three clear original image locations were chosen before seeing the model results. The predictions were 2.00, 3.61 and 4.12 source pixels away; these sparse checks are not a clip wide benchmark.

The final report flags three abrupt detector jumps for video review. A separate source frame audit found example detections on player footwear and advertising areas rather than a clear shuttle. The flags intentionally preserve the observations for review; mean and largest detected image motion can still include false detections. Neither is a fastest shot estimate. Physical shuttle speed in km/h remains null because this single camera pipeline cannot recover the airborne shuttle's height and depth.

## Dashboard and Windows commands

The Rally Lab serves locally and leaves footage and results on the project drive. Browser testing in Edge verified the 10.72 second annotated video, a download for the separate court map video, aligned source and court images at frame 535, all review tabs, and matching calibration and player selection for the prepared clip. A single player avoids drift between independent video controls. The browser reported no page errors or failed local requests. Streamlit tests exercised review callbacks, the coach clip span table, downloads, edit validation and source bound review saves.

From a PowerShell window open in the project folder, launch the demo with:

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/start_demo.ps1
```

Then open http://127.0.0.1:8501.

Run the included M5 analysis again with:

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/run_rally.ps1
```

The helper uses the completed cache when it matches source identity and tracking settings. Adding -Fresh performs both neural model runs again.

The calibration polygon was entered and previewed against this source video. I did not manually exercise the OpenCV corner click controls during this session. Automatic court proposals must be visually reviewed before use.

## Doubles and limits

The code and interface accept singles with two expected players or doubles with four. It never fabricates missing players. Movement in a doubles partnership summary needs human reviewed player labels and near or far court sides. A short real rear view check retained four visible doubles players for 56 frames but exposed a player leaving frame and reappearing with a new ID. A separate overhead sample missed an athlete. There is no full match doubles reliability benchmark yet.

Player position and speed depend on visible ankles, camera stability and calibration. Reviews of the court line showed small projection errors, but do not validate true player position. The three missing player frames, tracker confidence, gaps and movement exclusions need viewing against the match. The overlap setting was checked for this singles clip only; 0.7 stays the default for doubles because a stricter suppression threshold can remove close players.

Descriptive movement and occupancy can help a coach discuss court coverage and recovery after adding verified identities and manually selected rallies, contacts and targets. They do not infer technique quality, injury risk or guaranteed player improvement.
