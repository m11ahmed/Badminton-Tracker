"""Inspect actual encoded sample frames and compare output timestamps."""
import csv
import json
from pathlib import Path
import sys

import av
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from badminton_tracker.video import iter_rgb_frames, probe_video

def main():
    results = ROOT / "results" / "m1-sample"
    payload = json.loads((results / "shuttle.json").read_text(encoding="utf-8"))
    source = list(iter_rgb_frames(ROOT / "data" / "sample.mp4"))
    encoded = list(iter_rgb_frames(results / "annotated.mp4"))
    if len(source) != len(encoded) or len(encoded) != len(payload["frames"]):
        raise RuntimeError("Source, records and exported video frame counts differ.")
    largest_time_error = max(abs(left[1] - right[1]) for left, right in zip(source, encoded))
    if largest_time_error > 1/90000 + 1e-8:
        raise RuntimeError("Overlay timing differs from source beyond the output time base.")
    with (ROOT / "data" / "sample_ground_truth.csv").open(encoding="utf-8-sig", newline="") as file:
        labels = {int(row["Frame"]): row for row in csv.DictReader(file)}
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 19)
    except OSError:
        font = ImageFont.load_default()
    selected = [4, 16, 36, 48, 64, 80]
    cell_width, cell_height = 640, 410
    sheet = Image.new("RGB", (cell_width * 2, cell_height * 3 + 80), "#111923")
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 12), "M1 actual video frames: green = model, amber = estimate or low score", font=font, fill="white")
    draw.text((18, 40), "Magenta cross = public annotation, added only for this verification sheet", font=font, fill="#ee99ff")
    for position, index in enumerate(selected):
        image = Image.fromarray(encoded[index][2])
        label = labels[index]
        if int(label["Ball"]) == 1:
            x, y = float(label["x"]) * image.width, float(label["y"]) * image.height
            marking = ImageDraw.Draw(image)
            marking.line((x-12,y,x+12,y),fill="#ee66ff",width=3)
            marking.line((x,y-12,x,y+12),fill="#ee66ff",width=3)
        image = image.resize((640,360))
        row = payload["frames"][index]
        x0, y0 = (position % 2)*cell_width, (position // 2)*cell_height+80
        sheet.paste(image,(x0,y0))
        draw.text((x0+12,y0+365),f"Frame {index} | {row['timestamp_s']:.3f}s | {row['source']}",font=font,fill="white")
    destination = ROOT / "visualizations" / "m1-contact-sheet.jpg"
    sheet.save(destination,quality=94)
    report = {
        "source_frames":len(source), "output_frames":len(encoded),
        "maximum_source_output_timestamp_difference_s":largest_time_error,
        "output_probe":probe_video(results/"annotated.mp4"),
        "contact_sheet":str(destination),
        "inspection_scope":"Actual decoded encoded-output frames and timing; browser playback was not tested.",
    }
    (results/"video_verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__ == "__main__":
    main()
