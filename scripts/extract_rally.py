"""Extract a bounded video interval without changing its frame timing.

Video is reencoded as H264 for broad playback support. Audio is not copied.
Outputs and staging stay in the G drive project. Existing outputs are kept.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
for name in ("TEMP", "TMP"):
    os.environ[name] = str(ROOT / "working" / "tmp")
(ROOT / "working" / "tmp").mkdir(parents=True, exist_ok=True)
import av


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(video: Path, start: float, end: float, output: Path, source_url: str | None = None) -> dict:
    video = video.resolve()
    output = output.resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Output must remain inside the project: {ROOT}")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must have an .mp4 extension.")
    if not all(math.isfinite(v) for v in (start, end)) or start < 0 or end <= start:
        raise ValueError("Choose finite times with 0 <= start < end.")
    if end - start > 300:
        raise ValueError("Extract at most 300 seconds at once for bounded processing.")
    provenance = output.with_name(output.stem + "_provenance.json")
    if output.exists() or provenance.exists():
        raise FileExistsError("Video or provenance already exists. Choose a new output name.")
    if not video.is_file():
        raise FileNotFoundError(video)
    stage = ROOT / "working" / f"rally_validation_extract_{uuid.uuid4().hex}.mp4"
    time_base = Fraction(1, 90000)
    source_times = []
    try:
        with av.open(str(video)) as source:
            original = source.streams.video[0]
            original.codec_context.thread_count = 2
            rate = original.average_rate
            if not rate or rate <= 0:
                raise ValueError("Video must report a positive frame rate.")
            if original.duration is not None:
                duration = float(original.duration * original.time_base)
                if start >= duration or end > duration + 1 / float(rate):
                    raise ValueError(f"Requested interval exceeds source duration {duration:.3f}s.")
            else:
                duration = None
            source_codec = original.codec_context.name
            if start:
                source.seek(int(start / original.time_base), stream=original, backward=True, any_frame=False)
            with av.open(str(stage), "w") as destination:
                encoded = destination.add_stream("libx264", rate=rate)
                encoded.width = original.width
                encoded.height = original.height
                encoded.pix_fmt = "yuv420p"
                encoded.time_base = time_base
                encoded.codec_context.time_base = time_base
                encoded.codec_context.thread_count = 2
                encoded.options = {"crf": "18", "preset": "veryfast"}
                first_time = None
                last_pts = None
                for frame in source.decode(original):
                    if frame.pts is None or frame.time_base is None:
                        raise ValueError("Source timestamps are required; no FPS fabrication is performed.")
                    absolute = Fraction(frame.pts) * frame.time_base
                    timestamp = float(absolute)
                    if timestamp < start - 1e-8:
                        continue
                    if timestamp >= end - 1e-8:
                        break
                    if first_time is None:
                        first_time = absolute
                    relative = absolute - first_time
                    pts = round(relative / time_base)
                    if last_pts is not None and pts <= last_pts:
                        raise ValueError("Source timestamps are not strictly increasing.")
                    source_times.append(timestamp)
                    frame.pts = pts
                    frame.time_base = time_base
                    frame = frame.reformat(format="yuv420p")
                    frame.pts = pts
                    frame.time_base = time_base
                    for packet in encoded.encode(frame):
                        destination.mux(packet)
                    last_pts = pts
                if not source_times:
                    raise ValueError("The selected interval contains no decoded frames.")
                for packet in encoded.encode():
                    destination.mux(packet)
        decoded_times = []
        with av.open(str(stage)) as checked:
            stream = checked.streams.video[0]
            stream.codec_context.thread_count = 2
            for frame in checked.decode(stream):
                decoded_times.append(float(frame.pts * frame.time_base))
            output_rate = float(stream.average_rate)
            output_dimensions = [stream.width, stream.height]
        if len(decoded_times) != len(source_times):
            raise RuntimeError("Extracted frame count does not match selected source frames.")
        differences = [abs(actual - (wanted - source_times[0])) for actual, wanted in zip(decoded_times, source_times)]
        maximum_error = max(differences)
        if maximum_error > 1 / 90000 + 1e-8:
            raise RuntimeError("Encoded timestamps do not preserve source frame intervals.")
        record = {
            "schema_version": 1,
            "source_video": str(video),
            "source_video_sha256": sha256(video),
            "source_url": source_url,
            "requested_start_s": start,
            "requested_end_s_exclusive": end,
            "source_first_frame_timestamp_s": source_times[0],
            "source_last_frame_timestamp_s": source_times[-1],
            "source_codec": source_codec,
            "source_duration_s": duration,
            "source_encoded_fps": float(rate),
            "capture_frame_rate_status": "Encoded frame rate verified; original camera capture rate is not established.",
            "output_video": str(output),
            "output_video_sha256": sha256(stage),
            "output_codec": "h264",
            "output_dimensions": output_dimensions,
            "output_encoded_fps": output_rate,
            "frame_count": len(decoded_times),
            "first_output_timestamp_s": decoded_times[0],
            "last_output_timestamp_s": decoded_times[-1],
            "max_timestamp_preservation_error_s": maximum_error,
            "timing_policy": "Preserve selected source presentation intervals, relative to first selected frame; no frame rate conversion.",
            "audio": "not copied",
            "validation_scope": "Full decoded frame count and timestamps; rally selection must be inspected separately.",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        # The exclusive creation rechecks that outputs were not created by another run.
        with output.open("xb") as handle, stage.open("rb") as data:
            for chunk in iter(lambda: data.read(1024 * 1024), b""):
                handle.write(chunk)
        with provenance.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
        return record
    finally:
        if stage.exists():
            stage.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start", type=float, required=True, help="Source start time in seconds, inclusive")
    parser.add_argument("--end", type=float, required=True, help="Source end time in seconds, exclusive; maximum interval 300 seconds")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-url", help="Optional original video attribution URL")
    args = parser.parse_args()
    try:
        record = extract(args.video, args.start, args.end, args.out, args.source_url)
    except (OSError, ValueError, RuntimeError, av.error.FFmpegError) as error:
        parser.exit(2, f"Rally extraction failed: {error}\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())