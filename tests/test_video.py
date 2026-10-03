"""Real decode/encode round trips check timing, alignment and browser format.

Generated fixture visuals are software tests, not badminton tracking evidence.
"""
from fractions import Fraction

import av
import numpy as np
import pytest

from badminton_tracker.types import Detection
from badminton_tracker.video import VideoError, iter_rgb_frames, probe_video, write_overlay

PTS_MILLISECONDS = [1250, 1290, 1370, 1490]
RELATIVE_TIMES = [0.0, 0.04, 0.12, 0.24]


def make_video(path, *, width=320, height=180, pixel_format="yuv420p"):
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=25)
        stream.width, stream.height = width, height
        stream.pix_fmt = pixel_format
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        stream.codec_context.max_b_frames = 0
        for index, pts in enumerate(PTS_MILLISECONDS):
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            rgb[:, :, :] = (30 + index * 20, 50, 80)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def detections():
    return [Detection(i, timestamp, 160.0, 110.0, 0.9, "detected", False) for i, timestamp in enumerate(RELATIVE_TIMES)]


def test_source_pts_and_metadata_take_precedence_over_fps_hint(project_tmp_path):
    source = project_tmp_path / "source.mp4"
    make_video(source)
    records = list(iter_rgb_frames(source, fps_hint=60))
    assert [record[0] for record in records] == list(range(4))
    assert [record[1] for record in records] == pytest.approx(RELATIVE_TIMES, abs=1e-6)
    assert records[0][2].shape == (180, 320, 3)
    metadata = probe_video(source, fps_hint=60)
    assert metadata["fps"] == metadata["metadata_fps"]
    assert metadata["fps_source"] != "fps_hint"
    assert metadata["timing_warnings"]
    assert len(list(iter_rgb_frames(source, max_frames=2))) == 2


def test_overlay_is_decodable_h264_and_preserves_irregular_timestamps(project_tmp_path):
    source = project_tmp_path / "source.mp4"
    overlay = project_tmp_path / "overlay.mp4"
    make_video(source)
    result = write_overlay(source, overlay, detections(), fps_hint=60)
    assert result["frames_written"] == 4
    with av.open(str(overlay)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        frames = list(container.decode(video=0))
    assert len(frames) == 4
    assert frames[0].format.name == "yuv420p"
    timestamps = [float(frame.pts * frame.time_base) for frame in frames]
    assert timestamps == pytest.approx(RELATIVE_TIMES, abs=1 / 90000)
    # The drawn green marker is below the banner and survives lossy encoding.
    marker_region = frames[0].to_ndarray(format="rgb24")[101:120, 151:170]
    assert marker_region[:, :, 1].max() > 150


def test_odd_source_dimensions_are_padded_without_scaling(project_tmp_path):
    source = project_tmp_path / "odd.mp4"
    overlay = project_tmp_path / "padded.mp4"
    make_video(source, width=319, height=179, pixel_format="yuv444p")
    result = write_overlay(source, overlay, detections())
    assert (result["source_width"], result["source_height"]) == (319, 179)
    assert (result["encoded_width"], result["encoded_height"]) == (320, 180)
    with av.open(str(overlay)) as container:
        frame = next(container.decode(video=0))
    assert (frame.width, frame.height) == (320, 180)


def test_overlay_rejects_records_with_wrong_frame_timing(project_tmp_path):
    source = project_tmp_path / "source.mp4"
    make_video(source)
    records = detections()
    records[1].timestamp_s = 1 / 60
    with pytest.raises(VideoError, match="do not match"):
        write_overlay(source, project_tmp_path / "invalid.mp4", records)


@pytest.mark.parametrize("hint", [0, -30, float("nan"), float("inf")])
def test_invalid_fps_hint_is_rejected(project_tmp_path, hint):
    source = project_tmp_path / "source.mp4"
    make_video(source)
    with pytest.raises(VideoError, match="positive finite"):
        probe_video(source, fps_hint=hint)
