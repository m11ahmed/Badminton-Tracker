"""M2 exports, verified cache reuse and encoded video integration contracts.

Encoded fixtures are software tests and do not establish detection accuracy.
"""
from copy import deepcopy
import csv
import json

import av
import numpy as np
import pytest

from badminton_tracker.cli import (
    DEFAULTS, _export_players, _load_cached_shuttle, _load_player_roi, _sha256,
)
from badminton_tracker.players import COCO17_KEYPOINT_NAMES, make_frame_record
from badminton_tracker.types import Detection
from badminton_tracker.video import (
    VideoError, _draw_players, iter_rgb_frames, probe_video, write_overlay,
)
from test_video import RELATIVE_TIMES, make_video


def pose_player(track_id=17):
    points = [{"name": name, "x": None, "y": None,
               "confidence": 0.1, "visible": False}
              for name in COCO17_KEYPOINT_NAMES]
    for name, x, y in (("left_shoulder", 210.0, 115.0),
                       ("left_elbow", 230.0, 130.0),
                       ("left_wrist", 250.0, 150.0)):
        points[COCO17_KEYPOINT_NAMES.index(name)].update(
            x=x, y=y, confidence=0.9, visible=True)
    return {"track_id": track_id, "bbox": [200.0, 100.0, 280.0, 170.0],
            "confidence": 0.9, "keypoints": points, "ankle_midpoint": None,
            "low_confidence": True, "identity_source": "bytetrack"}


def shuttle_records():
    return [Detection(i, timestamp, 160.0, 110.0, 0.9, "detected", False)
            for i, timestamp in enumerate(RELATIVE_TIMES)]


def player_records():
    return [make_frame_record(i, timestamp, [] if i == 1 else [pose_player()], 2)
            for i, timestamp in enumerate(RELATIVE_TIMES)]


def test_player_json_retains_empty_frames_csv_retains_ids_and_null_pose(project_tmp_path):
    frames = player_records()
    context = {"milestone": "M2", "identity_definition": "ByteTrack fragment"}
    _export_players(project_tmp_path, frames, context)
    exported = json.loads((project_tmp_path / "players.json").read_text(encoding="utf-8"))
    assert exported["schema_version"] == "1.0"
    assert exported["metadata"] == context
    assert exported["frames"] == frames
    assert exported["frames"][1]["players"] == []
    assert "no_players_observed" in exported["frames"][1]["flags"]
    with (project_tmp_path / "players.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [int(row["frame_index"]) for row in rows] == [0, 2, 3]
    assert {row["track_id"] for row in rows} == {"17"}
    assert all(row["identity_source"] == "bytetrack" for row in rows)
    assert all(row["ankle_midpoint_x"] == row["ankle_midpoint_y"] == "" for row in rows)
    assert all(row["right_ankle_x"] == row["right_ankle_y"] == "" for row in rows)
    assert all(row["right_ankle_visible"] == "False" for row in rows)
    assert rows[0]["left_wrist_x"] == "250.0"
    assert rows[0]["left_wrist_visible"] == "True"


def test_all_empty_frames_export_as_json_frames_and_csv_header_only(project_tmp_path):
    frames = [make_frame_record(i, timestamp, [], 4)
              for i, timestamp in enumerate(RELATIVE_TIMES)]
    _export_players(project_tmp_path, frames, {})
    exported = json.loads((project_tmp_path / "players.json").read_text(encoding="utf-8"))
    assert len(exported["frames"]) == 4
    assert all(frame["players"] == [] and frame["count_mismatch"] for frame in exported["frames"])
    with (project_tmp_path / "players.csv").open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert "track_id" in reader.fieldnames
        assert "right_ankle_confidence" in reader.fieldnames
        assert list(reader) == []


@pytest.fixture
def valid_cache(project_tmp_path):
    source = project_tmp_path / "source.mp4"
    make_video(source)
    metadata = probe_video(source, fps_hint=60)
    settings = dict(DEFAULTS)
    frames = [row.to_dict() for row in shuttle_records()]
    # A previously interpolated value is never recycled as a fresh observation.
    frames[1].update(x=161.0, y=111.0, confidence=0.2, source="interpolated",
                     low_confidence=True, raw_x=None, raw_y=None, raw_confidence=0.2)
    payload = {"schema_version": "1.0", "metadata": {
        "input_video_sha256": _sha256(source), "video": deepcopy(metadata),
        "settings": deepcopy(settings), "model": {"name": "test-model"},
        "weights": "fixture.pt", "weights_sha256": "a" * 64,
    }, "frames": frames}
    cache = project_tmp_path / "shuttle.json"
    cache.write_text(json.dumps(payload), encoding="utf-8")
    return source, cache, metadata, settings, payload


def test_cache_reuses_verified_raw_observations_and_reverts_old_interpolation(valid_cache):
    source, cache, metadata, settings, _ = valid_cache
    raw, report = _load_cached_shuttle(cache, source, metadata, settings, 60)
    assert len(raw) == 4
    assert raw[0].source == "detected"
    assert raw[1].source == "missing" and raw[1].x is None and raw[1].y is None
    assert raw[1].low_confidence is True
    assert raw[1].raw_confidence == pytest.approx(0.2)
    assert [row.timestamp_s for row in raw] == pytest.approx(RELATIVE_TIMES)
    assert report["input_sha256_verified"] is True
    assert report["source_timestamps_verified"] is True
    assert report["decoded_frames_verified"] == 4
    assert report["interpolation_recomputed"] is True


def test_cache_prefix_request_still_checks_complete_source_timing(valid_cache):
    source, cache, metadata, settings, payload = valid_cache
    settings["max_frames"] = 2
    raw, report = _load_cached_shuttle(cache, source, metadata, settings, None)
    assert len(raw) == 2 and report["decoded_frames_verified"] == 4
    payload["frames"][-1]["timestamp_s"] += 0.01
    cache.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="timestamps"):
        _load_cached_shuttle(cache, source, metadata, settings, None)


@pytest.mark.parametrize("corruption, message", [
    ("legacy", "no recorded input SHA256"),
    ("sha", "SHA256 does not match"),
    ("metadata", "metadata mismatch"),
    ("settings", "parameter mismatch"),
    ("timing", "timestamps"),
    ("index", "indices"),
    ("short", "complete source frame count"),
    ("long", "frame count"),
    ("missing_raw", "raw detector fields"),
    ("partial_coordinate", "both be present or absent"),
    ("nonfinite", "finite numbers or null"),
])
def test_cache_rejects_unverified_or_misaligned_data(valid_cache, corruption, message):
    source, cache, metadata, settings, payload = valid_cache
    if corruption == "legacy":
        payload["metadata"].pop("input_video_sha256")
    elif corruption == "sha":
        payload["metadata"]["input_video_sha256"] = "0" * 64
    elif corruption == "metadata":
        payload["metadata"]["video"]["width"] += 2
    elif corruption == "settings":
        payload["metadata"]["settings"]["heatmap_threshold"] = 0.1
    elif corruption == "timing":
        payload["frames"][1]["timestamp_s"] = 1 / 60
    elif corruption == "index":
        payload["frames"][2]["frame_index"] = 7
    elif corruption == "short":
        payload["frames"].pop()
    elif corruption == "long":
        payload["frames"].append(deepcopy(payload["frames"][-1]))
    elif corruption == "missing_raw":
        payload["frames"][0].pop("raw_confidence")
    elif corruption == "partial_coordinate":
        payload["frames"][0]["raw_y"] = None
    elif corruption == "nonfinite":
        payload["frames"][0]["raw_x"] = float("nan")
    cache.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _load_cached_shuttle(cache, source, metadata, settings, None)


def test_roi_file_supports_explicit_polygon_and_rejects_self_intersection(project_tmp_path):
    path = project_tmp_path / "roi.json"
    points = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
    path.write_text(json.dumps({"points": points}), encoding="utf-8")
    assert _load_player_roi(path) == [tuple(point) for point in points]
    # This crossing polygon has nonzero algebraic area, so area alone cannot
    # reject it. The explicit intersection check must reject it.
    path.write_text(json.dumps([[0, 0], [0.8, 1], [1, 0], [0, 0.8]]), encoding="utf-8")
    with pytest.raises(ValueError, match="self intersect"):
        _load_player_roi(path)
    assert _load_player_roi(None) is None


def test_drawing_omits_invisible_coordinates_even_when_numeric():
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    item = pose_player()
    item["bbox"] = [20, 20, 40, 40]
    for point in item["keypoints"]:
        point.update(x=None, y=None, visible=False)
    nose = item["keypoints"][0]
    nose.update(x=250.0, y=150.0, confidence=0.1, visible=False)
    wrist = item["keypoints"][9]
    wrist.update(x=100.0, y=150.0, confidence=0.9, visible=True)
    _draw_players(image, make_frame_record(0, 0, [item], 2))
    assert image[145:156, 95:106].max() > 0
    assert image[145:156, 245:256].max() == 0


def test_combined_overlay_is_h264_draws_players_and_preserves_irregular_pts(project_tmp_path):
    source = project_tmp_path / "source.mp4"
    overlay = project_tmp_path / "m2.mp4"
    make_video(source)
    info = write_overlay(source, overlay, shuttle_records(), fps_hint=60,
                         player_frames=player_records())
    assert info["frames_written"] == 4 and info["player_layer"] is True
    with av.open(str(overlay)) as container:
        assert container.streams.video[0].codec_context.name == "h264"
        frames = list(container.decode(video=0))
    assert len(frames) == 4 and frames[0].format.name == "yuv420p"
    assert [float(frame.pts * frame.time_base) for frame in frames] == pytest.approx(
        RELATIVE_TIMES, abs=1 / 90000)
    input_rgb = list(iter_rgb_frames(source))[0][2]
    output_rgb = frames[0].to_ndarray(format="rgb24")
    # Player rectangle exterior is separate from the shuttle marker and banner.
    region = np.s_[120:145, 197:204]
    assert np.abs(output_rgb[region].astype(float) - input_rgb[region]).mean() > 25


@pytest.mark.parametrize("corruption", ["count", "time", "index", "observed"])
def test_overlay_rejects_misaligned_player_records_before_writing(project_tmp_path, corruption):
    source = project_tmp_path / "source.mp4"
    overlay = project_tmp_path / "bad.mp4"
    make_video(source)
    frames = player_records()
    if corruption == "count":
        frames.pop()
    elif corruption == "time":
        frames[1]["timestamp_s"] = 1 / 60
    elif corruption == "index":
        frames[1]["frame_index"] = 2
    elif corruption == "observed":
        frames[1]["observed_player_count"] = 2
    with pytest.raises(VideoError, match="Player"):
        write_overlay(source, overlay, shuttle_records(), player_frames=frames)
    assert not overlay.exists()
