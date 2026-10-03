"""M3 projection and video tests; generated fixtures are software evidence only."""
from copy import deepcopy
import json

import av
import numpy as np
import pytest

from badminton_tracker.court import create_calibration
from badminton_tracker.court_tracking import project_players_on_court
from badminton_tracker.minimap import render_minimap, write_minimap_video
from badminton_tracker.players import COCO17_KEYPOINT_NAMES, make_frame_record
from badminton_tracker.types import Detection
from badminton_tracker.video import VideoError, write_overlay
from test_court import perspective_calibration, synthetic_image_points
from test_video import RELATIVE_TIMES, make_video


def player_at_image_point(x, y, track_id=17):
    points = [{"name": name, "x": float(x), "y": float(y - 20),
               "confidence": 0.9, "visible": True} for name in COCO17_KEYPOINT_NAMES]
    points[15].update(x=float(x - 2), y=float(y))
    points[16].update(x=float(x + 2), y=float(y))
    return {"track_id": track_id, "bbox": [float(x - 20), float(y - 60),
            float(x + 20), float(y + 5)], "confidence": 0.9,
            "keypoints": points, "ankle_midpoint": [float(x), float(y)],
            "low_confidence": False, "identity_source": "bytetrack"}


def player_at_known_court_point(x_m, y_m, track_id=17):
    x, y = synthetic_image_points([[x_m, y_m]])[0]
    return player_at_image_point(x, y, track_id)


def test_projection_preserves_raw_pose_ids_and_outside_coordinates():
    players = [player_at_known_court_point(3.05, 6.7, 17),
               player_at_known_court_point(0.2, 6.7, 92),
               player_at_known_court_point(-0.25, 6.7, 103)]
    frames = [make_frame_record(0, 0.0, players, 2)]
    original = deepcopy(frames)
    projected, summary = project_players_on_court(frames, perspective_calibration(), mode="singles")
    assert frames == original
    assert projected is not frames and projected[0]["players"] is not frames[0]["players"]
    assert [player["track_id"] for player in projected[0]["players"]] == [17, 92, 103]
    positions = [player["court_position"] for player in projected[0]["players"]]
    assert [positions[0]["x_m"], positions[0]["y_m"]] == pytest.approx([3.05, 6.7], abs=2e-5)
    assert positions[0]["in_doubles_court"] is True and positions[0]["in_singles_court"] is True
    assert positions[1]["in_doubles_court"] is True and positions[1]["in_singles_court"] is False
    assert positions[2]["x_m"] == pytest.approx(-0.25, abs=2e-5)
    assert positions[2]["in_doubles_court"] is False and positions[2]["in_singles_court"] is False
    assert all(position["quality_flags"] for position in positions)
    for old, new in zip(frames[0]["players"], projected[0]["players"]):
        assert new["keypoints"] == old["keypoints"]
        assert new["bbox"] == old["bbox"]
        assert new["ankle_midpoint"] == old["ankle_midpoint"]
    assert projected[0]["observed_player_count"] == 3 and projected[0]["count_mismatch"]
    assert isinstance(summary, dict)


@pytest.mark.parametrize("x_m, expected", [(0.45, False), (0.47, True), (5.63, True), (5.65, False)])
def test_singles_sidelines_use_centered_5_point_18_meter_width(x_m, expected):
    frame = make_frame_record(0, 0.0, [player_at_known_court_point(x_m, 6.7)], 2)
    projected, _ = project_players_on_court([frame], perspective_calibration(), mode="singles")
    position = projected[0]["players"][0]["court_position"]
    assert position["in_doubles_court"] is True
    assert position["in_singles_court"] is expected


def test_missing_ankle_and_empty_frame_stay_missing_without_bbox_fallback():
    observed = player_at_known_court_point(3.05, 6.7)
    missing = deepcopy(observed)
    missing["ankle_midpoint"] = None
    missing["keypoints"][15].update(x=None, y=None, visible=False)
    missing["low_confidence"] = True
    frames = [make_frame_record(0, 0.0, [observed], 2),
              make_frame_record(1, 0.04, [missing], 2),
              make_frame_record(2, 0.12, [], 2),
              make_frame_record(3, 0.24, [observed], 2)]
    projected, _ = project_players_on_court(frames, perspective_calibration())
    assert projected[0]["players"][0]["court_position"] is not None
    assert projected[1]["players"][0]["court_position"] is None
    assert projected[1]["players"][0]["court_flags"]
    assert projected[2]["players"] == []
    assert [len(frame["players"]) for frame in projected] == [1, 1, 0, 1]
    assert projected[3]["players"][0]["track_id"] == 17


def test_projection_horizon_returns_flagged_null_instead_of_nonfinite_json():
    frame = make_frame_record(0, 0.0, [player_at_image_point(100.0, -1000.0)], 2)
    projected, _ = project_players_on_court([frame], perspective_calibration())
    player = projected[0]["players"][0]
    assert player["court_position"] is None
    assert player["court_flags"]
    json.dumps(projected, allow_nan=False)


def video_fixture_records():
    calibration = create_calibration([[20, 30], [300, 30], [300, 170], [20, 170]], 320, 180)
    frames = [make_frame_record(index, timestamp, [] if index == 1 else [player_at_image_point(220, 145)], 2)
              for index, timestamp in enumerate(RELATIVE_TIMES)]
    projected, _ = project_players_on_court(frames, calibration)
    shuttle = [Detection(index, timestamp, 160.0, 110.0, 0.9, "detected", False)
               for index, timestamp in enumerate(RELATIVE_TIMES)]
    return calibration, projected, shuttle


def test_standalone_minimap_is_h264_and_preserves_source_irregular_pts(project_tmp_path):
    calibration, frames, _ = video_fixture_records()
    output = project_tmp_path / "minimap.mp4"
    result = write_minimap_video(output, frames, fps=60, calibration=calibration)
    assert isinstance(result, dict)
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        decoded = list(container.decode(video=0))
    assert len(decoded) == 4 and decoded[0].format.name == "yuv420p"
    assert [float(frame.pts * frame.time_base) for frame in decoded] == pytest.approx(
        RELATIVE_TIMES, abs=1 / 90000)
    assert decoded[0].width == 320
    assert not np.array_equal(decoded[0].to_ndarray(format="rgb24"), decoded[1].to_ndarray(format="rgb24"))


def test_combined_minimap_appends_panel_without_scaling_source_pixels(project_tmp_path):
    source = project_tmp_path / "source.mp4"
    make_video(source)
    calibration, frames, shuttle = video_fixture_records()
    original = deepcopy(frames)
    old_path, combined_path = project_tmp_path / "m2.mp4", project_tmp_path / "m3.mp4"
    m2 = write_overlay(source, old_path, shuttle, player_frames=frames)
    m3 = write_overlay(source, combined_path, shuttle, player_frames=frames, court_calibration=calibration)
    with av.open(str(old_path)) as container:
        old = next(container.decode(video=0)).to_ndarray(format="rgb24")
    with av.open(str(combined_path)) as container:
        assert container.streams.video[0].codec_context.name == "h264"
        decoded = list(container.decode(video=0))
    assert old.shape[1] == 320
    assert decoded[0].width == 640
    assert [float(frame.pts * frame.time_base) for frame in decoded] == pytest.approx(
        RELATIVE_TIMES, abs=1 / 90000)
    # Compare the original player box edge, away from calibration lines/header.
    new = decoded[0].to_ndarray(format="rgb24")
    assert np.abs(new[133:145, 197:204].astype(float) - old[133:145, 197:204]).mean() < 8
    assert m2["source_width"] == m3["source_width"] == 320
    assert frames == original


def test_minimap_can_render_empty_and_missing_positions_without_fake_dots():
    empty = make_frame_record(0, 0.0, [], 2)
    unknown = player_at_image_point(150, 120)
    unknown["court_position"] = None
    unknown["court_flags"] = ["ankle_midpoint_unavailable"]
    missing = make_frame_record(0, 0.0, [unknown], 2)
    for frame in (empty, missing):
        image = render_minimap(frame, width=320, height=720)
        assert image.shape == (720, 320, 3) and image.dtype == np.uint8
        assert np.isfinite(image).all()


@pytest.mark.parametrize("corruption", ["duplicate_time", "wrong_index", "nonfinite_time"])
def test_minimap_rejects_invalid_frame_timing(project_tmp_path, corruption):
    calibration, frames, _ = video_fixture_records()
    if corruption == "duplicate_time":
        frames[1]["timestamp_s"] = frames[0]["timestamp_s"]
    elif corruption == "wrong_index":
        frames[1]["frame_index"] = 9
    else:
        frames[1]["timestamp_s"] = float("nan")
    with pytest.raises((VideoError, ValueError)):
        write_minimap_video(project_tmp_path / "bad.mp4", frames, fps=30, calibration=calibration)



def test_minimap_trails_break_at_absence_missing_position_or_off_map():
    state = {}
    def frame(index, x_m=3.05, *, present=True, positioned=True):
        item = player_at_image_point(150, 120)
        item["court_position"] = ({"x_m": x_m, "y_m": 6.7,
            "in_doubles_court": 0 <= x_m <= 6.1,
            "in_singles_court": 0.46 <= x_m <= 5.64,
            "quality_flags": ["approximate_floor_projection"]} if positioned else None)
        return make_frame_record(index, index * 0.04, [item] if present else [], 2)
    render_minimap(frame(0), trail_state=state)
    render_minimap(frame(1, 3.2), trail_state=state)
    assert set(state) == {17} and len(state[17]["points"]) == 2
    render_minimap(frame(2, present=False), trail_state=state)
    assert state == {}
    render_minimap(frame(3), trail_state=state)
    assert len(state[17]["points"]) == 1
    render_minimap(frame(4, positioned=False), trail_state=state)
    assert state == {}
    render_minimap(frame(5), trail_state=state)
    render_minimap(frame(6, 100.0), trail_state=state)
    assert state == {}
    external = frame(7, -0.25)
    before = deepcopy(external)
    render_minimap(external, trail_state=state)
    assert list(state[17]["points"]) == [(-0.25, 6.7)]
    assert external == before


def test_automatic_unreviewed_calibration_flags_every_valid_floor_position():
    original = perspective_calibration()
    automatic = create_calibration(original.corners_px.tolist(), 1500, 1300,
        method="auto-lines", metadata={"manually_reviewed": False})
    frame = make_frame_record(0, 0.0, [player_at_known_court_point(3.05, 6.7)], 2)
    projected, summary = project_players_on_court([frame], automatic)
    flags = projected[0]["players"][0]["court_position"]["quality_flags"]
    assert "unconfirmed_court_calibration" in flags
    assert "approximate_floor_projection" in flags
    assert "unconfirmed_court_calibration" in projected[0]["court_quality_flags"]
    assert summary["calibration_status"] == "unconfirmed_automatic_proposal"


@pytest.fixture
def verified_player_cache(project_tmp_path):
    from pathlib import Path
    import yaml
    from badminton_tracker.cli import DEFAULTS, _sha256
    from badminton_tracker.video import probe_video
    source = project_tmp_path / "source.mp4"
    make_video(source)
    metadata = probe_video(source)
    settings = dict(DEFAULTS)
    tracker = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/bytetrack.yaml").read_text(encoding="utf-8-sig"))
    tracker["track_high_thresh"] = max(tracker["track_high_thresh"], settings["player_confidence"])
    tracker["new_track_thresh"] = max(tracker["new_track_thresh"], settings["player_confidence"])
    roi = [[0, 0], [1, 0], [1, 1], [0, 1]]
    _, frames, _ = video_fixture_records()
    # Stale court fields represent a prior calibration; fresh M3 must recompute.
    for frame in frames:
        for player in frame["players"]:
            player["court_position"]["x_m"] = 999.0
    payload = {"schema_version": "1.0", "metadata": {
        "input_video_sha256": _sha256(source), "video": deepcopy(metadata),
        "settings": deepcopy(settings), "player_roi": roi,
        "player_model": {"tracker": "ByteTrack", "weights_sha256": "a" * 64,
                         "tracker_settings": tracker},
    }, "frames": frames}
    path = project_tmp_path / "players.json"
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    return source, path, metadata, settings, payload


def test_player_cache_revalidates_input_and_strips_stale_court_positions(verified_player_cache):
    from badminton_tracker.cli import _load_cached_players
    source, path, metadata, settings, payload = verified_player_cache
    raw, report = _load_cached_players(path, source, metadata, settings, None)
    assert report["input_sha256_verified"] and report["source_timestamps_verified"]
    assert report["decoded_frames_verified"] == 4
    assert report["cached_roi"] == payload["metadata"]["player_roi"]
    assert report["court_positions_discarded"] is True
    assert all("court_quality_flags" not in frame for frame in raw)
    assert all("court_position" not in player and "court_flags" not in player
               for frame in raw for player in frame["players"])
    assert [player["track_id"] for frame in raw for player in frame["players"]] == [17, 17, 17]
    first = create_calibration([[20, 30], [300, 30], [300, 170], [20, 170]], 320, 180)
    changed = create_calibration([[60, 30], [260, 30], [260, 170], [60, 170]], 320, 180)
    original_positions, _ = project_players_on_court(raw, first)
    new_positions, _ = project_players_on_court(raw, changed)
    assert original_positions[0]["players"][0]["court_position"]["x_m"] == pytest.approx(200 * 6.1 / 280)
    assert new_positions[0]["players"][0]["court_position"]["x_m"] == pytest.approx(160 * 6.1 / 200)
    assert raw[0]["players"][0]["keypoints"] == payload["frames"][0]["players"][0]["keypoints"]
    assert payload["frames"][0]["players"][0]["court_position"]["x_m"] == 999


def test_player_cache_prefix_checks_timing_beyond_requested_prefix(verified_player_cache):
    from badminton_tracker.cli import _load_cached_players
    source, path, metadata, settings, payload = verified_player_cache
    settings["max_frames"] = 2
    raw, report = _load_cached_players(path, source, metadata, settings, None)
    assert len(raw) == 2 and report["decoded_frames_verified"] == 4
    payload["frames"][-1]["timestamp_s"] += 0.01
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source PTS"):
        _load_cached_players(path, source, metadata, settings, None)


@pytest.mark.parametrize("corruption,message", [
    ("digest", "SHA256"), ("dimensions", "metadata mismatch"),
    ("mode", "parameter mismatch"), ("pose_size", "parameter mismatch"),
    ("tracker", "tracker settings mismatch"), ("time", "source PTS"),
    ("short", "complete source frame count"), ("long", "frame count"),
    ("missing_joint", "seventeen"), ("midpoint", "midpoint is inconsistent"),
    ("missing_ankle", "missing ankle"), ("invented_id", "positive ByteTrack"),
    ("invisible_coordinate", "null coordinates"), ("counts", "inconsistent player counts"),
])
def test_player_cache_rejects_content_pose_or_identity_mismatch(verified_player_cache, corruption, message):
    from badminton_tracker.cli import _load_cached_players
    source, path, metadata, settings, payload = verified_player_cache
    item = payload["frames"][0]["players"][0]
    if corruption == "digest":
        payload["metadata"]["input_video_sha256"] = "0" * 64
    elif corruption == "dimensions":
        payload["metadata"]["video"]["width"] += 1
    elif corruption == "mode":
        payload["metadata"]["settings"]["players_mode"] = "doubles"
    elif corruption == "pose_size":
        payload["metadata"]["settings"]["pose_image_size"] += 64
    elif corruption == "tracker":
        payload["metadata"]["player_model"]["tracker_settings"]["track_buffer"] += 1
    elif corruption == "time":
        payload["frames"][2]["timestamp_s"] = 0.1
    elif corruption == "short":
        payload["frames"].pop()
    elif corruption == "long":
        payload["frames"].append(deepcopy(payload["frames"][-1]))
    elif corruption == "missing_joint":
        item["keypoints"].pop()
    elif corruption == "midpoint":
        item["ankle_midpoint"][0] += 3
    elif corruption == "missing_ankle":
        item["keypoints"][15].update(x=None, y=None, visible=False)
    elif corruption == "invented_id":
        item["track_id"] = 0
    elif corruption == "invisible_coordinate":
        item["keypoints"][0]["visible"] = False
    elif corruption == "counts":
        payload["frames"][0]["observed_player_count"] = 2
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _load_cached_players(path, source, metadata, settings, None)


def test_player_cache_rejects_changed_roi_but_accepts_matching_explicit_roi(verified_player_cache):
    from badminton_tracker.cli import _load_cached_players
    source, path, metadata, settings, payload = verified_player_cache
    _load_cached_players(path, source, metadata, settings, None, roi=payload["metadata"]["player_roi"])
    with pytest.raises(ValueError, match="ROI mismatch"):
        _load_cached_players(path, source, metadata, settings, None,
                             roi=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])


def test_court_export_keeps_empty_frames_and_missing_positions_as_null(project_tmp_path):
    import csv
    from badminton_tracker.cli import _export_court_positions
    calibration, frames, _ = video_fixture_records()
    frames[2]["players"][0]["court_position"] = None
    frames[2]["players"][0]["court_flags"] = ["ankle_midpoint_unavailable"]
    _export_court_positions(project_tmp_path, frames, {"milestone": "M3"})
    payload = json.loads((project_tmp_path / "court_positions.json").read_text(encoding="utf-8"))
    assert len(payload["frames"]) == 4 and payload["frames"][1]["players"] == []
    item = payload["frames"][2]["players"][0]
    assert item["track_id"] == 17 and item["x_m"] is None and item["y_m"] is None
    assert "ankle_midpoint_unavailable" in item["quality_flags"]
    with (project_tmp_path / "court_positions.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [int(row["frame_index"]) for row in rows] == [0, 2, 3]
    assert rows[1]["x_m"] == rows[1]["y_m"] == ""
    assert all(row["track_id"] == "17" for row in rows)
