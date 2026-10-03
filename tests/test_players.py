"""M2 selection, confidence and real ByteTrack association regression tests.

These fixtures test software contracts; synthetic boxes are not model accuracy
or proof that doubles footage is tracked correctly.
"""
import numpy as np
import pytest

from badminton_tracker.players import (
    COCO17_KEYPOINT_NAMES,
    _IndexedDetections,
    _create_tracker,
    ankle_midpoint,
    format_keypoints,
    make_frame_record,
    select_player_indices,
    summarize_player_tracks,
    validate_player_roi,
)


def valid_pose():
    pose = np.tile([120.0, 80.0, 0.9], (17, 1))
    pose[15] = [100.0, 150.0, 0.8]
    pose[16] = [140.0, 154.0, 0.95]
    return pose


def player(track_id):
    points = format_keypoints(valid_pose())
    return {
        "track_id": track_id, "bbox": [90.0, 40.0, 150.0, 160.0],
        "confidence": 0.9, "keypoints": points,
        "ankle_midpoint": ankle_midpoint(points), "low_confidence": False,
        "identity_source": "bytetrack",
    }


def test_coco_order_and_null_coordinates_for_untrusted_keypoints():
    raw = valid_pose()
    raw[0] = [10, 20, 0.29]
    raw[1] = [0, 0, 0.95]
    raw[2] = [float("nan"), 50, 0.95]
    raw[3] = [50, float("inf"), 0.95]
    raw[4] = [50, 50, float("nan")]
    raw[5] = [301, 100, 0.95]
    raw[6] = [80, -1, 0.95]
    points = format_keypoints(raw, threshold=0.3, image_shape=(180, 300))
    assert [point["name"] for point in points] == list(COCO17_KEYPOINT_NAMES)
    assert len(points) == 17
    for point in points[:7]:
        assert point["visible"] is False
        assert point["x"] is None and point["y"] is None
        assert np.isfinite(point["confidence"])
    assert points[0]["confidence"] == pytest.approx(0.29)
    assert points[7]["visible"] is True
    assert points[7]["x"] == 120.0


def test_threshold_is_inclusive_but_both_ankles_are_required():
    raw = valid_pose()
    raw[15, 2] = 0.3
    points = format_keypoints(raw, threshold=0.3)
    assert ankle_midpoint(points) == pytest.approx([120.0, 152.0])
    raw[15, 2] = 0.299
    assert ankle_midpoint(format_keypoints(raw, threshold=0.3)) is None
    raw[15, 2] = 0.9
    raw[16] = [0, 0, 0.95]
    assert ankle_midpoint(format_keypoints(raw)) is None


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_keypoint_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        format_keypoints(valid_pose(), threshold=threshold)


@pytest.mark.parametrize("roi", [
    [[0, 0], [1, 1]],
    [[0, 0], [0.5, 0.5], [1, 1]],
    [[0, 0], [1, 0], [1, 1], [0, 0]],
    [[0, 0], [1.1, 0], [1, 1]],
    [[0, 0], [float("nan"), 0], [1, 1]],
])
def test_invalid_or_degenerate_roi_is_rejected(roi):
    with pytest.raises(ValueError, match="ROI"):
        validate_player_roi(roi)


def test_roi_selection_preserves_original_indices_and_does_not_cap_counts():
    boxes = np.array([[0, 30, 20, 150], [65, 30, 95, 150],
                      [270, 20, 290, 150], [205, 20, 235, 150]], np.float32)
    poses = np.zeros((4, 17, 3), np.float32)
    roi = [[0.2, 0.2], [0.8, 0.2], [0.8, 1], [0.2, 1]]
    assert select_player_indices(boxes, poses, roi, 300, 180) == [1, 3]
    assert select_player_indices(boxes, poses, None, 300, 180) == [0, 1, 2, 3]
    boxes = np.tile([65, 30, 95, 150], (5, 1))
    poses = np.zeros((5, 17, 3), np.float32)
    assert select_player_indices(boxes, poses, roi, 300, 180) == list(range(5))


@pytest.mark.parametrize("expected, ids, mismatch", [
    (2, [], True), (2, [17], True), (2, [17, 92], False),
    (2, [17, 92, 103], True), (4, [17, 92, 103], True),
    (4, [17, 92, 103, 209], False), (4, [17, 92, 103, 209, 310], True),
])
def test_count_warnings_preserve_all_raw_tracker_ids(expected, ids, mismatch):
    observed = [player(identity) for identity in ids]
    record = make_frame_record(3, 0.12, observed, expected)
    assert record["observed_player_count"] == len(ids)
    assert record["count_mismatch"] is mismatch
    assert ("player_count_mismatch" in record["flags"]) is mismatch
    assert [item["track_id"] for item in record["players"]] == ids
    assert record["players"] == observed
    if not ids:
        assert record["players"] == []
        assert "no_players_observed" in record["flags"]


def test_track_summary_reports_fragments_and_gaps_without_filling_identity():
    frames = [make_frame_record(0, 0.0, [player(17), player(92)], 2),
              make_frame_record(1, 0.04, [player(17)], 2),
              make_frame_record(2, 0.08, [player(17), player(92)], 2),
              make_frame_record(3, 0.12, [player(17), player(103)], 2)]
    summary = summarize_player_tracks(frames, 2)
    assert summary["unique_track_ids"] == 3
    assert [item["track_id"] for item in summary["track_fragments"]] == [17, 92, 103]
    second = summary["track_fragments"][1]
    assert second["observed_frames"] == 2
    assert second["gaps"] == [{"after_frame": 0, "before_frame": 2, "missing_frames": 1}]
    assert summary["count_mismatch_frame_indices"] == [1]
    assert summary["identity_switch_rate"] is None
    assert summary["warnings"]
    assert len(frames[1]["players"]) == 1


@pytest.fixture
def tracker_factory(monkeypatch, project_tmp_path):
    # Set this before the first Ultralytics import to keep generated app settings
    # in this project's G drive test directory.
    config_dir = project_tmp_path / "ultralytics"
    config_dir.mkdir()
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("YOLO_AUTOINSTALL", "false")
    settings = {"tracker_type": "bytetrack", "track_high_thresh": 0.25,
                "track_low_thresh": 0.1, "new_track_thresh": 0.25,
                "track_buffer": 30, "match_thresh": 0.8, "fuse_score": True}
    return lambda: _create_tracker(settings, 30.0)


def associated_sources(tracks):
    return {int(track[4]): int(track[-1]) for track in tracks}


def test_real_bytetrack_pose_indices_survive_roi_and_high_low_confidence_splits(tracker_factory):
    tracker = tracker_factory()
    original_boxes = np.array([[0, 30, 20, 150], [65, 30, 95, 150],
                               [270, 20, 290, 150], [205, 20, 235, 150]], np.float32)
    original_poses = np.zeros((4, 17, 3), np.float32)
    roi = [[0.2, 0.2], [0.8, 0.2], [0.8, 1], [0.2, 1]]
    accepted = select_player_indices(original_boxes, original_poses, roi, 300, 180)
    assert accepted == [1, 3]
    first = _IndexedDetections(original_boxes[accepted], [0.9, 0.9], [0, 0], accepted)
    assert associated_sources(tracker.update(first)) == {1: 1, 2: 3}
    # Player1 now falls into ByteTrack's separate LOW confidence association
    # stage, while player2 enters HIGH. Both must still address original poses.
    second = _IndexedDetections(original_boxes[accepted], [0.15, 0.9], [0, 0], accepted)
    assert associated_sources(tracker.update(second)) == {1: 1, 2: 3}
    # Model detection order changes; physical boxes keep their raw track IDs.
    third = _IndexedDetections(original_boxes[[3, 1]], [0.9, 0.9], [0, 0], [3, 1])
    assert associated_sources(tracker.update(third)) == {1: 1, 2: 3}


def test_real_bytetrack_does_not_emit_synthetic_players_in_empty_frame(tracker_factory):
    tracker = tracker_factory()
    first = _IndexedDetections([[65, 30, 95, 150]], [0.9], [0], [7])
    assert associated_sources(tracker.update(first)) == {1: 7}
    empty = _IndexedDetections(np.empty((0, 4)), [], [], [])
    assert len(tracker.update(empty)) == 0
    recovered = tracker.update(first)
    assert associated_sources(recovered) == {1: 7}


def test_real_bytetrack_state_and_id_counter_reset_for_each_video(tracker_factory):
    first = tracker_factory()
    boxes = _IndexedDetections([[65, 30, 95, 150], [205, 20, 235, 150]],
                               [0.9, 0.9], [0, 0], [1, 3])
    assert set(associated_sources(first.update(boxes))) == {1, 2}
    assert first.frame_id == 1
    second = tracker_factory()
    assert second.frame_id == 0
    assert not second.tracked_stracks and not second.lost_stracks
    assert set(associated_sources(second.update(boxes))) == {1, 2}


@pytest.mark.parametrize("threshold", [0,-.1,1.1,float("nan"),float("inf"),True,None,".5"])
def test_invalid_box_nms_iou_threshold_is_rejected(threshold):
    from badminton_tracker.players import validate_box_nms_iou
    with pytest.raises(ValueError,match="NMS IoU"):
        validate_box_nms_iou(threshold)


def test_lower_real_yolo_nms_iou_removes_nested_duplicate_and_preserves_distinct_people(tracker_factory):
    import torch
    from ultralytics.utils.nms import non_max_suppression
    # Independent boxes: main athlete, nested duplicate IoU=.64, distinct athlete.
    prediction = torch.tensor([[[50.,50.,100.,100.,.95],
                                [50.,50.,80.,80.,.85],
                                [250.,50.,100.,100.,.90]]]).transpose(1,2)
    default = non_max_suppression(prediction.clone(),conf_thres=.1,iou_thres=.7,nc=1)[0]
    lower = non_max_suppression(prediction.clone(),conf_thres=.1,iou_thres=.5,nc=1)[0]
    assert len(default) == 3
    assert len(lower) == 2
    np.testing.assert_allclose(lower[:, :4].numpy(),[[0,0,100,100],[200,0,300,100]],atol=1e-6)
    assert sorted(lower[:,4].tolist()) == pytest.approx([.90,.95])


def test_tracking_passes_requested_nms_iou_to_model_and_records_it(monkeypatch,project_tmp_path,tracker_factory):
    import torch
    import ultralytics
    from ultralytics.engine.results import Results
    import badminton_tracker.players as module
    image = np.zeros((200,300,3),dtype=np.uint8)
    metadata = {"fps":30.,"frame_count":1,"width":300,"height":200}
    monkeypatch.setattr(module,"probe_video",lambda *args:metadata)
    monkeypatch.setattr(module,"iter_rgb_frames",lambda *args,**kwargs:iter([(0,0.,image)]))
    calls=[]
    class FakePoseModel:
        def __init__(self,*args,**kwargs): pass
        def predict(self,**kwargs):
            calls.append(kwargs)
            boxes = torch.tensor([[90.,40.,150.,160.,.9,0.]])
            poses = torch.tensor(np.array([valid_pose()]),dtype=torch.float32)
            return [Results(image,path="synthetic",names={0:"person"},boxes=boxes,keypoints=poses)]
    monkeypatch.setattr(ultralytics,"YOLO",FakePoseModel)
    weight = project_tmp_path/"unit_fake_weights.pt"
    weight.write_bytes(b"mock fixture; never loaded")
    roi = [[.1,.1],[.8,.1],[.8,1],[.1,1]]
    records, context = module.track_players(project_tmp_path/"fake.mp4",weight,player_roi=roi,device="cpu",box_nms_iou=.5)
    assert calls[0]["iou"] == .5
    assert context["box_nms_iou"] == .5
    assert "no player-count cap" in context["box_nms_policy"]
    assert len(records) == 1 and len(records[0]["players"]) == 1
