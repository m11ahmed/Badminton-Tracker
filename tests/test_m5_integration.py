"""M5 CLI contracts and complete cached-source integration."""
import json
from pathlib import Path

import pytest

from badminton_tracker.cli import _load_settings, _parser, main

ROOT = Path(__file__).resolve().parents[1]


def test_m5_requires_geometry_and_review_is_not_accepted_by_earlier_milestones():
    parser = _parser()
    with pytest.raises(ValueError, match="needs --court-calibration"):
        _load_settings(parser.parse_args(["--video", "data/sample.mp4", "--milestone", "m5"]))
    with pytest.raises(ValueError, match="Review notes require"):
        _load_settings(parser.parse_args(["--video", "data/sample.mp4", "--milestone", "m2", "--review-file", "notes.json"]))
    valid = _load_settings(parser.parse_args(["--video", "data/sample.mp4", "--milestone", "m5", "--court-calibration", "configs/sample_court.json", "--players-mode", "doubles"]))
    assert valid["players_mode"] == "doubles"


def test_invalid_review_json_fails_before_inference_and_preserves_existing_files(project_tmp_path):
    review = project_tmp_path / "notes.json"
    review.write_text("[]", encoding="utf-8")
    output = project_tmp_path / "output"
    code = main(["--video", str(ROOT / "data/sample.mp4"), "--milestone", "m5",
        "--court-calibration", str(ROOT / "configs/sample_court.json"),
        "--review-file", str(review), "--out", str(output)])
    assert code == 2 and not output.exists()
    assert review.read_text(encoding="utf-8") == "[]"


def test_m5_exports_real_sample_observations_and_honest_speed_units(project_tmp_path):
    cache = ROOT / "results/m3-sample"
    if not (cache / "players.json").is_file():
        pytest.skip("Local evidence fixture is unavailable; not a neural accuracy test.")
    output = project_tmp_path / "m5"
    code = main(["--video", str(ROOT / "data/sample.mp4"), "--milestone", "m5",
        "--court-calibration", str(ROOT / "configs/sample_court.json"),
        "--player-roi", str(ROOT / "configs/sample_player_roi.json"),
        "--shuttle-results", str(cache), "--player-results", str(cache),
        "--device", "cpu", "--threads", "2", "--out", str(output)])
    assert code == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and summary["provenance"]["milestone"] == "M5"
    assert summary["overlay"]["movement_layer"] is True
    for name in summary["outputs"]:
        assert (output / name).is_file() and (output / name).stat().st_size > 0, name
    insights = json.loads((output / "insights.json").read_text(encoding="utf-8"))
    assert len(insights["shuttle_motion"]) == 81
    assert {row["track_id"] for row in insights["player_motion"]} == {1, 2}
    assert all(row.get("physical_speed_km_h") is None for row in insights["shuttle_motion"])
    assert summary["insights"]["physical_shuttle_speed_km_h"] is None
    assert "court_x_m" in (output / "players.csv").read_text().splitlines()[0]
    assert json.loads((output / "players.json").read_text())["frames"][0]["players"][0]["keypoints"] == json.loads((cache / "players.json").read_text())["frames"][0]["players"][0]["keypoints"]

def test_review_from_a_different_video_is_rejected_before_neural_inference(project_tmp_path):
    notes = project_tmp_path / "review.json"
    notes.write_text(json.dumps({"schema_version": 1, "source_video_sha256": "f" * 64}), encoding="utf-8")
    output = project_tmp_path / "review_output"
    code = main(["--video", str(ROOT / "data/sample.mp4"), "--milestone", "m5",
        "--court-calibration", str(ROOT / "configs/sample_court.json"),
        "--review-file", str(notes), "--out", str(output)])
    assert code == 2 and not output.exists()


def test_changed_pose_overlap_setting_rejects_existing_cache(project_tmp_path):
    cache = ROOT / "results/m3-sample"
    if not (cache / "players.json").is_file():
        pytest.skip("Local evidence fixture is unavailable.")
    output = project_tmp_path / "changed_iou"
    code = main(["--video", str(ROOT / "data/sample.mp4"), "--milestone", "m5",
        "--court-calibration", str(ROOT / "configs/sample_court.json"),
        "--shuttle-results", str(cache), "--player-results", str(cache),
        "--pose-nms-iou", "0.5", "--out", str(output)])
    assert code == 2 and not output.exists()


@pytest.mark.parametrize("threshold", ["0", "1.1", "nan"])
def test_invalid_pose_overlap_threshold_fails_before_inference(threshold):
    with pytest.raises(ValueError, match="pose_nms_iou"):
        _load_settings(_parser().parse_args(["--video", "data/sample.mp4", "--pose-nms-iou", threshold]))
