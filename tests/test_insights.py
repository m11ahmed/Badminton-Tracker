"""Independent known-answer tests for quality-aware movement and exports."""
from copy import deepcopy
import csv
import json
import math

import numpy as np
import pytest
from PIL import Image

from badminton_tracker.court import create_calibration
from badminton_tracker.insights import analyze_insights, export_insights, validate_review
from badminton_tracker.types import Detection


def calibration(method="manual", reviewed=False):
    return create_calibration([[100,100],[800,100],[800,700],[100,700]], 1000, 800,
                              method=method, metadata={"manually_reviewed": reviewed})


def player(raw_id, x, y=3, low=False):
    return {"track_id": raw_id, "low_confidence": low,
            "court_position": {"x_m": x, "y_m": y, "quality_flags": ["approximate_floor_projection"]},
            "court_flags": []}


def frames(times, positions=None):
    if positions is None:
        positions = [[player(1, 1 + timestamp)] for timestamp in times]
    return [{"frame_index": index, "timestamp_s": timestamp, "players": observations,
             "flags": [], "count_mismatch": False} for index, (timestamp, observations) in enumerate(zip(times, positions))]


def shuttles(times, positions=None):
    positions = positions or [[100 + 100 * timestamp, 100] for timestamp in times]
    return [Detection(index, timestamp, x, y, .9, "detected", False) for index, (timestamp, (x, y)) in enumerate(zip(times, positions))]


def test_constant_velocity_variable_source_pts_has_known_distance_and_time_average():
    times = [0., .03, .09, .13, .20]
    observations = frames(times, [[player(1, 1 + 1.5*t)] for t in times])
    result = analyze_insights(shuttles(times), observations, calibration())
    summary = result["players"][0]
    assert summary["distance_m"] == pytest.approx(.3)
    assert summary["valid_motion_time_s"] == pytest.approx(.2)
    assert summary["average_speed_m_s"] == pytest.approx(1.5)
    assert summary["max_speed_m_s"] == pytest.approx(1.5)
    assert sum(summary["zones"].values()) == pytest.approx(.2)
    assert np.sum(summary["heatmap"]["values_s"]) == pytest.approx(.2)
    assert result["shuttle"]["average_image_speed_px_s"] == pytest.approx(100)


def test_stationary_pose_jitter_is_not_accumulated_into_distance():
    times = [index / 50 for index in range(30)]
    observations = frames(times, [[player(1, 2 + (.01 if index % 2 else -.01), 3 + (.01 if index % 3 else -.01))] for index in range(len(times))])
    summary = analyze_insights([], observations, calibration())["players"][0]
    assert summary["distance_m"] == pytest.approx(0)
    assert summary["average_speed_m_s"] == pytest.approx(0)
    assert summary["motion_intervals"] == 29


def test_fast_valid_footwork_is_preserved():
    times = [index / 50 for index in range(10)]
    observations = frames(times, [[player(1, 1 + 8*t)] for t in times])
    summary = analyze_insights([], observations, calibration())["players"][0]
    assert summary["average_speed_m_s"] == pytest.approx(8)
    assert summary["distance_m"] == pytest.approx(1.44)


def test_missing_player_breaks_motion_and_does_not_bridge_gap():
    times = [0., .04, .08, .12, .16]
    observations = frames(times, [[player(1, 1)], [player(1, 1.04)], [], [player(1, 2)], [player(1, 2.04)]])
    result = analyze_insights([], observations, calibration())
    assert result["players"][0]["distance_m"] == pytest.approx(.08)
    assert result["players"][0]["valid_motion_time_s"] == pytest.approx(.08)
    assert result["player_motion"][2]["speed_m_s"] is None
    assert "observation_gap" in result["player_motion"][2]["quality_flags"]


def test_low_confidence_position_is_excluded_with_neighbor_intervals():
    times = [0., .04, .08, .12, .16]
    observations = frames(times)
    observations[2]["players"][0]["low_confidence"] = True
    result = analyze_insights([], observations, calibration())
    assert result["players"][0]["distance_m"] == pytest.approx(.08)
    assert result["player_motion"][2]["valid_position"] is False
    assert result["player_motion"][3]["speed_m_s"] is None


def test_teleport_is_rejected_before_smoothing_can_hide_it():
    times = [0., .04, .08, .12, .16]
    observations = frames(times, [[player(1,x)] for x in [1,1.04,5,1.12,1.16]])
    result = analyze_insights([], observations, calibration())
    assert result["players"][0]["distance_m"] == pytest.approx(.08)
    assert result["player_motion"][2]["speed_m_s"] is None
    assert result["player_motion"][3]["speed_m_s"] is None
    assert "implausible_position_jump" in result["player_motion"][2]["quality_flags"]


def test_source_time_jump_breaks_motion_without_replacing_pts_with_fps():
    times = [0., .02, .04, 1., 1.02]
    result = analyze_insights([], frames(times), calibration())
    assert result["players"][0]["valid_motion_time_s"] == pytest.approx(.06)
    assert result["player_motion"][3]["speed_m_s"] is None


def test_explicit_camera_cut_breaks_motion_and_shuttle_speed():
    times = [0., .04, .08, .12, .16]
    observations = frames(times)
    observations[2]["flags"] = ["camera_cut"]
    result = analyze_insights(shuttles(times), observations, calibration())
    assert result["players"][0]["distance_m"] == pytest.approx(.08)
    assert result["shuttle_motion"][2]["image_speed_px_s"] is None
    assert result["shuttle_motion"][3]["image_speed_px_s"] is None
    assert result["shuttle_motion"][4]["image_speed_px_s"] == pytest.approx(100)


def test_unknown_position_has_no_bbox_fallback():
    observations = frames([0., .04, .08])
    observations[1]["players"][0]["court_position"] = None
    observations[1]["players"][0]["bbox_xyxy"] = [100,100,200,300]
    result = analyze_insights([], observations, calibration())
    assert result["players"][0]["distance_m"] is None
    assert result["player_motion"][1]["x_m"] is None
    assert result["player_motion"][1]["valid_position"] is False


def test_unconfirmed_automatic_calibration_disables_metres_but_retains_image_speed():
    times = [0., .02, .04]
    result = analyze_insights(shuttles(times), frames(times), calibration("auto_lines"))
    assert result["quality"]["calibration_reviewed"] is False
    assert result["players"][0]["distance_m"] is None
    assert result["players"][0]["average_speed_m_s"] is None
    assert result["shuttle"]["average_image_speed_px_s"] == pytest.approx(100)


def test_manually_reviewed_automatic_calibration_can_measure_movement():
    result = analyze_insights([], frames([0., .04, .08, .12]), calibration("auto_lines", True))
    assert result["quality"]["calibration_reviewed"] is True
    assert result["players"][0]["distance_m"] == pytest.approx(.12)


def test_image_speed_time_weighted_average_differs_from_arithmetic_speed_average():
    times = [0., .02, .10]
    detections = shuttles(times, [[100,100],[110,100],[130,100]])
    result = analyze_insights(detections, frames(times), calibration())
    assert result["shuttle_motion"][1]["image_speed_px_s"] == pytest.approx(500)
    assert result["shuttle_motion"][2]["image_speed_px_s"] == pytest.approx(250)
    assert result["shuttle"]["average_image_speed_px_s"] == pytest.approx(300)
    assert result["shuttle"]["max_image_speed_px_s"] == pytest.approx(500)
    assert result["shuttle"]["physical_speed_km_h"] is None
    assert all(row["physical_speed_km_h"] is None for row in result["shuttle_motion"])


def test_interpolated_and_low_confidence_shuttles_never_generate_speed():
    times = [index*.02 for index in range(7)]
    detections = shuttles(times)
    detections[2] = Detection(2, .04, 104,100,.9,"interpolated",False)
    detections[4].low_confidence = True
    result = analyze_insights(detections, frames(times), calibration())
    assert [row["frame_index"] for row in result["shuttle_motion"] if row["image_speed_px_s"] is not None] == [1,6]
    assert result["shuttle"]["valid_motion_intervals"] == 2


def test_shuttle_coordinates_outside_source_image_are_excluded():
    times = [0., .02, .04]
    result = analyze_insights(shuttles(times, [[100,100],[1100,100],[102,100]]), frames(times), calibration())
    assert result["shuttle"]["valid_motion_intervals"] == 0
    assert "shuttle_outside_image" in result["shuttle_motion"][1]["quality_flags"]



def test_large_observed_shuttle_jump_is_flagged_but_not_hidden_from_motion_exports():
    times = [0.0, .02]
    result = analyze_insights(shuttles(times, [[100, 100], [300, 100]]), frames(times), calibration())
    interval = result["shuttle_motion"][1]
    assert interval["valid_observation"] is True
    assert interval["image_speed_px_s"] == pytest.approx(10000)
    assert "large_image_jump_review" in interval["quality_flags"]
    assert any("does not distinguish a fast shuttle from a false detection" in warning
               for warning in result["quality"]["warnings"])
    assert result["shuttle"]["max_image_speed_px_s"] == pytest.approx(10000)

def test_raw_identity_fragments_are_not_joined_even_with_manual_labels():
    times = [0., .04, .08, .12]
    observations = frames(times, [[player(1,1)],[player(1,1.04)],[player(8,4)],[player(8,4.04)]])
    review = {"identity_map":{"1":{"player_label":"Alice","side":"far"},"8":{"player_label":"Alice","side":"far"}}}
    result = analyze_insights([], observations, calibration(), review=review)
    assert len(result["players"]) == 2
    assert result["identified_players"][0]["distance_m"] == pytest.approx(.08)
    assert result["identified_players"][0]["valid_motion_time_s"] == pytest.approx(.08)
    assert result["identified_players"][0]["track_ids"] == [1,8]


def test_doubles_spacing_requires_reviewed_identity_and_side_mapping():
    times = [0., .02, .04]
    positions = [[player(1,1,3),player(2,4,3),player(3,1,10),player(4,4,10)] for _ in times]
    observations = frames(times, positions)
    result = analyze_insights([], observations, calibration(), mode="doubles")
    assert result["doubles_pair_spacing"]["status"] == "unavailable"
    review = {"identity_map": {str(i):{"player_label":f"Player {i}","side":"far" if i<3 else "near"} for i in range(1,5)}}
    result = analyze_insights([], observations, calibration(), mode="doubles", review=review)
    assert result["doubles_pair_spacing"]["status"] == "available"
    assert len(result["doubles_pair_spacing"]["frames"]) == 6
    assert all(pair["median_spacing_m"] == pytest.approx(3) for pair in result["doubles_pair_spacing"]["pairs"])


def test_doubles_spacing_excludes_frames_with_a_missing_fourth_player():
    times = [0., .02, .04]
    positions = [[player(1,1,3),player(2,4,3),player(3,1,10),player(4,4,10)] for _ in times]
    positions[1] = positions[1][:3]
    review = {"identity_map": {str(i):{"player_label":f"Player {i}","side":"far" if i<3 else "near"} for i in range(1,5)}}
    result = analyze_insights([], frames(times,positions), calibration(), mode="doubles", review=review)
    assert {row["frame_index"] for row in result["doubles_pair_spacing"]["frames"]} == {0,2}


def test_manual_rally_boundaries_exclude_intervals_crossing_start():
    times = [index*.04 for index in range(6)]
    review = {"rallies":[{"id":"rally1","start_frame":2,"end_frame":5,"notes":"Reviewed segment"}]}
    result = analyze_insights([], frames(times), calibration(), review=review)
    assert result["rallies"][0]["duration_s"] == pytest.approx(.12)
    assert result["rallies"][0]["players"][0]["distance_m"] == pytest.approx(.12)
    assert result["rallies"][0]["boundary_source"] == "manual_review"


def test_without_rally_review_whole_clip_is_explicitly_unsegmented():
    result = analyze_insights([], frames([0.,.04,.08]), calibration())
    assert result["rallies"][0]["id"] == "unsegmented_clip"
    assert "not_automatically_classified" in result["rallies"][0]["boundary_source"]


def test_manual_recovery_target_known_time_and_no_automatic_contact_inference():
    times = [index*.1 for index in range(11)]
    observations = frames(times, [[player(1,1+timestamp)] for timestamp in times])
    review = {"identity_map":{"1":{"player_label":"Alice"}},"events":[{"frame_index":0,"kind":"contact","player_label":"Alice"}],"recovery_targets":[{"player_label":"Alice","contact_frame":0,"target_x_m":2,"target_y_m":3,"radius_m":.11,"until_frame":10}]}
    result = analyze_insights([], observations, calibration(), review=review)
    assert result["recovery"][0]["recovery_time_s"] == pytest.approx(.9)
    assert result["recovery"][0]["reached_frame"] == 9
    assert result["review"]["events"][0]["kind"] == "contact"
    assert analyze_insights([], observations, calibration())["recovery"] == []


def test_recovery_target_does_not_cross_missing_observations():
    times = [index*.1 for index in range(11)]
    observations = frames(times)
    observations[5]["players"] = []
    review = {"identity_map":{"1":{"player_label":"Alice"}},"recovery_targets":[{"player_label":"Alice","contact_frame":0,"target_x_m":2,"target_y_m":3,"radius_m":.11}]}
    result = analyze_insights([], observations, calibration(), review=review)
    assert result["recovery"][0]["status"] == "unavailable"
    assert result["recovery"][0]["recovery_time_s"] is None


def test_outside_court_coordinates_are_not_clamped_into_heatmap():
    times = [0.,.04,.08,.12]
    result = analyze_insights([], frames(times,[[player(1,-.2+timestamp)] for timestamp in times]), calibration())
    summary = result["players"][0]
    assert summary["zones"]["outside_court"] == pytest.approx(.12)
    assert np.sum(summary["heatmap"]["values_s"]) == 0
    assert result["player_motion"][0]["x_m"] == -.2


@pytest.mark.parametrize("review", [
    {"identity_map":{"99":{"player_label":"Alice"}}},
    {"identity_map":{"1":{"player_label":"Alice","side":"left"}}},
    {"schema_version":2},
    {"rallies":[{"id":"one","start_frame":0,"end_frame":2},{"id":"two","start_frame":2,"end_frame":3}]},
    {"events":[{"frame_index":99,"kind":"contact"}]},
    {"unknown":True},
    {"identity_map":{"1":{"player_label":"Alice"}},"recovery_targets":[{"player_label":"Alice","contact_frame":0,"target_x_m":float("nan"),"target_y_m":3}]},
])
def test_malformed_review_rejected(review):
    with pytest.raises(ValueError):
        validate_review(review,frames([0.,.04,.08,.12]))


def test_simultaneous_fragments_cannot_map_to_same_athlete():
    observations = frames([0.,.04],[[player(1,1),player(2,3)],[player(1,1),player(2,3)]])
    review = {"identity_map":{"1":{"player_label":"Alice"},"2":{"player_label":"Alice"}}}
    with pytest.raises(ValueError,match="simultaneous"):
        analyze_insights([],observations,calibration(),review=review)


@pytest.mark.parametrize("times", [[0.,0.],[0.,float("nan")],[.04,0.]])
def test_invalid_source_timestamps_rejected(times):
    with pytest.raises(ValueError,match="timestamp"):
        analyze_insights([],frames(times),calibration())


def test_shuttle_and_player_timestamps_must_align():
    detections = shuttles([0.,.02,.04])
    detections[1].timestamp_s = .021
    with pytest.raises(ValueError,match="align"):
        analyze_insights(detections,frames([0.,.02,.04]),calibration())


def test_inputs_not_mutated_by_analysis():
    observations = frames([0.,.04,.08,.12])
    detections = shuttles([0.,.04,.08,.12])
    original = deepcopy(observations)
    analyze_insights(detections,observations,calibration())
    assert observations == original
    assert detections[1].source == "detected"


def test_export_artifacts_values_images_and_overwrite_guard(project_tmp_path):
    times = [0.,.04,.08,.12]
    insights = analyze_insights(shuttles(times),frames(times),calibration())
    outputs = export_insights(project_tmp_path,insights)
    assert len(outputs) == 7
    assert all((project_tmp_path/name).is_file() and (project_tmp_path/name).stat().st_size > 0 for name in outputs.values())
    stored = json.loads((project_tmp_path/"insights.json").read_text())
    assert stored["players"][0]["distance_m"] == pytest.approx(.12)
    with (project_tmp_path/"shuttle_motion.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows[1]["physical_speed_km_h"] == ""
    assert float(rows[1]["image_speed_px_s"]) == pytest.approx(100)
    with Image.open(project_tmp_path/"heatmaps/track_1.png") as image:
        assert image.size == (440,880)
        assert np.asarray(image).std() > 0
    assert "Physical shuttle speed in km/h is unavailable" in (project_tmp_path/"coaching_review.md").read_text()
    with pytest.raises(ValueError,match="overwrite"):
        export_insights(project_tmp_path,insights)


def test_empty_inputs_export_without_fake_players(project_tmp_path):
    result = analyze_insights([],[],None)
    assert result["players"] == []
    assert result["shuttle"]["physical_speed_km_h"] is None
    assert result["rallies"] == []
    assert len(export_insights(project_tmp_path,result)) == 6

def test_unknown_or_numerically_low_shuttle_confidence_is_excluded_even_if_flag_is_false():
    times = [0., .02, .04, .06]
    detections = shuttles(times)
    detections[1].confidence = None
    detections[2].confidence = .2
    result = analyze_insights(detections, frames(times), calibration())
    assert result["shuttle"]["valid_motion_intervals"] == 0
    assert "shuttle_confidence_unavailable" in result["shuttle_motion"][1]["quality_flags"]
    assert "low_confidence_shuttle" in result["shuttle_motion"][2]["quality_flags"]


@pytest.mark.parametrize("review", [
    {"events":[{"frame_index":0,"kind":"contact","player_label":[]}]},
    {"recovery_targets":[{"player_label":{},"contact_frame":0}]},
    {"schema_version":True},
])
def test_wrong_review_value_types_raise_validation_error_not_typeerror(review):
    with pytest.raises(ValueError):
        validate_review(review,frames([0.,.04,.08,.12]))


@pytest.mark.parametrize("digest", ["short", "g"*64, 4, None])
def test_invalid_review_source_binding_rejected(digest):
    with pytest.raises(ValueError,match="source_video_sha256"):
        validate_review({"source_video_sha256":digest},frames([0.,.04,.08]))


def test_review_source_binding_normalized_and_mismatched_calibration_rejected():
    observations = frames([0.,.04,.08])
    review = {"source_video_sha256":"A"*64,"identity_map":{"1":{"player_label":"Alice"}}}
    assert validate_review(review,observations)["source_video_sha256"] == "a"*64
    cal = calibration()
    cal.metadata["input_video_sha256"] = "b"*64
    with pytest.raises(ValueError,match="SHA256"):
        analyze_insights([],observations,cal,review=review)
    cal.metadata["input_video_sha256"] = "a"*64
    result = analyze_insights([],observations,cal,review=review)
    assert not any("no source_video_sha256" in warning for warning in result["quality"]["warnings"])


def test_only_actual_unbound_review_annotations_emit_source_warning():
    observations = frames([0.,.04,.08])
    result = analyze_insights([],observations,calibration())
    assert not any("source_video_sha256" in warning for warning in result["quality"]["warnings"])
    result = analyze_insights([],observations,calibration(),review={"identity_map":{"1":{"player_label":"Alice"}}})
    assert any("no source_video_sha256" in warning for warning in result["quality"]["warnings"])


def test_doubles_spacing_excludes_implausible_position_jump_frame():
    times = [0.,.02,.04,.06]
    positions = [[player(1,1,3),player(2,4,3),player(3,1,10),player(4,4,10)] for _ in times]
    positions[1][0]["court_position"]["x_m"] = 5
    review = {"identity_map":{str(i):{"player_label":f"Player {i}","side":"far" if i<3 else "near"} for i in range(1,5)}}
    result = analyze_insights([],frames(times,positions),calibration(),mode="doubles",review=review)
    assert 1 not in {row["frame_index"] for row in result["doubles_pair_spacing"]["frames"]}


def test_recovery_contact_on_rejected_jump_cannot_report_zero_seconds():
    times = [0.,.02,.04,.06]
    observations = frames(times,[[player(1,x)] for x in [1,5,1.04,1.06]])
    review = {"identity_map":{"1":{"player_label":"Alice"}},"recovery_targets":[{"player_label":"Alice","contact_frame":1,"target_x_m":5,"target_y_m":3,"radius_m":.1}]}
    result = analyze_insights([],observations,calibration(),review=review)
    assert result["recovery"][0]["status"] == "unavailable"
    assert result["recovery"][0]["recovery_time_s"] is None
