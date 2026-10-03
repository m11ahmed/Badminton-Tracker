"""Gap filling must preserve observed evidence and use elapsed time."""
import json

import pytest

from badminton_tracker.types import Detection
from badminton_tracker.postprocess import interpolate_short_gaps


def observed(index, timestamp, x, y, low=False):
    return Detection(index, timestamp, x, y, 0.9, "detected", low)


def missing(index, timestamp):
    return Detection(index, timestamp, None, None, 0.2, "missing", True)


def test_detection_export_uses_null_for_missing_and_preserves_raw_score():
    record = missing(0, 0.0).to_dict()
    exported = json.loads(json.dumps(record, allow_nan=False))
    assert exported["x"] is None and exported["y"] is None
    assert exported["raw_x"] is None and exported["raw_y"] is None
    assert exported["raw_confidence"] == 0.2
    assert exported["low_confidence"] is True


def test_unknown_provenance_source_is_rejected():
    with pytest.raises(ValueError, match="source"):
        Detection(0, 0.0, None, None, None, "invented", True)


def test_short_gap_uses_time_not_frame_index_and_keeps_raw_evidence():
    original = [observed(0, 0.0, 0.0, 0.0), missing(1, 0.1), missing(2, 0.3), observed(3, 0.4, 100.0, 80.0)]
    before = [item.to_dict() for item in original]
    result = interpolate_short_gaps(original, max_gap_frames=2)
    assert (result[1].x, result[1].y) == pytest.approx((25.0, 20.0))
    assert (result[2].x, result[2].y) == pytest.approx((75.0, 60.0))
    for item in result[1:3]:
        assert item.source == "interpolated"
        assert item.confidence is None
        assert item.low_confidence is True
        assert item.raw_x is None and item.raw_y is None
        assert item.raw_confidence == 0.2
    assert [item.to_dict() for item in original] == before
    assert result[0].to_dict() == before[0]
    assert result[-1].to_dict() == before[-1]


def test_long_gap_and_unbounded_edges_remain_missing():
    records = [missing(0, 0.0), observed(1, 0.1, 20, 30), missing(2, 0.2), missing(3, 0.3), missing(4, 0.4), observed(5, 0.5, 100, 90), missing(6, 0.6)]
    result = interpolate_short_gaps(records, max_gap_frames=2)
    assert [item.to_dict() for item in result] == [item.to_dict() for item in records]


@pytest.mark.parametrize("weak_endpoint", [0, 2])
def test_uncertain_endpoint_does_not_fill_gap(weak_endpoint):
    records = [observed(0, 0.0, 20, 30), missing(1, 0.1), observed(2, 0.2, 100, 90)]
    records[weak_endpoint].low_confidence = True
    assert interpolate_short_gaps(records, max_gap_frames=2)[1].source == "missing"


def test_zero_gap_limit_disables_interpolation():
    records = [observed(0, 0.0, 20, 30), missing(1, 0.1), observed(2, 0.2, 100, 90)]
    assert interpolate_short_gaps(records, max_gap_frames=0)[1].source == "missing"

@pytest.mark.parametrize("invalid_time", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_timing_is_rejected_before_interpolation(invalid_time):
    records = [observed(0, 0.0, 20, 30), missing(1, invalid_time), observed(2, 0.2, 100, 90)]
    with pytest.raises(ValueError, match="timestamp"):
        interpolate_short_gaps(records, max_gap_frames=2)


def test_skipped_frame_index_is_rejected_before_interpolation():
    records = [observed(0, 0.0, 20, 30), missing(2, 0.1), observed(3, 0.2, 100, 90)]
    with pytest.raises(ValueError, match="consecutive"):
        interpolate_short_gaps(records, max_gap_frames=2)
