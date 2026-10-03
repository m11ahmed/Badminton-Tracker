"""Boundary tests mock network output and assert only temporal bookkeeping."""
import json

import numpy as np
import pytest
import torch

from badminton_tracker import tracknet


def test_final_window_padding_never_exports_fictitious_frames(monkeypatch, project_tmp_path):
    frames = []
    timestamps = [0.0, 0.04, 0.11, 0.16, 0.24]
    for index, timestamp in enumerate(timestamps):
        rgb = np.zeros((24, 32, 3), dtype=np.uint8)
        rgb[:, :, :] = (10 + index, 40 + index, 80 + index)
        frames.append((index, timestamp, rgb))

    class StubModel:
        def __init__(self):
            self.calls = []

        def __call__(self, inputs):
            self.calls.append(inputs.clone())
            heatmaps = torch.zeros((inputs.shape[0], 4, tracknet.HEIGHT, tracknet.WIDTH))
            heatmaps[:, :, 20:23, 40:43] = 0.9
            return heatmaps

    model = StubModel()
    vendor = project_tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "provenance.json").write_text(json.dumps({"purpose": "test fixture"}), encoding="utf-8")
    weights = project_tmp_path / "mock_weights.pt"
    weights.write_bytes(b"only used to check provenance hashing")
    monkeypatch.setattr(tracknet, "VENDOR_ROOT", vendor)
    monkeypatch.setattr(tracknet, "probe_video", lambda *args, **kwargs: {"width": 32, "height": 24})
    monkeypatch.setattr(tracknet, "iter_rgb_frames", lambda *args, **kwargs: iter(frames))
    monkeypatch.setattr(tracknet, "_load_model", lambda *args: (model, torch.device("cpu"), 4, ""))
    records, metadata = tracknet.track_video(project_tmp_path / "mock.mp4", weights, batch_size=2)
    assert [record.frame_index for record in records] == list(range(5))
    assert [record.timestamp_s for record in records] == timestamps
    assert metadata["padded_frame_count"] == 3
    assert metadata["frames_analyzed"] == 5
    assert len(model.calls) == 1
    inputs = model.calls[0]
    assert inputs.shape == (2, 12, tracknet.HEIGHT, tracknet.WIDTH)
    # Every artificial frame in the final window repeats the last real RGB.
    for frame_channel_start in (0, 3, 6, 9):
        assert inputs[1, frame_channel_start:frame_channel_start + 3, 0, 0].tolist() == pytest.approx([14 / 255, 44 / 255, 84 / 255])
    # No synthetic predictions may escape merely because batch windows are padded.
    assert len(records) == len(frames)
