"""Pretrained TrackNetV3 tracking module, with explicit timing and missing values.

This milestone uses the authors' TrackNet module only. InpaintNet trajectory
rectification and their overlapping sequence ensemble are not included.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .types import Detection
from .video import iter_rgb_frames, probe_video

WIDTH = 512
HEIGHT = 288
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "tracknetv3"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_heatmap(
    heatmap: np.ndarray,
    source_width: int,
    source_height: int,
    threshold: float = 0.5,
) -> tuple[float | None, float | None, float]:
    """Use the upstream largest bounding rectangle rule; empty means missing.

    The score is the peak of the selected region, an uncalibrated model score.
    Integer truncation and scale order match the official prediction code.
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2 or heatmap.size == 0:
        raise ValueError("A heatmap must be a nonempty two dimensional array.")
    if not np.isfinite(heatmap).all():
        raise ValueError("Model produced a nonfinite heatmap.")
    if not 0 <= threshold <= 1:
        raise ValueError("Heatmap threshold must be between zero and one.")
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Source dimensions must be positive.")
    mask = (heatmap > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, float(heatmap.max())
    rectangles = [cv2.boundingRect(contour) for contour in contours]
    x, y, width, height = max(rectangles, key=lambda rect: rect[2] * rect[3])
    score = float(heatmap[y:y + height, x:x + width].max())
    model_x = int(x + width / 2)
    model_y = int(y + height / 2)
    source_x = int(model_x * source_width / heatmap.shape[1])
    source_y = int(model_y * source_height / heatmap.shape[0])
    return float(source_x), float(source_y), score


def _resized_rgb(rgb: np.ndarray) -> np.ndarray:
    # PIL's RGB default resize is bicubic, matching upstream dataset.py.
    return np.asarray(Image.fromarray(rgb).resize((WIDTH, HEIGHT)), dtype=np.uint8)


def estimate_background(
    video_path: Path,
    *,
    sample_count: int,
    max_frames: int | None = None,
    fps_hint: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministic bounded reservoir; the median is taken at model resolution.

    Upstream computes the median at source resolution and then resizes it. This
    implementation resizes first to keep background memory independent of video
    resolution and duration. That preprocessing difference is recorded in output.
    """
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or not 1 <= sample_count <= 512:
        raise ValueError("Background samples must be between 1 and 512.")
    reservoir: list[np.ndarray] = []
    generator = np.random.default_rng(0)
    decoded_count = 0
    for index, _, rgb in iter_rgb_frames(video_path, max_frames=max_frames, fps_hint=fps_hint):
        decoded_count += 1
        if len(reservoir) < sample_count:
            reservoir.append(_resized_rgb(rgb))
        else:
            replace = int(generator.integers(0, index + 1))
            if replace < sample_count:
                reservoir[replace] = _resized_rgb(rgb)
    if not reservoir:
        raise ValueError("The input video has no decodable frames.")
    background = np.median(np.stack(reservoir), axis=0).astype(np.uint8)
    return background, {
        "sampling": "deterministic reservoir sampling across analyzed frames",
        "sample_seed": 0,
        "samples_requested": sample_count,
        "samples_used": len(reservoir),
        "frames_decoded_for_background": decoded_count,
        "median_resolution": [WIDTH, HEIGHT],
        "upstream_preprocessing_difference": "frames resized before median instead of source resolution median then resize",
    }


def _load_model(weights_path: Path, device: str):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is missing. Install the project's requirements first.") from error
    if not weights_path.is_file():
        raise FileNotFoundError(f"TrackNet checkpoint does not exist: {weights_path}. Run scripts/download_models.py first.")
    if device == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected = device
    try:
        torch_device = torch.device(selected)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"Invalid PyTorch device: {device}") from error
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu or install CUDA enabled PyTorch.")
    if torch_device.type not in {"cuda", "cpu"}:
        raise ValueError("This Windows milestone supports cpu, cuda, or auto devices.")
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError(
            "Could not safely load the TrackNet checkpoint with weights_only=True. "
            "Use the official TrackNet_best.pt checkpoint or a compatible tensor and primitive dictionary; "
            "unrestricted pickle loading is not enabled. " + str(error)
        ) from error
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("param_dict"), dict):
        raise ValueError("Checkpoint must contain the official param_dict and model state dictionary.")
    parameters = checkpoint["param_dict"]
    sequence_length = parameters.get("seq_len")
    background_mode = parameters.get("bg_mode", "")
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or not 1 <= sequence_length <= 64:
        raise ValueError("Checkpoint param_dict.seq_len must be an integer between 1 and 64.")
    if background_mode not in {"", "concat"}:
        raise ValueError(
            f"Checkpoint background mode {background_mode!r} is unsupported. "
            "Use the official concat checkpoint; subtraction checkpoint preprocessing is not implemented."
        )
    model_name = parameters.get("model_name", "TrackNet")
    if model_name != "TrackNet":
        raise ValueError(f"Expected a TrackNet checkpoint, received {model_name!r}.")
    if not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Checkpoint is missing the model tensor state dictionary.")
    model_file = VENDOR_ROOT / "model.py"
    specification = importlib.util.spec_from_file_location("badminton_vendor_tracknetv3", model_file)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"The vendored TrackNet implementation is missing: {model_file}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    input_channels = 3 * (sequence_length + (background_mode == "concat"))
    model = module.TrackNet(in_dim=input_channels, out_dim=sequence_length)
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as error:
        raise ValueError("Checkpoint tensors do not match the authors' TrackNet architecture: " + str(error)) from error
    model.to(torch_device).eval()
    return model, torch_device, sequence_length, background_mode


def _sequence_tensor(frames: list[np.ndarray], background: np.ndarray | None) -> np.ndarray:
    channels = [np.moveaxis(frame, -1, 0) for frame in frames]
    if background is not None:
        channels.insert(0, np.moveaxis(background, -1, 0))
    return np.concatenate(channels, axis=0).astype(np.float32) / 255.0


def track_video(
    video_path: Path,
    weights_path: Path,
    *,
    device: str = "auto",
    batch_size: int = 1,
    max_frames: int | None = None,
    confidence_threshold: float = 0.5,
    heatmap_threshold: float = 0.5,
    background_samples: int = 32,
    fps_hint: float | None = None,
) -> tuple[list[Detection], dict[str, Any]]:
    """Track every analyzed frame using nonoverlapping checkpoint length windows.

    Only the incomplete final window is padded by repeating its last frame.
    Padded predictions are discarded so frame indexes and timestamps stay aligned.
    Scores below confidence_threshold are retained with low_confidence=True;
    an empty thresholded heatmap produces null coordinates, never a fake origin.
    """
    import torch

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
        raise ValueError("Batch size must be between 1 and 32; use 1 for limited CPU memory.")
    if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0):
        raise ValueError("Max frames must be positive when provided.")
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("Confidence threshold must be between zero and one.")
    if not math.isfinite(heatmap_threshold) or not 0 <= heatmap_threshold <= 1:
        raise ValueError("Heatmap threshold must be between zero and one.")
    if isinstance(background_samples, bool) or not isinstance(background_samples, int) or not 1 <= background_samples <= 512:
        raise ValueError("Background samples must be between 1 and 512.")
    video_path, weights_path = Path(video_path), Path(weights_path)
    metadata = dict(probe_video(video_path, fps_hint=fps_hint))
    model, torch_device, sequence_length, background_mode = _load_model(weights_path, device)
    start = time.perf_counter()
    background = None
    background_metadata = {"sampling": "not used for checkpoint with no background channels"}
    if background_mode:
        print(f"Estimating background from at most {background_samples} samples...", flush=True)
        background, background_metadata = estimate_background(
            video_path, sample_count=background_samples, max_frames=max_frames, fps_hint=fps_hint,
        )
    detections: list[Detection] = []
    batch_inputs: list[np.ndarray] = []
    batch_records: list[list[tuple[int, float, int, int]]] = []
    sequence_frames: list[np.ndarray] = []
    sequence_records: list[tuple[int, float, int, int]] = []
    padded_frames = 0
    last_progress_count = 0

    def infer_batch() -> None:
        nonlocal last_progress_count
        if not batch_inputs:
            return
        inputs = torch.from_numpy(np.stack(batch_inputs)).to(torch_device)
        with torch.inference_mode():
            heatmaps = model(inputs).detach().cpu().numpy()
        if heatmaps.shape != (len(batch_inputs), sequence_length, HEIGHT, WIDTH):
            raise RuntimeError(f"Unexpected TrackNet output shape: {heatmaps.shape}")
        for heatmap_sequence, records in zip(heatmaps, batch_records):
            for heatmap, (frame_index, timestamp_s, source_width, source_height) in zip(heatmap_sequence, records):
                x, y, score = decode_heatmap(heatmap, source_width, source_height, heatmap_threshold)
                missing = x is None or y is None
                detections.append(Detection(
                    frame_index=frame_index, timestamp_s=timestamp_s, x=x, y=y,
                    confidence=score, source="missing" if missing else "detected",
                    low_confidence=missing or score < confidence_threshold,
                ))
        if last_progress_count == 0 or len(detections) - last_progress_count >= 128:
            print(f"Tracked {len(detections)} frames on {torch_device}...", flush=True)
            last_progress_count = len(detections)
        batch_inputs.clear()
        batch_records.clear()

    def queue_sequence() -> None:
        batch_inputs.append(_sequence_tensor(sequence_frames, background))
        batch_records.append(sequence_records.copy())
        sequence_frames.clear()
        sequence_records.clear()
        if len(batch_inputs) >= batch_size:
            infer_batch()

    for frame_index, timestamp_s, rgb in iter_rgb_frames(video_path, max_frames=max_frames, fps_hint=fps_hint):
        source_height, source_width = rgb.shape[:2]
        if (source_width, source_height) != (metadata["width"], metadata["height"]):
            raise ValueError("Frame dimensions change inside this video; split it into fixed camera clips first.")
        sequence_frames.append(_resized_rgb(rgb))
        sequence_records.append((frame_index, timestamp_s, source_width, source_height))
        if len(sequence_frames) == sequence_length:
            queue_sequence()
    if sequence_frames:
        padded_frames = sequence_length - len(sequence_frames)
        sequence_frames.extend([sequence_frames[-1]] * padded_frames)
        queue_sequence()
    infer_batch()
    if not detections:
        raise ValueError("The input video has no decodable frames.")
    elapsed = time.perf_counter() - start
    provenance = json.loads((VENDOR_ROOT / "provenance.json").read_text(encoding="utf-8"))
    metadata.update({
        "model": "TrackNetV3 authors' TrackNet trajectory prediction module",
        "device": str(torch_device),
        "checkpoint": str(weights_path.resolve()),
        "checkpoint_sha256": file_sha256(weights_path),
        "upstream": provenance,
        "sequence_length": sequence_length,
        "background_mode": background_mode,
        "background": background_metadata,
        "model_input_size": [WIDTH, HEIGHT],
        "inference_mode": "nonoverlap",
        "inpaintnet_enabled": False,
        "overlapping_ensemble_enabled": False,
        "final_window_padding": "repeat last frame; padded predictions discarded",
        "padded_frame_count": padded_frames,
        "heatmap_threshold": heatmap_threshold,
        "confidence_threshold": confidence_threshold,
        "confidence_meaning": "selected region peak heatmap score; uncalibrated, not a probability",
        "coordinate_method": "largest contour bounding rectangle center with upstream integer truncation and source pixel scaling",
        "frames_analyzed": len(detections),
        "batch_size": batch_size,
        "processing_seconds": elapsed,
        "processing_frames_per_second": len(detections) / elapsed if elapsed else None,
        "warnings": [
            "Nonoverlapping sequence inference omits the authors' overlapping heatmap ensemble.",
            "InpaintNet trajectory rectification is not applied.",
            "Model scores are uncalibrated and detection accuracy has not been measured on this input.",
            "Use one fixed camera rally per clip; camera cuts or pans can invalidate the estimated background.",
        ],
    })
    if background_mode:
        metadata["warnings"].append("Background estimation uses a bounded sample and takes its median after resizing.")
    return detections, metadata