"""Shared records for shuttle tracking and exports."""
from dataclasses import dataclass, asdict


@dataclass
class Detection:
    frame_index: int
    timestamp_s: float
    x: float | None
    y: float | None
    confidence: float | None
    source: str
    low_confidence: bool
    raw_x: float | None = None
    raw_y: float | None = None
    raw_confidence: float | None = None

    def __post_init__(self):
        if self.source not in {"detected", "missing", "interpolated"}:
            raise ValueError(f"Unknown detection source: {self.source}")
        if self.source != "interpolated":
            if self.raw_x is None:
                self.raw_x = self.x
            if self.raw_y is None:
                self.raw_y = self.y
            if self.raw_confidence is None:
                self.raw_confidence = self.confidence

    def to_dict(self) -> dict:
        return asdict(self)