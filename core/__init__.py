"""NeuroLens+ Core Module - Eye tracking and biomarker extraction."""

from .tracker import EyeTracker
from .calibration import Calibrator
from .mapping import GazeMapper
from .validity import ValidityChecker, InvalidReason
from .logging import FrameLogger, SummaryLogger
from .replay import ReplayEngine
from .utils import (
    compute_velocity,
    compute_ear,
    detect_blink,
    rolling_fps,
    clamp_value,
    compute_rms,
    compute_bcea,
)

__all__ = [
    "EyeTracker",
    "Calibrator",
    "GazeMapper",
    "ValidityChecker",
    "InvalidReason",
    "FrameLogger",
    "SummaryLogger",
    "ReplayEngine",
    "compute_velocity",
    "compute_ear",
    "detect_blink",
    "rolling_fps",
    "clamp_value",
    "compute_rms",
    "compute_bcea",
]
