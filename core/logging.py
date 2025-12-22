"""Logging module for NeuroLens+ eye tracking system."""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import platform
import subprocess


@dataclass
class SessionMetadata:
    """Session metadata container."""
    
    session_id: str = ""
    timestamp: str = ""
    device_info: Dict[str, str] = field(default_factory=dict)
    screen_width: int = 1920
    screen_height: int = 1080
    config: Dict[str, Any] = field(default_factory=dict)
    git_commit: str = ""
    
    # Calibration metrics
    calibration_accepted: bool = False
    calibration_reason: str = ""
    calibration_error_mean_px: float = 0.0
    calibration_error_max_px: float = 0.0
    
    # Thresholds used
    thresholds: Dict[str, float] = field(default_factory=dict)
    
    # FPS stats
    fps_mean: float = 0.0
    fps_median: float = 0.0
    fps_min: float = 0.0
    fps_max: float = 0.0
    
    # Valid rates per task
    valid_rates: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'session_id': self.session_id,
            'timestamp': self.timestamp,
            'device_info': self.device_info,
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'config': self.config,
            'git_commit': self.git_commit,
            'calibration_accepted': self.calibration_accepted,
            'calibration_reason': self.calibration_reason,
            'calibration_error_mean_px': self.calibration_error_mean_px,
            'calibration_error_max_px': self.calibration_error_max_px,
            'thresholds': self.thresholds,
            'fps_mean': self.fps_mean,
            'fps_median': self.fps_median,
            'fps_min': self.fps_min,
            'fps_max': self.fps_max,
            'valid_rates': self.valid_rates
        }
    
    def save(self, filepath: str):
        """Save metadata to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def create(
        cls,
        session_id: str,
        screen_width: int,
        screen_height: int,
        config: Optional[Dict[str, Any]] = None
    ) -> 'SessionMetadata':
        """Create new session metadata."""
        meta = cls()
        meta.session_id = session_id
        meta.timestamp = datetime.now().isoformat()
        meta.screen_width = screen_width
        meta.screen_height = screen_height
        meta.config = config or {}
        
        # Get device info
        meta.device_info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'machine': platform.machine(),
            'python_version': platform.python_version()
        }
        
        # Try to get git commit
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                meta.git_commit = result.stdout.strip()
        except Exception:
            meta.git_commit = "unknown"
        
        return meta


# Frame log columns (exact order as specified)
FRAME_LOG_COLUMNS = [
    'session_id',
    'task',
    'trial_id',
    'state',
    't_epoch_s',
    'frame_idx',
    'fps_est',
    'screen_w',
    'screen_h',
    'target_x',
    'target_y',
    'face_present',
    'blink_flag',
    'valid_sample',
    'gaze_x_norm',
    'gaze_y_norm',
    'anchor_x_norm',
    'anchor_y_norm',
    'gaze_x_norm_comp',
    'gaze_y_norm_comp',
    'gaze_x_px_raw',
    'gaze_y_px_raw',
    'gaze_x_px_comp',
    'gaze_y_px_comp',
    'vel_x_px_s',
    'vel_y_px_s',
    'pupil_proxy_left',
    'pupil_proxy_right',
    'pupil_proxy_mean',
    'clamp_flag_x',
    'clamp_flag_y',
    'out_of_range_flag',
    'dropout_gap_s'
]


class FrameLogger:
    """
    Logger for per-frame eye tracking data.
    
    Writes high-frequency frame data to CSV for signal audit/replay.
    """
    
    def __init__(
        self,
        filepath: str,
        session_id: str,
        task_name: str
    ):
        """
        Initialize frame logger.
        
        Args:
            filepath: Path to output CSV file
            session_id: Session identifier
            task_name: Name of the task
        """
        self.filepath = filepath
        self.session_id = session_id
        self.task_name = task_name
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Open file and write header
        self.file = open(filepath, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=FRAME_LOG_COLUMNS)
        self.writer.writeheader()
        
        self.frame_count = 0
    
    def log_frame(
        self,
        trial_id: int,
        state: str,
        target_x: float,
        target_y: float,
        frame_data: Dict[str, Any]
    ):
        """
        Log a single frame.
        
        Args:
            trial_id: Current trial ID
            state: Current state (e.g., 'FORE', 'JUMP', 'POST')
            target_x: Target x position in pixels
            target_y: Target y position in pixels
            frame_data: Dictionary with frame tracking data
        """
        row = {
            'session_id': self.session_id,
            'task': self.task_name,
            'trial_id': trial_id,
            'state': state,
            'target_x': target_x,
            'target_y': target_y
        }
        
        # Add frame data
        for col in FRAME_LOG_COLUMNS:
            if col not in row:
                row[col] = frame_data.get(col, '')
        
        self.writer.writerow(row)
        self.frame_count += 1
    
    def flush(self):
        """Flush buffer to disk."""
        self.file.flush()
    
    def close(self):
        """Close the file."""
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SummaryLogger:
    """
    Logger for per-trial/per-task summary data.
    
    Writes biomarkers and validity information.
    """
    
    def __init__(
        self,
        filepath: str,
        session_id: str,
        task_name: str,
        columns: List[str]
    ):
        """
        Initialize summary logger.
        
        Args:
            filepath: Path to output CSV file
            session_id: Session identifier
            task_name: Name of the task
            columns: List of column names for this task
        """
        self.filepath = filepath
        self.session_id = session_id
        self.task_name = task_name
        self.columns = ['session_id', 'task'] + columns
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Open file and write header
        self.file = open(filepath, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.columns)
        self.writer.writeheader()
        
        self.trial_count = 0
    
    def log_trial(self, trial_data: Dict[str, Any]):
        """
        Log a single trial summary.
        
        Args:
            trial_data: Dictionary with trial biomarkers and validity
        """
        row = {
            'session_id': self.session_id,
            'task': self.task_name
        }
        
        # Add trial data
        for col in self.columns:
            if col not in row:
                row[col] = trial_data.get(col, '')
        
        self.writer.writerow(row)
        self.trial_count += 1
    
    def flush(self):
        """Flush buffer to disk."""
        self.file.flush()
    
    def close(self):
        """Close the file."""
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class DebugLogger:
    """
    Debug logger for detailed diagnostic information.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize debug logger.
        
        Args:
            filepath: Path to output log file
        """
        self.filepath = filepath
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.file = open(filepath, 'w')
        self.log(f"Debug log started at {datetime.now().isoformat()}")
    
    def log(self, message: str, level: str = "INFO"):
        """
        Log a debug message.
        
        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR, DEBUG)
        """
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.file.write(f"[{timestamp}] [{level}] {message}\n")
        self.file.flush()
    
    def info(self, message: str):
        """Log info message."""
        self.log(message, "INFO")
    
    def warning(self, message: str):
        """Log warning message."""
        self.log(message, "WARNING")
    
    def error(self, message: str):
        """Log error message."""
        self.log(message, "ERROR")
    
    def debug(self, message: str):
        """Log debug message."""
        self.log(message, "DEBUG")
    
    def close(self):
        """Close the file."""
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_session_directory(base_path: str, session_id: str) -> str:
    """
    Create session directory structure.
    
    Args:
        base_path: Base data directory
        session_id: Session identifier
    
    Returns:
        Path to session directory
    """
    session_dir = os.path.join(base_path, 'sessions', session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def generate_session_id() -> str:
    """Generate unique session ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# Summary columns for each task type
FIXATION_SUMMARY_COLUMNS = [
    'trial_id',
    'trial_start_s',
    'trial_end_s',
    'target_x',
    'target_y',
    'fixation_stability_rms_px',
    'fixation_bcea_px2',
    'microsaccade_rate_per_min',
    'drift_velocity_px_s',
    'percent_time_on_target',
    'blink_rate_per_min',
    'valid_fraction',
    'valid',
    'invalid_reason',
    'quality_score',
    'calibration_quality',
    'fps_median',
    'clamp_rate'
]

PURSUIT_SUMMARY_COLUMNS = [
    'trial_id',
    'trial_start_s',
    'trial_end_s',
    'direction',
    'target_velocity_px_s',
    'pursuit_gain',
    'pursuit_latency_ms',
    'catch_up_saccade_count',
    'catch_up_saccade_rate_per_s',
    'position_error_mean_px',
    'position_error_rmse_px',
    'phase_lag_ms',
    'valid_fraction',
    'valid',
    'invalid_reason',
    'quality_score',
    'calibration_quality',
    'fps_median',
    'clamp_rate'
]

SACCADE_SUMMARY_COLUMNS = [
    'trial_id',
    'trial_start_s',
    'trial_end_s',
    'target_x',
    'target_y',
    'direction',
    'eccentricity_px',
    'saccade_latency_ms',
    'saccade_duration_ms',
    'peak_velocity_px_s',
    'saccade_amplitude_px',
    'gain',
    'landing_error_px',
    'overshoot_px',
    'undershoot_px',
    'corrective_saccade_count',
    'valid_fraction',
    'valid',
    'invalid_reason',
    'quality_score',
    'calibration_quality',
    'fps_median',
    'clamp_rate'
]

ANTISACCADE_SUMMARY_COLUMNS = [
    'trial_id',
    'trial_start_s',
    'trial_end_s',
    'stimulus_x',
    'stimulus_y',
    'target_x',
    'target_y',
    'direction',
    'antisaccade_latency_ms',
    'direction_error',
    'correction_time_ms',
    'inhibition_success',
    'peak_velocity_px_s',
    'amplitude_px',
    'landing_error_px',
    'valid_fraction',
    'valid',
    'invalid_reason',
    'quality_score',
    'calibration_quality',
    'fps_median',
    'clamp_rate'
]

GRID_SUMMARY_COLUMNS = [
    'trial_id',
    'point_name',
    'trial_start_s',
    'trial_end_s',
    'target_x',
    'target_y',
    'mean_gaze_error_px',
    'rmse_gaze_error_px',
    'dwell_stability_rms_px',
    'valid_fraction',
    'valid',
    'invalid_reason',
    'quality_score',
    'calibration_quality',
    'fps_median',
    'clamp_rate'
]

GRID_SESSION_COLUMNS = [
    'grid_accuracy_mean_px',
    'grid_accuracy_max_px',
    'systematic_bias_x_px',
    'systematic_bias_y_px',
    'gaze_map_linearity_r2',
    'valid_fraction',
    'valid',
    'invalid_reason',
    'quality_score'
]

VISUAL_SEARCH_SUMMARY_COLUMNS = [
    'trial_id',
    'trial_start_s',
    'trial_end_s',
    'search_duration_s',
    'target_found',
    'search_time_ms',
    'blink_count',
    'blink_rate_per_min',
    'blink_duration_mean_ms',
    'blink_duration_std_ms',
    'interblink_interval_mean_s',
    'interblink_interval_std_s',
    'interblink_interval_cv',
    'blink_burstiness',
    'gaze_presence_pct',
    'valid_fraction',
    'valid',
    'invalid_reason',
    'quality_score',
    'calibration_quality',
    'fps_median',
    'clamp_rate'
]
