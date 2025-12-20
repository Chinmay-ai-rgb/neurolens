"""Gaze mapping module for NeuroLens+ eye tracking system."""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from .calibration import CalibrationResult


@dataclass
class MappedGaze:
    """Container for mapped gaze data."""
    
    # Compensated normalized gaze
    gaze_x_norm_comp: float = np.nan
    gaze_y_norm_comp: float = np.nan
    
    # Mapped pixel positions
    gaze_x_px_comp: float = np.nan
    gaze_y_px_comp: float = np.nan
    
    # Clamping flags
    clamp_flag_x: int = 0
    clamp_flag_y: int = 0
    out_of_range_flag: int = 0
    
    # Velocity (computed from mapped positions)
    vel_x_px_s: float = np.nan
    vel_y_px_s: float = np.nan


class GazeMapper:
    """
    Maps raw gaze coordinates to screen pixels with head compensation and clamping.
    
    Uses calibration parameters to transform normalized gaze coordinates
    to screen pixel coordinates, with proper handling of out-of-range values.
    """
    
    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        calibration: Optional[CalibrationResult] = None
    ):
        """
        Initialize gaze mapper.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            calibration: Optional calibration result to use
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Default mapping (identity-ish)
        self.slope_x = screen_width
        self.intercept_x = 0.0
        self.slope_y = screen_height
        self.intercept_y = 0.0
        
        # Baseline anchor
        self.baseline_anchor_x_norm = 0.5
        self.baseline_anchor_y_norm = 0.5
        
        # State for velocity computation
        self.last_x_px: Optional[float] = None
        self.last_y_px: Optional[float] = None
        self.last_timestamp: Optional[float] = None
        
        # Apply calibration if provided
        if calibration is not None:
            self.set_calibration(calibration)
    
    def set_calibration(self, calibration: CalibrationResult):
        """
        Set mapping parameters from calibration result.
        
        Args:
            calibration: CalibrationResult with mapping parameters
        """
        self.slope_x = calibration.slope_x
        self.intercept_x = calibration.intercept_x
        self.slope_y = calibration.slope_y
        self.intercept_y = calibration.intercept_y
        self.baseline_anchor_x_norm = calibration.baseline_anchor_x_norm
        self.baseline_anchor_y_norm = calibration.baseline_anchor_y_norm
    
    def set_baseline_anchor(self, x_norm: float, y_norm: float):
        """Set baseline anchor position for head compensation."""
        self.baseline_anchor_x_norm = x_norm
        self.baseline_anchor_y_norm = y_norm
    
    def map_gaze(
        self,
        gaze_x_norm: float,
        gaze_y_norm: float,
        anchor_x_norm: float,
        anchor_y_norm: float,
        timestamp: Optional[float] = None
    ) -> MappedGaze:
        """
        Map normalized gaze to screen pixels with head compensation.
        
        Args:
            gaze_x_norm: Raw gaze x position (normalized 0-1)
            gaze_y_norm: Raw gaze y position (normalized 0-1)
            anchor_x_norm: Current anchor x position (normalized)
            anchor_y_norm: Current anchor y position (normalized)
            timestamp: Optional timestamp for velocity computation
        
        Returns:
            MappedGaze with compensated and clamped coordinates
        """
        result = MappedGaze()
        
        # Handle invalid input
        if np.isnan(gaze_x_norm) or np.isnan(gaze_y_norm):
            return result
        
        # Head motion compensation
        if not np.isnan(anchor_x_norm) and not np.isnan(anchor_y_norm):
            result.gaze_x_norm_comp = gaze_x_norm - (anchor_x_norm - self.baseline_anchor_x_norm)
            result.gaze_y_norm_comp = gaze_y_norm - (anchor_y_norm - self.baseline_anchor_y_norm)
        else:
            result.gaze_x_norm_comp = gaze_x_norm
            result.gaze_y_norm_comp = gaze_y_norm
        
        # Check out of range before mapping
        if result.gaze_x_norm_comp < 0 or result.gaze_x_norm_comp > 1:
            result.out_of_range_flag = 1
        if result.gaze_y_norm_comp < 0 or result.gaze_y_norm_comp > 1:
            result.out_of_range_flag = 1
        
        # Apply linear mapping
        mapped_x = self.slope_x * result.gaze_x_norm_comp + self.intercept_x
        mapped_y = self.slope_y * result.gaze_y_norm_comp + self.intercept_y
        
        # Clamp to screen bounds
        result.gaze_x_px_comp, result.clamp_flag_x = self._clamp(
            mapped_x, 0, self.screen_width - 1
        )
        result.gaze_y_px_comp, result.clamp_flag_y = self._clamp(
            mapped_y, 0, self.screen_height - 1
        )
        
        # Compute velocity if we have previous data
        if (timestamp is not None and 
            self.last_timestamp is not None and
            self.last_x_px is not None and
            self.last_y_px is not None):
            
            dt = timestamp - self.last_timestamp
            if 0.001 < dt < 0.5:  # Valid dt range
                result.vel_x_px_s = (result.gaze_x_px_comp - self.last_x_px) / dt
                result.vel_y_px_s = (result.gaze_y_px_comp - self.last_y_px) / dt
        
        # Update state for next velocity computation
        if not np.isnan(result.gaze_x_px_comp) and not np.isnan(result.gaze_y_px_comp):
            self.last_x_px = result.gaze_x_px_comp
            self.last_y_px = result.gaze_y_px_comp
            self.last_timestamp = timestamp
        
        return result
    
    def map_gaze_simple(
        self,
        gaze_x_norm_comp: float,
        gaze_y_norm_comp: float
    ) -> Tuple[float, float, int, int]:
        """
        Simple mapping without velocity computation.
        
        Args:
            gaze_x_norm_comp: Compensated gaze x (normalized)
            gaze_y_norm_comp: Compensated gaze y (normalized)
        
        Returns:
            Tuple of (x_px, y_px, clamp_flag_x, clamp_flag_y)
        """
        if np.isnan(gaze_x_norm_comp) or np.isnan(gaze_y_norm_comp):
            return (np.nan, np.nan, 0, 0)
        
        # Apply linear mapping
        mapped_x = self.slope_x * gaze_x_norm_comp + self.intercept_x
        mapped_y = self.slope_y * gaze_y_norm_comp + self.intercept_y
        
        # Clamp to screen bounds
        x_px, clamp_x = self._clamp(mapped_x, 0, self.screen_width - 1)
        y_px, clamp_y = self._clamp(mapped_y, 0, self.screen_height - 1)
        
        return (x_px, y_px, clamp_x, clamp_y)
    
    def inverse_map(
        self,
        x_px: float,
        y_px: float
    ) -> Tuple[float, float]:
        """
        Inverse mapping from pixels to normalized coordinates.
        
        Args:
            x_px: X position in pixels
            y_px: Y position in pixels
        
        Returns:
            Tuple of (x_norm_comp, y_norm_comp)
        """
        if abs(self.slope_x) < 1e-10 or abs(self.slope_y) < 1e-10:
            return (np.nan, np.nan)
        
        x_norm = (x_px - self.intercept_x) / self.slope_x
        y_norm = (y_px - self.intercept_y) / self.slope_y
        
        return (x_norm, y_norm)
    
    def _clamp(
        self,
        value: float,
        min_val: float,
        max_val: float
    ) -> Tuple[float, int]:
        """
        Clamp value to range and return flag.
        
        Returns:
            Tuple of (clamped_value, was_clamped)
        """
        if np.isnan(value):
            return (value, 0)
        
        if value < min_val:
            return (min_val, 1)
        elif value > max_val:
            return (max_val, 1)
        else:
            return (value, 0)
    
    def reset_velocity_state(self):
        """Reset velocity computation state."""
        self.last_x_px = None
        self.last_y_px = None
        self.last_timestamp = None
    
    def compute_clamp_rate(
        self,
        clamp_flags_x: list,
        clamp_flags_y: list
    ) -> float:
        """
        Compute clamp rate from lists of clamp flags.
        
        Args:
            clamp_flags_x: List of x clamp flags
            clamp_flags_y: List of y clamp flags
        
        Returns:
            Clamp rate (0-1)
        """
        if not clamp_flags_x and not clamp_flags_y:
            return 0.0
        
        total = len(clamp_flags_x) + len(clamp_flags_y)
        clamped = sum(clamp_flags_x) + sum(clamp_flags_y)
        
        return clamped / total if total > 0 else 0.0


class VelocityComputer:
    """
    Dedicated velocity computation with filtering and spike detection.
    """
    
    def __init__(
        self,
        min_dt: float = 0.001,
        max_dt: float = 0.5,
        max_velocity: float = 10000.0,  # px/s - spike threshold
        smoothing_window: int = 3
    ):
        """
        Initialize velocity computer.
        
        Args:
            min_dt: Minimum valid dt
            max_dt: Maximum valid dt
            max_velocity: Maximum valid velocity (spike threshold)
            smoothing_window: Window size for smoothing
        """
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.max_velocity = max_velocity
        self.smoothing_window = smoothing_window
        
        # State
        self.positions_x: list = []
        self.positions_y: list = []
        self.timestamps: list = []
    
    def add_sample(
        self,
        x_px: float,
        y_px: float,
        timestamp: float
    ):
        """Add a position sample."""
        self.positions_x.append(x_px)
        self.positions_y.append(y_px)
        self.timestamps.append(timestamp)
        
        # Keep only recent samples
        max_samples = self.smoothing_window * 2
        if len(self.positions_x) > max_samples:
            self.positions_x = self.positions_x[-max_samples:]
            self.positions_y = self.positions_y[-max_samples:]
            self.timestamps = self.timestamps[-max_samples:]
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get current velocity estimate.
        
        Returns:
            Tuple of (vel_x, vel_y) in px/s
        """
        if len(self.positions_x) < 2:
            return (np.nan, np.nan)
        
        # Use last two valid samples
        x1, x2 = self.positions_x[-2], self.positions_x[-1]
        y1, y2 = self.positions_y[-2], self.positions_y[-1]
        t1, t2 = self.timestamps[-2], self.timestamps[-1]
        
        if np.isnan(x1) or np.isnan(x2) or np.isnan(y1) or np.isnan(y2):
            return (np.nan, np.nan)
        
        dt = t2 - t1
        if dt < self.min_dt or dt > self.max_dt:
            return (np.nan, np.nan)
        
        vel_x = (x2 - x1) / dt
        vel_y = (y2 - y1) / dt
        
        # Spike detection
        if abs(vel_x) > self.max_velocity:
            vel_x = np.nan
        if abs(vel_y) > self.max_velocity:
            vel_y = np.nan
        
        return (vel_x, vel_y)
    
    def get_velocity_magnitude(self) -> float:
        """Get velocity magnitude in px/s."""
        vel_x, vel_y = self.get_velocity()
        if np.isnan(vel_x) or np.isnan(vel_y):
            return np.nan
        return np.sqrt(vel_x**2 + vel_y**2)
    
    def reset(self):
        """Reset state."""
        self.positions_x = []
        self.positions_y = []
        self.timestamps = []
