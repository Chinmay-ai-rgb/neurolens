"""Calibration module for NeuroLens+ eye tracking system."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import time


class CalibrationReason(Enum):
    """Reasons for calibration acceptance/rejection."""
    ACCEPTED = "accepted"
    INSUFFICIENT_RANGE_X = "insufficient_range_x"
    INSUFFICIENT_RANGE_Y = "insufficient_range_y"
    HIGH_ERROR = "high_error"
    DEGENERATE_MAPPING = "degenerate_mapping"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    NO_FACE_DETECTED = "no_face_detected"
    TOO_MANY_BLINKS = "too_many_blinks"
    LOW_SLOPE = "low_slope"


@dataclass
class CalibrationPoint:
    """Data for a single calibration point."""
    target_x_px: float
    target_y_px: float
    target_x_norm: float
    target_y_norm: float
    gaze_x_norm_samples: List[float] = field(default_factory=list)
    gaze_y_norm_samples: List[float] = field(default_factory=list)
    anchor_x_norm_samples: List[float] = field(default_factory=list)
    anchor_y_norm_samples: List[float] = field(default_factory=list)
    valid_samples: int = 0
    total_samples: int = 0
    
    @property
    def gaze_x_norm_mean(self) -> float:
        if not self.gaze_x_norm_samples:
            return np.nan
        return np.nanmean(self.gaze_x_norm_samples)
    
    @property
    def gaze_y_norm_mean(self) -> float:
        if not self.gaze_y_norm_samples:
            return np.nan
        return np.nanmean(self.gaze_y_norm_samples)
    
    @property
    def anchor_x_norm_mean(self) -> float:
        if not self.anchor_x_norm_samples:
            return np.nan
        return np.nanmean(self.anchor_x_norm_samples)
    
    @property
    def anchor_y_norm_mean(self) -> float:
        if not self.anchor_y_norm_samples:
            return np.nan
        return np.nanmean(self.anchor_y_norm_samples)


@dataclass
class CalibrationResult:
    """Results from calibration procedure."""
    accepted: bool = False
    reason: CalibrationReason = CalibrationReason.INSUFFICIENT_SAMPLES
    
    # Baseline anchor (from center point)
    baseline_anchor_x_norm: float = 0.5
    baseline_anchor_y_norm: float = 0.5
    
    # Mapping parameters (linear: px = slope * norm_comp + intercept)
    slope_x: float = 1.0
    intercept_x: float = 0.0
    slope_y: float = 1.0
    intercept_y: float = 0.0
    
    # Quality metrics
    calibration_error_px: Dict[str, float] = field(default_factory=dict)
    mean_error_px: float = np.nan
    max_error_px: float = np.nan
    r_squared_x: float = np.nan
    r_squared_y: float = np.nan
    
    # Raw data
    points_used: List[str] = field(default_factory=list)
    num_points: int = 0
    
    # Thresholds used
    min_range_threshold: float = 0.01
    max_error_threshold: float = 120.0
    min_slope_threshold: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'accepted': self.accepted,
            'reason': self.reason.value,
            'baseline_anchor_x_norm': self.baseline_anchor_x_norm,
            'baseline_anchor_y_norm': self.baseline_anchor_y_norm,
            'slope_x': self.slope_x,
            'intercept_x': self.intercept_x,
            'slope_y': self.slope_y,
            'intercept_y': self.intercept_y,
            'calibration_error_px': self.calibration_error_px,
            'mean_error_px': self.mean_error_px,
            'max_error_px': self.max_error_px,
            'r_squared_x': self.r_squared_x,
            'r_squared_y': self.r_squared_y,
            'points_used': self.points_used,
            'num_points': self.num_points,
            'min_range_threshold': self.min_range_threshold,
            'max_error_threshold': self.max_error_threshold,
            'min_slope_threshold': self.min_slope_threshold,
        }
    
    def save(self, filepath: str):
        """Save calibration result to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'CalibrationResult':
        """Load calibration result from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        result = cls()
        result.accepted = data.get('accepted', False)
        result.reason = CalibrationReason(data.get('reason', 'insufficient_samples'))
        result.baseline_anchor_x_norm = data.get('baseline_anchor_x_norm', 0.5)
        result.baseline_anchor_y_norm = data.get('baseline_anchor_y_norm', 0.5)
        result.slope_x = data.get('slope_x', 1.0)
        result.intercept_x = data.get('intercept_x', 0.0)
        result.slope_y = data.get('slope_y', 1.0)
        result.intercept_y = data.get('intercept_y', 0.0)
        result.calibration_error_px = data.get('calibration_error_px', {})
        result.mean_error_px = data.get('mean_error_px', np.nan)
        result.max_error_px = data.get('max_error_px', np.nan)
        result.r_squared_x = data.get('r_squared_x', np.nan)
        result.r_squared_y = data.get('r_squared_y', np.nan)
        result.points_used = data.get('points_used', [])
        result.num_points = data.get('num_points', 0)
        
        return result


class Calibrator:
    """
    Calibration system for gaze mapping.
    
    Supports:
    - Center-only baseline (always performed)
    - 3-point horizontal calibration (center, left, right)
    - 9-point full calibration
    """
    
    # Standard calibration point positions (normalized)
    POINT_POSITIONS = {
        'center': (0.5, 0.5),
        'left': (0.2, 0.5),
        'right': (0.8, 0.5),
        'top': (0.5, 0.2),
        'bottom': (0.5, 0.8),
        'top_left': (0.2, 0.2),
        'top_right': (0.8, 0.2),
        'bottom_left': (0.2, 0.8),
        'bottom_right': (0.8, 0.8),
    }
    
    # Calibration modes
    MODE_CENTER_ONLY = ['center']
    MODE_3_POINT = ['center', 'left', 'right']
    MODE_5_POINT = ['center', 'left', 'right', 'top', 'bottom']
    MODE_9_POINT = ['center', 'left', 'right', 'top', 'bottom',
                    'top_left', 'top_right', 'bottom_left', 'bottom_right']
    
    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        min_range_threshold: float = 0.01,
        max_error_threshold: float = 120.0,
        min_slope_threshold: float = 100.0,
        min_samples_per_point: int = 10
    ):
        """
        Initialize calibrator.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            min_range_threshold: Minimum gaze range between points (normalized)
            max_error_threshold: Maximum acceptable error per point (pixels)
            min_slope_threshold: Minimum acceptable slope magnitude
            min_samples_per_point: Minimum valid samples required per point
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.min_range_threshold = min_range_threshold
        self.max_error_threshold = max_error_threshold
        self.min_slope_threshold = min_slope_threshold
        self.min_samples_per_point = min_samples_per_point
        
        # Calibration data
        self.points: Dict[str, CalibrationPoint] = {}
        self.current_point: Optional[str] = None
        self.result: Optional[CalibrationResult] = None
    
    def get_point_positions(self, mode: str = '3_point') -> List[Tuple[str, float, float]]:
        """
        Get calibration point positions for a given mode.
        
        Args:
            mode: 'center', '3_point', '5_point', or '9_point'
        
        Returns:
            List of (name, x_norm, y_norm) tuples
        """
        if mode == 'center':
            point_names = self.MODE_CENTER_ONLY
        elif mode == '3_point':
            point_names = self.MODE_3_POINT
        elif mode == '5_point':
            point_names = self.MODE_5_POINT
        elif mode == '9_point':
            point_names = self.MODE_9_POINT
        else:
            point_names = self.MODE_3_POINT
        
        return [(name, *self.POINT_POSITIONS[name]) for name in point_names]
    
    def start_calibration(self, mode: str = '3_point'):
        """
        Start a new calibration session.
        
        Args:
            mode: Calibration mode ('center', '3_point', '5_point', '9_point')
        """
        self.points = {}
        self.result = None
        
        positions = self.get_point_positions(mode)
        for name, x_norm, y_norm in positions:
            self.points[name] = CalibrationPoint(
                target_x_px=x_norm * self.screen_width,
                target_y_px=y_norm * self.screen_height,
                target_x_norm=x_norm,
                target_y_norm=y_norm
            )
        
        self.current_point = None
    
    def set_current_point(self, point_name: str):
        """Set the current calibration point being collected."""
        if point_name in self.points:
            self.current_point = point_name
    
    def add_sample(
        self,
        gaze_x_norm: float,
        gaze_y_norm: float,
        anchor_x_norm: float,
        anchor_y_norm: float,
        valid: bool = True
    ):
        """
        Add a gaze sample to the current calibration point.
        
        Args:
            gaze_x_norm: Gaze x position (normalized)
            gaze_y_norm: Gaze y position (normalized)
            anchor_x_norm: Anchor x position (normalized)
            anchor_y_norm: Anchor y position (normalized)
            valid: Whether this is a valid sample
        """
        if self.current_point is None or self.current_point not in self.points:
            return
        
        point = self.points[self.current_point]
        point.total_samples += 1
        
        if valid and not np.isnan(gaze_x_norm) and not np.isnan(gaze_y_norm):
            point.gaze_x_norm_samples.append(gaze_x_norm)
            point.gaze_y_norm_samples.append(gaze_y_norm)
            point.anchor_x_norm_samples.append(anchor_x_norm)
            point.anchor_y_norm_samples.append(anchor_y_norm)
            point.valid_samples += 1
    
    def compute_calibration(self) -> CalibrationResult:
        """
        Compute calibration parameters from collected samples.
        
        Returns:
            CalibrationResult with mapping parameters and quality metrics
        """
        result = CalibrationResult()
        result.min_range_threshold = self.min_range_threshold
        result.max_error_threshold = self.max_error_threshold
        result.min_slope_threshold = self.min_slope_threshold
        
        # Check we have center point
        if 'center' not in self.points:
            result.reason = CalibrationReason.INSUFFICIENT_SAMPLES
            return result
        
        center = self.points['center']
        
        # Check center has enough samples
        if center.valid_samples < self.min_samples_per_point:
            result.reason = CalibrationReason.INSUFFICIENT_SAMPLES
            return result
        
        # Set baseline anchor from center point
        result.baseline_anchor_x_norm = center.anchor_x_norm_mean
        result.baseline_anchor_y_norm = center.anchor_y_norm_mean
        
        # Collect valid points
        valid_points = []
        for name, point in self.points.items():
            if point.valid_samples >= self.min_samples_per_point:
                valid_points.append((name, point))
                result.points_used.append(name)
        
        result.num_points = len(valid_points)
        
        if result.num_points < 1:
            result.reason = CalibrationReason.INSUFFICIENT_SAMPLES
            return result
        
        # Compute compensated gaze for each point
        target_x_px = []
        target_y_px = []
        gaze_x_comp = []
        gaze_y_comp = []
        
        for name, point in valid_points:
            # Compensate gaze using baseline anchor
            comp_x = point.gaze_x_norm_mean - (point.anchor_x_norm_mean - result.baseline_anchor_x_norm)
            comp_y = point.gaze_y_norm_mean - (point.anchor_y_norm_mean - result.baseline_anchor_y_norm)
            
            target_x_px.append(point.target_x_px)
            target_y_px.append(point.target_y_px)
            gaze_x_comp.append(comp_x)
            gaze_y_comp.append(comp_y)
        
        target_x_px = np.array(target_x_px)
        target_y_px = np.array(target_y_px)
        gaze_x_comp = np.array(gaze_x_comp)
        gaze_y_comp = np.array(gaze_y_comp)
        
        # Check for degeneracy in X
        if result.num_points >= 2:
            x_range = np.max(gaze_x_comp) - np.min(gaze_x_comp)
            if x_range < self.min_range_threshold:
                result.reason = CalibrationReason.INSUFFICIENT_RANGE_X
                # Still compute with default mapping
                result.slope_x = self.screen_width
                result.intercept_x = 0
            else:
                # Linear regression for X mapping
                result.slope_x, result.intercept_x, result.r_squared_x = self._linear_fit(
                    gaze_x_comp, target_x_px
                )
        else:
            # Single point - use screen-width scaling centered on that point
            result.slope_x = self.screen_width
            result.intercept_x = target_x_px[0] - gaze_x_comp[0] * self.screen_width
        
        # Check for degeneracy in Y
        if result.num_points >= 2:
            y_range = np.max(gaze_y_comp) - np.min(gaze_y_comp)
            if y_range < self.min_range_threshold:
                result.reason = CalibrationReason.INSUFFICIENT_RANGE_Y
                result.slope_y = self.screen_height
                result.intercept_y = 0
            else:
                # Linear regression for Y mapping
                result.slope_y, result.intercept_y, result.r_squared_y = self._linear_fit(
                    gaze_y_comp, target_y_px
                )
        else:
            result.slope_y = self.screen_height
            result.intercept_y = target_y_px[0] - gaze_y_comp[0] * self.screen_height
        
        # Check slope magnitude
        if abs(result.slope_x) < self.min_slope_threshold:
            result.reason = CalibrationReason.LOW_SLOPE
        
        # Compute calibration errors
        errors = []
        for name, point in valid_points:
            comp_x = point.gaze_x_norm_mean - (point.anchor_x_norm_mean - result.baseline_anchor_x_norm)
            comp_y = point.gaze_y_norm_mean - (point.anchor_y_norm_mean - result.baseline_anchor_y_norm)
            
            mapped_x = result.slope_x * comp_x + result.intercept_x
            mapped_y = result.slope_y * comp_y + result.intercept_y
            
            error = np.sqrt((mapped_x - point.target_x_px)**2 + (mapped_y - point.target_y_px)**2)
            result.calibration_error_px[name] = error
            errors.append(error)
        
        result.mean_error_px = np.mean(errors)
        result.max_error_px = np.max(errors)
        
        # Check error threshold
        if result.max_error_px > self.max_error_threshold:
            result.reason = CalibrationReason.HIGH_ERROR
        
        # Accept if no rejection reason was set
        if result.reason == CalibrationReason.INSUFFICIENT_SAMPLES and result.num_points >= 1:
            result.accepted = True
            result.reason = CalibrationReason.ACCEPTED
        elif result.reason in [CalibrationReason.INSUFFICIENT_RANGE_X, 
                               CalibrationReason.INSUFFICIENT_RANGE_Y,
                               CalibrationReason.HIGH_ERROR,
                               CalibrationReason.LOW_SLOPE]:
            # These are warnings but we can still use the calibration
            # Only reject if truly degenerate
            if result.num_points >= 3 and result.mean_error_px < self.max_error_threshold * 2:
                result.accepted = True
        
        self.result = result
        return result
    
    def _linear_fit(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Perform linear regression.
        
        Returns:
            Tuple of (slope, intercept, r_squared)
        """
        if len(x) < 2:
            return (1.0, 0.0, np.nan)
        
        # Remove NaN values
        valid = ~(np.isnan(x) | np.isnan(y))
        x = x[valid]
        y = y[valid]
        
        if len(x) < 2:
            return (1.0, 0.0, np.nan)
        
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x**2)
        
        denom = n * sum_x2 - sum_x**2
        if abs(denom) < 1e-10:
            return (1.0, 0.0, np.nan)
        
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        
        # R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        
        if ss_tot < 1e-10:
            r_squared = 1.0 if ss_res < 1e-10 else 0.0
        else:
            r_squared = 1 - ss_res / ss_tot
        
        return (slope, intercept, r_squared)
    
    def get_quality_score(self) -> float:
        """
        Get overall calibration quality score (0-1).
        """
        if self.result is None:
            return 0.0
        
        if not self.result.accepted:
            return 0.2  # Low score for rejected calibration
        
        # Score based on error and R-squared
        error_score = max(0, 1 - self.result.mean_error_px / self.max_error_threshold)
        
        r2_x = self.result.r_squared_x if not np.isnan(self.result.r_squared_x) else 0.5
        r2_y = self.result.r_squared_y if not np.isnan(self.result.r_squared_y) else 0.5
        r2_score = (r2_x + r2_y) / 2
        
        # Weight error more heavily
        quality = 0.7 * error_score + 0.3 * r2_score
        
        return max(0.0, min(1.0, quality))
