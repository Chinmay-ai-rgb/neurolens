"""Validity checking module for NeuroLens+ eye tracking system."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class InvalidReason(Enum):
    """Reasons for trial/sample invalidity."""
    VALID = "valid"
    NO_FACE = "no_face"
    TOO_MANY_BLINKS = "too_many_blinks"
    LOW_FPS = "low_fps"
    CALIBRATION_REJECTED = "calibration_rejected"
    CLAMP_RATE_HIGH = "clamp_rate_high"
    NO_MEANINGFUL_MOVEMENT = "no_meaningful_movement"
    WRONG_DIRECTION = "wrong_direction"
    LATENCY_OUT_OF_RANGE = "latency_out_of_range"
    DURATION_OUT_OF_RANGE = "duration_out_of_range"
    PEAK_VEL_SPIKE = "peak_vel_spike"
    DROPOUT_DURING_CRITICAL_WINDOW = "dropout_during_critical_window"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    TARGET_NOT_REACHED = "target_not_reached"
    ANTICIPATORY_RESPONSE = "anticipatory_response"
    SKIPPED = "skipped"
    OTHER = "other"


@dataclass
class ValidityThresholds:
    """Thresholds for validity checking."""
    
    # Face/blink thresholds
    min_face_presence_rate: float = 0.70
    max_blink_rate: float = 0.30
    
    # FPS thresholds
    min_fps: float = 15.0
    low_fps_fraction_threshold: float = 0.20
    
    # Clamp thresholds (relaxed for webcam-based tracking)
    max_clamp_rate_per_trial: float = 0.25  # Increased from 0.05 for webcam noise
    max_clamp_rate_session: float = 0.10  # Increased from 0.02 for webcam noise
    
    # Saccade thresholds (adjusted for webcam physics)
    saccade_latency_min_ms: float = 50.0
    saccade_latency_max_ms: float = 900.0
    saccade_duration_min_ms: float = 15.0
    saccade_duration_max_ms: float = 400.0
    saccade_min_amplitude_fraction: float = 0.15
    saccade_velocity_threshold: float = 30.0  # px/s
    # Peak velocity threshold: 576px saccade at 30fps = ~17000 px/s expected
    saccade_max_peak_velocity: float = 20000.0  # px/s (increased from 5000 for webcam)
    
    # Antisaccade thresholds
    antisaccade_latency_min_ms: float = 80.0
    antisaccade_latency_max_ms: float = 1000.0
    
    # Fixation thresholds (relaxed for webcam-based tracking)
    fixation_max_deviation_px: float = 300.0  # Increased from 100px for webcam accuracy
    fixation_min_valid_fraction: float = 0.80
    
    # Pursuit thresholds
    pursuit_min_valid_fraction: float = 0.70
    pursuit_min_gain: float = 0.3
    pursuit_max_gain: float = 1.5
    
    # Grid thresholds (relaxed for webcam-based tracking)
    grid_max_error_px: float = 250.0  # Increased from 120px for webcam accuracy
    grid_min_valid_fraction: float = 0.80
    
    # Dropout thresholds
    critical_window_dropout_max_ms: float = 100.0
    
    # Quality targets
    target_valid_trial_rate: float = 0.85
    min_valid_trial_rate: float = 0.70


@dataclass
class TrialValidity:
    """Validity result for a single trial."""
    valid: bool = False
    reason: InvalidReason = InvalidReason.OTHER
    quality_score: float = 0.0
    
    # Component scores
    face_presence_rate: float = 0.0
    blink_rate: float = 0.0
    clamp_rate: float = 0.0
    fps_median: float = 0.0
    valid_fraction: float = 0.0
    
    # Additional info
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'valid': 1 if self.valid else 0,
            'invalid_reason': self.reason.value,
            'quality_score': self.quality_score,
            'face_presence_rate': self.face_presence_rate,
            'blink_rate': self.blink_rate,
            'clamp_rate': self.clamp_rate,
            'fps_median': self.fps_median,
            'valid_fraction': self.valid_fraction,
            **self.details
        }


class ValidityChecker:
    """
    Checks validity of trials and sessions based on quality metrics.
    """
    
    def __init__(self, thresholds: Optional[ValidityThresholds] = None):
        """
        Initialize validity checker.
        
        Args:
            thresholds: Optional custom thresholds
        """
        self.thresholds = thresholds or ValidityThresholds()
    
    def check_sample_validity(
        self,
        face_present: int,
        blink_flag: int,
        dt: float,
        clamp_flag_x: int = 0,
        clamp_flag_y: int = 0
    ) -> Tuple[bool, InvalidReason]:
        """
        Check validity of a single sample.
        
        Returns:
            Tuple of (is_valid, reason)
        """
        if face_present == 0:
            return (False, InvalidReason.NO_FACE)
        
        if blink_flag == 1:
            return (False, InvalidReason.TOO_MANY_BLINKS)
        
        if dt < 0.001 or dt > 0.5:
            return (False, InvalidReason.LOW_FPS)
        
        return (True, InvalidReason.VALID)
    
    def check_trial_validity_generic(
        self,
        face_present_flags: List[int],
        blink_flags: List[int],
        fps_values: List[float],
        clamp_flags_x: List[int],
        clamp_flags_y: List[int],
        valid_sample_flags: List[int]
    ) -> TrialValidity:
        """
        Check generic trial validity based on quality metrics.
        
        Args:
            face_present_flags: List of face detection flags
            blink_flags: List of blink flags
            fps_values: List of FPS estimates
            clamp_flags_x: List of x clamp flags
            clamp_flags_y: List of y clamp flags
            valid_sample_flags: List of valid sample flags
        
        Returns:
            TrialValidity result
        """
        result = TrialValidity()
        
        n_samples = len(face_present_flags)
        if n_samples == 0:
            result.reason = InvalidReason.INSUFFICIENT_SAMPLES
            return result
        
        # Compute metrics
        result.face_presence_rate = sum(face_present_flags) / n_samples
        result.blink_rate = sum(blink_flags) / n_samples
        result.clamp_rate = (sum(clamp_flags_x) + sum(clamp_flags_y)) / (2 * n_samples)
        result.fps_median = np.nanmedian(fps_values) if fps_values else 0.0
        result.valid_fraction = sum(valid_sample_flags) / n_samples
        
        # Check thresholds
        if result.face_presence_rate < self.thresholds.min_face_presence_rate:
            result.reason = InvalidReason.NO_FACE
            result.valid = False
        elif result.blink_rate > self.thresholds.max_blink_rate:
            result.reason = InvalidReason.TOO_MANY_BLINKS
            result.valid = False
        elif result.clamp_rate > self.thresholds.max_clamp_rate_per_trial:
            result.reason = InvalidReason.CLAMP_RATE_HIGH
            result.valid = False
        elif result.fps_median < self.thresholds.min_fps:
            result.reason = InvalidReason.LOW_FPS
            result.valid = False
        else:
            result.valid = True
            result.reason = InvalidReason.VALID
        
        # Compute quality score
        result.quality_score = self._compute_quality_score(result)
        
        return result
    
    def check_saccade_validity(
        self,
        latency_ms: float,
        duration_ms: float,
        amplitude_px: float,
        eccentricity_px: float,
        peak_velocity: float,
        direction_correct: bool,
        dropout_in_critical_window: bool,
        generic_validity: TrialValidity
    ) -> TrialValidity:
        """
        Check saccade trial validity.
        
        Args:
            latency_ms: Saccade latency in ms
            duration_ms: Saccade duration in ms
            amplitude_px: Saccade amplitude in pixels
            eccentricity_px: Target eccentricity in pixels
            peak_velocity: Peak velocity in px/s
            direction_correct: Whether saccade went in correct direction
            dropout_in_critical_window: Whether face was lost in critical window
            generic_validity: Generic validity result
        
        Returns:
            TrialValidity result
        """
        result = TrialValidity(
            face_presence_rate=generic_validity.face_presence_rate,
            blink_rate=generic_validity.blink_rate,
            clamp_rate=generic_validity.clamp_rate,
            fps_median=generic_validity.fps_median,
            valid_fraction=generic_validity.valid_fraction
        )
        
        # Start with generic validity
        if not generic_validity.valid:
            result.valid = False
            result.reason = generic_validity.reason
            result.quality_score = generic_validity.quality_score
            return result
        
        # Check saccade-specific criteria
        if dropout_in_critical_window:
            result.reason = InvalidReason.DROPOUT_DURING_CRITICAL_WINDOW
            result.valid = False
        elif not direction_correct:
            result.reason = InvalidReason.WRONG_DIRECTION
            result.valid = False
        elif latency_ms < self.thresholds.saccade_latency_min_ms:
            result.reason = InvalidReason.ANTICIPATORY_RESPONSE
            result.valid = False
        elif latency_ms > self.thresholds.saccade_latency_max_ms:
            result.reason = InvalidReason.LATENCY_OUT_OF_RANGE
            result.valid = False
        elif duration_ms < self.thresholds.saccade_duration_min_ms:
            result.reason = InvalidReason.DURATION_OUT_OF_RANGE
            result.valid = False
        elif duration_ms > self.thresholds.saccade_duration_max_ms:
            result.reason = InvalidReason.DURATION_OUT_OF_RANGE
            result.valid = False
        elif abs(amplitude_px) < self.thresholds.saccade_min_amplitude_fraction * abs(eccentricity_px):
            result.reason = InvalidReason.NO_MEANINGFUL_MOVEMENT
            result.valid = False
        elif peak_velocity > self.thresholds.saccade_max_peak_velocity:
            result.reason = InvalidReason.PEAK_VEL_SPIKE
            result.valid = False
        else:
            result.valid = True
            result.reason = InvalidReason.VALID
        
        # Store details
        result.details = {
            'latency_ms': latency_ms,
            'duration_ms': duration_ms,
            'amplitude_px': amplitude_px,
            'eccentricity_px': eccentricity_px,
            'peak_velocity': peak_velocity,
            'direction_correct': direction_correct
        }
        
        result.quality_score = self._compute_quality_score(result)
        
        return result
    
    def check_antisaccade_validity(
        self,
        latency_ms: float,
        direction_error: bool,
        correction_time_ms: Optional[float],
        generic_validity: TrialValidity
    ) -> TrialValidity:
        """
        Check antisaccade trial validity.
        
        Args:
            latency_ms: Antisaccade latency in ms
            direction_error: Whether initial movement was toward stimulus
            correction_time_ms: Time to correct if error occurred
            generic_validity: Generic validity result
        
        Returns:
            TrialValidity result
        """
        result = TrialValidity(
            face_presence_rate=generic_validity.face_presence_rate,
            blink_rate=generic_validity.blink_rate,
            clamp_rate=generic_validity.clamp_rate,
            fps_median=generic_validity.fps_median,
            valid_fraction=generic_validity.valid_fraction
        )
        
        if not generic_validity.valid:
            result.valid = False
            result.reason = generic_validity.reason
            result.quality_score = generic_validity.quality_score
            return result
        
        # Check antisaccade-specific criteria
        if latency_ms < self.thresholds.antisaccade_latency_min_ms:
            result.reason = InvalidReason.ANTICIPATORY_RESPONSE
            result.valid = False
        elif latency_ms > self.thresholds.antisaccade_latency_max_ms:
            result.reason = InvalidReason.LATENCY_OUT_OF_RANGE
            result.valid = False
        else:
            result.valid = True
            result.reason = InvalidReason.VALID
        
        result.details = {
            'latency_ms': latency_ms,
            'direction_error': direction_error,
            'correction_time_ms': correction_time_ms,
            'inhibition_success': not direction_error
        }
        
        result.quality_score = self._compute_quality_score(result)
        
        return result
    
    def check_fixation_validity(
        self,
        rms_deviation_px: float,
        generic_validity: TrialValidity
    ) -> TrialValidity:
        """
        Check fixation trial validity.
        
        Args:
            rms_deviation_px: RMS deviation from target in pixels
            generic_validity: Generic validity result
        
        Returns:
            TrialValidity result
        """
        result = TrialValidity(
            face_presence_rate=generic_validity.face_presence_rate,
            blink_rate=generic_validity.blink_rate,
            clamp_rate=generic_validity.clamp_rate,
            fps_median=generic_validity.fps_median,
            valid_fraction=generic_validity.valid_fraction
        )
        
        if not generic_validity.valid:
            result.valid = False
            result.reason = generic_validity.reason
            result.quality_score = generic_validity.quality_score
            return result
        
        if generic_validity.valid_fraction < self.thresholds.fixation_min_valid_fraction:
            result.reason = InvalidReason.INSUFFICIENT_SAMPLES
            result.valid = False
        elif rms_deviation_px > self.thresholds.fixation_max_deviation_px:
            result.reason = InvalidReason.TARGET_NOT_REACHED
            result.valid = False
        else:
            result.valid = True
            result.reason = InvalidReason.VALID
        
        result.details = {'rms_deviation_px': rms_deviation_px}
        result.quality_score = self._compute_quality_score(result)
        
        return result
    
    def check_pursuit_validity(
        self,
        gain: float,
        generic_validity: TrialValidity
    ) -> TrialValidity:
        """
        Check smooth pursuit trial validity.
        
        Args:
            gain: Pursuit gain (eye velocity / target velocity)
            generic_validity: Generic validity result
        
        Returns:
            TrialValidity result
        """
        result = TrialValidity(
            face_presence_rate=generic_validity.face_presence_rate,
            blink_rate=generic_validity.blink_rate,
            clamp_rate=generic_validity.clamp_rate,
            fps_median=generic_validity.fps_median,
            valid_fraction=generic_validity.valid_fraction
        )
        
        if not generic_validity.valid:
            result.valid = False
            result.reason = generic_validity.reason
            result.quality_score = generic_validity.quality_score
            return result
        
        if generic_validity.valid_fraction < self.thresholds.pursuit_min_valid_fraction:
            result.reason = InvalidReason.INSUFFICIENT_SAMPLES
            result.valid = False
        elif gain < self.thresholds.pursuit_min_gain or gain > self.thresholds.pursuit_max_gain:
            result.reason = InvalidReason.NO_MEANINGFUL_MOVEMENT
            result.valid = False
        else:
            result.valid = True
            result.reason = InvalidReason.VALID
        
        result.details = {'gain': gain}
        result.quality_score = self._compute_quality_score(result)
        
        return result
    
    def check_grid_point_validity(
        self,
        error_px: float,
        generic_validity: TrialValidity
    ) -> TrialValidity:
        """
        Check grid point validity.
        
        Args:
            error_px: Gaze error from target in pixels
            generic_validity: Generic validity result
        
        Returns:
            TrialValidity result
        """
        result = TrialValidity(
            face_presence_rate=generic_validity.face_presence_rate,
            blink_rate=generic_validity.blink_rate,
            clamp_rate=generic_validity.clamp_rate,
            fps_median=generic_validity.fps_median,
            valid_fraction=generic_validity.valid_fraction
        )
        
        if not generic_validity.valid:
            result.valid = False
            result.reason = generic_validity.reason
            result.quality_score = generic_validity.quality_score
            return result
        
        if generic_validity.valid_fraction < self.thresholds.grid_min_valid_fraction:
            result.reason = InvalidReason.INSUFFICIENT_SAMPLES
            result.valid = False
        elif error_px > self.thresholds.grid_max_error_px:
            result.reason = InvalidReason.TARGET_NOT_REACHED
            result.valid = False
        else:
            result.valid = True
            result.reason = InvalidReason.VALID
        
        result.details = {'error_px': error_px}
        result.quality_score = self._compute_quality_score(result)
        
        return result
    
    def _compute_quality_score(self, validity: TrialValidity) -> float:
        """
        Compute quality score from validity metrics.
        
        Returns:
            Quality score in [0, 1]
        """
        if not validity.valid:
            # Reduced score for invalid trials
            base_score = 0.3
        else:
            base_score = 0.7
        
        # Component scores
        face_score = min(1.0, validity.face_presence_rate / self.thresholds.min_face_presence_rate)
        blink_score = max(0.0, 1.0 - validity.blink_rate / self.thresholds.max_blink_rate)
        clamp_score = max(0.0, 1.0 - validity.clamp_rate / self.thresholds.max_clamp_rate_per_trial)
        fps_score = min(1.0, validity.fps_median / 30.0)
        valid_score = validity.valid_fraction
        
        # Weighted combination
        component_score = (
            0.25 * face_score +
            0.20 * blink_score +
            0.20 * clamp_score +
            0.15 * fps_score +
            0.20 * valid_score
        )
        
        # Combine with base score
        quality = base_score * 0.4 + component_score * 0.6
        
        return max(0.0, min(1.0, quality))
    
    def compute_session_validity(
        self,
        trial_validities: List[TrialValidity]
    ) -> Dict[str, Any]:
        """
        Compute session-level validity metrics.
        
        Args:
            trial_validities: List of trial validity results
        
        Returns:
            Dictionary with session validity metrics
        """
        if not trial_validities:
            return {
                'valid_trial_count': 0,
                'total_trial_count': 0,
                'valid_trial_rate': 0.0,
                'meets_target': False,
                'meets_minimum': False,
                'mean_quality_score': 0.0,
                'recommendation': 'No trials recorded'
            }
        
        valid_count = sum(1 for v in trial_validities if v.valid)
        total_count = len(trial_validities)
        valid_rate = valid_count / total_count
        
        mean_quality = np.mean([v.quality_score for v in trial_validities])
        
        meets_target = valid_rate >= self.thresholds.target_valid_trial_rate
        meets_minimum = valid_rate >= self.thresholds.min_valid_trial_rate
        
        # Generate recommendation
        if meets_target:
            recommendation = 'Session quality is good'
        elif meets_minimum:
            recommendation = 'Session quality is acceptable but below target'
        else:
            # Analyze common failure reasons
            reason_counts: Dict[InvalidReason, int] = {}
            for v in trial_validities:
                if not v.valid:
                    reason_counts[v.reason] = reason_counts.get(v.reason, 0) + 1
            
            if reason_counts:
                most_common = max(reason_counts, key=reason_counts.get)
                recommendation = f'Recalibration recommended. Most common issue: {most_common.value}'
            else:
                recommendation = 'Recalibration recommended'
        
        return {
            'valid_trial_count': valid_count,
            'total_trial_count': total_count,
            'valid_trial_rate': valid_rate,
            'meets_target': meets_target,
            'meets_minimum': meets_minimum,
            'mean_quality_score': mean_quality,
            'recommendation': recommendation
        }
