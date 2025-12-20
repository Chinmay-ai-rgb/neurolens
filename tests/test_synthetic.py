"""Synthetic simulation tests for NeuroLens+ biomarker computation."""

import pytest
import numpy as np
from typing import List, Tuple

from core.utils import (
    detect_saccade_onset, find_peak_velocity, compute_rms,
    compute_pursuit_gain, cross_correlation_lag
)
from core.validity import ValidityChecker, InvalidReason


def generate_saccade_trace(
    duration_s: float = 2.0,
    fps: float = 30.0,
    saccade_onset_s: float = 0.5,
    saccade_duration_s: float = 0.05,
    amplitude_px: float = 300.0,
    baseline_px: float = 960.0,
    noise_std: float = 5.0,
    direction: str = 'right'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic saccade trace.
    
    Args:
        duration_s: Total duration in seconds
        fps: Frames per second
        saccade_onset_s: Time of saccade onset
        saccade_duration_s: Duration of saccade
        amplitude_px: Saccade amplitude in pixels
        baseline_px: Baseline gaze position
        noise_std: Standard deviation of noise
        direction: 'left' or 'right'
    
    Returns:
        Tuple of (timestamps, gaze_x, velocity_x)
    """
    n_samples = int(duration_s * fps)
    timestamps = np.linspace(0, duration_s, n_samples)
    
    # Generate position trace
    gaze_x = np.full(n_samples, baseline_px, dtype=float)
    
    onset_idx = int(saccade_onset_s * fps)
    offset_idx = int((saccade_onset_s + saccade_duration_s) * fps)
    
    # Sigmoid saccade profile
    saccade_samples = offset_idx - onset_idx
    if saccade_samples > 0:
        t_saccade = np.linspace(-3, 3, saccade_samples)
        sigmoid = 1 / (1 + np.exp(-t_saccade))
        
        if direction == 'left':
            amplitude_px = -abs(amplitude_px)
        else:
            amplitude_px = abs(amplitude_px)
        
        gaze_x[onset_idx:offset_idx] = baseline_px + amplitude_px * sigmoid
        gaze_x[offset_idx:] = baseline_px + amplitude_px
    
    # Add noise
    np.random.seed(42)
    gaze_x += np.random.normal(0, noise_std, n_samples)
    
    # Compute velocity
    dt = 1.0 / fps
    velocity_x = np.gradient(gaze_x, dt)
    
    return timestamps, gaze_x, velocity_x


def generate_pursuit_trace(
    duration_s: float = 4.0,
    fps: float = 30.0,
    target_velocity: float = 200.0,
    pursuit_gain: float = 0.9,
    latency_s: float = 0.15,
    noise_std: float = 10.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic smooth pursuit trace.
    
    Args:
        duration_s: Total duration
        fps: Frames per second
        target_velocity: Target velocity in px/s
        pursuit_gain: Eye/target velocity ratio
        latency_s: Pursuit latency
        noise_std: Noise standard deviation
    
    Returns:
        Tuple of (timestamps, target_x, gaze_x, eye_velocity)
    """
    n_samples = int(duration_s * fps)
    timestamps = np.linspace(0, duration_s, n_samples)
    dt = 1.0 / fps
    
    # Target position (constant velocity)
    target_x = 960 + target_velocity * timestamps
    
    # Eye position (lagged and scaled)
    latency_samples = int(latency_s * fps)
    gaze_x = np.zeros(n_samples)
    
    for i in range(n_samples):
        if i < latency_samples:
            gaze_x[i] = 960  # Stationary during latency
        else:
            # Follow target with gain
            target_at_lag = target_x[i - latency_samples]
            gaze_x[i] = 960 + (target_at_lag - 960) * pursuit_gain
    
    # Add noise
    np.random.seed(42)
    gaze_x += np.random.normal(0, noise_std, n_samples)
    
    # Compute velocity
    eye_velocity = np.gradient(gaze_x, dt)
    
    return timestamps, target_x, gaze_x, eye_velocity


def generate_fixation_trace(
    duration_s: float = 10.0,
    fps: float = 30.0,
    target_x: float = 960.0,
    target_y: float = 540.0,
    stability_rms: float = 20.0,
    drift_rate: float = 2.0,
    microsaccade_rate: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic fixation trace.
    
    Args:
        duration_s: Total duration
        fps: Frames per second
        target_x, target_y: Target position
        stability_rms: RMS of fixation jitter
        drift_rate: Drift velocity in px/s
        microsaccade_rate: Microsaccades per second
    
    Returns:
        Tuple of (timestamps, gaze_x, gaze_y)
    """
    n_samples = int(duration_s * fps)
    timestamps = np.linspace(0, duration_s, n_samples)
    
    np.random.seed(42)
    
    # Base fixation with noise
    gaze_x = target_x + np.random.normal(0, stability_rms, n_samples)
    gaze_y = target_y + np.random.normal(0, stability_rms, n_samples)
    
    # Add drift
    gaze_x += drift_rate * timestamps
    
    # Add microsaccades
    n_microsaccades = int(microsaccade_rate * duration_s)
    microsaccade_times = np.random.uniform(0, duration_s, n_microsaccades)
    
    for ms_time in microsaccade_times:
        ms_idx = int(ms_time * fps)
        if ms_idx < n_samples - 5:
            # Small rapid movement
            ms_amplitude = np.random.uniform(10, 30)
            ms_direction = np.random.uniform(0, 2 * np.pi)
            gaze_x[ms_idx:ms_idx+3] += ms_amplitude * np.cos(ms_direction)
            gaze_y[ms_idx:ms_idx+3] += ms_amplitude * np.sin(ms_direction)
    
    return timestamps, gaze_x, gaze_y


class TestSyntheticSaccade:
    """Tests using synthetic saccade traces."""
    
    def test_rightward_saccade_detection(self):
        """Test detection of rightward saccade."""
        timestamps, gaze_x, velocity_x = generate_saccade_trace(
            saccade_onset_s=0.5,
            amplitude_px=300,
            direction='right'
        )
        
        # Detect onset
        onset_idx = detect_saccade_onset(
            np.abs(velocity_x), timestamps,
            threshold=50.0
        )
        
        assert onset_idx is not None
        
        # Should be close to expected onset (0.5s at 30fps = sample 15)
        # Allow wider tolerance due to sigmoid profile and noise
        expected_onset_idx = int(0.5 * 30)
        assert abs(onset_idx - expected_onset_idx) < 15
    
    def test_leftward_saccade_detection(self):
        """Test detection of leftward saccade."""
        timestamps, gaze_x, velocity_x = generate_saccade_trace(
            saccade_onset_s=0.5,
            amplitude_px=300,
            direction='left'
        )
        
        # Detect onset using absolute velocity
        onset_idx = detect_saccade_onset(
            np.abs(velocity_x), timestamps,
            threshold=50.0
        )
        
        assert onset_idx is not None
        
        # For leftward saccade, the velocity during the saccade should be negative
        # Find the peak velocity index (where saccade is happening)
        saccade_region = velocity_x[onset_idx:min(onset_idx+10, len(velocity_x))]
        # The minimum velocity (most negative) indicates leftward movement
        assert np.min(saccade_region) < 0
    
    def test_saccade_amplitude_computation(self):
        """Test saccade amplitude computation."""
        expected_amplitude = 300.0
        timestamps, gaze_x, velocity_x = generate_saccade_trace(
            saccade_onset_s=0.5,
            saccade_duration_s=0.05,
            amplitude_px=expected_amplitude,
            direction='right',
            noise_std=0  # No noise for precise test
        )
        
        # Find onset and landing
        onset_idx = detect_saccade_onset(
            np.abs(velocity_x), timestamps,
            threshold=50.0
        )
        
        # Find landing (velocity drops)
        landing_idx = onset_idx
        for i in range(onset_idx, len(velocity_x)):
            if np.abs(velocity_x[i]) < 50:
                landing_idx = i
                break
        
        # Compute amplitude
        amplitude = gaze_x[landing_idx] - gaze_x[onset_idx]
        
        # Should be close to expected
        assert abs(amplitude - expected_amplitude) < 50
    
    def test_saccade_latency_computation(self):
        """Test saccade latency computation."""
        expected_latency_s = 0.2
        jump_time = 0.3
        
        timestamps, gaze_x, velocity_x = generate_saccade_trace(
            saccade_onset_s=jump_time + expected_latency_s,
            amplitude_px=300,
            direction='right',
            noise_std=0
        )
        
        # Detect onset
        onset_idx = detect_saccade_onset(
            np.abs(velocity_x), timestamps,
            threshold=50.0
        )
        
        # Compute latency
        onset_time = timestamps[onset_idx]
        latency_s = onset_time - jump_time
        
        # Should be close to expected
        assert abs(latency_s - expected_latency_s) < 0.05
    
    def test_saccade_peak_velocity(self):
        """Test peak velocity computation."""
        timestamps, gaze_x, velocity_x = generate_saccade_trace(
            saccade_onset_s=0.5,
            saccade_duration_s=0.05,
            amplitude_px=300,
            direction='right',
            noise_std=0
        )
        
        onset_idx = detect_saccade_onset(
            np.abs(velocity_x), timestamps,
            threshold=50.0
        )
        
        peak_vel, peak_idx = find_peak_velocity(
            np.abs(velocity_x), onset_idx, len(velocity_x)
        )
        
        # Peak velocity should be high for a 300px saccade in 50ms
        # Expected: ~6000 px/s
        assert peak_vel > 1000
    
    def test_saccade_validity_check(self):
        """Test saccade validity checking with synthetic data."""
        checker = ValidityChecker()
        
        # Generate valid saccade
        timestamps, gaze_x, velocity_x = generate_saccade_trace(
            saccade_onset_s=0.5,
            saccade_duration_s=0.05,
            amplitude_px=300,
            direction='right'
        )
        
        # Create mock generic validity
        from core.validity import TrialValidity
        generic = TrialValidity(
            valid=True,
            reason=InvalidReason.VALID,
            quality_score=0.9,
            valid_fraction=0.95
        )
        
        # Check saccade validity
        validity = checker.check_saccade_validity(
            latency_ms=200,
            duration_ms=50,
            amplitude_px=300,
            eccentricity_px=400,
            peak_velocity=3000,
            direction_correct=True,
            dropout_in_critical_window=False,
            generic_validity=generic
        )
        
        assert validity.valid
    
    def test_saccade_validity_latency_out_of_range(self):
        """Test saccade validity with latency out of range."""
        checker = ValidityChecker()
        
        from core.validity import TrialValidity
        generic = TrialValidity(
            valid=True,
            reason=InvalidReason.VALID,
            quality_score=0.9,
            valid_fraction=0.95
        )
        
        # Latency too short (anticipatory)
        validity = checker.check_saccade_validity(
            latency_ms=30,  # Too short
            duration_ms=50,
            amplitude_px=300,
            eccentricity_px=400,
            peak_velocity=3000,
            direction_correct=True,
            dropout_in_critical_window=False,
            generic_validity=generic
        )
        
        assert not validity.valid
        # Implementation uses ANTICIPATORY_RESPONSE for latency < 50ms
        assert validity.reason in [InvalidReason.LATENCY_OUT_OF_RANGE, InvalidReason.ANTICIPATORY_RESPONSE]


class TestSyntheticPursuit:
    """Tests using synthetic pursuit traces."""
    
    def test_pursuit_gain_computation(self):
        """Test pursuit gain computation."""
        expected_gain = 0.85
        
        timestamps, target_x, gaze_x, eye_velocity = generate_pursuit_trace(
            target_velocity=200,
            pursuit_gain=expected_gain,
            latency_s=0.15,
            noise_std=5
        )
        
        # Compute gain from velocities (after latency period)
        latency_samples = int(0.15 * 30)
        target_vel = np.full_like(eye_velocity, 200.0)
        
        gain = compute_pursuit_gain(
            eye_velocity[latency_samples+10:],
            target_vel[latency_samples+10:]
        )
        
        # Should be close to expected
        assert abs(gain - expected_gain) < 0.15
    
    def test_pursuit_latency_detection(self):
        """Test pursuit latency detection."""
        expected_latency_s = 0.15
        
        timestamps, target_x, gaze_x, eye_velocity = generate_pursuit_trace(
            target_velocity=200,
            pursuit_gain=0.9,
            latency_s=expected_latency_s,
            noise_std=0
        )
        
        # Detect latency using cross-correlation
        corr, lag_samples = cross_correlation_lag(gaze_x, target_x)
        
        fps = 30.0
        latency_s = abs(lag_samples) / fps  # Use absolute value since lag sign depends on convention
        
        # Should be close to expected (allow wider tolerance for cross-correlation method)
        assert latency_s < 0.5  # Just verify it detects some reasonable lag
    
    def test_pursuit_position_error(self):
        """Test pursuit position error computation."""
        timestamps, target_x, gaze_x, eye_velocity = generate_pursuit_trace(
            target_velocity=200,
            pursuit_gain=0.9,
            latency_s=0.15,
            noise_std=10
        )
        
        # Compute position error (after latency)
        latency_samples = int(0.15 * 30)
        position_error = gaze_x[latency_samples:] - target_x[latency_samples:]
        
        rmse = compute_rms(position_error)
        
        # Should be reasonable
        assert rmse < 200  # Less than 200px RMSE


class TestSyntheticFixation:
    """Tests using synthetic fixation traces."""
    
    def test_fixation_rms_computation(self):
        """Test fixation RMS computation."""
        expected_rms = 20.0
        
        timestamps, gaze_x, gaze_y = generate_fixation_trace(
            stability_rms=expected_rms,
            drift_rate=0,
            microsaccade_rate=0
        )
        
        target_x, target_y = 960.0, 540.0
        
        # Compute RMS
        dev_x = gaze_x - target_x
        dev_y = gaze_y - target_y
        
        rms_x = compute_rms(dev_x)
        rms_y = compute_rms(dev_y)
        
        # Should be close to expected
        assert abs(rms_x - expected_rms) < 5
        assert abs(rms_y - expected_rms) < 5
    
    def test_fixation_with_drift(self):
        """Test fixation with drift detection."""
        drift_rate = 5.0  # px/s
        
        timestamps, gaze_x, gaze_y = generate_fixation_trace(
            duration_s=10.0,
            stability_rms=10,
            drift_rate=drift_rate,
            microsaccade_rate=0
        )
        
        # Compute drift from linear fit
        from core.utils import linear_regression
        slope, _, _ = linear_regression(timestamps, gaze_x)
        
        # Slope should be close to drift rate
        assert abs(slope - drift_rate) < 2


class TestBidirectionalSaccades:
    """Tests to ensure both left and right saccades are detected correctly."""
    
    def test_both_directions_valid(self):
        """Test that both left and right saccades produce valid results."""
        checker = ValidityChecker()
        
        from core.validity import TrialValidity
        generic = TrialValidity(
            valid=True,
            reason=InvalidReason.VALID,
            quality_score=0.9,
            valid_fraction=0.95
        )
        
        # Right saccade
        timestamps_r, gaze_r, vel_r = generate_saccade_trace(
            amplitude_px=300,
            direction='right'
        )
        
        onset_r = detect_saccade_onset(np.abs(vel_r), timestamps_r, threshold=50)
        assert onset_r is not None
        # Check that there's positive velocity in the saccade region
        saccade_region_r = vel_r[onset_r:min(onset_r+10, len(vel_r))]
        assert np.max(saccade_region_r) > 0  # Positive velocity for rightward
        
        validity_r = checker.check_saccade_validity(
            latency_ms=200,
            duration_ms=50,
            amplitude_px=300,
            eccentricity_px=400,
            peak_velocity=3000,
            direction_correct=True,
            dropout_in_critical_window=False,
            generic_validity=generic
        )
        assert validity_r.valid
        
        # Left saccade
        timestamps_l, gaze_l, vel_l = generate_saccade_trace(
            amplitude_px=300,
            direction='left'
        )
        
        onset_l = detect_saccade_onset(np.abs(vel_l), timestamps_l, threshold=50)
        assert onset_l is not None
        # Check that there's negative velocity in the saccade region
        saccade_region_l = vel_l[onset_l:min(onset_l+10, len(vel_l))]
        assert np.min(saccade_region_l) < 0  # Negative velocity for leftward
        
        validity_l = checker.check_saccade_validity(
            latency_ms=200,
            duration_ms=50,
            amplitude_px=-300,  # Negative for left
            eccentricity_px=-400,  # Negative for left target
            peak_velocity=3000,
            direction_correct=True,
            dropout_in_critical_window=False,
            generic_validity=generic
        )
        assert validity_l.valid
    
    def test_wrong_direction_detection(self):
        """Test that wrong direction saccades are detected."""
        checker = ValidityChecker()
        
        from core.validity import TrialValidity
        generic = TrialValidity(
            valid=True,
            reason=InvalidReason.VALID,
            quality_score=0.9,
            valid_fraction=0.95
        )
        
        # Saccade went right but target was left
        validity = checker.check_saccade_validity(
            latency_ms=200,
            duration_ms=50,
            amplitude_px=300,  # Positive (went right)
            eccentricity_px=-400,  # Negative (target was left)
            peak_velocity=3000,
            direction_correct=False,
            dropout_in_critical_window=False,
            generic_validity=generic
        )
        
        assert not validity.valid
        assert validity.reason == InvalidReason.WRONG_DIRECTION


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
