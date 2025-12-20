"""Utility functions for NeuroLens+ eye tracking system."""

import numpy as np
from typing import Tuple, Optional, List
from collections import deque
import time


def compute_velocity(
    x_curr: float,
    y_curr: float,
    x_prev: float,
    y_prev: float,
    dt: float,
    min_dt: float = 0.001,
    max_dt: float = 0.5
) -> Tuple[float, float]:
    """
    Compute velocity in pixels/second with dt guards.
    
    Args:
        x_curr, y_curr: Current position in pixels
        x_prev, y_prev: Previous position in pixels
        dt: Time delta in seconds
        min_dt: Minimum valid dt (to avoid division by near-zero)
        max_dt: Maximum valid dt (to avoid stale data)
    
    Returns:
        Tuple of (vel_x, vel_y) in pixels/second, or (NaN, NaN) if dt invalid
    """
    if dt < min_dt or dt > max_dt or np.isnan(dt):
        return (np.nan, np.nan)
    
    vel_x = (x_curr - x_prev) / dt
    vel_y = (y_curr - y_prev) / dt
    
    return (vel_x, vel_y)


def compute_ear(eye_landmarks: List[Tuple[float, float]]) -> float:
    """
    Compute Eye Aspect Ratio (EAR) for blink detection.
    
    Uses 6 landmarks around the eye:
    - Points 0, 3: horizontal extremes (left, right corners)
    - Points 1, 5: upper lid
    - Points 2, 4: lower lid
    
    EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
    
    Args:
        eye_landmarks: List of 6 (x, y) tuples for eye landmarks
    
    Returns:
        EAR value (lower = more closed)
    """
    if len(eye_landmarks) < 6:
        return np.nan
    
    p0, p1, p2, p3, p4, p5 = eye_landmarks[:6]
    
    # Vertical distances
    v1 = np.sqrt((p1[0] - p5[0])**2 + (p1[1] - p5[1])**2)
    v2 = np.sqrt((p2[0] - p4[0])**2 + (p2[1] - p4[1])**2)
    
    # Horizontal distance
    h = np.sqrt((p0[0] - p3[0])**2 + (p0[1] - p3[1])**2)
    
    if h < 1e-6:
        return np.nan
    
    ear = (v1 + v2) / (2.0 * h)
    return ear


def detect_blink(
    ear_left: float,
    ear_right: float,
    threshold: float = 0.21
) -> bool:
    """
    Detect blink based on EAR values.
    
    Args:
        ear_left: Left eye EAR
        ear_right: Right eye EAR
        threshold: EAR threshold below which is considered a blink
    
    Returns:
        True if blink detected
    """
    if np.isnan(ear_left) and np.isnan(ear_right):
        return True  # No eye data = treat as blink
    
    ear_mean = np.nanmean([ear_left, ear_right])
    return ear_mean < threshold


class RollingFPS:
    """Rolling FPS calculator with configurable window."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps: deque = deque(maxlen=window_size)
        self.last_time: Optional[float] = None
    
    def update(self, timestamp: Optional[float] = None) -> float:
        """
        Update with new frame and return current FPS estimate.
        
        Args:
            timestamp: Optional timestamp, uses time.time() if not provided
        
        Returns:
            Rolling FPS estimate
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.timestamps.append(timestamp)
        
        if len(self.timestamps) < 2:
            return 30.0  # Default estimate
        
        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span <= 0:
            return 30.0
        
        fps = (len(self.timestamps) - 1) / time_span
        return fps


def rolling_fps(timestamps: List[float], window: int = 30) -> float:
    """
    Compute rolling FPS from list of timestamps.
    
    Args:
        timestamps: List of timestamps
        window: Window size for rolling calculation
    
    Returns:
        FPS estimate
    """
    if len(timestamps) < 2:
        return 30.0
    
    recent = timestamps[-window:] if len(timestamps) > window else timestamps
    time_span = recent[-1] - recent[0]
    
    if time_span <= 0:
        return 30.0
    
    return (len(recent) - 1) / time_span


def clamp_value(
    value: float,
    min_val: float,
    max_val: float
) -> Tuple[float, bool]:
    """
    Clamp value to range and return clamped flag.
    
    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Returns:
        Tuple of (clamped_value, was_clamped)
    """
    if np.isnan(value):
        return (value, False)
    
    if value < min_val:
        return (min_val, True)
    elif value > max_val:
        return (max_val, True)
    else:
        return (value, False)


def compute_rms(values: np.ndarray) -> float:
    """
    Compute Root Mean Square of values.
    
    Args:
        values: Array of values
    
    Returns:
        RMS value
    """
    values = np.asarray(values)
    valid = values[~np.isnan(values)]
    
    if len(valid) == 0:
        return np.nan
    
    return np.sqrt(np.mean(valid**2))


def compute_bcea(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    p: float = 0.68
) -> float:
    """
    Compute Bivariate Contour Ellipse Area (BCEA).
    
    BCEA = 2 * pi * chi2_val * sigma_x * sigma_y * sqrt(1 - rho^2)
    
    Args:
        x_positions: Array of x positions
        y_positions: Array of y positions
        p: Probability level (default 0.68 for 1 SD)
    
    Returns:
        BCEA in squared units (px^2 if inputs are in px)
    """
    from scipy import stats
    
    x = np.asarray(x_positions)
    y = np.asarray(y_positions)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask]
    y = y[valid_mask]
    
    if len(x) < 3:
        return np.nan
    
    # Compute standard deviations
    sigma_x = np.std(x, ddof=1)
    sigma_y = np.std(y, ddof=1)
    
    # Compute correlation
    if sigma_x < 1e-10 or sigma_y < 1e-10:
        return np.nan
    
    rho = np.corrcoef(x, y)[0, 1]
    if np.isnan(rho):
        rho = 0
    
    # Chi-squared value for given probability
    chi2_val = stats.chi2.ppf(p, df=2)
    
    # BCEA formula
    bcea = 2 * np.pi * chi2_val * sigma_x * sigma_y * np.sqrt(1 - rho**2)
    
    return bcea


def compute_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def smooth_signal(
    signal: np.ndarray,
    window_size: int = 5,
    method: str = "median"
) -> np.ndarray:
    """
    Smooth a signal using specified method.
    
    Args:
        signal: Input signal array
        window_size: Smoothing window size
        method: "median" or "mean"
    
    Returns:
        Smoothed signal
    """
    from scipy.ndimage import median_filter, uniform_filter1d
    
    signal = np.asarray(signal, dtype=float)
    
    if method == "median":
        return median_filter(signal, size=window_size, mode='nearest')
    elif method == "mean":
        return uniform_filter1d(signal, size=window_size, mode='nearest')
    else:
        return signal


def detect_saccade_onset(
    velocities: np.ndarray,
    timestamps: np.ndarray,
    threshold: float = 30.0,
    min_duration_samples: int = 2
) -> Optional[int]:
    """
    Detect saccade onset based on velocity threshold.
    
    Args:
        velocities: Array of velocity magnitudes
        timestamps: Array of timestamps
        threshold: Velocity threshold in px/s
        min_duration_samples: Minimum consecutive samples above threshold
    
    Returns:
        Index of saccade onset, or None if not detected
    """
    above_threshold = velocities > threshold
    
    consecutive = 0
    for i, above in enumerate(above_threshold):
        if above:
            consecutive += 1
            if consecutive >= min_duration_samples:
                return i - min_duration_samples + 1
        else:
            consecutive = 0
    
    return None


def find_peak_velocity(
    velocities: np.ndarray,
    start_idx: int,
    end_idx: int
) -> Tuple[float, int]:
    """
    Find peak velocity within a window.
    
    Args:
        velocities: Array of velocity magnitudes
        start_idx: Start index of window
        end_idx: End index of window
    
    Returns:
        Tuple of (peak_velocity, peak_index)
    """
    window = velocities[start_idx:end_idx]
    if len(window) == 0:
        return (np.nan, -1)
    
    peak_idx_local = np.nanargmax(window)
    peak_vel = window[peak_idx_local]
    
    return (peak_vel, start_idx + peak_idx_local)


def compute_pursuit_gain(
    eye_velocity: np.ndarray,
    target_velocity: np.ndarray,
    valid_mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute smooth pursuit gain (eye velocity / target velocity).
    
    Args:
        eye_velocity: Array of eye velocities
        target_velocity: Array of target velocities
        valid_mask: Optional mask for valid samples
    
    Returns:
        Mean pursuit gain
    """
    if valid_mask is None:
        valid_mask = np.ones(len(eye_velocity), dtype=bool)
    
    # Only compute where target is moving
    moving_mask = np.abs(target_velocity) > 10  # px/s threshold
    combined_mask = valid_mask & moving_mask
    
    if not np.any(combined_mask):
        return np.nan
    
    eye_vel = eye_velocity[combined_mask]
    target_vel = target_velocity[combined_mask]
    
    # Avoid division by zero
    valid_target = np.abs(target_vel) > 1e-6
    if not np.any(valid_target):
        return np.nan
    
    gains = eye_vel[valid_target] / target_vel[valid_target]
    
    return np.nanmean(gains)


def cross_correlation_lag(
    signal1: np.ndarray,
    signal2: np.ndarray,
    max_lag_samples: int = 30
) -> Tuple[float, int]:
    """
    Compute cross-correlation and find lag at peak.
    
    Args:
        signal1: First signal (e.g., eye position)
        signal2: Second signal (e.g., target position)
        max_lag_samples: Maximum lag to search
    
    Returns:
        Tuple of (correlation_at_peak, lag_in_samples)
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(signal1) | np.isnan(signal2))
    s1 = signal1[valid_mask]
    s2 = signal2[valid_mask]
    
    if len(s1) < 10:
        return (np.nan, 0)
    
    # Normalize signals
    s1 = (s1 - np.mean(s1)) / (np.std(s1) + 1e-10)
    s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-10)
    
    # Compute cross-correlation
    correlation = np.correlate(s1, s2, mode='full')
    
    # Find peak within allowed lag range
    mid = len(correlation) // 2
    search_start = max(0, mid - max_lag_samples)
    search_end = min(len(correlation), mid + max_lag_samples + 1)
    
    search_region = correlation[search_start:search_end]
    peak_idx_local = np.argmax(search_region)
    peak_idx = search_start + peak_idx_local
    
    lag = peak_idx - mid
    peak_corr = correlation[peak_idx] / len(s1)
    
    return (peak_corr, lag)


def compute_microsaccade_rate(
    velocities: np.ndarray,
    timestamps: np.ndarray,
    threshold: float = 50.0,
    min_duration_ms: float = 6.0,
    max_duration_ms: float = 50.0
) -> float:
    """
    Compute microsaccade rate during fixation.
    
    Args:
        velocities: Array of velocity magnitudes
        timestamps: Array of timestamps
        threshold: Velocity threshold for microsaccade detection
        min_duration_ms: Minimum microsaccade duration
        max_duration_ms: Maximum microsaccade duration
    
    Returns:
        Microsaccade rate per minute
    """
    if len(velocities) < 3 or len(timestamps) < 3:
        return np.nan
    
    total_time_s = timestamps[-1] - timestamps[0]
    if total_time_s <= 0:
        return np.nan
    
    # Detect velocity peaks above threshold
    above_threshold = velocities > threshold
    
    # Find microsaccade events
    microsaccade_count = 0
    in_event = False
    event_start_idx = 0
    
    for i, above in enumerate(above_threshold):
        if above and not in_event:
            in_event = True
            event_start_idx = i
        elif not above and in_event:
            in_event = False
            # Check duration
            if i > event_start_idx and event_start_idx < len(timestamps) and i < len(timestamps):
                duration_ms = (timestamps[i] - timestamps[event_start_idx]) * 1000
                if min_duration_ms <= duration_ms <= max_duration_ms:
                    microsaccade_count += 1
    
    # Convert to rate per minute
    rate_per_min = microsaccade_count / (total_time_s / 60.0)
    
    return rate_per_min


def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Simple linear regression.
    
    Args:
        x: Independent variable
        y: Dependent variable
    
    Returns:
        Tuple of (slope, intercept, r_squared)
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask]
    y = y[valid_mask]
    
    if len(x) < 2:
        return (np.nan, np.nan, np.nan)
    
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x**2)
    
    denom = n * sum_x2 - sum_x**2
    if abs(denom) < 1e-10:
        return (np.nan, np.nan, np.nan)
    
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


def compute_quality_score(
    valid_fraction: float,
    calibration_quality: float,
    clamp_rate: float,
    fps_median: float,
    face_presence_rate: float,
    target_fps: float = 30.0
) -> float:
    """
    Compute overall quality score for a trial.
    
    Args:
        valid_fraction: Fraction of valid samples
        calibration_quality: Calibration quality score (0-1)
        clamp_rate: Rate of clamped samples
        fps_median: Median FPS during trial
        face_presence_rate: Rate of face detection
        target_fps: Target FPS for scoring
    
    Returns:
        Quality score in [0, 1]
    """
    # Weight factors
    w_valid = 0.30
    w_calib = 0.25
    w_clamp = 0.15
    w_fps = 0.15
    w_face = 0.15
    
    # Normalize components
    score_valid = min(1.0, valid_fraction)
    score_calib = min(1.0, calibration_quality)
    score_clamp = max(0.0, 1.0 - clamp_rate * 10)  # Penalize clamp rate
    score_fps = min(1.0, fps_median / target_fps)
    score_face = min(1.0, face_presence_rate)
    
    # Weighted sum
    quality = (
        w_valid * score_valid +
        w_calib * score_calib +
        w_clamp * score_clamp +
        w_fps * score_fps +
        w_face * score_face
    )
    
    return max(0.0, min(1.0, quality))
