"""
Saccade Task for NeuroLens+ eye tracking system.

FIXES APPLIED:
1) Amplitude:
   - Per-trial: store BOTH signed and absolute amplitude
       saccade_amplitude_signed_px (directional)
       saccade_amplitude_px        (absolute; ML-safe)
   - Session biomarker mean_amplitude_px uses ABS amplitude.

2) Landing selection + landing error (the big fix):
   - Landing index is NO LONGER "max displacement" (brittle).
   - Landing index is:
       (a) first stable within landing radius for >= stable_frames, OR
       (b) closest-approach to target fallback.
   - Uses SMOOTHED mapped gaze (x,y) for landing detection and endpoint.
   - landing_error_px is Euclidean distance from (land_gx, land_gy) to (target_x, target_y).
   - No clamps/caps.

3) Finite-only arrays:
   - Filter gaze/vel/timestamp arrays to finite values before detection.

4) Summary aggregation:
   - Biomarkers computed ONLY from trials where valid==1 AND values are finite.

5) Latency clock:
   - Uses jump_time as stimulus onset (kept).

NOTE ABOUT CSV COLUMNS:
- If your SummaryLogger writes only columns listed in SACCADE_SUMMARY_COLUMNS,
  you MUST add "saccade_amplitude_signed_px" to that list if you want it in summary_saccade.csv.
  If you don't add it, it will simply not appear (or could error depending on logger).
"""

import numpy as np
import time
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .base import BaseTask, TaskConfig, TaskResult
from core.logging import FrameLogger, SummaryLogger, SACCADE_SUMMARY_COLUMNS
from core.validity import InvalidReason
from core.utils import detect_saccade_onset_robust, find_peak_velocity


class SaccadeState(Enum):
    """Saccade trial state machine states."""
    FORE = "FORE"   # Foreperiod - fixation at center
    JUMP = "JUMP"   # Target jump - stimulus onset (not used as a loop state here)
    POST = "POST"   # Post-saccade recording
    NEXT = "NEXT"   # Inter-trial interval


@dataclass
class SaccadeConfig(TaskConfig):
    """Configuration for saccade task."""

    # Trial settings
    n_trials: int = 20

    # Target positions (normalized x coordinates)
    left_target_x_norm: float = 0.2
    right_target_x_norm: float = 0.8
    center_x_norm: float = 0.5
    target_y_norm: float = 0.5

    # Timing (seconds)
    foreperiod_min: float = 1.0
    foreperiod_max: float = 1.8
    post_duration: float = 1.2
    inter_trial_interval: float = 0.5

    # Saccade detection thresholds (webcam-tolerant)
    velocity_threshold: float = 150.0  # px/s for onset detection
    min_velocity_samples: int = 3      # consecutive samples above threshold

    # Validity thresholds (kept for checker)
    latency_min_ms: float = 50.0
    latency_max_ms: float = 900.0
    duration_min_ms: float = 15.0
    duration_max_ms: float = 400.0
    min_amplitude_fraction: float = 0.15
    max_peak_velocity: float = 5000.0  # px/s (your checker may override in ValidityThresholds)
    critical_window_ms: float = 250.0  # dropout window after jump


class SaccadeTask(BaseTask):
    """
    Saccade Task - Measure saccadic eye movements.

    State machine: FORE -> POST -> NEXT
    """

    TASK_NAME = "saccade"

    def __init__(self, config: Optional[SaccadeConfig] = None):
        super().__init__(config or SaccadeConfig())
        self.config: SaccadeConfig = self.config
        self.trial_directions: List[str] = []

    def get_instructions(self) -> List[str]:
        return [
            "SACCADE TASK",
            "",
            "A white dot will appear in the center of the screen.",
            "Keep your eyes on the center dot.",
            "",
            "When the dot JUMPS to the left or right,",
            "move your eyes to follow it as quickly as possible.",
            "",
            f"There will be {self.config.n_trials} trials.",
            "",
            "Press SPACE to start.",
            "Press Q or ESC to quit at any time.",
            "Press S to skip a trial."
        ]

    def get_summary_columns(self) -> List[str]:
        return SACCADE_SUMMARY_COLUMNS

    def _generate_trial_sequence(self):
        """Generate randomized trial sequence with balanced directions."""
        n_left = self.config.n_trials // 2
        n_right = self.config.n_trials - n_left
        self.trial_directions = (['left'] * n_left) + (['right'] * n_right)
        random.shuffle(self.trial_directions)

    @staticmethod
    def _finite(vals: List[float]) -> np.ndarray:
        """Return numpy array filtered to finite values."""
        arr = np.array(vals, dtype=float)
        return arr[np.isfinite(arr)]

    @staticmethod
    def _smooth_1d(x: np.ndarray, k: int = 5) -> np.ndarray:
        """Simple moving average smoothing (same length)."""
        if x.size < k or k < 2:
            return x
        kernel = np.ones(k, dtype=float) / float(k)
        return np.convolve(x, kernel, mode="same")

    @staticmethod
    def _first_stable_landing_idx(
        gx: np.ndarray,
        gy: np.ndarray,
        onset_idx: int,
        search_end: int,
        target_x: float,
        target_y: float,
        landing_radius_px: float,
        stable_frames: int = 2
    ) -> Optional[int]:
        """
        Return first index where gaze stays within landing_radius for stable_frames.
        If not found, return index of closest approach to target within window.
        """
        if onset_idx >= search_end:
            return None

        streak = 0
        best_i: Optional[int] = None
        best_d = np.inf

        for i in range(onset_idx, search_end):
            dx = gx[i] - target_x
            dy = gy[i] - target_y
            d = float(np.sqrt(dx * dx + dy * dy))

            # closest-approach fallback
            if d < best_d:
                best_d = d
                best_i = i

            # stable landing condition
            if d <= landing_radius_px:
                streak += 1
                if streak >= stable_frames:
                    return i
            else:
                streak = 0

        return best_i

    def run_task(self) -> TaskResult:
        """Run the saccade task."""
        result = TaskResult()

        # Generate trial sequence
        self._generate_trial_sequence()

        # Calculate target positions
        center_x = self.config.center_x_norm * self.config.screen_width
        center_y = self.config.target_y_norm * self.config.screen_height
        left_x = self.config.left_target_x_norm * self.config.screen_width
        right_x = self.config.right_target_x_norm * self.config.screen_width

        # Initialize loggers
        self.frame_logger = FrameLogger(
            f"{self.session_dir}/frame_log_{self.TASK_NAME}.csv",
            self.config.session_id,
            self.TASK_NAME
        )
        self.summary_logger = SummaryLogger(
            f"{self.session_dir}/summary_{self.TASK_NAME}.csv",
            self.config.session_id,
            self.TASK_NAME,
            self.get_summary_columns()
        )

        result.frame_log_csv_path = self.frame_logger.filepath
        result.summary_csv_path = self.summary_logger.filepath

        self.running = True
        trial_summaries: List[Dict[str, Any]] = []

        for trial_idx in range(self.config.n_trials):
            if not self.ui.running or not self.running:
                break

            self.current_trial = trial_idx + 1
            self.clear_trial_buffer()

            direction = self.trial_directions[trial_idx]
            target_x = left_x if direction == 'left' else right_x

            trial_data = self._run_trial(
                trial_idx, direction, center_x, center_y, target_x
            )

            if trial_data:
                self.summary_logger.log_trial(trial_data)
                trial_summaries.append(trial_data)

                self.debug_logger.info(
                    f"Trial {trial_idx + 1} ({direction}): "
                    f"valid={trial_data.get('valid', 0)}, "
                    f"lat={trial_data.get('saccade_latency_ms', np.nan):.1f}ms, "
                    f"amp_abs={trial_data.get('saccade_amplitude_px', np.nan):.1f}px, "
                    f"amp_signed={trial_data.get('saccade_amplitude_signed_px', np.nan):.1f}px, "
                    f"land_err={trial_data.get('landing_error_px', np.nan):.1f}px"
                )

        # Aggregate results
        result.n_trials = len(trial_summaries)
        result.n_valid_trials = sum(1 for t in trial_summaries if t.get('valid', 0) == 1)
        result.valid_rate = (result.n_valid_trials / result.n_trials) if result.n_trials > 0 else 0.0

        # Direction-specific warnings
        left_trials = [t for t in trial_summaries if t.get('direction') == 'left']
        right_trials = [t for t in trial_summaries if t.get('direction') == 'right']
        left_valid = sum(1 for t in left_trials if t.get('valid', 0) == 1)
        right_valid = sum(1 for t in right_trials if t.get('valid', 0) == 1)

        if left_trials and (left_valid / len(left_trials)) < 0.5:
            msg = f"Low validity for leftward trials: {left_valid}/{len(left_trials)}"
            result.warnings.append(msg)
            self.debug_logger.warning(msg)

        if right_trials and (right_valid / len(right_trials)) < 0.5:
            msg = f"Low validity for rightward trials: {right_valid}/{len(right_trials)}"
            result.warnings.append(msg)
            self.debug_logger.warning(msg)

        # ---- FIX 3: aggregate ONLY valid AND finite ----
        valid_trials = [t for t in trial_summaries if t.get('valid', 0) == 1]
        if valid_trials:
            lat = self._finite([t.get('saccade_latency_ms', np.nan) for t in valid_trials])
            dur = self._finite([t.get('saccade_duration_ms', np.nan) for t in valid_trials])
            pvel = self._finite([t.get('peak_velocity_px_s', np.nan) for t in valid_trials])

            # ---- FIX 1: ABS amplitude for summary ----
            amp_abs = self._finite([t.get('saccade_amplitude_px', np.nan) for t in valid_trials])

            gain = self._finite([t.get('gain', np.nan) for t in valid_trials])
            land = self._finite([t.get('landing_error_px', np.nan) for t in valid_trials])

            result.biomarkers = {
                'mean_latency_ms': float(np.nanmean(lat)) if lat.size else np.nan,
                'mean_duration_ms': float(np.nanmean(dur)) if dur.size else np.nan,
                'mean_peak_velocity': float(np.nanmean(pvel)) if pvel.size else np.nan,
                'mean_amplitude_px': float(np.nanmean(amp_abs)) if amp_abs.size else np.nan,
                'mean_gain': float(np.nanmean(gain)) if gain.size else np.nan,
                'mean_landing_error_px': float(np.nanmean(land)) if land.size else np.nan,
            }

        qs = self._finite([t.get('quality_score', 0.0) for t in trial_summaries])
        result.mean_quality_score = float(np.nanmean(qs)) if qs.size else 0.0

        # Show results + save metadata
        self._show_results(result)
        self._save_metadata(result)

        result.success = True
        result.session_json_path = f"{self.session_dir}/meta.json"
        return result

    def _run_trial(
        self,
        trial_idx: int,
        direction: str,
        center_x: float,
        center_y: float,
        target_x: float
    ) -> Optional[Dict[str, Any]]:
        """Run a single saccade trial."""
        trial_start = time.time()

        foreperiod = random.uniform(self.config.foreperiod_min, self.config.foreperiod_max)

        post_gaze_x: List[float] = []
        post_gaze_y: List[float] = []
        post_vel_x: List[float] = []
        post_vel_y: List[float] = []
        post_timestamps: List[float] = []

        state = SaccadeState.FORE
        fore_end = trial_start + foreperiod

        # === FOREPERIOD ===
        while time.time() < fore_end and self.ui.running:
            current_time = time.time()

            events = self.ui.process_events()
            if events['quit']:
                self.running = False
                return None
            if events['key_s']:
                return self._create_skipped_trial(trial_idx, direction, center_x, center_y, target_x, trial_start)

            ret, frame = self.cap.read()
            if not ret:
                continue

            frame_data = self.process_frame_with_mapping(frame, current_time)
            self.log_frame(trial_idx, state.value, center_x, center_y, frame_data)

            self.ui.clear_screen()
            self.ui.draw_target(center_x, center_y)
            self.ui.update_quality_indicators(
                face_detected=frame_data.get('face_present', 0) == 1,
                fps=frame_data.get('fps_est', 30.0),
                quality_score=frame_data.get('valid_sample', 0),
                trial_count=self.current_trial,
                total_trials=self.config.n_trials
            )
            self.ui.draw_quality_indicators()
            self.ui.update_display()
            self.ui.tick(60)

        # ---- target onset timestamp ----
        jump_time = time.time()
        state = SaccadeState.POST

        baseline_gaze_x: Optional[float] = None
        baseline_gaze_y: Optional[float] = None

        post_end = jump_time + self.config.post_duration
        dropout_in_critical = False
        critical_end = jump_time + (self.config.critical_window_ms / 1000.0)

        # === POST (record saccade) ===
        while time.time() < post_end and self.ui.running:
            current_time = time.time()

            events = self.ui.process_events()
            if events['quit']:
                self.running = False
                return None
            if events['key_s']:
                return self._create_skipped_trial(trial_idx, direction, center_x, center_y, target_x, trial_start)

            ret, frame = self.cap.read()
            if not ret:
                continue

            frame_data = self.process_frame_with_mapping(frame, current_time)
            self.log_frame(trial_idx, state.value, target_x, center_y, frame_data)

            if current_time < critical_end and frame_data.get('face_present', 0) == 0:
                dropout_in_critical = True

            if frame_data.get('valid_sample', 0) == 1:
                gx = frame_data.get('gaze_x_px_comp', np.nan)
                gy = frame_data.get('gaze_y_px_comp', np.nan)

                if baseline_gaze_x is None and np.isfinite(gx):
                    baseline_gaze_x = float(gx)
                if baseline_gaze_y is None and np.isfinite(gy):
                    baseline_gaze_y = float(gy)

                post_gaze_x.append(float(gx) if np.isfinite(gx) else np.nan)
                post_gaze_y.append(float(gy) if np.isfinite(gy) else np.nan)
                post_vel_x.append(float(frame_data.get('vel_x_px_s', np.nan)))
                post_vel_y.append(float(frame_data.get('vel_y_px_s', np.nan)))
                post_timestamps.append(float(current_time))

            self.ui.clear_screen()
            self.ui.draw_target(target_x, center_y)
            self.ui.update_quality_indicators(
                face_detected=frame_data.get('face_present', 0) == 1,
                fps=frame_data.get('fps_est', 30.0),
                quality_score=frame_data.get('valid_sample', 0),
                trial_count=self.current_trial,
                total_trials=self.config.n_trials
            )
            self.ui.draw_quality_indicators()
            self.ui.update_display()
            self.ui.tick(60)

        trial_end = time.time()

        trial_data = self._compute_biomarkers(
            trial_idx, direction, center_x, center_y, target_x,
            trial_start, trial_end, jump_time,
            post_gaze_x, post_gaze_y, post_vel_x, post_vel_y, post_timestamps,
            baseline_gaze_x, baseline_gaze_y,
            dropout_in_critical
        )

        time.sleep(self.config.inter_trial_interval)
        return trial_data

    def _compute_biomarkers(
        self,
        trial_idx: int,
        direction: str,
        center_x: float,
        center_y: float,
        target_x: float,
        trial_start: float,
        trial_end: float,
        jump_time: float,
        gaze_x: List[float],
        gaze_y: List[float],
        vel_x: List[float],
        vel_y: List[float],
        timestamps: List[float],
        baseline_gaze_x: Optional[float],
        baseline_gaze_y: Optional[float],
        dropout_in_critical: bool
    ) -> Dict[str, Any]:
        """Compute saccade biomarkers from collected samples."""
        generic_validity = self.compute_trial_validity()

        gaze_x = np.array(gaze_x, dtype=float)
        gaze_y = np.array(gaze_y, dtype=float)
        vel_x = np.array(vel_x, dtype=float)
        vel_y = np.array(vel_y, dtype=float)
        timestamps = np.array(timestamps, dtype=float)

        # ---- FIX: finite-only filter ----
        finite_mask = np.isfinite(gaze_x) & np.isfinite(gaze_y) & np.isfinite(timestamps)
        gaze_x = gaze_x[finite_mask]
        gaze_y = gaze_y[finite_mask]
        vel_x = vel_x[finite_mask]
        vel_y = vel_y[finite_mask]
        timestamps = timestamps[finite_mask]

        eccentricity_x = target_x - center_x
        eccentricity = eccentricity_x  # checker expects scalar px

        latency_ms = np.nan
        duration_ms = np.nan
        peak_velocity = np.nan

        amp_signed = np.nan
        amp_abs = np.nan
        gain = np.nan

        landing_error = np.nan
        overshoot = 0.0
        undershoot = 0.0
        corrective_count = 0
        direction_correct = False

        if len(gaze_x) >= 3 and len(timestamps) >= 3:
            vel_mag = np.sqrt(vel_x ** 2 + vel_y ** 2)
            vel_x_abs = np.abs(vel_x)

            baseline_x = float(baseline_gaze_x) if baseline_gaze_x is not None else float(gaze_x[0])

            onset_idx = detect_saccade_onset_robust(
                gaze_x, vel_x_abs, timestamps,
                jump_time=jump_time,
                baseline_gaze_x=baseline_x,
                eccentricity=eccentricity,
                velocity_threshold=self.config.velocity_threshold,
                min_latency_ms=self.config.latency_min_ms,
                min_duration_samples=self.config.min_velocity_samples
            )

            if onset_idx is not None and onset_idx < len(timestamps):
                onset_time = timestamps[onset_idx]
                onset_gx = gaze_x[onset_idx]
                onset_gy = gaze_y[onset_idx]

                latency_ms = (onset_time - jump_time) * 1000.0

                # Window: up to 400ms after onset (~12 samples @ 30fps)
                max_duration_samples = int(0.4 * 30)
                search_end = min(onset_idx + max_duration_samples, len(gaze_x))

                # ---- FIX: smooth gaze for landing robustness ----
                gx_s = self._smooth_1d(gaze_x, k=5)
                gy_s = self._smooth_1d(gaze_y, k=5)

                # ---- FIX: stable landing OR closest-approach fallback ----
                landing_radius = max(80.0, 0.18 * abs(eccentricity))  # webcam-friendly
                landing_idx = self._first_stable_landing_idx(
                    gx=gx_s,
                    gy=gy_s,
                    onset_idx=onset_idx,
                    search_end=search_end,
                    target_x=target_x,
                    target_y=center_y,
                    landing_radius_px=landing_radius,
                    stable_frames=2
                )
                if landing_idx is None:
                    landing_idx = onset_idx

                # Peak velocity within onset->landing window (95th percentile for noise robustness)
                if landing_idx > onset_idx:
                    vel_window = vel_mag[onset_idx:landing_idx + 1]
                    if vel_window.size:
                        peak_velocity = float(np.percentile(vel_window, 95))
                    else:
                        peak_vel, _ = find_peak_velocity(vel_mag, onset_idx, len(vel_mag))
                        peak_velocity = float(peak_vel)
                else:
                    peak_vel, _ = find_peak_velocity(vel_mag, onset_idx, len(vel_mag))
                    peak_velocity = float(peak_vel)

                if landing_idx < len(timestamps):
                    landing_time = timestamps[landing_idx]
                    land_gx = gx_s[landing_idx]
                    land_gy = gy_s[landing_idx]

                    duration_ms = (landing_time - onset_time) * 1000.0

                    # ---- FIX 1: signed + abs amplitude ----
                    amp_signed = float(land_gx - onset_gx)
                    amp_abs = float(abs(amp_signed))

                    expected_sign = np.sign(eccentricity) if eccentricity != 0 else 0.0
                    actual_sign = np.sign(amp_signed) if amp_signed != 0 else 0.0
                    direction_correct = (expected_sign == actual_sign) if expected_sign != 0 else True

                    if abs(eccentricity) > 1:
                        gain = float(amp_signed / eccentricity)

                    # ---- FIX 2: Euclidean landing error to target (mapped, smoothed endpoint) ----
                    dx = float(land_gx - target_x)
                    dy = float(land_gy - center_y)
                    landing_error = float(np.sqrt(dx * dx + dy * dy))

                    # Over/undershoot (x-based)
                    if direction_correct:
                        if abs(land_gx - center_x) > abs(eccentricity):
                            overshoot = float(abs(land_gx - target_x))
                        else:
                            undershoot = float(abs(land_gx - target_x))
                    else:
                        undershoot = float(abs(eccentricity))

                    # Corrective saccades: count velocity-threshold crossings after landing
                    if landing_idx < len(vel_x_abs) - 1:
                        post_landing_vel = vel_x_abs[landing_idx + 1:]
                        in_sacc = False
                        for v in post_landing_vel:
                            if v > self.config.velocity_threshold and not in_sacc:
                                corrective_count += 1
                                in_sacc = True
                            elif v < self.config.velocity_threshold * 0.5:
                                in_sacc = False

        # Validity checker uses signed amplitude for direction correctness logic
        saccade_validity = self.validity_checker.check_saccade_validity(
            latency_ms=latency_ms,
            duration_ms=duration_ms,
            amplitude_px=amp_signed if np.isfinite(amp_signed) else 0.0,
            eccentricity_px=eccentricity,
            peak_velocity=peak_velocity if np.isfinite(peak_velocity) else 0.0,
            direction_correct=direction_correct,
            dropout_in_critical_window=dropout_in_critical,
            generic_validity=generic_validity
        )
        self.trial_validities.append(saccade_validity)

        calib_quality = self.calibrator.get_quality_score() if self.calibrator else 0.0

        return {
            'trial_id': trial_idx,
            'trial_start_s': trial_start,
            'trial_end_s': trial_end,
            'target_x': target_x,
            'target_y': center_y,
            'direction': direction,
            'eccentricity_px': eccentricity,
            'saccade_latency_ms': latency_ms,
            'saccade_duration_ms': duration_ms,
            'peak_velocity_px_s': peak_velocity,

            # FIX 1 outputs
            'saccade_amplitude_px': amp_abs,                 # ABS (ML-safe)
            'saccade_amplitude_signed_px': amp_signed,       # signed (diagnostic)

            'gain': gain,

            # FIX 2 output
            'landing_error_px': landing_error,

            'overshoot_px': overshoot,
            'undershoot_px': undershoot,
            'corrective_saccade_count': corrective_count,

            'valid_fraction': generic_validity.valid_fraction,
            'valid': 1 if saccade_validity.valid else 0,
            'invalid_reason': saccade_validity.reason.value,
            'quality_score': saccade_validity.quality_score,
            'calibration_quality': calib_quality,
            'fps_median': generic_validity.fps_median,
            'clamp_rate': generic_validity.clamp_rate
        }

    def _create_skipped_trial(
        self,
        trial_idx: int,
        direction: str,
        center_x: float,
        center_y: float,
        target_x: float,
        trial_start: float
    ) -> Dict[str, Any]:
        """Create a skipped trial record."""
        return {
            'trial_id': trial_idx,
            'trial_start_s': trial_start,
            'trial_end_s': time.time(),
            'target_x': target_x,
            'target_y': center_y,
            'direction': direction,
            'eccentricity_px': target_x - center_x,
            'saccade_latency_ms': np.nan,
            'saccade_duration_ms': np.nan,
            'peak_velocity_px_s': np.nan,
            'saccade_amplitude_px': np.nan,
            'saccade_amplitude_signed_px': np.nan,
            'gain': np.nan,
            'landing_error_px': np.nan,
            'overshoot_px': np.nan,
            'undershoot_px': np.nan,
            'corrective_saccade_count': 0,
            'valid_fraction': 0.0,
            'valid': 0,
            'invalid_reason': InvalidReason.SKIPPED.value,
            'quality_score': 0.0,
            'calibration_quality': 0.0,
            'fps_median': 0.0,
            'clamp_rate': 0.0
        }

    def _show_results(self, result: TaskResult):
        """Show task results screen."""
        metrics = {
            'Valid Trials': f"{result.n_valid_trials}/{result.n_trials}",
            'Mean Latency (ms)': result.biomarkers.get('mean_latency_ms', np.nan),
            'Mean Duration (ms)': result.biomarkers.get('mean_duration_ms', np.nan),
            'Mean Peak Velocity (px/s)': result.biomarkers.get('mean_peak_velocity', np.nan),
            'Mean Amplitude (px)': result.biomarkers.get('mean_amplitude_px', np.nan),
            'Mean Gain': result.biomarkers.get('mean_gain', np.nan),
            'Mean Landing Error (px)': result.biomarkers.get('mean_landing_error_px', np.nan),
        }

        self.ui.draw_results("Saccade Task Results", metrics, result.valid_rate)

        while self.ui.running:
            events = self.ui.process_events()
            if events['quit'] or events['space']:
                break
            self.ui.tick(60)

    def _save_metadata(self, result: TaskResult):
        """Save session metadata."""
        from core.logging import SessionMetadata

        meta = SessionMetadata.create(
            self.config.session_id,
            self.config.screen_width,
            self.config.screen_height,
            config={
                'task': self.TASK_NAME,
                'n_trials': self.config.n_trials,
                'left_target_x_norm': self.config.left_target_x_norm,
                'right_target_x_norm': self.config.right_target_x_norm,
                'foreperiod_range': [self.config.foreperiod_min, self.config.foreperiod_max],
                'post_duration': self.config.post_duration,
                'velocity_threshold': self.config.velocity_threshold,
            }
        )

        if self.calibration:
            meta.calibration_accepted = self.calibration.accepted
            meta.calibration_reason = self.calibration.reason.value
            meta.calibration_error_mean_px = self.calibration.mean_error_px
            meta.calibration_error_max_px = self.calibration.max_error_px

        meta.valid_rates[self.TASK_NAME] = result.valid_rate

        all_fps = [f.get('fps_est', 30.0) for f in self.trial_frames]
        if all_fps:
            meta.fps_mean = float(np.mean(all_fps))
            meta.fps_median = float(np.median(all_fps))
            meta.fps_min = float(np.min(all_fps))
            meta.fps_max = float(np.max(all_fps))

        meta.save(f"{self.session_dir}/meta.json")
