"""
Anti-Saccade Task for NeuroLens+ eye tracking system.

FIXES APPLIED:
1) Landing error:
   - Now computed as Euclidean distance from mapped gaze endpoint (x,y) to the CORRECT target (x,y).
   - No clamps/caps. (This removes the repeated 273.6 / 272.6 artifact.)
   - Uses endpoint sample at landing_idx (mapped gaze), not raw.

2) Direction error + inhibition success:
   - Detect the FIRST saccade after stimulus onset (either direction) using a direction-agnostic onset.
   - Classify direction_error by comparing initial signed displacement vs stimulus side.
   - inhibition_success = 1 iff direction_error == 0.

3) Correction time:
   - If direction_error==1, find time from wrong landing to first time gaze moves into correct hemifield
     (and toward the correct target) with a displacement threshold.
   - Outputs correction_time_ms only for error trials.

4) Amplitude:
   - Stores BOTH signed and absolute amplitude:
       amplitude_signed_px (directional)
       amplitude_px        (absolute; ML-safe)
   - Keeps existing summary columns unchanged (amplitude_px stays, now ABS).

NOTES:
- ANTISACCADE_SUMMARY_COLUMNS currently contains "amplitude_px". This file keeps that column but
  changes its meaning to ABS amplitude (ML-safe). If you want to log the signed amplitude too,
  add "amplitude_signed_px" to ANTISACCADE_SUMMARY_COLUMNS.
- Uses gaze_y too for landing error (requires logging/collecting gaze_y in POST).
"""

import numpy as np
import time
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, replace
from enum import Enum

from .base import BaseTask, TaskConfig, TaskResult
from core.logging import FrameLogger, SummaryLogger, ANTISACCADE_SUMMARY_COLUMNS
from core.validity import InvalidReason
from core.utils import find_peak_velocity, detect_saccade_onset_robust


class AntisaccadeState(Enum):
    """Antisaccade trial state machine states."""
    FORE = "FORE"          # Foreperiod - fixation at center
    STIMULUS = "STIMULUS"  # Stimulus appears (look away!)
    POST = "POST"          # Post-response recording


# Latency validity constants (override config values for validity checks)
STIMULUS_ONSET_GRACE_MS = 50.0
MIN_LATENCY_MS = 110.0
MAX_LATENCY_MS = 1500.0


@dataclass
class AntisaccadeConfig(TaskConfig):
    """Configuration for antisaccade task."""

    # Trial settings
    n_trials: int = 20

    # Target positions
    stimulus_x_norm_left: float = 0.2
    stimulus_x_norm_right: float = 0.8
    center_x_norm: float = 0.5
    target_y_norm: float = 0.5

    # Timing
    foreperiod_min: float = 1.2
    foreperiod_max: float = 2.8
    post_duration: float = 1.5
    inter_trial_interval: float = 0.5

    # Detection thresholds (webcam)
    velocity_threshold: float = 220.0  # raise from 150 for webcam noise
    min_velocity_samples: int = 2
    min_displacement_px: float = 45.0
    displacement_threshold_fraction: float = 0.10

    # Windows
    direction_error_window_ms: float = 200.0  # initial error window after stimulus
    correction_window_s: float = 1.0          # search window for correction after wrong landing

    # Validity thresholds
    latency_min_ms: float = 80.0
    latency_max_ms: float = 1000.0


class AntisaccadeTask(BaseTask):
    """
    Anti-Saccade Task - Measure inhibitory control.

    Participant must look AWAY from the stimulus (opposite direction).
    """

    TASK_NAME = "antisaccade"

    def __init__(self, config: Optional[AntisaccadeConfig] = None):
        super().__init__(config or AntisaccadeConfig())
        self.config: AntisaccadeConfig = self.config
        self.trial_directions: List[str] = []

    def get_instructions(self) -> List[str]:
        return [
            "ANTI-SACCADE TASK",
            "",
            "A white dot will appear in the center.",
            "Keep your eyes on the center dot.",
            "",
            "When a dot appears on the LEFT or RIGHT,",
            "look to the OPPOSITE side as quickly as possible.",
            "",
            "Example: If dot appears on LEFT, look RIGHT.",
            "",
            f"There will be {self.config.n_trials} trials.",
            "",
            "Press SPACE to start.",
            "Press Q or ESC to quit."
        ]

    def get_summary_columns(self) -> List[str]:
        return ANTISACCADE_SUMMARY_COLUMNS

    def _generate_trial_sequence(self):
        n_left = self.config.n_trials // 2
        n_right = self.config.n_trials - n_left
        self.trial_directions = ['left'] * n_left + ['right'] * n_right
        random.shuffle(self.trial_directions)

    @staticmethod
    def _finite(vals: List[float]) -> np.ndarray:
        arr = np.array(vals, dtype=float)
        return arr[np.isfinite(arr)]

    def run_task(self) -> TaskResult:
        result = TaskResult()

        self._generate_trial_sequence()

        center_x = self.config.center_x_norm * self.config.screen_width
        center_y = self.config.target_y_norm * self.config.screen_height
        stim_left_x = self.config.stimulus_x_norm_left * self.config.screen_width
        stim_right_x = self.config.stimulus_x_norm_right * self.config.screen_width

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
            stimulus_x = stim_left_x if direction == 'left' else stim_right_x
            target_x = stim_right_x if direction == 'left' else stim_left_x  # correct target is opposite

            trial_data = self._run_trial(
                trial_idx, direction,
                center_x, center_y, stimulus_x, target_x
            )

            if trial_data:
                self.summary_logger.log_trial(trial_data)
                trial_summaries.append(trial_data)

                self.debug_logger.info(
                    f"Trial {trial_idx + 1} ({direction}): "
                    f"valid={trial_data.get('valid', 0)}, "
                    f"dir_error={trial_data.get('direction_error', 0)}, "
                    f"lat={trial_data.get('antisaccade_latency_ms', np.nan):.1f}ms, "
                    f"corr={trial_data.get('correction_time_ms', np.nan)}"
                )

        result.n_trials = len(trial_summaries)
        result.n_valid_trials = sum(1 for t in trial_summaries if t.get('valid', 0) == 1)
        result.valid_rate = (result.n_valid_trials / result.n_trials) if result.n_trials > 0 else 0.0

        # ---- Aggregate ONLY valid + finite ----
        valid_trials = [t for t in trial_summaries if t.get('valid', 0) == 1]
        if valid_trials:
            lat = self._finite([t.get('antisaccade_latency_ms', np.nan) for t in valid_trials])
            inh = np.array([t.get('inhibition_success', 0) for t in valid_trials], dtype=float)
            derr = np.array([t.get('direction_error', 0) for t in valid_trials], dtype=float)

            # correction time only for error trials
            corr = self._finite([
                t.get('correction_time_ms', np.nan)
                for t in valid_trials
                if t.get('direction_error', 0) == 1
            ])

            result.biomarkers = {
                'mean_latency_ms': float(np.nanmean(lat)) if lat.size else np.nan,
                'inhibition_success_rate': float(np.nanmean(inh)) if inh.size else 0.0,
                'direction_error_rate': float(np.nanmean(derr)) if derr.size else 1.0,
                'mean_correction_time_ms': float(np.nanmean(corr)) if corr.size else np.nan,
            }

        qs = self._finite([t.get('quality_score', 0.0) for t in trial_summaries])
        result.mean_quality_score = float(np.nanmean(qs)) if qs.size else 0.0

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
        stimulus_x: float,
        target_x: float
    ) -> Optional[Dict[str, Any]]:
        trial_start = time.time()
        foreperiod = random.uniform(self.config.foreperiod_min, self.config.foreperiod_max)

        # Data collection (POST)
        post_gaze_x: List[float] = []
        post_gaze_y: List[float] = []
        post_vel_x: List[float] = []
        post_vel_y: List[float] = []
        post_timestamps: List[float] = []

        # Baseline gaze during FORE
        fore_gaze_x: List[float] = []
        fore_gaze_y: List[float] = []

        state = AntisaccadeState.FORE
        fore_end = trial_start + foreperiod

        # === FOREPERIOD ===
        while time.time() < fore_end and self.ui.running:
            current_time = time.time()

            events = self.ui.process_events()
            if events['quit']:
                self.running = False
                return None
            if events.get('key_s', False):
                return self._create_skipped_trial(trial_idx, direction, center_x, center_y, stimulus_x, target_x, trial_start)

            ret, frame = self.cap.read()
            if not ret:
                continue

            frame_data = self.process_frame_with_mapping(frame, current_time)
            self.log_frame(trial_idx, state.value, center_x, center_y, frame_data)

            if frame_data.get('valid_sample', 0) == 1:
                fore_gaze_x.append(frame_data.get('gaze_x_px_comp', np.nan))
                fore_gaze_y.append(frame_data.get('gaze_y_px_comp', np.nan))

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

        # === STIMULUS ONSET ===
        state = AntisaccadeState.POST  # directly into POST collection
        stimulus_time = None
        stimulus_time_set = False
        post_end = None

        while self.ui.running:
            current_time = time.time()
            
            if post_end is None:
                post_end = current_time + self.config.post_duration
            elif current_time >= post_end:
                break

            events = self.ui.process_events()
            if events['quit']:
                self.running = False
                return None
            if events.get('key_s', False):
                return self._create_skipped_trial(trial_idx, direction, center_x, center_y, stimulus_x, target_x, trial_start)

            ret, frame = self.cap.read()
            if not ret:
                continue

            # Set stimulus_time on FIRST POST frame after successful cap.read()
            if not stimulus_time_set:
                stimulus_time = current_time
                stimulus_time_set = True
                post_end = stimulus_time + self.config.post_duration

            frame_data = self.process_frame_with_mapping(frame, current_time)
            self.log_frame(trial_idx, state.value, target_x, center_y, frame_data)

            if frame_data.get('valid_sample', 0) == 1:
                post_gaze_x.append(frame_data.get('gaze_x_px_comp', np.nan))
                post_gaze_y.append(frame_data.get('gaze_y_px_comp', np.nan))
                post_vel_x.append(frame_data.get('vel_x_px_s', np.nan))
                post_vel_y.append(frame_data.get('vel_y_px_s', np.nan))
                post_timestamps.append(current_time)

            # Draw stimulus (red) only (participant must look opposite)
            self.ui.clear_screen()
            self.ui.draw_target(stimulus_x, center_y, color=(255, 100, 100))
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
        
        # Ensure stimulus_time is set (should be, but safety check)
        if stimulus_time is None:
            stimulus_time = trial_end

        baseline_x = float(np.nanmean(fore_gaze_x)) if fore_gaze_x else float(center_x)
        baseline_y = float(np.nanmean(fore_gaze_y)) if fore_gaze_y else float(center_y)

        trial_data = self._compute_biomarkers(
            trial_idx, direction,
            center_x, center_y, stimulus_x, target_x,
            trial_start, trial_end, stimulus_time,
            post_gaze_x, post_gaze_y, post_vel_x, post_vel_y, post_timestamps,
            baseline_x, baseline_y
        )

        time.sleep(self.config.inter_trial_interval)
        return trial_data

    def _compute_biomarkers(
        self,
        trial_idx: int,
        direction: str,
        center_x: float,
        center_y: float,
        stimulus_x: float,
        target_x: float,
        trial_start: float,
        trial_end: float,
        stimulus_time: float,
        gaze_x: List[float],
        gaze_y: List[float],
        vel_x: List[float],
        vel_y: List[float],
        timestamps: List[float],
        baseline_gaze_x: float,
        baseline_gaze_y: float
    ) -> Dict[str, Any]:
        generic_validity = self.compute_trial_validity()

        gx = np.array(gaze_x, dtype=float)
        gy = np.array(gaze_y, dtype=float)
        vx = np.array(vel_x, dtype=float)
        vy = np.array(vel_y, dtype=float)
        ts = np.array(timestamps, dtype=float)

        # Stimulus side sign (+ right, - left)
        stim_sign = float(np.sign(stimulus_x - center_x)) if stimulus_x != center_x else 0.0
        # Correct antisaccade direction is opposite
        correct_sign = -stim_sign if stim_sign != 0 else 0.0

        # Eccentricity to correct target (used only for thresholding)
        correct_ecc = float(target_x - center_x)

        latency_ms = np.nan
        direction_error = 0
        inhibition_success = 0
        correction_time_ms = np.nan

        peak_velocity = np.nan
        amp_signed = np.nan
        amp_abs = np.nan
        landing_error = np.nan

        if gx.size >= 3 and ts.size >= 3:
            vel_mag = np.sqrt(vx * vx + vy * vy)
            vel_x_abs = np.abs(vx)

            # ---- Detect FIRST saccade onset in either direction (direction-agnostic) ----
            # Use correct_ecc (magnitude) but onset detection itself is robust to direction via displacement thresholds.
            onset_idx = detect_saccade_onset_robust(
                gx,
                vel_x_abs,  # horizontal emphasis
                ts,
                jump_time=stimulus_time,
                baseline_gaze_x=baseline_gaze_x,
                eccentricity=correct_ecc,  # magnitude reference
                velocity_threshold=self.config.velocity_threshold,
                displacement_threshold_fraction=self.config.displacement_threshold_fraction,
                min_displacement_px=self.config.min_displacement_px,
                min_latency_ms=50.0,
                min_duration_samples=self.config.min_velocity_samples
            )

            if onset_idx is not None and onset_idx < ts.size:
                onset_x = gx[onset_idx]
                
                # ---- Persistence guard: reject single-frame noise and micro-movements ----
                PERSIST_FRAMES = 5
                PERSIST_DISP_PX = 30.0
                persist_end = min(onset_idx + PERSIST_FRAMES, gx.size)
                persist_disp = gx[persist_end - 1] - onset_x
                move_sign = float(np.sign(persist_disp)) if abs(persist_disp) >= 8.0 else 0.0
                
                # Reject if movement doesn't persist: sign inconsistent or displacement too small
                if move_sign == 0.0 or abs(persist_disp) < PERSIST_DISP_PX:
                    onset_idx = None
                
            if onset_idx is not None and onset_idx < ts.size:
                # ---- Early-latency confirmation: reject noise-triggered early onsets ----
                candidate_latency_ms = float((ts[onset_idx] - stimulus_time) * 1000.0)
                
                if candidate_latency_ms < 140.0:
                    # For early latencies, require stronger displacement confirmation
                    confirm_end = min(onset_idx + PERSIST_FRAMES, gx.size)
                    confirm_disp = gx[confirm_end - 1] - baseline_gaze_x
                    confirm_sign = float(np.sign(confirm_disp)) if abs(confirm_disp) >= 8.0 else 0.0
                    
                    # Require displacement from baseline in movement direction exceeds 45px
                    if confirm_sign == 0.0 or abs(confirm_disp) < 45.0:
                        onset_idx = None
                
            if onset_idx is not None and onset_idx < ts.size:
                onset_t = ts[onset_idx]
                latency_ms = float((onset_t - stimulus_time) * 1000.0)

                # Find initial landing within ~400ms after onset using displacement in whichever direction is taken
                max_duration_samples = int(0.4 * 30)
                search_end = min(onset_idx + max_duration_samples, gx.size)

                onset_x = gx[onset_idx]
                onset_y = gy[onset_idx] if onset_idx < gy.size else np.nan

                # Determine initial movement sign from early displacement (first few frames)
                # Require abs(initial_dx) >= 12 px to avoid tiny noise triggering detection
                early_end = min(onset_idx + 2, gx.size - 1)
                initial_dx = gx[early_end] - onset_x if early_end > onset_idx else 0.0
                move_sign = float(np.sign(initial_dx)) if abs(initial_dx) >= 12.0 else 0.0

                # If move_sign is 0 due to noise or tiny displacement, choose sign of max displacement from onset in window
                if move_sign == 0.0:
                    disp = gx[onset_idx:search_end] - onset_x
                    if disp.size:
                        move_sign = float(np.sign(disp[np.argmax(np.abs(disp))])) if np.any(np.isfinite(disp)) else 0.0

                # If still unknown, assume correct_sign (rare)
                if move_sign == 0.0:
                    move_sign = correct_sign if correct_sign != 0 else 1.0

                # Landing = max displacement in the movement direction
                max_disp = 0.0
                landing_idx = onset_idx
                for i in range(onset_idx, search_end):
                    disp_i = (gx[i] - onset_x) * move_sign
                    if disp_i > max_disp:
                        max_disp = float(disp_i)
                        landing_idx = i

                # Peak velocity (95th percentile in onset->landing)
                if landing_idx > onset_idx:
                    vel_window = vel_mag[onset_idx:landing_idx + 1]
                    if vel_window.size:
                        peak_velocity = float(np.percentile(vel_window, 95))
                    else:
                        pv, _ = find_peak_velocity(vel_mag, onset_idx, vel_mag.size)
                        peak_velocity = float(pv)
                else:
                    pv, _ = find_peak_velocity(vel_mag, onset_idx, vel_mag.size)
                    peak_velocity = float(pv)

                land_x = gx[landing_idx]
                land_y = gy[landing_idx] if landing_idx < gy.size else np.nan
                land_t = ts[landing_idx]

                # ---- Direction error classification (first saccade only) ----
                # Initial signed displacement from onset to landing
                dx0 = float(land_x - onset_x) if np.isfinite(land_x) else 0.0
                first_sign = float(np.sign(dx0)) if dx0 != 0 else move_sign

                # Error if first saccade goes TOWARD stimulus side
                direction_error = 1 if (stim_sign != 0.0 and first_sign == stim_sign) else 0
                inhibition_success = 1 if direction_error == 0 else 0

                # ---- Amplitude (signed + abs) ----
                amp_signed = float(dx0)
                amp_abs = float(abs(amp_signed))

                # ---- Landing error to CORRECT target (always) ----
                # FIX: Euclidean distance from mapped endpoint to correct target point
                if np.isfinite(land_x) and np.isfinite(land_y):
                    dx = float(land_x - target_x)
                    dy = float(land_y - center_y)
                    landing_error = float(np.sqrt(dx * dx + dy * dy))
                elif np.isfinite(land_x):
                    landing_error = float(abs(land_x - target_x))

                # ---- Correction time (only if error) ----
                if direction_error == 1:
                    # Search after wrong landing for entry into correct hemifield / movement toward correct target
                    # Condition: displacement from baseline in correct direction exceeds threshold.
                    corr_deadline = land_t + float(self.config.correction_window_s)
                    thresh = float(self.config.min_displacement_px)

                    for i in range(landing_idx + 1, gx.size):
                        if ts[i] > corr_deadline:
                            break
                        if not np.isfinite(gx[i]):
                            continue
                        disp_correct = (gx[i] - baseline_gaze_x) * correct_sign
                        if disp_correct > thresh:
                            correction_time_ms = float((ts[i] - land_t) * 1000.0)
                            break

        # ---- Latency validity gates (before calling validity_checker) ----
        latency_finite = np.isfinite(latency_ms)
        invalid_reason_local = None
        valid_local = 1
        
        if latency_finite:
            # Compute effective latency (with grace offset) for anticipatory check only
            effective_latency_ms = max(0.0, latency_ms - STIMULUS_ONSET_GRACE_MS)
            
            # Check anticipatory (using effective latency)
            if effective_latency_ms < MIN_LATENCY_MS:
                invalid_reason_local = InvalidReason.ANTICIPATORY_RESPONSE
                valid_local = 0
            # Check too slow (using raw latency)
            elif latency_ms > MAX_LATENCY_MS:
                invalid_reason_local = InvalidReason.LATENCY_OUT_OF_RANGE
                valid_local = 0
        else:
            # NaN latency - do not run latency-based checks
            invalid_reason_local = InvalidReason.INSUFFICIENT_SAMPLES
            valid_local = 0

        # Only call validity_checker if passed local latency gates
        if valid_local == 1 and latency_finite:
            antisaccade_validity = self.validity_checker.check_antisaccade_validity(
                latency_ms=latency_ms,  # Pass raw latency (no grace subtraction)
                direction_error=(direction_error == 1),
                correction_time_ms=correction_time_ms,
                generic_validity=generic_validity
            )
        else:
            # Create new validity object with local latency reason override
            antisaccade_validity = replace(
                generic_validity,
                valid=False,
                reason=invalid_reason_local
            )
        
        self.trial_validities.append(antisaccade_validity)

        calib_quality = self.calibrator.get_quality_score() if self.calibrator else 0.0

        result_dict = {
            'trial_id': trial_idx,
            'trial_start_s': trial_start,
            'trial_end_s': trial_end,
            'stimulus_x': stimulus_x,
            'stimulus_y': center_y,
            'target_x': target_x,
            'target_y': center_y,
            'direction': direction,
            'antisaccade_latency_ms': latency_ms,  # Raw latency (no grace subtraction in CSV)
            'direction_error': direction_error,
            'correction_time_ms': correction_time_ms,
            'inhibition_success': inhibition_success,
            'peak_velocity_px_s': peak_velocity,

            # amplitude_px stays in CSV as ABS (ML-safe)
            'amplitude_px': amp_abs,
            # if you add this column in ANTISACCADE_SUMMARY_COLUMNS, it will log too
            'amplitude_signed_px': amp_signed,

            'landing_error_px': landing_error,
            'valid_fraction': generic_validity.valid_fraction,
            'valid': 1 if antisaccade_validity.valid else 0,
            'invalid_reason': antisaccade_validity.reason.value,
            'quality_score': antisaccade_validity.quality_score,
            'calibration_quality': calib_quality,
            'fps_median': generic_validity.fps_median,
            'clamp_rate': generic_validity.clamp_rate
        }
        
        # ---- Debug logging for invalid trials only ----
        if result_dict['valid'] == 0:
            effective_lat = max(0.0, latency_ms - STIMULUS_ONSET_GRACE_MS) if latency_finite else np.nan
            self.debug_logger.info(
                f"INVALID trial {trial_idx}: "
                f"raw_lat={latency_ms:.1f}ms, "
                f"eff_lat={effective_lat:.1f}ms, "
                f"dir_err={direction_error}, "
                f"reason={antisaccade_validity.reason.value}, "
                f"clamp={generic_validity.clamp_rate:.3f}, "
                f"valid_frac={generic_validity.valid_fraction:.3f}"
            )
        
        return result_dict

    def _create_skipped_trial(
        self,
        trial_idx: int,
        direction: str,
        center_x: float,
        center_y: float,
        stimulus_x: float,
        target_x: float,
        trial_start: float
    ) -> Dict[str, Any]:
        return {
            'trial_id': trial_idx,
            'trial_start_s': trial_start,
            'trial_end_s': time.time(),
            'stimulus_x': stimulus_x,
            'stimulus_y': center_y,
            'target_x': target_x,
            'target_y': center_y,
            'direction': direction,
            'antisaccade_latency_ms': np.nan,
            'direction_error': 0,
            'correction_time_ms': np.nan,
            'inhibition_success': 0,
            'peak_velocity_px_s': np.nan,
            'amplitude_px': np.nan,
            'amplitude_signed_px': np.nan,
            'landing_error_px': np.nan,
            'valid_fraction': 0.0,
            'valid': 0,
            'invalid_reason': InvalidReason.SKIPPED.value,
            'quality_score': 0.0,
            'calibration_quality': 0.0,
            'fps_median': 0.0,
            'clamp_rate': 0.0
        }

    def _show_results(self, result: TaskResult):
        metrics = {
            'Valid Trials': f"{result.n_valid_trials}/{result.n_trials}",
            'Mean Latency (ms)': result.biomarkers.get('mean_latency_ms', np.nan),
            'Inhibition Success Rate': f"{result.biomarkers.get('inhibition_success_rate', 0) * 100:.1f}%",
            'Direction Error Rate': f"{result.biomarkers.get('direction_error_rate', 0) * 100:.1f}%",
            'Mean Correction Time (ms)': result.biomarkers.get('mean_correction_time_ms', np.nan),
        }

        self.ui.draw_results("Anti-Saccade Results", metrics, result.valid_rate)

        while self.ui.running:
            events = self.ui.process_events()
            if events['quit'] or events['space']:
                break
            self.ui.tick(60)

    def _save_metadata(self, result: TaskResult):
        from core.logging import SessionMetadata

        meta = SessionMetadata.create(
            self.config.session_id,
            self.config.screen_width,
            self.config.screen_height,
            config={
                'task': self.TASK_NAME,
                'n_trials': self.config.n_trials,
                'velocity_threshold': self.config.velocity_threshold,
                'min_displacement_px': self.config.min_displacement_px,
                'direction_error_window_ms': self.config.direction_error_window_ms,
                'correction_window_s': self.config.correction_window_s,
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


def _sanity_check_latency_rules():
    """
    Sanity check function to verify latency classification rules.
    
    This function tests expected classifications:
    - latency=70ms -> anticipatory (after grace 70-50=20 <110)
    - latency=160ms -> not anticipatory (after grace 160-50=110 >=110, and <=1500)
    - latency=1164ms -> not latency_out_of_range (<=1500)
    - latency=1600ms -> latency_out_of_range (>1500)
    - latency=NaN -> handled separately (not anticipatory, not latency_out_of_range)
    """
    # Test case 1: 70ms -> anticipatory (70 - 50 = 20 < 110)
    lat_70 = 70.0
    eff_70 = max(0.0, lat_70 - STIMULUS_ONSET_GRACE_MS)
    assert eff_70 < MIN_LATENCY_MS, f"70ms should be anticipatory (eff={eff_70} < {MIN_LATENCY_MS})"
    
    # Test case 2: 160ms -> not anticipatory (160 - 50 = 110 >= 110)
    lat_160 = 160.0
    eff_160 = max(0.0, lat_160 - STIMULUS_ONSET_GRACE_MS)
    assert eff_160 >= MIN_LATENCY_MS, f"160ms should NOT be anticipatory (eff={eff_160} >= {MIN_LATENCY_MS})"
    assert lat_160 <= MAX_LATENCY_MS, f"160ms should NOT be out of range (<= {MAX_LATENCY_MS})"
    
    # Test case 3: 1164ms -> not latency_out_of_range (<=1500)
    lat_1164 = 1164.0
    assert lat_1164 <= MAX_LATENCY_MS, f"1164ms should NOT be out of range (<= {MAX_LATENCY_MS})"
    
    # Test case 4: 1600ms -> latency_out_of_range (>1500)
    lat_1600 = 1600.0
    assert lat_1600 > MAX_LATENCY_MS, f"1600ms should be out of range (> {MAX_LATENCY_MS})"
    
    # Test case 5: NaN -> not anticipatory and not latency_out_of_range (handled separately)
    lat_nan = np.nan
    assert not np.isfinite(lat_nan), "NaN latency should be handled separately"
    
    print("✓ All latency rule sanity checks passed")


if __name__ == "__main__":
    _sanity_check_latency_rules()
