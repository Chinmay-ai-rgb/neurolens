"""Fixation Task for NeuroLens+ eye tracking system."""

import numpy as np
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base import BaseTask, TaskConfig, TaskResult
from core.logging import FrameLogger, SummaryLogger, FIXATION_SUMMARY_COLUMNS
from core.validity import InvalidReason
from core.utils import compute_rms, compute_bcea, compute_microsaccade_rate


@dataclass
class FixationConfig(TaskConfig):
    """Configuration for fixation task."""

    # Trial settings
    n_trials: int = 5
    fixation_duration: float = 10.0  # seconds per trial

    # Target settings
    target_x_norm: float = 0.5  # Center
    target_y_norm: float = 0.5

    # On-target thresholding
    target_radius_mode: str = "adaptive"   # "fixed" or "adaptive"
    target_radius_px: float = 100.0        # fallback if fixed
    adaptive_radius_k: float = 2.5         # radius = k * calib_err
    adaptive_radius_min_px: float = 75.0
    adaptive_radius_max_px: float = 300.0

    # Biomarker thresholds
    microsaccade_velocity_threshold: float = 80.0  # px/s (reasonable starting point)


class FixationTask(BaseTask):
    """
    Fixation Task - Measure fixation stability.

    Participant fixates on a central target for extended periods.
    Measures:
    - Fixation stability (RMS)
    - BCEA (Bivariate Contour Ellipse Area)
    - Microsaccade rate
    - Drift velocity (slow component)
    - Percent time on target
    - Blink rate
    """

    TASK_NAME = "fixation"

    def __init__(self, config: Optional[FixationConfig] = None):
        """Initialize fixation task."""
        super().__init__(config or FixationConfig())
        self.config: FixationConfig = self.config

    def get_instructions(self) -> List[str]:
        """Get fixation task instructions."""
        return [
            "FIXATION TASK",
            "",
            "A white dot will appear in the center of the screen.",
            "Keep your eyes fixed on the dot.",
            "Try not to move your eyes or head.",
            "Blink naturally when needed.",
            "",
            f"Each trial lasts {self.config.fixation_duration:.0f} seconds.",
            f"There will be {self.config.n_trials} trials.",
            "",
            "Press SPACE to start.",
            "Press Q or ESC to quit at any time."
        ]

    def get_summary_columns(self) -> List[str]:
        """Get summary CSV columns."""
        return FIXATION_SUMMARY_COLUMNS

    def run_task(self) -> TaskResult:
        """Run the fixation task."""
        result = TaskResult()

        # Calculate target position
        target_x = self.config.target_x_norm * self.config.screen_width
        target_y = self.config.target_y_norm * self.config.screen_height

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
        trial_summaries = []

        for trial_idx in range(self.config.n_trials):
            if not self.ui.running or not self.running:
                break

            self.current_trial = trial_idx + 1
            self.clear_trial_buffer()

            trial_data = self._run_trial(trial_idx, target_x, target_y)

            if trial_data:
                self.summary_logger.log_trial(trial_data)
                trial_summaries.append(trial_data)

                validity = self.trial_validities[-1] if self.trial_validities else None
                if validity:
                    self.debug_logger.info(
                        f"Trial {trial_idx + 1}: valid={validity.valid}, "
                        f"reason={validity.reason.value}, "
                        f"rms={trial_data.get('fixation_stability_rms_px', np.nan):.1f}px, "
                        f"on_target={trial_data.get('percent_time_on_target', np.nan):.1f}%, "
                        f"radius={trial_data.get('target_radius_used_px', np.nan):.0f}px, "
                        f"blink={trial_data.get('blink_rate_per_min', np.nan):.1f}/min"
                    )

        # Aggregate results
        result.n_trials = len(trial_summaries)
        result.n_valid_trials = sum(1 for t in trial_summaries if t.get('valid', 0) == 1)
        result.valid_rate = result.n_valid_trials / result.n_trials if result.n_trials > 0 else 0.0

        if trial_summaries:
            valid_trials = [t for t in trial_summaries if t.get('valid', 0) == 1]
            if valid_trials:
                result.biomarkers = {
                    'mean_rms_px': np.nanmean([t.get('fixation_stability_rms_px', np.nan) for t in valid_trials]),
                    'mean_bcea_px2': np.nanmean([t.get('fixation_bcea_px2', np.nan) for t in valid_trials]),
                    'mean_microsaccade_rate': np.nanmean([t.get('microsaccade_rate_per_min', np.nan) for t in valid_trials]),
                    'mean_drift_velocity': np.nanmean([t.get('drift_velocity_px_s', np.nan) for t in valid_trials]),
                    'mean_percent_on_target': np.nanmean([t.get('percent_time_on_target', np.nan) for t in valid_trials]),
                }

        result.mean_quality_score = np.nanmean([t.get('quality_score', 0) for t in trial_summaries])

        self._show_results(result)
        self._save_metadata(result)

        result.success = True
        result.session_json_path = f"{self.session_dir}/meta.json"
        return result

    def _run_trial(
        self,
        trial_idx: int,
        target_x: float,
        target_y: float
    ) -> Optional[Dict[str, Any]]:
        """Run a single fixation trial."""
        trial_start = time.time()
        trial_end = trial_start + self.config.fixation_duration

        gaze_x_samples: List[float] = []
        gaze_y_samples: List[float] = []
        velocity_samples: List[float] = []
        timestamps: List[float] = []

        # Blink tracking: prefer frame_data blink_flag; fallback to self.trial_frames later
        blink_stream: List[int] = []

        while time.time() < trial_end and self.ui.running:
            current_time = time.time()

            events = self.ui.process_events()
            if events['quit']:
                self.running = False
                return None
            if events['key_s']:
                return self._create_skipped_trial(trial_idx, target_x, target_y, trial_start)

            ret, frame = self.cap.read()
            if not ret:
                continue

            frame_data = self.process_frame_with_mapping(frame, current_time)

            self.log_frame(trial_idx, "FIXATION", target_x, target_y, frame_data)

            # capture blink flag directly from frame_data if present
            blink_stream.append(int(frame_data.get("blink_flag", 0)))

            if frame_data.get('valid_sample', 0) == 1:
                gx = frame_data.get('gaze_x_px_comp', np.nan)
                gy = frame_data.get('gaze_y_px_comp', np.nan)

                gaze_x_samples.append(gx)
                gaze_y_samples.append(gy)
                timestamps.append(current_time)

                # Prefer upstream velocity; fallback to finite-difference from gaze samples
                vel_x = frame_data.get('vel_x_px_s', np.nan)
                vel_y = frame_data.get('vel_y_px_s', np.nan)

                if not np.isnan(vel_x) and not np.isnan(vel_y):
                    velocity_samples.append(float(np.sqrt(vel_x**2 + vel_y**2)))
                else:
                    if len(gaze_x_samples) >= 2:
                        dt = timestamps[-1] - timestamps[-2]
                        if dt > 1e-6:
                            dx = gaze_x_samples[-1] - gaze_x_samples[-2]
                            dy = gaze_y_samples[-1] - gaze_y_samples[-2]
                            velocity_samples.append(float(np.sqrt(dx**2 + dy**2) / dt))

            # UI update
            self.ui.clear_screen()
            self.ui.draw_target(target_x, target_y)

            progress = (current_time - trial_start) / self.config.fixation_duration
            self.ui.draw_progress_bar(
                progress,
                self.config.screen_width // 2,
                self.config.screen_height - 50
            )

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

        trial_data = self._compute_biomarkers(
            trial_idx=trial_idx,
            target_x=target_x,
            target_y=target_y,
            trial_start=trial_start,
            trial_end=time.time(),
            gaze_x=gaze_x_samples,
            gaze_y=gaze_y_samples,
            velocities=velocity_samples,
            timestamps=timestamps,
            blink_stream=blink_stream
        )
        return trial_data

    def _compute_biomarkers(
        self,
        trial_idx: int,
        target_x: float,
        target_y: float,
        trial_start: float,
        trial_end: float,
        gaze_x: List[float],
        gaze_y: List[float],
        velocities: List[float],
        timestamps: List[float],
        blink_stream: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Compute fixation biomarkers from collected samples."""

        validity = self.compute_trial_validity()
        self.trial_validities.append(validity)

        gaze_x = np.array(gaze_x, dtype=float)
        gaze_y = np.array(gaze_y, dtype=float)
        velocities = np.array(velocities, dtype=float)
        timestamps = np.array(timestamps, dtype=float)

        dev_x = gaze_x - target_x
        dev_y = gaze_y - target_y

        # RMS stability
        rms_x = compute_rms(dev_x)
        rms_y = compute_rms(dev_y)
        rms_total = np.sqrt(rms_x**2 + rms_y**2) if not np.isnan(rms_x) and not np.isnan(rms_y) else np.nan

        # BCEA
        bcea = compute_bcea(gaze_x, gaze_y)

        # Microsaccade rate (velocity threshold crossing; conservative)
        if velocities.size > 0 and timestamps.size > 0:
            microsaccade_rate = compute_microsaccade_rate(
                velocities, timestamps,
                threshold=self.config.microsaccade_velocity_threshold
            )
        else:
            microsaccade_rate = np.nan

        # Drift velocity (slow component): median of the lower 80% of instantaneous speeds
        drift_velocity = np.nan
        if gaze_x.size > 2 and timestamps.size > 2:
            dt = np.diff(timestamps)
            dx = np.diff(gaze_x)
            dy = np.diff(gaze_y)

            valid_dt = dt > 1e-6
            if np.any(valid_dt):
                inst_speed = np.sqrt(dx[valid_dt]**2 + dy[valid_dt]**2) / dt[valid_dt]
                inst_speed = inst_speed[np.isfinite(inst_speed)]
                if inst_speed.size > 0:
                    cutoff = np.percentile(inst_speed, 80)
                    slow = inst_speed[inst_speed <= cutoff]
                    drift_velocity = float(np.median(slow)) if slow.size > 0 else float(np.median(inst_speed))

        # Percent time on target (adaptive radius)
        radius_px = float(self.config.target_radius_px)  # fallback
        percent_on_target = np.nan
        if gaze_x.size > 0:
            distances = np.sqrt(dev_x**2 + dev_y**2)

            if getattr(self.config, "target_radius_mode", "fixed") == "adaptive":
                calib_err = None

                # Prefer calibration mean error if available
                if getattr(self, "calibrator", None) is not None and hasattr(self.calibrator, "mean_error_px"):
                    calib_err = self.calibrator.mean_error_px
                elif getattr(self, "calibration", None) is not None and hasattr(self.calibration, "mean_error_px"):
                    calib_err = self.calibration.mean_error_px

                # Fallback: use dispersion proxy
                if calib_err is None or np.isnan(calib_err) or calib_err <= 0:
                    calib_err = float(np.nanmedian(distances)) if distances.size else np.nan

                if calib_err is not None and not np.isnan(calib_err) and calib_err > 0:
                    radius_px = float(self.config.adaptive_radius_k * calib_err)
                    radius_px = float(np.clip(
                        radius_px,
                        self.config.adaptive_radius_min_px,
                        self.config.adaptive_radius_max_px
                    ))

            on_target = distances <= radius_px
            percent_on_target = float(np.mean(on_target) * 100.0)

        # Blink rate: use blink_stream (frame_data) first; fallback to trial_frames
        blink_rate = np.nan
        total_duration = trial_end - trial_start
        if total_duration > 0:
            flags = None

            if blink_stream is not None and len(blink_stream) > 1:
                flags = blink_stream
            else:
                # fallback to self.trial_frames if blink_flag exists there
                tf_flags = [int(f.get('blink_flag', 0)) for f in getattr(self, "trial_frames", [])]
                if len(tf_flags) > 1:
                    flags = tf_flags

            if flags is not None and len(flags) > 1:
                blink_onsets = sum(
                    1 for i in range(1, len(flags))
                    if flags[i] == 1 and flags[i - 1] == 0
                )
                blink_rate = float(blink_onsets / (total_duration / 60.0))

        # Fixation-specific validity
        fixation_validity = self.validity_checker.check_fixation_validity(rms_total, validity)

        # Calibration quality
        calib_quality = self.calibrator.get_quality_score() if getattr(self, "calibrator", None) else 0.0

        return {
            'trial_id': trial_idx,
            'trial_start_s': trial_start,
            'trial_end_s': trial_end,
            'target_x': target_x,
            'target_y': target_y,
            'target_radius_used_px': radius_px,
            'fixation_stability_rms_px': rms_total,
            'fixation_bcea_px2': bcea,
            'microsaccade_rate_per_min': microsaccade_rate,
            'drift_velocity_px_s': drift_velocity,
            'percent_time_on_target': percent_on_target,
            'blink_rate_per_min': blink_rate,
            'valid_fraction': validity.valid_fraction,
            'valid': 1 if fixation_validity.valid else 0,
            'invalid_reason': fixation_validity.reason.value,
            'quality_score': fixation_validity.quality_score,
            'calibration_quality': calib_quality,
            'fps_median': validity.fps_median,
            'clamp_rate': validity.clamp_rate
        }

    def _create_skipped_trial(
        self,
        trial_idx: int,
        target_x: float,
        target_y: float,
        trial_start: float
    ) -> Dict[str, Any]:
        """Create a skipped trial record."""
        return {
            'trial_id': trial_idx,
            'trial_start_s': trial_start,
            'trial_end_s': time.time(),
            'target_x': target_x,
            'target_y': target_y,
            'target_radius_used_px': np.nan,
            'fixation_stability_rms_px': np.nan,
            'fixation_bcea_px2': np.nan,
            'microsaccade_rate_per_min': np.nan,
            'drift_velocity_px_s': np.nan,
            'percent_time_on_target': np.nan,
            'blink_rate_per_min': np.nan,
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
            'Mean RMS (px)': result.biomarkers.get('mean_rms_px', np.nan),
            'Mean BCEA (px²)': result.biomarkers.get('mean_bcea_px2', np.nan),
            'Microsaccade Rate (/min)': result.biomarkers.get('mean_microsaccade_rate', np.nan),
            'Drift Velocity (px/s)': result.biomarkers.get('mean_drift_velocity', np.nan),
            'Time on Target (%)': result.biomarkers.get('mean_percent_on_target', np.nan),
        }

        self.ui.draw_results("Fixation Task Results", metrics, result.valid_rate)

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
                'fixation_duration': self.config.fixation_duration,
                'target_radius_mode': getattr(self.config, "target_radius_mode", "fixed"),
                'target_radius_px': self.config.target_radius_px,
                'adaptive_radius_k': getattr(self.config, "adaptive_radius_k", None),
                'adaptive_radius_min_px': getattr(self.config, "adaptive_radius_min_px", None),
                'adaptive_radius_max_px': getattr(self.config, "adaptive_radius_max_px", None),
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
