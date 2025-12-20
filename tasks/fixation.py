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
    target_radius_px: float = 100.0  # For on-target calculation
    
    # Biomarker thresholds
    microsaccade_velocity_threshold: float = 50.0  # px/s


class FixationTask(BaseTask):
    """
    Fixation Task - Measure fixation stability.
    
    Participant fixates on a central target for extended periods.
    Measures:
    - Fixation stability (RMS)
    - BCEA (Bivariate Contour Ellipse Area)
    - Microsaccade rate
    - Drift velocity
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
            
            # Run single trial
            trial_data = self._run_trial(trial_idx, target_x, target_y)
            
            if trial_data:
                # Log trial summary
                self.summary_logger.log_trial(trial_data)
                trial_summaries.append(trial_data)
                
                # Track validity
                validity = self.trial_validities[-1] if self.trial_validities else None
                if validity:
                    self.debug_logger.info(
                        f"Trial {trial_idx + 1}: valid={validity.valid}, "
                        f"reason={validity.reason.value}, "
                        f"rms={trial_data.get('fixation_stability_rms_px', np.nan):.1f}px"
                    )
        
        # Compute aggregate results
        result.n_trials = len(trial_summaries)
        result.n_valid_trials = sum(1 for t in trial_summaries if t.get('valid', 0) == 1)
        result.valid_rate = result.n_valid_trials / result.n_trials if result.n_trials > 0 else 0.0
        
        # Aggregate biomarkers
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
        
        # Show results
        self._show_results(result)
        
        # Save session metadata
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
        """
        Run a single fixation trial.
        
        Args:
            trial_idx: Trial index
            target_x: Target x position
            target_y: Target y position
        
        Returns:
            Trial summary dictionary
        """
        trial_start = time.time()
        trial_end = trial_start + self.config.fixation_duration
        
        # Data collection
        gaze_x_samples = []
        gaze_y_samples = []
        velocity_samples = []
        timestamps = []
        
        while time.time() < trial_end and self.ui.running:
            current_time = time.time()
            
            # Process events
            events = self.ui.process_events()
            if events['quit']:
                self.running = False
                return None
            if events['key_s']:
                # Skip trial
                return self._create_skipped_trial(trial_idx, target_x, target_y, trial_start)
            
            # Capture and process frame
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame_data = self.process_frame_with_mapping(frame, current_time)
            
            # Log frame
            self.log_frame(trial_idx, "FIXATION", target_x, target_y, frame_data)
            
            # Collect valid samples for biomarker computation
            if frame_data.get('valid_sample', 0) == 1:
                gaze_x_samples.append(frame_data.get('gaze_x_px_comp', np.nan))
                gaze_y_samples.append(frame_data.get('gaze_y_px_comp', np.nan))
                
                vel_x = frame_data.get('vel_x_px_s', 0)
                vel_y = frame_data.get('vel_y_px_s', 0)
                if not np.isnan(vel_x) and not np.isnan(vel_y):
                    velocity_samples.append(np.sqrt(vel_x**2 + vel_y**2))
                
                timestamps.append(current_time)
            
            # Update UI
            self.ui.clear_screen()
            self.ui.draw_target(target_x, target_y)
            
            # Progress indicator
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
        
        # Compute biomarkers
        trial_data = self._compute_biomarkers(
            trial_idx, target_x, target_y, trial_start, time.time(),
            gaze_x_samples, gaze_y_samples, velocity_samples, timestamps
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
        timestamps: List[float]
    ) -> Dict[str, Any]:
        """Compute fixation biomarkers from collected samples."""
        
        # Get trial validity
        validity = self.compute_trial_validity()
        self.trial_validities.append(validity)
        
        # Convert to arrays
        gaze_x = np.array(gaze_x)
        gaze_y = np.array(gaze_y)
        velocities = np.array(velocities)
        timestamps = np.array(timestamps)
        
        # Compute deviations from target
        dev_x = gaze_x - target_x
        dev_y = gaze_y - target_y
        
        # RMS stability
        rms_x = compute_rms(dev_x)
        rms_y = compute_rms(dev_y)
        rms_total = np.sqrt(rms_x**2 + rms_y**2) if not np.isnan(rms_x) and not np.isnan(rms_y) else np.nan
        
        # BCEA
        bcea = compute_bcea(gaze_x, gaze_y)
        
        # Microsaccade rate
        if len(velocities) > 0 and len(timestamps) > 0:
            microsaccade_rate = compute_microsaccade_rate(
                velocities, timestamps,
                threshold=self.config.microsaccade_velocity_threshold
            )
        else:
            microsaccade_rate = np.nan
        
        # Drift velocity
        if len(gaze_x) > 1 and len(timestamps) > 1:
            duration = timestamps[-1] - timestamps[0]
            if duration > 0:
                drift_x = (gaze_x[-1] - gaze_x[0]) / duration
                drift_y = (gaze_y[-1] - gaze_y[0]) / duration
                drift_velocity = np.sqrt(drift_x**2 + drift_y**2)
            else:
                drift_velocity = np.nan
        else:
            drift_velocity = np.nan
        
        # Percent time on target
        if len(gaze_x) > 0:
            distances = np.sqrt(dev_x**2 + dev_y**2)
            on_target = distances <= self.config.target_radius_px
            percent_on_target = np.sum(on_target) / len(on_target) * 100
        else:
            percent_on_target = np.nan
        
        # Blink rate
        blink_flags = [f.get('blink_flag', 0) for f in self.trial_frames]
        total_duration = trial_end - trial_start
        if total_duration > 0 and len(blink_flags) > 1:
            blink_onsets = sum(1 for i in range(1, len(blink_flags)) 
                             if blink_flags[i] == 1 and blink_flags[i-1] == 0)
            blink_rate = blink_onsets / (total_duration / 60)
        else:
            blink_rate = np.nan
        
        # Check fixation-specific validity
        fixation_validity = self.validity_checker.check_fixation_validity(
            rms_total, validity
        )
        
        # Get calibration quality
        calib_quality = self.calibrator.get_quality_score() if self.calibrator else 0.0
        
        return {
            'trial_id': trial_idx,
            'trial_start_s': trial_start,
            'trial_end_s': trial_end,
            'target_x': target_x,
            'target_y': target_y,
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
        
        # Wait for space
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
                'target_radius_px': self.config.target_radius_px,
            }
        )
        
        if self.calibration:
            meta.calibration_accepted = self.calibration.accepted
            meta.calibration_reason = self.calibration.reason.value
            meta.calibration_error_mean_px = self.calibration.mean_error_px
            meta.calibration_error_max_px = self.calibration.max_error_px
        
        meta.valid_rates[self.TASK_NAME] = result.valid_rate
        
        # FPS stats from trial frames
        all_fps = [f.get('fps_est', 30.0) for f in self.trial_frames]
        if all_fps:
            meta.fps_mean = np.mean(all_fps)
            meta.fps_median = np.median(all_fps)
            meta.fps_min = np.min(all_fps)
            meta.fps_max = np.max(all_fps)
        
        meta.save(f"{self.session_dir}/meta.json")
