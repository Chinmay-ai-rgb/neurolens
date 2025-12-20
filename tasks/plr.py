"""Pupillary Light Reflex (PLR) Task for NeuroLens+ eye tracking system."""

import numpy as np
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base import BaseTask, TaskConfig, TaskResult
from core.logging import FrameLogger, SummaryLogger, PLR_SUMMARY_COLUMNS
from core.validity import InvalidReason


@dataclass
class PLRConfig(TaskConfig):
    """Configuration for PLR task."""
    
    # Trial settings
    n_trials: int = 5
    
    # Timing
    baseline_duration: float = 3.0  # Dark adaptation before flash
    flash_duration: float = 0.5     # Duration of light flash
    recovery_duration: float = 5.0  # Recording after flash
    inter_trial_interval: float = 3.0
    
    # Flash intensity (0-1)
    flash_intensity: float = 1.0
    
    # PLR thresholds
    min_constriction: float = 0.02  # Minimum detectable constriction
    max_latency_ms: float = 500.0


class PLRTask(BaseTask):
    """
    Pupillary Light Reflex (PLR) Task.
    
    Measures pupil response to light stimulus (screen flash).
    Uses iris diameter as pupil proxy since webcam cannot directly measure pupil.
    
    Measures:
    - Constriction latency
    - Constriction amplitude
    - Constriction velocity
    - Dilation recovery time
    - Left/right asymmetry
    """
    
    TASK_NAME = "plr"
    
    def __init__(self, config: Optional[PLRConfig] = None):
        """Initialize PLR task."""
        super().__init__(config or PLRConfig())
        self.config: PLRConfig = self.config
    
    def get_instructions(self) -> List[str]:
        """Get PLR task instructions."""
        return [
            "PUPILLARY LIGHT REFLEX TEST",
            "",
            "The screen will flash bright white.",
            "Keep your eyes open and look at the center.",
            "Your pupil response will be measured.",
            "",
            "Note: This uses iris diameter as a proxy",
            "since webcams cannot directly measure pupil size.",
            "",
            f"There will be {self.config.n_trials} trials.",
            "",
            "Press SPACE to start.",
            "Press Q or ESC to quit."
        ]
    
    def get_summary_columns(self) -> List[str]:
        """Get summary CSV columns."""
        return PLR_SUMMARY_COLUMNS
    
    def run_task(self) -> TaskResult:
        """Run the PLR task."""
        result = TaskResult()
        
        center_x = self.config.screen_width // 2
        center_y = self.config.screen_height // 2
        
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
            
            trial_data = self._run_trial(trial_idx, center_x, center_y)
            
            if trial_data:
                self.summary_logger.log_trial(trial_data)
                trial_summaries.append(trial_data)
                
                self.debug_logger.info(
                    f"Trial {trial_idx + 1}: "
                    f"valid={trial_data.get('valid', 0)}, "
                    f"constriction={trial_data.get('constriction_amplitude', np.nan):.4f}"
                )
        
        # Compute results
        result.n_trials = len(trial_summaries)
        result.n_valid_trials = sum(1 for t in trial_summaries if t.get('valid', 0) == 1)
        result.valid_rate = result.n_valid_trials / result.n_trials if result.n_trials > 0 else 0.0
        
        valid_trials = [t for t in trial_summaries if t.get('valid', 0) == 1]
        if valid_trials:
            result.biomarkers = {
                'mean_constriction_latency_ms': np.nanmean([t.get('constriction_latency_ms', np.nan) for t in valid_trials]),
                'mean_constriction_amplitude': np.nanmean([t.get('constriction_amplitude', np.nan) for t in valid_trials]),
                'mean_constriction_velocity': np.nanmean([t.get('constriction_velocity', np.nan) for t in valid_trials]),
                'mean_recovery_time_ms': np.nanmean([t.get('dilation_recovery_time_ms', np.nan) for t in valid_trials]),
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
        center_x: float,
        center_y: float
    ) -> Optional[Dict[str, Any]]:
        """Run a single PLR trial."""
        
        trial_start = time.time()
        
        # Data collection
        pupil_samples = []
        pupil_left_samples = []
        pupil_right_samples = []
        timestamps = []
        
        flash_time = None
        
        # === BASELINE (dark screen) ===
        baseline_end = trial_start + self.config.baseline_duration
        baseline_pupil = []
        
        while time.time() < baseline_end and self.ui.running:
            current_time = time.time()
            
            events = self.ui.process_events()
            if events['quit']:
                self.running = False
                return None
            if events['key_s']:
                return self._create_skipped_trial(trial_idx, trial_start)
            
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame_data = self.process_frame_with_mapping(frame, current_time)
            self.log_frame(trial_idx, "BASELINE", center_x, center_y, frame_data)
            
            if frame_data.get('valid_sample', 0) == 1:
                pupil = frame_data.get('pupil_proxy_mean', np.nan)
                if not np.isnan(pupil):
                    baseline_pupil.append(pupil)
                    pupil_samples.append(pupil)
                    pupil_left_samples.append(frame_data.get('pupil_proxy_left', np.nan))
                    pupil_right_samples.append(frame_data.get('pupil_proxy_right', np.nan))
                    timestamps.append(current_time)
            
            # Dark screen with small fixation point
            self.ui.clear_screen()
            self.ui.draw_target(center_x, center_y, radius=5)
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
        
        # === FLASH ===
        flash_time = time.time()
        flash_end = flash_time + self.config.flash_duration
        
        while time.time() < flash_end and self.ui.running:
            current_time = time.time()
            
            events = self.ui.process_events()
            if events['quit']:
                self.running = False
                return None
            
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame_data = self.process_frame_with_mapping(frame, current_time)
            self.log_frame(trial_idx, "FLASH", center_x, center_y, frame_data)
            
            if frame_data.get('valid_sample', 0) == 1:
                pupil = frame_data.get('pupil_proxy_mean', np.nan)
                if not np.isnan(pupil):
                    pupil_samples.append(pupil)
                    pupil_left_samples.append(frame_data.get('pupil_proxy_left', np.nan))
                    pupil_right_samples.append(frame_data.get('pupil_proxy_right', np.nan))
                    timestamps.append(current_time)
            
            # Bright flash
            self.ui.draw_flash(self.config.flash_intensity)
            self.ui.update_display()
            self.ui.tick(60)
        
        # === RECOVERY ===
        recovery_end = flash_time + self.config.flash_duration + self.config.recovery_duration
        
        while time.time() < recovery_end and self.ui.running:
            current_time = time.time()
            
            events = self.ui.process_events()
            if events['quit']:
                self.running = False
                return None
            
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame_data = self.process_frame_with_mapping(frame, current_time)
            self.log_frame(trial_idx, "RECOVERY", center_x, center_y, frame_data)
            
            if frame_data.get('valid_sample', 0) == 1:
                pupil = frame_data.get('pupil_proxy_mean', np.nan)
                if not np.isnan(pupil):
                    pupil_samples.append(pupil)
                    pupil_left_samples.append(frame_data.get('pupil_proxy_left', np.nan))
                    pupil_right_samples.append(frame_data.get('pupil_proxy_right', np.nan))
                    timestamps.append(current_time)
            
            # Dark screen again
            self.ui.clear_screen()
            self.ui.draw_target(center_x, center_y, radius=5)
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
        
        # Compute biomarkers
        trial_data = self._compute_biomarkers(
            trial_idx, trial_start, trial_end, flash_time,
            pupil_samples, pupil_left_samples, pupil_right_samples,
            timestamps, baseline_pupil
        )
        
        time.sleep(self.config.inter_trial_interval)
        
        return trial_data
    
    def _compute_biomarkers(
        self,
        trial_idx: int,
        trial_start: float,
        trial_end: float,
        flash_time: float,
        pupil: List[float],
        pupil_left: List[float],
        pupil_right: List[float],
        timestamps: List[float],
        baseline_pupil: List[float]
    ) -> Dict[str, Any]:
        """Compute PLR biomarkers."""
        
        generic_validity = self.compute_trial_validity()
        
        pupil = np.array(pupil)
        pupil_left = np.array(pupil_left)
        pupil_right = np.array(pupil_right)
        timestamps = np.array(timestamps)
        
        # Baseline
        baseline = np.nanmean(baseline_pupil) if baseline_pupil else np.nan
        
        # Find post-flash data
        post_flash_mask = timestamps >= flash_time
        
        constriction_latency_ms = np.nan
        constriction_amplitude = np.nan
        constriction_velocity = np.nan
        recovery_time_ms = np.nan
        min_pupil = np.nan
        asymmetry = np.nan
        
        if post_flash_mask.sum() >= 3 and not np.isnan(baseline):
            post_pupil = pupil[post_flash_mask]
            post_times = timestamps[post_flash_mask]
            
            # Find minimum (max constriction)
            min_idx = np.nanargmin(post_pupil)
            min_pupil = post_pupil[min_idx]
            min_time = post_times[min_idx]
            
            # Constriction amplitude
            constriction_amplitude = baseline - min_pupil
            
            # Constriction latency
            constriction_latency_ms = (min_time - flash_time) * 1000
            
            # Constriction velocity (max negative derivative)
            if len(post_pupil) > 1:
                pupil_deriv = np.gradient(post_pupil, post_times)
                constriction_velocity = np.nanmin(pupil_deriv)
            
            # Recovery time (to 75% of baseline)
            recovery_threshold = min_pupil + 0.75 * constriction_amplitude
            for i in range(min_idx, len(post_pupil)):
                if post_pupil[i] >= recovery_threshold:
                    recovery_time_ms = (post_times[i] - min_time) * 1000
                    break
            
            # Asymmetry
            baseline_left = np.nanmean(pupil_left[~post_flash_mask]) if (~post_flash_mask).sum() > 0 else np.nan
            baseline_right = np.nanmean(pupil_right[~post_flash_mask]) if (~post_flash_mask).sum() > 0 else np.nan
            
            if not np.isnan(baseline_left) and not np.isnan(baseline_right) and (baseline_left + baseline_right) > 0:
                asymmetry = abs(baseline_left - baseline_right) / ((baseline_left + baseline_right) / 2)
        
        # Check validity
        plr_validity = self.validity_checker.check_plr_validity(
            constriction_amplitude if not np.isnan(constriction_amplitude) else 0,
            constriction_latency_ms if not np.isnan(constriction_latency_ms) else 1000,
            generic_validity
        )
        self.trial_validities.append(plr_validity)
        
        calib_quality = self.calibrator.get_quality_score() if self.calibrator else 0.0
        
        return {
            'trial_id': trial_idx,
            'trial_start_s': trial_start,
            'trial_end_s': trial_end,
            'flash_time_s': flash_time,
            'constriction_latency_ms': constriction_latency_ms,
            'constriction_amplitude': constriction_amplitude,
            'constriction_velocity': constriction_velocity,
            'dilation_recovery_time_ms': recovery_time_ms,
            'baseline_pupil_proxy': baseline,
            'min_pupil_proxy': min_pupil,
            'asymmetry_left_right': asymmetry,
            'valid_fraction': generic_validity.valid_fraction,
            'valid': 1 if plr_validity.valid else 0,
            'invalid_reason': plr_validity.reason.value,
            'quality_score': plr_validity.quality_score,
            'calibration_quality': calib_quality,
            'fps_median': generic_validity.fps_median,
            'clamp_rate': generic_validity.clamp_rate
        }
    
    def _create_skipped_trial(
        self,
        trial_idx: int,
        trial_start: float
    ) -> Dict[str, Any]:
        """Create skipped trial record."""
        return {
            'trial_id': trial_idx,
            'trial_start_s': trial_start,
            'trial_end_s': time.time(),
            'flash_time_s': np.nan,
            'constriction_latency_ms': np.nan,
            'constriction_amplitude': np.nan,
            'constriction_velocity': np.nan,
            'dilation_recovery_time_ms': np.nan,
            'baseline_pupil_proxy': np.nan,
            'min_pupil_proxy': np.nan,
            'asymmetry_left_right': np.nan,
            'valid_fraction': 0.0,
            'valid': 0,
            'invalid_reason': InvalidReason.SKIPPED.value,
            'quality_score': 0.0,
            'calibration_quality': 0.0,
            'fps_median': 0.0,
            'clamp_rate': 0.0
        }
    
    def _show_results(self, result: TaskResult):
        """Show results screen."""
        metrics = {
            'Valid Trials': f"{result.n_valid_trials}/{result.n_trials}",
            'Mean Constriction Latency (ms)': result.biomarkers.get('mean_constriction_latency_ms', np.nan),
            'Mean Constriction Amplitude': result.biomarkers.get('mean_constriction_amplitude', np.nan),
            'Mean Constriction Velocity': result.biomarkers.get('mean_constriction_velocity', np.nan),
            'Mean Recovery Time (ms)': result.biomarkers.get('mean_recovery_time_ms', np.nan),
        }
        
        self.ui.draw_results("PLR Test Results", metrics, result.valid_rate)
        
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
                'flash_duration': self.config.flash_duration,
                'recovery_duration': self.config.recovery_duration,
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
            meta.fps_mean = np.mean(all_fps)
            meta.fps_median = np.median(all_fps)
            meta.fps_min = np.min(all_fps)
            meta.fps_max = np.max(all_fps)
        
        meta.save(f"{self.session_dir}/meta.json")
