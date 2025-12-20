"""Smooth Pursuit Task for NeuroLens+ eye tracking system."""

import numpy as np
import time
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .base import BaseTask, TaskConfig, TaskResult
from core.logging import FrameLogger, SummaryLogger, PURSUIT_SUMMARY_COLUMNS
from core.validity import InvalidReason
from core.utils import compute_pursuit_gain, cross_correlation_lag, compute_rms


class PursuitState(Enum):
    """Pursuit trial state machine states."""
    FIXATION = "FIXATION"  # Initial fixation
    PURSUIT = "PURSUIT"    # Target moving
    END = "END"            # Trial end


@dataclass
class PursuitConfig(TaskConfig):
    """Configuration for smooth pursuit task."""
    
    # Trial settings
    n_trials: int = 8  # 4 left, 4 right
    
    # Target motion
    target_velocity_px_s: float = 200.0  # pixels per second
    pursuit_duration: float = 4.0  # seconds of pursuit
    initial_fixation: float = 1.0  # seconds
    
    # Target positions
    start_x_norm: float = 0.5  # Start at center
    end_x_norm_left: float = 0.15
    end_x_norm_right: float = 0.85
    target_y_norm: float = 0.5
    
    # Catch-up saccade detection
    catch_up_velocity_threshold: float = 100.0  # px/s


class PursuitTask(BaseTask):
    """
    Smooth Pursuit Task - Measure smooth pursuit eye movements.
    
    Participant tracks a smoothly moving target.
    Measures:
    - Pursuit gain
    - Pursuit latency
    - Catch-up saccade count/rate
    - Position error (mean, RMSE)
    - Phase lag
    """
    
    TASK_NAME = "pursuit"
    
    def __init__(self, config: Optional[PursuitConfig] = None):
        """Initialize pursuit task."""
        super().__init__(config or PursuitConfig())
        self.config: PursuitConfig = self.config
        
        # Trial sequence
        self.trial_directions: List[str] = []
    
    def get_instructions(self) -> List[str]:
        """Get pursuit task instructions."""
        return [
            "SMOOTH PURSUIT TASK",
            "",
            "A white dot will appear and begin moving slowly.",
            "Follow the dot smoothly with your eyes.",
            "Try to keep your eyes on the dot as it moves.",
            "",
            f"There will be {self.config.n_trials} trials.",
            "",
            "Press SPACE to start.",
            "Press Q or ESC to quit at any time.",
            "Press S to skip a trial."
        ]
    
    def get_summary_columns(self) -> List[str]:
        """Get summary CSV columns."""
        return PURSUIT_SUMMARY_COLUMNS
    
    def _generate_trial_sequence(self):
        """Generate randomized trial sequence."""
        n_left = self.config.n_trials // 2
        n_right = self.config.n_trials - n_left
        
        self.trial_directions = ['left'] * n_left + ['right'] * n_right
        random.shuffle(self.trial_directions)
    
    def run_task(self) -> TaskResult:
        """Run the smooth pursuit task."""
        result = TaskResult()
        
        self._generate_trial_sequence()
        
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
            
            direction = self.trial_directions[trial_idx]
            
            trial_data = self._run_trial(trial_idx, direction)
            
            if trial_data:
                self.summary_logger.log_trial(trial_data)
                trial_summaries.append(trial_data)
                
                self.debug_logger.info(
                    f"Trial {trial_idx + 1} ({direction}): "
                    f"valid={trial_data.get('valid', 0)}, "
                    f"gain={trial_data.get('pursuit_gain', np.nan):.2f}"
                )
        
        # Compute results
        result.n_trials = len(trial_summaries)
        result.n_valid_trials = sum(1 for t in trial_summaries if t.get('valid', 0) == 1)
        result.valid_rate = result.n_valid_trials / result.n_trials if result.n_trials > 0 else 0.0
        
        valid_trials = [t for t in trial_summaries if t.get('valid', 0) == 1]
        if valid_trials:
            result.biomarkers = {
                'mean_gain': np.nanmean([t.get('pursuit_gain', np.nan) for t in valid_trials]),
                'mean_latency_ms': np.nanmean([t.get('pursuit_latency_ms', np.nan) for t in valid_trials]),
                'mean_catch_up_rate': np.nanmean([t.get('catch_up_saccade_rate_per_s', np.nan) for t in valid_trials]),
                'mean_position_error_px': np.nanmean([t.get('position_error_mean_px', np.nan) for t in valid_trials]),
                'mean_phase_lag_ms': np.nanmean([t.get('phase_lag_ms', np.nan) for t in valid_trials]),
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
        direction: str
    ) -> Optional[Dict[str, Any]]:
        """Run a single pursuit trial."""
        
        trial_start = time.time()
        
        # Calculate positions
        center_x = self.config.start_x_norm * self.config.screen_width
        center_y = self.config.target_y_norm * self.config.screen_height
        
        if direction == 'left':
            end_x = self.config.end_x_norm_left * self.config.screen_width
            velocity = -self.config.target_velocity_px_s
        else:
            end_x = self.config.end_x_norm_right * self.config.screen_width
            velocity = self.config.target_velocity_px_s
        
        # Data collection
        gaze_x_samples = []
        gaze_y_samples = []
        target_x_samples = []
        vel_x_samples = []
        timestamps = []
        
        state = PursuitState.FIXATION
        current_target_x = center_x
        pursuit_start_time = None
        
        # === INITIAL FIXATION ===
        fixation_end = trial_start + self.config.initial_fixation
        
        while time.time() < fixation_end and self.ui.running:
            current_time = time.time()
            
            events = self.ui.process_events()
            if events['quit']:
                self.running = False
                return None
            if events['key_s']:
                return self._create_skipped_trial(trial_idx, direction, trial_start)
            
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame_data = self.process_frame_with_mapping(frame, current_time)
            self.log_frame(trial_idx, state.value, current_target_x, center_y, frame_data)
            
            self.ui.clear_screen()
            self.ui.draw_target(current_target_x, center_y)
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
        
        # === PURSUIT ===
        state = PursuitState.PURSUIT
        pursuit_start_time = time.time()
        pursuit_end = pursuit_start_time + self.config.pursuit_duration
        
        while time.time() < pursuit_end and self.ui.running:
            current_time = time.time()
            elapsed = current_time - pursuit_start_time
            
            events = self.ui.process_events()
            if events['quit']:
                self.running = False
                return None
            if events['key_s']:
                return self._create_skipped_trial(trial_idx, direction, trial_start)
            
            # Update target position
            current_target_x = center_x + velocity * elapsed
            current_target_x = np.clip(current_target_x, 0, self.config.screen_width - 1)
            
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame_data = self.process_frame_with_mapping(frame, current_time)
            self.log_frame(trial_idx, state.value, current_target_x, center_y, frame_data)
            
            # Collect samples
            if frame_data.get('valid_sample', 0) == 1:
                gaze_x_samples.append(frame_data.get('gaze_x_px_comp', np.nan))
                gaze_y_samples.append(frame_data.get('gaze_y_px_comp', np.nan))
                target_x_samples.append(current_target_x)
                vel_x_samples.append(frame_data.get('vel_x_px_s', np.nan))
                timestamps.append(current_time)
            
            self.ui.clear_screen()
            self.ui.draw_target(current_target_x, center_y)
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
            trial_idx, direction, trial_start, trial_end, pursuit_start_time,
            gaze_x_samples, target_x_samples, vel_x_samples, timestamps,
            velocity
        )
        
        return trial_data
    
    def _compute_biomarkers(
        self,
        trial_idx: int,
        direction: str,
        trial_start: float,
        trial_end: float,
        pursuit_start: float,
        gaze_x: List[float],
        target_x: List[float],
        vel_x: List[float],
        timestamps: List[float],
        target_velocity: float
    ) -> Dict[str, Any]:
        """Compute pursuit biomarkers."""
        
        generic_validity = self.compute_trial_validity()
        
        gaze_x = np.array(gaze_x)
        target_x = np.array(target_x)
        vel_x = np.array(vel_x)
        timestamps = np.array(timestamps)
        
        # Pursuit gain
        target_vel_array = np.full_like(vel_x, target_velocity)
        gain = compute_pursuit_gain(vel_x, target_vel_array)
        
        # Position error
        position_error = gaze_x - target_x
        error_mean = np.nanmean(np.abs(position_error))
        error_rmse = compute_rms(position_error)
        
        # Phase lag
        fps_est = 30.0
        if len(timestamps) > 1:
            fps_est = len(timestamps) / (timestamps[-1] - timestamps[0])
        
        corr, lag_samples = cross_correlation_lag(gaze_x, target_x)
        phase_lag_ms = (lag_samples / fps_est) * 1000 if fps_est > 0 else np.nan
        
        # Catch-up saccades
        catch_up_count = 0
        in_saccade = False
        for v in np.abs(vel_x):
            if v > self.config.catch_up_velocity_threshold and not in_saccade:
                catch_up_count += 1
                in_saccade = True
            elif v < self.config.catch_up_velocity_threshold * 0.5:
                in_saccade = False
        
        duration_s = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1.0
        catch_up_rate = catch_up_count / duration_s if duration_s > 0 else np.nan
        
        # Pursuit latency
        latency_ms = np.nan
        for i, v in enumerate(vel_x):
            if np.sign(v) == np.sign(target_velocity) and abs(v) > 10:
                latency_ms = (timestamps[i] - pursuit_start) * 1000
                break
        
        # Check validity
        pursuit_validity = self.validity_checker.check_pursuit_validity(
            gain if not np.isnan(gain) else 0,
            generic_validity
        )
        self.trial_validities.append(pursuit_validity)
        
        calib_quality = self.calibrator.get_quality_score() if self.calibrator else 0.0
        
        return {
            'trial_id': trial_idx,
            'trial_start_s': trial_start,
            'trial_end_s': trial_end,
            'direction': direction,
            'target_velocity_px_s': target_velocity,
            'pursuit_gain': gain,
            'pursuit_latency_ms': latency_ms,
            'catch_up_saccade_count': catch_up_count,
            'catch_up_saccade_rate_per_s': catch_up_rate,
            'position_error_mean_px': error_mean,
            'position_error_rmse_px': error_rmse,
            'phase_lag_ms': phase_lag_ms,
            'valid_fraction': generic_validity.valid_fraction,
            'valid': 1 if pursuit_validity.valid else 0,
            'invalid_reason': pursuit_validity.reason.value,
            'quality_score': pursuit_validity.quality_score,
            'calibration_quality': calib_quality,
            'fps_median': generic_validity.fps_median,
            'clamp_rate': generic_validity.clamp_rate
        }
    
    def _create_skipped_trial(
        self,
        trial_idx: int,
        direction: str,
        trial_start: float
    ) -> Dict[str, Any]:
        """Create skipped trial record."""
        return {
            'trial_id': trial_idx,
            'trial_start_s': trial_start,
            'trial_end_s': time.time(),
            'direction': direction,
            'target_velocity_px_s': self.config.target_velocity_px_s,
            'pursuit_gain': np.nan,
            'pursuit_latency_ms': np.nan,
            'catch_up_saccade_count': 0,
            'catch_up_saccade_rate_per_s': np.nan,
            'position_error_mean_px': np.nan,
            'position_error_rmse_px': np.nan,
            'phase_lag_ms': np.nan,
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
            'Mean Gain': result.biomarkers.get('mean_gain', np.nan),
            'Mean Latency (ms)': result.biomarkers.get('mean_latency_ms', np.nan),
            'Catch-up Rate (/s)': result.biomarkers.get('mean_catch_up_rate', np.nan),
            'Position Error (px)': result.biomarkers.get('mean_position_error_px', np.nan),
            'Phase Lag (ms)': result.biomarkers.get('mean_phase_lag_ms', np.nan),
        }
        
        self.ui.draw_results("Smooth Pursuit Results", metrics, result.valid_rate)
        
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
                'target_velocity_px_s': self.config.target_velocity_px_s,
                'pursuit_duration': self.config.pursuit_duration,
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
