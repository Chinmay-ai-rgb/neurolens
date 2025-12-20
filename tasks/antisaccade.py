"""Anti-Saccade Task for NeuroLens+ eye tracking system."""

import numpy as np
import time
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .base import BaseTask, TaskConfig, TaskResult
from core.logging import FrameLogger, SummaryLogger, ANTISACCADE_SUMMARY_COLUMNS
from core.validity import InvalidReason
from core.utils import detect_saccade_onset, find_peak_velocity


class AntisaccadeState(Enum):
    """Antisaccade trial state machine states."""
    FORE = "FORE"      # Foreperiod - fixation at center
    STIMULUS = "STIMULUS"  # Stimulus appears (look away!)
    POST = "POST"      # Post-response recording


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
    foreperiod_min: float = 1.0
    foreperiod_max: float = 1.8
    post_duration: float = 1.5
    inter_trial_interval: float = 0.5
    
    # Detection thresholds
    velocity_threshold: float = 30.0
    direction_error_window_ms: float = 200.0  # Window to detect direction error
    
    # Validity thresholds
    latency_min_ms: float = 80.0
    latency_max_ms: float = 1000.0


class AntisaccadeTask(BaseTask):
    """
    Anti-Saccade Task - Measure inhibitory control.
    
    Participant must look AWAY from the stimulus (opposite direction).
    Measures:
    - Antisaccade latency
    - Direction errors (prosaccade errors)
    - Correction time
    - Inhibition success rate
    """
    
    TASK_NAME = "antisaccade"
    
    def __init__(self, config: Optional[AntisaccadeConfig] = None):
        """Initialize antisaccade task."""
        super().__init__(config or AntisaccadeConfig())
        self.config: AntisaccadeConfig = self.config
        
        self.trial_directions: List[str] = []
    
    def get_instructions(self) -> List[str]:
        """Get antisaccade task instructions."""
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
        """Get summary CSV columns."""
        return ANTISACCADE_SUMMARY_COLUMNS
    
    def _generate_trial_sequence(self):
        """Generate randomized trial sequence."""
        n_left = self.config.n_trials // 2
        n_right = self.config.n_trials - n_left
        
        self.trial_directions = ['left'] * n_left + ['right'] * n_right
        random.shuffle(self.trial_directions)
    
    def run_task(self) -> TaskResult:
        """Run the antisaccade task."""
        result = TaskResult()
        
        self._generate_trial_sequence()
        
        # Calculate positions
        center_x = self.config.center_x_norm * self.config.screen_width
        center_y = self.config.target_y_norm * self.config.screen_height
        stim_left_x = self.config.stimulus_x_norm_left * self.config.screen_width
        stim_right_x = self.config.stimulus_x_norm_right * self.config.screen_width
        
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
            stimulus_x = stim_left_x if direction == 'left' else stim_right_x
            target_x = stim_right_x if direction == 'left' else stim_left_x  # Opposite!
            
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
                    f"error={trial_data.get('direction_error', 0)}, "
                    f"latency={trial_data.get('antisaccade_latency_ms', np.nan):.1f}ms"
                )
        
        # Compute results
        result.n_trials = len(trial_summaries)
        result.n_valid_trials = sum(1 for t in trial_summaries if t.get('valid', 0) == 1)
        result.valid_rate = result.n_valid_trials / result.n_trials if result.n_trials > 0 else 0.0
        
        valid_trials = [t for t in trial_summaries if t.get('valid', 0) == 1]
        if valid_trials:
            inhibition_success = sum(1 for t in valid_trials if t.get('inhibition_success', 0) == 1)
            result.biomarkers = {
                'mean_latency_ms': np.nanmean([t.get('antisaccade_latency_ms', np.nan) for t in valid_trials]),
                'inhibition_success_rate': inhibition_success / len(valid_trials) if valid_trials else 0,
                'direction_error_rate': 1 - (inhibition_success / len(valid_trials)) if valid_trials else 1,
                'mean_correction_time_ms': np.nanmean([t.get('correction_time_ms', np.nan) for t in valid_trials if t.get('direction_error', 0) == 1]),
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
        direction: str,
        center_x: float,
        center_y: float,
        stimulus_x: float,
        target_x: float
    ) -> Optional[Dict[str, Any]]:
        """Run a single antisaccade trial."""
        
        trial_start = time.time()
        foreperiod = random.uniform(self.config.foreperiod_min, self.config.foreperiod_max)
        
        # Data collection
        post_gaze_x = []
        post_vel_x = []
        post_timestamps = []
        
        stimulus_time = None
        state = AntisaccadeState.FORE
        
        # === FOREPERIOD ===
        fore_end = trial_start + foreperiod
        
        while time.time() < fore_end and self.ui.running:
            current_time = time.time()
            
            events = self.ui.process_events()
            if events['quit']:
                self.running = False
                return None
            if events['key_s']:
                return self._create_skipped_trial(trial_idx, direction, center_x, center_y, stimulus_x, target_x, trial_start)
            
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
        
        # === STIMULUS (show peripheral stimulus) ===
        stimulus_time = time.time()
        state = AntisaccadeState.STIMULUS
        
        # === POST (record response) ===
        post_end = stimulus_time + self.config.post_duration
        state = AntisaccadeState.POST
        
        while time.time() < post_end and self.ui.running:
            current_time = time.time()
            
            events = self.ui.process_events()
            if events['quit']:
                self.running = False
                return None
            if events['key_s']:
                return self._create_skipped_trial(trial_idx, direction, center_x, center_y, stimulus_x, target_x, trial_start)
            
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame_data = self.process_frame_with_mapping(frame, current_time)
            self.log_frame(trial_idx, state.value, target_x, center_y, frame_data)
            
            if frame_data.get('valid_sample', 0) == 1:
                post_gaze_x.append(frame_data.get('gaze_x_px_comp', np.nan))
                post_vel_x.append(frame_data.get('vel_x_px_s', np.nan))
                post_timestamps.append(current_time)
            
            # Show stimulus AND correct target location
            self.ui.clear_screen()
            self.ui.draw_target(stimulus_x, center_y, color=(255, 100, 100))  # Red stimulus
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
            trial_idx, direction, center_x, center_y, stimulus_x, target_x,
            trial_start, trial_end, stimulus_time,
            post_gaze_x, post_vel_x, post_timestamps
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
        vel_x: List[float],
        timestamps: List[float]
    ) -> Dict[str, Any]:
        """Compute antisaccade biomarkers."""
        
        generic_validity = self.compute_trial_validity()
        
        gaze_x = np.array(gaze_x)
        vel_x = np.array(vel_x)
        timestamps = np.array(timestamps)
        
        # Expected directions
        stimulus_direction = np.sign(stimulus_x - center_x)  # Direction TO stimulus
        correct_direction = -stimulus_direction  # Opposite direction
        
        latency_ms = np.nan
        direction_error = 0
        correction_time_ms = np.nan
        inhibition_success = 0
        peak_velocity = np.nan
        amplitude = np.nan
        landing_error = np.nan
        
        if len(gaze_x) >= 3:
            vel_x_abs = np.abs(vel_x)
            
            # Detect first saccade
            onset_idx = detect_saccade_onset(
                vel_x_abs, timestamps,
                threshold=self.config.velocity_threshold
            )
            
            if onset_idx is not None and onset_idx < len(timestamps):
                onset_time = timestamps[onset_idx]
                latency_ms = (onset_time - stimulus_time) * 1000
                
                # Determine initial saccade direction
                # Look at velocity sign at onset
                initial_vel_sign = np.sign(vel_x[onset_idx])
                
                # Check if initial movement was toward stimulus (error)
                if initial_vel_sign == stimulus_direction:
                    direction_error = 1
                    inhibition_success = 0
                    
                    # Look for correction
                    for i in range(onset_idx + 1, len(vel_x)):
                        if np.sign(vel_x[i]) == correct_direction and vel_x_abs[i] > self.config.velocity_threshold:
                            correction_time_ms = (timestamps[i] - onset_time) * 1000
                            break
                else:
                    direction_error = 0
                    inhibition_success = 1
                
                # Peak velocity
                peak_vel, peak_idx = find_peak_velocity(vel_x_abs, onset_idx, len(vel_x_abs))
                peak_velocity = peak_vel
                
                # Find landing
                landing_idx = peak_idx
                for i in range(peak_idx, len(vel_x_abs)):
                    if vel_x_abs[i] < self.config.velocity_threshold:
                        landing_idx = i
                        break
                
                if landing_idx < len(gaze_x):
                    landing_gaze = gaze_x[landing_idx]
                    amplitude = landing_gaze - gaze_x[onset_idx]
                    landing_error = abs(landing_gaze - target_x)
        
        # Check validity
        antisaccade_validity = self.validity_checker.check_antisaccade_validity(
            latency_ms=latency_ms if not np.isnan(latency_ms) else 0,
            direction_error=direction_error == 1,
            correction_time_ms=correction_time_ms,
            generic_validity=generic_validity
        )
        self.trial_validities.append(antisaccade_validity)
        
        calib_quality = self.calibrator.get_quality_score() if self.calibrator else 0.0
        
        return {
            'trial_id': trial_idx,
            'trial_start_s': trial_start,
            'trial_end_s': trial_end,
            'stimulus_x': stimulus_x,
            'stimulus_y': center_y,
            'target_x': target_x,
            'target_y': center_y,
            'direction': direction,
            'antisaccade_latency_ms': latency_ms,
            'direction_error': direction_error,
            'correction_time_ms': correction_time_ms,
            'inhibition_success': inhibition_success,
            'peak_velocity_px_s': peak_velocity,
            'amplitude_px': amplitude,
            'landing_error_px': landing_error,
            'valid_fraction': generic_validity.valid_fraction,
            'valid': 1 if antisaccade_validity.valid else 0,
            'invalid_reason': antisaccade_validity.reason.value,
            'quality_score': antisaccade_validity.quality_score,
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
        stimulus_x: float,
        target_x: float,
        trial_start: float
    ) -> Dict[str, Any]:
        """Create skipped trial record."""
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
        """Show results screen."""
        metrics = {
            'Valid Trials': f"{result.n_valid_trials}/{result.n_trials}",
            'Mean Latency (ms)': result.biomarkers.get('mean_latency_ms', np.nan),
            'Inhibition Success Rate': f"{result.biomarkers.get('inhibition_success_rate', 0)*100:.1f}%",
            'Direction Error Rate': f"{result.biomarkers.get('direction_error_rate', 0)*100:.1f}%",
            'Mean Correction Time (ms)': result.biomarkers.get('mean_correction_time_ms', np.nan),
        }
        
        self.ui.draw_results("Anti-Saccade Results", metrics, result.valid_rate)
        
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
