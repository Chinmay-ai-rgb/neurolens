"""Saccade Task for NeuroLens+ eye tracking system."""

import numpy as np
import time
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .base import BaseTask, TaskConfig, TaskResult
from core.logging import FrameLogger, SummaryLogger, SACCADE_SUMMARY_COLUMNS
from core.validity import InvalidReason
from core.utils import detect_saccade_onset, find_peak_velocity


class SaccadeState(Enum):
    """Saccade trial state machine states."""
    FORE = "FORE"  # Foreperiod - fixation at center
    JUMP = "JUMP"  # Target jump - stimulus onset
    POST = "POST"  # Post-saccade recording
    NEXT = "NEXT"  # Inter-trial interval


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
    
    # Saccade detection thresholds (increased for webcam noise)
    velocity_threshold: float = 150.0  # px/s for onset detection (increased from 30 for webcam noise)
    min_velocity_samples: int = 3  # Consecutive samples above threshold
    
    # Validity thresholds
    latency_min_ms: float = 50.0
    latency_max_ms: float = 900.0
    duration_min_ms: float = 15.0
    duration_max_ms: float = 400.0
    min_amplitude_fraction: float = 0.15
    max_peak_velocity: float = 5000.0  # px/s
    critical_window_ms: float = 250.0  # Dropout window after jump


class SaccadeTask(BaseTask):
    """
    Saccade Task - Measure saccadic eye movements.
    
    State machine: FORE -> JUMP -> POST -> NEXT
    
    Participant fixates at center, then makes a saccade to a peripheral target.
    Measures:
    - Saccade latency
    - Saccade duration
    - Peak velocity
    - Amplitude
    - Gain
    - Landing error
    - Corrective saccades
    """
    
    TASK_NAME = "saccade"
    
    def __init__(self, config: Optional[SaccadeConfig] = None):
        """Initialize saccade task."""
        super().__init__(config or SaccadeConfig())
        self.config: SaccadeConfig = self.config
        
        # Trial sequence (randomized left/right)
        self.trial_directions: List[str] = []
    
    def get_instructions(self) -> List[str]:
        """Get saccade task instructions."""
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
        """Get summary CSV columns."""
        return SACCADE_SUMMARY_COLUMNS
    
    def _generate_trial_sequence(self):
        """Generate randomized trial sequence with balanced directions."""
        n_left = self.config.n_trials // 2
        n_right = self.config.n_trials - n_left
        
        self.trial_directions = ['left'] * n_left + ['right'] * n_right
        random.shuffle(self.trial_directions)
    
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
        trial_summaries = []
        
        for trial_idx in range(self.config.n_trials):
            if not self.ui.running or not self.running:
                break
            
            self.current_trial = trial_idx + 1
            self.clear_trial_buffer()
            
            # Get target for this trial
            direction = self.trial_directions[trial_idx]
            target_x = left_x if direction == 'left' else right_x
            
            # Run single trial
            trial_data = self._run_trial(
                trial_idx, direction,
                center_x, center_y, target_x
            )
            
            if trial_data:
                # Log trial summary
                self.summary_logger.log_trial(trial_data)
                trial_summaries.append(trial_data)
                
                # Log to debug
                self.debug_logger.info(
                    f"Trial {trial_idx + 1} ({direction}): "
                    f"valid={trial_data.get('valid', 0)}, "
                    f"latency={trial_data.get('saccade_latency_ms', np.nan):.1f}ms, "
                    f"amplitude={trial_data.get('saccade_amplitude_px', np.nan):.1f}px"
                )
        
        # Compute aggregate results
        result.n_trials = len(trial_summaries)
        result.n_valid_trials = sum(1 for t in trial_summaries if t.get('valid', 0) == 1)
        result.valid_rate = result.n_valid_trials / result.n_trials if result.n_trials > 0 else 0.0
        
        # Check for direction-specific issues
        left_trials = [t for t in trial_summaries if t.get('direction') == 'left']
        right_trials = [t for t in trial_summaries if t.get('direction') == 'right']
        
        left_valid = sum(1 for t in left_trials if t.get('valid', 0) == 1)
        right_valid = sum(1 for t in right_trials if t.get('valid', 0) == 1)
        
        if left_trials and left_valid / len(left_trials) < 0.5:
            result.warnings.append(f"Low validity for leftward trials: {left_valid}/{len(left_trials)}")
            self.debug_logger.warning(f"Low validity for leftward trials: {left_valid}/{len(left_trials)}")
        
        if right_trials and right_valid / len(right_trials) < 0.5:
            result.warnings.append(f"Low validity for rightward trials: {right_valid}/{len(right_trials)}")
            self.debug_logger.warning(f"Low validity for rightward trials: {right_valid}/{len(right_trials)}")
        
        # Aggregate biomarkers from valid trials
        valid_trials = [t for t in trial_summaries if t.get('valid', 0) == 1]
        if valid_trials:
            result.biomarkers = {
                'mean_latency_ms': np.nanmean([t.get('saccade_latency_ms', np.nan) for t in valid_trials]),
                'mean_duration_ms': np.nanmean([t.get('saccade_duration_ms', np.nan) for t in valid_trials]),
                'mean_peak_velocity': np.nanmean([t.get('peak_velocity_px_s', np.nan) for t in valid_trials]),
                'mean_amplitude_px': np.nanmean([t.get('saccade_amplitude_px', np.nan) for t in valid_trials]),
                'mean_gain': np.nanmean([t.get('gain', np.nan) for t in valid_trials]),
                'mean_landing_error_px': np.nanmean([t.get('landing_error_px', np.nan) for t in valid_trials]),
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
        direction: str,
        center_x: float,
        center_y: float,
        target_x: float
    ) -> Optional[Dict[str, Any]]:
        """
        Run a single saccade trial.
        
        State machine: FORE -> JUMP -> POST
        
        Args:
            trial_idx: Trial index
            direction: 'left' or 'right'
            center_x: Center fixation x position
            center_y: Center fixation y position
            target_x: Target x position
        
        Returns:
            Trial summary dictionary
        """
        trial_start = time.time()
        
        # Randomize foreperiod
        foreperiod = random.uniform(
            self.config.foreperiod_min,
            self.config.foreperiod_max
        )
        
        # Data collection for biomarker computation
        post_gaze_x = []
        post_gaze_y = []
        post_vel_x = []
        post_vel_y = []
        post_timestamps = []
        
        jump_time = None
        state = SaccadeState.FORE
        
        # === FOREPERIOD ===
        fore_end = trial_start + foreperiod
        
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
            
            # Update UI - show center target
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
        
        # === JUMP (target appears at peripheral location) ===
        jump_time = time.time()
        state = SaccadeState.JUMP
        
        # Record baseline gaze at jump time
        baseline_gaze_x = None
        
        # === POST (record saccade) ===
        post_end = jump_time + self.config.post_duration
        state = SaccadeState.POST
        
        dropout_in_critical = False
        critical_end = jump_time + self.config.critical_window_ms / 1000.0
        
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
            
            # Check for dropout in critical window
            if current_time < critical_end and frame_data.get('face_present', 0) == 0:
                dropout_in_critical = True
            
            # Collect valid samples for biomarker computation
            if frame_data.get('valid_sample', 0) == 1:
                gaze_x = frame_data.get('gaze_x_px_comp', np.nan)
                gaze_y = frame_data.get('gaze_y_px_comp', np.nan)
                
                if baseline_gaze_x is None and not np.isnan(gaze_x):
                    baseline_gaze_x = gaze_x
                
                post_gaze_x.append(gaze_x)
                post_gaze_y.append(gaze_y)
                post_vel_x.append(frame_data.get('vel_x_px_s', np.nan))
                post_vel_y.append(frame_data.get('vel_y_px_s', np.nan))
                post_timestamps.append(current_time)
            
            # Update UI - show target at new position
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
        
        # Compute biomarkers
        trial_data = self._compute_biomarkers(
            trial_idx, direction, center_x, center_y, target_x,
            trial_start, trial_end, jump_time,
            post_gaze_x, post_gaze_y, post_vel_x, post_vel_y, post_timestamps,
            baseline_gaze_x, dropout_in_critical
        )
        
        # Inter-trial interval
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
        dropout_in_critical: bool
    ) -> Dict[str, Any]:
        """Compute saccade biomarkers from collected samples."""
        
        # Get generic trial validity
        generic_validity = self.compute_trial_validity()
        
        # Convert to arrays
        gaze_x = np.array(gaze_x)
        gaze_y = np.array(gaze_y)
        vel_x = np.array(vel_x)
        vel_y = np.array(vel_y)
        timestamps = np.array(timestamps)
        
        # Compute eccentricity
        eccentricity = target_x - center_x
        
        # Initialize biomarkers
        latency_ms = np.nan
        duration_ms = np.nan
        peak_velocity = np.nan
        amplitude = np.nan
        gain = np.nan
        landing_error = np.nan
        overshoot = 0.0
        undershoot = 0.0
        corrective_count = 0
        direction_correct = False
        
        # Need sufficient samples
        if len(gaze_x) >= 3 and len(timestamps) >= 3:
            # Compute velocity magnitude
            vel_mag = np.sqrt(vel_x**2 + vel_y**2)
            
            # Use x-velocity for direction detection (horizontal saccades)
            vel_x_abs = np.abs(vel_x)
            
            # Detect saccade onset
            onset_idx = detect_saccade_onset(
                vel_x_abs, timestamps,
                threshold=self.config.velocity_threshold,
                min_duration_samples=self.config.min_velocity_samples
            )
            
            if onset_idx is not None and onset_idx < len(timestamps):
                onset_time = timestamps[onset_idx]
                onset_gaze_x = gaze_x[onset_idx]
                
                # Latency
                latency_ms = (onset_time - jump_time) * 1000
                
                # Find peak velocity
                peak_vel, peak_idx = find_peak_velocity(vel_mag, onset_idx, len(vel_mag))
                peak_velocity = peak_vel
                
                # Find landing (velocity drops below threshold after peak)
                landing_idx = peak_idx
                for i in range(peak_idx, len(vel_x_abs)):
                    if vel_x_abs[i] < self.config.velocity_threshold:
                        landing_idx = i
                        break
                
                if landing_idx < len(timestamps):
                    landing_time = timestamps[landing_idx]
                    landing_gaze_x = gaze_x[landing_idx]
                    
                    # Duration
                    duration_ms = (landing_time - onset_time) * 1000
                    
                    # Amplitude (signed)
                    amplitude = landing_gaze_x - onset_gaze_x
                    
                    # Direction check
                    expected_sign = np.sign(eccentricity)
                    actual_sign = np.sign(amplitude)
                    direction_correct = (expected_sign == actual_sign)
                    
                    # Gain
                    if abs(eccentricity) > 1:
                        gain = amplitude / eccentricity
                    
                    # Landing error
                    landing_error = abs(landing_gaze_x - target_x)
                    
                    # Over/undershoot
                    if direction_correct:
                        if abs(landing_gaze_x - center_x) > abs(eccentricity):
                            overshoot = abs(landing_gaze_x - target_x)
                        else:
                            undershoot = abs(landing_gaze_x - target_x)
                    else:
                        undershoot = abs(eccentricity)
                    
                    # Count corrective saccades
                    if landing_idx < len(vel_x_abs) - 1:
                        post_landing_vel = vel_x_abs[landing_idx+1:]
                        in_saccade = False
                        for v in post_landing_vel:
                            if v > self.config.velocity_threshold and not in_saccade:
                                corrective_count += 1
                                in_saccade = True
                            elif v < self.config.velocity_threshold * 0.5:
                                in_saccade = False
        
        # Check saccade-specific validity
        saccade_validity = self.validity_checker.check_saccade_validity(
            latency_ms=latency_ms,
            duration_ms=duration_ms,
            amplitude_px=amplitude if not np.isnan(amplitude) else 0,
            eccentricity_px=eccentricity,
            peak_velocity=peak_velocity if not np.isnan(peak_velocity) else 0,
            direction_correct=direction_correct,
            dropout_in_critical_window=dropout_in_critical,
            generic_validity=generic_validity
        )
        
        self.trial_validities.append(saccade_validity)
        
        # Get calibration quality
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
            'saccade_amplitude_px': amplitude,
            'gain': gain,
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
        
        # FPS stats
        all_fps = [f.get('fps_est', 30.0) for f in self.trial_frames]
        if all_fps:
            meta.fps_mean = np.mean(all_fps)
            meta.fps_median = np.median(all_fps)
            meta.fps_min = np.min(all_fps)
            meta.fps_max = np.max(all_fps)
        
        meta.save(f"{self.session_dir}/meta.json")
