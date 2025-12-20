"""Blink Rate Monitoring Task for NeuroLens+ eye tracking system."""

import numpy as np
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base import BaseTask, TaskConfig, TaskResult
from core.logging import FrameLogger, SummaryLogger, BLINK_SUMMARY_COLUMNS
from core.validity import InvalidReason


@dataclass
class BlinkConfig(TaskConfig):
    """Configuration for blink monitoring task."""
    
    # Monitoring duration
    monitoring_duration: float = 60.0  # 1 minute
    
    # Number of monitoring periods
    n_periods: int = 3
    
    # Break between periods
    break_duration: float = 10.0


class BlinkTask(BaseTask):
    """
    Blink Rate Monitoring Task.
    
    Monitors spontaneous blink rate during passive viewing.
    Measures:
    - Blink rate per minute
    - Blink duration (mean, P95)
    - Interblink interval
    - Blink irregularity (CV)
    """
    
    TASK_NAME = "blink"
    
    def __init__(self, config: Optional[BlinkConfig] = None):
        """Initialize blink task."""
        super().__init__(config or BlinkConfig())
        self.config: BlinkConfig = self.config
    
    def get_instructions(self) -> List[str]:
        """Get blink task instructions."""
        return [
            "BLINK RATE MONITORING",
            "",
            "Look at the center of the screen.",
            "Blink naturally - do not try to control your blinks.",
            "Stay relaxed and comfortable.",
            "",
            f"Each monitoring period lasts {self.config.monitoring_duration:.0f} seconds.",
            f"There will be {self.config.n_periods} periods with short breaks.",
            "",
            "Press SPACE to start.",
            "Press Q or ESC to quit."
        ]
    
    def get_summary_columns(self) -> List[str]:
        """Get summary CSV columns."""
        return BLINK_SUMMARY_COLUMNS
    
    def run_task(self) -> TaskResult:
        """Run the blink monitoring task."""
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
        period_summaries = []
        
        for period_idx in range(self.config.n_periods):
            if not self.ui.running or not self.running:
                break
            
            self.current_trial = period_idx + 1
            self.clear_trial_buffer()
            
            period_data = self._run_period(period_idx, center_x, center_y)
            
            if period_data:
                self.summary_logger.log_trial(period_data)
                period_summaries.append(period_data)
                
                self.debug_logger.info(
                    f"Period {period_idx + 1}: "
                    f"valid={period_data.get('valid', 0)}, "
                    f"blink_rate={period_data.get('blink_rate_per_min', np.nan):.1f}/min"
                )
            
            # Break between periods (except after last)
            if period_idx < self.config.n_periods - 1:
                self._show_break()
        
        # Compute results
        result.n_trials = len(period_summaries)
        result.n_valid_trials = sum(1 for p in period_summaries if p.get('valid', 0) == 1)
        result.valid_rate = result.n_valid_trials / result.n_trials if result.n_trials > 0 else 0.0
        
        valid_periods = [p for p in period_summaries if p.get('valid', 0) == 1]
        if valid_periods:
            result.biomarkers = {
                'mean_blink_rate_per_min': np.nanmean([p.get('blink_rate_per_min', np.nan) for p in valid_periods]),
                'mean_blink_duration_ms': np.nanmean([p.get('blink_duration_mean_ms', np.nan) for p in valid_periods]),
                'mean_interblink_interval_s': np.nanmean([p.get('interblink_interval_mean_s', np.nan) for p in valid_periods]),
                'mean_irregularity_cv': np.nanmean([p.get('blink_irregularity_cv', np.nan) for p in valid_periods]),
            }
        
        result.mean_quality_score = np.nanmean([p.get('quality_score', 0) for p in period_summaries])
        
        self._show_results(result)
        self._save_metadata(result)
        
        result.success = True
        result.session_json_path = f"{self.session_dir}/meta.json"
        
        return result
    
    def _run_period(
        self,
        period_idx: int,
        center_x: float,
        center_y: float
    ) -> Optional[Dict[str, Any]]:
        """Run a single monitoring period."""
        
        period_start = time.time()
        period_end = period_start + self.config.monitoring_duration
        
        # Blink detection state
        blink_starts = []
        blink_ends = []
        in_blink = False
        last_blink_start = None
        
        timestamps = []
        
        while time.time() < period_end and self.ui.running:
            current_time = time.time()
            
            events = self.ui.process_events()
            if events['quit']:
                self.running = False
                return None
            if events['key_s']:
                return self._create_skipped_period(period_idx, period_start)
            
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame_data = self.process_frame_with_mapping(frame, current_time)
            self.log_frame(period_idx, "MONITOR", center_x, center_y, frame_data)
            
            timestamps.append(current_time)
            
            # Blink detection
            blink_flag = frame_data.get('blink_flag', 0)
            
            if blink_flag == 1 and not in_blink:
                # Blink onset
                in_blink = True
                last_blink_start = current_time
                blink_starts.append(current_time)
            elif blink_flag == 0 and in_blink:
                # Blink offset
                in_blink = False
                blink_ends.append(current_time)
            
            # Update UI
            self.ui.clear_screen()
            self.ui.draw_target(center_x, center_y, radius=10)
            
            # Progress bar
            progress = (current_time - period_start) / self.config.monitoring_duration
            self.ui.draw_progress_bar(
                progress,
                self.config.screen_width // 2,
                self.config.screen_height - 50
            )
            
            # Blink counter
            self.ui.draw_text(
                f"Blinks detected: {len(blink_starts)}",
                self.config.screen_width // 2,
                100,
                font_size="medium"
            )
            
            self.ui.update_quality_indicators(
                face_detected=frame_data.get('face_present', 0) == 1,
                fps=frame_data.get('fps_est', 30.0),
                quality_score=frame_data.get('valid_sample', 0),
                trial_count=self.current_trial,
                total_trials=self.config.n_periods
            )
            self.ui.draw_quality_indicators()
            self.ui.update_display()
            self.ui.tick(60)
        
        # Handle blink at end
        if in_blink:
            blink_ends.append(time.time())
        
        actual_end = time.time()
        
        # Compute biomarkers
        period_data = self._compute_biomarkers(
            period_idx, period_start, actual_end,
            blink_starts, blink_ends, timestamps
        )
        
        return period_data
    
    def _compute_biomarkers(
        self,
        period_idx: int,
        period_start: float,
        period_end: float,
        blink_starts: List[float],
        blink_ends: List[float],
        timestamps: List[float]
    ) -> Dict[str, Any]:
        """Compute blink biomarkers."""
        
        generic_validity = self.compute_trial_validity()
        
        duration_s = period_end - period_start
        n_blinks = len(blink_starts)
        
        # Blink rate
        blink_rate = n_blinks / (duration_s / 60) if duration_s > 0 else np.nan
        
        # Blink durations
        durations_ms = []
        for start, end in zip(blink_starts, blink_ends):
            dur = (end - start) * 1000
            if dur > 0 and dur < 1000:  # Reasonable blink duration
                durations_ms.append(dur)
        
        duration_mean = np.mean(durations_ms) if durations_ms else np.nan
        duration_p95 = np.percentile(durations_ms, 95) if len(durations_ms) > 1 else duration_mean
        
        # Interblink intervals
        intervals = []
        if n_blinks > 1:
            for i in range(1, len(blink_starts)):
                interval = blink_starts[i] - blink_ends[i-1]
                if interval > 0:
                    intervals.append(interval)
        
        interval_mean = np.mean(intervals) if intervals else np.nan
        
        # Irregularity (coefficient of variation)
        if len(intervals) > 1:
            interval_std = np.std(intervals)
            irregularity_cv = interval_std / interval_mean if interval_mean > 0 else np.nan
        else:
            irregularity_cv = np.nan
        
        # Simple validity check
        valid = 1 if (generic_validity.valid_fraction >= 0.7 and n_blinks >= 3) else 0
        reason = InvalidReason.VALID if valid else InvalidReason.INSUFFICIENT_SAMPLES
        
        self.trial_validities.append(generic_validity)
        
        calib_quality = self.calibrator.get_quality_score() if self.calibrator else 0.0
        
        return {
            'trial_id': period_idx,
            'trial_start_s': period_start,
            'trial_end_s': period_end,
            'blink_rate_per_min': blink_rate,
            'blink_duration_mean_ms': duration_mean,
            'blink_duration_p95_ms': duration_p95,
            'interblink_interval_mean_s': interval_mean,
            'blink_irregularity_cv': irregularity_cv,
            'valid_fraction': generic_validity.valid_fraction,
            'valid': valid,
            'invalid_reason': reason.value,
            'quality_score': generic_validity.quality_score,
            'calibration_quality': calib_quality,
            'fps_median': generic_validity.fps_median,
            'clamp_rate': generic_validity.clamp_rate
        }
    
    def _create_skipped_period(
        self,
        period_idx: int,
        period_start: float
    ) -> Dict[str, Any]:
        """Create skipped period record."""
        return {
            'trial_id': period_idx,
            'trial_start_s': period_start,
            'trial_end_s': time.time(),
            'blink_rate_per_min': np.nan,
            'blink_duration_mean_ms': np.nan,
            'blink_duration_p95_ms': np.nan,
            'interblink_interval_mean_s': np.nan,
            'blink_irregularity_cv': np.nan,
            'valid_fraction': 0.0,
            'valid': 0,
            'invalid_reason': InvalidReason.SKIPPED.value,
            'quality_score': 0.0,
            'calibration_quality': 0.0,
            'fps_median': 0.0,
            'clamp_rate': 0.0
        }
    
    def _show_break(self):
        """Show break screen between periods."""
        break_end = time.time() + self.config.break_duration
        
        while time.time() < break_end and self.ui.running:
            remaining = int(break_end - time.time())
            
            events = self.ui.process_events()
            if events['quit']:
                self.running = False
                return
            if events['space']:
                return  # Skip break
            
            self.ui.clear_screen()
            self.ui.draw_text(
                "Break",
                self.config.screen_width // 2,
                self.config.screen_height // 2 - 50,
                font_size="large"
            )
            self.ui.draw_text(
                f"Next period in {remaining} seconds",
                self.config.screen_width // 2,
                self.config.screen_height // 2 + 20,
                font_size="medium"
            )
            self.ui.draw_text(
                "Press SPACE to continue early",
                self.config.screen_width // 2,
                self.config.screen_height - 50,
                font_size="small"
            )
            self.ui.update_display()
            self.ui.tick(60)
    
    def _show_results(self, result: TaskResult):
        """Show results screen."""
        metrics = {
            'Valid Periods': f"{result.n_valid_trials}/{result.n_trials}",
            'Mean Blink Rate (/min)': result.biomarkers.get('mean_blink_rate_per_min', np.nan),
            'Mean Blink Duration (ms)': result.biomarkers.get('mean_blink_duration_ms', np.nan),
            'Mean Interblink Interval (s)': result.biomarkers.get('mean_interblink_interval_s', np.nan),
            'Irregularity (CV)': result.biomarkers.get('mean_irregularity_cv', np.nan),
        }
        
        self.ui.draw_results("Blink Monitoring Results", metrics, result.valid_rate)
        
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
                'n_periods': self.config.n_periods,
                'monitoring_duration': self.config.monitoring_duration,
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
