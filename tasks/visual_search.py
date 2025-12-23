"""Visual Search Task for NeuroLens+ eye tracking system.

This task covertly measures blink biomarkers while the user performs a visual search.
The user is NOT told about blink measurement - they simply search for targets in images.
This produces natural, unbiased blink patterns.
"""

import numpy as np
import time
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from .base import BaseTask, TaskConfig, TaskResult
from core.logging import FrameLogger, SummaryLogger
from core.validity import InvalidReason


# Summary columns for visual search task (covert blink measurement)
VISUAL_SEARCH_SUMMARY_COLUMNS = [
    'trial_id',
    'trial_start_s',
    'trial_end_s',
    'search_duration_s',
    'target_found',
    'search_time_ms',
    'blink_count',
    'blink_rate_per_min',
    'blink_duration_proxy_mean_ms',  # Renamed to indicate proxy measurement
    'blink_duration_proxy_std_ms',
    'blink_duration_valid_count',  # Number of blinks with valid duration (80-500ms)
    'interblink_interval_mean_s',
    'interblink_interval_std_s',
    'interblink_interval_cv',
    'blink_burstiness',
    'gaze_presence_pct',
    'valid_fraction',
    'valid',
    'invalid_reason',
    'qc_status',  # PASS / WARN / FAIL
    'qc_flags',  # Comma-separated list of QC issues
    'blink_rate_confidence',  # HIGH / LOW (based on trial duration)
    'quality_score',
    'calibration_quality',
    'fps_median',
    'clamp_rate'
]


# QC status levels
class QCStatus:
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


# Minimum trial duration for high-confidence blink rate
MIN_TRIAL_DURATION_FOR_HIGH_CONFIDENCE = 10.0  # seconds


@dataclass
class VisualSearchConfig(TaskConfig):
    """Configuration for visual search task (covert blink measurement)."""
    
    # Search duration per trial
    search_duration: float = 45.0  # seconds
    
    # Number of search trials
    n_trials: int = 2
    
    # Grid size for visual search
    grid_rows: int = 4
    grid_cols: int = 5
    
    # Target appearance
    target_symbol: str = "O"  # Target to find
    distractor_symbols: List[str] = field(default_factory=lambda: ["Q", "C", "G", "D"])
    
    # Minimum blinks required for valid trial
    min_blinks: int = 3
    
    # Break between trials
    break_duration: float = 5.0


class VisualSearchTask(BaseTask):
    """
    Visual Search Task - Covert Blink Measurement.
    
    User searches for target symbols in a grid while blinks are
    measured silently in the background. This produces natural,
    unbiased blink patterns without the Hawthorne effect.
    
    Measures (covertly):
    - Blink rate per minute
    - Blink duration (mean, std)
    - Interblink interval (mean, std, CV)
    - Blink burstiness (clustering)
    - Gaze presence percentage
    """
    
    TASK_NAME = "visual_search"
    
    def __init__(self, config: Optional[VisualSearchConfig] = None):
        """Initialize visual search task."""
        super().__init__(config or VisualSearchConfig())
        self.config: VisualSearchConfig = self.config
    
    def get_instructions(self) -> List[str]:
        """Get visual search task instructions.
        
        NOTE: Instructions do NOT mention blink measurement.
        """
        return [
            "VISUAL SEARCH TASK",
            "",
            f"Find the letter '{self.config.target_symbol}' in the grid.",
            "Press SPACE when you find it.",
            "",
            "The grid will contain many similar letters.",
            "Search carefully - accuracy matters more than speed.",
            "",
            f"There will be {self.config.n_trials} search trials.",
            f"Each trial lasts up to {int(self.config.search_duration)} seconds.",
            "",
            "Press SPACE to start.",
            "Press Q or ESC to quit."
        ]
    
    def get_summary_columns(self) -> List[str]:
        """Get summary CSV columns."""
        return VISUAL_SEARCH_SUMMARY_COLUMNS
    
    def run_task(self) -> TaskResult:
        """Run the visual search task."""
        result = TaskResult()
        
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
            
            trial_data = self._run_trial(trial_idx)
            
            if trial_data:
                self.summary_logger.log_trial(trial_data)
                trial_summaries.append(trial_data)
                
                self.debug_logger.info(
                    f"Trial {trial_idx + 1}: "
                    f"valid={trial_data.get('valid', 0)}, "
                    f"blink_rate={trial_data.get('blink_rate_per_min', np.nan):.1f}/min, "
                    f"gaze_presence={trial_data.get('gaze_presence_pct', np.nan):.1f}%"
                )
            
            # Break between trials (except after last)
            if trial_idx < self.config.n_trials - 1:
                self._show_break()
        
        # Compute results with session-level aggregation
        result.n_trials = len(trial_summaries)
        result.n_valid_trials = sum(1 for t in trial_summaries if t.get('valid', 0) == 1)
        result.valid_rate = result.n_valid_trials / result.n_trials if result.n_trials > 0 else 0.0
        
        # Session-level QC aggregation
        all_qc_statuses = [t.get('qc_status', QCStatus.FAIL) for t in trial_summaries]
        all_qc_flags = []
        for t in trial_summaries:
            flags = t.get('qc_flags', '')
            if flags:
                all_qc_flags.extend(flags.split(','))
        
        # Determine session-level QC status
        if all(s == QCStatus.PASS for s in all_qc_statuses):
            session_qc_status = QCStatus.PASS
        elif any(s == QCStatus.FAIL for s in all_qc_statuses):
            session_qc_status = QCStatus.FAIL
        else:
            session_qc_status = QCStatus.WARN
        
        # Count high-confidence trials
        high_confidence_trials = [t for t in trial_summaries if t.get('blink_rate_confidence') == 'HIGH']
        
        valid_trials = [t for t in trial_summaries if t.get('valid', 0) == 1]
        if valid_trials:
            # Session-level biomarker aggregation
            result.biomarkers = {
                # Blink rate (weighted by confidence)
                'session_blink_rate_per_min': np.nanmean([t.get('blink_rate_per_min', np.nan) for t in valid_trials]),
                'session_blink_rate_high_conf': np.nanmean([t.get('blink_rate_per_min', np.nan) for t in high_confidence_trials]) if high_confidence_trials else np.nan,
                
                # Blink duration (proxy)
                'session_blink_duration_proxy_ms': np.nanmean([t.get('blink_duration_proxy_mean_ms', np.nan) for t in valid_trials]),
                
                # Interblink interval
                'session_interblink_interval_s': np.nanmean([t.get('interblink_interval_mean_s', np.nan) for t in valid_trials]),
                'session_interblink_cv': np.nanmean([t.get('interblink_interval_cv', np.nan) for t in valid_trials]),
                
                # Burstiness
                'session_blink_burstiness': np.nanmean([t.get('blink_burstiness', np.nan) for t in valid_trials]),
                
                # Engagement
                'session_gaze_presence_pct': np.nanmean([t.get('gaze_presence_pct', np.nan) for t in valid_trials]),
                
                # QC summary
                'session_qc_status': session_qc_status,
                'session_qc_flags': list(set(all_qc_flags)),  # Unique flags
                'n_high_confidence_trials': len(high_confidence_trials),
                'n_pass_trials': sum(1 for s in all_qc_statuses if s == QCStatus.PASS),
                'n_warn_trials': sum(1 for s in all_qc_statuses if s == QCStatus.WARN),
                'n_fail_trials': sum(1 for s in all_qc_statuses if s == QCStatus.FAIL),
            }
        
        result.mean_quality_score = np.nanmean([t.get('quality_score', 0) for t in trial_summaries])
        
        self._show_results(result)
        self._save_metadata(result)
        
        result.success = True
        result.session_json_path = f"{self.session_dir}/meta.json"
        
        return result
    
    def _generate_grid(self) -> Tuple[List[List[str]], Tuple[int, int]]:
        """Generate search grid with target and distractors.
        
        Returns:
            Tuple of (grid, target_position)
        """
        grid = []
        
        # Place distractors
        for row in range(self.config.grid_rows):
            grid_row = []
            for col in range(self.config.grid_cols):
                symbol = random.choice(self.config.distractor_symbols)
                grid_row.append(symbol)
            grid.append(grid_row)
        
        # Place target at random position
        target_row = random.randint(0, self.config.grid_rows - 1)
        target_col = random.randint(0, self.config.grid_cols - 1)
        grid[target_row][target_col] = self.config.target_symbol
        
        return grid, (target_row, target_col)
    
    def _run_trial(self, trial_idx: int) -> Optional[Dict[str, Any]]:
        """Run a single visual search trial."""
        
        trial_start = time.time()
        trial_end = trial_start + self.config.search_duration
        
        # Generate search grid
        grid, target_pos = self._generate_grid()
        
        # Blink detection state
        blink_starts = []
        blink_ends = []
        in_blink = False
        
        # Gaze tracking
        valid_samples = 0
        total_samples = 0
        
        target_found = False
        search_time_ms = np.nan
        
        center_x = self.config.screen_width // 2
        center_y = self.config.screen_height // 2
        
        while time.time() < trial_end and self.ui.running and not target_found:
            current_time = time.time()
            
            events = self.ui.process_events()
            if events['quit']:
                self.running = False
                return None
            if events['space']:
                # User found target
                target_found = True
                search_time_ms = (current_time - trial_start) * 1000
                break
            if events['key_s']:
                return self._create_skipped_trial(trial_idx, trial_start)
            
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame_data = self.process_frame_with_mapping(frame, current_time)
            self.log_frame(trial_idx, "SEARCH", center_x, center_y, frame_data)
            
            total_samples += 1
            if frame_data.get('valid_sample', 0) == 1:
                valid_samples += 1
            
            # Covert blink detection
            blink_flag = frame_data.get('blink_flag', 0)
            
            if blink_flag == 1 and not in_blink:
                # Blink onset
                in_blink = True
                blink_starts.append(current_time)
            elif blink_flag == 0 and in_blink:
                # Blink offset
                in_blink = False
                blink_ends.append(current_time)
            
            # Draw search grid
            self._draw_search_grid(grid, target_pos if target_found else None)
            
            # Progress indicator (time remaining)
            remaining = max(0, trial_end - current_time)
            self.ui.draw_text(
                f"Time: {int(remaining)}s",
                self.config.screen_width - 80,
                30,
                font_size="small"
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
        
        # Handle blink at end
        if in_blink:
            blink_ends.append(time.time())
        
        actual_end = time.time()
        actual_duration = actual_end - trial_start
        
        # Compute biomarkers
        trial_data = self._compute_biomarkers(
            trial_idx, trial_start, actual_end, actual_duration,
            blink_starts, blink_ends,
            valid_samples, total_samples,
            target_found, search_time_ms
        )
        
        # Show feedback briefly
        if target_found:
            self._show_feedback("Found!", success=True)
        else:
            self._show_feedback("Time's up!", success=False)
        
        return trial_data
    
    def _draw_search_grid(
        self,
        grid: List[List[str]],
        highlight_pos: Optional[Tuple[int, int]] = None
    ):
        """Draw the visual search grid."""
        self.ui.clear_screen()
        
        # Calculate grid dimensions with safe margins
        margin = 0.1  # 10% margin
        usable_width = self.config.screen_width * (1 - 2 * margin)
        usable_height = self.config.screen_height * (1 - 2 * margin)
        
        cell_width = usable_width / self.config.grid_cols
        cell_height = usable_height / self.config.grid_rows
        
        start_x = self.config.screen_width * margin
        start_y = self.config.screen_height * margin
        
        # Draw grid cells
        for row in range(self.config.grid_rows):
            for col in range(self.config.grid_cols):
                x = start_x + col * cell_width + cell_width / 2
                y = start_y + row * cell_height + cell_height / 2
                
                symbol = grid[row][col]
                
                # Highlight target if found
                if highlight_pos and (row, col) == highlight_pos:
                    color = (0, 230, 118)  # Green
                else:
                    color = (215, 227, 255)  # Default text color
                
                self.ui.draw_text(symbol, int(x), int(y), font_size="large", color=color)
        
        # Instructions at bottom
        self.ui.draw_text(
            f"Find '{self.config.target_symbol}' - Press SPACE when found",
            self.config.screen_width // 2,
            self.config.screen_height - 30,
            font_size="small"
        )
    
    def _compute_biomarkers(
        self,
        trial_idx: int,
        trial_start: float,
        trial_end: float,
        duration_s: float,
        blink_starts: List[float],
        blink_ends: List[float],
        valid_samples: int,
        total_samples: int,
        target_found: bool,
        search_time_ms: float
    ) -> Dict[str, Any]:
        """Compute blink biomarkers (covertly measured).
        
        Blink duration is labeled as 'proxy' because webcam-based detection
        has limited temporal resolution. Valid duration range: 80-500ms.
        """
        
        generic_validity = self.compute_trial_validity()
        
        n_blinks = len(blink_starts)
        
        # Blink rate
        blink_rate = n_blinks / (duration_s / 60) if duration_s > 0 else np.nan
        
        # Blink rate confidence based on trial duration
        # Short trials (<10s) have LOW confidence for blink rate
        blink_rate_confidence = "HIGH" if duration_s >= MIN_TRIAL_DURATION_FOR_HIGH_CONFIDENCE else "LOW"
        
        # Blink durations (proxy measurement)
        # Valid range: 80-500ms (physiologically plausible blink duration)
        all_durations_ms = []
        valid_durations_ms = []
        for start, end in zip(blink_starts, blink_ends):
            dur = (end - start) * 1000
            all_durations_ms.append(dur)
            # Only count durations within valid physiological range
            if 80 <= dur <= 500:
                valid_durations_ms.append(dur)
        
        duration_proxy_mean = np.mean(valid_durations_ms) if valid_durations_ms else np.nan
        duration_proxy_std = np.std(valid_durations_ms) if len(valid_durations_ms) > 1 else np.nan
        duration_valid_count = len(valid_durations_ms)
        
        # Interblink intervals
        intervals = []
        if n_blinks > 1:
            for i in range(1, len(blink_starts)):
                interval = blink_starts[i] - blink_ends[i-1] if i-1 < len(blink_ends) else blink_starts[i] - blink_starts[i-1]
                if interval > 0:
                    intervals.append(interval)
        
        interval_mean = np.mean(intervals) if intervals else np.nan
        interval_std = np.std(intervals) if len(intervals) > 1 else np.nan
        
        # Coefficient of variation (irregularity)
        interval_cv = interval_std / interval_mean if interval_mean and interval_mean > 0 else np.nan
        
        # Burstiness: ratio of short intervals to expected
        # High burstiness = blinks clustered together
        if intervals and interval_mean > 0:
            short_threshold = interval_mean * 0.5
            n_short = sum(1 for i in intervals if i < short_threshold)
            burstiness = n_short / len(intervals) if intervals else 0
        else:
            burstiness = np.nan
        
        # Gaze presence (engagement metric)
        gaze_presence = (valid_samples / total_samples * 100) if total_samples > 0 else 0
        
        # QC flags and status
        qc_flags = []
        
        # Check for short trial duration
        if duration_s < MIN_TRIAL_DURATION_FOR_HIGH_CONFIDENCE:
            qc_flags.append("short_trial_duration")
        
        # Check for insufficient blinks
        if n_blinks < self.config.min_blinks:
            qc_flags.append("insufficient_blinks")
        
        # Check for low gaze presence
        if gaze_presence < 50:
            qc_flags.append("low_gaze_presence")
        
        # Check for low valid fraction
        if generic_validity.valid_fraction < 0.7:
            qc_flags.append("low_valid_fraction")
        
        # Check for high clamp rate
        if generic_validity.clamp_rate > 0.25:
            qc_flags.append("high_clamp_rate")
        
        # Check for low FPS
        if generic_validity.fps_median < 15:
            qc_flags.append("low_fps")
        
        # Check for few valid blink durations
        if n_blinks > 0 and duration_valid_count < n_blinks * 0.5:
            qc_flags.append("many_invalid_blink_durations")
        
        # Determine QC status
        if len(qc_flags) == 0:
            qc_status = QCStatus.PASS
        elif any(f in qc_flags for f in ["insufficient_blinks", "low_gaze_presence", "low_valid_fraction"]):
            qc_status = QCStatus.FAIL
        else:
            qc_status = QCStatus.WARN
        
        # Validity check (for backward compatibility)
        valid = 1 if qc_status != QCStatus.FAIL else 0
        
        if valid == 0:
            if n_blinks < self.config.min_blinks:
                reason = InvalidReason.INSUFFICIENT_SAMPLES
            elif gaze_presence < 50:
                reason = InvalidReason.LOW_TRACKING_QUALITY
            else:
                reason = InvalidReason.GENERIC_INVALID
        else:
            reason = InvalidReason.VALID
        
        self.trial_validities.append(generic_validity)
        
        calib_quality = self.calibrator.get_quality_score() if self.calibrator else 0.0
        
        return {
            'trial_id': trial_idx,
            'trial_start_s': trial_start,
            'trial_end_s': trial_end,
            'search_duration_s': duration_s,
            'target_found': 1 if target_found else 0,
            'search_time_ms': search_time_ms,
            'blink_count': n_blinks,
            'blink_rate_per_min': blink_rate,
            'blink_duration_proxy_mean_ms': duration_proxy_mean,
            'blink_duration_proxy_std_ms': duration_proxy_std,
            'blink_duration_valid_count': duration_valid_count,
            'interblink_interval_mean_s': interval_mean,
            'interblink_interval_std_s': interval_std,
            'interblink_interval_cv': interval_cv,
            'blink_burstiness': burstiness,
            'gaze_presence_pct': gaze_presence,
            'valid_fraction': generic_validity.valid_fraction,
            'valid': valid,
            'invalid_reason': reason.value,
            'qc_status': qc_status,
            'qc_flags': ','.join(qc_flags) if qc_flags else '',
            'blink_rate_confidence': blink_rate_confidence,
            'quality_score': generic_validity.quality_score,
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
            'search_duration_s': 0,
            'target_found': 0,
            'search_time_ms': np.nan,
            'blink_count': 0,
            'blink_rate_per_min': np.nan,
            'blink_duration_proxy_mean_ms': np.nan,
            'blink_duration_proxy_std_ms': np.nan,
            'blink_duration_valid_count': 0,
            'interblink_interval_mean_s': np.nan,
            'interblink_interval_std_s': np.nan,
            'interblink_interval_cv': np.nan,
            'blink_burstiness': np.nan,
            'gaze_presence_pct': 0,
            'valid_fraction': 0.0,
            'valid': 0,
            'invalid_reason': InvalidReason.SKIPPED.value,
            'qc_status': QCStatus.FAIL,
            'qc_flags': 'skipped',
            'blink_rate_confidence': 'LOW',
            'quality_score': 0.0,
            'calibration_quality': 0.0,
            'fps_median': 0.0,
            'clamp_rate': 0.0
        }
    
    def _show_feedback(self, message: str, success: bool):
        """Show brief feedback after trial."""
        color = (0, 230, 118) if success else (255, 196, 0)
        
        feedback_end = time.time() + 1.5  # 1.5 second feedback
        
        while time.time() < feedback_end and self.ui.running:
            events = self.ui.process_events()
            if events['quit']:
                return
            
            self.ui.clear_screen()
            self.ui.draw_text(
                message,
                self.config.screen_width // 2,
                self.config.screen_height // 2,
                font_size="large",
                color=color
            )
            self.ui.update_display()
            self.ui.tick(60)
    
    def _show_break(self):
        """Show break screen between trials."""
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
                f"Next search in {remaining} seconds",
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
        """Show results screen.
        
        NOTE: Results shown are search performance, not blink metrics.
        Blink metrics are recorded but not displayed to maintain covert measurement.
        """
        # Show search performance (not blink metrics)
        metrics = {
            'Trials Completed': f"{result.n_trials}",
            'Valid Trials': f"{result.n_valid_trials}/{result.n_trials}",
            'Quality Score': f"{result.mean_quality_score:.2f}",
        }
        
        self.ui.draw_results("Visual Search Complete", metrics, result.valid_rate)
        
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
                'search_duration': self.config.search_duration,
                'grid_size': f"{self.config.grid_rows}x{self.config.grid_cols}",
                'covert_blink_measurement': True,
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
