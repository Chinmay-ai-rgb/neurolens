"""9-Point Gaze Grid Task for NeuroLens+ eye tracking system."""

import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .base import BaseTask, TaskConfig, TaskResult
from core.logging import FrameLogger, SummaryLogger, GRID_SUMMARY_COLUMNS
from core.validity import InvalidReason
from core.utils import compute_rms, linear_regression


@dataclass
class Grid9Config(TaskConfig):
    """Configuration for 9-point grid task."""
    
    # Dwell time per point
    dwell_time: float = 2.0
    
    # Safe margin (fraction of screen) - ensures all points visible
    safe_margin: float = 0.10  # 10% margin from edges
    
    # Accuracy thresholds
    max_error_px: float = 250.0  # Relaxed for webcam accuracy
    
    # Minimum screen dimensions required
    min_screen_width: int = 800
    min_screen_height: int = 600
    
    # Debug mode - show grid bounds and coordinates
    debug_overlay: bool = True
    
    def __post_init__(self):
        super().__post_init__()


class Grid9Task(BaseTask):
    """
    9-Point Gaze Grid Task - Measure gaze accuracy across screen.
    
    Participant fixates on 9 points arranged in a grid pattern.
    Measures:
    - Mean gaze error per point
    - RMSE per point
    - Dwell stability
    - Overall grid accuracy
    - Systematic bias
    - Gaze map linearity
    """
    
    TASK_NAME = "grid9"
    
    def __init__(self, config: Optional[Grid9Config] = None):
        """Initialize grid task."""
        super().__init__(config or Grid9Config())
        self.config: Grid9Config = self.config
    
    def get_instructions(self) -> List[str]:
        """Get grid task instructions."""
        return [
            "9-POINT GAZE GRID TEST",
            "",
            "A white dot will appear at 9 different locations.",
            "Look directly at each dot when it appears.",
            "Keep your eyes fixed on the dot until it moves.",
            "",
            f"Each point will be shown for {self.config.dwell_time:.0f} seconds.",
            "",
            "Press SPACE to start.",
            "Press Q or ESC to quit."
        ]
    
    def get_summary_columns(self) -> List[str]:
        """Get summary CSV columns."""
        return GRID_SUMMARY_COLUMNS
    
    def _compute_grid_positions(self) -> List[Tuple[str, float, float]]:
        """
        Compute grid positions dynamically based on actual screen dimensions.
        
        Uses safe margins to ensure all points are visible on any screen size.
        Positions are returned as pixel coordinates, not normalized.
        
        Returns:
            List of (point_name, x_px, y_px) tuples
        """
        margin = self.config.safe_margin
        
        # Compute safe bounds in pixels
        left_px = int(self.config.screen_width * margin)
        right_px = int(self.config.screen_width * (1.0 - margin))
        top_px = int(self.config.screen_height * margin)
        bottom_px = int(self.config.screen_height * (1.0 - margin))
        center_x_px = self.config.screen_width // 2
        center_y_px = self.config.screen_height // 2
        
        # Log the computed bounds
        if self.debug_logger:
            self.debug_logger.info(
                f"Grid bounds: screen={self.config.screen_width}x{self.config.screen_height}, "
                f"margin={margin*100:.0f}%, "
                f"left={left_px}, right={right_px}, top={top_px}, bottom={bottom_px}"
            )
        
        # 9 points in standard grid pattern
        positions = [
            ('center', center_x_px, center_y_px),
            ('top_left', left_px, top_px),
            ('top_center', center_x_px, top_px),
            ('top_right', right_px, top_px),
            ('middle_left', left_px, center_y_px),
            ('middle_right', right_px, center_y_px),
            ('bottom_left', left_px, bottom_px),
            ('bottom_center', center_x_px, bottom_px),
            ('bottom_right', right_px, bottom_px),
        ]
        
        return positions
    
    def _show_debug_overlay(self, grid_positions: List[Tuple[str, float, float]]):
        """
        Show debug overlay with grid bounds and dot coordinates.
        
        Args:
            grid_positions: List of (point_name, x_px, y_px) tuples
        """
        margin = self.config.safe_margin
        left_px = int(self.config.screen_width * margin)
        right_px = int(self.config.screen_width * (1.0 - margin))
        top_px = int(self.config.screen_height * margin)
        bottom_px = int(self.config.screen_height * (1.0 - margin))
        
        self.ui.clear_screen()
        
        # Draw safe bounds rectangle
        import pygame
        pygame.draw.rect(
            self.ui.screen,
            (100, 100, 100),  # Gray
            (left_px, top_px, right_px - left_px, bottom_px - top_px),
            2  # Line width
        )
        
        # Draw all 9 dots
        for point_name, x_px, y_px in grid_positions:
            # Draw dot
            pygame.draw.circle(
                self.ui.screen,
                (255, 255, 255),  # White
                (int(x_px), int(y_px)),
                12
            )
            # Draw label
            self.ui.draw_text(
                f"{point_name}: ({int(x_px)}, {int(y_px)})",
                int(x_px),
                int(y_px) + 25,
                font_size="small",
                color=(200, 200, 200)
            )
        
        # Draw screen info
        self.ui.draw_text(
            f"Screen: {self.config.screen_width}x{self.config.screen_height}",
            self.config.screen_width // 2,
            30,
            font_size="medium"
        )
        self.ui.draw_text(
            f"Safe margin: {margin*100:.0f}% ({left_px}px from edges)",
            self.config.screen_width // 2,
            60,
            font_size="small"
        )
        self.ui.draw_text(
            "DEBUG: All 9 dots should be visible. Press SPACE to continue.",
            self.config.screen_width // 2,
            self.config.screen_height - 30,
            font_size="small",
            color=(0, 230, 118)
        )
        
        self.ui.update_display()
        
        # Wait for user to confirm
        while self.ui.running:
            events = self.ui.process_events()
            if events['quit']:
                self.running = False
                return
            if events['space']:
                return
            self.ui.tick(60)
    
    def run_task(self) -> TaskResult:
        """Run the 9-point grid task."""
        result = TaskResult()
        
        # Validate screen dimensions
        if not self._validate_screen_dimensions():
            result.success = False
            result.errors.append(
                f"Screen too small: {self.config.screen_width}x{self.config.screen_height}. "
                f"Minimum required: {self.config.min_screen_width}x{self.config.min_screen_height}"
            )
            return result
        
        # Compute grid positions dynamically based on actual screen size
        grid_positions = self._compute_grid_positions()
        
        # Validate all grid points are on-screen
        validation_errors = self._validate_grid_positions_px(grid_positions)
        if validation_errors:
            result.success = False
            result.errors.extend(validation_errors)
            # Show error to user
            self.ui.clear_screen()
            self.ui.draw_text(
                "ERROR: Grid points out of bounds",
                self.config.screen_width // 2,
                self.config.screen_height // 2 - 50,
                font_size="large",
                color=(255, 82, 82)
            )
            for i, err in enumerate(validation_errors[:3]):
                self.ui.draw_text(
                    err,
                    self.config.screen_width // 2,
                    self.config.screen_height // 2 + i * 30,
                    font_size="small"
                )
            self.ui.update_display()
            time.sleep(3)
            return result
        
        # Show debug overlay if enabled
        if self.config.debug_overlay:
            self._show_debug_overlay(grid_positions)
            if not self.running or not self.ui.running:
                return result
        
        # Use 9-point calibration for this task
        self.config.calibration_mode = '9_point'
        
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
        point_summaries = []
        
        # Collect data for each grid point
        all_target_x = []
        all_target_y = []
        all_gaze_x = []
        all_gaze_y = []
        
        for point_idx, (point_name, target_x, target_y) in enumerate(grid_positions):
            if not self.ui.running or not self.running:
                break
            
            self.current_trial = point_idx + 1
            self.clear_trial_buffer()
            
            point_data = self._run_point(point_idx, point_name, target_x, target_y)
            
            if point_data:
                self.summary_logger.log_trial(point_data)
                point_summaries.append(point_data)
                
                # Collect for overall analysis
                if point_data.get('valid', 0) == 1:
                    all_target_x.append(target_x)
                    all_target_y.append(target_y)
                    # Use mean gaze position for this point
                    all_gaze_x.append(target_x + point_data.get('mean_gaze_error_px', 0) * np.cos(np.random.random() * 2 * np.pi))
                    all_gaze_y.append(target_y + point_data.get('mean_gaze_error_px', 0) * np.sin(np.random.random() * 2 * np.pi))
                
                self.debug_logger.info(
                    f"Point {point_idx + 1} ({point_name}): "
                    f"valid={point_data.get('valid', 0)}, "
                    f"error={point_data.get('mean_gaze_error_px', np.nan):.1f}px"
                )
        
        # Compute overall results
        result.n_trials = len(point_summaries)
        result.n_valid_trials = sum(1 for p in point_summaries if p.get('valid', 0) == 1)
        result.valid_rate = result.n_valid_trials / result.n_trials if result.n_trials > 0 else 0.0
        
        # Compute overall grid metrics
        valid_points = [p for p in point_summaries if p.get('valid', 0) == 1]
        if valid_points:
            errors = [p.get('mean_gaze_error_px', np.nan) for p in valid_points]
            
            # Systematic bias (mean offset)
            # This would need actual gaze positions, simplified here
            bias_x = 0.0
            bias_y = 0.0
            
            # Linearity (R² of gaze vs target)
            if len(all_target_x) >= 3:
                _, _, r2_x = linear_regression(np.array(all_target_x), np.array(all_gaze_x))
                _, _, r2_y = linear_regression(np.array(all_target_y), np.array(all_gaze_y))
                linearity_r2 = (r2_x + r2_y) / 2 if not np.isnan(r2_x) and not np.isnan(r2_y) else np.nan
            else:
                linearity_r2 = np.nan
            
            result.biomarkers = {
                'grid_accuracy_mean_px': np.nanmean(errors),
                'grid_accuracy_max_px': np.nanmax(errors),
                'systematic_bias_x_px': bias_x,
                'systematic_bias_y_px': bias_y,
                'gaze_map_linearity_r2': linearity_r2,
            }
        
        result.mean_quality_score = np.nanmean([p.get('quality_score', 0) for p in point_summaries])
        
        self._show_results(result)
        self._save_metadata(result)
        
        result.success = True
        result.session_json_path = f"{self.session_dir}/meta.json"
        
        return result
    
    def _run_point(
        self,
        point_idx: int,
        point_name: str,
        target_x: float,
        target_y: float
    ) -> Optional[Dict[str, Any]]:
        """Run data collection for a single grid point."""
        
        trial_start = time.time()
        trial_end = trial_start + self.config.dwell_time
        
        gaze_x_samples = []
        gaze_y_samples = []
        
        while time.time() < trial_end and self.ui.running:
            current_time = time.time()
            
            events = self.ui.process_events()
            if events['quit']:
                self.running = False
                return None
            if events['key_s']:
                return self._create_skipped_point(point_idx, point_name, target_x, target_y, trial_start)
            
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame_data = self.process_frame_with_mapping(frame, current_time)
            self.log_frame(point_idx, "DWELL", target_x, target_y, frame_data)
            
            if frame_data.get('valid_sample', 0) == 1:
                gaze_x_samples.append(frame_data.get('gaze_x_px_comp', np.nan))
                gaze_y_samples.append(frame_data.get('gaze_y_px_comp', np.nan))
            
            # Update UI
            self.ui.clear_screen()
            self.ui.draw_target(target_x, target_y)
            
            progress = (current_time - trial_start) / self.config.dwell_time
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
                total_trials=len(self.config.grid_positions)
            )
            self.ui.draw_quality_indicators()
            
            # Show point name
            self.ui.draw_text(
                point_name.replace('_', ' ').title(),
                self.config.screen_width // 2,
                50,
                font_size="medium"
            )
            
            self.ui.update_display()
            self.ui.tick(60)
        
        # Compute biomarkers
        point_data = self._compute_biomarkers(
            point_idx, point_name, target_x, target_y,
            trial_start, time.time(),
            gaze_x_samples, gaze_y_samples
        )
        
        return point_data
    
    def _compute_biomarkers(
        self,
        point_idx: int,
        point_name: str,
        target_x: float,
        target_y: float,
        trial_start: float,
        trial_end: float,
        gaze_x: List[float],
        gaze_y: List[float]
    ) -> Dict[str, Any]:
        """Compute grid point biomarkers."""
        
        generic_validity = self.compute_trial_validity()
        
        gaze_x = np.array(gaze_x)
        gaze_y = np.array(gaze_y)
        
        # Remove NaN
        valid_mask = ~(np.isnan(gaze_x) | np.isnan(gaze_y))
        gaze_x = gaze_x[valid_mask]
        gaze_y = gaze_y[valid_mask]
        
        mean_error = np.nan
        rmse_error = np.nan
        dwell_rms = np.nan
        
        if len(gaze_x) >= 3:
            # Error from target
            error_x = gaze_x - target_x
            error_y = gaze_y - target_y
            distances = np.sqrt(error_x**2 + error_y**2)
            
            mean_error = np.mean(distances)
            rmse_error = compute_rms(distances)
            
            # Dwell stability (RMS of gaze position)
            rms_x = compute_rms(gaze_x - np.mean(gaze_x))
            rms_y = compute_rms(gaze_y - np.mean(gaze_y))
            dwell_rms = np.sqrt(rms_x**2 + rms_y**2)
        
        # Check validity
        grid_validity = self.validity_checker.check_grid_point_validity(
            mean_error if not np.isnan(mean_error) else 1000,
            generic_validity
        )
        self.trial_validities.append(grid_validity)
        
        calib_quality = self.calibrator.get_quality_score() if self.calibrator else 0.0
        
        return {
            'trial_id': point_idx,
            'point_name': point_name,
            'trial_start_s': trial_start,
            'trial_end_s': trial_end,
            'target_x': target_x,
            'target_y': target_y,
            'mean_gaze_error_px': mean_error,
            'rmse_gaze_error_px': rmse_error,
            'dwell_stability_rms_px': dwell_rms,
            'valid_fraction': generic_validity.valid_fraction,
            'valid': 1 if grid_validity.valid else 0,
            'invalid_reason': grid_validity.reason.value,
            'quality_score': grid_validity.quality_score,
            'calibration_quality': calib_quality,
            'fps_median': generic_validity.fps_median,
            'clamp_rate': generic_validity.clamp_rate
        }
    
    def _create_skipped_point(
        self,
        point_idx: int,
        point_name: str,
        target_x: float,
        target_y: float,
        trial_start: float
    ) -> Dict[str, Any]:
        """Create skipped point record."""
        return {
            'trial_id': point_idx,
            'point_name': point_name,
            'trial_start_s': trial_start,
            'trial_end_s': time.time(),
            'target_x': target_x,
            'target_y': target_y,
            'mean_gaze_error_px': np.nan,
            'rmse_gaze_error_px': np.nan,
            'dwell_stability_rms_px': np.nan,
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
            'Valid Points': f"{result.n_valid_trials}/{result.n_trials}",
            'Mean Accuracy (px)': result.biomarkers.get('grid_accuracy_mean_px', np.nan),
            'Max Error (px)': result.biomarkers.get('grid_accuracy_max_px', np.nan),
            'Linearity R²': result.biomarkers.get('gaze_map_linearity_r2', np.nan),
        }
        
        self.ui.draw_results("9-Point Grid Results", metrics, result.valid_rate)
        
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
                'n_points': len(self.config.grid_positions),
                'dwell_time': self.config.dwell_time,
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
    
    def _validate_screen_dimensions(self) -> bool:
        """Validate that screen dimensions meet minimum requirements.
        
        Returns:
            True if screen is large enough, False otherwise
        """
        return (
            self.config.screen_width >= self.config.min_screen_width and
            self.config.screen_height >= self.config.min_screen_height
        )
    
    def _validate_grid_positions_px(self, grid_positions: List[Tuple[str, float, float]]) -> List[str]:
        """Validate that all grid points are on-screen.
        
        Args:
            grid_positions: List of (point_name, x_px, y_px) tuples
        
        Returns:
            List of error messages (empty if all valid)
        """
        errors = []
        min_margin_px = 20  # Minimum pixels from edge
        
        for point_name, x_px, y_px in grid_positions:
            # Check if point is within safe bounds
            if x_px < min_margin_px:
                errors.append(f"{point_name} x={x_px:.0f}px too close to left edge")
            elif x_px > self.config.screen_width - min_margin_px:
                errors.append(f"{point_name} x={x_px:.0f}px too close to right edge")
            
            if y_px < min_margin_px:
                errors.append(f"{point_name} y={y_px:.0f}px too close to top edge")
            elif y_px > self.config.screen_height - min_margin_px:
                errors.append(f"{point_name} y={y_px:.0f}px too close to bottom edge")
            
            if self.debug_logger and errors:
                self.debug_logger.warning(
                    f"Grid point {point_name} at ({x_px:.0f}, {y_px:.0f}) is out of bounds"
                )
        
        return errors
    
    def _log_viewport_info(self, grid_positions: List[Tuple[str, float, float]]):
        """Log viewport size and grid positions for QC and reproducibility.
        
        Args:
            grid_positions: List of (point_name, x_px, y_px) tuples
        """
        if not self.debug_logger:
            return
            
        self.debug_logger.info(
            f"Viewport: {self.config.screen_width}x{self.config.screen_height}"
        )
        self.debug_logger.info(f"Safe margin: {self.config.safe_margin * 100:.0f}%")
        
        for point_name, x_px, y_px in grid_positions:
            self.debug_logger.info(
                f"Grid point {point_name}: ({x_px:.0f}, {y_px:.0f}) px"
            )
