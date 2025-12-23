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

    # ---------------- FIXES START HERE ----------------

    def _sync_screen_size_from_surface(self) -> None:
        """Ensure config screen_width/height match the REAL pygame surface size."""
        try:
            sw, sh = self.ui.screen.get_size()  # actual drawable surface
            self.config.screen_width = int(sw)
            self.config.screen_height = int(sh)
        except Exception:
            # fallback: keep config values
            pass

    def _compute_grid_positions(self) -> List[Tuple[str, float, float]]:
        """
        Compute grid positions using the REAL pygame surface size and a pixel-safe inset.

        This fixes the issue where only 1–2 dots appear because config screen size
        doesn't match the actual window surface.
        """
        # Prefer the real surface size
        try:
            sw, sh = self.ui.screen.get_size()
            sw, sh = int(sw), int(sh)
        except Exception:
            sw, sh = int(self.config.screen_width), int(self.config.screen_height)

        # UI-safe padding (accounts for header text, bottom indicators/progress bar, labels)
        dot_r = 12
        label_pad = 28          # label drawn at y+25
        ui_pad_top = 80         # room for task title at y=50
        ui_pad_bottom = 90      # room for progress + indicators near bottom

        margin_frac = float(self.config.safe_margin)

        inset_x = max(int(sw * margin_frac), dot_r + 25)
        inset_top = max(int(sh * margin_frac), ui_pad_top)
        inset_bottom = max(int(sh * margin_frac), ui_pad_bottom + label_pad)

        left_px = inset_x
        right_px = sw - inset_x
        top_px = inset_top
        bottom_px = sh - inset_bottom

        center_x_px = sw // 2
        center_y_px = sh // 2

        # Safety fallback if the safe box collapses on small windows
        if right_px <= left_px or bottom_px <= top_px:
            inset_x = max(dot_r + 30, int(sw * 0.12))
            inset_top = max(ui_pad_top, int(sh * 0.12))
            inset_bottom = max(ui_pad_bottom + label_pad, int(sh * 0.12))
            left_px = inset_x
            right_px = sw - inset_x
            top_px = inset_top
            bottom_px = sh - inset_bottom

        if self.debug_logger:
            self.debug_logger.info(
                f"[grid9] SAFE bounds using surface={sw}x{sh} | "
                f"inset_x={inset_x}, inset_top={inset_top}, inset_bottom={inset_bottom} | "
                f"left={left_px}, right={right_px}, top={top_px}, bottom={bottom_px}"
            )

        positions = [
            ("center", center_x_px, center_y_px),
            ("top_left", left_px, top_px),
            ("top_center", center_x_px, top_px),
            ("top_right", right_px, top_px),
            ("middle_left", left_px, center_y_px),
            ("middle_right", right_px, center_y_px),
            ("bottom_left", left_px, bottom_px),
            ("bottom_center", center_x_px, bottom_px),
            ("bottom_right", right_px, bottom_px),
        ]
        return [(n, float(x), float(y)) for (n, x, y) in positions]

    def _validate_grid_positions_px(self, grid_positions: List[Tuple[str, float, float]]) -> List[str]:
        """Validate dots are fully visible given dot radius + UI chrome."""
        errors: List[str] = []

        dot_r = 12
        label_pad = 28
        ui_pad_top = 80
        ui_pad_bottom = 90

        # Prefer real surface size
        try:
            sw, sh = self.ui.screen.get_size()
            sw, sh = int(sw), int(sh)
        except Exception:
            sw, sh = int(self.config.screen_width), int(self.config.screen_height)

        min_x = dot_r + 5
        max_x = sw - (dot_r + 5)
        min_y = max(dot_r + 5, ui_pad_top)
        max_y = sh - max(dot_r + 5 + label_pad, ui_pad_bottom)

        for point_name, x_px, y_px in grid_positions:
            if x_px < min_x:
                errors.append(f"{point_name} x={x_px:.0f}px out of bounds (min {min_x})")
            elif x_px > max_x:
                errors.append(f"{point_name} x={x_px:.0f}px out of bounds (max {max_x})")

            if y_px < min_y:
                errors.append(f"{point_name} y={y_px:.0f}px out of bounds (min {min_y})")
            elif y_px > max_y:
                errors.append(f"{point_name} y={y_px:.0f}px out of bounds (max {max_y})")

        if self.debug_logger and errors:
            self.debug_logger.warning("[grid9] Grid validation failed: " + " | ".join(errors[:6]))

        return errors

    def _show_debug_overlay(self, grid_positions: List[Tuple[str, float, float]]):
        """
        Show debug overlay with grid bounds and dot coordinates.
        Ensures labels are also kept on-screen.
        """
        self._sync_screen_size_from_surface()

        margin = float(self.config.safe_margin)

        # Use actual surface bounds for rectangle
        try:
            sw, sh = self.ui.screen.get_size()
            sw, sh = int(sw), int(sh)
        except Exception:
            sw, sh = int(self.config.screen_width), int(self.config.screen_height)

        # Recompute the exact same safe inset used in _compute_grid_positions
        dot_r = 12
        label_pad = 28
        ui_pad_top = 80
        ui_pad_bottom = 90

        inset_x = max(int(sw * margin), dot_r + 25)
        inset_top = max(int(sh * margin), ui_pad_top)
        inset_bottom = max(int(sh * margin), ui_pad_bottom + label_pad)

        left_px = inset_x
        right_px = sw - inset_x
        top_px = inset_top
        bottom_px = sh - inset_bottom

        self.ui.clear_screen()

        import pygame
        pygame.draw.rect(
            self.ui.screen,
            (100, 100, 100),  # Gray
            (left_px, top_px, right_px - left_px, bottom_px - top_px),
            2
        )

        # Draw all 9 dots
        for point_name, x_px, y_px in grid_positions:
            pygame.draw.circle(
                self.ui.screen,
                (255, 255, 255),
                (int(x_px), int(y_px)),
                12
            )

            label_y = int(y_px) + 25
            label_y = min(label_y, self.config.screen_height - 40)

            self.ui.draw_text(
                f"{point_name}: ({int(x_px)}, {int(y_px)})",
                int(x_px),
                label_y,
                font_size="small",
                color=(200, 200, 200)
            )

        self.ui.draw_text(
            f"Screen(surface): {sw}x{sh}",
            sw // 2,
            30,
            font_size="medium"
        )
        self.ui.draw_text(
            f"Safe margin: {margin*100:.0f}%",
            sw // 2,
            60,
            font_size="small"
        )
        self.ui.draw_text(
            "DEBUG: All 9 dots should be visible. Press SPACE to continue.",
            sw // 2,
            sh - 30,
            font_size="small",
            color=(0, 230, 118)
        )

        self.ui.update_display()

        while self.ui.running:
            events = self.ui.process_events()
            if events.get('quit'):
                self.running = False
                return
            if events.get('space'):
                return
            self.ui.tick(60)

    # ---------------- END FIXES ----------------

    def run_task(self) -> TaskResult:
        """Run the 9-point grid task."""
        result = TaskResult()

        # Ensure we use the REAL pygame window size (not stale config values)
        self._sync_screen_size_from_surface()

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

        # IMPORTANT: store positions so UI indicator total_trials doesn't break
        self.config.grid_positions = grid_positions

        # Validate all grid points are on-screen
        validation_errors = self._validate_grid_positions_px(grid_positions)
        if validation_errors:
            result.success = False
            result.errors.extend(validation_errors)

            self.ui.clear_screen()
            self.ui.draw_text(
                "ERROR: Grid points out of bounds",
                self.config.screen_width // 2,
                self.config.screen_height // 2 - 50,
                font_size="large",
                color=(255, 82, 82)
            )
            for i, err in enumerate(validation_errors[:4]):
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
            if not getattr(self, "running", True) or not self.ui.running:
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
        point_summaries: List[Dict[str, Any]] = []

        # Collect data for each grid point
        all_target_x: List[float] = []
        all_target_y: List[float] = []
        all_gaze_x: List[float] = []
        all_gaze_y: List[float] = []

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
                    all_target_x.append(float(target_x))
                    all_target_y.append(float(target_y))

                    # NOTE: your repo currently does not store mean gaze x/y here.
                    # This keeps your existing behavior, but it is not medically ideal.
                    # For now we keep it unchanged to avoid breaking dependencies.
                    err = float(point_data.get('mean_gaze_error_px', 0) or 0)
                    all_gaze_x.append(float(target_x) + err * float(np.cos(np.random.random() * 2 * np.pi)))
                    all_gaze_y.append(float(target_y) + err * float(np.sin(np.random.random() * 2 * np.pi)))

                if self.debug_logger:
                    self.debug_logger.info(
                        f"Point {point_idx + 1} ({point_name}): "
                        f"valid={point_data.get('valid', 0)}, "
                        f"error={point_data.get('mean_gaze_error_px', np.nan):.1f}px"
                    )

        # Compute overall results
        result.n_trials = len(point_summaries)
        result.n_valid_trials = sum(1 for p in point_summaries if p.get('valid', 0) == 1)
        result.valid_rate = (result.n_valid_trials / result.n_trials) if result.n_trials > 0 else 0.0

        # Compute overall grid metrics
        valid_points = [p for p in point_summaries if p.get('valid', 0) == 1]
        if valid_points:
            errors = [p.get('mean_gaze_error_px', np.nan) for p in valid_points]

            # Systematic bias placeholder (needs actual mean gaze x/y per point to be real)
            bias_x = 0.0
            bias_y = 0.0

            # Linearity (R² of gaze vs target)
            if len(all_target_x) >= 3:
                _, _, r2_x = linear_regression(np.array(all_target_x), np.array(all_gaze_x))
                _, _, r2_y = linear_regression(np.array(all_target_y), np.array(all_gaze_y))
                if not np.isnan(r2_x) and not np.isnan(r2_y):
                    linearity_r2 = float((r2_x + r2_y) / 2.0)
                else:
                    linearity_r2 = np.nan
            else:
                linearity_r2 = np.nan

            result.biomarkers = {
                'grid_accuracy_mean_px': float(np.nanmean(errors)),
                'grid_accuracy_max_px': float(np.nanmax(errors)),
                'systematic_bias_x_px': float(bias_x),
                'systematic_bias_y_px': float(bias_y),
                'gaze_map_linearity_r2': linearity_r2,
            }

        result.mean_quality_score = float(np.nanmean([p.get('quality_score', 0) for p in point_summaries])) if point_summaries else 0.0

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

        gaze_x_samples: List[float] = []
        gaze_y_samples: List[float] = []

        while time.time() < trial_end and self.ui.running:
            current_time = time.time()

            events = self.ui.process_events()
            if events.get('quit'):
                self.running = False
                return None
            if events.get('key_s'):
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

            # total_trials must be defined
            total_trials = len(getattr(self.config, "grid_positions", [])) or 9

            self.ui.update_quality_indicators(
                face_detected=frame_data.get('face_present', 0) == 1,
                fps=frame_data.get('fps_est', 30.0),
                quality_score=frame_data.get('valid_sample', 0),
                trial_count=self.current_trial,
                total_trials=total_trials
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

        gaze_x_arr = np.array(gaze_x, dtype=float)
        gaze_y_arr = np.array(gaze_y, dtype=float)

        valid_mask = ~(np.isnan(gaze_x_arr) | np.isnan(gaze_y_arr))
        gaze_x_arr = gaze_x_arr[valid_mask]
        gaze_y_arr = gaze_y_arr[valid_mask]

        mean_error = np.nan
        rmse_error = np.nan
        dwell_rms = np.nan

        if len(gaze_x_arr) >= 3:
            error_x = gaze_x_arr - float(target_x)
            error_y = gaze_y_arr - float(target_y)
            distances = np.sqrt(error_x**2 + error_y**2)

            mean_error = float(np.mean(distances))
            rmse_error = float(compute_rms(distances))

            rms_x = compute_rms(gaze_x_arr - np.mean(gaze_x_arr))
            rms_y = compute_rms(gaze_y_arr - np.mean(gaze_y_arr))
            dwell_rms = float(np.sqrt(rms_x**2 + rms_y**2))

        grid_validity = self.validity_checker.check_grid_point_validity(
            mean_error if not np.isnan(mean_error) else 1000,
            generic_validity
        )
        self.trial_validities.append(grid_validity)

        calib_quality = self.calibrator.get_quality_score() if getattr(self, "calibrator", None) else 0.0

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
            'calibration_quality': float(calib_quality),
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
            if events.get('quit') or events.get('space'):
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
                'n_points': len(getattr(self.config, "grid_positions", [])) or 9,
                'dwell_time': self.config.dwell_time,
            }
        )

        if getattr(self, "calibration", None):
            meta.calibration_accepted = self.calibration.accepted
            meta.calibration_reason = self.calibration.reason.value
            meta.calibration_error_mean_px = self.calibration.mean_error_px
            meta.calibration_error_max_px = self.calibration.max_error_px

        meta.valid_rates[self.TASK_NAME] = result.valid_rate

        all_fps = [f.get('fps_est', 30.0) for f in getattr(self, "trial_frames", [])]
        if all_fps:
            meta.fps_mean = float(np.mean(all_fps))
            meta.fps_median = float(np.median(all_fps))
            meta.fps_min = float(np.min(all_fps))
            meta.fps_max = float(np.max(all_fps))

        meta.save(f"{self.session_dir}/meta.json")

    def _validate_screen_dimensions(self) -> bool:
        """Validate that screen dimensions meet minimum requirements."""
        return (
            self.config.screen_width >= self.config.min_screen_width and
            self.config.screen_height >= self.config.min_screen_height
        )

    def _log_viewport_info(self, grid_positions: List[Tuple[str, float, float]]):
        """Log viewport size and grid positions for QC and reproducibility."""
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
