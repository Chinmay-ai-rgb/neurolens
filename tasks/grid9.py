#!/usr/bin/env python3
"""
9-Point Gaze Grid Task for NeuroLens+ eye tracking system.

Final fixes included:
- Uses REAL pygame surface size (not stale config size)
- Redesigns geometry as exact 3×3 lattice inside UI-safe bounds
- Prevents freezing on debug overlay (auto-continue + multiple keys)
- Runs BaseTask calibration properly (calls run_calibration)
- Rewrites validity rules for webcam realism:
    * tiered PASS/WARN/FAIL thresholds based on screen diagonal
    * gates on valid_fraction, clamp_rate, min_samples
- Computes usable_fov_score + pass/warn/fail counts
- Decides whether Grid9 should feed ML (defaults to QC/diagnostic unless strong)
"""

import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from .base import BaseTask, TaskConfig, TaskResult
from core.logging import FrameLogger, SummaryLogger, GRID_SUMMARY_COLUMNS
from core.validity import InvalidReason
from core.utils import compute_rms, linear_regression


@dataclass
class Grid9Config(TaskConfig):
    """Configuration for 9-point grid task."""
    # Dwell time per point
    dwell_time: float = 2.0

    # Base “margin” fraction used in UI-safe bounds
    safe_margin: float = 0.10

    # Minimum screen dimensions required
    min_screen_width: int = 800
    min_screen_height: int = 600

    # Show overlay first
    debug_overlay: bool = True

    # --- Webcam-realistic validity tuning ---
    # Minimum data quality
    min_valid_fraction_grid: float = 0.80
    min_samples_grid: int = 10

    # Allow higher clamp rate for webcam (ValidityThresholds allows 0.25/trial;
    # grid tends to clamp more at edges; we allow up to 0.30 before invalidating)
    max_clamp_rate_grid: float = 0.30

    # PASS/WARN/FAIL error tiers expressed as fraction of screen diagonal.
    # Example 1366×768 diag≈1567px:
    #   pass 0.12 → ~188px
    #   warn 0.18 → ~282px
    #   fail 0.26 → ~407px
    pass_error_frac: float = 0.12
    warn_error_frac: float = 0.18
    fail_error_frac: float = 0.26

    # Dwell stability tiers (RMS of gaze around mean) as fraction of diagonal
    pass_stability_frac: float = 0.06
    warn_stability_frac: float = 0.10

    # Calibration gate for ML eligibility (NOT for running the task)
    min_calibration_quality: float = 0.60

    def __post_init__(self):
        super().__post_init__()


class Grid9Task(BaseTask):
    """9-Point Gaze Grid Task - Measure gaze accuracy across screen."""
    TASK_NAME = "grid9"

    def __init__(self, config: Optional[Grid9Config] = None):
        super().__init__(config or Grid9Config())
        self.config: Grid9Config = self.config

    def get_instructions(self) -> List[str]:
        return [
            "9-POINT GAZE GRID TEST",
            "",
            "A dot will appear at 9 different locations.",
            "Look directly at each dot when it appears.",
            "Keep your eyes fixed on the dot until it moves.",
            "",
            f"Each point will be shown for {self.config.dwell_time:.0f} seconds.",
            "",
            "Press SPACE to start.",
            "Press Q or ESC to quit."
        ]

    def get_summary_columns(self) -> List[str]:
        return GRID_SUMMARY_COLUMNS

    # ---------------- Screen + geometry ----------------

    def _sync_screen_size_from_surface(self) -> Tuple[int, int]:
        """Ensure config screen_width/height match the REAL pygame surface size."""
        try:
            sw, sh = self.ui.screen.get_size()
            sw, sh = int(sw), int(sh)
            self.config.screen_width = sw
            self.config.screen_height = sh
            return sw, sh
        except Exception:
            return int(self.config.screen_width), int(self.config.screen_height)

    def _safe_bounds(self, sw: int, sh: int) -> Tuple[int, int, int, int]:
        """
        Compute UI-safe bounds for dot centers (pixels).
        Accounts for top title text + bottom bar/indicators + labels.
        """
        dot_r = 12
        label_pad = 28
        ui_pad_top = 80
        ui_pad_bottom = 90

        margin_frac = float(self.config.safe_margin)

        inset_x = max(int(sw * margin_frac), dot_r + 25)
        inset_top = max(int(sh * margin_frac), ui_pad_top)
        inset_bottom = max(int(sh * margin_frac), ui_pad_bottom + label_pad)

        left_px = inset_x
        right_px = sw - inset_x
        top_px = inset_top
        bottom_px = sh - inset_bottom

        # Fallback if window is small and safe box collapses
        if right_px <= left_px or bottom_px <= top_px:
            inset_x = max(dot_r + 30, int(sw * 0.12))
            inset_top = max(ui_pad_top, int(sh * 0.12))
            inset_bottom = max(ui_pad_bottom + label_pad, int(sh * 0.12))
            left_px = inset_x
            right_px = sw - inset_x
            top_px = inset_top
            bottom_px = sh - inset_bottom

        return int(left_px), int(right_px), int(top_px), int(bottom_px)

    def _compute_grid_positions(self) -> List[Tuple[str, float, float]]:
        """
        Exact 3×3 lattice inside UI-safe bounds using REAL surface size.
        """
        try:
            sw, sh = self.ui.screen.get_size()
            sw, sh = int(sw), int(sh)
        except Exception:
            sw, sh = int(self.config.screen_width), int(self.config.screen_height)

        left_px, right_px, top_px, bottom_px = self._safe_bounds(sw, sh)

        # Exact midpoints inside safe rectangle
        x_mid = int(round((left_px + right_px) / 2.0))
        y_mid = int(round((top_px + bottom_px) / 2.0))

        xs = [left_px, x_mid, right_px]
        ys = [top_px, y_mid, bottom_px]

        positions = [
            ("top_left", xs[0], ys[0]),
            ("top_center", xs[1], ys[0]),
            ("top_right", xs[2], ys[0]),
            ("middle_left", xs[0], ys[1]),
            ("center", xs[1], ys[1]),
            ("middle_right", xs[2], ys[1]),
            ("bottom_left", xs[0], ys[2]),
            ("bottom_center", xs[1], ys[2]),
            ("bottom_right", xs[2], ys[2]),
        ]

        if self.debug_logger:
            self.debug_logger.info(
                f"[grid9] surface={sw}x{sh} safe=(L={left_px},R={right_px},T={top_px},B={bottom_px}) "
                f"xs={xs} ys={ys}"
            )

        return [(n, float(x), float(y)) for (n, x, y) in positions]

    def _validate_grid_positions_px(self, grid_positions: List[Tuple[str, float, float]]) -> List[str]:
        """Validate that all points are within safe bounds."""
        errors: List[str] = []
        sw, sh = self._sync_screen_size_from_surface()
        left_px, right_px, top_px, bottom_px = self._safe_bounds(sw, sh)

        for name, x, y in grid_positions:
            if x < left_px or x > right_px:
                errors.append(f"{name} x={x:.0f}px out of safe bounds [{left_px}, {right_px}]")
            if y < top_px or y > bottom_px:
                errors.append(f"{name} y={y:.0f}px out of safe bounds [{top_px}, {bottom_px}]")

        if self.debug_logger and errors:
            self.debug_logger.warning("[grid9] Grid validation failed: " + " | ".join(errors[:6]))
        return errors

    def _show_debug_overlay(self, grid_positions: List[Tuple[str, float, float]]) -> None:
        """
        Show all 9 dots and then continue (never blocks forever).
        Auto-continues after 1.5s.
        """
        sw, sh = self._sync_screen_size_from_surface()
        left_px, right_px, top_px, bottom_px = self._safe_bounds(sw, sh)

        self.ui.clear_screen()

        import pygame
        pygame.draw.rect(
            self.ui.screen,
            (100, 100, 100),
            (left_px, top_px, right_px - left_px, bottom_px - top_px),
            2
        )

        dot_r = 12
        for point_name, x_px, y_px in grid_positions:
            pygame.draw.circle(
                self.ui.screen,
                (255, 255, 255),
                (int(x_px), int(y_px)),
                dot_r
            )
            label_y = min(int(y_px) + 25, sh - 40)
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
            "DEBUG: Dots visible. SPACE/ENTER/CLICK to continue (auto-continues).",
            sw // 2,
            sh - 30,
            font_size="small",
            color=(0, 230, 118)
        )

        self.ui.update_display()

        start = time.time()
        while self.ui.running:
            events = self.ui.process_events()
            if events.get("quit") or events.get("key_q") or events.get("escape"):
                self.running = False
                return

            # Accept several possible keys depending on UI mapping
            if (
                events.get("space")
                or events.get("enter")
                or events.get("return")
                or events.get("mouse_down")
                or events.get("mouse_click")
            ):
                return

            if time.time() - start >= 1.5:
                return

            self.ui.tick(60)

    # ---------------- Main task ----------------

    def run_task(self) -> TaskResult:
        """Run the 9-point grid task."""
        result = TaskResult()

        # Always sync to real surface size
        self._sync_screen_size_from_surface()

        if not self._validate_screen_dimensions():
            result.success = False
            result.errors.append(
                f"Screen too small: {self.config.screen_width}x{self.config.screen_height}. "
                f"Minimum required: {self.config.min_screen_width}x{self.config.min_screen_height}"
            )
            return result

        # Compute + store positions
        grid_positions = self._compute_grid_positions()
        self.config.grid_positions = grid_positions  # important for total_trials

        # Validate geometry
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
            time.sleep(2.5)
            return result

        self.running = True

        # Debug overlay
        if self.config.debug_overlay:
            self._show_debug_overlay(grid_positions)
            if not self.ui.running or not self.running:
                return result

        # Force calibration mode for this task and run calibration
        self.config.calibration_mode = "9_point"
        try:
            ok = self.run_calibration()
            if not ok:
                # User quit calibration or UI closed
                result.success = False
                result.errors.append("Calibration aborted.")
                return result
            self.calibration = getattr(self.calibrator, "result", None)
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"[grid9] Calibration failed: {e}")
            # Do not hard-fail; run as diagnostic
            self.calibration = getattr(self.calibrator, "result", None)

        calib_quality = float(self.calibrator.get_quality_score()) if self.calibrator else 0.0

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

        point_summaries: List[Dict[str, Any]] = []

        # For overall analysis (linearity)
        all_target_x: List[float] = []
        all_target_y: List[float] = []
        all_gaze_x: List[float] = []
        all_gaze_y: List[float] = []

        for point_idx, (point_name, target_x, target_y) in enumerate(grid_positions):
            if not self.ui.running or not self.running:
                break

            self.current_trial = point_idx + 1
            self.clear_trial_buffer()

            point_data = self._run_point(point_idx, point_name, target_x, target_y, calib_quality)

            if point_data:
                self.summary_logger.log_trial(point_data)
                point_summaries.append(point_data)

                if point_data.get("valid", 0) == 1:
                    all_target_x.append(float(target_x))
                    all_target_y.append(float(target_y))
                    # Use mean gaze position if available, else fallback to target +/- error (rare)
                    gx = point_data.get("mean_gaze_x_px", np.nan)
                    gy = point_data.get("mean_gaze_y_px", np.nan)
                    if not np.isnan(gx) and not np.isnan(gy):
                        all_gaze_x.append(float(gx))
                        all_gaze_y.append(float(gy))
                    else:
                        err = float(point_data.get("mean_gaze_error_px", 0) or 0)
                        all_gaze_x.append(float(target_x) + err)
                        all_gaze_y.append(float(target_y) + err)

        # Aggregate
        result.n_trials = len(point_summaries)
        result.n_valid_trials = sum(1 for p in point_summaries if p.get("valid", 0) == 1)
        result.valid_rate = (result.n_valid_trials / result.n_trials) if result.n_trials > 0 else 0.0
        result.mean_quality_score = float(np.nanmean([p.get("quality_score", 0) for p in point_summaries])) if point_summaries else 0.0

        # PASS/WARN/FAIL counts
        pass_count = sum(1 for p in point_summaries if p.get("grid_tier") == "PASS")
        warn_count = sum(1 for p in point_summaries if p.get("grid_tier") == "WARN")
        fail_count = sum(1 for p in point_summaries if p.get("grid_tier") == "FAIL")

        valid_points = [p for p in point_summaries if p.get("valid", 0) == 1]
        if valid_points:
            errors = [p.get("mean_gaze_error_px", np.nan) for p in valid_points]
            mean_err = float(np.nanmean(errors))
            max_err = float(np.nanmax(errors))

            # Linearity based on mean gaze x/y per point
            if len(all_target_x) >= 3 and len(all_gaze_x) == len(all_target_x):
                _, _, r2_x = linear_regression(np.array(all_target_x), np.array(all_gaze_x))
                _, _, r2_y = linear_regression(np.array(all_target_y), np.array(all_gaze_y))
                linearity_r2 = float((r2_x + r2_y) / 2.0) if (not np.isnan(r2_x) and not np.isnan(r2_y)) else np.nan
            else:
                linearity_r2 = np.nan

            # Usable FOV score: (pass + 0.5*warn) / 9
            usable_fov = float((pass_count + 0.5 * warn_count) / 9.0) if result.n_trials > 0 else 0.0

            # ML eligibility decision:
            # - Grid9 is QC-heavy; only allow ML if calibration + coverage are strong.
            # - Otherwise mark diagnostic_only.
            ml_eligible = int(
                (calib_quality >= float(self.config.min_calibration_quality))
                and (pass_count >= 7)
                and (fail_count == 0)
                and (result.valid_rate >= 0.85)
            )
            diagnostic_only = 1 if ml_eligible == 0 else 0

            result.biomarkers = {
                "grid_accuracy_mean_px": mean_err,
                "grid_accuracy_max_px": max_err,
                "gaze_map_linearity_r2": linearity_r2,
                "grid_pass_count": int(pass_count),
                "grid_warn_count": int(warn_count),
                "grid_fail_count": int(fail_count),
                "usable_fov_score": usable_fov,
                "grid_ml_eligible": int(ml_eligible),
                "grid_diagnostic_only": int(diagnostic_only),
                "calibration_quality_grid9": float(calib_quality),
            }

        self._show_results(result)
        self._save_metadata(result)

        result.success = True
        result.session_json_path = f"{self.session_dir}/meta.json"
        return result

    # ---------------- Per-point ----------------

    def _run_point(
        self,
        point_idx: int,
        point_name: str,
        target_x: float,
        target_y: float,
        calib_quality: float
    ) -> Optional[Dict[str, Any]]:
        """Collect dwell samples for one point."""
        trial_start = time.time()
        trial_end = trial_start + float(self.config.dwell_time)

        gaze_x_samples: List[float] = []
        gaze_y_samples: List[float] = []

        while time.time() < trial_end and self.ui.running:
            now = time.time()

            events = self.ui.process_events()
            if events.get("quit") or events.get("key_q") or events.get("escape"):
                self.running = False
                return None
            if events.get("key_s"):
                return self._create_skipped_point(point_idx, point_name, target_x, target_y, trial_start, calib_quality)

            ret, frame = self.cap.read()
            if not ret:
                continue

            frame_data = self.process_frame_with_mapping(frame, now)
            self.log_frame(point_idx, "DWELL", target_x, target_y, frame_data)

            if frame_data.get("valid_sample", 0) == 1:
                gaze_x_samples.append(frame_data.get("gaze_x_px_comp", np.nan))
                gaze_y_samples.append(frame_data.get("gaze_y_px_comp", np.nan))

            # UI
            self.ui.clear_screen()
            self.ui.draw_target(target_x, target_y)

            progress = (now - trial_start) / float(self.config.dwell_time)
            self.ui.draw_progress_bar(
                progress,
                int(self.config.screen_width) // 2,
                int(self.config.screen_height) - 50
            )

            total_trials = len(getattr(self.config, "grid_positions", [])) or 9
            self.ui.update_quality_indicators(
                face_detected=frame_data.get("face_present", 0) == 1,
                fps=frame_data.get("fps_est", 30.0),
                quality_score=frame_data.get("valid_sample", 0),
                trial_count=self.current_trial,
                total_trials=total_trials
            )
            self.ui.draw_quality_indicators()

            self.ui.draw_text(
                point_name.replace("_", " ").title(),
                int(self.config.screen_width) // 2,
                50,
                font_size="medium"
            )

            self.ui.update_display()
            self.ui.tick(60)

        return self._compute_biomarkers(
            point_idx, point_name, target_x, target_y,
            trial_start, time.time(),
            gaze_x_samples, gaze_y_samples,
            calib_quality
        )

    def _compute_biomarkers(
        self,
        point_idx: int,
        point_name: str,
        target_x: float,
        target_y: float,
        trial_start: float,
        trial_end: float,
        gaze_x: List[float],
        gaze_y: List[float],
        calib_quality: float
    ) -> Dict[str, Any]:
        """Compute per-point grid biomarkers with webcam-realistic validity."""
        generic_validity = self.compute_trial_validity()

        gx = np.array(gaze_x, dtype=float)
        gy = np.array(gaze_y, dtype=float)

        # Remove NaN
        mask = ~(np.isnan(gx) | np.isnan(gy))
        gx = gx[mask]
        gy = gy[mask]

        n_samples = int(len(gx))

        mean_error = np.nan
        rmse_error = np.nan
        dwell_rms = np.nan
        mean_gx = np.nan
        mean_gy = np.nan

        if n_samples >= 3:
            mean_gx = float(np.mean(gx))
            mean_gy = float(np.mean(gy))

            dx = gx - float(target_x)
            dy = gy - float(target_y)
            dist = np.sqrt(dx**2 + dy**2)

            mean_error = float(np.mean(dist))
            rmse_error = float(compute_rms(dist))

            rms_x = compute_rms(gx - np.mean(gx))
            rms_y = compute_rms(gy - np.mean(gy))
            dwell_rms = float(np.sqrt(rms_x**2 + rms_y**2))

        # --- Webcam-realistic tiering ---
        sw = float(self.config.screen_width)
        sh = float(self.config.screen_height)
        diag = float(np.hypot(sw, sh))

        pass_err = float(self.config.pass_error_frac) * diag
        warn_err = float(self.config.warn_error_frac) * diag
        fail_err = float(self.config.fail_error_frac) * diag

        pass_stab = float(self.config.pass_stability_frac) * diag
        warn_stab = float(self.config.warn_stability_frac) * diag

        vf = float(generic_validity.valid_fraction)
        cr = float(generic_validity.clamp_rate)

        valid = True
        reason = InvalidReason.VALID.value
        tier = "PASS"
        quality = 1.0

        # Hard gates
        if n_samples < int(self.config.min_samples_grid):
            valid = False
            reason = InvalidReason.INSUFFICIENT_SAMPLES.value
            tier = "FAIL"
            quality = 0.0
        elif vf < float(self.config.min_valid_fraction_grid):
            valid = False
            reason = InvalidReason.INSUFFICIENT_SAMPLES.value
            tier = "FAIL"
            quality = 0.2
        elif cr > float(self.config.max_clamp_rate_grid):
            valid = False
            reason = InvalidReason.CLAMP_RATE_HIGH.value
            tier = "FAIL"
            quality = 0.4
        else:
            # If error can't be computed, fail safely
            if np.isnan(mean_error) or np.isnan(dwell_rms):
                valid = False
                reason = InvalidReason.INSUFFICIENT_SAMPLES.value
                tier = "FAIL"
                quality = 0.0
            else:
                # Tiered grading
                if (mean_error <= pass_err) and (dwell_rms <= pass_stab):
                    tier = "PASS"
                    quality = 1.0
                elif (mean_error <= warn_err) and (dwell_rms <= warn_stab):
                    tier = "WARN"
                    quality = 0.75
                elif mean_error <= fail_err:
                    tier = "WARN"
                    quality = 0.55
                else:
                    # Too far from target
                    valid = False
                    reason = InvalidReason.TARGET_NOT_REACHED.value
                    tier = "FAIL"
                    quality = 0.35

        return {
            "trial_id": point_idx,
            "point_name": point_name,
            "trial_start_s": trial_start,
            "trial_end_s": trial_end,
            "target_x": float(target_x),
            "target_y": float(target_y),

            "mean_gaze_error_px": mean_error,
            "rmse_gaze_error_px": rmse_error,
            "dwell_stability_rms_px": dwell_rms,

            # helpful for overall linearity
            "mean_gaze_x_px": mean_gx,
            "mean_gaze_y_px": mean_gy,

            "valid_fraction": vf,
            "valid": 1 if valid else 0,
            "invalid_reason": reason,
            "quality_score": float(quality),

            # grid-specific extra columns (safe even if ignored by logger)
            "grid_tier": tier,
            "grid_pass_err_px": pass_err,
            "grid_warn_err_px": warn_err,
            "grid_fail_err_px": fail_err,
            "grid_pass_stab_px": pass_stab,
            "grid_warn_stab_px": warn_stab,

            "calibration_quality": float(calib_quality),
            "fps_median": float(generic_validity.fps_median),
            "clamp_rate": float(cr),
        }

    def _create_skipped_point(
        self,
        point_idx: int,
        point_name: str,
        target_x: float,
        target_y: float,
        trial_start: float,
        calib_quality: float
    ) -> Dict[str, Any]:
        """Create skipped point record."""
        return {
            "trial_id": point_idx,
            "point_name": point_name,
            "trial_start_s": trial_start,
            "trial_end_s": time.time(),
            "target_x": float(target_x),
            "target_y": float(target_y),

            "mean_gaze_error_px": np.nan,
            "rmse_gaze_error_px": np.nan,
            "dwell_stability_rms_px": np.nan,
            "mean_gaze_x_px": np.nan,
            "mean_gaze_y_px": np.nan,

            "valid_fraction": 0.0,
            "valid": 0,
            "invalid_reason": InvalidReason.SKIPPED.value,
            "quality_score": 0.0,

            "grid_tier": "FAIL",
            "calibration_quality": float(calib_quality),
            "fps_median": 0.0,
            "clamp_rate": 0.0,
        }

    # ---------------- UI + metadata ----------------

    def _show_results(self, result: TaskResult):
        """Show results screen."""
        metrics = {
            "Valid Points": f"{result.n_valid_trials}/{result.n_trials}",
            "Mean Accuracy (px)": result.biomarkers.get("grid_accuracy_mean_px", np.nan),
            "Max Error (px)": result.biomarkers.get("grid_accuracy_max_px", np.nan),
            "Linearity R²": result.biomarkers.get("gaze_map_linearity_r2", np.nan),
            "Calibration Q": result.biomarkers.get("calibration_quality_grid9", np.nan),
            "Pass/Warn/Fail": f"{result.biomarkers.get('grid_pass_count', 0)}/"
                             f"{result.biomarkers.get('grid_warn_count', 0)}/"
                             f"{result.biomarkers.get('grid_fail_count', 0)}",
            "ML Eligible": result.biomarkers.get("grid_ml_eligible", 0),
        }

        self.ui.draw_results("9-Point Grid Results", metrics, result.valid_rate)

        while self.ui.running:
            events = self.ui.process_events()
            if events.get("quit") or events.get("space") or events.get("enter") or events.get("return"):
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
                "task": self.TASK_NAME,
                "n_points": len(getattr(self.config, "grid_positions", [])) or 9,
                "dwell_time": float(self.config.dwell_time),
                "min_calibration_quality": float(self.config.min_calibration_quality),
            }
        )

        if getattr(self, "calibration", None):
            meta.calibration_accepted = bool(self.calibration.accepted)
            meta.calibration_reason = self.calibration.reason.value
            meta.calibration_error_mean_px = float(self.calibration.mean_error_px)
            meta.calibration_error_max_px = float(self.calibration.max_error_px)

        meta.valid_rates[self.TASK_NAME] = float(result.valid_rate)

        all_fps = [f.get("fps_est", 30.0) for f in getattr(self, "trial_frames", [])]
        if all_fps:
            meta.fps_mean = float(np.mean(all_fps))
            meta.fps_median = float(np.median(all_fps))
            meta.fps_min = float(np.min(all_fps))
            meta.fps_max = float(np.max(all_fps))

        meta.save(f"{self.session_dir}/meta.json")

    def _validate_screen_dimensions(self) -> bool:
        """Validate that screen dimensions meet minimum requirements."""
        return (
            int(self.config.screen_width) >= int(self.config.min_screen_width) and
            int(self.config.screen_height) >= int(self.config.min_screen_height)
        )
