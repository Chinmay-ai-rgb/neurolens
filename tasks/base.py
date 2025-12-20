"""Base task class for NeuroLens+ eye tracking tasks."""

import cv2
import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from pathlib import Path

from core.tracker import EyeTracker, FrameData
from core.calibration import Calibrator, CalibrationResult
from core.mapping import GazeMapper, VelocityComputer
from core.validity import ValidityChecker, InvalidReason, TrialValidity
from core.logging import (
    FrameLogger, SummaryLogger, DebugLogger,
    SessionMetadata, create_session_directory, generate_session_id
)
from ui.ui import NeuroLensUI, Colors


@dataclass
class TaskConfig:
    """Base configuration for all tasks."""
    
    # Session info
    session_id: str = ""
    data_dir: str = "data"
    
    # Screen settings
    screen_width: int = 1920
    screen_height: int = 1080
    fullscreen: bool = False
    
    # Target settings
    target_radius: int = 24
    
    # Timing
    instruction_duration: float = 3.0
    
    # Calibration
    calibration_mode: str = "3_point"  # 'center', '3_point', '5_point', '9_point'
    calibration_dwell_time: float = 2.0
    
    # Quality thresholds
    min_fps: float = 15.0
    max_clamp_rate: float = 0.05
    
    def __post_init__(self):
        if not self.session_id:
            self.session_id = generate_session_id()


@dataclass
class TaskResult:
    """Result from running a task."""
    
    success: bool = False
    summary_csv_path: str = ""
    frame_log_csv_path: str = ""
    session_json_path: str = ""
    
    # Summary statistics
    n_trials: int = 0
    n_valid_trials: int = 0
    valid_rate: float = 0.0
    mean_quality_score: float = 0.0
    
    # Aggregated biomarkers
    biomarkers: Dict[str, Any] = field(default_factory=dict)
    
    # Errors/warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class BaseTask(ABC):
    """
    Abstract base class for all eye tracking tasks.
    
    Provides common functionality:
    - Camera initialization
    - Eye tracking setup
    - Calibration
    - Frame logging
    - UI management
    """
    
    TASK_NAME = "base"
    
    def __init__(self, config: Optional[TaskConfig] = None):
        """
        Initialize base task.
        
        Args:
            config: Task configuration
        """
        self.config = config or TaskConfig()
        
        # Components (initialized in setup)
        self.ui: Optional[NeuroLensUI] = None
        self.tracker: Optional[EyeTracker] = None
        self.calibrator: Optional[Calibrator] = None
        self.mapper: Optional[GazeMapper] = None
        self.velocity_computer: Optional[VelocityComputer] = None
        self.validity_checker: Optional[ValidityChecker] = None
        
        # Camera
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Logging
        self.frame_logger: Optional[FrameLogger] = None
        self.summary_logger: Optional[SummaryLogger] = None
        self.debug_logger: Optional[DebugLogger] = None
        self.session_dir: str = ""
        
        # Calibration result
        self.calibration: Optional[CalibrationResult] = None
        
        # State
        self.running = False
        self.current_trial = 0
        self.trial_validities: List[TrialValidity] = []
        
        # Frame data buffer for current trial
        self.trial_frames: List[Dict[str, Any]] = []
    
    def setup(self) -> bool:
        """
        Set up task components.
        
        Returns:
            True if setup successful
        """
        try:
            # Create session directory
            self.session_dir = create_session_directory(
                self.config.data_dir,
                self.config.session_id
            )
            
            # Initialize debug logger
            self.debug_logger = DebugLogger(
                f"{self.session_dir}/debug_{self.TASK_NAME}.log"
            )
            self.debug_logger.info(f"Setting up {self.TASK_NAME} task")
            
            # Initialize camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.debug_logger.error("Failed to open camera")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Initialize UI
            from ui.ui import UIConfig
            ui_config = UIConfig(
                screen_width=self.config.screen_width,
                screen_height=self.config.screen_height,
                fullscreen=self.config.fullscreen,
                target_radius=self.config.target_radius
            )
            self.ui = NeuroLensUI(ui_config)
            
            # Update screen size from UI (may have changed for fullscreen)
            self.config.screen_width, self.config.screen_height = self.ui.get_screen_size()
            
            # Initialize tracker
            self.tracker = EyeTracker(
                screen_width=self.config.screen_width,
                screen_height=self.config.screen_height
            )
            
            # Initialize calibrator
            self.calibrator = Calibrator(
                screen_width=self.config.screen_width,
                screen_height=self.config.screen_height
            )
            
            # Initialize mapper
            self.mapper = GazeMapper(
                screen_width=self.config.screen_width,
                screen_height=self.config.screen_height
            )
            
            # Initialize velocity computer
            self.velocity_computer = VelocityComputer()
            
            # Initialize validity checker
            self.validity_checker = ValidityChecker()
            
            self.debug_logger.info("Setup complete")
            return True
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Setup failed: {e}")
            return False
    
    def run_calibration(self) -> bool:
        """
        Run calibration procedure.
        
        Returns:
            True if calibration successful
        """
        from ui.ui import CalibrationUI
        
        self.debug_logger.info(f"Starting calibration ({self.config.calibration_mode})")
        
        calib_ui = CalibrationUI(self.ui)
        
        # Start calibration
        self.calibrator.start_calibration(self.config.calibration_mode)
        points = self.calibrator.get_point_positions(self.config.calibration_mode)
        
        for point_name, x_norm, y_norm in points:
            x_px = x_norm * self.config.screen_width
            y_px = y_norm * self.config.screen_height
            
            self.calibrator.set_current_point(point_name)
            
            # Show point and wait for fixation
            start_time = time.time()
            samples_collected = 0
            target_samples = int(self.config.calibration_dwell_time * 30)  # Assume 30 fps
            
            while samples_collected < target_samples and self.ui.running:
                # Process events
                events = self.ui.process_events()
                if events['quit']:
                    return False
                if events['key_s']:
                    break  # Skip this point
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Process frame
                frame_data = self.tracker.process_frame(frame)
                
                # Add sample if valid
                if frame_data.face_present and not frame_data.blink_flag:
                    self.calibrator.add_sample(
                        frame_data.gaze_x_norm,
                        frame_data.gaze_y_norm,
                        frame_data.anchor_x_norm,
                        frame_data.anchor_y_norm,
                        valid=True
                    )
                    samples_collected += 1
                
                # Update UI
                progress = samples_collected / target_samples
                calib_ui.show_calibration_point(
                    x_px, y_px, point_name,
                    collecting=True,
                    progress=progress
                )
                
                self.ui.update_quality_indicators(
                    face_detected=frame_data.face_present == 1,
                    fps=frame_data.fps_est,
                    quality_score=progress
                )
                
                self.ui.tick(60)
        
        # Compute calibration
        self.calibration = self.calibrator.compute_calibration()
        
        # Apply calibration to mapper and tracker
        if self.calibration.accepted:
            self.mapper.set_calibration(self.calibration)
            self.tracker.set_baseline_anchor(
                self.calibration.baseline_anchor_x_norm,
                self.calibration.baseline_anchor_y_norm
            )
        
        # Show result
        calib_ui.show_calibration_result(
            self.calibration.accepted,
            self.calibration.reason.value,
            self.calibration.mean_error_px
        )
        
        # Save calibration
        self.calibration.save(f"{self.session_dir}/calibration.json")
        
        self.debug_logger.info(
            f"Calibration {'accepted' if self.calibration.accepted else 'rejected'}: "
            f"{self.calibration.reason.value}, error={self.calibration.mean_error_px:.1f}px"
        )
        
        # Wait for user input
        while self.ui.running:
            events = self.ui.process_events()
            if events['quit']:
                return False
            if events['space']:
                break
            if events['key_r']:
                return self.run_calibration()  # Retry
            self.ui.tick(60)
        
        return self.calibration.accepted
    
    def process_frame_with_mapping(self, frame: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """
        Process a frame and apply gaze mapping.
        
        Args:
            frame: Camera frame
            timestamp: Frame timestamp
        
        Returns:
            Dictionary with all frame data
        """
        # Get raw tracking data
        frame_data = self.tracker.process_frame(frame, timestamp)
        
        # Apply mapping
        mapped = self.mapper.map_gaze(
            frame_data.gaze_x_norm,
            frame_data.gaze_y_norm,
            frame_data.anchor_x_norm,
            frame_data.anchor_y_norm,
            timestamp
        )
        
        # Update frame data with mapped values
        frame_data.gaze_x_norm_comp = mapped.gaze_x_norm_comp
        frame_data.gaze_y_norm_comp = mapped.gaze_y_norm_comp
        frame_data.gaze_x_px_comp = mapped.gaze_x_px_comp
        frame_data.gaze_y_px_comp = mapped.gaze_y_px_comp
        frame_data.clamp_flag_x = mapped.clamp_flag_x
        frame_data.clamp_flag_y = mapped.clamp_flag_y
        frame_data.out_of_range_flag = mapped.out_of_range_flag
        frame_data.vel_x_px_s = mapped.vel_x_px_s
        frame_data.vel_y_px_s = mapped.vel_y_px_s
        
        return frame_data.to_dict()
    
    def log_frame(
        self,
        trial_id: int,
        state: str,
        target_x: float,
        target_y: float,
        frame_data: Dict[str, Any]
    ):
        """
        Log a frame to the frame logger.
        
        Args:
            trial_id: Current trial ID
            state: Current state
            target_x: Target x position
            target_y: Target y position
            frame_data: Frame data dictionary
        """
        if self.frame_logger:
            self.frame_logger.log_frame(trial_id, state, target_x, target_y, frame_data)
        
        # Also store in trial buffer
        frame_data['trial_id'] = trial_id
        frame_data['state'] = state
        frame_data['target_x'] = target_x
        frame_data['target_y'] = target_y
        self.trial_frames.append(frame_data)
    
    def compute_trial_validity(self) -> TrialValidity:
        """
        Compute validity for current trial from buffered frames.
        
        Returns:
            TrialValidity result
        """
        if not self.trial_frames:
            return TrialValidity(valid=False, reason=InvalidReason.INSUFFICIENT_SAMPLES)
        
        # Extract quality metrics
        face_flags = [f.get('face_present', 0) for f in self.trial_frames]
        blink_flags = [f.get('blink_flag', 0) for f in self.trial_frames]
        fps_values = [f.get('fps_est', 30.0) for f in self.trial_frames]
        clamp_x = [f.get('clamp_flag_x', 0) for f in self.trial_frames]
        clamp_y = [f.get('clamp_flag_y', 0) for f in self.trial_frames]
        valid_flags = [f.get('valid_sample', 0) for f in self.trial_frames]
        
        return self.validity_checker.check_trial_validity_generic(
            face_flags, blink_flags, fps_values, clamp_x, clamp_y, valid_flags
        )
    
    def clear_trial_buffer(self):
        """Clear the trial frame buffer."""
        self.trial_frames = []
    
    def show_instructions(self, title: str, lines: List[str]):
        """
        Show task instructions.
        
        Args:
            title: Instruction title
            lines: Instruction lines
        """
        self.ui.draw_instructions(title, lines)
        
        # Wait for space
        while self.ui.running:
            events = self.ui.process_events()
            if events['quit']:
                self.running = False
                return
            if events['space']:
                return
            self.ui.tick(60)
    
    @abstractmethod
    def get_summary_columns(self) -> List[str]:
        """Get column names for summary CSV."""
        pass
    
    @abstractmethod
    def run_task(self) -> TaskResult:
        """
        Run the task.
        
        Returns:
            TaskResult with paths and statistics
        """
        pass
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        
        if self.tracker:
            self.tracker.release()
        
        if self.frame_logger:
            self.frame_logger.close()
        
        if self.summary_logger:
            self.summary_logger.close()
        
        if self.debug_logger:
            self.debug_logger.close()
        
        if self.ui:
            self.ui.close()
    
    def run(self) -> TaskResult:
        """
        Main entry point - setup, calibrate, run task, cleanup.
        
        Returns:
            TaskResult
        """
        result = TaskResult()
        
        try:
            # Setup
            if not self.setup():
                result.errors.append("Setup failed")
                return result
            
            # Show instructions
            self.show_instructions(
                f"{self.TASK_NAME.title()} Task",
                self.get_instructions()
            )
            
            if not self.ui.running:
                return result
            
            # Calibration
            if not self.run_calibration():
                result.warnings.append("Calibration not accepted")
            
            if not self.ui.running:
                return result
            
            # Run task
            result = self.run_task()
            
        except Exception as e:
            result.errors.append(f"Task error: {e}")
            if self.debug_logger:
                self.debug_logger.error(f"Task error: {e}")
        
        finally:
            self.cleanup()
        
        return result
    
    def get_instructions(self) -> List[str]:
        """Get task-specific instructions. Override in subclasses."""
        return [
            "Follow the target dot with your eyes.",
            "Try to keep your head still.",
            "Blink naturally.",
            "",
            "Press SPACE to start.",
            "Press Q or ESC to quit.",
            "Press S to skip a trial."
        ]
