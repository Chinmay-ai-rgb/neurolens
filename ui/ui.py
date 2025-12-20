"""Pygame-based UI for NeuroLens+ eye tracking system."""

import pygame
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import time


class Colors:
    """UI color scheme as specified."""
    BACKGROUND = (11, 15, 26)  # #0B0F1A - very dark navy
    TARGET_DOT = (255, 255, 255)  # #FFFFFF - white
    INSTRUCTION_TEXT = (215, 227, 255)  # #D7E3FF - light blue
    SUCCESS = (0, 230, 118)  # #00E676 - green
    WARNING = (255, 196, 0)  # #FFC400 - amber
    ERROR = (255, 82, 82)  # #FF5252 - red
    CALIBRATION_DOT = (79, 195, 247)  # #4FC3F7 - light blue
    
    # Additional UI colors
    QUALITY_BAR_BG = (30, 40, 60)
    QUALITY_BAR_LOW = (255, 82, 82)
    QUALITY_BAR_MED = (255, 196, 0)
    QUALITY_BAR_HIGH = (0, 230, 118)
    
    FACE_INDICATOR_ON = (0, 230, 118)
    FACE_INDICATOR_OFF = (255, 82, 82)
    
    FPS_TEXT = (150, 160, 180)
    TRIAL_TEXT = (180, 190, 210)


class UIState(Enum):
    """UI state machine states."""
    MENU = "menu"
    INSTRUCTIONS = "instructions"
    CALIBRATION = "calibration"
    TASK_RUNNING = "task_running"
    TASK_PAUSED = "task_paused"
    RESULTS = "results"
    QUIT = "quit"


@dataclass
class UIConfig:
    """UI configuration."""
    screen_width: int = 1920
    screen_height: int = 1080
    fullscreen: bool = False
    target_radius: int = 24
    font_size_large: int = 48
    font_size_medium: int = 32
    font_size_small: int = 24
    quality_bar_width: int = 200
    quality_bar_height: int = 20


class NeuroLensUI:
    """
    Pygame-based UI for NeuroLens+ eye tracking tasks.
    
    Provides:
    - Fullscreen/windowed mode toggle
    - Target dot rendering
    - Instructions display
    - Quality indicators (face detection, FPS, quality bar)
    - Keyboard controls
    """
    
    def __init__(self, config: Optional[UIConfig] = None):
        """
        Initialize UI.
        
        Args:
            config: Optional UI configuration
        """
        self.config = config or UIConfig()
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Set up display
        self._setup_display()
        
        # Load fonts
        self.font_large = pygame.font.Font(None, self.config.font_size_large)
        self.font_medium = pygame.font.Font(None, self.config.font_size_medium)
        self.font_small = pygame.font.Font(None, self.config.font_size_small)
        
        # State
        self.state = UIState.MENU
        self.running = True
        self.fullscreen = self.config.fullscreen
        
        # Quality indicators
        self.face_detected = False
        self.current_fps = 0.0
        self.quality_score = 0.0
        self.trial_count = 0
        self.total_trials = 0
        
        # Target position
        self.target_x = self.config.screen_width // 2
        self.target_y = self.config.screen_height // 2
        self.target_visible = False
        self.target_color = Colors.TARGET_DOT
        
        # Clock for frame timing
        self.clock = pygame.time.Clock()
    
    def _setup_display(self):
        """Set up pygame display."""
        flags = pygame.DOUBLEBUF
        if self.config.fullscreen:
            flags |= pygame.FULLSCREEN
            info = pygame.display.Info()
            self.config.screen_width = info.current_w
            self.config.screen_height = info.current_h
        
        self.screen = pygame.display.set_mode(
            (self.config.screen_width, self.config.screen_height),
            flags
        )
        pygame.display.set_caption("NeuroLens+")
        
        # Hide mouse cursor during tasks
        pygame.mouse.set_visible(False)
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        self.fullscreen = not self.fullscreen
        self.config.fullscreen = self.fullscreen
        self._setup_display()
    
    def get_screen_size(self) -> Tuple[int, int]:
        """Get current screen size."""
        return (self.config.screen_width, self.config.screen_height)
    
    def process_events(self) -> Dict[str, Any]:
        """
        Process pygame events.
        
        Returns:
            Dictionary with event information:
            - 'quit': True if quit requested
            - 'space': True if space pressed
            - 'escape': True if escape pressed
            - 'key_r': True if R pressed (recalibrate)
            - 'key_s': True if S pressed (skip)
            - 'key_f': True if F pressed (fullscreen toggle)
        """
        events = {
            'quit': False,
            'space': False,
            'escape': False,
            'key_r': False,
            'key_s': False,
            'key_f': False,
            'key_q': False
        }
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events['quit'] = True
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    events['space'] = True
                elif event.key == pygame.K_ESCAPE:
                    events['escape'] = True
                    events['quit'] = True
                elif event.key == pygame.K_q:
                    events['key_q'] = True
                    events['quit'] = True
                elif event.key == pygame.K_r:
                    events['key_r'] = True
                elif event.key == pygame.K_s:
                    events['key_s'] = True
                elif event.key == pygame.K_f:
                    events['key_f'] = True
                    self.toggle_fullscreen()
        
        return events
    
    def clear_screen(self):
        """Clear screen with background color."""
        self.screen.fill(Colors.BACKGROUND)
    
    def draw_target(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        radius: Optional[int] = None,
        color: Optional[Tuple[int, int, int]] = None
    ):
        """
        Draw target dot.
        
        Args:
            x: X position (uses stored position if None)
            y: Y position (uses stored position if None)
            radius: Dot radius (uses config if None)
            color: Dot color (uses stored color if None)
        """
        if x is not None:
            self.target_x = int(x)
        if y is not None:
            self.target_y = int(y)
        if radius is None:
            radius = self.config.target_radius
        if color is None:
            color = self.target_color
        
        pygame.draw.circle(
            self.screen,
            color,
            (self.target_x, self.target_y),
            radius
        )
        self.target_visible = True
    
    def draw_calibration_target(
        self,
        x: float,
        y: float,
        radius: Optional[int] = None,
        collecting: bool = False
    ):
        """
        Draw calibration target with optional collection indicator.
        
        Args:
            x: X position
            y: Y position
            radius: Dot radius
            collecting: Whether currently collecting samples
        """
        if radius is None:
            radius = self.config.target_radius
        
        color = Colors.CALIBRATION_DOT if collecting else Colors.TARGET_DOT
        
        # Draw outer ring when collecting
        if collecting:
            pygame.draw.circle(
                self.screen,
                Colors.SUCCESS,
                (int(x), int(y)),
                radius + 8,
                3
            )
        
        pygame.draw.circle(
            self.screen,
            color,
            (int(x), int(y)),
            radius
        )
    
    def draw_text(
        self,
        text: str,
        x: float,
        y: float,
        color: Optional[Tuple[int, int, int]] = None,
        font_size: str = "medium",
        center: bool = True
    ):
        """
        Draw text on screen.
        
        Args:
            text: Text to draw
            x: X position
            y: Y position
            color: Text color
            font_size: "large", "medium", or "small"
            center: Whether to center text at position
        """
        if color is None:
            color = Colors.INSTRUCTION_TEXT
        
        if font_size == "large":
            font = self.font_large
        elif font_size == "small":
            font = self.font_small
        else:
            font = self.font_medium
        
        surface = font.render(text, True, color)
        rect = surface.get_rect()
        
        if center:
            rect.center = (int(x), int(y))
        else:
            rect.topleft = (int(x), int(y))
        
        self.screen.blit(surface, rect)
    
    def draw_instructions(
        self,
        title: str,
        lines: List[str],
        footer: str = "Press SPACE to continue"
    ):
        """
        Draw instruction screen.
        
        Args:
            title: Title text
            lines: List of instruction lines
            footer: Footer text
        """
        self.clear_screen()
        
        # Title
        self.draw_text(
            title,
            self.config.screen_width // 2,
            100,
            font_size="large"
        )
        
        # Instructions
        y_start = 200
        line_height = 40
        for i, line in enumerate(lines):
            self.draw_text(
                line,
                self.config.screen_width // 2,
                y_start + i * line_height,
                font_size="medium"
            )
        
        # Footer
        self.draw_text(
            footer,
            self.config.screen_width // 2,
            self.config.screen_height - 100,
            color=Colors.SUCCESS,
            font_size="medium"
        )
        
        self.update_display()
    
    def draw_quality_indicators(self):
        """Draw quality indicators (face detection, FPS, quality bar)."""
        margin = 20
        
        # Face detection indicator
        face_color = Colors.FACE_INDICATOR_ON if self.face_detected else Colors.FACE_INDICATOR_OFF
        pygame.draw.circle(
            self.screen,
            face_color,
            (margin + 10, margin + 10),
            10
        )
        self.draw_text(
            "Face" if self.face_detected else "No Face",
            margin + 30,
            margin + 10,
            color=face_color,
            font_size="small",
            center=False
        )
        
        # FPS indicator
        fps_text = f"FPS: {self.current_fps:.1f}"
        fps_color = Colors.SUCCESS if self.current_fps >= 25 else (
            Colors.WARNING if self.current_fps >= 15 else Colors.ERROR
        )
        self.draw_text(
            fps_text,
            margin,
            margin + 35,
            color=fps_color,
            font_size="small",
            center=False
        )
        
        # Trial counter
        if self.total_trials > 0:
            trial_text = f"Trial: {self.trial_count}/{self.total_trials}"
            self.draw_text(
                trial_text,
                margin,
                margin + 60,
                color=Colors.TRIAL_TEXT,
                font_size="small",
                center=False
            )
        
        # Quality bar
        bar_x = self.config.screen_width - self.config.quality_bar_width - margin
        bar_y = margin
        
        # Background
        pygame.draw.rect(
            self.screen,
            Colors.QUALITY_BAR_BG,
            (bar_x, bar_y, self.config.quality_bar_width, self.config.quality_bar_height)
        )
        
        # Fill based on quality
        fill_width = int(self.quality_score * self.config.quality_bar_width)
        if self.quality_score >= 0.7:
            fill_color = Colors.QUALITY_BAR_HIGH
        elif self.quality_score >= 0.4:
            fill_color = Colors.QUALITY_BAR_MED
        else:
            fill_color = Colors.QUALITY_BAR_LOW
        
        pygame.draw.rect(
            self.screen,
            fill_color,
            (bar_x, bar_y, fill_width, self.config.quality_bar_height)
        )
        
        # Border
        pygame.draw.rect(
            self.screen,
            Colors.INSTRUCTION_TEXT,
            (bar_x, bar_y, self.config.quality_bar_width, self.config.quality_bar_height),
            1
        )
        
        # Quality label
        quality_text = f"Quality: {int(self.quality_score * 100)}%"
        self.draw_text(
            quality_text,
            bar_x + self.config.quality_bar_width // 2,
            bar_y + self.config.quality_bar_height + 15,
            color=Colors.INSTRUCTION_TEXT,
            font_size="small"
        )
    
    def update_quality_indicators(
        self,
        face_detected: bool,
        fps: float,
        quality_score: float,
        trial_count: int = 0,
        total_trials: int = 0
    ):
        """
        Update quality indicator values.
        
        Args:
            face_detected: Whether face is detected
            fps: Current FPS
            quality_score: Quality score (0-1)
            trial_count: Current trial number
            total_trials: Total number of trials
        """
        self.face_detected = face_detected
        self.current_fps = fps
        self.quality_score = quality_score
        self.trial_count = trial_count
        self.total_trials = total_trials
    
    def draw_countdown(self, seconds: int):
        """
        Draw countdown number.
        
        Args:
            seconds: Seconds remaining
        """
        self.draw_text(
            str(seconds),
            self.config.screen_width // 2,
            self.config.screen_height // 2,
            color=Colors.WARNING,
            font_size="large"
        )
    
    def draw_message(
        self,
        message: str,
        message_type: str = "info"
    ):
        """
        Draw a message overlay.
        
        Args:
            message: Message text
            message_type: "info", "success", "warning", or "error"
        """
        if message_type == "success":
            color = Colors.SUCCESS
        elif message_type == "warning":
            color = Colors.WARNING
        elif message_type == "error":
            color = Colors.ERROR
        else:
            color = Colors.INSTRUCTION_TEXT
        
        # Semi-transparent background
        overlay = pygame.Surface((self.config.screen_width, 100))
        overlay.set_alpha(200)
        overlay.fill(Colors.BACKGROUND)
        self.screen.blit(overlay, (0, self.config.screen_height // 2 - 50))
        
        self.draw_text(
            message,
            self.config.screen_width // 2,
            self.config.screen_height // 2,
            color=color,
            font_size="medium"
        )
    
    def draw_flash(self, intensity: float = 1.0):
        """
        Draw a white flash (for PLR task).
        
        Args:
            intensity: Flash intensity (0-1)
        """
        brightness = int(255 * intensity)
        self.screen.fill((brightness, brightness, brightness))
    
    def draw_progress_bar(
        self,
        progress: float,
        x: float,
        y: float,
        width: int = 300,
        height: int = 20
    ):
        """
        Draw a progress bar.
        
        Args:
            progress: Progress value (0-1)
            x: X position (center)
            y: Y position (center)
            width: Bar width
            height: Bar height
        """
        bar_x = int(x - width // 2)
        bar_y = int(y - height // 2)
        
        # Background
        pygame.draw.rect(
            self.screen,
            Colors.QUALITY_BAR_BG,
            (bar_x, bar_y, width, height)
        )
        
        # Fill
        fill_width = int(progress * width)
        pygame.draw.rect(
            self.screen,
            Colors.SUCCESS,
            (bar_x, bar_y, fill_width, height)
        )
        
        # Border
        pygame.draw.rect(
            self.screen,
            Colors.INSTRUCTION_TEXT,
            (bar_x, bar_y, width, height),
            1
        )
    
    def draw_menu(self, options: List[str], selected: int = 0):
        """
        Draw menu screen.
        
        Args:
            options: List of menu options
            selected: Currently selected option index
        """
        self.clear_screen()
        
        # Title
        self.draw_text(
            "NeuroLens+",
            self.config.screen_width // 2,
            100,
            font_size="large"
        )
        
        self.draw_text(
            "Eye Biomarker Platform",
            self.config.screen_width // 2,
            150,
            color=Colors.FPS_TEXT,
            font_size="medium"
        )
        
        # Options
        y_start = 250
        line_height = 50
        
        for i, option in enumerate(options):
            color = Colors.SUCCESS if i == selected else Colors.INSTRUCTION_TEXT
            prefix = "> " if i == selected else "  "
            self.draw_text(
                prefix + option,
                self.config.screen_width // 2,
                y_start + i * line_height,
                color=color,
                font_size="medium"
            )
        
        # Footer
        self.draw_text(
            "Use UP/DOWN to select, SPACE to confirm, Q to quit",
            self.config.screen_width // 2,
            self.config.screen_height - 50,
            color=Colors.FPS_TEXT,
            font_size="small"
        )
        
        self.update_display()
    
    def draw_results(
        self,
        title: str,
        metrics: Dict[str, Any],
        valid_rate: float
    ):
        """
        Draw results screen.
        
        Args:
            title: Results title
            metrics: Dictionary of metric names and values
            valid_rate: Valid trial rate (0-1)
        """
        self.clear_screen()
        
        # Title
        self.draw_text(
            title,
            self.config.screen_width // 2,
            80,
            font_size="large"
        )
        
        # Valid rate with color coding
        rate_color = Colors.SUCCESS if valid_rate >= 0.85 else (
            Colors.WARNING if valid_rate >= 0.70 else Colors.ERROR
        )
        self.draw_text(
            f"Valid Trial Rate: {valid_rate*100:.1f}%",
            self.config.screen_width // 2,
            140,
            color=rate_color,
            font_size="medium"
        )
        
        # Metrics
        y_start = 200
        line_height = 35
        
        for i, (name, value) in enumerate(metrics.items()):
            if isinstance(value, float):
                if np.isnan(value):
                    value_str = "N/A"
                else:
                    value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            
            self.draw_text(
                f"{name}: {value_str}",
                self.config.screen_width // 2,
                y_start + i * line_height,
                font_size="small"
            )
        
        # Footer
        self.draw_text(
            "Press SPACE to continue",
            self.config.screen_width // 2,
            self.config.screen_height - 50,
            color=Colors.SUCCESS,
            font_size="medium"
        )
        
        self.update_display()
    
    def update_display(self):
        """Update display (flip buffers)."""
        pygame.display.flip()
    
    def tick(self, fps: int = 60) -> float:
        """
        Tick clock and return delta time.
        
        Args:
            fps: Target FPS
        
        Returns:
            Delta time in seconds
        """
        return self.clock.tick(fps) / 1000.0
    
    def wait_for_key(self, key: int = pygame.K_SPACE, timeout: float = None) -> bool:
        """
        Wait for a specific key press.
        
        Args:
            key: Key to wait for
            timeout: Optional timeout in seconds
        
        Returns:
            True if key pressed, False if timeout or quit
        """
        start_time = time.time()
        
        while self.running:
            events = self.process_events()
            
            if events['quit']:
                return False
            
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == key:
                    return True
            
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            self.tick(60)
        
        return False
    
    def close(self):
        """Close UI and cleanup."""
        pygame.quit()
        self.running = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class CalibrationUI:
    """
    Calibration-specific UI helpers.
    """
    
    def __init__(self, ui: NeuroLensUI):
        """
        Initialize calibration UI.
        
        Args:
            ui: Main UI instance
        """
        self.ui = ui
    
    def show_calibration_point(
        self,
        x: float,
        y: float,
        point_name: str,
        collecting: bool = False,
        progress: float = 0.0
    ):
        """
        Show calibration point with status.
        
        Args:
            x: X position
            y: Y position
            point_name: Name of calibration point
            collecting: Whether currently collecting
            progress: Collection progress (0-1)
        """
        self.ui.clear_screen()
        
        # Draw target
        self.ui.draw_calibration_target(x, y, collecting=collecting)
        
        # Draw point name
        self.ui.draw_text(
            f"Look at the dot ({point_name})",
            self.ui.config.screen_width // 2,
            50,
            font_size="medium"
        )
        
        # Draw progress if collecting
        if collecting:
            self.ui.draw_progress_bar(
                progress,
                self.ui.config.screen_width // 2,
                self.ui.config.screen_height - 50
            )
        
        # Draw quality indicators
        self.ui.draw_quality_indicators()
        
        self.ui.update_display()
    
    def show_calibration_result(
        self,
        accepted: bool,
        reason: str,
        error_px: float
    ):
        """
        Show calibration result.
        
        Args:
            accepted: Whether calibration was accepted
            reason: Reason for acceptance/rejection
            error_px: Mean calibration error in pixels
        """
        self.ui.clear_screen()
        
        if accepted:
            self.ui.draw_text(
                "Calibration Successful",
                self.ui.config.screen_width // 2,
                self.ui.config.screen_height // 2 - 50,
                color=Colors.SUCCESS,
                font_size="large"
            )
            self.ui.draw_text(
                f"Mean Error: {error_px:.1f} px",
                self.ui.config.screen_width // 2,
                self.ui.config.screen_height // 2 + 20,
                font_size="medium"
            )
        else:
            self.ui.draw_text(
                "Calibration Failed",
                self.ui.config.screen_width // 2,
                self.ui.config.screen_height // 2 - 50,
                color=Colors.ERROR,
                font_size="large"
            )
            self.ui.draw_text(
                f"Reason: {reason}",
                self.ui.config.screen_width // 2,
                self.ui.config.screen_height // 2 + 20,
                font_size="medium"
            )
            self.ui.draw_text(
                "Press R to retry calibration",
                self.ui.config.screen_width // 2,
                self.ui.config.screen_height // 2 + 70,
                color=Colors.WARNING,
                font_size="medium"
            )
        
        self.ui.draw_text(
            "Press SPACE to continue",
            self.ui.config.screen_width // 2,
            self.ui.config.screen_height - 50,
            color=Colors.SUCCESS,
            font_size="medium"
        )
        
        self.ui.update_display()
