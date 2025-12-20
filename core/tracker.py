"""Eye tracking module using MediaPipe FaceMesh."""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
import time

from .utils import compute_ear, detect_blink, RollingFPS


# MediaPipe FaceMesh landmark indices
# Left eye landmarks for EAR calculation
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
# Right eye landmarks for EAR calculation  
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Iris landmarks (MediaPipe iris refinement)
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Iris boundary landmarks for diameter estimation
LEFT_IRIS_BOUNDARY = [469, 470, 471, 472]
RIGHT_IRIS_BOUNDARY = [474, 475, 476, 477]

# Anchor points for head compensation
NOSE_TIP = 1
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


@dataclass
class FrameData:
    """Container for per-frame tracking data."""
    
    # Timing
    t_epoch_s: float = 0.0
    frame_idx: int = 0
    fps_est: float = 30.0
    dt: float = 0.033
    
    # Screen info
    screen_w: int = 1920
    screen_h: int = 1080
    
    # Face detection
    face_present: int = 0
    face_confidence: float = np.nan
    
    # Blink detection
    blink_flag: int = 0
    ear_left: float = np.nan
    ear_right: float = np.nan
    
    # Raw iris positions (normalized 0-1)
    left_iris_x_norm: float = np.nan
    left_iris_y_norm: float = np.nan
    right_iris_x_norm: float = np.nan
    right_iris_y_norm: float = np.nan
    
    # Combined gaze (mean of both eyes)
    gaze_x_norm: float = np.nan
    gaze_y_norm: float = np.nan
    
    # Raw pixel positions (before compensation)
    gaze_x_px_raw: float = np.nan
    gaze_y_px_raw: float = np.nan
    
    # Anchor points for head compensation
    anchor_x_norm: float = np.nan
    anchor_y_norm: float = np.nan
    
    # Compensated gaze (after head motion compensation)
    gaze_x_norm_comp: float = np.nan
    gaze_y_norm_comp: float = np.nan
    
    # Mapped and clamped pixel positions
    gaze_x_px_comp: float = np.nan
    gaze_y_px_comp: float = np.nan
    
    # Velocity
    vel_x_px_s: float = np.nan
    vel_y_px_s: float = np.nan
    
    # Pupil proxy (iris diameter)
    pupil_proxy_left: float = np.nan
    pupil_proxy_right: float = np.nan
    pupil_proxy_mean: float = np.nan
    
    # Lighting proxy
    lighting_proxy: float = np.nan
    
    # Quality flags
    valid_sample: int = 0
    clamp_flag_x: int = 0
    clamp_flag_y: int = 0
    out_of_range_flag: int = 0
    dropout_gap_s: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            't_epoch_s': self.t_epoch_s,
            'frame_idx': self.frame_idx,
            'fps_est': self.fps_est,
            'dt': self.dt,
            'screen_w': self.screen_w,
            'screen_h': self.screen_h,
            'face_present': self.face_present,
            'face_confidence': self.face_confidence,
            'blink_flag': self.blink_flag,
            'ear_left': self.ear_left,
            'ear_right': self.ear_right,
            'left_iris_x_norm': self.left_iris_x_norm,
            'left_iris_y_norm': self.left_iris_y_norm,
            'right_iris_x_norm': self.right_iris_x_norm,
            'right_iris_y_norm': self.right_iris_y_norm,
            'gaze_x_norm': self.gaze_x_norm,
            'gaze_y_norm': self.gaze_y_norm,
            'gaze_x_px_raw': self.gaze_x_px_raw,
            'gaze_y_px_raw': self.gaze_y_px_raw,
            'anchor_x_norm': self.anchor_x_norm,
            'anchor_y_norm': self.anchor_y_norm,
            'gaze_x_norm_comp': self.gaze_x_norm_comp,
            'gaze_y_norm_comp': self.gaze_y_norm_comp,
            'gaze_x_px_comp': self.gaze_x_px_comp,
            'gaze_y_px_comp': self.gaze_y_px_comp,
            'vel_x_px_s': self.vel_x_px_s,
            'vel_y_px_s': self.vel_y_px_s,
            'pupil_proxy_left': self.pupil_proxy_left,
            'pupil_proxy_right': self.pupil_proxy_right,
            'pupil_proxy_mean': self.pupil_proxy_mean,
            'lighting_proxy': self.lighting_proxy,
            'valid_sample': self.valid_sample,
            'clamp_flag_x': self.clamp_flag_x,
            'clamp_flag_y': self.clamp_flag_y,
            'out_of_range_flag': self.out_of_range_flag,
            'dropout_gap_s': self.dropout_gap_s,
        }


class EyeTracker:
    """
    Eye tracking using MediaPipe FaceMesh with iris refinement.
    
    Provides per-frame gaze estimation, blink detection, and pupil proxy.
    """
    
    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        blink_threshold: float = 0.21,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize eye tracker.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            blink_threshold: EAR threshold for blink detection
            min_detection_confidence: MediaPipe detection confidence
            min_tracking_confidence: MediaPipe tracking confidence
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.blink_threshold = blink_threshold
        
        # Initialize MediaPipe FaceMesh with iris refinement
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enable iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # State tracking
        self.frame_idx = 0
        self.fps_calculator = RollingFPS(window_size=30)
        self.last_timestamp: Optional[float] = None
        self.last_face_timestamp: Optional[float] = None
        self.last_gaze_x_px: Optional[float] = None
        self.last_gaze_y_px: Optional[float] = None
        
        # Baseline values (set during calibration)
        self.baseline_anchor_x_norm: float = 0.5
        self.baseline_anchor_y_norm: float = 0.5
        
    def set_baseline_anchor(self, x_norm: float, y_norm: float):
        """Set baseline anchor position for head compensation."""
        self.baseline_anchor_x_norm = x_norm
        self.baseline_anchor_y_norm = y_norm
    
    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> FrameData:
        """
        Process a single frame and extract eye tracking data.
        
        Args:
            frame: BGR image from webcam
            timestamp: Optional timestamp (uses time.time() if not provided)
        
        Returns:
            FrameData with all extracted signals
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Calculate timing
        dt = 0.033  # Default
        if self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
        
        fps_est = self.fps_calculator.update(timestamp)
        
        # Initialize frame data
        data = FrameData(
            t_epoch_s=timestamp,
            frame_idx=self.frame_idx,
            fps_est=fps_est,
            dt=dt,
            screen_w=self.screen_width,
            screen_h=self.screen_height
        )
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with FaceMesh
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            data.face_present = 1
            data.face_confidence = 1.0  # MediaPipe doesn't expose confidence directly
            
            # Extract landmarks
            landmarks = face_landmarks.landmark
            
            # Get iris positions
            data.left_iris_x_norm = landmarks[LEFT_IRIS_CENTER].x
            data.left_iris_y_norm = landmarks[LEFT_IRIS_CENTER].y
            data.right_iris_x_norm = landmarks[RIGHT_IRIS_CENTER].x
            data.right_iris_y_norm = landmarks[RIGHT_IRIS_CENTER].y
            
            # Combined gaze (mean of both eyes)
            data.gaze_x_norm = (data.left_iris_x_norm + data.right_iris_x_norm) / 2
            data.gaze_y_norm = (data.left_iris_y_norm + data.right_iris_y_norm) / 2
            
            # Raw pixel positions
            data.gaze_x_px_raw = data.gaze_x_norm * self.screen_width
            data.gaze_y_px_raw = data.gaze_y_norm * self.screen_height
            
            # Anchor point (average of inner eye corners)
            anchor_x = (landmarks[LEFT_EYE_INNER].x + landmarks[RIGHT_EYE_INNER].x) / 2
            anchor_y = (landmarks[LEFT_EYE_INNER].y + landmarks[RIGHT_EYE_INNER].y) / 2
            data.anchor_x_norm = anchor_x
            data.anchor_y_norm = anchor_y
            
            # Head-compensated gaze
            data.gaze_x_norm_comp = data.gaze_x_norm - (anchor_x - self.baseline_anchor_x_norm)
            data.gaze_y_norm_comp = data.gaze_y_norm - (anchor_y - self.baseline_anchor_y_norm)
            
            # Check out of range before clamping
            if data.gaze_x_norm_comp < 0 or data.gaze_x_norm_comp > 1:
                data.out_of_range_flag = 1
            if data.gaze_y_norm_comp < 0 or data.gaze_y_norm_comp > 1:
                data.out_of_range_flag = 1
            
            # EAR calculation for blink detection
            left_eye_pts = self._get_eye_landmarks(landmarks, LEFT_EYE_INDICES)
            right_eye_pts = self._get_eye_landmarks(landmarks, RIGHT_EYE_INDICES)
            
            data.ear_left = compute_ear(left_eye_pts)
            data.ear_right = compute_ear(right_eye_pts)
            data.blink_flag = 1 if detect_blink(data.ear_left, data.ear_right, self.blink_threshold) else 0
            
            # Pupil proxy (iris diameter)
            data.pupil_proxy_left = self._compute_iris_diameter(landmarks, LEFT_IRIS_BOUNDARY)
            data.pupil_proxy_right = self._compute_iris_diameter(landmarks, RIGHT_IRIS_BOUNDARY)
            data.pupil_proxy_mean = np.nanmean([data.pupil_proxy_left, data.pupil_proxy_right])
            
            # Lighting proxy (mean intensity around eyes)
            data.lighting_proxy = self._compute_lighting_proxy(frame, landmarks)
            
            # Update last face timestamp
            self.last_face_timestamp = timestamp
            
        else:
            # No face detected
            data.face_present = 0
            data.blink_flag = 1  # Treat as blink when no face
            
            # Calculate dropout gap
            if self.last_face_timestamp is not None:
                data.dropout_gap_s = timestamp - self.last_face_timestamp
        
        # Determine valid sample
        data.valid_sample = 1 if (
            data.face_present == 1 and
            data.blink_flag == 0 and
            0.001 < dt < 0.5
        ) else 0
        
        # Update state
        self.frame_idx += 1
        self.last_timestamp = timestamp
        
        return data
    
    def _get_eye_landmarks(
        self,
        landmarks,
        indices: List[int]
    ) -> List[Tuple[float, float]]:
        """Extract eye landmarks as list of (x, y) tuples."""
        return [(landmarks[i].x, landmarks[i].y) for i in indices]
    
    def _compute_iris_diameter(
        self,
        landmarks,
        boundary_indices: List[int]
    ) -> float:
        """
        Compute iris diameter from boundary landmarks.
        
        Returns diameter in normalized coordinates.
        """
        if len(boundary_indices) < 4:
            return np.nan
        
        pts = [(landmarks[i].x, landmarks[i].y) for i in boundary_indices]
        
        # Compute horizontal and vertical diameters
        h_diameter = np.sqrt((pts[0][0] - pts[2][0])**2 + (pts[0][1] - pts[2][1])**2)
        v_diameter = np.sqrt((pts[1][0] - pts[3][0])**2 + (pts[1][1] - pts[3][1])**2)
        
        # Return mean diameter
        return (h_diameter + v_diameter) / 2
    
    def _compute_lighting_proxy(
        self,
        frame: np.ndarray,
        landmarks
    ) -> float:
        """
        Compute lighting proxy from mean grayscale intensity around eyes.
        """
        h, w = frame.shape[:2]
        
        # Get eye region bounding box
        left_eye_x = int(landmarks[LEFT_EYE_OUTER].x * w)
        right_eye_x = int(landmarks[RIGHT_EYE_OUTER].x * w)
        eye_y_top = int(min(landmarks[LEFT_EYE_INDICES[1]].y, landmarks[RIGHT_EYE_INDICES[1]].y) * h)
        eye_y_bottom = int(max(landmarks[LEFT_EYE_INDICES[4]].y, landmarks[RIGHT_EYE_INDICES[4]].y) * h)
        
        # Clamp to frame bounds
        x1 = max(0, min(left_eye_x, right_eye_x) - 20)
        x2 = min(w, max(left_eye_x, right_eye_x) + 20)
        y1 = max(0, eye_y_top - 10)
        y2 = min(h, eye_y_bottom + 10)
        
        if x2 <= x1 or y2 <= y1:
            return np.nan
        
        # Extract region and compute mean intensity
        region = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        
        return float(np.mean(gray))
    
    def reset(self):
        """Reset tracker state."""
        self.frame_idx = 0
        self.last_timestamp = None
        self.last_face_timestamp = None
        self.last_gaze_x_px = None
        self.last_gaze_y_px = None
        self.fps_calculator = RollingFPS(window_size=30)
    
    def release(self):
        """Release resources."""
        self.face_mesh.close()
