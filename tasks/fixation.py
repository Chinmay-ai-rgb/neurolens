import cv2
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Tuple
from core.eye_tracker import EyeTracker


class FixationTask:
    def __init__(self, output_dir: str = "data/raw_logs"):
        self.tracker = EyeTracker()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cap = None
        
    def run_task(self, ipd: float, fixation_duration: float = 10.0, cross_duration: float = 2.0):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames_data = []
        start_time = None
        
        cv2.namedWindow("Fixation Task", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Fixation Task", 1280, 720)
        
        phase = "cross"
        phase_start = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_flipped = cv2.flip(frame, 1)
            current_time = time.time()
            
            if phase == "cross":
                if current_time - phase_start < cross_duration:
                    self.draw_cross(frame_flipped, width, height)
                    cv2.putText(frame_flipped, "Fixation cross - prepare to focus", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    phase = "fixation"
                    phase_start = current_time
                    start_time = current_time
            
            if phase == "fixation":
                elapsed = current_time - phase_start
                if elapsed < fixation_duration:
                    self.draw_fixation_dot(frame_flipped, width, height)
                    cv2.putText(frame_flipped, f"Focus on green dot - {fixation_duration - elapsed:.1f}s remaining", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    break
            
            data = self.tracker.process_frame(frame_flipped)
            
            if data["tracking_valid"] and data["ipd_px"] is not None:
                frame_data = {
                    "timestamp": data["timestamp"],
                    "left_x": data["left_pupil"][0] if data["left_pupil"] else np.nan,
                    "left_y": data["left_pupil"][1] if data["left_pupil"] else np.nan,
                    "right_x": data["right_pupil"][0] if data["right_pupil"] else np.nan,
                    "right_y": data["right_pupil"][1] if data["right_pupil"] else np.nan,
                    "ipd_px": data["ipd_px"] if data["ipd_px"] else ipd
                }
                
                for idx in range(468):
                    if idx in data["landmarks"]:
                        frame_data[f"landmark_{idx}_x"] = data["landmarks"][idx][0]
                        frame_data[f"landmark_{idx}_y"] = data["landmarks"][idx][1]
                    else:
                        frame_data[f"landmark_{idx}_x"] = np.nan
                        frame_data[f"landmark_{idx}_y"] = np.nan
                
                frames_data.append(frame_data)
            else:
                cv2.putText(frame_flipped, "Tracking lost - reposition face", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Fixation Task", frame_flipped)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        if frames_data:
            df = pd.DataFrame(frames_data)
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"fixation_{timestamp_str}.csv"
            df.to_csv(output_file, index=False)
            return str(output_file)
        else:
            raise RuntimeError("No valid tracking data collected")
    
    def draw_cross(self, frame: np.ndarray, width: int, height: int, size: int = 30):
        center_x, center_y = width // 2, height // 2
        cv2.line(frame, (center_x - size, center_y), (center_x + size, center_y), (255, 255, 255), 3)
        cv2.line(frame, (center_x, center_y - size), (center_x, center_y + size), (255, 255, 255), 3)
    
    def draw_fixation_dot(self, frame: np.ndarray, width: int, height: int, radius: int = 5):
        center_x, center_y = width // 2, height // 2
        cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), -1)

