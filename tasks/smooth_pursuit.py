import cv2
import numpy as np
import pandas as pd
import time
from pathlib import Path
from core.eye_tracker import EyeTracker


class SmoothPursuitTask:
    def __init__(self, output_dir: str = "data/raw_logs"):
        self.tracker = EyeTracker()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cap = None
        
    def run_task(self, ipd: float, duration: float = 10.0, frequency: float = 0.4, amplitude: float = 0.3):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames_data = []
        start_time = time.time()
        
        cv2.namedWindow("Smooth Pursuit Task", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smooth Pursuit Task", 1280, 720)
        
        center_x = width // 2
        center_y = height // 2
        amplitude_px = width * amplitude
        
        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_flipped = cv2.flip(frame, 1)
            elapsed = time.time() - start_time
            
            target_x = center_x + amplitude_px * np.sin(2 * np.pi * frequency * elapsed)
            target_y = center_y
            
            target_velocity_x = amplitude_px * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * elapsed)
            target_velocity_y = 0.0
            
            self.draw_target(frame_flipped, int(target_x), int(target_y))
            cv2.putText(frame_flipped, f"Follow the moving dot - {duration - elapsed:.1f}s remaining", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            data = self.tracker.process_frame(frame_flipped)
            
            if data["tracking_valid"] and data["left_pupil"]:
                eye_x = data["left_pupil"][0]
                eye_y = data["left_pupil"][1]
                
                if frames_data:
                    dt = data["timestamp"] - frames_data[-1]["timestamp"]
                    if dt > 0:
                        eye_velocity_x = (eye_x - frames_data[-1]["left_x"]) / dt
                        eye_velocity_y = (eye_y - frames_data[-1]["left_y"]) / dt
                    else:
                        eye_velocity_x = 0.0
                        eye_velocity_y = 0.0
                else:
                    eye_velocity_x = 0.0
                    eye_velocity_y = 0.0
                
                frame_data = {
                    "timestamp": data["timestamp"],
                    "left_x": eye_x,
                    "left_y": eye_y,
                    "right_x": data["right_pupil"][0] if data["right_pupil"] else np.nan,
                    "right_y": data["right_pupil"][1] if data["right_pupil"] else np.nan,
                    "target_x": target_x,
                    "target_y": target_y,
                    "eye_velocity_x": eye_velocity_x,
                    "eye_velocity_y": eye_velocity_y,
                    "target_velocity_x": target_velocity_x,
                    "target_velocity_y": target_velocity_y,
                    "ipd_px": data["ipd_px"] if data["ipd_px"] else ipd
                }
                frames_data.append(frame_data)
            else:
                cv2.putText(frame_flipped, "Tracking lost - reposition face", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Smooth Pursuit Task", frame_flipped)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        if frames_data:
            df = pd.DataFrame(frames_data)
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"smooth_pursuit_{timestamp_str}.csv"
            df.to_csv(output_file, index=False)
            return str(output_file)
        else:
            raise RuntimeError("No valid tracking data collected")
    
    def draw_target(self, frame: np.ndarray, x: int, y: int, radius: int = 8):
        cv2.circle(frame, x, y, radius, (0, 255, 0), -1)
        cv2.circle(frame, x, y, radius + 3, (255, 255, 255), 2)

