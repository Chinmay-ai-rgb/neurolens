import cv2
import numpy as np
import time
from pathlib import Path
from typing import Tuple
from core.eye_tracker import EyeTracker


class IPDCalibration:
    def __init__(self, output_dir: str = "data/raw_logs"):
        self.tracker = EyeTracker()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cap = None
        
    def run_calibration(self, duration: float = 5.0) -> float:
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        ipd_measurements = []
        start_time = time.time()
        
        cv2.namedWindow("IPD Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("IPD Calibration", 1280, 720)
        
        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_flipped = cv2.flip(frame, 1)
            data = self.tracker.process_frame(frame_flipped)
            
            instruction_text = "Look directly at the camera. Keep your head still."
            cv2.putText(frame_flipped, instruction_text, (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if data["tracking_valid"] and data["ipd_px"] is not None:
                ipd_measurements.append(data["ipd_px"])
                cv2.putText(frame_flipped, f"IPD: {data['ipd_px']:.2f} px", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                if data["left_pupil"] and data["right_pupil"]:
                    cv2.circle(frame_flipped, (int(data["left_pupil"][0]), int(data["left_pupil"][1])), 5, (0, 0, 255), -1)
                    cv2.circle(frame_flipped, (int(data["right_pupil"][0]), int(data["right_pupil"][1])), 5, (0, 0, 255), -1)
            else:
                cv2.putText(frame_flipped, "Tracking lost - keep face in frame", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("IPD Calibration", frame_flipped)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        if ipd_measurements:
            mean_ipd = np.mean(ipd_measurements)
            return mean_ipd
        else:
            raise RuntimeError("No valid IPD measurements collected")
    
    def validate_eye_in_box(self, data: dict, frame_shape: Tuple[int, int]) -> bool:
        if not data["tracking_valid"]:
            return False
        
        if data["left_pupil"] is None or data["right_pupil"] is None:
            return False
        
        margin_x = frame_shape[1] * 0.1
        margin_y = frame_shape[0] * 0.1
        
        left_in_bounds = (margin_x < data["left_pupil"][0] < frame_shape[1] - margin_x and
                         margin_y < data["left_pupil"][1] < frame_shape[0] - margin_y)
        
        right_in_bounds = (margin_x < data["right_pupil"][0] < frame_shape[1] - margin_x and
                          margin_y < data["right_pupil"][1] < frame_shape[0] - margin_y)
        
        return left_in_bounds and right_in_bounds

