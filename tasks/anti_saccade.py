import cv2
import numpy as np
import pandas as pd
import time
import random
from pathlib import Path
from core.eye_tracker import EyeTracker


class AntiSaccadeTask:
    def __init__(self, output_dir: str = "data/raw_logs"):
        self.tracker = EyeTracker()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cap = None
        
    def degrees_to_pixels(self, degrees: float, ipd: float, screen_distance_mm: float = 600, 
                         screen_width_px: int = 1280, screen_width_mm: float = 340) -> float:
        radians = np.deg2rad(degrees)
        distance_mm = screen_distance_mm * np.tan(radians)
        pixels = (distance_mm / screen_width_mm) * screen_width_px
        return pixels
    
    def run_task(self, ipd: float, num_trials: int = 15, amplitude_deg: float = 10.0):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames_data = []
        trial_info = []
        
        cv2.namedWindow("Anti-Saccade Task", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Anti-Saccade Task", 1280, 720)
        
        for trial in range(num_trials):
            direction = random.choice(["left", "right"])
            amplitude_px = self.degrees_to_pixels(amplitude_deg, ipd)
            
            target_x = width // 2 + amplitude_px if direction == "right" else width // 2 - amplitude_px
            target_y = height // 2
            
            correct_direction_x = width // 2 - amplitude_px if direction == "right" else width // 2 + amplitude_px
            
            fixation_start = time.time()
            stimulus_shown = False
            stimulus_onset_time = None
            initial_eye_position = None
            
            while time.time() - fixation_start < 2.0:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_flipped = cv2.flip(frame, 1)
                self.draw_fixation_dot(frame_flipped, width, height)
                cv2.putText(frame_flipped, f"Trial {trial + 1}/{num_trials} - Fixate on center", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame_flipped, "When target appears, look AWAY from it", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                data = self.tracker.process_frame(frame_flipped)
                
                if data["tracking_valid"] and data["left_pupil"]:
                    initial_eye_position = data["left_pupil"][0]
                
                self._record_frame(data, frames_data, ipd)
                
                cv2.imshow("Anti-Saccade Task", frame_flipped)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return None
                
                if time.time() - fixation_start > 1.5 and not stimulus_shown:
                    stimulus_shown = True
                    stimulus_onset_time = time.time()
            
            first_saccade_detected = False
            first_saccade_time = None
            first_saccade_direction = None
            correction_detected = False
            correction_time = None
            
            response_window_start = time.time()
            
            while time.time() - response_window_start < 2.0:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_flipped = cv2.flip(frame, 1)
                self.draw_target(frame_flipped, int(target_x), int(target_y))
                cv2.putText(frame_flipped, f"Trial {trial + 1}/{num_trials} - Look AWAY from target", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                data = self.tracker.process_frame(frame_flipped)
                
                if data["tracking_valid"] and data["left_pupil"] and initial_eye_position:
                    current_x = data["left_pupil"][0]
                    
                    if frames_data:
                        dt = data["timestamp"] - frames_data[-1]["timestamp"]
                        if dt > 0:
                            velocity = abs(current_x - frames_data[-1]["left_x"]) / dt
                        else:
                            velocity = 0.0
                    else:
                        velocity = 0.0
                    
                    movement = current_x - initial_eye_position
                    
                    if velocity > 50 and not first_saccade_detected:
                        first_saccade_detected = True
                        first_saccade_time = data["timestamp"]
                        
                        if (direction == "right" and movement > 20) or (direction == "left" and movement < -20):
                            first_saccade_direction = "incorrect"
                        else:
                            first_saccade_direction = "correct"
                    
                    if first_saccade_direction == "incorrect" and not correction_detected:
                        if (direction == "right" and movement < -20) or (direction == "left" and movement > 20):
                            if velocity > 50:
                                correction_detected = True
                                correction_time = data["timestamp"]
                
                self._record_frame(data, frames_data, ipd, stimulus_onset_time, target_x, direction)
                cv2.imshow("Anti-Saccade Task", frame_flipped)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return None
            
            trial_data = {
                "trial": trial + 1,
                "target_direction": direction,
                "first_saccade_time": first_saccade_time if first_saccade_detected else np.nan,
                "first_saccade_direction": first_saccade_direction if first_saccade_detected else "none",
                "error": first_saccade_direction == "incorrect" if first_saccade_detected else np.nan,
                "correction_time": correction_time if correction_detected else np.nan,
                "correction_latency": (correction_time - first_saccade_time) if (correction_detected and first_saccade_detected) else np.nan
            }
            trial_info.append(trial_data)
            
            time.sleep(0.5)
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        if frames_data:
            df = pd.DataFrame(frames_data)
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"anti_saccade_{timestamp_str}.csv"
            df.to_csv(output_file, index=False)
            
            trial_df = pd.DataFrame(trial_info)
            trial_file = self.output_dir / f"anti_saccade_trials_{timestamp_str}.csv"
            trial_df.to_csv(trial_file, index=False)
            
            return str(output_file)
        else:
            raise RuntimeError("No valid tracking data collected")
    
    def _record_frame(self, data: dict, frames_data: list, ipd: float, 
                     stimulus_onset: float = None, target_x: float = None, direction: str = None):
        frame_data = {
            "timestamp": data["timestamp"],
            "left_x": data["left_pupil"][0] if data["left_pupil"] else np.nan,
            "left_y": data["left_pupil"][1] if data["left_pupil"] else np.nan,
            "right_x": data["right_pupil"][0] if data["right_pupil"] else np.nan,
            "right_y": data["right_pupil"][1] if data["right_pupil"] else np.nan,
            "ipd_px": data["ipd_px"] if data["ipd_px"] else ipd,
            "stimulus_onset": stimulus_onset if stimulus_onset else np.nan,
            "target_x": target_x if target_x else np.nan,
            "target_direction": direction if direction else ""
        }
        frames_data.append(frame_data)
    
    def draw_fixation_dot(self, frame: np.ndarray, width: int, height: int, radius: int = 5):
        center_x, center_y = width // 2, height // 2
        cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), -1)
    
    def draw_target(self, frame: np.ndarray, x: int, y: int, radius: int = 8):
        cv2.circle(frame, int(x), int(y), radius, (255, 0, 0), -1)
        cv2.circle(frame, int(x), int(y), radius + 3, (255, 255, 255), 2)

