import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Optional, Tuple

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


class EyeTracker:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.left_pupil_idx = 468
        self.right_pupil_idx = 473
        self.left_eye_contour = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_contour = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
    def extract_pupil_position(self, landmarks, eye_indices: list, img_shape: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        if not landmarks:
            return None
        
        eye_points = []
        for idx in eye_indices:
            if idx < len(landmarks):
                x = landmarks[idx].x * img_shape[1]
                y = landmarks[idx].y * img_shape[0]
                eye_points.append((x, y))
        
        if len(eye_points) < 4:
            return None
        
        eye_center_x = np.mean([p[0] for p in eye_points])
        eye_center_y = np.mean([p[1] for p in eye_points])
        
        return (eye_center_x, eye_center_y)
    
    def calculate_ipd(self, landmarks, img_shape: Tuple[int, int]) -> Optional[float]:
        if landmarks is None or len(landmarks) < 473:
            return None
        
        left_eye = self.extract_pupil_position(landmarks, [33, 133], img_shape)
        right_eye = self.extract_pupil_position(landmarks, [362, 263], img_shape)
        
        if left_eye is None or right_eye is None:
            return None
        
        ipd = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
        return ipd
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        timestamp = cv2.getTickCount() / cv2.getTickFrequency()
        
        if results.multi_face_landmarks:
            landmarks_3d = results.multi_face_landmarks[0].landmark
            landmarks_dict = {}
            
            for idx, landmark in enumerate(landmarks_3d):
                landmarks_dict[idx] = (
                    landmark.x * frame.shape[1],
                    landmark.y * frame.shape[0]
                )
            
            left_pupil = None
            right_pupil = None
            
            if self.left_pupil_idx < len(landmarks_3d):
                left_landmark = landmarks_3d[self.left_pupil_idx]
                left_pupil = (left_landmark.x * frame.shape[1], left_landmark.y * frame.shape[0])
            
            if self.right_pupil_idx < len(landmarks_3d):
                right_landmark = landmarks_3d[self.right_pupil_idx]
                right_pupil = (right_landmark.x * frame.shape[1], right_landmark.y * frame.shape[0])
            
            ipd = self.calculate_ipd(landmarks_3d, frame.shape)
            
            return {
                "timestamp": timestamp,
                "left_pupil": left_pupil,
                "right_pupil": right_pupil,
                "landmarks": landmarks_dict,
                "ipd_px": ipd,
                "tracking_valid": True
            }
        else:
            return {
                "timestamp": timestamp,
                "left_pupil": None,
                "right_pupil": None,
                "landmarks": {},
                "ipd_px": None,
                "tracking_valid": False
            }
    
    def release(self):
        self.face_mesh.close()

