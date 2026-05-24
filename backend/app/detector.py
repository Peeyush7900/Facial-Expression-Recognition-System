import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Any

class FaceLandmarkDetector:
    """
    Wrapper around MediaPipe Face Mesh to perform face detection,
    landmark grouping, and face cropping.
    """
    # Group indices according to MediaPipe Face Mesh specification
    LANDMARK_GROUPS = {
        "left_eye": [33, 160, 158, 133, 153, 144, 163, 7, 145, 154, 155],
        "right_eye": [362, 385, 387, 263, 373, 380, 390, 249, 382, 381, 374],
        "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
        "right_eyebrow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
        "lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 95,
                 78, 191, 80, 81, 82, 13, 312, 311, 310, 415],
        "jawline": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 152, 
                    148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    }

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # Use static_image_mode=False for better performance in video stream;
        # Re-initialize on demand or run dynamically.
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_image(self, img_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Processes an RGB image to detect a face, extract its bounding box,
        and group landmarks.
        
        Returns:
            Dict containing:
                - 'face_detected': bool
                - 'bbox': [xmin, ymin, width, height] in pixel coordinates
                - 'landmarks': Dict of feature name to list of [x, y, z] points
                - 'cropped_face_gray': 48x48 numpy array or None
                - 'cropped_face_rgb': RGB crop or None
                - 'crop_coords': (ymin, ymax, xmin, xmax) pixel bounds of crop
        """
        h, w, _ = img_rgb.shape
        results = self.face_mesh.process(img_rgb)
        
        if not results.multi_face_landmarks:
            return {"face_detected": False, "bbox": None, "landmarks": None, "cropped_face_gray": None, "cropped_face_rgb": None}
            
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert landmarks to pixel coords and find min/max for bounding box
        x_coords = []
        y_coords = []
        all_landmarks = []
        
        for lm in face_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            x_coords.append(cx)
            y_coords.append(cy)
            all_landmarks.append([cx, cy, lm.z])
            
        # Get bounding box coordinates
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        
        # Ensure bounding box is within image bounds
        xmin = max(0, xmin)
        xmax = min(w, xmax)
        ymin = max(0, ymin)
        ymax = min(h, ymax)
        
        bbox_w = xmax - xmin
        bbox_h = ymax - ymin
        
        # Group landmarks by facial feature
        grouped_landmarks = {}
        for group_name, indices in self.LANDMARK_GROUPS.items():
            grouped_landmarks[group_name] = []
            for idx in indices:
                if idx < len(all_landmarks):
                    # Keep x, y normalized or absolute? Let's return absolute pixels for easier frontend overlay,
                    # along with original coordinates
                    lm = face_landmarks.landmark[idx]
                    grouped_landmarks[group_name].append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "px": int(lm.x * w),
                        "py": int(lm.y * h)
                    })
                    
        # Head Pose estimation using key indices
        lm_4 = face_landmarks.landmark[4]   # Nose tip
        lm_6 = face_landmarks.landmark[6]   # Nose bridge
        lm_152 = face_landmarks.landmark[152] # Chin
        lm_234 = face_landmarks.landmark[234] # Left face boundary
        lm_454 = face_landmarks.landmark[454] # Right face boundary

        # Yaw
        d_left = abs(lm_4.x - lm_234.x)
        d_right = abs(lm_454.x - lm_4.x)
        yaw_ratio = (d_left - d_right) / max(0.001, d_left + d_right)
        yaw = float(np.clip(yaw_ratio * 50.0, -45.0, 45.0))

        # Roll
        left_eye_pts = grouped_landmarks["left_eye"]
        right_eye_pts = grouped_landmarks["right_eye"]
        left_eye_x = sum(pt["x"] for pt in left_eye_pts) / len(left_eye_pts)
        left_eye_y = sum(pt["y"] for pt in left_eye_pts) / len(left_eye_pts)
        right_eye_x = sum(pt["x"] for pt in right_eye_pts) / len(right_eye_pts)
        right_eye_y = sum(pt["y"] for pt in right_eye_pts) / len(right_eye_pts)
        
        roll_rad = np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x)
        roll = float(np.clip(roll_rad * (180.0 / np.pi), -45.0, 45.0))

        # Pitch
        d_top = abs(lm_6.y - lm_4.y)
        d_bottom = abs(lm_4.y - lm_152.y)
        pitch_ratio = d_top / max(0.001, d_bottom)
        pitch = float(np.clip((pitch_ratio - 0.35) * 80.0, -30.0, 30.0))

        # Extract a padded face crop for CNN emotion model
        # Add 15% padding around the bounding box
        pad_x = int(bbox_w * 0.15)
        pad_y = int(bbox_h * 0.15)
        
        crop_xmin = max(0, xmin - pad_x)
        crop_xmax = min(w, xmax + pad_x)
        crop_ymin = max(0, ymin - pad_y)
        crop_ymax = min(h, ymax + pad_y)
        
        cropped_rgb = img_rgb[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        
        # Convert to grayscale and resize to 48x48 (FER-2013 standard)
        cropped_gray_48 = None
        if cropped_rgb.size > 0:
            gray = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2GRAY)
            cropped_gray_48 = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
            
        return {
            "face_detected": True,
            "bbox": [xmin, ymin, bbox_w, bbox_h],
            "landmarks": grouped_landmarks,
            "pose": {
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll
            },
            "cropped_face_gray": cropped_gray_48,
            "cropped_face_rgb": cropped_rgb,
            "crop_coords": (crop_ymin, crop_ymax, crop_xmin, crop_xmax)
        }

    def close(self):
        self.face_mesh.close()
