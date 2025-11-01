"""
Pose detection module using MediaPipe Pose.
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional, Tuple, Dict
from config import Config


class PoseDetector:
    """
    MediaPipe Pose detection wrapper with keypoint extraction.
    """
    
    # MediaPipe Pose landmarks indices
    LANDMARKS = {
        'nose': 0,
        'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
        'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
        'left_ear': 7, 'right_ear': 8,
        'mouth_left': 9, 'mouth_right': 10,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_pinky': 17, 'right_pinky': 18,
        'left_index': 19, 'right_index': 20,
        'left_thumb': 21, 'right_thumb': 22,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28,
        'left_heel': 29, 'right_heel': 30,
        'left_foot_index': 31, 'right_foot_index': 32
    }
    
    # Key connections for visualization
    CONNECTIONS = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7),  # left eye
        (0, 4), (4, 5), (5, 6), (6, 8),  # right eye
        (0, 9), (0, 10),  # mouth
        # Upper body
        (11, 12),  # shoulders
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (11, 23), (12, 24),  # torso
        (23, 24),  # hips
        # Lower body
        (23, 25), (25, 27),  # left leg
        (24, 26), (26, 28),  # right leg
        (27, 29), (27, 31),  # left foot
        (28, 30), (28, 32),  # right foot
        # Hand details
        (15, 17), (15, 19), (15, 21),  # left hand
        (17, 19), (19, 21),  # left hand
        (16, 18), (16, 20), (16, 22),  # right hand
        (18, 20), (20, 22),  # right hand
    ]
    
    def __init__(self, config: Config):
        """
        Initialize pose detector.
        
        Args:
            config: Application configuration object
        """
        self.config = config
        self.pose_config = config.pose
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0, 1, or 2
            enable_segmentation=False,
            min_detection_confidence=self.pose_config.min_detection_confidence,
            min_tracking_confidence=self.pose_config.min_tracking_confidence,
            smooth_landmarks=True
        )
    
    def detect(self, frame: np.ndarray) -> Optional[List[Dict]]:
        """
        Detect poses in frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of detected persons with landmarks, each as dict with:
            - landmarks: numpy array of shape (33, 3) with (x, y, visibility)
            - bounding_box: (x_min, y_min, x_max, y_max)
            - num_visible_keypoints: number of visible keypoints
        """
        if frame is None:
            return None
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform pose estimation
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks is None:
            return []
        
        # Extract landmarks for all detected persons
        # Note: MediaPipe Pose typically detects one person at a time
        persons = []
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Convert landmarks to numpy array (33 landmarks, 3 values each: x, y, visibility)
            landmarks_array = np.array([
                [lm.x, lm.y, lm.visibility] for lm in landmarks
            ])
            
            # Calculate bounding box
            visible_points = landmarks_array[landmarks_array[:, 2] > 0.5]
            if len(visible_points) > 0:
                x_coords = visible_points[:, 0]
                y_coords = visible_points[:, 1]
                
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                
                # Convert to pixel coordinates
                height, width = frame.shape[:2]
                bounding_box = (
                    int(x_min * width), int(y_min * height),
                    int(x_max * width), int(y_max * height)
                )
                
                # Count visible keypoints
                num_visible = np.sum(landmarks_array[:, 2] > 0.5)
                
                persons.append({
                    'landmarks': landmarks_array,
                    'bounding_box': bounding_box,
                    'num_visible_keypoints': num_visible,
                    'height': bounding_box[3] - bounding_box[1],
                    'completeness': num_visible / 33.0
                })
        
        return persons
    
    def draw_skeleton(self, frame: np.ndarray, persons: List[Dict], 
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2) -> np.ndarray:
        """
        Draw skeleton on frame.
        
        Args:
            frame: Input BGR frame
            persons: List of detected persons
            color: BGR color for skeleton
            thickness: Line thickness
            
        Returns:
            Frame with skeleton overlay
        """
        if frame is None or not persons:
            return frame
        
        # Draw each detected person
        for person in persons:
            landmarks = person['landmarks']
            height, width = frame.shape[:2]
            
            # Draw connections
            for connection in self.CONNECTIONS:
                idx1, idx2 = connection
                lm1 = landmarks[idx1]
                lm2 = landmarks[idx2]
                
                # Only draw if both landmarks are visible
                if lm1[2] > 0.5 and lm2[2] > 0.5:
                    pt1 = (int(lm1[0] * width), int(lm1[1] * height))
                    pt2 = (int(lm2[0] * width), int(lm2[1] * height))
                    cv2.line(frame, pt1, pt2, color, thickness)
            
            # Draw keypoints
            for landmark in landmarks:
                if landmark[2] > 0.5:  # visibility threshold
                    x = int(landmark[0] * width)
                    y = int(landmark[1] * height)
                    cv2.circle(frame, (x, y), 5, color, -1)
        
        return frame
    
    def draw_bounding_boxes(self, frame: np.ndarray, persons: List[Dict],
                           color: Tuple[int, int, int] = (255, 0, 0),
                           thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes around detected persons.
        
        Args:
            frame: Input BGR frame
            persons: List of detected persons
            color: BGR color for boxes
            thickness: Line thickness
            
        Returns:
            Frame with bounding boxes drawn
        """
        if frame is None or not persons:
            return frame
        
        for person in persons:
            bbox = person['bounding_box']
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
            
            # Draw label
            label = f"Person: {person['num_visible_keypoints']} kpts"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
        return frame
    
    def cleanup(self):
        """Release pose detector resources."""
        if hasattr(self, 'pose'):
            self.pose.close()

