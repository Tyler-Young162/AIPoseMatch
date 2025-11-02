"""
Pose matching module for comparing detected poses with target poses.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import mediapipe as mp


class PoseMatcher:
    """
    Matches detected poses with target poses and calculates similarity scores.
    """
    
    # Key body parts for comparison (躯干和四肢)
    # Using MediaPipe Pose landmark indices
    TORSO_AND_LIMBS = {
        # Torso (躯干)
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_hip': 23,
        'right_hip': 24,
        # Left arm (左臂)
        'left_elbow': 13,
        'left_wrist': 15,
        # Right arm (右臂)
        'right_elbow': 14,
        'right_wrist': 16,
        # Left leg (左腿)
        'left_knee': 25,
        'left_ankle': 27,
        # Right leg (右腿)
        'right_knee': 26,
        'right_ankle': 28,
    }
    
    # Connections for visualization (only torso and limbs)
    RELEVANT_CONNECTIONS = [
        # Torso
        (11, 12),  # shoulders
        (11, 23), (12, 24),  # shoulders to hips
        (23, 24),  # hips
        # Left arm
        (11, 13), (13, 15),  # left arm
        # Right arm
        (12, 14), (14, 16),  # right arm
        # Left leg
        (23, 25), (25, 27),  # left leg
        # Right leg
        (24, 26), (26, 28),  # right leg
    ]
    
    def __init__(self, pose_folder: str = "Pose"):
        """
        Initialize pose matcher.
        
        Args:
            pose_folder: Path to folder containing target pose images
        """
        self.pose_folder = Path(pose_folder)
        self.target_poses = []
        self.current_pose_index = 0
        
        # Initialize MediaPipe Pose for target pose detection
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load target poses
        self._load_target_poses()
    
    def _load_target_poses(self):
        """Load and detect poses from target pose images."""
        if not self.pose_folder.exists():
            print(f"[WARN] Pose folder not found: {self.pose_folder}")
            return
        
        # Find all pose images
        pose_images = sorted(self.pose_folder.glob("*.png")) + sorted(self.pose_folder.glob("*.jpg"))
        pose_images += sorted(self.pose_folder.glob("*.PNG")) + sorted(self.pose_folder.glob("*.JPG"))
        
        if len(pose_images) == 0:
            print(f"[WARN] No pose images found in {self.pose_folder}")
            return
        
        print(f"[INFO] Loading {len(pose_images)} target pose images...")
        
        for pose_path in pose_images:
            # Load image
            img = cv2.imread(str(pose_path))
            if img is None:
                print(f"[WARN] Failed to load image: {pose_path}")
                continue
            
            # Detect pose in image
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb_img)
            
            if results.pose_landmarks is None:
                print(f"[WARN] No pose detected in: {pose_path.name}")
                continue
            
            # Extract landmarks
            landmarks = np.array([
                [lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark
            ])
            
            # Store target pose
            self.target_poses.append({
                'name': pose_path.stem,
                'path': str(pose_path),
                'image': img,
                'landmarks': landmarks
            })
            
            print(f"[OK] Loaded target pose: {pose_path.name}")
        
        if len(self.target_poses) > 0:
            print(f"[OK] Successfully loaded {len(self.target_poses)} target poses")
        else:
            print(f"[ERROR] No valid target poses loaded")
    
    def get_current_target_pose(self) -> Optional[Dict]:
        """Get current target pose."""
        if len(self.target_poses) == 0:
            return None
        return self.target_poses[self.current_pose_index]
    
    def switch_to_next_pose(self):
        """Switch to next target pose (circular)."""
        if len(self.target_poses) == 0:
            return
        self.current_pose_index = (self.current_pose_index + 1) % len(self.target_poses)
        print(f"[INFO] Switched to pose: {self.target_poses[self.current_pose_index]['name']}")
    
    def switch_to_previous_pose(self):
        """Switch to previous target pose (circular)."""
        if len(self.target_poses) == 0:
            return
        self.current_pose_index = (self.current_pose_index - 1) % len(self.target_poses)
        print(f"[INFO] Switched to pose: {self.target_poses[self.current_pose_index]['name']}")
    
    def normalize_pose(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize pose landmarks to make comparison scale and position invariant.
        
        Args:
            landmarks: Array of shape (33, 3) with (x, y, visibility)
            
        Returns:
            Normalized landmarks focusing on torso and limbs
        """
        # Get relevant landmarks (torso and limbs)
        relevant_indices = list(self.TORSO_AND_LIMBS.values())
        relevant_landmarks = landmarks[relevant_indices]
        
        # Filter by visibility
        visible_mask = relevant_landmarks[:, 2] > 0.5
        if np.sum(visible_mask) < 4:  # Need at least 4 points
            return None
        
        visible_landmarks = relevant_landmarks[visible_mask]
        
        # Calculate center (use shoulder and hip average for stability)
        shoulder_hip_indices = [0, 1, 2, 3]  # shoulder and hip indices in relevant_landmarks
        if np.sum(visible_mask[:4]) >= 2:
            center_landmarks = relevant_landmarks[shoulder_hip_indices]
            center_visible = relevant_landmarks[shoulder_hip_indices, 2] > 0.5
            if np.sum(center_visible) >= 2:
                center = np.mean(center_landmarks[center_visible, :2], axis=0)
            else:
                center = np.mean(visible_landmarks[:, :2], axis=0)
        else:
            center = np.mean(visible_landmarks[:, :2], axis=0)
        
        # Calculate scale (use shoulder-hip distance as reference)
        if np.sum(visible_mask[:4]) >= 2:
            shoulder_hip_points = relevant_landmarks[shoulder_hip_indices][relevant_landmarks[shoulder_hip_indices, 2] > 0.5]
            if len(shoulder_hip_points) >= 2:
                distances = []
                for i in range(len(shoulder_hip_points)):
                    for j in range(i + 1, len(shoulder_hip_points)):
                        dist = np.linalg.norm(shoulder_hip_points[i, :2] - shoulder_hip_points[j, :2])
                        distances.append(dist)
                scale = np.mean(distances) if len(distances) > 0 else 1.0
            else:
                scale = 1.0
        else:
            # Use max distance between visible points
            if len(visible_landmarks) >= 2:
                max_dist = 0
                for i in range(len(visible_landmarks)):
                    for j in range(i + 1, len(visible_landmarks)):
                        dist = np.linalg.norm(visible_landmarks[i, :2] - visible_landmarks[j, :2])
                        max_dist = max(max_dist, dist)
                scale = max_dist if max_dist > 0 else 1.0
            else:
                scale = 1.0
        
        # Normalize
        normalized = relevant_landmarks.copy()
        if scale > 0:
            normalized[:, :2] = (normalized[:, :2] - center) / scale
        
        return normalized
    
    def calculate_pose_similarity(self, detected_landmarks: np.ndarray, 
                                   target_landmarks: np.ndarray) -> Tuple[float, Dict]:
        """
        Calculate similarity score between detected pose and target pose.
        
        Args:
            detected_landmarks: Detected pose landmarks (33, 3)
            target_landmarks: Target pose landmarks (33, 3)
            
        Returns:
            Tuple of (score 0-100, detailed match info for each joint)
        """
        # Normalize both poses
        norm_detected = self.normalize_pose(detected_landmarks)
        norm_target = self.normalize_pose(target_landmarks)
        
        if norm_detected is None or norm_target is None:
            return 0.0, {}
        
        # Calculate per-joint similarity
        joint_scores = {}
        relevant_indices = list(self.TORSO_AND_LIMBS.values())
        relevant_names = list(self.TORSO_AND_LIMBS.keys())
        
        total_score = 0.0
        count = 0
        
        for idx, name in zip(relevant_indices, relevant_names):
            # Get normalized positions for this joint
            det_idx = relevant_indices.index(idx) if idx in relevant_indices else -1
            if det_idx < 0 or det_idx >= len(norm_detected) or det_idx >= len(norm_target):
                continue
            
            det_joint = norm_detected[det_idx]
            target_joint = norm_target[det_idx]
            
            # Check visibility
            if det_joint[2] < 0.5 or target_joint[2] < 0.5:
                joint_scores[name] = 0.0
                continue
            
            # Calculate distance
            dist = np.linalg.norm(det_joint[:2] - target_joint[:2])
            
            # Convert distance to similarity score (0-1)
            # Using exponential decay: score = exp(-k * dist)
            # Adjust k to control sensitivity (larger k = more sensitive)
            k = 3.0
            similarity = np.exp(-k * dist)
            
            joint_scores[name] = similarity
            total_score += similarity
            count += 1
        
        # Average score and convert to 0-100
        if count > 0:
            avg_score = (total_score / count) * 100.0
        else:
            avg_score = 0.0
        
        return avg_score, joint_scores
    
    def get_joint_color(self, similarity: float) -> Tuple[int, int, int]:
        """
        Get color for joint based on similarity.
        Blue (default, low similarity) -> Yellow (high similarity)
        
        Args:
            similarity: Similarity score (0.0-1.0)
            
        Returns:
            BGR color tuple
        """
        # Blue to Yellow gradient
        # Blue: (255, 0, 0) in BGR
        # Yellow: (0, 255, 255) in BGR
        
        # Interpolate between blue and yellow
        blue = np.array([255, 0, 0])
        yellow = np.array([0, 255, 255])
        
        color = blue + (yellow - blue) * similarity
        color = np.clip(color, 0, 255).astype(int)
        
        return tuple(color.tolist())
    
    def draw_skeleton_with_matching(self, frame: np.ndarray, 
                                     detected_person: Dict,
                                     target_pose: Optional[Dict] = None,
                                     thickness: int = 3) -> np.ndarray:
        """
        Draw skeleton with color-coded joints based on match quality.
        
        Args:
            frame: Input BGR frame
            detected_person: Detected person data with landmarks
            target_pose: Target pose data (if None, uses current target)
            thickness: Line thickness
            
        Returns:
            Frame with colored skeleton overlay
        """
        if frame is None or detected_person is None:
            return frame
        
        if target_pose is None:
            target_pose = self.get_current_target_pose()
        
        detected_landmarks = detected_person['landmarks']
        height, width = frame.shape[:2]
        
        # Calculate similarity if target pose available
        joint_similarities = {}
        if target_pose is not None:
            _, joint_similarities = self.calculate_pose_similarity(
                detected_landmarks, target_pose['landmarks']
            )
        
        # Draw connections with color based on match
        for connection in self.RELEVANT_CONNECTIONS:
            idx1, idx2 = connection
            lm1 = detected_landmarks[idx1]
            lm2 = detected_landmarks[idx2]
            
            if lm1[2] > 0.5 and lm2[2] > 0.5:
                pt1 = (int(lm1[0] * width), int(lm1[1] * height))
                pt2 = (int(lm2[0] * width), int(lm2[1] * height))
                
                # Get average similarity for this connection
                # Find which joints these correspond to
                avg_sim = 0.0
                count = 0
                
                for name, idx in self.TORSO_AND_LIMBS.items():
                    if idx == idx1:
                        if name in joint_similarities:
                            avg_sim += joint_similarities[name]
                            count += 1
                    if idx == idx2:
                        if name in joint_similarities:
                            avg_sim += joint_similarities[name]
                            count += 1
                
                if count > 0:
                    avg_sim = avg_sim / count
                else:
                    avg_sim = 0.0  # Default blue
                
                # Get color
                color = self.get_joint_color(avg_sim)
                cv2.line(frame, pt1, pt2, color, thickness)
        
        # Draw keypoints with color
        for name, idx in self.TORSO_AND_LIMBS.items():
            landmark = detected_landmarks[idx]
            if landmark[2] > 0.5:  # visible
                x = int(landmark[0] * width)
                y = int(landmark[1] * height)
                
                # Get similarity for this joint
                similarity = joint_similarities.get(name, 0.0)
                color = self.get_joint_color(similarity)
                
                cv2.circle(frame, (x, y), 6, color, -1)
                cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)  # White outline
        
        return frame
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'pose_detector'):
            self.pose_detector.close()

