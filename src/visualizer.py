"""
Visualization module for displaying results in real-time.
"""
import cv2
import numpy as np
from typing import Optional, List, Dict
from config import Config


class Visualizer:
    """
    Handles real-time visualization of pose detection and matting results.
    """
    
    def __init__(self, config: Config):
        """
        Initialize visualizer.
        
        Args:
            config: Application configuration object
        """
        self.config = config
        self.display_config = config.display
        
        self.fps_frames = []
        self.fps_counter = 0
        self.fps_time = None
    
    def calculate_fps(self) -> float:
        """
        Calculate current FPS.
        
        Returns:
            Current FPS value
        """
        import time
        current_time = time.time()
        
        if self.fps_time is None:
            self.fps_time = current_time
            self.fps_counter = 0
            return 0.0
        
        self.fps_counter += 1
        
        # Update FPS every second
        if current_time - self.fps_time >= 1.0:
            fps = self.fps_counter / (current_time - self.fps_time)
            self.fps_time = current_time
            self.fps_counter = 0
            return fps
        
        return -1.0  # Not enough time elapsed yet
    
    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Draw FPS counter on frame.
        
        Args:
            frame: Input BGR frame
            fps: FPS value to display
            
        Returns:
            Frame with FPS overlay
        """
        if frame is None:
            return frame
        
        if fps < 0:
            return frame
        
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def draw_status_info(self, frame: np.ndarray, 
                        best_person: Optional[Dict] = None,
                        num_detected: int = 0) -> np.ndarray:
        """
        Draw status information on frame.
        
        Args:
            frame: Input BGR frame
            best_person: Best selected person data
            num_detected: Number of detected persons
            
        Returns:
            Frame with status overlay
        """
        if frame is None:
            return frame
        
        # Detection info
        info_y = 60
        line_height = 25
        
        # Number of detections
        cv2.putText(frame, f"Detections: {num_detected}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Best person info
        if best_person is not None:
            scores = best_person.get('_selection_scores', {})
            info_y += line_height
            
            cv2.putText(frame, 
                       f"Score: {scores.get('total', 0):.3f} "
                       f"(C:{scores.get('completeness', 0):.2f} "
                       f"H:{scores.get('height', 0):.2f} "
                       f"Ct:{scores.get('centeredness', 0):.2f})",
                       (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Keypoint count
            info_y += line_height
            num_kpts = best_person.get('num_visible_keypoints', 0)
            cv2.putText(frame, f"Keypoints: {num_kpts}/33", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        else:
            info_y += line_height
            cv2.putText(frame, "No person detected", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def create_multi_panel(self, 
                          original_frame: np.ndarray,
                          roi_frame: np.ndarray,
                          pose_frame: np.ndarray,
                          matting_frame: np.ndarray) -> np.ndarray:
        """
        Create multi-panel display showing all results.
        
        Args:
            original_frame: Original frame with ROI overlay
            roi_frame: Extracted ROI
            pose_frame: ROI with skeleton overlay
            matting_frame: ROI with matting overlay
            
        Returns:
            Combined multi-panel display
        """
        # Resize all frames to same height
        target_height = 480
        target_width = 640
        
        if original_frame is not None:
            original_frame = cv2.resize(original_frame, (target_width, target_height))
        else:
            original_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        if roi_frame is not None:
            roi_frame = cv2.resize(roi_frame, (target_width, target_height))
        else:
            roi_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        if pose_frame is not None:
            pose_frame = cv2.resize(pose_frame, (target_width, target_height))
        else:
            pose_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        if matting_frame is not None:
            matting_frame = cv2.resize(matting_frame, (target_width, target_height))
        else:
            matting_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Create labels
        def add_label(img, text):
            """Add label to image."""
            label_img = img.copy()
            cv2.putText(label_img, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return label_img
        
        # Add labels
        original_frame = add_label(original_frame, "Original")
        roi_frame = add_label(roi_frame, "ROI")
        pose_frame = add_label(pose_frame, "Pose")
        matting_frame = add_label(matting_frame, "Matting")
        
        # Combine into 2x2 grid
        top_row = np.hstack([original_frame, roi_frame])
        bottom_row = np.hstack([pose_frame, matting_frame])
        combined = np.vstack([top_row, bottom_row])
        
        return combined
    
    def draw_keypoint_labels(self, frame: np.ndarray, 
                           landmarks: np.ndarray,
                           labels: Optional[List[str]] = None) -> np.ndarray:
        """
        Draw keypoint labels on frame.
        
        Args:
            frame: Input BGR frame
            landmarks: Landmarks array (N, 3) with x, y, visibility
            labels: Optional list of label names
            
        Returns:
            Frame with keypoint labels
        """
        if frame is None or landmarks is None:
            return frame
        
        height, width = frame.shape[:2]
        
        # Default keypoint names if not provided
        if labels is None:
            labels = [f"K{i}" for i in range(len(landmarks))]
        
        # Draw labels for visible keypoints
        for i, (landmark, label) in enumerate(zip(landmarks, labels)):
            if landmark[2] > 0.5:  # visibility
                x = int(landmark[0] * width)
                y = int(landmark[1] * height)
                
                # Draw label
                cv2.putText(frame, label, (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        return frame
    
    def visualize_complete(self,
                          original_frame: np.ndarray,
                          roi_frame: np.ndarray,
                          best_person: Optional[Dict],
                          alpha_matte: Optional[np.ndarray],
                          matting_result: Optional[np.ndarray],
                          num_detected: int) -> np.ndarray:
        """
        Create complete visualization with all components.
        
        Args:
            original_frame: Original frame with overlays
            roi_frame: Extracted ROI
            best_person: Best selected person data
            alpha_matte: Alpha matte
            matting_result: Composited matting result
            num_detected: Number of detected persons
            
        Returns:
            Complete visualization
        """
        # Draw pose skeleton on ROI if person detected
        pose_frame = roi_frame.copy() if roi_frame is not None else None
        
        # Draw matting result
        matting_frame = None
        if matting_result is not None:
            matting_frame = matting_result
        elif roi_frame is not None:
            matting_frame = roi_frame
        
        # Calculate FPS
        fps = self.calculate_fps()
        
        # Draw FPS and status on original frame
        original_frame = self.draw_fps(original_frame, fps)
        original_frame = self.draw_status_info(original_frame, best_person, num_detected)
        
        # Create multi-panel display
        if self.display_config.show_roi and self.display_config.show_skeleton and self.display_config.show_matting:
            display = self.create_multi_panel(
                original_frame, roi_frame, pose_frame, matting_frame
            )
        else:
            # Simple single-panel display
            display = original_frame.copy()
            if roi_frame is not None:
                display = roi_frame
        
        return display

