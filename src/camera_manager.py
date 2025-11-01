"""
Camera capture and Region of Interest (ROI) management module.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Union
from config import Config, ROIConfig


class CameraManager:
    """
    Manages webcam capture and ROI extraction.
    """
    
    def __init__(self, config: Config):
        """
        Initialize camera manager.
        
        Args:
            config: Application configuration object
        """
        self.config = config
        self.camera_config = config.camera
        self.roi_config = config.roi
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize camera capture.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Release existing camera if any
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            print(f"  正在打开摄像头 {self.camera_config.device_id}...")
            self.cap = cv2.VideoCapture(self.camera_config.device_id)
            
            print(f"  检查摄像头状态...")
            if not self.cap.isOpened():
                print(f"[ERROR] 无法打开摄像头 {self.camera_config.device_id}")
                return False
            
            print(f"  设置摄像头参数...")
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_config.fps)
            
            # Verify settings
            print(f"  验证摄像头设置...")
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"[ERROR] 摄像头初始化错误: {e}")
            return False
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.is_initialized = False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame from camera.
        
        Returns:
            Tuple of (success, frame) where frame is BGR numpy array
        """
        if not self.is_initialized or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def get_roi_coords(self, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """
        Get pixel coordinates of ROI from normalized coordinates.
        
        Args:
            frame_shape: Shape of frame as (height, width, channels)
            
        Returns:
            Tuple of (x, y, width, height) in pixels
        """
        height, width = frame_shape[:2]
        
        x_min = int(self.roi_config.x_min * width)
        x_max = int(self.roi_config.x_max * width)
        y_min = int(self.roi_config.y_min * height)
        y_max = int(self.roi_config.y_max * height)
        
        return x_min, y_min, x_max - x_min, y_max - y_min
    
    def extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract Region of Interest from frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Extracted ROI region
        """
        if frame is None:
            return None
        
        x, y, w, h = self.get_roi_coords(frame.shape)
        roi = frame[y:y+h, x:x+w].copy()
        return roi
    
    def draw_roi_overlay(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw ROI rectangle overlay on frame.
        
        Args:
            frame: Input BGR frame
            color: BGR color for rectangle
            
        Returns:
            Frame with ROI overlay drawn
        """
        if frame is None:
            return frame
        
        x, y, w, h = self.get_roi_coords(frame.shape)
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw corner markers
        corner_size = 10
        corners = [
            (x, y), (x + w, y),  # top corners
            (x, y + h), (x + w, y + h)  # bottom corners
        ]
        
        for corner in corners:
            cv2.circle(frame, corner, corner_size, color, -1)
        
        return frame
    
    def get_frame_info(self) -> dict:
        """
        Get current frame information.
        
        Returns:
            Dictionary with frame width, height, and FPS
        """
        if not self.is_initialized or self.cap is None:
            return {}
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        return {
            'width': width,
            'height': height,
            'fps': fps
        }
    
    def update_roi(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """
        Update ROI coordinates at runtime.
        
        Args:
            x_min: Left boundary (0.0-1.0)
            x_max: Right boundary (0.0-1.0)
            y_min: Top boundary (0.0-1.0)
            y_max: Bottom boundary (0.0-1.0)
        """
        self.roi_config.x_min = max(0.0, min(1.0, x_min))
        self.roi_config.x_max = max(0.0, min(1.0, x_max))
        self.roi_config.y_min = max(0.0, min(1.0, y_min))
        self.roi_config.y_max = max(0.0, min(1.0, y_max))

