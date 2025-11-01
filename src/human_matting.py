"""
Human matting module using RVM (Robust Video Matting).
"""
import cv2
import numpy as np
import torch
from typing import Optional, Tuple
from pathlib import Path
from config import Config

# Try to import RVM model
try:
    from rvm_model import load_rvm_model, convert_to_rgb_tensor, convert_from_tensor
    RVM_AVAILABLE = True
except ImportError:
    RVM_AVAILABLE = False
    print("Warning: RVM model module not available, using simplified matting")


class HumanMatting:
    """
    GPU-accelerated human matting using RVM or similar.
    """
    
    def __init__(self, config: Config):
        """
        Initialize human matting module.
        
        Args:
            config: Application configuration object
        """
        self.config = config
        self.matting_config = config.matting
        
        # Check CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Human matting device: {self.device}")
        
        # Model will be initialized on first use
        self.model = None
        self.is_initialized = False
        
        # State for temporal consistency (if using video matting)
        self.refiner = None
    
    def initialize(self) -> bool:
        """
        Initialize matting model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.matting_config.model.lower() == "rvm":
                # Try to load RVM
                if RVM_AVAILABLE:
                    # Try different model paths
                    model_paths = [
                        "models/rvm_mobilenetv3.pth",
                        "models/rvm_resnet50.pth",
                        f"{Path(__file__).parent.parent}/models/rvm_mobilenetv3.pth",
                        f"{Path(__file__).parent.parent}/models/rvm_resnet50.pth"
                    ]
                    
                    model_loaded = False
                    for model_path in model_paths:
                        if Path(model_path).exists():
                            print(f"Attempting to load RVM model from: {model_path}")
                            variant = 'mobilenetv3' if 'mobilenetv3' in model_path else 'resnet50'
                            self.model = load_rvm_model(model_path, variant, self.device)
                            if self.model is not None:
                                model_loaded = True
                                self.model_type = "rvm_real"
                                break
                    
                    if not model_loaded:
                        print("RVM model file not found. Please download it using:")
                        print("  python download_rvm_model.py")
                        print("Or visit: https://github.com/PeterL1n/RobustVideoMatting/releases")
                        print("Using simplified matting approach.")
                        self.model = "simple"
                        self.model_type = "simple"
                else:
                    print("RVM module not available.")
                    print("Using simplified matting approach.")
                    self.model = "simple"
                    self.model_type = "simple"
                
                self.is_initialized = True
                return True
            else:
                print(f"Unknown matting model: {self.matting_config.model}")
                return False
                
        except Exception as e:
            print(f"Error initializing matting model: {e}")
            print("Using simplified matting approach.")
            self.model = "simple"
            self.model_type = "simple"
            self.is_initialized = True
            return True
    
    def cleanup(self):
        """Release matting model resources."""
        if self.model is not None and isinstance(self.model, torch.nn.Module):
            del self.model
        self.is_initialized = False
    
    def simple_matting(self, frame: np.ndarray) -> np.ndarray:
        """
        Simple matting using color-based segmentation.
        Fallback when RVM is not available.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Alpha matte (grayscale, 0-255) with white=foreground, black=background
        """
        if frame is None:
            return None
        
        # Convert to different color spaces for better segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Use HSV to create a rough matte
        # This is a very simple approach - assumes skin tone detection
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin tones
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Apply Gaussian blur for smoother edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask
    
    def filter_incomplete_figures(self, alpha_matte: np.ndarray, 
                                  bbox: Optional[tuple] = None) -> np.ndarray:
        """
        Filter out incomplete human figures based on size criteria.
        
        Args:
            alpha_matte: Alpha matte (0-255)
            bbox: Optional bounding box tuple (x_min, y_min, x_max, y_max)
            
        Returns:
            Filtered alpha matte
        """
        if alpha_matte is None:
            return None
        
        # Find contours in the matte
        contours, _ = cv2.findContours(alpha_matte, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return alpha_matte
        
        # Calculate minimum area threshold
        frame_area = alpha_matte.shape[0] * alpha_matte.shape[1]
        min_area = frame_area * (self.matting_config.min_person_height_ratio ** 2)
        
        # Create a mask for valid contours only
        filtered_mask = np.zeros_like(alpha_matte)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                cv2.fillPoly(filtered_mask, [contour], 255)
        
        return filtered_mask
    
    def extract_matte(self, frame: np.ndarray, 
                     bbox: Optional[tuple] = None) -> Optional[np.ndarray]:
        """
        Extract alpha matte from frame.
        
        Args:
            frame: Input BGR frame
            bbox: Optional bounding box to limit processing area
            
        Returns:
            Alpha matte (grayscale, 0-255) with white=foreground, black=background
        """
        if not self.is_initialized:
            if not self.initialize():
                return None
        
        if frame is None:
            return None
        
        # Use appropriate matting method
        if self.model_type == "rvm_real" and RVM_AVAILABLE:
            alpha_matte = self._extract_rvm_matte(frame)
        else:
            alpha_matte = self.simple_matting(frame)
        
        # Apply filtering if enabled
        if self.matting_config.filter_incomplete and alpha_matte is not None:
            alpha_matte = self.filter_incomplete_figures(alpha_matte, bbox)
        
        return alpha_matte
    
    def _extract_rvm_matte(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract alpha matte using RVM model.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Alpha matte (grayscale, 0-255)
        """
        try:
            # Convert to tensor
            tensor = convert_to_rgb_tensor(frame, self.device)
            
            # Resize if needed for performance
            original_size = frame.shape[:2]
            downsample = self.matting_config.downsample_ratio
            
            if downsample < 1.0:
                h, w = tensor.shape[2:]
                new_h, new_w = int(h * downsample), int(w * downsample)
                tensor_resized = torch.nn.functional.interpolate(
                    tensor, size=(new_h, new_w), mode='bilinear', align_corners=False
                )
            else:
                tensor_resized = tensor
            
            # Inference
            with torch.no_grad():
                alpha, _ = self.model(tensor_resized)
            
            # Resize back to original size
            if downsample < 1.0:
                alpha = torch.nn.functional.interpolate(
                    alpha, size=original_size, mode='bilinear', align_corners=False
                )
            
            # Convert to numpy
            alpha_np = alpha.squeeze().cpu().numpy()
            
            # Clip and convert to 0-255
            alpha_np = np.clip(alpha_np, 0, 1)
            alpha_matte = (alpha_np * 255).astype(np.uint8)
            
            return alpha_matte
            
        except Exception as e:
            print(f"Error in RVM matting: {e}")
            # Fallback to simple matting
            return self.simple_matting(frame)
    
    def composite(self, frame: np.ndarray, alpha_matte: np.ndarray,
                  background: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Composite frame with alpha matte onto background.
        
        Args:
            frame: Input BGR frame
            alpha_matte: Alpha matte (0-255)
            background: Optional background image (same size as frame), defaults to black
            
        Returns:
            Composited result
        """
        if frame is None or alpha_matte is None:
            return frame
        
        # Normalize alpha to 0-1
        alpha = alpha_matte.astype(np.float32) / 255.0
        
        # Add alpha channel if needed
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        if background is None:
            # Use black background
            background = np.zeros_like(frame)
        
        # Expand alpha to 3 channels
        alpha = np.expand_dims(alpha, axis=2)
        alpha = np.repeat(alpha, 3, axis=2)
        
        # Composite
        result = (frame * alpha + background * (1 - alpha)).astype(np.uint8)
        
        return result
    
    def process(self, frame: np.ndarray, 
                bbox: Optional[tuple] = None,
                background: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Complete matting pipeline: extract and composite.
        
        Args:
            frame: Input BGR frame
            bbox: Optional bounding box
            background: Optional background image
            
        Returns:
            Tuple of (alpha_matte, composited_result)
        """
        alpha_matte = self.extract_matte(frame, bbox)
        composited = self.composite(frame, alpha_matte, background)
        
        return alpha_matte, composited

