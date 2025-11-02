"""
Human matting module using RVM (Robust Video Matting).
"""
import cv2
import numpy as np
import torch
from typing import Optional, Tuple
from pathlib import Path
from config import Config

# Try to import official RVM first, then fallback to simplified
try:
    from rvm_official import (
        OFFICIAL_RVM_AVAILABLE,
        load_official_rvm_model,
        convert_frame_to_tensor as convert_to_rgb_tensor_official,
        convert_tensor_to_alpha
    )
    OFFICIAL_RVM_AVAILABLE = OFFICIAL_RVM_AVAILABLE
except ImportError:
    OFFICIAL_RVM_AVAILABLE = False

# Try to import simplified RVM model (fallback)
try:
    from rvm_model import load_rvm_model, convert_to_rgb_tensor, convert_from_tensor
    RVM_AVAILABLE = True
except ImportError:
    RVM_AVAILABLE = False

# Overall RVM availability
RVM_AVAILABLE = OFFICIAL_RVM_AVAILABLE or RVM_AVAILABLE


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
        
        # Determine device based on configuration
        device_config = self.matting_config.device.lower()
        
        if device_config == "cuda":
            # Force CUDA - check if available
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"✓ 使用CUDA模式: {torch.cuda.get_device_name(0)}")
            else:
                print("=" * 70)
                print("⚠ 错误: 配置要求使用CUDA，但CUDA不可用！")
                print("=" * 70)
                print("\n当前PyTorch版本:", torch.__version__)
                print("检测到您安装的是CPU版本的PyTorch。")
                print("\n要启用CUDA支持，请执行以下步骤：")
                print("\n1. 卸载当前PyTorch:")
                print("   pip uninstall torch torchvision torchaudio")
                print("\n2. 根据您的CUDA版本安装GPU版本PyTorch:")
                print("   访问: https://pytorch.org/get-started/locally/")
                print("\n   例如，CUDA 11.8:")
                print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                print("\n   或 CUDA 12.1:")
                print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                print("\n3. 验证安装:")
                print("   python -c \"import torch; print(torch.cuda.is_available())\"")
                print("=" * 70)
                print("\n将回退到CPU模式...")
                self.device = torch.device("cpu")
        elif device_config == "cpu":
            self.device = torch.device("cpu")
            print(f"使用CPU模式（配置指定）")
        else:  # "auto" or default
            # Auto-detect: use CUDA if available, otherwise CPU
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"✓ 自动检测: 使用CUDA模式 ({torch.cuda.get_device_name(0)})")
            else:
                self.device = torch.device("cpu")
                print(f"使用CPU模式（自动检测，CUDA不可用）")
                print("提示: 如需使用CUDA，请在config.yaml中设置 device: 'cuda'")
        
        print(f"设备: {self.device}")
        
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
        print(f"\n{'='*60}")
        print("[DEBUG INIT] Initializing matting model...")
        print(f"[DEBUG INIT] Config model: {self.matting_config.model}")
        print(f"[DEBUG INIT] OFFICIAL_RVM_AVAILABLE: {OFFICIAL_RVM_AVAILABLE}")
        print(f"[DEBUG INIT] Simplified RVM_AVAILABLE: {RVM_AVAILABLE}")
        print(f"[DEBUG INIT] Device: {self.device}")
        print(f"{'='*60}")
        
        try:
            if self.matting_config.model.lower() == "rvm":
                # Try to load official RVM first (highest priority)
                if OFFICIAL_RVM_AVAILABLE:
                    print("[DEBUG INIT] Official RVM available, attempting to load...")
                    model_paths = [
                        "models/rvm_mobilenetv3.pth",
                        "models/rvm_resnet50.pth",
                        f"{Path(__file__).parent.parent}/models/rvm_mobilenetv3.pth",
                        f"{Path(__file__).parent.parent}/models/rvm_resnet50.pth"
                    ]
                    
                    for model_path in model_paths:
                        if Path(model_path).exists():
                            variant = 'mobilenetv3' if 'mobilenetv3' in model_path else 'resnet50'
                            print(f"[DEBUG INIT] Trying to load official RVM {variant} from: {model_path}")
                            
                            official_model, _ = load_official_rvm_model(model_path, variant, self.device)
                            if official_model is not None:
                                self.model = official_model
                                self.model_type = "rvm_official"
                                print(f"[DEBUG INIT] ✓ Official RVM loaded successfully!")
                                self.is_initialized = True
                                print(f"[DEBUG INIT] Initialization complete, model_type: {self.model_type}")
                                print(f"{'='*60}\n")
                                return True
                    
                    print("[DEBUG INIT] Official RVM available but model loading failed, trying simplified...")
                
                # Fallback to simplified RVM
                if RVM_AVAILABLE:
                    # Try different model paths
                    model_paths = [
                        "models/rvm_mobilenetv3.pth",
                        "models/rvm_resnet50.pth",
                        f"{Path(__file__).parent.parent}/models/rvm_mobilenetv3.pth",
                        f"{Path(__file__).parent.parent}/models/rvm_resnet50.pth"
                    ]
                    
                    print(f"[DEBUG INIT] Checking {len(model_paths)} possible model paths...")
                    model_loaded = False
                    for model_path in model_paths:
                        path_obj = Path(model_path)
                        exists = path_obj.exists()
                        print(f"[DEBUG INIT]   Path: {model_path} - {'EXISTS' if exists else 'NOT FOUND'}")
                        if exists:
                            file_size = path_obj.stat().st_size / (1024 * 1024)  # MB
                            print(f"[DEBUG INIT]     File size: {file_size:.1f} MB")
                            print(f"[DEBUG INIT] Attempting to load RVM model from: {model_path}")
                            variant = 'mobilenetv3' if 'mobilenetv3' in model_path else 'resnet50'
                            print(f"[DEBUG INIT] Variant: {variant}")
                            self.model = load_rvm_model(model_path, variant, self.device)
                            if self.model is not None:
                                model_loaded = True
                                self.model_type = "rvm_real"
                                print(f"[DEBUG INIT] ✓ Model loaded successfully, type: {self.model_type}")
                                break
                            else:
                                print(f"[DEBUG INIT] ✗ Model loading failed from {model_path}")
                    
                    if not model_loaded:
                        print("[DEBUG INIT] RVM model file not found. Please download it using:")
                        print("  python download_rvm_model.py")
                        print("Or visit: https://github.com/PeterL1n/RobustVideoMatting/releases")
                        print("Using simplified matting approach.")
                        self.model = "simple"
                        self.model_type = "simple"
                else:
                    print("[DEBUG INIT] RVM module not available.")
                    print("[DEBUG INIT] Using simplified matting approach.")
                    self.model = "simple"
                    self.model_type = "simple"
                
                self.is_initialized = True
                print(f"[DEBUG INIT] Initialization complete, model_type: {self.model_type}")
                print(f"{'='*60}\n")
                return True
            else:
                print(f"[DEBUG INIT] Unknown matting model: {self.matting_config.model}")
                print(f"{'='*60}\n")
                return False
                
        except Exception as e:
            import traceback
            print(f"[DEBUG INIT] ERROR initializing matting model: {e}")
            print(f"[DEBUG INIT] Traceback:")
            traceback.print_exc()
            print("[DEBUG INIT] Using simplified matting approach.")
            self.model = "simple"
            self.model_type = "simple"
            self.is_initialized = True
            print(f"{'='*60}\n")
            return True
    
    def cleanup(self):
        """Release matting model resources."""
        try:
            # 检查model属性是否存在
            if hasattr(self, 'model') and self.model is not None:
                # 只有当model是torch模块时才需要删除
                if isinstance(self.model, torch.nn.Module):
                    del self.model
                    self.model = None
                # 如果是字符串（如"simple"），直接设置为None
                elif isinstance(self.model, str):
                    self.model = None
        except Exception as e:
            # 忽略清理错误，避免影响程序退出
            pass
        
        # 安全设置is_initialized标志
        if hasattr(self, 'is_initialized'):
            self.is_initialized = False
    
    def simple_matting(self, frame: np.ndarray, bbox: Optional[tuple] = None, 
                      landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simple matting using improved color-based segmentation.
        Uses GrabCut if bbox is provided, otherwise uses HSV-based detection.
        
        Args:
            frame: Input BGR frame
            bbox: Optional bounding box tuple (x_min, y_min, x_max, y_max)
            landmarks: Optional pose landmarks for better ROI
            
        Returns:
            Alpha matte (grayscale, 0-255) with white=foreground, black=background
        """
        if frame is None:
            return None
        
        # Simple version: Use GrabCut if bbox provided, otherwise HSV
        if bbox is not None:
            return self._grabcut_matting(frame, bbox)
        else:
            return self._hsv_matting(frame)
    
    def _hsv_matting(self, frame: np.ndarray) -> np.ndarray:
        """Simple HSV-based skin tone detection."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return mask
    
    def _grabcut_matting(self, frame: np.ndarray, bbox: Optional[tuple]) -> np.ndarray:
        """Simple GrabCut-based matting with bbox."""
        if frame is None:
            return None
        
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if bbox is None:
            return self._hsv_matting(frame)
        
        x_min, y_min, x_max, y_max = bbox
        # Expand bbox slightly
        expand_ratio = 0.05
        x_min = max(0, int(x_min - (x_max - x_min) * expand_ratio))
        y_min = max(0, int(y_min - (y_max - y_min) * expand_ratio))
        x_max = min(w, int(x_max + (x_max - x_min) * expand_ratio))
        y_max = min(h, int(y_max + (y_max - y_min) * expand_ratio))
        
        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return mask
        
        try:
            gc_mask = np.zeros(roi.shape[:2], np.uint8)
            gc_bgd_model = np.zeros((1, 65), np.float64)
            gc_fgd_model = np.zeros((1, 65), np.float64)
            
            # Center region as likely foreground
            center_h, center_w = roi.shape[:2]
            inner_y1 = int(center_h * 0.25)
            inner_x1 = int(center_w * 0.25)
            inner_y2 = int(center_h * 0.75)
            inner_x2 = int(center_w * 0.75)
            
            gc_mask[inner_y1:inner_y2, inner_x1:inner_x2] = cv2.GC_PR_FGD
            gc_mask[0:3, :] = cv2.GC_BGD
            gc_mask[-3:, :] = cv2.GC_BGD
            gc_mask[:, 0:3] = cv2.GC_BGD
            gc_mask[:, -3:] = cv2.GC_BGD
            
            cv2.grabCut(roi, gc_mask, None, gc_bgd_model, gc_fgd_model, 
                       5, cv2.GC_INIT_WITH_MASK)
            
            result_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            
            # Simple cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            result_mask = cv2.GaussianBlur(result_mask, (3, 3), 0)
            
            mask[y_min:y_max, x_min:x_max] = result_mask
            
        except Exception as e:
            print(f"[DEBUG] GrabCut failed: {e}, using HSV fallback")
            return self._hsv_matting(frame)
        
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
                     bbox: Optional[tuple] = None,
                     landmarks: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Extract alpha matte from frame.
        
        Args:
            frame: Input BGR frame
            bbox: Optional bounding box to limit processing area
            
        Returns:
            Alpha matte (grayscale, 0-255) with white=foreground, black=background
        """
        if not self.is_initialized:
            print("[DEBUG MATTING] Model not initialized, initializing...")
            if not self.initialize():
                print("[DEBUG MATTING] Initialization failed!")
                return None
            print(f"[DEBUG MATTING] Model initialized, type: {self.model_type}")
        
        if frame is None:
            print("[DEBUG MATTING] Input frame is None!")
            return None
        
        print(f"[DEBUG MATTING] Extracting matte, frame shape: {frame.shape}, bbox: {bbox}")
        print(f"[DEBUG MATTING] Model type: {self.model_type}, RVM_AVAILABLE: {RVM_AVAILABLE}")
        
        # Get landmarks if not provided but available from stored data
        if landmarks is None and hasattr(self, '_last_person_data'):
            person_data = getattr(self, '_last_person_data', None)
            if person_data and 'landmarks' in person_data:
                landmarks = person_data['landmarks']
        
        print(f"[DEBUG MATTING] Using landmarks: {landmarks is not None}")
        
        # Use appropriate matting method
        if self.model_type == "rvm_official" and OFFICIAL_RVM_AVAILABLE:
            print("[DEBUG MATTING] Using OFFICIAL RVM model")
            alpha_matte = self._extract_official_rvm_matte(frame)
        elif self.model_type == "rvm_real" and RVM_AVAILABLE:
            print("[DEBUG MATTING] Using simplified RVM model")
            alpha_matte = self._extract_rvm_matte(frame)
            
            # Auto-fallback check: if RVM output has no variation, use simple matting
            if alpha_matte is not None:
                alpha_range = alpha_matte.max() - alpha_matte.min()
                if alpha_range < 10:  # Less than 10 levels of difference in 0-255 range
                    print(f"[DEBUG MATTING] ⚠ Simplified RVM output has insufficient variation (range={alpha_range:.1f})")
                    print(f"[DEBUG MATTING] Auto-falling back to simple matting")
                    alpha_matte = self.simple_matting(frame, bbox, landmarks)
                    if alpha_matte is not None:
                        print(f"[DEBUG MATTING] Simple matting range: [{alpha_matte.min()}, {alpha_matte.max()}]")
        else:
            print(f"[DEBUG MATTING] Using simple matting (model_type={self.model_type})")
            alpha_matte = self.simple_matting(frame, bbox, landmarks)
        
        if alpha_matte is None:
            print("[DEBUG MATTING] Alpha matte is None after extraction!")
        else:
            print(f"[DEBUG MATTING] Alpha matte extracted, shape: {alpha_matte.shape}, dtype: {alpha_matte.dtype}")
            print(f"[DEBUG MATTING] Alpha matte range: [{alpha_matte.min()}, {alpha_matte.max()}], mean: {alpha_matte.mean():.1f}")
        
        # Apply filtering if enabled
        if self.matting_config.filter_incomplete and alpha_matte is not None:
            old_shape = alpha_matte.shape
            alpha_matte = self.filter_incomplete_figures(alpha_matte, bbox)
            if alpha_matte is not None:
                print(f"[DEBUG MATTING] After filtering, shape: {alpha_matte.shape}, range: [{alpha_matte.min()}, {alpha_matte.max()}]")
            else:
                print(f"[DEBUG MATTING] Filtering removed all content (was shape {old_shape})")
        
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
            print("[DEBUG RVM] Starting RVM matte extraction...")
            
            if self.model is None:
                print("[DEBUG RVM] ERROR: Model is None!")
                return None
            
            print(f"[DEBUG RVM] Model device: {next(self.model.parameters()).device}")
            print(f"[DEBUG RVM] Input frame shape: {frame.shape}")
            
            # Convert to tensor
            tensor = convert_to_rgb_tensor(frame, self.device)
            print(f"[DEBUG RVM] Tensor shape: {tensor.shape}, device: {tensor.device}, dtype: {tensor.dtype}")
            print(f"[DEBUG RVM] Tensor value range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
            
            # Resize if needed for performance
            original_size = frame.shape[:2]
            downsample = self.matting_config.downsample_ratio
            print(f"[DEBUG RVM] Downsample ratio: {downsample}, original size: {original_size}")
            
            if downsample < 1.0:
                h, w = tensor.shape[2:]
                new_h, new_w = int(h * downsample), int(w * downsample)
                print(f"[DEBUG RVM] Resizing tensor from {h}x{w} to {new_h}x{new_w}")
                tensor_resized = torch.nn.functional.interpolate(
                    tensor, size=(new_h, new_w), mode='bilinear', align_corners=False
                )
            else:
                tensor_resized = tensor
                print(f"[DEBUG RVM] No resizing, using original tensor size")
            
            # Inference
            print("[DEBUG RVM] Running model inference...")
            with torch.no_grad():
                alpha, fgr = self.model(tensor_resized)
            
            print(f"[DEBUG RVM] Model output - alpha shape: {alpha.shape}, fgr shape: {fgr.shape}")
            print(f"[DEBUG RVM] Alpha raw range: [{alpha.min().item():.4f}, {alpha.max().item():.4f}], mean: {alpha.mean().item():.4f}")
            print(f"[DEBUG RVM] Alpha std: {alpha.std().item():.4f}")
            
            # Resize back to original size
            if downsample < 1.0:
                print(f"[DEBUG RVM] Resizing alpha back to {original_size}")
                alpha = torch.nn.functional.interpolate(
                    alpha, size=original_size, mode='bilinear', align_corners=False
                )
            
            # Convert to numpy
            alpha_np = alpha.squeeze().cpu().numpy()
            print(f"[DEBUG RVM] Alpha numpy shape: {alpha_np.shape}, dtype: {alpha_np.dtype}")
            print(f"[DEBUG RVM] Alpha numpy range before clip: [{alpha_np.min():.4f}, {alpha_np.max():.4f}], mean: {alpha_np.mean():.4f}")
            
            # Check if alpha is all zeros or all ones (indicates model issue)
            alpha_range = alpha_np.max() - alpha_np.min()
            if alpha_range < 0.01:
                print(f"[DEBUG RVM] ⚠ WARNING: Alpha matte has no variation! (range < 0.01)")
                print(f"[DEBUG RVM] This suggests the model is not working correctly.")
                print(f"[DEBUG RVM] Possible causes:")
                print(f"[DEBUG RVM]   1. Model weights don't match architecture (check initialization)")
                print(f"[DEBUG RVM]   2. Model is not properly trained for this task")
                print(f"[DEBUG RVM]   3. Decoder architecture too simple")
                print(f"[DEBUG RVM] Consider using simplified matting or fixing model architecture.")
            elif alpha_range < 0.1:
                print(f"[DEBUG RVM] ⚠ WARNING: Alpha matte has very low variation! (range={alpha_range:.4f})")
                print(f"[DEBUG RVM] Model output may not be useful for matting.")
            
            # Clip and convert to 0-255
            alpha_np = np.clip(alpha_np, 0, 1)
            alpha_matte = (alpha_np * 255).astype(np.uint8)
            
            print(f"[DEBUG RVM] Final alpha matte - shape: {alpha_matte.shape}, dtype: {alpha_matte.dtype}")
            print(f"[DEBUG RVM] Final alpha matte range: [{alpha_matte.min()}, {alpha_matte.max()}], mean: {alpha_matte.mean():.1f}")
            
            return alpha_matte
            
        except Exception as e:
            import traceback
            print(f"[DEBUG RVM] ERROR in RVM matting: {e}")
            print(f"[DEBUG RVM] Traceback:")
            traceback.print_exc()
            # Fallback to simple matting
            print("[DEBUG RVM] Falling back to simple matting...")
            return self.simple_matting(frame)
    
    def _extract_official_rvm_matte(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract alpha matte using official RVM model.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Alpha matte (grayscale, 0-255)
        """
        try:
            print("[DEBUG OFFICIAL RVM] Starting official RVM matte extraction...")
            
            if self.model is None:
                print("[DEBUG OFFICIAL RVM] ERROR: Model is None!")
                return None
            
            print(f"[DEBUG OFFICIAL RVM] Model device: {next(self.model.parameters()).device}")
            print(f"[DEBUG OFFICIAL RVM] Input frame shape: {frame.shape}")
            
            # Convert to tensor using official converter
            tensor = convert_to_rgb_tensor_official(frame, self.device)
            print(f"[DEBUG OFFICIAL RVM] Tensor shape: {tensor.shape}, device: {tensor.device}")
            
            original_size = frame.shape[:2]
            downsample = self.matting_config.downsample_ratio
            
            # Official RVM inference
            # Official RVM handles downsampling internally via downsample_ratio parameter
            # forward(src, r1=None, r2=None, r3=None, r4=None, downsample_ratio=1, segmentation_pass=False)
            # Returns: [fgr, pha, r1, r2, r3, r4] where pha is alpha
            print(f"[DEBUG OFFICIAL RVM] Running official RVM inference (downsample_ratio={downsample})...")
            with torch.no_grad():
                # Use recursive states from previous frame if available (for temporal consistency)
                r1 = getattr(self, '_last_r1', None)
                r2 = getattr(self, '_last_r2', None)
                r3 = getattr(self, '_last_r3', None)
                r4 = getattr(self, '_last_r4', None)
                
                output = self.model(
                    tensor,
                    r1=r1, r2=r2, r3=r3, r4=r4,
                    downsample_ratio=downsample if downsample < 1.0 else 1.0,
                    segmentation_pass=False
                )
                
                # Output is: [fgr, pha, r1, r2, r3, r4]
                fgr, alpha, r1, r2, r3, r4 = output
                
                # Store recursive states for next frame (for temporal consistency)
                # For now, we'll ignore them since we process frames independently
                self._last_r1, self._last_r2, self._last_r3, self._last_r4 = r1, r2, r3, r4
            
            print(f"[DEBUG OFFICIAL RVM] Model output - alpha shape: {alpha.shape if hasattr(alpha, 'shape') else 'N/A'}")
            
            # Alpha from official RVM is already in correct shape [1, 1, H, W] with values [0, 1]
            print(f"[DEBUG OFFICIAL RVM] Alpha shape: {alpha.shape}")
            print(f"[DEBUG OFFICIAL RVM] Alpha range: [{alpha.min().item():.4f}, {alpha.max().item():.4f}], mean: {alpha.mean().item():.4f}")
            
            # Official RVM handles downsampling internally, so alpha should match input size
            # But let's check and resize if needed
            if alpha.shape[2:] != original_size:
                print(f"[DEBUG OFFICIAL RVM] Resizing alpha from {alpha.shape[2:]} to {original_size}")
                alpha = torch.nn.functional.interpolate(
                    alpha, size=original_size, mode='bilinear', align_corners=False
                )
            
            # Convert to numpy alpha matte
            alpha_matte = convert_tensor_to_alpha(alpha)
            
            print(f"[DEBUG OFFICIAL RVM] Final alpha matte - shape: {alpha_matte.shape}, "
                  f"range: [{alpha_matte.min()}, {alpha_matte.max()}], mean: {alpha_matte.mean():.1f}")
            
            # Check quality
            alpha_range = alpha_matte.max() - alpha_matte.min()
            if alpha_range < 10:
                print(f"[DEBUG OFFICIAL RVM] ⚠ WARNING: Alpha matte has low variation (range={alpha_range:.1f})")
            else:
                print(f"[DEBUG OFFICIAL RVM] ✓ Alpha matte looks good (range={alpha_range:.1f})")
            
            return alpha_matte
            
        except Exception as e:
            import traceback
            print(f"[DEBUG OFFICIAL RVM] ERROR: {e}")
            print(f"[DEBUG OFFICIAL RVM] Traceback:")
            traceback.print_exc()
            print("[DEBUG OFFICIAL RVM] Falling back to simple matting...")
            return self.simple_matting(frame)
    
    def composite(self, frame: np.ndarray, alpha_matte: np.ndarray,
                  background: Optional[np.ndarray] = None,
                  style: str = "original") -> np.ndarray:
        """
        Composite frame with alpha matte onto background.
        
        Args:
            frame: Input BGR frame
            alpha_matte: Alpha matte (0-255)
            background: Optional background image (same size as frame), defaults to black
            
        Returns:
            Composited result
        """
        print(f"[DEBUG COMPOSITE] Starting composite, frame shape: {frame.shape if frame is not None else None}")
        print(f"[DEBUG COMPOSITE] Alpha matte shape: {alpha_matte.shape if alpha_matte is not None else None}")
        
        if frame is None or alpha_matte is None:
            print("[DEBUG COMPOSITE] ERROR: frame or alpha_matte is None, returning frame")
            return frame
        
        # Check size match
        if frame.shape[:2] != alpha_matte.shape[:2]:
            print(f"[DEBUG COMPOSITE] WARNING: Size mismatch! frame: {frame.shape[:2]}, alpha: {alpha_matte.shape[:2]}")
            print("[DEBUG COMPOSITE] Resizing alpha matte to match frame...")
            alpha_matte = cv2.resize(alpha_matte, (frame.shape[1], frame.shape[0]))
        
        # Normalize alpha to 0-1
        alpha = alpha_matte.astype(np.float32) / 255.0
        print(f"[DEBUG COMPOSITE] Alpha normalized range: [{alpha.min():.3f}, {alpha.max():.3f}], mean: {alpha.mean():.3f}")
        
        # Add alpha channel if needed
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        if background is None:
            # Use a colored background to make matting more visible
            # Green background for better contrast
            background = np.zeros_like(frame)
            background[:, :, 1] = 255  # Green channel = 255 (bright green)
            print("[DEBUG COMPOSITE] Using green background")
        else:
            print(f"[DEBUG COMPOSITE] Using provided background, shape: {background.shape}")
        
        # Expand alpha to 3 channels
        alpha = np.expand_dims(alpha, axis=2)
        alpha = np.repeat(alpha, 3, axis=2)
        print(f"[DEBUG COMPOSITE] Alpha expanded shape: {alpha.shape}")
        
        # Apply style
        if style.lower() == "silhouette":
            # 剪影风格：人物区域为纯黑色
            foreground = np.zeros_like(frame)  # 纯黑色
            print("[DEBUG COMPOSITE] Using silhouette style (black foreground)")
        else:
            # 原图风格：人物保持原色
            foreground = frame
            print("[DEBUG COMPOSITE] Using original style (original colors)")
        
        # Composite: foreground * alpha + background * (1 - alpha)
        result = (foreground * alpha + background * (1 - alpha)).astype(np.uint8)
        print(f"[DEBUG COMPOSITE] Composite result shape: {result.shape}, dtype: {result.dtype}")
        print(f"[DEBUG COMPOSITE] Result value range: [{result.min()}, {result.max()}]")
        
        return result
    
    def process(self, frame: np.ndarray, 
                bbox: Optional[tuple] = None,
                background: Optional[np.ndarray] = None,
                landmarks: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Complete matting pipeline: extract and composite.
        
        Args:
            frame: Input BGR frame
            bbox: Optional bounding box
            background: Optional background image
            landmarks: Optional pose landmarks for better ROI
            
        Returns:
            Tuple of (alpha_matte, composited_result)
        """
        print(f"\n{'='*60}")
        print(f"[DEBUG PROCESS] Starting matting process")
        print(f"[DEBUG PROCESS] Frame shape: {frame.shape if frame is not None else None}")
        print(f"[DEBUG PROCESS] Bbox: {bbox}")
        print(f"[DEBUG PROCESS] Has landmarks: {landmarks is not None}")
        print(f"[DEBUG PROCESS] Model type: {getattr(self, 'model_type', 'unknown')}")
        print(f"{'='*60}")
        
        # Use landmarks if provided, otherwise try to get from stored data
        if landmarks is None and hasattr(self, '_last_person_data'):
            person_data = getattr(self, '_last_person_data', None)
            if person_data and 'landmarks' in person_data:
                landmarks = person_data['landmarks']
        
        alpha_matte = self.extract_matte_with_landmarks(frame, bbox, landmarks)
        
        if alpha_matte is None:
            print("[DEBUG PROCESS] Alpha matte is None, cannot composite")
            return None, None
        
        # Get style from config
        style = getattr(self.matting_config, 'style', 'original')
        composited = self.composite(frame, alpha_matte, background, style=style)
        
        print(f"[DEBUG PROCESS] Process complete - alpha_matte: {alpha_matte.shape if alpha_matte is not None else None}, "
              f"composited: {composited.shape if composited is not None else None}")
        print(f"{'='*60}\n")
        
        return alpha_matte, composited
    
    def extract_matte_with_landmarks(self, frame: np.ndarray, 
                                    bbox: Optional[tuple] = None,
                                    landmarks: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Extract matte with landmarks support."""
        return self.extract_matte(frame, bbox, landmarks)

