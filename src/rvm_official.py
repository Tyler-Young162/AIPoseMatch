"""
官方RVM (Robust Video Matting) 集成封装
如果RobustVideoMatting仓库存在，使用官方实现；否则回退到简化版本
"""
import sys
import os
from pathlib import Path
import torch
import numpy as np
from typing import Optional, Tuple

# Try to import official RVM
OFFICIAL_RVM_AVAILABLE = False
MattingNetwork = None
load_mobilenetv3 = None
load_resnet50 = None

try:
    # Check if RobustVideoMatting directory exists
    project_root = Path(__file__).parent.parent
    rvm_path = project_root / "RobustVideoMatting"
    
    if rvm_path.exists() and (rvm_path / "model" / "model.py").exists():
        print(f"[RVM Official] Found RobustVideoMatting repository at: {rvm_path}")
        # Add to path
        sys.path.insert(0, str(rvm_path))
        
        # Import official RVM
        from model import MattingNetwork as OfficialMattingNetwork
        
        MattingNetwork = OfficialMattingNetwork
        OFFICIAL_RVM_AVAILABLE = True
        print("[RVM Official] ✓ Official RVM MattingNetwork imported successfully")
        
        # Try to import helper functions (optional)
        try:
            from inference_utils import convert_video, convert_image
            print("[RVM Official] ✓ Inference utilities imported")
        except ImportError:
            print("[RVM Official] Inference utilities not available (not critical)")
            
    else:
        print(f"[RVM Official] RobustVideoMatting repository not found at: {rvm_path}")
        print("[RVM Official] Please clone the repository:")
        print("  git clone https://github.com/PeterL1n/RobustVideoMatting.git")
        
except ImportError as e:
    print(f"[RVM Official] Could not import official RVM: {e}")
    OFFICIAL_RVM_AVAILABLE = False

# Import cv2
try:
    import cv2
except ImportError:
    print("Warning: OpenCV not available")
    cv2 = None


def load_official_rvm_model(model_path: str, variant='mobilenetv3', device='cuda'):
    """
    Load official RVM model using the official implementation.
    
    Args:
        model_path: Path to model .pth file
        variant: Model variant 'mobilenetv3' or 'resnet50'
        device: Device to load model on
        
    Returns:
        Loaded model and converter function
    """
    if not OFFICIAL_RVM_AVAILABLE:
        print("[RVM Official] Official RVM not available")
        return None, None
    
    if not Path(model_path).exists():
        print(f"[RVM Official] Model file not found: {model_path}")
        return None, None
    
    try:
        print(f"[RVM Official] Loading official RVM model from: {model_path}")
        print(f"[RVM Official] Variant: {variant}, Device: {device}")
        
        # Load model using official implementation
        # Official RVM uses 'deep_guided_filter' refiner by default
        if variant == 'mobilenetv3':
            model = MattingNetwork(variant='mobilenetv3', refiner='deep_guided_filter', pretrained_backbone=False)
        else:
            model = MattingNetwork(variant='resnet50', refiner='deep_guided_filter', pretrained_backbone=False)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights (should match now!)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        total_keys = len(state_dict.keys())
        loaded_keys = total_keys - len(missing_keys)
        load_ratio = loaded_keys / total_keys if total_keys > 0 else 0
        
        print(f"[RVM Official] Weight loading: {loaded_keys}/{total_keys} ({load_ratio*100:.1f}%)")
        
        if load_ratio < 0.95:  # Less than 95% loaded is concerning
            print(f"[RVM Official] ⚠ Warning: Some weights not loaded ({len(missing_keys)} missing)")
            if missing_keys:
                print(f"[RVM Official] Missing keys: {list(missing_keys)[:5]}")
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        print(f"[RVM Official] ✓ Official RVM model loaded successfully!")
        
        return model, None  # Return model and converter (converter handled separately)
        
    except Exception as e:
        import traceback
        print(f"[RVM Official] Error loading model: {e}")
        print(f"[RVM Official] Traceback:")
        traceback.print_exc()
        return None, None


def convert_frame_to_tensor(frame: np.ndarray, device='cuda'):
    """
    Convert BGR numpy array to RGB tensor for official RVM.
    
    Args:
        frame: BGR numpy array [H, W, 3]
        device: Target device
        
    Returns:
        RGB tensor [1, 3, H, W] normalized to [0, 1]
    """
    if cv2 is None:
        raise ImportError("OpenCV is required")
    
    # BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float()
    tensor = tensor / 255.0  # Normalize to [0, 1]
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    # Move to device
    tensor = tensor.to(device)
    
    return tensor


def convert_tensor_to_alpha(alpha_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert alpha tensor to numpy array.
    
    Args:
        alpha_tensor: Alpha tensor [1, 1, H, W] or [1, H, W] in range [0, 1]
        
    Returns:
        Alpha matte as uint8 array [H, W] in range [0, 255]
    """
    # Move to CPU and convert to numpy
    if alpha_tensor.dim() == 4:
        alpha = alpha_tensor.squeeze(0).squeeze(0).cpu().numpy()
    elif alpha_tensor.dim() == 3:
        alpha = alpha_tensor.squeeze(0).cpu().numpy()
    else:
        alpha = alpha_tensor.cpu().numpy()
    
    # Clip and convert to 0-255
    alpha = np.clip(alpha, 0, 1)
    alpha_matte = (alpha * 255).astype(np.uint8)
    
    return alpha_matte

