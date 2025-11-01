"""
RVM (Robust Video Matting) model integration.
Adapted from: https://github.com/PeterL1n/RobustVideoMatting
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, resnet50
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np

# Import cv2
try:
    import cv2
except ImportError:
    print("Warning: OpenCV not available")
    cv2 = None


class MattingNetwork(nn.Module):
    """
    RVM Matting Network.
    Simplified version for integration.
    """
    
    def __init__(self, variant='mobilenetv3'):
        """
        Initialize RVM model.
        
        Args:
            variant: Model variant 'mobilenetv3' or 'resnet50'
        """
        super().__init__()
        self.variant = variant
        
        # Build backbone
        if variant == 'mobilenetv3':
            self.backbone = mobilenet_v3_large(pretrained=True).features
            self.backbone_range = [4, 10, 15]
        elif variant == 'resnet50':
            self.backbone = resnet50(pretrained=True)
            self.backbone_range = None
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        # Create feature extractor
        if variant == 'mobilenetv3':
            self.feature_extractor = create_feature_extractor(
                self.backbone,
                return_nodes={
                    f'features.{i}': f'features.{i}' 
                    for i in self.backbone_range
                }
            )
        else:
            self.feature_extractor = None
        
        # RVM specific layers (simplified)
        # In real implementation, these would be ConvGRU and other layers
        # For now, we use a simplified decoder
        
        # Simplified decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(960, 128, 3, padding=1),  # Adjust input channels based on variant
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 3, padding=1),  # Alpha + foreground
            nn.Sigmoid()
        )
    
    def forward(self, src, *args, **kwargs):
        """
        Forward pass.
        
        Args:
            src: Source image tensor [B, 3, H, W]
            
        Returns:
            Tuple of (alpha, foreground) or simplified output
        """
        if self.variant == 'mobilenetv3':
            return self._forward_mobilenetv3(src)
        else:
            return self._forward_resnet50(src)
    
    def _forward_mobilenetv3(self, src):
        """Forward pass for MobileNetV3 variant."""
        # Extract features
        features_dict = self.feature_extractor(src)
        
        # Get the final feature map
        feat = features_dict['features.15']
        
        # Decode
        output = self.decoder(feat)
        
        # Upsample to original size
        if output.shape[2:] != src.shape[2:]:
            output = torch.nn.functional.interpolate(
                output, size=src.shape[2:], mode='bilinear', align_corners=False
            )
        
        # Split alpha and foreground
        alpha = output[:, 0:1]
        fgr = output[:, 1:2].repeat(1, 3, 1, 1) * src
        
        return alpha, fgr
    
    def _forward_resnet50(self, src):
        """Forward pass for ResNet50 variant."""
        # Simplified implementation
        feat = self.backbone(src)
        
        # Decode (needs proper implementation)
        output = self.decoder(feat)
        
        if output.shape[2:] != src.shape[2:]:
            output = torch.nn.functional.interpolate(
                output, size=src.shape[2:], mode='bilinear', align_corners=False
            )
        
        alpha = output[:, 0:1]
        fgr = output[:, 1:2].repeat(1, 3, 1, 1) * src
        
        return alpha, fgr


def load_rvm_model(model_path: str, variant='mobilenetv3', device='cuda'):
    """
    Load RVM model from checkpoint.
    
    Args:
        model_path: Path to model .pth file
        variant: Model variant
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    from pathlib import Path
    
    # Check if model file exists
    if not Path(model_path).exists():
        print(f"Warning: Model file not found: {model_path}")
        print("Using simplified fallback model")
        return None
    
    try:
        # Create model
        model = MattingNetwork(variant=variant)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights (strict=False to handle missing keys)
        model.load_state_dict(state_dict, strict=False)
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        print(f"✓ RVM model loaded from {model_path}")
        return model
        
    except Exception as e:
        print(f"Error loading RVM model: {e}")
        print("Using simplified fallback model")
        return None


def convert_to_rgb_tensor(frame, device='cuda'):
    """
    Convert BGR numpy array to RGB tensor.
    
    Args:
        frame: BGR numpy array [H, W, 3]
        device: Target device
        
    Returns:
        RGB tensor [1, 3, H, W]
    """
    if cv2 is None:
        raise ImportError("OpenCV is required for convert_to_rgb_tensor")
    
    # BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # To tensor
    tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float()
    tensor = tensor / 255.0
    tensor = tensor.unsqueeze(0)
    
    # To device
    tensor = tensor.to(device)
    
    return tensor


def convert_from_tensor(tensor):
    """
    Convert tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        Numpy array
    """
    # To CPU
    tensor = tensor.cpu()
    
    # Squeeze batch dimension if present
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    
    # Permute if needed
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    
    # To numpy
    array = tensor.numpy()
    
    # Clip to valid range
    array = np.clip(array, 0, 1)
    
    return array

