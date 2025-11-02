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
            # Load ResNet50 and remove classifier to get feature maps
            resnet = resnet50(pretrained=True)
            # Remove avgpool and fc layers, keep only feature extraction
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
            self.backbone_range = None
            # ResNet50 feature map has 2048 channels
            resnet_channels = 2048
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
            decoder_input_channels = 960  # MobileNetV3 final features
        else:
            self.feature_extractor = None
            decoder_input_channels = resnet_channels
        
        # RVM specific layers (simplified)
        # In real implementation, these would be ConvGRU and other layers
        # For now, we use a simplified decoder
        
        # IMPORTANT: This decoder is a SIMPLIFIED implementation
        # The official RVM uses a much more complex architecture with:
        # - ConvGRU for temporal consistency
        # - Multiple refinement stages
        # - Different layer names and structures
        # 
        # Our simplified decoder will NOT match the checkpoint keys,
        # so weights won't load properly. This is why RVM output is uniform.
        #
        # To fix this, we need to either:
        # 1. Use the official RVM implementation
        # 2. Train our own model with this architecture
        # 3. Manually map checkpoint keys (complex)
        
        # Simplified decoder (won't match official weights)
        self.decoder = nn.Sequential(
            # First stage: reduce channels significantly
            nn.Conv2d(decoder_input_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Second stage
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Third stage
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Final stage: output alpha + foreground
            nn.Conv2d(64, 2, 3, padding=1),
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
        # Extract feature maps (not classification output)
        # backbone now returns feature maps of shape [B, 2048, H', W']
        feat = self.backbone(src)
        
        # Decode feature maps to alpha matte
        output = self.decoder(feat)
        
        # Upsample to original input size
        if output.shape[2:] != src.shape[2:]:
            output = torch.nn.functional.interpolate(
                output, size=src.shape[2:], mode='bilinear', align_corners=False
            )
        
        # Split alpha and foreground
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
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        # Check if most weights were loaded
        total_keys = len(state_dict.keys())
        loaded_keys = total_keys - len(missing_keys)
        load_ratio = loaded_keys / total_keys if total_keys > 0 else 0
        
        # Detailed analysis of checkpoint structure
        print(f"\n{'='*70}")
        print(f"[RVM DEBUG] Model Weight Loading Analysis")
        print(f"{'='*70}")
        print(f"Total keys in checkpoint: {total_keys}")
        print(f"Successfully loaded: {loaded_keys}")
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        print(f"Match ratio: {load_ratio*100:.1f}%")
        
        # Analyze checkpoint key patterns
        checkpoint_keys = list(state_dict.keys())
        key_patterns = {}
        for key in checkpoint_keys[:20]:  # Analyze first 20 keys
            parts = key.split('.')
            if len(parts) > 0:
                prefix = parts[0]
                key_patterns[prefix] = key_patterns.get(prefix, 0) + 1
        
        print(f"\n[RVM DEBUG] Checkpoint key patterns (first 20):")
        for prefix, count in sorted(key_patterns.items()):
            example_key = next((k for k in checkpoint_keys if k.startswith(prefix)), "")
            print(f"  {prefix}* : {count} keys (e.g., {example_key[:50]})")
        
        # Show all missing keys (not just examples)
        if missing_keys:
            print(f"\n[RVM DEBUG] All missing keys ({len(missing_keys)}):")
            missing_list = list(missing_keys)
            for i, key in enumerate(missing_list[:20]):  # Show first 20
                print(f"  {i+1}. {key}")
            if len(missing_list) > 20:
                print(f"  ... and {len(missing_list) - 20} more")
        
        # Show unexpected keys (keys in model but not in checkpoint)
        if unexpected_keys:
            print(f"\n[RVM DEBUG] Unexpected keys ({len(unexpected_keys)}):")
            unexpected_list = list(unexpected_keys)
            for i, key in enumerate(unexpected_list[:10]):  # Show first 10
                print(f"  {i+1}. {key}")
            if len(unexpected_list) > 10:
                print(f"  ... and {len(unexpected_list) - 10} more")
        
        print(f"{'='*70}\n")
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        if load_ratio < 0.5:  # Less than 50% of weights loaded
            print(f"[WARN] Only {loaded_keys}/{total_keys} weights loaded ({load_ratio*100:.1f}%)")
            print(f"   This means the model architecture doesn't match the checkpoint!")
            print(f"   The decoder layers are using RANDOM initialization, which explains")
            print(f"   why the output is nearly uniform (no useful matting).")
            print(f"\n   SOLUTIONS:")
            print(f"   1. Use the official RVM implementation from GitHub")
            print(f"   2. Or use simplified matting (which is working)")
        else:
            print(f"[OK] RVM model loaded from {model_path} ({load_ratio*100:.1f}% weights matched)")
        
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

