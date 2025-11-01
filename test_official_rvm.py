"""
测试官方RVM集成是否正常工作
"""
import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from rvm_official import OFFICIAL_RVM_AVAILABLE, load_official_rvm_model
    print(f"✓ RVM official module imported")
    print(f"  OFFICIAL_RVM_AVAILABLE: {OFFICIAL_RVM_AVAILABLE}")
except Exception as e:
    print(f"✗ Failed to import rvm_official: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

if not OFFICIAL_RVM_AVAILABLE:
    print("\n⚠ Official RVM not available")
    print("Please make sure RobustVideoMatting repository is cloned:")
    print("  python setup_official_rvm.ps1")
    print("  or")
    print("  git clone https://github.com/PeterL1n/RobustVideoMatting.git")
    sys.exit(1)

print("\n测试加载模型...")
model_path = "models/rvm_mobilenetv3.pth"

if not Path(model_path).exists():
    print(f"⚠ Model file not found: {model_path}")
    print("Please download it using: python download_rvm_model.py")
    sys.exit(1)

# Check CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n使用设备: {device}")

try:
    model, _ = load_official_rvm_model(model_path, variant='mobilenetv3', device=device)
    if model is None:
        print("✗ Model loading failed")
        sys.exit(1)
    
    print("✓ Model loaded successfully!")
    
    # Test inference with dummy input
    print("\n测试推理...")
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    with torch.no_grad():
        output = model(dummy_input, downsample_ratio=1.0)
        fgr, alpha, r1, r2, r3, r4 = output
        
        print(f"✓ Inference successful!")
        print(f"  Output shapes:")
        print(f"    fgr (foreground): {fgr.shape}")
        print(f"    alpha: {alpha.shape}")
        print(f"    Alpha range: [{alpha.min().item():.4f}, {alpha.max().item():.4f}], mean: {alpha.mean().item():.4f}")
        
        alpha_range = alpha.max().item() - alpha.min().item()
        if alpha_range > 0.1:
            print(f"  ✓ Alpha matte has good variation (range={alpha_range:.4f})")
        else:
            print(f"  ⚠ Alpha matte has low variation (range={alpha_range:.4f})")
    
    print("\n" + "="*70)
    print("✓ 官方RVM集成测试通过！")
    print("="*70)
    print("\n现在可以运行主程序，官方RVM将自动使用。")
    print("运行: python run_with_rvm.py")
    
except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

