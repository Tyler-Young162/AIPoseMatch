"""
Simple dependency check script.
"""
import sys

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")
    print("-" * 50)
    
    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'mediapipe': 'MediaPipe',
        'numpy': 'NumPy',
        'yaml': 'PyYAML',
        'scipy': 'SciPy',
        'PIL': 'Pillow'
    }
    
    all_ok = True
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            all_ok = False
    
    print("-" * 50)
    
    if all_ok:
        print("All dependencies are installed!")
    else:
        print("Some dependencies are missing. Please install them:")
        print("  pip install -r requirements.txt")
        return False
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✓ CUDA is available")
            print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        else:
            print(f"\n! CUDA is NOT available (will use CPU)")
    except:
        print(f"\n! Could not check CUDA status")
    
    # Check OpenCV version and camera
    try:
        import cv2
        print(f"\n✓ OpenCV version: {cv2.__version__}")
        
        # Check if any camera is available
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print(f"✓ Camera detected")
            cap.release()
        else:
            print(f"! No camera detected")
    except Exception as e:
        print(f"! Could not check camera: {e}")
    
    return all_ok

if __name__ == "__main__":
    success = check_dependencies()
    sys.exit(0 if success else 1)

