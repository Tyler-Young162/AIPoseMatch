"""
Script to download RVM model files.
"""
import os
import urllib.request
from pathlib import Path


def download_file(url: str, save_path: str, description: str = ""):
    """Download a file from URL."""
    print(f"Downloading {description}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"✓ {description} downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download {description}: {e}")
        return False


def main():
    """Download RVM model files."""
    print("RVM Model Downloader")
    print("=" * 50)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # RVM model URLs from official GitHub releases
    model_files = [
        {
            "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0/rvm_mobilenetv3.pth",
            "path": "models/rvm_mobilenetv3.pth",
            "description": "RVM MobileNetV3 model (lightweight)"
        },
        {
            "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0/rvm_resnet50.pth",
            "path": "models/rvm_resnet50.pth",
            "description": "RVM ResNet50 model (high quality)"
        }
    ]
    
    # Download model files
    for file_info in model_files:
        file_path = Path(file_info["path"])
        
        # Check if file already exists
        if file_path.exists():
            print(f"⚠ {file_info['description']} already exists, skipping...")
            continue
        
        download_file(
            file_info["url"],
            file_info["path"],
            file_info["description"]
        )
    
    print("\n" + "=" * 50)
    print("Download complete!")
    print("Note: You may need to download the model manually if the automatic download fails.")
    print("Visit: https://github.com/PeterL1n/RobustVideoMatting/releases")


if __name__ == "__main__":
    main()

