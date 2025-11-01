"""
Script to download RVM model files.
"""
import os
import urllib.request
from pathlib import Path
import sys


def download_file(url: str, save_path: str, description: str = ""):
    """Download a file from URL with progress bar."""
    print(f"\n正在下载 {description}...")
    print(f"URL: {url}")
    print(f"保存路径: {save_path}")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    def show_progress(block_num, block_size, total_size):
        """Show download progress."""
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = '=' * filled + '-' * (bar_length - filled)
        size_mb = total_size / (1024 * 1024)
        downloaded_mb = min(downloaded, total_size) / (1024 * 1024)
        sys.stdout.write(f'\r[{bar}] {percent:.1f}% ({downloaded_mb:.1f}/{size_mb:.1f} MB)')
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, save_path, show_progress)
        print(f"\n✓ {description} 下载成功！")
        return True
    except Exception as e:
        print(f"\n✗ 下载失败 {description}: {e}")
        return False


def main():
    """Download RVM model files."""
    print("RVM Model Downloader")
    print("=" * 50)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # RVM model URLs from official GitHub releases
    # Try multiple possible URLs as the release version may vary
    model_files = [
        {
            "urls": [
                "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.1.0/rvm_mobilenetv3.pth",
                "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth",
                "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0/rvm_mobilenetv3.pth",
                "https://github.com/PeterL1n/RobustVideoMatting/releases/latest/download/rvm_mobilenetv3.pth",
            ],
            "path": "models/rvm_mobilenetv3.pth",
            "description": "RVM MobileNetV3 model (lightweight)"
        },
        {
            "urls": [
                "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.1.0/rvm_resnet50.pth",
                "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth",
                "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0/rvm_resnet50.pth",
                "https://github.com/PeterL1n/RobustVideoMatting/releases/latest/download/rvm_resnet50.pth",
            ],
            "path": "models/rvm_resnet50.pth",
            "description": "RVM ResNet50 model (high quality)"
        }
    ]
    
    # Download model files
    for file_info in model_files:
        file_path = Path(file_info["path"])
        
        # Check if file already exists
        if file_path.exists():
            file_size = file_path.stat().st_size / (1024 * 1024)
            print(f"⚠ {file_info['description']} 已存在 ({file_size:.1f} MB)，跳过下载...")
            continue
        
        # Try multiple URLs until one works
        downloaded = False
        for url in file_info["urls"]:
            print(f"\n尝试下载链接: {url}")
            if download_file(url, file_info["path"], file_info["description"]):
                downloaded = True
                break
        
        if not downloaded:
            print(f"\n✗ 所有下载链接都失败了")
            print(f"请手动下载 {file_info['description']}")
            print(f"访问: https://github.com/PeterL1n/RobustVideoMatting/releases")
            print(f"将文件保存到: {file_info['path']}")
    
    print("\n" + "=" * 50)
    print("Download complete!")
    print("Note: You may need to download the model manually if the automatic download fails.")
    print("Visit: https://github.com/PeterL1n/RobustVideoMatting/releases")


if __name__ == "__main__":
    main()

