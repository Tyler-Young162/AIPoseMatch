"""
帮助脚本：检查和安装CUDA版本的PyTorch
"""
import sys
import subprocess

def check_cuda_pytorch():
    """检查当前PyTorch版本和CUDA支持"""
    print("=" * 70)
    print("PyTorch CUDA 检查工具")
    print("=" * 70)
    
    try:
        import torch
        print(f"\n当前PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("✓ CUDA可用！")
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
            print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"  设备数量: {torch.cuda.device_count()}")
            return True
        else:
            print("\n⚠ CUDA不可用")
            
            # 检查是否是CPU版本
            if "+cpu" in torch.__version__:
                print("检测到您安装的是CPU版本的PyTorch。")
                print("\n要启用CUDA支持，需要安装GPU版本的PyTorch。")
                return False
            else:
                print("PyTorch版本似乎支持CUDA，但无法检测到GPU设备。")
                print("请检查：")
                print("1. NVIDIA驱动是否已安装")
                print("2. CUDA Toolkit是否已安装")
                print("3. GPU是否支持CUDA")
                return False
                
    except ImportError:
        print("\n错误: PyTorch未安装")
        print("请先安装PyTorch")
        return False

def install_cuda_pytorch():
    """提供安装CUDA版本PyTorch的指令"""
    print("\n" + "=" * 70)
    print("安装CUDA版本PyTorch的步骤")
    print("=" * 70)
    
    print("\n1. 首先卸载当前的PyTorch（如果已安装）:")
    print("   pip uninstall torch torchvision torchaudio")
    
    print("\n2. 根据您的CUDA版本选择安装命令:")
    print("\n   CUDA 11.8:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n   CUDA 12.1:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    print("\n   CUDA 12.4:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    
    print("\n   或访问官网选择适合您系统的版本:")
    print("   https://pytorch.org/get-started/locally/")
    
    print("\n3. 验证安装:")
    print("   python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\"")
    
    print("\n" + "=" * 70)
    
    # 询问是否自动安装
    response = input("\n是否要自动卸载并安装CUDA版本的PyTorch? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\n正在卸载当前PyTorch...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], 
                         check=True)
            print("✓ 卸载完成")
            
            cuda_version = input("\n请输入您的CUDA版本 (11.8/12.1/12.4，或按Enter使用12.1): ").strip()
            if not cuda_version:
                cuda_version = "12.1"
            
            if cuda_version == "11.8":
                url = "https://download.pytorch.org/whl/cu118"
            elif cuda_version == "12.1":
                url = "https://download.pytorch.org/whl/cu121"
            elif cuda_version == "12.4":
                url = "https://download.pytorch.org/whl/cu124"
            else:
                print(f"未知的CUDA版本: {cuda_version}，使用12.1")
                url = "https://download.pytorch.org/whl/cu121"
            
            print(f"\n正在安装CUDA {cuda_version}版本的PyTorch...")
            subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", 
                          "--index-url", url], check=True)
            
            print("\n✓ 安装完成！请重新运行此脚本验证安装。")
            
        except subprocess.CalledProcessError as e:
            print(f"\n错误: 安装过程失败: {e}")
            print("请手动按照上面的步骤安装。")
    else:
        print("\n请按照上面的步骤手动安装。")

if __name__ == "__main__":
    print("\n正在检查PyTorch CUDA支持...")
    cuda_available = check_cuda_pytorch()
    
    if not cuda_available:
        install_cuda_pytorch()
        
        # 再次检查
        print("\n" + "=" * 70)
        print("重新检查...")
        print("=" * 70)
        check_cuda_pytorch()
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    print("\n提示:")
    print("- 如果CUDA可用，您的程序将自动使用GPU加速")
    print("- 如果仍不可用，请检查NVIDIA驱动和CUDA Toolkit")
    print("=" * 70)

