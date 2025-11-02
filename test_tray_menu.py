"""
测试托盘菜单点击功能
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("测试托盘菜单功能...")
print("=" * 70)

try:
    from run_with_tray import TrayApp
    
    # 创建托盘应用实例
    app = TrayApp()
    
    # 测试菜单函数
    print("\n[1] 测试 open_single_window...")
    app.open_single_window(None, None)
    
    # 等待1秒
    import time
    time.sleep(1)
    
    print("\n[2] 测试 open_dual_window...")
    app.open_dual_window(None, None)
    
    # 等待1秒
    time.sleep(1)
    
    print("\n测试完成")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

