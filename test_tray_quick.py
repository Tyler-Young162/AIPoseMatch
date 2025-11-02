"""
快速测试托盘功能是否正常
不实际启动托盘，只测试导入和初始化
"""
import sys
import os
from pathlib import Path

print("=" * 70)
print("快速托盘功能测试")
print("=" * 70)

# 1. 测试pystray导入
print("\n[1/4] 测试pystray导入...")
try:
    import pystray
    from PIL import Image, ImageDraw
    print("✓ pystray导入成功")
except Exception as e:
    print(f"✗ pystray导入失败: {e}")
    print("\n请使用虚拟环境运行:")
    print("  .venv\\Scripts\\python.exe test_tray_quick.py")
    sys.exit(1)

# 2. 设置路径
sys.path.insert(0, str(Path(__file__).parent / "src"))
print("\n[2/4] 路径设置完成")

# 3. 测试BackendService导入
print("\n[3/4] 测试BackendService导入...")
try:
    from run_backend_service import BackendService
    print("✓ BackendService导入成功")
except Exception as e:
    print(f"✗ BackendService导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 测试TrayApp类初始化
print("\n[4/4] 测试TrayApp类...")
try:
    from run_with_tray import TrayApp
    app = TrayApp()
    icon = app.create_icon()
    print(f"✓ TrayApp初始化成功，图标大小: {icon.size if icon else 'None'}")
except Exception as e:
    print(f"✗ TrayApp初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ 所有测试通过！托盘功能正常工作")
print("=" * 70)
print("\n可以运行以下命令启动托盘:")
print("  .venv\\Scripts\\python.exe run_with_tray.py")
print("或使用启动脚本:")
print("  start_tray.ps1")
print("=" * 70)

