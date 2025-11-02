"""
最简单的pystray导入测试
"""
print("开始测试...")

# 完全模拟run_with_tray.py的导入顺序
import sys
import os
from pathlib import Path
import threading
import time

print("1. 基础模块导入完成")

# 在设置sys.path之前导入pystray
try:
    print("2. 尝试导入pystray...")
    import pystray
    from PIL import Image, ImageDraw
    HAS_PYSTRAY = True
    print(f"   [OK] HAS_PYSTRAY = {HAS_PYSTRAY}")
except ImportError as e:
    HAS_PYSTRAY = False
    print(f"   [FAIL] ImportError: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    HAS_PYSTRAY = False
    print(f"   [FAIL] Exception: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# 然后设置sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))
print("3. sys.path已设置")

# 尝试导入其他模块（可能触发编码错误）
try:
    print("4. 尝试导入backend_service...")
    from run_backend_service import BackendService
    print("   [OK] BackendService导入成功")
except Exception as e:
    print(f"   [FAIL] BackendService导入失败: {e}")
    import traceback
    traceback.print_exc()

# 再次检查pystray
print(f"\n最终状态: HAS_PYSTRAY = {HAS_PYSTRAY}")

