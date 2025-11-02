"""
测试完整的导入链，模拟run_with_tray.py的导入过程
"""
import sys
from pathlib import Path

print("测试导入链...")
print("=" * 70)

# 模拟run_with_tray.py的路径设置
sys.path.insert(0, str(Path(__file__).parent / "src"))
print(f"[1] sys.path已设置: {sys.path[0]}")

# 测试pystray导入（在设置路径之前）
print("\n[2] 测试pystray导入（在设置路径之前）...")
try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_PYSTRAY_1 = True
    print(f"    HAS_PYSTRAY = {HAS_PYSTRAY_1}")
except Exception as e:
    HAS_PYSTRAY_1 = False
    print(f"    FAILED: {e}")
    import traceback
    traceback.print_exc()

# 测试导入backend_service（这会触发整个导入链）
print("\n[3] 测试导入run_backend_service（这会触发其他模块的导入）...")
try:
    from run_backend_service import BackendService
    print("    BackendService导入成功")
except Exception as e:
    print(f"    BackendService导入失败: {e}")
    import traceback
    traceback.print_exc()

# 再次测试pystray导入（在导入backend_service之后）
print("\n[4] 测试pystray导入（在导入backend_service之后）...")
try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_PYSTRAY_2 = True
    print(f"    HAS_PYSTRAY = {HAS_PYSTRAY_2}")
except Exception as e:
    HAS_PYSTRAY_2 = False
    print(f"    FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print(f"导入前: HAS_PYSTRAY = {HAS_PYSTRAY_1}")
print(f"导入后: HAS_PYSTRAY = {HAS_PYSTRAY_2}")

