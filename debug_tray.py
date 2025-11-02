"""
直接测试run_with_tray.py的导入逻辑
"""
import sys
print(f"Python路径: {sys.executable}")
print(f"Python版本: {sys.version}")
print("=" * 70)

# 直接复制run_with_tray.py的导入代码
print("\n[1] 测试导入逻辑（和run_with_tray.py完全一致）...")
try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_PYSTRAY = True
    print(f"[OK] HAS_PYSTRAY = {HAS_PYSTRAY}")
    print(f"    pystray模块: {pystray}")
    print(f"    pystray.Icon: {pystray.Icon}")
except ImportError as e:
    HAS_PYSTRAY = False
    print(f"[ERROR] ImportError: {e}")
    print(f"    HAS_PYSTRAY = {HAS_PYSTRAY}")
    import traceback
    traceback.print_exc()
except Exception as e:
    HAS_PYSTRAY = False
    print(f"[ERROR] 其他错误: {type(e).__name__}: {e}")
    print(f"    HAS_PYSTRAY = {HAS_PYSTRAY}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("[2] 检查HAS_PYSTRAY的值...")
print(f"    HAS_PYSTRAY = {HAS_PYSTRAY}")

if not HAS_PYSTRAY:
    print("\n[3] 尝试手动导入...")
    try:
        import pystray
        print("    ✓ 手动导入pystray成功")
        print(f"    模块位置: {pystray.__file__}")
    except Exception as e:
        print(f"    ✗ 手动导入也失败: {e}")

print("\n" + "=" * 70)

