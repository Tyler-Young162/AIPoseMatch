"""
测试pystray导入逻辑（模拟run_with_tray.py的导入方式）
"""
import sys
from pathlib import Path

# 模拟run_with_tray.py的导入逻辑
print("测试pystray导入逻辑...")
print("=" * 70)

try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_PYSTRAY = True
    print("[OK] pystray导入成功")
    print(f"    HAS_PYSTRAY = {HAS_PYSTRAY}")
except ImportError as e:
    HAS_PYSTRAY = False
    print(f"[ERROR] pystray导入失败（ImportError）: {e}")
    print(f"    HAS_PYSTRAY = {HAS_PYSTRAY}")
    import traceback
    traceback.print_exc()
except Exception as e:
    HAS_PYSTRAY = False
    print(f"[ERROR] pystray导入失败（其他错误）: {type(e).__name__}: {e}")
    print(f"    HAS_PYSTRAY = {HAS_PYSTRAY}")
    import traceback
    traceback.print_exc()

print("=" * 70)

# 测试创建图标
if HAS_PYSTRAY:
    print("\n测试创建图标...")
    try:
        # 创建一个简单的图标
        image = Image.new('RGB', (64, 64), color='blue')
        draw = ImageDraw.Draw(image)
        draw.ellipse([20, 15, 44, 25], fill='white')
        draw.line([32, 25, 32, 45], fill='white', width=3)
        
        icon = pystray.Icon("test", image, "Test Icon")
        print("[OK] 图标创建成功")
    except Exception as e:
        print(f"[ERROR] 创建图标失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n跳过图标创建测试（pystray不可用）")

