"""
诊断pystray导入问题
"""
import sys
import os

print("=" * 70)
print("pystray导入诊断工具")
print("=" * 70)

# 1. 检查Python环境
print("\n[1] Python环境信息：")
print(f"  Python版本: {sys.version}")
print(f"  Python路径: {sys.executable}")
print(f"  工作目录: {os.getcwd()}")

# 2. 检查pystray安装
print("\n[2] 检查pystray安装：")
try:
    import pip
    installed_packages = [p.key for p in pip.get_installed_distributions()]
    if 'pystray' in installed_packages:
        print("  ✓ pystray在已安装包列表中")
    else:
        print("  ✗ pystray不在已安装包列表中")
except:
    try:
        import pkg_resources
        installed_packages = [p.key for p in pkg_resources.working_set]
        if 'pystray' in installed_packages:
            print("  ✓ pystray在已安装包列表中")
            # 获取版本
            pkg = pkg_resources.get_distribution('pystray')
            print(f"  版本: {pkg.version}")
        else:
            print("  ✗ pystray不在已安装包列表中")
    except Exception as e:
        print(f"  无法检查已安装包: {e}")

# 3. 尝试导入pystray
print("\n[3] 尝试导入pystray：")
try:
    import pystray
    print("  ✓ pystray导入成功")
    print(f"  pystray模块路径: {pystray.__file__}")
    print(f"  pystray模块属性: {dir(pystray)[:10]}")
except ImportError as e:
    print(f"  ✗ pystray导入失败: {e}")
    print(f"  错误类型: {type(e).__name__}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"  ✗ 导入时出现其他错误: {e}")
    print(f"  错误类型: {type(e).__name__}")
    import traceback
    traceback.print_exc()

# 4. 检查依赖（PIL/Pillow）
print("\n[4] 检查依赖库：")
try:
    from PIL import Image, ImageDraw
    print("  ✓ PIL/Pillow导入成功")
except ImportError as e:
    print(f"  ✗ PIL/Pillow导入失败: {e}")
    print("  注意: pystray依赖PIL/Pillow")

# 5. 检查sys.path
print("\n[5] Python模块搜索路径：")
for i, path in enumerate(sys.path[:5]):
    print(f"  [{i}] {path}")

# 6. 尝试重新安装
print("\n[6] 建议的解决方案：")
print("  如果pystray未安装或导入失败，请运行：")
print("    pip install pystray")
print("  或者如果使用虚拟环境：")
print("    .venv\\Scripts\\pip install pystray")

print("\n" + "=" * 70)

