"""
快速诊断后台服务启动问题
逐步测试各模块加载
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("="*70)
print("快速诊断后台服务")
print("="*70)

try:
    print("\n[1/6] 导入配置模块...")
    sys.stdout.flush()
    from config import Config
    print("✓ Config导入成功")
    
    print("\n[2/6] 加载配置文件...")
    sys.stdout.flush()
    config = Config.load_from_yaml("config.yaml")
    print("✓ 配置文件加载成功")
    
    print("\n[3/6] 导入Unity通信模块...")
    sys.stdout.flush()
    from unity_communication import UnityCommunication
    print("✓ Unity通信模块导入成功")
    
    print("\n[4/6] 创建Unity通信实例...")
    sys.stdout.flush()
    unity_comm = UnityCommunication()
    print("✓ Unity通信实例创建成功")
    print("  提示：这里不会连接，只是创建对象")
    
    print("\n[5/6] 导入其他核心模块...")
    sys.stdout.flush()
    from pose_matcher import PoseMatcher
    print("✓ PoseMatcher导入成功")
    
    print("\n[6/6] 测试姿态匹配器初始化...")
    sys.stdout.flush()
    pose_matcher = PoseMatcher(pose_folder="Pose")
    print(f"✓ 姿态匹配器初始化成功，加载了 {len(pose_matcher.target_poses)} 个姿态")
    
    print("\n" + "="*70)
    print("✓ 所有基本模块加载成功！")
    print("="*70)
    print("\n结论：基本模块无问题，卡住可能是在摄像头初始化或读取帧时")
    print("     建议使用 --show-window 参数查看详细进度")
    print("="*70)
    
except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    print("="*70)

