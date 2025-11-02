"""
测试Unity连接设置
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("="*70)
print("测试Unity连接设置")
print("="*70)

try:
    from unity_communication import UnityCommunication
    
    print("\n[1] 创建UnityCommunication实例...")
    unity_comm = UnityCommunication()
    print("✓ 创建成功")
    
    print("\n[2] 调用connect()...")
    result = unity_comm.connect()
    
    if result:
        print("✓ connect()返回True - 服务器已启动")
        print("   等待Unity连接...")
        import time
        time.sleep(3)
        
        if unity_comm.is_connected:
            print("✓ Unity已连接")
        else:
            print("✗ Unity未连接（可能需要启动Unity）")
    else:
        print("✗ connect()返回False - 服务器启动失败")
    
    print("\n[3] 断开连接...")
    unity_comm.disconnect()
    print("✓ 已断开")
    
except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)

