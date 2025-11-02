"""
最简单地测试托盘启动窗口功能
"""
import sys
import threading
from pathlib import Path
import time

# 不设置无缓冲，改为手动flush

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("="*70)
print("测试托盘启动窗口功能")
print("="*70)

def test_app():
    print("\n[线程] 开始执行...")
    sys.stdout.flush()
    try:
        print("[线程] 导入AIPoseMatchRVMDebug...")
        sys.stdout.flush()
        from run_with_rvm import AIPoseMatchRVMDebug
        print("[线程] 导入成功")
        sys.stdout.flush()
        
        print("[线程] 创建实例...")
        sys.stdout.flush()
        app = AIPoseMatchRVMDebug("config.yaml")
        print("[线程] 实例创建成功")
        sys.stdout.flush()
        
        print("[线程] 跳过app.run()（避免阻塞测试）")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"[线程] 错误: {e}")
        sys.stdout.flush()
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\n[主线程] 创建测试线程...")
    sys.stdout.flush()
    thread = threading.Thread(target=test_app, daemon=True)
    thread.start()
    
    print("[主线程] 等待线程...")
    sys.stdout.flush()
    time.sleep(3)
    
    print("\n[主线程] 测试完成")
    sys.stdout.flush()
    print("="*70)

