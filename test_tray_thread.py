"""
测试线程是否正确启动并输出
"""
import sys
import threading
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("测试托盘线程功能...")
print("=" * 70)

def test_thread():
    import sys
    print(f"[Thread] 线程开始执行 - Python路径: {sys.executable}")
    print("[Thread] 尝试导入...")
    sys.stdout.flush()
    
    try:
        from run_with_rvm import AIPoseMatchRVMDebug
        print("[Thread] 导入成功")
        sys.stdout.flush()
        
        # 尝试初始化（但不运行）
        print("[Thread] 开始初始化...")
        sys.stdout.flush()
        app = AIPoseMatchRVMDebug("config.yaml")
        print("[Thread] 初始化成功！")
        print("[Thread] 注意：这里没有调用app.run()，所以不会显示窗口")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"[Thread] 错误: {e}")
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
    finally:
        print("[Thread] 线程结束")
        sys.stdout.flush()

# 创建daemon线程
print("\n[Main] 创建线程...")
thread = threading.Thread(target=test_thread, daemon=True)
thread.start()
print("[Main] 线程已启动")

# 等待线程执行
print("[Main] 等待线程执行...")
time.sleep(3)

print("[Main] 测试完成")
print("=" * 70)

