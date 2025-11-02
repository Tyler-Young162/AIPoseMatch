"""
AI Pose Match - 带系统托盘的主程序
默认启动时不显示窗口，只显示系统托盘图标
双击图标打开预览窗口
"""
import sys
import os
from pathlib import Path
import threading
import time

# 首先尝试导入pystray（在其他导入之前，避免被其他错误影响）
try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_PYSTRAY = True
    print("[DEBUG] pystray导入成功，HAS_PYSTRAY = True")
except ImportError as e:
    HAS_PYSTRAY = False
    print(f"[WARN] pystray未安装，无法使用系统托盘功能")
    print(f"      导入错误: {e}")
    print("      请运行: pip install pystray")
    import traceback
    traceback.print_exc()
except Exception as e:
    HAS_PYSTRAY = False
    print(f"[WARN] pystray导入时出现其他错误: {type(e).__name__}: {e}")
    print("      请运行: pip install pystray")
    import traceback
    traceback.print_exc()

# Add src directory to path（在pystray导入之后）
sys.path.insert(0, str(Path(__file__).parent / "src"))


class TrayApp:
    """
    系统托盘应用管理器
    """
    def __init__(self):
        self.tray_icon = None
        self.app_window = None
        self.app_thread = None
        self.is_dual_mode = False
        self.last_click_time = 0
        self.click_timeout = 0.5  # 双击检测时间窗口（秒）
        
    def create_icon(self):
        """创建托盘图标"""
        # 创建一个简单的图标
        image = Image.new('RGB', (64, 64), color='blue')
        draw = ImageDraw.Draw(image)
        # 绘制一个简单的姿势图标（简化版）
        draw.ellipse([20, 15, 44, 25], fill='white')  # 头部
        draw.line([32, 25, 32, 45], fill='white', width=3)  # 躯干
        draw.line([32, 32, 25, 40], fill='white', width=3)  # 左臂
        draw.line([32, 32, 39, 40], fill='white', width=3)  # 右臂
        draw.line([32, 45, 27, 55], fill='white', width=3)  # 左腿
        draw.line([32, 45, 37, 55], fill='white', width=3)  # 右腿
        return image
    
    def on_click(self, icon, item):
        """处理图标点击事件（实现双击检测）"""
        import time
        current_time = time.time()
        
        # 检测双击：两次点击间隔小于timeout
        if current_time - self.last_click_time < self.click_timeout:
            # 双击：打开窗口
            if self.app_window is None or not self.app_window.is_alive():
                self.open_window()
            self.last_click_time = 0  # 重置，避免三次点击触发
        else:
            # 单击：记录时间，等待可能的第二次点击
            self.last_click_time = current_time
    
    def open_window(self, dual_mode=False, backend_mode=False, show_window=True):
        """
        打开应用窗口
        
        Args:
            dual_mode: 是否双摄像头模式
            backend_mode: 是否后台服务模式（不显示窗口，推送Unity）
            show_window: 后台模式下是否显示调试窗口
        """
        print(f"[DEBUG] open_window被调用: dual_mode={dual_mode}, backend_mode={backend_mode}, show_window={show_window}")
        
        if self.app_window is not None and self.app_window.is_alive():
            print("[DEBUG] 应用窗口已在运行")
            return
        
        self.is_dual_mode = dual_mode
        
        def run_app():
            import sys
            print(f"[DEBUG] run_app线程开始: dual_mode={dual_mode}, backend_mode={backend_mode}, show_window={show_window}")
            sys.stdout.flush()
            try:
                if backend_mode:
                    # 后台服务模式（默认模式）
                    print("[DEBUG] 启动后台服务模式")
                    sys.stdout.flush()
                    from run_backend_service import BackendService
                    import argparse
                    parser = argparse.ArgumentParser()
                    parser.add_argument('--config', type=str, default='config.yaml')
                    parser.add_argument('--show-window', action='store_true', default=show_window)
                    args = parser.parse_args()
                    
                    service = BackendService(config_path=args.config, show_window=args.show_window)
                    service.run()
                elif dual_mode:
                    # 运行双摄像头模式（带窗口）
                    print("[DEBUG] 启动双摄像头模式")
                    sys.stdout.flush()
                    from run_dual_camera import DualCameraApp
                    import argparse
                    parser = argparse.ArgumentParser()
                    parser.add_argument('--config', type=str, default='config.yaml')
                    args = parser.parse_args()
                    
                    app = DualCameraApp(args.config)
                    app.run()
                else:
                    # 运行单摄像头模式
                    print("[DEBUG] 启动单摄像头模式")
                    sys.stdout.flush()
                    print("[DEBUG] 导入AIPoseMatchRVMDebug...")
                    sys.stdout.flush()
                    from run_with_rvm import AIPoseMatchRVMDebug
                    print("[DEBUG] 导入成功")
                    sys.stdout.flush()
                    import argparse
                    parser = argparse.ArgumentParser()
                    parser.add_argument('--config', type=str, default='config.yaml')
                    args = parser.parse_args()
                    
                    print(f"[DEBUG] 创建AIPoseMatchRVMDebug实例...")
                    sys.stdout.flush()
                    app = AIPoseMatchRVMDebug(args.config)
                    print("[DEBUG] 实例创建成功，调用app.run()")
                    sys.stdout.flush()
                    app.run()
            except Exception as e:
                print(f"[ERROR] 启动应用失败: {e}")
                sys.stdout.flush()
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
        
        print(f"[DEBUG] 创建线程...")
        self.app_thread = threading.Thread(target=run_app, daemon=True)
        self.app_thread.start()
        print(f"[DEBUG] 线程已启动，线程ID: {self.app_thread.ident}, 存活: {self.app_thread.is_alive()}")
        self.app_window = self.app_thread
    
    def open_single_window(self, icon, item):
        """打开单摄像头窗口"""
        print("[DEBUG] 收到打开单摄像头窗口请求")
        self.open_window(dual_mode=False)
    
    def open_dual_window(self, icon, item):
        """打开双摄像头窗口"""
        print("[DEBUG] 收到打开双摄像头窗口请求")
        self.open_window(dual_mode=True)
    
    def quit_app(self, icon, item):
        """退出应用"""
        icon.stop()
        os._exit(0)
    
    def run(self):
        """运行托盘应用"""
        print(f"[DEBUG] TrayApp.run()被调用，HAS_PYSTRAY = {HAS_PYSTRAY}")
        print(f"[DEBUG] HAS_PYSTRAY类型: {type(HAS_PYSTRAY)}")
        
        if not HAS_PYSTRAY:
            print("\n" + "="*70)
            print("[错误] 无法启动系统托盘，缺少pystray库")
            print("="*70)
            print("\n【原因分析】")
            print("  您可能使用了系统Python而不是虚拟环境")
            print(f"  当前Python路径: {sys.executable}")
            print(f"  当前工作目录: {os.getcwd()}")
            print("\n【解决方案 - 请选择其一】")
            print("\n方案1：使用虚拟环境（推荐）⭐")
            print("  .venv\\Scripts\\python.exe run_with_tray.py")
            print("\n方案2：使用启动脚本（最简单）⭐")
            print("  在PowerShell中运行: .\\start_tray.ps1")
            print("  或在CMD中运行: start_tray.bat")
            print("\n方案3：手动激活虚拟环境")
            print("  .venv\\Scripts\\activate")
            print("  python run_with_tray.py")
            print("\n方案4：在系统Python安装pystray（不推荐）")
            print("  pip install pystray")
            print("  然后直接运行: python run_with_tray.py")
            print("="*70)
            print("\n尝试继续运行（不使用托盘模式）...")
            # 直接启动应用窗口（后台服务模式）
            self.open_window(dual_mode=True, backend_mode=True, show_window=False)
            if self.app_thread:
                self.app_thread.join()
            return
        
        print("[DEBUG] HAS_PYSTRAY = True，继续创建托盘图标...")
        
        menu = pystray.Menu(
            pystray.MenuItem("启动后台服务（默认）", lambda: self.open_window(dual_mode=True, backend_mode=True, show_window=False)),
            pystray.MenuItem("启动后台服务（带调试窗口）", lambda: self.open_window(dual_mode=True, backend_mode=True, show_window=True)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("打开单摄像头窗口", self.open_single_window),
            pystray.MenuItem("打开双摄像头窗口", self.open_dual_window),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出", self.quit_app)
        )
        
        icon_image = self.create_icon()
        
        # 创建图标
        # pystray在Windows上：左键单击显示菜单，可以通过on_activated处理双击
        def on_activated(icon, item=None):
            """处理图标激活事件（双击）"""
            if self.app_window is None or not self.app_window.is_alive():
                # 默认打开单摄像头窗口
                self.open_window(dual_mode=False)
        
        self.tray_icon = pystray.Icon(
            "AI Pose Match",
            icon_image,
            "AI Pose Match - 姿态比对系统\n双击图标打开预览窗口\n右键显示菜单",
            menu
        )
        
        # 设置默认动作（双击）- 启动后台服务
        def on_activated_default(icon, item=None):
            """默认动作：启动后台服务（不显示窗口）"""
            if self.app_window is None or not self.app_window.is_alive():
                self.open_window(dual_mode=True, backend_mode=True, show_window=False)
        
        self.tray_icon.on_activated = on_activated_default
        
        # 运行托盘图标（阻塞调用，直到退出）
        print("[OK] 系统托盘图标已显示在任务栏")
        self.tray_icon.run()
    
    def stop(self):
        """停止托盘应用"""
        if self.tray_icon:
            self.tray_icon.stop()


def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Pose Match - 带系统托盘版本")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--no-tray', action='store_true',
                       help='不使用系统托盘，直接打开窗口')
    parser.add_argument('--dual', action='store_true',
                       help='启动双摄像头模式')
    args = parser.parse_args()
    
    if args.no_tray:
        # 不使用托盘，直接启动窗口
        if args.dual:
            from run_dual_camera import DualCameraApp
            app = DualCameraApp(args.config)
            app.run()
        else:
            from run_with_rvm import AIPoseMatchRVMDebug
            app = AIPoseMatchRVMDebug(args.config)
            app.run()
    else:
        # 使用系统托盘
        print("=" * 70)
        print("AI Pose Match - 系统托盘模式")
        print("=" * 70)
        print("\n程序正在启动...")
        print("启动后，请查看系统托盘（任务栏右下角）是否有图标。")
        print("\n操作说明：")
        print("  - 双击托盘图标：打开单摄像头预览窗口")
        print("  - 右键托盘图标：显示菜单")
        print("    - 打开单摄像头窗口")
        print("    - 打开双摄像头窗口")
        print("    - 退出")
        print("\n提示：程序在后台运行，关闭此窗口不会退出程序。")
        print("要退出程序，请右键托盘图标选择'退出'。")
        print("=" * 70)
        print("\n正在初始化系统托盘...\n")
        
        tray_app = TrayApp()
        
        # 默认启动后台服务（不显示窗口，推送Unity）
        # 如果指定了--dual，启动后台服务
        # 如果指定了--no-tray，直接运行（在main函数中处理）
        print("[INFO] 默认模式：后台服务（不显示窗口，推送Unity）")
        print("[INFO] 双击托盘图标或右键菜单可以打开调试窗口")
        tray_app.open_window(dual_mode=True, backend_mode=True, show_window=False)
        
        try:
            print("[INFO] 系统托盘已启动，程序在后台运行中...")
            print("[INFO] 请查看系统托盘图标（任务栏右下角）\n")
            tray_app.run()
        except KeyboardInterrupt:
            print("\n程序退出")
            tray_app.stop()


if __name__ == "__main__":
    main()

