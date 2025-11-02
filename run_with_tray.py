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

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_PYSTRAY = True
except ImportError:
    HAS_PYSTRAY = False
    print("[WARN] pystray未安装，无法使用系统托盘功能")
    print("      请运行: pip install pystray")


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
    
    def open_window(self, dual_mode=False):
        """打开应用窗口"""
        if self.app_window is not None and self.app_window.is_alive():
            print("应用窗口已在运行")
            return
        
        self.is_dual_mode = dual_mode
        
        def run_app():
            if dual_mode:
                # 运行双摄像头模式
                from run_dual_camera import DualCameraApp
                import argparse
                parser = argparse.ArgumentParser()
                parser.add_argument('--config', type=str, default='config.yaml')
                args = parser.parse_args()
                
                app = DualCameraApp(args.config)
                app.run()
            else:
                # 运行单摄像头模式
                from run_with_rvm import AIPoseMatchRVMDebug
                import argparse
                parser = argparse.ArgumentParser()
                parser.add_argument('--config', type=str, default='config.yaml')
                args = parser.parse_args()
                
                app = AIPoseMatchRVMDebug(args.config)
                app.run()
        
        self.app_thread = threading.Thread(target=run_app, daemon=True)
        self.app_thread.start()
        self.app_window = self.app_thread
    
    def open_single_window(self, icon, item):
        """打开单摄像头窗口"""
        self.open_window(dual_mode=False)
    
    def open_dual_window(self, icon, item):
        """打开双摄像头窗口"""
        self.open_window(dual_mode=True)
    
    def quit_app(self, icon, item):
        """退出应用"""
        icon.stop()
        os._exit(0)
    
    def run(self):
        """运行托盘应用"""
        if not HAS_PYSTRAY:
            print("[ERROR] 无法启动系统托盘，缺少pystray库")
            print("请运行: pip install pystray")
            # 直接启动应用窗口
            self.open_window()
            if self.app_thread:
                self.app_thread.join()
            return
        
        menu = pystray.Menu(
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
        
        # 设置默认动作（双击）
        self.tray_icon.on_activated = on_activated
        
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
        print("  • 双击托盘图标：打开单摄像头预览窗口")
        print("  • 右键托盘图标：显示菜单")
        print("    - 打开单摄像头窗口")
        print("    - 打开双摄像头窗口")
        print("    - 退出")
        print("\n提示：程序在后台运行，关闭此窗口不会退出程序。")
        print("要退出程序，请右键托盘图标选择'退出'。")
        print("=" * 70)
        print("\n正在初始化系统托盘...\n")
        
        tray_app = TrayApp()
        
        # 如果指定了双摄像头模式，自动打开双摄像头窗口
        if args.dual:
            print("[INFO] 检测到 --dual 参数，将自动打开双摄像头窗口...")
            tray_app.open_window(dual_mode=True)
        
        try:
            print("[INFO] 系统托盘已启动，程序在后台运行中...")
            print("[INFO] 请查看系统托盘图标（任务栏右下角）\n")
            tray_app.run()
        except KeyboardInterrupt:
            print("\n程序退出")
            tray_app.stop()


if __name__ == "__main__":
    main()

