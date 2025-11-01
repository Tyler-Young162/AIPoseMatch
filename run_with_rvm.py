"""
AI Pose Match - RVM增强版一键启动程序
带完整调试信息和键盘控制功能
"""
import sys
import cv2
import numpy as np
from pathlib import Path
import time
from PIL import Image, ImageDraw, ImageFont

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from camera_manager import CameraManager
from pose_detector import PoseDetector
from person_selector import PersonSelector
from human_matting import HumanMatting
from visualizer import Visualizer


class AIPoseMatchRVMDebug:
    """
    AI Pose Match with RVM - 带调试功能的增强版
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize application."""
        print("=" * 70)
        print("AI Pose Match - RVM增强版启动程序")
        print("=" * 70)
        
        # Load configuration
        print("\n[1/7] 加载配置文件...")
        try:
            self.config = Config.load_from_yaml(config_path)
            print(f"[OK] 配置文件加载成功: {config_path}")
        except Exception as e:
            print(f"[ERROR] 配置文件加载失败: {e}")
            sys.exit(1)
        
        # Check CUDA
        print("\n[2/7] 检查GPU/CUDA状态...")
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            print(f"[OK] CUDA可用")
            print(f"  GPU设备: {device_name}")
            print(f"  设备数量: {device_count}")
        else:
            print("[WARN] CUDA不可用，将使用CPU（速度较慢）")
        
        # Initialize components
        print("\n[3/7] 初始化摄像头...")
        self.camera = CameraManager(self.config)
        if not self.camera.initialize():
            print("[ERROR] 摄像头初始化失败")
            sys.exit(1)
        
        # Get camera info
        cam_info = self.camera.get_frame_info()
        print(f"[OK] 摄像头初始化成功")
        print(f"  分辨率: {cam_info['width']}x{cam_info['height']}")
        print(f"  帧率: {cam_info['fps']:.1f} FPS")
        
        print("\n[4/7] 初始化骨骼检测器...")
        self.pose_detector = PoseDetector(self.config)
        print("[OK] 骨骼检测器初始化成功")
        
        print("\n[5/7] 初始化人物选择器...")
        self.person_selector = PersonSelector(self.config)
        print("[OK] 人物选择器初始化成功")
        
        print("\n[6/7] 初始化人体抠像模块...")
        self.matting = HumanMatting(self.config)
        matting_initialized = self.matting.initialize()
        if matting_initialized:
            if self.matting.model_type == "rvm_real":
                print("[OK] RVM模型加载成功（高质量抠像）")
            else:
                print("[OK] 简化版抠像初始化成功")
        else:
            print("[WARN] 抠像模块初始化失败，使用简化方案")
        
        print("\n[7/7] 初始化可视化模块...")
        self.visualizer = Visualizer(self.config)
        print("[OK] 可视化模块初始化成功")
        
        # Debug mode flags
        self.debug_flags = {
            'show_camera_full': False,      # 1 - 显示完整摄像头画面
            'show_roi': True,               # 2 - 显示裁剪后的ROI
            'show_pose_debug': False,       # 3 - 显示骨骼检测调试信息
            'show_matting_debug': False,    # 4 - 显示抠像调试信息
            'show_all': False,              # 5 - 显示所有调试信息
            'show_fps': True,               # F - 切换FPS显示
            'show_hints': True,             # H - 切换提示信息显示
        }
        
        # Camera switching
        self.current_camera_id = self.config.camera.device_id
        self.max_camera_tries = 5  # Try up to 5 cameras
        
        # Statistics
        self.stats = {
            'frame_count': 0,
            'detection_count': 0,
            'avg_fps': 0.0,
            'last_time': time.time(),
            'fps_history': []
        }
        
        print("\n" + "=" * 70)
        print("初始化完成！")
        print("=" * 70)
        self._print_controls()
    
    def _print_controls(self):
        """Print keyboard controls."""
        print("\n" + "=" * 70)
        print("键盘控制说明")
        print("=" * 70)
        print("数字键:")
        print("1          - 切换显示完整摄像头画面")
        print("2          - 切换显示裁剪后的ROI区域")
        print("3          - 切换骨骼检测调试信息")
        print("4          - 切换抠像模块调试信息")
        print("5          - 切换所有调试信息（开/关）")
        print("C / c      - 切换摄像头")
        print("F / f      - 切换FPS显示")
        print("H / h      - 切换提示信息显示")
        print("S / s      - 保存当前帧到文件")
        print("R / r      - 重置统计信息")
        print("Q / q      - 退出程序")
        print("=" * 70)
        print("\n提示：按数字键查看不同模式，按'C'切换摄像头，按'Q'退出")
        print("=" * 70 + "\n")
    
    def _get_debug_display(self, key: str):
        """Get debug display mode for a key."""
        modes = {
            '1': 'show_camera_full',
            '2': 'show_roi',
            '3': 'show_pose_debug',
            '4': 'show_matting_debug',
            '5': 'show_all',
            'f': 'show_fps',
            'h': 'show_hints',
        }
        return modes.get(key.lower())
    
    def _put_chinese_text(self, img, text, position, font_size=20, color=(255, 255, 255)):
        """
        Draw Chinese text on image using PIL.
        
        Args:
            img: BGR image (numpy array)
            text: Chinese text to draw
            position: (x, y) tuple
            font_size: Font size
            color: BGR color tuple
            
        Returns:
            Image with Chinese text drawn
        """
        try:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_image)
            
            # Try to load Chinese font, fallback to default if not available
            try:
                # Try to use system Chinese fonts
                font_paths = [
                    "C:/Windows/Fonts/msyh.ttc",  # Microsoft YaHei
                    "C:/Windows/Fonts/simsun.ttc",  # SimSun
                    "C:/Windows/Fonts/simhei.ttf",  # SimHei
                ]
                font = None
                for font_path in font_paths:
                    if Path(font_path).exists():
                        font = ImageFont.truetype(font_path, font_size)
                        break
                
                if font is None:
                    # Fallback to default font
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # Draw text
            # Convert BGR to RGB for PIL
            rgb_color = (color[2], color[1], color[0])
            draw.text(position, text, font=font, fill=rgb_color)
            
            # Convert back to BGR
            img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return img_bgr
            
        except Exception as e:
            # If PIL fails, fallback to OpenCV (will show ?)
            print(f"Error drawing Chinese text: {e}")
            return img
    
    def _switch_camera(self):
        """Switch to next available camera."""
        old_id = self.current_camera_id
        
        # Try next cameras
        for i in range(self.max_camera_tries):
            self.current_camera_id = (self.current_camera_id + 1) % self.max_camera_tries
            
            # Try to initialize new camera (release is handled inside initialize)
            self.config.camera.device_id = self.current_camera_id
            if self.camera.initialize():
                print(f"[OK] 切换到摄像头 {self.current_camera_id}")
                cam_info = self.camera.get_frame_info()
                print(f"  分辨率: {cam_info['width']}x{cam_info['height']}")
                return True
        
        # If all failed, revert
        print(f"[ERROR] 未找到可用的摄像头，保持使用摄像头 {old_id}")
        self.current_camera_id = old_id
        self.config.camera.device_id = old_id
        if self.camera.initialize():
            print("已恢复原摄像头")
        return False
    
    def _draw_status_overlay(self, frame: np.ndarray, title: str = "") -> np.ndarray:
        """Draw status information overlay on frame."""
        if not self.debug_flags['show_hints']:
            return frame
        
        overlay = frame.copy()
        height, width = overlay.shape[:2]
        
        # Draw semi-transparent background for text
        overlay_box = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(overlay_box, (10, 10), (500, 200), (0, 0, 0), -1)
        overlay = cv2.addWeighted(overlay, 1.0, overlay_box, 0.5, 0)
        
        # Current mode indicator using Chinese font
        y_offset = 35
        
        if title:
            overlay = self._put_chinese_text(overlay, title, (20, y_offset), font_size=24, color=(0, 255, 255))
            y_offset += 40
        
        # Active debug modes
        modes = []
        if self.debug_flags['show_camera_full']:
            modes.append("[1]完整画面")
        if self.debug_flags['show_roi']:
            modes.append("[2]ROI")
        if self.debug_flags['show_pose_debug']:
            modes.append("[3]骨骼")
        if self.debug_flags['show_matting_debug']:
            modes.append("[4]抠像")
        if self.debug_flags['show_all']:
            modes.append("[5]全部")
        
        mode_text = " | ".join(modes) if modes else "[默认模式]"
        overlay = self._put_chinese_text(overlay, f"模式: {mode_text}", (20, y_offset), 
                                        font_size=18, color=(255, 255, 255))
        y_offset += 30
        
        # Camera info
        overlay = self._put_chinese_text(overlay, f"摄像头: {self.current_camera_id}", (20, y_offset),
                                        font_size=18, color=(0, 255, 0))
        y_offset += 30
        
        # Matting model
        model_type = "RVM" if self.matting.model_type == "rvm_real" else "简化版"
        overlay = self._put_chinese_text(overlay, f"抠像: {model_type}", (20, y_offset),
                                        font_size=18, color=(0, 255, 0))
        y_offset += 30
        
        # FPS
        if self.debug_flags['show_fps']:
            overlay = self._put_chinese_text(overlay, f"FPS: {self.stats['avg_fps']:.1f}", (20, y_offset),
                                            font_size=18, color=(0, 255, 0))
        
        return overlay
    
    def _update_stats(self):
        """Update statistics."""
        self.stats['frame_count'] += 1
        current_time = time.time()
        elapsed = current_time - self.stats['last_time']
        
        if elapsed >= 1.0:  # Update every second
            fps = self.stats['frame_count'] / elapsed
            self.stats['avg_fps'] = fps
            self.stats['fps_history'].append(fps)
            if len(self.stats['fps_history']) > 30:  # Keep last 30 seconds
                self.stats['fps_history'].pop(0)
            self.stats['frame_count'] = 0
            self.stats['last_time'] = current_time
    
    def _print_debug_info(self, frame_type: str, data: dict):
        """Print debug information based on display type."""
        if frame_type == "pose" and self.debug_flags['show_pose_debug']:
            print(f"\n--- 骨骼检测调试信息 ---")
            if data.get('persons'):
                print(f"检测到人数: {len(data['persons'])}")
                for i, person in enumerate(data['persons']):
                    print(f"  人物 {i+1}:")
                    print(f"    可见关键点: {person.get('num_visible_keypoints', 0)}/33")
                    print(f"    完整性: {person.get('completeness', 0):.2%}")
                    print(f"    高度: {person.get('height', 0)} pixels")
                    bbox = person.get('bounding_box', ())
                    print(f"    边界框: {bbox}")
                
                if data.get('best_person'):
                    best = data['best_person']
                    scores = best.get('_selection_scores', {})
                    print(f"\n最佳人物:")
                    print(f"  总评分: {scores.get('total', 0):.3f}")
                    print(f"  完整性: {scores.get('completeness', 0):.2f}")
                    print(f"  高度: {scores.get('height', 0):.2f}")
                    print(f"  居中度: {scores.get('centeredness', 0):.2f}")
            else:
                print("未检测到人物")
            print("--- 调试信息结束 ---\n")
        
        elif frame_type == "matting" and self.debug_flags['show_matting_debug']:
            print(f"\n--- 抠像模块调试信息 ---")
            print(f"模型类型: {self.matting.model_type}")
            print(f"设备: {self.matting.device}")
            print(f"降采样比例: {self.matting.matting_config.downsample_ratio}")
            print(f"过滤不完全: {self.matting.matting_config.filter_incomplete}")
            if data.get('alpha_shape'):
                print(f"Alpha通道形状: {data['alpha_shape']}")
            if data.get('processing_time'):
                print(f"处理时间: {data['processing_time']*1000:.2f} ms")
            print("--- 调试信息结束 ---\n")
        
        elif self.debug_flags['show_all']:
            # Print all debug info
            if data:
                print(f"\n--- 综合调试信息 (帧 #{self.stats['frame_count']}) ---")
                if 'persons' in data:
                    print(f"检测到人数: {len(data.get('persons', []))}")
                if 'best_person' in data:
                    print(f"最佳人物已选择: {bool(data['best_person'])}")
                print(f"平均FPS: {self.stats['avg_fps']:.1f}")
                print("--- 调试信息结束 ---\n")
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """Process a single frame with debug info."""
        result = {}
        
        # Draw ROI overlay on original
        if self.debug_flags['show_camera_full'] or self.debug_flags['show_all']:
            original_with_roi = self.camera.draw_roi_overlay(frame.copy())
        else:
            original_with_roi = frame.copy()
        
        # Extract ROI
        roi_frame = self.camera.extract_roi(frame)
        result['roi'] = roi_frame
        
        # Pose detection
        if roi_frame is not None:
            detection_start = time.time()
            all_persons = self.pose_detector.detect(roi_frame)
            detection_time = time.time() - detection_start
            self.stats['detection_count'] += len(all_persons)
            
            result['persons'] = all_persons
            
            # Person selection
            if all_persons:
                roi_center = self.person_selector.get_roi_center(self.config.roi)
                best_person = self.person_selector.select_best_person(all_persons, roi_center)
                result['best_person'] = best_person
                
                # Pose visualization
                if self.debug_flags['show_pose_debug'] or self.debug_flags['show_all']:
                    roi_with_pose = self.pose_detector.draw_skeleton(roi_frame.copy(), [best_person])
                    result['pose_frame'] = roi_with_pose
                
                # Matting
                if best_person:
                    matting_start = time.time()
                    bbox = best_person.get('bounding_box')
                    alpha_matte, matting_result = self.matting.process(roi_frame, bbox)
                    matting_time = time.time() - matting_start
                    
                    result['alpha'] = alpha_matte
                    result['matting_result'] = matting_result
                    result['alpha_shape'] = alpha_matte.shape if alpha_matte is not None else None
                    result['processing_time'] = matting_time
        
        # Print debug info
        self._print_debug_info("pose", result)
        self._print_debug_info("matting", result)
        
        return result
    
    def create_display(self, original_frame: np.ndarray, processed_data: dict) -> np.ndarray:
        """Create display based on debug flags."""
        displays = []
        
        # Mode 1: Full camera view
        if self.debug_flags['show_camera_full'] or self.debug_flags['show_all']:
            disp = original_frame.copy()
            disp = self._draw_status_overlay(disp, "完整摄像头画面")
            displays.append(("完整摄像头画面", disp))
        
        # Mode 2: ROI region
        if self.debug_flags['show_roi'] or self.debug_flags['show_all']:
            roi = processed_data.get('roi')
            if roi is not None:
                disp = roi.copy()
                disp = self._draw_status_overlay(disp, "ROI区域")
                displays.append(("ROI区域", disp))
        
        # Mode 3: Pose detection
        if self.debug_flags['show_pose_debug'] or self.debug_flags['show_all']:
            pose_frame = processed_data.get('pose_frame')
            if pose_frame is not None:
                disp = pose_frame.copy()
                disp = self._draw_status_overlay(disp, "骨骼检测")
                displays.append(("骨骼检测", disp))
        
        # Mode 4: Matting
        if self.debug_flags['show_matting_debug'] or self.debug_flags['show_all']:
            matting_result = processed_data.get('matting_result')
            if matting_result is not None:
                disp = matting_result.copy()
                disp = self._draw_status_overlay(disp, "抠像结果")
                displays.append(("抠像结果", disp))
        
        if not displays:
            # Default: show everything in grid
            combined = self.visualizer.visualize_complete(
                original_frame,
                processed_data.get('roi'),
                processed_data.get('best_person'),
                processed_data.get('alpha'),
                processed_data.get('matting_result'),
                len(processed_data.get('persons', []))
            )
            combined = self._draw_status_overlay(combined, "AI Pose Match")
            return combined
        
        # Show individual displays
        for title, disp in displays:
            cv2.imshow(title, disp)
        
        return None
    
    def run(self):
        """Main application loop."""
        print("\n开始运行...\n")
        
        try:
            while True:
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    print("读取帧失败，继续尝试...")
                    continue
                
                # Process frame
                processed_data = self.process_frame(frame)
                
                # Create display
                combined_display = self.create_display(frame, processed_data)
                if combined_display is not None:
                    cv2.imshow("AI Pose Match - RVM增强版", combined_display)
                
                # Update stats
                self._update_stats()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                # Check for exit (only user-initiated)
                if key == ord('q') or key == ord('Q') or key == 27:  # ESC key also exits
                    print("\n用户退出请求")
                    break
                
                # Handle other key presses
                if key == ord('1'):
                    self.debug_flags['show_camera_full'] = not self.debug_flags['show_camera_full']
                    print(f"完整摄像头画面: {'开启' if self.debug_flags['show_camera_full'] else '关闭'}")
                
                elif key == ord('2'):
                    self.debug_flags['show_roi'] = not self.debug_flags['show_roi']
                    print(f"ROI区域显示: {'开启' if self.debug_flags['show_roi'] else '关闭'}")
                
                elif key == ord('3'):
                    self.debug_flags['show_pose_debug'] = not self.debug_flags['show_pose_debug']
                    print(f"骨骼检测调试: {'开启' if self.debug_flags['show_pose_debug'] else '关闭'}")
                
                elif key == ord('4'):
                    self.debug_flags['show_matting_debug'] = not self.debug_flags['show_matting_debug']
                    print(f"抠像模块调试: {'开启' if self.debug_flags['show_matting_debug'] else '关闭'}")
                
                elif key == ord('5'):
                    self.debug_flags['show_all'] = not self.debug_flags['show_all']
                    print(f"所有调试信息: {'开启' if self.debug_flags['show_all'] else '关闭'}")
                
                elif key == ord('f') or key == ord('F'):
                    self.debug_flags['show_fps'] = not self.debug_flags['show_fps']
                    print(f"FPS显示: {'开启' if self.debug_flags['show_fps'] else '关闭'}")
                
                elif key == ord('h') or key == ord('H'):
                    self.debug_flags['show_hints'] = not self.debug_flags['show_hints']
                    print(f"提示信息显示: {'开启' if self.debug_flags['show_hints'] else '关闭'}")
                
                elif key == ord('c') or key == ord('C'):
                    self._switch_camera()
                
                elif key == ord('s') or key == ord('S'):
                    output_path = f"debug_frame_{self.stats['frame_count']:05d}.jpg"
                    if combined_display is not None:
                        cv2.imwrite(output_path, combined_display)
                    else:
                        # Save ROI if available
                        if processed_data.get('roi') is not None:
                            cv2.imwrite(output_path, processed_data['roi'])
                    print(f"已保存帧到: {output_path}")
                
                elif key == ord('r') or key == ord('R'):
                    self.stats = {
                        'frame_count': 0,
                        'detection_count': 0,
                        'avg_fps': 0.0,
                        'last_time': time.time(),
                        'fps_history': []
                    }
                    print("统计信息已重置")
        
        except KeyboardInterrupt:
            print("\n程序被中断")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        print("\n清理资源...")
        self.camera.release()
        self.pose_detector.cleanup()
        self.matting.cleanup()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print("程序统计信息")
        print("=" * 70)
        if len(self.stats['fps_history']) > 0:
            avg_fps = sum(self.stats['fps_history']) / len(self.stats['fps_history'])
            print(f"平均FPS: {avg_fps:.2f}")
            print(f"最高FPS: {max(self.stats['fps_history']):.2f}")
            print(f"最低FPS: {min(self.stats['fps_history']):.2f}")
        print(f"总检测次数: {self.stats['detection_count']}")
        print("=" * 70)
        print("程序已退出")


def main():
    """Application entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Pose Match - RVM增强版")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径 (默认: config.yaml)')
    parser.add_argument('--rvm', action='store_true',
                       help='强制使用RVM模式')
    args = parser.parse_args()
    
    # Check if RVM models exist
    if args.rvm:
        model_paths = [
            "models/rvm_mobilenetv3.pth",
            "models/rvm_resnet50.pth"
        ]
        models_exist = any(Path(p).exists() for p in model_paths)
        if not models_exist:
            print("警告: 未找到RVM模型文件")
            print("请运行: python download_rvm_model.py")
            print("或使用 --no-rvm 参数运行简化版本")
            response = input("是否继续使用简化版本? (y/n): ")
            if response.lower() != 'y':
                return
    
    # Create and run application
    app = AIPoseMatchRVMDebug(args.config)
    app.run()


if __name__ == "__main__":
    main()

