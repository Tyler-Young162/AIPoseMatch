"""
AI Pose Match - 双摄像头比赛模式
支持同时打开两个摄像头进行姿态比对比赛
"""
import sys
import cv2
import numpy as np
from pathlib import Path
import time
import threading
from PIL import Image, ImageDraw, ImageFont

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from camera_manager import CameraManager
from pose_detector import PoseDetector
from person_selector import PersonSelector
from human_matting import HumanMatting
from visualizer import Visualizer
from pose_matcher import PoseMatcher


class DualCameraView:
    """
    单个摄像头视图处理器
    """
    def __init__(self, camera_id: int, config: Config, pose_matcher: PoseMatcher, window_name: str):
        """
        初始化单个摄像头视图
        
        Args:
            camera_id: 摄像头ID
            config: 配置对象
            pose_matcher: 共享的姿态比对器
            window_name: 窗口名称
        """
        self.camera_id = camera_id
        self.config = config
        self.pose_matcher = pose_matcher
        self.window_name = window_name
        
        # 创建摄像头配置副本并修改device_id
        from copy import deepcopy
        self.camera_config = deepcopy(config)
        self.camera_config.camera.device_id = camera_id
        
        # 初始化组件
        self.camera = CameraManager(self.camera_config)
        self.pose_detector = PoseDetector(self.camera_config)
        self.person_selector = PersonSelector(self.camera_config)
        self.matting = HumanMatting(self.camera_config)
        self.visualizer = Visualizer(self.camera_config)
        
        # 统计信息
        self.stats = {
            'frame_count': 0,
            'detection_count': 0,
            'avg_fps': 0.0,
            'last_time': time.time(),
            'fps_history': []
        }
        
        self.current_match_score = 0.0
        self.is_running = False
        
    def initialize(self) -> bool:
        """初始化摄像头和模块"""
        print(f"\n初始化摄像头 {self.camera_id}...")
        if not self.camera.initialize():
            print(f"[ERROR] 摄像头 {self.camera_id} 初始化失败")
            return False
        
        cam_info = self.camera.get_frame_info()
        print(f"[OK] 摄像头 {self.camera_id} 初始化成功")
        print(f"  分辨率: {cam_info['width']}x{cam_info['height']}")
        print(f"  帧率: {cam_info['fps']:.1f} FPS")
        
        # 初始化抠像模块
        matting_initialized = self.matting.initialize()
        if matting_initialized:
            print(f"[OK] 摄像头 {self.camera_id} 抠像模块初始化成功")
        
        return True
    
    def _put_chinese_text(self, img, text, position, font_size=20, color=(255, 255, 255)):
        """绘制中文文本"""
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_image)
            
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simsun.ttc",
                "C:/Windows/Fonts/simhei.ttf",
            ]
            font = None
            for font_path in font_paths:
                if Path(font_path).exists():
                    font = ImageFont.truetype(font_path, font_size)
                    break
            
            if font is None:
                font = ImageFont.load_default()
            
            rgb_color = (color[2], color[1], color[0])
            draw.text(position, text, font=font, fill=rgb_color)
            
            img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return img_bgr
        except Exception as e:
            print(f"Error drawing Chinese text: {e}")
            return img
    
    def _update_stats(self):
        """更新统计信息"""
        self.stats['frame_count'] += 1
        current_time = time.time()
        elapsed = current_time - self.stats['last_time']
        
        if elapsed >= 1.0:
            fps = self.stats['frame_count'] / elapsed
            self.stats['avg_fps'] = fps
            self.stats['fps_history'].append(fps)
            if len(self.stats['fps_history']) > 30:
                self.stats['fps_history'].pop(0)
            self.stats['frame_count'] = 0
            self.stats['last_time'] = current_time
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """处理单帧"""
        result = {}
        
        # 提取ROI
        roi_frame = self.camera.extract_roi(frame)
        result['roi'] = roi_frame
        
        # 姿态检测
        if roi_frame is not None:
            all_persons = self.pose_detector.detect(roi_frame)
            self.stats['detection_count'] += len(all_persons)
            result['persons'] = all_persons
            
            if all_persons:
                roi_center = self.person_selector.get_roi_center(self.config.roi)
                best_person = self.person_selector.select_best_person(all_persons, roi_center)
                result['best_person'] = best_person
                
                if best_person:
                    # 使用共享的pose_matcher进行比对
                    target_pose = self.pose_matcher.get_current_target_pose()
                    if target_pose is not None:
                        score, _ = self.pose_matcher.calculate_pose_similarity(
                            best_person['landmarks'], target_pose['landmarks']
                        )
                        self.current_match_score = score
                        result['match_score'] = score
                        
                        roi_with_pose = self.pose_matcher.draw_skeleton_with_matching(
                            roi_frame.copy(), best_person, target_pose
                        )
                    else:
                        roi_with_pose = self.pose_detector.draw_skeleton(roi_frame.copy(), [best_person])
                        self.current_match_score = 0.0
                    result['pose_frame'] = roi_with_pose
                else:
                    self.current_match_score = 0.0
                
                # 抠像
                if best_person:
                    bbox = best_person.get('bounding_box')
                    landmarks = best_person.get('landmarks')
                    self.matting._last_person_data = best_person
                    alpha_matte, matting_result = self.matting.process(roi_frame, bbox, None, landmarks)
                    result['alpha'] = alpha_matte
                    result['matting_result'] = matting_result
        
        return result
    
    def create_display(self, original_frame: np.ndarray, processed_data: dict) -> np.ndarray:
        """创建显示画面（只显示ROI区域，裁剪掉黑色部分）"""
        h, w = original_frame.shape[:2]
        roi_config = self.config.roi
        x_min = int(roi_config.x_min * w)
        x_max = int(roi_config.x_max * w)
        y_min = int(roi_config.y_min * h)
        y_max = int(roi_config.y_max * h)
        
        target_h = y_max - y_min
        target_w = x_max - x_min
        
        # 获取是否检测到人物
        best_person = processed_data.get('best_person')
        matting_result = processed_data.get('matting_result')
        alpha_matte = processed_data.get('alpha')
        
        # 如果没有检测到人物，显示透明背景（绿色）
        if best_person is None or matting_result is None:
            # 创建绿色背景（透明效果）
            display = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            display[:, :, 1] = 255  # 绿色通道 = 255
        else:
            # 有人物：应用抠像结果
            if matting_result.shape[:2] != (target_h, target_w):
                matting_resized = cv2.resize(matting_result, (target_w, target_h))
            else:
                matting_resized = matting_result.copy()
            display = matting_resized
        
        # 叠加骨骼
        best_person = processed_data.get('best_person')
        if best_person is not None:
            target_pose = self.pose_matcher.get_current_target_pose()
            if target_pose is not None:
                display = self.pose_matcher.draw_skeleton_with_matching(
                    display, best_person, target_pose
                )
            else:
                display = self.pose_detector.draw_skeleton(display, [best_person])
        
        # 添加状态信息（相对于ROI尺寸）
        display_h, display_w = display.shape[:2]
        fps = self.visualizer.calculate_fps()
        if fps > 0:
            display = self.visualizer.draw_fps(display, fps)
        
        display = self.visualizer.draw_status_info(
            display, best_person, len(processed_data.get('persons', []))
        )
        
        # 添加摄像头标识和匹配评分（相对于ROI尺寸）
        camera_label = f"摄像头 {self.camera_id}"
        display = self._put_chinese_text(display, camera_label, (10, display_h - 60),
                                        font_size=20, color=(0, 255, 255))
        
        target_pose = self.pose_matcher.get_current_target_pose()
        if target_pose is not None:
            pose_image = target_pose.get('image')
            if pose_image is not None:
                # 右上角显示目标姿态预览（使用display尺寸）
                preview_size = min(150, int(display_w * 0.25))  # 预览大小不超过窗口的25%
                preview_x = display_w - preview_size - 10
                preview_y = 10
                
                pose_h, pose_w = pose_image.shape[:2]
                aspect_ratio = pose_w / pose_h
                if aspect_ratio > 1:
                    preview_w = preview_size
                    preview_h = int(preview_size / aspect_ratio)
                else:
                    preview_h = preview_size
                    preview_w = int(preview_size * aspect_ratio)
                
                # 确保不超出边界
                if preview_x + preview_w > display_w:
                    preview_w = display_w - preview_x - 10
                    preview_h = int(preview_w / aspect_ratio)
                if preview_y + preview_h > display_h - 50:  # 留出空间显示文字
                    preview_h = display_h - preview_y - 50
                    preview_w = int(preview_h * aspect_ratio)
                
                pose_resized = cv2.resize(pose_image, (preview_w, preview_h))
                
                if preview_x + preview_w <= display_w and preview_y + preview_h <= display_h - 30:
                    overlay = display.copy()
                    cv2.rectangle(overlay,
                                (max(0, preview_x - 5), max(0, preview_y - 5)),
                                (min(display_w, preview_x + preview_w + 5), 
                                 min(display_h, preview_y + preview_h + 30)),
                                (0, 0, 0), -1)
                    display = cv2.addWeighted(display, 0.3, overlay, 0.7, 0)
                    
                    cv2.rectangle(display,
                                (preview_x - 2, preview_y - 2),
                                (preview_x + preview_w + 2, preview_y + preview_h + 2),
                                (255, 255, 255), 2)
                    
                    display[preview_y:preview_y + preview_h,
                           preview_x:preview_x + preview_w] = pose_resized
                    
                    pose_name = target_pose['name']
                    text_y = preview_y + preview_h + 15
                    if text_y < display_h - 10:
                        display = self._put_chinese_text(display, pose_name,
                                                        (preview_x, text_y),
                                                        font_size=12, color=(255, 255, 255))
            
            # 显示匹配评分（如果检测到人物）
            if best_person is not None:
                score = processed_data.get('match_score', self.current_match_score)
                score_text = f"匹配度: {score:.1f}分"
                score_x = max(10, display_w - 200)
                display = self._put_chinese_text(display, score_text,
                                                (score_x, 30),
                                                font_size=20, color=(0, 255, 255))
        
        return display
    
    def run(self):
        """运行单个摄像头视图循环"""
        self.is_running = True
        print(f"\n摄像头 {self.camera_id} 开始运行...\n")
        
        try:
            # 设置窗口位置（让两个窗口并排显示）
            # 摄像头0在左侧，摄像头1在右侧
            window_width = 0  # 将在第一次显示时设置
            window_height = 0
            
            while self.is_running:
                # 检查退出标志
                if not self.is_running:
                    break
                ret, frame = self.camera.read()
                if not ret:
                    print(f"摄像头 {self.camera_id} 读取帧失败，继续尝试...")
                    continue
                
                processed_data = self.process_frame(frame)
                combined_display = self.create_display(frame, processed_data)
                
                # 第一次显示时设置窗口位置和大小
                if window_width == 0:
                    display_h, display_w = combined_display.shape[:2]
                    window_width = display_w
                    window_height = display_h
                    
                    # 创建窗口并设置位置
                    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.window_name, window_width, window_height)
                    
                    # 计算窗口位置
                    screen_offset_x = 50  # 距离屏幕左边的距离
                    screen_offset_y = 50  # 距离屏幕上边的距离
                    window_spacing = 20   # 两个窗口之间的间距
                    
                    if self.camera_id == 0:
                        # 摄像头0在左侧
                        window_x = screen_offset_x
                    else:
                        # 摄像头1在右侧
                        window_x = screen_offset_x + window_width + window_spacing
                    
                    cv2.moveWindow(self.window_name, window_x, screen_offset_y)
                
                cv2.imshow(self.window_name, combined_display)
                
                # 检查窗口是否被用户关闭（通过窗口关闭按钮）
                try:
                    window_prop = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
                    if window_prop < 0:
                        # 窗口已被用户关闭
                        print(f"\n摄像头 {self.camera_id} 窗口被用户关闭")
                        self.is_running = False
                        break
                except:
                    # 窗口不存在，退出循环
                    self.is_running = False
                    break
                
                self._update_stats()
                
                # 处理键盘输入
                key_full = cv2.waitKey(1)
                
                if key_full == -1:
                    pass
                else:
                    key_code = key_full & 0xFF
                    
                    # ESC键或Q键退出
                    if key_code == ord('q') or key_code == ord('Q') or key_code == 27:
                        print(f"\n摄像头 {self.camera_id} 窗口关闭 (ESC/Q)")
                        self.is_running = False
                        break
                    
                    # 支持箭头键切换目标姿态（两个窗口同步）
                    if key_full & 0xFF == 0xFF:
                        extended_code = (key_full >> 8) & 0xFF
                        if extended_code == 37:  # Left arrow
                            self.pose_matcher.switch_to_previous_pose()
                        elif extended_code == 39:  # Right arrow
                            self.pose_matcher.switch_to_next_pose()
                    elif key_full == 81:  # Alternative left
                        self.pose_matcher.switch_to_previous_pose()
                    elif key_full == 83:  # Alternative right
                        self.pose_matcher.switch_to_next_pose()
                    elif key_code == ord('a') or key_code == ord('A'):
                        self.pose_matcher.switch_to_previous_pose()
                    elif key_code == ord('d') or key_code == ord('D'):
                        self.pose_matcher.switch_to_next_pose()
        
        except Exception as e:
            print(f"[ERROR] 摄像头 {self.camera_id} 运行错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                self.cleanup()
            except Exception as e:
                print(f"[ERROR] 清理摄像头 {self.camera_id} 时出错: {e}")
    
    def cleanup(self):
        """清理资源"""
        print(f"\n清理摄像头 {self.camera_id} 资源...")
        self.camera.release()
        self.pose_detector.cleanup()
        self.matting.cleanup()
        
        # 检查窗口是否存在再销毁，避免Null pointer错误
        try:
            # 检查窗口是否存在
            window_prop = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
            if window_prop >= 0:  # 窗口存在
                cv2.destroyWindow(self.window_name)
        except cv2.error:
            # 窗口不存在或已被销毁，忽略错误
            pass
        except Exception as e:
            # 其他异常也忽略
            print(f"[WARN] 销毁窗口时出现异常（可忽略）: {e}")


class DualCameraApp:
    """
    双摄像头应用程序
    """
    def __init__(self, config_path: str = "config.yaml"):
        """初始化双摄像头应用"""
        print("=" * 70)
        print("AI Pose Match - 双摄像头比赛模式")
        print("=" * 70)
        
        # 加载配置
        self.config = Config.load_from_yaml(config_path)
        
        # 初始化共享的姿态比对器
        print("\n初始化姿态比对模块...")
        self.pose_matcher = PoseMatcher(pose_folder="Pose")
        if len(self.pose_matcher.target_poses) > 0:
            print(f"[OK] 已加载 {len(self.pose_matcher.target_poses)} 个目标姿态")
        
        # 创建两个摄像头视图
        self.view1 = DualCameraView(0, self.config, self.pose_matcher, "AI Pose Match - 摄像头 0")
        self.view2 = DualCameraView(1, self.config, self.pose_matcher, "AI Pose Match - 摄像头 1")
        
        self.is_running = False
    
    def initialize(self) -> bool:
        """初始化所有摄像头"""
        print("\n初始化双摄像头...")
        
        if not self.view1.initialize():
            return False
        
        if not self.view2.initialize():
            self.view1.cleanup()
            return False
        
        print("\n" + "=" * 70)
        print("初始化完成！")
        print("=" * 70)
        print("\n键盘控制：")
        print("  ESC/Q    - 关闭当前窗口或退出程序")
        print("  ←/→ 或 A/D - 切换目标姿态（两个窗口同步）")
        print("=" * 70 + "\n")
        
        return True
    
    def run(self):
        """运行双摄像头应用"""
        if not self.initialize():
            print("[ERROR] 初始化失败")
            return
        
        self.is_running = True
        thread1 = None
        thread2 = None
        
        try:
            # 在两个独立线程中运行两个摄像头
            thread1 = threading.Thread(target=self.view1.run, daemon=False)
            thread2 = threading.Thread(target=self.view2.run, daemon=False)
            
            thread1.start()
            thread2.start()
            
            # 给窗口一些时间初始化
            time.sleep(0.5)
            
            # 主线程等待线程完成
            while self.is_running:
                # 检查线程是否还在运行
                if not thread1.is_alive() and not thread2.is_alive():
                    # 两个线程都已结束（用户关闭了窗口或按了ESC）
                    break
                
                # 检查窗口是否还存在（使用更宽松的检查，避免误判）
                # 注意：不频繁检查，避免干扰窗口显示
                try:
                    prop0 = cv2.getWindowProperty("AI Pose Match - 摄像头 0", cv2.WND_PROP_VISIBLE)
                    prop1 = cv2.getWindowProperty("AI Pose Match - 摄像头 1", cv2.WND_PROP_VISIBLE)
                    # 只有在窗口确实被关闭（返回-1）时才退出
                    # prop < 0 表示窗口不存在，prop == 0 可能只是最小化
                    if prop0 < 0 and prop1 < 0:
                        # 两个窗口都不存在了，设置标志让线程退出
                        self.is_running = False
                        self.view1.is_running = False
                        self.view2.is_running = False
                        break
                except cv2.error:
                    # 窗口不存在（getWindowProperty抛出异常）
                    # 检查线程是否还在运行，如果还在运行可能是暂时的错误
                    if not thread1.is_alive() and not thread2.is_alive():
                        # 线程已结束，说明窗口确实关闭了
                        break
                except Exception:
                    # 其他异常忽略，继续运行
                    pass
                
                time.sleep(0.5)  # 降低检查频率，避免干扰
            
            # 等待线程结束（最多等待2秒）
            if thread1 and thread1.is_alive():
                thread1.join(timeout=2.0)
            if thread2 and thread2.is_alive():
                thread2.join(timeout=2.0)
                
        except KeyboardInterrupt:
            print("\n程序被中断")
            self.is_running = False
            self.view1.is_running = False
            self.view2.is_running = False
        except Exception as e:
            print(f"[ERROR] 运行错误: {e}")
        finally:
            # 确保设置退出标志
            self.is_running = False
            self.view1.is_running = False
            self.view2.is_running = False
            time.sleep(0.2)  # 给线程一点时间响应退出信号
            self.cleanup()
    
    def cleanup(self):
        """清理所有资源"""
        print("\n清理所有资源...")
        self.is_running = False
        self.view1.is_running = False
        self.view2.is_running = False
        time.sleep(0.5)  # 给线程时间退出
        
        # 清理各个视图
        try:
            self.view1.cleanup()
        except Exception as e:
            print(f"[WARN] 清理摄像头0时出错: {e}")
        
        try:
            self.view2.cleanup()
        except Exception as e:
            print(f"[WARN] 清理摄像头1时出错: {e}")
        
        if hasattr(self, 'pose_matcher'):
            try:
                self.pose_matcher.cleanup()
            except Exception as e:
                print(f"[WARN] 清理姿态比对器时出错: {e}")
        
        # 尝试销毁所有窗口，忽略错误
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"[WARN] 销毁窗口时出错（可忽略）: {e}")
        
        print("清理完成")


def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Pose Match - 双摄像头比赛模式")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径 (默认: config.yaml)')
    args = parser.parse_args()
    
    app = DualCameraApp(args.config)
    app.run()


if __name__ == "__main__":
    main()

