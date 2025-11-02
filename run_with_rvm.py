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
from pose_matcher import PoseMatcher


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
        
        print("\n[7/8] 初始化可视化模块...")
        self.visualizer = Visualizer(self.config)
        print("[OK] 可视化模块初始化成功")
        
        print("\n[8/8] 初始化姿态比对模块...")
        self.pose_matcher = PoseMatcher(pose_folder="Pose")
        if len(self.pose_matcher.target_poses) > 0:
            print(f"[OK] 姿态比对模块初始化成功，已加载 {len(self.pose_matcher.target_poses)} 个目标姿态")
        else:
            print("[WARN] 姿态比对模块初始化成功，但未找到目标姿态图片")
        
        # Display mode flags - 默认234都开启
        self.display_flags = {
            'show_roi_crop': True,          # 2 - 显示裁剪区域（无效区域黑色）
            'show_skeleton': True,          # 3 - 显示骨骼信息
            'show_matting': True,           # 4 - 显示抠像效果
            'show_fps': True,               # F - 切换FPS显示
            'show_hints': True,             # H - 切换提示信息显示
        }
        
        # Legacy debug flags (for compatibility)
        self.debug_flags = {
            'show_camera_full': False,
            'show_roi': True,
            'show_pose_debug': True,
            'show_matting_debug': True,
            'show_all': False,
            'show_fps': True,
            'show_hints': True,
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
        
        # Pose matching score
        self.current_match_score = 0.0
        
        print("\n" + "=" * 70)
        print("初始化完成！")
        print("=" * 70)
        self._print_controls()
    
    def _print_controls(self):
        """Print keyboard controls."""
        print("\n" + "=" * 70)
        print("键盘控制说明")
        print("=" * 70)
        print("显示模式（合并窗口）:")
        print("2          - 切换ROI裁剪显示（无效区域黑色）")
        print("3          - 切换骨骼信息显示")
        print("4          - 切换抠像效果显示（绿色背景）")
        print("其他控制:")
        print("1          - 切换显示完整摄像头画面（调试用）")
        print("5          - 切换所有调试信息（开/关）")
        print("C / c      - 切换摄像头")
        print("F / f      - 切换FPS显示")
        print("H / h      - 切换提示信息显示")
        print("← / →     - 切换目标姿态（左右箭头键）")
        print("A / D      - 切换目标姿态（左/右，备用方案）")
        print("S / s      - 保存当前帧到文件")
        print("R / r      - 重置统计信息")
        print("Q / q      - 退出程序")
        print("=" * 70)
        print("\n默认模式：ROI裁剪 + 骨骼 + 抠像（234都开启）")
        print("提示：按2/3/4键切换各功能显示，按'Q'退出")
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
                
                # Always prepare pose frame for potential display with matching
                if best_person:
                    # Use pose matcher to draw skeleton with color coding
                    target_pose = self.pose_matcher.get_current_target_pose()
                    if target_pose is not None:
                        # Calculate matching score
                        score, _ = self.pose_matcher.calculate_pose_similarity(
                            best_person['landmarks'], target_pose['landmarks']
                        )
                        self.current_match_score = score
                        result['match_score'] = score
                        
                        # Draw skeleton with color-coded matching
                        roi_with_pose = self.pose_matcher.draw_skeleton_with_matching(
                            roi_frame.copy(), best_person, target_pose
                        )
                    else:
                        # No target pose, use default skeleton drawing
                        roi_with_pose = self.pose_detector.draw_skeleton(roi_frame.copy(), [best_person])
                        self.current_match_score = 0.0
                    result['pose_frame'] = roi_with_pose
                else:
                    self.current_match_score = 0.0
                
                # Matting
                if best_person:
                    matting_start = time.time()
                    bbox = best_person.get('bounding_box')
                    landmarks = best_person.get('landmarks')
                    
                    # Store person data for matting module to access
                    self.matting._last_person_data = best_person
                    
                    alpha_matte, matting_result = self.matting.process(roi_frame, bbox, None, landmarks)
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
        """
        创建合并的单一显示窗口
        根据2、3、4的开关状态组合显示效果
        
        默认：ROI裁剪(2) + 骨骼(3) + 抠像(4) 都开启
        - 模式2开启：显示ROI区域，无效区域为黑色
        - 模式3开启：叠加骨骼信息
        - 模式4开启：应用抠像效果（绿色背景）
        """
        h, w = original_frame.shape[:2]
        
        roi_frame = processed_data.get('roi')
        best_person = processed_data.get('best_person')
        alpha_matte = processed_data.get('alpha')
        matting_result = processed_data.get('matting_result')
        
        # 计算ROI在原图中的位置
        roi_config = self.config.roi
        x_min = int(roi_config.x_min * w)
        x_max = int(roi_config.x_max * w)
        y_min = int(roi_config.y_min * h)
        y_max = int(roi_config.y_max * h)
        
        # 确定基础显示内容
        if self.display_flags['show_roi_crop'] and roi_frame is not None:
            # 模式2开启：显示ROI区域，无效区域为黑色
            display = np.zeros((h, w, 3), dtype=np.uint8)
            
            # 调整ROI大小以匹配实际位置
            target_h = y_max - y_min
            target_w = x_max - x_min
            
            if roi_frame.shape[:2] != (target_h, target_w):
                roi_resized = cv2.resize(roi_frame, (target_w, target_h))
            else:
                roi_resized = roi_frame.copy()
            
            # 放置ROI到对应位置
            display[y_min:y_max, x_min:x_max] = roi_resized
            
            # 当前工作区域是ROI区域
            working_area = display[y_min:y_max, x_min:x_max].copy()
            working_h, working_w = working_area.shape[:2]
        else:
            # 模式2关闭：显示完整原始画面
            display = original_frame.copy()
            working_area = display.copy()
            working_h, working_w = h, w
            # 注意：x_min, y_min, x_max, y_max 仍保留用于坐标转换
        
        # 应用抠像效果（模式4）
        if self.display_flags['show_matting'] and matting_result is not None:
            # matting_result已经是合成后的结果（人物原色+绿色背景）
            # 它基于ROI frame，所以需要调整大小以匹配ROI区域
            target_roi_h = y_max - y_min
            target_roi_w = x_max - x_min
            
            if matting_result.shape[:2] != (target_roi_h, target_roi_w):
                matting_resized = cv2.resize(matting_result, (target_roi_w, target_roi_h))
            else:
                matting_resized = matting_result.copy()
            
            if self.display_flags['show_roi_crop']:
                # ROI模式下，直接替换ROI区域
                display[y_min:y_max, x_min:x_max] = matting_resized
            else:
                # 完整画面模式下，将抠像效果叠加到ROI区域
                # 创建alpha mask用于混合
                if alpha_matte is not None:
                    # 使用alpha matte进行混合
                    alpha_resized = cv2.resize(alpha_matte, (target_roi_w, target_roi_h))
                    if alpha_resized.ndim == 2:
                        alpha_3ch = np.expand_dims(alpha_resized, axis=2)
                        alpha_3ch = np.repeat(alpha_3ch, 3, axis=2) / 255.0
                    else:
                        alpha_3ch = alpha_resized / 255.0
                    
                    # 混合：原图 * (1-alpha) + 抠像 * alpha
                    roi_original = display[y_min:y_max, x_min:x_max].astype(np.float32)
                    matting_float = matting_resized.astype(np.float32)
                    blended = (roi_original * (1 - alpha_3ch) + matting_float * alpha_3ch).astype(np.uint8)
                    display[y_min:y_max, x_min:x_max] = blended
                else:
                    # 没有alpha，直接替换
                    display[y_min:y_max, x_min:x_max] = matting_resized
        
        # 叠加骨骼信息（模式3）
        # 如果同时开启了抠像和骨骼，在抠像基础上绘制骨骼，而不是替换
        pose_frame = processed_data.get('pose_frame')
        if self.display_flags['show_skeleton'] and best_person is not None:
            if self.display_flags['show_roi_crop']:
                # ROI模式下，在现有的display基础上绘制骨骼（如果已经应用了抠像，会叠加在抠像上）
                roi_section = display[y_min:y_max, x_min:x_max].copy()
                target_pose = self.pose_matcher.get_current_target_pose()
                if target_pose is not None:
                    roi_with_skeleton = self.pose_matcher.draw_skeleton_with_matching(
                        roi_section, best_person, target_pose
                    )
                else:
                    roi_with_skeleton = self.pose_detector.draw_skeleton(roi_section, [best_person])
                display[y_min:y_max, x_min:x_max] = roi_with_skeleton
            else:
                # 完整画面模式：需要将landmarks从ROI坐标系转换到完整画面坐标系
                # landmarks是归一化的，相对于ROI frame（0-1相对于ROI）
                # 需要转换为相对于完整画面的归一化坐标
                roi_w_ratio = (x_max - x_min) / w
                roi_h_ratio = (y_max - y_min) / h
                
                # 创建转换后的person数据
                converted_person = best_person.copy()
                converted_landmarks = best_person['landmarks'].copy()
                
                # 转换每个landmark的坐标：从ROI归一化坐标 -> 完整画面归一化坐标
                for i in range(len(converted_landmarks)):
                    # 原坐标是相对于ROI的（0-1）
                    roi_x = converted_landmarks[i][0]
                    roi_y = converted_landmarks[i][1]
                    
                    # 转换为完整画面的归一化坐标
                    full_x = (x_min / w) + (roi_x * roi_w_ratio)
                    full_y = (y_min / h) + (roi_y * roi_h_ratio)
                    
                    converted_landmarks[i][0] = full_x
                    converted_landmarks[i][1] = full_y
                
                converted_person['landmarks'] = converted_landmarks
                
                # 在完整画面上绘制骨骼（使用匹配颜色）
                target_pose = self.pose_matcher.get_current_target_pose()
                if target_pose is not None:
                    display = self.pose_matcher.draw_skeleton_with_matching(
                        display, converted_person, target_pose
                    )
                else:
                    display = self.pose_detector.draw_skeleton(display, [converted_person])
        
        # 添加状态信息
        fps = self.visualizer.calculate_fps()
        if fps > 0 and self.display_flags['show_fps']:
            display = self.visualizer.draw_fps(display, fps)
        
        display = self.visualizer.draw_status_info(
            display,
            best_person,
            len(processed_data.get('persons', []))
        )
        
        # 添加匹配评分和目标姿态信息
        target_pose = self.pose_matcher.get_current_target_pose()
        if target_pose is not None:
            # 在右上角显示目标姿态图片预览
            pose_image = target_pose.get('image')
            if pose_image is not None:
                # 计算预览区域大小（右上角）
                preview_size = 180  # 预览图片大小（稍微小一点，避免遮挡）
                preview_x = w - preview_size - 10
                preview_y = 10
                
                # 调整图片大小（保持宽高比）
                pose_h, pose_w = pose_image.shape[:2]
                aspect_ratio = pose_w / pose_h
                if aspect_ratio > 1:
                    # 横向图片
                    preview_w = preview_size
                    preview_h = int(preview_size / aspect_ratio)
                else:
                    # 竖向图片
                    preview_h = preview_size
                    preview_w = int(preview_size * aspect_ratio)
                
                # 确保预览区域不超过屏幕
                if preview_x + preview_w > w:
                    preview_w = w - preview_x - 10
                    preview_h = int(preview_w / aspect_ratio)
                if preview_y + preview_h > h:
                    preview_h = h - preview_y - 80  # 留出空间显示文字
                    preview_w = int(preview_h * aspect_ratio)
                
                pose_resized = cv2.resize(pose_image, (preview_w, preview_h))
                
                # 在预览图片位置添加半透明背景（避免完全覆盖）
                overlay = display.copy()
                cv2.rectangle(overlay, 
                            (preview_x - 5, preview_y - 5),
                            (preview_x + preview_w + 5, preview_y + preview_h + 35),
                            (0, 0, 0), -1)
                display = cv2.addWeighted(display, 0.3, overlay, 0.7, 0)
                
                # 在预览图片周围绘制白色边框
                cv2.rectangle(display, 
                            (preview_x - 2, preview_y - 2),
                            (preview_x + preview_w + 2, preview_y + preview_h + 2),
                            (255, 255, 255), 2)
                
                # 放置预览图片
                display[preview_y:preview_y + preview_h, 
                       preview_x:preview_x + preview_w] = pose_resized
                
                # 在预览图片下方显示名称和评分
                pose_name = target_pose['name']
                text_y = preview_y + preview_h + 20
                display = self._put_chinese_text(display, f"目标: {pose_name}", 
                                                (preview_x, text_y),
                                                font_size=14, color=(255, 255, 255))
                
                # 显示匹配评分（如果检测到人物）
                if best_person is not None:
                    score = processed_data.get('match_score', self.current_match_score)
                    score_text = f"匹配度: {score:.1f}分"
                    display = self._put_chinese_text(display, score_text, 
                                                    (preview_x, text_y + 20),
                                                    font_size=16, color=(0, 255, 255))
        
        # 添加模式指示（左下角）
        mode_text = []
        if self.display_flags['show_roi_crop']:
            mode_text.append("ROI")
        if self.display_flags['show_skeleton']:
            mode_text.append("骨骼")
        if self.display_flags['show_matting']:
            mode_text.append("抠像")
        
        mode_str = " | ".join(mode_text) if mode_text else "无"
        display = self._put_chinese_text(display, f"模式: {mode_str}", (10, h - 20),
                                        font_size=18, color=(255, 255, 255))
        
        return display
    
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
                
                # Create merged display (always returns a valid display)
                combined_display = self.create_display(frame, processed_data)
                cv2.imshow("AI Pose Match - RVM增强版", combined_display)
                
                # Update stats
                self._update_stats()
                
                # Handle keyboard input
                key = cv2.waitKey(1)
                
                # Handle arrow keys - OpenCV arrow key handling
                # Arrow keys on Windows may return 0xFF followed by specific codes
                key_code = key & 0xFF
                if key == -1:
                    pass  # No key pressed
                elif key_code == 0xFF:  # Extended key
                    # Check the second byte (upper 8 bits)
                    extended_code = (key >> 8) & 0xFF
                    if extended_code == 37:  # Left arrow
                        self.pose_matcher.switch_to_previous_pose()
                    elif extended_code == 39:  # Right arrow
                        self.pose_matcher.switch_to_next_pose()
                elif key_code == ord('a') or key_code == ord('A'):  # Alternative: A key for previous
                    self.pose_matcher.switch_to_previous_pose()
                elif key_code == ord('d') or key_code == ord('D'):  # Alternative: D key for next
                    self.pose_matcher.switch_to_next_pose()
                elif key_code == ord('q') or key_code == ord('Q') or key == 27:  # ESC key also exits
                    print("\n用户退出请求")
                    break
                
                # Handle other key presses (mask to get ASCII)
                key_ascii = key_code
                if key_ascii == ord('1'):
                    self.debug_flags['show_camera_full'] = not self.debug_flags['show_camera_full']
                    print(f"完整摄像头画面: {'开启' if self.debug_flags['show_camera_full'] else '关闭'}")
                
                elif key_ascii == ord('2'):
                    self.display_flags['show_roi_crop'] = not self.display_flags['show_roi_crop']
                    self.debug_flags['show_roi'] = self.display_flags['show_roi_crop']
                    print(f"ROI裁剪显示 (2): {'开启' if self.display_flags['show_roi_crop'] else '关闭'}")
                
                elif key_ascii == ord('3'):
                    self.display_flags['show_skeleton'] = not self.display_flags['show_skeleton']
                    self.debug_flags['show_pose_debug'] = self.display_flags['show_skeleton']
                    print(f"骨骼显示 (3): {'开启' if self.display_flags['show_skeleton'] else '关闭'}")
                
                elif key_ascii == ord('4'):
                    self.display_flags['show_matting'] = not self.display_flags['show_matting']
                    self.debug_flags['show_matting_debug'] = self.display_flags['show_matting']
                    print(f"抠像效果 (4): {'开启' if self.display_flags['show_matting'] else '关闭'}")
                
                elif key_ascii == ord('5'):
                    self.debug_flags['show_all'] = not self.debug_flags['show_all']
                    print(f"所有调试信息: {'开启' if self.debug_flags['show_all'] else '关闭'}")
                
                elif key_ascii == ord('f') or key_ascii == ord('F'):
                    self.debug_flags['show_fps'] = not self.debug_flags['show_fps']
                    print(f"FPS显示: {'开启' if self.debug_flags['show_fps'] else '关闭'}")
                
                elif key_ascii == ord('h') or key_ascii == ord('H'):
                    self.debug_flags['show_hints'] = not self.debug_flags['show_hints']
                    print(f"提示信息显示: {'开启' if self.debug_flags['show_hints'] else '关闭'}")
                
                elif key_ascii == ord('c') or key_ascii == ord('C'):
                    self._switch_camera()
                
                elif key_ascii == ord('s') or key_ascii == ord('S'):
                    output_path = f"debug_frame_{self.stats['frame_count']:05d}.jpg"
                    if combined_display is not None:
                        cv2.imwrite(output_path, combined_display)
                    else:
                        # Save ROI if available
                        if processed_data.get('roi') is not None:
                            cv2.imwrite(output_path, processed_data['roi'])
                    print(f"已保存帧到: {output_path}")
                
                elif key_ascii == ord('r') or key_ascii == ord('R'):
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
        if hasattr(self, 'pose_matcher'):
            self.pose_matcher.cleanup()
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

