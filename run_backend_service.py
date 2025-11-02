"""
AI Pose Match - 后台服务模式
默认在后台运行双摄像头，不显示窗口，通过Unity通信模块推送数据
"""
import sys
import time
import threading
from pathlib import Path
from typing import Tuple

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from camera_manager import CameraManager
from pose_detector import PoseDetector
from person_selector import PersonSelector
from human_matting import HumanMatting
from pose_matcher import PoseMatcher
from unity_communication import UnityCommunication
import cv2
import numpy as np


class BackendService:
    """
    后台服务模式
    在后台运行双摄像头，推送数据到Unity
    """
    
    def __init__(self, config_path: str = "config.yaml", 
                 unity_host: str = "127.0.0.1", 
                 unity_video_port: int = 8888,
                 unity_control_port: int = 8889,
                 show_window: bool = False):
        """
        初始化后台服务
        
        Args:
            config_path: 配置文件路径
            unity_host: Unity服务器地址
            unity_video_port: Unity视频流端口
            unity_control_port: Unity控制命令端口
            show_window: 是否显示调试窗口（默认False）
        """
        print("=" * 70)
        print("AI Pose Match - 后台服务模式")
        print("=" * 70)
        
        # 加载配置
        self.config = Config.load_from_yaml(config_path)
        self.show_window = show_window
        
        # 初始化Unity通信
        print("\n初始化Unity通信...")
        self.unity_comm = UnityCommunication(
            host=unity_host,
            video_port=unity_video_port,
            control_port=unity_control_port
        )
        self.unity_comm.on_pose_switch = self._handle_pose_switch
        
        # 初始化姿态比对器
        print("\n初始化姿态比对模块...")
        self.pose_matcher = PoseMatcher(pose_folder="Pose")
        if len(self.pose_matcher.target_poses) > 0:
            print(f"[OK] 已加载 {len(self.pose_matcher.target_poses)} 个目标姿态")
        
        # 创建摄像头0的配置（临时修改device_id）
        import copy
        config0 = copy.deepcopy(self.config)
        config0.camera.device_id = 0
        
        # 初始化摄像头0
        print("\n初始化摄像头0...")
        self.camera0 = CameraManager(config0)
        self.pose_detector0 = PoseDetector(self.config)
        self.person_selector0 = PersonSelector(self.config)
        self.matting0 = HumanMatting(self.config)
        
        # 创建摄像头1的配置（临时修改device_id）
        config1 = copy.deepcopy(self.config)
        config1.camera.device_id = 1
        
        # 初始化摄像头1
        print("\n初始化摄像头1...")
        self.camera1 = CameraManager(config1)
        self.pose_detector1 = PoseDetector(self.config)
        self.person_selector1 = PersonSelector(self.config)
        self.matting1 = HumanMatting(self.config)
        
        # 初始化状态
        self.is_running = False
        self.current_scores = [0.0, 0.0]  # [camera0_score, camera1_score]
    
    def initialize(self) -> bool:
        """初始化所有组件"""
        # 初始化摄像头
        if not self.camera0.initialize():
            print("[ERROR] 摄像头0初始化失败")
            return False
        
        if not self.camera1.initialize():
            print("[ERROR] 摄像头1初始化失败")
            self.camera0.release()
            return False
        
        print("[OK] 摄像头0初始化成功")
        print("[OK] 摄像头1初始化成功")
        
        # 初始化抠像模块
        if not self.matting0.initialize():
            print("[ERROR] 摄像头0抠像模块初始化失败")
            return False
        
        if not self.matting1.initialize():
            print("[ERROR] 摄像头1抠像模块初始化失败")
            self.matting0.cleanup()
            return False
        
        print("[OK] 摄像头0抠像模块初始化成功")
        print("[OK] 摄像头1抠像模块初始化成功")
        
        # 连接到Unity
        if not self.unity_comm.connect():
            print("[WARN] 无法连接到Unity，将在无Unity连接模式下运行")
            print("       请确保Unity已启动并监听以下端口：")
            print(f"       视频流: {self.unity_comm.video_port}")
            print(f"       控制命令: {self.unity_comm.control_port}")
        
        print("\n" + "=" * 70)
        print("初始化完成！")
        print("=" * 70)
        if self.show_window:
            print("\n调试窗口已启用")
            print("键盘控制：")
            print("  ESC/Q    - 退出程序")
            print("  ←/→ 或 A/D - 切换目标姿态")
        else:
            print("\n后台服务模式：不显示窗口")
            print("数据将推送到Unity")
        print("=" * 70 + "\n")
        
        return True
    
    def _handle_pose_switch(self, pose_index: int, direction: str):
        """处理Unity发送的pose切换命令"""
        if direction == 'next':
            self.pose_matcher.switch_to_next_pose()
            print(f"[Unity] 切换到下一个姿态: {self.pose_matcher.get_current_target_pose_name()}")
        elif direction == 'previous':
            self.pose_matcher.switch_to_previous_pose()
            print(f"[Unity] 切换到上一个姿态: {self.pose_matcher.get_current_target_pose_name()}")
        elif pose_index is not None:
            # 直接切换指定索引
            current_index = self.pose_matcher.current_pose_index
            target_poses = self.pose_matcher.target_poses
            if 0 <= pose_index < len(target_poses):
                self.pose_matcher.current_pose_index = pose_index
                print(f"[Unity] 切换到姿态索引 {pose_index}: {self.pose_matcher.get_current_target_pose_name()}")
    
    def process_frame(self, camera_id: int, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        处理单帧
        
        Args:
            camera_id: 摄像头ID (0 或 1)
            frame: 输入帧
            
        Returns:
            (处理后的帧, 匹配分数)
        """
        # 选择对应的组件
        if camera_id == 0:
            pose_detector = self.pose_detector0
            person_selector = self.person_selector0
            matting = self.matting0
        else:
            pose_detector = self.pose_detector1
            person_selector = self.person_selector1
            matting = self.matting1
        
        # 选择对应的摄像头
        camera = self.camera0 if camera_id == 0 else self.camera1
        
        # 提取ROI
        roi_frame = camera.extract_roi(frame)
        
        if roi_frame is None:
            # 没有ROI，返回绿色背景
            target_h = int((self.config.roi.y_max - self.config.roi.y_min) * frame.shape[0])
            target_w = int((self.config.roi.x_max - self.config.roi.x_min) * frame.shape[1])
            display = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            display[:, :, 1] = 255  # 绿色背景
            return display, 0.0
        
        # 姿态检测
        all_persons = pose_detector.detect(roi_frame)
        match_score = 0.0
        
        # 选择最佳人物
        best_person = None
        if all_persons:
            roi_center = person_selector.get_roi_center(self.config.roi)
            best_person = person_selector.select_best_person(all_persons, roi_center)
        
        # 抠像
        matting_result = None
        alpha_matte = None
        if best_person:
            bbox = best_person.get('bounding_box')
            landmarks = best_person.get('landmarks')
            matting._last_person_data = best_person
            alpha_matte, matting_result = matting.process(roi_frame, bbox, None, landmarks)
            
            # 姿态比对
            target_pose = self.pose_matcher.get_current_target_pose()
            if target_pose is not None and best_person.get('landmarks') is not None:
                score, _ = self.pose_matcher.calculate_pose_similarity(
                    best_person['landmarks'], target_pose['landmarks']
                )
                match_score = score
        
        # 创建显示画面
        target_h = roi_frame.shape[0]
        target_w = roi_frame.shape[1]
        
        # 如果没有人物，显示绿色背景
        if best_person is None or matting_result is None:
            display = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            display[:, :, 1] = 255  # 绿色背景
        else:
            # 调整matting结果大小
            if matting_result.shape[:2] != (target_h, target_w):
                matting_result = cv2.resize(matting_result, (target_w, target_h))
            display = matting_result.copy()
        
        # 绘制骨骼（如果有）
        if best_person is not None:
            target_pose = self.pose_matcher.get_current_target_pose()
            if target_pose is not None:
                display = self.pose_matcher.draw_skeleton_with_matching(
                    display, best_person, target_pose
                )
            else:
                display = pose_detector.draw_skeleton(display, [best_person])
        
        return display, match_score
    
    def run(self):
        """运行后台服务"""
        if not self.initialize():
            print("[ERROR] 初始化失败")
            return
        
        self.is_running = True
        
        # 创建调试窗口（如果需要）
        if self.show_window:
            cv2.namedWindow("摄像头 0 (调试)", cv2.WINDOW_NORMAL)
            cv2.namedWindow("摄像头 1 (调试)", cv2.WINDOW_NORMAL)
        
        print("\n后台服务开始运行...")
        print("按 ESC 或 Q 键退出（如果启用调试窗口）\n")
        
        try:
            while self.is_running:
                # 读取两路摄像头
                ret0, frame0 = self.camera0.read()
                ret1, frame1 = self.camera1.read()
                
                if not ret0 or not ret1:
                    time.sleep(0.1)
                    continue
                
                # 处理两路帧
                display0, score0 = self.process_frame(0, frame0)
                display1, score1 = self.process_frame(1, frame1)
                
                self.current_scores = [score0, score1]
                
                # 获取当前姿态名称
                target_pose = self.pose_matcher.get_current_target_pose()
                pose_name = target_pose['name'] if target_pose else ""
                
                # 推送到Unity（分别发送两个摄像头的数据）
                if self.unity_comm.is_connected:
                    # 发送摄像头0
                    self.unity_comm.send_frame(0, display0, score0, pose_name)
                    # 发送摄像头1
                    self.unity_comm.send_frame(1, display1, score1, pose_name)
                
                # 显示调试窗口（如果需要）
                if self.show_window:
                    cv2.imshow("摄像头 0 (调试)", display0)
                    cv2.imshow("摄像头 1 (调试)", display1)
                    
                    # 处理键盘输入
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q') or key == 27:
                        print("\n退出请求")
                        self.is_running = False
                        break
                    elif key == ord('a') or key == ord('A'):
                        self.pose_matcher.switch_to_previous_pose()
                    elif key == ord('d') or key == ord('D'):
                        self.pose_matcher.switch_to_next_pose()
        
        except KeyboardInterrupt:
            print("\n程序被中断")
        except Exception as e:
            print(f"[ERROR] 运行错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")
        self.is_running = False
        
        # 断开Unity连接
        self.unity_comm.disconnect()
        
        # 清理摄像头
        self.camera0.release()
        self.camera1.release()
        
        # 清理检测器
        self.pose_detector0.cleanup()
        self.pose_detector1.cleanup()
        
        # 清理抠像
        self.matting0.cleanup()
        self.matting1.cleanup()
        
        # 清理窗口
        if self.show_window:
            cv2.destroyAllWindows()
        
        print("清理完成")


def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Pose Match - 后台服务模式")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--unity-host', type=str, default='127.0.0.1',
                       help='Unity服务器地址')
    parser.add_argument('--unity-video-port', type=int, default=8888,
                       help='Unity视频流端口')
    parser.add_argument('--unity-control-port', type=int, default=8889,
                       help='Unity控制命令端口')
    parser.add_argument('--show-window', action='store_true',
                       help='显示调试窗口')
    
    args = parser.parse_args()
    
    service = BackendService(
        config_path=args.config,
        unity_host=args.unity_host,
        unity_video_port=args.unity_video_port,
        unity_control_port=args.unity_control_port,
        show_window=args.show_window
    )
    
    service.run()


if __name__ == "__main__":
    main()

