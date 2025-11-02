"""
独立的RVM抠像效果测试程序
专门用于测试和查看RVM模型的抠像质量
"""
import cv2
import numpy as np
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from human_matting import HumanMatting

def create_test_backgrounds(frame_shape):
    """创建测试背景"""
    h, w = frame_shape[:2]
    
    backgrounds = {
        '绿色': np.zeros((h, w, 3), dtype=np.uint8),
        '蓝色': np.zeros((h, w, 3), dtype=np.uint8),
        '红色': np.zeros((h, w, 3), dtype=np.uint8),
        '棋盘格': np.zeros((h, w, 3), dtype=np.uint8),
        '白色': np.ones((h, w, 3), dtype=np.uint8) * 255,
        '黑色': np.zeros((h, w, 3), dtype=np.uint8),
    }
    
    backgrounds['绿色'][:, :, 1] = 255  # BGR: Green
    backgrounds['蓝色'][:, :, 0] = 255  # BGR: Blue
    backgrounds['红色'][:, :, 2] = 255  # BGR: Red
    
    # Checkerboard pattern
    square_size = 20
    for y in range(0, h, square_size):
        for x in range(0, w, square_size):
            if (x // square_size + y // square_size) % 2 == 0:
                backgrounds['棋盘格'][y:y+square_size, x:x+square_size] = [255, 255, 255]
            else:
                backgrounds['棋盘格'][y:y+square_size, x:x+square_size] = [0, 0, 0]
    
    return backgrounds

def main():
    print("=" * 70)
    print("RVM 抠像效果测试程序")
    print("=" * 70)
    
    # Load config
    config_path = "config.yaml"
    if not Path(config_path).exists():
        print(f"⚠ 配置文件不存在，使用默认配置")
        config = Config.get_default()
    else:
        config = Config.load_from_yaml(config_path)
    
    # Force CUDA if available
    if torch.cuda.is_available():
        config.matting.device = "cuda"
        print(f"✓ 检测到CUDA，将使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        config.matting.device = "cpu"
        print("⚠ 未检测到CUDA，将使用CPU")
    
    print(f"\n配置:")
    print(f"  设备: {config.matting.device}")
    print(f"  降采样比例: {config.matting.downsample_ratio}")
    print(f"  模型: {config.matting.model}")
    
    # Initialize matting
    print(f"\n初始化抠像模块...")
    matting = HumanMatting(config)
    if not matting.initialize():
        print("✗ 初始化失败！")
        return
    
    print(f"✓ 初始化成功，模型类型: {matting.model_type}")
    
    # Initialize camera
    print(f"\n初始化摄像头...")
    cap = cv2.VideoCapture(config.camera.device_id)
    if not cap.isOpened():
        print(f"✗ 无法打开摄像头 {config.camera.device_id}")
        print("提示: 尝试修改config.yaml中的device_id (0, 1, 2...)")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.resolution[1])
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ 摄像头已打开: {w}x{h}")
    
    # Create test backgrounds
    backgrounds = create_test_backgrounds((h, w))
    bg_names = list(backgrounds.keys())
    current_bg_index = 0
    
    print(f"\n" + "=" * 70)
    print("操作说明:")
    print("  SPACE  - 切换背景")
    print("  S      - 保存当前帧")
    print("  +/-    - 调整降采样比例 (速度 vs 质量)")
    print("  Q/ESC  - 退出")
    print("=" * 70 + "\n")
    
    frame_count = 0
    fps_time = None
    fps_counter = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠ 无法读取摄像头画面")
                break
            
            frame_count += 1
            
            # Calculate FPS
            import time
            current_time = time.time()
            if fps_time is None:
                fps_time = current_time
            fps_counter += 1
            if current_time - fps_time >= 1.0:
                fps = fps_counter / (current_time - fps_time)
                fps_counter = 0
                fps_time = current_time
            else:
                fps = -1
            
            # Process matting
            alpha_matte, composited = matting.process(frame)
            
            if alpha_matte is None or composited is None:
                print("⚠ 抠像处理失败，跳过此帧")
                continue
            
            # Get current background
            bg_name = bg_names[current_bg_index]
            background = backgrounds[bg_name]
            
            # Resize background if needed
            if background.shape[:2] != frame.shape[:2]:
                background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
            
            # Composite with test background
            result_with_bg = matting.composite(frame, alpha_matte, background)
            
            # Create display
            # Left: Original, Center: Alpha matte, Right: Composited result
            h_display = h
            w_display = w * 3
            
            display = np.zeros((h_display, w_display, 3), dtype=np.uint8)
            
            # Original frame
            display[:, 0:w] = frame
            
            # Alpha matte (as grayscale)
            alpha_display = cv2.cvtColor(alpha_matte, cv2.COLOR_GRAY2BGR)
            display[:, w:w*2] = alpha_display
            
            # Composited result
            display[:, w*2:w*3] = result_with_bg
            
            # Add labels
            cv2.putText(display, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "Alpha Matte", (w + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display, f"Result (BG: {bg_name})", (w*2 + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Add info
            info_y = h_display - 80
            info_text = [
                f"Model: {matting.model_type}",
                f"Device: {config.matting.device}",
                f"Downsample: {config.matting.downsample_ratio}",
                f"Alpha range: [{alpha_matte.min()}, {alpha_matte.max()}], mean: {alpha_matte.mean():.1f}",
            ]
            
            if fps > 0:
                info_text.append(f"FPS: {fps:.1f}")
            
            for i, text in enumerate(info_text):
                cv2.putText(display, text, (10, info_y + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show display
            cv2.imshow("RVM Matting Test", display)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # SPACE - switch background
                current_bg_index = (current_bg_index + 1) % len(bg_names)
                print(f"切换背景: {bg_names[current_bg_index]}")
            elif key == ord('s') or key == ord('S'):  # Save frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"rvm_test_original_{timestamp}.jpg", frame)
                cv2.imwrite(f"rvm_test_alpha_{timestamp}.jpg", alpha_matte)
                cv2.imwrite(f"rvm_test_result_{timestamp}.jpg", result_with_bg)
                print(f"✓ 已保存: rvm_test_*_{timestamp}.jpg")
            elif key == ord('+') or key == ord('='):  # Increase downsample (faster)
                config.matting.downsample_ratio = max(0.1, config.matting.downsample_ratio - 0.05)
                print(f"降采样比例: {config.matting.downsample_ratio:.2f} (更快)")
            elif key == ord('-') or key == ord('_'):  # Decrease downsample (better quality)
                config.matting.downsample_ratio = min(1.0, config.matting.downsample_ratio + 0.05)
                print(f"降采样比例: {config.matting.downsample_ratio:.2f} (更好质量)")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        matting.cleanup()
        print("\n✓ 测试完成")

if __name__ == "__main__":
    main()

