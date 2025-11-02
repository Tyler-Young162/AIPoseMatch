"""
检查系统可用的摄像头数量
"""
import cv2

print("="*70)
print("检查可用摄像头")
print("="*70)

available_cameras = []
for i in range(10):  # 检查0-9
    cap = cv2.VideoCapture(i)
    ret, frame = cap.read()
    if ret:
        available_cameras.append(i)
        print(f"✓ 摄像头 {i}: 可用")
    cap.release()

print(f"\n总共找到 {len(available_cameras)} 个可用摄像头")
print(f"设备ID: {available_cameras}")

if len(available_cameras) < 2:
    print("\n⚠️  警告: 只有1个摄像头，无法运行双摄像头模式")
    print("建议: 使用单摄像头模式或修改代码让两个摄像头复用设备0")
else:
    print("\n✓ 摄像头数量足够，可以运行双摄像头模式")

print("="*70)

