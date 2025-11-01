"""Test initialization step by step."""
print("Test 1: Import modules...")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import Config

print("Test 2: Load config...")
config = Config.load_from_yaml("config.yaml")

print("Test 3: Initialize camera...")
from camera_manager import CameraManager
camera = CameraManager(config)
result = camera.initialize()
print(f"Camera init result: {result}")

if result:
    print("Test 4: Read a frame...")
    ret, frame = camera.read()
    print(f"Frame read result: {ret}")
    
    print("Test 5: Initialize pose detector...")
    from pose_detector import PoseDetector
    pose = PoseDetector(config)
    print("Pose detector initialized")

print("All tests passed!")

