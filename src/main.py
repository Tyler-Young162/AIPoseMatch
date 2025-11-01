"""
Main application entry point for AI Pose Match.
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from camera_manager import CameraManager
from pose_detector import PoseDetector
from person_selector import PersonSelector
from human_matting import HumanMatting
from visualizer import Visualizer


class AIPoseMatch:
    """
    Main application class that integrates all components.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize AI Pose Match application.
        
        Args:
            config_path: Path to configuration YAML file
        """
        print("Initializing AI Pose Match...")
        
        # Load configuration
        self.config = Config.load_from_yaml(config_path)
        
        # Initialize components
        self.camera = CameraManager(self.config)
        self.pose_detector = PoseDetector(self.config)
        self.person_selector = PersonSelector(self.config)
        self.matting = HumanMatting(self.config)
        self.visualizer = Visualizer(self.config)
        
        self.running = False
        self.frame_count = 0
    
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if successful, False otherwise
        """
        print("Initializing camera...")
        if not self.camera.initialize():
            print("Failed to initialize camera")
            return False
        
        print("Initializing matting...")
        # Matting will be initialized on first use if needed
        
        print("Initialization complete!")
        return True
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Visualization result
        """
        # Draw ROI overlay on original frame
        original_frame = frame.copy()
        if self.config.display.show_roi:
            original_frame = self.camera.draw_roi_overlay(original_frame)
        
        # Extract ROI
        roi_frame = self.camera.extract_roi(frame)
        
        if roi_frame is None:
            return self.visualizer.visualize_complete(
                original_frame, None, None, None, None, 0
            )
        
        # Detect poses in ROI
        all_persons = self.pose_detector.detect(roi_frame)
        num_detected = len(all_persons)
        
        # Select best person
        best_person = None
        alpha_matte = None
        matting_result = None
        
        if all_persons:
            # Get ROI center for person selection
            roi_center = self.person_selector.get_roi_center(self.config.roi)
            best_person = self.person_selector.select_best_person(all_persons, roi_center)
            
            # Visualize skeleton on ROI
            if best_person and self.config.display.show_skeleton:
                roi_frame_with_pose = self.pose_detector.draw_skeleton(roi_frame.copy(), [best_person])
            else:
                roi_frame_with_pose = roi_frame.copy()
            
            # Perform matting if person detected
            if best_person and self.config.display.show_matting:
                bbox = best_person.get('bounding_box')
                alpha_matte, matting_result = self.matting.process(roi_frame, bbox)
                
                # If matting succeeded, overlay on original background
                if matting_result is not None:
                    # Can apply to original frame or ROI
                    pass
        else:
            roi_frame_with_pose = roi_frame.copy()
        
        # Create complete visualization
        display = self.visualizer.visualize_complete(
            original_frame,
            roi_frame_with_pose,
            best_person,
            alpha_matte,
            matting_result,
            num_detected
        )
        
        return display
    
    def run(self):
        """
        Main application loop.
        """
        if not self.initialize():
            print("Failed to initialize application")
            return
        
        self.running = True
        print("\nStarting AI Pose Match...")
        print("Controls:")
        print("  Press 'Q' to quit")
        print("  Press 'S' to save current frame")
        print("  Press 'R' to toggle ROI display")
        print("\nRunning...\n")
        
        try:
            while self.running:
                # Read frame from camera
                ret, frame = self.camera.read()
                
                if not ret:
                    print("Failed to read frame from camera, continuing...")
                    continue
                
                # Process frame
                display = self.process_frame(frame)
                
                # Show result
                window_name = "AI Pose Match"
                cv2.imshow(window_name, display)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q') or key == 27:  # ESC key also exits
                    print("Quit requested")
                    self.running = False
                    break
                elif key == ord('s') or key == ord('S'):
                    # Save current frame
                    output_path = f"output_frame_{self.frame_count:05d}.jpg"
                    cv2.imwrite(output_path, display)
                    print(f"Saved frame to: {output_path}")
                elif key == ord('r') or key == ord('R'):
                    # Toggle ROI display
                    self.config.display.show_roi = not self.config.display.show_roi
                    print(f"ROI display: {self.config.display.show_roi}")
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        print("\nCleaning up...")
        self.camera.release()
        self.pose_detector.cleanup()
        self.matting.cleanup()
        cv2.destroyAllWindows()
        print("Cleanup complete")


def main():
    """Application entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Pose Match - Human Pose Detection & Matting")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()
    
    # Create and run application
    app = AIPoseMatch(args.config)
    app.run()


if __name__ == "__main__":
    main()

