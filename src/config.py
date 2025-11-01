"""
Configuration management module for AI Pose Match application.
"""
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class CameraConfig:
    """Camera capture settings."""
    device_id: int = 0
    resolution: Tuple[int, int] = (1280, 720)
    fps: int = 30


@dataclass
class ROIConfig:
    """Region of Interest settings."""
    x_min: float = 0.4
    x_max: float = 0.6
    y_min: float = 0.0
    y_max: float = 0.8


@dataclass
class PoseConfig:
    """Pose detection settings."""
    model: str = "mediapipe"
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class PersonSelectionConfig:
    """Person selection scoring weights."""
    completeness_weight: float = 0.4
    height_weight: float = 0.3
    centeredness_weight: float = 0.3
    min_keypoints_visible: int = 8


@dataclass
class MattingConfig:
    """Human matting settings."""
    model: str = "rvm"
    device: str = "auto"  # "cuda", "cpu", or "auto" (auto-detect)
    downsample_ratio: float = 0.25
    filter_incomplete: bool = True
    min_person_height_ratio: float = 0.3


@dataclass
class DisplayConfig:
    """Display and visualization settings."""
    window_width: int = 1920
    window_height: int = 1080
    show_roi: bool = True
    show_skeleton: bool = True
    show_matting: bool = True


@dataclass
class Config:
    """Main configuration container."""
    camera: CameraConfig
    roi: ROIConfig
    pose: PoseConfig
    person_selection: PersonSelectionConfig
    matting: MattingConfig
    display: DisplayConfig

    @classmethod
    def load_from_yaml(cls, filepath: str = "config.yaml") -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            filepath: Path to YAML configuration file
            
        Returns:
            Config object with loaded settings
        """
        config_path = Path(filepath)
        
        if not config_path.exists():
            # Create default config file
            config = cls.get_default()
            cls.save_to_yaml(config, filepath)
            return config
        
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls(
            camera=CameraConfig(**data.get('camera', {})),
            roi=ROIConfig(**data.get('roi', {})),
            pose=PoseConfig(**data.get('pose', {})),
            person_selection=PersonSelectionConfig(**data.get('person_selection', {})),
            matting=MattingConfig(**data.get('matting', {})),
            display=DisplayConfig(**data.get('display', {}))
        )
    
    @classmethod
    def get_default(cls) -> 'Config':
        """Get default configuration."""
        return cls(
            camera=CameraConfig(),
            roi=ROIConfig(),
            pose=PoseConfig(),
            person_selection=PersonSelectionConfig(),
            matting=MattingConfig(),
            display=DisplayConfig()
        )
    
    @classmethod
    def save_to_yaml(cls, config: 'Config', filepath: str = "config.yaml"):
        """
        Save configuration to YAML file.
        
        Args:
            config: Config object to save
            filepath: Path to save YAML file
        """
        data = {
            'camera': {
                'device_id': config.camera.device_id,
                'resolution': list(config.camera.resolution),
                'fps': config.camera.fps
            },
            'roi': {
                'x_min': config.roi.x_min,
                'x_max': config.roi.x_max,
                'y_min': config.roi.y_min,
                'y_max': config.roi.y_max
            },
            'pose': {
                'model': config.pose.model,
                'min_detection_confidence': config.pose.min_detection_confidence,
                'min_tracking_confidence': config.pose.min_tracking_confidence
            },
            'person_selection': {
                'completeness_weight': config.person_selection.completeness_weight,
                'height_weight': config.person_selection.height_weight,
                'centeredness_weight': config.person_selection.centeredness_weight,
                'min_keypoints_visible': config.person_selection.min_keypoints_visible
            },
            'matting': {
                'model': config.matting.model,
                'device': config.matting.device,
                'downsample_ratio': config.matting.downsample_ratio,
                'filter_incomplete': config.matting.filter_incomplete,
                'min_person_height_ratio': config.matting.min_person_height_ratio
            },
            'display': {
                'window_width': config.display.window_width,
                'window_height': config.display.window_height,
                'show_roi': config.display.show_roi,
                'show_skeleton': config.display.show_skeleton,
                'show_matting': config.display.show_matting
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

