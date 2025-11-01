"""
Person selection module for choosing the best person from multiple detections.
"""
import numpy as np
from typing import List, Dict, Optional
from config import Config


class PersonSelector:
    """
    Selects the best person from multiple detections based on scoring criteria.
    """
    
    def __init__(self, config: Config):
        """
        Initialize person selector.
        
        Args:
            config: Application configuration object
        """
        self.config = config
        self.selection_config = config.person_selection
    
    def calculate_completeness_score(self, person: Dict) -> float:
        """
        Calculate completeness score based on visible keypoints.
        
        Args:
            person: Person data dictionary
            
        Returns:
            Completeness score (0-1)
        """
        return person.get('completeness', 0.0)
    
    def calculate_height_score(self, persons: List[Dict]) -> float:
        """
        Calculate normalized height score (relative to tallest person).
        
        Args:
            persons: List of all detected persons
            
        Returns:
            Function that returns height score for a given person
        """
        if not persons:
            return lambda p: 0.0
        
        max_height = max(p.get('height', 0) for p in persons)
        if max_height == 0:
            return lambda p: 0.0
        
        return lambda p: p.get('height', 0) / max_height
    
    def calculate_centeredness_score(self, person: Dict, roi_center: tuple) -> float:
        """
        Calculate how centered the person is within the ROI.
        
        Args:
            person: Person data dictionary
            roi_center: (x, y) center of ROI in normalized coordinates (0-1)
            
        Returns:
            Centeredness score (0-1), higher is more centered
        """
        landmarks = person.get('landmarks', np.array([]))
        
        if len(landmarks) == 0:
            return 0.0
        
        # Get visible landmarks
        visible_landmarks = landmarks[landmarks[:, 2] > 0.5]
        
        if len(visible_landmarks) == 0:
            return 0.0
        
        # Calculate person center (use all visible landmarks)
        person_center_x = np.mean(visible_landmarks[:, 0])
        person_center_y = np.mean(visible_landmarks[:, 1])
        
        # Calculate distance from ROI center (normalized)
        dx = person_center_x - roi_center[0]
        dy = person_center_y - roi_center[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Normalize: score is higher when closer to center
        # Using a normalized distance (max possible in 1x1 square is sqrt(2)/2 from corner)
        max_distance = np.sqrt(2) / 2
        centeredness = 1.0 - min(distance / max_distance, 1.0)
        
        return centeredness
    
    def select_best_person(self, persons: List[Dict], roi_center: tuple) -> Optional[Dict]:
        """
        Select the best person from multiple detections.
        
        Args:
            persons: List of detected persons
            roi_center: (x, y) center of ROI in normalized coordinates
            
        Returns:
            Best person dictionary or None if no valid person found
        """
        if not persons:
            return None
        
        # Filter persons by minimum keypoint requirement
        valid_persons = [
            p for p in persons 
            if p.get('num_visible_keypoints', 0) >= self.selection_config.min_keypoints_visible
        ]
        
        if not valid_persons:
            return None
        
        # Calculate height score function (needs all persons for normalization)
        get_height_score = self.calculate_height_score(valid_persons)
        
        # Calculate scores for each person
        scored_persons = []
        for person in valid_persons:
            completeness = self.calculate_completeness_score(person)
            height = get_height_score(person)
            centeredness = self.calculate_centeredness_score(person, roi_center)
            
            # Weighted combination
            total_score = (
                self.selection_config.completeness_weight * completeness +
                self.selection_config.height_weight * height +
                self.selection_config.centeredness_weight * centeredness
            )
            
            scored_persons.append({
                'person': person,
                'completeness': completeness,
                'height': height,
                'centeredness': centeredness,
                'total_score': total_score
            })
        
        # Sort by total score and return the best one
        scored_persons.sort(key=lambda x: x['total_score'], reverse=True)
        best = scored_persons[0]['person']
        
        # Add debug info
        best['_selection_scores'] = {
            'completeness': scored_persons[0]['completeness'],
            'height': scored_persons[0]['height'],
            'centeredness': scored_persons[0]['centeredness'],
            'total': scored_persons[0]['total_score']
        }
        
        return best
    
    def get_roi_center(self, roi_config) -> tuple:
        """
        Get the center of ROI in normalized coordinates.
        
        Args:
            roi_config: ROI configuration object
            
        Returns:
            (x, y) tuple in normalized coordinates (0-1)
        """
        center_x = (roi_config.x_min + roi_config.x_max) / 2.0
        center_y = (roi_config.y_min + roi_config.y_max) / 2.0
        return (center_x, center_y)

