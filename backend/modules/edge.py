"""
Edge detection modules
"""
import cv2
import numpy as np
from core.base_module import BaseCVModule


class EdgeDetectionModule(BaseCVModule):
    """Canny edge detection"""
    
    @property
    def module_id(self) -> str:
        return "edge_detection"
    
    @property
    def display_name(self) -> str:
        return "Edge Detection (Canny)"
    
    @property
    def description(self) -> str:
        return "Detect edges using Canny algorithm"
    
    @property
    def category(self) -> str:
        return "detection"
    
    @property
    def parameters(self):
        return [
            {
                "name": "threshold1",
                "type": "float",
                "default": 100.0,
                "min": 0.0,
                "max": 500.0,
                "description": "First threshold for hysteresis procedure"
            },
            {
                "name": "threshold2",
                "type": "float",
                "default": 200.0,
                "min": 0.0,
                "max": 500.0,
                "description": "Second threshold for hysteresis procedure"
            },
            {
                "name": "aperture_size",
                "type": "int",
                "default": 3,
                "min": 3,
                "max": 7,
                "description": "Aperture size for Sobel operator"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        threshold1 = kwargs.get("threshold1", 100.0)
        threshold2 = kwargs.get("threshold2", 200.0)
        aperture_size = kwargs.get("aperture_size", 3)
        
        # Ensure aperture_size is odd and valid
        if aperture_size not in [3, 5, 7]:
            aperture_size = 3
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=aperture_size)
        
        # Convert back to BGR for consistency
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

