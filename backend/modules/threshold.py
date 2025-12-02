"""
Threshold operation modules
"""
import cv2
import numpy as np
from core.base_module import BaseCVModule


class ThresholdModule(BaseCVModule):
    """Binary threshold"""
    
    @property
    def module_id(self) -> str:
        return "threshold"
    
    @property
    def display_name(self) -> str:
        return "Threshold"
    
    @property
    def description(self) -> str:
        return "Apply binary threshold"
    
    @property
    def category(self) -> str:
        return "filter"
    
    @property
    def parameters(self):
        return [
            {
                "name": "threshold_value",
                "type": "int",
                "default": 127,
                "min": 0,
                "max": 255,
                "description": "Threshold value"
            },
            {
                "name": "max_value",
                "type": "int",
                "default": 255,
                "min": 0,
                "max": 255,
                "description": "Maximum value to use with THRESH_BINARY"
            },
            {
                "name": "threshold_type",
                "type": "select",
                "default": "binary",
                "options": ["binary", "binary_inv", "trunc", "tozero", "tozero_inv"],
                "description": "Threshold type"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        threshold_value = kwargs.get("threshold_value", 127)
        max_value = kwargs.get("max_value", 255)
        threshold_type = kwargs.get("threshold_type", "binary")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Map threshold type
        type_map = {
            "binary": cv2.THRESH_BINARY,
            "binary_inv": cv2.THRESH_BINARY_INV,
            "trunc": cv2.THRESH_TRUNC,
            "tozero": cv2.THRESH_TOZERO,
            "tozero_inv": cv2.THRESH_TOZERO_INV
        }
        
        cv_type = type_map.get(threshold_type, cv2.THRESH_BINARY)
        _, thresh = cv2.threshold(gray, threshold_value, max_value, cv_type)
        
        # Convert back to BGR for consistency
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

