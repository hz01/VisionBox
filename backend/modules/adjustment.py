"""
Image adjustment modules
"""
import cv2
import numpy as np
from core.base_module import BaseCVModule


class BrightnessModule(BaseCVModule):
    """Adjust brightness"""
    
    @property
    def module_id(self) -> str:
        return "brightness"
    
    @property
    def display_name(self) -> str:
        return "Brightness"
    
    @property
    def description(self) -> str:
        return "Adjust image brightness"
    
    @property
    def category(self) -> str:
        return "adjustment"
    
    @property
    def parameters(self):
        return [
            {
                "name": "value",
                "type": "int",
                "default": 0,
                "min": -100,
                "max": 100,
                "description": "Brightness adjustment value"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        value = kwargs.get("value", 0)
        result = image.astype(np.float32) + value
        return np.clip(result, 0, 255).astype(np.uint8)


class ContrastModule(BaseCVModule):
    """Adjust contrast"""
    
    @property
    def module_id(self) -> str:
        return "contrast"
    
    @property
    def display_name(self) -> str:
        return "Contrast"
    
    @property
    def description(self) -> str:
        return "Adjust image contrast"
    
    @property
    def category(self) -> str:
        return "adjustment"
    
    @property
    def parameters(self):
        return [
            {
                "name": "alpha",
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 3.0,
                "description": "Contrast control (1.0 = no change, >1.0 = more contrast)"
            },
            {
                "name": "beta",
                "type": "int",
                "default": 0,
                "min": -100,
                "max": 100,
                "description": "Brightness control (0 = no change)"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        alpha = kwargs.get("alpha", 1.0)
        beta = kwargs.get("beta", 0)
        
        result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return result

