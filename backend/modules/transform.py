"""
Image transformation modules
"""
import cv2
import numpy as np
from core.base_module import BaseCVModule


class GrayscaleModule(BaseCVModule):
    """Convert to grayscale"""
    
    @property
    def module_id(self) -> str:
        return "grayscale"
    
    @property
    def display_name(self) -> str:
        return "Grayscale"
    
    @property
    def description(self) -> str:
        return "Convert image to grayscale"
    
    @property
    def category(self) -> str:
        return "transform"
    
    @property
    def parameters(self):
        return []
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return image


class ResizeModule(BaseCVModule):
    """Resize image"""
    
    @property
    def module_id(self) -> str:
        return "resize"
    
    @property
    def display_name(self) -> str:
        return "Resize"
    
    @property
    def description(self) -> str:
        return "Resize image to specified dimensions"
    
    @property
    def category(self) -> str:
        return "transform"
    
    @property
    def parameters(self):
        return [
            {
                "name": "width",
                "type": "int",
                "default": 640,
                "min": 1,
                "max": 10000,
                "description": "Target width"
            },
            {
                "name": "height",
                "type": "int",
                "default": 480,
                "min": 1,
                "max": 10000,
                "description": "Target height"
            },
            {
                "name": "interpolation",
                "type": "select",
                "default": "linear",
                "options": ["linear", "cubic", "area", "nearest"],
                "description": "Interpolation method"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        width = kwargs.get("width", 640)
        height = kwargs.get("height", 480)
        interpolation = kwargs.get("interpolation", "linear")
        
        interp_map = {
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "nearest": cv2.INTER_NEAREST
        }
        
        interp = interp_map.get(interpolation, cv2.INTER_LINEAR)
        return cv2.resize(image, (width, height), interpolation=interp)


class RotateModule(BaseCVModule):
    """Rotate image"""
    
    @property
    def module_id(self) -> str:
        return "rotate"
    
    @property
    def display_name(self) -> str:
        return "Rotate"
    
    @property
    def description(self) -> str:
        return "Rotate image by specified angle"
    
    @property
    def category(self) -> str:
        return "transform"
    
    @property
    def parameters(self):
        return [
            {
                "name": "angle",
                "type": "float",
                "default": 90.0,
                "min": -360.0,
                "max": 360.0,
                "description": "Rotation angle in degrees (counter-clockwise)"
            },
            {
                "name": "scale",
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 5.0,
                "description": "Scale factor"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        angle = kwargs.get("angle", 90.0)
        scale = kwargs.get("scale", 1.0)
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(image, rotation_matrix, (w, h))


class FlipModule(BaseCVModule):
    """Flip image"""
    
    @property
    def module_id(self) -> str:
        return "flip"
    
    @property
    def display_name(self) -> str:
        return "Flip"
    
    @property
    def description(self) -> str:
        return "Flip image horizontally, vertically, or both"
    
    @property
    def category(self) -> str:
        return "transform"
    
    @property
    def parameters(self):
        return [
            {
                "name": "direction",
                "type": "select",
                "default": "horizontal",
                "options": ["horizontal", "vertical", "both"],
                "description": "Flip direction"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        direction = kwargs.get("direction", "horizontal")
        
        flip_code = {
            "horizontal": 1,
            "vertical": 0,
            "both": -1
        }.get(direction, 1)
        
        return cv2.flip(image, flip_code)

