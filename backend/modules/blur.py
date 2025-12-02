"""
Blur operation modules
"""
import cv2
import numpy as np
from core.base_module import BaseCVModule


class BlurModule(BaseCVModule):
    """Simple box blur"""
    
    @property
    def module_id(self) -> str:
        return "blur"
    
    @property
    def display_name(self) -> str:
        return "Blur"
    
    @property
    def description(self) -> str:
        return "Apply simple box blur filter"
    
    @property
    def category(self) -> str:
        return "filter"
    
    @property
    def parameters(self):
        return [
            {
                "name": "kernel_size",
                "type": "int",
                "default": 5,
                "min": 3,
                "max": 51,
                "description": "Kernel size (must be odd number)"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        kernel_size = kwargs.get("kernel_size", 5)
        # Ensure odd number
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.blur(image, (kernel_size, kernel_size))


class GaussianBlurModule(BaseCVModule):
    """Gaussian blur"""
    
    @property
    def module_id(self) -> str:
        return "gaussian_blur"
    
    @property
    def display_name(self) -> str:
        return "Gaussian Blur"
    
    @property
    def description(self) -> str:
        return "Apply Gaussian blur filter"
    
    @property
    def category(self) -> str:
        return "filter"
    
    @property
    def parameters(self):
        return [
            {
                "name": "kernel_size",
                "type": "int",
                "default": 5,
                "min": 3,
                "max": 51,
                "description": "Kernel size (must be odd number)"
            },
            {
                "name": "sigma_x",
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 10.0,
                "description": "Gaussian kernel standard deviation in X direction"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        kernel_size = kwargs.get("kernel_size", 5)
        sigma_x = kwargs.get("sigma_x", 0.0)
        
        # Ensure odd number
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x)

