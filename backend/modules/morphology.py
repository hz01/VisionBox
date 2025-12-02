"""
Morphological operations modules
"""
import cv2
import numpy as np
from core.base_module import BaseCVModule


class MorphologyModule(BaseCVModule):
    """Morphological operations (erosion, dilation, opening, closing)"""
    
    @property
    def module_id(self) -> str:
        return "morphology"
    
    @property
    def display_name(self) -> str:
        return "Morphological Operations"
    
    @property
    def description(self) -> str:
        return "Apply morphological operations: erosion, dilation, opening, or closing"
    
    @property
    def category(self) -> str:
        return "filter"
    
    @property
    def parameters(self):
        return [
            {
                "name": "operation",
                "type": "select",
                "default": "dilation",
                "options": ["erosion", "dilation", "opening", "closing"],
                "description": "Type of morphological operation"
            },
            {
                "name": "kernel_size",
                "type": "int",
                "default": 5,
                "min": 3,
                "max": 31,
                "description": "Size of the structuring element kernel"
            },
            {
                "name": "kernel_shape",
                "type": "select",
                "default": "rectangle",
                "options": ["rectangle", "ellipse", "cross"],
                "description": "Shape of the structuring element"
            },
            {
                "name": "iterations",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 10,
                "description": "Number of times the operation is applied"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        operation = kwargs.get("operation", "dilation")
        kernel_size = kwargs.get("kernel_size", 5)
        kernel_shape = kwargs.get("kernel_shape", "rectangle")
        iterations = kwargs.get("iterations", 1)
        
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create structuring element
        if kernel_shape == "rectangle":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        elif kernel_shape == "ellipse":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        elif kernel_shape == "cross":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply morphological operation
        if operation == "erosion":
            result = cv2.erode(gray, kernel, iterations=iterations)
        elif operation == "dilation":
            result = cv2.dilate(gray, kernel, iterations=iterations)
        elif operation == "opening":
            result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == "closing":
            result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        else:
            result = gray
        
        # Convert back to BGR if original was color
        if len(image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return result


class MorphologyGradientModule(BaseCVModule):
    """Morphological gradient (difference between dilation and erosion)"""
    
    @property
    def module_id(self) -> str:
        return "morphology_gradient"
    
    @property
    def display_name(self) -> str:
        return "Morphological Gradient"
    
    @property
    def description(self) -> str:
        return "Compute morphological gradient (dilation - erosion)"
    
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
                "max": 31,
                "description": "Size of the structuring element kernel"
            },
            {
                "name": "kernel_shape",
                "type": "select",
                "default": "rectangle",
                "options": ["rectangle", "ellipse", "cross"],
                "description": "Shape of the structuring element"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        kernel_size = kwargs.get("kernel_size", 5)
        kernel_shape = kwargs.get("kernel_shape", "rectangle")
        
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create structuring element
        if kernel_shape == "rectangle":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        elif kernel_shape == "ellipse":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        elif kernel_shape == "cross":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Compute morphological gradient
        result = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Convert back to BGR if original was color
        if len(image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return result

