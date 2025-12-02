"""
Base abstractions for CV modules
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np


class BaseCVModule(ABC):
    """Abstract base class for all computer vision operation modules"""
    
    @property
    @abstractmethod
    def module_id(self) -> str:
        """Unique identifier for the module (e.g., 'blur', 'edge_detection')"""
        pass
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable display name (e.g., 'Gaussian Blur')"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the module does"""
        pass
    
    @property
    @abstractmethod
    def category(self) -> str:
        """Category of the module (e.g., 'filter', 'transform', 'detection')"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> List[Dict[str, Any]]:
        """List of parameter definitions
        
        Returns:
            List of parameter dicts with keys:
            - name: str - parameter name
            - type: str - parameter type ('int', 'float', 'str', 'bool', 'select')
            - default: Any - default value
            - min: Optional[float] - minimum value (for numeric types)
            - max: Optional[float] - maximum value (for numeric types)
            - options: Optional[List] - options for 'select' type
            - description: str - parameter description
        """
        pass
    
    @abstractmethod
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Process the input image
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            **kwargs: Module-specific parameters
            
        Returns:
            Processed image as numpy array (BGR format)
        """
        pass
    
    def validate_parameters(self, **kwargs) -> Dict[str, Any]:
        """Validate and normalize parameters
        
        Args:
            **kwargs: Input parameters
            
        Returns:
            Validated and normalized parameters
            
        Raises:
            ValueError: If parameters are invalid
        """
        validated = {}
        param_defs = {p["name"]: p for p in self.parameters}
        
        for param_name, param_def in param_defs.items():
            value = kwargs.get(param_name, param_def.get("default"))
            
            # Type conversion
            param_type = param_def["type"]
            if param_type == "int":
                value = int(value)
                if "min" in param_def:
                    value = max(value, int(param_def["min"]))
                if "max" in param_def:
                    value = min(value, int(param_def["max"]))
            elif param_type == "float":
                value = float(value)
                if "min" in param_def:
                    value = max(value, float(param_def["min"]))
                if "max" in param_def:
                    value = min(value, float(param_def["max"]))
            elif param_type == "bool":
                value = bool(value)
            elif param_type == "select":
                if value not in param_def.get("options", []):
                    raise ValueError(f"Invalid value '{value}' for parameter '{param_name}'. Must be one of {param_def.get('options')}")
            
            validated[param_name] = value
        
        return validated
    
    def get_schema(self) -> Dict[str, Any]:
        """Get complete module schema for API/UI"""
        return {
            "id": self.module_id,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category,
            "parameters": self.parameters
        }

