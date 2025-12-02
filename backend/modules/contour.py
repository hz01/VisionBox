"""
Contour detection and analysis modules
"""
import cv2
import numpy as np
from core.base_module import BaseCVModule


class ContourDetectionModule(BaseCVModule):
    """Detect and draw contours"""
    
    @property
    def module_id(self) -> str:
        return "contour_detection"
    
    @property
    def display_name(self) -> str:
        return "Contour Detection"
    
    @property
    def description(self) -> str:
        return "Detect and draw contours in the image"
    
    @property
    def category(self) -> str:
        return "detection"
    
    @property
    def parameters(self):
        return [
            {
                "name": "mode",
                "type": "select",
                "default": "external",
                "options": ["external", "list", "tree", "ccomp"],
                "description": "Contour retrieval mode"
            },
            {
                "name": "method",
                "type": "select",
                "default": "simple",
                "options": ["simple", "none", "tc89_l1", "tc89_kcos"],
                "description": "Contour approximation method"
            },
            {
                "name": "draw_contours",
                "type": "bool",
                "default": True,
                "description": "Draw detected contours on the image"
            },
            {
                "name": "min_area",
                "type": "int",
                "default": 0,
                "min": 0,
                "max": 1000000,
                "description": "Minimum contour area to filter small contours"
            },
            {
                "name": "max_contours",
                "type": "int",
                "default": 100,
                "min": 1,
                "max": 1000,
                "description": "Maximum number of contours to draw"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        mode = kwargs.get("mode", "external")
        method = kwargs.get("method", "simple")
        draw_contours = kwargs.get("draw_contours", True)
        min_area = kwargs.get("min_area", 0)
        max_contours = kwargs.get("max_contours", 100)
        
        # Map mode string to OpenCV constant
        mode_map = {
            "external": cv2.RETR_EXTERNAL,
            "list": cv2.RETR_LIST,
            "tree": cv2.RETR_TREE,
            "ccomp": cv2.RETR_CCOMP
        }
        retrieval_mode = mode_map.get(mode, cv2.RETR_EXTERNAL)
        
        # Map method string to OpenCV constant
        method_map = {
            "simple": cv2.CHAIN_APPROX_SIMPLE,
            "none": cv2.CHAIN_APPROX_NONE,
            "tc89_l1": cv2.CHAIN_APPROX_TC89_L1,
            "tc89_kcos": cv2.CHAIN_APPROX_TC89_KCOS
        }
        approx_method = method_map.get(method, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, hierarchy = cv2.findContours(binary, retrieval_mode, approx_method)
        
        # Filter contours by area
        if min_area > 0:
            contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        # Sort by area (largest first) and limit
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_contours]
        
        # Draw contours on original image
        result = image.copy()
        if draw_contours:
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        return result


class ContourAnalysisModule(BaseCVModule):
    """Detect contours and draw bounding boxes/centroids"""
    
    @property
    def module_id(self) -> str:
        return "contour_analysis"
    
    @property
    def display_name(self) -> str:
        return "Contour Analysis"
    
    @property
    def description(self) -> str:
        return "Detect contours and draw bounding boxes, centroids, and area information"
    
    @property
    def category(self) -> str:
        return "detection"
    
    @property
    def parameters(self):
        return [
            {
                "name": "draw_boxes",
                "type": "bool",
                "default": True,
                "description": "Draw bounding boxes around contours"
            },
            {
                "name": "draw_centroids",
                "type": "bool",
                "default": True,
                "description": "Draw centroids of contours"
            },
            {
                "name": "min_area",
                "type": "int",
                "default": 100,
                "min": 0,
                "max": 1000000,
                "description": "Minimum contour area to filter small contours"
            },
            {
                "name": "max_contours",
                "type": "int",
                "default": 50,
                "min": 1,
                "max": 500,
                "description": "Maximum number of contours to analyze"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        draw_boxes = kwargs.get("draw_boxes", True)
        draw_centroids = kwargs.get("draw_centroids", True)
        min_area = kwargs.get("min_area", 100)
        max_contours = kwargs.get("max_contours", 50)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        if min_area > 0:
            contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        # Sort by area (largest first) and limit
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_contours]
        
        # Draw on original image
        result = image.copy()
        
        for i, contour in enumerate(contours):
            # Draw bounding box
            if draw_boxes:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Calculate and draw centroid
            if draw_centroids:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(result, (cx, cy), 5, (255, 0, 0), -1)
                    
                    # Draw area text
                    area = cv2.contourArea(contour)
                    cv2.putText(result, f"{int(area)}", (cx - 20, cy - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result

