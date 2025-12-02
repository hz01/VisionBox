"""
Feature detection modules (SIFT, SURF, ORB)
"""
import cv2
import numpy as np
from core.base_module import BaseCVModule


class SIFTFeatureDetectionModule(BaseCVModule):
    """SIFT (Scale-Invariant Feature Transform) feature detection"""
    
    @property
    def module_id(self) -> str:
        return "sift_features"
    
    @property
    def display_name(self) -> str:
        return "SIFT Features"
    
    @property
    def description(self) -> str:
        return "Detect and draw SIFT keypoints and descriptors"
    
    @property
    def category(self) -> str:
        return "detection"
    
    @property
    def parameters(self):
        return [
            {
                "name": "n_features",
                "type": "int",
                "default": 0,
                "min": 0,
                "max": 10000,
                "description": "Number of best features to retain (0 = unlimited)"
            },
            {
                "name": "contrast_threshold",
                "type": "float",
                "default": 0.04,
                "min": 0.0,
                "max": 1.0,
                "description": "Contrast threshold for filtering weak features"
            },
            {
                "name": "edge_threshold",
                "type": "float",
                "default": 10.0,
                "min": 1.0,
                "max": 50.0,
                "description": "Edge threshold for filtering edge-like features"
            },
            {
                "name": "sigma",
                "type": "float",
                "default": 1.6,
                "min": 0.5,
                "max": 5.0,
                "description": "Sigma of Gaussian applied to input image"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        n_features = kwargs.get("n_features", 0)
        contrast_threshold = kwargs.get("contrast_threshold", 0.04)
        edge_threshold = kwargs.get("edge_threshold", 10.0)
        sigma = kwargs.get("sigma", 1.6)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create SIFT detector
        sift = cv2.SIFT_create(
            nfeatures=n_features if n_features > 0 else 0,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        # Draw keypoints on the image
        result = cv2.drawKeypoints(
            image,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        return result


class ORBFeatureDetectionModule(BaseCVModule):
    """ORB (Oriented FAST and Rotated BRIEF) feature detection"""
    
    @property
    def module_id(self) -> str:
        return "orb_features"
    
    @property
    def display_name(self) -> str:
        return "ORB Features"
    
    @property
    def description(self) -> str:
        return "Detect and draw ORB keypoints and descriptors"
    
    @property
    def category(self) -> str:
        return "detection"
    
    @property
    def parameters(self):
        return [
            {
                "name": "n_features",
                "type": "int",
                "default": 500,
                "min": 100,
                "max": 10000,
                "description": "Maximum number of features to retain"
            },
            {
                "name": "scale_factor",
                "type": "float",
                "default": 1.2,
                "min": 1.1,
                "max": 2.0,
                "description": "Pyramid decimation ratio"
            },
            {
                "name": "n_levels",
                "type": "int",
                "default": 8,
                "min": 1,
                "max": 16,
                "description": "Number of pyramid levels"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        n_features = kwargs.get("n_features", 500)
        scale_factor = kwargs.get("scale_factor", 1.2)
        n_levels = kwargs.get("n_levels", 8)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create ORB detector
        orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels
        )
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        # Draw keypoints on the image
        result = cv2.drawKeypoints(
            image,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=0
        )
        
        return result


class FASTFeatureDetectionModule(BaseCVModule):
    """FAST (Features from Accelerated Segment Test) corner detection"""
    
    @property
    def module_id(self) -> str:
        return "fast_features"
    
    @property
    def display_name(self) -> str:
        return "FAST Corners"
    
    @property
    def description(self) -> str:
        return "Detect corners using FAST algorithm"
    
    @property
    def category(self) -> str:
        return "detection"
    
    @property
    def parameters(self):
        return [
            {
                "name": "threshold",
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 100,
                "description": "Threshold on difference between intensity of center pixel and pixels on a circle"
            },
            {
                "name": "nonmax_suppression",
                "type": "bool",
                "default": True,
                "description": "Apply non-maximum suppression"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        threshold = kwargs.get("threshold", 10)
        nonmax_suppression = kwargs.get("nonmax_suppression", True)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create FAST detector
        fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmax_suppression)
        
        # Detect keypoints
        keypoints = fast.detect(gray, None)
        
        # Draw keypoints on the image
        result = cv2.drawKeypoints(
            image,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=0
        )
        
        return result

