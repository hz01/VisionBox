"""
Image information and analysis module
"""
import cv2
import numpy as np
import logging
from core.base_module import BaseCVModule

logger = logging.getLogger(__name__)


class ImageInfoModule(BaseCVModule):
    """Display image information and statistics"""
    
    # Class variable to store info for the last processed image
    last_info = None
    
    @property
    def module_id(self) -> str:
        return "image_info"
    
    @property
    def display_name(self) -> str:
        return "Image Info"
    
    @property
    def description(self) -> str:
        return "Display detailed information about the image (dimensions, channels, statistics, etc.)"
    
    @property
    def category(self) -> str:
        return "analysis"
    
    @property
    def parameters(self):
        return [
            {
                "name": "include_statistics",
                "type": "bool",
                "default": True,
                "description": "Include pixel value statistics (min, max, mean, std)"
            },
            {
                "name": "include_histogram",
                "type": "bool",
                "default": False,
                "description": "Include histogram information"
            },
            {
                "name": "include_color_info",
                "type": "bool",
                "default": True,
                "description": "Include color space and channel information"
            },
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Analyze image and return it unchanged (info is logged/returned in metadata)
        The actual info display should be handled by the frontend
        """
        include_statistics = kwargs.get("include_statistics", True)
        include_histogram = kwargs.get("include_histogram", False)
        include_color_info = kwargs.get("include_color_info", True)
        
        # Get basic dimensions
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        dtype = str(image.dtype)
        
        # Calculate image size in bytes
        image_size_bytes = image.nbytes
        image_size_mb = image_size_bytes / (1024 * 1024)
        
        # Determine color space
        if channels == 1:
            color_space = "Grayscale"
        elif channels == 3:
            color_space = "BGR (OpenCV default)"
        elif channels == 4:
            color_space = "BGRA (with alpha)"
        else:
            color_space = f"Multi-channel ({channels} channels)"
        
        # Build info dictionary
        info = {
            "dimensions": {
                "width": int(width),
                "height": int(height),
                "channels": int(channels),
                "aspect_ratio": round(width / height, 4) if height > 0 else 0
            },
            "format": {
                "dtype": dtype,
                "color_space": color_space,
                "size_bytes": int(image_size_bytes),
                "size_mb": round(image_size_mb, 2)
            }
        }
        
        # Add statistics if requested
        if include_statistics:
            if channels == 1:
                info["statistics"] = {
                    "min": float(np.min(image)),
                    "max": float(np.max(image)),
                    "mean": float(np.mean(image)),
                    "std": float(np.std(image)),
                    "median": float(np.median(image))
                }
            else:
                info["statistics"] = {}
                for i, channel_name in enumerate(["B", "G", "R"] if channels == 3 else [f"Channel_{i}" for i in range(channels)]):
                    channel = image[:, :, i]
                    info["statistics"][channel_name] = {
                        "min": float(np.min(channel)),
                        "max": float(np.max(channel)),
                        "mean": float(np.mean(channel)),
                        "std": float(np.std(channel)),
                        "median": float(np.median(channel))
                    }
        
        # Add histogram info if requested
        if include_histogram:
            if channels == 1:
                hist = cv2.calcHist([image], [0], None, [256], [0, 256])
                info["histogram"] = {
                    "bins": 256,
                    "range": [0, 256],
                    "non_zero_bins": int(np.count_nonzero(hist))
                }
            else:
                info["histogram"] = {}
                for i, channel_name in enumerate(["B", "G", "R"] if channels == 3 else [f"Channel_{i}" for i in range(channels)]):
                    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                    info["histogram"][channel_name] = {
                        "bins": 256,
                        "range": [0, 256],
                        "non_zero_bins": int(np.count_nonzero(hist))
                    }
        
        # Add color info if requested
        if include_color_info:
            info["color_info"] = {
                "color_space": color_space,
                "channels": int(channels),
                "has_alpha": channels == 4
            }
        
        # Log info (for debugging)
        logger.info(f"Image Info: {width}x{height}, {channels} channels, {dtype}, {image_size_mb:.2f} MB")
        
        # Store info in class variable so it can be accessed by the pipeline executor
        ImageInfoModule.last_info = info
        
        return image

