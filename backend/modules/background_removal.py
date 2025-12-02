"""
Background removal modules using various models
"""
import cv2
import numpy as np
from core.base_module import BaseCVModule


def _check_mediapipe_available():
    """Check if MediaPipe is available at runtime"""
    try:
        import mediapipe as mp
        return True
    except ImportError:
        return False


class AutoBackgroundRemoverModule(BaseCVModule):
    """Automatic background removal using various models"""
    
    @property
    def module_id(self) -> str:
        return "auto_background_remover"
    
    @property
    def display_name(self) -> str:
        return "Auto Background Remover"
    
    @property
    def description(self) -> str:
        return "Remove background automatically using MediaPipe, U2Net, or other models"
    
    @property
    def category(self) -> str:
        return "detection"
    
    @property
    def parameters(self):
        available_models = []
        
        # MediaPipe is always available (will check at runtime)
        available_models.append("mediapipe")
        
        # Could add more models here in the future
        # if _check_u2net_available():
        #     available_models.append("u2net")
        
        return [
            {
                "name": "model",
                "type": "select",
                "default": "mediapipe",
                "options": available_models,
                "description": "Background removal model to use"
            },
            # MediaPipe parameters
            {
                "name": "mediapipe_model_selection",
                "type": "select",
                "default": "0",
                "options": ["0", "1"],
                "description": "[MediaPipe] Model selection (0=general, 1=landscape)"
            },
            {
                "name": "mediapipe_min_detection_confidence",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "[MediaPipe] Minimum confidence for detection"
            },
            # Common parameters
            {
                "name": "background_type",
                "type": "select",
                "default": "transparent",
                "options": ["transparent", "white", "black", "blur", "color"],
                "description": "Type of background replacement"
            },
            {
                "name": "background_color_r",
                "type": "int",
                "default": 255,
                "min": 0,
                "max": 255,
                "description": "[Color background] Red component (0-255)"
            },
            {
                "name": "background_color_g",
                "type": "int",
                "default": 255,
                "min": 0,
                "max": 255,
                "description": "[Color background] Green component (0-255)"
            },
            {
                "name": "background_color_b",
                "type": "int",
                "default": 255,
                "min": 0,
                "max": 255,
                "description": "[Color background] Blue component (0-255)"
            },
            {
                "name": "blur_strength",
                "type": "int",
                "default": 55,
                "min": 5,
                "max": 255,
                "description": "[Blur background] Blur kernel size (must be odd)"
            },
            {
                "name": "edge_smoothing",
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 10.0,
                "description": "Edge smoothing amount (0=no smoothing)"
            }
        ]
    
    def _remove_background_mediapipe(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Remove background using MediaPipe Selfie Segmentation"""
        if not _check_mediapipe_available():
            raise ValueError(
                "MediaPipe is not installed. Please install it with: pip install mediapipe"
            )
        
        import mediapipe as mp
        
        model_selection = int(kwargs.get("mediapipe_model_selection", "0"))
        min_detection_confidence = kwargs.get("mediapipe_min_detection_confidence", 0.5)
        background_type = kwargs.get("background_type", "transparent")
        blur_strength = kwargs.get("blur_strength", 55)
        edge_smoothing = kwargs.get("edge_smoothing", 0.0)
        bg_color_r = kwargs.get("background_color_r", 255)
        bg_color_g = kwargs.get("background_color_g", 255)
        bg_color_b = kwargs.get("background_color_b", 255)
        
        # Ensure blur strength is odd
        if blur_strength % 2 == 0:
            blur_strength += 1
        
        selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(rgb_image)
        
        if results.segmentation_mask is None:
            # No segmentation available, return original or error
            raise ValueError("Could not generate segmentation mask. Try adjusting parameters.")
        
        # Get mask
        mask = results.segmentation_mask
        
        # Apply edge smoothing if requested
        if edge_smoothing > 0:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            # Apply Gaussian blur for smoothing
            kernel_size = int(edge_smoothing * 2) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            mask_uint8 = cv2.GaussianBlur(mask_uint8, (kernel_size, kernel_size), 0)
            mask = mask_uint8.astype(np.float32) / 255.0
        
        # Convert mask to 3-channel
        mask_3channel = np.stack((mask,) * 3, axis=-1)
        
        # Apply background based on type
        if background_type == "transparent":
            # Create BGRA image
            result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            result[:, :, 3] = (mask * 255).astype(np.uint8)
            # Convert back to BGR (alpha will be lost but that's expected for pipeline)
            # For true transparency, would need to handle RGBA throughout pipeline
            # For now, use white background with transparency simulation
            white_bg = np.ones_like(image) * 255
            result_bgr = (mask_3channel * image + (1 - mask_3channel) * white_bg).astype(np.uint8)
            return result_bgr
        elif background_type == "white":
            background = np.ones_like(image) * 255
        elif background_type == "black":
            background = np.zeros_like(image)
        elif background_type == "blur":
            background = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
        else:  # color
            background = np.zeros_like(image)
            background[:, :, 0] = bg_color_b  # BGR format
            background[:, :, 1] = bg_color_g
            background[:, :, 2] = bg_color_r
        
        # Blend foreground with background
        result = (mask_3channel * image + (1 - mask_3channel) * background).astype(np.uint8)
        
        selfie_segmentation.close()
        return result
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        model = kwargs.get("model", "mediapipe")
        
        if model == "mediapipe":
            return self._remove_background_mediapipe(image, **kwargs)
        else:
            raise ValueError(f"Unknown model: {model}. Available models: mediapipe")


class BackgroundRemoverU2NetModule(BaseCVModule):
    """Background removal using U2Net model (placeholder for future implementation)"""
    
    @property
    def module_id(self) -> str:
        return "background_remover_u2net"
    
    @property
    def display_name(self) -> str:
        return "Background Remover (U2Net)"
    
    @property
    def description(self) -> str:
        return "Remove background using U2Net deep learning model (requires model file)"
    
    @property
    def category(self) -> str:
        return "detection"
    
    @property
    def parameters(self):
        return [
            {
                "name": "model_path",
                "type": "str",
                "default": "",
                "description": "Path to U2Net model file"
            },
            {
                "name": "background_type",
                "type": "select",
                "default": "transparent",
                "options": ["transparent", "white", "black", "blur"],
                "description": "Type of background replacement"
            }
        ]
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # Placeholder for U2Net implementation
        # Would require: torch, u2net model file, etc.
        raise NotImplementedError(
            "U2Net background removal is not yet implemented. "
            "Use MediaPipe model instead, or implement U2Net integration."
        )

