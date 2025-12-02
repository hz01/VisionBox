"""
Image utility functions
"""
import numpy as np
import cv2
from PIL import Image
import io
import base64
from typing import Tuple, Optional


def numpy_to_base64(image: np.ndarray, format: str = "PNG") -> str:
    """Convert numpy array image to base64 string
    
    Args:
        image: Image as numpy array (BGR format)
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        Base64 encoded string
    """
    # Convert BGR to RGB for PIL
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    pil_image = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


def base64_to_numpy(base64_str: str) -> np.ndarray:
    """Convert base64 string to numpy array
    
    Args:
        base64_str: Base64 encoded image string (with or without data URI prefix)
        
    Returns:
        Image as numpy array (BGR format)
    """
    # Remove data URI prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    img_data = base64.b64decode(base64_str)
    pil_image = Image.open(io.BytesIO(img_data))
    
    # Convert to RGB if needed
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    # Convert to numpy array and then to BGR for OpenCV
    img_array = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_bgr


def file_to_numpy(file_bytes: bytes) -> np.ndarray:
    """Convert uploaded file bytes to numpy array
    
    Args:
        file_bytes: File bytes
        
    Returns:
        Image as numpy array (BGR format)
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Could not decode image file")
    
    return image


def numpy_to_bytes(image: np.ndarray, format: str = "PNG") -> bytes:
    """Convert numpy array to bytes
    
    Args:
        image: Image as numpy array (BGR format)
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        Image as bytes
    """
    # Convert BGR to RGB for encoding
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Determine extension
    ext = ".png" if format.upper() == "PNG" else ".jpg"
    
    # Encode
    success, buffer = cv2.imencode(ext, image)
    if not success:
        raise ValueError(f"Could not encode image as {format}")
    
    return buffer.tobytes()


def validate_image(image: np.ndarray) -> Tuple[bool, Optional[str]]:
    """Validate image array
    
    Args:
        image: Image as numpy array
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if image is None:
        return False, "Image is None"
    
    if not isinstance(image, np.ndarray):
        return False, "Image must be a numpy array"
    
    if image.size == 0:
        return False, "Image is empty"
    
    if len(image.shape) < 2 or len(image.shape) > 3:
        return False, f"Invalid image shape: {image.shape}"
    
    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        return False, f"Invalid number of channels: {image.shape[2]}"
    
    return True, None

