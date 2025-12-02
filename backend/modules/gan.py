"""
GAN (Generative Adversarial Network) modules for image generation
"""
import cv2
import numpy as np
import logging
from typing import Optional
from core.base_module import BaseCVModule
import os
from utils.model_manager import (
    get_available_models, get_model_path, download_model, is_model_cached
)

logger = logging.getLogger(__name__)


def _check_torch_available():
    """Check if PyTorch is available"""
    try:
        import torch
        return True
    except ImportError:
        return False


def _get_device():
    """Get the appropriate device (CUDA if available, else CPU)"""
    if _check_torch_available():
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    return 'cpu'


def _check_tensorflow_available():
    """Check if TensorFlow is available"""
    try:
        import tensorflow as tf
        return True
    except ImportError:
        return False


class GANImageGenerationModule(BaseCVModule):
    """Generate images using GAN models"""
    
    @property
    def module_id(self) -> str:
        return "gan_generation"
    
    @property
    def display_name(self) -> str:
        return "GAN Image Generation"
    
    @property
    def description(self) -> str:
        return "Generate images using Generative Adversarial Networks (GANs)"
    
    @property
    def category(self) -> str:
        return "generation"
    
    @property
    def parameters(self):
        # Get all available models from model manager (show all, filter at runtime)
        available_models_info = get_available_models()
        model_options = [model_info["id"] for model_info in available_models_info]
        
        if not model_options:
            model_options = ["noise_pattern"]
        
        return [
            {
                "name": "model",
                "type": "select",
                "default": model_options[0],
                "options": model_options,
                "description": "GAN model to use for generation (will auto-download if needed)"
            },
            # Generation parameters
            {
                "name": "width",
                "type": "int",
                "default": 512,
                "min": 64,
                "max": 2048,
                "description": "Width of generated image"
            },
            {
                "name": "height",
                "type": "int",
                "default": 512,
                "min": 64,
                "max": 2048,
                "description": "Height of generated image"
            },
            {
                "name": "seed",
                "type": "int",
                "default": -1,
                "min": -1,
                "max": 999999,
                "description": "Random seed (-1 for random)"
            },
            # DCGAN parameters
            {
                "name": "dcgan_latent_dim",
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 512,
                "description": "[DCGAN] Latent dimension size"
            },
            # StyleGAN3 parameters
            {
                "name": "stylegan3_variant",
                "type": "select",
                "default": "ffhq",
                "options": ["ffhq", "metfaces", "afhq", "cars"],
                "description": "[StyleGAN3] Model variant to use"
            },
            {
                "name": "stylegan3_truncation",
                "type": "float",
                "default": 0.7,
                "min": 0.0,
                "max": 1.0,
                "description": "[StyleGAN3] Truncation value for latent space"
            },
            # Progressive GAN parameters
            {
                "name": "progressive_gan_latent_dim",
                "type": "int",
                "default": 512,
                "min": 64,
                "max": 1024,
                "description": "[Progressive GAN] Latent dimension size"
            },
            # BigGAN parameters
            {
                "name": "biggan_latent_dim",
                "type": "int",
                "default": 128,
                "min": 64,
                "max": 512,
                "description": "[BigGAN] Latent dimension size"
            },
            {
                "name": "biggan_class_id",
                "type": "int",
                "default": 0,
                "min": 0,
                "max": 1000,
                "description": "[BigGAN] Class ID for conditional generation"
            },
            # WGAN parameters
            {
                "name": "wgan_latent_dim",
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 512,
                "description": "[WGAN] Latent dimension size"
            },
            # Noise pattern parameters
            {
                "name": "noise_type",
                "type": "select",
                "default": "perlin",
                "options": ["perlin", "simplex", "fractal", "cellular"],
                "description": "[Noise Pattern] Type of noise to generate"
            },
            {
                "name": "noise_scale",
                "type": "float",
                "default": 0.1,
                "min": 0.01,
                "max": 1.0,
                "description": "[Noise Pattern] Scale of noise pattern"
            },
            {
                "name": "noise_octaves",
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 8,
                "description": "[Noise Pattern] Number of octaves for fractal noise"
            },
            {
                "name": "color_mode",
                "type": "select",
                "default": "rgb",
                "options": ["rgb", "grayscale", "hsv"],
                "description": "Color mode for generated image"
            }
        ]
    
    def _generate_noise_pattern(self, width: int, height: int, **kwargs) -> np.ndarray:
        """Generate image from noise patterns (fallback when no GAN models available)"""
        noise_type = kwargs.get("noise_type", "perlin")
        noise_scale = kwargs.get("noise_scale", 0.1)
        noise_octaves = kwargs.get("noise_octaves", 4)
        color_mode = kwargs.get("color_mode", "rgb")
        seed = kwargs.get("seed", -1)
        
        if seed >= 0:
            np.random.seed(seed)
        
        if noise_type == "perlin" or noise_type == "simplex":
            # Simple Perlin-like noise using gradients
            x = np.linspace(0, width * noise_scale, width)
            y = np.linspace(0, height * noise_scale, height)
            X, Y = np.meshgrid(x, y)
            
            # Generate noise using multiple octaves
            noise = np.zeros((height, width))
            for i in range(noise_octaves):
                freq = 2 ** i
                amp = 1.0 / freq
                noise += amp * np.sin(X * freq) * np.cos(Y * freq)
            
            # Normalize to 0-255
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)
            noise = (noise * 255).astype(np.uint8)
            
        elif noise_type == "fractal":
            # Fractal noise
            noise = np.zeros((height, width))
            for i in range(noise_octaves):
                freq = 2 ** i
                amp = 1.0 / freq
                layer = np.random.rand(height, width) * amp
                noise += layer
            
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)
            noise = (noise * 255).astype(np.uint8)
            
        else:  # cellular
            # Cellular/Voronoi-like pattern
            num_points = 20
            points = np.random.rand(num_points, 2) * [width, height]
            noise = np.zeros((height, width))
            
            for y in range(height):
                for x in range(width):
                    distances = np.sqrt(np.sum((points - [x, y]) ** 2, axis=1))
                    noise[y, x] = np.min(distances)
            
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)
            noise = (noise * 255).astype(np.uint8)
        
        # Convert to color based on mode
        if color_mode == "grayscale":
            result = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
        elif color_mode == "hsv":
            # Create HSV image
            hsv = np.zeros((height, width, 3), dtype=np.uint8)
            hsv[:, :, 0] = (noise * 180 / 255).astype(np.uint8)  # Hue
            hsv[:, :, 1] = 255  # Saturation
            hsv[:, :, 2] = noise  # Value
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:  # rgb
            # Create RGB from noise
            result = np.zeros((height, width, 3), dtype=np.uint8)
            result[:, :, 0] = noise  # B
            result[:, :, 1] = (noise * 0.7).astype(np.uint8)  # G
            result[:, :, 2] = (noise * 0.5).astype(np.uint8)  # R
        
        return result
    
    def _get_or_download_model(self, model_id: str, variant: Optional[str] = None) -> str:
        """Get model path, downloading if necessary (with optional variant for StyleGAN3)"""
        # Check if cached
        model_path = get_model_path(model_id, variant)
        if model_path:
            return str(model_path)
        
        # Download model
        try:
            model_path = download_model(model_id, variant=variant)
            return str(model_path)
        except Exception as e:
            raise ValueError(
                f"Failed to get model {model_id} (variant: {variant or 'default'}): {str(e)}\n"
                f"Model will be downloaded automatically on first use. "
                f"Make sure you have the required framework installed."
            )
    
    def _generate_dcgan(self, width: int, height: int, **kwargs) -> np.ndarray:
        """Generate image using DCGAN (requires PyTorch and model)"""
        if not _check_torch_available():
            raise ValueError(
                "PyTorch is not installed. Install it with: pip install torch\n"
                "Or use 'noise_pattern' model instead."
            )
        
        import torch
        import torch.nn as nn
        
        latent_dim = kwargs.get("dcgan_latent_dim", 100)
        seed = kwargs.get("seed", -1)
        
        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Get or download model
        model_path = self._get_or_download_model("dcgan")
        
        # Determine device (CUDA if available, else CPU)
        device = _get_device()
        logger.info(f"Using device: {device}")
        
        # Generate random noise
        z = torch.randn(1, latent_dim)
        if device == 'cuda':
            z = z.cuda()
        
        # Load pre-trained model
        try:
            model = torch.load(model_path, map_location=device)
            if hasattr(model, 'eval'):
                model.eval()
            if device == 'cuda' and hasattr(model, 'cuda'):
                model = model.cuda()
            
            with torch.no_grad():
                if callable(model):
                    fake_image = model(z)
                else:
                    # Try to find generator in model dict
                    if isinstance(model, dict):
                        generator = model.get('generator') or model.get('G')
                        if generator:
                            if device == 'cuda' and hasattr(generator, 'cuda'):
                                generator = generator.cuda()
                            fake_image = generator(z)
                        else:
                            raise ValueError("Could not find generator in model file")
                    else:
                        fake_image = model(z)
                
                # Convert to numpy (move to CPU first)
                if isinstance(fake_image, torch.Tensor):
                    img = fake_image[0].cpu().numpy()
                else:
                    img = fake_image[0]
                
                # Handle different tensor shapes
                if len(img.shape) == 3:
                    img = np.transpose(img, (1, 2, 0))
                
                # Normalize to 0-255
                img = (img - img.min()) / (img.max() - img.min() + 1e-10)
                img = (img * 255).astype(np.uint8)
                
                # Resize to requested dimensions
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    img = cv2.resize(img, (width, height))
                    if img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                return img
        except Exception as e:
            raise ValueError(f"Failed to load DCGAN model: {str(e)}")
    
    def _generate_stylegan3(self, width: int, height: int, **kwargs) -> np.ndarray:
        """Generate image using StyleGAN3"""
        variant = kwargs.get("stylegan3_variant", "ffhq")
        return self._generate_stylegan_generic("stylegan3", width, height, variant=variant, **kwargs)
    
    def _generate_stylegan_generic(self, model_id: str, width: int, height: int, variant: Optional[str] = None, **kwargs) -> np.ndarray:
        """Generic StyleGAN generation"""
        if not _check_torch_available():
            raise ValueError(
                "PyTorch is not installed. Install it with: pip install torch\n"
                "Or use 'noise_pattern' model instead."
            )
        
        import torch
        
        truncation_param = f"{model_id}_truncation"
        truncation = kwargs.get(truncation_param, 0.7)
        seed = kwargs.get("seed", -1)
        
        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Get or download model (with variant for StyleGAN3)
        model_path = self._get_or_download_model(model_id, variant=variant)
        
        # Determine device (CUDA if available, else CPU)
        device = _get_device()
        logger.info(f"Using device: {device} for {model_id}")
        
        try:
            # Check if torch_utils is available (required for StyleGAN3)
            try:
                import torch_utils
            except ImportError:
                raise ValueError(
                    f"{model_id.upper()} requires the official StyleGAN3 implementation.\n"
                    "To use StyleGAN3, you need to install the official repository:\n"
                    "  git clone https://github.com/NVlabs/stylegan3.git\n"
                    "  cd stylegan3\n"
                    "  pip install -r requirements.txt\n"
                    "Then add the stylegan3 directory to your PYTHONPATH.\n\n"
                    "Alternatively, use 'noise_pattern' or 'dcgan' models which don't require additional setup."
                )
            
            # Load model (StyleGAN models are typically pickle files)
            import pickle
            import sys
            import os
            
            # Try to load the model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Generate random latent vector
            z = torch.randn(1, 512)  # StyleGAN typically uses 512-dim latent
            if device == 'cuda':
                z = z.cuda()
            
            # This is a placeholder - actual StyleGAN generation would require
            # the official StyleGAN implementation's generator
            raise NotImplementedError(
                f"{model_id.upper()} model generation is not yet fully implemented. "
                "The model file was loaded, but image generation requires the official StyleGAN3 generator. "
                "Please use 'noise_pattern' or 'dcgan' models for now, or install the full StyleGAN3 implementation."
            )
        except NotImplementedError:
            raise
        except ValueError as ve:
            # Re-raise ValueError (our custom error messages)
            raise
        except ImportError as ie:
            raise ValueError(
                f"{model_id.upper()} requires additional dependencies: {str(ie)}\n"
                "StyleGAN3 models need the official StyleGAN3 implementation. "
                "Please install it or use 'noise_pattern'/'dcgan' models instead."
            )
        except Exception as e:
            error_msg = str(e)
            if "torch_utils" in error_msg or "No module named 'torch_utils'" in error_msg:
                raise ValueError(
                    f"{model_id.upper()} requires the official StyleGAN3 implementation.\n"
                    "To use StyleGAN3, you need to install the official repository:\n"
                    "  git clone https://github.com/NVlabs/stylegan3.git\n"
                    "  cd stylegan3\n"
                    "  pip install -r requirements.txt\n"
                    "Then add the stylegan3 directory to your PYTHONPATH.\n\n"
                    "Alternatively, use 'noise_pattern' or 'dcgan' models which don't require additional setup."
                )
            raise ValueError(f"Failed to load {model_id} model: {error_msg}")
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate a new image using GAN.
        Note: This generates a new image, ignoring the input image.
        """
        model = kwargs.get("model", "noise_pattern")
        width = kwargs.get("width", 512)
        height = kwargs.get("height", 512)
        
        # Remove width and height from kwargs to avoid duplicate arguments
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["width", "height"]}
        
        # Generate based on selected model
        if model == "noise_pattern":
            return self._generate_noise_pattern(width, height, **filtered_kwargs)
        elif model == "dcgan":
            return self._generate_dcgan(width, height, **filtered_kwargs)
        elif model == "stylegan3":
            return self._generate_stylegan3(width, height, **filtered_kwargs)
        elif model == "progressive_gan":
            # Progressive GAN implementation - use DCGAN as fallback
            logger.warning("Progressive GAN full implementation requires model files. Using DCGAN as fallback.")
            return self._generate_dcgan(width, height, **filtered_kwargs)
        elif model == "biggan":
            raise NotImplementedError(
                "BigGAN is not yet implemented. "
                "Please use 'noise_pattern' or 'dcgan' models."
            )
        elif model == "wgan":
            raise NotImplementedError(
                "WGAN is not yet implemented. "
                "Please use 'noise_pattern' or 'dcgan' models."
            )
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def validate_parameters(self, **kwargs) -> dict:
        """Override to handle special validation for generation module"""
        validated = super().validate_parameters(**kwargs)
        
        # Ensure width and height are reasonable
        width = validated.get("width", 512)
        height = validated.get("height", 512)
        
        # Round to nearest even number for some models
        validated["width"] = (width // 2) * 2
        validated["height"] = (height // 2) * 2
        
        return validated

