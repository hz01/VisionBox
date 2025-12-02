"""
Model manager for downloading and caching GAN models
"""
import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, List
import urllib.request
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Model cache directory
CACHE_DIR = Path.home() / ".visionbox" / "models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Model metadata file
METADATA_FILE = CACHE_DIR / "model_metadata.json"


# Available GAN models configuration
GAN_MODELS = {
    "noise_pattern": {
        "name": "Noise Pattern",
        "description": "Procedural noise-based generation (no dependencies)",
        "requires_download": False,
        "framework": None,
    },
    "dcgan": {
        "name": "DCGAN",
        "description": "Deep Convolutional GAN",
        "requires_download": True,
        "framework": "pytorch",
        "url": "https://github.com/pytorch/examples/raw/master/dcgan/main.py",  # Placeholder
        "model_file": "dcgan_generator.pth",
    },
    "stylegan3": {
        "name": "StyleGAN3",
        "description": "Style-based GAN v3",
        "requires_download": True,
        "framework": "pytorch",
        "variants": {
            "ffhq": {
                "name": "FFHQ (Faces)",
                "description": "High-quality face generation",
                "url": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl",
                "model_file": "stylegan3-r-ffhq-1024x1024.pkl",
                "resolution": 1024,
            },
            "metfaces": {
                "name": "MetFaces",
                "description": "Artistic face generation",
                "url": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-metfaces-1024x1024.pkl",
                "model_file": "stylegan3-r-metfaces-1024x1024.pkl",
                "resolution": 1024,
            },
            "afhq": {
                "name": "AFHQ (Animals)",
                "description": "Animal face generation (cats, dogs, wildlife)",
                "url": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl",
                "model_file": "stylegan3-r-afhqv2-512x512.pkl",
                "resolution": 512,
            },
            "cars": {
                "name": "Cars",
                "description": "Car generation",
                "url": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-t-afhqv2-512x512.pkl",  # Placeholder, actual cars model URL needed
                "model_file": "stylegan3-r-cars-512x512.pkl",
                "resolution": 512,
            },
        },
        # Default variant
        "url": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl",
        "model_file": "stylegan3-r-ffhq-1024x1024.pkl",
    },
    "progressive_gan": {
        "name": "Progressive GAN",
        "description": "Progressive Growing GAN",
        "requires_download": True,
        "framework": "tensorflow",
        "url": "",  # Placeholder
        "model_file": "progressive_gan.pb",
    },
    "biggan": {
        "name": "BigGAN",
        "description": "Large Scale GAN Training",
        "requires_download": True,
        "framework": "pytorch",
        "url": "",  # Placeholder
        "model_file": "biggan.pth",
    },
    "wgan": {
        "name": "WGAN",
        "description": "Wasserstein GAN",
        "requires_download": True,
        "framework": "pytorch",
        "url": "",  # Placeholder
        "model_file": "wgan.pth",
    },
}


def load_metadata() -> Dict:
    """Load model metadata from cache"""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load model metadata: {e}")
    return {}


def save_metadata(metadata: Dict):
    """Save model metadata to cache"""
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save model metadata: {e}")


def get_model_path(model_id: str, variant: Optional[str] = None) -> Optional[Path]:
    """Get the cached path for a model (with optional variant for StyleGAN3)"""
    metadata = load_metadata()
    
    # Check metadata first
    key = f"{model_id}:{variant}" if variant else model_id
    if key in metadata and "path" in metadata[key]:
        path = Path(metadata[key]["path"])
        if path.exists():
            return path
    
    # Fallback: check file system for StyleGAN3 variants
    if model_id == "stylegan3" and variant:
        if model_id not in GAN_MODELS:
            return None
        config = GAN_MODELS[model_id]
        if "variants" in config and variant in config["variants"]:
            variant_config = config["variants"][variant]
            model_file = variant_config["model_file"]
            cache_path = CACHE_DIR / model_id / variant / model_file
            if cache_path.exists():
                return cache_path
    
    return None


def is_model_cached(model_id: str, variant: Optional[str] = None) -> bool:
    """Check if a model is already cached (with optional variant for StyleGAN3)"""
    return get_model_path(model_id, variant) is not None


def download_model(model_id: str, variant: Optional[str] = None, progress_callback=None) -> Path:
    """Download and cache a GAN model (with optional variant for StyleGAN3)"""
    if model_id not in GAN_MODELS:
        raise ValueError(f"Unknown model: {model_id}")
    
    model_config = GAN_MODELS[model_id]
    
    if not model_config["requires_download"]:
        raise ValueError(f"Model {model_id} does not require downloading")
    
    # Handle StyleGAN3 variants
    if model_id == "stylegan3" and variant:
        if "variants" not in model_config or variant not in model_config["variants"]:
            raise ValueError(f"Unknown StyleGAN3 variant: {variant}")
        variant_config = model_config["variants"][variant]
        url = variant_config["url"]
        model_file = variant_config["model_file"]
        cache_path = CACHE_DIR / model_id / variant / model_file
    else:
        url = model_config.get("url", "")
        model_file = model_config["model_file"]
        cache_path = CACHE_DIR / model_id / model_file
    
    # Check if already cached
    cached_path = get_model_path(model_id, variant)
    if cached_path:
        logger.info(f"Model {model_id} (variant: {variant or 'default'}) already cached at {cached_path}")
        return cached_path
    
    # Check framework availability
    if model_config["framework"] == "pytorch":
        try:
            import torch
        except ImportError:
            raise ValueError(
                f"PyTorch is required for {model_id}. Install with: pip install torch"
            )
    elif model_config["framework"] == "tensorflow":
        try:
            import tensorflow as tf
        except ImportError:
            raise ValueError(
                f"TensorFlow is required for {model_id}. Install with: pip install tensorflow"
            )
    
    # Validate URL
    if not url:
        raise ValueError(
            f"Model {model_id} (variant: {variant or 'default'}) download URL not configured. "
            "This model is not yet available for automatic download."
        )
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {model_id} (variant: {variant or 'default'}) from {url}...")
    
    try:
        def report_progress(block_num, block_size, total_size):
            if progress_callback and total_size > 0:
                downloaded = block_num * block_size
                progress = min(downloaded / total_size, 1.0)
                progress_callback(progress)
        
        urllib.request.urlretrieve(url, cache_path, reporthook=report_progress)
        
        # Save metadata
        metadata = load_metadata()
        key = f"{model_id}:{variant}" if variant else model_id
        metadata[key] = {
            "path": str(cache_path),
            "downloaded": True,
            "url": url,
        }
        save_metadata(metadata)
        
        logger.info(f"Successfully downloaded {model_id} (variant: {variant or 'default'}) to {cache_path}")
        return cache_path
        
    except Exception as e:
        logger.error(f"Failed to download {model_id} (variant: {variant or 'default'}): {e}")
        raise ValueError(f"Failed to download model {model_id} (variant: {variant or 'default'}): {str(e)}")


def get_available_models() -> List[Dict]:
    """Get list of available GAN models with their status"""
    available = []
    
    for model_id, config in GAN_MODELS.items():
        model_info = {
            "id": model_id,
            "name": config["name"],
            "description": config["description"],
            "framework": config["framework"],
            "cached": is_model_cached(model_id),
            "requires_download": config["requires_download"],
        }
        available.append(model_info)
    
    return available


def get_model_info(model_id: str, variant: Optional[str] = None) -> Optional[Dict]:
    """Get information about a specific model (with optional variant for StyleGAN3)"""
    if model_id not in GAN_MODELS:
        return None
    
    config = GAN_MODELS[model_id]
    info = {
        "id": model_id,
        "name": config["name"],
        "description": config["description"],
        "framework": config["framework"],
        "cached": is_model_cached(model_id, variant),
        "requires_download": config["requires_download"],
        "path": str(get_model_path(model_id, variant)) if is_model_cached(model_id, variant) else None,
    }
    
    # Add variant information for StyleGAN3
    if model_id == "stylegan3" and "variants" in config:
        info["variants"] = {
            variant_id: {
                "name": variant_config["name"],
                "description": variant_config["description"],
                "resolution": variant_config.get("resolution", 1024),
                "cached": is_model_cached(model_id, variant_id),
            }
            for variant_id, variant_config in config["variants"].items()
        }
    
    return info

