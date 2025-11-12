"""
Configuration file for AgriMind ML Inference
Contains all hyperparameters, paths, and model configurations.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

@dataclass
class DatasetConfig:
    """Configuration for individual datasets"""
    name: str
    path: Path
    classes: List[str]
    image_size: Tuple[int, int]
    
# Dataset configurations
DATASETS = {
    "plantvillage": DatasetConfig(
        name="PlantVillage",
        path=DATA_DIR / "PlantVillage",
        classes=[],  # Will be auto-detected
        image_size=(224, 224)
    ),
    "crop_diseases": DatasetConfig(
        name="Crop Diseases",
        path=DATA_DIR / "Crop Diseases",
        classes=[],
        image_size=(224, 224)
    ),
    "rice_leaf_diseases": DatasetConfig(
        name="Rice Leaf Diseases",
        path=DATA_DIR / "rice_leaf_diseases",
        classes=[],
        image_size=(224, 224)
    )
}

@dataclass
class ModelConfig:
    """Model configuration"""
    # Model architecture
    backbone: str = "efficientnet_b3"
    pretrained: bool = True
    num_classes: int = 100  # Will be updated based on datasets
    dropout_rate: float = 0.3
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 30
    early_stopping_patience: int = 7
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    
    # Data augmentation
    use_mixup: bool = True
    use_cutmix: bool = True
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Validation
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Model saving
    save_best_only: bool = True
    monitor_metric: str = "val_accuracy"
    mode: str = "max"

@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    # Basic augmentations
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.1
    rotation_limit: int = 20
    
    # Color augmentations
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    saturation_limit: float = 0.2
    hue_shift_limit: int = 20
    
    # Geometric augmentations
    shift_limit: float = 0.1
    scale_limit: float = 0.1
    
    # Advanced augmentations
    gaussian_blur: float = 0.1
    motion_blur: float = 0.1
    gaussian_noise: float = 0.1
    
    # Normalization (ImageNet stats)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_image_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: List[str] = None
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Global configurations
MODEL_CONFIG = ModelConfig()
AUGMENTATION_CONFIG = AugmentationConfig()
API_CONFIG = APIConfig()

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "agrimind.log"),
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
}

# Environment variables
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "agrimind-plant-disease")
CUDA_DEVICE = os.getenv("CUDA_DEVICE", "0")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Model paths
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
LATEST_MODEL_PATH = MODELS_DIR / "latest_model.pth"
ONNX_MODEL_PATH = MODELS_DIR / "model.onnx"

# Class mapping file
CLASS_MAPPING_PATH = MODELS_DIR / "class_mapping.json"
