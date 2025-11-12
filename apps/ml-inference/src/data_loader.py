"""
Data loading utilities for plant disease detection.
"""

import os
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

from config import AUGMENTATION_CONFIG, BASE_DIR

class PlantDiseaseDataset(Dataset):
    """
    Dataset class for plant disease images.
    """
    
    def __init__(
        self, 
        data_dir: Path, 
        transform: Optional[Any] = None,
        class_mapping: Optional[Dict[str, int]] = None
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load or create class mapping
        if class_mapping is None:
            mapping_file = BASE_DIR / "models" / "class_mapping.json"
            if mapping_file.exists():
                with open(mapping_file, 'r') as f:
                    self.class_mapping = json.load(f)
            else:
                self.class_mapping = self._create_class_mapping()
        else:
            self.class_mapping = class_mapping
        
        self.classes = list(self.class_mapping.keys())
        self.num_classes = len(self.classes)
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
    def _create_class_mapping(self) -> Dict[str, int]:
        """Create class mapping from directory structure."""
        classes = []
        for item in self.data_dir.iterdir():
            if item.is_dir():
                classes.append(item.name)
        
        classes = sorted(classes)
        return {cls: idx for idx, cls in enumerate(classes)}
    
    def _load_samples(self):
        """Load all image paths and corresponding labels."""
        samples = []
        
        for class_name, class_idx in self.class_mapping.items():
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                continue
                
            for img_path in class_dir.glob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    samples.append((str(img_path), class_idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            if ALBUMENTATIONS_AVAILABLE and 'albumentations' in str(type(self.transform)):
                # Albumentations transform
                image_np = np.array(image)
                transformed = self.transform(image=image_np)
                image = transformed['image']
            else:
                # PyTorch transforms
                image = self.transform(image)
        
        return image, label


def get_train_transforms(image_size: Tuple[int, int] = (224, 224)):
    """Get training data augmentation transforms."""
    config = AUGMENTATION_CONFIG
    
    if ALBUMENTATIONS_AVAILABLE:
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=config.horizontal_flip),
            A.VerticalFlip(p=config.vertical_flip),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=config.rotation_limit, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=config.shift_limit,
                scale_limit=config.scale_limit,
                rotate_limit=config.rotation_limit,
                p=0.5
            ),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=config.brightness_limit,
                    contrast_limit=config.contrast_limit,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=config.hue_shift_limit,
                    sat_shift_limit=config.saturation_limit,
                    val_shift_limit=0.1,
                    p=1.0
                ),
            ], p=0.7),
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussNoise(std_range=(0.2, 0.5), mean_range=(0.0, 0.0), p=1.0),
            ], p=config.gaussian_blur),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(0.05, 0.125),
                hole_width_range=(0.05, 0.125),
                fill=0,
                p=0.3
            ),
            A.Normalize(mean=config.mean, std=config.std),
            ToTensorV2(),
        ])
    else:
        # Fallback to PyTorch transforms
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=config.horizontal_flip),
            transforms.RandomVerticalFlip(p=config.vertical_flip),
            transforms.RandomRotation(degrees=config.rotation_limit),
            transforms.ColorJitter(
                brightness=config.brightness_limit,
                contrast=config.contrast_limit,
                saturation=config.saturation_limit,
                hue=config.hue_shift_limit / 360.0
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ])


def get_val_transforms(image_size: Tuple[int, int] = (224, 224)):
    """Get validation/test transforms (no augmentation)."""
    config = AUGMENTATION_CONFIG
    
    if ALBUMENTATIONS_AVAILABLE:
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=config.mean, std=config.std),
            ToTensorV2(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ])


def create_data_loaders(
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    """
    
    processed_dir = BASE_DIR / "data" / "processed"
    
    # Check if processed data exists
    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_dir}. "
            "Please run data preprocessing first: python src/data_preprocessing.py"
        )
    
    # Get transforms
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    
    # Create datasets
    train_dataset = PlantDiseaseDataset(
        data_dir=processed_dir / "train",
        transform=train_transform
    )
    
    val_dataset = PlantDiseaseDataset(
        data_dir=processed_dir / "val",
        transform=val_transform,
        class_mapping=train_dataset.class_mapping
    )
    
    test_dataset = PlantDiseaseDataset(
        data_dir=processed_dir / "test",
        transform=val_transform,
        class_mapping=train_dataset.class_mapping
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Data loaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"  Classes: {train_dataset.num_classes}")
    
    return train_loader, val_loader, test_loader


class MultiDatasetLoader:
    """
    Custom data loader that can handle multiple datasets with different formats.
    """
    
    def __init__(self, datasets_config: Dict, batch_size: int = 32):
        self.datasets_config = datasets_config
        self.batch_size = batch_size
        self.datasets = {}
        self.loaders = {}
        
        self._load_datasets()
    
    def _load_datasets(self):
        """Load individual datasets."""
        for name, config in self.datasets_config.items():
            try:
                dataset = PlantDiseaseDataset(
                    data_dir=config['path'],
                    transform=config.get('transform')
                )
                
                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=config.get('shuffle', True),
                    num_workers=config.get('num_workers', 4)
                )
                
                self.datasets[name] = dataset
                self.loaders[name] = loader
                
            except Exception as e:
                print(f"Failed to load dataset {name}: {e}")
    
    def get_combined_loader(self):
        """Get a combined data loader from all datasets."""
        all_samples = []
        
        for dataset in self.datasets.values():
            all_samples.extend(dataset.samples)
        
        # Create a combined dataset
        # This is a simplified version - you might want to implement
        # more sophisticated dataset combination logic
        pass


def calculate_dataset_statistics(data_loader: DataLoader) -> Dict[str, float]:
    """Calculate mean and std for dataset normalization."""
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for data, _ in data_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'total_samples': total_samples
    }
