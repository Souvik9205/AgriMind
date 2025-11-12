"""
Data preprocessing pipeline for AgriMind plant disease detection.
Handles multiple datasets and creates unified training data.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import DATASETS, MODEL_CONFIG, BASE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocesses and unifies multiple plant disease datasets."""
    
    def __init__(self):
        self.processed_dir = BASE_DIR / "data" / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
        self.train_dir = self.processed_dir / "train"
        self.val_dir = self.processed_dir / "val"
        self.test_dir = self.processed_dir / "test"
        
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def scan_dataset_structure(self, dataset_path: Path) -> Dict:
        """Scan dataset directory structure and identify classes."""
        if not dataset_path.exists():
            logger.warning(f"Dataset path does not exist: {dataset_path}")
            return {"classes": [], "structure": "unknown", "total_images": 0}
        
        classes = []
        total_images = 0
        structure = "unknown"
        
        # Check for common dataset structures
        if (dataset_path / "train").exists() and (dataset_path / "test").exists():
            structure = "train_test_split"
            for split in ["train", "test", "val"]:
                split_dir = dataset_path / split
                if split_dir.exists():
                    split_classes = [d.name for d in split_dir.iterdir() if d.is_dir()]
                    classes.extend(split_classes)
        
        elif any(d.is_dir() and len(list(d.glob("*.jpg"))) > 0 for d in dataset_path.iterdir()):
            structure = "class_folders"
            classes = [d.name for d in dataset_path.iterdir() 
                      if d.is_dir() and len(list(d.glob("*.*"))) > 0]
        
        # Count total images
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            total_images += len(list(dataset_path.rglob(ext)))
        
        classes = list(set(classes))
        
        return {
            "classes": sorted(classes),
            "structure": structure,
            "total_images": total_images
        }
    
    def process_plantvillage_dataset(self) -> List[Tuple[str, str]]:
        """Process PlantVillage dataset."""
        dataset_path = DATASETS["plantvillage"].path
        logger.info(f"Processing PlantVillage dataset from {dataset_path}")
        
        image_label_pairs = []
        
        # Look for the actual dataset directory
        possible_paths = [
            dataset_path,
            dataset_path / "PlantVillage",
            dataset_path / "plant-disease",
            dataset_path / "color"
        ]
        
        actual_path = None
        for path in possible_paths:
            if path.exists() and any(d.is_dir() for d in path.iterdir()):
                actual_path = path
                break
        
        if not actual_path:
            logger.warning(f"Could not find PlantVillage dataset structure in {dataset_path}")
            return image_label_pairs
        
        # Process class directories
        for class_dir in actual_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for img_path in class_dir.glob("*.*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        image_label_pairs.append((str(img_path), class_name))
        
        logger.info(f"Found {len(image_label_pairs)} images in PlantVillage dataset")
        return image_label_pairs
    

    def process_crop_diseases_dataset(self) -> List[Tuple[str, str]]:
        """Process Crop Diseases dataset."""
        dataset_path = DATASETS["crop_diseases"].path
        logger.info(f"Processing Crop Diseases dataset from {dataset_path}")
        
        image_label_pairs = []
        
        # Direct class folders structure
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir() and class_dir.name != ".DS_Store":
                class_name = class_dir.name
                for img_path in class_dir.glob("*.*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        image_label_pairs.append((str(img_path), class_name))
        
        logger.info(f"Found {len(image_label_pairs)} images in Crop Diseases dataset")
        return image_label_pairs
    
    def process_rice_leaf_diseases_dataset(self) -> List[Tuple[str, str]]:
        """Process Rice leaf diseases dataset."""
        dataset_path = DATASETS["rice_leaf_diseases"].path
        logger.info(f"Processing Rice leaf diseases dataset from {dataset_path}")
        
        image_label_pairs = []
        
        # Look for common structures
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for img_path in class_dir.glob("*.*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        image_label_pairs.append((str(img_path), class_name))
        
        logger.info(f"Found {len(image_label_pairs)} images in Rice leaf dataset")
        return image_label_pairs
    
    def create_unified_dataset(self):
        """Create a unified dataset from all sources."""
        logger.info("Creating unified dataset...")
        
        all_image_pairs = []
        
        # Process each dataset
        datasets_processors = [
            self.process_plantvillage_dataset,
            self.process_bengal_leaf_disease_dataset,
            self.process_crop_diseases_dataset,
            self.process_rice_leaf_diseases_dataset
        ]
        
        for processor in datasets_processors:
            pairs = processor()
            all_image_pairs.extend(pairs)
        
        if not all_image_pairs:
            logger.error("No images found in any dataset!")
            return
        
        # Analyze class distribution
        class_counts = Counter([pair[1] for pair in all_image_pairs])
        logger.info(f"Total classes found: {len(class_counts)}")
        logger.info(f"Total images: {len(all_image_pairs)}")
        
        # Filter out classes with too few samples
        min_samples = 10
        valid_classes = {cls for cls, count in class_counts.items() if count >= min_samples}
        filtered_pairs = [(img, cls) for img, cls in all_image_pairs if cls in valid_classes]
        
        logger.info(f"After filtering (min {min_samples} samples): {len(valid_classes)} classes, {len(filtered_pairs)} images")
        
        # Create class mapping
        class_to_idx = {cls: idx for idx, cls in enumerate(sorted(valid_classes))}
        
        # Save class mapping
        with open(BASE_DIR / "models" / "class_mapping.json", 'w') as f:
            json.dump(class_to_idx, f, indent=2)
        
        # Split data
        train_pairs, temp_pairs = train_test_split(
            filtered_pairs, 
            test_size=MODEL_CONFIG.validation_split + MODEL_CONFIG.test_split,
            stratify=[pair[1] for pair in filtered_pairs],
            random_state=42
        )
        
        val_pairs, test_pairs = train_test_split(
            temp_pairs,
            test_size=MODEL_CONFIG.test_split / (MODEL_CONFIG.validation_split + MODEL_CONFIG.test_split),
            stratify=[pair[1] for pair in temp_pairs],
            random_state=42
        )
        
        logger.info(f"Dataset split: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")
        
        # Copy images to processed directories
        self.copy_images_to_splits(train_pairs, val_pairs, test_pairs, class_to_idx)
        
        # Create metadata
        metadata = {
            "num_classes": len(class_to_idx),
            "class_mapping": class_to_idx,
            "class_counts": dict(class_counts),
            "dataset_splits": {
                "train": len(train_pairs),
                "val": len(val_pairs),
                "test": len(test_pairs)
            }
        }
        
        with open(self.processed_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Dataset preprocessing completed successfully!")
    
    def copy_images_to_splits(self, train_pairs, val_pairs, test_pairs, class_to_idx):
        """Copy images to train/val/test directories."""
        splits = [
            (train_pairs, self.train_dir, "train"),
            (val_pairs, self.val_dir, "validation"),
            (test_pairs, self.test_dir, "test")
        ]
        
        for pairs, split_dir, split_name in splits:
            logger.info(f"Copying {split_name} images...")
            
            # Create class directories
            for class_name in class_to_idx.keys():
                (split_dir / class_name).mkdir(exist_ok=True)
            
            # Copy images
            for img_path, class_name in tqdm(pairs, desc=f"Copying {split_name}"):
                src_path = Path(img_path)
                if src_path.exists():
                    # Create unique filename to avoid conflicts
                    dst_filename = f"{src_path.stem}_{hash(img_path) % 10000}{src_path.suffix}"
                    dst_path = split_dir / class_name / dst_filename
                    
                    try:
                        shutil.copy2(src_path, dst_path)
                    except Exception as e:
                        logger.warning(f"Failed to copy {src_path}: {e}")
    
    def validate_images(self, directory: Path):
        """Validate that all images can be loaded and are not corrupted."""
        logger.info(f"Validating images in {directory}...")
        
        corrupted_images = []
        for img_path in directory.rglob("*.*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                try:
                    img = Image.open(img_path)
                    img.verify()  # Verify image integrity
                except Exception as e:
                    logger.warning(f"Corrupted image: {img_path} - {e}")
                    corrupted_images.append(img_path)
        
        # Remove corrupted images
        for img_path in corrupted_images:
            try:
                img_path.unlink()
                logger.info(f"Removed corrupted image: {img_path}")
            except Exception as e:
                logger.error(f"Failed to remove {img_path}: {e}")
        
        logger.info(f"Validation complete. Removed {len(corrupted_images)} corrupted images.")
    
    def generate_dataset_report(self):
        """Generate a comprehensive dataset report."""
        logger.info("Generating dataset report...")
        
        report = {
            "datasets_scanned": {},
            "processed_statistics": {}
        }
        
        # Scan original datasets
        for name, config in DATASETS.items():
            if config.path.exists():
                scan_result = self.scan_dataset_structure(config.path)
                report["datasets_scanned"][name] = scan_result
        
        # Analyze processed data
        if (self.processed_dir / "metadata.json").exists():
            with open(self.processed_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
                report["processed_statistics"] = metadata
        
        # Save report
        with open(self.processed_dir / "dataset_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Dataset report saved to dataset_report.json")
        
        # Print summary
        print("\n" + "="*60)
        print("DATASET PREPROCESSING SUMMARY")
        print("="*60)
        
        for name, stats in report["datasets_scanned"].items():
            print(f"{name}:")
            print(f"  Classes: {len(stats['classes'])}")
            print(f"  Images: {stats['total_images']}")
            print(f"  Structure: {stats['structure']}")
        
        if "processed_statistics" in report and report["processed_statistics"]:
            stats = report["processed_statistics"]
            print(f"\nUnified Dataset:")
            print(f"  Total Classes: {stats['num_classes']}")
            print(f"  Train Images: {stats['dataset_splits']['train']}")
            print(f"  Val Images: {stats['dataset_splits']['val']}")
            print(f"  Test Images: {stats['dataset_splits']['test']}")

def main():
    """Main preprocessing function."""
    preprocessor = DataPreprocessor()
    
    # Generate initial report
    preprocessor.generate_dataset_report()
    
    # Create unified dataset
    preprocessor.create_unified_dataset()
    
    # Validate processed images
    for split_dir in [preprocessor.train_dir, preprocessor.val_dir, preprocessor.test_dir]:
        if split_dir.exists():
            preprocessor.validate_images(split_dir)
    
    # Generate final report
    preprocessor.generate_dataset_report()

if __name__ == "__main__":
    main()
