"""
Data Preprocessing Pipeline
Prepares custom training data for model training
"""

import os
import shutil
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from PIL import Image
import json
import numpy as np
from typing import Dict, List, Tuple
import cv2

class DataPreprocessor:
    """Handle data preprocessing and splitting"""
    
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw"
        self.processed_dir = self.data_root / "processed"
        self.train_dir = self.data_root / "train"
        self.val_dir = self.data_root / "validation"
        self.test_dir = self.data_root / "test"
        
        # Image settings
        self.target_size = (224, 224)
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
    def setup_directories(self):
        """Create necessary directories"""
        for dir_path in [self.processed_dir, self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def validate_raw_data(self) -> Dict[str, int]:
        """Validate and count raw data"""
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_dir}")
            
        class_counts = {}
        total_images = 0
        
        print("🔍 Scanning raw data...")
        
        for class_dir in self.raw_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            image_files = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in self.supported_formats
            ]
            
            count = len(image_files)
            class_counts[class_name] = count
            total_images += count
            
            print(f"   📁 {class_name}: {count} images")
            
        if not class_counts:
            raise ValueError("No valid image classes found in raw data directory")
            
        print(f"\n📊 Total: {total_images} images across {len(class_counts)} classes")
        
        # Check minimum requirements
        min_images_per_class = 20
        insufficient_classes = [
            cls for cls, count in class_counts.items()
            if count < min_images_per_class
        ]
        
        if insufficient_classes:
            print(f"\n⚠️  Warning: These classes have < {min_images_per_class} images:")
            for cls in insufficient_classes:
                print(f"   - {cls}: {class_counts[cls]} images")
            print("   Consider adding more images for better model performance")
            
        return class_counts
        
    def preprocess_image(self, image_path: Path, output_path: Path) -> bool:
        """
        Preprocess a single image
        
        Args:
            image_path: Source image path
            output_path: Target image path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"   ❌ Cannot read image: {image_path.name}")
                return False
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Normalize and enhance
            image = self.enhance_image(image)
            
            # Convert back to PIL and save
            pil_image = Image.fromarray(image)
            pil_image.save(output_path, quality=95, optimize=True)
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error processing {image_path.name}: {e}")
            return False
            
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply image enhancements"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Slight sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        image = cv2.filter2D(image, -1, kernel * 0.1 + np.eye(3) * 0.9)
        
        # Ensure values are in valid range
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
        
    def process_all_images(self, class_counts: Dict[str, int]):
        """Process all images"""
        print("\n🔄 Processing images...")
        
        processed_counts = {}
        
        for class_name, original_count in class_counts.items():
            class_raw_dir = self.raw_dir / class_name
            class_processed_dir = self.processed_dir / class_name
            class_processed_dir.mkdir(exist_ok=True)
            
            print(f"\n📂 Processing {class_name}...")
            
            processed_count = 0
            
            for image_file in class_raw_dir.iterdir():
                if image_file.suffix.lower() not in self.supported_formats:
                    continue
                    
                output_file = class_processed_dir / f"{image_file.stem}.jpg"
                
                if self.preprocess_image(image_file, output_file):
                    processed_count += 1
                    
            processed_counts[class_name] = processed_count
            print(f"   ✅ {processed_count}/{original_count} images processed")
            
        return processed_counts
        
    def split_data(self, processed_counts: Dict[str, int], 
                   train_ratio: float = 0.7, val_ratio: float = 0.15, 
                   test_ratio: float = 0.15):
        """Split processed data into train/val/test sets"""
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
            
        print(f"\n📊 Splitting data (Train: {train_ratio:.0%}, Val: {val_ratio:.0%}, Test: {test_ratio:.0%})...")
        
        split_stats = {}
        
        for class_name, count in processed_counts.items():
            if count == 0:
                continue
                
            print(f"\n📂 Splitting {class_name} ({count} images)...")
            
            # Get all image files
            class_processed_dir = self.processed_dir / class_name
            image_files = list(class_processed_dir.glob("*.jpg"))
            
            if len(image_files) < 3:
                print(f"   ⚠️  Too few images for proper splitting, copying to train only")
                self._copy_files(image_files, self.train_dir / class_name)
                split_stats[class_name] = {"train": len(image_files), "val": 0, "test": 0}
                continue
            
            # First split: separate test set
            train_val_files, test_files = train_test_split(
                image_files, test_size=test_ratio, random_state=42
            )
            
            # Second split: separate train and validation
            if len(train_val_files) < 2:
                train_files, val_files = train_val_files, []
            else:
                val_size = val_ratio / (train_ratio + val_ratio)
                train_files, val_files = train_test_split(
                    train_val_files, test_size=val_size, random_state=42
                )
            
            # Copy files to respective directories
            for split_name, files, target_dir in [
                ("train", train_files, self.train_dir),
                ("val", val_files, self.val_dir),
                ("test", test_files, self.test_dir)
            ]:
                target_class_dir = target_dir / class_name
                self._copy_files(files, target_class_dir)
                
            split_stats[class_name] = {
                "train": len(train_files),
                "val": len(val_files),
                "test": len(test_files)
            }
            
            print(f"   ✅ Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
            
        return split_stats
        
    def _copy_files(self, files: List[Path], target_dir: Path):
        """Copy files to target directory"""
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in files:
            target_path = target_dir / file_path.name
            shutil.copy2(file_path, target_path)
            
    def save_metadata(self, class_counts: Dict[str, int], 
                     processed_counts: Dict[str, int],
                     split_stats: Dict[str, Dict[str, int]]):
        """Save preprocessing metadata"""
        metadata = {
            "preprocessing_info": {
                "target_size": self.target_size,
                "supported_formats": list(self.supported_formats),
                "enhancement_applied": True
            },
            "original_counts": class_counts,
            "processed_counts": processed_counts,
            "split_statistics": split_stats,
            "class_names": list(processed_counts.keys()),
            "num_classes": len(processed_counts)
        }
        
        metadata_path = self.data_root / "preprocessing_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"\n💾 Metadata saved to: {metadata_path}")
        return metadata
        
    def run_pipeline(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                    test_ratio: float = 0.15):
        """Run the complete preprocessing pipeline"""
        print("🚀 Starting data preprocessing pipeline...\n")
        
        try:
            # Setup
            self.setup_directories()
            
            # Validate and count raw data
            class_counts = self.validate_raw_data()
            
            # Process images
            processed_counts = self.process_all_images(class_counts)
            
            # Split data
            split_stats = self.split_data(processed_counts, train_ratio, val_ratio, test_ratio)
            
            # Save metadata
            metadata = self.save_metadata(class_counts, processed_counts, split_stats)
            
            # Summary
            print("\n" + "="*60)
            print("📋 PREPROCESSING COMPLETE")
            print("="*60)
            
            total_processed = sum(processed_counts.values())
            total_train = sum(stats["train"] for stats in split_stats.values())
            total_val = sum(stats["val"] for stats in split_stats.values())
            total_test = sum(stats["test"] for stats in split_stats.values())
            
            print(f"📊 Total images processed: {total_processed}")
            print(f"🎯 Classes: {len(processed_counts)}")
            print(f"🚂 Train set: {total_train} images")
            print(f"✅ Validation set: {total_val} images")
            print(f"🧪 Test set: {total_test} images")
            print("\n🎉 Data is ready for training!")
            
            return metadata
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            raise

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Preprocess plant disease images for training')
    parser.add_argument('--data-root', default='data', 
                       help='Root directory containing raw data (default: data)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("❌ Error: Train, validation, and test ratios must sum to 1.0")
        return 1
    
    try:
        preprocessor = DataPreprocessor(args.data_root)
        preprocessor.run_pipeline(args.train_ratio, args.val_ratio, args.test_ratio)
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
