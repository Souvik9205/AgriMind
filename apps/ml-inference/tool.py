#!/usr/bin/env python3
"""
AgriMind Plant Disease Detection Tool

A standalone tool for predicting plant diseases from images using pre-trained models.
This tool can be used independently without API endpoints.

Usage examples:
    # Predict single image
    python tool.py predict path/to/image.jpg
    
    # Predict single image with custom model
    python tool.py predict path/to/image.jpg --model path/to/model.pth
    
    # Predict multiple images in a directory
    python tool.py batch path/to/images/ --output results.csv
    
    # Train a new model (requires dataset)
    python tool.py train --data path/to/dataset/
    
    # Get model information
    python tool.py info
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict
import csv

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from inference import PlantDiseasePredictor, predict_image
    from config import MODELS_DIR, CLASS_MAPPING_PATH
    from train import PlantDiseaseTrainer
    from data_preprocessing import main as preprocess_data
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the ml-inference directory")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def predict_single_image(args) -> None:
    """Predict disease for a single image."""
    try:
        result = predict_image(
            image_path=args.image_path,
            model_path=args.model,
            top_k=args.top_k
        )
        
        print(f"\n🔍 Analysis Results for: {Path(args.image_path).name}")
        print("=" * 50)
        print(f"🎯 Predicted Disease: {result['predicted_class']}")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        
        if len(result['top_predictions']) > 1:
            print(f"\n📈 Top {len(result['top_predictions'])} Predictions:")
            for i, pred in enumerate(result['top_predictions'], 1):
                print(f"   {i}. {pred['class']}: {pred['confidence']:.1%}")
        
        # Save results if output file specified
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Results saved to: {output_path}")
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)

def predict_batch(args) -> None:
    """Predict diseases for multiple images in a directory."""
    try:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Load predictor
        predictor = PlantDiseasePredictor(model_path=args.model)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"**/*{ext}"))
            image_files.extend(input_dir.glob(f"**/*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No image files found in {input_dir}")
            return
        
        print(f"\n🔍 Processing {len(image_files)} images...")
        print("=" * 50)
        
        results = []
        successful_predictions = 0
        
        for i, image_file in enumerate(image_files, 1):
            try:
                result = predictor.predict(str(image_file))
                
                results.append({
                    'image_path': str(image_file.relative_to(input_dir)),
                    'image_name': image_file.name,
                    'predicted_class': result['predicted_class'],
                    'confidence': result['confidence'],
                    'status': 'success'
                })
                
                print(f"   {i:3d}. ✅ {image_file.name[:40]:<40} → {result['predicted_class']} ({result['confidence']:.1%})")
                successful_predictions += 1
                
            except Exception as e:
                results.append({
                    'image_path': str(image_file.relative_to(input_dir)),
                    'image_name': image_file.name,
                    'predicted_class': 'ERROR',
                    'confidence': 0.0,
                    'status': f'error: {str(e)}'
                })
                
                print(f"   {i:3d}. ❌ {image_file.name[:40]:<40} → Error: {e}")
        
        print("=" * 50)
        print(f"✅ Successfully processed: {successful_predictions}/{len(image_files)} images")
        
        # Save results
        if args.output:
            output_path = Path(args.output)
            
            if output_path.suffix.lower() == '.csv':
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['image_name', 'image_path', 'predicted_class', 'confidence', 'status']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
            
            print(f"💾 Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        sys.exit(1)

def show_model_info(args) -> None:
    """Show information about the loaded model."""
    try:
        predictor = PlantDiseasePredictor(model_path=args.model)
        info = predictor.get_model_info()
        
        print("\n🤖 Model Information")
        print("=" * 50)
        print(f"📁 Model Path: {info['model_path']}")
        print(f"🏗️  Architecture: {info['backbone']}")
        print(f"🎯 Number of Classes: {info['num_classes']}")
        print(f"📊 Total Parameters: {info['total_parameters']:,}")
        print(f"🔧 Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"💻 Device: {info['device']}")
        
        # Show class mapping
        with open(CLASS_MAPPING_PATH, 'r') as f:
            class_mapping = json.load(f)
        
        print(f"\n📝 Disease Classes ({len(class_mapping)} total):")
        for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
            print(f"   {class_id:2d}. {class_name}")
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        sys.exit(1)

def train_model(args) -> None:
    """Train a new model."""
    try:
        print("\n🚀 Starting Model Training")
        print("=" * 50)
        
        if args.data:
            print(f"📁 Dataset Path: {args.data}")
            # Here you would typically preprocess the data first
            # preprocess_data()
        
        # Initialize trainer
        config_overrides = {}
        if args.epochs:
            config_overrides['epochs'] = args.epochs
        if args.batch_size:
            config_overrides['batch_size'] = args.batch_size
        if args.learning_rate:
            config_overrides['learning_rate'] = args.learning_rate
        
        trainer = PlantDiseaseTrainer(config=config_overrides)
        
        print("🏋️  Training model...")
        trainer.train()
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="AgriMind Plant Disease Detection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Predict single image
    predict_parser = subparsers.add_parser('predict', help='Predict disease for a single image')
    predict_parser.add_argument('image_path', help='Path to image file')
    predict_parser.add_argument('--model', help='Path to model file (uses default if not specified)')
    predict_parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions to show')
    predict_parser.add_argument('--output', help='Save results to JSON file')
    
    # Batch predict
    batch_parser = subparsers.add_parser('batch', help='Predict diseases for multiple images')
    batch_parser.add_argument('input_dir', help='Directory containing images')
    batch_parser.add_argument('--model', help='Path to model file (uses default if not specified)')
    batch_parser.add_argument('--output', help='Save results to CSV or JSON file')
    
    # Model info
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.add_argument('--model', help='Path to model file (uses default if not specified)')
    
    # Train model
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data', help='Path to training dataset')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    if args.command == 'predict':
        predict_single_image(args)
    elif args.command == 'batch':
        predict_batch(args)
    elif args.command == 'info':
        show_model_info(args)
    elif args.command == 'train':
        train_model(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
