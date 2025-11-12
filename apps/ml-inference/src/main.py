"""
Main application entry point for AgriMind ML Inference.
"""

import argparse
import logging
from pathlib import Path

from config import BASE_DIR
from data_preprocessing import main as preprocess_main
from train import main as train_main
from evaluate import main as evaluate_main

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_project():
    """Set up the project structure and initial configuration."""
    logger.info("Setting up AgriMind ML Inference project...")
    
    # Create necessary directories
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'logs',
        'results/plots',
        'results/reports'
    ]
    
    for dir_path in directories:
        (BASE_DIR / dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Project setup completed!")

def download_datasets():
    """Download and prepare datasets."""
    logger.info("Starting dataset download...")
    
    try:
        import subprocess
        result = subprocess.run([
            'python', 'scripts/download_datasets.py'
        ], cwd=BASE_DIR, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Dataset download completed successfully!")
            print(result.stdout)
        else:
            logger.error(f"Dataset download failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to run dataset download script: {e}")
        return False
    
    return True

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='AgriMind ML Inference Pipeline')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Set up project structure')
    
    # Download datasets command
    download_parser = subparsers.add_parser('download', help='Download datasets')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess datasets')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--resume', type=str, help='Resume training from checkpoint')
    train_parser.add_argument('--config', type=str, help='Path to config file')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    evaluate_parser.add_argument('--model', type=str, help='Path to model checkpoint')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict plant disease from image')
    predict_parser.add_argument('image_path', help='Path to image file')
    predict_parser.add_argument('--model', type=str, help='Path to model checkpoint')
    predict_parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions')
    predict_parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    
    # Batch predict command
    batch_predict_parser = subparsers.add_parser('batch-predict', help='Predict multiple images')
    batch_predict_parser.add_argument('input_dir', help='Directory containing images')
    batch_predict_parser.add_argument('--model', type=str, help='Path to model checkpoint')
    batch_predict_parser.add_argument('--output', type=str, help='Output CSV file for results')
    batch_predict_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    

    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full ML pipeline')
    pipeline_parser.add_argument('--skip-download', action='store_true', 
                                help='Skip dataset download')
    pipeline_parser.add_argument('--skip-preprocess', action='store_true', 
                                help='Skip preprocessing')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    if args.command == 'setup':
        setup_project()
    
    elif args.command == 'download':
        if not download_datasets():
            logger.error("Dataset download failed!")
            return
    
    elif args.command == 'preprocess':
        logger.info("Starting data preprocessing...")
        preprocess_main()
    
    elif args.command == 'train':
        logger.info("Starting model training...")
        
        # Load custom config if provided
        if args.config:
            logger.info(f"Loading config from {args.config}")
            # Implement config loading logic here
        
        if args.resume:
            logger.info(f"Resuming training from {args.resume}")
            # Implement resume logic here
        
        train_main()
    
    elif args.command == 'evaluate':
        logger.info("Starting model evaluation...")
        
        if args.model:
            logger.info(f"Evaluating model: {args.model}")
            # Pass model path to evaluator
        
        evaluate_main()
    
    elif args.command == 'predict':
        logger.info("Making prediction...")
        
        from inference import predict_image
        import json
        
        try:
            result = predict_image(
                image_path=args.image_path,
                model_path=args.model,
                top_k=args.top_k
            )
            
            print(f"\nPrediction for: {args.image_path}")
            print(f"Predicted class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            
            print(f"\nTop {args.top_k} predictions:")
            for i, pred in enumerate(result['top_predictions'], 1):
                print(f"{i}. {pred['class']}: {pred['confidence']:.4f}")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Results saved to {args.output}")
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
    
    elif args.command == 'batch-predict':
        logger.info("Running batch prediction...")
        
        from inference import load_predictor
        import csv
        from pathlib import Path
        
        try:
            predictor = load_predictor(model_path=args.model)
            
            # Get all image files
            input_path = Path(args.input_dir)
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(input_path.glob(f"**/*{ext}"))
                image_files.extend(input_path.glob(f"**/*{ext.upper()}"))
            
            logger.info(f"Found {len(image_files)} images")
            
            # Make predictions
            results = []
            for image_file in image_files:
                try:
                    result = predictor.predict(str(image_file))
                    results.append({
                        'image_path': str(image_file),
                        'predicted_class': result['predicted_class'],
                        'confidence': result['confidence']
                    })
                    print(f"✓ {image_file.name}: {result['predicted_class']} ({result['confidence']:.4f})")
                except Exception as e:
                    logger.warning(f"Failed to predict {image_file}: {e}")
                    results.append({
                        'image_path': str(image_file),
                        'predicted_class': 'ERROR',
                        'confidence': 0.0
                    })
            
            # Save results
            if args.output:
                with open(args.output, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['image_path', 'predicted_class', 'confidence'])
                    writer.writeheader()
                    writer.writerows(results)
                logger.info(f"Results saved to {args.output}")
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")

    
    elif args.command == 'pipeline':
        logger.info("Running full ML pipeline...")
        
        # Setup project
        setup_project()
        
        # Download datasets
        if not args.skip_download:
            if not download_datasets():
                logger.error("Pipeline stopped due to download failure")
                return
        
        # Preprocess data
        if not args.skip_preprocess:
            logger.info("Running data preprocessing...")
            preprocess_main()
        
        # Train model
        logger.info("Training model...")
        train_main()
        
        # Evaluate model
        logger.info("Evaluating model...")
        evaluate_main()
        
        logger.info("Full pipeline completed successfully!")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
