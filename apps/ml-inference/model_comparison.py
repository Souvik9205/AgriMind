"""
Model Comparison Script
Compare the performance of the current model vs Hugging Face model
"""

import argparse
from pathlib import Path
import time
import json
from typing import Dict, List
import pandas as pd

# Import all predictors
from huggingface_predictor import HuggingFacePredictor
from local_predictor import LocalModelPredictor

def compare_predictions(image_paths: List[str], output_file: str = None) -> pd.DataFrame:
    """
    Compare predictions from both models
    
    Args:
        image_paths: List of image paths to test
        output_file: Optional CSV file to save results
        
    Returns:
        DataFrame with comparison results
    """
    print("🔄 Initializing models...")
    
    # Initialize predictors
    try:
        local_pred = LocalModelPredictor()
        if local_pred.is_available():
            print("✅ Local predictor loaded")
        else:
            print("📭 No local models available")
            local_pred = None
    except Exception as e:
        print(f"❌ Error loading local predictor: {e}")
        local_pred = None
    
    try:
        hf_pred = HuggingFacePredictor()
        print("✅ Hugging Face predictor loaded")
    except Exception as e:
        print(f"❌ Error loading HF predictor: {e}")
        hf_pred = None
    
    results = []
    
    print(f"\n🧪 Testing on {len(image_paths)} images:")
    print("=" * 80)
    
    for i, img_path in enumerate(image_paths, 1):
        img_name = Path(img_path).name
        print(f"\n📷 [{i}/{len(image_paths)}] Testing: {img_name}")
        
        result = {
            'image_name': img_name,
            'image_path': img_path
        }
        
        # Test local predictor
        if local_pred:
            try:
                start_time = time.time()
                local_result = local_pred.infer(img_path)
                local_time = time.time() - start_time
                
                result.update({
                    'local_prediction': local_result['label'],
                    'local_confidence': local_result['confidence'],
                    'local_time': local_time
                })
                
                model_name = local_result['model_info']['model_name']
                print(f"   🏠 Local Model ({model_name}): {local_result['label']} ({local_result['confidence']:.1%})")
                
            except Exception as e:
                result.update({
                    'local_prediction': 'Error',
                    'local_confidence': 0.0,
                    'local_time': 0.0,
                    'local_error': str(e)
                })
                print(f"   ❌ Local Model Error: {e}")
        
        # Test HuggingFace predictor
        if hf_pred:
            try:
                start_time = time.time()
                hf_result = hf_pred.infer(img_path)
                hf_time = time.time() - start_time
                
                result.update({
                    'hf_prediction': hf_result['label'],
                    'hf_confidence': hf_result['confidence'],
                    'hf_time': hf_time
                })
                
                print(f"   🤗 HuggingFace: {hf_result['label']} ({hf_result['confidence']:.1%})")
                
            except Exception as e:
                result.update({
                    'hf_prediction': 'Error',
                    'hf_confidence': 0.0,
                    'hf_time': 0.0,
                    'hf_error': str(e)
                })
                print(f"   ❌ HuggingFace Error: {e}")
        
        # Compare predictions
        if 'local_prediction' in result and 'hf_prediction' in result:
            same_prediction = result['local_prediction'].lower() == result['hf_prediction'].lower()
            result['predictions_match'] = same_prediction
            
            if same_prediction:
                print(f"   ✅ Both models agree!")
            else:
                print(f"   🔄 Different predictions")
                
        # Show performance comparison
        if 'local_time' in result and 'hf_time' in result:
            if result['local_time'] < result['hf_time']:
                faster_model = "Local"
                time_diff = result['hf_time'] - result['local_time']
            else:
                faster_model = "HuggingFace"
                time_diff = result['local_time'] - result['hf_time']
            print(f"   ⚡ {faster_model} model is {time_diff:.2f}s faster")
        
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print_comparison_summary(df)
    
    # Save to CSV if requested
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
    
    return df

def print_comparison_summary(df: pd.DataFrame):
    """Print a summary of the comparison results"""
    
    print("\n" + "=" * 80)
    print("📊 COMPARISON SUMMARY")
    print("=" * 80)
    
    total_images = len(df)
    print(f"📷 Total images tested: {total_images}")
    
    if 'local_prediction' in df.columns and 'hf_prediction' in df.columns:
        # Agreement rate
        agreement_rate = df['predictions_match'].sum() / total_images * 100
        print(f"🤝 Agreement rate: {agreement_rate:.1f}%")
        
        # Performance comparison
        if 'local_time' in df.columns and 'hf_time' in df.columns:
            avg_local_time = df['local_time'].mean()
            avg_hf_time = df['hf_time'].mean()
            
            print(f"⏱️ Average inference time:")
            print(f"   Local Model:  {avg_local_time:.3f}s")
            print(f"   HuggingFace:  {avg_hf_time:.3f}s")
            
            if avg_local_time > 0 and avg_hf_time > 0:
                speedup = avg_hf_time / avg_local_time
                faster_model = "Local" if speedup > 1 else "HuggingFace"
                print(f"   🏃 {faster_model} is {abs(speedup):.1f}x faster")
        
        # Confidence comparison
        if 'local_confidence' in df.columns and 'hf_confidence' in df.columns:
            avg_local_conf = df['local_confidence'].mean()
            avg_hf_conf = df['hf_confidence'].mean()
            
            print(f"📊 Average confidence:")
            print(f"   Local Model:  {avg_local_conf:.1%}")
            print(f"   HuggingFace:  {avg_hf_conf:.1%}")
    
    # Show unique predictions for each available model
    if 'local_prediction' in df.columns:
        print(f"\n🏠 Local Model unique predictions:")
        local_preds = df['local_prediction'].value_counts().head(10)
        for pred, count in local_preds.items():
            percentage = count / total_images * 100
            print(f"   {pred}: {count} ({percentage:.1f}%)")
    
    if 'hf_prediction' in df.columns:        
        print(f"\n🤗 HuggingFace unique predictions:")
        hf_preds = df['hf_prediction'].value_counts().head(10)
        for pred, count in hf_preds.items():
            percentage = count / total_images * 100
            print(f"   {pred}: {count} ({percentage:.1f}%)")

def analyze_disease_diversity(df: pd.DataFrame):
    """Analyze the diversity of predicted diseases"""
    
    print("\n📈 DISEASE DIVERSITY ANALYSIS")
    print("=" * 50)
    
    if 'simple_prediction' in df.columns:
        simple_unique = df['simple_prediction'].nunique()
        print(f"🔹 Simple Model predicts {simple_unique} different diseases")
    
    if 'hf_prediction' in df.columns:
        hf_unique = df['hf_prediction'].nunique()
        print(f"🤗 HuggingFace predicts {hf_unique} different diseases")
    
    # Check for the "all healthy potato" issue
    if 'simple_prediction' in df.columns:
        potato_healthy_count = df['simple_prediction'].str.contains('Potato.*Healthy', case=False, na=False).sum()
        if potato_healthy_count > len(df) * 0.8:  # More than 80% predicted as healthy potato
            print(f"⚠️ WARNING: Simple model predicts 'Healthy Potato' for {potato_healthy_count}/{len(df)} images ({potato_healthy_count/len(df)*100:.1f}%)")
            print("   This suggests the model is biased towards healthy potato predictions.")

def main():
    parser = argparse.ArgumentParser(description='Compare Simple vs HuggingFace Plant Disease Models')
    parser.add_argument('--test-dir', default='test_images', help='Directory containing test images')
    parser.add_argument('--images', nargs='+', help='Specific image files to test')
    parser.add_argument('--output', help='CSV file to save results')
    parser.add_argument('--max-images', type=int, default=20, help='Maximum number of images to test')
    
    args = parser.parse_args()
    
    # Collect image paths
    image_paths = []
    
    if args.images:
        # Use specified images
        image_paths = args.images
    else:
        # Use test directory
        test_dir = Path(args.test_dir)
        if test_dir.exists():
            # Get image files
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_paths.extend(test_dir.glob(ext))
            
            # Limit number of images
            if len(image_paths) > args.max_images:
                image_paths = image_paths[:args.max_images]
                print(f"⚠️ Limited to {args.max_images} images for comparison")
            
            # Convert to strings
            image_paths = [str(p) for p in image_paths]
        else:
            print(f"❌ Test directory not found: {test_dir}")
            return 1
    
    if not image_paths:
        print("❌ No images found to test")
        return 1
    
    # Run comparison
    try:
        df = compare_predictions(image_paths, args.output)
        analyze_disease_diversity(df)
        
        print(f"\n✅ Comparison completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
